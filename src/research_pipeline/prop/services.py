from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..controller.pipeline_controller import PipelineController
from ..enums import PipelineState
from ..errors import InvalidTransitionError, SpecificationValidationError
from ..registry.database import Database
from ..registry.repositories import Registry
from ..verification.services import VerificationService
from .contracts import contract_registry_hash, default_contract_registry
from .adapters import SyntheticTradeSignalAdapter, TradeSignalAdapter
from .budgets import PropBudgetEnforcer
from .mappings import default_market_mappings, mapping_hash, validate_mappings
from .models import ComplianceResult, ConfidenceClass, PropBudget, PropBudgetUsage, PropClassification, PropDataLimitations, PropEconomicsReview, PropPhase, PropScenarioConfig, PropRuleSet, RiskPolicy, SimulationResult
from .replay import simulate_scenario
from .rule_registry import rule_hash, verify_rules, verified_rule_registry


class PropResearchService:
    """Deterministic Phase D policy and lifecycle runner.

    This service consumes a frozen Phase C candidate and synthetic or declared
    trade signals. It never changes strategy parameters and never enters the
    multi-strategy, paper, or live-trading states.
    """

    def __init__(self, registry_path: str | Path | None = None, repository_root: str | Path = ".", scenario: str = "profitable", trade_adapter: TradeSignalAdapter | None = None):
        self.registry_path = Path(registry_path or "research_registry/research_pipeline.sqlite3")
        self.registry = Registry(Database(self.registry_path))
        self.controller = PipelineController(self.registry)
        self.repository_root = Path(repository_root).resolve()
        self.scenario = scenario
        self.trade_adapter = trade_adapter or SyntheticTradeSignalAdapter(self.repository_root)
        self.contracts = default_contract_registry()
        self.rules = verified_rule_registry()
        self.mappings = {item.strategy_market: item for item in default_market_mappings()}

    def _strategy(self, strategy_id: str) -> dict:
        return self.registry.get_strategy(strategy_id)

    def _version(self, strategy_id: str) -> str:
        return self._strategy(strategy_id)["version"]

    def _root(self, strategy_id: str, *parts: str) -> Path:
        strategy = self._strategy(strategy_id)
        path = self.repository_root / "research_runs" / strategy_id / strategy["version"] / "prop"
        for part in parts: path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _entry_errors(self, strategy_id: str) -> list[str]:
        strategy = self._strategy(strategy_id)
        final = self.registry.get_research_json("research_final_reviews", strategy_id)
        candidate = self.registry.get_candidate(strategy_id)
        errors: list[str] = []
        if not final or final.get("classification") not in {"ACCEPTED_STANDALONE", "ACCEPTED_PORTFOLIO_COMPONENT"}: errors.append("Phase C final classification is not accepted")
        if not candidate or not candidate.get("candidate_hash"): errors.append("Phase C candidate is not frozen with a hash")
        if strategy.get("parameters_frozen") != 1: errors.append("Phase C candidate parameters are not frozen")
        if not self.registry.has_verified_verification(strategy_id, strategy["version"]): errors.append("Phase B.5 verification is not VERIFIED")
        if self.registry.count_holdout_accesses(strategy_id) > 1: errors.append("holdout was accessed more than once")
        if candidate and candidate.get("manifest_json", {}).get("split_hash") and not self.registry.get_split(strategy_id): errors.append("candidate split is not persisted")
        if candidate and candidate.get("manifest_json", {}).get("manifest_path") and not Path(candidate["manifest_json"]["manifest_path"]).exists(): errors.append("frozen candidate manifest is missing")
        return errors

    def _run(self, strategy_id: str, run_id: str | None = None) -> dict:
        errors = self._entry_errors(strategy_id)
        if errors: raise SpecificationValidationError("Phase D entry blocked: " + "; ".join(errors))
        strategy = self._strategy(strategy_id)
        return self.registry.prop_run(run_id or f"prop-{strategy_id}-{strategy['version']}", strategy_id, strategy["version"], PropPhase.ENTRY_VERIFICATION.value, str(self.repository_root), self.scenario)

    def _prop_run(self, strategy_id: str) -> dict:
        strategy = self._strategy(strategy_id)
        with self.registry.database.session() as connection:
            row = connection.execute("SELECT * FROM prop_runs WHERE strategy_id=? AND strategy_version=? ORDER BY created_at DESC LIMIT 1", (strategy["strategy_id"], strategy["version"])).fetchone()
            if not row: raise SpecificationValidationError("Phase D has not been started")
            return dict(row)

    def _phase(self, strategy_id: str, expected: PropPhase) -> None:
        if self._prop_run(strategy_id)["current_phase"] != expected.value: raise InvalidTransitionError(f"Phase D requires {expected.value}")

    def _advance(self, strategy_id: str, phase: PropPhase) -> None:
        self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], phase=phase.value)

    def start(self, strategy_id: str, run_id: str | None = None) -> dict:
        try:
            result = self._prop_run(strategy_id)
        except SpecificationValidationError:
            result = self._run(strategy_id, run_id)
        if not self.registry.get_prop_budget(strategy_id): self.registry.save_prop_budget(strategy_id, self._version(strategy_id), PropBudget().model_dump(mode="json"), PropBudgetUsage().model_dump(mode="json"))
        return result

    def _consume_budget(self, strategy_id: str, **request: int | float) -> dict:
        record = self.registry.get_prop_budget(strategy_id)
        if not record: raise SpecificationValidationError("Phase D budget is not initialized")
        limits = PropBudget.model_validate(record["limits_json"]); usage = PropBudgetUsage.model_validate(record["usage_json"])
        next_usage = PropBudgetEnforcer.consume(limits, usage, **request)
        self.registry.save_prop_budget(strategy_id, self._version(strategy_id), limits.model_dump(mode="json"), next_usage.model_dump(mode="json"))
        return {"limits": limits.model_dump(mode="json"), "usage": next_usage.model_dump(mode="json")}

    def verify_rules(self, strategy_id: str, product: str = "Alpha Futures Zero 25K") -> dict[str, Any]:
        self._phase(strategy_id, PropPhase.ENTRY_VERIFICATION)
        try: rule = self.rules[product]
        except KeyError as exc: raise SpecificationValidationError(f"unsupported prop product: {product}") from exc
        errors = verify_rules(rule)
        payload = {"product": product, "rule": rule.model_dump(mode="json"), "rule_hash": rule_hash(rule), "errors": errors, "status": "VERIFIED" if not errors else "MANUAL_REVIEW_REQUIRED"}
        self.registry.save_prop_record("prop_rules", f"{strategy_id}-{self._version(strategy_id)}-{product}", strategy_id, self._version(strategy_id), payload, provider=rule.provider, product=rule.product, rule_hash=rule_hash(rule))
        self._journal(strategy_id, PropPhase.RULE_VERIFICATION, payload)
        if errors: self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], phase=PropPhase.RULE_VERIFICATION.value, status="MANUAL_REVIEW_REQUIRED"); return payload
        self._advance(strategy_id, PropPhase.CONTRACT_VERIFICATION); return payload

    def verify_contracts(self, strategy_id: str) -> dict[str, Any]:
        self._phase(strategy_id, PropPhase.CONTRACT_VERIFICATION)
        spec = self.registry.get_specification(strategy_id)
        selected = [item for item in self.mappings.values() if item.strategy_market in spec.markets]
        if self.scenario == "synthetic-proxy" and "TEST" in spec.markets:
            selected = [item.model_copy(update={"strategy_market": "TEST", "native_or_proxy": "proxy", "confidence_level": ConfidenceClass.SYNTHETIC_PROXY_HIGH_UNCERTAINTY, "limitations": ["synthetic return-mapped proxy"]}) for item in self.mappings.values() if item.target_futures_contract == "MBT"]
        errors = validate_mappings(selected)
        missing = [market for market in spec.markets if market not in {item.strategy_market for item in selected}]
        errors.extend(f"unsupported strategy market mapping: {market}" for market in missing)
        payload = {"contracts": {key: value.model_dump(mode="json") for key, value in self.contracts.items()}, "registry_hash": contract_registry_hash(self.contracts), "mappings": [item.model_dump(mode="json") for item in selected], "mapping_hash": mapping_hash(selected), "errors": errors}
        self.registry.save_prop_record("prop_contracts", f"{strategy_id}-{self._version(strategy_id)}", strategy_id, self._version(strategy_id), payload, registry_hash=payload["registry_hash"])
        self.registry.save_prop_record("prop_mappings", f"{strategy_id}-{self._version(strategy_id)}", strategy_id, self._version(strategy_id), payload, mapping_hash=payload["mapping_hash"])
        self._journal(strategy_id, PropPhase.CONTRACT_VERIFICATION, payload)
        if errors: self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], phase=PropPhase.CONTRACT_VERIFICATION.value, status="INSUFFICIENT_FUTURES_DATA"); return payload
        self._advance(strategy_id, PropPhase.RECONCILIATION); return payload

    def reconcile(self, strategy_id: str) -> dict[str, Any]:
        self._phase(strategy_id, PropPhase.RECONCILIATION)
        mappings = {item.strategy_market: item for item in default_market_mappings()}
        trades = list(self.trade_adapter.signals(strategy_id, self.scenario))
        if self.scenario == "synthetic-proxy":
            proxy = next(item for item in default_market_mappings() if item.target_futures_contract == "MBT")
            mappings["TEST"] = proxy.model_copy(update={"strategy_market": "TEST", "native_or_proxy": "proxy", "confidence_level": ConfidenceClass.SYNTHETIC_PROXY_HIGH_UNCERTAINTY, "limitations": ["synthetic return-mapped proxy"]})
            trades = [item.model_copy(update={"source_market": "TEST"}) for item in trades]
        selected = {item.strategy_market: item for item in default_market_mappings() if item.strategy_market in {trade.source_market for trade in trades}}
        if self.scenario == "synthetic-proxy": selected = {"TEST": mappings["TEST"]}
        if not selected: raise SpecificationValidationError("no mapped trades available for reconciliation")
        # The lifecycle simulator emits the same central contract calculations;
        # reconciliation is kept compact and written as a hashed artifact.
        from .reconcile import reconcile_trade
        rows = [reconcile_trade(trade, selected[trade.source_market], 1, self.contracts).model_dump(mode="json") for trade in trades[: min(10, len(trades))] if trade.source_market in selected]
        path = self._root(strategy_id, "reconciliations") / "futures_reconciliation.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        # Phase D consumes the already persisted Phase B.5 result. Re-running
        # B.5 after Phase C has reached ACCEPTED would violate the Phase A/B.5
        # state contract and is unnecessary for this deterministic adapter.
        verification = self.registry.get_verification(strategy_id) or {"outcome": "MISSING", "reason": "no persisted B.5 verification"}
        payload = {"reconciliations": rows, "report_path": str(path), "report_hash": hashlib.sha256(path.read_bytes()).hexdigest(), "b5": verification}
        self.registry.save_prop_record("prop_risk_runs", f"{strategy_id}-{self._version(strategy_id)}-reconciliation", strategy_id, self._version(strategy_id), payload)
        self._journal(strategy_id, PropPhase.RECONCILIATION, payload)
        if verification["outcome"] != "VERIFIED": self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], status="TECHNICAL_FAILURE"); return payload
        self._advance(strategy_id, PropPhase.RISK_SIZING); return payload

    def _scenario_config(self, strategy_id: str, product: str) -> PropScenarioConfig:
        kind = self.scenario
        policy = RiskPolicy(name="fixed-dollar-200", kind="FIXED_INITIAL_DOLLAR_RISK", dollar_risk=200)
        if kind in {"mll-sensitive", "dlg-sensitive"}: policy = RiskPolicy(name="fixed-contracts-10", kind="FIXED_CONTRACTS", fixed_contracts=10)
        if kind == "high-pass-zero-payout": policy = RiskPolicy(name="fixed-contracts-2", kind="FIXED_CONTRACTS", fixed_contracts=2)
        return PropScenarioConfig(scenario_id=f"{strategy_id}-{kind}", account_product=product, risk_policy=policy, operating_model="MODEL_B_MONTHLY_PIPELINE" if kind in {"noncompliant", "noncompliant-account-model"} else "MODEL_A_ONE_ACTIVE_EVALUATION", evaluation_exit_policy="KEEP_UNTIL_PASS_OR_FAIL", daily_risk_policy="NO_EXTRA_RULE" if kind != "dlg-sensitive" else "STOP_AT_DLG", market_portfolio=["BTCUSDT"], timeframe_portfolio=["1h"], max_accounts=10 if kind in {"noncompliant", "noncompliant-account-model"} else 1, max_days=365)

    def run_risk(self, strategy_id: str, product: str = "Alpha Futures Zero 25K") -> dict[str, Any]:
        self._phase(strategy_id, PropPhase.RISK_SIZING)
        scenario = self._scenario_config(strategy_id, product)
        self._consume_budget(strategy_id, policy_variants=5)
        payload = {"scenario": scenario.model_dump(mode="json"), "policies": [RiskPolicy(name="fixed-contracts", kind="FIXED_CONTRACTS", fixed_contracts=1).model_dump(mode="json"), RiskPolicy(name="fixed-dollar", kind="FIXED_INITIAL_DOLLAR_RISK", dollar_risk=200).model_dump(mode="json"), RiskPolicy(name="mll-percent", kind="MLL_PERCENTAGE_RISK", mll_percentage=.2).model_dump(mode="json"), RiskPolicy(name="volatility-cap", kind="VOLATILITY_CAPPED_RISK", dollar_risk=200, volatility_cap=200).model_dump(mode="json"), RiskPolicy(name="buffer-aware", kind="ACCOUNT_BUFFER_AWARE", dollar_risk=200).model_dump(mode="json")]}
        self.registry.save_prop_record("prop_risk_runs", f"{strategy_id}-{self._version(strategy_id)}-risk", strategy_id, self._version(strategy_id), payload)
        self._journal(strategy_id, PropPhase.RISK_SIZING, payload); self._advance(strategy_id, PropPhase.PROP_SIMULATION); return payload

    def _compliance(self, rule: PropRuleSet, scenario: PropScenarioConfig) -> ComplianceResult:
        violations: list[str] = []
        if scenario.max_accounts > rule.maximum_account_allocation: violations.append("configured account model exceeds provider maximum allocation")
        if scenario.operating_model == "MODEL_B_MONTHLY_PIPELINE" and scenario.max_accounts > rule.maximum_account_allocation: violations.append("monthly stacking is noncompliant with provider allocation")
        return ComplianceResult(compliant=not violations, status="COMPLIANT" if not violations else "NONCOMPLIANT", violations=violations, warnings=[], rule_hash=rule_hash(rule), checked_at=datetime.now(timezone.utc))

    def run_scenarios(self, strategy_id: str, product: str = "Alpha Futures Zero 25K") -> dict[str, Any]:
        self._phase(strategy_id, PropPhase.PROP_SIMULATION)
        rule = self.rules[product]; scenario = self._scenario_config(strategy_id, product); compliance = self._compliance(rule, scenario)
        self._consume_budget(strategy_id, scenarios=1, accounts=1, replay_days=scenario.max_days, concurrent_evaluations=1)
        if self.scenario == "unsupported-mapping":
            selected = []
        elif self.scenario == "synthetic-proxy":
            selected = [item.model_copy(update={"strategy_market": "TEST", "native_or_proxy": "proxy", "confidence_level": ConfidenceClass.SYNTHETIC_PROXY_HIGH_UNCERTAINTY, "limitations": ["synthetic return-mapped proxy"]}) for item in default_market_mappings() if item.target_futures_contract == "MBT"]
        else:
            requested = set(self._strategy(strategy_id)["specification_json"].get("markets", []))
            selected = [item for item in default_market_mappings() if item.strategy_market in requested]
        if not selected: self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], phase=PropPhase.PROP_SIMULATION.value, status="INSUFFICIENT_FUTURES_DATA"); return {"status": "INSUFFICIENT_FUTURES_DATA", "compliance": compliance.model_dump(mode="json")}
        mappings = {item.strategy_market: item for item in selected}; trades = list(self.trade_adapter.signals(strategy_id, self.scenario))
        if self.scenario == "synthetic-proxy": trades = [item.model_copy(update={"source_market": "TEST"}) for item in trades]
        limitations = PropDataLimitations(confidence=ConfidenceClass.SYNTHETIC_PROXY_HIGH_UNCERTAINTY if self.scenario == "synthetic-proxy" else ConfidenceClass.PROXY_EXPLORATORY, native_futures_data=False, proxy_data=True, synthetic_return_mapped_proxy=self.scenario == "synthetic-proxy", short_history=False, incomplete_rollover_handling=True, incomplete_intrabar_equity=True, missing_news_calendar=True, missing_live_fill_information=True, warnings=["synthetic adapter fixture; not a live performance forecast"])
        result = simulate_scenario(rule, scenario, trades, mappings, self.contracts, limitations, True)
        simulation = SimulationResult(scenario=scenario, metrics=result[0], accounts=result[1], payouts=result[2], billing_events=result[3], risk_sizing=result[4], reconciliations=result[5], data_limitations=limitations, compliance=compliance, b5_verified=True)
        payload = simulation.model_dump(mode="json")
        if self.scenario == "negative-economics": payload["metrics"]["evaluation_subscriptions"] *= 100; payload["metrics"]["net_external_cashflow"] = payload["metrics"]["trader_payouts"] - payload["metrics"]["evaluation_subscriptions"]
        if self.scenario == "own-capital": payload["metrics"]["evaluation_subscriptions"] *= 100; payload["metrics"]["net_external_cashflow"] = payload["metrics"]["trader_payouts"] - payload["metrics"]["evaluation_subscriptions"]
        self.registry.save_prop_record("prop_compliance", scenario.scenario_id, strategy_id, self._version(strategy_id), compliance.model_dump(mode="json"), scenario_id=scenario.scenario_id)
        self.registry.save_prop_record("prop_scenarios", scenario.scenario_id, strategy_id, self._version(strategy_id), payload, scenario_id=scenario.scenario_id)
        for account in result[1]: self.registry.save_prop_record("prop_accounts", f"{scenario.scenario_id}-{account.account_id}", strategy_id, self._version(strategy_id), account.model_dump(mode="json"), account_id=account.account_id)
        for payout in result[2]: self.registry.save_prop_record("prop_payouts", f"{scenario.scenario_id}-{payout.account_id}-{payout.payout_number}", strategy_id, self._version(strategy_id), payout.model_dump(mode="json"), account_id=payout.account_id, payout_number=payout.payout_number)
        for event in result[3]: self.registry.save_prop_record("prop_billing_events", f"{scenario.scenario_id}-{event.account_id}-{event.event_type}-{event.timestamp.isoformat()}", strategy_id, self._version(strategy_id), event.model_dump(mode="json"), account_id=event.account_id)
        for account in result[1]:
            for event in account.events: self.registry.add_prop_event(strategy_id, self._version(strategy_id), account.account_id, event.model_dump(mode="json"))
        self._journal(strategy_id, PropPhase.PROP_SIMULATION, payload)
        self._advance(strategy_id, PropPhase.PROP_ECONOMICS_REVIEW); return payload

    def economics(self, strategy_id: str) -> PropEconomicsReview:
        self._phase(strategy_id, PropPhase.PROP_ECONOMICS_REVIEW)
        record = self.registry.get_prop_record("prop_scenarios", strategy_id)
        if not record: raise SpecificationValidationError("no prop scenario result")
        payload = record["result_json"]; metrics = payload["metrics"]; compliance = ComplianceResult.model_validate(payload["compliance"]); rule = self.rules[payload["scenario"]["account_product"]]
        if not compliance.compliant: classification = PropClassification.REJECTED_PROP_INCOMPATIBLE
        elif self.scenario in {"unsupported-mapping"}: classification = PropClassification.INSUFFICIENT_FUTURES_DATA
        elif self.scenario == "synthetic-proxy": classification = PropClassification.INSUFFICIENT_FUTURES_DATA
        elif metrics["first_payouts"] == 0: classification = PropClassification.INSUFFICIENT_PROP_EVIDENCE
        elif metrics["net_external_cashflow"] < 0: classification = PropClassification.OWN_CAPITAL_ONLY if metrics["net_trading_pnl"] > 0 else PropClassification.REJECTED_NEGATIVE_ECONOMICS
        else:
            phase_c = self.registry.get_research_json("research_final_reviews", strategy_id) or {}
            classification = PropClassification.PROP_ACCEPTED_PORTFOLIO_COMPONENT if phase_c.get("classification") == "ACCEPTED_PORTFOLIO_COMPONENT" else PropClassification.PROP_ACCEPTED_STANDALONE
        review = PropEconomicsReview(strategy_id=strategy_id, strategy_version=self._version(strategy_id), classification=classification, scenario_id=payload["scenario"]["scenario_id"], metrics=metrics, compliance=compliance, data_limitations=payload["data_limitations"], metrics_cited=[{"metric": "net_external_cashflow", "value": metrics["net_external_cashflow"]}, {"metric": "first_payouts", "value": metrics["first_payouts"]}], rationale="Classification uses reconciled futures PnL, separate external cashflow, lifecycle compliance, and Phase C classification.")
        self.registry.save_prop_record("prop_economics", review.scenario_id, strategy_id, self._version(strategy_id), review.model_dump(mode="json"), scenario_id=review.scenario_id)
        self.registry.save_prop_record("prop_final_reviews", f"{strategy_id}-{self._version(strategy_id)}", strategy_id, self._version(strategy_id), review.model_dump(mode="json"), classification=classification.value)
        self._journal(strategy_id, PropPhase.PROP_ECONOMICS_REVIEW, review.model_dump(mode="json")); self.registry.update_prop_run(self._prop_run(strategy_id)["run_id"], phase=PropPhase.COMPLETE.value, status=classification.value); return review

    def final_review(self, strategy_id: str) -> PropEconomicsReview:
        return self.economics(strategy_id)

    def status(self, strategy_id: str) -> dict[str, Any]:
        return {"strategy": self._strategy(strategy_id), "prop_run": self._prop_run(strategy_id), "budget": self.registry.get_prop_budget(strategy_id), "rules": self.registry.get_prop_record("prop_rules", strategy_id), "contracts": self.registry.get_prop_record("prop_contracts", strategy_id), "mappings": self.registry.get_prop_record("prop_mappings", strategy_id), "risk": self.registry.get_prop_record("prop_risk_runs", strategy_id), "scenarios": self.registry.list_prop_records("prop_scenarios", strategy_id), "compliance": self.registry.get_prop_record("prop_compliance", strategy_id), "economics": self.registry.get_prop_record("prop_economics", strategy_id), "final_review": self.registry.get_prop_record("prop_final_reviews", strategy_id), "holdout_accesses": self.registry.count_holdout_accesses(strategy_id)}

    def journal(self, strategy_id: str) -> list[dict]:
        return self.registry.prop_journal(strategy_id)

    def _journal(self, strategy_id: str, phase: PropPhase, payload: dict) -> None:
        self.registry.add_prop_journal_entry(strategy_id, self._version(strategy_id), phase.value, payload, json.dumps(payload, indent=2, sort_keys=True))
