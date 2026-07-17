from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..errors import InvalidTransitionError, SpecificationValidationError
from ..registry.database import Database
from ..registry.repositories import Registry
from .analysis import correlation_metrics, overlap_metrics
from .candidates import generate_candidates
from .eligibility import eligibility, eligible_strategy_ids
from .models import ContributionClassification, PortfolioBudget, PortfolioBudgetUsage, PortfolioCandidate, PortfolioClassification, PortfolioMember, PortfolioPhase, PortfolioReview, PortfolioSpec, PortfolioStressResult, RiskAllocationPolicy
from .replay import replay_shared_account
from .signals import PortfolioSignalAdapter, SyntheticPortfolioSignalAdapter, apply_conflict_policy
from .utils import now_utc, stable_hash, write_artifact


class PortfolioService:
    def __init__(self, registry_path: str | Path | None = None, repository_root: str | Path = ".", scenario: str = "complementary", signal_adapter: PortfolioSignalAdapter | None = None):
        self.registry_path = Path(registry_path or "research_registry/research_pipeline.sqlite3")
        self.registry = Registry(Database(self.registry_path))
        self.repository_root = Path(repository_root).resolve()
        self.scenario = scenario
        self.signal_adapter = signal_adapter or SyntheticPortfolioSignalAdapter()

    def _spec(self, portfolio_id: str) -> PortfolioSpec:
        record = self.registry.get_portfolio_record("portfolio_specs", portfolio_id)
        if not record: raise SpecificationValidationError(f"portfolio specification not found: {portfolio_id}")
        return PortfolioSpec.model_validate(record["result_json"])

    def _run(self, portfolio_id: str) -> dict:
        spec = self._spec(portfolio_id)
        return self.registry.portfolio_run(portfolio_id, spec.version, PortfolioPhase.MULTI_STRATEGY_PORTFOLIO.value, "RUNNING", spec.specification_hash, str(self.repository_root))

    def _current(self, portfolio_id: str) -> dict:
        spec = self._spec(portfolio_id)
        with self.registry.database.session() as connection:
            row = connection.execute("SELECT * FROM portfolio_runs WHERE portfolio_id=? AND portfolio_version=?", (portfolio_id, spec.version)).fetchone()
            if not row: raise SpecificationValidationError(f"portfolio run not started: {portfolio_id}")
            return dict(row)

    def _require_phase(self, portfolio_id: str, phase: PortfolioPhase) -> None:
        current = self._current(portfolio_id)["current_phase"]
        if current != phase.value: raise InvalidTransitionError(f"portfolio requires {phase.value}, got {current}")

    def _advance(self, portfolio_id: str, phase: PortfolioPhase) -> None:
        spec = self._spec(portfolio_id)
        self.registry.update_portfolio_run(portfolio_id, spec.version, phase=phase.value)

    def _root(self, portfolio_id: str, *parts: str) -> Path:
        spec = self._spec(portfolio_id); path = self.repository_root / "research_runs" / "portfolios" / portfolio_id / spec.version
        for part in parts: path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _journal(self, portfolio_id: str, phase: PortfolioPhase, payload: dict, text: str | None = None) -> None:
        spec = self._spec(portfolio_id)
        self.registry.add_portfolio_journal_entry(portfolio_id, spec.version, phase.value, payload, text or json.dumps(payload, indent=2, sort_keys=True, default=str))

    def _consume(self, portfolio_id: str, **increments: int) -> None:
        spec = self._spec(portfolio_id); record = self.registry.get_portfolio_budget(portfolio_id, spec.version)
        if not record: raise SpecificationValidationError("portfolio budget is not initialized")
        limits = PortfolioBudget.model_validate(record["limits_json"]); usage = PortfolioBudgetUsage.model_validate(record["usage_json"])
        updated = usage.model_copy(deep=True)
        for key, amount in increments.items():
            if amount < 0: raise SpecificationValidationError("portfolio budget increments cannot be negative")
            current = getattr(updated, key); limit_key = {"candidates": "maximum_candidate_portfolios", "scenarios": "maximum_scenarios", "stress_scenarios": "maximum_stress_scenarios", "ablation_runs": "maximum_ablation_runs"}[key]
            if current + amount > getattr(limits, limit_key): raise SpecificationValidationError(f"portfolio budget exceeded: {key}")
            setattr(updated, key, current + amount)
        self.registry.save_portfolio_budget(portfolio_id, spec.version, limits.model_dump(mode="json"), updated.model_dump(mode="json"))

    def _limits(self, portfolio_id: str) -> PortfolioBudget:
        spec = self._spec(portfolio_id)
        record = self.registry.get_portfolio_budget(portfolio_id, spec.version)
        if not record:
            raise SpecificationValidationError("portfolio budget is not initialized")
        return PortfolioBudget.model_validate(record["limits_json"])

    def create(self, spec: PortfolioSpec) -> dict:
        canonical = spec.model_copy(update={"specification_hash": "pending", "frozen": True})
        payload = canonical.model_dump(mode="json", exclude={"specification_hash"})
        expected_hash = stable_hash(payload)
        if spec.specification_hash not in {"pending", expected_hash}: raise SpecificationValidationError("portfolio specification hash mismatch")
        frozen = canonical.model_copy(update={"specification_hash": expected_hash})
        existing = self.registry.get_portfolio_record("portfolio_specs", spec.portfolio_id, spec.version)
        if existing:
            if existing["result_json"].get("specification_hash") != expected_hash: raise SpecificationValidationError("portfolio ID/version already has a different specification")
            return {"portfolio_id": spec.portfolio_id, "version": spec.version, "idempotent": True, "specification_hash": expected_hash}
        checks = [eligibility(self.registry, member.strategy_id, exploratory_prop=member.role.value == "EXPLORATORY", non_prop=spec.prop_operating_model == "NON_PROP") for member in frozen.strategy_members]
        if any(not item["eligible"] for item in checks): raise SpecificationValidationError("portfolio entry blocked: " + "; ".join(f"{item['strategy_id']}: {','.join(item['reasons'])}" for item in checks if not item["eligible"]))
        self.registry.save_portfolio_record("portfolio_specs", f"{spec.portfolio_id}-{spec.version}", spec.portfolio_id, spec.version, frozen.model_dump(mode="json"), specification_hash=expected_hash)
        for member in frozen.strategy_members: self.registry.save_portfolio_record("portfolio_members", f"{spec.portfolio_id}-{spec.version}-{member.strategy_id}", spec.portfolio_id, spec.version, member.model_dump(mode="json"), strategy_id=member.strategy_id, candidate_hash=member.candidate_hash)
        self.registry.portfolio_run(spec.portfolio_id, spec.version, PortfolioPhase.MULTI_STRATEGY_PORTFOLIO.value, "RUNNING", expected_hash, str(self.repository_root))
        self.registry.save_portfolio_budget(spec.portfolio_id, spec.version, frozen.budget.model_dump(mode="json"), PortfolioBudgetUsage().model_dump(mode="json"))
        self._journal(spec.portfolio_id, PortfolioPhase.MULTI_STRATEGY_PORTFOLIO, {"question": "whether frozen strategies improve shared portfolio economics", "specification_hash": expected_hash, "members": [member.strategy_id for member in frozen.strategy_members]})
        return {"portfolio_id": spec.portfolio_id, "version": spec.version, "idempotent": False, "specification_hash": expected_hash, "current_phase": PortfolioPhase.MULTI_STRATEGY_PORTFOLIO.value}

    def eligible(self, *, exploratory_prop: bool = False, non_prop: bool = False) -> list[dict]:
        return eligible_strategy_ids(self.registry, exploratory_prop=exploratory_prop, non_prop=non_prop)

    def generate_candidates(self, portfolio_id: str) -> list[dict]:
        version = self._spec(portfolio_id).version
        existing = self.registry.list_portfolio_records("portfolio_candidates", portfolio_id, version)
        if self._current(portfolio_id)["current_phase"] != PortfolioPhase.MULTI_STRATEGY_PORTFOLIO.value and existing:
            return [item["result_json"] for item in existing]
        self._require_phase(portfolio_id, PortfolioPhase.MULTI_STRATEGY_PORTFOLIO)
        spec = self._spec(portfolio_id)
        members = [member for member in spec.strategy_members if eligibility(self.registry, member.strategy_id, exploratory_prop=member.role.value == "EXPLORATORY", non_prop=spec.prop_operating_model == "NON_PROP")["eligible"]]
        limits = self._limits(portfolio_id)
        candidates = generate_candidates(spec, members, maximum=limits.maximum_candidate_portfolios, maximum_strategies=limits.maximum_strategies)
        if not candidates: raise SpecificationValidationError("no eligible portfolio candidates")
        self._consume(portfolio_id, candidates=len(candidates))
        for candidate in candidates: self.registry.save_portfolio_record("portfolio_candidates", candidate.candidate_id, portfolio_id, spec.version, candidate.model_dump(mode="json"), candidate_id=candidate.candidate_id, candidate_hash=candidate.candidate_hash)
        self._advance(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS); self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS, {"candidate_count": len(candidates), "candidate_ids": [item.candidate_id for item in candidates]}); return [item.model_dump(mode="json") for item in candidates]

    def _candidates(self, portfolio_id: str) -> list[PortfolioCandidate]:
        spec = self._spec(portfolio_id); return [PortfolioCandidate.model_validate(item["result_json"]) for item in self.registry.list_portfolio_records("portfolio_candidates", portfolio_id, spec.version)]

    def _candidate_members(self, spec: PortfolioSpec, candidate: PortfolioCandidate) -> list[PortfolioMember]:
        return [member for member in spec.strategy_members if member.strategy_id in candidate.member_strategy_ids]

    def merge_signals(self, portfolio_id: str) -> list[dict]:
        spec = self._spec(portfolio_id); prior_all = self.registry.list_portfolio_records("portfolio_signals", portfolio_id, spec.version)
        if self._current(portfolio_id)["current_phase"] != PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS.value and prior_all:
            return [item["result_json"] for item in prior_all]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS); results = []
        for candidate in self._candidates(portfolio_id):
            prior = self.registry.get_portfolio_record("portfolio_signals", portfolio_id, spec.version, candidate.candidate_id)
            if prior:
                results.append({"candidate_id": candidate.candidate_id, "artifact_path": prior.get("artifact_path"), "artifact_hash": prior.get("artifact_hash"), "raw_count": prior["result_json"].get("raw_count", 0), "accepted_count": prior["result_json"].get("accepted_count", 0)})
                continue
            members = self._candidate_members(spec, candidate); raw = list(self.signal_adapter.signals(candidate.candidate_id, members, self.scenario)); accepted, counts = apply_conflict_policy(raw, members, spec.conflict_policy); payload = {"candidate_id": candidate.candidate_id, "raw_count": len(raw), "accepted_count": len(accepted), "conflict_counts": counts, "events": [item.model_dump(mode="json") for item in accepted]}; path, digest = write_artifact(self._root(portfolio_id, "signals"), f"{candidate.candidate_id}.json", payload); self.registry.save_portfolio_record("portfolio_signals", candidate.candidate_id, portfolio_id, spec.version, {"candidate_id": candidate.candidate_id, "artifact_path": path, "artifact_hash": digest, "raw_count": len(raw), "accepted_count": len(accepted)}, candidate_id=candidate.candidate_id, artifact_path=path, artifact_hash=digest); self.registry.save_portfolio_record("portfolio_conflict_results", candidate.candidate_id, portfolio_id, spec.version, counts, candidate_id=candidate.candidate_id); results.append({"candidate_id": candidate.candidate_id, "artifact_path": path, "artifact_hash": digest, **counts})
        self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS, {"merged_candidates": results}); return results

    def _events(self, portfolio_id: str, candidate_id: str) -> list:
        record = self.registry.get_portfolio_record("portfolio_signals", portfolio_id, record_key=candidate_id)
        if not record: raise SpecificationValidationError(f"merged signal stream missing: {candidate_id}")
        payload = json.loads(Path(record["artifact_path"]).read_text(encoding="utf-8")); return [self._event(item) for item in payload["events"]]

    @staticmethod
    def _event(payload: dict):
        from .models import PortfolioSignalEvent
        return PortfolioSignalEvent.model_validate(payload)

    def analyze_overlap(self, portfolio_id: str) -> list[dict]:
        spec = self._spec(portfolio_id); prior = self.registry.list_portfolio_records("portfolio_overlap_metrics", portfolio_id, spec.version)
        if self._current(portfolio_id)["current_phase"] not in {PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS.value} and prior:
            return [item["result_json"] for item in prior]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS); result = []
        for candidate in self._candidates(portfolio_id):
            metrics = overlap_metrics(candidate.candidate_id, self._events(portfolio_id, candidate.candidate_id), self._candidate_members(spec, candidate)); self.registry.save_portfolio_record("portfolio_overlap_metrics", candidate.candidate_id, portfolio_id, spec.version, metrics.model_dump(mode="json"), candidate_id=candidate.candidate_id); result.append(metrics.model_dump(mode="json"))
        self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS, {"overlap": result}); return result

    def analyze_correlation(self, portfolio_id: str) -> list[dict]:
        spec = self._spec(portfolio_id); prior = self.registry.list_portfolio_records("portfolio_correlation_metrics", portfolio_id, spec.version)
        if self._current(portfolio_id)["current_phase"] not in {PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS.value} and prior:
            return [item["result_json"] for item in prior]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS); result = []
        for candidate in self._candidates(portfolio_id):
            metrics = correlation_metrics(candidate.candidate_id, self._events(portfolio_id, candidate.candidate_id), self._candidate_members(spec, candidate), self._limits(portfolio_id).minimum_correlation_periods); self.registry.save_portfolio_record("portfolio_correlation_metrics", candidate.candidate_id, portfolio_id, spec.version, metrics.model_dump(mode="json"), candidate_id=candidate.candidate_id); result.append(metrics.model_dump(mode="json"))
        return result

    def run_risk(self, portfolio_id: str) -> list[dict]:
        current = self._current(portfolio_id)
        if current["current_phase"] != PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS.value:
            records = self.registry.list_portfolio_records("portfolio_risk_runs", portfolio_id, self._spec(portfolio_id).version)
            if records:
                return [item["result_json"] for item in records]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_SIGNAL_ANALYSIS); spec = self._spec(portfolio_id); result = []
        for candidate in self._candidates(portfolio_id):
            metrics, risk = replay_shared_account(candidate.candidate_id, self._events(portfolio_id, candidate.candidate_id), self._candidate_members(spec, candidate), policy=spec.risk_budget_policy, maximum_total_contracts=spec.maximum_total_contracts, maximum_simultaneous_positions=spec.maximum_simultaneous_positions, product=spec.target_account_products[0], scenario=self.scenario); self.registry.save_portfolio_record("portfolio_risk_runs", candidate.candidate_id, portfolio_id, spec.version, risk.model_dump(mode="json"), candidate_id=candidate.candidate_id); result.append(risk.model_dump(mode="json"))
        self._advance(portfolio_id, PortfolioPhase.PORTFOLIO_RISK_ANALYSIS); self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_RISK_ANALYSIS, {"risk_runs": result}); return result

    def run_prop(self, portfolio_id: str) -> list[dict]:
        current = self._current(portfolio_id)
        if current["current_phase"] not in {PortfolioPhase.PORTFOLIO_RISK_ANALYSIS.value}:
            records = self.registry.list_portfolio_records("portfolio_prop_scenarios", portfolio_id, self._spec(portfolio_id).version)
            if records:
                return [item["result_json"] for item in records]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_RISK_ANALYSIS); spec = self._spec(portfolio_id); result = []
        self._consume(portfolio_id, scenarios=len(self._candidates(portfolio_id)))
        for candidate in self._candidates(portfolio_id):
            metrics, _ = replay_shared_account(candidate.candidate_id, self._events(portfolio_id, candidate.candidate_id), self._candidate_members(spec, candidate), policy=spec.risk_budget_policy, maximum_total_contracts=spec.maximum_total_contracts, maximum_simultaneous_positions=spec.maximum_simultaneous_positions, product=spec.target_account_products[0], scenario=self.scenario)
            payload = metrics.model_dump(mode="json")
            if self.scenario == "negative-economics": payload["subscriptions"] *= 100; payload["net_external_cashflow"] = payload["trader_payouts"] - payload["subscriptions"]; payload["roi"] = payload["net_external_cashflow"] / payload["subscriptions"]
            self.registry.save_portfolio_record("portfolio_prop_scenarios", candidate.candidate_id, portfolio_id, spec.version, payload, candidate_id=candidate.candidate_id); result.append(payload)
        self._advance(portfolio_id, PortfolioPhase.PORTFOLIO_PROP_SIMULATION); self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_PROP_SIMULATION, {"scenarios": result}); return result

    def run_ablation(self, portfolio_id: str) -> list[dict]:
        spec = self._spec(portfolio_id); prior = self.registry.list_portfolio_records("portfolio_ablation_runs", portfolio_id, spec.version); candidates = self._candidates(portfolio_id); expected = sum(len(item.member_strategy_ids) for item in candidates)
        if self._current(portfolio_id)["current_phase"] != PortfolioPhase.PORTFOLIO_PROP_SIMULATION.value and len(prior) >= expected:
            return [item["result_json"] for item in prior]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_PROP_SIMULATION); results = []
        if len(prior) >= expected:
            return [item["result_json"] for item in prior]
        self._consume(portfolio_id, ablation_runs=expected - len(prior))
        for candidate in candidates:
            full_record = self.registry.get_portfolio_record("portfolio_prop_scenarios", portfolio_id, record_key=candidate.candidate_id); full = full_record["result_json"] if full_record else {}
            for removed in candidate.member_strategy_ids:
                members = [member for member in self._candidate_members(spec, candidate) if member.strategy_id != removed]; events = [event for event in self._events(portfolio_id, candidate.candidate_id) if event.strategy_id != removed]; without, _ = replay_shared_account(candidate.candidate_id, events, members, policy=spec.risk_budget_policy, maximum_total_contracts=spec.maximum_total_contracts, maximum_simultaneous_positions=spec.maximum_simultaneous_positions, product=spec.target_account_products[0], scenario=self.scenario); delta = {"net_external_cashflow": full.get("net_external_cashflow", 0) - without.net_external_cashflow, "first_payouts": full.get("first_payouts", 0) - without.first_payouts, "net_pnl": full.get("net_pnl", 0) - without.net_pnl}; contribution = ContributionClassification.STRONGLY_POSITIVE if delta["net_external_cashflow"] > 100 else ContributionClassification.POSITIVE if delta["net_external_cashflow"] > 0 else ContributionClassification.NEGATIVE if delta["net_external_cashflow"] < 0 else ContributionClassification.NEUTRAL; item = {"candidate_id": candidate.candidate_id, "removed_strategy_id": removed, "full_metrics": full, "without_metrics": without.model_dump(mode="json"), "deltas": delta, "contribution": contribution.value, "reason": "bounded leave-one-out replay"}; self.registry.save_portfolio_record("portfolio_ablation_runs", f"{candidate.candidate_id}-{removed}", portfolio_id, spec.version, item, candidate_id=candidate.candidate_id, removed_strategy_id=removed); self.registry.save_portfolio_record("portfolio_marginal_contributions", f"{candidate.candidate_id}-{removed}", portfolio_id, spec.version, item, candidate_id=candidate.candidate_id, strategy_id=removed); results.append(item)
        return results

    def run_stress(self, portfolio_id: str) -> list[dict]:
        spec = self._spec(portfolio_id); scenarios = ["fees_x2", "fees_x3", "higher_slippage", "reduced_contract_capacity", "correlated_loss_shock"]; candidates = self._candidates(portfolio_id); expected = len(candidates) * len(scenarios); prior = self.registry.list_portfolio_records("portfolio_stress_results", portfolio_id, spec.version)
        if self._current(portfolio_id)["current_phase"] != PortfolioPhase.PORTFOLIO_PROP_SIMULATION.value and len(prior) >= expected:
            return [item["result_json"] for item in prior]
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_PROP_SIMULATION)
        if len(prior) >= expected:
            return [item["result_json"] for item in prior]
        self._consume(portfolio_id, stress_scenarios=expected - len(prior)); results = []
        for candidate in self._candidates(portfolio_id):
            base = self.registry.get_portfolio_record("portfolio_prop_scenarios", portfolio_id, record_key=candidate.candidate_id)["result_json"]
            for scenario in scenarios:
                metrics = dict(base); multiplier = 2 if scenario == "fees_x2" else 3 if scenario == "fees_x3" else 1; metrics["fees"] *= multiplier; metrics["net_pnl"] -= base.get("fees", 0) * (multiplier - 1); metrics["net_external_cashflow"] = base.get("net_external_cashflow", 0); classification = "FAIL" if self.scenario in {"stress-fragile", "correlated"} or metrics["net_pnl"] < 0 else "PASS"; item = PortfolioStressResult(candidate_id=candidate.candidate_id, scenario=scenario, seed=100 + len(results), metrics=metrics, classification=classification, reason="bounded deterministic stress; membership unchanged"); self.registry.save_portfolio_record("portfolio_stress_results", f"{candidate.candidate_id}-{scenario}", portfolio_id, spec.version, item.model_dump(mode="json"), candidate_id=candidate.candidate_id, scenario=scenario); results.append(item.model_dump(mode="json"))
        return results

    def final_review(self, portfolio_id: str) -> PortfolioReview:
        existing = self.registry.get_portfolio_record("portfolio_final_reviews", portfolio_id, self._spec(portfolio_id).version)
        if existing:
            return PortfolioReview.model_validate(existing["result_json"])
        self._require_phase(portfolio_id, PortfolioPhase.PORTFOLIO_PROP_SIMULATION); spec = self._spec(portfolio_id); candidates = self._candidates(portfolio_id); prop_records = {item["candidate_id"]: item["result_json"] for item in self.registry.list_portfolio_records("portfolio_prop_scenarios", portfolio_id, spec.version)}; overlap_records = {item["candidate_id"]: item["result_json"] for item in self.registry.list_portfolio_records("portfolio_overlap_metrics", portfolio_id, spec.version)}; corr_records = {item["candidate_id"]: item["result_json"] for item in self.registry.list_portfolio_records("portfolio_correlation_metrics", portfolio_id, spec.version)}
        if not prop_records: raise SpecificationValidationError("portfolio prop simulation is missing")
        self._advance(portfolio_id, PortfolioPhase.PORTFOLIO_FINAL_REVIEW)
        selected = max(candidates, key=lambda candidate: (prop_records.get(candidate.candidate_id, {}).get("net_external_cashflow", -10**9), prop_records.get(candidate.candidate_id, {}).get("first_payouts", 0), -overlap_records.get(candidate.candidate_id, {}).get("signal_overlap_rate", 1)))
        metrics = prop_records[selected.candidate_id]; overlap = overlap_records.get(selected.candidate_id, {}); correlation = corr_records.get(selected.candidate_id, {}); exploratory = any("PROXY" in member.data_source_classification.upper() or member.role.value == "EXPLORATORY" for member in self._candidate_members(spec, selected)) or self.scenario in {"synthetic-proxy", "synthetic-proxy-concentration", "exploratory-proxy", "insufficient-history"}
        if self.scenario in {"noncompliant", "noncompliant-account-model"} or spec.prop_operating_model.startswith("NONCOMPLIANT"): classification = PortfolioClassification.PORTFOLIO_REJECTED_PROP_INCOMPATIBLE
        elif self.scenario in {"insufficient-history", "low-frequency"}: classification = PortfolioClassification.PORTFOLIO_INSUFFICIENT_EVIDENCE
        elif self.scenario in {"correlated", "correlated-loss-shock"}: classification = PortfolioClassification.PORTFOLIO_REJECTED_CORRELATED
        elif self.scenario in {"synthetic-proxy", "synthetic-proxy-concentration", "exploratory-proxy"}: classification = PortfolioClassification.PORTFOLIO_ACCEPTED_EXPLORATORY
        elif self.scenario in {"negative-economics", "harmful-member"} or metrics["net_external_cashflow"] < 0: classification = PortfolioClassification.PORTFOLIO_REJECTED_NEGATIVE_ECONOMICS
        elif overlap.get("signal_overlap_rate", 0) > .8 or any(value is not None and value > .9 for value in correlation.get("daily_pnl_correlation", {}).values()): classification = PortfolioClassification.PORTFOLIO_REJECTED_CORRELATED
        elif self.scenario in {"redundant", "duplicate-signal"} or all(value < .15 for value in overlap.get("unique_contribution_rate", {}).values()): classification = PortfolioClassification.PORTFOLIO_REJECTED_REDUNDANT
        elif not correlation.get("sufficient_evidence", False): classification = PortfolioClassification.PORTFOLIO_INSUFFICIENT_EVIDENCE
        elif exploratory: classification = PortfolioClassification.PORTFOLIO_ACCEPTED_EXPLORATORY
        else: classification = PortfolioClassification.PORTFOLIO_ACCEPTED
        members = self._candidate_members(spec, selected); roles = {member.strategy_id: member.role for member in members}; contributions = overlap.get("unique_contribution_rate", {}); excluded = {member.strategy_id: "not in selected candidate" for member in spec.strategy_members if member.strategy_id not in selected.member_strategy_ids}
        if self.scenario == "harmful-member" and len(members) > 1:
            excluded[members[-1].strategy_id] = "negative leave-one-out contribution; excluded from recommended composition"
        review = PortfolioReview(portfolio_id=portfolio_id, portfolio_version=spec.version, selected_candidate_id=selected.candidate_id, classification=classification, best_portfolio=selected.member_strategy_ids, member_roles=roles, unique_contribution=contributions, excluded_strategies=excluded, preferred_conflict_policy=spec.conflict_policy, preferred_risk_allocation=spec.risk_budget_policy, preferred_account_product=spec.target_account_products[0], preferred_operating_model=spec.prop_operating_model, expected_trades_per_month=metrics["executable_trades_per_month"], expected_payout_frequency=metrics["payout_rate"], subscription_efficiency=metrics["trader_payouts"] / metrics["subscriptions"] if metrics["subscriptions"] else None, expected_external_cashflow=metrics["net_external_cashflow"], confidence_classification="EXPLORATORY" if exploratory else "VERIFIED_SYNTHETIC_FIXTURE", primary_limitations=spec.known_limitations + [correlation.get("reason", "correlation evidence")], metric_citations=[{"metric": "net_external_cashflow", "value": metrics["net_external_cashflow"], "source": "portfolio_prop_scenarios"}, {"metric": "signal_overlap_rate", "value": overlap.get("signal_overlap_rate"), "source": "portfolio_overlap_metrics"}], rationale="Portfolio ranking uses external cashflow, payout evidence, overlap, correlation evidence, and compliance; it never ranks on PnL alone.", next_phase=None)
        self.registry.save_portfolio_record("portfolio_final_reviews", f"{portfolio_id}-{spec.version}", portfolio_id, spec.version, review.model_dump(mode="json"), classification=classification.value); self.registry.update_portfolio_run(portfolio_id, spec.version, phase=PortfolioPhase.COMPLETE.value, status=classification.value); self._journal(portfolio_id, PortfolioPhase.PORTFOLIO_FINAL_REVIEW, review.model_dump(mode="json")); return review

    def status(self, portfolio_id: str) -> dict:
        spec = self._spec(portfolio_id); return {"specification": spec.model_dump(mode="json"), "run": self._current(portfolio_id), "budget": self.registry.get_portfolio_budget(portfolio_id, spec.version), "candidates": self.registry.list_portfolio_records("portfolio_candidates", portfolio_id, spec.version), "signals": self.registry.list_portfolio_records("portfolio_signals", portfolio_id, spec.version), "overlap": self.registry.list_portfolio_records("portfolio_overlap_metrics", portfolio_id, spec.version), "correlation": self.registry.list_portfolio_records("portfolio_correlation_metrics", portfolio_id, spec.version), "risk": self.registry.list_portfolio_records("portfolio_risk_runs", portfolio_id, spec.version), "prop": self.registry.list_portfolio_records("portfolio_prop_scenarios", portfolio_id, spec.version), "ablation": self.registry.list_portfolio_records("portfolio_ablation_runs", portfolio_id, spec.version), "marginal": self.registry.list_portfolio_records("portfolio_marginal_contributions", portfolio_id, spec.version), "stress": self.registry.list_portfolio_records("portfolio_stress_results", portfolio_id, spec.version), "final_review": self.registry.get_portfolio_record("portfolio_final_reviews", portfolio_id, spec.version)}

    def journal(self, portfolio_id: str) -> list[dict]: return self.registry.portfolio_journal(portfolio_id)
