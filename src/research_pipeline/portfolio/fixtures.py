from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..prop.fixtures import run_prop_dry_run
from .models import ConflictPolicy, PortfolioMember, PortfolioMemberRole, PortfolioSpec, RiskAllocationPolicy
from .service import PortfolioService

PORTFOLIO_FIXTURE_SCENARIOS = {
    "complementary", "correlated", "duplicate-signal", "opposite-conflicts",
    "low-frequency", "harmful-member", "redundant", "negative-economics",
    "payout-improving", "noncompliant-account-model", "insufficient-history",
    "synthetic-proxy-concentration", "contract-limit-congestion", "stress-fragile",
    "exploratory-proxy", "synthetic-proxy", "noncompliant",
}


def make_portfolio_spec(registry, portfolio_id: str, scenario: str) -> PortfolioSpec:
    members = []
    classifications = [PortfolioMemberRole.CORE, PortfolioMemberRole.DIVERSIFIER]
    for index, strategy_id in enumerate((f"{portfolio_id}-a", f"{portfolio_id}-b")):
        strategy = registry.get_strategy(strategy_id); candidate = registry.get_candidate(strategy_id); prop = registry.get_prop_record("prop_final_reviews", strategy_id); final = registry.get_research_json("research_final_reviews", strategy_id)
        data_class = "SYNTHETIC_PROXY_HIGH_UNCERTAINTY" if scenario in {"synthetic-proxy", "synthetic-proxy-concentration", "exploratory-proxy"} else "PROXY_EXPLORATORY"
        role = PortfolioMemberRole.EXPLORATORY if scenario in {"synthetic-proxy", "synthetic-proxy-concentration", "exploratory-proxy"} else classifications[index]
        members.append(PortfolioMember(strategy_id=strategy_id, strategy_version=strategy["version"], candidate_hash=candidate["candidate_hash"], phase_c_classification=final["classification"], phase_d_classification=prop["result_json"]["classification"], markets=["BTCUSDT"], timeframes=["1h"], expected_trades_per_month=20, data_source_classification=data_class, confidence_level="SYNTHETIC_PROXY_HIGH_UNCERTAINTY" if "PROXY" in data_class else "PROXY_EXPLORATORY", role=role, priority=index, confidence_score=.8 - index * .1))
    model = "NONCOMPLIANT_MONTHLY_PIPELINE" if scenario in {"noncompliant", "noncompliant-account-model"} else "MODEL_A_ONE_ACTIVE_EVALUATION"
    return PortfolioSpec(portfolio_id=portfolio_id, version="phase-e-1", name=f"Phase E {scenario}", description="Deterministic synthetic portfolio fixture.", strategy_members=members, strategy_candidate_hashes={item.strategy_id: item.candidate_hash for item in members}, strategy_code_commits={item.strategy_id: None for item in members}, dataset_hashes={item.strategy_id: "synthetic-phase-e" for item in members}, target_markets=["BTCUSDT"], target_timeframes=["1h"], target_account_products=["Alpha Futures Zero 25K"], conflict_policy=ConflictPolicy.SKIP_CONFLICT if scenario == "opposite-conflicts" else ConflictPolicy.FIRST_SIGNAL_WINS, risk_budget_policy=RiskAllocationPolicy.EQUAL_RISK, duplicate_exposure_rules={"crypto_btc": "aggregate_quantities_and_cap"}, maximum_simultaneous_positions=10, maximum_total_contracts=10, maximum_strategy_risk_contribution=200, session_assumptions=["UTC", "fully settled synthetic events"], prop_operating_model=model, known_limitations=["synthetic fixture", "no native futures rollover"], creation_timestamp=datetime.now(timezone.utc), specification_hash="pending")


def run_portfolio_dry_run(registry_path: str | Path, repository_root: str | Path, portfolio_id: str = "phase-e-dry-run", scenario: str = "complementary") -> dict:
    if scenario not in PORTFOLIO_FIXTURE_SCENARIOS:
        raise ValueError(f"unsupported Phase E fixture scenario: {scenario}")
    root = Path(repository_root); registry_path = Path(registry_path)
    for strategy_id in (f"{portfolio_id}-a", f"{portfolio_id}-b"):
        run_prop_dry_run(registry_path, root, strategy_id, "profitable")
    service = PortfolioService(registry_path, root, scenario)
    spec = make_portfolio_spec(service.registry, portfolio_id, scenario); service.create(spec)
    service.generate_candidates(portfolio_id); service.merge_signals(portfolio_id); service.analyze_overlap(portfolio_id); service.analyze_correlation(portfolio_id); service.run_risk(portfolio_id); service.run_prop(portfolio_id); service.run_ablation(portfolio_id); service.run_stress(portfolio_id); review = service.final_review(portfolio_id)
    return {"portfolio_id": portfolio_id, "scenario": scenario, "classification": review.classification.value, "selected_candidate": review.selected_candidate_id, "members": review.best_portfolio, "journal_entries": len(service.journal(portfolio_id)), "status": service.status(portfolio_id)}
