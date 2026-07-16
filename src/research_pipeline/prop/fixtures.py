from __future__ import annotations

from pathlib import Path

from ..research.fixtures import run_phase_c_dry_run
from .services import PropResearchService


def run_prop_dry_run(registry_path: str | Path, repository_root: str | Path, strategy_id: str, scenario: str = "profitable", product: str = "Alpha Futures Zero 25K") -> dict:
    # A completed fixture is a durable idempotency result, not a reason to
    # restart Phase C against its terminal state.
    existing = PropResearchService(registry_path, repository_root=repository_root, scenario=scenario)
    try:
        current = existing._prop_run(strategy_id)
        if current["current_phase"] == "COMPLETE":
            review = existing.registry.get_prop_record("prop_final_reviews", strategy_id)
            return {"strategy_id": strategy_id, "scenario": scenario, "final_classification": (review or {}).get("classification", current["status"]), "holdout_accesses": existing.registry.count_holdout_accesses(strategy_id), "journal_entries": len(existing.journal(strategy_id)), "idempotent": True, "status": existing.status(strategy_id)}
    except Exception:
        pass
    markets = ["UNSUPPORTED"] if scenario == "unsupported-mapping" else ["TEST"] if scenario == "synthetic-proxy" else ["BTCUSDT"]
    phase_c_scenario = "strong-stable"
    run_phase_c_dry_run(registry_path, repository_root, strategy_id, phase_c_scenario, markets=markets)
    service = PropResearchService(registry_path, repository_root=repository_root, scenario=scenario)
    service.start(strategy_id, f"prop-dry-run-{strategy_id}-{scenario}")
    rules = service.verify_rules(strategy_id, product)
    if rules["status"] != "VERIFIED": return {"strategy_id": strategy_id, "scenario": scenario, "final_classification": "MANUAL_REVIEW_REQUIRED", "status": service.status(strategy_id)}
    contracts = service.verify_contracts(strategy_id)
    if contracts.get("errors"): return {"strategy_id": strategy_id, "scenario": scenario, "final_classification": "INSUFFICIENT_FUTURES_DATA", "status": service.status(strategy_id)}
    service.reconcile(strategy_id)
    service.run_risk(strategy_id, product)
    simulation = service.run_scenarios(strategy_id, product)
    if service._prop_run(strategy_id)["current_phase"] == "PROP_ECONOMICS_REVIEW":
        review = service.economics(strategy_id)
        classification = review.classification.value
    else: classification = service._prop_run(strategy_id)["status"]
    return {"strategy_id": strategy_id, "scenario": scenario, "final_classification": classification, "holdout_accesses": service.registry.count_holdout_accesses(strategy_id), "journal_entries": len(service.journal(strategy_id)), "simulation": simulation}
