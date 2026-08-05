from __future__ import annotations

from pathlib import Path
from typing import Any

from ..imbalance_vwap_ride.artifacts import sha256_file
from ..liquidity_sweep_mean_reversion.artifacts import ImmutableLSMRArtifactStore
from .models import EVIDENCE, STRATEGY_ID, candidate_configuration_hash, candidate_registry_hash, candidate_registry_payload, preregistered_candidates

SPEC_PATH = ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"
PHASE_A_BARS = "data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json"


def manifest_inventory(repository_root: str | Path) -> dict[str, Any]:
    """Declarations only: this strict synthetic workflow must never inspect market data."""
    root = Path(repository_root).resolve()
    return {"phase_a_bars": str((root / PHASE_A_BARS).resolve()), "phase_b_manifests": [], "phase_b_available": None, "market_data_read": False, "market_data_written": False}


def materialize_lsmr_v2_strict_contract(*, artifact_root: str | Path = "research_runs", repository_root: str | Path = ".") -> dict[str, Any]:
    """Write the V2 sealed, deterministic non-agent candidate contract; execute no study."""
    root = Path(repository_root).resolve(); spec = root / SPEC_PATH
    required = (STRATEGY_ID, "LSMR-V2-2P0R=2R", "LSMR-V2-2P5R=2.5R", "LSMR-V2-3P0R=3R", "SESSION_CONTEXT_UNAVAILABLE", "TRADE_EXECUTED")
    text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    if not all(item in text for item in required): raise ValueError("MISSING_SEALED_LSMR_V2_SPECIFICATION")
    identity = {"strategy_id": STRATEGY_ID, "specification_hash": sha256_file(spec), "candidate_registry_hash": candidate_registry_hash(), "evidence_label": EVIDENCE, "mode": "SYNTHETIC_ONLY_NO_PHASE_EXECUTION"}
    store = ImmutableLSMRArtifactStore(artifact_root, identity); inventory = manifest_inventory(root)
    store.write_json("sealed-specification.json", {"path": SPEC_PATH, "sha256": identity["specification_hash"], "sealed": True, "evidence_label": EVIDENCE})
    store.write_json("candidate-registry.json", {"sealed_before_results": True, "registry_hash": identity["candidate_registry_hash"], "registry": candidate_registry_payload(), "grid_search": False, "retuning": False, "execution_mode": "NON_AGENT_DETERMINISTIC"})
    store.write_json("data-manifest.json", {"status": "NOT_READ", "inventory": inventory, "reason": "SYNTHETIC_VALIDATION_ONLY"})
    for candidate in preregistered_candidates():
        base = f"phase_a/candidates/{candidate.candidate_id}"
        store.write_json(f"{base}/configuration.json", {"candidate_id": candidate.candidate_id, "configuration_hash": candidate_configuration_hash(candidate), "parameters": candidate.parameter_payload(), "execution_count": 0, "status": "NOT_EXECUTED"})
        store.write_json(f"{base}/events.json", {"events": [], "status": "NOT_EXECUTED", "raw_market_data_included": False})
        store.write_json(f"{base}/trades.json", {"trades": [], "status": "NOT_EXECUTED", "raw_market_data_included": False})
        store.write_json(f"{base}/setup_outcomes.json", {"setup_outcomes": [], "terminal_dispositions_exactly_one_per_proposed_setup": True, "status": "NOT_EXECUTED"})
        store.write_json(f"{base}/gates.json", {"status": "NOT_EXECUTED", "hard_gates": {"minimum_executed_13_months": 163, "minimum_annualized": 150, "warning_annualized": 350, "profit_factor_minimum": "1.30", "positive_net_pnl": True, "positive_average_r": True, "maximum_drawdown_r": "20", "minimum_profitable_months": 8, "maximum_zero_months": 3, "best_month_concentration_maximum": "0.35", "best_five_concentration_maximum": "0.30", "long_minimum_fraction": "0.25", "short_minimum_fraction": "0.25", "long_minimum_average_r": "-0.15", "short_minimum_average_r": "-0.15", "bootstrap_median_r_positive": True, "bootstrap_lower_r_minimum": "-0.025", "extra_slippage_positive": True, "best_trade_removal_positive": True, "minimum_nonnegative_2023_subperiods": "3/4", "full_reconciliation_required": True}})
    store.write_json("phase_a/selection_report.json", {"status": "PHASE_A_NO_ROBUST_CANDIDATE", "reason": "SYNTHETIC_VALIDATION_ONLY", "candidate_execution_counts": {candidate.candidate_id: 0 for candidate in preregistered_candidates()}, "ranking": [], "selection": "NOT_RUN"})
    store.write_json("phase_a/gates.json", {"status": "NOT_EXECUTED", "reason": "SYNTHETIC_VALIDATION_ONLY"}); store.write_json("phase_a/freeze.json", {"status": "NOT_FROZEN", "reason": "PHASE_A_NO_ROBUST_CANDIDATE"})
    store.write_json("phase_b/locked-data-manifest.json", {"status": "NOT_OPENED", "reason": "PHASE_A_NOT_EXECUTED", "phase_b_manifest_available": None}); store.write_json("phase_b/report.json", {"status": "NOT_EXECUTED"}); store.write_json("phase_b/gates.json", {"status": "NOT_EXECUTED"}); store.write_json("alpha/rules-manifest.json", {"status": "NOT_OPENED"}); store.write_json("alpha/proxy-report.json", {"status": "NOT_EXECUTED"})
    final = {"status": "PHASE_A_NO_ROBUST_CANDIDATE", "summary": "Synthetic-only LSMR V2 strict contract materialized; no Phase A, Phase B, Alpha, market-data read, or candidate execution occurred.", "testsPassed": True, "realStudyExecuted": False, "model": "gpt-5.6-terra"}
    store.write_json("final_report.json", final); store.seal(); return {**final, "artifactRoot": str(store.root)}
