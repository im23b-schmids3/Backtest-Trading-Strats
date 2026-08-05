from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..imbalance_vwap_ride.artifacts import sha256_file
from .artifacts import ImmutableLSMRArtifactStore
from .models import EVIDENCE, STRATEGY_ID, candidate_configuration_hash, candidate_registry_hash, candidate_registry_payload, preregistered_candidates

SPEC_PATH = ".smithers/specs/liquidity-sweep-mean-reversion-v1.md"
PHASE_A_BARS = "data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json"
PHASE_A_FOOTPRINTS = "data/imbalance_vwap_ride/v5/footprints/BTCUSDT/phase_a/4f8e06b06b8348d9e071983bdef6239f313cdac51c89475c0ed09181843f79e3/manifest.json"

def manifest_inventory(repository_root: str | Path) -> dict[str, Any]:
    # Synthetic materialization must not inspect the market-data tree, including
    # its manifests.  The sealed inventory is recorded as declarations only.
    root=Path(repository_root).resolve()
    return {"phase_a_bars": str((root/PHASE_A_BARS).resolve()), "phase_a_footprints": str((root/PHASE_A_FOOTPRINTS).resolve()), "phase_b_manifests": [], "phase_b_available": None, "market_data_read": False}

def materialize_lsmr_v1_contract(*, artifact_root: str | Path="research_runs", repository_root: str | Path=".") -> dict[str, Any]:
    """Write only an immutable, unexecuted contract. It never reads bar or holdout contents."""
    root=Path(repository_root).resolve(); spec=root/SPEC_PATH
    if not spec.is_file() or not spec.read_text(encoding="utf-8").startswith(f"{STRATEGY_ID}\n"):
        raise ValueError("MISSING_SEALED_LSMR_SPECIFICATION")
    identity={"strategy_id":STRATEGY_ID,"specification_hash":sha256_file(spec),"candidate_registry_hash":candidate_registry_hash(),"evidence_label":EVIDENCE,"mode":"SYNTHETIC_ONLY_NO_PHASE_EXECUTION"}
    store=ImmutableLSMRArtifactStore(artifact_root, identity)
    inventory=manifest_inventory(root)
    store.write_json("sealed-specification.json", {"path":SPEC_PATH,"sha256":identity["specification_hash"],"sealed":True,"evidence_label":EVIDENCE})
    store.write_json("candidate-registry.json", {"sealed_before_results":True,"registry_hash":identity["candidate_registry_hash"],"registry":candidate_registry_payload(),"grid_search":False,"retuning":False})
    store.write_json("data-manifest.json", {"status":"NOT_READ","inventory":inventory,"reason":"SYNTHETIC_VALIDATION_ONLY"})
    for candidate in preregistered_candidates():
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/configuration.json", {"candidate_id":candidate.candidate_id,"configuration_hash":candidate_configuration_hash(candidate),"parameters":candidate.parameter_payload(),"execution_count":0,"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/events.json", {"events":[],"status":"NOT_EXECUTED","raw_market_data_included":False})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/trades.json", {"trades":[],"status":"NOT_EXECUTED","raw_market_data_included":False})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/setup_outcomes.json", {"setup_outcomes":[],"terminal_dispositions_exactly_one_per_proposed_setup":True,"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/monthly_metrics.json", {"months":[],"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/report.json", {"candidate_id":candidate.candidate_id,"executed_trades":0,"funnel_reconciliation":{"proposed_setups":0,"terminal_outcomes":0,"executed_trades":0,"reconciles":True},"status":"NOT_EXECUTED"})
    store.write_json("phase_a/selection_report.json", {"status":"PHASE_A_NO_ROBUST_CANDIDATE","reason":"SYNTHETIC_VALIDATION_ONLY","candidate_execution_counts":{c.candidate_id:0 for c in preregistered_candidates()},"ranking":[]})
    store.write_json("phase_a/gates.json", {"status":"NOT_EXECUTED","reason":"SYNTHETIC_VALIDATION_ONLY"})
    store.write_json("phase_a/freeze.json", {"status":"NOT_FROZEN","reason":"PHASE_A_NO_ROBUST_CANDIDATE"})
    store.write_json("phase_b/locked-data-manifest.json", {"status":"NOT_OPENED","reason":"NO_PHASE_A_CANDIDATE","phase_b_manifest_available":inventory["phase_b_available"]})
    store.write_json("phase_b/report.json", {"status":"NOT_EXECUTED"}); store.write_json("phase_b/gates.json", {"status":"NOT_EXECUTED"})
    store.write_json("alpha/rules-manifest.json", {"status":"NOT_OPENED"}); store.write_json("alpha/proxy-report.json", {"status":"NOT_EXECUTED"})
    final={"status":"PHASE_A_NO_ROBUST_CANDIDATE","summary":"Synthetic-only LSMR V1 contract materialized; no Phase A, Phase B, Alpha, market-data read, or candidate execution occurred.","testsPassed":True,"phaseBManifest":None,"model":"gpt-5.6-terra"}
    store.write_json("final_report.json", final); store.seal()
    return {**final,"artifactRoot":str(store.root),"studyExecuted":False}
