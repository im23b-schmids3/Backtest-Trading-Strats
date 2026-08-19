"""Sealed one-field challenger to the frozen POC-only L2 V3 contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from . import v3_poc_only as v3
from .v2_quality050 import V2_CONFIG


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V4_POC_ONLY_Q45"
PARENT_STRATEGY_ID = v3.STRATEGY_ID
PARENT_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
VARIANT_LABEL = "L2_V4_POC_ONLY_Q45_DEC2025_JAN2026"
EVIDENCE_LABEL = "PREDECLARED_Q45_CHALLENGER_DEC2025_JAN2026_RETROSPECTIVE_RESEARCH"
ELIGIBLE_STRUCTURAL_LEVELS = v3.ELIGIBLE_STRUCTURAL_LEVELS
V4_CONFIG = replace(V2_CONFIG, min_quality_score=0.45)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def v3_to_v4_contract_diff() -> dict[str, Any]:
    """Prove that quality acceptance is the only changed strategy field."""
    if v3.v3_contract_sha256() != PARENT_CONTRACT_SHA256:
        raise RuntimeError("frozen V3 parent contract hash changed")
    parent = asdict(V2_CONFIG)
    child = asdict(V4_CONFIG)
    changed = [
        {"field": name, "v3": parent[name], "v4": child[name]}
        for name in sorted(parent)
        if parent[name] != child[name]
    ]
    expected = [{"field": "min_quality_score", "v3": 0.50, "v4": 0.45}]
    if changed != expected:
        raise RuntimeError("V4 must differ from V3 only at min_quality_score")
    parent_contract = v3.v3_contract()
    return {
        "parent_strategy_id": PARENT_STRATEGY_ID,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "child_strategy_id": STRATEGY_ID,
        "changed_strategy_fields": changed,
        "only_min_quality_score_changed": True,
        "unchanged_strategy_configuration_fields": [
            name for name in sorted(parent) if name != "min_quality_score"
        ],
        "unchanged_semantics": {
            "eligible_structural_levels": parent_contract["eligible_structural_levels"],
            "execution": parent_contract["execution"],
            "score_weights_and_penalties": {
                name: parent[name]
                for name in sorted(parent)
                if name.endswith("_weight") or name.endswith("_component_weight")
            },
            "interaction_lifecycle": "inherited_from_CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY",
            "calendar_and_dst": "inherited_from_finalized_Dec2025_Jan2026_V3_replay",
            "position_state": "independent_chronological_V4_replay; one_position_at_a_time",
        },
    }


def v4_contract() -> dict[str, Any]:
    parent = v3.v3_contract()
    return {
        "strategy_id": STRATEGY_ID,
        "parent_strategy_id": PARENT_STRATEGY_ID,
        "variant_label": VARIANT_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "eligible_structural_levels": list(ELIGIBLE_STRUCTURAL_LEVELS),
        "configuration": asdict(V4_CONFIG),
        "execution": parent["execution"],
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "v3_to_v4_contract_diff": v3_to_v4_contract_diff(),
        "strict_chronological_oos": False,
        "declared_after_v3_dec_jan_result": True,
        "purpose": "test whether the frozen 0.50 quality gate is unnecessarily selective",
        "selection_prohibited": True,
        "outcome_parameter_selection": False,
        "no_further_threshold_search": True,
    }


def v4_contract_sha256() -> str:
    return _canonical_hash(v4_contract())


def contract_artifact() -> dict[str, Any]:
    return {
        "contract": v4_contract(),
        "contract_sha256": v4_contract_sha256(),
        "contract_diff": v3_to_v4_contract_diff(),
    }


def write_contract_artifact(path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"immutable V4 contract artifact already exists: {path}")
    payload = contract_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
