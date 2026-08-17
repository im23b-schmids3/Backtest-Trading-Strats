"""Sealed, predeclared POC-only L2 V3 validation contract.

V3 is deliberately not a replay entry point.  It records the one permitted
semantic change from the frozen V2 quality-0.50 contract: only prior-RTH POC
may instantiate a structural interaction.  All scoring and execution literals
are inherited byte-for-byte from V2.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from . import v2_quality050 as v2
from .model import LEVEL_NAMES, StructuralLevel


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY"
PARENT_STRATEGY_ID = v2.STRATEGY_ID
VARIANT_LABEL = "L2_V3_POC_ONLY_PREDECLARED"
EVIDENCE_LABEL = "POST_HOC_POC_ONLY_PREDECLARED_FUTURE_VALIDATION_NOT_VALIDATED"
ELIGIBLE_STRUCTURAL_LEVELS = ("PRIOR_RTH_POC",)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def filter_eligible_levels(levels: Iterable[StructuralLevel]) -> tuple[StructuralLevel, ...]:
    """Select only the predeclared prior-session POC without changing levels."""
    selected: list[StructuralLevel] = []
    for level in levels:
        if level.name not in LEVEL_NAMES:
            raise ValueError(f"unknown structural level: {level.name}")
        if level.name in ELIGIBLE_STRUCTURAL_LEVELS:
            selected.append(level)
    return tuple(selected)


def v2_to_v3_contract_diff() -> dict[str, Any]:
    """Machine-readable proof that structural eligibility is the sole change."""
    return {
        "parent_strategy_id": PARENT_STRATEGY_ID,
        "parent_contract_sha256": v2.v2_contract_sha256(),
        "child_strategy_id": STRATEGY_ID,
        "changed_strategy_fields": [{
            "field": "eligible_structural_levels",
            "v2": list(LEVEL_NAMES),
            "v3": list(ELIGIBLE_STRUCTURAL_LEVELS),
        }],
        "only_structural_eligibility_changed": True,
        "unchanged": {
            "configuration": v2.v2_contract()["configuration"],
            "execution": v2.v2_contract()["execution"],
            "feature_and_score_semantics": "inherited_from_CMEOrderflowAbsorption.ES_L2_V2",
            "interaction_lifecycle": "inherited_from_CMEOrderflowAbsorption.ES_L2_V2",
        },
    }


def v3_contract() -> dict[str, Any]:
    """Canonical pre-run V3 contract; no outcome claim is encoded here."""
    inherited = v2.v2_contract()
    return {
        "strategy_id": STRATEGY_ID,
        "parent_strategy_id": PARENT_STRATEGY_ID,
        "variant_label": VARIANT_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "eligible_structural_levels": list(ELIGIBLE_STRUCTURAL_LEVELS),
        "configuration": inherited["configuration"],
        "execution": inherited["execution"],
        "parent_contract_sha256": v2.v2_contract_sha256(),
        "v2_to_v3_contract_diff": v2_to_v3_contract_diff(),
        "post_hoc_context_only": {
            "may_2026": "May 6 POC trades: +4.4091R; DEVELOPMENT evidence only",
            "retro_june_july_2026": "8 POC trades: +2.6756R; 0 unresolved; retrospective robustness only",
            "seen_august_2026": "1 accepted POC setup; 0 confirmations/trades; seen data only",
        },
        "selection_prohibited": True,
        "outcome_parameter_selection": False,
        "requires_fresh_untouched_validation": True,
    }


def v3_contract_sha256() -> str:
    return _canonical_hash(v3_contract())

