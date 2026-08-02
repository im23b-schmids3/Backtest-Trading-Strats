from __future__ import annotations

from typing import Any

from ..registry.repositories import Registry

ALLOWED_PHASE_C = {"ACCEPTED_STANDALONE", "ACCEPTED_PORTFOLIO_COMPONENT"}
ALLOWED_PHASE_D = {"PROP_ACCEPTED_STANDALONE", "PROP_ACCEPTED_PORTFOLIO_COMPONENT", "OWN_CAPITAL_ONLY", "INSUFFICIENT_PROP_EVIDENCE"}
BLOCKED_PHASE_D = {"REJECTED_PROP_INCOMPATIBLE", "REJECTED_NEGATIVE_ECONOMICS", "TECHNICAL_FAILURE", "MANUAL_REVIEW_REQUIRED"}


def eligibility(registry: Registry, strategy_id: str, *, exploratory_prop: bool = False, non_prop: bool = False) -> dict[str, Any]:
    try:
        strategy = registry.get_strategy(strategy_id)
        final = registry.get_research_json("research_final_reviews", strategy_id)
        candidate = registry.get_candidate(strategy_id)
        verification = registry.get_verification(strategy_id)
        prop = registry.get_prop_record("prop_final_reviews", strategy_id)
    except Exception as exc:
        return {"strategy_id": strategy_id, "eligible": False, "reasons": [f"registry lookup failed: {exc}"]}
    reasons: list[str] = []
    phase_c = (final or {}).get("classification")
    phase_d = (prop or {}).get("result_json", {}).get("classification")
    if strategy.get("approval_status") not in {"APPROVED", "APPROVED_IMMUTABLE"} and strategy.get("current_phase") not in {"ACCEPTED", "FINAL_REVIEW"}:
        reasons.append("strategy specification is not approved")
    if strategy.get("parameters_frozen") != 1: reasons.append("candidate parameters are not frozen")
    if not candidate or not candidate.get("candidate_hash"): reasons.append("frozen candidate hash missing")
    if not verification or verification.get("outcome") != "VERIFIED": reasons.append("B.5 verification is not VERIFIED")
    if not all((registry.get_research_json(table, strategy_id) or {}).get("status") in {"PASS", "COMPLETED"} or (registry.get_research_json(table, strategy_id) or {}).get("classification") not in {"REJECTED", "FAIL"} for table in ("research_walk_forward", "research_holdout", "research_stress", "research_throughput")):
        reasons.append("Phase C validation evidence is incomplete")
    if phase_c not in ALLOWED_PHASE_C: reasons.append(f"Phase C classification is not portfolio eligible: {phase_c}")
    if phase_d in BLOCKED_PHASE_D or (phase_d not in ALLOWED_PHASE_D and not non_prop): reasons.append(f"Phase D classification is not portfolio eligible: {phase_d}")
    if phase_d == "OWN_CAPITAL_ONLY" and not non_prop: reasons.append("OWN_CAPITAL_ONLY requires explicit non-prop mode")
    if phase_d == "INSUFFICIENT_PROP_EVIDENCE" and not exploratory_prop and not non_prop: reasons.append("insufficient prop evidence requires exploratory mode")
    return {"strategy_id": strategy_id, "strategy_version": strategy.get("version"), "candidate_hash": (candidate or {}).get("candidate_hash"), "phase_c_classification": phase_c, "phase_d_classification": phase_d, "eligible": not reasons, "reasons": reasons}


def eligible_strategy_ids(registry: Registry, *, exploratory_prop: bool = False, non_prop: bool = False) -> list[dict[str, Any]]:
    return [eligibility(registry, item["strategy_id"], exploratory_prop=exploratory_prop, non_prop=non_prop) for item in registry.list_strategies()]
