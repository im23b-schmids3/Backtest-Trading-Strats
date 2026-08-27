"""Read-only funnel diagnostic for the frozen Asia W04 baseline.

This module deliberately reuses :mod:`asia_w04_replay` for source binding,
MBO reconstruction, interaction lifecycle, scoring, confirmation, and
execution.  It adds observations and reports only; it does not provide an
alternative strategy path.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import asia_w04_replay as asia


DIAGNOSTIC_ID = "CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY_DIAGNOSTIC"
OUTPUT_ROOT = Path("research_runs") / DIAGNOSTIC_ID
BASELINE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY")
SCHEMA_VERSION = "ASIA_W04_FUNNEL_DIAGNOSTIC_V1"
NY_UNAVAILABLE = "NY_COMPARISON_NOT_AVAILABLE_WITHOUT_REPLAY"
Q = Decimal("0.45")
PENALTY_WEIGHT = Decimal(str(asia.W04_CONFIG.false_refill_penalty_weight))

G_COMPONENTS: tuple[tuple[str, str, Decimal], ...] = (
    ("G1", "aggression_score", Decimal("0.20")),
    ("G2", "restoration_score", Decimal("0.10")),
    ("G3", "price_resistance_score", Decimal("0.30")),
    ("G4", "persistence_score", Decimal("0.20")),
    ("G5", "multi_level_support_score", Decimal("0.20")),
)
PRIMITIVE_REASON_ORDER = (
    "INSUFFICIENT_RELEVANT_AGGRESSION",
    "NO_GENUINE_CONSUME_RESTORE",
    "PRICE_PROGRESS_NOT_RESISTED",
)
REJECTION_REASON_ORDER = (*PRIMITIVE_REASON_ORDER, "L2_QUALITY_BELOW_THRESHOLD")


class AsiaW04DiagnosticError(RuntimeError):
    """Raised when the diagnostic cannot reproduce the frozen population."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        raise AsiaW04DiagnosticError(f"missing or invalid baseline CSV: {path}") from exc


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: Iterable[float | int | Decimal | None]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numbers:
        return {
            "count": 0, "mean": None, "median": None, "p10": None,
            "p25": None, "p75": None, "p90": None, "min": None, "max": None,
        }
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "p10": _percentile(numbers, 0.10),
        "p25": _percentile(numbers, 0.25),
        "p75": _percentile(numbers, 0.75),
        "p90": _percentile(numbers, 0.90),
        "min": min(numbers),
        "max": max(numbers),
    }


def _reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    found = tuple(reason for reason in str(row.get("rejection_reasons") or "").split(";") if reason)
    if any(reason not in REJECTION_REASON_ORDER for reason in found):
        raise AsiaW04DiagnosticError(f"unknown frozen rejection reason: {found}")
    if found != tuple(reason for reason in REJECTION_REASON_ORDER if reason in found):
        raise AsiaW04DiagnosticError(f"frozen rejection reason order changed: {found}")
    return found


def quality_bucket(score: float | Decimal) -> str:
    value = Decimal(str(score))
    if value >= Decimal("0.45"):
        return "ACCEPTED_SCORE"
    if value >= Decimal("0.40"):
        return "NEAR_MISS_0P40_TO_0P45"
    if value >= Decimal("0.30"):
        return "MEDIUM_MISS_0P30_TO_0P40"
    return "LOW_BELOW_0P30"


def weighted_contributions(row: Mapping[str, Any]) -> dict[str, float]:
    output = {
        name: float(Decimal(str(row[field])) * weight)
        for name, field, weight in G_COMPONENTS
    }
    output["FALSE_REFILL_PENALTY"] = -float(
        Decimal(str(row["false_refill_penalty"])) * PENALTY_WEIGHT
    )
    return output


def counterfactual_one_feature_lifts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure Q-distance using one G at a time and the frozen [0, 1] domain.

    Only below-Q rejected rows are included.  Reaching Q here is mathematical;
    it does not cure a primitive hard-gate rejection.
    """
    below_q = [
        row for row in rows
        if not bool(row["accepted"]) and Decimal(str(row["w04_quality_score"])) < Q
    ]
    output: list[dict[str, Any]] = []
    for name, field, weight in G_COMPONENTS:
        feasible_raw: list[float] = []
        feasible_weighted: list[float] = []
        for row in below_q:
            positive = sum(
                Decimal(str(row[other_field])) * other_weight
                for _other_name, other_field, other_weight in G_COMPONENTS
            )
            pre_clamp = positive - Decimal(str(row["false_refill_penalty"])) * PENALTY_WEIGHT
            weighted_gap = max(Decimal("0"), Q - pre_clamp)
            raw_gap = weighted_gap / weight
            if Decimal(str(row[field])) + raw_gap <= Decimal("1") + Decimal("1e-15"):
                feasible_raw.append(float(raw_gap))
                feasible_weighted.append(float(weighted_gap))
        output.append({
            "component": name,
            "field": field,
            "weight": float(weight),
            "below_q_rejected_count": len(below_q),
            "mathematically_reachable_count": len(feasible_raw),
            "median_required_raw_g_increase": statistics.median(feasible_raw) if feasible_raw else None,
            "median_required_weighted_contribution_increase": (
                statistics.median(feasible_weighted) if feasible_weighted else None
            ),
            "interpretation_limit": "QUALITY_ONLY; PRIMITIVE_HARD_GATES_REMAIN_UNCHANGED",
        })
    return output


def _direction(row: Mapping[str, Any]) -> str:
    value = str(row["direction"])
    if value == "BUYER_ABSORPTION":
        return "LONG"
    if value == "SELLER_ABSORPTION":
        return "SHORT"
    raise AsiaW04DiagnosticError(f"interaction lacks a frozen direction: {value}")


def _json_counts(values: Iterable[str]) -> str:
    return json.dumps(dict(sorted(Counter(values).items())), sort_keys=True, separators=(",", ":"))


def _terminal_reason(row: Mapping[str, Any]) -> str:
    reasons = _reasons(row)
    if bool(row["accepted"]):
        if reasons:
            raise AsiaW04DiagnosticError("accepted interaction contains rejection reasons")
        return "SETUP_ACCEPTED"
    if not reasons:
        raise AsiaW04DiagnosticError("rejected interaction has no frozen rejection reason")
    # This is the exact predicate order in asia_w04_replay._interaction_row.
    return reasons[0]


def _valid_five_g(row: Mapping[str, Any]) -> bool:
    for _name, field, _weight in G_COMPONENTS:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return False
    return True


def enrich_interactions(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for session in sessions:
        if not bool(session["eligible"]):
            if session.get("interactions"):
                raise AsiaW04DiagnosticError("profile-only Asia session emitted interactions")
            continue
        indexes = {str(row["interaction_id"]): row for row in session["indexes"]}
        terminals = {str(key): str(value) for key, value in dict(session["terminal_outcomes"]).items()}
        trades = {str(row["interaction_id"]): row for row in session["trades"]}
        for raw in session["interactions"]:
            row = dict(raw)
            identifier = str(row["interaction_id"])
            if identifier in seen:
                raise AsiaW04DiagnosticError(f"duplicate interaction_id: {identifier}")
            seen.add(identifier)
            valid = _valid_five_g(row)
            if not valid:
                raise AsiaW04DiagnosticError(f"invalid five-G vector: {identifier}")
            accepted = bool(row["accepted"])
            setup_terminal = _terminal_reason(row)
            if accepted:
                if identifier not in terminals or identifier not in indexes:
                    raise AsiaW04DiagnosticError(f"accepted interaction lacks lifecycle audit: {identifier}")
                final_terminal = terminals[identifier]
                confirmation_passed = indexes[identifier].get("derived_first_confirmation_timestamp_ns") is not None
                if final_terminal == "TRADE_EXECUTED" and identifier not in trades:
                    raise AsiaW04DiagnosticError(f"executed setup lacks trade: {identifier}")
                if identifier in trades and final_terminal != "TRADE_EXECUTED":
                    raise AsiaW04DiagnosticError(f"trade lacks executed disposition: {identifier}")
            else:
                if identifier in terminals or identifier in trades:
                    raise AsiaW04DiagnosticError(f"rejected interaction leaked into setup lifecycle: {identifier}")
                final_terminal = setup_terminal
                confirmation_passed = False
            duration_ns = int(row["interaction_end_ns"]) - int(row["interaction_start_ns"])
            if duration_ns < 0:
                raise AsiaW04DiagnosticError(f"negative interaction duration: {identifier}")
            contributions = weighted_contributions(row)
            direction = _direction(row)
            start_price = float(row["level_price"])
            end_price = float(row["interaction_end_price"])
            row.update({
                "interaction_start_utc": asia._iso_ns(int(row["interaction_start_ns"])),
                "interaction_end_utc": asia._iso_ns(int(row["interaction_end_ns"])),
                "month": str(row["session_date"])[:7],
                "trade_direction": direction,
                "interaction_duration_ns": duration_ns,
                "interaction_duration_seconds": duration_ns / 1_000_000_000,
                "interaction_start_price": start_price,
                "interaction_start_price_semantics": "FROZEN_LEVEL_ANCHOR_INITIALIZATION",
                "distance_from_prior_asia_poc_points": end_price - float(row["level_price"]),
                "valid_five_g": valid,
                "directional_candidate": True,
                "plus_gate_present_in_frozen_w04": False,
                "quality_bucket": quality_bucket(row["w04_quality_score"]),
                "setup_terminal_disposition": setup_terminal,
                "final_pipeline_disposition": final_terminal,
                "confirmation_passed": confirmation_passed,
                "trade_executed": identifier in trades,
                "all_rejection_reasons": ";".join(_reasons(row)),
                "g1_weighted_contribution": contributions["G1"],
                "g2_weighted_contribution": contributions["G2"],
                "g3_weighted_contribution": contributions["G3"],
                "g4_weighted_contribution": contributions["G4"],
                "g5_weighted_contribution": contributions["G5"],
                "false_refill_weighted_penalty": contributions["FALSE_REFILL_PENALTY"],
            })
            enriched.append(row)
    enriched.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["interaction_id"])))
    return enriched


def _metric_row(
    *, group: str, metric_type: str, component: str, field: str,
    weight: Decimal | None, values: Sequence[float],
) -> dict[str, Any]:
    return {
        "group": group,
        "metric_type": metric_type,
        "component": component,
        "field": field,
        "weight": float(weight) if weight is not None else None,
        **distribution(values),
    }


def component_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "ALL_VALID_FEATURE_INTERACTIONS": list(rows),
        "REJECTED_INTERACTIONS": [row for row in rows if not bool(row["accepted"])],
        "ACCEPTED_SETUPS": [row for row in rows if bool(row["accepted"])],
        "LONG_SIDE_CANDIDATES": [row for row in rows if row["trade_direction"] == "LONG"],
        "SHORT_SIDE_CANDIDATES": [row for row in rows if row["trade_direction"] == "SHORT"],
    }
    output: list[dict[str, Any]] = []
    for group, selected in groups.items():
        for name, field, weight in G_COMPONENTS:
            output.append(_metric_row(
                group=group, metric_type="RAW_G", component=name, field=field,
                weight=weight, values=[float(row[field]) for row in selected],
            ))
            output.append(_metric_row(
                group=group, metric_type="WEIGHTED_CONTRIBUTION", component=name,
                field=f"{name.lower()}_weighted_contribution", weight=weight,
                values=[float(row[f"{name.lower()}_weighted_contribution"]) for row in selected],
            ))
        output.append(_metric_row(
            group=group, metric_type="QUALITY", component="QUALITY", field="w04_quality_score",
            weight=None, values=[float(row["w04_quality_score"]) for row in selected],
        ))
    return output


def quality_bucket_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = (
        "ACCEPTED_SCORE", "NEAR_MISS_0P40_TO_0P45",
        "MEDIUM_MISS_0P30_TO_0P40", "LOW_BELOW_0P30",
    )
    output: list[dict[str, Any]] = []
    for bucket in ordered:
        selected = [row for row in rows if row["quality_bucket"] == bucket]
        item: dict[str, Any] = {
            "bucket": bucket,
            "count": len(selected),
            "accepted_setups": sum(bool(row["accepted"]) for row in selected),
            "long_count": sum(row["trade_direction"] == "LONG" for row in selected),
            "short_count": sum(row["trade_direction"] == "SHORT" for row in selected),
            "session_date_counts": _json_counts(str(row["session_date"]) for row in selected),
        }
        for prefix, field in (("quality", "w04_quality_score"), *(
            (name.lower(), field) for name, field, _weight in G_COMPONENTS
        )):
            for stat, value in distribution(float(row[field]) for row in selected).items():
                item[f"{prefix}_{stat}"] = value
        output.append(item)
    return output


def _reason_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    categories: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    for reason in REJECTION_REASON_ORDER:
        categories.append((
            "OVERLAPPING_FROZEN_REJECTION", reason,
            [row for row in rows if reason in _reasons(row)],
        ))
        categories.append((
            "EXCLUSIVE_SETUP_DISPOSITION", reason,
            [row for row in rows if row["setup_terminal_disposition"] == reason],
        ))
    for termination in sorted({str(row["termination"]) for row in rows}):
        categories.append((
            "INTERACTION_COMPLETION_MECHANISM", termination,
            [row for row in rows if str(row["termination"]) == termination],
        ))
    for scope, reason, selected in categories:
        duration = distribution(float(row["interaction_duration_seconds"]) for row in selected)
        output.append({
            "scope": scope,
            "reason": reason,
            "count": len(selected),
            "percent_of_raw": 100.0 * len(selected) / len(rows) if rows else 0.0,
            "duration_median_seconds": duration["median"],
            "duration_p25_seconds": duration["p25"],
            "duration_p75_seconds": duration["p75"],
            "month_counts": _json_counts(str(row["month"]) for row in selected),
            "direction_counts": _json_counts(str(row["trade_direction"]) for row in selected),
        })
    return output


def _funnel_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw = len(rows)
    primitive_pass = [row for row in rows if not str(row.get("non_quality_rejection_reasons") or "")]
    score_pass = [row for row in rows if Decimal(str(row["w04_quality_score"])) >= Q]
    accepted = [row for row in rows if bool(row["accepted"])]
    confirmation_passed = [row for row in accepted if bool(row["confirmation_passed"])]
    confirmation_failed = [row for row in accepted if not bool(row["confirmation_passed"])]
    traded = [row for row in accepted if bool(row["trade_executed"])]
    stages = [
        ("POPULATION", "RAW_INTERACTION", raw),
        ("LIFECYCLE", "INTERACTION_COMPLETED", raw),
        ("FEATURE", "VALID_FIVE_G_VECTOR", sum(bool(row["valid_five_g"]) for row in rows)),
        ("DIRECTION", "DIRECTIONAL_CANDIDATE", sum(bool(row["directional_candidate"]) for row in rows)),
        ("INDEPENDENT", "PRIMITIVE_HARD_ELIGIBILITY_PASS", len(primitive_pass)),
        ("INDEPENDENT", "QUALITY_SCORE_AT_OR_ABOVE_Q", len(score_pass)),
        ("INDEPENDENT", "QUALITY_BELOW_Q", raw - len(score_pass)),
        ("SEQUENTIAL", "SETUP_ACCEPTED", len(accepted)),
        ("SEQUENTIAL", "CONFIRMATION_FAILED", len(confirmation_failed)),
        ("SEQUENTIAL", "CONFIRMATION_PASSED", len(confirmation_passed)),
        ("SEQUENTIAL", "TRADE_EXECUTED", len(traded)),
    ]
    output = [
        {"row_type": row_type, "stage_or_disposition": stage, "count": count,
         "percent_of_raw": 100.0 * count / raw if raw else 0.0}
        for row_type, stage, count in stages
    ]
    for disposition, count in sorted(Counter(str(row["setup_terminal_disposition"]) for row in rows).items()):
        output.append({
            "row_type": "MUTUALLY_EXCLUSIVE_SETUP_TERMINAL",
            "stage_or_disposition": disposition,
            "count": count,
            "percent_of_raw": 100.0 * count / raw if raw else 0.0,
        })
    for disposition, count in sorted(Counter(str(row["final_pipeline_disposition"]) for row in rows).items()):
        output.append({
            "row_type": "MUTUALLY_EXCLUSIVE_FINAL_PIPELINE_TERMINAL",
            "stage_or_disposition": disposition,
            "count": count,
            "percent_of_raw": 100.0 * count / raw if raw else 0.0,
        })
    return output


def _group_counts(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in sorted({str(row[field]) for row in rows}):
        selected = [row for row in rows if str(row[field]) == value]
        accepted = [row for row in selected if bool(row["accepted"])]
        output.append({
            "group_field": field,
            "group_value": value,
            "raw_interactions": len(selected),
            "rejected_interactions": len(selected) - len(accepted),
            "accepted_setups": len(accepted),
            "acceptance_rate": len(accepted) / len(selected) if selected else 0.0,
            "confirmation_passed": sum(bool(row["confirmation_passed"]) for row in accepted),
            "confirmation_failed": sum(not bool(row["confirmation_passed"]) for row in accepted),
            "trades": sum(bool(row["trade_executed"]) for row in accepted),
            "quality_median": distribution(float(row["w04_quality_score"]) for row in selected)["median"],
        })
    return output


def _baseline_contract(baseline_root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    summary = asia._read_json(baseline_root / "summary.json")
    setups = _read_csv(baseline_root / "setup-ledger.csv")
    trades = _read_csv(baseline_root / "trade-ledger.csv")
    expected = {
        "strategy_id": asia.STRATEGY_ID,
        "eligible_session_count": 46,
        "raw_interactions": 742,
        "accepted_setups": 9,
        "confirmations_passed": 2,
        "confirmations_failed": 7,
    }
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches or len(setups) != 9 or len(trades) != 2:
        raise AsiaW04DiagnosticError(f"published Asia baseline does not match the sealed diagnostic population: {mismatches}")
    return summary, setups, trades


def analyze_sessions(
    sessions: Sequence[Mapping[str, Any]],
    *, baseline_summary: Mapping[str, Any],
    baseline_setups: Sequence[Mapping[str, Any]],
    baseline_trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = enrich_interactions(sessions)
    accepted = [row for row in rows if bool(row["accepted"])]
    confirmation_passed = [row for row in accepted if bool(row["confirmation_passed"])]
    confirmation_failed = [row for row in accepted if not bool(row["confirmation_passed"])]
    traded = [row for row in accepted if bool(row["trade_executed"])]
    expected_ids = {str(row["interaction_id"]) for row in baseline_setups}
    actual_ids = {str(row["interaction_id"]) for row in accepted}
    expected_trades = {str(row["interaction_id"]) for row in baseline_trades}
    actual_trades = {str(row["interaction_id"]) for row in traded}
    baseline_terminals = {str(row["interaction_id"]): str(row["terminal_disposition"]) for row in baseline_setups}
    actual_terminals = {str(row["interaction_id"]): str(row["final_pipeline_disposition"]) for row in accepted}
    if expected_ids != actual_ids or expected_trades != actual_trades or baseline_terminals != actual_terminals:
        raise AsiaW04DiagnosticError("diagnostic setup/trade identities do not reproduce the published baseline")
    exact_counts = (
        len(rows), len(accepted), len(confirmation_passed), len(confirmation_failed), len(traded)
    )
    if exact_counts != (742, 9, 2, 7, 2):
        raise AsiaW04DiagnosticError(f"Asia diagnostic population mismatch: {exact_counts}")
    setup_dispositions = Counter(str(row["setup_terminal_disposition"]) for row in rows)
    final_dispositions = Counter(str(row["final_pipeline_disposition"]) for row in rows)
    if sum(setup_dispositions.values()) != 742 or sum(final_dispositions.values()) != 742:
        raise AsiaW04DiagnosticError("mutually exclusive interaction terminal reconciliation failed")
    independent_rejections = Counter(reason for row in rows for reason in _reasons(row))
    if dict(sorted(independent_rejections.items())) != dict(baseline_summary["pre_quality_rejection_counts"]):
        raise AsiaW04DiagnosticError("diagnostic rejection reasons do not reproduce baseline summary")

    g_rows = component_summary_rows(rows)
    rejected_g = {
        row["component"]: row for row in g_rows
        if row["group"] == "REJECTED_INTERACTIONS" and row["metric_type"] == "RAW_G"
    }
    accepted_g = {
        row["component"]: row for row in g_rows
        if row["group"] == "ACCEPTED_SETUPS" and row["metric_type"] == "RAW_G"
    }
    rejected_contrib = {
        row["component"]: row for row in g_rows
        if row["group"] == "REJECTED_INTERACTIONS" and row["metric_type"] == "WEIGHTED_CONTRIBUTION"
    }
    weakness = sorted(
        ({
            "component": name,
            "rejected_median": rejected_g[name]["median"],
            "accepted_median": accepted_g[name]["median"],
            "accepted_minus_rejected_median": (
                float(accepted_g[name]["median"]) - float(rejected_g[name]["median"])
            ),
        } for name, _field, _weight in G_COMPONENTS),
        key=lambda row: (-row["accepted_minus_rejected_median"], row["component"]),
    )
    contribution_ranking = sorted(
        ({"component": name, "rejected_mean_weighted_contribution": rejected_contrib[name]["mean"]}
         for name, _field, _weight in G_COMPONENTS),
        key=lambda row: (float(row["rejected_mean_weighted_contribution"]), row["component"]),
    )
    buckets = quality_bucket_rows(rows)
    near_miss_count = next(row["count"] for row in buckets if row["bucket"] == "NEAR_MISS_0P40_TO_0P45")
    low_count = next(row["count"] for row in buckets if row["bucket"] == "LOW_BELOW_0P30")
    quality_below = sum(Decimal(str(row["w04_quality_score"])) < Q for row in rows)
    lifecycle = Counter(str(row["termination"]) for row in rows)
    groups = [*_group_counts(rows, "trade_direction"), *_group_counts(rows, "month"), *_group_counts(rows, "session_date")]
    summary = {
        "status": "ASIA_W04_FUNNEL_DIAGNOSTIC_COMPLETE",
        "diagnostic_id": DIAGNOSTIC_ID,
        "strategy_id": asia.STRATEGY_ID,
        "config_id": asia.CONFIG_ID,
        "evidence_label": asia.EVIDENCE_LABEL,
        "schema_version": SCHEMA_VERSION,
        "strategy_semantics_changed": False,
        "optimization_performed": False,
        "network_calls": 0,
        "downloads": 0,
        "raw_interactions": len(rows),
        "rejected_before_feature_scoring": 0,
        "valid_five_g_interactions": sum(bool(row["valid_five_g"]) for row in rows),
        "directional_candidates": sum(bool(row["directional_candidate"]) for row in rows),
        "separate_plus_gate_present": False,
        "plus_gate_explanation": "Frozen W04 assigns direction at interaction open and accepts only when primitive predicates and Q pass; it has no separate PLUS classifier.",
        "quality_below_q": quality_below,
        "quality_at_or_above_q": len(rows) - quality_below,
        "primitive_hard_failure_count": sum(bool(str(row.get("non_quality_rejection_reasons") or "")) for row in rows),
        "accepted_setups": len(accepted),
        "confirmations_failed": len(confirmation_failed),
        "confirmations_passed": len(confirmation_passed),
        "executed_trades": len(traded),
        "independent_overlapping_rejection_counts": dict(sorted(independent_rejections.items())),
        "mutually_exclusive_setup_terminal_dispositions": dict(sorted(setup_dispositions.items())),
        "mutually_exclusive_final_pipeline_dispositions": dict(sorted(final_dispositions.items())),
        "interaction_completion_mechanisms": dict(sorted(lifecycle.items())),
        "setup_terminal_reconciliation": sum(setup_dispositions.values()) == len(rows),
        "final_pipeline_terminal_reconciliation": sum(final_dispositions.values()) == len(rows),
        "confirmation_reconciliation": len(confirmation_passed) + len(confirmation_failed) == len(accepted),
        "trade_reconciliation": len(traded) == len(baseline_trades),
        "largest_independent_rejection_reason": (
            max(independent_rejections.items(), key=lambda item: (item[1], item[0]))[0]
            if independent_rejections else None
        ),
        "attrition_classification": "QUALITY_AND_PRIMITIVE_ELIGIBILITY_NOT_LIFECYCLE",
        "descriptive_g_weakness_ranking": weakness,
        "rejected_weighted_contribution_ranking_low_to_high": contribution_ranking,
        "counterfactual_one_feature_lift": counterfactual_one_feature_lifts(rows),
        "quality_bucket_counts": {row["bucket"]: row["count"] for row in buckets},
        "most_rejected_far_below_q": low_count > near_miss_count,
        "group_counts": groups,
        "ny_comparison_status": NY_UNAVAILABLE,
        "baseline_identity_reconciliation": True,
        "baseline_summary_sha256": None,
    }
    return {
        "summary": summary,
        "interaction_rows": rows,
        "funnel_rows": _funnel_rows(rows),
        "rejection_rows": _reason_rows(rows),
        "g_rows": g_rows,
        "bucket_rows": buckets,
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    terminals = summary["mutually_exclusive_setup_terminal_dispositions"]
    weakness = summary["descriptive_g_weakness_ranking"]
    contribution = summary["rejected_weighted_contribution_ranking_low_to_high"]
    buckets = summary["quality_bucket_counts"]
    lifecycle = summary["interaction_completion_mechanisms"]
    groups = summary["group_counts"]
    direction = [row for row in groups if row["group_field"] == "trade_direction"]
    months = [row for row in groups if row["group_field"] == "month"]
    return "\n".join([
        f"# {DIAGNOSTIC_ID}", "",
        "Read-only diagnosis of the exact published Asia W04 population. No strategy rule, weight, threshold, session, confirmation, stop, target, or sizing behavior changed.", "",
        "## Exact funnel", "",
        f"- Raw completed interactions: **{summary['raw_interactions']}**",
        f"- Rejected before five-G scoring: **{summary['rejected_before_feature_scoring']}**",
        f"- Valid five-G vectors / directional candidates: **{summary['valid_five_g_interactions']} / {summary['directional_candidates']}**",
        f"- Scores below Q=0.45: **{summary['quality_below_q']}**; scores at/above Q: **{summary['quality_at_or_above_q']}**",
        f"- Accepted setups: **{summary['accepted_setups']}**",
        f"- Confirmation failed / passed: **{summary['confirmations_failed']} / {summary['confirmations_passed']}**",
        f"- Executed trades: **{summary['executed_trades']}**", "",
        "### Mutually exclusive setup dispositions", "",
        *[f"- `{reason}`: {count}" for reason, count in terminals.items()], "",
        "## Why attrition occurs", "",
        f"Largest overlapping frozen reason: `{summary['largest_independent_rejection_reason']}`.",
        "All 742 raw records are already successfully completed interaction objects with valid five-G vectors and an aggressor-derived direction. Completion mechanisms are audit labels, not rejection gates. The implementation has no separate PLUS classifier after direction assignment.",
        f"Classification: `{summary['attrition_classification']}`.", "",
        "## Descriptive G comparison", "",
        "Accepted-minus-rejected median gaps (largest first):", "",
        *[f"- {row['component']}: rejected={row['rejected_median']:.6f}, accepted={row['accepted_median']:.6f}, gap={row['accepted_minus_rejected_median']:.6f}" for row in weakness], "",
        "Rejected mean weighted contributions (lowest first):", "",
        *[f"- {row['component']}: {row['rejected_mean_weighted_contribution']:.6f}" for row in contribution], "",
        "These are descriptive diagnostics only. They do not imply a weight or threshold change.", "",
        "## Fixed quality-distance buckets", "",
        *[f"- `{bucket}`: {count}" for bucket, count in buckets.items()], "",
        "## Lifecycle completion mechanisms", "",
        *[f"- `{reason}`: {count}" for reason, count in lifecycle.items()], "",
        "## Direction and month", "",
        *[f"- {row['group_value']}: raw={row['raw_interactions']}, accepted={row['accepted_setups']}, confirmation pass={row['confirmation_passed']}" for row in direction], "",
        *[f"- {row['group_value']}: raw={row['raw_interactions']}, accepted={row['accepted_setups']}, confirmation pass={row['confirmation_passed']}" for row in months], "",
        "## NY comparison", "",
        f"`{summary['ny_comparison_status']}`", "",
        "The protected NY research directory has aggregate matrix artifacts but no directly comparable frozen per-interaction W04 feature ledger. No NY replay was run.", "",
        "## Interpretation limits", "",
        "No recommendation, strategy change, parameter selection, optimization, data acquisition, Databento call, or PnL experiment was performed.", "",
    ])


def _report_html(summary: Mapping[str, Any]) -> str:
    terminals = "".join(
        f"<tr><td><code>{html.escape(reason)}</code></td><td>{count}</td></tr>"
        for reason, count in summary["mutually_exclusive_setup_terminal_dispositions"].items()
    )
    weakness = "".join(
        f"<tr><td>{row['component']}</td><td>{row['rejected_median']:.6f}</td><td>{row['accepted_median']:.6f}</td><td>{row['accepted_minus_rejected_median']:.6f}</td></tr>"
        for row in summary["descriptive_g_weakness_ranking"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{DIAGNOSTIC_ID}</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem;color:#17202a}}.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem}}.card{{border:1px solid #ccd6df;border-radius:8px;padding:1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd6df;padding:.45rem;text-align:left}}code{{background:#eef1f5;padding:.1rem .25rem}}.warn{{font-weight:700;color:#8a2b06}}</style></head><body>
<h1>{DIAGNOSTIC_ID}</h1><p>Deterministic read-only diagnostic of the exact frozen Asia W04 population.</p>
<div class='grid'><div class='card'><b>Raw</b><br>{summary['raw_interactions']}</div><div class='card'><b>Accepted</b><br>{summary['accepted_setups']}</div><div class='card'><b>Confirm pass</b><br>{summary['confirmations_passed']}</div><div class='card'><b>Confirm fail</b><br>{summary['confirmations_failed']}</div><div class='card'><b>Trades</b><br>{summary['executed_trades']}</div></div>
<h2>Exclusive setup dispositions</h2><table><tr><th>Disposition</th><th>Count</th></tr>{terminals}</table>
<h2>Descriptive G weakness</h2><table><tr><th>G</th><th>Rejected median</th><th>Accepted median</th><th>Gap</th></tr>{weakness}</table>
<p>Largest overlapping rejection reason: <code>{html.escape(str(summary['largest_independent_rejection_reason']))}</code>.</p>
<p>NY: <code>{NY_UNAVAILABLE}</code>.</p><p class='warn'>No strategy rule was selected or changed. No optimization, network call, download, or PnL experiment occurred.</p>
</body></html>"""


def _write_artifacts(staging: Path, analysis: Mapping[str, Any]) -> None:
    summary = dict(analysis["summary"])
    asia._write_json(staging / "summary.json", summary)
    asia._write_csv(staging / "interaction-funnel.csv", analysis["funnel_rows"])
    asia._write_csv(staging / "interaction-features.csv", analysis["interaction_rows"])
    asia._write_csv(staging / "rejection-reasons.csv", analysis["rejection_rows"])
    asia._write_csv(staging / "g-component-summary.csv", analysis["g_rows"])
    asia._write_csv(staging / "quality-buckets.csv", analysis["bucket_rows"])
    (staging / "diagnostic-report.md").write_text(_report_markdown(summary), encoding="utf-8")
    (staging / "diagnostic-report.html").write_text(_report_html(summary), encoding="utf-8")


def run_diagnostic(
    *, repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    baseline_root: Path = BASELINE_ROOT,
    audit_root: Path = asia.AUDIT_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_root = (output_root if output_root.is_absolute() else repository_root / output_root).resolve()
    baseline_root = (baseline_root if baseline_root.is_absolute() else repository_root / baseline_root).resolve()
    audit_root = (audit_root if audit_root.is_absolute() else repository_root / audit_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable Asia diagnostic output already exists: {output_root}")
    baseline_summary, baseline_setups, baseline_trades = _baseline_contract(baseline_root)
    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    contract_path = staging / "diagnostic-contract.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": asia.STRATEGY_ID,
        "config_id": asia.CONFIG_ID,
        "weights": {key: str(value) for key, value in asia.W04_WEIGHTS.items()},
        "quality_threshold": str(asia.QUALITY_THRESHOLD),
        "baseline_summary_sha256": asia._sha256(baseline_root / "summary.json"),
        "network_calls": 0,
        "downloads": 0,
    }
    if contract_path.is_file() and asia._read_json(contract_path) != contract:
        raise AsiaW04DiagnosticError("incompatible existing Asia diagnostic checkpoint contract")
    asia._write_json(contract_path, contract)

    sessions = asia.load_audit_sessions(audit_root)
    bindings = asia.source_bindings(repository_root, sessions)
    source_verification = asia.verify_source_bindings(bindings)
    sessions_by_day = {row.day: row for row in sessions}
    results: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for binding in bindings:
        binding_sessions = [sessions_by_day[day] for day in binding.days]
        if binding.shared:
            batch = asia._process_shared_binding(
                binding, binding_sessions, staging=staging, previous_result=previous,
            )
            results.extend(batch)
            previous = batch[-1]
        else:
            session = binding_sessions[0]
            print(
                f"ASIA_DIAGNOSTIC_SESSION {len(results) + 1:02d}/{asia.EXPECTED_SOURCE_SESSIONS:02d} "
                f"{session.day} eligible={str(session.eligible).lower()}",
                flush=True,
            )
            current = asia._process_daily_binding(
                binding, session, staging=staging, previous_result=previous,
            )
            results.append(current)
            previous = current
    results.sort(key=lambda row: str(row["session_date"]))
    if [str(row["session_date"]) for row in results] != [row.day for row in sessions]:
        raise AsiaW04DiagnosticError("diagnostic results do not match the audited source chronology")
    analysis = analyze_sessions(
        results,
        baseline_summary=baseline_summary,
        baseline_setups=baseline_setups,
        baseline_trades=baseline_trades,
    )
    analysis["summary"]["baseline_summary_sha256"] = contract["baseline_summary_sha256"]
    analysis["summary"]["source_verification"] = source_verification
    analysis["summary"]["eligible_session_count"] = sum(bool(row["eligible"]) for row in results)
    analysis["summary"]["source_session_count"] = len(results)
    _write_artifacts(staging, analysis)
    work = staging / "_work"
    if work.exists():
        shutil.rmtree(work)
    checkpoints = staging / "_checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    run_manifest = {
        "status": analysis["summary"]["status"],
        "diagnostic_id": DIAGNOSTIC_ID,
        "strategy_id": asia.STRATEGY_ID,
        "artifact_hashes": {
            path.name: asia._sha256(path)
            for path in sorted(staging.iterdir()) if path.is_file()
        },
        "network_calls": 0,
        "downloads": 0,
    }
    asia._write_json(staging / "run-manifest.json", run_manifest)
    os.rename(staging, output_root)
    return {**analysis["summary"], "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--audit-root", type=Path, default=asia.AUDIT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_diagnostic(
            repository_root=args.repository_root,
            output_root=args.output_root,
            baseline_root=args.baseline_root,
            audit_root=args.audit_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps({
        "status": result["status"],
        "raw_interactions": result["raw_interactions"],
        "accepted_setups": result["accepted_setups"],
        "confirmations_passed": result["confirmations_passed"],
        "executed_trades": result["executed_trades"],
        "output_root": result["output_root"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
