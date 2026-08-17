"""Read-only rejection-funnel diagnostic for published frozen L2 V1 artifacts.

This module intentionally knows only CSV/JSON artifact schemas.  It imports no
Databento code and cannot open a DBN, construct a book, score an interaction, or
alter an L2 result.  It explains the qualification decisions already recorded
by the completed May 2026 replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .model import L2Config


ARTIFACT_NAME = "CMEOrderflowAbsorption.ES_L2_V1_MAY_2026"
OUTPUT_NAME = "rejection_funnel"


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _quantile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = list(values)
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "count": len(numbers), "mean": mean(numbers), "median": median(numbers),
        "p25": _quantile(numbers, .25), "p75": _quantile(numbers, .75),
        "p90": _quantile(numbers, .90), "p95": _quantile(numbers, .95),
        "p99": _quantile(numbers, .99), "min": min(numbers), "max": max(numbers),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gate_status(row: dict[str, Any], config: L2Config = L2Config()) -> dict[str, bool]:
    """Re-evaluate only the frozen hard qualification predicates from a ledger row."""
    volume = _float(row.get("directional_aggressive_volume")) >= config.min_relevant_aggressive_volume
    executions = _float(row.get("relevant_execution_count")) >= config.min_relevant_execution_count
    restoration = _float(row.get("consume_restore_cycles")) >= config.min_consume_restore_cycles
    rejection = (
        _float(row.get("maximum_through_level_progress_ticks")) <= config.max_through_level_progress_ticks
        or _float(row.get("interaction_rejection_ticks")) >= config.min_rejection_ticks
    )
    quality = _float(row.get("l2_absorption_quality_score")) >= config.min_quality_score
    return {
        "relevant_aggressive_volume": volume,
        "relevant_execution_count": executions,
        "consume_restore": restoration,
        "rejection": rejection,
        # Qualification has no further evidence-family hard gate.  Retaining
        # this explicit pass-through stage makes the requested funnel auditable.
        "other_evidence_family": True,
        "quality": quality,
    }


def _failed_gates(status: dict[str, bool]) -> list[str]:
    return [name for name in ("relevant_aggressive_volume", "relevant_execution_count", "consume_restore", "rejection", "quality")
            if not status[name]]


def _reason_combo(failed: list[str]) -> str:
    labels = {
        "relevant_aggressive_volume": "AGGRESSIVE_VOLUME",
        "relevant_execution_count": "EXECUTION_COUNT",
        "consume_restore": "RESTORATION",
        "rejection": "REJECTION",
        "quality": "QUALITY",
    }
    return " + ".join(labels[name] for name in failed) if failed else "ACCEPTED"


def _quality_without_penalty(row: dict[str, Any], config: L2Config) -> float:
    raw = (
        _float(row.get("aggression_score")) * config.aggression_weight
        + _float(row.get("restoration_score")) * config.restoration_weight
        + _float(row.get("price_resistance_score")) * config.price_resistance_weight
        + _float(row.get("persistence_score")) * config.persistence_weight
        + _float(row.get("multi_level_support_score")) * config.multi_level_support_weight
    )
    return min(1.0, max(0.0, raw))


def _near_miss(row: dict[str, Any], gate: str, config: L2Config) -> dict[str, Any]:
    values: dict[str, tuple[float, float | str, float]] = {
        "relevant_aggressive_volume": (
            _float(row.get("directional_aggressive_volume")), config.min_relevant_aggressive_volume,
            config.min_relevant_aggressive_volume - _float(row.get("directional_aggressive_volume")),
        ),
        "relevant_execution_count": (
            _float(row.get("relevant_execution_count")), config.min_relevant_execution_count,
            config.min_relevant_execution_count - _float(row.get("relevant_execution_count")),
        ),
        "consume_restore": (
            _float(row.get("consume_restore_cycles")), config.min_consume_restore_cycles,
            config.min_consume_restore_cycles - _float(row.get("consume_restore_cycles")),
        ),
        "rejection": (
            _float(row.get("interaction_rejection_ticks")),
            f">={config.min_rejection_ticks} when progress>{config.max_through_level_progress_ticks}",
            config.min_rejection_ticks - _float(row.get("interaction_rejection_ticks")),
        ),
        "quality": (
            _float(row.get("l2_absorption_quality_score")), config.min_quality_score,
            config.min_quality_score - _float(row.get("l2_absorption_quality_score")),
        ),
    }
    value, threshold, distance = values[gate]
    return {"interaction_id": row["interaction_id"], "date": row.get("date"), "gate": gate,
            "value": value, "threshold": threshold, "distance_from_threshold": distance,
            "maximum_through_level_progress_ticks": _float(row.get("maximum_through_level_progress_ticks")),
            "rejection_reasons": row.get("rejection_reasons", "")}


def analyze_rows(
    interactions: list[dict[str, Any]], setups: list[dict[str, Any]], trades: list[dict[str, Any]],
    config: L2Config = L2Config(),
) -> dict[str, Any]:
    """Return a deterministic, ledger-only explanation of qualification outcomes."""
    setup_by_id = {row["interaction_id"]: row for row in setups}
    if len(setup_by_id) != len(setups) or len({row["interaction_id"] for row in interactions}) != len(interactions):
        raise ValueError("published setup/interactions ledgers contain duplicate interaction_id values")
    if set(setup_by_id) != {row["interaction_id"] for row in interactions}:
        raise ValueError("published setup and interaction ledgers do not reconcile")

    enriched: list[dict[str, Any]] = []
    for row in interactions:
        status = gate_status(row, config)
        expected_accepted = all(status.values())
        ledger = setup_by_id[row["interaction_id"]]
        if expected_accepted != _bool(ledger.get("accepted")):
            raise ValueError(f"frozen qualification ledger mismatch for {row['interaction_id']}")
        enriched.append({**row, **ledger, "_gates": status, "_failed": _failed_gates(status)})

    total = len(enriched)
    sequential: list[dict[str, Any]] = []
    survivors = enriched
    stage_names = [
        ("completed_interaction", None),
        ("relevant_aggressive_volume_passes", "relevant_aggressive_volume"),
        ("relevant_execution_count_passes", "relevant_execution_count"),
        ("consume_restore_requirement_passes", "consume_restore"),
        ("rejection_requirement_passes", "rejection"),
        ("all_other_evidence_family_requirements_pass", "other_evidence_family"),
        ("quality_score_passes", "quality"),
    ]
    for stage, gate in stage_names:
        if gate is not None:
            survivors = [row for row in survivors if row["_gates"][gate]]
        sequential.append({"stage": stage, "count": len(survivors), "percent_of_completed": 100.0 * len(survivors) / total if total else 0.0})
    accepted = [row for row in enriched if not row["_failed"]]
    confirmation_passed = [row for row in accepted if row.get("confirmation_status") == "ENTRY"]
    traded_ids = {row["interaction_id"] for row in trades}
    if not traded_ids.issubset({row["interaction_id"] for row in confirmation_passed}):
        raise ValueError("trade ledger references a setup without passed confirmation")
    sequential.extend([
        {"stage": "final_l2_setup_accepted", "count": len(accepted), "percent_of_completed": 100.0 * len(accepted) / total if total else 0.0},
        {"stage": "confirmation_passes", "count": len(confirmation_passed), "percent_of_completed": 100.0 * len(confirmation_passed) / total if total else 0.0},
        {"stage": "actual_trade_occurs", "count": len(traded_ids), "percent_of_completed": 100.0 * len(traded_ids) / total if total else 0.0},
    ])

    rejected = [row for row in enriched if row["_failed"]]
    independent = {
        gate: sum(not row["_gates"][gate] for row in enriched)
        for gate in ("relevant_aggressive_volume", "relevant_execution_count", "consume_restore", "rejection", "quality")
    }
    exact_reasons = Counter(
        reason for row in rejected for reason in str(row.get("rejection_reasons", "")).split(";") if reason
    )
    combinations = Counter(_reason_combo(row["_failed"]) for row in rejected)
    near_rows = [_near_miss(row, row["_failed"][0], config) for row in rejected if len(row["_failed"]) == 1]
    near_summary = {
        gate: {"count": len(items), "value_distribution": _distribution([_float(item["value"]) for item in items]),
               "distance_distribution": _distribution([_float(item["distance_from_threshold"]) for item in items])}
        for gate in ("relevant_aggressive_volume", "relevant_execution_count", "consume_restore", "rejection", "quality")
        for items in [[item for item in near_rows if item["gate"] == gate]]
    }
    quality_values = [_float(row.get("l2_absorption_quality_score")) for row in enriched]
    restoration_values = {
        "depth_restoration_count": [_float(row.get("depth_restoration_count")) for row in enriched],
        "consume_restore_cycles": [_float(row.get("consume_restore_cycles")) for row in enriched],
        "cumulative_restored_volume": [_float(row.get("cumulative_restored_volume")) for row in enriched],
        "restoration_to_consumption_ratio": [_float(row.get("restoration_to_consumption_ratio")) for row in enriched],
        "mean_restoration_latency_ms": [_float(row.get("mean_restoration_latency_ms")) for row in enriched],
    }
    positive_evidence = [row for row in enriched if all(row["_gates"][key] for key in (
        "relevant_aggressive_volume", "relevant_execution_count", "consume_restore", "rejection"))]
    penalty_suppressed = [row for row in positive_evidence if not row["_gates"]["quality"] and _quality_without_penalty(row, config) >= config.min_quality_score]
    failed_confirmations = [row for row in accepted if row.get("confirmation_status") == "FAILED"]

    return {
        "diagnostic_scope": "READ_ONLY_PUBLISHED_L2_V1_ARTIFACTS_ONLY",
        "strategy_or_parameter_changes": False,
        "total_interactions": total,
        "frozen_config": {
            "min_relevant_aggressive_volume": config.min_relevant_aggressive_volume,
            "min_relevant_execution_count": config.min_relevant_execution_count,
            "min_consume_restore_cycles": config.min_consume_restore_cycles,
            "min_rejection_ticks": config.min_rejection_ticks,
            "max_through_level_progress_ticks": config.max_through_level_progress_ticks,
            "min_quality_score": config.min_quality_score,
        },
        "funnel": sequential,
        "independent_gate_failures": independent,
        "exact_rejection_reasons": dict(sorted(exact_reasons.items())),
        "top_rejection_combinations": [{"combination": name, "count": count,
                                        "percent_of_all": 100.0 * count / total if total else 0.0,
                                        "percent_of_rejected": 100.0 * count / len(rejected) if rejected else 0.0}
                                       for name, count in combinations.most_common(20)],
        "one_gate_near_misses": {"count": len(near_rows), "by_gate": near_summary},
        "quality_score": {
            "counts_at_or_above": {str(value): sum(score >= value for score in quality_values)
                                     for value in (.20, .30, .40, .45, .50, .55, .60)},
            "distribution": _distribution(quality_values),
        },
        "restoration": {
            "cycle_counts": {"0": sum(value == 0 for value in restoration_values["consume_restore_cycles"]),
                             ">=1": sum(value >= 1 for value in restoration_values["consume_restore_cycles"]),
                             ">=2": sum(value >= 2 for value in restoration_values["consume_restore_cycles"]),
                             ">=3": sum(value >= 3 for value in restoration_values["consume_restore_cycles"])},
            "distributions": {name: _distribution(values) for name, values in restoration_values.items()},
            "fails_only_no_cycle": sum(row["_failed"] == ["consume_restore"] for row in rejected),
        },
        "price_resistance": {
            "rejection_tick_counts": {"=0": sum(_float(row.get("interaction_rejection_ticks")) == 0 for row in enriched),
                                      ">0": sum(_float(row.get("interaction_rejection_ticks")) > 0 for row in enriched),
                                      ">=0.25": sum(_float(row.get("interaction_rejection_ticks")) >= .25 for row in enriched),
                                      ">=1": sum(_float(row.get("interaction_rejection_ticks")) >= 1 for row in enriched),
                                      ">=2": sum(_float(row.get("interaction_rejection_ticks")) >= 2 for row in enriched),
                                      ">=3": sum(_float(row.get("interaction_rejection_ticks")) >= 3 for row in enriched)},
            "maximum_through_level_progress_ticks": _distribution([_float(row.get("maximum_through_level_progress_ticks")) for row in enriched]),
            "price_resistance_score": _distribution([_float(row.get("price_resistance_score")) for row in enriched]),
            "fails_only_rejection": sum(row["_failed"] == ["rejection"] for row in rejected),
        },
        "persistence": {
            "defended_price_present_fraction": _distribution([_float(row.get("defended_price_present_fraction")) for row in enriched]),
            "persistence_score": _distribution([_float(row.get("persistence_score")) for row in enriched]),
            "zero_score_count": sum(_float(row.get("persistence_score")) == 0 for row in enriched),
            "zero_present_fraction_count": sum(_float(row.get("defended_price_present_fraction")) == 0 for row in enriched),
            "zero_depth_mean_count": sum(_float(row.get("defended_depth_time_weighted_mean")) == 0 for row in enriched),
            "duration_ms_when_score_zero": _distribution([(_float(row.get("interaction_end_ns")) - _float(row.get("interaction_start_ns"))) / 1_000_000 for row in enriched if _float(row.get("persistence_score")) == 0]),
        },
        "false_refill_penalty": {
            "false_refill_penalty": _distribution([_float(row.get("false_refill_penalty")) for row in enriched]),
            "rapid_cancel_ratio": _distribution([_float(row.get("rapid_cancel_ratio")) for row in enriched]),
            "unexecuted_add_volume": _distribution([_float(row.get("unexecuted_add_volume")) for row in enriched]),
            "positive_evidence_but_penalty_suppressed": len(penalty_suppressed),
        },
        "accepted_setups": [{
            "date": row.get("date"), "interaction_id": row["interaction_id"], "direction": row.get("direction"),
            "structural_level": row.get("level"), "quality_score": _float(row.get("l2_absorption_quality_score")),
            "aggression_score": _float(row.get("aggression_score")), "restoration_score": _float(row.get("restoration_score")),
            "price_resistance_score": _float(row.get("price_resistance_score")), "persistence_score": _float(row.get("persistence_score")),
            "multi_level_support_score": _float(row.get("multi_level_support_score")), "false_refill_penalty": _float(row.get("false_refill_penalty")),
            "consume_restore_cycles": _float(row.get("consume_restore_cycles")), "rejection_ticks": _float(row.get("interaction_rejection_ticks")),
            "confirmation_status": row.get("confirmation_status"), "terminal_reason": row.get("terminal_reason"),
        } for row in accepted],
        "confirmation": {
            "accepted": len(accepted), "passed": len(confirmation_passed), "failed": len(failed_confirmations), "trades": len(traded_ids),
            "failed_price_path_mechanics": [{
                "interaction_id": row["interaction_id"], "status": "UNAVAILABLE_FROM_PUBLISHED_ARTIFACTS",
                "reason": "setup-ledger records only terminal confirmation status; it has no +5s..+15s execution-path observations",
            } for row in failed_confirmations],
        },
        "near_miss_rows": near_rows,
    }


def _markdown(summary: dict[str, Any]) -> str:
    funnel = summary["funnel"]
    independent = summary["independent_gate_failures"]
    top = summary["top_rejection_combinations"]
    return "\n".join([
        "# Frozen L2 V1 May 2026 rejection-funnel diagnostic", "",
        "Read-only analysis of published CSV/JSON artifacts. No DBN was opened, no replay was run, and no strategy rule was changed.", "",
        "## Funnel", "", "| Stage | Count | % of completed |", "|---|---:|---:|",
        *[f"| {row['stage']} | {row['count']} | {row['percent_of_completed']:.2f}% |" for row in funnel], "",
        "## Independent gate failures", "",
        *[f"- {name}: {count}" for name, count in independent.items()], "",
        "## Top multi-failure combinations", "",
        *[f"- {row['combination']}: {row['count']} ({row['percent_of_rejected']:.2f}% of rejected)" for row in top], "",
        "## Confirmation bottleneck", "",
        f"- Accepted: {summary['confirmation']['accepted']}; passed: {summary['confirmation']['passed']}; failed: {summary['confirmation']['failed']}; trades: {summary['confirmation']['trades']}.",
        "- The published artifacts do not preserve the execution-path observations required for +5s..+15s maxima; this diagnostic intentionally reports those values as unavailable rather than opening a DBN.", "",
        "## Interpretation limits", "",
        "All distributions are descriptive. This diagnostic selects no threshold, rule, or candidate and performs no PnL optimization.", "",
    ])


def materialize(artifact_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Read the published ledgers and write a new diagnostic subdirectory once."""
    output = output_dir or artifact_root / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    interactions = _read_csv(artifact_root / "interaction-features.csv")
    setups = _read_csv(artifact_root / "setup-ledger.csv")
    trades = _read_csv(artifact_root / "trade-ledger.csv")
    if not (artifact_root / "summary.json").is_file():
        raise FileNotFoundError("published summary.json is required")
    summary = analyze_rows(interactions, setups, trades)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps({key: value for key, value in summary.items() if key != "near_miss_rows"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "funnel.csv", summary["funnel"], ["stage", "count", "percent_of_completed"])
    rejection_rows = [
        {"row_type": "exact_reason", "reason_or_combination": key, "count": value,
         "percent_of_all": 100.0 * value / summary["total_interactions"],
         "percent_of_rejected": 100.0 * value / (summary["total_interactions"] - len(summary["accepted_setups"]))}
        for key, value in summary["exact_rejection_reasons"].items()
    ] + [
        {"row_type": "combination", "reason_or_combination": row["combination"], "count": row["count"],
         "percent_of_all": row["percent_of_all"], "percent_of_rejected": row["percent_of_rejected"]}
        for row in summary["top_rejection_combinations"]
    ]
    _write_csv(output / "rejection-reasons.csv", rejection_rows, ["row_type", "reason_or_combination", "count", "percent_of_all", "percent_of_rejected"])
    _write_csv(output / "near-misses.csv", summary["near_miss_rows"], ["interaction_id", "date", "gate", "value", "threshold", "distance_from_threshold", "maximum_through_level_progress_ticks", "rejection_reasons"])
    (output / "diagnostic-report.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only frozen L2 V1 published-artifact rejection diagnostic")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = materialize(args.artifact_root, args.output_dir)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"total_interactions": result["total_interactions"], "accepted_setups": len(result["accepted_setups"]),
                      "output_dir": str(args.output_dir or args.artifact_root / OUTPUT_NAME)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
