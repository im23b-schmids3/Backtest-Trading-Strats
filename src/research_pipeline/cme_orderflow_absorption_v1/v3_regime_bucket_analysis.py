"""CSV-only monotonicity diagnostic for the compact V3 context population."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import v3_long_short_regime_diagnostic as context


SOURCE = context.DEFAULT_OUTPUT / "setup-context.csv"
DEFAULT_OUTPUT = context.artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V3_DIAGNOSTIC/regime_bucket_analysis"
FEATURES = (
    "direction_normalized_price_move_5m_ticks",
    "direction_normalized_price_move_15m_ticks",
    "direction_normalized_price_move_30m_ticks",
    "recent_range_5m_ticks",
    "execution_count_15s",
    "executed_volume_15s",
)
MOMENTUM_FEATURES = set(FEATURES[:3])
POPULATIONS = (
    ("ALL", {}),
    ("AUGUST_SEEN", {"period": "AUGUST_SEEN"}),
    ("RETRO_JUNE_JULY", {"period": "RETRO_JUNE_JULY"}),
    ("RETRO_JUNE_JULY_BUYER_ABSORPTION", {"period": "RETRO_JUNE_JULY", "direction": "BUYER_ABSORPTION"}),
    ("RETRO_JUNE_JULY_SELLER_ABSORPTION", {"period": "RETRO_JUNE_JULY", "direction": "SELLER_ABSORPTION"}),
    ("AUGUST_SEEN_BUYER_ABSORPTION", {"period": "AUGUST_SEEN", "direction": "BUYER_ABSORPTION"}),
    ("AUGUST_SEEN_SELLER_ABSORPTION", {"period": "AUGUST_SEEN", "direction": "SELLER_ABSORPTION"}),
)


class BucketAnalysisError(RuntimeError):
    pass


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "p25": _percentile(values, 0.25), "p75": _percentile(values, 0.75), "min": min(values), "max": max(values)}


def select(rows: list[dict[str, str]], **where: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == value for key, value in where.items())]


def assign_quartiles(rows: list[dict[str, str]], feature: str) -> tuple[dict[str, list[dict[str, str]]], int]:
    """Return balanced value-ranked quartiles; immutable source order resolves ties."""
    usable = [(value, index, row) for index, row in enumerate(rows) if (value := _number(row.get(feature))) is not None]
    ordered = sorted(usable, key=lambda item: (item[0], item[1]))
    buckets = {f"Q{index}": [] for index in range(1, 5)}
    for index, (_, _, row) in enumerate(ordered):
        buckets[f"Q{min(4, (index * 4) // len(ordered) + 1)}"].append(row)
    return buckets, len(rows) - len(ordered)


def _traded_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("trade_outcome") in {"WIN", "LOSS"} and _number(row.get("r_multiple")) is not None]


def leave_one_out(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    trades = [_number(row["r_multiple"]) for row in _traded_rows(rows)]
    values = [value for value in trades if value is not None]
    total = sum(values)
    if len(values) < 3:
        return {"traded_rows": len(values), "original_total_r": total if values else None, "best_case_remove_worst_r": None, "worst_case_remove_best_r": None}
    return {"traded_rows": len(values), "original_total_r": total, "best_case_remove_worst_r": total - min(values), "worst_case_remove_best_r": total - max(values)}


def bucket_metrics(rows: list[dict[str, str]], feature: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _number(row.get(feature))) is not None]
    trades = _traded_rows(rows)
    r_values = [_number(row["r_multiple"]) for row in trades]
    r_summary = _summary([value for value in r_values if value is not None])
    absorption = _summary([value for row in rows if (value := _number(row.get("absorption_score"))) is not None])
    replenishment = _summary([value for row in rows if (value := _number(row.get("replenishment_score"))) is not None])
    wins = sum(row["trade_outcome"] == "WIN" for row in trades)
    return {"feature_min": min(values) if values else None, "feature_max": max(values) if values else None, "total_setups": len(rows), "traded_setups": len(trades), "wins": wins, "losses": len(trades) - wins, "win_rate": wins / len(trades) if trades else None, "average_r": r_summary["mean"], "median_r": r_summary["median"], "total_r": sum(value for value in r_values if value is not None) if r_values else None, "mean_absorption_score": absorption["mean"], "median_absorption_score": absorption["median"], "mean_replenishment_score": replenishment["mean"], "median_replenishment_score": replenishment["median"], "leave_one_out": leave_one_out(rows)}


def _span(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return max(known) - min(known) if known else None


def classify_monotonicity(feature: str, buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    means = [buckets[f"Q{index}"]["average_r"] for index in range(1, 5)]
    traded_counts = [buckets[f"Q{index}"]["traded_setups"] for index in range(1, 5)]
    observed = [value for value in means if value is not None]
    # A bucket relationship cannot be called broad when even one quartile is
    # carried by a single trade. This is a diagnostic reporting safeguard, not
    # a strategy threshold or a selection criterion.
    if len(observed) < 2 or any(count < 2 for count in traded_counts):
        label = "INSUFFICIENT_SAMPLE"
    else:
        expected_increasing = feature in MOMENTUM_FEATURES  # Q1 is most adverse momentum.
        signs = [right - left for left, right in zip(means, means[1:]) if left is not None and right is not None]
        correct = [value >= 0 if expected_increasing else value <= 0 for value in signs]
        endpoint_correct = means[-1] >= means[0] if expected_increasing else means[-1] <= means[0]
        label = "CLEAR_MONOTONIC_PATTERN" if len(observed) == 4 and signs and all(correct) else "PARTIAL_MONOTONIC_PATTERN" if endpoint_correct and correct and sum(correct) > len(correct) / 2 else "NO_MONOTONIC_PATTERN"
    return {"label": label, "feature_order": "Q1 most adverse to Q4 least adverse" if feature in MOMENTUM_FEATURES else "Q1 lowest to Q4 highest range/activity", "average_r_by_bucket": means, "traded_setups_by_bucket": traded_counts, "outcome_span": max(observed) - min(observed) if observed else None, "absorption_median_span": _span([buckets[f"Q{index}"]["median_absorption_score"] for index in range(1, 5)]), "replenishment_median_span": _span([buckets[f"Q{index}"]["median_replenishment_score"] for index in range(1, 5)])}


def _report(payload: dict[str, Any]) -> str:
    def section(population: str) -> list[str]:
        return [f"- `{feature}`: `{result['label']}`; average R Q1→Q4 = {result['average_r_by_bucket']}; absorption/replenishment median spans = {result['absorption_median_span']} / {result['replenishment_median_span']}." for feature, result in payload["monotonicity"][population].items()]
    return "\n".join(["# Compact regime monotonicity / bucket diagnostic", "", "Only `setup-context.csv` was read. No DBN scan, interaction reconstruction, PLUS rescoring, trade replay, PnL optimization, threshold selection, or strategy-rule change occurred.", "", "## Method", "", "Each predeclared feature is ranked within each population and split into four approximately equal-count buckets. Equal feature values are resolved only by immutable source-row order to keep counts balanced; reported min/max boundaries make tied-value buckets visible. WIN/LOSS rows alone contribute outcome metrics; non-traded setups remain counted.", "", "## RETRO BUYER assessment", "", *section("RETRO_JUNE_JULY_BUYER_ABSORPTION"), "", "## AUGUST BUYER assessment", "", *section("AUGUST_SEEN_BUYER_ABSORPTION"), "", "## Interpretation", "", "These are exploratory descriptive bucket patterns. No numerical cutoff, filter, target, confirmation method, or trading rule is selected. See leave-one-out totals in the machine-readable results."]) + "\n"


def materialize(*, source: Path = SOURCE, output_dir: Path = DEFAULT_OUTPUT, replace_existing: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not replace_existing:
        raise BucketAnalysisError(f"immutable output already exists: {output_dir}")
    if not source.is_file():
        raise BucketAnalysisError(f"missing compact context source: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    original = [dict(row) for row in rows]
    required = {"period", "direction", "trade_outcome", "r_multiple", "absorption_score", "replenishment_score", *FEATURES}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise BucketAnalysisError(f"compact context schema missing: {','.join(sorted(missing))}")
    result_rows: list[dict[str, Any]] = []
    monotonicity: dict[str, dict[str, Any]] = {}
    population_counts: dict[str, int] = {}
    for name, where in POPULATIONS:
        population = select(rows, **where)
        population_counts[name] = len(population)
        monotonicity[name] = {}
        for feature in FEATURES:
            assigned, missing_value_count = assign_quartiles(population, feature)
            metrics = {bucket: bucket_metrics(bucket_rows, feature) for bucket, bucket_rows in assigned.items()}
            monotonicity[name][feature] = classify_monotonicity(feature, metrics)
            for bucket, metric in metrics.items():
                result_rows.append({"population": name, "feature": feature, "bucket": bucket, "population_setups": len(population), "missing_feature_values": missing_value_count, **metric})
    payload = {"diagnostic_type": "CSV_ONLY_REGIME_MONOTONICITY_BUCKET_ANALYSIS", "source": str(source), "source_row_count": len(rows), "source_population_unchanged": rows == original, "primary_population": "TICK_3_TARGET_2_5R", "features": list(FEATURES), "population_counts": population_counts, "tie_handling": "feature value, then immutable source-row order; no numeric threshold selected", "strategy_semantics_changed": False, "pnl_optimization_performed": False, "selection_prohibited": True, "bucket_results": result_rows, "monotonicity": monotonicity}
    output_dir.mkdir(parents=True, exist_ok=replace_existing)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "monotonicity.json").write_text(json.dumps(monotonicity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["population", "feature", "bucket", "population_setups", "missing_feature_values", "feature_min", "feature_max", "total_setups", "traded_setups", "wins", "losses", "win_rate", "average_r", "median_r", "total_r", "mean_absorption_score", "median_absorption_score", "mean_replenishment_score", "median_replenishment_score", "leave_one_out"]
    with (output_dir / "bucket-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in result_rows:
            data = dict(row); data["leave_one_out"] = json.dumps(data["leave_one_out"], sort_keys=True, separators=(",", ":")); writer.writerow(data)
    (output_dir / "diagnostic-report.md").write_text(_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV-only V3 regime bucket analysis")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-existing", action="store_true", help="replace an existing diagnostic report directory")
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(source=args.source, output_dir=args.output_dir, replace_existing=args.replace_existing), indent=2, sort_keys=True))
    except BucketAnalysisError as exc:
        print(f"ERROR: {exc}"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
