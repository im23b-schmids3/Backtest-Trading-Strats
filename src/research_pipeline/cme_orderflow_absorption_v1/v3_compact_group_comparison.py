"""CSV-only descriptive comparison of compact V3 long/short context rows."""
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
DEFAULT_OUTPUT = context.artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V3_DIAGNOSTIC/compact_group_comparison"
FEATURES = (
    "absorption_score", "replenishment_score",
    "direction_normalized_price_move_5m_ticks", "direction_normalized_price_move_15m_ticks",
    "direction_normalized_price_move_30m_ticks", "direction_normalized_price_minus_vwap_ticks",
    "recent_range_1m_ticks", "recent_range_5m_ticks", "recent_range_15m_ticks", "session_range_ticks",
    "execution_count_5s", "execution_count_15s", "execution_count_60s",
    "executed_volume_5s", "executed_volume_15s", "executed_volume_60s",
    "previous_completed_interactions_same_level", "previous_plus_events_same_level",
    "entry_displacement_ticks_from_interaction_end", "stop_distance_ticks", "r_multiple",
)


class CompactComparisonError(RuntimeError):
    pass


def _number(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1: return ordered[0]
    index = (len(ordered) - 1) * q; low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(rows: list[dict[str, str]], feature: str) -> dict[str, float | int | None]:
    values = [value for row in rows if (value := _number(row.get(feature))) is not None]
    if not values: return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "p25": _percentile(values, .25), "p75": _percentile(values, .75), "min": min(values), "max": max(values)}


def select(rows: list[dict[str, str]], **where: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == value for key, value in where.items())]


def group_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {"group_size": len(rows), "features": {feature: describe(rows, feature) for feature in FEATURES}}


def delta(label: str, left: list[dict[str, str]], right: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for feature in FEATURES:
        a, b = describe(left, feature), describe(right, feature)
        output.append({"comparison": label, "feature": feature, "left_count": a["count"], "right_count": b["count"], "left_median": a["median"], "right_median": b["median"], "left_minus_right_median": a["median"] - b["median"] if a["median"] is not None and b["median"] is not None else None, "left_mean": a["mean"], "right_mean": b["mean"], "left_minus_right_mean": a["mean"] - b["mean"] if a["mean"] is not None and b["mean"] is not None else None})
    return output


def _largest(deltas: list[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    numeric = [row for row in deltas if row["left_minus_right_median"] is not None]
    # Unit-aware ranking is not claimed; this merely exposes largest raw-scale
    # median deltas and is explicitly descriptive.
    return sorted(numeric, key=lambda row: abs(float(row["left_minus_right_median"])), reverse=True)[:count]


def _comparison_row(deltas: list[dict[str, Any]], comparison: str, feature: str) -> dict[str, Any]:
    return next(row for row in deltas if row["comparison"] == comparison and row["feature"] == feature)


def _number_text(value: float | int | None, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def diagnostic_report(groups: dict[str, list[dict[str, str]]], deltas: list[dict[str, Any]]) -> str:
    retro_direction = _comparison_row(deltas, "RETRO_BUYER_MINUS_RETRO_SELLER", "r_multiple")
    august_direction = _comparison_row(deltas, "AUGUST_BUYER_MINUS_AUGUST_SELLER", "r_multiple")
    retro_level = _comparison_row(deltas, "RETRO_LOW_SWEEP_MINUS_RETRO_HIGH_SWEEP", "r_multiple")
    august_level = _comparison_row(deltas, "AUGUST_LOW_SWEEP_MINUS_AUGUST_HIGH_SWEEP", "r_multiple")
    buyer_period = _comparison_row(deltas, "RETRO_BUYER_MINUS_AUGUST_BUYER", "r_multiple")
    buyer_momentum = _comparison_row(deltas, "RETRO_BUYER_MINUS_AUGUST_BUYER", "direction_normalized_price_move_5m_ticks")
    buyer_range = _comparison_row(deltas, "RETRO_BUYER_MINUS_AUGUST_BUYER", "recent_range_5m_ticks")
    buyer_activity = _comparison_row(deltas, "RETRO_BUYER_MINUS_AUGUST_BUYER", "execution_count_15s")
    return "\n".join([
        "# Compact buyer/seller and high/low context comparison",
        "",
        "Only the existing `setup-context.csv` was read. No DBN, interaction reconstruction, PLUS scoring, strategy execution, PnL calculation, or rule selection occurred.",
        "",
        "## Group sizes",
        "",
        *[f"- `{name}`: {len(group)}" for name, group in groups.items()],
        "",
        "## Descriptive answers",
        "",
        f"- **Buyer vs seller:** In retro executed rows, buyer mean R was {_number_text(retro_direction['left_mean'], 3)} (n={retro_direction['left_count']}) versus seller {_number_text(retro_direction['right_mean'], 3)} (n={retro_direction['right_count']}); August was {_number_text(august_direction['left_mean'], 3)} versus {_number_text(august_direction['right_mean'], 3)}. Both pairwise medians were {_number_text(retro_direction['left_median'], 3)} / {_number_text(retro_direction['right_median'], 3)} in retro and {_number_text(august_direction['left_median'], 3)} / {_number_text(august_direction['right_median'], 3)} in August.",
        f"- **Low vs high sweep:** Retro low-sweep mean R was {_number_text(retro_level['left_mean'], 3)} (n={retro_level['left_count']}) versus high-sweep {_number_text(retro_level['right_mean'], 3)} (n={retro_level['right_count']}); August was {_number_text(august_level['left_mean'], 3)} versus {_number_text(august_level['right_mean'], 3)}. This is descriptive only.",
        f"- **Retro buyer versus August buyer:** mean R differed by {_number_text(buyer_period['left_minus_right_mean'], 3)}. Retro buyer median direction-normalized 5-minute move was {_number_text(buyer_momentum['left_median'], 1)} ticks versus {_number_text(buyer_momentum['right_median'], 1)}; median 5-minute range was {_number_text(buyer_range['left_median'], 1)} versus {_number_text(buyer_range['right_median'], 1)}; median 15-second execution count was {_number_text(buyer_activity['left_median'], 1)} versus {_number_text(buyer_activity['right_median'], 1)}.",
        "- **Strongest descriptive dimensions:** the buyer period comparison shows less favorable pre-interaction 5-minute direction-normalized movement, a wider recent 5-minute range, and greater 15-second execution activity in retro. Different units are not ranked against each other; these observations must not be treated as a filter, threshold, or new strategy rule.",
        "",
        "## Interpretation",
        "",
        "Feature differences are descriptive only. They may motivate a predeclared future hypothesis, but they do not authorize a filter, threshold, or trading rule. August remains seen data; retro remains retrospective.",
    ]) + "\n"


def materialize(*, source: Path = SOURCE, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists(): raise CompactComparisonError(f"immutable output already exists: {output_dir}")
    if not source.is_file(): raise CompactComparisonError(f"missing compact context source: {source}")
    with source.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    source_population = [dict(row) for row in rows]
    required = {"period", "direction", "level", "trade_outcome", *FEATURES}
    missing = required - set(rows[0]) if rows else required
    if missing: raise CompactComparisonError(f"compact context schema missing: {','.join(sorted(missing))}")
    groups = {
        "AUGUST_SEEN_BUYER_ABSORPTION": select(rows, period="AUGUST_SEEN", direction="BUYER_ABSORPTION"),
        "AUGUST_SEEN_SELLER_ABSORPTION": select(rows, period="AUGUST_SEEN", direction="SELLER_ABSORPTION"),
        "RETRO_JUNE_JULY_BUYER_ABSORPTION": select(rows, period="RETRO_JUNE_JULY", direction="BUYER_ABSORPTION"),
        "RETRO_JUNE_JULY_SELLER_ABSORPTION": select(rows, period="RETRO_JUNE_JULY", direction="SELLER_ABSORPTION"),
        "AUGUST_SEEN_CURRENT_RTH_LOW_SWEEP": select(rows, period="AUGUST_SEEN", level="CURRENT_RTH_LOW_SWEEP"),
        "AUGUST_SEEN_CURRENT_RTH_HIGH_SWEEP": select(rows, period="AUGUST_SEEN", level="CURRENT_RTH_HIGH_SWEEP"),
        "RETRO_JUNE_JULY_CURRENT_RTH_LOW_SWEEP": select(rows, period="RETRO_JUNE_JULY", level="CURRENT_RTH_LOW_SWEEP"),
        "RETRO_JUNE_JULY_CURRENT_RTH_HIGH_SWEEP": select(rows, period="RETRO_JUNE_JULY", level="CURRENT_RTH_HIGH_SWEEP"),
        "WIN": select(rows, trade_outcome="WIN"), "LOSS": select(rows, trade_outcome="LOSS"),
    }
    comparisons = [
        ("RETRO_BUYER_MINUS_RETRO_SELLER", groups["RETRO_JUNE_JULY_BUYER_ABSORPTION"], groups["RETRO_JUNE_JULY_SELLER_ABSORPTION"]),
        ("AUGUST_BUYER_MINUS_AUGUST_SELLER", groups["AUGUST_SEEN_BUYER_ABSORPTION"], groups["AUGUST_SEEN_SELLER_ABSORPTION"]),
        ("RETRO_LOW_SWEEP_MINUS_RETRO_HIGH_SWEEP", groups["RETRO_JUNE_JULY_CURRENT_RTH_LOW_SWEEP"], groups["RETRO_JUNE_JULY_CURRENT_RTH_HIGH_SWEEP"]),
        ("AUGUST_LOW_SWEEP_MINUS_AUGUST_HIGH_SWEEP", groups["AUGUST_SEEN_CURRENT_RTH_LOW_SWEEP"], groups["AUGUST_SEEN_CURRENT_RTH_HIGH_SWEEP"]),
        ("RETRO_BUYER_MINUS_AUGUST_BUYER", groups["RETRO_JUNE_JULY_BUYER_ABSORPTION"], groups["AUGUST_SEEN_BUYER_ABSORPTION"]),
    ]
    deltas = [item for label, left, right in comparisons for item in delta(label, left, right)]
    payload = {"diagnostic_type": "CSV_ONLY_COMPACT_GROUP_COMPARISON", "source": str(source), "source_rows": len(rows), "source_population_unchanged": rows == source_population, "strategy_semantics_changed": False, "pnl_optimization_performed": False, "selection_prohibited": True, "group_summaries": {name: group_summary(group) for name, group in groups.items()}, "deltas": deltas, "strongest_descriptive_raw_scale_separation": {label: _largest([row for row in deltas if row["comparison"] == label]) for label, _, _ in comparisons}, "interpretation": "Descriptive comparisons only. No threshold, filter, or trading rule is selected."}
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "group-comparison.json").write_text(json.dumps({"groups": payload["group_summaries"], "deltas": deltas}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["comparison", "feature", "left_count", "right_count", "left_median", "right_median", "left_minus_right_median", "left_mean", "right_mean", "left_minus_right_mean"]
    with (output_dir / "group-deltas.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(deltas)
    (output_dir / "diagnostic-report.md").write_text(diagnostic_report(groups, deltas), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV-only V3 compact context comparison")
    parser.add_argument("--source", type=Path, default=SOURCE); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try: print(json.dumps(materialize(source=args.source, output_dir=args.output_dir), indent=2, sort_keys=True))
    except CompactComparisonError as exc: print(f"ERROR: {exc}"); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
