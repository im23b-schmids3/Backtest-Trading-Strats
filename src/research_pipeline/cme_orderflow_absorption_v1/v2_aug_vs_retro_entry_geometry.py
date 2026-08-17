"""Read-only entry-geometry extension for the V2 August-versus-retro study.

It consumes existing completed-trade artifacts and the path-replay functions.
It cannot emit a signal, alter an outcome, or select a rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import v2_aug_vs_retro_diagnostic as artifacts
from . import v2_aug_vs_retro_path_replay as paths


DEFAULT_OUTPUT = artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_DIAGNOSTIC/aug_vs_retro_entry_geometry"
TICK = 0.25


class EntryGeometryError(RuntimeError):
    pass


def _long(direction: str) -> bool:
    if direction in {"BUYER_ABSORPTION", "LONG"}:
        return True
    if direction in {"SELLER_ABSORPTION", "SHORT"}:
        return False
    raise EntryGeometryError(f"unknown direction: {direction!r}")


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value in (None, "") else float(value)


def entry_geometry(row: dict[str, Any]) -> dict[str, Any]:
    """Add direction-aware descriptive geometry to an immutable recorded trade."""
    entry = _number(row, "entry")
    end = _number(row, "interaction_end_price")
    confirmation = _number(row, "confirmation_price")
    stop, target = _number(row, "stop"), _number(row, "target")
    if entry is None or end is None or stop is None or target is None:
        raise EntryGeometryError("recorded trade lacks required entry/interaction/stop/target price")
    sign = 1.0 if _long(str(row["direction"])) else -1.0
    displacement_points = sign * (entry - end)
    confirmation_displacement_points = sign * (entry - confirmation) if confirmation is not None else None
    stop_distance_points, target_distance_points = abs(entry - stop), abs(target - entry)
    enriched = dict(row)
    enriched.update({
        "interaction_end_price": end,
        "confirmation_price": confirmation,
        "entry": entry,
        "entry_displacement_points_from_interaction_end": displacement_points,
        "entry_displacement_ticks_from_interaction_end": displacement_points / TICK,
        "entry_displacement_ticks_from_confirmation_price": confirmation_displacement_points / TICK if confirmation_displacement_points is not None else None,
        "zone_low": _number(row, "zone_low"),
        "zone_high": _number(row, "zone_high"),
        "stop": stop,
        "target": target,
        "stop_distance_points_from_entry": stop_distance_points,
        "stop_distance_ticks_from_entry": stop_distance_points / TICK,
        "target_distance_points_from_entry": target_distance_points,
        "target_distance_ticks_from_entry": target_distance_points / TICK,
        "target_r": target_distance_points / stop_distance_points if stop_distance_points else None,
    })
    return enriched


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p25": None, "p75": None}
    return {
        "count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values), "p25": _percentile(values, 0.25), "p75": _percentile(values, 0.75),
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y))
    return numerator / denominator if denominator else None


def period_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "confirmation_favorable_ticks", "entry_displacement_ticks_from_interaction_end",
        "entry_displacement_ticks_from_confirmation_price", "stop_distance_ticks_from_entry",
        "target_distance_ticks_from_entry",
    )
    paired = [(float(row["entry_displacement_ticks_from_interaction_end"]), float(row["mfe_ticks"])) for row in rows if row.get("entry_displacement_ticks_from_interaction_end") is not None and row.get("mfe_ticks") is not None]
    return {
        "trade_count": len(rows),
        "distributions": {field: _distribution(rows, field) for field in fields},
        "entry_displacement_to_mfe_pearson": _pearson([item[0] for item in paired], [item[1] for item in paired]),
        "entry_displacement_to_mfe_pair_count": len(paired),
    }


def materialize(*, output_dir: Path = DEFAULT_OUTPUT, retro_data_root: Path = paths.DEFAULT_RETRO_DATA) -> dict[str, Any]:
    """Run the declared local read-only scans and write a new diagnostic tree."""
    if output_dir.exists():
        raise EntryGeometryError(f"immutable diagnostic output already exists: {output_dir}")
    august_source, retro_source = artifacts.DEFAULT_AUGUST_ROOT, artifacts.DEFAULT_RETRO_ROOT
    august_rows = [artifacts._augment_trade(row, period="AUGUST_SEEN_3R", execution_model="MES_PROXY_EXECUTION_FROM_ES_MBO") for row in artifacts._read_csv(august_source / "trades_3_0R.csv")]
    retro_signals = {row["interaction_id"]: row for row in artifacts._read_csv(retro_source / "plus-signals.csv")}
    retro_rows = [artifacts._augment_trade(row, period="RETRO_JUNE_JULY_2P5R", execution_model="NATIVE_MES_MBP1_FALLBACK", interaction=retro_signals.get(row["interaction_id"])) for row in artifacts._read_csv(retro_source / "trades.csv")]
    replayed = paths.replay_recorded_trade_paths(august_trades=august_rows, retro_trades=retro_rows, retro_data_root=retro_data_root)
    geometry_rows = [entry_geometry(row) for row in replayed]
    august_geometry = [row for row in geometry_rows if row["period"] == "AUGUST_SEEN_3R"]
    retro_geometry = [row for row in geometry_rows if row["period"] == "RETRO_JUNE_JULY_2P5R"]
    periods = {"august": period_geometry(august_geometry), "retro": period_geometry(retro_geometry)}
    questions = {
        "retro_entries_farther_from_interaction_end": "DESCRIPTIVE: compare entry_displacement_ticks_from_interaction_end distributions; no cutoff or rule is selected.",
        "retro_entries_extended_at_entry": "DESCRIPTIVE: inspect positive entry displacement distributions only.",
        "confirmation_wait_more_chasing": "DESCRIPTIVE: compare entry_displacement_ticks_from_confirmation_price; it is not a change to the 15-second wait.",
        "displacement_associated_with_lower_mfe": "DESCRIPTIVE_ONLY Pearson association in period summaries; it is not a causal claim or rule.",
        "retro_stop_distances_different": "DESCRIPTIVE: compare stop_distance_ticks_from_entry, which retains the existing zone-stop semantics.",
    }
    summary = {
        "diagnostic_type": "READ_ONLY_ENTRY_GEOMETRY_AND_PATH_REPLAY",
        "strategy_semantics_changed": False,
        "pnl_optimization_performed": False,
        "new_strategy_rule_selected": False,
        "august_interpretation": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE",
        "retro_interpretation": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST",
        "field_availability": {"august": "ALL_REQUESTED_FIELDS_FROM_LEDGER_AND_ES_MBO_REPLAY", "retro": "ALL_REQUESTED_FIELDS_FROM_LEDGER_PLUS_ES_MBO/MES_MBP1_REPLAY"},
        "execution_models": {"august_mes": "MES_PROXY_EXECUTION_FROM_ES_MBO", "retro_mes": "NATIVE_MES_MBP1_FALLBACK"},
        "periods": periods,
        "questions": questions,
    }
    output_dir.mkdir(parents=True)
    fields = sorted({key for row in geometry_rows for key in row})
    with (output_dir / "trade-entry-geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(geometry_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "period-entry-geometry.json").write_text(json.dumps(periods, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# V2 August vs retro entry geometry", "",
        "Descriptive only. August is seen-data research, not fresh OOS. June/July is retrospective robustness, not strict chronological OOS. No entry cutoff, filter, or strategy rule was selected.", "",
        "## Field definitions", "",
        "- `interaction_end_price`, `zone_low`, and `zone_high`: raw Databento `1e9` values converted once to ES points.",
        "- LONG displacement: `(entry - interaction_end_price) / 0.25`; SHORT displacement: `(interaction_end_price - entry) / 0.25`. Positive values are farther in the expected direction.",
        "- Confirmation-to-entry displacement applies the same direction-aware formula against `confirmation_price`.",
        "- Stop/target distances are absolute ES-point differences from entry, displayed also in ticks.",
        "- MFE/MAE retains the path-replay executable marks: long `bid - 1 tick`, short `ask + 1 tick`, only between recorded entry and exit.", "",
        "## Period distributions", "", "```json", json.dumps(periods, indent=2, sort_keys=True), "```", "",
        "## Limitations", "",
        "August MES observations are ES-MBO proxy marks; retro MES observations are native MES MBP-1. The difference is preserved, not normalized away. Any observed subgroup or association is descriptive only and cannot become a new rule without fresh validation.",
    ]
    (output_dir / "diagnostic-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V2 entry-geometry diagnostic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retro-data-root", type=Path, default=paths.DEFAULT_RETRO_DATA)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(output_dir=args.output_dir, retro_data_root=args.retro_data_root), indent=2, sort_keys=True))
    except EntryGeometryError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
