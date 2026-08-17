"""Read-only score-validity diagnostic for published L2 V2 May artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


OUTPUT_NAME = "score_validity"
COMPONENTS = (
    "aggression_score", "restoration_score", "price_resistance_score", "persistence_score",
    "multi_level_support_score", "false_refill_penalty", "l2_absorption_quality_score",
)
RAW_FEATURES = (
    "directional_aggressive_volume", "relevant_execution_count", "depth_restoration_count",
    "consume_restore_cycles", "cumulative_restored_volume", "restoration_to_consumption_ratio",
    "mean_restoration_latency_ms", "executed_to_initial_displayed_ratio",
    "defended_price_present_fraction", "maximum_through_level_progress_ticks",
    "interaction_rejection_ticks", "depth_imbalance_1", "depth_imbalance_3", "depth_imbalance_5",
    "multi_level_ofi", "rapid_cancel_ratio",
)
BUCKETS = (
    ("0.50_to_0.52", .50, .52), ("0.52_to_0.54", .52, .54),
    ("0.54_to_0.56", .54, .56), ("0.56_to_0.58", .56, .58),
    ("0.58_or_higher", .58, math.inf),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        return 0.0
    return float(value)


def _quantile(values: Iterable[float], point: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * point
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    return {"count": len(values), "mean": mean(values), "median": median(values), "p25": _quantile(values, .25),
            "p75": _quantile(values, .75), "p90": _quantile(values, .90), "min": min(values), "max": max(values)}


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end + 1) / 2.0
        for index in range(start, end + 1):
            ranks[indexed[index][0]] = rank
        start = end + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = mean(rx), mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def bucket_for(score: float) -> str:
    for name, lower, upper in BUCKETS:
        if lower <= score < upper:
            return name
    raise ValueError(f"quality score outside frozen V2 bucket domain: {score}")


def _performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [row for row in rows if row.get("trade") is not None]
    values = [_number(row["trade"], "r_multiple") for row in trades]
    pnl = [_number(row["trade"], "net_pnl_usd") for row in trades]
    wins, losses = sum(value > 0 for value in pnl), sum(value < 0 for value in pnl)
    gross_win, gross_loss = sum(value for value in pnl if value > 0), abs(sum(value for value in pnl if value < 0))
    return {"accepted_setups": len(rows), "confirmations": sum(row["outcome"] == "CONFIRMED" for row in rows),
            "confirmation_rate": sum(row["outcome"] == "CONFIRMED" for row in rows) / len(rows) if rows else 0.0,
            "trades": len(trades), "wins": wins, "losses": losses, "win_rate": wins / len(trades) if trades else 0.0,
            "total_r": sum(values), "average_r": sum(values) / len(values) if values else None,
            "median_r": median(values) if values else None, "profit_factor": gross_win / gross_loss if gross_loss else None}


def _component_rows(groups: dict[str, list[dict[str, Any]]], fields: Iterable[str]) -> list[dict[str, Any]]:
    return [{"component": field, "population": population, **stats([_number(row, field) for row in rows])}
            for field in fields for population, rows in groups.items()]


def _quartiles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (_number(row, "l2_absorption_quality_score"), row["interaction_id"]))
    if len(ordered) % 4:
        raise ValueError("accepted setup population must divide exactly into four fixed quartiles")
    width = len(ordered) // 4
    return {name: ordered[index * width:(index + 1) * width] for index, name in enumerate(("bottom_quartile", "Q2", "Q3", "top_quartile"))}


def _classification(quality_rho: float | None, quartiles: dict[str, dict[str, Any]], trade_count: int) -> str:
    """Fixed descriptive label, not a trading gate or parameter-selection rule."""
    if trade_count < 12 or quality_rho is None:
        return "INSUFFICIENT_SAMPLE"
    bottom, top = quartiles["bottom_quartile"]["average_r"], quartiles["top_quartile"]["average_r"]
    if quality_rho >= .25 and bottom is not None and top is not None and top > bottom:
        return "CLEAR_POSITIVE_RANKING"
    if quality_rho >= .10 and bottom is not None and top is not None and top > bottom:
        return "PARTIAL_POSITIVE_RANKING"
    if quality_rho <= -.10:
        return "INVERSE_RANKING"
    return "NO_USEFUL_RANKING"


def analyze(setups: list[dict[str, str]], trades: list[dict[str, str]]) -> dict[str, Any]:
    accepted = [dict(row) for row in setups if row.get("accepted") == "True"]
    if len(accepted) != 56:
        raise ValueError("published V2 setup ledger does not contain the sealed 56 accepted setups")
    trade_by_id = {row["interaction_id"]: dict(row) for row in trades}
    if len(trade_by_id) != len(trades):
        raise ValueError("published V2 trade ledger contains duplicate interaction_id values")
    records: list[dict[str, Any]] = []
    for row in accepted:
        interaction_id = row["interaction_id"]
        terminal = row.get("terminal_reason")
        if interaction_id in trade_by_id:
            outcome = "CONFIRMED"
        elif terminal == "CONFIRMATION_WINDOW_EXPIRED":
            outcome = "CONFIRMATION_WINDOW_EXPIRED"
        elif terminal == "COMPLIANCE_BLOCK_ACTIVE_POSITION":
            outcome = "COMPLIANCE_BLOCK_ACTIVE_POSITION"
        else:
            raise ValueError(f"accepted setup has unrecognised terminal state: {interaction_id}")
        direction = {"BUYER_ABSORPTION": "LONG", "SELLER_ABSORPTION": "SHORT"}.get(row.get("direction"))
        if direction is None:
            raise ValueError(f"accepted setup has unrecognised absorption direction: {interaction_id}")
        records.append({**row, "direction": direction, "outcome": outcome, "trade": trade_by_id.get(interaction_id)})
    if sum(row["outcome"] == "CONFIRMED" for row in records) != 32:
        raise ValueError("published V2 confirmation/trade count does not reconcile to 32")

    buckets = {name: [row for row in records if bucket_for(_number(row, "l2_absorption_quality_score")) == name] for name, _, _ in BUCKETS}
    quartile_rows = _quartiles(records)
    bucket_metrics = {name: _performance(rows) for name, rows in buckets.items()}
    quartile_metrics = {name: _performance(rows) for name, rows in quartile_rows.items()}
    component_groups = {
        "accepted": records,
        "confirmed": [row for row in records if row["outcome"] == "CONFIRMED"],
        "confirmation_expired": [row for row in records if row["outcome"] == "CONFIRMATION_WINDOW_EXPIRED"],
        "compliance_blocked": [row for row in records if row["outcome"] == "COMPLIANCE_BLOCK_ACTIVE_POSITION"],
        "winning_trade": [row for row in records if row.get("trade") and _number(row["trade"], "net_pnl_usd") > 0],
        "losing_trade": [row for row in records if row.get("trade") and _number(row["trade"], "net_pnl_usd") < 0],
    }
    components = _component_rows(component_groups, COMPONENTS)
    trade_records = [row for row in records if row.get("trade")]
    raw = _component_rows({"winning_trade": component_groups["winning_trade"], "losing_trade": component_groups["losing_trade"]}, RAW_FEATURES)
    correlations = {field: spearman([_number(row, field) for row in trade_records], [_number(row["trade"], "r_multiple") for row in trade_records])
                    for field in COMPONENTS}
    separation: list[dict[str, Any]] = []
    for field in COMPONENTS:
        all_values = [_number(row, field) for row in records]
        iqr = (_quantile(all_values, .75) or 0.0) - (_quantile(all_values, .25) or 0.0)
        win_median = median([_number(row, field) for row in component_groups["winning_trade"]])
        loss_median = median([_number(row, field) for row in component_groups["losing_trade"]])
        separation.append({"component": field, "winner_minus_loser_median": win_median - loss_median,
                           "absolute_iqr_normalized_median_separation": abs(win_median - loss_median) / iqr if iqr else 0.0,
                           "spearman_with_r": correlations[field]})
    separation.sort(key=lambda row: (-row["absolute_iqr_normalized_median_separation"], row["component"]))
    level_direction = {
        name: {"direction": {key: sum(row.get("direction") == key for row in rows) for key in ("LONG", "SHORT")},
               "level": {key: sum(row.get("level") == key for row in rows) for key in ("PRIOR_RTH_HIGH", "PRIOR_RTH_LOW", "PRIOR_RTH_POC", "PRIOR_RTH_VAH", "PRIOR_RTH_VAL")}}
        for name, rows in buckets.items()
    }
    loo = {}
    for name, rows in buckets.items():
        r_values = [_number(row["trade"], "r_multiple") for row in rows if row.get("trade")]
        if len(r_values) >= 3:
            loo[name] = {"trade_count": len(r_values), "original_total_r": sum(r_values),
                         "without_best_trade_total_r": sum(r_values) - max(r_values),
                         "without_worst_trade_total_r": sum(r_values) - min(r_values)}
    poc_val = [row for row in records if row.get("level") in {"PRIOR_RTH_POC", "PRIOR_RTH_VAL"}]
    quality_rho = correlations["l2_absorption_quality_score"]
    return {
        "strategy_id": "CMEOrderflowAbsorption.ES_L2_V2", "evidence_label": "MAY_DEVELOPMENT_V2_QUALITY_0_50_NOT_OOS_EVIDENCE",
        "diagnostic_scope": "PUBLISHED_LEDGER_ONLY_DESCRIPTIVE_SCORE_VALIDITY", "strategy_parameters_mutated": False,
        "accepted_setup_quality_score": stats([_number(row, "l2_absorption_quality_score") for row in records]),
        "quality_buckets": bucket_metrics, "quartiles": quartile_metrics,
        "quality_ranking_classification": _classification(quality_rho, quartile_metrics, len(trade_records)),
        "component_comparisons": components, "component_spearman_with_r": correlations,
        "component_winner_loser_separation": separation, "raw_feature_winner_loser_comparisons": raw,
        "confirmation_outcomes": {name: sum(row["outcome"] == name for row in records) for name in ("CONFIRMED", "CONFIRMATION_WINDOW_EXPIRED", "COMPLIANCE_BLOCK_ACTIVE_POSITION")},
        "bucket_level_direction_context": level_direction, "leave_one_out_by_bucket": loo,
        "poc_val_descriptive_appendix": {"setups": len(poc_val), **_performance(poc_val),
                                          "quality_score_distribution": stats([_number(row, "l2_absorption_quality_score") for row in poc_val])},
    }


def _rows_for_csv(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets = [{"bucket": name, **metrics, **summary["bucket_level_direction_context"][name]["direction"],
                **summary["bucket_level_direction_context"][name]["level"]} for name, metrics in summary["quality_buckets"].items()]
    components = [*summary["component_comparisons"], *[
        {"component": row["component"], "population": "winner_loser_separation", **row} for row in summary["component_winner_loser_separation"]
    ]]
    return buckets, components, summary["raw_feature_winner_loser_comparisons"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    quality = summary["accepted_setup_quality_score"]
    buckets = summary["quality_buckets"]
    quartiles = summary["quartiles"]
    separation = summary["component_winner_loser_separation"]
    strongest, weakest = separation[0], separation[-1]
    confirmation_quality = {row["population"]: row for row in summary["component_comparisons"]
                            if row["component"] == "l2_absorption_quality_score" and row["population"] in {"confirmed", "confirmation_expired"}}
    poc_val = summary["poc_val_descriptive_appendix"]
    return "\n".join([
        "# L2 V2 May development score-validity diagnostic", "",
        "Published-ledger-only descriptive analysis. May is development evidence, not OOS evidence. No score, weight, threshold, or execution behavior was changed.", "",
        f"Accepted-score distribution: min={quality['min']:.6f}, median={quality['median']:.6f}, mean={quality['mean']:.6f}, max={quality['max']:.6f}.",
        f"Quality-to-R Spearman: {summary['component_spearman_with_r']['l2_absorption_quality_score']}; classification: `{summary['quality_ranking_classification']}`.", "",
        "## Fixed quality buckets", "",
        *[f"- {name}: setups={row['accepted_setups']}, trades={row['trades']}, total_R={row['total_r']:.6f}, PF={row['profit_factor']}" for name, row in buckets.items()], "",
        "## Fixed rank quartiles", "",
        *[f"- {name}: setups={row['accepted_setups']}, trades={row['trades']}, total_R={row['total_r']:.6f}, average_R={row['average_r']}" for name, row in quartiles.items()], "",
        "## Component and confirmation context", "",
        f"- Largest winner/loser median separation under the fixed IQR-normalized descriptive measure: {strongest['component']} ({strongest['absolute_iqr_normalized_median_separation']:.6f}).",
        f"- Weakest: {weakest['component']} ({weakest['absolute_iqr_normalized_median_separation']:.6f}); Spearman={weakest['spearman_with_r']}.",
        f"- Quality mean, confirmed vs expired: {confirmation_quality['confirmed']['mean']:.6f} vs {confirmation_quality['confirmation_expired']['mean']:.6f}.",
        f"- POC+VAL descriptive appendix: setups={poc_val['setups']}, trades={poc_val['trades']}, wins/losses={poc_val['wins']}/{poc_val['losses']}, total_R={poc_val['total_r']:.6f}.", "",
        "## Leave-one-out", "",
        *[f"- {name}: original={row['original_total_r']:.6f}R; without best={row['without_best_trade_total_r']:.6f}R; without worst={row['without_worst_trade_total_r']:.6f}R." for name, row in summary["leave_one_out_by_bucket"].items()], "",
        "No bucket, component, level, direction, or POC/VAL result is a selected rule.", "",
    ])


def materialize(artifact_root: Path) -> dict[str, Any]:
    output = artifact_root / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"immutable score-validity output already exists: {output}")
    summary = analyze(_read_csv(artifact_root / "setup-ledger.csv"), _read_csv(artifact_root / "trade-ledger.csv"))
    output.mkdir(parents=True, exist_ok=False)
    buckets, components, raw = _rows_for_csv(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "quality-buckets.csv", buckets)
    _write_csv(output / "component-comparison.csv", components)
    _write_csv(output / "raw-feature-comparison.csv", raw)
    (output / "diagnostic-report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only published-ledger L2 V2 score-validity diagnostic")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = materialize(args.artifact_root)
    except (FileExistsError, FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"classification": summary["quality_ranking_classification"],
                      "quality_spearman_with_r": summary["component_spearman_with_r"]["l2_absorption_quality_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
