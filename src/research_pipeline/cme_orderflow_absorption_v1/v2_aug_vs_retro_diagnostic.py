"""Read-only comparison of existing August research and V2 retro artifacts.

This module deliberately consumes only already-materialized CSV/JSON artifacts.
It never opens a DBN, reconstructs interactions, selects a rule, or changes a
trade.  Trade-path excursions require a causal market-data replay and therefore
remain explicitly unavailable in this artifact-only diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUGUST_ROOT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_aug_target_matrix"
DEFAULT_RETRO_ROOT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RETRO_HOLDOUT/2026-06-23_2026-07-17-fixed"
DEFAULT_OUTPUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_DIAGNOSTIC/aug_vs_retro"
RAW_PRICE_SCALE = 1_000_000_000
TICK = 0.25

COMPARISON_COLUMNS = (
    "period", "date", "interaction_id", "direction", "level",
    "absorption_score", "replenishment_score", "interaction_end",
    "interaction_end_price", "confirmation_timestamp", "confirmation_price", "confirmation_favorable_ticks", "entry_timestamp",
    "entry_utc", "entry", "stop", "target", "exit_timestamp", "exit_reason",
    "zone_low", "zone_high",
    "r_multiple", "instrument", "contracts", "initial_risk_usd",
    "stop_distance_ticks", "duration_seconds", "seconds_confirmation_to_entry",
    "mfe_ticks", "mae_ticks", "maximum_favorable_r", "maximum_adverse_r",
    "seconds_to_mfe", "reached_0_5r", "reached_1_0r", "reached_1_5r",
    "reached_2_0r", "reached_2_5r", "reached_3_0r", "excursion_status",
    "target_r", "execution_model",
)


class DiagnosticError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _utc_timestamp(timestamp_ns: int | None) -> str | None:
    if timestamp_ns is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(timestamp_ns / RAW_PRICE_SCALE, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _target_r(row: dict[str, Any]) -> float | None:
    explicit = _float(row, "target_multiple")
    if explicit is not None:
        return explicit
    entry, stop, target = (_float(row, key) for key in ("entry", "stop", "target"))
    if entry is None or stop is None or target is None or entry == stop:
        return None
    return abs(target - entry) / abs(entry - stop)


def _raw_price_to_points(row: dict[str, Any], key: str) -> float | None:
    value = _float(row, key)
    return value / RAW_PRICE_SCALE if value is not None else None


def _initial_risk(row: dict[str, Any]) -> float | None:
    value = _float(row, "one_contract_initial_risk_usd")
    if value is not None:
        return value
    entry, stop = _float(row, "entry"), _float(row, "stop")
    instrument = row.get("instrument")
    if entry is None or stop is None or instrument not in {"ES", "MES"}:
        return None
    point_value, commission = (50.0, 3.0) if instrument == "ES" else (5.0, 1.25)
    # The sealed model's adverse stop-side exit adds one ES tick beyond stop.
    return (abs(entry - stop) + TICK) * point_value + 2 * commission


def _augment_trade(row: dict[str, Any], *, period: str, execution_model: str, interaction: dict[str, str] | None = None) -> dict[str, Any]:
    combined: dict[str, Any] = {**(interaction or {}), **row}
    entry_ns, exit_ns = _int(combined, "entry_timestamp"), _int(combined, "exit_timestamp")
    confirmation_ns = _int(combined, "confirmation_timestamp")
    entry, stop = _float(combined, "entry"), _float(combined, "stop")
    result: dict[str, Any] = {
        "period": period,
        "date": combined.get("date"),
        "interaction_id": combined.get("interaction_id"),
        "direction": combined.get("direction"),
        "level": combined.get("level"),
        "absorption_score": _float(combined, "absorption_score"),
        "replenishment_score": _float(combined, "replenishment_score"),
        "interaction_end": _int(combined, "interaction_end") or _int(combined, "end_ns"),
        "interaction_end_price": _raw_price_to_points(combined, "end_price"),
        "confirmation_timestamp": confirmation_ns,
        "confirmation_favorable_ticks": _float(combined, "confirmation_favorable_ticks"),
        "confirmation_price": _float(combined, "confirmation_price"),
        "entry_timestamp": entry_ns,
        "entry_utc": _utc_timestamp(entry_ns),
        "entry": entry,
        "zone_low": _raw_price_to_points(combined, "zone_low"),
        "zone_high": _raw_price_to_points(combined, "zone_high"),
        "stop": stop,
        "target": _float(combined, "target"),
        "exit_timestamp": exit_ns,
        "exit_reason": combined.get("exit_reason"),
        "r_multiple": _float(combined, "r_multiple"),
        "instrument": combined.get("instrument"),
        "contracts": _int(combined, "contracts"),
        "initial_risk_usd": _initial_risk(combined),
        "stop_distance_ticks": abs(entry - stop) / TICK if entry is not None and stop is not None else None,
        "duration_seconds": (exit_ns - entry_ns) / RAW_PRICE_SCALE if entry_ns is not None and exit_ns is not None else None,
        "seconds_confirmation_to_entry": (entry_ns - confirmation_ns) / RAW_PRICE_SCALE if entry_ns is not None and confirmation_ns is not None else None,
        # No extrema are contained in an immutable trade ledger.  Replaying
        # source DBNs is intentionally not part of this artifact-only analysis.
        "mfe_ticks": None, "mae_ticks": None, "maximum_favorable_r": None,
        "maximum_adverse_r": None, "seconds_to_mfe": None,
        "reached_0_5r": None, "reached_1_0r": None, "reached_1_5r": None,
        "reached_2_0r": None, "reached_2_5r": None, "reached_3_0r": None,
        "excursion_status": "NOT_COMPUTED_NO_CAUSAL_PRICE_PATH_REPLAY",
        "target_r": _target_r(combined),
        "execution_model": execution_model,
    }
    return result


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def _categorical_summary(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) if row.get(field) is not None else "UNAVAILABLE")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _trade_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "numeric": {field: _numeric_summary(rows, field) for field in (
            "absorption_score", "replenishment_score", "confirmation_favorable_ticks",
            "stop_distance_ticks", "initial_risk_usd", "duration_seconds", "mfe_ticks", "mae_ticks",
        )},
        "categorical": {field: _categorical_summary(rows, field) for field in ("direction", "level", "instrument", "exit_reason")},
        "entry_utc_hour": _categorical_summary(
            [{"entry_hour": row["entry_utc"][11:13] if row.get("entry_utc") else None} for row in rows], "entry_hour"
        ),
    }


def _period_payload(summary: dict[str, Any], trades: list[dict[str, Any]], *, source: str, raw_interactions: int, plus_count: int, confirmations_passed: int, confirmations_failed: int) -> dict[str, Any]:
    trade_count = len(trades)
    target_exits = sum(row["exit_reason"] == "TARGET" for row in trades)
    stop_exits = sum(row["exit_reason"] == "STOP" for row in trades)
    return {
        "source": source,
        "interpretation": summary.get("interpretation"),
        "raw_interactions": raw_interactions,
        "plus_count": plus_count,
        "plus_rate": plus_count / raw_interactions if raw_interactions else None,
        "confirmations_passed": confirmations_passed,
        "confirmations_failed": confirmations_failed,
        "confirmation_pass_rate": confirmations_passed / (confirmations_passed + confirmations_failed) if confirmations_passed + confirmations_failed else None,
        "completed_trades": trade_count,
        "completed_trade_per_confirmation_rate": trade_count / confirmations_passed if confirmations_passed else None,
        "target_exits": target_exits,
        "stop_exits": stop_exits,
        "target_exit_rate": target_exits / trade_count if trade_count else None,
        "total_r": sum(float(row["r_multiple"]) for row in trades if row["r_multiple"] is not None),
        "distributions": _trade_distribution(trades),
        "summary_status": summary.get("status"),
    }


def build_comparison(*, august_root: Path = DEFAULT_AUGUST_ROOT, retro_root: Path = DEFAULT_RETRO_ROOT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Materialize the immutable-artifact diagnostic without reading market data."""
    if output_dir.exists():
        raise DiagnosticError(f"diagnostic output already exists: {output_dir}")
    august_summary = _read_json(august_root / "summary.json")
    retro_summary = _read_json(retro_root / "summary.json")
    august_rows = _read_csv(august_root / "trades_3_0R.csv")
    retro_rows = _read_csv(retro_root / "trades.csv")
    retro_interactions = {row["interaction_id"]: row for row in _read_csv(retro_root / "plus-signals.csv")}

    august = [_augment_trade(row, period="AUGUST_SEEN_3R", execution_model="MES_PROXY_EXECUTION_FROM_ES_MBO") for row in august_rows]
    retro = [_augment_trade(row, period="RETRO_JUNE_JULY_2P5R", execution_model="NATIVE_MES_MBP1_FALLBACK", interaction=retro_interactions.get(row["interaction_id"])) for row in retro_rows]
    all_trades = august + retro

    august_period = _period_payload(
        august_summary, august, source=str(august_root), raw_interactions=1430,
        plus_count=int(august_summary["v1_plus_input"]), confirmations_passed=int(august_summary["confirmations_passed"]),
        confirmations_failed=int(august_summary["confirmations_failed"]),
    )
    retro_period = _period_payload(
        retro_summary, retro, source=str(retro_root), raw_interactions=int(retro_summary["raw_interactions"]),
        plus_count=int(retro_summary["plus_count"]), confirmations_passed=int(retro_summary["confirmations_passed"]),
        confirmations_failed=int(retro_summary["confirmations_failed"]),
    )
    hypothesis = {
        "statement": "V1 PLUS and 15-second confirmation occur at similar rates in both periods, but post-confirmation continuation is materially weaker in June/July.",
        "plus_rate_comparison": {"august": august_period["plus_rate"], "retro": retro_period["plus_rate"], "difference": august_period["plus_rate"] - retro_period["plus_rate"]},
        "confirmation_pass_rate_comparison": {"august": august_period["confirmation_pass_rate"], "retro": retro_period["confirmation_pass_rate"], "difference": august_period["confirmation_pass_rate"] - retro_period["confirmation_pass_rate"]},
        "completed_trade_per_confirmation_comparison": {"august": august_period["completed_trade_per_confirmation_rate"], "retro": retro_period["completed_trade_per_confirmation_rate"]},
        "post_confirmation_continuation_comparable": False,
        "finding": "SUPPORTED_ONLY_AS_A_DESCRIPTIVE_OUTCOME_DIFFERENCE: PLUS rates are near-equal and both periods execute roughly 91% of passed confirmations, but target exits/R are not apples-to-apples because August used 3R with MES proxy execution and retro used 2.5R with native MES fallback. MFE/MAE is unavailable without a causal replay.",
    }
    payload = {
        "diagnostic_type": "READ_ONLY_ARTIFACT_COMPARISON",
        "strategy_semantics_changed": False,
        "pnl_optimization_performed": False,
        "new_strategy_rule_selected": False,
        "august_interpretation": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE",
        "retro_interpretation": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST",
        "mfe_mae": {"computed": False, "reason": "Trade artifacts retain entry/exit only; no causal DBN price-path replay was performed."},
        "apples_to_apples_limitations": [
            "August comparator is the existing 3R target-matrix artifact; retro uses the frozen 2.5R runner.",
            "August MES execution is proxied from ES MBO; retro MES fallback uses native MES MBP-1.",
            "Retro has one unresolved tail interaction and its summary is explicitly partial.",
            "An August 2.5R price-path outcome requires a full causal replay and was intentionally not computed.",
        ],
        "periods": {"august": august_period, "retro": retro_period},
        "hypothesis": hypothesis,
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "trade-comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_trades)
    (output_dir / "period-comparison.json").write_text(json.dumps(payload["periods"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# August vs retrospective V2 diagnostic", "",
        "This is a descriptive, artifact-only comparison. August is seen-data research, not fresh OOS. June/July is retrospective robustness, not strict chronological OOS. No new strategy rule was selected and no PnL optimization was performed.", "",
        "## Rate comparison", "",
        "| Period | Raw interactions | PLUS | PLUS rate | Passed confirmations | Pass rate | Completed trades | Total R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| August seen (3R) | {august_period['raw_interactions']} | {august_period['plus_count']} | {august_period['plus_rate']:.4%} | {august_period['confirmations_passed']} | {august_period['confirmation_pass_rate']:.4%} | {august_period['completed_trades']} | {august_period['total_r']:.4f} |",
        f"| June/July retro (2.5R) | {retro_period['raw_interactions']} | {retro_period['plus_count']} | {retro_period['plus_rate']:.4%} | {retro_period['confirmations_passed']} | {retro_period['confirmation_pass_rate']:.4%} | {retro_period['completed_trades']} | {retro_period['total_r']:.4f} |",
        "",
        "## Hypothesis", "", hypothesis["finding"], "",
        "## Excursions", "",
        "MFE, MAE, maximum favorable/adverse R, threshold-reached flags, and time-to-MFE are `NOT_COMPUTED_NO_CAUSAL_PRICE_PATH_REPLAY` for both periods. They are not recoverable from trade ledgers alone, and no full DBN replay was run.", "",
        "## Apples-to-apples limitations", "",
        *[f"- {item}" for item in payload["apples_to_apples_limitations"]], "",
        "Any descriptive subgroup in `period-comparison.json` must not be treated as a new rule without fresh validation.",
    ]
    (output_dir / "diagnostic-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V2 August-versus-retro artifact diagnostic; no DBN replay")
    parser.add_argument("--august-root", type=Path, default=DEFAULT_AUGUST_ROOT)
    parser.add_argument("--retro-root", type=Path, default=DEFAULT_RETRO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        print(json.dumps(build_comparison(august_root=args.august_root, retro_root=args.retro_root, output_dir=args.output_dir), indent=2, sort_keys=True))
    except DiagnosticError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
