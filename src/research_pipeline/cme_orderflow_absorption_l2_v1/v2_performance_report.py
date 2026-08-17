"""Ledger-only reporting repair for a completed L2 V2 May development replay.

No source-market reader is imported here.  This module can only aggregate the
already published CSV/JSON artifacts and replace their stale derived summary
and Markdown metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V2"
EVIDENCE_LABEL = "MAY_DEVELOPMENT_V2_QUALITY_0_50_NOT_OOS_EVIDENCE"
VARIANT_LABEL = "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _performance(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    values = [float(row["r_multiple"]) for row in ordered]
    pnl = [float(row["net_pnl_usd"]) for row in ordered]
    wins, losses = sum(value > 0 for value in pnl), sum(value < 0 for value in pnl)
    profits, losses_usd = sum(value for value in pnl if value > 0), abs(sum(value for value in pnl if value < 0))
    equity = peak = 0.0
    max_drawdown = 0.0
    longest = current = 0
    for value, cash in zip(values, pnl):
        equity += value; peak = max(peak, equity); max_drawdown = min(max_drawdown, equity - peak)
        if cash < 0:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return {
        "completed_trades": len(ordered), "wins": wins, "losses": losses,
        "win_rate": wins / len(ordered) if ordered else 0.0,
        "total_r": sum(values), "average_r": sum(values) / len(values) if values else 0.0,
        "median_r": float(median(values)) if values else None, "net_pnl_usd": sum(pnl),
        "profit_factor": profits / losses_usd if losses_usd else None,
        "max_cumulative_drawdown_r": max_drawdown, "longest_losing_streak": longest,
        "es_trades": sum(row["instrument"] == "ES" for row in ordered),
        "mes_trades": sum(row["instrument"] == "MES" for row in ordered),
        "stop_exits": sum(row["exit_reason"] == "STOP" for row in ordered),
        "target_exits": sum(row["exit_reason"] == "TARGET" for row in ordered),
        "cutoff_exits": sum(row["exit_reason"] == "HARD_CUTOFF_2245" for row in ordered),
    }


def _breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _performance(value) for key, value in sorted(groups.items())}


def build_summary(
    *, setups: list[dict[str, str]], trades: list[dict[str, str]], daily: list[dict[str, str]],
    existing_summary: dict[str, Any],
) -> dict[str, Any]:
    if any(row.get("strategy_id") != STRATEGY_ID for row in daily):
        raise ValueError("published daily ledger does not identify L2 V2")
    accepted = [row for row in setups if row.get("accepted") == "True"]
    entered = [row for row in accepted if row.get("confirmation_status") == "ENTRY"]
    expired = [row for row in accepted if row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED"]
    blocked = [row for row in accepted if row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION"]
    unresolved = [row for row in accepted if not row.get("terminal_reason")]
    trade_ids = {row["interaction_id"] for row in trades}
    entered_ids = {row["interaction_id"] for row in entered}
    if len(trades) != len(trade_ids) or trade_ids != entered_ids:
        raise ValueError("published V2 trade/setup ledgers do not reconcile one entered setup to one trade")
    if len(accepted) != len(entered) + len(expired) + len(blocked) + len(unresolved):
        raise ValueError("accepted V2 setup disposition counts do not reconcile")
    completed_interactions = sum(int(row["interactions_completed"]) for row in daily)
    if completed_interactions != len(setups):
        raise ValueError("published daily and setup interaction counts do not reconcile")
    if sum(int(row["accepted_setups"]) for row in daily) != len(accepted):
        raise ValueError("published daily and setup accepted counts do not reconcile")

    performance = _performance(trades)
    by_day = _breakdown(trades, "date")
    day_rows = [{"date": key, **value} for key, value in by_day.items()]
    best = max(day_rows, key=lambda row: (row["net_pnl_usd"], row["date"]), default=None)
    worst = min(day_rows, key=lambda row: (row["net_pnl_usd"], row["date"]), default=None)
    absolute_total = sum(abs(row["net_pnl_usd"]) for row in day_rows)
    largest_days = sorted(day_rows, key=lambda row: (-abs(row["net_pnl_usd"]), row["date"]))[:2]
    concentration = {
        "basis": "absolute_daily_net_pnl_usd",
        "largest_day_share": abs(largest_days[0]["net_pnl_usd"]) / absolute_total if largest_days and absolute_total else 0.0,
        "largest_two_day_share": sum(abs(row["net_pnl_usd"]) for row in largest_days) / absolute_total if absolute_total else 0.0,
        "dominated_by_one_or_two_days": (sum(abs(row["net_pnl_usd"]) for row in largest_days) / absolute_total >= .5) if absolute_total else False,
    }
    frozen_contract = dict(existing_summary.get("frozen_contract", {}))
    frozen_contract["strategy_id"] = STRATEGY_ID
    frozen_contract["evidence_label"] = EVIDENCE_LABEL
    return {
        "strategy_id": STRATEGY_ID,
        "variant_label": VARIANT_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "first_run_policy": "L2_V2_MAY_DEVELOPMENT_REPLAY; NOT_OOS_EVIDENCE; NO_OUTCOME_BASED_PARAMETER_SELECTION",
        "metadata_repair": {
            "prior_strategy_label": existing_summary.get("strategy_id"),
            "prior_first_run_policy": existing_summary.get("first_run_policy"),
            "underlying_ledgers_modified": False,
        },
        "frozen_contract": frozen_contract,
        "counts": {
            "completed_interactions": completed_interactions,
            "accepted_setups": len(accepted),
            "confirmations_passed": len(entered),
            "confirmations_failed_window_expired": len(expired),
            "compliance_blocks_active_position": len(blocked),
            "completed_trades": len(trades),
            "unresolved_trades": len(unresolved),
            "accepted_to_confirmed_conversion_rate": len(entered) / len(accepted) if accepted else 0.0,
        },
        "performance": performance,
        "breakdowns": {"day": by_day, "direction": _breakdown(trades, "direction"),
                       "structural_level": _breakdown(trades, "level"), "instrument": _breakdown(trades, "instrument")},
        "day_highlights": {
            "best_day": best, "worst_day": worst,
            "profitable_trading_days": sum(row["net_pnl_usd"] > 0 for row in day_rows),
            "losing_trading_days": sum(row["net_pnl_usd"] < 0 for row in day_rows),
            "flat_trading_days": sum(row["net_pnl_usd"] == 0 for row in day_rows),
            "concentration": concentration,
        },
        "reporting_scope": "PUBLISHED_LEDGER_AGGREGATION_ONLY",
        "pnl_optimization_performed": False,
    }


def _report(summary: dict[str, Any]) -> str:
    counts, performance, highlights = summary["counts"], summary["performance"], summary["day_highlights"]
    return "\n".join([
        "# CMEOrderflowAbsorption.ES_L2_V2 — May development replay", "",
        f"Variant: `{summary['variant_label']}`", f"Evidence: `{summary['evidence_label']}`", "",
        "This is May development evidence, not OOS evidence. This report is a ledger-only metadata and performance aggregation repair; no market data was replayed.", "",
        "## Counts", "",
        f"- Completed interactions: {counts['completed_interactions']}",
        f"- Accepted setups: {counts['accepted_setups']}",
        f"- Confirmations passed / window-expired / compliance-blocked: {counts['confirmations_passed']} / {counts['confirmations_failed_window_expired']} / {counts['compliance_blocks_active_position']}",
        f"- Completed / unresolved trades: {counts['completed_trades']} / {counts['unresolved_trades']}", "",
        "## Performance", "",
        f"- Wins/losses: {performance['wins']}/{performance['losses']} ({performance['win_rate']:.2%})",
        f"- Total R / average R / median R: {performance['total_r']:.6f} / {performance['average_r']:.6f} / {performance['median_r']:.6f}",
        f"- Net PnL: ${performance['net_pnl_usd']:.2f}; profit factor: {performance['profit_factor']}",
        f"- Maximum cumulative drawdown: {performance['max_cumulative_drawdown_r']:.6f} R; longest losing streak: {performance['longest_losing_streak']}",
        f"- ES/MES: {performance['es_trades']}/{performance['mes_trades']}; stop/target/cutoff exits: {performance['stop_exits']}/{performance['target_exits']}/{performance['cutoff_exits']}", "",
        "No strategy parameter, execution rule, or outcome-based selection was changed.", "",
    ])


def materialize(artifact_root: Path) -> dict[str, Any]:
    """Replace only stale derived report files using the immutable published ledgers."""
    setups = _read_csv(artifact_root / "setup-ledger.csv")
    trades = _read_csv(artifact_root / "trade-ledger.csv")
    daily = _read_csv(artifact_root / "daily-results.csv")
    existing = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    summary = build_summary(setups=setups, trades=trades, daily=daily, existing_summary=existing)
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_root / "diagnostic-report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ledger-only L2 V2 published replay reporting repair")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = materialize(args.artifact_root)
    except (FileNotFoundError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"counts": summary["counts"], "performance": summary["performance"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
