"""Frozen L2 V2 extension over explicitly owned local inputs.

The inventory path is deliberately filesystem-only: it does not import
Databento, open a DBN, or inspect a published strategy outcome.  The replay
path is an explicit manual command and reuses the frozen V2 configuration.
It has special evidence handling for sources that terminate at 16:00 UTC;
that boundary is *not* a replacement for the frozen 22:45 UTC hard cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from . import historical_runner as historical
from .v2_quality050 import STRATEGY_ID, V2_CONFIG, v2_contract_sha256


OUTPUT_NAME = "CMEOrderflowAbsorption.ES_L2_V2_EXTENDED_EXISTING_DATA"
MAY_LABEL = "MAY_DEVELOPMENT_V2_QUALITY_0_50_NOT_OOS_EVIDENCE"
RETRO_LABEL = "RETRO_JUNE_JULY_L2_V2_INCOMPLETE_TAIL_ROBUSTNESS"
AUGUST_LABEL = "SEEN_AUG_L2_V2_NOT_FRESH_OOS_EVIDENCE"
RETRO_DATES = (
    "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30",
    "2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
    "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17",
)
RETRO_PRIOR_RTH = {
    day: ("2026-06-22" if index == 0 else RETRO_DATES[index - 1])
    for index, day in enumerate(RETRO_DATES)
}
AUGUST_DATES = ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07")


def _path(root: Path, relative: str) -> str:
    return str((root / relative).resolve())


def _exists(root: Path, relative: str) -> bool:
    return (root / relative).is_file()


def _session_row(*, period: str, date: str, evidence_label: str, es_mbo: str | None,
                 es_start: str | None, es_end: str | None, profile: str | None,
                 mes: str | None, mes_start: str | None, mes_end: str | None,
                 status: str, reason: str | None, root: Path) -> dict[str, Any]:
    return {
        "period": period,
        "date": date,
        "evidence_label": evidence_label,
        "es_mbo_path": _path(root, es_mbo) if es_mbo else None,
        "es_mbo_path_exists": _exists(root, es_mbo) if es_mbo else False,
        "es_mbo_start_utc": es_start,
        "es_mbo_end_utc_exclusive": es_end,
        "snapshot_initialization_viability": "REPLAY_MUST_VERIFY_F_SNAPSHOT_THROUGH_F_LAST",
        "prior_rth_profile_path": _path(root, profile) if profile else None,
        "prior_rth_profile_available": _exists(root, profile) if profile else False,
        "mes_native_mbp1_path": _path(root, mes) if mes else None,
        "mes_native_available": _exists(root, mes) if mes else False,
        "mes_start_utc": mes_start,
        "mes_end_utc_exclusive": mes_end,
        "execution_coverage_end_utc_exclusive": es_end,
        "full_trade_replay_status": status,
        "limitation": reason,
    }


def build_inventory(repository_root: Path) -> dict[str, Any]:
    """Return the declared input inventory without opening a market-data file."""
    rows: list[dict[str, Any]] = []
    may_root = "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
    for day in historical.MAY_DATES:
        prior = historical.MAY_PRIOR_RTH[day]
        rows.append(_session_row(
            period="may", date=day, evidence_label=MAY_LABEL,
            es_mbo=f"{may_root}/es_mbo/ESM6_{day}_000000_224501_mbo.dbn.zst",
            es_start=f"{day}T00:00:00Z", es_end=f"{day}T22:45:01Z",
            profile=f"{may_root}/es_prior_rth_trades/ESM6_{prior}_133000_200000_trades.dbn.zst",
            mes=f"{may_root}/mes_mbp1/MESM6_{day}_133000_224501_mbp1.dbn.zst",
            mes_start=f"{day}T13:30:00Z", mes_end=f"{day}T22:45:01Z",
            status="FULLY_EXECUTABLE", reason="PUBLISHED_V2_REPLAY_REUSED_NO_RERUN", root=repository_root,
        ))
    retro_root = "data/cme_orderflow_absorption_v2_holdout"
    for day in RETRO_DATES:
        prior = RETRO_PRIOR_RTH[day]
        rows.append(_session_row(
            period="retro_june_july", date=day, evidence_label=RETRO_LABEL,
            es_mbo=f"{retro_root}/es_mbo/ESU6_{day}_0000_1600_mbo.dbn.zst",
            es_start=f"{day}T00:00:00Z", es_end=f"{day}T16:00:00Z",
            profile=f"{retro_root}/es_rth_trades/ESU6_{prior}_1330_2000_trades.dbn.zst",
            mes=f"{retro_root}/mes_mbp1/MESU6_{day}_1300_1600_mbp1.dbn.zst",
            mes_start=f"{day}T13:00:00Z", mes_end=f"{day}T16:00:00Z",
            status="PARTIAL_TAIL_ONLY", reason="SOURCE_ENDS_16_UTC_BEFORE_FROZEN_2245_CUTOFF", root=repository_root,
        ))
    august_mbo = "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
    for day in AUGUST_DATES:
        # The frozen L2 profile contract consumes declared prior-RTH ES trades.
        # An MBO file is not silently repurposed as a profile source here.
        rows.append(_session_row(
            period="august_seen", date=day, evidence_label=AUGUST_LABEL,
            es_mbo=august_mbo, es_start="2026-08-03T00:00:00Z", es_end="2026-08-08T00:00:00Z",
            profile=None, mes=None, mes_start=None, mes_end=None,
            status="MISSING_REQUIRED_CONTEXT",
            reason="NO_DECLARED_L2_PRIOR_RTH_ES_TRADES_AND_NO_NATIVE_MES_MBP1", root=repository_root,
        ))
    for row in rows:
        if row["full_trade_replay_status"] == "FULLY_EXECUTABLE" and not (
            row["es_mbo_path_exists"] and row["prior_rth_profile_available"] and row["mes_native_available"]
        ):
            row["full_trade_replay_status"] = "UNUSABLE"
            row["limitation"] = "DECLARED_MAY_INPUT_MISSING"
        elif row["full_trade_replay_status"] == "PARTIAL_TAIL_ONLY" and not (
            row["es_mbo_path_exists"] and row["prior_rth_profile_available"] and row["mes_native_available"]
        ):
            row["full_trade_replay_status"] = "MISSING_REQUIRED_CONTEXT"
            row["limitation"] = "DECLARED_RETRO_INPUT_MISSING"
    return {
        "strategy_id": STRATEGY_ID,
        "variant_label": "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50",
        "v2_contract_sha256": v2_contract_sha256(),
        "inventory_scope": "FILESYSTEM_AND_DECLARED_ARTIFACT_PATHS_ONLY_NO_DBN_OPEN_NO_STRATEGY_OUTCOME_READ",
        "session_count": len(rows),
        "sessions": rows,
    }


def materialize_inventory(repository_root: Path, output_root: Path) -> dict[str, Any]:
    """Create the immutable availability audit and empty period directories."""
    if output_root.exists():
        raise FileExistsError(f"extended L2 V2 output already exists: {output_root}")
    inventory = build_inventory(repository_root)
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("may", "retro_june_july", "august_seen", "combined_descriptive"):
        (output_root / name).mkdir()
    (output_root / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Frozen L2 V2 existing-data availability audit", "", "This audit opened no DBN and read no strategy outcomes.", ""]
    for period in ("may", "retro_june_july", "august_seen"):
        items = [row for row in inventory["sessions"] if row["period"] == period]
        states: dict[str, int] = {}
        for row in items:
            states[row["full_trade_replay_status"]] = states.get(row["full_trade_replay_status"], 0) + 1
        lines.append(f"- `{period}`: {len(items)} sessions; {states}")
    lines.extend(["", "May is reused from its published V2 replay. Retro sessions are source-tail incomplete; a future replay must never force-flat at 16:00 UTC. August has no declared L2 prior-RTH trade profile inputs and no native MES feed, so it is not replayed under this frozen contract.", ""])
    (output_root / "inventory-report.md").write_text("\n".join(lines), encoding="utf-8")
    return inventory


def _run_retro_session(day: str, repository_root: Path) -> historical.HistoricalL2Runner:
    """Replay a declared 16:00-ended source without manufacturing a session exit."""
    root = repository_root / "data/cme_orderflow_absorption_v2_holdout"
    profile = root / "es_rth_trades" / f"ESU6_{RETRO_PRIOR_RTH[day]}_1330_2000_trades.dbn.zst"
    es_path = root / "es_mbo" / f"ESU6_{day}_0000_1600_mbo.dbn.zst"
    mes_path = root / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst"
    levels = historical._profile_levels_from_declared_trades(profile)
    runner = historical.HistoricalL2Runner(
        date=day, evidence_label=RETRO_LABEL, levels=levels, config=V2_CONFIG,
        strategy_id=STRATEGY_ID, require_native_mes_for_fallback=True,
    )
    adapter = historical.HistoricalMBOToMBP10Adapter()
    es_iter, mes_iter = iter(historical._stream_private_mbo(es_path)), iter(historical._stream_mes_quotes(mes_path))
    es, mes, records = historical._next(es_iter), historical._next(mes_iter), 0
    start_ns, source_end_ns = historical._clock_ns(day, historical.RTH_START_SECONDS), historical._clock_ns(day, 16 * 3600)
    while es is not None or mes is not None:
        es_ts = es.timestamp_ns if es is not None else 2**63 - 1
        mes_ts = mes[0] if mes is not None else 2**63 - 1
        if mes_ts < es_ts:
            if mes_ts >= start_ns:
                runner.observe_mes_quote(*mes)
            mes = historical._next(mes_iter)
            continue
        record, es = es, historical._next(es_iter)
        records += 1
        public = adapter.feed(record, materialize_public=record.timestamp_ns >= start_ns)
        if public is not None and public.timestamp_ns >= start_ns:
            runner.observe_public(public)
        if records % 5_000_000 == 0:
            print(f"  {day} records={records:,} completed={len(runner.interaction_ledger):,} accepted={sum(bool(row['accepted']) for row in runner.setup_ledger):,}", flush=True)
    adapter.finish()
    runner.source_integrity_diagnostics = adapter.source_integrity_diagnostics()
    runner.mark_source_end_incomplete(source_end_ns)
    return runner


def _as_rows(runners: Iterable[historical.HistoricalL2Runner], attribute: str) -> list[dict[str, Any]]:
    return [row for runner in runners for row in getattr(runner, attribute)]


def _subset(setups: list[dict[str, Any]], trades: list[dict[str, Any]], *, levels: set[str]) -> dict[str, Any]:
    selected = [row for row in setups if row.get("level") in levels]
    realized = [row for row in trades if row.get("level") in levels]
    performance = historical._performance(realized)
    return {
        "methodology": "POST_RUN_SUBSET_OF_THE_SAME_ALL_LEVEL_POSITION_STATE_NOT_A_RERUN",
        "levels": sorted(levels),
        "accepted_setups": sum(str(row.get("accepted")) == "True" or row.get("accepted") is True for row in selected),
        "confirmed_setups": sum(row.get("confirmation_timestamp_ns") not in (None, "") for row in selected),
        "completed_trades": performance["completed_trades"], "wins": performance["wins"], "losses": performance["losses"],
        "total_r": performance["total_r"], "average_r": performance["average_r"], "profit_factor": performance["profit_factor"],
        "es_trades": performance["es_trades"], "mes_trades": performance["mes_trades"],
        "unresolved": sum(str(row.get("terminal_reason", "")).endswith("SOURCE_INCOMPLETE") or row.get("terminal_reason") == "UNRESOLVED_SOURCE_END" for row in selected),
    }


def _write_period(output: Path, runners: list[historical.HistoricalL2Runner], *, label: str) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable extended period output already exists: {output}")
    for runner in runners:
        runner.refresh_setup_ledger()
    interactions, setups, trades = _as_rows(runners, "interaction_ledger"), _as_rows(runners, "setup_ledger"), _as_rows(runners, "trade_ledger")
    unresolved = _as_rows(runners, "source_end_unresolved")
    mes_unavailable = _as_rows(runners, "mes_execution_unavailable")
    performance = historical._performance(trades)
    metrics = {
        "raw_interactions": len(interactions), "completed_interactions": len(interactions),
        "accepted_setups": sum(bool(row["accepted"]) for row in setups), "rejected_setups": sum(not bool(row["accepted"]) for row in setups),
        "confirmations_passed": sum(row.get("confirmation_timestamp_ns") is not None for row in setups),
        "confirmation_expiries": sum(row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED" for row in setups),
        "active_position_blocks": sum(row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for row in setups),
        "unresolved_source_end_trades": sum(row.get("reason") == "UNRESOLVED_SOURCE_END" for row in unresolved),
        "execution_unresolved_source_incomplete": sum(row.get("reason") == "EXECUTION_UNRESOLVED_SOURCE_INCOMPLETE" for row in unresolved),
        "mes_execution_unavailable": len(mes_unavailable),
        "result_completeness": "INCOMPLETE" if unresolved else "COMPLETE",
        **performance,
    }
    result = {
        "strategy_id": STRATEGY_ID, "variant_label": "L2_V2_MAY_DEVELOPMENT_QUALITY_0_50", "evidence_label": label,
        "v2_contract_sha256": v2_contract_sha256(), "metrics": metrics,
        "poc_val_descriptive_subset": {
            "PRIOR_RTH_POC": _subset(setups, trades, levels={"PRIOR_RTH_POC"}),
            "PRIOR_RTH_VAL": _subset(setups, trades, levels={"PRIOR_RTH_VAL"}),
            "POC_PLUS_VAL": _subset(setups, trades, levels={"PRIOR_RTH_POC", "PRIOR_RTH_VAL"}),
        },
        "source_end_policy": "NO_1600_FORCED_EXIT_NO_PRICE_EXTRAPOLATION" if unresolved else "NOT_APPLICABLE",
    }
    output.mkdir(parents=True, exist_ok=True)
    historical._rows_write(output / "interaction-features.csv", interactions, ["interaction_id"])
    historical._rows_write(output / "setup-ledger.csv", setups, ["setup_id"])
    historical._rows_write(output / "trade-ledger.csv", trades, ["trade_id"])
    historical._rows_write(output / "unresolved-source-end.csv", unresolved, ["setup_id"])
    historical._rows_write(output / "mes-execution-unavailable.csv", mes_unavailable, ["setup_id"])
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "diagnostic-report.md").write_text(
        f"# {STRATEGY_ID} — {label}\n\nResult completeness: `{metrics['result_completeness']}`. "
        "This is the unchanged all-level strategy; POC/VAL is a post-run subset only.\n",
        encoding="utf-8",
    )
    return result


def _published_may_reference(repository_root: Path) -> dict[str, Any]:
    """Read only published May artifacts; never invoke the May replay."""
    path = repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V2_MAY_DEVELOPMENT_REPLAY/summary.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("strategy_id") != STRATEGY_ID or source.get("evidence_label") != MAY_LABEL:
        raise RuntimeError("published May artifact does not bind frozen L2 V2 evidence")
    levels = source.get("breakdowns", {}).get("structural_level", {})
    return {
        "strategy_id": STRATEGY_ID, "evidence_label": MAY_LABEL, "reused_without_replay": True,
        "metrics": {**source["counts"], **source["performance"], "result_completeness": "COMPLETE"},
        "poc_val_descriptive_subset": {
            name: levels.get(name, {}) for name in ("PRIOR_RTH_POC", "PRIOR_RTH_VAL")
        },
    }


def _cross_period_row(name: str, result: dict[str, Any] | None, *, status: str | None = None) -> dict[str, Any]:
    if result is None:
        return {"period": name, "all_level_trades": None, "all_level_total_r": None, "all_level_pf": None,
                "poc_val_trades": None, "poc_val_total_r": None, "poc_val_pf": None,
                "unresolved_count": None, "mes_unavailable_count": None, "completeness": status}
    metrics = result["metrics"]
    subset = result.get("poc_val_descriptive_subset", {}).get("POC_PLUS_VAL", {})
    if not subset:  # Published May preserves POC and VAL separately.
        poc = result.get("poc_val_descriptive_subset", {}).get("PRIOR_RTH_POC", {})
        val = result.get("poc_val_descriptive_subset", {}).get("PRIOR_RTH_VAL", {})
        subset = {
            "completed_trades": int(poc.get("completed_trades", 0)) + int(val.get("completed_trades", 0)),
            "total_r": float(poc.get("total_r", 0.0)) + float(val.get("total_r", 0.0)),
            "profit_factor": None,
        }
    return {"period": name, "all_level_trades": metrics.get("completed_trades"), "all_level_total_r": metrics.get("total_r"),
            "all_level_pf": metrics.get("profit_factor"), "poc_val_trades": subset.get("completed_trades"),
            "poc_val_total_r": subset.get("total_r"), "poc_val_pf": subset.get("profit_factor"),
            "unresolved_count": metrics.get("unresolved_source_end_trades", metrics.get("unresolved_trades", 0)),
            "mes_unavailable_count": metrics.get("mes_execution_unavailable", 0),
            "completeness": metrics.get("result_completeness", "COMPLETE")}


def replay_owned_usable_periods(repository_root: Path, output_root: Path) -> dict[str, Any]:
    """Explicit manual replay; never invoked by the inventory command."""
    inventory_path = output_root / "inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError("run --inventory first to bind the declared owned-input inventory")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("v2_contract_sha256") != v2_contract_sha256():
        raise RuntimeError("frozen V2 contract hash changed after inventory; refusing replay")
    retro = [row for row in inventory["sessions"] if row["period"] == "retro_june_july"]
    if any(row["full_trade_replay_status"] != "PARTIAL_TAIL_ONLY" for row in retro):
        raise RuntimeError("retro replay requires every declared source-tail session to remain usable")
    runners: list[historical.HistoricalL2Runner] = []
    for index, day in enumerate(RETRO_DATES, start=1):
        print(f"=== L2 V2 RETRO {index:02d}/{len(RETRO_DATES)} {day} ===", flush=True)
        runners.append(_run_retro_session(day, repository_root))
    retro_result = _write_period(output_root / "retro_june_july", runners, label=RETRO_LABEL)
    may_result = _published_may_reference(repository_root)
    (output_root / "may" / "reused-published-summary.json").write_text(
        json.dumps(may_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    august_status = {
        "strategy_id": STRATEGY_ID, "evidence_label": AUGUST_LABEL, "replayed": False,
        "status": "MISSING_REQUIRED_CONTEXT",
        "reason": "NO_DECLARED_L2_PRIOR_RTH_ES_TRADES_AND_NO_NATIVE_MES_MBP1",
    }
    (output_root / "august_seen" / "not-replayed.json").write_text(
        json.dumps(august_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = [
        _cross_period_row("MAY", may_result),
        _cross_period_row("RETRO_JUNE_JULY", retro_result),
        _cross_period_row("AUGUST_SEEN", None, status="NOT_REPLAYED_MISSING_REQUIRED_CONTEXT"),
    ]
    combined = {
        "strategy_id": STRATEGY_ID,
        "combined_descriptive_only": True,
        "periods": {"may": may_result, "retro_june_july": retro_result, "august_seen": august_status},
        "cross_period_table": table,
        "warnings": ["MAY is reused and not rerun.", "Retro source tails are incomplete.", "August is not replayed because required L2 profile context and native MES execution inputs are absent."],
    }
    (output_root / "combined_descriptive" / "summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "combined_descriptive" / "cross-period-table.csv").write_text(
        "\n".join([",".join(table[0].keys())] + [",".join("" if row[key] is None else str(row[key]) for key in table[0]) for row in table]) + "\n",
        encoding="utf-8",
    )
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen L2 V2 existing-data inventory and explicit manual replay")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true")
    mode.add_argument("--replay", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_inventory(args.repository_root, args.output_root) if args.inventory else replay_owned_usable_periods(args.repository_root, args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (FileExistsError, FileNotFoundError, RuntimeError, historical.HistoricalReplayError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
