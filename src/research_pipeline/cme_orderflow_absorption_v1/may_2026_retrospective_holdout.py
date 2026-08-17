"""Sealed two-strategy May-2026 retrospective holdout runner.

This runner downloads nothing.  It validates the immutable acquisition manifest,
streams each declared local DBN once, and evaluates only the predeclared frozen
V2 baseline and V3 +3-tick/3R candidate against the shared causal L3 stream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterator

from . import v2_retro_holdout_runner as v2
from . import v3_tick_trigger_target_matrix as v3
from .analysis import Diagnostics, RTH_END, RTH_START, day_and_seconds, volume_profile
from .engine import BookStateError, CausalMBOBook
from .oos_backtest_runner import load_development_calibration


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
DEFAULT_OUTPUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_MAY_2026_RETROSPECTIVE_HOLDOUT"
MANIFEST_NAME = "acquisition-manifest.json"
EVIDENCE_LABEL = "UNSEEN_MAY_2026_RETROSPECTIVE_HOLDOUT"
STRATEGY_V2 = "CMEOrderflowAbsorption.ES_V2_BASELINE"
STRATEGY_V3 = "CMEOrderflowAbsorption.ES_V3_TICK_3_TARGET_3R"
TARGET_DAYS = (
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
)
PRIOR_RTH = {day: ("2026-05-01" if day == TARGET_DAYS[0] else TARGET_DAYS[index - 1])
             for index, day in enumerate(TARGET_DAYS)}
RAW_PRICE_SCALE = v2.RAW_PRICE_SCALE
TICK = v2.TICK
ENTRY_LATENCY_NS = v2.ENTRY_LATENCY_NS
SIGNAL_CUTOFF_SECONDS = RTH_END
HARD_CUTOFF_CLOCK = "22:45:00"


class MayHoldoutError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _cutoff_ns(day: str) -> int:
    return int(datetime.fromisoformat(f"{day}T{HARD_CUTOFF_CLOCK}+00:00").timestamp() * 1_000_000_000)


def _expected_relative_paths(day: str) -> dict[str, str]:
    prior = PRIOR_RTH[day]
    return {
        "ES_MBO_L3": f"es_mbo/ESM6_{day}_000000_224501_mbo.dbn.zst",
        "MES_NATIVE_EXECUTION": f"mes_mbp1/MESM6_{day}_133000_224501_mbp1.dbn.zst",
        "ES_PRIOR_RTH_PROFILE": f"es_prior_rth_trades/ESM6_{prior}_133000_200000_trades.dbn.zst",
    }


def verify_manifest(data_root: Path) -> dict[str, Any]:
    """Hash-verify exactly the sealed 15 × 3 source files before replay."""
    manifest_path = data_root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MayHoldoutError("missing or unreadable May acquisition manifest") from exc
    if manifest.get("manifest_kind") != "MAY_2026_ES_MES_COST_PROXY_ACQUISITION" or manifest.get("data_acquired") is not True:
        raise MayHoldoutError("May acquisition manifest is not sealed/acquired")
    identity = manifest.get("request_identity", {})
    if identity.get("target_rth_dates") != list(TARGET_DAYS):
        raise MayHoldoutError("May target RTH dates differ from the approved 15-session set")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 45:
        raise MayHoldoutError("May manifest must contain exactly 45 files")
    labels = Counter(str(item.get("label")) for item in files.values())
    if labels != Counter({"ES_MBO_L3": 15, "MES_NATIVE_EXECUTION": 15, "ES_PRIOR_RTH_PROFILE": 15}):
        raise MayHoldoutError("May manifest does not contain the required 15/15/15 package")
    expected: dict[str, dict[str, str]] = {}
    for day in TARGET_DAYS:
        for label, relative in _expected_relative_paths(day).items():
            expected[relative] = {"label": label, "session_date": day if label != "ES_PRIOR_RTH_PROFILE" else PRIOR_RTH[day]}
    if set(files) != set(expected):
        raise MayHoldoutError("May manifest has missing, extra, or unapproved file paths")
    verified: list[dict[str, Any]] = []
    for relative, expected_fields in sorted(expected.items()):
        record = files[relative]
        path = (data_root / relative).resolve()
        if data_root.resolve() not in path.parents or not path.is_file():
            raise MayHoldoutError(f"missing sealed May input: {relative}")
        if record.get("label") != expected_fields["label"] or record.get("session_date") != expected_fields["session_date"]:
            raise MayHoldoutError(f"manifest identity mismatch: {relative}")
        if record.get("schema") not in {"mbo", "mbp-1", "trades"} or record.get("bytes") != path.stat().st_size:
            raise MayHoldoutError(f"manifest size/schema mismatch: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != record.get("sha256"):
            raise MayHoldoutError(f"manifest SHA-256 mismatch: {relative}")
        verified.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": actual_hash, **expected_fields})
    return {"manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "verified_files": verified}


def _stream(path: Path) -> Iterator[object]:
    from databento import DBNStore
    yield from DBNStore.from_file(path)


def _next(iterator: Iterator[object]) -> object | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _profile(path: Path) -> dict[str, int]:
    rows = [(int(record.price), int(record.size)) for record in _stream(path)
            if int(getattr(record, "price", 0)) > 0 and int(getattr(record, "size", 0)) > 0]
    profile = volume_profile(rows)
    if profile is None:
        raise MayHoldoutError("missing prior-RTH volume profile")
    return {"PRIOR_RTH_HIGH": profile["high"], "PRIOR_RTH_LOW": profile["low"],
            "PRIOR_RTH_POC": profile["poc"], "PRIOR_RTH_VAH": profile["vah"], "PRIOR_RTH_VAL": profile["val"]}


def _rth(timestamp_ns: int) -> bool:
    _, seconds = day_and_seconds(timestamp_ns)
    return RTH_START <= seconds < SIGNAL_CUTOFF_SECONDS


def _close_v3_at_cutoff(cell: v3.Cell, timestamp: int, quote: tuple[float, float] | None) -> None:
    position = cell.position
    if position is None or quote is None:
        return
    long = position.pending.row["direction"] == "BUYER_ABSORPTION"
    reference = quote[0] if long else quote[1]
    exit_fill = reference - TICK if long else reference + TICK
    points = exit_fill - position.entry if long else position.entry - exit_fill
    point_value = v2.ES_POINT_VALUE if position.instrument == "ES" else v2.MES_POINT_VALUE
    commission = v2.ES_COMMISSION if position.instrument == "ES" else v2.MES_COMMISSION
    gross = points * point_value * position.contracts
    fees = position.entry_commission + commission * position.contracts
    total_risk = position.one_contract_risk * position.contracts
    cell.trades.append({
        "period": cell.period, "cell_id": cell.spec.cell_id, "trigger_ticks": cell.spec.trigger_ticks,
        "target_r": cell.spec.target_r, "date": position.pending.row["date"],
        "interaction_id": position.pending.row["interaction_id"], "direction": position.pending.row["direction"],
        "level": position.pending.row["level"], "interaction_end": position.pending.row["interaction_end"],
        "trigger_timestamp": position.pending.trigger_timestamp, "trigger_price": position.pending.trigger_price,
        "trigger_favorable_ticks": position.pending.trigger_favorable_ticks, "entry_timestamp": position.entry_timestamp,
        "entry": position.entry, "stop": position.stop, "target": position.target, "instrument": position.instrument,
        "contracts": position.contracts, "exit_timestamp": timestamp, "exit": exit_fill,
        "exit_reason": "CUTOFF_FORCED_FLAT", "gross_usd": gross, "commission_usd": fees,
        "net_usd": gross - fees, "r_multiple": (gross - fees) / total_risk,
        "entry_displacement_ticks_from_interaction_end": ((position.entry - float(position.pending.row["end_price"]) / RAW_PRICE_SCALE) / TICK if long else (float(position.pending.row["end_price"]) / RAW_PRICE_SCALE - position.entry) / TICK),
        "stop_distance_ticks": abs(position.entry - position.stop) / TICK,
        "seconds_interaction_end_to_trigger": ((position.pending.trigger_timestamp - int(position.pending.row["interaction_end"])) / RAW_PRICE_SCALE if position.pending.trigger_timestamp is not None else None),
        "seconds_trigger_to_entry": ((position.entry_timestamp - position.pending.trigger_timestamp) / RAW_PRICE_SCALE if position.pending.trigger_timestamp is not None else None),
    })
    cell._audit(position.pending, "TRADE_CLOSED", exit_reason="CUTOFF_FORCED_FLAT")
    cell.position = None


def _close_v2_at_cutoff(state: v2.DayState, timestamp: int, quote: tuple[float, float] | None) -> None:
    position = state.position
    if position is None or quote is None:
        return
    reference = quote[0] if position.signal.row["direction"] == "BUYER_ABSORPTION" else quote[1]
    v2._close_position(state, position, timestamp, reference, "CUTOFF_FORCED_FLAT")


def _paths(data_root: Path, day: str) -> tuple[Path, Path, Path]:
    relative = _expected_relative_paths(day)
    return data_root / relative["ES_MBO_L3"], data_root / relative["MES_NATIVE_EXECUTION"], data_root / relative["ES_PRIOR_RTH_PROFILE"]


def _enrich_v2_trades(state: v2.DayState, confirmation: dict[str, dict[str, float | int | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in state.trades:
        signal = state.pending[trade["interaction_id"]].row
        end_price = float(signal["end_price"]) / RAW_PRICE_SCALE
        long = trade["direction"] == "BUYER_ABSORPTION"
        meta = confirmation.get(trade["interaction_id"], {})
        rows.append({**trade, "strategy_id": STRATEGY_V2, "target_r": 2.5, "interaction_end": signal["interaction_end"],
                     "confirmation_timestamp": meta.get("timestamp"), "confirmation_price": meta.get("price"),
                     "confirmation_favorable_ticks": meta.get("favorable_ticks"),
                     "entry_displacement_ticks_from_interaction_end": ((float(trade["entry"]) - end_price) / TICK if long else (end_price - float(trade["entry"])) / TICK),
                     "stop_distance_ticks": abs(float(trade["entry"]) - float(trade["stop"])) / TICK})
    return rows


def _enrich_v3_trades(cell: v3.Cell) -> list[dict[str, Any]]:
    return [{**trade, "strategy_id": STRATEGY_V3, "confirmation_timestamp": trade.get("trigger_timestamp"),
             "confirmation_price": trade.get("trigger_price"), "confirmation_favorable_ticks": trade.get("trigger_favorable_ticks")}
            for trade in cell.trades]


def _replay_day(day: str, data_root: Path, calibration: dict[str, Any], contract: dict[str, Any]) -> tuple[v2.DayState, v3.Cell, dict[str, dict[str, float | int | None]]]:
    es_path, mes_path, prior_path = _paths(data_root, day)
    diagnostics = Diagnostics(); diagnostics.levels[day] = _profile(prior_path)
    v2_state = v2.DayState(day=day, prior_day=PRIOR_RTH[day], diagnostics=diagnostics)
    v3_cell = v3.Cell(v3.MatrixSpec(3, 3.0), EVIDENCE_LABEL, "NATIVE_MES_MBP1_FALLBACK")
    confirmation: dict[str, dict[str, float | int | None]] = {}
    book = CausalMBOBook(); es_iter, mes_iter = iter(_stream(es_path)), iter(_stream(mes_path))
    es, mes = _next(es_iter), _next(mes_iter)
    es_quote: tuple[float, float] | None = None; mes_quote: tuple[float, float] | None = None
    cutoff_done = False; records = 0
    while es is not None or mes is not None:
        es_ts = int(getattr(es, "ts_recv", getattr(es, "ts_event", 2**63 - 1))) if es is not None else 2**63 - 1
        mes_ts = int(getattr(mes, "ts_event", 2**63 - 1)) if mes is not None else 2**63 - 1
        timestamp = min(es_ts, mes_ts)
        if not cutoff_done and timestamp >= _cutoff_ns(day):
            _close_v2_at_cutoff(v2_state, _cutoff_ns(day), es_quote if v2_state.position and v2_state.position.instrument == "ES" else mes_quote)
            _close_v3_at_cutoff(v3_cell, _cutoff_ns(day), es_quote if v3_cell.position and v3_cell.position.instrument == "ES" else mes_quote)
            cutoff_done = True
        if mes_ts < es_ts:
            mes_quote = v2._valid_mes(mes)
            if mes_quote is not None:
                v2._manage_mes(v2_state, mes_ts, *mes_quote); v3_cell.manage(mes_ts, *mes_quote)
            if _rth(mes_ts):
                v2._enter_mes(v2_state, mes_ts, mes_quote); v3_cell.expire(mes_ts); v3_cell.try_retro_mes_entry(mes_ts, mes_quote)
            mes = _next(mes_iter)
            continue
        record, es = es, _next(es_iter); records += 1
        action = v2._code(getattr(record, "action", "N"))
        if action in {"N", "NONE"}:
            continue
        applied = book.apply(action=action, side=v2._code(getattr(record, "side", "")), price=int(record.price), size=int(record.size),
                             order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv),
                             channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
        event_ts = int(record.ts_event); es_quote = v2._valid_es(book)
        if es_quote is not None:
            v2._manage_es(v2_state, event_ts, *es_quote); v3_cell.manage(event_ts, *es_quote)
        if _rth(int(record.ts_recv)):
            diagnostics.observe(record, applied, book.spread())
            prior_plus = len(v2_state.plus)
            if applied and applied.executed:
                v2._new_completed(v2_state, calibration, contract)
                new_plus = v2_state.plus[prior_plus:]
                if new_plus:
                    v3_cell.add_signals([{**row, "date": day} for row in new_plus])
                execution_price = v2.raw_price_to_points(record.price)
                before = {identifier: pending.state for identifier, pending in v2_state.pending.items()}
                v2._confirm_and_enter_es(v2_state, event_ts, execution_price, es_quote)
                for identifier, pending in v2_state.pending.items():
                    if before.get(identifier) == "AWAITING_CONFIRMATION" and pending.state == "AWAITING_ES_ENTRY":
                        end = v2.raw_price_to_points(pending.row["end_price"])
                        favorable = (execution_price - end) / TICK
                        if pending.row["direction"] == "SELLER_ABSORPTION": favorable = -favorable
                        confirmation[identifier] = {"timestamp": event_ts, "price": execution_price, "favorable_ticks": favorable}
                v3_cell.observe_execution(event_ts, int(record.price))
            v3_cell.expire(event_ts)
            v3_cell.try_retro_es_entry(event_ts, es_quote)
        if records % 5_000_000 == 0:
            print(f"  {day} records={records:,} interactions={len(v2_state.interactions):,} plus={len(v2_state.plus):,}", flush=True)
    diagnostics.finalize()
    prior_plus = len(v2_state.plus)
    v2._new_completed(v2_state, calibration, contract)
    if v2_state.plus[prior_plus:]:
        v3_cell.add_signals([{**row, "date": day} for row in v2_state.plus[prior_plus:]])
    if not cutoff_done:
        _close_v2_at_cutoff(v2_state, _cutoff_ns(day), es_quote if v2_state.position and v2_state.position.instrument == "ES" else mes_quote)
        _close_v3_at_cutoff(v3_cell, _cutoff_ns(day), es_quote if v3_cell.position and v3_cell.position.instrument == "ES" else mes_quote)
    return v2_state, v3_cell, confirmation


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "median": float(median(values)) if values else None}


def _performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: (int(row["exit_timestamp"]), str(row["interaction_id"])))
    rs = [float(row["r_multiple"]) for row in ordered]; net = [float(row["net_usd"]) for row in ordered]
    wins = sum(value > 0 for value in net); losses = sum(value < 0 for value in net)
    profit, loss = sum(value for value in net if value > 0), abs(sum(value for value in net if value < 0))
    equity = peak = drawdown = 0.0
    for value in rs:
        equity += value; peak = max(peak, equity); drawdown = min(drawdown, equity - peak)
    return {"completed_trades": len(trades), "wins": wins, "losses": losses, "win_rate": wins / len(trades) if trades else 0.0,
            "total_r": sum(rs), "average_r": sum(rs) / len(rs) if rs else 0.0, "median_r": float(median(rs)) if rs else None,
            "net_pnl": sum(net), "profit_factor": profit / loss if loss else None, "max_cumulative_drawdown_r": drawdown,
            "es_trades": sum(row["instrument"] == "ES" for row in trades), "mes_trades": sum(row["instrument"] == "MES" for row in trades),
            "target_exits": sum(row["exit_reason"] == "TARGET" for row in trades), "stop_exits": sum(row["exit_reason"] == "STOP" for row in trades),
            "hard_cutoff_exits": sum(row["exit_reason"] == "CUTOFF_FORCED_FLAT" for row in trades),
            "confirmation_seconds": _stats([(int(row["confirmation_timestamp"]) - int(row["interaction_end"])) / RAW_PRICE_SCALE for row in trades if row.get("confirmation_timestamp") is not None]),
            "entry_displacement_ticks": _stats([float(row["entry_displacement_ticks_from_interaction_end"]) for row in trades]),
            "stop_distance_ticks": _stats([float(row["stop_distance_ticks"]) for row in trades])}


def _breakdown(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades: groups[str(trade[field])].append(trade)
    return {key: {"trades": len(rows), "total_r": sum(float(row["r_multiple"]) for row in rows),
                  "net_pnl": sum(float(row["net_usd"]) for row in rows)} for key, rows in sorted(groups.items())}


def _strategy_summary(*, strategy_id: str, raw: int, plus: int, passed: int, failed: int,
                      audit: list[dict[str, Any]], trades: list[dict[str, Any]], unresolved: list[str]) -> dict[str, Any]:
    return {"strategy_id": strategy_id, "evidence_label": EVIDENCE_LABEL, "raw_interactions": raw,
            "v1_plus_setups": plus, "confirmations_passed": passed, "confirmations_failed": failed,
            "blocked_setups": sum(row.get("outcome") == "POSITION_ALREADY_OPEN" for row in audit),
            "unresolved_trades": len(unresolved), "unresolved_interaction_ids": unresolved,
            "breakdown": {field: _breakdown(trades, field) for field in ("date", "direction", "level", "instrument")},
            **_performance(trades)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))}) or fallback
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key in fields} for row in rows])


def _report(v2_summary: dict[str, Any], v3_summary: dict[str, Any]) -> str:
    def answer() -> str:
        return ("Neither" if v2_summary["net_pnl"] <= 0 and v3_summary["net_pnl"] <= 0 else
                "V2 baseline" if v2_summary["net_pnl"] > 0 and v3_summary["net_pnl"] <= 0 else
                "V3 candidate" if v3_summary["net_pnl"] > 0 and v2_summary["net_pnl"] <= 0 else "Both")
    return "\n".join([
        "# May 2026 frozen L3 retrospective holdout", "",
        f"Evidence label: `{EVIDENCE_LABEL}`. This is unseen May data but not strict chronological OOS because frozen L3 calibration used later July 2026 development data.",
        "", "No parameter, threshold, score, PLUS, level, lifecycle, sizing, or execution change was selected from this result.", "",
        "| Strategy | Trades | Total R | Net PnL | Win rate | Max drawdown R |", "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| V2 fixed-15s +1 / 2.5R | {v2_summary['completed_trades']} | {v2_summary['total_r']:.4f} | {v2_summary['net_pnl']:.2f} | {v2_summary['win_rate']:.2%} | {v2_summary['max_cumulative_drawdown_r']:.4f} |",
        f"| V3 immediate +3 / 3R | {v3_summary['completed_trades']} | {v3_summary['total_r']:.4f} | {v3_summary['net_pnl']:.2f} | {v3_summary['win_rate']:.2%} | {v3_summary['max_cumulative_drawdown_r']:.4f} |",
        "", f"Profitability: {answer()} is profitable by net PnL over this 15-session retrospective holdout.",
        f"Higher total R: {'V2' if v2_summary['total_r'] > v3_summary['total_r'] else 'V3' if v3_summary['total_r'] > v2_summary['total_r'] else 'tie'}.",
        f"Lower drawdown: {'V2' if v2_summary['max_cumulative_drawdown_r'] > v3_summary['max_cumulative_drawdown_r'] else 'V3' if v3_summary['max_cumulative_drawdown_r'] > v2_summary['max_cumulative_drawdown_r'] else 'tie'}.",
        "Direction, structural-level, daily concentration, and independent-trade counts are descriptive breakdowns in `summary.json`; they do not establish a new rule.", "",
    ])


def run(*, data_root: Path = DATA_ROOT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists(): raise MayHoldoutError("immutable May holdout output directory already exists")
    source = verify_manifest(data_root)
    calibration = load_development_calibration()
    contract = json.loads(v2.CONTRACT.read_text(encoding="utf-8"))
    all_v2_trades: list[dict[str, Any]] = []; all_v3_trades: list[dict[str, Any]] = []
    v2_audit: list[dict[str, Any]] = []; v3_audit: list[dict[str, Any]] = []; daily: list[dict[str, Any]] = []
    raw = plus = v2_passed = v2_failed = v3_passed = v3_failed = 0; v2_unresolved: list[str] = []; v3_unresolved: list[str] = []
    for index, day in enumerate(TARGET_DAYS, 1):
        print(f"=== MAY DAY {index:02d}/15 {day} ===", flush=True)
        state, cell, confirmations = _replay_day(day, data_root, calibration, contract)
        state_v2_trades = _enrich_v2_trades(state, confirmations); state_v3_trades = _enrich_v3_trades(cell)
        tail_v2, tail_v3 = v2._tail(state), cell.tail()
        raw += len(state.interactions); plus += len(state.plus); v2_passed += state.confirmations_passed; v2_failed += state.confirmations_failed
        v3_passed += cell.confirmations_passed; v3_failed += cell.confirmations_failed
        all_v2_trades.extend(state_v2_trades); all_v3_trades.extend(state_v3_trades); v2_audit.extend(state.audit); v3_audit.extend(cell.audit)
        v2_unresolved.extend(tail_v2); v3_unresolved.extend(tail_v3)
        for strategy, trades, passed, failed, tail in ((STRATEGY_V2, state_v2_trades, state.confirmations_passed, state.confirmations_failed, tail_v2), (STRATEGY_V3, state_v3_trades, cell.confirmations_passed, cell.confirmations_failed, tail_v3)):
            daily.append({"date": day, "previous_completed_rth": PRIOR_RTH[day], "strategy_id": strategy,
                          "raw_interactions": len(state.interactions), "v1_plus_setups": len(state.plus),
                          "confirmations_passed": passed, "confirmations_failed": failed, "completed_trades": len(trades),
                          "total_r": sum(float(row["r_multiple"]) for row in trades), "net_pnl": sum(float(row["net_usd"]) for row in trades),
                          "unresolved_count": len(tail), "unresolved_interaction_ids": ";".join(tail)})
    v2_summary = _strategy_summary(strategy_id=STRATEGY_V2, raw=raw, plus=plus, passed=v2_passed, failed=v2_failed, audit=v2_audit, trades=all_v2_trades, unresolved=sorted(set(v2_unresolved)))
    v3_summary = _strategy_summary(strategy_id=STRATEGY_V3, raw=raw, plus=plus, passed=v3_passed, failed=v3_failed, audit=v3_audit, trades=all_v3_trades, unresolved=sorted(set(v3_unresolved)))
    payload = {"study": "CMEOrderflowAbsorption.ES_MAY_2026_RETROSPECTIVE_HOLDOUT", "evidence_label": EVIDENCE_LABEL,
               "strict_chronological_oos": False, "frozen_calibration_source": "DEVELOPMENT_ONLY_LATER_JULY_2026",
               "test_days": list(TARGET_DAYS), "manifest_verification": source, "strategy_semantics_changed": False,
               "strategies": {"v2_baseline": v2_summary, "v3_tick_3_target_3r": v3_summary}}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "strategy-comparison.csv", [v2_summary, v3_summary], ["strategy_id"])
    _write_csv(output_dir / "trade-ledger.csv", all_v2_trades + all_v3_trades, ["strategy_id"])
    _write_csv(output_dir / "daily-results.csv", daily, ["date", "strategy_id"])
    (output_dir / "diagnostic-report.md").write_text(_report(v2_summary, v3_summary), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only frozen V2/V3 L3 strategies on sealed May-2026 local data")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        print(json.dumps(run(data_root=args.data_root, output_dir=args.output_dir), indent=2, sort_keys=True))
    except (MayHoldoutError, BookStateError, v2.RetroReplayError) as exc:
        print(f"ERROR: {exc}"); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
