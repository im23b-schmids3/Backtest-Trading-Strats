"""Predeclared six-cell V2 tick-trigger × target research matrix.

This research runner consumes the already-materialized V1 PLUS populations.
It never reconstructs interactions or recalibrates scores.  Only confirmation
timing changes inside the six sealed matrix cells; all other execution behavior
uses the existing V2 runners' fixed price, stop, risk, cap, and tail rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import v2_aug_vs_retro_diagnostic as artifacts
from . import v2_aug_vs_retro_path_replay as paths
from .oos_backtest_runner import CausalMBOBook, _cutoff, _valid, day_and_seconds, is_snapshot_record, load_frozen, prices
from .v2_retro_holdout_runner import (
    CONFIRMATION_NS, ENTRY_LATENCY_NS, ES_COMMISSION, ES_POINT_VALUE,
    MES_COMMISSION, MES_POINT_VALUE, RAW_PRICE_SCALE, TICK, _code,
    _valid_es, _valid_mes, choose_es_first, initial_prices, size_for_instrument,
)
from .v2_target_matrix_runner import DBN as AUGUST_ES_MBO, action_value, load_seen_aug_plus, size_trade_with_mes_fallback


ROOT = artifacts.ROOT
DEFAULT_OUTPUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V3_RESEARCH/tick_trigger_target_matrix"
DEFAULT_RETRO_DATA = paths.DEFAULT_RETRO_DATA
PROGRESS_EVERY = 5_000_000
RAW_TICK = 250_000_000


class MatrixError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatrixSpec:
    trigger_ticks: int
    target_r: float

    @property
    def cell_id(self) -> str:
        return f"TICK_{self.trigger_ticks}_TARGET_{str(self.target_r).replace('.', '_')}R"


SEALED_MATRIX = tuple(MatrixSpec(trigger, target) for trigger in (1, 2, 3) for target in (2.5, 3.0))


@dataclass
class Pending:
    row: dict[str, Any]
    state: str = "AWAITING_TRIGGER"
    trigger_timestamp: int | None = None
    trigger_price: float | None = None
    trigger_favorable_ticks: float | None = None
    entry_ready_ns: int | None = None

    @property
    def deadline(self) -> int:
        return int(self.row["interaction_end"]) + CONFIRMATION_NS


@dataclass
class Position:
    pending: Pending
    instrument: str
    contracts: int
    entry_timestamp: int
    entry: float
    stop: float
    target: float
    one_contract_risk: float
    entry_commission: float


@dataclass
class Cell:
    spec: MatrixSpec
    period: str
    execution_model: str
    pending: dict[str, Pending] = field(default_factory=dict)
    position: Position | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    confirmations_passed: int = 0
    confirmations_failed: int = 0

    def __post_init__(self) -> None:
        self.pending = dict(self.pending)

    def add_signals(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            identifier = str(row["interaction_id"])
            if identifier in self.pending:
                raise MatrixError(f"duplicate PLUS signal: {identifier}")
            self.pending[identifier] = Pending(dict(row))

    def _audit(self, pending: Pending, outcome: str, **extra: object) -> None:
        if any(row["interaction_id"] == pending.row["interaction_id"] for row in self.audit):
            return
        self.audit.append({"period": self.period, "cell_id": self.spec.cell_id, "interaction_id": pending.row["interaction_id"], "date": pending.row["date"], "outcome": outcome, **extra})

    def _favorable_ticks(self, raw_execution_price: int) -> float:
        raise RuntimeError("use observe_execution with a pending signal")

    def expire(self, timestamp: int) -> None:
        for pending in self.pending.values():
            if pending.state == "AWAITING_TRIGGER" and timestamp > pending.deadline:
                pending.state = "CONFIRMATION_FAILED"
                self.confirmations_failed += 1
                self._audit(pending, "CONFIRMATION_FAILED")

    def observe_execution(self, timestamp: int, raw_execution_price: int) -> None:
        """The first ES execution at each cell's predeclared favorable threshold."""
        self.expire(timestamp)
        price = raw_execution_price / RAW_PRICE_SCALE
        for pending in self.pending.values():
            if pending.state != "AWAITING_TRIGGER" or timestamp < int(pending.row["interaction_end"]) or timestamp > pending.deadline:
                continue
            end = float(pending.row["end_price"]) / RAW_PRICE_SCALE
            favorable = (price - end) / TICK
            if pending.row["direction"] == "SELLER_ABSORPTION":
                favorable = -favorable
            if favorable >= self.spec.trigger_ticks:
                pending.state = "AWAITING_ES_ENTRY"
                pending.trigger_timestamp, pending.trigger_price = timestamp, price
                pending.trigger_favorable_ticks, pending.entry_ready_ns = favorable, timestamp + ENTRY_LATENCY_NS
                self.confirmations_passed += 1

    def _target(self, direction: str, entry: float, stop: float) -> float:
        return entry + self.spec.target_r * (entry - stop) if direction == "BUYER_ABSORPTION" else entry - self.spec.target_r * (stop - entry)

    def _enter(self, pending: Pending, instrument: str, timestamp: int, p: dict[str, float], sizing: dict[str, Any]) -> None:
        pending.state = f"ENTERED_{instrument}"
        p = dict(p); p["target"] = self._target(pending.row["direction"], p["entry"], p["stop"])
        commission = ES_COMMISSION if instrument == "ES" else MES_COMMISSION
        self.position = Position(pending, instrument, int(sizing["contracts"]), timestamp, p["entry"], p["stop"], p["target"], float(sizing["one_contract_initial_risk_usd"]), commission * int(sizing["contracts"]))

    def try_august_entry(self, timestamp: int, bbo: tuple[float, float] | None) -> None:
        """Existing August behavior: ES BBO supplies MES-proxy fallback execution."""
        if bbo is None:
            return
        for pending in sorted(self.pending.values(), key=lambda item: (item.entry_ready_ns or 2**63 - 1, item.row["interaction_end"], item.row["interaction_id"])):
            if pending.state != "AWAITING_ES_ENTRY" or pending.entry_ready_ns is None or timestamp < pending.entry_ready_ns:
                continue
            if self.position is not None:
                pending.state = "POSITION_BLOCKED"; self._audit(pending, "POSITION_ALREADY_OPEN"); continue
            p = prices(pending.row["direction"], bbo[0], bbo[1], int(pending.row["zone_low"]) / RAW_PRICE_SCALE, int(pending.row["zone_high"]) / RAW_PRICE_SCALE)
            sizing = size_trade_with_mes_fallback(p)
            if not int(sizing["contracts"]):
                pending.state = "INSUFFICIENT_RISK"; self._audit(pending, "INSUFFICIENT_RISK_BUDGET_FOR_ES_AND_MES"); continue
            self._enter(pending, str(sizing["instrument"]), timestamp, p, sizing)

    def try_retro_es_entry(self, timestamp: int, bbo: tuple[float, float] | None) -> None:
        if bbo is None:
            return
        for pending in sorted(self.pending.values(), key=lambda item: (item.entry_ready_ns or 2**63 - 1, item.row["interaction_end"], item.row["interaction_id"])):
            if pending.state != "AWAITING_ES_ENTRY" or pending.entry_ready_ns is None or timestamp < pending.entry_ready_ns:
                continue
            if self.position is not None:
                pending.state = "POSITION_BLOCKED"; self._audit(pending, "POSITION_ALREADY_OPEN"); continue
            zone_low, zone_high = int(pending.row["zone_low"]) / RAW_PRICE_SCALE, int(pending.row["zone_high"]) / RAW_PRICE_SCALE
            p, sizing = choose_es_first(pending.row["direction"], bbo[0], bbo[1], zone_low, zone_high)
            if int(sizing["contracts"]):
                self._enter(pending, "ES", timestamp, p, sizing)
            else:
                pending.state = "AWAITING_MES_ENTRY"

    def try_retro_mes_entry(self, timestamp: int, bbo: tuple[float, float] | None) -> None:
        if bbo is None:
            return
        for pending in sorted(self.pending.values(), key=lambda item: (item.entry_ready_ns or 2**63 - 1, item.row["interaction_end"], item.row["interaction_id"])):
            if pending.state != "AWAITING_MES_ENTRY" or pending.entry_ready_ns is None or timestamp < pending.entry_ready_ns:
                continue
            if self.position is not None:
                pending.state = "POSITION_BLOCKED"; self._audit(pending, "POSITION_ALREADY_OPEN"); continue
            p = initial_prices(pending.row["direction"], bbo[0], bbo[1], int(pending.row["zone_low"]) / RAW_PRICE_SCALE, int(pending.row["zone_high"]) / RAW_PRICE_SCALE)
            sizing = size_for_instrument(p, "MES")
            if not int(sizing["contracts"]):
                pending.state = "INSUFFICIENT_RISK"; self._audit(pending, "INSUFFICIENT_RISK_BUDGET_FOR_ES_AND_MES"); continue
            self._enter(pending, "MES", timestamp, p, sizing)

    def manage(self, timestamp: int, bid: float, ask: float) -> None:
        position = self.position
        if position is None:
            return
        long = position.pending.row["direction"] == "BUYER_ABSORPTION"
        stop_hit = (long and bid <= position.stop) or (not long and ask >= position.stop)
        target_hit = (long and bid >= position.target) or (not long and ask <= position.target)
        if not (stop_hit or target_hit):
            return
        # Frozen same-bar behavior: stop wins when both sides are reached.
        reason = "STOP" if stop_hit else "TARGET"
        reference = position.stop if stop_hit else position.target
        exit_fill = reference - TICK if long else reference + TICK
        points = exit_fill - position.entry if long else position.entry - exit_fill
        point_value = ES_POINT_VALUE if position.instrument == "ES" else MES_POINT_VALUE
        commission = ES_COMMISSION if position.instrument == "ES" else MES_COMMISSION
        gross = points * point_value * position.contracts
        fees = position.entry_commission + commission * position.contracts
        total_risk = position.one_contract_risk * position.contracts
        self.trades.append({
            "period": self.period, "cell_id": self.spec.cell_id, "trigger_ticks": self.spec.trigger_ticks, "target_r": self.spec.target_r,
            "date": position.pending.row["date"], "interaction_id": position.pending.row["interaction_id"], "direction": position.pending.row["direction"], "level": position.pending.row["level"],
            "interaction_end": position.pending.row["interaction_end"], "trigger_timestamp": position.pending.trigger_timestamp, "trigger_price": position.pending.trigger_price, "trigger_favorable_ticks": position.pending.trigger_favorable_ticks,
            "entry_timestamp": position.entry_timestamp, "entry": position.entry, "stop": position.stop, "target": position.target, "instrument": position.instrument, "contracts": position.contracts,
            "exit_timestamp": timestamp, "exit": exit_fill, "exit_reason": reason, "gross_usd": gross, "commission_usd": fees, "net_usd": gross - fees, "r_multiple": (gross - fees) / total_risk,
            "entry_displacement_ticks_from_interaction_end": ((position.entry - float(position.pending.row["end_price"]) / RAW_PRICE_SCALE) / TICK if long else (float(position.pending.row["end_price"]) / RAW_PRICE_SCALE - position.entry) / TICK),
            "stop_distance_ticks": abs(position.entry - position.stop) / TICK,
            "seconds_interaction_end_to_trigger": (position.pending.trigger_timestamp - int(position.pending.row["interaction_end"])) / RAW_PRICE_SCALE if position.pending.trigger_timestamp is not None else None,
            "seconds_trigger_to_entry": (position.entry_timestamp - position.pending.trigger_timestamp) / RAW_PRICE_SCALE if position.pending.trigger_timestamp is not None else None,
        })
        self._audit(position.pending, "TRADE_CLOSED", exit_reason=reason)
        self.position = None

    def tail(self) -> list[str]:
        unresolved = [pending.row["interaction_id"] for pending in self.pending.values() if pending.state in {"AWAITING_TRIGGER", "AWAITING_ES_ENTRY", "AWAITING_MES_ENTRY"}]
        if self.position is not None:
            unresolved.append(self.position.pending.row["interaction_id"])
        return sorted(set(unresolved))


def create_cells(period: str, rows: list[dict[str, Any]], execution_model: str) -> list[Cell]:
    cells = [Cell(spec, period, execution_model) for spec in SEALED_MATRIX]
    for cell in cells:
        cell.add_signals(rows)
    return cells


def _close_cutoff(cells: list[Cell], timestamp: int, quote: tuple[float, float] | None) -> None:
    # Current source ends before the frozen cutoff; this retained guard only
    # prevents fabricated exits if future source coverage includes the cutoff.
    if quote is None:
        return
    for cell in cells:
        position = cell.position
        if position is None:
            continue
        long = position.pending.row["direction"] == "BUYER_ABSORPTION"
        cell.manage(timestamp, quote[0] if long else position.stop, quote[1] if not long else position.stop)


def _scan_august(cells: list[Cell]) -> None:
    from databento import DBNStore
    _, manifest = load_frozen(); book = CausalMBOBook()
    for count, record in enumerate(DBNStore.from_file(AUGUST_ES_MBO), start=1):
        if is_snapshot_record(record, manifest):
            book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False); continue
        if action_value(record) == "N":
            continue
        applied = book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False)
        quote = _valid(book); timestamp = int(record.ts_recv)
        for cell in cells:
            cell.expire(timestamp)
            if applied is not None and applied.executed:
                cell.observe_execution(timestamp, int(record.price))
            cell.try_august_entry(timestamp, quote)
            if quote is not None:
                cell.manage(timestamp, *quote)
        if count % PROGRESS_EVERY == 0:
            print(f"[tick matrix August] records={count:,}", flush=True)


def _scan_retro_day(cells: list[Cell], date: str, data_root: Path) -> None:
    from databento import DBNStore
    es_path, mes_path = paths._retro_paths(data_root, date)
    es_iter, mes_iter = iter(DBNStore.from_file(es_path)), iter(DBNStore.from_file(mes_path))
    es, mes = next(es_iter, None), next(mes_iter, None); book = CausalMBOBook(); count = 0
    while es is not None or mes is not None:
        es_ts = int(getattr(es, "ts_recv", getattr(es, "ts_event", 2**63 - 1))) if es is not None else 2**63 - 1
        mes_ts = int(getattr(mes, "ts_event", 2**63 - 1)) if mes is not None else 2**63 - 1
        if mes_ts < es_ts:
            quote = _valid_mes(mes)
            for cell in cells:
                cell.expire(mes_ts); cell.try_retro_mes_entry(mes_ts, quote)
                if quote is not None: cell.manage(mes_ts, *quote)
            mes = next(mes_iter, None); continue
        record, es = es, next(es_iter, None); count += 1
        action = _code(getattr(record, "action", "N"))
        if action in {"N", "NONE"}:
            continue
        applied = book.apply(action=action, side=_code(getattr(record, "side", "")), price=int(record.price), size=int(record.size), order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv), channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
        timestamp, quote = int(record.ts_event), _valid_es(book)
        for cell in cells:
            cell.expire(timestamp)
            if applied is not None and applied.executed:
                cell.observe_execution(timestamp, int(record.price))
            cell.try_retro_es_entry(timestamp, quote)
            if quote is not None: cell.manage(timestamp, *quote)
        if count % PROGRESS_EVERY == 0:
            print(f"[tick matrix retro] {date}: records={count:,}", flush=True)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    values = sorted(values); middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"count": len(values), "mean": sum(values) / len(values), "median": median}


def _breakdown(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades: groups[str(trade[field])].append(trade)
    return {key: {"trades": len(rows), "total_r": sum(float(row["r_multiple"]) for row in rows), "net_pnl": sum(float(row["net_usd"]) for row in rows)} for key, rows in sorted(groups.items())}


def result(cell: Cell) -> dict[str, Any]:
    trades = cell.trades; wins = [trade for trade in trades if float(trade["net_usd"]) > 0]; losses = [trade for trade in trades if float(trade["net_usd"]) < 0]
    profit = sum(float(trade["net_usd"]) for trade in wins); loss = abs(sum(float(trade["net_usd"]) for trade in losses)); tail = cell.tail()
    return {
        "period": cell.period, "cell_id": cell.spec.cell_id, "trigger_ticks": cell.spec.trigger_ticks, "target_r": cell.spec.target_r,
        "plus_count": len(cell.pending), "confirmations_passed": cell.confirmations_passed, "confirmations_failed": cell.confirmations_failed,
        "trades": len(trades), "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / len(trades) if trades else 0.0,
        "total_r": sum(float(trade["r_multiple"]) for trade in trades), "average_r": sum(float(trade["r_multiple"]) for trade in trades) / len(trades) if trades else 0.0,
        "net_pnl": sum(float(trade["net_usd"]) for trade in trades), "profit_factor": profit / loss if loss else None,
        "es_trades": sum(trade["instrument"] == "ES" for trade in trades), "mes_trades": sum(trade["instrument"] == "MES" for trade in trades),
        "target_exits": sum(trade["exit_reason"] == "TARGET" for trade in trades), "stop_exits": sum(trade["exit_reason"] == "STOP" for trade in trades),
        "cutoff_outcomes": sum(trade["exit_reason"] == "CUTOFF_FORCED_FLAT" for trade in trades), "unresolved_count": len(tail), "unresolved_interaction_ids": tail,
        "entry_diagnostics": {"entry_displacement_ticks": _stats([float(trade["entry_displacement_ticks_from_interaction_end"]) for trade in trades]), "stop_distance_ticks": _stats([float(trade["stop_distance_ticks"]) for trade in trades]), "seconds_interaction_end_to_trigger": _stats([float(trade["seconds_interaction_end_to_trigger"]) for trade in trades if trade["seconds_interaction_end_to_trigger"] is not None]), "seconds_trigger_to_entry": _stats([float(trade["seconds_trigger_to_entry"]) for trade in trades if trade["seconds_trigger_to_entry"] is not None])},
        "breakdown": {field: _breakdown(trades, field) for field in ("date", "direction", "level", "instrument")},
        "execution_model": cell.execution_model,
        "status": "INCOMPLETE_TAIL_DATA_REQUIRED" if tail else "COMPLETE_RESEARCH_MATRIX_CELL",
    }


def _baseline() -> dict[str, Any]:
    retro_summary = json.loads((artifacts.DEFAULT_RETRO_ROOT / "summary.json").read_text(encoding="utf-8"))
    return {"definition": "fixed_15_second_horizon__first_ES_execution_at_or_after_horizon__minimum_1_tick__2ms_latency__2p5R", "retro": {"artifact": str(artifacts.DEFAULT_RETRO_ROOT / "summary.json"), "trades": retro_summary.get("es_trades", 0) + retro_summary.get("mes_trades", 0), "total_r": retro_summary.get("total_r"), "status": retro_summary.get("status")}, "august": {"status": "UNAVAILABLE_NO_MATERIALIZED_FIXED_15S_2P5R_ARTIFACT"}}


def materialize(*, output_dir: Path = DEFAULT_OUTPUT, retro_data_root: Path = DEFAULT_RETRO_DATA) -> dict[str, Any]:
    if output_dir.exists(): raise MatrixError(f"immutable output already exists: {output_dir}")
    august_rows = [{**row, "date": row["date"]} for row in load_seen_aug_plus()]
    retro_rows = artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "plus-signals.csv")
    august_cells = create_cells("AUGUST_SEEN", august_rows, "MES_PROXY_EXECUTION_FROM_ES_MBO")
    retro_cells = create_cells("RETRO_JUNE_JULY", retro_rows, "NATIVE_MES_MBP1_FALLBACK")
    _scan_august(august_cells)
    for date in sorted({str(row["date"]) for row in retro_rows}): _scan_retro_day(retro_cells, date, retro_data_root)
    results = [result(cell) for cell in august_cells + retro_cells]; trades = [trade for cell in august_cells + retro_cells for trade in cell.trades]
    payload = {"study": "CMEOrderflowAbsorption.ES_V3_RESEARCH.TICK_TRIGGER_TARGET_MATRIX", "predeclared_cells": [{"trigger_ticks": spec.trigger_ticks, "target_r": spec.target_r, "cell_id": spec.cell_id} for spec in SEALED_MATRIX], "strategy_semantics_changed_outside_confirmation_trigger": False, "pnl_optimization_performed": False, "selection_prohibited": True, "interpretations": {"august": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE", "retro": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST"}, "baseline_context": _baseline(), "execution_models": {"august": "MES_PROXY_EXECUTION_FROM_ES_MBO", "retro": "NATIVE_MES_MBP1_FALLBACK"}, "results": results}
    output_dir.mkdir(parents=True)
    for name, rows in (("matrix-results.csv", results), ("trade-ledger.csv", trades)):
        fields = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))}) or ["cell_id"]
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key in fields} for row in rows])
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "period-comparison.json").write_text(json.dumps({"august": [row for row in results if row["period"] == "AUGUST_SEEN"], "retro": [row for row in results if row["period"] == "RETRO_JUNE_JULY"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text("# Frozen tick-trigger × target matrix\n\nThis research uses exactly six predeclared cells: +1/+2/+3 ES favorable ticks crossed with 2.5R/3.0R targets. August is seen data; retro is retrospective data. No result selects a V3 rule, and any future V3 hypothesis requires untouched OOS validation.\n\nOnly the confirmation moment changes: the first causal ES execution reaching the cell threshold inside the fixed 15-second window, followed by the same 2ms latency and existing execution model. No PLUS, score, level, stop, sizing, cap, direction, time, or target variants beyond the six declared cells are used.\n\nAugust MES remains ES-MBO proxy execution; retro MES remains native MES MBP-1. Incomplete source-end cells are reported fail-closed.\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen six-cell V2 tick-trigger × target research matrix")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--retro-data-root", type=Path, default=DEFAULT_RETRO_DATA)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(output_dir=args.output_dir, retro_data_root=args.retro_data_root), indent=2, sort_keys=True))
    except MatrixError as exc:
        print(f"ERROR: {exc}"); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
