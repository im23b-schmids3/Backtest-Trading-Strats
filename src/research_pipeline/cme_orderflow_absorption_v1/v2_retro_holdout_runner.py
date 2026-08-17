"""Frozen-parameter retrospective ES/MES robustness replay.

This is deliberately *not* chronological OOS: the DEVELOPMENT_ONLY score
calibration comes from a later July block.  It never obtains data, recalibrates
scores, or changes the frozen V1 PLUS selection.  Input files end at 16:00 UTC;
any unresolved setup or position at that boundary fails closed with exit code 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .analysis import Diagnostics, RTH_END, RTH_START, day_and_seconds, volume_profile
from .engine import BookStateError, CausalMBOBook
from .oos_backtest_runner import _score, load_development_calibration, plus_only

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data/cme_orderflow_absorption_v2_holdout"
CALIBRATION = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/development-score-calibration.json"
CONTRACT = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/backtest-contract.json"
DEFAULT_OUTPUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RETRO_HOLDOUT/2026-06-23_2026-07-17"

STUDY_ID = "CMEOrderflowAbsorption.ES_V2_RETRO_HOLDOUT"
INTERPRETATION = "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST"
RAW_TICK = 250_000_000
RAW_PRICE_SCALE = 1_000_000_000
TICK = 0.25
CONFIRMATION_NS = 15_000_000_000
ENTRY_LATENCY_NS = 2_000_000
RISK_BUDGET = 250.0
TARGET_R = 2.5
ES_POINT_VALUE, MES_POINT_VALUE = 50.0, 5.0
ES_COMMISSION, MES_COMMISSION = 3.0, 1.25
ES_CAP, MES_CAP = 6, 60

TEST_DAYS = (
    "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30",
    "2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
    "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17",
)
PRIOR_RTH = {
    "2026-06-23": "2026-06-22", "2026-06-24": "2026-06-23", "2026-06-25": "2026-06-24",
    "2026-06-26": "2026-06-25", "2026-06-29": "2026-06-26", "2026-06-30": "2026-06-29",
    "2026-07-01": "2026-06-30", "2026-07-02": "2026-07-01", "2026-07-06": "2026-07-02",
    "2026-07-07": "2026-07-06", "2026-07-08": "2026-07-07", "2026-07-09": "2026-07-08",
    "2026-07-10": "2026-07-09", "2026-07-13": "2026-07-10", "2026-07-14": "2026-07-13",
    "2026-07-15": "2026-07-14", "2026-07-16": "2026-07-15", "2026-07-17": "2026-07-16",
}


class RetroReplayError(RuntimeError):
    pass


def _code(value: object) -> str:
    value = getattr(value, "value", value)
    text = str(value)
    return text.rsplit(".", 1)[-1] if "." in text else text


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _ns(day: str, clock: str = "16:00:00") -> int:
    return int(datetime.fromisoformat(f"{day}T{clock}+00:00").timestamp() * 1_000_000_000)


def _paths(data_root: Path, day: str) -> tuple[Path, Path, Path]:
    prior = PRIOR_RTH[day]
    return (
        data_root / "es_mbo" / f"ESU6_{day}_0000_1600_mbo.dbn.zst",
        data_root / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst",
        data_root / "es_rth_trades" / f"ESU6_{prior}_1330_2000_trades.dbn.zst",
    )


def _profile_levels(rows: list[tuple[int, int]]) -> dict[str, int]:
    profile = volume_profile(rows)
    if profile is None:
        raise RetroReplayError("missing prior-RTH executed-volume profile")
    return {"PRIOR_RTH_HIGH": profile["high"], "PRIOR_RTH_LOW": profile["low"], "PRIOR_RTH_POC": profile["poc"], "PRIOR_RTH_VAH": profile["vah"], "PRIOR_RTH_VAL": profile["val"]}


def raw_price_to_points(price: int | float) -> float:
    """Convert a Databento fixed-point price to ES/MES trading points once."""
    return float(price) / RAW_PRICE_SCALE


def interaction_zone_points(row: dict[str, Any]) -> tuple[float, float]:
    """Return completed-interaction zone bounds on the BBO/price-arithmetic scale.

    Interaction summaries preserve Databento raw fixed-point prices.  Both ES
    and MES BBO helpers already return conventional point prices, so convert
    the zone only at this execution boundary.
    """
    return raw_price_to_points(row["zone_low"]), raw_price_to_points(row["zone_high"])


def initial_prices(direction: str, bid: float, ask: float, zone_low: float, zone_high: float) -> dict[str, float]:
    """Frozen 5-tick zone stop, 2.5R target, one adverse tick each side."""
    if bid <= 0 or ask <= bid:
        raise RetroReplayError("invalid executable market")
    if direction == "BUYER_ABSORPTION":
        entry_reference, entry, stop = ask, ask + TICK, zone_low - 5 * TICK
        stop_exit = stop - TICK
        target = entry + TARGET_R * (entry - stop)
        return {"direction": "LONG", "entry_reference": entry_reference, "entry": entry, "stop": stop, "stop_exit": stop_exit, "target": target}
    if direction == "SELLER_ABSORPTION":
        entry_reference, entry, stop = bid, bid - TICK, zone_high + 5 * TICK
        stop_exit = stop + TICK
        target = entry - TARGET_R * (stop - entry)
        return {"direction": "SHORT", "entry_reference": entry_reference, "entry": entry, "stop": stop, "stop_exit": stop_exit, "target": target}
    raise RetroReplayError("unknown absorption direction")


def size_for_instrument(prices: dict[str, float], instrument: str) -> dict[str, float | int | str]:
    if instrument not in {"ES", "MES"}:
        raise ValueError("unsupported instrument")
    point_value, commission, cap = (ES_POINT_VALUE, ES_COMMISSION, ES_CAP) if instrument == "ES" else (MES_POINT_VALUE, MES_COMMISSION, MES_CAP)
    raw = abs(prices["entry_reference"] - prices["stop"]) * point_value
    slippage = (abs(prices["entry"] - prices["entry_reference"]) + abs(prices["stop_exit"] - prices["stop"])) * point_value
    one_contract = abs(prices["entry"] - prices["stop_exit"]) * point_value + 2 * commission
    risk_based = math.floor(RISK_BUDGET / one_contract)
    contracts = min(risk_based, cap)
    return {"instrument": instrument, "contracts": contracts, "risk_based_contracts": risk_based, "account_max_contracts": cap, "raw_price_risk_usd": raw, "slippage_contribution_usd": slippage, "one_contract_initial_risk_usd": one_contract, "initial_risk_usd": contracts * one_contract}


def choose_es_first(direction: str, es_bid: float, es_ask: float, zone_low: float, zone_high: float) -> tuple[dict[str, float], dict[str, float | int | str]]:
    prices = initial_prices(direction, es_bid, es_ask, zone_low, zone_high)
    sizing = size_for_instrument(prices, "ES")
    return prices, sizing


def _valid_es(book: CausalMBOBook) -> tuple[float, float] | None:
    bid, ask = book.best_bid(), book.best_ask()
    if bid is None or ask is None or bid <= 0 or ask <= bid or book.depth["B"][bid] < 1 or book.depth["A"][ask] < 1:
        return None
    return bid / RAW_PRICE_SCALE, ask / RAW_PRICE_SCALE


def _valid_mes(record: object) -> tuple[float, float] | None:
    levels = getattr(record, "levels", ())
    if not levels:
        return None
    level = levels[0]
    bid, ask = int(level.bid_px), int(level.ask_px)
    if bid <= 0 or ask <= bid or int(level.bid_sz) < 1 or int(level.ask_sz) < 1:
        return None
    return bid / RAW_PRICE_SCALE, ask / RAW_PRICE_SCALE


@dataclass
class PendingSignal:
    row: dict[str, Any]
    confirmation_due_ns: int
    state: str = "AWAITING_CONFIRMATION"
    entry_ready_ns: int | None = None
    mes_decision_ns: int | None = None


@dataclass
class Position:
    signal: PendingSignal
    instrument: str
    contracts: int
    entry_ns: int
    entry: float
    stop: float
    target: float
    one_contract_risk: float
    entry_commission: float


@dataclass
class DayState:
    day: str
    prior_day: str
    diagnostics: Diagnostics
    book: CausalMBOBook = field(default_factory=CausalMBOBook)
    completed_seen: int = 0
    pending: dict[str, PendingSignal] = field(default_factory=dict)
    position: Position | None = None
    interactions: list[dict[str, Any]] = field(default_factory=list)
    plus: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    high: int = 0
    strong: int = 0
    confirmations_passed: int = 0
    confirmations_failed: int = 0

    def audit_once(self, interaction_id: str, outcome: str, **details: object) -> None:
        if any(row["interaction_id"] == interaction_id for row in self.audit):
            return
        self.audit.append({"date": self.day, "interaction_id": interaction_id, "outcome": outcome, **details})


def _selection_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("interaction_id", "interaction_end", "level", "absorption_score", "replenishment_score", "direction", "zone_low", "zone_high")}


def _new_completed(state: DayState, calibration: dict[str, Any], contract: dict[str, Any]) -> None:
    # ``Diagnostics.interaction_rows`` serializes every completed interaction.
    # The replay calls this from a per-record loop, so only summarize the new
    # append-only slice.  This preserves both ordering and summary content.
    completed_items = state.diagnostics.completed[state.completed_seen:]
    state.completed_seen = len(state.diagnostics.completed)
    if not completed_items:
        return
    completed = [interaction.summary() for interaction in completed_items]
    scored = _score(completed, calibration)
    state.interactions.extend(scored)
    for row in scored:
        if row["absorption_score"] >= contract["frozen_selection"]["absorption_p95"]:
            state.high += 1
        if row["replenishment_score"] >= contract["frozen_selection"]["replenishment_p95"]:
            state.strong += 1
        if plus_only(contract, _selection_row(row)):
            state.plus.append(row)
            state.pending[row["interaction_id"]] = PendingSignal(row=row, confirmation_due_ns=int(row["interaction_end"]) + CONFIRMATION_NS)


def _close_position(state: DayState, position: Position, timestamp: int, reference: float, reason: str) -> None:
    if position.instrument == "ES":
        exit_fill = reference - TICK if position.signal.row["direction"] == "BUYER_ABSORPTION" else reference + TICK
        point_value, commission = ES_POINT_VALUE, ES_COMMISSION
    else:
        exit_fill = reference - TICK if position.signal.row["direction"] == "BUYER_ABSORPTION" else reference + TICK
        point_value, commission = MES_POINT_VALUE, MES_COMMISSION
    signed_points = exit_fill - position.entry if position.signal.row["direction"] == "BUYER_ABSORPTION" else position.entry - exit_fill
    gross = signed_points * point_value * position.contracts
    commissions = (position.entry_commission + commission * position.contracts)
    net = gross - commissions
    total_risk = position.one_contract_risk * position.contracts
    state.trades.append({
        "date": state.day, "interaction_id": position.signal.row["interaction_id"], "direction": position.signal.row["direction"],
        "level": position.signal.row["level"], "absorption_score": position.signal.row["absorption_score"], "replenishment_score": position.signal.row["replenishment_score"],
        "instrument": position.instrument, "contracts": position.contracts, "entry_timestamp": position.entry_ns,
        "exit_timestamp": timestamp, "entry": position.entry, "exit": exit_fill, "stop": position.stop, "target": position.target,
        "exit_reason": reason, "gross_usd": gross, "commission_usd": commissions, "net_usd": net, "r_multiple": net / total_risk,
    })
    state.audit_once(position.signal.row["interaction_id"], "TRADE_CLOSED", exit_reason=reason)
    state.position = None


def _manage_es(state: DayState, ts: int, bid: float, ask: float) -> None:
    position = state.position
    if position is None or position.instrument != "ES":
        return
    if position.signal.row["direction"] == "BUYER_ABSORPTION":
        if bid <= position.stop:
            _close_position(state, position, ts, position.stop, "STOP")
        elif bid >= position.target:
            _close_position(state, position, ts, position.target, "TARGET")
    else:
        if ask >= position.stop:
            _close_position(state, position, ts, position.stop, "STOP")
        elif ask <= position.target:
            _close_position(state, position, ts, position.target, "TARGET")


def _manage_mes(state: DayState, ts: int, bid: float, ask: float) -> None:
    position = state.position
    if position is None or position.instrument != "MES":
        return
    if position.signal.row["direction"] == "BUYER_ABSORPTION":
        if bid <= position.stop:
            _close_position(state, position, ts, position.stop, "STOP")
        elif bid >= position.target:
            _close_position(state, position, ts, position.target, "TARGET")
    else:
        if ask >= position.stop:
            _close_position(state, position, ts, position.stop, "STOP")
        elif ask <= position.target:
            _close_position(state, position, ts, position.target, "TARGET")


def _confirm_and_enter_es(state: DayState, ts: int, execution_price: float | None, bbo: tuple[float, float] | None) -> None:
    for pending in list(state.pending.values()):
        if pending.state == "AWAITING_CONFIRMATION" and execution_price is not None and ts >= pending.confirmation_due_ns:
            favorable = (execution_price - raw_price_to_points(pending.row["end_price"])) / TICK
            if pending.row["direction"] == "SELLER_ABSORPTION":
                favorable = -favorable
            if favorable < 1:
                pending.state = "CONFIRMATION_FAILED"
                state.confirmations_failed += 1
                state.audit_once(pending.row["interaction_id"], "CONFIRMATION_FAILED")
            else:
                pending.state, pending.entry_ready_ns = "AWAITING_ES_ENTRY", ts + ENTRY_LATENCY_NS
                state.confirmations_passed += 1
        if pending.state != "AWAITING_ES_ENTRY" or pending.entry_ready_ns is None or ts < pending.entry_ready_ns:
            continue
        if state.position is not None:
            pending.state = "POSITION_BLOCKED"
            state.audit_once(pending.row["interaction_id"], "POSITION_ALREADY_OPEN")
            continue
        if bbo is None:
            continue
        zone_low, zone_high = interaction_zone_points(pending.row)
        prices, es_size = choose_es_first(pending.row["direction"], bbo[0], bbo[1], zone_low, zone_high)
        if int(es_size["contracts"]) >= 1:
            pending.state = "ENTERED_ES"
            state.position = Position(pending, "ES", int(es_size["contracts"]), ts, prices["entry"], prices["stop"], prices["target"], float(es_size["one_contract_initial_risk_usd"]), ES_COMMISSION * int(es_size["contracts"]))
        else:
            pending.state, pending.mes_decision_ns = "AWAITING_MES_ENTRY", ts


def _enter_mes(state: DayState, ts: int, bbo: tuple[float, float] | None) -> None:
    if bbo is None:
        return
    for pending in list(state.pending.values()):
        if pending.state != "AWAITING_MES_ENTRY" or pending.mes_decision_ns is None or ts < pending.mes_decision_ns:
            continue
        if state.position is not None:
            pending.state = "POSITION_BLOCKED"
            state.audit_once(pending.row["interaction_id"], "POSITION_ALREADY_OPEN")
            continue
        zone_low, zone_high = interaction_zone_points(pending.row)
        prices = initial_prices(pending.row["direction"], bbo[0], bbo[1], zone_low, zone_high)
        sizing = size_for_instrument(prices, "MES")
        if int(sizing["contracts"]) < 1:
            pending.state = "INSUFFICIENT_RISK"
            state.audit_once(pending.row["interaction_id"], "INSUFFICIENT_RISK_BUDGET_FOR_ES_AND_MES")
            continue
        pending.state = "ENTERED_MES"
        state.position = Position(pending, "MES", int(sizing["contracts"]), ts, prices["entry"], prices["stop"], prices["target"], float(sizing["one_contract_initial_risk_usd"]), MES_COMMISSION * int(sizing["contracts"]))


def _stream_dbn(path: Path) -> Iterator[object]:
    from databento import DBNStore
    yield from DBNStore.from_file(path)


def _load_prior_rth(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for record in _stream_dbn(path):
        price, size = int(getattr(record, "price", 0)), int(getattr(record, "size", 0))
        if price > 0 and size > 0:
            rows.append((price, size))
    return rows


def _next(iterator: Iterator[object]) -> object | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _replay_day(day: str, es_path: Path, mes_path: Path, prior_path: Path, calibration: dict[str, Any], contract: dict[str, Any]) -> DayState:
    prior_levels = _profile_levels(_load_prior_rth(prior_path))
    diagnostics = Diagnostics()
    diagnostics.levels[day] = prior_levels
    state = DayState(day=day, prior_day=PRIOR_RTH[day], diagnostics=diagnostics)
    es_iter, mes_iter = iter(_stream_dbn(es_path)), iter(_stream_dbn(mes_path))
    es, mes = _next(es_iter), _next(mes_iter)
    records = 0
    while es is not None or mes is not None:
        es_ts = int(getattr(es, "ts_recv", getattr(es, "ts_event", 2**63 - 1))) if es is not None else 2**63 - 1
        mes_ts = int(getattr(mes, "ts_event", 2**63 - 1)) if mes is not None else 2**63 - 1
        if mes_ts < es_ts:
            quote = _valid_mes(mes)
            _manage_mes(state, mes_ts, *(quote or (0.0, 0.0))) if quote else None
            _enter_mes(state, mes_ts, quote)
            mes = _next(mes_iter)
            continue
        record, es = es, _next(es_iter)
        records += 1
        action = _code(getattr(record, "action", "N"))
        if action not in {"N", "NONE"}:
            applied = state.book.apply(action=action, side=_code(getattr(record, "side", "")), price=int(record.price), size=int(record.size), order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv), channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
            _, seconds = day_and_seconds(int(record.ts_recv))
            if RTH_START <= seconds < RTH_END:
                bbo = _valid_es(state.book)
                if bbo:
                    _manage_es(state, int(record.ts_event), *bbo)
                diagnostics.observe(record, applied, state.book.spread())
                # In Diagnostics, an interaction can close only from an
                # executed market observation (or ``finalize`` below).  A/C/M
                # book events merely enrich an active interaction, so querying
                # completions after them cannot change strategy state.
                if applied and applied.executed:
                    _new_completed(state, calibration, contract)
                execution = raw_price_to_points(record.price) if applied and applied.executed else None
                _confirm_and_enter_es(state, int(record.ts_event), execution, bbo)
        if records % 5_000_000 == 0:
            print(f"  records={records:,} raw_interactions={len(state.interactions):,} plus={len(state.plus):,}", flush=True)
    diagnostics.finalize()
    _new_completed(state, calibration, contract)
    return state


def _tail(state: DayState) -> list[str]:
    unresolved = [pending.row["interaction_id"] for pending in state.pending.values() if pending.state in {"AWAITING_CONFIRMATION", "AWAITING_ES_ENTRY", "AWAITING_MES_ENTRY"}]
    if state.position is not None:
        unresolved.append(state.position.signal.row["interaction_id"])
    return sorted(set(unresolved))


def _write_csv(path: Path, rows: list[dict[str, Any]], default: list[str]) -> None:
    fields = sorted({key for row in rows for key in row}) or default
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(float(row["net_usd"]) > 0 for row in trades)
    losses = sum(float(row["net_usd"]) < 0 for row in trades)
    gross_profit = sum(float(row["net_usd"]) for row in trades if float(row["net_usd"]) > 0)
    gross_loss = sum(float(row["net_usd"]) for row in trades if float(row["net_usd"]) < 0)
    return {"wins": wins, "losses": losses, "win_rate": wins / len(trades) if trades else 0.0, "net_pnl": sum(float(row["net_usd"]) for row in trades), "gross_profit": gross_profit, "gross_loss": gross_loss, "profit_factor": gross_profit / abs(gross_loss) if gross_loss else None, "total_r": sum(float(row["r_multiple"]) for row in trades), "average_r": sum(float(row["r_multiple"]) for row in trades) / len(trades) if trades else 0.0}


def _breakdown(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade[field])].append(trade)
    return {key: {"trades": len(rows), "net_pnl": sum(float(row["net_usd"]) for row in rows), "total_r": sum(float(row["r_multiple"]) for row in rows)} for key, rows in sorted(grouped.items())}


def _manifest(data_root: Path) -> dict[str, Any]:
    days = []
    for day in TEST_DAYS:
        es, mes, prior = _paths(data_root, day)
        for path in (es, mes, prior):
            if not path.is_file():
                raise RetroReplayError(f"missing sealed input: {path}")
        days.append({"date": day, "previous_completed_rth": PRIOR_RTH[day], "es_mbo": {"path": str(es), "sha256": _hash(es)}, "mes_mbp1": {"path": str(mes), "sha256": _hash(mes)}, "prior_rth_trades": {"path": str(prior), "sha256": _hash(prior)}})
    return {"study": STUDY_ID, "interpretation": INTERPRETATION, "score_calibration_source": "DEVELOPMENT_ONLY", "oos_rank_recomputation_count": 0, "frozen_calibration": {"path": str(CALIBRATION), "sha256": _hash(CALIBRATION)}, "strategy_parameters": {"confirmation_seconds": 15, "minimum_favorable_ticks": 1, "no_early_invalidation": True, "post_confirmation_latency_ms": 2, "stop_ticks": 5, "target_r": 2.5, "risk_budget_usd": 250, "es_commission_per_side": 3.0, "mes_commission_per_side": 1.25, "apex_max_es": 6, "apex_max_mes": 60, "hard_cutoff_utc": "22:45:00"}, "days": days}


def run(*, data_root: Path = DATA_ROOT, output_dir: Path = DEFAULT_OUTPUT) -> int:
    if output_dir.exists():
        raise RetroReplayError(f"immutable output path already exists: {output_dir}")
    manifest = _manifest(data_root)
    calibration, contract = load_development_calibration(), json.loads(CONTRACT.read_text(encoding="utf-8"))
    all_interactions: list[dict[str, Any]] = []
    all_plus: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_audit: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    tails: dict[str, list[str]] = {}
    raw = high = strong = confirmations_passed = confirmations_failed = 0
    for index, day in enumerate(TEST_DAYS, start=1):
        print(f"=== DAY {index:02d}/18 {day} ===", flush=True)
        es, mes, prior = _paths(data_root, day)
        print("context ready", flush=True)
        state = _replay_day(day, es, mes, prior, calibration, contract)
        tail = _tail(state)
        raw += len(state.interactions); high += state.high; strong += state.strong
        confirmations_passed += state.confirmations_passed; confirmations_failed += state.confirmations_failed
        all_interactions.extend(state.interactions); all_plus.extend(state.plus); all_trades.extend(state.trades); all_audit.extend(state.audit)
        if tail:
            tails[day] = tail
        daily.append({"date": day, "previous_completed_rth": PRIOR_RTH[day], "raw_interactions": len(state.interactions), "high": state.high, "strong": state.strong, "plus": len(state.plus), "confirmations_passed": state.confirmations_passed, "confirmations_failed": state.confirmations_failed, "trades": len(state.trades), "tail_required": bool(tail), "unresolved_interaction_ids": ";".join(tail)})
        print(f"reconstruction complete\nraw interactions = {len(state.interactions)}\nPLUS = {len(state.plus)}\nexecution replay complete\ntrades = {len(state.trades)}\ntail required = {bool(tail)}", flush=True)
    status = "TAIL_DATA_REQUIRED" if tails else "RECONCILED_RETRO_ROBUSTNESS_REPLAY"
    summary = {"status": status, "interpretation": INTERPRETATION, "test_day_count": len(TEST_DAYS), "raw_interactions": raw, "high_count": high, "strong_count": strong, "plus_count": len(all_plus), "confirmations_passed": confirmations_passed, "confirmations_failed": confirmations_failed, "es_trades": sum(row["instrument"] == "ES" for row in all_trades), "mes_trades": sum(row["instrument"] == "MES" for row in all_trades), "tail_required_dates": tails, "score_calibration_source": "DEVELOPMENT_ONLY", "oos_rank_recomputation_count": 0, "performance_by_day": _breakdown(all_trades, "date"), "performance_by_direction": _breakdown(all_trades, "direction"), "performance_by_instrument": _breakdown(all_trades, "instrument"), "performance_by_level": _breakdown(all_trades, "level"), **_performance(all_trades)}
    if tails:
        summary["partial_metrics_label"] = "COMPLETED_BEFORE_1600_PARTIAL_METRICS"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "daily-summary.csv", daily, ["date", "tail_required"])
    _write_csv(output_dir / "interactions.csv", all_interactions, ["interaction_id"])
    _write_csv(output_dir / "plus-signals.csv", all_plus, ["interaction_id"])
    _write_csv(output_dir / "trades.csv", all_trades, ["interaction_id"])
    _write_csv(output_dir / "audit.csv", all_audit, ["interaction_id", "outcome"])
    return 2 if tails else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen ES/MES retrospective robustness replay; no acquisition path")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        return run(data_root=args.data_root, output_dir=args.output_dir)
    except (RetroReplayError, BookStateError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
