from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .models import LSMRV2Config, TERMINAL_DISPOSITIONS


def _d(value: Any) -> Decimal: return Decimal(str(value))
def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values); middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
def _time(bar: dict[str, Any]) -> datetime:
    value = bar["timestamp"]
    return value.astimezone(timezone.utc) if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
def _hash(prefix: str, payload: dict[str, Any]) -> str: return prefix + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:20]


def structure_id(*, direction: str, reference_level: Decimal, reference_start_index: int, reference_end_index: int) -> str:
    return _hash("lsmr-v2-structure-", {"direction": direction, "reference_level": str(reference_level), "reference_start_index": reference_start_index, "reference_end_index": reference_end_index})


def setup_id(*, direction: str, structure: str, sweep_timestamp: Any) -> str:
    return _hash("lsmr-v2-", {"direction": direction, "structure_id": structure, "sweep_timestamp": str(sweep_timestamp)})


def detect_setups(bars: list[dict[str, Any]], config: LSMRV2Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create one unresolved setup per direction/reference and retain repeat extremes as events."""
    proposed: list[dict[str, Any]] = []; events: list[dict[str, Any]] = []; active: dict[tuple[str, str], dict[str, Any]] = {}
    for index in range(config.reference_bars, len(bars)):
        prior = bars[index-config.reference_bars:index]
        for direction, level in (("LONG", min(_d(x["low"]) for x in prior)), ("SHORT", max(_d(x["high"]) for x in prior))):
            penetration = max(level * config.penetration_fraction, config.price_tick * 4)
            extreme = _d(bars[index]["low"] if direction == "LONG" else bars[index]["high"])
            swept = extreme <= level - penetration if direction == "LONG" else extreme >= level + penetration
            if not swept: continue
            sid = structure_id(direction=direction, reference_level=level, reference_start_index=index-config.reference_bars, reference_end_index=index-1)
            key = (direction, str(level))
            if key in active:
                setup = active[key]; previous = _d(setup["extreme"])
                updated = min(previous, extreme) if direction == "LONG" else max(previous, extreme)
                setup["extreme"] = str(updated)
                setup["event_history"].append({"event": "EXTREME_UPDATED", "bar_index": index, "extreme": str(updated)})
                events.append({"event": "EXTREME_UPDATED", "setup_id": setup["setup_id"], "bar_index": index, "extreme": str(updated)})
                continue
            setup = {"setup_id": setup_id(direction=direction, structure=sid, sweep_timestamp=bars[index]["timestamp"]), "structure_id": sid, "direction": direction, "reference": str(level), "extreme": str(extreme), "sweep_index": index, "event_history": [{"event": "PROPOSED_SETUP", "bar_index": index, "extreme": str(extreme)}]}
            active[key] = setup; proposed.append(setup); events.append({"event": "PROPOSED_SETUP", "setup_id": setup["setup_id"], "bar_index": index})
    return proposed, events


def terminal_disposition(setup: dict[str, Any], bars: list[dict[str, Any]], config: LSMRV2Config) -> tuple[str, dict[str, Any] | None]:
    sweep = int(setup["sweep_index"]); direction = setup["direction"]; level = _d(setup["reference"]); extreme = _d(setup["extreme"])
    for index in range(sweep, min(sweep + config.reclaim_window_bars, len(bars))):
        bar = bars[index]; stamp = _time(bar); close = _d(bar["close"]); open_ = _d(bar["open"]); spread = _d(bar["high"]) - _d(bar["low"])
        extreme = min(extreme, _d(bar["low"])) if direction == "LONG" else max(extreme, _d(bar["high"]))
        reclaimed = close > level if direction == "LONG" else close < level
        if not reclaimed: continue
        if spread == 0 or abs(close - open_) < spread * config.body_fraction or (direction == "LONG" and close <= open_) or (direction == "SHORT" and close >= open_): return "CANDLE_REJECTED", None
        history = [_d(x["volume"]) for x in bars[max(0, index-config.volume_lookback_bars):index]]
        if len(history) != config.volume_lookback_bars or _d(bar["volume"]) < _median(history) * config.volume_multiple: return "VOLUME_REJECTED", None
        slope_index = index - config.vwap_slope_bars
        if slope_index < 0 or _time(bars[slope_index]).date() != stamp.date(): return "SESSION_CONTEXT_UNAVAILABLE", None
        if abs(_d(bar["daily_vwap"]) - _d(bars[slope_index]["daily_vwap"])) / close > config.maximum_vwap_slope_fraction: return "REGIME_REJECTED", None
        if abs(close - _d(bar["daily_vwap"])) / close > config.maximum_vwap_proximity_fraction: return "VWAP_PROXIMITY_REJECTED", None
        if stamp.hour == config.utc_force_flat_hour and stamp.minute == config.utc_force_flat_minute: return "SESSION_ENDED", None
        if index + 1 >= len(bars): return "NO_EXECUTABLE_ENTRY", None
        entry_bar = bars[index + 1]
        if _time(entry_bar).date() != stamp.date(): return "SESSION_ENDED", None
        entry = _d(entry_bar["open"]); stop = extreme - config.stop_buffer_ticks * config.price_tick if direction == "LONG" else extreme + config.stop_buffer_ticks * config.price_tick
        distance = abs(entry - stop)
        if distance < entry * config.minimum_stop_fraction or distance > entry * config.maximum_stop_fraction: return "STOP_DISTANCE_REJECTED", None
        return "TRADE_EXECUTED", {"setup_id": setup["setup_id"], "trade_id": _hash("lsmr-v2-trade-", {"setup_id": setup["setup_id"], "entry_timestamp": str(entry_bar["timestamp"])}), "direction": direction, "reclaim_index": index, "entry_price": str(entry), "initial_stop_price": str(stop), "target_price": str(entry + config.target_r_multiple * distance if direction == "LONG" else entry - config.target_r_multiple * distance)}
    return "RECLAIM_WINDOW_EXPIRED", None


def simulate_trade(*, setup: dict[str, Any], reclaim_index: int, bars: list[dict[str, Any]], config: LSMRV2Config) -> tuple[str, dict[str, Any] | None]:
    """Pure V2 execution: one next-open position, stop-first, time/session exits, and verified cost inputs."""
    entry_index = reclaim_index + 1
    if entry_index >= len(bars) or _time(bars[entry_index]).date() != _time(bars[reclaim_index]).date(): return "NO_EXECUTABLE_ENTRY", None
    direction = setup["direction"]; sign = Decimal(1) if direction == "LONG" else Decimal(-1)
    raw_entry = _d(bars[entry_index]["open"]); entry = raw_entry + sign * config.market_slippage_ticks * config.price_tick
    extreme = _d(setup["extreme"]); stop = extreme - config.stop_buffer_ticks * config.price_tick if direction == "LONG" else extreme + config.stop_buffer_ticks * config.price_tick
    risk = abs(entry - stop)
    if risk < entry * config.minimum_stop_fraction or risk > entry * config.maximum_stop_fraction: return "STOP_DISTANCE_REJECTED", None
    target = entry + sign * config.target_r_multiple * risk; exit_price = _d(bars[entry_index]["close"]); exit_index = entry_index; reason = "TIME_STOP"; stopped = False
    for index in range(entry_index, min(len(bars), entry_index + config.time_stop_bars)):
        bar = bars[index]; stamp = _time(bar)
        if stamp.date() != _time(bars[entry_index]).date(): reason = "SESSION_ENDED"; break
        stop_hit = _d(bar["low"]) <= stop if direction == "LONG" else _d(bar["high"]) >= stop
        target_hit = _d(bar["high"]) >= target if direction == "LONG" else _d(bar["low"]) <= target
        if stop_hit: exit_price = stop - sign * config.stop_slippage_ticks * config.price_tick; exit_index = index; reason = "STOP_FIRST_AMBIGUITY" if target_hit else "STOP"; stopped = True; break
        if target_hit: exit_price = target; exit_index = index; reason = "TARGET"; break
        exit_price = _d(bar["close"]); exit_index = index
        if stamp.hour == config.utc_force_flat_hour and stamp.minute == config.utc_force_flat_minute: reason = "UTC_FORCE_FLAT"; break
    fees = config.taker_fee_rate * (abs(entry) + abs(exit_price)); slippage = abs(entry - raw_entry) + (abs(exit_price - stop) if stopped else Decimal())
    return "TRADE_EXECUTED", {"setup_id": setup["setup_id"], "trade_id": _hash("lsmr-v2-trade-", {"setup_id": setup["setup_id"], "entry_timestamp": str(bars[entry_index]["timestamp"])}), "direction": direction, "entry_price": str(entry), "initial_stop_price": str(stop), "target_price": str(target), "exit_price": str(exit_price), "exit_reason": reason, "same_bar_policy": config.same_bar_policy, "gross_pnl": str(sign * (exit_price - raw_entry)), "fees": str(fees), "slippage_cost": str(slippage), "entry_timestamp": _time(bars[entry_index]).isoformat(), "exit_timestamp": _time(bars[exit_index]).isoformat()}


def evaluate_setups(bars: list[dict[str, Any]], config: LSMRV2Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically apply duplicate-reference and post-outcome cooldown controls."""
    proposed, events = detect_setups(bars, config); outcomes: list[dict[str, Any]] = []; trades: list[dict[str, Any]] = []
    cooldown_until = {"LONG": -1, "SHORT": -1}; terminal_references: set[tuple[str, str]] = set()
    for setup in sorted(proposed, key=lambda item: (item["sweep_index"], item["setup_id"])):
        reference_key = (setup["direction"], setup["reference"])
        if reference_key in terminal_references:
            disposition, trade = "DUPLICATE_REFERENCE_SUPPRESSED", None
        elif int(setup["sweep_index"]) <= cooldown_until[setup["direction"]]:
            disposition, trade = "COOLDOWN_BLOCKED", None
        else:
            disposition, trade = terminal_disposition(setup, bars, config)
            terminal_references.add(reference_key); cooldown_until[setup["direction"]] = int(setup["sweep_index"]) + config.cooldown_bars
            if disposition == "TRADE_EXECUTED" and trade:
                disposition, trade = simulate_trade(setup=setup, reclaim_index=trade["reclaim_index"], bars=bars, config=config)
        outcomes.append({"setup_id": setup["setup_id"], "disposition": disposition})
        setup["event_history"].append({"event": "TERMINAL", "disposition": disposition})
        events.append({"event": "TERMINAL", "setup_id": setup["setup_id"], "disposition": disposition})
        if trade: trades.append(trade)
    validate_setup_audit(proposed, events, trades, outcomes)
    return proposed, events, trades, outcomes


def validate_setup_audit(proposed: list[dict[str, Any]], events: list[dict[str, Any]], trades: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> None:
    ids = [item["setup_id"] for item in proposed]; known = set(ids)
    if len(ids) != len(known): raise AssertionError("LSMR V2 setup IDs must be unique")
    if any(not item.get("structure_id") or not item.get("direction") or "reference" not in item or "extreme" not in item or not item.get("event_history") for item in proposed): raise AssertionError("LSMR V2 setup audit fields are required")
    if any(item.get("setup_id") not in known for item in events + trades): raise AssertionError("LSMR V2 event or trade references an unknown setup")
    if len(outcomes) != len(known) or {item.get("setup_id") for item in outcomes} != known: raise AssertionError("LSMR V2 requires one terminal disposition per setup")
    if any(item.get("disposition") not in TERMINAL_DISPOSITIONS for item in outcomes): raise AssertionError("LSMR V2 terminal disposition is invalid")
    executed = {item["setup_id"] for item in outcomes if item["disposition"] == "TRADE_EXECUTED"}
    if executed != {item["setup_id"] for item in trades} or any(not item.get("trade_id") for item in trades): raise AssertionError("LSMR V2 executed setups must exactly reconcile to trades")
