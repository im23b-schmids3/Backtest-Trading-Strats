from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

from .models import LSMRConfig, TERMINAL_DISPOSITIONS

def _d(value: Any) -> Decimal: return Decimal(str(value))
def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values); middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
def _time(bar: dict[str, Any]) -> datetime:
    value = bar["timestamp"]
    if isinstance(value, datetime): return value.astimezone(timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
def setup_id(*, direction: str, sweep_timestamp: Any, reference_level: Any) -> str:
    payload = json.dumps({"direction": direction, "sweep_timestamp": str(sweep_timestamp), "reference_level": str(reference_level)}, sort_keys=True, separators=(",", ":"))
    return "lsmr-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:20]

def validate_setup_audit(proposed: list[dict[str, Any]], events: list[dict[str, Any]], trades: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> None:
    ids = [item["setup_id"] for item in proposed]
    known = set(ids)
    if len(ids) != len(known): raise AssertionError("LSMR setup IDs must be unique")
    if any(item.get("setup_id") not in known for item in events + trades): raise AssertionError("LSMR event or trade references an unknown setup")
    if len(outcomes) != len(known) or {item.get("setup_id") for item in outcomes} != known: raise AssertionError("LSMR requires one terminal disposition per setup")
    if any(item.get("disposition") not in TERMINAL_DISPOSITIONS for item in outcomes): raise AssertionError("LSMR terminal disposition is invalid")
    executed = {item["setup_id"] for item in outcomes if item["disposition"] == "EXECUTED"}
    if executed != {item["setup_id"] for item in trades}: raise AssertionError("LSMR executed setups must exactly reconcile to trades")

def detect_setups(bars: list[dict[str, Any]], config: LSMRConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect sealed sweep/reclaim proposals without executing a trade."""
    proposed: list[dict[str, Any]] = []; events: list[dict[str, Any]] = []
    for index in range(max(config.reference_bars, 24), len(bars)):
        bar = bars[index]; prior = bars[index-config.reference_bars:index]
        for direction, level in (("LONG", min(_d(x["low"]) for x in prior)), ("SHORT", max(_d(x["high"]) for x in prior))):
            penetration = max(level * config.penetration_fraction, config.price_tick * 2)
            swept = _d(bar["low"]) <= level - penetration if direction == "LONG" else _d(bar["high"]) >= level + penetration
            if swept:
                item = {"setup_id": setup_id(direction=direction, sweep_timestamp=bar["timestamp"], reference_level=level), "direction": direction, "sweep_index": index, "reference_level": str(level), "sweep_extreme": str(_d(bar["low"]) if direction == "LONG" else _d(bar["high"])), "source_structure": {"reference_start_index": index-config.reference_bars, "reference_end_index": index-1, "reference_level": str(level)}}
                proposed.append(item); events.append({"event": "PROPOSED_SETUP", "setup_id": item["setup_id"], "bar_index": index})
    return proposed, events

def terminal_disposition(setup: dict[str, Any], bars: list[dict[str, Any]], config: LSMRConfig) -> tuple[str, dict[str, Any] | None]:
    sweep = int(setup["sweep_index"]); level = _d(setup["reference_level"]); direction = setup["direction"]
    extreme = _d(setup["sweep_extreme"])
    for index in range(sweep, min(sweep + config.reclaim_window_bars, len(bars))):
        bar = bars[index]; body = abs(_d(bar["close"]) - _d(bar["open"])); spread = _d(bar["high"]) - _d(bar["low"])
        extreme = min(extreme, _d(bar["low"])) if direction == "LONG" else max(extreme, _d(bar["high"]))
        volume_history = [_d(x["volume"]) for x in bars[max(0, index-10):index]]
        reclaimed = (_d(bar["close"]) > level and _d(bar["close"]) > _d(bar["open"])) if direction == "LONG" else (_d(bar["close"]) < level and _d(bar["close"]) < _d(bar["open"]))
        if not reclaimed: continue
        if not volume_history or body < spread * config.body_fraction or _d(bar["volume"]) < _median(volume_history) * config.volume_multiple: continue
        if abs(_d(bar["daily_vwap"]) - _d(bars[index-24]["daily_vwap"])) / _d(bar["close"]) > config.maximum_vwap_slope_fraction: return "REGIME_REJECTED", None
        if _time(bar).hour == config.utc_force_flat_hour and _time(bar).minute == config.utc_force_flat_minute: return "SESSION_ENDED", None
        if index + 1 >= len(bars): return "NO_EXECUTABLE_ENTRY", None
        if _time(bars[index+1]).date() != _time(bar).date(): return "SESSION_ENDED", None
        entry = _d(bars[index+1]["open"]); stop = extreme - config.stop_buffer_ticks * config.price_tick if direction == "LONG" else extreme + config.stop_buffer_ticks * config.price_tick
        distance = abs(entry-stop)
        if distance < entry * config.minimum_stop_fraction or distance > entry * config.maximum_stop_fraction: return "STOP_DISTANCE_REJECTED", None
        return "EXECUTED", {"setup_id": setup["setup_id"], "direction": direction, "reclaim_index": index, "entry_timestamp": _time(bars[index+1]).isoformat(), "entry_price": str(entry), "initial_stop_price": str(stop), "target_price": str(entry + config.target_r_multiple*distance if direction == "LONG" else entry-config.target_r_multiple*distance)}
    return "RECLAIM_WINDOW_EXPIRED", None

def simulate_trade(*, setup: dict[str, Any], reclaim_index: int, bars: list[dict[str, Any]], config: LSMRConfig) -> tuple[str, dict[str, Any] | None]:
    """Pure sealed execution primitive: next-open, stop-first, time/session exits and costs."""
    entry_index=reclaim_index+1
    if entry_index >= len(bars) or _time(bars[entry_index]).date() != _time(bars[reclaim_index]).date(): return "NO_EXECUTABLE_ENTRY", None
    direction=setup["direction"]; sign=Decimal(1) if direction=="LONG" else Decimal(-1)
    raw_entry=_d(bars[entry_index]["open"]); entry=raw_entry + sign*config.market_slippage_ticks*config.price_tick
    extreme=_d(setup["sweep_extreme"]); stop=extreme-config.stop_buffer_ticks*config.price_tick if direction=="LONG" else extreme+config.stop_buffer_ticks*config.price_tick
    risk=abs(entry-stop)
    if risk < entry*config.minimum_stop_fraction or risk > entry*config.maximum_stop_fraction: return "STOP_DISTANCE_REJECTED", None
    target=entry + sign*config.target_r_multiple*risk; exit_price=_d(bars[entry_index]["close"]); exit_index=entry_index; reason="TIME_STOP"; stopped=False
    for index in range(entry_index, min(len(bars), entry_index+config.time_stop_bars)):
        bar=bars[index]; stamp=_time(bar)
        if stamp.date()!=_time(bars[entry_index]).date(): reason="SESSION_ENDED"; break
        stop_hit=(_d(bar["low"])<=stop) if direction=="LONG" else (_d(bar["high"])>=stop)
        target_hit=(_d(bar["high"])>=target) if direction=="LONG" else (_d(bar["low"])<=target)
        if stop_hit: exit_price=stop-sign*config.stop_slippage_ticks*config.price_tick; exit_index=index; reason="STOP_FIRST_AMBIGUITY" if target_hit else "STOP"; stopped=True; break
        if target_hit: exit_price=target; exit_index=index; reason="TARGET"; break
        exit_price=_d(bar["close"]); exit_index=index
        if stamp.hour==config.utc_force_flat_hour and stamp.minute==config.utc_force_flat_minute: reason="UTC_FORCE_FLAT"; break
    gross=sign*(exit_price-raw_entry); fees=config.taker_fee_rate*(abs(entry)+abs(exit_price)); slippage=abs(entry-raw_entry)+ (abs(exit_price-stop) if stopped else Decimal())
    return "EXECUTED", {"setup_id":setup["setup_id"],"direction":direction,"entry_price":str(entry),"initial_stop_price":str(stop),"target_price":str(target),"exit_price":str(exit_price),"exit_reason":reason,"same_bar_policy":config.same_bar_policy,"gross_pnl":str(gross),"fees":str(fees),"slippage_cost":str(slippage),"net_pnl":str(gross-fees-slippage),"net_r":str((gross-fees-slippage)/risk),"entry_timestamp":_time(bars[entry_index]).isoformat(),"exit_timestamp":_time(bars[exit_index]).isoformat()}
