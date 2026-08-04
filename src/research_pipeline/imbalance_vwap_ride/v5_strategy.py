"""Literal V5 execution primitives for locally materialized price-scaled zones.

The study runner deliberately does not invoke these primitives in this offline
implementation node.  Keeping them here makes the execution contract testable
without downloading a single aggregate trade or placing any order.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

from .strategy import _d, _quantize, _ts
from .v5_models import ImbalanceVWAPRideV5Config


def simulate_v5_long_trade(*, zone: dict[str, Any], signal_bar: dict[str, Any], entry_index: int,
                           bars: list[dict[str, Any]], config: ImbalanceVWAPRideV5Config) -> tuple[str, dict[str, Any] | None]:
    """Apply next-bar, stop-first, UTC-flat V5 execution using the zone's bin size."""
    if zone.get("direction", "LONG") != "LONG":
        raise ValueError("V5 rejects non-long zones")
    if entry_index >= len(bars):
        return "NO_EXECUTABLE_ENTRY", None
    signal_time = _ts(signal_bar["bar_start_utc"])
    entry_bar = bars[entry_index]
    entry_time = _ts(entry_bar["bar_start_utc"])
    if entry_time != signal_time + timedelta(minutes=5) or entry_time.date() != signal_time.date():
        return "NO_EXECUTABLE_ENTRY", None
    size = _d(zone["bin_size_usd"])
    if size <= 0:
        return "INVALID_ZONE_BIN_SIZE", None
    quantity = (config.quantity_btc / config.quantity_step).to_integral_value(rounding=ROUND_FLOOR) * config.quantity_step
    if quantity < config.minimum_quantity:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None
    reference_entry = _d(entry_bar["open"])
    entry = _quantize(reference_entry + config.market_slippage_ticks * config.price_tick, config.price_tick, ROUND_CEILING)
    stop = _quantize(_d(zone["bottom"]) - config.stop_buffer_bins * size, config.price_tick, ROUND_FLOOR)
    risk = entry - stop
    if risk <= 0:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None
    target = _quantize(entry + config.target_r_multiple * risk, config.price_tick, ROUND_CEILING)
    day = entry_time.date()
    reference_exit = _d(entry_bar["close"]); exit_price = reference_exit - config.market_slippage_ticks * config.price_tick
    exit_bar = entry_bar; exit_reason = "UTC_DAY_FORCE_FLAT"; ambiguity = False
    for bar in bars[entry_index:]:
        if _ts(bar["bar_start_utc"]).date() != day: break
        hit_stop, hit_target = _d(bar["low"]) <= stop, _d(bar["high"]) >= target
        if hit_stop: # Contractually before target on an ambiguous bar.
            reference_exit = stop; exit_price = stop - config.stop_slippage_ticks * config.price_tick
            exit_bar = bar; exit_reason = "STOP_FIRST_AMBIGUITY" if hit_target else "STOP"; ambiguity = hit_target; break
        if hit_target:
            reference_exit = exit_price = target; exit_bar = bar; exit_reason = "TARGET"; break
        reference_exit = _d(bar["close"]); exit_price = reference_exit - config.market_slippage_ticks * config.price_tick; exit_bar = bar
    fees = config.taker_fee_rate * (entry + exit_price) * quantity
    slippage = abs(entry-reference_entry)*quantity + abs(exit_price-reference_exit)*quantity
    gross = (reference_exit-reference_entry)*quantity; net = gross-fees-slippage
    trade = {"setup_id":zone.get("setup_id"), "direction":"LONG", "source_bar_bin_size_usd":str(size), "bin_size_usd":str(size), "entry_price":str(entry), "initial_stop_price":str(stop), "target_price":str(target), "actual_risk_distance":str(risk), "gross_pnl":str(gross), "net_pnl":str(net), "fees":str(fees), "slippage_cost":str(slippage), "total_costs":str(fees+slippage), "gross_r":str(gross/(risk*quantity)), "net_r":str(net/(risk*quantity)), "exit_reason":exit_reason, "same_bar_ambiguity":ambiguity, "entry_timestamp":entry_time.isoformat(), "exit_timestamp":_ts(exit_bar["bar_end_utc"]).isoformat(), "quantity_btc":str(quantity), "candidate_id":config.candidate_id}
    return "TRADE_EXECUTED", trade
