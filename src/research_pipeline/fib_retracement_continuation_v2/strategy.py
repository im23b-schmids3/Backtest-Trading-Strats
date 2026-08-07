from datetime import timedelta
from decimal import Decimal
from .ids import setup_id
def fib_price(direction, low, high, ratio):
    if low >= high: raise ValueError("FIB_RANGE_INVALID")
    return high-ratio*(high-low) if direction == "LONG" else low+ratio*(high-low)
def touch(direction, bar, limit): return bar.low <= limit if direction == "LONG" else bar.high >= limit
def causal_setups(bars, candidate):
    result=[]; low_anchor=high_anchor=None
    for extreme in bars:
        if low_anchor is None or extreme.low < low_anchor.low: low_anchor=extreme
        if high_anchor is None or extreme.high > high_anchor.high: high_anchor=extreme
        for direction, anchor, aprice, eprice in (("LONG",low_anchor,low_anchor.low,extreme.high),("SHORT",high_anchor,high_anchor.high,extreme.low)):
            span=abs(eprice-aprice)
            if span >= candidate.min_distance and span/aprice >= candidate.min_move:
                sid=setup_id(candidate.candidate_id,candidate.symbol,candidate.timeframe,direction,anchor.timestamp,extreme.timestamp)
                low,high=sorted((aprice,eprice))
                result.append({"setup_id":sid,"fib_range_id":sid,"direction":direction,"anchor_timestamp":anchor.timestamp,"extreme_timestamp":extreme.timestamp,"low":low,"high":high,"active_timestamp":None,"version":0})
                if direction == "LONG": low_anchor=extreme
                else: high_anchor=extreme
    return result
def expire_reason(setup, bar, candidate):
    if bar.timestamp-setup["anchor_timestamp"] > timedelta(days=candidate.anchor_age_days): return "ENTRY_EXPIRED"
    if setup["direction"] == "LONG" and bar.low < setup["low"] or setup["direction"] == "SHORT" and bar.high > setup["high"]: return "ENTRY_EXPIRED"
