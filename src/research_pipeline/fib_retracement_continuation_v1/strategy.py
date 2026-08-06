from __future__ import annotations
from decimal import Decimal
from datetime import timedelta
from typing import Any
from .constants import ENTRY_RATIO, STOP_RATIO, TARGET_RATIOS
from .ids import setup_id, impulse_id, fib_range_id
from .models import Bar, Candidate

def d(x: Any) -> Decimal: return x if isinstance(x, Decimal) else Decimal(str(x))
def fib_price(direction: str, low: Decimal, high: Decimal, ratio: Decimal) -> Decimal:
 if low >= high: raise ValueError("FIB_RANGE_INVALID")
 return high-ratio*(high-low) if direction=="LONG" else low+ratio*(high-low)
def touch(direction: str, bar: Bar, limit: Decimal) -> bool: return bar.low<=limit if direction=="LONG" else bar.high>=limit
def create_setup(candidate: Candidate, direction: str, anchor: Bar, extreme: Bar) -> dict[str,Any]:
 anchor_price=anchor.low if direction=="LONG" else anchor.high
 extreme_price=extreme.high if direction=="LONG" else extreme.low
 sid=setup_id(candidate.candidate_id,direction,anchor.timestamp,anchor_price)
 iid=impulse_id(sid,extreme.timestamp,extreme_price)
 low,high=(anchor_price,extreme_price) if direction=="LONG" else (extreme_price,anchor_price)
 if low>=high: return {"setup_id":sid,"direction":direction,"anchor_timestamp":anchor.timestamp,"anchor_price":anchor_price,"terminal":"FIB_RANGE_INVALID"}
 return {"setup_id":sid,"impulse_id":iid,"fib_range_id":fib_range_id(iid,low,high),"direction":direction,"anchor_timestamp":anchor.timestamp,"extreme_timestamp":extreme.timestamp,"low":low,"high":high,"active_timestamp":None,"version":0}
def causal_setups(bars: list[Bar], candidate: Candidate) -> list[dict[str,Any]]:
 """active_wick_lifecycle: an anchor is promoted only by a later favorable extreme."""
 result=[]; low_anchor=None; high_anchor=None
 for bar in bars:
  if low_anchor is None or bar.low < low_anchor.low: low_anchor=bar
  if high_anchor is None or bar.high > high_anchor.high: high_anchor=bar
  if low_anchor and bar.high>low_anchor.low and (bar.high-low_anchor.low)>=candidate.min_distance and (bar.high-low_anchor.low)/low_anchor.low>=candidate.min_move:
   result.append(create_setup(candidate,"LONG",low_anchor,bar)); low_anchor=bar
  if high_anchor and bar.low<high_anchor.high and (high_anchor.high-bar.low)>=candidate.min_distance and (high_anchor.high-bar.low)/high_anchor.high>=candidate.min_move:
   result.append(create_setup(candidate,"SHORT",high_anchor,bar)); high_anchor=bar
 return result
def expire_reason(setup:dict[str,Any], bar:Bar, candidate:Candidate)->str|None:
 age=bar.timestamp-setup["anchor_timestamp"]
 if age>timedelta(days=candidate.anchor_age_days): return "ENTRY_EXPIRED"
 if setup["direction"]=="LONG" and bar.low<setup["low"]: return "ENTRY_EXPIRED"
 if setup["direction"]=="SHORT" and bar.high>setup["high"]: return "ENTRY_EXPIRED"
 return None
