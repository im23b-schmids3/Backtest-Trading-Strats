from __future__ import annotations
import hashlib, json
from decimal import Decimal
from datetime import datetime
from typing import Any
def canonical(value: Any) -> str:
 def default(x: Any):
  if isinstance(x, Decimal): return format(x, "f")
  if isinstance(x, datetime): return x.isoformat().replace("+00:00", "Z")
  raise TypeError(type(x).__name__)
 return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)
def ident(prefix: str, payload: dict[str, Any]) -> str: return prefix + hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
def setup_id(candidate, direction, anchor_timestamp, anchor_price, extreme_timestamp=None, extreme_price=None):
 """Stable identity for one proposed anchor-to-extreme setup.

 The impulse coordinates are optional only to preserve the public helper's
 backwards-compatible call shape.  Production setup creation always supplies
 them, so distinct impulses from one anchor cannot share a lifecycle ID.
 """
 payload={"candidate":candidate,"direction":direction,"anchor_timestamp":anchor_timestamp,"anchor_price":anchor_price}
 if extreme_timestamp is not None: payload["extreme_timestamp"]=extreme_timestamp
 if extreme_price is not None: payload["extreme_price"]=extreme_price
 return ident("setup-", payload)
def impulse_id(setup, extreme_timestamp, extreme_price): return ident("impulse-", {"setup_id":setup,"extreme_timestamp":extreme_timestamp,"extreme_price":extreme_price})
def fib_range_id(impulse, low, high): return ident("fib-", {"impulse_id":impulse,"low":low,"high":high})
def order_id(fib, version, active_timestamp): return ident("order-", {"fib_range_id":fib,"version":version,"active_timestamp":active_timestamp})
def trade_id(order, entry_fill_timestamp): return ident("trade-", {"order_id":order,"entry_fill_timestamp":entry_fill_timestamp})
def exit_leg_id(trade, ordinal): return ident("exit-", {"trade_id":trade,"leg_ordinal":ordinal})
def event_id(candidate, timestamp, kind, subject_id, ordinal): return ident("event-", {"candidate":candidate,"timestamp":timestamp,"event_kind":kind,"subject_id":subject_id,"ordinal":ordinal})
