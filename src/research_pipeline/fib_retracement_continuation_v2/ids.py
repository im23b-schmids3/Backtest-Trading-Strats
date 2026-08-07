"""V2 lifecycle identities include all economic and timing provenance."""
import hashlib, json
from datetime import datetime
from decimal import Decimal
def _canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=lambda x: format(x,"f") if isinstance(x,Decimal) else x.isoformat().replace("+00:00","Z") if isinstance(x,datetime) else (_ for _ in ()).throw(TypeError()))
def ident(kind, **value): return kind + "-" + hashlib.sha256(_canon(value).encode()).hexdigest()[:16]
def setup_id(candidate, symbol, timeframe, direction, anchor, extreme): return ident("setup",candidate=candidate,symbol=symbol,signal_timeframe=timeframe,execution_timeframe="1m",direction=direction,anchor=anchor,impulse_extreme=extreme)
def order_id(setup, candidate, timestamp): return ident("order",setup_id=setup,candidate=candidate,event_timestamp=timestamp)
def trade_id(order, candidate, timestamp): return ident("trade",order_id=order,candidate=candidate,event_timestamp=timestamp)
def exit_leg_id(trade, candidate, timestamp, ordinal): return ident("exit",trade_id=trade,candidate=candidate,event_timestamp=timestamp,ordinal=ordinal)
