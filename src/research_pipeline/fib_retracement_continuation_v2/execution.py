from decimal import Decimal, ROUND_DOWN
from .ids import order_id, trade_id, exit_leg_id
from .strategy import fib_price, touch
ENTRY_RATIO=Decimal(".900"); STOP_RATIO=Decimal("1.020"); TARGETS=(Decimal(".786"),Decimal(".618"),Decimal(".500"),Decimal(".236"),Decimal(".050")); FRACTIONS=(Decimal(".30"),Decimal(".25"),Decimal(".20"),Decimal(".15"),Decimal(".10"))
def _adverse(raw,direction,kind,rate): return raw*(1+rate if (direction=="LONG") == (kind=="ENTRY") else 1-rate)
def submit_order(setup, active_timestamp, version=1):
    x={**setup,"active_timestamp":active_timestamp,"version":version}; x["order_id"]=order_id(x["setup_id"], x["setup_id"], active_timestamp); return x
def execute_order(order,bar,candidate,equity,assumptions):
    limit=fib_price(order["direction"],order["low"],order["high"],ENTRY_RATIO); stop=fib_price(order["direction"],order["low"],order["high"],STOP_RATIO)
    if not touch(order["direction"],bar,limit): return None,None
    entry=_adverse(limit,order["direction"],"ENTRY",assumptions.slippage_rate); q=((equity*assumptions.risk_fraction/abs(entry-stop))/assumptions.quantity_step).to_integral_value(rounding=ROUND_DOWN)*assumptions.quantity_step
    if q<=0:return None,"STOP_DISTANCE_REJECTED"
    fee=abs(q*entry)*assumptions.fee_rate
    return {"trade_id":trade_id(order["order_id"],order["setup_id"],bar.timestamp),"order_id":order["order_id"],"setup_id":order["setup_id"],"direction":order["direction"],"low":order["low"],"high":order["high"],"entry_timestamp":bar.timestamp,"entry_raw_price":limit,"entry_price":entry,"initial_stop_price":stop,"current_stop":stop,"quantity":q,"remaining_quantity":q,"entry_fee":fee,"fees":fee,"slippage_cost":abs(entry-limit)*q,"legs":[]},None
def _exit(trade,raw,quantity,timestamp,reason,ordinal,assumptions):
    fill=_adverse(raw,trade["direction"],"EXIT",assumptions.slippage_rate); sign=1 if trade["direction"]=="LONG" else -1; fee=abs(quantity*fill)*assumptions.fee_rate
    leg={"exit_leg_id":exit_leg_id(trade["trade_id"],trade["setup_id"],timestamp,ordinal),"trade_id":trade["trade_id"],"ordinal":ordinal,"timestamp":timestamp,"reason":reason,"quantity":quantity,"raw_price":raw,"fill_price":fill,"fee":fee,"slippage_cost":abs(fill-raw)*quantity,"gross_pnl":sign*(fill-trade["entry_price"])*quantity}
    leg["net_pnl"]=leg["gross_pnl"]-fee; trade["remaining_quantity"]-=quantity; trade["fees"]+=fee; trade["slippage_cost"]+=leg["slippage_cost"]; trade["legs"].append(leg); return leg
def process_position(trade,bar,candidate,assumptions):
    stop=trade["current_stop"]
    if (trade["direction"]=="LONG" and bar.low<=stop) or (trade["direction"]=="SHORT" and bar.high>=stop): return [_exit(trade,stop,trade["remaining_quantity"],bar.timestamp,"STOP",len(trade["legs"])+1,assumptions)]
    result=[]
    for n,(ratio,fraction) in enumerate(zip(TARGETS,FRACTIONS),1):
        if n in trade.setdefault("hit_targets", set()):
            continue
        target=fib_price(trade["direction"],trade["low"],trade["high"],ratio)
        hit=bar.high>=target if trade["direction"]=="LONG" else bar.low<=target
        if hit:
            q=min(trade["remaining_quantity"],trade["quantity"]*fraction if n<5 else trade["remaining_quantity"]); result.append(_exit(trade,target,q,bar.timestamp,f"TP{n}",len(trade["legs"])+1,assumptions))
            trade["hit_targets"].add(n)
            if n==1: trade["pending_moved_stop"]=fib_price(trade["direction"],trade["low"],trade["high"],candidate.post_tp1_ratio); trade["moved_stop_effective_after"]=bar.timestamp
            if trade["remaining_quantity"]<=0: break
    if trade.get("pending_moved_stop") is not None and bar.timestamp>trade["moved_stop_effective_after"]: trade["current_stop"]=trade.pop("pending_moved_stop")
    return result
