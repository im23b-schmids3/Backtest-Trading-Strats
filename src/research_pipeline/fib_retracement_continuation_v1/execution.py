from __future__ import annotations
from decimal import Decimal, ROUND_DOWN
from typing import Any
from .constants import ENTRY_RATIO, STOP_RATIO, TARGET_RATIOS, TARGET_FRACTIONS
from .strategy import fib_price, touch, d
from .ids import order_id, trade_id, exit_leg_id
from .models import Bar, Candidate, ExecutionAssumptions

def adverse(raw:Decimal,direction:str,kind:str, slip:Decimal)->Decimal:
 # Long entry/short exit worsen upwards; short entry/long exit worsen downwards.
 up=(direction=="LONG" and kind=="ENTRY") or (direction=="SHORT" and kind=="EXIT")
 return raw*(Decimal(1)+slip) if up else raw*(Decimal(1)-slip)
def floor_step(q:Decimal, step:Decimal)->Decimal: return (q/step).to_integral_value(rounding=ROUND_DOWN)*step
def submit_order(setup:dict[str,Any], active_timestamp, version:int=1)->dict[str,Any]:
 setup={**setup,"active_timestamp":active_timestamp,"version":version}; setup["order_id"]=order_id(setup["fib_range_id"],version,active_timestamp); return setup
def execute_order(order:dict[str,Any], bar:Bar, candidate:Candidate, equity:Decimal, assumptions:ExecutionAssumptions)->tuple[dict[str,Any]|None,str|None]:
 if bar.timestamp < order["active_timestamp"]: return None,None
 direction=order["direction"]; limit=fib_price(direction,order["low"],order["high"],ENTRY_RATIO); stop=fib_price(direction,order["low"],order["high"],STOP_RATIO)
 if not touch(direction,bar,limit): return None,None
 entry=adverse(limit,direction,"ENTRY",assumptions.slippage_rate); quantity=floor_step(equity*assumptions.risk_fraction/abs(entry-stop), assumptions.quantity_step)
 if quantity<=0: return None,"STOP_DISTANCE_REJECTED"
 fee=abs(quantity*entry)*assumptions.fee_rate
 return {"trade_id":trade_id(order["order_id"],bar.timestamp),"order_id":order["order_id"],"setup_id":order["setup_id"],"direction":direction,"low":order["low"],"high":order["high"],"entry_timestamp":bar.timestamp,"entry_raw_price":limit,"entry_price":entry,"initial_stop_price":stop,"current_stop":stop,"quantity":quantity,"remaining_quantity":quantity,"entry_fee":fee,"fees":fee,"slippage_cost":abs(entry-limit)*quantity,"legs":[],"tp1_filled":False},None
def _exit(trade:dict[str,Any], raw:Decimal, quantity:Decimal, timestamp, reason:str, ordinal:int, assumptions:ExecutionAssumptions)->dict[str,Any]:
 fill=adverse(raw,trade["direction"],"EXIT",assumptions.slippage_rate); sign=Decimal(1) if trade["direction"]=="LONG" else Decimal(-1); fee=abs(quantity*fill)*assumptions.fee_rate
 leg={"exit_leg_id":exit_leg_id(trade["trade_id"],ordinal),"trade_id":trade["trade_id"],"ordinal":ordinal,"timestamp":timestamp,"reason":reason,"quantity":quantity,"raw_price":raw,"fill_price":fill,"fee":fee,"slippage_cost":abs(fill-raw)*quantity,"gross_pnl":sign*(fill-trade["entry_price"])*quantity,"net_pnl":sign*(fill-trade["entry_price"])*quantity-fee}
 trade["remaining_quantity"]-=quantity; trade["fees"]+=fee; trade["slippage_cost"]+=leg["slippage_cost"]; trade["legs"].append(leg); return leg
def process_position(trade:dict[str,Any], bar:Bar, candidate:Candidate, assumptions:ExecutionAssumptions, *, allow_moved_stop:bool=True)->list[dict[str,Any]]:
 """Current stop comes first. TP1 moved stop is installed for following bars only."""
 direction=trade["direction"]; stop=trade["current_stop"]; stop_hit=bar.low<=stop if direction=="LONG" else bar.high>=stop
 if stop_hit: return [_exit(trade,stop,trade["remaining_quantity"],bar.timestamp,"STOP",len(trade["legs"])+1,assumptions)]
 result=[]
 for index,(ratio,fraction) in enumerate(zip(TARGET_RATIOS,TARGET_FRACTIONS),1):
  if any(x["reason"]==f"TP{index}" for x in trade["legs"]): continue
  target=fib_price(direction,trade.get("low",Decimal(0)),trade.get("high",Decimal(0)),ratio) if "low" in trade else None
  if target is None: continue
  hit=bar.high>=target if direction=="LONG" else bar.low<=target
  if hit:
   q=trade["quantity"]*fraction if index<5 else trade["remaining_quantity"]
   q=min(q,trade["remaining_quantity"]); result.append(_exit(trade,target,q,bar.timestamp,f"TP{index}",len(trade["legs"])+1,assumptions))
   if index==1: trade["pending_moved_stop"]=fib_price(direction,trade["low"],trade["high"],candidate.post_tp1_ratio); trade["moved_stop_effective_after"]=bar.timestamp
   if trade["remaining_quantity"]<=0: break
 if allow_moved_stop and trade.get("pending_moved_stop") is not None and bar.timestamp>trade["moved_stop_effective_after"]: trade["current_stop"]=trade.pop("pending_moved_stop")
 return result
