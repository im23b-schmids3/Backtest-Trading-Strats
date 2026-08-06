from __future__ import annotations
from decimal import Decimal
def close_trade(trade:dict)->dict:
 gross=sum((x["gross_pnl"] for x in trade["legs"]),Decimal())
 net=gross-trade["entry_fee"]-sum((x["fee"] for x in trade["legs"]),Decimal())
 risk=abs(trade["entry_price"]-trade["initial_stop_price"])*trade["quantity"]
 return {**trade,"gross_pnl":gross,"net_pnl":net,"realized_r":net/risk if risk else Decimal(),"closed":True}
def compounded_equity(opening:Decimal,trades:list[dict])->Decimal: return opening+sum((x["net_pnl"] for x in trades),Decimal())
