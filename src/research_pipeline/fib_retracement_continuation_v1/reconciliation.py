from __future__ import annotations
from decimal import Decimal
def reconcile(setups:list[dict], outcomes:list[dict], orders:list[dict], trades:list[dict], opening_equity:Decimal)->dict:
 setup_ids=[x["setup_id"] for x in setups]; outcome_ids=[x.get("setup_id") for x in outcomes]
 one_terminal=len(setup_ids)==len(outcomes)==len(set(outcome_ids)) and set(setup_ids)==set(outcome_ids)
 executed={x["setup_id"] for x in outcomes if x.get("disposition")=="TRADE_EXECUTED"}
 trade_ids={x.get("setup_id") for x in trades}; quantities=all(t["quantity"]==sum((x["quantity"] for x in t["legs"]),Decimal())+t["remaining_quantity"] for t in trades)
 pnl=all(t["net_pnl"]==t["gross_pnl"]-t["entry_fee"]-sum((x["fee"] for x in t["legs"]),Decimal()) for t in trades)
 final=opening_equity+sum((x["net_pnl"] for x in trades),Decimal())
 ok=one_terminal and executed==trade_ids and quantities and pnl
 return {"reconciles":ok,"one_terminal_outcome":one_terminal,"trade_outcomes_reconcile":executed==trade_ids,"quantity_conservation":quantities,"pnl_reconcile":pnl,"final_equity":final}
