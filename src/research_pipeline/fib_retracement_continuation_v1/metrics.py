from __future__ import annotations
from decimal import Decimal
from .constants import EVIDENCE_LABELS
def evidence_label(count:int)->str:
 for low,high,label in EVIDENCE_LABELS:
  if count>=low and (high is None or count<=high): return label
 return "LOW_FREQUENCY_DEVELOPMENT_EVIDENCE"
def metrics(trades:list[dict], opening_equity:Decimal)->dict:
 nets=[x["net_pnl"] for x in trades]; positives=sum((x for x in nets if x>0),Decimal()); losses=-sum((x for x in nets if x<0),Decimal())
 equity=opening_equity; peak=equity; dd=Decimal()
 for value in nets: equity+=value; peak=max(peak,equity); dd=max(dd,(peak-equity)/peak*100 if peak else Decimal())
 return {"executed_trade_count":len(trades),"net_pnl":sum(nets,Decimal()),"profit_factor":positives/losses if losses else Decimal("Infinity"),"average_net_r":sum((x["realized_r"] for x in trades),Decimal())/len(trades) if trades else Decimal(),"maximum_drawdown_percent":dd,"final_equity":equity,"evidence_label":evidence_label(len(trades))}
def gates(value:dict, trades:list[dict], reconciles:bool, extra_slippage:Decimal=Decimal(".0002"))->dict:
 net=value["net_pnl"]; stress=net-sum((abs(t["entry_price"])*t["quantity"]*extra_slippage+sum((abs(x["fill_price"])*x["quantity"]*extra_slippage for x in t["legs"]),Decimal()) for t in trades),Decimal()); best=max((x["net_pnl"] for x in trades),default=Decimal())
 checks={"positive_net_after_costs":net>0,"profit_factor_at_least_1_30":value["profit_factor"]>=Decimal("1.30"),"positive_average_net_r":value["average_net_r"]>0,"maximum_drawdown_at_most_20_percent":value["maximum_drawdown_percent"]<=20,"additional_slippage_positive":stress>0,"best_trade_removal_positive":net-best>0,"full_reconciliation":reconciles}
 return {"passed":all(checks.values()),"hard_gates":checks,"evidence_label":value["evidence_label"],"additional_slippage_net_pnl":stress,"best_trade_removal_net_pnl":net-best}
