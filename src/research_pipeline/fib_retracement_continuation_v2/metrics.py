from decimal import Decimal
def metrics(trades,opening_equity):
    nets=[x["net_pnl"] for x in trades]; wins=sum((x for x in nets if x>0),Decimal()); losses=-sum((x for x in nets if x<0),Decimal()); equity=opening_equity; peak=equity; dd=Decimal()
    for net in nets: equity+=net; peak=max(peak,equity); dd=max(dd,(peak-equity)/peak*100 if peak else Decimal())
    return {"executed_trade_count":len(trades),"net_pnl":sum(nets,Decimal()),"profit_factor":wins/losses if losses else Decimal("Infinity"),"average_net_r":sum((x["realized_r"] for x in trades),Decimal())/len(trades) if trades else Decimal(),"maximum_drawdown_percent":dd,"final_equity":equity,"evidence_label":"LOW_FREQUENCY_DEVELOPMENT_EVIDENCE" if len(trades)<30 else "MODERATE_DEVELOPMENT_EVIDENCE" if len(trades)<60 else "FULL_DEVELOPMENT_GATE_ELIGIBILITY"}
def gates(value,trades,reconciles):
    net=value["net_pnl"]; best=max((x["net_pnl"] for x in trades),default=Decimal()); checks={"positive_net_after_costs":net>0,"profit_factor_at_least_1_30":value["profit_factor"]>=Decimal("1.30"),"positive_average_net_r":value["average_net_r"]>0,"maximum_drawdown_at_most_20_percent":value["maximum_drawdown_percent"]<=20,"best_trade_removal_positive":net-best>0,"full_reconciliation":reconciles}
    return {"passed":all(checks.values()),"hard_gates":checks,"evidence_label":value["evidence_label"]}
