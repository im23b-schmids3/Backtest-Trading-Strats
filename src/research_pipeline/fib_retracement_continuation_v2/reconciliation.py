from decimal import Decimal
def reconcile(setups,outcomes,orders,trades,opening_equity,*,final_equity=None,events=None):
    setup_ids=[x.get("setup_id") for x in setups]; outcome_ids=[x.get("setup_id") for x in outcomes]; order_ids={x.get("order_id") for x in orders}; trade_ids={x.get("trade_id") for x in trades}
    unique=len(setup_ids)==len(set(setup_ids)) and None not in setup_ids; one=unique and len(outcome_ids)==len(set(outcome_ids)) and set(setup_ids)==set(outcome_ids)
    quantity=all(t["quantity"]==sum((x["quantity"] for x in t["legs"]),Decimal())+t["remaining_quantity"] for t in trades); gross=all(t["gross_pnl"]==sum((x["gross_pnl"] for x in t["legs"]),Decimal()) for t in trades); net=all(t["net_pnl"]==t["gross_pnl"]-t["entry_fee"]-sum((x["fee"] for x in t["legs"]),Decimal()) for t in trades)
    expected=opening_equity+sum((t["net_pnl"] for t in trades),Decimal()); equity=expected==(expected if final_equity is None else final_equity); links=all(x.get("setup_id") in setup_ids for x in orders) and all(x.get("order_id") in order_ids for x in trades)
    checks={"unique_setup_ids":unique,"one_terminal_outcome":one,"order_parent_links":links,"trade_parent_links":links,"quantity_conservation":quantity,"gross_pnl_reconcile":gross,"pnl_reconcile":net,"equity_reconcile":equity,"event_links":all(e.get("setup_id") in setup_ids for e in (events or []))}
    return {"reconciles":all(checks.values()),**checks,"final_equity":expected if final_equity is None else final_equity,"expected_final_equity":expected}
