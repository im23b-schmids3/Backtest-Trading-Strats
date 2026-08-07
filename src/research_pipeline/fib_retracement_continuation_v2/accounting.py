from decimal import Decimal
def close_trade(trade):
    gross=sum((x["gross_pnl"] for x in trade["legs"]),Decimal()); net=gross-trade["entry_fee"]-sum((x["fee"] for x in trade["legs"]),Decimal()); risk=abs(trade["entry_price"]-trade["initial_stop_price"])*trade["quantity"]
    return {**trade,"gross_pnl":gross,"net_pnl":net,"realized_r":net/risk if risk else Decimal(),"closed":True}
