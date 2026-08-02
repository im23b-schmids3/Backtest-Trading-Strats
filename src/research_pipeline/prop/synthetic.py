from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import TradeSignal


def synthetic_trades(scenario: str = "profitable") -> list[TradeSignal]:
    start = datetime(2020, 1, 2, 14, tzinfo=timezone.utc)
    result: list[TradeSignal] = []
    count = 30 if scenario in {"profitable", "positive", "own-capital", "negative-economics", "noncompliant"} else 15
    for index in range(count):
        day = start + timedelta(days=index)
        if scenario in {"mll-sensitive", "billing-after-failure", "account-trading-after-failure"}:
            entry, stop, exit_price = 10000, 9000, 8000
        elif scenario == "dlg-sensitive":
            entry, stop, exit_price = 10000, 9000, 4000
        elif scenario in {"high-pass-zero-payout", "low-frequency"}:
            entry, stop, exit_price = 10000, 9000, 11000
        elif scenario == "payout-reconciliation-defect":
            entry, stop, exit_price = 10000, 9000, 11000
        else:
            entry, stop, exit_price = 10000, 9000, 11000
        market = "UNSUPPORTED" if scenario == "unsupported-mapping" else ("TEST" if scenario == "synthetic-proxy" else "BTCUSDT")
        result.append(TradeSignal(trade_id=f"synthetic-{scenario}-{index:03d}", timestamp=day, exit_timestamp=day + timedelta(minutes=30), source_market=market, timeframe="1h", direction="LONG", entry_price=entry, initial_stop_price=stop, exit_price=exit_price, source_return=(exit_price - entry) / entry, fees=1.0, slippage=0.5, trade_legs=[{"leg": 1, "fraction": 1.0}]))
    if scenario in {"profitable", "positive", "own-capital", "negative-economics", "noncompliant"}:
        # Two trades per day create five qualified winning days after pass.
        expanded: list[TradeSignal] = []
        for item in result[:20]:
            expanded.append(item)
            expanded.append(item.model_copy(update={"trade_id": item.trade_id + "-b", "timestamp": item.timestamp + timedelta(hours=1), "exit_timestamp": item.exit_timestamp + timedelta(hours=1)}))
        result = expanded
    if scenario == "low-frequency": result = result[:3]
    if scenario == "high-pass-zero-payout": result = result[:8]
    return result
