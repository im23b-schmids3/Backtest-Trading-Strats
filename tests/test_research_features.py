import pandas as pd

from fib_backtester.research.pipeline import trade_features


def test_decision_features_do_not_use_entry_candle_or_future_data():
    index = pd.date_range("2025-01-01", periods=35, freq="4h", tz="UTC")
    close = pd.Series(range(100, 135), index=index, dtype=float)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10.0}, index=index)
    trade = pd.DataFrame([{"fill_timestamp": index[25], "side": "long", "targets_hit": 1, "net_pnl": 5.0, "risk_budget": 100.0, "holding_hours": 4.0}])
    baseline = trade_features(trade, bars, "BTC", "4h")
    changed = bars.copy(); changed.loc[index[25]:, ["open", "high", "low", "close", "volume"]] = [10_000, 10_001, 9_999, 10_000, 1_000_000]
    candidate = trade_features(trade, changed, "BTC", "4h")
    for column in ("ema50_gap", "ema200_gap", "atr_pct", "rsi14", "roc10", "volume_ratio"):
        assert candidate.loc[0, column] == baseline.loc[0, column]
