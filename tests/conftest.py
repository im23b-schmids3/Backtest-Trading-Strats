import pandas as pd


def ohlcv(highs, lows=None):
    lows = lows if lows is not None else [value - 1 for value in highs]
    index = pd.date_range("2025-01-01", periods=len(highs), freq="h", tz="UTC")
    return pd.DataFrame({"open": lows, "high": highs, "low": lows, "close": highs, "volume": 1.0}, index=index)
