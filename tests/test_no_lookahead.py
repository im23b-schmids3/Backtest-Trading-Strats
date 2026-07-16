import pandas as pd

from fib_backtester.strategy.swings import confirmed_swings


def test_prefix_cannot_know_future_pivot():
    index = pd.date_range("2025-01-01", periods=7, freq="h", tz="UTC")
    full = pd.DataFrame({"open": [1]*7, "high": [1, 2, 3, 10, 3, 2, 1], "low": [0, 1, 2, 4, 2, 1, 0], "close": [1]*7, "volume": [1]*7}, index=index)
    assert not [s for s in confirmed_swings(full.iloc[:5], 2) if s.pivot_index == 3]
    assert [s for s in confirmed_swings(full, 2) if s.pivot_index == 3 and s.confirmation_index == 5]
