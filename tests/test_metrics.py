import pandas as pd

from fib_backtester.backtest.metrics import calculate_metrics


def test_sharpe_frequency_scales_with_bar_interval():
    daily_index = pd.date_range("2025-01-01", periods=4, freq="d", tz="UTC")
    hourly_index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    values = [100, 101, 99, 102]
    daily = pd.DataFrame({"timestamp": daily_index, "equity": values})
    hourly = pd.DataFrame({"timestamp": hourly_index, "equity": values})
    assert calculate_metrics(pd.DataFrame(), hourly, 100)["sharpe_ratio"] > calculate_metrics(pd.DataFrame(), daily, 100)["sharpe_ratio"]
