import pandas as pd
import pytest

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.config import AssetConfig, RunConfig


def test_lower_timeframe_replay_requires_explicit_data():
    index = pd.date_range("2025-01-01", periods=2, freq="4h", tz="UTC")
    bars = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [.5, .5], "close": [1, 1], "volume": [1, 1]}, index=index)
    config = RunConfig(assets=["BTC"], execution_policy="lower_timeframe_replay", asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    with pytest.raises(ValueError, match="needs replay"):
        BacktestEngine(config).run({"BTC": bars})
