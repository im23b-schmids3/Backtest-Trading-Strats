import pandas as pd

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.signals import setup_from_swings
from fib_backtester.strategy.swings import Swing


def test_order_submission_is_next_candle_open_after_confirmation():
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": [1]*8, "high": [2]*8, "low": [.5]*8, "close": [1]*8, "volume": [1]*8}, index=index)
    config = RunConfig(assets=["BTC"], min_pivot_distance=5, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    engine = BacktestEngine(config); engine._bars = {"BTC": bars}; engine._prior_swings = {"BTC": []}
    low = Swing("low", 0, 1, 10, index[0], index[1]); high = Swing("high", 5, 6, 20, index[5], index[6])
    engine._swings_by_confirmation = {"BTC": {1: [low], 6: [high]}}
    engine._publish_setups("BTC", 1, index[1]); engine._publish_setups("BTC", 6, index[6])
    assert engine.orders["BTC"].submission_time == index[7]
    assert engine.orders["BTC"].active_from_index == 7
