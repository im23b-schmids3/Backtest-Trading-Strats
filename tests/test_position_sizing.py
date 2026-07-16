import pandas as pd
import pytest

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.models import Order
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing


def test_position_risks_two_percent_of_realized_equity():
    time = pd.Timestamp("2025-01-01", tz="UTC")
    first = Swing("low", 0, 2, 10, time, time)
    second = Swing("high", 10, 12, 20, time, time)
    setup = Setup("s", "long", first, second, levels("long", 10, 20), time)
    config = RunConfig(assets=["BTC"], leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    engine = BacktestEngine(config); engine._marks["BTC"] = 20
    engine.orders["BTC"] = Order("BTC", setup, time, 0, 0)
    engine._process_order("BTC", 0, time, pd.Series({"low": 11, "high": 12}))
    position = engine.portfolio.positions["BTC"]
    assert abs(position.entry_price - position.initial_stop) * position.quantity == pytest.approx(200)
