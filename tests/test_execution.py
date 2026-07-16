import pandas as pd

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.models import Position
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing


def _setup():
    time = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 2, 10, time, time + pd.Timedelta(hours=2))
    high = Swing("high", 10, 12, 20, time + pd.Timedelta(hours=10), time + pd.Timedelta(hours=12))
    return Setup("test", "long", low, high, levels("long", 10, 20), high.confirmation_time)


def _engine(policy="conservative"):
    config = RunConfig(assets=["BTC"], execution_policy=policy, leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    return BacktestEngine(config)


def test_conservative_stop_wins_when_stop_and_target_share_bar():
    engine, setup = _engine(), _setup()
    position = Position("BTC", setup, 100, setup.fib.entry, setup.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"), pd.Timestamp("2025-01-02", tz="UTC"), 200, setup.fib.stop, setup.fib.stop, 100, 0, 0)
    engine.portfolio.positions["BTC"] = position
    engine.portfolio.reserved_notional = position.quantity * position.entry_price
    engine._process_position("BTC", pd.Timestamp("2025-01-03", tz="UTC"), pd.Series({"low": 9.7, "high": 12.2}))
    assert not engine.portfolio.positions
    assert engine.portfolio.closed[0]["exit_reason"] == "stop_before_tp1"


def test_tp1_moves_stop_and_reduces_position():
    engine, setup = _engine(), _setup()
    position = Position("BTC", setup, 100, setup.fib.entry, setup.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"), pd.Timestamp("2025-01-02", tz="UTC"), 200, setup.fib.stop, setup.fib.stop, 100, 0, 0)
    engine.portfolio.positions["BTC"] = position
    engine.portfolio.reserved_notional = position.quantity * position.entry_price
    engine._process_position("BTC", pd.Timestamp("2025-01-03", tz="UTC"), pd.Series({"low": 11.3, "high": 12.2}))
    assert position.remaining == 85
    assert position.current_stop == setup.fib.post_tp1_stop
    assert position.target_done[0]
