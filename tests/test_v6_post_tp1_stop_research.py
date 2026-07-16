import pandas as pd
import pytest

from fib_backtester.backtest.models import Position
from fib_backtester.backtest.v6_post_tp1_stop_engine import StrategyV6PostTP1StopResearchEngine, _fib_price
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing
from fib_backtester.strategy.v4_take_profit_research import PROFILES, profile_levels
from fib_backtester.strategy.v6_post_tp1_stop_research import POST_TP1_POLICIES


def _setup():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 1, 10, t, t + pd.Timedelta(hours=1))
    high = Swing("high", 5, 6, 20, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    return Setup("test", "long", low, high, levels("long", 10, 20), high.confirmation_time)


def _config():
    return RunConfig(assets=["BTC"], timeframes=["1h"], min_pivot_distance=4,
                     asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})


@pytest.mark.parametrize("policy,ratio", [("A", .880), ("B", .900), ("C", .786), ("D", None), ("E", .618)])
def test_every_policy_has_exact_post_tp1_rule(policy, ratio):
    assert POST_TP1_POLICIES[policy].fib_ratio == ratio


def test_same_candle_stop_touch_cannot_retroactively_close_after_tp1():
    setup = profile_levels(_setup(), PROFILES["B"], .900)
    engine = StrategyV6PostTP1StopResearchEngine(_config(), .01, POST_TP1_POLICIES["C"])
    engine.portfolio.positions["BTC"] = Position(
        "BTC", setup, 1.0, setup.fib.entry, setup.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"), 100.0, setup.fib.stop, setup.fib.stop, 1.0, 0.0, 0.0,
    )
    timestamp = pd.Timestamp("2025-01-02", tz="UTC")
    bar = pd.Series({"open": 11.0, "high": 12.2, "low": 12.0, "close": 12.1, "volume": 1.0})
    engine._process_position("BTC", timestamp, bar)
    assert engine.portfolio.positions["BTC"].target_done[0]
    assert engine.portfolio.positions["BTC"].current_stop == pytest.approx(_fib_price("long", 10, 20, .786))
    assert not engine.portfolio.closed


def test_no_stop_movement_keeps_initial_stop_after_tp1():
    setup = profile_levels(_setup(), PROFILES["B"], .900)
    engine = StrategyV6PostTP1StopResearchEngine(_config(), .01, POST_TP1_POLICIES["D"])
    engine.portfolio.positions["BTC"] = Position(
        "BTC", setup, 1.0, setup.fib.entry, setup.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"), 100.0, setup.fib.stop, setup.fib.stop, 1.0, 0.0, 0.0,
    )
    initial = setup.fib.stop
    bar = pd.Series({"open": 11.0, "high": 12.2, "low": 11.5, "close": 12.1, "volume": 1.0})
    engine._process_position("BTC", pd.Timestamp("2025-01-02", tz="UTC"), bar)
    assert engine.portfolio.positions["BTC"].target_done[0]
    assert engine.portfolio.positions["BTC"].current_stop == pytest.approx(initial)

