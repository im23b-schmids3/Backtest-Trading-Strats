import pandas as pd
import pytest

from fib_backtester.backtest.models import Position
from fib_backtester.backtest.v6_5_post_tp1_stop_placement_engine import StrategyV65PostTP1StopPlacementEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing
from fib_backtester.strategy.v4_take_profit_research import PROFILES, profile_levels
from fib_backtester.strategy.v6_5_post_tp1_stop_placement import STOP_PLACEMENT_POLICIES, STOP_PLACEMENT_ORDER


def _setup():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 1, 10, t, t + pd.Timedelta(hours=1))
    high = Swing("high", 5, 6, 20, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    return Setup("test", "long", low, high, levels("long", 10, 20), high.confirmation_time)


def _config():
    return RunConfig(assets=["BTC"], timeframes=["1h"], min_pivot_distance=4,
                     asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})


def test_requested_placement_set_is_exact():
    assert STOP_PLACEMENT_ORDER == ("no_stop_movement", "Fib 0.900", "Fib 0.890", "Fib 0.880", "Fib 0.870", "Fib 0.860", "Fib 0.850", "Fib 0.840", "Fib 0.830", "Fib 0.820", "Fib 0.786")
    assert STOP_PLACEMENT_POLICIES["Fib 0.900"].fib_ratio == pytest.approx(.9)
    assert STOP_PLACEMENT_POLICIES["Fib 0.786"].fib_ratio == pytest.approx(.786)
    assert STOP_PLACEMENT_POLICIES["no_stop_movement"].fib_ratio is None


@pytest.mark.parametrize("policy_name,ratio", [(name, STOP_PLACEMENT_POLICIES[name].fib_ratio) for name in STOP_PLACEMENT_ORDER if name != "no_stop_movement"])
def test_each_moved_stop_is_applied_only_after_tp1(policy_name, ratio):
    setup = profile_levels(_setup(), PROFILES["B"], .900)
    engine = StrategyV65PostTP1StopPlacementEngine(_config(), .01, STOP_PLACEMENT_POLICIES[policy_name])
    engine.portfolio.positions["BTC"] = Position(
        "BTC", setup, 1.0, setup.fib.entry, setup.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"), 100.0, setup.fib.stop, setup.fib.stop, 1.0, 0.0, 0.0,
    )
    bar = pd.Series({"open": 11.0, "high": 12.2, "low": 12.0, "close": 12.1, "volume": 1.0})
    engine._process_position("BTC", pd.Timestamp("2025-01-02", tz="UTC"), bar)
    position = engine.portfolio.positions["BTC"]
    assert position.target_done[0]
    assert position.current_stop == pytest.approx(20 - ratio * 10)
    assert not engine.portfolio.closed

