import pandas as pd
import pytest

from fib_backtester.backtest.models import Order
from fib_backtester.backtest.v3_entry_engine import StrategyV3EntryResearchEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing
from fib_backtester.strategy.v3_entry_research import ENTRY_LEVELS, entry_research_levels, setup_with_entry_level


def _setup(side="long"):
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 1, 10, t, t + pd.Timedelta(hours=1))
    high = Swing("high", 5, 6, 20, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    first, second = (low, high) if side == "long" else (high, low)
    return Setup("test", side, first, second, levels(side, 10, 20), second.confirmation_time)


def _engine():
    config = RunConfig(
        assets=["BTC"], timeframes=["1h"], leverage=2,
        asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)},
    )
    return StrategyV3EntryResearchEngine(config, .01, .882)


@pytest.mark.parametrize("entry_level", ENTRY_LEVELS)
@pytest.mark.parametrize("side", ["long", "short"])
def test_each_entry_level_uses_only_the_requested_fibonacci_ratio(entry_level, side):
    baseline = levels(side, 10, 20)
    changed = entry_research_levels(side, 10, 20, entry_level)
    expected = 20 - entry_level * 10 if side == "long" else 10 + entry_level * 10
    assert changed.entry == pytest.approx(expected)
    assert changed.low == baseline.low and changed.high == baseline.high
    assert changed.stop == baseline.stop
    assert changed.targets == baseline.targets
    assert changed.post_tp1_stop == baseline.post_tp1_stop


def test_baseline_entry_is_identical_to_v2_and_setup_identity_is_preserved():
    original = _setup()
    changed = setup_with_entry_level(original, .882)
    assert changed.identifier == original.identifier
    assert changed.first == original.first and changed.second == original.second
    assert changed.fib == original.fib


@pytest.mark.parametrize("entry_level", ENTRY_LEVELS)
def test_position_size_recalculates_from_unchanged_two_percent_stop_risk(entry_level):
    engine = _engine()
    setup = setup_with_entry_level(_setup(), entry_level)
    engine._marks["BTC"] = 15
    engine._bars = {"BTC": pd.DataFrame(index=pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"))}
    order = Order("BTC", setup, engine._bars["BTC"].index[1], 1, 0)
    assert engine._fill_v2_order(order, engine._bars["BTC"].index[1])
    expected = 10_000 * .02 / abs(setup.fib.entry - setup.fib.stop)
    assert engine.portfolio.positions["BTC"].quantity == pytest.approx(expected)


def test_v3_order_submission_remains_next_candle_after_lifecycle_event():
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    bars = pd.DataFrame({
        "open": [10] * 8, "high": [10.1, 10.2, 10.3, 10.4, 11, 12, 10.3, 14],
        "low": [10, 10.1, 10.1, 10.2, 10.5, 11.2, 10.2, 13],
        "close": [10] * 8, "volume": [1] * 8,
    }, index=index)
    config = RunConfig(assets=["BTC"], timeframes=["1h"], min_pivot_distance=4,
                       leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    engine = StrategyV3EntryResearchEngine(config, .05, .900)
    engine.run({"BTC": bars})
    submitted = [row for row in engine.lifecycle_history if row["action"] in {"activate", "update"}]
    assert submitted
    assert all(pd.Timestamp(row["order_submission"]) > pd.Timestamp(row["timestamp"]) for row in submitted)
