import pandas as pd
import pytest

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine, _research_setup
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing
from fib_backtester.strategy.v4_take_profit_research import TakeProfitProfile
from fib_backtester.strategy.v7_frozen_validation import (
    FROZEN_ENTRY,
    FROZEN_INITIAL_STOP,
    FROZEN_POST_TP1_STOP,
    FROZEN_TP_FRACTIONS,
    FROZEN_TP_RATIOS,
)


def _setup(side="long"):
    t = pd.Timestamp("2025-01-01", tz="UTC")
    if side == "long":
        first = Swing("low", 0, 1, 10, t, t + pd.Timedelta(hours=1))
        second = Swing("high", 5, 6, 20, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    else:
        first = Swing("high", 0, 1, 20, t, t + pd.Timedelta(hours=1))
        second = Swing("low", 5, 6, 10, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    return Setup("v7-test", side, first, second, levels(side, 10, 20), second.confirmation_time)


def _config():
    return RunConfig(
        assets=["BTC"],
        timeframes=["1h"],
        min_pivot_distance=4,
        asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0.001, 0.0002)},
    )


def test_frozen_specification_is_exact():
    assert FROZEN_ENTRY == pytest.approx(.900)
    assert FROZEN_INITIAL_STOP == pytest.approx(1.020)
    assert FROZEN_POST_TP1_STOP == pytest.approx(.820)
    assert FROZEN_TP_RATIOS == (.786, .618, .500, .236, .050)
    assert FROZEN_TP_FRACTIONS == (.30, .25, .20, .15, .10)


@pytest.mark.parametrize("side", ["long", "short"])
def test_frozen_validation_levels_preserve_fibonacci_convention(side):
    profile = TakeProfitProfile("B", FROZEN_TP_RATIOS, FROZEN_TP_FRACTIONS)
    setup = _research_setup(_setup(side), .900, 1.020, .820, profile)
    distance = 10.0
    if side == "long":
        assert setup.fib.entry == pytest.approx(20 - .900 * distance)
        assert setup.fib.stop == pytest.approx(20 - 1.020 * distance)
        assert setup.fib.post_tp1_stop == pytest.approx(20 - .820 * distance)
    else:
        assert setup.fib.entry == pytest.approx(10 + .900 * distance)
        assert setup.fib.stop == pytest.approx(10 + 1.020 * distance)
        assert setup.fib.post_tp1_stop == pytest.approx(10 + .820 * distance)


def test_default_engine_contains_no_validation_perturbation():
    engine = StrategyV7FrozenValidationEngine(_config(), .005)
    assert engine.entry_level == pytest.approx(.900)
    assert engine.initial_stop == pytest.approx(1.020)
    assert engine.post_tp1_stop == pytest.approx(.820)
    assert engine.profile.fractions == FROZEN_TP_FRACTIONS
    assert engine.fee_multiplier == pytest.approx(1.0)
    assert engine.slippage_multiplier == pytest.approx(1.0)
    assert engine.delay_bars == 0
    assert engine.missed_fill_probability == pytest.approx(0.0)


def test_stress_cost_shocks_are_adverse_and_deterministic():
    baseline = StrategyV7FrozenValidationEngine(_config(), .005)
    stressed = StrategyV7FrozenValidationEngine(_config(), .005, fee_multiplier=3, slippage_multiplier=3)
    assert stressed._costs("BTC").fee_rate == pytest.approx(baseline._costs("BTC").fee_rate * 3)
    assert stressed._costs("BTC").slippage_rate == pytest.approx(baseline._costs("BTC").slippage_rate * 3)


def test_adverse_fill_cushion_is_explicit():
    engine = StrategyV7FrozenValidationEngine(_config(), .005, adverse_fill_extra_slippage=.0005)
    assert engine._costs("BTC").slippage_rate == pytest.approx(.0007)
