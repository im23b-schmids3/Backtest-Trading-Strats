import pandas as pd
import pytest

from fib_backtester.backtest.v3_entry_engine import StrategyV3EntryResearchEngine
from fib_backtester.backtest.v4_take_profit_engine import StrategyV4TakeProfitResearchEngine, _atr
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing
from fib_backtester.strategy.v4_take_profit_research import PROFILES, profile_levels


def _setup():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 1, 10, t, t + pd.Timedelta(hours=1))
    high = Swing("high", 5, 6, 20, t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6))
    return Setup("test", "long", low, high, levels("long", 10, 20), high.confirmation_time)


def _config():
    return RunConfig(assets=["BTC"], timeframes=["1h"], min_pivot_distance=4, leverage=2,
                     asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})


def test_profile_a_matches_fixed_v3_targets_and_entry():
    original = _setup()
    changed = profile_levels(original, PROFILES["A"])
    expected = levels("long", 10, 20)
    assert changed.fib.entry == pytest.approx(11.0)
    assert changed.fib.stop == expected.stop
    assert changed.fib.targets == expected.targets
    assert changed.fib.post_tp1_stop == expected.post_tp1_stop


@pytest.mark.parametrize("profile,ratios,fractions", [
    ("B", (.786, .618, .5, .236, .05), (.30, .25, .20, .15, .10)),
    ("C", (.786, .5, 0.0), (.40, .30, .30)),
    ("D", (.5, .236, 0.0), (.20, .20, .20)),
    ("E", (.5, .236, 0.0), (.30, .30, .20)),
])
def test_profiles_have_exact_target_ratios_and_fractions(profile, ratios, fractions):
    assert PROFILES[profile].ratios == ratios
    assert PROFILES[profile].fractions == fractions
    assert sum(fractions) + PROFILES[profile].runner_fraction == pytest.approx(1.0)


def test_atr_uses_only_closed_bars_and_requested_length():
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    bars = pd.DataFrame({"high": [11, 13, 14, 16], "low": [9, 10, 12, 13], "close": [10, 12, 13, 15]}, index=index)
    atr = _atr(bars, 2)
    true_ranges = [2, 3, 2, 3]
    assert pd.isna(atr.iloc[0])
    assert atr.iloc[1] == pytest.approx(sum(true_ranges[:2]) / 2)
    assert atr.iloc[2] == pytest.approx(sum(true_ranges[1:3]) / 2)


def test_profile_a_matches_v3_execution_on_same_bars():
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    bars = pd.DataFrame({
        "open": [10] * 8, "high": [10.1, 10.2, 10.3, 10.4, 11, 12, 10.3, 14],
        "low": [10, 10.1, 10.1, 10.2, 10.5, 11.2, 10.2, 13],
        "close": [10] * 8, "volume": [1] * 8,
    }, index=index)
    v3_trades, _ = StrategyV3EntryResearchEngine(_config(), .05, .900).run({"BTC": bars})
    v4_trades, _ = StrategyV4TakeProfitResearchEngine(_config(), .05, PROFILES["A"]).run({"BTC": bars})
    assert len(v3_trades) == len(v4_trades)
    if len(v3_trades):
        assert v3_trades.net_pnl.iloc[0] == pytest.approx(v4_trades.net_pnl.iloc[0])
        assert v3_trades.exit_reason.iloc[0] == v4_trades.exit_reason.iloc[0]
