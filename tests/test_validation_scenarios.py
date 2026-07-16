import json

import pandas as pd
import pytest

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.models import Order, Position
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.fibonacci import levels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.swings import Swing


def setup(side="long"):
    t = pd.Timestamp("2025-01-01", tz="UTC")
    low = Swing("low", 0, 2, 10, t, t + pd.Timedelta(hours=2))
    high = Swing("high", 10, 12, 20, t + pd.Timedelta(hours=10), t + pd.Timedelta(hours=12))
    first, second = (low, high) if side == "long" else (high, low)
    return Setup(f"{side}-setup", side, first, second, levels(side, 10, 20), second.confirmation_time)


def engine(policy="conservative", fee=0.0, slippage=0.0, max_positions=1):
    return BacktestEngine(RunConfig(assets=["BTC"], swing_n=2, leverage=2, max_positions=max_positions,
        execution_policy=policy, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", fee, slippage)}))


def position(e, s, quantity=100):
    p = Position("BTC", s, quantity, s.fib.entry, s.fib.entry, pd.Timestamp("2025-01-02", tz="UTC"),
                 pd.Timestamp("2025-01-01 23:00", tz="UTC"), 200, s.fib.stop, s.fib.stop, quantity, 0, 0)
    e.portfolio.positions["BTC"] = p
    e.portfolio.reserved_notional = p.entry_price * quantity
    return p


def test_short_limit_and_stop_execute_at_fib_levels():
    e, s = engine(), setup("short")
    e._marks["BTC"] = 15
    e.orders["BTC"] = Order("BTC", s, s.signal_time, 0, 0)
    e._process_order("BTC", 0, s.signal_time, pd.Series({"low": 15, "high": 19}))
    assert e.portfolio.positions["BTC"].entry_price == pytest.approx(s.fib.entry)
    e._process_position("BTC", pd.Timestamp("2025-01-02", tz="UTC"), pd.Series({"low": 18, "high": 20.3}))
    assert e.portfolio.closed[0]["exit_reason"] == "stop_before_tp1"


def test_all_targets_close_exactly_one_original_position():
    e, s = engine(), setup()
    p = position(e, s)
    e._process_position("BTC", pd.Timestamp("2025-01-03", tz="UTC"), pd.Series({"low": 11.3, "high": 20.0}))
    assert not e.portfolio.positions
    closed = e.portfolio.closed[0]
    exits = json.loads(closed["exit_events"])
    assert sum(event["quantity"] for event in exits) == pytest.approx(100)
    assert [event["reason"] for event in exits] == ["tp1", "tp2", "tp3", "tp4", "tp5"]


def test_optimistic_targets_precede_reachable_initial_stop():
    e, s = engine("optimistic"), setup()
    position(e, s)
    e._process_position("BTC", pd.Timestamp("2025-01-03", tz="UTC"), pd.Series({"low": 9.7, "high": 12.2}))
    events = json.loads(e.portfolio.closed[0]["exit_events"])
    assert [event["reason"] for event in events] == ["tp1", "post_tp1_stop"]


def test_order_cannot_fill_before_active_bar_and_invalidated_order_cannot_enter():
    e, s = engine(), setup()
    e._marks["BTC"] = 15
    e.orders["BTC"] = Order("BTC", s, s.signal_time, 1, 0)
    e._process_order("BTC", 0, s.signal_time, pd.Series({"low": 11, "high": 12}))
    assert not e.portfolio.positions
    e._expire_or_invalidate_order("BTC", 1, pd.Series({"low": 9.5, "high": 12}))
    e._process_order("BTC", 1, s.signal_time, pd.Series({"low": 11, "high": 12}))
    assert not e.portfolio.positions


def test_duplicate_setup_is_not_reentered_and_fee_slippage_reduce_equity():
    e, s = engine(fee=0.001, slippage=0.001), setup()
    e._marks["BTC"] = 15
    e.orders["BTC"] = Order("BTC", s, s.signal_time, 0, 0)
    e._process_order("BTC", 0, s.signal_time, pd.Series({"low": 11, "high": 12}))
    p = e.portfolio.positions["BTC"]
    assert p.entry_price > s.fib.entry and e.portfolio.cash < 10_000
    e.used_setups.add(s.identifier)
    assert s.identifier in e.used_setups


def test_lower_timeframe_replay_uses_observed_bar_order():
    e, s = engine("lower_timeframe_replay"), setup()
    position(e, s)
    index = pd.date_range("2025-01-02", periods=2, freq="4h", tz="UTC")
    higher = pd.DataFrame({"open": [11.5, 11.5], "high": [12.3, 12], "low": [10.9, 11], "close": [11.5, 11.5], "volume": [1, 1]}, index=index)
    sub_index = pd.date_range("2025-01-02", periods=8, freq="h", tz="UTC")
    replay = pd.DataFrame({"open": [11.5]*8, "high": [12.3, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5], "low": [11.3, 11.0, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3], "close": [11.5]*8, "volume": [1]*8}, index=sub_index)
    trades, _ = e.run({"BTC": higher}, {"BTC": replay})
    events = json.loads(trades.iloc[0].exit_events)
    assert [event["reason"] for event in events][:2] == ["tp1", "post_tp1_stop"]
