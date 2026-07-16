import pandas as pd

from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.strategy.v2_swings import active_wick_lifecycle


def _bars(high, low):
    index = pd.date_range("2025-01-01", periods=len(high), freq="h", tz="UTC")
    return pd.DataFrame({"open": low, "high": high, "low": low, "close": high, "volume": 1.0}, index=index)


def _long_events(result):
    return [event for event in result.events if event.side == "long" and event.action in {"activate", "update"}]


def test_long_anchor_and_identifier_stay_fixed_across_higher_high_updates():
    bars = _bars([10.1, 10.2, 10.3, 10.4, 11.0, 12.0], [10, 10.1, 10.1, 10.2, 10.5, 11.2])
    events = _long_events(active_wick_lifecycle(bars, 4, .05))
    assert [event.action for event in events] == ["activate", "update"]
    assert events[0].setup.identifier == events[1].setup.identifier
    assert events[0].setup.first.price == events[1].setup.first.price == 10
    assert events[1].setup.second.price == 12
    assert events[1].setup.fib.entry != events[0].setup.fib.entry


def test_short_anchor_stays_fixed_across_lower_low_updates():
    bars = _bars([20, 19.9, 19.8, 19.7, 19.5, 18.8], [19.9, 19.8, 19.7, 19.6, 18.8, 17.5])
    events = [event for event in active_wick_lifecycle(bars, 4, .05).events if event.side == "short" and event.action in {"activate", "update"}]
    assert [event.action for event in events] == ["activate", "update"]
    assert events[0].setup.identifier == events[1].setup.identifier
    assert events[0].setup.first.price == events[1].setup.first.price == 20
    assert events[1].setup.second.price == 17.5


def test_anchor_break_invalidates_but_favorable_extreme_does_not():
    bars = _bars([10.1, 10.2, 10.3, 10.4, 11, 12, 11], [10, 10.1, 10.1, 10.1, 10.5, 11.2, 9.9])
    result = active_wick_lifecycle(bars, 4, .05)
    long = _long_events(result)
    invalid = [event for event in result.events if event.side == "long" and event.action == "invalidate"]
    assert len(long) == 2 and invalid[0].reason == "anchor_low_broken"


def test_replacement_is_submitted_after_extreme_candle_and_old_order_cancelled():
    bars = _bars([10.1, 10.2, 10.3, 10.4, 11, 12, 10.3], [10, 10.1, 10.1, 10.2, 10.5, 11.2, 10.2])
    config = RunConfig(assets=["BTC"], min_pivot_distance=4, leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    engine = StrategyV2Engine(config, .05); engine.run({"BTC": bars})
    history = [row for row in engine.lifecycle_history if row["side"] == "long"]
    update = next(row for row in history if row["action"] == "update")
    cancellation = next(row for row in history if row["action"] == "cancelled" and row["reason"] == "active_swing_extreme_updated")
    assert cancellation["timestamp"] == update["timestamp"]
    assert update["order_submission"] > update["timestamp"]
    assert engine.diagnostics["BTC"]["unique_active_setups"] == 1


def test_filled_setup_ignores_later_extreme_updates():
    bars = _bars([10.1, 10.2, 10.3, 10.4, 11, 12, 10.3, 14], [10, 10.1, 10.1, 10.2, 10.5, 11.2, 10.2, 13])
    config = RunConfig(assets=["BTC"], min_pivot_distance=4, leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", 0, 0)})
    engine = StrategyV2Engine(config, .05); engine.run({"BTC": bars})
    fills = [row for row in engine.lifecycle_history if row["action"] == "filled" and row["side"] == "long"]
    assert fills
    # Any later high event exists in the constructor but cannot produce an order after fill.
    filled_id = fills[0]["setup_id"]
    orders_after_fill = [row for row in engine.lifecycle_history if row["action"] in {"activate", "update"} and row["setup_id"] == filled_id and row["timestamp"] > fills[0]["timestamp"]]
    assert not orders_after_fill


def test_significant_higher_lows_create_independent_generations():
    bars = _bars(
        [10.1, 12.0, 11.5, 11.6, 13.0, 12.4, 12.5, 14.0],
        [10.0, 11.0, 10.8, 11.2, 12.0, 11.7, 12.1, 13.0],
    )
    events = [event for event in active_wick_lifecycle(bars, 2, .05).events if event.side == "long" and event.action == "activate"]
    assert len(events) == 3
    assert [event.setup.first.price for event in events] == [10.0, 10.8, 11.7]
    assert len({event.setup.identifier for event in events}) == 3


def test_anchor_max_age_invalidates_old_setup():
    bars = _bars([10.1, 10.2, 12.0, 12.1, 12.2], [10.0, 10.1, 10.1, 10.2, 10.3])
    result = active_wick_lifecycle(bars, 2, .05, max_anchor_age_days=.125)
    invalidations = [event for event in result.events if event.side == "long" and event.action == "invalidate"]
    assert any(event.reason == "anchor_max_age" for event in invalidations)
    assert result.diagnostics["max_anchor_age_invalidations"] >= 1


def test_v2_completed_trade_log_has_frozen_lifecycle_and_execution_fields():
    bars = _bars([10.1, 10.2, 10.3, 10.4, 11, 12, 10.3, 14], [10, 10.1, 10.1, 10.2, 10.5, 11.2, 10.2, 13])
    config = RunConfig(assets=["BTC"], min_pivot_distance=4, leverage=2, asset_configs={"BTC": AssetConfig("BTC/USDT", "binance", .001, .0002)})
    trades, _ = StrategyV2Engine(config, .05).run({"BTC": bars})
    required = {"asset", "strategy", "distance_parameter", "minimum_move_parameter", "setup_id", "order_version", "side",
                "anchor_timestamp", "extreme_timestamp", "entry_timestamp", "entry_candle_index", "entry_price", "stop_price",
                "tp_prices", "exit_timestamp", "exit_candle_index", "exit_price", "exit_reason", "gross_pnl", "net_pnl",
                "r_multiple", "fees", "slippage_cost", "swing_updates_before_fill"}
    assert required.issubset(trades.columns)
    assert len(trades) == 1
    assert trades.iloc[0].order_version > 1
