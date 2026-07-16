from __future__ import annotations

import json

import pandas as pd

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.models import Order, Position
from fib_backtester.strategy.v2_swings import V2Construction, active_wick_lifecycle


class StrategyV2Engine(BacktestEngine):
    """V2 lifecycle with independent generations and unchanged execution arithmetic."""

    def __init__(self, config, min_move: float):
        super().__init__(config)
        self.min_move = min_move
        self.construction: dict[str, V2Construction] = {}
        self.v2_orders: dict[tuple[str, str, str], Order] = {}
        self.diagnostics: dict[str, dict[str, float]] = {}
        self.lifecycle_history: list[dict] = []
        self.order_versions_by_setup: dict[str, int] = {}
        self.trade_context: dict[str, dict] = {}

    def run(self, data: dict[str, pd.DataFrame], replay_data=None):
        self._bars = data
        for asset, bars in data.items():
            timeframe = self.config.timeframes[0] if self.config.timeframes else None
            max_age = self.config.max_anchor_age_days.get(timeframe) if timeframe else None
            self.construction[asset] = active_wick_lifecycle(
                bars, self.config.min_pivot_distance, self.min_move, max_age
            )
            self.diagnostics[asset] = {
                **self.construction[asset].diagnostics,
                "initial_orders": 0, "replacement_orders": 0, "order_versions": 0,
                "cancellations_extreme_updated": 0, "expired_setups": 0,
                "max_anchor_age_invalidations_before_entry": 0, "filled_orders": 0,
                "filled_after_update": 0, "updates_before_filled_orders": 0,
                "executed_trades": 0, "conflicting_position_cancellations": 0,
                "average_order_versions_per_setup": 0.0,
                "average_setup_lifetime_hours": 0.0,
                "average_final_extreme_to_entry_hours": 0.0,
                "average_anchor_age_hours": 0.0,
                "average_setups_per_trend": 0.0,
                "average_trades_per_trend": 0.0,
            }

        rows = []
        for timestamp, asset, index in sorted((t, a, i) for a, b in data.items() for i, t in enumerate(b.index)):
            bar = data[asset].iloc[index]
            self._marks[asset] = float(bar.close)
            self._process_v2_orders(asset, index, timestamp, bar)
            self._process_position(asset, timestamp, bar)
            self._apply_events(asset, index)
            rows.append({"timestamp": timestamp, "equity": self.portfolio.equity(self._marks), "asset_event": asset})

        for asset, position in list(self.portfolio.positions.items()):
            final = data[asset].iloc[-1]
            self._close(position, position.remaining, float(final.close), data[asset].index[-1], "end_of_test")

        for asset in data:
            trades = pd.DataFrame(self.portfolio.closed)
            self.diagnostics[asset]["executed_trades"] = int((trades.asset == asset).sum()) if not trades.empty else 0
            setups = max(self.diagnostics[asset]["unique_active_setups"], 1)
            versions = self.diagnostics[asset]["initial_orders"] + self.diagnostics[asset]["replacement_orders"]
            self.diagnostics[asset]["order_versions"] = versions
            self.diagnostics[asset]["average_order_versions_per_setup"] = versions / setups
            relevant = [row for row in self.lifecycle_history if row["asset"] == asset]
            starts = {row["setup_id"]: pd.Timestamp(row["timestamp"]) for row in relevant if row["action"] == "activate"}
            ends = {}
            updates = {}
            for row in relevant:
                if row["action"] in {"update", "activate"}:
                    updates[row["setup_id"]] = pd.Timestamp(row["timestamp"])
                if row["action"] in {"filled", "invalidated"}:
                    ends[row["setup_id"]] = pd.Timestamp(row["timestamp"])
            lifetimes = [(ends[key] - value).total_seconds() / 3600 for key, value in starts.items() if key in ends]
            fill_delays = [
                (ends[key] - updates[key]).total_seconds() / 3600
                for key in ends if key in updates and any(row["action"] == "filled" and row["setup_id"] == key for row in relevant)
            ]
            for row in relevant:
                if row["action"] == "filled":
                    prior_updates = [
                        item for item in relevant
                        if item["setup_id"] == row["setup_id"] and item["action"] == "update"
                        and pd.Timestamp(item["timestamp"]) < pd.Timestamp(row["timestamp"])
                    ]
                    if prior_updates:
                        self.diagnostics[asset]["filled_after_update"] += 1
                        self.diagnostics[asset]["updates_before_filled_orders"] += len(prior_updates)
            self.diagnostics[asset]["average_setup_lifetime_hours"] = float(pd.Series(lifetimes).mean()) if lifetimes else 0.0
            self.diagnostics[asset]["average_final_extreme_to_entry_hours"] = float(pd.Series(fill_delays).mean()) if fill_delays else 0.0
            activates = [row for row in relevant if row["action"] == "activate"]
            trends = {(row["side"], row.get("trend_id", 0)) for row in activates}
            self.diagnostics[asset]["average_setups_per_trend"] = len(activates) / len(trends) if trends else 0.0
            fills = [row for row in relevant if row["action"] == "filled"]
            filled_trends = {(row["side"], row.get("trend_id", 0)) for row in fills}
            self.diagnostics[asset]["average_trades_per_trend"] = len(fills) / len(filled_trends) if filled_trends else 0.0
            ages = [
                (pd.Timestamp(row["timestamp"]) - pd.Timestamp(row["anchor_timestamp"])).total_seconds() / 3600
                for row in fills if row.get("anchor_timestamp") is not None
            ]
            self.diagnostics[asset]["average_anchor_age_hours"] = float(pd.Series(ages).mean()) if ages else 0.0

        if data:
            rows.append({"timestamp": max(b.index[-1] for b in data.values()), "equity": self.portfolio.equity(self._marks), "asset_event": "terminal"})
        trades = self._enrich_trade_log(pd.DataFrame(self.portfolio.closed))
        return trades, pd.DataFrame(rows).drop_duplicates("timestamp", keep="last")

    def _apply_events(self, asset, index):
        for event in self.construction[asset].events_by_index.get(index, []):
            key = (asset, event.side, event.setup_id)
            previous = self.v2_orders.get(key)
            if event.action == "invalidate":
                if previous:
                    self._cancel(key, event.reason or "invalidated", event.index)
                if event.reason == "anchor_max_age" and event.setup_id not in self.used_setups:
                    self.diagnostics[asset]["max_anchor_age_invalidations_before_entry"] += 1
                self.lifecycle_history.append({
                    "asset": asset, "side": event.side, "setup_id": event.setup_id,
                    "trend_id": event.trend_id, "action": "invalidated",
                    "timestamp": self._bars[asset].index[index], "reason": event.reason,
                })
                continue
            if event.setup_id in self.used_setups:
                continue  # A filled setup remains frozen.
            if previous:
                self._cancel(key, "active_swing_extreme_updated", event.index)
                self.diagnostics[asset]["cancellations_extreme_updated"] += 1
            if index + 1 >= len(self._bars[asset]):
                continue
            order = Order(asset, event.setup, self._bars[asset].index[index + 1], index + 1, index)
            self.v2_orders[key] = order
            self.order_versions_by_setup[event.setup_id] = self.order_versions_by_setup.get(event.setup_id, 0) + 1
            if event.action == "activate":
                self.diagnostics[asset]["initial_orders"] += 1
            else:
                self.diagnostics[asset]["replacement_orders"] += 1
            self.lifecycle_history.append({
                "asset": asset, "side": event.side, "setup_id": event.setup_id,
                "trend_id": event.trend_id, "action": event.action,
                "timestamp": self._bars[asset].index[index], "entry": event.setup.fib.entry,
                "stop": event.setup.fib.stop, "order_submission": order.submission_time,
            })

    def _cancel(self, key, reason, index):
        order = self.v2_orders.pop(key, None)
        if order:
            self.lifecycle_history.append({
                "asset": order.asset, "side": order.setup.side, "setup_id": order.setup.identifier,
                "action": "cancelled", "timestamp": self._bars[order.asset].index[index],
                "reason": reason, "entry": order.setup.fib.entry,
            })

    def _process_v2_orders(self, asset, index, timestamp, bar):
        for key, order in list(self.v2_orders.items()):
            if key[0] != asset or index < order.active_from_index:
                continue
            side = order.setup.side
            if self.config.entry_max_age_bars is not None and index - order.created_index > self.config.entry_max_age_bars:
                self._cancel(key, "expired", index)
                self.diagnostics[asset]["expired_setups"] += 1
                continue
            if asset in self.portfolio.positions:
                continue
            touched = float(bar.low) <= order.setup.fib.entry if side == "long" else float(bar.high) >= order.setup.fib.entry
            if not touched or not self._fill_v2_order(order, timestamp):
                continue
            self.v2_orders.pop(key, None)
            self.diagnostics[asset]["filled_orders"] += 1
            opposite_side = "short" if side == "long" else "long"
            for opposite in [candidate for candidate in self.v2_orders if candidate[0] == asset and candidate[1] == opposite_side]:
                self._cancel(opposite, "conflicting_open_position", index)
                self.diagnostics[asset]["conflicting_position_cancellations"] += 1

    def _fill_v2_order(self, order, timestamp):
        asset, setup = order.asset, order.setup
        costs = self._costs(asset)
        entry = costs.fill_price(setup.fib.entry, setup.side, "entry")
        distance = abs(entry - setup.fib.stop)
        equity = self.portfolio.equity(self._marks)
        budget = equity * .02
        if distance <= 0 or not pd.notna(distance):
            return False
        quantity = budget / distance
        notional = quantity * entry
        if len(self.portfolio.positions) >= self.config.max_positions or self.portfolio.reserved_notional + notional > equity * self.config.leverage or self.portfolio.planned_risk() + budget > equity * self.config.max_total_risk_fraction:
            return False
        fee = costs.fee(quantity, entry)
        self.portfolio.cash -= fee
        self.portfolio.reserved_notional += notional
        self.portfolio.positions[asset] = Position(asset, setup, quantity, entry, setup.fib.entry, timestamp, order.submission_time, budget, setup.fib.stop, setup.fib.stop, quantity, fee, abs(entry - setup.fib.entry) * quantity)
        self.used_setups.add(setup.identifier)
        trend_id = next((row.get("trend_id", 0) for row in reversed(self.lifecycle_history) if row["setup_id"] == setup.identifier), 0)
        self.lifecycle_history.append({"asset": asset, "side": setup.side, "setup_id": setup.identifier, "trend_id": trend_id, "action": "filled", "timestamp": timestamp, "entry": entry, "fib_low": setup.fib.low, "fib_high": setup.fib.high, "anchor_timestamp": setup.first.pivot_time})
        self.trade_context[setup.identifier] = {
            "strategy": "Strategy_V2", "distance_parameter": self.config.min_pivot_distance,
            "minimum_move_parameter": self.min_move, "order_version": self.order_versions_by_setup.get(setup.identifier, 1),
            "anchor_timestamp": setup.first.pivot_time, "extreme_timestamp": setup.second.pivot_time,
            "trend_id": trend_id, "entry_candle_index": int(self._bars[asset].index.get_loc(timestamp)),
            "stop_price": setup.fib.stop, "tp_prices": json.dumps(setup.fib.targets),
            "swing_updates_before_fill": max(0, self.order_versions_by_setup.get(setup.identifier, 1) - 1),
        }
        return True

    def _enrich_trade_log(self, trades: pd.DataFrame) -> pd.DataFrame:
        """One row per completed position, enriched with frozen V2 state at fill."""
        if trades.empty:
            return trades
        rows = []
        for trade in trades.to_dict("records"):
            context = self.trade_context.get(trade["setup_id"], {})
            exit_time = pd.Timestamp(trade["exit_timestamp"])
            bars = self._bars[trade["asset"]]
            try:
                exit_index = int(bars.index.get_loc(exit_time))
            except KeyError:
                exit_index = None
            rows.append({
                **trade, **context, "timeframe": None, "entry_timestamp": trade["fill_timestamp"],
                "exit_candle_index": exit_index, "exit_price": trade["average_exit_price"],
                "anchor_age_hours": (pd.Timestamp(trade["fill_timestamp"]) - pd.Timestamp(context["anchor_timestamp"])).total_seconds() / 3600,
                "r_multiple": trade["net_pnl"] / trade["risk_budget"] if trade["risk_budget"] else None,
            })
        return pd.DataFrame(rows)
