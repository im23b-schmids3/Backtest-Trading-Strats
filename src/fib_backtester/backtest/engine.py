from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from fib_backtester.config import RunConfig
from fib_backtester.strategy.signals import Setup, setup_from_swings
from fib_backtester.strategy.swings import Swing, confirmed_swings
from .execution import CostModel
from .models import Order, Position, TP_FRACTIONS
from .portfolio import Portfolio


class BacktestEngine:
    """Deterministic event engine: signals occur on confirmations and activate next bar."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.portfolio = Portfolio(config.initial_cash)
        self.orders: dict[str, Order] = {}
        self.used_setups: set[str] = set()
        self._marks: dict[str, float] = {}
        self._bars: dict[str, pd.DataFrame] = {}
        self._swings_by_confirmation: dict[str, dict[int, list[Swing]]] = {}
        self._prior_swings: dict[str, list[Swing]] = {}

    def run(self, data: dict[str, pd.DataFrame], replay_data: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._bars = data
        for asset, bars in data.items():
            confirm: dict[int, list[Swing]] = {}
            for swing in confirmed_swings(bars, self.config.swing_n):
                confirm.setdefault(swing.confirmation_index, []).append(swing)
            self._swings_by_confirmation[asset] = confirm
            self._prior_swings[asset] = []
        events = sorted((time, asset, i) for asset, bars in data.items() for i, time in enumerate(bars.index))
        equity_rows: list[dict] = []
        for timestamp, asset, index in events:
            bar = data[asset].iloc[index]
            self._marks[asset] = float(bar.close)
            execution_bars = [(timestamp, bar)]
            if self.config.execution_policy == "lower_timeframe_replay":
                if not replay_data or asset not in replay_data:
                    raise ValueError(f"lower_timeframe_replay needs replay bars for {asset}")
                next_time = data[asset].index[index + 1] if index + 1 < len(data[asset]) else timestamp + self._inferred_step(data[asset])
                replay = replay_data[asset].loc[(replay_data[asset].index >= timestamp) & (replay_data[asset].index < next_time)]
                if replay.empty:
                    raise ValueError(f"lower_timeframe_replay has no bars for {asset} at {timestamp}")
                execution_bars = [(when, replay.loc[when]) for when in replay.index]
            for execution_time, execution_bar in execution_bars:
                self._expire_or_invalidate_order(asset, index, execution_bar)
                self._process_order(asset, index, execution_time, execution_bar)
                self._process_position(asset, execution_time, execution_bar)
            self._publish_setups(asset, index, timestamp)
            equity_rows.append({"timestamp": timestamp, "equity": self.portfolio.equity(self._marks), "asset_event": asset})
        # force all remaining positions at final available close
        for asset, position in list(self.portfolio.positions.items()):
            final = data[asset].iloc[-1]
            self._close(position, position.remaining, float(final.close), data[asset].index[-1], "end_of_test")
        if data:
            terminal = max(bars.index[-1] for bars in data.values())
            equity_rows.append({"timestamp": terminal, "equity": self.portfolio.equity(self._marks), "asset_event": "terminal"})
        return pd.DataFrame(self.portfolio.closed), pd.DataFrame(equity_rows).drop_duplicates("timestamp", keep="last")

    def _publish_setups(self, asset: str, index: int, timestamp: object) -> None:
        for second in self._swings_by_confirmation[asset].get(index, []):
            candidates = [s for s in self._prior_swings[asset] if s.kind != second.kind and s.pivot_index < second.pivot_index]
            if candidates:
                setup = setup_from_swings(candidates[-1], second, self.config.min_pivot_distance)
                if setup and (self.config.reentry or setup.identifier not in self.used_setups):
                    if index + 1 >= len(self._bars[asset]):
                        continue
                    # OHLC timestamps mark candle open; the confirmation candle closes as the next candle opens.
                    submission_time = self._bars[asset].index[index + 1]
                    self.orders[asset] = Order(asset, setup, submission_time, index + 1, index)
            self._prior_swings[asset].append(second)

    def _expire_or_invalidate_order(self, asset: str, index: int, bar: pd.Series) -> None:
        order = self.orders.get(asset)
        if not order:
            return
        if self.config.entry_max_age_bars is not None and index - order.created_index > self.config.entry_max_age_bars:
            del self.orders[asset]
            return
        boundary = order.setup.fib.low if order.setup.side == "long" else order.setup.fib.high
        broken = float(bar.low) < boundary if order.setup.side == "long" else float(bar.high) > boundary
        if broken:
            del self.orders[asset]

    def _process_order(self, asset: str, index: int, timestamp: object, bar: pd.Series) -> None:
        order = self.orders.get(asset)
        if not order or index < order.active_from_index or asset in self.portfolio.positions:
            return
        if len(self.portfolio.positions) >= self.config.max_positions:
            return
        limit = order.setup.fib.entry
        touched = float(bar.low) <= limit if order.setup.side == "long" else float(bar.high) >= limit
        if not touched:
            return
        costs = self._costs(asset)
        entry = costs.fill_price(limit, order.setup.side, "entry")
        stop_distance = abs(entry - order.setup.fib.stop)
        equity = self.portfolio.equity(self._marks)
        risk_budget = equity * 0.02
        if stop_distance <= 0 or not pd.notna(stop_distance):
            del self.orders[asset]
            return
        quantity = risk_budget / stop_distance
        notional = quantity * entry
        if self.portfolio.reserved_notional + notional > equity * self.config.leverage:
            del self.orders[asset]
            return
        if self.portfolio.planned_risk() + risk_budget > equity * self.config.max_total_risk_fraction:
            del self.orders[asset]
            return
        fee = costs.fee(quantity, entry)
        self.portfolio.cash -= fee
        self.portfolio.reserved_notional += notional
        self.portfolio.positions[asset] = Position(asset, order.setup, quantity, entry, limit, timestamp, order.submission_time, risk_budget, order.setup.fib.stop, order.setup.fib.stop, quantity, fee, abs(entry - limit) * quantity)
        self.used_setups.add(order.setup.identifier)
        del self.orders[asset]

    def _process_position(self, asset: str, timestamp: object, bar: pd.Series) -> None:
        position = self.portfolio.positions.get(asset)
        if not position:
            return
        stop_hit = float(bar.low) <= position.current_stop if position.side == "long" else float(bar.high) >= position.current_stop
        targets_hit = [
            (float(bar.high) >= target if position.side == "long" else float(bar.low) <= target)
            for target in position.setup.fib.targets
        ]
        # Conservative lets any adverse stop precede a reachable target; optimistic reverses it.
        if stop_hit and (self.config.execution_policy == "conservative" or not any(targets_hit)):
            self._close(position, position.remaining, position.current_stop, timestamp, "post_tp1_stop" if any(position.target_done) else "stop_before_tp1")
            return
        for target_index, hit in enumerate(targets_hit):
            if hit and not position.target_done[target_index] and position.remaining > 1e-12:
                quantity = min(position.quantity * TP_FRACTIONS[target_index], position.remaining)
                self._close(position, quantity, position.setup.fib.targets[target_index], timestamp, f"tp{target_index + 1}", final=False)
                position.target_done[target_index] = True
                if target_index == 0 and asset in self.portfolio.positions:
                    position.current_stop = position.setup.fib.post_tp1_stop
        if asset in self.portfolio.positions and stop_hit:
            self._close(position, position.remaining, position.current_stop, timestamp, "post_tp1_stop" if any(position.target_done) else "stop_before_tp1")

    def _close(self, position: Position, quantity: float, raw_price: float, timestamp: object, reason: str, final: bool = True) -> None:
        costs = self._costs(position.asset)
        price = costs.fill_price(raw_price, position.side, "exit")
        fee = costs.fee(quantity, price)
        direction = 1 if position.side == "long" else -1
        self.portfolio.cash += direction * (price - position.entry_price) * quantity - fee
        self.portfolio.reserved_notional -= position.entry_price * quantity
        position.remaining = max(0.0, position.remaining - quantity)
        position.exit_value += price * quantity
        position.exit_qty += quantity
        position.exit_fee += fee
        position.slippage_cost += abs(price - raw_price) * quantity
        position.stop_reason = reason
        position.exit_events.append({"timestamp": str(timestamp), "reason": reason, "quantity": quantity, "raw_price": raw_price, "fill_price": price, "fee": fee})
        if position.remaining <= 1e-10 or final:
            if position.remaining > 1e-10:  # defensive: only used for forced closure
                raise RuntimeError("final close must close remaining quantity")
            average_exit = position.exit_value / position.exit_qty
            gross = (1 if position.side == "long" else -1) * (average_exit - position.entry_price) * position.quantity
            total_fees = position.entry_fee + position.exit_fee
            self.portfolio.closed.append({
                "asset": position.asset, "setup_id": position.setup.identifier, "side": position.side,
                "signal_timestamp": position.setup.signal_time, "order_submission_timestamp": position.order_submission_time,
                "fill_timestamp": position.entry_time, "exit_timestamp": timestamp,
                "first_pivot_timestamp": position.setup.first.pivot_time, "first_confirmation_timestamp": position.setup.first.confirmation_time,
                "second_pivot_timestamp": position.setup.second.pivot_time, "second_confirmation_timestamp": position.setup.second.confirmation_time,
                "entry_price": position.entry_price, "average_exit_price": average_exit, "quantity": position.quantity,
                "fib_low": position.setup.fib.low, "fib_high": position.setup.fib.high, "initial_stop": position.initial_stop,
                "targets": json.dumps(position.setup.fib.targets),
                "gross_pnl": gross, "fees": total_fees, "slippage_cost": position.slippage_cost,
                "net_pnl": gross - total_fees, "exit_reason": reason, "targets_hit": sum(position.target_done),
                "exit_events": json.dumps(position.exit_events),
                "holding_hours": (pd.Timestamp(timestamp) - pd.Timestamp(position.entry_time)).total_seconds() / 3600,
                "risk_budget": position.risk_budget,
            })
            self.portfolio.positions.pop(position.asset, None)

    def _costs(self, asset: str) -> CostModel:
        ac = self.config.asset_configs[asset]
        return CostModel(ac.fee_rate, ac.slippage_rate)

    @staticmethod
    def _inferred_step(bars: pd.DataFrame) -> pd.Timedelta:
        if len(bars) < 2:
            raise ValueError("cannot replay a single aggregate candle")
        return pd.Timestamp(bars.index[-1]) - pd.Timestamp(bars.index[-2])
