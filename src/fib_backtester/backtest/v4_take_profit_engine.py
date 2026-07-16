from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fib_backtester.backtest.v3_entry_engine import StrategyV3EntryResearchEngine
from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.strategy.v4_take_profit_research import TakeProfitProfile, profile_levels


class StrategyV4TakeProfitResearchEngine(StrategyV3EntryResearchEngine):
    """V3 0.900 entry with research-only take-profit and runner profiles."""

    strategy_version = "Strategy_V4_TakeProfitResearch"

    def __init__(self, config, min_move: float, profile: TakeProfitProfile, atr_length: int | None = None, atr_multiplier: float | None = None):
        super().__init__(config, min_move, .900)
        self.profile = profile
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self._atr: dict[str, pd.Series] = {}
        self._runner: dict[str, dict] = {}
        self._trade_research: dict[str, dict] = {}

    def run(self, data, replay_data=None):
        self._atr = {asset: _atr(bars, self.atr_length) if self.profile.runner else pd.Series(np.nan, index=bars.index) for asset, bars in data.items()}
        self._runner = {}
        self._trade_research = {}
        return super().run(data, replay_data)

    def _apply_events(self, asset, index):
        events = self.construction[asset].events_by_index.get(index, [])
        transformed = [
            event if event.setup is None else event.__class__(
                event.action, event.side, event.setup_id, event.index,
                profile_levels(event.setup, self.profile, .900), event.reason, event.trend_id,
            )
            for event in events
        ]
        self.construction[asset].events_by_index[index] = transformed
        try:
            StrategyV2Engine._apply_events(self, asset, index)
        finally:
            self.construction[asset].events_by_index[index] = events

    def _process_position(self, asset, timestamp, bar):
        position = self.portfolio.positions.get(asset)
        if not position:
            return
        state = self._trade_research.setdefault(position.setup.identifier, {"max_mfe_r": 0.0, "runner_exit": False, "runner_outperformed": False})
        risk = abs(position.entry_price - position.initial_stop)
        if risk > 0:
            favorable = float(bar.high) - position.entry_price if position.side == "long" else position.entry_price - float(bar.low)
            state["max_mfe_r"] = max(state["max_mfe_r"], favorable / risk)

        stop_hit = float(bar.low) <= position.current_stop if position.side == "long" else float(bar.high) >= position.current_stop
        target_count = len(self.profile.ratios)
        targets_hit = [
            (float(bar.high) >= position.setup.fib.targets[i] if position.side == "long" else float(bar.low) <= position.setup.fib.targets[i])
            for i in range(target_count)
        ]
        if stop_hit and (self.config.execution_policy == "conservative" or not any(targets_hit)):
            reason = "atr_trailing_stop" if asset in self._runner else ("post_tp1_stop" if any(position.target_done) else "stop_before_tp1")
            if reason == "atr_trailing_stop":
                state["runner_exit"] = True
                target = position.setup.fib.targets[target_count - 1]
                state["runner_outperformed"] = position.current_stop > target if position.side == "long" else position.current_stop < target
            self._close(position, position.remaining, position.current_stop, timestamp, reason)
            return

        for target_index, hit in enumerate(targets_hit):
            if hit and not position.target_done[target_index] and position.remaining > 1e-12:
                quantity = min(position.quantity * self.profile.fractions[target_index], position.remaining)
                self._close(position, quantity, position.setup.fib.targets[target_index], timestamp, f"tp{target_index + 1}", final=False)
                position.target_done[target_index] = True
                if target_index == 0 and asset in self.portfolio.positions:
                    position.current_stop = position.setup.fib.post_tp1_stop
                if target_index == target_count - 1 and self.profile.runner and asset in self.portfolio.positions:
                    self._runner[asset] = {
                        "extreme": float(bar.high) if position.side == "long" else float(bar.low),
                        "target": position.setup.fib.targets[target_count - 1],
                    }

        if asset in self.portfolio.positions and asset in self._runner:
            self._update_runner_stop(asset, timestamp, bar)

    def _update_runner_stop(self, asset, timestamp, bar):
        position = self.portfolio.positions.get(asset)
        if not position:
            return
        state = self._runner[asset]
        state["extreme"] = max(state["extreme"], float(bar.high)) if position.side == "long" else min(state["extreme"], float(bar.low))
        try:
            atr = float(self._atr[asset].loc[pd.Timestamp(timestamp)])
        except (KeyError, TypeError, ValueError):
            return
        if not np.isfinite(atr):
            return
        candidate = state["extreme"] - self.atr_multiplier * atr if position.side == "long" else state["extreme"] + self.atr_multiplier * atr
        position.current_stop = max(position.current_stop, candidate) if position.side == "long" else min(position.current_stop, candidate)

    def _close(self, position, quantity, raw_price, timestamp, reason, final=True):
        before = len(self.portfolio.closed)
        super()._close(position, quantity, raw_price, timestamp, reason, final)
        if len(self.portfolio.closed) > before:
            record = self.portfolio.closed[-1]
            research = self._trade_research.pop(record["setup_id"], {})
            record.update({
                "max_mfe_r": research.get("max_mfe_r", 0.0),
                "runner_exit": research.get("runner_exit", False),
                "runner_outperformed": research.get("runner_outperformed", False),
                "profile": self.profile.name, "atr_length": self.atr_length, "atr_multiplier": self.atr_multiplier,
            })
            self._runner.pop(position.asset, None)


def _atr(bars: pd.DataFrame, length: int | None) -> pd.Series:
    if length is None:
        return pd.Series(np.nan, index=bars.index)
    previous_close = bars.close.shift(1)
    true_range = pd.concat([
        bars.high - bars.low,
        (bars.high - previous_close).abs(),
        (bars.low - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(length, min_periods=length).mean()
