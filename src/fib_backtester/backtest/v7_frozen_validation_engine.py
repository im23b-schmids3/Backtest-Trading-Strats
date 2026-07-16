from __future__ import annotations

from dataclasses import replace

import numpy as np

from fib_backtester.backtest.execution import CostModel
from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.backtest.v6_5_post_tp1_stop_placement_engine import StrategyV65PostTP1StopPlacementEngine
from fib_backtester.strategy.fibonacci import FibLevels
from fib_backtester.strategy.signals import Setup
from fib_backtester.strategy.v4_take_profit_research import TakeProfitProfile
from fib_backtester.strategy.v6_5_post_tp1_stop_placement import StopPlacementPolicy
from fib_backtester.strategy.v7_frozen_validation import (
    FROZEN_ENTRY,
    FROZEN_INITIAL_STOP,
    FROZEN_POST_TP1_STOP,
    FROZEN_TP_FRACTIONS,
    FROZEN_TP_RATIOS,
    ADVERSE_FILL_EXTRA_SLIPPAGE,
)


class StrategyV7FrozenValidationEngine(StrategyV65PostTP1StopPlacementEngine):
    """V6.5 execution with a frozen default and validation-only controls.

    The default constructor is exactly the frozen V7 strategy.  The optional
    arguments are used only by the V7 sensitivity and stress harness; no older
    strategy implementation is modified.
    """

    strategy_version = "Strategy_V7_FrozenValidation"

    def __init__(
        self,
        config,
        min_move: float,
        *,
        entry_level: float = FROZEN_ENTRY,
        initial_stop: float = FROZEN_INITIAL_STOP,
        post_tp1_stop: float = FROZEN_POST_TP1_STOP,
        tp_fractions: tuple[float, ...] = FROZEN_TP_FRACTIONS,
        fee_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
        missed_fill_probability: float = 0.0,
        delay_bars: int = 0,
        adverse_fill_extra_slippage: float = 0.0,
        random_seed: int = 42,
    ):
        policy = StopPlacementPolicy(
            "Fib 0.820" if post_tp1_stop == FROZEN_POST_TP1_STOP else f"Fib {post_tp1_stop:.3f}",
            post_tp1_stop,
            "Frozen V7 post-TP1 stop",
        )
        super().__init__(config, min_move, policy)
        if len(tp_fractions) != len(FROZEN_TP_RATIOS) or abs(sum(tp_fractions) - 1.0) > 1e-9:
            raise ValueError("V7 TP fractions must contain five values summing to one")
        if not 0 < entry_level < 1 or not 1.0 <= initial_stop <= 1.1 or not 0 < post_tp1_stop < 1:
            raise ValueError("invalid V7 Fibonacci perturbation")
        self.entry_level = float(entry_level)
        self.initial_stop = float(initial_stop)
        self.post_tp1_stop = float(post_tp1_stop)
        self.profile = TakeProfitProfile("B", FROZEN_TP_RATIOS, tuple(float(v) for v in tp_fractions))
        self.fee_multiplier = float(fee_multiplier)
        self.slippage_multiplier = float(slippage_multiplier)
        self.missed_fill_probability = float(missed_fill_probability)
        self.delay_bars = int(delay_bars)
        self.adverse_fill_extra_slippage = float(adverse_fill_extra_slippage)
        self._rng = np.random.default_rng(random_seed)
        self.missed_fill_attempts = 0

    def _apply_events(self, asset, index):
        # Use the V2 event processor directly, exactly as V6/V6.5 do, while
        # replacing only the validation harness' temporary Fibonacci levels.
        events = self.construction[asset].events_by_index.get(index, [])
        transformed = [
            event if event.setup is None else event.__class__(
                event.action,
                event.side,
                event.setup_id,
                event.index,
                _research_setup(event.setup, self.entry_level, self.initial_stop, self.post_tp1_stop, self.profile),
                event.reason,
                event.trend_id,
            )
            for event in events
        ]
        self.construction[asset].events_by_index[index] = transformed
        try:
            StrategyV2Engine._apply_events(self, asset, index)
        finally:
            self.construction[asset].events_by_index[index] = events

        if self.delay_bars:
            for order in self.v2_orders.values():
                if order.created_index == index:
                    order.active_from_index += self.delay_bars

    def _fill_v2_order(self, order, timestamp):
        if self.missed_fill_probability > 0 and self._rng.random() < self.missed_fill_probability:
            self.missed_fill_attempts += 1
            return False
        return super()._fill_v2_order(order, timestamp)

    def _costs(self, asset: str) -> CostModel:
        base = super()._costs(asset)
        return CostModel(
            base.fee_rate * self.fee_multiplier,
            base.slippage_rate * self.slippage_multiplier + self.adverse_fill_extra_slippage,
        )


def _research_setup(setup: Setup, entry_level: float, initial_stop: float, post_tp1_stop: float, profile: TakeProfitProfile) -> Setup:
    low, high = setup.fib.low, setup.fib.high
    distance = high - low
    price = (lambda ratio: high - ratio * distance) if setup.side == "long" else (lambda ratio: low + ratio * distance)
    fib = FibLevels(
        side=setup.side,
        low=low,
        high=high,
        entry=price(entry_level),
        stop=price(initial_stop),
        targets=tuple(price(ratio) for ratio in profile.ratios),
        post_tp1_stop=price(post_tp1_stop),
    )
    return replace(setup, fib=fib)
