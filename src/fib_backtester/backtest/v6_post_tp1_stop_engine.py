from __future__ import annotations

from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.backtest.v4_take_profit_engine import StrategyV4TakeProfitResearchEngine
from fib_backtester.strategy.v4_take_profit_research import PROFILES
from fib_backtester.strategy.v4_take_profit_research import profile_levels
from fib_backtester.strategy.v6_post_tp1_stop_research import PostTP1StopPolicy


class StrategyV6PostTP1StopResearchEngine(StrategyV4TakeProfitResearchEngine):
    """V4 Profile B with the post-TP1 stop rule as the only research variable."""

    strategy_version = "Strategy_V6_PostTP1StopResearch"

    def __init__(self, config, min_move: float, policy: PostTP1StopPolicy):
        super().__init__(config, min_move, PROFILES["B"])
        self.policy = policy

    def _apply_events(self, asset, index):
        events = self.construction[asset].events_by_index.get(index, [])
        transformed = [
            event if event.setup is None else event.__class__(
                event.action, event.side, event.setup_id, event.index,
                profile_levels(event.setup, PROFILES["B"], .900), event.reason, event.trend_id,
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
        # This is intentionally the V4 conservative sequence: inspect the original
        # stop first, then process TP fills. A stop moved by TP1 is only active on
        # the next candle and can never retroactively fill on the TP1 candle.
        stop_hit = float(bar.low) <= position.current_stop if position.side == "long" else float(bar.high) >= position.current_stop
        targets_hit = [
            (float(bar.high) >= target if position.side == "long" else float(bar.low) <= target)
            for target in position.setup.fib.targets
        ]
        if stop_hit and (self.config.execution_policy == "conservative" or not any(targets_hit)):
            self._close(position, position.remaining, position.current_stop, timestamp, "post_tp1_stop" if any(position.target_done) else "stop_before_tp1")
            return
        for target_index, hit in enumerate(targets_hit):
            if hit and not position.target_done[target_index] and position.remaining > 1e-12:
                quantity = min(position.quantity * self.profile.fractions[target_index], position.remaining)
                self._close(position, quantity, position.setup.fib.targets[target_index], timestamp, f"tp{target_index + 1}", final=False)
                position.target_done[target_index] = True
                if target_index == 0 and asset in self.portfolio.positions and self.policy.fib_ratio is not None:
                    position.current_stop = _fib_price(position.side, position.setup.fib.low, position.setup.fib.high, self.policy.fib_ratio)

    def _close(self, position, quantity, raw_price, timestamp, reason, final=True):
        before = len(self.portfolio.closed)
        super()._close(position, quantity, raw_price, timestamp, reason, final)
        if len(self.portfolio.closed) > before:
            self.portfolio.closed[-1].update({
                "post_tp1_policy": self.policy.name,
                "post_tp1_stop_fib_ratio": self.policy.fib_ratio,
                "post_tp1_stop_description": self.policy.description,
            })


def _fib_price(side, low, high, ratio):
    distance = high - low
    return high - ratio * distance if side == "long" else low + ratio * distance
