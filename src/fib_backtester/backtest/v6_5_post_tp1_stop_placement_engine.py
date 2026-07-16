from __future__ import annotations

from fib_backtester.backtest.v6_post_tp1_stop_engine import StrategyV6PostTP1StopResearchEngine
from fib_backtester.strategy.v6_5_post_tp1_stop_placement import StopPlacementPolicy


class StrategyV65PostTP1StopPlacementEngine(StrategyV6PostTP1StopResearchEngine):
    """V6 Profile-B execution with a research-only post-TP1 stop placement."""

    strategy_version = "Strategy_V6_5_PostTP1StopPlacement"

    def __init__(self, config, min_move: float, policy: StopPlacementPolicy):
        super().__init__(config, min_move, policy)

