from __future__ import annotations

from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.strategy.v3_entry_research import setup_with_entry_level


class StrategyV3EntryResearchEngine(StrategyV2Engine):
    """V2 execution/lifecycle machinery with an entry-only research override."""

    strategy_version = "Strategy_V3_EntryResearch"

    def __init__(self, config, min_move: float, entry_level: float):
        super().__init__(config, min_move)
        self.entry_level = entry_level

    def _apply_events(self, asset, index):
        events = self.construction[asset].events_by_index.get(index, [])
        transformed = [
            event if event.setup is None else event.__class__(
                event.action, event.side, event.setup_id, event.index,
                setup_with_entry_level(event.setup, self.entry_level), event.reason, event.trend_id,
            )
            for event in events
        ]
        self.construction[asset].events_by_index[index] = transformed
        try:
            super()._apply_events(asset, index)
        finally:
            self.construction[asset].events_by_index[index] = events
