from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import TradeSignal
from .synthetic import synthetic_trades


class TradeSignalAdapter(Protocol):
    """Boundary for a real strategy-research trade-signal provider.

    Implementations must return chronological, fully described settled or
    settleable signals. They must not mutate the frozen strategy candidate.
    """

    def signals(self, strategy_id: str, scenario: str) -> Sequence[TradeSignal]: ...


class SyntheticTradeSignalAdapter:
    """Explicit fixture adapter; never claims native futures evidence."""

    def __init__(self, repository_root: str | Path = "."):
        self.repository_root = Path(repository_root)

    def signals(self, strategy_id: str, scenario: str) -> Sequence[TradeSignal]:
        return synthetic_trades(scenario)
