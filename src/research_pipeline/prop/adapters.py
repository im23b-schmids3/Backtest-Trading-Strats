from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import TradeSignal
from .synthetic import synthetic_trades
from ..compliance import ComplianceEvaluator, PropFirmPolicy


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


class AlertComplianceAdapter:
    """Alert-path facade using the same evaluator as the native backtest."""

    def __init__(self, policy: PropFirmPolicy, evaluator: ComplianceEvaluator | None = None):
        self.policy = policy
        self.evaluator = evaluator or ComplianceEvaluator()

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate_alert(policy=self.policy, **kwargs)
