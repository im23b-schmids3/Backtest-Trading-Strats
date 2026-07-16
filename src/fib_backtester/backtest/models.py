from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fib_backtester.strategy.signals import Setup

TP_FRACTIONS = (0.15, 0.20, 0.20, 0.25, 0.20)


@dataclass
class Order:
    asset: str
    setup: Setup
    submission_time: object
    active_from_index: int
    created_index: int


@dataclass
class Position:
    asset: str
    setup: Setup
    quantity: float
    entry_price: float
    entry_raw_price: float
    entry_time: object
    order_submission_time: object
    risk_budget: float
    initial_stop: float
    current_stop: float
    remaining: float
    entry_fee: float
    slippage_cost: float
    target_done: list[bool] = field(default_factory=lambda: [False] * 5)
    exit_value: float = 0.0
    exit_qty: float = 0.0
    exit_fee: float = 0.0
    stop_reason: str | None = None
    exit_events: list[dict] = field(default_factory=list)

    @property
    def side(self) -> Literal["long", "short"]:
        return self.setup.side
