from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Literal

Direction = Literal["LONG", "SHORT"]
@dataclass(frozen=True)
class Candidate:
 candidate_id: str; symbol: str; timeframe: str; post_tp1_ratio: Decimal; min_distance: int; min_move: Decimal; anchor_age_days: int
@dataclass(frozen=True)
class Bar:
 timestamp: datetime; open: Decimal; high: Decimal; low: Decimal; close: Decimal; volume: Decimal = Decimal(0)
@dataclass(frozen=True)
class ExecutionAssumptions:
 fee_rate: Decimal = Decimal(".001"); slippage_rate: Decimal = Decimal(".0002"); quantity_step: Decimal = Decimal(".001"); risk_fraction: Decimal = Decimal(".02"); opening_equity: Decimal = Decimal("10000")
