from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    fee_rate: float
    slippage_rate: float

    def fill_price(self, raw: float, side: str, action: str) -> float:
        """Apply slippage in the direction harmful to the position."""
        buy = (side == "long" and action == "entry") or (side == "short" and action == "exit")
        return raw * (1 + self.slippage_rate if buy else 1 - self.slippage_rate)

    def fee(self, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.fee_rate
