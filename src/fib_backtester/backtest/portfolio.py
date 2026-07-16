from __future__ import annotations

from dataclasses import dataclass, field

from .models import Position


@dataclass
class Portfolio:
    cash: float
    reserved_notional: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[dict] = field(default_factory=list)

    def equity(self, marks: dict[str, float]) -> float:
        unrealized = 0.0
        for asset, position in self.positions.items():
            mark = marks.get(asset, position.entry_price)
            direction = 1 if position.side == "long" else -1
            unrealized += direction * (mark - position.entry_price) * position.remaining
        return self.cash + unrealized

    def planned_risk(self) -> float:
        return sum(abs(p.entry_price - p.current_stop) * p.remaining for p in self.positions.values())
