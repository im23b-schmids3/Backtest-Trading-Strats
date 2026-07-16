from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .fibonacci import FibLevels, levels
from .swings import Swing


@dataclass(frozen=True)
class Setup:
    identifier: str
    side: Literal["long", "short"]
    first: Swing
    second: Swing
    fib: FibLevels
    signal_time: object


def setup_from_swings(first: Swing, second: Swing, min_distance: int) -> Setup | None:
    if first.kind == second.kind or second.pivot_index <= first.pivot_index:
        return None
    if second.pivot_index - first.pivot_index < min_distance:
        return None
    if first.kind == "low" and second.kind == "high":
        side, low, high = "long", first.price, second.price
    elif first.kind == "high" and second.kind == "low":
        side, low, high = "short", second.price, first.price
    else:
        return None
    if low >= high:
        return None
    return Setup(
        f"{side}-{first.pivot_time.isoformat()}-{second.pivot_time.isoformat()}", side, first, second,
        levels(side, low, high), second.confirmation_time,
    )
