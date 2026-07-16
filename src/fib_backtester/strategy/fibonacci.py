from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FibLevels:
    side: Literal["long", "short"]
    low: float
    high: float
    entry: float
    stop: float
    targets: tuple[float, float, float, float, float]
    post_tp1_stop: float


def levels(side: Literal["long", "short"], low: float, high: float) -> FibLevels:
    if not 0 < low < high:
        raise ValueError("Fibonacci range must have positive low < high")
    r = high - low
    price = (lambda f: high - f * r) if side == "long" else (lambda f: low + f * r)
    return FibLevels(side, low, high, price(0.882), price(1.02), tuple(price(f) for f in (0.786, 0.618, 0.5, 0.236, 0.05)), price(0.88))
