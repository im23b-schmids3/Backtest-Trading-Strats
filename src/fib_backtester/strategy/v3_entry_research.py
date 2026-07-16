from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .fibonacci import FibLevels
from .signals import Setup


ENTRY_LEVELS = (0.850, 0.882, 0.900, 0.935)


def entry_research_levels(side: Literal["long", "short"], low: float, high: float, entry_level: float) -> FibLevels:
    """Return V2-identical Fibonacci levels with only the entry ratio changed."""
    if entry_level not in ENTRY_LEVELS:
        raise ValueError(f"entry_level must be one of {ENTRY_LEVELS}")
    if not 0 < low < high:
        raise ValueError("Fibonacci range must have positive low < high")
    distance = high - low
    price = (lambda ratio: high - ratio * distance) if side == "long" else (lambda ratio: low + ratio * distance)
    return FibLevels(
        side=side,
        low=low,
        high=high,
        entry=price(entry_level),
        stop=price(1.02),
        targets=tuple(price(ratio) for ratio in (0.786, 0.618, 0.5, 0.236, 0.05)),
        post_tp1_stop=price(0.88),
    )


def setup_with_entry_level(setup: Setup, entry_level: float) -> Setup:
    """Copy a setup while changing only its Fib entry price."""
    fib = entry_research_levels(setup.side, setup.fib.low, setup.fib.high, entry_level)
    return replace(setup, fib=fib)
