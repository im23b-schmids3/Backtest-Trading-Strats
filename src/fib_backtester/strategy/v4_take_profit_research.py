from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .fibonacci import FibLevels
from .signals import Setup
from .v3_entry_research import setup_with_entry_level


@dataclass(frozen=True)
class TakeProfitProfile:
    name: str
    ratios: tuple[float, ...]
    fractions: tuple[float, ...]
    runner_fraction: float = 0.0
    runner: bool = False


PROFILES = {
    "A": TakeProfitProfile("A", (.786, .618, .500, .236, .050), (.15, .20, .20, .25, .20)),
    "B": TakeProfitProfile("B", (.786, .618, .500, .236, .050), (.30, .25, .20, .15, .10)),
    "C": TakeProfitProfile("C", (.786, .500, .000), (.40, .30, .30)),
    "D": TakeProfitProfile("D", (.500, .236, .000), (.20, .20, .20), .40, True),
    "E": TakeProfitProfile("E", (.500, .236, .000), (.30, .30, .20), .20, True),
}

ATR_SETTINGS = ((14, 2.0), (14, 2.5), (14, 3.0), (21, 2.0), (21, 2.5), (21, 3.0))


def profile_levels(setup: Setup, profile: TakeProfitProfile, entry_level: float = .900) -> Setup:
    """Copy V3's fixed-entry setup with only the TP profile changed."""
    baseline = setup_with_entry_level(setup, entry_level)
    low, high = baseline.fib.low, baseline.fib.high
    distance = high - low
    price = (lambda ratio: high - ratio * distance) if baseline.side == "long" else (lambda ratio: low + ratio * distance)
    targets = tuple(price(ratio) for ratio in profile.ratios)
    # Keep the shared Position contract's five target slots; inactive slots are unreachable.
    if baseline.side == "long":
        targets = targets + (float("inf"),) * (5 - len(targets))
    else:
        targets = targets + (float("-inf"),) * (5 - len(targets))
    fib = FibLevels(
        side=baseline.side, low=low, high=high, entry=baseline.fib.entry,
        stop=baseline.fib.stop, targets=targets, post_tp1_stop=baseline.fib.post_tp1_stop,
    )
    return type(baseline)(baseline.identifier, baseline.side, baseline.first, baseline.second, fib, baseline.signal_time)
