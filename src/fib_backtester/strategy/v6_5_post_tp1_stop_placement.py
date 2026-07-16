from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopPlacementPolicy:
    name: str
    fib_ratio: float | None
    description: str


STOP_PLACEMENT_POLICIES = {
    "no_stop_movement": StopPlacementPolicy("no_stop_movement", None, "Retain the initial Fib 1.02 stop."),
    "Fib 0.900": StopPlacementPolicy("Fib 0.900", 0.900, "Configured entry level; practical break-even before costs."),
    "Fib 0.890": StopPlacementPolicy("Fib 0.890", 0.890, "Post-TP1 stop placement."),
    "Fib 0.880": StopPlacementPolicy("Fib 0.880", 0.880, "Current V6 baseline."),
    "Fib 0.870": StopPlacementPolicy("Fib 0.870", 0.870, "Post-TP1 stop placement."),
    "Fib 0.860": StopPlacementPolicy("Fib 0.860", 0.860, "Post-TP1 stop placement."),
    "Fib 0.850": StopPlacementPolicy("Fib 0.850", 0.850, "Post-TP1 stop placement."),
    "Fib 0.840": StopPlacementPolicy("Fib 0.840", 0.840, "Post-TP1 stop placement."),
    "Fib 0.830": StopPlacementPolicy("Fib 0.830", 0.830, "Post-TP1 stop placement."),
    "Fib 0.820": StopPlacementPolicy("Fib 0.820", 0.820, "Post-TP1 stop placement."),
    "Fib 0.786": StopPlacementPolicy("Fib 0.786", 0.786, "TP1 price; exploratory aggressive profit lock."),
}

STOP_PLACEMENT_ORDER = (
    "no_stop_movement", "Fib 0.900", "Fib 0.890", "Fib 0.880", "Fib 0.870",
    "Fib 0.860", "Fib 0.850", "Fib 0.840", "Fib 0.830", "Fib 0.820", "Fib 0.786",
)

