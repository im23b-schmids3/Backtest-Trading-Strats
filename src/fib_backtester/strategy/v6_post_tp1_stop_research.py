from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostTP1StopPolicy:
    name: str
    fib_ratio: float | None
    description: str


POST_TP1_POLICIES = {
    "A": PostTP1StopPolicy("A", 0.880, "Current baseline: move remaining stop to Fib 0.880."),
    "B": PostTP1StopPolicy("B", 0.900, "Configured entry level; practical break-even before fees and slippage."),
    "C": PostTP1StopPolicy("C", 0.786, "Profit lock at TP1."),
    "D": PostTP1StopPolicy("D", None, "No stop movement; retain the initial Fib 1.02 stop."),
    "E": PostTP1StopPolicy("E", 0.618, "Exploratory stronger profit lock at Fib 0.618."),
}

