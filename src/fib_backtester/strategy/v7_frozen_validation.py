"""Constants for the frozen V7 validation specification.

This module deliberately contains no optimization logic.  The candidate distance
and move values are the already-established research grid; selection is performed
by the validation harness inside training windows only.
"""

from __future__ import annotations

from dataclasses import dataclass


FROZEN_ENTRY = 0.900
FROZEN_INITIAL_STOP = 1.020
FROZEN_POST_TP1_STOP = 0.820
FROZEN_TP_RATIOS = (0.786, 0.618, 0.500, 0.236, 0.050)
FROZEN_TP_FRACTIONS = (0.30, 0.25, 0.20, 0.15, 0.10)

# Existing V2/V6 research candidates.  These are selected only inside training
# windows; they are not new optimization values.
VALIDATION_DISTANCES = (4, 7, 10, 13, 16)
VALIDATION_MIN_MOVES = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
VALIDATION_CANDIDATES = tuple(
    (distance, minimum_move)
    for distance in VALIDATION_DISTANCES
    for minimum_move in VALIDATION_MIN_MOVES
)

STABILITY_ENTRY_LEVELS = (0.89, 0.90, 0.91)
STABILITY_POST_TP1_STOPS = (0.81, 0.82, 0.83)
STABILITY_INITIAL_STOPS = (1.01, 1.02, 1.03)
STABILITY_TP_ALLOCATIONS = {
    "30/25/20/15/10": (0.30, 0.25, 0.20, 0.15, 0.10),
    "35/25/20/10/10": (0.35, 0.25, 0.20, 0.10, 0.10),
    "25/25/20/15/15": (0.25, 0.25, 0.20, 0.15, 0.15),
}

STRESS_SCENARIOS = (
    "baseline",
    "2x_fees",
    "3x_fees",
    "2x_slippage",
    "3x_slippage",
    "missed_fills_5pct",
    "missed_fills_10pct",
    "delayed_execution",
    "adverse_fills",
)

# Adverse fills are defined explicitly because the request does not prescribe a
# numerical shock: one additional 5 bp adverse execution cushion is applied on
# top of the configured slippage, on every fill.
ADVERSE_FILL_EXTRA_SLIPPAGE = 0.0005


@dataclass(frozen=True)
class FrozenSpecification:
    entry_level: float = FROZEN_ENTRY
    initial_stop: float = FROZEN_INITIAL_STOP
    post_tp1_stop: float = FROZEN_POST_TP1_STOP
    tp_ratios: tuple[float, ...] = FROZEN_TP_RATIOS
    tp_fractions: tuple[float, ...] = FROZEN_TP_FRACTIONS


FROZEN_SPECIFICATION = FrozenSpecification()
