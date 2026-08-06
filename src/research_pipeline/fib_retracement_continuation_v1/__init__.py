"""Sealed FibRetracementContinuation prospective V1 implementation.

This package contains deterministic mechanics and synthetic-only test support.
It deliberately has no implicit market-data discovery.
"""
from .constants import STRATEGY_ID, CANDIDATES, TERMINAL_OUTCOMES
from .runner import run_synthetic, materialize_synthetic, run_development, run_holdout

__all__ = ["STRATEGY_ID", "CANDIDATES", "TERMINAL_OUTCOMES", "run_synthetic", "materialize_synthetic", "run_development", "run_holdout"]
