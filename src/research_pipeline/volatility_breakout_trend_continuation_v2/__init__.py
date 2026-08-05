"""Sealed VBTC V2 infrastructure; real Phase B is deliberately absent."""
from .runner import evaluate_bars, materialize_synthetic_contract, run_phase_a
__all__ = ["evaluate_bars", "materialize_synthetic_contract", "run_phase_a"]
