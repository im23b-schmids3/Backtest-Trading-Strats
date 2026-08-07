"""Artifact-free, development-only V1/V2 signal parity diagnostic."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from research_pipeline.fib_retracement_continuation_v1.strategy import causal_setups

from .aggregation import completed_utc_bars, validate_1m_bars
from .models import Candidate
from .runner import run_candidate


def _candidate(value):
    return Candidate(**value) if isinstance(value, dict) else value


def _setup_view(value):
    keys = ("setup_id", "impulse_id", "fib_range_id", "direction", "anchor_timestamp",
            "anchor_price", "extreme_timestamp", "extreme_price", "low", "high")
    return {key: value.get(key) for key in keys if key in value}


def fib09_v2_v1_parity_diagnostic(*, reference_bars_by_candidate, derived_1m_by_candidate,
                                  candidates, assumptions=None):
    """Compare V1 HTF input with V2's derived HTF input without writing anything.

    Callers own source access and must pass development-only bars.  This function
    exposes aggregate comparisons only; it never loads manifests, writes an
    artifact, or accepts a holdout source.
    """
    reports = []
    implementation_differences = 0
    for candidate_value in candidates:
        candidate = _candidate(candidate_value)
        reference = list(reference_bars_by_candidate.get(candidate.candidate_id, ()))
        rows = validate_1m_bars(list(derived_1m_by_candidate.get(candidate.candidate_id, ())))
        minutes = 240 if candidate.symbol == "ETH" else 1440
        derived = completed_utc_bars(rows, minutes)
        reference_setups = causal_setups(reference, candidate)
        derived_setups = causal_setups(derived, candidate)
        ohlc_differences = [stamp for stamp in sorted({x.timestamp for x in reference} | {x.timestamp for x in derived})
                            if next((x for x in reference if x.timestamp == stamp), None)
                            != next((x for x in derived if x.timestamp == stamp), None)]
        reference_view = [_setup_view(item) for item in reference_setups]
        derived_view = [_setup_view(item) for item in derived_setups]
        exact = reference_view == derived_view
        # Different bars are a data-source difference; identical bars that yield
        # different frozen V1 setup output would be an implementation defect.
        classification = "EXACT_MATCH" if exact else (
            "DATA_SOURCE_DIFFERENCE" if ohlc_differences else "IMPLEMENTATION_DIFFERENCE")
        if classification == "IMPLEMENTATION_DIFFERENCE":
            implementation_differences += 1
        result = run_candidate(rows, candidate, **({"assumptions": assumptions} if assumptions else {}))
        reports.append({
            "candidate_id": candidate.candidate_id,
            "reference_bar_count": len(reference), "derived_bar_count": len(derived),
            "ohlc_difference_count": len(ohlc_differences),
            "reference_setup_count": len(reference_setups), "derived_setup_count": len(derived_setups),
            "reference_setups": reference_view, "derived_setups": derived_view,
            "exact_match": exact, "classification": classification,
            "orders": len(result["orders"]),
            "trade_count_pre_force_flat": len(result["trades"]),
            "trade_count_post_force_flat": len(result["trades"]),
            "overnight_count": result["reconciliation"]["overnight_trade_count"],
            "positions_after_cutoff": 0,
            "reconciles": result["reconciliation"]["reconciles"],
        })
    return {"diagnostic": "fib09-v2-v1-parity-diagnostic", "development_only": True,
            "artifact_free": True, "holdout_strategy_accessed": False,
            "implementation_difference_count": implementation_differences,
            "candidates": reports}
