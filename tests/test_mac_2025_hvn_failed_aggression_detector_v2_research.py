from __future__ import annotations

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_hvn_failed_aggression_detector_v2_research import (
    SPEC,
    canonical_hash,
    depth_response,
    final_status,
    _forward_rows,
    qualifies_effort,
    state_transition_invariants,
    touch_row_matches,
    weak_impact,
)


def test_frozen_specification_hash_is_deterministic_and_outcome_blind():
    assert canonical_hash(SPEC) == canonical_hash(SPEC)
    assert SPEC["constraints"]["profitability_optimization"] is False
    assert SPEC["constraints"]["threshold_search"] is False


@pytest.mark.parametrize(
    "volume,threshold,count,persistence,expected",
    [(100, 50, 2, .5, True), (99, 50, 2, 1, False), (100, 50, 1, 1, False),
     (100, None, 2, 1, False), (100, 50, 2, .49, False)],
)
def test_effort_requires_sustained_attack_and_threshold(volume, threshold, count, persistence, expected):
    assert qualifies_effort(volume, threshold, count, persistence) is expected


def test_weak_impact_is_tick_based_and_keeps_boundary_inclusive():
    assert weak_impact(1.0)
    assert not weak_impact(1.00001)


def test_depth_response_requires_original_price_persistence_or_observed_recovery():
    assert depth_response([5, 1], False, False)
    assert depth_response([5, 0], True, True)
    assert not depth_response([5, 0], True, False)
    assert not depth_response([], False, True)


def test_touch_match_uses_exact_receive_time_and_touch_price():
    class Ep:
        t0 = 100
        touch_price = 10.0

    class BookRow:
        def __getitem__(self, key):
            if key == "action": return b"M"
            if key == "flags": return 0
            if key.startswith("bid_px_"): return 10_000_000_000 if key.endswith("00") else 0
            if key.startswith("bid_sz_"): return 3 if key.endswith("00") else 0
            if key.startswith("ask_px_"): return 10_250_000_000 if key.endswith("00") else 0
            if key.startswith("ask_sz_"): return 4 if key.endswith("00") else 0
            raise KeyError(key)

    row = BookRow()
    assert touch_row_matches(Ep(), row, 100)
    assert not touch_row_matches(Ep(), row, 101)


def test_rejection_cannot_precede_candidate_and_transition_clock_is_recv():
    rows = [
        {"transition": "HVN_TOUCH", "ts_recv": 100, "ts_event": 99},
        {"transition": "AGGRESSION_OBSERVED", "ts_recv": 101, "ts_event": 100},
        {"transition": "AGGRESSION_EFFORT_ESTABLISHED", "ts_recv": 102, "ts_event": 101},
        {"transition": "PRICE_IMPACT_EVALUATED", "ts_recv": 103, "ts_event": 102},
        {"transition": "DEPTH_RESPONSE_EVALUATED", "ts_recv": 104, "ts_event": 103},
        {"transition": "FAILED_AGGRESSION_CANDIDATE", "ts_recv": 105, "ts_event": 104},
        {"transition": "REJECTION_OBSERVED", "ts_recv": 106, "ts_event": 105},
    ]
    state_transition_invariants(rows)
    with pytest.raises(ValueError, match="cannot precede candidate"):
        state_transition_invariants([rows[0], rows[-1]])


def test_outcome_horizons_are_frozen_and_fixed():
    assert SPEC["forward"]["horizons_ms"] == [1000, 5000, 10000, 30000]


def test_forward_rows_use_each_fixed_horizon_endpoint_and_censor_independently():
    class Ep:
        episode_id = "d|z|t"
        date = "d"
    state = {"ep": Ep(), "candidate": {}, "continuation": None, "trackers": [{
        "stage": "FAILED_AGGRESSION_CANDIDATE", "ts_recv": 100, "ts_event": 99,
        "reference_mid": 100.0, "direction": 1,
        "horizons": {
            1000: {"covered": True, "gap": False, "end": 1000100, "last_mid": 100.25, "mfe": 1.0, "mae": -.5},
            5000: {"covered": False, "gap": False, "end": 5000100, "last_mid": 100.0, "mfe": 0.0, "mae": -1.0},
            10000: {"covered": False, "gap": True, "end": 10000100, "last_mid": 99.75, "mfe": 0.0, "mae": -1.0},
            30000: {"covered": True, "gap": False, "end": 30000100, "last_mid": 100.5, "mfe": 2.0, "mae": -1.0},
        },
    }]}
    rows = _forward_rows([{"states": [state]}])
    assert [r["valid"] for r in rows] == [True, False, False, True]
    assert rows[0]["markout_ticks"] == 1.0
    assert rows[-1]["markout_ticks"] == 2.0


def test_first_terminal_transition_resolves_candidate_continuation_rejection_path():
    assert final_status(10, 20, 30) == "CONTINUATION"
    assert final_status(10, 30, 20) == "FAILED_AGGRESSION_REJECTED"
    assert final_status(10, None, 20) == "FAILED_AGGRESSION_REJECTED"
    assert final_status(10, None, None) == "FAILED_AGGRESSION_NO_REJECTION"
    assert final_status(None, None, None) == "UNRESOLVED"
    assert final_status(10, 20, 20) == "CONTINUATION"
