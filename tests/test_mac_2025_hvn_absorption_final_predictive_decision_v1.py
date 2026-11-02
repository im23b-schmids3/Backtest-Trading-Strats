from __future__ import annotations

from src.research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_hvn_absorption_final_predictive_decision_v1 as study


def _path(*rows):
    return [{"landmark_ns": 0, **r} for r in rows]


def test_first_passage_is_direction_adjusted_and_symmetric():
    long = study.first_passage_result(100.0, 1, 2, _path({"ts_recv": 1_000_000_000, "mid": 100.5, "valid": True}), 30_000_000_000)
    short = study.first_passage_result(100.0, -1, 2, _path({"ts_recv": 1_000_000_000, "mid": 99.5, "valid": True}), 30_000_000_000)
    assert long["outcome"] == short["outcome"] == "REVERSAL_FIRST"


def test_first_passage_same_receive_time_opposite_crossing_is_ambiguous():
    result = study.first_passage_result(100.0, 1, 2, _path(
        {"ts_recv": 1_000_000_000, "mid": 100.5, "valid": True},
        {"ts_recv": 1_000_000_000, "mid": 99.5, "valid": True},
    ), 30_000_000_000)
    assert result["outcome"] == "TIED_OR_ORDER_AMBIGUOUS"


def test_first_passage_censors_invalid_quotes_and_large_gaps():
    invalid = study.first_passage_result(100.0, 1, 2, _path(
        {"ts_recv": 500_000_000, "mid": 100.0, "valid": True},
        {"ts_recv": 900_000_000, "mid": None, "valid": False},
    ), 30_000_000_000)
    gap = study.first_passage_result(100.0, 1, 2, _path(
        {"ts_recv": 3_000_000_001, "mid": 100.0, "valid": True},
    ), 30_000_000_000)
    assert invalid["outcome"] == "SOURCE_CENSORED"
    assert gap["outcome"] == "SOURCE_CENSORED"


def test_matching_is_deterministic_and_does_not_reuse_controls():
    rows = [
        {"episode_id": "t1", "date": "d1", "direction": 1, "local_hour_bucket": "09:00", "t": True, "c": False,
         "recent_30s_mid_range_ticks": 2, "recent_30s_mid_change_ticks": 0, "recent_30s_trade_count": 3,
         "recent_30s_trade_volume": 4, "spread_ticks": 1, "zone_width_ticks": 4, "distance_to_setup_boundary_ticks": 1, "history_seconds": 30},
        {"episode_id": "t2", "date": "d1", "direction": 1, "local_hour_bucket": "09:00", "t": True, "c": False,
         "recent_30s_mid_range_ticks": 2.1, "recent_30s_mid_change_ticks": 0, "recent_30s_trade_count": 3,
         "recent_30s_trade_volume": 4, "spread_ticks": 1, "zone_width_ticks": 4, "distance_to_setup_boundary_ticks": 1, "history_seconds": 30},
        {"episode_id": "c1", "date": "d1", "direction": 1, "local_hour_bucket": "09:00", "t": False, "c": True,
         "recent_30s_mid_range_ticks": 2.05, "recent_30s_mid_change_ticks": 0, "recent_30s_trade_count": 3,
         "recent_30s_trade_volume": 4, "spread_ticks": 1, "zone_width_ticks": 4, "distance_to_setup_boundary_ticks": 1, "history_seconds": 30},
    ]
    first, _ = study.greedy_match(rows, "t", "c")
    second, _ = study.greedy_match(rows, "t", "c")
    assert first == second
    assert len(first) == 1
    assert len({control for _, control, _ in first}) == len(first)


def test_bootstrap_is_seeded_and_reports_valid_replicates():
    clusters = {"d1": [1.0, -1.0], "d2": [1.0], "d3": [-1.0]}
    a = study._bootstrap_date_effect(clusters, replicates=500, seed=41)
    b = study._bootstrap_date_effect(clusters, replicates=500, seed=41)
    assert a == b
    assert a["valid_replicates"] == 500
