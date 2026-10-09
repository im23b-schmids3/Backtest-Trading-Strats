from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_jj_fair_pricing_expanded_v2 as study
from src.research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import EVENT_DTYPE


def _events(rows):
    out = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, (ts, bid, ask, px, size, aggr) in enumerate(rows):
        out[i] = (ts, bid, ask, px, size, aggr, 2)
    return out


def test_session_bounds_are_timezone_aware_across_dst():
    am_spring, _ = study._session_ns("2025-03-03", "NY_AM")
    am_october, _ = study._session_ns("2025-10-07", "NY_AM")
    pm_spring, _ = study._session_ns("2025-03-03", "NY_PM")
    assert datetime.fromtimestamp(am_spring / 1e9, timezone.utc).hour == 14
    assert datetime.fromtimestamp(am_october / 1e9, timezone.utc).hour == 13
    assert datetime.fromtimestamp(pm_spring / 1e9, timezone.utc).hour == 19


def test_displacement_candle_requires_body_wick_and_opposite_color():
    prior = {"open": 101.0, "close": 100.5, "high": 101.25, "low": 100.25}
    valid = {"open": 100.75, "close": 101.5, "high": 101.75, "low": 100.5}
    assert study.displacement_candle(valid, prior, 1)["valid"]
    no_wick_break = {**valid, "close": 101.0}
    assert not study.displacement_candle(no_wick_break, prior, 1)["valid"]
    same_color_prior = {**prior, "close": 101.1}
    assert not study.displacement_candle(valid, same_color_prior, 1)["valid"]


def test_reversion_direction_points_toward_anchor_after_continuation_phase():
    # Above-anchor price reverts short; below-anchor price reverts long.
    assert study._model_direction(100.0, 101.0, 15) == ("FAIR_PRICE_REVERSION", -1)
    assert study._model_direction(100.0, 99.0, 15) == ("FAIR_PRICE_REVERSION", 1)
    assert study._model_direction(100.0, 101.0, 14) == ("OPENING_CONTINUATION", 1)
    assert study._model_direction(100.0, 100.0, 15) is None


def test_bos_uses_only_completed_two_prior_bars_and_strict_inequality():
    bm = {
        1: {"low": 100.0, "high": 101.0, "close": 100.5},
        2: {"low": 99.75, "high": 101.25, "close": 100.5},
        3: {"low": 99.5, "high": 101.0, "close": 99.5},
    }
    assert study.v1.bos_for_bar(bm, 3, 1)["confirmed"]
    bm[3]["close"] = 99.75
    bm[2]["low"] = 99.75
    bm[3]["close"] = 99.75
    assert not study.v1.bos_for_bar(bm, 3, 1)["confirmed"]
    assert not study.v1.bos_for_bar({1: bm[1], 3: bm[3]}, 3, 1)["confirmed"]


def test_excursion_map_resets_on_anchor_touch_and_uses_causal_prefix():
    ts = np.arange(5, dtype=np.int64)
    prices = np.array([100.25, 101.0, 100.0, 99.75, 100.5])
    ids, starts, info = study._episode_map(ts, prices, 100.0)
    assert ids[0] == ids[1] != ids[3]
    assert starts[1] == 0
    assert info[ids[0]][0] == 0
    assert ids[2] != ids[1]


def test_causal_flow_features_exclude_future_records_and_mark_depth_unavailable():
    t = 10_000_000_000
    ev = _events([
        (t - 40_000_000_000, 99.75, 100.0, 99.75, 3, -1),
        (t - 20_000_000_000, 99.75, 100.0, 100.0, 5, 1),
        (t + 1_000_000, 100.0, 100.25, 100.25, 50, 1),
    ])
    feat = study.causal_features(ev, t, t - 60_000_000_000, t + 60_000_000_000, 1)
    assert feat["delta_30s_contracts"] == 5
    assert feat["session_cvd_contracts"] == 2
    assert feat["top5_depth_imbalance"] is None
    assert feat["normalized_mlofi_persistence"] is None


def test_sequential_cell_suppresses_signal_while_position_open():
    start = study._session_ns("2025-03-03", "NY_AM")[0]
    end = start + 60 * 60 * 1_000_000_000
    s1 = start + 30 * 60 * 1_000_000_000
    s2 = s1 + 5_000_000
    s3 = s1 + 20_000_000
    ev = _events([
        (s1 + 3_000_000, 100.0, 100.25, np.nan, 0, 0),
        (s1 + 10_000_000, 101.5, 101.75, np.nan, 0, 0),
        (s3 + 3_000_000, 100.0, 100.25, np.nan, 0, 0),
        (s3 + 10_000_000, 101.5, 101.75, np.nan, 0, 0),
    ])
    signals = [
        {"signal_timestamp_ns": s1, "direction_sign": 1, "direction": "LONG", "date": "2025-03-03", "period": "SPRING_2025", "session": "NY_AM", "model": "OPENING_CONTINUATION", "trigger": "BOS_ONLY", "signal_id": f"s{i}", "anchor_price": 95.0, "episode_adverse_extreme": 100.0, "minute_index": 30}
        for i in range(1)
    ] + [
        {"signal_timestamp_ns": s2, "direction_sign": 1, "direction": "LONG", "date": "2025-03-03", "period": "SPRING_2025", "session": "NY_AM", "model": "OPENING_CONTINUATION", "trigger": "BOS_ONLY", "signal_id": "s2", "anchor_price": 95.0, "episode_adverse_extreme": 100.0, "minute_index": 30},
        {"signal_timestamp_ns": s3, "direction_sign": 1, "direction": "LONG", "date": "2025-03-03", "period": "SPRING_2025", "session": "NY_AM", "model": "OPENING_CONTINUATION", "trigger": "BOS_ONLY", "signal_id": "s3", "anchor_price": 95.0, "episode_adverse_extreme": 100.0, "minute_index": 30},
    ]
    bars = [{"minute_index": 28, "low": 99.0, "high": 100.0, "close": 100.0}, {"minute_index": 29, "low": 99.0, "high": 100.0, "close": 100.0}]
    result = study.execute_sequential_cell(ev, signals, bars, start, end, "EPISODE_STRUCTURAL", "FIXED_1R")
    assert result["overlap_suppressed_signals"] == 1
    assert len(result["trades"]) == 2
    assert result["trades"][0]["entry_timestamp_ns"] >= s1 + 2_000_000


def test_zero_trade_metrics_are_explicit_and_safe():
    result = study._metrics([])
    assert result["trade_count"] == 0
    assert result["total_net_r"] == 0
    assert result["average_net_r"] is None


def test_feature_analysis_reports_periods_and_aggression_reversal_without_threshold_search():
    rows = [
        {"period": "SPRING_2025", "model": "OPENING_CONTINUATION", "local_delta_30s_supportive_fraction": 0.5,
         "aggression_reversal": True, "mean_net_r_across_executed_fixed_cells": 1.0},
        {"period": "SPRING_2025", "model": "OPENING_CONTINUATION", "local_delta_30s_supportive_fraction": -0.5,
         "aggression_reversal": False, "mean_net_r_across_executed_fixed_cells": -1.0},
        {"period": "OCTOBER_2025", "model": "FAIR_PRICE_REVERSION", "local_delta_30s_supportive_fraction": 0.25,
         "aggression_reversal": True, "mean_net_r_across_executed_fixed_cells": -0.5},
    ]
    result = study._feature_analysis(rows)
    assert result["period_relationships"]["local_delta_30s_supportive_fraction"]["SPRING_2025"]["n"] == 2
    assert result["aggression_reversal_outcomes"]["SPRING_2025"]["REVERSAL_TRUE"]["signals"] == 1
    assert result["aggression_reversal_outcomes"]["OCTOBER_2025"]["REVERSAL_TRUE"]["mean_event_net_r"] == -0.5
    assert "TOP5_DEPTH_IMBALANCE" in result["unavailable"]
