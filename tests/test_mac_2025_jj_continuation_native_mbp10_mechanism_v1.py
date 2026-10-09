from __future__ import annotations

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_jj_continuation_native_mbp10_mechanism_v1 as study


def test_frozen_pair_filter_and_target_timestamps_are_deterministic() -> None:
    rows = [{"estimand": "DISPLACEMENT_ONLY", "signal_id": "2025-03-03|NY_AM|X|1|1741012320000000000",
             "date": "2025-03-03", "period": "SPRING_2025", "session": "NY_AM", "direction": "LONG",
             "signal_minute": "1", "control_minute": "4", "signal_net_5m_ticks": "-2",
             "control_net_5m_ticks": "1", "signal_minus_control_5m_ticks": "-3",
             "signal_mfe_net_ticks": "3", "signal_mae_ticks": "5", "control_mfe_net_ticks": "2", "control_mae_ticks": "1"}]
    targets = study._targets(rows)["2025-03-03"]
    assert [r["role"] for r in targets] == ["SIGNAL", "CONTROL"]
    assert targets[0]["timestamp_ns"] == 1741012320000000000
    assert targets[1]["timestamp_ns"] == study._session_start_ns("2025-03-03", "NY_AM") + 5 * 60 * study.NS


def test_native_window_merging_is_stable_and_keeps_disjoint_windows() -> None:
    ns = study.NS
    events = [{"timestamp_ns": 100 * ns}, {"timestamp_ns": 160 * ns}, {"timestamp_ns": 700 * ns}]
    assert study._merged_intervals(events) == [(35 * ns, 461 * ns), (635 * ns, 1001 * ns)]


def test_contract_freezes_predictor_path_and_marks_postsignal_as_explanatory() -> None:
    contract = study._input_contract()
    assert contract["native_time"].startswith("DBN ts_recv")
    assert "post_signal_label" in contract
    assert contract["threshold_search"] is False
    assert contract["optimization"] is False


def test_mechanism_decision_requires_cross_period_coherence() -> None:
    fields = ("aggressive_flow_30s", "price_impact_ticks_per_directional_contract",
              "top5_imbalance_directional", "mlofi_top5_500ms_directional",
              "resiliency_restore_to_consumption_ratio")
    analysis = {"periods": {feature: {period: {"paired_difference": {"mean": None},
        "paired_feature_difference_vs_markout_difference": {"pearson": None},
        "paired_feature_outcome_correlation_date_cluster_ci95": None} for period in ("SPRING_2025", "OCTOBER_2025")}
        for feature in fields}}
    result = study._scientific_decision([], analysis, {})
    assert result["decision"] == "INSUFFICIENT_VALID_DEPTH_EVIDENCE"


def test_exact_fee_and_slippage_units_are_declared() -> None:
    contract = study._input_contract()
    assert "0.48" in contract["markout_execution"]
    assert "one adverse tick each side" in contract["markout_execution"]


def test_secondary_displacement_population_is_separate_and_frozen() -> None:
    signals = [{"model": "OPENING_CONTINUATION", "trigger": "DISPLACEMENT_CANDLE",
        "signal_id": f"2025-03-{i:02d}|NY_AM|X|{i}", "date": "2025-03-03",
        "period": "SPRING_2025", "session": "NY_AM", "direction": "LONG",
        "direction_sign": 1, "signal_timestamp_ns": i} for i in range(1, 143)]
    signals.append({**signals[0], "trigger": "BOS_ONLY", "signal_id": "excluded"})
    targets = study._secondary_targets(signals)
    flattened = [event for events in targets.values() for event in events]
    assert len(flattened) == 142
    assert all(event["role"] == "SECONDARY_SIGNAL" for event in flattened)
    assert all(event["pair_id"].startswith("SECONDARY|") for event in flattened)


def test_postsignal_comparison_is_paired_and_keeps_period_identity() -> None:
    rows = []
    for role, mid, opposing in (("SIGNAL", -2.0, -4.0), ("CONTROL", 1.0, 2.0)):
        rows.append({"pair_id": "p1", "role": role, "date": "2025-03-03", "period": "SPRING_2025",
            "window_end_seconds": 5, "complete": True, "directional_mid_change_ticks": mid,
            "opposing_depth_change": opposing, "directional_aggressive_contracts": 10.0,
            "directional_top5_price_keyed_ofi": 5.0})
    result = study._analyze_postsignal(rows)
    five = result["windows"]["5s"]
    assert five["complete_matched_paths"] == 1
    assert five["paired_signal_minus_control"]["SPRING_2025"]["directional_mid_change_ticks"]["mean"] == -3.0
    assert result["semantics"].startswith("post-signal explanatory outcomes")
