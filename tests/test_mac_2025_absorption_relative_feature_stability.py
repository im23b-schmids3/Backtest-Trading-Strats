from __future__ import annotations

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_feature_stability as study


def test_primary_variables_are_exactly_the_four_predeclared_families():
    assert tuple(study.FEATURES) == (
        "TREND_EFFICIENCY_5S", "TREND_EFFICIENCY_30S",
        "AGGRESSION_TO_DEPTH_1S", "AGGRESSION_TO_DEPTH_2S",
        "RELATIVE_TRADE_INTENSITY_5S", "RELATIVE_TRADE_INTENSITY_30S",
        "DEPTH_NORMALIZED_MLOFI_1S", "DEPTH_NORMALIZED_MLOFI_5S",
    )
    assert set(study.FAMILIES) == set(study.norm.LIVE_TO_TAPE.values())


def test_fixed_bucket_edges_are_not_searched():
    assert [study._bucket(x, "quintile") for x in (0, .2, .4, .6, .8, .999)] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5"]
    assert [study._bucket(x, "tercile") for x in (0, 1 / 3, 2 / 3, .999)] == ["LOW", "MID", "HIGH", "HIGH"]
    assert [study._bucket(x, "broad") for x in (0, .199, .2, .799, .8, 1)] == ["LOW_0_20", "LOW_0_20", "MID_20_80", "MID_20_80", "HIGH_80_100", "HIGH_80_100"]


def test_custom_percentiles_use_only_prior_date_history():
    history = {"TRADES_PER_SECOND_30S": {"EUROPE:0": [1.0, 2.0]}}
    event = {"interaction_start_ns": study.baseline._session_windows("2025-04-01")["EUROPE"][0] + 1,
             "session": "EUROPE", "TRADES_PER_SECOND_30S": 2.0}
    sample = {"timestamp_ns": event["interaction_start_ns"], "session": "EUROPE", "TRADES_PER_SECOND_30S": 100.0}
    study._rank_custom([event], [sample], history, "2025-04-01")
    assert event["TRADE_INTENSITY_30S_TOD_PERCENTILE"] == 1.0
    assert 100.0 in history["TRADES_PER_SECOND_30S"]["EUROPE:0"]


def test_mlofi_state_is_explicit_and_direction_oriented():
    rows = [
        {"direction": "BUYER_ABSORPTION", "MLOFI_1S_DEPTH_NORMALIZED": 2.0,
         "MLOFI_5S_DEPTH_NORMALIZED": -1.0, "MLOFI_PERSISTENCE_DIRECTIONAL": 0.5},
        {"direction": "SELLER_ABSORPTION", "MLOFI_1S_DEPTH_NORMALIZED": 2.0,
         "MLOFI_5S_DEPTH_NORMALIZED": -1.0, "MLOFI_PERSISTENCE_DIRECTIONAL": -0.5},
        {"direction": "SELLER_ABSORPTION", "MLOFI_PERSISTENCE_DIRECTIONAL": 0.0},
    ]
    study._derive_mlofi_states(rows)
    assert rows[0]["MLOFI_1S_DIRECTIONAL"] == 2.0
    assert rows[1]["MLOFI_1S_DIRECTIONAL"] == -2.0
    assert [r["MLOFI_REVERSAL_STATE"] for r in rows] == ["OPPOSES_REVERSAL", "SUPPORTS_REVERSAL", "NEUTRAL"]


def test_per_date_checkpoint_ranks_are_cleared_before_recalibration():
    event = {"ER_5S_GLOBAL_PERCENTILE": .9, "TRADE_INTENSITY_30S_TOD_PERCENTILE": .1,
             "MLOFI_1S_TOD_PERCENTILE": .8, "ER_5S": .5}
    study._clear_ranks([event])
    assert "ER_5S_GLOBAL_PERCENTILE" not in event
    assert "TRADE_INTENSITY_30S_TOD_PERCENTILE" not in event
    assert "MLOFI_1S_TOD_PERCENTILE" not in event
    assert event["ER_5S"] == .5


def test_overlap_clusters_are_deterministic_and_do_not_mutate_family_events():
    base = {"interaction_start_ns": 10, "price": 5000.0, "direction": "BUYER_ABSORPTION",
            "family": "EUROPE|EUROPE|CURRENT|HIGH", "event_id": "a", "markouts_ticks": {"5000": 1.0}}
    twin = {**base, "family": "NY|NY|PRIOR|POC", "event_id": "b", "markouts_ticks": {"5000": 3.0}}
    report, dedup = study._overlap([base, twin])
    assert report["overlap_cluster_count"] == 1
    assert report["duplicate_excess_events"] == 1
    assert len(dedup) == 1 and dedup[0]["event_id"] == "a"
    assert len({base["family"], twin["family"]}) == 2


def test_fixed_prototype_gate_definitions_are_only_predeclared_gates():
    gates = study._gate_report([])
    assert tuple(gates["gates"]) == study.GATE_NAMES
    assert set(gates["gates"]) == {"BASELINE", "A", "B", "C", "D", "E"}
    assert "exclude high ER_5S tercile" in gates["definitions"]["A"]
    assert "retain middle/high 5s relative trade intensity" in gates["definitions"]["E"]


def test_only_two_predeclared_coarse_interactions_are_built():
    result = study._interactions([])
    assert set(result) == {"INTERACTION_A_TREND_X_MLOFI", "INTERACTION_B_AGGRESSION_X_TREND"}
    assert len(result["INTERACTION_A_TREND_X_MLOFI"]["cells"]) == 9
    assert len(result["INTERACTION_B_AGGRESSION_X_TREND"]["cells"]) == 9
    assert "MLOFI_REVERSAL_STATE" in result["INTERACTION_A_TREND_X_MLOFI"]["features"]


def test_prototype_gate_b_excludes_only_opposing_normalized_mlofi_state():
    events = [
        {"date": "2025-03-03", "period": "SPRING_2025", "live_family": study.FAMILIES[0],
         "MLOFI_REVERSAL_STATE": "OPPOSES_REVERSAL", "ER_5S_GLOBAL_PERCENTILE": .4,
         "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE": .5, "TRADE_INTENSITY_TOD_PERCENTILE": .5,
         "markouts_ticks": {}, "mfe_mae_ticks": {}, "barriers": {}},
        {"date": "2025-03-03", "period": "SPRING_2025", "live_family": study.FAMILIES[0],
         "MLOFI_REVERSAL_STATE": "SUPPORTS_REVERSAL", "ER_5S_GLOBAL_PERCENTILE": .4,
         "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE": .5, "TRADE_INTENSITY_TOD_PERCENTILE": .5,
         "markouts_ticks": {}, "mfe_mae_ticks": {}, "barriers": {}},
    ]
    gates = study._gate_report(events)["gates"]
    assert gates["B"]["overall"]["event_count"] == 1
    assert gates["B"]["event_retention"] == .5


def test_daily_relationship_reports_all_requested_markout_horizons_and_sign_counts():
    events = []
    for day in ("2025-03-03", "2025-03-04"):
        for idx, value in enumerate((.1, .2, .4, .7, .9)):
            events.append({"date": day, "interaction_start_ns": idx, "session": "EUROPE",
                           "ER_5S_GLOBAL_PERCENTILE": value,
                           "markouts_ticks": {str(h): float(idx) for h in study.DAILY_HORIZONS}})
    report = study._daily_relationship(events, "ER_5S_GLOBAL_PERCENTILE", "quintile")
    assert set(report["dates"]) == {"2025-03-03", "2025-03-04"}
    assert set(report["dates"]["2025-03-03"]["buckets"]["Q1"]["markouts"]) == {"2000", "5000", "10000", "30000"}
    assert report["positive_dates"] == 2


def test_permutation_is_deterministic_and_preserves_group_counts():
    rows = []
    for family in study.FAMILIES[:1]:
        for i in range(10):
            rows.append({"date": "2025-03-03", "period": "SPRING_2025", "live_family": family,
                         "ER_5S_GLOBAL_PERCENTILE": i / 10,
                         "markouts_ticks": {"5000": float(i)}})
    first = study._permutation(rows, repetitions=5)
    second = study._permutation(rows, repetitions=5)
    assert first == second
    assert first["TREND_EFFICIENCY_5S"]["daily_effect_count"] == 1


def test_manifest_periods_are_limited_to_march_april_and_october():
    coverage = study._manifest_and_inputs()[2]
    assert all(d.startswith(("2025-03-", "2025-04-")) for d in coverage["spring_dates"])
    assert all(d.startswith("2025-10-") for d in coverage["october_dates"])
    assert coverage["dependency_used_as_target"] is False
