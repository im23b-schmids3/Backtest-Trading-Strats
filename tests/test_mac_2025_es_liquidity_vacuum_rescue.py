from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_liquidity_vacuum_rescue as rescue
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_liquidity_vacuum_v1 as fixed
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_normalization as relative
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import EVENT_DTYPE


def _parameters(**changes):
    params = {"pressure_window_ms": 500, "pressure_percentile": .90,
              "depth_depletion_threshold": .50, "refill_weakness_threshold": .50,
              "flow_persistence_threshold": .70, "max_favorable_move_before_entry_ticks": 4,
              "stop_ticks": 6, "target_r": 2.0}
    return {**params, **changes}


def _tape(day="2025-03-03"):
    t = fixed.baseline._session_windows(day)["ASIA"][0] + 3_600_000_000_000
    tape = np.zeros(3, dtype=EVENT_DTYPE)
    tape[0] = (t, 99.75, 100.00, np.nan, 0, 0, 0)
    tape[1] = (t+2_000_000, 99.75, 100.00, np.nan, 0, 0, 0)
    tape[2] = (t+1_000_000_000, 103.50, 103.75, np.nan, 0, 0, 0)
    return t, tape


def test_search_space_has_exactly_eight_frozen_discrete_dimensions():
    assert len(rescue.SEARCH_SPACE) == 8
    assert rescue.TRIAL_CAP == 500
    assert rescue.SEARCH_SPACE["pressure_window_ms"] == (250, 500, 750, 1000)
    assert rescue.SEARCH_SPACE["pressure_percentile"] == tuple(round(i/100, 2) for i in range(85,98))
    assert rescue.SEARCH_SPACE["depth_depletion_threshold"][-1] == .65
    assert rescue.SEARCH_SPACE["refill_weakness_threshold"][-1] == .80
    assert rescue.SEARCH_SPACE["stop_ticks"] == tuple(range(4,11))
    assert rescue.SEARCH_SPACE["target_r"][-1] == 3.0
    for name, values in rescue.SEARCH_SPACE.items():
        assert rescue.validate_parameters(_parameters(**{name: values[0]}))
        assert rescue.validate_parameters(_parameters(**{name: values[-1]}))
    with pytest.raises(rescue.RescueError):
        rescue.validate_parameters({**_parameters(), "ninth_dimension": 1})
    with pytest.raises(rescue.RescueError):
        rescue.validate_parameters(_parameters(depth_depletion_threshold=.70))


def test_spring_only_preregistration_and_immutable_formula(tmp_path: Path):
    assert rescue.PREREG["optimization_dates"] == list(fixed.SPRING_DATES)
    assert set(rescue.PREREG["optimization_dates"]).isdisjoint(fixed.OCTOBER_DATES)
    assert rescue.PREREG["october_role"].startswith("SECONDARY_DEV")
    p = tmp_path / "optimization-preregistration.json"
    rescue._write_once(p, rescue.PREREG)
    rescue._write_once(p, rescue.PREREG)
    with pytest.raises(rescue.RescueError):
        rescue._write_once(p, {**rescue.PREREG, "trial_cap": 1000})


def test_objective_is_deterministic_and_enforces_all_three_sample_minima():
    metrics = {"trades": 100, "active_dates": 8, "active_weeks": 5,
               "mean_trade_R": .2, "median_daily_R": .1, "median_weekly_R": .4,
               "max_drawdown_R": 3.0}
    assert rescue.robust_objective(metrics) == pytest.approx(.2+.02+.04-.03)
    assert rescue.robust_objective(metrics) == rescue.robust_objective(dict(metrics))
    for key, value in (("trades",99), ("active_dates",7), ("active_weeks",4)):
        assert rescue.robust_objective({**metrics, key:value}) == -1000.0


def test_fast_clustering_matches_fixed_v1_across_sessions():
    ts = np.asarray([0,1,2_000_000_000,2_000_000_001,5_000_000_000,5_000_000_001])
    sessions = np.asarray([0,0,0,0,1,1], dtype=np.int8)
    pressure = np.asarray([4,-5,6,-7,8,-9], dtype=float)
    raw, expected = fixed.cluster_pressure_events(ts, pressure, sessions, 3)
    assert rescue._cluster_greedy(raw, ts, sessions).tolist() == expected.tolist()


def test_entry_chase_is_signed_from_exact_event_mid_and_buckets_are_fixed():
    t, _ = _tape()
    base = {"date":"2025-03-03", "event_start_time_ns":t, "entry_time_ns":t+2_000_000, "pressure":5.0,
            "entry_direction":"LONG", "entry_price":100.25, "net_R":-1.0,
            "mfe_ticks":1.0,"mae_ticks":6.0,
            "markouts_ticks":{str(h): -1.0 for h in fixed.HORIZONS_MS}}
    short = {**base, "event_start_time_ns":t+1, "pressure":-5.0,
             "entry_direction":"SHORT", "entry_price":98.0}
    result = rescue._phase0_chase([base,short], {("2025-03-03",t,5.0):100.0,
                                                  ("2025-03-03",t+1,-5.0):100.0})
    assert result["buckets"]["0_to_2"]["trade_count"] == 1
    assert result["buckets"]["gt6_to_10"]["trade_count"] == 1
    assert result["trade_count"] == 2


def test_duplicate_provider_timestamp_is_acceptable_only_with_unique_event_mid():
    rows = np.zeros(2, dtype=relative.COMPACT_DTYPE)
    rows["ts"] = 123
    rows["mid"] = 100.0
    rows["ask5"] = 40.0
    trade = {"event_start_time_ns":123, "pressure":5.0,
             "entry_direction":"LONG", "event_depth":40.0}
    key = ("2025-03-03",123,5.0)
    assert rescue._event_mid_lookup(rows,np.asarray([5.0,5.0]),[trade],"2025-03-03")[key] == 100.0
    rows["mid"][1] = 100.25
    with pytest.raises(rescue.RescueError):
        rescue._event_mid_lookup(rows,np.asarray([5.0,5.0]),[trade],"2025-03-03")


def test_prospective_chase_gate_and_tick_rounded_geometry_keep_execution_semantics():
    t,tape = _tape()
    signal = {"timestamp_ns":t,"signal_anchor_ns":t,"session":0,"direction":1,
              "event_start_mid":100.0, "pressure":5.0,"pressure_percentile":.90,
              "depletion":.6,"recovery_ratio":.2,"refill_weakness":.8,
              "mlofi_persistence":.75,"baseline_depth":100,"event_depth":40,
              "confirmation_change_ticks":0}
    accepted = fixed._entry_and_exit(tape,signal,-10**30,stop_ticks=5,target_r=1.25,
                                     max_favorable_move_before_entry_ticks=2)
    assert accepted is not None
    assert accepted["favorable_move_before_entry_ticks"] == pytest.approx(1.0)
    assert accepted["target_ticks"] == 6
    assert accepted["stop_price"] == pytest.approx(99.0)
    assert accepted["target_price"] == pytest.approx(101.75)
    assert fixed._entry_and_exit(tape,signal,-10**30,stop_ticks=5,target_r=1.25,
                                 max_favorable_move_before_entry_ticks=0) is None


def test_prepared_day_attrition_and_no_duplicate_position():
    t,tape = _tape()
    features = np.zeros(2,dtype=rescue.FEATURE_DTYPE)
    features["row_index"] = [5,6]
    features["timestamp_ns"] = [t,t+1]
    features["session"] = 0
    features["direction"] = 1
    features["pressure"] = 5
    features["event_mid"] = 100
    features["baseline_depth"] = 100
    features["event_depth"] = 40
    features["depletion"] = .6
    features["recovery_ratio"] = .2
    features["persistence"] = .75
    features["confirmation_ticks"] = 0
    features["signal_anchor_ns"] = t
    features["chase_ticks"] = 1
    features["has_entry"] = True
    result = rescue.evaluate_prepared_day("2025-03-03",_parameters(),features,np.asarray([5,6]),10,tape)
    assert result["attrition"]["after_depletion"] == 2
    assert result["attrition"]["after_max_chase"] == 2
    assert result["attrition"]["actual_entries"] == 1
    assert result["trades"][0]["exit_reason"] == "TARGET"
    reject = rescue.evaluate_prepared_day("2025-03-03",_parameters(max_favorable_move_before_entry_ticks=2),
                                          features,np.asarray([5,6]),10,tape)
    assert reject["attrition"]["actual_entries"] == 1


def test_neighbor_generation_and_plateau_classification_are_deterministic():
    center = _parameters()
    neighbors = rescue.immediate_neighbors(center)
    assert len(neighbors) == len({rescue._canonical_hash(n) for n in neighbors})
    assert all(sum(n[k] != center[k] for k in center) == 1 for n in neighbors)
    center_row = {"objective":1.0}
    rows = [{"hard_screen":{"pass":True},"metrics":{"net_R":3.0},"objective":.9} for _ in neighbors]
    assert rescue.classify_neighbors(center_row,rows) == "ROBUST_PLATEAU"
    records = [{"number":i,"objective":1.0-i*.01,"parameters":center,
                "hard_screen":{"pass":True},"metrics":{"net_R":5.0,"mean_trade_R":.1,
                "PF":1.5,"max_drawdown_R":2.0,"trades":120,"positive_day_fraction":.6,
                "positive_week_fraction":.7}} for i in range(3)]
    plateau = rescue._plateaus(records)
    assert plateau["passed_spring_hard_screen"] == 3
    assert plateau["clusters"][0]["trial_count"] == 3


def test_vacuum_identity_and_flow_comparison():
    loose = _parameters(depth_depletion_threshold=.20,refill_weakness_threshold=.30,
                        flow_persistence_threshold=.50,max_favorable_move_before_entry_ticks=10)
    assert rescue._vacuum_identity(loose,{"after_price_confirmation":60,"clustered_events":100})["collapsed"]
    assert not rescue._vacuum_identity(_parameters(),{"after_price_confirmation":60,"clustered_events":100})["collapsed"]
    flow = {str(h):{"mean_ticks":.3} for h in fixed.HORIZONS_MS}
    mark = {str(h):{"mean_ticks":-.2} for h in fixed.HORIZONS_MS}
    sample = {"flow_only":flow,"markouts":{"all":mark},"attrition":{"clustered_events":100},
              "performance":{"trade_count":10}}
    assert rescue._flow_comparison(sample,sample)["vacuum_incremental_value"] == "VACUUM_COMPLEXITY_NOT_JUSTIFIED"


def test_optuna_resume_never_exceeds_hard_cap_and_uses_only_spring(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(rescue, "TRIAL_CAP", 2)
    rescue._write_once(tmp_path / "optimization-preregistration.json", rescue.PREREG)
    seen = []
    metrics = {"trades":100,"active_dates":8,"active_weeks":5,"mean_trade_R":.1,
               "median_daily_R":0.0,"median_weekly_R":0.0,"max_drawdown_R":1.0,
               "PF":1.3,"net_R":10.0,"positive_active_week_fraction":.8,
               "maximum_positive_day_share":.2,"maximum_positive_week_share":.3,
               "gross_positive_R":20.0,"long_R":5.0,"short_R":5.0}
    def evaluate(configs, dates, sources, root, catalog, diagnostics=False):
        seen.append(tuple(dates))
        assert tuple(dates) == rescue.fixed.SPRING_DATES
        return [{"metrics":dict(metrics)} for _ in configs]
    monkeypatch.setattr(rescue,"_evaluate_configs",evaluate)
    first = rescue._run_optuna(tmp_path,{}, {})
    second = rescue._run_optuna(tmp_path,{}, {})
    assert len(first) == len(second) == 2
    assert len(seen) == 1
    assert set(seen[0]).isdisjoint(rescue.fixed.OCTOBER_DATES)
    study = rescue._study(tmp_path)
    extra = study.ask(fixed_distributions=rescue._distributions())
    study.tell(extra, 0.0)
    with pytest.raises(rescue.RescueError, match="trial cap exceeded"):
        rescue._run_optuna(tmp_path,{}, {})
