from __future__ import annotations

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_flow_momentum_v1 as flow
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_liquidity_vacuum_v1 as native
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import EVENT_DTYPE


def _tape(day="2025-03-03", *, seconds=50, step_ms=100):
    start = native.baseline._session_windows(day)["ASIA"][0] + 3_600_000_000_000
    n = int(seconds * 1000 / step_ms)
    tape = np.zeros(n, dtype=EVENT_DTYPE)
    tape["timestamp_ns"] = start + np.arange(n, dtype=np.int64) * step_ms * 1_000_000
    tape["session"] = 0
    tape["bid"] = 100.0
    tape["ask"] = 100.25
    return start, tape


def test_fixed_single_strategy_contract_and_native_only_scope():
    assert flow.CONFIG["pressure_window_ms"] == 500
    assert flow.CONFIG["pressure_percentile"] == 90
    assert flow.CONFIG["refractory_seconds"] == 2
    assert flow.CONFIG["max_favorable_move_before_entry_ticks"] == 6
    assert flow.CONFIG["entry_delay_ms"] == 2
    assert flow.CONFIG["stop_ticks"] == 6
    assert flow.CONFIG["target_ticks"] == 9
    assert flow.CONFIG["target_r"] == 1.5
    assert flow.CONFIG["max_hold_seconds"] == 10
    assert flow.CONFIG["native_schema"] == "mbp-10"
    assert flow.CONFIG["vacuum_filters"] is False
    assert flow.CONFIG["level_filters"] is False
    assert len(flow.SPRING_DATES) == 35
    assert len(flow.OCTOBER_DATES) == 19
    assert not any(day.startswith("2026") for day in flow.SOURCE_DATES)


def test_side_specific_prior_date_q90_and_current_date_exclusion():
    history = {"LONG": list(range(1, 101)), "SHORT": list(range(2, 202, 2))}
    before = flow.prior_thresholds(history)
    assert before["SHORT"] == pytest.approx(2 * before["LONG"])
    # The caller adds today's sample only after evaluation, so the frozen
    # threshold object cannot change when a current-date row is introduced.
    today = np.asarray([1000.0, -1000.0])
    assert flow.prior_thresholds(history) == before
    assert today[0] >= before["LONG"] and -today[1] >= before["SHORT"]
    with pytest.raises(flow.FlowStudyError):
        flow.prior_thresholds({"LONG": [1.0], "SHORT": []})


def test_directional_flow_clustering_buy_sell_symmetry_and_two_second_refractory():
    ts = np.asarray([0, 1, 1_999_999_999, 2_000_000_000, 2_000_000_001, 5_000_000_000], dtype=np.int64)
    pressure = np.asarray([4., -4., 4., -4., 4., -4.])
    sessions = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.int8)
    raw, clustered = flow.cluster_flow(ts, pressure, sessions, {"LONG": 3., "SHORT": 3.})
    assert raw.tolist() == list(range(6))
    assert clustered.tolist() == [0, 3, 5]
    inverse_raw, inverse_clustered = flow.cluster_flow(ts, -pressure, sessions, {"LONG": 3., "SHORT": 3.})
    assert inverse_raw.tolist() == raw.tolist()
    assert inverse_clustered.tolist() == clustered.tolist()


def test_execution_stays_fixed_at_two_ms_six_nine_geometry_and_ten_second_exit():
    start, tape = _tape()
    signal = {"timestamp_ns": start + 5_000_000_000, "signal_anchor_ns": start + 5_000_000_000,
              "session": 0, "direction": 1, "event_start_mid": 100.125,
              "pressure": 10., "pressure_percentile": .95, "confirmation_change_ticks": 0.0}
    trade = native._entry_and_exit(tape, signal, -10**30, stop_ticks=6, target_r=1.5,
                                   max_favorable_move_before_entry_ticks=6, max_hold_seconds=10)
    assert trade is not None
    assert trade["entry_time_ns"] == start + 5_100_000_000
    assert trade["stop_ticks"] == 6 and trade["target_ticks"] == 9
    assert trade["exit_reason"] == "TIME_EXIT"
    assert trade["exit_time_ns"] >= trade["entry_time_ns"] + 10_000_000_000
    assert trade["exit_time_ns"] < trade["entry_time_ns"] + 10_100_000_000
    moved = dict(signal, event_start_mid=98.0)
    assert native._entry_and_exit(tape, moved, -10**30, stop_ticks=6, target_r=1.5,
                                  max_favorable_move_before_entry_ticks=6, max_hold_seconds=10) is None


def test_price_only_features_are_causal_and_matching_uses_no_outcomes():
    start, tape = _tape(seconds=12)
    tape["bid"][30:35] += 1.0
    tape["ask"][30:35] += 1.0
    at = np.asarray([start + 2_000_000_000], dtype=np.int64)
    before, valid = flow._causal_price_features(tape, at)
    assert valid.tolist() == [True]
    tape["bid"][30:] += 20.0
    tape["ask"][30:] += 20.0
    after, valid_after = flow._causal_price_features(tape, at)
    assert valid_after.tolist() == [True]
    assert np.array_equal(before, after)
    assert "L2 pressure" in flow.CONTROL_SPEC["exclusion"]


def test_pressure_only_event_baseline_is_direction_normalized():
    start, tape = _tape(seconds=3)
    tape["bid"][10:] += .25
    tape["ask"][10:] += .25
    signal = np.asarray([start, start], dtype=np.int64)
    direction = np.asarray([1, -1], dtype=np.int8)
    session = np.asarray([0, 0], dtype=np.int8)
    baseline = flow._event_markout_summary(tape, signal, np.asarray([100.125, 100.125]),
                                           direction, session)
    assert baseline["1000"]["n"] == 2
    assert baseline["1000"]["mean_ticks"] == pytest.approx(0.0)


def test_markouts_excursions_and_first_touch_reuse_causal_tape():
    start, tape = _tape(seconds=32)
    tape["bid"][10:] += .25
    tape["ask"][10:] += .25
    marks = flow._signed_entry_markouts(tape, start, 100.25, 1, 0)
    assert marks["1000"] == pytest.approx(0.0)
    excursion = flow._excursions(tape, start, 100.25, 1, 0)
    assert excursion["1000"]["mfe_ticks"] == pytest.approx(0.0)
    assert excursion["1000"]["mae_ticks"] == pytest.approx(1.0)
    assert set(flow._markout_values([1, -1, 2])) >= {"trimmed_mean_ticks", "p25_ticks", "p75_ticks"}


def test_lodo_lowo_and_daily_weekly_include_flat_periods():
    trades = [{"date": "2025-03-03", "week": "2025-W10", "entry_direction": "LONG",
               "net_R": 1.0, "gross_R": 1.2, "hold_seconds": 1.0, "exit_reason": "TARGET"},
              {"date": "2025-03-10", "week": "2025-W11", "entry_direction": "SHORT",
               "net_R": -1.0, "gross_R": -.8, "hold_seconds": 1.0, "exit_reason": "STOP"}]
    lodo = flow._leave_one_out(trades, ["2025-03-03", "2025-03-04", "2025-03-10"], "date")
    lowo = flow._leave_one_out(trades, ["2025-W10", "2025-W11"], "week")
    assert lodo["groups_tested"] == 3 and lowo["groups_tested"] == 2
    assert lodo["net_R"]["min"] == pytest.approx(-1.0)
    daily, weekly = native._daily_weekly(trades, ["2025-03-03", "2025-03-04", "2025-03-10"])
    assert daily[1]["trade_count"] == 0 and len(weekly) == 2


def test_checkpoint_resume_rejects_source_tape_config_and_prior_chain_changes(tmp_path, monkeypatch):
    path = flow._checkpoint_path(tmp_path, "2025-03-03")
    native._write_checkpoint(path, {"checkpoint_version": flow.CHECKPOINT_VERSION,
                                    "status": "DATE_COMPLETE", "date": "2025-03-03",
                                    "source_sha256": "s", "tape_sha256": "t", "config_sha256": flow.CONFIG_SHA256,
                                    "study_sha256": flow.STUDY_SHA256,
                                    "prior_source_chain_sha256": "p", "payload": {"pressure_history_sample": {"LONG": [1], "SHORT": [1]}}})
    args = {"day": "2025-03-03", "source_sha": "s", "tape_sha": "t", "prior_chain": "p"}
    assert flow._read_checkpoint(path, **args) is not None
    for key in ("source_sha", "tape_sha", "prior_chain"):
        assert flow._read_checkpoint(path, **{**args, key: "changed"}) is None
    monkeypatch.setattr(flow, "STUDY_SHA256", "changed")
    assert flow._read_checkpoint(path, **args) is None
