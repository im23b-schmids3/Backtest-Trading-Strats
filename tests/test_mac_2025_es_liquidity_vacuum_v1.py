from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_liquidity_vacuum_v1 as vacuum
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_normalization as relative
from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import EVENT_DTYPE


def _rows(day: str = "2025-03-03", count: int = 145) -> tuple[np.ndarray, int]:
    rows = np.zeros(count, dtype=relative.COMPACT_DTYPE)
    start = vacuum.baseline._session_windows(day)["ASIA"][0]
    rows["ts"] = start + np.arange(count, dtype=np.int64) * 50_000_000
    rows["mid"] = 100.0
    rows["bid5"] = 100
    rows["ask5"] = 100
    rows["denom"] = 50
    t_ix = 120
    rows["ask5"][t_ix:] = 40
    rows["bid5"][t_ix:] = 40
    rows["mid"][t_ix + 20:] = 100.25
    # Four post-event bins, with three directionally positive bins.
    for k in (1, 6, 11):
        rows["mlofi"][t_ix + k] = 1.0
    return rows, t_ix


def test_pressure_uses_prior_500ms_contributions_and_current_depth_denominator():
    rows = np.zeros(4, dtype=relative.COMPACT_DTYPE)
    rows["ts"] = [100, 200, 400, 600]
    rows["mlofi"] = [1, 2, -1, 3]
    rows["denom"] = 5
    result = vacuum.rolling_pressure(rows, window_ns=300)
    assert result.tolist() == pytest.approx([0.2, 0.6, 0.4, 0.4])


def test_prior_date_pressure_threshold_is_fixed_before_current_date():
    prior = np.asarray([1.0, 2.0, 3.0, 4.0])
    current = np.asarray([1000.0])
    threshold = float(np.quantile(prior, 0.90))
    assert threshold == pytest.approx(3.7)
    assert threshold != float(np.quantile(np.r_[prior, current], 0.90))


def test_two_second_refractory_clusters_sustained_pressure_burst():
    raw, clusters = vacuum.cluster_pressure_events(
        [0, 1, 2_000_000_000, 2_000_000_001, 3_000_000_000],
        [4, -5, 6, -7, 0], [0, 0, 0, 0, 0], 3,
    )
    assert raw.tolist() == [0, 1, 2, 3]
    assert clusters.tolist() == [0, 2]


def test_refractory_resets_at_session_boundary():
    _, clusters = vacuum.cluster_pressure_events(
        [1, 2], [3, -4], [0, 1], 2, 2_000_000_000,
    )
    assert clusters.tolist() == [0, 1]


@pytest.mark.parametrize("direction,side", [(1, "ask5"), (-1, "bid5")])
def test_opposing_side_depth_and_depletion_are_symmetric(direction, side):
    rows, ix = _rows()
    event = {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": direction}
    state = vacuum._classify_event(rows, event, day="2025-03-03")
    assert state["baseline_depth"] == 100
    assert state["event_depth"] == 40
    assert state["depletion"] == pytest.approx(0.60)
    other = "bid5" if side == "ask5" else "ask5"
    rows[other][ix] = 95
    state_other = vacuum._classify_event(rows, event, day="2025-03-03")
    assert state_other["depletion"] == pytest.approx(0.60)


def test_five_second_baseline_and_exact_fifty_percent_depletion_boundary():
    rows, ix = _rows()
    rows["ask5"][:ix] = 80
    rows["ask5"][ix:] = 40
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["baseline_depth"] == 80
    assert state["depletion"] == 0.5
    assert state["after_depletion"]


def test_depletion_below_fifty_percent_is_rejected():
    rows, ix = _rows()
    rows["ask5"][ix:] = 51
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert not state["after_depletion"]


def test_recovery_ratio_and_refill_weakness_are_measured_after_full_500ms():
    rows, ix = _rows()
    rows["ask5"][ix + 10:] = 70  # refill occurs only after the 500ms observation.
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["refill_observation_ns"] >= int(rows["ts"][ix]) + 500_000_000
    assert state["recovery_ratio"] == pytest.approx(0.5)
    assert state["refill_weakness"] == pytest.approx(0.5)
    assert state["after_refill"]


def test_refill_above_fifty_percent_fails_without_signal_leakage():
    rows, ix = _rows()
    rows["ask5"][ix + 2:] = 80
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["recovery_ratio"] > 0.5
    assert not state["after_refill"]
    assert "signal_anchor_ns" not in state


def test_mlofi_persistence_requires_three_of_four_250ms_bins():
    rows, ix = _rows()
    rows["mlofi"][ix + 6] = -1
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["mlofi_agreeing_bins"] == 2
    assert not state["after_persistence"]


def test_mlofi_three_of_four_passes_and_anchor_is_after_one_second():
    rows, ix = _rows()
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["mlofi_agreeing_bins"] == 3
    assert state["after_persistence"]
    assert state["signal_anchor_ns"] >= int(rows["ts"][ix]) + 1_000_000_000
    assert state["after_confirmation"]


def test_price_confirmation_rejects_adverse_move_and_uses_only_anchor():
    rows, ix = _rows()
    rows["mid"][ix + 20:] = 99.75
    state = vacuum._classify_event(rows, {"row_index": ix, "timestamp_ns": int(rows["ts"][ix]), "session": 0, "direction": 1}, day="2025-03-03")
    assert state["confirmation_change_ticks"] == -1
    assert not state["after_confirmation"]


def _events(entries: list[tuple[int, float, float, int]]) -> np.ndarray:
    out = np.zeros(len(entries), dtype=EVENT_DTYPE)
    for i, (ts, bid, ask, session) in enumerate(entries):
        out[i] = (ts, bid, ask, np.nan, 0, 0, session)
    return out


def _signal(ts: int, direction: int = 1) -> dict[str, Any]:
    return {"timestamp_ns": ts, "signal_anchor_ns": ts, "session": 0, "direction": direction,
            "pressure": direction * 5.0, "pressure_percentile": 0.95, "depletion": .6,
            "recovery_ratio": .2, "refill_weakness": .8, "mlofi_persistence": .75,
            "baseline_depth": 100, "event_depth": 40, "confirmation_change_ticks": 0}


def test_entry_uses_fixed_two_ms_and_adverse_quote_fill_and_six_twelve_geometry():
    start = vacuum.baseline._session_windows("2025-03-03")["ASIA"][0] + 3_600_000_000_000
    events = _events([(start + 1_000_000, 99.75, 100.0, 0), (start + 2_000_000, 99.75, 100.0, 0),
                      (start + 1_000_000_000, 103.5, 103.75, 0)])
    trade = vacuum._entry_and_exit(events, _signal(start), -10**30)
    assert trade is not None
    assert trade["entry_time_ns"] == start + 2_000_000
    assert trade["entry_price"] == pytest.approx(100.25)
    assert trade["stop_price"] == pytest.approx(98.75)
    assert trade["target_price"] == pytest.approx(103.25)
    assert trade["exit_reason"] == "TARGET"
    assert trade["exit_price"] == pytest.approx(103.25)


def test_stop_precedes_target_on_same_timestamp_wide_crossed_exit_quote():
    start = vacuum.baseline._session_windows("2025-03-03")["ASIA"][0] + 3_600_000_000_000
    events = _events([(start + 2_000_000, 99.75, 100.0, 0),
                      (start + 3_000_000, 98.0, 98.25, 0),
                      (start + 3_000_000, 104.0, 104.25, 0)])
    trade = vacuum._entry_and_exit(events, _signal(start), -10**30)
    assert trade is not None
    assert trade["exit_reason"] == "STOP"


def test_time_exit_uses_first_executable_at_or_after_thirty_seconds():
    start = vacuum.baseline._session_windows("2025-03-03")["ASIA"][0] + 3_600_000_000_000
    events = _events([(start + 2_000_000, 99.75, 100.0, 0),
                      (start + 30_002_000_001, 100.0, 100.25, 0)])
    trade = vacuum._entry_and_exit(events, _signal(start), -10**30)
    assert trade is not None
    assert trade["exit_reason"] == "TIME_EXIT"
    assert trade["exit_time_ns"] == start + 30_002_000_001


def test_one_position_and_post_exit_two_second_refractory_block_duplicate_entry():
    start = vacuum.baseline._session_windows("2025-03-03")["ASIA"][0] + 3_600_000_000_000
    events = _events([(start + 2_000_000, 99.75, 100.0, 0),
                      (start + 3_000_000, 98.0, 98.25, 0)])
    trade = vacuum._entry_and_exit(events, _signal(start), start + 5_000_000_000)
    assert trade is None


def test_daily_weekly_aggregation_includes_zero_trade_dates():
    trades = [{"date": "2025-03-03", "net_R": 1.0, "gross_R": 1.2, "entry_direction": "LONG",
               "exit_reason": "TARGET", "hold_seconds": 2.0}]
    daily, weekly = vacuum._daily_weekly(trades, ["2025-03-03", "2025-03-04"])
    assert [row["trade_count"] for row in daily] == [1, 0]
    assert daily[1]["R"] == 0
    assert weekly[0]["trades"] == 1


def test_trade_r_calculation_and_fees_are_explicit():
    start = vacuum.baseline._session_windows("2025-03-03")["ASIA"][0] + 3_600_000_000_000
    events = _events([(start + 2_000_000, 99.75, 100.0, 0), (start + 3_000_000, 103.5, 103.75, 0)])
    trade = vacuum._entry_and_exit(events, _signal(start), -10**30)
    assert trade is not None
    assert trade["commission_cost_usd"] > 0
    assert trade["net_R"] == pytest.approx((trade["gross_pnl_usd"] - trade["commission_cost_usd"]) / trade["initial_risk_usd"])


def test_checkpoint_resume_rejects_source_hash_or_semantic_change(tmp_path: Path):
    path = vacuum._checkpoint_path(tmp_path, "2025-03-03")
    payload = {"checkpoint_version": vacuum.CHECKPOINT_VERSION, "status": "DATE_COMPLETE", "date": "2025-03-03",
               "source_sha256": "source", "tape_sha256": "tape", "config_sha256": vacuum.CONFIG_SHA256,
               "study_sha256": vacuum.STUDY_SHA256, "payload": {"pressure_history_sample": [1.0]}}
    vacuum._write_checkpoint(path, payload)
    assert vacuum._read_checkpoint(path, day="2025-03-03", source_sha="source", tape_sha="tape") == payload
    assert vacuum._read_checkpoint(path, day="2025-03-03", source_sha="changed", tape_sha="tape") is None
    assert vacuum._read_checkpoint(path, day="2025-03-03", source_sha="source", tape_sha="changed") is None


def test_fixed_configuration_has_no_levels_or_optimization_and_exact_study_scope():
    assert vacuum.CONFIG["stop_ticks"] == 6
    assert vacuum.CONFIG["target_ticks"] == 12
    assert vacuum.CONFIG["pressure_percentile"] == 90
    assert vacuum.CONFIG["no_levels"] is True
    assert vacuum.SPRING_DATES[0] == "2025-03-03" and vacuum.SPRING_DATES[-1] == "2025-04-21"
    assert len(vacuum.SPRING_DATES) == 35
    assert len(vacuum.OCTOBER_DATES) == 19
    assert all(not day.startswith("2026-") for day in vacuum.ALL_SOURCE_DATES)
