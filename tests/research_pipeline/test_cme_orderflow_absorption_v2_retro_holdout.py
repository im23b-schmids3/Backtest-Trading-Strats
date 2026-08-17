from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import research_pipeline.cme_orderflow_absorption_v1.v2_retro_holdout_runner as retro
import research_pipeline.cme_orderflow_absorption_v1.v2_aug_vs_retro_diagnostic as comparison
import research_pipeline.cme_orderflow_absorption_v1.v2_aug_vs_retro_path_replay as path_replay
import research_pipeline.cme_orderflow_absorption_v1.v2_aug_vs_retro_entry_geometry as geometry
import research_pipeline.cme_orderflow_absorption_v1.v2_favorable_tick_timing as timing
import research_pipeline.cme_orderflow_absorption_v1.v3_tick_trigger_target_matrix as matrix
import research_pipeline.cme_orderflow_absorption_v1.v3_long_short_regime_diagnostic as regime
import research_pipeline.cme_orderflow_absorption_v1.v3_compact_group_comparison as compact
import research_pipeline.cme_orderflow_absorption_v1.v3_regime_bucket_analysis as buckets
import quote_fresh_es_oos_cost as fresh_quote
import acquire_may_2026_es_mes_cost_proxy as may_acquisition
from research_pipeline.cme_orderflow_absorption_v1.analysis import Diagnostics


def row(interaction_id: str = "i1", direction: str = "BUYER_ABSORPTION") -> dict:
    return {"interaction_id": interaction_id, "direction": direction, "end_price": 7_500_000_000_000,
            "zone_low": 7_499_000_000_000, "zone_high": 7_501_000_000_000, "level": "PRIOR_RTH_POC",
            "absorption_score": .9, "replenishment_score": .9}


def test_explicit_prior_session_map_preserves_july_sixth_holiday_mapping():
    assert retro.PRIOR_RTH["2026-07-06"] == "2026-07-02"
    assert tuple(retro.PRIOR_RTH) == retro.TEST_DAYS


def test_five_tick_stop_and_two_point_five_r_target_are_frozen():
    long = retro.initial_prices("BUYER_ABSORPTION", 7500.0, 7500.25, 7499.0, 7501.0)
    short = retro.initial_prices("SELLER_ABSORPTION", 7500.0, 7500.25, 7499.0, 7501.0)
    assert long["stop"] == 7497.75
    assert long["target"] == long["entry"] + 2.5 * (long["entry"] - long["stop"])
    assert short["stop"] == 7502.25
    assert short["target"] == short["entry"] - 2.5 * (short["stop"] - short["entry"])


def test_es_first_sizing_and_apex_cap_never_increase_size(monkeypatch):
    prices, es = retro.choose_es_first("BUYER_ABSORPTION", 100.0, 100.25, 99.0, 101.0)
    assert es["instrument"] == "ES"
    monkeypatch.setattr(retro, "RISK_BUDGET", 100_000.0)
    capped = retro.size_for_instrument(prices, "ES")
    assert capped["risk_based_contracts"] > retro.ES_CAP
    assert capped["contracts"] == retro.ES_CAP


def test_native_mes_fallback_is_extracted_from_level_zero_and_capped(monkeypatch):
    @dataclass
    class Level:
        bid_px: int = 100_000_000_000
        ask_px: int = 100_250_000_000
        bid_sz: int = 1
        ask_sz: int = 1
    @dataclass
    class Record:
        levels: tuple = (Level(),)
    assert retro._valid_mes(Record()) == (100.0, 100.25)
    prices = retro.initial_prices("BUYER_ABSORPTION", 100.0, 100.25, 99.0, 101.0)
    monkeypatch.setattr(retro, "RISK_BUDGET", 100_000.0)
    sizing = retro.size_for_instrument(prices, "MES")
    assert sizing["contracts"] == retro.MES_CAP


def test_raw_interaction_zone_is_converted_to_points_before_long_and_short_stop_sizing():
    raw = {"zone_low": 7_748_500_000_000, "zone_high": 7_749_000_000_000}
    zone_low, zone_high = retro.interaction_zone_points(raw)
    assert zone_low == 7748.50
    assert zone_high == 7749.00

    long = retro.initial_prices("BUYER_ABSORPTION", 7750.0, 7750.25, zone_low, zone_high)
    short = retro.initial_prices("SELLER_ABSORPTION", 7750.0, 7750.25, zone_low, zone_high)
    assert long["stop"] == 7747.25
    assert short["stop"] == 7750.25
    assert retro.size_for_instrument(long, "ES")["contracts"] >= 1
    assert retro.size_for_instrument(short, "ES")["contracts"] >= 1


def test_first_es_execution_at_or_after_fifteen_seconds_and_two_ms_entry_latency():
    state = retro.DayState("2026-06-23", "2026-06-22", Diagnostics())
    pending = retro.PendingSignal(row(), confirmation_due_ns=15_000_000_000)
    state.pending["i1"] = pending
    retro._confirm_and_enter_es(state, 14_999_999_999, 7500.25, (7500.0, 7500.25))
    assert pending.state == "AWAITING_CONFIRMATION"
    retro._confirm_and_enter_es(state, 15_000_000_000, 7500.25, (7500.0, 7500.25))
    assert pending.entry_ready_ns == 15_002_000_000 and state.position is None
    retro._confirm_and_enter_es(state, 15_001_999_999, None, (7500.0, 7500.25))
    assert state.position is None
    retro._confirm_and_enter_es(state, 15_002_000_000, None, (7500.0, 7500.25))
    assert state.position is not None and state.position.instrument == "ES"


def test_confirmed_raw_zone_uses_es_then_native_mes_fallback_when_es_risk_does_not_fit():
    state = retro.DayState("2026-06-23", "2026-06-22", Diagnostics())
    pending = retro.PendingSignal(
        row("mes", "BUYER_ABSORPTION") | {"zone_low": 7_494_000_000_000},
        confirmation_due_ns=0,
        state="AWAITING_MES_ENTRY",
        mes_decision_ns=0,
    )
    state.pending["mes"] = pending
    retro._enter_mes(state, 1, (7500.0, 7500.25))
    assert state.position is not None
    assert state.position.instrument == "MES"
    assert state.position.contracts >= 1


def test_confirmation_failure_does_not_early_invalidate_before_horizon():
    state = retro.DayState("2026-06-23", "2026-06-22", Diagnostics())
    pending = retro.PendingSignal(row(), confirmation_due_ns=15)
    state.pending["i1"] = pending
    retro._confirm_and_enter_es(state, 14, 7400.0, (7500.0, 7500.25))
    assert pending.state == "AWAITING_CONFIRMATION"
    retro._confirm_and_enter_es(state, 15, 7400.0, (7500.0, 7500.25))
    assert pending.state == "CONFIRMATION_FAILED"


def test_tail_required_is_fail_closed_for_pending_or_open_position():
    state = retro.DayState("2026-06-23", "2026-06-22", Diagnostics())
    state.pending["i1"] = retro.PendingSignal(row(), confirmation_due_ns=1)
    assert retro._tail(state) == ["i1"]
    assert ("TAIL_DATA_REQUIRED" if retro._tail(state) else "RECONCILED_RETRO_ROBUSTNESS_REPLAY") == "TAIL_DATA_REQUIRED"


def test_no_final_performance_status_when_tail_unresolved():
    tails = {"2026-06-23": ["i1"]}
    status = "TAIL_DATA_REQUIRED" if tails else "RECONCILED_RETRO_ROBUSTNESS_REPLAY"
    assert status != "RECONCILED_RETRO_ROBUSTNESS_REPLAY"


def test_incremental_completion_path_never_resummarizes_historical_interactions(monkeypatch):
    class SummaryProbe:
        def __init__(self, interaction_id: str):
            self.interaction_id = interaction_id
            self.calls = 0

        def summary(self):
            self.calls += 1
            return {"interaction_id": self.interaction_id, "end_ns": 1, "level": "PRIOR_RTH_POC",
                    "level_price": 7_500_000_000_000, "end_price": 7_500_000_000_000}

    state = retro.DayState("2026-06-23", "2026-06-22", Diagnostics())
    first, second, third = SummaryProbe("first"), SummaryProbe("second"), SummaryProbe("third")
    state.diagnostics.completed.extend((first, second))
    monkeypatch.setattr(retro, "_score", lambda rows, calibration: [
        {**item, "absorption_score": 0.0, "replenishment_score": 0.0,
         "direction": "BUYER_ABSORPTION", "interaction_end": item["end_ns"],
         "zone_low": 7500.0, "zone_high": 7500.0}
        for item in rows
    ])
    monkeypatch.setattr(retro, "plus_only", lambda contract, selection: False)

    contract = {"frozen_selection": {"absorption_p95": 1.0, "replenishment_p95": 1.0}}
    retro._new_completed(state, {}, contract)
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)
    for _ in range(4):
        retro._new_completed(state, {}, contract)
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)

    state.diagnostics.completed.append(third)
    retro._new_completed(state, {}, contract)
    assert (first.calls, second.calls, third.calls) == (1, 1, 1)


def test_artifact_only_august_retro_comparison_preserves_execution_limits_and_marks_excursions_unavailable(tmp_path):
    august, retro_root, output = tmp_path / "august", tmp_path / "retro", tmp_path / "diagnostic"
    august.mkdir(); retro_root.mkdir()
    august_summary = {"interpretation": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE", "v1_plus_input": 21, "confirmations_passed": 11, "confirmations_failed": 10, "status": "READY"}
    retro_summary = {"interpretation": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST", "raw_interactions": 3403, "plus_count": 51, "confirmations_passed": 21, "confirmations_failed": 30, "status": "TAIL_DATA_REQUIRED"}
    (august / "summary.json").write_text(json.dumps(august_summary), encoding="utf-8")
    (retro_root / "summary.json").write_text(json.dumps(retro_summary), encoding="utf-8")
    augmented = {
        "interaction_id": "aug-1", "date": "2026-08-03", "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC", "absorption_score": "0.9", "replenishment_score": "0.8", "interaction_end": "1000000000", "confirmation_timestamp": "16000000000", "confirmation_favorable_ticks": "2", "entry_timestamp": "16002000000", "entry": "7750.5", "stop": "7747.25", "target": "7760.25", "exit_timestamp": "20000000000", "exit_reason": "TARGET", "r_multiple": "2.9", "instrument": "ES", "contracts": "1", "one_contract_initial_risk_usd": "181", "target_multiple": "3.0",
    }
    retro_trade = {"interaction_id": "retro-1", "date": "2026-06-23", "direction": "SELLER_ABSORPTION", "level": "PRIOR_RTH_HIGH", "absorption_score": "0.91", "replenishment_score": "0.82", "entry": "7500", "stop": "7503", "target": "7492.5", "entry_timestamp": "16000000000", "exit_timestamp": "19000000000", "exit_reason": "STOP", "r_multiple": "-1", "instrument": "MES", "contracts": "5"}
    fields = list(augmented)
    with (august / "trades_3_0R.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow(augmented)
    with (retro_root / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retro_trade)); writer.writeheader(); writer.writerow(retro_trade)
    plus = {**retro_trade, "interaction_end": "12000000000", "zone_low": "7499000000000", "zone_high": "7501000000000"}
    with (retro_root / "plus-signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plus)); writer.writeheader(); writer.writerow(plus)

    result = comparison.build_comparison(august_root=august, retro_root=retro_root, output_dir=output)
    rows = list(csv.DictReader((output / "trade-comparison.csv").open(encoding="utf-8")))
    assert result["strategy_semantics_changed"] is False
    assert result["periods"]["august"]["plus_rate"] == pytest.approx(21 / 1430)
    assert len(rows) == 2 and {row["excursion_status"] for row in rows} == {"NOT_COMPUTED_NO_CAUSAL_PRICE_PATH_REPLAY"}
    assert rows[0]["confirmation_favorable_ticks"] == "2.0"
    assert rows[1]["confirmation_timestamp"] == ""
    assert (output / "summary.json").is_file() and (output / "period-comparison.json").is_file() and (output / "diagnostic-report.md").is_file()


def test_long_path_mfe_mae_milestones_and_exit_boundary_are_causal():
    state = path_replay.PathState({"direction": "BUYER_ABSORPTION", "entry_timestamp": 10, "exit_timestamp": 30, "entry": 100.0, "stop": 98.0})
    state.observe(9, 200.0, 200.25)
    state.observe(11, 103.0, 103.25)
    state.observe(20, 97.0, 97.25)
    state.observe(31, 500.0, 500.25)
    result = state.materialize()
    assert result["mfe_ticks"] == 11.0 and result["mae_ticks"] == 13.0
    assert result["reached_0_25r"] and result["reached_0_5r"] and result["reached_1_0r"]
    assert result["seconds_entry_to_mfe"] == pytest.approx(1e-9)


def test_short_path_uses_executable_ask_and_records_milestones():
    state = path_replay.PathState({"direction": "SELLER_ABSORPTION", "entry_timestamp": 10, "exit_timestamp": 30, "entry": 100.0, "stop": 102.0})
    state.observe(12, 96.75, 97.0)
    state.observe(20, 103.0, 103.25)
    result = state.materialize()
    assert result["mfe_ticks"] == 11.0 and result["mae_ticks"] == 14.0
    assert result["maximum_favorable_r"] == pytest.approx(11 / 9)
    assert result["maximum_adverse_r"] == pytest.approx(14 / 9)


def test_august_two_point_five_r_target_construction_and_tail_non_inference():
    assert path_replay.target_2p5("BUYER_ABSORPTION", 100.0, 98.0) == 105.0
    assert path_replay.target_2p5("SELLER_ABSORPTION", 100.0, 102.0) == 95.0
    assert path_replay.UNRESOLVED_TAIL_ID == "2026-07-13:PRIOR_RTH_VAL:7596000000000:0097"


def test_long_entry_geometry_uses_direction_aware_displacement_and_point_prices():
    row = {"direction": "BUYER_ABSORPTION", "interaction_end_price": 7748.50, "confirmation_price": 7749.00,
           "entry": 7750.25, "zone_low": 7748.50, "zone_high": 7749.25, "stop": 7747.25, "target": 7757.75,
           "mfe_ticks": 16.0, "mae_ticks": 4.0}
    result = geometry.entry_geometry(row)
    assert result["entry_displacement_points_from_interaction_end"] == 1.75
    assert result["entry_displacement_ticks_from_interaction_end"] == 7.0
    assert result["entry_displacement_ticks_from_confirmation_price"] == 5.0
    assert result["stop_distance_ticks_from_entry"] == 12.0
    assert result["target_distance_ticks_from_entry"] == 30.0
    assert result["target_r"] == 2.5


def test_short_entry_geometry_reverses_displacement_sign_without_changing_geometry():
    row = {"direction": "SELLER_ABSORPTION", "interaction_end_price": 7750.00, "confirmation_price": 7749.50,
           "entry": 7748.25, "zone_low": 7748.0, "zone_high": 7750.0, "stop": 7751.25, "target": 7740.75,
           "mfe_ticks": 8.0, "mae_ticks": 3.0}
    result = geometry.entry_geometry(row)
    assert result["entry_displacement_ticks_from_interaction_end"] == 7.0
    assert result["entry_displacement_ticks_from_confirmation_price"] == 5.0
    assert result["stop_distance_points_from_entry"] == 3.0
    assert result["target_distance_points_from_entry"] == 7.5
    assert result["target_r"] == 2.5


def test_long_first_favorable_tick_reaches_are_first_and_limited_to_fifteen_seconds():
    end = 100_000_000_000
    state = timing.TimingState({"interaction_id": "long", "period": "TEST", "date": "2026-01-01", "direction": "BUYER_ABSORPTION", "level": "L", "interaction_end": end, "end_price": 7_500_000_000_000}, "TEST")
    state.observe_execution(end + 1, 7_500_250_000_000)
    state.observe_execution(end + 2, 7_500_500_000_000)
    state.observe_execution(end + 3, 7_500_750_000_000)
    state.observe_execution(end + timing.CONFIRMATION_NS + 1, 7_503_000_000_000)
    assert state.first_reach[1] == (end + 1, 7500.25)
    assert state.first_reach[2] == (end + 2, 7500.5)
    assert state.first_reach[3] == (end + 3, 7500.75)
    assert state.first_reach[4] is None and state.confirmation_timestamp is None


def test_short_first_favorable_tick_timing_and_unreached_thresholds():
    end = 100_000_000_000
    state = timing.TimingState({"interaction_id": "short", "period": "TEST", "date": "2026-01-01", "direction": "SELLER_ABSORPTION", "level": "L", "interaction_end": end, "end_price": 7_500_000_000_000}, "TEST")
    state.observe_execution(end + 1, 7_499_750_000_000)
    state.observe_execution(end + 2, 7_499_500_000_000)
    state.observe_execution(end + 3, 7_499_250_000_000)
    state.observe_after_horizon_execution(end + timing.CONFIRMATION_NS, 7_499_500_000_000)
    assert state.first_reach[1] == (end + 1, 7499.75)
    assert state.first_reach[2] == (end + 2, 7499.5)
    assert state.first_reach[3] == (end + 3, 7499.25)
    assert state.first_reach[4] is None
    assert state.confirmation_favorable_ticks == 2.0


def test_mes_completed_trade_uses_es_confirmation_recovery(monkeypatch, tmp_path):
    end = 100_000_000_000
    row = {"interaction_id": "mes", "date": "2026-06-23", "direction": "BUYER_ABSORPTION", "interaction_end": end,
           "interaction_end_price": 7500.0, "end_price": 7_500_000_000_000, "entry_timestamp": 200, "exit_timestamp": 300,
           "entry": 7501.0, "stop": 7499.0, "target": 7506.0, "instrument": "MES"}
    def es_scan(path, states, *, august_snapshot_contract, confirmation_states=None):
        assert states == [] and confirmation_states is not None and confirmation_states[0].row["instrument"] == "MES"
        confirmation_states[0].observe_confirmation_execution(end + path_replay.CONFIRMATION_NS, 7_500_250_000_000)
    monkeypatch.setattr(path_replay, "scan_es_mbo_path", es_scan)
    monkeypatch.setattr(path_replay, "scan_mes_mbp1_path", lambda *args: None)
    monkeypatch.setattr(path_replay, "_retro_paths", lambda root, date: (tmp_path / "es", tmp_path / "mes"))
    result = path_replay.replay_recorded_trade_paths(august_trades=[], retro_trades=[row])
    assert result[0]["confirmation_favorable_ticks"] == 1.0


def matrix_row(identifier="matrix", direction="BUYER_ABSORPTION"):
    return {"interaction_id": identifier, "date": "2026-08-03", "direction": direction,
            "level": "CURRENT_RTH_HIGH_SWEEP", "interaction_end": 100_000_000_000,
            "end_price": 7_500_000_000_000, "zone_low": 7_499_000_000_000,
            "zone_high": 7_501_000_000_000}


def test_matrix_is_exactly_the_six_predeclared_tick_target_cells():
    assert [(item.trigger_ticks, item.target_r) for item in matrix.SEALED_MATRIX] == [
        (1, 2.5), (1, 3.0), (2, 2.5), (2, 3.0), (3, 2.5), (3, 3.0),
    ]


def test_matrix_long_trigger_is_first_qualifying_execution_and_keeps_two_ms_latency():
    cell = matrix.Cell(matrix.MatrixSpec(2, 2.5), "TEST", "TEST"); cell.add_signals([matrix_row()])
    pending = cell.pending["matrix"]
    cell.observe_execution(100_000_000_001, 7_500_250_000_000)
    assert pending.state == "AWAITING_TRIGGER"
    cell.observe_execution(100_000_000_002, 7_500_500_000_000)
    assert pending.state == "AWAITING_ES_ENTRY"
    assert pending.trigger_timestamp == 100_000_000_002
    assert pending.entry_ready_ns == 100_000_000_002 + matrix.ENTRY_LATENCY_NS


def test_matrix_long_three_tick_and_short_symmetric_trigger_behavior():
    long = matrix.Cell(matrix.MatrixSpec(3, 3.0), "TEST", "TEST"); long.add_signals([matrix_row("long")])
    short = matrix.Cell(matrix.MatrixSpec(3, 3.0), "TEST", "TEST"); short.add_signals([matrix_row("short", "SELLER_ABSORPTION")])
    long.observe_execution(100_000_000_001, 7_500_500_000_000)
    long.observe_execution(100_000_000_002, 7_500_750_000_000)
    short.observe_execution(100_000_000_001, 7_499_500_000_000)
    short.observe_execution(100_000_000_002, 7_499_250_000_000)
    assert long.pending["long"].trigger_timestamp == 100_000_000_002
    assert short.pending["short"].trigger_timestamp == 100_000_000_002


def test_matrix_trigger_after_window_fails_without_early_adverse_invalidation():
    cell = matrix.Cell(matrix.MatrixSpec(1, 2.5), "TEST", "TEST"); cell.add_signals([matrix_row()])
    pending = cell.pending["matrix"]
    cell.observe_execution(100_000_000_001, 7_499_000_000_000)
    assert pending.state == "AWAITING_TRIGGER"
    cell.observe_execution(pending.deadline + 1, 7_501_000_000_000)
    assert pending.state == "CONFIRMATION_FAILED" and cell.confirmations_failed == 1


def test_matrix_target_stop_and_cells_keep_independent_position_state(monkeypatch):
    first = matrix.Cell(matrix.MatrixSpec(1, 2.5), "TEST", "TEST")
    second = matrix.Cell(matrix.MatrixSpec(1, 3.0), "TEST", "TEST")
    first.add_signals([matrix_row("first")]); second.add_signals([matrix_row("second")])
    for cell in (first, second):
        cell.observe_execution(100_000_000_001, 7_500_250_000_000)
    monkeypatch.setattr(matrix, "size_trade_with_mes_fallback", lambda p: {"instrument": "ES", "contracts": 1, "one_contract_initial_risk_usd": 100.0})
    ts = 100_000_000_001 + matrix.ENTRY_LATENCY_NS
    first.try_august_entry(ts, (7500.0, 7500.25)); second.try_august_entry(ts, (7500.0, 7500.25))
    assert first.position is not None and second.position is not None
    assert first.position.stop == 7497.75 and first.position.target == 7507.375
    assert second.position.target == 7508.75


def test_matrix_es_first_mes_fallback_and_fixed_point_zone_conversion(monkeypatch):
    cell = matrix.Cell(matrix.MatrixSpec(1, 2.5), "TEST", "TEST"); cell.add_signals([matrix_row()])
    pending = cell.pending["matrix"]; cell.observe_execution(100_000_000_001, 7_500_250_000_000)
    captured = {}
    def size(p):
        captured.update(p); return {"instrument": "MES", "contracts": 4, "one_contract_initial_risk_usd": 25.0}
    monkeypatch.setattr(matrix, "size_trade_with_mes_fallback", size)
    cell.try_august_entry(pending.entry_ready_ns, (7500.0, 7500.25))
    assert cell.position is not None and cell.position.instrument == "MES"
    assert captured["stop"] == 7497.75


def test_matrix_tail_is_not_inferred_at_source_end():
    cell = matrix.Cell(matrix.MatrixSpec(1, 2.5), "TEST", "TEST"); cell.add_signals([matrix_row("tail")])
    assert cell.tail() == ["tail"]


def test_causal_context_trends_ranges_activity_and_vwap_exclude_future_execution():
    context = regime.ContextAccumulator()
    base = 1_000_000_000_000
    context.observe_execution(base, 7_500_000_000_000, 2)
    context.observe_execution(base + 300_000_000_000, 7_501_000_000_000, 1)
    before_future = context.snapshot(timestamp=base + 300_000_000_000, end_price_raw=7_501_000_000_000, direction="BUYER_ABSORPTION")
    context.observe_execution(base + 301_000_000_000, 7_600_000_000_000, 100)
    frozen = before_future
    assert frozen["price_move_5m_ticks"] == 4.0
    assert frozen["session_vwap"] == pytest.approx((7500 * 2 + 7501) / 3)
    assert frozen["session_high"] == 7501.0 and frozen["session_low"] == 7500.0
    assert frozen["recent_range_5m_ticks"] == 4.0
    assert frozen["execution_count_60s"] == 1


def test_direction_normalization_and_causal_level_history_exclude_current_and_future():
    assert regime.direction_normalize(-8.0, "BUYER_ABSORPTION") == -8.0
    assert regime.direction_normalize(-8.0, "SELLER_ABSORPTION") == 8.0
    signals = [
        {"interaction_id": "first", "date": "2026-01-01", "level": "CURRENT_RTH_LOW_SWEEP", "interaction_end": 10},
        {"interaction_id": "second", "date": "2026-01-01", "level": "CURRENT_RTH_LOW_SWEEP", "interaction_end": 20},
        {"interaction_id": "other", "date": "2026-01-01", "level": "CURRENT_RTH_HIGH_SWEEP", "interaction_end": 15},
    ]
    levels = regime.previous_level_counts(signals)
    interactions = [
        {"date": "2026-01-01", "level": "CURRENT_RTH_LOW_SWEEP", "interaction_end": "5"},
        {"date": "2026-01-01", "level": "CURRENT_RTH_LOW_SWEEP", "interaction_end": "20"},
        {"date": "2026-01-01", "level": "CURRENT_RTH_LOW_SWEEP", "interaction_end": "30"},
    ]
    completed = regime.previous_completed_interaction_counts(interactions, signals)
    assert levels["first"]["previous_plus_events_same_level"] == 0
    assert levels["second"]["previous_plus_events_same_level"] == 1
    assert completed["first"] == 1 and completed["second"] == 1


def test_grouping_is_descriptive_and_does_not_change_the_signal_rows():
    rows = [{"absorption_score": 0.8, "replenishment_score": 0.9, "period": "AUGUST_SEEN", "direction": "BUYER_ABSORPTION", "level": "CURRENT_RTH_LOW_SWEEP", "trade_outcome": "WIN"}]
    original = [dict(row) for row in rows]
    result = regime._group(rows)
    assert result["absorption_score"]["count"] == 1
    assert rows == original


def compact_row(**overrides):
    row = {"period": "RETRO_JUNE_JULY", "direction": "BUYER_ABSORPTION", "level": "CURRENT_RTH_LOW_SWEEP", "trade_outcome": "LOSS"}
    row.update({feature: "1" for feature in compact.FEATURES}); row.update(overrides)
    return row


def test_compact_group_filtering_period_direction_level_and_numeric_deltas():
    rows = [compact_row(absorption_score="2"), compact_row(direction="SELLER_ABSORPTION", level="CURRENT_RTH_HIGH_SWEEP", absorption_score="1"), compact_row(period="AUGUST_SEEN", absorption_score="4")]
    buyer = compact.select(rows, period="RETRO_JUNE_JULY", direction="BUYER_ABSORPTION")
    seller = compact.select(rows, period="RETRO_JUNE_JULY", direction="SELLER_ABSORPTION")
    assert len(buyer) == 1 and buyer[0]["level"] == "CURRENT_RTH_LOW_SWEEP"
    row = next(item for item in compact.delta("TEST", buyer, seller) if item["feature"] == "absorption_score")
    assert row["left_median"] == 2.0 and row["right_median"] == 1.0 and row["left_minus_right_median"] == 1.0


def test_compact_missing_numeric_fields_and_grouping_leave_population_unchanged():
    rows = [compact_row(executed_volume_60s=""), compact_row(direction="SELLER_ABSORPTION", trade_outcome="WIN")]
    original = [dict(row) for row in rows]
    summary = compact.group_summary(rows)
    assert summary["features"]["executed_volume_60s"]["count"] == 1
    assert rows == original
    assert "threshold" not in json.dumps(summary).lower()


def test_compact_report_answers_groups_without_selecting_a_rule():
    rows = [compact_row(), compact_row(direction="SELLER_ABSORPTION"), compact_row(period="AUGUST_SEEN")]
    groups = {
        "AUGUST_SEEN_BUYER_ABSORPTION": compact.select(rows, period="AUGUST_SEEN", direction="BUYER_ABSORPTION"),
        "AUGUST_SEEN_SELLER_ABSORPTION": compact.select(rows, period="AUGUST_SEEN", direction="SELLER_ABSORPTION"),
        "RETRO_JUNE_JULY_BUYER_ABSORPTION": compact.select(rows, period="RETRO_JUNE_JULY", direction="BUYER_ABSORPTION"),
        "RETRO_JUNE_JULY_SELLER_ABSORPTION": compact.select(rows, period="RETRO_JUNE_JULY", direction="SELLER_ABSORPTION"),
        "AUGUST_SEEN_CURRENT_RTH_LOW_SWEEP": compact.select(rows, period="AUGUST_SEEN", level="CURRENT_RTH_LOW_SWEEP"),
        "AUGUST_SEEN_CURRENT_RTH_HIGH_SWEEP": [],
        "RETRO_JUNE_JULY_CURRENT_RTH_LOW_SWEEP": compact.select(rows, period="RETRO_JUNE_JULY", level="CURRENT_RTH_LOW_SWEEP"),
        "RETRO_JUNE_JULY_CURRENT_RTH_HIGH_SWEEP": [],
        "WIN": [], "LOSS": rows,
    }
    comparisons = [
        ("RETRO_BUYER_MINUS_RETRO_SELLER", groups["RETRO_JUNE_JULY_BUYER_ABSORPTION"], groups["RETRO_JUNE_JULY_SELLER_ABSORPTION"]),
        ("AUGUST_BUYER_MINUS_AUGUST_SELLER", groups["AUGUST_SEEN_BUYER_ABSORPTION"], groups["AUGUST_SEEN_SELLER_ABSORPTION"]),
        ("RETRO_LOW_SWEEP_MINUS_RETRO_HIGH_SWEEP", groups["RETRO_JUNE_JULY_CURRENT_RTH_LOW_SWEEP"], groups["RETRO_JUNE_JULY_CURRENT_RTH_HIGH_SWEEP"]),
        ("AUGUST_LOW_SWEEP_MINUS_AUGUST_HIGH_SWEEP", groups["AUGUST_SEEN_CURRENT_RTH_LOW_SWEEP"], groups["AUGUST_SEEN_CURRENT_RTH_HIGH_SWEEP"]),
        ("RETRO_BUYER_MINUS_AUGUST_BUYER", groups["RETRO_JUNE_JULY_BUYER_ABSORPTION"], groups["AUGUST_SEEN_BUYER_ABSORPTION"]),
    ]
    deltas = [item for label, left, right in comparisons for item in compact.delta(label, left, right)]
    report = compact.diagnostic_report(groups, deltas)
    assert "## Descriptive answers" in report
    assert "new strategy rule" in report


def bucket_row(**overrides):
    value = compact_row(trade_outcome="WIN", r_multiple="1", absorption_score="0.8", replenishment_score="0.81")
    value.update({feature: "1" for feature in buckets.FEATURES})
    value.update(overrides)
    return value


def test_bucket_assignment_is_deterministic_balanced_and_handles_equal_values():
    rows = [bucket_row(interaction_id=f"row-{index}", recent_range_5m_ticks="5") for index in range(5)]
    assigned, missing = buckets.assign_quartiles(rows, "recent_range_5m_ticks")
    assert missing == 0
    assert [len(assigned[f"Q{index}"]) for index in range(1, 5)] == [2, 1, 1, 1]
    assert [row["interaction_id"] for row in assigned["Q1"]] == ["row-0", "row-1"]


def test_bucket_missing_values_outcomes_and_population_separation():
    rows = [
        bucket_row(period="RETRO_JUNE_JULY", direction="BUYER_ABSORPTION", r_multiple="2"),
        bucket_row(period="RETRO_JUNE_JULY", direction="SELLER_ABSORPTION", trade_outcome="LOSS", r_multiple="-1"),
        bucket_row(period="AUGUST_SEEN", direction="BUYER_ABSORPTION", trade_outcome="NOT_TRADED", r_multiple="", execution_count_15s=""),
    ]
    buyer = buckets.select(rows, period="RETRO_JUNE_JULY", direction="BUYER_ABSORPTION")
    seller = buckets.select(rows, period="RETRO_JUNE_JULY", direction="SELLER_ABSORPTION")
    assert len(buyer) == 1 and len(seller) == 1
    _, missing = buckets.assign_quartiles(rows, "execution_count_15s")
    assert missing == 1
    metrics = buckets.bucket_metrics(buyer, "execution_count_15s")
    assert metrics["traded_setups"] == 1 and metrics["average_r"] == 2.0


def test_bucket_leave_one_out_and_monotonic_classification_without_rule_selection():
    rows = [bucket_row(r_multiple="-1"), bucket_row(r_multiple="1"), bucket_row(r_multiple="2")]
    sensitivity = buckets.leave_one_out(rows)
    assert sensitivity == {"traded_rows": 3, "original_total_r": 2.0, "best_case_remove_worst_r": 3.0, "worst_case_remove_best_r": 0.0}
    monotonic = buckets.classify_monotonicity("recent_range_5m_ticks", {
        "Q1": {"average_r": 2.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q2": {"average_r": 1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q3": {"average_r": 0.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q4": {"average_r": -1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
    })
    assert monotonic["label"] == "CLEAR_MONOTONIC_PATTERN"
    assert "threshold" not in json.dumps(monotonic).lower()
    endpoint_reversal = buckets.classify_monotonicity("direction_normalized_price_move_5m_ticks", {
        "Q1": {"average_r": 1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q2": {"average_r": -1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q3": {"average_r": -1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
        "Q4": {"average_r": -1.0, "traded_setups": 2, "median_absorption_score": 0.8, "median_replenishment_score": 0.8},
    })
    assert endpoint_reversal["label"] == "NO_MONOTONIC_PATTERN"


def test_bucket_materialization_preserves_source_rows_and_selects_no_rule(tmp_path):
    rows = [bucket_row(interaction_id=f"row-{index}", r_multiple=str(index - 3), trade_outcome="WIN" if index % 2 else "LOSS", recent_range_5m_ticks=str(index)) for index in range(8)]
    source = tmp_path / "setup-context.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    before = source.read_text(encoding="utf-8")
    result = buckets.materialize(source=source, output_dir=tmp_path / "output")
    assert source.read_text(encoding="utf-8") == before
    assert result["source_row_count"] == 8 and result["source_population_unchanged"]
    assert result["selection_prohibited"] and not result["pnl_optimization_performed"]


def test_fresh_quote_previous_completed_rth_skips_weekend_and_documented_closure():
    assert fresh_quote.previous_completed_rth(fresh_quote.date(2026, 8, 17)) == fresh_quote.date(2026, 8, 14)
    assert fresh_quote.previous_completed_rth(fresh_quote.date(2026, 9, 8), [fresh_quote.date(2026, 9, 7)]) == fresh_quote.date(2026, 9, 4)
    sessions = fresh_quote.rth_sessions(fresh_quote.date(2026, 8, 17), 20, [fresh_quote.date(2026, 9, 7)])
    assert sessions[0] == fresh_quote.date(2026, 8, 17) and sessions[-1] == fresh_quote.date(2026, 9, 14)
    assert fresh_quote.date(2026, 9, 7) not in sessions


def test_fresh_quote_plan_has_only_explicit_required_windows_and_components():
    components = fresh_quote.component_plan(start=fresh_quote.date(2026, 8, 17), sessions=5, es_symbol="ESU6", mes_symbol="MESU6")
    mbo = [item for item in components if item.label == "ES_MBO_L3"]
    mes = [item for item in components if item.label == "MES_NATIVE_EXECUTION"]
    prior = [item for item in components if item.label == "ES_PRIOR_RTH_PROFILE"]
    assert len(mbo) == len(mes) == len(prior) == 5
    assert all(item.start.endswith("T00:00:00Z") and "snapshot" in item.purpose.lower() for item in mbo)
    assert all(item.schema == "mbp-1" and item.start.endswith("T13:30:00Z") for item in mes)
    assert all(item.schema == "trades" and item.start.endswith("T13:30:00Z") and item.end.endswith("T20:00:00Z") for item in prior)


def test_fresh_quote_cost_uses_metadata_only_and_reconciles_total():
    class Metadata:
        def __init__(self): self.calls = []
        def get_cost(self, **request): self.calls.append(request); return 1.25
    class Client:
        def __init__(self): self.metadata = Metadata()
        @property
        def timeseries(self): raise AssertionError("quote-only path must not expose downloads")
        @property
        def batch(self): raise AssertionError("quote-only path must not expose downloads")
    components = fresh_quote.component_plan(start=fresh_quote.date(2026, 8, 17), sessions=5, es_symbol="ESU6", mes_symbol="MESU6")
    client = Client()
    result = fresh_quote.quote_components(components, client)
    assert result["download_api_invoked"] is False
    assert len(client.metadata.calls) == len(components)
    assert result["total_estimated_usd"] == pytest.approx(1.25 * len(components))


def test_fresh_quote_payload_is_fresh_and_names_both_frozen_strategies():
    payload = fresh_quote.plan_payload(start=fresh_quote.date(2026, 8, 17), sessions=5, es_symbol="ESU6", mes_symbol="MESU6", closed_rth_dates=fresh_quote.DEFAULT_CLOSED_RTH_DATES)
    assert payload["label"] == "FRESH_UNTOUCHED_OOS" and payload["download_api_invoked"] is False
    assert {item["strategy_id"] for item in payload["strategies"]} == {"CMEOrderflowAbsorption.ES_V2_BASELINE_FRESH_OOS", "CMEOrderflowAbsorption.ES_V3_TICK_3_TARGET_3R_FRESH_OOS"}


def test_may_cost_proxy_uses_exact_completed_rth_sessions_and_memorial_day():
    closed = fresh_quote.COST_PROXY_CLOSED_RTH_DATES
    expected = {
        5: ("2026-05-04", "2026-05-08"),
        10: ("2026-05-04", "2026-05-15"),
        15: ("2026-05-04", "2026-05-22"),
        20: ("2026-05-04", "2026-06-01"),
    }
    for count, (first, last) in expected.items():
        sessions = fresh_quote.rth_sessions(fresh_quote.COST_PROXY_START, count, closed)
        assert sessions[0].isoformat() == first and sessions[-1].isoformat() == last
        assert fresh_quote.date(2026, 5, 25) not in sessions
    assert fresh_quote.previous_completed_rth(fresh_quote.date(2026, 5, 4), closed) == fresh_quote.date(2026, 5, 1)
    assert fresh_quote.previous_completed_rth(fresh_quote.date(2026, 5, 26), closed) == fresh_quote.date(2026, 5, 22)


def test_may_cost_proxy_is_explicitly_non_evidence_and_requires_symbol_resolution():
    payload = fresh_quote.plan_payload(
        start=fresh_quote.COST_PROXY_START,
        sessions=20,
        es_symbol=fresh_quote.COST_PROXY_ES_CANDIDATE,
        mes_symbol=fresh_quote.COST_PROXY_MES_CANDIDATE,
        closed_rth_dates=fresh_quote.COST_PROXY_CLOSED_RTH_DATES,
        mode="cost-proxy",
    )
    assert payload["label"] == "HISTORICAL_COST_PROXY_ONLY_NOT_STRATEGY_EVIDENCE"
    assert payload["not_strategy_evidence"] is True
    assert payload["not_eligible_for_signal_or_performance_research"] is True
    assert payload["symbol_resolution_required_before_cost_quote"] is True
    assert payload["download_api_invoked"] is False and payload["cost_api_invoked"] is False


def test_cost_proxy_resolves_raw_contracts_with_symbology_before_quote():
    class Symbology:
        def __init__(self): self.calls = []
        def resolve(self, **request):
            self.calls.append(request)
            return {"status": "OK", "raw_symbol": request["symbols"][0], "instrument_id": 123, "partial": [], "not_found": []}
    class Client:
        def __init__(self): self.symbology = Symbology()
        @property
        def metadata(self): raise AssertionError("this test must stop before a cost request")
        @property
        def timeseries(self): raise AssertionError("cost-proxy planning must not download")
        @property
        def batch(self): raise AssertionError("cost-proxy planning must not download")
    components = fresh_quote.component_plan(
        start=fresh_quote.COST_PROXY_START,
        sessions=20,
        es_symbol="ESM6",
        mes_symbol="MESM6",
        closed_rth_dates=fresh_quote.COST_PROXY_CLOSED_RTH_DATES,
        allow_historical_cost_proxy=True,
    )
    client = Client()
    resolutions = fresh_quote.resolve_raw_symbols_for_quote(components=components, client=client)
    assert {entry["raw_symbol"] for entry in resolutions} == {"ESM6", "MESM6"}
    assert {call["stype_out"] for call in client.symbology.calls} == {"instrument_id"}
    assert all(call["dataset"] == "GLBX.MDP3" and call["stype_in"] == "raw_symbol" for call in client.symbology.calls)


def test_symbology_accepts_numeric_success_and_date_scoped_raw_symbol_mapping():
    request = {
        "dataset": "GLBX.MDP3", "symbols": ["ESM6"], "stype_in": "raw_symbol",
        "stype_out": "instrument_id", "start_date": "2026-05-01", "end_date": "2026-05-08",
    }
    result = {
        "status": 0,
        "partial": [],
        "not_found": [],
        "result": {
            "ESM6": [
                {"d0": "2026-05-01", "d1": "2026-05-04", "s": "123"},
                {"d0": "2026-05-04", "d1": "2026-05-08", "s": "123"},
            ],
        },
    }
    assert fresh_quote._validated_symbology_result(result=result, symbol="ESM6", request=request)["status"] == 0


def test_symbology_not_found_and_incompatible_partial_fail_closed_without_secret_output():
    request = {
        "dataset": "GLBX.MDP3", "symbols": ["MESM6"], "stype_in": "raw_symbol",
        "stype_out": "instrument_id", "start_date": "2026-05-04", "end_date": "2026-05-08",
    }
    for result, expected in (
        ({"status": 0, "partial": [], "not_found": ["MESM6"], "result": {}}, "did not find"),
        ({"status": 0, "partial": ["2026-05-07"], "not_found": [], "result": {"MESM6": [{"d0": "2026-05-04"}]}}, "partial"),
    ):
        with pytest.raises(fresh_quote.SymbologyValidationError, match=expected) as raised:
            fresh_quote._validated_symbology_result(result=result, symbol="MESM6", request=request)
        rendered = json.dumps(raised.value.diagnostics)
        assert "DATABENTO_API_KEY" not in rendered and "secret" not in rendered.lower()
        assert raised.value.diagnostics["requested"]["stype_in"] == "raw_symbol"
        assert raised.value.diagnostics["requested"]["stype_out"] == "instrument_id"


def test_may_acquisition_has_exact_sealed_15_session_component_windows():
    components = may_acquisition._components()
    mbo = [item for item in components if item.label == "ES_MBO_L3"]
    mes = [item for item in components if item.label == "MES_NATIVE_EXECUTION"]
    prior = [item for item in components if item.label == "ES_PRIOR_RTH_PROFILE"]
    assert len(components) == 45 and len(mbo) == len(mes) == len(prior) == 15
    assert [item.session_date for item in mbo] == [
        "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
        "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
        "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
    ]
    assert prior[0].session_date == "2026-05-01"
    assert all(item.symbol == "ESM6" and item.start.endswith("T00:00:00Z") and item.end.endswith("T22:45:01Z") for item in mbo)
    assert all(item.symbol == "MESM6" and item.start.endswith("T13:30:00Z") and item.end.endswith("T22:45:01Z") for item in mes)
    assert all(item.symbol == "ESM6" and item.start.endswith("T13:30:00Z") and item.end.endswith("T20:00:00Z") for item in prior)


def _may_quote_client(*, allow_download: bool = False):
    class Symbology:
        def resolve(self, **request):
            symbol = request["symbols"][0]
            return {"status": 0, "partial": [], "not_found": [], "result": {symbol: [{"d0": request["start_date"], "d1": request["end_date"], "s": "123"}]}}
    class Metadata:
        def __init__(self): self.calls = []
        def get_cost(self, **request): self.calls.append(request); return 41.81 / 45
    class Timeseries:
        def __init__(self): self.calls = []
        def get_range(self, **request):
            self.calls.append(request)
            if not allow_download:
                raise AssertionError("download must require --download")
            Path(request["path"]).write_bytes(b"DBN")
    class Client:
        def __init__(self):
            self.symbology = Symbology(); self.metadata = Metadata(); self.timeseries = Timeseries()
    return Client()


def test_may_acquisition_requotes_before_any_download_and_has_no_download_mode(tmp_path):
    client = _may_quote_client()
    result = may_acquisition.acquire(root=tmp_path / "package", client=client, download=False)
    assert result["estimated_total_usd"] == "41.81"
    assert result["download_api_invoked"] is False
    assert len(client.metadata.calls) == 45 and not client.timeseries.calls
    assert not (tmp_path / "package").exists()


def test_may_acquisition_atomic_manifest_resume_skips_only_hash_verified_files(tmp_path):
    root = tmp_path / "package"
    first_client = _may_quote_client(allow_download=True)
    first = may_acquisition.acquire(root=root, client=first_client, download=True)
    manifest = json.loads((root / may_acquisition.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert first["download_api_invoked"] is True and len(first_client.timeseries.calls) == 45
    assert manifest["data_acquired"] is True and len(manifest["files"]) == 45
    second_client = _may_quote_client(allow_download=False)
    second = may_acquisition.acquire(root=root, client=second_client, download=True)
    assert len(second_client.timeseries.calls) == 0
    assert {item["status"] for item in second["files"]} == {"SKIPPED_VERIFIED"}


def test_may_acquisition_rejects_material_quote_change_before_creating_output(tmp_path):
    client = _may_quote_client()
    client.metadata.get_cost = lambda **_request: 2.0
    with pytest.raises(may_acquisition.AcquisitionError, match="differs materially"):
        may_acquisition.acquire(root=tmp_path / "package", client=client, download=True)
    assert not (tmp_path / "package").exists() and not client.timeseries.calls
