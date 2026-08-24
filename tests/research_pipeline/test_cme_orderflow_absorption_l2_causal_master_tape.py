from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as tape
from research_pipeline.cme_orderflow_absorption_l2_v1 import native_hard_flat_preflight as boundary_audit
from research_pipeline.cme_orderflow_absorption_l2_v1.model import Execution


DAY = "2025-12-01"
END = 1_000_000_000


def _interaction(
    identifier: str = "I1", *, score: float = 0.60, direction: str = "BUYER_ABSORPTION",
    zone_low: float = 100.0, zone_high: float = 100.5, primitive_rejection: str = "",
) -> dict[str, object]:
    source = f"PRIOR_RTH_POC:100.00:{identifier}"
    return {
        "interaction_id": f"{DAY}|{source}", "source_interaction_id": source,
        "session_date": DAY, "interaction_start_ns": END - 1_000_000_000,
        "interaction_end_ns": END, "direction": direction, "level": "PRIOR_RTH_POC",
        "level_price": 100.0, "interaction_end_price": 100.0,
        "zone_low": zone_low, "zone_high": zone_high, "termination": "VICINITY_TIMEOUT",
        "aggression_score": score, "restoration_score": score,
        "price_resistance_score": score, "persistence_score": score,
        "multi_level_support_score": score, "false_refill_penalty": 0.0,
        "original_v3_quality_score": score,
        "original_v3_rejection_reasons": primitive_rejection,
        "non_quality_rejection_reasons": primitive_rejection,
        "weights_label": "L2_V1_PREDECLARED_RESEARCH_WEIGHTS",
    }


def _event(
    ordinal: int, timestamp_ns: int, *, stream: str = "ES", event_type: str = "ES_BBO",
    source_index: int | None = None, es_bid: float | None = 100.5, es_ask: float | None = 100.75,
    mes_bid: float | None = 100.5, mes_ask: float | None = 100.75,
    execution_price: float | None = None, aggressor: str | None = None,
    hard_flat_reason: str | None = None, book_state: str = "EXECUTABLE",
) -> dict[str, object]:
    return {
        "session_date": DAY, "event_ordinal": ordinal, "timestamp_ns": timestamp_ns,
        "stream": stream, "stream_priority": tape.STREAM_PRIORITY[stream],
        "source_index": ordinal + 1 if source_index is None else source_index,
        "event_type": event_type, "es_bid": es_bid, "es_ask": es_ask,
        "mes_bid": mes_bid, "mes_ask": mes_ask,
        "execution_price": execution_price, "execution_size": 1 if execution_price is not None else None,
        "execution_aggressor": aggressor, "book_state": book_state,
        "entry_probe_count": 0, "es_quote_timestamp_ns": timestamp_ns if es_bid is not None else None,
        "mes_quote_timestamp_ns": timestamp_ns if mes_bid is not None else None,
        "hard_flat_reason": hard_flat_reason,
    }


def _write_master(root: Path, interactions: list[dict[str, object]], events: list[dict[str, object]]) -> Path:
    (root / "causal-event-tape").mkdir(parents=True)
    tape._write_small_parquet(root / "interaction-master.parquet", interactions)
    tape._write_small_parquet(root / "causal-event-tape" / f"{DAY}.parquet", events)
    tape._write_small_parquet(root / "interaction-event-index.parquet", [
        {
            "interaction_id": row["interaction_id"], "session_date": DAY,
            "event_partition": f"causal-event-tape/{DAY}.parquet",
            "confirmation_start_ns": int(row["interaction_end_ns"]) + 5_000_000_000,
            "confirmation_end_ns_inclusive": int(row["interaction_end_ns"]) + 15_000_000_000,
            "counterfactual_path_end_ns": events[-1]["timestamp_ns"],
        }
        for row in interactions
    ])
    (root / "calendar.json").write_text(json.dumps({"target_sessions": [DAY]}), encoding="utf-8")
    return root


def _complete_building(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, interaction_count: int = 1,
) -> Path:
    monkeypatch.setattr(tape, "_expected_session_days", lambda: (DAY,))
    monkeypatch.setattr(tape.parent, "EXPECTED_SESSION_COUNT", 1)
    monkeypatch.setattr(tape, "EXPECTED_INTERACTIONS", interaction_count)
    root.mkdir(parents=True)
    interactions = [_interaction(str(index)) for index in range(interaction_count)]
    event = _event(0, END + 20_000_000_000)
    event_writer = tape.AtomicParquetStream(root / "causal-event-tape" / f"{DAY}.parquet")
    event_writer.append(event)
    event_artifact = event_writer.close()
    index_rows = [
        {
            "interaction_id": row["interaction_id"], "session_date": DAY,
            "event_partition": f"causal-event-tape/{DAY}.parquet",
            "session_first_event_ordinal": 0, "session_last_event_ordinal": 0,
            "confirmation_start_ns": int(row["interaction_end_ns"]) + 5_000_000_000,
            "confirmation_end_ns_inclusive": int(row["interaction_end_ns"]) + 15_000_000_000,
            "derived_first_confirmation_timestamp_ns": None,
            "derived_first_confirmation_price": None, "entry_ready_ns": None,
            "entry_observation_event_ordinal": None,
            "counterfactual_path_end_ns": event["timestamp_ns"],
        }
        for row in interactions
    ]
    session_interactions = tape._write_small_parquet(
        root / "session-builds" / DAY / "interactions.parquet", interactions,
    )
    session_index = tape._write_small_parquet(
        root / "session-builds" / DAY / "interaction-index.parquet", index_rows,
    )
    tape._write_small_parquet(root / "interaction-master.parquet", interactions)
    tape._write_small_parquet(root / "interaction-event-index.parquet", index_rows)
    tape._write_json(root / "session-builds" / f"{DAY}.json", {
        "session_date": DAY, "status": "SESSION_CAUSAL_TAPE_COMPLETE",
        "decoded_source_records": 1, "completed_interactions": interaction_count,
        "stored_events": 1, "confirmed_interactions": 0, "entry_probe_interactions": 0,
        "artifacts": {
            "events": event_artifact,
            "interactions": session_interactions,
            "index": session_index,
        },
    })
    tape._write_json(root / "calendar.json", {
        "calendar_semantics_version": tape.CALENDAR_SEMANTICS_VERSION,
        "target_sessions": [DAY], "sessions": [{"day": DAY}],
    })
    tape._write_json(root / "metadata.json", {
        "artifact_kind": "CAUSAL_MASTER_TAPE",
        "event_schema_version": tape.EVENT_SCHEMA_VERSION,
        "ordering_rule_version": tape.ORDERING_RULE_VERSION,
        "v3_contract_sha256": tape.V3_CONTRACT_SHA256,
        "v4_contract_sha256": tape.V4_CONTRACT_SHA256,
        "network_calls": 0, "downloads": 0,
        "threshold_matrix_executed": False,
        "weight_grid_generated_or_evaluated": False,
    })
    gates = {
        "status": "EXACT_REPRODUCTION_GATES_PASS",
        "v3_december": {"metrics": deepcopy(tape.EXPECTED_GATES["V3_DECEMBER"])},
        "v3_full": {"metrics": deepcopy(tape.EXPECTED_GATES["V3_FULL"])},
        "v4_full": {"metrics": deepcopy(tape.EXPECTED_GATES["V4_FULL"])},
        "threshold_matrix_executed": False,
        "weight_grid_generated_or_evaluated": False,
    }
    tape._write_json(root / "reproduction-gates.json", gates)
    tape._write_json(root / "build-summary.json", {
        "status": "CAUSAL_MASTER_TAPE_VALID", "sessions": 1,
        "completed_interactions": interaction_count, "indexed_interactions": interaction_count,
        "stored_events": 1, "heavy_dbn_passes": 1,
        "threshold_performance_matrix_executed": False,
        "weight_grid_generated_or_evaluated": False,
    })
    (root / "diagnostic-report.md").write_text("validated fixture\n", encoding="utf-8")
    return root


def test_checkpoint_paths_from_pre_publish_building_root_resolve_in_final_root(tmp_path: Path):
    final_root = tmp_path / "FINAL"
    final_root.mkdir()
    target = final_root / "causal-event-tape" / f"{DAY}.parquet"
    raw = Path("research_runs") / "FINAL.building" / "causal-event-tape" / f"{DAY}.parquet"
    assert tape._checkpoint_target(final_root, raw) == target.resolve()


def _long_target_events() -> list[dict[str, object]]:
    confirmation = END + 5_000_000_000
    return [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, aggressor="BUY"),
        _event(1, confirmation + 1_000_000, es_bid=100.75, es_ask=101.0),
        _event(2, confirmation + 2_000_000, es_bid=100.5, es_ask=100.75),
        _event(3, confirmation + 3_000_000, es_bid=108.0, es_ask=108.25),
        _event(4, END + 20_000_000_000, stream="CALENDAR", event_type="HARD_FLAT",
               hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE"),
    ]


def test_all_pre_quality_interactions_receive_windows_even_when_rejected_or_low_quality():
    tracker = tape.CausalWindowTracker()
    rows = [
        _interaction("LOW", score=0.20),
        _interaction("REJECT", score=0.90, primitive_rejection="NO_GENUINE_CONSUME_RESTORE"),
        _interaction("BLOCKED", score=0.90),
    ]
    for row in rows:
        tracker.register(row)
    tracker.observe_es_execution(Execution(END + 5_000_000_000, 100.75, 1, "BUY"))
    due = tracker.due_entry_probes(END + 5_002_000_000)
    tracker.bind_entry_probe(due, 7)
    index = tracker.index_rows(day=DAY, first_event=0, last_event=10, cutoff_ns=END + 20_000_000_000)
    assert len(index) == 3
    assert all(row["derived_first_confirmation_timestamp_ns"] == END + 5_000_000_000 for row in index)
    assert all(row["entry_observation_event_ordinal"] == 7 for row in index)


@pytest.mark.parametrize("offset", [5_000_000_000, 15_000_000_000])
def test_confirmation_boundaries_are_inclusive(offset: int):
    tracker = tape.CausalWindowTracker()
    tracker.register(_interaction())
    tracker.observe_es_execution(Execution(END + offset, 100.75, 1, "BUY"))
    row = next(iter(tracker.windows.values()))
    assert row.confirmation_timestamp_ns == END + offset


def test_first_plus_three_execution_is_frozen_and_later_reaches_do_not_replace_it():
    tracker = tape.CausalWindowTracker()
    tracker.register(_interaction())
    tracker.observe_es_execution(Execution(END + 5_000_000_000, 100.50, 1, "BUY"))
    first = END + 6_000_000_000
    tracker.observe_es_execution(Execution(first, 100.75, 1, "BUY"))
    tracker.observe_es_execution(Execution(END + 7_000_000_000, 101.25, 1, "BUY"))
    row = next(iter(tracker.windows.values()))
    assert row.confirmation_timestamp_ns == first and row.confirmation_price == 100.75


def test_first_entry_probe_obeys_two_millisecond_latency():
    tracker = tape.CausalWindowTracker()
    tracker.register(_interaction())
    confirmation = END + 5_000_000_000
    tracker.observe_es_execution(Execution(confirmation, 100.75, 1, "BUY"))
    assert tracker.due_entry_probes(confirmation + 1_999_999) == []
    assert tracker.due_entry_probes(confirmation + 2_000_000) == [_interaction()["interaction_id"]]


def test_shared_event_order_is_mes_then_es_then_calendar_and_source_stable():
    rows = [
        _event(0, 10, stream="MES", source_index=1),
        _event(1, 10, stream="MES", source_index=2),
        _event(2, 10, stream="ES", source_index=1),
        _event(3, 10, stream="CALENDAR", event_type="HARD_FLAT", source_index=0),
    ]
    assert tape.validate_event_order(rows) == 4
    with pytest.raises(tape.CausalMasterTapeError, match="ordering"):
        tape.validate_event_order([rows[2], rows[0]])


def test_long_target_path_es_first_and_latency_are_replayed_without_dbn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _write_master(tmp_path / "master", [_interaction()], _long_target_events())
    monkeypatch.setattr(tape.native, "_stream_native_mbp10_records", lambda *_: pytest.fail("DBN opened"))
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["dbn_files_opened"] == 0 and result["network_calls"] == 0
    assert result["metrics"]["completed_trades"] == 1
    trade = result["trades"][0]
    assert trade["instrument"] == "ES"
    assert trade["entry_timestamp_ns"] == END + 5_002_000_000
    assert trade["exit_reason"] == "TARGET"


def test_mes_fallback_uses_native_mes_path_when_es_does_not_fit(tmp_path: Path):
    confirmation = END + 5_000_000_000
    events = [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, aggressor="BUY"),
        _event(1, confirmation + 2_000_000, stream="MES", es_bid=100.5, es_ask=100.75,
               mes_bid=100.5, mes_ask=100.75),
        _event(2, confirmation + 3_000_000, stream="MES", es_bid=100.5, es_ask=100.75,
               mes_bid=140.0, mes_ask=140.25),
        _event(3, END + 20_000_000_000, stream="CALENDAR", event_type="HARD_FLAT",
               es_bid=100.5, es_ask=100.75, mes_bid=140.0, mes_ask=140.25,
               hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE"),
    ]
    root = _write_master(tmp_path / "master", [_interaction(zone_low=90.0)], events)
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["metrics"]["completed_trades"] == 1
    assert result["trades"][0]["instrument"] == "MES"


def test_stop_is_evaluated_before_target_on_same_es_observation(tmp_path: Path):
    events = _long_target_events()
    events[3] = _event(3, END + 5_003_000_000, es_bid=98.0, es_ask=108.25)
    root = _write_master(tmp_path / "master", [_interaction()], events)
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["trades"][0]["exit_reason"] == "STOP"


def test_hard_flat_uses_last_causal_quote_and_calendar_reason(tmp_path: Path):
    confirmation = END + 5_000_000_000
    cutoff = END + 20_000_000_000
    events = [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, aggressor="BUY"),
        _event(1, confirmation + 2_000_000),
        _event(2, cutoff - 500_000_000, es_bid=101.0, es_ask=101.25),
        _event(3, cutoff, stream="CALENDAR", event_type="HARD_FLAT",
               es_bid=101.0, es_ask=101.25, hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE"),
    ]
    root = _write_master(tmp_path / "master", [_interaction()], events)
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["trades"][0]["exit_reason"] == "HARD_FLAT_SCHEDULED_CLOSE"
    assert result["trades"][0]["exit_timestamp_ns"] == cutoff


def test_native_liquidation_boundary_accepts_last_executable_bbo_without_exact_cutoff_record():
    cutoff = 10_000_000_000
    evidence = tape.resolve_native_hard_flat_boundary(
        cutoff_ns=cutoff,
        es_quote=(6_500.00, 6_500.25),
        es_quote_timestamp_ns=cutoff - 750_000_000,
        mes_quote=(6_500.00, 6_500.25),
        mes_quote_timestamp_ns=cutoff - 250_000_000,
    )
    assert evidence["es_quote_timestamp_ns"] == cutoff - 750_000_000
    assert evidence["mes_quote_timestamp_ns"] == cutoff - 250_000_000
    assert evidence["exact_hard_flat_record_required"] is False
    assert evidence["invented_quote_count"] == 0


def test_native_liquidation_boundary_drops_stale_instrument_but_keeps_valid_selected_instrument():
    cutoff = 10_000_000_000
    evidence = tape.resolve_native_hard_flat_boundary(
        cutoff_ns=cutoff,
        es_quote=(6_500.00, 6_500.25),
        es_quote_timestamp_ns=cutoff - tape.historical.CUTOFF_QUOTE_LOOKBACK_NS - 1,
        mes_quote=(6_500.00, 6_500.25),
        mes_quote_timestamp_ns=cutoff - 1,
    )
    assert evidence["es_quote"] is None and evidence["es_quote_timestamp_ns"] is None
    assert evidence["mes_quote"] == (6_500.00, 6_500.25)


def test_native_liquidation_boundary_rejects_truly_premature_source_end():
    cutoff = 10_000_000_000
    stale = cutoff - tape.historical.CUTOFF_QUOTE_LOOKBACK_NS - 1
    with pytest.raises(tape.CausalMasterTapeError, match="lacks any executable BBO"):
        tape.resolve_native_hard_flat_boundary(
            cutoff_ns=cutoff,
            es_quote=(6_500.00, 6_500.25),
            es_quote_timestamp_ns=stale,
            mes_quote=(6_500.00, 6_500.25),
            mes_quote_timestamp_ns=stale,
        )


def test_source_exhaustion_after_valid_liquidation_evidence_completes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cutoff = 10_000_000_000
    start = 2_000_000_000

    class Adapter:
        state = "EXECUTABLE"
        first_valid_book_ns = 1

        def feed(self, record, *, expected_non_executable=False):
            assert expected_non_executable is False
            return SimpleNamespace(snapshot=object(), update=None, execution=None)

        def assert_executable_at_boundary(self):
            return None

    class Engine:
        def __init__(self, *_args):
            self.completed = []

        def advance(self, _timestamp): pass
        def observe_snapshot(self, _snapshot, _update): pass
        def observe_execution(self, _execution): pass
        def finish_rth(self, _timestamp): pass

    es_rows = [SimpleNamespace(ts_recv=1_000_000_000), SimpleNamespace(ts_recv=cutoff - 500_000_000)]
    mes_rows = [(cutoff - 250_000_000, 6_500.00, 6_500.25)]
    monkeypatch.setattr(tape.native, "NativeMBP10Adapter", Adapter)
    monkeypatch.setattr(tape, "L2InteractionEngine", Engine)
    monkeypatch.setattr(tape.native, "_stream_native_mbp10_records", lambda _path: iter(es_rows))
    monkeypatch.setattr(tape.historical, "_stream_mes_quotes", lambda _path: iter(mes_rows))
    monkeypatch.setattr(tape.historical, "_quote", lambda _snapshot: (6_500.00, 6_500.25))

    result = tape._build_session(tape.SessionBuildSpec(
        "2026-08-12", "es.dbn", "mes.dbn", 6_500.00, start, cutoff,
        "HARD_CUTOFF_2245", str(tmp_path), maintenance_mode="NONE",
    ))
    assert result["status"] == "SESSION_CAUSAL_TAPE_COMPLETE"
    assert result["hard_flat_completion_mode"] == "SOURCE_ENDED_WITH_SUFFICIENT_LIQUIDATION_EVIDENCE"
    events = tape._read_parquet_rows(tmp_path / "causal-event-tape" / "2026-08-12.parquet")
    assert events[-1]["event_type"] == "HARD_FLAT"
    assert events[-1]["timestamp_ns"] == cutoff
    assert events[-1]["es_quote_timestamp_ns"] == cutoff - 500_000_000
    assert events[-1]["mes_quote_timestamp_ns"] == cutoff - 250_000_000


def test_aug10_14_calendar_contract_keeps_frozen_dst_and_scheduled_close_semantics():
    assert tuple(tape.native.TARGET_DATES) == (
        "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14",
    )
    for day in tape.native.TARGET_DATES[:-1]:
        assert tape.native.effective_hard_flat_seconds(day) == 22 * 60 * 60 + 45 * 60
        start, end = tape.native.liquidation_window_ns(day)
        assert end - start == tape.historical.CUTOFF_QUOTE_LOOKBACK_NS
    assert tape.native.effective_hard_flat_seconds("2026-08-14") == 21 * 60 * 60


def test_existing_native_mbp10_non_executable_state_machine_is_unchanged():
    scale = tape.native.RAW_PRICE_SCALE

    def record(timestamp: int, bid: float, ask: float):
        level = SimpleNamespace(
            bid_px=int(bid * scale), ask_px=int(ask * scale),
            bid_sz=10, ask_sz=10, bid_ct=1, ask_ct=1,
        )
        return SimpleNamespace(
            ts_recv=timestamp, ts_event=timestamp, action="M", side="B",
            price=int(bid * scale), size=1, levels=(level,),
        )

    adapter = tape.native.NativeMBP10Adapter()
    assert adapter.feed(record(1, 6_500.00, 6_500.25)) is not None
    assert adapter.state == "EXECUTABLE"
    assert adapter.feed(record(2, 6_500.25, 6_500.25)) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE" and adapter.previous is None
    reopened = adapter.feed(record(3, 6_500.00, 6_500.25))
    assert reopened is not None and adapter.state == "EXECUTABLE"
    assert adapter.last_transient == {
        "start_timestamp_ns": 2, "reopen_timestamp_ns": 3, "non_executable_records": 1,
    }


def test_native_hard_flat_preflight_contract_has_no_strategy_optimizer_or_pnl_path():
    source = Path(boundary_audit.__file__).read_text(encoding="utf-8").lower()
    assert "timeseries.get_range" not in source
    assert "metadata.get_cost" not in source
    assert "historical(" not in source
    assert "l2interactionengine" not in source
    assert '"strategy_logic_executed": false' in source
    assert '"pnl_calculated": false' in source


def test_non_executable_transition_clears_stale_bbo_and_prevents_entry(tmp_path: Path):
    confirmation = END + 5_000_000_000
    events = [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, aggressor="BUY"),
        _event(1, confirmation + 1_000_000, event_type="BOOK_NON_EXECUTABLE",
               es_bid=None, es_ask=None, mes_bid=None, mes_ask=None, book_state="WAITING_FOR_REOPEN_BOOK"),
        _event(2, confirmation + 2_000_000, es_bid=100.5, es_ask=100.75),
        _event(3, END + 20_000_000_000, stream="CALENDAR", event_type="HARD_FLAT",
               hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE"),
    ]
    root = _write_master(tmp_path / "master", [_interaction()], events)
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["metrics"]["completed_trades"] == 0


def test_transient_non_executable_book_suspends_without_changing_confirmation(tmp_path: Path):
    confirmation = END + 5_000_000_000
    events = [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, aggressor="BUY"),
        _event(1, confirmation + 1_000_000, event_type="BOOK_NON_EXECUTABLE",
               es_bid=None, es_ask=None, mes_bid=None, mes_ask=None,
               book_state="TEMPORARILY_NON_EXECUTABLE"),
        _event(2, confirmation + 2_000_000, es_bid=100.5, es_ask=100.75),
        _event(3, confirmation + 3_000_000, es_bid=108.0, es_ask=108.25),
        _event(4, END + 20_000_000_000, stream="CALENDAR", event_type="HARD_FLAT",
               hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE"),
    ]
    root = _write_master(tmp_path / "master", [_interaction()], events)
    result = tape.replay_configuration(root, threshold="0.50")
    assert result["metrics"]["completed_trades"] == 1
    assert result["trades"][0]["exit_reason"] == "TARGET"


def test_overlapping_setups_share_events_and_position_state_is_independent_per_threshold(tmp_path: Path):
    interactions = [_interaction("A", score=0.60), _interaction("B", score=0.46)]
    root = _write_master(tmp_path / "master", interactions, _long_target_events())
    q50_first = tape.replay_configuration(root, threshold="0.50")
    q45 = tape.replay_configuration(root, threshold="0.45")
    q50_second = tape.replay_configuration(root, threshold="0.50")
    assert q50_first["metrics"] == q50_second["metrics"]
    assert q50_first["accepted_setups"] == 1
    assert q45["accepted_setups"] == 2
    assert q45["metrics"]["completed_trades"] == 1


@pytest.mark.parametrize("threshold", ["0.35", "0.40", "0.45", "0.50", "0.55", "0.60"])
def test_all_predeclared_thresholds_are_supported_without_rounding(threshold: str):
    assert tape.normalize_threshold(threshold) == Decimal(threshold)


def test_raw_components_are_sufficient_for_future_legal_weight_recomputation():
    row = _interaction(score=0.0)
    row.update({
        "aggression_score": 1.0, "restoration_score": 0.5,
        "price_resistance_score": 0.25, "persistence_score": 0.75,
        "multi_level_support_score": 0.0, "false_refill_penalty": 0.1,
    })
    weights = {
        "aggression_score": "0.20", "restoration_score": "0.20",
        "price_resistance_score": "0.20", "persistence_score": "0.20",
        "multi_level_support_score": "0.20",
    }
    expected = 0.2 + 0.1 + 0.05 + 0.15 - 0.1 * tape.V2_CONFIG.false_refill_penalty_weight
    assert tape.recompute_quality(row, weights) == pytest.approx(expected)


def test_exact_reproduction_metric_gate_detects_any_difference():
    actual = dict(tape.EXPECTED_GATES["V3_FULL"])
    tape._assert_metrics("V3_FULL", actual, tape.EXPECTED_GATES["V3_FULL"])
    actual["completed_trades"] = 64
    with pytest.raises(tape.CausalMasterTapeError, match="completed_trades"):
        tape._assert_metrics("V3_FULL", actual, tape.EXPECTED_GATES["V3_FULL"])


def test_module_has_no_network_download_optimizer_or_outcome_artifact_dependency():
    source = Path(tape.__file__).read_text(encoding="utf-8").lower()
    assert "timeseries.get_range" not in source
    assert "metadata.get_cost" not in source
    assert "historical(" not in source
    assert "optimizer" not in source
    assert "trade-ledger.csv" not in source
    assert "interaction-features.csv" not in source


def test_immutable_parquet_collision_is_rejected(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    tape._write_small_parquet(path, [{"value": 1}])
    with pytest.raises(tape.CausalMasterTapeError, match="overwrite"):
        tape._write_small_parquet(path, [{"value": 2}])


def test_completed_building_is_validated_and_atomically_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    output = tmp_path / "master"
    result = tape.finalize_building_root(building_root=building, output_root=output)
    assert result["status"] == "CAUSAL_MASTER_TAPE_PUBLISHED"
    assert result["file_count"] == 11
    assert result["completed_interactions"] == 1
    assert result["dbn_files_opened"] == 0
    assert output.is_dir() and not building.exists()


def test_finalize_only_never_opens_dbn_or_runs_source_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    monkeypatch.setattr(
        tape.parent, "verify_data_preflight",
        lambda *_args, **_kwargs: pytest.fail("source preflight was called"),
    )
    monkeypatch.setattr(
        tape.native, "_stream_native_mbp10_records",
        lambda *_args, **_kwargs: pytest.fail("DBN was opened"),
    )
    result = tape.finalize_building_root(building_root=building, output_root=tmp_path / "master")
    assert result["dbn_files_opened"] == 0


def test_parallel_session_executor_is_joined_before_return(monkeypatch: pytest.MonkeyPatch):
    lifecycle: list[str] = []

    class Future:
        def result(self):
            lifecycle.append("result")
            return {"session_date": DAY}

    class Executor:
        def __init__(self, *, max_workers: int):
            assert max_workers == 2
            lifecycle.append("created")

        def submit(self, _function, _spec):
            lifecycle.append("submitted")
            return Future()

        def shutdown(self, *, wait: bool, cancel_futures: bool):
            assert wait is True and cancel_futures is False
            lifecycle.append("shutdown")

    monkeypatch.setattr(tape, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(tape, "as_completed", lambda futures: list(futures))
    spec = tape.SessionBuildSpec(DAY, "es", "mes", 100.0, 1, 2, "HARD_FLAT", "staging")
    assert tape._run_session_builds([spec], 2) == [{"session_date": DAY}]
    assert lifecycle[-1] == "shutdown"


def test_parquet_footer_and_incremental_readers_close_handles(monkeypatch: pytest.MonkeyPatch):
    import pyarrow.parquet as pq

    readers: list[object] = []

    class Batch:
        def to_pylist(self):
            return [{"value": 1}]

    class Reader:
        def __init__(self):
            self.metadata = SimpleNamespace(num_rows=1, num_row_groups=1)
            self.schema_arrow = SimpleNamespace(names=["value"])
            self.closed = False
            readers.append(self)

        def iter_batches(self, *, batch_size: int):
            assert batch_size > 0
            yield Batch()

        def close(self, *, force: bool):
            assert force is True
            self.closed = True

    monkeypatch.setattr(pq, "ParquetFile", lambda _path: Reader())
    assert tape._parquet_descriptor(Path("footer.parquet"))["rows"] == 1
    assert tape._read_parquet_rows(Path("rows.parquet")) == [{"value": 1}]
    iterator = tape._iter_parquet_rows(Path("iter.parquet"))
    assert next(iterator) == {"value": 1}
    iterator.close()
    assert len(readers) == 3 and all(reader.closed for reader in readers)


def test_finalize_destination_collision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    output = tmp_path / "master"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        tape.finalize_building_root(building_root=building, output_root=output)
    assert building.is_dir()


def test_incomplete_building_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    (building / "build-summary.json").unlink()
    with pytest.raises(tape.CausalMasterTapeError, match="missing artifact"):
        tape.validate_building_root(building)


def test_missing_session_parquet_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    (building / "causal-event-tape" / f"{DAY}.parquet").unlink()
    with pytest.raises(tape.CausalMasterTapeError, match="missing artifact"):
        tape.validate_building_root(building)


def test_corrupt_session_parquet_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    (building / "causal-event-tape" / f"{DAY}.parquet").write_bytes(b"not parquet")
    with pytest.raises(tape.CausalMasterTapeError, match="footer is unreadable"):
        tape.validate_building_root(building)


@pytest.mark.parametrize("suffix", [".part", ".tmp"])
def test_unfinished_part_or_tmp_fails_closed(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    (building / f"unfinished{suffix}").write_text("partial", encoding="utf-8")
    with pytest.raises(tape.CausalMasterTapeError, match="unfinished staging"):
        tape.validate_building_root(building)


def test_exact_interaction_population_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    building = _complete_building(tmp_path / "master.building", monkeypatch)
    monkeypatch.setattr(tape, "EXPECTED_INTERACTIONS", 2)
    summary_path = building / "build-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["completed_interactions"] = summary["indexed_interactions"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(tape.CausalMasterTapeError, match="interaction population mismatch"):
        tape.validate_building_root(building)


def test_reproduction_cli_reads_only_explicit_final_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    final = tmp_path / "final-master"
    final.mkdir()
    building = tmp_path / "final-master.building"
    building.mkdir()
    observed: list[Path] = []

    def reproduce(root: Path):
        observed.append(root)
        return {"status": "EXACT_REPRODUCTION_GATES_PASS"}

    monkeypatch.setattr(tape, "reproduction_gates", reproduce)
    assert tape.main(["reproduce", "--master-root", str(final)]) == 0
    assert observed == [final]
    assert "EXACT_REPRODUCTION_GATES_PASS" in capsys.readouterr().out
