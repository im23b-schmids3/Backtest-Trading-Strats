from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import all_period_weight_q_research as old_allp
from research_pipeline.cme_orderflow_absorption_l2_v1 import berlin_hardflat_all_period as runner
from research_pipeline.cme_orderflow_absorption_l2_v1 import berlin_hardflat_execution as execution
from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix


UTC = timezone.utc


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)


def _regular(
    ordinal: int, timestamp_ns: int, *, stream: str = "ES",
    es_bid: float | None = 100.0, es_ask: float | None = 100.25,
    mes_bid: float | None = 100.0, mes_ask: float | None = 100.25,
    es_source_ns: int | None = None, mes_source_ns: int | None = None,
) -> dict[str, object]:
    return {
        "session_date": "2026-07-13", "event_ordinal": ordinal,
        "timestamp_ns": timestamp_ns, "stream": stream,
        "event_type": f"{stream}_BBO", "book_state": "EXECUTABLE",
        "es_bid": es_bid, "es_ask": es_ask, "mes_bid": mes_bid, "mes_ask": mes_ask,
        "es_quote_timestamp_ns": timestamp_ns if es_source_ns is None else es_source_ns,
        "mes_quote_timestamp_ns": timestamp_ns if mes_source_ns is None else mes_source_ns,
        "hard_flat_reason": None,
    }


def _boundary(ordinal: int, timestamp_ns: int, state: str = "TEMPORARILY_NON_EXECUTABLE") -> dict[str, object]:
    return {
        **_regular(ordinal, timestamp_ns), "stream": "CALENDAR",
        "event_type": "BOOK_NON_EXECUTABLE", "book_state": state,
        "es_bid": None, "es_ask": None, "mes_bid": None, "mes_ask": None,
        "es_quote_timestamp_ns": None, "mes_quote_timestamp_ns": None,
    }


def _terminal(ordinal: int, timestamp_ns: int, event_type: str = "HARD_FLAT") -> dict[str, object]:
    return {
        **_regular(ordinal, timestamp_ns), "stream": "CALENDAR", "event_type": event_type,
        "hard_flat_reason": "HARD_CUTOFF_2245" if event_type == "HARD_FLAT" else "SOURCE_END",
    }


def _interaction(direction: str = "BUYER_ABSORPTION") -> dict[str, object]:
    source = "PRIOR_RTH_POC:100.00:0001"
    return {
        "interaction_id": f"2026-07-13|{source}", "source_interaction_id": source,
        "session_date": "2026-07-13", "interaction_end_ns": _ns("2026-07-13T19:59:00Z"),
        "direction": direction, "level": "PRIOR_RTH_POC",
        "zone_low": 99.0, "zone_high": 101.0,
    }


def _tape(
    middle: list[dict[str, object]] | None = None, *, terminal_time: str = "2026-07-13T22:45:00Z",
    terminal_type: str = "HARD_FLAT",
) -> execution.BerlinSessionCausalTape:
    start = _ns("2026-07-13T20:00:00Z")
    rows = [_regular(0, start), *(middle or [])]
    rows.append(_terminal(len(rows), _ns(terminal_time), terminal_type))
    return execution.BerlinSessionCausalTape("2026-07-13", rows)


def _index(entry_ordinal: int) -> dict[str, object]:
    interaction = _interaction()
    return {
        "interaction_id": interaction["interaction_id"],
        "derived_first_confirmation_timestamp_ns": _ns("2026-07-13T19:59:59Z"),
        "entry_observation_event_ordinal": entry_ordinal,
        "counterfactual_path_end_ns": _ns("2026-07-13T22:45:00Z"),
    }


def test_cest_berlin_2245_is_2045_utc():
    assert execution.berlin_hard_flat_utc("2026-07-13").isoformat() == "2026-07-13T20:45:00+00:00"


def test_cet_berlin_2245_is_2145_utc():
    assert execution.berlin_hard_flat_utc("2026-01-12").isoformat() == "2026-01-12T21:45:00+00:00"


def test_european_dst_transition_is_date_specific():
    assert execution.berlin_hard_flat_utc("2026-03-28").hour == 21
    assert execution.berlin_hard_flat_utc("2026-03-30").hour == 20


def test_us_eu_dst_mismatch_does_not_change_berlin_definition():
    assert execution.berlin_hard_flat_utc("2026-03-16").isoformat() == "2026-03-16T21:45:00+00:00"


def test_open_long_is_liquidated_by_berlin_hard_flat():
    exact = _regular(1, _ns("2026-07-13T20:45:00Z"), es_bid=101.0, es_ask=101.25)
    trade = _tape([exact]).entry_outcome(_interaction(), 0).trade
    assert trade is not None and trade["exit_reason"] == "HARD_FLAT_BERLIN"
    assert trade["exit_timestamp_ns"] == _ns("2026-07-13T20:45:00Z")


def test_long_hard_flat_uses_bid_then_normal_adverse_slippage():
    trade = _tape([_regular(1, _ns("2026-07-13T20:45:00Z"), es_bid=101.0, es_ask=101.25)]).entry_outcome(
        _interaction(), 0,
    ).trade
    assert trade is not None
    assert trade["liquidation_reference_price"] == 101.0
    assert trade["exit"] == 100.75


def test_short_hard_flat_uses_ask_then_normal_adverse_slippage():
    trade = _tape([_regular(1, _ns("2026-07-13T20:45:00Z"), es_bid=99.0, es_ask=99.25)]).entry_outcome(
        _interaction("SELLER_ABSORPTION"), 0,
    ).trade
    assert trade is not None
    assert trade["liquidation_reference_price"] == 99.25
    assert trade["exit"] == 99.5


def test_working_order_is_cancelled_at_hard_flat():
    tape = _tape()
    interaction = _interaction()
    # Due at the synthetic Berlin terminal rather than an executable event.
    index = _index(tape.contract_terminal_ordinal)
    result = execution.simulate_berlin_session(
        tape, [interaction], {str(interaction["interaction_id"]): index},
    )
    assert result.unresolved == 0
    assert result.terminal_outcomes[str(interaction["interaction_id"])] == "CANCELLED_AT_HARD_FLAT_BERLIN"


def test_entry_one_second_before_hard_flat_remains_allowed():
    hard_flat_ns = execution.berlin_hard_flat_ns("2026-07-13")
    tape = _tape([_regular(1, hard_flat_ns - 1_000_000_000)])
    outcome = tape.entry_outcome(_interaction(), 1)
    assert outcome.trade is not None
    assert outcome.trade["entry_timestamp_ns"] == hard_flat_ns - 1_000_000_000


def test_no_entry_at_or_after_hard_flat():
    tape = _tape([_regular(1, _ns("2026-07-13T20:45:00Z"))])
    outcome = tape.entry_outcome(_interaction(), 1)
    assert outcome.terminal_reason == "ENTRY_BLOCKED_AT_OR_AFTER_HARD_FLAT_BERLIN"


def test_no_position_reaches_maintenance():
    hard = execution.berlin_hard_flat_ns("2026-07-13")
    maintenance, _ = execution.maintenance_window_ns("2026-07-13")
    assert hard < maintenance
    trade = _tape([_regular(1, hard)]).entry_outcome(_interaction(), 0).trade
    assert trade is not None and int(trade["exit_timestamp_ns"]) < maintenance


@pytest.mark.parametrize("gap_ns", [100_000_000, 1_000_000_000])
def test_short_reconstruction_gap_resumes(gap_ns: int):
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _boundary(1, start + 1),
        _regular(2, start + gap_ns, es_bid=100.25, es_ask=100.5),
        _regular(3, execution.berlin_hard_flat_ns("2026-07-13"), es_bid=101.0, es_ask=101.25),
    ])
    trade = tape.entry_outcome(_interaction(), 0).trade
    assert trade is not None and trade["exit_reason"] == "HARD_FLAT_BERLIN"


def test_exactly_three_second_gap_resumes():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _boundary(1, start + 1), _regular(2, start + 3_000_000_000),
        _regular(3, execution.berlin_hard_flat_ns("2026-07-13")),
    ])
    assert tape.entry_outcome(_interaction(), 0).trade["exit_reason"] == "HARD_FLAT_BERLIN"
    assert execution.EXECUTION_CONTRACT["execution_contract"]["gap_at_exact_threshold"].startswith("RESUME")


def test_gap_over_three_seconds_force_flats_at_last_valid_bbo():
    start = _ns("2026-07-13T20:00:00Z")
    trade = _tape([
        _boundary(1, start + 1),
        _regular(2, start + 3_000_000_001, es_bid=105.0, es_ask=105.25),
    ]).entry_outcome(_interaction(), 0).trade
    assert trade is not None
    assert trade["exit_reason"] == "DATA_GAP_3S_FORCE_FLAT"
    assert trade["liquidation_reference_price"] == 100.0
    assert trade["gap_timeout_timestamp_ns"] == start + 3_000_000_000


def test_same_timestamp_reopen_is_ordered_after_boundary_not_before_it():
    start = _ns("2026-07-13T20:00:00Z")
    reopen = start + 4_000_000_000
    trade = _tape([
        _boundary(1, reopen),
        _regular(2, reopen, es_bid=105.0, es_ask=105.25),
    ]).entry_outcome(_interaction(), 0).trade
    assert trade is not None
    assert trade["exit_reason"] == "DATA_GAP_3S_FORCE_FLAT"
    assert trade["liquidation_reference_price"] == 100.0


def test_no_stop_or_target_is_invented_from_other_stream_during_gap():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _boundary(1, start + 1),
        _regular(2, start + 1_000_000_000, stream="MES", es_bid=50.0, es_ask=50.25),
        _regular(3, start + 2_000_000_000, es_bid=100.0, es_ask=100.25),
        _regular(4, execution.berlin_hard_flat_ns("2026-07-13"), es_bid=100.0, es_ask=100.25),
    ])
    assert tape.entry_outcome(_interaction(), 0).trade["exit_reason"] == "HARD_FLAT_BERLIN"


def test_source_end_forces_flat_at_last_valid_bbo():
    trade = _tape(terminal_time="2026-07-13T20:00:10Z", terminal_type="SOURCE_END").entry_outcome(
        _interaction(), 0,
    ).trade
    assert trade is not None and trade["exit_reason"] == "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO"


def test_long_source_end_uses_bid():
    trade = _tape(terminal_time="2026-07-13T20:00:10Z", terminal_type="SOURCE_END").entry_outcome(
        _interaction(), 0,
    ).trade
    assert trade["liquidation_reference_price"] == 100.0 and trade["exit"] == 99.75


def test_short_source_end_uses_ask():
    trade = _tape(terminal_time="2026-07-13T20:00:10Z", terminal_type="SOURCE_END").entry_outcome(
        _interaction("SELLER_ABSORPTION"), 0,
    ).trade
    assert trade["liquidation_reference_price"] == 100.25 and trade["exit"] == 100.5


def test_missing_defensible_entry_bbo_fails_closed(monkeypatch: pytest.MonkeyPatch):
    tape = _tape()
    monkeypatch.setattr(tape, "_entry_quote", lambda _instrument, _index: None)
    with pytest.raises(execution.UnpricedSourceIntegrityFailure, match="UNPRICED_SOURCE_INTEGRITY_FAILURE"):
        tape.entry_outcome(_interaction(), 0)


def test_locked_or_crossed_book_never_becomes_executable():
    row = _regular(0, _ns("2026-07-13T20:00:00Z"), es_bid=100.25, es_ask=100.25)
    tape = execution.BerlinSessionCausalTape(
        "2026-07-13", [row, _terminal(1, _ns("2026-07-13T22:45:00Z"))],
    )
    assert tape.entry_outcome(_interaction(), 0).terminal_reason == "WAIT_FOR_ES_EXECUTABLE_QUOTE"


def test_es_and_mes_last_valid_bbo_are_independent():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _regular(1, start + 1, stream="MES", es_bid=100.0, es_ask=100.25, mes_bid=99.5, mes_ask=99.75),
    ])
    assert tape._last_quote("ES", start + 1).bid == 100.0
    assert tape._last_quote("MES", start + 1).bid == 99.5
    assert tape._last_quote("ES", start + 1).observation_timestamp_ns == start


def test_every_entered_trade_has_one_explicit_terminal_exit():
    trade = _tape([_regular(1, execution.berlin_hard_flat_ns("2026-07-13"))]).entry_outcome(
        _interaction(), 0,
    ).trade
    assert trade is not None and trade["exit_reason"] in execution.TERMINAL_EXIT_REASONS


def test_session_reconciliation_has_zero_unresolved():
    tape = _tape([_regular(1, execution.berlin_hard_flat_ns("2026-07-13"))])
    interaction = _interaction()
    result = execution.simulate_berlin_session(
        tape, [interaction], {str(interaction["interaction_id"]): _index(0)},
    )
    assert result.unresolved == 0 and len(result.trades) == 1


def test_historical_v3_hash_and_corrected_hash_are_separate():
    assert execution.HISTORICAL_V3_CONTRACT_SHA256 == old_allp.V3_CONTRACT_SHA256
    assert execution.HISTORICAL_V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert execution.CONTRACT_SHA256 != execution.HISTORICAL_V3_CONTRACT_SHA256


def test_weight_grid_remains_exactly_23256():
    assert len(matrix.generate_weight_grid()) == 3_876
    assert matrix.QUALITY_THRESHOLDS == tuple(Decimal(value) for value in ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60"))
    assert len(matrix.configuration_registry()) == 23_256


def test_corrected_modules_have_no_databento_or_network_import():
    source = inspect.getsource(execution) + inspect.getsource(runner)
    assert "import databento" not in source.lower()
    assert "timeseries.get_range" not in source


def test_checkpoint_binding_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "EXPECTED_GRID_COUNT", 1)
    root = tmp_path / "checkpoints"
    root.mkdir()
    binding = {"period_id": "SYNTHETIC", "execution_contract_sha256": execution.CONTRACT_SHA256}
    checkpoint = root / "synthetic"
    checkpoint.mkdir()
    runner._write_parquet(checkpoint / "period-results.parquet", [{"config_id": "one"}])
    runner._write_json(checkpoint / "checkpoint.json", {
        "binding": {**binding, "execution_contract_sha256": "wrong"},
        "binding_sha256": runner._canonical_hash({**binding, "execution_contract_sha256": "wrong"}),
        "rows_sha256": runner._sha256(checkpoint / "period-results.parquet"),
    })
    with pytest.raises(runner.CorrectedAllPeriodError, match="binding mismatch"):
        runner._load_checkpoint(root, binding)


def test_checkpoint_round_trip_is_atomic_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "EXPECTED_GRID_COUNT", 1)
    root = tmp_path / "checkpoints"
    binding = {"period_id": "SYNTHETIC", "execution_contract_sha256": execution.CONTRACT_SHA256}
    rows = [{"config_id": "one", "unresolved": 0}]
    (root / "synthetic.part").mkdir(parents=True)
    (root / "synthetic.part" / "incomplete").write_text("not published", encoding="utf-8")
    runner._publish_checkpoint(root, binding, rows)
    assert runner._load_checkpoint(root, binding) == rows
    assert (root / "synthetic").is_dir()
    assert not list(root.glob("*.part"))


def test_corrected_contract_preserves_signal_constants():
    signal = execution.EXECUTION_CONTRACT["signal_contract"]
    assert signal["eligible_levels"] == ["PRIOR_RTH_POC"]
    assert signal["confirmation_favorable_ticks"] == 3
    assert signal["confirmation_horizon_seconds"] == 15
    assert signal["entry_latency_ms"] == 2
    assert execution.EXECUTION_CONTRACT["execution_contract"]["target_r"] == 3.0
    assert execution.EXECUTION_CONTRACT["execution_contract"]["fixed_risk_usd"] == 250.0


def test_sealed_contract_artifact_matches_runtime_contract():
    path = Path(
        "docs/research_pipeline/cme_orderflow_absorption_l2_v1/"
        "berlin-hardflat-execution-contract.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    documented_hash = payload.pop("contract_sha256")
    assert payload == execution.EXECUTION_CONTRACT
    assert documented_hash == execution.CONTRACT_SHA256


def test_path_classification_maps_explicit_terminal_reasons():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([_boundary(1, start + 1), _regular(2, start + 3_000_000_001)])
    overlap = {"entry_timestamp_ns": start, "entry_ordinal": 0}
    classification, reason, trade = runner._classify_corrected_path(tape, _interaction(), overlap)
    assert classification == "C_DATA_GAP_3S_FORCE_FLAT"
    assert reason == "DATA_GAP_3S_FORCE_FLAT" and trade is not None


def test_maintenance_without_preceding_hard_flat_is_fatal():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _boundary(1, execution.maintenance_window_ns("2026-07-13")[0], "MAINTENANCE"),
        _regular(2, execution.maintenance_window_ns("2026-07-13")[1]),
    ])
    # The corrected 20:45 hard flat precedes this boundary and has no native
    # observation until after maintenance; the invariant must fail loudly.
    with pytest.raises(execution.OpenPositionAtMaintenance):
        tape.entry_outcome(_interaction(), 0)


def test_earlier_data_gap_force_flat_prevents_later_maintenance_crossing():
    start = _ns("2026-07-13T20:00:00Z")
    maintenance_start, maintenance_end = execution.maintenance_window_ns("2026-07-13")
    tape = _tape([
        _boundary(1, start + 1),
        _boundary(2, maintenance_start, "MAINTENANCE"),
        _regular(3, maintenance_end),
    ])
    trade = tape.entry_outcome(_interaction(), 0).trade
    assert trade is not None
    assert trade["exit_reason"] == "DATA_GAP_3S_FORCE_FLAT"
    assert int(trade["exit_timestamp_ns"]) < maintenance_start


def test_setup_due_during_open_position_gap_is_cancelled_not_orphaned():
    start = _ns("2026-07-13T20:00:00Z")
    tape = _tape([
        _boundary(1, start + 1),
        _regular(2, start + 1_000_000_000, stream="MES"),
        _regular(3, start + 4_000_000_000),
    ])
    first = _interaction()
    second = {**first}
    second_source = "PRIOR_RTH_POC:100.00:0002"
    second["source_interaction_id"] = second_source
    second["interaction_id"] = f"2026-07-13|{second_source}"
    first_index = _index(0)
    second_index = {
        **_index(2), "interaction_id": second["interaction_id"],
    }
    result = execution.simulate_berlin_session(
        tape, [first, second], {
            str(first["interaction_id"]): first_index,
            str(second["interaction_id"]): second_index,
        },
    )
    assert len(result.trades) == 1
    assert result.trades[0]["exit_reason"] == "DATA_GAP_3S_FORCE_FLAT"
    assert result.terminal_outcomes[str(second["interaction_id"])] == "COMPLIANCE_BLOCK_ACTIVE_POSITION"
    assert result.unresolved == 0


def test_source_end_trade_records_price_source_and_decision_timestamps():
    trade = _tape(terminal_time="2026-07-13T20:00:10Z", terminal_type="SOURCE_END").entry_outcome(
        _interaction(), 0,
    ).trade
    assert trade["price_source_timestamp_ns"] == _ns("2026-07-13T20:00:00Z")
    assert trade["liquidation_decision_timestamp_ns"] == _ns("2026-07-13T20:00:10Z")


def _semantic_trade(
    setup_number: int, *, entry_ns: int, exit_ns: int, exit_reason: str = "STOP",
    entry: float = 100.0, exit_price: float = 99.0, day: str = "2026-07-13",
) -> dict[str, object]:
    source = f"PRIOR_RTH_POC:100.00:{setup_number:04d}"
    setup_id = f"L2:{source}"
    return {
        "contracts": 1, "date": day, "direction": "LONG",
        "entry": entry, "entry_timestamp_ns": entry_ns,
        "exit": exit_price, "exit_reason": exit_reason, "exit_timestamp_ns": exit_ns,
        "instrument": "ES", "interaction_id": source, "level": "PRIOR_RTH_POC",
        "net_pnl_usd": -1.0, "r_multiple": -1.0, "setup_id": setup_id,
        "stop": 99.0, "target": 103.0, "total_costs_usd": 6.0,
        "trade_id": f"L2T:{setup_id}",
    }


def _semantic_fixture(*setup_numbers: int, day: str = "2026-07-13"):
    signals: set[str] = set()
    setups: dict[str, dict[str, object]] = {}
    indexes: dict[str, dict[str, object]] = {}
    for number in setup_numbers:
        source = f"PRIOR_RTH_POC:100.00:{number:04d}"
        key = f"{day}|{source}"
        confirmation = 1_000 + number
        signals.add(key)
        setups[key] = {
            "date": day, "interaction_id": source,
            "setup_id": f"L2:{source}", "direction": "BUYER_ABSORPTION",
            "level": "PRIOR_RTH_POC", "confirmation_timestamp_ns": confirmation,
            "confirmation_price": 100.75, "terminal_reason": "ENTRY",
        }
        indexes[key] = {
            "derived_first_confirmation_timestamp_ns": confirmation,
            "derived_first_confirmation_price": 100.75,
        }
    return signals, setups, indexes


def _analyze(
    *, setup_numbers=(1,), old_trades=(), new_trades=(), old_signals=None,
    new_signals=None, corrected_terminal=None, indexes_mutator=None,
    day="2026-07-13", hard_flat_evidence_overrides=None,
):
    signals, setups, indexes = _semantic_fixture(*setup_numbers, day=day)
    if indexes_mutator is not None:
        indexes_mutator(indexes)
    return runner.analyze_semantic_period(
        period_id="SYNTHETIC",
        historical_signal_ids=signals if old_signals is None else set(old_signals),
        corrected_signal_ids=signals if new_signals is None else set(new_signals),
        historical_setups=setups,
        corrected_indexes=indexes,
        historical_trades=runner._unique_by_setup(old_trades, artifact="old"),
        corrected_trades=runner._unique_by_setup(new_trades, artifact="new"),
        corrected_terminal_outcomes=(
            {key: "TRADE_EXECUTED" for key in signals}
            if corrected_terminal is None else corrected_terminal
        ),
        hard_flat_evidence_overrides=hard_flat_evidence_overrides,
    )


def test_semantic_diff_accepts_exit_change_without_downstream_effect():
    old = _semantic_trade(1, entry_ns=100, exit_ns=300)
    new = {**old, "exit_reason": "HARD_FLAT_BERLIN", "exit_timestamp_ns": 200}
    result = _analyze(old_trades=(old,), new_trades=(new,))
    assert result["classification_counts"][
        "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT"
    ] == 1
    assert result["blocking_membership_changes"] == 0


def test_semantic_diff_accepts_earlier_corrected_exit_unblocking_later_setup():
    old_predecessor = _semantic_trade(1, entry_ns=50, exit_ns=300)
    new_predecessor = {
        **old_predecessor, "exit_reason": "DATA_GAP_3S_FORCE_FLAT", "exit_timestamp_ns": 100,
    }
    newly_unblocked = _semantic_trade(2, entry_ns=200, exit_ns=250)
    result = _analyze(
        setup_numbers=(1, 2), old_trades=(old_predecessor,),
        new_trades=(new_predecessor, newly_unblocked),
    )
    assert result["classification_counts"][
        "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT"
    ] == 1
    changed = [row for row in result["rows"] if row["setup_id"].endswith("0002")][0]
    assert changed["causal_predecessor_setup_id"].endswith("0001")


def test_semantic_diff_accepts_later_corrected_exit_blocking_later_setup():
    old_predecessor = _semantic_trade(1, entry_ns=50, exit_ns=100)
    new_predecessor = {
        **old_predecessor, "exit_reason": "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
        "exit_timestamp_ns": 300,
    }
    historically_entered = _semantic_trade(2, entry_ns=200, exit_ns=250)
    result = _analyze(
        setup_numbers=(1, 2), old_trades=(old_predecessor, historically_entered),
        new_trades=(new_predecessor,),
    )
    assert result["blocking_membership_changes"] == 1


def test_semantic_diff_rejects_signal_mutation():
    signals, setups, indexes = _semantic_fixture(1)
    result = runner.analyze_semantic_period(
        period_id="SYNTHETIC", historical_signal_ids=signals,
        corrected_signal_ids=set(), historical_setups=setups,
        corrected_indexes=indexes, historical_trades={}, corrected_trades={},
        corrected_terminal_outcomes={},
    )
    assert result["signal_differences"] == sorted(signals)
    assert result["classification_counts"][
        "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
    ] == 1


def test_semantic_diff_rejects_confirmation_mutation():
    def mutate(indexes):
        next(iter(indexes.values()))["derived_first_confirmation_price"] = 101.0

    result = _analyze(indexes_mutator=mutate)
    assert len(result["confirmation_differences"]) == 1
    assert result["classification_counts"][
        "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
    ] == 1


def test_semantic_diff_rejects_unexplained_common_entry_mutation():
    old = _semantic_trade(1, entry_ns=100, exit_ns=300)
    new = {**old, "entry": 100.25}
    result = _analyze(old_trades=(old,), new_trades=(new,))
    assert result["common_entry_mutations"] == 1
    assert result["unexpected_entry_changes"] == 1


def test_semantic_diff_requires_explicit_causal_predecessor_for_membership_change():
    old = _semantic_trade(1, entry_ns=100, exit_ns=300)
    key = "2026-07-13|PRIOR_RTH_POC:100.00:0001"
    result = _analyze(
        old_trades=(old,), new_trades=(),
        corrected_terminal={key: "CANCELLED_AT_HARD_FLAT_BERLIN"},
    )
    assert result["blocking_membership_changes"] == 0
    assert result["unexplained_membership_changes"] == 1


def test_semantic_diff_classifies_may_three_second_force_flat():
    old = _semantic_trade(1, entry_ns=100, exit_ns=300)
    new = {**old, "exit_reason": "DATA_GAP_3S_FORCE_FLAT", "exit_timestamp_ns": 200}
    result = _analyze(old_trades=(old,), new_trades=(new,))
    assert result["classification_counts"][
        "C_SAME_ENTRY_DIFFERENT_EXIT_DUE_3S_DATA_GAP"
    ] == 1


def test_semantic_diff_classifies_berlin_hard_flat():
    old = _semantic_trade(1, entry_ns=100, exit_ns=300)
    new = {**old, "exit_reason": "HARD_FLAT_BERLIN", "exit_timestamp_ns": 200}
    result = _analyze(old_trades=(old,), new_trades=(new,))
    assert result["classification_counts"][
        "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT"
    ] == 1


def _cancelled_hard_flat_membership(
    *, day: str = "2026-07-13", offset_ns: int = 0,
    evidence_overrides=None, new_signals=None, indexes_mutator=None,
):
    hard_flat_ns = execution.berlin_hard_flat_ns(day)
    old = _semantic_trade(
        1, day=day, entry_ns=hard_flat_ns + offset_ns,
        exit_ns=hard_flat_ns + offset_ns + 10_000_000_000,
    )
    key = f"{day}|PRIOR_RTH_POC:100.00:0001"
    return _analyze(
        day=day,
        old_trades=(old,),
        new_trades=(),
        new_signals=new_signals,
        corrected_terminal={key: "CANCELLED_AT_HARD_FLAT_BERLIN"},
        indexes_mutator=indexes_mutator,
        hard_flat_evidence_overrides=(
            {key: evidence_overrides} if evidence_overrides else None
        ),
    )


def test_semantic_h_one_second_before_flat_remains_disallowed():
    result = _cancelled_hard_flat_membership(offset_ns=-1_000_000_000)
    assert result["classification_counts"]["H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"] == 0
    assert result["classification_counts"]["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"] == 1


def test_semantic_h_exactly_at_flat_is_cancelled_without_fill():
    result = _cancelled_hard_flat_membership(offset_ns=0)
    row = result["rows"][0]
    assert row["classification"] == "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
    assert row["seconds_after_hard_flat"] == 0.0
    assert row["corrected_fill_occurred"] is False
    assert row["corrected_all_working_orders_cancelled"] is True


def test_semantic_h_one_second_after_flat_is_cancelled_without_predecessor():
    result = _cancelled_hard_flat_membership(offset_ns=1_000_000_000)
    row = result["rows"][0]
    assert row["classification"] == "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
    assert row["causal_predecessor_setup_id"] is None
    assert row["seconds_after_hard_flat"] == 1.0
    assert result["blocking_membership_changes"] == 0


def test_semantic_h_uses_summer_berlin_conversion():
    result = _cancelled_hard_flat_membership(day="2026-08-10")
    row = result["rows"][0]
    assert row["hard_flat_utc"] == "2026-08-10T20:45:00+00:00"
    assert row["hard_flat_local"] == "2026-08-10T22:45:00+02:00"


def test_semantic_h_uses_winter_berlin_conversion():
    result = _cancelled_hard_flat_membership(day="2026-01-28")
    row = result["rows"][0]
    assert row["hard_flat_utc"] == "2026-01-28T21:45:00+00:00"
    assert row["hard_flat_local"] == "2026-01-28T22:45:00+01:00"


def test_semantic_h_rejects_arbitrary_corrected_only_membership():
    day = "2026-07-13"
    hard_flat_ns = execution.berlin_hard_flat_ns(day)
    new = _semantic_trade(
        1, day=day, entry_ns=hard_flat_ns + 1_000_000_000,
        exit_ns=hard_flat_ns + 2_000_000_000,
    )
    result = _analyze(day=day, old_trades=(), new_trades=(new,))
    assert result["classification_counts"]["H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"] == 0
    assert result["classification_counts"]["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"] == 1


def test_semantic_h_requires_unchanged_signal_and_confirmation():
    signal_changed = _cancelled_hard_flat_membership(new_signals=set())
    assert signal_changed["classification_counts"][
        "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
    ] == 1

    def mutate(indexes):
        next(iter(indexes.values()))["derived_first_confirmation_price"] = 101.0

    confirmation_changed = _cancelled_hard_flat_membership(indexes_mutator=mutate)
    assert confirmation_changed["classification_counts"][
        "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
    ] == 1


def test_semantic_h_requires_no_corrected_position():
    result = _cancelled_hard_flat_membership(
        evidence_overrides={"corrected_position_created": True},
    )
    assert result["classification_counts"]["H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"] == 0
    assert result["classification_counts"]["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"] == 1


def test_semantic_h_requires_no_surviving_working_order():
    result = _cancelled_hard_flat_membership(
        evidence_overrides={
            "corrected_later_order_survived": True,
            "corrected_all_working_orders_cancelled": False,
        },
    )
    assert result["classification_counts"]["H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"] == 0
    assert result["classification_counts"]["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"] == 1


def test_corrected_hard_flat_invariant_detects_entry_at_boundary():
    day = "2026-07-13"
    hard_flat_ns = execution.berlin_hard_flat_ns(day)
    trade = _semantic_trade(
        1, day=day, entry_ns=hard_flat_ns,
        exit_ns=hard_flat_ns + 1_000_000_000,
    )
    invariants = runner.corrected_hard_flat_invariants(
        corrected_trades={f"{day}|{trade['interaction_id']}": trade},
        semantic_rows=(),
    )
    assert invariants["corrected_entries_at_or_after_hard_flat_count"] == 1


def test_corrected_hard_flat_invariant_detects_position_reaching_maintenance():
    day = "2026-07-13"
    hard_flat_ns = execution.berlin_hard_flat_ns(day)
    maintenance_start_ns = execution.maintenance_window_ns(day)[0]
    trade = _semantic_trade(
        1, day=day, entry_ns=hard_flat_ns - 1_000_000_000,
        exit_ns=maintenance_start_ns,
    )
    invariants = runner.corrected_hard_flat_invariants(
        corrected_trades={f"{day}|{trade['interaction_id']}": trade},
        semantic_rows=(),
    )
    assert invariants["corrected_positions_open_after_hard_flat_count"] == 1
    assert invariants["corrected_positions_reaching_maintenance_count"] == 1


def _offline_report_fixture(tmp_path: Path):
    optimizer = tmp_path / "optimizer"
    baseline = tmp_path / "baseline"
    optimizer.mkdir()
    baseline.mkdir()
    period_ids = [period.period_id for period in old_allp.PERIODS]
    source_groups = {
        period.period_id: period.source_group for period in old_allp.PERIODS
    }
    baseline_rs = [0.5, 1.0, 2.0, 0.0, 3.0, -4.0, 15.97094008334436]
    baseline_periods = []
    for index, (period_id, total_r) in enumerate(zip(period_ids, baseline_rs)):
        trades = 12 if index < 6 else 16
        wins = 4
        baseline_periods.append({
            "period_id": period_id,
            "source_group": source_groups[period_id],
            "sessions": [3, 15, 18, 4, 5, 22, 20][index],
            "completed_trades": trades,
            "wins": wins,
            "losses": trades - wins,
            "win_rate": wins / trades,
            "total_r": total_r,
            "average_r": total_r / trades,
            "net_pnl_usd": total_r * 200,
            "profit_factor": 1.1 if total_r >= 0 else 0.8,
            "max_cumulative_drawdown_r": -float(index + 1),
            "es_trades": trades // 2,
            "mes_trades": trades - trades // 2,
            "target_exits": wins,
            "stop_exits": trades - wins,
            "hard_flat_berlin_exits": 1 if period_id == "DECEMBER_2025" else 0,
            "data_gap_3s_force_flat_exits": 1 if period_id == "MAY_2026" else 0,
            "source_end_force_flat_exits": 0,
            "unresolved": 0,
        })
    runner._write_json(baseline / "summary.json", {
        "status": "CORRECTED_BERLIN_HARDFLAT_BASELINE_COMPLETE",
        "execution_contract_sha256": runner.CONTRACT_SHA256,
        "aggregate": {
            "sessions": 87, "trades": 88, "wins": 30, "losses": 58,
            "total_r": 18.47094008334436, "net_pnl_usd": 4375.5,
            "profit_factor": 1.3340522588895463,
            "max_cumulative_drawdown_r": -12.946378636204262,
            "es_trades": 52, "mes_trades": 36, "target_exits": 29,
            "stop_exits": 57, "hard_flat_berlin_exits": 1,
            "data_gap_3s_force_flat_exits": 1,
            "source_end_force_flat_exits": 0, "unresolved": 0,
            "integrity_failures": 0,
        },
        "periods": baseline_periods,
    })
    aggregate_rows = []
    period_rows = []
    neighbor_rows = []
    source_rows = []
    for config_index, config_id in enumerate(runner.COMPARISON_CONFIG_IDS):
        expected_trades, expected_r, expected_pnl = runner.EXPECTED_COMPARISON_AGGREGATES[config_id]
        weight_map = {
            "W04-02-06-04-04-Q45": (0.20, 0.10, 0.30, 0.20, 0.20, 0.45),
            "W04-02-02-04-08-Q50": (0.20, 0.10, 0.10, 0.20, 0.40, 0.50),
            "W03-02-08-03-04-Q45": (0.15, 0.10, 0.40, 0.15, 0.20, 0.45),
            "W05-01-01-05-08-Q50": (0.25, 0.05, 0.05, 0.25, 0.40, 0.50),
        }[config_id]
        aggregate_rows.append({
            "config_id": config_id,
            "G1": weight_map[0], "G2": weight_map[1], "G3": weight_map[2],
            "G4": weight_map[3], "G5": weight_map[4],
            "quality_threshold": weight_map[5],
            "total_trades": expected_trades, "wins": 30, "losses": expected_trades - 30,
            "total_r": expected_r, "total_net_pnl_usd": expected_pnl,
            "weighted_average_r": expected_r / expected_trades,
            "aggregate_profit_factor": 1.4 + config_index / 10,
            "positive_periods": 6, "negative_periods": 1, "periods_with_trades": 7,
            "median_period_r": 2.0, "worst_period_r": -1.0 + config_index / 10,
            "best_period_r": 10.0, "worst_period_max_drawdown_r": -5.0,
            "native_mbp10_total_r": expected_r * 0.6,
            "native_mbp10_trade_count": expected_trades // 2,
            "mbo_derived_total_r": expected_r * 0.4,
            "mbo_derived_trade_count": expected_trades - expected_trades // 2,
            "unresolved": 0, "es_trades": expected_trades // 2,
            "mes_trades": expected_trades - expected_trades // 2,
            "target_exits": 30, "stop_exits": expected_trades - 30,
            "cutoff_exits": 0, "hard_flat_berlin_exits": 1,
            "data_gap_3s_force_flat_exits": 1, "source_end_force_flat_exits": 0,
            "integrity_failures": 0,
            "v3_reference_trades": 90,
            "v3_reference_total_r": 16.39685541515214,
            "v3_reference_net_pnl_usd": 3903.5,
            "trade_delta_vs_v3": expected_trades - 90,
            "total_r_delta_vs_v3": expected_r - 16.39685541515214,
            "net_pnl_delta_vs_v3": expected_pnl - 3903.5,
        })
        neighbor_rows.append({
            "config_id": config_id,
            "median_neighbor_total_r": 20.0 + config_index,
            "worst_neighbor_total_r": 10.0 + config_index,
            "proportion_neighbors_aggregate_positive": 1.0,
            "proportion_neighbors_at_least_five_positive_periods": 0.8,
        })
        source_rows.append({
            "config_id": config_id,
            "native_mbp10_total_r": expected_r * 0.6,
            "mbo_derived_total_r": expected_r * 0.4,
        })
        for period_index, period_id in enumerate(period_ids):
            total_r = expected_r / 7 if period_index != 5 else -0.5
            period_rows.append({
                "period_id": period_id, "source_group": source_groups[period_id],
                "sessions": [3, 15, 18, 4, 5, 22, 20][period_index],
                "config_id": config_id,
                "G1": weight_map[0], "G2": weight_map[1], "G3": weight_map[2],
                "G4": weight_map[3], "G5": weight_map[4],
                "quality_threshold": weight_map[5],
                "trades": 10, "wins": 4, "losses": 6, "win_rate": 0.4,
                "total_r": total_r, "average_r": total_r / 10,
                "net_pnl_usd": total_r * 200, "profit_factor": 1.2,
                "max_cumulative_drawdown_r": -2.0,
                "es_trades": 5, "mes_trades": 5, "target_exits": 4,
                "stop_exits": 6, "hard_flat_berlin_exits": 0,
                "data_gap_3s_force_flat_exits": 0,
                "source_end_force_flat_exits": 0, "unresolved": 0,
            })
    runner._write_csv(optimizer / "weight-q-aggregate-results.csv", aggregate_rows)
    runner._write_parquet(optimizer / "weight-q-period-results.parquet", period_rows)
    runner._write_csv(optimizer / "neighbor-robustness.csv", neighbor_rows)
    runner._write_csv(optimizer / "source-model-robustness.csv", source_rows)
    runner._write_json(optimizer / "plateau-analysis.json", {"status": "EXISTING"})
    runner._write_json(optimizer / "summary.json", {
        "status": "CORRECTED_ALL_PERIOD_WEIGHT_Q_RESEARCH_COMPLETE_NO_SELECTION",
        "execution_contract_sha256": runner.CONTRACT_SHA256,
        "configuration_count": 4, "network_calls": 0, "downloads": 0,
        "dbn_files_opened": 0,
    })
    return optimizer, baseline


def test_offline_reporting_requires_corrected_baseline_hash(tmp_path):
    _optimizer, baseline = _offline_report_fixture(tmp_path)
    payload = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    payload["execution_contract_sha256"] = "0" * 64
    runner._write_json(baseline / "summary.json", payload)
    with pytest.raises(runner.CorrectedAllPeriodError, match="hash mismatch"):
        runner.load_corrected_berlin_baseline(tmp_path, baseline)


def test_offline_reporting_uses_berlin_reference_and_preserves_raw_values(tmp_path):
    optimizer, baseline = _offline_report_fixture(tmp_path)
    before = runner._read_csv(optimizer / "weight-q-aggregate-results.csv")
    before_digest = runner._strategy_value_digest(before)
    result = runner.run_offline_reporting(
        repository_root=tmp_path, optimizer_root=optimizer, baseline_root=baseline,
    )
    after = runner._read_csv(optimizer / "weight-q-aggregate-results.csv")
    assert runner._strategy_value_digest(after) == before_digest
    selected = next(row for row in after if row["config_id"] == "W04-02-02-04-08-Q50")
    assert "v3_reference_total_r" not in selected
    assert float(selected["historical_v3_reference_total_r"]) == 16.39685541515214
    assert float(selected["berlin_v3_reference_total_r"]) == 18.47094008334436
    assert float(selected["total_r_delta_vs_berlin_v3"]) == pytest.approx(
        26.721885295430262 - 18.47094008334436
    )
    assert result["aggregate_strategy_values_changed"] is False


def test_offline_reporting_reads_existing_parquet_only_and_never_replays(tmp_path, monkeypatch):
    optimizer, baseline = _offline_report_fixture(tmp_path)
    parquet_calls = []
    original_read = runner._read_parquet

    def tracked(path):
        parquet_calls.append(Path(path))
        return original_read(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("market-data/replay path invoked")

    monkeypatch.setattr(runner, "_read_parquet", tracked)
    monkeypatch.setattr(runner.allp, "_period_bundles", forbidden)
    monkeypatch.setattr(runner.BerlinSessionCausalTape, "from_parquet", forbidden)
    result = runner.run_offline_reporting(
        repository_root=tmp_path, optimizer_root=optimizer, baseline_root=baseline,
    )
    assert parquet_calls == [optimizer / "weight-q-period-results.parquet"]
    assert result["candidate_period_rows"] == 40
    assert result["dbn_files_opened"] == result["network_calls"] == result["downloads"] == 0
    summaries = runner._read_csv(optimizer / "candidate-summary.csv")
    assert [row["config_id"] for row in summaries] == [
        runner.BERLIN_BASELINE_CONFIG_ID, *runner.COMPARISON_CONFIG_IDS,
    ]


def test_offline_reporting_missing_candidate_fails_closed(tmp_path):
    optimizer, baseline = _offline_report_fixture(tmp_path)
    rows = runner._read_csv(optimizer / "weight-q-aggregate-results.csv")[:-1]
    runner._write_csv(optimizer / "weight-q-aggregate-results.csv", rows)
    summary = runner._read_json(optimizer / "summary.json")
    summary["configuration_count"] = len(rows)
    runner._write_json(optimizer / "summary.json", summary)
    with pytest.raises(runner.CorrectedAllPeriodError, match="candidate missing"):
        runner.run_offline_reporting(
            repository_root=tmp_path, optimizer_root=optimizer, baseline_root=baseline,
        )


def test_offline_reporting_missing_period_fails_closed(tmp_path):
    optimizer, baseline = _offline_report_fixture(tmp_path)
    rows = runner._read_parquet(optimizer / "weight-q-period-results.parquet")
    target = runner.COMPARISON_CONFIG_IDS[0]
    removed = False
    kept = []
    for row in rows:
        if not removed and row["config_id"] == target:
            removed = True
            continue
        kept.append(row)
    runner._write_parquet(optimizer / "weight-q-period-results.parquet", kept)
    with pytest.raises(runner.CorrectedAllPeriodError, match="period rows missing"):
        runner.run_offline_reporting(
            repository_root=tmp_path, optimizer_root=optimizer, baseline_root=baseline,
        )
