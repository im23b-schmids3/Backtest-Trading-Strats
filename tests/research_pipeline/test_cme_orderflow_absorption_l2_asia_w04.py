from __future__ import annotations

import csv
import inspect
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import asia_w04_replay as asia
from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import historical_runner as historical
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix
from research_pipeline.cme_orderflow_absorption_l2_v1.model import (
    ENTRY_LATENCY_NS,
    EXIT_RESET_NS,
    INACTIVITY_NS,
    MAX_CONFIRMATION_NS,
    MIN_CONFIRMATION_NS,
    STOP_BUFFER_TICKS,
    TARGET_R,
    TICK,
    Execution,
    initial_prices,
    size_for_instrument,
)
from research_pipeline.cme_orderflow_absorption_l2_v1.v2_quality050 import V2_CONFIG


ELIGIBLE_DATES = (
    "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
    "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30",
    "2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08",
    "2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
    "2026-07-16", "2026-07-17", "2026-07-20", "2026-07-21", "2026-07-22",
    "2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29",
    "2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05",
    "2026-08-06", "2026-08-07",
)


def _period(day: str) -> str:
    if day.startswith("2026-05"):
        return "MAY_2026_MBO_DERIVED"
    if day <= "2026-07-17":
        return "RETRO_JUNE_JULY_2026_MBO_DERIVED"
    if day <= "2026-07-31":
        return "JULY_20_31_PILOT_MBO"
    return "AUGUST_03_07_SHARED_MBO"


def _audit_fixture(root: Path) -> Path:
    dates = ["2026-05-04", *ELIGIBLE_DATES[:14], "2026-06-23", *ELIGIBLE_DATES[14:]]
    prior: str | None = None
    rows = []
    for day in dates:
        missing = day in {"2026-05-04", "2026-06-23"}
        rows.append({
            "session_date": day,
            "period": _period(day),
            "source_model": "MBO_DERIVED_MBP10",
            "prior_asia_session": "" if missing else prior,
            "classification": asia.PROFILE_ONLY_CLASSIFICATION if missing else asia.ELIGIBLE_CLASSIFICATION,
        })
        prior = day
    # The June source starts a new contiguous audited block.
    rows_by_day = {row["session_date"]: row for row in rows}
    rows_by_day["2026-06-24"]["prior_asia_session"] = "2026-06-23"
    audit = root / "audit"
    audit.mkdir()
    with (audit / "session-coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (audit / "summary.json").write_text(json.dumps({
        "audit_id": "CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT",
        "market_data_opened": False,
        "strategy_replay_executed": False,
        "candidate_session_universe": {
            "missing_prior_asia_poc_dates": ["2026-05-04", "2026-06-23"],
            "es_and_prior_poc_present_but_mes_missing_dates": list(ELIGIBLE_DATES),
        },
    }), encoding="utf-8")
    return audit


def _event(
    ordinal: int,
    timestamp_ns: int,
    *,
    bid: float = 100.0,
    ask: float = 100.25,
    hard: bool = False,
) -> dict[str, object]:
    if hard:
        timestamp_ns = asia._clock_ns("2026-01-01", asia.ASIA_END_SECONDS)
    return {
        "session_date": "2026-01-01",
        "event_ordinal": ordinal,
        "timestamp_ns": timestamp_ns,
        "stream": "CALENDAR" if hard else "ES",
        "stream_priority": 2 if hard else 1,
        "source_index": 0 if hard else ordinal + 1,
        "event_type": "HARD_FLAT" if hard else "ES_BBO",
        "es_bid": bid,
        "es_ask": ask,
        "mes_bid": None,
        "mes_ask": None,
        "execution_price": None,
        "execution_size": None,
        "execution_aggressor": None,
        "book_state": "EXECUTABLE",
        "entry_probe_count": 0,
        "es_quote_timestamp_ns": timestamp_ns,
        "mes_quote_timestamp_ns": None,
        "hard_flat_reason": asia.ASIA_HARD_FLAT_REASON if hard else None,
    }


def _interaction(identifier: str, direction: str, zone_low: float, zone_high: float) -> dict[str, object]:
    return {
        "interaction_id": identifier,
        "source_interaction_id": identifier.rsplit("|", 1)[-1],
        "session_date": "2026-01-01",
        "direction": direction,
        "level": asia.ASIA_LEVEL_NAME,
        "zone_low": zone_low,
        "zone_high": zone_high,
    }


def test_audit_fixture_yields_exact_46_sessions_and_excludes_profile_only_days(tmp_path: Path) -> None:
    sessions = asia.load_audit_sessions(_audit_fixture(tmp_path))
    assert len(sessions) == 48
    eligible = [row.day for row in sessions if row.eligible]
    assert tuple(eligible) == ELIGIBLE_DATES
    assert "2026-05-04" not in eligible
    assert "2026-06-23" not in eligible


def test_asia_window_is_exactly_half_open_utc() -> None:
    day = "2026-05-05"
    assert asia.in_asia_window(day, asia._clock_ns(day, 0))
    assert asia.in_asia_window(day, asia._clock_ns(day, 8 * 3600) - 1)
    assert not asia.in_asia_window(day, asia._clock_ns(day, 8 * 3600))


def test_prior_asia_poc_uses_size_weighting_and_lower_tick_tie_break() -> None:
    assert asia.prior_asia_poc({40_000: 9, 40_001: 10}) == 10_000.25
    assert asia.prior_asia_poc({40_000: 10, 40_001: 10}) == 10_000.00
    with pytest.raises(asia.AsiaReplayError, match="no ES executions"):
        asia.prior_asia_poc({})


def test_only_prior_asia_poc_is_accepted() -> None:
    assert asia.AsiaStructuralLevel(asia.ASIA_LEVEL_NAME, 5_000.25).name == asia.ASIA_LEVEL_NAME
    with pytest.raises(asia.AsiaReplayError, match="only PRIOR_ASIA_SESSION_POC"):
        asia.AsiaStructuralLevel("PRIOR_RTH_POC", 5_000.25)


def test_w04_weights_quality_and_primitive_contract_are_frozen() -> None:
    assert tuple(asia.W04_WEIGHTS.values()) == tuple(map(Decimal, ("0.20", "0.10", "0.30", "0.20", "0.20")))
    assert asia.QUALITY_THRESHOLD == Decimal("0.45")
    excluded = {
        "min_quality_score", "aggression_weight", "restoration_weight",
        "price_resistance_weight", "persistence_weight", "multi_level_support_weight",
        "weights_label",
    }
    assert {key: value for key, value in asdict(asia.W04_CONFIG).items() if key not in excluded} == {
        key: value for key, value in asdict(V2_CONFIG).items() if key not in excluded
    }


def test_interaction_and_execution_constants_remain_frozen() -> None:
    assert TICK == 0.25
    assert INACTIVITY_NS == 60_000_000_000
    assert EXIT_RESET_NS == 1_000_000_000
    assert MIN_CONFIRMATION_NS == 5_000_000_000
    assert MAX_CONFIRMATION_NS == 15_000_000_000
    assert ENTRY_LATENCY_NS == 2_000_000
    assert STOP_BUFFER_TICKS == 5
    assert TARGET_R == 3.0


def test_confirmation_uses_first_qualifying_es_execution_without_early_invalidation() -> None:
    tracker = master.CausalWindowTracker()
    tracker.register({
        "interaction_id": "i", "interaction_end_ns": 0,
        "interaction_end_price": 100.0, "direction": "BUYER_ABSORPTION",
    })
    assert tracker.observe_es_execution(Execution(1_000_000_000, 90.0, 1, "SELL")) == []
    assert tracker.observe_es_execution(Execution(5_000_000_000, 100.50, 1, "BUY")) == []
    assert tracker.observe_es_execution(Execution(6_000_000_000, 100.75, 1, "BUY")) == ["i"]
    window = tracker.windows["i"]
    assert window.confirmation_timestamp_ns == 6_000_000_000
    assert window.confirmation_price == 100.75
    assert window.entry_ready_ns == 6_002_000_000
    assert tracker.observe_es_execution(Execution(7_000_000_000, 101.0, 1, "BUY")) == []


def test_confirmation_inclusive_at_15_seconds_and_entry_probe_after_two_ms() -> None:
    tracker = master.CausalWindowTracker()
    tracker.register({
        "interaction_id": "i", "interaction_end_ns": 0,
        "interaction_end_price": 100.0, "direction": "SELLER_ABSORPTION",
    })
    assert tracker.observe_es_execution(Execution(15_000_000_000, 99.25, 1, "SELL")) == ["i"]
    assert tracker.due_entry_probes(15_001_999_999) == []
    assert tracker.due_entry_probes(15_002_000_000) == ["i"]


@pytest.mark.parametrize(
    ("direction", "expected_stop", "expected_target"),
    (("BUYER_ABSORPTION", 98.75, 104.75), ("SELLER_ABSORPTION", 101.25, 94.25)),
)
def test_long_short_stop_and_three_r_geometry(direction: str, expected_stop: float, expected_target: float) -> None:
    prices = initial_prices(direction, 99.75, 100.0, 100.0, 100.0)
    assert prices["stop"] == expected_stop
    assert prices["target"] == expected_target


def test_es_is_preferred_when_one_contract_fits() -> None:
    prices = initial_prices("BUYER_ABSORPTION", 99.75, 100.0, 99.75, 100.0)
    sizing = size_for_instrument(prices, "ES")
    assert sizing["contracts"] >= 1
    assert sizing["contracts"] <= 6
    assert sizing["estimated_initial_risk_usd"] <= 250.0


def test_apex_caps_remain_six_es_and_sixty_mes() -> None:
    prices = {"entry": 100.0, "stop_exit": 99.75}
    assert size_for_instrument(prices, "ES")["contracts"] <= 6
    assert size_for_instrument(prices, "MES")["contracts"] <= 60


def test_mes_proxy_uses_es_price_path_when_es_does_not_fit() -> None:
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000),
        _event(1, 2_000_000_000, bid=101.0, ask=101.25, hard=True),
    ])
    outcome = tape.entry_outcome(
        _interaction("2026-01-01|wide", "BUYER_ABSORPTION", 95.0, 100.0), 0,
    )
    assert outcome.trade is not None
    trade = outcome.trade
    assert trade["instrument"] == "MES_PROXY_FROM_ES"
    assert trade["source_instrument"] == "ES"
    assert trade["entry"] == 100.50
    assert trade["contracts"] <= 60
    assert trade["estimated_initial_risk_usd"] <= 250.0


def test_es_native_source_label_is_explicit_when_es_fits() -> None:
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000),
        _event(1, 2_000_000_000, hard=True),
    ])
    outcome = tape.entry_outcome(
        _interaction("2026-01-01|tight", "SELLER_ABSORPTION", 100.0, 100.25), 0,
    )
    assert outcome.trade is not None
    assert outcome.trade["instrument"] == "ES_NATIVE_SOURCE"
    assert outcome.trade["source_instrument"] == "ES"


def test_proxy_tape_requires_no_native_mes_quotes() -> None:
    rows = [_event(0, 1_000_000_000), _event(1, 2_000_000_000, hard=True)]
    assert all(row["mes_bid"] is None and row["mes_ask"] is None for row in rows)
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", rows)
    assert list(tape.mes_bid) == list(tape.es_bid)
    assert list(tape.mes_ask) == list(tape.es_ask)


def test_one_active_position_and_terminal_reconciliation_are_inherited() -> None:
    rows = [
        _event(0, 1_000_000_000),
        _event(1, 1_500_000_000),
        _event(2, 2_000_000_000, hard=True),
    ]
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", rows)
    first = _interaction("2026-01-01|a", "BUYER_ABSORPTION", 99.75, 100.0)
    second = _interaction("2026-01-01|b", "BUYER_ABSORPTION", 99.75, 100.0)
    indexes = {
        "2026-01-01|a": {
            "derived_first_confirmation_timestamp_ns": 500_000_000,
            "entry_observation_event_ordinal": 0,
        },
        "2026-01-01|b": {
            "derived_first_confirmation_timestamp_ns": 500_000_000,
            "entry_observation_event_ordinal": 0,
        },
    }
    result = matrix.simulate_independent_session(tape, [first, second], indexes)
    assert len(result.terminal_outcomes) == 2
    assert sum(value == "TRADE_EXECUTED" for value in result.terminal_outcomes.values()) == 1
    assert sum(value == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for value in result.terminal_outcomes.values()) == 1
    assert len(result.trades) == 1


def test_semantic_diff_contains_only_the_six_predeclared_fields() -> None:
    document = asia.semantic_diff_document()
    assert document["status"] == "PASS"
    assert set(document["differences"]) == asia.ALLOWED_SEMANTIC_DIFFERENCES
    assert document["unexpected_difference_fields"] == []


def test_reports_use_the_exact_sealed_disclaimer() -> None:
    summary = {
        "status": "ASIA_W04_REPLAY_COMPLETE",
        "eligible_session_count": 0,
        "raw_interactions": 0,
        "accepted_setups": 0,
        "confirmations_passed": 0,
        "confirmations_failed": 0,
        "unresolved": 0,
        "performance": {
            "completed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "net_pnl_usd": 0.0,
            "profit_factor": None,
            "max_cumulative_drawdown_r": 0.0,
        },
        "execution_model_counts": {},
        "risk_budget_audit": {"pass": True, "violation_count": 0, "maximum_usd": None},
        "reconciliation": {"pass": True},
        "semantic_diff_status": "PASS",
        "ny_artifacts_mutated": False,
        "network_calls": 0,
        "downloads": 0,
    }
    assert asia.DISCLAIMER in asia._report_markdown(summary)
    assert asia.DISCLAIMER in asia._report_html(summary)


def test_performance_counts_explicit_es_native_and_mes_proxy_labels() -> None:
    common = {
        "exit_timestamp_ns": 1,
        "r_multiple": -1.0,
        "net_pnl_usd": -10.0,
        "exit_reason": "STOP",
    }
    performance = asia._performance([
        {**common, "trade_id": "es", "instrument": "ES_NATIVE_SOURCE", "execution_model": "ES_NATIVE_SOURCE"},
        {**common, "trade_id": "mes", "instrument": "MES_PROXY_FROM_ES", "execution_model": "MES_PROXY_FROM_ES"},
    ])
    assert performance["es_trades"] == 1
    assert performance["mes_trades"] == 1


def test_new_output_root_is_isolated_from_ny_research() -> None:
    assert "ASIA" in str(asia.OUTPUT_ROOT)
    assert "BERLIN_HARDFLAT" not in str(asia.OUTPUT_ROOT)


def test_existing_output_collision_fails_before_source_processing(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable Asia output root"):
        asia.run_replay(repository_root=tmp_path, output_root=output, audit_root=tmp_path / "audit")


def test_module_has_no_network_or_download_api_path() -> None:
    source = inspect.getsource(asia)
    assert "metadata.get_cost" not in source
    assert "timeseries.get_range" not in source
    assert "Historical(" not in source
    assert "DATABENTO_API_KEY" not in source


def test_trade_identifiers_are_deterministic_and_globally_session_scoped() -> None:
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000), _event(1, 2_000_000_000, hard=True),
    ])
    interaction = _interaction("2026-01-01|stable", "BUYER_ABSORPTION", 99.75, 100.0)
    one = tape.entry_outcome(interaction, 0).trade
    two = tape.entry_outcome(interaction, 0).trade
    assert one is not None and two is not None
    assert one["setup_id"] == two["setup_id"] == "ASIA:2026-01-01|stable"
    assert one["trade_id"] == two["trade_id"] == "ASIA_T:2026-01-01|stable"


def test_hard_flat_uses_explicit_asia_reason_and_source_bbo() -> None:
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000, bid=100.0, ask=100.25),
        _event(1, asia._clock_ns("2026-01-01", asia.ASIA_END_SECONDS) - 1, bid=100.5, ask=100.75),
        _event(2, 2_000_000_000, bid=100.5, ask=100.75, hard=True),
    ])
    outcome = tape.entry_outcome(
        _interaction("2026-01-01|flat", "BUYER_ABSORPTION", 99.75, 100.0), 0,
    )
    assert outcome.trade is not None
    assert outcome.trade["exit_reason"] == asia.ASIA_HARD_FLAT_REASON
    assert outcome.trade["exit"] == 100.25  # hard-flat bid minus one adverse ES tick


def test_risk_audit_detects_no_budget_overrun_for_valid_sizing() -> None:
    audit = asia._risk_audit([
        {"trade_id": "a", "estimated_initial_risk_usd": 249.0},
        {"trade_id": "b", "estimated_initial_risk_usd": 250.0},
    ])
    assert audit["pass"] is True
    assert audit["violation_count"] == 0
    assert audit["maximum_usd"] == 250.0


def test_risk_audit_fails_closed_on_overrun() -> None:
    audit = asia._risk_audit([{"trade_id": "bad", "estimated_initial_risk_usd": 250.01}])
    assert audit["pass"] is False
    assert audit["violation_trade_ids"] == ["bad"]


def test_profile_only_session_builds_poc_without_creating_strategy_state(tmp_path: Path) -> None:
    day = "2026-05-04"
    base = asia._clock_ns(day, 0)
    spec = asia.SessionSpec(
        day, "MAY_2026_MBO_DERIVED", "MBO_DERIVED_MBP10", None, False,
        base, asia._clock_ns(day, asia.ASIA_END_SECONDS), tmp_path / "source.dbn", tmp_path,
    )
    state = asia._AsiaMboState(spec, None)
    state.observe(historical.PrivateMBORecord(base, "R", "B", 5_000.0, 0, 0, historical.F_SNAPSHOT))
    state.observe(historical.PrivateMBORecord(base + 1, "A", "B", 5_000.0, 100, 1, historical.F_SNAPSHOT))
    state.observe(historical.PrivateMBORecord(
        base + 2, "A", "A", 5_000.25, 100, 2,
        historical.F_SNAPSHOT | historical.F_LAST,
    ))
    state.observe(historical.PrivateMBORecord(base + 1_000_000_000, "T", "B", 5_000.0, 7, 1))
    state.observe(historical.PrivateMBORecord(spec.cutoff_ns, "T", "B", 5_000.0, 99, 1))
    result = state.finish()
    assert result["session_poc"] == 5_000.0
    assert result["profile_execution_count"] == 1
    assert result["profile_execution_volume"] == 7
    assert result["raw_interactions"] == 0
    assert result["trades"] == []


@pytest.mark.parametrize(
    "generic_reason",
    sorted(asia._GENERIC_HARD_FLAT_UNRESOLVED),
)
def test_complete_asia_hard_flat_classifies_late_entry_as_cutoff(generic_reason: str) -> None:
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000),
        _event(1, 2_000_000_000, hard=True),
    ])
    result = matrix.SessionResult(
        accepted_setups=1,
        confirmations=1,
        unresolved=1,
        terminal_outcomes={"late": generic_reason},
    )
    asia._classify_asia_entry_cutoff(result, tape=tape)
    assert result.unresolved == 0
    assert result.terminal_outcomes == {"late": asia.ASIA_ENTRY_CUTOFF_REASON}
    assert result.other_terminal == {asia.ASIA_ENTRY_CUTOFF_REASON: 1}


def test_source_incomplete_unresolved_is_not_reclassified_as_asia_cutoff() -> None:
    source_end = _event(1, 2_000_000_000)
    source_end["event_type"] = "SOURCE_END"
    tape = matrix.SessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000), source_end,
    ])
    result = matrix.SessionResult(
        accepted_setups=1,
        confirmations=1,
        unresolved=1,
        terminal_outcomes={"late": "UNRESOLVED_NO_LATER_EVENT"},
    )
    asia._classify_asia_entry_cutoff(result, tape=tape)
    assert result.unresolved == 1
    assert result.terminal_outcomes == {"late": "UNRESOLVED_NO_LATER_EVENT"}
    assert result.other_terminal == {}


def test_corrected_w04_gap_contract_force_flats_after_three_seconds() -> None:
    boundary = _event(1, 2_000_000_000)
    boundary.update({
        "event_type": "BOOK_NON_EXECUTABLE",
        "book_state": "TEMPORARILY_NON_EXECUTABLE",
    })
    tape = asia.AsiaProxySessionCausalTape("2026-01-01", [
        _event(0, 1_000_000_000),
        boundary,
        _event(2, 6_000_000_000, bid=100.0, ask=100.25),
        _event(3, 7_000_000_000, hard=True),
    ])
    outcome = tape.entry_outcome(
        _interaction("2026-01-01|gap", "BUYER_ABSORPTION", 95.0, 100.0), 0,
    )
    assert outcome.trade is not None
    assert outcome.trade["exit_reason"] == "DATA_GAP_3S_FORCE_FLAT"
    assert outcome.open_until_source_end is False
