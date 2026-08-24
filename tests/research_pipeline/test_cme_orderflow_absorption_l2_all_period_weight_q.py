from __future__ import annotations

import json
import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import all_period_weight_q_research as research
from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import durable_boundary_reconciliation as boundary_audit
from research_pipeline.cme_orderflow_absorption_l2_v1 import mbo_v3_reproduction_reconciliation as reconciliation
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix


def _period_row(
    period_id: str, *, total_r: float, trades: int = 10, pnl: float = 100.0,
    source_group: str | None = None, unresolved: int = 0,
) -> dict[str, object]:
    group = source_group or research.SOURCE_GROUP[period_id]
    return {
        "period_id": period_id, "source_group": group, "config_id": "W04-04-04-04-04-Q50",
        "G1": 0.2, "G2": 0.2, "G3": 0.2, "G4": 0.2, "G5": 0.2,
        "quality_threshold": 0.5, "trades": trades, "wins": trades // 2,
        "losses": trades - trades // 2, "total_r": total_r, "net_pnl_usd": pnl,
        "gross_profit_usd": max(pnl, 0.0) + 200.0, "gross_loss_usd": 200.0,
        "profit_factor": (max(pnl, 0.0) + 200.0) / 200.0,
        "max_cumulative_drawdown_r": -abs(total_r) / 2,
        "unresolved": unresolved, "es_trades": trades, "mes_trades": 0,
        "target_exits": trades // 2, "stop_exits": trades - trades // 2,
        "hard_cutoff_exits": 0,
    }


def _event(ordinal: int, timestamp: int, event_type: str = "ES_BBO") -> dict[str, object]:
    return {
        "session_date": "2026-07-13", "event_ordinal": ordinal, "timestamp_ns": timestamp,
        "stream": "CALENDAR" if event_type == "SOURCE_END" else "ES",
        "event_type": event_type, "book_state": "EXECUTABLE",
        "es_bid": 100.0, "es_ask": 100.25, "mes_bid": 100.0, "mes_ask": 100.25,
        "es_quote_timestamp_ns": timestamp, "mes_quote_timestamp_ns": timestamp,
        "hard_flat_reason": "SOURCE_END_INCOMPLETE_NO_FORCED_EXIT" if event_type == "SOURCE_END" else None,
    }


def _interaction() -> dict[str, object]:
    source = "PRIOR_RTH_POC:100.00:0001"
    return {
        "interaction_id": f"2026-07-13|{source}", "source_interaction_id": source,
        "session_date": "2026-07-13", "interaction_end_ns": 1,
        "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC",
        "zone_low": 99.0, "zone_high": 100.0,
    }


def _boundary_event(ordinal: int, timestamp: int, state: str) -> dict[str, object]:
    return {
        **_event(ordinal, timestamp, "BOOK_NON_EXECUTABLE"),
        "stream": "CALENDAR", "book_state": state,
        "es_bid": None, "es_ask": None, "mes_bid": None, "mes_ask": None,
        "es_quote_timestamp_ns": None, "mes_quote_timestamp_ns": None,
    }


def _hard_event(ordinal: int, timestamp: int) -> dict[str, object]:
    return {
        **_event(ordinal, timestamp, "HARD_FLAT"),
        "stream": "CALENDAR", "hard_flat_reason": "HARD_CUTOFF_2245",
    }


def _confirmed_index(entry_ordinal: int = 0) -> dict[str, object]:
    row = _interaction()
    return {
        "interaction_id": row["interaction_id"],
        "derived_first_confirmation_timestamp_ns": 5,
        "entry_observation_event_ordinal": entry_ordinal,
        "counterfactual_path_end_ns": 100,
    }


def test_inventory_declares_exactly_seven_periods_87_sessions_and_source_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    master_root = tmp_path / research.DEC_JAN_MASTER
    master_root.mkdir(parents=True)
    for name in ("interaction-master.parquet", "interaction-event-index.parquet"):
        (master_root / name).write_bytes(b"sealed")
    december = [f"2025-12-{day:02d}" for day in range(1, 23)]
    january = [f"2026-01-{day:02d}" for day in range(2, 22)]
    (master_root / "calendar.json").write_text(
        json.dumps({"target_sessions": december + january}), encoding="utf-8",
    )
    source_rows = [{"period_id": period.period_id, "status": "READY_FROM_EXISTING_LOCAL_DATA", "missing_paths": []}
                   for period in research.PERIODS]
    monkeypatch.setattr(research.cross, "build_source_inventory", lambda _root: {"periods": source_rows})
    inventory = research.build_source_inventory(tmp_path, tmp_path / "tapes")
    assert inventory["period_count"] == 7
    assert inventory["session_count"] == 87
    assert sum(row["heavy_local_pass_required"] for row in inventory["periods"]) == 5
    assert {row["source_group"] for row in inventory["periods"]} == {"NATIVE_MBP10", "MBO_DERIVED"}
    assert [row["period_id"] for row in inventory["periods"] if row["compact_tape_reusable"]] == [
        "DECEMBER_2025", "JANUARY_2026",
    ]


def test_grid_and_quality_registry_remain_exact():
    assert len(matrix.generate_weight_grid()) == 3_876
    assert matrix.QUALITY_THRESHOLDS == tuple(Decimal(value) for value in ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60"))
    assert len(matrix.configuration_registry()) == 23_256


def test_all_seven_exact_v3_reproduction_gates_and_aggregate():
    rows = []
    sessions = {"APRIL_2026": 3, "MAY_2026": 15, "RETRO_JUNE_JULY_2026": 18,
                "AUGUST_03_06_2026": 4, "AUGUST_10_14_2026": 5,
                "DECEMBER_2025": 22, "JANUARY_2026": 20}
    for period_id, expected in research.EXPECTED_V3.items():
        rows.append({"period_id": period_id, "sessions": sessions[period_id], **expected})
    gates = research.assert_reproduction_gates(rows)
    assert gates["status"] == "ALL_SEVEN_V3_REPRODUCTION_GATES_PASS"
    assert gates["aggregate"] == research.EXPECTED_V3_AGGREGATE
    rows[0]["total_r"] = float(rows[0]["total_r"]) + 0.01
    with pytest.raises(research.AllPeriodResearchError, match="V3_REPRODUCTION_GATE_FAILED:APRIL_2026"):
        research.assert_reproduction_gates(rows)


def _documented_baseline() -> dict[str, object]:
    document = {
        "status": "MBO_V3_REPRODUCTION_RECONCILIATION_COMPLETE",
        "classification": research.REPRODUCTION_BASELINE_LABEL,
        "v3_contract_sha256": research.V3_CONTRACT_SHA256,
        "v3_parameter_changes": False, "weight_changes": False,
        "quality_threshold_changes": False, "optimizer_executed": False,
        "weight_q_configurations_evaluated": 0, "network_calls": 0, "downloads": 0,
        "optimizer_gate_update_required": True,
        "optimizer_permitted_after_documented_gate_validation": True,
        "baseline_updates": {
            "MAY_2026": {
                "label": research.REPRODUCTION_BASELINE_LABEL,
                "old": dict(research.OLD_MBO_V3_BASELINES["MAY_2026"]),
                "corrected": dict(research.EXPECTED_V3["MAY_2026"]),
            }
        },
        "periods": {},
        "native_periods": {},
    }
    for period_id in research.MBO_REPRODUCTION_PERIOD_IDS:
        document["periods"][period_id] = {
            "canonical_decision": (
                "SUPERSEDED_BY_MBO_PUBLIC_BOOK_INTEGRITY_CLARIFICATION"
                if period_id == "MAY_2026" else "OLD_PUBLISHED_BASELINE_REMAINS_EXACT"
            ),
            "corrected_semantics": {
                "sessions": research.EXPECTED_V3_SESSIONS[period_id],
                **research.EXPECTED_V3[period_id],
                **research.EXPECTED_MBO_REPRODUCTION_DETAILS[period_id],
            },
        }
    for period_id in research.NATIVE_REPRODUCTION_PERIOD_IDS:
        document["native_periods"][period_id] = {
            "status": "UNCHANGED",
            "published": {
                "sessions": research.EXPECTED_V3_SESSIONS[period_id],
                **research.EXPECTED_V3[period_id],
            },
        }
    canonical = {
        "classification": document["classification"],
        "v3_contract_sha256": document["v3_contract_sha256"],
        "baseline_updates": document["baseline_updates"],
    }
    document["canonical_baseline_sha256"] = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    assert document["canonical_baseline_sha256"] == research.EXPECTED_REPRODUCTION_BASELINE_SHA256
    return document


def test_corrected_baseline_requires_documented_source_integrity_scope_and_no_optimizer():
    document = _documented_baseline()
    assert research.validate_corrected_baseline_document(document)["classification"] == research.REPRODUCTION_BASELINE_LABEL
    for mutation, match in (
        (("classification", "UNLABELED"), "CLASSIFICATION"),
        (("optimizer_executed", True), "OPTIMIZER_CONTAMINATION"),
        (("network_calls", 1), "NETWORK_CONTAMINATION"),
        (("v3_contract_sha256", "0" * 64), "STRATEGY_HASH"),
        (("canonical_baseline_sha256", "0" * 64), "CANONICAL_HASH"),
    ):
        invalid = {**document, mutation[0]: mutation[1]}
        with pytest.raises(research.AllPeriodResearchError, match=match):
            research.validate_corrected_baseline_document(invalid)

    missing_hash = dict(document)
    missing_hash.pop("canonical_baseline_sha256")
    with pytest.raises(research.AllPeriodResearchError, match="CANONICAL_HASH"):
        research.validate_corrected_baseline_document(missing_hash)


def test_corrected_baseline_cannot_silently_update_another_period_or_metric():
    document = _documented_baseline()
    document["baseline_updates"] = {
        **document["baseline_updates"],
        "RETRO_JUNE_JULY_2026": {"label": research.REPRODUCTION_BASELINE_LABEL},
    }
    with pytest.raises(research.AllPeriodResearchError, match="SCOPE_INVALID"):
        research.validate_corrected_baseline_document(document)
    metric_tamper = _documented_baseline()
    metric_tamper["baseline_updates"]["MAY_2026"]["corrected"]["trades"] = 6
    with pytest.raises(research.AllPeriodResearchError, match="NEW_METRIC_INVALID:trades"):
        research.validate_corrected_baseline_document(metric_tamper)


def test_old_may_baseline_is_rejected_and_corrected_may_is_selected():
    corrected = _documented_baseline()
    rows = research.reproduction_rows_from_baseline_document(corrected)
    may = next(row for row in rows if row["period_id"] == "MAY_2026")
    assert may["trades"] == 5
    assert may["total_r"] == pytest.approx(1.9545454545454546)
    assert may["net_pnl_usd"] == pytest.approx(467.5)
    assert may["gate_source"] == "HASH_BOUND_CORRECTED_MBO_RECONCILIATION"

    stale = _documented_baseline()
    stale["periods"]["MAY_2026"]["corrected_semantics"].update(
        research.OLD_MBO_V3_BASELINES["MAY_2026"],
    )
    with pytest.raises(research.AllPeriodResearchError, match="METRIC_INVALID:MAY_2026:trades"):
        research.validate_corrected_baseline_document(stale)


@pytest.mark.parametrize("period_id", ("RETRO_JUNE_JULY_2026", "AUGUST_03_06_2026"))
def test_unchanged_mbo_period_gate_is_hash_bound_and_exact(period_id: str):
    document = _documented_baseline()
    document["periods"][period_id]["corrected_semantics"]["total_r"] += 0.01
    with pytest.raises(research.AllPeriodResearchError, match=f"METRIC_INVALID:{period_id}:total_r"):
        research.validate_corrected_baseline_document(document)


@pytest.mark.parametrize("period_id", research.NATIVE_REPRODUCTION_PERIOD_IDS)
def test_native_period_gates_remain_unchanged(period_id: str):
    document = _documented_baseline()
    document["native_periods"][period_id]["published"]["trades"] += 1
    with pytest.raises(research.AllPeriodResearchError, match=f"METRIC_INVALID:{period_id}:trades"):
        research.validate_corrected_baseline_document(document)


def test_canonical_artifact_rows_reconcile_to_90_trade_aggregate():
    rows = research.reproduction_rows_from_baseline_document(_documented_baseline())
    result = research.assert_reproduction_gates(rows)
    assert result["aggregate_gate"]["status"] == "PASS"
    assert result["aggregate"] == research.EXPECTED_V3_AGGREGATE
    assert result["aggregate"]["trades"] == 90


def test_preflight_writes_report_without_grid_or_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    document = _documented_baseline()
    monkeypatch.setattr(research, "load_corrected_baseline_document", lambda _root: document)
    report_root = tmp_path / "preflight"
    result = research.run_reproduction_preflight(repository_root=tmp_path, report_root=report_root)
    assert result["optimizer_permitted"] is True
    assert result["configuration_count_evaluated"] == 0
    assert result["network_calls"] == result["downloads"] == 0
    assert (report_root / "preflight-summary.json").is_file()
    assert (report_root / "preflight-report.html").is_file()


def test_optimizer_cannot_start_before_reproduction_preflight_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    reached_inventory = False

    def fail_preflight(*, repository_root: Path) -> dict[str, object]:
        raise research.AllPeriodResearchError("V3_REPRODUCTION_GATE_FAILED:MAY_2026")

    def inventory_was_reached(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal reached_inventory
        reached_inventory = True
        return {}

    monkeypatch.setattr(research, "require_reproduction_preflight", fail_preflight)
    monkeypatch.setattr(research, "build_source_inventory", inventory_was_reached)
    with pytest.raises(research.AllPeriodResearchError, match="V3_REPRODUCTION_GATE_FAILED:MAY_2026"):
        research.run_optimizer(
            repository_root=tmp_path,
            tape_root=tmp_path / "tapes",
            output_root=tmp_path / "output",
        )
    assert reached_inventory is False


def test_all_period_preflight_source_has_no_provider_or_download_api():
    source = Path(research.__file__).read_text(encoding="utf-8")
    assert "metadata.get_cost" not in source
    assert "timeseries.get_range" not in source


def test_old_vs_corrected_trade_comparison_preserves_common_trades_and_isolates_old_only():
    shared = {
        "date": "2026-05-07", "interaction_id": "I1", "entry_timestamp_ns": 10,
        "instrument": "MES", "entry": 100.0, "stop": 101.0, "target": 97.5,
        "exit_timestamp_ns": 20, "exit": 97.25, "exit_reason": "TARGET",
        "r_multiple": 2.5, "net_pnl_usd": 600.0,
    }
    removed = {**shared, "date": "2026-05-13", "interaction_id": reconciliation.MAY_DIVERGENCE_ID}
    result = reconciliation.compare_trades([shared, removed], [shared])
    assert result["common_trade_count"] == result["unchanged_common_trade_count"] == 1
    assert len(result["old_only_trades"]) == 1
    assert result["corrected_only_trades"] == []


def test_corrected_public_book_suppresses_locked_atomic_state_and_reopens_fresh():
    from research_pipeline.cme_orderflow_absorption_l2_v1.historical_runner import (
        F_LAST,
        F_SNAPSHOT,
        HistoricalMBOToMBP10Adapter,
        PrivateMBORecord,
    )

    adapter = HistoricalMBOToMBP10Adapter()
    adapter.feed(PrivateMBORecord(1, "R", "B", 99.75, 0, 0, F_SNAPSHOT, 99_750_000_000))
    adapter.feed(PrivateMBORecord(1, "A", "B", 99.75, 2, 1, F_SNAPSHOT, 99_750_000_000))
    adapter.feed(PrivateMBORecord(1, "A", "A", 100.25, 2, 4, F_SNAPSHOT, 100_250_000_000))
    opened = adapter.feed(PrivateMBORecord(
        1, "A", "A", 100.0, 2, 2, F_SNAPSHOT | F_LAST, 100_000_000_000,
    ))
    assert opened is not None and adapter.state == "EXECUTABLE"
    locked = adapter.feed(PrivateMBORecord(2, "A", "B", 100.0, 1, 3, 0, 100_000_000_000))
    assert locked is None and adapter.state == "TEMPORARILY_NON_EXECUTABLE"
    reopened = adapter.feed(PrivateMBORecord(2, "C", "A", 100.0, 2, 2, 128, 100_000_000_000))
    assert reopened is not None and adapter.state == "EXECUTABLE"
    assert reopened.update is None
    assert reopened.snapshot.bids[0].price < reopened.snapshot.asks[0].price


def test_source_end_keeps_open_position_unresolved_instead_of_inventing_hard_flat():
    row = _interaction()
    tape = matrix.SessionCausalTape("2026-07-13", [_event(0, 10), _event(1, 20, "SOURCE_END")])
    index = {
        "interaction_id": row["interaction_id"], "derived_first_confirmation_timestamp_ns": 5,
        "entry_observation_event_ordinal": 0, "counterfactual_path_end_ns": 20,
    }
    result = matrix.simulate_independent_session(tape, [row], {str(row["interaction_id"]): index})
    assert result.trades == []
    assert result.unresolved == 1
    assert result.terminal_outcomes[row["interaction_id"]] == "UNRESOLVED_SOURCE_END"


@pytest.mark.parametrize(
    ("post_reopen_bid", "expected_resolution"),
    ((109.0, "TARGET"), (97.5, "STOP")),
)
def test_open_position_at_expected_maintenance_is_fail_closed_even_with_later_path(
    post_reopen_bid: float, expected_resolution: str,
):
    row = _interaction()
    events = [
        _event(0, 10),
        _boundary_event(1, 20, "MAINTENANCE"),
        {**_event(2, 30), "es_bid": 100.0, "es_ask": 100.25},
        {**_event(3, 40), "es_bid": post_reopen_bid, "es_ask": post_reopen_bid + 0.25},
        _hard_event(4, 50),
    ]
    tape = matrix.SessionCausalTape("2026-07-13", events)
    plan = boundary_audit._entry_plan(tape, row, 0)
    assert plan is not None
    assert boundary_audit._diagnostic_exit_without_boundary(tape, plan)["resolution"] == expected_resolution
    result = matrix.simulate_independent_session(
        tape, [row], {str(row["interaction_id"]): _confirmed_index()},
    )
    assert result.trades == [] and result.unresolved == 1
    assert result.terminal_outcomes[row["interaction_id"]] == (
        "POSITION_UNRESOLVED_EXPECTED_SCHEDULED_MAINTENANCE"
    )


def test_temporary_reconstruction_with_open_position_is_typed_and_fail_closed():
    row = _interaction()
    tape = matrix.SessionCausalTape("2026-07-13", [
        _event(0, 10),
        _boundary_event(1, 20, "TEMPORARILY_NON_EXECUTABLE"),
        {**_event(2, 30), "es_bid": 109.0, "es_ask": 109.25},
        _hard_event(3, 40),
    ])
    result = matrix.simulate_independent_session(
        tape, [row], {str(row["interaction_id"]): _confirmed_index()},
    )
    assert result.trades == [] and result.unresolved == 1
    assert result.terminal_outcomes[row["interaction_id"]] == (
        "POSITION_UNRESOLVED_TEMPORARY_BOOK_RECONSTRUCTION"
    )


def test_temporary_reconstruction_before_entry_preserves_confirmation_and_uses_fresh_quote():
    row = _interaction()
    tape = matrix.SessionCausalTape("2026-07-13", [
        _event(0, 10),
        _boundary_event(1, 20, "TEMPORARILY_NON_EXECUTABLE"),
        _event(2, 30),
        {**_event(3, 40), "es_bid": 109.0, "es_ask": 109.25},
        _hard_event(4, 50),
    ])
    result = matrix.simulate_independent_session(
        tape, [row], {str(row["interaction_id"]): _confirmed_index(2)},
    )
    assert len(result.trades) == 1
    assert result.trades[0]["entry_timestamp_ns"] == 30


def test_waiting_for_reopen_boundary_is_not_silently_resumed_with_open_position():
    row = _interaction()
    tape = matrix.SessionCausalTape("2026-07-13", [
        _event(0, 10),
        _boundary_event(1, 20, "WAITING_FOR_REOPEN_BOOK"),
        {**_event(2, 30), "es_bid": 109.0, "es_ask": 109.25},
        _hard_event(3, 40),
    ])
    result = matrix.simulate_independent_session(
        tape, [row], {str(row["interaction_id"]): _confirmed_index()},
    )
    assert result.unresolved == 1
    assert result.terminal_outcomes[row["interaction_id"]] == (
        "POSITION_UNRESOLVED_UNRESOLVED_INVALID_BOOK"
    )


def test_boundary_reopen_evidence_is_instrument_specific_and_never_uses_stale_bbo():
    tape = matrix.SessionCausalTape("2026-07-13", [
        _event(0, 10),
        _boundary_event(1, 20, "MAINTENANCE"),
        {**_event(2, 30), "mes_bid": None, "mes_ask": None},
        {**_event(3, 40), "stream": "MES", "es_bid": 100.0, "es_ask": 100.25,
         "mes_bid": 99.75, "mes_ask": 100.0},
        _hard_event(4, 50),
    ])
    boundary = tape.non_executable_boundaries[0]
    assert boundary.es_reopen_ordinal == 2
    assert boundary.mes_reopen_ordinal == 3
    assert boundary.es_reopen_timestamp_ns == 30
    assert boundary.mes_reopen_timestamp_ns == 40


def test_unknown_non_executable_state_rejects_tape_instead_of_weakening_integrity():
    with pytest.raises(matrix.WeightQResearchError, match="unsupported non-executable tape state"):
        matrix.SessionCausalTape("2026-07-13", [
            _event(0, 10), _boundary_event(1, 20, "UNKNOWN_STATE"), _hard_event(2, 30),
        ])


def test_boundary_diagnostic_does_not_change_strategy_grid_or_select_a_configuration():
    assert research.V3_CONTRACT_SHA256 == "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
    assert len(matrix.configuration_registry()) == 23_256
    assert boundary_audit.FROZEN_POSITION_POLICY["EXPECTED_SCHEDULED_MAINTENANCE"] == (
        "FAIL_CLOSED_IF_POSITION_OPEN"
    )
    source = Path(boundary_audit.__file__).read_text(encoding="utf-8")
    assert "timeseries.get_range" not in source
    assert "automatic_strategy_selection" not in source


def test_period_portfolio_state_is_not_concatenated():
    first = matrix.ConfigurationAccumulator((4, 4, 4, 4, 4), Decimal("0.50"))
    second = matrix.ConfigurationAccumulator((4, 4, 4, 4, 4), Decimal("0.50"))
    first.equity_r = first.peak_r = 12.0
    assert second.equity_r == 0.0
    assert second.peak_r == 0.0
    assert first is not second


def test_aggregate_metrics_positive_periods_worst_period_and_source_split():
    values = {
        "APRIL_2026": 1.0, "MAY_2026": -2.0, "RETRO_JUNE_JULY_2026": 3.0,
        "AUGUST_03_06_2026": -1.0, "AUGUST_10_14_2026": 4.0,
        "DECEMBER_2025": 5.0, "JANUARY_2026": 6.0,
    }
    aggregate = research.aggregate_configuration_periods([
        _period_row(period.period_id, total_r=values[period.period_id]) for period in research.PERIODS
    ])
    assert aggregate["total_trades"] == 70
    assert aggregate["total_r"] == 16.0
    assert aggregate["positive_periods"] == 5
    assert aggregate["negative_periods"] == 2
    assert aggregate["worst_period_r"] == -2.0
    assert aggregate["native_mbp10_total_r"] == 16.0
    assert aggregate["mbo_derived_total_r"] == 0.0
    assert aggregate["guard_60"] is True
    assert aggregate["trade_delta_vs_v3"] == 70 - research.EXPECTED_V3_AGGREGATE["trades"]
    assert aggregate["total_r_delta_vs_v3"] == pytest.approx(16.0 - research.EXPECTED_V3_AGGREGATE["total_r"])


def _aggregate_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for units, q in matrix.configuration_registry():
        total_r = sum(units[index] * (index + 1) for index in range(5)) / 20.0 - float(q)
        rows.append({
            "config_id": matrix.config_id(units, q),
            **{f"G{index}": value * 0.05 for index, value in enumerate(units, start=1)},
            "quality_threshold": float(q), "total_r": total_r,
            "positive_periods": 5, "worst_period_r": -1.0,
            "worst_period_max_drawdown_r": -2.0, "guard_60": True,
        })
    return rows


def test_neighbor_robustness_uses_weight_transfer_and_adjacent_quality():
    rows = _aggregate_grid()
    robustness = research.build_neighbor_robustness(rows)
    central = next(row for row in robustness if row["config_id"] == matrix.config_id((4, 4, 4, 4, 4), "0.50"))
    assert central["weight_neighbor_count"] == 20
    assert central["quality_neighbor_count"] == 2
    assert central["combined_neighbor_count"] == 22
    assert 0.0 <= central["proportion_neighbors_aggregate_positive"] <= 1.0


def test_plateau_guards_are_descriptive_and_never_select_strategy():
    rows = _aggregate_grid()
    neighbors = research.build_neighbor_robustness(rows)
    result = research.plateau_analysis(rows, neighbors)
    assert result["automatic_strategy_selection"] is False
    assert result["selected_configuration"] is None
    assert set(result["strict_views"]) == {"A", "B", "C"}


def test_module_has_no_download_or_databento_entry_point_and_no_auto_selection():
    assert not hasattr(research, "download")
    assert not hasattr(research, "databento")
    assert research.NO_AUTOMATIC_SELECTION is True
    assert research.EXPECTED_HEAVY_PERIOD_PASSES == 5
    assert research.EXPECTED_CONFIGURATION_COUNT == 23_256
