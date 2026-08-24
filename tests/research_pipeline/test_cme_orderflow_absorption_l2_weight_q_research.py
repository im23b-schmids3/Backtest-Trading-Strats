from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import causal_master_tape as master
from research_pipeline.cme_orderflow_absorption_l2_v1 import weight_q_research as matrix


DAY = "2025-12-01"


def _interaction(identifier: str, *, score: float = 0.60, penalty: float = 0.0) -> dict[str, object]:
    source = f"PRIOR_RTH_POC:100.00:{identifier}"
    return {
        "interaction_id": f"{DAY}|{source}", "source_interaction_id": source,
        "session_date": DAY, "interaction_start_ns": 1, "interaction_end_ns": 2,
        "direction": "BUYER_ABSORPTION", "level": "PRIOR_RTH_POC",
        "level_price": 100.0, "interaction_end_price": 100.0,
        "zone_low": 99.0, "zone_high": 100.0,
        "aggression_score": score, "restoration_score": score,
        "price_resistance_score": score, "persistence_score": score,
        "multi_level_support_score": score, "false_refill_penalty": penalty,
        "non_quality_rejection_reasons": "",
    }


def _event(
    ordinal: int, timestamp: int, *, stream: str = "ES", event_type: str = "ES_BBO",
    es_bid: float | None = 100.0, es_ask: float | None = 100.25,
    mes_bid: float | None = 100.0, mes_ask: float | None = 100.25,
    execution_price: float | None = None, execution_aggressor: str | None = None,
) -> dict[str, object]:
    return {
        "session_date": DAY, "event_ordinal": ordinal, "timestamp_ns": timestamp,
        "stream": stream, "stream_priority": master.STREAM_PRIORITY[stream],
        "source_index": ordinal + 1, "event_type": event_type,
        "es_bid": es_bid, "es_ask": es_ask, "mes_bid": mes_bid, "mes_ask": mes_ask,
        "execution_price": execution_price,
        "execution_size": 1 if execution_price is not None else None,
        "execution_aggressor": execution_aggressor,
        "book_state": "EXECUTABLE", "hard_flat_reason": None,
        "es_quote_timestamp_ns": timestamp if es_bid is not None else None,
        "mes_quote_timestamp_ns": timestamp if mes_bid is not None else None,
    }


def _hard(ordinal: int, timestamp: int) -> dict[str, object]:
    row = _event(ordinal, timestamp, stream="CALENDAR", event_type="HARD_FLAT")
    row["hard_flat_reason"] = "HARD_FLAT_SCHEDULED_CLOSE"
    return row


def _index(row: dict[str, object], due_ordinal: int = 0) -> dict[str, object]:
    return {
        "interaction_id": row["interaction_id"], "session_date": DAY,
        "derived_first_confirmation_timestamp_ns": 5,
        "derived_first_confirmation_price": 100.75,
        "entry_ready_ns": 7, "entry_observation_event_ordinal": due_ordinal,
    }


def _result_row(
    units: tuple[int, int, int, int, int], q: str, *, trades: int = 20,
    total_r: float = 2.0, pf: float = 1.2, drawdown: float = -3.0,
) -> dict[str, object]:
    return {
        "config_id": matrix.config_id(units, q),
        **{f"G{index}": value * 0.05 for index, value in enumerate(units, start=1)},
        "quality_threshold": float(q), "trades": trades, "total_r": total_r,
        "profit_factor": pf, "max_cumulative_drawdown_r": drawdown,
    }


def test_weight_grid_has_exact_frozen_cardinality_and_constraints():
    rows = matrix.generate_weight_grid()
    assert len(rows) == 3_876
    assert len(set(rows)) == 3_876
    assert all(sum(row) == 20 and min(row) >= 1 for row in rows)


def test_quality_registry_has_six_thresholds_and_23256_unique_configurations():
    assert matrix.QUALITY_THRESHOLDS == tuple(
        matrix.Decimal(value) for value in ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60")
    )
    registry = matrix.configuration_registry()
    assert len(registry) == 23_256
    assert len({matrix.config_id(weights, q) for weights, q in registry}) == 23_256


def test_score_recomputation_preserves_frozen_false_refill_penalty():
    row = _interaction("SCORE", score=0.60, penalty=0.20)
    units = (4, 4, 4, 4, 4)
    expected = 0.60 - 0.20 * master.V2_CONFIG.false_refill_penalty_weight
    assert matrix.recompute_grid_score(row, units) == pytest.approx(expected)
    assert matrix.recompute_grid_score(row, units) == pytest.approx(
        master.recompute_quality(row, matrix.unit_weights(units))
    )


def test_threshold_application_is_inclusive_and_keeps_primitive_rejection():
    row = _interaction("THRESHOLD", score=0.50)
    assert master.interaction_is_accepted(row, threshold="0.50", weights=matrix.unit_weights((4, 4, 4, 4, 4)))
    row["non_quality_rejection_reasons"] = "NO_GENUINE_CONSUME_RESTORE"
    assert not master.interaction_is_accepted(row, threshold="0.35", weights=matrix.unit_weights((4, 4, 4, 4, 4)))


def test_vectorized_threshold_boundary_is_resolved_with_exact_frozen_decimal_semantics():
    row = _interaction("BOUNDARY", score=0.50)
    mask = matrix._accepted_mask(
        [row], np.asarray([True]), np.asarray([0.50 - 5e-13]),
        (4, 4, 4, 4, 4), matrix.Decimal("0.50"),
    )
    assert mask.tolist() == [True]


def test_independent_portfolio_chronology_and_one_position_blocking():
    first, second = _interaction("A"), _interaction("B")
    indexes = {str(first["interaction_id"]): _index(first), str(second["interaction_id"]): _index(second)}
    # The first setup remains open on ordinal 1, so the second setup is blocked
    # there; the first then reaches its target on ordinal 2.
    tape = matrix.SessionCausalTape(DAY, [
        _event(0, 10), _event(1, 11, es_bid=100.25, es_ask=100.50),
        _event(2, 12, es_bid=109.0, es_ask=109.25), _hard(3, 13),
    ])
    together = matrix.simulate_independent_session(tape, [first, second], indexes)
    second_only = matrix.simulate_independent_session(
        tape, [second], {str(second["interaction_id"]): indexes[str(second["interaction_id"])]},
    )
    assert len(together.trades) == 1
    assert together.active_position_blocks == 1
    assert together.trades[0]["interaction_id"] == first["source_interaction_id"]
    assert len(second_only.trades) == 1
    assert second_only.active_position_blocks == 0
    assert second_only.trades[0]["interaction_id"] == second["source_interaction_id"]


def test_later_same_event_setup_is_deferred_and_may_enter_after_prior_exit():
    first, second = _interaction("A"), _interaction("B")
    indexes = {str(first["interaction_id"]): _index(first), str(second["interaction_id"]): _index(second)}
    # The first trade exits before entry attempts on ordinal 1, allowing the
    # still-confirmed second setup to enter on that exact event.
    tape = matrix.SessionCausalTape(DAY, [
        _event(0, 10), _event(1, 11, es_bid=109.0, es_ask=109.25), _hard(2, 12),
    ])
    result = matrix.simulate_independent_session(tape, [first, second], indexes)
    assert len(result.trades) == 2
    assert result.active_position_blocks == 0
    assert result.trades[0]["exit_reason"] == "TARGET"
    assert result.trades[1]["entry_timestamp_ns"] == 11


def test_compact_session_matches_canonical_event_replay_on_synthetic_fixture(tmp_path: Path):
    end = 1_000_000_000
    row = _interaction("EQUIVALENCE")
    row["interaction_end_ns"] = end
    confirmation = end + 5_000_000_000
    events = [
        _event(0, confirmation, event_type="ES_EXECUTION", execution_price=100.75, execution_aggressor="BUY"),
        _event(1, confirmation + 2_000_000, es_bid=100.0, es_ask=100.25),
        _event(2, confirmation + 3_000_000, es_bid=109.0, es_ask=109.25),
        _hard(3, confirmation + 4_000_000),
    ]
    root = tmp_path / "master"
    (root / "causal-event-tape").mkdir(parents=True)
    master._write_small_parquet(root / "interaction-master.parquet", [row])
    master._write_small_parquet(root / "causal-event-tape" / f"{DAY}.parquet", events)
    (root / "calendar.json").write_text(json.dumps({"target_sessions": [DAY]}), encoding="utf-8")
    canonical = master.replay_configuration(root, threshold="0.50", session_prefix="2025-12")
    compact_index = _index(row, due_ordinal=1)
    compact_index["derived_first_confirmation_timestamp_ns"] = confirmation
    compact = matrix.simulate_independent_session(
        matrix.SessionCausalTape(DAY, events), [row], {str(row["interaction_id"]): compact_index},
    )
    assert len(canonical["trades"]) == len(compact.trades) == 1
    fields = (
        "trade_id", "entry_timestamp_ns", "entry", "stop", "target",
        "exit_timestamp_ns", "exit", "exit_reason", "instrument", "contracts",
        "net_pnl_usd", "r_multiple",
    )
    assert {name: canonical["trades"][0][name] for name in fields} == {
        name: compact.trades[0][name] for name in fields
    }


def test_weight_quality_and_combined_neighbors_are_exact():
    units = (4, 4, 4, 4, 4)
    weight = matrix.weight_neighbors(units)
    quality = matrix.quality_neighbors("0.50")
    combined = matrix.combined_neighbor_ids(units, "0.50")
    assert len(weight) == 20
    assert quality == (matrix.Decimal("0.45"), matrix.Decimal("0.55"))
    assert len(combined) == 22
    assert all(sum(row) == 20 and min(row) >= 1 for row in weight)


def test_plateau_clustering_and_isolated_peak_are_descriptive_only():
    a, b, isolated = (4, 4, 4, 4, 4), (3, 5, 4, 4, 4), (1, 1, 1, 1, 16)
    rows = [
        _result_row(a, "0.50", total_r=4.0, pf=1.4),
        _result_row(b, "0.50", total_r=3.0, pf=1.3),
        _result_row(isolated, "0.60", total_r=20.0, pf=2.0),
    ]
    robustness = [
        {"config_id": rows[0]["config_id"], "proportion_neighbors_profitable": 0.8,
         "median_neighbor_total_r": 2.0, "worst_neighbor_total_r": 1.0},
        {"config_id": rows[1]["config_id"], "proportion_neighbors_profitable": 0.8,
         "median_neighbor_total_r": 2.0, "worst_neighbor_total_r": 1.0},
        {"config_id": rows[2]["config_id"], "proportion_neighbors_profitable": 0.2,
         "median_neighbor_total_r": -1.0, "worst_neighbor_total_r": -3.0},
    ]
    payload, representatives, peaks = matrix.plateau_analysis(rows, robustness)
    assert payload["connected_plateau_count"] == 1
    assert payload["selected_configuration"] is None
    assert payload["automatic_v5_selection"] is False
    assert representatives[0]["plateau_configuration_count"] == 2
    assert [row["config_id"] for row in peaks] == [rows[2]["config_id"]]


def test_ranked_tables_apply_fifteen_trade_guard():
    eligible = _result_row((4, 4, 4, 4, 4), "0.50", trades=15, total_r=2.0)
    excluded = _result_row((3, 5, 4, 4, 4), "0.50", trades=14, total_r=99.0)
    robustness = [
        {"config_id": row["config_id"], "median_neighbor_total_r": 1.0,
         "worst_neighbor_total_r": 0.0, "proportion_neighbors_profitable": 1.0}
        for row in (eligible, excluded)
    ]
    tables = matrix._ranked_tables([eligible, excluded], robustness)
    assert [row["config_id"] for row in tables["top-total-r.csv"]] == [eligible["config_id"]]


def test_december_parquet_filter_never_exposes_january(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"session_date": "2025-12-31", "value": 1},
        {"session_date": "2026-01-02", "value": 999},
    ]), path)
    assert matrix._load_filtered_parquet(path, ("2025-12-31",)) == [
        {"session_date": "2025-12-31", "value": 1}
    ]


def test_v3_december_reproduction_gate_fails_closed():
    payload = {
        "metrics": deepcopy(master.EXPECTED_GATES["V3_DECEMBER"]),
        "dbn_files_opened": 0, "network_calls": 0,
    }
    matrix._assert_v3_gate(payload)
    payload["metrics"]["completed_trades"] = 37
    with pytest.raises(master.CausalMasterTapeError, match="V3_DECEMBER reproduction mismatch"):
        matrix._assert_v3_gate(payload)


def test_offline_contract_and_no_automatic_v5_selection(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("source/network path must not be called")

    monkeypatch.setattr(master.native, "_stream_native_mbp10_records", forbidden)
    monkeypatch.setattr(master.historical, "_stream_mes_quotes", forbidden)
    row = _interaction("OFFLINE")
    tape = matrix.SessionCausalTape(DAY, [
        _event(0, 10), _event(1, 11, es_bid=109.0, es_ask=109.25), _hard(2, 12),
    ])
    result = matrix.simulate_independent_session(
        tape, [row], {str(row["interaction_id"]): _index(row)},
    )
    assert len(result.trades) == 1
    assert matrix.NO_AUTOMATIC_V5_SELECTION is True
    payload, _representatives, _peaks = matrix.plateau_analysis([], [])
    assert payload["automatic_v5_selection"] is False
    assert payload["selected_configuration"] is None
    # The implementation's final contract is explicit and contains no source
    # acquisition entry point; DBN/network counters are immutable zeros.
    assert matrix.STRATEGY_ID.endswith("WEIGHT_Q_RESEARCH_DEC2025")
    assert not hasattr(matrix, "download")
