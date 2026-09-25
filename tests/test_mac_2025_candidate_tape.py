from pathlib import Path

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape import (
    CandidateTape,
    EventSpool,
    EVENT_DTYPE,
    FEATURE_NAMES,
    TAPE_VERSION,
    build_candidate_tape,
    classify_parameters,
    compare_trade_ledgers,
    evaluate_candidate_tape,
    evaluate_weight_q_matrix,
    load_tape,
    write_tape,
)
from research_pipeline.cme_orderflow_absorption_l2_v1.model import L2Config


def _candidate() -> dict:
    row = {
        "candidate_id": "ASIA|POC:0001",
        "interaction_id": "ASIA|POC:0001",
        "date": "2025-03-03",
        "level": "ASIA|ASIA|PRIOR|POC",
        "direction": "BUYER_ABSORPTION",
        "interaction_end_ns": 1_000_000_000,
        "interaction_end_price": 100.0,
        "zone_low": 99.5,
        "zone_high": 100.0,
        "accepted": True,
    }
    for name in (
        "directional_aggressive_volume", "relevant_execution_count", "consume_restore_cycles",
        "maximum_through_level_progress_ticks", "interaction_rejection_ticks",
    ):
        row[name] = {"directional_aggressive_volume": 100, "relevant_execution_count": 3,
                     "consume_restore_cycles": 2, "maximum_through_level_progress_ticks": 1.0,
                     "interaction_rejection_ticks": 1.0}[name]
    for name in (
        "aggression_score", "restoration_score", "price_resistance_score",
        "persistence_score", "multi_level_support_score", "false_refill_penalty",
    ):
        row[name] = 0.8 if name != "false_refill_penalty" else 0.0
    return row


def _tape() -> CandidateTape:
    events = np.zeros(5, dtype=EVENT_DTYPE)
    events[:] = [
        (1_000_000_000, 100.0, 100.25, np.nan, 0, 0, 0),
        (6_000_000_000, 100.0, 100.25, 100.75, 10, -1, 0),
        (7_000_000_000, 100.0, 100.25, np.nan, 0, 0, 0),
        (8_000_000_000, 98.0, 98.25, np.nan, 0, 0, 0),
        (9_000_000_000, 98.0, 98.25, np.nan, 0, 0, 0),
    ]
    return CandidateTape({
        "tape_version": TAPE_VERSION, "date": "2025-03-03", "event_count": len(events),
        "feature_names": list(FEATURE_NAMES), "source_sha256": "source", "semantic_sha256": "semantic",
        "bbo_path_complete": True,
    }, (_candidate(),), events)


def test_offline_evaluator_does_not_need_dbn_and_reconstructs_trade_path() -> None:
    result = evaluate_candidate_tape(_tape())
    assert result["dbn_required"] is False
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit_reason"] == "STOP"


def test_confirmation_does_not_cross_runner_session_boundary() -> None:
    row = _candidate()
    row["trading_session"] = "EUROPE"
    events = _tape().events.copy()
    events[1]["session"] = 2  # next NY runner, not EUROPE
    tape = CandidateTape(_tape().metadata, (row,), events)
    assert evaluate_candidate_tape(tape)["trades"] == []


def test_entry_after_confirmation_deadline_is_rejected() -> None:
    row = _candidate()
    row["trading_session"] = "ASIA"
    row["interaction_end_ns"] = 1_000_000_000
    events = np.zeros(4, dtype=EVENT_DTYPE)
    events[:] = [
        (1_000_000_000, 99.0, 99.25, np.nan, 0, 0, 0),
        (15_999_000_000, 99.0, 99.25, 104.0, 1, 1, 0),
        (16_001_000_000, 99.0, 99.25, 104.0, 1, 1, 0),
        (16_002_000_000, 99.0, 99.25, np.nan, 0, 0, 0),
    ]
    tape = CandidateTape(_tape().metadata, (row,), events)
    assert evaluate_candidate_tape(tape)["trades"] == []


def test_same_timestamp_entries_use_causal_confirmation_order() -> None:
    first = _candidate()
    first["candidate_id"] = first["interaction_id"] = "ASIA|POC:0001"
    second = dict(first)
    second["candidate_id"] = second["interaction_id"] = "ASIA|POC:0002"
    second["direction"] = "SELLER_ABSORPTION"
    candidates = (second, first)
    events = np.zeros(4, dtype=EVENT_DTYPE)
    events[:] = [
        (1_000_000_000, 100.0, 100.25, np.nan, 0, 0, 0),
        (6_000_000_000, 100.0, 100.25, 100.75, 1, -1, 0),
        (6_000_000_000, 100.0, 100.25, 99.25, 1, 1, 0),
        (7_000_000_000, 100.0, 100.25, np.nan, 0, 0, 0),
    ]
    causal_tape = CandidateTape(_tape().metadata, candidates, events)
    result = evaluate_candidate_tape(causal_tape)
    assert result["trades"][0]["setup_id"] == "L2:ASIA|POC:0001"


def test_tape_round_trip_preserves_source_and_feature_contract(tmp_path: Path) -> None:
    path = tmp_path / "candidate-tape.npz"
    write_tape(path, _tape())
    loaded = load_tape(path, source_sha256="source", semantic_sha256="semantic")
    assert loaded.metadata["tape_version"] == TAPE_VERSION
    assert loaded.candidates[0]["interaction_id"] == "ASIA|POC:0001"
    assert loaded.events["timestamp_ns"].tolist() == _tape().events["timestamp_ns"].tolist()
    assert loaded.feature_matrix.shape == (1, len(FEATURE_NAMES))


def test_tape_rejects_stale_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "candidate-tape.npz"
    write_tape(path, _tape())
    try:
        load_tape(path, source_sha256="changed")
    except Exception as exc:
        assert "source hash" in str(exc)
    else:
        raise AssertionError("stale source tape was accepted")


def test_weight_q_population_uses_vectorized_candidate_matrix() -> None:
    tape = _tape()
    population = evaluate_weight_q_matrix(
        tape,
        np.asarray([[0.28, 0.25, 0.22, 0.12, 0.13], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        np.asarray([0.55, 0.55]),
    )
    assert population.shape == (1, 2)
    assert population[0, 0]
    assert not population[0, 1]


def test_parameter_classification_keeps_lifecycle_changes_out_of_posthoc_path() -> None:
    rows = {row["parameter"]: row for row in classify_parameters(L2Config())}
    assert rows["aggression_weight"]["class"] == "A"
    assert rows["confirmation_volume_threshold"]["class"] == "B"
    assert rows["entry_delay"]["class"] == "B"
    assert rows["stop_ticks"]["class"] == "B"
    assert rows["setup_timeout"]["class"] == "C"


def test_trade_ledger_comparison_is_deterministic() -> None:
    expected = [{"setup_id": "x", "entry_timestamp_ns": 1, "exit_timestamp_ns": 2,
                 "exit_reason": "STOP", "instrument": "ES", "contracts": 1,
                 "entry": 1.0, "stop": 0.0, "target": 2.0, "r_multiple": -1.0}]
    assert compare_trade_ledgers(expected, expected)["pass"]


def test_event_spool_drops_unchanged_quotes_but_keeps_execution_and_entry_probe(tmp_path: Path) -> None:
    spool = EventSpool()
    try:
        spool.append({"timestamp_ns": 1, "bid": 100, "ask": 100.25, "session": "ASIA", "execution_size": 0})
        spool.append({"timestamp_ns": 2, "bid": 100, "ask": 100.25, "session": "ASIA", "execution_size": 0})
        spool.append({"timestamp_ns": 3, "bid": 100, "ask": 100.25, "session": "ASIA",
                      "execution_price": 100, "execution_size": 1, "aggressor": "SELL"})
        spool.append({"timestamp_ns": 3_000_003, "bid": 100, "ask": 100.25, "session": "ASIA", "execution_size": 0})
        rows = spool.to_array()
        assert rows["timestamp_ns"].tolist() == [1, 3, 3_000_003]
        assert spool.bbo_transition_count == 1
        assert spool.execution_event_count == 1
    finally:
        spool.close()


def test_sparse_tape_is_rejected_for_exact_class_b_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "sparse-tape.npz"
    sparse = _tape()
    sparse.metadata.pop("bbo_path_complete")
    write_tape(path, sparse)
    try:
        load_tape(path)
    except Exception as exc:
        assert "complete executable BBO path" in str(exc)
    else:
        raise AssertionError("sparse tape was accepted")
