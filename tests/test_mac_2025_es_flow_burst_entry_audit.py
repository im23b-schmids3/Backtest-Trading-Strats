from __future__ import annotations

import gzip
import json

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_flow_burst_entry_audit as audit


def test_causal_duration_and_sign_resets():
    # Above at 0; below at 100ms; next crossings 700ms, 1.3s and 3.5s.
    ts = np.array([0, 100, 700, 800, 1300, 1400, 3500, 3600, 4200], dtype=np.int64) * 1_000_000
    pressure = np.array([1.1, .2, 1.1, .2, 1.1, .2, 1.1, -.1, 1.1])
    sessions = np.zeros(len(ts), dtype=np.int8)
    bursts = audit.burst_indices(ts, pressure, sessions, {"LONG": 1.0, "SHORT": 1.0})
    assert bursts["A"].tolist() == [0, 2, 4, 6, 8]
    assert bursts["B"].tolist() == [0, 6]
    assert bursts["C"].tolist() == [0, 6]
    assert bursts["D"].tolist() == [0, 8]
    assert audit.time_since_previous(ts, sessions, pressure, bursts["C"]) == {0: None, 6: 3.5}


def test_q98_is_single_c_rule_and_session_reset():
    ts = np.array([0, 100, 600, 700, 3100, 3200], dtype=np.int64) * 1_000_000
    pressure = np.array([2.1, .1, 2.1, .1, 2.1, 2.1])
    sessions = np.array([0, 0, 0, 0, 0, 1], dtype=np.int8)
    assert audit.burst_indices(ts, pressure, sessions, {"LONG": 2., "SHORT": 2.}, q98=True)["E"].tolist() == [0, 4, 5]


def test_same_direction_ordinal_and_buckets():
    pressure = np.array([1., 1., 1., -1., -1.])
    sessions = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    old = np.arange(5, dtype=np.int64)
    bursts = np.array([0, 2, 3], dtype=np.int64)
    assert audit.ordinal_assignments(old, bursts, pressure, sessions) == {0: 1, 1: 2, 2: 1, 3: 1, 4: 2}
    assert audit._bucket_since(None) == "NO_PREVIOUS_BURST"
    assert audit._bucket_since(2.0) == "2_TO_5S"


def test_execution_path_and_exact_decomposition(monkeypatch):
    dtype = [("ts", "i8"), ("mid", "f8")]
    rows = np.array([(0, 100.0)], dtype=dtype)
    tape_dtype = [("timestamp_ns", "i8"), ("session", "i1"), ("bid", "f8"), ("ask", "f8")]
    tape = np.array([(0, 0, 99.875, 100.125),
                     (2_000_000, 0, 100.125, 100.375),
                     (252_000_000, 0, 100.375, 100.625),
                     (502_000_000, 0, 100.625, 100.875),
                     (1_002_000_000, 0, 100.875, 101.125),
                     (2_002_000_000, 0, 101.125, 101.375),
                     (5_002_000_000, 0, 101.375, 101.625),
                     (10_002_000_000, 0, 101.625, 101.875)], dtype=tape_dtype)
    monkeypatch.setattr(audit.flow, "_quote_arrays", lambda x: (x["timestamp_ns"], x["session"], x["bid"], x["ask"]))
    features = audit._feature_rows("2025-03-03", rows, tape, np.array([1.]),
                                   np.array([0]), {v: np.array([0]) for v in "ABCDE"}, np.array([0]))
    p = features[0]["paths"]["500"]
    assert p["raw"] == 3.0
    assert p["quote"] == 1.0
    assert p["actual"] == 0.0
    assert np.isclose(p["actual"], p["raw"]+p["horizon_shift"]-p["price_move_during_2ms"]+p["bid_ask_execution_cost"]-1)
    assert np.isclose(p["actual"], p["quote"]-1)
    assert features[0]["entry_time_ns"] == 2_000_000


def test_checkpoint_source_hash_invalidation(tmp_path):
    path = tmp_path / "date.json.gz"
    row = {"status": "DATE_COMPLETE", "version": audit.CHECKPOINT_VERSION,
           "date": "2025-03-03", "source_sha256": "source", "tape_sha256": "tape",
           "history_sha256": "history", "study_sha256": audit.STUDY_SHA256, "payload": {}}
    with gzip.open(path, "wt") as out:
        json.dump(row, out)
    assert audit._read_checkpoint(path, day="2025-03-03", source_sha="source", tape_sha="tape", history_sha="history") == row
    assert audit._read_checkpoint(path, day="2025-03-03", source_sha="changed", tape_sha="tape", history_sha="history") is None


def test_future_pressure_does_not_change_earlier_burst():
    ts = np.array([0, 100, 700, 2500, 3000], dtype=np.int64) * 1_000_000
    sessions = np.zeros(5, dtype=np.int8)
    a = np.array([1.2, .2, 1.3, .2, 1.2])
    b = a.copy(); b[-1] = -3.0
    threshold = {"LONG": 1.0, "SHORT": 1.0}
    for variant in "ABCD":
        assert audit.burst_indices(ts, a, sessions, threshold)[variant][0] == 0
        assert audit.burst_indices(ts, b, sessions, threshold)[variant][0] == 0


def test_buy_sell_burst_symmetry():
    ts = np.arange(6, dtype=np.int64) * 600_000_000
    pressure = np.array([1.2, .2, 1.2, -.2, 1.2, .2])
    sessions = np.zeros(6, dtype=np.int8)
    threshold = {"LONG": 1.0, "SHORT": 1.0}
    long = audit.burst_indices(ts, pressure, sessions, threshold)
    short = audit.burst_indices(ts, -pressure, sessions, threshold)
    for variant in "ABCD":
        assert long[variant].tolist() == short[variant].tolist()


def test_period_and_daily_weekly_aggregation(tmp_path):
    def record(day: str, value: float):
        path = {str(h): {"raw": value+2, "quote": value+1, "actual": value,
                         "horizon_shift": 0., "price_move_during_2ms": 0.,
                         "bid_ask_execution_cost": -1.} for h in audit.HORIZONS_MS}
        return {"date": day, "session": 0, "direction": "LONG", "variants": list("ABCDE"),
                "old_clustered": True, "ordinal_by_variant": {v: 1 for v in "ABCD"},
                "time_since_previous_same_direction_burst_seconds": {v: None for v in "ABCDE"},
                "paths": path, "mfe_10s_ticks": value+1, "mae_10s_ticks": value-1}
    payloads = [{"date": day, "old_raw_threshold_hits": 1, "old_clustered_events": 1,
                 "old_actual_entries": 0, "burst_counts": {v: 1 for v in "ABCDE"},
                 "features": [record(day, result)]}
                for day, result in (("2025-03-03", 1.), ("2025-10-07", -1.))]
    result = audit._aggregate(payloads, tmp_path)
    assert result["spring_october"]["C"]["SPRING_2025"]["actual"]["500"]["mean_ticks"] == 1
    assert result["spring_october"]["C"]["OCTOBER_2025"]["actual"]["500"]["mean_ticks"] == -1
    assert result["daily_results"]["C"][0]["date"] == "2025-03-03"
    assert len(result["weekly_results"]["C"]) == 2
