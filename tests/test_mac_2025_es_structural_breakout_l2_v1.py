from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_structural_breakout_l2_v1 as study


TAPE_DTYPE = [("timestamp_ns", "i8"), ("bid", "f8"), ("ask", "f8"),
              ("execution_price", "f8"), ("execution_size", "i8"),
              ("aggressor", "i1"), ("session", "i1")]


def _row(t, price, size=1):
    return (t, price-.125, price+.125, price, size, 1, 2)


def test_trade_only_or_boundaries_strict_break_and_uniqueness():
    start, end, close = study._windows("2025-03-03")
    tape = np.array([_row(start-1, 200.), _row(start, 100.),
                     _row(end-1, 101.), _row(end, 101.),
                     _row(end+1, 101.25), _row(end+2, 101.5),
                     _row(end+3, 100.), _row(end+4, 99.75),
                     _row(close, 90.)], dtype=TAPE_DTYPE)
    opening, events = study.opening_range_events("2025-03-03", tape)
    assert opening["high"] == 101.
    assert opening["low"] == 100.
    assert [e["direction"] for e in events] == ["LONG", "SHORT"]
    assert events[0]["trade_price"] == 101.25
    assert events[0]["timestamp_ns"] == end+1
    assert events[1]["timestamp_ns"] == end+4
    assert opening["status"] == "BOTH_DIRECTIONS"


def test_or_ignores_quote_only_row_and_boundary_equality():
    start, end, _ = study._windows("2025-03-03")
    tape = np.array([_row(start, 100.), (start+1, 0., 1000., 999., 0, 0, 2),
                     _row(end-1, 101.), _row(end, 101.), _row(end+1, 100.)], dtype=TAPE_DTYPE)
    opening, events = study.opening_range_events("2025-03-03", tape)
    assert opening["high"] == 101.
    assert events == []


def test_ineligible_day_rejected():
    with pytest.raises(study.BreakoutStudyError):
        study._period("2026-09-18")


def test_chronological_terciles_and_refill_direction():
    history = {"mlofi_500ms": [1., 2., 3.], "refill_recovery_500ms": [1., 2., 3.]}
    assert study._prior_quantile(history, "mlofi_500ms", 4.) == ("HIGH", True)
    assert study._prior_quantile(history, "refill_recovery_500ms", 0.) == ("LOW", True)
    assert study._prior_quantile({"mlofi_500ms": []}, "mlofi_500ms", 4.) == (None, None)


def test_bounded_depth_depletion_and_no_future():
    dtype = [("ts", "i8"), ("ask5", "f8")]
    start = 1_000_000_000_000
    ts = start+np.arange(0, 601, dtype=np.int64)*50_000_000
    rows = np.zeros(len(ts), dtype=dtype)
    rows["ts"] = ts
    rows["ask5"] = 100
    assert study._sample_depth_baseline(rows, start+30_000_000_000, start, "ask5") == 100
    rows["ask5"][-1] = 1
    assert study._sample_depth_baseline(rows, start+30_000_000_000, start, "ask5") == 100


def test_execution_decomposition_and_direction_symmetry():
    start, end, close = study._windows("2025-03-03")
    tape = np.array([_row(end, 100.), _row(end+2_000_000, 100.25),
                     _row(end+502_000_000, 100.75),
                     _row(end+1_002_000_000, 101.),
                     _row(end+2_002_000_000, 101.25),
                     _row(end+5_002_000_000, 101.5),
                     _row(end+10_002_000_000, 102.),
                     _row(end+30_002_000_000, 102.25),
                     _row(end+60_002_000_000, 102.5)], dtype=TAPE_DTYPE)
    opening = {"rth_close_ns": close}
    event = {"timestamp_ns": end, "sign": 1, "tape_index": 0, "date": "2025-03-03"}
    path = study._path_analysis(tape, event, opening)
    p = path["paths"]["500"]
    assert np.isclose(p["actual"], p["raw"]+p["horizon_shift"]-p["pre_entry_price_move"]+p["bid_ask_effect"]-1)
    assert np.isclose(p["actual"], p["quote"]-1)
    assert path["barriers"]["4:-4@10000ms"]["result"] == "FAVORABLE_FIRST"
    assert path["excursions"]["10000"]["mfe"] >= 4
    mirrored = tape.copy()
    for field in ("bid", "ask", "execution_price"):
        mirrored[field] = -tape[field]
    mirrored["bid"] = -tape["ask"]
    mirrored["ask"] = -tape["bid"]
    short = study._path_analysis(mirrored, {**event, "sign": -1}, opening)
    assert np.isclose(short["paths"]["500"]["actual"], p["actual"])


def test_bucket_sample_and_permutation_safety():
    assert study._markout([])["n"] == 0
    assert study._permutation([], "mlofi_500ms", np.random.default_rng(1))["status"] == "INSUFFICIENT_SAMPLE"


def test_return_inside_requires_subsequent_trade_not_inside_midpoint():
    _, t, close = study._windows("2025-03-03")
    # The anchor midpoint is inside the trade-defined OR, but every trade is
    # still beyond its HIGH. This is not a post-event return.
    tape = np.array([(t, 100.75, 101.00, 101.25, 1, 1, 2),
                     (t+10_000_000_000, 100.75, 101.00, 101.25, 1, 1, 2),
                     (t+60_000_000_000, 100.75, 101.00, 101.25, 1, 1, 2)], dtype=TAPE_DTYPE)
    opening = {"high": 101., "low": 100., "rth_close_ns": close}
    event = {"timestamp_ns": t, "sign": 1}
    assert study._topology(tape, event, opening, 0, 100.875) == "STAGNATION"
    tape[1]["execution_price"] = 100.75
    assert study._topology(tape, event, opening, 0, 100.875) == "RETURN_INSIDE_OPENING_RANGE"


def test_delayed_refill_cannot_change_event_anchor_features():
    start, event_t, close = study._windows("2025-03-03")
    ts = np.arange(event_t-30_000_000_000, event_t+550_000_000,
                   50_000_000, dtype=np.int64)
    from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_normalization as relative
    rows = np.zeros(len(ts), dtype=relative.COMPACT_DTYPE)
    rows["ts"] = ts
    rows["mid"] = 100.
    rows["bid5"] = 100.
    rows["ask5"] = 100.
    rows["denom"] = 100.
    rows["mlofi"] = 1.
    rows["action"] = 2
    rows["ask5"][np.searchsorted(ts, event_t, side="left")-1] = 50.
    tape = np.array([_row(x, 100.) for x in np.arange(event_t-10_000_000_000, event_t+500_000_000,
                                                        500_000_000, dtype=np.int64)], dtype=TAPE_DTYPE)
    event = {"timestamp_ns": event_t, "sign": 1, "overshoot_ticks": 1., "date": "2025-03-03"}
    opening = {"or_start_ns": start, "rth_close_ns": close, "width_ticks": 8.}
    before, delayed_before = study._features(rows, tape, event, opening)
    changed = rows.copy()
    changed["ask5"][ts >= event_t] = 200.
    after, delayed_after = study._features(changed, tape, event, opening)
    assert before == after
    assert delayed_before["refill_recovery_500ms"] != delayed_after["refill_recovery_500ms"]


def test_checkpoint_resume_and_invalidations(tmp_path):
    path = tmp_path / "checkpoint.json.gz"
    row = {"status": "DATE_COMPLETE", "version": study.CHECKPOINT_VERSION,
           "date": "2025-03-03", "source_sha256": "source", "tape_sha256": "tape",
           "history_sha256": "history", "config_sha256": study.CONFIG_SHA256,
           "payload": {}}
    with gzip.open(path, "wt") as out:
        json.dump(row, out)
    assert study._read_checkpoint(path, "2025-03-03", "source", "tape", "history") == row
    assert study._read_checkpoint(path, "2025-03-03", "wrong", "tape", "history") is None
    assert study._read_checkpoint(path, "2025-03-03", "source", "tape", "wrong") is None
    row["config_sha256"] = "wrong"
    with gzip.open(path, "wt") as out:
        json.dump(row, out)
    assert study._read_checkpoint(path, "2025-03-03", "source", "tape", "history") is None


def test_price_only_match_excludes_outcomes_and_l2_from_distance():
    def item(day, support, outcome):
        features = {k: 1. for k in study.PRICE_KEYS}
        return {"date": day, "period": "SPRING_2025", "direction": "LONG",
                "timestamp_ns": 1, "event_anchor_support_count": support,
                "event_anchor_features": features,
                "paths": {"10000": {"raw": outcome}}}
    rows = [item("2025-03-03", 4, 3.), item("2025-03-04", 0, 1.),
            item("2025-03-05", 0, -2.)]
    result = study._price_only_control(rows)
    assert result["matched_pairs"] == 1
    assert result["incremental_ticks"] == 2.
    assert result["matching_uses_outcomes"] is False


def test_permutation_seed_and_leave_out_are_deterministic():
    dates = ("2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06",
             "2025-03-07", "2025-03-10")
    events = []
    for n in range(30):
        high = n % 2 == 0
        events.append({"date": dates[n % len(dates)], "period": "SPRING_2025",
            "direction": "LONG" if n % 3 else "SHORT",
            "event_anchor_support_count": 4 if high else 0,
            "prior_date_terciles": {"mlofi_500ms": "HIGH" if high else "LOW",
                                    "depth_depletion": "HIGH" if high else "LOW"},
            "paths": {"10000": {"raw": 1. if high else -1.}}})
    a = study._permutation(events, "mlofi_500ms", np.random.default_rng(2))
    b = study._permutation(events, "mlofi_500ms", np.random.default_rng(2))
    assert a == b
    assert a["status"] == "COMPLETE"
    assert study._leave_out(events, "date")["base"]["groups_tested"] == 6
    assert study._leave_out(events, "week")["base"]["groups_tested"] == 2
