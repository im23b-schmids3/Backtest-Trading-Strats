from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_jj_fair_pricing_reversion_bos_v1 as study


def _tape(rows):
    dtype = [
        ("timestamp_ns", "i8"), ("bid", "f8"), ("ask", "f8"),
        ("execution_price", "f8"), ("execution_size", "f8"),
    ]
    return np.asarray(rows, dtype=dtype)


def _utc_ns(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000)


def test_ny_open_conversion_is_dst_safe():
    assert study.ny_open_ns("2025-03-03") == _utc_ns("2025-03-03T14:30:00Z")
    assert study.ny_open_ns("2025-10-07") == _utc_ns("2025-10-07T13:30:00Z")


def test_opening_anchor_is_first_valid_trade_inside_opening_minute():
    opening = study.ny_open_ns("2025-03-03")
    result = study.opening_anchor(
        [opening - 1, opening, opening + 1_000_000_000, opening + 61_000_000_000],
        [99.0, float("nan"), 100.25, 101.0], opening,
    )
    assert result == {"price": 100.25, "timestamp_ns": opening + 1_000_000_000, "status": "VALID"}
    missing = study.opening_anchor([opening + 60_000_000_000], [100.0], opening)
    assert missing["status"] == "MISSING_OPENING_MINUTE_TRADE"


def test_trade_bars_aggregate_observed_trades_without_synthesizing_minutes():
    opening = study.ny_open_ns("2025-03-03")
    bars = study.build_one_minute_bars(
        [opening + 2, opening + 10_000_000_000, opening + 60_000_000_000],
        [100.0, 101.0, 102.0], [1, 2, 3], opening, opening + 180_000_000_000,
    )
    assert [b["minute_index"] for b in bars] == [0, 1]
    assert (bars[0]["open"], bars[0]["high"], bars[0]["low"], bars[0]["close"], bars[0]["volume"]) == (100.0, 101.0, 100.0, 101.0, 3.0)
    assert bars[1]["start_ns"] == opening + study.MINUTE_NS


def test_displacement_uses_completed_close_not_intraminute_extreme():
    # The bar high hits 32 ticks, but a completed close below the threshold does not qualify.
    assert not study.is_displaced_close(100.0, 107.75, 1)
    assert study.is_displaced_close(100.0, 108.0, 1)
    assert study.is_displaced_close(100.0, 92.0, -1)
    assert not study.is_displaced_close(100.0, 92.25, -1)


def test_episode_detection_waits_for_completed_candle_close():
    day = "2025-03-03"
    opening = study.ny_open_ns(day)
    anchor = {"price": 100.0, "timestamp_ns": opening, "status": "VALID"}
    intraminute_only = _tape([
        (opening + 61_000_000_000, 107.75, 108.0, 108.0, 1.0),
        (opening + 119_000_000_000, 107.5, 107.75, 107.75, 1.0),
    ])
    _, episodes, _, _ = study._episode_and_candidates(day, intraminute_only, anchor)
    assert episodes == []

    completed_close = _tape([
        (opening + 61_000_000_000, 108.0, 108.25, 108.0, 1.0),
        (opening + 119_000_000_000, 108.0, 108.25, 108.25, 1.0),
    ])
    _, episodes, _, _ = study._episode_and_candidates(day, completed_close, anchor)
    assert len(episodes) == 1
    assert episodes[0]["detected_at_ns"] == opening + 2 * study.MINUTE_NS
    assert episodes[0]["displacement_distance_ticks"] == 33.0


def test_bos_uses_only_two_immediately_prior_completed_candles():
    bars = {
        8: {"low": 100.0, "high": 105.0, "close": 104.0},
        9: {"low": 101.0, "high": 106.0, "close": 105.0},
        10: {"low": 99.0, "high": 104.0, "close": 99.75},
        # An extreme future candle must not affect the already evaluated BOS.
        11: {"low": 1.0, "high": 200.0, "close": 1.0},
    }
    result = study.bos_for_bar(bars, 10, 1)
    assert result["confirmed"] and result["reference_level"] == 100.0
    assert result["prior_minute_indices"] == [8, 9]
    assert study.bos_for_bar({9: bars[9], 10: bars[10]}, 10, 1)["status"] == "MISSING_REFERENCE_CANDLE"


def test_remaining_distance_boundary_and_structural_stops():
    assert study.classify_bos_distance(16.0) == "BOS_CONFIRMED"
    assert study.classify_bos_distance(15.999) == "BOS_TOO_CLOSE_TO_ANCHOR"
    assert study.classify_bos_distance(16.0, True) == "DIRECTION_ALREADY_TRADED"
    assert study.structural_stop(1, 100.0, 110.0) == 110.25
    assert study.structural_stop(-1, 100.0, 90.0) == 89.75


def test_stop_first_precedence_is_explicit():
    assert study.resolve_exit_trigger(True, True) == "STOP"
    assert study.resolve_exit_trigger(False, True) == "TARGET"
    assert study.resolve_exit_trigger(False, False) is None


def test_entry_waits_two_ms_uses_executable_quote_and_adverse_tick():
    signal = 1_000_000_000
    tape = _tape([
        (signal + 1_000_000, 100.0, 100.25, 0.0, 0.0),
        (signal + 2_000_000, 100.0, 100.25, 0.0, 0.0),
        (signal + 5_000_000, 101.0, 101.25, 0.0, 0.0),
    ])
    result = study.simulate_entry_and_path(
        tape, day="2025-03-03", signal_ns=signal, signal_close=100.0,
        direction=1, anchor=101.0, stop=99.75,
        last_entry_ns=signal + 10_000_000_000,
        exit_deadline_ns=signal + 20_000_000_000,
        session_end_ns=signal + 30_000_000_000,
    )
    assert result["status"] == "EXECUTED"
    assert result["entry_timestamp_ns"] == signal + 2_000_000
    assert result["entry_quote"] == 100.25 and result["entry_price"] == 100.50
    assert result["outcome"] == "TARGET"
    assert result["exit_quote"] == 101.0 and result["exit_price"] == 100.75
    assert result["fees_usd"] == 6.0


def test_target_requires_executable_side_and_missing_quotes_fail_closed():
    signal = 1_000_000_000
    # Ask crossing the anchor is not a long target; the bid has not reached it.
    tape = _tape([
        (signal + 2_000_000, 100.0, 100.25, 0.0, 0.0),
        (signal + 3_000_000, 100.75, 101.25, 0.0, 0.0),
    ])
    censored = study.simulate_entry_and_path(
        tape, day="2025-03-03", signal_ns=signal, signal_close=100.0,
        direction=1, anchor=101.0, stop=99.75,
        last_entry_ns=signal + 10_000_000_000,
        exit_deadline_ns=signal + 4_000_000,
        session_end_ns=signal + 20_000_000,
    )
    assert censored["status"] == "CENSORED_NO_RELIABLE_EXIT_QUOTE"
    crossed = _tape([(signal + 2_000_000, 101.0, 100.75, 0.0, 0.0)])
    no_entry = study.simulate_entry_and_path(
        crossed, day="2025-03-03", signal_ns=signal, signal_close=100.0,
        direction=1, anchor=101.0, stop=99.75,
        last_entry_ns=signal + 10_000_000_000,
        exit_deadline_ns=signal + 20_000_000_000,
        session_end_ns=signal + 30_000_000_000,
    )
    assert no_entry["status"] == "NO_VALID_ENTRY_QUOTE_BEFORE_LAST_ENTRY"


def test_daily_aggregation_counts_censored_fills_but_excludes_unknown_pnl():
    metrics = study._stats([
        {"status": "EXECUTED", "net_r": 0.5, "gross_r": 0.6, "net_pnl_usd": 28.0,
         "outcome": "TARGET", "mfe_ticks": 8, "mae_ticks": 2},
        {"status": "CENSORED_NO_RELIABLE_EXIT_QUOTE"},
    ])
    assert metrics["events"] == 2
    assert metrics["executable_trades"] == 2
    assert metrics["completed_trade_outcomes"] == 1
    assert metrics["censored_after_entry"] == 1
    assert metrics["net_total_r"] == 0.5


def test_authoritative_date_lists_and_source_coverage_gate(monkeypatch, tmp_path):
    assert len(study.SPRING_DATES) == 35
    assert len(study.OCTOBER_DATES) == 19
    paths = {day: tmp_path / f"{day}.dbn.zst" for day in study.TARGET_DATES}
    rows = {day: {"sha256": "0" * 64} for day in study.TARGET_DATES}
    monkeypatch.setattr(study.native, "_source_catalog", lambda root: (paths, rows))
    checked_paths, checked_rows = study._validate_sources(tmp_path)
    assert set(checked_paths) == set(study.TARGET_DATES)
    assert set(checked_rows) == set(study.TARGET_DATES)
    monkeypatch.setattr(study.native, "_source_catalog", lambda root: (dict(list(paths.items())[:-1]), rows))
    with pytest.raises(study.FairPriceStudyError, match="does not cover"):
        study._validate_sources(tmp_path)


def test_artifact_writers_are_readable_and_hashable(tmp_path):
    json_path = tmp_path / "manifest.json"
    gzip_path = tmp_path / "events.jsonl.gz"
    study._write_json(json_path, {"status": "PASS", "count": 1})
    study._write_jsonl_gz(gzip_path, [{"event": 1}, {"event": 2}])
    assert json.loads(json_path.read_text()) == {"count": 1, "status": "PASS"}
    with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
        assert [json.loads(line) for line in f] == [{"event": 1}, {"event": 2}]
    assert study._sha(json_path) and study._sha(gzip_path)
    assert not list(tmp_path.glob(".*.tmp"))
