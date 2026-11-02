from __future__ import annotations

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_hvn_native_mbp10_absorption_mechanism_v1 import (
    ATTACK_AFTER_ONSET_NS,
    Episode,
    _book,
    _depth_at,
    _depth_delta,
    attack_onset_eligible,
    attack_window_contains,
    preentry_feature_complete,
    native_order_window_indices,
    timestamp_gap_exceeds,
    trade_aggressor,
)


def _row(*, bid=100.0, ask=100.25, bid_size=10, ask_size=12, flags=128):
    names = ["flags"]
    for side in ("bid", "ask"):
        for suffix in ("px", "sz"):
            names.extend(f"{side}_{suffix}_{i:02d}" for i in range(10))
    dtype = [(n, "i8" if "px" in n else "u4") for n in names]
    row = np.zeros((), dtype=dtype)
    row["flags"] = flags
    for i in range(10):
        row[f"bid_px_{i:02d}"] = int((bid - i * .25) * 1e9)
        row[f"ask_px_{i:02d}"] = int((ask + i * .25) * 1e9)
        row[f"bid_sz_{i:02d}"] = bid_size if i == 0 else 0
        row[f"ask_sz_{i:02d}"] = ask_size if i == 0 else 0
    return row


def test_native_trade_aggressor_side_is_action_dependent():
    assert trade_aggressor("T", "A") == "SELL"
    assert trade_aggressor("T", "B") == "BUY"
    assert trade_aggressor("T", "N") == "UNKNOWN"
    assert trade_aggressor("C", "A") == "NOT_A_TRADE"


def test_native_record_order_is_preserved_for_equal_timestamps():
    # Equal-time messages remain in their original DBN row order.
    records = [(10, 7, "T"), (10, 8, "C"), (11, 9, "A")]
    indices = native_order_window_indices(np.array([r[0] for r in records]), [(10, 10)])
    assert [records[i][2] for i in indices] == ["T", "C"]
    assert [records[i][1] for i in indices] == [7, 8]


def test_book_snapshot_quote_movement_and_top_depth_are_read_from_each_record():
    before = _book(_row(bid=100, ask=100.25, bid_size=10, ask_size=12))
    after = _book(_row(bid=100.25, ask=100.5, bid_size=8, ask_size=14))
    assert before["valid"] and after["valid"]
    assert before["mid"] == 100.125
    assert after["bid"] == 100.25 and after["ask"] == 100.5
    assert after["top5_bid"] == 8 and after["top10_ask"] == 14


def test_price_aligned_zone_depth_and_visible_change_during_trade_sequence():
    ep = Episode(episode_id="e", date="d", zone_id="z", t0=0, direction=1, low=100, high=100.25, poc=100, origin="fixture", v1_status="x", v1_heavy=False, v1_heavy_size=None, v1_threshold=None, v1_heavy_ns=None, v1_heavy_price=None, touch_price=100.0, approach="")
    before = _book(_row(bid=100.25, bid_size=10))
    after = _book(_row(bid=100.25, bid_size=6))
    restored = _book(_row(bid=100.25, bid_size=9))
    assert _depth_at(before, ep) == 10
    assert _depth_at(after, ep) == 6
    assert _depth_delta(before, after, ep) == -4
    assert _depth_delta(after, restored, ep) == 3
    # The positive delta is observable aggregate recovery, not an order identity.


def test_invalid_crossed_or_maybe_bad_book_is_not_counted_as_valid():
    assert not _book(_row(bid=100.5, ask=100.25))["valid"]
    assert not _book(_row(flags=128 | 4))["valid"]
    assert _book(_row())["valid"]


def test_attack_window_has_fixed_inclusive_boundaries_and_no_outcome_dependency():
    onset = 1_000_000_000
    assert attack_window_contains(onset, onset)
    assert attack_window_contains(onset + ATTACK_AFTER_ONSET_NS, onset)
    assert not attack_window_contains(onset - 1, onset)
    assert not attack_window_contains(onset + ATTACK_AFTER_ONSET_NS + 1, onset)
    assert attack_onset_eligible(onset, onset)
    assert not attack_onset_eligible(onset + 15_000_000_001, onset)


def test_preentry_feature_gate_rejects_features_that_finish_after_entry():
    onset = 10_000_000_000
    assert preentry_feature_complete(onset, onset + ATTACK_AFTER_ONSET_NS)
    assert not preentry_feature_complete(onset, onset + ATTACK_AFTER_ONSET_NS - 1)
    assert not preentry_feature_complete(None, onset + ATTACK_AFTER_ONSET_NS)


def test_timestamp_gap_diagnostic_is_boundary_exact():
    assert not timestamp_gap_exceeds(100, 100 + 2_000_000_000)
    assert timestamp_gap_exceeds(100, 100 + 2_000_000_001)
    assert not timestamp_gap_exceeds(None, 100)


def test_missing_or_one_sided_book_levels_fail_closed():
    row = _row()
    row["ask_sz_00"] = 0
    assert not _book(row)["valid"]
