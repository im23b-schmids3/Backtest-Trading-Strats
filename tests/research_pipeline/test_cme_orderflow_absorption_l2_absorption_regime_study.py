from __future__ import annotations

import numpy as np
import hashlib

from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    absorption_regime_study as study,
    mac_2025_mlofi_event_study as mlofi,
)


def _mbp_dtype():
    fields = [("ts_recv", "<i8"), ("action", "S1"), ("side", "S1"),
              ("price", "<i8"), ("size", "<i4")]
    for side in ("bid", "ask"):
        for level in range(10):
            fields += [(f"{side}_px_{level:02d}", "<i8"),
                       (f"{side}_sz_{level:02d}", "<i4")]
    return np.dtype(fields)


def _book_row(ts: int, bid_top_size: int, *, action=b"R", side=b"N",
              price=0, size=0):
    row = np.zeros((), dtype=_mbp_dtype())
    row["ts_recv"], row["action"], row["side"] = ts, action, side
    row["price"], row["size"] = price, size
    scale = study.RAW_PRICE_SCALE
    for i in range(10):
        row[f"bid_px_{i:02d}"] = (100 - i) * scale
        row[f"bid_sz_{i:02d}"] = bid_top_size if i == 0 else 10
        row[f"ask_px_{i:02d}"] = (101 + i) * scale
        row[f"ask_sz_{i:02d}"] = 10
    return row


def test_direction_normalization_and_mfe_mae_are_symmetric():
    assert study.direction_normalize(3, "SELLER_ABSORPTION") == 3
    assert study.direction_normalize(-3, "BUYER_ABSORPTION") == 3
    assert study.mfe_mae([-2, 1, 4, -3]) == (4.0, 3.0)
    assert study.mfe_mae([]) == (0.0, 0.0)


def test_recovery_and_aggression_metrics_fail_closed_on_empty_denominators():
    assert study.recovery_fraction(10, 5, 8) == 0.6
    assert study.recovery_fraction(5, 5, 8) is None
    assert study.aggression_depth_ratio(12, 4) == 3
    assert study.aggression_depth_ratio(12, 0) is None


def test_blocked_resiliency_index_returns_exact_first_crossing():
    values = np.random.default_rng(17).integers(0, 20, size=4097).astype(np.float64)
    index = study._max_tree(values)
    for start, end, threshold in ((0, 4097, 17), (63, 130, 9), (64, 128, 21),
                                  (4090, 4097, 4), (105, 106, 19)):
        expected_hits = np.flatnonzero(values[start:end] >= threshold)
        expected = start + int(expected_hits[0]) if len(expected_hits) else None
        assert study._first_at_least(index, values, start, end, threshold) == expected


def test_mlofi_bins_are_direction_normalized_and_keep_predeclared_flow_state():
    positive = study.mlofi_persistence([1, 2, 0, 1], 1, 10)
    negative = study.mlofi_persistence([-1, -2, 0, -1], -1, 10)
    assert positive["signed_persistence_relative_to_reversal"] == 0.4
    assert negative["signed_persistence_relative_to_reversal"] == 0.4
    assert positive["flow_state"] == "FLOW_SUPPORTS_REVERSAL"


def test_native_mbp_top5_update_matches_validated_price_keyed_mlofi_semantics():
    scale = study.RAW_PRICE_SCALE
    batch = np.array([
        _book_row(1, 10),
        _book_row(2, 12, action=b"M", side=b"B", price=100 * scale, size=12),
        _book_row(3, 12, action=b"M", side=b"B", price=90 * scale, size=12),
    ], dtype=_mbp_dtype())
    out, _, _, _ = study._batch_market_columns(batch, None, False, None)
    assert out["executable"].tolist() == [True, True, True]
    assert out["ofi"][1] == 2.0
    # 90 is outside visible top ten on both snapshots and must not be rank 0.
    assert out["ofi"][2] == 0.0

    before = mlofi.snapshot_from_row(batch[0])
    after = mlofi.snapshot_from_row(batch[1])
    expected = mlofi.account_mbp_event(before, after, "M", "B", 100 * scale, 12)
    assert sum(expected[:5][i] / (i + 1) for i in range(5)) == out["ofi"][1]


def test_native_mbp_trade_uses_passive_side_top5_rank():
    scale = study.RAW_PRICE_SCALE
    batch = np.array([
        _book_row(1, 10),
        _book_row(2, 10, action=b"T", side=b"B", price=101 * scale, size=7),
    ], dtype=_mbp_dtype())
    out, _, _, _ = study._batch_market_columns(batch, None, False, None)
    assert out["ofi"][1] == 7.0
    reference = mlofi.account_mbp_event(mlofi.snapshot_from_row(batch[0]),
                                        mlofi.snapshot_from_row(batch[1]),
                                        "T", "B", 101 * scale, 7)
    assert sum(reference[i] / (i + 1) for i in range(5)) == out["ofi"][1]


def test_pre_event_features_exclude_the_interaction_timestamp_and_future_book():
    ts = np.asarray([10, 14, 16], dtype=np.int64) * 1_000_000_000
    market = {
        "ts": ts,
        "mid_half_ticks": np.asarray([800, 802, 798], dtype=np.int32),
        "bid_depth": np.asarray([10.0, 12.0, 1000.0]),
        "ask_depth": np.asarray([10.0, 10.0, 1000.0]),
        "denom": np.asarray([10.0, 11.0, 1000.0]),
        "ofi": np.asarray([1.0, 2.0, 10000.0]),
        "buy_volume": np.asarray([1.0, 2.0, 10000.0]),
        "sell_volume": np.asarray([0.0, 0.0, 0.0]),
    }
    for key in ("ofi", "buy_volume", "sell_volume"):
        market[f"{key.split('_')[0]}_prefix" if key != "ofi" else "ofi_prefix"] = study._prefix(market[key])
    market["buy_prefix"] = study._prefix(market["buy_volume"])
    market["sell_prefix"] = study._prefix(market["sell_volume"])
    steps = np.zeros(3)
    steps[1:] = np.diff(market["mid_half_ticks"] / 2.0) ** 2
    market["sq_mid_prefix"] = study._prefix(steps)
    grid = np.asarray([13, 14], dtype=np.int64) * 1_000_000_000
    market.update({"depth_grid_ts": grid, "depth_grid_bid": np.asarray([10.0, 12.0]),
                   "depth_grid_ask": np.asarray([10.0, 10.0]),
                   "b_episodes": np.asarray([], dtype=np.int64),
                   "a_episodes": np.asarray([], dtype=np.int64),
                   "b_max_tree": study._max_tree(market["bid_depth"]),
                   "a_max_tree": study._max_tree(market["ask_depth"])})
    result = study._pre_features({"interaction_start_ns": 15_000_000_000,
                                  "direction": "SELLER_ABSORPTION"}, market, 0, 20_000_000_000)
    assert result["feature_end_ns"] == 14_000_000_000
    assert result["feature_end_ns"] < result["feature_cutoff_ns"]
    assert result["pre_1s_raw_inverse_level_top5_ofi"] == 2.0


def test_expanding_buckets_use_prior_dates_only_for_same_date_rows():
    initial = {("F", "pre_1s_nMLOFI"): [float(i) for i in range(20)]}
    rows = [{"family": "F", "date": "2025-12-01", "pre_1s_nMLOFI": 1.0},
            {"family": "F", "date": "2025-12-01", "pre_1s_nMLOFI": 100.0}]
    study._assign_expanding_buckets(rows, initial)
    assert rows[0]["bucket_pre_1s_nMLOFI"] == "Q1"
    assert rows[1]["bucket_pre_1s_nMLOFI"] == "Q5"
    assert len(initial[("F", "pre_1s_nMLOFI")]) == 22


def test_cross_month_contrast_requires_date_separate_q1_q5_support():
    def month_tree(q1, q5):
        return {"families": {"F": {
            "Q1": {"markouts": {"markout_2000ms_ticks": {"count": 5, "mean": q1}}},
            "Q5": {"markouts": {"markout_2000ms_ticks": {"count": 5, "mean": q5}}},
        }}}
    result = study._cross_month_bucket_contrast({"by_month": {
        "2025-12": month_tree(-0.5, 0.5), "2026-01": month_tree(-0.2, 0.8)}})
    assert result["comparable_families"] == 1
    assert result["families"]["F"]["months"]["2025-12"]["q5_minus_q1_ticks"] == 1.0
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"


def test_bucket_summary_accepts_stream_and_keeps_month_level_inputs_small():
    rows = ({"family": "F", "date": "2025-12-01", "bucket_x": "Q1",
             "markout_2000ms_ticks": float(i), "mfe_1000ms_ticks": 1.0,
             "mae_1000ms_ticks": 0.0, "barrier_1_1_outcome": "FAVORABLE_FIRST",
             "barrier_1_1_seconds": 0.2} for i in range(5))
    result = study._bucket_summaries(rows, "x", "bucket_x", "test")
    summary = result["families"]["F"]["Q1"]
    assert summary["event_count"] == 5
    assert summary["markouts"]["markout_2000ms_ticks"]["mean"] == 2.0
    assert summary["barriers"]["1_1"]["favorable_first_probability"] == 1.0


def test_first_touch_is_ordered_and_checkpoint_requires_exact_identity():
    assert study.first_touch([1, 2], [0.5, -1], 1, 1)["outcome"] == "ADVERSE_FIRST"
    assert study.first_touch([1], [1], 1, 1)["outcome"] == "FAVORABLE_FIRST"
    checkpoint = {"status": "DATE_COMPLETE", "date": "2025-12-02",
                  "source_sha256": "s", "config_sha256": "c", "output_sha256": "o"}
    assert study.checkpoint_matches(checkpoint, date="2025-12-02", source_sha256="s",
                                    config_sha256="c", output_sha256="o")
    assert not study.checkpoint_matches(checkpoint, date="2025-12-03", source_sha256="s",
                                        config_sha256="c", output_sha256="o")
    assert study.interaction_within_session(10, 20, (10, 20))
    assert study.interaction_within_session(19, 20, (10, 20))
    assert not study.interaction_within_session(20, 20, (10, 20))
    assert not study.interaction_within_session(19, 21, (10, 20))


def test_gzip_event_artifact_is_byte_deterministic(tmp_path):
    rows = [{"event_id": "d:f:i", "value": 1.25}]
    left, right = tmp_path / "left.jsonl.gz", tmp_path / "right.jsonl.gz"
    study._jsonl_gz_write(left, rows)
    study._jsonl_gz_write(right, rows)
    assert hashlib.sha256(left.read_bytes()).digest() == hashlib.sha256(right.read_bytes()).digest()


def test_json_writer_normalizes_numpy_scalars(tmp_path):
    path = tmp_path / "summary.json"
    study._json_write(path, {"flag": np.bool_(True), "count": np.int64(3),
                             "value": np.float64(1.25)})
    assert path.read_text(encoding="utf-8") == '{\n  "count": 3,\n  "flag": true,\n  "value": 1.25\n}\n'


def test_frozen_identity_and_no_pnl_contract_are_explicit():
    assert study.EXPECTED_CONFIG_SHA256 == "99c4af7f7b03cf6a255781524f7a6c2a32bd992d9785dbfb294db6a07fbc7448"
    definitions = study._feature_definitions()
    assert definitions["no_strategy_pnl"] is True
    assert definitions["dec_jan_are_not_untouched_oos"] is True
    assert len(study.BARRIER_PAIRS) == 5
