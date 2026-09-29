from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_event_study import (
    BookSnapshot,
    COMBO_INDEX,
    CompactFeatures,
    FEATURE_CACHE_VERSION,
    MAIN_VARIANT,
    TICK_POINTS,
    _aggregate_date_payloads,
    _atomic_npz,
    _barrier_counts,
    _feature_cache_paths,
    _fixed_view,
    _iter_event_signal_batches,
    _load_feature_cache,
    _markouts,
    _percentile_bounds,
    _signals_from_rolling,
    account_mbp_event,
    first_touch_outcome,
    forward_markout_ticks,
    signal_values,
    variant_index,
    weighted_resting_depth,
)


def book(*, bid_size: int = 10, ask_size: int = 10,
         bid_price: int = 100_000_000_000, ask_price: int = 100_250_000_000) -> BookSnapshot:
    return BookSnapshot(1, (bid_price,), (bid_size,), (ask_price,), (ask_size,))


def test_bid_add_and_remove_signs() -> None:
    previous = book(bid_size=10)
    added = book(bid_size=12)
    removed = book(bid_size=8)
    assert account_mbp_event(previous, added, "A", "B", added.bid_prices[0], 2)[0] == 2
    assert account_mbp_event(previous, removed, "C", "B", removed.bid_prices[0], 2)[0] == -2


def test_ask_add_and_remove_signs() -> None:
    previous = book(ask_size=10)
    added = book(ask_size=12)
    removed = book(ask_size=8)
    assert account_mbp_event(previous, added, "A", "A", added.ask_prices[0], 2)[0] == -2
    assert account_mbp_event(previous, removed, "C", "A", removed.ask_prices[0], 2)[0] == 2


def test_repricing_does_not_infer_cancel_at_untouched_rank() -> None:
    previous = BookSnapshot(1, (100_000_000_000, 99_750_000_000), (10, 20),
                            (100_250_000_000,), (10,))
    current = BookSnapshot(2, (100_250_000_000, 100_000_000_000), (4, 10),
                           (100_500_000_000,), (10,))
    contribution = account_mbp_event(previous, current, "A", "B", 100_250_000_000, 4)
    assert contribution[0] == 4
    assert sum(contribution[1:]) == 0


def test_trade_uses_execution_once_not_book_delta() -> None:
    previous = book(ask_size=10)
    current = book(ask_size=9)
    contribution = account_mbp_event(previous, current, "T", "B", current.ask_prices[0], 1)
    assert contribution[0] == 1
    assert sum(contribution[1:]) == 0


def test_depth_normalization_uses_current_book_only() -> None:
    sums = {window: (1.0,) + (0.0,) * 9 for window in (250, 500, 1000, 2000)}
    shallow = book(bid_size=10, ask_size=10)
    deep = book(bid_size=100, ask_size=100)
    shallow_values = signal_values(sums, shallow)
    deep_values = signal_values(sums, deep)
    inverse_depth = variant_index(5, "INVERSE_LEVEL", 500, "DEPTH_NORMALIZED")
    assert shallow_values[inverse_depth] > deep_values[inverse_depth]
    assert weighted_resting_depth(shallow, (1.0,)) == 10


def test_variant_identity_and_fixed_definitions_are_deterministic() -> None:
    assert variant_index(*MAIN_VARIANT) == 35
    assert TICK_POINTS == 0.25


def test_train_manifest_path_is_not_an_oos_path() -> None:
    manifest = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE/candidate-tapes/train-tape-manifest.json")
    assert "TRAIN" in manifest.name.upper() or "train" in manifest.name
    assert "OOS" not in str(manifest).upper()


def test_forward_markout_uses_first_state_at_or_after_horizon() -> None:
    states = [(1_000_000_000, 100.0, "ASIA"), (1_100_000_000, 100.25, "ASIA"),
              (1_300_000_000, 100.5, "ASIA")]
    assert forward_markout_ticks(1_000_000_000, 100.0, "ASIA", 100, states) == 1.0


def test_forward_markout_does_not_cross_session_boundary() -> None:
    states = [(1_100_000_000, 100.25, "EUROPE")]
    assert forward_markout_ticks(1_000_000_000, 100.0, "ASIA", 100, states) is None


def test_barrier_first_touch_is_deterministic_and_direction_neutral() -> None:
    states = [(1_100_000_000, 100.25, "ASIA"), (1_200_000_000, 99.75, "ASIA")]
    assert first_touch_outcome(1_000_000_000, 100.0, "ASIA", 1, states) == 1
    same_event = [(1_100_000_000, 100.0, "ASIA")]
    assert first_touch_outcome(1_000_000_000, 100.0, "ASIA", 1, same_event) == 0


def test_expanding_percentile_contract_is_train_only() -> None:
    # The event-study implementation assigns a bucket before adding the
    # current observation to its reservoir; no later/OOS value can affect it.
    from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_event_study import Reservoir
    reservoir = Reservoir(100, 7)
    for value in range(100):
        reservoir.add(float(value))
    before = reservoir.decile(50.0)
    reservoir.add(10_000.0)
    assert before is not None


def compact_features(*, timestamp: np.ndarray, contribution: np.ndarray | None = None,
                     midpoint: np.ndarray | None = None) -> CompactFeatures:
    size = timestamp.size
    return CompactFeatures(
        timestamp_ns=timestamp.astype(np.int64), session=np.zeros(size, dtype=np.int8),
        mid_sum_raw=(midpoint if midpoint is not None else np.full(size, 200_250_000_000, dtype=np.int64)),
        microprice=np.full(size, 100_125_000_000.0), depth_sum=np.full((size, 10), 20, dtype=np.uint32),
        contribution=(contribution if contribution is not None else np.zeros((size, 10), dtype=np.int32)),
        event_sample=np.ones(size, dtype=np.uint8), count=size, raw_rows=size,
    )


def test_cumulative_rolling_windows_equal_naive_reference() -> None:
    timestamp = np.asarray((0, 100_000_000, 300_000_000, 700_000_000), dtype=np.int64)
    contribution = np.zeros((4, 10), dtype=np.int32)
    contribution[:, 0] = (1, 2, -1, 3)
    features = compact_features(timestamp=timestamp, contribution=contribution)
    batches = list(_iter_event_signal_batches(features, block_rows=2))
    signals = np.vstack([batch[4] for batch in batches])
    index = variant_index(3, "EQUAL", 500, "RAW")
    expected = []
    for row, now in enumerate(timestamp):
        expected.append(sum(contribution[column, 0] for column, then in enumerate(timestamp) if then >= now - 500_000_000 and column <= row))
    assert signals[:, index].tolist() == expected


def test_weighted_variants_equal_direct_reference() -> None:
    rolling = {window: np.zeros((1, 9), dtype=np.float64) for window in (250, 500, 1000, 2000)}
    rolling[500][0, COMBO_INDEX[(5, "INVERSE_LEVEL")]] = 1.0 + 2.0 / 2.0 + 3.0 / 3.0
    depth = np.full((1, 10), 20, dtype=np.uint32)
    values = _signals_from_rolling(rolling, depth)
    raw = variant_index(5, "INVERSE_LEVEL", 500, "RAW")
    normalized = variant_index(5, "INVERSE_LEVEL", 500, "DEPTH_NORMALIZED")
    assert values[0, raw] == pytest.approx(3.0)
    assert values[0, normalized] == pytest.approx(0.3)


def test_vector_markouts_equal_forward_search_reference() -> None:
    timestamp = np.asarray((0, 100_000_000, 300_000_000), dtype=np.int64)
    midpoint = np.asarray((200_000_000_000, 200_500_000_000, 201_000_000_000), dtype=np.int64)
    markouts = _markouts(timestamp[:1], midpoint[:1], timestamp, midpoint)
    assert markouts[0, 0] == pytest.approx(1.0)
    assert markouts[0, 1] == pytest.approx(2.0)


def test_fixed_asof_view_never_uses_future_state() -> None:
    from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_only_train_baseline as baseline

    start, _ = baseline._session_windows("2025-03-03")["ASIA"]
    features = compact_features(timestamp=np.asarray((start + 50_000_000, start + 150_000_000), dtype=np.int64))
    fixed = _fixed_view("2025-03-03", features)
    early = np.flatnonzero(fixed.timestamp_ns <= start + 200_000_000)
    assert fixed.timestamp_ns[early].tolist() == [start + 100_000_000, start + 200_000_000]
    assert fixed.state_index[early].tolist() == [0, 1]


def test_optimized_barrier_counts_equal_naive_first_touch() -> None:
    from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_only_train_baseline as baseline

    start, _ = baseline._session_windows("2025-03-03")["ASIA"]
    timestamp = np.asarray((start, start + 50_000_000, start + 150_000_000, start + 250_000_000), dtype=np.int64)
    midpoint = np.asarray((200_000_000_000, 200_500_000_000, 199_500_000_000, 201_000_000_000), dtype=np.int64)
    features = compact_features(timestamp=timestamp, midpoint=midpoint)
    counts = _barrier_counts(features, _fixed_view("2025-03-03", features))
    outcomes = []
    states = [(int(ts), float(mid) / 2_000_000_000.0, "ASIA") for ts, mid in zip(timestamp, midpoint)]
    for index in range(timestamp.size):
        outcomes.append(first_touch_outcome(int(timestamp[index]), float(midpoint[index]) / 2_000_000_000.0,
                                            "ASIA", 1, states[index + 1:]))
    assert int(counts[0, 0, 0, 0]) == len(outcomes)
    assert int(counts[0, 0, 0, 1]) == sum(value > 0 for value in outcomes)
    assert int(counts[0, 0, 0, 2]) == sum(value < 0 for value in outcomes)


def test_cache_invalidates_on_source_hash_or_feature_version(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    feature_path, manifest_path = _feature_cache_paths(root, "2025-03-03")
    features = compact_features(timestamp=np.asarray((1, 2), dtype=np.int64))
    _atomic_npz(feature_path, timestamp_ns=features.timestamp_ns, session=features.session,
                mid_sum_raw=features.mid_sum_raw, microprice=features.microprice,
                depth_sum=features.depth_sum, contribution=features.contribution, event_sample=features.event_sample)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"status": "COMPLETE", "feature_cache_version": FEATURE_CACHE_VERSION,
                                         "feature_semantic_sha256": __import__(
                                             "research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_event_study",
                                             fromlist=["_cache_semantic_sha256"])._cache_semantic_sha256(),
                                         "source_sha256": "right", "raw_rows": 2, "relevant_state_events": 2}))
    assert _load_feature_cache(root, "2025-03-03", "right") is not None
    assert _load_feature_cache(root, "2025-03-03", "wrong") is None
    payload = json.loads(manifest_path.read_text()); payload["feature_cache_version"] = "wrong"
    manifest_path.write_text(json.dumps(payload))
    assert _load_feature_cache(root, "2025-03-03", "right") is None


def test_parallel_and_serial_aggregation_order_are_identical() -> None:
    cube = __import__(
        "research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_event_study",
        fromlist=["StatsCube"]).StatsCube()
    cube.count[0, 0, 0, 0] = 3
    payload = {"date": "a", "event_view_samples": 3, "fixed_interval_view_samples": 0,
               "stats": cube.export(), "extreme_stats": __import__(
                   "research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_event_study",
                   fromlist=["StatsCube"]).StatsCube(buckets=4).export(),
               "control_count": np.zeros((2, 3, 3, 7), dtype=np.int64).tolist(),
               "control_sum": np.zeros((2, 3, 3, 7), dtype=np.float64).tolist(),
               "regime_count": np.zeros((2, 3, 3, 7), dtype=np.int64).tolist(),
               "regime_sum": np.zeros((2, 3, 3, 7), dtype=np.float64).tolist(),
               "barrier_count": np.zeros((2, 72, 3, 4), dtype=np.int64).tolist()}
    serial = _aggregate_date_payloads([payload, {**payload, "date": "b"}])[0].count
    parallel_merged_in_date_order = _aggregate_date_payloads([{**payload, "date": "a"}, {**payload, "date": "b"}])[0].count
    assert np.array_equal(serial, parallel_merged_in_date_order)
