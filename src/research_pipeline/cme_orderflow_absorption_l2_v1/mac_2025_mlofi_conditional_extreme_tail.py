"""TRAIN-only conditional extreme-tail MLOFI event study.

This module reads only the sealed compact caches produced by the completed
MLOFI event study.  It deliberately contains no entry, exit, PnL, or
optimization logic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_mlofi_event_study as base


RUN_ID = "CMEOrderflow_MLOFI_CONDITIONAL_EXTREME_TAIL_TRAIN_V1"
OUTPUT_ROOT = Path("research_runs") / RUN_ID
SOURCE_ROOT = Path("research_runs/CMEOrderflow_MLOFI_EVENT_STUDY_TRAIN_V1")
CACHE_ROOT = SOURCE_ROOT / "feature-cache"
TAIL_LABELS = ("0-2.5", "2.5-5", "5-10", "10-25", "25-40", "40-60",
               "60-75", "75-90", "90-95", "95-97.5", "97.5-100")
TAIL_PCTS = np.asarray((0.025, 0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95, 0.975), dtype=float)
WINDOWS_MS = (250, 500, 1_000, 2_000)
HORIZONS_MS = (100, 250, 500, 1_000, 2_000, 5_000)
PATH_WINDOWS_MS = (500, 1_000, 2_000, 5_000)
EXTREME_BUCKETS = (0, 1, 2, 8, 9, 10)
GROUPS = (
    "EUROPE_LOW_MEDIUM_VOL_LOW_MEDIUM_DEPTH",
    "EUROPE_LOW_MEDIUM_VOL_HIGH_DEPTH",
    "EUROPE_HIGH_VOL",
    "NY_LOW_MEDIUM_VOL",
    "UNCONDITIONAL",
)
GROUP_DEFINITIONS = {
    GROUPS[0]: "session=EUROPE and volatility_class in {LOW,MEDIUM} and depth_class in {LOW,MEDIUM}",
    GROUPS[1]: "session=EUROPE and volatility_class in {LOW,MEDIUM} and depth_class=HIGH",
    GROUPS[2]: "session=EUROPE and volatility_class=HIGH",
    GROUPS[3]: "session=NY and volatility_class in {LOW,MEDIUM}",
    GROUPS[4]: "all eligible sessions and all volatility/depth classes",
}
METRICS = ("count", "sum", "sumsq", "positive", "negative", "zero")
RESERVOIR = 256
TICK_SCALE = 2.0  # midpoint is stored as a half-tick integer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _splitmix64(values: np.ndarray, salt: int) -> np.ndarray:
    salt_u64 = np.uint64((int(salt) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    x = np.asarray(values, dtype=np.uint64) + salt_u64
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _summary_from_stats(cell: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    count, total, sumsq, positive, negative, zero = (float(v) for v in cell)
    if not count:
        return {"sample_count": 0}
    mean = total / count
    variance = max(0.0, sumsq / count - mean * mean)
    values = np.asarray(samples[np.isfinite(samples)], dtype=float)
    if values.size:
        q25, median, q75 = (float(x) for x in np.quantile(values, (0.25, 0.5, 0.75)))
        trimmed_values = values[(values >= np.quantile(values, 0.10)) & (values <= np.quantile(values, 0.90))]
        trimmed = float(trimmed_values.mean()) if trimmed_values.size else float(values.mean())
    else:
        q25 = median = q75 = trimmed = None
    se = math.sqrt(variance / count)
    return {
        "sample_count": int(count), "mean_markout_ticks": mean,
        "median_markout_ticks": median, "trimmed_mean_markout_ticks": trimmed,
        "p25_markout_ticks": q25, "p75_markout_ticks": q75,
        "std_markout_ticks": math.sqrt(variance), "standard_error": se,
        "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
        "positive_fraction": positive / count, "negative_fraction": negative / count,
        "zero_fraction": zero / count, "quantiles_are_reservoir_estimates": True,
    }


class TailAccumulator:
    """Exact moments plus deterministic bounded samples for quantiles."""

    def __init__(self, groups: int = len(GROUPS), views: int = 2) -> None:
        shape = (groups, views, len(WINDOWS_MS), len(TAIL_LABELS), len(HORIZONS_MS), len(METRICS))
        self.exact = np.zeros(shape, dtype=np.float64)
        cells = groups * views * len(WINDOWS_MS) * len(TAIL_LABELS) * len(HORIZONS_MS)
        self.priorities = np.full((cells, RESERVOIR), np.iinfo(np.uint64).max, dtype=np.uint64)
        self.samples = np.full((cells, RESERVOIR), np.nan, dtype=np.float32)
        self.shape = shape[:-1]

    def _cell(self, group: int, view: int, window: int, bucket: int, horizon: int) -> int:
        return int(np.ravel_multi_index((group, view, window, bucket, horizon), self.shape))

    def update(self, group: int, view: int, window: int, buckets: np.ndarray, values: np.ndarray,
               keys: np.ndarray) -> None:
        for bucket in range(len(TAIL_LABELS)):
            selected = buckets == bucket
            if not selected.any():
                continue
            for horizon in range(values.shape[1]):
                outcome = values[:, horizon]
                valid = selected & np.isfinite(outcome)
                if not valid.any():
                    continue
                value = outcome[valid]
                cell = self._cell(group, view, window, bucket, horizon)
                self.exact[group, view, window, bucket, horizon, 0] += value.size
                self.exact[group, view, window, bucket, horizon, 1] += value.sum()
                self.exact[group, view, window, bucket, horizon, 2] += np.square(value).sum()
                self.exact[group, view, window, bucket, horizon, 3] += np.count_nonzero(value > 0)
                self.exact[group, view, window, bucket, horizon, 4] += np.count_nonzero(value < 0)
                self.exact[group, view, window, bucket, horizon, 5] += np.count_nonzero(value == 0)
                self._sample(cell, value, keys[valid], group, view, window, bucket, horizon)

    def _sample(self, cell: int, values: np.ndarray, keys: np.ndarray, *salt_parts: int) -> None:
        priorities = _splitmix64(keys, 17 + sum((index + 1) * int(value) for index, value in enumerate(salt_parts)))
        if values.size > RESERVOIR:
            chosen = np.argpartition(priorities, RESERVOIR - 1)[:RESERVOIR]
            priorities, values = priorities[chosen], values[chosen]
        old_priority = self.priorities[cell]
        old_values = self.samples[cell]
        candidate_priority = np.concatenate((old_priority[old_priority != np.iinfo(np.uint64).max], priorities))
        candidate_values = np.concatenate((old_values[np.isfinite(old_values)], values.astype(np.float32, copy=False)))
        if candidate_priority.size > RESERVOIR:
            chosen = np.argpartition(candidate_priority, RESERVOIR - 1)[:RESERVOIR]
            candidate_priority, candidate_values = candidate_priority[chosen], candidate_values[chosen]
        self.priorities[cell] = np.iinfo(np.uint64).max
        self.samples[cell] = np.nan
        self.priorities[cell, :candidate_priority.size] = candidate_priority
        self.samples[cell, :candidate_values.size] = candidate_values

    def merge(self, other: "TailAccumulator") -> None:
        self.exact += other.exact
        for cell in range(self.priorities.shape[0]):
            priorities = np.concatenate((self.priorities[cell][self.priorities[cell] != np.iinfo(np.uint64).max],
                                          other.priorities[cell][other.priorities[cell] != np.iinfo(np.uint64).max]))
            values = np.concatenate((self.samples[cell][np.isfinite(self.samples[cell])],
                                     other.samples[cell][np.isfinite(other.samples[cell])]))
            if priorities.size > RESERVOIR:
                chosen = np.argpartition(priorities, RESERVOIR - 1)[:RESERVOIR]
                priorities, values = priorities[chosen], values[chosen]
            self.priorities[cell] = np.iinfo(np.uint64).max
            self.samples[cell] = np.nan
            self.priorities[cell, :priorities.size] = priorities
            self.samples[cell, :values.size] = values

    def rows(self, directional: "TailAccumulator | None" = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group, group_name in enumerate(GROUPS):
            for view, view_name in enumerate(("EVENT_VIEW", "FIXED_INTERVAL_VIEW")):
                for window, window_ms in enumerate(WINDOWS_MS):
                    for bucket, bucket_name in enumerate(TAIL_LABELS):
                        for horizon, horizon_ms in enumerate(HORIZONS_MS):
                            row = {"group": group_name, "view": view_name, "observation_window_ms": window_ms,
                                   "bucket": bucket_name, "horizon_ms": horizon_ms,
                                   "signed": _summary_from_stats(self.exact[group, view, window, bucket, horizon],
                                                                 self.samples[self._cell(group, view, window, bucket, horizon)])}
                            if directional is not None:
                                row["direction_normalized"] = _summary_from_stats(
                                    directional.exact[group, view, window, bucket, horizon],
                                    directional.samples[directional._cell(group, view, window, bucket, horizon)])
                            rows.append(row)
        return rows


def _calibration_extended(features: base.CompactFeatures) -> dict[str, Any]:
    values: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    remaining = base.PERCENTILE_CALIBRATION_EVENTS
    calibration_end = 0
    variant_indices = [base.VARIANT_INDEX[(5, "INVERSE_LEVEL", window, "DEPTH_NORMALIZED")] for window in WINDOWS_MS]
    for _, timestamp, _, flag, signals, global_index in base._iter_event_signal_batches(features):
        selected = np.flatnonzero(flag)
        take = min(remaining, selected.size)
        if take:
            current = selected[:take]
            values.append(signals[current][:, variant_indices])
            indices.append(global_index[current])
            calibration_end = int(timestamp[current[-1]])
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("insufficient causal calibration states")
    signal_values = np.concatenate(values)
    calibration_index = np.concatenate(indices)
    depth = base._denominators(np.asarray(features.array("depth_sum")[calibration_index]))[:, base.COMBO_INDEX[(5, "INVERSE_LEVEL")]]
    first_session = int(features.array("session")[calibration_index[0]])
    start, end = next((start, end) for code, start, end in base._session_slices(features) if code == first_session)
    timestamps = np.asarray(features.array("timestamp_ns")[start:end], dtype=np.int64)
    midpoints = np.asarray(features.array("mid_sum_raw")[start:end], dtype=np.int64)
    calibration_ts = np.asarray(features.array("timestamp_ns")[calibration_index], dtype=np.int64)
    calibration_mid = np.asarray(features.array("mid_sum_raw")[calibration_index], dtype=np.int64)
    momentum = base._momentum_ticks(calibration_ts, calibration_mid, timestamps, midpoints)
    quantiles = list(TAIL_PCTS)
    return {
        "calibration_event_count": int(signal_values.shape[0]),
        "calibration_end_timestamp_ns": calibration_end,
        "tail_percentile_cuts": {str(window): np.quantile(signal_values[:, index], quantiles).tolist()
                                  for index, window in enumerate(WINDOWS_MS)},
        "depth_percentile_cuts_q10_to_q90": np.quantile(depth, np.arange(0.1, 1.0, 0.1)).tolist(),
        "volatility_percentile_cuts_q10_to_q90": np.quantile(np.abs(momentum), np.arange(0.1, 1.0, 0.1)).tolist(),
        "depth_class_definition": "min(2, searchsorted(q10..q90, weighted inverse-level depth, right) * 3 // 10)",
        "volatility_class_definition": "min(2, searchsorted(q10..q90, abs(1-second momentum ticks), right) * 3 // 10)",
    }


def _classes(depth: np.ndarray, volatility: np.ndarray, calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    depth_class = np.minimum(2, np.searchsorted(np.asarray(calibration["depth_percentile_cuts_q10_to_q90"]), depth, side="right") * 3 // 10)
    vol_class = np.minimum(2, np.searchsorted(np.asarray(calibration["volatility_percentile_cuts_q10_to_q90"]), volatility, side="right") * 3 // 10)
    return depth_class.astype(np.int8), vol_class.astype(np.int8)


def _group_mask(group: int, session: np.ndarray, depth_class: np.ndarray, vol_class: np.ndarray) -> np.ndarray:
    europe = session == baseline.SESSION_ORDER.index("EUROPE")
    ny = session == baseline.SESSION_ORDER.index("NY")
    if group == 0:
        return europe & (vol_class < 2) & (depth_class < 2)
    if group == 1:
        return europe & (vol_class < 2) & (depth_class == 2)
    if group == 2:
        return europe & (vol_class == 2)
    if group == 3:
        return ny & (vol_class < 2)
    return np.ones(session.shape[0], dtype=bool)


def _bucket(signal: np.ndarray, cuts: list[float]) -> np.ndarray:
    # Existing decile semantics use strict greater-than cuts; equal values
    # stay in the lower interval, making the tail partition deterministic.
    return np.searchsorted(np.asarray(cuts), signal, side="left").astype(np.int8)


def _midpoint_half_ticks(mid_sum_raw: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(mid_sum_raw, dtype=np.float64) / 250_000_000.0).astype(np.int32)


def _append_records(records: dict[tuple[int, int, int], list[np.ndarray]], *, view: int, window: int,
                    bucket: np.ndarray, signal: np.ndarray, timestamp: np.ndarray, midpoint: np.ndarray,
                    local_index: np.ndarray, keys: np.ndarray) -> None:
    for bucket_id in EXTREME_BUCKETS:
        selected = bucket == bucket_id
        if not selected.any():
            continue
        # time, midpoint-half-ticks, direction, local state index, priority key
        record = np.column_stack((timestamp[selected], midpoint[selected],
                                  np.where(signal[selected] >= 0, 1, -1), local_index[selected], keys[selected]))
        records.setdefault((view, window, bucket_id), []).append(record.astype(np.int64, copy=False))


def _markout_and_signal_rows(features: base.CompactFeatures, calibration: dict[str, Any], *, view: int,
                             timestamp: np.ndarray, midpoint: np.ndarray, session: int,
                             state_index: np.ndarray, signals: np.ndarray, event_flag: np.ndarray,
                             tail: TailAccumulator, directional: TailAccumulator,
                             records: dict[tuple[int, int, int], list[np.ndarray]]) -> None:
    eligible = event_flag.astype(bool) if view == 0 else np.ones(timestamp.shape[0], dtype=bool)
    if view == 0:
        eligible &= timestamp > calibration["calibration_end_timestamp_ns"]
    else:
        eligible &= timestamp > calibration["calibration_end_timestamp_ns"]
    if not eligible.any():
        return
    timestamp = timestamp[eligible]
    midpoint = midpoint[eligible]
    state_index = state_index[eligible]
    signals = signals[eligible]
    session_array = np.full(timestamp.shape[0], session, dtype=np.int8)
    all_ts = np.asarray(features.array("timestamp_ns"), dtype=np.int64)
    all_mid = np.asarray(features.array("mid_sum_raw"), dtype=np.int64)
    start, end = next((start, end) for code, start, end in base._session_slices(features) if code == session)
    outcomes = base._markouts(timestamp, midpoint, all_ts[start:end], all_mid[start:end])[:, :len(HORIZONS_MS)]
    depth = base._denominators(np.asarray(features.array("depth_sum")[state_index]))[:, base.COMBO_INDEX[(5, "INVERSE_LEVEL")]]
    momentum = base._momentum_ticks(timestamp, midpoint, all_ts[start:end], all_mid[start:end])
    depth_class, vol_class = _classes(depth, np.abs(momentum), calibration)
    keys = state_index.astype(np.int64, copy=False)
    midpoint_half = _midpoint_half_ticks(midpoint)
    for window_index, window_ms in enumerate(WINDOWS_MS):
        signal = signals[:, base.VARIANT_INDEX[(5, "INVERSE_LEVEL", window_ms, "DEPTH_NORMALIZED")]]
        buckets = _bucket(signal, calibration["tail_percentile_cuts"][str(window_ms)])
        directional_values = outcomes * np.where(signal[:, None] >= 0, 1.0, -1.0)
        for group in range(len(GROUPS)):
            mask = _group_mask(group, session_array, depth_class, vol_class)
            if mask.any():
                tail.update(group, view, window_index, buckets[mask], outcomes[mask], keys[mask])
                directional.update(group, view, window_index, buckets[mask], directional_values[mask], keys[mask])
        if _group_mask(0, session_array, depth_class, vol_class).any():
            primary = _group_mask(0, session_array, depth_class, vol_class)
            _append_records(records, view=view, window=window_index, bucket=buckets[primary], signal=signal[primary],
                            timestamp=timestamp[primary], midpoint=midpoint_half[primary],
                            local_index=state_index[primary] - start, keys=keys[primary])


class RangeRMQ:
    """Block range min/max with returned arg indices for cached price paths."""

    def __init__(self, values: np.ndarray, block_size: int = 512) -> None:
        self.values = np.asarray(values, dtype=np.int32)
        self.block = block_size
        self.blocks = (self.values.size + block_size - 1) // block_size
        self.block_min = np.empty(self.blocks, dtype=np.int32)
        self.block_max = np.empty(self.blocks, dtype=np.int32)
        self.block_min_i = np.empty(self.blocks, dtype=np.int32)
        self.block_max_i = np.empty(self.blocks, dtype=np.int32)
        self.prefix_min = np.empty_like(self.values)
        self.prefix_max = np.empty_like(self.values)
        self.prefix_min_i = np.empty_like(self.values)
        self.prefix_max_i = np.empty_like(self.values)
        self.suffix_min = np.empty_like(self.values)
        self.suffix_max = np.empty_like(self.values)
        self.suffix_min_i = np.empty_like(self.values)
        self.suffix_max_i = np.empty_like(self.values)
        for block in range(self.blocks):
            start, end = block * block_size, min(self.values.size, (block + 1) * block_size)
            part = self.values[start:end]
            min_local, max_local = int(np.argmin(part)), int(np.argmax(part))
            self.block_min[block], self.block_max[block] = part[min_local], part[max_local]
            self.block_min_i[block], self.block_max_i[block] = start + min_local, start + max_local
            self.prefix_min[start:end] = np.minimum.accumulate(part)
            self.prefix_max[start:end] = np.maximum.accumulate(part)
            self.suffix_min[start:end] = np.minimum.accumulate(part[::-1])[::-1]
            self.suffix_max[start:end] = np.maximum.accumulate(part[::-1])[::-1]
            size = end - start
            prefix_min_changes = np.flatnonzero(np.r_[True, self.prefix_min[start + 1:end] != self.prefix_min[start:end - 1]])
            prefix_max_changes = np.flatnonzero(np.r_[True, self.prefix_max[start + 1:end] != self.prefix_max[start:end - 1]])
            prefix_positions = np.arange(size)
            self.prefix_min_i[start:end] = start + prefix_min_changes[np.searchsorted(prefix_min_changes, prefix_positions, side="right") - 1]
            self.prefix_max_i[start:end] = start + prefix_max_changes[np.searchsorted(prefix_max_changes, prefix_positions, side="right") - 1]
            reversed_part = part[::-1]
            reversed_min = np.minimum.accumulate(reversed_part)
            reversed_max = np.maximum.accumulate(reversed_part)
            suffix_min_changes = np.flatnonzero(np.r_[True, reversed_min[1:] != reversed_min[:-1]])
            suffix_max_changes = np.flatnonzero(np.r_[True, reversed_max[1:] != reversed_max[:-1]])
            reversed_positions = np.arange(size)
            suffix_min_reversed_i = suffix_min_changes[np.searchsorted(suffix_min_changes, reversed_positions, side="right") - 1]
            suffix_max_reversed_i = suffix_max_changes[np.searchsorted(suffix_max_changes, reversed_positions, side="right") - 1]
            self.suffix_min_i[start:end] = start + size - 1 - suffix_min_reversed_i[::-1]
            self.suffix_max_i[start:end] = start + size - 1 - suffix_max_reversed_i[::-1]
        self.min_levels = [self.block_min]
        self.max_levels = [self.block_max]
        self.min_index_levels = [self.block_min_i]
        self.max_index_levels = [self.block_max_i]
        span = 1
        while span * 2 <= self.blocks:
            previous_min, previous_max = self.min_levels[-1], self.max_levels[-1]
            left_min, right_min = previous_min[:-span], previous_min[span:]
            left_max, right_max = previous_max[:-span], previous_max[span:]
            left_min_i, right_min_i = self.min_index_levels[-1][:-span], self.min_index_levels[-1][span:]
            left_max_i, right_max_i = self.max_index_levels[-1][:-span], self.max_index_levels[-1][span:]
            self.min_levels.append(np.minimum(left_min, right_min))
            self.max_levels.append(np.maximum(left_max, right_max))
            self.min_index_levels.append(np.where(left_min <= right_min, left_min_i, right_min_i))
            self.max_index_levels.append(np.where(left_max >= right_max, left_max_i, right_max_i))
            span *= 2

    def query(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_value = np.full(left.shape, np.iinfo(np.int32).min, dtype=np.int32)
        min_value = np.full(left.shape, np.iinfo(np.int32).max, dtype=np.int32)
        max_index = np.full(left.shape, -1, dtype=np.int32)
        min_index = np.full(left.shape, -1, dtype=np.int32)
        valid = right > left
        if not valid.any():
            return max_value, max_index, min_value, min_index
        same = valid & ((left // self.block) == ((right - 1) // self.block))
        for row in np.flatnonzero(same):
            lo, hi = int(left[row]), int(right[row])
            part = self.values[lo:hi]
            max_value[row], max_index[row] = int(part.max()), lo + int(part.argmax())
            min_value[row], min_index[row] = int(part.min()), lo + int(part.argmin())
        cross = valid & ~same
        if cross.any():
            rows = np.flatnonzero(cross)
            lo, hi = left[rows], right[rows]
            candidate_max = np.column_stack((self.suffix_max[lo], self.prefix_max[hi - 1]))
            candidate_max_i = np.column_stack((self.suffix_max_i[lo], self.prefix_max_i[hi - 1]))
            candidate_min = np.column_stack((self.suffix_min[lo], self.prefix_min[hi - 1]))
            candidate_min_i = np.column_stack((self.suffix_min_i[lo], self.prefix_min_i[hi - 1]))
            block_lo = lo // self.block + 1
            block_hi = (hi - 1) // self.block
            middle = block_lo < block_hi
            if middle.any():
                middle_rows = np.flatnonzero(middle)
                lengths = block_hi[middle] - block_lo[middle]
                levels = np.array([int(length).bit_length() - 1 for length in lengths], dtype=np.int64)
                for level in range(len(self.min_levels)):
                    selected = middle_rows[levels == level]
                    if not selected.size:
                        continue
                    span = 1 << level
                    first = block_lo[middle][levels == level]
                    last = block_hi[middle][levels == level] - span
                    first_max_i = self.max_index_levels[level][first]
                    last_max_i = self.max_index_levels[level][last]
                    first_min_i = self.min_index_levels[level][first]
                    last_min_i = self.min_index_levels[level][last]
                    first_max = self.max_levels[level][first]
                    last_max = self.max_levels[level][last]
                    first_min = self.min_levels[level][first]
                    last_min = self.min_levels[level][last]
                    update_max = (first_max > candidate_max[selected, 0]) | ((first_max == candidate_max[selected, 0]) & (first_max_i < candidate_max_i[selected, 0]))
                    candidate_max[selected, 0] = np.where(update_max, first_max, candidate_max[selected, 0])
                    candidate_max_i[selected, 0] = np.where(update_max, first_max_i, candidate_max_i[selected, 0])
                    update_min = (first_min < candidate_min[selected, 0]) | ((first_min == candidate_min[selected, 0]) & (first_min_i < candidate_min_i[selected, 0]))
                    candidate_min[selected, 0] = np.where(update_min, first_min, candidate_min[selected, 0])
                    candidate_min_i[selected, 0] = np.where(update_min, first_min_i, candidate_min_i[selected, 0])
                    update_max = (last_max > candidate_max[selected, 0]) | ((last_max == candidate_max[selected, 0]) & (last_max_i < candidate_max_i[selected, 0]))
                    candidate_max[selected, 0] = np.where(update_max, last_max, candidate_max[selected, 0])
                    candidate_max_i[selected, 0] = np.where(update_max, last_max_i, candidate_max_i[selected, 0])
                    update_min = (last_min < candidate_min[selected, 0]) | ((last_min == candidate_min[selected, 0]) & (last_min_i < candidate_min_i[selected, 0]))
                    candidate_min[selected, 0] = np.where(update_min, last_min, candidate_min[selected, 0])
                    candidate_min_i[selected, 0] = np.where(update_min, last_min_i, candidate_min_i[selected, 0])
            max_choice = np.argmax(candidate_max, axis=1)
            min_choice = np.argmin(candidate_min, axis=1)
            max_value[rows] = candidate_max[np.arange(rows.size), max_choice]
            min_value[rows] = candidate_min[np.arange(rows.size), min_choice]
            max_index[rows] = candidate_max_i[np.arange(rows.size), max_choice]
            min_index[rows] = candidate_min_i[np.arange(rows.size), min_choice]
        return max_value, max_index, min_value, min_index


def _append_path_stats(path: dict[str, Any], *, view: int, window: int, bucket: int,
                       mfe: np.ndarray, mae: np.ndarray, keys: np.ndarray) -> None:
    cell = (view, window, bucket)
    entry = path.setdefault("mfe_mae", {}).setdefault(str(cell), {
        "count": 0, "sum_mfe": 0.0, "sum_mae": 0.0, "sumsq_mfe": 0.0, "sumsq_mae": 0.0,
        "mfe_ge_1": 0, "mfe_ge_2": 0, "mfe_ge_4": 0, "mae_ge_1": 0, "mae_ge_2": 0, "mae_ge_4": 0,
        "mfe_samples": [], "mae_samples": [],
    })
    n = int(mfe.size)
    entry["count"] += n; entry["sum_mfe"] += float(mfe.sum()); entry["sum_mae"] += float(mae.sum())
    entry["sumsq_mfe"] += float(np.square(mfe).sum()); entry["sumsq_mae"] += float(np.square(mae).sum())
    entry["mfe_ge_1"] += int(np.count_nonzero(mfe >= 1)); entry["mfe_ge_2"] += int(np.count_nonzero(mfe >= 2)); entry["mfe_ge_4"] += int(np.count_nonzero(mfe >= 4))
    entry["mae_ge_1"] += int(np.count_nonzero(mae >= 1)); entry["mae_ge_2"] += int(np.count_nonzero(mae >= 2)); entry["mae_ge_4"] += int(np.count_nonzero(mae >= 4))
    # Deterministic bounded samples are enough for descriptive quantiles;
    # moments and threshold probabilities above remain exact.
    take = np.arange(0, n, max(1, n // 256), dtype=int)[:256]
    entry["mfe_samples"].extend(mfe[take].astype(float).tolist())
    entry["mae_samples"].extend(mae[take].astype(float).tolist())
    entry["mfe_samples"] = entry["mfe_samples"][:4096]
    entry["mae_samples"] = entry["mae_samples"][:4096]


def _path_analysis(features: base.CompactFeatures, records: dict[tuple[int, int, int], list[np.ndarray]], path: dict[str, Any]) -> None:
    for session, start, end in base._session_slices(features):
        session_records: dict[tuple[int, int, int], np.ndarray] = {}
        for key, pieces in records.items():
            view, window, bucket = key
            values = np.concatenate(pieces) if pieces else np.empty((0, 5), dtype=np.int64)
            values = values[(values[:, 3] >= start) & (values[:, 3] < end)] if values.size else values
            if values.size:
                values[:, 3] -= start
                session_records[key] = values
        if not session_records:
            continue
        timestamps = np.asarray(features.array("timestamp_ns")[start:end], dtype=np.int64)
        midpoint = _midpoint_half_ticks(np.asarray(features.array("mid_sum_raw")[start:end], dtype=np.int64))
        rmq = RangeRMQ(midpoint)
        for (view, window, bucket), values in session_records.items():
            obs_time = values[:, 0]
            current = values[:, 1].astype(np.int32)
            direction = values[:, 2].astype(np.int32)
            left = np.searchsorted(timestamps, obs_time, side="right")
            right = np.searchsorted(timestamps, obs_time + PATH_WINDOWS_MS[window] * 1_000_000, side="right")
            max_value, max_index, min_value, min_index = rmq.query(left, right)
            valid = max_index >= 0
            if not valid.any():
                continue
            favorable = np.where(direction[valid] > 0, max_value[valid] - current[valid], current[valid] - min_value[valid]) / TICK_SCALE
            adverse = np.where(direction[valid] > 0, current[valid] - min_value[valid], max_value[valid] - current[valid]) / TICK_SCALE
            _append_path_stats(path, view=view, window=window, bucket=bucket, mfe=favorable.astype(float),
                               mae=adverse.astype(float), keys=values[valid, 4])
            path.setdefault("time_to_mfe_mae", {}).setdefault(str((view, window, bucket)), {"mfe_ms": [], "mae_ms": []})
            times = path["time_to_mfe_mae"][str((view, window, bucket))]
            mfe_time = (timestamps[max_index[valid]] - obs_time[valid]) / 1_000_000.0
            mae_time = (timestamps[min_index[valid]] - obs_time[valid]) / 1_000_000.0
            times["mfe_ms"].extend(mfe_time.astype(float).tolist()[:4096])
            times["mae_ms"].extend(mae_time.astype(float).tolist()[:4096])
            path.setdefault("time_to_levels", {}).setdefault(str((view, window, bucket)), {
                "favorable_1_ms": [], "favorable_2_ms": [], "adverse_1_ms": [],
            })
            level_times = path["time_to_levels"][str((view, window, bucket))]
            path_start, path_end = left[valid], right[valid]
            path_direction = direction[valid]
            path_current = current[valid]
            targets = (
                (path_current + path_direction * 2, path_direction > 0, "favorable_1_ms"),
                (path_current + path_direction * 4, path_direction > 0, "favorable_2_ms"),
                (path_current - path_direction * 2, path_direction < 0, "adverse_1_ms"),
            )
            for target, upward, name in targets:
                crossing = _first_crossing_indices(rmq, target.astype(np.int32), path_start, path_end, upward=True)
                downward_crossing = _first_crossing_indices(rmq, target.astype(np.int32), path_start, path_end, upward=False)
                crossing = np.where(upward, crossing, downward_crossing)
                hit = crossing >= 0
                if hit.any():
                    level_times[name].extend(((timestamps[crossing[hit]] - obs_time[valid][hit]) / 1_000_000.0).astype(float).tolist()[:4096])


def _first_crossing_indices(rmq: RangeRMQ, target: np.ndarray, start: np.ndarray, end: np.ndarray,
                            *, upward: bool) -> np.ndarray:
    """Return the first cached state crossing each target, or -1.

    The search uses the vectorized RMQ for existence checks and then a bounded
    binary search for the first state.  This preserves first-touch semantics
    without one Python heap operation per observation.
    """
    result = np.full(target.shape, -1, dtype=np.int32)
    active = end > start
    if not active.any():
        return result
    rows = np.flatnonzero(active)
    lo0, hi0 = start[rows].astype(np.int64), end[rows].astype(np.int64)
    max_value, _, min_value, _ = rmq.query(lo0, hi0)
    hit = (max_value >= target[rows]) if upward else (min_value <= target[rows])
    if not hit.any():
        return result
    rows = rows[hit]
    lo = start[rows].astype(np.int64).copy()
    hi = end[rows].astype(np.int64).copy()
    wanted = target[rows]
    while np.any(lo < hi):
        middle = (lo + hi) // 2
        probe_hi = middle + 1
        max_value, _, min_value, _ = rmq.query(lo, probe_hi)
        reached = (max_value >= wanted) if upward else (min_value <= wanted)
        hi = np.where(reached, middle, hi)
        lo = np.where(reached, lo, middle + 1)
    result[rows] = lo.astype(np.int32)
    return result


def _barrier_time_records(features: base.CompactFeatures, records: dict[tuple[int, int, int], list[np.ndarray]], path: dict[str, Any]) -> None:
    """Exact first-touch directional barriers for selected tail records."""
    for session, start, end in base._session_slices(features):
        timestamps = np.asarray(features.array("timestamp_ns")[start:end], dtype=np.int64)
        midpoint = _midpoint_half_ticks(np.asarray(features.array("mid_sum_raw")[start:end], dtype=np.int64))
        rmq = RangeRMQ(midpoint)
        for (view, window, bucket), chunks in records.items():
            values = np.concatenate(chunks) if chunks else np.empty((0, 5), dtype=np.int64)
            if values.size:
                values = values[(values[:, 3] >= start) & (values[:, 3] < end)]
            if not values.size:
                continue
            values = values.copy()
            values[:, 3] -= start
            insert_at = values[:, 3].astype(np.int64) + 1 if view == 0 else np.searchsorted(timestamps, values[:, 0], side="right")
            active = insert_at < timestamps.size
            if not active.any():
                continue
            values = values[active]
            insert_at = insert_at[active]
            ends = np.minimum(
                np.searchsorted(timestamps, values[:, 0] + 10_000_000_000, side="right"),
                timestamps.size,
            ).astype(np.int64)
            current = values[:, 1].astype(np.int32)
            direction = values[:, 2].astype(np.int32)
            barrier = path.setdefault("barriers", {}).setdefault(
                str((view, window, bucket)), np.zeros((3, 4), dtype=np.int64).tolist()
            )
            barrier = np.asarray(barrier, dtype=np.int64)
            for level_index, level in enumerate((1, 2, 4)):
                favorable_target = current + direction * int(level * 2)
                adverse_target = current - direction * int(level * 2)
                favorable_upward = direction > 0
                adverse_upward = direction < 0
                favorable = np.where(
                    favorable_upward,
                    _first_crossing_indices(rmq, favorable_target, insert_at, ends, upward=True),
                    _first_crossing_indices(rmq, favorable_target, insert_at, ends, upward=False),
                )
                adverse = np.where(
                    adverse_upward,
                    _first_crossing_indices(rmq, adverse_target, insert_at, ends, upward=True),
                    _first_crossing_indices(rmq, adverse_target, insert_at, ends, upward=False),
                )
                favorable_hit = favorable >= 0
                adverse_hit = adverse >= 0
                tie = favorable_hit & adverse_hit & (favorable == adverse)
                positive = favorable_hit & (~adverse_hit | (favorable < adverse))
                negative = adverse_hit & (~favorable_hit | (adverse < favorable))
                barrier[level_index, 0] += values.shape[0]
                barrier[level_index, 1] += int(np.count_nonzero(positive))
                barrier[level_index, 2] += int(np.count_nonzero(negative))
                barrier[level_index, 3] += int(np.count_nonzero(tie))
            path["barriers"][str((view, window, bucket))] = barrier.tolist()


def _date_job(day: str) -> dict[str, Any]:
    manifest = json.loads((CACHE_ROOT / day / "cache-manifest.json").read_text())
    features = base._load_feature_cache(CACHE_ROOT, day, str(manifest["source_sha256"]))
    if features is None:
        raise RuntimeError(f"invalid compact cache: {day}")
    try:
        calibration = _calibration_extended(features)
        tail = TailAccumulator(); directional = TailAccumulator()
        records: dict[tuple[int, int, int], list[np.ndarray]] = {}
        for session, timestamp, midpoint, event_flag, signals, global_index in base._iter_event_signal_batches(features):
            _markout_and_signal_rows(features, calibration, view=0, timestamp=timestamp, midpoint=midpoint,
                                     session=session, state_index=global_index, signals=signals,
                                     event_flag=event_flag, tail=tail, directional=directional, records=records)
        fixed = base._fixed_view(day, features)
        fixed_signals, fixed_midpoint, _ = base._fixed_signals(features, fixed)
        for session, start, end in base._session_slices(features):
            selected = np.flatnonzero(fixed.session == session)
            if selected.size:
                _markout_and_signal_rows(features, calibration, view=1, timestamp=fixed.timestamp_ns[selected],
                                         midpoint=fixed_midpoint[selected], session=session,
                                         state_index=fixed.state_index[selected], signals=fixed_signals[selected],
                                         event_flag=np.ones(selected.size, dtype=bool), tail=tail,
                                         directional=directional, records=records)
        path: dict[str, Any] = {"day": day, "calibration": calibration}
        _path_analysis(features, records, path)
        _barrier_time_records(features, records, path)
        # Daily directional means for the primary regime; these remain exact
        # moments and are intentionally retained for stability diagnostics.
        primary_daily = {}
        for view in (0, 1):
            for window_index, window_ms in enumerate(WINDOWS_MS):
                for bucket in EXTREME_BUCKETS:
                    for horizon, horizon_ms in enumerate(HORIZONS_MS):
                        cell = tail._cell(0, view, window_index, bucket, horizon)
                        dcell = directional._cell(0, view, window_index, bucket, horizon)
                        primary_daily[str((view, window_ms, bucket, horizon_ms))] = {
                            "count": int(tail.exact[0, view, window_index, bucket, horizon, 0]),
                            "directional_count": int(directional.exact[0, view, window_index, bucket, horizon, 0]),
                            "directional_sum": float(directional.exact[0, view, window_index, bucket, horizon, 1]),
                        }
        path["primary_daily"] = primary_daily
        return {"day": day, "tail": tail, "directional": directional, "path": path}
    finally:
        features.close()


def _jsonable_job(job: dict[str, Any]) -> dict[str, Any]:
    return {"day": job["day"], "tail": job["tail"], "directional": job["directional"], "path": job["path"]}


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "day": job["day"],
        "tail_exact": job["tail"].exact.tolist(),
        "tail_priorities": job["tail"].priorities.tolist(),
        "tail_samples": job["tail"].samples.tolist(),
        "directional_exact": job["directional"].exact.tolist(),
        "directional_priorities": job["directional"].priorities.tolist(),
        "directional_samples": job["directional"].samples.tolist(),
        "path": job["path"],
    }


def _deserialize_job(payload: dict[str, Any]) -> dict[str, Any]:
    tail = TailAccumulator(); tail.exact = np.asarray(payload["tail_exact"]); tail.priorities = np.asarray(payload["tail_priorities"], dtype=np.uint64); tail.samples = np.asarray(payload["tail_samples"], dtype=np.float32)
    directional = TailAccumulator(); directional.exact = np.asarray(payload["directional_exact"]); directional.priorities = np.asarray(payload["directional_priorities"], dtype=np.uint64); directional.samples = np.asarray(payload["directional_samples"], dtype=np.float32)
    return {"day": payload["day"], "tail": tail, "directional": directional, "path": payload["path"]}


def _merge_path(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("mfe_mae", "time_to_mfe_mae"):
        for cell, value in source.get(key, {}).items():
            if key == "mfe_mae":
                item = target.setdefault(key, {}).setdefault(cell, {"count": 0, "sum_mfe": 0.0, "sum_mae": 0.0, "sumsq_mfe": 0.0, "sumsq_mae": 0.0, "mfe_ge_1": 0, "mfe_ge_2": 0, "mfe_ge_4": 0, "mae_ge_1": 0, "mae_ge_2": 0, "mae_ge_4": 0, "mfe_samples": [], "mae_samples": []})
                for metric in ("count", "sum_mfe", "sum_mae", "sumsq_mfe", "sumsq_mae", "mfe_ge_1", "mfe_ge_2", "mfe_ge_4", "mae_ge_1", "mae_ge_2", "mae_ge_4"):
                    item[metric] += value[metric]
                item["mfe_samples"] = (item["mfe_samples"] + value.get("mfe_samples", []))[:4096]
                item["mae_samples"] = (item["mae_samples"] + value.get("mae_samples", []))[:4096]
            else:
                item = target.setdefault(key, {}).setdefault(cell, {"mfe_ms": [], "mae_ms": []})
                item["mfe_ms"] = (item["mfe_ms"] + value.get("mfe_ms", []))[:4096]
                item["mae_ms"] = (item["mae_ms"] + value.get("mae_ms", []))[:4096]
    for cell, value in source.get("time_to_levels", {}).items():
        item = target.setdefault("time_to_levels", {}).setdefault(cell, {
            "favorable_1_ms": [], "favorable_2_ms": [], "adverse_1_ms": [],
        })
        for name in ("favorable_1_ms", "favorable_2_ms", "adverse_1_ms"):
            item[name] = (item[name] + value.get(name, []))[:4096]
    for cell, value in source.get("barriers", {}).items():
        target.setdefault("barriers", {}).setdefault(cell, np.zeros((3, 4), dtype=np.int64).tolist())
        target["barriers"][cell] = (np.asarray(target["barriers"][cell], dtype=np.int64) + np.asarray(value, dtype=np.int64)).tolist()


def _path_rows(path: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mfe_rows: list[dict[str, Any]] = []; time_rows: list[dict[str, Any]] = []; barrier_rows: list[dict[str, Any]] = []
    for cell, value in path.get("mfe_mae", {}).items():
        view, window, bucket = (int(part) for part in cell.strip("()").split(", "))
        count = value["count"]
        if not count: continue
        mfe = np.asarray(value["mfe_samples"], dtype=float); mae = np.asarray(value["mae_samples"], dtype=float)
        mfe_rows.append({"view": ("EVENT_VIEW", "FIXED_INTERVAL_VIEW")[view], "observation_window_ms": PATH_WINDOWS_MS[window], "bucket": TAIL_LABELS[bucket], "sample_count": count,
                         "mean_mfe_ticks": value["sum_mfe"] / count, "mean_mae_ticks": value["sum_mae"] / count,
                         "median_mfe_ticks": float(np.median(mfe)) if mfe.size else None, "median_mae_ticks": float(np.median(mae)) if mae.size else None,
                         "p_mfe_ge_1": value["mfe_ge_1"] / count, "p_mfe_ge_2": value["mfe_ge_2"] / count, "p_mfe_ge_4": value["mfe_ge_4"] / count,
                         "p_mae_ge_1": value["mae_ge_1"] / count, "p_mae_ge_2": value["mae_ge_2"] / count, "p_mae_ge_4": value["mae_ge_4"] / count,
                         "quantiles_are_bounded_samples": True})
    for cell, value in path.get("time_to_mfe_mae", {}).items():
        view, window, bucket = (int(part) for part in cell.strip("()").split(", "))
        time_rows.append({"view": ("EVENT_VIEW", "FIXED_INTERVAL_VIEW")[view], "observation_window_ms": PATH_WINDOWS_MS[window], "bucket": TAIL_LABELS[bucket],
                          "time_to_mfe_ms": _quantile_row(value.get("mfe_ms", [])), "time_to_mae_ms": _quantile_row(value.get("mae_ms", []))})
    for cell, value in path.get("time_to_levels", {}).items():
        view, window, bucket = (int(part) for part in cell.strip("()").split(", "))
        time_rows.append({"view": ("EVENT_VIEW", "FIXED_INTERVAL_VIEW")[view], "observation_window_ms": PATH_WINDOWS_MS[window], "bucket": TAIL_LABELS[bucket],
                          "time_to_favorable_1_ms": _quantile_row(value.get("favorable_1_ms", [])),
                          "time_to_favorable_2_ms": _quantile_row(value.get("favorable_2_ms", [])),
                          "time_to_adverse_1_ms": _quantile_row(value.get("adverse_1_ms", []))})
    for cell, value in path.get("barriers", {}).items():
        view, window, bucket = (int(part) for part in cell.strip("()").split(", "))
        counts = np.asarray(value, dtype=float)
        rows = []
        for level, threshold in enumerate((1, 2, 4)):
            total, positive, negative, tie = counts[level]
            rows.append({"barrier_ticks": threshold, "sample_count": int(total), "directional_win_probability": float(positive / total) if total else None,
                         "directional_loss_probability": float(negative / total) if total else None, "tie_or_same_event_probability": float(tie / total) if total else None,
                         "edge_over_50_percentage_points": float((positive / total - 0.5) * 100) if total else None})
        barrier_rows.append({"view": ("EVENT_VIEW", "FIXED_INTERVAL_VIEW")[view], "observation_window_ms": PATH_WINDOWS_MS[window], "bucket": TAIL_LABELS[bucket], "barriers": rows})
    return mfe_rows, time_rows, barrier_rows


def _quantile_row(values: list[float]) -> dict[str, Any]:
    if not values: return {"sample_count": 0}
    arr = np.asarray(values, dtype=float)
    return {"sample_count": int(arr.size), "p25": float(np.quantile(arr, .25)), "median": float(np.median(arr)), "p75": float(np.quantile(arr, .75)), "p95": float(np.quantile(arr, .95))}


def _daily_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for job in jobs:
        for view in (0, 1):
            for window_ms in (500, 1_000, 2_000):
                for horizon_ms in (500, 1_000, 2_000):
                    positive = job["path"].get("primary_daily", {}).get(str((view, window_ms, 9, horizon_ms)), {})
                    negative = job["path"].get("primary_daily", {}).get(str((view, window_ms, 0, horizon_ms)), {})
                    pmean = positive.get("directional_sum", 0.0) / positive.get("directional_count", 1) if positive.get("directional_count") else None
                    nmean = negative.get("directional_sum", 0.0) / negative.get("directional_count", 1) if negative.get("directional_count") else None
                    if pmean is not None or nmean is not None:
                        output.append({"date": job["day"], "view": ("EVENT_VIEW", "FIXED_INTERVAL_VIEW")[view], "observation_window_ms": window_ms, "horizon_ms": horizon_ms,
                                       "positive_tail_mean_directional_ticks": pmean, "negative_tail_mean_directional_ticks": nmean,
                                       "combined_mean_directional_ticks": float(np.nanmean([pmean, nmean])) if pmean is not None and nmean is not None else pmean or nmean,
                                       "positive_count": positive.get("directional_count", 0), "negative_count": negative.get("directional_count", 0)})
    return output


def _finalize_path(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, Any] = {}
    for job in jobs: _merge_path(merged, job["path"])
    return _path_rows(merged)


def run_study(*, workers: int = 2) -> dict[str, Any]:
    started = time.monotonic()
    source_summary = json.loads((SOURCE_ROOT / "summary.json").read_text())
    source_manifest = json.loads((SOURCE_ROOT / "run-manifest.json").read_text())
    if source_summary.get("status") != "PASS" or source_manifest.get("status") != "PASS":
        raise RuntimeError("completed MLOFI source run is not PASS")
    if source_summary.get("october_accessed") or source_summary.get("final_oos_accessed"):
        raise RuntimeError("source run violates TRAIN-only contract")
    dates = list(baseline.TRAIN_DATES)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:
        jobs = list(pool.map(_date_job, dates))
    jobs.sort(key=lambda job: job["day"])
    tail = TailAccumulator(); directional = TailAccumulator()
    for job in jobs:
        tail.merge(job["tail"]); directional.merge(job["directional"])
    tail_rows = tail.rows(directional)
    mfe_rows, time_rows, barrier_rows = _finalize_path(jobs)
    daily_rows = _daily_rows(jobs)
    source_hashes = {day: source_summary["source_sha256_by_date"][day] for day in dates}
    cache_hashes = {day: _sha256(CACHE_ROOT / day / "features.npz") for day in dates}
    sample_counts: dict[str, dict[str, dict[str, int]]] = {}
    sample_counts_by_horizon: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    active_dates: dict[str, dict[str, dict[str, list[str]]]] = {}
    for group_index, group_name in enumerate(GROUPS):
        sample_counts[group_name] = {}
        sample_counts_by_horizon[group_name] = {}
        active_dates[group_name] = {}
        for view_index, view_name in enumerate(("EVENT_VIEW", "FIXED_INTERVAL_VIEW")):
            sample_counts[group_name][view_name] = {}
            sample_counts_by_horizon[group_name][view_name] = {}
            active_dates[group_name][view_name] = {}
            for window_index, window_ms in enumerate(WINDOWS_MS):
                counts = {
                    str(horizon_ms): int(tail.exact[group_index, view_index, window_index, :, horizon_index, 0].sum())
                    for horizon_index, horizon_ms in enumerate(HORIZONS_MS)
                }
                sample_counts[group_name][view_name][str(window_ms)] = counts[str(HORIZONS_MS[0])]
                sample_counts_by_horizon[group_name][view_name][str(window_ms)] = counts
                active_dates[group_name][view_name][str(window_ms)] = [
                    job["day"] for job in jobs
                    if int(job["tail"].exact[group_index, view_index, window_index, :, 0, 0].sum()) > 0
                ]
    summary = {
        "run_id": RUN_ID, "status": "PASS", "source_run_id": source_summary["run_id"],
        "source_summary_sha256": _sha256(SOURCE_ROOT / "summary.json"),
        "source_run_manifest_sha256": _sha256(SOURCE_ROOT / "run-manifest.json"),
        "source_sha256_by_date": source_hashes, "feature_cache_sha256_by_date": cache_hashes,
        "train_dates": dates, "train_date_count": len(dates), "conditional_regime": GROUP_DEFINITIONS[GROUPS[0]],
        "group_definitions": GROUP_DEFINITIONS, "tail_labels": list(TAIL_LABELS), "tail_percentiles": TAIL_PCTS.tolist(),
        "observation_windows_ms": list(WINDOWS_MS), "forward_horizons_ms": list(HORIZONS_MS), "path_windows_ms": list(PATH_WINDOWS_MS),
        "calibration": "per-date first 100,000 chronological event samples; current observation excluded from its own thresholding; no later date is used",
        "sample_counts": sample_counts,
        "sample_counts_by_horizon_ms": sample_counts_by_horizon,
        "active_dates": active_dates,
        "elapsed_seconds": time.monotonic() - started, "optimization_performed": False, "strategy_backtest_performed": False,
        "pnl_calculated": False, "october_accessed": False, "dec_jan_accessed": False, "validation_accessed": False,
        "final_oos_accessed": False, "data_downloaded": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_ROOT / "summary.json", summary)
    input_artifacts = [SOURCE_ROOT / name for name in ("summary.json", "run-manifest.json", "signal-definition.json", "variant-results.json", "regime-results.json", "daily-stability.json")]
    _atomic_json(OUTPUT_ROOT / "run-manifest.json", {
        **summary,
        "input_artifacts": [str(path) for path in input_artifacts],
        "input_artifact_sha256": {str(path): _sha256(path) for path in input_artifacts},
    })
    _atomic_json(OUTPUT_ROOT / "tail-buckets.json", {"rows": tail_rows, "calibration_by_date": [job["path"]["calibration"] for job in jobs]})
    _atomic_json(OUTPUT_ROOT / "barrier-results.json", {"rows": barrier_rows, "barrier_horizon_ms": 10_000})
    _atomic_json(OUTPUT_ROOT / "mae-mfe-results.json", {"rows": mfe_rows, "path_windows_ms": list(PATH_WINDOWS_MS)})
    _atomic_json(OUTPUT_ROOT / "time-to-move.json", {"rows": time_rows, "definition": "first cached event state reaching the directional threshold; MFE/MAE timing uses the arg-extreme state in the path window"})
    _atomic_json(OUTPUT_ROOT / "daily-stability.json", {"rows": daily_rows, "near_zero_definition": "absolute combined directional markout < 0.05 ticks"})
    controls = []
    for group in GROUPS:
        for view in ("EVENT_VIEW", "FIXED_INTERVAL_VIEW"):
            for window in WINDOWS_MS:
                for bucket in (TAIL_LABELS[0], TAIL_LABELS[1], TAIL_LABELS[2], TAIL_LABELS[8], TAIL_LABELS[9], TAIL_LABELS[10]):
                    rows = [row for row in tail_rows if row["group"] == group and row["view"] == view and row["observation_window_ms"] == window and row["bucket"] == bucket and row["horizon_ms"] == 500]
                    if rows: controls.append(rows[0])
    _atomic_json(OUTPUT_ROOT / "regime-comparison.json", {"primary_and_controls": controls, "groups": GROUP_DEFINITIONS, "daily_rows": daily_rows})
    _atomic_json(OUTPUT_ROOT / "signal-definition.json", {"run_id": RUN_ID, "anchor": "TOP_5|INVERSE_LEVEL|DEPTH_NORMALIZED", "direction_normalization": "positive MLOFI uses future-current; negative MLOFI uses current-future", "regime_definition": GROUP_DEFINITIONS, "barrier": "first touch within 10 seconds; same-event opposing touch is tie", "no_strategy_or_pnl": True})
    report = [f"# {RUN_ID}", "", "TRAIN-only conditional descriptive tail study; no strategy, PnL, optimization, or non-TRAIN access.", "", f"Primary regime: {GROUP_DEFINITIONS[GROUPS[0]]}", "", "See JSON artifacts for complete tail, barrier, path, timing, and daily tables.", ""]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    summary = run_study(workers=args.workers)
    print(json.dumps({"MLOFI_CONDITIONAL_TAIL_STUDY": "PASS", "artifact_root": str(OUTPUT_ROOT), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
