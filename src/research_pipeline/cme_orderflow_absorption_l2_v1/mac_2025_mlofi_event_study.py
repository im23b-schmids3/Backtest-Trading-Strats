"""TRAIN-only multi-level order-flow imbalance event study.

This module is deliberately not a strategy runner.  It consumes the sealed
MAC 2025 Candidate Tape V2 manifest as an input/provenance contract and reads
the corresponding validated native ES MBP-10 files to retain the ten visible
book levels needed for MLOFI.  No candidate filter, entry, stop, target, or
PnL calculation is used here.

The event accounting is intentionally price-keyed rather than rank-diffed:
an A/C/M record contributes the displayed-size delta at the explicitly
changed price and a T record contributes the execution size on the passive
side.  Untouched levels that move in rank are not treated as cancellations.
Thus the same execution is never counted once as a trade and again as a
book-size delta.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import mac_2025_es_only_train_baseline as baseline
from .mac_2025_candidate_tape import TAPE_VERSION


RUN_ID = "CMEOrderflow_MLOFI_EVENT_STUDY_TRAIN_V1"
EXPECTED_TAPE_VERSION = "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE"
FEATURE_CACHE_VERSION = "mlofi-canonical-features-v4-price-keyed-ofi-asof-fixed"
PERCENTILE_CALIBRATION_EVENTS = 100_000
FEATURE_BLOCK_ROWS = 250_000
MAX_BARRIER_HORIZON_NS = 10_000_000_000
FIXED_INTERVAL_NS = 100_000_000
TICK_POINTS = 0.25
LEVELS = (3, 5, 10)
WINDOWS_MS = (250, 500, 1_000, 2_000)
HORIZONS_MS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000)
WEIGHTING = {
    "EQUAL": tuple(1.0 for _ in range(10)),
    "INVERSE_LEVEL": tuple(1.0 / (i + 1) for i in range(10)),
    # Fixed, interpretable half-life by displayed level; not optimized.
    "EXPONENTIAL_DECAY": tuple(0.5 ** i for i in range(10)),
}
DECILE_LABELS = ("0-10", "10-20", "20-30", "30-40", "40-50", "50-60",
                 "60-70", "70-80", "80-90", "90-100")
EXTREME_LABELS = ("0-2.5", "2.5-5", "95-97.5", "97.5-100")
MAIN_VARIANT = (5, "INVERSE_LEVEL", 500, "DEPTH_NORMALIZED")


class MLOFIError(RuntimeError):
    """Input, accounting, or artifact contract failure."""


@dataclass(frozen=True)
class BookSnapshot:
    timestamp_ns: int
    bid_prices: tuple[int, ...]
    bid_sizes: tuple[int, ...]
    ask_prices: tuple[int, ...]
    ask_sizes: tuple[int, ...]

    @property
    def executable(self) -> bool:
        return bool(self.bid_prices and self.ask_prices and self.ask_prices[0] > self.bid_prices[0])

    @property
    def mid(self) -> float:
        return (self.bid_prices[0] + self.ask_prices[0]) / 2_000_000_000.0

    def side(self, side: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.bid_prices, self.bid_sizes) if side == "B" else (self.ask_prices, self.ask_sizes)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def snapshot_from_row(row: Any, timestamp_ns: int | None = None) -> BookSnapshot:
    """Build a ten-level snapshot from one native DBN ndarray row."""
    def side(prefix: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        prices: list[int] = []
        sizes: list[int] = []
        for i in range(10):
            price = int(row[f"{prefix}_px_{i:02d}"])
            size = int(row[f"{prefix}_sz_{i:02d}"])
            if price <= 0 or size <= 0:
                continue
            prices.append(price)
            sizes.append(size)
        return tuple(prices), tuple(sizes)
    bids, bid_sizes = side("bid")
    asks, ask_sizes = side("ask")
    return BookSnapshot(int(timestamp_ns if timestamp_ns is not None else row["ts_recv"]),
                        bids, bid_sizes, asks, ask_sizes)


def _find_price(snapshot: BookSnapshot | None, side: str, price: int) -> tuple[int | None, int]:
    if snapshot is None or price <= 0:
        return None, 0
    prices, sizes = snapshot.side(side)
    try:
        index = prices.index(price)
    except ValueError:
        return None, 0
    return index, sizes[index]


def account_mbp_event(previous: BookSnapshot | None, current: BookSnapshot,
                      action: str, side: str, price: int, size: int) -> tuple[float, ...]:
    """Return signed OFI contribution by current/previous visible rank.

    Positive is bullish.  A/C/M uses the explicit price-level size delta;
    T uses the execution size on the passive side.  R contributes zero.
    Price-level disappearance caused only by a ladder shift is not inferred
    as a cancellation.
    """
    contribution = [0.0] * 10
    if action == "T" and side in {"B", "A"}:
        passive = "A" if side == "B" else "B"
        current_rank, _ = _find_price(current, passive, price)
        previous_rank, _ = _find_price(previous, passive, price)
        rank = current_rank if current_rank is not None else previous_rank
        if rank is not None and rank < 10:
            contribution[rank] = float(size if side == "B" else -size)
        return tuple(contribution)
    if action not in {"A", "C", "M"} or side not in {"B", "A"} or price <= 0:
        return tuple(contribution)
    current_rank, current_size = _find_price(current, side, price)
    previous_rank, previous_size = _find_price(previous, side, price)
    rank = current_rank if current_rank is not None else previous_rank
    if rank is None or rank >= 10:
        return tuple(contribution)
    delta = current_size - previous_size
    # A/C/M semantics are retained for auditability; the explicit size delta
    # is authoritative even when a provider emits an unusual modify record.
    contribution[rank] = float(delta if side == "B" else -delta)
    return tuple(contribution)


def weighted_resting_depth(snapshot: BookSnapshot, weights: Sequence[float]) -> float:
    """Current-time denominator: weighted mean of bid/ask displayed depth."""
    total = 0.0
    weight_total = 0.0
    for i, weight in enumerate(weights):
        bid = snapshot.bid_sizes[i] if i < len(snapshot.bid_sizes) else 0
        ask = snapshot.ask_sizes[i] if i < len(snapshot.ask_sizes) else 0
        total += float(weight) * (bid + ask) / 2.0
        weight_total += float(weight) if bid or ask else 0.0
    return total / weight_total if weight_total else 0.0


def signal_values(window_sums: Mapping[int, Sequence[float]], snapshot: BookSnapshot) -> tuple[float, ...]:
    """Return all 72 fixed MLOFI variants in deterministic variant order."""
    values: list[float] = []
    for levels in LEVELS:
        for weighting_name, weights in WEIGHTING.items():
            weights_used = weights[:levels]
            denominator = weighted_resting_depth(snapshot, weights_used)
            for window in WINDOWS_MS:
                raw = sum(float(window_sums[window][i]) * weights_used[i] for i in range(levels))
                values.append(raw)
                values.append(raw / denominator if denominator > 0 else 0.0)
    return tuple(values)


def variant_index(levels: int, weighting: str, window_ms: int, normalization: str) -> int:
    index = 0
    for current_levels in LEVELS:
        for current_weighting in WEIGHTING:
            for current_window in WINDOWS_MS:
                for current_norm in ("RAW", "DEPTH_NORMALIZED"):
                    if (current_levels, current_weighting, current_window, current_norm) == (levels, weighting, window_ms, normalization):
                        return index
                    index += 1
    raise KeyError((levels, weighting, window_ms, normalization))


def first_forward_midprice(observation_timestamp_ns: int, observation_session: str,
                           horizon_ms: int, future_states: Iterable[tuple[int, float, str]]) -> float | None:
    """Return the first same-session midpoint at/after a forward horizon."""
    target = observation_timestamp_ns + horizon_ms * 1_000_000
    for timestamp_ns, mid, session in future_states:
        if timestamp_ns < target:
            continue
        return float(mid) if session == observation_session else None
    return None


def forward_markout_ticks(observation_timestamp_ns: int, observation_mid: float,
                          observation_session: str, horizon_ms: int,
                          future_states: Iterable[tuple[int, float, str]]) -> float | None:
    mid = first_forward_midprice(observation_timestamp_ns, observation_session, horizon_ms, future_states)
    return None if mid is None else (mid - observation_mid) / TICK_POINTS


def first_touch_outcome(observation_timestamp_ns: int, observation_mid: float,
                        observation_session: str, barrier_ticks: float,
                        future_states: Iterable[tuple[int, float, str]],
                        max_horizon_ms: int = 10_000) -> int:
    """Return +1/-1 for first touch, 0 for tie/no touch/session cutoff."""
    expiry = observation_timestamp_ns + max_horizon_ms * 1_000_000
    up = observation_mid + barrier_ticks * TICK_POINTS
    down = observation_mid - barrier_ticks * TICK_POINTS
    for timestamp_ns, mid, session in future_states:
        if timestamp_ns <= observation_timestamp_ns:
            continue
        if timestamp_ns > expiry or session != observation_session:
            break
        bullish = mid >= up
        bearish = mid <= down
        if bullish and bearish:
            return 0
        if bullish:
            return 1
        if bearish:
            return -1
    return 0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.values: list[float] = []
        self.seen = 0
        self.random = random.Random(seed)
        self.cached: tuple[float, ...] = ()

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(float(value))
        else:
            slot = self.random.randrange(self.seen)
            if slot < self.capacity:
                self.values[slot] = float(value)
        if self.seen in (100, self.capacity) or self.seen % 100_000 == 0:
            self.cached = tuple(_quantile(self.values, q) for q in (0.025, 0.05, *[i / 10 for i in range(1, 10)], 0.95, 0.975))

    def decile(self, value: float) -> int | None:
        if self.seen < 100:
            return None
        bounds = self.cached[2:11]
        return min(9, bisect_right(bounds, value))

    def extreme(self, value: float) -> int | None:
        if self.seen < 100:
            return None
        if value <= self.cached[0]:
            return 0
        if value <= self.cached[1]:
            return 1
        if value >= self.cached[-1]:
            return 3
        if value >= self.cached[-2]:
            return 2
        return None


class StatsCube:
    """Compact sufficient statistics plus tiny deterministic value reservoirs."""
    def __init__(self, views: int = 2, buckets: int = 10, variants: int = 72) -> None:
        self.shape = (views, variants, buckets, len(HORIZONS_MS))
        self.count = np.zeros(self.shape, dtype=np.int64)
        self.sum = np.zeros(self.shape, dtype=np.float64)
        self.sumsq = np.zeros(self.shape, dtype=np.float64)
        self.positive = np.zeros(self.shape, dtype=np.int64)
        self.negative = np.zeros(self.shape, dtype=np.int64)
        self.zero = np.zeros(self.shape, dtype=np.int64)
        self.reservoirs = [[[] for _ in range(variants * buckets * len(HORIZONS_MS))] for _ in range(views)]

    def update(self, view: int, variant: int, bucket: int, horizon: int, value: float) -> None:
        idx = (view, variant, bucket, horizon)
        self.count[idx] += 1
        self.sum[idx] += value
        self.sumsq[idx] += value * value
        if value > 0:
            self.positive[idx] += 1
        elif value < 0:
            self.negative[idx] += 1
        else:
            self.zero[idx] += 1
        reservoir = self.reservoirs[view][(variant * self.shape[2] + bucket) * self.shape[3] + horizon]
        seen = int(self.count[idx])
        if len(reservoir) < 32:
            reservoir.append(float(value))
        elif seen % 100 == 0:
            slot = (seen * 1_103_515_245 + variant * 97 + bucket * 13 + horizon) % seen
            if slot < 32:
                reservoir[slot] = float(value)

    def update_observation(self, view: int, buckets: tuple[int | None, ...], horizon: int, value: float) -> None:
        """Vectorized bucket counters for one observation/horizon."""
        variants = np.arange(len(buckets), dtype=np.int64)
        valid = np.asarray([bucket is not None for bucket in buckets], dtype=bool)
        if not valid.any():
            return
        variant_values = variants[valid]
        bucket_values = np.asarray([bucket for bucket in buckets if bucket is not None], dtype=np.int64)
        np.add.at(self.count, (view, variant_values, bucket_values, horizon), 1)
        np.add.at(self.sum, (view, variant_values, bucket_values, horizon), value)
        np.add.at(self.sumsq, (view, variant_values, bucket_values, horizon), value * value)
        if value > 0:
            np.add.at(self.positive, (view, variant_values, bucket_values, horizon), 1)
        elif value < 0:
            np.add.at(self.negative, (view, variant_values, bucket_values, horizon), 1)
        else:
            np.add.at(self.zero, (view, variant_values, bucket_values, horizon), 1)
        # Robust summaries use sparse deterministic reservoirs; the exact
        # counters above remain complete for every observation.
        for variant, bucket in zip(variant_values.tolist(), bucket_values.tolist()):
            seen = int(self.count[view, variant, int(bucket), horizon])
            if seen <= 32 or seen % 100 == 0:
                self._reservoir_update(view, variant, int(bucket), horizon, value)

    def _reservoir_update(self, view: int, variant: int, bucket: int, horizon: int, value: float) -> None:
        reservoir = self.reservoirs[view][(variant * self.shape[2] + bucket) * self.shape[3] + horizon]
        seen = int(self.count[view, variant, bucket, horizon])
        if len(reservoir) < 32:
            reservoir.append(float(value))
        elif seen % 100 == 0:
            slot = (seen * 1_103_515_245 + variant * 97 + bucket * 13 + horizon) % seen
            if slot < 32:
                reservoir[slot] = float(value)

    def merge(self, other: "StatsCube") -> None:
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.positive += other.positive
        self.negative += other.negative
        self.zero += other.zero
        for view in range(2):
            for i, values in enumerate(other.reservoirs[view]):
                target = self.reservoirs[view][i]
                for value in values:
                    if len(target) < 32:
                        target.append(value)

    def export(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "count": self.count.tolist(), "sum": self.sum.tolist(), "sumsq": self.sumsq.tolist(),
            "positive": self.positive.tolist(), "negative": self.negative.tolist(), "zero": self.zero.tolist(),
            "reservoirs": self.reservoirs,
        }

    @classmethod
    def load(cls, payload: Mapping[str, Any]) -> "StatsCube":
        shape = tuple(payload["shape"])
        result = cls(views=shape[0], buckets=shape[2], variants=shape[1])
        for name in ("count", "sum", "sumsq", "positive", "negative", "zero"):
            setattr(result, name, np.asarray(payload[name]))
        result.reservoirs = payload["reservoirs"]
        return result


def _summary_from_cube(cube: StatsCube, view: int, variant: int, bucket: int, horizon: int) -> dict[str, Any]:
    count = int(cube.count[view, variant, bucket, horizon])
    if not count:
        return {"sample_count": 0}
    total = float(cube.sum[view, variant, bucket, horizon])
    mean = total / count
    variance = max(0.0, float(cube.sumsq[view, variant, bucket, horizon]) / count - mean * mean)
    se = math.sqrt(variance / count)
    values = sorted(cube.reservoirs[view][(variant * cube.shape[2] + bucket) * len(HORIZONS_MS) + horizon])
    trimmed = values[1:-1] if len(values) > 4 else values
    return {
        "sample_count": count, "mean_markout_ticks": mean,
        "median_markout_ticks": statistics.median(values) if values else None,
        "std_markout_ticks": math.sqrt(variance),
        "positive_markout_fraction": int(cube.positive[view, variant, bucket, horizon]) / count,
        "negative_markout_fraction": int(cube.negative[view, variant, bucket, horizon]) / count,
        "zero_markout_fraction": int(cube.zero[view, variant, bucket, horizon]) / count,
        "standard_error": se, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
        "trimmed_mean": sum(trimmed) / len(trimmed) if trimmed else None,
        "q25_markout_ticks": _quantile(values, 0.25), "q75_markout_ticks": _quantile(values, 0.75),
    }


@dataclass
class Observation:
    timestamp_ns: int
    mid: float
    signals: tuple[float, ...]
    buckets: tuple[int | None, ...]
    extremes: tuple[int | None, ...]
    view: int
    session: str
    momentum_class: int | None
    depth_class: int | None
    volatility_class: int | None
    markouts: list[float | None]
    barriers: list[list[int | None]]
    expiry_ns: int


def _percentile_class(reservoir: Reservoir, value: float) -> int | None:
    return reservoir.decile(value)


class DateProcessor:
    def __init__(self, day: str, *, seed: int = 20250301) -> None:
        self.day = day
        self.stats = StatsCube()
        self.extreme_stats = StatsCube(buckets=4)
        self.control_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
        self.control_sum = np.zeros_like(self.control_count, dtype=np.float64)
        self.regime_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
        self.regime_sum = np.zeros_like(self.regime_count, dtype=np.float64)
        self.barrier_count = np.zeros((2, 72, 3, 4), dtype=np.int64)
        self.reservoirs = [Reservoir(20_000, seed + i) for i in range(72)]
        self.depth_reservoir = Reservoir(20_000, seed + 1000)
        self.vol_reservoir = Reservoir(20_000, seed + 1001)
        self.window_sums = {window: np.zeros(10, dtype=np.float64) for window in WINDOWS_MS}
        self.contribution_history: dict[int, deque[tuple[int, tuple[float, ...]]]] = {
            window: deque() for window in WINDOWS_MS
        }
        self.pending_by_horizon: list[deque[Observation]] = [deque() for _ in HORIZONS_MS]
        self.expiry_heap: list[tuple[int, int, Observation]] = []
        self.up_heaps: list[list[tuple[float, int, Observation]]] = [[] for _ in (1, 2, 4)]
        self.down_heaps: list[list[tuple[float, int, Observation]]] = [[] for _ in (1, 2, 4)]
        self.next_id = 0
        self.previous: BookSnapshot | None = None
        self.current_session: str | None = None
        self.fixed_next_ns: int | None = None
        self.fixed_history: deque[tuple[int, float]] = deque(maxlen=32)
        self.event_samples = 0
        self.fixed_samples = 0
        self.unresolved_markouts = 0

    def _reset_session(self, session: str, start_ns: int) -> None:
        self.current_session = session
        self.previous = None
        self.window_sums = {window: np.zeros(10, dtype=np.float64) for window in WINDOWS_MS}
        for history in self.contribution_history.values():
            history.clear()
        self.fixed_next_ns = start_ns
        self.fixed_history.clear()
        self.pending_by_horizon = [deque() for _ in HORIZONS_MS]
        self.expiry_heap.clear(); self.up_heaps = [[] for _ in (1, 2, 4)]; self.down_heaps = [[] for _ in (1, 2, 4)]

    def _remove_old_contributions(self, timestamp_ns: int) -> None:
        for window in WINDOWS_MS:
            cutoff = timestamp_ns - window * 1_000_000
            history = self.contribution_history[window]
            while history and history[0][0] < cutoff:
                _, old = history.popleft()
                self.window_sums[window] -= old

    def _momentum(self, timestamp_ns: int, mid: float, horizon_ms: int) -> float:
        target = timestamp_ns - horizon_ms * 1_000_000
        prior = None
        for old_ts, old_mid in reversed(self.fixed_history):
            if old_ts <= target:
                prior = old_mid
                break
        return mid - prior if prior is not None else 0.0

    def _classes(self, signals: tuple[float, ...], snapshot: BookSnapshot) -> tuple[tuple[int | None, ...], tuple[int | None, ...], int | None, int | None, int | None]:
        buckets = tuple(self.reservoirs[i].decile(value) for i, value in enumerate(signals))
        extremes = tuple(self.reservoirs[i].extreme(value) for i, value in enumerate(signals))
        depth = weighted_resting_depth(snapshot, WEIGHTING["INVERSE_LEVEL"])
        depth_bucket = self.depth_reservoir.decile(depth)
        depth_class = None if depth_bucket is None else min(2, depth_bucket * 3 // 10)
        volatility = abs(self._momentum(snapshot.timestamp_ns, snapshot.mid, 1_000))
        volatility_bucket = self.vol_reservoir.decile(volatility)
        volatility_class = None if volatility_bucket is None else min(2, volatility_bucket * 3 // 10)
        momentum = self._momentum(snapshot.timestamp_ns, snapshot.mid, 1_000) / TICK_POINTS
        momentum_class = 0 if momentum < -1.0 else 2 if momentum > 1.0 else 1
        return buckets, extremes, momentum_class, depth_class, volatility_class

    def _add_reservoir_values(self, signals: tuple[float, ...], snapshot: BookSnapshot) -> None:
        for reservoir, value in zip(self.reservoirs, signals):
            reservoir.add(value)
        self.depth_reservoir.add(weighted_resting_depth(snapshot, WEIGHTING["INVERSE_LEVEL"]))
        self.vol_reservoir.add(abs(self._momentum(snapshot.timestamp_ns, snapshot.mid, 1_000)))

    def _new_observation(self, timestamp_ns: int, snapshot: BookSnapshot, view: int) -> None:
        signals = signal_values(self.window_sums, snapshot)
        buckets, extremes, momentum, depth, volatility = self._classes(signals, snapshot)
        observation = Observation(timestamp_ns, snapshot.mid, signals, buckets, extremes, view,
                                  self.current_session or "", momentum, depth, volatility,
                                  [None] * len(HORIZONS_MS), [[None] * 3 for _ in range(72)],
                                  timestamp_ns + MAX_BARRIER_HORIZON_NS)
        self.next_id += 1
        for index, horizon_ms in enumerate(HORIZONS_MS):
            self.pending_by_horizon[index].append(observation)
        heapq.heappush(self.expiry_heap, (observation.expiry_ns, self.next_id, observation))
        for barrier_index, barrier in enumerate((1, 2, 4)):
            heapq.heappush(self.up_heaps[barrier_index], (snapshot.mid + barrier * TICK_POINTS, self.next_id, observation))
            heapq.heappush(self.down_heaps[barrier_index], (-(snapshot.mid - barrier * TICK_POINTS), self.next_id, observation))
        self._add_reservoir_values(signals, snapshot)
        if view == 0:
            self.event_samples += 1
        else:
            self.fixed_samples += 1

    def _update_barriers(self, timestamp_ns: int, mid: float) -> None:
        def set_outcome(observation: Observation, barrier_index: int, outcome: int) -> None:
            for barriers in observation.barriers:
                barriers[barrier_index] = outcome

        for barrier_index, barrier in enumerate((1, 2, 4)):
            up = self.up_heaps[barrier_index]
            up_hits: list[Observation] = []
            while up and up[0][0] <= mid:
                _, _, observation = heapq.heappop(up)
                if timestamp_ns > observation.expiry_ns or observation.barriers[0][barrier_index] is not None:
                    continue
                up_hits.append(observation)
            down = self.down_heaps[barrier_index]
            down_hits: list[Observation] = []
            while down and -down[0][0] >= mid:
                _, _, observation = heapq.heappop(down)
                if timestamp_ns > observation.expiry_ns or observation.barriers[0][barrier_index] is not None:
                    continue
                down_hits.append(observation)
            down_ids = {id(observation) for observation in down_hits}
            for observation in up_hits:
                set_outcome(observation, barrier_index, 0 if id(observation) in down_ids else 1)
            up_ids = {id(observation) for observation in up_hits}
            for observation in down_hits:
                if id(observation) not in up_ids:
                    set_outcome(observation, barrier_index, -1)
        # Expiry is processed after touches at the same timestamp, so the
        # declared 10-second horizon is inclusive of the first state at that
        # timestamp.  Expired heap entries are stale-safe and cheap to pop.
        while self.expiry_heap and self.expiry_heap[0][0] < timestamp_ns:
            _, _, observation = heapq.heappop(self.expiry_heap)
            for barrier_index in range(3):
                if observation.barriers[0][barrier_index] is None:
                    set_outcome(observation, barrier_index, 0)

    def _resolve_markouts(self, timestamp_ns: int, mid: float) -> None:
        for horizon_index, pending in enumerate(self.pending_by_horizon):
            horizon_ns = HORIZONS_MS[horizon_index] * 1_000_000
            while pending and pending[0].timestamp_ns + horizon_ns <= timestamp_ns:
                observation = pending.popleft()
                if observation.markouts[horizon_index] is not None:
                    continue
                observation.markouts[horizon_index] = (mid - observation.mid) / TICK_POINTS
                self._record_observation(observation, horizon_index)

    def _record_observation(self, observation: Observation, horizon_index: int) -> None:
        value = float(observation.markouts[horizon_index])
        view = observation.view
        self.stats.update_observation(view, observation.buckets, horizon_index, value)
        self.extreme_stats.update_observation(view, observation.extremes, horizon_index, value)
        if observation.momentum_class is not None:
            main = variant_index(*MAIN_VARIANT)
            bucket = observation.buckets[main]
            if bucket is not None:
                signal_group = 0 if bucket <= 1 else 2 if bucket >= 8 else 1
                self.control_count[view, observation.momentum_class, signal_group, horizon_index] += 1
                self.control_sum[view, observation.momentum_class, signal_group, horizon_index] += value
        for regime_index, regime_class in enumerate((observation.momentum_class,
                                                      observation.depth_class,
                                                      observation.volatility_class)):
            if regime_class is not None:
                self.regime_count[view, regime_index, regime_class, horizon_index] += 1
                self.regime_sum[view, regime_index, regime_class, horizon_index] += value
        for variant, barriers in enumerate(observation.barriers):
            for barrier_index, outcome in enumerate(barriers):
                if outcome is not None:
                    self.barrier_count[view, variant, barrier_index, 0] += 1
                    self.barrier_count[view, variant, barrier_index, 1 if outcome > 0 else 2 if outcome < 0 else 3] += 1

    def feed(self, row: Any, session: str, session_start_ns: int) -> None:
        timestamp_ns = int(row["ts_recv"])
        if self.current_session != session:
            self._reset_session(session, session_start_ns)
        snapshot = snapshot_from_row(row, timestamp_ns)
        action = _text(row["action"])
        side = _text(row["side"])
        price = int(row["price"])
        size = int(row["size"])
        contribution = account_mbp_event(self.previous, snapshot, action, side, price, size)
        if snapshot.executable:
            self._update_barriers(timestamp_ns, snapshot.mid)
            self._resolve_markouts(timestamp_ns, snapshot.mid)
            if any(contribution):
                for window in WINDOWS_MS:
                    self.contribution_history[window].append((timestamp_ns, contribution))
                    self.window_sums[window] += contribution
            self._remove_old_contributions(timestamp_ns)
            if action in {"A", "C", "M", "T", "R"}:
                self._new_observation(timestamp_ns, snapshot, 0)
            if self.fixed_next_ns is None:
                self.fixed_next_ns = session_start_ns
            if timestamp_ns >= self.fixed_next_ns:
                self._new_observation(timestamp_ns, snapshot, 1)
                self.fixed_history.append((timestamp_ns, snapshot.mid))
                self.fixed_next_ns = timestamp_ns + FIXED_INTERVAL_NS
            self.previous = snapshot
        else:
            self.previous = None

    def finish(self) -> None:
        # Pending observations without a valid future state are intentionally
        # excluded from markout denominators, never filled with stale values.
        self.unresolved_markouts += sum(1 for pending in self.pending_by_horizon for _ in pending)

    def export(self) -> dict[str, Any]:
        return {
            "date": self.day, "event_view_samples": self.event_samples,
            "fixed_interval_view_samples": self.fixed_samples,
            "unresolved_markout_observations": self.unresolved_markouts,
            "stats": self.stats.export(), "extreme_stats": self.extreme_stats.export(),
            "control_count": self.control_count.tolist(), "control_sum": self.control_sum.tolist(),
            "regime_count": self.regime_count.tolist(), "regime_sum": self.regime_sum.tolist(),
            "barrier_count": self.barrier_count.tolist(),
        }


def _iso_ns(ns: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc).isoformat().replace("+00:00", "Z")


def _source_paths(data_root: Path) -> dict[str, Path]:
    _, requests = baseline._manifest(data_root)
    paths: dict[str, Path] = {}
    for day in baseline.TRAIN_DATES:
        path = baseline._source_path(data_root, requests, day)
        if not path.is_file():
            raise MLOFIError(f"missing TRAIN source: {path}")
        paths[day] = path
    return paths


def _validate_tape_contract(tape_manifest: Path, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    try:
        payload = json.loads(tape_manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MLOFIError(f"missing Candidate Tape V2 manifest: {tape_manifest}") from exc
    if payload.get("status") != "COMPLETE" or payload.get("tape_version") != EXPECTED_TAPE_VERSION:
        raise MLOFIError("Candidate Tape V2 manifest is not complete/current")
    if payload.get("validation_dates") is not None:
        raise MLOFIError("unexpected validation field in TRAIN tape manifest")
    expected = list(baseline.TRAIN_DATES)
    actual = payload.get("train_dates", payload.get("completed_dates", []))
    if actual != expected:
        raise MLOFIError("Candidate Tape V2 dates do not exactly match TRAIN")
    declared = payload.get("source_sha256_by_date", {})
    for day, digest in source_hashes.items():
        if declared and declared.get(day) != digest:
            raise MLOFIError(f"Candidate Tape/source hash mismatch: {day}")
    return payload


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats_rows(cube: StatsCube, *, extreme: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = EXTREME_LABELS if extreme else DECILE_LABELS
    bucket_count = len(labels)
    for view, view_name in enumerate(("EVENT_VIEW", "FIXED_INTERVAL_VIEW")):
        for variant in range(72):
            levels = LEVELS[variant // (3 * 4 * 2)]
            weighting = tuple(WEIGHTING)[(variant // (4 * 2)) % 3]
            window = WINDOWS_MS[(variant // 2) % 4]
            normalization = ("RAW", "DEPTH_NORMALIZED")[variant % 2]
            for bucket, label in enumerate(labels):
                for horizon, horizon_ms in enumerate(HORIZONS_MS):
                    row = _summary_from_cube(cube, view, variant, bucket, horizon)
                    row.update({"view": view_name, "levels": levels, "weighting": weighting,
                                "window_ms": window, "normalization": normalization,
                                "bucket": label, "horizon_ms": horizon_ms})
                    rows.append(row)
    return rows


def _aggregate_date_payloads(payloads: Sequence[Mapping[str, Any]]) -> tuple[StatsCube, StatsCube, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    stats = StatsCube(); extreme = StatsCube(buckets=4)
    control_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
    control_sum = np.zeros_like(control_count, dtype=np.float64)
    regime_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
    regime_sum = np.zeros_like(regime_count, dtype=np.float64)
    barrier_count = np.zeros((2, 72, 3, 4), dtype=np.int64)
    daily: dict[str, Any] = {}
    for payload in payloads:
        stats.merge(StatsCube.load(payload["stats"]))
        extreme.merge(StatsCube.load(payload["extreme_stats"]))
        control_count += np.asarray(payload["control_count"])
        control_sum += np.asarray(payload["control_sum"])
        regime_count += np.asarray(payload["regime_count"])
        regime_sum += np.asarray(payload["regime_sum"])
        barrier_count += np.asarray(payload["barrier_count"])
        daily[payload["date"]] = {"event_view_samples": payload["event_view_samples"],
                                   "fixed_interval_view_samples": payload["fixed_interval_view_samples"]}
    return stats, extreme, control_count, control_sum, regime_count, regime_sum, barrier_count, daily


def _daily_main_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    cube = StatsCube.load(payload["stats"])
    variant = variant_index(*MAIN_VARIANT)
    rows = []
    for horizon, horizon_ms in enumerate(HORIZONS_MS):
        bottom = _summary_from_cube(cube, 0, variant, 0, horizon)
        top = _summary_from_cube(cube, 0, variant, 9, horizon)
        rows.append({"date": payload["date"], "variant": "TOP_5|INVERSE_LEVEL|DEPTH_NORMALIZED",
                     "horizon_ms": horizon_ms,
                     "bottom_decile_mean_ticks": bottom.get("mean_markout_ticks"),
                     "top_decile_mean_ticks": top.get("mean_markout_ticks"),
                     "top_minus_bottom_ticks": (top.get("mean_markout_ticks", 0) - bottom.get("mean_markout_ticks", 0)
                                                if top.get("sample_count") and bottom.get("sample_count") else None),
                     "bottom_count": bottom.get("sample_count", 0), "top_count": top.get("sample_count", 0)})
    return rows


def _classification(daily_rows: Sequence[Mapping[str, Any]], bucket_rows: Sequence[Mapping[str, Any]]) -> str:
    spreads = [float(row["top_minus_bottom_ticks"]) for row in daily_rows
               if row.get("horizon_ms") in (500, 1_000, 2_000, 5_000) and row.get("top_minus_bottom_ticks") is not None]
    expected = sum(value > 0 for value in spreads)
    opposite = sum(value < 0 for value in spreads)
    if not spreads:
        return "INSUFFICIENT_EVIDENCE"
    positive_gradients = 0
    for horizon in HORIZONS_MS:
        rows = [row for row in bucket_rows if row["view"] == "EVENT_VIEW" and row["levels"] == 5
                and row["weighting"] == "INVERSE_LEVEL" and row["window_ms"] == 500
                and row["normalization"] == "DEPTH_NORMALIZED" and row["horizon_ms"] == horizon]
        means = [row.get("mean_markout_ticks") for row in rows if row.get("sample_count", 0)]
        if len(means) >= 5 and means[-1] > means[0]:
            positive_gradients += 1
    if expected / len(spreads) >= 0.70 and positive_gradients >= 4:
        return "STRONG_DIRECTIONAL_GRADIENT"
    if expected / len(spreads) <= 0.35 and positive_gradients <= 1:
        return "NO_STABLE_DIRECTIONAL_GRADIENT"
    if opposite and expected / len(spreads) < 0.60:
        return "MIXED_REGIME_DEPENDENT"
    return "WEAK_DIRECTIONAL_GRADIENT"


VARIANT_SPECS = tuple(
    (levels, weighting, window_ms, normalization)
    for levels in LEVELS
    for weighting in WEIGHTING
    for window_ms in WINDOWS_MS
    for normalization in ("RAW", "DEPTH_NORMALIZED")
)
COMBO_SPECS = tuple((levels, weighting) for levels in LEVELS for weighting in WEIGHTING)
COMBO_INDEX = {spec: index for index, spec in enumerate(COMBO_SPECS)}
VARIANT_INDEX = {spec: index for index, spec in enumerate(VARIANT_SPECS)}
COMBO_WEIGHTS = np.asarray(
    [list(WEIGHTING[weighting][:levels]) + [0.0] * (10 - levels) for levels, weighting in COMBO_SPECS],
    dtype=np.float64,
).T


@dataclass
class CompactFeatures:
    """Numeric, source-bound state sufficient for the whole MLOFI contract.

    The cache intentionally stores no order identifiers, order counts, raw
    action payloads, or full DBN rows.  It retains exactly the state required
    to reconstruct price-keyed L1--L10 OFI, the three depth normalizers, the
    midpoint/microprice diagnostics, and session-bounded forward outcomes.
    """

    timestamp_ns: np.ndarray
    session: np.ndarray
    mid_sum_raw: np.ndarray
    microprice: np.ndarray
    depth_sum: np.ndarray
    contribution: np.ndarray
    event_sample: np.ndarray
    count: int
    raw_rows: int
    work_root: Path | None = None

    def array(self, name: str) -> np.ndarray:
        return getattr(self, name)[:self.count]

    def close(self) -> None:
        if self.work_root is None:
            return
        for path in sorted(self.work_root.glob("*")):
            path.unlink(missing_ok=True)
        self.work_root.rmdir()
        self.work_root = None


@dataclass(frozen=True)
class FixedView:
    timestamp_ns: np.ndarray
    state_index: np.ndarray
    session: np.ndarray


def _feature_cache_paths(cache_root: Path, day: str) -> tuple[Path, Path]:
    root = cache_root / day
    return root / "features.npz", root / "cache-manifest.json"


def _cache_semantic_sha256() -> str:
    # The explicit accounting version is deliberate: changing Stage B output
    # formatting must not invalidate a source-feature cache, while changing
    # price-keyed accounting must bump FEATURE_CACHE_VERSION.
    return hashlib.sha256(FEATURE_CACHE_VERSION.encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_record_count(day: str, path: Path) -> int:
    """Read the sealed count from the acquisition manifest without DBN replay."""
    _, requests = baseline._manifest(baseline.DATA_ROOT)
    resolved = path.resolve()
    for row in requests.values():
        if not isinstance(row, dict):
            continue
        candidate = (baseline.DATA_ROOT / str(row.get("path", ""))).resolve()
        if candidate == resolved:
            count = row.get("verification", {}).get("record_count")
            if isinstance(count, int) and count > 0:
                return count
    raise MLOFIError(f"no sealed record count for feature extraction: {day} {path}")


def _load_feature_cache(cache_root: Path, day: str, source_sha256: str) -> CompactFeatures | None:
    feature_path, manifest_path = _feature_cache_paths(cache_root, day)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "COMPLETE" or manifest.get("feature_cache_version") != FEATURE_CACHE_VERSION
                or manifest.get("feature_semantic_sha256") != _cache_semantic_sha256()
                or manifest.get("source_sha256") != source_sha256):
            return None
        with np.load(feature_path, allow_pickle=False) as archive:
            names = ("timestamp_ns", "session", "mid_sum_raw", "microprice", "depth_sum", "contribution", "event_sample")
            if set(names) - set(archive.files):
                return None
            arrays = {name: np.asarray(archive[name]) for name in names}
        count = int(manifest["relevant_state_events"])
        if any(array.shape[0] != count for array in arrays.values()):
            return None
        return CompactFeatures(**arrays, count=count, raw_rows=int(manifest["raw_rows"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _persist_feature_cache(cache_root: Path, day: str, features: CompactFeatures, *, source_path: Path,
                           source_sha256: str, timings: Mapping[str, float]) -> None:
    feature_path, manifest_path = _feature_cache_paths(cache_root, day)
    _atomic_npz(
        feature_path,
        timestamp_ns=features.array("timestamp_ns"), session=features.array("session"),
        mid_sum_raw=features.array("mid_sum_raw"), microprice=features.array("microprice"),
        depth_sum=features.array("depth_sum"), contribution=features.array("contribution"),
        event_sample=features.array("event_sample"),
    )
    _json_write(manifest_path, {
        "status": "COMPLETE", "date": day, "feature_cache_version": FEATURE_CACHE_VERSION,
        "feature_semantic_sha256": _cache_semantic_sha256(), "source_path": str(source_path),
        "source_sha256": source_sha256, "raw_rows": features.raw_rows,
        "relevant_state_events": features.count,
        "event_view_samples": int(features.array("event_sample").sum()),
        "feature_columns": ["timestamp_ns", "session", "mid_sum_raw", "microprice", "depth_sum",
                            "contribution", "event_sample"],
        "stage_a_timings_seconds": dict(timings),
    })


def _session_codes(timestamp_ns: np.ndarray, windows: Mapping[str, tuple[int, int]]) -> np.ndarray:
    result = np.full(timestamp_ns.shape[0], -1, dtype=np.int8)
    for code, name in enumerate(baseline.SESSION_ORDER):
        start, end = windows[name]
        result[(timestamp_ns >= start) & (timestamp_ns < end)] = code
    return result


def _raw_book_arrays(batch: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.column_stack([batch[f"{prefix}_px_{index:02d}"] for index in range(10)]).astype(np.int64, copy=False),
        np.column_stack([batch[f"{prefix}_sz_{index:02d}"] for index in range(10)]).astype(np.int64, copy=False),
    )


def _packed_book_or_raise(prices: np.ndarray, sizes: np.ndarray, side: str) -> np.ndarray:
    """Return visible-level flags, rejecting a layout snapshot_from_row cannot vectorize safely."""
    visible = (prices > 0) & (sizes > 0)
    # Native MBP-10 records are packed by visible rank.  Rejecting an
    # unexpected sparse ladder is safer than silently treating raw column
    # rank as visible rank (the latter would change OFI semantics).
    if np.any(visible != np.cumprod(visible, axis=1).astype(bool)):
        raise MLOFIError(f"non-packed {side} MBP-10 ladder prevents causal compact extraction")
    return visible


def _visible_rank_and_size(prices: np.ndarray, sizes: np.ndarray, target_price: np.ndarray,
                           visible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match = (prices == target_price[:, None]) & visible
    found = match.any(axis=1)
    raw_rank = match.argmax(axis=1)
    size = np.take_along_axis(sizes, raw_rank[:, None], axis=1)[:, 0]
    # argmax returns zero when there is no match; the corresponding level's
    # size must not leak into the explicit price-keyed delta.
    size = np.where(found, size, 0)
    return raw_rank.astype(np.int64), size.astype(np.int64), found


def _vector_price_keyed_contribution(*, bid_px: np.ndarray, bid_sz: np.ndarray,
                                     ask_px: np.ndarray, ask_sz: np.ndarray,
                                     previous_bid_px: np.ndarray, previous_bid_sz: np.ndarray,
                                     previous_ask_px: np.ndarray, previous_ask_sz: np.ndarray,
                                     previous_valid: np.ndarray, action: np.ndarray,
                                     side: np.ndarray, price: np.ndarray, size: np.ndarray,
                                     current_valid: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of :func:`account_mbp_event` for packed MBP-10."""
    count = price.shape[0]
    result = np.zeros((count, 10), dtype=np.int32)
    bid_visible = _packed_book_or_raise(bid_px, bid_sz, "bid")
    ask_visible = _packed_book_or_raise(ask_px, ask_sz, "ask")
    previous_bid_visible = _packed_book_or_raise(previous_bid_px, previous_bid_sz, "previous_bid")
    previous_ask_visible = _packed_book_or_raise(previous_ask_px, previous_ask_sz, "previous_ask")
    side_bid = side == b"B"
    side_ask = side == b"A"
    book_action = np.isin(action, (b"A", b"C", b"M")) & (side_bid | side_ask) & current_valid
    trade_action = (action == b"T") & (side_bid | side_ask) & current_valid

    def assign(mask: np.ndarray, current_prices: np.ndarray, current_sizes: np.ndarray,
               current_visible: np.ndarray, prior_prices: np.ndarray, prior_sizes: np.ndarray,
               prior_visible: np.ndarray, signed_value: np.ndarray) -> None:
        current_rank, current_size, current_found = _visible_rank_and_size(
            current_prices, current_sizes, price, current_visible)
        prior_rank, prior_size, prior_found = _visible_rank_and_size(
            prior_prices, prior_sizes, price, prior_visible & previous_valid[:, None])
        rank = np.where(current_found, current_rank, prior_rank)
        eligible = mask & (current_found | prior_found) & (rank < 10)
        rows = np.flatnonzero(eligible)
        if rows.size:
            values = signed_value if signed_value.ndim else np.full(count, signed_value, dtype=np.int64)
            result[rows, rank[rows]] = values[rows].astype(np.int32, copy=False)

    # A/C/M: current displayed size minus the previous size at the explicit
    # price, signed bullish for bids and bearish for asks.
    bid_delta = np.zeros(count, dtype=np.int64)
    ask_delta = np.zeros(count, dtype=np.int64)
    for is_bid, current_prices, current_sizes, current_visible, prior_prices, prior_sizes, prior_visible, target in (
        (True, bid_px, bid_sz, bid_visible, previous_bid_px, previous_bid_sz, previous_bid_visible, bid_delta),
        (False, ask_px, ask_sz, ask_visible, previous_ask_px, previous_ask_sz, previous_ask_visible, ask_delta),
    ):
        current_rank, current_size, current_found = _visible_rank_and_size(current_prices, current_sizes, price, current_visible)
        _, prior_size, prior_found = _visible_rank_and_size(prior_prices, prior_sizes, price, prior_visible & previous_valid[:, None])
        target[:] = current_size - prior_size
        current_mask = book_action & (side_bid if is_bid else side_ask) & (current_found | prior_found)
        values = target if is_bid else -target
        assign(current_mask, current_prices, current_sizes, current_visible, prior_prices, prior_sizes, prior_visible, values)

    # T: use execution size directly on the passive side, never the book
    # delta, matching account_mbp_event exactly.
    assign(trade_action & side_bid, ask_px, ask_sz, ask_visible, previous_ask_px, previous_ask_sz,
           previous_ask_visible, size.astype(np.int64, copy=False))
    assign(trade_action & side_ask, bid_px, bid_sz, bid_visible, previous_bid_px, previous_bid_sz,
           previous_bid_visible, -size.astype(np.int64, copy=False))
    return result


def _extract_compact_features(day: str, path: Path, *, source_sha256: str, cache_root: Path | None) -> tuple[CompactFeatures, dict[str, float]]:
    """Stage A: exactly one DBN pass, one book reconstruction, no Python rows."""
    from databento import DBNStore

    timings = {name: 0.0 for name in ("dbn_decode", "book_reconstruction", "ofi_accounting", "feature_write", "cache_serialization")}
    record_count = _source_record_count(day, path)
    scratch_parent = cache_root.parent if cache_root is not None else Path(tempfile.gettempdir())
    scratch_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix=f".mlofi-{day}-", dir=scratch_parent))
    arrays = {
        "timestamp_ns": np.lib.format.open_memmap(work_root / "timestamp_ns.npy", mode="w+", dtype=np.int64, shape=(record_count,)),
        "session": np.lib.format.open_memmap(work_root / "session.npy", mode="w+", dtype=np.int8, shape=(record_count,)),
        "mid_sum_raw": np.lib.format.open_memmap(work_root / "mid_sum_raw.npy", mode="w+", dtype=np.int64, shape=(record_count,)),
        "microprice": np.lib.format.open_memmap(work_root / "microprice.npy", mode="w+", dtype=np.float64, shape=(record_count,)),
        "depth_sum": np.lib.format.open_memmap(work_root / "depth_sum.npy", mode="w+", dtype=np.uint32, shape=(record_count, 10)),
        "contribution": np.lib.format.open_memmap(work_root / "contribution.npy", mode="w+", dtype=np.int32, shape=(record_count, 10)),
        "event_sample": np.lib.format.open_memmap(work_root / "event_sample.npy", mode="w+", dtype=np.uint8, shape=(record_count,)),
    }
    windows = baseline._session_windows(day)
    offset = 0
    raw_rows = 0
    last: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int] | None = None
    iterator = DBNStore.from_file(path).to_ndarray(count=FEATURE_BLOCK_ROWS)
    while True:
        decoded_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        timings["dbn_decode"] += time.perf_counter() - decoded_started
        raw_rows += len(batch)
        book_started = time.perf_counter()
        timestamps = np.asarray(batch["ts_recv"], dtype=np.int64)
        bid_px, bid_sz = _raw_book_arrays(batch, "bid")
        ask_px, ask_sz = _raw_book_arrays(batch, "ask")
        bid_visible = _packed_book_or_raise(bid_px, bid_sz, "bid")
        ask_visible = _packed_book_or_raise(ask_px, ask_sz, "ask")
        bid_top = bid_px[:, 0]
        ask_top = ask_px[:, 0]
        valid = bid_visible[:, 0] & ask_visible[:, 0] & (ask_top > bid_top)
        sessions = _session_codes(timestamps, windows)
        in_scope = valid & (sessions >= 0)
        if last is None:
            previous_bid_px = np.zeros_like(bid_px); previous_bid_sz = np.zeros_like(bid_sz)
            previous_ask_px = np.zeros_like(ask_px); previous_ask_sz = np.zeros_like(ask_sz)
            previous_bid_px[1:], previous_bid_sz[1:] = bid_px[:-1], bid_sz[:-1]
            previous_ask_px[1:], previous_ask_sz[1:] = ask_px[:-1], ask_sz[:-1]
            previous_valid = np.zeros(len(batch), dtype=bool)
            previous_valid[1:] = valid[:-1]
            previous_session = -1
        else:
            previous_bid_px = np.empty_like(bid_px); previous_bid_sz = np.empty_like(bid_sz)
            previous_ask_px = np.empty_like(ask_px); previous_ask_sz = np.empty_like(ask_sz)
            previous_bid_px[0], previous_bid_sz[0], previous_ask_px[0], previous_ask_sz[0], last_valid, previous_session = last
            previous_bid_px[1:], previous_bid_sz[1:] = bid_px[:-1], bid_sz[:-1]
            previous_ask_px[1:], previous_ask_sz[1:] = ask_px[:-1], ask_sz[:-1]
            previous_valid = np.empty(len(batch), dtype=bool)
            previous_valid[0] = last_valid
            previous_valid[1:] = valid[:-1]
        prior_sessions = np.empty(len(batch), dtype=np.int8)
        prior_sessions[0] = previous_session
        prior_sessions[1:] = sessions[:-1]
        previous_valid &= prior_sessions == sessions
        actions = np.asarray(batch["action"])
        sides = np.asarray(batch["side"])
        prices = np.asarray(batch["price"], dtype=np.int64)
        sizes = np.asarray(batch["size"], dtype=np.int64)
        timings["book_reconstruction"] += time.perf_counter() - book_started
        ofi_started = time.perf_counter()
        contributions = _vector_price_keyed_contribution(
            bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
            previous_bid_px=previous_bid_px, previous_bid_sz=previous_bid_sz,
            previous_ask_px=previous_ask_px, previous_ask_sz=previous_ask_sz,
            previous_valid=previous_valid, action=actions, side=sides, price=prices, size=sizes,
            current_valid=in_scope,
        )
        event_sample = np.isin(actions, (b"A", b"C", b"M", b"T", b"R")) & in_scope
        timings["ofi_accounting"] += time.perf_counter() - ofi_started
        write_started = time.perf_counter()
        selected = np.flatnonzero(in_scope)
        end = offset + selected.size
        if end > record_count:
            raise MLOFIError(f"raw record count exceeded sealed manifest for {day}")
        top_depth = bid_sz[:, 0] + ask_sz[:, 0]
        micro = np.divide(
            ask_top.astype(np.float64) * bid_sz[:, 0] + bid_top.astype(np.float64) * ask_sz[:, 0],
            top_depth, out=np.zeros(len(batch), dtype=np.float64), where=top_depth > 0,
        )
        arrays["timestamp_ns"][offset:end] = timestamps[selected]
        arrays["session"][offset:end] = sessions[selected]
        arrays["mid_sum_raw"][offset:end] = bid_top[selected] + ask_top[selected]
        arrays["microprice"][offset:end] = micro[selected]
        arrays["depth_sum"][offset:end] = (bid_sz[selected] + ask_sz[selected]).astype(np.uint32, copy=False)
        arrays["contribution"][offset:end] = contributions[selected]
        arrays["event_sample"][offset:end] = event_sample[selected]
        offset = end
        last = (bid_px[-1].copy(), bid_sz[-1].copy(), ask_px[-1].copy(), ask_sz[-1].copy(), bool(valid[-1]), int(sessions[-1]))
        timings["feature_write"] += time.perf_counter() - write_started
    if raw_rows != record_count:
        raise MLOFIError(f"sealed record count mismatch for {day}: {raw_rows} != {record_count}")
    features = CompactFeatures(**arrays, count=offset, raw_rows=raw_rows, work_root=work_root)
    if cache_root is not None:
        persisted_started = time.perf_counter()
        _persist_feature_cache(cache_root, day, features, source_path=path, source_sha256=source_sha256, timings=timings)
        timings["cache_serialization"] = time.perf_counter() - persisted_started
    return features, timings


def _session_slices(features: CompactFeatures) -> Iterator[tuple[int, int, int]]:
    sessions = features.array("session")
    for code in range(len(baseline.SESSION_ORDER)):
        matching = np.flatnonzero(sessions == code)
        if not matching.size:
            continue
        start, end = int(matching[0]), int(matching[-1]) + 1
        if end - start != matching.size:
            raise MLOFIError("compact feature session rows are not contiguous")
        yield code, start, end


def _denominators(depth_sum: np.ndarray) -> np.ndarray:
    weights = COMBO_WEIGHTS
    numerator = (depth_sum.astype(np.float64, copy=False) / 2.0) @ weights
    active_weight = (depth_sum > 0).astype(np.float64) @ weights
    return np.divide(numerator, active_weight, out=np.zeros_like(numerator), where=active_weight > 0)


def _signals_from_rolling(rolling_by_window: Mapping[int, np.ndarray], depth_sum: np.ndarray) -> np.ndarray:
    count = depth_sum.shape[0]
    signals = np.empty((count, len(VARIANT_SPECS)), dtype=np.float64)
    denominator = _denominators(depth_sum)
    for window_ms, rolling in rolling_by_window.items():
        for combo, combo_index in COMBO_INDEX.items():
            raw_index = VARIANT_INDEX[(combo[0], combo[1], window_ms, "RAW")]
            normalized_index = raw_index + 1
            raw = rolling[:, combo_index]
            signals[:, raw_index] = raw
            signals[:, normalized_index] = np.divide(raw, denominator[:, combo_index], out=np.zeros(count), where=denominator[:, combo_index] > 0)
    return signals


def _iter_event_signal_batches(features: CompactFeatures, *, block_rows: int = FEATURE_BLOCK_ROWS) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield event-time signals with O(N) rolling cumulative accounting per session."""
    timestamp_all = features.array("timestamp_ns")
    contribution_all = features.array("contribution")
    depth_all = features.array("depth_sum")
    midpoint_all = features.array("mid_sum_raw")
    flags_all = features.array("event_sample")
    max_window_ns = max(WINDOWS_MS) * 1_000_000
    for session, session_start, session_end in _session_slices(features):
        running = np.zeros(len(COMBO_SPECS), dtype=np.float64)
        carry_ts = np.empty(0, dtype=np.int64)
        carry_cumulative = np.empty((0, len(COMBO_SPECS)), dtype=np.float64)
        before_carry = np.zeros(len(COMBO_SPECS), dtype=np.float64)
        for start in range(session_start, session_end, block_rows):
            end = min(session_end, start + block_rows)
            timestamp = np.asarray(timestamp_all[start:end], dtype=np.int64)
            contribution = np.asarray(contribution_all[start:end], dtype=np.float64)
            combo_contribution = contribution @ COMBO_WEIGHTS
            cumulative = np.cumsum(combo_contribution, axis=0) + running
            if carry_ts.size:
                combined_ts = np.concatenate((carry_ts, timestamp))
                combined_cumulative = np.concatenate((carry_cumulative, cumulative))
            else:
                combined_ts, combined_cumulative = timestamp, cumulative
            rolling: dict[int, np.ndarray] = {}
            for window_ms in WINDOWS_MS:
                starts = np.searchsorted(combined_ts, timestamp - window_ms * 1_000_000, side="left")
                previous = np.empty_like(cumulative)
                nonzero = starts > 0
                previous[nonzero] = combined_cumulative[starts[nonzero] - 1]
                previous[~nonzero] = before_carry
                rolling[window_ms] = cumulative - previous
            signals = _signals_from_rolling(rolling, np.asarray(depth_all[start:end]))
            cutoff = timestamp[-1] - max_window_ns
            keep = int(np.searchsorted(combined_ts, cutoff, side="left"))
            new_before = combined_cumulative[keep - 1].copy() if keep else before_carry.copy()
            carry_ts = combined_ts[keep:].copy()
            carry_cumulative = combined_cumulative[keep:].copy()
            before_carry = new_before
            running = cumulative[-1].copy()
            yield (session, timestamp, np.asarray(midpoint_all[start:end], dtype=np.int64),
                   np.asarray(flags_all[start:end], dtype=bool), signals, np.arange(start, end, dtype=np.int64))


def _markouts(timestamp: np.ndarray, midpoint: np.ndarray, session_timestamps: np.ndarray,
              session_midpoints: np.ndarray) -> np.ndarray:
    result = np.full((timestamp.shape[0], len(HORIZONS_MS)), np.nan, dtype=np.float64)
    for horizon, horizon_ms in enumerate(HORIZONS_MS):
        index = np.searchsorted(session_timestamps, timestamp + horizon_ms * 1_000_000, side="left")
        valid = index < session_timestamps.shape[0]
        if valid.any():
            result[valid, horizon] = (session_midpoints[index[valid]] - midpoint[valid]) / 500_000_000.0
    return result


def _percentile_bounds(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.quantile(signals, [index / 10.0 for index in range(1, 10)], axis=0),
        np.quantile(signals, [0.025, 0.05, 0.95, 0.975], axis=0),
    )


def _legacy_process_date(day: str, path: Path, *, seed: int = 20250301) -> dict[str, Any]:
    """Stream one validated MBP-10 date into compact sufficient statistics."""
    from databento import DBNStore
    windows = baseline._session_windows(day)
    processor = DateProcessor(day, seed=seed)
    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
        for row in batch:
            timestamp_ns = int(row["ts_recv"])
            session = next((name for name, (start, end) in windows.items() if start <= timestamp_ns < end), None)
            if session is None:
                continue
            processor.feed(row, session, windows[session][0])
    processor.finish()
    return processor.export()


def _append_reservoir_sample(cube: StatsCube, view: int, variant: int, bucket: int, horizon: int,
                             values: np.ndarray) -> None:
    """Keep a tiny deterministic descriptive sample without per-event objects."""
    reservoir = cube.reservoirs[view][(variant * cube.shape[2] + bucket) * cube.shape[3] + horizon]
    if len(reservoir) >= 32:
        return
    reservoir.extend(float(value) for value in values[:32 - len(reservoir)])


def _accumulate_cube(cube: StatsCube, *, view: int, signals: np.ndarray, markouts: np.ndarray,
                     decile_bounds: np.ndarray, extreme: bool = False) -> None:
    """Aggregate all 72 variants together, avoiding 72×7 Python loops.

    The flattened ``variant * bucket_count + bucket`` encoding turns each
    horizon into a small number of C-level bincount passes.  This is exact for
    count, sum, sum-of-squares, and sign counts; reservoirs remain deliberately
    tiny descriptive samples and are populated separately for the main output.
    """
    variant_count = signals.shape[1]
    bucket_count = cube.shape[2]
    if extreme:
        q025, q05, q95, q975 = decile_bounds
        buckets = np.full(signals.shape, -1, dtype=np.int8)
        buckets[signals <= q025[None, :]] = 0
        buckets[(signals > q025[None, :]) & (signals <= q05[None, :])] = 1
        buckets[(signals >= q95[None, :]) & (signals < q975[None, :])] = 2
        buckets[signals >= q975[None, :]] = 3
    else:
        # Broadcasting compares each signal only with its own nine causal
        # calibration bounds; no later observation enters those bounds.
        buckets = (signals[:, :, None] > decile_bounds.T[None, :, :]).sum(axis=2, dtype=np.int8)
    offsets = np.arange(variant_count, dtype=np.int64) * bucket_count
    for horizon in range(markouts.shape[1]):
        outcome = markouts[:, horizon]
        valid_rows = np.isfinite(outcome)
        if not valid_rows.any():
            continue
        selected_buckets = buckets[valid_rows]
        if extreme:
            selected = selected_buckets >= 0
            flat_index = (offsets[None, :] + selected_buckets)[selected]
            repeated_outcome = np.broadcast_to(outcome[valid_rows, None], selected_buckets.shape)[selected]
        else:
            flat_index = (offsets[None, :] + selected_buckets).ravel()
            repeated_outcome = np.broadcast_to(outcome[valid_rows, None], selected_buckets.shape).ravel()
        cells = variant_count * bucket_count
        count = np.bincount(flat_index, minlength=cells).reshape(variant_count, bucket_count)
        total = np.bincount(flat_index, weights=repeated_outcome, minlength=cells).reshape(variant_count, bucket_count)
        sumsq = np.bincount(flat_index, weights=repeated_outcome * repeated_outcome, minlength=cells).reshape(variant_count, bucket_count)
        positive = np.bincount(flat_index, weights=(repeated_outcome > 0), minlength=cells).reshape(variant_count, bucket_count)
        negative = np.bincount(flat_index, weights=(repeated_outcome < 0), minlength=cells).reshape(variant_count, bucket_count)
        zero = np.bincount(flat_index, weights=(repeated_outcome == 0), minlength=cells).reshape(variant_count, bucket_count)
        cube.count[view, :, :, horizon] += count
        cube.sum[view, :, :, horizon] += total
        cube.sumsq[view, :, :, horizon] += sumsq
        cube.positive[view, :, :, horizon] += positive.astype(np.int64, copy=False)
        cube.negative[view, :, :, horizon] += negative.astype(np.int64, copy=False)
        cube.zero[view, :, :, horizon] += zero.astype(np.int64, copy=False)


def _momentum_ticks(timestamp: np.ndarray, midpoint: np.ndarray, session_timestamps: np.ndarray,
                    session_midpoints: np.ndarray) -> np.ndarray:
    index = np.searchsorted(session_timestamps, timestamp - 1_000_000_000, side="right") - 1
    prior = midpoint.copy()
    valid = index >= 0
    if valid.any():
        prior[valid] = session_midpoints[index[valid]]
    return (midpoint - prior) / 500_000_000.0


def _accumulate_controls(*, view: int, signals: np.ndarray, markouts: np.ndarray, momentum_ticks: np.ndarray,
                         depth: np.ndarray, volatility: np.ndarray, main_bounds: np.ndarray,
                         depth_bounds: np.ndarray, volatility_bounds: np.ndarray,
                         control_count: np.ndarray, control_sum: np.ndarray,
                         regime_count: np.ndarray, regime_sum: np.ndarray) -> None:
    main = VARIANT_INDEX[MAIN_VARIANT]
    main_bucket = np.searchsorted(main_bounds, signals[:, main], side="right")
    signal_group = np.where(main_bucket <= 1, 0, np.where(main_bucket >= 8, 2, 1))
    momentum_class = np.where(momentum_ticks < -1.0, 0, np.where(momentum_ticks > 1.0, 2, 1))
    depth_class = np.minimum(2, np.searchsorted(depth_bounds, depth, side="right") * 3 // 10)
    volatility_class = np.minimum(2, np.searchsorted(volatility_bounds, volatility, side="right") * 3 // 10)
    for horizon in range(markouts.shape[1]):
        outcome = markouts[:, horizon]
        valid = np.isfinite(outcome)
        if not valid.any():
            continue
        pair = momentum_class[valid] * 3 + signal_group[valid]
        control_count[view, :, :, horizon] += np.bincount(pair, minlength=9).reshape(3, 3)
        control_sum[view, :, :, horizon] += np.bincount(pair, weights=outcome[valid], minlength=9).reshape(3, 3)
        for regime_index, classes in enumerate((momentum_class, depth_class, volatility_class)):
            values = classes[valid]
            regime_count[view, regime_index, :, horizon] += np.bincount(values, minlength=3)
            regime_sum[view, regime_index, :, horizon] += np.bincount(values, weights=outcome[valid], minlength=3)


def _fixed_view(day: str, features: CompactFeatures) -> FixedView:
    windows = baseline._session_windows(day)
    timestamps = features.array("timestamp_ns")
    target_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    session_parts: list[np.ndarray] = []
    for session, start, end in _session_slices(features):
        name = baseline.SESSION_ORDER[session]
        start_ns, end_ns = windows[name]
        targets = np.arange(start_ns, end_ns, FIXED_INTERVAL_NS, dtype=np.int64)
        local = np.searchsorted(timestamps[start:end], targets, side="right") - 1
        valid = local >= 0
        if valid.any():
            target_parts.append(targets[valid])
            index_parts.append((local[valid] + start).astype(np.int64, copy=False))
            session_parts.append(np.full(int(valid.sum()), session, dtype=np.int8))
    if not target_parts:
        raise MLOFIError(f"no fixed-interval as-of states for {day}")
    return FixedView(np.concatenate(target_parts), np.concatenate(index_parts), np.concatenate(session_parts))


def _fixed_signals(features: CompactFeatures, fixed: FixedView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive 100ms as-of signals without rereading raw MBP-10."""
    count = fixed.timestamp_ns.shape[0]
    result = np.empty((count, len(VARIANT_SPECS)), dtype=np.float64)
    timestamp_all = features.array("timestamp_ns")
    contribution_all = features.array("contribution")
    depth = np.asarray(features.array("depth_sum")[fixed.state_index])
    denominator = _denominators(depth)
    midpoint = np.asarray(features.array("mid_sum_raw")[fixed.state_index], dtype=np.int64)
    for session, start, end in _session_slices(features):
        which = np.flatnonzero(fixed.session == session)
        if not which.size:
            continue
        timestamp = np.asarray(timestamp_all[start:end], dtype=np.int64)
        local_index = fixed.state_index[which] - start
        target = fixed.timestamp_ns[which]
        contribution = np.asarray(contribution_all[start:end], dtype=np.float64)
        for combo, combo_index in COMBO_INDEX.items():
            instantaneous = contribution @ COMBO_WEIGHTS[:, combo_index]
            cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(instantaneous)))
            for window_ms in WINDOWS_MS:
                starts = np.searchsorted(timestamp, target - window_ms * 1_000_000, side="left")
                raw = cumulative[local_index + 1] - cumulative[starts]
                raw_variant = VARIANT_INDEX[(combo[0], combo[1], window_ms, "RAW")]
                result[which, raw_variant] = raw
                result[which, raw_variant + 1] = np.divide(
                    raw, denominator[which, combo_index], out=np.zeros(raw.shape[0]),
                    where=denominator[which, combo_index] > 0,
                )
    return result, midpoint, denominator[:, COMBO_INDEX[(5, "INVERSE_LEVEL")]]


def _barrier_counts(features: CompactFeatures, fixed: FixedView) -> np.ndarray:
    """Exact first-touch barriers with one shared numeric heap pass per session.

    An event observation is inserted *after* its current state is processed,
    so it can only see a future state.  Fixed observations are inserted just
    before the first later state, and at equal timestamps after that state,
    implementing the same strict future rule with as-of sampling.
    """
    total = np.zeros((2, 3, 4), dtype=np.int64)
    timestamp_all = features.array("timestamp_ns")
    midpoint_all = features.array("mid_sum_raw")
    event_flag_all = features.array("event_sample")
    offsets = (500_000_000, 1_000_000_000, 2_000_000_000)
    for session, start, end in _session_slices(features):
        timestamp = np.asarray(timestamp_all[start:end], dtype=np.int64)
        midpoint = np.asarray(midpoint_all[start:end], dtype=np.int64)
        event_flag = np.asarray(event_flag_all[start:end], dtype=bool)
        fixed_rows = np.flatnonzero(fixed.session == session)
        fixed_time = fixed.timestamp_ns[fixed_rows]
        state_count = end - start
        settled = np.zeros((state_count + fixed_rows.size, 3), dtype=bool)
        # Most observations share a small number of tick-aligned barrier
        # thresholds.  Heap keys are therefore *price levels*, with the
        # observation ids grouped behind each key, rather than six heap
        # entries per state event.  This preserves first-touch ordering while
        # removing tens of millions of redundant heap operations.
        up_heaps: list[list[int]] = [[] for _ in offsets]
        down_heaps: list[list[int]] = [[] for _ in offsets]
        up_waiting: list[dict[int, list[int]]] = [dict() for _ in offsets]
        down_waiting: list[dict[int, list[int]]] = [dict() for _ in offsets]

        def observation_time(identifier: int) -> int:
            return int(timestamp[identifier]) if identifier < state_count else int(fixed_time[identifier - state_count])

        def observation_view(identifier: int) -> int:
            return 0 if identifier < state_count else 1

        def add(identifier: int, observed_midpoint: int) -> None:
            view = observation_view(identifier)
            total[view, :, 0] += 1
            for barrier, offset in enumerate(offsets):
                up_target = int(observed_midpoint + offset)
                down_target = int(observed_midpoint - offset)
                up_ids = up_waiting[barrier].get(up_target)
                if up_ids is None:
                    up_waiting[barrier][up_target] = [identifier]
                    heapq.heappush(up_heaps[barrier], up_target)
                else:
                    up_ids.append(identifier)
                down_ids = down_waiting[barrier].get(down_target)
                if down_ids is None:
                    down_waiting[barrier][down_target] = [identifier]
                    heapq.heappush(down_heaps[barrier], -down_target)
                else:
                    down_ids.append(identifier)

        def resolve(identifier: int, barrier: int, outcome: int) -> None:
            if settled[identifier, barrier]:
                return
            settled[identifier, barrier] = True
            view = observation_view(identifier)
            total[view, barrier, 1 if outcome > 0 else 2 if outcome < 0 else 3] += 1

        fixed_cursor = 0
        for state_index, (now, current_midpoint) in enumerate(zip(timestamp.tolist(), midpoint.tolist())):
            while fixed_cursor < fixed_time.size and fixed_time[fixed_cursor] < now:
                prior_state = int(np.searchsorted(timestamp, fixed_time[fixed_cursor], side="right") - 1)
                if prior_state >= 0:
                    add(state_count + fixed_cursor, int(midpoint[prior_state]))
                fixed_cursor += 1
            for barrier in range(len(offsets)):
                up_hits: list[int] = []
                while up_heaps[barrier] and up_heaps[barrier][0] <= current_midpoint:
                    target = heapq.heappop(up_heaps[barrier])
                    identifiers = up_waiting[barrier].pop(target, ())
                    up_hits.extend(identifier for identifier in identifiers
                                   if not settled[identifier, barrier]
                                   and now <= observation_time(identifier) + MAX_BARRIER_HORIZON_NS)
                down_hits: list[int] = []
                while down_heaps[barrier] and -down_heaps[barrier][0] >= current_midpoint:
                    target = -heapq.heappop(down_heaps[barrier])
                    identifiers = down_waiting[barrier].pop(target, ())
                    down_hits.extend(identifier for identifier in identifiers
                                     if not settled[identifier, barrier]
                                     and now <= observation_time(identifier) + MAX_BARRIER_HORIZON_NS)
                down_set = set(down_hits)
                for identifier in up_hits:
                    resolve(identifier, barrier, 0 if identifier in down_set else 1)
                up_set = set(up_hits)
                for identifier in down_hits:
                    if identifier not in up_set:
                        resolve(identifier, barrier, -1)
            if event_flag[state_index]:
                add(state_index, int(current_midpoint))
            while fixed_cursor < fixed_time.size and fixed_time[fixed_cursor] == now:
                add(state_count + fixed_cursor, int(current_midpoint))
                fixed_cursor += 1
        # Any pending observation is an exact no-touch outcome at the session
        # boundary/end-of-file.  Counts are total - positive - negative - tie.
    total[:, :, 3] += total[:, :, 0] - total[:, :, 1] - total[:, :, 2] - total[:, :, 3]
    return np.repeat(total[:, None, :, :], len(VARIANT_SPECS), axis=1)


def _calibration(features: CompactFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    values: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    remaining = PERCENTILE_CALIBRATION_EVENTS
    calibration_end = 0
    for _, timestamp, _, flag, signals, global_index in _iter_event_signal_batches(features):
        selected = np.flatnonzero(flag)
        take = min(remaining, selected.size)
        if take:
            current = selected[:take]
            values.append(signals[current])
            indices.append(global_index[current])
            calibration_end = int(timestamp[current[-1]])
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise MLOFIError("insufficient event states for causal percentile calibration")
    signal_values = np.concatenate(values)
    calibration_index = np.concatenate(indices)
    depth = _denominators(np.asarray(features.array("depth_sum")[calibration_index]))[:, COMBO_INDEX[(5, "INVERSE_LEVEL")]]
    timestamp = np.asarray(features.array("timestamp_ns")[calibration_index], dtype=np.int64)
    midpoint = np.asarray(features.array("mid_sum_raw")[calibration_index], dtype=np.int64)
    # Calibration is from the first chronological event block.  It cannot use
    # October/OOS or any later TRAIN state to bucket an analysed observation.
    first_session = int(features.array("session")[calibration_index[0]])
    first_start, first_end = next((start, end) for code, start, end in _session_slices(features) if code == first_session)
    momentum = _momentum_ticks(timestamp, midpoint,
                               np.asarray(features.array("timestamp_ns")[first_start:first_end]),
                               np.asarray(features.array("mid_sum_raw")[first_start:first_end]))
    return (*_percentile_bounds(signal_values), np.quantile(depth, [index / 10.0 for index in range(1, 10)]),
            np.quantile(np.abs(momentum), [index / 10.0 for index in range(1, 10)]), calibration_end)


def process_date(day: str, path: Path, *, seed: int = 20250301, source_sha256: str | None = None,
                 cache_root: Path | None = None) -> dict[str, Any]:
    """Stage A/B MLOFI processing using compact arrays instead of Python observations."""
    started = time.perf_counter()
    source_sha256 = source_sha256 or _sha256(path)
    features = _load_feature_cache(cache_root, day, source_sha256) if cache_root is not None else None
    timings: dict[str, float] = {"feature_cache_load": 0.0}
    if features is None:
        features, stage_a = _extract_compact_features(day, path, source_sha256=source_sha256, cache_root=cache_root)
        timings.update(stage_a)
        timings["feature_cache_hit"] = 0.0
        # The just-extracted arrays are disk-backed staging memmaps.  Stage B
        # makes many bounded random/as-of reads, so rematerialize the compact
        # 157MB-on-disk cache into ordinary arrays before analysis.  This keeps
        # the one raw pass bounded while avoiding page-fault amplification in
        # the dense 72-variant aggregation pass.
        if cache_root is not None:
            features.close()
            materialize_started = time.perf_counter()
            features = _load_feature_cache(cache_root, day, source_sha256)
            if features is None:
                raise MLOFIError("atomic compact feature cache could not be reloaded")
            timings["feature_cache_materialize"] = time.perf_counter() - materialize_started
    else:
        timings["feature_cache_hit"] = 1.0
    try:
        calibration_started = time.perf_counter()
        decile_bounds, extreme_bounds, depth_bounds, volatility_bounds, calibration_end = _calibration(features)
        timings["percentile_calibration"] = time.perf_counter() - calibration_started
        stats = StatsCube(); extreme_stats = StatsCube(buckets=4)
        control_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
        control_sum = np.zeros_like(control_count, dtype=np.float64)
        regime_count = np.zeros((2, 3, 3, len(HORIZONS_MS)), dtype=np.int64)
        regime_sum = np.zeros_like(regime_count, dtype=np.float64)
        session_count = np.zeros((3, 2, len(HORIZONS_MS)), dtype=np.int64)
        session_sum = np.zeros_like(session_count, dtype=np.float64)
        micro_count = np.zeros(2, dtype=np.int64)
        micro_sum = np.zeros(2, dtype=np.float64)
        micro_sumsq = np.zeros(2, dtype=np.float64)
        event_samples = int(features.array("event_sample").sum())
        event_analyzed = 0
        event_started = time.perf_counter()
        timestamp_all = features.array("timestamp_ns")
        midpoint_all = features.array("mid_sum_raw")
        for session, timestamp, midpoint, event_flag, signals, global_index in _iter_event_signal_batches(features):
            session_start, session_end = next((start, end) for code, start, end in _session_slices(features) if code == session)
            usable = event_flag & (timestamp > calibration_end)
            if not usable.any():
                continue
            outcome = _markouts(timestamp, midpoint,
                                np.asarray(timestamp_all[session_start:session_end], dtype=np.int64),
                                np.asarray(midpoint_all[session_start:session_end], dtype=np.int64))
            signal_part, outcome_part = signals[usable], outcome[usable]
            _accumulate_cube(stats, view=0, signals=signal_part, markouts=outcome_part, decile_bounds=decile_bounds)
            _accumulate_cube(extreme_stats, view=0, signals=signal_part, markouts=outcome_part, decile_bounds=extreme_bounds, extreme=True)
            momentum = _momentum_ticks(timestamp[usable], midpoint[usable],
                                        np.asarray(timestamp_all[session_start:session_end], dtype=np.int64),
                                        np.asarray(midpoint_all[session_start:session_end], dtype=np.int64))
            depth = _denominators(np.asarray(features.array("depth_sum")[global_index[usable]]))[:, COMBO_INDEX[(5, "INVERSE_LEVEL")]]
            _accumulate_controls(view=0, signals=signal_part, markouts=outcome_part, momentum_ticks=momentum,
                                 depth=depth, volatility=np.abs(momentum), main_bounds=decile_bounds[:, VARIANT_INDEX[MAIN_VARIANT]],
                                 depth_bounds=depth_bounds, volatility_bounds=volatility_bounds,
                                 control_count=control_count, control_sum=control_sum,
                                 regime_count=regime_count, regime_sum=regime_sum)
            for horizon in range(len(HORIZONS_MS)):
                valid = np.isfinite(outcome_part[:, horizon])
                session_count[session, 0, horizon] += int(valid.sum())
                session_sum[session, 0, horizon] += float(outcome_part[valid, horizon].sum())
            micro = np.asarray(features.array("microprice")[global_index[usable]], dtype=np.float64)
            micro_offset = (2.0 * micro - midpoint[usable]) / 500_000_000.0
            micro_count[0] += micro_offset.size; micro_sum[0] += micro_offset.sum(); micro_sumsq[0] += np.square(micro_offset).sum()
            event_analyzed += int(usable.sum())
        timings["event_signals_markouts_statistics"] = time.perf_counter() - event_started

        fixed_started = time.perf_counter()
        fixed = _fixed_view(day, features)
        fixed_signals, fixed_midpoint, fixed_depth = _fixed_signals(features, fixed)
        fixed_usable = fixed.timestamp_ns > calibration_end
        fixed_analyzed = int(fixed_usable.sum())
        for session, start, end in _session_slices(features):
            selected = np.flatnonzero((fixed.session == session) & fixed_usable)
            if not selected.size:
                continue
            outcome = _markouts(fixed.timestamp_ns[selected], fixed_midpoint[selected],
                                np.asarray(timestamp_all[start:end], dtype=np.int64),
                                np.asarray(midpoint_all[start:end], dtype=np.int64))
            current_signals = fixed_signals[selected]
            _accumulate_cube(stats, view=1, signals=current_signals, markouts=outcome, decile_bounds=decile_bounds)
            _accumulate_cube(extreme_stats, view=1, signals=current_signals, markouts=outcome, decile_bounds=extreme_bounds, extreme=True)
            momentum = _momentum_ticks(fixed.timestamp_ns[selected], fixed_midpoint[selected],
                                        np.asarray(timestamp_all[start:end], dtype=np.int64),
                                        np.asarray(midpoint_all[start:end], dtype=np.int64))
            _accumulate_controls(view=1, signals=current_signals, markouts=outcome, momentum_ticks=momentum,
                                 depth=fixed_depth[selected], volatility=np.abs(momentum),
                                 main_bounds=decile_bounds[:, VARIANT_INDEX[MAIN_VARIANT]], depth_bounds=depth_bounds,
                                 volatility_bounds=volatility_bounds, control_count=control_count, control_sum=control_sum,
                                 regime_count=regime_count, regime_sum=regime_sum)
            for horizon in range(len(HORIZONS_MS)):
                valid = np.isfinite(outcome[:, horizon])
                session_count[session, 1, horizon] += int(valid.sum())
                session_sum[session, 1, horizon] += float(outcome[valid, horizon].sum())
            micro = np.asarray(features.array("microprice")[fixed.state_index[selected]], dtype=np.float64)
            micro_offset = (2.0 * micro - fixed_midpoint[selected]) / 500_000_000.0
            micro_count[1] += micro_offset.size; micro_sum[1] += micro_offset.sum(); micro_sumsq[1] += np.square(micro_offset).sum()
        timings["fixed_asof_sampling_markouts_statistics"] = time.perf_counter() - fixed_started

        barriers_started = time.perf_counter()
        barrier_count = _barrier_counts(features, fixed)
        timings["exact_first_touch_barriers"] = time.perf_counter() - barriers_started
        timings["total"] = time.perf_counter() - started
        return {
            "date": day, "raw_rows": features.raw_rows, "relevant_state_events": features.count,
            "event_view_samples": event_samples, "event_view_samples_analysed": event_analyzed,
            "fixed_interval_view_samples": int(fixed.timestamp_ns.size),
            "fixed_interval_view_samples_analysed": fixed_analyzed,
            "unresolved_markout_observations": 0,
            "percentile_calibration_events": PERCENTILE_CALIBRATION_EVENTS,
            "percentile_calibration_end_timestamp": _iso_ns(calibration_end),
            "stats": stats.export(), "extreme_stats": extreme_stats.export(),
            "control_count": control_count.tolist(), "control_sum": control_sum.tolist(),
            "regime_count": regime_count.tolist(), "regime_sum": regime_sum.tolist(),
            "barrier_count": barrier_count.tolist(), "session_diagnostics": {
                "count": session_count.tolist(), "sum": session_sum.tolist(),
                "session_order": list(baseline.SESSION_ORDER),
            },
            "microprice_diagnostics": {
                "count": micro_count.tolist(), "mean_offset_ticks": np.divide(micro_sum, micro_count, out=np.zeros(2), where=micro_count > 0).tolist(),
                "std_offset_ticks": np.sqrt(np.maximum(0.0, np.divide(micro_sumsq, micro_count, out=np.zeros(2), where=micro_count > 0)
                                                        - np.square(np.divide(micro_sum, micro_count, out=np.zeros(2), where=micro_count > 0)))).tolist(),
            },
            "timings_seconds": timings,
            "feature_cache_version": FEATURE_CACHE_VERSION,
        }
    finally:
        features.close()


def _process_date_job(job: tuple[str, str, int, str, str]) -> tuple[str, dict[str, Any]]:
    day, path, seed, source_sha256, cache_root = job
    return day, process_date(day, Path(path), seed=seed, source_sha256=source_sha256, cache_root=Path(cache_root))


def run_study(*, data_root: Path = baseline.DATA_ROOT,
              tape_manifest: Path = baseline.OUTPUT_ROOT / "candidate-tapes" / "train-tape-manifest.json",
              output_root: Path = Path("research_runs") / RUN_ID,
              cache_root: Path | None = None, workers: int = 1,
              force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    if TAPE_VERSION != EXPECTED_TAPE_VERSION:
        raise MLOFIError("repository Candidate Tape version mismatch")
    paths = _source_paths(data_root)
    source_hashes = {day: _sha256(path) for day, path in paths.items()}
    tape = _validate_tape_contract(tape_manifest, source_hashes)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root or output_root / "feature-cache"
    date_payload_by_day: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[str, str, int, str, str]] = []
    for index, day in enumerate(baseline.TRAIN_DATES, 1):
        destination = output_root / "per-date" / f"{day}.json"
        if destination.exists() and not force:
            try:
                cached = json.loads(destination.read_text())
                if cached.get("source_sha256") == source_hashes[day] and cached.get("status") == "PASS":
                    date_payload_by_day[day] = cached
                    print(f"MLOFI_DATE_RESUME_SKIP={day}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        jobs.append((day, str(paths[day]), 20250301 + index, source_hashes[day], str(cache_root)))

    # This Mac has 16GB RAM and each raw/exact date can hold several GB
    # transiently.  Two independent dates are the conservative bound;
    # result merging is always canonical TRAIN-date order regardless of finish
    # order.
    worker_count = max(1, min(2, int(workers), len(jobs) or 1))
    print(f"MLOFI_WORKERS={worker_count} REMAINING_DATES={[job[0] for job in jobs]}", flush=True)

    def checkpoint(day: str, payload: dict[str, Any]) -> None:
        payload.update({"status": "PASS", "source_path": str(paths[day]), "source_sha256": source_hashes[day],
                        "tape_version": EXPECTED_TAPE_VERSION})
        _json_write(output_root / "per-date" / f"{day}.json", payload)
        date_payload_by_day[day] = payload
        print(f"MLOFI_DATE_COMPLETE={day} event={payload['event_view_samples']} fixed={payload['fixed_interval_view_samples']}", flush=True)

    if worker_count == 1:
        for index, job in enumerate(jobs, 1):
            day = job[0]
            print(f"MLOFI_DATE_START={index}/{len(jobs)} {day}", flush=True)
            _, payload = _process_date_job(job)
            checkpoint(day, payload)
    elif jobs:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_process_date_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                day, payload = future.result()
                checkpoint(day, payload)

    date_payloads = [date_payload_by_day[day] for day in baseline.TRAIN_DATES]
    if len(date_payloads) != len(baseline.TRAIN_DATES):
        raise MLOFIError("not all TRAIN dates produced validated MLOFI checkpoints")
    stats, extreme, control_count, control_sum, regime_count, regime_sum, barrier_count, counts = _aggregate_date_payloads(date_payloads)
    bucket_rows = _stats_rows(stats)
    extreme_rows = _stats_rows(extreme, extreme=True)
    daily_rows = [row for payload in date_payloads for row in _daily_main_rows(payload)]
    main_index = variant_index(*MAIN_VARIANT)
    main_bucket_rows = [row for row in bucket_rows if row["levels"] == 5 and row["weighting"] == "INVERSE_LEVEL"
                        and row["window_ms"] == 500 and row["normalization"] == "DEPTH_NORMALIZED"]
    barrier_rows = []
    for view, view_name in enumerate(("EVENT_VIEW", "FIXED_INTERVAL_VIEW")):
        for variant in range(72):
            levels = LEVELS[variant // (3 * 4 * 2)]
            weighting = tuple(WEIGHTING)[(variant // (4 * 2)) % 3]
            window = WINDOWS_MS[(variant // 2) % 4]
            normalization = ("RAW", "DEPTH_NORMALIZED")[variant % 2]
            for barrier_index, barrier in enumerate((1, 2, 4)):
                count, positive, negative, tie = (int(barrier_count[view, variant, barrier_index, i]) for i in range(4))
                barrier_rows.append({"view": view_name, "levels": levels, "weighting": weighting,
                                     "window_ms": window, "normalization": normalization,
                                     "barrier_ticks": barrier, "sample_count": count,
                                     "positive_first_fraction": positive / count if count else None,
                                     "negative_first_fraction": negative / count if count else None,
                                     "tie_fraction": tie / count if count else None})
    daily_spreads = [float(row["top_minus_bottom_ticks"]) for row in daily_rows
                     if row["horizon_ms"] == 500 and row["top_minus_bottom_ticks"] is not None]
    regime_rows = []
    for view, view_name in enumerate(("EVENT_VIEW", "FIXED_INTERVAL_VIEW")):
        for regime_index, class_name in enumerate(("MOMENTUM", "DEPTH", "VOLATILITY")):
            for level in range(3):
                for horizon, horizon_ms in enumerate(HORIZONS_MS):
                    count = int(regime_count[view, regime_index, level, horizon])
                    total = float(regime_sum[view, regime_index, level, horizon])
                    regime_rows.append({"view": view_name, "regime": class_name, "level": level,
                                        "horizon_ms": horizon_ms, "sample_count": count,
                                        "mean_markout_ticks": total / count if count else None})
    artifact = {
        "run_id": RUN_ID, "status": "PASS", "train_date_count": len(baseline.TRAIN_DATES),
        "train_dates": list(baseline.TRAIN_DATES), "data_source": "validated native ES MBP-10 underlying Candidate Tape V2",
        "tape_version": EXPECTED_TAPE_VERSION, "tape_manifest": str(tape_manifest),
        "tape_manifest_sha256": _sha256(tape_manifest), "source_sha256_by_date": source_hashes,
        "feature_cache_version": FEATURE_CACHE_VERSION, "feature_cache_root": str(cache_root), "worker_count": worker_count,
        "total_event_view_samples": sum(int(p["event_view_samples"]) for p in date_payloads),
        "total_fixed_interval_samples": sum(int(p["fixed_interval_view_samples"]) for p in date_payloads),
        "effective_date_count": len(date_payloads), "elapsed_seconds": time.monotonic() - started,
        "optimization_performed": False, "strategy_backtest_performed": False, "pnl_calculated": False,
        "october_accessed": False, "dec_jan_accessed": False, "final_oos_accessed": False,
        "data_downloaded": False, "mlofi_signal_evidence": _classification(daily_rows, bucket_rows),
        "main_variant": {"levels": 5, "weighting": "INVERSE_LEVEL", "window_ms": 500, "normalization": "DEPTH_NORMALIZED"},
        "barrier_horizon_ms": 10_000,
    }
    _json_write(output_root / "run-manifest.json", {**artifact, "source_paths": {d: str(p) for d, p in paths.items()},
                                                      "raw_dbn_read_for_mlofi": True})
    _json_write(output_root / "summary.json", artifact)
    _json_write(output_root / "signal-definition.json", {
        "positive_direction": "bullish", "ofi_formula": "bid additions - bid removals - ask additions + ask removals",
        "accounting": "A/C/M explicit price-keyed displayed-size delta; T direct execution size on passive side; R zero; rank shifts are not inferred cancellations",
        "normalization": "weighted current-time mean of displayed bid/ask depth; zero denominator yields zero and invalid books are excluded",
        "event_view": "one sample after each executable A/C/M/T/R state change",
        "fixed_interval_view": "100ms cadence targets use the last executable state at or before T; state timestamps never exceed T",
        "markout": "first valid executable midprice state at or after observation timestamp plus horizon; no session crossing",
        "barrier": "first executable midprice touch within 10 seconds; same-event opposing touch is deterministic tie=0",
        "percentiles": "first 100,000 chronological event states calibrate deterministic per-date bounds; those calibration observations are excluded, so every analysed bucket is strictly no-future",
        "weighting": {name: list(values) for name, values in WEIGHTING.items()},
    })
    _json_write(output_root / "variant-results.json", {"bucket_results": bucket_rows, "extreme_results": extreme_rows,
                                                        "barrier_results": barrier_rows,
                                                        "barrier_counts": barrier_count.tolist()})
    _json_write(output_root / "bucket-results.json", {"deciles": bucket_rows, "extreme_tails": extreme_rows,
                                                       "control_stats": {"count": control_count.tolist(), "sum": control_sum.tolist()}})
    _json_write(output_root / "daily-stability.json", {"rows": daily_rows, "dates_expected_sign": sum(float(r["top_minus_bottom_ticks"]) > 0 for r in daily_rows if r["horizon_ms"] == 500 and r["top_minus_bottom_ticks"] is not None),
                                                        "dates_opposite_sign": sum(float(r["top_minus_bottom_ticks"]) < 0 for r in daily_rows if r["horizon_ms"] == 500 and r["top_minus_bottom_ticks"] is not None),
                                                        "median_daily_spread_ticks": statistics.median(daily_spreads) if daily_spreads else None,
                                                        "daily_spread_values_ticks": daily_spreads})
    _json_write(output_root / "regime-results.json", {
        "rows": regime_rows,
        "momentum_control": {"count": control_count.tolist(), "sum": control_sum.tolist()},
        "regime_count": regime_count.tolist(), "regime_sum": regime_sum.tolist(),
    })
    with (output_root / "bucket-results.csv").open("w", newline="") as handle:
        rows = bucket_rows
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    with (output_root / "daily-stability.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in daily_rows for key in row}))
        writer.writeheader(); writer.writerows(daily_rows)
    report_lines = [
        f"# {RUN_ID}", "", "TRAIN-only descriptive MLOFI event study; no strategy, entry, stop, target, or PnL.", "",
        f"- Dates: {len(baseline.TRAIN_DATES)}", f"- Event-view samples: {artifact['total_event_view_samples']}",
        f"- Fixed 100ms samples: {artifact['total_fixed_interval_samples']}",
        f"- Evidence classification: **{artifact['mlofi_signal_evidence']}**", "",
        "## Main variant", "", "| Horizon ms | Bucket | Count | Mean ticks | Median ticks | CI95 |", "|---:|---|---:|---:|---:|---|",
    ]
    for row in main_bucket_rows:
        if row.get("sample_count"):
            report_lines.append(f"| {row['horizon_ms']} | {row['bucket']} | {row['sample_count']} | {row['mean_markout_ticks']:.6f} | {row.get('median_markout_ticks')} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    report_lines += ["", "See `bucket-results.csv`, `daily-stability.csv`, `regime-results.json`, and `variant-results.json` for complete outputs.", ""]
    (output_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=baseline.DATA_ROOT)
    parser.add_argument("--tape-manifest", type=Path, default=baseline.OUTPUT_ROOT / "candidate-tapes" / "train-tape-manifest.json")
    parser.add_argument("--output-root", type=Path, default=Path("research_runs") / RUN_ID)
    parser.add_argument("--cache-root", type=Path, default=None, help="source-bound compact feature cache; defaults under output-root")
    parser.add_argument("--workers", type=int, default=1, help="independent date workers (conservatively capped at 2 on this 16GB Mac)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_study(data_root=args.data_root, tape_manifest=args.tape_manifest,
                       output_root=args.output_root, cache_root=args.cache_root,
                       workers=args.workers, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
