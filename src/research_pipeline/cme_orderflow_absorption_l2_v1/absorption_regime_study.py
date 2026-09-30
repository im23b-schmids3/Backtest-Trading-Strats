"""Dec-2025/Jan-2026 descriptive absorption-regime event study.

The frozen ten-family Candidate Tape V2 supplies the completed causal core
events.  Native ES MBP-10 is read locally for pre-event depth/flow features;
the already validated sparse BBO tape supplies forward markouts and paths.
This module does not evaluate executions, trades, or PnL and performs no
parameter search.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_mlofi_event_study as mlofi
from . import ten_family_dec_jan_robustness as frozen_run
from .model import L2Config


RUN_ID = "CMEOrderflow_ABSORPTION_REGIME_DEC_JAN_V1"
EXPECTED_CONFIG_SHA256 = frozen_run.EXPECTED_CONFIG_SHA256
CONFIG_PATH = frozen_run.CONFIG_PATH
PRIOR_RUN_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_TEN_FAMILY_DEC2025_JAN2026_"
    "ROBUSTNESS_SESSION_BOUNDARY_FIXED_RETRY1"
)
OUTPUT_ROOT = Path("research_runs") / RUN_ID
TICK_POINTS = 0.25
RAW_PRICE_SCALE = 1_000_000_000
FLOW_BIN_NS = 250_000_000
FLOW_BINS = 8
DEPTH_GRID_NS = 100_000_000
RESILIENCY_INDEX_BLOCK = 64
RESILIENCY_LOOKBACKS_NS = (30_000_000_000, 60_000_000_000, 120_000_000_000)
RECOVERY_HORIZONS_NS = (100_000_000, 250_000_000, 500_000_000,
                        1_000_000_000, 2_000_000_000)
MARKOUT_HORIZONS_MS = (250, 500, 1_000, 2_000, 5_000, 10_000, 30_000)
EXCURSION_HORIZONS_MS = (1_000, 2_000, 5_000, 10_000, 30_000)
BARRIER_PAIRS = ((1, 1), (2, 2), (4, 4), (8, 4), (12, 6))
MIN_EXPANDING_HISTORY = 20
MIN_DAILY_SAMPLE = 5
SESSION_NAMES = ("ASIA", "EUROPE", "NY")
SESSION_CODES = {name: index for index, name in enumerate(SESSION_NAMES)}


class RegimeStudyError(RuntimeError):
    """A frozen input, causal, checkpoint, or artifact invariant failed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _iso(ns: int | None) -> str | None:
    if ns is None:
        return None
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_ready(payload), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _jsonl_gz_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                for row in rows:
                    gz.write((json.dumps(row, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False) + "\n").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def direction_sign(direction: str) -> int:
    if direction == "SELLER_ABSORPTION":
        return 1
    if direction == "BUYER_ABSORPTION":
        return -1
    raise RegimeStudyError(f"unknown frozen absorption direction: {direction}")


def direction_normalize(price_change_ticks: float, direction: str) -> float:
    return float(direction_sign(direction) * price_change_ticks)


def recovery_fraction(depth_before: float, depth_min: float, depth_at_horizon: float) -> float | None:
    depleted = float(depth_before) - float(depth_min)
    if not math.isfinite(depleted) or depleted <= 0:
        return None
    return (float(depth_at_horizon) - float(depth_min)) / depleted


def mlofi_persistence(bins: Sequence[float], reversal_sign: int, denominator: float) -> dict[str, Any]:
    values = [float(v) for v in bins]
    total = sum(values)
    nonzero = [v for v in values if v != 0.0]
    same = (sum(1 for v in nonzero if (v > 0) == (total > 0)) / len(nonzero)
            if nonzero and total != 0 else (0.0 if nonzero else None))
    signed = (reversal_sign * total / denominator) if denominator > 0 else None
    if signed is None or signed == 0:
        state = "FLOW_NEUTRAL"
    elif signed > 0:
        state = "FLOW_SUPPORTS_REVERSAL"
    else:
        state = "FLOW_OPPOSES_REVERSAL"
    return {"bin_values": values, "same_sign_bin_fraction": same,
            "signed_persistence_relative_to_reversal": signed, "flow_state": state}


def aggression_depth_ratio(aggressive_volume: float, mean_depth: float) -> float | None:
    if mean_depth <= 0 or not math.isfinite(mean_depth):
        return None
    return float(aggressive_volume) / float(mean_depth)


def price_impact(mid_change_ticks: float, raw_weighted_ofi: float) -> float:
    return abs(float(mid_change_ticks)) / (abs(float(raw_weighted_ofi)) + 1e-9)


def volatility_state(value: float | None, prior_values: Sequence[float]) -> str:
    if value is None or not math.isfinite(float(value)) or len(prior_values) < MIN_EXPANDING_HISTORY:
        return "INSUFFICIENT_HISTORY"
    q20, q80, q95 = (float(np.quantile(np.asarray(prior_values, dtype=np.float64), q))
                     for q in (0.20, 0.80, 0.95))
    value = float(value)
    return "LOW" if value < q20 else "NORMAL" if value < q80 else "HIGH" if value < q95 else "EXTREME"


def expanding_quintile(value: float | None, prior_values: Sequence[float]) -> str:
    if value is None or not math.isfinite(float(value)) or len(prior_values) < MIN_EXPANDING_HISTORY:
        return "INSUFFICIENT_HISTORY"
    thresholds = np.quantile(np.asarray(prior_values, dtype=np.float64), (0.2, 0.4, 0.6, 0.8))
    return f"Q{int(np.searchsorted(thresholds, float(value), side='right')) + 1}"


def mfe_mae(direction_changes_ticks: Sequence[float]) -> tuple[float, float]:
    values = [float(v) for v in direction_changes_ticks]
    if not values:
        return 0.0, 0.0
    return max(0.0, max(values)), max(0.0, -min(values))


def first_touch(times_ns: Sequence[int], changes_ticks: Sequence[float],
                favorable: float, adverse: float) -> dict[str, Any]:
    if len(times_ns) != len(changes_ticks):
        raise ValueError("barrier path arrays differ in length")
    for timestamp, move in zip(times_ns, changes_ticks):
        up, down = float(move) >= favorable, float(move) <= -adverse
        if up and down:
            return {"outcome": "SIMULTANEOUS", "time_ns": int(timestamp)}
        if up:
            return {"outcome": "FAVORABLE_FIRST", "time_ns": int(timestamp)}
        if down:
            return {"outcome": "ADVERSE_FIRST", "time_ns": int(timestamp)}
    return {"outcome": "UNRESOLVED", "time_ns": None}


def checkpoint_matches(checkpoint: Mapping[str, Any], *, date: str,
                       source_sha256: str, config_sha256: str,
                       output_sha256: str | None,
                       candidate_tape_sha256: str | None = None,
                       semantic_sha256: str | None = None) -> bool:
    base = (checkpoint.get("status") == "DATE_COMPLETE"
            and checkpoint.get("date") == date
            and checkpoint.get("source_sha256") == source_sha256
            and checkpoint.get("config_sha256") == config_sha256
            and output_sha256 is not None
            and checkpoint.get("output_sha256") == output_sha256)
    return (base
            and (candidate_tape_sha256 is None
                 or checkpoint.get("candidate_tape_sha256") == candidate_tape_sha256)
            and (semantic_sha256 is None
                 or checkpoint.get("candidate_tape_semantic_sha256") == semantic_sha256))


def _matrix(batch: np.ndarray, name: str) -> np.ndarray:
    return np.column_stack([batch[f"{name}_{i:02d}"] for i in range(10)])


def _match_level(prices: np.ndarray, sizes: np.ndarray, price: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (prices > 0) & (sizes > 0)
    matches = valid & (prices == price[:, None]) & (price[:, None] > 0)
    found = matches.any(axis=1)
    pos = matches.argmax(axis=1)
    rank_matrix = np.cumsum(valid, axis=1, dtype=np.int16) - 1
    rank = np.take_along_axis(rank_matrix, pos[:, None], axis=1)[:, 0]
    size = np.take_along_axis(sizes, pos[:, None], axis=1)[:, 0]
    return found, np.where(found, rank, -1), np.where(found, size, 0)


def _batch_market_columns(batch: np.ndarray, previous: tuple[np.ndarray, ...] | None,
                         previous_executable: bool, previous_ts: int | None) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], bool, int]:
    ts = batch["ts_recv"].astype(np.int64, copy=False)
    bpx, apx, bsz, asz = (_matrix(batch, n) for n in
                            ("bid_px", "ask_px", "bid_sz", "ask_sz"))
    bv, av = (bpx > 0) & (bsz > 0), (apx > 0) & (asz > 0)
    br, ar = np.cumsum(bv, axis=1, dtype=np.int16) - 1, np.cumsum(av, axis=1, dtype=np.int16) - 1
    bid_count, ask_count = bv.sum(axis=1), av.sum(axis=1)
    bid_best = np.max(np.where(bv, bpx, 0), axis=1)
    ask_best = np.min(np.where(av, apx, np.iinfo(np.int64).max), axis=1)
    executable = (bid_count > 0) & (ask_count > 0) & (ask_best > bid_best)
    mid_half_ticks = np.where(executable,
        np.rint((bid_best.astype(np.float64) + ask_best.astype(np.float64)) *
                (4.0 / RAW_PRICE_SCALE)), 0).astype(np.int32)
    bid_depth = np.zeros(len(batch), dtype=np.float64)
    ask_depth = np.zeros(len(batch), dtype=np.float64)
    denominator = np.zeros(len(batch), dtype=np.float64)
    denom_weight = np.zeros(len(batch), dtype=np.float64)
    for rank in range(5):
        bi = np.sum(np.where(bv & (br == rank), bsz, 0), axis=1, dtype=np.float64)
        ai = np.sum(np.where(av & (ar == rank), asz, 0), axis=1, dtype=np.float64)
        bid_depth += bi
        ask_depth += ai
        weight = 1.0 / (rank + 1)
        denominator += weight * (bi + ai) / 2.0
        denom_weight += np.where((bi > 0) | (ai > 0), weight, 0.0)
    denominator = np.divide(denominator, denom_weight, out=np.zeros_like(denominator), where=denom_weight > 0)

    prev_bpx = np.zeros_like(bpx); prev_apx = np.zeros_like(apx)
    prev_bsz = np.zeros_like(bsz); prev_asz = np.zeros_like(asz)
    prev_ok = np.zeros(len(batch), dtype=bool)
    if previous is not None:
        pbpx, papx, pbsz, pasz = previous
        prev_bpx[0], prev_apx[0], prev_bsz[0], prev_asz[0] = pbpx, papx, pbsz, pasz
        prev_ok[0] = previous_executable
    if len(batch) > 1:
        prev_bpx[1:], prev_apx[1:] = bpx[:-1], apx[:-1]
        prev_bsz[1:], prev_asz[1:] = bsz[:-1], asz[:-1]
        prev_ok[1:] = executable[:-1]
    prev_bpx = np.where(prev_ok[:, None], prev_bpx, 0)
    prev_apx = np.where(prev_ok[:, None], prev_apx, 0)
    prev_bsz = np.where(prev_ok[:, None], prev_bsz, 0)
    prev_asz = np.where(prev_ok[:, None], prev_asz, 0)

    action, side = batch["action"], batch["side"]
    price = batch["price"].astype(np.int64, copy=False)
    size = batch["size"].astype(np.float64, copy=False)
    book_update = np.isin(action, (b"A", b"C", b"M")) & np.isin(side, (b"B", b"A"))
    side_bid = side == b"B"
    cp = np.where(side_bid[:, None], bpx, apx); cs = np.where(side_bid[:, None], bsz, asz)
    pp = np.where(side_bid[:, None], prev_bpx, prev_apx); ps = np.where(side_bid[:, None], prev_bsz, prev_asz)
    cf, cr, csz = _match_level(cp, cs, price)
    pf, pr, psz = _match_level(pp, ps, price)
    use_rank = np.where(cf, cr, pr)
    delta = np.where(cf, csz, 0.0) - np.where(pf, psz, 0.0)
    book_sign = np.where(side_bid, 1.0, -1.0)
    book_denominator = np.where(use_rank >= 0, use_rank + 1, 1)
    contribution = np.where(book_update & (use_rank >= 0) & (use_rank < 5),
                            book_sign * delta / book_denominator, 0.0)

    trade = action == b"T"
    buy = trade & (side == b"B"); sell = trade & (side == b"A")
    passive_px = np.where((side == b"B")[:, None], apx, bpx)
    passive_sz = np.where((side == b"B")[:, None], asz, bsz)
    prev_passive_px = np.where((side == b"B")[:, None], prev_apx, prev_bpx)
    prev_passive_sz = np.where((side == b"B")[:, None], prev_asz, prev_bsz)
    tf, tr, _ = _match_level(passive_px, passive_sz, price)
    tpf, tpr, _ = _match_level(prev_passive_px, prev_passive_sz, price)
    trade_rank = np.where(tf, tr, tpr)
    trade_denominator = np.where(trade_rank >= 0, trade_rank + 1, 1)
    contribution += np.where(trade & (side != b"N") & (trade_rank >= 0) & (trade_rank < 5),
                             np.where(side == b"B", size, -size) / trade_denominator, 0.0)
    contribution = np.where(executable, contribution, 0.0)

    output = {
        "ts": ts, "mid_half_ticks": mid_half_ticks,
        "bid_depth": bid_depth.astype(np.int32), "ask_depth": ask_depth.astype(np.int32),
        "denom": denominator, "ofi": contribution,
        "buy_volume": np.where(buy, size, 0).astype(np.int32),
        "sell_volume": np.where(sell, size, 0).astype(np.int32),
        "executable": executable,
    }
    last_previous = (bpx[-1].copy(), apx[-1].copy(), bsz[-1].copy(), asz[-1].copy())
    return output, last_previous, bool(executable[-1]), int(ts[-1]) if len(ts) else previous_ts


def _session_code_for(ts: np.ndarray, windows: Mapping[str, Sequence[int]]) -> np.ndarray:
    result = np.full(len(ts), -1, dtype=np.int8)
    for name, code in SESSION_CODES.items():
        start, end = map(int, windows[name])
        result[(ts >= start) & (ts < end)] = code
    return result


def _empty_session() -> dict[str, list[np.ndarray]]:
    return {k: [] for k in ("ts", "mid_half_ticks", "bid_depth", "ask_depth", "denom", "ofi", "buy_volume", "sell_volume")}


def _load_market_day(source_paths: Sequence[Path], session_windows: Mapping[str, Sequence[int]],
                     *, progress_date: str) -> dict[str, dict[str, np.ndarray]]:
    from databento import DBNStore

    collected = {name: _empty_session() for name in SESSION_NAMES}
    prev: tuple[np.ndarray, ...] | None = None
    prev_exec = False
    previous_ts: int | None = None
    total_records = 0
    last_ts: int | None = None
    for path in source_paths:
        store = DBNStore.from_file(path)
        for batch in store.to_ndarray(count=100_000):
            ts = batch["ts_recv"].astype(np.int64, copy=False)
            if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
                raise RegimeStudyError(f"timestamp regression within source {path}")
            if previous_ts is not None and len(ts) and int(ts[0]) < previous_ts:
                raise RegimeStudyError(f"timestamp regression across source partitions at {path}")
            raw, prev, prev_exec, previous_ts = _batch_market_columns(batch, prev, prev_exec, previous_ts)
            codes = _session_code_for(raw["ts"], session_windows)
            mask = raw["executable"] & (codes >= 0)
            for name, code in SESSION_CODES.items():
                selected = mask & (codes == code)
                if not selected.any():
                    continue
                bucket = collected[name]
                for key in bucket:
                    bucket[key].append(raw[key][selected].copy())
            total_records += len(batch)
            if len(ts):
                previous_ts = int(ts[-1]); last_ts = previous_ts
            if total_records and total_records % 5_000_000 < len(batch):
                print(f"REGIME_DATE_PROGRESS={progress_date} records={total_records}", flush=True)
    result: dict[str, dict[str, Any]] = {}
    for name, parts in collected.items():
        result[name] = {key: (np.concatenate(values) if values else np.zeros(0, dtype=np.float64))
                        for key, values in parts.items()}
        for values in parts.values():
            values.clear()
        parts.clear()
        ts = result[name]["ts"]
        if len(ts) and np.any(ts[1:] < ts[:-1]):
            raise RegimeStudyError(f"executable timestamp regression in {name}/{progress_date}")
        market = result[name]
        market["ofi_prefix"] = _prefix(market["ofi"])
        market["buy_prefix"] = _prefix(market["buy_volume"])
        market["sell_prefix"] = _prefix(market["sell_volume"])
        mid_ticks = market["mid_half_ticks"].astype(np.float64) / 2.0
        sq_step = np.zeros(len(ts), dtype=np.float64)
        if len(ts) > 1:
            sq_step[1:] = np.diff(mid_ticks) ** 2
        market["sq_mid_prefix"] = _prefix(sq_step)
        if len(ts):
            session_start, session_end = map(int, session_windows[name])
            grid, grid_bid, grid_ask = _depth_grid(
                ts, market["bid_depth"], market["ask_depth"], session_start, session_end)
        else:
            grid = np.zeros(0, dtype=np.int64)
            grid_bid = grid_ask = np.zeros(0, dtype=np.float64)
        market["depth_grid_ts"] = grid
        market["depth_grid_bid"] = grid_bid
        market["depth_grid_ask"] = grid_ask
        for side, depth_key in (("B", "bid_depth"), ("A", "ask_depth")):
            depth = market[depth_key]
            market[f"{side.lower()}_episodes"] = _episode_indices(ts, depth)
            market[f"{side.lower()}_max_tree"] = _max_tree(depth)
        # Prefix arrays and depth grids are the reusable representation. Drop
        # their source flow arrays so a single date remains memory-bounded.
        for key in ("ofi", "buy_volume", "sell_volume"):
            market.pop(key, None)
    print(f"REGIME_DATE_SOURCE_COMPLETE={progress_date} raw_records={total_records} last={_iso(last_ts)}", flush=True)
    return result


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))


def _interval_sum(ts: np.ndarray, cumulative: np.ndarray, start_ns: int, end_ns: int) -> float:
    left = int(np.searchsorted(ts, start_ns, side="left"))
    right = int(np.searchsorted(ts, end_ns, side="left"))
    return float(cumulative[right] - cumulative[left])


def _depth_grid(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                session_start: int, session_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(session_start, session_end, DEPTH_GRID_NS, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1
    valid = idx >= 0
    bid_out = np.full(len(grid), np.nan, dtype=np.float64)
    ask_out = np.full(len(grid), np.nan, dtype=np.float64)
    bid_out[valid], ask_out[valid] = bid[idx[valid]], ask[idx[valid]]
    return grid, bid_out, ask_out


def _episode_indices(ts: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if len(depth) < 2:
        return np.zeros(0, dtype=np.int64)
    prior = depth[:-1]
    drop = prior - depth[1:]
    return np.flatnonzero((prior > 0) & (drop >= np.maximum(1.0, prior * 0.10))) + 1


def _max_tree(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Small exact segment tree over fixed-size observation blocks."""
    starts = np.arange(0, len(values), RESILIENCY_INDEX_BLOCK, dtype=np.int64)
    blocks = np.maximum.reduceat(values, starts) if len(starts) else np.zeros(0, dtype=np.float64)
    size = 1
    while size < len(blocks):
        size <<= 1
    tree = np.full(size * 2, -np.inf, dtype=np.float64)
    if len(blocks):
        tree[size:size + len(blocks)] = blocks
    for index in range(size - 1, 0, -1):
        tree[index] = max(tree[index * 2], tree[index * 2 + 1])
    return tree, size, len(values)


def _first_at_least(tree_info: tuple[np.ndarray, int, int], values: np.ndarray,
                    start: int, end: int, threshold: float) -> int | None:
    """Leftmost index in [start,end), using block scans and an exact max tree."""
    tree, size, value_count = tree_info
    start, end = max(0, int(start)), min(int(end), value_count)
    if start >= end:
        return None
    first_full_block = (start + RESILIENCY_INDEX_BLOCK - 1) // RESILIENCY_INDEX_BLOCK
    first_full_index = min(end, first_full_block * RESILIENCY_INDEX_BLOCK)
    hits = np.flatnonzero(values[start:first_full_index] >= threshold)
    if len(hits):
        return start + int(hits[0])
    block_start = first_full_block
    block_end = end // RESILIENCY_INDEX_BLOCK
    def visit(node: int, left: int, right: int) -> int | None:
        if right <= block_start or left >= block_end or tree[node] < threshold:
            return None
        if right - left == 1:
            return left
        middle = (left + right) // 2
        found = visit(node * 2, left, middle)
        return found if found is not None else visit(node * 2 + 1, middle, right)
    result = visit(1, 0, size) if block_start < block_end else None
    if result is not None:
        left = result * RESILIENCY_INDEX_BLOCK
        right = min(end, left + RESILIENCY_INDEX_BLOCK)
        hits = np.flatnonzero(values[left:right] >= threshold)
        if len(hits):
            return left + int(hits[0])
    tail_start = max(first_full_index, block_end * RESILIENCY_INDEX_BLOCK)
    hits = np.flatnonzero(values[tail_start:end] >= threshold)
    return tail_start + int(hits[0]) if len(hits) else None


def _resiliency_features(market: Mapping[str, Any], side: str, start_ns: int,
                         event_index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ts = market["ts"]
    depth = market["bid_depth"] if side == "B" else market["ask_depth"]
    episodes = market[f"{side.lower()}_episodes"]
    tree_info = market[f"{side.lower()}_max_tree"]
    for window in RESILIENCY_LOOKBACKS_NS:
        cutoff = start_ns - window
        lo = int(np.searchsorted(ts[episodes], cutoff, side="left"))
        hi = int(np.searchsorted(ts[episodes], start_ns, side="left"))
        candidates = episodes[lo:hi]
        measurements: dict[int, list[float]] = {h: [] for h in RECOVERY_HORIZONS_NS}
        t50: list[float] = []; t80: list[float] = []
        for i in candidates:
            before = float(depth[i - 1]); minimum = float(depth[i]); lost = before - minimum
            if lost <= 0:
                continue
            for horizon in RECOVERY_HORIZONS_NS:
                target = int(ts[i]) + horizon
                j = int(np.searchsorted(ts, target, side="left"))
                if j < event_index and int(ts[j]) < start_ns:
                    value = recovery_fraction(before, minimum, float(depth[j]))
                    if value is not None:
                        measurements[horizon].append(value)
            for fraction, dest in ((0.5, t50), (0.8, t80)):
                hit = _first_at_least(tree_info, depth, int(i), event_index,
                                      minimum + lost * fraction)
                if hit is not None:
                    dest.append((int(ts[hit]) - int(ts[i])) / 1e6)
        suffix = f"{window // 1_000_000_000}s"
        for horizon, vals in measurements.items():
            out[f"resiliency_{suffix}_recovery_{horizon // 1_000_000}ms_median"] = (
                float(np.median(vals)) if vals else None)
        primary = measurements[500_000_000]
        out[f"resiliency_{suffix}_qualifying_episode_count"] = int(len(candidates))
        out[f"resiliency_{suffix}_episode_count"] = len(primary)
        out[f"resiliency_{suffix}_score"] = float(np.median(primary)) if primary else None
        out[f"resiliency_{suffix}_t50_median_ms"] = float(np.median(t50)) if t50 else None
        out[f"resiliency_{suffix}_t80_median_ms"] = float(np.median(t80)) if t80 else None
    out["resiliency_state"] = ("RESILIENCY_INSUFFICIENT" if out["resiliency_60s_score"] is None
                               else "MEASURED")
    return out


def _pre_features(event: Mapping[str, Any], market: Mapping[str, np.ndarray],
                  session_start: int, session_end: int) -> dict[str, Any]:
    ts = market["ts"]
    start = int(event["interaction_start_ns"])
    ix = int(np.searchsorted(ts, start, side="left"))
    result: dict[str, Any] = {"feature_cutoff_ns": start, "feature_end_ns": int(ts[ix - 1]) if ix else None}
    if ix == 0:
        return {**result, "feature_status": "NO_PRE_EVENT_BOOK"}
    result["feature_status"] = "OK"
    reversal = direction_sign(str(event["direction"]))
    denom = float(market["denom"][ix - 1])
    ofi_prefix = market["ofi_prefix"]
    buy_prefix, sell_prefix = market["buy_prefix"], market["sell_prefix"]
    raw_windows: dict[int, float] = {}
    for seconds in (1, 2, 5):
        raw = _interval_sum(ts, ofi_prefix, start - seconds * 1_000_000_000, start)
        raw_windows[seconds] = raw
        result[f"pre_{seconds}s_raw_inverse_level_top5_ofi"] = raw
        result[f"pre_{seconds}s_nMLOFI"] = raw / denom if denom > 0 else None
    bins: list[float] = []
    for idx in range(FLOW_BINS):
        left = start - 2_000_000_000 + idx * FLOW_BIN_NS
        right = left + FLOW_BIN_NS
        bins.append(_interval_sum(ts, ofi_prefix, left, right))
    persistence = mlofi_persistence(bins, reversal, denom)
    result.update(persistence)

    pressured_is_bid = reversal > 0
    pressured_depth = market["bid_depth"] if pressured_is_bid else market["ask_depth"]
    aggressive_prefix = sell_prefix if pressured_is_bid else buy_prefix
    grid = market["depth_grid_ts"]
    grid_bid, grid_ask = market["depth_grid_bid"], market["depth_grid_ask"]
    mean_depths: dict[int, float | None] = {}
    volumes: dict[int, float] = {}
    for seconds in (1, 2):
        left_t = start - seconds * 1_000_000_000
        sum_grid = int(np.searchsorted(grid, start, side="left"))
        left_grid = int(np.searchsorted(grid, left_t, side="left"))
        gdepth = grid_bid if pressured_is_bid else grid_ask
        values = gdepth[left_grid:sum_grid]
        values = values[np.isfinite(values)]
        mean_depth = float(np.mean(values)) if len(values) else None
        mean_depths[seconds] = mean_depth
        volume = _interval_sum(ts, aggressive_prefix, left_t, start)
        volumes[seconds] = volume
        result[f"pre_{seconds}s_pressured_aggressive_volume"] = volume
        result[f"pre_{seconds}s_mean_pressured_top5_depth_100ms_grid"] = mean_depth
        result[f"pre_{seconds}s_aggression_to_depth"] = (
            aggression_depth_ratio(volume, mean_depth) if mean_depth is not None else None)

    mid_ticks = market["mid_half_ticks"].astype(np.float64) / 2.0
    sq_prefix = market["sq_mid_prefix"]
    last_mid = float(mid_ticks[ix - 1])
    for seconds in (1, 2, 5):
        prior_ix = int(np.searchsorted(ts, start - seconds * 1_000_000_000, side="left")) - 1
        prior_mid = float(mid_ticks[prior_ix]) if prior_ix >= 0 else last_mid
        delta_ticks = direction_normalize(last_mid - prior_mid, str(event["direction"]))
        result[f"pre_{seconds}s_directional_mid_change_ticks"] = delta_ticks
        result[f"pre_{seconds}s_price_impact_per_raw_ofi"] = price_impact(delta_ticks, raw_windows.get(seconds, 0.0))
        result[f"pre_{seconds}s_price_movement_per_aggressive_contract"] = (
            abs(delta_ticks) / (volumes.get(seconds, 0.0) + 1e-9))
    result["pre_1s_price_impact_abs_ticks_per_raw_ofi"] = abs(
        result["pre_1s_directional_mid_change_ticks"]) / (abs(raw_windows[1]) + 1e-9)
    result["pre_2s_price_impact_abs_ticks_per_raw_ofi"] = abs(
        result["pre_2s_directional_mid_change_ticks"]) / (abs(raw_windows[2]) + 1e-9)

    for seconds in (30, 120):
        left = max(0, int(np.searchsorted(ts, start - seconds * 1_000_000_000, side="left")))
        variance = float(sq_prefix[ix] - sq_prefix[left])
        result[f"pre_{seconds}s_realized_mid_volatility_ticks"] = math.sqrt(max(0.0, variance))
    result.update(_resiliency_features(market, "B" if pressured_is_bid else "A", start, ix))
    result["feature_end_ns"] = int(ts[ix - 1])
    if result["feature_end_ns"] >= start:
        raise RegimeStudyError("pre-event feature cutoff violation")
    return result


def _path_outcomes(event: Mapping[str, Any], rows: np.ndarray,
                   session_end: int) -> dict[str, Any]:
    start = int(event["interaction_start_ns"])
    if not len(rows):
        return {"outcome_status": "NO_SESSION_PATH"}
    times = rows["timestamp_ns"].astype(np.int64, copy=False)
    mids = (rows["bid"].astype(np.float64) + rows["ask"].astype(np.float64)) / 2.0
    before = int(np.searchsorted(times, start, side="left")) - 1
    if before < 0:
        return {"outcome_status": "NO_PRE_EVENT_BBO"}
    base = float(mids[before])
    sign = direction_sign(str(event["direction"]))
    result: dict[str, Any] = {"outcome_status": "OK", "interaction_start_mid": base,
                              "interaction_price": float(event["interaction_end_price"]),
                              "outcome_reference_timestamp_ns": int(times[before])}
    for ms in MARKOUT_HORIZONS_MS:
        horizon_end = start + ms * 1_000_000
        if horizon_end >= session_end:
            result[f"markout_{ms}ms_ticks"] = None
        else:
            pos = int(np.searchsorted(times, horizon_end, side="right")) - 1
            result[f"markout_{ms}ms_ticks"] = (sign * (float(mids[pos]) - base) / TICK_POINTS) if pos >= 0 else None
    for ms in EXCURSION_HORIZONS_MS:
        horizon_end = start + ms * 1_000_000
        if horizon_end >= session_end:
            result[f"mfe_{ms}ms_ticks"] = result[f"mae_{ms}ms_ticks"] = None
            continue
        lo = int(np.searchsorted(times, start, side="right"))
        hi = int(np.searchsorted(times, horizon_end, side="right"))
        path = sign * (mids[lo:hi] - base) / TICK_POINTS
        fav, adv = mfe_mae(path)
        result[f"mfe_{ms}ms_ticks"], result[f"mae_{ms}ms_ticks"] = fav, adv
    horizon_end = start + 30_000_000_000
    if horizon_end < session_end:
        lo = int(np.searchsorted(times, start, side="right"))
        hi = int(np.searchsorted(times, horizon_end, side="right"))
        ts_path = times[lo:hi]
        moves = sign * (mids[lo:hi] - base) / TICK_POINTS
        for favorable, adverse in BARRIER_PAIRS:
            outcome = first_touch(ts_path, moves, favorable, adverse)
            tag = f"{favorable}_{adverse}"
            result[f"barrier_{tag}_outcome"] = outcome["outcome"]
            result[f"barrier_{tag}_time_ns"] = outcome["time_ns"]
            result[f"barrier_{tag}_seconds"] = ((int(outcome["time_ns"]) - start) / 1e9
                                                   if outcome["time_ns"] is not None else None)
    else:
        for favorable, adverse in BARRIER_PAIRS:
            tag = f"{favorable}_{adverse}"
            result[f"barrier_{tag}_outcome"] = None
            result[f"barrier_{tag}_time_ns"] = None
            result[f"barrier_{tag}_seconds"] = None
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    payload, digest, configs = frozen_run._config_payload(path)
    if digest != EXPECTED_CONFIG_SHA256:
        raise RegimeStudyError("frozen config hash mismatch")
    return payload, digest, configs


def _validate_prior_tapes(repository_root: Path, dates: Sequence[str],
                          sources: Mapping[str, tuple[Path, Path]], source_manifest: Mapping[str, Any],
                          config_sha: str,
                          configs: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Path], str]:
    run_root = repository_root / PRIOR_RUN_ROOT
    progress_path = run_root / "progress.json"
    summary_path = run_root / "summary.json"
    loaded_path = run_root / "loaded-configs.json"
    if not all(p.is_file() for p in (progress_path, summary_path, loaded_path)):
        raise RegimeStudyError("prior completed frozen run provenance artifacts are missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loaded = json.loads(loaded_path.read_text(encoding="utf-8"))
    persisted_semantic = str(progress.get("semantic_sha256", ""))
    if (progress.get("status") != "COMPLETE" or summary.get("status") != "PASS"
            or progress.get("config_sha256") != config_sha
            or loaded.get("config_sha256") != config_sha
            or not persisted_semantic
            or summary.get("evaluator_semantic_sha256") != persisted_semantic):
        raise RegimeStudyError("prior Dec/Jan candidate-tape run identity is not complete/frozen")
    loaded_families = {str(row.get("family")): row for row in loaded.get("families", [])}
    if set(loaded_families) != set(configs):
        raise RegimeStudyError("prior run loaded-family set differs from the frozen config")
    for family, frozen in configs.items():
        recorded = loaded_families[family]
        for key in ("class_a_config", "class_b_config", "entry_delay_ms"):
            if recorded.get(key) != frozen.get(key):
                raise RegimeStudyError(f"prior run loaded config differs for {family}/{key}")
    recorded = {str(row.get("date")): row for row in progress.get("tapes", [])}
    expected_dates = list(dates)
    if sorted(recorded) != sorted(expected_dates):
        raise RegimeStudyError("prior candidate tape date set differs from eligible normal dates")
    # The tape producer's semantic digest is part of the completed prior run.
    # Do not compare it with today's producer source hash: this study consumes
    # the sealed tapes and does not regenerate their candidate semantics.
    semantic = persisted_semantic
    tapes: dict[str, Path] = {}
    for day in dates:
        paths = sources[day]
        file_meta = {row["path"]: row["sha256"] for row in source_manifest["input_files"]}
        identity = frozen_run._canonical_sha([
            {"path": str(p.relative_to(repository_root)), "sha256": file_meta[str(p.relative_to(repository_root))]}
            for p in paths
        ])
        tape_path = run_root / "tapes" / f"{day}-{candidate_tape.TAPE_FILENAME}"
        sidecar_path = run_root / "tapes" / f"{day}-{candidate_tape.TAPE_MANIFEST_FILENAME}"
        record = recorded[day]
        if not tape_path.is_file() or not sidecar_path.is_file():
            raise RegimeStudyError(f"prior candidate tape is missing for {day}")
        tape_digest = sha256_file(tape_path)
        if record.get("tape_sha256") != tape_digest:
            raise RegimeStudyError(f"prior candidate tape hash mismatch for {day}")
        tape = candidate_tape.load_tape(tape_path, source_sha256=identity, semantic_sha256=semantic)
        if tape.metadata.get("date") != day or int(tape.metadata.get("candidate_count", -1)) != len(tape.candidates):
            raise RegimeStudyError(f"prior tape metadata mismatch for {day}")
        available = set(tape.metadata.get("available_families", []))
        allowed_missing = frozen_run.PRIOR_EUROPE_FAMILIES if day == "2025-12-01" else set()
        unexpected_missing = set(configs) - available - set(allowed_missing)
        if unexpected_missing:
            raise RegimeStudyError(f"frozen families unexpectedly unavailable on {day}: {sorted(unexpected_missing)}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar != tape.metadata:
            raise RegimeStudyError(f"prior tape sidecar differs from NPZ metadata for {day}")
        tapes[day] = tape_path
        del tape
    return tapes, semantic


def _core_quality(row: Mapping[str, Any], parameters: Mapping[str, Any]) -> tuple[float, bool]:
    score = candidate_tape._quality(row, parameters["weights"], parameters)
    return float(score), bool(candidate_tape._qualifies(row, parameters))


def interaction_within_session(start_ns: int, end_ns: int,
                               window: Sequence[int]) -> bool:
    """Completed tape interactions may terminate exactly at session close."""
    window_start, window_end = map(int, window)
    return window_start <= int(start_ns) < window_end and int(start_ns) <= int(end_ns) <= window_end


def _event_rows_for_date(day: str, tape: candidate_tape.CandidateTape,
                         configs: Mapping[str, Mapping[str, Any]],
                         market_by_session: Mapping[str, Mapping[str, np.ndarray]],
                         windows: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    candidates = [row for row in tape.candidates if row.get("family_id") in configs]
    candidates.sort(key=lambda row: (int(row["interaction_start_ns"]), str(row["family_id"]),
                                    int(row.get("candidate_ordinal", 0))))
    parameters_by_family = {
        family: candidate_tape._default_parameters(
            L2Config(**{field.name: config["class_a_config"][field.name]
                        for field in fields(L2Config)}))
        for family, config in configs.items()
    }
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    paths_by_session = {
        name: tape.events[tape.events["session"] == code]
        for name, code in SESSION_CODES.items()
    }
    for row in candidates:
        family = str(row["family_id"])
        session = str(row.get("trading_session", ""))
        if session not in SESSION_CODES:
            raise RegimeStudyError(f"candidate missing valid trading session: {day}/{family}")
        start_ns = int(row.get("interaction_start_ns", 0))
        end_ns = int(row.get("interaction_end_ns", 0))
        if start_ns <= 0 or end_ns < start_ns:
            raise RegimeStudyError(f"candidate interaction chronology invalid: {day}/{family}")
        start, end = map(int, windows[session])
        if not interaction_within_session(start_ns, end_ns, (start, end)):
            raise RegimeStudyError(f"candidate outside completed frozen session window: {day}/{family}")
        event_id = f"{day}:{family}:{row.get('interaction_id')}"
        if event_id in seen:
            raise RegimeStudyError(f"duplicate event identity {event_id}")
        seen.add(event_id)
        quality, class_a_qualified = _core_quality(row, parameters_by_family[family])
        event = {
            "event_id": event_id, "date": day, "timestamp_utc": _iso(start_ns),
            "family": family, "session": session, "direction": str(row["direction"]),
            "interaction_start_ns": start_ns, "interaction_start_utc": _iso(start_ns),
            "interaction_end_ns": end_ns, "interaction_end_utc": _iso(end_ns),
            "level": row.get("level"), "interaction_price": row.get("interaction_end_price"),
            "interaction_end_price": row.get("interaction_end_price"),
            "core_quality_score": quality, "frozen_class_a_qualified": class_a_qualified,
            "interaction_id": row.get("interaction_id"),
        }
        event.update(_pre_features(event, market_by_session[session], start, end))
        event.update(_path_outcomes(event, paths_by_session[session], end))
        output.append(event)
    return output


def _source_inputs(repository_root: Path, config_path: Path) -> tuple[list[str], dict[str, tuple[Path, Path]], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    payload, config_sha, configs = _load_config(config_path)
    dates, sources, source_manifest = frozen_run._source_plan(repository_root)
    if len(dates) != 42:
        raise RegimeStudyError(f"expected 42 intended calendar dates, found {len(dates)}")
    nonstandard = [day for day in dates if source_manifest["session_contract_by_date"][day]["scheduled_early_close"]]
    eligible = [day for day in dates if day not in nonstandard]
    if len(nonstandard) != 1 or nonstandard != ["2025-12-24"] or len(eligible) != 41:
        raise RegimeStudyError(f"unexpected Dec/Jan normal-session eligibility: {nonstandard}/{len(eligible)}")
    return eligible, sources, source_manifest, payload, config_sha, configs


def _valid_day_checkpoint(checkpoint_path: Path, event_path: Path, day: str,
                          source_sha: str, config_sha: str, tape_sha: str,
                          semantic_sha: str) -> bool:
    if not checkpoint_path.is_file() or not event_path.is_file():
        return False
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return checkpoint_matches(checkpoint, date=day, source_sha256=source_sha,
                              config_sha256=config_sha, output_sha256=sha256_file(event_path),
                              candidate_tape_sha256=tape_sha, semantic_sha256=semantic_sha)


def _validate_resume_root(output_root: Path, *, config_sha: str, semantic_sha: str,
                          dates: Sequence[str], source_manifest: Mapping[str, Any]) -> None:
    manifest_path = output_root / "run-manifest.json"
    if not manifest_path.is_file():
        raise RegimeStudyError("resume root has no run-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "COMPLETE":
        raise RegimeStudyError("completed immutable study cannot be resumed")
    expected = {
        "run_id": RUN_ID, "config_sha256": config_sha,
        "candidate_tape_semantic_sha256": semantic_sha,
        "eligible_dates": list(dates),
        "source_manifests": {"base": source_manifest["base_manifest_sha256"],
                             "dec_jan_extension": source_manifest["extension_manifest_sha256"]},
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RegimeStudyError(f"resume run identity differs in fields: {mismatches}")
    allowed_root_files = {
        "feature-definitions.json", "run-manifest.json", "source-coverage.json",
        "smoke-status.json", "summary.json", "report.md", "resiliency-results.json",
        "mlofi-persistence-results.json", "aggression-depth-impact-results.json",
        "volatility-results.json", "daily-stability.json", "monthly-stability.json",
        "daily-event-summary.json", "interaction-results.json", "family-results.json", "barrier-results.json",
        "mfe-mae-results.json",
    }
    allowed_dirs = {"dates", "checkpoints", "bucketed-dates"}
    unknown: list[str] = []
    for path in output_root.rglob("*"):
        rel = path.relative_to(output_root)
        parts = rel.parts
        if path.is_dir():
            if len(parts) != 1 or parts[0] not in allowed_dirs:
                unknown.append(str(rel))
        elif len(parts) == 1:
            if parts[0] not in allowed_root_files:
                unknown.append(str(rel))
        elif len(parts) == 2:
            if parts[0] == "checkpoints" and parts[1].endswith(".json"):
                continue
            if parts[0] in {"dates", "bucketed-dates"} and parts[1].endswith("-events.jsonl.gz"):
                continue
            unknown.append(str(rel))
        else:
            unknown.append(str(rel))
    if unknown:
        raise RegimeStudyError(f"unrecognized files in immutable resume root: {sorted(unknown)}")


def _process_date(repository_root: Path, output_root: Path, day: str,
                  sources: Mapping[str, tuple[Path, Path]], source_manifest: Mapping[str, Any],
                  tape_path: Path, semantic_sha: str,
                  configs: Mapping[str, Mapping[str, Any]], config_sha: str) -> dict[str, Any]:
    rel_by_path = {row["path"]: row["sha256"] for row in source_manifest["input_files"]}
    source_id = frozen_run._canonical_sha([
        {"path": str(p.relative_to(repository_root)), "sha256": rel_by_path[str(p.relative_to(repository_root))]}
        for p in sources[day]
    ])
    date_dir = output_root / "dates"
    event_path = date_dir / f"{day}-events.jsonl.gz"
    checkpoint_path = output_root / "checkpoints" / f"{day}.json"
    tape_sha = sha256_file(tape_path)
    if event_path.exists() or checkpoint_path.exists():
        if _valid_day_checkpoint(checkpoint_path, event_path, day, source_id, config_sha,
                                 tape_sha, semantic_sha):
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        raise RegimeStudyError(f"existing per-date output/checkpoint is stale or mismatched: {day}")
    tape = candidate_tape.load_tape(tape_path, source_sha256=source_id,
                                    semantic_sha256=semantic_sha)
    if tape.metadata.get("date") != day:
        raise RegimeStudyError(f"candidate tape date mismatch while processing {day}")
    windows = tape.metadata["session_windows"]
    market = _load_market_day(sources[day], windows, progress_date=day)
    rows = _event_rows_for_date(day, tape, configs, market, windows)
    del tape, market
    _jsonl_gz_write(event_path, rows)
    payload = {
        "status": "DATE_COMPLETE", "date": day, "source_sha256": source_id,
        "config_sha256": config_sha, "candidate_tape_sha256": sha256_file(
            repository_root / PRIOR_RUN_ROOT / "tapes" / f"{day}-{candidate_tape.TAPE_FILENAME}"),
        "candidate_tape_semantic_sha256": semantic_sha,
        "event_output": str(event_path.relative_to(repository_root)),
        "output_sha256": sha256_file(event_path), "event_count": len(rows),
        "events_by_family": dict(sorted(__import__("collections").Counter(r["family"] for r in rows).items())),
    }
    _json_write(checkpoint_path, payload)
    print(f"REGIME_DATE_COMPLETE={day} events={len(rows)} output_sha256={payload['output_sha256']}", flush=True)
    return payload


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


CONTINUOUS_FEATURES = (
    "resiliency_30s_score", "resiliency_60s_score", "resiliency_120s_score",
    "pre_1s_nMLOFI", "pre_2s_nMLOFI", "pre_5s_nMLOFI",
    "pre_1s_raw_inverse_level_top5_ofi", "pre_2s_raw_inverse_level_top5_ofi",
    "pre_5s_raw_inverse_level_top5_ofi", "signed_persistence_relative_to_reversal",
    "same_sign_bin_fraction", "pre_1s_aggression_to_depth", "pre_2s_aggression_to_depth",
    "pre_1s_price_impact_abs_ticks_per_raw_ofi", "pre_2s_price_impact_abs_ticks_per_raw_ofi",
    "pre_30s_realized_mid_volatility_ticks", "pre_120s_realized_mid_volatility_ticks",
    "pre_1s_price_movement_per_aggressive_contract", "pre_2s_price_movement_per_aggressive_contract",
)


def _assign_expanding_buckets(rows: list[dict[str, Any]], histories: dict[tuple[str, str], list[float]]) -> None:
    for row in rows:
        family = str(row["family"])
        for feature in CONTINUOUS_FEATURES:
            value = row.get(feature)
            history = histories.setdefault((family, feature), [])
            row[f"bucket_{feature}"] = expanding_quintile(value, history)
        rv = row.get("pre_30s_realized_mid_volatility_ticks")
        rv_history = histories.setdefault((family, "pre_30s_realized_mid_volatility_ticks"), [])
        row["volatility_state_30s"] = volatility_state(rv, rv_history)
        signed = row.get("signed_persistence_relative_to_reversal")
        row["resiliency_speed_state"] = "RESILIENCY_INSUFFICIENT"
        resiliency = row.get("resiliency_60s_score")
        hist_res = histories.setdefault((family, "resiliency_60s_score"), [])
        if resiliency is not None and len(hist_res) >= MIN_EXPANDING_HISTORY:
            row["resiliency_speed_state"] = "FAST" if float(resiliency) >= float(np.median(hist_res)) else "SLOW"
        row["flow_state"] = row.get("flow_state", "FLOW_NEUTRAL")
        aggr = row.get("pre_1s_aggression_to_depth")
        impact = row.get("pre_1s_price_impact_abs_ticks_per_raw_ofi")
        ah = histories.setdefault((family, "pre_1s_aggression_to_depth"), [])
        ih = histories.setdefault((family, "pre_1s_price_impact_abs_ticks_per_raw_ofi"), [])
        row["aggression_state"] = ("INSUFFICIENT_HISTORY" if aggr is None or len(ah) < MIN_EXPANDING_HISTORY
                                    else "HIGH" if float(aggr) >= float(np.median(ah)) else "LOW")
        row["price_impact_state"] = ("INSUFFICIENT_HISTORY" if impact is None or len(ih) < MIN_EXPANDING_HISTORY
                                     else "LOW" if float(impact) <= float(np.median(ih)) else "HIGH")
    # Only after every same-date event has received thresholds from prior dates
    # may this date contribute values to the next date's calibration history.
    for row in rows:
        family = str(row["family"])
        for feature in CONTINUOUS_FEATURES:
            value = row.get(feature)
            if value is not None and math.isfinite(float(value)):
                histories.setdefault((family, feature), []).append(float(value))


def _as_float(values: Sequence[Any]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    v = sorted(_as_float(values))
    if not v:
        return {"count": 0}
    arr = np.asarray(v, dtype=np.float64)
    trim = int(len(arr) * 0.05)
    trimmed = arr[trim:len(arr) - trim] if len(arr) - 2 * trim > 0 else arr
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "count": int(len(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "trimmed_mean_5pct": float(np.mean(trimmed)), "p25": float(np.quantile(arr, .25)),
        "p75": float(np.quantile(arr, .75)), "stddev": sd,
        "standard_error": sd / math.sqrt(len(arr)),
        "ci95_low": float(np.mean(arr) - 1.96 * sd / math.sqrt(len(arr))),
        "ci95_high": float(np.mean(arr) + 1.96 * sd / math.sqrt(len(arr))),
        "positive_fraction": float(np.mean(arr > 0)), "negative_fraction": float(np.mean(arr < 0)),
        "zero_fraction": float(np.mean(arr == 0)),
    }


_OUTCOME_FIELDS = tuple(
    [f"markout_{ms}ms_ticks" for ms in MARKOUT_HORIZONS_MS]
    + [name for ms in EXCURSION_HORIZONS_MS
       for name in (f"mfe_{ms}ms_ticks", f"mae_{ms}ms_ticks")]
    + [name for favorable, adverse in BARRIER_PAIRS
       for name in (f"barrier_{favorable}_{adverse}_outcome",
                    f"barrier_{favorable}_{adverse}_seconds")]
)


def _bucket_summaries(events: Iterable[Mapping[str, Any]], feature: str,
                     bucket_field: str, group_label: str) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        keys = {"family", "date", bucket_field, *_OUTCOME_FIELDS}
        groups[(str(row["family"]), str(row.get(bucket_field, "INSUFFICIENT_HISTORY")))].append(
            {key: row.get(key) for key in keys})
    result: dict[str, Any] = {"feature": feature, "bucket_field": bucket_field, "group": group_label,
                              "families": {}, "pooled_secondary": {}}
    horizons = [f"markout_{ms}ms_ticks" for ms in MARKOUT_HORIZONS_MS]
    for (family, bucket), rows in sorted(groups.items()):
        dest = result["families"].setdefault(family, {})
        summary = {"event_count": len(rows), "active_dates": len({r["date"] for r in rows}),
                   "markouts": {name: _distribution([r.get(name) for r in rows]) for name in horizons},
                   "mfe_mae": {}, "barriers": {}}
        for ms in EXCURSION_HORIZONS_MS:
            mfe = _as_float([r.get(f"mfe_{ms}ms_ticks") for r in rows])
            mae = _as_float([r.get(f"mae_{ms}ms_ticks") for r in rows])
            summary["mfe_mae"][str(ms)] = {
                "mfe": _distribution(mfe), "mae_magnitude": _distribution(mae),
                **{f"p_mfe_ge_{n}": float(np.mean(np.asarray(mfe) >= n)) if mfe else None
                   for n in (1, 2, 4, 8)},
                **{f"p_mae_ge_{n}": float(np.mean(np.asarray(mae) >= n)) if mae else None
                   for n in (1, 2, 4, 8)},
            }
        for favorable, adverse in BARRIER_PAIRS:
            tag = f"{favorable}_{adverse}"
            vals = [r.get(f"barrier_{tag}_outcome") for r in rows]
            valid = [x for x in vals if x is not None]
            times_f = [r.get(f"barrier_{tag}_seconds") for r in rows if r.get(f"barrier_{tag}_outcome") == "FAVORABLE_FIRST"]
            times_a = [r.get(f"barrier_{tag}_seconds") for r in rows if r.get(f"barrier_{tag}_outcome") == "ADVERSE_FIRST"]
            n = len(valid)
            summary["barriers"][tag] = {
                "sample_count": n,
                "favorable_first_probability": sum(x == "FAVORABLE_FIRST" for x in valid) / n if n else None,
                "adverse_first_probability": sum(x == "ADVERSE_FIRST" for x in valid) / n if n else None,
                "unresolved_probability": sum(x == "UNRESOLVED" for x in valid) / n if n else None,
                "median_seconds_to_favorable": float(np.median(_as_float(times_f))) if _as_float(times_f) else None,
                "median_seconds_to_adverse": float(np.median(_as_float(times_a))) if _as_float(times_a) else None,
            }
        daily: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            val = row.get("markout_2000ms_ticks")
            if val is not None:
                daily[str(row["date"])].append(float(val))
        day_means = [float(np.mean(v)) for v in daily.values() if len(v) >= MIN_DAILY_SAMPLE]
        summary["daily_stability"] = {
            "positive_dates": sum(x > 0 for x in day_means), "negative_dates": sum(x < 0 for x in day_means),
            "flat_dates": sum(x == 0 for x in day_means),
            "insufficient_dates": len(daily) - len(day_means),
            "median_daily_markout_2s": float(np.median(day_means)) if day_means else None,
            "p25_daily_markout_2s": float(np.quantile(day_means, .25)) if day_means else None,
            "p75_daily_markout_2s": float(np.quantile(day_means, .75)) if day_means else None,
        }
        dest[bucket] = summary
        result["pooled_secondary"].setdefault(bucket, []).extend(rows)
    for bucket, rows in list(result["pooled_secondary"].items()):
        result["pooled_secondary"][bucket] = {
            "event_count": len(rows), "active_dates": len({r["date"] for r in rows}),
            "markouts": {name: _distribution([r.get(name) for r in rows]) for name in horizons},
        }
    return result


def _monotonicity(feature_results: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family, buckets in feature_results.get("families", {}).items():
        entries = []
        for bucket, val in buckets.items():
            dist = val.get("markouts", {}).get("markout_2000ms_ticks", {})
            if bucket.startswith("Q") and dist.get("count", 0) >= MIN_DAILY_SAMPLE:
                entries.append((int(bucket[1:]), dist.get("mean")))
        entries.sort()
        values = [float(v) for _, v in entries if v is not None]
        if len(values) < 3:
            classification = "INSUFFICIENT_EVIDENCE"
        elif all(a <= b for a, b in zip(values, values[1:])) or all(a >= b for a, b in zip(values, values[1:])):
            classification = "MONOTONIC"
        elif max(values) - min(values) <= 0.25:
            classification = "NO_RELATIONSHIP"
        else:
            # A single interior turning point is considered coherent; multiple
            # sign reversals are not promoted to a relationship.
            signs = [np.sign(b - a) for a, b in zip(values, values[1:]) if b != a]
            reversals = sum(x != y for x, y in zip(signs, signs[1:]))
            classification = "COHERENT_NON_MONOTONIC" if reversals <= 1 else "UNSTABLE"
        by_family[family] = {"bucket_means_q_order": entries, "classification": classification}
    return by_family


def _cross_month_bucket_contrast(feature_result: Mapping[str, Any]) -> dict[str, Any]:
    months = feature_result.get("by_month", {})
    december = months.get("2025-12", {}).get("families", {})
    january = months.get("2026-01", {}).get("families", {})
    families = sorted(set(december) & set(january))
    details: dict[str, Any] = {}
    stable_signs: list[int] = []
    unstable = 0
    for family in families:
        contrasts: dict[str, Any] = {}
        deltas: list[float] = []
        for month, tree in (("2025-12", december), ("2026-01", january)):
            buckets = tree[family]
            vals: dict[str, float | None] = {}
            for bucket in ("Q1", "Q5"):
                dist = buckets.get(bucket, {}).get("markouts", {}).get("markout_2000ms_ticks", {})
                vals[bucket] = (float(dist["mean"]) if dist.get("count", 0) >= MIN_DAILY_SAMPLE
                                and dist.get("mean") is not None else None)
            delta = vals["Q5"] - vals["Q1"] if vals["Q5"] is not None and vals["Q1"] is not None else None
            contrasts[month] = {"q1_mean_2s_ticks": vals["Q1"],
                                "q5_mean_2s_ticks": vals["Q5"],
                                "q5_minus_q1_ticks": delta}
            if delta is not None:
                deltas.append(delta)
        same_direction = bool(len(deltas) == 2 and
                              (deltas[0] == 0 or deltas[1] == 0 or np.sign(deltas[0]) == np.sign(deltas[1])))
        if len(deltas) == 2:
            if same_direction:
                stable_signs.append(int(np.sign(np.mean(deltas))))
            else:
                unstable += 1
        details[family] = {"months": contrasts, "comparable_both_months": len(deltas) == 2,
                           "same_direction_across_months": same_direction}
    positives = sum(sign > 0 for sign in stable_signs)
    negatives = sum(sign < 0 for sign in stable_signs)
    flat = sum(sign == 0 for sign in stable_signs)
    return {
        "families": details, "comparable_families": sum(
            bool(item["comparable_both_months"]) for item in details.values()),
        "same_direction_family_count": len(stable_signs), "unstable_family_count": unstable,
        "positive_direction_family_count": positives,
        "negative_direction_family_count": negatives,
        "flat_direction_family_count": flat,
        "assessment": ("INSUFFICIENT_EVIDENCE" if len(stable_signs) < 3 else
                       "NO_Q1_Q5_DIFFERENCE" if positives == 0 and negatives == 0 else
                       "CROSS_MONTH_DIRECTION_COHERENT" if positives == 0 or negatives == 0 else
                       "FAMILY_DIRECTION_MIXED"),
    }


def _primary_research_decision(feature_results: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    contrasts = {name: result.get("cross_month_bucket_contrast", {})
                 for payload in feature_results.values() for name, result in payload.items()}
    coherent = [name for name, result in contrasts.items()
                if result.get("assessment") == "CROSS_MONTH_DIRECTION_COHERENT"]
    mixed = [name for name, result in contrasts.items()
             if result.get("assessment") == "FAMILY_DIRECTION_MIXED"]
    nontrivial = [name for name, result in contrasts.items()
                  if result.get("assessment") not in {"INSUFFICIENT_EVIDENCE", "NO_Q1_Q5_DIFFERENCE"}]
    if len(coherent) >= 3:
        decision = "ABSORPTION_REGIME_EFFECT_PROMISING_BUT_NOT_YET_STRONG"
    elif coherent:
        max_breadth = max(int(contrasts[name].get("same_direction_family_count", 0)) for name in coherent)
        decision = ("ABSORPTION_REGIME_EFFECT_FAMILY_SPECIFIC" if max_breadth <= 3
                    else "ABSORPTION_REGIME_EFFECT_PROMISING_BUT_NOT_YET_STRONG")
    elif len(mixed) >= 2:
        decision = "ABSORPTION_REGIME_EFFECT_UNSTABLE"
    elif len(nontrivial) == 0 and contrasts:
        # No preregistered minimum effect size exists; flat Q1/Q5 deltas alone
        # do not prove absence of a nonlinear relationship.
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        decision = "INSUFFICIENT_EVIDENCE"
    basis = {
        "descriptive_only": True,
        "feature_contrasts": {name: {
            "assessment": value.get("assessment"),
            "comparable_families": value.get("comparable_families"),
            "same_direction_family_count": value.get("same_direction_family_count"),
            "positive_direction_family_count": value.get("positive_direction_family_count"),
            "negative_direction_family_count": value.get("negative_direction_family_count"),
        } for name, value in contrasts.items()},
        "coherent_features": coherent, "mixed_features": mixed,
        "rule_selection_or_parameter_threshold": False,
        "note": "Cross-month Q5-minus-Q1 2-second markout direction is descriptive. Dec/Jan was previously used and is not untouched OOS.",
    }
    return decision, basis


def _daily_monthly(events: Iterable[Mapping[str, Any]],
                   bucket_fields: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    daily: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    monthly: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        month = str(row["date"])[:7]
        for field in bucket_fields:
            bucket = str(row.get(field, "INSUFFICIENT_HISTORY"))
            key = (str(row["family"]), field, bucket)
            compact = {"date": row["date"], "family": row["family"],
                       "markout_2000ms_ticks": row.get("markout_2000ms_ticks"),
                       "markout_5000ms_ticks": row.get("markout_5000ms_ticks"),
                       "markout_10000ms_ticks": row.get("markout_10000ms_ticks"),
                       "mfe_5000ms_ticks": row.get("mfe_5000ms_ticks"),
                       "mae_5000ms_ticks": row.get("mae_5000ms_ticks"),
                       "barrier_2_2_outcome": row.get("barrier_2_2_outcome")}
            daily[(str(row["date"]), *key)].append(compact)
            monthly[(month, *key)].append(compact)
            monthly[("COMBINED", *key)].append(compact)
    def pack(source: Mapping[tuple[str, str, str, str], list[Mapping[str, Any]]], date_key: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, rows in sorted(source.items()):
            period, family, field, bucket = key
            out.setdefault(period, {}).setdefault(family, {}).setdefault(field, {})[bucket] = {
                "event_count": len(rows),
                "mean_2s_markout": _distribution([r.get("markout_2000ms_ticks") for r in rows]).get("mean"),
                "median_2s_markout": _distribution([r.get("markout_2000ms_ticks") for r in rows]).get("median"),
                "mean_5s_markout": _distribution([r.get("markout_5000ms_ticks") for r in rows]).get("mean"),
                "mean_10s_markout": _distribution([r.get("markout_10000ms_ticks") for r in rows]).get("mean"),
                "mean_mfe_5s": _distribution([r.get("mfe_5000ms_ticks") for r in rows]).get("mean"),
                "mean_mae_5s": _distribution([r.get("mae_5000ms_ticks") for r in rows]).get("mean"),
            }
        return out
    daily_out: dict[str, Any] = {}
    for (day, family, field, bucket), rows in sorted(daily.items()):
        daily_out.setdefault(day, {}).setdefault(family, {}).setdefault(field, {})[bucket] = {
            "event_count": len(rows),
            "mean_2s_markout": _distribution([r.get("markout_2000ms_ticks") for r in rows]).get("mean"),
            "median_markout": _distribution([r.get("markout_2000ms_ticks") for r in rows]).get("median"),
            "mean_5s_markout": _distribution([r.get("markout_5000ms_ticks") for r in rows]).get("mean"),
            "mean_10s_markout": _distribution([r.get("markout_10000ms_ticks") for r in rows]).get("mean"),
            "mean_mfe_5s": _distribution([r.get("mfe_5000ms_ticks") for r in rows]).get("mean"),
            "mean_mae_5s": _distribution([r.get("mae_5000ms_ticks") for r in rows]).get("mean"),
            "barrier_2_2_favorable_first": _distribution([1.0 if r.get("barrier_2_2_outcome") == "FAVORABLE_FIRST" else 0.0
                                                            for r in rows if r.get("barrier_2_2_outcome") is not None]).get("mean"),
        }
    return daily_out, pack(monthly, False)


def _interaction_results(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    specs = {
        "INTERACTION_A_RESILIENCY_X_FLOW": ("resiliency_speed_state", "flow_state"),
        "INTERACTION_B_VOLATILITY_X_RESILIENCY": ("volatility_state_30s", "resiliency_speed_state"),
        "INTERACTION_C_AGGRESSION_X_IMPACT": ("aggression_state", "price_impact_state"),
    }
    result: dict[str, Any] = {}
    for name, (a, b) in specs.items():
        groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in events:
            groups[(str(row["family"]), str(row.get(a, "INSUFFICIENT")),
                    str(row.get(b, "INSUFFICIENT")))].append({
                        "date": row["date"], **{key: row.get(key) for key in (
                            "markout_2000ms_ticks", "markout_5000ms_ticks",
                            "mfe_5000ms_ticks", "mae_5000ms_ticks")}})
        result[name] = {
            "axes": [a, b],
            "families": {
                family: {
                    f"{x}__X__{y}": {
                        "event_count": len(rows), "active_dates": len({r["date"] for r in rows}),
                        "markout_2s": _distribution([r.get("markout_2000ms_ticks") for r in rows]),
                        "markout_5s": _distribution([r.get("markout_5000ms_ticks") for r in rows]),
                        "mfe_5s": _distribution([r.get("mfe_5000ms_ticks") for r in rows]),
                        "mae_5s": _distribution([r.get("mae_5000ms_ticks") for r in rows]),
                    } for (f, x, y), rows in groups.items() if f == family
                } for family in sorted({key[0] for key in groups})
            },
        }
    return result


def _interaction_results_for_spec(events: Iterable[Mapping[str, Any]], name: str,
                                  left: str, right: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        key = (str(row["family"]), str(row.get(left, "INSUFFICIENT")),
               str(row.get(right, "INSUFFICIENT")))
        groups[key].append({"date": row["date"], **{field: row.get(field) for field in (
            "markout_2000ms_ticks", "markout_5000ms_ticks", "mfe_5000ms_ticks",
            "mae_5000ms_ticks")}})
    output: dict[str, Any] = {"axes": [left, right], "families": {}}
    for family in sorted({key[0] for key in groups}):
        cells = {}
        for (group_family, x, y), rows in sorted(groups.items()):
            if group_family != family:
                continue
            cells[f"{x}__X__{y}"] = {
                "event_count": len(rows), "active_dates": len({row["date"] for row in rows}),
                "markout_2s": _distribution([row.get("markout_2000ms_ticks") for row in rows]),
                "markout_5s": _distribution([row.get("markout_5000ms_ticks") for row in rows]),
                "mfe_5s": _distribution([row.get("mfe_5000ms_ticks") for row in rows]),
                "mae_5s": _distribution([row.get("mae_5000ms_ticks") for row in rows]),
            }
        output["families"][family] = cells
    return output


def _feature_definitions() -> dict[str, Any]:
    return {
        "study": RUN_ID, "evidence_label": "REGIME_DIAGNOSTIC_ROBUSTNESS_SET",
        "dec_jan_are_not_untouched_oos": True, "direction_normalization": "+ is predicted reversal",
        "event_population": "All completed causally-valid frozen-family candidate-tape interactions for the ten selected family IDs; family-specific Class-A score/qualification recorded but not used to select the event sample.",
        "pre_event_invariant": "Only source rows with ts_recv < interaction_start_ns contribute. Fixed-depth averages use 100ms as-of samples whose timestamp is also < interaction_start_ns.",
        "mlofi": {
            "source": "validated mac_2025_mlofi_event_study.account_mbp_event semantics",
            "levels": "TOP5", "weighting": "INVERSE_LEVEL (1/(rank+1))",
            "normalization": "DEPTH_NORMALIZED using weighted mean of displayed bid/ask depth at the last executable snapshot strictly before event start",
            "actions": "A/C/M explicit price-level displayed-size delta; T aggressor size signed on passive side; R zero; rank keyed by updated price; non-executable current snapshot breaks previous-book linkage.",
            "flow_bins": "2s before event start split into 8 adjacent half-open 250ms bins",
        },
        "resiliency": {
            "lookbacks_seconds": [30, 60, 120], "qualifying_depletion": "one executable update with depth decline >= max(1 contract, 10% of immediately preceding TOP5 pressured-side depth)",
            "recovery": "(depth at first executable observation at/after depletion timestamp+h - immediate post-depletion depth)/(pre-depletion depth - immediate post-depletion depth); only h observation strictly before event start is admitted; denominator <=0 is insufficient.",
            "primary": "median 500ms recovery fraction of qualifying episodes in prior 60s; supporting 100/250/1000/2000ms and T50/T80; no qualifying observed 500ms episodes => RESILIENCY_INSUFFICIENT.",
            "episodes_are_independent_update_pulses": True,
        },
        "aggression_depth": {
            "windows_seconds": [1, 2], "aggression": "sell-aggressor volume / mean bid TOP5 depth for SELLER_ABSORPTION; buy-aggressor volume / mean ask TOP5 depth for BUYER_ABSORPTION",
            "mean_depth": "arithmetic mean of 100ms as-of executable TOP5 depth samples in the half-open pre-event window",
            "impact": "absolute direction-normalized mid change in ES ticks divided by absolute raw inverse-level TOP5 MLOFI plus 1e-9; price movement per aggressive contract is a separate descriptive ratio.",
        },
        "realized_volatility": "square root of summed squared executable-mid changes in ticks over preceding 30s and 120s; session-local; same session only.",
        "bucket_calibration": {
            "continuous": "expanding per-family empirical quintiles calibrated only on earlier eligible dates; at least 20 prior events; same-date events do not calibrate one another.",
            "volatility": "expanding per-family prior-date 20/80/95 percentiles => LOW/NORMAL/HIGH/EXTREME; at least 20 prior events.",
            "interaction_states": "resiliency FAST/SLOW relative to prior-date median; flow states by sign relative to predicted reversal; aggression and impact HIGH/LOW relative to prior-date median; no thresholds are tuned.",
        },
        "forward_path": {
            "reference": "last sparse executable ES BBO midpoint strictly before interaction start; markout midpoint is as-of target horizon (last state at/before horizon, carried forward).",
            "markouts_ms": list(MARKOUT_HORIZONS_MS), "mfe_mae_ms": list(EXCURSION_HORIZONS_MS),
            "barriers": [{"favorable_ticks": a, "adverse_ticks": b} for a, b in BARRIER_PAIRS],
            "barrier_horizon_seconds": 30,
        },
        "no_strategy_pnl": True,
    }


def _iter_bucketed(output_root: Path, dates: Sequence[str]) -> Iterable[dict[str, Any]]:
    for day in dates:
        path = output_root / "bucketed-dates" / f"{day}-events.jsonl.gz"
        yield from _read_jsonl_gz(path)


def _aggregate(output_root: Path, dates: Sequence[str], family_ids: Sequence[str],
               source_manifest: Mapping[str, Any], config_sha: str) -> dict[str, Any]:
    history: dict[tuple[str, str], list[float]] = {}
    bucketed_dir = output_root / "bucketed-dates"
    family_counts: dict[str, int] = {x: 0 for x in family_ids}
    daily_counts: dict[str, dict[str, int]] = {}
    for day in dates:
        path = output_root / "dates" / f"{day}-events.jsonl.gz"
        rows = list(_read_jsonl_gz(path))
        _assign_expanding_buckets(rows, history)
        _jsonl_gz_write(bucketed_dir / path.name, rows)
        for row in rows:
            family_counts[str(row["family"])] += 1
        daily_counts[day] = {f: sum(1 for row in rows if row["family"] == f) for f in family_ids}
        print(f"REGIME_BUCKETED_DATE={day} events={len(rows)}", flush=True)

    feature_outputs: dict[str, Any] = {}
    feature_for_file = {
        "resiliency-results.json": ["resiliency_30s_score", "resiliency_60s_score", "resiliency_120s_score"],
        "mlofi-persistence-results.json": ["pre_1s_nMLOFI", "pre_2s_nMLOFI", "pre_5s_nMLOFI",
                                           "pre_1s_raw_inverse_level_top5_ofi",
                                           "pre_2s_raw_inverse_level_top5_ofi",
                                           "pre_5s_raw_inverse_level_top5_ofi",
                                           "signed_persistence_relative_to_reversal", "same_sign_bin_fraction"],
        "aggression-depth-impact-results.json": ["pre_1s_pressured_aggressive_volume",
            "pre_2s_pressured_aggressive_volume", "pre_1s_mean_pressured_top5_depth_100ms_grid",
            "pre_2s_mean_pressured_top5_depth_100ms_grid", "pre_1s_aggression_to_depth",
            "pre_2s_aggression_to_depth", "pre_1s_price_impact_abs_ticks_per_raw_ofi",
            "pre_2s_price_impact_abs_ticks_per_raw_ofi", "pre_1s_price_movement_per_aggressive_contract",
            "pre_2s_price_movement_per_aggressive_contract"],
        "volatility-results.json": ["pre_30s_realized_mid_volatility_ticks", "pre_120s_realized_mid_volatility_ticks"],
    }
    for filename, features in feature_for_file.items():
        payload: dict[str, Any] = {}
        for feature in features:
            result = _bucket_summaries(_iter_bucketed(output_root, dates), feature,
                                       f"bucket_{feature}", feature)
            result["monotonicity"] = _monotonicity(result)
            by_month: dict[str, Any] = {}
            for month in ("2025-12", "2026-01"):
                month_rows = (row for row in _iter_bucketed(output_root, dates)
                              if str(row["date"]).startswith(month))
                month_result = _bucket_summaries(month_rows, feature,
                                                 f"bucket_{feature}", feature)
                month_result["monotonicity"] = _monotonicity(month_result)
                by_month[month] = month_result
            result["by_month"] = by_month
            result["cross_month_bucket_contrast"] = _cross_month_bucket_contrast(result)
            payload[feature] = result
        _json_write(output_root / filename, payload)
        feature_outputs[filename] = payload

    bucket_fields = [f"bucket_{name}" for name in CONTINUOUS_FEATURES]
    daily, monthly = _daily_monthly(_iter_bucketed(output_root, dates), bucket_fields)
    _json_write(output_root / "daily-stability.json", daily)
    _json_write(output_root / "monthly-stability.json", monthly)
    interaction_specs = (
        ("resiliency_speed_state", "flow_state"),
        ("volatility_state_30s", "resiliency_speed_state"),
        ("aggression_state", "price_impact_state"),
    )
    interaction: dict[str, Any] = {}
    for name, (left, right) in zip((
            "INTERACTION_A_RESILIENCY_X_FLOW",
            "INTERACTION_B_VOLATILITY_X_RESILIENCY",
            "INTERACTION_C_AGGRESSION_X_IMPACT"), interaction_specs):
        # The helper is deliberately called on a fresh disk iterator per
        # predeclared interaction; no all-event dictionary list is retained.
        interaction[name] = _interaction_results_for_spec(
            _iter_bucketed(output_root, dates), name, left, right)
    _json_write(output_root / "interaction-results.json", interaction)

    compact_by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in family_ids}
    month_rows: dict[str, list[dict[str, Any]]] = {"2025-12": [], "2026-01": []}
    daily_markouts: dict[str, list[float]] = defaultdict(list)
    event_count = 0
    for row in _iter_bucketed(output_root, dates):
        family = str(row["family"])
        compact = {key: row.get(key) for key in (
            "date", "family", "frozen_class_a_qualified", "core_quality_score",
            *_OUTCOME_FIELDS)}
        compact_by_family[family].append(compact)
        month_rows[str(row["date"])[:7]].append(compact)
        if row.get("markout_2000ms_ticks") is not None:
            daily_markouts[str(row["date"])].append(float(row["markout_2000ms_ticks"]))
        event_count += 1

    family_results: dict[str, Any] = {}
    for family in family_ids:
        rows = compact_by_family[family]
        q = [r.get("markout_2000ms_ticks") for r in rows]
        family_results[family] = {
            "event_count": len(rows), "active_dates": len({r["date"] for r in rows}),
            "class_a_qualified_count": sum(bool(r["frozen_class_a_qualified"]) for r in rows),
            "core_quality_distribution": _distribution([r["core_quality_score"] for r in rows]),
            "mean_2s_markout": _distribution(q).get("mean"),
            "resiliency_relationship": _monotonicity(feature_outputs["resiliency-results.json"]["resiliency_60s_score"]).get(family),
            "mlofi_relationship": _monotonicity(feature_outputs["mlofi-persistence-results.json"]["pre_5s_nMLOFI"]).get(family),
            "aggression_relationship": _monotonicity(feature_outputs["aggression-depth-impact-results.json"]["pre_1s_aggression_to_depth"]).get(family),
            "volatility_relationship": _monotonicity(feature_outputs["volatility-results.json"]["pre_30s_realized_mid_volatility_ticks"]).get(family),
        }
    _json_write(output_root / "family-results.json", family_results)
    daily_event_summary = {
        day: {"event_count": len(values), "mean_2s_markout": _distribution(values).get("mean"),
              "median_2s_markout": _distribution(values).get("median"),
              "positive_fraction": _distribution(values).get("positive_fraction"),
              "negative_fraction": _distribution(values).get("negative_fraction")}
        for day, values in sorted(daily_markouts.items())
    }
    _json_write(output_root / "daily-event-summary.json", daily_event_summary)

    barrier_payload: dict[str, Any] = {}; excursion_payload: dict[str, Any] = {}
    for family in family_ids:
        rows = compact_by_family[family]
        barrier_payload[family] = {}
        for favorable, adverse in BARRIER_PAIRS:
            tag = f"{favorable}_{adverse}"
            vals = [r.get(f"barrier_{tag}_outcome") for r in rows if r.get(f"barrier_{tag}_outcome") is not None]
            ftime = [r.get(f"barrier_{tag}_seconds") for r in rows if r.get(f"barrier_{tag}_outcome") == "FAVORABLE_FIRST"]
            atime = [r.get(f"barrier_{tag}_seconds") for r in rows if r.get(f"barrier_{tag}_outcome") == "ADVERSE_FIRST"]
            n = len(vals)
            barrier_payload[family][tag] = {
                "sample_count": n,
                "favorable_first_probability": sum(x == "FAVORABLE_FIRST" for x in vals)/n if n else None,
                "adverse_first_probability": sum(x == "ADVERSE_FIRST" for x in vals)/n if n else None,
                "unresolved_probability": sum(x == "UNRESOLVED" for x in vals)/n if n else None,
                "median_seconds_to_favorable": float(np.median(_as_float(ftime))) if _as_float(ftime) else None,
                "median_seconds_to_adverse": float(np.median(_as_float(atime))) if _as_float(atime) else None,
            }
        excursion_payload[family] = {
            str(ms): {"mfe": _distribution([r.get(f"mfe_{ms}ms_ticks") for r in rows]),
                      "mae_magnitude": _distribution([r.get(f"mae_{ms}ms_ticks") for r in rows]),
                      **{f"p_mfe_ge_{n}": float(np.mean([float(r[f"mfe_{ms}ms_ticks"]) >= n for r in rows if r.get(f"mfe_{ms}ms_ticks") is not None])) if any(r.get(f"mfe_{ms}ms_ticks") is not None for r in rows) else None for n in (1,2,4,8)},
                      **{f"p_mae_ge_{n}": float(np.mean([float(r[f"mae_{ms}ms_ticks"]) >= n for r in rows if r.get(f"mae_{ms}ms_ticks") is not None])) if any(r.get(f"mae_{ms}ms_ticks") is not None for r in rows) else None for n in (1,2,4,8)}}
            for ms in EXCURSION_HORIZONS_MS
        }
    _json_write(output_root / "barrier-results.json", barrier_payload)
    _json_write(output_root / "mfe-mae-results.json", excursion_payload)

    completed_dates = [day for day in dates
                       if (output_root / "checkpoints" / f"{day}.json").is_file()
                       and (output_root / "dates" / f"{day}-events.jsonl.gz").is_file()]
    if completed_dates != list(dates) or sum(family_counts.values()) != event_count or event_count <= 0:
        raise RegimeStudyError("aggregation invariant failed: completed-date/event/family counts do not reconcile")
    primary_decision, decision_basis = _primary_research_decision(feature_outputs)

    month_results: dict[str, Any] = {}
    for month in ("2025-12", "2026-01"):
        rows = month_rows[month]
        month_results[month] = {
            "event_count": len(rows), "families": len({r["family"] for r in rows}),
            "active_dates": len({r["date"] for r in rows}),
            "markout_2s": _distribution([r.get("markout_2000ms_ticks") for r in rows]),
            "markout_5s": _distribution([r.get("markout_5000ms_ticks") for r in rows]),
            "markout_30s": _distribution([r.get("markout_30000ms_ticks") for r in rows]),
        }
    month_results["COMBINED"] = {
        "event_count": event_count, "active_dates": sum(bool(v) for v in daily_counts.values()),
        "markout_2s": _distribution([r.get("markout_2000ms_ticks") for rows in compact_by_family.values() for r in rows]),
        "markout_5s": _distribution([r.get("markout_5000ms_ticks") for rows in compact_by_family.values() for r in rows]),
        "markout_30s": _distribution([r.get("markout_30000ms_ticks") for rows in compact_by_family.values() for r in rows]),
    }
    coverage_path = output_root / "source-coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    coverage.update({
        "status": "PASS", "intended_dates": 42, "eligible_dates": list(dates),
        "eligible_date_count": len(dates), "excluded_dates": [{"date": "2025-12-24",
        "reason": "SCHEDULED_EARLY_CLOSE_NOT_NORMAL_FULL_SESSION"}],
        "completed_dates": [d for d in dates if (output_root / "checkpoints" / f"{d}.json").is_file()],
        "source_manifest_sha256": source_manifest,
    })
    _json_write(output_root / "source-coverage.json", coverage)
    summary = {
        "status": "PASS", "study": RUN_ID, "dataset": "DEC_2025 + JAN_2026",
        "evidence_label": "REGIME_DIAGNOSTIC_ROBUSTNESS_SET",
        "dec_jan_are_not_untouched_oos": True, "frozen_config_sha256": config_sha,
        "eligible_dates": list(dates), "completed_dates": coverage["completed_dates"],
        "excluded_dates": coverage["excluded_dates"], "families": len(family_ids),
        "event_count": event_count, "event_count_by_family": family_counts,
        "daily_event_count_by_family": daily_counts, "monthly_results": month_results,
        "daily_event_summary": daily_event_summary,
        "feature_cross_month_contrasts": {
            feature: result["cross_month_bucket_contrast"]
            for payload in feature_outputs.values() for feature, result in payload.items()},
        "primary_research_decision": primary_decision,
        "primary_research_decision_basis": decision_basis,
        "daily_stability_date_count": len(daily), "no_strategy_pnl": True,
        "optimization_performed": False, "pnl_optimization_performed": False,
        "untouched_oos_accessed": False,
    }
    _json_write(output_root / "summary.json", summary)
    return summary


def _write_report(output_root: Path, summary: Mapping[str, Any], feature_results: Mapping[str, Any],
                  interactions: Mapping[str, Any], family_results: Mapping[str, Any]) -> None:
    lines = [
        "# ES absorption regime event study — Dec 2025 + Jan 2026", "",
        "This is a descriptive robustness-set event study, not untouched OOS or final validation. No strategy PnL, trade selection, or parameter optimization was performed.", "",
        f"- Eligible sessions: {len(summary['eligible_dates'])}; completed: {len(summary['completed_dates'])}; excluded: 2025-12-24 (scheduled early close).",
        f"- Frozen ten-family config SHA-256: `{summary['frozen_config_sha256']}`.",
        f"- Completed core events: {summary['event_count']:,}; families: {summary['families']}; ES native MBP-10 only.", "",
        f"- Primary descriptive classification: **{summary['primary_research_decision']}**.",
        "- Dec/Jan was previously used: this is a robustness-set diagnostic, not untouched OOS.",
        "- Class-A score and qualification are reported as labels and do not filter the event sample.",
        "- No strategy PnL, optimization, or parameter selection was performed.", "",
        "## Monthly event markouts", "",
        "| Period | Events | 2s mean | 2s median | 5s mean | 30s mean |", "|---|---:|---:|---:|---:|---:|",
    ]
    for period, row in summary["monthly_results"].items():
        m2, m5, m30 = row["markout_2s"], row["markout_5s"], row["markout_30s"]
        lines.append(f"| {period} | {row['event_count']} | {m2.get('mean')} | {m2.get('median')} | {m5.get('mean')} | {m30.get('mean')} |")
    lines += ["", "## First-wave pre-event variables", "",
              "Each continuous variable is bucketed per family using only earlier eligible dates; same-date interactions never calibrate one another.", ""]
    for fname, groups in feature_results.items():
        lines.append(f"### {fname}")
        for feature, res in groups.items():
            contrast = res["cross_month_bucket_contrast"]
            lines.append(f"- `{feature}`: {contrast['assessment']}; cross-month comparable families={contrast['comparable_families']}; same-direction family contrasts={contrast['same_direction_family_count']} (positive={contrast['positive_direction_family_count']}, negative={contrast['negative_direction_family_count']}).")
            for month, month_result in res.get("by_month", {}).items():
                cells = []
                for family, buckets in month_result.get("families", {}).items():
                    for bucket, metrics in buckets.items():
                        dist = metrics.get("markouts", {}).get("markout_2000ms_ticks", {})
                        if bucket.startswith("Q"):
                            cells.append(f"{family} {bucket}: n={dist.get('count', 0)}, mean={dist.get('mean')}")
                lines.append(f"  - {month} 2s bucket means: " + ("; ".join(cells) if cells else "no Q-bucket samples") + ".")
        lines.append("")
    lines += ["## Predeclared interactions", ""]
    for name, payload in interactions.items():
        event_n = sum(v["event_count"] for fam in payload["families"].values() for v in fam.values())
        lines.append(f"- {name}: {event_n} family/state observations; descriptive only.")
    lines += ["", "## Family-level event counts", "", "| Family | Events | Active dates | Class-A-qualified | 2s mean |", "|---|---:|---:|---:|---:|"]
    for family, row in family_results.items():
        lines.append(f"| {family} | {row['event_count']} | {row['active_dates']} | {row['class_a_qualified_count']} | {row['mean_2s_markout']} |")
    lines += ["", "## Additional artifacts", "",
        "Detailed distributions and date/family splits are in the four feature-results JSON files. Daily and monthly expanding-bucket summaries are in `daily-stability.json` and `monthly-stability.json`; raw per-day event summaries are in `daily-event-summary.json`; family results, predeclared interactions, first-touch barriers, and MFE/MAE are in their named JSON artifacts.",
        "", "## Caveats", "",
        "Pre-event features use timestamps strictly earlier than interaction start. Event markouts are carried-forward ES BBO midpoint states at the fixed horizon; sparse BBO path extrema and first-touch barriers use observed executable quote transitions. Resiliency depletion pulses use the documented 10%/one-contract criterion and are aggregate-book diagnostics, not queue inference.",
        "",
        "This two-month population has prior exposure and cannot support untouched-OOS or production claims. Any subgroup pattern is descriptive and was not converted into a rule.", ""]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(repository_root: Path = Path("."), *, config_path: Path = CONFIG_PATH,
        output_root: Path = OUTPUT_ROOT, resume: bool = False,
        smoke_date: str | None = None, smoke_only: bool = False) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = config_path if config_path.is_absolute() else repository_root / config_path
    output_root = output_root if output_root.is_absolute() else repository_root / output_root
    output_root = output_root.resolve()
    if output_root.exists() and not resume:
        raise RegimeStudyError(f"immutable study output root already exists: {output_root}")
    dates, sources, source_manifest, config_payload, config_sha, configs = _source_inputs(repository_root, config_path)
    if smoke_date is not None and smoke_date not in dates:
        raise RegimeStudyError(f"smoke date is not an eligible normal session: {smoke_date}")
    tapes, semantic_sha = _validate_prior_tapes(repository_root, dates, sources,
                                               source_manifest, config_sha, configs)
    if output_root.exists():
        _validate_resume_root(output_root, config_sha=config_sha, semantic_sha=semantic_sha,
                              dates=dates, source_manifest=source_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    _json_write(output_root / "feature-definitions.json", _feature_definitions())
    prior_run = json.loads((repository_root / PRIOR_RUN_ROOT / "run-manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "RUNNING", "run_id": RUN_ID, "config_sha256": config_sha,
        "expected_config_sha256": EXPECTED_CONFIG_SHA256, "candidate_tape_semantic_sha256": semantic_sha,
        "candidate_tape_semantic_provenance": "Prior completed run progress.json, summary.json, tape hashes, and tape sidecars agree; no current candidate-tape rebuild was performed.",
        "prior_run_root": str(PRIOR_RUN_ROOT), "prior_run_config_sha256": prior_run.get("config_sha256"),
        "source_manifests": {"base": source_manifest["base_manifest_sha256"],
                              "dec_jan_extension": source_manifest["extension_manifest_sha256"]},
        "source_roots": [str(p.relative_to(repository_root)) for p in (sources[dates[0]])],
        "source_input_files": [{key: item.get(key) for key in
                                 ("path", "sha256", "bytes", "schema", "symbol", "date")}
                                for item in source_manifest["input_files"]],
        "eligible_dates": dates, "intended_dates": 42,
        "excluded_dates": [{"date": "2025-12-24", "reason": "SCHEDULED_EARLY_CLOSE_NOT_NORMAL_FULL_SESSION"}],
        "dec_jan_are_not_untouched_oos": True, "data_downloaded": False,
        "untouched_oos_accessed": False, "optimization_performed": False,
        "pnl_optimization_performed": False, "strategy_pnl_calculated": False,
        "source_model": "NATIVE_MBP10", "session_order": list(SESSION_NAMES),
    }
    _json_write(output_root / "run-manifest.json", manifest)
    _write_source_coverage_preflight(output_root, dates, source_manifest, configs)
    to_process = [smoke_date] if smoke_only and smoke_date else dates
    for index, day in enumerate(to_process, 1):
        print(f"REGIME_DATE_START={index}/{len(to_process)} {day}", flush=True)
        _process_date(repository_root, output_root, day, sources, source_manifest,
                      tapes[day], semantic_sha, configs, config_sha)
    if smoke_only:
        _json_write(output_root / "smoke-status.json", {"status": "SMOKE_PASS", "date": smoke_date,
                    "config_sha256": config_sha, "source_manifest_sha256": {
                        "base": source_manifest["base_manifest_sha256"],
                        "extension": source_manifest["extension_manifest_sha256"]},
                    "event_count": json.loads((output_root / "checkpoints" / f"{smoke_date}.json").read_text())["event_count"]})
        return {"status": "SMOKE_PASS", "date": smoke_date}
    checkpoints = []
    for day in dates:
        checkpoint_path = output_root / "checkpoints" / f"{day}.json"
        event_path = output_root / "dates" / f"{day}-events.jsonl.gz"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else None
        rel_by_path = {row["path"]: row["sha256"] for row in source_manifest["input_files"]}
        source_id = frozen_run._canonical_sha([
            {"path": str(p.relative_to(repository_root)), "sha256": rel_by_path[str(p.relative_to(repository_root))]}
            for p in sources[day]
        ])
        tape_sha = sha256_file(tapes[day])
        if not checkpoint_matches(checkpoint or {}, date=day, source_sha256=source_id,
                                  config_sha256=config_sha,
                                  output_sha256=sha256_file(event_path) if event_path.is_file() else None,
                                  candidate_tape_sha256=tape_sha,
                                  semantic_sha256=semantic_sha):
            raise RegimeStudyError(f"required day missing/invalid at aggregation: {day}")
        checkpoints.append(checkpoint)
    family_ids = tuple(config_payload["families"][i]["family"] for i in range(len(config_payload["families"])))
    summary = _aggregate(output_root, dates, family_ids, {
        "base": source_manifest["base_manifest_sha256"],
        "dec_jan_extension": source_manifest["extension_manifest_sha256"],
    }, config_sha)
    feature_results = {
        "resiliency-results.json": json.loads((output_root / "resiliency-results.json").read_text(encoding="utf-8")),
        "mlofi-persistence-results.json": json.loads((output_root / "mlofi-persistence-results.json").read_text(encoding="utf-8")),
        "aggression-depth-impact-results.json": json.loads((output_root / "aggression-depth-impact-results.json").read_text(encoding="utf-8")),
        "volatility-results.json": json.loads((output_root / "volatility-results.json").read_text(encoding="utf-8")),
    }
    interactions = json.loads((output_root / "interaction-results.json").read_text(encoding="utf-8"))
    family_results = json.loads((output_root / "family-results.json").read_text(encoding="utf-8"))
    _write_report(output_root, summary, feature_results, interactions, family_results)
    manifest.update({"status": "COMPLETE", "completed_dates": dates,
                     "event_count": summary["event_count"],
                     "source_coverage_sha256": sha256_file(output_root / "source-coverage.json"),
                     "summary_sha256": sha256_file(output_root / "summary.json"),
                     "report_sha256": sha256_file(output_root / "report.md"),
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    _json_write(output_root / "run-manifest.json", manifest)
    return summary


def _write_source_coverage_preflight(output_root: Path, dates: Sequence[str],
                                     source_manifest: Mapping[str, Any],
                                     configs: Mapping[str, Mapping[str, Any]]) -> None:
    _json_write(output_root / "source-coverage.json", {
        "status": "SOURCE_PREFLIGHT_PASS", "intended_dates": 42, "eligible_dates": list(dates),
        "eligible_date_count": len(dates), "excluded_dates": [{"date": "2025-12-24",
        "reason": "SCHEDULED_EARLY_CLOSE_NOT_NORMAL_FULL_SESSION"}],
        "source_manifest_sha256": {"base": source_manifest["base_manifest_sha256"],
                                    "dec_jan_extension": source_manifest["extension_manifest_sha256"]},
        "source_models": ["NATIVE_MBP10"], "data_downloaded": False,
        "required_raw_files_verified_by_existing_runner": len(source_manifest["input_files"]),
        "source_input_files": [{key: item.get(key) for key in
                                ("path", "sha256", "bytes", "schema", "symbol", "date")}
                               for item in source_manifest["input_files"]],
        "candidate_family_count": len(configs), "family_ids": list(configs),
        "required_schema": "GLBX.MDP3 / mbp-10", "price_scale": RAW_PRICE_SCALE,
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-date", help="process only one eligible date for a real-data smoke")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.repository_root, config_path=args.config, output_root=args.output_root,
                     resume=args.resume, smoke_date=args.smoke_date, smoke_only=args.smoke_only)
        print(f"ABSORPTION_REGIME_STUDY_STATUS={result['status']}", flush=True)
        if result["status"] == "PASS":
            print(f"ABSORPTION_REGIME_EVENT_COUNT={result['event_count']}", flush=True)
        return 0
    except (RegimeStudyError, frozen_run.RobustnessRunError,
            candidate_tape.CandidateTapeError, OSError, ValueError, KeyError, TypeError) as exc:
        try:
            manifest_path = args.output_root / "run-manifest.json"
            if not args.output_root.is_absolute():
                manifest_path = args.repository_root / manifest_path
            if manifest_path.is_file():
                failed = json.loads(manifest_path.read_text(encoding="utf-8"))
                if failed.get("status") != "COMPLETE":
                    failed.update({"status": "FAILED", "failure": f"{type(exc).__name__}: {exc}"})
                    _json_write(manifest_path, failed)
        except (OSError, ValueError, TypeError):
            pass
        print(f"ABSORPTION_REGIME_STUDY_STATUS=FAIL error={exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
