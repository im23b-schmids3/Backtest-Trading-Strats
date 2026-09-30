"""Frozen-event April/October 2025 absorption context normalization study.

This is descriptive research, not a strategy optimizer. Candidate discovery is
read from the sealed MAC 2025 candidate tapes; pre-event market context is
streamed from the corresponding native ES MBP-10 files. Percentile references
are updated only after a date is complete, so a date never calibrates itself.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_mlofi_event_study as mlofi
from .mac_2025_candidate_tape import _default_parameters, _qualifies, _quality
from .model import L2Config

RUN_ID = "CMEOrderflow_ABSORPTION_RELATIVE_NORMALIZATION_APR_OCT_2025_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
DATA_ROOT = Path("data/databento/mac-2025-native-es-mbp10-final-reduced")
TRAIN_TAPE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE/candidate-tapes/tapes")
OCT_TAPE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928/october-candidate-tapes/tapes")
LIVE_CONFIG_PATH = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_TRAIN_20260928/live-config-source-snapshot.json")
EXPECTED_STRATEGY_MANIFEST_SHA = "6c55756af201a20e11bfb87753142f9886ebfa26c796a37598a726f0ef66f56f"
EXPECTED_RUNTIME_CONTRACT_SHA = "f48272561479da308554402cbf93c5975def829e4136fcd3122180fda611fbff"
EXPECTED_TAPE_SEMANTIC_SHA = "8a7d959d7a4ead731f7ca04960b48fd49d48818d505e92d81ea681219ee1023f"
PATH_OUTCOME_VERSION = 2
TICK = 0.25
TOD_BIN_NS = 300_000_000_000  # fixed five-minute session-relative bins
SAMPLE_STEP_NS = 5_000_000_000
RESILIENCY_SAMPLE_NS = 50_000_000
HORIZONS_MS = (250, 500, 1_000, 2_000, 5_000, 10_000, 30_000)
PATH_HORIZONS_MS = (1_000, 2_000, 5_000, 10_000, 30_000)
BARRIERS = ((1, 1), (2, 2), (4, 4), (8, 4), (12, 6))
FEATURES = (
    "RV_30S_RAW", "RV_120S_RAW", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSIVE_VOLUME_2S_RAW",
    "AGGRESSION_TO_DEPTH_1S", "AGGRESSION_TO_DEPTH_2S", "TOP5_SAME_SIDE_DEPTH_RAW",
    "TOP10_SAME_SIDE_DEPTH_RAW", "OPPOSITE_TOP5_DEPTH_RAW", "DEPTH_ASYMMETRY_RAW",
    "DEPTH_ASYMMETRY_DIRECTIONAL", "TRADES_PER_SECOND_5S", "TRADES_PER_SECOND_30S",
    "CONTRACTS_PER_SECOND_5S", "CONTRACTS_PER_SECOND_30S", "BOOK_UPDATES_PER_SECOND_5S",
    "BOOK_UPDATES_PER_SECOND_30S", "VELOCITY_2S_TICKS_PER_SECOND", "VELOCITY_10S_TICKS_PER_SECOND",
    "VELOCITY_30S_TICKS_PER_SECOND", "VELOCITY_2S_DIRECTIONAL", "VELOCITY_10S_DIRECTIONAL",
    "VELOCITY_30S_DIRECTIONAL", "ER_5S", "ER_30S", "ER_5S_DIRECTIONAL", "ER_30S_DIRECTIONAL",
    "MLOFI_1S_RAW", "MLOFI_5S_RAW", "MLOFI_1S_DEPTH_NORMALIZED", "MLOFI_5S_DEPTH_NORMALIZED",
    "MLOFI_PERSISTENCE_2S", "MLOFI_PERSISTENCE_DIRECTIONAL", "PRICE_IMPACT_PER_MLOFI_5S",
    "RESILIENCY_RECOVERY_250MS", "RESILIENCY_RECOVERY_500MS", "RESILIENCY_RECOVERY_1S",
    "RESILIENCY_RECOVERY_2S", "RESILIENCY_MEDIAN_500MS_PRIOR_60S",
)
PERCENTILE_SOURCE = {
    "RV_30S_RAW": ("RV_30S_TOD_PERCENTILE", "RV_30S_GLOBAL_PERCENTILE"),
    "RV_120S_RAW": ("RV_120S_TOD_PERCENTILE", "RV_120S_GLOBAL_PERCENTILE"),
    "AGGRESSIVE_VOLUME_1S_RAW": ("AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",),
    "AGGRESSIVE_VOLUME_2S_RAW": ("AGGRESSIVE_VOLUME_2S_TOD_PERCENTILE",),
    "AGGRESSION_TO_DEPTH_1S": ("AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE",),
    "AGGRESSION_TO_DEPTH_2S": ("AGGRESSION_TO_DEPTH_2S_TOD_PERCENTILE",),
    "TOP5_SAME_SIDE_DEPTH_RAW": ("TOP5_DEPTH_TOD_PERCENTILE",),
    "TOP10_SAME_SIDE_DEPTH_RAW": ("TOP10_DEPTH_TOD_PERCENTILE",),
    "TRADES_PER_SECOND_5S": ("TRADE_INTENSITY_TOD_PERCENTILE",),
    "CONTRACTS_PER_SECOND_5S": ("VOLUME_INTENSITY_TOD_PERCENTILE",),
    "BOOK_UPDATES_PER_SECOND_5S": ("UPDATE_INTENSITY_TOD_PERCENTILE",),
    "VELOCITY_2S_TICKS_PER_SECOND": ("VELOCITY_2S_TOD_PERCENTILE",),
    "VELOCITY_10S_TICKS_PER_SECOND": ("VELOCITY_10S_TOD_PERCENTILE",),
    "VELOCITY_30S_TICKS_PER_SECOND": ("VELOCITY_30S_TOD_PERCENTILE",),
    "ER_5S": ("ER_5S_GLOBAL_PERCENTILE",),
    "ER_30S": ("ER_30S_GLOBAL_PERCENTILE",),
    "MLOFI_5S_DEPTH_NORMALIZED": ("MLOFI_PERCENTILE_TOD",),
    "RESILIENCY_MEDIAN_500MS_PRIOR_60S": ("RESILIENCY_PERCENTILE",),
}
FAMILY_MAP = {
    "EU_CURRENT_HIGH_SWEEP": "EUROPE|EUROPE|CURRENT|HIGH",
    "EU_PRIOR_HIGH": "EUROPE|EUROPE|PRIOR|HIGH",
    "PRIOR_VAH": "EUROPE|EUROPE|PRIOR|VAH",
    "NY_W04": "NY|NY|PRIOR|POC",
}
LIVE_TO_TAPE = {"EUROPE|EUROPE|CURRENT|HIGH": "EU_CURRENT_HIGH_SWEEP",
                "EUROPE|EUROPE|PRIOR|HIGH": "EU_PRIOR_HIGH",
                "EUROPE|EUROPE|PRIOR|VAH": "PRIOR_VAH", "NY|NY|PRIOR|POC": "NY_W04"}
COMPACT_DTYPE = np.dtype([
    ("ts", "<i8"), ("mid", "<f8"), ("bid5", "<f8"), ("ask5", "<f8"),
    ("bid10", "<f8"), ("ask10", "<f8"), ("action", "i1"), ("side", "i1"),
    ("size", "<i4"), ("mlofi", "<f8"), ("denom", "<f8"),
])
ACTION_CODE = {b"T": 1, b"A": 2, b"C": 2, b"M": 2}
SIDE_CODE = {b"B": 1, b"A": -1}
INV5 = np.asarray([1.0, .5, 1 / 3, .25, .2, 0, 0, 0, 0, 0], dtype=np.float64)


class StudyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    os.replace(tmp, path)


def _load_live_configs() -> tuple[dict[str, Any], str]:
    payload = json.loads(LIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("strategy_manifest_sha256") != EXPECTED_STRATEGY_MANIFEST_SHA:
        raise StudyError("frozen production strategy manifest SHA mismatch")
    if payload.get("runtime_contract_sha256") != EXPECTED_RUNTIME_CONTRACT_SHA:
        raise StudyError("frozen runtime contract SHA mismatch")
    result = {}
    for row in payload.get("strategies", []):
        derived = row["derived_parameters"]
        live_key = derived["runtime_family_key"]["value"]
        canonical = FAMILY_MAP.get(live_key)
        if not canonical:
            continue
        if derived["runtime_structural_level"]["value"] not in {
            "CURRENT_EUROPE_HIGH_SWEEP", "PRIOR_EUROPE_HIGH", "PRIOR_EUROPE_VAH", "PRIOR_RTH_POC"
        }:
            raise StudyError(f"unexpected structural mapping for {live_key}")
        weights_raw = row["class_a_parameters"]["feature_weights"]
        weights = {f"{name}_weight": float(weights_raw[name]["value"]) for name in
                   ("aggression", "restoration", "price_resistance", "persistence", "multi_level_support")}
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise StudyError(f"frozen weights do not normalize for {live_key}")
        config = L2Config(**weights, min_quality_score=float(row["class_a_parameters"]["quality_threshold"]["value"]))
        result[canonical] = {"live_key": live_key, "runtime_strategy_identity": derived["runtime_strategy_identity"]["value"],
                             "structural_level": derived["runtime_structural_level"]["value"],
                             "config": _default_parameters(config), "strategy_manifest_sha256": payload["strategy_manifest_sha256"],
                             "runtime_contract_sha256": payload["runtime_contract_sha256"]}
    if set(result) != set(LIVE_TO_TAPE):
        raise StudyError(f"incomplete frozen strategy handoff: {sorted(result)}")
    return result, _sha(LIVE_CONFIG_PATH)


def _sources_and_tapes() -> tuple[dict[str, dict[str, Path]], dict[str, Path], dict[str, str]]:
    manifest, requests = baseline._manifest(DATA_ROOT)
    date_rows: dict[str, dict[str, Path]] = {}
    source_hashes: dict[str, str] = {}
    for period, start, end in (("APRIL_2025", "2025-04-01", "2025-04-30"),
                               ("OCTOBER_2025", "2025-10-01", "2025-10-31")):
        dates = sorted({str(r.get("session_date")) for r in requests.values()
                        if isinstance(r, dict) and r.get("session_date") and start <= str(r["session_date"]) <= end
                        and r.get("category") in {"TRAIN", "VALIDATION"}})
        expected = [str(x) for x in (baseline.TRAIN_DATES if period == "APRIL_2025" else []) if start <= x <= end]
        if period == "OCTOBER_2025":
            expected = sorted({str(r.get("session_date")) for r in requests.values()
                               if isinstance(r, dict) and r.get("category") == "VALIDATION"
                               and start <= str(r.get("session_date", "")) <= end})
        if dates != expected:
            raise StudyError(f"{period} source manifest dates differ from expected; found={dates}, expected={expected}")
        for day in dates:
            row = next((r for r in requests.values() if isinstance(r, dict) and r.get("session_date") == day
                        and r.get("category") in {"TRAIN", "VALIDATION"}), None)
            if not row:
                raise StudyError(f"missing source manifest row: {day}")
            path = DATA_ROOT / row["path"]
            if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
                raise StudyError(f"native source missing/size mismatch: {path}")
            digest = _sha(path)
            if digest != row.get("sha256"):
                raise StudyError(f"native source hash mismatch: {path}")
            if row.get("schema") != "mbp-10" or row.get("symbol") not in {"ESM5", "ESZ5"}:
                raise StudyError(f"unexpected native identity for {day}: {row.get('schema')} {row.get('symbol')}")
            from databento import DBNStore
            metadata = DBNStore.from_file(path).metadata
            expected_contract = "ESM5" if period == "APRIL_2025" else "ESZ5"
            if metadata.dataset != "GLBX.MDP3" or metadata.schema != "mbp-10" or expected_contract not in metadata.symbols:
                raise StudyError(f"DBN header identity mismatch for {day}: {metadata}")
            date_rows.setdefault(day, {})["period"] = Path(period)
            date_rows[day]["source"] = path
            date_rows[day]["source_sha256"] = Path(digest)
            source_hashes[day] = digest
    tapes: dict[str, Path] = {}
    for day in date_rows:
        period = date_rows[day]["period"].name
        path = (TRAIN_TAPE_ROOT if period == "APRIL_2025" else OCT_TAPE_ROOT) / f"{day}-candidate-tape.npz"
        if not path.is_file():
            raise StudyError(f"candidate tape absent for covered date {day}")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            tape_sha = str(meta.get("source_sha256", ""))
            semantic = str(meta.get("semantic_sha256", ""))
            if tape_sha != source_hashes[day] or semantic != EXPECTED_TAPE_SEMANTIC_SHA:
                raise StudyError(f"candidate tape/source/semantic mismatch for {day}")
        tapes[day] = path
    return date_rows, tapes, source_hashes


def _source_count(path: Path) -> int:
    _, reqs = baseline._manifest(DATA_ROOT)
    resolved = path.resolve()
    for r in reqs.values():
        if isinstance(r, dict) and (DATA_ROOT / str(r.get("path", ""))).resolve() == resolved:
            n = r.get("verification", {}).get("record_count")
            if isinstance(n, int) and n > 0:
                return n
    raise StudyError(f"native file lacks a sealed record count: {path}")


def _iter_chunks(path: Path, rows: int = 250_000) -> Iterable[np.ndarray]:
    from databento import DBNStore
    yield from DBNStore.from_file(path).to_ndarray(count=rows)


def _extract_compact(day: str, source: Path, cache_path: Path, source_sha: str) -> tuple[np.memmap, Path, dict[str, Any]]:
    """One bounded-memory source pass into an ephemeral compact structured mmap."""
    count = _source_count(source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(prefix=f"absnorm-{day}-", suffix=".bin", dir=cache_path.parent)
    os.close(fd)
    out = np.memmap(scratch, mode="w+", dtype=COMPACT_DTYPE, shape=(count,))
    windows = baseline._session_windows(day)
    offset = 0
    raw_rows = 0
    last = None
    total = {"raw_rows": 0, "scoped_rows": 0, "start_ns": None, "end_ns": None}
    for batch in _iter_chunks(source):
        n = len(batch); raw_rows += n
        ts = np.asarray(batch["ts_recv"], dtype=np.int64)
        bid_px, bid_sz = mlofi._raw_book_arrays(batch, "bid")
        ask_px, ask_sz = mlofi._raw_book_arrays(batch, "ask")
        bid_vis = mlofi._packed_book_or_raise(bid_px, bid_sz, "bid")
        ask_vis = mlofi._packed_book_or_raise(ask_px, ask_sz, "ask")
        valid = bid_vis[:, 0] & ask_vis[:, 0] & (ask_px[:, 0] > bid_px[:, 0])
        sess = mlofi._session_codes(ts, windows)
        scope = valid & (sess >= 0)
        if last is None:
            pbpx = np.zeros_like(bid_px); pbsz = np.zeros_like(bid_sz)
            papx = np.zeros_like(ask_px); pasz = np.zeros_like(ask_sz)
            pbpx[1:], pbsz[1:] = bid_px[:-1], bid_sz[:-1]
            papx[1:], pasz[1:] = ask_px[:-1], ask_sz[:-1]
            pvalid = np.zeros(n, dtype=bool); pvalid[1:] = valid[:-1]
            psess = np.full(n, -1, dtype=np.int8); psess[1:] = sess[:-1]
        else:
            pbpx = np.empty_like(bid_px); pbsz = np.empty_like(bid_sz)
            papx = np.empty_like(ask_px); pasz = np.empty_like(ask_sz)
            pbpx[0], pbsz[0], papx[0], pasz[0], last_valid, last_sess = last
            pbpx[1:], pbsz[1:] = bid_px[:-1], bid_sz[:-1]
            papx[1:], pasz[1:] = ask_px[:-1], ask_sz[:-1]
            pvalid = np.empty(n, dtype=bool); pvalid[0] = last_valid; pvalid[1:] = valid[:-1]
            psess = np.empty(n, dtype=np.int8); psess[0] = last_sess; psess[1:] = sess[:-1]
        pvalid &= psess == sess
        contribution = mlofi._vector_price_keyed_contribution(
            bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
            previous_bid_px=pbpx, previous_bid_sz=pbsz, previous_ask_px=papx,
            previous_ask_sz=pasz, previous_valid=pvalid, action=np.asarray(batch["action"]),
            side=np.asarray(batch["side"]), price=np.asarray(batch["price"], dtype=np.int64),
            size=np.asarray(batch["size"], dtype=np.int64), current_valid=scope)
        selected = np.flatnonzero(scope)
        end = offset + len(selected)
        if end > count:
            raise StudyError(f"native scoped count exceeds sealed count: {day}")
        rows = np.empty(len(selected), dtype=COMPACT_DTYPE)
        rows["ts"] = ts[selected]
        rows["mid"] = (bid_px[selected, 0] + ask_px[selected, 0]) / 2e9
        rows["bid5"] = bid_sz[selected, :5].sum(axis=1)
        rows["ask5"] = ask_sz[selected, :5].sum(axis=1)
        rows["bid10"] = bid_sz[selected].sum(axis=1)
        rows["ask10"] = ask_sz[selected].sum(axis=1)
        acts = np.asarray(batch["action"])[selected]
        sides = np.asarray(batch["side"])[selected]
        rows["action"] = np.select([acts == b"T", np.isin(acts, (b"A", b"C", b"M"))], [1, 2], default=0)
        rows["side"] = np.select([sides == b"B", sides == b"A"], [1, -1], default=0)
        rows["size"] = np.asarray(batch["size"], dtype=np.int32)[selected]
        rows["mlofi"] = contribution[selected].astype(np.float64) @ INV5
        w = INV5[:5]
        level_sum = (bid_sz[selected, :5] + ask_sz[selected, :5]).astype(np.float64)
        active = level_sum > 0
        den_num = ((level_sum / 2.0) * w).sum(axis=1)
        den = (active * w).sum(axis=1)
        rows["denom"] = np.divide(den_num, den, out=np.zeros_like(den_num), where=den > 0)
        out[offset:end] = rows
        offset = end
        if len(selected):
            total["start_ns"] = int(ts[selected[0]]) if total["start_ns"] is None else total["start_ns"]
            total["end_ns"] = int(ts[selected[-1]])
        last = (bid_px[-1].copy(), bid_sz[-1].copy(), ask_px[-1].copy(), ask_sz[-1].copy(), bool(valid[-1]), int(sess[-1]))
    out.flush()
    if raw_rows != count:
        del out
        Path(scratch).unlink(missing_ok=True)
        raise StudyError(f"raw count mismatch for {day}: {raw_rows} != {count}")
    total["raw_rows"] = raw_rows; total["scoped_rows"] = offset; total["source_sha256"] = source_sha
    if offset != count:
        # Re-open the temporary file with its actual written row count.
        del out
        out = np.memmap(scratch, mode="r", dtype=COMPACT_DTYPE, shape=(count,))
        return out[:offset], Path(scratch), total
    return out, Path(scratch), total


def _prefix_window(prefix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = prefix[hi].astype(np.float64, copy=True)
    mask = lo > 0
    result[mask] -= prefix[lo[mask] - 1]
    return result


def _rolling_features(rows: np.ndarray, query_ns: np.ndarray, direction: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized as-of context; every selected row has ts < query timestamp."""
    # A structured mmap timestamp field is strided. Copy once to a contiguous
    # vector so repeated as-of searches do not trigger random disk-page churn.
    ts = np.array(rows["ts"], dtype=np.int64, copy=True)
    ix = np.searchsorted(ts, query_ns, side="left") - 1
    valid = ix >= 1
    ix_safe = np.maximum(ix, 0)
    prevmid = np.asarray(rows["mid"], dtype=np.float64)
    diff_ticks = np.diff(prevmid, prepend=prevmid[0]) / TICK
    sq_prefix = np.cumsum(diff_ticks * diff_ticks)
    abs_prefix = np.cumsum(np.abs(diff_ticks))
    direction = np.asarray(direction, dtype=np.float64)
    result: dict[str, np.ndarray] = {name: np.full(len(query_ns), np.nan, dtype=np.float64) for name in FEATURES}
    result["_index"] = ix
    # Cumulative discrete event fields.
    is_trade = rows["action"] == 1
    is_update = rows["action"] == 2
    size = np.where(is_trade, rows["size"], 0).astype(np.int64)
    buyvol = np.where(is_trade & (rows["side"] == 1), size, 0)
    sellvol = np.where(is_trade & (rows["side"] == -1), size, 0)
    trade_prefix = np.cumsum(is_trade, dtype=np.int64)
    volume_prefix = np.cumsum(size, dtype=np.int64)
    buy_prefix = np.cumsum(buyvol, dtype=np.int64)
    sell_prefix = np.cumsum(sellvol, dtype=np.int64)
    update_prefix = np.cumsum(is_update, dtype=np.int64)
    mlofi_prefix = np.cumsum(np.asarray(rows["mlofi"], dtype=np.float64))
    windows = {"30": 30_000_000_000, "120": 120_000_000_000,
               "1": 1_000_000_000, "2": 2_000_000_000,
               "5": 5_000_000_000, "10": 10_000_000_000}
    for key, delta_ns in windows.items():
        lo = np.searchsorted(ts, query_ns - delta_ns, side="left")
        hi = ix_safe
        if key in {"30", "120"}:
            rv = np.sqrt(np.maximum(0.0, _prefix_window(sq_prefix, lo, hi)))
            result[f"RV_{key}S_RAW"] = np.where(valid, rv, np.nan)
        if key in {"1", "2"}:
            pbuy = _prefix_window(buy_prefix, lo, hi); psell = _prefix_window(sell_prefix, lo, hi)
            contra = np.where(direction >= 0, psell, pbuy)
            result[f"AGGRESSIVE_VOLUME_{key}S_RAW"] = np.where(valid, contra, np.nan)
            depth = np.where(direction >= 0, rows["bid5"][ix_safe], rows["ask5"][ix_safe]) / 5.0
            result[f"AGGRESSION_TO_DEPTH_{key}S"] = np.divide(contra, depth, out=np.full(len(query_ns), np.nan), where=depth > 0)
        if key in {"5", "30"}:
            seconds = delta_ns / 1e9
            trades = _prefix_window(trade_prefix, lo, hi)
            contracts = _prefix_window(volume_prefix, lo, hi)
            updates = _prefix_window(update_prefix, lo, hi)
            result[f"TRADES_PER_SECOND_{key}S"] = np.where(valid, trades / seconds, np.nan)
            result[f"CONTRACTS_PER_SECOND_{key}S"] = np.where(valid, contracts / seconds, np.nan)
            result[f"BOOK_UPDATES_PER_SECOND_{key}S"] = np.where(valid, updates / seconds, np.nan)
        if key in {"2", "5", "10", "30"}:
            current = prevmid[ix_safe]
            start_ix = np.maximum(lo, 0)
            move = (current - prevmid[start_ix]) / TICK
            velocity = move / (delta_ns / 1e9)
            result[f"VELOCITY_{key}S_TICKS_PER_SECOND"] = np.where(valid, velocity, np.nan)
            result[f"VELOCITY_{key}S_DIRECTIONAL"] = np.where(valid, velocity * direction, np.nan)
        if key in {"5", "30"}:
            current = prevmid[ix_safe]
            start_ix = np.maximum(lo, 0)
            net = np.abs(current - prevmid[start_ix]) / TICK
            path = _prefix_window(abs_prefix, lo, hi)
            er = np.divide(net, path, out=np.zeros(len(query_ns)), where=path > 0)
            result[f"ER_{key}S"] = np.where(valid, er, np.nan)
            result[f"ER_{key}S_DIRECTIONAL"] = np.where(valid, np.sign(current - prevmid[start_ix]) * direction * er, np.nan)
        if key in {"1", "2", "5"}:
            result[f"MLOFI_{key}S_RAW"] = np.where(valid, _prefix_window(mlofi_prefix, lo, hi), np.nan)
            denom = np.asarray(rows["denom"])[ix_safe]
            result[f"MLOFI_{key}S_DEPTH_NORMALIZED"] = np.divide(
                result[f"MLOFI_{key}S_RAW"], denom, out=np.full(len(query_ns), np.nan), where=denom > 0)
    bid5 = np.asarray(rows["bid5"])[ix_safe]; ask5 = np.asarray(rows["ask5"])[ix_safe]
    bid10 = np.asarray(rows["bid10"])[ix_safe]; ask10 = np.asarray(rows["ask10"])[ix_safe]
    result["TOP5_SAME_SIDE_DEPTH_RAW"] = np.where(direction >= 0, bid5, ask5)
    result["TOP10_SAME_SIDE_DEPTH_RAW"] = np.where(direction >= 0, bid10, ask10)
    result["OPPOSITE_TOP5_DEPTH_RAW"] = np.where(direction >= 0, ask5, bid5)
    asym = np.divide(bid5 - ask5, bid5 + ask5, out=np.zeros(len(query_ns)), where=(bid5 + ask5) > 0)
    result["DEPTH_ASYMMETRY_RAW"] = asym
    result["DEPTH_ASYMMETRY_DIRECTIONAL"] = asym * direction
    # Exact MLOFI accounting is the established price-keyed TOP5/INVERSE_LEVEL
    # variant; persistence uses eight preceding 250-ms contribution bins.
    persistence = np.full(len(query_ns), np.nan)
    good_query = ix >= 0
    if good_query.any():
        # Vectorize all 8 x 250-ms bin boundaries; no Python-level repeated
        # binary search over a multi-million-row day.
        q = query_ns[good_query]
        offsets = np.arange(8, -1, -1, dtype=np.int64) * 250_000_000
        edge_ix = np.searchsorted(ts, (q[:, None] - offsets[None, :]).ravel(), side="left").reshape(len(q), 9)
        cumulative_at_edge = np.zeros(edge_ix.shape, dtype=np.float64)
        has_prior = edge_ix > 0
        cumulative_at_edge[has_prior] = mlofi_prefix[edge_ix[has_prior] - 1]
        bins = np.diff(cumulative_at_edge, axis=1)
        signs = np.sign(bins)
        nonzero = signs != 0
        last_from_right = np.argmax(nonzero[:, ::-1], axis=1)
        has_nonzero = nonzero.any(axis=1)
        last_col = 7 - last_from_right
        last_sign = signs[np.arange(len(q)), last_col]
        matches = (signs == last_sign[:, None]) & nonzero
        persistence[good_query] = np.divide(matches.sum(axis=1), nonzero.sum(axis=1),
                                            out=np.zeros(len(q), dtype=np.float64), where=has_nonzero)
    result["MLOFI_PERSISTENCE_2S"] = persistence
    result["MLOFI_PERSISTENCE_DIRECTIONAL"] = persistence * np.sign(result["MLOFI_2S_DEPTH_NORMALIZED"]) * direction
    result["PRICE_IMPACT_PER_MLOFI_5S"] = np.divide(
        result["VELOCITY_5S_DIRECTIONAL"], np.abs(result["MLOFI_5S_DEPTH_NORMALIZED"]),
        out=np.full(len(query_ns), np.nan), where=np.abs(result["MLOFI_5S_DEPTH_NORMALIZED"]) > 1e-12)
    return result


def _resiliency(rows: np.ndarray, query_ns: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed 100-ms top-five aggregate-depth depletion/recovery episode proxy.

    Each sample's depletion is the positive fall from the prior 50-ms sample;
    recovery is the capped subsequent increase within the stated horizon. The
    episode is usable only once its full horizon is strictly before query time.
    """
    ts = np.array(rows["ts"], dtype=np.int64, copy=True)
    grid = np.arange(int(ts[0]), int(ts[-1]) + RESILIENCY_SAMPLE_NS, RESILIENCY_SAMPLE_NS, dtype=np.int64)
    ids = np.searchsorted(ts, grid, side="right") - 1
    ok = ids >= 0; grid = grid[ok]; ids = ids[ok]
    depth = (np.asarray(rows["bid5"])[ids] + np.asarray(rows["ask5"])[ids]).astype(np.float64)
    drop = np.maximum(0.0, np.r_[0.0, depth[:-1]] - depth)
    fields = {}
    for horizon_ms, label in ((250, "250MS"), (500, "500MS"), (1000, "1S"), (2000, "2S")):
        steps = horizon_ms // 50
        recovery = np.full(len(depth), np.nan)
        if len(depth) > steps:
            future_max = np.maximum.reduce([depth[i:len(depth)-steps+i] for i in range(steps + 1)])
            denom = drop[:len(future_max)]
            recovery[:len(future_max)] = np.divide(np.minimum(denom, np.maximum(0.0, future_max - depth[:len(future_max)])),
                                                   denom, out=np.zeros_like(denom), where=denom > 0)
        fields[label] = recovery
    med = np.full(len(query_ns), np.nan)
    horizon_values = {k: np.full(len(query_ns), np.nan) for k in fields}
    for j, q in enumerate(query_ns):
        upper = np.searchsorted(grid, q - 500_000_000, side="left")
        lower = np.searchsorted(grid, q - 60_000_000_000, side="left")
        vals = fields["500MS"][lower:upper]
        drop_window = drop[lower:upper]
        vals = vals[np.isfinite(vals) & (drop_window > 0)]
        if vals.size: med[j] = float(np.median(vals))
        for k, ar in fields.items():
            idx = np.searchsorted(grid, q - (int(k[:-2]) * 1_000_000 if k.endswith("MS") else int(k[:-1]) * 1_000_000_000), side="left") - 1
            if idx >= 0 and drop[idx] > 0 and np.isfinite(ar[idx]): horizon_values[k][j] = float(ar[idx])
    return {"RESILIENCY_RECOVERY_250MS": horizon_values["250MS"],
            "RESILIENCY_RECOVERY_500MS": horizon_values["500MS"],
            "RESILIENCY_RECOVERY_1S": horizon_values["1S"], "RESILIENCY_RECOVERY_2S": horizon_values["2S"],
            "RESILIENCY_MEDIAN_500MS_PRIOR_60S": med}


def _period(day: str) -> str:
    return "APRIL_2025" if day.startswith("2025-04-") else "OCTOBER_2025"


def _time_queries(day: str) -> np.ndarray:
    windows = baseline._session_windows(day)
    out = []
    for start, end in windows.values():
        out.extend(range(start, end, SAMPLE_STEP_NS))
    return np.asarray(out, dtype=np.int64)


def _path_outcomes(candidate: Mapping[str, Any], events: np.ndarray, session_windows: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    t = int(candidate["interaction_start_ns"])
    direction = 1 if candidate.get("direction") == "BUYER_ABSORPTION" else -1
    session = str(candidate.get("session", candidate.get("trading_session", "")))
    start, end = session_windows.get(session, (0, 0))
    et = np.asarray(events["timestamp_ns"], dtype=np.int64)
    start_ix = int(np.searchsorted(et, t, side="left"))
    if start_ix >= len(events): return {"markouts_ticks": {}, "mfe_mae_ticks": {}, "barriers": {}}
    # Midpoint at the last public state strictly before interaction_start.
    anchor_ix = max(0, start_ix - 1)
    anchor = (float(events["bid"][anchor_ix]) + float(events["ask"][anchor_ix])) / 2
    horizon_30s = t + 30_000_000_000
    upper = int(np.searchsorted(et, min(end, horizon_30s), side="right"))
    # Fixed-horizon markouts use the first quote at or after the horizon. The
    # path therefore needs one observation beyond the inclusive MFE/barrier
    # window; otherwise a non-exact 30s timestamp is always reported missing.
    if horizon_30s < end:
        upper = min(len(et), max(upper, int(np.searchsorted(et, horizon_30s, side="left")) + 1))
    path = events[start_ix:upper]
    path_t = et[start_ix:upper]
    mids = (path["bid"].astype(np.float64) + path["ask"].astype(np.float64)) / 2
    markouts = {}
    mfe_mae = {}
    for ms in HORIZONS_MS:
        i = int(np.searchsorted(path_t, t + ms * 1_000_000, side="left"))
        markouts[str(ms)] = None if i >= len(path_t) else float((mids[i] - anchor) / TICK * direction)
    for ms in PATH_HORIZONS_MS:
        take = path_t <= t + ms * 1_000_000
        signed = (mids[take] - anchor) / TICK * direction
        mfe_mae[str(ms)] = {"mfe": float(np.max(np.r_[0.0, signed])), "mae": float(np.max(np.r_[0.0, -signed]))} if signed.size else None
    barriers = {}
    for up, down in BARRIERS:
        key = f"+{up}/-{down}"
        end_ix = int(np.searchsorted(path_t, t + 30_000_000_000, side="right"))
        signed = (mids[:end_ix] - anchor) / TICK * direction
        hit = None
        for value in signed:
            pos, neg = value >= up, value <= -down
            if pos or neg:
                hit = "TIE" if pos and neg else "UP" if pos else "DOWN"
                break
        barriers[key] = hit or "NO_TOUCH"
    return {"markouts_ticks": markouts, "mfe_mae_ticks": mfe_mae, "barriers": barriers,
            "anchor_mid": anchor, "markout_path_source": "candidate tape ordered MBP-10 public path"}


def _candidate_rows(tape_path: Path, source_sha: str, day: str, configs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    from .mac_2025_candidate_tape import _json_default
    with np.load(tape_path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        raw_rows = json.loads(str(z["candidate_json"].item()))
        events = np.asarray(z["events"])
    if meta.get("date") != day or meta.get("source_sha256") != source_sha:
        raise StudyError(f"candidate tape source identity mismatch: {day}")
    if meta.get("semantic_sha256") != EXPECTED_TAPE_SEMANTIC_SHA:
        raise StudyError(f"candidate detector semantic mismatch: {day}")
    windows = baseline._session_windows(day)
    frozen = []
    for row in raw_rows:
        canonical = str(row.get("family_id", ""))
        live_key = LIVE_TO_TAPE.get(canonical)
        if live_key is None: continue
        cfg = configs[canonical]["config"]
        quality = _quality(row, cfg["weights"], cfg)
        if not _qualifies(row, cfg): continue
        start_ns = int(row.get("interaction_start_ns", 0))
        end_ns = int(row.get("interaction_end_ns", 0))
        if start_ns <= 0 or end_ns < start_ns:
            raise StudyError(f"malformed frozen event timestamps: {day} {row.get('interaction_id')}")
        item = {"event_id": f"{day}:{row['interaction_id']}", "date": day, "period": _period(day),
                "family": canonical, "live_family": live_key, "session": row.get("trading_session"),
                "direction": row.get("direction"), "interaction_start_ns": start_ns,
                "interaction_end_ns": end_ns, "level": row.get("reference_level"),
                "price": row.get("level_price", row.get("zone_low")), "core_quality_score": quality,
                "interaction_id": row.get("interaction_id"), "reference_session": row.get("reference_session"),
                "reference_day": row.get("reference_day"), "candidate_definition": "frozen completed causal interaction; live config _qualifies gate"}
        frozen.append(item)
    frozen.sort(key=lambda x: (int(x["interaction_start_ns"]), x["family"], x["event_id"]))
    return frozen, events, meta


def _add_percentiles(events: list[dict[str, Any]], samples: list[dict[str, Any]], history_global: dict[str, list[float]],
                     history_tod: dict[str, dict[str, list[float]]], day: str) -> None:
    windows = baseline._session_windows(day)
    source_features = set(PERCENTILE_SOURCE)
    global_sorted = {f: np.sort(np.asarray(history_global.get(f, []), dtype=np.float64)) for f in source_features}
    tod_sorted: dict[str, dict[str, np.ndarray]] = {}
    for feature in source_features:
        tod_sorted[feature] = {key: np.sort(np.asarray(values, dtype=np.float64))
                               for key, values in history_tod.get(feature, {}).items()}
    for row in events:
        ts = int(row["interaction_start_ns"])
        session = str(row.get("session", ""))
        s0 = windows.get(session, (0, 0))[0]
        key = f"{session}:{(ts - s0) // TOD_BIN_NS}"
        for feature in source_features:
            val = row.get(feature)
            if val is None or not math.isfinite(float(val)): continue
            for pctname in PERCENTILE_SOURCE[feature]:
                src = abs(float(val)) if "VELOCITY" in feature or "MLOFI" in feature else float(val)
                use_tod = pctname.endswith("TOD_PERCENTILE")
                dist = tod_sorted.get(feature, {}).get(key, np.empty(0)) if use_tod else global_sorted.get(feature, np.empty(0))
                row[pctname] = _rank_sorted(src, dist)
    # Seed all historical distributions only after every event of this date is ranked.
    for sample in samples:
        ts = int(sample["timestamp_ns"]); session = str(sample["session"])
        s0 = windows.get(session, (0, 0))[0]
        key = f"{session}:{(ts - s0) // TOD_BIN_NS}"
        for feature in source_features:
            value = sample.get(feature)
            if value is None or not math.isfinite(float(value)): continue
            history_global.setdefault(feature, []).append(float(value))
            history_tod.setdefault(feature, {}).setdefault(key, []).append(float(value))
    # Resiliency has a prior-date-only global percentile reference.
    feature = "RESILIENCY_MEDIAN_500MS_PRIOR_60S"
    hist_key = "_event_resiliency"
    for row in events:
        value = row.get(feature)
        row["RESILIENCY_PERCENTILE"] = _rank(float(value), history_global.get(hist_key, [])) if value is not None else None
    history_global.setdefault(hist_key, []).extend(float(s[feature]) for s in samples if s.get(feature) is not None and math.isfinite(float(s[feature])))


def _rank(value: float, distribution: Iterable[float]) -> float | None:
    arr = np.asarray(list(distribution), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size: return None
    return float(np.searchsorted(np.sort(arr), value, side="right") / arr.size)


def _rank_sorted(value: float, distribution: np.ndarray) -> float | None:
    if distribution.size == 0:
        return None
    return float(np.searchsorted(distribution, value, side="right") / distribution.size)


def _date_checkpoint_path(day: str) -> Path:
    return OUT_ROOT / "checkpoints" / f"{day}.json.gz"


def _extract_date(day: str, source: Path, tape_path: Path, source_sha: str,
                  configs: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    events, path_events, tape_meta = _candidate_rows(tape_path, source_sha, day, configs)
    count = _source_count(source)
    work_dir = OUT_ROOT / "checkpoints" / "_work"
    compact_path = work_dir / f"{day}.compact"
    rows, temp, coverage = _extract_compact(day, source, compact_path, source_sha)
    try:
        event_queries = np.asarray([int(e["interaction_start_ns"]) for e in events], dtype=np.int64)
        event_dir = np.asarray([1 if e["direction"] == "BUYER_ABSORPTION" else -1 for e in events], dtype=np.float64)
        base_sample_queries = _time_queries(day)
        # Both passive sides are represented in historical TOD references.
        sample_queries = np.repeat(base_sample_queries, 2)
        sample_dirs = np.tile(np.asarray([1.0, -1.0]), len(base_sample_queries))
        f_ev = _rolling_features(rows, event_queries, event_dir) if len(events) else {k: np.array([], dtype=float) for k in FEATURES}
        f_ev.update(_resiliency(rows, event_queries) if len(events) else {})
        f_sample = _rolling_features(rows, sample_queries, sample_dirs)
        f_sample.update(_resiliency(rows, sample_queries))
        for i, event in enumerate(events):
            for name in FEATURES:
                value = f_ev.get(name, np.full(len(events), np.nan))[i]
                event[name] = float(value) if np.isfinite(value) else None
            event.update(_path_outcomes(event, path_events, baseline._session_windows(day)))
        samples = []
        windows = baseline._session_windows(day)
        sess_codes = ((0, "ASIA"), (1, "EUROPE"), (2, "NY"))
        for j, q in enumerate(sample_queries):
            session = next((name for code, name in sess_codes if windows[name][0] <= q < windows[name][1]), None)
            if not session: continue
            item = {"timestamp_ns": int(q), "session": session}
            for name in FEATURES:
                value = f_sample.get(name, np.full(len(sample_queries), np.nan))[j]
                item[name] = float(value) if np.isfinite(value) else None
            samples.append(item)
        return {"date": day, "period": _period(day), "events": events, "samples": samples,
                "source": str(source), "source_sha256": source_sha, "tape": str(tape_path),
                "tape_sha256": _sha(tape_path), "tape_version": tape_meta.get("tape_version"),
                "path_outcome_version": PATH_OUTCOME_VERSION,
                "candidate_count_all": int(tape_meta.get("candidate_count", -1)),
                "frozen_event_count": len(events), "coverage": coverage}, time.perf_counter() - start
    finally:
        del rows
        temp.unlink(missing_ok=True)


def _valid_checkpoint(day: str, source_sha: str, tape_path: Path, config_sha: str) -> dict[str, Any] | None:
    p = _date_checkpoint_path(day)
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f: payload = json.load(f)
        if payload.get("status") == "COMPLETE" and payload.get("source_sha256") == source_sha \
                and payload.get("tape_sha256") == _sha(tape_path) and payload.get("config_sha256") == config_sha:
            return payload
    except (OSError, json.JSONDecodeError): pass
    return None


def _refresh_checkpoint_path_outcomes(payload: dict[str, Any], tape_path: Path, day: str) -> bool:
    """Repair derived path outcomes in a valid feature checkpoint without rereading DBN."""
    if payload.get("path_outcome_version") == PATH_OUTCOME_VERSION:
        return False
    try:
        with np.load(tape_path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"].item()))
            path_events = np.asarray(z["events"])
    except Exception as exc:
        raise StudyError(f"cannot refresh checkpoint path outcomes from {tape_path}: {exc}") from exc
    if metadata.get("date") != day or metadata.get("source_sha256") != payload.get("source_sha256") \
            or metadata.get("semantic_sha256") != EXPECTED_TAPE_SEMANTIC_SHA:
        raise StudyError(f"candidate tape identity mismatch while refreshing path outcomes: {day}")
    windows = baseline._session_windows(day)
    for event in payload.get("events", []):
        event.update(_path_outcomes(event, path_events, windows))
    payload["path_outcome_version"] = PATH_OUTCOME_VERSION
    return True


def _summarize_values(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if not len(x): return {"n": 0}
    trim = np.sort(x); trim = trim[int(.1 * len(trim)):max(int(.9 * len(trim)), int(.1 * len(trim)) + 1)]
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "trimmed_mean_10pct": float(np.mean(trim)), "p25": float(np.quantile(x, .25)),
            "p75": float(np.quantile(x, .75)), "se": sd / math.sqrt(len(x)),
            "ci95": [float(np.mean(x) - 1.96 * sd / math.sqrt(len(x))), float(np.mean(x) + 1.96 * sd / math.sqrt(len(x)))],
            "positive_fraction": float(np.mean(x > 0)), "negative_fraction": float(np.mean(x < 0)),
            "zero_fraction": float(np.mean(x == 0))}


def _bucket_index(values: np.ndarray, bins: int = 5) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.full(len(values), -1, dtype=np.int8)
    if finite.any():
        edges = np.quantile(values[finite], np.linspace(0, 1, bins + 1)[1:-1])
        output[finite] = np.searchsorted(edges, values[finite], side="right")
    return output


def _markout_table(events: list[dict[str, Any]], feature_name: str, relative: bool = False,
                   bins: int = 5) -> dict[str, Any]:
    feature = (PERCENTILE_SOURCE.get(feature_name, (feature_name,))[0] if relative else feature_name)
    data = [e for e in events if e.get(feature) is not None]
    if not data: return {"feature": feature_name, "n": 0, "buckets": {}}
    vals = np.asarray([float(e[feature]) for e in data])
    labels = _bucket_index(vals, bins)
    results = {}
    for b in range(bins):
        selected = [e for e, bi in zip(data, labels) if bi == b]
        by_horizon = {}
        for horizon in HORIZONS_MS:
            by_horizon[str(horizon)] = _summarize_values(e["markouts_ticks"].get(str(horizon)) for e in selected)
        results[f"Q{b+1}"] = {"event_count": len(selected), "active_dates": len({e["date"] for e in selected}),
                               "markouts": by_horizon}
    return {"feature": feature_name, "bucket_variable": feature, "n": len(data), "buckets": results}


def _daily_stability(events: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    # Date-local quintiles are descriptive neighbor-stability views, never fitted for selection.
    result = {}
    for day in sorted({e["date"] for e in events}):
        rows = [e for e in events if e["date"] == day and e.get(feature) is not None and e["markouts_ticks"].get("5000") is not None]
        if len(rows) < 10:
            result[day] = {"event_count": len(rows), "status": "INSUFFICIENT"}; continue
        v = np.asarray([float(e[feature]) for e in rows]); y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows])
        b = _bucket_index(v)
        hi, lo = y[b == 4], y[b == 0]
        buckets = {}
        for bi in range(5):
            selected = [e for e, label in zip(rows, b) if label == bi]
            buckets[f"Q{bi+1}"] = {"event_count": len(selected),
                "markouts": {str(h): _summarize_values(e["markouts_ticks"].get(str(h)) for e in selected)
                             for h in (2000, 5000, 10000, 30000)},
                "mfe_mae_5s": {k: _summarize_values(e.get("mfe_mae_ticks", {}).get("5000", {}).get(k) for e in selected)
                               for k in ("mfe", "mae")},
                "barriers_30s": _barrier_summary(selected)}
        result[day] = {"event_count": len(rows), "high_minus_low_5s": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None,
                       "high_n": int(len(hi)), "low_n": int(len(lo)), "positive_dates": None, "quintile_buckets": buckets}
    eff = [r["high_minus_low_5s"] for r in result.values() if r.get("high_minus_low_5s") is not None]
    for row in result.values():
        if row.get("high_minus_low_5s") is not None: row["positive_dates"] = row["high_minus_low_5s"] > 0
    return {"dates": result, "positive_dates": sum(x > 0 for x in eff), "negative_dates": sum(x < 0 for x in eff),
            "insufficient_dates": sum(r.get("status") == "INSUFFICIENT" for r in result.values()),
            "median_daily_effect": float(np.median(eff)) if eff else None,
            "p25_daily_effect": float(np.quantile(eff, .25)) if eff else None,
            "p75_daily_effect": float(np.quantile(eff, .75)) if eff else None}


def _neighbor_stability(events: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    rows = [e for e in events if e.get(feature) is not None and e.get("markouts_ticks", {}).get("5000") is not None]
    if not rows: return {"n": 0}
    values = np.asarray([float(e[feature]) for e in rows], dtype=np.float64)
    outcome = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows], dtype=np.float64)
    q20, q80 = np.quantile(values, [.2, .8])
    broad = np.where(values < q20, 0, np.where(values >= q80, 2, 1))
    output = {}
    for name, labels in (("QUINTILES", _bucket_index(values, 5)), ("TERCILES", _bucket_index(values, 3)),
                         ("BROAD_STATES_P20_P80", broad)):
        count = 5 if name == "QUINTILES" else 3
        groups = {}
        for b in range(count):
            y = outcome[labels == b]
            groups[f"B{b+1}"] = {"n": int(len(y)), "mean_5s_markout": float(y.mean()) if len(y) else None}
        hi, lo = outcome[labels == count - 1], outcome[labels == 0]
        output[name] = {"groups": groups, "high_minus_low_5s": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None}
    effects = [output[x]["high_minus_low_5s"] for x in ("QUINTILES", "TERCILES", "BROAD_STATES_P20_P80")]
    output["fragile_boundary_flag"] = bool(all(v is not None for v in effects) and
                                             len({int(np.sign(v)) for v in effects}) > 1)
    output["n"] = len(rows)
    return output


def _aggregate(all_events: list[dict[str, Any]], all_samples: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [f for f in FEATURES if f in all_events[0]] if all_events else []
    raw_distributions = {}; relative_distributions = {}; raw_vs = {}
    features = {}
    daily_keys = {"RV_30S_RAW", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSION_TO_DEPTH_1S", "TOP5_SAME_SIDE_DEPTH_RAW",
                  "TRADES_PER_SECOND_5S", "VELOCITY_10S_TICKS_PER_SECOND", "ER_5S", "MLOFI_PERSISTENCE_DIRECTIONAL",
                  "RESILIENCY_MEDIAN_500MS_PRIOR_60S"}
    for feature in numeric:
        raw_distributions[feature] = {p: _summarize_values(e.get(feature) for e in all_events if e["period"] == p) for p in ("APRIL_2025", "OCTOBER_2025")}
        relatives = PERCENTILE_SOURCE.get(feature, ())
        for rel in relatives:
            relative_distributions[rel] = {p: _summarize_values(e.get(rel) for e in all_events if e["period"] == p) for p in ("APRIL_2025", "OCTOBER_2025")}
        for rel in relatives:
            rel_table = _markout_table(all_events, feature, relative=True)
            raw_table = _markout_table(all_events, feature, relative=False)
            raw_vs[f"{feature}__{rel}"] = {"raw": raw_table, "relative": rel_table,
                                             "classification": _normalization_class(raw_table, rel_table, all_events, feature, rel)}
        features[feature] = {"raw_quintiles": _markout_table(all_events, feature),
                             "relative_quintiles": _markout_table(all_events, feature, relative=True) if relatives else None,
                             "daily_stability_raw": _daily_stability(all_events, feature) if feature in daily_keys else None,
                             "daily_stability_relative": _daily_stability(all_events, relatives[0]) if relatives and feature in daily_keys else None}
    return {"raw_feature_distributions": raw_distributions, "relative_feature_distributions": relative_distributions,
            "raw_vs_relative": raw_vs, "feature_results": features}


def _normalization_class(raw: Mapping[str, Any], rel: Mapping[str, Any], events: list[dict[str, Any]], raw_feature: str, rel_feature: str) -> str:
    # Conservative descriptive rubric: coherent top-bottom sign across periods/families and more daily agreement.
    def effect(table: Mapping[str, Any], period: str | None = None, fam: str | None = None) -> float | None:
        rows = [e for e in events if e.get(raw_feature if table is raw else rel_feature) is not None
                and (period is None or e["period"] == period) and (fam is None or e["family"] == fam)
                and e["markouts_ticks"].get("5000") is not None]
        if len(rows) < 12: return None
        vals = np.asarray([float(e[raw_feature if table is raw else rel_feature]) for e in rows])
        y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows]); bins = _bucket_index(vals)
        if not np.any(bins == 0) or not np.any(bins == 4): return None
        return float(y[bins == 4].mean() - y[bins == 0].mean())
    raw_e = [effect(raw, p) for p in ("APRIL_2025", "OCTOBER_2025")]
    rel_e = [effect(rel, p) for p in ("APRIL_2025", "OCTOBER_2025")]
    if any(x is None for x in raw_e + rel_e): return "INSUFFICIENT"
    raw_sign = raw_e[0] * raw_e[1] >= 0
    rel_sign = rel_e[0] * rel_e[1] >= 0
    raw_abs = min(abs(x) for x in raw_e); rel_abs = min(abs(x) for x in rel_e)
    if rel_sign and not raw_sign and rel_abs > 0: return "RELATIVE_CLEARLY_BETTER"
    if rel_sign and raw_sign and rel_abs > raw_abs * 1.15: return "RELATIVE_MODESTLY_BETTER"
    if raw_sign and not rel_sign: return "RAW_BETTER"
    if max(abs(x) for x in raw_e + rel_e) < .05: return "NO_RELATIONSHIP"
    return "SIMILAR"


def _feature_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = {}
    for feature, relatives in PERCENTILE_SOURCE.items():
        outputs[feature] = {"raw": _markout_table(events, feature),
                            "relative": {r: _markout_table(events, r) for r in relatives},
                            "classifications": {r: _normalization_class({}, {}, events, feature, r) for r in relatives}}
    # Scale-free variables without an explicit percentile counterpart.
    for feature in ("AGGRESSION_TO_DEPTH_1S", "DEPTH_ASYMMETRY_DIRECTIONAL", "ER_5S_DIRECTIONAL",
                    "MLOFI_5S_DEPTH_NORMALIZED", "RESILIENCY_RECOVERY_500MS"):
        outputs[feature] = {"relative": _markout_table(events, feature), "daily_stability": _daily_stability(events, feature)}
    return outputs


def _quantile_shift(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for feature in FEATURES:
        result[feature] = {}
        for period in ("APRIL_2025", "OCTOBER_2025"):
            vals = np.asarray([e[feature] for e in events if e["period"] == period and e.get(feature) is not None], dtype=np.float64)
            result[feature][period] = {str(q): float(np.quantile(vals, q)) for q in (.10, .25, .5, .75, .90, .95, .99)} if len(vals) else {"n": 0}
        a = result[feature]["APRIL_2025"].get("0.5"); o = result[feature]["OCTOBER_2025"].get("0.5")
        result[feature]["median_scale_ratio_october_over_april"] = float(o/a) if a not in (None, 0) and o is not None else None
    return result


def _tod_analysis(events: list[dict[str, Any]]) -> dict[str, Any]:
    bins = {"EUROPE": ("EUROPE",), "US_PREOPEN": (), "CASH_OPEN": (), "MORNING": (), "MIDDAY": (), "AFTERNOON": (), "CLOSE": ()}
    # Fixed UTC bins corresponding to NY regular session; April/October are both EDT.
    def economic(ts: int, session: str) -> str:
        sec = ts // 1_000_000_000 % 86400
        if session == "EUROPE" and sec < 13 * 3600: return "EUROPE"
        if 13 * 3600 <= sec < 13 * 3600 + 30 * 60: return "US_PREOPEN"
        if 13 * 3600 + 30 * 60 <= sec < 14 * 3600 + 30 * 60: return "CASH_OPEN"
        if 14 * 3600 + 30 * 60 <= sec < 16 * 3600: return "MORNING"
        if 16 * 3600 <= sec < 18 * 3600: return "MIDDAY"
        if 18 * 3600 <= sec < 19 * 3600 + 30 * 60: return "AFTERNOON"
        if 19 * 3600 + 30 * 60 <= sec < 20 * 3600: return "CLOSE"
        return "OTHER"
    result = {}
    for feature in ("RV_30S_RAW", "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",
                    "TOP5_SAME_SIDE_DEPTH_RAW", "TOP5_DEPTH_TOD_PERCENTILE", "TRADES_PER_SECOND_5S", "TRADE_INTENSITY_TOD_PERCENTILE"):
        result[feature] = {}
        for bucket in ("EUROPE", "US_PREOPEN", "CASH_OPEN", "MORNING", "MIDDAY", "AFTERNOON", "CLOSE"):
            rows = [e for e in events if economic(int(e["interaction_start_ns"]), str(e.get("session"))) == bucket]
            result[feature][bucket] = _summarize_values(e.get(feature) for e in rows)
    return {"fixed_bucket_semantics": "UTC: Europe 08:00-13:00; US preopen 13:00-13:30; cash open 13:30-14:30; morning 14:30-16:00; midday 16:00-18:00; afternoon 18:00-19:30; close 19:30-20:00",
            "features": result}


def _interaction_results(events: list[dict[str, Any]]) -> dict[str, Any]:
    return _tercile_interaction(events, "AGGRESSION_TO_DEPTH_1S", "ER_5S", "INTERACTION_1_AGGRESSION_DEPTH_X_EFFICIENCY") | \
           _tercile_interaction(events, "RV_30S_TOD_PERCENTILE", "MLOFI_PERSISTENCE_DIRECTIONAL", "INTERACTION_2_RV_PERCENTILE_X_MLOFI_PERSISTENCE")


def _tercile_interaction(events: list[dict[str, Any]], left: str, right: str, name: str) -> dict[str, Any]:
    valid = [e for e in events if e.get(left) is not None and e.get(right) is not None and e["markouts_ticks"].get("5000") is not None]
    if not valid: return {name: {"n": 0}}
    a = np.asarray([float(e[left]) for e in valid]); b = np.asarray([float(e[right]) for e in valid])
    qa, qb = np.quantile(a, [1/3, 2/3]), np.quantile(b, [1/3, 2/3])
    ai = np.searchsorted(qa, a); bi = np.searchsorted(qb, b)
    cells = {}
    for i in range(3):
        for j in range(3):
            rows = [e for e, x, y in zip(valid, ai, bi) if x == i and y == j]
            cells[f"T{i+1}_X_T{j+1}"] = {"n": len(rows), "5s_markout": _summarize_values(e["markouts_ticks"]["5000"] for e in rows)}
    return {name: {"n": len(valid), "left": left, "right": right, "fixed_tercile_edges": {left: qa.tolist(), right: qb.tolist()}, "cells": cells}}


def _prototype_gates(events: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = {}
    for fam in sorted(LIVE_TO_TAPE):
        for period in ("APRIL_2025", "OCTOBER_2025"):
            base = [e for e in events if e["family"] == fam and e["period"] == period]
            finite = lambda e, f: e.get(f) is not None and math.isfinite(float(e[f]))
            er_values = np.asarray([float(e["ER_30S"]) for e in base if e.get("ER_30S") is not None], dtype=np.float64)
            persist_values = np.asarray([float(e["MLOFI_PERSISTENCE_DIRECTIONAL"]) for e in base if e.get("MLOFI_PERSISTENCE_DIRECTIONAL") is not None], dtype=np.float64)
            er_cut = float(np.quantile(er_values, .8)) if len(er_values) else 1.0
            # Most opposing persistence is the lower tail of direction-signed persistence.
            opposing_cut = float(np.quantile(persist_values, .2)) if len(persist_values) else -1.0
            masks = {
                "BASELINE": lambda e: True,
                "A": lambda e: finite(e, "ER_30S") and float(e["ER_30S"]) <= er_cut,
                "B": lambda e: not (finite(e, "MLOFI_PERSISTENCE_DIRECTIONAL") and float(e["MLOFI_PERSISTENCE_DIRECTIONAL"]) < opposing_cut),
                "C": lambda e: finite(e, "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE") and .40 <= float(e["AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE"]) <= .90,
                "D": lambda e: not (finite(e, "RV_30S_TOD_PERCENTILE") and float(e["RV_30S_TOD_PERCENTILE"]) >= .95),
                "E": lambda e: finite(e, "ER_30S") and float(e["ER_30S"]) <= er_cut and not (finite(e, "MLOFI_PERSISTENCE_DIRECTIONAL") and float(e["MLOFI_PERSISTENCE_DIRECTIONAL"]) < opposing_cut),
                "F": lambda e: finite(e, "ER_30S") and float(e["ER_30S"]) <= er_cut and not (finite(e, "MLOFI_PERSISTENCE_DIRECTIONAL") and float(e["MLOFI_PERSISTENCE_DIRECTIONAL"]) < opposing_cut) and not (finite(e, "RV_30S_TOD_PERCENTILE") and float(e["RV_30S_TOD_PERCENTILE"]) >= .95),
            }
            result = {}
            for gate, predicate in masks.items():
                selected = [e for e in base if predicate(e)]
                result[gate] = {"event_count": len(selected), "retention_pct": 100 * len(selected) / len(base) if base else None,
                                "markouts": {str(h): _summarize_values(e["markouts_ticks"].get(str(h)) for e in selected) for h in (2000, 5000, 10000, 30000)},
                                "mfe_mae_5s": {k: _summarize_values(e["mfe_mae_ticks"].get("5000", {}).get(k) for e in selected) for k in ("mfe", "mae")},
                                "barriers": _barrier_summary(selected),
                                "daily_stability_5s": _daily_stability(selected, "RV_30S_TOD_PERCENTILE")}
            outputs[f"{fam}|{period}"] = result
    return outputs


def _prototype_gate_decision(gates: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the fixed-gate screen from cross-period markout evidence, not event presence."""
    coherent: dict[str, list[str]] = {}
    observed_families: set[str] = set()
    for key, result in gates.items():
        family, period = key.rsplit("|", 1)
        if period not in {"APRIL_2025", "OCTOBER_2025"}:
            continue
        baseline = result["BASELINE"]["markouts"]["5000"]
        if baseline.get("n", 0) >= 30:
            observed_families.add(family)
    for family in sorted({key.rsplit("|", 1)[0] for key in gates}):
        for gate in "ABCDEF":
            deltas = []
            for period in ("APRIL_2025", "OCTOBER_2025"):
                result = gates.get(f"{family}|{period}")
                if not result:
                    deltas = []
                    break
                base = result["BASELINE"]["markouts"]["5000"]
                selected = result[gate]["markouts"]["5000"]
                if base.get("n", 0) < 30 or selected.get("n", 0) < 20:
                    deltas = []
                    break
                deltas.append(float(selected["mean"]) - float(base["mean"]))
            if len(deltas) == 2 and all(delta > 0 for delta in deltas):
                coherent.setdefault(family, []).append(gate)
    evidence = {"minimum_baseline_n_per_period": 30, "minimum_gate_n_per_period": 20,
                "families_with_usable_two_period_baselines": sorted(observed_families),
                "families_with_at_least_one_gate_improving_5s_mean_in_both_periods": coherent}
    n_families = len(coherent)
    if n_families >= 2:
        return "WORTH_FURTHER_TESTING", evidence
    if n_families == 1:
        return "PROMISING_BUT_NEEDS_MORE_DATA", evidence
    if len(observed_families) < 2:
        return "PROMISING_BUT_NEEDS_MORE_DATA", evidence
    return "NOT_USEFUL", evidence


def _barrier_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: {outcome: sum(e.get("barriers", {}).get(key) == outcome for e in events) / len(events) if events else None
                  for outcome in ("UP", "DOWN", "TIE", "NO_TOUCH")} for key in (f"+{u}/-{d}" for u, d in BARRIERS)}


def _permutation(events: list[dict[str, Any]], repetitions: int = 199) -> dict[str, Any]:
    seed = 20250930
    rng = np.random.default_rng(seed)
    features = ("RV_30S_TOD_PERCENTILE", "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE", "ER_5S_GLOBAL_PERCENTILE", "MLOFI_PERSISTENCE_DIRECTIONAL")
    out = {}
    for feature in features:
        observed = []; null = []
        for day in sorted({e["date"] for e in events}):
            for family in sorted(LIVE_TO_TAPE):
                rows = [e for e in events if e["date"] == day and e["family"] == family and e.get(feature) is not None and e["markouts_ticks"].get("5000") is not None]
                if len(rows) < 10: continue
                vals = np.asarray([float(e[feature]) for e in rows]); y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows])
                labels = _bucket_index(vals)
                if not (np.any(labels == 0) and np.any(labels == 4)): continue
                observed.append(float(y[labels == 4].mean() - y[labels == 0].mean()))
                for _ in range(repetitions):
                    perm = rng.permutation(labels)
                    null.append(float(y[perm == 4].mean() - y[perm == 0].mean()))
        out[feature] = {"seed": seed, "repetitions_per_date_family": repetitions, "observed_high_minus_low_5s_effects": observed,
                        "null_high_minus_low_5s_effects": null,
                        "observed_mean_absolute": float(np.mean(np.abs(observed))) if observed else None,
                        "null_95_abs": float(np.quantile(np.abs(null), .95)) if null else None,
                        "exceeds_null_95": bool(np.mean(np.abs(observed)) > np.quantile(np.abs(null), .95)) if observed and null else None}
    return out


def _family_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for family in sorted(LIVE_TO_TAPE):
        subsets = [e for e in events if e["family"] == family]
        entry = {"periods": {}}
        for period in ("APRIL_2025", "OCTOBER_2025"):
            rows = [e for e in subsets if e["period"] == period]
            entry["periods"][period] = {"event_count": len(rows), "active_dates": len({r["date"] for r in rows}),
                "feature_results": _feature_summary(rows),
                "daily_stability": {f: _daily_stability(rows, f) for f in ("RV_30S_TOD_PERCENTILE", "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE", "ER_5S", "MLOFI_PERSISTENCE_DIRECTIONAL", "RESILIENCY_MEDIAN_500MS_PRIOR_60S")},
                "markouts": {str(h): _summarize_values(r["markouts_ticks"].get(str(h)) for r in rows) for h in HORIZONS_MS},
                "mfe_mae": {str(h): {k: _summarize_values(r["mfe_mae_ticks"].get(str(h), {}).get(k) for r in rows) for k in ("mfe", "mae")} for h in PATH_HORIZONS_MS},
                "barriers": _barrier_summary(rows)}
        out[FAMILY_MAP[LIVE_TO_TAPE[family]]] = {"live_family": LIVE_TO_TAPE[family], **entry}
    return out


def _date_manifest(date_rows: Mapping[str, Mapping[str, Path]], tapes: Mapping[str, Path], source_hashes: Mapping[str, str],
                   configs: Mapping[str, Any], config_sha: str) -> dict[str, Any]:
    april = sorted(d for d in date_rows if d.startswith("2025-04-"))
    october = sorted(d for d in date_rows if d.startswith("2025-10-"))
    oct_dependency = next((r for r in baseline._manifest(DATA_ROOT)[1].values() if isinstance(r, dict) and r.get("session_date") == "2025-10-06" and r.get("category") == "DEPENDENCY"), None)
    dep_ok = bool(oct_dependency and (DATA_ROOT / oct_dependency["path"]).is_file())
    return {"run_id": RUN_ID, "dataset": "GLBX.MDP3 ES native MBP-10", "periods": {"APRIL_2025": april, "OCTOBER_2025": october},
            "expected_april_dates": april, "expected_october_dates": october, "october_dependency_2025_10_06_present": dep_ok,
            "missing_dates": [], "data_downloaded": False, "oos_accessed": False, "dbn_source_used": True,
            "source_files": {d: {"path": str(date_rows[d]["source"]), "sha256": source_hashes[d],
                                  "bytes": date_rows[d]["source"].stat().st_size, "category": "TRAIN" if d.startswith("2025-04") else "VALIDATION"}
                              for d in sorted(date_rows)},
            "tape_files": {d: {"path": str(tapes[d]), "sha256": _sha(tapes[d])} for d in sorted(tapes)},
            "contract_by_period": {"APRIL_2025": "ESM5", "OCTOBER_2025": "ESZ5"}, "config_sha256": config_sha,
            "strategy_manifest_sha256": EXPECTED_STRATEGY_MANIFEST_SHA,
            "runtime_contract_sha256": EXPECTED_RUNTIME_CONTRACT_SHA,
            "frozen_live_configs": configs,
            "candidate_tape_versions": {"APRIL_2025": "stored train CandidateTape source; V2 semantic hash verified", "OCTOBER_2025": "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE"},
            "event_definition": "all completed causal interactions in the four exact live canonical families passing the frozen live Class-A technical and quality gates; event anchor is interaction_start_ns",
            "strict_causality": "market features use only records with ts_recv < interaction_start_ns; historical percentile distributions are appended only after the date finishes",
            "tod_scheme": "five-minute bins relative to repository ASIA/EUROPE/NY session start; UTC economic buckets are fixed in time-of-day-analysis.json",
            "normalization_limitations": ["No pre-April historical percentile seed was read; first target date has null prior-date percentiles.", "Resiliency is an aggregate top-five-depth recovery proxy sampled every 50ms, not an order-ID refill measure.", "Price impact per MLOFI is a descriptive ratio defined here, not claimed to be a separately validated existing production feature."],
            "mlofi_semantics": "existing price-keyed account_mbp_event equivalent; TOP5, INVERSE_LEVEL, DEPTH_NORMALIZED uses inverse rank weights 1/(rank+1) and the existing weighted current-time mean displayed depth denominator; eight prior 250ms contribution bins for persistence"}


def run(*, output_root: Path = OUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    global OUT_ROOT
    OUT_ROOT = output_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "checkpoints").mkdir(exist_ok=True)
    configs, config_sha = _load_live_configs()
    date_rows, tapes, source_hashes = _sources_and_tapes()
    coverage = _date_manifest(date_rows, tapes, source_hashes, configs, config_sha)
    _write_json(OUT_ROOT / "source-coverage.json", coverage)
    if not coverage["october_dependency_2025_10_06_present"]:
        raise StudyError("October 2025 profile dependency date 2025-10-06 is missing")
    order = sorted(date_rows)
    if smoke:
        order = ["2025-04-01"]
    history_global: dict[str, list[float]] = {}
    history_tod: dict[str, dict[str, list[float]]] = {}
    all_events: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    timings = {}; progress = {"completed_dates": [], "failed_dates": [], "resumed_dates": []}
    for day in order:
        print(f"[absnorm] date_start={day} position={len(progress['completed_dates']) + len(progress['resumed_dates']) + 1}/{len(order)}", flush=True)
        src_sha = source_hashes[day]; tape_path = tapes[day]
        cp = _valid_checkpoint(day, src_sha, tape_path, config_sha)
        if cp:
            payload = cp
            if _refresh_checkpoint_path_outcomes(payload, tape_path, day):
                _write_gzip_json(_date_checkpoint_path(day), payload)
                print(f"[absnorm] path_outcomes_refreshed={day} version={PATH_OUTCOME_VERSION}", flush=True)
            progress["resumed_dates"].append(day)
            print(f"[absnorm] date_checkpoint_reused={day} events={len(payload.get('events', []))}", flush=True)
        else:
            payload, elapsed = _extract_date(day, date_rows[day]["source"], tape_path, src_sha, configs)
            timings[day] = elapsed
            _add_percentiles(payload["events"], payload["samples"], history_global, history_tod, day)
            payload.update({"status": "COMPLETE", "config_sha256": config_sha, "source_sha256": src_sha,
                            "tape_sha256": _sha(tape_path), "elapsed_seconds": elapsed})
            _write_gzip_json(_date_checkpoint_path(day), payload)
            progress["completed_dates"].append(day)
            print(f"[absnorm] date_complete={day} events={len(payload['events'])} elapsed_seconds={elapsed:.1f}", flush=True)
        # Checkpoints store already-ranked events; history state is deterministically rebuilt from earlier completed checkpoint samples.
        if cp:
            # Rebuild the prior-only calibration state by adding this date's samples after reading its ranked events.
            _add_percentiles([], payload.get("samples", []), history_global, history_tod, day)
        all_events.extend(payload.get("events", [])); all_samples.extend(payload.get("samples", []))
        _write_json(OUT_ROOT / "checkpoints" / "progress.json", {**progress, "current_date": day,
                    "target_date_count": len(order), "source_manifest_sha256": _sha(DATA_ROOT / baseline.MANIFEST_NAME),
                    "config_sha256": config_sha})
    return _finish(coverage, all_events, all_samples, progress, timings, configs, config_sha, smoke)


def _finish(coverage: dict[str, Any], events: list[dict[str, Any]], samples: list[dict[str, Any]],
            progress: dict[str, Any], timings: dict[str, Any], configs: Mapping[str, Any], config_sha: str,
            smoke: bool) -> dict[str, Any]:
    if smoke:
        return {"status": "SMOKE_PASS" if events else "SMOKE_NO_EVENTS", "events": len(events), "dates": progress["completed_dates"]}
    if not events:
        raise StudyError("no frozen live-family events qualified in the requested periods")
    summary = _aggregate(events, samples)
    feature_results = _feature_summary(events)
    family_results = _family_report(events)
    daily = {f: _daily_stability(events, f) for f in (
        "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE", "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE",
        "TOP5_DEPTH_TOD_PERCENTILE", "TRADE_INTENSITY_TOD_PERCENTILE", "VELOCITY_10S_TOD_PERCENTILE",
        "ER_5S", "MLOFI_PERSISTENCE_DIRECTIONAL", "RESILIENCY_MEDIAN_500MS_PRIOR_60S")}
    mfeat = {"RV_30S_RAW": "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_RAW": "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",
             "TOP5_SAME_SIDE_DEPTH_RAW": "TOP5_DEPTH_TOD_PERCENTILE", "TRADES_PER_SECOND_5S": "TRADE_INTENSITY_TOD_PERCENTILE",
             "VELOCITY_10S_TICKS_PER_SECOND": "VELOCITY_10S_TOD_PERCENTILE", "ER_5S": "ER_5S_GLOBAL_PERCENTILE",
             "MLOFI_5S_DEPTH_NORMALIZED": "MLOFI_PERCENTILE_TOD"}
    raw_relative = {}
    for raw, rel in mfeat.items():
        raw_relative[raw] = {"raw_quintiles": _markout_table(events, raw), "relative_quintiles": _markout_table(events, rel),
                             "classification": _normalization_class({}, {}, events, raw, rel)}
    relative_counts = {}
    for value in [x["classification"] for x in raw_relative.values()]: relative_counts[value] = relative_counts.get(value, 0) + 1
    if relative_counts.get("RELATIVE_CLEARLY_BETTER", 0) >= 3:
        primary = "RELATIVE_NORMALIZATION_STRONGLY_IMPROVES_CROSS_REGIME_STABILITY"
    elif relative_counts.get("RELATIVE_MODESTLY_BETTER", 0) + relative_counts.get("RELATIVE_CLEARLY_BETTER", 0) >= 2:
        primary = "RELATIVE_NORMALIZATION_MODESTLY_IMPROVES_CROSS_REGIME_STABILITY"
    elif relative_counts.get("SIMILAR", 0) + relative_counts.get("RAW_BETTER", 0) >= 5:
        primary = "RAW_AND_RELATIVE_SIMILAR" if relative_counts.get("SIMILAR", 0) >= 4 else "RELATIVE_NORMALIZATION_UNSTABLE"
    elif relative_counts.get("NO_RELATIONSHIP", 0) >= 3:
        primary = "NO_USEFUL_RELATIVE_REGIME_EFFECT"
    elif relative_counts.get("INSUFFICIENT", 0) >= 3:
        primary = "INSUFFICIENT_EVIDENCE"
    else: primary = "ONLY_SPECIFIC_RELATIVE_FEATURES_HELP"
    evt_summary = {}
    for live in LIVE_TO_TAPE.values():
        evt_summary[live] = {p: sum(e["live_family"] == live and e["period"] == p for e in events) for p in ("APRIL_2025", "OCTOBER_2025")}
    prototype_results = _prototype_gates(events)
    prototype_decision, prototype_evidence = _prototype_gate_decision(prototype_results)
    events_jsonl = OUT_ROOT / "event-features.jsonl.gz"
    _write_gzip_json(events_jsonl, events)
    res = {
        "summary.json": {"run_id": RUN_ID, "status": "COMPLETE", "dataset": "APRIL_2025 + OCTOBER_2025",
                         "period_dates": coverage["periods"], "live_family_mapping": FAMILY_MAP,
                         "frozen_config_identity": "PASS", "event_counts": evt_summary, "relative_feature_classifications": raw_relative,
                         "primary_decision": primary, "prototype_gate_decision": prototype_decision,
                         "prototype_gate_evidence": prototype_evidence,
                         "oos_accessed": False, "optimization_performed": False, "parameter_search_performed": False, "pnl_optimization_performed": False,
                         "full_study": True, "smoke_date": "2025-04-01"},
        "report.md": _report_md(events, evt_summary, primary, raw_relative, coverage),
        "raw-feature-distributions.json": summary["raw_feature_distributions"],
        "relative-feature-distributions.json": summary["relative_feature_distributions"],
        "raw-vs-relative.json": summary["raw_vs_relative"],
        "volatility-results.json": {k: v for k, v in summary["feature_results"].items() if k.startswith("RV_")},
        "aggression-results.json": {k: v for k, v in summary["feature_results"].items() if "AGGRESS" in k},
        "depth-results.json": {k: v for k, v in summary["feature_results"].items() if "DEPTH" in k},
        "intensity-results.json": {k: v for k, v in summary["feature_results"].items() if "INTENSITY" in k or "SECOND" in k},
        "velocity-results.json": {k: v for k, v in summary["feature_results"].items() if "VELOCITY" in k},
        "trend-efficiency-results.json": {k: v for k, v in summary["feature_results"].items() if k.startswith("ER_")},
        "mlofi-results.json": {k: v for k, v in summary["feature_results"].items() if "MLOFI" in k},
        "resiliency-results.json": {k: v for k, v in summary["feature_results"].items() if "RESILIENCY" in k},
        "cross-regime-comparison.json": _quantile_shift(events),
        "time-of-day-analysis.json": _tod_analysis(events), "family-results.json": family_results,
        "mfe-mae-results.json": _mfe_mae(events), "barrier-results.json": _barriers_by_feature(events),
        "daily-stability.json": daily, "interaction-results.json": _interaction_results(events),
        "neighbor-stability.json": {feature: _neighbor_stability(events, feature) for feature in (
            "RV_30S_RAW", "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",
            "AGGRESSION_TO_DEPTH_1S", "TOP5_SAME_SIDE_DEPTH_RAW", "TOP5_DEPTH_TOD_PERCENTILE", "ER_5S",
            "MLOFI_5S_DEPTH_NORMALIZED", "MLOFI_PERSISTENCE_DIRECTIONAL", "RESILIENCY_MEDIAN_500MS_PRIOR_60S")},
        "prototype-gates.json": prototype_results, "permutation-results.json": _permutation(events),
    }
    for name, value in res.items():
        if name == "report.md":
            p = OUT_ROOT / name; p.write_text(value, encoding="utf-8")
        else: _write_json(OUT_ROOT / name, value)
    _write_json(OUT_ROOT / "summary.json", res["summary.json"])
    manifest = {"run_id": RUN_ID, "status": "COMPLETE", "dataset": "April + October 2025", "source_coverage": coverage,
                "frozen_config_sha256": config_sha, "frozen_strategy_manifest_sha256": EXPECTED_STRATEGY_MANIFEST_SHA,
                "completed_date_checkpoints": progress["completed_dates"] + progress["resumed_dates"],
                "date_processing_seconds": timings, "event_counts": evt_summary,
                "event_features_sha256": _sha(events_jsonl), "result_files_sha256": {p.name: _sha(p) for p in OUT_ROOT.glob("*.json") if p.name != "run-manifest.json"},
                "optimization_performed": False, "oos_accessed": False, "data_downloaded": False,
                "note": "April/October are diagnostic research periods, not final OOS."}
    _write_json(OUT_ROOT / "run-manifest.json", manifest)
    _write_json(OUT_ROOT / "checkpoints" / "progress.json", {**progress, "status": "COMPLETE", "summary": res["summary.json"]})
    return res["summary.json"]


def _mfe_mae(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for feature in ("RV_30S_RAW", "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",
                    "AGGRESSION_TO_DEPTH_1S", "TOP5_SAME_SIDE_DEPTH_RAW", "TOP5_DEPTH_TOD_PERCENTILE", "ER_5S", "MLOFI_5S_DEPTH_NORMALIZED"):
        out[feature] = {}
        selected = [e for e in events if e.get(feature) is not None]
        vals = np.asarray([float(e[feature]) for e in selected], dtype=np.float64)
        labels = _bucket_index(vals)
        for bucket in range(5):
            bucket_rows = [e for e, bi in zip(selected, labels) if bi == bucket]
            out[feature][f"Q{bucket+1}"] = {}
            for horizon in PATH_HORIZONS_MS:
                out[feature][f"Q{bucket+1}"][str(horizon)] = {}
                for measure in ("mfe", "mae"):
                    series = [e.get("mfe_mae_ticks", {}).get(str(horizon), {}).get(measure) for e in bucket_rows]
                    stats = _summarize_values(series)
                    finite = np.asarray([v for v in series if v is not None], dtype=np.float64)
                    for threshold in (1, 2, 4, 8):
                        stats[f"p_ge_{threshold}_ticks"] = float(np.mean(finite >= threshold)) if finite.size else None
                    out[feature][f"Q{bucket+1}"][str(horizon)][measure] = stats
    return out


def _barriers_by_feature(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for feature in ("RV_30S_RAW", "RV_30S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_1S_RAW", "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE",
                    "AGGRESSION_TO_DEPTH_1S", "TOP5_SAME_SIDE_DEPTH_RAW", "TOP5_DEPTH_TOD_PERCENTILE", "ER_5S", "MLOFI_5S_DEPTH_NORMALIZED"):
        rows = [e for e in events if e.get(feature) is not None]
        if not rows:
            out[feature] = {"n": 0, "quintiles": {}}
            continue
        labels = _bucket_index(np.asarray([float(e[feature]) for e in rows], dtype=np.float64))
        buckets = {}
        for b in range(5):
            chosen = [e for e, label in zip(rows, labels) if label == b]
            buckets[f"Q{b+1}"] = {"event_count": len(chosen), "first_touch_within_30s": _barrier_summary(chosen)}
        out[feature] = {"n": len(rows), "quintile_source": feature, "quintiles": buckets}
    return out


def _report_md(events: list[dict[str, Any]], counts: Mapping[str, Any], decision: str,
               comparisons: Mapping[str, Any], coverage: Mapping[str, Any]) -> str:
    lines = [f"# {RUN_ID}", "", "Diagnostic event-context study only; no optimization, strategy PnL, downloads, or 2026 OOS access.",
             "", f"Primary decision: **{decision}**", "", "## Frozen event counts", "", "| Live family | April | October |", "|---|---:|---:|"]
    for fam, row in counts.items(): lines.append(f"| {fam} | {row['APRIL_2025']} | {row['OCTOBER_2025']} |")
    lines += ["", "## Paired raw/relative classifications", "", "| Raw feature | Relative feature | Classification |", "|---|---|---|"]
    for feature, item in comparisons.items():
        for rel, result in item.get("relative_features", {}).items():
            lines.append(f"| {feature} | {rel} | {result} |")
        if "classification" in item: lines.append(f"| {feature} | {item.get('relative_feature','')} | {item['classification']} |")
    lines += ["", "## Methods and caveats", "", "- Event anchor is interaction_start_ns; context is strictly as-of the last native MBP row with ts_recv < anchor.",
              "- TOD percentile calibration uses earlier completed dates only, with fixed 5-minute session-relative bins. No same-day rows calibrate event ranks.",
              "- No pre-April source dates were opened; early-April percentile values are null until a previous eligible date exists.",
            "- Resiliency is an aggregate top-five depth recovery proxy, sampled at 50ms; it cannot identify individual-order refill causality.",
              "- MLOFI uses the established price-keyed TOP5/INVERSE_LEVEL/DEPTH_NORMALIZED accounting. Candidate path provides forward markouts.",
              "- Counts, daily stability, and per-date reports are descriptive; events are not independent trials.",
              "", "## Covered dates", "", f"April: {', '.join(coverage['periods']['APRIL_2025'])}",
              f"October: {', '.join(coverage['periods']['OCTOBER_2025'])}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true", help="process one April date only")
    args = parser.parse_args(argv)
    result = run(output_root=args.output_root, smoke=args.smoke)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
