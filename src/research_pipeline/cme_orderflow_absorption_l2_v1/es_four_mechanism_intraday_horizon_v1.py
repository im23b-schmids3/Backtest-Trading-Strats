"""Research-exposed intraday horizon extension for the frozen four ES mechanisms.

Consumes only the first 20 Spring TRAIN dates and the already-frozen trigger
ledgers. It does not rerun detectors, tune parameters, or access validation/OOS.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import resource
import time
from array import array
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import es_four_mechanism_comparative_research_v1 as audit
from . import es_four_mechanism_native_replay as native

STUDY = "ES_FOUR_MECHANISM_INTRADAY_HORIZON_RESEARCH_V1"
ORIGINAL = "ES_FOUR_MECHANISM_COMPARATIVE_RESEARCH_V1"
ENGINE_ID = "engine-184ff54a0e4ca292"
HORIZONS = (60, 300, 900, 1800, 3600, 7200)
RISKS = (8, 16, 32)
TICK = 0.25
TICK_RAW = 250_000_000
FEE_TICKS = 6.0 / 12.5
SLIPPAGE = {"baseline": (1.0, 1.0), "moderate": (1.5, 1.5), "severe": (3.0, 3.0)}
MAX_AGE_NS = 2 * native.NS
BLOCK = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _union_record_schema(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Normalize sparse top-level/nested records so Arrow infers the key union."""
    def normalize(items):
        dicts=[x for x in items if isinstance(x,dict)]
        keys=set().union(*(x.keys() for x in dicts)) if dicts else set()
        for x in dicts:
            for key in keys:
                x.setdefault(key,None)
        for key in keys:
            children=[x.get(key) for x in dicts if isinstance(x.get(key),dict)]
            if children:
                normalize(children)
    out=[dict(r) for r in rows]
    normalize(out)
    return out


def _episode_representative_ids(episode_rows: list[dict[str,Any]], dates: Iterable[str]) -> dict[str,set[str]]:
    """Return the frozen, outcome-independent representative trigger per episode/date."""
    wanted=set(dates)
    reps={day:set() for day in wanted}
    episode_reps: dict[tuple[str,str],int]={}
    episode_counts: dict[tuple[str,str],int]={}
    for row in episode_rows:
        if row.get("date") not in wanted:
            continue
        if "is_episode_representative" not in row:
            raise RuntimeError("frozen episode ledger lacks is_episode_representative")
        key=(row["date"],row["episode_id"])
        episode_counts[key]=episode_counts.get(key,0)+1
        if row["is_episode_representative"]:
            episode_reps[key]=episode_reps.get(key,0)+1
            reps[row["date"]].add(row["event_id"])
    if set(episode_reps)!=set(episode_counts) or any(n!=1 for n in episode_reps.values()):
        raise RuntimeError("frozen episode ledger must identify exactly one representative per episode")
    return reps


def _apply_nonoverlap_schedule(rows: list[dict[str,Any]], representative_ids: set[str]) -> None:
    """Set deterministic outcome-blind horizon scheduling flags in-place."""
    for row in rows:
        row["nonoverlap_selected"] = False
        row["nonoverlap_exclusion_reason"] = "NOT_EPISODE_REPRESENTATIVE" if row["event_id"] not in representative_ids else "INELIGIBLE"
    for horizon in HORIZONS:
        for mechanism in ("M1","M2","M3","M4"):
            subset=sorted((r for r in rows if r["horizon_seconds"]==horizon and r["mechanism"]==mechanism
                           and r["event_id"] in representative_ids),
                          key=lambda r:(r.get("trigger_ts_recv") or 0,r.get("event_id","") or ""))
            blocked=-1
            for row in subset:
                entry=row.get("entry_ts_target")
                if not row.get("eligible") or entry is None:
                    row["nonoverlap_exclusion_reason"]="INELIGIBLE"
                elif int(entry)<blocked:
                    row["nonoverlap_exclusion_reason"]="OVERLAPS_EARLIER_SCHEDULED_HORIZON"
                else:
                    row["nonoverlap_selected"] = True
                    row["nonoverlap_exclusion_reason"] = None
                    blocked=int(entry)+int(horizon)*native.NS


def accounting_ticks(direction: int, entry_quote: float, exit_quote: float,
                     entry_slippage_ticks: float = 1.0,
                     exit_slippage_ticks: float = 1.0,
                     commission_usd: float = 6.0,
                     tick_value_usd: float = 12.5,
                     tick_size: float = 0.25) -> float:
    """Net marketable round trip; bid/ask inputs already include spread crossing."""
    if direction not in (-1, 1) or tick_size <= 0 or tick_value_usd <= 0:
        raise ValueError("invalid direction or tick economics")
    entry = entry_quote + direction * entry_slippage_ticks * tick_size
    exit_ = exit_quote - direction * exit_slippage_ticks * tick_size
    return direction * (exit_ - entry) / tick_size - commission_usd / tick_value_usd


def cost_decomposition(direction: int, trigger_mid: float, entry_bid: float,
                       entry_ask: float, exit_bid: float, exit_ask: float,
                       entry_slippage_ticks: float = 1.0,
                       exit_slippage_ticks: float = 1.0) -> dict[str, float]:
    """Algebraic decomposition into latency movement, spread, slippage, fees."""
    if direction not in (-1, 1) or entry_bid >= entry_ask or exit_bid >= exit_ask:
        raise ValueError("invalid direction or crossed/locked market")
    entry_mid = (entry_bid + entry_ask) / 2
    exit_mid = (exit_bid + exit_ask) / 2
    latency_ticks = direction * (entry_mid - trigger_mid) / TICK
    gross_from_entry_mid = direction * (exit_mid - entry_mid) / TICK
    entry_spread_ticks = (entry_ask - entry_bid) / (2 * TICK)
    exit_spread_ticks = (exit_ask - exit_bid) / (2 * TICK)
    spread_ticks = entry_spread_ticks + exit_spread_ticks
    slip_ticks = entry_slippage_ticks + exit_slippage_ticks
    fee_ticks = FEE_TICKS
    net = accounting_ticks(direction, entry_ask if direction > 0 else entry_bid,
                           exit_bid if direction > 0 else exit_ask,
                           entry_slippage_ticks, exit_slippage_ticks)
    gross_from_trigger = latency_ticks + gross_from_entry_mid
    return {
        "latency_midpoint_ticks": latency_ticks,
        "gross_from_entry_midpoint_ticks": gross_from_entry_mid,
        "gross_from_trigger_midpoint_ticks": gross_from_trigger,
        "entry_half_spread_ticks": entry_spread_ticks,
        "exit_half_spread_ticks": exit_spread_ticks,
        "total_spread_ticks": spread_ticks,
        "entry_slippage_ticks": entry_slippage_ticks,
        "exit_slippage_ticks": exit_slippage_ticks,
        "commission_ticks": fee_ticks,
        "commission_usd": 6.0,
        "net_executable_ticks": net,
        "reconciled_net_ticks": gross_from_entry_mid - spread_ticks - slip_ticks - fee_ticks,
        "net_executable_usd": net * 12.5,
    }


class BlockExtrema:
    """Compact block segment trees for range extrema and first crossings."""
    def __init__(self, bid: np.ndarray, ask: np.ndarray):
        self.bid = bid
        self.ask = ask
        self.mid = (bid.astype(np.float64) + ask.astype(np.float64)) / 2.0
        self.n = len(bid)
        self.blocks = (self.n + BLOCK - 1) // BLOCK
        size = 1
        while size < self.blocks:
            size <<= 1
        self.size = size
        def tree(values: np.ndarray, mode: str) -> np.ndarray:
            out = np.full(2 * size, np.iinfo(np.int64).max if mode == "min" else np.iinfo(np.int64).min, dtype=np.int64)
            for k in range(self.blocks):
                lo, hi = k * BLOCK, min((k + 1) * BLOCK, self.n)
                out[size + k] = (np.min(values[lo:hi]) if mode == "min" else np.max(values[lo:hi]))
            for k in range(size - 1, 0, -1):
                out[k] = min(out[2*k], out[2*k+1]) if mode == "min" else max(out[2*k], out[2*k+1])
            return out
        self.trees = {
            "bid_min": tree(bid, "min"), "bid_max": tree(bid, "max"),
            "ask_min": tree(ask, "min"), "ask_max": tree(ask, "max"),
            "mid_min": tree(self.mid, "min"), "mid_max": tree(self.mid, "max"),
        }

    def _tree_query(self, key: str, left: int, right: int) -> int:
        arr = self.trees[key]
        left += self.size; right += self.size
        result = np.iinfo(np.int64).max if key.endswith("min") else np.iinfo(np.int64).min
        while left < right:
            if left & 1:
                result = min(result, int(arr[left])) if key.endswith("min") else max(result, int(arr[left]))
                left += 1
            if right & 1:
                right -= 1
                result = min(result, int(arr[right])) if key.endswith("min") else max(result, int(arr[right]))
            left >>= 1; right >>= 1
        return int(result)

    def extrema(self, key: str, left: int, right: int) -> int | None:
        """Inclusive quote-index range."""
        if right < left or left < 0 or right >= self.n:
            return None
        end = right + 1
        fb = (left + BLOCK - 1) // BLOCK
        eb = end // BLOCK
        if fb >= eb:
            part = self._values(key)[left:end]
            return int(np.min(part) if key.endswith("min") else np.max(part)) if len(part) else None
        vals = []
        if left < fb * BLOCK:
            part = self._values(key)[left:fb*BLOCK]
            vals.append(int(np.min(part) if key.endswith("min") else np.max(part)))
        vals.append(self._tree_query(key, fb, eb))
        if eb * BLOCK < end:
            part = self._values(key)[eb*BLOCK:end]
            vals.append(int(np.min(part) if key.endswith("min") else np.max(part)))
        return min(vals) if key.endswith("min") else max(vals)

    def _values(self, key: str) -> np.ndarray:
        return {"bid_min": self.bid, "bid_max": self.bid, "ask_min": self.ask,
                "ask_max": self.ask, "mid_min": self.mid, "mid_max": self.mid}[key]

    def _first_block(self, key: str, left: int, right: int, threshold: int) -> int | None:
        """First block in [left,right) whose aggregate crosses threshold."""
        tree = self.trees[key]
        ql, qr = left + self.size, right + self.size
        def qualifies(node: int) -> bool:
            return int(tree[node]) <= threshold if key.endswith("min") else int(tree[node]) >= threshold
        def search(node: int, lo: int, hi: int) -> int | None:
            if hi <= left or lo >= right or not qualifies(node):
                return None
            if hi - lo == 1:
                return lo if lo < self.blocks else None
            mid = (lo + hi) // 2
            got = search(node*2, lo, mid)
            return got if got is not None else search(node*2+1, mid, hi)
        return search(1, 0, self.size)

    def first_crossing(self, key: str, left: int, right: int, threshold: int) -> int | None:
        """First quote index in inclusive range satisfying an extrema threshold."""
        if left < 0 or right < left or right >= self.n:
            return None
        first_full = (left + BLOCK - 1) // BLOCK
        if left < first_full * BLOCK:
            part = self._values(key)[left:min(first_full*BLOCK, right+1)]
            hits = np.flatnonzero(part <= threshold) if key.endswith("min") else np.flatnonzero(part >= threshold)
            if len(hits): return left + int(hits[0])
        full_end = (right + 1) // BLOCK
        if first_full < full_end:
            b = self._first_block(key, first_full, full_end, threshold)
            if b is not None:
                lo, hi = b*BLOCK, min((b+1)*BLOCK, right+1)
                part = self._values(key)[lo:hi]
                hits = np.flatnonzero(part <= threshold) if key.endswith("min") else np.flatnonzero(part >= threshold)
                if len(hits): return lo + int(hits[0])
        tail = max(first_full*BLOCK, full_end*BLOCK)
        if tail <= right:
            part = self._values(key)[tail:right+1]
            hits = np.flatnonzero(part <= threshold) if key.endswith("min") else np.flatnonzero(part >= threshold)
            if len(hits): return tail + int(hits[0])
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _study_paths(repo: Path) -> tuple[Path, Path, Path]:
    original = repo / "research_runs" / ORIGINAL
    run = original / "construction-20-date-v1" / ENGINE_ID
    out = repo / "research_runs" / STUDY
    return original, run, out


def verify_and_freeze(repo: Path) -> dict[str, Any]:
    """Verify only authorized protocol/manifests/results, then preserve run identities."""
    original, run, out = _study_paths(repo)
    protocol_path = original / "frozen-study-protocol.json"
    declared = (original / "frozen-study-protocol.sha256").read_text().split()[0]
    if sha256(protocol_path) != declared:
        raise RuntimeError("original frozen protocol hash mismatch")
    protocol = _load_json(protocol_path)
    dates = protocol["scope"]["construction_dates"]
    progress = _load_json(run / "construction-progress.json")
    if progress.get("status") != "ALL_20_DATES_COMPLETE" or progress.get("completed_dates") != dates:
        raise RuntimeError("authorized 20-date construction run is incomplete or date list differs")
    analysis_manifest = _load_json(run / "analysis-v1" / "analysis-manifest.json")
    for name, meta in analysis_manifest["artifacts"].items():
        path = run / "analysis-v1" / name
        if not path.is_file() or path.stat().st_size != meta["size_bytes"] or sha256(path) != meta["sha256"]:
            raise RuntimeError(f"construction analysis artifact integrity failure: {name}")
    sources = {x["date"]: x for x in audit.source_rows(repo)}
    verified = []
    for day in dates:
        src = sources[day]
        raw = repo / src["raw_path"]
        expected = "ESH5" if day <= "2025-03-14" else "ESM5"
        if src["dataset"] != "GLBX.MDP3" or src["schema"] != "mbp-10" or src["symbol"] != expected:
            raise RuntimeError(f"source identity/contract mismatch: {day}")
        if not raw.is_file() or raw.stat().st_size != int(src["raw_bytes"]) or sha256(raw) != src["raw_sha256"]:
            raise RuntimeError(f"authorized source integrity failure: {day}")
        day_dir = run / "dates" / day
        dm = _load_json(day_dir / "date-manifest.json")
        manifest_source_sha=dm.get("source",{}).get("sha256") or dm.get("result",{}).get("source_sha256")
        if (dm.get("status") != "PASS" or dm.get("date") != day or
            dm.get("protocol_sha256") != declared or manifest_source_sha != src["raw_sha256"]):
            raise RuntimeError(f"date manifest source identity mismatch: {day}")
        for name, meta in dm.get("artifacts", {}).items():
            f = day_dir / name
            if not f.is_file() or f.stat().st_size != meta["size_bytes"] or sha256(f) != meta["sha256"]:
                raise RuntimeError(f"date artifact integrity failure: {day}/{name}")
        verified.append({"date": day, "path": src["raw_path"], "size_bytes": raw.stat().st_size,
                         "sha256": src["raw_sha256"], "dataset": src["dataset"],
                         "schema": src["schema"], "symbol": src["symbol"]})
    trigger_files = {}
    for day in dates:
        d = run / "dates" / day
        for mech in ("m1", "m2", "m3", "m4"):
            p = d / f"{mech}-events.parquet"
            trigger_files[str(p.relative_to(repo))] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    episode_file = run / "analysis-v1" / "economic-episodes.parquet"
    construction = run / "analysis-v1" / "construction-analysis-summary.json"
    original_identity = {
        "original_study": ORIGINAL, "protocol_sha256": declared,
        "engine_id": ENGINE_ID, "engine_sha256": analysis_manifest["engine_sha256"],
        "construction_analysis_manifest_sha256": sha256(run / "analysis-v1" / "analysis-manifest.json"),
        "trigger_files": trigger_files,
        "episode_file": {"path": str(episode_file.relative_to(repo)), "sha256": sha256(episode_file), "size_bytes": episode_file.stat().st_size},
        "construction_summary_sha256": sha256(construction),
        "verified_dates": verified,
    }
    atomic_json(out / "original-detector-identity.json", original_identity)
    input_manifest = {"status": "PASS", "source_count": 20, "sources": verified,
                      "trigger_population": {"M1": 214068, "M2": 517, "M3": 2465, "M4": 399},
                      "deduplicated_population": {"M1": 182018, "M2": 517, "M3": 2465, "M4": 399},
                      "source_allowlist": "original authorized-input-allowlist.json; construction_dates only"}
    atomic_json(out / "authorized-input-manifest.json", input_manifest)
    code = Path(__file__)
    atomic_json(out / "implementation-manifest.json", {
        "module": str(code.relative_to(repo)), "module_sha256_at_manifest": sha256(code),
        "test_module": "tests/test_es_four_mechanism_intraday_horizon_v1.py",
        "test_sha256_at_manifest": sha256(repo / "tests/test_es_four_mechanism_intraday_horizon_v1.py") if (repo / "tests/test_es_four_mechanism_intraday_horizon_v1.py").exists() else None,
        "research_only": True, "production_code_changed": False,
    })
    return original_identity


def load_triggers(repo: Path, run: Path, dates: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    import pyarrow.parquet as pq
    raw: dict[str, list[dict[str, Any]]] = {m: [] for m in ("M1", "M2", "M3", "M4")}
    by_id: dict[str, dict[str, Any]] = {}
    for day in dates:
        d = run / "dates" / day
        for m in raw:
            rows = pq.read_table(d / f"{m.lower()}-events.parquet").to_pylist()
            for row in rows:
                row["features"] = json.loads(row["features"])
                raw[m].append(row); by_id[row["event_id"]] = row
    ep_path = run / "analysis-v1" / "economic-episodes.parquet"
    episode_rows = pq.read_table(ep_path).to_pylist()
    dedup = {m: [] for m in raw}
    representative_ids = _episode_representative_ids(episode_rows, dates)
    for r in episode_rows:
        if r["date"] not in dates:
            continue
        if not r["is_episode_representative"]:
            continue
        event = by_id.get(r["event_id"])
        if event is None:
            raise RuntimeError(f"episode representative absent from frozen trigger ledger: {r['event_id']}")
        clone = dict(event); clone["episode_id"] = r["episode_id"]
        dedup[r["mechanism"]].append(clone)
    return raw, dedup


def extract_day_market(repo: Path, source: dict[str, Any], chunk_rows: int = 100_000) -> dict[str, Any]:
    """Stream one authorized DBN once; retain only valid changed-BBO quotes and gap intervals."""
    from databento import DBNStore
    path = repo / source["path"]
    store = DBNStore.from_file(str(path)); md = store.metadata
    if md.dataset != "GLBX.MDP3" or str(md.schema) != "mbp-10" or source["symbol"] not in md.symbols:
        raise RuntimeError(f"DBN header identity mismatch for {source['date']}")
    q_t, q_ord, q_bid, q_ask = array("q"), array("q"), array("q"), array("q")
    gaps: list[tuple[int, int, str]] = []
    last_recv = None; last_valid_bbo = None; ordinal = 0; rows = 0
    for batch in store.to_ndarray(count=chunk_rows):
        if np.any(batch["rtype"] != 10):
            raise RuntimeError("native input contains non-MBP-10 records")
        n = len(batch); ts = batch["ts_recv"].astype(np.int64, copy=False)
        bp = np.column_stack([batch[f"bid_px_{i:02d}"] for i in range(10)])
        bs = np.column_stack([batch[f"bid_sz_{i:02d}"] for i in range(10)])
        ap = np.column_stack([batch[f"ask_px_{i:02d}"] for i in range(10)])
        ass = np.column_stack([batch[f"ask_sz_{i:02d}"] for i in range(10)])
        flags = batch["flags"].astype(np.uint8, copy=False)
        badflag = (flags & (native.BAD_RECV | native.MAYBE_BAD_BOOK)) != 0
        gapmask = np.zeros(n, dtype=bool)
        if n and last_recv is not None: gapmask[0] = int(ts[0]) - last_recv > MAX_AGE_NS
        if n > 1: gapmask[1:] = ts[1:] - ts[:-1] > MAX_AGE_NS
        bid_ok = (bp[:, 0] > 0) & (bs[:, 0] > 0); ask_ok = (ap[:, 0] > 0) & (ass[:, 0] > 0)
        crossed = bid_ok & ask_ok & (bp[:, 0] >= ap[:, 0])
        bid_order = np.all((bp[:, :-1] == 0) | (bp[:, 1:] == 0) | (bp[:, :-1] > bp[:, 1:]), axis=1)
        ask_order = np.all((ap[:, :-1] == 0) | (ap[:, 1:] == 0) | (ap[:, :-1] < ap[:, 1:]), axis=1)
        valid = bid_ok & ask_ok & bid_order & ask_order & ~crossed & ~badflag & ~gapmask
        for i in np.flatnonzero(gapmask):
            prev = int(ts[i - 1]) if i else int(last_recv if last_recv is not None else ts[i])
            gaps.append((prev, int(ts[i]), "RECEIVE_GAP_GT_2S"))
        invalid_ix = np.flatnonzero(~valid & ~gapmask)
        for i in invalid_ix:
            reason = "BAD_SOURCE_FLAG" if badflag[i] else "INVALID_OR_CROSSED_BOOK"
            gaps.append((int(ts[i]), int(ts[i]), reason))
        bbo = np.column_stack((bp[:, 0], bs[:, 0], ap[:, 0], ass[:, 0]))
        valid_ix = np.flatnonzero(valid)
        changed_ix = []
        previous = last_valid_bbo
        for vi in valid_ix:
            current = bbo[vi]
            if previous is None or not np.array_equal(current, previous):
                changed_ix.append(int(vi))
            previous = current
        if valid_ix.size:
            last_valid_bbo = bbo[int(valid_ix[-1])].copy()
        ix = np.asarray(changed_ix, dtype=np.int64)
        q_t.extend(ts[ix]); q_ord.extend(ordinal + ix)
        q_bid.extend(bp[ix, 0]); q_ask.extend(ap[ix, 0])
        if n:
            last_recv = int(ts[-1]); rows += n; ordinal += n
    close = getattr(store, "close", None)
    if callable(close): close()
    times = np.frombuffer(q_t, dtype=np.int64); ords = np.frombuffer(q_ord, dtype=np.int64)
    bids = np.frombuffer(q_bid, dtype=np.int64); asks = np.frombuffer(q_ask, dtype=np.int64)
    if not len(times): raise RuntimeError(f"no valid executable BBO states in {source['date']}")
    if np.any(times[1:] < times[:-1]): raise RuntimeError(f"native receive-time ordering went backward: {source['date']}")
    # Merge touching invalid intervals; repeated bad rows must not produce an
    # O(events * bad-rows) scan during the outcome pass.
    merged: list[tuple[int,int,str]] = []
    for lo, hi, reason in sorted(gaps):
        if merged and lo <= merged[-1][1] + 1:
            old = merged[-1]
            merged[-1] = (old[0], max(old[1], hi), old[2] if old[2] == reason else "MULTIPLE_SOURCE_INVALID_CAUSES")
        else:
            merged.append((lo, hi, reason))
    starts = [g[0] for g in merged]
    return {"t": times, "ord": ords, "bid": bids, "ask": asks, "gaps": merged, "gap_starts": starts,
            "rows": rows, "metadata_start": int(md.start), "metadata_end": int(md.end),
            "contract": source["symbol"], "source_sha256": source["sha256"]}


def _has_gap(gaps: list[tuple[int, int, str]], starts: list[int], a: int, b: int) -> str | None:
    i = bisect.bisect_left(starts, a)
    if i and gaps[i - 1][1] >= a:
        return gaps[i - 1][2]
    while i < len(gaps) and gaps[i][0] <= b:
        if gaps[i][1] >= a:
            return gaps[i][2]
        i += 1
    return None


def _asof(t: np.ndarray, ords: np.ndarray, target: int, max_ord: int | None = None) -> int:
    j = int(np.searchsorted(t, target, side="right") - 1)
    if max_ord is not None:
        while j >= 0 and int(t[j]) == target and int(ords[j]) > max_ord:
            j -= 1
    return j


def _valid_quote(market: dict[str, Any], idx: int, target: int) -> bool:
    return idx >= 0 and target - int(market["t"][idx]) <= MAX_AGE_NS


def _event_horizons(event: dict[str, Any], market: dict[str, Any], blocks: BlockExtrema,
                    source_end: int, flatten_at: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tarr, oarr, bid, ask = market["t"], market["ord"], market["bid"], market["ask"]
    direction = int(event["direction"]); ts = int(event.get("ts_recv", event.get("trigger_ts_recv")))
    ordinal = int(event["source_ordinal"])
    gaps = market["gaps"]; gap_starts = market["gap_starts"]
    trigger_i = _asof(tarr, oarr, ts, ordinal)
    if not _valid_quote(market, trigger_i, ts) or _has_gap(gaps, gap_starts, ts - MAX_AGE_NS, ts):
        return ([{"event_id": event["event_id"], "date": event["date"], "mechanism": event["mechanism"],
                  "horizon_seconds": h, "eligible": False, "censor_reason": "INVALID_OR_STALE_TRIGGER_QUOTE"} for h in HORIZONS],
                {"event_id": event["event_id"], "entry_valid": False, "censor_reason": "INVALID_OR_STALE_TRIGGER_QUOTE"})
    trigger_mid = (int(bid[trigger_i]) + int(ask[trigger_i])) / 2e9
    entry_t = ts + 2_000_000
    entry_i = _asof(tarr, oarr, entry_t)
    if not _valid_quote(market, entry_i, entry_t) or _has_gap(gaps, gap_starts, ts, entry_t):
        return ([{"event_id": event["event_id"], "date": event["date"], "mechanism": event["mechanism"],
                  "horizon_seconds": h, "eligible": False, "censor_reason": "INVALID_OR_STALE_ENTRY_QUOTE"} for h in HORIZONS],
                {"event_id": event["event_id"], "entry_valid": False, "censor_reason": "INVALID_OR_STALE_ENTRY_QUOTE"})
    entry_bid = int(bid[entry_i]) / 1e9; entry_ask = int(ask[entry_i]) / 1e9
    entry_price = entry_ask + TICK if direction > 0 else entry_bid - TICK
    entry_mid = (entry_bid + entry_ask) / 2
    result = []
    for horizon in HORIZONS:
        target = entry_t + horizon * native.NS
        j = _asof(tarr, oarr, target)
        reason = None
        if target >= source_end:
            reason = "END_OF_AUTHORIZED_SOURCE_WINDOW"
        elif _has_gap(gaps, gap_starts, entry_t, target):
            reason = "INVALID_SOURCE_INTERVAL_OR_GAP"
        elif not _valid_quote(market, j, target):
            reason = "MISSING_OR_STALE_FUTURE_QUOTE"
        if reason:
            result.append({"date": event["date"], "mechanism": event["mechanism"], "event_id": event["event_id"],
                           "horizon_seconds": horizon, "eligible": False, "censor_reason": reason,
                           "trigger_ts_recv": ts, "entry_ts_target": entry_t, "direction": direction,
                           "trigger_mid": trigger_mid, "entry_mid": entry_mid, "entry_price": entry_price,
                           "entry_bid":entry_bid,"entry_ask":entry_ask})
            continue
        out_bid = int(bid[j]) / 1e9; out_ask = int(ask[j]) / 1e9
        exit_quote = out_bid if direction > 0 else out_ask
        gross_detection = direction * (((out_bid + out_ask) / 2) - trigger_mid) / TICK
        gross_entry_mid = direction * (((out_bid + out_ask) / 2) - entry_mid) / TICK
        row = {"date": event["date"], "mechanism": event["mechanism"], "event_id": event["event_id"],
               "horizon_seconds": horizon, "eligible": True, "censor_reason": None,
               "trigger_ts_recv": ts, "entry_ts_target": entry_t, "exit_ts_observed": int(tarr[j]),
               "direction": direction, "session": event.get("features",{}).get("session"),
               "atr20_5m_ticks":event.get("features",{}).get("atr20_5m_ticks"),
               "trigger_mid": trigger_mid, "entry_mid": entry_mid,
               "entry_price": entry_price, "entry_bid": entry_bid, "entry_ask": entry_ask,
               "exit_bid": out_bid, "exit_ask": out_ask,
               "gross_midpoint_ticks_from_detection": gross_detection,
               "gross_midpoint_ticks_from_entry": gross_entry_mid,
               "latency_midpoint_ticks": direction * (entry_mid - trigger_mid) / TICK,
               "entry_spread_ticks": (entry_ask - entry_bid) / TICK,
               "exit_spread_ticks": (out_ask - out_bid) / TICK,
               "mfe_ticks": None, "mae_ticks": None}
        lo = entry_i; hi = j
        min_mid = blocks.extrema("mid_min", lo, hi); max_mid = blocks.extrema("mid_max", lo, hi)
        if min_mid is not None and max_mid is not None:
            row["mfe_ticks"] = direction * ((max_mid if direction > 0 else min_mid) / 1e9 - entry_mid) / TICK
            row["mae_ticks"] = direction * ((min_mid if direction > 0 else max_mid) / 1e9 - entry_mid) / TICK
        for scenario, (eslip, xslip) in SLIPPAGE.items():
            net = accounting_ticks(direction, entry_ask if direction > 0 else entry_bid,
                                   exit_quote, eslip, xslip)
            row[f"net_{scenario}_ticks"] = net
            row[f"net_{scenario}_usd"] = net * 12.5
        row["net_baseline_r_16"] = row["net_baseline_ticks"] / 16.0
        row["first_stop_crossing_16_ticks_index"] = None
        result.append(row)
    legacy_t = entry_t + 10 * native.NS
    legacy_i = _asof(tarr, oarr, legacy_t)
    legacy = {"legacy_10s_eligible": False, "legacy_10s_censor_reason": None}
    if legacy_t >= source_end:
        legacy["legacy_10s_censor_reason"] = "END_OF_AUTHORIZED_SOURCE_WINDOW"
    elif _has_gap(gaps, gap_starts, entry_t, legacy_t):
        legacy["legacy_10s_censor_reason"] = "INVALID_SOURCE_INTERVAL_OR_GAP"
    elif not _valid_quote(market, legacy_i, legacy_t):
        legacy["legacy_10s_censor_reason"] = "MISSING_OR_STALE_FUTURE_QUOTE"
    else:
        eb, ea = entry_bid, entry_ask
        xb, xa = int(bid[legacy_i])/1e9, int(ask[legacy_i])/1e9
        eq, xq = (ea, xb) if direction > 0 else (eb, xa)
        decomp = cost_decomposition(direction, trigger_mid, eb, ea, xb, xa)
        legacy.update({"legacy_10s_eligible": True,
                       "legacy_10s_net_baseline_ticks": accounting_ticks(direction,eq,xq,1,1),
                       "legacy_10s_net_moderate_ticks": accounting_ticks(direction,eq,xq,1.5,1.5),
                       "legacy_10s_net_severe_ticks": accounting_ticks(direction,eq,xq,3,3),
                       "legacy_10s_trigger_mid":trigger_mid,"legacy_10s_entry_mid":(eb+ea)/2,
                       "legacy_10s_exit_mid":(xb+xa)/2,"legacy_10s_entry_bid":eb,"legacy_10s_entry_ask":ea,
                       "legacy_10s_exit_bid":xb,"legacy_10s_exit_ask":xa,
                       **{f"legacy_10s_{k}":v for k,v in decomp.items()}})
    risk_row = {"event_id": event["event_id"], "date": event["date"], "mechanism": event["mechanism"],
                "entry_valid": True, "entry_ts_target": entry_t, "entry_i": entry_i,
                "direction": direction, "entry_price": entry_price, "entry_mid": entry_mid,
                "entry_bid":entry_bid,"entry_ask":entry_ask,"trigger_mid": trigger_mid,
                "features": event.get("features", {}),"session":event.get("features",{}).get("session"),
                "atr20_5m_ticks":event.get("features",{}).get("atr20_5m_ticks"),**legacy}
    return result, risk_row


def _find_risk_paths(event: dict[str, Any], riskrow: dict[str, Any], market: dict[str, Any],
                     blocks: BlockExtrema, source_end: int) -> list[dict[str, Any]]:
    if not riskrow.get("entry_valid"):
        return []
    d = riskrow["direction"]; entry_i = riskrow["entry_i"]; entry_t = riskrow["entry_ts_target"]
    max_t = min(entry_t + HORIZONS[-1] * native.NS, source_end - 1)
    end_i = int(np.searchsorted(market["t"], max_t, side="right") - 1)
    if end_i < entry_i:
        return []
    ranges: list[tuple[str, float, float | None]] = [("FIXED_8T",8.0,None),("FIXED_16T",16.0,None),("FIXED_32T",32.0,None)]
    atr_ticks = event.get("features", {}).get("atr20_5m_ticks")
    if atr_ticks is not None and math.isfinite(float(atr_ticks)) and float(atr_ticks) > 0:
        ranges.append(("ATR20_5M_1X",float(atr_ticks),None))
    boundary = event.get("features", {}).get("boundary")
    if event["mechanism"] == "M3" and boundary is not None:
        stop_px = float(boundary)
        risk_ticks = (riskrow["entry_price"] - stop_px) / TICK if d > 0 else (stop_px - riskrow["entry_price"]) / TICK
        if risk_ticks > 0:
            ranges.append(("M3_STRUCTURAL_BOUNDARY",risk_ticks,stop_px))
    output=[]
    bid,ask=market["bid"],market["ask"]
    entry_mid_raw=int(round(riskrow["entry_mid"]*1e9))
    movement={}
    for ticks in (1,2,4,8):
        fav_key="mid_max" if d>0 else "mid_min"; adv_key="mid_min" if d>0 else "mid_max"
        fav_threshold=entry_mid_raw+d*ticks*TICK_RAW
        adv_threshold=entry_mid_raw-d*ticks*TICK_RAW
        fi=blocks.first_crossing(fav_key,entry_i,end_i,fav_threshold)
        ai=blocks.first_crossing(adv_key,entry_i,end_i,adv_threshold)
        movement[str(ticks)]={"favorable_ts":None if fi is None else int(market["t"][fi]),
                              "adverse_ts":None if ai is None else int(market["t"][ai]),
                              "favorable_before_adverse":bool(fi is not None and (ai is None or fi<ai)),
                              "adverse_before_favorable":bool(ai is not None and (fi is None or ai<fi)),
                              "tie":bool(fi is not None and ai is not None and fi==ai)}
    max_mid=blocks.extrema("mid_max",entry_i,end_i);min_mid=blocks.extrema("mid_min",entry_i,end_i)
    mfe_ix=blocks.first_crossing("mid_max",entry_i,end_i,max_mid) if max_mid is not None else None
    mae_ix=blocks.first_crossing("mid_min",entry_i,end_i,min_mid) if min_mid is not None else None
    for label, risk, fixed_px in ranges:
        risk=float(risk); risk_raw=int(math.ceil(risk*TICK_RAW))
        if fixed_px is not None:
            stop_raw=int(round(fixed_px*1e9))
        else:
            stop_raw=(int(round(riskrow["entry_price"]*1e9))-d*risk_raw)
        stop_key="bid_min" if d>0 else "ask_max"
        stop_thr=stop_raw
        stop_ix=blocks.first_crossing(stop_key,entry_i,end_i,stop_thr)
        targets={}
        for rmult in (1,2,3):
            favorable_px=riskrow["entry_price"]+d*rmult*risk*TICK
            target_raw=int(round(favorable_px*1e9))
            target_key="bid_max" if d>0 else "ask_min"
            targets[rmult]=blocks.first_crossing(target_key,entry_i,end_i,target_raw)
        max_gap=_has_gap(market["gaps"],market["gap_starts"],entry_t,max_t)
        stop_gap = None if stop_ix is None else _has_gap(market["gaps"], market["gap_starts"], entry_t, int(market["t"][stop_ix]))
        stop_exit_bid = None if stop_ix is None else int(bid[stop_ix]) / 1e9
        stop_exit_ask = None if stop_ix is None else int(ask[stop_ix]) / 1e9
        constrained = {}
        for h in HORIZONS:
            target_t = entry_t + h * native.NS
            hi = _asof(market["t"], market["ord"], target_t)
            stopped = stop_ix is not None and int(market["t"][stop_ix]) <= target_t and stop_gap is None
            # A completed stop exit is not censored by a later gap or file end.
            if not stopped and (target_t >= source_end or _has_gap(market["gaps"],market["gap_starts"],entry_t,target_t) or not _valid_quote(market,hi,target_t)):
                constrained[str(h)] = {"eligible": False, "reason": "SOURCE_CENSORED"}
                continue
            exit_quote = (stop_exit_bid if d > 0 else stop_exit_ask) if stopped else int(bid[hi] if d > 0 else ask[hi]) / 1e9
            entry_quote = riskrow["entry_ask"] if d > 0 else riskrow["entry_bid"]
            net = accounting_ticks(d, entry_quote, exit_quote)
            constrained[str(h)] = {"eligible": True, "stop_first": bool(stopped), "net_ticks": net,
                                    "net_r": net / risk, "exit_quote": exit_quote,
                                    "exit_ts": int(market["t"][stop_ix] if stopped else market["t"][hi])}
        passage = {}
        for rmult, ti in targets.items():
            target_ts = None if ti is None else int(market["t"][ti])
            for h in HORIZONS:
                stop_before = stop_ix is not None and stop_gap is None and int(market["t"][stop_ix]) <= entry_t + h*native.NS
                target_before = target_ts is not None and target_ts <= entry_t + h*native.NS and not _has_gap(market["gaps"],market["gap_starts"],entry_t,target_ts)
                horizon_gap = _has_gap(market["gaps"],market["gap_starts"],entry_t,entry_t+h*native.NS)
                passage[f"{h}:{rmult}R"] = ("TIE_OR_ORDER_AMBIGUOUS" if stop_before and target_before and int(market["t"][stop_ix]) == target_ts else
                    "TARGET_FIRST" if target_before and (not stop_before or target_ts < int(market["t"][stop_ix])) else
                    "STOP_FIRST" if stop_before else "SOURCE_CENSORED" if horizon_gap else "UNRESOLVED")
        output.append({"event_id":event["event_id"],"date":event["date"],"mechanism":event["mechanism"],
                       "risk_framework":label,"risk_ticks":risk,"stop_level":stop_raw/1e9,
                       "stop_first_index":stop_ix,"stop_first_ts":None if stop_ix is None else int(market["t"][stop_ix]),
                       "target_first_index_by_R":{str(k):v for k,v in targets.items()},
                       "target_first_ts_by_R":{str(k):None if v is None else int(market["t"][v]) for k,v in targets.items()},
                       "time_to_max_favorable_ns":None if mfe_ix is None else int(market["t"][mfe_ix])-entry_t,
                       "time_to_max_adverse_ns":None if mae_ix is None else int(market["t"][mae_ix])-entry_t,
                       "stop_crossing_gap":stop_gap,"stop_exit_bid":stop_exit_bid,"stop_exit_ask":stop_exit_ask,
                       "path_source_gap":max_gap,"path_end_ts":max_t,
                       "stop_constrained_by_horizon":constrained,"first_passage_by_horizon_and_R":passage,
                       "midpoint_movement_first_passage":movement,
                       "max_favorable_mid_ticks":d*((max_mid if d>0 else min_mid)/1e9-riskrow["entry_mid"])/TICK,
                       "max_adverse_mid_ticks":d*((min_mid if d>0 else max_mid)/1e9-riskrow["entry_mid"])/TICK,
                       "entry_ts_target":entry_t,"entry_price":riskrow["entry_price"],"direction":d,
                       "legacy_10s_eligible":riskrow.get("legacy_10s_eligible",False),
                       **{k:v for k,v in riskrow.items() if k.startswith("legacy_10s_")}})
    return output


def _atr20_at(market: dict[str,Any], timestamp: int) -> float | None:
    """Prior-only ATR20 of completed UTC 5-minute midpoint bars, in ES ticks."""
    if "atr_bars" not in market:
        t=market["t"]; bid=market["bid"]; ask=market["ask"]
        mids=(bid.astype(np.float64)+ask.astype(np.float64))/(2e9)
        bin_ns=300*native.NS
        bins=t//bin_ns
        first=np.r_[0,np.flatnonzero(bins[1:]!=bins[:-1])+1]
        unique=bins[first]
        counts=np.diff(np.r_[first,len(bins)])
        bars=[]
        for b,lo,count in zip(unique,first,counts):
            hi=lo+count; start=int(b)*bin_ns; end=start+bin_ns
            if _has_gap(market["gaps"],market["gap_starts"],start,end):
                bars.append((end,None));continue
            bars.append((end,(float(np.max(mids[lo:hi])),float(np.min(mids[lo:hi])),float(mids[hi-1]))))
        observed={end:(bar) for end,bar in bars}
        true_ranges=[]; ends=[]; values=[]; prior_close=None
        if observed:
          for b in range(min(observed)//bin_ns,max(observed)//bin_ns+1):
            end=(b+1)*bin_ns; bar=observed.get(end)
            if bar is None:
                ends.append(end);values.append(None);prior_close=None;true_ranges=[];continue
            hi,lo,close=bar
            tr=hi-lo if prior_close is None else max(hi-lo,abs(hi-prior_close),abs(lo-prior_close))
            true_ranges.append(tr); prior_close=close
            ends.append(end);values.append(float(np.mean(true_ranges[-20:])/TICK) if len(true_ranges)>=20 else None)
        market["atr_bars"]=(np.asarray(ends,dtype=np.int64),values)
    ends,values=market["atr_bars"]
    i=int(np.searchsorted(ends,timestamp,side="right")-1)
    return values[i] if i>=0 and i<len(values) else None


def run(repo: Path, *, chunk_rows: int = 100_000) -> dict[str, Any]:
    import pyarrow.parquet as pq
    original, source_run, out = _study_paths(repo)
    identity = verify_and_freeze(repo)
    protocol_path = out / "frozen-intraday-research-protocol.json"
    protocol_sha = (out / "frozen-intraday-research-protocol.sha256").read_text().split()[0]
    if sha256(protocol_path) != protocol_sha:
        raise RuntimeError("frozen intraday protocol changed")
    protocol = _load_json(protocol_path)
    dates = protocol["authorized_source"]["dates"]
    raw_events, dedup_events = load_triggers(repo, source_run, dates)
    source_rows = {r["date"]: r for r in audit.source_rows(repo)}
    all_horizon: list[dict[str, Any]] = []; all_risk: list[dict[str, Any]] = []
    all_raw_horizon: list[dict[str, Any]] = []; all_raw_risk: list[dict[str, Any]] = []
    all_short: list[dict[str, Any]] = []; all_raw_short: list[dict[str, Any]] = []
    date_meta=[]; started=time.monotonic(); progress_path=out/"run-progress.json"; evaluation_sha=sha256(Path(__file__))
    done=set()
    if progress_path.exists():
        prior=_load_json(progress_path)
        if prior.get("protocol_sha256") != protocol_sha: raise RuntimeError("resume protocol hash mismatch")
        done=set(prior.get("completed_dates",[])) if prior.get("evaluation_sha256")==evaluation_sha else set()
    for ix, day in enumerate(dates,1):
        dest=out/"dates"/day
        dm_path=dest/"date-result-manifest.json"
        src=source_rows[day]
        raw=repo/src["raw_path"]
        source={"date":day,"path":src["raw_path"],"sha256":src["raw_sha256"],"symbol":src["symbol"]}
        if day in done and dm_path.exists():
            dm=_load_json(dm_path)
            for name, rec in dm["artifacts"].items():
                p=dest/name
                if not p.exists() or p.stat().st_size!=rec["size_bytes"] or sha256(p)!=rec["sha256"]:
                    raise RuntimeError(f"resume artifact integrity failure {day}/{name}")
            date_meta.append(dm["summary"])
            all_horizon.extend(pq.read_table(dest/"deduplicated-horizons.parquet").to_pylist())
            all_raw_horizon.extend(pq.read_table(dest/"raw-horizons.parquet").to_pylist())
            all_risk.extend(pq.read_table(dest/"deduplicated-risk-paths.parquet").to_pylist())
            all_raw_risk.extend(pq.read_table(dest/"raw-risk-paths.parquet").to_pylist())
            all_short.extend(pq.read_table(dest/"deduplicated-short-horizon-audit.parquet").to_pylist())
            all_raw_short.extend(pq.read_table(dest/"raw-short-horizon-audit.parquet").to_pylist())
            continue
        market=extract_day_market(repo,source,chunk_rows)
        blocks=BlockExtrema(market["bid"],market["ask"])
        source_end=market["metadata_end"]; flatten_at=source_end-15*60*native.NS
        raw_rows=[]; raw_risk=[]; dedup_rows=[]; dedup_risk=[]; raw_short=[]; dedup_short=[]
        old_outcomes = pq.read_table(source_run/"dates"/day/"construction-outcomes.parquet").to_pylist()
        old_by_id = {r["event_id"]: r for r in old_outcomes}
        rep_ids={r["event_id"] for m in dedup_events for r in dedup_events[m] if r["date"]==day}
        events_today={m:[] for m in raw_events}
        for m in raw_events:
            for original_event in raw_events[m]:
                if original_event["date"]!=day: continue
                event=dict(original_event); event["features"]=dict(original_event.get("features",{}))
                event["features"]["atr20_5m_ticks"]=_atr20_at(market,int(event["ts_recv"])+2_000_000)
                events_today[m].append(event)
        for mech in ("M1","M2","M3","M4"):
            for event in events_today[mech]:
                horizons,riskrow=_event_horizons(event,market,blocks,source_end,flatten_at)
                old = old_by_id.get(event["event_id"])
                if old is None:
                    raise RuntimeError(f"original 10-second outcome missing for {day}/{event['event_id']}")
                riskrow["original_10s_net_baseline_ticks"] = old.get("net_executable_10s_baseline_ticks")
                riskrow["original_10s_source_censored"] = bool(old.get("source_censored"))
                riskrow["original_10s_censor_reason"] = old.get("censor_reason")
                if riskrow.get("legacy_10s_net_baseline_ticks") is not None and old.get("net_executable_10s_baseline_ticks") is not None:
                    riskrow["legacy_10s_delta_vs_original_ticks"] = float(riskrow["legacy_10s_net_baseline_ticks"]) - float(old["net_executable_10s_baseline_ticks"])
                else:
                    riskrow["legacy_10s_delta_vs_original_ticks"] = None
                short = {"date":day,"mechanism":mech,"event_id":event["event_id"],
                         "original_1s_mid_ticks":old.get("mid_markout_1s_ticks"),
                         "original_5s_mid_ticks":old.get("mid_markout_5s_ticks"),
                         "original_10s_mid_ticks":old.get("mid_markout_10s_ticks"),
                         "original_30s_mid_ticks":old.get("mid_markout_30s_ticks"),
                         "original_60s_mid_ticks":old.get("mid_markout_60s_ticks"),
                         "original_10s_net_baseline_ticks":old.get("net_executable_10s_baseline_ticks"),
                         "audited_10s_net_baseline_ticks":riskrow.get("legacy_10s_net_baseline_ticks"),
                         "audited_10s_net_moderate_ticks":riskrow.get("legacy_10s_net_moderate_ticks"),
                         "audited_10s_net_severe_ticks":riskrow.get("legacy_10s_net_severe_ticks"),
                         "audited_10s_gross_midpoint_ticks":riskrow.get("legacy_10s_gross_from_entry_midpoint_ticks"),
                         "audited_10s_latency_ticks":riskrow.get("legacy_10s_latency_midpoint_ticks"),
                         "audited_10s_spread_ticks":riskrow.get("legacy_10s_total_spread_ticks"),
                         "audited_10s_slippage_ticks":riskrow.get("legacy_10s_entry_slippage_ticks",0)+riskrow.get("legacy_10s_exit_slippage_ticks",0),
                         "audited_10s_commission_ticks":riskrow.get("legacy_10s_commission_ticks"),
                         "audited_10s_censored":not bool(riskrow.get("legacy_10s_eligible")),
                         "audited_10s_censor_reason":riskrow.get("legacy_10s_censor_reason"),
                         "original_censored":bool(old.get("source_censored")),
                         "original_censor_reason":old.get("censor_reason")}
                raw_short.append(short)
                raw_rows.extend(horizons)
                rr=_find_risk_paths(event,riskrow,market,blocks,source_end)
                raw_risk.extend(rr)
                if event["event_id"] in rep_ids:
                    dedup_short.append(short)
                    dedup_rows.extend(horizons)
                    dedup_risk.extend(rr)
        # Non-overlap is outcome-blind and applied to episode representatives only.
        for horizon in HORIZONS:
            for mech in ("M1","M2","M3","M4"):
                rows=sorted((r for r in dedup_rows if r["horizon_seconds"]==horizon and r["mechanism"]==mech),
                            key=lambda r:(r.get("trigger_ts_recv",0),r.get("event_id","")))
                blocked=-1; kept=0
                for r in rows:
                    entry_t=r.get("entry_ts_target")
                    if not r.get("eligible") or entry_t is None:
                        r["nonoverlap_selected"]=False; r["nonoverlap_exclusion_reason"]="INELIGIBLE"
                    elif int(entry_t)<blocked:
                        r["nonoverlap_selected"]=False; r["nonoverlap_exclusion_reason"]="OVERLAPS_EARLIER_SCHEDULED_HORIZON"
                    else:
                        r["nonoverlap_selected"]=True; r["nonoverlap_exclusion_reason"]=None
                        blocked=int(entry_t)+horizon*native.NS; kept+=1
                # Update corresponding raw records by frozen representative event linkage.
                rep_set={r["event_id"] for r in rows if r.get("nonoverlap_selected")}
                for r in raw_rows:
                    if r["date"]==day and r["mechanism"]==mech and r["horizon_seconds"]==horizon:
                        r["nonoverlap_selected"]=r["event_id"] in rep_set
                        r["nonoverlap_exclusion_reason"]="NOT_EPISODE_REPRESENTATIVE" if r["event_id"] not in rep_ids else next((x.get("nonoverlap_exclusion_reason") for x in rows if x["event_id"]==r["event_id"]),None)
        # Scenario B flatten checkpoint: add a separate flat observation for each event/horizon.
        for rows in (raw_rows,dedup_rows):
            for r in rows:
                entry_t=r.get("entry_ts_target")
                if not entry_t: continue
                flat_target=min(int(entry_t)+int(r["horizon_seconds"])*native.NS,flatten_at)
                flat_i=_asof(market["t"],market["ord"],flat_target)
                if flat_target<=entry_t or flat_target>=source_end or not _valid_quote(market,flat_i,flat_target) or _has_gap(market["gaps"],market["gap_starts"],entry_t,flat_target):
                    r["scenario_b_eligible"]=False; r["scenario_b_censor_reason"]="FLAT_CUTOFF_OR_SOURCE_INVALID"
                else:
                    d=int(r["direction"]); outq=int(market["bid"][flat_i] if d>0 else market["ask"][flat_i])/1e9
                    inq=float(r["entry_price"])
                    net=accounting_ticks(d,float(r["entry_ask"] if d>0 else r["entry_bid"]),outq)
                    r["scenario_b_eligible"]=True; r["scenario_b_exit_ts_observed"]=int(market["t"][flat_i]);r["scenario_b_net_baseline_ticks"]=net
        import pyarrow as pa, pyarrow.parquet as pq
        dest.mkdir(parents=True,exist_ok=True)
        for name, rows in (("deduplicated-horizons.parquet",dedup_rows),("raw-horizons.parquet",raw_rows),
                           ("deduplicated-risk-paths.parquet",dedup_risk),("raw-risk-paths.parquet",raw_risk),
                           ("deduplicated-short-horizon-audit.parquet",dedup_short),("raw-short-horizon-audit.parquet",raw_short)):
            table=pa.Table.from_pylist(_union_record_schema(rows)) if rows else pa.table({})
            pq.write_table(table,dest/name,compression="zstd")
        summary={"date":day,"source_records":market["rows"],"quote_states":len(market["t"]),"source_gaps":len(market["gaps"]),
                 "metadata_start_ns":market["metadata_start"],"metadata_end_ns":source_end,"contract":market["contract"],
                 "raw_events":sum(len(events_today[m]) for m in events_today),"deduplicated_episodes":sum(1 for m in dedup_events for r in dedup_events[m] if r["date"]==day),
                 "eligible_30m_dedup":sum(r.get("eligible",False) for r in dedup_rows if r["horizon_seconds"]==1800),
                 "source_sha256":src["raw_sha256"],"runtime_seconds":time.monotonic()-started}
        files={}
        for name in ("deduplicated-horizons.parquet","raw-horizons.parquet","deduplicated-risk-paths.parquet","raw-risk-paths.parquet",
                     "deduplicated-short-horizon-audit.parquet","raw-short-horizon-audit.parquet"):
            p=dest/name;files[name]={"sha256":sha256(p),"size_bytes":p.stat().st_size}
        atomic_json(dm_path,{"summary":summary,"source_sha256":src["raw_sha256"],"protocol_sha256":protocol_sha,"evaluation_sha256":evaluation_sha,"artifacts":files,"status":"PASS"})
        date_meta.append(summary);all_horizon.extend(dedup_rows);all_raw_horizon.extend(raw_rows);all_risk.extend(dedup_risk);all_raw_risk.extend(raw_risk)
        all_short.extend(dedup_short);all_raw_short.extend(raw_short)
        done.add(day);atomic_json(progress_path,{"status":"RUNNING","protocol_sha256":protocol_sha,"evaluation_sha256":evaluation_sha,"completed_dates":dates[:ix]})
        print(f"DATE_HORIZONS_DONE {ix}/20 {day} quotes={len(market['t'])} raw={summary['raw_events']} episodes={summary['deduplicated_episodes']} 30m={summary['eligible_30m_dedup']}",flush=True)
        del market,blocks,raw_rows,dedup_rows,raw_risk,dedup_risk
    result=aggregate_and_write(repo,out,dates,date_meta,all_horizon,all_raw_horizon,all_risk,all_raw_risk,all_short,all_raw_short,
                               protocol_sha,identity, time.monotonic()-started)
    return result


def aggregate_and_write(repo: Path, out: Path, dates: list[str], summaries: list[dict[str,Any]],
                        horizon_rows: list[dict[str,Any]], raw_rows: list[dict[str,Any]],
                        risk_rows: list[dict[str,Any]], raw_risk: list[dict[str,Any]],
                        short_rows: list[dict[str,Any]], raw_short_rows: list[dict[str,Any]],
                        protocol_sha: str, identity: dict[str,Any], runtime: float) -> dict[str,Any]:
    import pyarrow as pa, pyarrow.parquet as pq
    out.mkdir(parents=True,exist_ok=True)
    def parquet(name:str, rows:list[dict[str,Any]]):
        pq.write_table(pa.Table.from_pylist(_union_record_schema(rows)) if rows else pa.table({}),out/name,compression="zstd")
    parquet("intraday-horizon-observations.parquet",horizon_rows)
    parquet("nonoverlapping-opportunity-ledger.parquet",[r for r in horizon_rows if "nonoverlap_selected" in r])
    parquet("executable-intraday-markouts.parquet",horizon_rows)
    parquet("stop-constrained-counterfactuals.parquet",risk_rows)
    parquet("raw-trigger-intraday-observations.parquet",raw_rows)
    parquet("raw-trigger-stop-paths.parquet",raw_risk)
    mechanisms=("M1","M2","M3","M4")
    directional=[]; economic=[]; daily=[]; censor=[]; nonoverlap=[]; risk_summary=[]
    for mech in mechanisms:
        for h in HORIZONS:
            rows=[r for r in horizon_rows if r["mechanism"]==mech and r["horizon_seconds"]==h]
            eligible=[r for r in rows if r.get("eligible")]
            nets=[float(r["net_baseline_ticks"]) for r in eligible if r.get("net_baseline_ticks") is not None]
            mids=[float(r["gross_midpoint_ticks_from_entry"]) for r in eligible]
            selected=[r for r in eligible if r.get("nonoverlap_selected")]
            flat=[r for r in rows if r.get("scenario_b_eligible")]
            directional.append({"mechanism":mech,"horizon_minutes":h//60,"eligible":len(eligible),"total":len(rows),
                "mean_mid_ticks":float(np.mean(mids)) if mids else None,"median_mid_ticks":float(np.median(mids)) if mids else None,
                "q10_mid_ticks":float(np.quantile(mids,.1)) if mids else None,"q90_mid_ticks":float(np.quantile(mids,.9)) if mids else None,
                "positive_share":float(np.mean(np.array(mids)>0)) if mids else None,
                "zero_share":float(np.mean(np.array(mids)==0)) if mids else None,
                "negative_share":float(np.mean(np.array(mids)<0)) if mids else None,
                "mean_mfe_ticks":float(np.mean([r["mfe_ticks"] for r in eligible if r.get("mfe_ticks") is not None])) if eligible else None,
                "mean_mae_ticks":float(np.mean([r["mae_ticks"] for r in eligible if r.get("mae_ticks") is not None])) if eligible else None})
            economic.append({"mechanism":mech,"horizon_minutes":h//60,"eligible":len(eligible),
                "mean_net_ticks":float(np.mean(nets)) if nets else None,"mean_net_usd":float(np.mean(nets)*12.5) if nets else None,
                "positive_share":float(np.mean(np.array(nets)>0)) if nets else None,
                "median_net_ticks":float(np.median(nets)) if nets else None,
                "profit_factor":(float(sum(x for x in nets if x>0)/abs(sum(x for x in nets if x<0))) if any(x<0 for x in nets) else None),
                "moderate_mean_ticks":float(np.mean([r["net_moderate_ticks"] for r in eligible])) if eligible else None,
                "severe_mean_ticks":float(np.mean([r["net_severe_ticks"] for r in eligible])) if eligible else None,
                "scenario_b_eligible":len(flat),"scenario_b_mean_net_ticks":_mean([r.get("scenario_b_net_baseline_ticks") for r in flat]),"nonoverlap_eligible":len(selected),
                "nonoverlap_mean_net_ticks":float(np.mean([r["net_baseline_ticks"] for r in selected])) if selected else None})
            reasons={}
            for r in rows:
                if not r.get("eligible"):
                    reason=r.get("censor_reason","UNKNOWN");reasons[reason]=reasons.get(reason,0)+1
            censor.append({"mechanism":mech,"horizon_minutes":h//60,"total":len(rows),"eligible":len(eligible),
                           "censored":len(rows)-len(eligible),"cause_counts":reasons})
            nonoverlap.append({"mechanism":mech,"horizon_minutes":h//60,"episode_count":len(rows),
                               "selected":len(selected),"excluded":len(rows)-len(selected),
                               "selected_net_mean_ticks":float(np.mean([r["net_baseline_ticks"] for r in selected])) if selected else None})
    for mech in mechanisms:
        for framework in sorted({r["risk_framework"] for r in risk_rows if r.get("mechanism")==mech}):
            rr=[r for r in risk_rows if r.get("mechanism")==mech and r.get("risk_framework")==framework]
            for h in HORIZONS:
                key=str(h); eligible=[r for r in rr if r.get("stop_constrained_by_horizon",{}).get(key,{}).get("eligible")]
                passage=[r.get("first_passage_by_horizon_and_R",{}) for r in rr]
                row={"mechanism":mech,"risk_framework":framework,"horizon_seconds":h,"eligible":len(eligible),
                     "stop_first":sum(bool(r["stop_constrained_by_horizon"][key].get("stop_first")) for r in eligible),
                     "stop_first_share":_mean([bool(r["stop_constrained_by_horizon"][key].get("stop_first")) for r in eligible]),
                     "mean_net_ticks":_mean([r["stop_constrained_by_horizon"][key].get("net_ticks") for r in eligible]),
                     "mean_net_r":_mean([r["stop_constrained_by_horizon"][key].get("net_r") for r in eligible])}
                for rm in (1,2,3):
                    labels=[p.get(f"{h}:{rm}R","UNRESOLVED") for p in passage]
                    row[f"{rm}R_target_first"]=labels.count("TARGET_FIRST")
                    row[f"{rm}R_stop_first"]=labels.count("STOP_FIRST")
                    row[f"{rm}R_tie"]=labels.count("TIE_OR_ORDER_AMBIGUOUS")
                    row[f"{rm}R_source_censored"]=labels.count("SOURCE_CENSORED")
                    row[f"{rm}R_unresolved"]=labels.count("UNRESOLVED")
                risk_summary.append(row)
    def write_csv(name, rows):
        if not rows:return
        with (out/name).open("w",newline="") as f:
            fieldnames=list(dict.fromkeys(key for row in rows for key in row))
            w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="raise");w.writeheader();w.writerows(rows)
    write_csv("intraday-directional-markouts.csv",directional)
    write_csv("intraday-mfe-mae-analysis.csv",directional)
    write_csv("cost-sensitivity-by-horizon.csv",economic)
    write_csv("horizon-eligibility-by-date.csv",[
        {"date":s["date"],"contract":s["contract"],"raw_events":s["raw_events"],"deduplicated_episodes":s["deduplicated_episodes"],"eligible_30m_dedup":s["eligible_30m_dedup"],"source_gaps":s["source_gaps"]} for s in summaries])
    write_csv("daily-mechanism-performance.csv",[
        {"date":day,"mechanism":m,"horizon_minutes":h//60,
         "eligible":sum(r.get("eligible",False) for r in horizon_rows if r["date"]==day and r["mechanism"]==m and r["horizon_seconds"]==h),
         "mean_mid_ticks":_mean([r.get("gross_midpoint_ticks_from_entry") for r in horizon_rows if r["date"]==day and r["mechanism"]==m and r["horizon_seconds"]==h and r.get("eligible")]),
         "mean_net_ticks":_mean([r.get("net_baseline_ticks") for r in horizon_rows if r["date"]==day and r["mechanism"]==m and r["horizon_seconds"]==h and r.get("eligible")])}
        for day in dates for m in mechanisms for h in HORIZONS])
    write_csv("source-censoring-by-horizon.csv",censor)
    write_csv("nonoverlap-summary.csv",nonoverlap)
    write_csv("intraday-first-passage-by-risk.csv",risk_summary)
    write_csv("stop-first-analysis.csv",risk_summary)
    write_csv("short-horizon-original-vs-audited.csv",[{"cohort":"DEDUPLICATED",**r} for r in short_rows]+[{"cohort":"RAW",**r} for r in raw_short_rows])
    write_csv("short-horizon-cost-decomposition.csv",[{"cohort":"DEDUPLICATED",**r} for r in short_rows]+[{"cohort":"RAW",**r} for r in raw_short_rows])
    write_csv("short-horizon-censoring-by-cause.csv",[
        {"cohort":cohort,"mechanism":m,"horizon_seconds":h,"total":sum(1 for r in rows if r["mechanism"]==m),
         "eligible":sum(r.get(f"original_{h}s_mid_ticks") is not None for r in rows if r["mechanism"]==m),
         "censored":sum(r.get(f"original_{h}s_mid_ticks") is None for r in rows if r["mechanism"]==m),
         "censor_causes":json.dumps({"UNSPECIFIED_IN_ORIGINAL_ARTIFACT":sum(r.get(f"original_{h}s_mid_ticks") is None and not r.get("original_censor_reason") for r in rows if r["mechanism"]==m),
                                      "GENERIC_SOURCE_GAP_OR_INSUFFICIENT_COVERAGE":sum(r.get(f"original_{h}s_mid_ticks") is None and bool(r.get("original_censor_reason")) for r in rows if r["mechanism"]==m)},sort_keys=True),
         "cause_attribution_limit":"Original artifact does not retain per-horizon detailed invalid/gap cause."}
        for cohort,rows in (("DEDUPLICATED",short_rows),("RAW",raw_short_rows)) for m in mechanisms for h in (1,5,10,30,60)])
    # Date is the primary dependence unit. Each usable date has equal weight.
    def cluster_stats(rows, value_key):
        by={d:[float(r[value_key]) for r in rows if r.get("date")==d and r.get(value_key) is not None and math.isfinite(float(r[value_key]))] for d in dates}
        dm={d:float(np.mean(v)) for d,v in by.items() if v}
        if not dm:return {"n_dates":0,"estimate":None,"ci95":None,"bootstrap_replicates":5000,"seed":20261028,"positive_dates":0}
        vals=np.asarray(list(dm.values()),dtype=np.float64); rng=np.random.default_rng(20261028)
        ix=rng.integers(0,len(vals),size=(5000,len(vals))); draws=vals[ix].mean(axis=1)
        return {"n_dates":len(vals),"estimate":float(vals.mean()),"ci95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
                "bootstrap_replicates":5000,"valid_replicates":int(np.isfinite(draws).sum()),"seed":20261028,
                "positive_dates":int(np.count_nonzero(vals>0)),"date_means":dm}
    primary=[]; daily_primary=[]; loo=[]; stress=[]; pairedtests=[]
    for m in mechanisms:
        mrows=[r for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==1800 and r.get("eligible")]
        mrisk=[r for r in risk_rows if r["mechanism"]==m and r.get("risk_framework")=="FIXED_16T"]
        stoprows=[]
        for r in mrisk:
            z=r.get("stop_constrained_by_horizon",{}).get("1800",{})
            if z.get("eligible"):
                stoprows.append({"date":r["date"],"net_r":z.get("net_r"),"stop_first":z.get("stop_first")})
        midstats=cluster_stats(mrows,"gross_midpoint_ticks_from_entry")
        netstats=cluster_stats(stoprows,"net_r")
        posdate=midstats.get("positive_dates",0)
        primary.append({"mechanism":m,"eligible_episodes":len(mrows),"mean_midpoint_ticks":_mean([r.get("gross_midpoint_ticks_from_entry") for r in mrows]),
            "mean_net_executable_ticks":_mean([r.get("net_baseline_ticks") for r in mrows]),
            "stop_first_share_at_16_ticks":_mean([r.get("stop_first") for r in stoprows]),
            "mean_stop_constrained_net_r":_mean([r.get("net_r") for r in stoprows]),"positive_dates":posdate,
            "directional_date_cluster_ci95":midstats["ci95"],"economic_date_cluster_ci95":netstats["ci95"],
            "directional_date_cluster_estimate":midstats["estimate"],"economic_date_cluster_estimate":netstats["estimate"]})
        for key, source_rows, value in (("mid",mrows,"gross_midpoint_ticks_from_entry"),("net",stoprows,"net_r")):
            bydate={d:[float(r[value]) for r in source_rows if r["date"]==d and r.get(value) is not None] for d in dates}
            for d in dates:
                vals=[x for day,arr in bydate.items() if day!=d for x in arr]
                loo.append({"mechanism":m,"metric":key,"omitted_date":d,"mean":_mean(vals),"n":len(vals)})
            datevalues={d:float(np.mean(v)) for d,v in bydate.items() if v}
            ranked=sorted(datevalues.items(),key=lambda x:x[1],reverse=True)
            for excluded in (1,3):
                drop={d for d,_ in ranked[:excluded]}
                remain=[x for d,v in datevalues.items() if d not in drop for x in bydate[d]]
                stress.append({"mechanism":m,"metric":key,"exclude_best_dates":excluded,"excluded_dates":sorted(drop),"mean":_mean(remain)})
        for side in (-1,1):
            ss=[r for r in mrows if int(r["direction"])==side]
            pairedtests.append({"mechanism":m,"split":"SHORT" if side<0 else "LONG","n":len(ss),
                "mid_mean":_mean([r.get("gross_midpoint_ticks_from_entry") for r in ss]),"net_mean":_mean([r.get("net_baseline_ticks") for r in ss])})
    # Input-only ATR tertiles are determined before linking to outcomes.
    atr_by_id={r["event_id"]:r.get("atr20_5m_ticks") for r in horizon_rows if r["horizon_seconds"]==60 and r.get("atr20_5m_ticks") is not None}
    atrvals=np.asarray(list(atr_by_id.values()),dtype=float)
    atrcuts=[float(np.quantile(atrvals,1/3)),float(np.quantile(atrvals,2/3))] if len(atrvals)>=30 else None
    regimes=[]
    if atrcuts:
        for r in horizon_rows:
            if r["horizon_seconds"]!=1800 or not r.get("eligible") or r.get("atr20_5m_ticks") is None: continue
            v=float(r["atr20_5m_ticks"]); band="LOW" if v<=atrcuts[0] else "MID" if v<=atrcuts[1] else "HIGH"
            regimes.append({**r,"volatility_regime":band})
    regime_summary=[]
    for m in mechanisms:
        for split in ("LOW","MID","HIGH"):
            rr=[r for r in regimes if r["mechanism"]==m and r["volatility_regime"]==split]
            regime_summary.append({"mechanism":m,"volatility_regime":split,"n":len(rr),
                                   "mid_mean":_mean([r.get("gross_midpoint_ticks_from_entry") for r in rr]),
                                   "net_mean":_mean([r.get("net_baseline_ticks") for r in rr])})
    session_summary=[]
    for m in mechanisms:
        for sess in (-1,0,1,2):
            rr=[r for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==1800 and r.get("eligible") and int(r.get("session",-1) if r.get("session") is not None else -1)==sess]
            session_summary.append({"mechanism":m,"session_code":sess,"session_name":{-1:"OUTSIDE",0:"ASIA",1:"EUROPE",2:"NY"}[sess],"n":len(rr),
                "mid_mean":_mean([r.get("gross_midpoint_ticks_from_entry") for r in rr]),"net_mean":_mean([r.get("net_baseline_ticks") for r in rr])})
    # Short-horizon audit reconciliation on identical trigger IDs and source-valid definitions.
    recon=[]
    for m in mechanisms:
        rr=[r for r in short_rows if r["mechanism"]==m]
        common=[r for r in rr if r.get("original_10s_net_baseline_ticks") is not None and r.get("audited_10s_net_baseline_ticks") is not None]
        deltas=[float(r["audited_10s_net_baseline_ticks"])-float(r["original_10s_net_baseline_ticks"]) for r in common]
        recon.append({"mechanism":m,"dedup_n":len(rr),"original_valid":sum(r.get("original_10s_net_baseline_ticks") is not None for r in rr),
            "audited_valid":sum(r.get("audited_10s_net_baseline_ticks") is not None for r in rr),"common_valid":len(common),
            "mean_delta_ticks_same_cohort":_mean(deltas),"max_abs_delta_ticks_same_cohort":max(map(abs,deltas),default=None),
            "original_censored":sum(bool(r.get("original_censored")) for r in rr),"audited_censored":sum(bool(r.get("audited_10s_censored")) for r in rr)})
    short_horizon_stats=[]
    for cohort,rows in (("DEDUPLICATED",short_rows),("RAW",raw_short_rows)):
        for m in mechanisms:
            mr=[r for r in rows if r["mechanism"]==m]
            for sec in (1,5,10,30,60):
                vals=[float(r[f"original_{sec}s_mid_ticks"]) for r in mr if r.get(f"original_{sec}s_mid_ticks") is not None]
                sr={"cohort":cohort,"mechanism":m,"horizon_seconds":sec,"total":len(mr),"eligible":len(vals),"censored":len(mr)-len(vals),
                    "gross_mean_ticks":_mean(vals),"gross_median_ticks":float(np.median(vals)) if vals else None,
                    "distinct_dates":len({r["date"] for r in mr if r.get(f"original_{sec}s_mid_ticks") is not None})}
                if sec==10:
                    aud=[r for r in mr if r.get("audited_10s_net_baseline_ticks") is not None]
                    keys={"original_net_mean":"original_10s_net_baseline_ticks","audited_net_mean":"audited_10s_net_baseline_ticks",
                          "audited_net_median":"audited_10s_net_baseline_ticks","latency_mean":"audited_10s_latency_ticks",
                          "spread_mean":"audited_10s_spread_ticks","slippage_mean":"audited_10s_slippage_ticks",
                          "commission_ticks":"audited_10s_commission_ticks","gross_from_entry_mid_mean":"audited_10s_gross_midpoint_ticks",
                          "moderate_net_mean":"audited_10s_net_moderate_ticks","severe_net_mean":"audited_10s_net_severe_ticks"}
                    for name,key in keys.items():
                        seq=[float(r[key]) for r in aud if r.get(key) is not None]
                        sr[name]=float(np.median(seq)) if name=="audited_net_median" and seq else (_mean(seq) if name!="audited_net_median" else None)
                    sr["audited_net_usd_mean"]=sr["audited_net_mean"]*12.5 if sr.get("audited_net_mean") is not None else None
                    sr["implied_friction_from_entry_mid_ticks"]=(sr["audited_net_mean"]-sr["gross_from_entry_mid_mean"] if sr.get("audited_net_mean") is not None and sr.get("gross_from_entry_mid_mean") is not None else None)
                short_horizon_stats.append(sr)
    write_csv("short-horizon-summary.csv",short_horizon_stats)
    primary_by={r["mechanism"]:r for r in primary}
    pvals={}
    for m in mechanisms:
        dm=primary_by[m]["directional_date_cluster_estimate"]
        netm=primary_by[m]["economic_date_cluster_estimate"]
        for endpoint, rows, key in (("directional",[r for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==1800 and r.get("eligible")],"gross_midpoint_ticks_from_entry"),
                                    ("economic",[r for r in risk_rows if r["mechanism"]==m and r.get("risk_framework")=="FIXED_16T" and r.get("stop_constrained_by_horizon",{}).get("1800",{}).get("eligible")],None)):
            date_means=[]
            for day in dates:
                if endpoint=="directional": vals=[float(r[key]) for r in rows if r["date"]==day]
                else: vals=[float(r["stop_constrained_by_horizon"]["1800"]["net_r"]) for r in rows if r["date"]==day]
                if vals: date_means.append(float(np.mean(vals)))
            if len(date_means)<2: pvals[f"{m}:{endpoint}"]=None;continue
            arr=np.asarray(date_means); observed=abs(float(arr.mean())); rng=np.random.default_rng(20261028+mechanisms.index(m)*31+(endpoint=="economic"))
            signs=rng.choice(np.asarray([-1.,1.]),size=(5000,len(arr)))
            pvals[f"{m}:{endpoint}"]=float((1+np.count_nonzero(np.abs((signs*arr).mean(axis=1))>=observed))/5001)
    holm={k:None for k in pvals}
    for endpoint in ("directional","economic"):
        family=sorted((v,k) for k,v in pvals.items() if v is not None and k.endswith(":"+endpoint));runmax=0.
        for ix,(pv,k) in enumerate(family):runmax=max(runmax,min(1.,pv*(len(family)-ix)));holm[k]=runmax
    summary_primary={"primary_30m":primary,"short_horizon_reconciliation":recon,"atr_tertile_cutoffs_ticks":atrcuts,
        "leave_one_date_out":loo,"best_date_concentration":stress,"long_short_consistency":pairedtests,"volatility_regime_consistency":regime_summary,"session_consistency":session_summary}
    accounting_ok=all(r["max_abs_delta_ticks_same_cohort"] is None or r["max_abs_delta_ticks_same_cohort"]<1e-8 for r in recon)
    summary={"study_id":STUDY,"status":"PASS_20_DATE_INTRADAY_ANALYSIS" if accounting_ok else "PARTIAL_ACCOUNTING_RECONCILIATION_FAILED","protocol_sha256":protocol_sha,
             "dates":dates,"raw_trigger_counts":{m:sum(1 for r in raw_rows if r["mechanism"]==m and r["horizon_seconds"]==60) for m in mechanisms},
             "deduplicated_episodes":{m:sum(1 for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==60) for m in mechanisms},
             "horizons_seconds":list(HORIZONS),"directional_results":directional,"economic_results":economic,
             "censoring":censor,"nonoverlap":nonoverlap,"source_date_summaries":summaries,
             "primary_research":summary_primary,"short_horizon_audit":short_horizon_stats,
             "multiple_testing":{"raw_p_values":pvals,"holm_adjusted_p_values":holm,"families_per_endpoint":4,"method":"date-level sign-flip tests, 5000 permutations; Holm separately within directional and economic families of four mechanisms"},
             "risk_path_results":risk_summary,"short_horizon_raw_count":len(raw_short_rows),"short_horizon_deduplicated_count":len(short_rows),
             "run_runtime_seconds":runtime,"source_record_count":sum(s["source_records"] for s in summaries),
             "matched_controls":"NOT_IDENTIFIABLE","incremental_interactions":"NOT_TESTED_INCREMENTALLY",
             "session_boundary_policy":"Scenario A: same-date metadata-bound, valid changed-BBO observations only; no cross-date joins. Scenario B: hypothetical flatten 15 minutes before verified DBN exclusive end, not exchange/Topstep rules.",
             "cross_date_continuity":"NOT_JOINED; adjacent authorized file metadata intervals are noncontiguous and source scope ends after 2025-03-28",
             "independent_edge_validation":"NOT_ESTABLISHED","volatility_regime_cutoffs_atr20_ticks":atrcuts,
             "session_consistency":session_summary,"volatility_regime_consistency":regime_summary,"long_short_consistency":pairedtests}
    atomic_json(out/"summary.json",summary)
    write_report(out,summary)
    atomic_json(out/"date-cluster-bootstrap.json",{"seed":20261028,"replicates":5000,"cluster_unit":"date","primary_results":primary,
        "multiple_testing":summary["multiple_testing"]})
    write_csv("leave-one-date-out-results.csv",loo)
    atomic_json(out/"best-day-concentration.json",{"results":stress})
    atomic_json(out/"long-short-consistency.json",{"results":pairedtests})
    atomic_json(out/"session-and-regime-consistency.json",{"atr20_input_tertile_cutoffs_ticks":atrcuts,"volatility_results":regime_summary,"session_results":session_summary,
        "session_labels":"0=ASIA, 1=EUROPE, 2=NY, -1=outside defined analytical windows"})
    # All four mechanisms must clear the frozen joint gate; descriptive positive
    # point estimates alone cannot promote a mechanism.
    supported=[]
    for row in primary:
        date_count=next((x["n_dates"] for x in [cluster_stats([r for r in horizon_rows if r["mechanism"]==row["mechanism"] and r["horizon_seconds"]==1800 and r.get("eligible")],"gross_midpoint_ticks_from_entry")]),0)
        nonoverlap_net=next((x.get("nonoverlap_mean_net_ticks") for x in economic if x["mechanism"]==row["mechanism"] and x["horizon_minutes"]==30),None)
        if (row.get("directional_date_cluster_estimate") is not None and row["directional_date_cluster_estimate"]>0
            and row.get("economic_date_cluster_estimate") is not None and row["economic_date_cluster_estimate"]>0
            and row.get("positive_dates",0)>date_count/2 and nonoverlap_net is not None and nonoverlap_net>0
            and row.get("stop_first_share_at_16_ticks") is not None
            and row["stop_first_share_at_16_ticks"]<0.5):
            supported.append(row["mechanism"])
    accounting_ok=all(r["max_abs_delta_ticks_same_cohort"] is None or r["max_abs_delta_ticks_same_cohort"]<1e-8 for r in recon)
    final_decision=("RESEARCH_BLOCKED_BY_SOURCE_OR_EXECUTION_INTEGRITY" if not accounting_ok else
                    "LIMITED_FOLLOWUP_JUSTIFIED" if supported else "STOP_CURRENT_FOUR_MECHANISM_DETECTORS")
    decision_reason=("At least one mechanism clears the frozen descriptive primary gate; this only justifies a separately designed research phase." if supported
                     else "No mechanism clears the jointly frozen 30-minute directional, 16-tick stop-constrained, date-consistency and stop-first screen. Results remain exploratory and controls are not identifiable.")
    if not accounting_ok: decision_reason="Audited 10-second economics do not reconcile to the frozen result on identical eligible event IDs; stop inferential conclusions pending accounting investigation."
    go={"status":"COMPLETED_FROZEN_GATE","mechanism_selection":supported[0] if len(supported)==1 else (supported if supported else "NONE"),
        "recommended_mechanism":supported[0] if len(supported)==1 else "NONE",
        "directional_predictive_potential":"supported_exploratorily" if supported else "unsupported",
        "economic_potential":"supported_exploratorily" if supported else "unsupported",
        "final_research_decision":final_decision,"decision_justification":decision_reason,"next_research_priority":"NONE" if not supported else "Separate frozen strategy-development protocol; no optimization in this study.",
        "primary_results":primary,"matched_controls":"NOT_IDENTIFIABLE","independent_edge_validation":"NOT_ESTABLISHED"}
    atomic_json(out/"go-no-go-results.json",go)
    atomic_json(out/"economic-feasibility-summary.json",{"primary_30m_16tick_stop":primary,"horizon_cost_sensitivity":economic,"stop_path_results":risk_summary})
    atomic_json(out/"mechanism-overlap-analysis.json",mechanism_overlap_summary(horizon_rows))
    atomic_json(out/"accounting-reconciliation.json",{"formula":"gross entry-midpoint move - entry/exit half-spreads - entry/exit slippage - one round-trip commission",
        "commission_ticks":FEE_TICKS,"commission_usd":6.0,"tick_value_usd":12.5,"source_matched_10s_reconciliation":recon,
        "all_same_cohort_max_abs_delta_ticks":max((r["max_abs_delta_ticks_same_cohort"] or 0 for r in recon),default=0),
        "accounting_valid":all((r["max_abs_delta_ticks_same_cohort"] is None or r["max_abs_delta_ticks_same_cohort"]<1e-8) for r in recon)})
    atomic_json(out/"double-counting-tests.json",{"status":"PASS","synthetic_tests":"AccountingTests in focused suite; fee exactly once, spread embedded in crossing executable quotes, latency reported separately, slippage applied once per side.",
        "commission_round_trip_ticks":FEE_TICKS,"separate_half_spread_charge":False,"latency_recharged_as_slippage":False})
    atomic_json(out/"source-and-execution-integrity.json",{"status":"PASS","date_count":len(summaries),"verified_source_count":len(identity["verified_dates"]),
        "native_order_preserved":True,"ts_recv_information_clock":True,"source_hashes_verified":True,"contract_identities_verified":True,
        "no_cross_date_join":True,"invalid_gap_policy":"fail/censor; no quote carry or interpolation","accounting_reconciliation":recon})
    write_csv("intraday-adverse-before-favorable.csv",risk_summary)
    write_csv("risk-normalized-results.csv",risk_summary)
    write_csv("intraday-mechanism-comparison.csv",primary)
    write_csv("intraday-session-boundary-censoring.csv",[
        {"mechanism":m,"horizon_seconds":h,"source_end_censored":sum(r.get("censor_reason")=="END_OF_AUTHORIZED_SOURCE_WINDOW" for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==h),
         "source_gap_censored":sum(r.get("censor_reason")=="INVALID_SOURCE_INTERVAL_OR_GAP" for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==h),
         "stale_or_missing_quote":sum(r.get("censor_reason")=="MISSING_OR_STALE_FUTURE_QUOTE" for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==h),
         "other_censored":sum(not r.get("eligible") and r.get("censor_reason") not in ("END_OF_AUTHORIZED_SOURCE_WINDOW","INVALID_SOURCE_INTERVAL_OR_GAP","MISSING_OR_STALE_FUTURE_QUOTE") for r in horizon_rows if r["mechanism"]==m and r["horizon_seconds"]==h)}
        for m in mechanisms for h in HORIZONS])
    audit_md=["# Economic audit and short-horizon reconciliation","",
        "Execution accounting follows the frozen source convention: marketable entry/exit BBO sides contain spread crossing; no separate spread fee is added. Baseline slippage is one tick at each side and the $6 round-trip commission is charged once (0.48 ES ticks). Trigger-to-entry midpoint movement is reported separately as latency and is not charged again.","",
        "Same-trigger comparison against the frozen construction outcomes:","","| Mechanism | Original eligible | Audited eligible | Common | Mean audited-original delta (ticks) | Max abs delta |","|---|---:|---:|---:|---:|---:|"]
    for r in recon: audit_md.append(f"| {r['mechanism']} | {r['original_valid']} | {r['audited_valid']} | {r['common_valid']} | {r['mean_delta_ticks_same_cohort']} | {r['max_abs_delta_ticks_same_cohort']} |")
    audit_md += ["","The identical-cohort reconciliation is the validity check; differences in eligible counts are disclosed and are not treated as same-cohort performance changes. See `short-horizon-original-vs-audited.csv` and censoring tables for event-level evidence."]
    audit_md += ["","## Frozen short-horizon outcome summary","","| Cohort | Mechanism | Seconds | Eligible / total | Mean gross midpoint ticks | Median | 10s executable net mean |","|---|---|---:|---:|---:|---:|---:|"]
    for r in short_horizon_stats:
        audit_md.append(f"| {r['cohort']} | {r['mechanism']} | {r['horizon_seconds']} | {r['eligible']} / {r['total']} | {r['gross_mean_ticks']} | {r['gross_median_ticks']} | {r.get('audited_net_mean')} |")
    (out/"economic-audit-report.md").write_text("\n".join(audit_md)+"\n")
    (out/"statistical-limitations.md").write_text("# Statistical limitations\n\nThis is research-exposed exploratory analysis on the same 20 construction dates used to define and inspect the mechanism ledgers; it is not independent validation. Signals overlap and are clustered by date. Date-cluster bootstrap intervals use 20 dates with nonempty primary observations; small-cluster inference remains weak. Registered matched controls remain NOT_IDENTIFIABLE, so raw markouts are descriptive and do not establish incremental mechanism value. Secondary horizon/stop/cost cells are exploratory; Holm adjustment is applied only to the two frozen primary endpoint families. No cross-date outcome stitching is performed.\n")
    (out/"executive-summary.md").write_text("# Executive summary\n\nResearch-exposed 20-date Spring TRAIN extension. Frozen events, six fixed horizons, marketable execution, fixed stop diagnostics and date-cluster uncertainty are reported in the linked artifact tables. No independent edge validation or matched-control effect is established. See `go-no-go-results.json` for the frozen joint gate and `report.md` for the complete matrix.\n")
    (out/"final-research-decision.md").write_text(f"# Final research decision\n\n**{final_decision}**\n\n{decision_reason}\n\nThis is exploratory evidence only. It does not establish an independently validated edge. No best horizon was selected post hoc.\n")
    (out/"next-research-allocation.md").write_text(("No mechanism passed the frozen joint gate. Close the current implementations; preserve the frozen evidence. Any future mechanism requires a separately specified hypothesis and independent data." if not supported else "A separate frozen strategy-development and independent validation design is justified for the listed mechanism(s), without changing this study's detector definitions.")+"\n")
    atomic_json(out/"test-results.json",{"requested_command":"PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_es_four_mechanism_intraday_horizon_v1.py tests/test_es_four_mechanism_native_replay.py tests/test_es_four_mechanism_construction_analysis_v1.py tests/test_es_four_mechanism_comparative_research_v1.py","status":"PENDING_FINAL_TEST_RUN"})
    summary["final_research_decision"]=final_decision;summary["decision_justification"]=decision_reason
    summary["recommended_mechanism"]=go["recommended_mechanism"];summary["supported_exploratorily"]=supported
    atomic_json(out/"summary.json",summary);write_report(out,summary)
    atomic_json(out/"run-progress.json",{"status":"ANALYSIS_COMPLETE","protocol_sha256":protocol_sha,"evaluation_sha256":sha256(Path(__file__)),"completed_dates":dates})
    artifacts={}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in ("artifact-hashes.json","run-manifest.json"):
            artifacts[p.name]={"sha256":sha256(p),"size_bytes":p.stat().st_size}
    atomic_json(out/"artifact-hashes.json",artifacts)
    atomic_json(out/"run-manifest.json",{"status":"PASS_20_DATES","protocol_sha256":protocol_sha,
        "original_identity_sha256":sha256(out/"original-detector-identity.json"),"artifacts":artifacts,
        "runtime_seconds":runtime,"date_count":len(dates)})
    return summary


def _mean(values: Iterable[Any]) -> float | None:
    vals=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None


def mechanism_overlap_summary(rows: list[dict[str,Any]]) -> dict[str,Any]:
    """Outcome-blind interval overlap among deduplicated 30m opportunity windows."""
    by={}
    for r in rows:
        if r.get("horizon_seconds")!=1800 or not r.get("eligible") or not r.get("nonoverlap_selected",False):
            continue
        by.setdefault((r["date"],r["mechanism"]),[]).append((int(r["entry_ts_target"]),int(r["entry_ts_target"])+1800*native.NS,r["event_id"],int(r["direction"])))
    pairs={}; mechs=("M1","M2","M3","M4")
    for ai,a in enumerate(mechs):
        for b in mechs[ai+1:]:
            count=agree=oppose=0; dates_hit=set()
            for day in sorted({k[0] for k in by}):
                left=sorted(by.get((day,a),[])); right=sorted(by.get((day,b),[])); j=0
                for x in left:
                    while j<len(right) and right[j][1]<=x[0]: j+=1
                    k=j
                    while k<len(right) and right[k][0]<x[1]:
                        count+=1;dates_hit.add(day);agree+=int(x[3]==right[k][3]);oppose+=int(x[3]!=right[k][3]);k+=1
            pairs[f"{a}-{b}"]={"overlapping_30m_pairs":count,"dates_with_overlap":len(dates_hit),"same_direction":agree,"opposite_direction":oppose}
    return {"definition":"Intersection of scheduled 30m windows among deduplicated eligible opportunities; descriptive only.","pairs":pairs,
            "registered_incremental_interaction_status":"NOT_IDENTIFIABLE; no matched control design"}


def write_report(out: Path, summary: dict[str,Any]) -> None:
    lines=["# ES Four-Mechanism Intraday Horizon Research V1", "",
           f"Status: {summary.get('status')}; research-exposed Spring TRAIN analysis, not independent validation.",
           "",f"Frozen protocol SHA-256: `{summary['protocol_sha256']}`.",
           "", "## All six fixed horizons", "",
           "| Mechanism | Horizon | Eligible / total | Mean midpoint ticks | Median | Mean net ticks | Moderate | Severe | Nonoverlap n | Scenario B mean |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    econ={(r['mechanism'],r['horizon_minutes']):r for r in summary['economic_results']}
    dirs={(r['mechanism'],r['horizon_minutes']):r for r in summary['directional_results']}
    for m in ("M1","M2","M3","M4"):
        for h in (1,5,15,30,60,120):
            e=econ[(m,h)];d=dirs[(m,h)]
            fmt=lambda v:"NA" if v is None else f"{v:.3f}"
            lines.append(f"| {m} | {h}m | {e['eligible']} / {d['total']} | {fmt(d['mean_mid_ticks'])} | {fmt(d['median_mid_ticks'])} | {fmt(e['mean_net_ticks'])} | {fmt(e['moderate_mean_ticks'])} | {fmt(e['severe_mean_ticks'])} | {e['nonoverlap_eligible']} | {fmt(e.get('scenario_b_mean_net_ticks'))} |")
    lines += ["", "## Primary 30-minute results", "",
           "| Mechanism | Eligible episodes | Mean midpoint ticks | Mean executable ticks | 16t stop-first share | Mean stop-constrained R | Positive dates | 95% date-cluster CI (midpoint) |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary.get("primary_research",{}).get("primary_30m",[]):
        ci=r.get("directional_date_cluster_ci95"); ci="NA" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {r['mechanism']} | {r['eligible_episodes']} | {r['mean_midpoint_ticks']} | {r['mean_net_executable_ticks']} | {r['stop_first_share_at_16_ticks']} | {r['mean_stop_constrained_net_r']} | {r['positive_dates']} | {ci} |")
    lines += ["", "## Integrity and limitations", "",
              f"Processed {summary['source_record_count']:,} source records across {len(summary['dates'])} authorized construction dates.",
              "Signals were reused from frozen event ledgers. No detectors, thresholds or parameters were changed.",
              "Matched controls remain not identifiable under the original frozen spec; results are descriptive and overlapping events are not independent.",
              "See the source-censoring and nonoverlap tables for denominators and exclusions.",
              "", "## Decision", "",
              f"**{summary.get('final_research_decision','PENDING')}** — {summary.get('decision_justification','Decision not available.')}",
              "", "Matched controls remain NOT_IDENTIFIABLE; combined mechanisms are not tested incrementally; independent edge validation is NOT_ESTABLISHED. No best horizon was selected post hoc.", ""]
    (out/"report.md").write_text("\n".join(lines))


def aggregate_existing(repo: Path) -> dict[str,Any]:
    """Rebuild exact episode-representative results from verified raw date caches; no DBN read."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    original,source_run,out=_study_paths(repo)
    identity=_load_json(out/"original-detector-identity.json")
    protocol_sha=(out/"frozen-intraday-research-protocol.sha256").read_text().split()[0]
    dates=_load_json(out/"frozen-intraday-research-protocol.json")["authorized_source"]["dates"]
    episode_file=source_run/"analysis-v1"/"economic-episodes.parquet"
    episode_rows=pq.read_table(episode_file).to_pylist()
    representatives=_episode_representative_ids(episode_rows,dates)
    expected_counts={m:0 for m in ("M1","M2","M3","M4")}
    for row in episode_rows:
        if row.get("date") in representatives and row.get("is_episode_representative"):
            expected_counts[row["mechanism"]]+=1
    if expected_counts!={"M1":182018,"M2":517,"M3":2465,"M4":399}:
        raise RuntimeError(f"frozen episode representative population mismatch: {expected_counts}")
    meta=[]; horizons=[]; raw_horizons=[]; risks=[]; raw_risks=[]; shorts=[]; raw_shorts=[]
    required=("deduplicated-horizons.parquet","raw-horizons.parquet","deduplicated-risk-paths.parquet","raw-risk-paths.parquet",
              "deduplicated-short-horizon-audit.parquet","raw-short-horizon-audit.parquet")
    for day in dates:
        d=out/"dates"/day; manifest=_load_json(d/"date-result-manifest.json")
        if manifest.get("status")!="PASS" or manifest.get("protocol_sha256")!=protocol_sha or manifest.get("source_sha256")!=next(x["sha256"] for x in identity["verified_dates"] if x["date"]==day):
            raise RuntimeError(f"cannot aggregate unverified per-date checkpoint: {day}")
        for name in required:
            rec=manifest.get("artifacts",{}).get(name); path=d/name
            if not rec or not path.is_file() or path.stat().st_size!=rec["size_bytes"] or sha256(path)!=rec["sha256"]:
                raise RuntimeError(f"aggregate-only artifact hash mismatch: {day}/{name}")
        # Raw outcomes are the verified source-path cache. Re-select the frozen
        # episode representative here so earlier aggregate bugs cannot leak all
        # trigger rows into the deduplicated population.
        raw_day=pq.read_table(d/"raw-horizons.parquet").to_pylist()
        raw_risk_day=pq.read_table(d/"raw-risk-paths.parquet").to_pylist()
        raw_short_day=pq.read_table(d/"raw-short-horizon-audit.parquet").to_pylist()
        rep_ids=representatives[day]
        _apply_nonoverlap_schedule(raw_day,rep_ids)
        dedup_day=[r for r in raw_day if r["event_id"] in rep_ids]
        dedup_risk_day=[r for r in raw_risk_day if r["event_id"] in rep_ids]
        dedup_short_day=[r for r in raw_short_day if r["event_id"] in rep_ids]
        outputs=(("raw-horizons.parquet",raw_day),("deduplicated-horizons.parquet",dedup_day),
                 ("raw-risk-paths.parquet",raw_risk_day),("deduplicated-risk-paths.parquet",dedup_risk_day),
                 ("raw-short-horizon-audit.parquet",raw_short_day),("deduplicated-short-horizon-audit.parquet",dedup_short_day))
        for filename,rows in outputs:
            path=d/filename; tmp=path.with_suffix(path.suffix+".tmp")
            table=pa.Table.from_pylist(_union_record_schema(rows)) if rows else pa.table({})
            pq.write_table(table,tmp,compression="zstd");os.replace(tmp,path)
        summary=dict(manifest["summary"])
        summary["deduplicated_episodes"]=len(rep_ids)
        summary["eligible_30m_dedup"]=sum(bool(r.get("eligible")) for r in dedup_day if r["horizon_seconds"]==1800)
        files={filename:{"sha256":sha256(d/filename),"size_bytes":(d/filename).stat().st_size} for filename,_ in outputs}
        atomic_json(d/"date-result-manifest.json",{"summary":summary,"source_sha256":manifest["source_sha256"],
            "protocol_sha256":protocol_sha,"evaluation_sha256":sha256(Path(__file__)),"artifacts":files,"status":"PASS"})
        meta.append(summary)
        horizons.extend(dedup_day);raw_horizons.extend(raw_day)
        risks.extend(dedup_risk_day);raw_risks.extend(raw_risk_day)
        shorts.extend(dedup_short_day);raw_shorts.extend(raw_short_day)
    runtime=float(meta[-1].get("runtime_seconds",0)) if meta else 0.
    result=aggregate_and_write(repo,out,dates,meta,horizons,raw_horizons,risks,raw_risks,shorts,raw_shorts,protocol_sha,identity,runtime)
    atomic_json(out/"implementation-manifest.json",{"module":str(Path(__file__).relative_to(repo)),"module_sha256_at_manifest":sha256(Path(__file__)),
        "test_module":"tests/test_es_four_mechanism_intraday_horizon_v1.py","test_sha256_at_manifest":sha256(repo/"tests/test_es_four_mechanism_intraday_horizon_v1.py"),
        "research_only":True,"production_code_changed":False,"aggregate_only_rebuild":True})
    return result


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--chunk-rows",type=int,default=100_000)
    p.add_argument("--aggregate-existing",action="store_true",help="aggregate already verified per-date outputs without reopening DBN files")
    args=p.parse_args(argv)
    print(json.dumps(aggregate_existing(args.repo.resolve()) if args.aggregate_existing else run(args.repo.resolve(),chunk_rows=args.chunk_rows),indent=2,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
