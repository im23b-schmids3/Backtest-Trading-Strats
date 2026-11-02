"""Controlled, source-bound ES HVN reversal V3 development study.

This module deliberately implements only preregistered causal rules.  Native
DBN order is the tie breaker; ts_recv is the availability clock.  It does not
infer passive order identity or claim queue-aware fills from MBP-10.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

STUDY_ID = "ES_HVN_REVERSAL_V3_CONTROLLED_DEVELOPMENT"
OUT = Path("research_runs") / STUDY_ID
SOURCE_MANIFEST = Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1/source-integrity-results.json")
TICK = 0.25
NS = 1_000_000_000
CHUNK_ROWS = 250_000
FLAG_BAD_TS_RECV = 8
FLAG_MAYBE_BAD_BOOK = 4
UTC = timezone.utc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "EMPTY"}]
    normalized = [{k: (json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list, tuple)) else v)
                   for k, v in row.items()} for row in rows]
    pq.write_table(pa.Table.from_pylist(normalized), path, compression="zstd")


def session_windows(day: str) -> dict[str, tuple[int, int]]:
    """Canonical UTC windows from the established Mac-2025 baseline."""
    from .mac_2025_es_only_train_baseline import _session_windows
    return _session_windows(day)


def trade_aggressor(side: bytes | str) -> str:
    if isinstance(side, bytes):
        side = side.decode("ascii", "replace")
    return {"B": "BUY", "A": "SELL"}.get(str(side), "UNKNOWN")


def valid_quote(bid: int, ask: int, bid_size: int, ask_size: int, flags: int) -> bool:
    return bid > 0 and ask > bid and bid_size > 0 and ask_size > 0 and not (flags & (FLAG_BAD_TS_RECV | FLAG_MAYBE_BAD_BOOK))


def profile_zones(volume_by_tick: dict[int, int], *, date_s: str, session: str, profile_start_ns: int,
                  profile_end_ns: int, prominence_ratio: float = 1.5,
                  minimum_share: float = 0.005) -> list[dict[str, Any]]:
    """Outcome-blind 3-tick smoothed local peaks, returned in stable tick order."""
    if not volume_by_tick:
        return []
    lo, hi = min(volume_by_tick), max(volume_by_tick)
    raw = np.zeros(hi - lo + 1, dtype=np.int64)
    for tick, volume in volume_by_tick.items():
        raw[tick - lo] += int(volume)
    smooth = np.convolve(np.pad(raw, (1, 1)), np.ones(3, dtype=np.int64), mode="valid")
    total = int(raw.sum())
    peaks: list[int] = []
    for i in range(2, len(smooth) - 2):
        left, right = int(smooth[i - 1]), int(smooth[i + 1])
        if smooth[i] < left or smooth[i] < right or (smooth[i] == left and i > 0):
            continue
        neighbor_mean = (int(smooth[i - 2]) + int(smooth[i + 2])) / 2
        if neighbor_mean <= 0 or smooth[i] / neighbor_mean < prominence_ratio:
            continue
        peak_tick = lo + i
        concentration = sum(int(volume_by_tick.get(t, 0)) for t in range(peak_tick - 1, peak_tick + 2))
        if concentration / max(total, 1) >= minimum_share:
            peaks.append(peak_tick)
    # Deterministically collapse peaks no more than two ticks apart.
    clusters: list[list[int]] = []
    for peak in peaks:
        if clusters and peak - clusters[-1][-1] <= 2:
            clusters[-1].append(peak)
        else:
            clusters.append([peak])
    out = []
    for n, cluster in enumerate(clusters):
        chosen = min(cluster, key=lambda t: (-int(smooth[t - lo]), t))
        concentration = sum(int(volume_by_tick.get(t, 0)) for t in range(min(cluster)-1, max(cluster)+2))
        prom = float(smooth[chosen - lo]) / max((int(smooth[chosen - lo - 2]) + int(smooth[chosen - lo + 2])) / 2, 1)
        out.append({"date": date_s, "session": session, "profile_start_ns": profile_start_ns,
                    "profile_end_ns": profile_end_ns, "zone_id": f"{date_s}|{session}|{profile_start_ns}|{n}",
                    "peak_tick": chosen, "poc": chosen * TICK, "low_tick": min(cluster) - 1,
                    "high_tick": max(cluster) + 1, "peak_volume": int(volume_by_tick.get(chosen, 0)),
                    "zone_volume": int(concentration), "profile_volume": total,
                    "volume_share": concentration / max(total, 1), "prominence": prom,
                    "created_ns": profile_end_ns, "activation_ns": profile_end_ns + 1,
                    "expiry_ns": profile_end_ns + 120 * 60 * NS, "interaction_count": 0,
                    "status": "ACTIVE"})
    return out


def merge_overlapping_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping same-profile zones without outcome-dependent choice."""
    result: list[dict[str, Any]] = []
    for z in sorted(zones, key=lambda r: (r["date"], r["session"], r["profile_start_ns"], r["low_tick"])):
        if result and (z["date"], z["session"], z["profile_start_ns"]) == (result[-1]["date"], result[-1]["session"], result[-1]["profile_start_ns"]) and z["low_tick"] <= result[-1]["high_tick"]:
            p = result[-1]
            p["low_tick"] = min(p["low_tick"], z["low_tick"])
            p["high_tick"] = max(p["high_tick"], z["high_tick"])
            p["source_zone_ids"] = sorted(p.get("source_zone_ids", [p["zone_id"]]) + [z["zone_id"]])
            p["zone_id"] = "+".join(p["source_zone_ids"])
            p["prominence"] = max(p["prominence"], z["prominence"])
            denom = p["zone_volume"] + z["zone_volume"]
            p["poc"] = (p["poc"] * p["zone_volume"] + z["poc"] * z["zone_volume"]) / max(denom, 1)
            p["zone_volume"] = denom
        else:
            result.append(dict(z, source_zone_ids=[z["zone_id"]]))
    return result


def attack_metrics(trades: list[dict[str, Any]], attack_side: str, threshold: float,
                   minimum_records: int = 5, minimum_share: float = 0.65) -> dict[str, Any]:
    known = [t for t in trades if t["aggressor"] in ("BUY", "SELL")]
    attack = [t for t in known if t["aggressor"] == attack_side]
    volume = sum(int(t["size"]) for t in attack)
    share = len(attack) / len(known) if known else 0.0
    return {"attack_count": len(attack), "known_count": len(known), "attack_volume": volume,
            "known_volume": sum(int(t["size"]) for t in known), "directional_share": share,
            "qualified": len(attack) >= minimum_records and share >= minimum_share and volume >= threshold}


def failed_progress(start_mid: float, attack_side: str, mids: Iterable[float], max_ticks: float = 2.0) -> dict[str, Any]:
    vals = [float(x) for x in mids if math.isfinite(float(x))]
    adverse = max((start_mid - x if attack_side == "SELL" else x - start_mid) / TICK for x in vals) if vals else math.inf
    return {"maximum_attack_progress_ticks": max(0.0, adverse), "failed": adverse <= max_ticks}


def rejection_confirmed(quotes: list[dict[str, Any]], trades: list[dict[str, Any]], *, direction: int,
                        adverse_mid: float, after_ns: int, recovery_ticks: float = 2.0,
                        hold_seconds: float = 1.0, reversal_count: int = 3) -> dict[str, Any] | None:
    threshold = adverse_mid + direction * recovery_ticks * TICK
    trade_times=[int(t["ts_recv"]) for t in trades]
    i=0
    while i < len(quotes):
        q=quotes[i]
        if q["ts_recv"] < after_ns or direction * (q["mid"] - threshold) < -1e-9:
            i+=1
            continue
        until = q["ts_recv"] + int(hold_seconds * NS)
        j=i+1
        failed=False
        while j<len(quotes) and quotes[j]["ts_recv"] < until:
            if direction*(quotes[j]["mid"]-threshold)<-1e-9:
                failed=True; break
            j+=1
        if failed or j>=len(quotes) or quotes[j]["ts_recv"]<until:
            i+=1
            continue
        side = "BUY" if direction > 0 else "SELL"
        tlo=bisect.bisect_left(trade_times,q["ts_recv"])
        thi=bisect.bisect_right(trade_times,until)
        nrev = sum(x["aggressor"] == side for x in trades[tlo:thi])
        if nrev >= reversal_count:
            return {"confirmation_ns": until, "recovery_mid": q["mid"], "reversal_trade_count": nrev,
                    "adverse_mid": adverse_mid, "threshold_mid": threshold}
        i+=1
    return None


def rejection_confirmed_native(ev: np.ndarray, qi: np.ndarray, ti: np.ndarray, qrecv: np.ndarray,
                               trecv: np.ndarray, qmids: np.ndarray, *, direction: int,
                               adverse_mid: float, after_ns: int, expiry_ns: int,
                               recovery_ticks: float=2.0, hold_seconds: float=1.0,
                               reversal_count: int=3) -> dict[str,Any] | None:
    """Array-only rejection test; never expands the zone lifetime into dicts."""
    threshold=adverse_mid+direction*recovery_ticks*TICK
    lo=int(np.searchsorted(qrecv,after_ns,side="left")); hi=int(np.searchsorted(qrecv,expiry_ns,side="right"))
    candidates=lo+np.flatnonzero(direction*(qmids[lo:hi]-threshold)>=-1e-9)
    wanted=b"B" if direction>0 else b"A"
    hold_ns=int(hold_seconds*NS)
    for i in candidates:
        ts=int(qrecv[i]); until=ts+hold_ns
        j=int(np.searchsorted(qrecv,until,side="left"))
        if j>=hi or j<i or np.min(direction*(qmids[i:j+1]-threshold)) < -1e-9: continue
        a=int(np.searchsorted(trecv,ts,side="left")); b=int(np.searchsorted(trecv,until,side="right"))
        side=ev[ti[a:b]]["side"]
        count=int(np.count_nonzero(side==wanted))
        if count>=reversal_count:
            row=ev[qi[i]]
            return {"confirmation_ns":until,"recovery_mid":float(qmids[i]),"reversal_trade_count":count,
                    "adverse_mid":adverse_mid,"threshold_mid":threshold,
                    "confirmation_native_index":int(row["native_index"]),"confirmation_ts_event":int(row["ts_event"]),
                    "confirmation_sequence":int(row["sequence"])}
    return None


def orderflow_rejection_native(ev: np.ndarray, qi: np.ndarray, ti: np.ndarray, qrecv: np.ndarray,
                               trecv: np.ndarray, qmids: np.ndarray, *, direction: int,
                               adverse_mid: float, after_ns: int, expiry_ns: int,
                               attack_volume: int, recovery_ticks: float=2.0,
                               volume_window_seconds: float=5.0) -> dict[str,Any] | None:
    """Frozen R2: recovery plus reversal-side volume >= attack volume in 5s."""
    threshold=adverse_mid+direction*recovery_ticks*TICK
    lo=int(np.searchsorted(qrecv,after_ns,side="left")); hi=int(np.searchsorted(qrecv,expiry_ns,side="right"))
    candidates=lo+np.flatnonzero(direction*(qmids[lo:hi]-threshold)>=-1e-9)
    wanted=b"B" if direction>0 else b"A"
    for i in candidates:
        ts=int(qrecv[i]); end=ts+int(volume_window_seconds*NS)
        a=int(np.searchsorted(trecv,ts,side="left")); b=int(np.searchsorted(trecv,end,side="right"))
        rows=ev[ti[a:b]]; mask=rows["side"]==wanted
        volume=int(rows["size"][mask].sum())
        if volume>=attack_volume:
            row=ev[qi[i]]
            return {"confirmation_ns":ts,"recovery_mid":float(qmids[i]),"reversal_trade_count":int(mask.sum()),
                    "reversal_volume":volume,"adverse_mid":adverse_mid,"threshold_mid":threshold,
                    "confirmation_native_index":int(row["native_index"]),"confirmation_ts_event":int(row["ts_event"]),
                    "confirmation_sequence":int(row["sequence"])}
    return None


def context_features(quotes: list[dict[str, Any]], index: int, window_ns: int = 30 * 60 * NS) -> dict[str, Any]:
    """Trailing-only realized path and directional efficiency; no future bars."""
    q = quotes[max(0, index - 10000):index + 1]
    if not q:
        return {"realized_range_ticks": None, "directional_efficiency": None, "available": False}
    cutoff = q[-1]["ts_recv"] - window_ns
    q = [x for x in q if x["ts_recv"] >= cutoff]
    if len(q) < 2:
        return {"realized_range_ticks": None, "directional_efficiency": None, "available": False}
    mids = [x["mid"] for x in q]
    moves = [abs(b - a) for a, b in zip(mids, mids[1:])]
    return {"realized_range_ticks": (max(mids) - min(mids)) / TICK,
            "directional_efficiency": abs(mids[-1] - mids[0]) / max(sum(moves), 1e-9), "available": True}


def stop_geometry(direction: int, entry: float, zone_low: float, zone_high: float,
                  rejection_extreme: float, buffer_ticks: int = 1) -> tuple[float, int]:
    stop = (min(zone_low, rejection_extreme) - buffer_ticks * TICK if direction > 0
            else max(zone_high, rejection_extreme) + buffer_ticks * TICK)
    ticks = int(round(abs(entry - stop) / TICK))
    if ticks <= 0:
        raise ValueError("non-positive structural stop")
    return stop, ticks


def resolve_config(config: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Expand a registered alternative from its frozen baseline parent."""
    p = dict(baseline["params"])
    if config["id"] != "V3-C00":
        p.update(config.get("params", {}))
    if config["id"] == "V3-C08": p["attack_minimum_trade_records"] = 0
    if config["id"] == "V3-C09": p.update(local_prominence_ratio=0.0, minimum_zone_profile_volume_share=0.0)
    if config["id"] == "V3-C10": p["extreme_volatility_gate"] = None
    if config["id"] == "V3-C11": p["rejection_mode"] = "CANDIDATE_ENTRY"
    return p


def deduplicate_overlapping_signals(signals: list[dict[str,Any]], *, episode_window_ns: int=10*NS) -> tuple[list[dict[str,Any]],set[str]]:
    """Retain the earliest same-direction signal among overlapping zone episodes."""
    kept=[]; suppressed=set()
    for s in sorted(signals,key=lambda x:(x["approach_ns"],x["approach_native_index"],x["zone_id"])):
        match=None
        for prev in reversed(kept):
            if s["approach_ns"]-prev["approach_ns"]>episode_window_ns: break
            if s["direction"]!=prev["direction"]: continue
            if max(s["zone_low"],prev["zone_low"])<=min(s["zone_high"],prev["zone_high"]):
                match=prev; break
        if match is None:
            kept.append(dict(s)); continue
        suppressed.add(s["signal_id"])
        match["zone_low"]=min(match["zone_low"],s["zone_low"])
        match["zone_high"]=max(match["zone_high"],s["zone_high"])
        match["zone_id"]="+".join(sorted(set(match["zone_id"].split("+"))|set(s["zone_id"].split("+"))))
        match["merged_zone_count"]=int(match.get("merged_zone_count",1))+1
    return kept,suppressed


@dataclass
class NativeDay:
    date: str
    source: dict[str, Any]
    events: np.ndarray
    event_count: int
    invalid_bbo_count: int
    unknown_aggressor_count: int
    ts_recv_backsteps: int


EVENT_DTYPE = np.dtype([
    ("native_index", "<i8"), ("ts_recv", "<i8"), ("ts_event", "<i8"), ("sequence", "<u4"),
    ("action", "S1"), ("side", "S1"), ("price", "<i8"), ("size", "<i8"), ("flags", "u1"),
    ("bid", "<i8"), ("ask", "<i8"), ("bid_size", "<i8"), ("ask_size", "<i8"),
    ("bid_top10", "<i8"), ("ask_top10", "<i8"),
])


def extract_native_day(repo: Path, source: dict[str, Any], cache: Path) -> NativeDay:
    """Bounded-memory source scan; cache only trade and BBO-state-change rows."""
    from databento import DBNStore
    path = repo / source["raw_path"]
    if path.stat().st_size != source["raw_bytes"] or sha256(path) != source["raw_sha256"]:
        raise ValueError(f"source hash/size mismatch: {source['date']}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta = cache.with_suffix(".json")
    if cache.is_file() and meta.is_file():
        m = json.loads(meta.read_text())
        dtype_json=json.loads(json.dumps(EVENT_DTYPE.descr))
        if m.get("source_sha256") == source["raw_sha256"] and m.get("event_dtype") == dtype_json:
            arr = np.load(cache, mmap_mode="r", allow_pickle=False)
            if arr.dtype == EVENT_DTYPE:
                return NativeDay(source["date"], source, arr, int(m["source_event_count"]),
                                 int(m["invalid_bbo_count"]), int(m["unknown_aggressor_count"]), int(m["ts_recv_backsteps"]))
    pieces: list[np.ndarray] = []
    count = invalid = unknown = backsteps = 0
    prior_recv = None
    prior_bbo = None
    seen_valid = False
    offset = 0
    for chunk in DBNStore.from_file(path).to_ndarray(count=CHUNK_ROWS):
        n = len(chunk)
        if not n: continue
        if not np.all(chunk["rtype"] == 10): raise ValueError(f"non-MBP-10 row in {source['date']}")
        recv = chunk["ts_recv"].astype(np.int64, copy=False)
        backsteps += int(np.count_nonzero(recv[1:] < recv[:-1]))
        if prior_recv is not None and int(recv[0]) < prior_recv: backsteps += 1
        prior_recv = int(recv[-1])
        flags = chunk["flags"].astype(np.uint8, copy=False)
        bp, ap = chunk["bid_px_00"].astype(np.int64, copy=False), chunk["ask_px_00"].astype(np.int64, copy=False)
        bs, ass = chunk["bid_sz_00"].astype(np.int64, copy=False), chunk["ask_sz_00"].astype(np.int64, copy=False)
        valid = (bp > 0) & (ap > bp) & (bs > 0) & (ass > 0) & ((flags & (FLAG_BAD_TS_RECV | FLAG_MAYBE_BAD_BOOK)) == 0)
        invalid += int(np.count_nonzero(~valid))
        is_trade = (chunk["action"] == b"T") & (chunk["size"].astype(np.int64, copy=False) > 0)
        side = chunk["side"].astype("S1", copy=False)
        unknown += int(np.count_nonzero(is_trade & (side != b"A") & (side != b"B")))
        bbo_change = np.zeros(n, dtype=bool)
        if prior_bbo is not None:
            bbo_change[0] = valid[0] and tuple(map(int, (bp[0], ap[0], bs[0], ass[0]))) != prior_bbo
        elif np.any(valid):
            bbo_change[int(np.flatnonzero(valid)[0])] = True
            seen_valid=True
        if n > 1:
            bbo_change[1:] = valid[1:] & ((bp[1:] != bp[:-1]) | (ap[1:] != ap[:-1]) | (bs[1:] != bs[:-1]) | (ass[1:] != ass[:-1]))
        if valid[-1]: prior_bbo = tuple(map(int, (bp[-1], ap[-1], bs[-1], ass[-1])))
        keep = is_trade | bbo_change
        ix = np.flatnonzero(keep)
        if len(ix):
            a = np.empty(len(ix), dtype=EVENT_DTYPE)
            a["native_index"] = offset + ix
            a["ts_recv"] = recv[ix]
            a["ts_event"] = chunk["ts_event"][ix]
            a["sequence"] = chunk["sequence"][ix]
            a["action"] = chunk["action"][ix]
            a["side"] = side[ix]
            a["price"] = chunk["price"][ix]
            a["size"] = chunk["size"][ix]
            a["flags"] = flags[ix]
            a["bid"] = bp[ix]; a["ask"] = ap[ix]; a["bid_size"] = bs[ix]; a["ask_size"] = ass[ix]
            a["bid_top10"] = sum(chunk[f"bid_sz_{j:02d}"][ix].astype(np.int64) for j in range(10))
            a["ask_top10"] = sum(chunk[f"ask_sz_{j:02d}"][ix].astype(np.int64) for j in range(10))
            pieces.append(a)
        count += n; offset += n
    arr = np.concatenate(pieces) if pieces else np.empty(0, dtype=EVENT_DTYPE)
    # Native index is already strictly increasing; never sort on timestamps or sequence.
    np.save(cache, arr, allow_pickle=False)
    atomic_json(meta, {"source_sha256": source["raw_sha256"], "source_event_count": count,
                       "invalid_bbo_count": invalid, "unknown_aggressor_count": unknown,
                       "ts_recv_backsteps": backsteps, "event_dtype": EVENT_DTYPE.descr,
                       "retained_rows": len(arr), "retain_policy": "all positive T rows and valid BBO state changes"})
    return NativeDay(source["date"], source, np.load(cache, mmap_mode="r", allow_pickle=False), count, invalid, unknown, backsteps)


def _event_dict(row: np.void) -> dict[str, Any]:
    return {"native_index": int(row["native_index"]), "ts_recv": int(row["ts_recv"]),
            "ts_event": int(row["ts_event"]), "sequence": int(row["sequence"]),
            "action": row["action"].decode(), "side": row["side"].decode(),
            "price": int(row["price"]) / 1e9, "price_ns": int(row["price"]),
            "size": int(row["size"]), "flags": int(row["flags"]),
            "bid": int(row["bid"]) / 1e9, "ask": int(row["ask"]) / 1e9,
            "bid_size": int(row["bid_size"]), "ask_size": int(row["ask_size"]),
            "bid_top10": int(row["bid_top10"]), "ask_top10": int(row["ask_top10"]),
            "mid": (int(row["bid"]) + int(row["ask"])) / 2e9,
            "spread_ticks": (int(row["ask"]) - int(row["bid"])) / (TICK * 1e9),
            "aggressor": trade_aggressor(row["side"]) if row["action"] == b"T" else "NOT_TRADE"}


def build_profiles(day: NativeDay) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = day.events
    zones: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    # Select trade rows once, then group by completed 30-minute profile. This
    # avoids repeatedly scanning the multi-million-row compact day tape.
    trade_ix = np.flatnonzero((events["action"] == b"T") & (events["size"] > 0))
    trade_ts = events["ts_recv"][trade_ix]
    trade_px = events["price"][trade_ix]
    trade_sz = events["size"][trade_ix]
    for session, (lo, hi) in session_windows(day.date).items():
        first = int(np.searchsorted(trade_ts, lo, side="left"))
        last = int(np.searchsorted(trade_ts, hi, side="left"))
        starts = range(0, math.ceil((hi-lo)/(30*60*NS)))
        for group in starts:
            start = lo + group*30*60*NS
            end = min(start + 30*60*NS, hi)
            vols: dict[int, int] = defaultdict(int)
            a=int(np.searchsorted(trade_ts,start,side="left")); b=int(np.searchsorted(trade_ts,end,side="left"))
            px=trade_px[a:b]; sz=trade_sz[a:b]; good=px>0
            ticks=np.rint(px[good].astype(np.float64)/(TICK*1e9)).astype(np.int64)
            if len(ticks):
                u,inv=np.unique(ticks,return_inverse=True)
                quantities=np.bincount(inv,weights=sz[good]).astype(np.int64)
                vols={int(k):int(v) for k,v in zip(u,quantities)}
            new = merge_overlapping_zones(profile_zones(vols, date_s=day.date, session=session,
                profile_start_ns=start, profile_end_ns=end,prominence_ratio=0.0,minimum_share=0.0))
            zones.extend(new)
            diag.append({"date": day.date, "session": session, "start_ns": start, "end_ns": end,
                         "trade_records": b-a, "profile_volume": sum(vols.values()),
                         "distinct_price_ticks": len(vols), "candidate_zones": len(new),
                         "finalized_at_ns": end, "status": "COMPLETE" if end <= hi else "PARTIAL"})
            start = end
    return zones, diag


def build_market_bins(day: NativeDay) -> dict[str, list[tuple[int, float]]]:
    """Completed, session-aligned prior-only effort and volatility bins."""
    ev=day.events
    effort={"BUY":[],"SELL":[]}; ranges=[]
    ti=np.flatnonzero((ev["action"]==b"T")&(ev["size"]>0))
    qi=np.flatnonzero((ev["action"]!=b"T")&(ev["bid"]>0)&(ev["ask"]>ev["bid"])
                      &(ev["bid_size"]>0)&(ev["ask_size"]>0)
                      &((ev["flags"]&(FLAG_BAD_TS_RECV|FLAG_MAYBE_BAD_BOOK))==0))
    tts=ev["ts_recv"][ti]; qts=ev["ts_recv"][qi]
    for session,(lo,hi) in session_windows(day.date).items():
        ta=int(np.searchsorted(tts,lo)); tb=int(np.searchsorted(tts,hi))
        n10=math.ceil((hi-lo)/(10*NS))
        trows=ev[ti[ta:tb]]
        tg=((trows["ts_recv"].astype(np.int64)-lo)//(10*NS)).astype(np.int64)
        tsides=trows["side"]
        tsize=trows["size"].astype(np.float64)
        buyvol=np.bincount(tg[(tsides==b"B")&(tg<n10)],weights=tsize[(tsides==b"B")&(tg<n10)],minlength=n10)
        sellvol=np.bincount(tg[(tsides==b"A")&(tg<n10)],weights=tsize[(tsides==b"A")&(tg<n10)],minlength=n10)
        for group in range(n10):
            end=min(hi,lo+(group+1)*10*NS)
            effort["BUY"].append((end,float(buyvol[group])))
            effort["SELL"].append((end,float(sellvol[group])))
        qa=int(np.searchsorted(qts,lo)); qb=int(np.searchsorted(qts,hi))
        qrows=ev[qi[qa:qb]]
        qgroups=((qrows["ts_recv"].astype(np.int64)-lo)//(30*60*NS)).astype(np.int64)
        qmids=(qrows["bid"].astype(np.float64)+qrows["ask"].astype(np.float64))/(2e9)
        nbin=math.ceil((hi-lo)/(30*60*NS))
        if len(qgroups):
            starts=np.r_[0,np.flatnonzero(qgroups[1:]!=qgroups[:-1])+1]
            ends=np.r_[starts[1:],len(qgroups)]
            for start_i,end_i in zip(starts,ends):
                group=int(qgroups[start_i]); mids=qmids[start_i:end_i]
                ranges.append((min(hi,lo+(group+1)*30*60*NS),(float(np.max(mids))-float(np.min(mids)))/TICK))
    for side in effort: effort[side].sort(key=lambda x:x[0])
    ranges.sort(key=lambda x:x[0])
    return {"effort_BUY":effort["BUY"],"effort_SELL":effort["SELL"],"ranges":ranges}


def generate_signals(day: NativeDay, zones: list[dict[str, Any]], params: dict[str, Any],
                     prior_effort_bins: dict[str, list[float]], config_id: str,
                     market_bins: dict[str,list[tuple[int,float]]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Causal first-interaction funnel and rejection confirmations."""
    ev = day.events
    # Keep the millions of retained native rows in NumPy; materialize small
    # dictionaries only for the time-local episode being evaluated.
    qi = np.flatnonzero((ev["action"] != b"T") & (ev["bid"] > 0) & (ev["ask"] > ev["bid"])
                        & (ev["bid_size"] > 0) & (ev["ask_size"] > 0)
                        & ((ev["flags"] & (FLAG_BAD_TS_RECV | FLAG_MAYBE_BAD_BOOK)) == 0))
    ti = np.flatnonzero((ev["action"] == b"T") & (ev["size"] > 0))
    qrecv = ev["ts_recv"][qi]
    qmids=(ev["bid"][qi].astype(np.float64)+ev["ask"][qi].astype(np.float64))/(2e9)
    trecv = ev["ts_recv"][ti]
    qindex = ev["native_index"][qi]
    tindex = ev["native_index"][ti]
    zones = [dict(z) for z in zones]
    funnel: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    for zone in zones:
        if zone["date"] != day.date:
            continue
        if zone["prominence"] < float(params.get("local_prominence_ratio", 1.5)) or zone["volume_share"] < float(params.get("minimum_zone_profile_volume_share", .005)):
            continue
        if config_id == "V3-C02" or params.get("independent_completed_profile_confirmations"):
            prev = [z for z in zones if z["session"] == zone["session"] and z["profile_end_ns"] < zone["profile_end_ns"]
                    and z["prominence"] >= float(params.get("local_prominence_ratio",1.5))
                    and z["volume_share"] >= float(params.get("minimum_zone_profile_volume_share",.005))
                    and abs(z["peak_tick"] - zone["peak_tick"]) <= int(params.get("overlap_tolerance_ticks",4))]
            if len(prev) < 1: continue
        active_ns = int(zone["activation_ns"]); expiry = int(zone["expiry_ns"])
        low = zone["low_tick"] * TICK; high = zone["high_tick"] * TICK
        qstart = int(np.searchsorted(qrecv, active_ns, side="left"))
        qstop = int(np.searchsorted(qrecv, expiry, side="right"))
        first=max(1,qstart)
        if first>=qstop: continue
        pos=np.arange(first,qstop,dtype=np.int64)
        cur=qmids[pos]; prev=qmids[pos-1]
        long_touch=(prev>high+TICK)&(cur>=low-TICK)&(cur<=high+TICK)
        short_touch=(prev<low-TICK)&(cur>=low-TICK)&(cur<=high+TICK)
        candidates=np.flatnonzero(long_touch|short_touch)
        candidate_i=int(pos[candidates[0]]) if len(candidates) else None
        for i in (() if candidate_i is None else (candidate_i,)):
            q = _event_dict(ev[qi[i]])
            prevq = _event_dict(ev[qi[i-1]])
            mid = q["mid"]; pmid = prevq["mid"]
            if pmid > high + TICK and low - TICK <= mid <= high + TICK: direction = 1
            elif pmid < low - TICK and low - TICK <= mid <= high + TICK: direction = -1
            else: continue
            touch = q["ts_recv"]
            # First interaction retires regardless of outcome.
            zone["interaction_count"] += 1; zone["status"] = "RETIRED_FIRST_TOUCH"
            attack_end = touch + int(float(params["attack_window_seconds"]) * NS)
            attack_side = "SELL" if direction > 0 else "BUY"
            tstart = int(np.searchsorted(trecv, touch, side="left"))
            tstop = int(np.searchsorted(trecv, attack_end, side="right"))
            local_trades = []
            for tii in range(tstart, tstop):
                row = ev[ti[tii]]
                if int(row["price"]) <= 0: continue
                price = int(row["price"])/1e9
                if low-2*TICK <= price <= high+2*TICK:
                    local_trades.append(_event_dict(row))
            # Prior-only same-direction 10s bins. Bins are updated after each interaction.
            base_bins = [(end,v) for end,v in (market_bins or {}).get("effort_"+attack_side,[]) if end<=touch]
            threshold = float(np.quantile([v for _,v in base_bins], .75)) if len(base_bins) >= 20 else math.inf
            am = attack_metrics(local_trades, attack_side, threshold,
                int(params["attack_minimum_trade_records"]), float(params["attack_directional_trade_share"]))
            bin_vol = sum(t["size"] for t in local_trades if t["aggressor"] == attack_side)
            if config_id == "V3-C08": am["qualified"] = True
            attack_qlo = int(np.searchsorted(qrecv, touch, side="left"))
            attack_qhi = int(np.searchsorted(qrecv, attack_end, side="right"))
            mids = qmids[attack_qlo:attack_qhi]
            fp = failed_progress(mid, attack_side, mids, float(params["maximum_attack_direction_mid_progress_ticks"]))
            adverse_mid = (float(np.min(mids)) if direction > 0 else float(np.max(mids))) if len(mids) else mid
            record = {"date": day.date, "config_id": config_id, "zone_id": zone["zone_id"],
                      "session": zone["session"], "direction": direction, "approach_ns": touch,
                      "approach_native_index": q["native_index"], "approach_ts_event": q["ts_event"],
                      "approach_sequence": q["sequence"], "zone_low": low, "zone_high": high,
                      "zone_poc": zone["poc"], "zone_prominence": zone["prominence"],
                      "zone_volume_share": zone["volume_share"], "attack_end_ns": attack_end,
                      **am, **fp, "status": "INTERACTION"}
            context_start=int(np.searchsorted(qrecv,touch-30*60*NS,side="left"))
            context_end=int(np.searchsorted(qrecv,touch,side="right"))
            trailing_mids=qmids[context_start:context_end]
            prior_ranges=[v for end,v in (market_bins or {}).get("ranges",[]) if end<=touch][-20:]
            current_range=(float(np.max(trailing_mids))-float(np.min(trailing_mids)))/TICK if len(trailing_mids) else None
            vol_available=len(prior_ranges)>=20 and current_range is not None
            extreme=(vol_available and current_range>float(np.quantile(prior_ranges,.95)))
            trailing_eff=(abs(float(trailing_mids[-1]-trailing_mids[0]))/max(float(np.abs(np.diff(trailing_mids)).sum()),1e-9) if len(trailing_mids)>=2 else None)
            context={"trailing_30m_range_ticks":current_range,"prior_completed_range_bins":len(prior_ranges),
                     "extreme_volatility_reference_available":vol_available,"extreme_volatility":bool(extreme),
                     "directional_efficiency_30m":trailing_eff,"context_available":len(trailing_mids)>=2}
            record.update(context)
            if params.get("extreme_volatility_gate") and (not vol_available or extreme):
                record["status"]="CONTEXT_EXTREME_OR_UNAVAILABLE"; funnel.append(record); break
            countertrend=(trailing_eff is not None and direction*float(trailing_mids[-1]-trailing_mids[0])<0)
            trend_cut=params.get("block_countertrend_when_30m_efficiency_gte",params.get("countertrend_efficiency_gate"))
            if trend_cut is not None and countertrend and trailing_eff>=float(trend_cut):
                record["status"]="COUNTERTREND_CONTEXT_BLOCK"; funnel.append(record); break
            if am["qualified"]: record["status"] = "AGGRESSION_QUALIFIED"
            elif params.get("attack_minimum_trade_records", 1) > 0:
                record["status"] = "NO_MEANINGFUL_AGGRESSION"
            if not am["qualified"] or not fp["failed"]:
                funnel.append(record); break
            record["status"] = "FAILED_ATTACK_CANDIDATE"
            record["failed_attack_candidate"] = True
            if params.get("rejection_mode") == "CANDIDATE_ENTRY":
                conf = {"confirmation_ns": attack_end, "recovery_mid": mid, "reversal_trade_count": 0, "adverse_mid": adverse_mid}
            elif params.get("rejection_mode") == "R2_ORDERFLOW_LED":
                conf=orderflow_rejection_native(ev,qi,ti,qrecv,trecv,qmids,direction=direction,adverse_mid=adverse_mid,
                    after_ns=attack_end,expiry_ns=expiry,attack_volume=int(am["attack_volume"]),
                    recovery_ticks=float(params["rejection_recovery_ticks"]),volume_window_seconds=float(params.get("reversal_window_seconds",5)))
            elif params.get("rejection_mode") == "R3_COMBINED":
                conf = rejection_confirmed_native(ev,qi,ti,qrecv,trecv,qmids,direction=direction,adverse_mid=adverse_mid,
                    after_ns=attack_end,expiry_ns=expiry,recovery_ticks=float(params["rejection_recovery_ticks"]),
                    hold_seconds=float(params["rejection_hold_seconds"]),reversal_count=int(params["reversal_aggressor_trade_count"]))
                if conf:
                    revside = "BUY" if direction > 0 else "SELL"
                    a=int(np.searchsorted(trecv,conf["confirmation_ns"]-5*NS,side="left")); b=int(np.searchsorted(trecv,conf["confirmation_ns"],side="right"))
                    wanted=b"B" if revside=="BUY" else b"A"
                    rows=ev[ti[a:b]]; rv=int(rows["size"][rows["side"]==wanted].sum())
                    if rv < am["attack_volume"]: conf = None
            else:
                conf = rejection_confirmed_native(ev,qi,ti,qrecv,trecv,qmids,direction=direction,adverse_mid=adverse_mid,
                    after_ns=attack_end,expiry_ns=expiry,recovery_ticks=float(params["rejection_recovery_ticks"]),
                    hold_seconds=float(params["rejection_hold_seconds"]),reversal_count=int(params["reversal_aggressor_trade_count"]))
            record["status"] = "REJECTION_CONFIRMED" if conf else "REJECTION_NOT_CONFIRMED"
            if conf:
                record["signal_id"]=f"{config_id}|{day.date}|{zone['zone_id']}|{touch}"
                signals.append({**record, **conf, "signal_id": record["signal_id"],
                                "zone_created_ns": zone["created_ns"], "zone_activation_ns": zone["activation_ns"],
                                "entry_ns": conf["confirmation_ns"] + int(float(params["entry_latency_ms"])*1e6),
                                "rejection_extreme": adverse_mid})
            funnel.append(record); break
    signals,suppressed=deduplicate_overlapping_signals(signals)
    for row in funnel:
        if row.get("signal_id") in suppressed: row["status"]="DEDUPLICATED_OVERLAP"
    return funnel, signals


def simulate_signals(day: NativeDay, signals: list[dict[str, Any]], params: dict[str, Any],
                     *, entry_extra_ticks: float = 1, stop_extra_ticks: float = 1,
                     commission_usd: float = 6.0) -> list[dict[str, Any]]:
    """Conservative marketable-BBO entry and event-path stop/target simulation."""
    ev=day.events
    qi=np.flatnonzero((ev["action"]!=b"T")&(ev["bid"]>0)&(ev["ask"]>ev["bid"])&(ev["bid_size"]>0)&(ev["ask_size"]>0)&((ev["flags"]&(FLAG_BAD_TS_RECV|FLAG_MAYBE_BAD_BOOK))==0))
    ti=np.flatnonzero((ev["action"]==b"T")&(ev["size"]>0))
    qix=ev["native_index"][qi]; tix=ev["native_index"][ti]
    qrecv=ev["ts_recv"][qi]
    qspread=(ev["ask"][qi].astype(np.float64)-ev["bid"][qi].astype(np.float64))/(TICK*1e9)
    tprice=ev["price"][ti]
    out = []
    last_exit_index=-1
    for sig in sorted(signals, key=lambda x: x["entry_ns"]):
        direction = int(sig["direction"])
        qstart=int(np.searchsorted(qrecv,int(sig["entry_ns"]),side="left"))
        qstop=int(np.searchsorted(qrecv,max((b for _,b in session_windows(day.date).values())),side="right"))
        possible=np.flatnonzero(qspread[qstart:qstop]<=float(params["spread_gate_ticks"]))
        if not len(possible): continue
        qpos=qstart+int(possible[0]); q=ev[qi[qpos]]
        if int(q["native_index"])<=last_exit_index: continue
        entry_bid=int(q["bid"])/1e9; entry_ask=int(q["ask"])/1e9
        entry = (entry_ask + entry_extra_ticks*TICK) if direction > 0 else (entry_bid - entry_extra_ticks*TICK)
        stop, stop_ticks = stop_geometry(direction, entry, sig["zone_low"], sig["zone_high"], sig["rejection_extreme"], int(params["stop_buffer_ticks"]))
        risk_points = abs(entry-stop)
        target = entry + direction * float(params["target_r"]) * risk_points
        target_ticks = math.ceil(target/TICK - 1e-12) if direction > 0 else math.floor(target/TICK + 1e-12)
        target = target_ticks*TICK
        end_ns = max((hi for _, hi in session_windows(day.date).values()), default=q["ts_recv"])
        qend=int(np.searchsorted(qrecv,end_ns,side="right")); pathq=np.arange(qpos,qend,dtype=np.int64)
        start_native=int(q["native_index"])
        path_t0=int(np.searchsorted(tix,start_native,side="left")); path_t1=int(np.searchsorted(tix,int(ev["native_index"][qi[qend-1]]) if qend>qpos else start_native,side="right")) if qend>qpos else path_t0
        path_t=np.arange(path_t0,path_t1,dtype=np.int64)
        if direction>0:
            stops=pathq[ev["bid"][qi[pathq]]<=int(round(stop*1e9))]
            targets=path_t[tprice[path_t]>=int(round((target+TICK)*1e9))]
        else:
            stops=pathq[ev["ask"][qi[pathq]]>=int(round(stop*1e9))]
            targets=path_t[tprice[path_t]<=int(round((target-TICK)*1e9))]
        stop_native=int(qix[stops[0]]) if len(stops) else math.inf
        target_native=int(tix[targets[0]]) if len(targets) else math.inf
        if stop_native<=target_native and stop_native!=math.inf:
            row=ev[qi[int(stops[0])]]; exit_row={"reason":"STOP","ts_ns":int(row["ts_recv"]),"price":stop-stop_extra_ticks*TICK*direction,"native_index":stop_native}
        elif target_native!=math.inf:
            row=ev[ti[int(targets[0])]]; exit_row={"reason":"TARGET","ts_ns":int(row["ts_recv"]),"price":target,"native_index":target_native}
        elif qend>qpos:
            row=ev[qi[qend-1]]; exit_row={"reason":"SESSION_END","ts_ns":int(row["ts_recv"]),"price":int(row["bid" if direction>0 else "ask"])/1e9,"native_index":int(row["native_index"])}
        else: continue
        gross_points = direction*(exit_row["price"]-entry)
        gross_dollars = gross_points*50
        net_dollars = gross_dollars - commission_usd
        risk_dollars = risk_points*50
        exit_native=int(exit_row["native_index"])
        qpath_end=int(np.searchsorted(qix,exit_native,side="right"))
        path_mid=(ev["bid"][qi[qpos:qpath_end]].astype(np.float64)+ev["ask"][qi[qpos:qpath_end]].astype(np.float64))/(2e9)
        move_ticks=direction*(path_mid-entry)/TICK if len(path_mid) else np.asarray([0.0])
        out.append({**sig,"entry_fill_ns":int(q["ts_recv"]),"entry_native_index":start_native,
                    "entry_bid":entry_bid,"entry_ask":entry_ask,"entry_price":entry,
                    "stop_price":stop,"stop_ticks":stop_ticks,"target_price":target,
                    "exit_ns":exit_row["ts_ns"],"exit_price":exit_row["price"],"exit_reason":exit_row["reason"],
                    "gross_dollars":gross_dollars,"commission_usd":commission_usd,"net_dollars":net_dollars,
                    "gross_r":gross_dollars/risk_dollars,"net_r":net_dollars/risk_dollars,
                    "cost_r":(commission_usd+entry_extra_ticks*TICK*50)/risk_dollars,
                    "mfe_ticks":float(max(0.0,float(np.max(move_ticks)))),
                    "mae_ticks":float(max(0.0,-float(np.min(move_ticks)))),
                    "mfe_r":float(max(0.0,float(np.max(move_ticks)))/(risk_points/TICK)),
                    "mae_r":float(max(0.0,-float(np.min(move_ticks)))/(risk_points/TICK)),
                    "hold_seconds":(exit_row["ts_ns"]-int(q["ts_recv"]))/NS,
                    "status":"COMPLETED"})
        last_exit_index=int(exit_row["native_index"])
    return out


def performance(trades: list[dict[str, Any]], eligible_dates: list[str]) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades: by_day[t["date"]].append(t)
    daily = []
    for d in eligible_dates:
        rr = [float(t["net_r"]) for t in by_day.get(d, [])]
        daily.append({"date":d,"trades":len(rr),"net_r":sum(rr),"positive":sum(x>0 for x in rr),"negative":sum(x<0 for x in rr)})
    vals=[float(t["net_r"]) for t in trades]
    wins=[x for x in vals if x>0]; losses=[x for x in vals if x<0]
    eq=0.; peak=0.; dd=0.
    for x in vals: eq+=x; peak=max(peak,eq); dd=max(dd,peak-eq)
    return {"eligible_dates":len(eligible_dates),"active_dates":sum(x["trades"]>0 for x in daily),
            "positive_days":sum(x["net_r"]>0 for x in daily),"zero_trade_days":sum(x["trades"]==0 for x in daily),
            "trades":len(vals),"winners":len(wins),"losers":len(losses),"win_rate":len(wins)/len(vals) if vals else 0.0,
            "total_net_r":sum(vals),"mean_net_r_per_trade":statistics.mean(vals) if vals else 0.0,
            "median_net_r_per_trade":statistics.median(vals) if vals else 0.0,
            "profit_factor":sum(wins)/abs(sum(losses)) if losses else (math.inf if wins else 0.0),
            "max_drawdown_r":dd,"worst_trade_r":min(vals) if vals else 0.0,"best_trade_r":max(vals) if vals else 0.0,
            "median_stop_ticks":statistics.median([t["stop_ticks"] for t in trades]) if trades else None,
            "cost_per_trade_usd":statistics.mean([t["commission_usd"] for t in trades]) if trades else None,
            "daily":daily}


def verify_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p=root/OUT
    protocol=json.loads((p/"frozen-study-protocol.json").read_text())
    registry=json.loads((p/"configuration-registry.json").read_text())
    split=json.loads((p/"development-split-manifest.json").read_text())
    expected=(p/"frozen-study-protocol.sha256").read_text().split()[0]
    if sha256(p/"frozen-study-protocol.json") != expected: raise ValueError("frozen protocol hash mismatch")
    if len(registry.get("configurations",[]))>14: raise ValueError("configuration budget exceeded")
    if len(split.get("construction_dates",[]))!=20 or len(split.get("internal_assessment_dates",[]))!=10 or len(split.get("final_spring_research_audit_dates",[]))!=5:
        raise ValueError("invalid development split")
    return protocol, registry, split


def run(root: Path, *, max_configs: int | None = None) -> dict[str, Any]:
    started=time.monotonic(); root=root.resolve(); out=root/OUT; out.mkdir(parents=True,exist_ok=True)
    protocol, registry, split=verify_protocol(root)
    # Reverify independently against the sealed prior study source manifest.
    source_doc=json.loads((root/SOURCE_MANIFEST).read_text())
    sources=[]
    for s in source_doc["sources"]:
        p=root/s["raw_path"]
        if not p.is_file() or p.stat().st_size!=s["raw_bytes"] or sha256(p)!=s["raw_sha256"]:
            raise ValueError(f"source integrity failed: {s['date']}")
        expected_symbol="ESH5" if s["date"]<="2025-03-14" else "ESM5"
        if s.get("dataset")!="GLBX.MDP3" or s.get("schema")!="mbp-10" or s.get("symbol")!=expected_symbol:
            raise ValueError(f"source identity/roll mismatch for {s['date']}: {s}")
        sources.append(dict(s))
    if len(sources)!=35: raise ValueError(f"expected 35 source dates, got {len(sources)}")
    atomic_json(out/"source-integrity.json",{"status":"PASS","source_manifest_path":str(SOURCE_MANIFEST),"source_manifest_sha256":sha256(root/SOURCE_MANIFEST),"authorized_date_count":35,"sources":sources})
    all_dates=[s["date"] for s in sources]
    # Process one date at a time. The compact source tape is mmap-backed and
    # content-addressed, so interruption resumes without holding 35 tapes in RAM.
    cache_dir=out/".native-cache"
    config_rows=registry["configurations"]
    if max_configs is not None: config_rows=config_rows[:max_configs]
    baseline=next(c for c in registry["configurations"] if c["id"]=="V3-C00")
    all_zone_rows=[]; all_profile_diag=[]; all_funnel=[]; all_signal_rows=[]; all_trades=[]; results=[]; daily_rows=[]
    construction_dates=set(split["construction_dates"])
    signal_by_config={c["id"]:[] for c in config_rows}; trade_by_config={c["id"]:[] for c in config_rows}
    moderate_by_config={c["id"]:[] for c in config_rows}; severe_by_config={c["id"]:[] for c in config_rows}
    rr_by_target={target:[] for target in (1.0,1.5,2.0,2.25,2.5,2.75,3.0)}
    bins_history={"effort_BUY":[],"effort_SELL":[],"ranges":[]}
    zone_history=[]
    total_source_events=0; total_retained_events=0; invalid_bbo=unknown_aggressor=0
    for date_i,source in enumerate(sources,1):
        d=source["date"]
        print(f"V3_DATE_START {date_i}/35 {d}",flush=True)
        day=extract_native_day(root,source,cache_dir/f"{d}.npy")
        if day.ts_recv_backsteps: raise ValueError(f"ts_recv backstep prevents timestamp-indexed causal features: {d}")
        total_source_events+=day.event_count; total_retained_events+=len(day.events)
        invalid_bbo+=day.invalid_bbo_count; unknown_aggressor+=day.unknown_aggressor_count
        today_bins=build_market_bins(day)
        market_bins={k:bins_history[k]+today_bins[k] for k in bins_history}
        zones,diag=build_profiles(day)
        all_zone_rows.extend(zones); all_profile_diag.extend(diag)
        visible_zones=zone_history+zones
        for cfg in config_rows:
            params=resolve_config(cfg,baseline)
            funnel,signals=generate_signals(day,visible_zones,params,{},cfg["id"],market_bins)
            all_funnel.extend(funnel); signal_by_config[cfg["id"]].extend(signals); all_signal_rows.extend(signals)
            if d in construction_dates:
                trades=simulate_signals(day,signals,params)
                trade_by_config[cfg["id"]].extend(trades); all_trades.extend(trades)
                moderate_by_config[cfg["id"]].extend(simulate_signals(day,signals,params,entry_extra_ticks=1.5,stop_extra_ticks=1.5))
                severe_by_config[cfg["id"]].extend(simulate_signals(day,signals,params,entry_extra_ticks=3.0,stop_extra_ticks=3.0))
                if cfg["id"]=="V3-C00":
                    for target in rr_by_target:
                        rrparams=dict(params,target_r=target)
                        rr_by_target[target].extend(simulate_signals(day,signals,rrparams))
        zone_history.extend(zones)
        for k in bins_history: bins_history[k].extend(today_bins[k])
        print(f"V3_DATE_DONE {date_i}/35 {d} zones={len(zones)} signals={sum(len(v) for v in signal_by_config.values())}",flush=True)
        del day
    for cfg in config_rows:
        construction=trade_by_config[cfg["id"]]
        perf=performance(construction,split["construction_dates"])
        results.append({"config_id":cfg["id"],"kind":cfg["kind"],"components":cfg["components"],"construction":{k:v for k,v in perf.items() if k!="daily"},"parameters":resolve_config(cfg,baseline)})
        daily_rows += [{"config_id":cfg["id"],**x} for x in perf["daily"]]
        if cfg["id"]=="V3-C00": atomic_json(out/"baseline-performance.json",perf)
    write_parquet(out/"v3-hvn-zone-ledger.parquet",all_zone_rows)
    atomic_json(out/"v3-volume-profile-diagnostics.json",{"profile_count":len(all_profile_diag),"profiles":all_profile_diag})
    write_parquet(out/"v3-aggression-episode-ledger.parquet",[x for x in all_funnel if x.get("attack_count",0)>0])
    write_parquet(out/"v3-rejection-state-transitions.parquet",[x for x in all_funnel if x.get("status","").startswith("REJECTION") or x.get("status")=="FAILED_ATTACK_CANDIDATE"])
    write_parquet(out/"v3-context-feature-ledger.parquet",[{k:x.get(k) for k in ("date","config_id","zone_id","session","direction","approach_ns","status")} for x in all_funnel])
    write_parquet(out/"v3-full-candidate-funnel.parquet",all_funnel)
    write_rows(out/"v3-entry-intention-ledger.csv",all_signal_rows)
    write_rows(out/"v3-order-submission-ledger.csv",[{**x,"order_type":"MARKETABLE_BBO","status":"SUBMITTED"} for x in all_signal_rows])
    write_rows(out/"v3-execution-trade-ledger.csv",all_trades)
    write_rows(out/"configuration-results.csv",results)
    write_rows(out/"configuration-daily-results.csv",daily_rows)
    write_rows(out/"signal-frequency-and-retention.csv",[{"config_id":r["config_id"],"signals":sum(x["config_id"]==r["config_id"] for x in all_signal_rows),"filled_trades":sum(x["config_id"]==r["config_id"] for x in all_trades)} for r in results])
    # Apply the preregistered construction gate exactly; no later dates are read for selection.
    eligible=[]
    for r in results:
        m=r["construction"]
        daily=[x for x in daily_rows if x["config_id"]==r["config_id"]]
        positive=sum(x["net_r"]>0 for x in daily)
        pos_contrib=sum(max(0,x["net_r"]) for x in daily)
        max_day=max((max(0,x["net_r"]) for x in daily),default=0)
        moderate=performance(moderate_by_config[r["config_id"]],eligible_dates=split["construction_dates"])["total_net_r"]
        if (m["mean_net_r_per_trade"]>0 and positive>=11 and m["trades"]>=10 and m["profit_factor"]>=1.05 and (max_day <= .5*pos_contrib if pos_contrib else False) and moderate>0): eligible.append(r["config_id"])
    gate={"gate_0_source_integrity":"PASS","gate_1_causal_signal_pipeline":"PASS","gate_2_baseline_construction":results[0]["construction"],"gate_3_registered_configuration_selection":{"eligible_configs":eligible,"selected_config":None,"status":"NO_GO" if not eligible else "ELIGIBLE_FOR_FROZEN_SELECTION"},"internal_assessment":"NOT_RUN","final_spring_audit":"NOT_RUN"}
    atomic_json(out/"go-no-go-gate-results.json",gate)
    cost_sensitivity={"construction_dates":split["construction_dates"],"scenarios":{
        "BASELINE":{"entry_extra_ticks":1.0,"stop_extra_ticks":1.0,"commission_usd":6.0},
        "MODERATE":{"entry_extra_ticks":1.5,"stop_extra_ticks":1.5,"commission_usd":6.0},
        "SEVERE":{"entry_extra_ticks":3.0,"stop_extra_ticks":3.0,"commission_usd":6.0}},
        "by_config":{c["id"]:{"baseline_net_r":performance(trade_by_config[c["id"]],split["construction_dates"])["total_net_r"],
            "moderate_net_r":performance(moderate_by_config[c["id"]],split["construction_dates"])["total_net_r"],
            "severe_net_r":performance(severe_by_config[c["id"]],split["construction_dates"])["total_net_r"]} for c in config_rows}}
    atomic_json(out/"cost-sensitivity.json",cost_sensitivity)
    write_rows(out/"rr-grid-results.csv",[{"target_r":r,"metrics":performance(ts,split["construction_dates"])} for r,ts in rr_by_target.items()])
    atomic_json(out/"mfe-mae-analysis.json",{"construction_trade_count":len(all_trades),"mean_mfe_ticks":statistics.mean([x["mfe_ticks"] for x in all_trades]) if all_trades else 0.0,
        "median_mfe_ticks":statistics.median([x["mfe_ticks"] for x in all_trades]) if all_trades else 0.0,
        "mean_mae_ticks":statistics.mean([x["mae_ticks"] for x in all_trades]) if all_trades else 0.0,
        "median_mae_ticks":statistics.median([x["mae_ticks"] for x in all_trades]) if all_trades else 0.0,
        "mean_mfe_r":statistics.mean([x["mfe_r"] for x in all_trades]) if all_trades else 0.0,
        "mean_mae_r":statistics.mean([x["mae_r"] for x in all_trades]) if all_trades else 0.0})
    summary={"study_id":STUDY_ID,"status":"NO_GO" if not eligible else "CONSTRUCTION_COMPLETE","source_integrity":"PASS","dates":len(all_dates),"construction_dates":20,"internal_dates":10,"final_audit_dates":5,"source_events_processed":total_source_events,"retained_tape_events":total_retained_events,"invalid_bbo_records":invalid_bbo,"unknown_aggressor_records":unknown_aggressor,"zones":len(all_zone_rows),"interactions":len(all_funnel),"qualified_attacks":sum(x.get("qualified",False) for x in all_funnel),"failed_attack_candidates":sum(x.get("status")=="FAILED_ATTACK_CANDIDATE" for x in all_funnel),"confirmed_rejections":sum(x.get("status")=="REJECTION_CONFIRMED" for x in all_funnel),"signals":len(all_signal_rows),"trades":len(all_trades),"configs_evaluated":len(results),"eligible_configurations":eligible,"elapsed_seconds":time.monotonic()-started,"stages":{"stage0":"PASS","stage1":"PASS","stage2":"PASS","stage3":"PASS","stage4":"PASS","stage5":"SKIPPED","internal_assessment":"NOT_RUN","final_spring_audit":"NOT_RUN"}}
    atomic_json(out/"summary.json",summary)
    return summary


def simulate_signals_for_cost(day_cache: dict[str,NativeDay], signals: list[dict[str,Any]], params:dict[str,Any], entry:float, stop:float) -> list[dict[str,Any]]:
    out=[]
    for d,day in day_cache.items():
        out += simulate_signals(day,[s for s in signals if s["date"]==d],params,entry_extra_ticks=entry,stop_extra_ticks=stop)
    return out


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,default=Path.cwd())
    ap.add_argument("--max-configs",type=int,default=None,help="diagnostic cap; default executes frozen registry")
    ns=ap.parse_args(argv)
    result=run(ns.repo,max_configs=ns.max_configs)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
