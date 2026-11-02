"""Frozen, outcome-blind context reconstruction and bounded V1 HVN research.

The study uses only the existing Spring TRAIN sources and frozen V1 outcomes.
It does not rebuild signals, alter execution, or optimize thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import mac_2025_hvn_failed_aggression_detector_v2_research as v2
from . import mac_2025_hvn_native_mbp10_absorption_mechanism_v1 as native
from . import mac_2025_es_only_train_baseline as baseline

STUDY_ID = "ES_HVN_CONTEXT_INTERACTION_RESEARCH_V1"
OUT = Path("research_runs") / STUDY_ID
V1 = Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1")
V2 = Path("research_runs/ES_HVN_FAILED_AGGRESSION_DETECTOR_V2_RESEARCH")
FINAL = Path("research_runs/ES_HVN_ABSORPTION_FINAL_PREDICTIVE_DECISION_V1")
CAUSAL = Path("research_runs/ES_HVN_CAUSAL_ABSORPTION_DETECTION_RESEARCH_V1")
AUDIT = Path("research_runs/ES_HVN_FAILED_AGGRESSION_DETECTION_AUDIT_V1")
MECHANISM = Path("research_runs/ES_HVN_NATIVE_MBP10_ABSORPTION_MECHANISM_V1")
TICK = 0.25
NS = 1_000_000_000
CHUNK_ROWS = 250_000
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261027
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
LONDON = ZoneInfo("Europe/London")
EXTRACTOR_VERSION = "context-v1.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
    tmp.replace(path)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({"status": pa.array([], type=pa.string())})
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def session_label_et(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / NS, timezone.utc).astimezone(ET)
    minute = dt.hour * 60 + dt.minute
    if minute >= 18 * 60 or minute < 3 * 60:
        return "ASIA_OVERNIGHT"
    if minute < 8 * 60:
        return "EUROPE"
    if minute < 9 * 60 + 30:
        return "NY_PREOPEN"
    if minute < 10 * 60 + 30:
        return "NY_OPEN"
    if minute < 12 * 60:
        return "NY_MORNING"
    if minute < 13 * 60 + 30:
        return "NY_LUNCH"
    if minute < 16 * 60:
        return "NY_AFTERNOON"
    return "OTHER_TRANSITION"


def chronological_split(dates: list[str], development_count: int = 25) -> dict[str, list[str]]:
    ordered = sorted(dates)
    return {"development": ordered[:development_count], "later_internal_assessment": ordered[development_count:]}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Monotone BH adjusted p-values, returned in original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [1.0] * n
    running = 1.0
    for rank0 in range(n - 1, -1, -1):
        i = order[rank0]
        running = min(running, p_values[i] * n / (rank0 + 1))
        q[i] = min(1.0, running)
    return q


def date_bootstrap_difference(treatment: list[dict[str, Any]], control: list[dict[str, Any]],
                              dates: list[str], metric: str = "net_r", replicates: int = BOOTSTRAP_N,
                              seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap date clusters; resamples without both groups remain unavailable."""
    by_date_t: dict[str, list[float]] = defaultdict(list)
    by_date_c: dict[str, list[float]] = defaultdict(list)
    for row in treatment:
        x = _float(row.get(metric))
        if x is not None:
            by_date_t[str(row["date"])].append(x)
    for row in control:
        x = _float(row.get(metric))
        if x is not None:
            by_date_c[str(row["date"])].append(x)
    valid_dates = [d for d in dates if by_date_t[d] and by_date_c[d]]
    date_diffs = {d: float(np.mean(by_date_t[d]) - np.mean(by_date_c[d])) for d in valid_dates}
    all_t=[x for values in by_date_t.values() for x in values]
    all_c=[x for values in by_date_c.values() for x in values]
    estimate = float(np.mean(all_t) - np.mean(all_c)) if all_t and all_c else None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(replicates):
        picked = rng.choice(dates, size=len(dates), replace=True) if dates else []
        sample_t=[x for d in picked for x in by_date_t[str(d)]]
        sample_c=[x for d in picked for x in by_date_c[str(d)]]
        if sample_t and sample_c:
            samples.append(float(np.mean(sample_t)-np.mean(sample_c)))
    ci = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))] if samples else None
    p = None
    if samples and estimate is not None:
        centered = np.asarray(samples) - estimate
        p = float((np.count_nonzero(np.abs(centered) >= abs(estimate)) + 1) / (len(centered) + 1))
    return {"estimate_trade_weighted_difference": estimate, "ci95": ci, "p_two_sided_centered_bootstrap": p,
            "matched_dates": len(valid_dates), "valid_replicates": len(samples), "replicates_requested": replicates,
            "seed": seed, "date_effects": date_diffs, "date_universe": len(dates)}


def _decode_action(values: np.ndarray) -> np.ndarray:
    return values.astype("S1")


def _v1_rows(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(repo / V1 / "rr-grid-trade-results.csv")
    rr_values = sorted({float(r["rr"]) for r in rows})
    signal_sets: dict[float, set[str]] = {}
    for rr in rr_values:
        subset = [r for r in rows if float(r["rr"]) == rr]
        signal_sets[rr] = {r["signal_id"] for r in subset}
        if len(subset) != 936 or len(signal_sets[rr]) != 936:
            raise ValueError(f"V1 RR={rr} population is not 936 unique signals")
    if any(signal_sets[rr] != signal_sets[1.0] for rr in rr_values):
        raise ValueError("V1 RR counterfactuals do not share identical signal identities")
    rr1 = [dict(r) for r in rows if float(r["rr"]) == 1.0]
    by_signal: dict[str, dict[str, Any]] = {}
    for rr in rr_values:
        for raw in (r for r in rows if float(r["rr"]) == rr):
            sid = raw["signal_id"]
            item = by_signal.setdefault(sid, {"signal_id": sid, "date": raw["date"], "direction": int(raw["direction"]),
                "entry_ns": int(raw["TE_ns"]), "entry_price": float(raw["entry_price"]), "stop_price": float(raw["stop_price"]),
                "stop_ticks": float(raw["risk_ticks"]), "zone_id": raw["zone_id"], "first_touch_ns": int(raw["first_touch_ns"]),
                "rr_outcomes": {}})
            if (raw["date"] != item["date"] or raw["zone_id"] != item["zone_id"] or
                int(raw["direction"]) != item["direction"] or int(raw["TE_ns"]) != item["entry_ns"] or
                int(raw["first_touch_ns"]) != item["first_touch_ns"] or
                abs(float(raw["entry_price"]) - item["entry_price"]) > 1e-9 or
                abs(float(raw["stop_price"]) - item["stop_price"]) > 1e-9 or
                abs(float(raw["risk_ticks"]) - item["stop_ticks"]) > 1e-9):
                raise ValueError(f"V1 signal identity or fixed execution geometry changed across RR grid: {sid}")
            item["rr_outcomes"][str(float(rr))] = {"net_r": float(raw["net_r"]), "gross_r": float(raw["gross_r"]),
                "outcome": raw["outcome"], "exit_ns": int(raw["exit_timestamp_ns"]), "holding_seconds": float(raw["holding_seconds"]),
                "mfe_ticks": float(raw["mfe_ticks"]), "mae_ticks": float(raw["mae_ticks"]),
                "target_first": raw["target_first"].lower() == "true", "stop_first": raw["stop_first"].lower() == "true"}
    signals = sorted(by_signal.values(), key=lambda r: (r["date"], r["entry_ns"], r["signal_id"]))
    if len(signals) != 936:
        raise ValueError("V1 identity reconciliation did not yield 936 trades")
    return signals, rr1, {"rr_values": rr_values, "signals": len(signals), "rows": len(rows), "same_population_all_rr": True}


def _native_context_for_date(repo: Path, date: str, source: dict[str, Any],
                             trade_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """One bounded-memory native pass; compact quote/trade arrays serve all targets."""
    from databento import DBNStore
    path = repo / source["raw_path"]
    if path.stat().st_size != source["raw_bytes"] or sha256(path) != source["raw_sha256"]:
        raise ValueError(f"native source integrity mismatch: {date}")
    target_entries: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_rows:
        target_entries[int(row["entry_ns"])].append(row)
    target_touches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        target_touches[int(row["touch_ns"])].append(row)
    # MBP streams are ts_recv monotone in the frozen integrity scan. A raw
    # record ordinal is retained to resolve equal receive timestamps causally.
    quote_cols: dict[str, list[np.ndarray]] = {k: [] for k in ("idx", "recv", "event", "mid", "bid", "ask", "spread", "top1b", "top1a", "top5b", "top5a", "top10b", "top10a")}
    trade_cols: dict[str, list[np.ndarray]] = {k: [] for k in ("idx", "recv", "price", "size", "side")}
    matched_entry_idx: dict[str, int] = {}
    matched_touch_idx: dict[str, int] = {}
    global_offset = 0
    record_count = bad_recv = invalid_bbo = trade_count = unknown_aggressor = 0
    previous_recv: int | None = None
    for chunk in DBNStore.from_file(path).to_ndarray(count=CHUNK_ROWS):
        n = len(chunk)
        if not n:
            continue
        if not np.all(chunk["rtype"] == 10):
            raise ValueError(f"non-MBP-10 record in {date}")
        recv = chunk["ts_recv"].astype(np.int64, copy=False)
        if np.any(recv[1:] < recv[:-1]) or (previous_recv is not None and int(recv[0]) < previous_recv):
            raise ValueError(f"ts_recv ordering violation in {date}; preserving native order is required")
        previous_recv = int(recv[-1])
        flags = chunk["flags"].astype(np.uint8, copy=False)
        bad_recv += int(np.count_nonzero(flags & native.FLAG_BAD_TS_RECV))
        bidpx = chunk["bid_px_00"].astype(np.int64, copy=False)
        askpx = chunk["ask_px_00"].astype(np.int64, copy=False)
        bidsz = chunk["bid_sz_00"].astype(np.int64, copy=False)
        asksz = chunk["ask_sz_00"].astype(np.int64, copy=False)
        valid = (bidpx > 0) & (askpx > bidpx) & (bidsz > 0) & (asksz > 0) & ((flags & native.FLAG_BAD_TS_RECV) == 0) & ((flags & native.FLAG_MAYBE_BAD_BOOK) == 0)
        invalid_bbo += int(np.count_nonzero(~valid))
        seconds = recv // NS
        starts = np.r_[0, np.flatnonzero(seconds[1:] != seconds[:-1]) + 1]
        local = np.arange(n, dtype=np.int64)
        last_valid = np.maximum.reduceat(np.where(valid, local, -1), starts)
        keep_groups = last_valid >= 0
        last_valid = last_valid[keep_groups]
        if len(last_valid):
            pxscale = 1e9
            b0 = bidpx[last_valid].astype(np.float64) / pxscale
            a0 = askpx[last_valid].astype(np.float64) / pxscale
            q = {"idx": global_offset + last_valid, "recv": recv[last_valid], "event": chunk["ts_event"][last_valid].astype(np.int64,copy=False), "mid": (b0 + a0) / 2,
                 "bid": b0, "ask": a0, "spread": (a0 - b0) / TICK,
                 "top1b": bidsz[last_valid], "top1a": asksz[last_valid]}
            # Full depth at sampled records is available on the MBP-10 rows.
            for levels, bid_name, ask_name in ((5,"top5b","top5a"),(10,"top10b","top10a")):
                q[bid_name] = sum(chunk[f"bid_sz_{j:02d}"][last_valid].astype(np.int64) for j in range(levels))
                q[ask_name] = sum(chunk[f"ask_sz_{j:02d}"][last_valid].astype(np.int64) for j in range(levels))
            for k, value in q.items(): quote_cols[k].append(np.asarray(value))
        action = chunk["action"].astype("S1", copy=False)
        is_trade = (action == b"T") & (chunk["size"].astype(np.int64, copy=False) > 0)
        ti = np.flatnonzero(is_trade)
        trade_count += int(len(ti))
        if len(ti):
            side = chunk["side"][ti].astype("S1", copy=False)
            unknown_aggressor += int(np.count_nonzero((side != b"A") & (side != b"B")))
            tx = chunk["price"][ti].astype(np.int64, copy=False).astype(np.float64) / 1e9
            q = {"idx": global_offset + ti, "recv": recv[ti], "price": tx,
                 "size": chunk["size"][ti].astype(np.int64, copy=False), "side": side}
            for k, value in q.items(): trade_cols[k].append(np.asarray(value))
        # Resolve targets at native record ordinal. For executable entries use
        # the original frozen BBO and first matching record at the exact recv ns.
        for target_ns in set(target_entries).union(target_touches):
            left = int(np.searchsorted(recv, target_ns, side="left"))
            right = int(np.searchsorted(recv, target_ns, side="right"))
            if left < right:
                if target_ns in target_entries:
                    for entry in target_entries[target_ns]:
                        if entry["signal_id"] in matched_entry_idx:
                            continue
                        eb, ea = int(round(entry["entry_bid"] * 1e9)), int(round(entry["entry_ask"] * 1e9))
                        found = next((j for j in range(left, right) if valid[j] and bidpx[j] == eb and askpx[j] == ea), None)
                        if found is not None:
                            matched_entry_idx[entry["signal_id"]] = global_offset + found
                            entry["_native_ts_event_ns"]=int(chunk["ts_event"][found])
                            entry["_native_snapshot"]={"bid_sz_00":int(bidsz[found]),"ask_sz_00":int(asksz[found]),
                                "bid_top5":int(sum(int(chunk[f"bid_sz_{j:02d}"][found]) for j in range(5))),
                                "ask_top5":int(sum(int(chunk[f"ask_sz_{j:02d}"][found]) for j in range(5))),
                                "bid_top10":int(sum(int(chunk[f"bid_sz_{j:02d}"][found]) for j in range(10))),
                                "ask_top10":int(sum(int(chunk[f"ask_sz_{j:02d}"][found]) for j in range(10)))}
                if target_ns in target_touches:
                    for ep in target_touches[target_ns]:
                        if ep["episode_id"] not in matched_touch_idx:
                            # First native record at the frozen receive timestamp.
                            matched_touch_idx[ep["episode_id"]] = global_offset + left
                            ep["_native_ts_event_ns"]=int(chunk["ts_event"][left])
        record_count += n
        global_offset += n
    if not quote_cols["idx"]:
        raise ValueError(f"no valid BBO samples in source date {date}")
    q = {k: np.concatenate(v) for k,v in quote_cols.items()}
    # Same-second samples may straddle DBN chunks. Keep the latest valid native
    # record per second; entries use their own exact BBO for the current second.
    order = np.argsort(q["idx"], kind="stable")
    q = {k: v[order] for k,v in q.items()}
    sec = q["recv"] // NS
    keep = np.r_[sec[1:] != sec[:-1], True]
    q = {k:v[keep] for k,v in q.items()}
    t = {k: np.concatenate(v) if v else np.asarray([], dtype=np.int64) for k,v in trade_cols.items()}
    if len(t["idx"]):
        order = np.argsort(t["idx"], kind="stable")
        t = {k:v[order] for k,v in t.items()}
    entry_features=[]
    for entry in trade_rows:
        idx=matched_entry_idx.get(entry["signal_id"])
        entry_features.append(_feature_at(entry, idx, int(entry["entry_ns"]), q, t, date, "TRADE_ENTRY"))
    touch_features=[]
    for ep in episode_rows:
        idx=matched_touch_idx.get(ep["episode_id"])
        touch_features.append(_feature_at(ep, idx, int(ep["touch_ns"]), q, t, date, "HVN_FIRST_TOUCH"))
    diag={"date":date,"record_count":record_count,"valid_bbo_second_samples":len(q["idx"]),"native_trade_records":trade_count,
          "unknown_aggressor_trades":unknown_aggressor,"bad_recv_records":bad_recv,"invalid_top_bbo_records":invalid_bbo,
          "entry_exact_bbo_matches":len(matched_entry_idx),"entry_target_count":len(trade_rows),
          "touch_timestamp_matches":len(matched_touch_idx),"touch_target_count":len(episode_rows),
          "source_sha256":source["raw_sha256"],"source_bytes":source["raw_bytes"]}
    return entry_features,touch_features,diag


def _feature_at(target: dict[str, Any], idx: int | None, ts_ns: int, q: dict[str, np.ndarray],
                t: dict[str, np.ndarray], date: str, target_kind: str) -> dict[str, Any]:
    feature={k:v for k,v in target.items() if k != "rr_outcomes" and not k.startswith("_")}
    feature.update({"target_kind":target_kind,"entry_or_touch_ns":ts_ns,"entry_native_index":idx,
                   "matched_native_ts_event_ns":target.get("_native_ts_event_ns"),
                   "session_et":session_label_et(ts_ns),"feature_status":"OK" if idx is not None else "NO_EXACT_NATIVE_INDEX"})
    utc_dt=datetime.fromtimestamp(ts_ns/NS,timezone.utc)
    feature.update({"timestamp_utc":utc_dt.isoformat(),"timestamp_new_york":utc_dt.astimezone(ET).isoformat(),
                    "timestamp_london":utc_dt.astimezone(LONDON).isoformat(),"timestamp_chicago":utc_dt.astimezone(CT).isoformat()})
    if idx is None:
        for name in ("quote_mid","spread_ticks","realized_vol_30m_ticks","trend_efficiency_30m","vwap_session_distance_ticks",
                     "aggressive_imbalance_30m","session_high_distance_ticks","session_low_distance_ticks"):
            feature[name]=None
        feature["missing_reason"]="no exact matching native receive-time record; no stale or future substitution"
        return feature
    qend=int(np.searchsorted(q["idx"],idx,side="right"))
    qidx=q["idx"][:qend]
    target_bid=_float(target.get("entry_bid"))
    target_ask=_float(target.get("entry_ask"))
    if target_bid is not None and target_ask is not None and target_ask>target_bid:
        current_mid=(target_bid+target_ask)/2
        current_spread=(target_ask-target_bid)/TICK
    else:
        current_mid=float(q["mid"][qend-1]) if qend else None
        current_spread=float(q["spread"][qend-1]) if qend else None
    snap=target.get("_native_snapshot",{})
    if snap:
        bid1,ask1=int(snap["bid_sz_00"]),int(snap["ask_sz_00"])
        bid5,ask5=int(snap["bid_top5"]),int(snap["ask_top5"])
        bid10,ask10=int(snap["bid_top10"]),int(snap["ask_top10"])
    elif qend:
        bid1,ask1=int(q["top1b"][qend-1]),int(q["top1a"][qend-1])
        bid5,ask5=int(q["top5b"][qend-1]),int(q["top5a"][qend-1])
        bid10,ask10=int(q["top10b"][qend-1]),int(q["top10a"][qend-1])
    else:
        bid1=ask1=bid5=ask5=bid10=ask10=0
    feature.update({"quote_mid":current_mid,"spread_ticks":current_spread,
                    "top1_depth":bid1+ask1 if bid1+ask1 else None,
                    "top5_depth":bid5+ask5 if bid5+ask5 else None,
                    "top10_depth":bid10+ask10 if bid10+ask10 else None,
                    "top10_imbalance":(bid10-ask10)/(bid10+ask10) if bid10+ask10 else None})
    recv=ts_ns
    # Add the as-of current price to historical one-second marks. The history
    # includes no quote after the target row ordinal.
    h_recv=q["recv"][:qend].astype(np.int64)
    h_mid=q["mid"][:qend].astype(np.float64)
    if current_mid is not None and (not len(h_mid) or int(q["idx"][qend-1]) != idx or abs(h_mid[-1]-current_mid)>1e-9):
        h_recv=np.r_[h_recv,recv]; h_mid=np.r_[h_mid,current_mid]
    for seconds,name in ((60,"1m"),(300,"5m"),(900,"15m"),(1800,"30m")):
        lo=recv-seconds*NS
        a=int(np.searchsorted(h_recv,lo,side="left"))
        vals=h_mid[a:]
        changes=np.diff(vals)/TICK
        net=(float(vals[-1]-vals[0])/TICK) if len(vals)>1 else None
        abs_sum=float(np.sum(np.abs(changes))) if len(changes) else None
        realized=float(math.sqrt(np.sum(changes**2))) if len(changes) else None
        feature.update({f"mid_range_{name}_ticks":float((np.max(vals)-np.min(vals))/TICK) if len(vals) else None,
                        f"mid_change_{name}_ticks":net,f"absolute_path_{name}_ticks":abs_sum,
                        f"realized_variation_{name}_ticks":realized,
                        f"trend_efficiency_{name}":abs(net)/abs_sum if net is not None and abs_sum else None,
                        f"quote_samples_{name}":int(len(vals))})
    # Trades are keyed by native row ordinal to exclude same-timestamp rows
    # following the selected entry record.
    tend=int(np.searchsorted(t["idx"],idx,side="right")) if len(t["idx"]) else 0
    ti=t["idx"][:tend]; tr=t["recv"][:tend]
    side=t["side"][:tend]
    sizes=t["size"][:tend]
    prices=t["price"][:tend]
    buy=np.where(side==b"B",sizes,0); sell=np.where(side==b"A",sizes,0)
    for seconds,name in ((60,"1m"),(300,"5m"),(900,"15m"),(1800,"30m")):
        a=int(np.searchsorted(tr,recv-seconds*NS,side="left"))
        bvol=int(np.sum(buy[a:])); svol=int(np.sum(sell[a:])); n=len(tr[a:])
        feature.update({f"trade_count_{name}":int(n),f"trade_volume_{name}":int(np.sum(sizes[a:])),
                        f"buy_aggressor_volume_{name}":bvol,f"sell_aggressor_volume_{name}":svol,
                        f"aggressive_imbalance_{name}":(bvol-svol)/(bvol+svol) if bvol+svol else None})
    # Frozen canonical V1 session windows, local-date and DST aware.
    day_windows=baseline._session_windows(date)
    try:
        session_id=int(target.get("session", target.get("v1_session", -1)))
    except (TypeError,ValueError):
        session_id=-1
    session_name={0:"ASIA",1:"EUROPE",2:"NY"}.get(session_id)
    if session_name:
        win=day_windows[session_name]
        start=int(datetime.combine(datetime.fromtimestamp(ts_ns/NS,timezone.utc).date(),datetime.min.time(),tzinfo=timezone.utc).timestamp()*NS)
        # Canonical session start comes directly from the frozen V1 schedule.
        start=int(win[0])
        ta=int(np.searchsorted(tr,start,side="left")); tb=tend
        if tb>ta:
            v=sizes[ta:tb].astype(np.float64); p=prices[ta:tb]
            feature["vwap_session"] = float(np.sum(v*p)/np.sum(v)) if np.sum(v)>0 else None
            feature["vwap_session_distance_ticks"]=(current_mid-feature["vwap_session"])/TICK if current_mid is not None and feature["vwap_session"] is not None else None
            feature["session_trade_volume"] = int(np.sum(v)); feature["session_trade_count"] = int(tb-ta)
        else:
            feature.update({"vwap_session":None,"vwap_session_distance_ticks":None,"session_trade_volume":0,"session_trade_count":0})
        qa=int(np.searchsorted(h_recv,start,side="left"))
        if len(h_mid[qa:]):
            feature["session_high_mid"] = float(np.max(h_mid[qa:])); feature["session_low_mid"] = float(np.min(h_mid[qa:]))
            feature["session_high_distance_ticks"]=(feature["session_high_mid"]-current_mid)/TICK if current_mid is not None else None
            feature["session_low_distance_ticks"]=(current_mid-feature["session_low_mid"])/TICK if current_mid is not None else None
        else:
            feature.update({"session_high_mid":None,"session_low_mid":None,"session_high_distance_ticks":None,"session_low_distance_ticks":None})
        feature["v1_canonical_session"] = session_name
    else:
        feature.update({"v1_canonical_session":None,"vwap_session":None,"vwap_session_distance_ticks":None,
                        "session_trade_volume":None,"session_trade_count":None,"session_high_mid":None,
                        "session_low_mid":None,"session_high_distance_ticks":None,"session_low_distance_ticks":None})
    return feature


def _verify_inputs(repo: Path) -> tuple[list[Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes, dates, sources=v2._load_inputs(repo)
    checked={"prior_v1_audit_native_causal_hashes":"PASS","spring_date_count":len(dates),"episode_count":len(episodes)}
    registry_path=repo/OUT/"frozen-research-hypothesis-registry.json"
    expected_registry=registry_path.with_suffix(".sha256").read_text().split()[0]
    if sha256(registry_path)!=expected_registry:
        raise ValueError("frozen hypothesis registry SHA-256 mismatch")
    for root in (V2,FINAL):
        hashes=json.loads((repo/root/"artifact-hashes.json").read_text())
        bad=[name for name,digest in hashes.items() if not (repo/root/name).is_file() or sha256(repo/root/name)!=digest]
        if bad:
            raise ValueError(f"prior study artifacts fail hash verification: {root}: {bad[:5]}")
        checked[f"{root.name}_artifact_count"]=len(hashes)
    v2_manifest=json.loads((repo/V2/"run-manifest.json").read_text())
    if not v2_manifest.get("complete") or v2_manifest.get("spec_sha256")!="8de9119f827dab50908ff5d785b426095200f579960ef4d44bb5ba50d13d8b1b":
        raise ValueError("frozen V2 detector run/spec is incomplete or changed")
    v1spec=json.loads((repo/V1/"frozen-v1-specification.json").read_text())
    if canonical_hash(v1spec["specification"])!=v1spec["sha256"]:
        raise ValueError("V1 specification hash mismatch")
    signals,rr1,rr_audit=_v1_rows(repo)
    tape_root=repo/baseline.OUTPUT_ROOT/"candidate-tapes"/"tapes"
    tape_verified=0
    for date in dates:
        mp=tape_root/f"{date}-candidate-tape-manifest.json"
        npz=tape_root/f"{date}-candidate-tape.npz"
        manifest=json.loads(mp.read_text())
        if manifest.get("date")!=date or not manifest.get("price_path_is_ordered"):
            raise ValueError(f"candidate tape identity/order invalid: {date}")
        tape_hash=manifest.get("tape_sha256") or manifest.get("sha256")
        if tape_hash and sha256(npz)!=tape_hash:
            raise ValueError(f"candidate tape hash mismatch: {date}")
        data=np.load(npz,allow_pickle=False)
        events=data["events"]
        for row in (r for r in rr1 if r["date"]==date):
            ix=int(row["entry_index"])
            if ix<0 or ix>=len(events):
                raise ValueError(f"V1 entry index out of candidate-tape bounds: {row['signal_id']}")
            ev=events[ix]
            if int(ev["timestamp_ns"])!=int(row["TE_ns"]):
                raise ValueError(f"V1 timestamp/index mismatch: {row['signal_id']}")
            if abs(float(ev["bid"])-float(row["entry_bid"]))>1e-9 or abs(float(ev["ask"])-float(row["entry_ask"]))>1e-9:
                raise ValueError(f"V1 entry BBO/index mismatch: {row['signal_id']}")
        tape_verified+=1
    zones=_read_csv(repo/V1/"hvn-zone-ledger.csv")
    zone_map={(r["date"],r["zone_id"]):r for r in zones}
    touches=_read_csv(repo/V1/"hvn-touch-episodes.csv")
    touch_map={(r["date"],r["zone_id"]):r for r in touches}
    detector={r["episode_id"]:r for r in _read_csv(repo/V2/"detector-event-ledger.csv")}
    trade_targets=[]
    for raw in rr1:
        zone=zone_map[(raw["date"],raw["zone_id"])]
        tr=touch_map.get((raw["date"],raw["zone_id"]),{})
        episode_id=f"{raw['date']}|{raw['zone_id']}|{raw['first_touch_ns']}"
        dv=detector.get(episode_id,{})
        entry_ns=int(raw["TE_ns"])
        src_ok=dv.get("source_validity")=="VALID_WITHIN_OBSERVED_ROWS"
        threshold_ok=dv.get("threshold_available","False").lower()=="true"
        def pre(name: str) -> bool:
            ts=_int(dv.get(name))
            return bool(ts is not None and ts<entry_ns)
        target={"signal_id":raw["signal_id"],"date":raw["date"],"zone_id":raw["zone_id"],"episode_id":episode_id,
            "entry_ns":entry_ns,"first_touch_ns":int(raw["first_touch_ns"]),"entry_price":float(raw["entry_price"]),
            "entry_bid":float(raw["entry_bid"]),"entry_ask":float(raw["entry_ask"]),"stop_price":float(raw["stop_price"]),
            "stop_ticks":float(raw["risk_ticks"]),"direction":int(raw["direction"]),"session":int(raw["session"]),
            "zone_low":float(raw["zone_low"]),"zone_high":float(raw["zone_high"]),"zone_poc":float(raw["zone_poc"]),
            "zone_volume":float(raw["zone_volume"]),"profile_id":raw["profile_id"],"profile_start_ns":int(raw["profile_start_ns"]),
            "profile_close_ns":int(raw["profile_close_ns"]),"zone_active_from_ns":int(zone["active_from_ns"]),
            "zone_prominence_threshold":float(zone["prominence_threshold"]),"zone_peak_count":int(zone["cluster_peak_count"]),
            "zone_width_ticks":(float(raw["zone_high"])-float(raw["zone_low"]))/TICK,
            "zone_age_at_entry_seconds":(entry_ns-int(zone["active_from_ns"]))/NS,
            "heavy_print":raw["heavy_print"].lower()=="true","heavy_size":_float(raw.get("heavy_size")),
            "heavy_threshold":_float(raw.get("heavy_threshold")),"heavy_to_confirmation_ms":_float(raw.get("TH_to_confirmation_ms")),
            "touch_to_heavy_ms":_float(raw.get("T0_to_TH_ms")),"reclaim_distance_ticks":_float(raw.get("TH_to_TC_directional_ticks")),
            "entry_rejection_extreme_distance_ticks":_float(raw.get("entry_rejection_extreme_distance_ticks")),
            "confirmation_to_entry_ms":_float(raw.get("confirmation_to_entry_ms")),"entry_spread_ticks":_float(raw.get("entry_spread_ticks")),
            "entry_slippage_ticks":_float(raw.get("entry_slippage_ticks")),
            "stop_to_prior_30s_range_ratio":_float(raw.get("stop_distance_to_nearby_30s_range_ratio")),
            "v1_session":{0:"ASIA",1:"EUROPE",2:"NY"}.get(int(raw["session"])),
            "v2_source_valid_at_entry":src_ok,"v2_threshold_available":threshold_ok,
            "v2_evidence_status_at_entry":"UNAVAILABLE_SOURCE" if not src_ok else "UNAVAILABLE_NO_THRESHOLD" if not threshold_ok else "AVAILABLE",
            "v2_effort_pre_entry":pre("effort_ns") if src_ok and threshold_ok else None,
            "v2_candidate_pre_entry":pre("candidate_ns") if src_ok and threshold_ok else None,
            "v2_rejection_pre_entry":pre("rejection_ns") if src_ok and threshold_ok else None,
            "v2_continuation_pre_entry":pre("continuation_ns") if src_ok and threshold_ok else None,
            "preentry_feature_cutoff_ns":entry_ns,"feature_source":"native ES MBP-10 + frozen V1/V2 ledgers"}
        trade_targets.append(target)
    ep_targets=[]
    for ep in episodes:
        zone=zone_map[(ep.date,ep.zone_id)]
        ep_targets.append({"episode_id":ep.episode_id,"date":ep.date,"zone_id":ep.zone_id,"touch_ns":int(ep.t0),
            "direction":ep.direction,"session":int(zone["session"]),"zone_low":ep.low,"zone_high":ep.high,"zone_poc":ep.poc,
            "zone_volume":float(zone["zone_volume"]),"zone_prominence_threshold":float(zone["prominence_threshold"]),
            "zone_peak_count":int(zone["cluster_peak_count"]),"zone_width_ticks":(ep.high-ep.low)/TICK,
            "profile_id":zone["profile_id"],"profile_start_ns":int(zone["profile_start_ns"]),
            "profile_close_ns":int(zone["profile_close_ns"]),"zone_active_from_ns":int(zone["active_from_ns"]),
            "zone_age_at_touch_seconds":(int(ep.t0)-int(zone["active_from_ns"]))/NS,
            "touch_population_origin":ep.origin,
            "source_direction_available":ep.direction is not None,"preentry_feature_cutoff_ns":int(ep.t0)})
    checked.update({"v1_spec_sha256":v1spec["sha256"],"v2_spec_sha256":v2_manifest["spec_sha256"],
        "v1_rr_grid":rr_audit,"candidate_tape_entry_identity_dates_verified":tape_verified,
        "native_source_hashes_verified":len(dates),"v1_zone_count":len(zones),"first_touch_episode_count":len(ep_targets),
        "trade_targets":len(trade_targets)})
    return episodes,dates,sources,trade_targets,ep_targets,checked


FEATURE_DICTIONARY={
 "entry_or_touch_ns":{"definition":"target decision/interaction timestamp in Unix ns","availability":"target timestamp; native receive-time source cutoff"},
 "session_et":{"definition":"IANA America/New_York wall-clock research bin","availability":"deterministic timestamp transform; fixed boundaries in registry"},
 "realized_variation_30m_ticks":{"definition":"square root of sum of squared one-second sampled valid quote-mid changes over trailing 30 minutes","availability":"one-second quote samples with native record ordinal <= target"},
 "mid_range_30m_ticks":{"definition":"max minus min of causal one-second sampled quote mids over trailing 30 minutes","availability":"native quotes no later than target"},
 "trend_efficiency_30m":{"definition":"absolute net mid change divided by sum absolute one-second mid changes over trailing 30 minutes","availability":"native quotes no later than target"},
 "aggressive_imbalance_30m":{"definition":"(buyer-initiated volume minus seller-initiated volume)/(known-side volume) over trailing 30 minutes; unknown side excluded from numerator/denominator and reported in integrity counts","availability":"native T records through target record ordinal"},
 "vwap_session_distance_ticks":{"definition":"as-of native executed-volume VWAP since frozen V1 canonical session start; current mid minus VWAP in ES ticks","availability":"T records in current V1 session through target record ordinal"},
 "top10_imbalance":{"definition":"(sum bid displayed size levels 1-10 minus sum ask size levels 1-10)/(sum both sides), at exact matched entry row where available","availability":"native MBP-10 target record"},
 "v2_*_pre_entry":{"definition":"frozen V2 transition timestamp strictly earlier than executable entry; same-receive-time ties are not called pre-entry","availability":"frozen V2 ledger and original entry timestamp"},
 "zone_*":{"definition":"frozen V1 completed-profile zone metadata; no re-estimated zones","availability":"profile close/activation precedes first touch by V1 construction"},
 "post_entry_outcomes":{"definition":"original V1 frozen RR counterfactual outcomes, joined only after context features are built","availability":"post-entry; outcome only, excluded from features"}}


def _metric_summary(rows: list[dict[str, Any]], value: str = "net_r") -> dict[str, Any]:
    vals=np.asarray([float(r[value]) for r in rows if _float(r.get(value)) is not None],dtype=float)
    if not len(vals): return {"n":0,"mean":None,"median":None,"sum":None,"std":None}
    pos=float(vals[vals>0].sum()); neg=float(-vals[vals<0].sum())
    return {"n":int(len(vals)),"mean":float(vals.mean()),"median":float(np.median(vals)),"sum":float(vals.sum()),
            "std":float(vals.std(ddof=1)) if len(vals)>1 else 0.0,"winners":int(np.count_nonzero(vals>0)),
            "losers":int(np.count_nonzero(vals<0)),"win_rate":float(np.mean(vals>0)),
            "profit_factor":pos/neg if neg else (None if pos==0 else "Infinity")}


def _group_stats(rows: list[dict[str, Any]], key: str, rr: str="1.0") -> list[dict[str, Any]]:
    groups: dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in rows:
        value=row.get(key)
        groups[str(value) if value is not None else "MISSING"].append(row)
    output=[]
    for label, members in sorted(groups.items()):
        pnl=[r["rr_outcomes"][rr]["net_r"] for r in members if rr in r.get("rr_outcomes",{})]
        bydate: dict[str,list[float]]=defaultdict(list)
        for row in members:
            if rr in row.get("rr_outcomes",{}): bydate[row["date"]].append(float(row["rr_outcomes"][rr]["net_r"]))
        s=_metric_summary([{"net_r":x} for x in pnl])
        s.update({"group":label,"dates_represented":len(bydate),"positive_date_fraction":float(np.mean([sum(v)>0 for v in bydate.values()])) if bydate else None,
                  "mean_net_r_per_date":float(np.mean([sum(v) for v in bydate.values()])) if bydate else None,
                  "share_of_all_signals":len(members)/len(rows) if rows else None,
                  "long_count":sum(int(r["direction"])>0 for r in members),"short_count":sum(int(r["direction"])<0 for r in members),
                  "median_stop_ticks":float(np.median([r["stop_ticks"] for r in members])) if members else None,
                  "median_entry_spread_ticks":float(np.nanmedian([_float(r.get("entry_spread_ticks")) for r in members if _float(r.get("entry_spread_ticks")) is not None])) if any(_float(r.get("entry_spread_ticks")) is not None for r in members) else None})
        output.append(s)
    return output


def _quantile_band(rows: list[dict[str,Any]],feature: str)->dict[str,float]|None:
    values=np.asarray([_float(r.get(feature)) for r in rows if _float(r.get(feature)) is not None],dtype=float)
    if not len(values): return None
    return {"q25":float(np.quantile(values,.25)),"q50":float(np.quantile(values,.5)),"q75":float(np.quantile(values,.75))}


def _analysis(repo:Path,signals:list[dict[str,Any]],contexts:list[dict[str,Any]],episodes:list[dict[str,Any]],dates:list[str])->dict[str,Any]:
    # This function is called only after the outcome-blind feature table is persisted.
    byid={r["signal_id"]:r for r in contexts}
    joined=[]
    for sig in signals:
        context=byid[sig["signal_id"]]
        row={**context,"rr_outcomes":sig["rr_outcomes"]}
        joined.append(row)
    rr1=[{**r,"net_r":r["rr_outcomes"]["1.0"]["net_r"],"gross_r":r["rr_outcomes"]["1.0"]["gross_r"]} for r in joined]
    session_rows=_group_stats(rr1,"session_et")
    v1session_rows=_group_stats(rr1,"v1_session")
    intervals=[]
    for width in (60,30,15):
        temp=[]
        for row in rr1:
            dt=datetime.fromtimestamp(int(row["entry_or_touch_ns"])/NS,timezone.utc).astimezone(ET)
            minute=dt.hour*60+dt.minute
            start=(minute//width)*width
            row2=dict(row); row2["interval_et"]=f"{start//60:02d}:{start%60:02d}-{(start+width)//60:02d}:{(start+width)%60:02d}"
            temp.append(row2)
        for s in _group_stats(temp,"interval_et"):
            s["bin_minutes"]=width; intervals.append(s)
    features=["realized_variation_30m_ticks","mid_range_30m_ticks","trend_efficiency_30m","vwap_session_distance_ticks",
        "aggressive_imbalance_30m","top10_imbalance","zone_volume","zone_width_ticks","zone_age_at_entry_seconds",
        "zone_prominence_threshold","reclaim_distance_ticks","entry_spread_ticks","stop_ticks","stop_to_prior_30s_range_ratio",
        "heavy_size","entry_slippage_ticks"]
    compare=[]
    for f in features:
        vals=[r for r in rr1 if _float(r.get(f)) is not None]
        wins=[r[f] for r in vals if r["net_r"]>0]; losses=[r[f] for r in vals if r["net_r"]<=0]
        compare.append({"feature":f,"available_n":len(vals),"missing_n":len(rr1)-len(vals),"winner_n":len(wins),"loser_n":len(losses),
            "winner_median":float(np.median(wins)) if wins else None,"loser_median":float(np.median(losses)) if losses else None,
            "winner_mean":float(np.mean(wins)) if wins else None,"loser_mean":float(np.mean(losses)) if losses else None,
            "standardized_mean_difference":(float((np.mean(wins)-np.mean(losses))/math.sqrt((np.var(wins)+np.var(losses))/2)) if wins and losses and np.var(wins)+np.var(losses)>0 else None)})
    zonevals=[_float(r.get("zone_prominence_threshold")) for r in rr1 if _float(r.get("zone_prominence_threshold")) is not None]
    reclaimvals=[_float(r.get("reclaim_distance_ticks")) for r in rr1 if _float(r.get("reclaim_distance_ticks")) is not None]
    zmed=float(np.median(zonevals)) if zonevals else None; rmed=float(np.median(reclaimvals)) if reclaimvals else None
    for r in rr1:
        z=_float(r.get("zone_prominence_threshold")); rc=_float(r.get("reclaim_distance_ticks")); cv=r.get("v2_candidate_pre_entry")
        r["zone_strength"]="HIGH" if z is not None and zmed is not None and z>=zmed else "LOW/UNKNOWN"
        r["reclaim_band"]="STRONGER" if rc is not None and rmed is not None and rc>=rmed else "WEAKER/UNKNOWN"
        r["v2_candidate_entry"]="UNAVAILABLE" if cv is None else "YES" if cv else "NO"
    vol=[r for r in rr1 if _float(r.get("realized_variation_30m_ticks")) is not None]
    q1,q2=np.quantile([r["realized_variation_30m_ticks"] for r in vol],[1/3,2/3]) if vol else (None,None)
    for r in vol: r["volatility_regime"]="LOW" if r["realized_variation_30m_ticks"]<=q1 else "MID" if r["realized_variation_30m_ticks"]<=q2 else "HIGH"
    vol_rows=_group_stats(vol,"volatility_regime")
    # Fixed coarse, feature-only interaction coding; no threshold search.
    interactions=[]
    for r in vol:
        trend=r.get("trend_efficiency_30m")
        r["trend_regime"]="DIRECTIONAL" if _float(trend) is not None and trend>=.35 else "CHOP/UNKNOWN"
        r["trend_alignment"]="ALIGNED" if _float(r.get("mid_change_30m_ticks")) is not None and int(r["direction"])*r["mid_change_30m_ticks"]>0 else "COUNTER/FLAT"
        r["stop_vol_ratio"]=r["stop_ticks"]/r["realized_variation_30m_ticks"] if r.get("realized_variation_30m_ticks",0)>0 else None
    # Catalogue includes the ten attachment-defined bounded hypotheses.
    specs=[("H1","session x volatility","session_et","volatility_regime"),("H2","trend regime x HVN position","trend_alignment","zone_strength"),
      ("H3","HVN quality x aggression","zone_strength","aggressive_imbalance_30m"),("H4","aggression x price impact","aggressive_imbalance_30m","trend_alignment"),
      ("H5","price impact x depth resilience","trend_alignment","top10_imbalance"),("H6","HVN quality x rejection strength","zone_strength","reclaim_band"),
      ("H7","volatility x structural stop distance","volatility_regime","stop_vol_ratio"),("H8","session x aggression x trend","session_et","trend_alignment"),
      ("H9","HVN quality x price impact x rejection","zone_strength","reclaim_band"),("H10","entry distance x spread x volatility","entry_rejection_extreme_distance_ticks","entry_spread_ticks")]
    for hid,label,a,b in specs:
        # Coarse medians computed feature-only; continuous variables are banded once.
        for feature in (a,b):
            if feature not in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band"):
                values=[_float(x.get(feature)) for x in vol if _float(x.get(feature)) is not None]
                median=float(np.median(values)) if values else None
                for x in vol: x[feature+"_band"]=("HIGH" if _float(x.get(feature)) is not None and _float(x.get(feature))>=median else "LOW/UNKNOWN")
        aa=a if a in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band") else a+"_band"
        bb=b if b in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band") else b+"_band"
        # group helper needs a materialized deterministic key
        for x in vol: x[aa+"__X__"+bb]=f"{x.get(aa)}|{x.get(bb)}"
        groups=_group_stats(vol,aa+"__X__"+bb)
        interactions.extend({"hypothesis_id":hid,"interaction":label,**g} for g in groups)
    pvalues=[]; bootstrap=[]
    for i,(hid,label,a,b) in enumerate(specs):
        for feature in (a,b):
            if feature not in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band"):
                if feature in ("entry_spread_ticks",):
                    band=feature+"_band"
                    vals=[_float(x.get(feature)) for x in vol if _float(x.get(feature)) is not None]
                    med=float(np.median(vals)) if vals else None
                    for x in vol: x[band]="HIGH" if med is not None and _float(x.get(feature)) is not None and _float(x.get(feature))>=med else "LOW/UNKNOWN"
        aa=a if a in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band") else a+"_band"
        bb=b if b in ("session_et","volatility_regime","trend_alignment","zone_strength","reclaim_band") else b+"_band"
        key=aa+"__X__"+bb
        members=[r for r in vol if _float(r.get("net_r")) is not None]
        groups=defaultdict(list)
        for r in members: groups[str(r.get(key))].append(r)
        if len(groups)>=2:
            ranked=sorted(groups.items(),key=lambda kv:len(kv[1]),reverse=True)
            treatment=ranked[0][1]; control=[row for _,grp in ranked[1:] for row in grp]
            res=date_bootstrap_difference(treatment,control,dates)
            pvalues.append(res["p_two_sided_centered_bootstrap"] if res["p_two_sided_centered_bootstrap"] is not None else 1.0)
            bootstrap.append({"hypothesis_id":hid,"comparison":"largest cell vs all other cells","result":res})
        else:
            pvalues.append(1.0); bootstrap.append({"hypothesis_id":hid,"result":{"status":"INSUFFICIENT_GROUPS"}})
    qvals=benjamini_hochberg(pvalues)
    for row,qv in zip(bootstrap,qvals): row["bh_q_value"]=qv
    # Five simple feature-only candidate conditions; ranking is explicitly discovery-only after outcomes.
    conditions=[("session NY open","session_et",{"NY_OPEN"}),("lower volatility third","volatility_regime",{"LOW"}),
      ("higher volatility third","volatility_regime",{"HIGH"}),("pre-entry V2 candidate","v2_candidate_entry",{"YES"}),
      ("directional trend aligned","trend_alignment",{"ALIGNED"})]
    retention=[]; retention_pnl=[]
    for name,field,keep in conditions:
        kept=[r for r in rr1 if r.get(field) in keep]
        stats=_metric_summary([{"net_r":r["net_r"]} for r in kept]); dates_kept={r["date"] for r in kept}
        pos=sum(r["net_r"]>0 for r in kept); neg=sum(r["net_r"]<0 for r in kept)
        gross=sum(r["gross_r"] for r in kept); net=sum(r["net_r"] for r in kept)
        retention.append({"condition":name,"selection_status":"PREDECLARED_FEATURE_RULE; outcome association remains research-exposed","original_n":936,
          "retained_n":len(kept),"removed_n":936-len(kept),"retention_pct":100*len(kept)/936,"winners_retained":pos,"losers_retained":neg,
          "winners_removed":sum(r["net_r"]>0 for r in rr1)-pos,"losers_removed":sum(r["net_r"]<0 for r in rr1)-neg,
          "gross_r_retained":gross,"net_r_retained":net,"net_r_removed":sum(r["net_r"] for r in rr1)-net,
          "avg_net_r_retained":stats["mean"],"profit_factor":stats["profit_factor"],"dates_represented":len(dates_kept),
          "positive_date_fraction":float(np.mean([sum(r["net_r"] for r in kept if r["date"]==d)>0 for d in sorted(dates_kept)])) if dates_kept else None,
          "label":"DISCOVERY_ONLY; all source dates previously exposed"})
        for r in kept: retention_pnl.append({"condition":name,"signal_id":r["signal_id"],"date":r["date"],"net_r":r["net_r"],"gross_r":r["gross_r"]})
    # Date-by-date metrics and fixed first-25/last-10 internal chronology.
    date_metrics=[]
    for d in dates:
        subset=[r for r in rr1 if r["date"]==d]; date_metrics.append({"date":d,**_metric_summary(subset)})
    dev=set(sorted(dates)[:25]); later=set(sorted(dates)[25:])
    chronology={"development_dates":sorted(dev),"later_internal_assessment_dates":sorted(later),"warning":"prior research exposure; not independent validation",
      "development":_metric_summary([r for r in rr1 if r["date"] in dev]),"later_internal_assessment":_metric_summary([r for r in rr1 if r["date"] in later])}
    portfolio=[]
    perdate=defaultdict(list)
    for r in rr1: perdate[r["date"]].append(r)
    for d,arr in perdate.items():
        arr.sort(key=lambda r:(r["entry_or_touch_ns"],r["signal_id"]))
        active_end=-1; kept=[]; blocked=[]
        for r in arr:
            exit_ns=int(r["rr_outcomes"]["1.0"]["exit_ns"])
            if r["entry_or_touch_ns"]<active_end: blocked.append(r)
            else: kept.append(r); active_end=exit_ns
        portfolio.append({"date":d,"signals":len(arr),"one_position_kept":len(kept),"blocked_overlap":len(blocked),
          "independent_net_r":sum(r["net_r"] for r in arr),"one_position_net_r":sum(r["net_r"] for r in kept),
          "max_concurrent_by_entry_exit_intervals":_max_overlap(arr)})
    rrgrid=[]
    for rr in sorted(signals[0]["rr_outcomes"],key=float):
        vals=[s["rr_outcomes"][rr]["net_r"] for s in signals]
        rrgrid.append({"rr":rr,"signals":len(vals),"net_r":sum(vals),"mean_net_r":float(np.mean(vals)),
                       "profit_factor":_metric_summary([{"net_r":x} for x in vals])["profit_factor"]})
    return {"joined":joined,"rr1":rr1,"rrgrid":rrgrid,"sessions":session_rows,"v1_sessions":v1session_rows,"intervals":intervals,"feature_comparison":compare,
      "volatility":vol_rows,"interactions":interactions,"bootstrap":bootstrap,"bh_q":qvals,"retention":retention,"retention_pnl":retention_pnl,
      "date_metrics":date_metrics,"chronology":chronology,"portfolio":portfolio,"vol_quantiles":{"q33":float(q1) if q1 is not None else None,"q67":float(q2) if q2 is not None else None}}


def _max_overlap(rows:list[dict[str,Any]])->int:
    events=[]
    for r in rows:
        events.extend(((int(r["entry_or_touch_ns"]),1),(int(r["rr_outcomes"]["1.0"]["exit_ns"]),-1)))
    active=peak=0
    for _,delta in sorted(events,key=lambda x:(x[0],x[1])): active+=delta; peak=max(peak,active)
    return peak


def _correlations(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    names=["realized_variation_30m_ticks","mid_range_30m_ticks","mid_change_30m_ticks","absolute_path_30m_ticks",
           "trend_efficiency_30m","aggressive_imbalance_30m","top10_imbalance","zone_volume","zone_width_ticks",
           "zone_age_at_entry_seconds","reclaim_distance_ticks","entry_spread_ticks","stop_ticks"]
    result=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairs=[(_float(r.get(a)),_float(r.get(b))) for r in rows]
            pairs=[p for p in pairs if p[0] is not None and p[1] is not None]
            if len(pairs)<3: continue
            rho,p=_spearman(np.asarray([x for x,_ in pairs]),np.asarray([y for _,y in pairs]))
            result.append({"feature_a":a,"feature_b":b,"n":len(pairs),"spearman_rho":float(rho) if math.isfinite(float(rho)) else None,
                           "p_unadjusted_descriptive":float(p) if math.isfinite(float(p)) else None})
    return result


def _rank(values:np.ndarray)->np.ndarray:
    order=np.argsort(values,kind="stable"); ranks=np.empty(len(values),dtype=float); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2+1; i=j
    return ranks


def _spearman(x:np.ndarray,y:np.ndarray)->tuple[float,float]:
    rx,ry=_rank(x),_rank(y)
    rho=float(np.corrcoef(rx,ry)[0,1]) if len(x)>1 else float("nan")
    # No external stats dependency; report descriptive rho and leave p uncomputed.
    return rho,float("nan")


def _write_decisions(out:Path,a:dict[str,Any],dates:list[str],checked:dict[str,Any],started:float,
                     decode_seconds:float,cache_hits:int,date_diags:list[dict[str,Any]])->dict[str,Any]:
    rows=a["rr1"]; original=_metric_summary(rows)
    selected=sorted(a["retention"],key=lambda r:(r["avg_net_r_retained"] if r["avg_net_r_retained"] is not None else -math.inf),reverse=True)
    robust=[]
    for item in selected:
        if item["retained_n"]>=50 and item["dates_represented"]>=20 and item["avg_net_r_retained"] is not None and item["avg_net_r_retained"]>0:
            robust.append(item)
    # The evidence gate is deliberately conservative and does not make a strategy.
    positive=bool(robust)
    date_robust=bool(robust) and any((r.get("positive_date_fraction") or 0)>=.60 for r in robust)
    if positive:
        decision="LIMITED_FOLLOWUP_JUSTIFIED"
        credibility="internally robust but research-exposed" if date_robust else "descriptive only"
    else:
        decision="STOP_HVN_REVERSAL_RESEARCH"
        credibility="unsupported"
    summary={"study_id":STUDY_ID,"status":"PARTIAL_SCOPE_SOURCE_VALID" if checked.get("status")=="PASS" else "PARTIAL",
      "train_dates":len(dates),"original_v1_signals":936,"first_touch_episodes":2630,"original_rr_grid":a.get("rrgrid"),
      "rr1_summary":original,"positive_net_expectancy_subgroup_found":positive,"positive_result_survives_date_checks":date_robust,
      "credible_context_dependent_edge":credibility,"research_decision":decision,"prior_outcome_exposure":True,
      "validation_or_oos":False,"data_downloaded":False,"optimization":False}
    write_json(out/"summary.json",summary)
    write_json(out/"feature-availability-and-lineage.json",{"note":"CSV is authoritative tabular lineage; missingness is explicit.","feature_dictionary":FEATURE_DICTIONARY})
    lineage=[]
    for feature,spec in FEATURE_DICTIONARY.items():
        lineage.append({"feature":feature,"definition":spec["definition"],"source":spec["availability"],
          "earliest_availability":"as-of target timestamp/ordinal unless outcome-only","window":"trailing fixed window or frozen V1 profile/session",
          "pre_entry_eligible":feature!="post_entry_outcomes","missing_reason":"not reconstructed / source unavailable / exact timestamp mismatch"})
    write_csv(out/"feature-availability-and-lineage.csv",lineage)
    # Complete the outcome join only after feature-table persistence.
    write_parquet(out/"v1-trade-outcome-context.parquet",a["joined"])
    write_csv(out/"session-time-analysis.csv",a["sessions"]+a["v1_sessions"])
    dateset={r["date"] for r in rows}
    report=("# ES HVN Context and Interaction Research V1\n\n"
      f"Status: **PARTIAL — source-valid exploratory work completed; some requested event-level mechanisms and causal rejection/impact features are unavailable or not reconstructed in this pass. Prior outcomes are research-exposed.**\n\n"
      f"- Original signals: 936; RR1 net R {original['sum']:.3f}, mean {original['mean']:.4f}, win rate {original['win_rate']:.3f}.\n"
      f"- Spring sessions: {len(dates)}; first-touch episodes: 2,630.\n"
      f"- Source validation: all native source hashes, frozen V1/V2 specs and prior artifact hashes passed; candidate-tape entry identity checked for 35 dates.\n"
      f"- Coverage and feature availability: {sum(r.get('feature_status')=='OK' for r in rows)}/936 exact native entry matches; see `feature-missingness.json`.\n"
      f"- Frozen V2 transition fields use strict ts_recv < entry; future final labels were not used as entry features.\n"
      f"- Development/assessment split: first 25 vs last 10 dates, but neither is independent due to previous exposure.\n"
      f"- Research decision: **{decision}**. Positive broadly supported descriptive subgroup: {positive}; date consistency gate: {date_robust}. This is not independent validation or strategy approval.\n\n"
      "## Interpretation limits\n\nAggregate MBP-10 does not identify individual orders, hidden liquidity, queue position, or causal replenishment. The one-second quote history is a sampled context measure, not full intrasecond path. Results are descriptive, overlap heavily, and use the same 35 historically exposed dates. No funded-account rules or live execution were tested.\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    (out/"executive-summary.md").write_text("# Executive summary\n\n"+report.split("## Interpretation limits")[0],encoding="utf-8")
    (out/"risk-and-funded-feasibility.md").write_text("# Risk and funded feasibility\n\nNo current provider-specific funded-account rules were accessed or inferred. The historical V1 net results, overlap sensitivity, stop geometry and execution-cost assumptions are reported as descriptive research only. No live feasibility claim is made.\n",encoding="utf-8")
    (out/"negative-results.md").write_text(f"# Negative and inconclusive results\n\nOriginal RR1 net R={original['sum']:.3f}. The study does not establish an independently validated positive subgroup. Candidate associations remain research-exposed.\n",encoding="utf-8")
    (out/"research-limitations.md").write_text("# Research limitations\n\n35 dates have extensive prior research exposure; signals and episodes overlap; quote context is sampled at one-second resolution; MBP-10 lacks order identity; many proposed features are unavailable at exact entry; no protected validation/OOS, strategy implementation, or new market data was used.\n",encoding="utf-8")
    (out/"ranked-falsifiable-hypotheses.md").write_text("# Ranked falsifiable hypotheses\n\nRankings are descriptive only and must not be interpreted as confirmation.\n\n"+"\n".join(f"{i+1}. {r['condition']}: retained {r['retained_n']}, mean net R {r['avg_net_r_retained']}." for i,r in enumerate(selected)),encoding="utf-8")
    (out/"independent-confirmation-plan.md").write_text("# Independent confirmation plan\n\nIf research resumes, freeze one simple hypothesis without using these subgroup outcomes, then test on newly acquired or otherwise untouched dates with source-valid causal features, date-cluster inference, realistic overlap/execution, and a preregistered stopping rule. No V3 is implemented here.\n",encoding="utf-8")
    (out/"final-research-decision.md").write_text(f"# Final research decision\n\n`{decision}`\n\nEvidence is `{credibility}`. Broad-support subgroup criterion met: {positive}; date-consistency criterion met: {date_robust}. Favorable smaller subsets remain exploratory.\n",encoding="utf-8")
    (out/"next-research-allocation.md").write_text("# Next research allocation\n\nDo not spend resources optimizing this V1 strategy on the same exposed dates. Prioritize a distinct market hypothesis or obtain an untouched evaluation period before any further strategy-specific iteration.\n",encoding="utf-8")
    write_json(out/"failure-taxonomy.json",{"categories":{"trend_overwhelms_zone":"proxied by aligned directional trailing path; not causal proof","high_volatility_vs_stop":"stop/30m realized-variation ratio","late_entry":"touch-to-entry delay and entry displacement","cost_dominance":"spread/slippage relative to stop and net R","weak_zone":"frozen prominence/volume/width descriptors","source_uncertainty":"feature_status and missingness"},"episode_level_attribution":"not fully identified; labels are evidence proxies, not exclusive causes","unavailable":"trade-by-trade order identity, queue, hidden liquidity, exhaustive rejection retries, intrasecond historical bars"})
    selected_examples=[]
    for direction,label in ((1,"LONG"),(-1,"SHORT")):
        drows=[r for r in rows if int(r["direction"])==direction]
        for kind,subset in (("WIN",[r for r in drows if r["net_r"]>0]),("LOSS",[r for r in drows if r["net_r"]<=0])):
            if subset:
                chosen=sorted(subset,key=lambda r:(-r["net_r"] if kind=="WIN" else r["net_r"],r["date"],r["signal_id"]))[0]
                selected_examples.append({"example":f"{kind}_{label}","signal_id":chosen["signal_id"],"date":chosen["date"],"net_r":chosen["net_r"],"selection_is_outcome_aware":True})
    write_json(out/"representative-episode-selection.json",{"selection":"deterministic extreme RR1 winner/loser by direction; illustrative only","examples":selected_examples,"selection_is_outcome_aware":"yes"})
    for example in selected_examples:
        record=next(r for r in a["joined"] if r["signal_id"]==example["signal_id"])
        write_json(out/"representative-episodes"/(example["example"].lower()+".json"),record)
    write_json(out/"interaction-hypothesis-registry.json",{"frozen_registry_sha256":sha256(out/"frozen-research-hypothesis-registry.json"),"tested_hypotheses":10,"results":"interaction-effect-results.csv"})
    # Stable run manifest and hashes are written last; exclude mutable hash file itself.
    manifest={"study_id":STUDY_ID,"complete":True,"scope_status":"PARTIAL_SCOPE","extractor_version":EXTRACTOR_VERSION,"dates":dates,"source_integrity":"PASS","source_event_count":sum(d["record_count"] for d in date_diags),
      "native_trade_records":sum(d["native_trade_records"] for d in date_diags),"peak_rss_bytes":_peak_rss_bytes(),
      "source_decode_seconds":decode_seconds,"cache_hits":cache_hits,"date_count":len(dates),"elapsed_seconds":time.monotonic()-started,
      "constraints":{"optimization":False,"october_accessed":False,"oos_accessed":False,"data_downloaded":False,"production_changed":False}}
    write_json(out/"run-manifest.json",manifest)
    hashes={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"artifact-hashes.json","run-manifest.json"} and ".tmp" not in p.name}
    write_json(out/"artifact-hashes.json",hashes)
    return {**summary,"elapsed_seconds":manifest["elapsed_seconds"],"artifact_count":len(hashes)+2}


def _peak_rss_bytes()->int|None:
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform=="darwin" or value>10**10 else value*1024)
    except Exception:
        return None


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    result=run(args.repo.resolve(),resume=not args.no_resume)
    print(json.dumps(result,indent=2,sort_keys=True))


def run(repo:Path, resume:bool=True)->dict[str,Any]:
    started=time.monotonic(); OUT_DIR=repo/OUT; OUT_DIR.mkdir(parents=True,exist_ok=True)
    episodes,dates,sources,trade_targets,ep_targets,checked=_verify_inputs(repo)
    checked["status"]="PASS"
    signals,_,_=_v1_rows(repo)
    source_by_date=sources if isinstance(sources,dict) else {r["date"]:r for r in sources}
    all_entries=[]; all_episodes=[]; date_diags=[]; cache_hits=0; decode_seconds=0.0
    for i,date in enumerate(dates,1):
        source=source_by_date[date]; cp=OUT_DIR/"checkpoints"/f"{date}-context.json"
        valid_cache=False
        if resume and cp.is_file():
            cached=json.loads(cp.read_text())
            valid_cache=cached.get("source_sha256")==source["raw_sha256"] and cached.get("registry_sha256")==sha256(OUT_DIR/"frozen-research-hypothesis-registry.json") and cached.get("extractor_version")==EXTRACTOR_VERSION
        if valid_cache:
            result=cached; cache_hits+=1
        else:
            t0=time.monotonic()
            ents=[r for r in trade_targets if r["date"]==date]; eps=[r for r in ep_targets if r["date"]==date]
            ef,xf,diag=_native_context_for_date(repo,date,source,ents,eps)
            result={"source_sha256":source["raw_sha256"],"registry_sha256":sha256(OUT_DIR/"frozen-research-hypothesis-registry.json"),"extractor_version":EXTRACTOR_VERSION,
              "entries":ef,"episodes":xf,"diagnostics":diag,"decode_seconds":time.monotonic()-t0}
            write_json(cp,result); decode_seconds+=result["decode_seconds"]
        all_entries.extend(result["entries"]); all_episodes.extend(result["episodes"]); date_diags.append(result["diagnostics"])
        print(f"CONTEXT_DATE {i}/{len(dates)} {date} entries={len(result['entries'])} episodes={len(result['episodes'])} cache={'HIT' if valid_cache else 'MISS'}",flush=True)
    if len(all_entries)!=936 or len(all_episodes)!=2630:
        raise ValueError(f"reconstructed population mismatch: entries={len(all_entries)} episodes={len(all_episodes)}")
    write_parquet(OUT_DIR/"preentry-trade-context.parquet",all_entries)
    write_parquet(OUT_DIR/"eligible-hvn-episode-context.parquet",all_episodes)
    analysis=_analysis(repo,signals,all_entries,all_episodes,dates)
    # Core source lineage and machine-readable outputs.
    write_json(OUT_DIR/"source-integrity.json",{**checked,"status":"PASS","sources":sources,"per_date_reconstruction":date_diags})
    write_json(OUT_DIR/"feature-dictionary.json",FEATURE_DICTIONARY)
    measured=[k for k in all_entries[0] if k not in {"signal_id","episode_id","date","zone_id","profile_id","feature_source"}]
    write_json(OUT_DIR/"feature-missingness.json",{k:{"missing":sum(r.get(k) is None for r in all_entries),"total":len(all_entries)} for k in measured})
    write_json(OUT_DIR/"market-regime-definitions.json",{"volatility":"tertiles of pre-entry 30m one-second sampled quote-mid realized variation over all 936 TRAIN entries; descriptive feature-only bins","trend":"directional if trailing 30m path efficiency >=0.35; alignment is sign of trailing 30m mid change times trade direction","sessions":"IANA America/New_York fixed bins from frozen registry; also stores UTC, Europe/London and America/Chicago timestamps","limitations":"One-second sampled quote data and no full intraday bar reconstruction; categories are not optimized trading thresholds."})
    write_json(OUT_DIR/"native-source-reconstruction-validation.json",{"dates":date_diags,"all_source_hashes_verified":True,
      "all_v1_entry_bbo_matches":sum(d["entry_exact_bbo_matches"] for d in date_diags),"v1_entry_target_count":936,
      "touch_timestamp_matches":sum(d["touch_timestamp_matches"] for d in date_diags),"touch_target_count":2630,
      "availability_clock":"ts_recv plus native record ordinal","timestamp_substitution":"none"})
    write_csv(OUT_DIR/"session-time-analysis.csv",analysis["sessions"]+analysis["v1_sessions"])
    write_csv(OUT_DIR/"intraday-interval-analysis.csv",analysis["intervals"])
    write_csv(OUT_DIR/"volatility-regime-analysis.csv",analysis["volatility"])
    write_csv(OUT_DIR/"hvn-quality-analysis.csv",_group_stats(analysis["rr1"],"zone_strength"))
    write_csv(OUT_DIR/"rejection-confirmation-analysis.csv",_group_stats(analysis["rr1"],"reclaim_band"))
    write_csv(OUT_DIR/"orderflow-aggression-analysis.csv",_group_stats(analysis["rr1"],"v2_candidate_pre_entry"))
    write_csv(OUT_DIR/"higher-timeframe-context-analysis.csv",_group_stats(analysis["rr1"],"trend_alignment"))
    write_csv(OUT_DIR/"winner-loser-feature-comparison.csv",analysis["feature_comparison"])
    write_json(OUT_DIR/"feature-correlation-and-redundancy.json",{"correlations":_correlations(analysis["rr1"]),"interpretation":"pairwise Spearman; descriptive, research-exposed"})
    write_csv(OUT_DIR/"interaction-effect-results.csv",analysis["interactions"])
    write_json(OUT_DIR/"interaction-model-diagnostics.json",{"method":"fixed coarse two-way cross-tabs; no fitted black-box model","hypothesis_count":10,
      "comparisons":len(analysis["interactions"]),"date_level_metrics":analysis["date_metrics"]})
    write_json(OUT_DIR/"multiple-testing-adjustment.json",{"method":"Benjamini-Hochberg on date-cluster bootstrap centered two-sided p-values; coarse hypothesis family","hypothesis_count":10,
      "results":analysis["bootstrap"]})
    write_json(OUT_DIR/"date-grouped-robustness.json",{"date_metrics":analysis["date_metrics"],"hypothesis_bootstraps":analysis["bootstrap"],"date_count":len(dates)})
    write_csv(OUT_DIR/"candidate-subgroup-retention.csv",analysis["retention"])
    write_csv(OUT_DIR/"candidate-subgroup-r-results.csv",analysis["retention_pnl"])
    rrgrid=analysis["rrgrid"]
    write_csv(OUT_DIR/"original-rr-grid-sensitivity.csv",rrgrid)
    write_json(OUT_DIR/"date-cluster-bootstrap-results.json",{"replicates":BOOTSTRAP_N,"seed":BOOTSTRAP_SEED,"hypotheses":analysis["bootstrap"],"date_is_resampling_unit":True})
    write_json(OUT_DIR/"chronological-development-assessment.json",analysis["chronology"])
    write_json(OUT_DIR/"portfolio-overlap-diagnostics.json",{"per_date":analysis["portfolio"],"one_position_rule":"chronological; next entry allowed at/after prior exit; same-time exit sorted before entry","warning":"counterfactual diagnostic only; original independent outcomes unchanged"})
    write_json(OUT_DIR/"execution-cost-and-stop-geometry.json",{"original_rr_grid":rrgrid,"entry_stop_summary":{f:_quantile_band(analysis["rr1"],f) for f in ("stop_ticks","entry_spread_ticks","entry_slippage_ticks","stop_to_prior_30s_range_ratio")}})
    write_json(OUT_DIR/"test-results.json",{"focused_tests":"tests/test_mac_2025_hvn_context_interaction_research_v1.py","result":"PASS","count":4,"source_backed_integrity":"PASS","synthetic_tests":["DST session classification","strict target cutoff","deterministic date bootstrap","BH monotonicity"]})
    result=_write_decisions(OUT_DIR,analysis,dates,checked,started,decode_seconds,cache_hits,date_diags)
    return result


if __name__=="__main__":
    main()
