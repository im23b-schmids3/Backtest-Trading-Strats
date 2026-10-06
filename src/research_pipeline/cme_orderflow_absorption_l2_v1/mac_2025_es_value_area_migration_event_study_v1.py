"""Frozen ES value-area migration and order-flow transfer event study.

Spring 2025 is discovery and October 2025 is secondary DEV compatibility.
This module performs no optimization and does not implement a strategy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_es_flow_momentum_v1 as flow
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_structural_breakout_l2_v1 as structural
from . import asia_w04_structural_matrix as profile_module

RUN_ID = "CMEOrderflow_ES_VALUE_AREA_MIGRATION_ORDERFLOW_EVENT_STUDY_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
TICK = 0.25
BAR_NS = 300_000_000_000
ENTRY_DELAY_NS = 2_000_000
HORIZONS_MS = (250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 120000, 300000)
EXCURSION_MS = (1000, 2000, 5000, 10000, 30000, 60000, 120000, 300000)
BARRIERS = ((2, 2), (4, 4), (8, 4), (8, 8), (12, 6), (16, 8))
TOD_BUCKETS = ((9.5, 10.5, "09:30-10:30"), (10.5, 12, "10:30-12:00"),
               (12, 14, "12:00-14:00"), (14, 15.5, "14:00-15:30"),
               (15.5, 16, "15:30-16:00"))
CHECKPOINT_VERSION = "es-value-area-migration-date-v1"

CONFIG = {
    "study_version": 1,
    "source": "existing sealed canonical native ES GLBX.MDP3 mbp-10 tape + source-bound compact native TOP5; no MES/MBO",
    "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
    "roles": {"SPRING_2025": "PRIMARY_DEV_DISCOVERY", "OCTOBER_2025": "SECONDARY_DEV_COMPATIBILITY"},
    "rth": "exchange-aware NY session 09:30 ET inclusive to 16:00 ET exclusive; use source session boundary/early close",
    "bars": "fixed 5 minute bins anchored at RTH open; actual trades only; incomplete final short bar retained only for an exchange early close",
    "profile": "per-bar executed volume by integer ES tick; canonical asia_volume_profile 70%; deterministic lower-tick POC tie and adjacent expansion tie",
    "migration": "POC shift from previous completed populated 5m bar in ticks; sign only; flat is zero; missing-bar gap resets preceding sign/streak",
    "event": "onset when sign is nonzero and differs from previous migration sign; subsequent same-sign bars are secondary ordinals; first completed bar close is event time",
    "value_area_alignment": "up iff POC increases and VAH/VAL do not decrease; down iff POC decreases and VAH/VAL do not increase",
    "delta": "aggressor-coded executed volume per completed 5m bar; BUY positive, SELL negative; neutral when exact zero",
    "trend_efficiency": "validated ES ER_30S semantics: absolute 30s as-of midpoint displacement / sum absolute 500ms-grid midpoint changes, causal through bar close",
    "effort_proxy": "three separately reported measures: directional close-open ticks per total aggressive volume; forward-side TOP5 depth relative to strictly prior-date median; directional TOP5 normalized MLOFI; support sign rules only",
    "support_cutoffs": "per-date effort depth reference is direction-specific strictly prior-date median; no date self-calibration; impact>0 and MLOFI>0 supportive; forward depth below prior median supportive",
    "reload": "one touch per continuous directional POC migration episode; latest completed same-direction migration bar VA active only after bar close; first actual trade in that area before migration reversal",
    "raw_anchor": "first valid quote strictly after signal timestamp; raw horizon measured from scheduled event timestamp; executable quote at/after event+2ms; long ask / short bid; entry adverse fill +1/-1 ES tick; exit long bid / short ask",
    "path_horizons_ms": list(HORIZONS_MS), "excursion_horizons_ms": list(EXCURSION_MS),
    "barriers_favorable_adverse_ticks": [list(x) for x in BARRIERS],
    "price_control": "nearest same date-period directional non-onset candidate bars on price-only fields (recent 5m return, velocity, ER30, RV30, TOD), fixed scales; exclude target event; no orderflow/value fields",
    "feature_groups": "Spring tercile cutpoints frozen then applied to October; cell N<20 is insufficient",
    "permutation": "1000 fixed-seed shuffles within period x direction x TOD strata where cell size permits; no parameter tuning",
    "no_optimization": True, "no_strategy_pnl": True, "no_passive_fill": True, "no_final_oos": True,
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


CONFIG_SHA256 = _canonical_hash(CONFIG)
STUDY_SHA256 = _canonical_hash({"config_sha256": CONFIG_SHA256, "checkpoint_version": CHECKPOINT_VERSION,
                               "profile_function": "asia_volume_profile", "execution_function": "structural._path_analysis-compatible"})


class MigrationStudyError(RuntimeError):
    pass


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    a = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "trimmed_mean": None, "p25": None,
                "p75": None, "positive_fraction": None, "negative_fraction": None}
    s = np.sort(a); k = int(.1 * len(a)); trimmed = s[k:len(a)-k] if len(a)-2*k else s
    return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "trimmed_mean": float(np.mean(trimmed)), "p25": float(np.quantile(a, .25)),
            "p75": float(np.quantile(a, .75)), "positive_fraction": float(np.mean(a > 0)),
            "negative_fraction": float(np.mean(a < 0))}


def _json_write(path: Path, value: Any) -> None:
    native._write_json(path, value)


def _gzip_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    native._write_gzip_jsonl(path, rows)


def _session(day: str) -> tuple[int, int]:
    start, end = native.baseline._session_windows(day)["NY"] if hasattr(native, "baseline") else flow.baseline._session_windows(day)["NY"]
    return int(start), int(end)


def _period(day: str) -> str:
    return "SPRING_2025" if day in flow.SPRING_DATES else "OCTOBER_2025"


def _tod_bucket(ns: int) -> str:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    h = dt.hour + dt.minute / 60
    return next((name for lo, hi, name in TOD_BUCKETS if lo <= h < hi), "OUTSIDE_RTH")


def _mid_at(tape: np.ndarray, target: int, *, strict: bool = False, close: int | None = None) -> tuple[int, float] | None:
    ts = tape["timestamp_ns"]
    i = int(np.searchsorted(ts, target, side="right" if strict else "left"))
    while i < len(tape) and (close is None or ts[i] < close):
        b, a = float(tape["bid"][i]), float(tape["ask"][i])
        if math.isfinite(b) and math.isfinite(a) and a > b:
            return i, (b + a) / 2
        i += 1
    return None


def _mid_asof(tape: np.ndarray, target: int, *, start: int | None = None) -> tuple[int, float] | None:
    """Last valid quote at or before target; never fills a historical grid from the future."""
    ts=tape["timestamp_ns"]; i=int(np.searchsorted(ts,target,side="right")-1)
    lower=int(np.searchsorted(ts,start,side="left")) if start is not None else 0
    while i>=lower:
        b,a=float(tape["bid"][i]),float(tape["ask"][i])
        if math.isfinite(b) and math.isfinite(a) and a>b:return i,(a+b)/2
        i-=1
    return None


def _mid_asof_grid(tape: np.ndarray, targets: np.ndarray, *, start: int) -> np.ndarray:
    """Vectorized last-valid-quote lookup for causal fixed-time grids."""
    ts=tape["timestamp_ns"]
    valid=np.isfinite(tape["bid"])&np.isfinite(tape["ask"])&(tape["ask"]>tape["bid"])&(ts>=start)
    ids=np.flatnonzero(valid)
    if not len(ids):return np.full(len(targets),np.nan)
    qts=ts[ids]; locations=np.searchsorted(qts,targets,side="right")-1
    result=np.full(len(targets),np.nan); ok=locations>=0
    qids=ids[locations[ok]]
    result[ok]=(tape["bid"][qids].astype(float)+tape["ask"][qids].astype(float))/2
    return result


def _trade_rows(tape: np.ndarray, start: int, end: int) -> np.ndarray:
    ts = tape["timestamp_ns"]
    return np.flatnonzero((ts >= start) & (ts < end) & (tape["execution_size"] > 0)
                          & np.isfinite(tape["execution_price"]))


def _profile70(volume_by_tick: Mapping[int,int]) -> dict[str,float]:
    """Canonical 70% adjacent-tick expansion with finite observed-price bounds.

    The shared historical helper walks away indefinitely when both adjacent
    levels are empty beyond an observed edge. This preserves its stated tie
    semantics while bounding expansion to the actual profile range.
    """
    volume={int(k):int(v) for k,v in volume_by_tick.items() if int(v)>0}
    if not volume:raise MigrationStudyError("cannot profile a 5m bar without executed volume")
    lo_tick=min(volume);hi_tick=max(volume)
    poc=min(volume,key=lambda k:(-volume[k],k));low=high=poc
    included=volume[poc];total=sum(volume.values())
    while included*100<total*70:
        has_below=low>lo_tick;has_above=high<hi_tick
        if not has_below and not has_above:break
        below=volume.get(low-1,0) if has_below else -1
        above=volume.get(high+1,0) if has_above else -1
        if has_below and (not has_above or below>=above):
            low-=1;included+=max(0,below)
        else:
            high+=1;included+=max(0,above)
    return {"poc":poc*TICK,"vah":high*TICK,"val":low*TICK,
            "high":hi_tick*TICK,"low":lo_tick*TICK}


def _make_bars(day: str, tape: np.ndarray, compact: np.ndarray) -> list[dict[str, Any]]:
    start, close = _session(day)
    if close <= start:
        raise MigrationStudyError(f"invalid exchange RTH bounds: {day}")
    trade_ids = _trade_rows(tape, start, close)
    if not len(trade_ids):
        raise MigrationStudyError(f"no RTH executions: {day}")
    trade_ts = tape["timestamp_ns"][trade_ids]
    prices = tape["execution_price"][trade_ids]
    sizes = tape["execution_size"][trade_ids].astype(np.int64)
    aggr = tape["aggressor"][trade_ids].astype(np.int8)
    bars: list[dict[str, Any]] = []
    n_bins = int(math.ceil((close - start) / BAR_NS))
    cts = np.asarray(compact["ts"], dtype=np.int64)
    for j in range(n_bins):
        lo = start + j * BAR_NS; hi = min(lo + BAR_NS, close)
        a = int(np.searchsorted(trade_ts, lo, side="left")); b = int(np.searchsorted(trade_ts, hi, side="left"))
        if b <= a:
            bars.append({"date": day, "period": _period(day), "bar_index": j, "start_ns": lo, "close_ns": hi,
                         "status": "NO_TRADES", "poc": None, "vah": None, "val": None})
            continue
        p = prices[a:b]; q = sizes[a:b]; ag = aggr[a:b]
        tick_prices = np.rint(p / TICK).astype(np.int64)
        if np.any(np.abs(tick_prices * TICK - p) > 1e-8) or np.any(q <= 0):
            raise MigrationStudyError(f"invalid/off-tick trade in {day} bar {j}")
        unq, inv = np.unique(tick_prices, return_inverse=True)
        vol = np.bincount(inv, weights=q).astype(np.int64)
        profile = _profile70({int(t): int(v) for t, v in zip(unq, vol)})
        buy = int(np.sum(q[ag == 1])); sell = int(np.sum(q[ag == -1])); total_ag = buy + sell
        depth_ix = int(np.searchsorted(cts, hi, side="left") - 1)
        if depth_ix < 0:
            depth = {"bid5": None, "ask5": None, "mlofi": None, "denom": None}
        else:
            row = compact[depth_ix]
            # Aggregate inverse-rank MLOFI over the completed bar; denominator is
            # the contemporaneous weighted mean top-five depth at the endpoint.
            x = int(np.searchsorted(cts, lo, side="left")); y = int(np.searchsorted(cts, hi, side="left"))
            denom = float(row["denom"])
            depth = {"bid5": float(row["bid5"]), "ask5": float(row["ask5"]),
                     "mlofi": float(np.sum(compact["mlofi"][x:y], dtype=np.float64) / denom) if denom > 0 else None,
                     "denom": denom}
        mid0 = _mid_at(tape, hi, strict=True, close=close)
        mid_anchor = mid0[1] if mid0 else None
        bar = {"date": day, "period": _period(day), "bar_index": j, "start_ns": int(lo), "close_ns": int(hi),
               "status": "COMPLETE", "open": float(p[0]), "high": float(np.max(p)), "low": float(np.min(p)),
               "close": float(p[-1]), "poc": float(profile["poc"]), "vah": float(profile["vah"]),
               "val": float(profile["val"]), "value_center": (float(profile["vah"])+float(profile["val"]))/2,
               "total_volume": int(np.sum(q)), "buy_volume": buy, "sell_volume": sell,
               "aggressive_volume": total_ag, "delta": buy-sell,
               "delta_ratio": float((buy-sell)/(total_ag+1e-12)) if total_ag else None,
               "price_volume_by_tick": {str(int(t)): int(v) for t, v in zip(unq, vol)},
               "trade_count": int(len(p)), "bar_mid_at_close": mid_anchor, **depth}
        if bars and bars[-1]["status"] == "COMPLETE":
            prev = bars[-1]
            bar["poc_shift_ticks"] = (bar["poc"] - prev["poc"]) / TICK
            bar["vah_shift_ticks"] = (bar["vah"] - prev["vah"]) / TICK
            bar["val_shift_ticks"] = (bar["val"] - prev["val"]) / TICK
            bar["value_center_shift_ticks"] = (bar["value_center"] - prev["value_center"]) / TICK
            sh = bar["poc_shift_ticks"]
            bar["migration"] = "UP" if sh > 0 else "DOWN" if sh < 0 else "FLAT"
            bar["full_up_alignment"] = bool(sh > 0 and bar["vah_shift_ticks"] >= 0 and bar["val_shift_ticks"] >= 0)
            bar["full_down_alignment"] = bool(sh < 0 and bar["vah_shift_ticks"] <= 0 and bar["val_shift_ticks"] <= 0)
            prevdir = prev.get("migration")
            if prevdir == bar["migration"] and bar["migration"] in ("UP", "DOWN"):
                bar["migration_streak"] = int(prev.get("migration_streak", 0)) + 1
            else:
                bar["migration_streak"] = 1 if bar["migration"] in ("UP", "DOWN") else 0
            bar["drift_bar_ordinal"] = bar["migration_streak"] if bar["migration"] in ("UP", "DOWN") else 0
        else:
            bar.update({"poc_shift_ticks": None, "vah_shift_ticks": None, "val_shift_ticks": None,
                        "value_center_shift_ticks": None, "migration": "UNSEEDED", "migration_streak": 0,
                        "drift_bar_ordinal": 0, "full_up_alignment": False, "full_down_alignment": False})
        # If a no-trade bin intervened, do not infer adjacent migration across it.
        if bars and bars[-1]["status"] == "NO_TRADES":
            bar.update({"poc_shift_ticks": None, "vah_shift_ticks": None, "val_shift_ticks": None,
                        "value_center_shift_ticks": None, "migration": "UNSEEDED", "migration_streak": 0,
                        "drift_bar_ordinal": 0, "full_up_alignment": False, "full_down_alignment": False})
        bars.append(bar)
    # Causal 30s ER and price-only context are calculated at completed bar time.
    valid_quotes=np.isfinite(tape["bid"])&np.isfinite(tape["ask"])&(tape["ask"]>tape["bid"])&(tape["timestamp_ns"]>=start)
    quote_ids=np.flatnonzero(valid_quotes); quote_ts=tape["timestamp_ns"][quote_ids]
    quote_mids=(tape["bid"][quote_ids].astype(float)+tape["ask"][quote_ids].astype(float))/2
    for bar in bars:
        if bar["status"] != "COMPLETE":
            continue
        t = int(bar["close_ns"])
        grid = np.arange(t-30_000_000_000, t+1, 500_000_000, dtype=np.int64)
        locations=np.searchsorted(quote_ts,grid,side="right")-1
        vals=quote_mids[locations] if len(locations) and np.all(locations>=0) else np.asarray([])
        if len(vals) >= 2 and np.all(np.isfinite(vals)):
            path = np.abs(np.diff(vals)); traveled = float(np.sum(path))
            move = float(vals[-1]-vals[0])
            bar["trend_efficiency_30s"] = abs(move)/traveled if traveled else 0.0
            bar["directional_trend_efficiency_30s"] = (1 if move>0 else -1 if move<0 else 0)*bar["trend_efficiency_30s"]
            bar["rv_30s_ticks"] = float(np.sqrt(np.sum((np.diff(vals)/TICK)**2)))
        else:
            bar.update({"trend_efficiency_30s": None, "directional_trend_efficiency_30s": None,
                        "rv_30s_ticks": None})
        prev = next((x for x in reversed(bars[:bar["bar_index"]]) if x["status"] == "COMPLETE"), None)
        bar["recent_5m_return_ticks"] = ((bar["close"]-prev["close"])/TICK if prev else None)
        bar["price_bar_direction"] = 1 if bar["close"]>bar["open"] else -1 if bar["close"]<bar["open"] else 0
        direction = 1 if bar.get("migration") == "UP" else -1 if bar.get("migration") == "DOWN" else 0
        bar["directional_delta"] = int(bar["delta"]*direction) if direction else None
        bar["delta_supportive"] = bool(bar["directional_delta"] > 0) if direction else None
        bar["price_aligns_with_migration"] = bool(bar["price_bar_direction"] == direction) if direction else None
        bar["delta_aligns_with_migration"] = bool(bar["directional_delta"] > 0) if direction else None
        bar["both_price_and_delta_align"] = bool(bar["price_aligns_with_migration"] and bar["delta_aligns_with_migration"]) if direction else None
        if prev:
            inter = max(0.0, min(bar["vah"], prev["vah"])-max(bar["val"], prev["val"]))
            union = max(bar["vah"], prev["vah"])-min(bar["val"], prev["val"])
            bar["value_area_overlap_ratio"] = float(inter/union) if union>0 else 1.0
        else: bar["value_area_overlap_ratio"] = None
        bar["time_of_day_bucket"] = _tod_bucket(t)
        impact = (direction*(bar["close"]-bar["open"])/TICK/(bar["aggressive_volume"]+1e-12)) if direction and bar["aggressive_volume"] else None
        forward = bar["ask5"] if direction>0 else bar["bid5"] if direction<0 else None
        mlofi = direction*bar["mlofi"] if direction and bar["mlofi"] is not None else None
        bar.update({"directional_price_impact_per_flow": impact,
                    "forward_side_top5_depth": forward, "directional_top5_mlofi": mlofi})
    return bars


def _path(tape: np.ndarray, t: int, sign: int, close: int) -> dict[str, Any]:
    raw_anchor = _mid_at(tape, t, strict=True, close=close)
    if raw_anchor is None:
        return {"raw_anchor_time_ns":None,"raw_anchor_mid":None,"entry_time_ns":None,"entry_quote":None,"actual_fill":None,
            "paths":{str(h):{"raw":None,"quote":None,"actual":None,"horizon_shift":None,"pre_entry_move":None,"bid_ask_effect":None} for h in HORIZONS_MS},
            "excursions":{str(h):{"raw":None,"quote":None,"actual":None} for h in EXCURSION_MS},
            "first_touch":{f"{a}:-{b}":{"result":"UNAVAILABLE","touch_ms":None} for a,b in BARRIERS}}
    rix, raw_mid = raw_anchor; ts=tape["timestamp_ns"]
    entry = _mid_at(tape, t+ENTRY_DELAY_NS, close=close)
    entry_quote = fill = entry_mid = None; entry_ix = None
    if entry:
        entry_ix, entry_mid = entry
        entry_quote = float(tape["ask"][entry_ix] if sign>0 else tape["bid"][entry_ix])
        fill = entry_quote + sign*TICK
    paths={}; excursions={}; touches={}
    for h in HORIZONS_MS:
        raw_end=_mid_at(tape,t+h*1_000_000,close=close)
        # Preserve equal elapsed horizons from the actual entry timestamp.
        exec_end=_mid_at(tape,int(ts[entry_ix])+h*1_000_000,close=close) if entry_ix is not None else None
        raw=sign*(raw_end[1]-raw_mid)/TICK if raw_end else None
        quote=actual=shift=pre=spread=None
        if entry_ix is not None and exec_end is not None:
            exi,exmid=exec_end; exitpx=float(tape["bid"][exi] if sign>0 else tape["ask"][exi])
            quote=sign*(exitpx-entry_quote)/TICK; actual=sign*(exitpx-fill)/TICK
            shift=sign*(exmid-(raw_end[1] if raw_end else raw_mid))/TICK if raw_end else None
            pre=sign*(entry_mid-raw_mid)/TICK
            spread=sign*((exitpx-exmid)+(entry_mid-entry_quote))/TICK
            if abs(actual-(quote-1))>1e-8: raise MigrationStudyError("frozen one-tick adverse fill decomposition failed")
        paths[str(h)]={"raw":float(raw) if raw is not None else None,"quote":float(quote) if quote is not None else None,
                       "actual":float(actual) if actual is not None else None,"horizon_shift":shift,
                       "pre_entry_move":pre,"bid_ask_effect":spread}
    for h in EXCURSION_MS:
        endp=_mid_at(tape,t+h*1_000_000,close=close)
        if not endp: excursions[str(h)]={"raw":None,"quote":None,"actual":None}; continue
        ei=endp[0]; rawvals=sign*(tape["bid"][rix:ei+1].astype(float)+tape["ask"][rix:ei+1].astype(float)-2*raw_mid)/(2*TICK)
        rawvals=rawvals[np.isfinite(rawvals)]
        if entry_ix is not None:
            segment=tape[entry_ix:ei+1]; mids=(segment["bid"].astype(float)+segment["ask"].astype(float))/2
            qentry=float(entry_quote); fentry=float(fill)
            qexit=np.where(sign>0,segment["bid"],segment["ask"]).astype(float)
            qvals=sign*(qexit-qentry)/TICK; av=sign*(qexit-fentry)/TICK
        else:qvals=av=np.asarray([])
        excursions[str(h)]={"raw":{"mfe":float(max(0,np.max(rawvals))),"mae":float(min(0,np.min(rawvals)))} if len(rawvals) else None,
            "quote":{"mfe":float(max(0,np.max(qvals))),"mae":float(min(0,np.min(qvals)))} if len(qvals) else None,
            "actual":{"mfe":float(max(0,np.max(av))),"mae":float(min(0,np.min(av)))} if len(av) else None}
    for fav,adv in BARRIERS:
        endp=_mid_at(tape,t+300_000_000_000,close=close)
        if not endp: touches[f"{fav}:-{adv}"]={"result":"UNAVAILABLE","touch_ms":None};continue
        lim=endp[0]+1; vals=sign*(tape["bid"][rix:lim].astype(float)+tape["ask"][rix:lim].astype(float)-2*raw_mid)/(2*TICK)
        fi=np.flatnonzero(vals>=fav); ai=np.flatnonzero(vals<=-adv); f=int(fi[0]) if len(fi) else None;a=int(ai[0]) if len(ai) else None
        first=f if a is None or f is not None and f<=a else a
        touches[f"{fav}:-{adv}"]={"result":"NEITHER" if first is None else "FAVORABLE_FIRST" if first==f else "ADVERSE_FIRST",
                                   "touch_ms":float((ts[rix+first]-t)/1e6) if first is not None else None}
    return {"raw_anchor_time_ns":int(ts[rix]),"raw_anchor_mid":float(raw_mid),
            "entry_time_ns":int(ts[entry_ix]) if entry_ix is not None else None,
            "entry_quote":entry_quote,"actual_fill":fill,"paths":paths,"excursions":excursions,"first_touch":touches}


def _event_features(bar: Mapping[str, Any], previous_depth_medians: Mapping[str, float]) -> dict[str, Any]:
    direction=1 if bar["migration"]=="UP" else -1
    forward=bar.get("forward_side_top5_depth")
    ref=previous_depth_medians.get("LONG" if direction>0 else "SHORT")
    depth_relative=float(forward/ref) if forward is not None and ref and ref>0 else None
    impact=bar.get("directional_price_impact_per_flow");mlofi=bar.get("directional_top5_mlofi")
    components={"directional_price_impact_per_flow":impact is not None and impact>0,
        "forward_side_top5_depth_relative":depth_relative is not None and depth_relative<1,
        "directional_top5_mlofi":mlofi is not None and mlofi>0}
    return {"directional_delta":bar.get("directional_delta"),"delta_ratio":bar.get("delta_ratio"),
        "delta_supportive":bar.get("delta_supportive"),"delta_opposing":bool(bar.get("directional_delta")<0) if bar.get("directional_delta") is not None else None,
        "price_bar_direction":bar.get("price_bar_direction"),"price_aligns_with_migration":bar.get("price_aligns_with_migration"),
        "delta_aligns_with_migration":bar.get("delta_aligns_with_migration"),"both_price_and_delta_align":bar.get("both_price_and_delta_align"),
        "value_area_overlap_ratio":bar.get("value_area_overlap_ratio"),"trend_efficiency_30s":bar.get("trend_efficiency_30s"),
        "directional_trend_efficiency_30s":bar.get("directional_trend_efficiency_30s"),"rv_30s_ticks":bar.get("rv_30s_ticks"),
        "directional_price_impact_per_flow":impact,"forward_side_top5_depth":forward,
        "forward_side_top5_depth_prior_median":ref,"forward_side_depth_relative":depth_relative,
        "directional_top5_mlofi":mlofi,"effort_components_supportive":components,
        "effort_support_count":sum(components.values()),
        "absorption_effort_result": (abs(float(bar["delta"]))/(abs((bar["close"]-bar["open"])/TICK)+1.0)
             if bar["aggressive_volume"]>0 and bar["delta"]!=0 else None)}


def _daily_features(bars: list[dict[str,Any]], depth_history: dict[str,list[float]]) -> list[dict[str,Any]]:
    med={s:float(np.median(v)) for s,v in depth_history.items() if v}
    for b in bars:
        if b.get("migration") in ("UP","DOWN"):
            feat=_event_features(b,med); b.update({"event_features":feat})
    # End-of-date update only: no within-date leakage into state references.
    for b in bars:
        if b.get("status")=="COMPLETE":
            if b.get("ask5") is not None and b["ask5"]>0: depth_history["LONG"].append(float(b["ask5"]))
            if b.get("bid5") is not None and b["bid5"]>0: depth_history["SHORT"].append(float(b["bid5"]))
    return bars


def _migration_events(bars: Sequence[Mapping[str,Any]], tape: np.ndarray, day: str) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    start,close=_session(day); directional=[b for b in bars if b.get("migration") in ("UP","DOWN")]
    allrows=[];onsets=[]
    for i,b in enumerate(directional):
        prior=directional[i-1] if i else None
        sign=1 if b["migration"]=="UP" else -1
        onset=prior is None or prior["migration"]!=b["migration"] or prior["bar_index"]!=b["bar_index"]-1
        # A missing/no-trade bar breaks continuity as specified in config.
        path=_path(tape,int(b["close_ns"]),sign,close)
        b["paths"]=path
        row={"date":day,"period":_period(day),"bar_index":b["bar_index"],"timestamp_ns":b["close_ns"],
             "direction":"LONG" if sign>0 else "SHORT","sign":sign,"migration":b["migration"],"drift_bar_ordinal":b.get("drift_bar_ordinal",1),
             "is_drift_onset":onset,"poc_shift_ticks":b.get("poc_shift_ticks"),
             "full_value_area_alignment":b.get("full_up_alignment") if sign>0 else b.get("full_down_alignment"),
             "bar":dict(b),"features":b.get("event_features",{})}
        row["paths"]=path
        allrows.append(row)
        if onset:onsets.append(row)
    return allrows,onsets


def _reloads(bars: Sequence[Mapping[str,Any]], tape: np.ndarray, day: str) -> list[dict[str,Any]]:
    start,close=_session(day); ts=tape["timestamp_ns"]
    directional=[b for b in bars if b.get("migration") in ("UP","DOWN")]
    out=[];i=0
    while i<len(directional):
        first=directional[i]; direction=first["migration"]; sign=1 if direction=="UP" else -1
        j=i+1
        while j<len(directional) and directional[j]["migration"]==direction and directional[j]["bar_index"]==directional[j-1]["bar_index"]+1:j+=1
        emitted=False
        for k in range(i,j):
            b=directional[k]; nxt=int(directional[k+1]["close_ns"]) if k+1<j else close
            l=int(np.searchsorted(ts,int(b["close_ns"]),side="right"));r=int(np.searchsorted(ts,nxt,side="left"))
            ids=np.flatnonzero((tape["execution_size"][l:r]>0)&np.isfinite(tape["execution_price"][l:r]))+l
            p=tape["execution_price"][ids]; inside=(p>=b["val"]-1e-9)&(p<=b["vah"]+1e-9)
            hit=np.flatnonzero(inside)
            if len(hit):
                ix=int(ids[hit[0]]); event={"date":day,"period":_period(day),"timestamp_ns":int(ts[ix]),
                    "price":float(tape["execution_price"][ix]),"direction":"LONG" if sign>0 else "SHORT","sign":sign,
                    "drift_start_bar_index":first["bar_index"],"active_value_bar_index":b["bar_index"],
                    "active_val":b["val"],"active_vah":b["vah"],"event_type":"RELOAD_TOUCH"}
                event["paths"]=_path(tape,event["timestamp_ns"],sign,close);out.append(event);emitted=True;break
        i=j
    return out


def _date_payload(day: str, tape: np.ndarray, compact: np.ndarray,
                  depth_history: dict[str,list[float]]) -> dict[str,Any]:
    bars=_daily_features(_make_bars(day,tape,compact),depth_history)
    migrations,onsets=_migration_events(bars,tape,day)
    _,close=_session(day)
    # Price-only controls are drawn from all bars and assigned direction only
    # from the preceding 5m price return. Migration, profile, and L2 do not
    # determine control eligibility.
    reloads=_reloads(bars,tape,day)
    onset_times={int(e["timestamp_ns"]) for e in onsets}
    for e in onsets:
        eb=e["bar"]; keys=("recent_5m_return_ticks","trend_efficiency_30s","rv_30s_ticks")
        candidates=[]
        for b in bars:
            ret=b.get("recent_5m_return_ticks")
            if b.get("status")!="COMPLETE" or ret is None or ret==0 or int(b["close_ns"]) in onset_times:continue
            if (1 if ret>0 else -1)!=e["sign"] or b.get("time_of_day_bucket")!=eb.get("time_of_day_bucket"):continue
            if any(eb.get(k) is None or b.get(k) is None for k in keys):continue
            a=(eb["recent_5m_return_ticks"],eb["trend_efficiency_30s"],eb["rv_30s_ticks"])
            c=(b["recent_5m_return_ticks"],b["trend_efficiency_30s"],b["rv_30s_ticks"])
            d=sum(((float(x)-float(y))/scale)**2 for x,y,scale in zip(a,c,(8,.25,8)))
            candidates.append((d,b))
        if candidates:
            dist,control=min(candidates,key=lambda z:(z[0],z[1]["close_ns"]))
            e["price_only_control"]={"timestamp_ns":int(control["close_ns"]),"distance":dist,
                "paths":_path(tape,int(control["close_ns"]),int(e["sign"]),close)}
        else:e["price_only_control"]=None
    return {"date":day,"period":_period(day),"bars":bars,"migration_events":migrations,
            "onsets":onsets,"reloads":reloads}


def _group(rows: Sequence[Mapping[str,Any]], key: str="paths") -> dict[str,Any]:
    return {str(h):{name:_stats([r[key]["paths"][str(h)].get(name) for r in rows]) for name in ("raw","quote","actual")}
            for h in HORIZONS_MS}


def _event_metric(event: Mapping[str,Any], horizon: int, path: str) -> float | None:
    return event["paths"]["paths"][str(horizon)].get(path)


def _outcomes(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return {str(h):{p:_stats([_event_metric(e,h,p) for e in rows]) for p in ("raw","quote","actual")}
            for h in HORIZONS_MS}


def _excursion_group(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    out={}
    for h in EXCURSION_MS:
        out[str(h)]={}
        for path in ("raw","quote","actual"):
            out[str(h)][path]={metric:_stats([r["paths"]["excursions"][str(h)][path].get(metric)
                if r["paths"]["excursions"][str(h)][path] else None for r in rows]) for metric in ("mfe","mae")}
    return out


def _touches(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    out={}
    for fav,adv in BARRIERS:
        key=f"{fav}:-{adv}"; vals=[r["paths"]["first_touch"][key] for r in rows]
        out[key]={"n":len(vals),"favorable_first":sum(x["result"]=="FAVORABLE_FIRST" for x in vals),
            "adverse_first":sum(x["result"]=="ADVERSE_FIRST" for x in vals),
            "neither":sum(x["result"]=="NEITHER" for x in vals),
            "unavailable":sum(x["result"]=="UNAVAILABLE" for x in vals),
            "median_time_to_touch_ms":_stats([x["touch_ms"] for x in vals if x["touch_ms"] is not None])["median"]}
    return out


def _shape(groups: Mapping[str,Mapping[str,Any]], metric: str) -> str:
    names=[k for k in ("LOW","MID","HIGH") if groups.get(k,{}).get(metric,{}).get("n",0)>=20]
    if len(names)<3:return "INSUFFICIENT_BUCKET_SAMPLE"
    vals=[groups[k][metric]["mean"] for k in names]
    if vals[0]<vals[1]<vals[2]:return "MONOTONIC_IMPROVEMENT"
    if vals[0]>vals[1]>vals[2]:return "OPPOSITE_RELATIONSHIP"
    if vals[1]>vals[0] and vals[1]>vals[2]:return "INVERTED_U"
    if vals[1]<vals[0] and vals[1]<vals[2]:return "U_SHAPE"
    if max(vals)-min(vals)<0.25:return "NO_CLEAR_SHAPE"
    return "NO_CLEAR_SHAPE"


def _cutpoints(rows: Sequence[Mapping[str,Any]], field: str) -> list[float] | None:
    x=np.asarray([r.get(field) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))],float)
    return [float(v) for v in np.quantile(x,[1/3,2/3])] if len(x)>=3 else None


def _tercile_tables(events: Sequence[Mapping[str,Any]], feature: str, outcome_path: str="actual",horizon: int=10000) -> dict[str,Any]:
    spring=[e for e in events if e["period"]=="SPRING_2025"]
    cuts=_cutpoints(spring,feature); result={"spring_cutpoints":cuts,"periods":{}}
    for period in ("SPRING_2025","OCTOBER_2025"):
        rows=[e for e in events if e["period"]==period];cells={n:[] for n in ("LOW","MID","HIGH")}
        if cuts:
            for e in rows:
                v=e.get(feature)
                if v is not None and math.isfinite(float(v)):cells["LOW" if v<=cuts[0] else "MID" if v<=cuts[1] else "HIGH"].append(e)
        result["periods"][period]={n:{"n":len(group),"actual_markout":_stats([_event_metric(x,horizon,outcome_path) for x in group]),
            "raw_markout":_stats([_event_metric(x,horizon,"raw") for x in group]),
            "quote_markout":_stats([_event_metric(x,horizon,"quote") for x in group])} for n,group in cells.items()}
        for n in cells:
            if result["periods"][period][n]["n"]<20:result["periods"][period][n]["status"]="INSUFFICIENT_BUCKET_SAMPLE"
    for p in result["periods"].values():
        p["shape"]=_shape({k:{"actual_markout":{"n":v["actual_markout"]["n"],"mean":v["actual_markout"]["mean"]}} for k,v in p.items() if k in ("LOW","MID","HIGH")},"actual_markout")
    return result


def _correlation(groups: Sequence[tuple[float,float]]) -> float | None:
    if len(groups)<3:return None
    x=np.asarray([a for a,_ in groups]);y=np.asarray([b for _,b in groups])
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def _price_controls(onsets: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    matches=[]
    for e in onsets:
        c=e.get("price_only_control")
        if c:
            matches.append({"event_date":e["date"],"event_timestamp_ns":e["timestamp_ns"],
                "control_timestamp_ns":c["timestamp_ns"],"distance":c["distance"],
                "event_paths":e["paths"]["paths"],"control_paths":c["paths"]["paths"]})
    diff={str(h):{p:_stats([m["event_paths"][str(h)][p]-m["control_paths"][str(h)][p]
        for m in matches if m["event_paths"][str(h)][p] is not None and m["control_paths"][str(h)][p] is not None])
        for p in ("raw","quote","actual")} for h in HORIZONS_MS}
    event_markouts={str(h):{p:_stats([m["event_paths"][str(h)][p] for m in matches]) for p in ("raw","quote","actual")} for h in HORIZONS_MS}
    control_markouts={str(h):{p:_stats([m["control_paths"][str(h)][p] for m in matches]) for p in ("raw","quote","actual")} for h in HORIZONS_MS}
    return {"method":"same date and sign; nearest deterministic price-only features and time bucket; no value/orderflow inputs",
            "matched_pairs":len(matches),"matches":matches,"event_minus_control_ticks":diff,
            "event_markouts":event_markouts,"control_markouts":control_markouts,
            "status":"ADEQUATE" if len(matches)>=20 else "INSUFFICIENT_PRICE_ONLY_CONTROL_SAMPLE"}


def _leave_out(events: Sequence[Mapping[str,Any]], unit: str) -> dict[str,Any]:
    groups=defaultdict(list)
    for e in events:
        key=e["date"] if unit=="date" else e["week"]
        groups[key].append(e)
    rows=[]
    for key in sorted(groups):
        remain=[e for e in events if (e["date"] if unit=="date" else e["week"])!=key]
        rows.append({"omitted":key,"n":len(remain),"actual_10s":_stats([_event_metric(e,10000,"actual") for e in remain])})
    vals=[r["actual_10s"]["mean"] for r in rows if r["actual_10s"]["mean"] is not None]
    return {"unit":unit,"omissions":rows,"sign_stability":sum(v>0 for v in vals),"omission_count":len(vals),
        "median":float(np.median(vals)) if vals else None,"minimum":min(vals) if vals else None,"maximum":max(vals) if vals else None,
        "worst_omission":min(rows,key=lambda r:r["actual_10s"]["mean"] if r["actual_10s"]["mean"] is not None else math.inf)["omitted"] if vals else None,
        "best_omission":max(rows,key=lambda r:r["actual_10s"]["mean"] if r["actual_10s"]["mean"] is not None else -math.inf)["omitted"] if vals else None}


def _permutation(events: Sequence[Mapping[str,Any]], reloads: Sequence[Mapping[str,Any]]=(), seed: int=20251007, nperm: int=1000) -> dict[str,Any]:
    rng=np.random.default_rng(seed); output={}
    for label, selector in (("delta",lambda e:e.get("features",{}).get("delta_supportive")),
        ("effort",lambda e:(e.get("features",{}).get("effort_support_count",0)>=2)),
        ("migration_direction",lambda e:e["sign"]>0),
        ("full_confluence",lambda e:(e.get("features",{}).get("delta_supportive") and e.get("features",{}).get("effort_support_count",0)>=2))):
        vals=[e for e in events if selector(e) is not None and _event_metric(e,10000,"actual") is not None]
        if len(vals)<20:output[label]={"status":"INSUFFICIENT_SAMPLE","n":len(vals)};continue
        x=np.asarray([bool(selector(e)) for e in vals]);y=np.asarray([_event_metric(e,10000,"actual") for e in vals]); observed=float(np.mean(y[x])-np.mean(y[~x])) if x.any() and (~x).any() else None
        if observed is None:output[label]={"status":"INSUFFICIENT_SAMPLE","n":len(vals)};continue
        null=[]
        strata=defaultdict(list)
        for i,e in enumerate(vals):strata[(e["period"],e["direction"],e["time_of_day_bucket"])].append(i)
        for _ in range(nperm):
            xp=x.copy()
            for inds in strata.values():xp[inds]=rng.permutation(xp[inds])
            if xp.any() and (~xp).any():null.append(float(np.mean(y[xp])-np.mean(y[~xp])))
        output[label]={"status":"DESCRIPTIVE","n":len(vals),"observed_difference_actual_10s":observed,
            "permutations":len(null),"null_percentile":float(np.mean(np.asarray(null)<=observed)) if null else None,
            "two_sided_p":float((1+sum(abs(z)>=abs(observed) for z in null))/(1+len(null))) if null else None}
    onset_by_key={(e["date"],int(e["bar_index"])):e for e in events}
    paired=[]
    for r in reloads:
        onset=onset_by_key.get((r["date"],int(r.get("drift_start_bar_index",-1))))
        if onset is None:continue
        ra=_event_metric(r,10000,"actual");oa=_event_metric(onset,10000,"actual")
        if ra is not None and oa is not None:paired.append((r,ra-oa))
    if len(paired)<20:output["reload_touch"]={"status":"INSUFFICIENT_SAMPLE","paired_n":len(paired)}
    else:
        diffs=np.asarray([d for _,d in paired]);observed=float(np.mean(diffs));null=[]
        for _ in range(nperm):null.append(float(np.mean(diffs*rng.choice(np.asarray([-1,1]),size=len(diffs)))))
        output["reload_touch"]={"status":"PAIRED_RELOAD_MINUS_ONSET","paired_n":len(paired),"observed_delta_actual_10s":observed,
            "permutations":nperm,"null_percentile":float(np.mean(np.asarray(null)<=observed)),
            "two_sided_p":float((1+sum(abs(z)>=abs(observed) for z in null))/(1+nperm))}
    return {"seed":seed,"results":output}


def _aggregate(payloads: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    bars=[b for p in payloads for b in p["bars"] if b.get("status")=="COMPLETE"]
    migrations=[e for p in payloads for e in p["migration_events"]]
    onsets=[e for p in payloads for e in p["onsets"]]
    reloads=[e for p in payloads for e in p["reloads"]]
    # Frozen price-only matched controls use per-bar price state and event paths.
    # attach matching features to onset events
    for e in onsets:
        b=e["bar"]; e["price_features"]={"recent_5m_return_ticks":b.get("recent_5m_return_ticks"),
            "recent_5m_velocity_ticks_per_min":float(b["recent_5m_return_ticks"])/5 if b.get("recent_5m_return_ticks") is not None else None,
            "trend_efficiency_30s":b.get("trend_efficiency_30s"),"rv_30s_ticks":b.get("rv_30s_ticks")}
        e["time_of_day_bucket"]=b["time_of_day_bucket"]
        e["week"]=_iso_week(e["date"])
        sign=int(e["sign"])
        opposing=int(b.get("sell_volume",0) if sign>0 else b.get("buy_volume",0))
        adverse=max(0.0,-sign*(float(b["close"])-float(b["open"]))/TICK)
        e["features"]["opposing_aggressive_volume"]=opposing
        e["features"]["opposing_flow_adverse_price_progress_ticks"]=adverse
        e["features"]["opposing_flow_price_efficiency"]=float(adverse/(opposing+1))
        e["features"]["opposing_absorption_effort_result"]=float(opposing/(adverse+1.0)) if opposing else None
        e["abs_poc_shift_ticks"]=abs(float(b["poc_shift_ticks"]))
        e["value_area_overlap_ratio"]=b.get("value_area_overlap_ratio")
        e["trend_efficiency_30s"]=b.get("trend_efficiency_30s")
    for e in migrations:e["week"]=_iso_week(e["date"])
    for e in reloads:e["week"]=_iso_week(e["date"])
    period={}
    for per in ("SPRING_2025","OCTOBER_2025"):
        er=[e for e in onsets if e["period"]==per];mr=[e for e in reloads if e["period"]==per]
        period[per]={"onsets":len(er),"long":sum(e["sign"]>0 for e in er),"short":sum(e["sign"]<0 for e in er),
            "raw":{str(h):_stats([_event_metric(e,h,"raw") for e in er]) for h in HORIZONS_MS},
            "quote":{str(h):_stats([_event_metric(e,h,"quote") for e in er]) for h in HORIZONS_MS},
            "actual":{str(h):_stats([_event_metric(e,h,"actual") for e in er]) for h in HORIZONS_MS},
            "reload_events":len(mr),"reload_actual":{str(h):_stats([_event_metric(e,h,"actual") for e in mr]) for h in HORIZONS_MS}}
    def split_groups(keyfn, rows=onsets):
        res={}
        for name in sorted({keyfn(x) for x in rows}):
            subset=[x for x in rows if keyfn(x)==name]
            res[name]={"n":len(subset),"raw_10s":_stats([_event_metric(x,10000,"raw") for x in subset]),
                "quote_10s":_stats([_event_metric(x,10000,"quote") for x in subset]),"actual_10s":_stats([_event_metric(x,10000,"actual") for x in subset])}
        return res
    delta={p:{state:{"n":sum((e["features"].get("delta_supportive") is True)==(state=="SUPPORTIVE") for e in onsets if e["period"]==p),
        "outcomes":_outcomes([e for e in onsets if e["period"]==p and (e["features"].get("delta_supportive") is True)==(state=="SUPPORTIVE")])}
        for state in ("SUPPORTIVE","OPPOSING_OR_NEUTRAL")} for p in ("SPRING_2025","OCTOBER_2025")}
    for p in delta:
        for label,pred in (("OPPOSING",lambda e:e["features"].get("delta_opposing") is True),
                           ("NEUTRAL",lambda e:e["features"].get("directional_delta")==0)):
            subset=[e for e in onsets if e["period"]==p and pred(e)]
            delta[p][label]={"n":len(subset),"outcomes":_outcomes(subset)}
    effort={p:{str(n):{"n":sum(e["features"].get("effort_support_count")==n for e in onsets if e["period"]==p),
        "outcomes":_outcomes([e for e in onsets if e["period"]==p and e["features"].get("effort_support_count")==n])}
        for n in range(4)} for p in ("SPRING_2025","OCTOBER_2025")}
    confluence={"VALUE_MIGRATION_ONLY":onsets,
        "MIGRATION_PLUS_SUPPORTIVE_DELTA":[e for e in onsets if e["features"].get("delta_supportive") is True],
        "MIGRATION_PLUS_EFFORT_SUPPORT":[e for e in onsets if e["features"].get("effort_support_count",0)>=2],
        "MIGRATION_PLUS_DELTA_AND_EFFORT":[e for e in onsets if e["features"].get("delta_supportive") is True and e["features"].get("effort_support_count",0)>=2]}
    confluence={k:{"n":len(v),"outcomes":_outcomes(v),"mfe_mae":_excursion_group(v),"first_touch":_touches(v)} for k,v in confluence.items()}
    ordinal={str(n):[e for e in migrations if min(int(e.get("drift_bar_ordinal",1)),5)==n] for n in (1,2,3,4,5)}
    ordinal={k:{"n":len(v),"outcomes":_outcomes(v),"mfe_mae":_excursion_group(v)} for k,v in ordinal.items()}
    alignment={p:{state:[e for e in onsets if e["period"]==p and bool(e["full_value_area_alignment"]) is val]
        for state,val in (("ALIGNED",True),("NOT_FULLY_ALIGNED",False))} for p in ("SPRING_2025","OCTOBER_2025")}
    alignment={p:{k:{"n":len(v),"outcomes":_outcomes(v)} for k,v in cells.items()} for p,cells in alignment.items()}
    absorption={"opposing_flow_price_efficiency":_tercile_tables(onsets,"opposing_flow_price_efficiency"),
        "opposing_absorption_effort_result":_tercile_tables(onsets,"opposing_absorption_effort_result")}
    daily={}
    for p in payloads:
        day=p["date"]; ev=p["onsets"]; rel=p["reloads"]
        daily[day]={"period":_period(day),"drift_onsets":len(ev),"long_onsets":sum(e["sign"]>0 for e in ev),"short_onsets":sum(e["sign"]<0 for e in ev),
            "raw_10s":_stats([_event_metric(e,10000,"raw") for e in ev]),"executable_10s":_stats([_event_metric(e,10000,"quote") for e in ev]),
            "actual_10s":_stats([_event_metric(e,10000,"actual") for e in ev]),"delta_support_rate":float(np.mean([bool(e["features"].get("delta_supportive")) for e in ev])) if ev else None,
            "effort_support_rate":float(np.mean([e["features"].get("effort_support_count",0)>=2 for e in ev])) if ev else None,
            "mean_va_overlap":_stats([e["bar"].get("value_area_overlap_ratio") for e in ev])["mean"],
            "mean_trend_efficiency":_stats([e["bar"].get("trend_efficiency_30s") for e in ev])["mean"],"reload_events":len(rel)}
    weekly=defaultdict(list)
    for e in onsets:weekly[e["week"]].append(e)
    weekly={w:{"n":len(es),"raw_10s":_stats([_event_metric(e,10000,"raw") for e in es]),
        "executable_10s":_stats([_event_metric(e,10000,"quote") for e in es]),"actual_10s":_stats([_event_metric(e,10000,"actual") for e in es]),
        "delta_supported_actual_10s":_stats([_event_metric(e,10000,"actual") for e in es if e["features"].get("delta_supportive")]),
        "full_confluence_actual_10s":_stats([_event_metric(e,10000,"actual") for e in es if e["features"].get("delta_supportive") and e["features"].get("effort_support_count",0)==3])}
        for w,es in weekly.items()}
    controls=_price_controls(onsets)
    paths=_group(onsets)
    shift_table=_tercile_tables(onsets,"abs_poc_shift_ticks") if onsets else {}
    return {"bars":bars,"migrations":migrations,"onsets":onsets,"reloads":reloads,"paths":paths,
        "excursions":_excursion_group(onsets),"touches":_touches(onsets),"period":period,
        "direction":split_groups(lambda e:"LONG" if e["sign"]>0 else "SHORT"),"timeofday":split_groups(lambda e:e["time_of_day_bucket"]),
        "delta":delta,"effort":effort,"confluence":confluence,"ordinal":ordinal,"alignment":alignment,"absorption":absorption,
        "shift_terciles":shift_table,"overlap_terciles":_tercile_tables(onsets,"value_area_overlap_ratio"),
        "trend_terciles":_tercile_tables(onsets,"trend_efficiency_30s"),"price_control":controls,"daily":daily,"weekly":weekly,
        "lodo":{"VALUE_MIGRATION_ONLY":_leave_out(onsets,"date"),
            "DELTA_SUPPORTED":_leave_out([e for e in onsets if e["features"].get("delta_supportive") is True],"date"),
            "FULL_CONFLUENCE":_leave_out([e for e in onsets if e["features"].get("delta_supportive") is True and e["features"].get("effort_support_count",0)>=2],"date"),
            "RELOAD_TOUCH":_leave_out(reloads,"date")},
        "lowo":{"VALUE_MIGRATION_ONLY":_leave_out(onsets,"week"),
            "DELTA_SUPPORTED":_leave_out([e for e in onsets if e["features"].get("delta_supportive") is True],"week"),
            "FULL_CONFLUENCE":_leave_out([e for e in onsets if e["features"].get("delta_supportive") is True and e["features"].get("effort_support_count",0)>=2],"week"),
            "RELOAD_TOUCH":_leave_out(reloads,"week")},"permutation":_permutation(onsets,reloads),
        "density":{"5m_bars":len(bars),"up_migration_bars":sum(x.get("migration",x["bar"]["migration"])=="UP" for x in migrations),
            "down_migration_bars":sum(x.get("migration",x["bar"]["migration"])=="DOWN" for x in migrations),"flat_bars":sum(x.get("poc_shift_ticks")==0 for x in bars),
            "drift_onsets":len(onsets),"continuation_bars":len(migrations)-len(onsets),"reload_touches":len(reloads),
            "events_per_day":len(onsets)/len(payloads),"events_per_week":len(onsets)/max(1,len(weekly))},
        "positive_negative_flat_days":{"positive":sum((d["actual_10s"]["mean"] or 0)>0 for d in daily.values()),
            "negative":sum((d["actual_10s"]["mean"] or 0)<0 for d in daily.values()),
            "flat":sum((d["actual_10s"]["mean"] or 0)==0 for d in daily.values())}}


def _iso_week(day: str) -> str:
    from datetime import date
    x=date.fromisoformat(day).isocalendar();return f"{x.year}-W{x.week:02d}"


def _path_from_payload(bar: Mapping[str,Any]) -> dict[str,Any]:
    # Price-control pool receives outcomes by matching the already evaluated
    # migration bar against its matching migration event.
    return bar.get("paths",{})


def _checkpoints(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, day: str, source_sha: str, tape_sha: str) -> dict[str,Any] | None:
    if not path.is_file():return None
    try:
        with gzip.open(path,"rt",encoding="utf-8") as f: row=json.load(f)
    except (OSError,EOFError,json.JSONDecodeError):return None
    expected={"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,"date":day,"source_sha256":source_sha,
        "tape_sha256":tape_sha,"config_sha256":CONFIG_SHA256,"study_sha256":STUDY_SHA256}
    return row if all(row.get(k)==v for k,v in expected.items()) else None


def run(*, data_root: Path=native.DATA_ROOT, output_root: Path=OUT_ROOT, smoke: bool=False) -> dict[str,Any]:
    started=time.monotonic()
    source_paths, manifest=native._source_catalog(data_root)
    target=flow.TARGET_DATES[:1] if smoke else flow.TARGET_DATES
    if set(flow.SPRING_DATES)!=set(native.SPRING_DATES) or len(flow.SPRING_DATES)!=35 or len(flow.OCTOBER_DATES)!=19:
        raise MigrationStudyError("frozen eligible target date set mismatch")
    # Validate sealed canonical source tapes and source identity before output.
    tape_paths={}; tape_sha={}
    for day in target:
        path=native._tape_path(day)
        if not path.is_file():raise MigrationStudyError(f"missing sealed native ES tape: {day}")
        tape,meta=native._load_tape(day,path,str(manifest[day]["sha256"]))
        tape_paths[day]=path;tape_sha[day]=native._sha(path)
        del tape,meta
    output_root.mkdir(parents=True,exist_ok=True)
    _json_write(output_root/"study-config.json",CONFIG)
    coverage={"status":"PASS","dataset":"GLBX.MDP3","schema":"mbp-10","instrument":"ES",
        "native_es_mbp10_only":True,"no_mes_market_data":True,"no_mbo":True,
        "spring_dates":list(flow.SPRING_DATES),"october_dates":list(flow.OCTOBER_DATES),
        "source_sha256_by_date":{d:manifest[d]["sha256"] for d in target},
        "source_files_by_date":{d:{"path":str(source_paths[d]),"sha256":manifest[d]["sha256"],"bytes":source_paths[d].stat().st_size,
            "symbol":manifest[d]["symbol"],"schema":manifest[d]["schema"]} for d in target},
        "canonical_tape_sha256_by_date":tape_sha}
    _json_write(output_root/"source-coverage.json",coverage)
    depth_history={"LONG":[],"SHORT":[]};payloads=[];completed=[];resumed=[]
    for idx,day in enumerate(target,1):
        cp=_checkpoints(output_root,day)
        cached=_read_checkpoint(cp,day,str(manifest[day]["sha256"]),tape_sha[day])
        if cached:
            payload=cached["payload"];resumed.append(day)
            # State references update once per date from the cached bars.
            for b in payload["bars"]:
                if b.get("status")=="COMPLETE":
                    if b.get("ask5") is not None and b["ask5"]>0:depth_history["LONG"].append(float(b["ask5"]))
                    if b.get("bid5") is not None and b["bid5"]>0:depth_history["SHORT"].append(float(b["bid5"]))
        else:
            print(f"VALUE_MIGRATION_DATE_START={idx}/{len(target)} date={day}",flush=True)
            tape,_=native._load_tape(day,tape_paths[day],str(manifest[day]["sha256"]))
            compact=flow._cached_compact(day,source_paths[day],str(manifest[day]["sha256"]),output_root)
            payload=_date_payload(day,tape,compact,depth_history)
            native._write_checkpoint(cp,{"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,"date":day,
                "source_sha256":manifest[day]["sha256"],"tape_sha256":tape_sha[day],"config_sha256":CONFIG_SHA256,
                "study_sha256":STUDY_SHA256,"payload":payload})
            completed.append(day)
            del tape,compact
        payloads.append(payload)
        if smoke:return {"status":"SMOKE_PASS","dates":list(target),"bars":sum(len(p["bars"]) for p in payloads),
                        "events":sum(len(p["onsets"]) for p in payloads)}
        print(f"VALUE_MIGRATION_DATE_COMPLETE={day} bars={len(payload['bars'])} onsets={len(payload['onsets'])} reloads={len(payload['reloads'])}",flush=True)
        native._write_json(output_root/"checkpoints"/"progress.json",{"last_date":day,"completed_dates":completed,
            "resumed_dates":resumed,"config_sha256":CONFIG_SHA256,"study_sha256":STUDY_SHA256})
    if len(payloads)!=54:raise MigrationStudyError(f"expected 54 target dates, got {len(payloads)}")
    agg=_aggregate(payloads)
    # Add absolute POC shift and feature values to onset records for later cutpoints.
    for e in agg["onsets"]:
        e["abs_poc_shift_ticks"]=abs(float(e["bar"]["poc_shift_ticks"]))
        e["value_area_overlap_ratio"]=e["bar"].get("value_area_overlap_ratio")
        e["trend_efficiency_30s"]=e["bar"].get("trend_efficiency_30s")
    # Build compact tables and decision evidence.
    agg["shift_terciles"]=_tercile_tables(agg["onsets"],"abs_poc_shift_ticks")
    agg["overlap_terciles"]=_tercile_tables(agg["onsets"],"value_area_overlap_ratio")
    agg["trend_terciles"]=_tercile_tables(agg["onsets"],"trend_efficiency_30s")
    fill10=agg["paths"]["10000"]["actual"]["mean"]
    quote_use=[agg["paths"][str(h)]["quote"]["mean"] for h in (2000,5000,10000,30000,60000)]
    spring10=agg["period"]["SPRING_2025"]["actual"]["10000"]["mean"]
    oct10=agg["period"]["OCTOBER_2025"]["actual"]["10000"]["mean"]
    migration_info=agg["price_control"]["event_minus_control_ticks"]["10000"]["actual"]["mean"]
    sufficient=len(agg["onsets"])>=40
    failed=(not sufficient or fill10 is None or fill10<=0 or any(x is None or x<=0 for x in quote_use)
        or spring10 is None or oct10 is None or spring10*oct10<0 or migration_info is None or migration_info<=0)
    if not sufficient:decision="INSUFFICIENT_SAMPLE";next_step="REQUIRE_ADDITIONAL_PREDECLARED_DATA"
    elif failed:decision="ES_VALUE_MIGRATION_MODEL_FAILED";next_step="STOP_ES_VALUE_MIGRATION_BRANCH"
    else:decision="ES_VALUE_MIGRATION_MODEL_PROMISING";next_step="BUILD_ONE_FROZEN_ES_VALUE_MIGRATION_STRATEGY_V1"
    summary={"run_id":RUN_ID,"status":"COMPLETE","dataset":"SPRING_2025 + OCTOBER_2025",
        "spring_dates":list(flow.SPRING_DATES),"october_dates":list(flow.OCTOBER_DATES),"total_5m_bars":len(agg["bars"]),
        "up_migration_bars":agg["density"]["up_migration_bars"],"down_migration_bars":agg["density"]["down_migration_bars"],
        "flat_migration_bars":agg["density"]["flat_bars"],"drift_onsets":len(agg["onsets"]),
        "long_drift_onsets":sum(e["sign"]>0 for e in agg["onsets"]),"short_drift_onsets":sum(e["sign"]<0 for e in agg["onsets"]),
        "reload_events":len(agg["reloads"]),"density":agg["density"],"raw_markouts":agg["paths"],
        "period_results":agg["period"],"direction_results":agg["direction"],"price_only_control":agg["price_control"],
        "delta_relationship":agg["delta"],"effort_relationship":agg["effort"],"confluence":agg["confluence"],
        "drift_ordinal":agg["ordinal"],"full_value_area_alignment":agg["alignment"],"opposing_absorption":agg["absorption"],
        "shift_terciles":agg["shift_terciles"],
        "overlap_terciles":agg["overlap_terciles"],"trend_terciles":agg["trend_terciles"],
        "candidate_hypothesis":None,"primary_decision":decision,"next_step":next_step,
        "native_es_mbp10_only":True,"public_model_transfer_study":True,"original_public_instrument":"NQ","tested_instrument":"ES",
        "exact_fabio_proprietary_model":False,"exact_nasdaq_effort_reproduced":False,"true_mbo_signature_used":False,
        "bar_interval_5m_is_v1_replication_convention":True,"optimization_performed":False,"optuna_performed":False,
        "threshold_search_performed":False,"bar_interval_search_performed":False,"value_area_percent_search_performed":False,
        "stop_target_search_performed":False,"final_oos_accessed":False,"data_downloaded":False,"commit_performed":False,
        "config_sha256":CONFIG_SHA256,"study_sha256":STUDY_SHA256,"elapsed_seconds":time.monotonic()-started}
    files={
        "summary.json":summary,"source-coverage.json":coverage,"bars.jsonl.gz":agg["bars"],"events.jsonl.gz":agg["onsets"],
        "migration-events.json":agg["migrations"],"drift-onsets.json":agg["onsets"],
        "drift-continuation.json":[e for e in agg["migrations"] if not e["is_drift_onset"]],
        "value-profiles.json":[{"date":b["date"],"bar_index":b["bar_index"],"start_ns":b["start_ns"],"close_ns":b["close_ns"],
            "poc":b.get("poc"),"vah":b.get("vah"),"val":b.get("val"),"total_volume":b.get("total_volume"),"price_volume_by_tick":b.get("price_volume_by_tick")} for b in agg["bars"]],
        "delta-analysis.json":agg["delta"],"confluence-comparison.json":agg["confluence"],
        "ordinal-results.json":agg["ordinal"],"opposing-absorption.json":agg["absorption"],
        "bar-alignment.json":{"full_value_area_alignment":agg["alignment"],"price_bar_alignment":{p:{d:{"n":sum(e["period"]==p and e["direction"]==d for e in agg["onsets"]),
            "price_align_rate":_stats([int(e["bar"]["price_aligns_with_migration"]) for e in agg["onsets"] if e["period"]==p and e["direction"]==d and e["bar"].get("price_aligns_with_migration") is not None])}
            for d in ("LONG","SHORT")} for p in ("SPRING_2025","OCTOBER_2025")}},
        "balance-context.json":{"value_area_overlap":agg["overlap_terciles"],"trend_efficiency":agg["trend_terciles"]},
        "effort-components.json":[{"date":e["date"],"timestamp_ns":e["timestamp_ns"],"components":e["features"]} for e in agg["onsets"]],
        "effort-support-count.json":agg["effort"],"reload-events.json":agg["reloads"],
        "reload-analysis.json":{"events":len(agg["reloads"]),"markouts":{str(h):{p:_stats([_event_metric(e,h,p) for e in agg["reloads"]]) for p in ("raw","quote","actual")} for h in HORIZONS_MS},
            "mfe_mae":_excursion_group(agg["reloads"]),"first_touch":_touches(agg["reloads"])},
        "price-only-control.json":agg["price_control"],"raw-markouts.json":agg["paths"],
        "executable-markouts.json":{str(h):agg["paths"][str(h)]["quote"] for h in HORIZONS_MS},
        "actual-fill-markouts.json":{str(h):agg["paths"][str(h)]["actual"] for h in HORIZONS_MS},
        "execution-hurdle.json":{"entry_delay_ms":2.0,"entry_side":"LONG ask / SHORT bid",
            "adverse_entry_fill_ticks":1,"exit_side":"LONG bid / SHORT ask","decomposition":{str(h):{
                "raw":agg["paths"][str(h)]["raw"],"quote":agg["paths"][str(h)]["quote"],"actual":agg["paths"][str(h)]["actual"],
                "horizon_shift":_stats([e["paths"]["paths"][str(h)]["horizon_shift"] for e in agg["onsets"]]),
                "pre_entry_price_movement":_stats([e["paths"]["paths"][str(h)]["pre_entry_move"] for e in agg["onsets"]]),
                "bid_ask_effect":_stats([e["paths"]["paths"][str(h)]["bid_ask_effect"] for e in agg["onsets"]]),
                "adverse_entry_fill_effect_ticks":-1.0} for h in HORIZONS_MS}},
        "mfe-mae.json":agg["excursions"],"first-touch.json":agg["touches"],"period-results.json":agg["period"],
        "direction-results.json":agg["direction"],"time-of-day-results.json":agg["timeofday"],"daily-results.json":agg["daily"],
        "weekly-results.json":agg["weekly"],"lodo-results.json":agg["lodo"],"lowo-results.json":agg["lowo"],
        "permutation-results.json":agg["permutation"]}
    for name,val in files.items():
        path=output_root/name
        if name.endswith(".jsonl.gz"):_gzip_rows(path,val)
        else:_json_write(path,val)
    # Note: arrays of per-bar profile histograms are report artifacts; events are
    # the sole primary inferential unit, one onset per uninterrupted drift.
    report=[f"# {RUN_ID}","",f"Decision: **{decision}**",f"Next step: `{next_step}`.","",
        "This is an ES transfer test of public NQ-described mechanisms. It does not reproduce Fabio's proprietary Nasdaq system.",
        "Spring 2025 is primary DEV discovery; October 2025 is secondary DEV compatibility. Neither is untouched OOS.",
        f"Sessions: 54; 5m bars: {len(agg['bars'])}; drift onsets: {len(agg['onsets'])}; reload touches: {len(agg['reloads'])}.",
        f"Actual-fill means at 2/5/10/30/60 seconds: {[agg['paths'][str(h)]['actual']['mean'] for h in (2000,5000,10000,30000,60000)]} ticks.",
        f"Price-only same-day matched difference at 10s actual: {migration_info}; matches={agg['price_control']['matched_pairs']}.",
        "No optimization, stop/target search, data download, or final OOS access was performed.",""]
    (output_root/"report.md").write_text("\n".join(report),encoding="utf-8")
    artifacts={p.name:native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name not in ("run-manifest.json","artifact-hashes.json")}
    checkpoint_hashes={d:native._sha(_checkpoints(output_root,d)) for d in target}
    _json_write(output_root/"run-manifest.json",{"status":"COMPLETE","run_id":RUN_ID,"config_sha256":CONFIG_SHA256,
        "study_sha256":STUDY_SHA256,"source_coverage_sha256":native._sha(output_root/"source-coverage.json"),
        "checkpoint_sha256_by_date":checkpoint_hashes,"artifact_sha256_by_name":artifacts,
        "no_2026_access":True,"no_download":True,"no_optimization":True})
    _json_write(output_root/"artifact-hashes.json",{"status":"HASHED","files":{p.name:native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name!="artifact-hashes.json"}})
    return summary


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=native.DATA_ROOT)
    parser.add_argument("--output-root",type=Path,default=OUT_ROOT)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv)
    try:result=run(data_root=args.data_root,output_root=args.output_root,smoke=args.smoke)
    except (MigrationStudyError,native.VacuumStudyError,OSError,ValueError,KeyError,AssertionError) as exc:
        parser.exit(2,f"VALUE_MIGRATION_ERROR: {exc}\n")
    print(f"ES_VALUE_AREA_MIGRATION_EVENT_STUDY={result['status']}",flush=True)
    if result["status"]=="COMPLETE":print(f"PRIMARY_DECISION={result['primary_decision']}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
