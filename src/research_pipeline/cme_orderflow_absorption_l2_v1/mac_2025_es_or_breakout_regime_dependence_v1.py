"""Frozen, descriptive Spring/October 2025 ES opening-range regime audit.

No event selection, strategy PnL, parameter fitting, threshold search, or OOS data.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_es_structural_breakout_l2_v1 as parent
from . import mac_2025_es_structural_breakout_entry_timing_audit as entry
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_flow_momentum_v1 as flow

RUN_ID = "CMEOrderflow_ES_OR_BREAKOUT_REGIME_DEPENDENCE_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
FEATURES = ("or_width", "rv_10s", "trend_efficiency_10s", "price_velocity_10s",
            "minutes_after_10", "directional_top5_mlofi_500ms")
HORIZONS = parent.HORIZONS_MS
TERCILES = ("LOW", "MID", "HIGH")
PATH_CLASSES = ("FAST_CONTINUATION", "CONTINUATION_THEN_RETURN", "IMMEDIATE_REJECTION",
                "STAGNATION_THEN_RETURN", "NO_RETURN_BEFORE_CLOSE", "OPPOSITE_RANGE_BREAK", "UNCLASSIFIED")
CONFIG = {
    "version": 2, "parent_run_id": parent.RUN_ID, "parent_config_sha256": parent.CONFIG_SHA256,
    "entry_audit_run_id": entry.RUN_ID, "entry_audit_config_sha256": entry.CONFIG_SHA256,
    "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
    "spring_role": "PRIMARY_DISCOVERY", "october_role": "SECONDARY_DEV_COMPATIBILITY",
    "event_stream": "hash-verified frozen first strict OR breakouts; no regeneration/change",
    "price_features": "reuse parent strict-pre-event 10-second 500ms-midpoint-grid RV, signed trend efficiency and velocity; no RV30 exists in validated parent",
    "mlofi": "reuse parent strictly pre-event 500ms directional normalized TOP5 price-keyed MLOFI",
    "feature_order": list(FEATURES), "feature_terciles": "Spring pooled feature cutpoints q1/3,q2/3 frozen before October outcomes; chronological prior-date OR-width percentile separately",
    "minimum_tercile_n": 10, "minimum_interaction_cell_n": 10,
    "outcome": "raw signed 10s midpoint markout; 2s,5s,30s,60s compatibility; quote/fill reference only",
    "return_inside": "first subsequent actual trade LONG <= OR high or SHORT >= OR low through 16:00 ET exclusive",
    "favorable_expansion": "maximum subsequent actual trade distance beyond boundary before first return; breakout trade included",
    "good_expansion": "+8 ES ticks before first return; if no return, +8 before RTH close is GOOD, otherwise UNRESOLVED",
    "topology_priority": "opposite OR break any time before close; then no-return; then +8 within 2s; then +8 before return; then return <=2s; then later return; else unclassified",
    "topology_thresholds": {"favorable_ticks": 8, "fast_ms": 2000},
    "statistics": "tercile means/medians/counts; descriptive shape only; no fitted classifier",
    "shape_rule": "strict tercile mean ordering with >=10 each: monotonic +/-; middle local max/min; saturation if adjacent gap <=25% full range; otherwise no clear shape",
    "compatibility": "same sign high-low Spring/October with >=10 in each extreme; opposite sign=OPPOSITE; otherwise insufficient",
    "composition_rule": "Spring/October tercile share shift >=0.15 maximum; within-tercile raw10s mean difference >=2 ticks in >=2 adequately sampled cells",
    "interactions": ["trend_efficiency_10s x price_velocity_10s", "trend_efficiency_10s x directional_top5_mlofi_500ms"],
    "permutation": "1000 fixed-seed shuffles of outcomes within Spring period x direction, for ER/velocity/MLOFI high-low and ER vs good-expansion; no model selection",
    "decision": "regime hypothesis only if >=10 per extreme in both periods, same-sign raw high-low, positive better-state quote10s in both, and positive leave-one-week-out Spring effect throughout; if coherent Spring but October inadequate/marginal => possible; if raw state effect but quote nonpositive => too small after execution; else period not explained",
    "no_optimization": True, "no_entry_rule_change": True, "no_2026": True,
}
CONFIG_SHA256 = entry._hash(CONFIG)
CHECKPOINT_VERSION = "es-or-breakout-regime-date-v2"


class RegimeError(RuntimeError):
    pass


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    a = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
    if not len(a): return {"n": 0, "mean": None, "median": None, "trimmed_mean": None,
                           "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    trimmed = np.sort(a)[int(.1*len(a)):len(a)-int(.1*len(a))]
    return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
        "trimmed_mean": float(np.mean(trimmed)), "p25": float(np.quantile(a,.25)),
        "p75": float(np.quantile(a,.75)), "p90": float(np.quantile(a,.9)),
        "min": float(np.min(a)), "max": float(np.max(a))}


def _mean(values: Sequence[float | None]) -> float | None:
    return _stats(values)["mean"]


def _features(e: Mapping[str, Any]) -> dict[str, float | None]:
    f = e["event_anchor_features"]
    start, or_end, _ = parent._windows(e["date"])
    assert or_end == start+30*60*1_000_000_000
    return {"or_width": float(e["opening_range_width_ticks"]),
        "rv_10s": f["realized_volatility_10s"],
        "trend_efficiency_10s": f["trend_efficiency_10s"],
        "price_velocity_10s": f["price_velocity_10s"],
        "minutes_after_10": float((int(e["timestamp_ns"])-or_end)/60_000_000_000),
        "directional_top5_mlofi_500ms": f["mlofi_500ms"]}


def _anchor_index(tape: np.ndarray, e: Mapping[str, Any]) -> int:
    ix = int(e["tape_index"])
    ts = tape["timestamp_ns"]
    while ix < len(tape) and ts[ix] < parent._windows(e["date"])[2]:
        if (ts[ix] == int(e["raw_anchor_time_ns"]) and
            math.isfinite(float(tape["bid"][ix])) and math.isfinite(float(tape["ask"][ix])) and
            tape["ask"][ix] > tape["bid"][ix]):
            return ix
        ix += 1
    raise RegimeError("frozen raw midpoint anchor not found")


def _lifecycle(tape: np.ndarray, e: Mapping[str, Any]) -> dict[str, Any]:
    ts = tape["timestamp_ns"]
    t = int(e["timestamp_ns"]); ix = int(e["tape_index"]); sign = int(e["sign"])
    _, _, close = parent._windows(e["date"])
    end = int(np.searchsorted(ts, close, side="left"))
    if ix >= end: raise RegimeError("event outside RTH")
    trade_ix = np.flatnonzero((tape["execution_size"][ix:end] > 0) &
                             np.isfinite(tape["execution_price"][ix:end]))+ix
    if not len(trade_ix) or trade_ix[0] != ix: raise RegimeError("frozen breakout is not an actual trade")
    prices = tape["execution_price"][trade_ix]
    boundary = float(e["boundary"]); high = float(e["opening_range_high"]); low = float(e["opening_range_low"])
    distance = sign*(prices-boundary)/parent.TICK
    inside = prices <= high if sign > 0 else prices >= low
    inside[0] = False
    ret_pos = np.flatnonzero(inside)
    ret_at = int(ret_pos[0]) if len(ret_pos) else None
    before = distance[:ret_at] if ret_at is not None else distance
    if not len(before): raise RegimeError("empty pre-return expansion")
    best = float(np.max(before)); max_at = int(np.argmax(before))
    if best < 1-1e-9: raise RegimeError("strict breakout distance < one tick")
    first8 = np.flatnonzero(before >= 8-1e-9)
    good = "GOOD_EXPANSION_EVENT" if len(first8) else (
        "WEAK_OR_FAILED_EVENT" if ret_at is not None else "UNRESOLVED")
    opposite = prices < low if sign > 0 else prices > high
    opposite[0] = False
    opposite_ix = np.flatnonzero(opposite)
    return_ms = float((ts[trade_ix[ret_at]]-t)/1e6) if ret_at is not None else None
    time_8_ms = float((ts[trade_ix[int(first8[0])]]-t)/1e6) if len(first8) else None
    if len(opposite_ix): topology = "OPPOSITE_RANGE_BREAK"
    elif ret_at is None: topology = "NO_RETURN_BEFORE_CLOSE"
    elif len(first8) and time_8_ms is not None and time_8_ms <= 2000: topology = "FAST_CONTINUATION"
    elif len(first8): topology = "CONTINUATION_THEN_RETURN"
    elif return_ms is not None and return_ms <= 2000: topology = "IMMEDIATE_REJECTION"
    elif return_ms is not None: topology = "STAGNATION_THEN_RETURN"
    else: topology = "UNCLASSIFIED"
    anchor = _anchor_index(tape,e); mid = (tape["bid"]+tape["ask"])/2
    raw_mid = float(e["raw_anchor_mid"])
    excursions = {}
    for h in HORIZONS:
        stop = parent._future(ts,t+h*1_000_000,close)
        if stop is None: excursions[str(h)] = None; continue
        idx = np.arange(anchor,stop+1)
        signed = sign*(mid[idx]-raw_mid)/parent.TICK
        valid = np.isfinite(signed)
        idx = idx[valid]; signed = signed[valid]
        if not len(signed): excursions[str(h)] = None; continue
        imax = int(np.argmax(signed)); imin = int(np.argmin(signed))
        excursions[str(h)] = {"mfe_ticks": float(max(0,signed[imax])),
            "mae_ticks": float(min(0,signed[imin])),
            "time_to_mfe_ms": float((ts[idx[imax]]-t)/1e6),
            "time_to_mae_ms": float((ts[idx[imin]]-t)/1e6)}
    return {"return_inside_ms": return_ms,
        "return_status": "RETURN_INSIDE_OR" if ret_at is not None else "NO_RETURN_BEFORE_RTH_CLOSE",
        "time_spent_outside_before_first_return_ms": return_ms if return_ms is not None else float((close-t)/1e6),
        "favorable_expansion_before_return_ticks": best,
        "time_to_max_expansion_before_return_ms": float((ts[trade_ix[max_at]]-t)/1e6),
        "max_adverse_distance_before_return_ticks": float(min(0,np.min(
            sign*((prices[:ret_at] if ret_at is not None else prices)-float(e["trade_price"]))
            /parent.TICK))),
        "opposite_or_break_before_close": bool(len(opposite_ix)),
        "time_to_opposite_break_ms": float((ts[trade_ix[opposite_ix[0]]]-t)/1e6) if len(opposite_ix) else None,
        "good_expansion_class": good, "time_to_8_ticks_ms": time_8_ms,
        "topology": topology, "excursions": excursions}


def evaluate_date(day: str, tape: np.ndarray, frozen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if day not in flow.TARGET_DATES: raise RegimeError("ineligible date")
    result = []
    for e in frozen:
        if e["date"] != day or e["period"] != parent._period(day): raise RegimeError("frozen event date/period mismatch")
        if e["direction"] not in ("LONG", "SHORT") or e["overshoot_ticks"] != 1:
            raise RegimeError("frozen event semantics mismatch")
        result.append({"date":day,"period":e["period"],"direction":e["direction"],
            "timestamp_ns":e["timestamp_ns"],"tape_index":e["tape_index"],
            "features":_features(e), "time_of_day_bucket":e["time_of_day_bucket"],
            "paths":e["paths"], "parent_topology":e["topology"],
            **_lifecycle(tape,e)})
    if len(result)>2 or len({e["direction"] for e in result})!=len(result):
        raise RegimeError("more than one first event per direction")
    return {"date":day,"events":result}


def _path_mean(events: Sequence[Mapping[str, Any]], h: int, path: str) -> float | None:
    return _mean([e["paths"][str(h)][path] for e in events])


def _period(events: Sequence[Mapping[str, Any]], period: str) -> list[Mapping[str, Any]]:
    return [e for e in events if e["period"] == period]


def _cutpoints(spring: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    result = {}
    for feature in FEATURES:
        values = [e["features"][feature] for e in spring if e["features"][feature] is not None]
        if len(values) < 3: raise RegimeError(f"insufficient Spring feature observations: {feature}")
        result[feature] = [float(x) for x in np.quantile(values,[1/3,2/3])]
    return result


def _tercile(value: float | None, cuts: Sequence[float]) -> str | None:
    if value is None or not math.isfinite(value): return None
    return "LOW" if value <= cuts[0] else ("MID" if value <= cuts[1] else "HIGH")


def _assign_features(events: list[dict[str, Any]], cuts: Mapping[str, Sequence[float]]) -> None:
    history = []
    for day in flow.TARGET_DATES:
        rows = [e for e in events if e["date"]==day]
        for e in rows:
            e["feature_terciles"] = {f:_tercile(e["features"][f],cuts[f]) for f in FEATURES}
            width=e["features"]["or_width"]
            e["prior_date_or_width_percentile"] = (float(100*np.mean(np.asarray(history)<=width))
                if history else None)
        history.extend(e["features"]["or_width"] for e in rows)


def _feature_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {"n":n,"status":"SUFFICIENT" if n>=10 else "INSUFFICIENT_BUCKET_SAMPLE",
        "raw_markouts": {str(h): _stats([e["paths"][str(h)]["raw"] for e in rows])
            for h in (2000,5000,10000,30000,60000)},
        "quote_10s":_stats([e["paths"]["10000"]["quote"] for e in rows]),
        "actual_10s":_stats([e["paths"]["10000"]["actual"] for e in rows]),
        "good_expansion_rate":sum(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT" for e in rows)/n if n else None,
        "favorable_expansion_before_return":_stats([e["favorable_expansion_before_return_ticks"] for e in rows]),
        "return_inside_ms":_stats([e["return_inside_ms"] for e in rows]),
        "topology_counts":{c:sum(e["topology"]==c for e in rows) for c in PATH_CLASSES},
        "mfe_mae":{str(h):{key:_stats([e["excursions"][str(h)][key] for e in rows
            if e["excursions"][str(h)] is not None]) for key in ("mfe_ticks","mae_ticks","time_to_mfe_ms")}
            for h in (10000,30000,60000)}}


def _feature_table(events: Sequence[Mapping[str, Any]],
                   periods: Sequence[str]) -> dict[str, Any]:
    return {feature:{period:{label:_feature_cell([e for e in _period(events,period)
        if e["feature_terciles"][feature]==label]) for label in TERCILES}
        for period in periods}
        for feature in FEATURES}


def _shape(cells: Mapping[str, Any]) -> str:
    if any(cells[k]["n"]<10 for k in TERCILES): return "INSUFFICIENT_SAMPLE"
    means = [cells[k]["raw_markouts"]["10000"]["mean"] for k in TERCILES]
    a,b,c = means
    if None in means: return "INSUFFICIENT_SAMPLE"
    span=max(means)-min(means)
    if span < 1: return "NO_CLEAR_SHAPE"
    if a<b<c: return "SATURATION" if abs(c-b)<=.25*span else "MONOTONIC_POSITIVE"
    if a>b>c: return "SATURATION" if abs(b-a)<=.25*span else "MONOTONIC_NEGATIVE"
    if b>a and b>c: return "INVERTED_U"
    if b<a and b<c: return "U_SHAPED"
    return "NO_CLEAR_SHAPE"


def _spring_interpretation(feature_table: Mapping[str, Any]) -> dict[str, Any]:
    return {f:{"shape":_shape(feature_table[f]["SPRING_2025"]),
        "high_minus_low_raw_10s":(
            feature_table[f]["SPRING_2025"]["HIGH"]["raw_markouts"]["10000"]["mean"]-
            feature_table[f]["SPRING_2025"]["LOW"]["raw_markouts"]["10000"]["mean"])
        if feature_table[f]["SPRING_2025"]["HIGH"]["n"] and
           feature_table[f]["SPRING_2025"]["LOW"]["n"] else None}
        for f in FEATURES}


def _compatibility(table: Mapping[str, Any], spring: Mapping[str, Any]) -> dict[str, Any]:
    result={}
    for f in FEATURES:
        oct_cells=table[f]["OCTOBER_2025"]
        if any(oct_cells[k]["n"]<10 for k in ("LOW","HIGH")):
            label="INSUFFICIENT"
        else:
            delta=oct_cells["HIGH"]["raw_markouts"]["10000"]["mean"]-oct_cells["LOW"]["raw_markouts"]["10000"]["mean"]
            prior=spring[f]["high_minus_low_raw_10s"]
            if prior is None or abs(prior)<1: label="NO_EFFECT"
            elif delta*prior<0: label="OPPOSITE"
            elif abs(delta)<1: label="NO_EFFECT"
            elif abs(delta)<.5*abs(prior): label="WEAKER_BUT_COMPATIBLE"
            else: label="SAME_DIRECTION"
        result[f]=label
    return result


def _composition(events: Sequence[Mapping[str, Any]], table: Mapping[str, Any]) -> dict[str, Any]:
    result={}
    for f in FEATURES:
        s=table[f]["SPRING_2025"]; o=table[f]["OCTOBER_2025"]
        ns=sum(s[k]["n"] for k in TERCILES); no=sum(o[k]["n"] for k in TERCILES)
        shares={k:{"spring":s[k]["n"]/ns if ns else None,"october":o[k]["n"]/no if no else None} for k in TERCILES}
        adequate=[k for k in TERCILES if s[k]["n"]>=10 and o[k]["n"]>=10]
        composition=any(abs(shares[k]["spring"]-shares[k]["october"])>=.15 for k in TERCILES)
        conditional=sum(abs(s[k]["raw_markouts"]["10000"]["mean"]-o[k]["raw_markouts"]["10000"]["mean"])>=2
                        for k in adequate)>=2
        classification=("INSUFFICIENT" if len(adequate)<2 else "BOTH" if composition and conditional else
            "COMPOSITION_DIFFERENCE_ONLY" if composition else "CONDITIONAL_OUTCOME_DIFFERENCE_ONLY" if conditional else "NEITHER")
        result[f]={"classification":classification,"tercile_shares":shares,"adequate_conditional_cells":adequate,
            "spring_conditional_raw10s":{k:s[k]["raw_markouts"]["10000"]["mean"] for k in TERCILES},
            "october_conditional_raw10s":{k:o[k]["raw_markouts"]["10000"]["mean"] for k in TERCILES}}
    return result


def _interaction(events: Sequence[Mapping[str, Any]], second: str) -> dict[str, Any]:
    result={}
    for period in ("SPRING_2025","OCTOBER_2025"):
        result[period]={}
        for a in TERCILES:
            result[period][a]={}
            for b in TERCILES:
                rows=[e for e in _period(events,period) if e["feature_terciles"]["trend_efficiency_10s"]==a
                      and e["feature_terciles"][second]==b]
                result[period][a][b]={"n":len(rows),"status":"SUFFICIENT" if len(rows)>=10
                    else "INSUFFICIENT_INTERACTION_SAMPLE",
                    "raw_10s":_path_mean(rows,10000,"raw"),
                    "good_expansion_rate":_feature_cell(rows)["good_expansion_rate"]}
    return result


def _good_groups(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result={}
    for period in ("SPRING_2025","OCTOBER_2025"):
        result[period]={}
        for label in ("GOOD_EXPANSION_EVENT","WEAK_OR_FAILED_EVENT","UNRESOLVED"):
            rows=[e for e in _period(events,period) if e["good_expansion_class"]==label]
            result[period][label]={"n":len(rows),"features":{f:_stats([e["features"][f] for e in rows]) for f in FEATURES},
                "tercile_counts":{f:{k:sum(e["feature_terciles"][f]==k for e in rows) for k in TERCILES}
                    for f in FEATURES}}
        good=[e for e in _period(events,period) if e["good_expansion_class"]=="GOOD_EXPANSION_EVENT"]
        weak=[e for e in _period(events,period) if e["good_expansion_class"]=="WEAK_OR_FAILED_EVENT"]
        effect={}
        for f in FEATURES:
            a=np.asarray([e["features"][f] for e in good if e["features"][f] is not None],dtype=float)
            b=np.asarray([e["features"][f] for e in weak if e["features"][f] is not None],dtype=float)
            if len(a)<2 or len(b)<2: effect[f]=None;continue
            pooled=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
            effect[f]=float((np.mean(a)-np.mean(b))/pooled) if pooled>0 else None
        result[period]["standardized_good_minus_weak_effects"]=effect
    return result


def _permutation(spring: Sequence[Mapping[str, Any]], seed: int = 20251005) -> dict[str, Any]:
    rng=np.random.default_rng(seed)
    direction=np.asarray([e["direction"] for e in spring])
    raw=np.asarray([e["paths"]["10000"]["raw"] for e in spring],dtype=float)
    good=np.asarray([e["good_expansion_class"]=="GOOD_EXPANSION_EVENT" for e in spring],dtype=float)
    result={}
    for f,outcome,label in (("trend_efficiency_10s",raw,"raw10s"),
                             ("price_velocity_10s",raw,"raw10s"),
                             ("directional_top5_mlofi_500ms",raw,"raw10s"),
                             ("trend_efficiency_10s",good,"good_expansion")):
        states=np.asarray([e["feature_terciles"][f] for e in spring])
        high=states=="HIGH"; low=states=="LOW"
        if high.sum()<10 or low.sum()<10:
            result[f+"_vs_"+label]={"status":"INSUFFICIENT_BUCKET_SAMPLE"};continue
        observed=float(np.mean(outcome[high])-np.mean(outcome[low]))
        null=[]
        for _ in range(1000):
            shuffled=outcome.copy()
            for d in ("LONG","SHORT"):
                ix=np.flatnonzero(direction==d)
                shuffled[ix]=rng.permutation(shuffled[ix])
            null.append(float(np.mean(shuffled[high])-np.mean(shuffled[low])))
        null_a=np.asarray(null)
        result[f+"_vs_"+label]={"status":"DESCRIPTIVE_PERMUTATION", "observed_high_minus_low":observed,
            "null_percentile":float(np.mean(null_a<=observed)),
            "two_sided_p":float((1+np.sum(np.abs(null_a)>=abs(observed)))/(len(null_a)+1)),
            "permutations":1000,"seed":seed,"strata":"Spring direction"}
    return result


def _week(day: str) -> str:
    y,w,_=date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _date_week_tables(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any],dict[str, Any]]:
    daily={}
    for day in flow.TARGET_DATES:
        rows=[e for e in events if e["date"]==day]
        daily[day]={"n":len(rows),"period":parent._period(day),"raw_10s":_path_mean(rows,10000,"raw"),
            "mfe_10s":_stats([e["excursions"]["10000"]["mfe_ticks"] for e in rows]),
            "median_return_inside_ms":_stats([e["return_inside_ms"] for e in rows])["median"],
            "good_expansion_count":sum(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT" for e in rows),
            "feature_states":[{"direction":e["direction"],"terciles":e["feature_terciles"]} for e in rows]}
    weekly={}
    for week in sorted({_week(d) for d in flow.TARGET_DATES}):
        rows=[e for e in events if _week(e["date"])==week]
        weekly[week]={"n":len(rows),"raw_10s":_path_mean(rows,10000,"raw"),
            "good_expansion_rate":_mean([float(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT") for e in rows]),
            "median_return_inside_ms":_stats([e["return_inside_ms"] for e in rows])["median"],
            "feature_distributions":{f:_stats([e["features"][f] for e in rows]) for f in FEATURES}}
    return daily,weekly


def _stability(spring: Sequence[Mapping[str, Any]], by: str) -> dict[str, Any]:
    # ER is the first predeclared directional-state variable; never choose a
    # different predictor after inspecting October or leave-out results.
    keys=list(flow.SPRING_DATES) if by=="date" else sorted({_week(d) for d in flow.SPRING_DATES})
    rows={}
    for key in keys:
        subset=[e for e in spring if (e["date"] if by=="date" else _week(e["date"]))!=key]
        hi=[e for e in subset if e["feature_terciles"]["trend_efficiency_10s"]=="HIGH"]
        lo=[e for e in subset if e["feature_terciles"]["trend_efficiency_10s"]=="LOW"]
        rows[key]={"n":len(subset),"raw_10s":_path_mean(subset,10000,"raw"),
            "er_high_minus_low_raw_10s":(_path_mean(hi,10000,"raw")-_path_mean(lo,10000,"raw"))
                if hi and lo else None,
            "er_high_minus_low_good_rate":(_mean([float(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT") for e in hi])-
                _mean([float(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT") for e in lo]))
                if hi and lo else None}
    summary={}
    for field in ("raw_10s","er_high_minus_low_raw_10s","er_high_minus_low_good_rate"):
        valid={k:v[field] for k,v in rows.items() if v[field] is not None}
        summary[field]={"tested":len(valid),"positive":sum(x>0 for x in valid.values()),
            "negative":sum(x<0 for x in valid.values()),"median":_stats(list(valid.values()))["median"],
            "min":min(valid.values()) if valid else None,"max":max(valid.values()) if valid else None,
            "worst_omitted":min(valid,key=valid.get) if valid else None,
            "best_omitted":max(valid,key=valid.get) if valid else None}
    return {"predeclared_feature":"trend_efficiency_10s","rows":rows,"sign_stability":summary}


def _decision(table: Mapping[str, Any], spring_shape: Mapping[str, Any],
              compatibility: Mapping[str, Any], lowo: Mapping[str, Any]) -> dict[str, Any]:
    candidates=[]
    for f in FEATURES:
        s=table[f]["SPRING_2025"]; o=table[f]["OCTOBER_2025"]
        effect=spring_shape[f]["high_minus_low_raw_10s"]
        if (spring_shape[f]["shape"] in ("MONOTONIC_POSITIVE","MONOTONIC_NEGATIVE","SATURATION") and
            effect is not None and abs(effect)>=2 and
            all(s[k]["n"]>=10 for k in ("LOW","HIGH"))):
            better="HIGH" if effect>0 else "LOW"
            quote_s=s[better]["quote_10s"]["mean"]
            quote_o=o[better]["quote_10s"]["mean"]
            candidates.append({"feature":f,"better_state":better,"raw_effect":effect,
                "spring_quote_10s":quote_s,"october_quote_10s":quote_o,
                "compatibility":compatibility[f],
                "adequate_october_extremes":all(o[k]["n"]>=10 for k in ("LOW","HIGH"))})
    # One deterministic prospective hypothesis at most; declaration does not
    # convert it into a filter or permit re-use of these dates for testing.
    qualified=[x for x in candidates if x["adequate_october_extremes"] and
        x["compatibility"] in ("SAME_DIRECTION","WEAKER_BUT_COMPATIBLE") and
        x["spring_quote_10s"] is not None and x["spring_quote_10s"]>0 and
        x["october_quote_10s"] is not None and x["october_quote_10s"]>0]
    # Require robustness for the predeclared ER hypothesis only. Other features
    # have no leave-out proof in this frozen protocol, so none can be promoted.
    stable=lowo["sign_stability"]["er_high_minus_low_raw_10s"]
    accepted=[x for x in qualified if x["feature"]=="trend_efficiency_10s" and
              stable["tested"]>0 and stable["negative"]==0 and stable["positive"]==stable["tested"]]
    if accepted:
        decision="REGIME_HYPOTHESIS_IDENTIFIED"
        candidate={"feature":"trend_efficiency_10s","state":accepted[0]["better_state"],
            "definition":"Spring-frozen ER tercile under the existing causal 10s definition; hypothesis only, not an entry gate"}
        next_step="FREEZE_ONE_REGIME_HYPOTHESIS_AND_RUN_ONE_FRESH_CALIBRATION_TEST"
    elif candidates and any(x["compatibility"] not in ("OPPOSITE",) for x in candidates):
        if all((x["spring_quote_10s"] or -math.inf)<=0 and (x["october_quote_10s"] or -math.inf)<=0 for x in candidates):
            decision="STRUCTURAL_EDGE_TOO_SMALL_AFTER_EXECUTION"
            next_step="STOP_30M_OR_BREAKOUT_BRANCH_UNDER_CURRENT_EXECUTION_MODEL"
        else:
            decision="REGIME_EFFECT_POSSIBLE_BUT_UNCONFIRMED"
            next_step="ONE_FRESH_CALIBRATION_TEST_ONLY"
        candidate=None
    else:
        decision="PERIOD_DIFFERENCE_NOT_EXPLAINED"
        next_step="STOP_30M_OR_BREAKOUT_BRANCH"
        candidate=None
    return {"primary_decision":decision,"next_step":next_step,"candidate_hypothesis":candidate,
            "spring_coherent_features":candidates,"qualified_features":qualified}


def _checkpoint(root: Path, day: str) -> Path:
    return root/"checkpoints"/f"{day}.json.gz"


def _read_checkpoint(path: Path, day: str, source_sha: str, tape_sha: str,
                     parent_sha: str, entry_sha: str) -> dict[str, Any] | None:
    if not path.is_file(): return None
    try:
        with gzip.open(path,"rt",encoding="utf-8") as f: row=json.load(f)
    except (OSError,EOFError,json.JSONDecodeError): return None
    expected={"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,"date":day,
        "source_sha256":source_sha,"tape_sha256":tape_sha,"parent_manifest_sha256":parent_sha,
        "entry_manifest_sha256":entry_sha,"config_sha256":CONFIG_SHA256}
    return row if all(row.get(k)==v for k,v in expected.items()) else None


def _entry_manifest(root: Path, parent_sha: str) -> str:
    path=root/"run-manifest.json"; m=json.loads(path.read_text())
    if (m.get("status")!="COMPLETE" or m.get("config_sha256")!=entry.CONFIG_SHA256 or
        m.get("parent_manifest_sha256")!=parent_sha):
        raise RegimeError("entry audit incomplete or changed")
    for name,digest in m["artifact_sha256_by_name"].items():
        if native._sha(root/name)!=digest: raise RegimeError(f"entry audit artifact changed: {name}")
    for day,digest in m["checkpoint_sha256_by_date"].items():
        if native._sha(root/"checkpoints"/f"{day}.json.gz")!=digest:
            raise RegimeError(f"entry audit date checkpoint changed: {day}")
    return native._sha(path)


def run(*, data_root: Path=native.DATA_ROOT, parent_root: Path=parent.OUT_ROOT,
        entry_root: Path=entry.OUT_ROOT, output_root: Path=OUT_ROOT,
        smoke: bool=False) -> dict[str, Any]:
    started=time.monotonic()
    _,coverage,frozen,parent_sha=entry._parent_package(parent_root)
    entry_sha=_entry_manifest(entry_root,parent_sha)
    _,catalog=native._source_catalog(data_root)
    if {d:catalog[d]["sha256"] for d in native.ALL_SOURCE_DATES}!=coverage["source_sha256_by_date"]:
        raise RegimeError("native source differs from frozen parent")
    days=flow.TARGET_DATES[:1] if smoke else flow.TARGET_DATES
    output_root.mkdir(parents=True,exist_ok=True)
    native._write_json(output_root/"study-config.json",CONFIG)
    payloads=[]
    for pos,day in enumerate(days,1):
        tape_path=native._tape_path(day); tape_sha=native._sha(tape_path)
        if tape_sha!=coverage["tape_sha256_by_date"][day]: raise RegimeError(f"tape hash mismatch: {day}")
        source_sha=catalog[day]["sha256"]
        cp=_checkpoint(output_root,day)
        cached=_read_checkpoint(cp,day,source_sha,tape_sha,parent_sha,entry_sha)
        if cached:
            payload=cached["payload"]
            print(f"REGIME_DATE_RESUME={day}",flush=True)
        else:
            print(f"REGIME_DATE_START={pos}/{len(days)} date={day}",flush=True)
            tape,_=native._load_tape(day,tape_path,source_sha)
            payload=evaluate_date(day,tape,frozen.get(day,[]))
            native._write_checkpoint(cp,{"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,
                "date":day,"source_sha256":source_sha,"tape_sha256":tape_sha,
                "parent_manifest_sha256":parent_sha,"entry_manifest_sha256":entry_sha,
                "config_sha256":CONFIG_SHA256,"payload":payload})
            print(f"REGIME_DATE_COMPLETE={day} events={len(payload['events'])}",flush=True)
        payloads.append(payload)
    if smoke:return {"status":"SMOKE_PASS","dates":list(days),"events":sum(len(p["events"]) for p in payloads)}
    events=[e for p in payloads for e in p["events"]]
    if len(events)!=77 or sum(e["direction"]=="LONG" for e in events)!=38:
        raise RegimeError("frozen event-count parity failure")
    spring=_period(events,"SPRING_2025"); october=_period(events,"OCTOBER_2025")
    cuts=_cutpoints(spring)
    _assign_features(events,cuts)
    spring_table=_feature_table(spring,("SPRING_2025",))
    spring_interpretation=_spring_interpretation(spring_table)
    # This artifact is finalized before any October compatibility/decision work.
    native._write_json(output_root/"spring-feature-interpretation.json",spring_interpretation)
    october_table=_feature_table(october,("OCTOBER_2025",))
    table={f:{**spring_table[f],**october_table[f]} for f in FEATURES}
    compatibility=_compatibility(table,spring_interpretation)
    composition=_composition(events,table)
    interactions={"trend_efficiency_x_price_velocity":_interaction(events,"price_velocity_10s"),
        "trend_efficiency_x_mlofi":_interaction(events,"directional_top5_mlofi_500ms")}
    perm=_permutation(spring)
    daily,weekly=_date_week_tables(events)
    lodo=_stability(spring,"date"); lowo=_stability(spring,"week")
    decision=_decision(table,spring_interpretation,compatibility,lowo)
    periods={p:{"n":len(rows),"raw_markouts":{str(h):_stats([e["paths"][str(h)]["raw"] for e in rows]) for h in HORIZONS},
        "quote_10s":_stats([e["paths"]["10000"]["quote"] for e in rows]),
        "actual_10s":_stats([e["paths"]["10000"]["actual"] for e in rows]),
        "return_inside_n":sum(e["return_inside_ms"] is not None for e in rows),
        "return_inside_ms":_stats([e["return_inside_ms"] for e in rows]),
        "favorable_expansion_before_return":_stats([e["favorable_expansion_before_return_ticks"] for e in rows]),
        "good_expansion_n":sum(e["good_expansion_class"]=="GOOD_EXPANSION_EVENT" for e in rows),
        "topology_counts":{c:sum(e["topology"]==c for e in rows) for c in PATH_CLASSES},
        "mfe_mae":{str(h):{k:_stats([e["excursions"][str(h)][k] for e in rows if e["excursions"][str(h)] is not None])
            for k in ("mfe_ticks","mae_ticks","time_to_mfe_ms","time_to_mae_ms")} for h in HORIZONS}}
        for p,rows in (("SPRING_2025",spring),("OCTOBER_2025",october))}
    source={"status":"PASS","native_es_mbp10_only":True,"no_mes":True,"no_mbo":True,
        "parent_manifest_sha256":parent_sha,"entry_manifest_sha256":entry_sha,
        "parent_source_coverage_sha256":native._sha(parent_root/"source-coverage.json"),
        "spring_dates":list(flow.SPRING_DATES),"october_dates":list(flow.OCTOBER_DATES),
        "source_sha256_by_date":coverage["source_sha256_by_date"],
        "tape_sha256_by_date":coverage["tape_sha256_by_date"]}
    summary={"run_id":RUN_ID,"status":"COMPLETE","total_events":77,"long_events":38,"short_events":39,
        "periods":periods,"spring_feature_interpretation":spring_interpretation,
        "october_compatibility":compatibility,"composition_vs_conditional":composition,
        "decision":decision,"config_sha256":CONFIG_SHA256,
        "optimization_performed":False,"optuna_performed":False,"threshold_search_performed":False,
        "feature_search_performed":False,"entry_rule_changed":False,
        "stop_target_search_performed":False,"final_oos_accessed":False,
        "data_downloaded":False,"commit_performed":False,"elapsed_seconds":time.monotonic()-started}
    native._write_gzip_jsonl(output_root/"events.jsonl.gz",events)
    files={"summary.json":summary,"source-coverage.json":source,
        "breakout-lifecycle.json":periods,"return-inside-analysis.json":{p:{k:v[k] for k in ("return_inside_n","return_inside_ms")}
            for p,v in periods.items()},
        "expansion-before-return.json":{p:v["favorable_expansion_before_return"] for p,v in periods.items()},
        "topology-results.json":{p:v["topology_counts"] for p,v in periods.items()},
        "feature-values.json":[{"date":e["date"],"direction":e["direction"],
            "features":e["features"],"prior_date_or_width_percentile":e["prior_date_or_width_percentile"]} for e in events],
        "feature-terciles.json":{"spring_cutpoints":cuts,"cells":table,"spring_interpretation":spring_interpretation,
            "october_compatibility":compatibility},
        "or-width-results.json":table["or_width"],"rv-results.json":table["rv_10s"],
        "trend-efficiency-results.json":table["trend_efficiency_10s"],
        "price-velocity-results.json":table["price_velocity_10s"],
        "breakout-time-results.json":table["minutes_after_10"],
        "mlofi-results.json":table["directional_top5_mlofi_500ms"],
        "good-expansion-results.json":_good_groups(events),
        "period-feature-distributions.json":{p:{f:_stats([e["features"][f] for e in rows]) for f in FEATURES}
            for p,rows in (("SPRING_2025",spring),("OCTOBER_2025",october))},
        "composition-vs-conditional.json":composition,"interaction-results.json":interactions,
        "permutation-results.json":perm,
        "raw-vs-executable.json":{p:{k:v[k] for k in ("raw_markouts","quote_10s","actual_10s")}
            for p,v in periods.items()},
        "daily-results.json":daily,"weekly-results.json":weekly,
        "lodo-results.json":lodo,"lowo-results.json":lowo}
    for name,value in files.items():native._write_json(output_root/name,value)
    report=[f"# {RUN_ID}","", "Spring = primary discovery; October = secondary DEV compatibility.",
        "Only six frozen pre-breakout features; no strategy or feature search.",
        f"Events: 77 (38 LONG, 39 SHORT). Spring raw 10s {periods['SPRING_2025']['raw_markouts']['10000']['mean']:.3f}; October raw 10s {periods['OCTOBER_2025']['raw_markouts']['10000']['mean']:.3f} ticks.",
        f"Full-RTH first trade return inside OR: Spring {periods['SPRING_2025']['return_inside_n']}/51; October {periods['OCTOBER_2025']['return_inside_n']}/26. This differs from the prior 60-second topology window by design.",
        f"Median pre-return expansion: Spring {periods['SPRING_2025']['favorable_expansion_before_return']['median']:.1f}; October {periods['OCTOBER_2025']['favorable_expansion_before_return']['median']:.1f} ES ticks. +8-tick good events: Spring {periods['SPRING_2025']['good_expansion_n']}/51; October {periods['OCTOBER_2025']['good_expansion_n']}/26.",
        "The canonical validated pre-event price controls use one trailing 10-second 500ms-grid RV, ER, and signed velocity. RV30 was not present in the validated parent event package.",
        "Spring low-ER and low-velocity terciles show stronger raw 10-second markouts; October extreme-tercile cells fail the preregistered N>=10 compatibility rule and their lower-state quote markouts are negative.",
        "Spring high-minus-low directional 500ms MLOFI raw effect is only +0.176 tick; no incremental L2 claim is supported. Both fixed 3x3 interactions are sparse.",
        "Spring-only feature interpretation was written before October outcome aggregation. Spring tercile cutpoints are reused unchanged for October; no threshold or feature is selected as a trading gate.",
        f"Primary decision: {decision['primary_decision']}.",
        f"Candidate hypothesis: {decision['candidate_hypothesis'] or 'NONE'}.",
        f"Next step: {decision['next_step']}.", ""]
    (output_root/"report.md").write_text("\n".join(report),encoding="utf-8")
    artifacts={p.name:native._sha(p) for p in output_root.iterdir() if p.is_file()
        and p.name not in ("run-manifest.json","artifact-hashes.json")}
    native._write_json(output_root/"run-manifest.json",{"status":"COMPLETE","run_id":RUN_ID,
        "config_sha256":CONFIG_SHA256,"parent_manifest_sha256":parent_sha,
        "entry_manifest_sha256":entry_sha,"source_coverage_sha256":native._sha(output_root/"source-coverage.json"),
        "checkpoint_sha256_by_date":{d:native._sha(_checkpoint(output_root,d)) for d in days},
        "artifact_sha256_by_name":artifacts,"no_2026_access":True,"no_download":True,"no_optimization":True})
    native._write_json(output_root/"artifact-hashes.json",{"status":"HASHED","files":
        {p.name:native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name!="artifact-hashes.json"}})
    return summary


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=native.DATA_ROOT)
    parser.add_argument("--parent-root",type=Path,default=parent.OUT_ROOT)
    parser.add_argument("--entry-root",type=Path,default=entry.OUT_ROOT)
    parser.add_argument("--output-root",type=Path,default=OUT_ROOT)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv)
    try:result=run(data_root=args.data_root,parent_root=args.parent_root,
        entry_root=args.entry_root,output_root=args.output_root,smoke=args.smoke)
    except (RegimeError,entry.EntryAuditError,parent.BreakoutStudyError,
            native.VacuumStudyError,OSError,ValueError,KeyError,AssertionError) as exc:
        parser.exit(2,f"REGIME_ERROR: {exc}\n")
    print(f"ES_OR_BREAKOUT_REGIME_DEPENDENCE_STUDY={result['status']}",flush=True)
    if result["status"]=="COMPLETE":print(f"PRIMARY_DECISION={result['decision']['primary_decision']}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
