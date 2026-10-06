"""Public-model ES OR/profile/retrace event audit; never proprietary IVB reproduction."""
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
from . import mac_2025_es_structural_breakout_entry_timing_audit as prior
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_flow_momentum_v1 as flow
from . import asia_w04_structural_matrix as profile_module

RUN_ID="CMEOrderflow_FABERVAALE_PUBLIC_IVB_REPLICATION_EVENT_STUDY_V1"
OUT_ROOT=Path("research_runs")/RUN_ID
HORIZONS=parent.HORIZONS_MS
EXCURSIONS=parent.EXCURSION_MS
BARRIERS=parent.BARRIERS
COMPONENTS=("aggressive_delta_ratio","opposing_flow_price_efficiency",
            "directional_top5_mlofi","passive_defense_ratio")
PERIODS=("ALL","SPRING_2025","OCTOBER_2025","LONG","SHORT")
CONFIG={
    "version":2,"spring_dates":list(flow.SPRING_DATES),"october_dates":list(flow.OCTOBER_DATES),
    "roles":{"SPRING_2025":"PRIMARY_DEV_DISCOVERY","OCTOBER_2025":"SECONDARY_DEV_COMPATIBILITY"},
    "source":"hash-bound existing native ES GLBX.MDP3 mbp-10 canonical tapes and compact TOP5 MLOFI; no MES/MBO",
    "parent_run_id":parent.RUN_ID,"parent_config_sha256":parent.CONFIG_SHA256,
    "opening_range":"09:30:00 ET inclusive to 10:00:00 ET exclusive, actual ES trade high/low",
    "profile":"actual trade volume by integer ES tick during OR; canonical asia_volume_profile 70%; lower POC tie, adjacent expansion lower tie",
    "frame":"earliest of parent frozen first strict LONG/SHORT breakouts after 10:00; one direction only; no flip",
    "zone":{"LONG":"[POC,VAH]","SHORT":"[VAL,POC]"},
    "retrace":"first subsequent actual trade whose price is inclusively inside directional zone; one event per session; no gap-cross inference",
    "confirmation":"retrace trade index through retrace timestamp +500ms inclusive; confirmed raw anchor strictly AFTER +500ms to avoid equal-timestamp order ambiguity; no post-window leakage",
    "aggression":"signed actual executed size, tape aggressor +1 BUY/-1 SELL; ratio signed delta/(total+1e-12)",
    "effort_result":"max adverse midpoint progress from retrace event through confirmation in ticks/(opposing aggressive size+1); missing if no opposing aggressive size",
    "mlofi":"signed sum validated price-keyed inverse-rank TOP5 MLOFI compact rows on [retrace timestamp,confirmation timestamp]/as-of positive weighted TOP5 denominator",
    "passive_defense":"(defending-side TOP5 size as-of confirm - at retrace)/max(retrace TOP5 size,1); positive replenishment; zero stable",
    "support":"delta ratio>0; opposing volume>0 AND adverse progress<=0; MLOFI>0; passive defense ratio>=0. No fitted weights or gates",
    "support_groups":{"LOW":[0,1],"MID":[2],"HIGH":[3,4]},
    "component_terciles":"Spring discovery cutpoints q1/3 and q2/3 on valid confirmed events; unchanged in October; cell N<10 insufficient",
    "price_only":"nearest LOW-support comparator for each HIGH-support event, same period/direction, with replacement; fixed scaled squared distance of pre-retrace expansion/8, breakout-to-retrace seconds/120, normalized zone depth/0.33, pre-retrace price velocity/2, RV/5, distance to POC/4; no outcomes or L2 used",
    "location":"fixed thirds of directional [POC,VA boundary] zone; zero-width zone reported as POC_ONLY",
    "timing_buckets_seconds":[30,120,300,900],
    "anchors":"retrace raw mid first valid at/after trade row; confirmed raw mid first valid strictly after t+500ms; quote first valid at/after confirmation timestamp+2ms; actual fill entry-side quote plus adverse 1 ES tick",
    "reload":"from confirmation raw mid, first +8 ES ticks before trade through invalidation edge = SUCCESSFUL; invalidation first = STRUCTURAL_INVALIDATION; +4 then failure = PARTIAL; -2 within 2s = IMMEDIATE_FAILURE; neither +/-2 within 60s = STAGNATION; else UNRESOLVED; no stop/PnL",
    "barriers":[{"favorable":a,"adverse":b} for a,b in BARRIERS],
    "permutation":"1000 fixed-seed 20251006 outcome shuffles within period x direction; five predeclared component/support-count tests; descriptive",
    "decision":"minimum 10 complete retraces in each period; promising requires positive actual-fill 10s both periods, adequate LOW/HIGH support cells, positive support effect both, >=8 price-only matched pairs and positive incremental effect, leave-out sign stability, one coherent component; possible-but-unconfirmed when Spring support high-low quote effect >2 ticks but support cells too small; no outcome-based reselection",
    "no_optimization":True,"no_passive_fill":True,"no_proprietary_targets":True,"no_2026":True,
}
CONFIG_SHA256=prior._hash(CONFIG)
CHECKPOINT_VERSION="public-ivb-date-v2"


class PublicIVBError(RuntimeError):pass


def _stats(values: Sequence[float | None]) -> dict[str,Any]:
    a=np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))],dtype=float)
    if not len(a):return {"n":0,"mean":None,"median":None,"trimmed_mean":None,"p25":None,
                          "p75":None,"positive_fraction":None,"negative_fraction":None,"min":None,"max":None}
    k=int(.1*len(a)); sorted_a=np.sort(a)
    trimmed=sorted_a[k:len(a)-k]
    return {"n":len(a),"mean":float(np.mean(a)),"median":float(np.median(a)),
        "trimmed_mean":float(np.mean(trimmed)),"p25":float(np.quantile(a,.25)),
        "p75":float(np.quantile(a,.75)),"positive_fraction":float(np.mean(a>0)),
        "negative_fraction":float(np.mean(a<0)),"min":float(np.min(a)),"max":float(np.max(a))}


def _split(events: Sequence[Mapping[str,Any]], label: str) -> list[Mapping[str,Any]]:
    if label=="ALL":return list(events)
    return [e for e in events if e["period"]==label or e["direction"]==label]


def _profile(day: str,tape: np.ndarray) -> dict[str,Any]:
    start,end,close=parent._windows(day)
    ts=tape["timestamp_ns"]
    trade=(tape["execution_size"]>0)&np.isfinite(tape["execution_price"])
    idx=np.flatnonzero(trade&(ts>=start)&(ts<end))
    if not len(idx):raise PublicIVBError(f"no actual OR trades: {day}")
    ticks=np.rint(tape["execution_price"][idx]/parent.TICK).astype(np.int64)
    if np.any(np.abs(ticks*parent.TICK-tape["execution_price"][idx])>1e-8):
        raise PublicIVBError("off-tick actual OR trade")
    size=np.asarray(tape["execution_size"][idx],dtype=np.int64)
    if np.any(size<=0):raise PublicIVBError("invalid executed volume")
    prices,inv=np.unique(ticks,return_inverse=True)
    volume=np.bincount(inv,weights=size).astype(np.int64)
    by_tick={int(p):int(v) for p,v in zip(prices,volume)}
    profile=profile_module.asia_volume_profile(by_tick)
    if profile["high"]!=float(np.max(tape["execution_price"][idx])) or profile["low"]!=float(np.min(tape["execution_price"][idx])):
        raise PublicIVBError("profile trade-price OR mismatch")
    return {"date":day,"or_start_ns":start,"or_end_ns":end,"rth_close_ns":close,
        "or_high":profile["high"],"or_low":profile["low"],"poc":profile["poc"],
        "vah":profile["vah"],"val":profile["val"],
        "or_width_ticks":(profile["high"]-profile["low"])/parent.TICK,
        "profile_total_volume":int(np.sum(size)),"price_volume_by_tick":by_tick}


def _frame(day: str, frozen: Sequence[Mapping[str,Any]], profile: Mapping[str,Any]) -> dict[str,Any]:
    if not frozen:return {"date":day,"status":"NO_BREAKOUT"}
    for e in frozen:
        if e["date"]!=day or e["period"]!=parent._period(day):raise PublicIVBError("frozen event identity mismatch")
        if e["opening_range_high"]!=profile["or_high"] or e["opening_range_low"]!=profile["or_low"]:
            raise PublicIVBError("frozen actual-trade OR differs from profile")
    first=min(frozen,key=lambda e:(e["timestamp_ns"],e["tape_index"]))
    sign=int(first["sign"])
    if sign not in (-1,1) or first["direction"] not in ("LONG","SHORT"):
        raise PublicIVBError("invalid frozen breakout direction")
    low,high=(profile["poc"],profile["vah"]) if sign>0 else (profile["val"],profile["poc"])
    return {"date":day,"status":"FRAME","direction":first["direction"],"sign":sign,
        "breakout_timestamp_ns":int(first["timestamp_ns"]),"breakout_tape_index":int(first["tape_index"]),
        "breakout_price":float(first["trade_price"]),"zone_low":float(low),"zone_high":float(high),
        "invalidation_reference":float(profile["val"] if sign>0 else profile["vah"]),
        "opposite_first_breakout_observed":len(frozen)>1,
        "opposite_first_breakout_timestamp_ns":min((int(e["timestamp_ns"]) for e in frozen if e is not first),default=None)}


def _retrace(tape: np.ndarray, frame: Mapping[str,Any], profile: Mapping[str,Any]) -> dict[str,Any]:
    if frame["status"]!="FRAME":return {"status":"NO_FRAME"}
    ts=tape["timestamp_ns"]
    ix=int(frame["breakout_tape_index"])
    close=int(profile["rth_close_ns"])
    end=int(np.searchsorted(ts,close,side="left"))
    ids=np.flatnonzero((tape["execution_size"][ix+1:end]>0)&
        np.isfinite(tape["execution_price"][ix+1:end]))+ix+1
    prices=tape["execution_price"][ids]
    entry=np.flatnonzero((prices>=frame["zone_low"]-1e-8)&(prices<=frame["zone_high"]+1e-8))
    if not len(entry):return {"status":"NO_RETRACE","date":frame["date"],"direction":frame["direction"]}
    pos=int(entry[0]); ridx=int(ids[pos]); t=int(ts[ridx]); sign=int(frame["sign"])
    pre=prices[:pos]
    signed=sign*(pre-float(frame["breakout_price"]))/parent.TICK
    expansion=float(max(0,np.max(signed))) if len(signed) else 0.0
    adverse=float(min(0,np.min(signed))) if len(signed) else 0.0
    price=float(tape["execution_price"][ridx]); poc=float(profile["poc"])
    zone_width=(float(frame["zone_high"])-float(frame["zone_low"]))/parent.TICK
    # Outer boundary -> POC, normalized 0..1.
    outer=float(frame["zone_high"] if sign>0 else frame["zone_low"])
    depth=(sign*(outer-price)/parent.TICK/zone_width) if zone_width>0 else None
    location=("POC_ONLY" if depth is None else "OUTER_THIRD" if depth<1/3
        else "MIDDLE_THIRD" if depth<2/3 else "POC_THIRD")
    elapsed=(t-int(frame["breakout_timestamp_ns"]))/1e9
    timing=("LE_30S" if elapsed<=30 else "30_TO_120S" if elapsed<=120 else
        "2_TO_5M" if elapsed<=300 else "5_TO_15M" if elapsed<=900 else "GT_15M")
    after=prices[pos:]
    through_poc=bool(np.any(after<=poc)) if sign>0 else bool(np.any(after>=poc))
    crossed_zone=bool(np.any(after<float(frame["zone_low"]))) if sign>0 else bool(np.any(after>float(frame["zone_high"])))
    return {"status":"RETRACE","date":frame["date"],"direction":frame["direction"],
        "timestamp_ns":t,"tape_index":ridx,"price":price,
        "breakout_to_retrace_seconds":float(elapsed),"timing_bucket":timing,
        "max_favorable_expansion_before_retrace_ticks":expansion,
        "max_adverse_move_before_retrace_ticks":adverse,
        "distance_breakout_to_retrace_ticks":sign*(price-float(frame["breakout_price"]))/parent.TICK,
        "distance_to_poc_ticks":abs(price-poc)/parent.TICK,
        "distance_to_outer_va_ticks":abs(price-outer)/parent.TICK,
        "zone_depth_fraction":float(depth) if depth is not None else None,
        "location_bucket":location,"poc_touched_at_entry":abs(price-poc)<1e-8,
        "poc_touched_or_crossed_after_entry":through_poc,"crossed_entire_directional_zone":crossed_zone}


def _valid_quote(tape: np.ndarray, start_ix: int, close: int) -> int | None:
    ts=tape["timestamp_ns"]
    ix=start_ix
    while ix<len(tape) and ts[ix]<close:
        if (math.isfinite(float(tape["bid"][ix])) and math.isfinite(float(tape["ask"][ix]))
            and tape["ask"][ix]>tape["bid"][ix]):return ix
        ix+=1
    return None


def _anchor(tape: np.ndarray, t: int, close: int, floor: int=0) -> int | None:
    ix=max(floor,int(np.searchsorted(tape["timestamp_ns"],t,side="left")))
    return _valid_quote(tape,ix,close)


def _orderflow(tape: np.ndarray, rows: np.ndarray, retrace: Mapping[str,Any],
               frame: Mapping[str,Any], profile: Mapping[str,Any]) -> dict[str,Any]:
    t=int(retrace["timestamp_ns"]); end=t+500_000_000; sign=int(frame["sign"])
    if end>=int(profile["rth_close_ns"]):return {"status":"CONFIRMATION_OUTSIDE_RTH"}
    ix=int(retrace["tape_index"])
    ts=tape["timestamp_ns"]
    stop=int(np.searchsorted(ts,end,side="right"))
    segment=tape[ix:stop]
    traded=(segment["execution_size"]>0)&np.isin(segment["aggressor"],(-1,1))
    signed=segment["aggressor"][traded].astype(np.int64)*sign
    sizes=segment["execution_size"][traded].astype(np.int64)
    supportive=int(np.sum(sizes[signed>0])); opposing=int(np.sum(sizes[signed<0])); total=supportive+opposing
    delta=supportive-opposing
    ratio=float(delta/(total+1e-12)) if total else None
    start_ix=_valid_quote(tape,ix,int(profile["rth_close_ns"]))
    confirm_ix=_anchor(tape,end,int(profile["rth_close_ns"]),floor=ix)
    if start_ix is None or confirm_ix is None:raise PublicIVBError("missing causal confirmation BBO")
    if int(ts[start_ix])>end:raise PublicIVBError("no BBO during causal confirmation window")
    mid0=float((tape["bid"][start_ix]+tape["ask"][start_ix])/2)
    mids=(segment["bid"].astype(float)+segment["ask"].astype(float))/2
    adverse=float(max(0,-np.nanmin(sign*(mids-mid0)/parent.TICK))) if len(mids) else 0.0
    efficiency=float(adverse/(opposing+1)) if opposing else None
    book_ts=np.asarray(rows["ts"],dtype=np.int64)
    lo=int(np.searchsorted(book_ts,t,side="left"))
    hi=int(np.searchsorted(book_ts,end,side="right"))
    initial=int(np.searchsorted(book_ts,t,side="right")-1)
    final=int(np.searchsorted(book_ts,end,side="right")-1)
    if initial<0 or final<initial:raise PublicIVBError("missing causal TOP5 depth at retrace")
    denom=float(rows["denom"][final])
    mlofi=float(sign*np.sum(rows["mlofi"][lo:hi],dtype=np.float64)/denom) if denom>0 else None
    defend="bid5" if sign>0 else "ask5"
    first_depth=float(rows[defend][initial]); last_depth=float(rows[defend][final])
    if first_depth<=0 or last_depth<=0:raise PublicIVBError("nonpositive defending TOP5 depth")
    defense=float((last_depth-first_depth)/max(first_depth,1))
    support={"aggressive_delta_ratio":ratio is not None and ratio>0,
        "opposing_flow_price_efficiency":opposing>0 and adverse<=0,
        "directional_top5_mlofi":mlofi is not None and mlofi>0,
        "passive_defense_ratio":defense>=0}
    count=sum(support.values())
    group="LOW" if count<=1 else "MID" if count==2 else "HIGH"
    return {"status":"COMPLETE","confirmation_timestamp_ns":end,
        "supportive_aggressive_volume":supportive,"opposing_aggressive_volume":opposing,
        "total_aggressive_volume":total,"directional_aggressive_delta":delta,
        "aggressive_delta_ratio":ratio,"adverse_price_progress_ticks":adverse,
        "opposing_flow_price_efficiency":efficiency,"directional_top5_mlofi":mlofi,
        "passive_defense_ratio":defense,"defending_depth_start":first_depth,
        "defending_depth_end":last_depth,"component_support":support,
        "support_count":count,"support_group":group,
        "book_last_ts_ns":int(book_ts[final]),"book_rows_in_window":hi-lo}


def _markouts(tape: np.ndarray, t: int, sign: int, close: int,
              *, executable: bool, floor: int=0,
              entry_reference_ns: int | None=None) -> dict[str,Any]:
    ts=tape["timestamp_ns"]
    anchor=_anchor(tape,t,close,floor=floor)
    if anchor is None:raise PublicIVBError("missing raw markout anchor")
    bid=tape["bid"].astype(float,copy=False); ask=tape["ask"].astype(float,copy=False)
    mid=(bid+ask)/2
    raw_mid=float(mid[anchor])
    entry_target=(entry_reference_ns if entry_reference_ns is not None else t)+2_000_000
    entry_ix=_anchor(tape,entry_target,close,floor=anchor) if executable else None
    quote=float(ask[entry_ix] if sign>0 else bid[entry_ix]) if entry_ix is not None else None
    fill=quote+sign*parent.TICK if quote is not None else None
    paths={}; excursions={}; touches={}
    for h in HORIZONS:
        raw_ix=parent._future(ts,t+h*1_000_000,close)
        end_ix=parent._future(ts,int(ts[entry_ix])+h*1_000_000,close) if entry_ix is not None else None
        raw=float(sign*(mid[raw_ix]-raw_mid)/parent.TICK) if raw_ix is not None and math.isfinite(float(mid[raw_ix])) else None
        q=a=shift=pre=spread=None
        if executable and raw is not None and entry_ix is not None and end_ix is not None:
            exit_px=float(bid[end_ix] if sign>0 else ask[end_ix])
            q=float(sign*(exit_px-quote)/parent.TICK)
            a=float(sign*(exit_px-fill)/parent.TICK)
            shift=float(sign*(mid[end_ix]-mid[raw_ix])/parent.TICK)
            pre=float(sign*(mid[entry_ix]-raw_mid)/parent.TICK)
            spread=float(sign*((exit_px-mid[end_ix])+(mid[entry_ix]-quote))/parent.TICK)
            if abs(a-(raw+shift-pre+spread-1))>1e-8:raise PublicIVBError("signal-to-fill decomposition mismatch")
        paths[str(h)]={"raw":raw,"quote":q,"actual":a,"horizon_shift":shift,
            "pre_entry_price_move":pre,"bid_ask_effect":spread}
    for h in EXCURSIONS:
        end_ix=parent._future(ts,t+h*1_000_000,close)
        if end_ix is None:excursions[str(h)]=None;continue
        signed=sign*(mid[anchor:end_ix+1]-raw_mid)/parent.TICK
        signed=signed[np.isfinite(signed)]
        excursions[str(h)]={"mfe":float(max(0,np.max(signed))),
            "mae":float(min(0,np.min(signed)))} if len(signed) else None
    horizon=60_000
    end_ix=parent._future(ts,t+horizon*1_000_000,close)
    for favorable,adverse in BARRIERS:
        key=f"{favorable}:-{adverse}@{horizon}ms"
        if end_ix is None:touches[key]={"result":"UNAVAILABLE","touch_ms":None};continue
        signed=sign*(mid[anchor:end_ix+1]-raw_mid)/parent.TICK
        fi=np.flatnonzero(signed>=favorable-1e-9)
        ai=np.flatnonzero(signed<=-adverse+1e-9)
        f=int(fi[0]) if len(fi) else None; a=int(ai[0]) if len(ai) else None
        first=f if a is None or f is not None and f<=a else a
        touches[key]={"result":"NEITHER" if first is None else
            "FAVORABLE_FIRST" if first==f else "ADVERSE_FIRST",
            "touch_ms":float((ts[anchor+first]-t)/1e6) if first is not None else None}
    return {"raw_anchor_time_ns":int(ts[anchor]),"raw_anchor_mid":raw_mid,
        "entry_time_ns":int(ts[entry_ix]) if entry_ix is not None else None,
        "entry_quote":quote,"actual_fill":fill,"paths":paths,
        "excursions":excursions,"first_touch":touches}


def _structural_outcomes(tape: np.ndarray, frame: Mapping[str,Any],
                         retrace: Mapping[str,Any], profile: Mapping[str,Any],
                         confirmed: Mapping[str,Any]) -> dict[str,Any]:
    ts=tape["timestamp_ns"];t=int(retrace["timestamp_ns"]);sign=int(frame["sign"])
    close=int(profile["rth_close_ns"])
    end=int(np.searchsorted(ts,close,side="left"))
    ids=np.flatnonzero((tape["execution_size"][int(retrace["tape_index"]):end]>0)&
        np.isfinite(tape["execution_price"][int(retrace["tape_index"]):end]))+int(retrace["tape_index"])
    p=tape["execution_price"][ids]
    inv=float(frame["invalidation_reference"])
    invalid=p<inv if sign>0 else p>inv
    invpos=np.flatnonzero(invalid)
    first_inv=int(invpos[0]) if len(invpos) else None
    before=p[:first_inv] if first_inv is not None else p
    extreme=sign*(p-float(frame["breakout_price"]))
    reach_prior=bool(np.any(extreme>=-1e-8))
    new_extreme=bool(np.any(extreme>1e-8))
    poc=float(profile["poc"])
    through_poc=bool(np.any(p<poc)) if sign>0 else bool(np.any(p>poc))
    boundary=float(profile["or_high"] if sign>0 else profile["or_low"])
    reach_boundary=bool(np.any(p>=boundary)) if sign>0 else bool(np.any(p<=boundary))
    opposite=float(profile["or_low"] if sign>0 else profile["or_high"])
    through_or=bool(np.any(p<opposite)) if sign>0 else bool(np.any(p>opposite))
    confirm_t=int(confirmed["confirmation_timestamp_ns"])
    raw_mid=float(confirmed["raw_anchor_mid"])
    # Post-confirmation trade prices, not retrace-to-confirmation observations.
    post_ids=ids[ts[ids]>confirm_t]
    post_p=tape["execution_price"][post_ids]
    post_signed=sign*(post_p-raw_mid)/parent.TICK
    post_invalid=post_p<inv if sign>0 else post_p>inv
    invp=np.flatnonzero(post_invalid);inv_at=int(invp[0]) if len(invp) else None
    success=np.flatnonzero(post_signed>=8-1e-9)
    succ_at=int(success[0]) if len(success) else None
    plus4=np.flatnonzero(post_signed>=4-1e-9)
    minus2=np.flatnonzero(post_signed<=-2+1e-9)
    before_invalid=post_signed[:inv_at] if inv_at is not None else post_signed
    if succ_at is not None and (inv_at is None or succ_at<inv_at):label="SUCCESSFUL_RELOAD"
    elif inv_at is not None:label="STRUCTURAL_INVALIDATION"
    elif len(plus4) and len(minus2) and plus4[0]<minus2[0]:label="PARTIAL_RELOAD_THEN_FAILURE"
    elif len(minus2) and (ts[post_ids[minus2[0]]]-confirm_t)<=2_000_000_000:label="IMMEDIATE_FAILURE"
    elif not len(plus4) and not len(minus2) and len(post_ids) and ts[post_ids[-1]]>=confirm_t+60_000_000_000:
        label="STAGNATION"
    else:label="UNRESOLVED"
    return {"prior_breakout_extreme_reached":reach_prior,"new_favorable_extreme":new_extreme,
        "traded_through_poc":through_poc,"invalidation_reference_reached":first_inv is not None,
        "time_to_invalidation_ms":float((ts[ids[first_inv]]-t)/1e6) if first_inv is not None else None,
        "max_favorable_excursion_before_invalidation_ticks":float(max(0,np.max(sign*(before-float(retrace["price"]))/parent.TICK))) if len(before) else 0.0,
        "or_breakout_boundary_reached":reach_boundary,"opposite_or_edge_broken":through_or,
        "reload_classification":label,
        "post_confirmation_plus8_before_invalidation":label=="SUCCESSFUL_RELOAD",
        "post_confirmation_max_favorable_before_invalidation_ticks":float(max(0,np.max(before_invalid))) if len(before_invalid) else 0.0}


def evaluate_date(day: str, tape: np.ndarray, rows: np.ndarray,
                  frozen: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    if day not in flow.TARGET_DATES:raise PublicIVBError("ineligible date")
    profile=_profile(day,tape)
    frame=_frame(day,frozen,profile)
    if frame["status"]!="FRAME":return {"date":day,"profile":profile,"frame":frame,"retrace":{"status":"NO_FRAME"},"event":None}
    retrace=_retrace(tape,frame,profile)
    if retrace["status"]!="RETRACE":return {"date":day,"profile":profile,"frame":frame,"retrace":retrace,"event":None}
    t=int(retrace["timestamp_ns"]);sign=int(frame["sign"]);close=int(profile["rth_close_ns"])
    retrace_path=_markouts(tape,t,sign,close,executable=False,floor=int(retrace["tape_index"]))
    orderflow=_orderflow(tape,rows,retrace,frame,profile)
    if orderflow["status"]!="COMPLETE":
        return {"date":day,"profile":profile,"frame":frame,"retrace":retrace,
                "event":{"status":"UNCONFIRMED_END_OF_RTH","retrace_anchor":retrace_path}}
    confirmed_path=_markouts(tape,int(orderflow["confirmation_timestamp_ns"])+1,sign,close,
        executable=True,floor=int(retrace["tape_index"]),
        entry_reference_ns=int(orderflow["confirmation_timestamp_ns"]))
    price_controls=parent._price_controls(tape,{"timestamp_ns":t,"sign":sign},int(profile["or_start_ns"]))
    structural=_structural_outcomes(tape,frame,retrace,profile,
        {"confirmation_timestamp_ns":orderflow["confirmation_timestamp_ns"],
         "raw_anchor_mid":confirmed_path["raw_anchor_mid"]})
    event={"status":"COMPLETE","date":day,"period":parent._period(day),
        "direction":frame["direction"],"sign":sign,"profile":profile,"frame":frame,
        "retrace":retrace,"orderflow":orderflow,"price_controls":price_controls,
        "retrace_anchor":retrace_path,"confirmed_anchor":confirmed_path,
        "structural_outcomes":structural}
    return {"date":day,"profile":profile,"frame":frame,"retrace":retrace,"event":event}


def _path_table(events: Sequence[Mapping[str,Any]],anchor: str,path: str) -> dict[str,Any]:
    return {label:{str(h):_stats([e[anchor]["paths"][str(h)][path] for e in _split(events,label)])
        for h in HORIZONS} for label in PERIODS}


def _excursion_table(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return {label:{anchor:{str(h):{kind:_stats([e[anchor]["excursions"][str(h)][kind]
        for e in _split(events,label) if e[anchor]["excursions"][str(h)] is not None])
        for kind in ("mfe","mae")} for h in EXCURSIONS}
        for anchor in ("retrace_anchor","confirmed_anchor")} for label in PERIODS}


def _touch_table(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    keys=[f"{a}:-{b}@60000ms" for a,b in BARRIERS]
    return {label:{anchor:{key:{"n":len(group),"favorable_first":sum(x=="FAVORABLE_FIRST" for x in group),
        "adverse_first":sum(x=="ADVERSE_FIRST" for x in group),"neither":sum(x=="NEITHER" for x in group),
        "unavailable":sum(x=="UNAVAILABLE" for x in group),
        "touch_ms":_stats([e[anchor]["first_touch"][key]["touch_ms"] for e in _split(events,label)])}
        for key in keys for group in ([e[anchor]["first_touch"][key]["result"] for e in _split(events,label)],)}
        for anchor in ("retrace_anchor","confirmed_anchor")} for label in PERIODS}


def _component_values(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return {label:{c:_stats([e["orderflow"][c] for e in _split(events,label)]) for c in COMPONENTS}
        for label in PERIODS}


def _cuts(spring: Sequence[Mapping[str,Any]]) -> dict[str,list[float] | None]:
    out={}
    for c in COMPONENTS:
        values=[e["orderflow"][c] for e in spring if e["orderflow"][c] is not None]
        out[c]=[float(x) for x in np.quantile(values,[1/3,2/3])] if len(values)>=3 else None
    return out


def _assign_terciles(events: Sequence[dict[str,Any]],cuts: Mapping[str,Sequence[float] | None]) -> None:
    for e in events:
        e["component_terciles"]={}
        for c in COMPONENTS:
            value=e["orderflow"][c]; q=cuts[c]
            e["component_terciles"][c]=(None if value is None or q is None else
                "LOW" if value<=q[0] else "MID" if value<=q[1] else "HIGH")


def _group_metrics(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return {"n":len(rows),"status":"SUFFICIENT" if len(rows)>=10 else "INSUFFICIENT_BUCKET_SAMPLE",
        "retrace_raw_10s":_stats([e["retrace_anchor"]["paths"]["10000"]["raw"] for e in rows]),
        "confirmed_raw_10s":_stats([e["confirmed_anchor"]["paths"]["10000"]["raw"] for e in rows]),
        "quote_10s":_stats([e["confirmed_anchor"]["paths"]["10000"]["quote"] for e in rows]),
        "actual_10s":_stats([e["confirmed_anchor"]["paths"]["10000"]["actual"] for e in rows]),
        "mfe_10s":_stats([e["confirmed_anchor"]["excursions"]["10000"]["mfe"] for e in rows]),
        "mae_10s":_stats([e["confirmed_anchor"]["excursions"]["10000"]["mae"] for e in rows]),
        "plus8_before_invalidation_rate":(sum(e["structural_outcomes"]["post_confirmation_plus8_before_invalidation"] for e in rows)/len(rows)) if rows else None,
        "reload_classification_counts":{name:sum(e["structural_outcomes"]["reload_classification"]==name for e in rows)
            for name in ("SUCCESSFUL_RELOAD","PARTIAL_RELOAD_THEN_FAILURE","IMMEDIATE_FAILURE",
                         "STAGNATION","STRUCTURAL_INVALIDATION","UNRESOLVED")}}


def _component_terciles(events: Sequence[Mapping[str,Any]],cuts: Mapping[str,Any]) -> dict[str,Any]:
    return {c:{label:{bucket:_group_metrics([e for e in _split(events,label)
        if e["component_terciles"][c]==bucket]) for bucket in ("LOW","MID","HIGH")}
        for label in ("SPRING_2025","OCTOBER_2025")}
        for c in COMPONENTS}


def _support_table(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return {label:{bucket:_group_metrics([e for e in _split(events,label)
        if e["orderflow"]["support_group"]==bucket]) for bucket in ("LOW","MID","HIGH")}
        for label in PERIODS}


def _relationship(cells: Mapping[str,Any], *, invert: bool=False) -> str:
    if any(cells[k]["n"]<10 for k in ("LOW","MID","HIGH")):return "INSUFFICIENT_SAMPLE"
    a,b,c=[cells[k]["quote_10s"]["mean"] for k in ("LOW","MID","HIGH")]
    if None in (a,b,c):return "INSUFFICIENT_SAMPLE"
    if invert:a,b,c=c,b,a
    if a<b<c:return "MONOTONIC_IMPROVEMENT"
    if a>b>c:return "OPPOSITE_RELATIONSHIP"
    if b>a and b>c:return "INVERTED_U"
    if abs(c-b)<.25*max(abs(c-a),1e-12) and c>a:return "SATURATION"
    return "NO_CLEAR_SHAPE"


def _permutation(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    rng=np.random.default_rng(20251006)
    y=np.asarray([e["confirmed_anchor"]["paths"]["10000"]["quote"] for e in events],dtype=float)
    strata=np.asarray([e["period"]+"|"+e["direction"] for e in events])
    result={}
    for c in (*COMPONENTS,"support_count"):
        values=np.asarray([e["orderflow"][c] for e in events],dtype=float)
        valid=np.isfinite(values)&np.isfinite(y)
        if valid.sum()<20 or len(np.unique(values[valid]))<2:
            result[c]={"status":"INSUFFICIENT_SAMPLE","n":int(valid.sum())};continue
        observed=prior.spearman(values[valid],y[valid])["rho"]
        if observed is None:
            result[c]={"status":"INSUFFICIENT_SAMPLE","n":int(valid.sum())};continue
        null=[]
        for _ in range(1000):
            shuffled=y.copy()
            for s in np.unique(strata):
                ix=np.flatnonzero((strata==s)&valid)
                shuffled[ix]=rng.permutation(shuffled[ix])
            rho=prior.spearman(values[valid],shuffled[valid])["rho"]
            if rho is not None:null.append(rho)
        n=np.asarray(null,dtype=float)
        result[c]={"status":"DESCRIPTIVE_PERMUTATION","n":int(valid.sum()),
            "observed_spearman":float(observed),"null_percentile":float(np.mean(n<=observed)),
            "two_sided_p":float((1+np.sum(np.abs(n)>=abs(observed)))/(len(n)+1)),
            "permutations":len(n),"seed":20251006,"strata":"period x direction"}
    return result


def _price_only_control(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    def vector(e: Mapping[str,Any]) -> list[float] | None:
        r=e["retrace"];f=e["price_controls"]
        vals=[r["max_favorable_expansion_before_retrace_ticks"],
            r["breakout_to_retrace_seconds"],r["zone_depth_fraction"],
            f["price_velocity_10s"],f["realized_volatility_10s"],r["distance_to_poc_ticks"]]
        if any(v is None or not math.isfinite(v) for v in vals):return None
        return [float(a)/scale for a,scale in zip(vals,(8,120,.33,2,5,4))]
    matched=[]
    for e in events:
        if e["orderflow"]["support_group"]!="HIGH" or vector(e) is None:continue
        pool=[x for x in events if x["orderflow"]["support_group"]=="LOW" and
              x["period"]==e["period"] and x["direction"]==e["direction"] and vector(x) is not None]
        if not pool:continue
        v=np.asarray(vector(e)); control=min(pool,key=lambda x:(float(np.sum((v-np.asarray(vector(x)))**2)),x["date"]))
        matched.append((e,control))
    positive=[a["confirmed_anchor"]["paths"]["10000"]["quote"] for a,_ in matched]
    controls=[b["confirmed_anchor"]["paths"]["10000"]["quote"] for _,b in matched]
    delta=[a-b for a,b in zip(positive,controls)]
    return {"status":"DESCRIPTIVE_ONLY" if len(matched)>=8 else "INSUFFICIENT_PRICE_ONLY_CONTROL_SAMPLE",
        "matched_pairs":len(matched),"high_support_quote_10s":_stats(positive),
        "matched_low_support_quote_10s":_stats(controls),"incremental_ticks":_stats(delta),
        "match_fields":["period","direction","pre_retrace_expansion","breakout_to_retrace_seconds",
            "zone_depth","pre_retrace_price_velocity","pre_retrace_rv","distance_to_poc"],
        "matching_uses_outcomes":False,"matching_uses_l2":False}


def _week(day: str) -> str:
    y,w,_=date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _daily_weekly(payloads: Sequence[Mapping[str,Any]]) -> tuple[dict[str,Any],dict[str,Any]]:
    daily={}
    for p in payloads:
        e=p["event"];r=p["retrace"];f=p["frame"]
        daily[p["date"]]={"period":parent._period(p["date"]),"frame_direction":f.get("direction"),
            "breakout_timestamp_ns":f.get("breakout_timestamp_ns"),"retrace_status":r["status"],
            "retrace_timestamp_ns":r.get("timestamp_ns"),
            "profile_levels":{k:p["profile"][k] for k in ("or_high","or_low","poc","vah","val")},
            "orderflow_support":e["orderflow"]["component_support"] if e and e["status"]=="COMPLETE" else None,
            "retrace_raw_10s":e["retrace_anchor"]["paths"]["10000"]["raw"] if e and e["status"]=="COMPLETE" else None,
            "confirmed_quote_10s":e["confirmed_anchor"]["paths"]["10000"]["quote"] if e and e["status"]=="COMPLETE" else None,
            "reload_classification":e["structural_outcomes"]["reload_classification"] if e and e["status"]=="COMPLETE" else None}
    weekly={}
    for week in sorted({_week(d) for d in flow.TARGET_DATES}):
        rows=[v for d,v in daily.items() if _week(d)==week]
        weekly[week]={"sessions":len(rows),"frames":sum(r["frame_direction"] is not None for r in rows),
            "retraces":sum(r["retrace_status"]=="RETRACE" for r in rows),
            "retrace_raw_10s":_stats([r["retrace_raw_10s"] for r in rows]),
            "confirmed_quote_10s":_stats([r["confirmed_quote_10s"] for r in rows])}
    return daily,weekly


def _effect(events: Sequence[Mapping[str,Any]]) -> float | None:
    high=[e["confirmed_anchor"]["paths"]["10000"]["quote"] for e in events
          if e["orderflow"]["support_group"]=="HIGH"]
    low=[e["confirmed_anchor"]["paths"]["10000"]["quote"] for e in events
         if e["orderflow"]["support_group"]=="LOW"]
    return float(np.mean(high)-np.mean(low)) if len(high)>=8 and len(low)>=8 else None


def _leave_out(events: Sequence[Mapping[str,Any]], by: str) -> dict[str,Any]:
    keys=list(flow.TARGET_DATES) if by=="date" else sorted({_week(d) for d in flow.TARGET_DATES})
    rows={}
    for key in keys:
        remaining=[e for e in events if (e["date"] if by=="date" else _week(e["date"]))!=key]
        rows[key]={"n":len(remaining),"support_high_minus_low_quote_10s":_effect(remaining),
            "all_confirmed_quote_10s":_stats([e["confirmed_anchor"]["paths"]["10000"]["quote"]
                for e in remaining])["mean"]}
    valid={k:v["support_high_minus_low_quote_10s"] for k,v in rows.items()
           if v["support_high_minus_low_quote_10s"] is not None}
    return {"rows":rows,"adequate_omissions":len(valid),
        "sign_stability":{"positive":sum(x>0 for x in valid.values()),
            "negative":sum(x<0 for x in valid.values()),"median":_stats(list(valid.values()))["median"],
            "minimum":min(valid.values()) if valid else None,"maximum":max(valid.values()) if valid else None,
            "worst_omitted":min(valid,key=valid.get) if valid else None,
            "best_omitted":max(valid,key=valid.get) if valid else None}}


def _execution_hurdle(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    result={}
    for label in PERIODS:
        result[label]={}
        for h in HORIZONS:
            paths=[e["confirmed_anchor"]["paths"][str(h)] for e in _split(events,label)]
            valid=[p for p in paths if p["actual"] is not None]
            result[label][str(h)]={"n":len(valid),"raw":_stats([p["raw"] for p in valid]),
                "quote":_stats([p["quote"] for p in valid]),"actual":_stats([p["actual"] for p in valid]),
                "pre_entry_price_effect":_stats([-p["pre_entry_price_move"] for p in valid]),
                "bid_ask_effect":_stats([p["bid_ask_effect"] for p in valid]),
                "horizon_alignment_effect":_stats([p["horizon_shift"] for p in valid]),
                "adverse_entry_tick_effect":-1 if valid else None}
    return result


def _decision(events: Sequence[Mapping[str,Any]], support: Mapping[str,Any],
              control: Mapping[str,Any], component_table: Mapping[str,Any],
              lodo: Mapping[str,Any], lowo: Mapping[str,Any]) -> dict[str,Any]:
    spring=_split(events,"SPRING_2025"); october=_split(events,"OCTOBER_2025")
    if len(spring)<10 or len(october)<10:
        return {"primary_decision":"INSUFFICIENT_SAMPLE","next_step":"REQUIRE_ADDITIONAL_PREDECLARED_DATA",
            "candidate_hypothesis":None,"orderflow_adds_incremental_information":"insufficient",
            "execution_hurdle_cleared":False}
    means={period:{path:_stats([e["confirmed_anchor"]["paths"]["10000"][path] for e in rows])["mean"]
        for path in ("raw","quote","actual")}
        for period,rows in (("SPRING_2025",spring),("OCTOBER_2025",october))}
    baseline_positive=all(means[p]["actual"] is not None and means[p]["actual"]>0
                          for p in ("SPRING_2025","OCTOBER_2025"))
    support_adequate=all(support[p][k]["n"]>=10 for p in ("SPRING_2025","OCTOBER_2025")
                         for k in ("LOW","HIGH"))
    support_effects={p:(support[p]["HIGH"]["quote_10s"]["mean"]-
        support[p]["LOW"]["quote_10s"]["mean"]) if support[p]["HIGH"]["n"] and support[p]["LOW"]["n"] else None
        for p in ("SPRING_2025","OCTOBER_2025")}
    price_increment=control["incremental_ticks"]["mean"]
    incremental=(support_adequate and all(x is not None and x>0 for x in support_effects.values())
        and control["matched_pairs"]>=8 and price_increment is not None and price_increment>0
        and lodo["adequate_omissions"]>0 and lowo["adequate_omissions"]>0
        and lodo["sign_stability"]["negative"]==0 and lowo["sign_stability"]["negative"]==0)
    if baseline_positive and incremental:
        # No component is promoted automatically merely because a support sum
        # works. A component-level causal mechanism must independently survive.
        coherent=[c for c in COMPONENTS if all(_relationship(component_table[c][p],
            invert=(c=="opposing_flow_price_efficiency")) in ("MONOTONIC_IMPROVEMENT","SATURATION")
            for p in ("SPRING_2025","OCTOBER_2025"))]
        if len(coherent)==1:
            decision="PUBLIC_IVB_REPLICATION_PROMISING"
            candidate={"component":coherent[0],"status":"HYPOTHESIS_ONLY_NOT_A_GATE"}
            next_step="BUILD_ONE_FROZEN_PUBLIC_IVB_STRATEGY_V1"
        else:
            decision="PROFILE_RETRACEMENT_PROMISING_ORDERFLOW_UNCONFIRMED"
            candidate=None;next_step="TEST_ONE_FROZEN_PROFILE_RETRACE_STRATEGY_WITHOUT_ORDERFLOW_OPTIMIZATION"
    elif baseline_positive:
        decision="PROFILE_RETRACEMENT_PROMISING_ORDERFLOW_UNCONFIRMED"
        candidate=None;next_step="TEST_ONE_FROZEN_PROFILE_RETRACE_STRATEGY_WITHOUT_ORDERFLOW_OPTIMIZATION"
    elif support_effects["SPRING_2025"] is not None and support_effects["SPRING_2025"]>2 and not support_adequate:
        decision="ORDERFLOW_POSSIBLY_USEFUL_BUT_UNCONFIRMED"
        candidate=None;next_step="FREEZE_AT_MOST_ONE_ORDERFLOW_HYPOTHESIS_FOR_FRESH_CALIBRATION"
    else:
        decision="PUBLIC_IVB_REPLICATION_FAILED"
        candidate=None;next_step="STOP_PUBLIC_IVB_REPLICATION_BRANCH"
    return {"primary_decision":decision,"next_step":next_step,"candidate_hypothesis":candidate,
        "orderflow_adds_incremental_information":True if incremental else "insufficient" if not support_adequate else False,
        "execution_hurdle_cleared":baseline_positive,"period_confirmed_10s_means":means,
        "support_high_minus_low_quote_10s":support_effects,"price_only_control_status":control["status"]}


def _checkpoint(root: Path, day: str) -> Path:
    return root/"checkpoints"/f"{day}.json.gz"


def _read_checkpoint(path: Path, day: str, source_sha: str, tape_sha: str,
                     parent_sha: str, entry_sha: str) -> dict[str,Any] | None:
    if not path.is_file():return None
    try:
        with gzip.open(path,"rt",encoding="utf-8") as f:row=json.load(f)
    except (OSError,EOFError,json.JSONDecodeError):return None
    expected={"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,"date":day,
        "source_sha256":source_sha,"tape_sha256":tape_sha,
        "parent_manifest_sha256":parent_sha,"entry_manifest_sha256":entry_sha,
        "config_sha256":CONFIG_SHA256}
    return row if all(row.get(k)==v for k,v in expected.items()) else None


def _entry_manifest(root: Path, parent_sha: str) -> str:
    path=root/"run-manifest.json";m=json.loads(path.read_text())
    if m.get("status")!="COMPLETE" or m.get("config_sha256")!=prior.CONFIG_SHA256 or m.get("parent_manifest_sha256")!=parent_sha:
        raise PublicIVBError("entry audit incomplete, changed, or parent-unbound")
    for name,digest in m["artifact_sha256_by_name"].items():
        if native._sha(root/name)!=digest:raise PublicIVBError(f"entry audit artifact hash mismatch: {name}")
    return native._sha(path)


def run(*,data_root: Path=native.DATA_ROOT,parent_root: Path=parent.OUT_ROOT,
        entry_root: Path=prior.OUT_ROOT,output_root: Path=OUT_ROOT,
        smoke: bool=False) -> dict[str,Any]:
    started=time.monotonic()
    paths,source_manifest=native._source_catalog(data_root)
    _,coverage,frozen,parent_sha=prior._parent_package(parent_root)
    entry_sha=_entry_manifest(entry_root,parent_sha)
    if {d:source_manifest[d]["sha256"] for d in native.ALL_SOURCE_DATES}!=coverage["source_sha256_by_date"]:
        raise PublicIVBError("native ES source catalog changed from parent")
    days=flow.TARGET_DATES[:1] if smoke else flow.TARGET_DATES
    output_root.mkdir(parents=True,exist_ok=True)
    native._write_json(output_root/"study-config.json",CONFIG)
    payloads=[]
    for pos,day in enumerate(days,1):
        source_sha=source_manifest[day]["sha256"]
        tape_path=native._tape_path(day);tape_sha=native._sha(tape_path)
        if tape_sha!=coverage["tape_sha256_by_date"][day]:raise PublicIVBError(f"canonical tape hash mismatch: {day}")
        cp=_checkpoint(output_root,day)
        cached=_read_checkpoint(cp,day,source_sha,tape_sha,parent_sha,entry_sha)
        if cached:
            payload=cached["payload"]
            print(f"PUBLIC_IVB_DATE_RESUME={day}",flush=True)
        else:
            print(f"PUBLIC_IVB_DATE_START={pos}/{len(days)} date={day}",flush=True)
            tape,_=native._load_tape(day,tape_path,source_sha)
            compact=flow._cached_compact(day,paths[day],source_sha,flow.OUT_ROOT)
            payload=evaluate_date(day,tape,compact,frozen.get(day,[]))
            native._write_checkpoint(cp,{"status":"DATE_COMPLETE","version":CHECKPOINT_VERSION,
                "date":day,"source_sha256":source_sha,"tape_sha256":tape_sha,
                "parent_manifest_sha256":parent_sha,"entry_manifest_sha256":entry_sha,
                "config_sha256":CONFIG_SHA256,"payload":payload})
            print(f"PUBLIC_IVB_DATE_COMPLETE={day} frame={payload['frame']['status']} retrace={payload['retrace']['status']}",flush=True)
        payloads.append(payload)
    if smoke:return {"status":"SMOKE_PASS","dates":list(days),
                     "retraces":sum(p["retrace"]["status"]=="RETRACE" for p in payloads)}
    if len(payloads)!=54:raise PublicIVBError("incomplete 54-date source set")
    frames=[p for p in payloads if p["frame"]["status"]=="FRAME"]
    retraces=[p for p in payloads if p["retrace"]["status"]=="RETRACE"]
    events=[p["event"] for p in retraces if p["event"] and p["event"]["status"]=="COMPLETE"]
    if any(p["event"] and p["event"]["status"]!="COMPLETE" for p in retraces):
        raise PublicIVBError("retrace lacks complete 500ms confirmation")
    if len({e["date"] for e in events})!=len(events):raise PublicIVBError("duplicate retrace event per session")
    spring=_split(events,"SPRING_2025")
    cuts=_cuts(spring)
    _assign_terciles(events,cuts)
    # Spring cutpoints/interpretation are sealed before October outcome tables.
    spring_component={c:{"SPRING_2025":{k:_group_metrics([e for e in spring if e["component_terciles"][c]==k])
        for k in ("LOW","MID","HIGH")}} for c in COMPONENTS}
    spring_interpretation={c:_relationship(spring_component[c]["SPRING_2025"],
        invert=(c=="opposing_flow_price_efficiency")) for c in COMPONENTS}
    native._write_json(output_root/"spring-component-interpretation.json",
        {"cutpoints":cuts,"shape":spring_interpretation})
    component_table=_component_terciles(events,cuts)
    support=_support_table(events)
    price_control=_price_only_control(events)
    perm=_permutation(events)
    retrace_raw=_path_table(events,"retrace_anchor","raw")
    confirmed_raw=_path_table(events,"confirmed_anchor","raw")
    quote=_path_table(events,"confirmed_anchor","quote")
    actual=_path_table(events,"confirmed_anchor","actual")
    excursions=_excursion_table(events)
    touches=_touch_table(events)
    hurdle=_execution_hurdle(events)
    daily,weekly=_daily_weekly(payloads)
    lodo=_leave_out(events,"date");lowo=_leave_out(events,"week")
    decision=_decision(events,support,price_control,component_table,lodo,lowo)
    location={k:_group_metrics([e for e in events if e["retrace"]["location_bucket"]==k])
        for k in ("OUTER_THIRD","MIDDLE_THIRD","POC_THIRD","POC_ONLY")}
    timing={k:_group_metrics([e for e in events if e["retrace"]["timing_bucket"]==k])
        for k in ("LE_30S","30_TO_120S","2_TO_5M","5_TO_15M","GT_15M")}
    period_results={p:{"n":len(rows),"retrace_anchor":_group_metrics(rows),
        "support_counts":{k:sum(e["orderflow"]["support_group"]==k for e in rows)
            for k in ("LOW","MID","HIGH")},
        "successful_reload_rate":_group_metrics(rows)["plus8_before_invalidation_rate"]}
        for p,rows in (("SPRING_2025",_split(events,"SPRING_2025")),
                       ("OCTOBER_2025",_split(events,"OCTOBER_2025")))}
    direction_results={d:{"n":len(_split(events,d)),"metrics":_group_metrics(_split(events,d))}
        for d in ("LONG","SHORT")}
    if min(direction_results[d]["n"] for d in ("LONG","SHORT"))<10:direction_class="INSUFFICIENT"
    else:
        a=direction_results["LONG"]["metrics"]["quote_10s"]["mean"]
        b=direction_results["SHORT"]["metrics"]["quote_10s"]["mean"]
        direction_class="OPPOSITE" if a*b<0 else "SYMMETRIC" if abs(a-b)<1 else "LONG_DOMINANT" if a>b else "SHORT_DOMINANT"
    direction_results["classification"]=direction_class
    source={"status":"PASS","dataset":"GLBX.MDP3","schema":"mbp-10","instrument":"ES",
        "native_es_mbp10_only":True,"no_mes":True,"no_mbo":True,
        "spring_dates":list(flow.SPRING_DATES),"october_dates":list(flow.OCTOBER_DATES),
        "parent_manifest_sha256":parent_sha,"entry_manifest_sha256":entry_sha,
        "source_sha256_by_date":coverage["source_sha256_by_date"],
        "tape_sha256_by_date":coverage["tape_sha256_by_date"]}
    summary={"run_id":RUN_ID,"status":"COMPLETE","total_sessions":54,
        "first_breakout_frames":len(frames),"long_frames":sum(p["frame"]["direction"]=="LONG" for p in frames),
        "short_frames":sum(p["frame"]["direction"]=="SHORT" for p in frames),
        "profile_retracements":len(retraces),"confirmed_retracements":len(events),
        "no_retrace_sessions":len(frames)-len(retraces),
        "retrace_rate":len(retraces)/len(frames) if frames else None,
        "breakout_to_retrace_seconds":_stats([e["retrace"]["breakout_to_retrace_seconds"] for e in events]),
        "retrace_anchor_markouts_all":retrace_raw["ALL"],
        "confirmed_raw_markouts_all":confirmed_raw["ALL"],
        "executable_markouts_all":quote["ALL"],"actual_fill_markouts_all":actual["ALL"],
        "period_results":period_results,"direction_results":direction_results,
        "spring_component_interpretation":spring_interpretation,
        "component_relationships":{c:{p:_relationship(component_table[c][p],
            invert=(c=="opposing_flow_price_efficiency")) for p in ("SPRING_2025","OCTOBER_2025")}
            for c in COMPONENTS},
        "support_count_relationship":{p:_relationship(support[p]) for p in ("SPRING_2025","OCTOBER_2025")},
        "decision":decision,"config_sha256":CONFIG_SHA256,
        "public_model_replication":True,"exact_proprietary_fabervaale_model":False,
        "proprietary_deepcharts_targets_reproduced":False,"native_es_mbp10_only":True,
        "spring_role":"PRIMARY_DEV_DISCOVERY","october_role":"SECONDARY_DEV_COMPATIBILITY",
        "optimization_performed":False,"optuna_performed":False,"threshold_search_performed":False,
        "stop_target_search_performed":False,"opening_range_search_performed":False,
        "profile_zone_search_performed":False,"passive_fill_assumption_used":False,
        "final_oos_accessed":False,"data_downloaded":False,"commit_performed":False,
        "elapsed_seconds":time.monotonic()-started}
    native._write_gzip_jsonl(output_root/"events.jsonl.gz",events)
    files={"summary.json":summary,"source-coverage.json":source,
        "opening-range-profiles.json":{p["date"]:p["profile"] for p in payloads},
        "breakout-frames.json":{p["date"]:p["frame"] for p in payloads},
        "profile-retracements.json":{p["date"]:p["retrace"] for p in payloads},
        "no-retrace-sessions.json":[p["date"] for p in payloads if p["frame"]["status"]=="FRAME" and p["retrace"]["status"]=="NO_RETRACE"],
        "breakout-to-retrace-path.json":{e["date"]:{k:e["retrace"][k] for k in
            ("breakout_to_retrace_seconds","max_favorable_expansion_before_retrace_ticks",
             "max_adverse_move_before_retrace_ticks","distance_breakout_to_retrace_ticks")}
            for e in events},
        "profile-location-analysis.json":location,"retrace-timing-analysis.json":timing,
        "orderflow-components.json":_component_values(events),
        "component-terciles.json":{"spring_cutpoints":cuts,"cells":component_table,
            "spring_shape":spring_interpretation},
        "orderflow-support-count.json":support,"price-only-control.json":price_control,
        "retrace-anchor-markouts.json":retrace_raw,"confirmed-markouts.json":confirmed_raw,
        "executable-markouts.json":quote,"actual-fill-markouts.json":actual,
        "execution-hurdle.json":hurdle,"mfe-mae.json":excursions,"first-touch.json":touches,
        "reload-classification.json":{p:{name:sum(e["structural_outcomes"]["reload_classification"]==name for e in _split(events,p))
            for name in ("SUCCESSFUL_RELOAD","PARTIAL_RELOAD_THEN_FAILURE","IMMEDIATE_FAILURE",
                         "STAGNATION","STRUCTURAL_INVALIDATION","UNRESOLVED")} for p in PERIODS},
        "period-results.json":period_results,"direction-results.json":direction_results,
        "daily-results.json":daily,"weekly-results.json":weekly,
        "lodo-results.json":lodo,"lowo-results.json":lowo,"permutation-results.json":perm}
    for name,value in files.items():native._write_json(output_root/name,value)
    report=[f"# {RUN_ID}","","Publicly reconstructable OR + executed-volume profile + first retracement convention only.",
        "This is not the proprietary Fabervaale/DeepCharts IVB system; no protection levels or target model were inferred.",
        f"Sessions: 54; first directional frames: {len(frames)}; first fixed-zone retracements: {len(retraces)}; no retrace: {len(frames)-len(retraces)}.",
        "Spring and October are both examined DEV blocks; neither is untouched OOS.",
        f"Primary decision: {decision['primary_decision']}.",
        f"Next step: {decision['next_step']}.",""]
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
    parser.add_argument("--entry-root",type=Path,default=prior.OUT_ROOT)
    parser.add_argument("--output-root",type=Path,default=OUT_ROOT)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv)
    try:result=run(data_root=args.data_root,parent_root=args.parent_root,
        entry_root=args.entry_root,output_root=args.output_root,smoke=args.smoke)
    except (PublicIVBError,prior.EntryAuditError,parent.BreakoutStudyError,
            native.VacuumStudyError,OSError,ValueError,KeyError,AssertionError) as exc:
        parser.exit(2,f"PUBLIC_IVB_ERROR: {exc}\n")
    print(f"FABERVAALE_PUBLIC_IVB_REPLICATION_EVENT_STUDY={result['status']}",flush=True)
    if result["status"]=="COMPLETE":print(f"PRIMARY_DECISION={result['decision']['primary_decision']}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
