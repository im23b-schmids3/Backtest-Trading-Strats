"""Read-only diagnostics for the sealed 2025 Fair Pricing V2 event study."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_jj_fair_pricing_expanded_v2 as v2
from . import mac_2025_jj_fair_pricing_reversion_bos_v1 as v1
from .mac_2025_candidate_tape import load_tape

V2 = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_EXPANDED_V2")
OUT = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_V2_MECHANISM_DIAGNOSTIC")
TICK = .25
HORIZONS = (10, 30, 60, 120, 300, 600, 900)
FLOW_FIELDS = ("local_delta_30s_supportive_fraction", "local_delta_2m_supportive_fraction",
               "session_cvd_supportive_fraction", "opposing_aggression_fraction_2m",
               "delta_30s_normalized", "delta_2m_normalized", "session_cvd_normalized",
               "price_progress_toward_ticks_2m", "effort_without_result",
               "price_impact_ticks_per_100_aggressive_contracts")


class DiagnosticError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    a = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)), "p10": float(np.quantile(a, .1)),
            "p90": float(np.quantile(a, .9)), "min": float(np.min(a)), "max": float(np.max(a))}


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _bootstrap(values_by_date: Mapping[str, Sequence[float]], seed: int = 31817, reps: int = 1500) -> dict[str, Any]:
    dates = sorted(values_by_date)
    all_values = [float(x) for d in dates for x in values_by_date[d]]
    if len(dates) < 2 or not all_values:
        return {"date_clusters": len(dates), "mean": _mean(all_values), "ci95": None}
    rng = random.Random(seed); vals = []
    for _ in range(reps):
        sample = [x for d in (rng.choice(dates) for _ in dates) for x in values_by_date[d]]
        if sample: vals.append(float(statistics.mean(sample)))
    return {"date_clusters": len(dates), "mean": float(statistics.mean(all_values)),
            "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
            "method": "date-cluster percentile bootstrap; signals within a session-date remain clustered"}


def _verify_source() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest, hashes, source = (_json(V2 / n) for n in ("run-manifest.json", "artifact-hashes.json", "source-coverage.json"))
    if manifest.get("status") != "COMPLETE" or hashes.get("status") != "HASHED":
        raise DiagnosticError("frozen V2 run is not complete/hashed")
    bad_manifest = [n for n, digest in manifest["files"].items() if _sha(V2 / n) != digest]
    bad_artifacts = [n for n, digest in hashes["files"].items() if _sha(V2 / n) != digest]
    # The run-manifest field names the upstream V1 coverage input, while the
    # source-coverage.json bytes are independently pinned in manifest["files"].
    config = _json(V2 / "study-config.json")
    upstream_ref = source.get("source_coverage_manifest", {})
    upstream_path = Path(upstream_ref["path"]) if upstream_ref.get("path") else None
    upstream_ok = bool(upstream_path and upstream_path.is_file()
                       and _sha(upstream_path) == upstream_ref.get("sha256")
                       and upstream_ref.get("sha256") == manifest.get("source_coverage_sha256")
                       and config.get("source_coverage_sha256") == manifest.get("source_coverage_sha256"))
    signals, features, trades = (_jsonl(V2 / n) for n in ("all-signals.jsonl.gz", "causal-features.jsonl.gz", "executed-trades.jsonl.gz"))
    smap = {str(x["signal_id"]): x for x in signals}; fmap = {str(x["signal_id"]): x for x in features}
    ids = {str(x["signal_id"]) for x in trades}
    errors = []
    if len(smap) != len(signals): errors.append("duplicate signal_id")
    if set(smap) != set(fmap): errors.append("signal/features ID mismatch")
    if not ids.issubset(smap): errors.append("trade references unknown signal_id")
    for sid, s in smap.items():
        f = fmap.get(sid)
        if f and any(f.get(k) != s.get(k) for k in ("date", "session", "model", "trigger")):
            errors.append(f"signal/feature identity mismatch: {sid}")
        if f and f.get("timestamp_ns") != s.get("signal_timestamp_ns"):
            errors.append(f"signal/feature timestamp mismatch: {sid}")
        d, a, c = int(s["direction_sign"]), float(s["anchor_price"]), float(s["signal_close"])
        correct = d * (a - c) > 0 if s["model"] == "FAIR_PRICE_REVERSION" else d * (c - a) > 0
        if not correct: errors.append(f"model direction not aligned to anchor: {sid}")
    trade_errors=[]
    for t in trades:
        s=smap.get(str(t["signal_id"]))
        if s is None: continue
        expected_direction="LONG" if int(s["direction_sign"])>0 else "SHORT"
        if t.get("direction")!=expected_direction: trade_errors.append(f"trade direction mismatch: {t['signal_id']}")
        for k in ("date","session","model","trigger"):
            if t.get(k)!=s.get(k): trade_errors.append(f"trade {k} mismatch: {t['signal_id']}")
        if t.get("signal_timestamp_ns")!=s.get("signal_timestamp_ns"):
            trade_errors.append(f"trade signal timestamp mismatch: {t['signal_id']}")
        if int(t["entry_timestamp_ns"]) < int(s["signal_timestamp_ns"]):
            trade_errors.append(f"trade entry precedes signal: {t['signal_id']}")
    dates = {x["date"] for x in source.get("files", [])}
    if len(dates) != 54 or dates != set(manifest.get("completed_dates", [])):
        errors.append("source coverage dates do not equal 54 completed dates")
    integrity = {
        "v2_run_manifest_status": manifest["status"], "v2_summary_status": _json(V2 / "summary.json").get("status"),
        "artifact_hashes_verified": not bad_manifest and not bad_artifacts,
        "manifest_hash_mismatches": bad_manifest, "artifact_hash_mismatches": bad_artifacts,
        "v2_source_coverage_file_hash_matches_manifest": _sha(V2 / "source-coverage.json") == manifest["files"].get("source-coverage.json"),
        "upstream_source_coverage_sha256_matches": upstream_ok,
        "upstream_source_coverage_path": str(upstream_path) if upstream_path else None,
        "dates": len(dates), "signal_rows": len(signals), "unique_signals": len(smap),
        "trade_rows_across_exit_cells": len(trades), "signals_with_any_executed_cell": len(ids),
        "signal_alignment_errors": errors,
        "trade_alignment_errors": trade_errors,
        "verified_v2_file_sha256": dict(manifest["files"]),
        "verified_artifact_hash_manifest_sha256": _sha(V2 / "artifact-hashes.json"),
        "verified_upstream_source_coverage_sha256": upstream_ref.get("sha256"),
        "reversion_direction_checks": {"n": sum(s["model"] == "FAIR_PRICE_REVERSION" for s in signals),
                                        "pass": sum(s["model"] == "FAIR_PRICE_REVERSION" and int(s["direction_sign"])*(float(s["anchor_price"])-float(s["signal_close"])) > 0 for s in signals)},
        "continuation_direction_checks": {"n": sum(s["model"] == "OPENING_CONTINUATION" for s in signals),
                                          "pass": sum(s["model"] == "OPENING_CONTINUATION" and int(s["direction_sign"])*(float(s["signal_close"])-float(s["anchor_price"])) > 0 for s in signals)},
        "trade_signal_ids_are_subset": ids.issubset(smap),
        "input_tape_contract": "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE; BBO/trades/aggressor, no depth ladder"
    }
    return integrity, source, signals, features, trades


def _directional_bos(bm: Mapping[int, Mapping[str, Any]], minute: int, direction: int) -> bool:
    if minute-1 not in bm or minute-2 not in bm: return False
    a, b, c = bm[minute-1], bm[minute-2], bm[minute]
    return float(c["close"]) > max(float(a["high"]),float(b["high"])) if direction > 0 else float(c["close"]) < min(float(a["low"]),float(b["low"]))


def _combined_trigger_audit() -> dict[str, Any]:
    bull = {1:{"open":100.25,"close":100.5,"high":101.0,"low":100.0},2:{"open":100.5,"close":100.25,"high":100.75,"low":99.75},3:{"open":100.5,"close":101.25,"high":101.5,"low":100.25}}
    bear = {1:{"open":100.5,"close":100.25,"high":101.0,"low":99.0},2:{"open":100.25,"close":100.5,"high":100.75,"low":100.0},3:{"open":100.5,"close":98.75,"high":100.75,"low":98.5}}
    cases={}
    for name,bm,d in (("bull",bull,1),("bear",bear,-1)):
        disp=v2.displacement_candle(bm[3],bm[2],d)["valid"]
        cases[name]={"displacement":disp,"correct_directional_bos":_directional_bos(bm,3,d),
                     "v2_called_v1_bos":v1.bos_for_bar(bm,3,d)["confirmed"]}
    no_bos={1:{"open":100.25,"close":100.5,"high":102.0,"low":100.0},2:bull[2],3:bull[3]}
    cases["displacement_without_bos"]={"displacement":v2.displacement_candle(no_bos[3],no_bos[2],1)["valid"],"correct_directional_bos":_directional_bos(no_bos,3,1)}
    return {"classification":"IMPLEMENTATION_DEFECT",
        "proof":"V2 calls v1.bos_for_bar(bar_map, minute, intended_trade_direction). In V1, direction>0 tests current close below the two prior lows; direction<0 tests close above the two prior highs. Both are opposite-direction close-throughs. A same-direction displacement close beyond a prior high/low cannot also satisfy the inverted test.",
        "synthetic_sequences":cases,
        "correct_same_candle_combination":"FEASIBLE in bullish and bearish examples; displacement and two-bar directional BOS are compatible",
        "sequential_displacement_then_bos":"Distinct later trigger with different entry timing/path; not implemented or backtested",
        "historical_v2_bos_only":"INVALID for intended-direction inference",
        "historical_v2_bos_plus_displacement_zero":"Implementation artifact, not market evidence"}


def _load_day_tapes(source: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tapes={}; checks=[]
    for row in source["files"]:
        p=Path(row["candidate_tape_path"])
        ok=p.is_file() and _sha(p)==row["candidate_tape_sha256"]
        if not ok: raise DiagnosticError(f"candidate tape missing/hash mismatch: {p}")
        tape=load_tape(p,source_sha256=row["candidate_tape_source_sha256"],semantic_sha256=row["candidate_tape_semantic_sha256"])
        if tape.metadata.get("date")!=row["date"]: raise DiagnosticError(f"tape date mismatch: {row['date']}")
        tapes[row["date"]]=tape; checks.append({"date":row["date"],"sha256":row["candidate_tape_sha256"],"verified":True})
    return tapes,checks


def _quote_path(tape: Any, signal_ns: int, direction: int, end_ns: int, anchor: float) -> dict[str, Any]:
    e=tape.events; ts=e["timestamp_ns"].astype(np.int64,copy=False); bid=e["bid"].astype(float,copy=False); ask=e["ask"].astype(float,copy=False)
    valid=np.isfinite(bid)&np.isfinite(ask)&(bid>0)&(ask>=bid)&(ts<end_ns)
    ix=np.flatnonzero(valid); qts=ts[ix]; qb=bid[ix]; qa=ask[ix]
    entry_i=int(np.searchsorted(qts,signal_ns+2_000_000,side="left"))
    if entry_i>=len(qts): return {"entry_available":False,"horizons":{str(h):None for h in HORIZONS}}
    entry_ns=int(qts[entry_i]); entry_bid=float(qb[entry_i]); entry_ask=float(qa[entry_i]); entry_mid=(entry_bid+entry_ask)/2
    entry_fill=entry_ask+TICK if direction>0 else entry_bid-TICK
    last=int(np.searchsorted(qts,end_ns,side="left")); qts2=qts[entry_i:last]; b2=qb[entry_i:last]; a2=qa[entry_i:last]
    net_path=direction*((b2-TICK if direction>0 else a2+TICK)-entry_fill)/TICK-6.0/12.5
    horizon={}
    for sec in HORIZONS:
        j=int(np.searchsorted(qts,entry_ns+sec*1_000_000_000,side="right"))-1
        if j<entry_i or j>=last: horizon[str(sec)]=None; continue
        mid=(qb[j]+qa[j])/2; outpx=qb[j]-TICK if direction>0 else qa[j]+TICK
        horizon[str(sec)]={"mid_directional_ticks":direction*(mid-entry_mid)/TICK,
                           "executable_net_ticks":direction*(outpx-entry_fill)/TICK-6.0/12.5,"quote_timestamp_ns":int(qts[j])}
    trade_mask=(ts>=entry_ns)&(ts<end_ns)&(e["execution_size"]>0)&np.isfinite(e["execution_price"])
    trade_ix=np.flatnonzero(trade_mask); px=e["execution_price"][trade_ix].astype(float,copy=False)
    revisits=[]; prev=entry_mid
    for ti,p in zip(trade_ix,px):
        if abs(float(p)-anchor)<1e-9 or (prev-anchor)*(float(p)-anchor)<0: revisits.append(int(ts[ti]))
        prev=float(p)
    return {"entry_available":True,"entry_timestamp_ns":entry_ns,"entry_bid":entry_bid,"entry_ask":entry_ask,
            "entry_fill":entry_fill,"entry_mid":entry_mid,"horizons":horizon,
            "mfe_executable_net_ticks":max(0.0,float(np.max(net_path))) if len(net_path) else None,
            "mae_executable_ticks":max(0.0,-float(np.min(net_path))) if len(net_path) else None,
            "anchor_revisit_ns":revisits[0] if revisits else None,"anchor_revisits_ns":revisits,
            "seconds_to_anchor_revisit":(revisits[0]-entry_ns)/1e9 if revisits else None}


def _atr14(bars: Sequence[Mapping[str,Any]], minute: int) -> float|None:
    bm={int(b["minute_index"]):b for b in bars}; vals=[]
    for i in range(minute-13,minute+1):
        if i not in bm or i-1 not in bm:return None
        b,p=bm[i],bm[i-1]
        vals.append(max(float(b["high"])-float(b["low"]),abs(float(b["high"])-float(p["close"])),abs(float(b["low"])-float(p["close"]))) / TICK)
    return statistics.mean(vals) if vals else None


def _event_paths(signals: Sequence[Mapping[str,Any]], source: Mapping[str,Any]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    tapes,tape_checks=_load_day_tapes(source); by=defaultdict(list)
    for s in signals:by[(s["date"],s["session"])].append(s)
    out=[]
    for (day,session),ss in sorted(by.items()):
        tape=tapes[day]; start,end=v2._session_ns(day,session); bars,tts,px,_,_=v2._bars_for_session(tape.events,start,end)
        for s in ss:
            d=int(s["direction_sign"]); minute=int(s["minute_index"]); path=_quote_path(tape,int(s["signal_timestamp_ns"]),d,end,float(s["anchor_price"]))
            entry=path.get("entry_fill"); stop=float(s["episode_adverse_extreme"])-d*TICK if s.get("episode_adverse_extreme") is not None else None
            distance=abs(float(s["signal_close"])-float(s["anchor_price"]))/TICK
            atr14=_atr14(bars,minute)
            entry_ns=path.get("entry_timestamp_ns")
            # Operational definition: after executable entry, find the first completed
            # minute whose close-to-close move opposes the predicted direction, but
            # only after at least one post-entry minute first moved in that direction.
            first_favorable=None; first_reversal=None
            if entry_ns is not None:
                for bi in range(minute+1,len(bars)):
                    b=bars[bi]; prev=bars[bi-1]
                    if int(b["end_ns"])<int(entry_ns): continue
                    move=d*(float(b["close"])-float(prev["close"]))
                    if move>0 and first_favorable is None: first_favorable=int(b["end_ns"])
                    elif move<0 and first_favorable is not None:
                        first_reversal=int(b["end_ns"]); break
            out.append({"signal_id":s["signal_id"],"date":day,"period":s["period"],"session":session,"model":s["model"],"trigger":s["trigger"],
                "direction":s["direction"],"direction_sign":d,"minute_since_open":minute,"anchor_price":s["anchor_price"],"signal_close":s["signal_close"],
                "anchor_distance_ticks":distance,"episode_max_displacement_ticks":s.get("v1_episode_max_displacement_ticks"),"features":s.get("features",{}),**path,
                "pre_entry_directional_move_ticks":d*(float(path["entry_mid"])-float(s["signal_close"]))/TICK if path.get("entry_available") else None,
                "seconds_to_first_post_entry_favorable_minute":(first_favorable-entry_ns)/1e9 if first_favorable is not None else None,
                "seconds_to_first_post_favorable_reversal_minute":(first_reversal-entry_ns)/1e9 if first_reversal is not None else None,
                "episode_structural_stop_distance_ticks":d*(entry-stop)/TICK if entry is not None and stop is not None else None,
                "atr14_ticks":atr14,
                "structural_risk_to_atr14_ratio":(d*(entry-stop)/TICK)/atr14 if entry is not None and stop is not None and atr14 else None})
    return out,tape_checks


def _match_controls(paths: list[dict[str,Any]],signals: Sequence[Mapping[str,Any]],source: Mapping[str,Any]) -> dict[str,Any]:
    tapes={r["date"]:load_tape(Path(r["candidate_tape_path"]),source_sha256=r["candidate_tape_source_sha256"],
                                semantic_sha256=r["candidate_tape_semantic_sha256"]) for r in source["files"]}
    sig_minutes=defaultdict(set)
    sig_by_group=defaultdict(list)
    for s in signals:
        key=(s["date"],s["session"]);sig_minutes[key].add(int(s["minute_index"]))
        sig_by_group[key].append(s)
    path_by={p["signal_id"]:p for p in paths}
    matches=[]
    for (day,session),group_signals in sorted(sig_by_group.items()):
        start,end=v2._session_ns(day,session);tape=tapes[day];bars,_,_,_,_=v2._bars_for_session(tape.events,start,end)
        bm={int(b["minute_index"]):b for b in bars}; anchor=float(group_signals[0]["anchor_price"]); used=set()
        controls={model:[] for model in v2.MODELS}
        for b in bars:
            m=int(b["minute_index"]); model="OPENING_CONTINUATION" if m<v2.CONTINUATION_MINUTES else "FAIR_PRICE_REVERSION"
            close=float(b["close"])
            if model not in controls or m in sig_minutes[(day,session)] or m-2 not in bm or close==anchor:continue
            side=1 if close>anchor else -1;direction=side if model=="OPENING_CONTINUATION" else -side
            prior=float(bm[m-2]["close"]);ret=close-prior;trend=1 if ret>0 else -1 if ret<0 else 0
            atr=_atr14(bars,m)
            if atr is None or atr<=0:continue
            controls[model].append({"minute":m,"timestamp_ns":int(b["end_ns"]),"close":close,"direction":direction,"trend":trend,
                                    "atr":float(atr),"dist":abs(close-anchor)/TICK})
        selected=[s for s in group_signals if s["trigger"]=="DISPLACEMENT_CANDLE"]
        selected.sort(key=lambda s:(int(s["minute_index"]),str(s["signal_id"])))
        for s in selected:
            p=path_by[str(s["signal_id"])]
            if not p.get("entry_available"):continue
            m=int(s["minute_index"]);model=str(s["model"])
            if m-2 not in bm:continue
            r2=float(bm[m]["close"])-float(bm[m-2]["close"]);trend=1 if r2>0 else -1 if r2<0 else 0
            atr=p.get("atr14_ticks");dist=float(p["anchor_distance_ticks"])
            if atr is None or atr<=0:continue
            candidates=[]
            for c in controls[model]:
                if c["minute"] in used or c["direction"]!=int(s["direction_sign"]) or c["trend"]!=trend or abs(c["minute"]-m)>5:continue
                vol=abs(math.log(c["atr"]/float(atr)));dd=abs(c["dist"]-dist);caliper=max(8.0,.5*dist)
                score=abs(c["minute"]-m)/5+vol/.5+dd/caliper
                candidates.append((score,abs(c["minute"]-m),c["minute"],c))
            if not candidates:continue
            c=min(candidates,key=lambda x:(x[0],x[1],x[2]))[3];used.add(c["minute"])
            control_path=_quote_path(tape,c["timestamp_ns"],int(s["direction_sign"]),end,anchor)
            if not control_path.get("entry_available"):continue
            p["control_match"]={"control_minute":c["minute"],"control_timestamp_ns":c["timestamp_ns"],"control_atr14_ticks":c["atr"],
                "control_anchor_distance_ticks":c["dist"],"control_path":control_path,
                "fixed_matching_rules":{"same_date_session_phase_direction_and_trailing_2m_direction":True,"time_caliper_minutes":5,
                    "absolute_log_atr14_ratio_caliper":.5,"anchor_distance_difference_caliper_ticks":max(8,.5*dist),"without_replacement":True}}
            matches.append(p)
    return {"matched_signals":len(matches),"matched_by_class_period":{
        f"{model}|{period}":sum(p["model"]==model and p["period"]==period for p in matches)
        for model in v2.MODELS for period in ("SPRING_2025","OCTOBER_2025")},
        "unmatched_by_class_period":{
        f"{model}|{period}":sum(p["model"]==model and p["trigger"]=="DISPLACEMENT_CANDLE" and p["period"]==period for p in paths)-sum(p["model"]==model and p["period"]==period for p in matches)
        for model in v2.MODELS for period in ("SPRING_2025","OCTOBER_2025")},
        "rules":"same date/session/phase, intended direction, and trailing 2m direction; within 5 minutes; absolute log ATR14 ratio <=0.50; anchor distance diff <= max(8 ticks, 50% signal distance); nearest normalized covariate distance; no replacement; no future outcomes used"}


def _paired_control_results(paths: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    result={}
    for model in v2.MODELS:
        for period in ("SPRING_2025","OCTOBER_2025"):
            matched=[p for p in paths if p["model"]==model and p["trigger"]=="DISPLACEMENT_CANDLE" and p["period"]==period and p.get("control_match")]
            by_h={}
            for h in HORIZONS:
                by_date=defaultdict(list)
                for p in matched:
                    a=p["horizons"].get(str(h));b=p["control_match"]["control_path"]["horizons"].get(str(h))
                    if a and b:by_date[p["date"]].append(a["executable_net_ticks"]-b["executable_net_ticks"])
                by_h[str(h)]=_bootstrap(by_date,seed=31817+h)
            result[f"{model}|{period}"]={"matched_pairs":len(matched),"unmatched":sum(p["model"]==model and p["trigger"]=="DISPLACEMENT_CANDLE" and p["period"]==period for p in paths)-len(matched),
                "paired_signal_minus_control_net_ticks":by_h}
    return result


def _corr(x:Sequence[float],y:Sequence[float])->float|None:
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None


def _flow_analysis(paths: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    result={}
    for model in v2.MODELS:
        for period in ("SPRING_2025","OCTOBER_2025"):
            rows=[p for p in paths if p["model"]==model and p["trigger"]=="DISPLACEMENT_CANDLE" and p["period"]==period and p.get("control_match") and p["horizons"].get("300") and p["control_match"]["control_path"]["horizons"].get("300")]
            rel={}
            for field in FLOW_FIELDS:
                pairs=[]
                for p in rows:
                    x=p.get("features",{}).get(field)
                    if x is None or not math.isfinite(float(x)):continue
                    y=p["horizons"]["300"]["executable_net_ticks"]-p["control_match"]["control_path"]["horizons"]["300"]["executable_net_ticks"]
                    pairs.append((float(x),float(y)))
                by_date=defaultdict(list)
                for p in rows:
                    x=p.get("features",{}).get(field)
                    if x is None or not math.isfinite(float(x)):continue
                    y=p["horizons"]["300"]["executable_net_ticks"]-p["control_match"]["control_path"]["horizons"]["300"]["executable_net_ticks"]
                    by_date[p["date"]].append((float(x),float(y)))
                dates=sorted(by_date);rng=random.Random(31817+len(field));boot=[]
                for _ in range(800):
                    sample=[z for day in (rng.choice(dates) for _ in dates) for z in by_date[day]]
                    br=_corr([x for x,_ in sample],[y for _,y in sample])
                    if br is not None:boot.append(br)
                rel[field]={"matched_pairs":len(pairs),"pearson_r_with_matched_5m_excess_ticks":_corr([x for x,_ in pairs],[y for _,y in pairs]),
                            "ci95_date_cluster_bootstrap":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))] if boot else None,
                            "date_clusters":len(dates)}
            reversal={}
            for val in (True,False):
                selected=[p for p in rows if p.get("features",{}).get("aggression_reversal") is val]
                by_date=defaultdict(list)
                for p in selected:
                    by_date[p["date"]].append(p["horizons"]["300"]["executable_net_ticks"]-
                        p["control_match"]["control_path"]["horizons"]["300"]["executable_net_ticks"])
                reversal[str(val)]={"n":sum(map(len,by_date.values())),"mean_matched_5m_excess_ticks":_mean([x for xs in by_date.values() for x in xs]),
                    "date_cluster_bootstrap":_bootstrap(by_date,seed=441+int(val))}
            result[f"{model}|{period}"]={"matched_signal_count":len(rows),"flow_relationships":rel,"aggression_reversal":reversal}
    return {"method":"continuous features related to matched 5m executable excess return; date/session/time/volatility/direction/anchor distance controlled by matching; descriptive only",
            "by_model_period":result,"TOP5_imbalance":"UNAVAILABLE in sealed BBO tape; no raw depth reconstruction",
            "normalized_MLOFI":"UNAVAILABLE: no compatible event-time feature cache; not reconstructed"}


def _summaries(paths: Sequence[Mapping[str,Any]],trades:Sequence[Mapping[str,Any]]) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    cont={};rev={};daily=defaultdict(list)
    for model in v2.MODELS:
        for trigger in v2.TRIGGERS:
            for period in ("SPRING_2025","OCTOBER_2025"):
                for session in ("NY_AM","NY_PM"):
                    rows=[p for p in paths if p["model"]==model and p["trigger"]==trigger and p["period"]==period and p["session"]==session]
                    horizons={}
                    for h in HORIZONS:
                        vals=[p["horizons"][str(h)]["executable_net_ticks"] for p in rows if p.get("horizons",{}).get(str(h))]
                        mid=[p["horizons"][str(h)]["mid_directional_ticks"] for p in rows if p.get("horizons",{}).get(str(h))]
                        horizons[str(h)]={**_stats(vals),"positive_fraction":sum(x>0 for x in vals)/len(vals) if vals else None,
                            "mid_directional_ticks":_stats(mid),"positive_mid_direction_fraction":sum(x>0 for x in mid)/len(mid) if mid else None}
                    group={"signals":len(rows),"entry_quotes":sum(p.get("entry_available",False) for p in rows),"forward_net_ticks":horizons,
                           "MFE_ticks":_stats([p.get("mfe_executable_net_ticks") for p in rows]),"MAE_ticks":_stats([p.get("mae_executable_ticks") for p in rows]),
                           "initial_anchor_distance_ticks":_stats([p["anchor_distance_ticks"] for p in rows]),"pre_entry_move_ticks":_stats([p.get("pre_entry_directional_move_ticks") for p in rows])}
                    group["seconds_to_first_post_entry_favorable_minute"]=_stats([p.get("seconds_to_first_post_entry_favorable_minute") for p in rows])
                    group["seconds_to_first_post_favorable_reversal_minute"]=_stats([p.get("seconds_to_first_post_favorable_reversal_minute") for p in rows])
                    group["seconds_to_anchor_revisit"]=_stats([p.get("seconds_to_anchor_revisit") for p in rows])
                    dest=cont if model=="OPENING_CONTINUATION" else rev
                    dest[f"{trigger}|{period}|{session}"]=group
                    for p in rows:daily[(p["date"],session,model,trigger)].append(p)
    cell={(t["signal_id"],t["stop_family"],t["target_model"]):t for t in trades}
    for key,group in rev.items():
        trigger,period,session=key.split("|")
        rows=[p for p in paths if p["model"]=="FAIR_PRICE_REVERSION" and p["trigger"]==trigger and p["period"]==period and p["session"]==session]
        cell_rows=[cell[(p["signal_id"],"EPISODE_STRUCTURAL","FAIR_PRICE_ANCHOR")] for p in rows if (p["signal_id"],"EPISODE_STRUCTURAL","FAIR_PRICE_ANCHOR") in cell]
        stopped=[t for t in cell_rows if t["outcome"]=="STOP"]
        revisits=[]; stop_exceeded=[]; pre_entry_favorable=[]
        for t in stopped:
            p=next((x for x in rows if x["signal_id"]==t["signal_id"]),None)
            revisits.append(bool(p and any(int(ts)>int(t["exit_timestamp_ns"]) for ts in p.get("anchor_revisits_ns",[]))))
        for p in rows:
            risk=p.get("episode_structural_stop_distance_ticks"); mae=p.get("mae_executable_ticks")
            if risk is not None and mae is not None: stop_exceeded.append(float(mae)>float(risk))
            move=p.get("pre_entry_directional_move_ticks")
            if move is not None: pre_entry_favorable.append(float(move)>0)
        group.update({"episode_structural_fair_anchor_cell_executions":len(cell_rows),
            "cell_outcomes":{o:sum(t["outcome"]==o for t in cell_rows) for o in ("TARGET","STOP","SESSION_END")},
            "target_before_stop_fraction":sum(t["outcome"]=="TARGET" for t in cell_rows)/len(cell_rows) if cell_rows else None,
            "stop_distance_ticks":_stats([t["stop_distance_ticks"] for t in cell_rows]),
            "target_to_stop_ratio":_stats([t["target_distance_ticks"]/t["stop_distance_ticks"] for t in cell_rows if t["stop_distance_ticks"]>0]),
            "episode_structural_stop_distance_ticks":_stats([p.get("episode_structural_stop_distance_ticks") for p in rows]),
            "structural_risk_to_atr14_ratio":_stats([p.get("structural_risk_to_atr14_ratio") for p in rows]),
            "seconds_to_cell_exit":_stats([(t["exit_timestamp_ns"]-t["entry_timestamp_ns"])/1e9 for t in cell_rows]),
            "seconds_to_stop_for_stop_outcomes":_stats([(t["exit_timestamp_ns"]-t["entry_timestamp_ns"])/1e9 for t in cell_rows if t["outcome"]=="STOP"]),
            "net_r_one_cell_per_signal":_stats([t["net_r"] for t in cell_rows]),
            "gross_r_one_cell_per_signal":_stats([t["gross_r"] for t in cell_rows]),
            "fees_usd_one_cell_per_signal":_stats([t["fees_usd"] for t in cell_rows]),
            "anchor_revisited_after_stop_fraction":sum(revisits)/len(revisits) if revisits else None,
            "adverse_path_exceeded_structural_risk_fraction":sum(stop_exceeded)/len(stop_exceeded) if stop_exceeded else None,
            "favorable_move_occurred_before_executable_entry_fraction":sum(pre_entry_favorable)/len(pre_entry_favorable) if pre_entry_favorable else None,
            "warning":"one predeclared structural-stop/fair-anchor cell only; BOS-only excluded from inference"})
    for key,group in cont.items():
        trigger,period,session=key.split("|")
        rows=[p for p in paths if p["model"]=="OPENING_CONTINUATION" and p["trigger"]==trigger and p["period"]==period and p["session"]==session]
        grid={}
        for stop_family in v2.STOP_FAMILIES:
            for target_model in ("FIXED_1R","FIXED_1.5R","FIXED_2R"):
                cell_rows=[cell[(p["signal_id"],stop_family,target_model)] for p in rows
                           if (p["signal_id"],stop_family,target_model) in cell]
                grid[f"{stop_family}|{target_model}"]={"executions":len(cell_rows),
                    "distinct_signals":len({t["signal_id"] for t in cell_rows}),
                    "outcomes":{o:sum(t["outcome"]==o for t in cell_rows) for o in ("TARGET","STOP","SESSION_END")},
                    "target_before_stop_fraction":sum(t["outcome"]=="TARGET" for t in cell_rows)/len(cell_rows) if cell_rows else None,
                    "mean_net_r":_mean([float(t["net_r"]) for t in cell_rows]),
                    "stop_first_precedence":"same quote observation resolving both stop and target is recorded STOP"}
        group["fixed_exit_grid_descriptive_not_independent"] = grid
    dayrows=[]
    dates=sorted({p["date"] for p in paths})
    for day in dates:
        for session in ("NY_AM","NY_PM"):
            for model in v2.MODELS:
                for trigger in v2.TRIGGERS:
                    rows=daily.get((day,session,model,trigger),[])
                    key=(day,session,model,trigger)
                    vals=[(p["horizons"].get("300") or {}).get("executable_net_ticks") for p in rows]
                    dayrows.append({"date":key[0],"session":key[1],"model":key[2],"trigger":key[3],"raw_signals":len(rows),
                        "entry_quotes":sum(p.get("entry_available",False) for p in rows),"mean_5m_net_ticks":_mean([x for x in vals if x is not None])})
    return cont,rev,{"date_session_model_trigger":dayrows,"unique_signal_path_rows":len(paths),"trade_rows_across_exit_grid":len(trades),
                     "unique_signal_ids_executed_anywhere":len({t["signal_id"] for t in trades})}


def _write_daily(rows: Sequence[Mapping[str,Any]], path: Path) -> None:
    fields=("date","session","model","trigger","raw_signals","entry_quotes","mean_5m_net_ticks")
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:r.get(k) for k in fields})


def _classify(paired: Mapping[str,Any],model: str,positive_cells: bool) -> str:
    periods=[paired.get(f"{model}|{p}",{}).get("paired_signal_minus_control_net_ticks",{}).get("300",{}) for p in ("SPRING_2025","OCTOBER_2025")]
    if any(x.get("date_clusters",0)<8 or x.get("ci95") is None for x in periods):
        return "REVERSION_EVIDENCE_INCONCLUSIVE" if model=="FAIR_PRICE_REVERSION" else "CONTINUATION_EVIDENCE_INCONCLUSIVE"
    if all(x["mean"]>0 and x["ci95"][0]>0 for x in periods):
        return "REVERSION_SIGNAL_HAS_PREDICTIVE_INFORMATION" if model=="FAIR_PRICE_REVERSION" else "CONTINUATION_SIGNAL_HAS_PREDICTIVE_INFORMATION"
    if model=="FAIR_PRICE_REVERSION":
        return "REVERSION_IS_DESCRIPTIVE_NOT_ECONOMICALLY_EXPLOITABLE" if not positive_cells else "REVERSION_SIGNAL_UNSUPPORTED"
    return "CONTINUATION_POSITIVE_CELLS_EXPLAINED_BY_SELECTION" if positive_cells else "CONTINUATION_SIGNAL_UNSUPPORTED"


def _failure_decomposition(rev: Mapping[str,Any], paths: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    """Overlapping diagnostic flags from one predeclared reversion exit cell."""
    out={}
    for key,g in rev.items():
        if not key.startswith("DISPLACEMENT_CANDLE|"): continue
        out[key]={"no_positive_mid_direction_fraction":
                    (1-g["forward_net_ticks"]["300"]["positive_mid_direction_fraction"]) if g["forward_net_ticks"]["300"]["positive_mid_direction_fraction"] is not None else None,
                  "favorable_move_before_entry_fraction":g.get("favorable_move_occurred_before_executable_entry_fraction"),
                  "adverse_path_exceeded_structural_risk_fraction":g.get("adverse_path_exceeded_structural_risk_fraction"),
                  "target_to_stop_ratio":g.get("target_to_stop_ratio"),
                  "execution_costs":{"per_side_adverse_plus_half_fee_ticks":1+3/12.5,"round_trip_explicit_adverse_ticks":2,
                      "round_trip_fee_ticks":6/12.5,"round_trip_adverse_plus_fee_ticks":2+6/12.5,
                      "note":"Variable bid/ask spread is embedded in executable-side pricing; one adverse tick each side plus $6 round-trip fees; attribution is descriptive, not causal"},
                  "anchor_revisited_after_stop_fraction":g.get("anchor_revisited_after_stop_fraction"),
                  "session_date_concentration_not_additive":True,
                  "categories_overlap":True}
    concentration={}
    for period in ("SPRING_2025","OCTOBER_2025"):
        for session in ("NY_AM","NY_PM"):
            rows=[p for p in paths if p["model"]=="FAIR_PRICE_REVERSION" and p["trigger"]=="DISPLACEMENT_CANDLE"
                  and p["period"]==period and p["session"]==session]
            by_date=defaultdict(int)
            for p in rows: by_date[p["date"]]+=1
            counts=sorted(by_date.values(),reverse=True)
            concentration[f"{period}|{session}"]={"signals":len(rows),"active_dates":len(by_date),
                "largest_date_signal_share":counts[0]/len(rows) if rows else None,
                "top_5_date_signal_share":sum(counts[:5])/len(rows) if rows else None,
                "interpretation":"concentration of signal counts only, not PnL attribution"}
    return {"method":"descriptive flags from displacement-only reversion signals and one preregistered EPISODE_STRUCTURAL|FAIR_PRICE_ANCHOR exit cell; no additive attribution",
            "by_cell":out,"date_session_concentration":concentration,
            "unexplained":"This dataset cannot uniquely attribute outcomes to mechanism; the BBO tape omits depth/order-book state, and sampled exit-cell outcomes are not causal counterfactuals."}


def _report(result: Mapping[str,Any]) -> str:
    integ=result["integrity"]; decisions=result["decisions"]; paired=result["matched_controls"]["results"]
    lines=["# ES JJ Fair Pricing V2 mechanism diagnostic","",
      "## Verdict and scope","",f"- Status: **{result['status']}**; development-period diagnostic only.",
      f"- Reversion: `{decisions['reversion']}`.",f"- Continuation: `{decisions['continuation']}`.",
      f"- Combined trigger: `{decisions['combined_trigger']}`.",
      "- No optimization, new strategy backtest, raw DBN read, depth reconstruction, or 2026/OOS access.",
      "- BOS-only historical signals are not interpretable as intended-direction BOS because V2 called the opposite-direction V1 helper; combined-trigger zero is an implementation defect, not evidence of impossibility.","",
      "## Integrity and units","",f"- 54 dates; {integ['signal_rows']} unique signal rows; {integ['trade_rows_across_exit_cells']} hypothetical executions across fixed exit configurations; {integ['signals_with_any_executed_cell']} distinct signals executed at least once.",
      "- Exit-cell rows are repeated counterfactual configurations, not independent observations. Signal-path analysis uses one row per signal.",
      "- Candidate tapes are sealed BBO/trade/aggressor tapes without depth ladders. Source MBP-10 was not read.",
      "- Forward diagnostic path: first valid BBO at/after signal+2ms; marketable side plus one adverse tick; $6 RT fee; endpoint executable side plus one adverse tick. This is a fixed descriptive path, not the V2 stop/target strategy simulation.","",
      "## Matched controls","","Controls are same date/session/phase/model, intended direction and trailing 2-minute direction; within 5 minutes; log ATR14 ratio ≤0.50; anchor-distance difference ≤ max(8 ticks, 50% of signal distance); deterministic nearest normalized distance; without replacement. Matching uses no future outcome. Confidence intervals resample trading dates, keeping within-date observations clustered.","",
      "| Model / period | Pairs | 5m paired excess net ticks | Date-cluster 95% CI |","|---|---:|---:|---|"]
    for model in v2.MODELS:
      for period in ("SPRING_2025","OCTOBER_2025"):
        cell=paired.get(f"{model}|{period}",{});h=cell.get("paired_signal_minus_control_net_ticks",{}).get("300",{})
        ci=h.get("ci95");lines.append(f"| {model} / {period} | {cell.get('matched_pairs',0)} | {h.get('mean')} | {ci} |")
    lines += ["","## Failure-topology interpretation","","Reversion timing, anchor distance, pre-entry directional movement, MFE/MAE, target-before-stop, stop timing, anchor revisits after stop, and risk/ATR are reported by period/session/trigger. These are overlapping descriptions, not an additive causal decomposition. The single structural-stop/fair-anchor cell is predeclared and should not be generalized to the 126-cell grid.",
      "The operational first-reversal measure is the first completed minute after entry moving opposite the predicted direction, after at least one favorable completed minute; it is not a tick-level reversal label.",
      "Positive forward path at an isolated horizon or positive exit-cell PnL does not establish predictive information; matched-control effect and clustered uncertainty govern the decision.","",
      "## Order flow and periods","","Order-flow relationships use matched 5-minute excess outcomes, period-separated estimates, date-cluster bootstrap intervals, and no threshold search or ML. They remain exploratory with multiple comparisons. TOP5 imbalance and normalized MLOFI are unavailable.",
      "Spring and October are reported separately. Both are previously researched development periods, not untouched OOS.","",
      "## Integrity flags","",f"- Artifact hashes verified: {integ['artifact_hashes_verified']}",f"- Signal alignment errors: {integ['signal_alignment_errors']}",
      f"- Candidate tape rows hash-verified: {integ.get('candidate_tapes_verified')}","",
      "## No production conclusion","","These results do not establish a production-ready edge. The mechanism is not fully identified without independent data, depth features, and a preregistered follow-up; no further trigger strategy was implemented or backtested.",""]
    return "\n".join(lines)


def run(*,output_root:Path=OUT,force:bool=False)->dict[str,Any]:
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise DiagnosticError(f"output exists; pass --force to regenerate: {output_root}")
    output_root.mkdir(parents=True,exist_ok=True)
    integrity,source,signals,features,trades=_verify_source()
    paths,tape_checks=_event_paths(signals,source)
    match_meta=_match_controls(paths,signals,source)
    paired=_paired_control_results(paths)
    flow=_flow_analysis(paths)
    cont,rev,daily=_summaries(paths,trades)
    failure=_failure_decomposition(rev,paths)
    combo=_json(V2/"session-model-trigger-results.json")["combinations"]
    actual=defaultdict(int)
    for s in signals:actual[(s["session"],s["model"],s["trigger"])] += 1
    expected={(r["session"],r["model"],r["trigger"]):r["raw_qualified_signals"] for r in combo}
    integrity["signal_counts_match_frozen_combo_summary"]=all(actual.get(k,0)==n for k,n in expected.items()) and all(k in expected for k in actual)
    integrity["candidate_tapes_verified"]=len(tape_checks)
    integrity["signal_path_rows"]=len(paths)
    integrity["trade_alignment"]={"rows_across_fixed_exit_grid":len(trades),"distinct_signal_ids":len({t["signal_id"] for t in trades}),
        "all_trade_signal_ids_present":all(t["signal_id"] in {s["signal_id"] for s in signals} for t in trades)}
    integrity["invalid_bos_signal_counts"]={m:sum(s["model"]==m and s["trigger"]=="BOS_ONLY" for s in signals) for m in v2.MODELS}
    integrity["zero_combined_trigger_count"]=sum(s["trigger"]=="BOS_PLUS_DISPLACEMENT" for s in signals)
    integrity["status"]="PARTIAL_BOS_SEMANTICS_DEFECT"
    trigger=_combined_trigger_audit()
    reversion_positive=any(g.get("net_r_one_cell_per_signal",{}).get("mean") is not None and g["net_r_one_cell_per_signal"]["mean"]>0 for g in rev.values())
    continuation_positive=any(g.get("forward_net_ticks",{}).get("300",{}).get("mean") is not None and g["forward_net_ticks"]["300"]["mean"]>0 for g in cont.values() if g.get("signals",0)>0)
    decisions={"reversion":_classify(paired,"FAIR_PRICE_REVERSION",positive_cells=reversion_positive),
               "continuation":_classify(paired,"OPENING_CONTINUATION",positive_cells=continuation_positive),
               "combined_trigger":trigger["classification"]}
    summary_v2=_json(V2/"summary.json")
    spring=_json(V2/"spring-results.json");october=_json(V2/"october-results.json")
    period={"spring_dates":spring["dates"],"october_dates":october["dates"],"spring_exit_cells":spring["metrics"],
            "october_exit_cells":october["metrics"],"same_sign_exit_cells":summary_v2["spring_october_compatibility"]["same_sign_cells"],
            "total_exit_cells":summary_v2["spring_october_compatibility"]["cell_count"],"interpretation":"development-period descriptive compatibility only"}
    result={"study_id":"ES_JJ_FAIR_PRICING_V2_MECHANISM_DIAGNOSTIC","status":"PARTIAL",
        "v2_summary":{k:summary_v2[k] for k in ("dates_processed","valid_ny_am_sessions","valid_ny_pm_sessions","total_raw_signals","total_sequential_trades_across_independent_fixed_exit_configurations","unique_signals_executed_at_least_once")},
        "integrity":integrity,"combined_trigger":trigger,"continuation":cont,"reversion":rev,"matched_controls":{"matching":match_meta,"results":paired},
        "orderflow":flow,"spring_october":period,"failure_decomposition":failure,"decisions":decisions,
        "scope":{"parameter_optimization_performed":False,"new_strategy_backtest_performed":False,"final_oos_accessed":False,
                  "2026_data_accessed":False,"production_files_modified":False,"commit_performed":False}}
    _write(output_root/"implementation-integrity.json",integrity)
    _write(output_root/"combined-trigger-feasibility.json",trigger)
    _write(output_root/"reversion-failure-topology.json",rev)
    _write(output_root/"continuation-event-study.json",cont)
    _write(output_root/"matched-control-comparison.json",result["matched_controls"])
    _write(output_root/"orderflow-mechanism-analysis.json",flow)
    _write(output_root/"spring-october-comparison.json",period)
    _write(output_root/"failure-decomposition.json",failure)
    _write_daily(daily["date_session_model_trigger"],output_root/"daily-mechanism-results.csv")
    _write(output_root/"summary.json",result)
    (output_root/"report.md").write_text(_report(result),encoding="utf-8")
    limitations=("# Limitations and next step\n\n"
      "This is a development-only, read-only diagnostic over sealed 2025 Candidate Tape V2 BBO/trade/aggressor records. "
      "It does not read raw DBN, reconstruct depth, access 2026/OOS, optimize, or evaluate a new strategy. The 54 sessions are not untouched OOS.\n\n"
      "The V2 intended-direction BOS call is semantically inverted through the V1 helper. Therefore historical BOS-only outcomes and the zero BOS-plus-displacement count cannot support directional BOS conclusions. The synthetic feasibility check proves compatible same-candle examples exist; no sequential trigger variant was implemented.\n\n"
      "Matched controls are deterministic and causal on listed covariates, but limited common support, small date-cluster counts, matching choices, and exploratory multiple comparisons constrain inference. Confidence intervals are descriptive date-cluster bootstraps, not confirmatory p-values. Order-flow associations are not causal. TOP5 imbalance and normalized MLOFI are unavailable.\n\n"
      "Next step: correct and separately preregister directional BOS semantics, validate its signal catalog with synthetic tests, then assess the new trigger on genuinely untouched data before any economic claim. Do not select parameters or production behavior from this report.\n")
    (output_root/"limitations-and-next-step.md").write_text(limitations,encoding="utf-8")
    artifacts=sorted(p for p in output_root.iterdir() if p.is_file() and p.name!="artifact-hashes.json")
    _write(output_root/"artifact-hashes.json",{"status":"HASHED","study_id":"ES_JJ_FAIR_PRICING_V2_MECHANISM_DIAGNOSTIC",
        "files":{p.name:_sha(p) for p in artifacts}})
    result["artifact_hashes"]={p.name:_sha(p) for p in artifacts}
    return result


def main(argv: Sequence[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=OUT)
    parser.add_argument("--force",action="store_true",help="regenerate this diagnostic output directory")
    args=parser.parse_args(argv)
    try:
        result=run(output_root=args.output_root,force=args.force)
    except (DiagnosticError, OSError, KeyError, ValueError) as exc:
        parser.exit(2,f"ERROR: {exc}\n")
    print(json.dumps({"status":result["status"],"output_root":str(args.output_root),
        "decisions":result["decisions"],"matched_pairs":result["matched_controls"]["matching"]["matched_signals"],
        "hash_manifest":str(args.output_root/"artifact-hashes.json")},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
