"""Aggregate the frozen 20-date four-mechanism construction outputs."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .es_four_mechanism_construction_v1 import STUDY_ROOT,_sha,_read_json,_atomic_json
from .es_four_mechanism_native_replay import _session_code

MECHANISMS=("M1","M2","M3","M4")
HORIZONS=(1,5,10,30,60)
BARRIERS=(2,4,6,8)
SEED=20261028
BOOTSTRAPS=5000


def _episode_annotation(events:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[str,str]]:
    """Apply the outcome-independent episode grouping frozen before aggregation."""
    rules={"M1":{"window_seconds":2,"key":"date/session; source protocol's 2s trigger cluster"},
           "M2":{"window_seconds":2,"key":"date/session/direction; repeated qualified thinning inside refractory interval"},
           "M3":{"window_seconds":1800,"key":"date/session/direction/reference boundary; trailing range episode"},
           "M4":{"window_seconds":0,"key":"one event per detector compression-expansion state cycle"}}
    by_mech={m:[] for m in MECHANISMS}
    for e in events:by_mech[e["mechanism"]].append(e)
    ledger=[]; event_to_episode={}
    for mech in MECHANISMS:
        rows=sorted(by_mech[mech],key=lambda e:(e["date"],int(e["ts_recv"]),int(e["source_ordinal"])))
        last={}; episode_no=0
        for e in rows:
            f=json.loads(e.get("features") or "{}")
            sess=int(f["session"]) if mech=="M1" and "session" in f else _session_code(e["date"],int(e["ts_recv"]))
            boundary=round(float(f.get("boundary",0.0)),9) if mech=="M3" else None
            key=(e["date"],sess)
            if mech in ("M2","M3"):key+=(int(e["direction"]),)
            if mech=="M3":key+=(boundary,)
            window=rules[mech]["window_seconds"]*1_000_000_000
            prior=last.get(key)
            if prior is None or window==0 or int(e["ts_recv"])-prior[0]>window:
                episode_no+=1; episode_id=f"{e['date']}:{mech}:EP{episode_no:07d}"
                count=1
            else:
                episode_id=prior[1]; count=prior[2]+1
            last[key]=(int(e["ts_recv"]),episode_id,count)
            event_to_episode[e["event_id"]]=episode_id
            ledger.append({"date":e["date"],"mechanism":mech,"episode_id":episode_id,
                "event_id":e["event_id"],"ts_recv":int(e["ts_recv"]),"source_ordinal":int(e["source_ordinal"]),
                "direction":int(e["direction"]),"session":sess,"is_episode_representative":count==1,
                "grouping_rule":rules[mech]["key"]})
    return ledger,rules,event_to_episode


def _write_episode_parquet(path:Path,rows:list[dict[str,Any]]):
    schema=pa.schema([("date",pa.string()),("mechanism",pa.string()),("episode_id",pa.string()),
        ("event_id",pa.string()),("ts_recv",pa.int64()),("source_ordinal",pa.int64()),
        ("direction",pa.int8()),("session",pa.int8()),("is_episode_representative",pa.bool_()),
        ("grouping_rule",pa.string())])
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),path,compression="zstd")


def _date_bootstrap(rows_by_date:dict[str,list[dict[str,Any]]], dates:list[str], value_key:str,
                    *, success_key:str|None=None, denom_key:str|None=None)->dict[str,Any]:
    usable=[]
    for day in dates:
        rows=[r for r in rows_by_date.get(day,[]) if r.get(value_key) is not None]
        if rows:
            if success_key:
                usable.append((day,sum(bool(r.get(success_key)) for r in rows),len(rows),None))
            else:
                vals=np.asarray([float(r[value_key]) for r in rows],dtype=np.float64)
                usable.append((day,float(vals.sum()),len(vals),float(vals.mean())))
    if not usable:return {"n_dates":0,"estimate":None,"ci95":None,"bootstrap_replicates":BOOTSTRAPS,"seed":SEED}
    rng=np.random.default_rng(SEED)
    indices=rng.integers(0,len(usable),size=(BOOTSTRAPS,len(usable)))
    draws=np.empty(BOOTSTRAPS,dtype=np.float64)
    if success_key:
        success=np.asarray([x[1] for x in usable],dtype=np.float64)
        den=np.asarray([x[2] for x in usable],dtype=np.float64)
        for i,idx in enumerate(indices):draws[i]=success[idx].sum()/den[idx].sum() if den[idx].sum() else np.nan
        obs=float(success.sum()/den.sum())
    else:
        sums=np.asarray([x[1] for x in usable],dtype=np.float64)
        nums=np.asarray([x[2] for x in usable],dtype=np.float64)
        for i,idx in enumerate(indices):draws[i]=sums[idx].sum()/nums[idx].sum() if nums[idx].sum() else np.nan
        obs=float(sums.sum()/nums.sum())
    finite=draws[np.isfinite(draws)]
    return {"n_dates":len(usable),"dates":[x[0] for x in usable],"estimate":obs,
            "ci95":[float(np.quantile(finite,.025)),float(np.quantile(finite,.975))] if len(finite) else None,
            "bootstrap_replicates":BOOTSTRAPS,"seed":SEED,"resampling_unit":"whole trading date",
            "date_metrics":{x[0]:({"successes":x[1],"denominator":x[2]} if success_key else {"sum":x[1],"count":x[2],"mean":x[3]}) for x in usable}}


def _signflip_p(date_means:list[float])->float|None:
    if len(date_means)<2:return None
    vals=np.asarray(date_means,dtype=np.float64); observed=abs(float(vals.mean()))
    rng=np.random.default_rng(SEED)
    signs=rng.choice(np.asarray([-1.0,1.0]),size=(BOOTSTRAPS,len(vals)))
    null=np.abs((signs*vals).mean(axis=1))
    return float((1+np.count_nonzero(null>=observed))/(BOOTSTRAPS+1))


def _holm(pvalues:dict[str,float|None])->dict[str,float|None]:
    valid=sorted(((p,k) for k,p in pvalues.items() if p is not None))
    out={k:None for k in pvalues}; running=0.0; m=len(valid)
    for i,(p,k) in enumerate(valid):
        running=max(running,min(1.0,(m-i)*p)); out[k]=running
    return out


def _overlap(events:list[dict[str,Any]])->dict[str,Any]:
    by_day_mech=defaultdict(list)
    for e in events:by_day_mech[(e["date"],e["mechanism"])].append(e)
    result={}
    for i,a in enumerate(MECHANISMS):
        for b in MECHANISMS[i+1:]:
            exact=within=agree=oppose=0
            for day in sorted({e["date"] for e in events}):
                left=sorted(by_day_mech[(day,a)],key=lambda e:int(e["ts_recv"]))
                right=sorted(by_day_mech[(day,b)],key=lambda e:int(e["ts_recv"]))
                rt=[int(e["ts_recv"]) for e in right]
                for x in left:
                    t=int(x["ts_recv"]); lo=bisect.bisect_left(rt,t-1_000_000_000); hi=bisect.bisect_right(rt,t+1_000_000_000)
                    for y in right[lo:hi]:
                        within+=1; exact+=int(t==int(y["ts_recv"]))
                        if int(x["direction"]) and int(y["direction"]):
                            agree+=int(int(x["direction"])==int(y["direction"]))
                            oppose+=int(int(x["direction"])==-int(y["direction"]))
            result[f"{a}-{b}"]={"pairs_within_1s":within,"pairs_exact_timestamp":exact,
                                 "directionally_agreeing":agree,"directionally_opposing":oppose}
    return result


def analyze(repo:Path,run_root:Path)->dict[str,Any]:
    study=repo/STUDY_ROOT; protocol=_read_json(study/"frozen-study-protocol.json")
    dates=list(protocol["scope"]["construction_dates"])
    progress=_read_json(run_root/"construction-progress.json")
    if progress.get("status")!="ALL_20_DATES_COMPLETE" or progress.get("completed_dates")!=dates:
        raise RuntimeError("20-date construction run is incomplete")
    all_events=[]; all_outcomes=[]; all_source=[]; equivalences=[]; runtimes=[]
    for day in dates:
        d=run_root/"dates"/day; manifest=_read_json(d/"date-manifest.json")
        if manifest.get("status")!="PASS" or manifest["source"]["date"]!=day:
            raise RuntimeError(f"date manifest invalid: {day}")
        if manifest["protocol_sha256"]!=_sha(study/"frozen-study-protocol.json"):
            raise RuntimeError(f"protocol hash mismatch in {day}")
        for name,rec in manifest["artifacts"].items():
            p=d/name
            if not p.is_file() or p.stat().st_size!=rec["size_bytes"] or _sha(p)!=rec["sha256"]:
                raise RuntimeError(f"artifact hash mismatch {day}/{name}")
        result=_read_json(d/"source-validity.json"); all_source.append(result)
        equivalences.append(_read_json(d/"native-depth-replay-equivalence.json"))
        runtimes.append(float(manifest["runtime_seconds"]))
        for m in MECHANISMS:
            table=pq.read_table(d/f"{m.lower()}-events.parquet")
            all_events.extend(table.to_pylist())
        all_outcomes.extend(pq.read_table(d/"construction-outcomes.parquet").to_pylist())
    ids=[e["event_id"] for e in all_events]
    if len(ids)!=len(set(ids)):raise RuntimeError("duplicate raw detector event IDs")
    outcome_ids=[r["event_id"] for r in all_outcomes]
    if len(outcome_ids)!=len(set(outcome_ids)) or set(ids)!=set(outcome_ids):
        raise RuntimeError("outcome ledger does not have exactly one row per trigger")
    base=run_root/"analysis-v1"; base.mkdir(parents=True,exist_ok=True)
    # Freeze the outcome-blind episode rule before any aggregate estimates.
    episode_ledger,episode_rules,event_to_episode=_episode_annotation(all_events)
    _atomic_json(base/"outcome-independent-episode-rules.json",{
        "frozen_before_aggregate_analysis":True,"rules":episode_rules,
        "note":"The M1 two-second cluster comes from the sealed protocol. M2 uses its frozen two-second refractory envelope. M3 groups attempts at the same boundary inside the 30-minute reference-range window. M4 emits at most once per compression-expansion state cycle.",
        "no_thresholds_or_rules_changed":True})
    _write_episode_parquet(base/"economic-episodes.parquet",episode_ledger)
    _write_episode_parquet(base/"raw-trigger-episode-map.parquet",episode_ledger)
    outcomes_by_id={r["event_id"]:r for r in all_outcomes}
    unique_ids={}
    for x in episode_ledger:
        if x["is_episode_representative"]:unique_ids[x["event_id"]]=x["episode_id"]
    unique_outcomes=[outcomes_by_id[event_id] for event_id in unique_ids]
    event_table=pq.read_table(pa.Table.from_pylist(all_events).to_batches()[0]) if False else None
    # Combined raw ledgers are preserved separately from deduplicated episodes.
    for m in MECHANISMS:
        rows=[e for e in all_events if e["mechanism"]==m]
        pq.write_table(pa.Table.from_pylist(rows),base/f"{m.lower()}-events.parquet",compression="zstd")
    pq.write_table(pq.read_table if False else pa.Table.from_pylist(all_outcomes),base/"one-date-forward-outcomes.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pylist(unique_outcomes),base/"deduplicated-episode-outcomes.parquet",compression="zstd")
    overlap=_overlap(all_events)
    _atomic_json(base/"cross-mechanism-overlap-diagnostics.json",{
        "count_window":"absolute trigger timestamp difference <=1s; exact timestamp also reported",
        "pairs":overlap})
    raw_by_mech={m:[e for e in all_events if e["mechanism"]==m] for m in MECHANISMS}
    outcome_by_mech={m:[r for r in all_outcomes if r["mechanism"]==m] for m in MECHANISMS}
    episode_by_mech={m:[x for x in episode_ledger if x["mechanism"]==m] for m in MECHANISMS}
    event_by_id={e["event_id"]:e for e in all_events}
    unique_outcome_by_mech={m:[outcomes_by_id[event_id] for event_id in unique_ids
                              if event_by_id[event_id]["mechanism"]==m] for m in MECHANISMS}
    diagnostics={}; source_censor={}; passage={}; exec_cost={}; raw_outcome_summary={}
    per_date={}; inferential={}; primary_ps={}; predictive_ps={}
    for m in MECHANISMS:
        evs=raw_by_mech[m]; outs=outcome_by_mech[m]; eps=episode_by_mech[m]; uouts=unique_outcome_by_mech[m]
        event_dates=defaultdict(list); out_dates=defaultdict(list)
        for e in evs:event_dates[e["date"]].append(e)
        for o in uouts:out_dates[o["date"]].append(o)
        gaps=[]; hours=Counter(); bursts=Counter(); interval=[]
        for day,rows in event_dates.items():
            ordered=sorted(rows,key=lambda e:(int(e["ts_recv"]),int(e["source_ordinal"])))
            prev=None
            for e in ordered:
                t=int(e["ts_recv"]); sess=_session_code(day,t)
                if m=="M1":sess=int(json.loads(e["features"]).get("session",sess))
                hours[f"session_{sess}"]+=1
                hour=datetime.fromtimestamp(t/1e9,tz=timezone.utc).strftime("%H")
                hours[f"utc_hour_{hour}"]+=1
                bursts[(day,t//60_000_000_000)]+=1
                if prev is not None:interval.append((t-int(prev["ts_recv"]))/1e9)
                prev=e
        valid=sum(bool(e.get("source_valid")) for e in evs)
        censor=Counter((r.get("censor_reason") or "NONE") for r in outs if r.get("source_censored"))
        per_horizon={}
        for sec in HORIZONS:
            key=f"mid_markout_{sec}s_ticks"; vals=[float(r[key]) for r in uouts if r.get(key) is not None]
            per_horizon[str(sec)]={"n_valid":len(vals),"n_total":len(uouts),"n_missing":len(uouts)-len(vals),
                "mean":float(np.mean(vals)) if vals else None,"median":float(np.median(vals)) if vals else None,
                "positive_share":float(np.mean(np.asarray(vals)>0)) if vals else None}
        by_barrier={}
        for b in BARRIERS:
            c=Counter(r.get(f"barrier_{b}_first") for r in uouts)
            by_barrier[str(b)]={k:int(c.get(k,0)) for k in ("PREDICTED_FIRST","OPPOSITE_FIRST","TIE_OR_ORDER_AMBIGUOUS","UNRESOLVED","SOURCE_CENSORED")}
        costs={}
        for scenario in ("baseline","moderate","severe"):
            key=f"net_executable_10s_{scenario}_ticks"; vals=[float(r[key]) for r in uouts if r.get(key) is not None]
            costs[scenario]={"n":len(vals),"n_censored":len(uouts)-len(vals),
                "mean_ticks":float(np.mean(vals)) if vals else None,"median_ticks":float(np.median(vals)) if vals else None,
                "positive_share":float(np.mean(np.asarray(vals)>0)) if vals else None}
        raw_costs={}
        for scenario in ("baseline","moderate","severe"):
            key=f"net_executable_10s_{scenario}_ticks"; vals=[float(r[key]) for r in outs if r.get(key) is not None]
            raw_costs[scenario]={"n":len(vals),"n_censored":len(outs)-len(vals),
                "mean_ticks":float(np.mean(vals)) if vals else None,"median_ticks":float(np.median(vals)) if vals else None,
                "positive_share":float(np.mean(np.asarray(vals)>0)) if vals else None}
        raw_horizons={}
        for sec in HORIZONS:
            key=f"mid_markout_{sec}s_ticks"; vals=[float(r[key]) for r in outs if r.get(key) is not None]
            raw_horizons[str(sec)]={"n_valid":len(vals),"n_total":len(outs),
                "mean":float(np.mean(vals)) if vals else None,"median":float(np.median(vals)) if vals else None}
        raw_barriers={}
        for b in BARRIERS:
            c=Counter(r.get(f"barrier_{b}_first") for r in outs)
            raw_barriers[str(b)]={k:int(c.get(k,0)) for k in ("PREDICTED_FIRST","OPPOSITE_FIRST","TIE_OR_ORDER_AMBIGUOUS","UNRESOLVED","SOURCE_CENSORED")}
        raw_outcome_summary[m]={"raw_trigger_count":len(outs),"forward_horizons":raw_horizons,
                                "first_passage":raw_barriers,"executable_cost_scenarios":raw_costs}
        daily_net={}; daily_pass={}
        for day in dates:
            rows=[r for r in out_dates.get(day,[]) if r.get("net_executable_10s_baseline_ticks") is not None]
            if rows:
                vals=np.asarray([r["net_executable_10s_baseline_ticks"] for r in rows],dtype=float)
                daily_net[day]={"n":len(vals),"mean":float(vals.mean()),"sum":float(vals.sum())}
            p_rows=[r for r in out_dates.get(day,[]) if r.get("barrier_4_first") in ("PREDICTED_FIRST","OPPOSITE_FIRST")]
            if p_rows:daily_pass[day]={"success":sum(r["barrier_4_first"]=="PREDICTED_FIRST" for r in p_rows),"n":len(p_rows)}
        net_boot=_date_bootstrap({d:[r for r in out_dates.get(d,[]) if r.get("net_executable_10s_baseline_ticks") is not None] for d in dates},dates,"net_executable_10s_baseline_ticks")
        passage_rows_by_day={d:[r for r in out_dates.get(d,[]) if r.get("barrier_4_first") in ("PREDICTED_FIRST","OPPOSITE_FIRST")] for d in dates}
        # Date-cluster CI for resolved predicted-side share.
        for d,rows in passage_rows_by_day.items():
            for r in rows:r["_predicted_success"]=r["barrier_4_first"]=="PREDICTED_FIRST"
        pass_boot=_date_bootstrap(passage_rows_by_day,dates,"_predicted_success",success_key="_predicted_success")
        means=[v["mean"] for v in daily_net.values()]
        pval=_signflip_p(means); primary_ps[m]=pval
        predictive_daily=[daily_pass[d]["success"]/daily_pass[d]["n"]-.5 for d in daily_pass if daily_pass[d]["n"]]
        predictive_ps[m]=_signflip_p(predictive_daily)
        sum_net=sum(v["sum"] for v in daily_net.values())
        positive_sum=sum(max(0.0,v["sum"]) for v in daily_net.values())
        date_consistency=sum(v["mean"]>0 for v in daily_net.values())
        loo=[]
        for omitted in dates:
            sample=[r for d in dates if d!=omitted for r in out_dates.get(d,[]) if r.get("net_executable_10s_baseline_ticks") is not None]
            loo.append(float(np.mean([r["net_executable_10s_baseline_ticks"] for r in sample])) if sample else None)
        inferential[m]={"primary_economic_endpoint":"deduplicated mean net executable 10-second markout, baseline costs, ES ticks",
            "net_executable_10s_date_cluster_bootstrap":net_boot,
            "four_tick_resolved_predicted_share_date_cluster_bootstrap":pass_boot,
            "leave_one_date_out_net_mean_range":[min(x for x in loo if x is not None),max(x for x in loo if x is not None)] if any(x is not None for x in loo) else None,
            "positive_dates":date_consistency,"dates_with_primary_observations":len(daily_net),
            "positive_net_date_sum_share":positive_sum/sum_net if sum_net>0 else None,
            "best_positive_date_net_share":max((max(0,v["sum"]) for v in daily_net.values()),default=0)/positive_sum if positive_sum>0 else None,
            "daily_means_for_primary_signflip":daily_net,"unadjusted_signflip_p":pval,
            "four_tick_primary_denominators_by_date":daily_pass}
        inter_sorted=sorted(interval)
        diagnostics[m]={"raw_triggers":len(evs),"unique_episodes":len(eps)-sum(1 for x in eps if not x["is_episode_representative"]),
            "repeated_triggers_within_episode":sum(not x["is_episode_representative"] for x in eps),
            "source_valid_triggers":valid,"source_valid_share":valid/len(evs) if evs else None,
            "inter_trigger_seconds":{"n":len(inter_sorted),"median":float(np.median(inter_sorted)) if inter_sorted else None,
                "q10":float(np.quantile(inter_sorted,.1)) if inter_sorted else None,"q90":float(np.quantile(inter_sorted,.9)) if inter_sorted else None,
                "q99":float(np.quantile(inter_sorted,.99)) if inter_sorted else None},
            "active_session_and_utc_hour_counts":dict(hours),
            "one_minute_burst":{"nonempty_bins":len(bursts),"max_triggers_per_minute":max(bursts.values(),default=0),
                "q95_triggers_per_nonempty_minute":float(np.quantile(list(bursts.values()),.95)) if bursts else None}}
        source_censor[m]={"outcome_rows":len(outs),"source_censored_outcomes":sum(bool(r["source_censored"]) for r in outs),
            "source_censor_rate":float(np.mean([bool(r["source_censored"]) for r in outs])) if outs else None,
            "primary_censor_reasons":dict(censor),"forward_horizon_coverage":per_horizon,
            "source_censored_first_passage_by_barrier":{b:by_barrier[b]["SOURCE_CENSORED"] for b in by_barrier}}
        passage[m]={"deduplicated_episodes":len(uouts),"barriers_ticks":by_barrier,
            "primary_four_tick_resolved_n":by_barrier["4"]["PREDICTED_FIRST"]+by_barrier["4"]["OPPOSITE_FIRST"],
            "primary_four_tick_predicted_share_resolved":(by_barrier["4"]["PREDICTED_FIRST"]/(by_barrier["4"]["PREDICTED_FIRST"]+by_barrier["4"]["OPPOSITE_FIRST"])) if (by_barrier["4"]["PREDICTED_FIRST"]+by_barrier["4"]["OPPOSITE_FIRST"]) else None,
            "forward_horizons":per_horizon}
        exec_cost[m]=costs
        per_date[m]={}
        for day in dates:
            raw_n=sum(e["date"]==day for e in evs)
            epi_n=sum(x["date"]==day and x["is_episode_representative"] for x in eps)
            selected=out_dates.get(day,[])
            per_date[m][day]={"raw_triggers":raw_n,"unique_episodes":epi_n,
                "baseline_net_executable_10s_mean":daily_net.get(day,{}).get("mean"),
                "baseline_net_executable_10s_n":daily_net.get(day,{}).get("n",0),
                "four_tick_resolved":daily_pass.get(day,{}).get("n",0),
                "four_tick_predicted_first":daily_pass.get(day,{}).get("success",0),
                "four_tick_predicted_share":daily_pass.get(day,{}).get("success",0)/daily_pass[day]["n"] if day in daily_pass else None}
    holm=_holm(primary_ps); predictive_holm=_holm(predictive_ps)
    _atomic_json(base/"trigger-duplication-diagnostics.json",{"grouping_rules":episode_rules,"mechanisms":diagnostics})
    _atomic_json(base/"source-censoring-diagnostics.json",source_censor)
    _atomic_json(base/"four-tick-first-passage-summary.json",passage)
    _atomic_json(base/"executable-markout-cost-sensitivity.json",exec_cost)
    _atomic_json(base/"raw-trigger-outcome-summary.json",raw_outcome_summary)
    _atomic_json(base/"date-clustered-uncertainty.json",{"unit":"trading date","bootstrap_replicates":BOOTSTRAPS,
        "seed":SEED,"mechanisms":inferential,"holm_adjusted_economic_p_values":holm,
        "holm_adjusted_predictive_p_values":predictive_holm,"unadjusted_predictive_signflip_p_values":predictive_ps,
        "multiple_testing":"Separate Holm adjustments across the four mechanism economic endpoints and four primary four-tick directional endpoints; other horizons/barriers descriptive."})
    _atomic_json(base/"per-date-scorecard.json",per_date)
    controls={"status":"NOT_IDENTIFIABLE","mechanisms":{m:{"status":"NOT_IDENTIFIABLE",
        "reason":"Frozen protocol requires same-time causal-direction controls and names covariates, but contains no operational control-sampling algorithm, calipers/standardization model, common-support rule, or observation-age cutoff. No post-hoc control definition was introduced."} for m in MECHANISMS}}
    _atomic_json(base/"mechanism-versus-control.json",controls)
    interaction_registry=_read_json(study/"interaction-hypothesis-registry.json")
    interactions={"status":"NOT_IDENTIFIABLE_FOR_INCREMENTAL_EFFECT",
        "reason":"The frozen registry provides hypothesis labels only; it does not specify executable feature formulas, interaction estimators, or multiple-testing family beyond names. Incremental effects cannot be estimated without adding post-hoc definitions.",
        "registered_hypotheses":[{"id":h["id"],"definition":h["definition"],"status":"NOT_IDENTIFIABLE"} for h in interaction_registry["hypotheses"]],
        "descriptive_cross_mechanism_overlap":overlap}
    _atomic_json(base/"registered-interaction-results.json",interactions)
    total_records=sum(int(x["source_records_processed"]) for x in all_source)
    total_triggers={m:len(raw_by_mech[m]) for m in MECHANISMS}
    total_unique={m:sum(x["is_episode_representative"] for x in episode_by_mech[m]) for m in MECHANISMS}
    gate={"construction_dates_required":20,"construction_dates_completed":20,
        "minimum_construction_dates_gate":"PASS" if 20>=int(protocol["progression_gates"]["minimum_construction_dates"]) else "FAIL",
        "minimum_construction_episodes":int(protocol["progression_gates"]["minimum_construction_episodes"]),
        "unique_episodes_by_mechanism":total_unique,
        "episode_support_gate":{m:("PASS" if total_unique[m]>=int(protocol["progression_gates"]["minimum_construction_episodes"]) else "FAIL") for m in MECHANISMS},
        "source_validity_gate":"PASS" if all(diagnostics[m]["source_valid_share"]==1.0 for m in MECHANISMS) else "REVIEW",
        "positive_incremental_directional_effect":"NOT_IDENTIFIABLE",
        "primary_economic_endpoint_nonnegative_or_alternative_predeclared":{m:("PASS" if exec_cost[m]["baseline"]["mean_ticks"] is not None and exec_cost[m]["baseline"]["mean_ticks"]>=0 else "FAIL") for m in MECHANISMS},
        "no_best_date_dominance":"NOT_IDENTIFIABLE: protocol gives no numerical definition of dominance; descriptive date concentration is reported.",
        "controls_and_incremental_effect_gate":"NOT_IDENTIFIABLE",
        "interaction_gate":"NOT_IDENTIFIABLE",
        "selected_mechanism":"NONE","internal_assessment_ready":False,
        "decision":"NO_GO","decision_reason":"At least one frozen progression gate fails or is not identifiable; construction evidence does not justify selecting a mechanism for the 10-date internal assessment."}
    _atomic_json(base/"construction-gate-results.json",gate)
    aggregate={"status":"PASS_20_DATE_CONSTRUCTION_EXECUTED_WITH_UNIDENTIFIABLE_CONTROLS_INTERACTIONS",
        "dates":dates,"source_records_processed":total_records,"trigger_counts":total_triggers,
        "unique_episode_counts":total_unique,"source_censored_counts":{m:source_censor[m]["source_censored_outcomes"] for m in MECHANISMS},
        "baseline_net_executable_10s_mean_ticks":{m:exec_cost[m]["baseline"]["mean_ticks"] for m in MECHANISMS},
        "four_tick_predicted_share_resolved":{m:passage[m]["primary_four_tick_predicted_share_resolved"] for m in MECHANISMS},
        "checkpoint_equivalence_pass_count":sum(x.get("status")=="PASS" and x.get("book_state_equivalent") and x.get("detector_state_equivalent") for x in equivalences),
        "run_runtime_seconds":sum(runtimes),"controls":"NOT_IDENTIFIABLE","registered_interactions":"NOT_IDENTIFIABLE",
        "decision":"NO_GO","protocol_sha256":_sha(study/"frozen-study-protocol.json"),"engine_sha256":progress["engine_sha256"]}
    _atomic_json(base/"construction-analysis-summary.json",aggregate)
    source_summary={"dates":len(all_source),"source_records_processed":total_records,
        "source_gap_intervals":sum(int(x["source_gap_intervals_count"]) for x in all_source),
        "book_validity":{k:sum(int(x["book_validity"].get(k,0)) for x in all_source)
                         for k in all_source[0]["book_validity"]},
        "detector_source_validity":{k:sum(int(x["detector_source_validity"].get(k,0)) for x in all_source)
                                     for k in all_source[0]["detector_source_validity"]},
        "transition_counts":{m:dict(sum((Counter(x["transition_counts"].get(m,{})) for x in all_source),Counter())) for m in MECHANISMS}}
    _atomic_json(base/"source-validity-summary.json",source_summary)
    _atomic_json(base/"native-depth-checkpoint-equivalence.json",{
        "status":"PASS" if len(equivalences)==20 and all(x.get("status")=="PASS" and x.get("book_state_equivalent") and x.get("detector_state_equivalent") for x in equivalences) else "FAIL",
        "date_count":len(equivalences),"bounded_records_per_date":10000,"dates":equivalences,
        "synthetic_semantics_coverage":["depth-only updates","price migration","bid/ask quantity","snapshot/reset","invalid states","timestamp ties","unknown aggressor","checkpoint restoration"]})
    with (base/"comparison-scorecard.csv").open("w",newline="") as f:
        columns=["mechanism","raw_triggers","unique_episodes","repeated_triggers","source_censored_raw",
            "resolved_four_tick_n","predicted_first","opposite_first","predicted_share_resolved",
            "baseline_net_exec10_mean_ticks","moderate_net_exec10_mean_ticks","severe_net_exec10_mean_ticks",
            "date_cluster_ci95_low","date_cluster_ci95_high","control_status","interaction_status"]
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for m in MECHANISMS:
            b=passage[m]["barriers_ticks"]["4"]; ci=inferential[m]["net_executable_10s_date_cluster_bootstrap"]["ci95"]
            writer.writerow({"mechanism":m,"raw_triggers":diagnostics[m]["raw_triggers"],"unique_episodes":total_unique[m],
                "repeated_triggers":diagnostics[m]["repeated_triggers_within_episode"],
                "source_censored_raw":source_censor[m]["source_censored_outcomes"],
                "resolved_four_tick_n":passage[m]["primary_four_tick_resolved_n"],
                "predicted_first":b["PREDICTED_FIRST"],"opposite_first":b["OPPOSITE_FIRST"],
                "predicted_share_resolved":passage[m]["primary_four_tick_predicted_share_resolved"],
                "baseline_net_exec10_mean_ticks":exec_cost[m]["baseline"]["mean_ticks"],
                "moderate_net_exec10_mean_ticks":exec_cost[m]["moderate"]["mean_ticks"],
                "severe_net_exec10_mean_ticks":exec_cost[m]["severe"]["mean_ticks"],
                "date_cluster_ci95_low":ci[0] if ci else None,"date_cluster_ci95_high":ci[1] if ci else None,
                "control_status":"NOT_IDENTIFIABLE","interaction_status":"NOT_IDENTIFIABLE"})
    report=["# Four-Mechanism 20-Date Construction Comparison","",
        "Status: 20-date detector/outcome construction executed; matched controls and registered incremental interactions are not identifiable from the frozen operational definitions.","",
        f"Dates: {len(dates)}. Source records processed: {total_records:,}. Protocol SHA-256: `{aggregate['protocol_sha256']}`.","",
        "| Mechanism | Raw triggers | Unique episodes | Source-censored | 4-tick predicted first / resolved | Baseline net exec 10s (ticks) | Moderate | Severe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MECHANISMS:
        b=passage[m]["barriers_ticks"]["4"]
        report.append(f"| {m} | {diagnostics[m]['raw_triggers']} | {total_unique[m]} | {source_censor[m]['source_censored_outcomes']} | {b['PREDICTED_FIRST']} / {passage[m]['primary_four_tick_resolved_n']} | {exec_cost[m]['baseline']['mean_ticks']} | {exec_cost[m]['moderate']['mean_ticks']} | {exec_cost[m]['severe']['mean_ticks']} |")
    report.extend(["","Controls: `NOT_IDENTIFIABLE`. The sealed protocol gives covariate names but no control sampling algorithm, calipers/model, common-support rule, or observation-age cutoff.",
        "","Interactions: incremental effects `NOT_IDENTIFIABLE`. The frozen registry contains labels, not executable feature formulas or estimators; no post-hoc definitions were added.",
        "","Construction gate: `NO_GO`; no mechanism selected and internal assessment not authorized."])
    (base/"report.md").write_text("\n".join(report)+"\n")
    # Required artifact hash manifest for this analysis namespace.
    hashes={}
    for p in sorted(base.iterdir()):
        if p.is_file() and p.name!="analysis-manifest.json":hashes[p.name]={"sha256":_sha(p),"size_bytes":p.stat().st_size}
    _atomic_json(base/"analysis-manifest.json",{"status":"PASS","protocol_sha256":aggregate["protocol_sha256"],
        "engine_sha256":aggregate["engine_sha256"],"date_count":20,"trigger_counts":total_triggers,
        "outcome_rows":len(all_outcomes),"artifacts":hashes})
    return aggregate


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--run-root",type=Path,required=True)
    a=p.parse_args(argv)
    result=analyze(a.repo.resolve(),(a.repo.resolve()/a.run_root).resolve() if not a.run_root.is_absolute() else a.run_root)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
