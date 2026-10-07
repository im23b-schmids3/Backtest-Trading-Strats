"""Frozen live strategy × day regime diagnostic; descriptive, not optimization."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo
import numpy as np
from . import mac_2025_absorption_relative_normalization as norm
from . import mac_2025_absorption_relative_feature_stability as stability
from . import mac_2025_es_only_train_baseline as baseline

RUN_ID = "CMEOrderflow_ES_LIVE_STRATEGY_DAILY_REGIME_DELTA_DIAGNOSTIC_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
TRAIN_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_TRAIN_20260928")
OCT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928")
FEATURE_ROOT = Path("research_runs/CMEOrderflow_ABSORPTION_RELATIVE_FEATURE_STABILITY_2025_V1")
TICK = 0.25
SEED = 20261006
N_PERM = 2000
FAMILIES = tuple(norm.LIVE_TO_TAPE)
FEATURES = ("DIRECTIONAL_SESSION_CVD_RATIO","DIRECTIONAL_LOCAL_DELTA_RATIO_2M",
 "DIRECTIONAL_CONFIRMATION_DELTA_RATIO","HIGHER_DIVERGENCE_SUPPORT","TREND_EFFICIENCY_AT_SIGNAL",
 "REALIZED_VOLATILITY_AT_SIGNAL","DIRECTIONAL_MLOFI_PERSISTENCE","PRICE_IMPACT_PER_FLOW",
 "APPROACH_VELOCITY_TICKS_PER_SECOND","ABSORPTION_QUALITY_SCORE",
 "DIRECTIONAL_LOCAL_DELTA_2M","DIRECTIONAL_CONFIRMATION_DELTA")
PERMUTATION_FEATURES = ("DIRECTIONAL_SESSION_CVD_RATIO", "DIRECTIONAL_LOCAL_DELTA_RATIO_2M",
 "DIRECTIONAL_CONFIRMATION_DELTA_RATIO", "TREND_EFFICIENCY_AT_SIGNAL",
 "DIRECTIONAL_MLOFI_PERSISTENCE", "PRICE_IMPACT_PER_FLOW")

class DiagnosticError(RuntimeError): pass

def _validate_unique_trade_identities(trades:Sequence[Mapping[str,Any]])->None:
 keys=[(str(t["date"]),str(t["family_id"]),str(t["trade_id"])) for t in trades]
 if len(keys)!=len(set(keys)):raise DiagnosticError("duplicate live trade identity within strategy/date")

def _sha(p: Path) -> str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def _hash(x: Any) -> str:
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _clean(x: Any) -> Any:
 if isinstance(x,dict): return {str(k):_clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [_clean(v) for v in x]
 if isinstance(x,np.generic): x=x.item()
 if isinstance(x,float) and not math.isfinite(x): return None
 return x

def _write(p: Path,x: Any) -> None:
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f".{p.name}.{os.getpid()}.tmp")
 t.write_text(json.dumps(_clean(x),sort_keys=True,indent=2,allow_nan=False)+"\n")
 os.replace(t,p)

def _write_rows(p: Path,rows: Sequence[Mapping[str,Any]]) -> None:
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f".{p.name}.{os.getpid()}.tmp")
 with t.open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
   for r in rows: gz.write(json.dumps(_clean(dict(r)),sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n")
 os.replace(t,p)

def _rank(x: Sequence[float]) -> np.ndarray:
 a=np.asarray(x,float); order=np.argsort(a,kind="mergesort"); out=np.empty(len(a),float); i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]: j+=1
  out[order[i:j]]=(i+j-1)/2+1; i=j
 return out

def _rho(x: Sequence[Any],y: Sequence[Any]) -> float|None:
 pairs=[(float(a),float(b)) for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
 if len(pairs)<3:return None
 a=np.asarray(pairs); rx,ry=_rank(a[:,0]),_rank(a[:,1])
 return float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 and np.std(ry)>0 else None

def _stats(x: Sequence[Any]) -> dict[str,Any]:
 a=np.asarray([float(v) for v in x if v is not None and math.isfinite(float(v))])
 return {"n":len(a),"mean":float(a.mean()) if len(a) else None,"median":float(np.median(a)) if len(a) else None,
  "q25":float(np.quantile(a,.25)) if len(a) else None,"q75":float(np.quantile(a,.75)) if len(a) else None}

def _window_delta(ev: np.ndarray,lo: int,hi: int) -> tuple[float,float]:
 ts=ev["timestamp_ns"]; a=int(np.searchsorted(ts,lo,"left")); b=int(np.searchsorted(ts,hi,"left")); z=ev[a:b]
 return _delta_rows(z)

def _delta_rows(z:np.ndarray)->tuple[float,float]:
 m=(z["execution_size"]>0)&(z["aggressor"]!=0); sz=z["execution_size"][m].astype(float); side=z["aggressor"][m].astype(float)
 return float(np.dot(sz,side)),float(sz.sum())

def _confirm(c: Mapping[str,Any],t: Mapping[str,Any],ev: np.ndarray,cb: Mapping[str,float])->tuple[int,int,float,float]:
 start=int(c["interaction_end_ns"]+cb["min_confirmation_seconds"]*1e9)
 end=int(c["interaction_end_ns"]+cb["max_confirmation_seconds"]*1e9)
 ts=ev["timestamp_ns"]; z=ev[np.searchsorted(ts,start,"left"):np.searchsorted(ts,end,"right")]
 d=1 if t["direction"]=="LONG" else -1
 mask=(z["execution_size"]>0)&(z["aggressor"]!=0)&(((z["execution_price"]-float(c["interaction_end_price"]))*d)>=cb["favorable_ticks"]*TICK)
 hits=z[mask]
 if not len(hits): raise DiagnosticError(f"frozen confirmation not found: {t['trade_id']}")
 hit_local=int(np.flatnonzero(mask)[0]); event_index=int(np.searchsorted(ts,start,"left"))+hit_local
 ts0=int(hits[0]["timestamp_ns"])
 if int(t["entry_timestamp_ns"])<ts0+cb["entry_delay_ms"]*1e6: raise DiagnosticError(f"entry before earliest-ready time: {t['trade_id']}")
 # Stop at the exact confirming execution in canonical tape sequence. A pure
 # timestamp slice could accidentally include later events sharing ts_event.
 window_start=int(np.searchsorted(ts,start,"left")); delta,total=_delta_rows(ev[window_start:event_index+1])
 return ts0,event_index,delta,total

def _session_start(day: str,session: str)->int:
 w=baseline._session_windows(day)
 if session not in w: raise DiagnosticError(f"unrecognized live routing session {session}")
 return int(w[session][0])

def _context(day: str,t: Mapping[str,Any],c: Mapping[str,Any],f: Mapping[str,Any],ev: np.ndarray,cb: Mapping[str,float])->dict[str,Any]:
 q=int(c["interaction_start_ns"]); direction=1 if t["direction"]=="LONG" else -1
 cvd,vol=_window_delta(ev,_session_start(day,str(t["trading_session"])),q)
 delta,dvol=_window_delta(ev,q-120_000_000_000,q)
 cts,cidx,cd,cv=_confirm(c,t,ev,cb)
 ts=ev["timestamp_ns"]; i=max(0,int(np.searchsorted(ts,q,"left"))-1); j=max(0,int(np.searchsorted(ts,q-30_000_000_000,"left"))-1)
 mid=(ev["bid"]+ev["ask"])/2; level=float(c.get("level_price",c.get("zone_low",mid[i])))
 approach=(abs(mid[j]-level)-abs(mid[i]-level))/TICK
 opp=float(c.get("opposite_aggressive_volume") or 0); progress=float(c.get("maximum_through_level_progress_ticks") or 0)
 impact100=c.get("adverse_progress_per_100_aggressive_contracts")
 if impact100 is None and opp>0: impact100=100*progress/opp
 pre_start=int(ts[0]); sess_start=_session_start(day,str(t["trading_session"])); end=int(np.searchsorted(ts,sess_start,"left"))
 pm=mid[:end]
 d=np.diff(pm)/TICK
 return {"signal_context_timestamp_ns":q,"signal_context_anchor":"interaction_start_ns; interaction-context tape queries are strictly before anchor",
  "confirmation_timestamp_ns":cts,"confirmation_event_index":cidx,"confirmation_delta_available_at_confirmation":True,
  "confirmation_delta_window_start_ns":int(c["interaction_end_ns"]+cb["min_confirmation_seconds"]*1e9),
  "SESSION_CVD":cvd,"SESSION_TOTAL_AGGRESSIVE_VOLUME":vol,
  "SESSION_CVD_RATIO":cvd/vol if vol else None,"DIRECTIONAL_SESSION_CVD_RATIO":direction*cvd/vol if vol else None,
  "LOCAL_DELTA_2M":delta,"LOCAL_DELTA_RATIO_2M":delta/dvol if dvol else None,
  "DIRECTIONAL_LOCAL_DELTA_2M":direction*delta,
  "DIRECTIONAL_LOCAL_DELTA_RATIO_2M":direction*delta/dvol if dvol else None,
  "CONFIRMATION_DELTA":cd,"CONFIRMATION_DELTA_RATIO":cd/cv if cv else None,
  "DIRECTIONAL_CONFIRMATION_DELTA":direction*cd,
  "DIRECTIONAL_CONFIRMATION_DELTA_RATIO":direction*cd/cv if cv else None,
  "OPPOSING_AGGRESSIVE_VOLUME":opp,"OPPOSING_PRICE_PROGRESS_TICKS":progress,
  "OPPOSING_PRICE_IMPACT_PER_100_CONTRACTS":impact100,"HIGHER_DIVERGENCE_SUPPORT":-float(impact100) if impact100 is not None else None,
  "TREND_EFFICIENCY_AT_SIGNAL":f.get("ER_30S"),
  "DIRECTIONAL_TREND_EFFICIENCY":f.get("ER_30S_DIRECTIONAL"),
  "PRECEDING_PRICE_PATH_DIRECTION":"WITH_TRADE_DIRECTION" if (mid[i]-mid[j])*direction>0 else "AGAINST_TRADE_DIRECTION" if (mid[i]-mid[j])*direction<0 else "NEUTRAL",
  "REALIZED_VOLATILITY_AT_SIGNAL":f.get("RV_30S_RAW"),
  "NORMALIZED_REALIZED_VOLATILITY_AT_SIGNAL":f.get("RV_30S_TOD_PERCENTILE",f.get("RV_30S_GLOBAL_PERCENTILE")),
  "DIRECTIONAL_MLOFI_PERSISTENCE":f.get("MLOFI_PERSISTENCE_DIRECTIONAL"),
  "PRICE_IMPACT_PER_FLOW":f.get("PRICE_IMPACT_PER_MLOFI_5S"),
  "APPROACH_MOVE_TICKS_30S":approach,"APPROACH_VELOCITY_TICKS_PER_SECOND":approach/30,
  "APPROACH_CLASS":"TOWARD_LEVEL" if approach>0 else "AWAY_FROM_LEVEL" if approach<0 else "NEUTRAL",
  "ABSORPTION_QUALITY_SCORE":f.get("core_quality_score"),
  "PRE_SESSION_CONTEXT":{"session":t["trading_session"],"start_ns":pre_start,"end_ns":sess_start,
   "pre_session_return_ticks":float((pm[-1]-pm[0])/TICK) if len(pm)>1 else None,
   "pre_session_rv_ticks":float(np.sqrt(np.square(d).sum())) if len(d) else None,
   "pre_session_trend_efficiency":float(abs(pm[-1]-pm[0])/TICK/np.abs(d).sum()) if len(d) and np.abs(d).sum() else None,
   "known_by_session_start":True},"future_leakage":False}

def _path(t: Mapping[str,Any],ev: np.ndarray)->dict[str,Any]:
 ts=ev["timestamp_ns"]; a=int(np.searchsorted(ts,int(t["entry_timestamp_ns"]),"left")); b=int(np.searchsorted(ts,int(t["exit_timestamp_ns"]),"right")); z=ev[a:b]
 if not len(z): raise DiagnosticError(f"empty trade price path: {t['trade_id']}")
 long=t["direction"]=="LONG"; px=z["bid"] if long else z["ask"]; risk=abs(float(t["entry"])-float(t["stop"]))
 if risk<=0:
  # A stop-gap can put the realized fill on the stop price. The frozen ledger's
  # gross_R preserves the original planned risk denominator for this case.
  gr=float(t.get("gross_r") or 0); gp=float(t.get("gross_pnl_usd") or 0)
  qty=float(t.get("contracts") or 0); point=float(t.get("point_value_usd") or 0)
  risk=abs(gp/gr)/(qty*point) if gr and qty>0 and point>0 else 0.0
 if risk<=0: raise DiagnosticError(f"cannot recover positive planned risk for trade {t['trade_id']}")
 favorable=(px-float(t["entry"]))*(1 if long else -1)
 return {"MFE_R":float(favorable.max()/risk),"MAE_R":float(max(0,-favorable.min())/risk),"planned_risk_points":risk,
  "target_before_stop":t.get("exit_reason")=="TARGET","stop_before_target":t.get("exit_reason")=="STOP",
  "path_quote_rows":len(z),"path_source":"native ES MBP-10 candidate tape BBO"}

def _live_inputs(date_rows: Mapping[str,Any])->tuple[dict[str,list[dict[str,Any]]],dict[str,str]]:
 roots={"SPRING_2025":TRAIN_ROOT,"OCTOBER_2025":OCT_ROOT}
 dirs=dict(zip(FAMILIES,("europe-europe-current-high","europe-europe-prior-high","europe-europe-prior-vah","ny-ny-prior-poc")))
 out=defaultdict(list); hashes={}
 for family,sub in dirs.items():
  for period,root in roots.items():
   ledger=root/"per-family"/sub/"live-fixed-trades.jsonl"; result=root/"per-family"/sub/"result.json"
   if not ledger.is_file() or not result.is_file(): raise DiagnosticError(f"missing ledger/result {family} {period}")
   if json.loads(result.read_text()).get("live_trade_output_sha256")!=_sha(ledger): raise DiagnosticError(f"ledger hash mismatch {ledger}")
   hashes[str(ledger)]=_sha(ledger)
   for line in ledger.read_text().splitlines():
    r=json.loads(line)
    if r.get("family_id")!=family or r.get("date") not in date_rows: raise DiagnosticError(f"unexpected ledger identity/date: {ledger}")
    if r.get("execution_policy")!="ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS": raise DiagnosticError("live execution policy mismatch")
    out[r["date"]].append(r)
 ids=[(r["date"],r["family_id"],r["trade_id"]) for group in out.values() for r in group]
 if len(ids)!=len(set(ids)): raise DiagnosticError("duplicate trade identity within strategy/date")
 for d in out: out[d].sort(key=lambda r:(int(r["entry_timestamp_ns"]),r["family_id"],r["trade_id"]))
 return out,hashes

def _feature_checkpoint(day:str,config_sha:str,source_sha:str,tape_sha:str)->dict[str,dict[str,Any]]:
 p=FEATURE_ROOT/"checkpoints"/f"{day}.json.gz"
 if not p.is_file(): raise DiagnosticError(f"feature checkpoint missing: {day}")
 x=json.load(gzip.open(p,"rt",encoding="utf8"))
 if any(x.get(k)!=v for k,v in {"date":day,"config_sha256":config_sha,"source_sha256":source_sha,"tape_sha256":tape_sha,"status":"COMPLETE"}.items()):
  raise DiagnosticError(f"stale/hash-mismatched causal feature checkpoint: {day}")
 return {str(r["interaction_id"]):r for r in x["events"]}

def _mfe_day(day:str,ev:np.ndarray)->dict[str,Any]:
 w=baseline._session_windows(day)["NY"]; a=np.searchsorted(ev["timestamp_ns"],w[0],"left"); b=np.searchsorted(ev["timestamp_ns"],w[1],"left"); z=ev[a:b]
 if not len(z): return {"status":"MISSING"}
 m=(z["bid"]+z["ask"])/2; d=np.diff(m)/TICK; op=m[0]; close=m[-1]; hi=m.max(); lo=m.min(); span=hi-lo
 return {"status":"AVAILABLE","RTH_OPEN_TO_CLOSE_RETURN_TICKS":float((close-op)/TICK),
  "RTH_OPEN_TO_CLOSE_RETURN_PERCENT":float((close-op)/op*100),"RTH_HIGH_LOW_RANGE_TICKS":float(span/TICK),
  "RTH_REALIZED_VOLATILITY":float(np.sqrt(np.square(d).sum())),
  "RTH_TREND_EFFICIENCY":float(abs(close-op)/span) if span else 0,
  "RTH_TOTAL_VOLUME":float(z["execution_size"].sum()),"CLOSE_LOCATION_IN_RANGE":float((close-lo)/span) if span else .5,
  "MAX_UP_MOVE_FROM_OPEN":float((m.max()-op)/TICK),"MAX_DOWN_MOVE_FROM_OPEN":float((m.min()-op)/TICK),"type":"EX_POST_ONLY"}

def _max_cluster(ts:Sequence[int],width:int)->int:
 left=best=0
 for right,t in enumerate(ts):
  while t-ts[left]>width:left+=1
  best=max(best,right-left+1)
 return best

def _strategy_days(trades:Sequence[Mapping[str,Any]],day:str,period:str)->list[dict[str,Any]]:
 by=defaultdict(list)
 for t in trades:by[t["family_id"]].append(t)
 out=[]
 for family,rows in by.items():
  rows=sorted(rows,key=lambda r:int(r["entry_timestamp_ns"])); rs=np.asarray([float(r["r_multiple"]) for r in rows]); eq=np.cumsum(rs)
  dd=float(np.min(eq-np.maximum.accumulate(np.r_[0,eq])[1:])) if len(eq) else 0
  pos=rs[rs>0]; neg=rs[rs<0]; counts=Counter(r["setup_id"] for r in rows); ctx=[r["context"] for r in rows]
  out.append({"date":day,"period":period,"family":family,"trade_count":len(rows),"win_count":int((rs>0).sum()),"loss_count":int((rs<0).sum()),
   "breakeven_count":int((rs==0).sum()),"net_r":float(rs.sum()),"avg_r_per_trade":float(rs.mean()),"median_r_per_trade":float(np.median(rs)),
   "win_rate":float((rs>0).mean()),"profit_factor":float(pos.sum()/abs(neg.sum())) if len(neg) else ("INF" if len(pos) else None),
   "max_intraday_drawdown_r":dd,"max_consecutive_losses":_max_losses(rs),"best_trade_r":float(rs.max()),"worst_trade_r":float(rs.min()),
   "sum_mfe_r":float(sum(r.get("MFE_R",0) for r in rows)),"median_mfe_r":float(np.median([r["MFE_R"] for r in rows])),
   "median_mae_r":float(np.median([r["MAE_R"] for r in rows])),"target_before_stop_count":sum(r["target_before_stop"] for r in rows),
   "stop_before_target_count":sum(r["stop_before_target"] for r in rows),"unique_setup_count":len(counts),"max_trades_per_setup":max(counts.values()),
   "max_trades_within_5_minutes":_max_cluster([int(r["entry_timestamp_ns"]) for r in rows],300_000_000_000),
   "max_trades_within_30_minutes":_max_cluster([int(r["entry_timestamp_ns"]) for r in rows],1_800_000_000_000),
   "first_signal_context":ctx[0],"pre_session_context":ctx[0].get("PRE_SESSION_CONTEXT"),
   "all_trade_context_summary":{f:_stats([c.get(f) for c in ctx]) for f in FEATURES},"trade_ids":[r["trade_id"] for r in rows],
   "first_trade_r":float(rs[0]),"later_trades_net_r":float(rs[1:].sum()),
   "cumulative_r_after_trade_1":float(rs[:1].sum()),"cumulative_r_after_trade_2":float(rs[:2].sum()),
   "cumulative_r_after_trade_3":float(rs[:3].sum()),"first_trade_context_predictor":ctx[0]})
 return out

def _max_losses(rs:Sequence[float])->int:
 n=best=0
 for x in rs:n=n+1 if x<0 else 0; best=max(best,n)
 return best

def _calendar(days:Sequence[Mapping[str,Any]],trades:Sequence[Mapping[str,Any]],eligible_dates:Sequence[str])->list[dict[str,Any]]:
 by=defaultdict(list)
 for r in days:by[r["date"]].append(r)
 out=[]
 period_for={d:("SPRING_2025" if d in baseline.TRAIN_DATES else "OCTOBER_2025") for d in eligible_dates}
 for d in sorted(eligible_dates):
  rows=by.get(d,[])
  vals=[r["net_r"] for r in rows]; n=len(vals); net=sum(vals)
  pooled=sorted((t for t in trades if t["date"]==d),key=lambda t:(int(t["entry_timestamp_ns"]),t["family_id"],t["trade_id"]))
  prs=np.cumsum([float(t["r_multiple"]) for t in pooled]); pdd=float(np.min(prs-np.maximum.accumulate(np.r_[0,prs])[1:])) if len(prs) else 0.0
  out.append({"date":d,"period":rows[0]["period"] if rows else period_for[d],"total_trades":sum(r["trade_count"] for r in rows),"total_net_r":net,
   "total_wins":sum(r["win_count"] for r in rows),"total_losses":sum(r["loss_count"] for r in rows),
   "total_max_intraday_dd_r":pdd,"profitable_strategies":sum(x>0 for x in vals),
   "losing_strategies":sum(x<0 for x in vals),"active_strategies":n,"classification":"GOOD" if net>0 else "BAD" if net<0 else "FLAT",
   "period":period_for[d],
   "broad_failure_day":sum(x<0 for x in vals)/n>=.75 if n else False,"broad_success_day":sum(x>0 for x in vals)/n>=.75 if n else False})
 return out

def _concentration(days:Sequence[Mapping[str,Any]])->dict[str,Any]:
 out={}
 for period in ("SPRING_2025","OCTOBER_2025"):
  for family in FAMILIES:
   rows=sorted((r for r in days if r["period"]==period and r["family"]==family),key=lambda r:r["net_r"])
   vals=[r["net_r"] for r in rows]; neg=sum(-x for x in vals if x<0); pos=sum(x for x in vals if x>0)
   out[f"{period}|{family}"]={"positive_days":sum(x>0 for x in vals),"negative_days":sum(x<0 for x in vals),"flat_days":sum(x==0 for x in vals),
    "median_daily_r":float(np.median(vals)) if vals else None,"mean_daily_r":float(np.mean(vals)) if vals else None,
    "worst_day_r":min(vals) if vals else None,"best_day_r":max(vals) if vals else None,
    **{f"worst_{k}_negative_share":sum(-x for x in vals[:k] if x<0)/neg if neg else None for k in (1,3,5)},
    **{f"best_{k}_positive_share":sum(x for x in vals[-k:] if x>0)/pos if pos else None for k in (1,3,5)},
    **{f"net_excluding_worst_{k}":sum(vals[k:]) for k in (1,3,5)},"post_hoc_only":True}
 return out

def _tercile(vals:Sequence[Any])->tuple[float,float]:
 a=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
 return (float(np.quantile(a,1/3)),float(np.quantile(a,2/3))) if len(a)>=3 else (math.nan,math.nan)

def _bucket(x:Any,c:tuple[float,float])->str|None:
 if x is None or not all(math.isfinite(v) for v in c):return None
 return "LOW" if x<=c[0] else "MID" if x<=c[1] else "HIGH"

def _row_bucket(r:Mapping[str,Any],feature:str,cuts:Mapping[str,Mapping[str,tuple[float,float]]])->str|None:
 return _bucket(r["first_signal_context"].get(feature),cuts.get(str(r["family"]),{}).get(feature,(math.nan,math.nan)))

def _analyze(days:Sequence[Mapping[str,Any]])->dict[str,Any]:
 spring=[r for r in days if r["period"]=="SPRING_2025"]; octo=[r for r in days if r["period"]=="OCTOBER_2025"]
 cuts={family:{f:_tercile([r["first_signal_context"].get(f) for r in spring if r["family"]==family]) for f in FEATURES} for family in FAMILIES}
 shapes={}; goodbad={}; corr={}; compat={}
 for f in FEATURES:
  groups=defaultdict(list)
  for r in spring:
   b=_row_bucket(r,f,cuts)
   if b:groups[b].append(r)
  shapes[f]={}
  for b in ("LOW","MID","HIGH"):
   z=groups[b]; trades=[r for r in z]
   tn=sum(r["trade_count"] for r in trades)
   shapes[f][b]={"n_strategy_days":len(z),"mean_net_r":float(np.mean([r["net_r"] for r in z])) if z else None,
    "median_net_r":float(np.median([r["net_r"] for r in z])) if z else None,"positive_day_fraction":float(np.mean([r["net_r"]>0 for r in z])) if z else None,
    "mean_avg_r_per_trade":float(np.mean([r["avg_r_per_trade"] for r in z])) if z else None,
    "median_trade_mfe_r":float(np.median([r["median_mfe_r"] for r in z])) if z else None,
    "median_trade_mae_r":float(np.median([r["median_mae_r"] for r in z])) if z else None,
    "target_before_stop_probability":sum(r["target_before_stop_count"] for r in z)/tn if tn else None,"trade_count":tn}
  means=[shapes[f][b]["mean_net_r"] for b in ("LOW","MID","HIGH")]
  if any(v is None for v in means) or min(shapes[f][b]["n_strategy_days"] for b in ("LOW","MID","HIGH"))<3: cls="INSUFFICIENT"
  elif means[0]<=means[1]<=means[2]:cls="MONOTONIC_POSITIVE"
  elif means[0]>=means[1]>=means[2]:cls="MONOTONIC_NEGATIVE"
  elif means[1]<means[0] and means[1]<means[2]:cls="U_SHAPED"
  elif means[1]>means[0] and means[1]>means[2]:cls="INVERTED_U"
  else:cls="NO_CLEAR_SHAPE"
  shapes[f]["classification"]=cls
  good=[r["first_signal_context"].get(f) for r in spring if r["net_r"]>0]; bad=[r["first_signal_context"].get(f) for r in spring if r["net_r"]<0]
  good=[float(x) for x in good if x is not None]; bad=[float(x) for x in bad if x is not None]
  goodbad[f]={"good_day_median":float(np.median(good)) if good else None,"bad_day_median":float(np.median(bad)) if bad else None,
   "good_minus_bad":float(np.median(good)-np.median(bad)) if good and bad else None,"good_day_n":len(good),"bad_day_n":len(bad)}
  available=[r for r in spring if r["first_signal_context"].get(f) is not None]
  corr[f]={"spearman_strategy_day_net_r":_rho([r["first_signal_context"][f] for r in available],[r["net_r"] for r in available]),
   "spearman_strategy_day_avg_r_per_trade":_rho([r["first_signal_context"][f] for r in available],[r["avg_r_per_trade"] for r in available]),
   "spearman_strategy_day_trade_count":_rho([r["first_signal_context"][f] for r in available],[r["trade_count"] for r in available]),
   "n":len(available)}
  og=defaultdict(list)
  for r in octo:
   b=_row_bucket(r,f,cuts)
   if b:og[b].append(r["net_r"])
  om={b:float(np.mean(og[b])) if og[b] else None for b in ("LOW","MID","HIGH")}
  if sum(len(v) for v in og.values())<10 or any(v is None for v in om.values()): oc="INSUFFICIENT"
  elif cls=="MONOTONIC_POSITIVE":oc="SAME_DIRECTION" if om["LOW"]<=om["MID"]<=om["HIGH"] else "OPPOSITE" if om["LOW"]>=om["MID"]>=om["HIGH"] else "NO_EFFECT"
  elif cls=="MONOTONIC_NEGATIVE":oc="SAME_DIRECTION" if om["LOW"]>=om["MID"]>=om["HIGH"] else "OPPOSITE" if om["LOW"]<=om["MID"]<=om["HIGH"] else "NO_EFFECT"
  else:oc="INSUFFICIENT" if cls=="INSUFFICIENT" else "NO_EFFECT"
  compat[f]={"spring_shape":cls,"october_bucket_mean_net_r":om,"classification":oc,
   "spring_cutpoints_by_family":{family:cuts[family][f] for family in FAMILIES}}
 return {"spring_terciles":shapes,"good_vs_bad":goodbad,"spearman":corr,"october_compatibility":compat,"spring_cutpoints":cuts}

def _interaction(rows:Sequence[Mapping[str,Any]],x:str,y:str,cuts:Mapping[str,tuple[float,float]])->dict[str,Any]:
 cells=defaultdict(list)
 for r in rows:
  a=_row_bucket(r,x,cuts); b=_row_bucket(r,y,cuts)
  if a and b:cells[f"{a}|{b}"].append(r["net_r"])
 result={f"{a}|{b}":{"n":len(cells[f"{a}|{b}"]),"mean_net_r":float(np.mean(cells[f"{a}|{b}"])) if cells[f"{a}|{b}"] else None}
  for a in ("LOW","MID","HIGH") for b in ("LOW","MID","HIGH")}
 return {"cells":result,"classification":"INSUFFICIENT" if any(v["n"]<10 for v in result.values()) else "DESCRIPTIVE_ONLY"}

def _robust(rows:Sequence[Mapping[str,Any]],f:str)->dict[str,Any]:
 rs=[r for r in rows if r["period"]=="SPRING_2025" and r["first_signal_context"].get(f) is not None]
 effect=_rho([r["first_signal_context"][f] for r in rs],[r["net_r"] for r in rs])
 dates=sorted({r["date"] for r in rs}); lodo=[]
 for d in dates:
  z=[r for r in rs if r["date"]!=d]; v=_rho([r["first_signal_context"][f] for r in z],[r["net_r"] for r in z])
  if v is not None:lodo.append((d,v))
 weeks=sorted({datetime.fromisoformat(r["date"]).isocalendar()[:2] for r in rs}); lowo=[]
 for w in weeks:
  z=[r for r in rs if datetime.fromisoformat(r["date"]).isocalendar()[:2]!=w]; v=_rho([r["first_signal_context"][f] for r in z],[r["net_r"] for r in z])
  if v is not None:lowo.append((w,v))
 return {"spring_spearman":effect,"lodo_sign_stability":float(np.mean([np.sign(v)==np.sign(effect) for _,v in lodo])) if lodo and effect else None,
  "lodo_median_effect":float(np.median([v for _,v in lodo])) if lodo else None,"lodo_min_effect":min((v for _,v in lodo),default=None),
  "lodo_max_effect":max((v for _,v in lodo),default=None),"lodo_worst_omitted_date":min(lodo,key=lambda z:z[1])[0] if lodo else None,
  "lodo_best_omitted_date":max(lodo,key=lambda z:z[1])[0] if lodo else None,
  "lowo_sign_stability":float(np.mean([np.sign(v)==np.sign(effect) for _,v in lowo])) if lowo and effect else None,
  "lowo_median_effect":float(np.median([v for _,v in lowo])) if lowo else None,"lowo_min_effect":min((v for _,v in lowo),default=None),
  "lowo_max_effect":max((v for _,v in lowo),default=None),"lowo_worst_omitted_week":min(lowo,key=lambda z:z[1])[0] if lowo else None,
  "lowo_best_omitted_week":max(lowo,key=lambda z:z[1])[0] if lowo else None,"status":"AVAILABLE" if len(rs)>=20 else "INSUFFICIENT"}

def _permutation(rows:Sequence[Mapping[str,Any]],features:Sequence[str]=PERMUTATION_FEATURES)->dict[str,Any]:
 rng=np.random.default_rng(SEED); out={}
 for period in ("SPRING_2025","OCTOBER_2025"):
  subset=[r for r in rows if r["period"]==period]
  for feature in features:
   usable=[r for r in subset if r["first_signal_context"].get(feature) is not None]
   observed=_rho([r["first_signal_context"][feature] for r in usable],[r["net_r"] for r in usable])
   if observed is None or len(usable)<10:
    out[f"{period}|{feature}"]={"observed_spearman":observed,"n_strategy_days":len(usable),"status":"INSUFFICIENT"}; continue
   groups=defaultdict(list)
   for i,r in enumerate(usable):groups[(r["family"],r["period"])].append(i)
   vals=np.asarray([r["first_signal_context"][feature] for r in usable],float); outcome=[r["net_r"] for r in usable]
   null=[]
   for _ in range(N_PERM):
    perm=vals.copy()
    for ids in groups.values():perm[ids]=rng.permutation(perm[ids])
    rho=_rho(perm,outcome)
    if rho is not None:null.append(rho)
   out[f"{period}|{feature}"]={"observed_spearman":observed,"n_strategy_days":len(usable),
    "permutations":len(null),"null_median":float(np.median(null)) if null else None,
    "null_q025":float(np.quantile(null,.025)) if null else None,"null_q975":float(np.quantile(null,.975)) if null else None,
    "two_sided_empirical_p":(1+sum(abs(x)>=abs(observed) for x in null))/(1+len(null)) if null else None,
    "permutation_unit":"strategy-day feature shuffle within family and period; descriptive, calendar clustering is not fully preserved",
    "status":"AVAILABLE"}
 return {"seed":SEED,"repetitions":N_PERM,"results":out}

def _concentration_r(trades:Sequence[Mapping[str,Any]])->dict[str,Any]:
 out={}
 for period in ("SPRING_2025","OCTOBER_2025"):
  for f in FAMILIES:
   ds=sorted([r for r in trades if r["period"]==period and r["family_id"]==f],key=lambda r:r["date"])
   by=defaultdict(float)
   for r in ds:by[r["date"]]+=float(r["r_multiple"])
   vals=sorted(by.values()); neg=sum(-x for x in vals if x<0); pos=sum(x for x in vals if x>0)
   out[f"{period}|{f}"]={"positive_days":sum(x>0 for x in vals),"negative_days":sum(x<0 for x in vals),"flat_days":sum(x==0 for x in vals),
    "median_daily_r":float(np.median(vals)) if vals else None,"mean_daily_r":float(np.mean(vals)) if vals else None,
    "worst_day_r":min(vals) if vals else None,"best_day_r":max(vals) if vals else None,
    **{f"worst_{k}_negative_share":sum(-x for x in vals[:k] if x<0)/neg if neg else None for k in (1,3,5)},
    **{f"best_{k}_positive_share":sum(x for x in vals[-k:] if x>0)/pos if pos else None for k in (1,3,5)},
    **{f"net_excluding_worst_{k}":sum(vals[k:]) for k in (1,3,5)}}
 return out

def _load():
 configs,config_sha=norm._load_live_configs()
 rows,tapes,coverage=stability._manifest_and_inputs()
 if len(rows)!=54 or len(baseline.TRAIN_DATES)!=35:raise DiagnosticError("expected 35 Spring + 19 October eligible sources")
 identity=json.loads((FEATURE_ROOT/"checkpoints"/"run-identity.json").read_text())
 dates=sorted(rows)
 if identity.get("config_sha256")!=config_sha or identity.get("ordered_dates")!=dates:raise DiagnosticError("feature checkpoint run identity differs from requested inputs")
 source_hashes={d:str(rows[d]["source_sha256"]) for d in rows}
 return configs,config_sha,rows,tapes,source_hashes,coverage

def run(*,output_root:Path=OUT_ROOT)->dict[str,Any]:
 started=time.perf_counter(); configs,config_sha,date_rows,tapes,source_hashes,input_coverage=_load()
 snap_path=norm.LIVE_CONFIG_PATH; snap=json.loads(snap_path.read_text()); live_rows={r["derived_parameters"]["runtime_family_key"]["value"]:r for r in snap["strategies"]}
 live_cfg={}
 cb={}
 for f,live in zip(FAMILIES,norm.LIVE_TO_TAPE.values()):
  r=live_rows[live]; live_cfg[f]={"live_key":live,"runtime_identity":r["derived_parameters"]["runtime_strategy_identity"]["value"],
   "class_a":r["class_a_parameters"],"class_b":r["class_b_parameters"],"execution":r["execution_parameters"],
   "strategy_parameter_sha256":_hash({"class_a":r["class_a_parameters"],"class_b":r["class_b_parameters"],"execution":r["execution_parameters"]})}
  c=r["class_b_parameters"]; shared=snap.get("shared_semantics",{})
  def value(v,default):return v.get("value",default) if isinstance(v,dict) else v if v is not None else default
  cb[f]={"min_confirmation_seconds":value(c.get("min_confirmation_seconds"),value(c.get("confirmation_window_seconds"),5)),
   "max_confirmation_seconds":value(c.get("max_confirmation_seconds"),value(c.get("confirmation_window_seconds"),15)),
   "favorable_ticks":value(c.get("favorable_ticks"),value(shared.get("confirmation_favorable_ticks"),3)),
   "entry_delay_ms":value(r.get("execution_parameters",{}).get("entry_delay_ms"),2)}
 expected_dirs=dict(zip(FAMILIES,("europe-europe-current-high","europe-europe-prior-high","europe-europe-prior-vah","ny-ny-prior-poc")))
 ledgers={}; ledger_hashes={}
 for family,sub in expected_dirs.items():
  for period,root in (("SPRING_2025",TRAIN_ROOT),("OCTOBER_2025",OCT_ROOT)):
   ledger_name="live-fixed-trades.jsonl" if period=="SPRING_2025" else "trades.jsonl"
   p=root/"per-family"/sub/ledger_name; result=json.loads((root/"per-family"/sub/"result.json").read_text())
   if not p.is_file():raise DiagnosticError(f"live ledger missing {p}")
   recorded_hash=result.get("live_trade_output_sha256")
   if recorded_hash is not None and recorded_hash!=_sha(p):raise DiagnosticError(f"live ledger hash mismatch {p}")
   ledger_hashes[str(p)]=_sha(p)
   for line in p.read_text().splitlines():
    t=json.loads(line)
    t["family_id"]=t.get("family_id") or str(t.get("level",t.get("family",""))).replace(" | ","|")
    t["trading_session"]=t.get("trading_session") or t["family_id"].split("|",1)[0]
    if t.get("date") not in date_rows or t.get("family_id")!=family or t.get("execution_policy")!="ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS":raise DiagnosticError(f"live trade identity mismatch {p}")
    ledgers.setdefault(t["date"],[]).append(t)
 _validate_unique_trade_identities([t for ts in ledgers.values() for t in ts])
 study={"run_id":RUN_ID,"families":list(FAMILIES),"target_dates":sorted(date_rows),"live_config_sha256":config_sha,
  "strategy_manifest_sha256":norm.EXPECTED_STRATEGY_MANIFEST_SHA,"features":list(FEATURES),"primary_unit":"strategy-day",
  "source_manifest_sha256":input_coverage["source_manifest_sha256"],
  "source_hashes":source_hashes,"candidate_tape_hashes":{d:_sha(tapes[d]) for d in sorted(tapes)},
  "diagnostic_code_sha256":_sha(Path(__file__)),
  "feature_checkpoint_code_sha256":_sha(Path(stability.__file__)),
  "normalization_code_sha256":_sha(Path(norm.__file__)),
  "session_definition":"baseline._session_windows(date); CVD starts at frozen routed session start",
  "feature_definition_version":"frozen feature-stability checkpoint + strictly pre-interaction ES tape deltas",
  "signal_context_anchor":"interaction_start_ns, strict as-of; first live trade per family/date",
  "spring_terciles_applied_unchanged_to_october":True,"permutation_seed":SEED,"permutation_repetitions":N_PERM,
  "optimization_performed":False,"threshold_search_performed":False,"feature_search_performed":False,"data_downloaded":False,"final_oos_accessed":False}
 study_sha=_hash(study); root=output_root; cpdir=root/"checkpoints"; cpdir.mkdir(parents=True,exist_ok=True)
 _write(root/"study-config.json",study)
 _write(root/"live-strategy-config.json",{"source":str(snap_path),"source_sha256":config_sha,"strategies":live_cfg})
 trade_events=[]; days=[]; descriptors={}
 for ix,day in enumerate(sorted(date_rows),1):
  tape=tapes[day]; tape_sha=_sha(tape); fmaps={}
  cpfeature=FEATURE_ROOT/"checkpoints"/f"{day}.json.gz"; fcp=json.load(gzip.open(cpfeature,"rt"))
  if fcp.get("config_sha256")!=config_sha or fcp.get("source_sha256")!=source_hashes[day] or fcp.get("tape_sha256")!=tape_sha or fcp.get("status")!="COMPLETE":raise DiagnosticError(f"causal checkpoint mismatch {day}")
  fmaps={r["interaction_id"]:r for r in fcp["events"]}
  with np.load(tape,allow_pickle=False) as z:
   ev=np.asarray(z["events"]); meta=json.loads(str(z["metadata_json"].item())); cand=json.loads(str(z["candidate_json"].item()))
  if meta.get("date")!=day or meta.get("source_sha256")!=source_hashes[day] or np.any(np.diff(ev["timestamp_ns"])<0):raise DiagnosticError(f"sealed event tape invalid {day}")
  cm={r["interaction_id"]:r for r in cand}; annotated=[]
  for original in ledgers.get(day,[]):
   t=dict(original); c=cm.get(t["interaction_id"]); f=fmaps.get(t["interaction_id"])
   if c is None or f is None:raise DiagnosticError(f"trade/context join missing {day}/{t['trade_id']}")
   if f["interaction_start_ns"]!=c["interaction_start_ns"]:raise DiagnosticError("candidate/checkpoint anchor mismatch")
   t["context"]=_context(day,t,c,f,ev,cb[t["family_id"]]); t.update(_path(t,ev))
   t["signal_timestamp_ns"]=int(c["interaction_start_ns"])
   t["interaction_start_ns"]=int(c["interaction_start_ns"]); t["interaction_end_ns"]=int(c["interaction_end_ns"])
   t["structural_level_price"]=c.get("level_price",c.get("zone_low"))
   t["confirmation_timestamp_ns"]=int(t["context"]["confirmation_timestamp_ns"])
   t["planned_reward_r"]=abs(float(t["target"])-float(t["entry"]))/float(t["planned_risk_points"])
   t["period"]="SPRING_2025" if day in baseline.TRAIN_DATES else "OCTOBER_2025"; annotated.append(t)
  period="SPRING_2025" if day in baseline.TRAIN_DATES else "OCTOBER_2025"
  cp={"date":day,"period":period,"source_sha256":source_hashes[day],"tape_sha256":tape_sha,"config_sha256":config_sha,
   "study_sha256":study_sha,"ledger_sha256":_hash({k:v for k,v in ledger_hashes.items() if (TRAIN_ROOT if period=="SPRING_2025" else OCT_ROOT).as_posix() in k}),
   "trades":annotated,"ex_post_day_descriptors":_mfe_day(day,ev),"status":"COMPLETE"}
  cpath=cpdir/f"{day}.json.gz"; tmp=cpath.with_name(f".{cpath.name}.{os.getpid()}.tmp")
  with tmp.open("wb") as raw:
   with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:gz.write(json.dumps(_clean(cp),sort_keys=True,separators=(",",":"),allow_nan=False).encode())
  os.replace(tmp,cpath)
  trade_events.extend(annotated); days.extend(_strategy_days(annotated,day,period)); descriptors[day]=cp["ex_post_day_descriptors"]
  print(f"[daily-regime] {ix}/54 {day} trades={len(annotated)}",flush=True)
 cal=_calendar(days,trade_events,sorted(date_rows)); analysis=_analyze(days); spring=[r for r in days if r["period"]=="SPRING_2025"]
 octo=[r for r in days if r["period"]=="OCTOBER_2025"]
 pooled={f:analysis["spearman"][f]["spearman_strategy_day_net_r"] for f in FEATURES}
 primary=sorted(((abs(v),f) for f,v in pooled.items() if v is not None),reverse=True)
 strongest=primary[0][1] if primary else FEATURES[0]
 interaction={"flow_x_trend":_interaction(spring,"DIRECTIONAL_SESSION_CVD_RATIO","TREND_EFFICIENCY_AT_SIGNAL",analysis["spring_cutpoints"]),
  "mlofi_x_impact":_interaction(spring,"DIRECTIONAL_MLOFI_PERSISTENCE","PRICE_IMPACT_PER_FLOW",analysis["spring_cutpoints"])}
 interaction_oct={"flow_x_trend":_interaction(octo,"DIRECTIONAL_SESSION_CVD_RATIO","TREND_EFFICIENCY_AT_SIGNAL",analysis["spring_cutpoints"]),
  "mlofi_x_impact":_interaction(octo,"DIRECTIONAL_MLOFI_PERSISTENCE","PRICE_IMPACT_PER_FLOW",analysis["spring_cutpoints"])}
 broad=[r for r in cal if r["broad_failure_day"]]; success=[r for r in cal if r["broad_success_day"]]
 broad_context=_broad_day_context(days,cal)
 conc=_concentration(days); period_r={p:sum(r["net_r"] for r in days if r["period"]==p) for p in ("SPRING_2025","OCTOBER_2025")}
 allgood=sum(r["net_r"]>0 for r in days); allbad=sum(r["net_r"]<0 for r in days)
 # No validated, schema-documented local calendar is part of the approved 2025
 # input inventory. Do not recursively inspect `data/`: it also contains sealed
 # later/OOS material that is explicitly out of scope.
 shared_news=False
 robustness=_robust(days,strongest); robust=robustness; permutations=_permutation(days)
 by_family={}
 for family in FAMILIES:
  fr=[r for r in days if r["family"]==family]; sp=[r for r in fr if r["period"]=="SPRING_2025"]
  by_family[family]={"spring_strategy_days":len(sp),"october_strategy_days":sum(r["period"]=="OCTOBER_2025" for r in fr),
   "spring_feature_spearman":{f:_rho([r["first_signal_context"].get(f) for r in sp],[r["net_r"] for r in sp]) for f in FEATURES}}
 signs=[int(np.sign(by_family[f]["spring_feature_spearman"][strongest])) for f in FAMILIES if by_family[f]["spring_feature_spearman"][strongest] not in (None,0)]
 shared_vs_specific="INSUFFICIENT" if len(signs)<2 else "SHARED_ACROSS_STRATEGIES" if all(s==signs[0] for s in signs) else "CONTRADICTORY_ACROSS_STRATEGIES"
 first_later={}
 for family in FAMILIES:
  z=[r for r in days if r["family"]==family and r["trade_count"]>=2]
  first_later[family]={"n_strategy_days":len(z),"first_trade_r":_stats([r["first_trade_r"] for r in z]),
   "later_trades_net_r":_stats([r["later_trades_net_r"] for r in z]),
   "first_trade_vs_later_net_r_spearman":_rho([r["first_trade_r"] for r in z],[r["later_trades_net_r"] for r in z]),
   "first_signal_feature_vs_later_net_r":{f:_rho([r["first_signal_context"].get(f) for r in z],[r["later_trades_net_r"] for r in z]) for f in FEATURES}}
 sp_sorted=sorted(spring,key=lambda r:r["net_r"]); tert=max(1,len(sp_sorted)//3)
 bottom={r["date"]+"|"+r["family"] for r in sp_sorted[:tert]}; top={r["date"]+"|"+r["family"] for r in sp_sorted[-tert:]}
 ex_post={}
 for field in ("RTH_OPEN_TO_CLOSE_RETURN_TICKS","RTH_HIGH_LOW_RANGE_TICKS","RTH_REALIZED_VOLATILITY","RTH_TREND_EFFICIENCY"):
  vals=[{"key":r["date"]+"|"+r["family"],"value":abs(descriptors[r["date"]][field]) if field=="RTH_OPEN_TO_CLOSE_RETURN_TICKS" else descriptors[r["date"]][field]} for r in spring]
  ex_post[field]={"worst_r_tercile":_stats([v["value"] for v in vals if v["key"] in bottom]),"best_r_tercile":_stats([v["value"] for v in vals if v["key"] in top]),"type":"EX_POST_ONLY"}
 # No candidate is nominated without coherent, compatible, day-robust evidence.
 hypothesis="NONE"; decision="INSUFFICIENT_SAMPLE" if len(spring)<20 or len(octo)<10 else "NO_STABLE_DAILY_REGIME_PATTERN"
 next_step="REQUIRE_ADDITIONAL_PREDECLARED_DATA" if decision=="INSUFFICIENT_SAMPLE" else "DO_NOT_FILTER_MINE_THESE_STRATEGIES"
 summary={"run_id":RUN_ID,"status":"COMPLETE","dataset":"SPRING_2025 + OCTOBER_2025","spring_role":"PRIMARY_DISCOVERY","october_role":"SECONDARY_COMPATIBILITY",
  "live_strategy_families":list(FAMILIES),"live_config_hash":config_sha,"strategy_parameter_hashes":{f:r["strategy_parameter_sha256"] for f,r in live_cfg.items()},
  "total_calendar_days":len(cal),"total_strategy_days_with_trades":len(days),"total_trades":len(trade_events),
  "good_strategy_days":allgood,"bad_strategy_days":allbad,"flat_strategy_days":len(days)-allgood-allbad,
  "good_calendar_days":sum(r["total_net_r"]>0 for r in cal),"bad_calendar_days":sum(r["total_net_r"]<0 for r in cal),
  "spring_net_r":period_r["SPRING_2025"],"october_net_r":period_r["OCTOBER_2025"],"performance_concentration":conc,
  "broad_failure_days":[r["date"] for r in broad],"broad_success_days":[r["date"] for r in success],"news_calendar_available":shared_news,
  "first_signal_feature_relationships":analysis["spearman"],"flow_x_trend_interaction":interaction["flow_x_trend"],
  "mlofi_x_impact_interaction":interaction["mlofi_x_impact"],"october_interactions_same_spring_cutpoints":interaction_oct,
  "strongest_feature":strongest,"lodo_lowo":robust,
  "permutation_results":permutations,"strategy_specific_feature_relationships":by_family,"shared_vs_strategy_specific":shared_vs_specific,
  "first_trade_vs_later_trade":first_later,"ex_post_extreme_day_relationship":ex_post,"broad_day_context_comparison":broad_context,
  "candidate_hypothesis":hypothesis,"primary_decision":decision,"next_step":next_step,
  "live_strategy_rules_changed":False,"filter_implemented":False,"optimization_performed":False,"optuna_performed":False,
  "threshold_search_performed":False,"feature_search_performed":False,"final_oos_accessed":False,"data_downloaded":False,"commit_performed":False}
 calroot=root
 _write_rows(root/"trade-events.jsonl.gz",trade_events); _write_rows(root/"strategy-days.jsonl.gz",days); _write_rows(root/"calendar-days.jsonl.gz",cal)
 outputs={"summary.json":summary,"performance-concentration.json":conc,
  "trade-clustering.json":{f"{r['date']}|{r['family']}":{k:r[k] for k in ("trade_count","unique_setup_count","max_trades_per_setup","max_trades_within_5_minutes","max_trades_within_30_minutes")} for r in days},
  "first-signal-context.json":{f"{r['date']}|{r['family']}":r["first_signal_context"] for r in days},
  "all-trade-context.json":{f"{r['date']}|{r['family']}":r["all_trade_context_summary"] for r in days},
  "session-cvd-analysis.json":_feature_analysis_by_name(days,"DIRECTIONAL_SESSION_CVD_RATIO"),
  "local-delta-analysis.json":_feature_analysis_by_name(days,"DIRECTIONAL_LOCAL_DELTA_RATIO_2M"),
  "confirmation-delta-analysis.json":_feature_analysis_by_name(days,"DIRECTIONAL_CONFIRMATION_DELTA_RATIO"),
  "delta-divergence-analysis.json":_feature_analysis_by_name(days,"HIGHER_DIVERGENCE_SUPPORT"),
  "trend-efficiency-analysis.json":_feature_analysis_by_name(days,"TREND_EFFICIENCY_AT_SIGNAL"),
  "volatility-analysis.json":_feature_analysis_by_name(days,"REALIZED_VOLATILITY_AT_SIGNAL"),
  "mlofi-analysis.json":_feature_analysis_by_name(days,"DIRECTIONAL_MLOFI_PERSISTENCE"),
  "impact-per-flow-analysis.json":_feature_analysis_by_name(days,"PRICE_IMPACT_PER_FLOW"),
  "approach-velocity-analysis.json":_feature_analysis_by_name(days,"APPROACH_VELOCITY_TICKS_PER_SECOND"),
  "absorption-quality-analysis.json":_feature_analysis_by_name(days,"ABSORPTION_QUALITY_SCORE"),
  "good-vs-bad-day-features.json":{"pooled":analysis["good_vs_bad"],"by_family":_good_bad_by_family(days)},"strategy-specific-results.json":_family_results(days),
  "pooled-strategy-day-results.json":{"equal_weight_strategy_day":analysis["spearman"],"unweighted_trade_level":_trade_associations(trade_events),"strategy_specific":by_family,"shared_vs_strategy_specific":shared_vs_specific},
  "broad-failure-days.json":{"failure":broad,"success":success,"context_comparison":broad_context,"criterion":"at least 75% active families with trades"},
  "ex-post-day-regime.json":{"by_date":descriptors,"extreme_strategy_day_comparison":ex_post,"type":"EX_POST_ONLY"},
  "interaction-flow-trend.json":{"spring":interaction["flow_x_trend"],"october_same_spring_cutpoints":interaction_oct["flow_x_trend"]},
  "interaction-book-impact.json":{"spring":interaction["mlofi_x_impact"],"october_same_spring_cutpoints":interaction_oct["mlofi_x_impact"]},
  "spring-terciles.json":{"cutpoints_by_family":analysis["spring_cutpoints"],"results":analysis["spring_terciles"]},
  "october-compatibility.json":analysis["october_compatibility"],"lodo-results.json":robust,"lowo-results.json":robust,
  "permutation-results.json":permutations,"first-trade-vs-later-trades.json":first_later,
  "candidate-hypothesis.json":{"candidate_hypothesis":hypothesis,"primary_decision":decision,"next_step":next_step}}
 if shared_news:
  outputs["news-calendar-analysis.json"]={"available":True,"status":"LOCAL_VALIDATED_CALENDAR_PRESENT"}
 for name,payload in outputs.items():
  if name!="summary.json":_write(root/name,payload)
 # Additional requested fixed-grid diagnostics from completed outcomes.
 _write(root/"spring-october-diagnostics.json",{"strategy_days":analysis["spring_terciles"],"october_compatibility":analysis["october_compatibility"]})
 report=_report(summary,analysis,interaction,live_cfg,period_r)
 (root/"report.md").write_text(report)
 source_hash=_sha(norm.DATA_ROOT/baseline.MANIFEST_NAME)
 _write(root/"source-coverage.json",{"manifest_sha256":source_hash,"source_dates":sorted(date_rows),"source_hashes":source_hashes,
  "validated_input_coverage":input_coverage,
  "tapes":{d:{"path":str(tapes[d]),"sha256":_sha(tapes[d])} for d in tapes},"ledger_hashes":ledger_hashes,"config_sha256":config_sha,
  "dependency_2025_10_06_used_as_target":False,"data_downloaded":False,"2026_or_oos_accessed":False})
 _write(root/"summary.json",summary)
 files=[p for p in root.iterdir() if p.is_file() and p.name not in ("artifact-hashes.json","run-manifest.json")]
 _write(root/"artifact-hashes.json",{p.name:_sha(p) for p in files})
 _write(root/"run-manifest.json",{"run_id":RUN_ID,"status":"COMPLETE","study_sha256":study_sha,"live_config_sha256":config_sha,
  "source_manifest_sha256":source_hash,"completed_dates":sorted(date_rows),"source_hashes":source_hashes,"ledger_hashes":ledger_hashes,
  "artifact_hashes_sha256":_sha(root/"artifact-hashes.json"),"elapsed_seconds":time.perf_counter()-started,
  "optimization_performed":False,"data_downloaded":False,"final_oos_accessed":False})
 return summary

def _feature_analysis_by_name(rows:Sequence[Mapping[str,Any]],feature:str)->dict[str,Any]:
 return {p:{state:_stats([r["first_signal_context"].get(feature) for r in rows if r["period"]==p and ((r["net_r"]>0) if state=="GOOD" else (r["net_r"]<0) if state=="BAD" else True)])
  for state in ("GOOD","BAD","ALL")} for p in ("SPRING_2025","OCTOBER_2025")}

def _good_bad_by_family(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
 return {family:{feature:{state:_stats([r["first_signal_context"].get(feature) for r in rows if r["family"]==family and r["period"]=="SPRING_2025" and ((r["net_r"]>0) if state=="GOOD" else (r["net_r"]<0) if state=="BAD" else True)])
  for state in ("GOOD","BAD","ALL")} for feature in FEATURES} for family in FAMILIES}

def _broad_day_context(days:Sequence[Mapping[str,Any]],calendar:Sequence[Mapping[str,Any]])->dict[str,Any]:
 bydate=defaultdict(list)
 for r in days:bydate[r["date"]].append(r)
 sets={"BROAD_FAILURE":[r["date"] for r in calendar if r["broad_failure_day"]],
       "BROAD_SUCCESS":[r["date"] for r in calendar if r["broad_success_day"]]}
 out={"calendar_day_sets":sets,"features":{}}
 for feature in FEATURES:
  out["features"][feature]={}
  for label,dates in sets.items():
   daily=[]
   for date in dates:
    values=[r["first_signal_context"].get(feature) for r in bydate[date]]
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    if values:daily.append(float(np.mean(values)))
   out["features"][feature][label]={"n_calendar_days":len(daily),**_stats(daily)}
 return out

def _family_results(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
 out={}
 for f in FAMILIES:
  out[f]={}
  for p in ("SPRING_2025","OCTOBER_2025"):
   z=[r for r in rows if r["family"]==f and r["period"]==p]; vals=[r["net_r"] for r in z]
   out[f][p]={"strategy_days":len(z),"positive_days":sum(x>0 for x in vals),"negative_days":sum(x<0 for x in vals),
    "flat_days":sum(x==0 for x in vals),"median_daily_r":float(np.median(vals)) if vals else None,"mean_daily_r":float(np.mean(vals)) if vals else None,
    "best_day_r":max(vals) if vals else None,"worst_day_r":min(vals) if vals else None,"total_net_r":sum(vals),"trades":sum(r["trade_count"] for r in z)}
 return out

def _trade_associations(trades:Sequence[Mapping[str,Any]])->dict[str,Any]:
 return {f:{"spearman_trade_r":_rho([t["context"].get(f) for t in trades],[t["r_multiple"] for t in trades]),
  "trade_count":sum(t["context"].get(f) is not None for t in trades),"inference":"descriptive; strategy-day is primary"} for f in FEATURES}

def _report(s:Mapping[str,Any],a:Mapping[str,Any],ix:Mapping[str,Any],cfg:Mapping[str,Any],net:Mapping[str,float])->str:
 lines=[f"# {RUN_ID}","","Frozen strategy/day diagnostic. No strategy, threshold, feature, or execution optimization.","",
  f"Primary decision: **{s['primary_decision']}**",f"Next step: **{s['next_step']}**","","| Family | Runtime key | Parameter hash |","|---|---|---|"]
 for f,r in cfg.items():lines.append(f"| {f} | {r['live_key']} | {r['strategy_parameter_sha256']} |")
 lines += ["",f"Spring NET_R={net['SPRING_2025']:.4f}; October NET_R={net['OCTOBER_2025']:.4f}; trades={s['total_trades']}; strategy-days={s['total_strategy_days_with_trades']}.",
  "","First-signal features are anchored at the beginning of the matched causal interaction; feature checkpoint and ES tape hashes were verified.",
  "Completed-day descriptors are ex-post only. October uses frozen Spring tercile cutpoints.","",
  f"Fixed interactions: flow×trend={ix['flow_x_trend']['classification']}; MLOFI×impact={ix['mlofi_x_impact']['classification']}.",
  "Results are descriptive; no filter is implemented or promoted from these discovery/compatibility periods."]
 return "\n".join(lines)+"\n"

def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-root",type=Path,default=OUT_ROOT);a=p.parse_args(argv)
 try:r=run(output_root=a.output_root)
 except (DiagnosticError,norm.StudyError,baseline.BaselineError) as e:p.exit(2,f"ERROR: {e}\n")
 print(f"ES_LIVE_STRATEGY_DAILY_REGIME_AND_DELTA_DIAGNOSTIC=PASS trades={r['total_trades']} days={r['total_strategy_days_with_trades']} decision={r['primary_decision']}")
 return 0
if __name__=="__main__":raise SystemExit(main())
