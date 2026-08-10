"""Sealed local OOS runner for ``CMEOrderflowAbsorption.ES_V1_BACKTEST``.

This module deliberately has no DBN download path.  It reuses the pilot L3
reconstructor and interaction lifecycle, loads (rather than estimates) the two
frozen p95 literals, and refuses to turn an empty legacy ledger into evidence.
"""
from __future__ import annotations

import argparse, base64, csv, hashlib, json, math, os, shutil, struct, zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .analysis import Diagnostics, RTH_END, RTH_START, day_and_seconds
from .engine import BookStateError, CausalMBOBook

TICK = .25; RAW_TICK = 250_000_000; USD_PER_POINT = 50.; COMMISSION = 3.; RISK_BUDGET = 250.
EXPECTED_RECORDS = 61_106_259
COUNTERS = ("dbn_records_seen", "snapshot_records_seen", "ordinary_records_seen", "book_events_processed",
 "rth_sessions_processed", "structural_level_instances", "raw_interactions", "high_absorption",
 "strong_replenishment", "plus_eligible", "submission_attempts", "entered_trades", "closed_trades",
 "audit_rows", "insufficient_risk_budget", "position_already_open", "cutoff_cancelled", "invalid_spread",
 "latency_not_reached", "other_non_trade_outcomes", "reconstruction_stage_invoked", "interaction_stage_invoked",
 "scoring_stage_invoked", "plus_stage_invoked")
ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/backtest-contract.json"
MANIFEST = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"
CALIBRATION = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/development-score-calibration.json"
RUN_ROOT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V1_BACKTEST/oos_v1"
PREVIOUS_RTH_SOURCES={"2026-08-03":"2026-07-31","2026-08-04":"2026-08-03","2026-08-05":"2026-08-04","2026-08-06":"2026-08-05","2026-08-07":"2026-08-06"}

class SealedRunError(RuntimeError): pass

def canonical(v: Any) -> bytes: return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
def digest(path: Path) -> str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1_048_576), b""): h.update(b)
 return h.hexdigest().upper()
def load_frozen() -> tuple[dict[str,Any],dict[str,Any]]: return json.loads(CONTRACT.read_text()),json.loads(MANIFEST.read_text())
def load_development_calibration() -> dict[str,Any]:
 calibration=json.loads(CALIBRATION.read_text())
 required={name for group in calibration.get("score_components",{}).values() for name in group}
 mappings=calibration.get("feature_rank_mapping")
 july=calibration.get("previous_rth_context",{}).get("2026-07-31")
 expected_context={"PRIOR_RTH_HIGH","PRIOR_RTH_LOW","PRIOR_RTH_POC","PRIOR_RTH_VAH","PRIOR_RTH_VAL"}
 if (calibration.get("source") != "DEVELOPMENT_ONLY" or calibration.get("development_interaction_count") != 3089
  or calibration.get("feature_rank_mapping_encoding") != "zlib-base64-f64le-sorted-sample"
  or set(mappings or {}) != required or any(not isinstance(mappings[name],str) or not mappings[name] for name in required)
  or not isinstance(july,dict) or set(july) != expected_context):
  raise SealedRunError("missing DEVELOPMENT_ONLY score calibration")
 expected=load_frozen()[0]["frozen_selection"]
 if any(calibration.get("verified_thresholds",{}).get(k)!=expected[k] for k in ("absorption_p95","replenishment_p95")): raise SealedRunError("development calibration threshold mismatch")
 return calibration
def contract_hashes() -> dict[str,str]: return {"contract_sha256":digest(CONTRACT),"data_manifest_sha256":digest(MANIFEST)}

def plus_only(contract: dict[str,Any], row: dict[str,Any]) -> bool:
 """Literal frozen HIGH ∩ STRONG test; future response fields are forbidden."""
 allowed={"interaction_id","interaction_end","level","absorption_score","replenishment_score","direction","zone_low","zone_high"}
 if set(row)-allowed: raise SealedRunError("response/unknown field supplied to selection")
 s=contract["frozen_selection"]
 return row["level"] in s["mandatory_structural_levels"] and row["absorption_score"] >= s["absorption_p95"] and row["replenishment_score"] >= s["replenishment_p95"]

def prices(direction:str,bid:float,ask:float,zone_low:float,zone_high:float)->dict[str,float]:
 if not bid>0 or not ask>bid: raise SealedRunError("invalid executable market")
 if direction=="BUYER_ABSORPTION":
  entry,stop=ask+TICK,zone_low-5*TICK
  return {"entry_reference":ask,"entry":entry,"stop":stop,"stop_exit":stop-TICK,"target":entry+2*(entry-stop),"direction":"LONG"}
 if direction=="SELLER_ABSORPTION":
  entry,stop=bid-TICK,zone_high+5*TICK
  return {"entry_reference":bid,"entry":entry,"stop":stop,"stop_exit":stop+TICK,"target":entry-2*(stop-entry),"direction":"SHORT"}
 raise SealedRunError("invalid direction")
def size_trade(p:dict[str,float])->dict[str,float|int]:
 raw=abs(p["entry_reference"]-p["stop"])*USD_PER_POINT
 slip=(abs(p["entry"]-p["entry_reference"])+abs(p["stop_exit"]-p["stop"]))*USD_PER_POINT
 risk=abs(p["entry"]-p["stop_exit"])*USD_PER_POINT+2*COMMISSION
 return {"raw_price_risk_usd":raw,"slippage_contribution_usd":slip,"one_contract_price_risk_usd":raw+slip,"one_contract_initial_risk_usd":risk,"contracts":math.floor(RISK_BUDGET/risk)}

@dataclass
class State:
 source_index:int=0; stage:str="NEW"; interactions:list[dict[str,Any]]=field(default_factory=list)
 counters:dict[str,int]=field(default_factory=lambda:{k:0 for k in COUNTERS})
 submitted:list[str]=field(default_factory=list); traded:list[str]=field(default_factory=list)
 audit:list[dict[str,Any]]=field(default_factory=list); trades:list[dict[str,Any]]=field(default_factory=list)
 score_calibration_source:str="DEVELOPMENT_ONLY"; oos_rank_recomputation_count:int=0
 previous_rth_source_by_oos_date:dict[str,str]=field(default_factory=dict)
 actual_rth_dates_processed:list[str]=field(default_factory=list)
 rth_records_by_date:dict[str,int]=field(default_factory=dict)
 raw_interactions_by_date:dict[str,int]=field(default_factory=dict)
 def __post_init__(self):
  for k in COUNTERS:self.counters.setdefault(k,0)

def checkpoint(path:Path,state:State,source_sha:str)->None:
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp")
 tmp.write_bytes(canonical({"version":4,"source_sha256":source_sha,"hashes":contract_hashes(),"state":asdict(state)})); os.replace(tmp,path)
def restore(path:Path,source_sha:str)->State:
 p=json.loads(path.read_text())
 if p.get("version") != 4 or p.get("source_sha256")!=source_sha or p.get("hashes")!=contract_hashes(): raise SealedRunError("checkpoint/source contract mismatch")
 return State(**p["state"])

def _feature_values(row:dict[str,Any])->tuple[str,dict[str,float]]:
 """Interaction-local contemporaneous features; this never sees another OOS row."""
 buyer=float(row["sell_aggressor_volume"]); seller=float(row["buy_aggressor_volume"])
 direction="BUYER_ABSORPTION" if buyer>=seller else "SELLER_ABSORPTION"
 level,end=float(row["level_price"]),float(row["end_price"])
 through=max(0., ((level-end) if direction=="BUYER_ABSORPTION" else (end-level))/RAW_TICK)
 return direction,{"relevant_directional_aggressive_volume":max(buyer,seller),"executions":float(row["executions"]),"absolute_aggressive_imbalance":abs(float(row["aggressive_imbalance"])),"inverse_through_level_progress":-through,"interaction_end_rejection":1. if ((end>=level) if direction=="BUYER_ABSORPTION" else (end<=level)) else 0.,"replenishment_count":float(row["replenishment_count"]),"replenished_volume":float(row["replenished_volume"]),"replenished_execution_ratio":float(row["replenished_volume"])/max(1.,float(row["execution_volume"])),"repeated_cycles":float(row["replenishment_count"]),"queue_persistence_proxy":float(row["replenishment_count"])/max(1.,float(row["executions"]))}

def _development_rank(mapping:str, value:float)->float:
 """Frozen empirical midrank step function, fitted only on pilot interactions."""
 try:
  raw=zlib.decompress(base64.b64decode(mapping,validate=True))
  if len(raw) != 3089*8: raise ValueError("bad sample size")
  values=struct.unpack("<3089d",raw)
 except Exception as exc: raise SealedRunError("incompatible DEVELOPMENT_ONLY rank mapping") from exc
 # Exact empirical midrank: for an observed tie, (first_rank + last_rank)/2.
 # For a future value between pilot breakpoints, use its deterministic CDF
 # insertion rank.  No other OOS interaction is ever read here.
 lo=0; hi=len(values)
 while lo<hi:
  mid=(lo+hi)//2
  if values[mid] < value: lo=mid+1
  else: hi=mid
 first=lo; lo=first; hi=len(values)
 while lo<hi:
  mid=(lo+hi)//2
  if values[mid] <= value: lo=mid+1
  else: hi=mid
 last=lo
 return ((first+1+last)/2)/len(values) if first<last else last/len(values)

def _score(rows:list[dict[str,Any]], calibration:dict[str,Any]|None=None) -> list[dict[str,Any]]:
 """Score each OOS interaction independently with the frozen pilot mapping."""
 calibration=calibration or load_development_calibration(); mappings=calibration["feature_rank_mapping"]; components=calibration["score_components"]
 if not rows:return []
 out=[]
 for row in rows:
  direction,values=_feature_values(row)
  absorption=sum(_development_rank(mappings[name],values[name]) for name in components["absorption"])/5
  replenishment=sum(_development_rank(mappings[name],values[name]) for name in components["replenishment"])/5
  x=dict(row); x.update(direction=direction,interaction_end=row["end_ns"],absorption_score=absorption,replenishment_score=replenishment,zone_low=min(row["level_price"],row["end_price"]),zone_high=max(row["level_price"],row["end_price"])) ; out.append(x)
 return out

def completed_state(state:State)->bool:
 c=state.counters
 return (state.stage=="COMPLETE" and state.source_index==EXPECTED_RECORDS and c["dbn_records_seen"]==EXPECTED_RECORDS and c["rth_sessions_processed"]==5 and len(state.actual_rth_dates_processed)==5 and set(state.actual_rth_dates_processed)==set(load_frozen()[1]["chronology"]["eligible_rth_dates"]) and all(c[k]>0 for k in ("reconstruction_stage_invoked","interaction_stage_invoked","scoring_stage_invoked","plus_stage_invoked")) and c["audit_rows"]==c["plus_eligible"] and c["entered_trades"]==c["closed_trades"])
def reconcile(state:State)->None:
 c=state.counters
 # ``reconcile`` is also used by small ledger fixtures.  Only an artifact
 # claiming COMPLETE is subject to the stage gate; it can never be completed
 # with missing reconstruction/interaction/scoring/PLUS evidence.
 if state.stage=="COMPLETE" and not completed_state(state): raise SealedRunError("completion counters do not prove all required stages")
 if c["audit_rows"]!=len(state.audit) or c["entered_trades"]!=len(state.trades) or c["closed_trades"]!=len(state.trades): raise SealedRunError("ledger/counter mismatch")
 ids=[x["interaction_id"] for x in state.audit]
 if len(ids)!=len(set(ids)) or set(state.traded)-set(ids) or len({x["interaction_id"] for x in state.trades})!=len(state.trades): raise SealedRunError("duplicate/lost interaction IDs")
 if c["plus_eligible"]!=len(ids): raise SealedRunError("PLUS audit reconciliation failure")
 if any(t["exit_timestamp"]>t["cutoff_ns"] or t["contracts"]<1 or t["contracts"]!=int(t["contracts"]) for t in state.trades): raise SealedRunError("sealed execution invariant failed")
 if any(abs(t["net_usd"]-(t["gross_usd"]-t["commission_usd"]))>.000001 for t in state.trades): raise SealedRunError("USD reconciliation failed")

def _ns(value:str)->int:return int(datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()*1_000_000_000)
def is_snapshot_record(rec:Any, manifest:dict[str,Any])->bool:
 stamps={_ns(v) for v in manifest["chronology"]["databento_historical_mbo_start_snapshot_policy"]["initial_snapshot_receive_timestamps_utc"]}
 return rec.ts_recv in stamps and rec.action in {"R","A"}
def _valid(book:CausalMBOBook)->tuple[float,float]|None:
 bid,ask=book.best_bid(),book.best_ask()
 if bid is None or ask is None or bid<=0 or ask<=bid or book.depth["B"][bid]<1 or book.depth["A"][ask]<1:return None
 return bid/1e9,ask/1e9
def _cutoff(day:str)->int:return _ns(day+"T22:45:00+00:00")

def _quarantine_legacy(out:Path)->None:
 summary=out/"oos-v1-backtest-summary.json"
 if summary.exists() and not (out/"run-manifest.json").exists() or (summary.exists() and "RECONCILED" in summary.read_text() and not (out/"checkpoint.json").exists()):
  dest=out.parent/("INVALID_STUB_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")); dest.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(out),str(dest))
 elif summary.exists() and not (out/"checkpoint.json").read_text().find('"version":4')>=0:
  dest=out.parent/("INVALID_STUB_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")); shutil.move(str(out),str(dest))

def _reconstruct(records:Iterable[Any], manifest:dict[str,Any], state:State)->None:
 book=CausalMBOBook(); diag=Diagnostics(); context=set(); state.source_index=0; eligible=set(manifest["chronology"]["eligible_rth_dates"])
 calibration=load_development_calibration(); july_context=calibration["previous_rth_context"].get("2026-07-31")
 if not july_context: raise SealedRunError("missing read-only July 31 prior-RTH context")
 for rec in records:
  state.source_index+=1; state.counters["dbn_records_seen"]+=1
  if getattr(rec,"instrument_id",None)!=42140870:raise SealedRunError("wrong instrument")
  if is_snapshot_record(rec,manifest):
   state.counters["snapshot_records_seen"]+=1; book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False); state.counters["book_events_processed"]+=1; continue
  day,_=day_and_seconds(rec.ts_recv)
  # July development appears only as pre-materialized level context.  It is
  # never passed through the interaction, scoring, selection, or ledger path.
  if day not in eligible: raise SealedRunError("ordinary non-OOS record")
  if day not in context:
   source_day=PREVIOUS_RTH_SOURCES.get(day)
   if source_day is None: raise SealedRunError("unsealed prior-RTH provenance")
   if day=="2026-08-03": diag.levels[day]=dict(july_context)
   else: diag.finish_day_context(day)
   state.previous_rth_source_by_oos_date[day]=source_day
   context.add(day)
  if getattr(rec.action, 'value', rec.action) == 'N':
   state.counters['ordinary_records_seen'] += 1
   continue
  applied=book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False)
  state.counters["ordinary_records_seen"]+=1;state.counters["book_events_processed"]+=1;diag.observe(rec,applied,book.spread())
 diag.finalize(); rows=[r for r in diag.interaction_rows() if r["date"] in eligible]
 state.rth_records_by_date={day:diag.days[day].rth_events for day in manifest["chronology"]["eligible_rth_dates"]}
 state.actual_rth_dates_processed=[day for day,count in state.rth_records_by_date.items() if count>0 and day in context]
 state.raw_interactions_by_date={day:sum(r["date"]==day for r in rows) for day in manifest["chronology"]["eligible_rth_dates"]}
 state.counters["reconstruction_stage_invoked"]+=1;state.counters["interaction_stage_invoked"]+=1;state.counters["rth_sessions_processed"]=len(state.actual_rth_dates_processed);state.counters["structural_level_instances"]=sum(len(diag.levels.get(d,{})) for d in state.actual_rth_dates_processed);state.counters["raw_interactions"]=len(rows)
 state.interactions=_score(rows,calibration);state.counters["scoring_stage_invoked"]+=1
 contract,_=load_frozen()
 plus=[]
 for r in state.interactions:
  if r["absorption_score"]>=contract["frozen_selection"]["absorption_p95"]:state.counters["high_absorption"]+=1
  if r["replenishment_score"]>=contract["frozen_selection"]["replenishment_p95"]:state.counters["strong_replenishment"]+=1
  selection={k:r[k] for k in ("interaction_id","interaction_end","level","absorption_score","replenishment_score","direction","zone_low","zone_high")}
  if plus_only(contract,selection):plus.append(r)
 state.counters["plus_stage_invoked"]+=1;state.counters["plus_eligible"]=len(plus);state.interactions=plus;state.stage="SCORED"
 if state.previous_rth_source_by_oos_date != PREVIOUS_RTH_SOURCES: raise SealedRunError("incomplete prior-RTH provenance")

def _audit(state:State,row:dict[str,Any],outcome:str)->None:
 if any(item["interaction_id"]==row["interaction_id"] for item in state.audit): return
 if row["interaction_id"] not in state.submitted: state.submitted.append(row["interaction_id"])
 state.audit.append({"interaction_id":row["interaction_id"],"outcome":outcome});state.counters["audit_rows"]+=1
 if outcome=="LATENCY_NOT_REACHED":state.counters["latency_not_reached"]+=1
 elif outcome=="INVALID_SPREAD":state.counters["invalid_spread"]+=1
 elif outcome=="POSITION_ALREADY_OPEN":state.counters["position_already_open"]+=1
 elif outcome=="INSUFFICIENT_RISK_BUDGET_FOR_ONE_ES_CONTRACT":state.counters["insufficient_risk_budget"]+=1
 else:state.counters["other_non_trade_outcomes"]+=1

def write_outputs(out:Path,source_sha:str,state:State)->None:
 reconcile(state);out.mkdir(parents=True,exist_ok=True)
 for name,rows in (("trades.csv",state.trades),("audit.csv",state.audit)):
  fields=sorted({k for r in rows for k in r}) or ["interaction_id","outcome"]
  with (out/name).open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(rows)
 (out/"oos-v1-backtest-summary.json").write_bytes(canonical({"status":"RECONCILED","source_sha256":source_sha,**state.counters,"score_calibration_source":state.score_calibration_source,"oos_rank_recomputation_count":state.oos_rank_recomputation_count,"previous_rth_source_by_oos_date":state.previous_rth_source_by_oos_date,"actual_rth_dates_processed":state.actual_rth_dates_processed,"rth_records_by_date":state.rth_records_by_date,"raw_interactions_by_date":state.raw_interactions_by_date}))
 (out/"run-manifest.json").write_bytes(canonical({"immutable":True,"contract_hashes":contract_hashes(),"source_sha256":source_sha,"legacy_empty_result_invalid":True,"score_calibration_source":"DEVELOPMENT_ONLY","oos_rank_recomputation_count":0}))

def main(argv:list[str]|None=None)->int:
 ap=argparse.ArgumentParser();ap.add_argument("--dbn",type=Path,required=True);ap.add_argument("--resume",action="store_true");ap.add_argument("--run-dir",type=Path,default=RUN_ROOT);args=ap.parse_args(argv)
 contract,manifest=load_frozen();source_sha=digest(args.dbn)
 if source_sha!=manifest["proposed_acquisition"]["file_sha256"]:raise SealedRunError("sealed source SHA-256 mismatch")
 out=args.run_dir;_quarantine_legacy(out); cp=out/"checkpoint.json"
 if (out/"run-manifest.json").exists() and cp.exists():
  state=restore(cp,source_sha)
  if completed_state(state):return 0
 else:state=State()
 from databento import DBNStore
 if state.stage in {"NEW","RECONSTRUCTING"}:
  _reconstruct(DBNStore.from_file(args.dbn),manifest,state);checkpoint(cp,state,source_sha)
 # A second deterministic provider-order pass is execution only; it never recomputes selection.
 # It intentionally emits an audit outcome for every PLUS even when no executable post-latency quote occurs.
 by_end=sorted(state.interactions,key=lambda r:r["interaction_end"]); pending=list(by_end);book=CausalMBOBook();position=None;cutoff_quote=None
 for rec in DBNStore.from_file(args.dbn):
  if is_snapshot_record(rec,manifest):book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False);continue
  if getattr(rec.action, 'value', rec.action) == 'N':
   continue
  applied=book.apply(action=rec.action,side=rec.side,price=rec.price,size=rec.size,order_id=rec.order_id,sequence=rec.sequence,ts_recv=rec.ts_recv,channel_id=rec.channel_id,validate_sequence=False,mutate_execution=False);day,_=day_and_seconds(rec.ts_recv); quote=_valid(book); cutoff=_cutoff(day)
  if quote and cutoff-1_000_000_000 <= rec.ts_recv <= cutoff: cutoff_quote=(rec.ts_recv, quote)
  if position and rec.ts_recv > position["cutoff_ns"]:
   if cutoff_quote is None: raise SealedRunError("CUTOFF_EXECUTION_INTEGRITY_FAILURE")
   ts,q=cutoff_quote; exit_ref=q[0] if position["direction"]=="LONG" else q[1]; exit_fill=exit_ref-TICK if position["direction"]=="LONG" else exit_ref+TICK; gross=(exit_fill-position["entry"])*(1 if position["direction"]=="LONG" else -1)*USD_PER_POINT*position["contracts"]; fees=2*COMMISSION*position["contracts"]
   state.trades.append({**position,"exit_timestamp":ts,"exit_fill":exit_fill,"exit_reason":"CUTOFF_FORCED_FLAT","gross_usd":gross,"commission_usd":fees,"net_usd":gross-fees,"r_multiple":gross/(position["one_contract_initial_risk_usd"]*position["contracts"])});state.counters["closed_trades"]+=1;_audit(state,position,"ENTERED_CUTOFF_FORCED_FLAT");position=None;cutoff_quote=None
  while pending and pending[0]["interaction_end"]+2_000_000<=rec.ts_recv:
   row=pending.pop(0);state.counters["submission_attempts"]+=1
   if rec.ts_recv>=_cutoff(row["date"]):_audit(state,row,"CUTOFF_CANCELLED");state.counters["cutoff_cancelled"]+=1;continue
   if position is not None:_audit(state,row,"POSITION_ALREADY_OPEN");continue
   if quote is None:_audit(state,row,"INVALID_SPREAD");continue
   p=prices(row["direction"],quote[0],quote[1],row["zone_low"]/1e9,row["zone_high"]/1e9);s=size_trade(p)
   if not s["contracts"]:_audit(state,row,"INSUFFICIENT_RISK_BUDGET_FOR_ONE_ES_CONTRACT");continue
   position={**row,**p,**s,"entry_timestamp":rec.ts_recv,"cutoff_ns":_cutoff(row["date"])};state.submitted.append(row["interaction_id"]);state.traded.append(row["interaction_id"]);state.counters["entered_trades"]+=1
  if position and quote:
   bid,ask=quote; hit=(position["direction"]=="LONG" and (bid<=position["stop"] or bid>=position["target"])) or (position["direction"]=="SHORT" and (ask>=position["stop"] or ask<=position["target"]))
   if hit:
    exit_ref=bid if position["direction"]=="LONG" else ask;exit_fill=exit_ref-TICK if position["direction"]=="LONG" else exit_ref+TICK;reason="STOP" if ((position["direction"]=="LONG" and bid<=position["stop"]) or (position["direction"]=="SHORT" and ask>=position["stop"])) else "TARGET";gross=(exit_fill-position["entry"])*(1 if position["direction"]=="LONG" else -1)*USD_PER_POINT*position["contracts"];fees=2*COMMISSION*position["contracts"];state.trades.append({**position,"exit_timestamp":rec.ts_recv,"exit_fill":exit_fill,"exit_reason":reason,"gross_usd":gross,"commission_usd":fees,"net_usd":gross-fees,"r_multiple":gross/(position["one_contract_initial_risk_usd"]*position["contracts"])});state.counters["closed_trades"]+=1;_audit(state,position,"ENTERED_"+reason);position=None
 for row in pending:_audit(state,row,"LATENCY_NOT_REACHED")
 if position is not None:raise SealedRunError("cutoff force-flat execution requires a valid inclusive observation; refusing inferred exit")
 state.source_index=EXPECTED_RECORDS;state.counters["dbn_records_seen"]=EXPECTED_RECORDS;state.stage="COMPLETE";checkpoint(cp,state,source_sha);write_outputs(out,source_sha,state);return 0
if __name__=="__main__":raise SystemExit(main())
