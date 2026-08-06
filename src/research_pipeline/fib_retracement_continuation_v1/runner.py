from __future__ import annotations
import hashlib,json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from .constants import *
from .models import Candidate, Bar, ExecutionAssumptions
from .strategy import causal_setups, expire_reason
from .execution import submit_order,execute_order,process_position,_exit
from .accounting import close_trade
from .metrics import metrics,gates
from .reconciliation import reconcile
from .manifests import require_development_mode,verify_manifest,verify_chronology_manifest,ManifestError
from .loader import load_development_bars

def _json(x):
 if isinstance(x,Decimal): return format(x,"f")
 if hasattr(x,"isoformat"): return x.isoformat().replace("+00:00","Z")
 raise TypeError(type(x).__name__)
def _write(root:Path,name:str,payload:Any):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,sort_keys=True,indent=2,default=_json)+"\n",encoding="utf-8")
def _sealed(root:Path):
 p=root/SPEC_PATH; text=p.read_text(encoding="utf-8") if p.is_file() else ""
 if not text or not text.startswith("# "+STRATEGY_ID) or STRATEGY_ID not in text: raise ValueError("MISSING_SEALED_FIB09_PROSPECTIVE_SPECIFICATION")
 return p
def _candidate(row): return Candidate(**row)
def _store(root:Path,identity:dict):
 if root.exists(): raise FileExistsError("FIB09_V1_IMMUTABLE_ARTIFACT_ROOT_COLLISION")
 root.mkdir(parents=True); return root
def _seal(root:Path,identity:dict):
 files=[{"path":p.relative_to(root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob("*")) if p.is_file() and p.name!="integrity-manifest.json"]
 _write(root,"integrity-manifest.json",{"identity":identity,"files":files,"manifest_sha256":hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()})

def run_candidate(bars:list[Bar], candidate:Candidate, assumptions:ExecutionAssumptions=ExecutionAssumptions())->dict:
 setups=causal_setups(bars,candidate); orders=[];outcomes=[];events=[];trades=[];equity=assumptions.opening_equity; active=None; pending=[]
 for index,bar in enumerate(bars):
  for setup in [s for s in setups if s.get("extreme_timestamp")==bar.timestamp and "terminal" not in s]:
   if index+1>=len(bars): setup["terminal"]="SESSION_OR_DATA_END"; outcomes.append({"setup_id":setup["setup_id"],"disposition":setup["terminal"]}); continue
   pending.append(submit_order(setup,bars[index+1].timestamp,1)); orders.append(pending[-1]); events.append({"kind":"ORDER_SUBMITTED","setup_id":setup["setup_id"],"order_id":pending[-1]["order_id"],"timestamp":bar.timestamp})
  for order in list(pending):
   if bar.timestamp<order["active_timestamp"]: continue
   reason=expire_reason(order,bar,candidate)
   if active is not None: reason="ACTIVE_POSITION_BLOCKED"
   if reason: outcomes.append({"setup_id":order["setup_id"],"disposition":reason});pending.remove(order);continue
   trade,reject=execute_order(order,bar,candidate,equity,assumptions)
   if reject: outcomes.append({"setup_id":order["setup_id"],"disposition":reject});pending.remove(order);continue
   if trade:
    active=trade; pending.remove(order); outcomes.append({"setup_id":order["setup_id"],"disposition":"TRADE_EXECUTED"});events.append({"kind":"ORDER_FILLED","order_id":order["order_id"],"trade_id":trade["trade_id"],"timestamp":bar.timestamp})
  if active is not None and bar.timestamp>active["entry_timestamp"]:
   process_position(active,bar,candidate,assumptions)
   if active["remaining_quantity"]<=0:
    closed=close_trade(active);trades.append(closed);equity+=closed["net_pnl"];active=None
 if active is not None:
  _exit(active,bars[-1].close,active["remaining_quantity"],bars[-1].timestamp,"DATA_END_FORCE_CLOSE",len(active["legs"])+1,assumptions);closed=close_trade(active);trades.append(closed);equity+=closed["net_pnl"]
 for order in pending: outcomes.append({"setup_id":order["setup_id"],"disposition":"SESSION_OR_DATA_END"})
 for setup in setups:
  if setup.get("terminal") and not any(x["setup_id"]==setup["setup_id"] for x in outcomes): outcomes.append({"setup_id":setup["setup_id"],"disposition":setup["terminal"]})
 rec=reconcile(setups,outcomes,orders,trades,assumptions.opening_equity); met=metrics(trades,assumptions.opening_equity); return {"events":events,"setups":setups,"setup_outcomes":outcomes,"orders":orders,"trades":trades,"partial_exits":[leg for trade in trades for leg in trade["legs"]],"metrics":met,"reconciliation":rec,"gates":gates(met,trades,rec["reconciles"])}

def materialize_synthetic(*,artifact_root:str|Path,repository_root:str|Path)->dict:
 repo=Path(repository_root).resolve();spec=_sealed(repo);root=_store(Path(artifact_root).resolve(),{"mode":"SYNTHETIC"});identity={"strategy_id":STRATEGY_ID,"mode":"SYNTHETIC_ONLY","holdout_status":"LOCKED_NOT_OPENED"}
 _write(root,"sealed-specification.json",{"path":SPEC_PATH,"sha256":hashlib.sha256(spec.read_bytes()).hexdigest(),"sealed":True});_write(root,"candidate-registry.json",{"registry":CANDIDATES,"sealed_before_results":True});_write(root,"evidence-classification.json",{"PROSPECTIVE_V1_DECISION":["entry_expiry_time_stop","concurrency","costs_quantity_accounting"]});_write(root,"chronology-manifest.json",{"status":"SYNTHETIC_NO_HOLDOUT","holdout_status":"LOCKED_NOT_OPENED"});_write(root,"data-manifest.json",{"status":"NOT_READ","synthetic_only":True});_write(root,"execution-assumptions.json",asdict(ExecutionAssumptions()))
 for row in CANDIDATES:
  base=Path("candidates")/row["candidate_id"]
  for name,payload in (("events.json",[]),("setup-outcomes.json",[]),("orders.json",[]),("trades.json",[]),("partial-exits.json",[]),("monthly-metrics.json",{}),("report.json",{"status":"SYNTHETIC_NOT_EXECUTED"}),("gates.json",{"status":"SYNTHETIC_NOT_EXECUTED"}),("reconciliation.json",{"reconciles":True})):_write(root,base/name,payload)
 _write(root,"freeze.json",{"status":"FROZEN","holdout_status":"LOCKED_NOT_OPENED"});_write(root,"development-result.json",{"status":"NOT_EXECUTED","holdout_status":"LOCKED_NOT_OPENED","candidates":[]});_write(root,"final-report.json",{"status":"SYNTHETIC_ONLY","holdout_status":"LOCKED_NOT_OPENED"});_seal(root,identity);return {"artifact_root":str(root),"holdout_status":"LOCKED_NOT_OPENED"}
def run_synthetic(*,bars_by_candidate:dict[str,list[Bar]],artifact_root:str|Path,repository_root:str|Path)->dict:
 repo=Path(repository_root).resolve();spec=_sealed(repo);root=_store(Path(artifact_root).resolve(),{"mode":"SYNTHETIC_RUN"}); results=[]
 for row in CANDIDATES:
  item=run_candidate(bars_by_candidate.get(row["candidate_id"],[]),_candidate(row));base=Path("candidates")/row["candidate_id"]
  for key,name in (("events","events.json"),("setup_outcomes","setup-outcomes.json"),("orders","orders.json"),("trades","trades.json"),("partial_exits","partial-exits.json"),("metrics","monthly-metrics.json"),("metrics","report.json"),("gates","gates.json"),("reconciliation","reconciliation.json")):_write(root,base/name,item[key])
  results.append({"candidate_id":row["candidate_id"],"hard_gates":item["gates"]["hard_gates"],"evidence_label":item["metrics"]["evidence_label"],"reconciles":item["reconciliation"]["reconciles"]})
 _write(root,"sealed-specification.json",{"path":SPEC_PATH,"sha256":hashlib.sha256(spec.read_bytes()).hexdigest(),"sealed":True});_write(root,"candidate-registry.json",{"registry":CANDIDATES});_write(root,"evidence-classification.json",{"classification":"PROSPECTIVE_V1_DECISION"});_write(root,"chronology-manifest.json",{"synthetic":True,"holdout_status":"LOCKED_NOT_OPENED"});_write(root,"data-manifest.json",{"synthetic_only":True});_write(root,"execution-assumptions.json",asdict(ExecutionAssumptions()));_write(root,"freeze.json",{"holdout_status":"LOCKED_NOT_OPENED"});_write(root,"development-result.json",{"status":"SYNTHETIC","holdout_status":"LOCKED_NOT_OPENED","candidates":results});_write(root,"final-report.json",{"status":"SYNTHETIC","holdout_status":"LOCKED_NOT_OPENED"});_seal(root,{"strategy_id":STRATEGY_ID,"mode":"SYNTHETIC_RUN"});return {"artifact_root":str(root),"candidates":results,"holdout_status":"LOCKED_NOT_OPENED"}

def development_diagnostic(*,eth_manifest:str|Path,btc_manifest:str|Path)->dict:
 return {"eth":verify_manifest(eth_manifest,mode="development"),"btc":verify_manifest(btc_manifest,mode="development"),"holdout_status":"LOCKED_NOT_OPENED"}

def run_development(*, eth_manifest:str|Path, btc_manifest:str|Path, chronology_manifest:str|Path, artifact_root:str|Path, repository_root:str|Path)->dict:
 require_development_mode("development")
 root_path=Path(artifact_root); repo=Path(repository_root)
 if not root_path.is_absolute(): raise ValueError("FIB09_V1_ARTIFACT_ROOT_MUST_BE_ABSOLUTE")
 if root_path.exists(): raise FileExistsError("FIB09_V1_IMMUTABLE_ARTIFACT_ROOT_COLLISION")
 if not repo.is_absolute() or not repo.is_dir(): raise ValueError("FIB09_V1_REPOSITORY_ROOT_MUST_BE_ABSOLUTE_EXISTING")
 # Manifest-only verification occurs before the loader touches either source.
 chronology=verify_chronology_manifest(chronology_manifest,eth_manifest=eth_manifest,btc_manifest=btc_manifest)
 # The diagnostic is manifest-only; the chronology lock is validated before
 # either loader can decode a Parquet row.
 diagnostic=development_diagnostic(eth_manifest=eth_manifest,btc_manifest=btc_manifest)
 eth_bars,eth_data=load_development_bars(eth_manifest,development_start=chronology["development_start"],development_end=chronology["development_end"],chronology_claim=chronology["assets"]["ETH"]); btc_bars,btc_data=load_development_bars(btc_manifest,development_start=chronology["development_start"],development_end=chronology["development_end"],chronology_claim=chronology["assets"]["BTC"])
 spec=_sealed(repo); results=[]; candidate_results=[]
 try:
  for row in CANDIDATES:
   item=run_candidate(eth_bars if row["symbol"]=="ETH" else btc_bars,_candidate(row));
   if not item["reconciliation"]["reconciles"]: raise ManifestError("FIB09_V1_RECONCILIATION_FAILED")
   candidate_results.append((row,item))
   results.append({"candidate_id":row["candidate_id"],"executed_trade_count":item["metrics"]["executed_trade_count"],"hard_gates":item["gates"]["hard_gates"],"evidence_label":item["metrics"]["evidence_label"],"reconciles":True})
  root=_store(root_path.resolve(),{"mode":"DEVELOPMENT"})
  for row,item in candidate_results:
   base=Path("candidates")/row["candidate_id"]
   for key,name in (("events","events.json"),("setup_outcomes","setup-outcomes.json"),("orders","orders.json"),("trades","trades.json"),("partial_exits","partial-exits.json"),("metrics","monthly-metrics.json"),("metrics","report.json"),("gates","gates.json"),("reconciliation","reconciliation.json")):_write(root,base/name,item[key])
  _write(root,"sealed-specification.json",{"path":SPEC_PATH,"sha256":hashlib.sha256(spec.read_bytes()).hexdigest(),"sealed":True}); _write(root,"candidate-registry.json",{"registry":CANDIDATES,"sealed_before_results":True}); _write(root,"evidence-classification.json",{"classification":"PROSPECTIVE_V1_DECISION"}); _write(root,"chronology-manifest.json",{"source_path":chronology["chronology_manifest_path"],"holdout_status":"LOCKED_NOT_OPENED"}); _write(root,"data-manifest.json",{"eth":eth_data,"btc":btc_data,"rows_read":True}); _write(root,"execution-assumptions.json",asdict(ExecutionAssumptions())); _write(root,"freeze.json",{"holdout_status":"LOCKED_NOT_OPENED"}); _write(root,"development-result.json",{"status":"DEVELOPMENT_EXECUTED","holdout_status":"LOCKED_NOT_OPENED","candidates":results}); _write(root,"final-report.json",{"status":"DEVELOPMENT","holdout_status":"LOCKED_NOT_OPENED","candidates":results}); _seal(root,{"strategy_id":STRATEGY_ID,"mode":"DEVELOPMENT"})
 except Exception:
  # A failed immutable run is never presented as a completed artifact tree.
  raise
 return {"artifact_root":str(root),"rows_read":True,"candidate_count":len(results),"holdout_status":"LOCKED_NOT_OPENED"}
def run_holdout(**kwargs): raise ManifestError("LOCKED_HOLDOUT_NOT_AUTHORIZED")
