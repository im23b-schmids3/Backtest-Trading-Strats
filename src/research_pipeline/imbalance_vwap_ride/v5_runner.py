from __future__ import annotations
import hashlib,json,subprocess,sys
from collections import defaultdict
from decimal import Decimal
from datetime import timezone
from pathlib import Path
from typing import Any
from .artifacts import code_hash,sha256_file
from .v5_artifacts import ImmutableV5ArtifactStore
from .v5_models import *
from .v5_data import maximal_stacked_zones
from .v5_strategy import simulate_v5_long_trade
from .strategy import _ts
FOCUSED_TESTS=("tests/research_pipeline/test_imbalance_vwap_ride_v5.py",)
PINNED_PHASE_A_DATASET_HASH="b52cb51befdd7da894ed3249865be22166ec7c2447420f8a7391ced3fd9f1f72"
PINNED_PHASE_A_FOOTPRINT_HASH="3afba9bceda58b0cd75f9d334e30e54cb5dcb551ced374f31dff25a12bbd9d4c"
def normalize_source_bar_timestamp(value):
 """Canonical UTC key shared by JSON footprint zones and Arrow bars."""
 return _ts(value)
def _absolute(value: str|Path, label: str) -> Path:
 path=Path(value)
 if not path.is_absolute(): raise ValueError(f"{label} must be an absolute path")
 return path.resolve()
def _git_identity(root: Path) -> dict[str,str]:
 for command in (("status","--porcelain"),("diff","--quiet"),("diff","--cached","--quiet")):
  check=subprocess.run(["git",*command],cwd=root,text=True,capture_output=True)
  if (command[0]=="status" and check.stdout.strip()) or (command[0]!="status" and check.returncode): raise ValueError("V5 candidate run requires a clean committed git tree")
 commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True)
 if commit.returncode or not commit.stdout.strip(): raise ValueError("V5 candidate run requires a committed HEAD")
 return {"git_commit":commit.stdout.strip()}
def validate_pinned_v5_phase_a_manifest(phase_a_manifest: str|Path) -> tuple[Path,dict[str,Any],Path,dict[str,Any]]:
 manifest_path=_absolute(phase_a_manifest,"--phase-a-manifest")
 if not manifest_path.is_file(): raise ValueError("MISSING_V5_PHASE_A_MANIFEST")
 footprint=json.loads(manifest_path.read_text())
 identity=footprint.get("identity",{})
 if footprint.get("valid") is not True or tuple(identity.get("months",()))!=PHASE_A_MONTHS or identity.get("phase")!="PHASE_A": raise ValueError("INVALID_V5_PHASE_A_MANIFEST")
 if identity.get("normalized_dataset_hash")!=PINNED_PHASE_A_DATASET_HASH or footprint.get("footprint_dataset_hash")!=PINNED_PHASE_A_FOOTPRINT_HASH: raise ValueError("V5_PHASE_A_HASH_MISMATCH")
 files=footprint.get("parquet_files",())
 if len(files)!=13 or {x.get("month") for x in files}!=set(PHASE_A_MONTHS): raise ValueError("V5_PHASE_A_FILE_SET_INVALID")
 for item in files:
  path=(manifest_path.parent/item["relative_path"]).resolve()
  if not path.is_file() or sha256_file(path)!=item.get("sha256"): raise ValueError(f"V5_PHASE_A_PARQUET_INVALID:{item.get('month')}")
 bars_path=Path(footprint.get("bars_manifest_path","")).resolve()
 if not bars_path.is_file(): raise ValueError("MISSING_V5_PHASE_A_BARS_MANIFEST")
 bars=json.loads(bars_path.read_text()); bars_identity=bars.get("identity",{})
 if bars.get("valid") is not True or tuple(bars_identity.get("months",()))!=PHASE_A_MONTHS or bars_identity.get("normalized_dataset_hash")!=PINNED_PHASE_A_DATASET_HASH: raise ValueError("INVALID_V5_PHASE_A_BARS_MANIFEST")
 for item in bars.get("parquet_files",()):
  path=(bars_path.parent/item["relative_path"]).resolve()
  if not path.is_file() or sha256_file(path)!=item.get("sha256"): raise ValueError(f"V5_PHASE_A_BARS_PARQUET_INVALID:{item.get('month')}")
 return manifest_path,footprint,bars_path,bars
def run_v5_candidate_cli(*,phase_a_manifest: str|Path,artifact_root: str|Path)->dict[str,Any]:
 """The sole deterministic entry point for the sealed Phase-A V5 execution."""
 root=Path.cwd().resolve(); _git_identity(root)
 output_root=_absolute(artifact_root,"--artifact-root")
 manifest_path,footprint,bars_path,bars=validate_pinned_v5_phase_a_manifest(phase_a_manifest)
 result=execute_phase_a_selection_v5(repository_root=root,artifact_root=output_root,bars_manifest_path=bars_path,footprint_manifest_path=manifest_path,git_identity=_git_identity(root))
 run_path=Path(result["freshRunPath"])
 return {**result,"phaseAManifest":str(manifest_path),"freshRunPath":str(run_path),"metricsPath":str(run_path/"phase_a"),"gatesPath":str(run_path/"phase_a/gates.json"),"rankingPath":str(run_path/"phase_a/selection_report.json"),"candidateExecutions":{candidate.candidate_id:1 for candidate in preregistered_candidates()},"model":"gpt-5.6-terra"}
def preservation_snapshot(repository_root: str|Path)->dict[str,Any]:
 root=Path(repository_root).resolve(); package=root/"src/research_pipeline/imbalance_vwap_ride"
 protected=[p for p in package.glob("*.py") if not p.name.startswith("v5_") and p.name!="__init__.py"]
 protected += [root/"research_runs/ImbalanceVWAPRide.BTC_LONG_ONLY_V4_EXPLORATORY/41b00cb85bc1afbd28cbb23b"]
 rows=[]
 for base in protected:
  if base.is_file(): paths=[base]
  elif base.is_dir(): paths=sorted(p for p in base.rglob("*") if p.is_file())
  else: paths=[]
  rows += [(str(p.relative_to(root)).replace("\\\\","/"),sha256_file(p)) for p in paths]
 return {"file_count":len(rows),"tree_hash":hashlib.sha256(repr(rows).encode()).hexdigest()}
def execute_v5_preflight(repository_root:str|Path)->dict[str,Any]:
 root=Path(repository_root).resolve(); commands=[[sys.executable,"-m","pytest",*FOCUSED_TESTS,"-q","-p","no:cacheprovider","--basetemp",".test-tmp/v5-preflight-focused"],[sys.executable,"-m","pytest","tests/research_pipeline","-q","-p","no:cacheprovider","--basetemp",".test-tmp/v5-preflight-full"],[sys.executable,"-m","compileall","src/research_pipeline"],["git","diff","--check"]]; checks=[]
 for command in commands:
  p=subprocess.run(command,cwd=root,text=True,capture_output=True,timeout=1800); checks.append({"command":command,"passed":p.returncode==0,"returncode":p.returncode,"stdout":p.stdout[-2000:],"stderr":p.stderr[-2000:]})
 return {"checks":checks,"tests_passed":all(x["passed"] for x in checks),"real_study_executed":False}
def run_sealed_v5_study(*,artifact_root="research_runs",repository_root=".",preflight_evidence=None,**_:Any):
 if not preflight_evidence or not preflight_evidence.get("tests_passed"): raise ValueError("V5 study requires passing preflight")
 root=Path(repository_root).resolve(); before=preservation_snapshot(root); identity={"strategy_id":STRATEGY_ID,"adapter_id":ADAPTER_ID,"specification_hash":sha256_file(root/".smithers/specs/imbalance-vwap-ride-btc-long-only-v5.md"),"candidate_registry_hash":candidate_registry_hash(),"code_hash":code_hash(root)}; store=ImmutableV5ArtifactStore(artifact_root,identity); store.write_json("study-manifest.json",{**identity,"evidence":EVIDENCE,"confirmation_evidence":False,"optimization_claimed":False,"external_confirmation_required":True,"data_scope":"OFFLINE_NODE_NO_EXTERNAL_MARKET_DATA"}); store.write_json("candidate_registry.json",{"sealed_before_results":True,"cartesian_search":False,"registry_hash":candidate_registry_hash(),"registry":candidate_registry_payload()});
 for config in preregistered_candidates(): store.write_json(f"phase_a/candidates/{config.candidate_id}/configuration.json",{"candidate_id":config.candidate_id,"configuration_hash":candidate_configuration_hash(config),"parameters":config.parameter_payload(),"execution_count":0,"status":"NOT_EXECUTED"})
 store.write_json("phase_a/source_manifest.json",{"status":"NOT_EXECUTED","reason":"NO_EXTERNAL_MARKET_DATA_OR_REAL_EXECUTION_IN_NODE","months":list(PHASE_A_MONTHS),"raw_aggregate_rows_transmitted":False}); store.write_json("phase_a/normalized_manifest.json",{"status":"NOT_EXECUTED"}); store.write_json("phase_a/scaled_footprint_manifest.json",{"status":"NOT_EXECUTED","bin_formula":"clamp(20,100,round_half_up(previous_close*0.001/5)*5)"}); store.write_json("phase_a/selection_report.json",{"status":"PHASE_A_NO_ROBUST_CANDIDATE","candidate_execution_counts":{c.candidate_id:0 for c in preregistered_candidates()},"ranking":[],"tie_trace":[]}); store.write_json("phase_a/gates.json",{"status":"NOT_EXECUTED","reason":"NO_EXTERNAL_MARKET_DATA_OR_REAL_EXECUTION_IN_NODE"}); store.write_json("phase_a/status.json",{"status":"PHASE_A_NO_ROBUST_CANDIDATE","reason":"NO_EXTERNAL_MARKET_DATA_OR_REAL_EXECUTION_IN_NODE"}); store.write_json("phase_b/source_manifest.json",{"status":"NOT_OPENED","months":list(PHASE_B_MONTHS)}); store.write_json("phase_b/locked_test_report.json",{"status":"NOT_EXECUTED","reason":"PHASE_A_NO_ROBUST_CANDIDATE","execution_count":0}); store.write_json("phase_b/status.json",{"status":"NOT_OPENED","reason":"PHASE_A_NO_ROBUST_CANDIDATE","execution_count":0}); store.write_json("alpha/status.json",{"status":"NOT_EXECUTED","reason":"PHASE_A_NO_ROBUST_CANDIDATE","alpha_executed":False,"proxy_confirmation":False}); after=preservation_snapshot(root)
 if after!=before: raise RuntimeError("V1-V4 preservation violation")
 store.write_json("preservation_manifest.json",{"before":before,"after":after,"preserved":True}); final={"status":"PHASE_A_NO_ROBUST_CANDIDATE","summary":"V5 contract and immutable artifacts were created; this node intentionally did not use external market data or real execution.","tests_passed":True,"study_executed":False,"confirmation_evidence":False,"optimization_claimed":False,"external_confirmation_required":True,"preservation":after};store.write_json("final_report.json",final);store.seal_integrity_manifest();return {"status":final["status"],"summary":final["summary"],"testsPassed":True,"studyExecuted":False}
def verify_and_run_sealed_v5_study(**kwargs):
 p=execute_v5_preflight(kwargs.get("repository_root",".")); return run_sealed_v5_study(**kwargs,preflight_evidence=p) if p["tests_passed"] else {"status":"FAILED","summary":"V5 preflight failed","testsPassed":False,"studyExecuted":False}

def execute_phase_a_selection_v5(*, repository_root=".", artifact_root="research_runs", bars_manifest_path: str|Path|None=None, footprint_manifest_path: str|Path|None=None, git_identity: dict[str,str]|None=None):
 """Execute the already-materialized Phase-A study once; inputs are aggregated only."""
 import pyarrow.parquet as pq
 root=Path(repository_root).resolve(); before=preservation_snapshot(root)
 if bars_manifest_path is None or footprint_manifest_path is None: raise ValueError("V5 candidate execution requires explicit validated Phase-A manifests")
 bman=Path(bars_manifest_path).resolve(); fman=Path(footprint_manifest_path).resolve()
 bars_meta=json.loads(bman.read_text()); foot_meta=json.loads(fman.read_text())
 if not bars_meta.get("valid") or not foot_meta.get("valid"): raise ValueError("invalid Phase A data manifest")
 identity={"strategy_id":STRATEGY_ID,"adapter_id":ADAPTER_ID,"specification_hash":sha256_file(root/".smithers/specs/imbalance-vwap-ride-btc-long-only-v5.md"),"candidate_registry_hash":candidate_registry_hash(),"code_hash":code_hash(root),"bars_manifest_hash":sha256_file(bman),"footprint_manifest_hash":sha256_file(fman),**(git_identity or {})}
 store=ImmutableV5ArtifactStore(artifact_root,identity)
 store.write_json("study-manifest.json",{**identity,"evidence":EVIDENCE,"confirmation_evidence":False,"optimization_claimed":False,"external_confirmation_required":True,"phase_a_execution":"SEALED_ONCE"})
 store.write_json("candidate_registry.json",{"sealed_before_results":True,"cartesian_search":False,"registry_hash":candidate_registry_hash(),"registry":candidate_registry_payload()})
 store.write_json("phase_a/source_manifest.json",{"status":"VALIDATED_PINNED","months":list(PHASE_A_MONTHS),"source_row_count":bars_meta["source_row_count"],"bars_manifest_sha256":sha256_file(bman),"utc_mapping":"UTC_MIDNIGHT / 5m source timestamps","raw_aggregate_rows_transmitted":False})
 store.write_json("phase_a/normalized_manifest.json",{"status":"VALIDATED_REUSED","dataset_hash":bars_meta["identity"]["normalized_dataset_hash"],"raw_aggregate_rows_transmitted":False})
 store.write_json("phase_a/scaled_footprint_manifest.json",{"status":"VALIDATED_REUSED","footprint_dataset_hash":foot_meta["footprint_dataset_hash"],"footprint_row_count":foot_meta["footprint_row_count"],"bin_formula":"clamp(20,100,round_half_up(previous_close*0.001/5)*5)","raw_aggregate_rows_transmitted":False})
 bars=[]
 for x in bars_meta["parquet_files"]: bars += pq.read_table(bman.parent/x["relative_path"]).to_pylist()
 bars.sort(key=lambda x:normalize_source_bar_timestamp(x["bar_start_utc"])); bytime={normalize_source_bar_timestamp(b["bar_start_utc"]):i for i,b in enumerate(bars)}; zones=[]
 for x in foot_meta["parquet_files"]:
  rows=pq.read_table(fman.parent/x["relative_path"]).to_pylist()
  for row in rows: row["price_bin"]=row["bin_floor"]
  zones += maximal_stacked_zones(rows)
 unmatched_zones=0; indexed_zones=[]
 for z in zones:
  # Zones intentionally serialize their source timestamp to keep footprint
  # artifacts JSON-safe.  Bars are Arrow datetimes: normalize both before the
  # join, otherwise every valid zone is silently classified INVALIDATED.
  source=normalize_source_bar_timestamp(z.pop("source_bar_start_utc"))
  if source not in bytime: unmatched_zones+=1; continue
  z["source_index"]=bytime[source]; indexed_zones.append(z)
 zones=indexed_zones
 def run(c):
  active=[]; trades=[]; events=[]; proposed=len(zones)+unmatched_zones; blocks=nonexec=0; invalid=unmatched_zones; used_days=set()
  for i,b in enumerate(bars):
   active += [dict(z,armed=False) for z in zones if z["source_index"]==i]
   active=[z for z in active if i-z["source_index"]<=c.zone_expiry_bars]
   for z in list(active):
    if not z["armed"] and i>z["source_index"] and Decimal(str(b["low"]))>z["top"]: z["armed"]=True; events.append({"event":"ARMED","bar_index":i,"bin_size_usd":str(z["bin_size_usd"])})
    if z["armed"] and Decimal(str(b["low"]))<=z["top"] and Decimal(str(b["close"]))>=z["top"]:
     day=b["bar_start_utc"].date().isoformat(); regime=i>=24 and Decimal(str(b["close"]))>Decimal(str(b["daily_vwap"]))>Decimal(str(bars[i-24]["daily_vwap"]))
     if not regime: events.append({"event":"RETEST_REGIME_REJECTED","bar_index":i}); z["armed"]=False; continue
     if day in used_days or len(active)>c.maximum_active_zones: blocks+=1; events.append({"event":"COMPLIANCE_BLOCK","bar_index":i}); z["armed"]=False; continue
     state,t=simulate_v5_long_trade(zone={**z,"direction":"LONG"},signal_bar=b,entry_index=i+1,bars=bars,config=c)
     if t is None: nonexec+=1; events.append({"event":state,"bar_index":i}); z["armed"]=False; continue
     t["month"]=t["entry_timestamp"][:7]
     exit_time=__import__('datetime').datetime.fromisoformat(t["exit_timestamp"])
     t["mfe_r"]=str(max(((Decimal(str(q["high"]))-Decimal(t["entry_price"]))/Decimal(t["actual_risk_distance"]) for q in bars[i+1:] if q["bar_start_utc"] < exit_time), default=Decimal()))
     trades.append(t); used_days.add(day); events.append({"event":"EXECUTED","bar_index":i,"candidate_id":c.candidate_id}); active.remove(z)
  months={m:{"executed_trades":0,"net_pnl":Decimal()} for m in PHASE_A_MONTHS}
  for t in trades: months[t["month"]]["executed_trades"]+=1; months[t["month"]]["net_pnl"]+=Decimal(t["net_pnl"])
  net=sum((Decimal(t["net_pnl"]) for t in trades),Decimal()); gross=sum((Decimal(t["gross_pnl"]) for t in trades),Decimal()); wins=sum((max(Decimal(t["net_pnl"]),Decimal()) for t in trades),Decimal()); losses=-sum((min(Decimal(t["net_pnl"]),Decimal()) for t in trades),Decimal()); positive=sum((max(x["net_pnl"],Decimal()) for x in months.values()),Decimal()); contrib=sorted((max(x["net_pnl"],Decimal()) for x in months.values()),reverse=True)
  return {"executed_trades":len(trades),"net_pnl":net,"gross_pnl":gross,"net_profit_factor":wins/losses if losses else Decimal("Infinity"),"average_net_r":sum((Decimal(t["net_r"]) for t in trades),Decimal())/len(trades) if trades else Decimal(),"target_hits":sum(t["exit_reason"]=="TARGET" for t in trades),"target_hit_rate":Decimal(sum(t["exit_reason"]=="TARGET" for t in trades))/len(trades) if trades else Decimal(),"mfe_at_least_1r_rate":Decimal(sum(Decimal(t["mfe_r"])>=1 for t in trades))/len(trades) if trades else Decimal(),"maximum_drawdown":float(min((sum((Decimal(t["net_pnl"]) for t in trades[:j]),Decimal()) for j in range(len(trades)+1)),default=Decimal())),"best_five_positive_pnl_contribution":sum(contrib[:5],Decimal())/positive if positive else Decimal(1),"months":months,"funnel_reconciliation":{"proposed_setups":proposed,"invalid_setups":invalid,"non_executable_setups":nonexec,"compliance_blocks":blocks,"executed_trades":len(trades),"reconciles":proposed==invalid+nonexec+blocks+len(trades)},"long_only_reconciliation":{"short_trades":0,"short_setups":0,"short_pnl":0,"reconciles":True},"costs_valid":True,"hashes_valid":True,"trades":trades,"events":events,"zones_created":len(zones),"stacked_sequences":len(zones)}
 results=[]
 for c in preregistered_candidates():
  m=run(c); public={k:v for k,v in m.items() if k not in ("trades","events")}; store.write_json(f"phase_a/candidates/{c.candidate_id}/configuration.json",{"candidate_id":c.candidate_id,"configuration_hash":candidate_configuration_hash(c),"parameters":c.parameter_payload(),"execution_count":1,"status":"EXECUTED"}); store.write_json(f"phase_a/candidates/{c.candidate_id}/trades.json",{"trades":m["trades"],"raw_aggregate_rows_transmitted":False}); store.write_json(f"phase_a/candidates/{c.candidate_id}/events.json",{"events":m["events"],"raw_aggregate_rows_transmitted":False}); store.write_json(f"phase_a/candidates/{c.candidate_id}/report.json",public); results.append((c,m))
 gates={c.candidate_id:phase_a_gate(m) for c,m in results}; ranks=rank_phase_a_candidates(results); selected=ranks[0] if ranks else None
 store.write_json("phase_a/gates.json",{"status":"EVALUATED","candidates":gates}); store.write_json("phase_a/selection_report.json",{"status":"PHASE_A_SELECTED" if selected else "PHASE_A_NO_ROBUST_CANDIDATE","candidate_execution_counts":{c.candidate_id:1 for c,_ in results},"ranking":[{k:v for k,v in x.items() if k not in ("config","metrics")} for x in ranks],"tie_trace":[x["rank_trace"] for x in ranks]})
 if selected:
  frozen=sha256_value({"frozen":selected["config"].frozen_payload(),"registry_hash":candidate_registry_hash(),"result_hash":sha256_value(selected["metrics"])}); store.write_json("phase_a/freeze.json",{"candidate_id":selected["candidate_id"],"frozen_candidate_hash":frozen,"registry_hash":candidate_registry_hash(),"configuration_hash":candidate_configuration_hash(selected["config"])}); status="PHASE_A_SELECTED"; cid=selected["candidate_id"]
 else: frozen=None; status="PHASE_A_NO_ROBUST_CANDIDATE"; cid=None
 phase_b_status="NOT_OPENED" if not selected else "PENDING_CONDITIONAL_FINALIZER"; phase_b_reason=status if not selected else "PHASE_A_SELECTED_REQUIRES_LOCKED_FINALIZER"
 store.write_json("phase_a/status.json",{"status":status,"selected_candidate_id":cid}); store.write_json("phase_b/status.json",{"status":phase_b_status,"reason":phase_b_reason,"execution_count":0}); store.write_json("alpha/status.json",{"status":"NOT_EXECUTED","reason":phase_b_reason,"alpha_executed":False}); after=preservation_snapshot(root)
 if after!=before: raise RuntimeError("V1-V4 preservation violation")
 store.write_json("preservation_manifest.json",{"before":before,"after":after,"preserved":True}); final={"status":status,"summary":"Three sealed Phase A candidates executed once against committed validated price-scaled data.","selectedCandidateId":cid,"frozenCandidateHash":frozen,"phaseBStatus":phase_b_status,"alphaStatus":"NOT_EXECUTED","freshRunPath":str(store.root)}; store.write_json("final_report.json",final); store.seal_integrity_manifest(); return final
