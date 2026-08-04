from __future__ import annotations
import hashlib,subprocess,sys
from pathlib import Path
from typing import Any
from .artifacts import code_hash,sha256_file
from .v5_artifacts import ImmutableV5ArtifactStore
from .v5_models import *
FOCUSED_TESTS=("tests/research_pipeline/test_imbalance_vwap_ride_v5.py",)
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
