"""Resumable 20-date construction execution for frozen ES mechanism research."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any

from . import es_four_mechanism_comparative_research_v1 as audit
from .es_four_mechanism_native_replay import run_one_date, source_window_checkpoint_equivalence

STUDY="ES_FOUR_MECHANISM_COMPARATIVE_RESEARCH_V1"
STUDY_ROOT=Path("research_runs")/STUDY


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def _atomic_json(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)


def _read_json(path:Path):
    return json.loads(path.read_text())


def verify_authorized_inputs(repo:Path)->tuple[dict[str,Any],list[dict[str,Any]],str]:
    root=repo/STUDY_ROOT
    protocol_path=root/"frozen-study-protocol.json"
    declared=(root/"frozen-study-protocol.sha256").read_text().split()[0]
    protocol_sha=_sha(protocol_path)
    if protocol_sha!=declared:raise RuntimeError("frozen protocol SHA-256 mismatch")
    protocol=_read_json(protocol_path)
    split=_read_json(root/"date-split-manifest.json")
    dates=list(split.get("construction_dates",[]))
    if dates!=protocol["scope"]["construction_dates"] or len(dates)!=20:
        raise RuntimeError("construction date split differs from sealed protocol")
    appendix_sha=_sha(root/"implementation-appendix-v1.md")
    definitions_sha=_sha(root/"mechanism-definition-registry.json")
    interactions_sha=_sha(root/"interaction-hypothesis-registry.json")
    if _read_json(root/"mechanism-definition-registry.json")["frozen_protocol_sha256"]!=protocol_sha:
        raise RuntimeError("mechanism registry protocol identity mismatch")
    if _read_json(root/"interaction-hypothesis-registry.json")["frozen_protocol_sha256"]!=protocol_sha:
        raise RuntimeError("interaction registry protocol identity mismatch")
    sources={r["date"]:r for r in audit.source_rows(repo)}
    verified=[]
    for day in dates:
        source=sources.get(day)
        if source is None:raise RuntimeError(f"source missing from authorized manifest: {day}")
        raw=repo/source["raw_path"]
        if not raw.is_file() or raw.stat().st_size!=int(source["raw_bytes"]):
            raise RuntimeError(f"source missing/size mismatch: {day}")
        got=_sha(raw)
        if got!=source["raw_sha256"]:raise RuntimeError(f"source SHA-256 mismatch: {day}")
        expected="ESH5" if day<="2025-03-14" else "ESM5"
        if source["dataset"]!="GLBX.MDP3" or source["schema"]!="mbp-10" or source["symbol"]!=expected:
            raise RuntimeError(f"source dataset/schema/contract mismatch: {day}")
        verified.append({"date":day,"path":source["raw_path"],"size_bytes":raw.stat().st_size,
                         "sha256":got,"dataset":"GLBX.MDP3","schema":"mbp-10","symbol":expected})
        print(f"SOURCE_VERIFIED {day} {expected} sha256={got}",flush=True)
    return protocol,verified,appendix_sha+":"+definitions_sha+":"+interactions_sha


def _artifact_rows(date_dir:Path)->dict[str,dict[str,Any]]:
    result={}
    for p in sorted(date_dir.iterdir()):
        if p.is_file() and p.name not in ("date-manifest.json","date-checkpoint.json"):
            result[p.name]={"sha256":_sha(p),"size_bytes":p.stat().st_size}
    return result


def execute_construction(repo:Path, *, chunk_rows:int=100_000)->dict[str,Any]:
    protocol,verified,registry_hashes=verify_authorized_inputs(repo)
    code_path=repo/"src/research_pipeline/cme_orderflow_absorption_l2_v1/es_four_mechanism_native_replay.py"
    test_path=repo/"tests/test_es_four_mechanism_native_replay.py"
    code_sha=_sha(code_path); test_sha=_sha(test_path)
    engine_id=hashlib.sha256((code_sha+test_sha+registry_hashes).encode()).hexdigest()[:16]
    root=repo/STUDY_ROOT/"construction-20-date-v1"/f"engine-{engine_id}"
    dates_root=root/"dates"; dates_root.mkdir(parents=True,exist_ok=True)
    verified_by_date={x["date"]:x for x in verified}
    source_rows={x["date"]:x for x in audit.source_rows(repo)}
    progress_path=root/"construction-progress.json"
    completed=[]
    for i,day in enumerate(protocol["scope"]["construction_dates"],1):
        dest=dates_root/day
        manifest_path=dest/"date-manifest.json"
        if manifest_path.is_file():
            old=_read_json(manifest_path)
            if old.get("source_sha256")==verified_by_date[day]["sha256"] and old.get("engine_sha256")==code_sha:
                problems=[]
                for name,rec in old.get("artifacts",{}).items():
                    p=dest/name
                    if not p.is_file() or p.stat().st_size!=rec["size_bytes"] or _sha(p)!=rec["sha256"]:
                        problems.append(name)
                if not problems:
                    completed.append(day)
                    print(f"DATE_SKIP_VERIFIED {i}/20 {day}",flush=True)
                    _atomic_json(progress_path,{"status":"RUNNING","completed_dates":completed,"next_date":None if i==20 else protocol["scope"]["construction_dates"][i],"engine_sha256":code_sha})
                    continue
                raise RuntimeError(f"completed date output hash mismatch for {day}: {problems}")
            raise RuntimeError(f"existing date output has different source/engine identity: {dest}")
        temp=dates_root/f".{day}.tmp"
        if temp.exists():shutil.rmtree(temp)
        temp.mkdir(parents=True)
        print(f"DATE_BEGIN {i}/20 {day} {verified_by_date[day]['symbol']}",flush=True)
        started=time.monotonic()
        eq=source_window_checkpoint_equivalence(repo,source_rows[day],count=10_000)
        if eq.get("status")!="PASS" or not eq.get("detector_state_equivalent"):
            raise RuntimeError(f"source checkpoint equivalence failed: {day}: {eq}")
        result=run_one_date(repo,temp,day,chunk_rows,construction_outcomes=True,checkpoint_equivalence=eq)
        if result["source_sha256"]!=verified_by_date[day]["sha256"]:
            raise RuntimeError(f"source changed while processing {day}")
        runtime=time.monotonic()-started
        checkpoint={"date":day,"source_sha256":verified_by_date[day]["sha256"],
                    "engine_sha256":code_sha,"protocol_sha256":_sha(repo/STUDY_ROOT/"frozen-study-protocol.json"),
                    "next_ordinal":result["source_records_processed"],"status":"DATE_COMPLETE"}
        _atomic_json(temp/"date-checkpoint.json",checkpoint)
        artifacts=_artifact_rows(temp)
        date_manifest={"date":day,"source":verified_by_date[day],"engine_sha256":code_sha,
            "test_sha256":test_sha,"protocol_sha256":protocol["_verified_sha256"] if "_verified_sha256" in protocol else _sha(repo/STUDY_ROOT/"frozen-study-protocol.json"),
            "registry_hashes":registry_hashes,"result":result,"runtime_seconds":runtime,
            "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "artifacts":artifacts,"status":"PASS"}
        _atomic_json(temp/"date-manifest.json",date_manifest)
        if dest.exists():raise RuntimeError(f"refusing to overwrite existing date directory: {dest}")
        os.replace(temp,dest)
        completed.append(day)
        _atomic_json(progress_path,{"status":"RUNNING","completed_dates":completed,
            "next_date":None if i==20 else protocol["scope"]["construction_dates"][i],"engine_sha256":code_sha})
        print(f"DATE_DONE {i}/20 {day} records={result['source_records_processed']} triggers={result['trigger_counts']} runtime_sec={runtime:.1f}",flush=True)
    _atomic_json(progress_path,{"status":"ALL_20_DATES_COMPLETE","completed_dates":completed,
        "engine_sha256":code_sha,"protocol_sha256":_sha(repo/STUDY_ROOT/"frozen-study-protocol.json")})
    return {"root":str(root.relative_to(repo)),"engine_sha256":code_sha,"test_sha256":test_sha,
            "protocol_sha256":_sha(repo/STUDY_ROOT/"frozen-study-protocol.json"),
            "construction_dates_completed":len(completed),"dates":completed,"status":"DATES_COMPLETE"}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--chunk-rows",type=int,default=100_000)
    args=parser.parse_args(argv)
    result=execute_construction(args.repo.resolve(),chunk_rows=args.chunk_rows)
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
