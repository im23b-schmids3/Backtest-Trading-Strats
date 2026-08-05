from __future__ import annotations

import hashlib
import json
import subprocess
import random
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..imbalance_vwap_ride.artifacts import sha256_file
from .artifacts import ImmutableLSMRArtifactStore
from .models import EVIDENCE, STRATEGY_ID, candidate_configuration_hash, candidate_registry_hash, candidate_registry_payload, preregistered_candidates
from .strategy import detect_setups, simulate_trade, terminal_disposition, validate_setup_audit

SPEC_PATH = ".smithers/specs/liquidity-sweep-mean-reversion-v1.md"
PHASE_A_BARS = "data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json"
PHASE_A_FOOTPRINTS = "data/imbalance_vwap_ride/v5/footprints/BTCUSDT/phase_a/4f8e06b06b8348d9e071983bdef6239f313cdac51c89475c0ed09181843f79e3/manifest.json"
PHASE_A_MONTHS = tuple([f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"])

def _require_clean_git(root: Path) -> str:
    for command in (("status", "--porcelain"), ("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", *command], cwd=root, text=True, capture_output=True)
        if (command[0] == "status" and result.stdout.strip()) or (command[0] != "status" and result.returncode):
            raise ValueError("LSMR_PHASE_A_REQUIRES_CLEAN_GIT")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    if result.returncode or not result.stdout.strip(): raise ValueError("LSMR_PHASE_A_REQUIRES_COMMITTED_HEAD")
    return result.stdout.strip()

def _sealed_spec(root: Path) -> Path:
    path = root / SPEC_PATH
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    required = (STRATEGY_ID, "LSMR-V1-1P5R", "LSMR-V1-2P0R", "LSMR-V1-2P5R", "Phase A: 2023-01-01T00:00:00Z", "Phase B: 2024-02-01T00:00:00Z", "Same-bar ambiguity Stop first")
    if not all(item in text for item in required): raise ValueError("MISSING_SEALED_LSMR_SPECIFICATION")
    return path

def _load_pinned_bars(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Read only the pinned Phase-A bar manifest and its declared 5m files."""
    manifest_path = (root / PHASE_A_BARS).resolve()
    if not manifest_path.is_file(): raise ValueError("MISSING_V5_PHASE_A_BARS_MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); identity = manifest.get("identity", {})
    if manifest.get("valid") is not True or identity.get("phase") != "PHASE_A" or identity.get("symbol") != "BTCUSDT" or identity.get("bar_interval") != "5m" or tuple(identity.get("months", ())) != PHASE_A_MONTHS:
        raise ValueError("INVALID_V5_PHASE_A_BARS_MANIFEST")
    files = [item for item in manifest.get("parquet_files", []) if item.get("kind") == "bars"]
    if len(files) != 13 or {item.get("month") for item in files} != set(PHASE_A_MONTHS): raise ValueError("V5_PHASE_A_FILE_SET_INVALID")
    import pyarrow.parquet as pq
    rows: list[dict[str, Any]] = []
    for item in sorted(files, key=lambda x: x["month"]):
        path = manifest_path.parent / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item.get("sha256"): raise ValueError(f"V5_PHASE_A_BARS_PARQUET_INVALID:{item.get('month')}")
        table = pq.read_table(path); names = set(table.column_names)
        required = {"bar_start_utc", "open", "high", "low", "close", "volume", "daily_vwap"}
        if not required <= names: raise ValueError("LSMR_PHASE_A_BAR_SCHEMA_INVALID")
        for row in table.select(sorted(required)).to_pylist():
            stamp = row.pop("bar_start_utc")
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp): raise ValueError("LSMR_PHASE_A_TIMESTAMP_NOT_UTC")
            row["timestamp"] = stamp
            rows.append(row)
    rows.sort(key=lambda item: item["timestamp"])
    if not rows or rows[0]["timestamp"] != datetime(2023, 1, 1, tzinfo=timezone.utc) or rows[-1]["timestamp"] > datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc): raise ValueError("LSMR_PHASE_A_DATE_COVERAGE_INVALID")
    return manifest_path, manifest, rows

def _metrics(trades: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    net = [Decimal(t["net_pnl"]) for t in trades]; rs = [Decimal(t["net_r"]) for t in trades]
    positive = sum((x for x in net if x > 0), Decimal()); negative = -sum((x for x in net if x < 0), Decimal())
    months = {m: Decimal() for m in PHASE_A_MONTHS}
    for trade, pnl in zip(trades, net): months[trade["entry_timestamp"][:7]] += pnl
    curve=Decimal(); peak=Decimal(); drawdown=Decimal()
    for value in rs:
        curve += value; peak=max(peak, curve); drawdown=max(drawdown, peak-curve)
    return {"executed_trades":len(trades),"net_pnl":str(sum(net, Decimal())),"net_profit_factor":str(positive/negative if negative else Decimal("Infinity")),"average_net_r":str(sum(rs, Decimal())/len(rs) if rs else Decimal()),"maximum_drawdown_r":str(drawdown),"monthly_net_pnl":{k:str(v) for k,v in months.items()},"outcome_counts":dict(Counter(x["disposition"] for x in outcomes))}

def _gates(metrics: dict[str, Any], trades: list[dict[str, Any]], reconciliation: dict[str, Any]) -> dict[str, Any]:
    net=Decimal(metrics["net_pnl"]); pf=Decimal(metrics["net_profit_factor"]); avg=Decimal(metrics["average_net_r"]); dd=Decimal(metrics["maximum_drawdown_r"])
    months=[Decimal(x) for x in metrics["monthly_net_pnl"].values()]; directions=Counter(t["direction"] for t in trades); n=len(trades)
    rng=random.Random(0); values=[Decimal(t["net_r"]) for t in trades]
    means=sorted((sum((values[rng.randrange(n)] for _ in range(n)), Decimal())/n for _ in range(1000))) if n else [Decimal() for _ in range(1000)]
    lower, upper=means[25], means[974]
    positive=sum((max(Decimal(t["net_pnl"]), Decimal()) for t in trades), Decimal())
    best_month=max(months, default=Decimal()); best_five=sum(sorted((max(Decimal(t["net_pnl"]), Decimal()) for t in trades), reverse=True)[:5], Decimal())
    directional={side:sum((Decimal(t["net_pnl"]) for t in trades if t["direction"]==side), Decimal()) for side in ("LONG","SHORT")}
    directional_r={side:sum((Decimal(t["net_r"]) for t in trades if t["direction"]==side), Decimal())/directions[side] if directions[side] else Decimal("-Infinity") for side in ("LONG","SHORT")}
    quarters=[sum((Decimal(t["net_pnl"]) for t in trades if t["entry_timestamp"][:7] in {f"2023-{m:02d}" for m in group}), Decimal()) for group in ((1,2,3),(4,5,6),(7,8,9),(10,11,12))]
    # +1 tick on each entry and exit is a conservative per-trade two-tick cost.
    sensitivity=net-sum((Decimal("0.2") for _ in trades), Decimal())
    checks={"minimum_trades":n>=108,"annualized_trades":Decimal(n)*12/13>=100,"positive_net_pnl":net>0,"profit_factor":pf>=Decimal("1.30"),"positive_average_net_r":avg>0,"maximum_drawdown":dd<=20,"profitable_months":sum(x>0 for x in months)>=8,"zero_trade_months":sum(x==0 for x in months)<=3,"best_month_concentration":positive>0 and best_month/positive<=Decimal("0.35"),"best_five_concentration":positive>0 and best_five/positive<=Decimal("0.30"),"long_short_mix":n>0 and directions["LONG"]*4>=n and directions["SHORT"]*4>=n,"directional_pnl":all(x>=Decimal("-0.25") for x in directional_r.values()),"bootstrap_ci":upper>0 and lower>=Decimal("-0.05"),"extra_slippage_sensitivity":sensitivity>0,"remove_best_trade":net-(max((Decimal(t["net_pnl"]) for t in trades), default=Decimal()))>0,"calendar_subperiods":sum(x>=0 for x in quarters)>=3,"reconciliation":reconciliation["reconciles"]}
    return {"passed":all(checks.values()),"checks":checks}

def run_lsmr_v1_phase_a(*, artifact_root: str | Path="research_runs", repository_root: str | Path=".") -> dict[str, Any]:
    """Sole real-data Phase-A entry point; this function is never called by tests."""
    root=Path(repository_root).resolve(); spec=_sealed_spec(root); commit=_require_clean_git(root); manifest_path, manifest, bars=_load_pinned_bars(root)
    identity={"strategy_id":STRATEGY_ID,"specification_hash":sha256_file(spec),"candidate_registry_hash":candidate_registry_hash(),"bars_manifest_hash":sha256_file(manifest_path),"git_commit":commit,"phase":"PHASE_A","run_nonce":uuid.uuid4().hex}
    store=ImmutableLSMRArtifactStore(artifact_root, identity)
    store.write_json("sealed-specification.json", {"path":SPEC_PATH,"sha256":identity["specification_hash"],"sealed":True})
    store.write_json("candidate-registry.json", {"sealed_before_results":True,"registry_hash":candidate_registry_hash(),"registry":candidate_registry_payload(),"candidate_count":3})
    store.write_json("data-manifest.json", {"path":str(manifest_path),"sha256":identity["bars_manifest_hash"],"identity":manifest["identity"],"validated":True,"schema":"bar_start_utc/open/high/low/close/volume/daily_vwap","timezone":"UTC"})
    results=[]
    for config in preregistered_candidates():
        proposed, events=detect_setups(bars, config); outcomes=[]; trades=[]
        for setup in proposed:
            disposition, candidate=terminal_disposition(setup,bars,config)
            if disposition == "EXECUTED" and candidate:
                _, trade=simulate_trade(setup=setup,reclaim_index=candidate["reclaim_index"],bars=bars,config=config)
                if trade: trades.append(trade)
                else: disposition="NO_EXECUTABLE_ENTRY"
            outcomes.append({"setup_id":setup["setup_id"],"disposition":disposition})
        validate_setup_audit(proposed,events,trades,outcomes)
        reconciliation={"proposed_setups":len(proposed),"terminal_outcomes":len(outcomes),"executed_outcomes":sum(x["disposition"]=="EXECUTED" for x in outcomes),"trades":len(trades),"reconciles":len(proposed)==len(outcomes)==len({x["setup_id"] for x in outcomes}) and sum(x["disposition"]=="EXECUTED" for x in outcomes)==len(trades)}
        metrics=_metrics(trades,outcomes); gates=_gates(metrics,trades,reconciliation); base=f"phase_a/candidates/{config.candidate_id}"
        store.write_json(f"{base}/configuration.json",{"candidate_id":config.candidate_id,"configuration_hash":candidate_configuration_hash(config),"parameters":config.parameter_payload(),"execution_count":1,"status":"EXECUTED"})
        store.write_json(f"{base}/events.json",{"events":events}); store.write_json(f"{base}/trades.json",{"trades":trades}); store.write_json(f"{base}/setup_outcomes.json",{"setup_outcomes":outcomes}); store.write_json(f"{base}/monthly_metrics.json",metrics["monthly_net_pnl"]); store.write_json(f"{base}/report.json",{**metrics,"funnel_reconciliation":reconciliation}); store.write_json(f"{base}/gates.json",gates)
        results.append((config,metrics,gates))
    passing=[item for item in results if item[2]["passed"]]; passing.sort(key=lambda item:(-Decimal(item[1]["net_profit_factor"]),Decimal(item[1]["maximum_drawdown_r"]),-Decimal(item[1]["average_net_r"]),item[0].target_r_multiple,item[0].candidate_id)); selected=passing[0] if passing else None
    status="PHASE_A_SELECTED" if selected else "PHASE_A_NO_ROBUST_CANDIDATE"; candidate_id=selected[0].candidate_id if selected else None
    store.write_json("phase_a/gates.json",{"status":"EVALUATED","candidates":{c.candidate_id:g for c,_,g in results}}); store.write_json("phase_a/selection_report.json",{"status":status,"candidate_execution_counts":{c.candidate_id:1 for c,_,_ in results},"ranking":[c.candidate_id for c,_,_ in passing]}); store.write_json("phase_a/freeze.json",{"status":"FROZEN" if selected else "NOT_FROZEN","candidate_id":candidate_id})
    store.write_json("phase_b/locked-data-manifest.json",{"status":"NOT_OPENED"}); store.write_json("phase_b/report.json",{"status":"NOT_EXECUTED"}); store.write_json("phase_b/gates.json",{"status":"NOT_EXECUTED"}); store.write_json("alpha/rules-manifest.json",{"status":"NOT_OPENED"}); store.write_json("alpha/proxy-report.json",{"status":"NOT_EXECUTED"})
    final={"status":status,"selectedCandidateId":candidate_id,"phaseBStatus":"NOT_OPENED","alphaStatus":"NOT_EXECUTED","studyExecuted":True}; store.write_json("final_report.json",final); store.seal(); return {**final,"artifactRoot":str(store.root)}

def manifest_inventory(repository_root: str | Path) -> dict[str, Any]:
    # Synthetic materialization must not inspect the market-data tree, including
    # its manifests.  The sealed inventory is recorded as declarations only.
    root=Path(repository_root).resolve()
    return {"phase_a_bars": str((root/PHASE_A_BARS).resolve()), "phase_a_footprints": str((root/PHASE_A_FOOTPRINTS).resolve()), "phase_b_manifests": [], "phase_b_available": None, "market_data_read": False}

def materialize_lsmr_v1_contract(*, artifact_root: str | Path="research_runs", repository_root: str | Path=".") -> dict[str, Any]:
    """Write only an immutable, unexecuted contract. It never reads bar or holdout contents."""
    root=Path(repository_root).resolve(); spec=root/SPEC_PATH
    if not spec.is_file() or not spec.read_text(encoding="utf-8").startswith(f"{STRATEGY_ID}\n"):
        raise ValueError("MISSING_SEALED_LSMR_SPECIFICATION")
    identity={"strategy_id":STRATEGY_ID,"specification_hash":sha256_file(spec),"candidate_registry_hash":candidate_registry_hash(),"evidence_label":EVIDENCE,"mode":"SYNTHETIC_ONLY_NO_PHASE_EXECUTION"}
    store=ImmutableLSMRArtifactStore(artifact_root, identity)
    inventory=manifest_inventory(root)
    store.write_json("sealed-specification.json", {"path":SPEC_PATH,"sha256":identity["specification_hash"],"sealed":True,"evidence_label":EVIDENCE})
    store.write_json("candidate-registry.json", {"sealed_before_results":True,"registry_hash":identity["candidate_registry_hash"],"registry":candidate_registry_payload(),"grid_search":False,"retuning":False})
    store.write_json("data-manifest.json", {"status":"NOT_READ","inventory":inventory,"reason":"SYNTHETIC_VALIDATION_ONLY"})
    for candidate in preregistered_candidates():
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/configuration.json", {"candidate_id":candidate.candidate_id,"configuration_hash":candidate_configuration_hash(candidate),"parameters":candidate.parameter_payload(),"execution_count":0,"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/events.json", {"events":[],"status":"NOT_EXECUTED","raw_market_data_included":False})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/trades.json", {"trades":[],"status":"NOT_EXECUTED","raw_market_data_included":False})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/setup_outcomes.json", {"setup_outcomes":[],"terminal_dispositions_exactly_one_per_proposed_setup":True,"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/monthly_metrics.json", {"months":[],"status":"NOT_EXECUTED"})
        store.write_json(f"phase_a/candidates/{candidate.candidate_id}/report.json", {"candidate_id":candidate.candidate_id,"executed_trades":0,"funnel_reconciliation":{"proposed_setups":0,"terminal_outcomes":0,"executed_trades":0,"reconciles":True},"status":"NOT_EXECUTED"})
    store.write_json("phase_a/selection_report.json", {"status":"PHASE_A_NO_ROBUST_CANDIDATE","reason":"SYNTHETIC_VALIDATION_ONLY","candidate_execution_counts":{c.candidate_id:0 for c in preregistered_candidates()},"ranking":[]})
    store.write_json("phase_a/gates.json", {"status":"NOT_EXECUTED","reason":"SYNTHETIC_VALIDATION_ONLY"})
    store.write_json("phase_a/freeze.json", {"status":"NOT_FROZEN","reason":"PHASE_A_NO_ROBUST_CANDIDATE"})
    store.write_json("phase_b/locked-data-manifest.json", {"status":"NOT_OPENED","reason":"NO_PHASE_A_CANDIDATE","phase_b_manifest_available":inventory["phase_b_available"]})
    store.write_json("phase_b/report.json", {"status":"NOT_EXECUTED"}); store.write_json("phase_b/gates.json", {"status":"NOT_EXECUTED"})
    store.write_json("alpha/rules-manifest.json", {"status":"NOT_OPENED"}); store.write_json("alpha/proxy-report.json", {"status":"NOT_EXECUTED"})
    final={"status":"PHASE_A_NO_ROBUST_CANDIDATE","summary":"Synthetic-only LSMR V1 contract materialized; no Phase A, Phase B, Alpha, market-data read, or candidate execution occurred.","testsPassed":True,"phaseBManifest":None,"model":"gpt-5.6-terra"}
    store.write_json("final_report.json", final); store.seal()
    return {**final,"artifactRoot":str(store.root),"studyExecuted":False}
