from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import CANDIDATES, SPEC_HASH, Bar, HTFLevelLiquidityFVG, materialize_synthetic, phase_a_hard_gates, reconcile_events

PHASE_A_MANIFEST = Path(r"C:\Users\sandr\Trading-Bot-Fib\data\imbalance_vwap_ride\v5\bars\BTCUSDT\phase_a\6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f\manifest.json")

def materialize_htf_lfvg_v1_contract(*, artifact_root: str, repository_root: str) -> dict:
    return materialize_synthetic(Path(artifact_root), Path(repository_root))

def _utc(value: Any) -> datetime:
    parsed=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0: raise ValueError("Phase-A bars must have UTC timestamps")
    return parsed.astimezone(timezone.utc)

def _load_explicit_phase_a_bars(path: Path) -> list[Bar]:
    """The real interface accepts only a self-contained, explicitly shaped manifest.

    The seal deliberately leaves historical-manifest schema decisions unresolved;
    this prevents format guessing and fails closed rather than discovering files.
    """
    raw=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw,dict) or not isinstance(raw.get("bars"),list):
        raise ValueError("Phase-A manifest contract requires a self-contained 'bars' array; no data-path discovery is permitted")
    bars=[]
    for row in raw["bars"]:
        if not isinstance(row,dict): raise ValueError("invalid Phase-A bar record")
        bars.append(Bar(_utc(row["time"]),float(row["open"]),float(row["high"]),float(row["low"]),float(row["close"]),float(row.get("volume",0)),str(row.get("id",row["time"]))))
    start=datetime(2023,1,1,tzinfo=timezone.utc); end=datetime(2024,2,1,tzinfo=timezone.utc)
    if any(not(start<=b.time<end) for b in bars): raise ValueError("Phase-A bar lies outside sealed interval")
    return bars

def run_htf_lfvg_v1_phase_a(*, phase_a_bars_manifest: str, artifact_root: str, repository_root: str) -> dict:
    """Explicit-only deterministic executor; it has no Phase-B, fallback, or discovery path."""
    repo=Path(repository_root).resolve(); spec=repo/".smithers/specs/htf-level-liquidity-fvg-v1.md"
    import hashlib
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper()!=SPEC_HASH: raise RuntimeError("sealed specification hash mismatch")
    supplied=Path(phase_a_bars_manifest).resolve()
    # The explicit contract is identity, not a glob or an inferred equivalent.
    if str(supplied).lower()!=str(PHASE_A_MANIFEST).lower(): raise ValueError("Phase-A manifest does not match sealed unopened input contract")
    output=Path(artifact_root)
    if output.exists(): raise FileExistsError("immutable artifact collision")
    bars=_load_explicit_phase_a_bars(supplied)
    output.mkdir(parents=True)
    reports=[]
    for candidate_id in CANDIDATES:
        engine=HTFLevelLiquidityFVG(candidate_id,run_id=f"phase-a-{candidate_id}")
        for bar in bars: engine.feed(bar)
        # End-of-data is an exit reason, never a second setup disposition.
        if engine.position: engine._exit(bars[-1],bars[-1].close,engine.position["remaining"],"FORCED_END_OF_DATA_EXIT")
        if engine.setup: engine._finish(bars[-1].time, __import__("research_pipeline.htf_level_liquidity_fvg.core",fromlist=["TerminalDisposition"]).TerminalDisposition.MSS_WINDOW_EXPIRED)
        reconcile_events(engine.events,engine.outcomes,engine.trades)
        net=sum(x["net_pnl"] for x in engine.trades); gates=phase_a_hard_gates({"executed_trades":len(engine.trades),"days":396,"net_pnl":net,"immutable_artifacts":True,"funnel_reconciled":True})
        reports.append({"candidate_id":candidate_id,"executed_trades":len(engine.trades),"net_pnl":net,"gates":gates})
    (output/"phase-a-result.json").write_text(json.dumps({"specification_hash":SPEC_HASH,"candidates":reports,"phase_b":"NOT_OPENED"},sort_keys=True,indent=2)+"\n",encoding="utf-8")
    passed=[x for x in reports if x["gates"]["passed"]]
    return {"status":"FROZEN" if len(passed)==1 else "PHASE_A_NO_ROBUST_CANDIDATE","candidate_reports":reports,"phase_b":"NOT_OPENED","realStudyExecuted":True,"model":"gpt-5.6-terra"}
