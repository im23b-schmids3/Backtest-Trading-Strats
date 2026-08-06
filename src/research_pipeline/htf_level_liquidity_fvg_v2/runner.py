"""V2 command surfaces.  The real Phase-A entry point is intentionally locked."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from .core import CANDIDATES, SPEC_HASH, HTFLevelLiquidityFVG, TerminalDisposition, materialize_synthetic, reconcile_events
from research_pipeline.htf_level_liquidity_fvg.runner import load_synthetic_embedded_bars

PHASE_A_MANIFEST = r"C:\Users\sandr\Trading-Bot-Fib\data\imbalance_vwap_ride\v5\bars\BTCUSDT\phase_a\6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f\manifest.json"

def materialize_htf_lfvg_v2_contract(*, artifact_root: str, repository_root: str) -> dict[str, Any]:
    return materialize_synthetic(Path(artifact_root), Path(repository_root))

def synthetic_funnel_diagnostic(synthetic_manifest: str) -> dict[str, Any]:
    """Read-only: accepts synthetic JSON bars only and never writes artifacts."""
    raw = json.loads(Path(synthetic_manifest).read_text(encoding="utf-8"))
    if raw.get("synthetic_only") is not True: raise ValueError("diagnostic requires synthetic_only fixture")
    bars = load_synthetic_embedded_bars(raw); reports = []
    for cid in CANDIDATES:
        e = HTFLevelLiquidityFVG(cid, run_id=f"diagnostic-v2-{cid}")
        for bar in bars: e.feed(bar)
        if e.setup: e._finish(bars[-1].time if bars else __import__('datetime').datetime.now(__import__('datetime').timezone.utc), TerminalDisposition.MSS_WINDOW_EXPIRED)
        reconcile_events(e.events, e.outcomes, e.trades)
        reports.append({"candidateId":cid,"proposedSetups":len(e.outcomes),"executedTrades":len(e.trades),"terminalDispositionCounts":{d.value:sum(o["terminal_disposition"]==d.value for o in e.outcomes) for d in TerminalDisposition}})
    return {"candidateSummaries":reports,"realPhaseARun":False,"phaseBRun":False,"alphaRun":False,"artifactWritten":False,"syntheticOnly":True}

def run_htf_lfvg_v2_phase_a(*, phase_a_bars_manifest: str, artifact_root: str, repository_root: str) -> dict[str, Any]:
    """Reserved future contract: fail before opening any market-data path."""
    spec = Path(repository_root).resolve()/".smithers/specs/htf-level-liquidity-fvg-v2-relaxed-mss.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper() != SPEC_HASH: raise RuntimeError("sealed specification hash mismatch")
    if str(Path(phase_a_bars_manifest)) != PHASE_A_MANIFEST: raise ValueError("Phase-A manifest does not match sealed unopened input contract")
    if Path(artifact_root).exists(): raise FileExistsError("immutable artifact collision")
    raise RuntimeError("real Phase-A execution is locked; no market data was opened")
