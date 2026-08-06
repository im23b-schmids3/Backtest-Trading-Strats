"""V2 command surfaces, including the sealed deterministic Phase-A executor."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import CANDIDATES, SPEC_HASH, HTFLevelLiquidityFVG, TerminalDisposition, materialize_synthetic, phase_a_hard_gates, reconcile_events
from research_pipeline.htf_level_liquidity_fvg.runner import _funnel_candidate_summary, _load_explicit_phase_a_bars, load_synthetic_embedded_bars

PHASE_A_MANIFEST = r"C:\Users\sandr\Trading-Bot-Fib\data\imbalance_vwap_ride\v5\bars\BTCUSDT\phase_a\6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f\manifest.json"
PHASE_A_TOTAL_ROWS = 113_757
PHASE_A_DAYS = 396

def materialize_htf_lfvg_v2_contract(*, artifact_root: str, repository_root: str) -> dict[str, Any]:
    return materialize_synthetic(Path(artifact_root), Path(repository_root))

def synthetic_funnel_diagnostic(synthetic_manifest: str) -> dict[str, Any]:
    """Read-only: accepts synthetic JSON bars only and never writes artifacts."""
    raw = json.loads(Path(synthetic_manifest).read_text(encoding="utf-8"))
    if raw.get("synthetic_only") is not True:
        raise ValueError("diagnostic requires synthetic_only fixture")
    bars = load_synthetic_embedded_bars(raw)
    reports = []
    for cid in CANDIDATES:
        engine = HTFLevelLiquidityFVG(cid, run_id=f"diagnostic-v2-{cid}")
        for bar in bars: engine.feed(bar)
        if engine.setup:
            engine._finish(bars[-1].time if bars else __import__("datetime").datetime.now(__import__("datetime").timezone.utc), TerminalDisposition.MSS_WINDOW_EXPIRED)
        reconcile_events(engine.events, engine.outcomes, engine.trades)
        reports.append({"candidateId": cid, "proposedSetups": len(engine.outcomes), "executedTrades": len(engine.trades), "terminalDispositionCounts": {d.value: sum(o["terminal_disposition"] == d.value for o in engine.outcomes) for d in TerminalDisposition}})
    return {"candidateSummaries": reports, "realPhaseARun": False, "phaseBRun": False, "alphaRun": False, "artifactWritten": False, "syntheticOnly": True}

def _require_sealed_input(phase_a_bars_manifest: str) -> Path:
    supplied = Path(phase_a_bars_manifest)
    if not supplied.is_absolute():
        raise ValueError("Phase-A manifest path must be absolute")
    if supplied.resolve() != Path(PHASE_A_MANIFEST).resolve():
        raise ValueError("Phase-A manifest does not match sealed unopened input contract")
    return supplied.resolve()

def _load_sealed_v5_phase_a_bars(manifest_path: Path):
    """Reuse the proven V1 V5 parquet loader, adding V2's sealed total."""
    bars = _load_explicit_phase_a_bars(manifest_path)
    if len(bars) != PHASE_A_TOTAL_ROWS:
        raise ValueError("Phase-A total row count does not match sealed V5 contract")
    return bars

def _trade_metrics(engine: HTFLevelLiquidityFVG) -> dict[str, Any]:
    outcomes = {item["setup_id"]: item for item in engine.outcomes}; trades = engine.trades
    net = [float(item["net_pnl"]) for item in trades]; gross_profit = sum(value for value in net if value > 0); gross_loss = -sum(value for value in net if value < 0)
    # JSON artifacts cannot encode infinity; a no-loss sample is not a
    # sufficient finite profit-factor observation for the sealed hard gate.
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    net_r: list[float] = []; monthly: dict[str, float] = defaultdict(float)
    exit_time = {event.trade_id: event.time for event in engine.events if event.decision == "EXIT_FILLED" and event.trade_id}
    equity = peak = drawdown = 0.0
    for trade in trades:
        outcome = outcomes[trade["setup_id"]]; risk = abs(float(outcome["entry_price"]) - float(outcome["stop"])) * float(trade["quantity"])
        if risk > 0: net_r.append(float(trade["net_pnl"]) / risk)
        equity += float(trade["net_pnl"]); peak = max(peak, equity); drawdown = max(drawdown, peak - equity)
        timestamp = exit_time.get(trade["trade_id"])
        if timestamp: monthly[timestamp[:7]] += float(trade["net_pnl"])
    directions = Counter(str(item["direction"]) for item in trades)
    long_r = [value for value, trade in zip(net_r, trades) if trade["direction"] == "LONG"]; short_r = [value for value, trade in zip(net_r, trades) if trade["direction"] == "SHORT"]
    positive_months = [value for value in monthly.values() if value > 0]; positive_total = sum(positive_months)
    average_absolute_r = sum(abs(value) for value in net_r) / len(net_r) if net_r else 0.0
    return {"executed_trades": len(trades), "annualized_trades": len(trades) * 365.0 / PHASE_A_DAYS, "net_pnl": sum(net), "net_profit_factor": profit_factor, "average_net_r": sum(net_r) / len(net_r) if net_r else 0.0, "maximum_drawdown_r": drawdown / average_absolute_r if average_absolute_r else 0.0, "profit_factor": profit_factor, "max_drawdown_r": drawdown / average_absolute_r if average_absolute_r else 0.0, "profitable_months": len(positive_months), "zero_trade_months": 13 - len(monthly), "best_month_positive_pnl_share": max(positive_months, default=0.0) / positive_total if positive_total else 1.0, "best_five_positive_pnl_share": sum(sorted(positive_months, reverse=True)[:5]) / positive_total if positive_total else 1.0, "long_trades": directions["LONG"], "short_trades": directions["SHORT"], "long_share": directions["LONG"] / len(trades) if trades else 0.0, "short_share": directions["SHORT"] / len(trades) if trades else 0.0, "long_average_net_r": sum(long_r) / len(long_r) if long_r else -99.0, "short_average_net_r": sum(short_r) / len(short_r) if short_r else -99.0, "one_tick_stress_positive": False, "best_trade_removal_positive": False, "bootstrap_median_mean_net_r": 0.0, "bootstrap_lower_95_net_r": -99.0, "nonnegative_2023_subperiods": 0}

def run_htf_lfvg_v2_phase_a(*, phase_a_bars_manifest: str, artifact_root: str, repository_root: str) -> dict[str, Any]:
    """Execute the three sealed V2 candidates once, with no Phase-B/Alpha path."""
    repo = Path(repository_root)
    if not repo.is_absolute(): raise ValueError("repository root must be absolute")
    spec = repo.resolve() / ".smithers/specs/htf-level-liquidity-fvg-v2-relaxed-mss.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper() != SPEC_HASH: raise RuntimeError("sealed specification hash mismatch")
    manifest = _require_sealed_input(phase_a_bars_manifest); output = Path(artifact_root)
    if not output.is_absolute(): raise ValueError("artifact root must be absolute")
    if output.exists(): raise FileExistsError("immutable artifact collision")
    bars = _load_sealed_v5_phase_a_bars(manifest); output.mkdir(parents=True)
    reports = []
    for candidate_id in CANDIDATES:
        engine = HTFLevelLiquidityFVG(candidate_id, run_id=f"phase-a-v2-{candidate_id}")
        for bar in bars: engine.feed(bar)
        if engine.position: engine._exit(bars[-1], bars[-1].close, engine.position["remaining"], "FORCED_END_OF_DATA_EXIT")
        if engine.setup: engine._finish(bars[-1].time, TerminalDisposition.MSS_WINDOW_EXPIRED)
        reconcile_events(engine.events, engine.outcomes, engine.trades)
        metrics = _trade_metrics(engine); funnel = _funnel_candidate_summary(candidate_id, engine.events, engine.outcomes, engine.trades)
        gates = phase_a_hard_gates({**metrics, "immutable_artifacts": True, "funnel_reconciled": funnel["fullyReconciled"]})
        reports.append({"candidate_id": candidate_id, "executed_trades": metrics["executed_trades"], "annualized_trades": metrics["annualized_trades"], "net_pnl": metrics["net_pnl"], "net_profit_factor": metrics["net_profit_factor"], "average_net_r": metrics["average_net_r"], "maximum_drawdown_r": metrics["maximum_drawdown_r"], "long_trades": metrics["long_trades"], "short_trades": metrics["short_trades"], "funnel_reconciliation": funnel, "gate_results": gates, "terminal_disposition_counts": funnel["terminalDispositionCounts"], "phase_b_status": "NOT_OPENED"})
    payload = {"schema_version": "HTFLevelLiquidityFVG.BTC_LONG_SHORT_V2_RELAXED_MSS.phase-a-result.v1", "specification_hash": SPEC_HASH, "phase_a_total_rows": PHASE_A_TOTAL_ROWS, "candidates": reports, "phase_b_status": "NOT_OPENED", "real_phase_a_executed": True, "phase_b_executed": False, "alpha_executed": False}
    (output / "phase-a-result.json").write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "COMPLETED", "artifact_root": str(output), "phase_a_result": "phase-a-result.json", "phase_b_status": "NOT_OPENED", "realPhaseAExecuted": True, "phaseBExecuted": False, "alphaExecuted": False}
