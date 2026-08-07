from __future__ import annotations
import csv, hashlib, io, json, zipfile
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .aggregation import completed_utc_bars, validate_1m_bars
from .constants import CANDIDATES, DEVELOPMENT_END, DEVELOPMENT_START, NO_HOLDOUT_LOGICAL_EXPOSURE, STRATEGY_ID
from .manifests import ManifestError, verify_manifest
from .models import Bar, Candidate, ExecutionAssumptions
from .accounting import close_trade
from .execution import _exit, execute_order, process_position, submit_order
from .metrics import gates, metrics
from .reconciliation import reconcile
from .strategy import causal_setups, expire_reason

CUTOFF = time(22, 45)


def guard():
    return {"isolation": NO_HOLDOUT_LOGICAL_EXPOSURE, "development_start": DEVELOPMENT_START, "development_end": DEVELOPMENT_END, "holdout_status": "LOCKED_NOT_OPENED"}


def _candidate(value): return Candidate(**value) if isinstance(value, dict) else value
def _json(value):
    if hasattr(value, "isoformat"): return value.isoformat().replace("+00:00", "Z")
    return str(value)
def _write(root, name, payload):
    target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2, default=_json) + "\n", encoding="utf-8")
def _create_root(root):
    root = Path(root).resolve()
    if root.exists(): raise FileExistsError("FIB09_V2_IMMUTABLE_ARTIFACT_ROOT_COLLISION")
    root.mkdir(parents=True); return root
def _seal(root, identity):
    files = [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob("*")) if p.is_file()]
    _write(root, "integrity-manifest.json", {"identity": identity, "files": files, "manifestSha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})


def _at_or_after_cutoff(stamp: datetime) -> bool:
    return stamp.astimezone(timezone.utc).time() >= CUTOFF


def _close(active, bar, assumptions, trades, events):
    leg = _exit(active, bar.open, active["remaining_quantity"], bar.timestamp, "FORCED_SESSION_EXIT_2245", len(active["legs"]) + 1, assumptions)
    closed = close_trade(active); trades.append(closed)
    events.append({"kind": "FORCED_SESSION_EXIT_2245", "setup_id": active["setup_id"], "order_id": active["order_id"], "trade_id": active["trade_id"], "exit_leg_id": leg["exit_leg_id"], "timestamp": bar.timestamp})
    return closed


def run_candidate(rows: list[Bar], candidate: Candidate | dict, assumptions: ExecutionAssumptions = ExecutionAssumptions()) -> dict:
    """Calculate setups on completed HTF bars but execute only later 1m bars."""
    candidate = _candidate(candidate); rows = validate_1m_bars(rows)
    htf_minutes = 240 if candidate.symbol == "ETH" else 1440
    htf = completed_utc_bars(rows, htf_minutes)
    setups = causal_setups(htf, candidate)
    scheduled = {}
    for setup in setups:
        if "terminal" not in setup:
            # The HTF bar is known only after its last minute has closed.
            scheduled.setdefault(setup["extreme_timestamp"] + __import__("datetime").timedelta(minutes=htf_minutes), []).append(setup)
    events, orders, outcomes, trades, pending = [], [], [], [], []
    active = None; equity = assumptions.opening_equity
    for bar in rows:
        stamp = bar.timestamp.astimezone(timezone.utc)
        # This ordering is sealed: force flat at the OPEN before intrabar logic.
        if stamp.time() == CUTOFF:
            if active is not None:
                closed = _close(active, bar, assumptions, trades, events); equity += closed["net_pnl"]; active = None
            for order in list(pending):
                outcomes.append({"setup_id": order["setup_id"], "disposition": "SESSION_ENTRY_CUTOFF_2245"}); pending.remove(order)
            continue
        for setup in scheduled.get(stamp, []):
            if _at_or_after_cutoff(stamp):
                outcomes.append({"setup_id": setup["setup_id"], "disposition": "SESSION_ENTRY_CUTOFF_2245"}); continue
            order = submit_order(setup, stamp, 2); pending.append(order); orders.append(order)
            events.append({"kind": "ORDER_SUBMITTED", "setup_id": setup["setup_id"], "order_id": order["order_id"], "timestamp": stamp})
        for order in list(pending):
            if _at_or_after_cutoff(stamp):
                outcomes.append({"setup_id": order["setup_id"], "disposition": "SESSION_ENTRY_CUTOFF_2245"}); pending.remove(order); continue
            reason = "ACTIVE_POSITION_BLOCKED" if active is not None else expire_reason(order, bar, candidate)
            if reason:
                outcomes.append({"setup_id": order["setup_id"], "disposition": reason}); pending.remove(order); continue
            trade, rejected = execute_order(order, bar, candidate, equity, assumptions)
            if rejected:
                outcomes.append({"setup_id": order["setup_id"], "disposition": rejected}); pending.remove(order); continue
            if trade:
                if trade["entry_timestamp"].date() != stamp.date(): raise ManifestError("V2_SAME_DAY_ENTRY_FINAL_EXIT_INVARIANT")
                active = trade; pending.remove(order); outcomes.append({"setup_id": order["setup_id"], "disposition": "TRADE_EXECUTED"})
                events.append({"kind": "ORDER_FILLED", "setup_id": order["setup_id"], "order_id": order["order_id"], "trade_id": trade["trade_id"], "timestamp": stamp})
        if active is not None and stamp > active["entry_timestamp"]:
            process_position(active, bar, candidate, assumptions)
            if active["remaining_quantity"] <= 0:
                closed = close_trade(active)
                if closed["legs"][-1]["timestamp"].date() != active["entry_timestamp"].date(): raise ManifestError("V2_SAME_DAY_ENTRY_FINAL_EXIT_INVARIANT")
                trades.append(closed); equity += closed["net_pnl"]; active = None
    if active is not None:
        # Missing 22:45 bar is a data failure, never a substituted close.
        raise ManifestError("V2_REQUIRED_2245_FORCE_EXIT_BAR_MISSING")
    for order in pending: outcomes.append({"setup_id": order["setup_id"], "disposition": "SESSION_OR_DATA_END"})
    for setup in setups:
        if setup.get("terminal") and not any(x["setup_id"] == setup["setup_id"] for x in outcomes): outcomes.append({"setup_id": setup["setup_id"], "disposition": setup["terminal"]})
    result_metrics = metrics(trades, assumptions.opening_equity)
    rec = reconcile(setups, outcomes, orders, trades, assumptions.opening_equity, final_equity=equity, events=events)
    forced = sum(1 for leg in (leg for trade in trades for leg in trade["legs"]) if leg["reason"] == "FORCED_SESSION_EXIT_2245")
    return {"events": events, "setups": setups, "setup_outcomes": outcomes, "orders": orders, "trades": trades, "partial_exits": [leg for trade in trades for leg in trade["legs"]], "metrics": {**result_metrics, "forced_session_exit_count": forced}, "reconciliation": rec, "gates": gates(result_metrics, trades, rec["reconciles"]), **guard()}


def _materialize_result(root, results, mode):
    for row, item in results:
        base = Path("candidates") / row["candidate_id"]
        for key, name in (("events", "events.json"), ("setup_outcomes", "setup-outcomes.json"), ("orders", "orders.json"), ("trades", "trades.json"), ("partial_exits", "partial-exits.json"), ("metrics", "monthly-metrics.json"), ("gates", "gates.json"), ("reconciliation", "reconciliation.json")):
            _write(root, base / name, item[key])
    _write(root, "candidate-registry.json", {"strategy_id": STRATEGY_ID, "candidates": CANDIDATES})
    _write(root, "development-result.json", {"status": mode, "candidates": [{"candidate_id": r["candidate_id"], "reconciles": item["reconciliation"]["reconciles"]} for r, item in results], **guard()})
    _write(root, "final-report.json", {"status": mode, **guard()}); _seal(root, {"strategy_id": STRATEGY_ID, "mode": mode})


def materialize_synthetic(*, artifact_root, repository_root):
    root = _create_root(artifact_root)
    _write(root, "candidate-registry.json", {"strategy_id": STRATEGY_ID, "candidates": CANDIDATES, "synthetic_only": True})
    _write(root, "final-report.json", {"status": "SYNTHETIC_ONLY", **guard()}); _seal(root, {"strategy_id": STRATEGY_ID, "mode": "SYNTHETIC"})
    return {"artifact_root": str(root), **guard()}


def run_synthetic(*, bars_by_candidate, artifact_root, repository_root):
    root = _create_root(artifact_root); results = [(row, run_candidate(bars_by_candidate.get(row["candidate_id"], []), row)) for row in CANDIDATES]
    _materialize_result(root, results, "SYNTHETIC"); return {"artifact_root": str(root), "candidates": len(results), **guard()}


def development_diagnostic(*, eth_manifest, btc_manifest):
    return {"eth": verify_manifest(eth_manifest, symbol="ETHUSDT"), "btc": verify_manifest(btc_manifest, symbol="BTCUSDT"), **guard()}


def _development_only(rows):
    """Private predicate boundary: callers can receive development bars only."""
    start = datetime(2022, 1, 1, tzinfo=timezone.utc); end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [bar for bar in rows if start <= bar.timestamp < end]


def _load_development_rows(manifest_path, repository_root):
    """Future execution loader; does not expose source rows or holdout rows."""
    # Authenticate every declared immutable archive before opening any archive.
    verify_manifest(manifest_path, partition_root=(Path(repository_root) / "data" / "fib_prospective_v2").resolve())
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = []
    for part in manifest["partitions"]:
        first = datetime.fromisoformat(part["firstUtcTimestamp"].replace("Z", "+00:00"))
        if first >= datetime(2025, 1, 1, tzinfo=timezone.utc):
            break
        payload = Path(repository_root) / "data" / "fib_prospective_v2" / part["path"] / part["file"]
        if not payload.is_file(): raise ManifestError("V2_SEALED_PARTITION_PAYLOAD_MISSING")
        if hashlib.sha256(payload.read_bytes()).hexdigest() != part["sha256"]: raise ManifestError("V2_SEALED_PARTITION_HASH_MISMATCH")
        with zipfile.ZipFile(payload) as archive:
            names = archive.namelist()
            if len(names) != 1: raise ManifestError("V2_SEALED_PARTITION_ARCHIVE_INVALID")
            reader = csv.reader(io.TextIOWrapper(archive.open(names[0]), encoding="utf-8"))
            for raw in reader:
                if not raw or raw[0] == "open_time": continue
                stamp = datetime.fromtimestamp(int(raw[0]) / 1000, tz=timezone.utc)
                rows.append(Bar(stamp, *(Decimal(raw[i]) for i in (1, 2, 3, 4, 5))))
    return validate_1m_bars(_development_only(rows))


def run_development(*, eth_manifest, btc_manifest, artifact_root, repository_root):
    # The deliberately private loader/execution boundary is wired for the authorized future run.
    if not Path(repository_root).is_absolute() or not Path(repository_root).is_dir(): raise ManifestError("V2_REPOSITORY_ROOT_MUST_BE_ABSOLUTE_EXISTING")
    if not Path(artifact_root).is_absolute(): raise ManifestError("V2_ARTIFACT_ROOT_MUST_BE_ABSOLUTE")
    if Path(artifact_root).exists(): raise FileExistsError("FIB09_V2_IMMUTABLE_ARTIFACT_ROOT_COLLISION")
    development_diagnostic(eth_manifest=eth_manifest, btc_manifest=btc_manifest)
    eth = _load_development_rows(eth_manifest, repository_root)
    btc = _load_development_rows(btc_manifest, repository_root)
    results = []
    for row in CANDIDATES:
        item = run_candidate(eth if row["symbol"] == "ETH" else btc, row)
        if not item["reconciliation"]["reconciles"]: raise ManifestError("V2_RECONCILIATION_FAILED")
        results.append((row, item))
    root = _create_root(artifact_root); _materialize_result(root, results, "DEVELOPMENT_EXECUTED")
    return {"artifact_root": str(root), "candidate_count": len(results), **guard()}


def run_holdout(**kwargs): raise ManifestError("LOCKED_HOLDOUT_NOT_AUTHORIZED")
