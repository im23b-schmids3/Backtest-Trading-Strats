from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .core import CANDIDATES, SPEC_HASH, Bar, HTFLevelLiquidityFVG, materialize_synthetic, phase_a_hard_gates, reconcile_events


PHASE_A_MANIFEST = Path(r"C:\Users\sandr\Trading-Bot-Fib\data\imbalance_vwap_ride\v5\bars\BTCUSDT\phase_a\6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f\manifest.json")
PHASE_A_MONTHS = tuple([f"2023-{month:02d}" for month in range(1, 13)] + ["2024-01"])
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def materialize_htf_lfvg_v1_contract(*, artifact_root: str, repository_root: str) -> dict:
    """Synthetic-only artifact fixture entry point; it never opens market data."""
    return materialize_synthetic(Path(artifact_root), Path(repository_root))


def _utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Phase-A parquet timestamps must be datetimes")
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("Phase-A bars must have UTC timestamps")
    return value.astimezone(timezone.utc)


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, number = (int(part) for part in month.split("-"))
    start = datetime(year, number, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if number == 12 else datetime(year, number + 1, 1, tzinfo=timezone.utc)
    return start, end


def _safe_partition_path(manifest_path: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Phase-A partition path must be a non-empty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("Phase-A partition path is not safe and relative")
    base = manifest_path.parent.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Phase-A partition path escapes manifest directory") from exc
    return resolved


def _validated_phase_a_partitions(manifest_path: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Phase-A manifest is unreadable") from exc
    if not isinstance(raw, dict) or raw.get("valid") is not True:
        raise ValueError("Phase-A manifest must declare valid=true")
    identity = raw.get("identity")
    if not isinstance(identity, dict) or identity.get("phase") != "PHASE_A" or identity.get("symbol") != "BTCUSDT" or identity.get("bar_interval") != "5m" or identity.get("months") != list(PHASE_A_MONTHS):
        raise ValueError("Phase-A manifest identity does not match the sealed contract")
    files = raw.get("parquet_files")
    if not isinstance(files, list):
        raise ValueError("Phase-A manifest must declare parquet_files")
    if len(files) != len(PHASE_A_MONTHS):
        raise ValueError("Phase-A manifest must declare exactly one partition per sealed month")
    partitions: list[tuple[dict[str, Any], Path]] = []
    for expected_month, item in zip(PHASE_A_MONTHS, files):
        if not isinstance(item, dict) or item.get("kind") != "bars" or item.get("month") != expected_month:
            raise ValueError(f"Phase-A manifest requires the declared bars partition for {expected_month}")
        if not isinstance(item.get("row_count"), int) or item["row_count"] < 0 or not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError("Phase-A partition count or SHA256 is invalid")
        partitions.append((item, _safe_partition_path(manifest_path, item.get("relative_path"))))
    if not isinstance(raw.get("five_minute_bar_count"), int) or raw["five_minute_bar_count"] < 0:
        raise ValueError("Phase-A manifest total row count is invalid")
    return raw, partitions


def _load_explicit_phase_a_bars(path: Path) -> list[Bar]:
    """Load only the sealed V5 parquet partitions; embedded bars are never real input."""
    manifest_path = path.resolve()
    raw, partitions = _validated_phase_a_partitions(manifest_path)
    # Hash every declared file before opening any parquet rows.
    for item, partition_path in partitions:
        if not partition_path.is_file():
            raise ValueError(f"Phase-A partition is missing: {item['month']}")
        if hashlib.sha256(partition_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Phase-A partition SHA256 mismatch: {item['month']}")

    bars: list[Bar] = []
    total = 0
    previous: datetime | None = None
    for item, partition_path in partitions:
        try:
            table = pq.read_table(partition_path, columns=["bar_start_utc", "open", "high", "low", "close", "volume"])
        except Exception as exc:
            raise ValueError("Phase-A parquet does not have the V5 bar schema") from exc
        if table.num_rows != item["row_count"]:
            raise ValueError(f"Phase-A partition row count mismatch: {item['month']}")
        required = {"bar_start_utc", "open", "high", "low", "close", "volume"}
        if not required.issubset(table.column_names):
            raise ValueError("Phase-A parquet does not have the V5 bar schema")
        start, end = _month_bounds(item["month"])
        for row in table.to_pylist():
            timestamp = _utc(row["bar_start_utc"])
            if not start <= timestamp < end:
                raise ValueError(f"Phase-A bar is outside its sealed month: {item['month']}")
            if previous is not None and timestamp <= previous:
                raise ValueError("Phase-A timestamps must be strictly increasing in declared order")
            previous = timestamp
            bars.append(Bar(timestamp, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), timestamp.isoformat()))
        total += table.num_rows
    if total != raw["five_minute_bar_count"]:
        raise ValueError("Phase-A total row count mismatch")
    # Validation above is deliberately against manifest order, before this final ordering.
    return sorted(bars, key=lambda bar: bar.time)


def load_synthetic_embedded_bars(manifest: dict[str, Any]) -> list[Bar]:
    """Synthetic-test helper. Real Phase-A loading intentionally cannot call this."""
    rows = manifest.get("bars")
    if not isinstance(rows, list):
        raise ValueError("synthetic fixture requires a bars array")
    return [Bar(_utc(datetime.fromisoformat(str(row["time"]).replace("Z", "+00:00"))), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0)), str(row.get("id", row["time"]))) for row in rows]


def phase_a_diagnostic(path: Path = PHASE_A_MANIFEST) -> dict[str, Any]:
    """Read-only integrity summary for the sealed data contract."""
    bars = _load_explicit_phase_a_bars(path)
    times = [bar.time for bar in bars]
    return {"partitionCount": len(PHASE_A_MONTHS), "totalRows": len(bars), "minimumTimestamp": times[0].isoformat(), "maximumTimestamp": times[-1].isoformat(), "duplicateTimestamps": sum(right == left for left, right in zip(times, times[1:])), "nonIncreasingTimestamps": sum(right <= left for left, right in zip(times, times[1:])), "gapsGreaterThanFiveMinutes": sum(right - left > __import__("datetime").timedelta(minutes=5) for left, right in zip(times, times[1:]))}


def run_htf_lfvg_v1_phase_a(*, phase_a_bars_manifest: str, artifact_root: str, repository_root: str) -> dict:
    """Explicit-only deterministic executor; it has no Phase-B, fallback, or discovery path."""
    repo = Path(repository_root).resolve(); spec = repo / ".smithers/specs/htf-level-liquidity-fvg-v1.md"
    if hashlib.sha256(spec.read_bytes()).hexdigest().upper() != SPEC_HASH: raise RuntimeError("sealed specification hash mismatch")
    supplied = Path(phase_a_bars_manifest).resolve()
    if supplied != PHASE_A_MANIFEST.resolve(): raise ValueError("Phase-A manifest does not match sealed unopened input contract")
    output = Path(artifact_root)
    if output.exists(): raise FileExistsError("immutable artifact collision")
    bars = _load_explicit_phase_a_bars(supplied)
    output.mkdir(parents=True)
    reports = []
    for candidate_id in CANDIDATES:
        engine = HTFLevelLiquidityFVG(candidate_id, run_id=f"phase-a-{candidate_id}")
        for bar in bars: engine.feed(bar)
        if engine.position: engine._exit(bars[-1], bars[-1].close, engine.position["remaining"], "FORCED_END_OF_DATA_EXIT")
        if engine.setup: engine._finish(bars[-1].time, __import__("research_pipeline.htf_level_liquidity_fvg.core", fromlist=["TerminalDisposition"]).TerminalDisposition.MSS_WINDOW_EXPIRED)
        reconcile_events(engine.events, engine.outcomes, engine.trades)
        net = sum(x["net_pnl"] for x in engine.trades); gates = phase_a_hard_gates({"executed_trades": len(engine.trades), "days": 396, "net_pnl": net, "immutable_artifacts": True, "funnel_reconciled": True})
        reports.append({"candidate_id": candidate_id, "executed_trades": len(engine.trades), "net_pnl": net, "gates": gates})
    (output / "phase-a-result.json").write_text(json.dumps({"specification_hash": SPEC_HASH, "candidates": reports, "phase_b": "NOT_OPENED"}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    passed = [x for x in reports if x["gates"]["passed"]]
    return {"status": "FROZEN" if len(passed) == 1 else "PHASE_A_NO_ROBUST_CANDIDATE", "candidate_reports": reports, "phase_b": "NOT_OPENED", "realStudyExecuted": True, "model": "gpt-5.6-terra"}
