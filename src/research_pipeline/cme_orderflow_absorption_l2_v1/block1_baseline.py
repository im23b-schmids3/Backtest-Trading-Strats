"""Split-aware Block 1 baseline orchestration.

This module composes the existing Asia, Europe, and corrected Berlin/NY
implementations.  It owns no signal or execution logic.  The default runner
is deliberately limited to the sealed Block 1 date sets and publishes one
immutable run root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import all_period_weight_q_research as all_period
from . import asia_w04_replay as asia
from . import berlin_hardflat_all_period as berlin
from . import europe_w04_replay as europe
from . import historical_runner as historical


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_BLOCK1_BASELINE"
RUN_ID = "BLOCK1_TRAIN_BASELINE"
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_BLOCK1_TRAIN_BASELINE")
FORMAT_VERSION = "BLOCK1_BASELINE_ARTIFACTS_V1"
NY_SOURCE_MANIFEST = Path(
    "data/cme_orderflow_absorption_l2_v3/dec2025_jan2026/acquisition-manifest.json"
)
NY_DATE_SUBSET_ROOT = berlin.TRAIN_DATE_SUBSET_ROOT

TRAIN_DATES: tuple[str, ...] = (
    "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05",
    "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19",
    "2025-12-22", "2025-12-23", "2025-12-29",
    "2025-12-30", "2026-01-02", "2026-01-05", "2026-01-06",
    "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13",
    "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-20", "2026-01-21",
    "2026-01-22", "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28",
    "2026-01-29", "2026-01-30",
)
VALIDATION_DATES: tuple[str, ...] = (
    "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11",
    "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-18",
    "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22", "2026-06-24",
    "2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30", "2026-07-01",
    "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
    "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16",
    "2026-07-17", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
)
FINAL_OOS_PREFIX = "2026-09"
TRAIN_PROFILE_ONLY_CONTEXT_DATES: tuple[str, ...] = ("2025-12-24", "2025-12-26", "2025-12-31")
EXCLUDED_DATES: tuple[dict[str, str], ...] = (
    {
        "date": "2025-12-01",
        "reason": "PRIOR_ASIA_EUROPE_PROFILE_DEPENDENCY_INCOMPLETE",
    },
    {
        "date": "2025-12-24",
        "reason": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
    },
    {
        "date": "2025-12-26",
        "reason": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
    },
    {
        "date": "2025-12-31",
        "reason": "ASIA_HARD_FLAT_FRESH_BBO_EVIDENCE_UNAVAILABLE",
    },
)


class Block1BaselineError(RuntimeError):
    """The split or one of its immutable inputs is not safe to replay."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    part.replace(path)


def split_dates(split: str) -> tuple[str, ...]:
    if split == "train":
        return TRAIN_DATES
    if split == "validation":
        return VALIDATION_DATES
    raise Block1BaselineError(f"unknown Block 1 split: {split}")


def validate_split(split: str, dates: Sequence[str] | None = None) -> dict[str, Any]:
    expected = split_dates(split)
    observed = expected if dates is None else tuple(str(day) for day in dates)
    if observed != expected:
        raise Block1BaselineError(f"{split} dates do not exactly match the sealed Block 1 set")
    if len(observed) != len(set(observed)):
        raise Block1BaselineError(f"{split} contains duplicate dates")
    if set(TRAIN_DATES) & set(VALIDATION_DATES):
        raise Block1BaselineError("Block 1 train/validation overlap is non-empty")
    if any(day.startswith(FINAL_OOS_PREFIX) for day in (*TRAIN_DATES, *VALIDATION_DATES)):
        raise Block1BaselineError("final September 2026 OOS date reached Block 1")
    return {
        "split": split,
        "dates": list(observed),
        "count": len(observed),
        "excluded_dates": [dict(row) for row in EXCLUDED_DATES],
        "train_validation_overlap": [],
        "final_oos_accessed": False,
    }


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        status = []
    return {"commit": commit, "working_tree_dirty": bool(status), "status_entry_count": len(status)}


def _config_integrity() -> dict[str, Any]:
    asia_contract = asia.frozen_semantic_contracts()[1]
    europe_contract = europe.frozen_semantic_contracts()[1]
    ny_contract = berlin.EXECUTION_CONTRACT
    configs = {
        "asia_w04": {
            "strategy_id": asia.STRATEGY_ID,
            "config_id": asia.CONFIG_ID,
            "weights": {key: str(value) for key, value in asia.W04_WEIGHTS.items()},
            "quality_threshold": str(asia.QUALITY_THRESHOLD),
            "contract_hash": _canonical_hash(asia_contract),
        },
        "europe_w04": {
            "strategy_id": europe.STRATEGY_ID,
            "config_id": europe.CONFIG_ID,
            "weights": {key: str(value) for key, value in europe.W04_WEIGHTS.items()},
            "quality_threshold": str(europe.QUALITY_THRESHOLD),
            "contract_hash": _canonical_hash(europe_contract),
        },
        "ny_berlin": {
            "strategy_id": berlin.STRATEGY_ID,
            "contract_hash": berlin.CONTRACT_SHA256,
            "historical_v3_contract_sha256": berlin.HISTORICAL_V3_CONTRACT_SHA256,
            "execution_contract": ny_contract,
        },
    }
    return {"configs": configs, "config_hashes": {name: _canonical_hash(value) for name, value in configs.items()}}


def _required_source_paths(repository_root: Path) -> tuple[Path, ...]:
    return (
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT/summary.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT/session-coverage.csv",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_EUROPE_DATA_COVERAGE_AUDIT/summary.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_EUROPE_DATA_COVERAGE_AUDIT/session-coverage.csv",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT_ALL_PERIOD/summary.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT_ALL_PERIOD/periods/DECEMBER_2025.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT_ALL_PERIOD/periods/JANUARY_2026.json",
    )


def source_integrity(repository_root: Path) -> dict[str, Any]:
    missing = [str(path) for path in _required_source_paths(repository_root) if not path.is_file()]
    if missing:
        raise Block1BaselineError(f"required Block 1 source artifact is missing: {missing}")
    return {
        "status": "PASS",
        "files": [
            {"path": str(path.relative_to(repository_root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in _required_source_paths(repository_root)
        ],
        "network_calls": 0,
        "downloads": 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Block1BaselineError(f"expected JSON object: {path}")
    return value


def _active_contracts(day: str) -> tuple[str, str]:
    """Return the sealed Dec/Jan ES and MES contract pair for a target day."""
    if day <= "2025-12-16":
        return "ESZ5", "MESZ5"
    return "ESH6", "MESH6"


def audit_ny_source_paths(repository_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    """Audit manifest-bound NY inputs without opening DBNs or replaying them."""
    manifest_path = repository_root / NY_SOURCE_MANIFEST
    if not manifest_path.is_file():
        raise Block1BaselineError(f"NY acquisition manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    prior_by_day = manifest.get("prior_rth_by_target_session")
    if not isinstance(files, dict) or not isinstance(prior_by_day, dict):
        raise Block1BaselineError("NY acquisition manifest lacks verified files or prior-RTH mapping")
    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in files.values():
        if isinstance(record, dict) and record.get("target_session") in set(dates):
            rows_by_target[str(record["target_session"])].append(dict(record))
    audit_rows: list[dict[str, Any]] = []
    for day in dates:
        expected_es, expected_mes = _active_contracts(day)
        records = rows_by_target.get(str(day), [])
        purpose_counts = Counter(str(record.get("purpose")) for record in records)
        by_purpose = {str(record.get("purpose")): record for record in records}
        failures: list[str] = []
        checks = (
            ("ES_MBP10", "mbp-10", expected_es, "13:00:00Z", "22:45:01Z"),
            ("MES_MBP1", "mbp-1", expected_mes, "13:30:00Z", "22:45:01Z"),
        )
        paths: list[str] = []
        for purpose, schema, symbol, start_time, end_time in checks:
            if purpose_counts[purpose] != 1:
                failures.append(f"expected exactly one {purpose}, found {purpose_counts[purpose]}")
            record = by_purpose.get(purpose)
            if record is None:
                failures.append(f"missing {purpose}")
                continue
            path = (manifest_path.parent / str(record.get("local_path"))).resolve()
            paths.append(str(path))
            if record.get("dataset") != "GLBX.MDP3":
                failures.append(f"{purpose} dataset mismatch")
            if record.get("schema") != schema or record.get("raw_symbol") != symbol:
                failures.append(f"{purpose} contract/schema mismatch")
            if record.get("start_utc") != f"{day}T{start_time}" or record.get("end_utc") != f"{day}T{end_time}":
                failures.append(f"{purpose} UTC range mismatch")
            file_record = files.get(record.get("local_path"), {})
            if not path.is_file() or path.stat().st_size <= 0:
                failures.append(f"{purpose} file missing or empty")
            if file_record.get("bytes") is not None and path.is_file() and path.stat().st_size != int(file_record["bytes"]):
                failures.append(f"{purpose} byte count mismatch")
            if not file_record.get("sha256"):
                failures.append(f"{purpose} manifest SHA-256 missing")

        prior_day = str(prior_by_day.get(day, ""))
        prior = by_purpose.get("PRIOR_RTH_TRADES")
        if prior is None:
            failures.append("missing PRIOR_RTH_TRADES")
        else:
            prior_path = (manifest_path.parent / str(prior.get("local_path"))).resolve()
            paths.append(str(prior_path))
            expected_prior_es = _active_contracts(prior_day)[0] if prior_day else None
            if prior.get("schema") != "trades" or prior.get("raw_symbol") != expected_prior_es:
                failures.append("prior-RTH contract/schema mismatch")
            if prior.get("prior_rth_date") != prior_day:
                failures.append("prior-RTH date mismatch")
            if prior.get("start_utc") != f"{prior_day}T13:30:00Z" or prior.get("end_utc") != f"{prior_day}T20:00:00Z":
                failures.append("prior-RTH UTC range mismatch")
            file_record = files.get(prior.get("local_path"), {})
            if not prior_path.is_file() or prior_path.stat().st_size <= 0:
                failures.append("prior-RTH file missing or empty")
            if file_record.get("bytes") is not None and prior_path.is_file() and prior_path.stat().st_size != int(file_record["bytes"]):
                failures.append("prior-RTH byte count mismatch")
            if not file_record.get("sha256"):
                failures.append("prior-RTH manifest SHA-256 missing")
        audit_rows.append({
            "date": str(day),
            "classification": "NY_SOURCE_READY" if not failures else "NY_SOURCE_NOT_READY",
            "es_symbol": expected_es,
            "mes_symbol": expected_mes,
            "prior_rth_date": prior_day,
            "paths": paths,
            "failures": failures,
        })
    return {
        "status": "PASS" if all(not row["failures"] for row in audit_rows) else "FAIL",
        "classification_counts": dict(Counter(row["classification"] for row in audit_rows)),
        "dates": audit_rows,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


def _performance(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return historical._performance([dict(row) for row in trades])


def _published_ny_subset(repository_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    """Reuse the existing verified Berlin period artifacts, date-filtered.

    The current Berlin implementation already published these Dec/Jan results,
    but its compact event tapes are not present in the repository.  Replaying
    without those inputs would be a fabricated substitute, so this path is
    explicit, hash-bound reuse of the existing corrected period artifacts.
    """
    baseline_root = repository_root / berlin.BASELINE_ROOT
    baseline = berlin.load_corrected_berlin_baseline(repository_root, baseline_root)
    target = set(dates)
    periods = {str(row["period_id"]): row for row in baseline["periods"]}
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for period_id, filename in (("DECEMBER_2025", "DECEMBER_2025.json"), ("JANUARY_2026", "JANUARY_2026.json")):
        payload = _read_json(baseline_root / "periods" / filename)
        audit = payload.get("semantic_audit")
        if not isinstance(audit, dict):
            raise Block1BaselineError(f"NY Berlin period audit is missing: {period_id}")
        terminal = audit.get("corrected_terminal_outcomes")
        period_trades = audit.get("corrected_trades")
        if not isinstance(terminal, dict) or not isinstance(period_trades, list):
            raise Block1BaselineError(f"NY Berlin period audit schema is invalid: {period_id}")
        trades.extend(dict(row) for row in period_trades if str(row.get("date")) in target)
        by_day: dict[str, list[str]] = defaultdict(list)
        for setup_id, outcome in terminal.items():
            day = str(setup_id).split("|", 1)[0]
            if day in target:
                by_day[day].append(str(outcome))
        period_days = sorted(day for day in by_day if day in target)
        for day in period_days:
            day_trades = [row for row in trades if str(row.get("date")) == day]
            perf = _performance(day_trades)
            outcomes = by_day[day]
            rows.append({
                "session_date": day,
                "period_id": period_id,
                "source_model": str(periods[period_id].get("source_model")),
                "source_mode": "REUSED_PUBLISHED_BERLIN_PERIOD_ARTIFACT",
                "accepted_setups": len(outcomes),
                "confirmations_passed": sum(outcome == "TRADE_EXECUTED" for outcome in outcomes),
                "confirmation_failures": sum(outcome != "TRADE_EXECUTED" for outcome in outcomes),
                "raw_interactions": None,
                **perf,
            })
    existing_days = {str(row["session_date"]) for row in rows}
    missing = sorted(target - existing_days)
    source_audit = audit_ny_source_paths(repository_root, missing) if missing else {
        "status": "NOT_REQUIRED", "dates": [], "classification_counts": {},
    }
    if missing:
        if source_audit["status"] != "PASS":
            raise Block1BaselineError(f"NY source audit failed for missing artifact dates: {source_audit}")
        if not (repository_root / NY_DATE_SUBSET_ROOT).is_dir():
            berlin.run_date_subset(
                repository_root=repository_root,
                dates=missing,
                output_root=NY_DATE_SUBSET_ROOT,
            )
        subset = berlin.load_date_subset(repository_root, NY_DATE_SUBSET_ROOT, missing)
        rows.extend(dict(row, source_mode="REUSED_CORRECTED_BERLIN_DATE_SUBSET_ARTIFACT") for row in subset["sessions"])
        trades.extend(dict(row) for row in subset["trades"])
    if {str(row["session_date"]) for row in rows} != target:
        absent = sorted(target - {str(row["session_date"]) for row in rows})
        raise Block1BaselineError(f"NY Berlin artifacts do not cover requested dates after repair: {absent}")
    rows.sort(key=lambda row: str(row["session_date"]))
    trades.sort(key=lambda row: (str(row.get("date")), str(row.get("trade_id"))))
    return {
        "status": "PASS",
        "execution_mode": "REUSED_PUBLISHED_BERLIN_PERIOD_ARTIFACT",
        "strategy_id": berlin.STRATEGY_ID,
        "contract_hash": berlin.CONTRACT_SHA256,
        "sessions": rows,
        "trades": trades,
        "source_audit": source_audit,
        "source_period_artifact_hashes": {
            filename: _sha256(baseline_root / "periods" / filename)
            for filename in ("DECEMBER_2025.json", "JANUARY_2026.json")
        },
    }


def _result_artifact_rows(path: Path, filename: str) -> list[dict[str, Any]]:
    with (path / filename).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _family_result(name: str, root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    trades = _result_artifact_rows(root, "trade-ledger.csv")
    daily = _result_artifact_rows(root, "daily-results.csv")
    return {
        "family": name,
        "strategy_id": summary.get("strategy_id"),
        "config_id": summary.get("config_id"),
        "sessions": int(summary.get("eligible_session_count", len(daily))),
        "trades": int(summary.get("performance", {}).get("completed_trades", len(trades))),
        "winners": int(summary.get("performance", {}).get("wins", 0)),
        "losses": int(summary.get("performance", {}).get("losses", 0)),
        "net_r": float(summary.get("performance", {}).get("total_r", 0.0)),
        "net_pnl_usd": float(summary.get("performance", {}).get("net_pnl_usd", 0.0)),
        "max_drawdown_r": float(summary.get("performance", {}).get("max_cumulative_drawdown_r", 0.0)),
        "profit_factor": summary.get("performance", {}).get("profit_factor"),
        "profitable_session_ratio": sum(float(row.get("total_r") or 0.0) > 0 for row in daily) / len(daily) if daily else 0.0,
    }


def _ny_family_result(ny: Mapping[str, Any]) -> dict[str, Any]:
    trades = list(ny["trades"])
    perf = _performance(trades)
    sessions = list(ny["sessions"])
    return {
        "family": "NY_BERLIN",
        "strategy_id": ny["strategy_id"],
        "config_id": "CORRECTED_BERLIN_V3_BASELINE",
        "sessions": len(sessions),
        "trades": int(perf["completed_trades"]),
        "winners": int(perf["wins"]),
        "losses": int(perf["losses"]),
        "net_r": float(perf["total_r"]),
        "net_pnl_usd": float(perf["net_pnl_usd"]),
        "max_drawdown_r": float(perf["max_cumulative_drawdown_r"]),
        "profit_factor": perf["profit_factor"],
        "profitable_session_ratio": sum(float(row.get("total_r") or 0.0) > 0 for row in sessions) / len(sessions) if sessions else 0.0,
    }


def _aggregate(families: Sequence[Mapping[str, Any]], trades: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    perf = _performance(trades)
    return {
        "sessions": len(sessions),
        "trades": int(perf["completed_trades"]),
        "winners": int(perf["wins"]),
        "losses": int(perf["losses"]),
        "net_r": float(perf["total_r"]),
        "net_pnl_usd": float(perf["net_pnl_usd"]),
        "max_drawdown_r": float(perf["max_cumulative_drawdown_r"]),
        "profit_factor": perf["profit_factor"],
        "profitable_session_ratio": sum(float(row.get("total_r") or 0.0) > 0 for row in sessions) / len(sessions) if sessions else 0.0,
        "family_count": len(families),
    }


def run_block1(*, repository_root: Path, split: str = "train", output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    split_info = validate_split(split)
    output_root = (repository_root / output_root if not output_root.is_absolute() else output_root).resolve()
    staging = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"immutable Block 1 output already exists or is building: {output_root}")
    europe_audit = repository_root / europe.AUDIT_ROOT
    if not europe_audit.is_dir():
        europe.build_europe_coverage_audit(repository_root, europe_audit)
    source_report = source_integrity(repository_root)
    config_report = _config_integrity()
    git_report = _git_metadata(repository_root)
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        asia_result = asia.run_replay(
            repository_root=repository_root,
            output_root=staging / "asia-w04",
            audit_root=repository_root / asia.AUDIT_ROOT,
            session_dates=TRAIN_DATES if split == "train" else VALIDATION_DATES,
            profile_only_dates=TRAIN_PROFILE_ONLY_CONTEXT_DATES if split == "train" else (),
        )
        europe_result = europe.run_replay(
            repository_root=repository_root,
            output_root=staging / "europe-w04",
            audit_root=europe_audit,
            session_dates=TRAIN_DATES if split == "train" else VALIDATION_DATES,
            profile_only_dates=TRAIN_PROFILE_ONLY_CONTEXT_DATES if split == "train" else (),
        )
        ny_result = _published_ny_subset(repository_root, split_info["dates"])

        asia_root = staging / "asia-w04"
        europe_root = staging / "europe-w04"
        family_results = [
            _family_result("ASIA_W04", asia_root, asia_result),
            _family_result("EUROPE_W04", europe_root, europe_result),
            _ny_family_result(ny_result),
        ]
        all_trades = [
            *[dict(row, strategy_family="ASIA_W04") for row in _result_artifact_rows(asia_root, "trade-ledger.csv")],
            *[dict(row, strategy_family="EUROPE_W04") for row in _result_artifact_rows(europe_root, "trade-ledger.csv")],
            *[dict(row, strategy_family="NY_BERLIN") for row in ny_result["trades"]],
        ]
        all_sessions = [
            *[dict(row, strategy_family="ASIA_W04") for row in _result_artifact_rows(asia_root, "daily-results.csv")],
            *[dict(row, strategy_family="EUROPE_W04") for row in _result_artifact_rows(europe_root, "daily-results.csv")],
            *[dict(row, strategy_family="NY_BERLIN") for row in ny_result["sessions"]],
        ]
        all_trades.sort(key=lambda row: (str(row.get("date")), str(row.get("trade_id"))))
        all_sessions.sort(key=lambda row: (str(row.get("session_date", row.get("date"))), str(row.get("strategy_family"))))
        aggregate = _aggregate(family_results, all_trades, all_sessions)
        _write_json(staging / "ny-berlin" / "summary.json", ny_result)
        _write_json(staging / "strategy-results.json", {"strategies": family_results})
        _write_json(staging / "aggregate-results.json", aggregate)
        _write_csv(staging / "trade-ledger.csv", all_trades)
        _write_csv(staging / "daily-results.csv", all_sessions)
        runtime = time.monotonic() - started
        run_manifest = {
            "status": "BLOCK1_TRAIN_BASELINE_COMPLETE" if split == "train" else "BLOCK1_VALIDATION_BASELINE_COMPLETE",
            "format_version": FORMAT_VERSION,
            "run_id": RUN_ID if split == "train" else "BLOCK1_VALIDATION_BASELINE",
            "split": split,
            "dates": split_info,
            "exclusions": [dict(row) for row in EXCLUDED_DATES],
            "profile_only_context_dates": list(
                TRAIN_PROFILE_ONLY_CONTEXT_DATES if split == "train" else ()
            ),
            "repository": str(repository_root),
            "git": git_report,
            "configs": config_report,
            "source_integrity": source_report,
            "source_verification": {
                "asia_w04": asia_result.get("source_verification", []),
                "europe_w04": europe_result.get("source_verification", []),
                "ny_berlin": {
                    "period_artifacts": ny_result.get("source_period_artifact_hashes", {}),
                    "source_audit": ny_result.get("source_audit", {}),
                },
            },
            "strategy_families": ["ASIA_W04", "EUROPE_W04", "NY_BERLIN"],
            "source_models": {"ASIA_W04": "MBO_DERIVED_MBP10 or NATIVE_MBP10", "EUROPE_W04": "MBO_DERIVED_MBP10 or NATIVE_MBP10", "NY_BERLIN": "PUBLISHED_CORRECTED_BERLIN_PERIOD_ARTIFACT"},
            "ny_execution_mode": ny_result["execution_mode"],
            "runtime_seconds": runtime,
            "network_calls": 0,
            "downloads": 0,
            "validation_executed": split == "validation",
            "optimization_executed": False,
            "final_oos_accessed": False,
        }
        _write_json(staging / "run-manifest.json", run_manifest)
        os.rename(staging, output_root)
        return {**run_manifest, "aggregate": aggregate, "output_root": str(output_root)}
    except BaseException:
        # Preserve a failed staging root for diagnosis; never publish partial
        # output as a completed baseline.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_block1(repository_root=args.repository_root, split=args.split, output_root=args.output_root)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps({
        "status": result["status"], "split": result["split"],
        "sessions": result["dates"]["count"], "output_root": result["output_root"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
