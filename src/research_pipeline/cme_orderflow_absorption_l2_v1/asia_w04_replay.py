"""Deterministic Asia-session replay for the frozen W04 candidate.

This module is deliberately isolated from the NY/RTH implementations.  It
opens only the already sealed local ES MBO sources, reconstructs the public
MBP-10 view through the existing private adapter, and uses the existing W04
interaction, confirmation, entry, stop, target, cost, and portfolio logic.

The only execution concession is explicit and auditable: when one ES contract
does not fit the frozen USD 250 risk budget, MES economics are applied to the
same ES observations.  No MES quote, spread, timestamp, or price path is
invented.  Such trades are labelled ``MES_PROXY_FROM_ES`` throughout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import statistics
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import causal_master_tape as master
from . import berlin_hardflat_execution as corrected
from . import historical_runner as historical
from . import public_book_adapters as public_books
from . import weight_q_research as matrix
from .model import (
    ENTRY_LATENCY_NS,
    ES_CAP,
    ES_COMMISSION,
    ES_POINT_VALUE,
    EXIT_RESET_NS,
    INACTIVITY_NS,
    MAX_CONFIRMATION_NS,
    MES_CAP,
    MES_COMMISSION,
    MES_POINT_VALUE,
    MIN_CONFIRMATION_NS,
    RISK_BUDGET_USD,
    STOP_BUFFER_TICKS,
    TARGET_R,
    TICK,
    VICINITY_TICKS,
    Execution,
    L2Interaction,
    L2InteractionEngine,
    initial_prices,
    size_for_instrument,
)
from .v2_quality050 import V2_CONFIG


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY"
CONFIG_ID = "W04-02-06-04-04-Q45"
EVIDENCE_LABEL = "ASIA_W04_ES_NATIVE_MES_PROXY_RETROSPECTIVE_RESEARCH"
INTERPRETATION = "RETROSPECTIVE_RESEARCH_NOT_FRESH_OOS_EVIDENCE"
DISCLAIMER = (
    "THIS IS RETROSPECTIVE ASIA RESEARCH USING ES-BASED MES PROXY EXECUTION. "
    "IT IS NOT NATIVE MES EXECUTION AND NOT FRESH OOS EVIDENCE."
)
ASIA_LEVEL_NAME = "PRIOR_ASIA_SESSION_POC"
ASIA_START_SECONDS = 0
ASIA_END_SECONDS = 8 * 60 * 60
ASIA_HARD_FLAT_REASON = "HARD_FLAT_ASIA_0800"
ASIA_ENTRY_CUTOFF_REASON = "SESSION_ENTRY_CUTOFF_0800"
_GENERIC_HARD_FLAT_UNRESOLVED = frozenset({
    "UNRESOLVED_NO_ENTRY_OBSERVATION",
    "UNRESOLVED_NO_LATER_EVENT",
    "UNRESOLVED_AT_HARD_FLAT",
})
ASIA_LIQUIDATION_LOOKBACK_NS = 1_000_000_000
EXPECTED_ELIGIBLE_SESSIONS = 46
EXPECTED_SOURCE_SESSIONS = 48
ELIGIBLE_CLASSIFICATION = "MISSING_MES_EXECUTION"
PROFILE_ONLY_CLASSIFICATION = "MISSING_PRIOR_ASIA_POC"
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY")
AUDIT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT")

W04_WEIGHTS: dict[str, Decimal] = {
    "aggression_score": Decimal("0.20"),
    "restoration_score": Decimal("0.10"),
    "price_resistance_score": Decimal("0.30"),
    "persistence_score": Decimal("0.20"),
    "multi_level_support_score": Decimal("0.20"),
}
QUALITY_THRESHOLD = Decimal("0.45")
W04_CONFIG = replace(
    V2_CONFIG,
    min_quality_score=float(QUALITY_THRESHOLD),
    aggression_weight=float(W04_WEIGHTS["aggression_score"]),
    restoration_weight=float(W04_WEIGHTS["restoration_score"]),
    price_resistance_weight=float(W04_WEIGHTS["price_resistance_score"]),
    persistence_weight=float(W04_WEIGHTS["persistence_score"]),
    multi_level_support_weight=float(W04_WEIGHTS["multi_level_support_score"]),
    weights_label=CONFIG_ID,
)

ALLOWED_SEMANTIC_DIFFERENCES = frozenset({
    "strategy_identifier",
    "session_window",
    "structural_level",
    "position_management_cutoff",
    "mes_execution_model",
    "evidence_label",
})


class AsiaReplayError(RuntimeError):
    """The isolated Asia replay cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class AsiaStructuralLevel:
    """Isolated structural level that cannot be mistaken for a prior-RTH level."""

    name: str
    price: float

    def __post_init__(self) -> None:
        if self.name != ASIA_LEVEL_NAME:
            raise AsiaReplayError("Asia replay accepts only PRIOR_ASIA_SESSION_POC")
        value = float(self.price)
        if not 0.0 < value < 100_000.0 or not math.isclose(
            value / TICK, round(value / TICK), rel_tol=0.0, abs_tol=1e-9,
        ):
            raise AsiaReplayError("Asia POC must be a normalized ES tick price")
        object.__setattr__(self, "price", value)


@dataclass(frozen=True)
class AuditSession:
    day: str
    period: str
    source_model: str
    prior_day: str | None
    classification: str

    @property
    def eligible(self) -> bool:
        return self.classification == ELIGIBLE_CLASSIFICATION


@dataclass(frozen=True)
class SourceBinding:
    period: str
    path: Path
    days: tuple[str, ...]
    expected_sha256: str
    expected_bytes: int | None
    shared: bool
    extra_paths: tuple[Path, ...] = ()
    extra_sha256: tuple[str, ...] = ()
    extra_bytes: tuple[int, ...] = ()


@dataclass(frozen=True)
class SessionSpec:
    day: str
    period: str
    source_model: str
    prior_day: str | None
    eligible: bool
    start_ns: int
    cutoff_ns: int
    source_path: Path
    staging_root: Path


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AsiaReplayError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AsiaReplayError(f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(fields or dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _clock_ns(day: str, seconds: int) -> int:
    return historical._clock_ns(day, seconds)


def _iso_ns(value: int | None) -> str | None:
    if value is None:
        return None
    seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanoseconds:09d}Z"


def _date_from_ns(value: int) -> str:
    return datetime.fromtimestamp(int(value) // 1_000_000_000, tz=UTC).date().isoformat()


def in_asia_window(day: str, timestamp_ns: int) -> bool:
    return _clock_ns(day, ASIA_START_SECONDS) <= int(timestamp_ns) < _clock_ns(day, ASIA_END_SECONDS)


def prior_asia_poc(volume_by_tick: Mapping[int, int]) -> float:
    """Return the lower max-volume ES tick, matching the existing POC tie break."""
    valid = {int(tick): int(volume) for tick, volume in volume_by_tick.items() if int(volume) > 0}
    if not valid:
        raise AsiaReplayError("completed Asia session has no ES executions for POC construction")
    tick = min(valid, key=lambda value: (-valid[value], value))
    return tick * TICK


def _execution_tick(execution: Execution) -> int:
    value = execution.price / TICK
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise AsiaReplayError("ES execution is not aligned to the frozen 0.25-point tick")
    return int(rounded)


def frozen_semantic_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    unchanged = {
        "configuration_id": CONFIG_ID,
        "weights": {key: str(value) for key, value in W04_WEIGHTS.items()},
        "quality_threshold": str(QUALITY_THRESHOLD),
        "interaction_vicinity_ticks": VICINITY_TICKS,
        "interaction_timeout_seconds": INACTIVITY_NS / 1_000_000_000,
        "interaction_exit_reset_seconds": EXIT_RESET_NS / 1_000_000_000,
        "primitive_thresholds": {
            "min_relevant_aggressive_volume": W04_CONFIG.min_relevant_aggressive_volume,
            "min_relevant_execution_count": W04_CONFIG.min_relevant_execution_count,
            "min_consume_restore_cycles": W04_CONFIG.min_consume_restore_cycles,
            "max_through_level_progress_ticks": W04_CONFIG.max_through_level_progress_ticks,
            "min_rejection_ticks": W04_CONFIG.min_rejection_ticks,
        },
        "false_refill_penalty_weight": W04_CONFIG.false_refill_penalty_weight,
        "confirmation": {
            "first_eligible_seconds": MIN_CONFIRMATION_NS / 1_000_000_000,
            "last_eligible_seconds_inclusive": MAX_CONFIRMATION_NS / 1_000_000_000,
            "minimum_favorable_ticks": 3,
            "no_early_invalidation": True,
            "first_qualifying_execution": True,
        },
        "entry_latency_ms": ENTRY_LATENCY_NS / 1_000_000,
        "stop_buffer_ticks": STOP_BUFFER_TICKS,
        "target_r": TARGET_R,
        "risk_budget_usd": RISK_BUDGET_USD,
        "instrument_preference": "ES_FIRST_THEN_MES",
        "apex_caps": {"ES": ES_CAP, "MES": MES_CAP},
        "economics": {
            "ES": {"point_value": ES_POINT_VALUE, "commission_per_side": ES_COMMISSION},
            "MES": {"point_value": MES_POINT_VALUE, "commission_per_side": MES_COMMISSION},
        },
        "one_active_position": True,
        "execution_integrity": {
            "semantic_version": corrected.SEMANTIC_VERSION,
            "maximum_executable_bbo_gap_seconds": corrected.MAX_EXECUTABLE_BBO_GAP_NS / 1e9,
            "temporary_gap_at_or_below_limit": "RESUME_CAUSALLY",
            "temporary_gap_above_limit": "DATA_GAP_3S_FORCE_FLAT",
            "source_end": "FORCE_FLAT_LAST_VALID_BBO",
        },
    }
    ny = {
        **unchanged,
        "strategy_identifier": CONFIG_ID,
        "session_window": "EXISTING_NY_RTH_SESSION",
        "structural_level": "PRIOR_RTH_POC",
        "position_management_cutoff": "EXISTING_CORRECTED_BERLIN_HARD_FLAT",
        "mes_execution_model": "NATIVE_MES_MBP1",
        "evidence_label": "EXISTING_NY_W04_RESEARCH",
    }
    asia = {
        **unchanged,
        "strategy_identifier": STRATEGY_ID,
        "session_window": "[00:00:00,08:00:00)_UTC",
        "structural_level": ASIA_LEVEL_NAME,
        "position_management_cutoff": "08:00:00_UTC_LAST_VALID_BBO_IN_INCLUSIVE_PRECEDING_1S",
        "mes_execution_model": "MES_PROXY_FROM_ES_MONETARY_SCALING_ONLY",
        "evidence_label": EVIDENCE_LABEL,
    }
    return ny, asia


def semantic_diff_document() -> dict[str, Any]:
    ny, asia = frozen_semantic_contracts()
    differences = {
        key: {"ny_w04": ny.get(key), "asia_w04": asia.get(key)}
        for key in sorted(set(ny) | set(asia))
        if ny.get(key) != asia.get(key)
    }
    unexpected = sorted(set(differences) - ALLOWED_SEMANTIC_DIFFERENCES)
    missing = sorted(ALLOWED_SEMANTIC_DIFFERENCES - set(differences))
    if unexpected or missing:
        raise AsiaReplayError(
            f"Asia semantic isolation failed; unexpected={unexpected}, missing={missing}"
        )
    return {
        "status": "PASS",
        "reference": CONFIG_ID,
        "asia_strategy_id": STRATEGY_ID,
        "allowed_difference_fields": sorted(ALLOWED_SEMANTIC_DIFFERENCES),
        "differences": differences,
        "unexpected_difference_fields": [],
        "unchanged_contract_sha256": _canonical_sha256({
            key: value for key, value in asia.items() if key not in ALLOWED_SEMANTIC_DIFFERENCES
        }),
        "ny_contract_sha256": _canonical_sha256(ny),
        "asia_contract_sha256": _canonical_sha256(asia),
    }


def load_audit_sessions(audit_root: Path, *, include_native: bool = False) -> tuple[AuditSession, ...]:
    summary = _read_json(audit_root / "summary.json")
    if summary.get("audit_id") != "CMEOrderflowAbsorption.ES_L2_ASIA_DATA_COVERAGE_AUDIT":
        raise AsiaReplayError("wrong Asia coverage-audit identity")
    if summary.get("market_data_opened") is not False or summary.get("strategy_replay_executed") is not False:
        raise AsiaReplayError("Asia coverage audit provenance is not read-only")
    rows: list[AuditSession] = []
    native_dec_jan: list[AuditSession] = []
    try:
        with (audit_root / "session-coverage.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                classification = str(row["classification"])
                source_model = str(row["source_model"])
                period = str(row["period"])
                if include_native and source_model == public_books.NATIVE_MBP10 and period == "DEC2025_JAN2026_NATIVE":
                    native_dec_jan.append(AuditSession(
                        str(row["session_date"]), period, source_model, None, PROFILE_ONLY_CLASSIFICATION,
                    ))
                    continue
                if classification not in {ELIGIBLE_CLASSIFICATION, PROFILE_ONLY_CLASSIFICATION}:
                    continue
                rows.append(AuditSession(
                    str(row["session_date"]), str(row["period"]), str(row["source_model"]),
                    str(row["prior_asia_session"]) or None, classification,
                ))
    except OSError as exc:
        raise AsiaReplayError("Asia session coverage CSV is unavailable") from exc
    native_dec_jan.sort(key=lambda item: item.day)
    native_dec_jan = [
        replace(item, prior_day=(native_dec_jan[index - 1].day if index else None),
                classification=(PROFILE_ONLY_CLASSIFICATION if index == 0 else ELIGIBLE_CLASSIFICATION))
        for index, item in enumerate(native_dec_jan)
    ]
    rows.extend(native_dec_jan)
    rows.sort(key=lambda item: item.day)
    days = [row.day for row in rows]
    eligible = [row for row in rows if row.eligible]
    expected_source_count = EXPECTED_SOURCE_SESSIONS + len(native_dec_jan)
    expected_eligible_count = EXPECTED_ELIGIBLE_SESSIONS + max(0, len(native_dec_jan) - 1)
    if len(rows) != expected_source_count or len(eligible) != expected_eligible_count:
        raise AsiaReplayError(
            f"Asia audit chronology mismatch: source={len(rows)}, eligible={len(eligible)}"
        )
    if len(days) != len(set(days)) or days != sorted(days):
        raise AsiaReplayError("Asia source chronology is duplicate or unordered")
    excluded = [row.day for row in rows if not row.eligible]
    expected_excluded = list(summary.get("candidate_session_universe", {}).get("missing_prior_asia_poc_dates", ()))
    if native_dec_jan:
        expected_excluded.append(native_dec_jan[0].day)
    if excluded != sorted(expected_excluded):
        raise AsiaReplayError("profile-only dates disagree with the sealed coverage audit")
    expected_eligible = list(
        summary.get("candidate_session_universe", {}).get(
            "es_and_prior_poc_present_but_mes_missing_dates", (),
        )
    )
    expected_eligible = sorted(expected_eligible + [row.day for row in native_dec_jan[1:]])
    if [row.day for row in eligible] != expected_eligible:
        raise AsiaReplayError("eligible Asia sessions disagree with the sealed coverage audit")
    if any(row.source_model not in public_books.SUPPORTED_SOURCE_MODELS for row in rows):
        raise AsiaReplayError("Asia manifest declares an unsupported public-book source model")
    return tuple(rows)


def _may_binding(repository_root: Path, sessions: Sequence[AuditSession]) -> list[SourceBinding]:
    root = repository_root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
    manifest = _read_json(root / "acquisition-manifest.json")
    files = manifest.get("files")
    if manifest.get("data_acquired") is not True or not isinstance(files, dict):
        raise AsiaReplayError("sealed May acquisition manifest is invalid")
    output: list[SourceBinding] = []
    for session in sessions:
        relative = f"es_mbo/ESM6_{session.day}_000000_224501_mbo.dbn.zst"
        record = files.get(relative)
        if not isinstance(record, dict) or record.get("schema") != "mbo" or record.get("session_date") != session.day:
            raise AsiaReplayError(f"May ES MBO manifest binding is invalid: {session.day}")
        output.append(SourceBinding(
            session.period, root / relative, (session.day,), str(record["sha256"]),
            int(record["bytes"]), False,
        ))
    return output


def _retro_binding(repository_root: Path, sessions: Sequence[AuditSession]) -> list[SourceBinding]:
    manifest = _read_json(
        repository_root
        / "research_runs/CMEOrderflowAbsorption.ES_V2_RETRO_HOLDOUT/2026-06-23_2026-07-17-fixed/input-manifest.json"
    )
    records = {str(row["date"]): row for row in manifest.get("days", ()) if isinstance(row, dict)}
    output: list[SourceBinding] = []
    for session in sessions:
        record = records.get(session.day, {}).get("es_mbo")
        if not isinstance(record, dict):
            raise AsiaReplayError(f"retro ES MBO manifest binding is missing: {session.day}")
        path = repository_root / str(record["path"])
        output.append(SourceBinding(
            session.period, path, (session.day,), str(record["sha256"]), None, False,
        ))
    return output


def _dec_jan_native_binding(repository_root: Path, sessions: Sequence[AuditSession]) -> list[SourceBinding]:
    """Bind the declared native pre-NY file to the existing post-NY file."""
    pre_root = repository_root / "data/cme_orderflow_absorption_l2_v1/historical_completion/dec_jan_asia_europe"
    pre_manifest = _read_json(pre_root / "acquisition-manifest.json")
    post_root = repository_root / "data/cme_orderflow_absorption_l2_v3/dec2025_jan2026"
    post_manifest = _read_json(post_root / "acquisition-manifest.json")
    pre_files, post_files = pre_manifest.get("files"), post_manifest.get("files")
    if not isinstance(pre_files, dict) or not isinstance(post_files, dict):
        raise AsiaReplayError("native Dec/Jan source manifests are invalid")
    output: list[SourceBinding] = []
    for session in sessions:
        pre = next((row for row in pre_files.values() if isinstance(row, dict) and row.get("session_date") == session.day and row.get("schema") == "mbp-10"), None)
        post_key = next((key for key, row in post_files.items() if str(key).startswith("es_mbp10/") and isinstance(row, dict) and row.get("target_session") == session.day), None)
        post = post_files.get(post_key) if post_key is not None else None
        if not isinstance(pre, dict) or not isinstance(post, dict) or post_key is None:
            raise AsiaReplayError(f"native Dec/Jan MBP-10 binding is missing: {session.day}")
        output.append(SourceBinding(
            session.period, pre_root / str(pre["local_path"]), (session.day,), str(pre["sha256"]), int(pre["bytes"]), False,
            (post_root / str(post_key),), (str(post["sha256"]),), (int(post["bytes"]),),
        ))
    return output


def source_bindings(repository_root: Path, sessions: Sequence[AuditSession]) -> tuple[SourceBinding, ...]:
    grouped: dict[str, list[AuditSession]] = defaultdict(list)
    for session in sessions:
        grouped[session.period].append(session)
    baseline = {
        "MAY_2026_MBO_DERIVED",
        "RETRO_JUNE_JULY_2026_MBO_DERIVED",
        "JULY_20_31_PILOT_MBO",
        "AUGUST_03_07_SHARED_MBO",
    }
    allowed = baseline | {"DEC2025_JAN2026_NATIVE"}
    if not set(grouped).issubset(allowed) or not baseline.issubset(set(grouped)):
        raise AsiaReplayError(f"unexpected Asia source periods: {sorted(grouped)}")
    bindings = [
        *_may_binding(repository_root, grouped["MAY_2026_MBO_DERIVED"]),
        *_retro_binding(repository_root, grouped["RETRO_JUNE_JULY_2026_MBO_DERIVED"]),
    ]
    if "DEC2025_JAN2026_NATIVE" in grouped:
        bindings.extend(_dec_jan_native_binding(repository_root, grouped["DEC2025_JAN2026_NATIVE"]))
    pilot_manifest = _read_json(
        repository_root / "docs/research_pipeline/cme_orderflow_absorption_v1/mbo-pilot-manifest.json"
    )
    pilot_path = repository_root / "data/cme_orderflow_absorption_v1/ESU6/mbo/ESU6_2026-07-20_2026-08-01_mbo.dbn"
    bindings.append(SourceBinding(
        "JULY_20_31_PILOT_MBO", pilot_path,
        tuple(row.day for row in grouped["JULY_20_31_PILOT_MBO"]),
        str(pilot_manifest["dbn_sha256"]), int(pilot_manifest["dbn_bytes"]), True,
    ))
    oos_manifest = _read_json(
        repository_root / "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"
    )
    acquired = oos_manifest.get("proposed_acquisition", {})
    oos_path = repository_root / str(acquired.get("target_path"))
    bindings.append(SourceBinding(
        "AUGUST_03_07_SHARED_MBO", oos_path,
        tuple(row.day for row in grouped["AUGUST_03_07_SHARED_MBO"]),
        str(acquired.get("file_sha256")), int(acquired.get("file_bytes")), True,
    ))
    binding_by_day = {day: binding for binding in bindings for day in binding.days}
    if set(binding_by_day) != {row.day for row in sessions}:
        raise AsiaReplayError("source bindings do not preserve the audited 48-session chronology")
    return tuple(binding_by_day[row.day] for row in sessions)


def verify_source_bindings(bindings: Sequence[SourceBinding]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings, start=1):
        if not binding.path.is_file():
            raise AsiaReplayError(f"sealed local MBO source is missing: {binding.path}")
        actual_bytes = binding.path.stat().st_size
        if binding.expected_bytes is not None and actual_bytes != binding.expected_bytes:
            raise AsiaReplayError(f"sealed local MBO size mismatch: {binding.path}")
        print(
            f"ASIA_SOURCE_VERIFY {index:02d}/{len(bindings):02d} {binding.path.name}",
            flush=True,
        )
        actual_sha256 = _sha256(binding.path)
        if actual_sha256.lower() != binding.expected_sha256.lower():
            raise AsiaReplayError(f"sealed local MBO SHA-256 mismatch: {binding.path}")
        if len(binding.extra_paths) != len(binding.extra_sha256) or len(binding.extra_paths) != len(binding.extra_bytes):
            raise AsiaReplayError("native source binding hash metadata is incomplete")
        for extra_path, extra_hash, extra_size in zip(binding.extra_paths, binding.extra_sha256, binding.extra_bytes):
            if not extra_path.is_file() or extra_path.stat().st_size != extra_size or _sha256(extra_path).lower() != extra_hash.lower():
                raise AsiaReplayError(f"sealed native source hash/size mismatch: {extra_path}")
        verified.append({
            "period": binding.period,
            "path": str(binding.path),
            "days": list(binding.days),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "shared_source": binding.shared,
        })
    return verified


def _protected_ny_snapshot(repository_root: Path) -> dict[str, str]:
    paths = (
        repository_root / "src/research_pipeline/cme_orderflow_absorption_l2_v1/model.py",
        repository_root / "src/research_pipeline/cme_orderflow_absorption_l2_v1/berlin_hardflat_execution.py",
        repository_root / "src/research_pipeline/cme_orderflow_absorption_l2_v1/berlin_hardflat_all_period.py",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH_BERLIN_HARDFLAT/summary.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH_BERLIN_HARDFLAT/execution-contract.json",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH_BERLIN_HARDFLAT/candidate-summary.csv",
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH_BERLIN_HARDFLAT/candidate-period-comparison.csv",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AsiaReplayError(f"protected NY W04 evidence is missing: {missing}")
    return {str(path.relative_to(repository_root)).replace("\\", "/"): _sha256(path) for path in paths}


def _interaction_row(interaction: L2Interaction, day: str) -> dict[str, Any]:
    if interaction.end_ns is None or interaction.end_price is None:
        raise AsiaReplayError("only completed Asia interactions can be summarized")
    if interaction.level.name != ASIA_LEVEL_NAME:
        raise AsiaReplayError("RTH structural-level leakage detected in Asia interaction")
    features = interaction.feature_inputs()
    components = interaction.component_scores()
    row: dict[str, Any] = {
        "interaction_id": f"{day}|{interaction.interaction_id}",
        "source_interaction_id": interaction.interaction_id,
        "session_date": day,
        "interaction_start_ns": interaction.start_ns,
        "interaction_end_ns": interaction.end_ns,
        "direction": interaction.direction,
        "level": interaction.level.name,
        "level_price": interaction.level.price,
        "interaction_end_price": interaction.end_price,
        "zone_low": interaction.zone_low,
        "zone_high": interaction.zone_high,
        "termination": interaction.termination,
        **features,
        **components,
        "weights_label": CONFIG_ID,
    }
    score = master.recompute_quality(row, W04_WEIGHTS)
    primitive_reasons: list[str] = []
    if interaction.directional_aggressive_volume < W04_CONFIG.min_relevant_aggressive_volume or interaction.relevant_execution_count < W04_CONFIG.min_relevant_execution_count:
        primitive_reasons.append("INSUFFICIENT_RELEVANT_AGGRESSION")
    if interaction.consume_restore_cycles < W04_CONFIG.min_consume_restore_cycles:
        primitive_reasons.append("NO_GENUINE_CONSUME_RESTORE")
    if (
        float(features["maximum_through_level_progress_ticks"]) > W04_CONFIG.max_through_level_progress_ticks
        and float(features["interaction_rejection_ticks"]) < W04_CONFIG.min_rejection_ticks
    ):
        primitive_reasons.append("PRICE_PROGRESS_NOT_RESISTED")
    reasons = [*primitive_reasons]
    if Decimal(str(score)) < QUALITY_THRESHOLD:
        reasons.append("L2_QUALITY_BELOW_THRESHOLD")
    row.update({
        "w04_quality_score": score,
        "non_quality_rejection_reasons": ";".join(primitive_reasons),
        "rejection_reasons": ";".join(reasons),
        "accepted": not reasons,
    })
    return row


class AsiaProxySessionCausalTape(corrected.BerlinSessionCausalTape):
    """Reuse frozen execution logic while binding MES economics to the ES path."""

    def __init__(self, day: str, rows: Iterable[Mapping[str, Any]]) -> None:
        super().__init__(day, rows)
        cutoff_ns = _clock_ns(day, ASIA_END_SECONDS)
        if (
            str(self.original_terminal_event.get("event_type")) != "HARD_FLAT"
            or int(self.original_terminal_event.get("timestamp_ns") or -1) != cutoff_ns
        ):
            raise AsiaReplayError(f"Asia causal tape lacks its exact 08:00 terminal: {day}")
        self.hard_flat_timestamp_ns = cutoff_ns
        self.hard_flat_utc = datetime.fromtimestamp(cutoff_ns / 1e9, tz=UTC)
        self.hard_flat_local = self.hard_flat_utc
        self.contract_terminal_kind = "HARD_FLAT_ASIA_0800"
        self.contract_terminal_timestamp_ns = cutoff_ns
        self.contract_terminal_ordinal = int(self.original_terminal_event["event_ordinal"])
        self.source_end_event = None
        self.hard_event = {
            **self.original_terminal_event,
            "timestamp_ns": cutoff_ns,
            "event_type": "HARD_FLAT",
            "hard_flat_reason": ASIA_HARD_FLAT_REASON,
            "hard_flat_utc": self.hard_flat_utc.isoformat(),
        }
        self.mes_bid = array("d", self.es_bid)
        self.mes_ask = array("d", self.es_ask)
        self.mes_quote_timestamps = array("q", self.es_quote_timestamps)
        self._series[("MES", "bid")] = matrix._BlockExtrema(array("d", self.es_bid))
        self._series[("MES", "ask")] = matrix._BlockExtrema(array("d", self.es_ask))
        self._quotes["MES"] = self._quotes["ES"]
        self._quote_observation_timestamps["MES"] = self._quote_observation_timestamps["ES"]
        self._quote_event_ordinals["MES"] = self._quote_event_ordinals["ES"]
        if self.hard_event is not None:
            self.hard_event["mes_bid"] = self.hard_event.get("es_bid")
            self.hard_event["mes_ask"] = self.hard_event.get("es_ask")
            self.hard_event["mes_quote_timestamp_ns"] = self.hard_event.get("es_quote_timestamp_ns")

    def _terminal_candidate(
        self, *, instrument: str, direction: str,
        entry_seed: corrected.QuoteObservation,
    ) -> corrected.ExitObservation:
        observation = super()._terminal_candidate(
            instrument=instrument, direction=direction, entry_seed=entry_seed,
        )
        if observation.reason != "HARD_FLAT_BERLIN":
            return observation
        # Unlike the NY tapes, the Asia source is deliberately consumed only
        # through the exact cutoff.  Its price is the last valid pre-cutoff BBO,
        # while its portfolio transition belongs to the explicit hard-flat
        # event ordinal—not to the earlier quote's ordinal.
        return replace(observation, event_ordinal=self.contract_terminal_ordinal)

    def entry_outcome(self, interaction: Mapping[str, Any], event_ordinal: int) -> matrix.EntryOutcome:
        outcome = super().entry_outcome(interaction, event_ordinal)
        if outcome.trade is None:
            if outcome.terminal_reason == "ENTRY_BLOCKED_AT_OR_AFTER_HARD_FLAT_BERLIN":
                replacement = matrix.EntryOutcome(ASIA_ENTRY_CUTOFF_REASON)
                self._outcome_cache[(str(interaction["interaction_id"]), event_ordinal)] = replacement
                return replacement
            return outcome
        trade = outcome.trade
        if trade.get("exit_reason") == "HARD_FLAT_BERLIN":
            trade["exit_reason"] = ASIA_HARD_FLAT_REASON
        native_instrument = str(trade["instrument"])
        if native_instrument in {"ES_NATIVE_SOURCE", "MES_PROXY_FROM_ES"}:
            # ``SessionCausalTape`` caches entry outcomes.  Relabelling is an
            # idempotent presentation step and must not mutate a cached trade
            # a second time.
            return outcome
        if native_instrument not in {"ES", "MES"}:
            raise AsiaReplayError(f"unexpected frozen sizing instrument: {native_instrument}")
        execution_model = "ES_NATIVE_SOURCE" if native_instrument == "ES" else "MES_PROXY_FROM_ES"
        trade["source_instrument"] = "ES"
        trade["sizing_instrument"] = native_instrument
        trade["instrument"] = execution_model
        trade["execution_model"] = execution_model
        trade["interaction_id"] = str(interaction["interaction_id"])
        trade["setup_id"] = f"ASIA:{interaction['interaction_id']}"
        trade["trade_id"] = f"ASIA_T:{interaction['interaction_id']}"
        long = str(trade["direction"]) == "LONG"
        stop_exit = float(trade["stop"]) - TICK if long else float(trade["stop"]) + TICK
        sizing = size_for_instrument({
            "entry": float(trade["entry"]), "stop_exit": stop_exit,
        }, native_instrument)  # type: ignore[arg-type]
        trade.update({
            "risk_based_contracts": int(sizing["risk_based_contracts"]),
            "account_max_contracts": int(sizing["account_max_contracts"]),
            "one_contract_initial_risk_usd": float(sizing["one_contract_initial_risk_usd"]),
            "estimated_initial_risk_usd": float(sizing["estimated_initial_risk_usd"]),
            "risk_budget_usd": RISK_BUDGET_USD,
        })
        return outcome


def _classify_asia_entry_cutoff(
    session: matrix.SessionResult, *, tape: matrix.SessionCausalTape,
) -> None:
    """Turn generic hard-flat waiting states into the sealed Asia cutoff outcome.

    ``simulate_independent_session`` is shared with NY/RTH research and retains
    generic ``UNRESOLVED_*`` labels when a confirmed setup has no permissible
    entry observation before its terminal event.  A complete Asia tape has an
    explicit 08:00 hard-flat event, so those states are deterministically
    blocked by the sealed no-entry-at-or-after-08:00 rule; they are not source
    incompleteness.  True source-end and non-executable-boundary failures remain
    unresolved and continue to fail closed.
    """
    if tape.hard_event is None or tape.source_end_event is not None:
        return
    identifiers = [
        identifier
        for identifier, outcome in session.terminal_outcomes.items()
        if outcome in _GENERIC_HARD_FLAT_UNRESOLVED
    ]
    if not identifiers:
        return
    for identifier in identifiers:
        session.terminal_outcomes[identifier] = ASIA_ENTRY_CUTOFF_REASON
    count = len(identifiers)
    if count > session.unresolved:
        raise AsiaReplayError("Asia cutoff classification exceeds unresolved setup count")
    session.unresolved -= count
    session.other_terminal[ASIA_ENTRY_CUTOFF_REASON] = (
        session.other_terminal.get(ASIA_ENTRY_CUTOFF_REASON, 0) + count
    )


class _AsiaMboState:
    def __init__(self, spec: SessionSpec, prior_poc: float | None) -> None:
        self.spec = spec
        self.adapter = public_books.source_model_adapter(spec.source_model)
        self.profile_volume: Counter[int] = Counter()
        self.profile_execution_records = 0
        self.decoded_records = 0
        self.source_index = 0
        self.reached_cutoff = False
        self.latest_es_quote: tuple[float, float] | None = None
        self.latest_es_quote_ns: int | None = None
        self.prior_es_quote: tuple[float, float] | None = None
        self.interactions: list[dict[str, Any]] = []
        self.completed_seen = 0
        self.stored_events = 0
        self.ordinal = 0
        self.closed = False
        self.prior_poc = prior_poc
        self.writer: master.AtomicParquetStream | None = None
        self.engine: L2InteractionEngine | None = None
        self.tracker: master.CausalWindowTracker | None = None
        if spec.eligible:
            if prior_poc is None:
                raise AsiaReplayError(f"eligible Asia session lacks prior POC: {spec.day}")
            level = AsiaStructuralLevel(ASIA_LEVEL_NAME, prior_poc)
            self.engine = L2InteractionEngine([level], W04_CONFIG)  # type: ignore[list-item]
            self.tracker = master.CausalWindowTracker()
            work = spec.staging_root / "_work" / f"{spec.day}.parquet"
            if work.exists():
                work.unlink()
            part = work.with_suffix(work.suffix + ".part")
            if part.exists():
                part.unlink()
            self.writer = master.AtomicParquetStream(work)

    def _drain(self) -> None:
        if self.engine is None or self.tracker is None:
            return
        for interaction in self.engine.completed[self.completed_seen:]:
            row = _interaction_row(interaction, self.spec.day)
            self.interactions.append(row)
            self.tracker.register(row)
        self.completed_seen = len(self.engine.completed)

    def _append(
        self, *, timestamp_ns: int, event_type: str,
        execution: Execution | None = None, due: Sequence[str] = (),
        hard_flat_reason: str | None = None, book_state: str = "EXECUTABLE",
    ) -> None:
        if self.writer is None or self.tracker is None:
            return
        event_spec = master.SessionBuildSpec(
            self.spec.day, str(self.spec.source_path), "MES_PROXY_FROM_ES",
            float(self.prior_poc), self.spec.start_ns, self.spec.cutoff_ns,
            ASIA_HARD_FLAT_REASON, str(self.spec.staging_root), "ASIA_UTC",
        )
        self.writer.append(master._event_row(
            spec=event_spec,
            ordinal=self.ordinal,
            timestamp_ns=timestamp_ns,
            stream="CALENDAR" if event_type == "HARD_FLAT" else "ES",
            source_index=0 if event_type == "HARD_FLAT" else self.source_index,
            event_type=event_type,
            es_quote=self.latest_es_quote,
            mes_quote=None,
            execution=execution,
            book_state=book_state,
            entry_probe_count=len(due),
            es_quote_timestamp_ns=self.latest_es_quote_ns,
            mes_quote_timestamp_ns=None,
            hard_flat_reason=hard_flat_reason,
        ))
        self.tracker.bind_entry_probe(due, self.ordinal)
        self.ordinal += 1
        self.stored_events += 1

    def observe(self, record: object) -> None:
        if self.closed:
            raise AsiaReplayError(f"event routed to closed Asia session: {self.spec.day}")
        self.source_index += 1
        self.decoded_records += 1
        timestamp_ns = public_books.source_timestamp_ns(record)
        if timestamp_ns >= self.spec.cutoff_ns:
            self.reached_cutoff = True
            return
        previous_state = self.adapter.state
        public = self.adapter.feed(record, materialize_public=True)
        if public is None:
            if self.spec.eligible and self.adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                if previous_state != self.adapter.state or self.latest_es_quote is not None:
                    self.latest_es_quote = self.prior_es_quote = None
                    self.latest_es_quote_ns = None
                    self._append(
                        timestamp_ns=timestamp_ns,
                        event_type="BOOK_NON_EXECUTABLE",
                        book_state=self.adapter.state,
                    )
            return
        quote = historical._quote(public.snapshot)
        if quote is None:
            raise AsiaReplayError(f"MBO adapter emitted non-executable Asia BBO: {self.spec.day}")
        self.latest_es_quote = quote
        self.latest_es_quote_ns = public.timestamp_ns
        if public.execution is not None:
            self.profile_volume[_execution_tick(public.execution)] += int(public.execution.size)
            self.profile_execution_records += 1
        if not self.spec.eligible:
            return
        assert self.engine is not None and self.tracker is not None
        self.engine.advance(public.timestamp_ns)
        self._drain()
        self.engine.observe_snapshot(public.snapshot, public.update)
        if public.execution is not None:
            self.engine.observe_execution(public.execution)
            self._drain()
            self.tracker.observe_es_execution(public.execution)
        due = self.tracker.due_entry_probes(public.timestamp_ns)
        if quote != self.prior_es_quote or previous_state != "EXECUTABLE" or public.execution is not None or due:
            self._append(
                timestamp_ns=public.timestamp_ns,
                event_type="ES_EXECUTION" if public.execution is not None else "ES_BBO",
                execution=public.execution,
                due=due,
            )
            self.prior_es_quote = quote

    def finish(self) -> dict[str, Any]:
        if self.closed:
            raise AsiaReplayError(f"duplicate Asia session close: {self.spec.day}")
        if not self.reached_cutoff:
            raise AsiaReplayError(f"source did not reach the 08:00 UTC cutoff: {self.spec.day}")
        self.adapter.finish()
        session_poc = prior_asia_poc(self.profile_volume)
        result: dict[str, Any] = {
            "session_date": self.spec.day,
            "period": self.spec.period,
            "source_model": self.spec.source_model,
            "eligible": self.spec.eligible,
            "prior_asia_session": self.spec.prior_day,
            "prior_asia_poc": self.prior_poc,
            "session_poc": session_poc,
            "decoded_source_records": self.decoded_records,
            "profile_execution_count": self.profile_execution_records,
            "profile_execution_volume": sum(self.profile_volume.values()),
            "source_integrity_anomalies": len(self.adapter.source_integrity_diagnostics()),
            "raw_interactions": 0,
            "accepted_setups": 0,
            "confirmations_passed": 0,
            "confirmation_failures": 0,
            "terminal_outcomes": {},
            "interactions": [],
            "indexes": [],
            "trades": [],
            "unresolved": 0,
        }
        if not self.spec.eligible:
            self.closed = True
            return result
        assert self.engine is not None and self.tracker is not None and self.writer is not None
        self.engine.finish_rth(self.spec.cutoff_ns)
        self._drain()
        quote, quote_ns = master.liquidation_window_quote(
            cutoff_ns=self.spec.cutoff_ns,
            quote=self.latest_es_quote,
            quote_timestamp_ns=self.latest_es_quote_ns,
        )
        if quote is None or quote_ns is None:
            raise AsiaReplayError(
                f"08:00 hard flat lacks an ES BBO in the preceding inclusive second: {self.spec.day}"
            )
        self.latest_es_quote, self.latest_es_quote_ns = quote, quote_ns
        self._append(
            timestamp_ns=self.spec.cutoff_ns,
            event_type="HARD_FLAT",
            hard_flat_reason=ASIA_HARD_FLAT_REASON,
        )
        event_artifact = self.writer.close()
        indexes = self.tracker.index_rows(
            day=self.spec.day,
            first_event=0,
            last_event=self.ordinal - 1,
            cutoff_ns=self.spec.cutoff_ns,
        )
        if len(indexes) != len(self.interactions):
            raise AsiaReplayError(f"Asia interaction/index mismatch: {self.spec.day}")
        index_by_id = {str(row["interaction_id"]): row for row in indexes}
        accepted = [row for row in self.interactions if bool(row["accepted"])]
        tape = AsiaProxySessionCausalTape.from_parquet(self.spec.day, Path(event_artifact["path"]))
        session = matrix.simulate_independent_session(tape, accepted, index_by_id)
        _classify_asia_entry_cutoff(session, tape=tape)
        if session.unresolved != 0:
            unresolved = Counter(
                outcome
                for outcome in session.terminal_outcomes.values()
                if outcome.startswith("UNRESOLVED") or "UNRESOLVED" in outcome
            )
            raise AsiaReplayError(
                f"unresolved Asia setup/trade at session close: {self.spec.day}: "
                f"{dict(sorted(unresolved.items()))}"
            )
        if len(session.terminal_outcomes) != len(accepted):
            raise AsiaReplayError(f"accepted setup terminal mismatch: {self.spec.day}")
        for trade in session.trades:
            if float(trade["estimated_initial_risk_usd"]) > RISK_BUDGET_USD + 1e-9:
                raise AsiaReplayError(f"risk-budget overrun: {trade['trade_id']}")
            if int(trade["contracts"]) < 1:
                raise AsiaReplayError(f"zero-size executed trade: {trade['trade_id']}")
        Path(event_artifact["path"]).unlink()
        result.update({
            "raw_interactions": len(self.interactions),
            "accepted_setups": len(accepted),
            "confirmations_passed": session.confirmations,
            "confirmation_failures": session.confirmation_expiries,
            "terminal_outcomes": session.terminal_outcomes,
            "interactions": self.interactions,
            "indexes": indexes,
            "trades": session.trades,
            "unresolved": session.unresolved,
            "stored_events": self.stored_events,
            "event_tape_rows": int(event_artifact["rows"]),
            "event_tape_sha256_before_disposal": str(event_artifact["sha256"]),
        })
        self.closed = True
        return result

    def abort(self) -> None:
        if self.writer is not None and not self.closed:
            self.writer.abort()


def _session_checkpoint_path(staging: Path, day: str) -> Path:
    return staging / "_checkpoints" / f"{day}.json"


def _load_checkpoint(staging: Path, session: AuditSession) -> dict[str, Any] | None:
    path = _session_checkpoint_path(staging, session.day)
    if not path.is_file():
        return None
    payload = _read_json(path)
    if (
        payload.get("status") != "ASIA_SESSION_COMPLETE"
        or payload.get("strategy_id") != STRATEGY_ID
        or payload.get("session", {}).get("session_date") != session.day
        or bool(payload.get("session", {}).get("eligible")) != session.eligible
    ):
        raise AsiaReplayError(f"stale or incompatible Asia checkpoint: {session.day}")
    return dict(payload["session"])


def _save_checkpoint(staging: Path, result: Mapping[str, Any]) -> None:
    day = str(result["session_date"])
    _write_json(_session_checkpoint_path(staging, day), {
        "status": "ASIA_SESSION_COMPLETE",
        "strategy_id": STRATEGY_ID,
        "session": dict(result),
        "session_sha256": _canonical_sha256(result),
    })


def _spec(session: AuditSession, binding: SourceBinding, staging: Path) -> SessionSpec:
    return SessionSpec(
        session.day, session.period, session.source_model, session.prior_day,
        session.eligible, _clock_ns(session.day, ASIA_START_SECONDS),
        _clock_ns(session.day, ASIA_END_SECONDS), binding.path, staging,
    )


def _require_prior(session: AuditSession, previous_result: Mapping[str, Any] | None) -> float | None:
    if not session.eligible:
        return None
    if previous_result is None or str(previous_result["session_date"]) != session.prior_day:
        raise AsiaReplayError(
            f"prior Asia continuity mismatch for {session.day}: expected {session.prior_day}"
        )
    return float(previous_result["session_poc"])


def _process_daily_binding(
    binding: SourceBinding,
    session: AuditSession,
    *, staging: Path,
    previous_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(staging, session)
    if checkpoint is not None:
        return checkpoint
    state = _AsiaMboState(_spec(session, binding, staging), _require_prior(session, previous_result))
    next_progress = 5_000_000
    try:
        for record in public_books.stream_source(binding.path, session.source_model, binding.extra_paths):
            state.observe(record)
            if state.reached_cutoff:
                break
            if state.decoded_records >= next_progress:
                print(
                    f"ASIA_REPLAY {session.day} records={state.decoded_records:,} "
                    f"interactions={len(state.interactions):,}",
                    flush=True,
                )
                next_progress += 5_000_000
        result = state.finish()
        _save_checkpoint(staging, result)
        return result
    except BaseException:
        state.abort()
        raise


def _process_shared_binding(
    binding: SourceBinding,
    sessions: Sequence[AuditSession],
    *, staging: Path,
    previous_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    prior = previous_result
    pending: dict[str, AuditSession] = {}
    for session in sessions:
        checkpoint = _load_checkpoint(staging, session)
        if checkpoint is not None:
            if session.eligible:
                _require_prior(session, prior)
            results.append(checkpoint)
            prior = checkpoint
        else:
            pending[session.day] = session
    if not pending:
        return results
    states: dict[str, _AsiaMboState] = {}
    next_progress = 5_000_000
    records = 0
    try:
        source_model = sessions[0].source_model
        for record in public_books.stream_source(binding.path, source_model, binding.extra_paths):
            day = _date_from_ns(public_books.source_timestamp_ns(record))
            session = pending.get(day)
            if session is None:
                continue
            state = states.get(day)
            if state is None:
                # Every earlier audited session in this shared source must have
                # completed before a later date can be initialized.
                earlier = [item for item in sessions if item.day < day]
                for earlier_session in earlier:
                    restored = next(
                        (row for row in results if row["session_date"] == earlier_session.day), None,
                    )
                    if restored is not None:
                        prior = restored
                state = _AsiaMboState(_spec(session, binding, staging), _require_prior(session, prior))
                states[day] = state
            state.observe(record)
            records += 1
            if state.reached_cutoff:
                completed = state.finish()
                _save_checkpoint(staging, completed)
                results.append(completed)
                prior = completed
                pending.pop(day)
                states.pop(day)
                if not pending:
                    break
            if records >= next_progress:
                print(
                    f"ASIA_REPLAY {binding.period} records={records:,} "
                    f"completed={len(results):,}/{len(sessions):,}",
                    flush=True,
                )
                next_progress += 5_000_000
        if pending:
            raise AsiaReplayError(f"shared MBO source did not complete Asia sessions: {sorted(pending)}")
    except BaseException:
        for state in states.values():
            state.abort()
        raise
    by_day = {str(row["session_date"]): row for row in results}
    ordered = [by_day[session.day] for session in sessions]
    return ordered


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _performance(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = historical._performance([dict(row) for row in trades])
    r_values = [float(row["r_multiple"]) for row in trades]
    completed = len(trades)
    result.update({
        "trade_count": completed,
        "es_trades": sum(row.get("execution_model") == "ES_NATIVE_SOURCE" for row in trades),
        "mes_trades": sum(row.get("execution_model") == "MES_PROXY_FROM_ES" for row in trades),
        "win_rate": result["wins"] / completed if completed else 0.0,
        "average_r": statistics.mean(r_values) if r_values else 0.0,
        "median_r": statistics.median(r_values) if r_values else 0.0,
    })
    return result


def _group_metrics(
    sessions: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    session_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    trade_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for session in sessions:
        session_groups[str(session[key])].append(session)
    for trade in trades:
        trade_groups[str(trade[key])].append(trade)
    output: list[dict[str, Any]] = []
    for value in sorted(set(session_groups) | set(trade_groups)):
        grouped_sessions = session_groups.get(value, [])
        perf = _performance(trade_groups.get(value, []))
        output.append({
            "group_field": key,
            "group_value": value,
            "sessions": len(grouped_sessions),
            "raw_interactions": sum(int(row.get("raw_interactions", 0)) for row in grouped_sessions),
            "accepted_setups": sum(int(row.get("accepted_setups", 0)) for row in grouped_sessions),
            "confirmations_passed": sum(int(row.get("confirmations_passed", 0)) for row in grouped_sessions),
            **perf,
        })
    return output


def _trade_and_setup_rows(
    sessions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    setups: list[dict[str, Any]] = []
    for session in sessions:
        if not session["eligible"]:
            continue
        indexes = {str(row["interaction_id"]): row for row in session["indexes"]}
        interaction_rows = {str(row["interaction_id"]): row for row in session["interactions"]}
        trade_by_interaction = {str(row["interaction_id"]): row for row in session["trades"]}
        terminals = dict(session["terminal_outcomes"])
        accepted = [row for row in interaction_rows.values() if bool(row["accepted"])]
        if set(terminals) != {str(row["interaction_id"]) for row in accepted}:
            raise AsiaReplayError(f"setup terminal universe mismatch: {session['session_date']}")
        for row in accepted:
            identifier = str(row["interaction_id"])
            index = indexes[identifier]
            terminal = str(terminals[identifier])
            trade = trade_by_interaction.get(identifier)
            confirmation_price = index.get("derived_first_confirmation_price")
            end_price = float(row["interaction_end_price"])
            favorable = None
            if confirmation_price is not None:
                favorable = (
                    (float(confirmation_price) - end_price) / TICK
                    if row["direction"] == "BUYER_ABSORPTION"
                    else (end_price - float(confirmation_price)) / TICK
                )
            setup_id = f"ASIA:{identifier}"
            setup = {
                "session_date": session["session_date"],
                "period": session["period"],
                "source_model": session["source_model"],
                "setup_id": setup_id,
                "interaction_id": identifier,
                "direction": "LONG" if row["direction"] == "BUYER_ABSORPTION" else "SHORT",
                "level": row["level"],
                "prior_asia_session": session["prior_asia_session"],
                "prior_asia_poc": session["prior_asia_poc"],
                "interaction_start_ns": row["interaction_start_ns"],
                "interaction_start_utc": _iso_ns(int(row["interaction_start_ns"])),
                "interaction_end_ns": row["interaction_end_ns"],
                "interaction_end_utc": _iso_ns(int(row["interaction_end_ns"])),
                "interaction_end_price": row["interaction_end_price"],
                "zone_low": row["zone_low"],
                "zone_high": row["zone_high"],
                "quality_score": row["w04_quality_score"],
                "aggression_score": row["aggression_score"],
                "restoration_score": row["restoration_score"],
                "price_resistance_score": row["price_resistance_score"],
                "persistence_score": row["persistence_score"],
                "multi_level_support_score": row["multi_level_support_score"],
                "false_refill_penalty": row["false_refill_penalty"],
                "confirmation_timestamp_ns": index.get("derived_first_confirmation_timestamp_ns"),
                "confirmation_timestamp_utc": _iso_ns(index.get("derived_first_confirmation_timestamp_ns")),
                "confirmation_price": confirmation_price,
                "confirmation_favorable_ticks": favorable,
                "entry_ready_ns": index.get("entry_ready_ns"),
                "entry_ready_utc": _iso_ns(index.get("entry_ready_ns")),
                "terminal_disposition": terminal,
                "blocked_or_non_trade_reason": "" if terminal == "TRADE_EXECUTED" else terminal,
                "trade_id": trade.get("trade_id") if trade else None,
            }
            setups.append(setup)
            if trade is None:
                continue
            enriched = {
                **trade,
                "period": session["period"],
                "source_model": session["source_model"],
                "prior_asia_session": session["prior_asia_session"],
                "prior_asia_poc": session["prior_asia_poc"],
                "quality_score": row["w04_quality_score"],
                "aggression_score": row["aggression_score"],
                "restoration_score": row["restoration_score"],
                "price_resistance_score": row["price_resistance_score"],
                "persistence_score": row["persistence_score"],
                "multi_level_support_score": row["multi_level_support_score"],
                "false_refill_penalty": row["false_refill_penalty"],
                "interaction_start_ns": row["interaction_start_ns"],
                "interaction_start_utc": _iso_ns(int(row["interaction_start_ns"])),
                "interaction_end_ns": row["interaction_end_ns"],
                "interaction_end_utc": _iso_ns(int(row["interaction_end_ns"])),
                "interaction_end_price": row["interaction_end_price"],
                "zone_low": row["zone_low"],
                "zone_high": row["zone_high"],
                "confirmation_timestamp_ns": index.get("derived_first_confirmation_timestamp_ns"),
                "confirmation_timestamp_utc": _iso_ns(index.get("derived_first_confirmation_timestamp_ns")),
                "confirmation_price": confirmation_price,
                "confirmation_favorable_ticks": favorable,
                "entry_timestamp_utc": _iso_ns(int(trade["entry_timestamp_ns"])),
                "exit_timestamp_utc": _iso_ns(int(trade["exit_timestamp_ns"])),
                "blocked_or_non_trade_reason": "",
            }
            trades.append(enriched)
    setups.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["setup_id"])))
    trades.sort(key=lambda row: (int(row["entry_timestamp_ns"]), str(row["trade_id"])))
    if len({row["setup_id"] for row in setups}) != len(setups):
        raise AsiaReplayError("duplicate Asia setup ID")
    if len({row["trade_id"] for row in trades}) != len(trades):
        raise AsiaReplayError("duplicate Asia trade ID")
    if any(row["level"] != ASIA_LEVEL_NAME for row in setups):
        raise AsiaReplayError("RTH level leaked into setup ledger")
    return setups, trades


def _daily_rows(sessions: Sequence[Mapping[str, Any]], trades: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trades_by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        trades_by_day[str(trade["date"])].append(trade)
    output: list[dict[str, Any]] = []
    for session in sessions:
        if not session["eligible"]:
            continue
        day = str(session["session_date"])
        perf = _performance(trades_by_day.get(day, []))
        output.append({
            "session_date": day,
            "period": session["period"],
            "source_model": session["source_model"],
            "prior_asia_session": session["prior_asia_session"],
            "prior_asia_poc": session["prior_asia_poc"],
            "session_poc": session["session_poc"],
            "raw_interactions": session["raw_interactions"],
            "accepted_setups": session["accepted_setups"],
            "confirmations_passed": session["confirmations_passed"],
            "confirmation_failures": session["confirmation_failures"],
            "unresolved": session["unresolved"],
            **perf,
        })
    return output


def _risk_audit(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    risks = [float(row["estimated_initial_risk_usd"]) for row in trades]
    violations = [row["trade_id"] for row in trades if float(row["estimated_initial_risk_usd"]) > RISK_BUDGET_USD + 1e-9]
    return {
        "count": len(risks),
        "minimum_usd": min(risks) if risks else None,
        "maximum_usd": max(risks) if risks else None,
        "mean_usd": statistics.mean(risks) if risks else None,
        "median_usd": statistics.median(risks) if risks else None,
        "risk_budget_usd": RISK_BUDGET_USD,
        "violation_count": len(violations),
        "violation_trade_ids": violations,
        "pass": not violations,
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    perf = summary["performance"]
    risk = summary["risk_budget_audit"]
    return "\n".join([
        f"# {STRATEGY_ID}",
        "",
        f"Status: `{summary['status']}`  ",
        f"Evidence: `{EVIDENCE_LABEL}`",
        "",
        "## Frozen research contract",
        "",
        f"- Candidate: `{CONFIG_ID}` with weights 0.20 / 0.10 / 0.30 / 0.20 / 0.20 and Q=0.45.",
        f"- Session: `[00:00:00, 08:00:00) UTC`; level: `{ASIA_LEVEL_NAME}`.",
        "- Confirmation: first qualifying ES execution from +5s through +15s, at least +3 favorable ticks; 2ms latency.",
        f"- Stop: {STOP_BUFFER_TICKS} ticks beyond the completed zone; target: {TARGET_R:g}R; risk budget: USD {RISK_BUDGET_USD:.2f}.",
        "- Hard flat: 08:00 UTC using the last executable ES BBO in the inclusive preceding one-second window.",
        "- MES fallback is a monetary proxy from the exact ES price/timestamp path. It is not native MES evidence.",
        "",
        "## Population and funnel",
        "",
        f"- Eligible sessions: {summary['eligible_session_count']}",
        f"- Raw interactions: {summary['raw_interactions']}",
        f"- Accepted setups: {summary['accepted_setups']}",
        f"- Confirmations passed / failed: {summary['confirmations_passed']} / {summary['confirmations_failed']}",
        f"- Completed / unresolved trades: {perf['completed_trades']} / {summary['unresolved']}",
        "",
        "## Descriptive performance",
        "",
        f"- Wins / losses: {perf['wins']} / {perf['losses']}",
        f"- Win rate: {perf['win_rate']:.4%}",
        f"- Total / average / median R: {perf['total_r']:.6f} / {perf['average_r']:.6f} / {perf['median_r']:.6f}",
        f"- Net PnL: USD {perf['net_pnl_usd']:.2f}",
        f"- Profit factor: {perf['profit_factor']}",
        f"- Maximum cumulative drawdown: {perf['max_cumulative_drawdown_r']:.6f}R",
        f"- ES-native / MES-proxy trades: {summary['execution_model_counts'].get('ES_NATIVE_SOURCE', 0)} / {summary['execution_model_counts'].get('MES_PROXY_FROM_ES', 0)}",
        "",
        "## Integrity",
        "",
        f"- Setup terminal reconciliation: `{summary['reconciliation']['pass']}`",
        f"- Risk-budget audit: `{risk['pass']}`; violations={risk['violation_count']}; max={risk['maximum_usd']}",
        f"- Semantic diff: `{summary['semantic_diff_status']}`; NY artifact mutation: `{summary['ny_artifacts_mutated']}`.",
        f"- Network calls / downloads: {summary['network_calls']} / {summary['downloads']}",
        "",
        "## Interpretation",
        "",
        DISCLAIMER,
        "",
        "No parameter search, threshold optimization, NY-strategy modification, network acquisition, or automatic strategy selection occurred.",
        "",
    ])


def _report_html(summary: Mapping[str, Any]) -> str:
    perf = summary["performance"]
    cards = "".join(
        f"<div class='card'><strong>{html.escape(label)}</strong><br>{html.escape(str(value))}</div>"
        for label, value in (
            ("Sessions", summary["eligible_session_count"]),
            ("Accepted", summary["accepted_setups"]),
            ("Trades", perf["completed_trades"]),
            ("Total R", f"{perf['total_r']:.4f}"),
            ("Net PnL", f"${perf['net_pnl_usd']:.2f}"),
            ("Profit factor", perf["profit_factor"]),
        )
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{STRATEGY_ID}</title>
<style>body{{font:15px system-ui;margin:2rem;color:#17202a;max-width:1100px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem}}.card{{border:1px solid #ccd6df;border-radius:8px;padding:1rem}}code{{background:#eef1f5;padding:.1rem .25rem}}.warn{{font-weight:700;color:#8a2b06}}</style></head><body>
<h1>{STRATEGY_ID}</h1><p><code>{EVIDENCE_LABEL}</code></p><div class='grid'>{cards}</div>
<h2>Contract</h2><p>00:00-08:00 UTC, prior-Asia POC only, W04 Q45, +3 tick causal confirmation, 2ms latency, 5-tick zone stop, 3R target, USD 250 fixed risk, one position.</p>
<h2>Integrity</h2><p>Reconciliation: <code>{summary['reconciliation']['pass']}</code>. Risk audit: <code>{summary['risk_budget_audit']['pass']}</code>. Semantic diff: <code>{summary['semantic_diff_status']}</code>. Network/downloads: 0/0.</p>
<p class='warn'>{DISCLAIMER}</p>
</body></html>"""


def _materialize(
    *, staging: Path,
    audit_root: Path,
    source_verification: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    semantic_diff: Mapping[str, Any],
    ny_before: Mapping[str, str],
    ny_after: Mapping[str, str],
) -> dict[str, Any]:
    eligible_sessions = [row for row in sessions if bool(row["eligible"])]
    setups, trades = _trade_and_setup_rows(sessions)
    performance = _performance(trades)
    terminal_counts = Counter(row["terminal_disposition"] for row in setups)
    reconciliation = {
        "accepted_setups": len(setups),
        "terminal_dispositions": sum(terminal_counts.values()),
        "executed_dispositions": terminal_counts["TRADE_EXECUTED"],
        "trade_count": len(trades),
        "unique_setup_ids": len({row["setup_id"] for row in setups}),
        "unique_trade_ids": len({row["trade_id"] for row in trades}),
    }
    reconciliation["pass"] = (
        reconciliation["accepted_setups"] == reconciliation["terminal_dispositions"]
        == reconciliation["unique_setup_ids"]
        and reconciliation["executed_dispositions"] == reconciliation["trade_count"]
        == reconciliation["unique_trade_ids"]
    )
    if not reconciliation["pass"]:
        raise AsiaReplayError("aggregate setup/trade reconciliation failed")
    daily = _daily_rows(sessions, trades)
    period_rows = _group_metrics(eligible_sessions, trades, "period")
    month_sessions = [{**row, "month": str(row["session_date"])[:7]} for row in eligible_sessions]
    month_trades = [{**row, "month": str(row["date"])[:7]} for row in trades]
    month_rows = _group_metrics(month_sessions, month_trades, "month")
    direction_rows = _group_metrics([], trades, "direction")
    execution_rows = _group_metrics([], trades, "execution_model")
    source_rows = _group_metrics(eligible_sessions, trades, "source_model")
    period_results = [*period_rows, *month_rows, *direction_rows, *source_rows]
    execution_counts = Counter(str(row["execution_model"]) for row in trades)
    pre_quality_rejections: Counter[str] = Counter()
    for session in eligible_sessions:
        for row in session["interactions"]:
            for reason in str(row.get("rejection_reasons") or "").split(";"):
                if reason:
                    pre_quality_rejections[reason] += 1
    risk = _risk_audit(trades)
    if not risk["pass"]:
        raise AsiaReplayError("risk-budget audit failed")
    summary: dict[str, Any] = {
        "status": "ASIA_W04_REPLAY_COMPLETE",
        "strategy_id": STRATEGY_ID,
        "config_id": CONFIG_ID,
        "evidence_label": EVIDENCE_LABEL,
        "interpretation": INTERPRETATION,
        "eligible_session_count": len(eligible_sessions),
        "source_session_count": len(sessions),
        "profile_only_session_count": len(sessions) - len(eligible_sessions),
        "first_eligible_session": eligible_sessions[0]["session_date"],
        "last_eligible_session": eligible_sessions[-1]["session_date"],
        "session_contract": {
            "timezone": "UTC",
            "start_inclusive": "00:00:00",
            "end_exclusive": "08:00:00",
            "hard_flat": "08:00:00 using last valid ES BBO in [07:59:59,08:00:00]",
            "reference_level": ASIA_LEVEL_NAME,
        },
        "weights": {key: str(value) for key, value in W04_WEIGHTS.items()},
        "quality_threshold": str(QUALITY_THRESHOLD),
        "raw_interactions": sum(int(row["raw_interactions"]) for row in eligible_sessions),
        "accepted_setups": len(setups),
        "confirmations_passed": sum(int(row["confirmations_passed"]) for row in eligible_sessions),
        "confirmations_failed": len(setups) - sum(int(row["confirmations_passed"]) for row in eligible_sessions),
        "unresolved": sum(int(row["unresolved"]) for row in eligible_sessions),
        "terminal_disposition_counts": dict(sorted(terminal_counts.items())),
        "pre_quality_rejection_counts": dict(sorted(pre_quality_rejections.items())),
        "performance": performance,
        "execution_model_counts": dict(sorted(execution_counts.items())),
        "risk_budget_audit": risk,
        "reconciliation": reconciliation,
        "period_results": period_results,
        "daily_results": daily,
        "semantic_diff_status": semantic_diff["status"],
        "ny_artifacts_mutated": dict(ny_before) != dict(ny_after),
        "protected_ny_artifact_hashes": dict(ny_after),
        "source_verification": list(source_verification),
        "coverage_audit": {
            "path": str(audit_root),
            "summary_sha256": _sha256(audit_root / "summary.json"),
            "session_coverage_sha256": _sha256(audit_root / "session-coverage.csv"),
        },
        "network_calls": 0,
        "downloads": 0,
        "databento_api_calls": 0,
        "native_mes_files_opened": 0,
        "optimization_performed": False,
        "automatic_strategy_selection": False,
        "native_mes_evidence": False,
        "fresh_oos_evidence": False,
    }
    if summary["eligible_session_count"] != EXPECTED_ELIGIBLE_SESSIONS:
        raise AsiaReplayError("final output does not contain exactly 46 eligible sessions")
    if summary["ny_artifacts_mutated"]:
        raise AsiaReplayError("protected NY W04 source or artifact changed during Asia replay")
    eligible_rows = [{
        "session_date": row["session_date"],
        "period": row["period"],
        "source_model": row["source_model"],
        "prior_asia_session": row["prior_asia_session"],
        "prior_asia_poc": row["prior_asia_poc"],
        "session_poc": row["session_poc"],
        "execution_model": "ES_NATIVE_WITH_MES_MONETARY_PROXY_FALLBACK",
        "evidence_label": EVIDENCE_LABEL,
    } for row in eligible_sessions]
    _write_csv(staging / "eligible-sessions.csv", eligible_rows)
    _write_json(staging / "semantic-diff.json", semantic_diff)
    _write_csv(staging / "setup-ledger.csv", setups)
    _write_csv(staging / "trade-ledger.csv", trades)
    _write_csv(staging / "daily-results.csv", daily)
    _write_csv(staging / "period-results.csv", period_results)
    _write_csv(staging / "execution-model-breakdown.csv", execution_rows)
    _write_json(staging / "summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_report_markdown(summary), encoding="utf-8")
    (staging / "report.html").write_text(_report_html(summary), encoding="utf-8")
    return summary


def run_replay(
    *, repository_root: Path,
    output_root: Path = OUTPUT_ROOT,
    audit_root: Path = AUDIT_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_root = output_root if output_root.is_absolute() else repository_root / output_root
    audit_root = audit_root if audit_root.is_absolute() else repository_root / audit_root
    output_root = output_root.resolve()
    audit_root = audit_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable Asia output root already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    sessions = load_audit_sessions(audit_root, include_native=True)
    bindings = source_bindings(repository_root, sessions)
    semantic_diff = semantic_diff_document()
    ny_before = _protected_ny_snapshot(repository_root)
    source_verification = verify_source_bindings(bindings)
    sessions_by_day = {row.day: row for row in sessions}
    results: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for binding in bindings:
        binding_sessions = [sessions_by_day[day] for day in binding.days]
        if binding.shared:
            batch = _process_shared_binding(
                binding, binding_sessions, staging=staging, previous_result=previous,
            )
            results.extend(batch)
            previous = batch[-1]
        else:
            session = binding_sessions[0]
            print(
                f"ASIA_SESSION {len(results) + 1:02d}/{EXPECTED_SOURCE_SESSIONS:02d} "
                f"{session.day} eligible={str(session.eligible).lower()}",
                flush=True,
            )
            current = _process_daily_binding(
                binding, session, staging=staging, previous_result=previous,
            )
            results.append(current)
            previous = current
    results.sort(key=lambda row: str(row["session_date"]))
    if [str(row["session_date"]) for row in results] != [row.day for row in sessions]:
        raise AsiaReplayError("replay results do not match the audited source chronology")
    ny_after = _protected_ny_snapshot(repository_root)
    summary = _materialize(
        staging=staging,
        audit_root=audit_root,
        source_verification=source_verification,
        sessions=results,
        semantic_diff=semantic_diff,
        ny_before=ny_before,
        ny_after=ny_after,
    )
    work = staging / "_work"
    if work.exists():
        shutil.rmtree(work)
    checkpoints = staging / "_checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    run_manifest = {
        "status": summary["status"],
        "strategy_id": STRATEGY_ID,
        "config_id": CONFIG_ID,
        "evidence_label": EVIDENCE_LABEL,
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        },
        "network_calls": 0,
        "downloads": 0,
    }
    _write_json(staging / "run-manifest.json", run_manifest)
    os.rename(staging, output_root)
    return {**summary, "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_replay(
            repository_root=args.repository_root,
            output_root=args.output_root,
            audit_root=args.audit_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps({
        "status": result["status"],
        "strategy_id": result["strategy_id"],
        "eligible_sessions": result["eligible_session_count"],
        "trades": result["performance"]["completed_trades"],
        "total_r": result["performance"]["total_r"],
        "net_pnl_usd": result["performance"]["net_pnl_usd"],
        "output_root": result["output_root"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
