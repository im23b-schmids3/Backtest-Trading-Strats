"""Local historical replay for the frozen Dec-2025/Jan-2026 L2 V3 block.

The adapter audit and source-quality check are strategy-free.  The full replay
is a separate, manually invoked path gated by a passing audit artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from . import historical_runner as historical
from . import v3_poc_fresh_august_replay as native
from .dec2025_feb2026_quote import PROFILE_END, _session_end, build_sessions
from .model import (
    ENTRY_LATENCY_NS, ES_CAP, MAX_CONFIRMATION_NS, MES_CAP, MIN_CONFIRMATION_NS,
    RISK_BUDGET_USD, STOP_BUFFER_TICKS, TARGET_R, TICK, StructuralLevel,
)
from .v2_quality050 import V2_CONFIG
from .v3_poc_only import ELIGIBLE_STRUCTURAL_LEVELS, STRATEGY_ID, v3_contract, v3_contract_sha256


DATA_ROOT = Path("data/cme_orderflow_absorption_l2_v3/dec2025_jan2026")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO")
AUDIT_ROOT = Path("research_runs/L2_V3_DEC2025_JAN2026_DATA_PREFLIGHT")
AUDIT_OUTPUT = AUDIT_ROOT / "adapter-audit.json"
AUDIT_HTML = Path("docs/research_pipeline/cme_orderflow_absorption_v1/l2-v3-dec2025-jan2026-data-preflight.html")
SUPPLEMENT_DIRECTORY = "es_prior_rth_trades_supplement"
SUPPLEMENT_MANIFEST_NAME = "supplemental-acquisition-manifest.json"
EVIDENCE_LABEL = "RETROSPECTIVE_ROBUSTNESS_DEC2025_JAN2026_NOT_STRICT_OOS"
V3_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
FIRST_TARGET = date(2025, 12, 1)
LAST_TARGET = date(2026, 1, 30)
EXPECTED_SESSION_COUNT = 42
EXPECTED_COMPONENT_COUNT = 126
EXPECTED_PURPOSES = ("ES_MBP10", "MES_MBP1", "PRIOR_RTH_TRADES")
DEGRADED_SOURCE_DATE = date(2025, 11, 28)
DEGRADED_TARGET_DATE = date(2025, 12, 1)
RAW_PRICE_SCALE = 1_000_000_000
SOURCE_START_TOLERANCE_NS = 60_000_000_000
SOURCE_END_TOLERANCE_NS = 60_000_000_000
SOURCE_MAX_OBVIOUS_GAP_NS = 15 * 60 * 1_000_000_000
SOURCE_MIN_VALID_TRADES = 100
CUTOFF_QUOTE_LOOKBACK_NS = 1_000_000_000
NEW_YORK = ZoneInfo("America/New_York")
EARLY_RTH_CLOSE_DATES = frozenset({date(2025, 11, 28), date(2025, 12, 24)})
EXPECTED_SUPPLEMENT_COUNT = 40
EXPECTED_SUPPLEMENT_SYMBOLS = Counter({"ESZ5": 12, "ESH6": 28})
EXPECTED_SUPPLEMENT_TOTAL_USD = "2.396373689177"
PRE_QUALITY_RAW_COMPONENTS = (
    "directional_aggressive_volume",
    "opposite_aggressive_volume",
    "aggressive_volume_imbalance",
    "execution_count",
    "relevant_execution_count",
    "executed_volume_at_defended_price",
    "executed_volume_within_1_tick",
    "executed_volume_within_2_ticks",
    "execution_rate",
    "aggressive_volume_rate",
    "displayed_size_before_consumption",
    "size_consumed_by_execution",
    "minimum_displayed_size_after_consumption",
    "restored_size",
    "restoration_timestamp_ns",
    "depth_restoration_count",
    "restored_depth_volume",
    "mean_restoration_latency_ms",
    "median_restoration_latency_ms",
    "fastest_restoration_latency_ms",
    "consume_restore_cycles",
    "cumulative_consumed_volume",
    "cumulative_restored_volume",
    "restoration_to_consumption_ratio",
    "cumulative_executed_at_price",
    "initial_displayed_depth_at_price",
    "median_displayed_depth_at_price",
    "max_displayed_depth_at_price",
    "executed_to_initial_displayed_ratio",
    "executed_to_median_displayed_ratio",
    "maximum_through_level_progress_ticks",
    "final_through_level_progress_ticks",
    "interaction_rejection_ticks",
    "adverse_progress_per_100_aggressive_contracts",
    "aggressive_contracts_per_adverse_tick",
    "defended_price_present_fraction",
    "fraction_of_interaction_with_nonzero_defended_depth",
    "defended_depth_time_weighted_mean",
    "defended_depth_time_weighted_median",
    "defended_order_count_before_consumption",
    "minimum_order_count_after_consumption",
    "restored_order_count",
    "order_count_restoration_cycles",
    "bid_depth_1",
    "ask_depth_1",
    "bid_depth_3",
    "ask_depth_3",
    "bid_depth_5",
    "ask_depth_5",
    "depth_imbalance_1",
    "depth_imbalance_3",
    "depth_imbalance_5",
    "multi_level_ofi",
    "depth_recovery_100ms",
    "depth_recovery_250ms",
    "depth_recovery_500ms",
    "depth_recovery_1s",
    "unexecuted_add_volume",
    "rapid_cancel_volume",
    "rapid_cancel_ratio",
    "restoration_supported_by_execution_ratio",
    "restoration_away_from_defended_price_volume",
)


class DecJanReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(timestamp_ns: int) -> str:
    seconds, nanos = divmod(int(timestamp_ns), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanos:09d}Z"


def _ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DecJanReplayError(f"timestamp is not UTC: {value!r}")
    return int(parsed.timestamp() * 1_000_000_000)


def _ny_to_utc(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=NEW_YORK).astimezone(timezone.utc)


def _semantic_rth_window(day: date) -> tuple[datetime, datetime]:
    close = time(13, 0) if day in EARLY_RTH_CLOSE_DATES else time(16, 0)
    return _ny_to_utc(day, time(9, 30)), _ny_to_utc(day, close)


def _effective_hard_flat(day: date) -> datetime:
    if day in EARLY_RTH_CLOSE_DATES:
        return _ny_to_utc(day, time(13, 15))
    nominal = datetime.combine(day, time(22, 45), tzinfo=timezone.utc)
    maintenance_start = _ny_to_utc(day, time(17, 0))
    maintenance_end = _ny_to_utc(day, time(18, 0))
    return maintenance_start if maintenance_start <= nominal < maintenance_end else nominal


def _iso_seconds(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _interval_intersection(start: int, end: int, required_start: int, required_end: int) -> tuple[int, int] | None:
    left, right = max(start, required_start), min(end, required_end)
    return (left, right) if left < right else None


def _interval_payload(value: tuple[int, int] | None) -> dict[str, str] | None:
    if value is None:
        return None
    return {"start_utc": _utc(value[0]), "end_utc": _utc(value[1])}


def _load_manifest(data_root: Path) -> tuple[Path, dict[str, Any]]:
    path = data_root / "acquisition-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecJanReplayError("missing or unreadable Dec/Jan acquisition manifest") from exc
    if not isinstance(payload, dict):
        raise DecJanReplayError("acquisition manifest root is not an object")
    return path, payload


def _expected_sessions() -> tuple[Any, ...]:
    sessions = build_sessions(FIRST_TARGET, LAST_TARGET)
    if len(sessions) != EXPECTED_SESSION_COUNT:
        raise DecJanReplayError("frozen calendar no longer produces exactly 42 sessions")
    return sessions


def _validate_frozen_strategy() -> None:
    if v3_contract_sha256() != V3_CONTRACT_SHA256:
        raise DecJanReplayError("frozen V3 contract hash mismatch")
    if ELIGIBLE_STRUCTURAL_LEVELS != ("PRIOR_RTH_POC",):
        raise DecJanReplayError("V3 structural eligibility changed")
    execution = v3_contract()["execution"]
    expected = {
        "confirmation_window_seconds_inclusive": [5.0, 15.0],
        "confirmation_favorable_ticks": 3,
        "entry_latency_ms": 2.0,
        "stop_buffer_ticks": 5,
        "target_r": 3.0,
        "risk_budget_usd": 250.0,
        "es_first": True,
        "mes_fallback": True,
        "max_es_contracts": 6,
        "max_mes_contracts": 60,
    }
    if execution != expected or V2_CONFIG.min_quality_score != 0.50:
        raise DecJanReplayError("frozen V3 execution or quality threshold changed")
    if (
        MIN_CONFIRMATION_NS != 5_000_000_000 or MAX_CONFIRMATION_NS != 15_000_000_000
        or ENTRY_LATENCY_NS != 2_000_000 or STOP_BUFFER_TICKS != 5 or TARGET_R != 3.0
        or RISK_BUDGET_USD != 250.0 or ES_CAP != 6 or MES_CAP != 60
    ):
        raise DecJanReplayError("model execution literals no longer match frozen V3")


def _component_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != EXPECTED_COMPONENT_COUNT:
        raise DecJanReplayError("manifest must declare exactly 126 acquisition components")
    result: dict[str, Mapping[str, Any]] = {}
    for item in components:
        if not isinstance(item, dict) or not isinstance(item.get("local_path"), str):
            raise DecJanReplayError("malformed acquisition component")
        relative = str(item["local_path"])
        if relative in result:
            raise DecJanReplayError(f"duplicate acquisition component path: {relative}")
        result[relative] = item
    return result


def _session_inputs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_day: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in _component_map(manifest).values():
        target, purpose = str(item.get("target_session")), str(item.get("purpose"))
        if purpose not in EXPECTED_PURPOSES or purpose in by_day[target]:
            raise DecJanReplayError(f"invalid or duplicate component purpose for {target}: {purpose}")
        by_day[target][purpose] = item
    if any(set(items) != set(EXPECTED_PURPOSES) for items in by_day.values()):
        raise DecJanReplayError("every target session must declare ES, MES, and prior-RTH inputs")
    return dict(by_day)


def _validate_component_identity(day: date, expected: Any, items: Mapping[str, Mapping[str, Any]]) -> None:
    es, mes, prior = items["ES_MBP10"], items["MES_MBP1"], items["PRIOR_RTH_TRADES"]
    target_end = f"{day.isoformat()}T{expected.effective_end.isoformat()}Z"
    target_end_plus_one = _utc(_ns(target_end) + 1_000_000_000)
    target_end_plus_one = target_end_plus_one.replace(".000000000Z", "Z")
    prior_day = expected.previous_rth_date
    prior_end = min(PROFILE_END, _session_end(prior_day)).isoformat()
    requirements = (
        (es, "GLBX.MDP3", "mbp-10", f"{day}T13:00:00Z", target_end_plus_one, None),
        (mes, "GLBX.MDP3", "mbp-1", f"{day}T13:30:00Z", target_end_plus_one, None),
        (prior, "GLBX.MDP3", "trades", f"{prior_day}T13:30:00Z", f"{prior_day}T{prior_end}Z", str(prior_day)),
    )
    for item, dataset, schema, start, end, prior_value in requirements:
        if (
            item.get("dataset") != dataset or item.get("schema") != schema
            or item.get("start_utc") != start or item.get("end_utc") != end
            or item.get("prior_rth_date") != prior_value
        ):
            raise DecJanReplayError(f"component identity/window mismatch for {day}/{schema}")
    expected_symbols = (("ESZ5", "MESZ5") if day <= date(2025, 12, 16) else ("ESH6", "MESH6"))
    expected_prior_es = "ESZ5" if prior_day <= date(2025, 12, 16) else "ESH6"
    if (es.get("raw_symbol"), mes.get("raw_symbol"), prior.get("raw_symbol")) != (*expected_symbols, expected_prior_es):
        raise DecJanReplayError(f"contract-roll mapping mismatch for {day}")
    if day == date(2025, 12, 17) and (prior_day != date(2025, 12, 16) or prior.get("raw_symbol") != "ESZ5"):
        raise DecJanReplayError("December 17 prior-RTH roll transition changed")


def verify_acquisition_manifest(data_root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Verify all 126 manifest-bound inputs without parsing strategy outcomes."""
    _validate_frozen_strategy()
    manifest_path, manifest = _load_manifest(data_root)
    constraints = manifest.get("constraints", {})
    if (
        manifest.get("status") != "ACQUISITION_COMPLETE_VERIFIED"
        or manifest.get("strategy_id") != STRATEGY_ID
        or manifest.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or manifest.get("evidence_classification") != EVIDENCE_LABEL
        or manifest.get("target_session_count") != EXPECTED_SESSION_COUNT
        or manifest.get("first_target_session") != FIRST_TARGET.isoformat()
        or manifest.get("last_target_session") != LAST_TARGET.isoformat()
        or constraints.get("no_mbo") is not True
        or constraints.get("no_strategy_replay") is not True
        or constraints.get("no_outcomes_inspected") is not True
        or constraints.get("february_excluded") is not True
    ):
        raise DecJanReplayError("acquisition manifest does not bind the frozen Dec/Jan V3 package")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != EXPECTED_COMPONENT_COUNT:
        raise DecJanReplayError("acquisition manifest files must contain exactly 126 entries")
    component_by_path = _component_map(manifest)
    if set(files) != set(component_by_path):
        raise DecJanReplayError("manifest component/file paths disagree")
    sessions = _expected_sessions()
    by_day = _session_inputs(manifest)
    if list(by_day) != [item.session_date.isoformat() for item in sessions]:
        raise DecJanReplayError("manifest target sessions are missing, extra, duplicated, or unordered")
    prior_map = manifest.get("prior_rth_by_target_session")
    expected_prior = {str(item.session_date): str(item.previous_rth_date) for item in sessions}
    if prior_map != expected_prior or expected_prior[FIRST_TARGET.isoformat()] != "2025-11-28":
        raise DecJanReplayError("manifest prior-RTH mapping changed")

    counts: Counter[str] = Counter()
    verified: list[dict[str, Any]] = []
    total_bytes = 0
    for session in sessions:
        day = session.session_date
        items = by_day[str(day)]
        _validate_component_identity(day, session, items)
        for purpose, component in items.items():
            relative = str(component["local_path"])
            record, local = files[relative], data_root / relative
            if not isinstance(record, dict) or any(record.get(key) != component.get(key) for key in (
                "target_session", "prior_rth_date", "purpose", "dataset", "schema", "raw_symbol",
                "start_utc", "end_utc", "local_path",
            )):
                raise DecJanReplayError(f"manifest file/component disagreement: {relative}")
            if record.get("status") != "DOWNLOADED_VERIFIED" or not local.is_file():
                raise DecJanReplayError(f"required acquired input missing: {relative}")
            size = local.stat().st_size
            if size <= 0 or size != int(record.get("bytes", -1)):
                raise DecJanReplayError(f"input size mismatch: {relative}")
            if verify_hashes and _sha256(local) != record.get("sha256"):
                raise DecJanReplayError(f"input SHA-256 mismatch: {relative}")
            counts[purpose] += 1
            total_bytes += size
            verified.append({"relative_path": relative, "bytes": size, "sha256": record["sha256"]})
    if counts != Counter({purpose: EXPECTED_SESSION_COUNT for purpose in EXPECTED_PURPOSES}):
        raise DecJanReplayError("component cardinality is not 42/42/42")
    allowed = {"acquisition-manifest.json"} | set(files)
    actual = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and SUPPLEMENT_DIRECTORY not in path.relative_to(data_root).parts
    }
    if actual != allowed:
        raise DecJanReplayError("data root contains missing or undeclared files")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "files_verified": len(verified),
        "hashes_verified": verify_hashes,
        "total_bytes": total_bytes,
        "by_purpose": dict(counts),
        "target_sessions": [str(item.session_date) for item in sessions],
        "prior_rth_mapping": expected_prior,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "quality_threshold": V2_CONFIG.min_quality_score,
        "strategy_or_outcomes_accessed": False,
        "session_inputs": by_day,
    }


def verify_supplemental_manifest(data_root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Verify the 40 immutable 20:00-21:00 prior-RTH tails."""
    root = data_root / SUPPLEMENT_DIRECTORY
    manifest_path = root / SUPPLEMENT_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecJanReplayError("missing or unreadable supplemental acquisition manifest") from exc
    constraints = manifest.get("constraints", {})
    if (
        manifest.get("status") != "SUPPLEMENT_ACQUISITION_COMPLETE_VERIFIED"
        or manifest.get("strategy_id") != STRATEGY_ID
        or manifest.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or manifest.get("dataset") != "GLBX.MDP3"
        or manifest.get("schema") != "trades"
        or manifest.get("request_count") != EXPECTED_SUPPLEMENT_COUNT
        or manifest.get("verified_file_count") != EXPECTED_SUPPLEMENT_COUNT
        or manifest.get("symbol_counts") != dict(EXPECTED_SUPPLEMENT_SYMBOLS)
        or manifest.get("approved_quote", {}).get("total_usd") != EXPECTED_SUPPLEMENT_TOTAL_USD
        or constraints.get("supplemental_tails_only") is not True
        or constraints.get("no_overlapping_1430_2000_purchase") is not True
        or constraints.get("no_mbp_10") is not True
        or constraints.get("no_mbp_1") is not True
        or constraints.get("no_mbo") is not True
        or constraints.get("strategy_executed") is not False
        or constraints.get("outcomes_inspected") is not False
    ):
        raise DecJanReplayError("supplemental manifest identity or safety contract mismatch")
    requests, files = manifest.get("requests"), manifest.get("files")
    if not isinstance(requests, list) or len(requests) != 40 or not isinstance(files, dict) or len(files) != 40:
        raise DecJanReplayError("supplemental manifest must declare exactly 40 requests and files")

    source_artifacts = (
        (manifest.get("calendar_audit", {}), "calendar audit"),
        (manifest.get("approved_quote", {}), "supplement quote"),
    )
    for declaration, label in source_artifacts:
        path = Path(str(declaration.get("path", "")))
        if not path.is_file() or _sha256(path) != declaration.get("sha256"):
            raise DecJanReplayError(f"supplemental {label} binding mismatch")

    symbols: Counter[str] = Counter()
    by_target: dict[str, dict[str, Any]] = {}
    verified_files: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise DecJanReplayError(f"malformed supplemental request {index}")
        required = {
            "index": index,
            "dataset": "GLBX.MDP3",
            "schema": "trades",
            "stype_in": "raw_symbol",
        }
        if any(request.get(key) != value for key, value in required.items()):
            raise DecJanReplayError(f"supplemental request identity mismatch at index {index}")
        symbol, target, prior = str(request.get("raw_symbol")), str(request.get("target_session")), str(request.get("prior_rth_date"))
        start, end = str(request.get("start_utc")), str(request.get("end_utc"))
        if (
            symbol not in EXPECTED_SUPPLEMENT_SYMBOLS
            or start != f"{prior}T20:00:00Z"
            or end != f"{prior}T21:00:00Z"
            or target in by_target
        ):
            raise DecJanReplayError(f"supplemental symbol/window/target mismatch at index {index}")
        relative = str(request.get("local_path"))
        expected_name = f"{symbol}_{prior}_200000_210000_trades.dbn.zst"
        if relative != expected_name or Path(relative).name != relative:
            raise DecJanReplayError(f"supplemental filename mismatch at index {index}")
        record, local = files.get(relative), root / relative
        if not isinstance(record, dict):
            raise DecJanReplayError(f"supplemental file record missing: {relative}")
        compared = (
            "index", "target_session", "prior_rth_date", "dataset", "schema", "stype_in",
            "raw_symbol", "start_utc", "end_utc", "local_path",
        )
        if any(record.get(key) != request.get(key) for key in compared):
            raise DecJanReplayError(f"supplemental file/request disagreement: {relative}")
        if record.get("status") != "DOWNLOADED_VERIFIED" or not local.is_file():
            raise DecJanReplayError(f"supplemental file missing: {relative}")
        size = local.stat().st_size
        if size <= 0 or size != int(record.get("bytes", -1)):
            raise DecJanReplayError(f"supplemental file size mismatch: {relative}")
        if verify_hashes and _sha256(local) != record.get("sha256"):
            raise DecJanReplayError(f"supplemental file SHA-256 mismatch: {relative}")
        symbols[symbol] += 1
        by_target[target] = {**request, "absolute_path": str(local), "bytes": size, "sha256": record.get("sha256")}
        verified_files.append({"relative_path": relative, "bytes": size, "sha256": record.get("sha256")})
    if symbols != EXPECTED_SUPPLEMENT_SYMBOLS:
        raise DecJanReplayError("supplemental symbol cardinality is not 12 ESZ5 / 28 ESH6")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != {SUPPLEMENT_MANIFEST_NAME, *files.keys()}:
        raise DecJanReplayError("supplement directory contains missing or undeclared files")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "files_verified": len(verified_files),
        "hashes_verified": verify_hashes,
        "total_bytes": sum(int(row["bytes"]) for row in verified_files),
        "symbol_counts": dict(symbols),
        "approved_total_usd": EXPECTED_SUPPLEMENT_TOTAL_USD,
        "by_target_session": by_target,
        "strategy_or_outcomes_accessed": False,
    }


def _covers(required: tuple[int, int], intervals: Sequence[tuple[int, int]]) -> tuple[bool, list[tuple[int, int]]]:
    cursor, required_end = required
    missing: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor:
            missing.append((cursor, min(start, required_end)))
        cursor = max(cursor, end)
        if cursor >= required_end:
            break
    if cursor < required_end:
        missing.append((cursor, required_end))
    return not missing, missing


def build_data_sufficiency_matrix(
    base: Mapping[str, Any], supplement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconcile acquired declarations against canonical winter semantics."""
    rows: list[dict[str, Any]] = []
    supplement_by_target = supplement["by_target_session"]
    for day_text in base["target_sessions"]:
        day = date.fromisoformat(day_text)
        inputs = base["session_inputs"][day_text]
        es, mes, prior = inputs["ES_MBP10"], inputs["MES_MBP1"], inputs["PRIOR_RTH_TRADES"]
        prior_day = date.fromisoformat(str(prior["prior_rth_date"]))
        prior_start_dt, prior_end_dt = _semantic_rth_window(prior_day)
        prior_required = (_ns(_iso_seconds(prior_start_dt)), _ns(_iso_seconds(prior_end_dt)))
        base_interval = _interval_intersection(
            _ns(str(prior["start_utc"])), _ns(str(prior["end_utc"])), *prior_required,
        )
        supplemental = supplement_by_target.get(day_text)
        supplement_interval = None if supplemental is None else _interval_intersection(
            _ns(str(supplemental["start_utc"])), _ns(str(supplemental["end_utc"])), *prior_required,
        )
        effective_intervals = [value for value in (base_interval, supplement_interval) if value is not None]
        prior_sufficient, missing = _covers(prior_required, effective_intervals)

        rth_start, _ = _semantic_rth_window(day)
        hard_flat = _effective_hard_flat(day)
        es_required = (_ns(_iso_seconds(rth_start - timedelta(minutes=30))), _ns(_iso_seconds(hard_flat + timedelta(seconds=1))))
        mes_required = (_ns(_iso_seconds(rth_start)), es_required[1])
        es_acquired = (_ns(str(es["start_utc"])), _ns(str(es["end_utc"])))
        mes_acquired = (_ns(str(mes["start_utc"])), _ns(str(mes["end_utc"])))
        es_sufficient = es_acquired[0] <= es_required[0] and es_acquired[1] >= es_required[1]
        mes_sufficient = mes_acquired[0] <= mes_required[0] and mes_acquired[1] >= mes_required[1]
        final_start = min((value[0] for value in effective_intervals), default=None)
        final_end = max((value[1] for value in effective_intervals), default=None)
        rows.append({
            "session_date": day_text,
            "target_es_symbol": es["raw_symbol"],
            "target_mes_symbol": mes["raw_symbol"],
            "prior_rth_date": str(prior_day),
            "prior_rth_es_symbol": prior["raw_symbol"],
            "required_prior_rth_utc": _interval_payload(prior_required),
            "base_covered_range": _interval_payload(base_interval),
            "supplement_covered_range": _interval_payload(supplement_interval),
            "final_effective_covered_range": _interval_payload((final_start, final_end)) if final_start is not None and final_end is not None else None,
            "missing_prior_rth_ranges": [_interval_payload(value) for value in missing],
            "required_es_mbp10_utc": _interval_payload(es_required),
            "required_mes_mbp1_utc": _interval_payload(mes_required),
            "effective_hard_flat_utc": _iso_seconds(hard_flat),
            "es_mbp10_sufficient": es_sufficient,
            "mes_mbp1_sufficient": mes_sufficient,
            "prior_rth_sufficient": prior_sufficient,
            "overall_sufficient": es_sufficient and mes_sufficient and prior_sufficient,
        })
    return rows


def verify_data_preflight(data_root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    base = verify_acquisition_manifest(data_root, verify_hashes=verify_hashes)
    supplement = verify_supplemental_manifest(data_root, verify_hashes=verify_hashes)
    matrix = build_data_sufficiency_matrix(base, supplement)
    excluded = [row["session_date"] for row in matrix if not row["overall_sufficient"]]
    return {
        "status": "ALL_42_SESSIONS_DATA_SUFFICIENT" if not excluded and len(matrix) == 42 else "DATA_SUFFICIENCY_FAIL_CLOSED",
        "base": base,
        "supplement": supplement,
        "sufficiency_matrix": matrix,
        "sufficient_session_count": sum(bool(row["overall_sufficient"]) for row in matrix),
        "excluded_sessions": excluded,
        "strategy_or_outcomes_accessed": False,
    }


def _record_timestamp(record: object) -> int:
    value = getattr(record, "ts_event", None)
    if value is None:
        value = getattr(record, "ts_recv")
    return int(value)


def _stream_trade_records(path: Path) -> Iterable[object]:
    from databento import DBNStore
    yield from DBNStore.from_file(path)


def _profile_levels_from_trade_sources(
    sources: Sequence[Path], *, required_start_ns: int, required_end_ns: int,
) -> list[StructuralLevel]:
    """Build one profile from chronological sources while exposing only RTH."""
    by_price: Counter[int] = Counter()
    previous_timestamp: int | None = None
    included_records = 0
    for path in sources:
        for record in _stream_trade_records(path):
            timestamp = _record_timestamp(record)
            if timestamp < required_start_ns or timestamp >= required_end_ns:
                continue
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise DecJanReplayError("prior-RTH sources are not chronologically mergeable")
            previous_timestamp = timestamp
            price, size = int(getattr(record, "price", 0)), int(getattr(record, "size", 0))
            if price <= 0 or price >= historical.UNDEF_PRICE or size <= 0:
                raise DecJanReplayError("malformed trade inside semantic prior-RTH window")
            by_price[price] += size
            included_records += 1
    if not by_price or included_records < 1:
        raise DecJanReplayError("semantic prior-RTH sources cannot build a structural profile")
    tick = int(TICK * RAW_PRICE_SCALE)
    poc = min(by_price, key=lambda price: (-by_price[price], price))
    total, included, low, high = sum(by_price.values()), by_price[poc], poc, poc
    while included * 100 < total * 70:
        below, above = low - tick, high + tick
        if by_price.get(below, 0) >= by_price.get(above, 0):
            low, included = below, included + by_price.get(below, 0)
        else:
            high, included = above, included + by_price.get(above, 0)
    return [StructuralLevel(name, value / RAW_PRICE_SCALE) for name, value in (
        ("PRIOR_RTH_HIGH", max(by_price)), ("PRIOR_RTH_LOW", min(by_price)), ("PRIOR_RTH_POC", poc),
        ("PRIOR_RTH_VAH", high), ("PRIOR_RTH_VAL", low),
    )]


def _profile_source_spec(preflight: Mapping[str, Any], data_root: Path, day_text: str) -> dict[str, Any]:
    row = next(item for item in preflight["sufficiency_matrix"] if item["session_date"] == day_text)
    if not row["overall_sufficient"]:
        raise DecJanReplayError(f"profile source requested for insufficient session: {day_text}")
    base_item = preflight["base"]["session_inputs"][day_text]["PRIOR_RTH_TRADES"]
    paths = [data_root / str(base_item["local_path"])]
    supplemental = preflight["supplement"]["by_target_session"].get(day_text)
    if supplemental is not None:
        paths.append(Path(str(supplemental["absolute_path"])))
    required = row["required_prior_rth_utc"]
    return {
        "paths": tuple(paths),
        "required_start_ns": _ns(str(required["start_utc"])),
        "required_end_ns": _ns(str(required["end_utc"])),
    }


def _validated_merged_poc(preflight: Mapping[str, Any], data_root: Path, day_text: str) -> StructuralLevel:
    spec = _profile_source_spec(preflight, data_root, day_text)
    selected = [
        level for level in _profile_levels_from_trade_sources(
            spec["paths"],
            required_start_ns=spec["required_start_ns"],
            required_end_ns=spec["required_end_ns"],
        )
        if level.name in ELIGIBLE_STRUCTURAL_LEVELS
    ]
    if len(selected) != 1 or selected[0].name != "PRIOR_RTH_POC":
        raise DecJanReplayError("merged prior-RTH profile did not produce exactly one eligible POC")
    return selected[0]


def audit_degraded_nov28_source(data_root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Outcome-independent trade-stream integrity decision for the Dec-1 POC."""
    day_text = DEGRADED_TARGET_DATE.isoformat()
    spec = _profile_source_spec(preflight, data_root, day_text)
    path = spec["paths"][0]
    required_start, required_end = spec["required_start_ns"], spec["required_end_ns"]
    count = valid = malformed = decreasing = excluded_outside = gaps = 0
    first = last = previous = largest_gap = None
    for record in _stream_trade_records(path):
        timestamp = _record_timestamp(record)
        if timestamp < required_start or timestamp >= required_end:
            excluded_outside += 1
            continue
        count += 1
        if first is None:
            first = timestamp
        if previous is not None:
            if timestamp < previous:
                decreasing += 1
            delta = timestamp - previous
            if delta > SOURCE_MAX_OBVIOUS_GAP_NS:
                gaps += 1
            largest_gap = delta if largest_gap is None else max(largest_gap, delta)
        previous = last = timestamp
        price, size = int(getattr(record, "price", 0)), int(getattr(record, "size", 0))
        if 0 < price < historical.UNDEF_PRICE and size > 0:
            valid += 1
        else:
            malformed += 1
    profile_constructed = False
    try:
        level = _validated_merged_poc(preflight, data_root, day_text)
        profile_constructed = isinstance(level, StructuralLevel) and level.name == "PRIOR_RTH_POC"
    except (OSError, ValueError, historical.HistoricalReplayError, DecJanReplayError):
        profile_constructed = False
    start_gap = first - required_start if first is not None else None
    end_gap = required_end - last if last is not None else None
    usable = bool(
        count >= SOURCE_MIN_VALID_TRADES and valid == count and malformed == 0 and decreasing == 0
        and start_gap is not None and 0 <= start_gap <= SOURCE_START_TOLERANCE_NS
        and end_gap is not None and 0 < end_gap <= SOURCE_END_TOLERANCE_NS
        and gaps == 0 and profile_constructed
    )
    reasons = []
    for failed, reason in (
        (count < SOURCE_MIN_VALID_TRADES, "INSUFFICIENT_VALID_TRADE_RECORDS"),
        (malformed > 0 or valid != count, "MALFORMED_TRADE_RECORDS"),
        (decreasing > 0, "DECREASING_EVENT_TIMESTAMPS"),
        (start_gap is None or start_gap < 0 or start_gap > SOURCE_START_TOLERANCE_NS, "RTH_START_COVERAGE_INSUFFICIENT"),
        (end_gap is None or end_gap <= 0 or end_gap > SOURCE_END_TOLERANCE_NS, "RTH_END_COVERAGE_INSUFFICIENT"),
        (gaps > 0, "OBVIOUS_EVENT_GAP_OVER_15_MINUTES"),
        (not profile_constructed, "PRIOR_RTH_POC_NOT_CONSTRUCTIBLE"),
    ):
        if failed:
            reasons.append(reason)
    return {
        "source_date": str(DEGRADED_SOURCE_DATE),
        "target_session": str(DEGRADED_TARGET_DATE),
        "provider_condition": "DATABENTO_DEGRADED_DATA_WARNING",
        "decision": "USABLE_WITH_DEGRADED_SOURCE_WARNING" if usable else "SOURCE_QUALITY_FAIL_CLOSED",
        "usable": usable,
        "decision_reasons": reasons,
        "record_count": count,
        "semantic_rth_start_utc": _utc(required_start),
        "semantic_rth_end_utc": _utc(required_end),
        "excluded_outside_semantic_rth_count": excluded_outside,
        "valid_trade_record_count": valid,
        "malformed_record_count": malformed,
        "first_timestamp_utc": _utc(first) if first is not None else None,
        "last_timestamp_utc": _utc(last) if last is not None else None,
        "decreasing_timestamp_count": decreasing,
        "outside_declared_window_count": 0,
        "obvious_gap_count_over_15_minutes": gaps,
        "largest_inter_record_gap_ns": largest_gap,
        "profile_constructible": profile_constructed,
        "outcome_or_pnl_consulted": False,
    }


def _record_action_side(record: object) -> tuple[str, str]:
    return native._code(getattr(record, "action", "")), native._code(getattr(record, "side", ""))


def _maintenance_window(day: date) -> tuple[int, int]:
    """CME equity-index daily halt in UTC for this standard-time block."""
    start = historical._clock_ns(day.isoformat(), 22 * 3600)
    return start, start + 3600 * 1_000_000_000


def _in_maintenance(timestamp_ns: int, day: date) -> bool:
    start, end = _maintenance_window(day)
    return start <= timestamp_ns < end


def _episode_classification(expected_maintenance: bool, state: str) -> str:
    if expected_maintenance:
        return "SCHEDULED_MAINTENANCE"
    if state == "TEMPORARILY_NON_EXECUTABLE":
        return "ORDINARY_TEMPORARY_RECONSTRUCTION"
    return "ORDINARY_WAITING_FOR_REOPEN_BOOK"


def _audit_native_mbp10_session(task: tuple[int, str, str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Audit one independent session; safe for deterministic process dispatch."""
    index, data_root_text, day_text, item, sufficiency = task
    day = date.fromisoformat(day_text)
    path = Path(data_root_text) / str(item["local_path"])
    adapter = native.NativeMBP10Adapter()
    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    records = maintenance_episodes = temporary_episodes = 0
    last_executable_ns: int | None = None
    exceptions: list[dict[str, Any]] = []
    cutoff_ns = _ns(str(sufficiency["effective_hard_flat_utc"]))
    rth_start_ns = _ns(str(sufficiency["required_mes_mbp1_utc"]["start_utc"]))
    try:
        for record_number, record in enumerate(native._stream_native_mbp10_records(path), start=1):
            timestamp = native._timestamp(record)
            if timestamp >= cutoff_ns:
                break
            records = record_number
            action, side = _record_action_side(record)
            expected_maintenance = _in_maintenance(timestamp, day)
            public = adapter.feed(record, expected_non_executable=expected_maintenance)
            new_state = adapter.state
            if public is not None:
                last_executable_ns = timestamp
            if public is None and adapter.first_valid_book_ns is not None:
                if active is None:
                    classification = _episode_classification(expected_maintenance, new_state)
                    active = {
                        "session": day_text,
                        "classification": classification,
                        "start_record": record_number,
                        "end_record": record_number,
                        "start_timestamp_ns": timestamp,
                        "end_timestamp_ns": timestamp,
                        "record_count": 0,
                        "initiating_action": action,
                        "initiating_side": side,
                        "states_observed": [],
                    }
                active["end_record"] = record_number
                active["end_timestamp_ns"] = timestamp
                active["record_count"] += 1
                if new_state not in active["states_observed"]:
                    active["states_observed"].append(new_state)
            elif public is not None and active is not None:
                active.update({
                    "start_timestamp_utc": _utc(int(active["start_timestamp_ns"])),
                    "end_timestamp_utc": _utc(int(active["end_timestamp_ns"])),
                    "reopen_timestamp_ns": timestamp,
                    "reopen_timestamp_utc": _utc(timestamp),
                    "duration_ns": timestamp - int(active["start_timestamp_ns"]),
                    "reopen_bbo": {
                        "bid": public.snapshot.bids[0].price,
                        "ask": public.snapshot.asks[0].price,
                    },
                    "resolved": True,
                })
                episodes.append(active)
                if active["classification"] == "SCHEDULED_MAINTENANCE":
                    maintenance_episodes += 1
                else:
                    temporary_episodes += 1
                active = None
            if record_number % 2_000_000 == 0:
                print(
                    f"adapter-audit {index:02d}/42 {day_text} records={record_number:,} "
                    f"episodes={len(episodes)} state={adapter.state}",
                    file=sys.stderr, flush=True,
                )
        adapter.assert_initialized_before(rth_start_ns)
        adapter.assert_executable_at_boundary()
    except Exception as exc:
        exceptions.append({
            "type": type(exc).__name__, "message": str(exc), "record_number": records,
            "state": adapter.state,
        })
    unresolved: list[dict[str, Any]] = []
    if active is not None:
        active.update({
            "start_timestamp_utc": _utc(int(active["start_timestamp_ns"])),
            "end_timestamp_utc": _utc(int(active["end_timestamp_ns"])),
            "duration_ns": int(active["end_timestamp_ns"]) - int(active["start_timestamp_ns"]),
            "reopen_bbo": None,
            "resolved": False,
        })
        unresolved.append(active)
        episodes.append(active)
        if active["classification"] == "SCHEDULED_MAINTENANCE":
            maintenance_episodes += 1
        else:
            temporary_episodes += 1
    cutoff_quote_available = bool(
        last_executable_ns is not None and cutoff_ns - CUTOFF_QUOTE_LOOKBACK_NS <= last_executable_ns <= cutoff_ns
    )
    if not cutoff_quote_available:
        unresolved.append({
            "session": day_text,
            "classification": "NO_EXECUTABLE_BBO_IN_FROZEN_CUTOFF_WINDOW",
            "cutoff_timestamp_utc": _utc(cutoff_ns),
            "last_executable_timestamp_utc": _utc(last_executable_ns) if last_executable_ns else None,
            "resolved": False,
        })
    session = {
        "session": day_text,
        "records_processed": records,
        "maintenance_episodes": maintenance_episodes,
        "temporary_non_executable_episodes": temporary_episodes,
        "non_executable_episodes": episodes,
        "unresolved_episodes": unresolved,
        "adapter_exceptions": exceptions,
        "last_executable_timestamp_utc": _utc(last_executable_ns) if last_executable_ns else None,
        "cutoff_timestamp_utc": _utc(cutoff_ns),
        "cutoff_quote_available": cutoff_quote_available,
        "effective_hard_flat_boundary_available": cutoff_quote_available,
        "final_state": adapter.state,
        "stale_bbo_exposure_count": 0,
    }
    print(
        f"adapter-audit complete {index:02d}/42 {day_text} records={records:,} "
        f"maintenance={maintenance_episodes} temporary={temporary_episodes} "
        f"unresolved={len(unresolved)} exceptions={len(exceptions)}",
        file=sys.stderr, flush=True,
    )
    return session


def audit_native_mbp10(
    data_root: Path, *, audit_output: Path | None = None, audit_html: Path | None = None,
    verify_hashes: bool = True, workers: int = 1,
) -> dict[str, Any]:
    """Parse all 42 ES inputs through the repaired adapter, never strategy code."""
    if workers < 1:
        raise DecJanReplayError("adapter audit workers must be at least one")
    preflight = verify_data_preflight(data_root, verify_hashes=verify_hashes)
    if preflight["status"] != "ALL_42_SESSIONS_DATA_SUFFICIENT":
        raise DecJanReplayError(f"adapter audit blocked by insufficient sessions: {preflight['excluded_sessions']}")
    verification = preflight["base"]
    source_quality = audit_degraded_nov28_source(data_root, preflight)
    matrix = {row["session_date"]: row for row in preflight["sufficiency_matrix"]}
    tasks = [
        (index, str(data_root), day_text, verification["session_inputs"][day_text]["ES_MBP10"], matrix[day_text])
        for index, day_text in enumerate(verification["target_sessions"], start=1)
    ]
    if workers == 1:
        sessions = [_audit_native_mbp10_session(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            sessions = list(executor.map(_audit_native_mbp10_session, tasks))
    sessions.sort(key=lambda item: str(item["session"]))
    totals = Counter()
    for session in sessions:
        totals.update({
            "records": int(session["records_processed"]),
            "maintenance_episodes": int(session["maintenance_episodes"]),
            "temporary_non_executable_episodes": int(session["temporary_non_executable_episodes"]),
            "unresolved_episodes": len(session["unresolved_episodes"]),
            "adapter_exceptions": len(session["adapter_exceptions"]),
            "hard_flat_boundaries_available": int(bool(session["effective_hard_flat_boundary_available"])),
            "stale_bbo_exposures": int(session["stale_bbo_exposure_count"]),
        })
    passed = (
        len(sessions) == EXPECTED_SESSION_COUNT
        and totals["unresolved_episodes"] == 0
        and totals["adapter_exceptions"] == 0
        and totals["hard_flat_boundaries_available"] == EXPECTED_SESSION_COUNT
        and totals["stale_bbo_exposures"] == 0
        and source_quality["usable"] is True
    )
    payload = {
        "audit_kind": "READ_ONLY_NATIVE_MBP10_ADAPTER_AUDIT",
        "status": "ADAPTER_AUDIT_PASS" if passed else "ADAPTER_AUDIT_FAIL_CLOSED",
        "strategy_id": STRATEGY_ID,
        "evidence_label": EVIDENCE_LABEL,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "quality_threshold": V2_CONFIG.min_quality_score,
        "manifest_sha256": verification["manifest_sha256"],
        "supplement_manifest_verification": {
            key: value for key, value in preflight["supplement"].items() if key != "by_target_session"
        },
        "data_sufficiency_status": preflight["status"],
        "data_sufficiency_matrix": preflight["sufficiency_matrix"],
        "sufficient_session_count": preflight["sufficient_session_count"],
        "target_session_count": EXPECTED_SESSION_COUNT,
        "source_quality_nov28": source_quality,
        "source_quality_excluded_sessions": [] if source_quality["usable"] else [str(DEGRADED_TARGET_DATE)],
        "strategy_runner_invoked": False,
        "setups_created": False,
        "trades_created": False,
        "pnl_or_outcomes_accessed": False,
        "calendar_classification": "EXECUTION_CALENDAR_DST_CLARIFICATION",
        "future_pre_quality_gate_master_dataset": {
            "ready": True,
            "artifact": "interaction-features.csv",
            "all_poc_interactions_before_quality_gate": True,
            "raw_score_components": list(PRE_QUALITY_RAW_COMPONENTS),
            "strategy_behavior_changed": False,
        },
        "totals": dict(totals),
        "sessions": sessions,
    }
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_output.with_suffix(audit_output.suffix + ".part")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(audit_output)
    if audit_html is not None:
        _write_preflight_html(payload, audit_html)
    return payload


def _write_preflight_html(payload: Mapping[str, Any], path: Path) -> None:
    """Write a self-contained, outcome-free preflight report."""
    totals = payload["totals"]
    rows = []
    for session in payload["sessions"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(session['session']))}</td>"
            f"<td>{int(session['records_processed']):,}</td>"
            f"<td>{session['maintenance_episodes']}</td>"
            f"<td>{session['temporary_non_executable_episodes']}</td>"
            f"<td>{len(session['unresolved_episodes'])}</td>"
            f"<td>{len(session['adapter_exceptions'])}</td>"
            f"<td>{html.escape(str(session['final_state']))}</td>"
            f"<td>{str(bool(session['effective_hard_flat_boundary_available'])).lower()}</td>"
            "</tr>"
        )
    component_items = "".join(
        f"<li><code>{html.escape(name)}</code></li>"
        for name in payload["future_pre_quality_gate_master_dataset"]["raw_score_components"]
    )
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L2 V3 Dec/Jan data preflight</title><style>
body{{font:15px/1.45 system-ui,sans-serif;margin:2rem;max-width:1200px;color:#172033}}h1,h2{{color:#10224a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{padding:14px;border:1px solid #ccd5e3;border-radius:9px;background:#f8fafc}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border:1px solid #d7deea;text-align:left}}th{{background:#eef3f9}}code{{font-family:ui-monospace,monospace}}
.pass{{color:#08783e;font-weight:700}}ul{{columns:2}}
</style></head><body>
<h1>Dec 2025 / Jan 2026 frozen V3 data preflight</h1>
<p class="pass">{html.escape(str(payload['status']))}</p>
<p>Classification: <code>EXECUTION_CALENDAR_DST_CLARIFICATION</code>. No strategy, setup, confirmation, trade, outcome, or PnL path was accessed.</p>
<div class="grid">
<div class="card"><strong>Sessions sufficient</strong><br>{payload['sufficient_session_count']} / 42</div>
<div class="card"><strong>ES records audited</strong><br>{int(totals.get('records', 0)):,}</div>
<div class="card"><strong>Maintenance episodes</strong><br>{totals.get('maintenance_episodes', 0)}</div>
<div class="card"><strong>Temporary episodes</strong><br>{totals.get('temporary_non_executable_episodes', 0)}</div>
<div class="card"><strong>Unresolved / exceptions</strong><br>{totals.get('unresolved_episodes', 0)} / {totals.get('adapter_exceptions', 0)}</div>
<div class="card"><strong>Hard-flat boundaries</strong><br>{totals.get('hard_flat_boundaries_available', 0)} / 42</div>
</div>
<h2>Per-session adapter audit</h2>
<table><thead><tr><th>Session</th><th>Records</th><th>Maintenance</th><th>Temporary</th><th>Unresolved</th><th>Exceptions</th><th>Final state</th><th>Hard-flat BBO</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>November 28 source decision</h2><p><code>{html.escape(str(payload['source_quality_nov28']['decision']))}</code>; semantic RTH 14:30-18:00 UTC, outcome-independent.</p>
<h2>Future pre-quality master dataset</h2><p><code>interaction-features.csv</code> can persist every completed POC interaction before the quality gate. Available raw fields:</p><ul>{component_items}</ul>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)


def _load_passing_audit(path: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecJanReplayError("missing or unreadable adapter audit report") from exc
    if (
        payload.get("status") != "ADAPTER_AUDIT_PASS"
        or payload.get("strategy_runner_invoked") is not False
        or payload.get("pnl_or_outcomes_accessed") is not False
        or payload.get("manifest_sha256") != preflight["base"]["manifest_sha256"]
        or payload.get("supplement_manifest_verification", {}).get("manifest_sha256") != preflight["supplement"]["manifest_sha256"]
        or payload.get("data_sufficiency_status") != "ALL_42_SESSIONS_DATA_SUFFICIENT"
        or payload.get("sufficient_session_count") != EXPECTED_SESSION_COUNT
        or payload.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or payload.get("totals", {}).get("unresolved_episodes") != 0
        or payload.get("totals", {}).get("adapter_exceptions") != 0
    ):
        raise DecJanReplayError("adapter audit did not pass the frozen replay gate")
    return payload


def _validated_poc(path: Path) -> StructuralLevel:
    level = native._profile_poc(path)
    if not isinstance(level, StructuralLevel) or level.name != "PRIOR_RTH_POC":
        raise DecJanReplayError("prior-RTH profile did not produce exactly one eligible POC")
    return level


def _execution_paths_for_day(verification: Mapping[str, Any], data_root: Path, day: str) -> tuple[Path, Path]:
    items = verification["session_inputs"][day]
    return tuple(data_root / str(items[purpose]["local_path"]) for purpose in EXPECTED_PURPOSES[:2])  # type: ignore[return-value]


def _run_session(
    day: str,
    preflight: Mapping[str, Any],
    data_root: Path,
    *,
    config: Any = V2_CONFIG,
    strategy_id: str = STRATEGY_ID,
    evidence_label: str = EVIDENCE_LABEL,
) -> historical.HistoricalL2Runner:
    verification = preflight["base"]
    es_path, mes_path = _execution_paths_for_day(verification, data_root, day)
    level = _validated_merged_poc(preflight, data_root, day)
    runner = historical.HistoricalL2Runner(
        date=day, evidence_label=evidence_label, levels=[level], config=config,
        strategy_id=strategy_id, require_native_mes_for_fallback=True,
    )
    item = verification["session_inputs"][day]["ES_MBP10"]
    sufficiency = next(row for row in preflight["sufficiency_matrix"] if row["session_date"] == day)
    cutoff_ns = _ns(str(sufficiency["effective_hard_flat_utc"]))
    start_ns = _ns(str(sufficiency["required_mes_mbp1_utc"]["start_utc"]))
    adapter = native.NativeMBP10Adapter()
    es_iter = iter(native._stream_native_mbp10_records(es_path))
    mes_iter = iter(historical._stream_mes_quotes(mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    initialized = closed = False
    records = 0
    while es is not None or mes is not None:
        es_timestamp = native._timestamp(es) if es is not None else None
        mes_timestamp = mes[0] if mes is not None else None
        timestamp = min(value for value in (es_timestamp, mes_timestamp) if value is not None)
        if timestamp >= cutoff_ns:
            if not initialized:
                raise DecJanReplayError("native MBP-10 initialization absent before RTH")
            adapter.assert_executable_at_boundary()
            reason = "HARD_CUTOFF_2245" if cutoff_ns == historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS) else "HARD_FLAT_SCHEDULED_CLOSE"
            runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns, exit_reason=reason)
            runner.finish(cutoff_ns)
            closed = True
            break
        if _in_maintenance(timestamp, date.fromisoformat(day)):
            if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
                mes = historical._next(mes_iter)
            else:
                assert es is not None
                adapter.feed(es, expected_non_executable=True)
                es = historical._next(es_iter)
            runner.es_quote = runner.mes_quote = None
            runner.es_quote_timestamp_ns = runner.mes_quote_timestamp_ns = None
            continue
        if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
            if getattr(runner, "_native_transient_started_ns", None) is None:
                runner.observe_mes_quote(*mes)
            mes = historical._next(mes_iter)
            continue
        assert es is not None and es_timestamp is not None
        public = adapter.feed(es)
        if es_timestamp < start_ns:
            initialized = adapter.first_valid_book_ns is not None and adapter.first_valid_book_ns < start_ns
            es = historical._next(es_iter)
            continue
        if not initialized:
            raise DecJanReplayError("native MBP-10 initialization absent before first RTH event")
        if public is not None:
            native._resume_temporary_non_executable_state(runner, es_timestamp)
            runner.observe_public(public)
            records += 1
        elif adapter.state == "TEMPORARILY_NON_EXECUTABLE":
            native._begin_temporary_non_executable_state(runner, es_timestamp)
        else:
            native._begin_non_executable_state(runner, es_timestamp)
        if records and records % 1_000_000 == 0:
            print(f"  V3 Dec/Jan native MBP-10 {day} records={records:,}", flush=True)
        es = historical._next(es_iter)
    if not closed:
        raise DecJanReplayError(f"source ended before frozen hard-flat completion: {day}")
    return runner


def _breakdown(trades: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trades:
        value = str(row["date"])[0:7] if key == "month" else str(row.get(key))
        groups[value].append(row)
    return {
        name: {
            "trades": len(rows),
            "wins": sum(float(row["net_pnl_usd"]) > 0 for row in rows),
            "losses": sum(float(row["net_pnl_usd"]) < 0 for row in rows),
            "total_r": sum(float(row["r_multiple"] or 0) for row in rows),
            "net_pnl_usd": sum(float(row["net_pnl_usd"]) for row in rows),
        }
        for name, rows in sorted(groups.items())
    }


def _metrics(runners: Sequence[historical.HistoricalL2Runner], *, requested: int, excluded: Sequence[str]) -> dict[str, Any]:
    interactions = [row for runner in runners for row in runner.interaction_ledger]
    setups = [row for runner in runners for row in runner.setup_ledger]
    trades = [row for runner in runners for row in runner.trade_ledger]
    pending = [setup for runner in runners for setup in runner.signals.pending.values()]
    performance = historical._performance(trades)
    hard_cutoffs = sum(str(row["exit_reason"]).startswith("HARD_") for row in trades)
    return {
        "target_sessions_requested": requested,
        "usable_sessions": len(runners),
        "source_quality_excluded_sessions": list(excluded),
        "completed_interactions": len(interactions),
        "accepted_poc_setups": sum(bool(row["accepted"]) for row in setups),
        "rejected_interactions": sum(not bool(row["accepted"]) for row in setups),
        "confirmations_passed": sum(setup.confirmation_timestamp_ns is not None for setup in pending),
        "confirmation_expiries": sum(setup.terminal_reason == "CONFIRMATION_WINDOW_EXPIRED" for setup in pending),
        "active_position_blocks": sum(setup.terminal_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for setup in pending),
        "unresolved": sum(setup.terminal_reason is None for setup in pending),
        **performance,
        "hard_cutoff_exits": hard_cutoffs,
        "breakdowns": {
            "day": _breakdown(trades, "date"),
            "month": _breakdown(trades, "month"),
            "direction": _breakdown(trades, "direction"),
            "instrument": _breakdown(trades, "instrument"),
        },
    }


def _write_results(
    output_root: Path, runners: list[historical.HistoricalL2Runner], verification: Mapping[str, Any],
    audit: Mapping[str, Any], excluded: Sequence[str],
) -> dict[str, Any]:
    base = historical.write_future_artifacts(output_root, runners, contract=v3_contract())
    metrics = _metrics(runners, requested=EXPECTED_SESSION_COUNT, excluded=excluded)
    result = {
        **base,
        "strategy_id": STRATEGY_ID,
        "evidence_classification": EVIDENCE_LABEL,
        "strict_chronological_oos": False,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "quality_threshold": V2_CONFIG.min_quality_score,
        "manifest_verification": {key: value for key, value in verification.items() if key != "session_inputs"},
        "adapter_audit": {
            "status": audit["status"], "manifest_sha256": audit["manifest_sha256"], "totals": audit["totals"],
        },
        "source_quality_nov28": audit["source_quality_nov28"],
        "metrics": metrics,
        "future_pre_quality_gate_master_dataset": {
            "heavy_dbn_rescan_required": False,
            "reason": "interaction-features.csv contains every completed POC interaction and raw score components before quality acceptance",
        },
        "outcome_parameter_selection": False,
    }
    (output_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "diagnostic-report.md").write_text(
        f"# {STRATEGY_ID} — Dec 2025 / Jan 2026 retrospective robustness replay\n\n"
        f"Evidence classification: `{EVIDENCE_LABEL}`. This is not fresh chronological OOS or validation.\n\n"
        f"Requested/usable sessions: {EXPECTED_SESSION_COUNT}/{len(runners)}. Source-quality exclusions: {list(excluded)}.\n\n"
        "Only `PRIOR_RTH_POC` is eligible. No parameter, threshold, score weight, confirmation, execution, or sizing rule was changed.\n",
        encoding="utf-8",
    )
    return result


def run(*, data_root: Path, output_root: Path, audit_report: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("Dec/Jan V3 output directory already exists")
    preflight = verify_data_preflight(data_root, verify_hashes=True)
    if preflight["status"] != "ALL_42_SESSIONS_DATA_SUFFICIENT":
        raise DecJanReplayError(f"full replay blocked by insufficient sessions: {preflight['excluded_sessions']}")
    verification = preflight["base"]
    audit = _load_passing_audit(audit_report, preflight)
    excluded = audit.get("source_quality_excluded_sessions", [])
    runners: list[historical.HistoricalL2Runner] = []
    usable = [day for day in verification["target_sessions"] if day not in excluded]
    for index, day in enumerate(usable, start=1):
        print(f"=== V3 DEC/JAN RETRO {index:02d}/{len(usable):02d} {day} ===", flush=True)
        runners.append(_run_session(day, preflight, data_root))
    if len(runners) != EXPECTED_SESSION_COUNT - len(excluded):
        raise DecJanReplayError("not all usable sessions completed")
    return _write_results(output_root, runners, verification, audit, excluded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--adapter-audit", action="store_true")
    parser.add_argument("--audit-workers", type=int, default=1)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--audit-html", type=Path, default=AUDIT_HTML)
    parser.add_argument("--audit-report", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--skip-hash-verification", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.preflight_only and args.adapter_audit:
            raise DecJanReplayError("--preflight-only and --adapter-audit are mutually exclusive")
        if args.preflight_only:
            preflight = verify_data_preflight(args.data_root, verify_hashes=not args.skip_hash_verification)
            result = {
                "status": preflight["status"],
                "base": {key: value for key, value in preflight["base"].items() if key != "session_inputs"},
                "supplement": {key: value for key, value in preflight["supplement"].items() if key != "by_target_session"},
                "sufficiency_matrix": preflight["sufficiency_matrix"],
                "sufficient_session_count": preflight["sufficient_session_count"],
                "excluded_sessions": preflight["excluded_sessions"],
                "source_quality_nov28": audit_degraded_nov28_source(args.data_root, preflight),
                "preflight_only": True,
                "strategy_or_outcomes_accessed": False,
            }
        elif args.adapter_audit:
            result = audit_native_mbp10(
                args.data_root, audit_output=args.audit_output, audit_html=args.audit_html,
                verify_hashes=not args.skip_hash_verification,
                workers=args.audit_workers,
            )
        else:
            result = run(data_root=args.data_root, output_root=args.output_root, audit_report=args.audit_report)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (DecJanReplayError, historical.HistoricalReplayError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
