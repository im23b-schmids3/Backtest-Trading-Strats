"""Strict Dec-2025/Jan-2026 replay of the ten sealed MAC family configs.

This is a deterministic evaluation driver. It reads the frozen config JSON,
validates local manifest-bound ES MBP-10 partitions, reuses the Candidate Tape
V2 builder/evaluator, and performs no search or provider access.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_class_b_optuna as class_b_v1
from . import mac_2025_class_b_v2_calibration as class_b_v2
from . import mac_2025_es_only_train_baseline as baseline
from . import dec2025_feb2026_quote as quote_contract
from .model import L2ClassBConfig, L2Config


EXPECTED_CONFIG_SHA256 = "99c4af7f7b03cf6a255781524f7a6c2a32bd992d9785dbfb294db6a07fbc7448"
CONFIG_PATH = Path("frozen_configs/ten-family-frozen-configs.raw.json")
EXPECTED_FAMILIES = (
    "EUROPE|EUROPE|CURRENT|HIGH",
    "EUROPE|EUROPE|PRIOR|VAH",
    "EUROPE|EUROPE|PRIOR|HIGH",
    "NY|NY|PRIOR|POC",
    "ASIA|ASIA|CURRENT|HIGH",
    "EUROPE|ASIA|CURRENT|VAH",
    "EUROPE|ASIA|CURRENT|LOW",
    "EUROPE|RTH|PRIOR|VAL",
    "NY|EUROPE|CURRENT|HIGH",
    "NY|EUROPE|CURRENT|VAH",
)
BASE_ROOT = Path("data/cme_orderflow_absorption_l2_v3/dec2025_jan2026")
EXT_ROOT = Path("data/cme_orderflow_absorption_l2_v1/historical_completion/dec_jan_asia_europe")
MASTER_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_CAUSAL_MASTER_DEC2025_JAN2026")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_TEN_FAMILY_DEC2025_JAN2026_ROBUSTNESS_WINDOWS_FIXED")
PRIOR_EUROPE_FAMILIES = {
    "EUROPE|EUROPE|PRIOR|VAH", "EUROPE|EUROPE|PRIOR|HIGH",
}
STRICT_ROUTE_EQUIVALENCE_DATES = ("2025-12-02", "2026-01-02")
NON_STANDARD_SESSION_REASON = "SCHEDULED_EARLY_CLOSE_NOT_NORMAL_FULL_SESSION"
SESSION_BOUNDARY_FAILURE_DATES = (
    "2025-12-01", "2025-12-08", "2025-12-11", "2025-12-22",
    "2026-01-07", "2026-01-20", "2026-01-27", "2026-01-29",
)


class RobustnessRunError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessRunError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(result, dict):
        raise RobustnessRunError(f"expected JSON object: {path}")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key, value in row.items()
                      if not isinstance(value, (dict, list, tuple))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in (columns or ["empty"])})


def _config_payload(path: Path) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise RobustnessRunError(f"frozen config not found: {path}")
    digest = _sha256(path)
    if digest != EXPECTED_CONFIG_SHA256:
        raise RobustnessRunError(f"frozen config hash mismatch: {digest}")
    payload = _json(path)
    expected_metadata = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY",
        "tape_version": candidate_tape.TAPE_VERSION,
        "entry_delay_ms": 2.0, "validation_accessed": False,
        "final_oos_accessed": False, "data_downloaded": False,
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise RobustnessRunError(f"frozen config metadata mismatch: {key}")
    rows = payload.get("families")
    if not isinstance(rows, list) or tuple(row.get("family") for row in rows) != EXPECTED_FAMILIES:
        raise RobustnessRunError("frozen config family sequence mismatch")
    class_a_fields = {item.name for item in fields(L2Config)}
    class_b_fields = {item.name for item in fields(L2ClassBConfig)}
    configs: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        a = row.get("class_a_config")
        b = row.get("class_b_config")
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise RobustnessRunError(f"missing frozen Class-A/Class-B config: {family}")
        if class_a_fields - set(a):
            raise RobustnessRunError(f"Class-A fields missing for {family}: {sorted(class_a_fields-set(a))}")
        if class_b_fields != set(b):
            raise RobustnessRunError(f"Class-B field mismatch for {family}")
        if row.get("entry_delay_ms") != 2.0 or row.get("tape_version") != candidate_tape.TAPE_VERSION:
            raise RobustnessRunError(f"entry delay/tape metadata mismatch for {family}")
        try:
            class_a = L2Config(**{key: value for key, value in a.items() if key in class_a_fields})
            class_b = L2ClassBConfig(**b)
        except (TypeError, ValueError) as exc:
            raise RobustnessRunError(f"invalid frozen config for {family}: {exc}") from exc
        configs[family] = {
            "class_a_config": {item.name: getattr(class_a, item.name) for item in fields(L2Config)},
            "class_b_config": {item.name: getattr(class_b, item.name) for item in fields(L2ClassBConfig)},
            "entry_delay_ms": 2.0,
            "train_metrics": row.get("train_metrics"),
            "robustness_flags": row.get("robustness_flags", []),
        }
    return payload, digest, configs


def _verified_manifest_file(root: Path, item: Mapping[str, Any]) -> Path:
    relative = str(item.get("local_path", ""))
    path = root / relative
    if not path.is_file():
        raise RobustnessRunError(f"manifest source missing: {path}")
    if path.stat().st_size != int(item.get("bytes", -1)):
        raise RobustnessRunError(f"manifest source byte count mismatch: {path}")
    actual = _sha256(path)
    if actual != item.get("sha256"):
        raise RobustnessRunError(f"manifest source hash mismatch: {path}")
    return path


def _source_plan(repository_root: Path) -> tuple[list[str], dict[str, tuple[Path, Path]], dict[str, Any]]:
    base_path = repository_root / BASE_ROOT / "acquisition-manifest.json"
    ext_path = repository_root / EXT_ROOT / "acquisition-manifest.json"
    base = _json(base_path)
    extension = _json(ext_path)
    if base.get("status") != "ACQUISITION_COMPLETE_VERIFIED" or int(base.get("target_session_count", -1)) != 42:
        raise RobustnessRunError("Dec/Jan base acquisition manifest is incomplete")
    if extension.get("status") != "ACQUISITION_COMPLETE_VERIFIED" or extension.get("family") != "dec_jan_asia_europe":
        raise RobustnessRunError("Dec/Jan Asia/Europe extension manifest is incomplete")
    if len(base.get("files", {})) != 126 or len(extension.get("files", {})) != 43:
        raise RobustnessRunError("acquisition manifest file counts mismatch")
    target_dates = [str(day) for day in base.get("prior_rth_by_target_session", {}).keys()]
    if len(target_dates) != 42:
        target_dates = sorted({str(item.get("target_session")) for item in base["files"].values()
                               if item.get("purpose") == "ES_MBP10"})
    target_dates = sorted(day for day in target_dates if day and day != "None")
    if len(target_dates) != 42 or len(set(target_dates)) != 42:
        raise RobustnessRunError("expected 42 unique Dec/Jan target dates")
    calendar_sessions = quote_contract.build_sessions(
        date.fromisoformat(target_dates[0]), date.fromisoformat(target_dates[-1]),
    )
    calendar_by_day = {item.session_date.isoformat(): item for item in calendar_sessions}
    if list(calendar_by_day) != target_dates:
        raise RobustnessRunError("quote calendar and acquisition target dates differ")

    base_items = list(base["files"].values())
    ext_items = list(extension["files"].values())
    sources: dict[str, tuple[Path, Path]] = {}
    session_contract_by_date: dict[str, dict[str, Any]] = {}
    input_records: list[dict[str, Any]] = []
    from databento import DBNStore

    for day in target_dates:
        base_matches = [item for item in base_items if item.get("purpose") == "ES_MBP10" and item.get("target_session") == day]
        ext_matches = [item for item in ext_items if item.get("component") == "DEC_JAN_ES_MBP10" and item.get("session_date") == day]
        if len(base_matches) != 1 or len(ext_matches) != 1:
            raise RobustnessRunError(f"expected one ES partition of each type for {day}")
        b, e = base_matches[0], ext_matches[0]
        expected_symbol = "ESZ5" if day <= "2025-12-16" else "ESH6"
        calendar_session = calendar_by_day[day]
        day_date = date.fromisoformat(day)
        expected_end = quote_contract._iso(day_date, calendar_session.effective_end, plus_one_second=True)
        if (b.get("schema"), b.get("raw_symbol"), b.get("start_utc"), b.get("end_utc"), b.get("status")) != (
                "mbp-10", expected_symbol, f"{day}T13:00:00Z", expected_end, "DOWNLOADED_VERIFIED"):
            raise RobustnessRunError(f"NY ES source manifest identity/range mismatch: {day}")
        if (e.get("schema"), e.get("symbol"), e.get("start"), e.get("end"), e.get("status")) != (
                "mbp-10", expected_symbol, f"{day}T00:00:00Z", f"{day}T13:00:00Z", "DOWNLOADED_VERIFIED"):
            raise RobustnessRunError(f"pre-NY ES source manifest identity/range mismatch: {day}")
        base_file = _verified_manifest_file(repository_root / BASE_ROOT, b)
        ext_file = _verified_manifest_file(repository_root / EXT_ROOT, e)
        for file_path, item, schema_key, symbol_key in (
            (ext_file, e, "schema", "symbol"), (base_file, b, "schema", "raw_symbol"),
        ):
            store = DBNStore.from_file(file_path)
            meta = store.metadata
            if meta.dataset != "GLBX.MDP3" or meta.schema != item[schema_key] or item[symbol_key] not in meta.symbols:
                raise RobustnessRunError(f"DBN metadata mismatch: {file_path}")
        sources[day] = (ext_file, base_file)
        session_contract_by_date[day] = {
            "source_model": "NATIVE_MBP10",
            "raw_symbol": expected_symbol,
            "requested_source_start": str(e["start"]),
            "requested_source_end": str(b["end_utc"]),
            "scheduled_session_end": quote_contract._iso(day_date, calendar_session.effective_end),
            "scheduled_early_close": bool(calendar_session.shortened),
        }
        input_records.extend({"path": str(item_path.relative_to(repository_root)), "bytes": int(item["bytes"]),
                              "sha256": str(item["sha256"]), "symbol": expected_symbol,
                              "schema": "mbp-10", "date": day, "partition": label}
                             for item_path, item, label in ((ext_file, e, "00:00-13:00"),
                                                             (base_file, b, "13:00-effective-close")))

    prior_matches = [item for item in base_items if item.get("purpose") == "PRIOR_RTH_TRADES"
                     and item.get("prior_rth_date") == "2025-11-28" and item.get("target_session") == "2025-12-01"]
    if len(prior_matches) != 1:
        raise RobustnessRunError("Nov 28 prior-RTH trades dependency is unavailable")
    prior_item = prior_matches[0]
    prior_symbol = "ESZ5"
    if (prior_item.get("schema"), prior_item.get("raw_symbol"), prior_item.get("start_utc"), prior_item.get("end_utc")) != (
            "trades", prior_symbol, "2025-11-28T13:30:00Z", "2025-11-28T18:15:00Z"):
        raise RobustnessRunError("Nov 28 prior-RTH trades dependency identity/range mismatch")
    prior_file = _verified_manifest_file(repository_root / BASE_ROOT, prior_item)
    prior_meta = DBNStore.from_file(prior_file).metadata
    if prior_meta.dataset != "GLBX.MDP3" or prior_meta.schema != "trades" or prior_symbol not in prior_meta.symbols:
        raise RobustnessRunError("Nov 28 profile dependency DBN metadata mismatch")
    input_records.append({"path": str(prior_file.relative_to(repository_root)), "bytes": int(prior_item["bytes"]),
                          "sha256": str(prior_item["sha256"]), "symbol": prior_symbol,
                          "schema": "trades", "date": "2025-11-28", "partition": "prior-RTH-profile-only"})
    return target_dates, sources, {
        "base_manifest_path": str(base_path.relative_to(repository_root)),
        "base_manifest_sha256": _sha256(base_path),
        "extension_manifest_path": str(ext_path.relative_to(repository_root)),
        "extension_manifest_sha256": _sha256(ext_path),
        "base_status": base["status"], "extension_status": extension["status"],
        "input_files": input_records, "nov28_prior_rth_trades_path": prior_file,
        "nov28_prior_rth_trades_manifest_item": prior_item,
        "session_contract_by_date": session_contract_by_date,
    }


def _scan_source_last_timestamp_ns(source_paths: Sequence[Path], *, expected_end: str) -> int:
    """Read only receive timestamps to establish the final source observation."""
    from databento import DBNStore

    expected_end_ns = baseline._ns(expected_end)
    previous: int | None = None
    records = 0
    for source_path in source_paths:
        store = DBNStore.from_file(source_path)
        for batch in store.to_ndarray(count=1_000_000):
            names = set(batch.dtype.names or ())
            if "ts_recv" not in names:
                raise RobustnessRunError(f"source lacks ts_recv timestamps: {source_path}")
            timestamps = batch["ts_recv"]
            if len(timestamps) == 0:
                continue
            if previous is not None and int(timestamps[0]) < previous:
                raise RobustnessRunError(f"source timestamp regressed at partition boundary: {source_path}")
            if len(timestamps) > 1 and bool(np.any(timestamps[1:] < timestamps[:-1])):
                raise RobustnessRunError(f"source timestamps regress within partition: {source_path}")
            previous = int(timestamps[-1])
            records += len(timestamps)
            if previous >= expected_end_ns:
                raise RobustnessRunError(f"source timestamp reaches/exceeds requested exclusive end: {source_path}")
    if not records or previous is None:
        raise RobustnessRunError("empty native MBP-10 source while auditing session coverage")
    return previous


def _classify_session_coverage(day: str, *, contract: Mapping[str, Any], source_last_timestamp_ns: int,
                               normal_required_final_session_end: str) -> dict[str, Any]:
    """Classify calendar status independently from frozen strategy windows."""
    required_ns = baseline._ns(normal_required_final_session_end)
    source_end_ns = baseline._ns(str(contract["requested_source_end"]))
    source_last = baseline._iso(source_last_timestamp_ns)
    if source_last_timestamp_ns >= source_end_ns:
        raise RobustnessRunError(f"source last timestamp violates requested end for {day}")
    if bool(contract.get("scheduled_early_close")):
        session_type = "SCHEDULED_EARLY_CLOSE"
        reason = NON_STANDARD_SESSION_REASON
    elif source_last_timestamp_ns < required_ns:
        session_type = "INCOMPLETE_NORMAL_SESSION"
        reason = "NORMAL_SESSION_SOURCE_ENDS_BEFORE_FROZEN_NY_WINDOW"
    else:
        session_type = "NORMAL_FULL_SESSION"
        reason = None
    return {
        "date": day,
        "source_model": contract["source_model"],
        "raw_symbol": contract["raw_symbol"],
        "normal_required_final_session_end": normal_required_final_session_end,
        "requested_source_end": contract["requested_source_end"],
        "source_last_timestamp": source_last,
        "scheduled_session_end": contract["scheduled_session_end"],
        "scheduled_early_close": bool(contract["scheduled_early_close"]),
        "session_type": session_type,
        "eligibility": "EXCLUDED" if session_type in {"SCHEDULED_EARLY_CLOSE", "OTHER_NON_STANDARD_SESSION"}
        else "ELIGIBLE" if session_type == "NORMAL_FULL_SESSION" else "FAIL_CLOSED",
        "exclusion_reason": reason if session_type in {"SCHEDULED_EARLY_CLOSE", "OTHER_NON_STANDARD_SESSION"} else None,
        "failure_reason": reason if session_type == "INCOMPLETE_NORMAL_SESSION" else None,
    }


def _session_eligibility_audit(dates: Sequence[str], sources: Mapping[str, Sequence[Path]],
                               source_manifest: Mapping[str, Any],
                               reusable_tapes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    contracts = source_manifest.get("session_contract_by_date")
    if not isinstance(contracts, dict) or set(contracts) != set(dates):
        raise RobustnessRunError("per-date frozen session contracts are missing or unexpected")
    rows: list[dict[str, Any]] = []
    for day in dates:
        contract = contracts[day]
        ny_end_ns = baseline._session_windows(day)["NY"][1]
        required_end = baseline._iso(ny_end_ns)
        if day in reusable_tapes:
            last_ns = int(reusable_tapes[day]["source_last_timestamp_ns"])
        else:
            last_ns = _scan_source_last_timestamp_ns(sources[day], expected_end=str(contract["requested_source_end"]))
        row = _classify_session_coverage(day, contract=contract,
                                         source_last_timestamp_ns=last_ns,
                                         normal_required_final_session_end=required_end)
        row["source_paths"] = [str(path) for path in sources[day]]
        rows.append(row)
    standard = [row["date"] for row in rows if row["session_type"] == "NORMAL_FULL_SESSION"]
    excluded = [row["date"] for row in rows if row["session_type"] in {
        "SCHEDULED_EARLY_CLOSE", "OTHER_NON_STANDARD_SESSION",
    }]
    incomplete = [row["date"] for row in rows if row["session_type"] == "INCOMPLETE_NORMAL_SESSION"]
    return {
        "status": "PASS" if not incomplete else "FAIL",
        "intended_dates": list(dates),
        "normal_full_session_dates": standard,
        "non_standard_session_excluded_dates": excluded,
        "incomplete_normal_session_dates": incomplete,
        "session_count": len(rows),
        "excluded_session_count": len(excluded),
        "normal_full_session_count": len(standard),
        "per_date": rows,
        "classification_evidence": {
            "scheduled_calendar_source": "src/research_pipeline/cme_orderflow_absorption_l2_v1/dec2025_feb2026_quote.py:SHORTENED_SESSION_ENDS",
            "source_request_evidence": "hash-verified Dec/Jan acquisition manifests; shortened flag and requested UTC end must match the calendar",
            "strategy_window_source": "mac_2025_es_only_train_baseline._session_windows (unchanged)",
            "session_audit_document": "docs/research_pipeline/cme_orderflow_absorption_v1/l2-v3-dec2025-jan2026-dst-calendar-audit.md",
        },
    }


def _prior_profiles_by_date(dates: Sequence[str], profile_by_date: Mapping[str, Any],
                            initial_ny_profile: Any) -> dict[str, dict[str, Any]]:
    """Keep every intended session, including unscored dates, in causal context."""
    if not dates or any(day not in profile_by_date for day in dates[1:]):
        raise RobustnessRunError("prior-session profile context is incomplete")
    result: dict[str, dict[str, Any]] = {dates[0]: {"NY": initial_ny_profile}}
    for index, day in enumerate(dates[1:], 1):
        result[day] = profile_by_date[dates[index - 1]]
    return result


def _profiles_by_date(dates: Sequence[str], sources: Mapping[str, tuple[Path, Path]],
                      source_manifest: Mapping[str, Any], repository_root: Path
                      ) -> tuple[dict[str, dict[str, baseline.Profile]], dict[str, str]]:
    """Build complete day profiles from the verified native MBP-10 inputs.

    Older causal-master event tapes for this block were built before the
    pre-NY partitions were acquired and contain only the NY portion for Dec 1.
    Reusing them would leave Asia/Europe profiles empty, so this dependency
    pass uses the same raw-trade profile builder as the frozen baseline over
    the two declared, hash-verified adjacent source partitions.
    """
    result: dict[str, dict[str, baseline.Profile]] = {}
    hashes: dict[str, str] = {}
    for index, day in enumerate(dates, 1):
        profile_started = time.perf_counter()
        day_sources = sources[day]
        source_rows = [row for row in source_manifest["input_files"]
                       if row["date"] == day and row["schema"] == "mbp-10"]
        if len(source_rows) != 2:
            raise RobustnessRunError(f"profile source partition count mismatch: {day}")
        digest = _canonical_sha([{"path": row["path"], "sha256": row["sha256"]}
                                 for row in source_rows])
        result[day] = baseline._profile_only_day(day, day_sources[0], source_paths=day_sources)
        hashes[day] = digest
        print(f"PROFILE_SESSION={index}/{len(dates)} DATE={day} "
              f"ELAPSED={time.perf_counter() - profile_started:.1f}s", flush=True)
    return result, hashes


def _build_tape_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Build one date tape in an isolated worker; dates share no mutable state."""
    day = str(job["day"])
    source_paths = tuple(Path(item) for item in job["source_paths"])
    tape_path = Path(job["tape_path"])
    tape, raw_result = candidate_tape.build_candidate_tape(
        day, source_paths[0], job["prior_profiles"], job["current_profiles"],
        output_path=tape_path, source_sha256=str(job["source_sha256"]),
        semantic_sha256=str(job["semantic_sha256"]), config=L2Config(),
        source_paths=source_paths,
    )
    equivalence: dict[str, Any] | None = None
    if bool(job.get("check_strict_route_equivalence")):
        reference = baseline._route_day(
            day, source_paths[0], dict(job["prior_profiles"]), L2Config(),
            dict(job["current_profiles"]), capture_events=[], source_paths=source_paths,
        )
        equivalence = _compare_tape_to_reference(tape, reference, raw_result, day)
    return {
        "date": day, "tape_path": str(tape_path),
        "candidate_count": len(tape.candidates), "event_count": len(tape.events),
        "available_families": sorted(tape.metadata.get("available_families", [])),
        "equivalence": equivalence,
        "source_last_timestamp_ns": tape.metadata.get("source_last_timestamp_ns"),
        "last_strategy_timestamp_ns": tape.metadata.get("last_strategy_timestamp_ns"),
        "book_state_at_last_strategy_record": tape.metadata.get("book_state_at_last_strategy_record"),
        "post_session_terminal_state_accepted": tape.metadata.get("post_session_terminal_state_accepted"),
        "raw_replay_seconds": float(raw_result["timings"]["total_seconds"]),
    }


def _compare_tape_to_reference(tape: candidate_tape.CandidateTape,
                               reference: Mapping[str, Any], candidate_result: Mapping[str, Any],
                               day: str) -> dict[str, Any]:
    expected_candidates, _ = candidate_tape._candidate_payload(reference["interactions"])
    expected_events = _compact_reference_events(reference["market_events"])
    candidates_equal = tuple(tape.candidates) == tuple(expected_candidates)
    events_equal = _event_arrays_equal(tape.events, expected_events)
    setups_equal = reference["setups"] == candidate_result["setups"]
    trades_equal = reference["trades"] == candidate_result["trades"]
    profiles_equal = {
        name: reference["profiles"][name].volume_by_tick == candidate_result["profiles"][name].volume_by_tick
        for name in baseline.SESSION_ORDER
    }
    if not (candidates_equal and events_equal and setups_equal and trades_equal and all(profiles_equal.values())):
        details = {
            "candidates_equal": candidates_equal,
            "events_equal": events_equal,
            "setups_equal": setups_equal,
            "trades_equal": trades_equal,
            "profiles_equal": profiles_equal,
        }
        raise RobustnessRunError(
            f"strict route/tape equivalence failed for representative date {day}: "
            f"{json.dumps(details, sort_keys=True)}"
        )
    return {
        "status": "PASS", "candidate_count": len(tape.candidates), "event_count": len(tape.events),
        "setup_count": len(reference["setups"]), "trade_count": len(reference["trades"]),
        "candidates_equal": candidates_equal, "events_equal": events_equal,
        "setups_equal": setups_equal, "trades_equal": trades_equal,
        "profiles_equal": profiles_equal,
    }


def _compact_reference_events(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Apply the candidate tape's lossless sparse-event contract to reference rows."""
    spool = candidate_tape.EventSpool()
    try:
        for row in rows:
            spool.append(row)
        return spool.to_array()
    finally:
        spool.close()


def _event_arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare structured events exactly while treating matching NaNs as equal."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    names = left.dtype.names
    if names is None or names != right.dtype.names:
        return False
    for name in names:
        lhs, rhs = left[name], right[name]
        if lhs.dtype.kind == "f":
            if not np.array_equal(lhs, rhs, equal_nan=True):
                return False
        elif not np.array_equal(lhs, rhs):
            return False
    return True


def _profile_contract_sha(dates: Sequence[str], source_manifest: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Recompute the deterministic profile contract identity without replaying tapes."""
    rows = source_manifest["input_files"]
    hashes: dict[str, str] = {}
    for day in dates:
        source_rows = [row for row in rows if row.get("date") == day and row.get("schema") == "mbp-10"]
        if len(source_rows) != 2:
            raise RobustnessRunError(f"profile source partition count mismatch: {day}")
        hashes[day] = _canonical_sha([{"path": row["path"], "sha256": row["sha256"]}
                                      for row in source_rows])
    contract_sha = _canonical_sha({"profile_semantic_sha256": baseline._semantic_sha256(),
                                   "verified_source_sha256_by_date": hashes})
    return contract_sha, hashes


def _validate_existing_tape(repository_root: Path, job: Mapping[str, Any],
                            recorded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate one completed tape and its sidecar before allowing resume reuse."""
    day = str(job["day"])
    path = Path(job["tape_path"])
    sidecar = candidate_tape._tape_manifest_path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RobustnessRunError(f"candidate tape missing or empty: {day}")
    if not sidecar.is_file() or sidecar.stat().st_size <= 0:
        raise RobustnessRunError(f"candidate tape sidecar missing or empty: {day}")
    actual_sha = _sha256(path)
    if recorded is not None and recorded.get("tape_sha256") != actual_sha:
        raise RobustnessRunError(f"recorded tape SHA-256 mismatch: {day}")
    try:
        tape = candidate_tape.load_tape(
            path, source_sha256=str(job["source_sha256"]),
            semantic_sha256=str(job["semantic_sha256"]),
        )
        sidecar_metadata = _json(sidecar)
    except Exception as exc:
        raise RobustnessRunError(f"candidate tape load/sidecar validation failed for {day}: {exc}") from exc
    metadata = tape.metadata
    if sidecar_metadata != metadata:
        raise RobustnessRunError(f"candidate tape sidecar/embedded metadata mismatch: {day}")
    if (metadata.get("date") != day
            or metadata.get("source_sha256") != job["source_sha256"]
            or metadata.get("semantic_sha256") != job["semantic_sha256"]
            or tuple(metadata.get("source_paths", ())) != tuple(job["source_paths"])
            or int(metadata.get("candidate_count", -1)) != len(tape.candidates)
            or int(metadata.get("event_count", -1)) != len(tape.events)
            or "candidate_count" not in metadata
            or "event_count" not in metadata
            or metadata.get("bbo_path_complete") is not True
            or metadata.get("completed_strategy_sessions") != list(baseline.SESSION_ORDER)
            or metadata.get("book_state_at_last_strategy_record") != "EXECUTABLE"
            or int(metadata.get("source_last_timestamp_ns", -1)) < int(metadata.get("final_strategy_window_end_ns", 0))):
        raise RobustnessRunError(f"candidate tape metadata contract mismatch: {day}")
    if day in SESSION_BOUNDARY_FAILURE_DATES and metadata.get("post_session_terminal_state_accepted") is not True:
        raise RobustnessRunError(f"known post-session terminal-state case lacks acceptance evidence: {day}")
    expected_relative = str(path.relative_to(repository_root))
    if recorded is not None:
        if (recorded.get("date") != day or recorded.get("tape_path") != expected_relative
                or int(recorded.get("candidate_count", -1)) != len(tape.candidates)
                or int(recorded.get("event_count", -1)) != len(tape.events)
                or recorded.get("source_sha256") != job["source_sha256"]):
            raise RobustnessRunError(f"progress record does not match tape: {day}")
        if recorded.get("strict_route_equivalence") is not None:
            equivalence = recorded["strict_route_equivalence"]
        else:
            equivalence = None
    else:
        equivalence = None
    available = sorted(metadata.get("available_families", []))
    allowed_missing = PRIOR_EUROPE_FAMILIES if day == "2025-12-01" else set()
    unexpected_missing = set(EXPECTED_FAMILIES) - set(available) - allowed_missing
    if unexpected_missing:
        raise RobustnessRunError(f"frozen family unavailable on {day}: {sorted(unexpected_missing)}")
    return {
        "date": day,
        "source_files": [str(Path(item).relative_to(repository_root)) for item in job["source_paths"]],
        "source_sha256": str(job["source_sha256"]), "tape_path": expected_relative,
        "tape_sha256": actual_sha, "candidate_count": len(tape.candidates),
        "event_count": len(tape.events), "available_families": available,
        "raw_replay_seconds": float(metadata.get("raw_replay_seconds", 0.0)),
        "strict_route_equivalence": equivalence,
        "source_last_timestamp_ns": metadata.get("source_last_timestamp_ns"),
        "last_strategy_timestamp_ns": metadata.get("last_strategy_timestamp_ns"),
        "book_state_at_last_strategy_record": metadata.get("book_state_at_last_strategy_record"),
        "post_session_terminal_state_accepted": metadata.get("post_session_terminal_state_accepted"),
    }


def _resume_tape_inventory(repository_root: Path, output_root: Path, dates: Sequence[str],
                           jobs_by_day: Mapping[str, Mapping[str, Any]],
                           progress_identity: Mapping[str, Any], *,
                           intended_session_count: int | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return verified reusable tapes and invalid/missing dates without modifying files."""
    progress_path = output_root / "progress.json"
    if not progress_path.is_file():
        raise RobustnessRunError("resume requested but progress.json is missing")
    progress = _json(progress_path)
    if progress.get("status") not in {"BUILDING_CANDIDATE_TAPES", "FAILED", "RESUME_AUDITED"}:
        raise RobustnessRunError(f"run cannot resume from status: {progress.get('status')}")
    for key, expected in progress_identity.items():
        if progress.get(key) != expected:
            raise RobustnessRunError(f"resume progress identity mismatch: {key}")
    expected_intended_count = len(dates) if intended_session_count is None else intended_session_count
    if int(progress.get("total_sessions", -1)) != expected_intended_count:
        raise RobustnessRunError("resume total session count mismatch")
    recorded_rows = progress.get("tapes")
    if not isinstance(recorded_rows, list):
        raise RobustnessRunError("resume progress tape list is malformed")
    recorded_by_day: dict[str, Mapping[str, Any]] = {}
    for row in recorded_rows:
        day = str(row.get("date", "")) if isinstance(row, dict) else ""
        if day not in dates or day in recorded_by_day:
            raise RobustnessRunError(f"duplicate or unexpected progress date: {day}")
        recorded_by_day[day] = row
    ordered_recorded = [day for day in dates if day in recorded_by_day]
    if (int(progress.get("completed_sessions", -1)) != len(recorded_by_day)
            or progress.get("last_completed_date") != (ordered_recorded[-1] if ordered_recorded else None)):
        raise RobustnessRunError("resume progress counters do not reconcile to completed-date records")

    tape_dir = output_root / "tapes"
    expected_names: set[str] = set()
    artifacts_by_day: dict[str, set[str]] = {}
    for day in dates:
        path = Path(jobs_by_day[day]["tape_path"])
        names = {path.name, candidate_tape._tape_manifest_path(path).name}
        expected_names.update(names)
        artifacts_by_day[day] = names
    unexpected = {item.name for item in tape_dir.iterdir() if item.is_file()} - expected_names
    if unexpected:
        raise RobustnessRunError(f"unrecognized files in resume tape directory: {sorted(unexpected)}")

    reusable: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    for day in dates:
        path = Path(jobs_by_day[day]["tape_path"])
        sidecar = candidate_tape._tape_manifest_path(path)
        has_any = path.exists() or sidecar.exists()
        if not has_any:
            if day in recorded_by_day:
                invalid.append(day)
            continue
        try:
            reusable[day] = _validate_existing_tape(
                repository_root, jobs_by_day[day], recorded_by_day.get(day),
            )
        except Exception:
            # Existing broken artifacts are retained until the replacement is
            # fully written atomically; only this date is scheduled to rebuild.
            invalid.append(day)
    return reusable, invalid


def _nov28_rth_profile(repository_root: Path, source: Path, item: Mapping[str, Any]) -> baseline.Profile:
    from databento import DBNStore
    day = "2025-11-28"
    start, end = baseline._session_windows(day)["NY"]
    profile = baseline.Profile.create(day, "NY", start, end)
    store = DBNStore.from_file(source)
    iterator = store.to_ndarray(count=500_000)
    for batch in iterator:
        names = set(batch.dtype.names or ())
        timestamp_field = "ts_recv" if "ts_recv" in names else "ts_event" if "ts_event" in names else None
        if timestamp_field is None or not {"price", "size"}.issubset(names):
            raise RobustnessRunError("Nov 28 trades source schema lacks timestamp/price/size")
        for row in batch:
            if start <= int(row[timestamp_field]) < end:
                profile.add(int(row["price"]) / 1_000_000_000, int(row["size"]))
    if not profile.volume_by_tick:
        raise RobustnessRunError("Nov 28 completed RTH trades yielded an empty NY profile")
    return profile


def _metrics(trades: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    values = [float(row["r_multiple"]) for row in trades]
    base_metrics = class_b_v1._metrics(trades, dates)
    winners = sum(value > 0 for value in values)
    losers = sum(value < 0 for value in values)
    return {
        "net_r": base_metrics["net_r"], "profit_factor": base_metrics["profit_factor"],
        "profit_factor_cap": 10.0, "max_drawdown_r": base_metrics["max_drawdown_r"],
        "trades": len(values), "active_dates": base_metrics["active_dates"],
        "winners": winners, "losers": losers,
        "win_rate": winners / len(values) if values else 0.0,
        "average_r_per_trade": statistics.mean(values) if values else 0.0,
        "median_r_per_trade": statistics.median(values) if values else 0.0,
        "best_trade_r": max(values) if values else 0.0,
        "worst_trade_r": min(values) if values else 0.0,
    }


def _family_comparison(train: Mapping[str, Any], dec_jan: Mapping[str, Any]) -> dict[str, Any]:
    train_r = float(train["net_r"])
    train_trades = int(train["total_trades"])
    train_active = int(train["active_dates"])
    return {
        "train_net_r": train_r, "train_profit_factor": train["profit_factor"],
        "train_max_drawdown_r": train["max_drawdown_r"], "train_trades": train_trades,
        "train_active_dates": train_active,
        "delta_net_r": float(dec_jan["net_r"]) - train_r,
        "delta_pf": (float(dec_jan["profit_factor"]) - float(train["profit_factor"])
                     if dec_jan["profit_factor"] is not None and train["profit_factor"] is not None else None),
        "delta_dd_r": float(dec_jan["max_drawdown_r"]) - float(train["max_drawdown_r"]),
        "trade_count_ratio": int(dec_jan["trades"]) / train_trades if train_trades else None,
        "active_date_ratio": int(dec_jan["active_dates"]) / train_active if train_active else None,
        "generalization_ratio": float(dec_jan["net_r"]) / train_r if train_r else None,
    }


def _robustness_flags(train: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    if int(result["trades"]) < 8 or int(result["active_dates"]) < 6:
        return ["LOW_SAMPLE"]
    net = float(result["net_r"])
    pf = result["profit_factor"]
    if net > 0 and pf is not None and float(pf) > 1.0:
        return ["POSITIVE_ROBUSTNESS"]
    if net < 0 and (pf is None or float(pf) < 1.0):
        return ["NEGATIVE_ROBUSTNESS"]
    return ["MIXED_ROBUSTNESS"]


def _render_report(summary: Mapping[str, Any]) -> str:
    columns = ("FAMILY", "TRAIN R", "TRAIN TRADES", "DEC R", "JAN R", "DEC+JAN R",
               "DEC+JAN PF", "DEC+JAN DD", "DEC+JAN TRADES", "ACTIVE DATES",
               "GENERALIZATION RATIO", "ROBUSTNESS FLAGS")
    lines = ["# Ten-family Dec 2025 + Jan 2026 frozen-config robustness", "",
             f"Status: `{summary['status']}`", "",
             "This is a frozen-config retrospective robustness evaluation. No parameter search or config selection was run. "
             "PF is reported using the existing Candidate Tape V2 cap of 10. Dec 1 is omitted only for prior-Europe "
             "families because the Nov 28 Europe profile source has a documented coverage gap; the verified Nov 28 "
             "RTH trades source seeds the NY/RTH profile used by the other applicable families. Scheduled early-close "
             "dates are excluded from scored session tapes without changing frozen session windows and remain available "
             "for causal profile context.", "",
             f"Intended dates: {len(summary['intended_dates'])}; eligible normal full sessions: {summary['target_session_count']}",
             "Non-standard exclusions: " + json.dumps(summary.get("excluded_nonstandard_sessions", []), sort_keys=True), "",
             "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for family in summary["families"]:
        dec, jan, combined = family["december_2025"], family["january_2026"], family["dec_jan"]
        comp = family["train_comparison"]
        values = (family["family"], f"{comp['train_net_r']:.4f}", str(comp["train_trades"]),
                  f"{dec['net_r']:.4f}", f"{jan['net_r']:.4f}", f"{combined['net_r']:.4f}",
                  str(combined["profit_factor"]), f"{combined['max_drawdown_r']:.4f}",
                  str(combined["trades"]), str(combined["active_dates"]),
                  str(comp["generalization_ratio"]), ", ".join(family["robustness_flags"]))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Coverage", ""])
    for family in summary["families"]:
        coverage = family["coverage"]
        lines.extend([f"### {family['family']}", "",
                      f"Usable December dates ({len(coverage['usable_dec_dates'])}): " + ", ".join(coverage["usable_dec_dates"]),
                      f"Usable January dates ({len(coverage['usable_jan_dates'])}): " + ", ".join(coverage["usable_jan_dates"]),
                      f"Total usable: {coverage['total_usable_dates']}",
                      f"Missing: {json.dumps(coverage['missing_dates'], sort_keys=True)}", ""])
    lines.extend(["## Pooled summaries", "",
                  "```json", json.dumps(summary["pooled"], indent=2, sort_keys=True), "```", "",
                  "No October data informed selection. Final Aug/Sep 2026 OOS was not accessed.", ""])
    return "\n".join(lines)


def run(repository_root: Path = Path("."), *, config_path: Path = CONFIG_PATH,
        output_root: Path = OUTPUT_ROOT, workers: int = 4, resume: bool = False) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = (repository_root / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    output_root = (repository_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    payload, config_sha, configs = _config_payload(config_path)
    print(f"FROZEN_CONFIG_FOUND={config_path}", flush=True)
    print(f"FROZEN_CONFIG_SHA256={config_sha}", flush=True)
    print("FROZEN_CONFIG_SHA256_VERIFIED=true", flush=True)
    for family in EXPECTED_FAMILIES:
        print(f"FAMILY_CONFIG_LOADED={json.dumps({'family': family, **configs[family]}, sort_keys=True, separators=(',', ':'))}", flush=True)
    if output_root.exists() and not resume:
        raise RobustnessRunError(f"immutable output root already exists: {output_root}")
    if resume and (not output_root.is_dir() or not (output_root / "progress.json").is_file()):
        raise RobustnessRunError(f"resume root lacks a valid progress record: {output_root}")
    if workers < 1 or workers > 4:
        raise RobustnessRunError("workers must be between 1 and 4 to bound replay memory")
    if resume and workers != 1:
        raise RobustnessRunError("resume requires --workers 1 for deterministic sequential construction")

    dates, sources, source_manifest = _source_plan(repository_root)
    master_calendar = _json(repository_root / MASTER_ROOT / "calendar.json")
    calendar_dates = [str(item["day"]) for item in master_calendar.get("sessions", [])]
    if calendar_dates != dates:
        raise RobustnessRunError("acquisition and causal-master target calendars differ")
    semantic_sha = candidate_tape._semantic_sha256()
    profile_contract_hash, profile_hashes = _profile_contract_sha(dates, source_manifest)
    # Scheduled non-standard dates are unscored, but stay in `dates` for
    # causal profile construction and later-session context.
    scheduled_nonstandard_dates = [
        day for day in dates if bool(source_manifest["session_contract_by_date"][day]["scheduled_early_close"])
    ]
    eligible_candidate_dates = [day for day in dates if day not in scheduled_nonstandard_dates]

    # Construct source-bound job identities first, so existing tapes can be
    # audited and reused before any day is replayed or profile is regenerated.
    jobs_by_day: dict[str, dict[str, Any]] = {}
    for day in dates:
        chunks = sources[day]
        source_digest = _canonical_sha([
            {"path": str(path.relative_to(repository_root)), "sha256": next(
                row["sha256"] for row in source_manifest["input_files"]
                if row["path"] == str(path.relative_to(repository_root)))}
            for path in chunks
        ])
        tape_path = output_root / "tapes" / f"{day}-{candidate_tape.TAPE_FILENAME}"
        jobs_by_day[day] = {
            "day": day, "source_paths": [str(item) for item in chunks],
            "source_sha256": source_digest, "semantic_sha256": semantic_sha,
            "tape_path": str(tape_path),
            "check_strict_route_equivalence": day in STRICT_ROUTE_EQUIVALENCE_DATES,
        }

    progress_identity = {
        "config_sha256": config_sha, "semantic_sha256": semantic_sha,
        "profile_contract_sha256": profile_contract_hash,
        "source_manifest_sha256": {
            "base": source_manifest["base_manifest_sha256"],
            "dec_jan_extension": source_manifest["extension_manifest_sha256"],
        },
        "workers": workers,
    }
    if resume:
        existing_progress = _json(output_root / "progress.json")
        replay_reports_by_day, invalid_existing = _resume_tape_inventory(
            repository_root, output_root, eligible_candidate_dates, jobs_by_day, progress_identity,
            intended_session_count=len(dates),
        )
        completed_dates = [day for day in eligible_candidate_dates if day in replay_reports_by_day]
        missing_dates = [day for day in eligible_candidate_dates if day not in replay_reports_by_day]
        print(f"RECORDED_COMPLETED_SESSIONS={len(existing_progress.get('tapes', []))}", flush=True)
        print(f"VALID_EXISTING_TAPES={len(replay_reports_by_day)}", flush=True)
        print(f"INVALID_EXISTING_TAPES={json.dumps(invalid_existing)}", flush=True)
        print(f"COMPLETED_DATES={json.dumps(completed_dates)}", flush=True)
        print(f"MISSING_DATES={json.dumps(missing_dates)}", flush=True)
        print(f"NON_STANDARD_SESSION_EXCLUDED_DATES={json.dumps(scheduled_nonstandard_dates)}", flush=True)
        print(f"NEXT_DATE_TO_BUILD={missing_dates[0] if missing_dates else 'NONE'}", flush=True)
        print(f"EXISTING_PARTIAL_RUN_VALID={str(not invalid_existing).lower()}", flush=True)
    else:
        replay_reports_by_day = {}
        invalid_existing = []

    session_audit = _session_eligibility_audit(dates, sources, source_manifest, replay_reports_by_day)
    session_audit["source_manifest_sha256"] = {
        "base": source_manifest["base_manifest_sha256"],
        "dec_jan_extension": source_manifest["extension_manifest_sha256"],
    }
    session_audit_sha = _canonical_sha(session_audit)
    audit_path = output_root / "session-eligibility-audit.json"
    if resume:
        previous_audit_sha = existing_progress.get("session_eligibility_audit_sha256")
        if previous_audit_sha is not None and previous_audit_sha != session_audit_sha:
            raise RobustnessRunError("session eligibility audit changed since prior resume")
    _write_json(audit_path, session_audit)
    normal_full_dates = list(session_audit["normal_full_session_dates"])
    nonstandard_excluded_dates = list(session_audit["non_standard_session_excluded_dates"])
    incomplete_normal_dates = list(session_audit["incomplete_normal_session_dates"])
    print(f"INTENDED_DATES={json.dumps(dates)}", flush=True)
    print(f"NORMAL_FULL_SESSION_DATES={json.dumps(normal_full_dates)}", flush=True)
    print(f"NON_STANDARD_SESSION_EXCLUDED_DATES={json.dumps(nonstandard_excluded_dates)}", flush=True)
    print(f"INCOMPLETE_NORMAL_SESSION_DATES={json.dumps(incomplete_normal_dates)}", flush=True)
    print(f"SESSION_ELIGIBILITY_AUDIT={json.dumps(session_audit['per_date'], sort_keys=True)}", flush=True)
    if incomplete_normal_dates:
        failure = f"incomplete normal-session source coverage: {incomplete_normal_dates}"
        _write_json(output_root / "progress.json", {
            **progress_identity, "status": "FAILED", "completed_sessions": len(replay_reports_by_day),
            "total_sessions": len(dates), "eligible_total_sessions": len(normal_full_dates),
            "completed_dates": [day for day in normal_full_dates if day in replay_reports_by_day],
            "last_completed_date": max((day for day in normal_full_dates if day in replay_reports_by_day), default=None),
            "failure": failure, "session_eligibility_audit_sha256": session_audit_sha,
            "invalid_existing_tapes": invalid_existing,
            "tapes": [replay_reports_by_day[day] for day in normal_full_dates if day in replay_reports_by_day],
        })
        raise RobustnessRunError(failure)
    if set(normal_full_dates) != set(eligible_candidate_dates):
        raise RobustnessRunError("calendar eligibility and source-coverage classification disagree")
    completed_dates = [day for day in normal_full_dates if day in replay_reports_by_day]
    _write_json(output_root / "progress.json", {
        **progress_identity, "status": "RESUME_AUDITED" if resume else "SESSION_ELIGIBILITY_AUDITED",
        "completed_sessions": len(completed_dates), "total_sessions": len(dates),
        "eligible_total_sessions": len(normal_full_dates),
        "excluded_nonstandard_sessions": nonstandard_excluded_dates,
        "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
        "session_eligibility_audit_sha256": session_audit_sha,
        "completed_dates": completed_dates,
        "last_completed_date": completed_dates[-1] if completed_dates else None,
        "invalid_existing_tapes": invalid_existing,
        "previous_failure": ({"date": existing_progress.get("failed_date"),
                              "failure": existing_progress.get("failure"),
                              "resolved_as": NON_STANDARD_SESSION_REASON}
                             if resume and existing_progress.get("failed_date") in nonstandard_excluded_dates else None),
        "tapes": [replay_reports_by_day[day] for day in completed_dates],
    })

    profile_by_date, actual_profile_hashes = _profiles_by_date(dates, sources, source_manifest, repository_root)
    if actual_profile_hashes != profile_hashes:
        raise RobustnessRunError("profile dependency hashes changed during preflight")
    nov28_path = source_manifest["nov28_prior_rth_trades_path"]
    nov28_item = source_manifest["nov28_prior_rth_trades_manifest_item"]
    nov28_profile = _nov28_rth_profile(repository_root, nov28_path, nov28_item)
    prior_profiles_by_date = _prior_profiles_by_date(dates, profile_by_date, nov28_profile)

    jobs: list[dict[str, Any]] = []
    for day in normal_full_dates:
        job = dict(jobs_by_day[day])
        job["prior_profiles"] = prior_profiles_by_date[day]
        job["current_profiles"] = profile_by_date[day]
        jobs.append(job)
    jobs_by_day = {str(job["day"]): job for job in jobs}
    family_rows = {str(row["family"]): row for row in payload["families"]}
    trades_by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in EXPECTED_FAMILIES}
    usable_dates: dict[str, list[str]] = {family: [] for family in EXPECTED_FAMILIES}
    missing_dates: dict[str, dict[str, str]] = {family: {} for family in EXPECTED_FAMILIES}
    family_coverage: dict[str, dict[str, int]] = {
        family: {day: 0 for day in normal_full_dates} for family in EXPECTED_FAMILIES
    }
    replay_reports: list[dict[str, Any]] = [replay_reports_by_day[day]
                                            for day in normal_full_dates if day in replay_reports_by_day]
    build_results: dict[str, dict[str, Any]] = {}
    jobs_to_build = [job for job in jobs if str(job["day"]) not in replay_reports_by_day]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tapes").mkdir(parents=True, exist_ok=True)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    futures = {executor.submit(_build_tape_job, job): job for job in jobs_to_build}
    active_day: str | None = None
    try:
      for future in concurrent.futures.as_completed(futures):
        job = futures[future]
        day = str(job["day"])
        active_day = day
        result = future.result()
        tape_path = Path(result["tape_path"])
        tape = candidate_tape.load_tape(
            tape_path, source_sha256=str(job["source_sha256"]), semantic_sha256=semantic_sha,
        )
        available = set(tape.metadata.get("available_families", []))
        allowed_missing = PRIOR_EUROPE_FAMILIES if day == "2025-12-01" else set()
        unexpected_missing = set(EXPECTED_FAMILIES) - available - allowed_missing
        if unexpected_missing:
            raise RobustnessRunError(f"frozen family unavailable on {day}: {sorted(unexpected_missing)}")
        tape_report = {
            "date": day,
            "source_files": [str(path.relative_to(repository_root)) for path in sources[day]],
            "source_sha256": str(job["source_sha256"]),
            "tape_path": str(tape_path.relative_to(repository_root)),
            "tape_sha256": _sha256(tape_path), "candidate_count": len(tape.candidates),
            "event_count": len(tape.events), "available_families": sorted(available),
            "raw_replay_seconds": result["raw_replay_seconds"],
            "strict_route_equivalence": result["equivalence"],
            "source_last_timestamp_ns": result["source_last_timestamp_ns"],
            "last_strategy_timestamp_ns": result["last_strategy_timestamp_ns"],
            "book_state_at_last_strategy_record": result["book_state_at_last_strategy_record"],
            "post_session_terminal_state_accepted": result["post_session_terminal_state_accepted"],
        }
        if day in STRICT_ROUTE_EQUIVALENCE_DATES and result["equivalence"] is None:
            raise RobustnessRunError(f"strict route equivalence was not run for {day}")
        replay_reports_by_day[day] = tape_report
        build_results[day] = result
        replay_reports = [replay_reports_by_day[item] for item in normal_full_dates if item in replay_reports_by_day]
        completed = len(replay_reports_by_day)
        _write_json(output_root / "progress.json", {
            **progress_identity,
            "status": "BUILDING_CANDIDATE_TAPES", "completed_sessions": completed,
            "total_sessions": len(dates), "eligible_total_sessions": len(normal_full_dates),
            "excluded_nonstandard_sessions": nonstandard_excluded_dates,
            "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
            "session_eligibility_audit_sha256": session_audit_sha,
            "completed_dates": [row["date"] for row in replay_reports],
            "last_completed_date": replay_reports[-1]["date"] if replay_reports else None,
            "last_built_date": day, "invalid_existing_tapes": invalid_existing,
            "tapes": replay_reports,
        })
        print(f"CANDIDATE_TAPE_BUILD={completed}/{len(normal_full_dates)} DATE={day} "
              f"CANDIDATES={len(tape.candidates)} EVENTS={len(tape.events)} "
              f"ELAPSED={result['raw_replay_seconds']:.1f}s", flush=True)
    except BaseException as exc:
        current_reports = [replay_reports_by_day[item] for item in normal_full_dates if item in replay_reports_by_day]
        _write_json(output_root / "progress.json", {
            **progress_identity, "status": "FAILED",
            "completed_sessions": len(current_reports), "total_sessions": len(dates),
            "eligible_total_sessions": len(normal_full_dates),
            "excluded_nonstandard_sessions": nonstandard_excluded_dates,
            "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
            "session_eligibility_audit_sha256": session_audit_sha,
            "completed_dates": [row["date"] for row in current_reports],
            "last_completed_date": current_reports[-1]["date"] if current_reports else None,
            "failed_date": active_day, "failure": f"{type(exc).__name__}: {exc}",
            "invalid_existing_tapes": invalid_existing, "tapes": current_reports,
        })
        # Fail closed immediately; don't drain dozens of expensive queued
        # sessions after one required artifact/result has failed validation.
        for future in futures:
            future.cancel()
        processes = tuple(getattr(executor, "_processes", {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if set(replay_reports_by_day) != set(normal_full_dates):
        failure = (f"candidate tape build coverage mismatch: {len(replay_reports_by_day)}/"
                   f"{len(normal_full_dates)} eligible normal sessions")
        _write_json(output_root / "progress.json", {
            **progress_identity, "status": "FAILED", "completed_sessions": len(replay_reports_by_day),
            "total_sessions": len(dates), "eligible_total_sessions": len(normal_full_dates),
            "excluded_nonstandard_sessions": nonstandard_excluded_dates,
            "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
            "session_eligibility_audit_sha256": session_audit_sha,
            "completed_dates": [day for day in normal_full_dates if day in replay_reports_by_day],
            "last_completed_date": max((day for day in normal_full_dates if day in replay_reports_by_day), default=None),
            "failure": failure, "invalid_existing_tapes": invalid_existing,
            "tapes": [replay_reports_by_day[day] for day in normal_full_dates if day in replay_reports_by_day],
        })
        raise RobustnessRunError(failure)
    equivalence_failures = [day for day in STRICT_ROUTE_EQUIVALENCE_DATES
                            if replay_reports_by_day[day]["strict_route_equivalence"].get("status") != "PASS"]
    if equivalence_failures:
        failure = f"passing-date strict-route equivalence failed: {equivalence_failures}"
        _write_json(output_root / "progress.json", {
            **progress_identity, "status": "FAILED", "completed_sessions": len(normal_full_dates),
            "total_sessions": len(dates), "eligible_total_sessions": len(normal_full_dates),
            "excluded_nonstandard_sessions": nonstandard_excluded_dates,
            "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
            "session_eligibility_audit_sha256": session_audit_sha,
            "completed_dates": list(normal_full_dates),
            "last_completed_date": normal_full_dates[-1], "failure": failure,
            "invalid_existing_tapes": invalid_existing, "tapes": replay_reports,
        })
        raise RobustnessRunError(failure)
    replay_reports = [replay_reports_by_day[day] for day in normal_full_dates]
    _write_json(output_root / "progress.json", {
        **progress_identity, "status": "ALL_ELIGIBLE_NORMAL_SESSION_TAPES_BUILT",
        "completed_sessions": len(normal_full_dates), "total_sessions": len(dates),
        "eligible_total_sessions": len(normal_full_dates),
        "excluded_nonstandard_sessions": nonstandard_excluded_dates,
        "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
        "session_eligibility_audit_sha256": session_audit_sha,
        "completed_dates": list(normal_full_dates), "last_completed_date": normal_full_dates[-1],
        "candidate_tape_build_complete": True, "evaluation_started": False,
        "passing_date_equivalence": {day: replay_reports_by_day[day]["strict_route_equivalence"]
                                     for day in STRICT_ROUTE_EQUIVALENCE_DATES},
        "tapes": replay_reports,
    })
    print(f"ALL_ELIGIBLE_NORMAL_FULL_SESSION_DATES_BUILT=true ({len(normal_full_dates)}/{len(normal_full_dates)})", flush=True)
    print("PASSING_DATE_TAPE_EQUIVALENCE=PASS", flush=True)

    # Frozen evaluation is intentionally a second phase: it cannot begin until
    # every requested tape has been built, hash-checked, and family-validated.
    _write_json(output_root / "progress.json", {
        **progress_identity, "status": "EVALUATING_FROZEN_CONFIGS",
        "completed_sessions": len(normal_full_dates), "total_sessions": len(dates),
        "eligible_total_sessions": len(normal_full_dates),
        "excluded_nonstandard_sessions": nonstandard_excluded_dates,
        "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
        "session_eligibility_audit_sha256": session_audit_sha,
        "completed_dates": list(normal_full_dates), "last_completed_date": normal_full_dates[-1],
        "candidate_tape_build_complete": True, "evaluation_started": True,
        "passing_date_equivalence": {day: replay_reports_by_day[day]["strict_route_equivalence"]
                                     for day in STRICT_ROUTE_EQUIVALENCE_DATES},
        "tapes": replay_reports,
    })
    for eval_index, day in enumerate(normal_full_dates, 1):
        job = jobs_by_day[day]
        tape_path = Path(job["tape_path"])
        tape = candidate_tape.load_tape(
            tape_path, source_sha256=str(job["source_sha256"]), semantic_sha256=semantic_sha,
        )
        available = set(tape.metadata.get("available_families", []))
        for family in EXPECTED_FAMILIES:
            if family not in available:
                missing_dates[family][day] = "PRIOR_EUROPE_PROFILE_DEPENDENCY_INCOMPLETE_2025-11-28"
                continue
            usable_dates[family].append(day)
            family_tape = candidate_tape.CandidateTape(
                tape.metadata,
                tuple(row for row in tape.candidates if str(row.get("level", row.get("family_id"))) == family),
                tape.events,
            )
            family_coverage[family][day] = len(family_tape.candidates)
            class_a_fields = {item.name for item in fields(L2Config)}
            class_a = L2Config(**{key: value for key, value in configs[family]["class_a_config"].items()
                                  if key in class_a_fields})
            class_b = configs[family]["class_b_config"]
            _, family_trades = class_b_v2._evaluate_family_full(
                [tape], [day], family, class_a, class_b,
            )
            trades_by_family[family].extend(family_trades)
        print(f"FROZEN_FAMILY_EVALUATION={eval_index}/{len(normal_full_dates)} DATE={day}", flush=True)

    families_result: list[dict[str, Any]] = []
    for family in EXPECTED_FAMILIES:
        rows = trades_by_family[family]
        dec_dates = [day for day in usable_dates[family] if day.startswith("2025-12-")]
        jan_dates = [day for day in usable_dates[family] if day.startswith("2026-01-")]
        combined = _metrics(rows, usable_dates[family])
        dec = _metrics([row for row in rows if str(row["date"]).startswith("2025-12-")], dec_dates)
        jan = _metrics([row for row in rows if str(row["date"]).startswith("2026-01-")], jan_dates)
        train = family_rows[family]["train_metrics"]
        comparison = _family_comparison(train, combined)
        flags = _robustness_flags(train, combined)
        families_result.append({
            "family": family,
            "class_a_config": configs[family]["class_a_config"],
            "class_b_config": configs[family]["class_b_config"],
            "entry_delay_ms": 2.0,
            "coverage": {
                "usable_dec_dates": dec_dates, "usable_jan_dates": jan_dates,
                "total_usable_dates": len(usable_dates[family]),
                "missing_dates": missing_dates[family],
                "missing_reason": sorted(set(missing_dates[family].values())),
                "candidate_interactions_by_date": family_coverage[family],
            },
            "december_2025": dec, "january_2026": jan, "dec_jan": combined,
            "train_metrics": train, "train_comparison": comparison,
            "robustness_flags": flags,
        })

    pooled = _pooled(families_result)
    output = {
        "status": "PASS", "scope": "FROZEN_CONFIG_RETROSPECTIVE_ROBUSTNESS",
        "evidence_label": "DEC_JAN_RETROSPECTIVE_ROBUSTNESS_NOT_FINAL_OOS",
        "config_source": str(config_path.relative_to(repository_root)),
        "config_sha256": config_sha, "tape_version": candidate_tape.TAPE_VERSION,
        "evaluator_semantic_sha256": semantic_sha,
        "profile_source_contract_sha256": profile_contract_hash,
        "source_manifests": {key: value for key, value in source_manifest.items()
                              if key not in {"nov28_prior_rth_trades_path", "nov28_prior_rth_trades_manifest_item"}},
        "intended_dates": dates, "target_dates": normal_full_dates,
        "target_session_count": len(normal_full_dates),
        "excluded_nonstandard_sessions": [row for row in session_audit["per_date"]
                                          if row["date"] in nonstandard_excluded_dates],
        "session_eligibility_audit_path": str(audit_path.relative_to(repository_root)),
        "session_eligibility_audit_sha256": session_audit_sha,
        "dependency_dates": ["2025-11-28"], "dependency_scored": False,
        "dec1_profile_note": {
            "nov28_asia_europe_profile": "UNAVAILABLE_GAP_IN_2025-11-28_SOURCE",
            "nov28_rth_ny_profile": "SEEDED_FROM_VERIFIED_2025-11-28_RTH_TRADES",
            "excluded_families_on_dec1": sorted(PRIOR_EUROPE_FAMILIES),
        },
        "frozen_config_flags": {
            "optimization_performed": False, "config_reselection_performed": False,
            "frozen_configs_modified": False, "october_used_for_selection": False,
            "final_oos_accessed": False, "data_downloaded": False,
            "repository_committed": False,
        },
        "families": families_result, "pooled": pooled, "replay_sessions": replay_reports,
    }
    _write_json(output_root / "loaded-configs.json", {
        "config_path": str(config_path.relative_to(repository_root)), "config_sha256": config_sha,
        "families": [{"family": family, **configs[family]} for family in EXPECTED_FAMILIES],
    })
    _write_json(output_root / "summary.json", output)
    _write_json(output_root / "coverage.json", {
        row["family"]: row["coverage"] for row in families_result
    })
    _write_csv(output_root / "trade-ledger.csv", [row for family in EXPECTED_FAMILIES for row in trades_by_family[family]])
    (output_root / "report.md").write_text(_render_report(output), encoding="utf-8", newline="\n")
    _write_json(output_root / "run-manifest.json", {
        "status": "COMPLETE", "config_sha256": config_sha,
        "session_eligibility_audit_sha256": session_audit_sha,
        "intended_session_count": len(dates),
        "eligible_normal_session_count": len(normal_full_dates),
        "excluded_nonstandard_session_dates": nonstandard_excluded_dates,
        "source_manifest_sha256": {"base": source_manifest["base_manifest_sha256"],
                                    "dec_jan_extension": source_manifest["extension_manifest_sha256"]},
        "profile_source_sha256_by_date": profile_hashes,
        "semantic_sha256": semantic_sha,
        "optimization_performed": False, "downloads": 0, "network_calls": 0,
        "validation_accessed": False, "final_oos_accessed": False,
        "output_files": ["summary.json", "coverage.json", "loaded-configs.json", "trade-ledger.csv", "report.md",
                         "session-eligibility-audit.json"],
        "tapes": replay_reports,
    })
    print(f"ROBUSTNESS_OUTPUT_ROOT={output_root}", flush=True)
    print("TEN_FAMILY_DEC_JAN_ROBUSTNESS_RESULT = PASS", flush=True)
    for row in families_result:
        result = row["dec_jan"]
        print(f"{row['family']} | TRAIN_R={row['train_metrics']['net_r']:.4f} "
              f"TRAIN_TRADES={row['train_metrics']['total_trades']} DEC_R={row['december_2025']['net_r']:.4f} "
              f"JAN_R={row['january_2026']['net_r']:.4f} DEC_JAN_R={result['net_r']:.4f} "
              f"PF={result['profit_factor']} DD={result['max_drawdown_r']:.4f} "
              f"TRADES={result['trades']} ACTIVE_DATES={result['active_dates']} "
              f"GENERALIZATION={row['train_comparison']['generalization_ratio']} "
              f"FLAGS={','.join(row['robustness_flags'])}", flush=True)
    for flag in ("optimization_performed", "config_reselection_performed", "frozen_configs_modified",
                 "october_used_for_selection", "final_oos_accessed", "data_downloaded", "repository_committed"):
        print(f"{flag.upper()}={str(output['frozen_config_flags'][flag]).lower()}", flush=True)
    return output


def _pooled(family_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        combined = [row["dec_jan"] for row in rows]
        net_values = [float(item["net_r"]) for item in combined]
        pf_values = [float(item["profit_factor"]) for item in combined if item["profit_factor"] is not None]
        ratios = [float(row["train_comparison"]["generalization_ratio"]) for row in rows
                  if row["train_comparison"]["generalization_ratio"] is not None]
        return {
            "family_count": len(rows), "total_dec_jan_net_r": sum(net_values),
            "total_dec_jan_trades": sum(int(item["trades"]) for item in combined),
            "total_dec_jan_winners": sum(int(item["winners"]) for item in combined),
            "total_dec_jan_losers": sum(int(item["losers"]) for item in combined),
            "number_positive_families": sum(value > 0 for value in net_values),
            "number_negative_families": sum(value < 0 for value in net_values),
            "number_zero_families": sum(value == 0 for value in net_values),
            "number_low_sample_families": sum("LOW_SAMPLE" in row["robustness_flags"] for row in rows),
            "median_family_dec_jan_net_r": statistics.median(net_values) if net_values else 0.0,
            "median_family_dec_jan_pf": statistics.median(pf_values) if pf_values else None,
            "median_generalization_ratio": statistics.median(ratios) if ratios else None,
        }
    eligible = [row for row in family_rows if int(row["dec_jan"]["trades"]) >= 8
                and int(row["dec_jan"]["active_dates"]) >= 6]
    return {"all_10_families": summarize(family_rows),
            "excluding_low_sample": summarize(eligible),
            "low_sample_excluded_family_ids": [row["family"] for row in family_rows if row not in eligible]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="validate and reuse completed tapes in an existing run root")
    args = parser.parse_args(argv)
    try:
        run(args.repository_root, config_path=args.config, output_root=args.output_root,
            workers=args.workers, resume=args.resume)
    except (RobustnessRunError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"TEN_FAMILY_DEC_JAN_ROBUSTNESS_RESULT = FAIL: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
