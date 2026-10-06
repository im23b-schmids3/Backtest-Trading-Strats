"""Frozen public overnight-high/low sweep study using a levels-only OHLCV patch.

The patch contributes only hourly high/low values for 18:00 ET through midnight.
All signal, orderflow, quote, execution, and outcome events come from the sealed
native ES MBP-10 candidate tapes. Event-path calculations are delegated to the
existing frozen Sato public-study implementation; only the level source changes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_sato_overnight_ohlcv1h_download as download
from . import mac_2025_sato_overnight_ohlcv1h_quote_plan as quote_plan
from . import mac_2025_sato_prior_day_liquidity_sweep_v1 as engine

RUN_ID = "CMEOrderflow_SATO_ES_OVERNIGHT_LIQUIDITY_SWEEP_RECLAIM_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
RECON_ROOT = Path("research_runs/CMEOrderflow_SATO_ES_OVERNIGHT_LEVEL_RECONSTRUCTION_V1")
PATCH_MANIFEST = download.MANIFEST_PATH
PATCH_PLAN = download.QUOTE_PLAN_PATH
PATCH_ROOT = download.DATA_ROOT
DATES = tuple(engine.SPRING_DATES + engine.OCTOBER_DATES)
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
TICK = 0.25
RECONSTRUCTION_VERSION = "sato-overnight-high-low-reconstruction-v1"
STUDY_VERSION = "sato-overnight-liquidity-sweep-reclaim-v1"
CHECKPOINT_VERSION = "sato-overnight-levels-per-date-v1"


class OvernightStudyError(RuntimeError):
    """Source integrity, coverage, or frozen event-path contract failure."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=engine._json) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False, default=engine._json).encode() + b"\n")
    os.replace(temporary, path)


def _utc_midnight(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def _rth_open_utc(day: str) -> datetime:
    local = datetime.combine(datetime.fromisoformat(day).date(), dtime(9, 30), tzinfo=ET)
    return local.astimezone(UTC)


def expected_overnight_start_utc(day: str) -> datetime:
    """Timezone-derived prior-evening 18:00 ET start (DST-safe)."""
    current_date = datetime.fromisoformat(day).date()
    local = datetime.combine(current_date, dtime(18, 0), tzinfo=ET) - timedelta(days=1)
    return local.astimezone(UTC)


def combine_overnight_extremes(patch_highs: Sequence[float], patch_lows: Sequence[float],
                               native_high: float, native_low: float) -> dict[str, Any]:
    if not patch_highs or len(patch_highs) != len(patch_lows):
        raise OvernightStudyError("patch OHLC bars are empty or inconsistent")
    patch_high, patch_low = max(map(float, patch_highs)), min(map(float, patch_lows))
    high = max(patch_high, float(native_high))
    low = min(patch_low, float(native_low))
    if not all(math.isfinite(x) for x in (patch_high, patch_low, native_high, native_low, high, low)) or high < low:
        raise OvernightStudyError("invalid combined overnight extremes")
    return {
        "patch_high": patch_high, "patch_low": patch_low,
        "native_high": float(native_high), "native_low": float(native_low),
        "overnight_high": high, "overnight_low": low,
        "overnight_high_source": "PATCH" if patch_high > native_high else "NATIVE" if native_high > patch_high else "BOTH_EQUAL",
        "overnight_low_source": "PATCH" if patch_low < native_low else "NATIVE" if native_low < patch_low else "BOTH_EQUAL",
    }


def _read_csv_bars(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != download.CSV_FIELDS:
            raise OvernightStudyError(f"normalized patch CSV schema mismatch: {path}")
        return list(reader)


def validate_patch_and_reconstruct(*, write_artifacts: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail-closed verification of all 54 patch requests and 54 native tapes."""
    try:
        plan = download.load_and_validate_plan(PATCH_PLAN)
        patch_manifest = json.loads(PATCH_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OvernightStudyError(f"patch plan/manifest unreadable or invalid: {exc}") from exc
    requests = plan["requests"]
    request_by_date = {str(row["CURRENT_RTH_DATE"]): row for row in requests}
    entries = patch_manifest.get("requests")
    if patch_manifest.get("expected_request_count") != 54 or patch_manifest.get("expected_hourly_bar_count") != 103:
        raise OvernightStudyError("patch manifest expected totals are not 54 requests / 103 bars")
    if patch_manifest.get("invalid_request_count") != 0 or not isinstance(entries, list) or len(entries) != 54:
        raise OvernightStudyError("patch manifest is incomplete or has invalid requests")
    if set(request_by_date) != set(DATES) or len(request_by_date) != 54:
        raise OvernightStudyError("frozen patch plan date set is not the 54 target sessions")
    native_paths, native_rows, tape_hashes = engine._source_inventory()
    results: list[dict[str, Any]] = []
    total_bars = 0
    for entry in entries:
        day = str(entry.get("CURRENT_RTH_DATE", ""))
        if day not in request_by_date or entry.get("STATUS") not in {"VERIFIED", "SKIP_ALREADY_VERIFIED", "RECOVERED_VALID_PART"}:
            raise OvernightStudyError(f"unverified/unexpected patch manifest entry: {day}")
        request = request_by_date[day]
        raw_path = Path(str(entry.get("LOCAL_FILE", "")))
        csv_path = Path(str(entry.get("NORMALIZED_CSV", "")))
        if not raw_path.is_file() or not csv_path.is_file():
            raise OvernightStudyError(f"patch source missing for {day}: {raw_path} / {csv_path}")
        raw_hash = _sha(raw_path)
        if raw_hash != entry.get("SHA256"):
            raise OvernightStudyError(f"raw patch SHA-256 mismatch for {day}")
        bars = download.validate_dbn_file(raw_path, request)
        if len(bars) != int(request["EXPECTED_HOURLY_BAR_COUNT"]):
            raise OvernightStudyError(f"decoded patch bar count mismatch for {day}")
        csv_hash = _sha(csv_path)
        if csv_hash != entry.get("NORMALIZED_CSV_SHA256"):
            raise OvernightStudyError(f"normalized patch CSV SHA-256 mismatch for {day}")
        csv_bars = download.validate_bars(_read_csv_bars(csv_path), request)
        if csv_bars != bars:
            raise OvernightStudyError(f"normalized patch CSV differs from DBN decode for {day}")

        source_row = native_rows[day]
        native_path = native_paths[day]
        if request["RAW_ES_SYMBOL"] != source_row.get("symbol"):
            raise OvernightStudyError(f"patch/native raw contract mismatch for {day}: {request['RAW_ES_SYMBOL']} != {source_row.get('symbol')}")
        start = datetime.fromisoformat(str(request["PATCH_START_UTC"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(request["PATCH_END_UTC"]).replace("Z", "+00:00"))
        native_start = datetime.fromisoformat(str(source_row["start"]).replace("Z", "+00:00"))
        native_end = datetime.fromisoformat(str(source_row["end"]).replace("Z", "+00:00"))
        expected_start = expected_overnight_start_utc(day)
        midnight = _utc_midnight(day)
        rth_open = _rth_open_utc(day)
        boundary_valid = end == midnight == native_start
        if start != expected_start or end != midnight or native_start != midnight or native_end < rth_open:
            raise OvernightStudyError(f"patch/native coverage boundary or window invalid for {day}")
        event_tape, metadata = native._load_tape(day, native._tape_path(day), source_row["sha256"])
        ts = event_tape["timestamp_ns"].astype(np.int64, copy=False)
        lo_ns, open_ns = int(midnight.timestamp() * 1e9), int(rth_open.timestamp() * 1e9)
        trade_mask = ((ts >= lo_ns) & (ts < open_ns) & (event_tape["execution_size"] > 0)
                      & np.isfinite(event_tape["execution_price"]))
        trade_ids = np.flatnonzero(trade_mask)
        if not len(trade_ids) or int(ts[0]) > lo_ns or int(ts[-1]) < open_ns:
            raise OvernightStudyError(f"native tape does not cover midnight-to-RTH-open for {day}")
        native_prices = event_tape["execution_price"][trade_ids].astype(np.float64, copy=False)
        native_high, native_low = float(np.max(native_prices)), float(np.min(native_prices))
        level = combine_overnight_extremes(
            [float(bar["high"]) for bar in bars], [float(bar["low"]) for bar in bars], native_high, native_low)
        if any(abs(float(level[name]) / TICK - round(float(level[name]) / TICK)) > 1e-7
               for name in ("overnight_high", "overnight_low")):
            raise OvernightStudyError(f"reconstructed ONH/ONL off ES tick grid for {day}")
        patch_times = [str(bar["ts_event"]) for bar in bars]
        record = {
            "CURRENT_RTH_DATE": day, "RAW_ES_SYMBOL": request["RAW_ES_SYMBOL"],
            "OVERNIGHT_START_ET": expected_start.astimezone(ET).isoformat(),
            "PATCH_START_UTC": start.isoformat().replace("+00:00", "Z"),
            "PATCH_END_UTC": end.isoformat().replace("+00:00", "Z"),
            "NATIVE_START_UTC": native_start.isoformat().replace("+00:00", "Z"),
            "OVERNIGHT_END_ET": rth_open.astimezone(ET).isoformat(),
            "PATCH_BAR_COUNT": len(bars), "PATCH_BAR_TIMESTAMPS_UTC": patch_times,
            "PATCH_HIGH": level["patch_high"], "PATCH_LOW": level["patch_low"],
            "NATIVE_HIGH": native_high, "NATIVE_LOW": native_low,
            "OVERNIGHT_HIGH": level["overnight_high"], "OVERNIGHT_LOW": level["overnight_low"],
            "OVERNIGHT_RANGE_TICKS": (level["overnight_high"] - level["overnight_low"]) / TICK,
            "OVERNIGHT_HIGH_SOURCE": level["overnight_high_source"],
            "OVERNIGHT_LOW_SOURCE": level["overnight_low_source"],
            "PATCH_FILE": str(raw_path), "PATCH_SHA256": raw_hash,
            "PATCH_CSV": str(csv_path), "PATCH_CSV_SHA256": csv_hash,
            "NATIVE_FILE": str(native_path), "NATIVE_SOURCE_HASH": source_row["sha256"],
            "NATIVE_TAPE": str(native._tape_path(day)), "NATIVE_TAPE_SHA256": tape_hashes[day],
            "NATIVE_TRADE_COUNT_0000_TO_RTH_OPEN": int(len(trade_ids)),
            "NATIVE_FIRST_TRADE_UTC": datetime.fromtimestamp(int(ts[trade_ids[0]]) / 1e9, UTC).isoformat(),
            "NATIVE_LAST_TRADE_UTC": datetime.fromtimestamp(int(ts[trade_ids[-1]]) / 1e9, UTC).isoformat(),
            "BOUNDARY_VALID": boundary_valid, "STATUS": "VERIFIED",
        }
        if not boundary_valid:
            raise OvernightStudyError(f"patch/native boundary failed for {day}")
        results.append(record)
        total_bars += len(bars)
        del event_tape
    if len(results) != 54 or total_bars != 103:
        raise OvernightStudyError(f"reconstruction totals invalid: sessions={len(results)}, bars={total_bars}")
    results.sort(key=lambda row: row["CURRENT_RTH_DATE"])
    summary = {
        "reconstruction_version": RECONSTRUCTION_VERSION, "status": "PASS",
        "patch_request_count": len(results), "patch_bar_count": total_bars,
        "patch_invalid_requests": 0, "eligible_sessions": len(results),
        "patch_manifest_sha256": _sha(PATCH_MANIFEST), "patch_plan_sha256": _sha(PATCH_PLAN),
        "patch_request_set_sha256": patch_manifest["frozen_request_set_sha256"],
        "native_manifest_sha256": _sha(baseline.DATA_ROOT / baseline.MANIFEST_NAME),
        "boundary_valid_sessions": sum(bool(row["BOUNDARY_VALID"]) for row in results),
        "patch_determined_onh_count": sum(row["OVERNIGHT_HIGH_SOURCE"] == "PATCH" for row in results),
        "patch_determined_onl_count": sum(row["OVERNIGHT_LOW_SOURCE"] == "PATCH" for row in results),
        "patch_determined_either_count": sum("PATCH" in (row["OVERNIGHT_HIGH_SOURCE"], row["OVERNIGHT_LOW_SOURCE"]) for row in results),
        "native_midnight_to_open_determined_both_count": sum(row["OVERNIGHT_HIGH_SOURCE"] == row["OVERNIGHT_LOW_SOURCE"] == "NATIVE" for row in results),
        "sessions": results,
    }
    if write_artifacts:
        RECON_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(RECON_ROOT / "overnight-levels.json", summary)
        columns = [key for key in results[0] if key != "PATCH_BAR_TIMESTAMPS_UTC"]
        with (RECON_ROOT / "overnight-levels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in results:
                writer.writerow({key: json.dumps(row[key], separators=(",", ":")) if isinstance(row[key], (list, dict)) else row[key] for key in columns})
        coverage = {"schema": "mbp-10", "dataset": "GLBX.MDP3", "instrument": "ES",
                    "periods": ["SPRING_2025", "OCTOBER_2025"], "sessions": results,
                    "patch_is_levels_only": True, "native_mbp10_is_signal_and_execution_source": True}
        _write_json(RECON_ROOT / "source-coverage.json", coverage)
        recon_manifest = {key: value for key, value in summary.items() if key != "sessions"}
        recon_manifest.update({"sessions": [{k: row[k] for k in ("CURRENT_RTH_DATE", "RAW_ES_SYMBOL", "PATCH_SHA256", "NATIVE_SOURCE_HASH", "NATIVE_TAPE_SHA256", "BOUNDARY_VALID", "STATUS")} for row in results]})
        _write_json(RECON_ROOT / "reconstruction-manifest.json", recon_manifest)
        report = ("# Sato overnight level reconstruction\n\n"
                  f"Status: PASS; {len(results)} sessions, {total_bars} verified hourly bars.\n\n"
                  f"Exact patch/native midnight boundary: {summary['boundary_valid_sessions']}/54.\n\n"
                  f"Patch determined ONH: {summary['patch_determined_onh_count']}; ONL: {summary['patch_determined_onl_count']}; either: {summary['patch_determined_either_count']}.\n\n"
                  "The OHLCV-1h patch is used only for overnight level reconstruction. Native ES MBP-10 executions remain the source for signal, orderflow, quote and execution paths.\n")
        (RECON_ROOT / "report.md").write_text(report, encoding="utf-8")
        hashes = {path.name: _sha(path) for path in sorted(RECON_ROOT.iterdir()) if path.is_file() and path.name != "artifact-hashes.json"}
        _write_json(RECON_ROOT / "artifact-hashes.json", hashes)
    return results, summary


def _level_adapter(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"pdh": float(row["OVERNIGHT_HIGH"]), "pdl": float(row["OVERNIGHT_LOW"]),
            "midpoint": (float(row["OVERNIGHT_HIGH"]) + float(row["OVERNIGHT_LOW"])) / 2,
            "range_ticks": float(row["OVERNIGHT_RANGE_TICKS"]), "coverage_complete": True,
            "overnight_high": float(row["OVERNIGHT_HIGH"]), "overnight_low": float(row["OVERNIGHT_LOW"]),
            "level_source": "OHLCV1H_PATCH_PLUS_NATIVE_MBP10_TRADES"}


def _rename_event_level_fields(event: dict[str, Any]) -> dict[str, Any]:
    mapping = {"PDH": "ONH", "PDL": "ONL"}
    event["side"] = mapping.get(event.get("side"), event.get("side"))
    event["level_type"] = event["side"]
    event["level_family"] = "OVERNIGHT_HIGH_LOW"
    if "both_prior_day_sides_swept" in event:
        event["both_overnight_sides_swept"] = event.pop("both_prior_day_sides_swept")
    for old, new in (("prior_day_high", "overnight_high"), ("prior_day_low", "overnight_low"),
                     ("prior_day_midpoint", "overnight_midpoint"), ("prior_day_range_ticks", "overnight_range_ticks")):
        if old in event:
            event[new] = event.pop(old)
    first = event.get("first_touch")
    if isinstance(first, dict):
        for old, new in (("prior_midpoint", "overnight_midpoint"), ("opposite_prior_extreme", "opposite_overnight_extreme")):
            if old in first:
                first[new] = first.pop(old)
        for key, result_name in (("session_vwap", "VWAP_FIRST"), ("opposite_overnight_extreme", "TARGET2_FIRST")):
            target = first.get(key)
            if isinstance(target, dict) and target.get("result") == "TARGET_FIRST":
                target["result"] = result_name
    return event


def _checkpoint_key(day: str, level: Mapping[str, Any], native_sha: str, patch_manifest_sha: str,
                    patch_sha: str, study_sha: str) -> str:
    return _canonical_hash({"date": day, "overnight_level": level, "native_source_sha256": native_sha,
                            "patch_manifest_sha256": patch_manifest_sha, "patch_file_sha256": patch_sha,
                            "study_sha256": study_sha, "checkpoint_version": CHECKPOINT_VERSION})


def run(*, smoke: bool = False, resume: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    levels, reconstruction = validate_patch_and_reconstruct(write_artifacts=True)
    level_by_date = {row["CURRENT_RTH_DATE"]: row for row in levels}
    source_paths, source_rows, tape_hashes = engine._source_inventory()
    days = (DATES[0],) if smoke else DATES
    config = dict(engine.CONFIG)
    config.update({"study_version": STUDY_VERSION, "run_id": RUN_ID,
                   "model": "public overnight-high/low liquidity sweep + fast reclaim + public orderflow confirmation",
                   "source": "native GLBX.MDP3 ES mbp-10 only for signal/orderflow/quote/execution; Databento OHLCV-1h patch for ONH/ONL values only",
                   "prior_rth_level_source": "NOT_USED; replaced by complete overnight [prior-date 18:00 ET,current-date 09:30 ET) extrema",
                   "targets": "freeze current-session VWAP at reclaim and opposite overnight extreme; midpoint descriptive only",
                   "private_sato_levels_used": False, "true_mbo_used": False, "data_downloaded": False,
                   "level_family": "OVERNIGHT_HIGH_LOW", "overnight_start": "prior calendar date 18:00 America/New_York",
                   "overnight_end": "current RTH date 09:30 America/New_York exclusive",
                   "hourly_patch_use": "OHLCV1H is used only to reconstruct ONH/ONL; no signal/orderflow/execution fields",
                   "native_source": "GLBX.MDP3 ES mbp-10 canonical tape and actual executions",
                   "source_strategy_engine": "sato-prior-day public event-path functions with only level inputs replaced"})
    engine_study_sha = _canonical_hash({"config": engine.CONFIG, "version": engine.STUDY_VERSION})
    study_sha = _canonical_hash({"config": config, "engine_config_sha256": engine.CONFIG_SHA256,
                                 "engine_study_sha256": engine_study_sha})
    output = OUT_ROOT if not smoke else OUT_ROOT / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    patch_manifest_sha = _sha(PATCH_MANIFEST)
    all_events: list[dict[str, Any]] = []
    all_bars: list[dict[str, Any]] = []
    acceptance: list[dict[str, Any]] = []
    cp_hashes: dict[str, str] = {}
    for day in days:
        source_row = source_rows[day]
        level_row = level_by_date[day]
        cp = checkpoints / f"{day}.json.gz"
        cp_key = _checkpoint_key(day, level_row, source_row["sha256"], patch_manifest_sha,
                                 level_row["PATCH_SHA256"], study_sha)
        loaded = False
        if resume and cp.is_file():
            try:
                with gzip.open(cp, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("checkpoint_key") == cp_key:
                    events, bars, no_reclaim = payload["events"], payload["bars"], payload["acceptance"]
                    loaded = True
            except (OSError, ValueError, KeyError):
                loaded = False
        if not loaded:
            tape, _ = native._load_tape(day, native._tape_path(day), source_row["sha256"])
            events, bars, no_reclaim = engine._daily_run(day, "OVERNIGHT_LEVELS", tape, _level_adapter(level_row))
            del tape
            for event in events:
                _rename_event_level_fields(event)
                event.pop("previous_valid_rth_date", None)
                event["level_source"] = level_row["OVERNIGHT_HIGH_SOURCE"] if event["side"] == "ONH" else level_row["OVERNIGHT_LOW_SOURCE"]
            for event in no_reclaim:
                _rename_event_level_fields(event)
                event.pop("previous_valid_rth_date", None)
            for bar in bars:
                bar.pop("previous_valid_rth_date", None)
            temp = cp.with_name(f".{cp.name}.{os.getpid()}.tmp")
            with temp.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                    stream.write(json.dumps({"checkpoint_key": cp_key, "events": events, "bars": bars,
                                             "acceptance": no_reclaim}, sort_keys=True, allow_nan=False, default=engine._json).encode())
            os.replace(temp, cp)
        # Normalize cached outputs too: older compatible event checkpoints may
        # contain generic target labels, while exported overnight artifacts use
        # the preregistered VWAP_FIRST / TARGET2_FIRST terminology.
        for event in events:
            _rename_event_level_fields(event)
            event.pop("previous_valid_rth_date", None)
            event["level_source"] = level_row["OVERNIGHT_HIGH_SOURCE"] if event["side"] == "ONH" else level_row["OVERNIGHT_LOW_SOURCE"]
        for event in no_reclaim:
            _rename_event_level_fields(event)
            event.pop("previous_valid_rth_date", None)
        for bar in bars:
            bar.pop("previous_valid_rth_date", None)
        all_events.extend(events)
        all_bars.extend(bars)
        acceptance.extend(no_reclaim)
        cp_hashes[day] = _sha(cp)

    # Aggregate before relabeling because the shared frozen functions key on PDH/PDL.
    # Re-label on copies so exported artifacts carry the correct ONH/ONL semantics.
    aggregate_events = []
    for event in all_events:
        item = dict(event)
        original = item.get("side")
        item["side"] = {"ONH": "PDH", "ONL": "PDL"}.get(original, original)
        item["level_type"] = item["side"]
        item["previous_valid_rth_date"] = "OVERNIGHT_LEVELS"
        for new, old in (("prior_day_high", "overnight_high"), ("prior_day_low", "overnight_low"),
                         ("prior_day_midpoint", "overnight_midpoint"), ("prior_day_range_ticks", "overnight_range_ticks")):
            if new not in item and old in item:
                item[new] = item[old]
        aggregate_events.append(item)
    agg = engine._flatten_components(aggregate_events)
    groups = engine._event_groups(aggregate_events)
    lodo, lowo = engine._lodo_lowo(aggregate_events)
    permutations = engine._resampling(aggregate_events)
    price_control = engine._price_only_control(aggregate_events)
    outputs = {
        "five-minute-bars.jsonl.gz": all_bars,
        "breach-events.jsonl.gz": all_events,
        "reclaim-events.jsonl.gz": [e for e in all_events if e["reclaim_only"]],
        "acceptance-controls.jsonl.gz": acceptance,
        "raw-markouts.json": {g: {str(h): engine._stats([e.get("markouts", {}).get(str(h), {}).get("raw") for e in es]) for h in engine.MARKOUT_MS} for g, es in groups.items()},
        "executable-markouts.json": {g: {str(h): engine._stats([e.get("markouts", {}).get(str(h), {}).get("quote") for e in es]) for h in engine.MARKOUT_MS} for g, es in groups.items()},
        "actual-fill-markouts.json": {g: {str(h): engine._stats([e.get("markouts", {}).get(str(h), {}).get("actual") for e in es]) for h in engine.MARKOUT_MS} for g, es in groups.items()},
        "mfe-mae.json": {g: {str(h): {"raw_mfe": engine._stats([e.get("mfe_mae", {}).get(str(h), {}).get("raw", {}).get("mfe") for e in es]),
                                             "raw_mae": engine._stats([e.get("mfe_mae", {}).get(str(h), {}).get("raw", {}).get("mae") for e in es]),
                                             "actual_mfe": engine._stats([e.get("mfe_mae", {}).get(str(h), {}).get("actual", {}).get("mfe") for e in es]),
                                             "actual_mae": engine._stats([e.get("mfe_mae", {}).get(str(h), {}).get("actual", {}).get("mae") for e in es])} for h in engine.EXCURSION_MS} for g, es in groups.items()},
        "vwap-first-touch.json": {f"{e['date']}/{e['side']}": e.get("first_touch", {}).get("session_vwap") for e in all_events if e["reclaim_only"]},
        "overnight-midpoint-first-touch.json": {f"{e['date']}/{e['side']}": e.get("first_touch", {}).get("overnight_midpoint") for e in all_events if e["reclaim_only"]},
        "opposite-overnight-extreme-first-touch.json": {f"{e['date']}/{e['side']}": e.get("first_touch", {}).get("opposite_overnight_extreme") for e in all_events if e["reclaim_only"]},
        "period-first-touch.json": _period_first_touch(all_events),
        "event-groups.json": {"ALL_LEVEL_BREACHES": len(all_events),
                               "RECLAIMED_SWEEPS": sum(bool(e["reclaim_only"]) for e in all_events),
                               "PUBLIC_STACK_QUALIFIED_SWEEPS": sum(bool(e["public_stack_qualified"]) for e in all_events),
                               "NO_RECLAIM_ACCEPTANCE_CANDIDATES": len(acceptance)},
        "volume-spike-analysis.json": engine._component_effects(aggregate_events, "volume_spike"),
        "delta-divergence-analysis.json": engine._component_effects(aggregate_events, "delta_divergence"),
        "absorption-analysis.json": engine._component_effects(aggregate_events, "absorption"),
        "public-stack-analysis.json": {"summary": agg["groups"]["PUBLIC_STACK_QUALIFIED_SWEEPS"], "qualified_n": agg["event_counts"]["qualified"],
                                        "reclaim_only_n": agg["event_counts"]["reclaimed"], "unavailable_component_counts": agg["components"]},
        "acceptance-markouts.json": {"breakout_direction": {str(h): engine._stats([e.get("acceptance_markouts", {}).get(str(h)) for e in acceptance]) for h in engine.ACCEPTANCE_MS}},
        "sweep-extension.json": {"continuous_ticks": engine._stats([e["sweep_extension_ticks"] for e in all_events]), "terciles": engine._terciles(aggregate_events, "sweep_extension_ticks")},
        "overnight-range-analysis.json": {"continuous_ticks": engine._stats([e["overnight_range_ticks"] for e in all_events]),
                                           "terciles": engine._terciles(aggregate_events, "prior_day_range_ticks")},
        "time-results.json": {bucket: {"n": sum(e["breach_time_bucket"] == bucket for e in all_events),
                                          "actual_5m": engine._stats([e.get("markouts", {}).get("300000", {}).get("actual") for e in all_events if e["breach_time_bucket"] == bucket])}
                              for bucket in ("09:30-09:40", "09:40-09:50", "09:50-10:00")},
        "period-results.json": agg["period"], "side-results.json": {"ONH": agg["side"]["PDH"], "ONL": agg["side"]["PDL"]},
        "daily-results.json": _overnight_daily_rows(agg["daily"], level_by_date),
        "daily-robustness.json": _daily_robustness(all_events, level_by_date),
        "weekly-results.json": agg["weekly"], "lodo-results.json": lodo,
        "lowo-results.json": lowo, "permutation-results.json": permutations,
        "first-sweep-sensitivity.json": {"primary": agg["groups"],
                                           "first_sweep_of_session_only": engine._group_summary(_first_sweeps(aggregate_events), "markouts")},
        "reclaim-speed.json": {s: {"n": sum(e.get("reclaim_speed") == s for e in all_events),
                                      "actual_5m": engine._stats([e.get("markouts", {}).get("300000", {}).get("actual") for e in all_events if e.get("reclaim_speed") == s])}
                               for s in ("SAME_BAR", "NEXT_BAR", "THIRD_BAR")},
        "execution-hurdle.json": {"entry_delay_ns": engine.ENTRY_DELAY_NS, "one_adverse_tick_entry": True,
                                  "reclaim_actual_5m_ticks": engine._stats([e.get("markouts", {}).get("300000", {}).get("actual") for e in all_events if e["reclaim_only"]]),
                                  "qualified_actual_5m_ticks": engine._stats([e.get("markouts", {}).get("300000", {}).get("actual") for e in all_events if e["public_stack_qualified"]])},
    }
    for name, value in outputs.items():
        path = output / name
        if name.endswith(".jsonl.gz"):
            _write_jsonl_gz(path, value)
        else:
            _write_json(path, value)
    summary = {"study_id": RUN_ID, "study_version": STUDY_VERSION, "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
               "target_dates": list(days), "eligible_sessions": len(days), "event_counts": agg["event_counts"],
               "public_stack_qualified_n": agg["event_counts"]["qualified"], "candidate_hypothesis": "NONE",
               "orderflow_adds_incremental_information": "insufficient" if agg["event_counts"]["qualified"] < 20 else "requires_review",
               "primary_decision": "INSUFFICIENT_SAMPLE" if agg["event_counts"]["reclaimed"] < 20 or agg["event_counts"]["qualified"] < 20 else "REQUIRES_REVIEW",
               "next_step": "REQUIRE_ADDITIONAL_PREDECLARED_DATA" if agg["event_counts"]["reclaimed"] < 20 or agg["event_counts"]["qualified"] < 20 else "MANUAL_REVIEW_REQUIRED",
               "no_optimization": True, "data_downloaded": False, "oos_accessed": False,
               "patch_used_for_levels_only": True, "native_mbp10_used_for_signal_and_execution": True,
               "runtime_seconds": time.monotonic() - started}
    _write_json(output / "summary.json", summary)
    _write_json(output / "study-config.json", config)
    report = (f"# {RUN_ID}\n\nStatus: {summary['status']}\n\nEligible sessions: {len(days)}.\n\n"
              f"Event counts: {agg['event_counts']}\n\nPrimary decision: {summary['primary_decision']}; candidate hypothesis: NONE.\n\n"
              "Public-model replication only. OHLCV-1h contributes ONH/ONL only; native ES MBP-10 remains the signal, orderflow, quote and execution source. No optimization, private Sato model, true MBO, download, or OOS access.\n")
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {"study_id": RUN_ID, "study_version": STUDY_VERSION, "status": summary["status"],
                "config_sha256": _canonical_hash(config), "engine_config_sha256": engine.CONFIG_SHA256,
                "engine_study_sha256": engine_study_sha, "patch_manifest_sha256": patch_manifest_sha,
                "patch_request_set_sha256": reconstruction["patch_request_set_sha256"],
                "native_source_sha256_by_date": {d: source_rows[d]["sha256"] for d in days},
                "native_tape_sha256_by_date": {d: tape_hashes[d] for d in days},
                "patch_sha256_by_date": {d: level_by_date[d]["PATCH_SHA256"] for d in days},
                "checkpoint_sha256_by_date": cp_hashes, "eligible_dates": list(days),
                "no_validation_selection": True, "no_final_oos_access": True}
    _write_json(output / "run-manifest.json", manifest)
    hashes = {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("artifact-hashes.json", "run-manifest.json")}
    _write_json(output / "artifact-hashes.json", hashes)
    _write_json(output / "run-manifest.json", {**manifest, "artifact_sha256": hashes})
    return summary


def _first_sweeps(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    first: dict[str, Mapping[str, Any]] = {}
    for event in sorted(events, key=lambda row: row["breach_timestamp_ns"]):
        first.setdefault(str(event["date"]), event)
    return [dict(value) for value in first.values()]


def _overnight_daily_rows(rows: Sequence[Mapping[str, Any]], levels: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        day = str(row["date"])
        item = dict(row)
        item["onh_breach"] = item.pop("pdh_breach", False)
        item["onl_breach"] = item.pop("pdl_breach", False)
        item.update({key: levels[day][key] for key in ("OVERNIGHT_HIGH", "OVERNIGHT_LOW", "OVERNIGHT_HIGH_SOURCE", "OVERNIGHT_LOW_SOURCE")})
        item["events"] = [_rename_event_level_fields(dict(event)) for event in item.get("events", [])]
        result.append(item)
    return result


def _daily_robustness(events: Sequence[Mapping[str, Any]], levels: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, list[Mapping[str, Any]]] = {day: [] for day in levels}
    for event in events:
        by_date[str(event["date"])].append(event)
    rows = []
    for day, level in levels.items():
        daily_events = sorted(by_date[day], key=lambda event: event["breach_timestamp_ns"])
        rows.append({"date": day, "onh": level["OVERNIGHT_HIGH"], "onl": level["OVERNIGHT_LOW"],
                     "onh_source": level["OVERNIGHT_HIGH_SOURCE"], "onl_source": level["OVERNIGHT_LOW_SOURCE"],
                     "onh_breach": any(e["side"] == "ONH" for e in daily_events),
                     "onl_breach": any(e["side"] == "ONL" for e in daily_events),
                     "reclaims": sum(bool(e["reclaim_only"]) for e in daily_events),
                     "qualified_sweeps": sum(bool(e["public_stack_qualified"]) for e in daily_events),
                     "events": daily_events})
    return rows


def _period_first_touch(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for period in ("SPRING_2025", "OCTOBER_2025"):
        reclaimed = [event for event in events if event["period"] == period and event["reclaim_only"]]
        result[period] = {}
        for key, name in (("session_vwap", "VWAP"), ("overnight_midpoint", "OVERNIGHT_MIDPOINT"),
                          ("opposite_overnight_extreme", "OPPOSITE_OVERNIGHT_EXTREME")):
            counts: dict[str, int] = {}
            for event in reclaimed:
                touch = event.get("first_touch", {}).get(key)
                label = str(touch.get("result")) if isinstance(touch, dict) else "UNAVAILABLE"
                counts[label] = counts.get(label, 0) + 1
            result[period][name] = {"n": len(reclaimed), "results": counts}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("--run is required")
    print(json.dumps(run(smoke=args.smoke, resume=not args.no_resume), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
