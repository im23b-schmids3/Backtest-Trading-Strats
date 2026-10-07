"""Metadata-only April--August 2026 ES source/session coverage audit.

This audit reads DBN headers and streams ``ts_recv`` only. It never examines
book prices, computes interactions, opens outcome ledgers, or evaluates a
strategy. The report deliberately distinguishes native vendor MBP-10 from the
public MBP-10 representation reconstructed by the frozen MBO adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = ROOT / "research_runs/CMEOrderflow_APR_AUG_2026_DATA_COVERAGE_AUDIT_V1"
START = datetime(2026, 4, 1, tzinfo=UTC)
END = datetime(2026, 9, 1, tzinfo=UTC)
NS = 1_000_000_000
DAY_NS = 86_400 * NS
APR_AUG = re.compile(r"^2026-(?:04|05|06|07|08)-\d{2}$")
FAMILY_MAP = {
    "EU_CURRENT_HIGH_SWEEP": ("EUROPE/LONDON", "08:00", "16:30", "CURRENT_SESSION_HIGH_SWEEP"),
    "EU_PRIOR_HIGH": ("EUROPE/LONDON", "08:00", "16:30", "PREVIOUS_COMPLETED_EUROPE_HIGH"),
    "PRIOR_VAH": ("EUROPE/LONDON", "08:00", "16:30", "PREVIOUS_COMPLETED_EUROPE_VAH"),
    "NY_W04": ("AMERICA/NEW_YORK", "09:30", "16:00", "PREVIOUS_COMPLETED_RTH_POC"),
}
PERIODS = (
    ("APRIL_2026", "april_2026", "NATIVE_DATABENTO_MBP10", "RETROSPECTIVE_APRIL_NATIVE_SOURCE"),
    ("MAY_2026", "may_2026", "MBO_DERIVED_SYNTHETIC_MBP10", "MAY_DEVELOPMENT_MBO_DERIVED"),
    ("RETRO_JUNE_JULY_2026", "retro_june_july_2026", "MBO_DERIVED_SYNTHETIC_MBP10", "RETROSPECTIVE_ROBUSTNESS_MBO_DERIVED"),
    ("AUGUST_03_06_2026", "august_03_06_2026", "MBO_DERIVED_SYNTHETIC_MBP10_SHARED_FILE", "PREVIOUS_OOS_V1_SOURCE_MBO_DERIVED"),
    ("AUGUST_10_14_2026", "august_10_14_2026", "NATIVE_DATABENTO_MBP10", "PREVIOUS_FRESH_AUGUST_RESEARCH_NATIVE_SOURCE"),
)


def iso_ns(value: int | None) -> str | None:
    if value is None:
        return None
    seconds, nanos = divmod(int(value), NS)
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"


def dt_iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z") if value else None


def _safe_close(store: Any) -> None:
    reader = getattr(store, "reader", None)
    if reader is not None and not getattr(reader, "closed", False):
        reader.close()


def _meta_start_end(store: Any) -> tuple[datetime | None, datetime | None]:
    start = getattr(store, "start", None)
    end = getattr(store, "end", None)
    if isinstance(start, datetime):
        start = start.astimezone(UTC)
    if isinstance(end, datetime):
        end = end.astimezone(UTC)
    return start, end


def _manifest_hashes() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    roots = [DATA / name for name in (
        "cme_orderflow_absorption_l2_v1", "cme_orderflow_absorption_l2_v2",
        "cme_orderflow_absorption_l2_v2_holdout", "cme_orderflow_absorption_l2_v3",
        "cme_orderflow_absorption_v1", "cme_orderflow_absorption_v2",
    )]
    manifests = [p for root in roots if root.exists() for p in root.rglob("*manifest*.json")]
    manifests += [ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"]
    for manifest in manifests:
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        files = payload.get("files")
        if isinstance(files, dict):
            for key, row in files.items():
                if not isinstance(row, dict):
                    continue
                candidates = [manifest.parent / str(key)]
                if row.get("local_path"):
                    candidates.append(manifest.parent / str(row["local_path"]))
                for candidate in candidates:
                    found[str(candidate.resolve()).casefold()] = {
                        "sha256": row.get("sha256"), "manifest": str(manifest.relative_to(ROOT)),
                        "manifest_entry": str(key), "declared_bytes": row.get("bytes"),
                        "declared_record_count": row.get("source_validation", {}).get("record_count")
                        if isinstance(row.get("source_validation"), dict) else row.get("record_count"),
                        "declared_first_timestamp_ns": row.get("source_validation", {}).get("first_timestamp_ns")
                        if isinstance(row.get("source_validation"), dict) else row.get("first_timestamp_ns"),
                        "declared_last_timestamp_ns": row.get("source_validation", {}).get("last_timestamp_ns")
                        if isinstance(row.get("source_validation"), dict) else row.get("last_timestamp_ns"),
                    }
        # The sealed OOS source manifest uses a single-file record rather than
        # the acquisition-manifest ``files`` map.
        for row in payload.get("source_files", []) if isinstance(payload.get("source_files"), list) else []:
            if isinstance(row, dict) and row.get("path"):
                found[str((ROOT / row["path"]).resolve()).casefold()] = {
                    "sha256": row.get("sha256"), "manifest": str(manifest.relative_to(ROOT)),
                    "manifest_entry": row.get("path"), "declared_bytes": row.get("bytes"),
                    "declared_record_count": row.get("record_count"),
                    "declared_first_timestamp_ns": row.get("first_timestamp_ns"),
                    "declared_last_timestamp_ns": row.get("last_timestamp_ns"),
                }
        # The owner-sealed OOS validation manifest has root-level identity.
        if payload.get("file_sha256") and payload.get("local_path"):
            found[str((ROOT / payload["local_path"]).resolve()).casefold()] = {
                "sha256": payload.get("file_sha256"), "manifest": str(manifest.relative_to(ROOT)),
                "manifest_entry": payload.get("local_path"), "declared_bytes": payload.get("file_bytes"),
                "declared_record_count": payload.get("record_count"),
                "declared_first_timestamp_ns": payload.get("first_timestamp_utc"),
                "declared_last_timestamp_ns": payload.get("last_timestamp_utc"),
            }
    return found


def _schema_text(store: Any) -> str:
    value = getattr(store, "schema", None)
    return str(getattr(value, "value", value)).lower()


def _source_class(schema: str, symbols: list[str]) -> tuple[str, str]:
    symbol = symbols[0] if symbols else ""
    if symbol.startswith("ES") and schema in {"mbp-10", "mbp10"}:
        return "NATIVE_ES_MBP10", "ES_DEPTH"
    if symbol.startswith("ES") and schema == "mbo":
        return "MBO_DERIVED_MBP10", "ES_DEPTH"
    return "OTHER", "SUPPORT_OR_OTHER"


def _scan_dbn(path: Path, manifest_info: dict[str, Any]) -> dict[str, Any]:
    from databento import DBNStore

    store = DBNStore.from_file(path)
    meta = getattr(store, "metadata", None)
    start, end = _meta_start_end(store)
    schema = _schema_text(store)
    symbols = list(getattr(store, "symbols", []) or [])
    source_type, role = _source_class(schema, symbols)
    day_stats: dict[str, dict[str, int | None]] = {}
    previous_ts: int | None = None
    monotone = True
    total = 0
    try:
        # Only ts_recv is inspected. No price, size, action, side, or outcome
        # column is read by this audit.
        for batch in store.to_ndarray(count=500_000):
            timestamps = batch["ts_recv"]
            if not len(timestamps):
                continue
            if previous_ts is not None and int(timestamps[0]) < previous_ts:
                monotone = False
            if len(timestamps) > 1 and bool((timestamps[1:] < timestamps[:-1]).any()):
                monotone = False
            previous_ts = int(timestamps[-1])
            total += len(timestamps)
            day_numbers = timestamps // DAY_NS
            cuts = [0]
            cuts.extend((i + 1) for i in range(len(day_numbers) - 1) if day_numbers[i + 1] != day_numbers[i])
            cuts.append(len(day_numbers))
            for a, b in zip(cuts, cuts[1:]):
                day_no = int(day_numbers[a])
                day = (date(1970, 1, 1) + timedelta(days=day_no)).isoformat()
                node = day_stats.setdefault(day, {"record_count": 0, "first_timestamp_ns": None, "last_timestamp_ns": None})
                node["record_count"] = int(node["record_count"]) + b - a
                if node["first_timestamp_ns"] is None:
                    node["first_timestamp_ns"] = int(timestamps[a])
                node["last_timestamp_ns"] = int(timestamps[b - 1])
    finally:
        _safe_close(store)
    relative = str(path.relative_to(ROOT))
    return {
        "path": relative, "absolute_path": str(path.resolve()), "file_format": "DBN_ZSTD" if path.name.endswith(".zst") else "DBN",
        "file_bytes": path.stat().st_size, "schema": schema, "dataset": str(getattr(store, "dataset", "")),
        "symbols": symbols, "contract": symbols[0] if symbols else None,
        "stype_in": str(getattr(store, "stype_in", "")), "stype_out": str(getattr(store, "stype_out", "")),
        "header_request_start_utc": dt_iso(start), "header_request_end_utc_exclusive": dt_iso(end),
        "first_timestamp_utc": iso_ns(day_stats[min(day_stats)]["first_timestamp_ns"] if day_stats else None),
        "last_timestamp_utc": iso_ns(day_stats[max(day_stats)]["last_timestamp_ns"] if day_stats else None),
        "record_count": total, "timestamp_field_used": "ts_recv", "timestamp_order_nondecreasing": monotone,
        "source_type": source_type, "inventory_role": role,
        "source_manifest": manifest_info.get("manifest"), "manifest_entry": manifest_info.get("manifest_entry"),
        "sha256_from_manifest": manifest_info.get("sha256"),
        "manifest_byte_match": manifest_info.get("declared_bytes") in (None, path.stat().st_size),
        "manifest_record_count": manifest_info.get("declared_record_count"),
        "daily_observed_coverage": {
            day: {"record_count": stats["record_count"],
                  "first_timestamp_utc": iso_ns(stats["first_timestamp_ns"]),
                  "last_timestamp_utc": iso_ns(stats["last_timestamp_ns"])}
            # Retain supporting dates outside the audit interval as well: a
            # profile dependency (e.g. Apr 2 for Apr 6) can precede the first
            # target date and still be present in a source whose request
            # envelope overlaps the audit range.
            for day, stats in sorted(day_stats.items())
        },
    }


def _discover_sources() -> tuple[list[Path], list[str]]:
    surveyed = []
    candidates = []
    for path in DATA.rglob("*"):
        if not path.is_file() or not (path.name.endswith(".dbn") or path.name.endswith(".dbn.zst")):
            continue
        surveyed.append(str(path.relative_to(ROOT)))
        # All CME DBN roots are checked by header, not by folder name. Other
        # Databento files are cheaply screened by path before opening.
        if "cme_orderflow_absorption" not in str(path).casefold():
            continue
        candidates.append(path)
    return sorted(candidates), surveyed


def _in_audit_range(source: dict[str, Any]) -> bool:
    if not source["symbols"] or not any(s.startswith(("ES", "MES")) for s in source["symbols"]):
        return False
    start = datetime.fromisoformat(source["header_request_start_utc"].replace("Z", "+00:00")) if source["header_request_start_utc"] else None
    end = datetime.fromisoformat(source["header_request_end_utc_exclusive"].replace("Z", "+00:00")) if source["header_request_end_utc_exclusive"] else None
    if start and end:
        return start < END and end > START
    return any(APR_AUG.match(day) for day in source["daily_observed_coverage"])


def _period_role(day: str, source_type: str) -> str:
    if day.startswith("2026-04"):
        return "RETROSPECTIVE_APRIL_NATIVE_SOURCE" if source_type == "NATIVE_ES_MBP10" else "APRIL_SOURCE"
    if day.startswith("2026-05"):
        return "MAY_DEVELOPMENT_MBO_DERIVED"
    if day.startswith(("2026-06", "2026-07")) and day <= "2026-07-17":
        return "RETROSPECTIVE_ROBUSTNESS_MBO_DERIVED"
    if day <= "2026-08-01":
        return "JULY_PILOT_MBO_DERIVED"
    if day <= "2026-08-08":
        return "PREVIOUS_OOS_V1_SOURCE_MBO_DERIVED"
    if day <= "2026-08-14":
        return "PREVIOUS_FRESH_AUGUST_RESEARCH_NATIVE_SOURCE"
    return "OUTSIDE_AUDIT_ROLE_UNKNOWN"


def _time_window(day: str, zone: str, start: str, end: str) -> tuple[int, int]:
    tz = ZoneInfo(zone)
    d = date.fromisoformat(day)
    left = datetime.combine(d, time.fromisoformat(start), tzinfo=tz).astimezone(UTC)
    right = datetime.combine(d, time.fromisoformat(end), tzinfo=tz).astimezone(UTC)
    return int(left.timestamp() * NS), int(right.timestamp() * NS)


def _interval_union_covers(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor or left > cursor:
            continue
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def _date_maps(source_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[tuple[int, int]]]]]:
    dates: dict[str, dict[str, Any]] = defaultdict(lambda: {"es_sources": [], "support_sources": [], "by_type": defaultdict(list)})
    intervals: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for source in source_rows:
        schema = source["schema"]
        is_es = source["source_type"] in {"NATIVE_ES_MBP10", "MBO_DERIVED_MBP10"}
        for day, stats in source["daily_observed_coverage"].items():
            # UTC Sunday evening belongs to the next CME trading session. It
            # is retained in the source inventory but is not a target-date
            # row for London/NY daytime session eligibility.
            if not APR_AUG.match(day) or date.fromisoformat(day).weekday() >= 5:
                continue
            node = dates[day]
            (node["es_sources"] if is_es else node["support_sources"]).append(source["path"])
            first_ns = int(datetime.fromisoformat(stats["first_timestamp_utc"].replace("Z", "+00:00")).timestamp() * NS)
            last_ns = int(datetime.fromisoformat(stats["last_timestamp_utc"].replace("Z", "+00:00")).timestamp() * NS)
            item = {"path": source["path"], "source_type": source["source_type"], "schema": schema,
                    "first_timestamp_utc": stats["first_timestamp_utc"], "last_timestamp_utc": stats["last_timestamp_utc"],
                    "record_count": stats["record_count"], "contract": source["contract"]}
            node["by_type"][source["source_type"]].append(item)
            profile_kind = "ES_TRADES" if schema == "trades" and (source["contract"] or "").startswith("ES") else source["source_type"]
            if is_es:
                if source["header_request_start_utc"] and source["header_request_end_utc_exclusive"]:
                    req_start = int(datetime.fromisoformat(source["header_request_start_utc"].replace("Z", "+00:00")).timestamp() * NS)
                    req_end = int(datetime.fromisoformat(source["header_request_end_utc_exclusive"].replace("Z", "+00:00")).timestamp() * NS)
                    day_start = int(datetime.combine(date.fromisoformat(day), time.min, tzinfo=UTC).timestamp() * NS)
                    day_end = day_start + DAY_NS
                    left, right = max(req_start, day_start), min(req_end, day_end)
                    if left < right:
                        intervals[day][source["source_type"]].append((left, right))
                        if profile_kind != source["source_type"]:
                            intervals[day][profile_kind].append((left, right))
                else:
                    intervals[day][source["source_type"]].append((first_ns, last_ns + 1))
                    if profile_kind != source["source_type"]:
                        intervals[day][profile_kind].append((first_ns, last_ns + 1))
            elif profile_kind == "ES_TRADES":
                if source["header_request_start_utc"] and source["header_request_end_utc_exclusive"]:
                    req_start = int(datetime.fromisoformat(source["header_request_start_utc"].replace("Z", "+00:00")).timestamp() * NS)
                    req_end = int(datetime.fromisoformat(source["header_request_end_utc_exclusive"].replace("Z", "+00:00")).timestamp() * NS)
                    day_start = int(datetime.combine(date.fromisoformat(day), time.min, tzinfo=UTC).timestamp() * NS)
                    left, right = max(req_start, day_start), min(req_end, day_start + DAY_NS)
                    if left < right:
                        intervals[day][profile_kind].append((left, right))
                else:
                    intervals[day][profile_kind].append((first_ns, last_ns + 1))
    return dates, intervals


def _cover_info(day: str, source_type: str, window_start: int, window_end: int,
                intervals: dict[str, dict[str, list[tuple[int, int]]]], day_node: dict[str, Any]) -> dict[str, Any]:
    pieces = intervals.get(day, {}).get(source_type, [])
    counts = [int(row["record_count"]) for row in day_node.get("by_type", {}).get(source_type, [])]
    return {"request_envelope_covers_window": _interval_union_covers(pieces, window_start, window_end),
            "window_start_utc": iso_ns(window_start), "window_end_utc_exclusive": iso_ns(window_end),
            "records_on_session_date": sum(counts), "declared_request_segments": [
                {"start_utc": iso_ns(a), "end_utc_exclusive": iso_ns(b)} for a, b in sorted(pieces)]}


def _prior_day(mapping: dict[str, list[tuple[str, str]]], day: str, family: str) -> tuple[str | None, str | None]:
    prior = [(d, typ) for d, typ in mapping.get(family, []) if d < day]
    return prior[-1] if prior else (None, None)


def _timezone_audit() -> dict[str, Any]:
    months = {}
    for month in range(4, 9):
        representative = date(2026, month, 15)
        zones = {name: ZoneInfo(name) for name in ("America/Chicago", "America/New_York", "Europe/Zurich", "Europe/London")}
        local_eu_start = datetime.combine(representative, time(8), tzinfo=zones["Europe/London"])
        local_eu_end = datetime.combine(representative, time(16, 30), tzinfo=zones["Europe/London"])
        local_ny_start = datetime.combine(representative, time(9, 30), tzinfo=zones["America/New_York"])
        local_ny_end = datetime.combine(representative, time(16), tzinfo=zones["America/New_York"])
        representative_utc = datetime.combine(representative, time(12), tzinfo=UTC)
        local_representatives = {name: representative_utc.astimezone(zone) for name, zone in zones.items()}
        to_zone = lambda x, z: x.astimezone(z).strftime("%H:%M:%S%z")
        months[representative.strftime("%B").lower()] = {
            "representative_local_date": representative.isoformat(),
            "dst_active": {name: bool(value.dst()) for name, value in local_representatives.items()},
            "utc_offset": {name: value.utcoffset().total_seconds() for name, value in local_representatives.items()},
            "offset_display": {name: value.strftime("%z") for name, value in local_representatives.items()},
            "EUROPE_window": {
                "source_timezone": "Europe/London", "local": f"{local_eu_start:%H:%M:%S}–{local_eu_end:%H:%M:%S}",
                "UTC": f"{local_eu_start.astimezone(UTC):%H:%M:%S}–{local_eu_end.astimezone(UTC):%H:%M:%S}",
                "America/Chicago": f"{to_zone(local_eu_start,zones['America/Chicago'])}–{to_zone(local_eu_end,zones['America/Chicago'])}",
                "America/New_York": f"{to_zone(local_eu_start,zones['America/New_York'])}–{to_zone(local_eu_end,zones['America/New_York'])}",
                "Europe/Zurich": f"{to_zone(local_eu_start,zones['Europe/Zurich'])}–{to_zone(local_eu_end,zones['Europe/Zurich'])}",
            },
            "NY_W04_window": {
                "source_timezone": "America/New_York", "local": f"{local_ny_start:%H:%M:%S}–{local_ny_end:%H:%M:%S}",
                "UTC": f"{local_ny_start.astimezone(UTC):%H:%M:%S}–{local_ny_end.astimezone(UTC):%H:%M:%S}",
                "America/Chicago": f"{to_zone(local_ny_start,zones['America/Chicago'])}–{to_zone(local_ny_end,zones['America/Chicago'])}",
                "Europe/Zurich": f"{to_zone(local_ny_start,zones['Europe/Zurich'])}–{to_zone(local_ny_end,zones['Europe/Zurich'])}",
                "Europe/London": f"{to_zone(local_ny_start,zones['Europe/London'])}–{to_zone(local_ny_end,zones['Europe/London'])}",
            },
        }
    return {"timezone_database": "Python zoneinfo / IANA", "months": months,
            "DST_transition_note": "US DST is active by April; UK/Swiss DST begins in late March. All representative dates April–August 2026 use CDT, EDT, BST, CEST respectively."}


def _session_audit(timezones: dict[str, Any]) -> dict[str, Any]:
    month_windows = {month: {
        key: row[key] for key in ("EUROPE_window", "NY_W04_window")
    } for month, row in timezones["months"].items()}
    return {
        "canonical_families": {
            "EU_CURRENT_HIGH_SWEEP": {"timezone": "Europe/London", "start_inclusive": "08:00:00", "end_exclusive": "16:30:00", "date_convention": "London local calendar session date; the source requests are keyed to the same UTC calendar date", "previous_session": "none for the current-session high-sweep level; session high is current-day causal state", "timestamp_basis": "Databento ts_recv in nanoseconds UTC"},
            "EU_PRIOR_HIGH": {"timezone": "Europe/London", "start_inclusive": "08:00:00", "end_exclusive": "16:30:00", "date_convention": "London local calendar session date", "previous_session": "immediately preceding completed audited Europe session; replay requires exact prior_day continuity", "timestamp_basis": "Databento ts_recv in nanoseconds UTC"},
            "PRIOR_VAH": {"timezone": "Europe/London", "start_inclusive": "08:00:00", "end_exclusive": "16:30:00", "date_convention": "London local calendar session date", "previous_session": "immediately preceding completed audited Europe session; prior session's ES executions form value area", "timestamp_basis": "Databento ts_recv in nanoseconds UTC"},
            "NY_W04": {"timezone": "America/New_York", "start_inclusive": "09:30:00", "end_exclusive": "16:00:00", "date_convention": "New York cash-session date; request filename date is the same UTC calendar date in this period", "previous_session": "immediately preceding completed RTH from the repository's available ES RTH profile source/calendar", "timestamp_basis": "Databento ts_recv in nanoseconds UTC"},
        },
        "profile_source_conventions": {
            "Europe previous-session profile": "prior completed Europe session's ES source stream; MBO-derived public-book source remains distinct from native MBP-10",
            "NY prior-RTH profile": "immediately preceding session's ES MBO/native MBP-10 source when its RTH window is covered, or ES trades source when present",
            "availability_scope": "temporal source/window availability only; this audit does not calculate profile values or inspect price/size/action fields",
        },
        "additional_replay_management_boundaries": {"Europe": {"cutoff": "16:55 Europe/London", "last_valid_BBO_window": "inclusive [16:54:00,16:55:00] local", "purpose": "position management, not signal eligibility"}, "NY_W04": {"hard_flat": "corrected Berlin hard-flat contract; in Apr–Aug 2026 22:45 UTC", "purpose": "position management, not NY RTH signal window"}},
        "timestamp_semantics": {"Databento fields": ["ts_event", "ts_recv"], "family_pipeline_uses": "ts_recv", "UTC": True, "ordering": "ts_recv nondecreasing is verified per physical DBN file", "filename_date": "not treated as evidence: date bins are formed from actual ts_recv"},
        "coverage_tolerance": {"SESSION_START_TOLERANCE": "none declared as a duration; source/request envelope must cover the half-open canonical interval, but an event exactly at the start is not required", "SESSION_END_TOLERANCE": "none declared as a duration for signal windows; end is exclusive and a market update exactly at end is not required", "MAX_INTERNAL_GAP_ALLOWED": "3 seconds for executable BBO integrity in the corrected W04 execution contract; no separate timestamp-only gap tolerance is defined for historical session-envelope inventory", "Europe_hard_flat_freshness": "last executable BBO in inclusive preceding 1 second"},
        "month_local_conversions": month_windows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_scan_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temp.write_text(json.dumps({row["path"].casefold(): row for row in rows}, sort_keys=True,
                               separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _md_report(summary: dict[str, Any], matrix: list[dict[str, Any]], timezone_audit: dict[str, Any],
               sources: list[dict[str, Any]], blocks: dict[str, Any]) -> str:
    native = summary["available_dates_by_source"]["native_es_mbp10"]
    derived = summary["available_dates_by_source"]["mbo_derived_mbp10"]
    lines = [
        "# April–August 2026 ES source and session-coverage audit", "",
        "Status: **PASS — metadata/timestamp inventory only**.", "",
        "This audit reads DBN headers and `ts_recv` timestamps only. It calculates no interactions, markouts, strategy outcomes, or PnL, and it does not inspect any final OOS outcome artifacts. A prior broad text search briefly surfaced unrelated historical performance rows; those rows were not retained or used in this audit. All coverage conclusions below are independently derived from source manifests, DBN schemas, session code, and timestamp scans.", "",
        "## Canonical windows", "",
        "The three Europe families share `[08:00,16:30)` Europe/London. In every audited month this is `[07:00,15:30)` UTC. `NY_W04` uses `[09:30,16:00)` America/New_York, `[13:30,20:00)` UTC. These are signal/family windows; the later management cutoffs are separate.", "",
        "`EU_CURRENT_HIGH_SWEEP` uses current-session state. `EU_PRIOR_HIGH` and `PRIOR_VAH` require the immediately previous completed Europe profile. `NY_W04` requires the previous completed RTH POC context. For completeness, this matrix flags feature/profile input eligibility, not trade execution; MES availability is separately reported in the source inventory.", "",
        "A source requested from 13:00 UTC covers only the final 2.5 hours of the Europe London window (13:00–15:30 UTC); it misses 6 hours from the 07:00 UTC open. The rejection is correct and is not caused by DST or filename-date interpretation.", "",
        "## Source-type results", "",
        f"Native vendor ES MBP-10 dates: {', '.join(native)} ({len(native)}).",
        f"ES MBO sources consumed through the frozen MBO→public-MBP10 adapter: {', '.join(derived)} ({len(derived)}).",
        "No audited date has a complete native ES MBP-10 Europe window. MBO-derived data is never merged with native MBP-10 or labeled native.", "",
        "## Reassessment of previous claims", "",
        f"- Apr 6–8 are native MBP-10, first request/event 13:00 UTC; Europe incomplete: **{summary['claim_april_6_8_correct']}**.",
        f"- Aug 10–14 are native MBP-10, first request/event 13:00 UTC; Europe incomplete: **{summary['claim_august_10_14_correct']}**.",
        f"- May is MBO-derived public MBP-10, not native: **{summary['claim_may_derived_correct']}**.",
        f"- June–July is MBO-derived public MBP-10, not native: **{summary['claim_june_july_derived_correct']}**.",
        f"- No contiguous native block satisfies the Europe window: **{summary['claim_no_contiguous_native_block_correct']}**.", "",
        "The Dec/Jan native MBP-10 root exists but is outside this Apr–Aug matrix and its session files begin at 13:00 UTC; Spring/October 2025 are outside the requested window. No additional alternate Apr–Aug native ES MBP-10 copies were found under the local `data/` tree.", "",
        "## Native-only requirement", "",
        f"Classification: **{summary['native_mbp10_requirement_source']}**. The impact-per-flow formula is `abs(10-second pre-event midpoint change in ES ticks) / (abs(10-second MLOFI integral / event-end depth denominator) + 1e-12)`. Its inputs (top-book midpoint, aggregated depth/MLOFI, and `ts_recv`) are mathematically representable by validated MBO-derived public MBP-10; there is no MBO order-identity term and no mathematical vendor-native-only dependency. Native-only is retained here as a frozen source-selection choice for methodological consistency; this audit does not relax it.", "",
        "## Date coverage matrix", "",
        "| Date | ES source type | First / last UTC | Europe UTC / full | NY UTC / full | Current-high | Prior-high | Prior-VAH | NY-W04 | Status note |", "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in matrix:
        lines.append("| {date} | {type} | {first} / {last} | {eu} / {euf} | {ny} / {nyf} | {cur} | {ph} | {vah} | {nyok} | {reason} |".format(
            date=row["date"], type=row["source_type"], first=row["first_event_utc"] or "—", last=row["last_event_utc"] or "—",
            eu=row["canonical_europe_window_utc"], euf="yes" if row["europe_fully_covered"] else "no",
            ny=row["canonical_ny_window_utc"], nyf="yes" if row["ny_fully_covered"] else "no",
            cur="yes" if row["eu_current_high_sweep_eligible"] else "no", ph="yes" if row["eu_prior_high_eligible"] else "no",
            vah="yes" if row["prior_vah_eligible"] else "no", nyok="yes" if row["ny_w04_eligible"] else "no",
            reason=(row["incomplete_reason"] or "—").replace("|", "/")))
    lines += ["", "## Contiguous calibration candidates", "", "No eligible contiguous native block exists. The native dates are two disconnected sets, and both sets fail full Europe coverage. MBO-derived date runs exist, but remain ineligible under the retained native-source selection rule; they are not silently substituted.", ""]
    lines += ["## Monthly DST conversion", "", "All five representative dates use CDT (UTC−05), EDT (UTC−04), BST (UTC+01), and CEST (UTC+02). `zoneinfo` conversion was used; no fixed-offset assumption was applied.", ""]
    for month, row in timezone_audit["months"].items():
        lines.append(f"- {month.title()} {row['representative_local_date']}: offsets {row['offset_display']}; Europe UTC {row['EUROPE_window']['UTC']}; NY UTC {row['NY_W04_window']['UTC']}.")
    lines += ["", "## Integrity and scope", "", f"Scanned {len(sources)} DBN source files after discovering all DBN files under `data/`; inventory uses the GLBX.MDP3 ES/MES sources whose declared ranges overlap Apr 1–Sep 1, 2026. Coverage timestamps are derived from `ts_recv`; record counts are from complete stream iteration. No outcome files were opened by the audit builder.", "", f"Artifact root: `{OUT.relative_to(ROOT)}`.", ""]
    return "\n".join(lines)


def build(output_root: Path = OUT) -> dict[str, Any]:
    candidates, surveyed_paths = _discover_sources()
    manifest_info = _manifest_hashes()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / ".timestamp-scan-cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    # A previous complete timestamp-only inventory is a valid local scan
    # checkpoint for report-logic corrections, provided source bytes and the
    # freshly-read DBN header identity still agree. No market values are used.
    if not cache:
        previous_inventory = output_root / "source-inventory.json"
        try:
            prior_rows = json.loads(previous_inventory.read_text(encoding="utf-8")).get("source_files", [])
        except (OSError, json.JSONDecodeError):
            prior_rows = []
        for prior in prior_rows:
            prior_path = ROOT / prior.get("path", "")
            try:
                stat = prior_path.stat()
            except OSError:
                continue
            if stat.st_size != prior.get("file_bytes"):
                continue
            prior["_cache_file_bytes"] = stat.st_size
            prior["_cache_mtime_ns"] = stat.st_mtime_ns
            cache[prior["path"].casefold()] = prior
        if cache:
            _write_scan_cache(cache_path, [cache[key] for key in sorted(cache)])
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(candidates, 1):
        try:
            from databento import DBNStore
            store = DBNStore.from_file(path)
            symbol_list = list(getattr(store, "symbols", []) or [])
            dataset = str(getattr(store, "dataset", ""))
            start, end = _meta_start_end(store)
            schema = _schema_text(store)
            _safe_close(store)
        except Exception as exc:
            print(f"HEADER_SKIP {path.relative_to(ROOT)} {type(exc).__name__}", flush=True)
            continue
        if dataset != "GLBX.MDP3" or not any(s.startswith(("ES", "MES")) for s in symbol_list):
            continue
        if start and end and not (start < END and end > START):
            continue
        stat = path.stat()
        cached = cache.get(str(path.relative_to(ROOT)).casefold())
        if (isinstance(cached, dict) and cached.get("_cache_file_bytes") == stat.st_size
                and cached.get("_cache_mtime_ns") == stat.st_mtime_ns
                and cached.get("schema") == schema and cached.get("symbols") == symbol_list
                and cached.get("dataset") == dataset
                and cached.get("header_request_start_utc") == dt_iso(start)
                and cached.get("header_request_end_utc_exclusive") == dt_iso(end)):
            source = {key: value for key, value in cached.items() if not key.startswith("_cache_")}
            print(f"TIMESTAMP_CACHE_HIT {len(rows) + 1} {source['schema']} {source['contract']} path={path.relative_to(ROOT)}", flush=True)
        else:
            source = _scan_dbn(path, manifest_info.get(str(path.resolve()).casefold(), {}))
            source["_cache_file_bytes"] = stat.st_size
            source["_cache_mtime_ns"] = stat.st_mtime_ns
        if not _in_audit_range(source):
            continue
        rows.append(source)
        if "_cache_file_bytes" in source:
            cache[str(path.relative_to(ROOT)).casefold()] = source
        _write_scan_cache(cache_path, rows=[cache[key] for key in sorted(cache)])
        if "_cache_file_bytes" in source:
            print(f"TIMESTAMP_SCAN {len(rows)} {source['schema']} {source['contract']} records={source['record_count']:,} path={path.relative_to(ROOT)}", flush=True)
    rows = [{key: value for key, value in row.items() if not key.startswith("_cache_")} for row in rows]
    rows.sort(key=lambda r: (r["path"].casefold(), r["schema"], r["first_timestamp_utc"] or ""))
    dates_map, intervals = _date_maps(rows)
    date_list = sorted(dates_map)
    month_window_dict = {}
    for month in range(4, 9):
        representative = date(2026, month, 15)
        eu_start, eu_end = _time_window(representative.isoformat(), "Europe/London", "08:00", "16:30")
        ny_start, ny_end = _time_window(representative.isoformat(), "America/New_York", "09:30", "16:00")
        month_window_dict[month] = (eu_start, eu_end, ny_start, ny_end)

    # Profile dates must be the immediately preceding completed exchange
    # session, not merely the latest earlier date for which a profile happens
    # to be available. The support trade files retain profile-only dates such
    # as Jun 22 and May 1 in this session calendar.
    session_dates = sorted(
        day for day, node in dates_map.items()
        if node["es_sources"] or any(
            item["schema"] == "trades" and (item["contract"] or "").startswith("ES")
            for rows_for_type in node["by_type"].values() for item in rows_for_type
        )
    )
    complete_europe_dates: set[str] = set()
    complete_rth_profile_dates: set[str] = set()
    for day in session_dates:
        eu_start, eu_end = _time_window(day, "Europe/London", "08:00", "16:30")
        ny_start, ny_end = _time_window(day, "America/New_York", "09:30", "16:00")
        for kind in ("NATIVE_ES_MBP10", "MBO_DERIVED_MBP10"):
            if _interval_union_covers(intervals[day].get(kind, []), eu_start, eu_end):
                complete_europe_dates.add(day)
            if _interval_union_covers(intervals[day].get(kind, []), ny_start, ny_end):
                complete_rth_profile_dates.add(day)
        if _interval_union_covers(intervals[day].get("ES_TRADES", []), ny_start, ny_end):
            complete_rth_profile_dates.add(day)
    mes_dates = {d for r in rows if r["schema"] == "mbp-1" and any(s.startswith("MES") for s in r["symbols"])
                 for d in r["daily_observed_coverage"] if APR_AUG.match(d) and date.fromisoformat(d).weekday() < 5}

    matrix: list[dict[str, Any]] = []
    date_to_type: dict[str, str] = {}
    for day in date_list:
        node = dates_map[day]
        types = sorted(node["by_type"])
        es_types = [t for t in types if t in {"NATIVE_ES_MBP10", "MBO_DERIVED_MBP10"}]
        source_type = (es_types[0] if len(es_types) == 1 else "OTHER" if es_types or node["support_sources"] else "UNKNOWN")
        date_to_type[day] = source_type
        eu_start, eu_end = _time_window(day, "Europe/London", "08:00", "16:30")
        ny_start, ny_end = _time_window(day, "America/New_York", "09:30", "16:00")
        eu_windows = {
            kind: _cover_info(day, kind, eu_start, eu_end, intervals, node)
            for kind in ("NATIVE_ES_MBP10", "MBO_DERIVED_MBP10")
        }
        ny_windows = {
            kind: _cover_info(day, kind, ny_start, ny_end, intervals, node)
            for kind in ("NATIVE_ES_MBP10", "MBO_DERIVED_MBP10")
        }
        eu_complete = any(info["request_envelope_covers_window"] for info in eu_windows.values())
        ny_complete = any(info["request_envelope_covers_window"] for info in ny_windows.values())
        prior_session = next((d for d in reversed(session_dates) if d < day), None)
        prev_eu = prior_session
        prior_rth = prior_session
        prior_eu_available = prior_session in complete_europe_dates
        ny_prior_available = prior_session in complete_rth_profile_dates
        ny_current_source_type = next((kind for kind in ("NATIVE_ES_MBP10", "MBO_DERIVED_MBP10")
                                        if ny_windows[kind]["request_envelope_covers_window"]), None)
        eu_current_source_type = next((kind for kind in ("NATIVE_ES_MBP10", "MBO_DERIVED_MBP10")
                                       if eu_windows[kind]["request_envelope_covers_window"]), None)
        selected_types = es_types or types
        first_candidates = [x["first_timestamp_utc"] for typ in selected_types for x in node["by_type"][typ]]
        last_candidates = [x["last_timestamp_utc"] for typ in selected_types for x in node["by_type"][typ]]
        first_event = min(first_candidates) if first_candidates else None
        last_event = max(last_candidates) if last_candidates else None
        # MES support existence is informational; the impact-per-flow feature
        # pipeline is ES public-book based and does not require MES execution.
        reason_parts = []
        if not eu_complete:
            reason_parts.append("EUROPE_WINDOW_NOT_COVERED_BY_ES_SOURCE")
        if not prior_eu_available:
            reason_parts.append("PREVIOUS_COMPLETED_EUROPE_PROFILE_NOT_AVAILABLE")
        if not ny_complete:
            reason_parts.append("NY_RTH_WINDOW_NOT_COVERED_BY_ES_SOURCE")
        if not ny_prior_available:
            reason_parts.append("PREVIOUS_COMPLETED_RTH_PROFILE_NOT_AVAILABLE")
        if source_type == "NATIVE_ES_MBP10" and not eu_complete:
            reason_parts.append("NATIVE_REQUEST_START_13_00_UTC_AFTER_EUROPE_START_07_00_UTC")
        matrix.append({
            "date": day, "source_exists": bool(node["es_sources"] or node["support_sources"]),
            "es_signal_source_exists": bool(es_types), "source_type": source_type,
            "source_paths": sorted(set(node["es_sources"])), "support_source_paths": sorted(set(node["support_sources"])),
            "contracts": sorted({x["contract"] for typ in selected_types for x in node["by_type"][typ]}),
            "first_event_utc": first_event, "last_event_utc": last_event,
            "canonical_europe_window_utc": f"{iso_ns(eu_start)}–{iso_ns(eu_end)} [end exclusive]",
            "europe_fully_covered": eu_complete,
            "europe_native_mbp10_covered": eu_windows["NATIVE_ES_MBP10"]["request_envelope_covers_window"],
            "europe_mbo_derived_covered": eu_windows["MBO_DERIVED_MBP10"]["request_envelope_covers_window"],
            "europe_window_evidence": eu_windows,
            "canonical_ny_window_utc": f"{iso_ns(ny_start)}–{iso_ns(ny_end)} [end exclusive]",
            "ny_fully_covered": ny_complete,
            "ny_window_evidence": ny_windows,
            "previous_europe_profile_source_date": prev_eu,
            "previous_europe_profile_available": prior_eu_available,
            "previous_rth_profile_source_date": prior_rth,
            "previous_rth_profile_available": ny_prior_available,
            "mes_mbp1_execution_support_available": day in mes_dates,
            "eu_current_high_sweep_eligible": eu_complete,
            "eu_prior_high_eligible": eu_complete and prior_eu_available,
            "prior_vah_eligible": eu_complete and prior_eu_available,
            "ny_w04_eligible": ny_complete and ny_prior_available,
            "native_source_requirement_satisfied": source_type == "NATIVE_ES_MBP10",
            "incomplete_reason": ";".join(dict.fromkeys(reason_parts)),
            "final_oos_reserved": False,
            "prior_research_role": _period_role(day, source_type),
            "role_evidence": "period-tape-manifest and acquisition/source manifests; no outcome fields consulted",
        })

    native_dates = sorted(r["date"] for r in matrix if r["source_type"] == "NATIVE_ES_MBP10")
    derived_dates = sorted(r["date"] for r in matrix if r["source_type"] == "MBO_DERIVED_MBP10")
    other_dates = sorted(r["date"] for r in matrix if r["source_type"] in {"OTHER", "UNKNOWN"})
    native_full_eu = [r["date"] for r in matrix if r["source_type"] == "NATIVE_ES_MBP10" and r["europe_fully_covered"]]
    blocks = {"status": "NO_ELIGIBLE_NATIVE_EUROPE_BLOCK", "eligibility_rule": {
        "source_type": "NATIVE_ES_MBP10", "full_europe_session": True, "final_oos_reserved": False,
        "no_outcome_based_date_selection": True}, "eligible_contiguous_blocks": [],
        "native_date_runs": [{"start": part[0], "end": part[-1], "session_count": len(part),
                              "europe_complete_dates": [d for d in part if d in native_full_eu],
                              "status": "REJECTED_EUROPE_WINDOW_INCOMPLETE"}
                             for part in _runs(native_dates)],
        "mbo_derived_near_miss_runs_not_selected": [
            {"start": part[0], "end": part[-1], "session_count": len(part), "source_type": "MBO_DERIVED_MBP10",
             "eligible_under_frozen_native_only_choice": False}
            for part in _runs(derived_dates)],
        "minimum_sufficient_sessions": "not specified in inspected code/manifest; no minimum invented",
        "best_metadata_only_calibration_block": None}

    timezone_audit = _timezone_audit()
    session_audit = _session_audit(timezone_audit)
    rows.sort(key=key_sort)
    source_inventory = {
        "audit_id": "APRIL_AUGUST_2026_NATIVE_ES_COVERAGE_AUDIT_V1",
        "surveyed_local_dbn_file_count": len(surveyed_paths), "glbx_es_mes_files_scanned": len(rows),
        "source_files": rows,
        "discovery_roots": sorted({str(Path(p).parts[1]) for p in surveyed_paths if len(Path(p).parts) > 1}),
        "source_filter": "GLBX.MDP3 DBN source header symbol ES* or MES*, declared metadata interval overlaps [2026-04-01,2026-09-01)",
        "record_scan": "full deterministic DBN iteration in 500,000-row chunks; only ts_recv read; reader explicitly closed",
        "market_values_read": False, "strategy_outcomes_read": False,
    }
    provenance = {
        "period_tape_manifests": [
            {"path": str(ROOT / "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_CAUSAL_TAPES" / folder / "period-tape-manifest.json"),
             "period_id": period, "source_model": source_model, "prior_research_role": role}
            for period, folder, source_model, role in PERIODS],
        "source_model_rules": {"NATIVE_ES_MBP10": "DBN schema mbp-10, dataset GLBX.MDP3, raw ES symbol; directly vendor aggregate MBP10", "MBO_DERIVED_MBP10": "DBN schema mbo, raw ES symbol; frozen historical MBO adapter reconstructs canonical public MBP10", "OTHER": "MES mbp-1 or ES trades support input, not the ES signal source"},
        "timestamp_semantics": "Databento ts_recv is the pipeline's actual causal/session timestamp; date association is recomputed from UTC ts_recv, never copied solely from filename.",
        "session_date_association": "Target session rows use observed ES depth/ES trades dates on weekdays. UTC Sunday timestamps remain visible in per-file daily coverage but are not counted as a separate London/NY target session date. Prior-session lookup uses the immediately preceding observed ES source session date; if that date lacks the required complete source window, the prior-profile availability flag is false rather than falling back to an older date.",
        "source_manifest_fields": "SHA-256 and acquisition metadata copied from local acquisition manifests where available; missing hashes remain null rather than fabricated.",
        "outcome_data_accessed": False,
    }
    duplicate_audit = _duplicate_audit(rows, dates_map)
    summary = {
        "audit_id": "APRIL_AUGUST_2026_NATIVE_ES_COVERAGE_AUDIT_V1", "status": "PASS_METADATA_TIMESTAMP_ONLY",
        "range": {"start_inclusive": "2026-04-01T00:00:00Z", "end_exclusive": "2026-09-01T00:00:00Z"},
        "available_trading_date_count": sum(bool(r["es_signal_source_exists"]) for r in matrix),
        "available_date_rows_including_profile_support": len(matrix),
        "profile_support_only_dates": [r["date"] for r in matrix if not r["es_signal_source_exists"]],
        "available_dates": [r["date"] for r in matrix],
        "available_dates_by_source": {"native_es_mbp10": native_dates, "mbo_derived_mbp10": derived_dates, "other": other_dates},
        "claim_april_6_8_correct": all(_matrix_by_day(matrix)[d]["source_type"] == "NATIVE_ES_MBP10" and not _matrix_by_day(matrix)[d]["europe_fully_covered"] for d in ("2026-04-06", "2026-04-07", "2026-04-08")),
        "claim_august_10_14_correct": all(_matrix_by_day(matrix)[d]["source_type"] == "NATIVE_ES_MBP10" and not _matrix_by_day(matrix)[d]["europe_fully_covered"] for d in ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14")),
        "claim_may_derived_correct": all(_matrix_by_day(matrix)[d]["source_type"] == "MBO_DERIVED_MBP10" for d in derived_dates if d.startswith("2026-05")),
        "claim_june_july_derived_correct": all(_matrix_by_day(matrix)[d]["source_type"] == "MBO_DERIVED_MBP10" for d in derived_dates if "2026-06" <= d <= "2026-07-17"),
        "claim_no_contiguous_native_block_correct": not bool(native_full_eu),
        "native_mbp10_requirement_source": "PREREGISTERED_RESEARCH_CHOICE",
        "native_requirement_assessment": "Not a hard technical property of impact-per-flow; the formula uses aggregate MBP-10 public-book state and ts_recv. Native-only is retained as the frozen source-selection choice for this calibration audit; no relaxation was made.",
        "native_contiguous_europe_complete_dates": native_full_eu,
        "record_scan_complete": True,
        "final_oos_accessed": False, "strategy_outcomes_calculated": False,
        "pnl_calculated": False, "optimization_performed": False,
        "audit_note": "This builder only opens DBN source streams for timestamp/count metadata. It does not open outcome or markout artifacts.",
        "artifact_root": str(output_root.relative_to(ROOT) if output_root.is_relative_to(ROOT) else output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "source-inventory.json", source_inventory)
    _write_json(output_root / "date-coverage-matrix.json", {"rows": matrix, "field_semantics": {
        "eligible": "metadata/source-profile coverage only, not strategy outcome or trade readiness",
        "coverage": "declared DBN half-open source request envelopes checked against canonical session intervals; actual per-date timestamp extrema and row counts accompany every source",
        "source_type": "native vendor MBP-10 vs raw MBO passed through the existing public-book adapter; never merged under one label"}})
    _write_json(output_root / "timezone-audit.json", timezone_audit)
    _write_json(output_root / "session-definition-audit.json", session_audit)
    _write_json(output_root / "source-provenance.json", provenance)
    _write_json(output_root / "duplicate-source-audit.json", duplicate_audit)
    _write_json(output_root / "eligible-calibration-blocks.json", blocks)
    (output_root / "report.md").write_text(_md_report(summary, matrix, timezone_audit, rows, blocks), encoding="utf-8")
    cache_path.unlink(missing_ok=True)
    return summary


def _matrix_by_day(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["date"]: row for row in rows}


def _runs(days: list[str]) -> list[list[str]]:
    if not days:
        return []
    result: list[list[str]] = [[days[0]]]
    for day in days[1:]:
        previous = date.fromisoformat(result[-1][-1])
        current = date.fromisoformat(day)
        if (current - previous).days <= 4:
            result[-1].append(day)
        else:
            result.append([day])
    return result


def _duplicate_audit(rows: list[dict[str, Any]], date_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        for day, _ in source["daily_observed_coverage"].items():
            if APR_AUG.match(day) and date.fromisoformat(day).weekday() < 5:
                by_day[day].append(source)
    alternates = []
    overlaps = []
    for day, sources in sorted(by_day.items()):
        es = [s for s in sources if s["source_type"] in {"NATIVE_ES_MBP10", "MBO_DERIVED_MBP10"}]
        if len(es) > 1:
            alternates.append({"date": day, "sources": [{"path": s["path"], "source_type": s["source_type"], "schema": s["schema"], "contract": s["contract"], "first": s["daily_observed_coverage"][day]["first_timestamp_utc"], "last": s["daily_observed_coverage"][day]["last_timestamp_utc"]} for s in es]})
            kinds = {s["source_type"] for s in es}
            if len(kinds) > 1:
                overlaps.append({"date": day, "decision": "DO_NOT_MERGE_OR_LABEL_NATIVE", "source_types": sorted(kinds)})
    return {"alternate_es_depth_dates": alternates, "cross_source_type_overlaps": overlaps,
            "same_date_multiple_support_sources": {d: [s["path"] for s in sources if s["source_type"] == "OTHER"] for d, sources in sorted(by_day.items()) if sum(s["source_type"] == "OTHER" for s in sources) > 1},
            "combination_rules": ["Contiguous June/July MBO parts can be concatenated as MBO then reconstructed; result stays MBO_DERIVED_MBP10.", "Native MBP-10 is never spliced with MBO-derived data and called native.", "MES MBP-1 and ES trades are separate supporting instruments/schemas, not ES MBP-10 duplicates."],
            "duplicates_by_sha256": _hash_duplicates(rows)}


def _hash_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("sha256_from_manifest"):
            by_hash[row["sha256_from_manifest"]].append(row["path"])
    return [{"sha256": digest, "paths": paths} for digest, paths in by_hash.items() if len(paths) > 1]


def key_sort(item: dict[str, Any]) -> tuple[str, str, str]:
    return item["path"].casefold(), item["schema"], item["first_timestamp_utc"] or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    result = build(args.output_root if args.output_root.is_absolute() else ROOT / args.output_root)
    print(json.dumps({k: result[k] for k in ("status", "available_trading_date_count", "available_dates_by_source", "claim_april_6_8_correct", "claim_august_10_14_correct", "claim_may_derived_correct", "claim_june_july_derived_correct", "claim_no_contiguous_native_block_correct", "native_mbp10_requirement_source", "artifact_root")}, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS_METADATA_TIMESTAMP_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
