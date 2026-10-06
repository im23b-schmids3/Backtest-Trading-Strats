"""Build the local-only plan for the Sato missing overnight OHLCV patch."""
from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1h"
STYPE_IN = "raw_symbol"
ET = ZoneInfo("America/New_York")
EXPECTED_TOTAL_HOURS_SANITY = 103  # comparison only; never used to construct rows
STUDY_ID = "CMEOrderflow_SATO_ES_OVERNIGHT_OHLCV1H_QUOTE_PREP_V1"
DEFAULT_NATIVE_MANIFEST = Path("data/databento/mac-2025-native-es-mbp10-final-reduced/mac-2025-native-mbp-download-manifest.json")
DEFAULT_SATO_DIR = Path("research_runs/CMEOrderflow_SATO_ES_PRIOR_DAY_LIQUIDITY_SWEEP_RECLAIM_V1")
DEFAULT_OUTPUT_DIR = Path("research_runs") / STUDY_ID
ES_OUTRIGHT = re.compile(r"^ES[HMUZ][0-9]$")


class QuotePlanError(ValueError):
    """Raised when local source evidence cannot support a safe quote plan."""


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise QuotePlanError("naive datetime is not allowed")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuotePlanError(f"cannot read local evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QuotePlanError(f"expected JSON object in {path}")
    return value


def build_plan(
    repo_root: Path,
    *,
    native_manifest_path: Path = DEFAULT_NATIVE_MANIFEST,
    sato_dir: Path = DEFAULT_SATO_DIR,
) -> dict:
    """Derive exact missing bars from sealed local source evidence only."""
    root = repo_root.resolve()
    native_path = root / native_manifest_path
    sato_root = root / sato_dir
    native = _read_json(native_path)
    source_coverage = _read_json(sato_root / "source-coverage.json")
    run_manifest = _read_json(sato_root / "run-manifest.json")
    summary = _read_json(sato_root / "summary.json")

    if native.get("status") != "COMPLETE":
        raise QuotePlanError("native acquisition manifest is not complete")
    if source_coverage.get("dataset") != DATASET or source_coverage.get("schema") != "mbp-10":
        raise QuotePlanError("Sato coverage artifact is not for native ES GLBX.MDP3 mbp-10")
    if source_coverage.get("instrument") != "ES":
        raise QuotePlanError("Sato coverage artifact does not identify ES")
    if run_manifest.get("study_id") != "CMEOrderflow_SATO_ES_PRIOR_DAY_LIQUIDITY_SWEEP_RECLAIM_V1":
        raise QuotePlanError("unexpected Sato run-manifest identity")
    if summary.get("status") != "COMPLETE":
        raise QuotePlanError("Sato source study is not complete")

    dates = source_coverage.get("eligible_dates")
    target_dates = summary.get("target_dates")
    sessions = source_coverage.get("sessions")
    if not isinstance(dates, list) or dates != target_dates or len(dates) != 54:
        raise QuotePlanError("expected the exact 54 Sato target dates from source coverage")
    if sum(day.startswith(("2025-03", "2025-04")) for day in dates) != 35 or sum(day.startswith("2025-10") for day in dates) != 19:
        raise QuotePlanError("Sato target set does not contain exactly 35 Spring and 19 October dates")
    if not isinstance(sessions, list) or len(sessions) != len(dates):
        raise QuotePlanError("Sato session coverage is incomplete")
    current_hashes = run_manifest.get("source_sha256_by_date", {})
    native_requests = native.get("requests", {})
    by_date: dict[str, list[tuple[str, dict]]] = {}
    for request_id, request in native_requests.items():
        if request.get("category") in {"TRAIN", "VALIDATION"}:
            by_date.setdefault(request.get("session_date", ""), []).append((request_id, request))

    rows: list[dict] = []
    for session_date in dates:
        current = date.fromisoformat(session_date)
        matching = by_date.get(session_date, [])
        if len(matching) != 1:
            raise QuotePlanError(f"expected exactly one native ES current-date request for {session_date}; found {len(matching)}")
        request_id, native_request = matching[0]
        verification = native_request.get("verification", {})
        symbol = native_request.get("symbol")
        if native_request.get("status") != "VERIFIED":
            raise QuotePlanError(f"native source request is not verified for {session_date}")
        if native_request.get("schema") != "mbp-10" or verification.get("schema") != "mbp-10":
            raise QuotePlanError(f"native source schema mismatch for {session_date}")
        if verification.get("dataset") != DATASET or verification.get("symbols") != [symbol]:
            raise QuotePlanError(f"native source dataset/symbol metadata mismatch for {session_date}")
        if not isinstance(symbol, str) or not ES_OUTRIGHT.fullmatch(symbol):
            raise QuotePlanError(f"non-outright or missing ES raw symbol for {session_date}: {symbol!r}")
        if native_request.get("start") != f"{session_date}T00:00:00Z":
            raise QuotePlanError(f"native current-date source does not start at 00:00 UTC for {session_date}")
        if verification.get("first_timestamp_utc") != f"{session_date}T00:00:00.000000000Z":
            raise QuotePlanError(f"native source has a gap/overlap boundary before first record for {session_date}")
        if current_hashes.get(session_date) != native_request.get("sha256"):
            raise QuotePlanError(f"Sato run-manifest source hash does not bind native manifest for {session_date}")
        source_file = native_request.get("path")
        if not source_file:
            raise QuotePlanError(f"native source file path missing for {session_date}")
        absolute_source = root / "data/databento/mac-2025-native-es-mbp10-final-reduced" / source_file
        if not absolute_source.is_file():
            raise QuotePlanError(f"native source file is missing for {session_date}: {absolute_source}")

        evening_date = current - timedelta(days=1)
        overnight_start = datetime.combine(evening_date, time(18, 0), tzinfo=ET)
        patch_start = overnight_start.astimezone(timezone.utc)
        patch_end = datetime.combine(current, time(0, 0), tzinfo=timezone.utc)
        duration_seconds = int((patch_end - patch_start).total_seconds())
        if duration_seconds <= 0 or duration_seconds % 3600:
            raise QuotePlanError(f"patch is not a positive whole-hour interval for {session_date}")
        if patch_start.minute or patch_start.second or patch_end.minute or patch_end.second:
            raise QuotePlanError(f"patch boundaries are not hour-aligned for {session_date}")
        if patch_end != datetime.fromisoformat(native_request["start"].replace("Z", "+00:00")):
            raise QuotePlanError(f"patch end does not equal native source start for {session_date}")
        bar_starts = [
            _iso_utc(patch_start + timedelta(hours=offset))
            for offset in range(duration_seconds // 3600)
        ]
        rows.append({
            "CURRENT_RTH_DATE": session_date,
            "PRIOR_EVENING_ET_DATE": evening_date.isoformat(),
            "OVERNIGHT_START_ET": overnight_start.isoformat(timespec="seconds"),
            "PATCH_START_UTC": _iso_utc(patch_start),
            "PATCH_END_UTC": _iso_utc(patch_end),
            "MISSING_DURATION_MINUTES": duration_seconds // 60,
            "RAW_ES_SYMBOL": symbol,
            "SYMBOL_STATUS": "VERIFIED_LOCAL_NATIVE_RAW_SYMBOL",
            "STYPE_IN": STYPE_IN,
            "DATASET": DATASET,
            "SCHEMA": SCHEMA,
            "EXPECTED_HOURLY_BAR_COUNT": len(bar_starts),
            "EXPECTED_HOURLY_BAR_STARTS_UTC": bar_starts,
            "SOURCE_FILE": str(Path("data/databento/mac-2025-native-es-mbp10-final-reduced") / source_file),
            "SOURCE_FILE_START_UTC": native_request["start"],
            "SYMBOL_EVIDENCE_SOURCE": (
                f"{native_manifest_path} request {request_id} (VERIFIED symbol metadata), "
                f"{sato_dir}/run-manifest.json source_sha256_by_date[{session_date}]"
            ),
            "STATUS": "APPROVED_FOR_COST_QUOTE",
            "NOTES": "Local verified outright MBP-10 source binds symbol/hash; no remote resolution performed.",
            "SOURCE_FILE_SHA256": native_request["sha256"],
        })

    if len(rows) != 54:
        raise QuotePlanError(f"expected 54 requests, derived {len(rows)}")
    if any(row["DATASET"] != DATASET or row["SCHEMA"] != SCHEMA for row in rows):
        raise QuotePlanError("unexpected dataset/schema in derived plan")

    symbol_dates: dict[str, list[str]] = {}
    for row in rows:
        symbol_dates.setdefault(row["RAW_ES_SYMBOL"], []).append(row["CURRENT_RTH_DATE"])
    one_hour = [row["CURRENT_RTH_DATE"] for row in rows if row["EXPECTED_HOURLY_BAR_COUNT"] == 1]
    two_hour = [row["CURRENT_RTH_DATE"] for row in rows if row["EXPECTED_HOURLY_BAR_COUNT"] == 2]
    return {
        "status": "PREPARED_LOCAL_EVIDENCE_NO_QUOTE_EXECUTED",
        "study_id": STUDY_ID,
        "dataset": DATASET,
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "quote_endpoint": "Historical.metadata.get_cost",
        "download_endpoint_present": False,
        "data_downloaded": False,
        "api_key_used": False,
        "live_databento_call_executed": False,
        "quote_executed": False,
        "current_rth_dates": dates,
        "spring_dates": [day for day in dates if day.startswith("2025-03") or day.startswith("2025-04")],
        "october_dates": [day for day in dates if day.startswith("2025-10")],
        "one_hour_patch_sessions": one_hour,
        "two_hour_patch_sessions": two_hour,
        "total_missing_hours": sum(row["EXPECTED_HOURLY_BAR_COUNT"] for row in rows),
        "expected_total_hours_sanity": EXPECTED_TOTAL_HOURS_SANITY,
        "hour_count_matches_sanity_check": sum(row["EXPECTED_HOURLY_BAR_COUNT"] for row in rows) == EXPECTED_TOTAL_HOURS_SANITY,
        "distinct_raw_es_symbols": sorted(symbol_dates),
        "raw_symbols_and_date_ranges": {
            symbol: {"first_date": symbol_dates[symbol][0], "last_date": symbol_dates[symbol][-1], "dates": symbol_dates[symbol]}
            for symbol in sorted(symbol_dates)
        },
        "ambiguous_symbol_sessions": [],
        "source_manifest": str(native_manifest_path),
        "sato_source_coverage": str(sato_dir / "source-coverage.json"),
        "sato_run_manifest": str(sato_dir / "run-manifest.json"),
        "requests": rows,
        "cost_estimate_note": (
            "Historical.metadata.get_cost is an estimate in USD; billed cost is based on actual bytes delivered. "
            "All requested intervals are whole-hour multiples and therefore align to the provider's 10-minute quote granularity."
        ),
    }


def write_plan(plan: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "missing-overnight-hourly-quote-plan.json"
    csv_path = output_dir / "missing-overnight-hourly-quote-plan.csv"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = [
        "CURRENT_RTH_DATE", "PRIOR_EVENING_ET_DATE", "OVERNIGHT_START_ET", "PATCH_START_UTC",
        "PATCH_END_UTC", "MISSING_DURATION_MINUTES", "RAW_ES_SYMBOL", "STYPE_IN", "DATASET", "SCHEMA",
        "SYMBOL_STATUS",
        "EXPECTED_HOURLY_BAR_COUNT", "EXPECTED_HOURLY_BAR_STARTS_UTC", "SOURCE_FILE", "SOURCE_FILE_START_UTC",
        "SYMBOL_EVIDENCE_SOURCE", "STATUS", "NOTES", "SOURCE_FILE_SHA256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in plan["requests"]:
            writer.writerow({**row, "EXPECTED_HOURLY_BAR_STARTS_UTC": json.dumps(row["EXPECTED_HOURLY_BAR_STARTS_UTC"]),
                             "SYMBOL_EVIDENCE_SOURCE": row["SYMBOL_EVIDENCE_SOURCE"]})
    symbols = plan["raw_symbols_and_date_ranges"]
    one_hour_count = len(plan["one_hour_patch_sessions"])
    two_hour_count = len(plan["two_hour_patch_sessions"])
    report = [
        "# Sato missing overnight OHLCV-1h quote preparation",
        "",
        "Prepared from local manifests only. No Databento API call was made; no data was downloaded.",
        "",
        f"- Current RTH dates: {len(plan['current_rth_dates'])} (Spring {len(plan['spring_dates'])}, October {len(plan['october_dates'])})",
        f"- Quote requests: {len(plan['requests'])}, one per current RTH date",
        f"- Missing hourly bars: {plan['total_missing_hours']} ({one_hour_count} one-hour and {two_hour_count} two-hour sessions)",
        f"- Symbols: {', '.join('{} ({}–{})'.format(sym, info['first_date'], info['last_date']) for sym, info in symbols.items())}",
        "- Ambiguous symbol sessions: none; raw symbols are bound to the verified current-date native ES MBP-10 source and matching Sato source hash.",
        "- Quote endpoint: `Historical.metadata.get_cost` only.",
        "- No download endpoint/code is included in the quote script.",
        "",
        "OHLCV timestamps represent inclusive hourly-bar starts; all intervals are [start, end), ending exactly at the existing source start (00:00 UTC).",
        "The quote is a USD cost estimate; the eventual charge is based on bytes actually delivered.",
        "",
        f"Derived missing-hour total: {plan['total_missing_hours']}; expected sanity check: {plan['expected_total_hours_sanity']} "
        f"({'matches' if plan['hour_count_matches_sanity_check'] else 'differs'}).",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return json_path, csv_path, report_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    plan = build_plan(root)
    paths = write_plan(plan, root / DEFAULT_OUTPUT_DIR)
    print("\n".join(str(path.relative_to(root)) for path in paths))
