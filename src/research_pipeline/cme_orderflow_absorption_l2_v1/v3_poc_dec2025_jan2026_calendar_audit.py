"""DST-aware, outcome-free calendar audit for the frozen Dec/Jan L2 V3 block."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from .v3_poc_dec2025_jan2026_replay import (
    DATA_ROOT, EVIDENCE_LABEL, EXPECTED_SESSION_COUNT, V3_CONTRACT_SHA256,
    verify_acquisition_manifest,
)
from .v3_poc_only import STRATEGY_ID, v3_contract_sha256


AUDIT_JSON = Path("research_runs/L2_V3_DEC2025_JAN2026_CALENDAR_AUDIT/calendar-audit.json")
SUPPLEMENT_QUOTE_JSON = AUDIT_JSON.with_name("supplement-quote.json")
AUDIT_REPORT = Path("docs/research_pipeline/cme_orderflow_absorption_v1/l2-v3-dec2025-jan2026-dst-calendar-audit.md")
CLASSIFICATION = "EXECUTION_CALENDAR_DST_CLARIFICATION"
NEW_YORK = ZoneInfo("America/New_York")
CHICAGO = ZoneInfo("America/Chicago")
UTC = timezone.utc
NOMINAL_HARD_FLAT_UTC = time(22, 45)
NORMAL_RTH_OPEN_ET = time(9, 30)
NORMAL_RTH_CLOSE_ET = time(16, 0)
EARLY_RTH_CLOSE_ET = time(13, 0)
ES_WARMUP_START_ET = time(9, 0)
MAINTENANCE_START_ET = time(17, 0)
MAINTENANCE_END_ET = time(18, 0)
EARLY_FUTURES_CLOSE_ET = time(13, 15)
EARLY_CLOSE_DATES = frozenset({date(2025, 11, 28), date(2025, 12, 24)})
OFFICIAL_SOURCES = (
    "https://www.cmegroup.com/trading-hours.html",
    "https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/thanksgiving-holiday-settlement-times-2025.pdf",
    "https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/christmas-holiday-settlement-times-2025.pdf",
)


class CalendarAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuoteRequest:
    target_session: str
    prior_rth_date: str
    raw_symbol: str
    start_utc: str
    end_utc: str
    schema: str = "trades"
    dataset: str = "GLBX.MDP3"


def _local(day: date, clock: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, clock, tzinfo=zone)


def _utc(day: date, clock: time, zone: ZoneInfo = NEW_YORK) -> datetime:
    return _local(day, clock, zone).astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def timezone_state(day: date) -> str:
    return "DST" if _local(day, time(12), NEW_YORK).dst() else "STANDARD"


def rth_window(day: date) -> tuple[datetime, datetime]:
    close = EARLY_RTH_CLOSE_ET if day in EARLY_CLOSE_DATES else NORMAL_RTH_CLOSE_ET
    return _utc(day, NORMAL_RTH_OPEN_ET), _utc(day, close)


def maintenance_window(day: date) -> tuple[datetime, datetime]:
    return _utc(day, MAINTENANCE_START_ET), _utc(day, MAINTENANCE_END_ET)


def scheduled_futures_close(day: date) -> datetime:
    if day in EARLY_CLOSE_DATES:
        return _utc(day, EARLY_FUTURES_CLOSE_ET)
    return maintenance_window(day)[0]


def effective_hard_flat(day: date) -> datetime:
    """Apply the frozen UTC cutoff to the executable segments for that date.

    A nominal cutoff inside the daily maintenance pause moves to the last
    executable boundary before the pause.  If the market has reopened before
    22:45 UTC (summer), the nominal cutoff remains valid.  An early holiday
    close always bounds the cutoff.
    """
    nominal = datetime.combine(day, NOMINAL_HARD_FLAT_UTC, tzinfo=UTC)
    if day in EARLY_CLOSE_DATES:
        return min(nominal, scheduled_futures_close(day))
    pause_start, pause_end = maintenance_window(day)
    return pause_start if pause_start <= nominal < pause_end else nominal


def required_es_window(day: date) -> tuple[datetime, datetime]:
    return _utc(day, ES_WARMUP_START_ET), effective_hard_flat(day) + timedelta(seconds=1)


def required_mes_window(day: date) -> tuple[datetime, datetime]:
    return _utc(day, NORMAL_RTH_OPEN_ET), effective_hard_flat(day) + timedelta(seconds=1)


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise CalendarAuditError(f"non-UTC manifest timestamp: {value}")
    return parsed


def _missing(required: tuple[datetime, datetime], acquired: tuple[datetime, datetime]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    required_start, required_end = required
    acquired_start, acquired_end = acquired
    if acquired_start > required_start:
        end = min(acquired_start, required_end)
        if required_start < end:
            result.append({"start_utc": _iso(required_start), "end_utc": _iso(end),
                           "seconds": int((end - required_start).total_seconds()), "position": "PREFIX"})
    if acquired_end < required_end:
        start = max(acquired_end, required_start)
        if start < required_end:
            result.append({"start_utc": _iso(start), "end_utc": _iso(required_end),
                           "seconds": int((required_end - start).total_seconds()), "position": "TAIL"})
    return result


def _window(value: tuple[datetime, datetime]) -> dict[str, str]:
    return {"start_utc": _iso(value[0]), "end_utc": _iso(value[1])}


def _acquired(item: Mapping[str, Any]) -> tuple[datetime, datetime]:
    return _parse(str(item["start_utc"])), _parse(str(item["end_utc"]))


def _linear_proxy_cost(item: Mapping[str, Any], missing_seconds: int) -> Decimal:
    acquired = _acquired(item)
    seconds = Decimal(str((acquired[1] - acquired[0]).total_seconds()))
    cost = Decimal(str(item["fresh_component_cost_usd"]))
    return cost * Decimal(missing_seconds) / seconds


def time_semantics_audit() -> list[dict[str, Any]]:
    """Document current literals versus their canonical calendar meaning."""
    return [
        {"field": "RTH_START", "current_hardcoded_utc": "13:30", "intended_market_time": "US cash open",
         "eastern_local": "09:30", "chicago_local": "08:30", "dst_utc": "13:30", "standard_utc": "14:30",
         "dec_jan_sufficient": True},
        {"field": "RTH_END", "current_hardcoded_utc": "20:00", "intended_market_time": "US cash close",
         "eastern_local": "16:00 (13:00 early close)", "chicago_local": "15:00 (12:00 early close)",
         "dst_utc": "20:00", "standard_utc": "21:00", "dec_jan_sufficient": False},
        {"field": "PRIOR_RTH_PROFILE", "current_hardcoded_utc": "13:30-20:00",
         "intended_market_time": "prior completed US cash RTH", "eastern_local": "09:30-16:00",
         "chicago_local": "08:30-15:00", "dst_utc": "13:30-20:00", "standard_utc": "14:30-21:00",
         "dec_jan_sufficient": False},
        {"field": "INTERACTION_ELIGIBILITY", "current_hardcoded_utc": "13:30-20:00",
         "intended_market_time": "current US cash RTH", "eastern_local": "09:30-16:00",
         "chicago_local": "08:30-15:00", "dst_utc": "13:30-20:00", "standard_utc": "14:30-21:00",
         "dec_jan_sufficient": True},
        {"field": "ES_MBP10_WARMUP_START", "current_hardcoded_utc": "13:00",
         "intended_market_time": "30 minutes before US cash open", "eastern_local": "09:00",
         "chicago_local": "08:00", "dst_utc": "13:00", "standard_utc": "14:00",
         "dec_jan_sufficient": True},
        {"field": "MES_MBP1_REQUIRED_START", "current_hardcoded_utc": "13:30",
         "intended_market_time": "US cash open", "eastern_local": "09:30", "chicago_local": "08:30",
         "dst_utc": "13:30", "standard_utc": "14:30", "dec_jan_sufficient": True},
        {"field": "DAILY_MAINTENANCE", "current_hardcoded_utc": "21:00-22:00 summer / 22:00-23:00 winter",
         "intended_market_time": "CME daily maintenance", "eastern_local": "17:00-18:00",
         "chicago_local": "16:00-17:00", "dst_utc": "21:00-22:00", "standard_utc": "22:00-23:00",
         "dec_jan_sufficient": True},
        {"field": "HARD_FLAT", "current_hardcoded_utc": "nominal 22:45",
         "intended_market_time": "fixed UTC risk cutoff bounded by executable session calendar",
         "eastern_local": "18:45 EDT; nominal 17:45 EST, effective 17:00 EST when inside maintenance",
         "chicago_local": "17:45 CDT; nominal 16:45 CST, effective 16:00 CST when inside maintenance",
         "dst_utc": "22:45", "standard_utc": "22:00 effective", "dec_jan_sufficient": True},
        {"field": "SCHEDULED_HOLIDAY_CLOSE", "current_hardcoded_utc": "18:15 on 2025-12-24",
         "intended_market_time": "equity-index futures early close", "eastern_local": "13:15",
         "chicago_local": "12:15", "dst_utc": "17:15 if DST", "standard_utc": "18:15",
         "dec_jan_sufficient": True},
        {"field": "LIQUIDATION_WINDOW", "current_hardcoded_utc": "22:44:59-22:45:00",
         "intended_market_time": "inclusive final second ending at effective hard flat",
         "eastern_local": "calendar-derived", "chicago_local": "calendar-derived",
         "dst_utc": "22:44:59-22:45:00", "standard_utc": "21:59:59-22:00:00",
         "dec_jan_sufficient": True},
        {"field": "SOURCE_END", "current_hardcoded_utc": "22:45:01 normal",
         "intended_market_time": "one second beyond effective hard flat declaration",
         "eastern_local": "calendar-derived", "chicago_local": "calendar-derived",
         "dst_utc": "22:45:01", "standard_utc": "22:00:01 required",
         "dec_jan_sufficient": True},
    ]


def build_calendar_audit(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    if v3_contract_sha256() != V3_CONTRACT_SHA256:
        raise CalendarAuditError("frozen V3 contract hash mismatch")
    verification = verify_acquisition_manifest(data_root, verify_hashes=False)
    sessions: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    proxy_es = proxy_mes = proxy_prior = Decimal("0")
    for day_text in verification["target_sessions"]:
        day = date.fromisoformat(day_text)
        inputs = verification["session_inputs"][day_text]
        es_item, mes_item, prior_item = inputs["ES_MBP10"], inputs["MES_MBP1"], inputs["PRIOR_RTH_TRADES"]
        prior_day = date.fromisoformat(str(prior_item["prior_rth_date"]))
        required_rth = rth_window(day)
        required_prior = rth_window(prior_day)
        required_es = required_es_window(day)
        required_mes = required_mes_window(day)
        acquired_es, acquired_mes, acquired_prior = _acquired(es_item), _acquired(mes_item), _acquired(prior_item)
        es_missing = _missing(required_es, acquired_es)
        mes_missing = _missing(required_mes, acquired_mes)
        prior_missing = _missing(required_prior, acquired_prior)
        for missing in prior_missing:
            requests.append({
                "target_session": day_text,
                "prior_rth_date": prior_day.isoformat(),
                "dataset": "GLBX.MDP3",
                "schema": "trades",
                "stype_in": "raw_symbol",
                "raw_symbol": str(prior_item["raw_symbol"]),
                "start_utc": missing["start_utc"],
                "end_utc": missing["end_utc"],
            })
            proxy_prior += _linear_proxy_cost(prior_item, int(missing["seconds"]))
        pause = maintenance_window(day)
        flat = effective_hard_flat(day)
        sessions.append({
            "session_date": day_text,
            "timezone_state": timezone_state(day),
            "es_contract": es_item["raw_symbol"],
            "mes_contract": mes_item["raw_symbol"],
            "prior_rth_date": prior_day.isoformat(),
            "prior_rth_contract": prior_item["raw_symbol"],
            "intended_rth_local": "09:30-13:00 America/New_York" if day in EARLY_CLOSE_DATES else "09:30-16:00 America/New_York",
            "required_rth_utc": _window(required_rth),
            "required_prior_rth_utc": _window(required_prior),
            "acquired_prior_rth_utc": _window(acquired_prior),
            "prior_rth_missing_ranges": prior_missing,
            "required_es_mbp10_utc": _window(required_es),
            "acquired_es_mbp10_utc": _window(acquired_es),
            "es_missing_ranges": es_missing,
            "required_mes_mbp1_utc": _window(required_mes),
            "acquired_mes_mbp1_utc": _window(acquired_mes),
            "mes_missing_ranges": mes_missing,
            "maintenance_utc": _window(pause),
            "nominal_hard_flat_utc": _iso(datetime.combine(day, NOMINAL_HARD_FLAT_UTC, tzinfo=UTC)),
            "effective_hard_flat_utc": _iso(flat),
            "scheduled_close_utc": _iso(scheduled_futures_close(day)),
            "liquidation_window_utc": _window((flat - timedelta(seconds=1), flat)),
            "acquisition_sufficient": not (prior_missing or es_missing or mes_missing),
            "special_session": (
                "CHRISTMAS_EVE_SHORTENED" if day == date(2025, 12, 24)
                else "POST_THANKSGIVING_PRIOR_SOURCE" if prior_day == date(2025, 11, 28)
                else "POST_HOLIDAY_NORMAL" if day in {date(2025, 12, 26), date(2026, 1, 2)}
                else "POST_MLK_NORMAL" if day == date(2026, 1, 20)
                else None
            ),
        })
    affected = [row["session_date"] for row in sessions if not row["acquisition_sufficient"]]
    if len(sessions) != EXPECTED_SESSION_COUNT:
        raise CalendarAuditError("calendar audit did not produce exactly 42 sessions")
    if len(requests) != 40 or any(item["start_utc"][11:19] != "20:00:00" or item["end_utc"][11:19] != "21:00:00" for item in requests):
        raise CalendarAuditError("supplemental prior-RTH request set is not the expected 40 one-hour tails")
    return {
        "audit_kind": "L2_V3_DEC2025_JAN2026_DST_CALENDAR_AUDIT",
        "classification": CLASSIFICATION,
        "strategy_id": STRATEGY_ID,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "evidence_label": EVIDENCE_LABEL,
        "outcomes_accessed": False,
        "strategy_runner_invoked": False,
        "dbn_files_opened": False,
        "official_sources": list(OFFICIAL_SOURCES),
        "canonical_semantics": {
            "rth": "09:30-16:00 America/New_York; 09:30-13:00 on official US cash early-close dates",
            "hard_flat": "Nominal 22:45 UTC remains frozen; if it lies in a scheduled non-executable pause, use the last executable boundary before that pause; earlier holiday futures close also bounds it",
            "maintenance": "17:00-18:00 America/New_York (16:00-17:00 America/Chicago)",
            "liquidation_window": "inclusive [effective_hard_flat - 1 second, effective_hard_flat]",
            "source_end": "effective_hard_flat + 1 second declared coverage; actual executable BBO required in liquidation window",
        },
        "mapping_examples": {
            "summer_rth_utc": "13:30-20:00",
            "standard_rth_utc": "14:30-21:00",
            "summer_maintenance_utc": "21:00-22:00",
            "standard_maintenance_utc": "22:00-23:00",
            "summer_effective_hard_flat_utc": "22:45:00",
            "standard_effective_hard_flat_utc": "22:00:00",
        },
        "time_semantics_audit": time_semantics_audit(),
        "manifest_sha256": verification["manifest_sha256"],
        "target_session_count": len(sessions),
        "affected_session_count": len(affected),
        "affected_sessions": affected,
        "existing_acquisition_sufficient": not affected,
        "component_sufficiency": {
            "ES_MBP10": all(not row["es_missing_ranges"] for row in sessions),
            "MES_MBP1": all(not row["mes_missing_ranges"] for row in sessions),
            "PRIOR_RTH_TRADES": all(not row["prior_rth_missing_ranges"] for row in sessions),
        },
        "nov28": {
            "decision": "USABLE_WITH_DEGRADED_SOURCE_WARNING",
            "required_profile_utc": _window(rth_window(date(2025, 11, 28))),
            "acquired_declared_utc": _window(_acquired(verification["session_inputs"]["2025-12-01"]["PRIOR_RTH_TRADES"])),
            "previous_record_audit": {"record_count": 96045, "first_timestamp_utc": "2025-11-28T13:30:00.000000000Z", "last_timestamp_utc": "2025-11-28T18:14:59.632603429Z", "obvious_gaps_over_15_minutes": 0},
            "correct_shortened_rth_fully_covered": True,
            "outcome_dependent": False,
        },
        "supplemental_plan": {
            "status": "QUOTE_REQUIRED_NOT_ACQUIRED",
            "request_count": len(requests),
            "schemas": {"trades": len(requests), "mbp-10": 0, "mbp-1": 0, "mbo": 0},
            "requests": requests,
            "local_linear_duration_proxy_usd": {
                "ES_MBP10": str(proxy_es), "MES_MBP1": str(proxy_mes),
                "PRIOR_RTH_TRADES": str(proxy_prior), "total": str(proxy_es + proxy_mes + proxy_prior),
                "classification": "NON_BINDING_LOCAL_PROXY_NOT_DATABENTO_METADATA_QUOTE",
            },
            "official_metadata_quote_usd": None,
            "download_executed": False,
        },
        "sessions": sessions,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_supplement_requests(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen quote requests only when the complete contract matches."""
    if payload.get("strategy_id") != STRATEGY_ID:
        raise CalendarAuditError("supplement audit strategy identity mismatch")
    if payload.get("v3_contract_sha256") != V3_CONTRACT_SHA256:
        raise CalendarAuditError("supplement audit V3 contract hash mismatch")
    if payload.get("target_session_count") != EXPECTED_SESSION_COUNT:
        raise CalendarAuditError("supplement audit target session count mismatch")
    if payload.get("affected_session_count") != 40:
        raise CalendarAuditError("supplement audit affected session count mismatch")
    plan = payload.get("supplemental_plan")
    if not isinstance(plan, Mapping) or plan.get("request_count") != 40:
        raise CalendarAuditError("supplement audit must declare exactly 40 requests")
    raw_requests = plan.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 40:
        raise CalendarAuditError("supplement audit request list must contain exactly 40 requests")

    requests: list[dict[str, Any]] = []
    symbol_counts = {"ESZ5": 0, "ESH6": 0}
    identities: set[tuple[str, str, str, str]] = set()
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, Mapping):
            raise CalendarAuditError(f"supplement request {index} is not an object")
        item = dict(raw)
        if item.get("dataset") != "GLBX.MDP3" or item.get("schema") != "trades":
            raise CalendarAuditError(f"supplement request {index} has an invalid dataset/schema")
        if item.get("stype_in") != "raw_symbol":
            raise CalendarAuditError(f"supplement request {index} has an invalid input symbology")
        symbol = item.get("raw_symbol")
        if symbol not in symbol_counts:
            raise CalendarAuditError(f"supplement request {index} has an unexpected symbol")
        start = _parse(str(item.get("start_utc")))
        end = _parse(str(item.get("end_utc")))
        if start.date() != end.date() or start.time() != time(20, 0) or end.time() != time(21, 0):
            raise CalendarAuditError(f"supplement request {index} is not the frozen 20:00-21:00 UTC range")
        if end - start != timedelta(hours=1):
            raise CalendarAuditError(f"supplement request {index} is not exactly one hour")
        if item.get("prior_rth_date") != start.date().isoformat():
            raise CalendarAuditError(f"supplement request {index} prior-RTH date disagrees with its range")
        target_session = item.get("target_session")
        if not isinstance(target_session, str):
            raise CalendarAuditError(f"supplement request {index} has no target session")
        identity = (target_session, str(item["prior_rth_date"]), symbol, str(item["start_utc"]))
        if identity in identities:
            raise CalendarAuditError(f"supplement request {index} duplicates an existing request")
        identities.add(identity)
        symbol_counts[symbol] += 1
        requests.append(item)
    if symbol_counts != {"ESZ5": 12, "ESH6": 28}:
        raise CalendarAuditError(f"supplement request symbol counts mismatch: {symbol_counts}")
    return requests


def quote_supplement(client: Any, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    symbol_totals: dict[str, Decimal] = {"ESZ5": Decimal("0"), "ESH6": Decimal("0")}
    rows: list[dict[str, Any]] = []
    for item in requests:
        cost = Decimal(str(client.metadata.get_cost(
            dataset=item["dataset"], schema=item["schema"], stype_in="raw_symbol",
            symbols=[item["raw_symbol"]], start=item["start_utc"], end=item["end_utc"],
        )))
        symbol_totals[str(item["raw_symbol"])] += cost
        rows.append({**dict(item), "estimated_usd": str(cost)})
    return {
        "status": "SUPPLEMENT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED",
        "request_count": len(rows),
        "dataset": "GLBX.MDP3",
        "schema": "trades",
        "request_spec_sha256": _canonical_sha256([dict(item) for item in requests]),
        "symbol_subtotals_usd": {key: str(value) for key, value in symbol_totals.items()},
        "total_usd": str(sum(symbol_totals.values(), Decimal("0"))),
        "requests": rows,
        "download_requested": False,
        "timeseries_requests": 0,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_supplement_quote(
    *,
    audit_path: Path = AUDIT_JSON,
    quote_path: Path = SUPPLEMENT_QUOTE_JSON,
    environment: Mapping[str, str] | None = None,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Quote the existing frozen supplement without rebuilding or mutating it."""
    try:
        audit_bytes = audit_path.read_bytes()
        payload = json.loads(audit_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalendarAuditError(f"unable to read frozen calendar audit: {exc}") from exc
    requests = validate_frozen_supplement_requests(payload)
    env = os.environ if environment is None else environment
    api_key = env.get("DATABENTO_API_KEY")
    if not api_key:
        raise CalendarAuditError("DATABENTO_API_KEY is required for metadata quote")
    if client_factory is None:
        import databento as db
        client_factory = lambda key: db.Historical(key=key)
    client = client_factory(api_key)
    quote = quote_supplement(client, requests)
    quote["calendar_audit_path"] = str(audit_path)
    quote["calendar_audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    quote["v3_contract_sha256"] = V3_CONTRACT_SHA256
    _atomic_write_json(quote_path, quote)
    return quote


def _write_report(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# {STRATEGY_ID} Dec 2025 / Jan 2026 DST calendar audit", "",
        f"Classification: `{CLASSIFICATION}`. This is a structural, outcome-free audit.", "",
        "## Decision", "",
        "RTH is `09:30-16:00 America/New_York` (13:00 close on official cash early-close dates). The nominal hard flat remains the frozen `22:45 UTC`; calendar execution moves it to the last executable boundary when that instant lies inside a scheduled pause. Thus normal winter sessions use 22:00 UTC, while normal summer sessions remain 22:45 UTC after the 21:00-22:00 maintenance reopen.", "",
        f"The existing ES MBP-10 and MES MBP-1 files are sufficient. Prior-RTH trades are incomplete for {payload['affected_session_count']} sessions: each needs only the missing 20:00-21:00 UTC tail. November 28 and December 24 shortened profiles are already complete.", "",
        "No strategy runner, setup, confirmation, trade, outcome, R, or PnL path was accessed.", "",
        "## Time semantics", "",
        "| Field | Current UTC literal | Canonical meaning | Eastern | Chicago | DST UTC | Standard UTC | Dec/Jan sufficient |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["time_semantics_audit"]:
        lines.append(
            f"| {item['field']} | {item['current_hardcoded_utc']} | {item['intended_market_time']} | "
            f"{item['eastern_local']} | {item['chicago_local']} | {item['dst_utc']} | {item['standard_utc']} | "
            f"{str(item['dec_jan_sufficient']).lower()} |"
        )
    lines.extend([
        "",
        "## Per-session audit", "",
        "| Session | TZ | ES/MES | Prior RTH | Required prior RTH UTC | Acquired prior UTC | Missing | Required ES | Required MES | Maintenance | Effective flat | Sufficient |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for row in payload["sessions"]:
        prior_required = row["required_prior_rth_utc"]
        prior_acquired = row["acquired_prior_rth_utc"]
        missing = ", ".join(f"{item['start_utc'][11:19]}-{item['end_utc'][11:19]}" for item in row["prior_rth_missing_ranges"]) or "none"
        lines.append(
            f"| {row['session_date']} | {row['timezone_state']} | {row['es_contract']}/{row['mes_contract']} | {row['prior_rth_date']} {row['prior_rth_contract']} | "
            f"{prior_required['start_utc'][11:19]}-{prior_required['end_utc'][11:19]} | {prior_acquired['start_utc'][11:19]}-{prior_acquired['end_utc'][11:19]} | {missing} | "
            f"{row['required_es_mbp10_utc']['start_utc'][11:19]}-{row['required_es_mbp10_utc']['end_utc'][11:19]} | {row['required_mes_mbp1_utc']['start_utc'][11:19]}-{row['required_mes_mbp1_utc']['end_utc'][11:19]} | "
            f"{row['maintenance_utc']['start_utc'][11:19]}-{row['maintenance_utc']['end_utc'][11:19]} | {row['effective_hard_flat_utc'][11:19]} | {str(row['acquisition_sufficient']).lower()} |"
        )
    lines.extend(["", "## Supplemental plan", "",
                  f"Exactly {payload['supplemental_plan']['request_count']} `GLBX.MDP3` `trades` requests are required; no MBP-10, MBP-1, or MBO supplement is needed. The local duration proxy is `${payload['supplemental_plan']['local_linear_duration_proxy_usd']['total']}` and is not an official Databento quote. Run the quote-only command before acquisition.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=AUDIT_JSON)
    parser.add_argument("--report", type=Path, default=AUDIT_REPORT)
    parser.add_argument("--quote-output", type=Path, default=SUPPLEMENT_QUOTE_JSON)
    parser.add_argument("--quote-supplement", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.quote_supplement:
            quote = run_supplement_quote(audit_path=args.output, quote_path=args.quote_output)
            print(json.dumps({
                "status": quote["status"],
                "request_count": quote["request_count"],
                "ESZ5_subtotal_usd": quote["symbol_subtotals_usd"]["ESZ5"],
                "ESH6_subtotal_usd": quote["symbol_subtotals_usd"]["ESH6"],
                "total_usd": quote["total_usd"],
                "quote_artifact": str(args.quote_output),
            }, sort_keys=True))
            return 0
        payload = build_calendar_audit(args.data_root)
        _atomic_write_json(args.output, payload)
        _write_report(payload, args.report)
        print(json.dumps({"status": "SUPPLEMENT_REQUIRED", "audit": str(args.output), "report": str(args.report),
                          "affected_sessions": payload["affected_session_count"],
                          "supplemental_requests": payload["supplemental_plan"]["request_count"]}, sort_keys=True))
    except (CalendarAuditError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
