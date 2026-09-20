"""Quote-only planner for the historical Asia/Europe and NY completion work.

Only ``symbology.resolve`` and ``metadata.get_cost`` are used.  This module
does not import a replay runner and has no Databento timeseries/download path.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dec2025_feb2026_quote import (
    DATASET,
    CLOSED_RTH_DATES,
    SHORTENED_SESSION_ENDS,
    NORMAL_SESSION_END,
    _entries,
    _mapped_value_for_date,
    _validate_resolution,
)


OUTPUT_PATH = Path("research_runs/L2_HISTORICAL_COMPLETION_QUOTE/quote.json")
JUN_JUL_DATES = (
    "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
    "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
    "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
    "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
    "2026-07-16", "2026-07-17",
)
DEC_JAN_START = date(2025, 12, 1)
DEC_JAN_END = date(2026, 1, 30)
NOV28 = date(2025, 11, 28)


class CompletionQuoteError(RuntimeError):
    """Fail-closed quote planning error."""


@dataclass(frozen=True)
class Request:
    group: str
    component: str
    session_date: date
    symbol_key: date
    schema: str
    start: str
    end: str

    def api_request(self, symbols: Mapping[date, str]) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "schema": self.schema,
            "symbols": [symbols[self.symbol_key]],
            "stype_in": "raw_symbol",
            "start": self.start,
            "end": self.end,
        }


def _iso(day: date, clock: time, *, plus_one_second: bool = False) -> str:
    value = datetime.combine(day, clock, tzinfo=timezone.utc)
    if plus_one_second:
        value += timedelta(seconds=1)
    return value.isoformat().replace("+00:00", "Z")


def _is_completed_rth(day: date) -> bool:
    return day.weekday() < 5 and day not in CLOSED_RTH_DATES


def _session_end(day: date) -> time:
    return SHORTENED_SESSION_ENDS.get(day, NORMAL_SESSION_END)


def _dec_jan_dates() -> tuple[date, ...]:
    result: list[date] = []
    cursor = DEC_JAN_START
    while cursor <= DEC_JAN_END:
        if _is_completed_rth(cursor):
            result.append(cursor)
        cursor += timedelta(days=1)
    if len(result) != 42:
        raise CompletionQuoteError(f"Dec/Jan chronology expected 42 sessions, got {len(result)}")
    return tuple(result)


def build_requests() -> tuple[Request, ...]:
    requests: list[Request] = []
    for value in JUN_JUL_DATES:
        day = date.fromisoformat(value)
        requests.extend((
            Request("JUN_JUL", "JUN_JUL_ES_MBO", day, day, "mbo", _iso(day, time(16)), _iso(day, time(22, 45), plus_one_second=True)),
            Request("JUN_JUL", "JUN_JUL_MES_MBP1", day, day, "mbp-1", _iso(day, time(16)), _iso(day, time(22, 45), plus_one_second=True)),
        ))
    for day in _dec_jan_dates():
        requests.append(Request(
            "DEC_JAN", "DEC_JAN_ES_MBP10", day, day, "mbp-10",
            _iso(day, time(0)), _iso(day, time(13)),
        ))
    requests.append(Request(
        "NOV28_DEPENDENCY", "NOV28_PROFILE_MBP10", NOV28, NOV28, "mbp-10",
        _iso(NOV28, time(0)), _iso(NOV28, time(16, 30)),
    ))
    if len(requests) != 79:
        raise CompletionQuoteError(f"frozen request count mismatch: {len(requests)}")
    return tuple(requests)


def _resolve_es(client: object, required: Iterable[date]) -> dict[date, str]:
    days = sorted(set(required))
    if not days:
        raise CompletionQuoteError("no ES dates to resolve")
    continuous = _validate_resolution(client.symbology.resolve(  # type: ignore[attr-defined]
        dataset=DATASET, symbols=["ES.v.0"], stype_in="continuous", stype_out="instrument_id",
        start_date=days[0].isoformat(), end_date=(days[-1] + timedelta(days=1)).isoformat(),
    ), ["ES.v.0"])
    result: dict[date, str] = {}
    for day in days:
        es_id = int(_mapped_value_for_date(_entries(continuous, "ES.v.0"), day))
        payload = _validate_resolution(client.symbology.resolve(  # type: ignore[attr-defined]
            dataset=DATASET, symbols=[es_id], stype_in="instrument_id", stype_out="raw_symbol",
            start_date=day.isoformat(), end_date=(day + timedelta(days=1)).isoformat(),
        ), [es_id])
        result[day] = _mapped_value_for_date(_entries(payload, es_id), day)
    return result


def _resolve_mes(client: object, days: Iterable[date], es: Mapping[date, str]) -> dict[date, str]:
    result: dict[date, str] = {}
    for day in sorted(set(days)):
        symbol = es[day]
        if not symbol.startswith("ES"):
            raise CompletionQuoteError(f"unexpected ES outright symbol: {symbol}")
        mes_symbol = "MES" + symbol[2:]
        payload = _validate_resolution(client.symbology.resolve(  # type: ignore[attr-defined]
            dataset=DATASET, symbols=[mes_symbol], stype_in="raw_symbol", stype_out="instrument_id",
            start_date=day.isoformat(), end_date=(day + timedelta(days=1)).isoformat(),
        ), [mes_symbol])
        _mapped_value_for_date(_entries(payload, mes_symbol), day)
        result[day] = mes_symbol
    return result


def _quote_requests(client: object, requests: Sequence[Request], es: Mapping[date, str], mes: Mapping[date, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        symbols = mes if request.schema == "mbp-1" else es
        api = request.api_request(symbols)
        cost = Decimal(str(client.metadata.get_cost(**api)))  # type: ignore[attr-defined]
        if not cost.is_finite() or cost < 0:
            raise CompletionQuoteError(f"invalid cost for {request.component}: {cost}")
        rows.append({
            "group": request.group, "component": request.component,
            "session_date": request.session_date.isoformat(),
            "symbol": api["symbols"][0], "schema": request.schema,
            "start": request.start, "end": request.end,
            "cost_usd": str(cost), "request": api,
        })
    return rows


def _sum(rows: Iterable[Mapping[str, Any]]) -> Decimal:
    return sum((Decimal(str(row["cost_usd"])) for row in rows), Decimal(0))


def quote(client: object) -> dict[str, Any]:
    requests = build_requests()
    es_days = [r.symbol_key for r in requests if r.schema != "mbp-1"]
    mes_days = [r.symbol_key for r in requests if r.schema == "mbp-1"]
    es = _resolve_es(client, es_days)
    mes = _resolve_mes(client, mes_days, es)
    rows = _quote_requests(client, requests, es, mes)
    jun_jul = [r for r in rows if r["group"] == "JUN_JUL"]
    dec_jan = [r for r in rows if r["group"] == "DEC_JAN"]
    nov28 = [r for r in rows if r["group"] == "NOV28_DEPENDENCY"]
    es_mbo = [r for r in jun_jul if r["schema"] == "mbo"]
    mes_mbp1 = [r for r in jun_jul if r["schema"] == "mbp-1"]
    def group_total(group: str) -> Decimal:
        return _sum(r for r in rows if r["group"] == group)
    per_day = []
    for value in JUN_JUL_DATES:
        per_day.append({"date": value, "cost_usd": str(_sum(r for r in jun_jul if r["session_date"] == value))})
    for day in _dec_jan_dates():
        per_day.append({"date": day.isoformat(), "cost_usd": str(_sum(r for r in dec_jan if r["session_date"] == day.isoformat()))})
    jun_jul_total = _sum(jun_jul)
    dec_jan_total = _sum(dec_jan)
    total = _sum(rows)
    return {
        "status": "QUOTE_COMPLETE_NO_DATA_ACQUIRED",
        "quote_only": True, "market_data_downloaded": False, "strategy_outcomes_run": False,
        "requests": rows, "request_count": len(rows),
        "june_july_es_mbo_total_usd": str(_sum(es_mbo)),
        "june_july_mes_mbp1_total_usd": str(_sum(mes_mbp1)),
        "june_july_total_usd": str(jun_jul_total),
        "dec_jan_mbp10_total_usd": str(dec_jan_total),
        "nov28_profile_dependency_usd": str(_sum(nov28)),
        "dec_jan_total_usd": str(dec_jan_total + _sum(nov28)),
        "grand_total_usd": str(total),
        "cheapest_request_usd": str(min((Decimal(r["cost_usd"]) for r in rows), default=Decimal(0))),
        "most_expensive_request_usd": str(max((Decimal(r["cost_usd"]) for r in rows), default=Decimal(0))),
        "average_june_july_completion_cost_usd": str(jun_jul_total / Decimal(len(JUN_JUL_DATES))),
        "average_dec_jan_extension_cost_usd": str(dec_jan_total / Decimal(42)),
        "per_date_costs": per_day,
        "no_download_performed": True,
    }


def _print(payload: Mapping[str, Any]) -> None:
    print(f"JUN_JUL_ES_MBO_TOTAL_USD = {payload['june_july_es_mbo_total_usd']}")
    print(f"JUN_JUL_MES_MBP1_TOTAL_USD = {payload['june_july_mes_mbp1_total_usd']}")
    print(f"JUN_JUL_TOTAL_USD = {payload['june_july_total_usd']}")
    print(f"DEC_JAN_MBP10_TOTAL_USD = {payload['dec_jan_mbp10_total_usd']}")
    print(f"NOV28_PROFILE_DEPENDENCY_USD = {payload['nov28_profile_dependency_usd']}")
    print(f"DEC_JAN_TOTAL_USD = {payload['dec_jan_total_usd']}")
    print(f"GRAND_TOTAL_USD = {payload['grand_total_usd']}")
    print(f"REQUEST_COUNT = {payload['request_count']}")
    print(f"CHEAPEST_REQUEST_USD = {payload['cheapest_request_usd']}")
    print(f"MOST_EXPENSIVE_REQUEST_USD = {payload['most_expensive_request_usd']}")
    print(f"AVERAGE_JUN_JUL_COMPLETION_COST_USD = {payload['average_june_july_completion_cost_usd']}")
    print(f"AVERAGE_DEC_JAN_EXTENSION_COST_USD = {payload['average_dec_jan_extension_cost_usd']}")
    for row in payload["requests"]:
        print(f"REQUEST {row['session_date']} {row['symbol']} {row['schema']} {row['start']}..{row['end']} = {row['cost_usd']}")
    print("NO_DOWNLOAD_PERFORMED=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote", action="store_true", help="perform metadata-only quote calls")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if not args.quote:
        requests = build_requests()
        print(json.dumps({"status": "PLAN_ONLY_QUOTE_NOT_EXECUTED", "request_count": len(requests), "quote_flag_required": True}, indent=2))
        return 0
    if not os.environ.get("DATABENTO_API_KEY"):
        raise CompletionQuoteError("DATABENTO_API_KEY is required for metadata-only quoting")
    try:
        import databento as db
    except ImportError as exc:
        raise CompletionQuoteError("databento package is required for metadata-only quoting") from exc
    payload = quote(db.Historical(os.environ["DATABENTO_API_KEY"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(payload)
    print(f"WROTE {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
