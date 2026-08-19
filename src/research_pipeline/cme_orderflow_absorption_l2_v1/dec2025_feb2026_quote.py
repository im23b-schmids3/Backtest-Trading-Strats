"""Metadata-only cost planner for the frozen POC-only L2 V3 research block.

This module deliberately has no market-data or strategy-runner dependency.  Its
only Databento calls are ``symbology.resolve`` and ``metadata.get_cost``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .v3_poc_only import STRATEGY_ID, v3_contract_sha256


DATASET = "GLBX.MDP3"
EXPECTED_V3_HASH = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
START_DATE = date(2025, 12, 1)
END_DATE = date(2026, 2, 27)
BUDGET_USD = Decimal("115")
OUTPUT_ROOT = Path("research_runs/L2_V3_DEC2025_FEB2026_QUOTE")

# Explicit RTH calendar exceptions for the requested block and its first
# prior-session dependency.  U.S. holidays without a completed cash RTH are
# excluded.  Christmas Eve and the post-Thanksgiving session are completed,
# shortened sessions.  The MBP end is the known equity-index futures halt;
# prior-RTH trades stop at the earlier of 20:00 UTC and that halt.
CLOSED_RTH_DATES = frozenset({
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Presidents Day
})
SHORTENED_SESSION_ENDS = {
    date(2025, 11, 28): time(18, 15),
    date(2025, 12, 24): time(18, 15),
}
NORMAL_SESSION_END = time(22, 45)
PROFILE_END = time(20, 0)
CALENDAR_PROVENANCE = (
    "https://www.cmegroup.com/trading-hours.html",
    "https://www.cmegroup.com/trading/equity-index/rolldates.html",
)


class QuotePlanError(RuntimeError):
    """Fail-closed quote-planning error."""


@dataclass(frozen=True)
class Session:
    session_date: date
    previous_rth_date: date
    effective_end: time
    previous_rth_end: time

    @property
    def shortened(self) -> bool:
        return self.effective_end != NORMAL_SESSION_END


@dataclass(frozen=True)
class ResolvedSymbols:
    es_by_date: Mapping[date, str]
    mes_by_date: Mapping[date, str]
    es_instrument_id_by_date: Mapping[date, int]
    mes_instrument_id_by_date: Mapping[date, int]


def _iso(day: date, clock: time, *, plus_one_second: bool = False) -> str:
    value = datetime.combine(day, clock, tzinfo=timezone.utc)
    if plus_one_second:
        value += timedelta(seconds=1)
    return value.isoformat().replace("+00:00", "Z")


def _is_completed_rth(day: date) -> bool:
    return day.weekday() < 5 and day not in CLOSED_RTH_DATES


def _session_end(day: date) -> time:
    return SHORTENED_SESSION_ENDS.get(day, NORMAL_SESSION_END)


def _previous_completed_rth(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not _is_completed_rth(candidate):
        candidate -= timedelta(days=1)
    return candidate


def build_sessions(start: date = START_DATE, end: date = END_DATE) -> tuple[Session, ...]:
    """Build every eligible session in chronological order, without selection."""
    if start > end:
        raise QuotePlanError("start date follows end date")
    sessions: list[Session] = []
    cursor = start
    while cursor <= end:
        if _is_completed_rth(cursor):
            previous = _previous_completed_rth(cursor)
            sessions.append(Session(
                session_date=cursor,
                previous_rth_date=previous,
                effective_end=_session_end(cursor),
                previous_rth_end=min(PROFILE_END, _session_end(previous)),
            ))
        cursor += timedelta(days=1)
    if not sessions or sessions[0].session_date != start or sessions[-1].session_date != end:
        raise QuotePlanError("requested chronology does not begin and end on completed RTH sessions")
    if any(left.session_date >= right.session_date for left, right in zip(sessions, sessions[1:])):
        raise QuotePlanError("session chronology is not strictly increasing")
    return tuple(sessions)


def _response_dict(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        value = response.to_dict()  # type: ignore[union-attr]
        if isinstance(value, dict):
            return value
    raise QuotePlanError("Databento symbology response is not a mapping")


def _validate_resolution(response: object, requested: Sequence[str | int]) -> dict[str, Any]:
    payload = _response_dict(response)
    status = payload.get("status", 0)
    if status not in (0, "0", "OK", None):
        raise QuotePlanError(f"symbology resolution failed with status {status!r}")
    if payload.get("not_found"):
        raise QuotePlanError(f"symbology not found: {payload['not_found']!r}")
    if payload.get("partial"):
        raise QuotePlanError(f"symbology coverage is partial: {payload['partial']!r}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise QuotePlanError("symbology response has no result mapping")
    for symbol in requested:
        if str(symbol) not in {str(key) for key in result}:
            raise QuotePlanError(f"symbology response omitted {symbol!r}")
    return payload


def _entries(payload: Mapping[str, Any], key: str | int) -> list[Mapping[str, Any]]:
    result = payload["result"]
    raw = result.get(key, result.get(str(key)))
    if not isinstance(raw, list) or not raw:
        raise QuotePlanError(f"no mappings returned for {key!r}")
    if not all(isinstance(item, dict) for item in raw):
        raise QuotePlanError(f"malformed mappings returned for {key!r}")
    return raw


def _mapped_value_for_date(entries: Iterable[Mapping[str, Any]], day: date) -> str:
    matches: list[str] = []
    for entry in entries:
        try:
            d0 = date.fromisoformat(str(entry["d0"]))
            d1 = date.fromisoformat(str(entry["d1"]))
            value = str(entry["s"])
        except (KeyError, ValueError) as exc:
            raise QuotePlanError("malformed date-scoped symbology mapping") from exc
        if d0 <= day < d1:
            matches.append(value)
    if len(matches) != 1:
        raise QuotePlanError(f"ambiguous symbology mapping for {day}: {matches!r}")
    return matches[0]


def _matching_mes_symbol(es_symbol: str) -> str:
    match = re.fullmatch(r"ES([HMUZ]\d)", es_symbol)
    if match is None:
        raise QuotePlanError(f"resolved ES symbol is not an outright quarterly contract: {es_symbol!r}")
    return f"MES{match.group(1)}"


def resolve_symbols(client: object, sessions: Sequence[Session]) -> ResolvedSymbols:
    """Resolve volume-front ES and validate its matching MES contract per date."""
    required_es_dates = sorted({item.session_date for item in sessions} |
                               {item.previous_rth_date for item in sessions})
    required_mes_dates = sorted({item.session_date for item in sessions})
    start = required_es_dates[0]
    end_exclusive = required_es_dates[-1] + timedelta(days=1)
    request = {
        "dataset": DATASET,
        "symbols": ["ES.v.0"],
        "stype_in": "continuous",
        "stype_out": "instrument_id",
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
    }
    continuous = _validate_resolution(client.symbology.resolve(**request), ["ES.v.0"])  # type: ignore[attr-defined]
    continuous_entries = _entries(continuous, "ES.v.0")

    es_by_date: dict[date, str] = {}
    mes_by_date: dict[date, str] = {}
    es_ids: dict[date, int] = {}
    mes_ids: dict[date, int] = {}
    for day in required_es_dates:
        es_id = int(_mapped_value_for_date(continuous_entries, day))
        one_day_end = (day + timedelta(days=1)).isoformat()
        raw_request = {
            "dataset": DATASET,
            "symbols": [es_id],
            "stype_in": "instrument_id",
            "stype_out": "raw_symbol",
            "start_date": day.isoformat(),
            "end_date": one_day_end,
        }
        raw_payload = _validate_resolution(client.symbology.resolve(**raw_request), [es_id])  # type: ignore[attr-defined]
        es_symbol = _mapped_value_for_date(_entries(raw_payload, es_id), day)
        es_by_date[day], es_ids[day] = es_symbol, es_id

        if day in required_mes_dates:
            mes_symbol = _matching_mes_symbol(es_symbol)
            mes_request = {
                "dataset": DATASET,
                "symbols": [mes_symbol],
                "stype_in": "raw_symbol",
                "stype_out": "instrument_id",
                "start_date": day.isoformat(),
                "end_date": one_day_end,
            }
            mes_payload = _validate_resolution(client.symbology.resolve(**mes_request), [mes_symbol])  # type: ignore[attr-defined]
            mes_id = int(_mapped_value_for_date(_entries(mes_payload, mes_symbol), day))
            mes_by_date[day], mes_ids[day] = mes_symbol, mes_id

    return ResolvedSymbols(es_by_date, mes_by_date, es_ids, mes_ids)


def _component_requests(session: Session, symbols: ResolvedSymbols) -> tuple[dict[str, Any], ...]:
    day, prior = session.session_date, session.previous_rth_date
    return (
        {
            "component": "ES_MBP10_USD",
            "dataset": DATASET,
            "schema": "mbp-10",
            "symbols": [symbols.es_by_date[day]],
            "stype_in": "raw_symbol",
            "start": _iso(day, time(13, 0)),
            "end": _iso(day, session.effective_end, plus_one_second=True),
        },
        {
            "component": "MES_MBP1_USD",
            "dataset": DATASET,
            "schema": "mbp-1",
            "symbols": [symbols.mes_by_date[day]],
            "stype_in": "raw_symbol",
            "start": _iso(day, time(13, 30)),
            "end": _iso(day, session.effective_end, plus_one_second=True),
        },
        {
            "component": "PRIOR_RTH_TRADES_USD",
            "dataset": DATASET,
            "schema": "trades",
            "symbols": [symbols.es_by_date[prior]],
            "stype_in": "raw_symbol",
            "start": _iso(prior, time(13, 30)),
            "end": _iso(prior, session.previous_rth_end),
        },
    )


def _display(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def _sum_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Decimal]:
    keys = ("ES_MBP10_USD", "MES_MBP1_USD", "PRIOR_RTH_TRADES_USD")
    totals = {key: sum((Decimal(str(row["costs_usd"][key])) for row in rows), Decimal(0)) for key in keys}
    totals["TOTAL_USD"] = sum(totals.values(), Decimal(0))
    return totals


def _scenario(name: str, rows: Sequence[Mapping[str, Any]], *, budget: Decimal | None = None) -> dict[str, Any]:
    totals = _sum_rows(rows)
    total = totals["TOTAL_USD"]
    if budget is not None and total > budget:
        raise QuotePlanError(f"budget scenario {name} exceeds {budget}: {total}")
    return {
        "name": name,
        "session_count": len(rows),
        "start_date": rows[0]["target_session"] if rows else None,
        "end_date": rows[-1]["target_session"] if rows else None,
        "component_totals_usd": {key: str(value) for key, value in totals.items()},
        "total_usd_display": _display(total),
        "budget_usd": str(budget) if budget is not None else None,
        "remaining_budget_usd": str(budget - total) if budget is not None else None,
    }


def longest_prefix(rows: Sequence[Mapping[str, Any]], budget: Decimal = BUDGET_USD) -> tuple[Mapping[str, Any], ...]:
    """Return the longest chronological prefix under budget; never cherry-pick."""
    chosen: list[Mapping[str, Any]] = []
    total = Decimal(0)
    for row in rows:
        candidate = total + Decimal(str(row["session_total_usd"]))
        if candidate > budget:
            break
        chosen.append(row)
        total = candidate
    return tuple(chosen)


def _contract_rolls(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    previous: str | None = None
    for row in rows:
        current = str(row[field])
        if previous is not None and current != previous:
            changes.append({"effective_session": str(row["target_session"]), "from": previous, "to": current})
        previous = current
    return changes


def quote(client: object, sessions: Sequence[Session] | None = None) -> dict[str, Any]:
    """Resolve symbols and quote all components using metadata endpoints only."""
    if v3_contract_sha256() != EXPECTED_V3_HASH:
        raise QuotePlanError("frozen V3 contract hash changed")
    selected = tuple(sessions or build_sessions())
    symbols = resolve_symbols(client, selected)
    rows: list[dict[str, Any]] = []
    cumulative = Decimal(0)
    for session in selected:
        costs: dict[str, str] = {}
        for request in _component_requests(session, symbols):
            component = str(request["component"])
            api_request = {key: value for key, value in request.items() if key != "component"}
            cost = Decimal(str(client.metadata.get_cost(**api_request)))  # type: ignore[attr-defined]
            if not cost.is_finite() or cost < 0:
                raise QuotePlanError(f"invalid Databento cost for {component}: {cost}")
            costs[component] = str(cost)
        session_total = sum((Decimal(value) for value in costs.values()), Decimal(0))
        cumulative += session_total
        day, prior = session.session_date, session.previous_rth_date
        rows.append({
            "target_session": day.isoformat(),
            "es_raw_symbol": symbols.es_by_date[day],
            "es_instrument_id": symbols.es_instrument_id_by_date[day],
            "mes_raw_symbol": symbols.mes_by_date[day],
            "mes_instrument_id": symbols.mes_instrument_id_by_date[day],
            "prior_rth_date": prior.isoformat(),
            "prior_rth_es_raw_symbol": symbols.es_by_date[prior],
            "effective_session_end_utc": _iso(day, session.effective_end),
            "shortened_session": session.shortened,
            "costs_usd": costs,
            "session_total_usd": str(session_total),
            "session_total_usd_display": _display(session_total),
            "cumulative_total_usd": str(cumulative),
            "cumulative_total_usd_display": _display(cumulative),
        })

    monthly = {}
    for month in ("2025-12", "2026-01", "2026-02"):
        month_rows = [row for row in rows if str(row["target_session"]).startswith(month)]
        monthly[month] = _scenario(month, month_rows)
    dec_jan = [row for row in rows if str(row["target_session"]) < "2026-02-01"]
    jan_feb = [row for row in rows if str(row["target_session"]) >= "2026-01-01"]
    from_jan = [row for row in rows if str(row["target_session"]) >= "2026-01-02"]
    scenarios = [
        _scenario("FULL_DEC_JAN_FEB", rows),
        _scenario("DEC_PLUS_JAN", dec_jan),
        _scenario("JAN_PLUS_FEB", jan_feb),
        _scenario("LONGEST_PREFIX_FROM_2025_12_01_UNDER_115", longest_prefix(rows), budget=BUDGET_USD),
        _scenario("LONGEST_PREFIX_FROM_2026_01_02_UNDER_115", longest_prefix(from_jan), budget=BUDGET_USD),
    ]
    return {
        "status": "QUOTE_COMPLETE_NO_DATA_ACQUIRED",
        "strategy_id": STRATEGY_ID,
        "v3_contract_sha256": EXPECTED_V3_HASH,
        "eligible_structural_levels": ["PRIOR_RTH_POC"],
        "quote_only": True,
        "market_data_downloaded": False,
        "strategy_outcomes_run": False,
        "v4_created": False,
        "dataset": DATASET,
        "calendar_provenance": list(CALENDAR_PROVENANCE),
        "sessions": rows,
        "monthly_totals": monthly,
        "grand_total": _scenario("GRAND_TOTAL", rows),
        "scenarios": scenarios,
        "contract_rolls": {
            "ES": _contract_rolls(rows, "es_raw_symbol"),
            "MES": _contract_rolls(rows, "mes_raw_symbol"),
        },
        "cost_precision": "unrounded Decimal sums from metadata.get_cost; display values rounded to cents",
    }


def write_outputs(payload: Mapping[str, Any], output_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_root / "quote.json", output_root / "quote.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = (
        "target_session", "es_raw_symbol", "mes_raw_symbol", "prior_rth_date",
        "prior_rth_es_raw_symbol", "effective_session_end_utc", "shortened_session",
        "ES_MBP10_USD", "MES_MBP1_USD", "PRIOR_RTH_TRADES_USD",
        "session_total_usd", "cumulative_total_usd",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["sessions"]:  # type: ignore[index]
            flat = dict(row)
            flat.update(row["costs_usd"])
            writer.writerow({key: flat[key] for key in fields})
    return json_path, csv_path


def print_report(payload: Mapping[str, Any]) -> None:
    for row in payload["sessions"]:  # type: ignore[index]
        costs = row["costs_usd"]
        print(
            f"{row['target_session']} ES={row['es_raw_symbol']} MES={row['mes_raw_symbol']} "
            f"prior={row['prior_rth_date']} prior_ES={row['prior_rth_es_raw_symbol']} "
            f"ES_MBP10={costs['ES_MBP10_USD']} MES_MBP1={costs['MES_MBP1_USD']} "
            f"PRIOR_TRADES={costs['PRIOR_RTH_TRADES_USD']} total={row['session_total_usd']} "
            f"cumulative={row['cumulative_total_usd']}"
        )
    for month, summary in payload["monthly_totals"].items():  # type: ignore[index]
        print(f"MONTH {month}: {summary['component_totals_usd']['TOTAL_USD']} USD")
    print(f"GRAND TOTAL: {payload['grand_total']['component_totals_usd']['TOTAL_USD']} USD")  # type: ignore[index]
    for scenario in payload["scenarios"]:  # type: ignore[index]
        print(
            f"SCENARIO {scenario['name']}: sessions={scenario['session_count']} "
            f"range={scenario['start_date']}..{scenario['end_date']} "
            f"total={scenario['component_totals_usd']['TOTAL_USD']} "
            f"remaining={scenario['remaining_budget_usd']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote", action="store_true", help="perform metadata-only symbology and cost calls")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if not args.quote:
        plan = build_sessions()
        print(json.dumps({
            "status": "PLAN_ONLY_QUOTE_NOT_EXECUTED",
            "session_count": len(plan),
            "start_date": plan[0].session_date.isoformat(),
            "end_date": plan[-1].session_date.isoformat(),
            "quote_flag_required": True,
            "market_data_downloaded": False,
        }, indent=2, sort_keys=True))
        return 0
    if not os.environ.get("DATABENTO_API_KEY"):
        raise QuotePlanError("DATABENTO_API_KEY is required for metadata-only quoting")
    try:
        import databento as db
    except ImportError as exc:
        raise QuotePlanError("databento package is required for metadata-only quoting") from exc
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    payload = quote(client)
    paths = write_outputs(payload, args.output_root)
    print_report(payload)
    print(f"WROTE {paths[0]}\nWROTE {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
