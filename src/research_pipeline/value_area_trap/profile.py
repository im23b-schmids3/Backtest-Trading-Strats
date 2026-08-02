from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_FLOOR
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field

from ..schemas.strategy_spec import StrictModel
from .data import AggregateTrade

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
SESSION_LABEL = "US_CASH_WINDOW_PROXY"
UTC_SESSION_LABEL = "UTC_24H_SESSION"
US_CASH_SESSION_LABEL = "US_CASH_SESSION"
SessionDefinition = Literal["US_CASH_WINDOW_PROXY", "UTC_24H_SESSION", "US_CASH_SESSION"]

# This study is deliberately fixed to May--July 2026.  These are the NYSE
# full-day closures in that window.  Early closes do not occur in the window,
# so every eligible session uses the declared 09:30--16:00 local interval.
US_CASH_SESSION_HOLIDAYS_2026 = frozenset({
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth National Independence Day
    date(2026, 7, 3),   # Independence Day observed
})


def is_us_cash_trading_day(day: date) -> bool:
    """Return the pinned full-session eligibility used by the equity study."""

    return day.weekday() < 5 and day not in US_CASH_SESSION_HOLIDAYS_2026


class SessionProfile(StrictModel):
    session_date: date
    poc: Decimal
    vah: Decimal
    val: Decimal
    total_session_volume: Decimal
    value_area_volume: Decimal
    coverage_ratio: Decimal
    bucket_size: Decimal
    profile_hash: str
    source_dataset_hash: str
    session_label: str = SESSION_LABEL


class FiveMinuteBar(StrictModel):
    start_utc: datetime
    end_utc: datetime
    start_new_york: datetime
    end_new_york: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    total_volume: Decimal
    aggressive_buy_volume: Decimal
    aggressive_sell_volume: Decimal
    bar_delta: Decimal
    cumulative_volume_delta: Decimal
    trade_count: int
    vwap: Decimal
    session_date: date
    session_label: str = SESSION_LABEL


def session_bounds(
    day: date,
    session_definition: SessionDefinition = SESSION_LABEL,
) -> tuple[datetime, datetime]:
    """Return an explicit, timezone-aware session interval.

    ``UTC_24H_SESSION`` deliberately uses one UTC calendar day.  Its end is
    exclusive, so the final representable timestamp is 23:59:59.999999 UTC.
    The default remains the historical US cash-window proxy unchanged.
    """

    if session_definition == UTC_SESSION_LABEL:
        start = datetime.combine(day, time.min, UTC)
        return start, start + timedelta(days=1)
    start = datetime.combine(day, time(9, 30), NY)
    return start, datetime.combine(day, time(16), NY)


def session_date_for(
    timestamp: datetime,
    session_definition: SessionDefinition = SESSION_LABEL,
) -> date:
    """Resolve the deterministic session date without a fixed UTC offset."""

    zone = UTC if session_definition == UTC_SESSION_LABEL else NY
    return timestamp.astimezone(zone).date()


def in_session(
    timestamp: datetime,
    session_definition: SessionDefinition = SESSION_LABEL,
) -> bool:
    zone = UTC if session_definition == UTC_SESSION_LABEL else NY
    local = timestamp.astimezone(zone)
    if session_definition == US_CASH_SESSION_LABEL and not is_us_cash_trading_day(local.date()):
        return False
    start, end = session_bounds(local.date(), session_definition)
    return start <= local < end


def _bucket(value: Decimal, size: Decimal) -> Decimal:
    return (value / size).to_integral_value(rounding=ROUND_FLOOR) * size


def build_session_profile(trades: Iterable[AggregateTrade], session_date: date, *, bucket_size: Decimal = Decimal("10"), source_dataset_hash: str = "fixture", session_definition: SessionDefinition = SESSION_LABEL) -> SessionProfile | None:
    """70% VA expansion: select POC, then add exactly one larger adjacent bucket each iteration.

    Equal adjacent volumes choose the lower price; POC ties choose the price
    closest to the volume-weighted mean and then the lower price.
    """
    if session_definition == US_CASH_SESSION_LABEL and not is_us_cash_trading_day(session_date):
        return None
    start, end = session_bounds(session_date, session_definition)
    zone = UTC if session_definition == UTC_SESSION_LABEL else NY
    selected = [item for item in trades if start <= item.trade_time_utc.astimezone(zone) < end]
    if not selected:
        return None
    volumes: dict[Decimal, Decimal] = {}
    for item in selected:
        key = _bucket(item.price, bucket_size)
        volumes[key] = volumes.get(key, Decimal()) + item.quantity_base
    total = sum(volumes.values(), Decimal())
    mean = sum((item.price * item.quantity_base for item in selected), Decimal()) / total
    maximum = max(volumes.values())
    poc = min((price for price, volume in volumes.items() if volume == maximum), key=lambda price: (abs(price - mean), price))
    chosen = {poc}; cumulative = volumes[poc]; lower = poc - bucket_size; upper = poc + bucket_size
    while cumulative / total < Decimal("0.70") and (lower in volumes or upper in volumes):
        lower_volume = volumes.get(lower, Decimal("-1")); upper_volume = volumes.get(upper, Decimal("-1"))
        if lower_volume >= upper_volume:  # ties deliberately select lower.
            chosen.add(lower); cumulative += volumes[lower]; lower -= bucket_size
        else:
            chosen.add(upper); cumulative += volumes[upper]; upper += bucket_size
    payload = {str(price): str(volumes[price]) for price in sorted(volumes)}
    profile_payload = {"session": str(session_date), "bucket_size": str(bucket_size), "volumes": payload, "source_dataset_hash": source_dataset_hash}
    # Retain the baseline hash contract; non-baseline session definitions must
    # be distinguishable in their immutable profile provenance.
    if session_definition != SESSION_LABEL:
        profile_payload["session_definition"] = session_definition
    profile_hash = hashlib.sha256(json.dumps(profile_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SessionProfile(session_date=session_date, poc=poc, vah=max(chosen) + bucket_size, val=min(chosen), total_session_volume=total, value_area_volume=cumulative, coverage_ratio=cumulative / total, bucket_size=bucket_size, profile_hash=profile_hash, source_dataset_hash=source_dataset_hash, session_label=session_definition)


def build_session_profiles(trades: Iterable[AggregateTrade], *, bucket_size: Decimal = Decimal("10"), source_dataset_hash: str = "fixture", session_definition: SessionDefinition = SESSION_LABEL) -> dict[date, SessionProfile]:
    records = list(trades)
    dates = sorted({session_date_for(item.trade_time_utc, session_definition) for item in records})
    return {day: profile for day in dates if (profile := build_session_profile(records, day, bucket_size=bucket_size, source_dataset_hash=source_dataset_hash, session_definition=session_definition)) is not None}


def build_five_minute_bars(trades: Iterable[AggregateTrade], *, session_definition: SessionDefinition = SESSION_LABEL) -> list[FiveMinuteBar]:
    records = sorted(trades, key=lambda item: (item.trade_time_utc, item.aggregate_trade_id))
    buckets: dict[datetime, list[AggregateTrade]] = {}
    for item in records:
        zone = UTC if session_definition == UTC_SESSION_LABEL else NY
        local = item.trade_time_utc.astimezone(zone)
        if session_definition == US_CASH_SESSION_LABEL and not is_us_cash_trading_day(local.date()):
            continue
        start, end = session_bounds(local.date(), session_definition)
        if not start <= local < end:
            continue
        floor = local.replace(second=0, microsecond=0, minute=(local.minute // 5) * 5)
        buckets.setdefault(floor.astimezone(ZoneInfo("UTC")), []).append(item)
    bars: list[FiveMinuteBar] = []; cvd_by_day: dict[date, Decimal] = {}
    for start_utc, items in sorted(buckets.items()):
        zone = UTC if session_definition == UTC_SESSION_LABEL else NY
        local = start_utc.astimezone(zone); day = local.date()
        buy = sum((item.quantity_base for item in items if item.signed_quantity > 0), Decimal())
        sell = sum((-item.signed_quantity for item in items if item.signed_quantity < 0), Decimal())
        delta = buy - sell; cvd_by_day[day] = cvd_by_day.get(day, Decimal()) + delta
        volume = buy + sell
        bars.append(FiveMinuteBar(start_utc=start_utc, end_utc=start_utc + timedelta(minutes=5), start_new_york=start_utc.astimezone(NY), end_new_york=(start_utc + timedelta(minutes=5)).astimezone(NY), open=items[0].price, high=max(item.price for item in items), low=min(item.price for item in items), close=items[-1].price, total_volume=volume, aggressive_buy_volume=buy, aggressive_sell_volume=sell, bar_delta=delta, cumulative_volume_delta=cvd_by_day[day], trade_count=len(items), vwap=sum((item.price * item.quantity_base for item in items), Decimal()) / volume, session_date=day, session_label=session_definition))
    return bars
