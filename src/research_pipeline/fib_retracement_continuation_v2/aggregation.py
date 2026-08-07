"""Strict, causal UTC aggregation for the V2 1-minute execution engine."""
from datetime import timedelta, timezone
from decimal import Decimal

from .models import Bar


def _utc_minute(bar: Bar):
    stamp = bar.timestamp.astimezone(timezone.utc)
    if stamp.second or stamp.microsecond:
        raise ValueError("V2_UTC_MINUTE_CADENCE_INVALID")
    return stamp


def validate_1m_bars(rows):
    """Validate supplied bars without filling or reordering a single minute."""
    result = []
    previous = None
    for bar in rows:
        stamp = _utc_minute(bar)
        if previous is not None and stamp - previous != timedelta(minutes=1):
            raise ValueError("V2_1M_MISSING_OR_DUPLICATE_MINUTE")
        if min(bar.open, bar.high, bar.low, bar.close) <= 0 or bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            raise ValueError("V2_INVALID_OHLC")
        result.append(Bar(stamp, *(Decimal(str(getattr(bar, key))) for key in ("open", "high", "low", "close", "volume"))))
        previous = stamp
    return result


def completed_utc_bars(rows, minutes):
    if minutes not in (240, 1440):
        raise ValueError("V2_UNSUPPORTED_HIGHER_TIMEFRAME")
    rows = validate_1m_bars(rows)
    output, group, bucket = [], [], None
    for row in rows:
        stamp = row.timestamp
        start = stamp.replace(hour=(stamp.hour // 4) * 4, minute=0) if minutes == 240 else stamp.replace(hour=0, minute=0)
        if bucket is not None and start != bucket:
            if len(group) == minutes and group[0].timestamp == bucket and group[-1].timestamp == bucket + timedelta(minutes=minutes - 1):
                output.append(Bar(bucket, group[0].open, max(x.high for x in group), min(x.low for x in group), group[-1].close, sum((x.volume for x in group), Decimal(0))))
            group = []
        bucket = start
        group.append(row)
    # The final bucket is deliberately not emitted: it may still be forming.
    return output
