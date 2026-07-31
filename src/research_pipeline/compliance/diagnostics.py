from __future__ import annotations

from statistics import median
from typing import Any, Iterable

from pydantic import Field

from ..schemas.strategy_spec import StrictModel
from .models import HoldingTimePolicy


class ActivityDiagnostics(StrictModel):
    trade_count: int
    average_holding_minutes: float | None = None
    median_holding_minutes: float | None = None
    minimum_holding_minutes: float | None = None
    maximum_holding_minutes: float | None = None
    holding_duration_percentiles: dict[str, float] = Field(default_factory=dict)
    percentage_below_threshold: dict[str, float] = Field(default_factory=dict)
    average_favorable_excursion: float | None = None
    average_adverse_excursion: float | None = None
    trades_per_day: float | None = None
    maximum_trades_per_day: int | None = None
    short_trade_pnl_percentage: float | None = None
    short_small_movement_percentage: float | None = None
    classification: str = "INFORMATIONAL"
    warnings: list[str] = Field(default_factory=list)


def _value(item: Any, name: str, default: Any = None) -> Any:
    return getattr(item, name, item.get(name, default) if isinstance(item, dict) else default)


def calculate_activity_diagnostics(trades: Iterable[Any], policy: HoldingTimePolicy | None = None) -> ActivityDiagnostics:
    rows = list(trades)
    durations: list[float] = []
    by_day: dict[str, int] = {}
    short_pnl = 0.0
    total_pnl = 0.0
    short_small = 0
    favorable: list[float] = []
    adverse: list[float] = []
    warnings: list[str] = []
    threshold = (policy.short_duration_threshold_minutes[0] if policy and policy.short_duration_threshold_minutes else None)
    movement_limit = policy.small_price_movement_threshold if policy else None
    for item in rows:
        entry = _value(item, "entry_time", _value(item, "entry_timestamp"))
        exit_time = _value(item, "exit_time", _value(item, "exit_timestamp"))
        if entry is None or exit_time is None:
            warnings.append("one or more trades lack timestamps")
            continue
        if getattr(entry, "tzinfo", None) is None or getattr(exit_time, "tzinfo", None) is None:
            warnings.append("one or more trades lack timezone-aware timestamps")
            continue
        minutes = max(0.0, (exit_time - entry).total_seconds() / 60)
        durations.append(minutes)
        key = entry.date().isoformat()
        by_day[key] = by_day.get(key, 0) + 1
        pnl = float(_value(item, "net_pnl", 0) or 0)
        total_pnl += pnl
        mfe = _value(item, "favorable_excursion")
        mae = _value(item, "adverse_excursion")
        if mfe is not None:
            favorable.append(float(mfe))
        if mae is not None:
            adverse.append(float(mae))
        if threshold is not None and minutes <= threshold:
            short_pnl += pnl
            if movement_limit is not None:
                price = abs(float(_value(item, "entry", 0) or 0))
                gross = abs(float(_value(item, "gross_pnl", 0) or 0))
                movement = gross / max(price, 1e-12)
                if movement <= movement_limit:
                    short_small += 1
    if not durations:
        return ActivityDiagnostics(trade_count=len(rows), classification="INSUFFICIENT_DATA", warnings=warnings + ["holding-time timestamps unavailable"])
    ordered = sorted(durations)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]
    percentage = {}
    if policy:
        percentage = {str(value): sum(item <= value for item in durations) / len(durations) for value in policy.short_duration_threshold_minutes}
    return ActivityDiagnostics(trade_count=len(rows), average_holding_minutes=sum(durations) / len(durations), median_holding_minutes=median(durations), minimum_holding_minutes=min(durations), maximum_holding_minutes=max(durations), holding_duration_percentiles={"p25": percentile(.25), "p50": percentile(.5), "p75": percentile(.75), "p90": percentile(.9)}, percentage_below_threshold=percentage, average_favorable_excursion=(sum(favorable) / len(favorable) if favorable else None), average_adverse_excursion=(sum(adverse) / len(adverse) if adverse else None), trades_per_day=len(durations) / max(len(by_day), 1), maximum_trades_per_day=max(by_day.values(), default=0), short_trade_pnl_percentage=(short_pnl / total_pnl if total_pnl else None), short_small_movement_percentage=(short_small / len(durations) if movement_limit is not None and threshold is not None else None), classification="INFORMATIONAL", warnings=warnings)
