from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from statistics import mean
from typing import Iterable

from .models import PortfolioCorrelationMetrics, PortfolioMember, PortfolioOverlapMetrics, PortfolioSignalEvent


def _pearson(left: list[float], right: list[float], minimum: int) -> float | None:
    if len(left) < minimum or len(left) != len(right): return None
    mean_left, mean_right = mean(left), mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def overlap_metrics(candidate_id: str, events: Iterable[PortfolioSignalEvent], members: list[PortfolioMember], overlap_window_minutes: int = 60) -> PortfolioOverlapMetrics:
    ordered = sorted(list(events), key=lambda item: (item.entry_timestamp, item.signal_id))
    total = {member.strategy_id: sum(item.strategy_id == member.strategy_id for item in ordered) for member in members}
    exact = 0; same = 0; opposite = 0; unique_owner: dict[str, int] = defaultdict(int)
    for left, right in combinations(ordered, 2):
        if left.market != right.market: continue
        if abs((left.entry_timestamp - right.entry_timestamp).total_seconds()) > overlap_window_minutes * 60: continue
        if (left.entry_timestamp, left.direction, left.entry_price, left.exit_price) == (right.entry_timestamp, right.direction, right.entry_price, right.exit_price): exact += 1
        if left.direction == right.direction: same += 1
        else: opposite += 1
    for event in ordered: unique_owner[event.strategy_id] += 1
    unique = len({(item.market, item.entry_timestamp, item.direction) for item in ordered})
    total_signals = len(ordered)
    return PortfolioOverlapMetrics(candidate_id=candidate_id, total_signals_by_strategy=total, unique_portfolio_signals=unique, exact_duplicates=exact, same_direction_overlaps=same, opposite_signal_conflicts=opposite, exposure_skips=0, unique_contribution_rate={key: value / max(1, unique) for key, value in unique_owner.items()}, signal_overlap_rate=max(0, total_signals - unique) / max(1, total_signals), duplicate_exposure_rate=(same + exact) / max(1, total_signals), opposite_signal_conflict_rate=opposite / max(1, total_signals), simultaneous_position_rate=0)


def _period(timestamp: datetime, period: str) -> str:
    if period == "day": return timestamp.date().isoformat()
    if period == "week": return f"{timestamp.isocalendar().year}-W{timestamp.isocalendar().week:02d}"
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def _drawdown(series: list[float]) -> list[float]:
    peak = 0.0; result = []
    for value in series:
        peak = max(peak, value); result.append(peak - value)
    return result


def correlation_metrics(candidate_id: str, events: Iterable[PortfolioSignalEvent], members: list[PortfolioMember], minimum_periods: int = 20) -> PortfolioCorrelationMetrics:
    event_list = list(events)
    daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float)); weekly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float)); monthly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    outcomes: dict[str, list[float]] = defaultdict(list)
    for event in event_list:
        direction = 1 if event.direction == "LONG" else -1
        pnl = direction * (event.exit_price - event.entry_price) / max(1, abs(event.entry_price - event.stop))
        for period, target in (("day", daily), ("week", weekly), ("month", monthly)):
            target[_period(event.exit_timestamp, period)][event.strategy_id] += pnl
        outcomes[event.strategy_id].append(1.0 if pnl > 0 else 0.0)
    pairs = [f"{left}|{right}" for left, right in combinations(sorted(member.strategy_id for member in members), 2)]
    def aligned(source: dict[str, dict[str, float]]) -> tuple[dict[str, float | None], int]:
        result: dict[str, float | None] = {}
        periods = set(source)
        for key in pairs:
            left, right = key.split("|"); shared = sorted(period for period in periods if left in source[period] and right in source[period])
            result[key] = _pearson([source[item][left] for item in shared], [source[item][right] for item in shared], minimum_periods)
        return result, len(periods)
    daily_corr, daily_count = aligned(daily); weekly_corr, weekly_count = aligned(weekly); monthly_corr, monthly_count = aligned(monthly)
    trade_corr = {key: _pearson(outcomes[key.split("|")[0]], outcomes[key.split("|")[1]], minimum_periods) for key in pairs}
    drawdowns: dict[str, list[float]] = {member.strategy_id: _drawdown([daily[period].get(member.strategy_id, 0) for period in sorted(daily)]) for member in members}
    draw_corr = {key: _pearson(drawdowns[key.split("|")[0]], drawdowns[key.split("|")[1]], minimum_periods) for key in pairs}
    worst_days = [period for period, values in daily.items() if sum(values.values()) < 0]
    worst_overlap = sum(all(period in daily and daily[period].get(member.strategy_id, 0) < 0 for member in members) for period in worst_days) / max(1, len(worst_days))
    sufficient = all(value is not None for value in daily_corr.values()) if pairs else False
    return PortfolioCorrelationMetrics(candidate_id=candidate_id, aligned_daily_periods=daily_count, aligned_weekly_periods=weekly_count, aligned_monthly_periods=monthly_count, minimum_required_periods=minimum_periods, daily_pnl_correlation=daily_corr, weekly_pnl_correlation=weekly_corr, monthly_pnl_correlation=monthly_corr, trade_outcome_correlation=trade_corr, drawdown_correlation=draw_corr, worst_day_overlap=worst_overlap, worst_week_overlap=worst_overlap, losing_streak_overlap=worst_overlap, simultaneous_adverse_excursion_rate=worst_overlap, sufficient_evidence=sufficient, reason="aligned causal periods" if sufficient else "insufficient overlapping periods; low correlation is not diversification")
