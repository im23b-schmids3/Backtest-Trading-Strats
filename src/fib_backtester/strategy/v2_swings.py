from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .fibonacci import levels
from .signals import Setup
from .swings import Swing


DEFAULT_MAX_ANCHOR_AGE_DAYS = {"1h": 30.0, "4h": 60.0, "1d": 180.0}


@dataclass(frozen=True)
class LifecycleEvent:
    action: Literal["activate", "update", "invalidate"]
    side: Literal["long", "short"]
    setup_id: str
    index: int
    setup: Setup | None
    reason: str | None = None
    trend_id: int = 0


@dataclass
class V2Construction:
    events_by_index: dict[int, list[LifecycleEvent]] = field(default_factory=dict)
    diagnostics: dict[str, int | float] = field(default_factory=lambda: {
        "anchor_candidates": 0,
        "anchors_created": 0,
        "higher_low_candidates": 0,
        "lower_high_candidates": 0,
        "unique_active_setups": 0,
        "eligible_setups": 0,
        "higher_high_updates": 0,
        "lower_low_updates": 0,
        "fibonacci_recalculations": 0,
        "rejected_distance": 0,
        "rejected_move": 0,
        "anchor_break_invalidations": 0,
        "max_anchor_age_invalidations": 0,
        "setups_invalidated": 0,
        "trend_starts": 2,
    })
    events: list[LifecycleEvent] = field(default_factory=list)

    def emit(self, event: LifecycleEvent) -> None:
        self.events_by_index.setdefault(event.index, []).append(event)
        self.events.append(event)


@dataclass
class _State:
    side: Literal["long", "short"]
    anchor_index: int
    anchor_price: float
    extreme_index: int
    extreme_price: float
    trend_id: int
    setup_id: str | None = None
    creation_index: int | None = None


@dataclass
class _Trend:
    side: Literal["long", "short"]
    states: list[_State]
    extreme_index: int
    extreme_price: float
    trend_id: int = 0
    candidate_index: int | None = None
    candidate_price: float | None = None


def active_wick_lifecycle(
    ohlcv: pd.DataFrame,
    min_distance: int,
    min_move: float,
    max_anchor_age_days: float | None = DEFAULT_MAX_ANCHOR_AGE_DAYS["1h"],
) -> V2Construction:
    """Build causal V2 events with independent Fibonacci generations.

    Each direction keeps every still-valid setup independently.  A favorable
    extreme updates all of those setups, while a pullback creates a provisional
    higher-low/lower-high candidate.  That candidate becomes a new generation
    only after the existing distance and percentage-move filters are satisfied
    by the next favorable extreme.  Events are emitted at candle close and the
    engine acts on them from the following candle.
    """
    if min_distance < 1 or min_move <= 0:
        raise ValueError("min_distance and min_move must be positive")
    if max_anchor_age_days is not None and max_anchor_age_days <= 0:
        raise ValueError("max_anchor_age_days must be positive")
    if ohlcv.empty:
        return V2Construction()

    result = V2Construction()
    first = ohlcv.iloc[0]
    long_state = _State("long", 0, float(first.low), 0, float(first.high), 0)
    short_state = _State("short", 0, float(first.high), 0, float(first.low), 0)
    long = _Trend("long", [long_state], 0, float(first.high))
    short = _Trend("short", [short_state], 0, float(first.low))
    result.diagnostics["anchor_candidates"] = 2
    result.diagnostics["anchors_created"] = 2

    for i in range(1, len(ohlcv)):
        low, high = float(ohlcv.low.iloc[i]), float(ohlcv.high.iloc[i])
        long = _advance_trend(result, ohlcv, long, i, low, high, min_distance, min_move, max_anchor_age_days)
        short = _advance_trend(result, ohlcv, short, i, low, high, min_distance, min_move, max_anchor_age_days)
    return result


def _advance_trend(result, bars, trend, i, low, high, distance, move, max_age_days):
    _expire_old_states(result, bars, trend, i, max_age_days)
    _remove_broken_states(result, bars, trend, i, low, high)

    if not trend.states:
        trend = _start_new_trend(result, trend, i, low, high)

    if trend.side == "long":
        floor = min(state.anchor_price for state in trend.states)
        if low > floor and low < trend.extreme_price:
            if trend.candidate_index is None or low < trend.candidate_price:
                if trend.candidate_index is None:
                    result.diagnostics["anchor_candidates"] += 1
                    result.diagnostics["anchors_created"] += 1
                    result.diagnostics["higher_low_candidates"] += 1
                trend.candidate_index, trend.candidate_price = i, low
        if high > trend.extreme_price:
            previous_candidate = (trend.candidate_index, trend.candidate_price)
            trend.extreme_index, trend.extreme_price = i, high
            for state in trend.states:
                _advance_state(result, bars, state, i, high, distance, move)
            _maybe_create_generation(result, bars, trend, i, high, previous_candidate, distance, move)
    else:
        ceiling = max(state.anchor_price for state in trend.states)
        if high < ceiling and high > trend.extreme_price:
            if trend.candidate_index is None or high > trend.candidate_price:
                if trend.candidate_index is None:
                    result.diagnostics["anchor_candidates"] += 1
                    result.diagnostics["anchors_created"] += 1
                    result.diagnostics["lower_high_candidates"] += 1
                trend.candidate_index, trend.candidate_price = i, high
        if low < trend.extreme_price:
            previous_candidate = (trend.candidate_index, trend.candidate_price)
            trend.extreme_index, trend.extreme_price = i, low
            for state in trend.states:
                _advance_state(result, bars, state, i, low, distance, move)
            _maybe_create_generation(result, bars, trend, i, low, previous_candidate, distance, move)
    return trend


def _expire_old_states(result, bars, trend, i, max_age_days):
    if max_age_days is None:
        return
    limit = pd.Timedelta(days=max_age_days)
    kept = []
    for state in trend.states:
        age = pd.Timestamp(bars.index[i]) - pd.Timestamp(bars.index[state.anchor_index])
        if age <= limit:
            kept.append(state)
            continue
        if state.setup_id:
            result.emit(LifecycleEvent("invalidate", trend.side, state.setup_id, i, None, "anchor_max_age", state.trend_id))
            result.diagnostics["max_anchor_age_invalidations"] += 1
            result.diagnostics["setups_invalidated"] += 1
        trend.candidate_index = None
        trend.candidate_price = None
    trend.states = kept

    if trend.candidate_index is not None:
        age = pd.Timestamp(bars.index[i]) - pd.Timestamp(bars.index[trend.candidate_index])
        if age > limit:
            trend.candidate_index = None
            trend.candidate_price = None


def _remove_broken_states(result, bars, trend, i, low, high):
    if trend.side == "long":
        broken = [state for state in trend.states if low < state.anchor_price]
    else:
        broken = [state for state in trend.states if high > state.anchor_price]
    for state in broken:
        if state.setup_id:
            reason = "anchor_low_broken" if trend.side == "long" else "anchor_high_broken"
            result.emit(LifecycleEvent("invalidate", trend.side, state.setup_id, i, None, reason, state.trend_id))
            result.diagnostics["anchor_break_invalidations"] += 1
            result.diagnostics["setups_invalidated"] += 1
    if broken:
        trend.states = [state for state in trend.states if state not in broken]
        if trend.candidate_price is not None:
            if trend.side == "long" and low <= trend.candidate_price:
                trend.candidate_index = trend.candidate_price = None
            if trend.side == "short" and high >= trend.candidate_price:
                trend.candidate_index = trend.candidate_price = None


def _start_new_trend(result, trend, i, low, high):
    trend.trend_id += 1
    result.diagnostics["anchor_candidates"] += 1
    result.diagnostics["anchors_created"] += 1
    result.diagnostics["trend_starts"] += 1
    if trend.side == "long":
        state = _State("long", i, low, i, high, trend.trend_id)
        return _Trend("long", [state], i, high, trend.trend_id)
    state = _State("short", i, high, i, low, trend.trend_id)
    return _Trend("short", [state], i, low, trend.trend_id)


def _maybe_create_generation(result, bars, trend, i, extreme, candidate, distance, move):
    candidate_index, candidate_price = candidate
    if candidate_index is None or candidate_price is None or candidate_index >= i:
        return
    if trend.side == "long":
        relative = (extreme - candidate_price) / candidate_price
    else:
        relative = (candidate_price - extreme) / candidate_price
    if i - candidate_index < distance:
        result.diagnostics["rejected_distance"] += 1
        return
    if relative < move:
        result.diagnostics["rejected_move"] += 1
        return

    state = _State(trend.side, candidate_index, candidate_price, i, extreme, trend.trend_id)
    trend.states.append(state)
    trend.candidate_index = trend.candidate_price = None
    _emit_state_event(result, bars, state, i, "activate")


def _advance_state(result, bars, state, i, extreme, distance, move):
    state.extreme_index, state.extreme_price = i, extreme
    if i - state.anchor_index < distance:
        result.diagnostics["rejected_distance"] += 1
        return
    relative = (extreme - state.anchor_price) / state.anchor_price if state.side == "long" else (state.anchor_price - extreme) / state.anchor_price
    if relative < move:
        result.diagnostics["rejected_move"] += 1
        return
    _emit_state_event(result, bars, state, i, "activate" if state.setup_id is None else "update")


def _emit_state_event(result, bars, state, i, action):
    if state.setup_id is None:
        state.setup_id = f"v2-{state.side}-{bars.index[state.anchor_index].isoformat()}"
        state.creation_index = i
        result.diagnostics["unique_active_setups"] += 1
        result.diagnostics["eligible_setups"] += 1
    elif state.side == "long":
        result.diagnostics["higher_high_updates"] += 1
    else:
        result.diagnostics["lower_low_updates"] += 1
    result.diagnostics["fibonacci_recalculations"] += 1
    result.emit(LifecycleEvent(action, state.side, state.setup_id, i, _setup_from_state(bars, state, i), None, state.trend_id))


def _setup_from_state(bars: pd.DataFrame, state: _State, index: int) -> Setup:
    created = state.creation_index if state.creation_index is not None else index
    if state.side == "long":
        first = Swing("low", state.anchor_index, created, state.anchor_price, bars.index[state.anchor_index], bars.index[created])
        second = Swing("high", state.extreme_index, index, state.extreme_price, bars.index[state.extreme_index], bars.index[index])
        return Setup(state.setup_id or "", "long", first, second, levels("long", state.anchor_price, state.extreme_price), bars.index[created])
    first = Swing("high", state.anchor_index, created, state.anchor_price, bars.index[state.anchor_index], bars.index[created])
    second = Swing("low", state.extreme_index, index, state.extreme_price, bars.index[state.extreme_index], bars.index[index])
    return Setup(state.setup_id or "", "short", first, second, levels("short", state.extreme_price, state.anchor_price), bars.index[created])


# Compatibility alias for callers that previously imported the V2 constructor.
chronological_wick_setups = active_wick_lifecycle
