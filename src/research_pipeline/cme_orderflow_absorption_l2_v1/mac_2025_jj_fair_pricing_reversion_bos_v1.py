"""Fixed ES session-open displacement/BOS reversion mechanism study.

This is a mechanical operationalization of a public fair-pricing idea, not a
replication of any private strategy. Spring 2025 is exploratory; October 2025
is secondary compatibility/development data, not untouched OOS. The run uses
only hash-bound native ES MBP-10 candidate tapes and their actual ES trades.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from datetime import date, datetime, time as wall_time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import mac_2025_es_flow_momentum_v1 as flow
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_native_mbp_quote as plan
from .model import ES_COMMISSION, ES_POINT_VALUE

STUDY_ID = "ES_JJ_PUBLIC_FAIR_PRICING_REVERSION_BOS_EVENT_STUDY_V1"
RUN_ID = "CMEOrderflow_ES_JJ_FAIR_PRICING_REVERSION_BOS_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
DATA_ROOT = native.DATA_ROOT
TAPE_VERSION = "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE"
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NS = 1_000_000_000
MINUTE_NS = 60 * NS
TICK = 0.25
DISPLACEMENT_TICKS = 32
MIN_REMAINING_TICKS = 16
DETECTION_START_MINUTE = 1
DETECTION_END_MINUTE = 135  # 11:45 ET close boundary relative to 09:30.
BOS_START_MINUTE = 15       # 09:45 ET.
BOS_END_MINUTE = 150        # 12:00 ET.
LAST_ENTRY_MINUTE = 150     # 12:00 ET, inclusive.
EXIT_DEADLINE_MINUTE = 385  # 15:55 ET.
OPENING_MINUTE_END = MINUTE_NS
FORWARD_HORIZONS_SECONDS = (10, 60, 300, 900, 1800, 3600)

SPRING_DATES = tuple(flow.SPRING_DATES)
OCTOBER_DATES = tuple(flow.OCTOBER_DATES)
TARGET_DATES = SPRING_DATES + OCTOBER_DATES
SOURCE_DATES = tuple(native.ALL_SOURCE_DATES)

CONFIG = {
    "study_id": STUDY_ID,
    "description": "Mechanical public-theory operationalization; not JJ Simon's private/exact strategy.",
    "period_roles": {"SPRING_2025": "EXPLORATORY_DISCOVERY", "OCTOBER_2025": "SECONDARY_DEV_COMPATIBILITY_NOT_UNTOUCHED_OOS"},
    "dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES",
    "source": "Existing hash-bound native ES candidate tapes: actual T-record prices and validated executable BBO; no tape rebuild.",
    "clock": "Canonical receive-ordered timestamp_ns, preserving causal tape order; candle bars use actual ES T records only.",
    "anchor": "First actual ES trade at or after 09:30:00 America/New_York and before 09:31:00 ET; fixed for date.",
    "bar": "Trade-only 1-minute OHLCV, [minute_start, minute_end), completed at minute_end; missing trade minutes are not synthesized.",
    "displacement": {"threshold_ticks": DISPLACEMENT_TICKS, "up": "completed close >= anchor + 32 ticks", "down": "completed close <= anchor - 32 ticks", "detect_close_minutes_after_open": [DETECTION_START_MINUTE, DETECTION_END_MINUTE]},
    "episode_definition": "One episode begins on the first completed close crossing the directional 32-tick threshold; same-direction rearming requires a completed close back inside the threshold before another crossing. An episode is not regenerated each minute while price remains displaced.",
    "bos": {"up_displacement": "SHORT when completed close[i] < min(low[i-1], low[i-2])", "down_displacement": "LONG when completed close[i] > max(high[i-1], high[i-2])", "reference": "immediately preceding two calendar-minute bars; missing bar means no BOS confirmation on that minute", "eligible_close_minutes_after_open": [BOS_START_MINUTE, BOS_END_MINUTE], "signal_timestamp": "BOS candle end boundary"},
    "remaining_distance_ticks": MIN_REMAINING_TICKS,
    "windows_et": {"open": "09:30", "displacement_detection": "09:31-11:45", "bos_confirmation": "09:45-12:00", "last_entry": "12:00", "position_exit_deadline": "15:55"},
    "episode_expiry": ["anchor actual-trade touch before BOS", "12:00 ET confirmation window end", "a first executable BOS entry for the directional side already recorded"],
    "frequency": "At most one filled primary BOS entry and one control entry per reversion direction/date; primary uses first opposing BOS per displacement episode. Control uses first displacement episode/direction only.",
    "control": {"name": "DISPLACEMENT_ONLY_REVERSION_CONTROL", "entry": "first completed candle at/after 09:45 ET while first episode remains active and close still satisfies 32-tick displacement; no BOS required", "same_anchor_stop_target_costs": True, "different_entry_time_and_price_caveat": "Control contrast tests the complete BOS entry rule, not an isolated BOS feature."},
    "entry": "First valid executable BBO at or after signal+2ms; long ask / short bid, then one additional adverse ES tick.",
    "stop": "SHORT max actual trade price from displacement detection through BOS + 1 tick; LONG min actual trade price over that interval - 1 tick. Invalid adverse-side geometry is rejected.",
    "target": "Fixed 09:30 anchor; long triggers on executable bid >= anchor, short triggers on executable ask <= anchor; aggressive liquidation at quote with one adverse exit tick.",
    "same_quote_stop_target_precedence": "STOP first.",
    "exit": "First target/stop before 15:55 ET; otherwise first valid executable BBO at/after 15:55 ET. Missing reliable EOD quote is CENSORED, never a win.",
    "fees": {"commission_and_fees_usd_per_side_per_ES_contract": float(ES_COMMISSION), "authoritative_project_ES_fee_used": True, "point_value_usd": float(ES_POINT_VALUE)},
    "risk": "One ES contract reference economics; no account sizing. Initial risk includes adverse stop fill and round-trip fees. Gross_R=gross PnL/initial risk; net_R=net PnL/initial risk.",
    "markouts_seconds": list(FORWARD_HORIZONS_SECONDS),
    "markout_reference": "Raw trade price, executable-side quote, and quote markout from actual adverse entry fill; signal-clock horizons; actual-fill values are null before entry.",
    "policy_c": "Existing sealed tape is required to report complete BBO path; temporary non-executable source rows follow the existing Policy-C adapter behavior; no integrity gates weakened.",
    "excluded": ["MBO", "MES market data", "Delta/CVD/MLOFI/order-book features", "trend/volatility/news/regime filters", "VWAP/POC/VAH targets", "optimization or rule variants"],
}


class FairPriceStudyError(RuntimeError):
    """Source, causal, or execution contract could not be satisfied."""


def _hash_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


CONFIG_SHA256 = _hash_obj(CONFIG)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ns(dt: datetime) -> int:
    utc = dt.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc - epoch
    return (delta.days * 86400 + delta.seconds) * NS + delta.microseconds * 1000


def et_clock_ns(day: str, hour: int, minute: int, second: int = 0) -> int:
    local = datetime.combine(date.fromisoformat(day), wall_time(hour, minute, second), tzinfo=NY)
    return _ns(local)


def ny_open_ns(day: str) -> int:
    return et_clock_ns(day, 9, 30)


def period_for(day: str) -> str:
    if day in SPRING_DATES:
        return "SPRING_2025"
    if day in OCTOBER_DATES:
        return "OCTOBER_2025"
    raise FairPriceStudyError(f"date not in authoritative target lists: {day}")


def build_one_minute_bars(timestamps_ns: Sequence[int], prices: Sequence[float], sizes: Sequence[float],
                          open_ns: int, end_ns: int) -> list[dict[str, Any]]:
    """Create observed trade-only bars; do not synthesize no-trade minutes."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    px = np.asarray(prices, dtype=np.float64)
    sz = np.asarray(sizes, dtype=np.float64)
    if not (len(ts) == len(px) == len(sz)):
        raise FairPriceStudyError("trade timestamp/price/size arrays differ in length")
    if len(ts) and np.any(np.diff(ts) < 0):
        raise FairPriceStudyError("actual trade events are not causally ordered")
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(ts):
        if open_ns <= int(t) < end_ns:
            buckets[(int(t) - open_ns) // MINUTE_NS].append(i)
    bars = []
    for minute, indices in sorted(buckets.items()):
        vals = px[indices]
        if not np.all(np.isfinite(vals)):
            raise FairPriceStudyError(f"non-finite trade price in minute {minute}")
        bars.append({
            "minute_index": int(minute), "start_ns": open_ns + int(minute) * MINUTE_NS,
            "end_ns": open_ns + (int(minute) + 1) * MINUTE_NS,
            "open": float(vals[0]), "high": float(np.max(vals)), "low": float(np.min(vals)),
            "close": float(vals[-1]), "volume": float(np.sum(sz[indices])),
            "trade_count": len(indices), "first_trade_ns": int(ts[indices[0]]),
            "last_trade_ns": int(ts[indices[-1]]),
        })
    return bars


def opening_anchor(trade_ts: Sequence[int], trade_prices: Sequence[float], open_ns: int) -> dict[str, Any]:
    ts = np.asarray(trade_ts, dtype=np.int64)
    px = np.asarray(trade_prices, dtype=np.float64)
    lo = int(np.searchsorted(ts, open_ns, side="left"))
    hi = int(np.searchsorted(ts, open_ns + OPENING_MINUTE_END, side="left"))
    for i in range(lo, hi):
        if math.isfinite(float(px[i])) and float(px[i]) > 0:
            return {"price": float(px[i]), "timestamp_ns": int(ts[i]), "status": "VALID"}
    return {"price": None, "timestamp_ns": None, "status": "MISSING_OPENING_MINUTE_TRADE"}


def bos_for_bar(bar_map: Mapping[int, Mapping[str, Any]], minute_index: int,
                displacement_direction: int) -> dict[str, Any]:
    """Evaluate close-through against the two immediately prior completed minutes."""
    if minute_index - 1 not in bar_map or minute_index - 2 not in bar_map:
        return {"confirmed": False, "status": "MISSING_REFERENCE_CANDLE", "reference_level": None}
    one, two, current = bar_map[minute_index - 1], bar_map[minute_index - 2], bar_map[minute_index]
    if displacement_direction > 0:
        level = min(float(one["low"]), float(two["low"]))
        confirmed = float(current["close"]) < level
    else:
        level = max(float(one["high"]), float(two["high"]))
        confirmed = float(current["close"]) > level
    return {"confirmed": confirmed, "status": "CONFIRMED" if confirmed else "NO_CLOSE_THROUGH",
            "reference_level": level, "prior_minute_indices": [minute_index - 2, minute_index - 1]}


def distance_from_anchor_ticks(anchor: float, price: float, displacement_direction: int) -> float:
    return float(displacement_direction * (price - anchor) / TICK)


def is_displaced_close(anchor: float, close: float, displacement_direction: int) -> bool:
    """Whether a completed close meets the frozen 32-tick threshold."""
    return distance_from_anchor_ticks(anchor, close, displacement_direction) >= DISPLACEMENT_TICKS


def structural_stop(displacement_direction: int, anchor: float, extreme: float) -> float:
    del anchor  # Explicitly not an input to this fixed adverse-extreme stop.
    return float(extreme + TICK if displacement_direction > 0 else extreme - TICK)


def resolve_exit_trigger(stop_hit: bool, target_hit: bool) -> str | None:
    """Conservative ordering when one quote/event is classified as both barriers."""
    if stop_hit:
        return "STOP"
    if target_hit:
        return "TARGET"
    return None


def classify_bos_distance(distance_ticks: float, direction_already_traded: bool = False) -> str:
    if distance_ticks < MIN_REMAINING_TICKS:
        return "BOS_TOO_CLOSE_TO_ANCHOR"
    if direction_already_traded:
        return "DIRECTION_ALREADY_TRADED"
    return "BOS_CONFIRMED"


def _valid_quote(bid: float, ask: float) -> bool:
    return math.isfinite(bid) and math.isfinite(ask) and ask > bid


def _trade_arrays(tape: np.ndarray, open_ns: int, end_ns: int):
    ts = np.asarray(tape["timestamp_ns"], dtype=np.int64)
    mask = ((np.asarray(tape["execution_size"]) > 0) & np.isfinite(tape["execution_price"]) &
            (ts >= open_ns) & (ts < end_ns))
    ids = np.flatnonzero(mask)
    return ts[ids], np.asarray(tape["execution_price"][ids], dtype=np.float64), np.asarray(tape["execution_size"][ids], dtype=np.float64), ids


def _first_anchor_touch(ts: np.ndarray, prices: np.ndarray, start_ns: int, end_ns: int,
                        anchor: float, displacement_direction: int) -> int | None:
    lo = int(np.searchsorted(ts, start_ns, side="left"))
    hi = int(np.searchsorted(ts, end_ns, side="left"))
    p = prices[lo:hi]
    hit = p <= anchor if displacement_direction > 0 else p >= anchor
    ix = np.flatnonzero(hit)
    return int(ts[lo + int(ix[0])]) if len(ix) else None


def simulate_entry_and_path(tape: np.ndarray, *, day: str, signal_ns: int,
                            signal_close: float, direction: int, anchor: float,
                            stop: float, last_entry_ns: int, exit_deadline_ns: int,
                            session_end_ns: int) -> dict[str, Any]:
    """Use validated BBO, +2ms, one adverse tick, stop-first, fee-inclusive R."""
    ts = np.asarray(tape["timestamp_ns"], dtype=np.int64)
    bid = np.asarray(tape["bid"], dtype=np.float64)
    ask = np.asarray(tape["ask"], dtype=np.float64)
    ready_ns = signal_ns + 2_000_000
    ix = int(np.searchsorted(ts, ready_ns, side="left"))
    entry_ix = None
    while ix < len(ts) and ts[ix] <= last_entry_ns and ts[ix] < session_end_ns:
        if _valid_quote(float(bid[ix]), float(ask[ix])):
            entry_ix = ix
            break
        ix += 1
    if entry_ix is None:
        return {"status": "NO_VALID_ENTRY_QUOTE_BEFORE_LAST_ENTRY", "signal_timestamp_ns": signal_ns,
                "earliest_order_ready_ns": ready_ns, "entry_timestamp_ns": None}
    entry_quote = float(ask[entry_ix] if direction > 0 else bid[entry_ix])
    entry_fill = entry_quote + direction * TICK
    target_distance_ticks = direction * (anchor - entry_fill) / TICK
    stop_distance_ticks = direction * (entry_fill - stop) / TICK
    if target_distance_ticks <= 0:
        return {"status": "TARGET_NOT_AHEAD_AT_ENTRY", "signal_timestamp_ns": signal_ns,
                "entry_timestamp_ns": int(ts[entry_ix]), "entry_price": entry_fill,
                "entry_quote": entry_quote, "target_distance_ticks": target_distance_ticks}
    if stop_distance_ticks <= 0:
        return {"status": "INVALID_STRUCTURAL_STOP", "signal_timestamp_ns": signal_ns,
                "entry_timestamp_ns": int(ts[entry_ix]), "entry_price": entry_fill,
                "entry_quote": entry_quote, "stop_distance_ticks": stop_distance_ticks}

    outcome, exit_ix = None, None
    for j in range(entry_ix + 1, len(ts)):
        t = int(ts[j])
        if t >= session_end_ns:
            break
        if not _valid_quote(float(bid[j]), float(ask[j])):
            continue
        if t >= exit_deadline_ns:
            outcome, exit_ix = "END_OF_DAY", j
            break
        exit_ref = float(bid[j] if direction > 0 else ask[j])
        stop_hit = direction * (exit_ref - stop) <= 0
        target_hit = (float(bid[j]) >= anchor) if direction > 0 else (float(ask[j]) <= anchor)
        trigger = resolve_exit_trigger(stop_hit, target_hit)
        if trigger:
            outcome, exit_ix = trigger, j
            break
    if exit_ix is None:
        return {"status": "CENSORED_NO_RELIABLE_EXIT_QUOTE", "signal_timestamp_ns": signal_ns,
                "entry_timestamp_ns": int(ts[entry_ix]), "entry_price": entry_fill,
                "entry_quote": entry_quote, "stop_price": stop, "target_price": anchor,
                "target_distance_ticks": target_distance_ticks, "stop_distance_ticks": stop_distance_ticks}

    exit_ref = float(bid[exit_ix] if direction > 0 else ask[exit_ix])
    exit_fill = exit_ref - direction * TICK
    fee = 2.0 * float(ES_COMMISSION)
    risk_usd = abs(entry_fill - (stop - direction * TICK)) * float(ES_POINT_VALUE) + fee
    gross_usd = direction * (exit_fill - entry_fill) * float(ES_POINT_VALUE)
    net_usd = gross_usd - fee
    if risk_usd <= 0:
        raise FairPriceStudyError("nonpositive initial risk after fees")

    path_refs = np.asarray([(float(bid[j]) if direction > 0 else float(ask[j])) - direction * TICK
                            for j in range(entry_ix, exit_ix + 1)
                            if _valid_quote(float(bid[j]), float(ask[j]))], dtype=np.float64)
    signed_ticks = direction * (path_refs - entry_fill) / TICK
    mfe_ticks = float(max(0.0, np.max(signed_ticks))) if len(signed_ticks) else 0.0
    mae_ticks = float(max(0.0, -np.min(signed_ticks))) if len(signed_ticks) else 0.0
    risk_ticks_fee_adjusted = risk_usd / (float(ES_POINT_VALUE) * TICK)
    target_ts = int(ts[exit_ix]) if outcome == "TARGET" else None
    stop_ts = int(ts[exit_ix]) if outcome == "STOP" else None
    trade_ts, trade_px = np.asarray(tape["timestamp_ns"], dtype=np.int64), np.asarray(tape["execution_price"], dtype=np.float64)
    trade_mask = (np.asarray(tape["execution_size"]) > 0) & np.isfinite(trade_px)
    trade_ids = np.flatnonzero(trade_mask & (trade_ts >= signal_ns) & (trade_ts <= int(ts[exit_ix])))
    anchor_touched = bool(np.any(trade_px[trade_ids] <= anchor) if direction > 0 else np.any(trade_px[trade_ids] >= anchor))
    return {
        "status": "EXECUTED", "outcome": outcome, "signal_timestamp_ns": signal_ns,
        "entry_ready_timestamp_ns": ready_ns, "entry_timestamp_ns": int(ts[entry_ix]),
        "entry_quote": entry_quote, "entry_price": entry_fill,
        "exit_timestamp_ns": int(ts[exit_ix]), "exit_quote": exit_ref, "exit_price": exit_fill,
        "direction": "LONG" if direction > 0 else "SHORT", "anchor_price": anchor,
        "target_price": anchor, "stop_price": stop,
        "target_distance_ticks": float(target_distance_ticks), "structural_risk_ticks": float(stop_distance_ticks),
        "stop_ticks_including_exit_slippage": float(abs(entry_fill - (stop - direction * TICK)) / TICK),
        "initial_target_r_gross_geometry": float(target_distance_ticks / stop_distance_ticks),
        "initial_risk_usd": risk_usd, "fees_usd": fee,
        "gross_pnl_usd": gross_usd, "net_pnl_usd": net_usd,
        "gross_r": gross_usd / risk_usd, "net_r": net_usd / risk_usd,
        "mfe_ticks": mfe_ticks, "mae_ticks": mae_ticks,
        "mfe_r": mfe_ticks / risk_ticks_fee_adjusted,
        "mae_r": -mae_ticks / risk_ticks_fee_adjusted,
        "time_to_exit_seconds": (int(ts[exit_ix]) - int(ts[entry_ix])) / NS,
        "time_to_target_seconds": (int(ts[exit_ix]) - signal_ns) / NS if target_ts else None,
        "time_to_stop_seconds": (int(ts[exit_ix]) - signal_ns) / NS if stop_ts else None,
        "anchor_touch_before_exit": anchor_touched,
        "exit_timestamp_after_deadline": int(ts[exit_ix]) >= exit_deadline_ns,
    }


def _markouts(tape: np.ndarray, *, signal_ns: int, signal_price: float,
              direction: int, entry: Mapping[str, Any] | None, session_end_ns: int) -> dict[str, Any]:
    ts = np.asarray(tape["timestamp_ns"], dtype=np.int64)
    bid, ask = np.asarray(tape["bid"], dtype=np.float64), np.asarray(tape["ask"], dtype=np.float64)
    trade_mask = (np.asarray(tape["execution_size"]) > 0) & np.isfinite(tape["execution_price"])
    trade_ids = np.flatnonzero(trade_mask)
    tts = ts[trade_ids]
    tpx = np.asarray(tape["execution_price"][trade_ids], dtype=np.float64)
    result = {}
    for seconds in FORWARD_HORIZONS_SECONDS:
        target_ns = signal_ns + seconds * NS
        ti = int(np.searchsorted(tts, target_ns, side="left"))
        qi = int(np.searchsorted(ts, target_ns, side="left"))
        raw = float(direction * (tpx[ti] - signal_price) / TICK) if ti < len(tts) and tts[ti] < session_end_ns else None
        if qi < len(ts) and ts[qi] < session_end_ns and _valid_quote(float(bid[qi]), float(ask[qi])):
            exit_quote = float(bid[qi] if direction > 0 else ask[qi])
            executable = float(direction * (exit_quote - signal_price) / TICK)
        else:
            exit_quote, executable = None, None
        actual = None
        if entry and entry.get("status") == "EXECUTED" and qi < len(ts) and ts[qi] < session_end_ns and ts[qi] >= entry["entry_timestamp_ns"] and _valid_quote(float(bid[qi]), float(ask[qi])):
            exit_fill_ref = (float(bid[qi]) if direction > 0 else float(ask[qi])) - direction * TICK
            actual = float(direction * (exit_fill_ref - float(entry["entry_price"])) / TICK)
        result[str(seconds)] = {"raw_trade_markout_ticks": raw,
                                "executable_quote_markout_ticks_from_signal_close": executable,
                                "conservative_actual_fill_markout_ticks": actual,
                                "quote_at_horizon": exit_quote}
    return result


def _episode_and_candidates(day: str, tape: np.ndarray, anchor: Mapping[str, Any]):
    open_ns = ny_open_ns(day)
    last_observation = et_clock_ns(day, 15, 55)
    session_end = baseline._session_windows(day)["NY"][1]
    trade_ts, trade_px, trade_sz, _ = _trade_arrays(tape, open_ns, session_end)
    bars = build_one_minute_bars(trade_ts, trade_px, trade_sz, open_ns, session_end)
    bar_map = {int(b["minute_index"]): b for b in bars}
    episodes: list[dict[str, Any]] = []
    bos_events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    if anchor["status"] != "VALID":
        return bars, episodes, bos_events, controls
    anchor_px = float(anchor["price"])
    detection_limit_ns = open_ns + DETECTION_END_MINUTE * MINUTE_NS
    bos_start_ns = open_ns + BOS_START_MINUTE * MINUTE_NS
    bos_end_ns = open_ns + BOS_END_MINUTE * MINUTE_NS
    last_entry_ns = open_ns + LAST_ENTRY_MINUTE * MINUTE_NS
    started_control = {1: False, -1: False}
    consumed_direction = {1: False, -1: False}  # Reversion trade direction.
    for displacement_direction in (1, -1):
        active = None
        armed = True
        for bar in bars:
            minute = int(bar["minute_index"])
            end_ns = int(bar["end_ns"])
            if end_ns <= open_ns + DETECTION_START_MINUTE * MINUTE_NS:
                continue
            close_dist = distance_from_anchor_ticks(anchor_px, float(bar["close"]), displacement_direction)
            qualifies = is_displaced_close(anchor_px, float(bar["close"]), displacement_direction)
            if active is not None:
                touch = _first_anchor_touch(trade_ts, trade_px, int(active["detected_at_ns"]), end_ns,
                                            anchor_px, displacement_direction)
                if touch is not None:
                    active["anchor_revisit_timestamp_ns"] = touch
                    active["status"] = "EXPIRED_ANCHOR_TOUCHED_BEFORE_BOS"
                    active["expired_at_ns"] = touch
                    active = None
                    armed = False
                    continue
                # Update the structural extreme only with trades at/after detection.
                lo = int(np.searchsorted(trade_ts, int(active["detected_at_ns"]), side="left"))
                hi = int(np.searchsorted(trade_ts, end_ns, side="left"))
                prices_since = trade_px[lo:hi]
                if len(prices_since):
                    if displacement_direction > 0:
                        active["adverse_extreme"] = max(float(active["adverse_extreme"]), float(np.max(prices_since)))
                    else:
                        active["adverse_extreme"] = min(float(active["adverse_extreme"]), float(np.min(prices_since)))

                reversion_direction = -displacement_direction
                within_detection = end_ns <= detection_limit_ns
                within_bos = bos_start_ns <= end_ns <= bos_end_ns
                # Frozen displacement-only control: first qualifying completed close
                # after 09:45 on the first episode, before anchor revisit.
                if (active["episode_index"] == 1 and not started_control[displacement_direction]
                        and within_bos and qualifies):
                    started_control[displacement_direction] = True
                    controls.append({"date": day, "period": period_for(day),
                        "episode_id": active["episode_id"], "displacement_direction": "UP" if displacement_direction > 0 else "DOWN",
                        "direction": "LONG" if reversion_direction > 0 else "SHORT",
                        "signal_timestamp_ns": end_ns, "signal_close": float(bar["close"]),
                        "anchor_price": anchor_px, "distance_to_anchor_ticks": close_dist,
                        "stop_price": structural_stop(displacement_direction, anchor_px, float(active["adverse_extreme"])),
                        "minute_index": minute, "control_rule": "first completed candle >=09:45 still displaced; no BOS"})
                if within_bos and end_ns > int(active["detected_at_ns"]):
                    bos = bos_for_bar(bar_map, minute, displacement_direction)
                    if bos["confirmed"]:
                        status = classify_bos_distance(close_dist, consumed_direction[reversion_direction])
                        event = {"date": day, "period": period_for(day),
                            "episode_id": active["episode_id"], "minute_index": minute,
                            "displacement_direction": "UP" if displacement_direction > 0 else "DOWN",
                            "direction": "LONG" if reversion_direction > 0 else "SHORT",
                            "signal_timestamp_ns": end_ns, "bos_candle_close": float(bar["close"]),
                            "bos_reference_level": bos["reference_level"],
                            "prior_minute_indices": bos["prior_minute_indices"],
                            "anchor_price": anchor_px, "distance_to_anchor_ticks": close_dist,
                            "displacement_close": active["displacement_close"],
                            "displacement_distance_ticks": active["displacement_distance_ticks"],
                            "adverse_extreme_through_bos": float(active["adverse_extreme"]),
                            "stop_price": structural_stop(displacement_direction, anchor_px, float(active["adverse_extreme"])),
                            "status": status}
                        bos_events.append(event)
                        active["bos_timestamp_ns"] = end_ns
                        active["bos_event_index"] = len(bos_events) - 1
                        active["status"] = status
                        active = None
                        armed = False
                        continue
                if end_ns >= bos_end_ns:
                    active["status"] = "EXPIRED_CONFIRMATION_WINDOW"
                    active["expired_at_ns"] = bos_end_ns
                    active = None
                    armed = False
                continue

            # Rearm only after a completed candle has returned inside threshold.
            if not armed:
                if not qualifies:
                    armed = True
                continue
            if end_ns > detection_limit_ns or not qualifies:
                continue
            episode_index = 1 + sum(e["displacement_direction_sign"] == displacement_direction for e in episodes)
            active = {"date": day, "period": period_for(day),
                "episode_id": f"{day}:{'UP' if displacement_direction > 0 else 'DOWN'}:{episode_index:02d}",
                "episode_index": episode_index, "displacement_direction_sign": displacement_direction,
                "displacement_direction": "UP" if displacement_direction > 0 else "DOWN",
                "detected_at_ns": end_ns, "displacement_candle_start_ns": int(bar["start_ns"]),
                "displacement_close": float(bar["close"]), "anchor_price": anchor_px,
                "displacement_distance_ticks": close_dist,
                "adverse_extreme": float(bar["close"]), "status": "PENDING_BOS_OR_EXPIRY"}
            episodes.append(active)
            # Do not use the displacement candle itself as post-detection data.
        if active is not None:
            active["status"] = "EXPIRED_CONFIRMATION_WINDOW"
            active["expired_at_ns"] = bos_end_ns

    # Mechanism outcomes use actual trade returns through the fixed 15:55 observation endpoint.
    for episode in episodes:
        start = int(episode["detected_at_ns"])
        revisit = _first_anchor_touch(trade_ts, trade_px, start, last_observation, anchor_px,
                                      int(episode["displacement_direction_sign"]))
        episode["anchor_revisit"] = revisit is not None
        episode["anchor_revisit_timestamp_ns"] = revisit
        episode["time_to_anchor_revisit_seconds"] = (revisit - start) / NS if revisit is not None else None
        lo = int(np.searchsorted(trade_ts, start, side="left"))
        hi = int(np.searchsorted(trade_ts, last_observation, side="left"))
        after = trade_px[lo:hi]
        sign = int(episode["displacement_direction_sign"])
        if len(after):
            episode["maximum_displacement_ticks_through_1555"] = float((np.max(after) - anchor_px) / TICK if sign > 0 else (anchor_px - np.min(after)) / TICK)
        else:
            episode["maximum_displacement_ticks_through_1555"] = None
        bos_ns = episode.get("bos_timestamp_ns")
        episode["opposing_bos_before_anchor_revisit"] = bool(bos_ns is not None and (revisit is None or int(bos_ns) < int(revisit)))
        episode["time_from_displacement_to_bos_seconds"] = (int(bos_ns) - start) / NS if bos_ns is not None else None
    return bars, episodes, bos_events, controls


def _to_jsonable(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}


def _daily_row(day: str, tape: np.ndarray, *, source_ok: bool) -> dict[str, Any]:
    open_ns = ny_open_ns(day)
    close_ns = baseline._session_windows(day)["NY"][1]
    end_observe = et_clock_ns(day, 15, 55)
    trades_ts, trades_px, trades_sz, _ = _trade_arrays(tape, open_ns, close_ns)
    anchor = opening_anchor(trades_ts, trades_px, open_ns)
    period = period_for(day)
    if anchor["status"] != "VALID":
        daily = {"date": day, "period": period, "open_anchor": None,
            "displacement_up_count": 0, "displacement_down_count": 0,
            "bos_short_count": 0, "bos_long_count": 0, "bos_confirmed_events": 0,
            "bos_executable_trades": 0, "bos_net_r": 0.0, "bos_win_count": 0,
            "bos_loss_count": 0, "bos_target_first": 0, "bos_stop_first": 0,
            "bos_end_of_day_exits": 0, "bos_censored": 0,
            "bos_median_mfe_r": None, "bos_median_mae_r": None,
            "control_trades": 0, "control_net_r": 0.0,
            "bos_minus_control_r": 0.0,
            "data_coverage_status": anchor["status"], "open_anchor_timestamp_ns": None}
        return {"daily": daily, "anchor": anchor, "bars": [], "episodes": [],
                "bos_events": [], "bos_trades": [], "controls": [], "control_trades": []}
    bars, episodes, bos_events, controls = _episode_and_candidates(day, tape, anchor)
    last_entry_ns = open_ns + LAST_ENTRY_MINUTE * MINUTE_NS
    deadline_ns = end_observe
    # Execute events chronologically, permitting one successful filled entry per side/date.
    bos_trades: list[dict[str, Any]] = []
    consumed_direction: set[str] = set()
    for ev in sorted(bos_events, key=lambda x: (x["signal_timestamp_ns"], x["direction"])):
        if ev["status"] != "BOS_CONFIRMED":
            ev["execution_status"] = ev["status"]
            continue
        if ev["direction"] in consumed_direction:
            ev["execution_status"] = "DIRECTION_ALREADY_TRADED"
            continue
        direction = 1 if ev["direction"] == "LONG" else -1
        trade = simulate_entry_and_path(tape, day=day, signal_ns=int(ev["signal_timestamp_ns"]),
            signal_close=float(ev["bos_candle_close"]), direction=direction,
            anchor=float(anchor["price"]), stop=float(ev["stop_price"]),
            last_entry_ns=last_entry_ns, exit_deadline_ns=deadline_ns, session_end_ns=close_ns)
        ev["execution_status"] = trade["status"]
        ev["markouts"] = _markouts(tape, signal_ns=int(ev["signal_timestamp_ns"]),
            signal_price=float(ev["bos_candle_close"]), direction=direction,
            entry=trade, session_end_ns=close_ns)
        if trade["status"] in ("EXECUTED", "CENSORED_NO_RELIABLE_EXIT_QUOTE"):
            consumed_direction.add(ev["direction"])
            bos_trades.append({**ev, **trade, "trade_group": "BOS_CONFIRMED_REVERSION",
                               "entry_stop_distance_dollars_per_contract": trade.get("structural_risk_ticks", 0) * TICK * ES_POINT_VALUE})
    control_trades: list[dict[str, Any]] = []
    control_consumed: set[str] = set()
    for ev in sorted(controls, key=lambda x: (x["signal_timestamp_ns"], x["direction"])):
        if ev["direction"] in control_consumed:
            continue
        direction = 1 if ev["direction"] == "LONG" else -1
        trade = simulate_entry_and_path(tape, day=day, signal_ns=int(ev["signal_timestamp_ns"]),
            signal_close=float(ev["signal_close"]), direction=direction,
            anchor=float(anchor["price"]), stop=float(ev["stop_price"]),
            last_entry_ns=last_entry_ns, exit_deadline_ns=deadline_ns, session_end_ns=close_ns)
        ev["execution_status"] = trade["status"]
        ev["markouts"] = _markouts(tape, signal_ns=int(ev["signal_timestamp_ns"]),
            signal_price=float(ev["signal_close"]), direction=direction,
            entry=trade, session_end_ns=close_ns)
        if trade["status"] in ("EXECUTED", "CENSORED_NO_RELIABLE_EXIT_QUOTE"):
            control_consumed.add(ev["direction"])
            control_trades.append({**ev, **trade, "trade_group": "DISPLACEMENT_ONLY_REVERSION_CONTROL"})
    complete_bos = [x for x in bos_trades if x.get("status") == "EXECUTED"]
    complete_control = [x for x in control_trades if x.get("status") == "EXECUTED"]
    def net_total(rows):
        return float(sum(float(x["net_r"]) for x in rows))
    def count_reason(rows, reason):
        return sum(x.get("outcome") == reason for x in rows)
    direction_count = lambda d: sum(x["direction"] == d for x in bos_events)
    daily = {
        "date": day, "period": period, "open_anchor": float(anchor["price"]),
        "open_anchor_timestamp_ns": int(anchor["timestamp_ns"]),
        "displacement_up_count": sum(x["displacement_direction_sign"] > 0 for x in episodes),
        "displacement_down_count": sum(x["displacement_direction_sign"] < 0 for x in episodes),
        "bos_short_count": direction_count("SHORT"), "bos_long_count": direction_count("LONG"),
        "bos_confirmed_events": len(bos_events), "bos_executable_trades": len(bos_trades),
        "bos_net_r": net_total(complete_bos),
        "bos_win_count": sum(float(x["net_pnl_usd"]) > 0 for x in complete_bos),
        "bos_loss_count": sum(float(x["net_pnl_usd"]) < 0 for x in complete_bos),
        "bos_target_first": count_reason(complete_bos, "TARGET"),
        "bos_stop_first": count_reason(complete_bos, "STOP"),
        "bos_end_of_day_exits": count_reason(complete_bos, "END_OF_DAY"),
        "bos_censored": sum(x.get("status") == "CENSORED_NO_RELIABLE_EXIT_QUOTE" for x in bos_trades),
        "bos_median_mfe_r": float(np.median([x["mfe_r"] for x in complete_bos])) if complete_bos else None,
        "bos_median_mae_r": float(np.median([x["mae_r"] for x in complete_bos])) if complete_bos else None,
        "control_trades": len(control_trades), "control_net_r": net_total(complete_control),
        "bos_minus_control_r": net_total(complete_bos) - net_total(complete_control),
        "data_coverage_status": "PASS" if source_ok else "FAIL",
    }
    return {"daily": daily, "anchor": anchor, "bars": bars, "episodes": episodes,
            "bos_events": bos_events, "bos_trades": bos_trades,
            "controls": controls, "control_trades": control_trades}


def _stats(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    executed = [t for t in trades if t.get("status") == "EXECUTED"]
    censored = sum(t.get("status") == "CENSORED_NO_RELIABLE_EXIT_QUOTE" for t in trades)
    filled = len(executed) + censored
    rs = [float(t["net_r"]) for t in executed]
    gross_rs = [float(t["gross_r"]) for t in executed]
    netpnl = [float(t["net_pnl_usd"]) for t in executed]
    positive, negative = [x for x in netpnl if x > 0], [x for x in netpnl if x < 0]
    def med(key):
        values = [float(t[key]) for t in executed if t.get(key) is not None]
        return float(np.median(values)) if values else None
    n = len(executed)
    return {
        "events": len(trades), "executable_trades": filled,
        "completed_trade_outcomes": n, "censored_after_entry": censored,
        "gross_avg_r": float(np.mean(gross_rs)) if n else None,
        "net_avg_r": float(np.mean(rs)) if n else None,
        "net_total_r": float(sum(rs)), "net_total_usd": float(sum(netpnl)),
        "profit_factor_net_dollars": float(sum(positive) / abs(sum(negative))) if negative else ("infinite" if positive else None),
        "win_rate_net": float(sum(x > 0 for x in netpnl) / n) if n else None,
        "target_before_stop_count": sum(t.get("outcome") == "TARGET" for t in executed),
        "stop_before_target_count": sum(t.get("outcome") == "STOP" for t in executed),
        "end_of_day_exit_count": sum(t.get("outcome") == "END_OF_DAY" for t in executed),
        "target_before_stop_frequency": sum(t.get("outcome") == "TARGET" for t in executed) / n if n else None,
        "median_mfe_ticks": med("mfe_ticks"), "median_mfe_r": med("mfe_r"),
        "median_mae_ticks_magnitude": med("mae_ticks"), "median_mae_r": med("mae_r"),
        "median_target_distance_ticks": med("target_distance_ticks"),
        "median_structural_risk_ticks": med("structural_risk_ticks"),
        "median_initial_target_r": med("initial_target_r_gross_geometry"),
        "median_time_to_target_seconds": med("time_to_target_seconds"),
        "median_time_to_stop_seconds": med("time_to_stop_seconds"),
        "anchor_touch_before_exit_count": sum(bool(t.get("anchor_touch_before_exit")) for t in executed),
    }


def _rank(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _group_robustness(days: Sequence[str], trades: Sequence[Mapping[str, Any]], daily_by_date: Mapping[str, float], seed: int) -> dict[str, Any]:
    values = [float(daily_by_date.get(d, 0.0)) for d in days]
    active = [d for d in days if any(t["date"] == d and t.get("status") == "EXECUTED" for t in trades)]
    positive = [d for d in active if daily_by_date.get(d, 0.0) > 0]
    negative = [d for d in active if daily_by_date.get(d, 0.0) < 0]
    by_week: dict[str, float] = defaultdict(float)
    for day, value in zip(days, values):
        d = date.fromisoformat(day)
        by_week[f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"] += value
    # Date-cluster bootstrap: sample dates, never individual trades.
    rng = np.random.default_rng(seed)
    matrix = np.asarray(values, dtype=np.float64)
    boot = np.empty(5000, dtype=np.float64)
    if len(matrix):
        for i in range(len(boot)):
            boot[i] = float(np.mean(matrix[rng.integers(0, len(matrix), len(matrix))]))
        ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    else:
        ci = [None, None]
    observed = abs(float(np.mean(values))) if values else 0.0
    exceed = 0
    for _ in range(20000):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values)) if values else np.asarray([])
        exceed += abs(float(np.mean(matrix * signs))) >= observed - 1e-15 if values else 0
    p = (exceed + 1) / 20001 if values else None
    lodo_values = [float(np.mean([v for j, v in enumerate(values) if j != i])) for i in range(len(values)) if len(values) > 1]
    active_weeks = [w for w, total in by_week.items() if any(date.fromisoformat(d).isocalendar().week == int(w.split("W")[1]) and daily_by_date.get(d, 0) != 0 for d in days)]
    lowo = []
    for week in by_week:
        remaining = [v for d, v in zip(days, values) if f"{date.fromisoformat(d).isocalendar().year}-W{date.fromisoformat(d).isocalendar().week:02d}" != week]
        if remaining:
            lowo.append(float(np.mean(remaining)))
    n_active_weeks = sum(any(f"{date.fromisoformat(d).isocalendar().year}-W{date.fromisoformat(d).isocalendar().week:02d}" == w and daily_by_date.get(d, 0) != 0 for d in days) for w in by_week)
    return {
        "date_unit": True, "calendar_dates": len(days), "active_trade_dates": len(active),
        "positive_trading_days": len(positive), "negative_trading_days": len(negative),
        "zero_net_active_days": len(active) - len(positive) - len(negative),
        "daily_net_r_mean": float(np.mean(values)) if values else None,
        "daily_net_r_median": float(np.median(values)) if values else None,
        "best_daily_net_r": max(values, default=None), "worst_daily_net_r": min(values, default=None),
        "best_3_days_net_r": float(sum(sorted(values, reverse=True)[:3])),
        "worst_3_days_net_r": float(sum(sorted(values)[:3])),
        "weekly_net_r": dict(sorted(by_week.items())),
        "date_cluster_bootstrap_mean_daily_net_r_95pct_ci": ci,
        "date_sign_flip_two_sided_p_sanity": p,
        "resampling_unit": "trading date (all trades on a date remain together)",
        "lodo": {"status": "COMPUTED" if len(active) >= 5 else "INSUFFICIENT_SAMPLE",
                 "sign_stable": (all(x >= 0 for x in lodo_values) if np.mean(values) >= 0 else all(x <= 0 for x in lodo_values)) if len(active) >= 5 else None,
                 "minimum_leave_one_date_out_mean": min(lodo_values) if lodo_values and len(active) >= 5 else None,
                 "maximum_leave_one_date_out_mean": max(lodo_values) if lodo_values and len(active) >= 5 else None},
        "lowo": {"status": "COMPUTED" if n_active_weeks >= 4 else "INSUFFICIENT_SAMPLE",
                 "active_weeks": n_active_weeks,
                 "minimum_leave_one_week_out_mean": min(lowo) if lowo and n_active_weeks >= 4 else None,
                 "maximum_leave_one_week_out_mean": max(lowo) if lowo and n_active_weeks >= 4 else None,
                 "sign_stable": (all(x >= 0 for x in lowo) if np.mean(values) >= 0 else all(x <= 0 for x in lowo)) if n_active_weeks >= 4 else None},
    }


def aggregate(daily_rows: Sequence[Mapping[str, Any]], bos_trades: Sequence[Mapping[str, Any]],
              control_trades: Sequence[Mapping[str, Any]], bos_events: Sequence[Mapping[str, Any]],
              episodes: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    periods = {"SPRING_2025": SPRING_DATES, "OCTOBER_2025": OCTOBER_DATES, "POOLED_DEV": tuple(dates)}
    out: dict[str, Any] = {}
    for period, group_dates in periods.items():
        btr = [x for x in bos_trades if x["date"] in group_dates]
        ctr = [x for x in control_trades if x["date"] in group_dates]
        bevents = [x for x in bos_events if x["date"] in group_dates]
        eps = [x for x in episodes if x["date"] in group_dates]
        rows = [x for x in daily_rows if x["date"] in group_dates]
        bdaily = {x["date"]: float(x["bos_net_r"]) for x in rows}
        cdaily = {x["date"]: float(x["control_net_r"]) for x in rows}
        out[period] = {
            "dates": len(group_dates),
            "open_anchor_valid_dates": sum(x["open_anchor"] is not None for x in rows),
            "displacement_episodes": len(eps),
            "upward_displacements": sum(x["displacement_direction_sign"] > 0 for x in eps),
            "downward_displacements": sum(x["displacement_direction_sign"] < 0 for x in eps),
            "anchor_revisited_episodes": sum(bool(x.get("anchor_revisit")) for x in eps),
            "anchor_revisit_rate": sum(bool(x.get("anchor_revisit")) for x in eps) / len(eps) if eps else None,
            "median_time_to_anchor_revisit_seconds": float(np.median([x["time_to_anchor_revisit_seconds"] for x in eps if x.get("time_to_anchor_revisit_seconds") is not None])) if any(x.get("time_to_anchor_revisit_seconds") is not None for x in eps) else None,
            "bos_before_anchor_revisit_count": sum(bool(x.get("opposing_bos_before_anchor_revisit")) for x in eps),
            "bos_events_confirmed": len(bevents),
            "bos_too_close_to_anchor": sum(x.get("status") == "BOS_TOO_CLOSE_TO_ANCHOR" for x in bevents),
            "bos_strategy_eligible_events": sum(x.get("status") == "BOS_CONFIRMED" for x in bevents),
            "bos_suppressed_after_prior_entry": sum(x.get("execution_status") == "DIRECTION_ALREADY_TRADED" for x in bevents),
            "bos_unfilled_execution_or_invalid_stop_events": sum(
                x.get("status") == "BOS_CONFIRMED" and x.get("execution_status") not in
                ("EXECUTED", "CENSORED_NO_RELIABLE_EXIT_QUOTE", "DIRECTION_ALREADY_TRADED")
                for x in bevents),
            "bos": _stats(btr), "control": _stats(ctr),
            "bos_minus_control_net_avg_r": (_stats(btr)["net_avg_r"] - _stats(ctr)["net_avg_r"]) if _stats(btr)["net_avg_r"] is not None and _stats(ctr)["net_avg_r"] is not None else None,
            "bos_minus_control_net_total_r": _stats(btr)["net_total_r"] - _stats(ctr)["net_total_r"],
            "daily_bos_robustness": _group_robustness(group_dates, btr, bdaily, 20261008 + list(periods).index(period)),
            "daily_control_robustness": _group_robustness(group_dates, ctr, cdaily, 20262008 + list(periods).index(period)),
        }
    return out


def _mechanism_analysis(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mags = [float(e["maximum_displacement_ticks_through_1555"]) for e in episodes if e.get("maximum_displacement_ticks_through_1555") is not None]
    revisit = [float(bool(e.get("anchor_revisit"))) for e in episodes if e.get("maximum_displacement_ticks_through_1555") is not None]
    times = [float(e["time_to_anchor_revisit_seconds"]) for e in episodes if e.get("time_to_anchor_revisit_seconds") is not None and e.get("maximum_displacement_ticks_through_1555") is not None]
    time_mags = [float(e["maximum_displacement_ticks_through_1555"]) for e in episodes if e.get("time_to_anchor_revisit_seconds") is not None and e.get("maximum_displacement_ticks_through_1555") is not None]
    bos_before = [float(bool(e.get("opposing_bos_before_anchor_revisit"))) for e in episodes]
    return {
        "episodes": len(episodes),
        "anchor_revisited_count": sum(bool(e.get("anchor_revisit")) for e in episodes),
        "anchor_revisited_fraction": float(np.mean(revisit)) if revisit else None,
        "median_seconds_to_anchor_revisit": float(np.median(times)) if times else None,
        "opposing_bos_before_anchor_revisit_fraction": float(np.mean(bos_before)) if bos_before else None,
        "median_max_displacement_ticks": float(np.median(mags)) if mags else None,
        "spearman_max_displacement_vs_anchor_revisit": float(np.corrcoef(_rank(mags), _rank(revisit))[0, 1]) if len(mags) >= 3 and np.std(revisit) > 0 else None,
        "spearman_max_displacement_vs_revisit_time": float(np.corrcoef(_rank(time_mags), _rank(times))[0, 1]) if len(times) >= 3 and np.std(times) > 0 else None,
        "analysis_note": "Magnitude is retained continuously; no magnitude bins/thresholds were searched. Correlations are descriptive and episode-clustered, not causal or inferential.",
    }


def _validate_sources(data_root: Path):
    paths, rows = native._source_catalog(data_root)
    if tuple(SPRING_DATES) != tuple(native.SPRING_DATES) or len(SPRING_DATES) != 35 or len(OCTOBER_DATES) != 19 or len(TARGET_DATES) != 54:
        raise FairPriceStudyError("authoritative date-list counts do not match 35 Spring + 19 October")
    target_set = set(TARGET_DATES)
    if any(day not in paths for day in target_set) or len(target_set) != 54:
        raise FairPriceStudyError("source manifest does not cover the exact 54 target dates")
    return paths, rows


def _load_target_tape(day: str, source_sha: str):
    path = native._tape_path(day)
    tape, metadata = native._load_tape(day, path, source_sha)
    if metadata.get("tape_version") != TAPE_VERSION or not metadata.get("bbo_path_complete"):
        raise FairPriceStudyError(f"candidate tape is not BBO-complete V2: {day}")
    ts = np.asarray(tape["timestamp_ns"], dtype=np.int64)
    if len(ts) == 0 or np.any(np.diff(ts) < 0):
        raise FairPriceStudyError(f"empty or noncausally ordered tape: {day}")
    if not ("execution_price" in tape.dtype.names and "execution_size" in tape.dtype.names and "bid" in tape.dtype.names and "ask" in tape.dtype.names):
        raise FairPriceStudyError(f"candidate tape is missing trade or BBO fields: {day}")
    return path, tape, metadata


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False,
                                        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode() + b"\n")
    os.replace(tmp, path)


def _run_one(day: str, tape: np.ndarray, source_ok: bool = True):
    return _daily_row(day, tape, source_ok=source_ok)


def run(*, data_root: Path = DATA_ROOT, output_root: Path = OUT_ROOT,
        smoke_date: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    if smoke_date is not None and smoke_date not in TARGET_DATES:
        raise FairPriceStudyError(f"smoke date is not in the authoritative 54-date list: {smoke_date}")
    paths, source_rows = _validate_sources(data_root)
    target_dates = [smoke_date] if smoke_date else list(TARGET_DATES)
    daily_rows, all_episodes, all_bos_events, all_bos_trades, all_controls, all_control_trades = [], [], [], [], [], []
    coverage = {"dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES", "status": "PASS",
        "periods": {"SPRING_2025": list(SPRING_DATES), "OCTOBER_2025": list(OCTOBER_DATES)},
        "target_dates": list(TARGET_DATES), "dependency_dates_validated_by_manifest": list(native.DEPENDENCY_DATES),
        "MBO_used": False, "MES_market_data_used": False, "raw_data_downloaded": False,
        "source_manifest": {"path": str(data_root / baseline.MANIFEST_NAME),
                            "sha256": _sha(data_root / baseline.MANIFEST_NAME)},
        "source_files": {}, "candidate_tapes": {}}
    for i, day in enumerate(target_dates, 1):
        source_row = source_rows[day]
        path, tape_path, metadata = paths[day], None, None
        tape_path, tape, metadata = _load_target_tape(day, str(source_row["sha256"]))
        coverage["source_files"][day] = {"path": str(path), "sha256": source_row["sha256"],
            "bytes": path.stat().st_size, "symbol": source_row["symbol"], "schema": source_row["schema"],
            "category": source_row["category"], "dataset": "GLBX.MDP3",
            "record_count_manifest": source_row.get("verification", {}).get("record_count")}
        coverage["candidate_tapes"][day] = {"path": str(tape_path), "sha256": native._sha(tape_path),
            "source_sha256": metadata["source_sha256"], "semantic_sha256": metadata["semantic_sha256"],
            "tape_version": metadata["tape_version"], "event_count": int(metadata["event_count"]),
            "execution_event_count": int(metadata["execution_event_count"]),
            "bbo_event_count": int(metadata["bbo_event_count"]), "bbo_path_complete": metadata["bbo_path_complete"]}
        result = _run_one(day, tape)
        daily_rows.append(result["daily"])
        all_episodes.extend(result["episodes"])
        all_bos_events.extend(result["bos_events"])
        all_bos_trades.extend(result["bos_trades"])
        all_controls.extend(result["controls"])
        all_control_trades.extend(result["control_trades"])
        if smoke_date:
            return {"status": "SMOKE_PASS", "smoke_date": day, "daily": result["daily"],
                "displacements": len(result["episodes"]), "bos_events": len(result["bos_events"]),
                "bos_fills": len(result["bos_trades"]), "control_fills": len(result["control_trades"]),
                "elapsed_seconds": time.monotonic() - started}
        print(f"FAIR_PRICE_DATE_COMPLETE={i}/{len(target_dates)} date={day} displacements={len(result['episodes'])} bos={len(result['bos_events'])} bos_fills={len(result['bos_trades'])} control_fills={len(result['control_trades'])}", flush=True)

    if len(daily_rows) != 54 or set(x["date"] for x in daily_rows) != set(TARGET_DATES):
        raise FairPriceStudyError("full run did not produce exactly the 54 authoritative target dates")
    periods = aggregate(daily_rows, all_bos_trades, all_control_trades, all_bos_events, all_episodes, TARGET_DATES)
    if len(all_episodes) != (periods["SPRING_2025"]["displacement_episodes"]
                             + periods["OCTOBER_2025"]["displacement_episodes"]):
        raise FairPriceStudyError("period episode aggregation mismatch")
    mechanism = {"ALL": _mechanism_analysis(all_episodes),
        "SPRING_2025": _mechanism_analysis([e for e in all_episodes if e["date"] in SPRING_DATES]),
        "OCTOBER_2025": _mechanism_analysis([e for e in all_episodes if e["date"] in OCTOBER_DATES])}

    def window_result(period):
        return periods[period]
    bos_stats, control_stats = periods["POOLED_DEV"]["bos"], periods["POOLED_DEV"]["control"]
    enough = bos_stats["completed_trade_outcomes"] >= 10 and control_stats["completed_trade_outcomes"] >= 10
    economics_improved = (enough and bos_stats["net_avg_r"] is not None and control_stats["net_avg_r"] is not None
        and bos_stats["net_avg_r"] > control_stats["net_avg_r"]
        and (bos_stats["target_before_stop_frequency"] or 0) >= (control_stats["target_before_stop_frequency"] or 0))
    if not enough:
        bos_control_decision: bool | str = "INSUFFICIENT"
        causal_information: bool | str = "INSUFFICIENT"
    else:
        bos_control_decision = bool(economics_improved)
        causal_information = bool(economics_improved)

    period_signs = [math.copysign(1, periods[p]["bos"]["net_avg_r"]) if periods[p]["bos"]["net_avg_r"] not in (None, 0) else 0 for p in ("SPRING_2025", "OCTOBER_2025")]
    pooled_bos_robustness = periods["POOLED_DEV"]["daily_bos_robustness"]
    pooled_date_ci = pooled_bos_robustness["date_cluster_bootstrap_mean_daily_net_r_95pct_ci"]
    date_robust_positive = bool(pooled_date_ci and pooled_date_ci[0] is not None and pooled_date_ci[0] > 0
                                and pooled_bos_robustness["lodo"]["sign_stable"] is True)
    execution_hurdle_cleared = bool(
        bos_stats["completed_trade_outcomes"] >= 10
        and bos_stats["net_avg_r"] is not None and bos_stats["net_avg_r"] > 0
        and isinstance(bos_stats["profit_factor_net_dollars"], (int, float))
        and bos_stats["profit_factor_net_dollars"] > 1.0)
    if bos_stats["completed_trade_outcomes"] < 10:
        primary_decision, next_step = "INSUFFICIENT_SAMPLE", "FREEZE_HYPOTHESIS_AND_REQUIRE_FRESH_DATA"
    elif execution_hurdle_cleared and economics_improved and period_signs[0] > 0 and period_signs[1] > 0 and date_robust_positive:
        primary_decision, next_step = "FAIR_PRICE_BOS_MODEL_PROMISING", "FREEZE_ONE_BOS_STRATEGY_FOR_FRESH_VALIDATION"
    elif any(x["net_avg_r"] is not None and x["net_avg_r"] > 0 for x in (periods["SPRING_2025"]["bos"], periods["OCTOBER_2025"]["bos"])):
        primary_decision, next_step = "FAIR_PRICE_MECHANISM_POSSIBLE_BUT_UNCONFIRMED", "FREEZE_HYPOTHESIS_AND_REQUIRE_FRESH_DATA"
    elif bos_stats["net_avg_r"] is not None and bos_stats["net_avg_r"] <= 0:
        primary_decision, next_step = "FAIR_PRICE_BOS_MODEL_FAILED", "STOP_THIS_FAIR_PRICE_BOS_BRANCH"
    else:
        primary_decision, next_step = "INSUFFICIENT_SAMPLE", "FREEZE_HYPOTHESIS_AND_REQUIRE_FRESH_DATA"

    if period_signs[0] > 0 and period_signs[1] > 0:
        period_compatibility = "COMPATIBLE_POSITIVE"
    elif period_signs[0] < 0 and period_signs[1] < 0:
        period_compatibility = "COMPATIBLE_NONPOSITIVE"
    elif period_signs[0] < 0 < period_signs[1]:
        period_compatibility = "SPRING_CONTRADICTS_OCTOBER"
    elif period_signs[1] < 0 < period_signs[0]:
        period_compatibility = "OCTOBER_CONTRADICTS_SPRING"
    else:
        period_compatibility = "NOT_COMPARABLE_INSUFFICIENT_SAMPLE"

    coverage.update({"status": "PASS", "validated_target_date_count": len(daily_rows),
        "opening_anchor_valid_dates": sum(x["open_anchor"] is not None for x in daily_rows),
        "periods_read": ["SPRING_2025", "OCTOBER_2025"], "validation_is_not_untouched_oos": True})
    OUT_ROOT_LOCAL = output_root
    OUT_ROOT_LOCAL.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_ROOT_LOCAL / "study-config.json", {"config": CONFIG, "config_sha256": CONFIG_SHA256,
        "spring_dates": list(SPRING_DATES), "october_dates": list(OCTOBER_DATES),
        "spring_date_count": len(SPRING_DATES), "october_date_count": len(OCTOBER_DATES),
        "total_target_dates": len(TARGET_DATES), "optimization_performed": False,
        "actual_jj_private_strategy_replicated": False})
    _write_json(OUT_ROOT_LOCAL / "source-coverage.json", coverage)
    _write_jsonl_gz(OUT_ROOT_LOCAL / "displacement-events.jsonl.gz", all_episodes)
    _write_jsonl_gz(OUT_ROOT_LOCAL / "bos-events.jsonl.gz", all_bos_events)
    _write_jsonl_gz(OUT_ROOT_LOCAL / "bos-trades.jsonl.gz", all_bos_trades)
    _write_jsonl_gz(OUT_ROOT_LOCAL / "displacement-control-trades.jsonl.gz", all_control_trades)
    with (OUT_ROOT_LOCAL / "daily-results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(daily_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(daily_rows)
    _write_json(OUT_ROOT_LOCAL / "spring-results.json", periods["SPRING_2025"])
    _write_json(OUT_ROOT_LOCAL / "october-results.json", periods["OCTOBER_2025"])
    _write_json(OUT_ROOT_LOCAL / "control-comparison.json", {
        "BOS_CONFIRMED_REVERSION": periods["POOLED_DEV"]["bos"],
        "DISPLACEMENT_ONLY_REVERSION_CONTROL": periods["POOLED_DEV"]["control"],
        "by_period": {p: {"bos": periods[p]["bos"], "control": periods[p]["control"],
            "bos_minus_control_net_avg_r": periods[p]["bos_minus_control_net_avg_r"],
            "bos_minus_control_net_total_r": periods[p]["bos_minus_control_net_total_r"]}
            for p in ("SPRING_2025", "OCTOBER_2025")},
        "entry_timing_caveat": CONFIG["control"]["different_entry_time_and_price_caveat"],
        "bos_vs_control_economic_improvement": bos_control_decision,
        "execution_hurdle_cleared": execution_hurdle_cleared,
        "causal_bos_information_added": causal_information})
    _write_json(OUT_ROOT_LOCAL / "execution-analysis.json", {
        "bos": {"pooled": periods["POOLED_DEV"]["bos"], "spring": periods["SPRING_2025"]["bos"], "october": periods["OCTOBER_2025"]["bos"]},
        "control": {"pooled": periods["POOLED_DEV"]["control"], "spring": periods["SPRING_2025"]["control"], "october": periods["OCTOBER_2025"]["control"]},
        "period_metric_objects": {p: {"bos": periods[p]["bos"], "control": periods[p]["control"]} for p in periods},
        "forward_markout_horizons_seconds": list(FORWARD_HORIZONS_SECONDS),
        "markout_groups": _forward_summary(all_bos_trades, all_control_trades),
        "execution_cost_contract": CONFIG["fees"], "entry_exit_model": CONFIG["entry"],
        "censored_after_entry_bos_count": bos_stats["censored_after_entry"],
        "censored_after_entry_control_count": control_stats["censored_after_entry"]})
    robustness = {"BOS_CONFIRMED_REVERSION": {p: periods[p]["daily_bos_robustness"] for p in periods},
                  "DISPLACEMENT_ONLY_REVERSION_CONTROL": {p: periods[p]["daily_control_robustness"] for p in periods},
                  "mechanism": mechanism,
                  "sample_unit": "trading date; no trade-level independent-observation inference"}
    _write_json(OUT_ROOT_LOCAL / "robustness.json", robustness)

    summary = {
        "study_id": STUDY_ID, "run_id": RUN_ID, "status": "COMPLETE", "dataset": "SPRING_2025 + OCTOBER_2025",
        "spring_dates": list(SPRING_DATES), "october_dates": list(OCTOBER_DATES),
        "total_eligible_dates": len(TARGET_DATES),
        "open_anchor_valid_dates": coverage["opening_anchor_valid_dates"],
        "total_displacement_episodes": len(all_episodes),
        "upward_displacements": sum(x["displacement_direction_sign"] > 0 for x in all_episodes),
        "downward_displacements": sum(x["displacement_direction_sign"] < 0 for x in all_episodes),
        "bos_confirmed_events": len(all_bos_events),
        "bos_too_close_to_anchor": sum(x.get("status") == "BOS_TOO_CLOSE_TO_ANCHOR" for x in all_bos_events),
        "bos_executable_trades": bos_stats["executable_trades"],
        "control_executable_trades": control_stats["executable_trades"],
        "bos_metrics": bos_stats, "control_metrics": control_stats,
        "bos_target_before_stop": bos_stats["target_before_stop_count"],
        "bos_stop_before_target": bos_stats["stop_before_target_count"],
        "bos_end_of_day_exits": bos_stats["end_of_day_exit_count"],
        "bos_win_rate": bos_stats["win_rate_net"], "bos_gross_avg_r": bos_stats["gross_avg_r"],
        "bos_net_avg_r": bos_stats["net_avg_r"], "bos_net_total_r": bos_stats["net_total_r"],
        "bos_profit_factor": bos_stats["profit_factor_net_dollars"],
        "bos_median_mfe_r": bos_stats["median_mfe_r"], "bos_median_mae_r": bos_stats["median_mae_r"],
        "bos_median_stop_ticks": bos_stats["median_structural_risk_ticks"],
        "bos_median_target_distance_ticks": bos_stats["median_target_distance_ticks"],
        "bos_median_initial_target_r": bos_stats["median_initial_target_r"],
        "bos_median_time_to_target_seconds": bos_stats["median_time_to_target_seconds"],
        "control_net_avg_r": control_stats["net_avg_r"], "control_net_total_r": control_stats["net_total_r"],
        "bos_vs_control_economic_improvement": bos_control_decision,
        "spring_results": periods["SPRING_2025"], "october_results": periods["OCTOBER_2025"],
        "spring_october_compatibility": period_compatibility,
        "causal_bos_information_added": causal_information,
        "execution_hurdle_cleared": execution_hurdle_cleared,
        "date_robust_positive": date_robust_positive,
        "candidate_hypothesis": "NONE" if primary_decision in ("FAIR_PRICE_BOS_MODEL_FAILED", "INSUFFICIENT_SAMPLE") else "FIXED_32_TICK_DISPLACEMENT_PLUS_TWO_PRIOR_CANDLE_BOS_TO_0930_ANCHOR",
        "primary_decision": primary_decision, "next_step": next_step,
        "actual_jj_private_strategy_replicated": False, "price_action_only_signals": True,
        "mbp10_used_for_execution": True, "mbo_required": False,
        "delta_filters_used": False, "regime_filters_used": False,
        "parameter_optimization_performed": False, "2026_data_accessed": False,
        "final_oos_accessed": False, "data_downloaded": False,
        "production_config_changed": False, "commit_performed": False,
        "config_sha256": CONFIG_SHA256, "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(OUT_ROOT_LOCAL / "summary.json", summary)
    report = _make_report(summary, daily_rows, all_episodes, all_bos_events, all_bos_trades, all_control_trades, mechanism, periods)
    (OUT_ROOT_LOCAL / "report.md").write_text(report, encoding="utf-8")
    outputs = {p.name: _sha(p) for p in sorted(OUT_ROOT_LOCAL.iterdir()) if p.is_file() and p.name not in {"artifact-hashes.json", "run-manifest.json"}}
    _write_json(OUT_ROOT_LOCAL / "run-manifest.json", {
        "status": "COMPLETE", "study_id": STUDY_ID, "run_id": RUN_ID,
        "config_sha256": CONFIG_SHA256, "dataset": "GLBX.MDP3", "schema": "mbp-10",
        "source_sha256_by_date": {d: source_rows[d]["sha256"] for d in TARGET_DATES},
        "candidate_tape_sha256_by_date": {d: coverage["candidate_tapes"][d]["sha256"] for d in TARGET_DATES},
        "output_sha256_by_name": outputs,
        "complete_target_dates": list(TARGET_DATES), "no_2026_access": True,
        "no_download": True, "no_optimization": True})
    artifact_hashes = {p.name: _sha(p) for p in sorted(OUT_ROOT_LOCAL.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    _write_json(OUT_ROOT_LOCAL / "artifact-hashes.json", {"status": "HASHED", "files": artifact_hashes})
    return summary


def _forward_summary(bos_trades, control_trades):
    result = {}
    for name, rows in (("BOS", bos_trades), ("CONTROL", control_trades)):
        groups = [t for t in rows if t.get("status") == "EXECUTED"]
        result[name] = {str(h): {
            metric: _median([t.get("markouts", {}).get(str(h), {}).get(field) for t in groups])
            for metric, field in (("raw_trade_median_ticks", "raw_trade_markout_ticks"),
                                  ("executable_quote_median_ticks", "executable_quote_markout_ticks_from_signal_close"),
                                  ("actual_fill_median_ticks", "conservative_actual_fill_markout_ticks"))}
            | {"n": sum(t.get("markouts", {}).get(str(h), {}).get("raw_trade_markout_ticks") is not None for t in groups)}
            for h in FORWARD_HORIZONS_SECONDS}
    return result


def _median(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(clean)) if clean else None


def _make_report(summary, daily_rows, episodes, bos_events, bos_trades, control_trades, mechanism, periods):
    lines = [f"# {STUDY_ID}", "", "## Preregistered scope", "",
        "Mechanical public-theory operationalization; not JJ Simon's private strategy. Spring is exploratory; October is secondary development compatibility and is not untouched OOS.",
        "Native ES MBP-10 only, using the sealed source-bound V2 BBO/trade candidate tapes. No raw DBN rebuild, download, optimization, or 2026/OOS access.", "",
        "## Primary aggregate results", ""]
    for period in ("SPRING_2025", "OCTOBER_2025", "POOLED_DEV"):
        x = periods[period]
        lines.append(f"- **{period}:** {x['dates']} dates; anchors {x['open_anchor_valid_dates']}; displacement episodes {x['displacement_episodes']} (up {x['upward_displacements']}, down {x['downward_displacements']}); anchor revisits {x['anchor_revisited_episodes']} ({_fmt(x['anchor_revisit_rate'], pct=True)}); BOS confirmations {x['bos_events_confirmed']} (too-close {x['bos_too_close_to_anchor']}); BOS fills {x['bos']['executable_trades']} net R {x['bos']['net_total_r']:+.3f}, avg { _fmt(x['bos']['net_avg_r'])}, PF {_fmt(x['bos']['profit_factor_net_dollars'])}; control fills {x['control']['executable_trades']} net R {x['control']['net_total_r']:+.3f}, avg {_fmt(x['control']['net_avg_r'])}.")
    lines += ["", "Interpretation uses executed, fee-inclusive trades as the economic unit; date-cluster robustness is reported below. Censored positions are excluded from win/net totals and counted separately.", "",
        "## BOS vs displacement-only control", "",
        "The control is first qualifying completed candle at/after 09:45 while the first directional displacement remains active. It often enters at a different time/price; this compares the full BOS timing rule, not an isolated BOS indicator.", "",
        "| Period | BOS fills | BOS net avg R | BOS target-first | BOS median MFE / MAE R | Control fills | Control net avg R | Δ avg R |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for period in ("SPRING_2025", "OCTOBER_2025", "POOLED_DEV"):
        x = periods[period]; b, c = x["bos"], x["control"]
        delta = x["bos_minus_control_net_avg_r"]
        lines.append(f"| {period} | {b['executable_trades']} | {_fmt(b['net_avg_r'])} | {_fmt(b['target_before_stop_frequency'], pct=True)} | {_fmt(b['median_mfe_r'])} / {_fmt(b['median_mae_r'])} | {c['executable_trades']} | {_fmt(c['net_avg_r'])} | {_fmt(delta)} |")
    lines += ["", f"BOS/control relative comparison: **{summary['bos_vs_control_economic_improvement']}**; positive net execution hurdle cleared: **{summary['execution_hurdle_cleared']}**; causal BOS information added relative to the control: **{summary['causal_bos_information_added']}**.", "",
        "## Per-date results (all 54 dates, including zero-event dates)", "",
        "| Date | Period | Anchor | Up/down displacement | BOS short/long | BOS fills | BOS net R | Wins/losses | Target/stop first | Control fills | Control net R | Δ R | Coverage |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in daily_rows:
        lines.append(f"| {r['date']} | {r['period']} | {_fmt(r['open_anchor'])} | {r['displacement_up_count']}/{r['displacement_down_count']} | {r['bos_short_count']}/{r['bos_long_count']} | {r['bos_executable_trades']} | {r['bos_net_r']:+.3f} | {r['bos_win_count']}/{r['bos_loss_count']} | {r['bos_target_first']}/{r['bos_stop_first']} | {r['control_trades']} | {r['control_net_r']:+.3f} | {r['bos_minus_control_r']:+.3f} | {r['data_coverage_status']} |")
    lines += ["", "## Displacement/reversion mechanism", ""]
    for period, result in mechanism.items():
        lines.append(f"- {period}: n={result['episodes']}; anchor revisit fraction={_fmt(result['anchor_revisited_fraction'], pct=True)}; median revisit seconds={_fmt(result['median_seconds_to_anchor_revisit'])}; BOS-before-revisit fraction={_fmt(result['opposing_bos_before_anchor_revisit_fraction'], pct=True)}; median maximum displacement={_fmt(result['median_max_displacement_ticks'])} ticks; Spearman displacement/revisit={_fmt(result['spearman_max_displacement_vs_anchor_revisit'])}; displacement/time-to-revisit={_fmt(result['spearman_max_displacement_vs_revisit_time'])}.")
    lines += ["", "Magnitude diagnostics are continuous and descriptive; no alternate threshold or magnitude bins were tested.", "",
        "## Date-cluster robustness", ""]
    for group, x in (("BOS", periods["POOLED_DEV"]["daily_bos_robustness"]), ("CONTROL", periods["POOLED_DEV"]["daily_control_robustness"])):
        lines.append(f"- {group}: {x['positive_trading_days']} positive / {x['negative_trading_days']} negative active dates; worst 1/3 daily net R {_fmt(x['worst_daily_net_r'])}/{x['worst_3_days_net_r']:+.3f}; best 1/3 {_fmt(x['best_daily_net_r'])}/{x['best_3_days_net_r']:+.3f}; mean daily 95% date-bootstrap CI {x['date_cluster_bootstrap_mean_daily_net_r_95pct_ci']}; day sign-flip sanity p={x['date_sign_flip_two_sided_p_sanity']}; LODO={x['lodo']['status']}; LOWO={x['lowo']['status']}.")
    lines += ["", "## Execution and costs", "", f"ES fee: ${ES_COMMISSION:.2f}/side/contract; point value ${ES_POINT_VALUE:.2f}/point. Entry/exit include displayed executable side plus one adverse tick, and the initial risk denominator includes the adverse stop fill and round-trip fees.",
        "Stop is checked before target. Target is an executable quote condition toward the fixed open; no passive limit fill is assumed. A missing post-15:55 executable exit remains censored, not a win.", "",
        f"## Decision: {summary['primary_decision']}", "", f"Next step: `{summary['next_step']}`.", "",
        "## Scope flags", "", "- `ACTUAL_JJ_PRIVATE_STRATEGY_REPLICATED=false`", "- `PRICE_ACTION_ONLY_SIGNALS=true`", "- `MBP10_USED_FOR_EXECUTION=true`", "- `MBO_REQUIRED=false`", "- `DELTA_FILTERS_USED=false`", "- `REGIME_FILTERS_USED=false`", "- `PARAMETER_OPTIMIZATION_PERFORMED=false`", "- `2026_DATA_ACCESSED=false`", "- `FINAL_OOS_ACCESSED=false`", "- `DATA_DOWNLOADED=false`", "- `PRODUCTION_CONFIG_CHANGED=false`", "- `COMMIT_PERFORMED=false`", ""]
    return "\n".join(lines)


def _fmt(value, pct=False):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.1%}" if pct else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke-date", help="Run one authorized Spring/October date only; writes no final artifacts.")
    args = parser.parse_args(argv)
    try:
        result = run(data_root=args.data_root, output_root=args.output_root, smoke_date=args.smoke_date)
    except (FairPriceStudyError, native.VacuumStudyError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({"status": result["status"], "primary_decision": result.get("primary_decision"),
        "smoke_date": result.get("smoke_date"),
        "smoke_displacements": result.get("displacements"),
        "smoke_bos_events": result.get("bos_events"),
        "smoke_bos_fills": result.get("bos_fills"),
        "smoke_control_fills": result.get("control_fills"),
        "total_displacement_episodes": result.get("total_displacement_episodes"),
        "bos_executable_trades": result.get("bos_executable_trades"),
        "elapsed_seconds": result.get("elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
