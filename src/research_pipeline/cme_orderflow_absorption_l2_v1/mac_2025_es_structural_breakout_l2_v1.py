"""Preregistered native-ES 30-minute RTH opening-range breakout event study.

Structural events, not L2 thresholds, define the sample. October is secondary
DEV compatibility; no optimization, MES feed, strategy PnL, or 2026 data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_es_flow_momentum_v1 as flow
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_absorption_relative_normalization as relative

RUN_ID = "CMEOrderflow_ES_STRUCTURAL_BREAKOUT_L2_EVENT_STUDY_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
TICK = 0.25
HORIZONS_MS = (250, 500, 1000, 2000, 5000, 10000, 30000, 60000)
EXCURSION_MS = (1000, 2000, 5000, 10000, 30000, 60000)
BARRIERS = ((2, 2), (4, 4), (8, 4), (8, 8), (12, 6), (16, 8))
PRICE_KEYS = ("price_velocity_10s", "trend_efficiency_10s", "realized_volatility_10s",
              "opening_range_width_ticks", "overshoot_ticks")
ANCHOR_L2_KEYS = ("mlofi_500ms", "depth_depletion", "impact_per_flow_10s", "withdrawal_500ms")
ALL_L2_KEYS = ANCHOR_L2_KEYS + ("refill_recovery_500ms",)
CONFIG = {
    "version": 1, "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
    "roles": {"SPRING_2025": "PRIMARY_DISCOVERY", "OCTOBER_2025": "SECONDARY_DEV_COMPATIBILITY"},
    "native_dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES",
    "opening_range": "09:30:00 ET inclusive to 10:00:00 ET exclusive; actual trade prices only",
    "breakout": "first actual trade strictly beyond OR high/low per direction, 10:00:00 ET inclusive to 16:00:00 ET exclusive",
    "event_anchor": "first valid canonical BBO state at or after breakout trade row in tape order, same NY session; L2 controls use strictly pre-event native state",
    "feature_windows": {"mlofi_ms": [500, 2000], "depth_baseline_seconds": 30,
                        "depth_sample_ms": 50, "impact_seconds": 10, "withdrawal_ms": 500,
                        "price_control_seconds": 10, "refill_delay_ms": 500},
    "feature_formulas": {
        "mlofi": "direction*sum(price-keyed inverse-rank TOP5 MLOFI in trailing window)/contemporaneous weighted TOP5 depth denominator",
        "depth_depletion": "1-current opposing TOP5 depth/median of 50ms as-of opposing depth over prior 30s",
        "impact_per_flow": "directional midpoint move over prior 10s in ticks/absolute 10s normalized MLOFI integral; missing when flow magnitude <=1e-6",
        "withdrawal": "(opposing displayed TOP5 decreases-in-nontrade-updates minus increases)/(sum decreases+increases); aggregate-depth proxy, not exact order cancellation IDs",
        "refill": "(opposing depth as-of event+500ms minus event opposing depth)/(prior 30s median minus event depth); missing unless deficit >0",
        "price_controls": "causal 10s directional mid velocity, 10s trend efficiency, sqrt(sum squared 500ms mid changes), OR width, event trade overshoot",
    },
    "support": "event feature above its strictly prior-date median means supportive; for refill below prior-date median means weak refill; no outcome input",
    "buckets": "terciles from strictly prior-date values; bucket N<10 insufficient; descriptive quintiles only if >=90 valid observations",
    "delayed_clock": "refill observed through event+500ms; all refill-conditioned outcomes begin at event+500ms",
    "barriers": [{"favorable_ticks": a, "adverse_ticks": b} for a, b in BARRIERS],
    "barrier_max_horizons_ms": [10000, 60000],
    "topology": "within 60s priority: opposite actual trade beyond other OR boundary; subsequent actual trade back inside OR (LONG <= OR high, SHORT >= OR low); +4 midpoint ticks before -2 within 10s immediate continuation; +2 then -2 within 10s small continuation then failure; neither +/-2 within 10s stagnation; else unclassified",
    "price_control_matching": "same period and direction, exclude self; nearest by fixed scaled squared distance of velocity/2, efficiency/0.25, volatility/4, OR width/20, overshoot/4; no future outcome or L2 input; with replacement",
    "permutation": "1000 deterministic seed-20251005 feature-label shuffles within period x direction; no date-level shuffle because <=1 event per date/direction",
    "execution": "first valid quote at/after event+2ms, LONG ask+1 tick / SHORT bid-1 tick; exit-side bid/ask, no stop or target; commissions not included in tick markouts",
    "minimum_bucket_n": 10, "primary_tercile_outcome_ms": 10000,
    "decision": "PROMISING only if actual-fill 10s mean positive in both periods, base raw 10s positive in both, at least one anchor L2 high-low Spring effect positive and October nonnegative, matched price-only excess positive, and positive-date majority; FAILED if base raw and actual 10s nonpositive in both and no positive compatible L2 effect; otherwise MIXED; insufficient bucket samples are not evidence",
    "no_optimization": True,
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CONFIG_SHA256 = _hash(CONFIG)
CHECKPOINT_VERSION = "structural-breakout-l2-date-v1"


class BreakoutStudyError(RuntimeError):
    pass


def _period(day: str) -> str:
    if day in flow.SPRING_DATES:
        return "SPRING_2025"
    if day in flow.OCTOBER_DATES:
        return "OCTOBER_2025"
    raise BreakoutStudyError(f"ineligible target date: {day}")


def _windows(day: str) -> tuple[int, int, int]:
    start, end = baseline._session_windows(day)["NY"]
    return start, start + 30 * 60 * 1_000_000_000, end


def opening_range_events(day: str, tape: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Only actual ES executions define OR boundaries and first strict breaks."""
    start, or_end, close = _windows(day)
    ts = tape["timestamp_ns"]
    trade = (tape["execution_size"] > 0) & np.isfinite(tape["execution_price"])
    inside = np.flatnonzero(trade & (ts >= start) & (ts < or_end))
    if not len(inside):
        raise BreakoutStudyError(f"no opening-range executions: {day}")
    prices = tape["execution_price"]
    high = float(np.max(prices[inside])); low = float(np.min(prices[inside]))
    candidates = np.flatnonzero(trade & (ts >= or_end) & (ts < close))
    events = []
    for direction, predicate, boundary in (("LONG", prices[candidates] > high, high),
                                           ("SHORT", prices[candidates] < low, low)):
        found = candidates[predicate]
        if len(found):
            ix = int(found[0]); sign = 1 if direction == "LONG" else -1
            events.append({"date": day, "period": _period(day), "direction": direction,
                "sign": sign, "timestamp_ns": int(ts[ix]), "trade_price": float(prices[ix]),
                "boundary": boundary, "overshoot_ticks": float(sign*(prices[ix]-boundary)/TICK),
                "tape_index": ix})
    events.sort(key=lambda e: (e["timestamp_ns"], e["tape_index"]))
    label = "NO_BREAKOUT" if not events else ("BOTH_DIRECTIONS" if len(events) == 2
            else f"{events[0]['direction']}_ONLY")
    detail = {"date": day, "or_start_ns": start, "or_end_ns": or_end,
              "rth_close_ns": close, "high": high, "low": low,
              "width_ticks": float((high-low)/TICK), "width_points": high-low,
              "or_execution_count": len(inside), "status": label,
              "first_break_direction": events[0]["direction"] if events else None}
    return detail, events


def _asof(ts: np.ndarray, target: int) -> int:
    return int(np.searchsorted(ts, target, side="right")-1)


def _future(ts: np.ndarray, target: int, end: int) -> int | None:
    ix = int(np.searchsorted(ts, target, side="left"))
    return ix if ix < len(ts) and ts[ix] < end else None


def _sample_depth_baseline(rows: np.ndarray, event_ns: int, rth_start: int,
                           field: str) -> float | None:
    if event_ns-rth_start < 30_000_000_000:
        return None
    ts = np.asarray(rows["ts"], dtype=np.int64)
    grid = np.arange(event_ns-30_000_000_000, event_ns, 50_000_000, dtype=np.int64)
    ids = np.searchsorted(ts, grid, side="right")-1
    if np.any(ids < 0):
        return None
    values = np.asarray(rows[field][ids], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) >= 500 else None


def _prior_quantile(history: Mapping[str, list[float]], key: str,
                    value: float | None) -> tuple[str | None, bool | None]:
    values = history[key]
    if value is None or not math.isfinite(value) or len(values) < 3:
        return None, None
    q1, q2 = np.quantile(np.asarray(values, dtype=np.float64), [1/3, 2/3])
    bucket = "LOW" if value <= q1 else ("MID" if value <= q2 else "HIGH")
    median = float(np.median(values))
    supportive = value < median if key == "refill_recovery_500ms" else value > median
    return bucket, supportive


def _price_controls(tape: np.ndarray, event: Mapping[str, Any], start: int) -> dict[str, float | None]:
    ts = tape["timestamp_ns"]; mid = (tape["bid"]+tape["ask"])/2
    target = int(event["timestamp_ns"])
    grid = np.arange(target-10_000_000_000, target+1, 500_000_000, dtype=np.int64)
    ids = np.searchsorted(ts, grid, side="right")-1
    ids[-1] = int(np.searchsorted(ts, target, side="left")-1)
    if np.any(ids < 0) or ts[ids[0]] < start or np.any(~np.isfinite(mid[ids])):
        return {k: None for k in PRICE_KEYS[:3]}
    prices = mid[ids]
    diff = np.diff(prices)/TICK
    move = float(event["sign"]*(prices[-1]-prices[0])/TICK)
    traveled = float(np.sum(np.abs(diff)))
    return {"price_velocity_10s": move/10,
            "trend_efficiency_10s": move/traveled if traveled else 0.0,
            "realized_volatility_10s": float(np.sqrt(np.sum(diff*diff)))}


def _features(rows: np.ndarray, tape: np.ndarray, event: Mapping[str, Any],
              opening: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    ts = np.asarray(rows["ts"], dtype=np.int64)
    t = int(event["timestamp_ns"]); sign = int(event["sign"])
    at = int(np.searchsorted(ts, t, side="left")-1)
    if at < 0 or ts[at] < opening["or_start_ns"]:
        raise BreakoutStudyError(f"no causal L2 state for breakout: {event['date']}")
    opposite = "ask5" if sign > 0 else "bid5"
    depth = np.asarray(rows[opposite], dtype=np.float64)
    current = float(depth[at]); denom = float(rows["denom"][at])
    features: dict[str, float | None] = {}
    for ms, key in ((500, "mlofi_500ms"), (2000, "mlofi_2s")):
        lo = int(np.searchsorted(ts, t-ms*1_000_000, side="left"))
        total = float(np.sum(rows["mlofi"][lo:at+1], dtype=np.float64))
        features[key] = sign*total/denom if denom > 0 else None
    baseline_depth = _sample_depth_baseline(rows, t, opening["or_start_ns"], opposite)
    features["depth_depletion"] = (1-current/baseline_depth if baseline_depth and current > 0 else None)
    lo10 = _asof(ts, t-10_000_000_000)
    if lo10 >= 0 and ts[lo10] >= opening["or_start_ns"] and denom > 0:
        flow10 = float(np.sum(rows["mlofi"][lo10+1:at+1], dtype=np.float64))/denom
        move = sign*(float(rows["mid"][at])-float(rows["mid"][lo10]))/TICK
        features["impact_per_flow_10s"] = move/abs(flow10) if abs(flow10) > 1e-6 else None
    else:
        features["impact_per_flow_10s"] = None
    # Aggregate displayed withdrawal/addition from non-trade updates. Exact
    # order-ID cancellation is unavailable in MBP-10 and is never claimed.
    lo = max(1, int(np.searchsorted(ts, t-500_000_000, side="left")))
    ids = np.arange(lo, at+1)
    ids = ids[rows["action"][ids] == 2]
    delta = depth[ids]-depth[ids-1]
    removed = float(np.sum(np.maximum(-delta, 0)))
    added = float(np.sum(np.maximum(delta, 0)))
    features["withdrawal_500ms"] = ((removed-added)/(removed+added)
                                      if removed+added > 0 else None)
    features.update(_price_controls(tape, event, opening["or_start_ns"]))
    features["opening_range_width_ticks"] = float(opening["width_ticks"])
    features["overshoot_ticks"] = float(event["overshoot_ticks"])
    # As-of state at t+500ms is valid even if the last book update was earlier.
    later = _asof(ts, t+500_000_000)
    deficit = baseline_depth-current if baseline_depth is not None else None
    delayed = {"refill_recovery_500ms": (float((depth[later]-current)/deficit)
               if later >= at and deficit is not None and deficit > 0 and
               t+500_000_000 < opening["rth_close_ns"] else None),
               "observation_end_ns": t+500_000_000}
    return features, delayed


def _markout(values: Sequence[float]) -> dict[str, Any]:
    return flow._markout_values(values)


def _path_analysis(tape: np.ndarray, event: Mapping[str, Any],
                   opening: Mapping[str, Any]) -> dict[str, Any]:
    ts, _, bid, ask = flow._quote_arrays(tape)
    mid = (bid+ask)/2
    t = int(event["timestamp_ns"]); sign = int(event["sign"])
    close = int(opening["rth_close_ns"])
    anchor_ix = int(event["tape_index"])
    while anchor_ix < len(tape) and ts[anchor_ix] < close and not (
            np.isfinite(bid[anchor_ix]) and np.isfinite(ask[anchor_ix]) and ask[anchor_ix] > bid[anchor_ix]):
        anchor_ix += 1
    if anchor_ix >= len(tape) or ts[anchor_ix] >= close:
        raise BreakoutStudyError(f"no event-known BBO anchor: {event['date']}")
    raw_mid = float(mid[anchor_ix])
    entry_ix = _future(ts, t+2_000_000, close)
    if entry_ix is not None:
        while entry_ix < len(tape) and ts[entry_ix] < close and not (
                np.isfinite(bid[entry_ix]) and np.isfinite(ask[entry_ix]) and ask[entry_ix] > bid[entry_ix]):
            entry_ix += 1
        if entry_ix >= len(tape) or ts[entry_ix] >= close:
            entry_ix = None
    entry_quote = (float(ask[entry_ix] if sign > 0 else bid[entry_ix]) if entry_ix is not None else None)
    fill = entry_quote+sign*TICK if entry_quote is not None else None
    paths: dict[str, Any] = {}
    delayed_paths: dict[str, float | None] = {}
    delayed_ix = _future(ts, t+500_000_000, close)
    for h in HORIZONS_MS:
        raw_ix = _future(ts, t+h*1_000_000, close)
        exit_ix = _future(ts, int(ts[entry_ix])+h*1_000_000, close) if entry_ix is not None else None
        raw = (sign*(float(mid[raw_ix])-raw_mid)/TICK if raw_ix is not None and np.isfinite(mid[raw_ix]) else None)
        quote = actual = shift = pre_move = spread = None
        if entry_ix is not None and exit_ix is not None and raw is not None:
            exit_price = float(bid[exit_ix] if sign > 0 else ask[exit_ix])
            quote = sign*(exit_price-entry_quote)/TICK
            actual = sign*(exit_price-fill)/TICK
            shift = sign*(float(mid[exit_ix])-float(mid[raw_ix]))/TICK
            pre_move = sign*(float(mid[entry_ix])-raw_mid)/TICK
            spread = sign*((exit_price-float(mid[exit_ix]))+(float(mid[entry_ix])-entry_quote))/TICK
            if abs(actual-(raw+shift-pre_move+spread-1.0)) > 1e-8:
                raise BreakoutStudyError("signal-to-fill decomposition does not balance")
        paths[str(h)] = {"raw": raw, "quote": quote, "actual": actual,
                         "horizon_shift": shift, "pre_entry_price_move": pre_move,
                         "bid_ask_effect": spread}
        if delayed_ix is not None:
            after = _future(ts, int(ts[delayed_ix])+h*1_000_000, close)
            delayed_paths[str(h)] = (sign*(float(mid[after])-float(mid[delayed_ix]))/TICK
                                     if after is not None and np.isfinite(mid[after]) else None)
        else:
            delayed_paths[str(h)] = None
    excursions = {}; barriers = {}
    for h in EXCURSION_MS:
        end_ix = _future(ts, t+h*1_000_000, close)
        if end_ix is None:
            excursions[str(h)] = None
            continue
        signed = sign*(mid[anchor_ix:end_ix+1]-raw_mid)/TICK
        signed = signed[np.isfinite(signed)]
        excursions[str(h)] = {"mfe": float(max(0, np.max(signed))),
                              "mae": float(min(0, np.min(signed)))} if len(signed) else None
    for horizon in (10000, 60000):
        end_ix = _future(ts, t+horizon*1_000_000, close)
        for favorable, adverse in BARRIERS:
            key = f"{favorable}:-{adverse}@{horizon}ms"
            if end_ix is None:
                barriers[key] = {"result": "UNAVAILABLE", "touch_ms": None}; continue
            signed = sign*(mid[anchor_ix:end_ix+1]-raw_mid)/TICK
            fav = np.flatnonzero(signed >= favorable-1e-9)
            bad = np.flatnonzero(signed <= -adverse+1e-9)
            fi = int(fav[0]) if len(fav) else None
            bi = int(bad[0]) if len(bad) else None
            first = fi if bi is None or fi is not None and fi <= bi else bi
            result = "NEITHER" if first is None else ("FAVORABLE_FIRST" if first == fi else "ADVERSE_FIRST")
            barriers[key] = {"result": result,
                "touch_ms": float((ts[anchor_ix+first]-t)/1e6) if first is not None else None}
    return {"raw_anchor_mid": raw_mid, "raw_anchor_time_ns": int(ts[anchor_ix]),
            "entry_time_ns": int(ts[entry_ix]) if entry_ix is not None else None,
            "entry_quote": entry_quote, "actual_fill": fill,
            "paths": paths, "delayed_paths": delayed_paths,
            "excursions": excursions, "barriers": barriers,
            "_anchor_index": anchor_ix}


def _topology(tape: np.ndarray, event: Mapping[str, Any], opening: Mapping[str, Any],
              anchor_ix: int, anchor_mid: float) -> str:
    ts = tape["timestamp_ns"]; t = int(event["timestamp_ns"])
    end = _future(ts, t+60_000_000_000, int(opening["rth_close_ns"]))
    if end is None:
        return "UNCLASSIFIED"
    segment = tape[anchor_ix:end+1]
    sign = int(event["sign"])
    trades = segment["execution_size"] > 0
    opposite = (segment["execution_price"] < opening["low"] if sign > 0
                else segment["execution_price"] > opening["high"])
    if np.any(trades & opposite):
        return "OPPOSITE_RANGE_BREAK"
    mid = (segment["bid"]+segment["ask"])/2
    # The event trade itself is outside by construction. A midpoint already
    # inside the trade-defined OR is not evidence of a post-event return.
    inside_trade = (segment["execution_price"] <= opening["high"] if sign > 0
                    else segment["execution_price"] >= opening["low"])
    if np.any(trades & inside_trade):
        return "RETURN_INSIDE_OPENING_RANGE"
    ten = _future(ts, t+10_000_000_000, int(opening["rth_close_ns"]))
    if ten is None:
        return "UNCLASSIFIED"
    signed = sign*(mid[:ten-anchor_ix+1]-anchor_mid)/TICK
    plus4 = np.flatnonzero(signed >= 4-1e-9)
    plus2 = np.flatnonzero(signed >= 2-1e-9)
    minus2 = np.flatnonzero(signed <= -2+1e-9)
    if len(plus4) and (not len(minus2) or plus4[0] < minus2[0]):
        return "IMMEDIATE_CONTINUATION"
    if len(plus2) and len(minus2) and plus2[0] < minus2[0]:
        return "SMALL_CONTINUATION_THEN_FAILURE"
    if not len(plus2) and not len(minus2):
        return "STAGNATION"
    return "UNCLASSIFIED"


def _tod_bucket(day: str, timestamp_ns: int) -> str:
    start, _, _ = _windows(day)
    minute = (timestamp_ns-start)/60_000_000_000
    for hi, label in ((60, "10:00-10:30"), (120, "10:30-11:30"),
                      (270, "11:30-14:00"), (360, "14:00-15:30"),
                      (390, "15:30-16:00")):
        if minute < hi:
            return label
    raise BreakoutStudyError(f"event outside frozen RTH bins: {day}/{timestamp_ns}")


def evaluate_date(day: str, rows: np.ndarray, tape: np.ndarray,
                  history: Mapping[str, list[float]]) -> dict[str, Any]:
    opening, events = opening_range_events(day, tape)
    output = []
    for event in events:
        features, delayed = _features(rows, tape, event, opening)
        paths = _path_analysis(tape, event, opening)
        topology = _topology(tape, event, opening, paths["_anchor_index"], paths["raw_anchor_mid"])
        paths.pop("_anchor_index")
        buckets = {}; support = {}
        for key in ALL_L2_KEYS + ("opening_range_width_ticks",):
            value = delayed["refill_recovery_500ms"] if key == "refill_recovery_500ms" else features[key]
            buckets[key], support[key] = _prior_quantile(history, key, value)
        anchor_count = sum(support[k] for k in ANCHOR_L2_KEYS) if all(support[k] is not None for k in ANCHOR_L2_KEYS) else None
        delayed_count = (anchor_count+int(support["refill_recovery_500ms"])
                         if anchor_count is not None and support["refill_recovery_500ms"] is not None else None)
        t = event["timestamp_ns"]
        ts = tape["timestamp_ns"]
        distance = {"0": event["overshoot_ticks"]}
        for h in (100, 250, 500, 1000):
            ix = _future(ts, t+h*1_000_000, opening["rth_close_ns"])
            distance[str(h)] = (float(event["sign"]*((tape["bid"][ix]+tape["ask"][ix])/2-event["boundary"])/TICK)
                                if ix is not None else None)
        output.append({**event, "opening_range_high": opening["high"],
            "opening_range_low": opening["low"], "opening_range_width_ticks": opening["width_ticks"],
            "time_of_day_bucket": _tod_bucket(day, t), "distance_beyond_boundary_ticks": distance,
            "event_anchor_features": features, "delayed_features": delayed,
            "prior_date_terciles": buckets, "supportive": support,
            "event_anchor_support_count": anchor_count, "delayed_support_count": delayed_count,
            "topology": topology, **paths})
    return {"date": day, "period": _period(day), "opening_range": opening,
            "events": output}


def _values(events: Sequence[Mapping[str, Any]], horizon: int, path: str = "raw",
            *, delayed: bool = False) -> list[float]:
    if delayed:
        return [float(x) for e in events if (x := e["delayed_paths"][str(horizon)]) is not None]
    return [float(x) for e in events if (x := e["paths"][str(horizon)][path]) is not None]


def _markout_block(events: Sequence[Mapping[str, Any]], *, delayed: bool = False) -> dict[str, Any]:
    if delayed:
        return {str(h): _markout(_values(events, h, delayed=True)) for h in HORIZONS_MS}
    return {path: {str(h): _markout(_values(events, h, path)) for h in HORIZONS_MS}
            for path in ("raw", "quote", "actual")}


def _feature_rows(events: Sequence[Mapping[str, Any]], key: str,
                  period: str | None = None) -> dict[str, Any]:
    group = [e for e in events if period is None or e["period"] == period]
    buckets = {name: [e for e in group if e["prior_date_terciles"][key] == name]
               for name in ("LOW", "MID", "HIGH")}
    delayed = key == "refill_recovery_500ms"
    result = {name: {"n": len(rows), "status": "SUFFICIENT" if len(rows) >= 10 else "INSUFFICIENT_BUCKET_SAMPLE",
                     "markout_10s": _markout(_values(rows, 10000, delayed=delayed)),
                     "mfe_10s": _markout([e["excursions"]["10000"]["mfe"] for e in rows if e["excursions"]["10000"]]),
                     "mae_10s": _markout([e["excursions"]["10000"]["mae"] for e in rows if e["excursions"]["10000"]]),
                     "first_touch_4_vs_4_10s": {outcome: sum(e["barriers"]["4:-4@10000ms"]["result"] == outcome for e in rows)
                         for outcome in ("FAVORABLE_FIRST", "ADVERSE_FIRST", "NEITHER", "UNAVAILABLE")}}
              for name, rows in buckets.items()}
    low = result["LOW"]["markout_10s"]["mean_ticks"]
    mid = result["MID"]["markout_10s"]["mean_ticks"]
    high = result["HIGH"]["markout_10s"]["mean_ticks"]
    if any(result[name]["status"] != "SUFFICIENT" for name in result):
        shape = "INSUFFICIENT_SAMPLE"
    elif key == "refill_recovery_500ms" and low is not None and mid is not None and high is not None and low > mid > high:
        shape = "MONOTONIC_IMPROVEMENT"
    elif key == "refill_recovery_500ms" and low is not None and mid is not None and high is not None and low < mid < high:
        shape = "OPPOSITE_RELATIONSHIP"
    elif low is not None and mid is not None and high is not None and low < mid < high:
        shape = "MONOTONIC_IMPROVEMENT"
    elif low is not None and mid is not None and high is not None and high < mid < low:
        shape = "OPPOSITE_RELATIONSHIP"
    elif mid is not None and low is not None and high is not None and mid > max(low, high):
        shape = "INVERTED_U"
    elif low is not None and mid is not None and high is not None and low < mid and high >= mid:
        shape = "SATURATION"
    else:
        shape = "NO_CLEAR_SHAPE"
    return {"terciles": result, "shape": shape,
            "expected_favorable_direction": "LOW" if delayed else "HIGH",
            "high_minus_low_ticks": high-low if high is not None and low is not None else None,
            "valid_events": sum(len(v) for v in buckets.values()),
            "outcome_clock": "event+500ms" if delayed else "event"}


def _support_group(events: Sequence[Mapping[str, Any]], *, delayed: bool) -> dict[str, Any]:
    key = "delayed_support_count" if delayed else "event_anchor_support_count"
    buckets = {"0-1": [], "2-3": [], "4-5": []}
    exact = defaultdict(list)
    for e in events:
        value = e[key]
        if value is None:
            continue
        exact[str(value)].append(e)
        label = "0-1" if value <= 1 else ("2-3" if value <= 3 else "4-5")
        buckets[label].append(e)
    return {"groups": {label: {"n": len(rows), "status": "SUFFICIENT" if len(rows) >= 10 else "INSUFFICIENT_BUCKET_SAMPLE",
                              "markout_10s": _markout(_values(rows, 10000, delayed=delayed))}
                       for label, rows in buckets.items()},
            "exact_descriptive": {n: {"n": len(rows), "markout_10s": _markout(_values(rows, 10000, delayed=delayed))}
                                  for n, rows in sorted(exact.items())}}


def _barrier_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for horizon in (10000, 60000):
        for favorable, adverse in BARRIERS:
            key = f"{favorable}:-{adverse}@{horizon}ms"
            outcomes = [e["barriers"][key] for e in events]
            available = [x for x in outcomes if x["result"] != "UNAVAILABLE"]
            n = len(available)
            result[key] = {"n": n, "unavailable": len(outcomes)-n,
                "favorable_first": sum(x["result"] == "FAVORABLE_FIRST" for x in available)/n if n else None,
                "adverse_first": sum(x["result"] == "ADVERSE_FIRST" for x in available)/n if n else None,
                "neither": sum(x["result"] == "NEITHER" for x in available)/n if n else None,
                "median_first_touch_ms": float(np.median([x["touch_ms"] for x in available if x["touch_ms"] is not None]))
                    if any(x["touch_ms"] is not None for x in available) else None}
    return result


def _excursion_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(h): {key: _markout([e["excursions"][str(h)][key] for e in events if e["excursions"][str(h)] is not None])
                     for key in ("mfe", "mae")} for h in EXCURSION_MS}


def _execution_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for h in (500, 1000, 2000, 5000, 10000, 30000, 60000):
        valid = [e["paths"][str(h)] for e in events if e["paths"][str(h)]["actual"] is not None]
        mean = lambda k: float(np.mean([r[k] for r in valid])) if valid else None
        result[str(h)] = {"n": len(valid), "raw_signal_edge": mean("raw"),
            "executable_quote": mean("quote"), "actual_fill": mean("actual"),
            "pre_entry_price_movement": mean("pre_entry_price_move"),
            "pre_entry_impact": -mean("pre_entry_price_move") if valid else None,
            "bid_ask_effect": mean("bid_ask_effect"), "horizon_alignment": mean("horizon_shift"),
            "adverse_entry_tick": -1.0 if valid else None,
            "total_signal_to_fill_deterioration": mean("actual")-mean("raw") if valid else None}
    return result


def _price_only_control(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [e for e in events if e["event_anchor_support_count"] is not None and e["event_anchor_support_count"] >= 3]
    control = [e for e in events if e["event_anchor_support_count"] is not None and e["event_anchor_support_count"] <= 1]
    scales = (2., .25, 4., 20., 4.)
    pairs = []
    for item in selected:
        candidates = [e for e in control if e["period"] == item["period"] and e["direction"] == item["direction"]
                      and e["date"] != item["date"]
                      and all(e["event_anchor_features"][k] is not None and item["event_anchor_features"][k] is not None for k in PRICE_KEYS)]
        if not candidates:
            continue
        def distance(other):
            return sum(((item["event_anchor_features"][k]-other["event_anchor_features"][k])/scale)**2
                       for k, scale in zip(PRICE_KEYS, scales))
        match = min(candidates, key=lambda e: (distance(e), e["date"], e["timestamp_ns"]))
        pairs.append((item, match))
    paired = [(a["paths"]["10000"]["raw"], b["paths"]["10000"]["raw"]) for a, b in pairs]
    paired = [(a, b) for a, b in paired if a is not None and b is not None]
    return {"supported_n": len(selected), "low_support_control_n": len(control),
            "matched_pairs": len(paired), "unmatched": len(selected)-len(pairs),
            "supported_raw_10s": _markout([a for a, _ in paired]),
            "matched_price_only_raw_10s": _markout([b for _, b in paired]),
            "incremental_ticks": float(np.mean([a-b for a, b in paired])) if paired else None,
            "matching_uses_outcomes": False, "matching_uses_l2_values_for_distance": False}


def _effect(events: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if key == "base":
        values = _values(events, 10000)
        return float(np.mean(values)) if values else None
    if key == "support_count":
        high = _values([e for e in events if e["event_anchor_support_count"] is not None and e["event_anchor_support_count"] >= 3], 10000)
        low = _values([e for e in events if e["event_anchor_support_count"] is not None and e["event_anchor_support_count"] <= 1], 10000)
    else:
        high = _values([e for e in events if e["prior_date_terciles"][key] == "HIGH"], 10000,
                       delayed=key == "refill_recovery_500ms")
        low = _values([e for e in events if e["prior_date_terciles"][key] == "LOW"], 10000,
                      delayed=key == "refill_recovery_500ms")
    return float(np.mean(high)-np.mean(low)) if high and low else None


def _leave_out(events: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    keys = ("base", "mlofi_500ms", "depth_depletion", "support_count")
    groups = sorted({flow._week(e["date"]) if field == "week" else e["date"] for e in events})
    result = {}
    for key in keys:
        full = _effect(events, key)
        values = []
        for group in groups:
            subset = [e for e in events if (flow._week(e["date"]) if field == "week" else e["date"]) != group]
            effect = _effect(subset, key)
            if effect is not None:
                values.append((group, effect))
        ordered = sorted(values, key=lambda x: x[1])
        result[key] = {"full_effect": full, "groups_tested": len(values),
            "sign_stability": sum(x*full > 0 for _, x in values)/len(values) if values and full not in (None, 0) else None,
            "median": float(np.median([x for _, x in values])) if values else None,
            "minimum": ordered[0][1] if ordered else None,
            "maximum": ordered[-1][1] if ordered else None,
            "worst_omitted": ordered[0][0] if ordered else None,
            "best_omitted": ordered[-1][0] if ordered else None}
    return result


def _permutation(events: Sequence[Mapping[str, Any]], key: str, rng: np.random.Generator) -> dict[str, Any]:
    delayed = key == "refill_recovery_500ms" or key == "delayed_support_count"
    outcome = np.array([e["delayed_paths"]["10000"] if delayed else e["paths"]["10000"]["raw"] for e in events], dtype=object)
    if key.endswith("support_count"):
        label = np.array([e[key] for e in events], dtype=object)
        valid = np.array([y is not None and x is not None and (x <= 1 or x >= 3) for x, y in zip(label, outcome)], dtype=bool)
        high = np.array([x is not None and x >= 3 for x in label[valid]], dtype=bool)
    else:
        label = np.array([e["prior_date_terciles"][key] for e in events], dtype=object)
        valid = np.array([y is not None and x in ("LOW", "HIGH") for x, y in zip(label, outcome)], dtype=bool)
        high = np.asarray(label[valid] == "HIGH", dtype=bool)
    y = np.asarray(outcome[valid], dtype=np.float64)
    if len(y) < 20 or sum(high) < 5 or sum(~high) < 5:
        return {"status": "INSUFFICIENT_SAMPLE", "n": len(y), "observed": None}
    groups = np.array([e["period"]+"|"+e["direction"] for e, keep in zip(events, valid) if keep])
    observed = float(np.mean(y[high])-np.mean(y[~high]))
    null = []
    for _ in range(1000):
        shuffled = high.copy()
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            shuffled[ids] = rng.permutation(shuffled[ids])
        null.append(float(np.mean(y[shuffled])-np.mean(y[~shuffled])))
    return {"status": "COMPLETE", "n": len(y), "observed": observed,
        "null_mean": float(np.mean(null)), "null_p05": float(np.quantile(null, .05)),
        "null_p95": float(np.quantile(null, .95)),
        "percentile": float(np.mean(np.asarray(null) <= observed)),
        "two_sided_p_value": float((1+sum(abs(x) >= abs(observed) for x in null))/1001),
        "strata": "period x direction", "permutations": 1000}


def _aggregate(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = [e for p in payloads for e in p["events"]]
    open_rows = [p["opening_range"] for p in payloads]
    period = {name: [e for e in events if e["period"] == name]
              for name in ("SPRING_2025", "OCTOBER_2025")}
    direction = {name: [e for e in events if e["direction"] == name]
                 for name in ("LONG", "SHORT")}
    markouts = {"ALL": _markout_block(events),
                **{name: _markout_block(rows) for name, rows in {**period, **direction}.items()}}
    excursions = {"ALL": _excursion_summary(events),
                  **{name: _excursion_summary(rows) for name, rows in {**period, **direction}.items()}}
    barriers = {"ALL": _barrier_summary(events),
                **{name: _barrier_summary(rows) for name, rows in {**period, **direction}.items()}}
    terciles = {key: {"ALL": _feature_rows(events, key),
                      **{name: _feature_rows(rows, key) for name, rows in period.items()}}
                for key in ALL_L2_KEYS}
    supports = {name: {"ALL": _support_group(events, delayed=flag),
                       **{period_name: _support_group(rows, delayed=flag) for period_name, rows in period.items()}}
                for name, flag in (("event_anchor", False), ("delayed", True))}
    daily = []
    for p in payloads:
        e = p["events"]
        daily.append({"date": p["date"], "period": p["period"], "events": len(e),
            "long": sum(x["direction"] == "LONG" for x in e),
            "short": sum(x["direction"] == "SHORT" for x in e),
            "raw_10s": _markout(_values(e, 10000)), "actual_10s": _markout(_values(e, 10000, "actual")),
            "mfe_mae_10s": _excursion_summary(e)["10000"],
            "topology": {name: sum(x["topology"] == name for x in e) for name in
                ("IMMEDIATE_CONTINUATION", "SMALL_CONTINUATION_THEN_FAILURE", "STAGNATION",
                 "RETURN_INSIDE_OPENING_RANGE", "OPPOSITE_RANGE_BREAK", "UNCLASSIFIED")},
            "event_anchor_support_counts": {str(n): sum(x["event_anchor_support_count"] == n for x in e) for n in range(5)}})
    by_week = defaultdict(list)
    for event in events:
        by_week[flow._week(event["date"])].append(event)
    weekly = [{"week": name, "events": len(rows),
               "status": "SUFFICIENT" if len(rows) >= 10 else "INSUFFICIENT_WEEK_SAMPLE",
               "raw_10s": _markout(_values(rows, 10000)),
               "actual_10s": _markout(_values(rows, 10000, "actual"))}
              for name, rows in sorted(by_week.items())]
    topology = {name: {"n": len(group), "features": {key: _markout([
                    e["event_anchor_features"][key] for e in group if e["event_anchor_features"][key] is not None])
                    for key in ANCHOR_L2_KEYS}}
                for name, group in ((name, [e for e in events if e["topology"] == name])
                    for name in ("IMMEDIATE_CONTINUATION", "SMALL_CONTINUATION_THEN_FAILURE", "STAGNATION",
                                 "RETURN_INSIDE_OPENING_RANGE", "OPPOSITE_RANGE_BREAK", "UNCLASSIFIED"))}
    control = _price_only_control(events)
    rng = np.random.default_rng(20251005)
    permutation = {key: _permutation(events, key, rng) for key in
                   ("mlofi_500ms", "depth_depletion", "refill_recovery_500ms",
                    "event_anchor_support_count", "delayed_support_count")}
    spring = markouts["SPRING_2025"]["raw"]["10000"]["mean_ticks"]
    october = markouts["OCTOBER_2025"]["raw"]["10000"]["mean_ticks"]
    spring_fill = markouts["SPRING_2025"]["actual"]["10000"]["mean_ticks"]
    october_fill = markouts["OCTOBER_2025"]["actual"]["10000"]["mean_ticks"]
    compatible = [key for key in ANCHOR_L2_KEYS if (terciles[key]["SPRING_2025"]["high_minus_low_ticks"] is not None
        and terciles[key]["OCTOBER_2025"]["high_minus_low_ticks"] is not None
        and terciles[key]["SPRING_2025"]["high_minus_low_ticks"] > 0
        and terciles[key]["OCTOBER_2025"]["high_minus_low_ticks"] >= 0
        and all(terciles[key][p]["shape"] != "INSUFFICIENT_SAMPLE" for p in period))]
    positive_days = sum(x["actual_10s"]["mean_ticks"] is not None and x["actual_10s"]["mean_ticks"] > 0 for x in daily)
    if all(x is not None and x > 0 for x in (spring, october, spring_fill, october_fill)) and compatible and control["incremental_ticks"] is not None and control["incremental_ticks"] > 0 and positive_days > len(daily)/2:
        decision = "STRUCTURAL_BREAKOUT_L2_PROMISING"
        next_step = "BUILD_ONE_FROZEN_BREAKOUT_STRATEGY_V1"
    elif all(x is not None and x <= 0 for x in (spring, october, spring_fill, october_fill)) and not compatible:
        decision = "STRUCTURAL_BREAKOUT_L2_FAILED"
        next_step = "STOP_ES_STRUCTURAL_BREAKOUT_BRANCH"
    else:
        decision = "STRUCTURAL_BREAKOUT_L2_MIXED"
        next_step = "DIAGNOSE_WHETHER_STRUCTURE_OR_L2_IS_THE_LIMITING_FACTOR"
    execution = {"ALL": _execution_summary(events),
                 **{name: _execution_summary(rows) for name, rows in {**period, **direction}.items()}}
    return {"events": events, "opening_range_events": open_rows,
        "baseline_breakout": {name: {"n": len(rows), "markouts": markouts[name],
                              "mfe_mae": excursions[name], "barriers": barriers[name]}
                             for name, rows in {"ALL": events, **period, **direction}.items()},
        "markouts": markouts, "mfe_mae": excursions, "first_touch": barriers,
        "feature_terciles": terciles, "event_anchor_l2_support_count": supports["event_anchor"],
        "delayed_l2_support_count": supports["delayed"],
        "period_results": {p: {"n": len(e), "markouts": markouts[p]} for p, e in period.items()},
        "direction_results": {p: {"n": len(e), "markouts": markouts[p]} for p, e in direction.items()},
        "time_of_day_results": {name: {"n": len(group), "raw_10s": _markout(_values(group, 10000))}
             for name, group in ((name, [e for e in events if e["time_of_day_bucket"] == name])
                 for name in ("10:00-10:30", "10:30-11:30", "11:30-14:00", "14:00-15:30", "15:30-16:00"))},
        "opening_range_size_results": _feature_rows(events, "opening_range_width_ticks"),
        "failure_topology": topology, "price_only_control": control,
        "execution_hurdle": execution, "daily_results": daily, "weekly_results": weekly,
        "lodo_results": _leave_out(events, "date"), "lowo_results": _leave_out(events, "week"),
        "permutation_results": permutation, "compatible_l2_features": compatible,
        "positive_actual_10s_dates": positive_days,
        "primary_decision": decision, "next_step": next_step,
        "insufficient_for_l2_conclusion": not compatible or control["matched_pairs"] < 10}


def _checkpoint(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, day: str, source_sha: str, tape_sha: str,
                     history_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            row = json.load(source)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    expected = {"status": "DATE_COMPLETE", "version": CHECKPOINT_VERSION,
                "date": day, "source_sha256": source_sha, "tape_sha256": tape_sha,
                "history_sha256": history_sha, "config_sha256": CONFIG_SHA256}
    return row if all(row.get(k) == value for k, value in expected.items()) else None


def run(*, data_root: Path = native.DATA_ROOT, output_root: Path = OUT_ROOT,
        smoke: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    paths, source_manifest = native._source_catalog(data_root)
    target_days = flow.TARGET_DATES[:1] if smoke else flow.TARGET_DATES
    tape_sha = {}
    for day in target_days:
        path = native._tape_path(day)
        if not path.is_file():
            raise BreakoutStudyError(f"missing native canonical tape: {day}")
        tape_sha[day] = native._sha(path)
    coverage = {"status": "PASS", "dataset": "GLBX.MDP3", "schema": "mbp-10",
        "instrument": "ES", "native_es_only": True, "no_mes": True, "no_mbo": True,
        "spring_role": "PRIMARY_DISCOVERY", "october_role": "SECONDARY_DEV_COMPATIBILITY",
        "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
        "source_sha256_by_date": {day: source_manifest[day]["sha256"] for day in native.ALL_SOURCE_DATES},
        "tape_sha256_by_date": tape_sha}
    output_root.mkdir(parents=True, exist_ok=True)
    native._write_json(output_root / "study-config.json", CONFIG)
    native._write_json(output_root / "source-coverage.json", coverage)
    history = {key: [] for key in (*ALL_L2_KEYS, "opening_range_width_ticks")}
    payloads = []; completed = []; resumed = []
    chain = "NO_PRIOR_TARGET"
    for pos, day in enumerate(target_days, 1):
        source_sha = source_manifest[day]["sha256"]
        history_sha = _hash([chain, history])
        checkpoint_path = _checkpoint(output_root, day)
        cached = _read_checkpoint(checkpoint_path, day, source_sha, tape_sha[day], history_sha)
        if cached:
            payload = cached["payload"]
            resumed.append(day)
            print(f"BREAKOUT_DATE_RESUME={day}", flush=True)
        else:
            print(f"BREAKOUT_DATE_START={pos}/{len(target_days)} date={day}", flush=True)
            rows = flow._cached_compact(day, paths[day], source_sha, flow.OUT_ROOT)
            tape, _ = native._load_tape(day, native._tape_path(day), source_sha)
            payload = evaluate_date(day, rows, tape, history)
            del rows, tape
            native._write_checkpoint(checkpoint_path, {"status": "DATE_COMPLETE",
                "version": CHECKPOINT_VERSION, "date": day, "source_sha256": source_sha,
                "tape_sha256": tape_sha[day], "history_sha256": history_sha,
                "config_sha256": CONFIG_SHA256, "payload": payload})
            completed.append(day)
            print(f"BREAKOUT_DATE_COMPLETE={day} events={len(payload['events'])} status={payload['opening_range']['status']}", flush=True)
        # Current-day feature values become history only AFTER the entire date.
        for event in payload["events"]:
            for key in history:
                value = event["delayed_features"][key] if key == "refill_recovery_500ms" else event["event_anchor_features"][key]
                if value is not None and math.isfinite(value):
                    history[key].append(float(value))
        chain = _hash([chain, day, source_sha, tape_sha[day], CONFIG_SHA256])
        payloads.append(payload)
        native._write_json(output_root / "checkpoints" / "progress.json",
                           {"last_date": day, "completed_dates": completed, "resumed_dates": resumed,
                            "source_chain_sha256": chain, "config_sha256": CONFIG_SHA256})
    if smoke:
        return {"status": "SMOKE_PASS", "dates": list(target_days),
                "events": sum(len(p["events"]) for p in payloads)}
    if len(payloads) != len(flow.TARGET_DATES):
        raise BreakoutStudyError("incomplete target-date checkpoint set")
    results = _aggregate(payloads)
    events = results.pop("events")
    native._write_gzip_jsonl(output_root / "events.jsonl.gz", events)
    files = {
        "opening-range-events.json": results["opening_range_events"],
        "baseline-breakout.json": results["baseline_breakout"],
        "event-anchor-features.json": [{"date": e["date"], "direction": e["direction"],
            "features": e["event_anchor_features"], "prior_date_terciles": {k: e["prior_date_terciles"][k] for k in ANCHOR_L2_KEYS}}
            for e in events],
        "delayed-features.json": [{"date": e["date"], "direction": e["direction"],
            "features": e["delayed_features"], "prior_date_tercile": e["prior_date_terciles"]["refill_recovery_500ms"]}
            for e in events],
        "feature-terciles.json": results["feature_terciles"],
        "event-anchor-l2-support-count.json": results["event_anchor_l2_support_count"],
        "delayed-l2-support-count.json": results["delayed_l2_support_count"],
        "period-results.json": results["period_results"],
        "direction-results.json": results["direction_results"],
        "time-of-day-results.json": results["time_of_day_results"],
        "opening-range-size-results.json": results["opening_range_size_results"],
        "markouts.json": results["markouts"], "mfe-mae.json": results["mfe_mae"],
        "first-touch.json": results["first_touch"],
        "failure-topology.json": results["failure_topology"],
        "price-only-control.json": results["price_only_control"],
        "execution-hurdle.json": results["execution_hurdle"],
        "daily-results.json": results["daily_results"],
        "weekly-results.json": results["weekly_results"],
        "lodo-results.json": results["lodo_results"],
        "lowo-results.json": results["lowo_results"],
        "permutation-results.json": results["permutation_results"],
    }
    valid_quintiles = [key for key in ALL_L2_KEYS if results["feature_terciles"][key]["ALL"]["valid_events"] >= 90]
    if valid_quintiles:
        # Descriptive only; no search or rule selection. Chronological prior-date
        # quintile boundaries are recomputed from event feature history.
        quintile_history = {key: [] for key in valid_quintiles}
        quintile_rows = {key: [] for key in valid_quintiles}
        for p in payloads:
            for e in p["events"]:
                for key in valid_quintiles:
                    value = e["delayed_features"][key] if key == "refill_recovery_500ms" else e["event_anchor_features"][key]
                    prior = quintile_history[key]
                    if value is not None and len(prior) >= 5:
                        q = np.quantile(prior, [.2, .4, .6, .8])
                        bucket = int(np.searchsorted(q, value, side="left"))+1
                        quintile_rows[key].append({"date": e["date"], "bucket": bucket,
                            "outcome_10s": e["delayed_paths"]["10000"] if key == "refill_recovery_500ms" else e["paths"]["10000"]["raw"]})
                    if value is not None:
                        prior.append(value)
        files["feature-quintiles-descriptive.json"] = {key: {str(n): {"n": len(group),
            "markout_10s": _markout([x["outcome_10s"] for x in group if x["outcome_10s"] is not None])}
            for n, group in ((n, [r for r in quintile_rows[key] if r["bucket"] == n]) for n in range(1, 6))}
            for key in valid_quintiles}
    for name, value in files.items():
        native._write_json(output_root / name, value)
    n_long = sum(e["direction"] == "LONG" for e in events)
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "dataset": "SPRING_2025 + OCTOBER_2025",
        "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
        "spring_role": "PRIMARY_DISCOVERY", "october_role": "SECONDARY_DEV_COMPATIBILITY",
        "native_es_only": True, "structural_event": "OPENING_RANGE_BREAKOUT_09_30_TO_10_00_ET",
        "opening_range_price_source": "ACTUAL_ES_TRADES", "total_events": len(events),
        "long_events": n_long, "short_events": len(events)-n_long,
        "sessions_no_breakout": sum(p["opening_range"]["status"] == "NO_BREAKOUT" for p in payloads),
        "sessions_one_direction": sum(p["opening_range"]["status"] in ("LONG_ONLY", "SHORT_ONLY") for p in payloads),
        "sessions_both_directions": sum(p["opening_range"]["status"] == "BOTH_DIRECTIONS" for p in payloads),
        "base_raw_markouts": results["markouts"]["ALL"]["raw"],
        "actual_fill_markouts": results["markouts"]["ALL"]["actual"],
        "execution_hurdle": results["execution_hurdle"]["ALL"],
        "compatible_l2_features": results["compatible_l2_features"],
        "price_only_incremental_ticks": results["price_only_control"]["incremental_ticks"],
        "positive_actual_10s_dates": results["positive_actual_10s_dates"],
        "insufficient_for_l2_conclusion": results["insufficient_for_l2_conclusion"],
        "primary_decision": results["primary_decision"], "next_step": results["next_step"],
        "optimization_performed": False, "optuna_performed": False,
        "threshold_search_performed": False, "stop_target_search_performed": False,
        "breakout_window_search_performed": False, "level_search_performed": False,
        "final_oos_accessed": False, "data_downloaded": False,
        "config_sha256": CONFIG_SHA256, "elapsed_seconds": time.monotonic()-started}
    native._write_json(output_root / "summary.json", summary)
    report = [f"# {RUN_ID}", "", f"Decision: **{summary['primary_decision']}**", "",
        "Spring is primary discovery; October is secondary DEV compatibility, not untouched OOS.",
        f"Events: {len(events)} ({n_long} LONG, {len(events)-n_long} SHORT).",
        "OR boundaries use only actual ES trade prices, 09:30–10:00 ET; one first strict breakout per direction.",
        "MBP-10 displayed withdrawal/addition is an aggregate-depth proxy, not exact order-ID cancellation.",
        "Refill is delayed 500ms and is never paired with t0-anchored outcomes.",
        f"Raw 10s mean: {summary['base_raw_markouts']['10000']['mean_ticks']}; actual-fill 10s mean: {summary['actual_fill_markouts']['10000']['mean_ticks']}.",
        f"Next step: {summary['next_step']}", ""]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    artifacts = {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file()}
    native._write_json(output_root / "run-manifest.json", {"status": "COMPLETE", "run_id": RUN_ID,
        "config_sha256": CONFIG_SHA256,
        "source_coverage_sha256": native._sha(output_root / "source-coverage.json"),
        "checkpoint_sha256_by_date": {day: native._sha(_checkpoint(output_root, day)) for day in target_days},
        "artifact_sha256_by_name": artifacts,
        "no_2026_access": True, "no_download": True, "no_optimization": True})
    native._write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files":
        {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name != "artifact-hashes.json"}})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=native.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(data_root=args.data_root, output_root=args.output_root, smoke=args.smoke)
    except (BreakoutStudyError, native.VacuumStudyError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(f"ES_STRUCTURAL_BREAKOUT_L2_EVENT_STUDY={result['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
