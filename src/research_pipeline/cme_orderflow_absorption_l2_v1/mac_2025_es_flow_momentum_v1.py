"""Single frozen, native-ES directional-flow continuation study.

This module has no optimization path.  October is a second fixed evaluation
period, not a source of thresholds, configuration choices, or 2026 OOS data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_absorption_relative_normalization as relative
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from .model import ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS

RUN_ID = "CMEOrderflow_ES_FLOW_MOMENTUM_ONLY_V1_FIXED"
OUT_ROOT = Path("research_runs") / RUN_ID
SPRING_DATES = native.SPRING_DATES
OCTOBER_DATES = native.OCTOBER_DATES
TARGET_DATES = SPRING_DATES + OCTOBER_DATES
SOURCE_DATES = native.ALL_SOURCE_DATES
DEPENDENCY_DATES = native.DEPENDENCY_DATES
HORIZONS_MS = native.HORIZONS_MS
EXCURSION_HORIZONS_MS = (1000, 2000, 5000, 10000, 30000)
TICK = native.TICK
CHECKPOINT_VERSION = "flow-momentum-v1-date-v1"
CONFIG = {
    "pressure_window_ms": 500, "pressure_percentile": 90,
    "pressure_sides": "separate prior-date positive and negative-magnitude q90",
    "refractory_seconds": 2, "max_favorable_move_before_entry_ticks": 6,
    "price_confirmation_min_ticks": 0, "entry_delay_ms": 2.0,
    "stop_ticks": 6, "target_ticks": 9, "target_r": 1.5,
    "max_hold_seconds": 10, "post_exit_refractory_seconds": 2,
    "native_schema": "mbp-10", "dataset": "GLBX.MDP3", "instrument": "ES",
    "history_sample_per_side_per_date": 100_000,
    "vacuum_filters": False, "level_filters": False,
}
CONTROL_SPEC = {
    "version": 1,
    "pool": "every 2s in the same native canonical ES quote tape session; direction is sign of causal 500ms price move, or 2s move if zero",
    "features": "at-or-before timestamp ES mid-price returns over 500ms and 2s, and standard deviation of four past 500ms returns, all in ticks",
    "matching": "same date, session, UTC hour and price-only direction; minimize squared scaled distance using scales (2,4,2) ticks; tie by earlier timestamp",
    "exclusion": "matched control signal must be more than 2s from the flow event; no future outcome or L2 pressure enters matching",
    "execution": "first valid ES quote at/after control time+2ms, same adverse one-tick entry, then markouts from executable entry",
    "fallback": "if no price-only candidate in the same hour/direction, leave unmatched and report count; never use outcomes for fallback",
}
STUDY_SPEC = {
    "version": 1, "strategy": CONFIG, "control": CONTROL_SPEC,
    "flow_formula": "sum of validated inverse-rank price-keyed TOP5 MLOFI contributions on [t-500ms,t] / contemporaneous weighted mean TOP5 resting depth",
    "pressure_threshold": "separate side-magnitude q90 using only deterministic, equal-date-weighted prior-date samples; current date admitted after its evaluation",
    "event": "first raw hit in each two-second refractory window within each ASIA/EUROPE/NY session",
    "price_rule": "signed event mid minus as-of mid at t-500ms is at least zero",
    "execution": "V1 first executable quote at/after signal+2ms; adverse one-tick entry and exit; stop first; ES then MES sizing economics without MES data",
    "exits": "six-tick stop, nine-tick target, ten-second first executable time exit; no overlap; two-second post-exit refractory",
    "strength_shape": "descriptive buckets only; >=10 trades per bucket; plateau if 2s mean range <=0.5 tick; otherwise monotonic improvement, inverted U, extreme-tail exhaustion, or no clear shape",
}


def _hash_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CONFIG_SHA256 = _hash_obj(CONFIG)
STUDY_SHA256 = _hash_obj(STUDY_SPEC)


class FlowStudyError(RuntimeError):
    pass


def _sample_sides(pressure: np.ndarray) -> dict[str, list[float]]:
    return {"LONG": native._systematic_sample(pressure[pressure > 0], 100_000),
            "SHORT": native._systematic_sample(-pressure[pressure < 0], 100_000)}


def prior_thresholds(history: Mapping[str, Sequence[float]]) -> dict[str, float]:
    if not history["LONG"] or not history["SHORT"]:
        raise FlowStudyError("both directional pressure histories require prior dates")
    return {side: float(np.quantile(np.asarray(history[side], dtype=np.float64), .90))
            for side in ("LONG", "SHORT")}


def cluster_flow(ts: np.ndarray, pressure: np.ndarray, sessions: np.ndarray,
                 thresholds: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if not (len(ts) == len(pressure) == len(sessions)):
        raise FlowStudyError("flow arrays have mismatched lengths")
    hit = ((pressure >= thresholds["LONG"]) | (pressure <= -thresholds["SHORT"])) & (pressure != 0) & (sessions >= 0)
    raw = np.flatnonzero(hit)
    # Identical first-hit, session-local refractory semantics to fixed V1.
    clustered = []
    last = {}
    for ix in raw:
        session = int(sessions[ix]); t = int(ts[ix])
        if t - last.get(session, -10**30) >= 2_000_000_000:
            clustered.append(int(ix)); last[session] = t
    return raw, np.asarray(clustered, dtype=np.int64)


def _session_codes(day: str, ts: np.ndarray) -> np.ndarray:
    codes = np.full(len(ts), -1, dtype=np.int8)
    for code, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = baseline._session_windows(day)[name]
        codes[(ts >= start) & (ts < end)] = code
    return codes


def _quote_arrays(tape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.ascontiguousarray(tape["timestamp_ns"], dtype=np.int64),
            np.ascontiguousarray(tape["session"], dtype=np.int8),
            np.asarray(tape["bid"], dtype=np.float64), np.asarray(tape["ask"], dtype=np.float64))


def _entry_markout_arrays(tape: np.ndarray, signal_ns: np.ndarray, direction: np.ndarray,
                          session: np.ndarray, event_mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tt, ss, bid, ask = _quote_arrays(tape)
    ix = np.searchsorted(tt, signal_ns + 2_000_000, side="left")
    inside = ix < len(tt)
    safe = np.minimum(ix, max(0, len(tt)-1))
    valid = inside & (ss[safe] == session) & np.isfinite(bid[safe]) & np.isfinite(ask[safe]) & (ask[safe] > bid[safe])
    fill = np.where(direction > 0, ask[safe] + TICK, bid[safe] - TICK)
    chase = direction * (fill - event_mid) / TICK
    return valid, chase


def _markout_values(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean_ticks": None, "median_ticks": None, "trimmed_mean_ticks": None,
                "p25_ticks": None, "p75_ticks": None, "positive_fraction": None, "negative_fraction": None}
    ordered = np.sort(x); cut = int(len(x) * .10)
    trimmed = ordered[cut:len(x)-cut] if cut else ordered
    return {"n": len(x), "mean_ticks": float(np.mean(x)), "median_ticks": float(np.median(x)),
            "trimmed_mean_ticks": float(np.mean(trimmed)), "p25_ticks": float(np.quantile(x, .25)),
            "p75_ticks": float(np.quantile(x, .75)), "positive_fraction": float(np.mean(x > 0)),
            "negative_fraction": float(np.mean(x < 0))}


def _markout_summary(rows: Sequence[Mapping[str, Any]], key: str = "markouts_ticks") -> dict[str, Any]:
    return {str(h): _markout_values([row[key][str(h)] for row in rows
                                     if row.get(key, {}).get(str(h)) is not None]) for h in HORIZONS_MS}


def _signed_entry_markouts(tape: np.ndarray, entry_ns: int, entry_price: float, direction: int,
                           session: int) -> dict[str, float | None]:
    tt, ss, bid, ask = _quote_arrays(tape)
    lo, hi = native._session_bounds_ns_for_code(session, tape, sessions=ss)
    result = {}
    for horizon in HORIZONS_MS:
        ix = native._first_at_or_after(tt, entry_ns + horizon * 1_000_000, lo, hi)
        result[str(horizon)] = (None if ix is None or ss[ix] != session else
                                float(direction * ((bid[ix] if direction > 0 else ask[ix]) - entry_price) / TICK))
    return result


def _event_markout_summary(tape: np.ndarray, signal_ns: np.ndarray, event_mid: np.ndarray,
                           direction: np.ndarray, session: np.ndarray) -> dict[str, Any]:
    tt, ss, bid, ask = _quote_arrays(tape)
    mid = (bid + ask) / 2
    out = {}
    for horizon in HORIZONS_MS:
        ix = np.searchsorted(tt, signal_ns + horizon * 1_000_000, side="left")
        inside = ix < len(tt); safe = np.minimum(ix, max(0, len(tt)-1))
        good = inside & (ss[safe] == session) & np.isfinite(mid[safe])
        values = direction[good] * (mid[safe[good]] - event_mid[good]) / TICK
        out[str(horizon)] = _markout_values(values)
    return out


def _causal_price_features(tape: np.ndarray, at_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Price-only features using as-of quotes no later than each signal."""
    tt, ss, bid, ask = _quote_arrays(tape)
    mid = (bid + ask) / 2
    if not len(tt):
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    offsets = np.asarray((0, 500, 1000, 1500, 2000), dtype=np.int64) * 1_000_000
    indices = np.searchsorted(tt, at_ns[:, None] - offsets[None, :], side="right") - 1
    safe = np.maximum(indices, 0)
    values = mid[safe]
    valid = np.all(indices >= 0, axis=1) & np.all(ss[safe] == ss[safe[:, 0], None], axis=1)
    valid &= np.all(np.isfinite(values), axis=1)
    diff = np.diff(values, axis=1) / TICK
    features = np.column_stack(((values[:, 0] - values[:, 1]) / TICK,
                                (values[:, 0] - values[:, 4]) / TICK,
                                np.std(diff, axis=1)))
    return features, valid


def _control_pool(day: str, tape: np.ndarray) -> dict[str, np.ndarray]:
    windows = baseline._session_windows(day)
    grids, session_codes = [], []
    for code, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = windows[name]
        grid = np.arange(start + 2_000_000_000, end, 2_000_000_000, dtype=np.int64)
        grids.append(grid); session_codes.append(np.full(len(grid), code, dtype=np.int8))
    signals = np.concatenate(grids); sessions = np.concatenate(session_codes)
    features, valid = _causal_price_features(tape, signals)
    tt, ss, bid, ask = _quote_arrays(tape)
    eix = np.searchsorted(tt, signals + 2_000_000, side="left")
    inside = eix < len(tt); safe = np.minimum(eix, max(0, len(tt)-1))
    valid &= inside & (ss[safe] == sessions) & (ask[safe] > bid[safe]) & np.isfinite(ask[safe]) & np.isfinite(bid[safe])
    direction = np.sign(np.where(features[:, 0] != 0, features[:, 0], features[:, 1])).astype(np.int8)
    valid &= direction != 0
    return {"time": signals[valid], "session": sessions[valid], "hour": (signals[valid] // 3_600_000_000_000),
            "direction": direction[valid], "features": features[valid], "entry_ix": eix[valid]}


def _match_price_controls(day: str, tape: np.ndarray, trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fixed, outcome-blind nearest causal price-only control within hour."""
    pool = _control_pool(day, tape)
    tt, ss, bid, ask = _quote_arrays(tape)
    if not trades:
        return {"matched": 0, "unmatched": 0, "controls": [], "method": CONTROL_SPEC}
    at = np.asarray([int(t["event_start_time_ns"]) for t in trades], dtype=np.int64)
    features, feature_valid = _causal_price_features(tape, at)
    used: set[int] = set(); controls = []; unmatched = 0
    scale = np.asarray((2.0, 4.0, 2.0))
    for i, trade in enumerate(trades):
        if not feature_valid[i]:
            unmatched += 1; continue
        direction = 1 if trade["entry_direction"] == "LONG" else -1
        signal_ns = int(trade["event_start_time_ns"])
        group = np.flatnonzero((pool["session"] == int(trade["session_code"])) &
                               (pool["hour"] == signal_ns // 3_600_000_000_000) &
                               (pool["direction"] == direction) &
                               (np.abs(pool["time"] - signal_ns) > 2_000_000_000))
        if not len(group):
            unmatched += 1; continue
        distance = np.sum(((pool["features"][group] - features[i]) / scale) ** 2, axis=1)
        order = np.lexsort((pool["time"][group], distance))
        pick = next((int(group[j]) for j in order if int(group[j]) not in used), None)
        if pick is None:
            unmatched += 1; continue
        used.add(pick)
        ix = int(pool["entry_ix"][pick]); entry = float(ask[ix] + TICK if direction > 0 else bid[ix] - TICK)
        controls.append({"date": day, "flow_entry_time_ns": int(trade["entry_time_ns"]),
                         "signal_time_ns": int(pool["time"][pick]),
                         "entry_time_ns": int(tt[ix]), "direction": direction,
                         "distance": float(distance[np.where(group == pick)[0][0]]),
                         "markouts_ticks": _signed_entry_markouts(tape, int(tt[ix]), entry, direction,
                                                                   int(pool["session"][pick])),
                         "excursions_by_horizon": _excursions(tape, int(tt[ix]), entry, direction,
                                                              int(pool["session"][pick]))})
    return {"matched": len(controls), "unmatched": unmatched, "controls": controls, "method": CONTROL_SPEC}


def _excursions(tape: np.ndarray, entry_ns: int, entry_price: float, direction: int,
                session: int) -> dict[str, Any]:
    tt, ss, bid, ask = _quote_arrays(tape)
    lo, hi = native._session_bounds_ns_for_code(session, tape, sessions=ss)
    first = native._first_at_or_after(tt, entry_ns, lo, hi)
    result = {}
    for horizon in EXCURSION_HORIZONS_MS:
        end = native._first_at_or_after(tt, entry_ns + horizon * 1_000_000, lo, hi)
        if first is None or end is None or ss[end] != session:
            result[str(horizon)] = {"mfe_ticks": None, "mae_ticks": None}
            continue
        references = bid[first:end+1] if direction > 0 else ask[first:end+1]
        movement = direction * (references - entry_price) / TICK
        result[str(horizon)] = {"mfe_ticks": float(max(0.0, np.max(movement))),
                                "mae_ticks": float(max(0.0, -np.min(movement)))}
    return result


def evaluate_day(day: str, rows: np.ndarray, tape: np.ndarray,
                 history: Mapping[str, Sequence[float]], pressure: np.ndarray | None = None) -> dict[str, Any]:
    """One fixed decision path; all event gates use information known by entry."""
    pressure = native.rolling_pressure(rows, 500_000_000) if pressure is None else pressure
    thresholds = prior_thresholds(history)
    ts = np.ascontiguousarray(rows["ts"], dtype=np.int64)
    mid = np.asarray(rows["mid"], dtype=np.float64)
    sessions = _session_codes(day, ts)
    raw, clustered = cluster_flow(ts, pressure, sessions, thresholds)
    direction = np.where(pressure[clustered] > 0, 1, -1).astype(np.int8)
    signal_ns = ts[clustered]
    event_mid = mid[clustered]
    tt, ss, bid, ask = _quote_arrays(tape)
    bounds = {s: native._session_bounds_ns_for_code(s, tape, sessions=ss) for s in (0, 1, 2)}
    baseline_markouts = _event_markout_summary(tape, signal_ns, event_mid, direction, sessions[clustered])
    valid_quote, chase = _entry_markout_arrays(tape, signal_ns, direction, sessions[clustered], event_mid)
    after_chase = valid_quote & (chase <= 6 + 1e-12)
    prior_ix = np.searchsorted(ts, signal_ns - 500_000_000, side="right") - 1
    safe_prior = np.maximum(prior_ix, 0)
    prior_valid = (prior_ix >= 0) & (sessions[safe_prior] == sessions[clustered])
    prior_movement = direction * (event_mid - mid[safe_prior]) / TICK
    after_confirmation = after_chase & prior_valid & (prior_movement >= -1e-12)
    sides = {"LONG": pressure[clustered] > 0, "SHORT": pressure[clustered] < 0}
    counts: dict[str, dict[str, int]] = {}
    for side, mask in sides.items():
        sign = 1 if side == "LONG" else -1
        counts[side] = {"raw_pressure_observations": int(np.sum(pressure[raw] * sign > 0)),
                        "clustered_events": int(np.sum(mask)),
                        "after_max_chase": int(np.sum(after_chase & mask)),
                        "after_price_confirmation": int(np.sum(after_confirmation & mask)),
                        "after_position_refractory": 0, "actual_entries": 0}
    sorted_history = {side: np.sort(np.asarray(history[side], dtype=np.float64)) for side in sides}
    trades = []; busy_until = -10**30
    for j in np.flatnonzero(after_confirmation):
        event_time = int(signal_ns[j]); sign = int(direction[j])
        side = "LONG" if sign > 0 else "SHORT"
        if event_time < busy_until:
            continue
        counts[side]["after_position_refractory"] += 1
        pct = float(np.searchsorted(sorted_history[side], abs(float(pressure[clustered[j]])), side="right") /
                    len(sorted_history[side]))
        signal = {"timestamp_ns": event_time, "signal_anchor_ns": event_time,
                  "session": int(sessions[clustered[j]]), "direction": sign,
                  "event_start_mid": float(event_mid[j]), "pressure": float(pressure[clustered[j]]),
                  "pressure_percentile": pct, "confirmation_change_ticks": float(prior_movement[j])}
        trade = native._entry_and_exit(tape, signal, busy_until, tt, ss, bounds,
                                       stop_ticks=6, target_r=1.5,
                                       max_favorable_move_before_entry_ticks=6,
                                       max_hold_seconds=10)
        if trade is None:
            continue
        trade["session_code"] = int(sessions[clustered[j]])
        trade["flow_strength_percentile"] = pct
        trade["excursions_by_horizon"] = _excursions(tape, int(trade["entry_time_ns"]),
                                                     float(trade["entry_price"]), sign,
                                                     int(trade["session_code"]))
        trades.append(trade)
        counts[side]["actual_entries"] += 1
        busy_until = int(trade["exit_time_ns"]) + 2_000_000_000
    control = _match_price_controls(day, tape, trades)
    attrition = {key: sum(counts[side][key] for side in sides) for key in counts["LONG"]}
    print(f"FLOW_DATE_EVALUATED={day} raw={len(raw)} clustered={len(clustered)} chase={attrition['after_max_chase']} confirmed={attrition['after_price_confirmation']} trades={len(trades)} controls={control['matched']}", flush=True)
    return {"date": day, "thresholds": thresholds, "pressure_history_sample": _sample_sides(pressure),
            "pressure_observation_count": len(pressure), "pressure_event_baseline": baseline_markouts,
            "attrition": attrition, "attrition_by_direction": counts, "trades": trades,
            "price_only_control": control}


def _excursion_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for horizon in EXCURSION_HORIZONS_MS:
        entries = [r.get("excursions_by_horizon", {}).get(str(horizon), {}) for r in rows]
        mfe = [float(v["mfe_ticks"]) for v in entries if v.get("mfe_ticks") is not None]
        mae = [float(v["mae_ticks"]) for v in entries if v.get("mae_ticks") is not None]
        output[str(horizon)] = {"count": len(mfe), "mean_mfe_ticks": float(np.mean(mfe)) if mfe else None,
                                "median_mfe_ticks": float(np.median(mfe)) if mfe else None,
                                "mean_mae_ticks": float(np.mean(mae)) if mae else None,
                                "median_mae_ticks": float(np.median(mae)) if mae else None}
    return output


def _aggregate_baseline(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for period, days in (("ALL", TARGET_DATES), ("SPRING_2025", SPRING_DATES), ("OCTOBER_2025", OCTOBER_DATES)):
        selected = [p for p in payloads if p["date"] in days]
        out[period] = {}
        for h in HORIZONS_MS:
            rows = [p["pressure_event_baseline"][str(h)] for p in selected]
            n = sum(int(r["n"]) for r in rows)
            out[period][str(h)] = {"events": sum(p["attrition"]["clustered_events"] for p in selected),
                                   "n": n, "mean_ticks": (sum(r["n"] * r["mean_ticks"] for r in rows
                                                           if r["mean_ticks"] is not None) / n if n else None)}
    return out


def _strength_shape(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    edges = ((.90, .92), (.92, .94), (.94, .96), (.96, .98), (.98, 1.000001))
    buckets = {}
    for lo, hi in edges:
        name = f"{lo*100:.0f}-{min(hi,1)*100:.0f}"
        group = [t for t in trades if lo <= float(t["flow_strength_percentile"]) < hi]
        buckets[name] = {"trade_count": len(group),
                         "markouts": {str(h): _markout_values([t["markouts_ticks"][str(h)] for t in group
                                                               if t["markouts_ticks"].get(str(h)) is not None])
                                      for h in (500, 1000, 2000, 5000, 10000)},
                         "average_R": float(np.mean([t["net_R"] for t in group])) if group else None,
                         "mean_MFE_ticks": float(np.mean([t["mfe_ticks"] for t in group])) if group else None,
                         "mean_MAE_ticks": float(np.mean([t["mae_ticks"] for t in group])) if group else None}
    observed = [(key, value["markouts"]["2000"]["mean_ticks"]) for key, value in buckets.items()
                if value["trade_count"] >= 10 and value["markouts"]["2000"]["mean_ticks"] is not None]
    means = [v for _, v in observed]
    if len(means) < 3:
        shape = "NO_CLEAR_SHAPE"
    elif max(means) - min(means) <= .5:
        shape = "PLATEAU"
    elif all(means[i] <= means[i+1] for i in range(len(means)-1)):
        shape = "MONOTONIC_IMPROVEMENT"
    elif means[-1] < min(means[:-1]) and observed[-1][0] == "98-100":
        shape = "EXTREME_TAIL_EXHAUSTION"
    elif max(means[1:-1]) > max(means[0], means[-1]):
        shape = "INVERTED_U"
    else:
        shape = "NO_CLEAR_SHAPE"
    return {"classification": shape, "buckets": buckets, "diagnostic_only": True}


def _leave_one_out(trades: Sequence[Mapping[str, Any]], groups: Sequence[str], key: str) -> dict[str, Any]:
    full = native._summarize_trades(trades)
    base_sign = int(np.sign(full["net_R"]))
    rows = []
    for omitted in groups:
        remaining = [t for t in trades if t[key] != omitted]
        s = native._summarize_trades(remaining)
        rows.append({"omitted": omitted, "net_R": s["net_R"], "average_R": s["average_R"],
                     "trade_count": s["trade_count"]})
    net = [r["net_R"] for r in rows]
    avg = [r["average_R"] for r in rows if r["average_R"] is not None]
    best = max(rows, key=lambda r: r["net_R"]) if rows else None
    worst = min(rows, key=lambda r: r["net_R"]) if rows else None
    return {"groups_tested": len(rows), "sign_stability": (sum(int(np.sign(x)) == base_sign for x in net) / len(net) if net else None),
            "net_R": {"median": float(np.median(net)) if net else None, "min": min(net, default=None),
                      "max": max(net, default=None)},
            "average_R": {"median": float(np.median(avg)) if avg else None,
                          "min": min(avg, default=None), "max": max(avg, default=None)},
            "worst_omitted": worst, "best_omitted": best, "rows": rows}


def _week(day: str) -> str:
    iso = date.fromisoformat(day).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _aggregate(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trades = sorted((t for p in payloads for t in p["trades"]), key=lambda t: (t["entry_time_ns"], t["date"]))
    for t in trades:
        t["week"] = _week(t["date"])
    spring = [t for t in trades if t["date"] in SPRING_DATES]
    october = [t for t in trades if t["date"] in OCTOBER_DATES]
    longs = [t for t in trades if t["entry_direction"] == "LONG"]
    shorts = [t for t in trades if t["entry_direction"] == "SHORT"]
    perf = native._summarize_trades(trades)
    periods = {"SPRING_2025": native._summarize_trades(spring),
               "OCTOBER_2025": native._summarize_trades(october)}
    a, b = periods["SPRING_2025"]["net_R"], periods["OCTOBER_2025"]["net_R"]
    periods["classification"] = ("BOTH_POSITIVE" if a >= 0 and b >= 0 else
                                 "SPRING_ONLY" if a >= 0 else "OCTOBER_ONLY" if b >= 0 else "BOTH_NEGATIVE")
    directions = {"LONG": native._summarize_trades(longs), "SHORT": native._summarize_trades(shorts)}
    la, sh = directions["LONG"]["net_R"], directions["SHORT"]["net_R"]
    directions["classification"] = ("INSUFFICIENT" if not longs or not shorts else
                                    "SYMMETRIC" if la > 0 and sh > 0 else
                                    "LONG_DOMINANT" if la > 0 else "SHORT_DOMINANT" if sh > 0 else "OPPOSITE")
    daily, weekly = native._daily_weekly(trades, TARGET_DATES)
    for row in daily:
        group = [t for t in trades if t["date"] == row["date"]]
        for h in (2000, 5000):
            vals = [t["markouts_ticks"][str(h)] for t in group if t["markouts_ticks"].get(str(h)) is not None]
            row[f"mean_{h//1000}s_markout_ticks"] = float(np.mean(vals)) if vals else None
    day_r = [x["R"] for x in daily]
    daily_summary = {"positive_days": sum(x > 0 for x in day_r), "negative_days": sum(x < 0 for x in day_r),
                     "flat_days": sum(x == 0 for x in day_r), "mean_daily_R": float(np.mean(day_r)),
                     "median_daily_R": float(np.median(day_r)),
                     "best_5_days": sorted(daily, key=lambda x: (-x["R"], x["date"]))[:5],
                     "worst_5_days": sorted(daily, key=lambda x: (x["R"], x["date"]))[:5]}
    weekly_summary = {"positive_weeks": sum(x["R"] > 0 for x in weekly),
                      "negative_weeks": sum(x["R"] < 0 for x in weekly),
                      "flat_weeks": sum(x["R"] == 0 for x in weekly)}
    subsets = {"all": trades, "spring": spring, "october": october, "long": longs, "short": shorts,
               "wins": [t for t in trades if t["net_R"] > 0], "losses": [t for t in trades if t["net_R"] < 0]}
    marks = {key: _markout_summary(group) for key, group in subsets.items()}
    excursions = {key: _excursion_summary(group) for key, group in subsets.items()}
    first_touch = {key: {barrier: {outcome: sum(t["first_touch"][barrier] == outcome for t in group)
                                 for outcome in ("UP", "DOWN", "TIE", "NO_TOUCH")}
                          for barrier in ("+1/-1", "+2/-2", "+4/-4", "+8/-4")}
                   for key, group in subsets.items()}
    baseline = _aggregate_baseline(payloads)
    attrition = {period: {side: {key: sum(p["attrition_by_direction"][side][key] for p in payloads if p["date"] in days)
                                 for key in payloads[0]["attrition_by_direction"]["LONG"]}
                          for side in ("LONG", "SHORT")}
                 for period, days in (("SPRING", SPRING_DATES), ("OCTOBER", OCTOBER_DATES))}
    all_controls = [c for p in payloads for c in p["price_only_control"]["controls"]]
    flow_lookup = {(t["date"], int(t["entry_time_ns"])): t for t in trades}
    matched_flow = [flow_lookup[(c["date"], c["flow_entry_time_ns"])] for c in all_controls]
    flow_marks = _markout_summary(matched_flow); control_marks = _markout_summary(all_controls)
    comparison = {str(h): {"flow_mean_ticks": flow_marks[str(h)]["mean_ticks"],
                           "price_only_mean_ticks": control_marks[str(h)]["mean_ticks"],
                           "incremental_ticks": (flow_marks[str(h)]["mean_ticks"] - control_marks[str(h)]["mean_ticks"]
                                                 if flow_marks[str(h)]["mean_ticks"] is not None and control_marks[str(h)]["mean_ticks"] is not None else None)}
                  for h in (500, 1000, 2000, 5000, 10000)}
    controls = {"method": CONTROL_SPEC, "matched": len(all_controls),
                "unmatched": sum(p["price_only_control"]["unmatched"] for p in payloads),
                "flow_matched_markouts": flow_marks, "price_only_markouts": control_marks,
                "flow_matched_mfe_mae": _excursion_summary(matched_flow),
                "price_only_mfe_mae": _excursion_summary(all_controls), "comparison": comparison}
    lodo = _leave_one_out(trades, TARGET_DATES, "date")
    lowo = _leave_one_out(trades, sorted({_week(d) for d in TARGET_DATES}), "week")
    return {"trades": trades, "performance": perf, "period_results": periods,
            "direction_results": directions, "daily_results": daily, "daily_summary": daily_summary,
            "weekly_results": weekly, "weekly_summary": weekly_summary, "markouts": marks,
            "mfe_mae": excursions, "first_touch": first_touch, "pressure_event_baseline": baseline,
            "price_momentum_control": controls, "pressure_strength_diagnostic": _strength_shape(trades),
            "attrition": attrition, "lodo": lodo, "lowo": lowo}


def _decision(a: Mapping[str, Any]) -> tuple[str, str, bool, bool]:
    perf = a["performance"]; periods = a["period_results"]; dirs = a["direction_results"]
    mark = a["markouts"]["all"]; controls = a["price_momentum_control"]["comparison"]
    increments = [controls[str(h)]["incremental_ticks"] for h in (500, 1000, 2000, 5000, 10000)]
    l2_adds = all(v is not None and v > 0 for v in increments[:2]) and sum(v for v in increments if v is not None) > 0
    continuation = all(mark[str(h)]["mean_ticks"] is not None and mark[str(h)]["mean_ticks"] > 0 for h in (500, 1000))
    active_weeks = [x for x in a["weekly_results"] if x["trades"] > 0]
    compatible = periods["SPRING_2025"]["net_R"] >= 0 and periods["OCTOBER_2025"]["net_R"] >= 0
    reasonable_dd = perf["max_drawdown_R"] <= max(1.0, perf["net_R"])
    directional = dirs["LONG"]["net_R"] >= 0 and dirs["SHORT"]["net_R"] >= 0
    good = (perf["net_R"] > 0 and perf["average_R"] is not None and perf["average_R"] > 0
            and isinstance(perf["profit_factor"], (int, float)) and perf["profit_factor"] >= 1.10
            and compatible and reasonable_dd and directional and continuation and l2_adds
            and sum(x["R"] >= 0 for x in active_weeks) > len(active_weeks) / 2)
    if good:
        return "FLOW_MOMENTUM_V1_PROMISING", "RUN_SMALL_ROBUSTNESS_GRID_BEFORE_OPTUNA", l2_adds, continuation
    if perf["net_R"] > 0:
        return "FLOW_MOMENTUM_V1_MIXED", "DIAGNOSE_FLOW_STRENGTH_SHAPE_AND_REGIME_DEPENDENCE", l2_adds, continuation
    return "FLOW_MOMENTUM_V1_FAILED", "STOP_FLOW_MOMENTUM_BRANCH", l2_adds, continuation


def _checkpoint_path(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, *, day: str, source_sha: str, tape_sha: str,
                     prior_chain: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    expected = {"checkpoint_version": CHECKPOINT_VERSION, "date": day,
                "source_sha256": source_sha, "tape_sha256": tape_sha,
                "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256,
                "prior_source_chain_sha256": prior_chain, "status": "DATE_COMPLETE"}
    return record if all(record.get(key) == value for key, value in expected.items()) else None


def _cached_compact(day: str, source: Path, source_sha: str, root: Path) -> np.ndarray:
    cache_roots = (root / "_cache", Path("research_runs/CMEOrderflow_ES_LIQUIDITY_VACUUM_V1_RESCUE_OPTUNA/_cache"))
    for cache_root in cache_roots:
        path = cache_root / f"{day}.compact.bin"; meta_path = cache_root / f"{day}.compact.json"
        if not (path.is_file() and meta_path.is_file()):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (meta.get("source_sha256") == source_sha and meta.get("bytes") == path.stat().st_size
                and meta.get("dtype") == repr(relative.COMPACT_DTYPE.descr)
                and native._sha(path) == meta.get("sha256")):
            return np.memmap(path, dtype=relative.COMPACT_DTYPE, mode="r", shape=(int(meta["rows"]),))
    cache_root = root / "_cache"; cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{day}.compact.bin"
    rows, scratch, provenance = relative._extract_compact(day, source, path, source_sha)
    count = len(rows); del rows
    os.replace(scratch, path)
    native._write_json(cache_root / f"{day}.compact.json",
                       {"source_sha256": source_sha, "rows": count, "bytes": path.stat().st_size,
                        "dtype": repr(relative.COMPACT_DTYPE.descr), "sha256": native._sha(path),
                        "raw_rows": provenance["raw_rows"]})
    return np.memmap(path, dtype=relative.COMPACT_DTYPE, mode="r", shape=(count,))


def run(*, data_root: Path = native.DATA_ROOT, output_root: Path = OUT_ROOT,
        smoke: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    paths, manifest_rows = native._source_catalog(data_root)
    train_manifest = json.loads((native.TRAIN_TAPE_ROOT.parent / "train-tape-manifest.json").read_text())
    oct_manifest = json.loads((native.OCT_TAPE_ROOT.parent / "october-tape-manifest.json").read_text())
    if train_manifest.get("status") != "COMPLETE" or train_manifest.get("train_dates") != list(SPRING_DATES):
        raise FlowStudyError("Spring native ES tape manifest does not match frozen dates")
    if oct_manifest.get("status") != "COMPLETE" or oct_manifest.get("validation_dates") != list(OCTOBER_DATES):
        raise FlowStudyError("October native ES tape manifest does not match frozen dates")
    source_days = (DEPENDENCY_DATES[0], SPRING_DATES[0]) if smoke else SOURCE_DATES
    target_days = (SPRING_DATES[0],) if smoke else TARGET_DATES
    tape_hashes = {}
    for day in target_days:
        path = native._tape_path(day)
        declared = (train_manifest if day in SPRING_DATES else oct_manifest).get("source_sha256_by_date", {}).get(day)
        if not path.is_file() or declared != manifest_rows[day]["sha256"]:
            raise FlowStudyError(f"canonical tape source binding failure: {day}")
        tape_hashes[day] = native._sha(path)
    output_root.mkdir(parents=True, exist_ok=True)
    native._write_json(output_root / "strategy-config.json", CONFIG)
    native._write_json(output_root / "study-spec.json", STUDY_SPEC)
    coverage = {"status": "PASS", "dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES",
                "spring_dates": list(SPRING_DATES), "october_dates": list(OCTOBER_DATES),
                "dependency_dates": list(DEPENDENCY_DATES), "native_es_only": True,
                "no_mbo": True, "no_mes_market_data": True,
                "source_files": {day: {"path": str(paths[day]), "sha256": manifest_rows[day]["sha256"],
                                       "bytes": paths[day].stat().st_size,
                                       "symbol": manifest_rows[day]["symbol"],
                                       "category": manifest_rows[day]["category"]} for day in SOURCE_DATES},
                "canonical_tapes": {day: {"path": str(native._tape_path(day)), "sha256": tape_hashes[day]}
                                    for day in target_days}}
    native._write_json(output_root / "source-coverage.json", coverage)
    history: dict[str, list[float]] = {"LONG": [], "SHORT": []}
    chain = "NO_PRIOR_SOURCE"
    payloads = {}; completed = []; resumed = []
    for position, day in enumerate(source_days, 1):
        source_sha = str(manifest_rows[day]["sha256"])
        tape_sha = tape_hashes.get(day, "DEPENDENCY_ONLY")
        path = _checkpoint_path(output_root, day)
        cached = _read_checkpoint(path, day=day, source_sha=source_sha,
                                  tape_sha=tape_sha, prior_chain=chain)
        if cached is not None:
            payload = cached["payload"]
            resumed.append(day)
            print(f"FLOW_DATE_RESUME={day}", flush=True)
        else:
            print(f"FLOW_DATE_START={position}/{len(source_days)} date={day}", flush=True)
            rows = _cached_compact(day, paths[day], source_sha, output_root)
            pressure = native.rolling_pressure(rows, 500_000_000)
            if day in target_days:
                tape, _ = native._load_tape(day, native._tape_path(day), source_sha)
                payload = evaluate_day(day, rows, tape, history, pressure)
            else:
                payload = {"date": day, "dependency_only": True,
                           "pressure_history_sample": _sample_sides(pressure),
                           "pressure_observation_count": len(pressure)}
            del rows, pressure
            native._write_checkpoint(path, {"checkpoint_version": CHECKPOINT_VERSION,
                                            "status": "DATE_COMPLETE", "date": day,
                                            "source_sha256": source_sha, "tape_sha256": tape_sha,
                                            "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256,
                                            "prior_source_chain_sha256": chain, "payload": payload})
            completed.append(day)
        for side in ("LONG", "SHORT"):
            history[side].extend(payload["pressure_history_sample"][side])
        if day in target_days:
            payloads[day] = payload
        chain = _hash_obj([chain, source_sha])
        native._write_json(output_root / "checkpoints" / "progress.json",
                           {"last_date": day, "completed_dates": completed, "resumed_dates": resumed,
                            "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256,
                            "source_chain_sha256": chain})
    if set(payloads) != set(target_days):
        raise FlowStudyError(f"missing target checkpoints: {sorted(set(target_days)-set(payloads))}")
    if smoke:
        return {"status": "SMOKE_PASS", "dates": list(payloads),
                "trades": sum(len(p["trades"]) for p in payloads.values())}
    aggregate = _aggregate([payloads[day] for day in target_days])
    trades = aggregate.pop("trades")
    decision, next_step, l2_adds, executable = _decision(aggregate)
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "dataset": "SPRING_2025 + OCTOBER_2025",
               "spring_dates": list(SPRING_DATES), "october_dates": list(OCTOBER_DATES),
               "native_es_only": True, "flow_formula": STUDY_SPEC["flow_formula"],
               "strategy_config": CONFIG,
               "execution_model": {"entry": "first canonical ES quote at or after signal+2ms; long ask+1tick, short bid-1tick",
                                   "stop": "exit-side BBO crosses six-tick stop; adverse one-tick fill",
                                   "target": "exit-side BBO crosses nine-tick target; adverse one-tick fill",
                                   "same_timestamp_precedence": "STOP before TARGET",
                                   "time_exit": "first executable BBO at or after entry+10s; reject entry if full hold crosses session end",
                                   "fees": "ES $3/side/contract; MES fallback $1.25/side/contract if ES sizing yields zero",
                                   "economics": "ES $50/point; MES fallback $5/point; native ES price path only",
                                   "policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS},
               "performance": aggregate["performance"], "period_results": aggregate["period_results"],
               "direction_results": aggregate["direction_results"],
               "daily_summary": aggregate["daily_summary"], "weekly_summary": aggregate["weekly_summary"],
               "pressure_strength_shape": aggregate["pressure_strength_diagnostic"]["classification"],
               "l2_adds_incremental_information": l2_adds, "executable_at_500ms_plus": executable,
               "primary_decision": decision, "next_step": next_step,
               "optimization_performed": False, "optuna_performed": False,
               "threshold_search_performed": False, "vacuum_filters_used": False,
               "level_filters_used": False, "final_oos_accessed": False,
               "data_downloaded": False, "elapsed_seconds": time.monotonic()-started,
               "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256}
    native._write_gzip_jsonl(output_root / "trades.jsonl.gz", trades)
    files = {
        "summary.json": summary,
        "performance-summary.json": aggregate["performance"],
        "period-results.json": aggregate["period_results"],
        "direction-results.json": aggregate["direction_results"],
        "daily-results.json": {"summary": aggregate["daily_summary"], "dates": aggregate["daily_results"]},
        "weekly-results.json": {"summary": aggregate["weekly_summary"], "weeks": aggregate["weekly_results"]},
        "pressure-strength-diagnostic.json": aggregate["pressure_strength_diagnostic"],
        "pressure-event-baseline.json": aggregate["pressure_event_baseline"],
        "price-momentum-control.json": aggregate["price_momentum_control"],
        "markouts.json": aggregate["markouts"], "mfe-mae.json": aggregate["mfe_mae"],
        "first-touch.json": aggregate["first_touch"], "attrition.json": aggregate["attrition"],
        "lodo-results.json": aggregate["lodo"], "lowo-results.json": aggregate["lowo"],
    }
    for name, contents in files.items():
        native._write_json(output_root / name, contents)
    report = [f"# {RUN_ID}", "", f"Decision: **{decision}**", "",
              "Frozen native ES MBP-10, 500ms TOP5 pressure, side-specific prior-date q90, no Vacuum or level filter.",
              f"Trades: {aggregate['performance']['trade_count']}; net R: {aggregate['performance']['net_R']:.4f}; PF: {aggregate['performance']['profit_factor']}.",
              f"Spring R: {aggregate['period_results']['SPRING_2025']['net_R']:.4f}; October R: {aggregate['period_results']['OCTOBER_2025']['net_R']:.4f}.",
              f"L2 incremental to matched price-only controls: {l2_adds}; executable 500ms continuation: {executable}.",
              "", f"Next step: {next_step}", ""]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    checkpoint_hashes = {day: native._sha(_checkpoint_path(output_root, day)) for day in source_days}
    native._write_json(output_root / "run-manifest.json",
                       {"status": "COMPLETE", "run_id": RUN_ID, "config_sha256": CONFIG_SHA256,
                        "study_sha256": STUDY_SHA256, "source_coverage_sha256": native._sha(output_root / "source-coverage.json"),
                        "source_sha256_by_date": {day: manifest_rows[day]["sha256"] for day in source_days},
                        "tape_sha256_by_date": tape_hashes, "checkpoint_sha256_by_date": checkpoint_hashes,
                        "no_2026_access": True, "no_download": True, "no_optimization": True})
    hashes = {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name != "artifact-hashes.json"}
    native._write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files": hashes})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=native.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(data_root=args.data_root, output_root=args.output_root, smoke=args.smoke)
    except (FlowStudyError, native.VacuumStudyError, relative.StudyError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({"status": result["status"], "decision": result.get("primary_decision"),
                      "trades": result.get("performance", {}).get("trade_count", result.get("trades"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
