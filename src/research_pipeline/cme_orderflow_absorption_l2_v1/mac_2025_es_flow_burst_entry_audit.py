"""Fixed q90 burst-state and signal-to-execution audit on native ES MBP-10.

No position simulation, PnL, threshold search, or 2026 source is used here.
The only q98 path is variant E with the predeclared C reset rule.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_es_flow_momentum_v1 as flow
from . import mac_2025_es_liquidity_vacuum_v1 as native

RUN_ID = "CMEOrderflow_ES_FLOW_BURST_ENTRY_AUDIT_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
HORIZONS_MS = (250, 500, 1000, 2000, 5000, 10000)
BREAKDOWN_MS = HORIZONS_MS[:5]
VARIANTS = {"A": {"q": .90, "reset": "below_q90_for_500ms"},
            "B": {"q": .90, "reset": "below_q90_for_1000ms"},
            "C": {"q": .90, "reset": "below_q90_for_2000ms"},
            "D": {"q": .90, "reset": "pressure_zero_or_opposite_sign"},
            "E": {"q": .98, "reset": "below_q98_for_2000ms"}}
STUDY_SPEC = {
    "version": 1, "source_flow_config_sha256": flow.CONFIG_SHA256,
    "source_flow_study_sha256": flow.STUDY_SHA256,
    "data": "local native ES GLBX.MDP3 mbp-10 on exactly 35 Spring and 19 October 2025 target dates; two dependency dates only for prior history",
    "formula": flow.STUDY_SPEC["flow_formula"],
    "thresholds": "side-specific q90 prior-date samples; one q98 robustness path with identical prior-date chronology",
    "variants": VARIANTS,
    "reset_observation": "pressure state is as-of/piecewise constant between native MBP-10 rows; no reset inferred from a future row before its timestamp; reset at each named session boundary",
    "directional_reset": "LONG re-arms only after signed normalized pressure <=0; SHORT only after >=0",
    "q98_selected_before_outcomes": "C: the most conservative duration-based q90 reset; no outcome-based selection",
    "structural_reference": "C is the predeclared primary first-burst comparison; A/B/D are diagnostics, not performance-ranked candidates",
    "old_stream": "unchanged side-specific q90 hits and first-hit two-second session-local refractory from Flow Momentum V1",
    "ordinal": "old clustered same-direction event index within each causal burst; EVENT_5_PLUS combines ordinal >=5",
    "ordinal_classification": "using variant C 500ms executable markouts: first >= later+0.5 tick and first>0 STRONGLY_BETTER; >=0.25 MODESTLY_BETTER; later >= first+0.25 LATER_EVENTS_BETTER; |difference|<0.25 NO_EFFECT; otherwise MIXED",
    "execution": "at/after event+2ms first same-session executable ES BBO; quote path uses ask/bid; actual fill adds one adverse tick; all markouts use future exit-side bid/ask",
    "decomposition": "actual = raw_signal + horizon_shift - signed_pre_entry_mid_move + bid_ask_cost - one_extra_adverse_tick, using identical future quote for decomposition terms",
    "excursion": "directional maximum favorable/adverse exit-side quote movement from actual fill during first 10s; diagnostic only",
    "no_pnl": True, "no_optimization": True,
}


def _hash_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


STUDY_SHA256 = _hash_obj(STUDY_SPEC)
CHECKPOINT_VERSION = "flow-burst-entry-audit-date-v1"


class BurstAuditError(RuntimeError):
    pass


def _rising_and_falling(pressure: np.ndarray, sessions: np.ndarray, sign: int,
                        threshold: float) -> tuple[np.ndarray, np.ndarray]:
    above = (sign * pressure >= threshold) & (sessions >= 0)
    previous = np.zeros(len(above), dtype=bool)
    previous[1:] = above[:-1] & (sessions[1:] == sessions[:-1])
    rises = np.flatnonzero(above & ~previous)
    falls = np.flatnonzero(~above & previous)
    return rises, falls


def _duration_bursts(ts: np.ndarray, sessions: np.ndarray, rises: np.ndarray,
                     falls: np.ndarray, reset_ns: int) -> np.ndarray:
    if not len(rises):
        return np.empty(0, dtype=np.int64)
    preceding = np.searchsorted(falls, rises, side="left") - 1
    has_fall = preceding >= 0
    safe = np.maximum(preceding, 0)
    same_session = has_fall & (sessions[falls[safe]] == sessions[rises]) if len(falls) else np.zeros(len(rises), bool)
    start = np.zeros(len(rises), dtype=np.int64)
    if len(falls):
        start[same_session] = ts[falls[safe[same_session]]]
    fresh = ~same_session | ((ts[rises] - start) >= reset_ns)
    return rises[fresh]


def _sign_bursts(pressure: np.ndarray, sessions: np.ndarray, rises: np.ndarray,
                 sign: int) -> np.ndarray:
    if not len(rises):
        return np.empty(0, dtype=np.int64)
    reset_indices = np.flatnonzero(sign * pressure <= 0)
    before = np.searchsorted(reset_indices, rises, side="left") - 1
    latest = np.full(len(rises), -1, dtype=np.int64)
    good = before >= 0
    latest[good] = reset_indices[before[good]]
    selected = []
    last_start = -1; last_session = -1
    for ix, reset, session in zip(rises, latest, sessions[rises]):
        if int(session) != last_session or int(reset) > last_start:
            selected.append(int(ix)); last_start = int(ix); last_session = int(session)
    return np.asarray(selected, dtype=np.int64)


def burst_indices(ts: np.ndarray, pressure: np.ndarray, sessions: np.ndarray,
                  thresholds: Mapping[str, float], *, q98: bool = False) -> dict[str, np.ndarray]:
    """Causal first crossings; duration resets require observed below-q state."""
    if not (len(ts) == len(pressure) == len(sessions)):
        raise BurstAuditError("state arrays have different lengths")
    side_outputs: dict[str, list[np.ndarray]] = {v: [] for v in ("E",) if q98} if q98 else {v: [] for v in "ABCD"}
    for side, sign in (("LONG", 1), ("SHORT", -1)):
        rises, falls = _rising_and_falling(pressure, sessions, sign, thresholds[side])
        if q98:
            side_outputs["E"].append(_duration_bursts(ts, sessions, rises, falls, 2_000_000_000))
        else:
            for variant, ns in (("A", 500_000_000), ("B", 1_000_000_000), ("C", 2_000_000_000)):
                side_outputs[variant].append(_duration_bursts(ts, sessions, rises, falls, ns))
            side_outputs["D"].append(_sign_bursts(pressure, sessions, rises, sign))
    return {v: np.sort(np.concatenate(parts)).astype(np.int64) for v, parts in side_outputs.items()}


def time_since_previous(ts: np.ndarray, sessions: np.ndarray, pressure: np.ndarray,
                        burst_rows: np.ndarray) -> dict[int, float | None]:
    previous: dict[tuple[int, int], int] = {}; result = {}
    for ix in burst_rows:
        key = (int(sessions[ix]), 1 if pressure[ix] > 0 else -1)
        prior = previous.get(key)
        result[int(ix)] = (int(ts[ix]) - prior) / 1e9 if prior is not None else None
        previous[key] = int(ts[ix])
    return result


def ordinal_assignments(old_indices: np.ndarray, burst_rows: np.ndarray,
                        pressure: np.ndarray, sessions: np.ndarray) -> dict[int, int]:
    """Map old 2s-refractory events to ordinal within same-direction burst."""
    result: dict[int, int] = {}
    for session in (0, 1, 2):
        for sign in (1, -1):
            old = old_indices[(sessions[old_indices] == session) & (sign * pressure[old_indices] > 0)]
            starts = burst_rows[(sessions[burst_rows] == session) & (sign * pressure[burst_rows] > 0)]
            if not len(old):
                continue
            if not len(starts):
                raise BurstAuditError("old threshold event has no matching burst start")
            groups = np.searchsorted(starts, old, side="right") - 1
            if np.any(groups < 0):
                raise BurstAuditError("old event precedes first burst crossing")
            last_group = -1; ordinal = 0
            for ix, group in zip(old, groups):
                ordinal = ordinal + 1 if int(group) == last_group else 1
                result[int(ix)] = ordinal
                last_group = int(group)
    if len(result) != len(old_indices):
        raise BurstAuditError("ordinal mapping did not cover the old event stream")
    return result


def _bucket_since(seconds: float | None) -> str:
    if seconds is None: return "NO_PREVIOUS_BURST"
    if seconds < 2: return "LT_2S"
    if seconds < 5: return "2_TO_5S"
    if seconds < 10: return "5_TO_10S"
    if seconds < 30: return "10_TO_30S"
    return "GT_30S"


def _checkpoint_path(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, *, day: str, source_sha: str, tape_sha: str,
                     history_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt") as stream:
            row = json.load(stream)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    required = {"status": "DATE_COMPLETE", "version": CHECKPOINT_VERSION,
                "date": day, "source_sha256": source_sha, "tape_sha256": tape_sha,
                "history_sha256": history_sha, "study_sha256": STUDY_SHA256}
    return row if all(row.get(k) == v for k, v in required.items()) else None


def _feature_rows(day: str, rows: np.ndarray, tape: np.ndarray, pressure: np.ndarray,
                  sessions: np.ndarray, variants: Mapping[str, np.ndarray],
                  old: np.ndarray) -> list[dict[str, Any]]:
    """Evaluate the fixed quote paths for the union of old and fresh events."""
    unique = np.unique(np.concatenate([old, *variants.values()]))
    if not len(unique):
        return []
    ts = np.asarray(rows["ts"], dtype=np.int64)
    event_t = ts[unique]
    event_mid = np.asarray(rows["mid"], dtype=np.float64)[unique]
    sign = np.where(pressure[unique] > 0, 1, -1)
    session = sessions[unique]
    tt, ss, bid, ask = flow._quote_arrays(tape)
    if not len(tt):
        raise BurstAuditError(f"empty canonical quote tape: {day}")
    mid = (bid + ask) / 2
    entry_ix = np.searchsorted(tt, event_t + 2_000_000, side="left")
    safe_entry = np.minimum(entry_ix, len(tt)-1)
    valid_entry = ((entry_ix < len(tt)) & (ss[safe_entry] == session)
                   & np.isfinite(bid[safe_entry]) & np.isfinite(ask[safe_entry])
                   & (ask[safe_entry] > bid[safe_entry]))
    entry_time = tt[safe_entry]
    entry_mid = mid[safe_entry]
    entry_quote = np.where(sign > 0, ask[safe_entry], bid[safe_entry])
    fill = entry_quote + sign * flow.TICK
    paths: dict[int, dict[str, np.ndarray]] = {}
    for h in HORIZONS_MS:
        raw_ix = np.searchsorted(tt, event_t + h * 1_000_000, side="left")
        exit_ix = np.searchsorted(tt, entry_time + h * 1_000_000, side="left")
        raw_safe = np.minimum(raw_ix, len(tt)-1)
        exit_safe = np.minimum(exit_ix, len(tt)-1)
        raw_ok = ((raw_ix < len(tt)) & (ss[raw_safe] == session)
                  & np.isfinite(mid[raw_safe]))
        exit_ok = ((exit_ix < len(tt)) & (ss[exit_safe] == session)
                   & np.isfinite(bid[exit_safe]) & np.isfinite(ask[exit_safe])
                   & (ask[exit_safe] > bid[exit_safe]))
        raw = sign * (mid[raw_safe] - event_mid) / flow.TICK
        exit_quote = np.where(sign > 0, bid[exit_safe], ask[exit_safe])
        quote = sign * (exit_quote - entry_quote) / flow.TICK
        actual = sign * (exit_quote - fill) / flow.TICK
        # This horizon-shift term makes the decomposition an exact identity even
        # when t+h and entry_time+h refer to different quote observations.
        shift = sign * (mid[exit_safe] - mid[raw_safe]) / flow.TICK
        pre_move = sign * (entry_mid - event_mid) / flow.TICK
        spread = sign * ((exit_quote-mid[exit_safe]) + (entry_mid-entry_quote)) / flow.TICK
        valid = valid_entry & raw_ok & exit_ok
        if np.any(valid & (np.abs(actual-(raw+shift-pre_move+spread-1.0)) > 1e-8)):
            raise BurstAuditError(f"execution decomposition does not balance: {day}/{h}")
        paths[h] = {"valid": valid, "raw": raw, "quote": quote, "actual": actual,
                    "horizon_shift": shift, "price_move_during_2ms": pre_move,
                    "bid_ask_execution_cost": spread}
    memberships = {v: set(map(int, ix)) for v, ix in variants.items()}
    old_set = set(map(int, old))
    ordinals = {v: ordinal_assignments(old, ix, pressure, sessions)
                for v, ix in variants.items() if v != "E"}
    since = {v: time_since_previous(ts, sessions, pressure, ix)
             for v, ix in variants.items()}
    output = []
    for j, ix in enumerate(unique):
        member = [v for v, ids in memberships.items() if int(ix) in ids]
        result = {"date": day, "row_index": int(ix), "timestamp_ns": int(event_t[j]),
                  "session": int(session[j]), "direction": "LONG" if sign[j] > 0 else "SHORT",
                  "pressure": float(pressure[ix]), "event_mid": float(event_mid[j]),
                  "variants": member, "old_clustered": int(ix) in old_set,
                  "ordinal_by_variant": {v: n[int(ix)] for v, n in ordinals.items() if int(ix) in n},
                  "time_since_previous_same_direction_burst_seconds":
                  {v: since[v][int(ix)] for v in member},
                  "executable_entry_available": bool(valid_entry[j]),
                  "entry_time_ns": int(entry_time[j]) if valid_entry[j] else None,
                  "entry_delay_actual_ms": float((entry_time[j]-event_t[j])/1e6) if valid_entry[j] else None,
                  "paths": {}, "mfe_10s_ticks": None, "mae_10s_ticks": None}
        for h, arrays in paths.items():
            if bool(arrays["valid"][j]):
                result["paths"][str(h)] = {key: float(arrays[key][j]) for key in
                    ("raw", "quote", "actual", "horizon_shift", "price_move_during_2ms",
                     "bid_ask_execution_cost")}
            else:
                result["paths"][str(h)] = None
        if valid_entry[j] and result["paths"]["10000"] is not None:
            end_ix = int(np.searchsorted(tt, entry_time[j]+10_000_000_000, side="left"))
            quote_segment = bid[safe_entry[j]:end_ix+1] if sign[j] > 0 else ask[safe_entry[j]:end_ix+1]
            quote_segment = quote_segment[np.isfinite(quote_segment)]
            if len(quote_segment):
                excursions = sign[j]*(quote_segment-fill[j])/flow.TICK
                result["mfe_10s_ticks"] = float(np.max(excursions))
                result["mae_10s_ticks"] = float(np.min(excursions))
        output.append(result)
    return output


def evaluate_date(day: str, rows: np.ndarray, tape: np.ndarray,
                  history: Mapping[str, Sequence[float]],
                  prior_payload: Mapping[str, Any]) -> dict[str, Any]:
    pressure = native.rolling_pressure(rows, 500_000_000)
    ts = np.ascontiguousarray(rows["ts"], dtype=np.int64)
    sessions = flow._session_codes(day, ts)
    q90 = flow.prior_thresholds(history)
    if any(abs(q90[k]-prior_payload["thresholds"][k]) > 1e-12 for k in q90):
        raise BurstAuditError(f"prior-date q90 threshold parity failure: {day}")
    q98 = {side: float(np.quantile(np.asarray(history[side], dtype=np.float64), .98))
           for side in ("LONG", "SHORT")}
    raw, old = flow.cluster_flow(ts, pressure, sessions, q90)
    old_attrition = prior_payload["attrition"]
    if len(raw) != old_attrition["raw_pressure_observations"] or len(old) != old_attrition["clustered_events"]:
        raise BurstAuditError(f"old stream count parity failure: {day}")
    variants = burst_indices(ts, pressure, sessions, q90)
    variants.update(burst_indices(ts, pressure, sessions, q98, q98=True))
    for name, ids in variants.items():
        if len(ids) != len(np.unique(ids)) or np.any(sessions[ids] < 0):
            raise BurstAuditError(f"invalid burst segmentation: {day}/{name}")
    return {"date": day, "q90": q90, "q98": q98,
            "old_raw_threshold_hits": len(raw), "old_clustered_events": len(old),
            "old_actual_entries": old_attrition["actual_entries"],
            "burst_counts": {v: len(ix) for v, ix in variants.items()},
            "features": _feature_rows(day, rows, tape, pressure, sessions, variants, old)}


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _path_summary(records: Sequence[Mapping[str, Any]], path: str) -> dict[str, Any]:
    return {str(h): flow._markout_values([r["paths"][str(h)][path] for r in records
                   if r["paths"][str(h)] is not None]) for h in HORIZONS_MS}


def _classification_by_split(spring: float | None, october: float | None) -> str:
    if spring is None or october is None:
        return "INSUFFICIENT_EVENTS"
    if spring > 0 and october > 0:
        return "CONSISTENT"
    if spring < 0 and october < 0:
        return "NEGATIVE_BOTH"
    if spring > 0 and october <= 0:
        return "SPRING_ONLY"
    if spring <= 0 and october > 0:
        return "OCTOBER_ONLY"
    return "OPPOSITE"


def _aggregate(payloads: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    # Date-scoped checkpoints are read and emitted one at a time. Retain only
    # references needed for aggregate quantiles, not raw/native source rows.
    by_variant: dict[str, list[Mapping[str, Any]]] = {v: [] for v in VARIANTS}
    old: list[Mapping[str, Any]] = []
    daily: list[dict[str, Any]] = []
    count_audit = {"old_raw_threshold_hits": 0, "old_clustered_2s_events": 0,
                   "old_actual_entries": 0, "bursts": {v: 0 for v in VARIANTS}}
    for p in payloads:
        day = p["date"]
        for k, source in (("old_raw_threshold_hits", "old_raw_threshold_hits"),
                          ("old_clustered_2s_events", "old_clustered_events"),
                          ("old_actual_entries", "old_actual_entries")):
            count_audit[k] += int(p[source])
        counts = {v: int(p["burst_counts"][v]) for v in VARIANTS}
        for v in VARIANTS:
            count_audit["bursts"][v] += counts[v]
        daily.append({"date": day, "period": "SPRING_2025" if day in flow.SPRING_DATES else "OCTOBER_2025",
                      "week": flow._week(day), "old_raw_threshold_hits": p["old_raw_threshold_hits"],
                      "old_clustered_events": p["old_clustered_events"], "bursts": counts})
        for r in p["features"]:
            for v in r["variants"]:
                by_variant[v].append(r)
            if r["old_clustered"]:
                old.append(r)
    for v in VARIANTS:
        if len(by_variant[v]) != count_audit["bursts"][v]:
            raise BurstAuditError(f"variant membership/count mismatch: {v}")
    if len(old) != count_audit["old_clustered_2s_events"]:
        raise BurstAuditError("old event membership/count mismatch")
    variant_results = {}
    period_results = {}
    direction_results = {}
    markouts = {}
    mfe_mae = {}
    daily_results = {}
    weekly_results = {}
    time_since = {}
    decomposition = {}
    for v, records in by_variant.items():
        markouts[v] = {p: _path_summary(records, p) for p in ("raw", "quote", "actual")}
        mfe_mae[v] = {k: flow._markout_values([r[k] for r in records if r[k] is not None])
                      for k in ("mfe_10s_ticks", "mae_10s_ticks")}
        periods = {period: [r for r in records if (r["date"] in flow.SPRING_DATES) == (period == "SPRING_2025")]
                   for period in ("SPRING_2025", "OCTOBER_2025")}
        directions = {side: [r for r in records if r["direction"] == side] for side in ("LONG", "SHORT")}
        period_results[v] = {period: {"event_count": len(group), "actual": _path_summary(group, "actual"),
                                      "raw": _path_summary(group, "raw")}
                             for period, group in periods.items()}
        direction_results[v] = {side: {"event_count": len(group), "actual": _path_summary(group, "actual")}
                                for side, group in directions.items()}
        s = period_results[v]["SPRING_2025"]["actual"]["500"]["mean_ticks"]
        o = period_results[v]["OCTOBER_2025"]["actual"]["500"]["mean_ticks"]
        l = direction_results[v]["LONG"]["actual"]["500"]["mean_ticks"]
        sh = direction_results[v]["SHORT"]["actual"]["500"]["mean_ticks"]
        variant_results[v] = {"definition": VARIANTS[v], "events": len(records),
                              "events_per_day": len(records)/len(flow.TARGET_DATES),
                              "events_per_week": len(records)/len({flow._week(day) for day in flow.TARGET_DATES}),
                              "period_classification": _classification_by_split(s, o),
                              "direction_classification": _classification_by_split(l, sh).replace("SPRING_ONLY", "LONG_DOMINANT").replace("OCTOBER_ONLY", "SHORT_DOMINANT").replace("CONSISTENT", "SYMMETRIC"),
                              "markouts": markouts[v], "periods": period_results[v],
                              "directions": direction_results[v], "mfe_mae": mfe_mae[v]}
        daily_results[v] = []
        weekly_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in records:
            by_day[r["date"]].append(r)
            weekly_groups[flow._week(r["date"])].append(r)
        for day in flow.TARGET_DATES:
            group = by_day[day]
            daily_results[v].append({"date": day, "event_count": len(group),
                 "actual_mean_ticks": {str(h): _mean([r["paths"][str(h)]["actual"] for r in group
                     if r["paths"][str(h)] is not None]) for h in BREAKDOWN_MS}})
        weekly_results[v] = [{"week": week, "event_count": len(group),
                              "mean_500ms_ticks": _mean([r["paths"]["500"]["actual"] for r in group
                                  if r["paths"]["500"] is not None])}
                             for week, group in sorted(weekly_groups.items())]
        variant_results[v]["daily_500ms"] = {"positive_dates": sum(x["actual_mean_ticks"]["500"] is not None and x["actual_mean_ticks"]["500"] > 0 for x in daily_results[v]),
            "negative_dates": sum(x["actual_mean_ticks"]["500"] is not None and x["actual_mean_ticks"]["500"] < 0 for x in daily_results[v]),
            "median_daily_effect": float(np.median([x["actual_mean_ticks"]["500"] for x in daily_results[v] if x["actual_mean_ticks"]["500"] is not None]))}
        variant_results[v]["weekly_500ms"] = {"positive_weeks": sum(x["mean_500ms_ticks"] is not None and x["mean_500ms_ticks"] > 0 for x in weekly_results[v]),
            "negative_weeks": sum(x["mean_500ms_ticks"] is not None and x["mean_500ms_ticks"] < 0 for x in weekly_results[v])}
        time_since[v] = {bucket: {"event_count": len(group), "actual": {str(h): flow._markout_values(
            [r["paths"][str(h)]["actual"] for r in group if r["paths"][str(h)] is not None])
            for h in BREAKDOWN_MS[1:]}} for bucket, group in
            ((b, [r for r in records if _bucket_since(r["time_since_previous_same_direction_burst_seconds"][v]) == b])
             for b in ("NO_PREVIOUS_BURST", "LT_2S", "2_TO_5S", "5_TO_10S", "10_TO_30S", "GT_30S"))}
        decomposition[v] = {}
        for label, group in {"ALL": records, **periods, **directions}.items():
            decomposition[v][label] = {}
            for h in BREAKDOWN_MS:
                valid = [r["paths"][str(h)] for r in group if r["paths"][str(h)] is not None]
                decomposition[v][label][str(h)] = {"n": len(valid),
                    "raw_signal_edge": _mean([p["raw"] for p in valid]),
                    "executable_quote": _mean([p["quote"] for p in valid]),
                    "actual_fill": _mean([p["actual"] for p in valid]),
                    "price_move_during_2ms": _mean([p["price_move_during_2ms"] for p in valid]),
                    "price_move_edge_impact": _mean([-p["price_move_during_2ms"] for p in valid]),
                    "horizon_shift": _mean([p["horizon_shift"] for p in valid]),
                    "bid_ask_execution_cost": _mean([p["bid_ask_execution_cost"] for p in valid]),
                    "extra_adverse_entry_tick_cost": -1.0 if valid else None,
                    "total_signal_to_fill_deterioration": _mean([p["actual"]-p["raw"] for p in valid])}
    ordinal = {}
    for v in "ABCD":
        ordinal[v] = {}
        for label, criterion in (("EVENT_1", lambda n: n == 1), ("EVENT_2", lambda n: n == 2),
                                 ("EVENT_3", lambda n: n == 3), ("EVENT_4", lambda n: n == 4),
                                 ("EVENT_5_PLUS", lambda n: n >= 5)):
            group = [r for r in old if criterion(r["ordinal_by_variant"][v])]
            ordinal[v][label] = {"event_count": len(group), "actual": _path_summary(group, "actual"),
                "mfe_mae": {key: flow._markout_values([r[key] for r in group if r[key] is not None])
                            for key in ("mfe_10s_ticks", "mae_10s_ticks")},
                "spring": _path_summary([r for r in group if r["date"] in flow.SPRING_DATES], "actual"),
                "october": _path_summary([r for r in group if r["date"] in flow.OCTOBER_DATES], "actual"),
                "long": _path_summary([r for r in group if r["direction"] == "LONG"], "actual"),
                "short": _path_summary([r for r in group if r["direction"] == "SHORT"], "actual")}
    first = ordinal["C"]["EVENT_1"]["actual"]["500"]["mean_ticks"]
    later_values = [r["paths"]["500"]["actual"] for r in old if r["ordinal_by_variant"]["C"] > 1 and r["paths"]["500"] is not None]
    later = _mean(later_values)
    if first is None or later is None:
        ordinal_class = "MIXED"
    elif first > 0 and first-later >= .5:
        ordinal_class = "FIRST_EVENT_STRONGLY_BETTER"
    elif first > 0 and first-later >= .25:
        ordinal_class = "FIRST_EVENT_MODESTLY_BETTER"
    elif later-first >= .25:
        ordinal_class = "LATER_EVENTS_BETTER"
    elif abs(first-later) < .25:
        ordinal_class = "NO_ORDINAL_EFFECT"
    else:
        ordinal_class = "MIXED"
    c = variant_results["C"]
    c_s = c["periods"]["SPRING_2025"]; c_o = c["periods"]["OCTOBER_2025"]
    actual_good = all(x["actual"][h]["mean_ticks"] is not None and x["actual"][h]["mean_ticks"] > 0
                      for x in (c_s, c_o) for h in ("500", "1000"))
    raw_good = all(x["raw"]["500"]["mean_ticks"] is not None and x["raw"]["500"]["mean_ticks"] > 0
                   for x in (c_s, c_o))
    if actual_good and c["direction_classification"] not in ("OPPOSITE", "NEGATIVE_BOTH") and c["daily_500ms"]["positive_dates"] > c["daily_500ms"]["negative_dates"]:
        decision, next_step = "FIRST_BURST_FLOW_EDGE_SUPPORTED", "BUILD_FIRST_BURST_FLOW_STRATEGY_V2_FIXED"
    elif ordinal_class in ("FIRST_EVENT_STRONGLY_BETTER", "FIRST_EVENT_MODESTLY_BETTER") and first > 0:
        decision, next_step = "REPEATED_EVENTS_ARE_THE_PRIMARY_FAILURE", "BUILD_FIRST_BURST_ONLY_FIXED_PROTOTYPE_WITH_NO_OPTIMIZATION"
    elif raw_good and not actual_good:
        decision, next_step = "SIGNAL_EXISTS_BUT_EXECUTION_COST_EXCEEDS_EDGE", "STOP_LIVE_FLOW_STRATEGY_UNDER_CURRENT_EXECUTION_MODEL"
    else:
        decision, next_step = "FLOW_MOMENTUM_NOT_EXECUTABLE", "STOP_FLOW_MOMENTUM_BRANCH"
    count_audit["events_per_session"] = {}
    for v, records in by_variant.items():
        per_session = defaultdict(int)
        for record in records:
            per_session[(record["date"], int(record["session"]))] += 1
        counts = [per_session[(day, session)] for day in flow.TARGET_DATES for session in (0, 1, 2)]
        count_audit["events_per_session"][v] = {"session_count": len(counts),
            "mean": float(np.mean(counts)), "median": float(np.median(counts)),
            "p25": float(np.quantile(counts, .25)), "p75": float(np.quantile(counts, .75))}
    return {"variants": variant_results, "old_event_ordinal_analysis": {"by_variant": ordinal,
             "primary_variant": "C", "first_mean_500ms": first, "later_mean_500ms": later,
             "classification": ordinal_class}, "event_count_audit": count_audit,
            "signal_execution_decomposition": decomposition, "time_since_burst": time_since,
            "spring_october": period_results, "direction_results": direction_results,
            "daily_results": daily_results, "weekly_results": weekly_results,
            "mfe_mae": mfe_mae, "markouts": markouts,
            "primary_decision": decision, "next_step": next_step}


def run(*, data_root: Path = native.DATA_ROOT, output_root: Path = OUT_ROOT,
        flow_root: Path = flow.OUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    paths, source_manifest = native._source_catalog(data_root)
    flow_manifest_path = flow_root / "run-manifest.json"
    if not flow_manifest_path.is_file():
        raise BurstAuditError("completed fixed Flow Momentum V1 manifest is required")
    flow_manifest = json.loads(flow_manifest_path.read_text())
    if (flow_manifest.get("status") != "COMPLETE"
            or flow_manifest.get("config_sha256") != flow.CONFIG_SHA256
            or flow_manifest.get("study_sha256") != flow.STUDY_SHA256):
        raise BurstAuditError("fixed Flow Momentum V1 manifest/version mismatch")
    source_days = (flow.DEPENDENCY_DATES[0], flow.SPRING_DATES[0]) if smoke else flow.SOURCE_DATES
    target_days = (flow.SPRING_DATES[0],) if smoke else flow.TARGET_DATES
    tape_hashes = {}
    for day in target_days:
        path = native._tape_path(day)
        if not path.is_file():
            raise BurstAuditError(f"missing canonical tape: {day}")
        tape_hashes[day] = native._sha(path)
        if flow_manifest["tape_sha256_by_date"].get(day) != tape_hashes[day]:
            raise BurstAuditError(f"canonical tape hash mismatch: {day}")
    for day in source_days:
        if flow_manifest["source_sha256_by_date"].get(day) != source_manifest[day]["sha256"]:
            raise BurstAuditError(f"native source hash mismatch: {day}")
        prior_path = flow._checkpoint_path(flow_root, day)
        if not prior_path.is_file() or native._sha(prior_path) != flow_manifest["checkpoint_sha256_by_date"].get(day):
            raise BurstAuditError(f"fixed V1 checkpoint hash mismatch: {day}")
    output_root.mkdir(parents=True, exist_ok=True)
    native._write_json(output_root / "study-spec.json", STUDY_SPEC)
    coverage = {"status": "PASS", "dataset": "GLBX.MDP3", "schema": "mbp-10",
                "native_es_only": True, "spring_dates": list(flow.SPRING_DATES),
                "october_dates": list(flow.OCTOBER_DATES),
                "dependency_dates": list(flow.DEPENDENCY_DATES),
                "fixed_flow_manifest_sha256": native._sha(flow_manifest_path),
                "source_sha256_by_date": {d: source_manifest[d]["sha256"] for d in source_days},
                "tape_sha256_by_date": tape_hashes}
    native._write_json(output_root / "source-coverage.json", coverage)
    history: dict[str, list[float]] = {"LONG": [], "SHORT": []}
    chain = "NO_PRIOR_SOURCE"
    checkpoints = []
    completed = []; resumed = []
    for position, day in enumerate(source_days, 1):
        source_sha = source_manifest[day]["sha256"]
        tape_sha = tape_hashes.get(day, "DEPENDENCY_ONLY")
        prior_path = flow._checkpoint_path(flow_root, day)
        prior = flow._read_checkpoint(prior_path, day=day, source_sha=source_sha,
                                      tape_sha=tape_sha, prior_chain=chain)
        if prior is None:
            raise BurstAuditError(f"fixed V1 checkpoint chronology mismatch: {day}")
        history_sha = _hash_obj([chain, source_sha, flow.STUDY_SHA256, STUDY_SHA256])
        if day in target_days:
            checkpoint_path = _checkpoint_path(output_root, day)
            cached = _read_checkpoint(checkpoint_path, day=day, source_sha=source_sha,
                                      tape_sha=tape_sha, history_sha=history_sha)
            if cached is not None:
                resumed.append(day)
                print(f"BURST_DATE_RESUME={day}", flush=True)
            else:
                print(f"BURST_DATE_START={position}/{len(source_days)} date={day}", flush=True)
                rows = flow._cached_compact(day, paths[day], source_sha, flow_root)
                tape, _ = native._load_tape(day, native._tape_path(day), source_sha)
                payload = evaluate_date(day, rows, tape, history, prior["payload"])
                del rows, tape
                native._write_checkpoint(checkpoint_path, {"status": "DATE_COMPLETE",
                    "version": CHECKPOINT_VERSION, "date": day, "source_sha256": source_sha,
                    "tape_sha256": tape_sha, "history_sha256": history_sha,
                    "study_sha256": STUDY_SHA256, "payload": payload})
                cached = {"payload": payload}
                completed.append(day)
                print(f"BURST_DATE_COMPLETE={day} old={payload['old_clustered_events']} fresh_C={payload['burst_counts']['C']}", flush=True)
            checkpoints.append(checkpoint_path)
        for side in ("LONG", "SHORT"):
            history[side].extend(prior["payload"]["pressure_history_sample"][side])
        chain = flow._hash_obj([chain, source_sha])
        native._write_json(output_root / "checkpoints" / "progress.json",
                           {"last_date": day, "completed_dates": completed,
                            "resumed_dates": resumed, "study_sha256": STUDY_SHA256,
                            "source_chain_sha256": chain})
    if smoke:
        return {"status": "SMOKE_PASS", "dates": list(target_days),
                "checkpoints": len(checkpoints)}
    if len(checkpoints) != len(flow.TARGET_DATES):
        raise BurstAuditError("incomplete target-date checkpoint set")
    payloads = []
    for path in checkpoints:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payloads.append(json.load(stream)["payload"])
    results = _aggregate(payloads, output_root)
    def iter_features():
        for payload in payloads:
            yield from payload["features"]
    native._write_gzip_jsonl(output_root / "event-features.jsonl.gz", iter_features())
    for letter, filename in (("A", "burst-variant-a.json"), ("B", "burst-variant-b.json"),
                             ("C", "burst-variant-c.json"), ("D", "burst-variant-d.json"),
                             ("E", "burst-variant-e-q98.json")):
        native._write_json(output_root / filename, results["variants"][letter])
    for key, filename in (("old_event_ordinal_analysis", "old-event-ordinal-analysis.json"),
                          ("signal_execution_decomposition", "signal-execution-decomposition.json"),
                          ("time_since_burst", "time-since-burst.json"),
                          ("event_count_audit", "event-count-audit.json"),
                          ("spring_october", "spring-october.json"),
                          ("direction_results", "direction-results.json"),
                          ("daily_results", "daily-results.json"),
                          ("weekly_results", "weekly-results.json"),
                          ("mfe_mae", "mfe-mae.json"), ("markouts", "markouts.json")):
        native._write_json(output_root / filename, results[key])
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "dataset": "SPRING_2025 + OCTOBER_2025",
               "target_date_count": len(flow.TARGET_DATES), "native_es_only": True,
               "old_event_counts": results["event_count_audit"],
               "ordinal_classification": results["old_event_ordinal_analysis"]["classification"],
               "signal_to_execution": results["signal_execution_decomposition"]["C"]["ALL"],
               "variants": {v: {"events": r["events"], "actual_500ms": r["markouts"]["actual"]["500"]["mean_ticks"],
                       "actual_1s": r["markouts"]["actual"]["1000"]["mean_ticks"],
                       "period_classification": r["period_classification"]} for v, r in results["variants"].items()},
               "best_structural_first_burst_variant": "C", "q98_reset_rule": "C",
               "primary_decision": results["primary_decision"], "next_step": results["next_step"],
               "optimization_performed": False, "optuna_performed": False,
               "pnl_optimization_performed": False, "execution_model_changed": False,
               "final_oos_accessed": False, "data_downloaded": False,
               "elapsed_seconds": time.monotonic()-started, "study_sha256": STUDY_SHA256}
    native._write_json(output_root / "summary.json", summary)
    (output_root / "report.md").write_text("\n".join((f"# {RUN_ID}", "",
        f"Decision: {summary['primary_decision']}",
        "C is the predeclared structural reference; E applies q98 only with C's 2s reset.",
        "All paths use native ES MBP-10 canonical quotes and frozen 2ms/+1-tick entry semantics.",
        f"Old clustered events: {results['event_count_audit']['old_clustered_2s_events']}; C bursts: {results['variants']['C']['events']}.",
        f"Next step: {summary['next_step']}", "")), encoding="utf-8")
    artifact_hashes = {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file()}
    native._write_json(output_root / "run-manifest.json", {"status": "COMPLETE", "run_id": RUN_ID,
        "study_sha256": STUDY_SHA256, "source_coverage_sha256": native._sha(output_root / "source-coverage.json"),
        "checkpoint_sha256_by_date": {day: native._sha(path) for day, path in zip(target_days, checkpoints)},
        "artifact_sha256_by_name": artifact_hashes, "no_2026_access": True, "no_download": True,
        "no_optimization": True})
    native._write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files":
        {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name != "artifact-hashes.json"}})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=native.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--flow-root", type=Path, default=flow.OUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(data_root=args.data_root, output_root=args.output_root,
                     flow_root=args.flow_root, smoke=args.smoke)
    except (BurstAuditError, flow.FlowStudyError, native.VacuumStudyError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(f"ES_FLOW_BURST_ENTRY_AUDIT={result['status']}", flush=True)
    if result["status"] == "COMPLETE":
        print(f"PRIMARY_DECISION={result['primary_decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
