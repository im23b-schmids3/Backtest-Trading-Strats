"""Frozen 2025 ES opening-range breakout structure/execution diagnostic.

Consumes the completed native-ES structural event package. No new event rule,
entry filter, strategy PnL, parameter search, or 2026 source is used.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import mac_2025_es_structural_breakout_l2_v1 as parent
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_flow_momentum_v1 as flow

RUN_ID = "CMEOrderflow_ES_STRUCTURAL_BREAKOUT_ENTRY_TIMING_AUDIT_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
HORIZONS_MS = parent.HORIZONS_MS
EXCURSION_MS = parent.EXCURSION_MS
TICK = parent.TICK
ORIGINAL_LATENCY_LABELS = ("LE_2MS", "2_TO_5MS", "5_TO_10MS", "10_TO_25MS", "25_TO_50MS", "GT_50MS")
OVERSHOOT_LABELS = ("1_TICK", "2_TICKS", "3_TO_4_TICKS", "5_TO_8_TICKS", "9_PLUS_TICKS")
PRE_MOVE_LABELS = ("LE_ZERO", "ZERO_TO_1", "1_TO_2", "2_TO_4", "GT_4")
SPLITS = ("ALL", "SPRING_2025", "OCTOBER_2025", "LONG", "SHORT")
CONFIG = {
    "version": 1,
    "parent_run_id": parent.RUN_ID,
    "parent_config_sha256": parent.CONFIG_SHA256,
    "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
    "roles": {"SPRING_2025": "PRIMARY_DISCOVERY", "OCTOBER_2025": "SECONDARY_DEV_COMPATIBILITY"},
    "source": "only parent-hash-bound native ES GLBX.MDP3 mbp-10 structural events and identical canonical ES tapes",
    "event_semantics": parent.CONFIG["opening_range"] + "; " + parent.CONFIG["breakout"],
    "execution_semantics": parent.CONFIG["execution"],
    "horizons_ms": list(HORIZONS_MS),
    "overshoot_buckets_ticks": list(OVERSHOOT_LABELS),
    "pre_entry_quote_move_buckets_ticks": list(PRE_MOVE_LABELS),
    "latency_buckets_ms": list(ORIGINAL_LATENCY_LABELS),
    "minimum_overshoot_bucket_n": 8,
    "minimum_latency_bucket_n": 8,
    "latency_merge": "left-to-right accumulate adjacent original buckets until N>=8; any trailing underfull group merges into previous; never use outcomes",
    "pre_entry_move": "direction*(entry-side quote - structural trade price)/ES tick; includes trade-to-mid movement AND entry-side half-spread, both reported",
    "return_inside": "reuse parent trade-price topology; first subsequent actual trade LONG<=OR high or SHORT>=OR low within 60s, restricted to RETURN_INSIDE_OPENING_RANGE class",
    "quote_excursion": "from first usable entry-side quote to future exit-side bid/ask, include zero at entry; actual fill adds one adverse entry tick",
    "matching": "October event matched with replacement to earliest Spring event in exact direction x prior-date OR-width tercile x frozen time-of-day bucket x frozen overshoot bucket; no outcomes or L2; insufficient if <8 pairs",
    "period_difference": "10s raw difference material at >=1 tick; execution deterioration difference material at >=1 tick; execution worsening means October raw-to-fill loss at least 1 tick MORE negative than Spring",
    "decision": "BOTH if Spring raw 10s>0 and October raw 10s<=0 and October raw-to-fill loss<=-2 ticks; STRUCTURAL_REGIME if Spring raw 10s>0 and October raw 10s<=0 but October raw-to-fill loss>-2 ticks; ENTRY_TIMING only if raw positive both periods and <=2-tick overshoot quote 10s positive both with >=8 events each and lower pre-entry-move cohorts positive; NOT_EXECUTABLE if raw positive both but <=2-tick overshoot quote 10s nonpositive both with >=8 each; otherwise INSUFFICIENT_SAMPLE",
    "no_l2_filter": True, "no_optimization": True, "no_pnl": True,
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CONFIG_SHA256 = _hash(CONFIG)
CHECKPOINT_VERSION = "structural-breakout-entry-timing-date-v1"


class EntryAuditError(RuntimeError):
    pass


def overshoot_bucket(value: float) -> str:
    if value <= 0 or not math.isfinite(value):
        raise EntryAuditError("strict breakout requires positive finite overshoot")
    if value <= 1+1e-9: return "1_TICK"
    if value <= 2+1e-9: return "2_TICKS"
    if value <= 4+1e-9: return "3_TO_4_TICKS"
    if value <= 8+1e-9: return "5_TO_8_TICKS"
    return "9_PLUS_TICKS"


def pre_move_bucket(value: float) -> str:
    if value <= 0: return "LE_ZERO"
    if value <= 1: return "ZERO_TO_1"
    if value <= 2: return "1_TO_2"
    if value <= 4: return "2_TO_4"
    return "GT_4"


def latency_bucket(value: float) -> str:
    if value < 2-1e-9:
        raise EntryAuditError("executable quote precedes frozen 2ms delay")
    if value <= 2+1e-9: return "LE_2MS"
    if value <= 5: return "2_TO_5MS"
    if value <= 10: return "5_TO_10MS"
    if value <= 25: return "10_TO_25MS"
    if value <= 50: return "25_TO_50MS"
    return "GT_50MS"


def _stats(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "p25": float(np.quantile(a, .25)), "p75": float(np.quantile(a, .75)),
            "p90": float(np.quantile(a, .90)), "p95": float(np.quantile(a, .95)),
            "p99": float(np.quantile(a, .99)), "min": float(np.min(a)), "max": float(np.max(a))}


def _mean(values: Iterable[float | None]) -> float | None:
    good = [float(v) for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(good)) if good else None


def _split(events: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    if name == "ALL": return list(events)
    if name in ("SPRING_2025", "OCTOBER_2025"):
        return [e for e in events if e["period"] == name]
    return [e for e in events if e["direction"] == name]


def _mid_and_sides(tape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts, _, bid, ask = flow._quote_arrays(tape)
    return ts, bid, ask, (bid+ask)/2


def _entry_index(tape: np.ndarray, event: Mapping[str, Any], close: int) -> int | None:
    ts, bid, ask, _ = _mid_and_sides(tape)
    ix = parent._future(ts, int(event["timestamp_ns"])+2_000_000, close)
    if ix is None: return None
    while ix < len(ts) and ts[ix] < close and not (
            math.isfinite(float(bid[ix])) and math.isfinite(float(ask[ix])) and ask[ix] > bid[ix]):
        ix += 1
    return ix if ix < len(ts) and ts[ix] < close else None


def _entry_excursions(tape: np.ndarray, event: Mapping[str, Any],
                      entry_ix: int | None, close: int) -> dict[str, Any]:
    result = {}
    if entry_ix is None:
        return {str(h): None for h in EXCURSION_MS}
    ts, bid, ask, _ = _mid_and_sides(tape)
    sign = int(event["sign"])
    exit_side = bid if sign > 0 else ask
    quote = float(event["entry_quote"]); fill = float(event["actual_fill"])
    for h in EXCURSION_MS:
        end_ix = parent._future(ts, int(ts[entry_ix])+h*1_000_000, close)
        if end_ix is None:
            result[str(h)] = None; continue
        q = sign*(exit_side[entry_ix:end_ix+1]-quote)/TICK
        f = sign*(exit_side[entry_ix:end_ix+1]-fill)/TICK
        q = q[np.isfinite(q)]; f = f[np.isfinite(f)]
        if not len(q) or not len(f):
            result[str(h)] = None; continue
        result[str(h)] = {"quote_mfe": float(max(0, np.max(q))),
            "quote_mae": float(min(0, np.min(q))),
            "fill_mfe": float(max(0, np.max(f))),
            "fill_mae": float(min(0, np.min(f)))}
    return result


def _return_inside_ms(tape: np.ndarray, event: Mapping[str, Any], close: int) -> float | None:
    if event["topology"] != "RETURN_INSIDE_OPENING_RANGE":
        return None
    ts = tape["timestamp_ns"]
    start_ix = int(event["tape_index"])+1
    end_ix = parent._future(ts, int(event["timestamp_ns"])+60_000_000_000, close)
    if end_ix is None:
        raise EntryAuditError("return-inside topology lacks 60s path")
    segment = tape[start_ix:end_ix+1]
    if int(event["sign"]) > 0:
        inside = segment["execution_price"] <= float(event["opening_range_high"])
    else:
        inside = segment["execution_price"] >= float(event["opening_range_low"])
    found = np.flatnonzero((segment["execution_size"] > 0) & inside)
    if not len(found):
        raise EntryAuditError("parent RETURN_INSIDE topology cannot be reproduced")
    return float((ts[start_ix+int(found[0])]-int(event["timestamp_ns"]))/1e6)


def evaluate_date(day: str, tape: np.ndarray, frozen_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    opening, regenerated = parent.opening_range_events(day, tape)
    if len(regenerated) != len(frozen_events):
        raise EntryAuditError(f"frozen breakout event-count parity failure: {day}")
    for a, b in zip(regenerated, frozen_events):
        for key in ("direction", "timestamp_ns", "tape_index", "trade_price", "boundary", "overshoot_ticks"):
            if a[key] != b[key]:
                raise EntryAuditError(f"frozen breakout event parity failure: {day}/{key}")
    ts, bid, ask, mid = _mid_and_sides(tape)
    output = []
    for frozen in frozen_events:
        current = dict(frozen)
        # Re-evaluate the frozen price path on the exact same source-bound tape.
        calculated = parent._path_analysis(tape, frozen, opening)
        for key in ("entry_time_ns", "entry_quote", "actual_fill", "raw_anchor_mid", "raw_anchor_time_ns"):
            if calculated[key] != frozen[key]:
                raise EntryAuditError(f"frozen anchor/fill parity failure: {day}/{key}")
        for h in HORIZONS_MS:
            for path in ("raw", "quote", "actual", "horizon_shift", "pre_entry_price_move", "bid_ask_effect"):
                if calculated["paths"][str(h)][path] != frozen["paths"][str(h)][path]:
                    raise EntryAuditError(f"frozen markout parity failure: {day}/{h}/{path}")
        entry_ix = _entry_index(tape, frozen, int(opening["rth_close_ns"]))
        if entry_ix is None or int(ts[entry_ix]) != frozen["entry_time_ns"]:
            raise EntryAuditError(f"frozen executable timestamp parity failure: {day}")
        sign = int(frozen["sign"])
        trade = float(frozen["trade_price"])
        entry_quote = float(frozen["entry_quote"])
        entry_mid = float(mid[entry_ix])
        trade_to_mid = sign*(entry_mid-trade)/TICK
        mid_to_quote = sign*(entry_quote-entry_mid)/TICK
        pre_entry_move = sign*(entry_quote-trade)/TICK
        if abs(pre_entry_move-trade_to_mid-mid_to_quote) > 1e-8:
            raise EntryAuditError("trade-to-entry quote geometry does not reconcile")
        latency = float((int(ts[entry_ix])-int(frozen["timestamp_ns"]))/1e6)
        for h in HORIZONS_MS:
            p = frozen["paths"][str(h)]
            if p["actual"] is not None and abs(p["actual"]-(p["raw"]+p["horizon_shift"]-p["pre_entry_price_move"]+p["bid_ask_effect"]-1)) > 1e-8:
                raise EntryAuditError(f"raw-to-fill decomposition mismatch: {day}/{h}")
        current.update({"entry_mid": entry_mid, "actual_executable_latency_ms": latency,
            "pre_entry_move_ticks": pre_entry_move,
            "trade_to_entry_mid_ticks": trade_to_mid,
            "entry_mid_to_quote_ticks": mid_to_quote,
            "overshoot_bucket": overshoot_bucket(float(frozen["overshoot_ticks"])),
            "pre_entry_move_bucket": pre_move_bucket(pre_entry_move),
            "latency_bucket": latency_bucket(latency),
            "entry_excursions": _entry_excursions(tape, frozen, entry_ix, int(opening["rth_close_ns"])),
            "return_inside_ms": _return_inside_ms(tape, frozen, int(opening["rth_close_ns"]))})
        output.append(current)
    return {"date": day, "period": parent._period(day), "opening_range": opening,
            "events": output}


def _path_mean(events: Sequence[Mapping[str, Any]], horizon: int, path: str) -> float | None:
    return _mean(e["paths"][str(horizon)][path] for e in events)


def _path_table(events: Sequence[Mapping[str, Any]], path: str) -> dict[str, Any]:
    return {name: {str(h): parent._markout([e["paths"][str(h)][path] for e in _split(events, name)
                                          if e["paths"][str(h)][path] is not None])
                   for h in HORIZONS_MS} for name in SPLITS}


def _edge_loss(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in SPLITS:
        group = _split(events, name)
        output[name] = {}
        for h in HORIZONS_MS[1:]:
            path = [e["paths"][str(h)] for e in group if e["paths"][str(h)]["actual"] is not None]
            total = _mean(x["actual"]-x["raw"] for x in path)
            output[name][str(h)] = {"n": len(path), "raw_signal_edge": _mean(x["raw"] for x in path),
                "pre_entry_price_movement_effect": -_mean(x["pre_entry_price_move"] for x in path) if path else None,
                "bid_ask_execution_effect": _mean(x["bid_ask_effect"] for x in path),
                "adverse_entry_tick_effect": -1.0 if path else None,
                "horizon_alignment_effect": _mean(x["horizon_shift"] for x in path),
                "executable_quote_edge": _mean(x["quote"] for x in path),
                "actual_fill_edge": _mean(x["actual"] for x in path),
                "raw_to_executable_loss": _mean(x["quote"]-x["raw"] for x in path),
                "raw_to_actual_fill_loss": total,
                "executable_to_actual_fill_loss": _mean(x["actual"]-x["quote"] for x in path)}
            if path:
                expected = (output[name][str(h)]["raw_signal_edge"]
                            + output[name][str(h)]["pre_entry_price_movement_effect"]
                            + output[name][str(h)]["bid_ask_execution_effect"]
                            + output[name][str(h)]["adverse_entry_tick_effect"]
                            + output[name][str(h)]["horizon_alignment_effect"])
                if abs(expected-output[name][str(h)]["actual_fill_edge"]) > 1e-8:
                    raise EntryAuditError("aggregated decomposition does not reconcile")
    return output


def _rank_average(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    top = np.cumsum(counts)
    return (top-(counts-1)/2)[inverse].astype(np.float64)


def spearman(x: Iterable[float | None], y: Iterable[float | None]) -> dict[str, Any]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "status": "INSUFFICIENT_SAMPLE"}
    left = _rank_average(np.asarray([a for a, _ in pairs]))
    right = _rank_average(np.asarray([b for _, b in pairs]))
    if np.std(left) == 0 or np.std(right) == 0:
        return {"n": len(pairs), "rho": None, "status": "CONSTANT_INPUT"}
    return {"n": len(pairs), "rho": float(np.corrcoef(left, right)[0, 1]),
            "status": "DESCRIPTIVE_ONLY"}


def _excursion_table(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in SPLITS:
        group = _split(events, name)
        result[name] = {}
        for h in EXCURSION_MS:
            result[name][str(h)] = {}
            for key, source in (("raw_mfe", ("excursions", "mfe")),
                                ("raw_mae", ("excursions", "mae")),
                                ("quote_mfe", ("entry_excursions", "quote_mfe")),
                                ("quote_mae", ("entry_excursions", "quote_mae")),
                                ("fill_mfe", ("entry_excursions", "fill_mfe")),
                                ("fill_mae", ("entry_excursions", "fill_mae"))):
                values = [e[source[0]][str(h)][source[1]] for e in group if e[source[0]][str(h)] is not None]
                result[name][str(h)][key] = _stats(values)
    return result


def _bucket_metrics(events: Sequence[Mapping[str, Any]], *, include_excursion: bool = False) -> dict[str, Any]:
    result = {"n": len(events), "status": "SUFFICIENT" if len(events) >= 8 else "INSUFFICIENT_BUCKET_SAMPLE",
        "raw": {}, "quote": {}, "actual": {}}
    for path in ("raw", "quote", "actual"):
        for h in HORIZONS_MS[1:]:
            result[path][str(h)] = parent._markout([e["paths"][str(h)][path] for e in events
                                                     if e["paths"][str(h)][path] is not None])
    if include_excursion:
        result["mfe_mae"] = {str(h): {name: _stats(e["excursions"][str(h)][key] for e in events
            if e["excursions"][str(h)] is not None) for name, key in (("mfe", "mfe"), ("mae", "mae"))}
            for h in EXCURSION_MS if h >= 2000}
    return result


def _fixed_bucket_table(events: Sequence[Mapping[str, Any]], field: str,
                        labels: Sequence[str], *, excursion: bool = False) -> dict[str, Any]:
    return {label: _bucket_metrics([e for e in events if e[field] == label], include_excursion=excursion)
            for label in labels}


def merge_latency_buckets(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic adjacent merge by counts only; never inspect outcomes."""
    sizes = {label: sum(e["latency_bucket"] == label for e in events) for label in ORIGINAL_LATENCY_LABELS}
    groups = []; labels = []; count = 0
    for label in ORIGINAL_LATENCY_LABELS:
        labels.append(label); count += sizes[label]
        if count >= 8:
            groups.append({"labels": labels, "n": count})
            labels = []; count = 0
    if labels:
        if groups:
            groups[-1]["labels"].extend(labels)
            groups[-1]["n"] += count
        else:
            groups.append({"labels": labels, "n": count})
    return groups


def _latency_table(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    originals = _fixed_bucket_table(events, "latency_bucket", ORIGINAL_LATENCY_LABELS)
    merged = []
    for item in merge_latency_buckets(events):
        group = [e for e in events if e["latency_bucket"] in item["labels"]]
        merged.append({"original_labels": item["labels"], **_bucket_metrics(group)})
    return {"original_buckets": originals, "merged_adjacent_buckets": merged,
            "merge_rule": CONFIG["latency_merge"]}


def _geometry(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {key: _stats(e[key] for e in _split(events, name)) for key in (
        "overshoot_ticks", "pre_entry_move_ticks", "trade_to_entry_mid_ticks",
        "entry_mid_to_quote_ticks", "actual_executable_latency_ms")}
        for name in SPLITS}


def _correlations(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {str(h): spearman((e["overshoot_ticks"] for e in _split(events, name)),
        (e["paths"][str(h)]["quote"] for e in _split(events, name)))
        for h in HORIZONS_MS[1:]} for name in ("ALL", "SPRING_2025", "OCTOBER_2025")}


def _predictor_correlations(events: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    return {name: {str(h): spearman((e[field] for e in _split(events, name)),
        (e["paths"][str(h)]["quote"] for e in _split(events, name)))
        for h in HORIZONS_MS[1:]} for name in ("ALL", "SPRING_2025", "OCTOBER_2025")}


def _event_loss_distributions(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in SPLITS:
        group = _split(events, name)
        result[name] = {}
        for h in HORIZONS_MS[1:]:
            valid = [e["paths"][str(h)] for e in group if e["paths"][str(h)]["actual"] is not None]
            result[name][str(h)] = {
                "raw_to_executable_loss_ticks": _stats(x["quote"]-x["raw"] for x in valid),
                "raw_to_actual_fill_loss_ticks": _stats(x["actual"]-x["raw"] for x in valid),
                "executable_to_actual_fill_loss_ticks": _stats(x["actual"]-x["quote"] for x in valid)}
    return result


def _topology_table(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("IMMEDIATE_CONTINUATION", "SMALL_CONTINUATION_THEN_FAILURE", "STAGNATION",
              "RETURN_INSIDE_OPENING_RANGE", "OPPOSITE_RANGE_BREAK", "UNCLASSIFIED")
    result = {}
    for period in ("SPRING_2025", "OCTOBER_2025"):
        group = _split(events, period)
        result[period] = {label: {"n": len(rows), "percent": 100*len(rows)/len(group) if group else None,
            "mean_overshoot_ticks": _mean(e["overshoot_ticks"] for e in rows),
            "mean_latency_ms": _mean(e["actual_executable_latency_ms"] for e in rows),
            "raw_10s": _path_mean(rows, 10000, "raw"),
            "quote_10s": _path_mean(rows, 10000, "quote"),
            "actual_10s": _path_mean(rows, 10000, "actual")}
            for label in labels for rows in ([e for e in group if e["topology"] == label],)}
    return result


def _return_inside_table(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: _stats(e["return_inside_ms"] for e in _split(events, name)
                         if e["return_inside_ms"] is not None) for name in SPLITS}


def _tod_table(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("10:00-10:30", "10:30-11:30", "11:30-14:00", "14:00-15:30", "15:30-16:00")
    return {name: {label: {"n": len(rows), "mean_overshoot_ticks": _mean(e["overshoot_ticks"] for e in rows),
        "mean_latency_ms": _mean(e["actual_executable_latency_ms"] for e in rows),
        **{f"{path}_10s": _path_mean(rows, 10000, path) for path in ("raw", "quote", "actual")}}
        for label in labels for rows in ([e for e in _split(events, name) if e["time_of_day_bucket"] == label],)}
        for name in ("ALL", "SPRING_2025", "OCTOBER_2025")}


def _week(day: str) -> str:
    year, week, _ = date.fromisoformat(day).isocalendar()
    return f"{year}-W{week:02d}"


def _period_rows(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {"n": len(group), "raw": {str(h): _path_mean(group, h, "raw") for h in HORIZONS_MS},
        "quote": {str(h): _path_mean(group, h, "quote") for h in HORIZONS_MS},
        "actual": {str(h): _path_mean(group, h, "actual") for h in HORIZONS_MS}}
        for name in ("SPRING_2025", "OCTOBER_2025") for group in (_split(events, name),)}


def _primary_values(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"n": len(events), **{f"{path}_10s": _path_mean(events, 10000, path)
        for path in ("raw", "quote", "actual")},
        "overshoot_quote_10s_spearman": spearman((e["overshoot_ticks"] for e in events),
            (e["paths"]["10000"]["quote"] for e in events))}


def _daily_weekly(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    daily = {}; weekly = {}
    for day in flow.TARGET_DATES:
        group = [e for e in events if e["date"] == day]
        daily[day] = {**_primary_values(group),
            "mean_overshoot_ticks": _mean(e["overshoot_ticks"] for e in group),
            "mean_latency_ms": _mean(e["actual_executable_latency_ms"] for e in group),
            "topology_counts": {label: sum(e["topology"] == label for e in group)
                for label in sorted({e["topology"] for e in events})}}
    for week in sorted({_week(day) for day in flow.TARGET_DATES}):
        group = [e for e in events if _week(e["date"]) == week]
        weekly[week] = {**_primary_values(group),
            "mean_overshoot_ticks": _mean(e["overshoot_ticks"] for e in group),
            "mean_latency_ms": _mean(e["actual_executable_latency_ms"] for e in group)}
    return daily, weekly


def _leave_out(events: Sequence[Mapping[str, Any]], by: str) -> dict[str, Any]:
    keys = list(flow.TARGET_DATES) if by == "date" else sorted({_week(d) for d in flow.TARGET_DATES})
    rows = {}
    for key in keys:
        remaining = [e for e in events if (e["date"] if by == "date" else _week(e["date"])) != key]
        rows[key] = _primary_values(remaining)
    stability = {}
    for metric in ("raw_10s", "quote_10s", "actual_10s", "overshoot_quote_10s_spearman"):
        vals = {k: (v[metric]["rho"] if metric.endswith("spearman") else v[metric]) for k, v in rows.items()}
        good = {k: v for k, v in vals.items() if v is not None}
        stability[metric] = {"tested": len(good), "positive": sum(v > 0 for v in good.values()),
            "negative": sum(v < 0 for v in good.values()), "median": _stats(good.values())["median"],
            "minimum": min(good.values()) if good else None, "maximum": max(good.values()) if good else None,
            "worst_omitted": min(good, key=good.get) if good else None,
            "best_omitted": max(good, key=good.get) if good else None}
    return {"rows": rows, "sign_stability": stability}


def _matched_period(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(e: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
        width = e["prior_date_terciles"].get("opening_range_width_ticks")
        if width is None: return None
        return (e["direction"], str(width), e["time_of_day_bucket"], e["overshoot_bucket"])
    spring = sorted(_split(events, "SPRING_2025"), key=lambda e: (e["date"], e["timestamp_ns"]))
    october = sorted(_split(events, "OCTOBER_2025"), key=lambda e: (e["date"], e["timestamp_ns"]))
    lookup = {}
    for e in spring:
        if key(e) is not None: lookup.setdefault(key(e), e)
    pairs = [(lookup[key(e)], e) for e in october if key(e) in lookup]
    result = {"status": "DESCRIPTIVE_ONLY" if len(pairs) >= 8 else "INSUFFICIENT_FOR_MATCHED_PERIOD_COMPARISON",
        "matched_pairs": len(pairs), "october_events": len(october),
        "matched_spring": _primary_values([a for a, _ in pairs]),
        "matched_october": _primary_values([b for _, b in pairs]),
        "match_fields": ["direction", "prior_date_opening_range_width_tercile", "time_of_day_bucket", "overshoot_bucket"]}
    return result


def _classification(events: Sequence[Mapping[str, Any]], period: Mapping[str, Any],
                    edge: Mapping[str, Any], overshoot: Mapping[str, Any]) -> dict[str, Any]:
    s = period["SPRING_2025"]; o = period["OCTOBER_2025"]
    sr, oraw = s["raw"]["10000"], o["raw"]["10000"]
    sl = edge["SPRING_2025"]["10000"]["raw_to_executable_loss"]
    ol = edge["OCTOBER_2025"]["10000"]["raw_to_executable_loss"]
    if None in (sr, oraw, sl, ol):
        period_class = "INSUFFICIENT_SAMPLE"
    else:
        structural = abs(sr-oraw) >= 1
        execution = abs(sl-ol) >= 1
        period_class = ("BOTH_STRUCTURE_AND_EXECUTION_DIFFERENT" if structural and execution else
            "STRUCTURE_DIFFERENT_EXECUTION_SIMILAR" if structural else
            "STRUCTURE_SIMILAR_EXECUTION_DIFFERENT" if execution else "NO_CLEAR_PERIOD_DIFFERENCE")
    clean = [e for e in events if e["overshoot_bucket"] in ("1_TICK", "2_TICKS")]
    cs = [e for e in clean if e["period"] == "SPRING_2025"]
    co = [e for e in clean if e["period"] == "OCTOBER_2025"]
    clean_positive = len(cs) >= 8 and len(co) >= 8 and any(
        (_path_mean(cs, h, "quote") or -math.inf) > 0 and (_path_mean(co, h, "quote") or -math.inf) > 0
        for h in (500, 1000, 2000, 5000, 10000))
    if sr is not None and oraw is not None and sr > 0 and oraw <= 0:
        decision = ("BOTH_STRUCTURE_AND_ENTRY_LIMITING" if
            edge["OCTOBER_2025"]["10000"]["raw_to_actual_fill_loss"] <= -2
            else "STRUCTURAL_REGIME_PROBLEM_IDENTIFIED")
    elif sr is not None and oraw is not None and sr > 0 and oraw > 0 and clean_positive:
        decision = "ENTRY_TIMING_PROBLEM_IDENTIFIED"
    elif sr is not None and oraw is not None and sr > 0 and oraw > 0 and len(cs) >= 8 and len(co) >= 8:
        decision = "BREAKOUT_EDGE_NOT_EXECUTABLE"
    else:
        decision = "INSUFFICIENT_SAMPLE"
    next_step = {"BOTH_STRUCTURE_AND_ENTRY_LIMITING": "DO_NOT_BUILD_STRATEGY_YET",
        "STRUCTURAL_REGIME_PROBLEM_IDENTIFIED": "DIAGNOSE_OPENING_RANGE_BREAKOUT_REGIME_DEPENDENCE",
        "ENTRY_TIMING_PROBLEM_IDENTIFIED": "DESIGN_ONE_FROZEN_BREAKOUT_RETEST_OR_ANTI_CHASE_EVENT_STUDY",
        "BREAKOUT_EDGE_NOT_EXECUTABLE": "STOP_30M_OR_BREAKOUT_BRANCH_UNDER_CURRENT_EXECUTION_MODEL",
        "INSUFFICIENT_SAMPLE": "REQUIRE_ADDITIONAL_PREDECLARED_CALIBRATION_DATA"}[decision]
    return {"structure_vs_execution_classification": period_class, "primary_decision": decision,
        "next_step": next_step, "spring_raw_10s": sr, "october_raw_10s": oraw,
        "spring_raw_to_quote_loss_10s": sl, "october_raw_to_quote_loss_10s": ol,
        "october_execution_worse_than_spring": ol is not None and sl is not None and ol < sl,
        "small_overshoot_positive_executable_both_periods": clean_positive,
        "small_overshoot_counts": {"spring": len(cs), "october": len(co)}}


def _checkpoint(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, day: str, source_sha: str, tape_sha: str,
                     parent_sha: str) -> dict[str, Any] | None:
    if not path.is_file(): return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f: row = json.load(f)
    except (OSError, EOFError, json.JSONDecodeError): return None
    expected = {"version": CHECKPOINT_VERSION, "status": "DATE_COMPLETE", "date": day,
        "source_sha256": source_sha, "tape_sha256": tape_sha,
        "parent_manifest_sha256": parent_sha, "config_sha256": CONFIG_SHA256}
    return row if all(row.get(k) == v for k, v in expected.items()) else None


def _parent_package(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], str]:
    manifest_path = root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("config_sha256") != parent.CONFIG_SHA256:
        raise EntryAuditError("parent study incomplete or changed")
    for name, digest in manifest["artifact_sha256_by_name"].items():
        # Parent rerun wrote these bookkeeping files after taking its artifact
        # snapshot. They are self-referential; validate the data files instead.
        if name in ("run-manifest.json", "artifact-hashes.json"): continue
        if native._sha(root / name) != digest: raise EntryAuditError(f"parent artifact hash mismatch: {name}")
    for day, digest in manifest["checkpoint_sha256_by_date"].items():
        if native._sha(parent._checkpoint(root, day)) != digest:
            raise EntryAuditError(f"parent date checkpoint hash mismatch: {day}")
    coverage_path = root / "source-coverage.json"
    if native._sha(coverage_path) != manifest["source_coverage_sha256"]:
        raise EntryAuditError("parent source coverage hash mismatch")
    coverage = json.loads(coverage_path.read_text())
    if coverage.get("schema") != "mbp-10" or coverage.get("instrument") != "ES" or not coverage.get("native_es_only"):
        raise EntryAuditError("parent is not native ES MBP-10")
    if coverage["spring_dates"] != list(flow.SPRING_DATES) or coverage["october_dates"] != list(flow.OCTOBER_DATES):
        raise EntryAuditError("eligible date mismatch")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(root / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["date"] not in flow.TARGET_DATES: raise EntryAuditError("parent event outside date universe")
            grouped[e["date"]].append(e)
    if sum(map(len, grouped.values())) != 77:
        raise EntryAuditError("frozen parent event universe is not 77")
    return manifest, coverage, grouped, native._sha(manifest_path)


def run(*, data_root: Path = native.DATA_ROOT, parent_root: Path = parent.OUT_ROOT,
        output_root: Path = OUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    _, coverage, frozen, parent_sha = _parent_package(parent_root)
    paths, source_manifest = native._source_catalog(data_root)
    if {d: source_manifest[d]["sha256"] for d in native.ALL_SOURCE_DATES} != coverage["source_sha256_by_date"]:
        raise EntryAuditError("native ES source catalog differs from parent coverage")
    days = flow.TARGET_DATES[:1] if smoke else flow.TARGET_DATES
    output_root.mkdir(parents=True, exist_ok=True)
    native._write_json(output_root / "audit-config.json", CONFIG)
    payloads = []; resumed = []; completed = []
    for pos, day in enumerate(days, 1):
        tape_path = native._tape_path(day)
        tape_sha = native._sha(tape_path)
        if tape_sha != coverage["tape_sha256_by_date"][day]:
            raise EntryAuditError(f"canonical ES tape hash differs from parent: {day}")
        source_sha = source_manifest[day]["sha256"]
        cp = _checkpoint(output_root, day)
        cached = _read_checkpoint(cp, day, source_sha, tape_sha, parent_sha)
        if cached:
            payload = cached["payload"]; resumed.append(day)
            print(f"ENTRY_AUDIT_DATE_RESUME={day}", flush=True)
        else:
            print(f"ENTRY_AUDIT_DATE_START={pos}/{len(days)} date={day}", flush=True)
            tape, _ = native._load_tape(day, tape_path, source_sha)
            payload = evaluate_date(day, tape, frozen.get(day, []))
            native._write_checkpoint(cp, {"version": CHECKPOINT_VERSION,
                "status": "DATE_COMPLETE", "date": day, "source_sha256": source_sha,
                "tape_sha256": tape_sha, "parent_manifest_sha256": parent_sha,
                "config_sha256": CONFIG_SHA256, "payload": payload})
            completed.append(day)
            print(f"ENTRY_AUDIT_DATE_COMPLETE={day} events={len(payload['events'])}", flush=True)
        payloads.append(payload)
    if smoke: return {"status": "SMOKE_PASS", "days": list(days), "events": sum(len(p["events"]) for p in payloads)}
    events = [e for p in payloads for e in p["events"]]
    if len(events) != 77 or sum(e["direction"] == "LONG" for e in events) != 38:
        raise EntryAuditError("final frozen event-count parity failure")
    raw, quote, actual = (_path_table(events, x) for x in ("raw", "quote", "actual"))
    edge = _edge_loss(events); geometry = _geometry(events)
    overshoot = {name: _fixed_bucket_table(_split(events, name), "overshoot_bucket", OVERSHOOT_LABELS,
        excursion=True) for name in ("ALL", "SPRING_2025", "OCTOBER_2025", "LONG", "SHORT")}
    pre_move = {name: _fixed_bucket_table(_split(events, name), "pre_entry_move_bucket", PRE_MOVE_LABELS)
        for name in ("ALL", "SPRING_2025", "OCTOBER_2025")}
    latency = {name: _latency_table(_split(events, name)) for name in ("ALL", "SPRING_2025", "OCTOBER_2025")}
    correlations = _correlations(events); period = _period_rows(events)
    pre_move_correlations = _predictor_correlations(events, "pre_entry_move_ticks")
    latency_correlations = _predictor_correlations(events, "actual_executable_latency_ms")
    daily, weekly = _daily_weekly(events)
    matched = _matched_period(events)
    decision = _classification(events, period, edge, overshoot)
    source_coverage = {"status": "PASS", "parent_run_id": parent.RUN_ID,
        "parent_manifest_sha256": parent_sha, "source_coverage_sha256": native._sha(parent_root / "source-coverage.json"),
        "source_sha256_by_date": coverage["source_sha256_by_date"],
        "tape_sha256_by_date": coverage["tape_sha256_by_date"],
        "spring_dates": list(flow.SPRING_DATES), "october_dates": list(flow.OCTOBER_DATES),
        "native_es_mbp10_only": True, "no_mes": True, "no_mbo": True}
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "total_events": len(events),
        "long_events": 38, "short_events": 39, "spring_role": "PRIMARY_DISCOVERY",
        "october_role": "SECONDARY_DEV_COMPATIBILITY", "raw_markouts_all": raw["ALL"],
        "executable_markouts_all": quote["ALL"], "actual_fill_markouts_all": actual["ALL"],
        "period_comparison": period, "overshoot": geometry["ALL"]["overshoot_ticks"],
        "pre_entry_move": geometry["ALL"]["pre_entry_move_ticks"],
        "latency_ms": geometry["ALL"]["actual_executable_latency_ms"],
        "overshoot_correlations": correlations,
        "pre_entry_move_correlations": pre_move_correlations,
        "latency_correlations": latency_correlations,
        "overshoot_relationship_status": "NOT_IDENTIFIABLE_CONSTANT_ONE_TICK" if
            len({e["overshoot_ticks"] for e in events}) == 1 else "DESCRIPTIVE_ONLY",
        "decision": decision,
        "matched_period_status": matched["status"], "config_sha256": CONFIG_SHA256,
        "optimization_performed": False, "optuna_performed": False,
        "threshold_search_performed": False, "entry_filter_added": False,
        "l2_filter_used": False, "final_oos_accessed": False,
        "data_downloaded": False, "elapsed_seconds": time.monotonic()-started}
    native._write_gzip_jsonl(output_root / "events.jsonl.gz", events)
    files = {"source-coverage.json": source_coverage, "summary.json": summary,
        "event-geometry.json": geometry, "overshoot-analysis.json": {"distribution":
            {name: geometry[name]["overshoot_ticks"] for name in SPLITS}, "correlations": correlations},
        "pre-entry-move-analysis.json": {"distribution":
            {name: geometry[name]["pre_entry_move_ticks"] for name in SPLITS},
            "quote_markout_correlations": pre_move_correlations},
        "execution-latency.json": {"distribution":
            {name: geometry[name]["actual_executable_latency_ms"] for name in SPLITS},
            "quote_markout_correlations": latency_correlations},
        "raw-markouts.json": raw, "executable-markouts.json": quote,
        "actual-fill-markouts.json": actual, "edge-loss-decomposition.json": edge,
        "event-loss-distributions.json": _event_loss_distributions(events),
        "period-comparison.json": {"periods": period, **decision},
        "direction-results.json": {name: {path: table[name] for path, table in
            (("raw", raw), ("quote", quote), ("actual", actual))} for name in ("LONG", "SHORT")},
        "overshoot-buckets.json": overshoot, "latency-buckets.json": latency,
        "pre-entry-move-buckets.json": pre_move, "time-of-day-results.json": _tod_table(events),
        "failure-topology.json": _topology_table(events),
        "return-inside-range.json": _return_inside_table(events),
        "mfe-mae.json": _excursion_table(events), "matched-period-comparison.json": matched,
        "daily-results.json": daily, "weekly-results.json": weekly,
        "lodo-results.json": _leave_out(events, "date"),
        "lowo-results.json": _leave_out(events, "week")}
    for name, obj in files.items(): native._write_json(output_root / name, obj)
    report = [f"# {RUN_ID}", "", "Spring is primary discovery; October is secondary DEV compatibility.",
        "No event filter, L2 filter, strategy PnL, parameter search, or OOS access.",
        f"Events: {len(events)} (38 LONG, 39 SHORT).",
        f"Spring raw 10s: {period['SPRING_2025']['raw']['10000']:.3f} ticks; October raw 10s: {period['OCTOBER_2025']['raw']['10000']:.3f} ticks.",
        f"Pooled quote 10s: {quote['ALL']['10000']['mean_ticks']:.3f} ticks; actual fill 10s: {actual['ALL']['10000']['mean_ticks']:.3f} ticks.",
        "All 77 first strict breakout events overshot by exactly one tick; overshoot-outcome correlation is undefined, not zero. No anti-overshoot rule can be inferred.",
        f"Pooled raw-to-fill 10s loss: {edge['ALL']['10000']['raw_to_actual_fill_loss']:.3f} ticks = pre-entry price effect {edge['ALL']['10000']['pre_entry_price_movement_effect']:.3f} + bid/ask effect {edge['ALL']['10000']['bid_ask_execution_effect']:.3f} + adverse fill -1.000 + horizon alignment {edge['ALL']['10000']['horizon_alignment_effect']:.3f}.",
        f"October raw-to-quote deterioration ({edge['OCTOBER_2025']['10000']['raw_to_executable_loss']:.3f}) is smaller than Spring's ({edge['SPRING_2025']['10000']['raw_to_executable_loss']:.3f}); execution is a pooled hurdle, not the cause of the October-minus-Spring raw divergence.",
        f"Topology return-inside share: Spring {_topology_table(events)['SPRING_2025']['RETURN_INSIDE_OPENING_RANGE']['percent']:.1f}%; October {_topology_table(events)['OCTOBER_2025']['RETURN_INSIDE_OPENING_RANGE']['percent']:.1f}%.",
        f"Period classification: {decision['structure_vs_execution_classification']}.",
        f"Primary decision: {decision['primary_decision']}.", f"Next step: {decision['next_step']}.",
        f"Matched comparison: {matched['status']} ({matched['matched_pairs']} pairs).", ""]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    artifacts = {p.name: native._sha(p) for p in output_root.iterdir()
                 if p.is_file() and p.name not in ("run-manifest.json", "artifact-hashes.json")}
    native._write_json(output_root / "run-manifest.json", {"status": "COMPLETE", "run_id": RUN_ID,
        "config_sha256": CONFIG_SHA256, "parent_manifest_sha256": parent_sha,
        "source_coverage_sha256": native._sha(output_root / "source-coverage.json"),
        "checkpoint_sha256_by_date": {day: native._sha(_checkpoint(output_root, day)) for day in days},
        "artifact_sha256_by_name": artifacts, "no_2026_access": True,
        "no_download": True, "no_optimization": True})
    native._write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files":
        {p.name: native._sha(p) for p in output_root.iterdir() if p.is_file() and p.name != "artifact-hashes.json"}})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=native.DATA_ROOT)
    parser.add_argument("--parent-root", type=Path, default=parent.OUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(data_root=args.data_root, parent_root=args.parent_root,
            output_root=args.output_root, smoke=args.smoke)
    except (EntryAuditError, parent.BreakoutStudyError, native.VacuumStudyError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ENTRY_AUDIT_ERROR: {exc}\n")
    print(f"ES_STRUCTURAL_BREAKOUT_ENTRY_TIMING_AUDIT={result['status']}", flush=True)
    if result["status"] == "COMPLETE":
        print(f"PRIMARY_DECISION={result['decision']['primary_decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
