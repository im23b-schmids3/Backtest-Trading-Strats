"""Bounded, preregistered Spring-only rescue of Liquidity Vacuum V1.

October is used only for the fixed V1 chase diagnostic and, after a hashed
Spring-only candidate freeze, one secondary development evaluation.  Derived
native MBP-10 compact files are source-bound and never enter the repository's
strategy selection data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna

from . import mac_2025_absorption_relative_normalization as relative
from . import mac_2025_es_liquidity_vacuum_v1 as fixed
from . import mac_2025_es_only_train_baseline as baseline

RUN_ID = "CMEOrderflow_ES_LIQUIDITY_VACUUM_V1_RESCUE_OPTUNA"
OUT_ROOT = Path("research_runs") / RUN_ID
CACHE_VERSION = "vacuum-rescue-compact-v1"
_CACHE_VERIFIED: set[str] = set()
TRIAL_CAP = 500
BATCH_SIZE = 16
SEED = 20251005
STUDY_NAME = "ES_LIQUIDITY_VACUUM_V1_RESCUE_SPRING_ONLY_500_TPE"
PERCENTILES = tuple(round(i / 100, 2) for i in range(85, 98))
SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "pressure_window_ms": (250, 500, 750, 1000),
    "pressure_percentile": PERCENTILES,
    "depth_depletion_threshold": tuple(round(i / 100, 2) for i in range(20, 66, 5)),
    "refill_weakness_threshold": tuple(round(i / 100, 2) for i in range(30, 81, 5)),
    "flow_persistence_threshold": (0.50, 0.60, 0.70, 0.80, 0.90),
    "max_favorable_move_before_entry_ticks": (2, 4, 6, 8, 10),
    "stop_ticks": tuple(range(4, 11)),
    "target_r": (1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00),
}
OBJECTIVE_SPEC = {
    "version": "VACUUM_RESCUE_SPRING_ROBUST_OBJECTIVE_V1",
    "minimum_trades": 100, "minimum_active_dates": 8, "minimum_active_iso_weeks": 5,
    "ineligible_score": -1000.0,
    "formula": "mean_trade_R + 0.20*median_daily_R + 0.10*median_weekly_R - 0.01*max_drawdown_R",
    "spring_hard_screen": {
        "minimum_pf": 1.20, "minimum_net_R_exclusive": 0.0,
        "minimum_average_R_exclusive": 0.0, "minimum_median_daily_R": 0.0,
        "minimum_positive_active_week_fraction_exclusive": 0.50,
        "maximum_single_positive_day_share": 0.35,
        "maximum_single_positive_week_share": 0.50,
        "maximum_drawdown_as_fraction_of_gross_positive_R": 0.75,
        "minimum_each_direction_R": "-max(5.0, 0.25*gross_positive_R)",
    },
    "neighbor_rule": {
        "robust": "at least 60% pass hard screen, at least 75% positive, median objective >= center-0.25",
        "moderate": "at least 40% pass hard screen, at least 60% positive, median objective >= center-0.50",
    },
}
PREREG = {
    "run_id": RUN_ID, "trial_cap": TRIAL_CAP, "batch_size": BATCH_SIZE,
    "seed": SEED, "sampler": "multivariate TPESampler(n_startup_trials=64, seed=20251005)",
    "search_space": {key: list(values) for key, values in SEARCH_SPACE.items()},
    "objective": OBJECTIVE_SPEC,
    "optimization_dates": list(fixed.SPRING_DATES),
    "october_role": "SECONDARY_DEV_CHECK_AFTER_HASHED_SPRING_SELECTION_ONLY",
    "october_dates": list(fixed.OCTOBER_DATES),
    "fixed_semantics": {key: fixed.CONFIG[key] for key in (
        "event_refractory_seconds", "baseline_depth_seconds", "baseline_depth_sampling_ms",
        "refill_observation_ms", "mlofi_bins", "mlofi_bin_ms", "price_confirmation_min_ticks",
        "entry_delay_ms", "max_hold_seconds", "post_exit_refractory_seconds",
        "history_sample_per_date", "source_schema", "instrument")},
    "target_tick_rounding": "floor(stop_ticks*target_r + 0.5) to nearest valid ES tick",
    "vacuum_collapse_rule": "at least 3 of depletion<=0.25, refill<=0.35, persistence<=0.60, chase>=8; and at least 50% of clustered events pass confirmation",
    "flow_increment_rule": "candidate mean markout must exceed flow-only at both 500ms and 1000ms in both Spring and October",
    "source_mlofi_semantic_sha256": relative.EXPECTED_TAPE_SEMANTIC_SHA,
}


class RescueError(RuntimeError):
    pass


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_once(path: Path, payload: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RescueError(f"immutable artifact differs: {path}")
        return
    fixed._write_json(path, payload)


def _read_fixed() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = fixed.OUT_ROOT
    summary = json.loads((root / "summary.json").read_text())
    if (summary.get("status") != "COMPLETE" or summary.get("config_sha256") != fixed.CONFIG_SHA256
            or summary.get("spring_dates") != list(fixed.SPRING_DATES)
            or summary.get("october_dates") != list(fixed.OCTOBER_DATES)):
        raise RescueError("fixed V1 reference identity is invalid")
    with gzip.open(root / "trades.jsonl.gz", "rt") as stream:
        trades = [json.loads(line) for line in stream]
    if len(trades) != 520 or not math.isclose(sum(t["net_R"] for t in trades), -286.57754010695186, abs_tol=1e-8):
        raise RescueError("fixed V1 trade reference changed")
    return summary, trades


def _source_rows(days: Sequence[str], data_root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((data_root / baseline.MANIFEST_NAME).read_text())
    if manifest.get("status") != "COMPLETE":
        raise RescueError("native ES source manifest is incomplete")
    wanted = set(days)
    selected: dict[str, dict[str, Any]] = {}
    for row in manifest.get("requests", {}).values():
        day = str(row.get("session_date", ""))
        if day not in wanted or row.get("schema") != "mbp-10":
            continue
        if day in selected:
            raise RescueError(f"duplicate native ES source row for {day}")
        if row.get("symbol") != fixed._expected_contract(day):
            raise RescueError(f"native ES contract mismatch for {day}")
        path = data_root / str(row.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)) or fixed._sha(path) != row.get("sha256"):
            raise RescueError(f"native ES source integrity mismatch for {day}")
        from databento import DBNStore
        metadata = DBNStore.from_file(path).metadata
        if metadata.dataset != "GLBX.MDP3" or metadata.schema != "mbp-10" or row["symbol"] not in metadata.symbols:
            raise RescueError(f"native ES DBN metadata mismatch for {day}")
        selected[day] = {**row, "absolute_path": str(path)}
    if set(selected) != wanted:
        raise RescueError(f"missing native ES dates: {sorted(wanted-set(selected))}")
    return selected


def _cache_paths(root: Path, day: str) -> tuple[Path, Path]:
    return root / "_cache" / f"{day}.compact.bin", root / "_cache" / f"{day}.compact.json"


def _compact(day: str, row: Mapping[str, Any], root: Path) -> np.ndarray:
    data_path, meta_path = _cache_paths(root, day)
    if data_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (meta.get("source_sha256") == row["sha256"] and meta.get("bytes") == data_path.stat().st_size
                and meta.get("dtype") == repr(relative.COMPACT_DTYPE.descr)
                and meta.get("cache_version") == CACHE_VERSION):
            if day not in _CACHE_VERIFIED:
                if fixed._sha(data_path) != meta.get("sha256"):
                    raise RescueError(f"derived compact cache hash mismatch for {day}")
                _CACHE_VERIFIED.add(day)
            return np.memmap(data_path, dtype=relative.COMPACT_DTYPE, mode="r", shape=(int(meta["rows"]),))
    compact, scratch, provenance = relative._extract_compact(day, Path(row["absolute_path"]), data_path, row["sha256"])
    count = len(compact)
    del compact
    os.replace(scratch, data_path)
    fixed._write_json(meta_path, {"source_sha256": row["sha256"], "rows": count,
                                  "bytes": data_path.stat().st_size, "dtype": repr(relative.COMPACT_DTYPE.descr),
                                  "raw_rows": provenance["raw_rows"], "cache_version": CACHE_VERSION,
                                  "sha256": fixed._sha(data_path)})
    _CACHE_VERIFIED.add(day)
    return np.memmap(data_path, dtype=relative.COMPACT_DTYPE, mode="r", shape=(count,))


def _pressure_path(root: Path, day: str, window: int) -> Path:
    return root / "_cache" / f"{day}.pressure-{window}.npy"


def _pressure(day: str, window: int, rows: np.ndarray, root: Path) -> np.ndarray:
    path = _pressure_path(root, day, window)
    if path.is_file():
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(array) == len(rows):
            return array
    array = fixed.rolling_pressure(rows, window * 1_000_000)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _phase0_chase(trades: Sequence[Mapping[str, Any]], mids: Mapping[tuple[str, int, float], float]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = {label: [] for label in ("0_to_2", "gt2_to_4", "gt4_to_6", "gt6_to_10", "gt10", "negative")}
    all_rows = []
    for trade in trades:
        key = (str(trade["date"]), int(trade["event_start_time_ns"]), float(trade["pressure"]))
        if key not in mids:
            raise RescueError(f"could not recover exact original pressure event mid: {key}")
        direction = 1 if trade["entry_direction"] == "LONG" else -1
        chase = direction * (float(trade["entry_price"]) - mids[key]) / fixed.TICK
        row = {**trade, "original_flow_event_mid": mids[key], "favorable_move_before_entry_ticks": chase}
        all_rows.append(row)
        label = ("negative" if chase < 0 else "0_to_2" if chase <= 2 else "gt2_to_4" if chase <= 4
                 else "gt4_to_6" if chase <= 6 else "gt6_to_10" if chase <= 10 else "gt10")
        buckets[label].append(row)
    result: dict[str, Any] = {}
    for label, members in buckets.items():
        r = [float(t["net_R"]) for t in members]
        result[label] = {"trade_count": len(members),
                         "spring_count": sum(t["date"] in fixed.SPRING_DATES for t in members),
                         "october_count": sum(t["date"] in fixed.OCTOBER_DATES for t in members),
                         "markouts": fixed._markout_aggregate(members),
                         "win_rate": sum(x > 0 for x in r) / len(r) if r else None,
                         "average_R": float(np.mean(r)) if r else None,
                         "median_R": float(np.median(r)) if r else None,
                         "mean_MFE_ticks": float(np.mean([t["mfe_ticks"] for t in members])) if members else None,
                         "mean_MAE_ticks": float(np.mean([t["mae_ticks"] for t in members])) if members else None}
    ordered = [result[k] for k in ("0_to_2", "gt2_to_4", "gt4_to_6", "gt6_to_10", "gt10")]
    valid = [x for x in ordered if x["trade_count"] >= 10]
    if len(valid) >= 3 and all(valid[i]["average_R"] >= valid[i+1]["average_R"] for i in range(len(valid)-1)):
        classification = "CHASE_DELAY_STRONGLY_SUPPORTED"
    elif len(valid) >= 2 and valid[0]["average_R"] > valid[-1]["average_R"]:
        classification = "CHASE_DELAY_PARTIALLY_SUPPORTED"
    else:
        classification = "NO_CLEAR_CHASE_EFFECT"
    per_trade = [{"date": row["date"], "event_start_time_ns": row["event_start_time_ns"],
                  "entry_time_ns": row["entry_time_ns"], "entry_direction": row["entry_direction"],
                  "original_flow_event_mid": row["original_flow_event_mid"],
                  "entry_price": row["entry_price"],
                  "favorable_move_before_entry_ticks": row["favorable_move_before_entry_ticks"]}
                 for row in all_rows]
    return {"classification": classification, "buckets": result, "trade_count": len(all_rows),
            "per_trade": per_trade,
            "price_definition": "exact native MBP-10 event-row mid; prospective entry is existing adverse quote fill"}


def _event_mid_lookup(rows: np.ndarray, pressure: np.ndarray,
                      trades: Sequence[Mapping[str, Any]], day: str,
                      event_indices: Mapping[tuple[int, float], int] | None = None) -> dict[tuple[str, int, float], float]:
    ts = np.ascontiguousarray(rows["ts"], dtype=np.int64)
    out = {}
    for trade in trades:
        event_ns = int(trade["event_start_time_ns"])
        if event_indices is not None:
            index = event_indices.get((event_ns, float(trade["pressure"])))
            if index is None or not 0 <= index < len(rows):
                raise RescueError(f"fixed V1 checkpoint has no event index for {day} {event_ns}")
            if int(ts[index]) != event_ns or not math.isclose(float(pressure[index]), float(trade["pressure"]), abs_tol=1e-9):
                raise RescueError(f"fixed V1 checkpoint event index does not match native rows: {day} {event_ns}")
            side = "ask5" if trade["entry_direction"] == "LONG" else "bid5"
            if not math.isclose(float(rows[index][side]), float(trade["event_depth"]), abs_tol=1e-9):
                raise RescueError(f"fixed V1 checkpoint event depth does not match native rows: {day} {event_ns}")
            out[(day, event_ns, float(trade["pressure"]))] = float(rows[index]["mid"])
            continue
        lo = int(np.searchsorted(ts, event_ns, side="left"))
        hi = int(np.searchsorted(ts, event_ns, side="right"))
        candidates = [i for i in range(lo, hi) if math.isclose(float(pressure[i]), float(trade["pressure"]), abs_tol=1e-9)]
        side = "ask5" if trade["entry_direction"] == "LONG" else "bid5"
        candidates = [i for i in candidates if math.isclose(float(rows[i][side]),
                                                              float(trade["event_depth"]), abs_tol=1e-9)]
        mids = {float(rows[i]["mid"]) for i in candidates}
        if len(mids) != 1:
            raise RescueError(f"event timestamp/pressure/depth does not determine one event mid: {day} {event_ns}: {len(candidates)} rows, {len(mids)} mids")
        out[(day, event_ns, float(trade["pressure"]))] = mids.pop()
    return out


def sample_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {"pressure_window_ms": trial.suggest_categorical("pressure_window_ms", SEARCH_SPACE["pressure_window_ms"]),
            "pressure_percentile": trial.suggest_float("pressure_percentile", .85, .97, step=.01),
            "depth_depletion_threshold": trial.suggest_float("depth_depletion_threshold", .20, .65, step=.05),
            "refill_weakness_threshold": trial.suggest_float("refill_weakness_threshold", .30, .80, step=.05),
            "flow_persistence_threshold": trial.suggest_categorical("flow_persistence_threshold", SEARCH_SPACE["flow_persistence_threshold"]),
            "max_favorable_move_before_entry_ticks": trial.suggest_categorical("max_favorable_move_before_entry_ticks", SEARCH_SPACE["max_favorable_move_before_entry_ticks"]),
            "stop_ticks": trial.suggest_int("stop_ticks", 4, 10),
            "target_r": trial.suggest_categorical("target_r", SEARCH_SPACE["target_r"])}


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if set(parameters) != set(SEARCH_SPACE):
        raise RescueError("rescue search must have exactly eight active parameters")
    out = dict(parameters)
    for name, values in SEARCH_SPACE.items():
        if not any(math.isclose(float(out[name]), float(v), abs_tol=1e-9) for v in values):
            raise RescueError(f"search boundary violation: {name}={out[name]}")
    return out


def robust_objective(metrics: Mapping[str, Any]) -> float:
    if (metrics["trades"] < 100 or metrics["active_dates"] < 8 or metrics["active_weeks"] < 5):
        return -1000.0
    score = (float(metrics["mean_trade_R"]) + .20 * float(metrics["median_daily_R"])
             + .10 * float(metrics["median_weekly_R"]) - .01 * float(metrics["max_drawdown_R"]))
    if not math.isfinite(score):
        raise RescueError("non-finite robust objective")
    return score


def _sample_path(root: Path, day: str, window: int) -> Path:
    return root / "_cache" / f"{day}.sample-{window}.npy"


def _sample_pressure(day: str, window: int, rows: np.ndarray, root: Path) -> np.ndarray:
    path = _sample_path(root, day, window)
    if path.is_file():
        return np.load(path, allow_pickle=False)
    values = np.asarray(np.abs(_pressure(day, window, rows, root)), dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) > fixed.HISTORY_SAMPLE:
        values = values[np.linspace(0, len(values)-1, num=fixed.HISTORY_SAMPLE, dtype=np.int64)]
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return values


def _prepare_phase0(root: Path, sources: Mapping[str, Mapping[str, Any]],
                    fixed_trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mids: dict[tuple[str, int, float], float] = {}
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in fixed_trades:
        by_day[str(trade["date"])].append(trade)
    for day in (fixed.DEPENDENCY_DATES[0], *fixed.SPRING_DATES, *fixed.OCTOBER_DATES):
        rows = _compact(day, sources[day], root)
        windows = SEARCH_SPACE["pressure_window_ms"] if day in fixed.SPRING_DATES or day == fixed.DEPENDENCY_DATES[0] else (500,)
        for window in windows:
            _sample_pressure(day, int(window), rows, root)
        if day in by_day:
            with gzip.open(fixed._checkpoint_path(fixed.OUT_ROOT, day), "rt") as stream:
                checkpoint = json.load(stream)
            if checkpoint.get("source_sha256") != sources[day]["sha256"] or checkpoint.get("status") != "DATE_COMPLETE":
                raise RescueError(f"fixed V1 checkpoint integrity mismatch for {day}")
            indices = {(int(e["timestamp_ns"]), float(e["pressure"])): int(e["row_index"])
                       for e in checkpoint["payload"]["clustered_events"]}
            mids.update(_event_mid_lookup(rows, _pressure(day, 500, rows, root), by_day[day], day, indices))
        print(f"RESCUE_PHASE0_DATE={day} trades={len(by_day[day])} rows={len(rows)}", flush=True)
        del rows
    if len(mids) != len(fixed_trades):
        raise RescueError(f"recovered {len(mids)} of {len(fixed_trades)} event prices")
    result = _phase0_chase(fixed_trades, mids)
    fixed._write_json(root / "entry-chase-diagnostic.json", result)
    return result


def _threshold_catalog(root: Path, source_days: Sequence[str], target_days: Sequence[str],
                       windows: Sequence[int]) -> dict[tuple[str, int, float], float]:
    targets = set(target_days)
    catalog: dict[tuple[str, int, float], float] = {}
    for window in windows:
        history: list[np.ndarray] = []
        for day in source_days:
            if day in targets:
                if not history:
                    raise RescueError(f"no prior-date pressure history for {day} window={window}")
                q = np.quantile(np.concatenate(history), np.asarray(PERCENTILES, dtype=np.float64))
                for percentile, threshold in zip(PERCENTILES, q):
                    catalog[(day, int(window), percentile)] = float(threshold)
            path = _sample_path(root, day, int(window))
            if not path.is_file():
                raise RescueError(f"missing prior-date pressure sample: {path}")
            history.append(np.load(path, allow_pickle=False))
    return catalog


def _cluster_greedy(indices: np.ndarray, ts: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    """Vector-boundary equivalent of V1's first-hit two-second clustering."""
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    output: list[int] = []
    for session in (0, 1, 2):
        selected = indices[sessions[indices] == session]
        times = ts[selected]
        pos = 0
        while pos < len(selected):
            output.append(int(selected[pos]))
            pos = int(np.searchsorted(times, int(times[pos]) + 2_000_000_000, side="left"))
    return np.asarray(sorted(output), dtype=np.int64)


FEATURE_DTYPE = np.dtype([
    ("row_index", "<i8"), ("timestamp_ns", "<i8"), ("session", "i1"), ("direction", "i1"),
    ("pressure", "<f8"), ("event_mid", "<f8"), ("baseline_depth", "<f8"),
    ("event_depth", "<f8"), ("depletion", "<f8"), ("recovery_ratio", "<f8"),
    ("persistence", "<f8"), ("confirmation_ticks", "<f8"), ("signal_anchor_ns", "<i8"),
    ("chase_ticks", "<f8"), ("has_entry", "?"),
])


def _features_path(root: Path, day: str, window: int) -> Path:
    return root / "_cache" / f"{day}.features-{window}.npy"


def _cluster_path(root: Path, day: str, window: int, percentile: float) -> Path:
    return root / "_cache" / f"{day}.cluster-{window}-{int(round(percentile*100))}.npz"


def _prepare_window(day: str, window: int, rows: np.ndarray, tape: np.ndarray,
                    catalog: Mapping[tuple[str, int, float], float], root: Path) -> np.ndarray:
    feature_path = _features_path(root, day, window)
    cluster_paths = [_cluster_path(root, day, window, p) for p in PERCENTILES]
    if feature_path.is_file() and all(p.is_file() for p in cluster_paths):
        return np.load(feature_path, mmap_mode="r", allow_pickle=False)
    pressure = _pressure(day, window, rows, root)
    ts = np.ascontiguousarray(rows["ts"], dtype=np.int64)
    sessions = np.full(len(ts), -1, dtype=np.int8)
    for code, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = baseline._session_windows(day)[name]
        sessions[(ts >= start) & (ts < end)] = code
    low_threshold = min(catalog[(day, window, p)] for p in PERCENTILES)
    raw_low = np.flatnonzero((np.abs(pressure) >= low_threshold) & (pressure != 0) & (sessions >= 0))
    clusters: dict[float, np.ndarray] = {}
    for percentile in PERCENTILES:
        threshold = catalog[(day, window, percentile)]
        raw = raw_low[np.abs(pressure[raw_low]) >= threshold]
        clusters[percentile] = _cluster_greedy(raw, ts, sessions)
        path = _cluster_path(root, day, window, percentile)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as stream:
            np.savez(stream, indices=clusters[percentile], raw_count=np.asarray(len(raw)))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    union = np.unique(np.concatenate(list(clusters.values()))) if clusters else np.empty(0, dtype=np.int64)
    features = np.zeros(len(union), dtype=FEATURE_DTYPE)
    features["row_index"] = union
    grids = fixed._depth_baseline_grid(rows, day, timestamps=ts)
    tape_ts = np.ascontiguousarray(tape["timestamp_ns"], dtype=np.int64)
    tape_sessions = np.ascontiguousarray(tape["session"], dtype=np.int8)
    for j, index in enumerate(union):
        event = {"row_index": int(index), "timestamp_ns": int(ts[index]),
                 "session": int(sessions[index]), "direction": 1 if pressure[index] > 0 else -1}
        state = fixed._classify_event(rows, event, day=day, depth_grid=grids[event["session"]],
                                      timestamps=ts, minimum_depletion=-1e9,
                                      maximum_recovery_ratio=1e9, minimum_persistence=0.0)
        f = features[j]
        f["timestamp_ns"] = event["timestamp_ns"]; f["session"] = event["session"]
        f["direction"] = event["direction"]; f["pressure"] = float(pressure[index])
        f["event_mid"] = float(rows[index]["mid"])
        for dst, src in (("baseline_depth", "baseline_depth"), ("event_depth", "event_depth"),
                         ("depletion", "depletion"), ("recovery_ratio", "recovery_ratio"),
                         ("persistence", "mlofi_persistence"), ("confirmation_ticks", "confirmation_change_ticks")):
            f[dst] = float(state[src]) if state.get(src) is not None else math.nan
        anchor = state.get("signal_anchor_ns")
        f["signal_anchor_ns"] = int(anchor) if anchor is not None else -1
        f["chase_ticks"] = math.nan
        if anchor is not None:
            entry_ix = int(np.searchsorted(tape_ts, int(anchor) + 2_000_000, side="left"))
            if entry_ix < len(tape) and int(tape_sessions[entry_ix]) == event["session"]:
                bid, ask = float(tape[entry_ix]["bid"]), float(tape[entry_ix]["ask"])
                if math.isfinite(bid) and math.isfinite(ask) and ask > bid:
                    entry = ask + fixed.TICK if event["direction"] > 0 else bid - fixed.TICK
                    f["chase_ticks"] = event["direction"] * (entry - f["event_mid"]) / fixed.TICK
                    f["has_entry"] = True
        if (j+1) % 10000 == 0:
            print(f"RESCUE_FEATURE_PROGRESS date={day} window={window} {j+1}/{len(union)}", flush=True)
    temp = feature_path.with_name(feature_path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, feature_path)
    print(f"RESCUE_FEATURES_READY date={day} window={window} unique={len(union)}", flush=True)
    return np.load(feature_path, mmap_mode="r", allow_pickle=False)


def _signal(feature: np.void, percentile: float) -> dict[str, Any]:
    return {"timestamp_ns": int(feature["timestamp_ns"]), "session": int(feature["session"]),
            "direction": int(feature["direction"]), "signal_anchor_ns": int(feature["signal_anchor_ns"]),
            "event_start_mid": float(feature["event_mid"]), "pressure": float(feature["pressure"]),
            "pressure_percentile": percentile, "depletion": float(feature["depletion"]),
            "recovery_ratio": float(feature["recovery_ratio"]),
            "refill_weakness": 1.0 - float(feature["recovery_ratio"]),
            "mlofi_persistence": float(feature["persistence"]),
            "baseline_depth": float(feature["baseline_depth"]), "event_depth": float(feature["event_depth"]),
            "confirmation_change_ticks": float(feature["confirmation_ticks"])}


def evaluate_prepared_day(day: str, parameters: Mapping[str, Any], features: np.ndarray,
                          clustered: np.ndarray, raw_count: int, tape: np.ndarray,
                          *, diagnostics: bool = False) -> dict[str, Any]:
    """Evaluate only causal precomputed features; no source dates outside *day*."""
    params = validate_parameters(parameters)
    selected = features[np.searchsorted(features["row_index"], clustered)] if len(clustered) else features[:0]
    if len(selected) and not np.array_equal(selected["row_index"], clustered):
        raise RescueError(f"cluster cache and feature cache disagree for {day}")
    depletion = np.isfinite(selected["depletion"]) & (selected["depletion"] >= params["depth_depletion_threshold"])
    refill = depletion & np.isfinite(selected["recovery_ratio"]) & ((1.0-selected["recovery_ratio"]) >= params["refill_weakness_threshold"] - 1e-12)
    persistence = refill & np.isfinite(selected["persistence"]) & (selected["persistence"] >= params["flow_persistence_threshold"])
    chase = persistence & selected["has_entry"] & (selected["chase_ticks"] <= params["max_favorable_move_before_entry_ticks"])
    confirmation = chase & np.isfinite(selected["confirmation_ticks"]) & (selected["confirmation_ticks"] >= 0)
    attrition = {"pressure_events": int(raw_count), "clustered_events": int(len(clustered)),
                 "after_depletion": int(depletion.sum()), "after_refill": int(refill.sum()),
                 "after_persistence": int(persistence.sum()), "after_max_chase": int(chase.sum()),
                 "after_price_confirmation": int(confirmation.sum())}
    tape_ts = np.ascontiguousarray(tape["timestamp_ns"], dtype=np.int64)
    tape_sessions = np.ascontiguousarray(tape["session"], dtype=np.int8)
    bounds = {code: fixed._session_bounds_ns_for_code(code, tape, sessions=tape_sessions) for code in (0, 1, 2)}
    trades: list[dict[str, Any]] = []
    busy_until = -10**30
    for feature in selected[confirmation]:
        signal = _signal(feature, float(params["pressure_percentile"]))
        trade = fixed._entry_and_exit(tape, signal, busy_until, tape_ts, tape_sessions, bounds,
                                      stop_ticks=int(params["stop_ticks"]), target_r=float(params["target_r"]),
                                      max_favorable_move_before_entry_ticks=float(params["max_favorable_move_before_entry_ticks"]),
                                      include_diagnostics=diagnostics)
        if trade is not None:
            trades.append(trade)
            busy_until = int(trade["exit_time_ns"]) + 2_000_000_000
    attrition["actual_entries"] = len(trades)
    flow_rows: list[dict[str, Any]] = []
    if diagnostics:
        for feature in selected:
            flow_rows.append({"markouts": fixed._path_markouts(
                tape, int(feature["timestamp_ns"]), float(feature["event_mid"]),
                int(feature["direction"]), int(feature["session"]),
                event_timestamps=tape_ts, event_sessions=tape_sessions, session_bounds=bounds)})
    return {"date": day, "trades": trades, "attrition": attrition,
            "flow_only": fixed._flow_only_summary(tape, flow_rows) if diagnostics else None}


def _trial_metrics(payloads: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    trades = sorted((trade for p in payloads for trade in p["trades"]), key=lambda t: int(t["entry_time_ns"]))
    total = fixed._summarize_trades(trades)
    daily, weekly = fixed._daily_weekly(trades, dates)
    active_daily = [r for r in daily if r["trade_count"] > 0]
    active_weekly = [r for r in weekly if r["trades"] > 0]
    positive_daily = [max(0.0, float(r["R"])) for r in daily]
    positive_weekly = [max(0.0, float(r["R"])) for r in weekly]
    gross_positive = sum(max(0.0, float(t["net_R"])) for t in trades)
    long_r = sum(t["net_R"] for t in trades if t["entry_direction"] == "LONG")
    short_r = sum(t["net_R"] for t in trades if t["entry_direction"] == "SHORT")
    return {"trades": len(trades), "net_R": total["net_R"], "PF": total["profit_factor"],
            "max_drawdown_R": total["max_drawdown_R"],
            "mean_trade_R": total["average_R"] or 0.0,
            "median_trade_R": total["median_R"] or 0.0,
            "active_dates": len(active_daily), "active_weeks": len(active_weekly),
            "median_daily_R": float(np.median([r["R"] for r in daily])),
            "median_weekly_R": float(np.median([r["R"] for r in weekly])),
            "positive_day_fraction": sum(r["R"] > 0 for r in daily) / len(daily),
            "positive_week_fraction": sum(r["R"] > 0 for r in weekly) / len(weekly),
            "positive_active_week_fraction": (sum(r["R"] > 0 for r in active_weekly) / len(active_weekly)) if active_weekly else 0.0,
            "maximum_positive_day_share": max(positive_daily, default=0.0) / sum(positive_daily) if sum(positive_daily) > 0 else 1.0,
            "maximum_positive_week_share": max(positive_weekly, default=0.0) / sum(positive_weekly) if sum(positive_weekly) > 0 else 1.0,
            "gross_positive_R": gross_positive, "long_R": float(long_r), "short_R": float(short_r),
            "daily": daily, "weekly": weekly,
            "summary": total}


def spring_hard_screen(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["trades"] < 100: reasons.append("TRADES_BELOW_100")
    if metrics["active_dates"] < 8: reasons.append("ACTIVE_DATES_BELOW_8")
    if metrics["active_weeks"] < 5: reasons.append("ACTIVE_WEEKS_BELOW_5")
    pf = metrics["PF"]
    if not isinstance(pf, (int, float)) or pf < 1.20: reasons.append("PF_BELOW_1P20")
    if metrics["net_R"] <= 0: reasons.append("NET_R_NONPOSITIVE")
    if metrics["mean_trade_R"] <= 0: reasons.append("AVERAGE_R_NONPOSITIVE")
    if metrics["median_daily_R"] < 0: reasons.append("MEDIAN_DAILY_R_NEGATIVE")
    if metrics["positive_active_week_fraction"] <= .50: reasons.append("ACTIVE_WEEKS_NOT_MAJORITY_POSITIVE")
    if metrics["maximum_positive_day_share"] > .35: reasons.append("DAY_CONCENTRATION_OVER_35_PERCENT")
    if metrics["maximum_positive_week_share"] > .50: reasons.append("WEEK_CONCENTRATION_OVER_50_PERCENT")
    gross_positive = float(metrics["gross_positive_R"])
    if metrics["max_drawdown_R"] > .75 * gross_positive: reasons.append("DRAWDOWN_TOO_LARGE")
    direction_floor = -max(5.0, .25 * gross_positive)
    if metrics["long_R"] < direction_floor: reasons.append("LONG_CATASTROPHICALLY_NEGATIVE")
    if metrics["short_R"] < direction_floor: reasons.append("SHORT_CATASTROPHICALLY_NEGATIVE")
    return not reasons, reasons


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {"number": int(trial.number), "state": trial.state.name,
            "objective": float(trial.value) if trial.value is not None else None,
            "parameters": dict(trial.params), "metrics": trial.user_attrs.get("metrics"),
            "hard_screen": trial.user_attrs.get("hard_screen")}


def _evaluate_configs(configs: Sequence[Mapping[str, Any]], dates: Sequence[str],
                      source_rows: Mapping[str, Mapping[str, Any]], root: Path,
                      catalog: Mapping[tuple[str, int, float], float],
                      *, diagnostics: bool = False) -> list[dict[str, Any]]:
    payloads: list[list[dict[str, Any]]] = [[] for _ in configs]
    for day in dates:
        rows = _compact(day, source_rows[day], root)
        tape_path = fixed._tape_path(day)
        tape, _ = fixed._load_tape(day, tape_path, str(source_rows[day]["sha256"]))
        windows = sorted({int(params["pressure_window_ms"]) for params in configs})
        features_by_window = {window: _prepare_window(day, window, rows, tape, catalog, root) for window in windows}
        cluster_by = {}
        for params in configs:
            window = int(params["pressure_window_ms"]); percentile = round(float(params["pressure_percentile"]), 2)
            key = (window, percentile)
            if key not in cluster_by:
                with np.load(_cluster_path(root, day, window, percentile), allow_pickle=False) as archive:
                    cluster_by[key] = (np.asarray(archive["indices"]), int(archive["raw_count"].item()))
        for i, params in enumerate(configs):
            window = int(params["pressure_window_ms"]); percentile = round(float(params["pressure_percentile"]), 2)
            indices, raw_count = cluster_by[(window, percentile)]
            payloads[i].append(evaluate_prepared_day(day, params, features_by_window[window], indices,
                                                     raw_count, tape, diagnostics=diagnostics))
        print(f"RESCUE_EVAL_DATE={day} configs={len(configs)} diagnostics={diagnostics}", flush=True)
        del rows, tape, features_by_window
    return [{"metrics": _trial_metrics(p, dates), "date_payloads": p} for p in payloads]


def _study(root: Path) -> optuna.study.Study:
    path = (root / "optuna-study.db").resolve()
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=64, multivariate=True)
    study = optuna.create_study(study_name=STUDY_NAME, direction="maximize",
                                sampler=sampler, storage=f"sqlite:///{path}", load_if_exists=True)
    prereg_hash = fixed._sha(root / "optimization-preregistration.json")
    if study.user_attrs.get("preregistration_sha256") not in (None, prereg_hash):
        raise RescueError("existing Optuna study has a different preregistration")
    if study.user_attrs.get("spring_dates") not in (None, list(fixed.SPRING_DATES)):
        raise RescueError("existing Optuna study has a different date scope")
    if "preregistration_sha256" not in study.user_attrs:
        study.set_user_attr("preregistration_sha256", prereg_hash)
    if "spring_dates" not in study.user_attrs:
        study.set_user_attr("spring_dates", list(fixed.SPRING_DATES))
    return study


def _distributions() -> dict[str, optuna.distributions.BaseDistribution]:
    return {
        "pressure_window_ms": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["pressure_window_ms"]),
        "pressure_percentile": optuna.distributions.FloatDistribution(.85, .97, step=.01),
        "depth_depletion_threshold": optuna.distributions.FloatDistribution(.20, .65, step=.05),
        "refill_weakness_threshold": optuna.distributions.FloatDistribution(.30, .80, step=.05),
        "flow_persistence_threshold": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["flow_persistence_threshold"]),
        "max_favorable_move_before_entry_ticks": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["max_favorable_move_before_entry_ticks"]),
        "stop_ticks": optuna.distributions.IntDistribution(4, 10),
        "target_r": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["target_r"]),
    }


def _result_path(root: Path, number: int) -> Path:
    return root / "trial-results" / f"{number:03d}.json"


def _trial_result(number: int, parameters: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {k: v for k, v in result["metrics"].items() if k not in ("daily", "weekly", "summary")}
    eligible, reasons = spring_hard_screen(metrics)
    return {"number": int(number), "parameters": validate_parameters(parameters),
            "metrics": metrics, "objective": robust_objective(metrics),
            "hard_screen": {"pass": eligible, "reasons": reasons}}


def _run_optuna(root: Path, sources: Mapping[str, Mapping[str, Any]],
                catalog: Mapping[tuple[str, int, float], float]) -> list[dict[str, Any]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _study(root)
    distributions = _distributions()
    while True:
        trials = study.get_trials(deepcopy=False)
        if len(trials) > TRIAL_CAP:
            raise RescueError(f"trial cap exceeded: {len(trials)}>{TRIAL_CAP}")
        failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        if failed:
            raise RescueError(f"failed trials require inspection; no replacement trials started: {[t.number for t in failed]}")
        complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(complete) == TRIAL_CAP and not running:
            break
        if len(running) > BATCH_SIZE:
            raise RescueError("more than one outstanding trial batch")
        if not running:
            for _ in range(min(BATCH_SIZE, TRIAL_CAP-len(trials))):
                trial = study.ask(fixed_distributions=distributions)
                validate_parameters(trial.params)
            running = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.RUNNING]
        if not running:
            raise RescueError("study cannot make progress")
        configs = [validate_parameters(t.params) for t in running]
        cached = [(_result_path(root, t.number).is_file()) for t in running]
        missing = [(t, configs[i]) for i, t in enumerate(running) if not cached[i]]
        if missing:
            evaluated = _evaluate_configs([p for _, p in missing], fixed.SPRING_DATES, sources, root, catalog)
            for (trial, params), result in zip(missing, evaluated):
                fixed._write_json(_result_path(root, trial.number), _trial_result(trial.number, params, result))
        for trial in running:
            record = json.loads(_result_path(root, trial.number).read_text())
            if record["parameters"] != validate_parameters(trial.params):
                raise RescueError(f"trial-result parameter mismatch for {trial.number}")
            study.tell(trial.number, float(record["objective"]))
        print(f"RESCUE_OPTUNA_COMPLETE={len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{TRIAL_CAP}", flush=True)
    records = []
    for trial in study.get_trials(deepcopy=False):
        path = _result_path(root, trial.number)
        if trial.state != optuna.trial.TrialState.COMPLETE or not path.is_file():
            raise RescueError(f"trial {trial.number} is not durably complete")
        record = json.loads(path.read_text())
        if not math.isclose(float(record["objective"]), float(trial.value), abs_tol=1e-12):
            raise RescueError(f"objective differs from persistent Optuna study: {trial.number}")
        records.append(record)
    if len(records) != TRIAL_CAP:
        raise RescueError(f"expected {TRIAL_CAP} complete trials, got {len(records)}")
    fixed._write_json(root / "optuna-trials.json", {"study_name": STUDY_NAME, "complete": len(records),
                                                   "failed": 0, "trials": records})
    return records


def _region_key(parameters: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(parameters["pressure_window_ms"]),
            "P_LOW" if parameters["pressure_percentile"] < .90 else "P_MID" if parameters["pressure_percentile"] < .94 else "P_HIGH",
            "D_LOW" if parameters["depth_depletion_threshold"] <= .35 else "D_MID" if parameters["depth_depletion_threshold"] <= .50 else "D_HIGH",
            "R_LOW" if parameters["refill_weakness_threshold"] <= .45 else "R_MID" if parameters["refill_weakness_threshold"] <= .65 else "R_HIGH",
            "PERSIST_LOW" if parameters["flow_persistence_threshold"] <= .70 else "PERSIST_HIGH",
            "CHASE_TIGHT" if parameters["max_favorable_move_before_entry_ticks"] <= 4 else "CHASE_WIDE")


def _plateaus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [r for r in records if r["hard_screen"]["pass"]]
    ranked = sorted(records, key=lambda r: (-float(r["objective"]), int(r["number"])))
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in passed:
        groups[_region_key(row["parameters"])].append(row)
    clusters = []
    for key, members in groups.items():
        parameters = {name: [float(min(r["parameters"][name] for r in members)),
                             float(max(r["parameters"][name] for r in members))] for name in SEARCH_SPACE}
        metrics = [r["metrics"] for r in members]
        clusters.append({"region": list(key), "trial_count": len(members), "parameter_ranges": parameters,
                         "trial_numbers": sorted(r["number"] for r in members),
                         "median_objective": float(np.median([r["objective"] for r in members])),
                         "median_total_R": float(np.median([m["net_R"] for m in metrics])),
                         "median_avg_R": float(np.median([m["mean_trade_R"] for m in metrics])),
                         "median_PF": float(np.median([m["PF"] for m in metrics])),
                         "median_DD": float(np.median([m["max_drawdown_R"] for m in metrics])),
                         "median_trade_count": float(np.median([m["trades"] for m in metrics])),
                         "positive_day_fraction": float(np.median([m["positive_day_fraction"] for m in metrics])),
                         "positive_week_fraction": float(np.median([m["positive_week_fraction"] for m in metrics]))})
    clusters.sort(key=lambda c: (-c["trial_count"], -c["median_objective"], c["region"]))
    return {"passed_spring_hard_screen": len(passed), "best_trial": ranked[0],
            "clusters": clusters, "top_50": ranked[:50], "top_100": ranked[:100]}


def immediate_neighbors(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    center = validate_parameters(parameters)
    result: list[dict[str, Any]] = []
    for name, values in SEARCH_SPACE.items():
        position = next(i for i, value in enumerate(values) if math.isclose(float(center[name]), float(value), abs_tol=1e-9))
        for adjacent in (position-1, position+1):
            if 0 <= adjacent < len(values):
                candidate = dict(center); candidate[name] = values[adjacent]
                result.append(candidate)
    return result


def classify_neighbors(center: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]) -> str:
    if not neighbors:
        return "UNSTABLE"
    pass_fraction = sum(r["hard_screen"]["pass"] for r in neighbors) / len(neighbors)
    positive_fraction = sum(r["metrics"]["net_R"] > 0 for r in neighbors) / len(neighbors)
    median_objective = float(np.median([r["objective"] for r in neighbors]))
    if pass_fraction >= .60 and positive_fraction >= .75 and median_objective >= float(center["objective"])-.25:
        return "ROBUST_PLATEAU"
    if pass_fraction >= .40 and positive_fraction >= .60 and median_objective >= float(center["objective"])-.50:
        return "MODERATE_PLATEAU"
    if positive_fraction > 0:
        return "NEEDLE"
    return "UNSTABLE"


def _neighbor_stability(root: Path, records: Sequence[Mapping[str, Any]],
                        sources: Mapping[str, Mapping[str, Any]],
                        catalog: Mapping[tuple[str, int, float], float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = sorted((r for r in records if r["hard_screen"]["pass"]),
                      key=lambda r: (-float(r["objective"]), int(r["number"])))
    centers = []
    seen_regions = set()
    for row in accepted:
        region = _region_key(row["parameters"])
        if region not in seen_regions:
            centers.append(row); seen_regions.add(region)
        if len(centers) >= 3:
            break
    reports = []
    for center in centers:
        configs = immediate_neighbors(center["parameters"])
        evaluated = _evaluate_configs(configs, fixed.SPRING_DATES, sources, root, catalog)
        rows = [_trial_result(-1, config, result) for config, result in zip(configs, evaluated)]
        label = classify_neighbors(center, rows)
        reports.append({"center_trial": center["number"], "classification": label,
                        "neighbor_count": len(rows), "neighbors": rows,
                        "hard_screen_pass_fraction": sum(r["hard_screen"]["pass"] for r in rows) / len(rows),
                        "positive_fraction": sum(r["metrics"]["net_R"] > 0 for r in rows) / len(rows)})
    return reports, [r for r in centers if any(n["center_trial"] == r["number"] and
                  n["classification"] in ("ROBUST_PLATEAU", "MODERATE_PLATEAU") for n in reports)]


def _summarize_diagnostics(result: Mapping[str, Any], dates: Sequence[str]) -> dict[str, Any]:
    payloads = result["date_payloads"]
    trades = sorted((trade for p in payloads for trade in p["trades"]), key=lambda t: int(t["entry_time_ns"]))
    daily, weekly = fixed._daily_weekly(trades, dates)
    markouts, mfe_mae, first_touch = fixed._diagnostics(trades)
    direction = {side: fixed._summarize_trades([t for t in trades if t["entry_direction"] == side]) for side in ("LONG", "SHORT")}
    attrition = {key: sum(int(p["attrition"][key]) for p in payloads) for key in (
        "pressure_events", "clustered_events", "after_depletion", "after_refill",
        "after_persistence", "after_max_chase", "after_price_confirmation", "actual_entries")}
    flow = {}
    for horizon in fixed.HORIZONS_MS:
        h = str(horizon)
        rows = [(p["flow_only"]["markouts"][h]["sample_count"],
                 p["flow_only"]["markouts"][h]["mean_ticks"]) for p in payloads]
        count = sum(n for n, _ in rows)
        flow[h] = {"event_count": attrition["clustered_events"], "sample_count": count,
                   "mean_ticks": sum(n * mean for n, mean in rows if mean is not None) / count if count else None}
    return {"performance": fixed._summarize_trades(trades), "daily": daily, "weekly": weekly,
            "positive_days": sum(r["R"] > 0 for r in daily), "negative_days": sum(r["R"] < 0 for r in daily),
            "positive_weeks": sum(r["R"] > 0 for r in weekly), "negative_weeks": sum(r["R"] < 0 for r in weekly),
            "direction": direction, "markouts": markouts, "flow_only": flow,
            "mfe_mae": mfe_mae, "first_touch": first_touch, "attrition": attrition}


def _vacuum_identity(parameters: Mapping[str, Any], attrition: Mapping[str, int]) -> dict[str, Any]:
    flags = {
        "low_depletion": parameters["depth_depletion_threshold"] <= .25,
        "weak_refill_requirement": parameters["refill_weakness_threshold"] <= .35,
        "low_persistence": parameters["flow_persistence_threshold"] <= .60,
        "wide_chase_allowance": parameters["max_favorable_move_before_entry_ticks"] >= 8,
    }
    pass_fraction = (attrition["after_price_confirmation"] / attrition["clustered_events"]
                     if attrition["clustered_events"] else 0.0)
    return {"flags": flags, "confirmation_fraction_of_clustered": pass_fraction,
            "collapsed": sum(flags.values()) >= 3 and pass_fraction >= .50}


def _flow_comparison(spring: Mapping[str, Any], october: Mapping[str, Any] | None) -> dict[str, Any]:
    periods = {"SPRING": spring}
    if october is not None:
        periods["OCTOBER_DEV"] = october
    rows = {}
    for period, data in periods.items():
        flow = data["flow_only"]; vacuum = data["markouts"]["all"]
        rows[period] = {"flow_only_events": data["attrition"]["clustered_events"],
                        "vacuum_entries": data["performance"]["trade_count"],
                        "flow_only_markouts": {h: flow[h]["mean_ticks"] for h in ("500", "1000", "2000", "5000", "10000", "30000")},
                        "vacuum_entry_markouts": {h: vacuum[h]["mean_ticks"] for h in ("500", "1000", "2000", "5000", "10000", "30000")},
                        "incremental_ticks": {h: (vacuum[h]["mean_ticks"] - flow[h]["mean_ticks"]
                                                 if vacuum[h]["mean_ticks"] is not None and flow[h]["mean_ticks"] is not None else None)
                                              for h in ("500", "1000", "2000", "5000", "10000", "30000")}}
    material = all(data["incremental_ticks"][h] is not None and data["incremental_ticks"][h] > 0
                   for data in rows.values() for h in ("500", "1000")) and len(rows) == 2
    return {"periods": rows, "vacuum_incremental_value": "MATERIAL" if material else "VACUUM_COMPLEXITY_NOT_JUSTIFIED",
            "comparison_note": "Flow-only is event-mid anchored; vacuum is executable-entry anchored, matching fixed V1 conventions."}


def _decision(frozen: Sequence[Mapping[str, Any]], spring_by_number: Mapping[int, Mapping[str, Any]],
              october_by_number: Mapping[int, Mapping[str, Any]],
              flow_by_number: Mapping[int, Mapping[str, Any]],
              identity_by_number: Mapping[int, Mapping[str, Any]]) -> tuple[str, str]:
    if not frozen:
        return "LIQUIDITY_VACUUM_RESCUE_FAILED", "BUILD_FLOW_MOMENTUM_ONLY_V1"
    for row in frozen:
        number = int(row["number"])
        spring = spring_by_number[number]; october = october_by_number.get(number)
        if october is None:
            continue
        op = october["performance"]
        oct_compatible = (op["net_R"] > 0 and op["average_R"] is not None and op["average_R"] > 0
                          and isinstance(op["profit_factor"], (int, float)) and op["profit_factor"] >= 1.0)
        if (oct_compatible and not identity_by_number[number]["collapsed"]
                and flow_by_number[number]["vacuum_incremental_value"] == "MATERIAL"):
            return "LIQUIDITY_VACUUM_RESCUED", "FREEZE_CONFIG_AND_PLAN_ONE_FRESH_CALIBRATION_BLOCK"
    if any(identity_by_number[int(r["number"])]["collapsed"] for r in frozen):
        return "VACUUM_COLLAPSES_TO_FLOW_MOMENTUM", "BUILD_FLOW_MOMENTUM_ONLY_V1"
    return "LIQUIDITY_VACUUM_NOT_ROBUST", "STOP_OPTIMIZING_VACUUM"


def _prepare_october(root: Path, sources: Mapping[str, Mapping[str, Any]], windows: Sequence[int]) -> None:
    for day in (fixed.DEPENDENCY_DATES[1], *fixed.OCTOBER_DATES):
        rows = _compact(day, sources[day], root)
        for window in windows:
            _sample_pressure(day, int(window), rows, root)
        del rows
        print(f"RESCUE_OCTOBER_CACHE_READY={day}", flush=True)


def run(*, output_root: Path = OUT_ROOT, data_root: Path = fixed.DATA_ROOT) -> dict[str, Any]:
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    prereg_path = output_root / "optimization-preregistration.json"
    _write_once(prereg_path, PREREG)
    fixed_summary, fixed_trades = _read_fixed()
    phase0_days = (fixed.DEPENDENCY_DATES[0], *fixed.SPRING_DATES, *fixed.OCTOBER_DATES)
    sources = _source_rows(phase0_days, data_root)
    phase0_path = output_root / "entry-chase-diagnostic.json"
    phase0 = json.loads(phase0_path.read_text()) if phase0_path.exists() else _prepare_phase0(output_root, sources, fixed_trades)
    spring_catalog = _threshold_catalog(output_root, (fixed.DEPENDENCY_DATES[0], *fixed.SPRING_DATES),
                                        fixed.SPRING_DATES, SEARCH_SPACE["pressure_window_ms"])
    records = _run_optuna(output_root, sources, spring_catalog)
    plateau = _plateaus(records)
    neighbor_reports, selected = _neighbor_stability(output_root, records, sources, spring_catalog)
    fixed._write_json(output_root / "spring-plateau-analysis.json", plateau)
    fixed._write_json(output_root / "spring-neighbor-stability.json", {"evaluated_centers": neighbor_reports})
    ranked = sorted(records, key=lambda r: (-float(r["objective"]), int(r["number"])))
    best = ranked[0]
    spring_payload = {"best_trial": best, "hard_screen_pass_count": plateau["passed_spring_hard_screen"],
                      "selected_trial_numbers": [r["number"] for r in selected],
                      "complete_trials": len(records), "optimization_scope": "SPRING_2025_ONLY"}
    fixed._write_json(output_root / "spring-results.json", spring_payload)
    freeze_payload = {"status": "FROZEN", "selection_source": "SPRING_ONLY", "prereg_sha256": fixed._sha(prereg_path),
                      "trials_sha256_at_freeze": fixed._sha(output_root / "optuna-trials.json"),
                      "candidates": [{"number": r["number"], "parameters": r["parameters"],
                                      "spring_metrics": r["metrics"],
                                      "neighbor_classification": next(n["classification"] for n in neighbor_reports if n["center_trial"] == r["number"])}
                                     for r in selected[:3]]}
    freeze_path = output_root / "frozen-october-candidates.json"
    _write_once(freeze_path, freeze_payload)
    _write_once(output_root / "frozen-october-candidates.sha256.json", {"sha256": fixed._sha(freeze_path)})
    frozen = json.loads(freeze_path.read_text())["candidates"]
    if fixed._sha(freeze_path) != json.loads((output_root / "frozen-october-candidates.sha256.json").read_text())["sha256"]:
        raise RescueError("October candidate freeze hash mismatch")
    diagnostic_configs = [best["parameters"]]
    for candidate in frozen:
        if candidate["parameters"] not in diagnostic_configs:
            diagnostic_configs.append(candidate["parameters"])
    spring_details = _evaluate_configs(diagnostic_configs, fixed.SPRING_DATES, sources, output_root, spring_catalog,
                                       diagnostics=True)
    spring_by_config = {_canonical_hash(c): _summarize_diagnostics(result, fixed.SPRING_DATES)
                        for c, result in zip(diagnostic_configs, spring_details)}
    spring_by_number = {r["number"]: spring_by_config[_canonical_hash(r["parameters"])] for r in [best, *selected]}
    october_by_number: dict[int, dict[str, Any]] = {}
    if frozen:
        oct_path = output_root / "october-dev-results.json"
        if oct_path.exists():
            existing = json.loads(oct_path.read_text())
            if existing.get("freeze_sha256") != fixed._sha(freeze_path):
                raise RescueError("existing October dev result has different candidate freeze")
            october_by_number = {int(k): v for k, v in existing["candidates"].items()}
        else:
            october_sources = _source_rows((fixed.DEPENDENCY_DATES[1], *fixed.OCTOBER_DATES), data_root)
            sources.update(october_sources)
            windows = sorted({int(r["parameters"]["pressure_window_ms"]) for r in frozen})
            _prepare_october(output_root, sources, windows)
            october_catalog = _threshold_catalog(output_root, fixed.ALL_SOURCE_DATES,
                                                 fixed.OCTOBER_DATES, windows)
            results = _evaluate_configs([r["parameters"] for r in frozen], fixed.OCTOBER_DATES,
                                        sources, output_root, october_catalog, diagnostics=True)
            october_by_number = {int(r["number"]): _summarize_diagnostics(result, fixed.OCTOBER_DATES)
                                 for r, result in zip(frozen, results)}
            fixed._write_json(oct_path, {"status": "COMPLETE", "role": "SECONDARY_DEV_CHECK",
                                         "freeze_sha256": fixed._sha(freeze_path),
                                         "candidates": {str(k): v for k, v in october_by_number.items()}})
    else:
        _write_once(output_root / "october-dev-results.json",
                    {"status": "SKIPPED_NO_SPRING_PLATEAU", "role": "SECONDARY_DEV_CHECK",
                     "freeze_sha256": fixed._sha(freeze_path), "candidates": {}})
    flow_by_number: dict[int, dict[str, Any]] = {}
    identity_by_number: dict[int, dict[str, Any]] = {}
    for r in [best, *selected]:
        number = int(r["number"])
        spring = spring_by_number[number]; october = october_by_number.get(number)
        flow_by_number[number] = _flow_comparison(spring, october)
        identity_by_number[number] = _vacuum_identity(r["parameters"], spring["attrition"])
    decision, next_step = _decision(frozen, spring_by_number, october_by_number, flow_by_number, identity_by_number)
    fixed_attrition = fixed_summary["component_attrition"]
    fixed._write_json(output_root / "flow-only-comparison.json", {"by_trial_number": {str(k): v for k,v in flow_by_number.items()},
                                                            "fixed_v1_flow_only": fixed_summary["flow_only_baseline"]})
    fixed._write_json(output_root / "component-attrition.json", {"fixed_v1": fixed_attrition,
                       "best_spring": spring_by_number[best["number"]]["attrition"],
                       "frozen": {str(r["number"]): {"spring": spring_by_number[int(r["number"])]["attrition"],
                                                         "october_dev": october_by_number.get(int(r["number"]), {}).get("attrition")}
                                  for r in frozen}})
    fixed._write_json(output_root / "vacuum-identity-check.json", {"by_trial_number": {str(k): v for k,v in identity_by_number.items()}})
    for name, key in (("direction-results.json", "direction"), ("daily-results.json", "daily"),
                      ("weekly-results.json", "weekly"), ("markouts.json", "markouts"),
                      ("mfe-mae.json", "mfe_mae")):
        fixed._write_json(output_root / name, {"best_spring": spring_by_number[best["number"]][key],
                                             "frozen_spring": {str(r["number"]): spring_by_number[int(r["number"])][key] for r in frozen},
                                             "frozen_october_dev": {str(k): v[key] for k, v in october_by_number.items()}})
    best_number = int(frozen[0]["number"]) if frozen else int(best["number"])
    best_flow = flow_by_number[best_number]
    executable = (spring_by_number[best_number]["markouts"]["all"]["500"]["mean_ticks"] or 0) > 0
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "primary_decision": decision, "next_step": next_step,
               "fixed_v1_reference": {"trades": 520, "net_R": fixed_summary["performance"]["net_R"],
                                      "PF": fixed_summary["performance"]["profit_factor"]},
               "entry_chase_diagnostic": phase0, "optuna_trials_requested": TRIAL_CAP,
               "optuna_trials_completed": len(records), "optimization_data": "SPRING_2025_ONLY",
               "october_used_in_objective": False, "search_space": PREREG["search_space"],
               "spring_best_trial": best, "spring_robust_plateaus": plateau["clusters"],
               "frozen_october_candidates": frozen, "october_results": october_by_number,
               "best_candidate": {"config": next((r["parameters"] for r in frozen if int(r["number"]) == best_number), best["parameters"]),
                                  "spring": spring_by_number[best_number], "october": october_by_number.get(best_number)},
               "flow_only_comparison": best_flow, "vacuum_incremental_value": best_flow["vacuum_incremental_value"],
               "vacuum_collapsed_to_flow_momentum": any(v["collapsed"] for v in identity_by_number.values()),
               "executable_at_500ms_plus": bool(executable), "trial_cap_exceeded": False,
               "october_optimized": False, "final_oos_accessed": False,
               "new_features_added": False, "levels_added": False, "data_downloaded": False,
               "elapsed_seconds": time.monotonic()-started}
    fixed._write_json(output_root / "summary.json", summary)
    manifest = {"status": "COMPLETE", "run_id": RUN_ID, "prereg_sha256": fixed._sha(prereg_path),
                "freeze_sha256": fixed._sha(freeze_path), "study_sha256": fixed._sha(output_root / "optuna-study.db"),
                "source_sha256_by_date": {d: row["sha256"] for d,row in sources.items()},
                "fixed_v1_config_sha256": fixed.CONFIG_SHA256, "spring_dates": list(fixed.SPRING_DATES),
                "october_dev_dates": list(fixed.OCTOBER_DATES) if frozen else [],
                "no_2026_access": True, "no_downloads": True}
    fixed._write_json(output_root / "run-manifest.json", manifest)
    report = [f"# {RUN_ID}", "", f"Decision: **{decision}**", "",
              f"Optuna: {len(records)}/{TRIAL_CAP} complete Spring-only trials; October role: secondary DEV check.",
              f"Fixed V1: 520 trades, {fixed_summary['performance']['net_R']:.4f} net R, PF {fixed_summary['performance']['profit_factor']:.4f}.",
              f"Chase diagnostic: {phase0['classification']}.",
              f"Spring hard-screen passes: {plateau['passed_spring_hard_screen']}; frozen October candidates: {len(frozen)}.",
              f"Best Spring trial #{best['number']}: objective {best['objective']:.4f}, net R {best['metrics']['net_R']:.4f}, trades {best['metrics']['trades']}.",
              f"Flow comparison: {best_flow['vacuum_incremental_value']}; 500ms executable continuation positive: {executable}.",
              "", f"Next step: {next_step}", ""]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    hashes = {p.name: fixed._sha(p) for p in output_root.iterdir() if p.is_file() and p.name != "artifact-hashes.json"}
    fixed._write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files": hashes})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=fixed.DATA_ROOT)
    args = parser.parse_args(argv)
    result = run(output_root=args.output_root, data_root=args.data_root)
    print(json.dumps({"decision": result["primary_decision"], "completed": result["optuna_trials_completed"],
                      "frozen_candidates": len(result["frozen_october_candidates"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
