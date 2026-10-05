"""Frozen native-ES liquidity-vacuum continuation prototype for MAC 2025.

This is a one-pass-per-date, non-optimizing strategy study.  It uses only the
sealed native ES MBP-10 archives and their source-bound canonical quote tapes.
The top-five flow accounting is delegated to the repository's validated
price-keyed MLOFI implementation.
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
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import mac_2025_absorption_relative_normalization as relative
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_native_mbp_quote as plan
from .mac_2025_candidate_tape import TAPE_VERSION
from .model import ES_COMMISSION, ES_POINT_VALUE, ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS, size_for_instrument

RUN_ID = "CMEOrderflow_ES_LIQUIDITY_VACUUM_STRATEGY_V1_FIXED"
OUT_ROOT = Path("research_runs") / RUN_ID
DATA_ROOT = baseline.DATA_ROOT
TRAIN_TAPE_ROOT = baseline.OUTPUT_ROOT / "candidate-tapes" / "tapes"
OCT_TAPE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928/october-candidate-tapes/tapes")
DEPENDENCY_DATES = ("2025-02-28", "2025-10-06")
SPRING_DATES = tuple(baseline.TRAIN_DATES)
OCTOBER_DATES = tuple(d.isoformat() for d in plan.validation_dates())
ALL_TARGET_DATES = SPRING_DATES + OCTOBER_DATES
ALL_SOURCE_DATES = (DEPENDENCY_DATES[0], *SPRING_DATES, DEPENDENCY_DATES[1], *OCTOBER_DATES)
CONFIG = {
    "pressure_window_ms": 500, "pressure_percentile": 90,
    "pressure_metric": "absolute directional top-5 inverse-rank price-keyed MLOFI sum / contemporaneous weighted mean top-5 resting depth",
    "pressure_history": "equal-date-weighted deterministic systematic sample of prior source-date observations only; cumulative chronology, including Feb-28 and Oct-06 dependencies",
    "event_refractory_seconds": 2, "baseline_depth_seconds": 5,
    "baseline_depth_sampling_ms": 50,
    "minimum_depletion": 0.50, "refill_observation_ms": 500,
    "maximum_recovery_ratio": 0.50, "mlofi_persistence_window_ms": 1000,
    "mlofi_bins": 4, "mlofi_bin_ms": 250, "minimum_persistence": 0.75,
    "price_confirmation_min_ticks": 0, "entry_delay_ms": 2.0,
    "stop_ticks": 6, "target_ticks": 12, "target_r": 2.0,
    "max_hold_seconds": 30, "post_exit_refractory_seconds": 2,
    "history_sample_per_date": 100_000, "source_schema": "mbp-10", "instrument": "ES",
    "no_levels": True, "no_mbo": True, "no_mes_market_data": True,
}
CONFIG_SHA256 = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
STUDY_SPEC = {
    "version": 1, "config_sha256": CONFIG_SHA256,
    "source_scope": "2025 Spring TRAIN + October weekdays + only Feb-28 and Oct-06 percentile dependencies",
    "pressure": "validated TOP5 INVERSE_LEVEL price-keyed contribution; 500ms rolling sum divided by current denominator",
    "event": "every signed pressure observation at/above prior-date-only q90; first event, then 2s refractory clustering",
    "depletion": "opposing top5 depth against the median of 100 causal 50ms as-of depth samples in [t-5s,t); at least 50% depleted",
    "refill": "minimum opposing depth on [t,t+500ms]; first executable state at/after 500ms; recovery ratio <=0.5",
    "persistence": "directional normalized top5 MLOFI in four consecutive post-event 250ms bins; 3/4 agree",
    "signal": "first state at/after t+1s after full refill and persistence are known; direction-normalized mid change from event start >=0",
    "execution": "historical runner adverse entry offset one tick beyond ask/bid; 2ms then first executable canonical ES quote; stop/target trigger on exit-side BBO; adverse one-tick exit; stop wins same-event ambiguity; $50/point, $3/side; existing risk-budget contract sizing, MES fallback economics only if ES sizing is zero",
    "exits": "6-tick stop, 12-tick target, 30s first executable time exit; one position, no overlap, 2s post-exit refractory",
}
STUDY_SHA256 = hashlib.sha256(json.dumps(STUDY_SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
HORIZONS_MS = (250, 500, 1_000, 2_000, 5_000, 10_000, 30_000)
TICK = 0.25
HISTORY_SAMPLE = int(CONFIG["history_sample_per_date"])
CHECKPOINT_VERSION = "liquidity-vacuum-date-checkpoint-v1"


class VacuumStudyError(RuntimeError):
    """Frozen source, causal, checkpoint, or execution contract violation."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default).encode() + b"\n")
    os.replace(temp, path)


def _expected_contract(day: str) -> str:
    return plan.contract_for(date.fromisoformat(day))[0]


def _source_catalog(data_root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    manifest_path = data_root / baseline.MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VacuumStudyError(f"unreadable native-ES acquisition manifest: {manifest_path}") from exc
    requests = manifest.get("requests")
    if manifest.get("status") != "COMPLETE" or not isinstance(requests, dict):
        raise VacuumStudyError("native-ES acquisition manifest is not COMPLETE")
    rows: dict[str, dict[str, Any]] = {}
    for row in requests.values():
        if not isinstance(row, dict) or row.get("schema") != "mbp-10":
            continue
        day = str(row.get("session_date", ""))
        category = str(row.get("category", ""))
        if (day in SPRING_DATES and category == "TRAIN") or (day in OCTOBER_DATES and category == "VALIDATION") or (day in DEPENDENCY_DATES and category == "DEPENDENCY"):
            if day in rows:
                raise VacuumStudyError(f"duplicate native ES manifest row: {day}")
            rows[day] = row
    if set(rows) != set(ALL_SOURCE_DATES):
        raise VacuumStudyError(f"native ES source date mismatch; missing={sorted(set(ALL_SOURCE_DATES)-set(rows))}, extra={sorted(set(rows)-set(ALL_SOURCE_DATES))}")
    paths: dict[str, Path] = {}
    for day in ALL_SOURCE_DATES:
        row = rows[day]
        path = data_root / str(row.get("path", ""))
        if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size != int(row.get("bytes", -1)):
            raise VacuumStudyError(f"missing, empty, or size-mismatched native source: {path}")
        if row.get("symbol") != _expected_contract(day):
            raise VacuumStudyError(f"contract mismatch for {day}: manifest={row.get('symbol')} expected={_expected_contract(day)}")
        actual = _sha(path)
        if actual != row.get("sha256"):
            raise VacuumStudyError(f"native source hash mismatch: {path}")
        from databento import DBNStore
        metadata = DBNStore.from_file(path).metadata
        if metadata.dataset != "GLBX.MDP3" or metadata.schema != "mbp-10" or _expected_contract(day) not in metadata.symbols:
            raise VacuumStudyError(f"native DBN header identity mismatch for {day}: {metadata}")
        paths[day] = path
    return paths, rows


def _tape_path(day: str) -> Path:
    return (TRAIN_TAPE_ROOT if day in SPRING_DATES else OCT_TAPE_ROOT) / f"{day}-candidate-tape.npz"


def _load_tape(day: str, path: Path, source_sha: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise VacuumStudyError(f"missing sealed canonical ES event path: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        events = np.asarray(archive["events"])
    if metadata.get("date") != day or metadata.get("source_sha256") != source_sha:
        raise VacuumStudyError(f"candidate tape source/date mismatch: {day}")
    if metadata.get("semantic_sha256") != relative.EXPECTED_TAPE_SEMANTIC_SHA or metadata.get("tape_version") != TAPE_VERSION:
        raise VacuumStudyError(f"candidate tape semantics/version mismatch: {day}")
    if len(events):
        if np.any(np.diff(events["timestamp_ns"].astype(np.int64, copy=False)) < 0):
            raise VacuumStudyError(f"unordered canonical event path: {day}")
        if np.any(np.diff(events["session"].astype(np.int8, copy=False)) < 0):
            raise VacuumStudyError(f"canonical event session ordering is invalid: {day}")
    return events, metadata


def _systematic_sample(values: np.ndarray, cap: int = HISTORY_SAMPLE) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) > cap:
        # Fixed phase avoids random sampling and is stable across resumes.
        indices = np.linspace(0, len(values) - 1, num=cap, dtype=np.int64)
        values = values[indices]
    return values.tolist()


def rolling_pressure(rows: np.ndarray, window_ns: int = 500_000_000) -> np.ndarray:
    """Prior/current-inclusive rolling price-keyed TOP5 normalized MLOFI."""
    ts = np.asarray(rows["ts"], dtype=np.int64)
    flow = np.asarray(rows["mlofi"], dtype=np.float64)
    denom = np.asarray(rows["denom"], dtype=np.float64)
    if not len(ts):
        return np.empty(0, dtype=np.float64)
    prefix = np.empty(len(flow) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(flow, out=prefix[1:])
    left = np.searchsorted(ts, ts - int(window_ns), side="left")
    sums = prefix[1:] - prefix[left]
    return np.divide(sums, denom, out=np.zeros_like(sums), where=denom > 0)


def _session_bounds(day: str, session_code: int) -> tuple[int, int]:
    name = ("ASIA", "EUROPE", "NY")[int(session_code)]
    return baseline._session_windows(day)[name]


def _first_at_or_after(ts: np.ndarray, target: int, lo: int, hi: int) -> int | None:
    index = max(lo, int(np.searchsorted(ts, target, side="left")))
    return index if index < hi else None


def _depth_baseline_grid(rows: np.ndarray, day: str,
                         timestamps: np.ndarray | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """50ms as-of depth series, matching the repository's validated grid cadence."""
    ts = timestamps if timestamps is not None else np.ascontiguousarray(rows["ts"], dtype=np.int64)
    result = {}
    step = int(CONFIG["baseline_depth_sampling_ms"]) * 1_000_000
    for code, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = baseline._session_windows(day)[name]
        grid = np.arange(start + step, end, step, dtype=np.int64)
        ids = np.searchsorted(ts, grid, side="right") - 1
        valid = (ids >= 0) & (ts[np.maximum(ids, 0)] >= start)
        bid_values = np.full(len(grid), np.nan, dtype=np.float64)
        bid_values[valid] = np.asarray(rows["bid5"])[ids[valid]]
        ask_values = np.full(len(grid), np.nan, dtype=np.float64)
        ask_values[valid] = np.asarray(rows["ask5"])[ids[valid]]
        # Store both sides; tuple shape remains compact and date/session local.
        result[code] = (grid, np.column_stack((bid_values, ask_values)))
    return result


def _prior_depth_median(grid_ns: np.ndarray, depth_grid: np.ndarray, event_ns: int,
                        session_start_ns: int, side_index: int) -> float | None:
    if event_ns - session_start_ns < 5_000_000_000:
        return None
    low = int(np.searchsorted(grid_ns, event_ns - 5_000_000_000, side="left"))
    high = int(np.searchsorted(grid_ns, event_ns, side="left"))
    values = np.asarray(depth_grid[low:high, side_index], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return None
    median = float(np.median(values))
    return median if math.isfinite(median) and median > 0 else None


def _path_markouts(events: np.ndarray, timestamp_ns: int, anchor_mid: float, direction: int,
                   session: int, horizons: Sequence[int] = HORIZONS_MS,
                   event_timestamps: np.ndarray | None = None,
                   event_sessions: np.ndarray | None = None,
                   session_bounds: Mapping[int, tuple[int, int]] | None = None) -> dict[str, float | None]:
    ts = event_timestamps if event_timestamps is not None else np.ascontiguousarray(events["timestamp_ns"], dtype=np.int64)
    session_arr = event_sessions if event_sessions is not None else np.ascontiguousarray(events["session"], dtype=np.int8)
    if session_bounds is not None:
        start, end = session_bounds[int(session)]
    else:
        start, end = _session_bounds_ns_for_code(session, events, sessions=session_arr)
    out: dict[str, float | None] = {}
    for horizon in horizons:
        ix = _first_at_or_after(ts, timestamp_ns + int(horizon) * 1_000_000, start, end)
        out[str(horizon)] = None if ix is None or session_arr[ix] != session else float(direction * (((float(events[ix]["bid"]) + float(events[ix]["ask"])) / 2 - anchor_mid) / TICK))
    return out


def cluster_pressure_events(timestamp_ns: Sequence[int], signed_pressure: Sequence[float],
                            session_codes: Sequence[int], threshold: float,
                            refractory_ns: int = 2_000_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Return raw threshold-hit indices and first-hit refractory clusters."""
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    pressure = np.asarray(signed_pressure, dtype=np.float64)
    sessions = np.asarray(session_codes, dtype=np.int8)
    if not (len(ts) == len(pressure) == len(sessions)):
        raise VacuumStudyError("pressure clustering arrays have different lengths")
    raw = np.flatnonzero((np.abs(pressure) >= float(threshold)) & (pressure != 0) & (sessions >= 0))
    clustered: list[int] = []
    last_by_session: dict[int, int] = {}
    for index in raw:
        code = int(sessions[index]); timestamp = int(ts[index])
        if timestamp - last_by_session.get(code, -10**30) >= int(refractory_ns):
            clustered.append(int(index)); last_by_session[code] = timestamp
    return raw, np.asarray(clustered, dtype=np.int64)


def _session_bounds_ns_for_code(session: int, events: np.ndarray,
                                sessions: np.ndarray | None = None) -> tuple[int, int]:
    if not len(events):
        return 0, 0
    sessions = sessions if sessions is not None else np.ascontiguousarray(events["session"], dtype=np.int8)
    # Candidate Tape V2 orders events by timestamp and labels the three
    # non-overlapping session windows monotonically 0, 1, 2.
    lo = int(np.searchsorted(sessions, int(session), side="left"))
    hi = int(np.searchsorted(sessions, int(session), side="right"))
    return lo, hi


def _flow_only_summary(events: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"event_count": len(candidates), "markouts": {}}
    for horizon in HORIZONS_MS:
        values = [row["markouts"].get(str(horizon)) for row in candidates]
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        result["markouts"][str(horizon)] = {"sample_count": len(finite), "mean_ticks": float(np.mean(finite)) if finite else None}
    return result


def _classify_event(rows: np.ndarray, event: Mapping[str, Any], *, day: str,
                    depth_grid: tuple[np.ndarray, np.ndarray] | None = None,
                    timestamps: np.ndarray | None = None,
                    minimum_depletion: float = 0.50,
                    maximum_recovery_ratio: float = 0.50,
                    minimum_persistence: float = 0.75) -> dict[str, Any]:
    """Apply depletion, full 500ms refill, 1s persistence, then price gate."""
    ts = timestamps if timestamps is not None else np.ascontiguousarray(rows["ts"], dtype=np.int64)
    i = int(event["row_index"]); t = int(event["timestamp_ns"]); sess = int(event["session"])
    windows = baseline._session_windows(day)
    name = ("ASIA", "EUROPE", "NY")[sess]
    session_lo, session_end = windows[name]
    right = int(np.searchsorted(ts, session_end, side="left"))
    side = "ask5" if int(event["direction"]) > 0 else "bid5"
    depth = np.asarray(rows[side], dtype=np.float64)
    depth_grid = depth_grid or _depth_baseline_grid(rows, day, timestamps=ts)[sess]
    grid_ns, grid_depth = depth_grid
    baseline_depth = _prior_depth_median(grid_ns, grid_depth, t, session_lo, 1 if side == "ask5" else 0)
    if baseline_depth is None:
        return {"after_depletion": False, "reject_reason": "INSUFFICIENT_PRIOR_DEPTH_HISTORY"}
    current_depth = float(depth[i])
    depletion = 1.0 - current_depth / baseline_depth if baseline_depth > 0 else None
    base = {"baseline_depth": baseline_depth, "event_depth": current_depth, "depletion": depletion}
    if depletion is None or depletion < minimum_depletion:
        return {**base, "after_depletion": False, "reject_reason": "DEPLETION_BELOW_THRESHOLD"}
    after500 = _first_at_or_after(ts, t + 500_000_000, i, right)
    if after500 is None:
        return {**base, "after_depletion": True, "after_refill": False, "reject_reason": "REFILL_WINDOW_CROSSES_SESSION_OR_DATA_END"}
    min_ix = int(i + np.argmin(depth[i:after500 + 1]))
    d_min = float(depth[min_ix]); d_after = float(depth[after500])
    denom = baseline_depth - d_min
    recovery = (d_after - d_min) / denom if denom > 0 else 1.0
    recovery = max(0.0, recovery)
    refill_weakness = 1.0 - recovery
    base.update({"d_min": d_min, "d_after_500ms": d_after, "recovery_ratio": recovery,
                 "refill_weakness": refill_weakness, "refill_observation_ns": int(ts[after500])})
    if recovery > maximum_recovery_ratio:
        return {**base, "after_depletion": True, "after_refill": False, "reject_reason": "REFILL_RECOVERED_OVER_THRESHOLD"}
    # Four post-event bins; each bin's own end-depth denominator produces the
    # same current-time normalization as the validated MLOFI implementation.
    mlofi = np.asarray(rows["mlofi"], dtype=np.float64)
    depth_denom = np.asarray(rows["denom"], dtype=np.float64)
    bins = []
    for bin_i in range(4):
        lo_t = t + bin_i * 250_000_000
        hi_t = t + (bin_i + 1) * 250_000_000
        lo = int(np.searchsorted(ts, lo_t, side="right"))
        hi = int(np.searchsorted(ts, hi_t, side="right"))
        if hi <= lo:
            bins.append(0.0)
        else:
            d = float(depth_denom[hi - 1])
            bins.append(float(np.sum(mlofi[lo:hi]) / d) if d > 0 else 0.0)
    agrees = sum(int(int(event["direction"]) * value > 0) for value in bins)
    persistence = agrees / 4.0
    base.update({"mlofi_bin_values": bins, "mlofi_agreeing_bins": agrees, "mlofi_persistence": persistence})
    if persistence < minimum_persistence:
        return {**base, "after_depletion": True, "after_refill": True, "after_persistence": False,
                "reject_reason": "MLOFI_PERSISTENCE_BELOW_THRESHOLD"}
    anchor_ix = _first_at_or_after(ts, t + 1_000_000_000, i, right)
    if anchor_ix is None:
        return {**base, "after_depletion": True, "after_refill": True, "after_persistence": True,
                "after_confirmation": False, "reject_reason": "CONFIRMATION_WINDOW_CROSSES_SESSION_OR_DATA_END"}
    event_mid = float(rows[i]["mid"]); anchor_mid = float(rows[anchor_ix]["mid"])
    signed_change_ticks = int(event["direction"]) * (anchor_mid - event_mid) / TICK
    base.update({"signal_anchor_ns": int(ts[anchor_ix]), "event_start_mid": event_mid,
                 "signal_anchor_mid": anchor_mid, "confirmation_change_ticks": signed_change_ticks})
    if signed_change_ticks < 0:
        return {**base, "after_depletion": True, "after_refill": True, "after_persistence": True,
                "after_confirmation": False, "reject_reason": "PRICE_REJECTED_EVENT_DIRECTION"}
    return {**base, "after_depletion": True, "after_refill": True, "after_persistence": True,
            "after_confirmation": True, "reject_reason": None}


def _entry_and_exit(events: np.ndarray, signal: Mapping[str, Any], busy_until: int,
                    event_timestamps: np.ndarray | None = None,
                    event_sessions: np.ndarray | None = None,
                    session_bounds: Mapping[int, tuple[int, int]] | None = None,
                    stop_ticks: int = 6, target_r: float = 2.0,
                    max_favorable_move_before_entry_ticks: float | None = None,
                    include_diagnostics: bool = True,
                    max_hold_seconds: int = 30) -> dict[str, Any] | None:
    direction = int(signal["direction"]); signal_ns = int(signal["signal_anchor_ns"])
    ready = signal_ns + 2_000_000
    if signal_ns < busy_until:
        return None
    ts = event_timestamps if event_timestamps is not None else np.ascontiguousarray(events["timestamp_ns"], dtype=np.int64)
    sessions = event_sessions if event_sessions is not None else np.ascontiguousarray(events["session"], dtype=np.int8)
    session = int(signal["session"])
    if session_bounds is not None:
        lo, hi = session_bounds[int(session)]
    else:
        lo, hi = _session_bounds_ns_for_code(session, events, sessions=sessions)
    if lo >= hi:
        return None
    entry_ix = _first_at_or_after(ts, ready, lo, hi)
    if entry_ix is None or int(events[entry_ix]["session"]) != session:
        return None
    entry_event = events[entry_ix]
    bid, ask = float(entry_event["bid"]), float(entry_event["ask"])
    if not (math.isfinite(bid) and math.isfinite(ask) and ask > bid):
        return None
    # Historical runner convention: pay one extra tick beyond the displayed
    # entry-side quote. Geometry is fixed from this executable entry fill.
    entry = ask + TICK if direction > 0 else bid - TICK
    event_mid = signal.get("event_start_mid")
    chase_ticks = (direction * (entry - float(event_mid)) / TICK) if event_mid is not None else None
    if max_favorable_move_before_entry_ticks is not None:
        if chase_ticks is None or chase_ticks > max_favorable_move_before_entry_ticks:
            return None
    target_ticks = int(math.floor(stop_ticks * target_r + 0.5))
    stop = entry - direction * stop_ticks * TICK
    target = entry + direction * target_ticks * TICK
    stop_exit = stop - direction * TICK
    prices = {"entry": entry, "stop_exit": stop_exit}
    sizing = size_for_instrument(prices, "ES"); instrument = "ES"
    if int(sizing["contracts"]) < 1:
        sizing = size_for_instrument(prices, "MES"); instrument = "MES"
    if int(sizing["contracts"]) < 1:
        return None
    hold_deadline = int(entry_event["timestamp_ns"]) + int(max_hold_seconds) * 1_000_000_000
    # Avoid opening if the full fixed holding interval cannot be observed
    # within the execution session.
    from datetime import datetime, timezone
    entry_day = datetime.fromtimestamp(int(entry_event["timestamp_ns"]) / 1e9, tz=timezone.utc).date().isoformat()
    session_name = ("ASIA", "EUROPE", "NY")[session]
    session_end = baseline._session_windows(entry_day)[session_name][1]
    if hold_deadline >= session_end:
        return None
    exit_ix = None; reason = None
    for j in range(entry_ix, hi):
        event = events[j]
        if int(event["session"]) != session:
            break
        t = int(event["timestamp_ns"])
        exit_ref = float(event["bid"] if direction > 0 else event["ask"])
        # Stop precedence is explicit if both conditions can appear on one
        # timestamp/gapped quote update.
        if direction * (exit_ref - stop) <= 0:
            exit_ix, reason = j, "STOP"
            break
        if direction * (exit_ref - target) >= 0:
            exit_ix, reason = j, "TARGET"
            break
        if t >= hold_deadline:
            exit_ix, reason = j, "TIME_EXIT"
            break
    if exit_ix is None:
        return None
    exit_event = events[exit_ix]
    exit_ref = float(exit_event["bid"] if direction > 0 else exit_event["ask"])
    exit_price = exit_ref - direction * TICK
    contracts = int(sizing["contracts"])
    point_value, commission = (ES_POINT_VALUE, ES_COMMISSION) if instrument == "ES" else (5.0, 1.25)
    gross_points = direction * (exit_price - entry)
    gross_usd = gross_points * point_value * contracts
    fees = 2 * commission * contracts
    initial_risk = abs(entry - stop_exit) * point_value * contracts + fees
    gross_r = gross_usd / initial_risk if initial_risk else 0.0
    net_r = (gross_usd - fees) / initial_risk if initial_risk else 0.0
    if not include_diagnostics:
        return {"date": entry_day, "entry_direction": "LONG" if direction > 0 else "SHORT",
                "entry_time_ns": int(entry_event["timestamp_ns"]),
                "exit_time_ns": int(exit_event["timestamp_ns"]), "entry_price": entry,
                "exit_price": exit_price, "exit_reason": reason, "net_R": net_r,
                "gross_R": gross_r, "hold_seconds": (int(exit_event["timestamp_ns"]) - int(entry_event["timestamp_ns"])) / 1e9,
                "favorable_move_before_entry_ticks": chase_ticks}
    path_start = entry_ix; path_end = exit_ix + 1
    refs = np.asarray([float(events[k]["bid"] if direction > 0 else events[k]["ask"]) for k in range(path_start, path_end)], dtype=np.float64)
    signed_excursions = direction * (refs - entry) / TICK
    markouts: dict[str, float | None] = {}
    for horizon in HORIZONS_MS:
        mx = _first_at_or_after(ts, int(entry_event["timestamp_ns"]) + horizon * 1_000_000, entry_ix, hi)
        if mx is None or int(events[mx]["session"]) != session:
            markouts[str(horizon)] = None
        else:
            ref = float(events[mx]["bid"] if direction > 0 else events[mx]["ask"])
            markouts[str(horizon)] = direction * (ref - entry) / TICK
    diagnostic_ix = _first_at_or_after(ts, int(entry_event["timestamp_ns"]) + 30_000_000_000, entry_ix, hi)
    diagnostic_end = diagnostic_ix + 1 if diagnostic_ix is not None else hi
    diagnostic_refs = np.asarray([float(events[k]["bid"] if direction > 0 else events[k]["ask"])
                                  for k in range(entry_ix, diagnostic_end)
                                  if int(events[k]["session"]) == session], dtype=np.float64)
    diagnostic_excursions = direction * (diagnostic_refs - entry) / TICK
    first_touch: dict[str, str] = {}
    for up, down in ((1, 1), (2, 2), (4, 4), (8, 4)):
        outcome = "NO_TOUCH"
        for value in diagnostic_excursions:
            hit_up, hit_down = value >= up, value <= -down
            if hit_up or hit_down:
                outcome = "TIE" if hit_up and hit_down else "UP" if hit_up else "DOWN"
                break
        first_touch[f"+{up}/-{down}"] = outcome
    return {
        "date": entry_day, "event_start_time_ns": int(signal["timestamp_ns"]),
        "signal_anchor_ns": signal_ns, "entry_time_ns": int(entry_event["timestamp_ns"]),
        "entry_price": entry, "entry_direction": "LONG" if direction > 0 else "SHORT",
        "event_start_mid": float(event_mid) if event_mid is not None else None,
        "favorable_move_before_entry_ticks": chase_ticks,
        "stop_ticks": int(stop_ticks), "target_ticks": target_ticks,
        "pressure": signal["pressure"], "pressure_percentile": signal["pressure_percentile"],
        "depletion": signal.get("depletion"), "recovery_ratio": signal.get("recovery_ratio"),
        "refill_weakness": signal.get("refill_weakness"), "mlofi_persistence": signal.get("mlofi_persistence"),
        "baseline_depth": signal.get("baseline_depth"), "event_depth": signal.get("event_depth"),
        "signal_confirmation_ticks": signal.get("confirmation_change_ticks"),
        "stop_price": stop, "target_price": target, "exit_time_ns": int(exit_event["timestamp_ns"]),
        "exit_price": exit_price, "exit_reason": reason, "hold_seconds": (int(exit_event["timestamp_ns"]) - int(entry_event["timestamp_ns"])) / 1e9,
        "instrument": instrument, "contracts": contracts, "gross_points": gross_points,
        "gross_R": gross_r, "net_R": net_r, "gross_pnl_usd": gross_usd,
        "commission_cost_usd": fees, "initial_risk_usd": initial_risk,
        "markouts_ticks": markouts,
        "mfe_ticks": float(max(0.0, np.max(signed_excursions))) if len(signed_excursions) else 0.0,
        "mae_ticks": float(max(0.0, -np.min(signed_excursions))) if len(signed_excursions) else 0.0,
        "first_touch": first_touch,
    }


def evaluate_day(day: str, rows: np.ndarray, tape_events: np.ndarray, threshold: float,
                 historical_pressure: Sequence[float], pressure: np.ndarray | None = None) -> dict[str, Any]:
    # numpy memmap views do not accept custom attrs: date passed via module-level
    # array metadata is handled in wrapper below.
    phase_start = time.perf_counter()
    pressure = rolling_pressure(rows) if pressure is None else pressure
    event_timestamps = np.ascontiguousarray(tape_events["timestamp_ns"], dtype=np.int64)
    event_sessions = np.ascontiguousarray(tape_events["session"], dtype=np.int8)
    event_session_bounds = {code: _session_bounds_ns_for_code(code, tape_events, sessions=event_sessions)
                            for code in (0, 1, 2)}
    history = np.asarray(historical_pressure, dtype=np.float64)
    # Date-bound session codes are generated without filtering/retiming events.
    ts = np.ascontiguousarray(rows["ts"], dtype=np.int64)
    sess_codes = np.full(len(ts), -1, dtype=np.int8)
    for code, name in enumerate(("ASIA", "EUROPE", "NY")):
        start, end = baseline._session_windows(day)[name]
        sess_codes[(ts >= start) & (ts < end)] = code
    raw_indices, cluster_indices = cluster_pressure_events(ts, pressure, sess_codes, threshold)
    print(f"VACUUM_EVENTS date={day} raw={len(raw_indices)} clustered={len(cluster_indices)} pressure_seconds={time.perf_counter()-phase_start:.1f}", flush=True)
    clustered: list[dict[str, Any]] = []
    sorted_history = np.sort(history) if len(history) else np.empty(0)
    for idx in cluster_indices:
        session = int(sess_codes[idx]); t = int(ts[idx])
        magnitude = abs(float(pressure[idx]))
        percentile = float(np.searchsorted(sorted_history, magnitude, side="right") / len(sorted_history)) if len(sorted_history) else None
        clustered.append({"row_index": int(idx), "timestamp_ns": t, "session": session,
                          "direction": 1 if pressure[idx] > 0 else -1,
                          "pressure": float(pressure[idx]), "pressure_percentile": percentile})

    flow_rows = []
    for event in clustered:
        idx = int(event["row_index"]); s = int(event["session"]); t = int(event["timestamp_ns"])
        anchor = float(rows[idx]["mid"])
        flow_rows.append({"timestamp_ns": t, "session": s,
                          "markouts": _path_markouts(tape_events, t, anchor, int(event["direction"]), s,
                                                     event_timestamps=event_timestamps,
                                                     event_sessions=event_sessions,
                                                     session_bounds=event_session_bounds)})
    print(f"VACUUM_FLOW_BASELINE_READY date={day} events={len(flow_rows)} seconds={time.perf_counter()-phase_start:.1f}", flush=True)

    attrition: dict[str, Any] = {"raw_pressure_events": int(len(raw_indices)), "clustered_events": len(clustered)}
    attrition_by_direction: dict[str, dict[str, int]] = {
        "1": {"raw_pressure_events": int(np.count_nonzero(pressure[raw_indices] > 0)), "clustered_events": sum(e["direction"] > 0 for e in clustered)},
        "-1": {"raw_pressure_events": int(np.count_nonzero(pressure[raw_indices] < 0)), "clustered_events": sum(e["direction"] < 0 for e in clustered)},
    }
    by_period_direction: dict[str, dict[str, int]] = {}
    passed: list[dict[str, Any]] = []
    baseline_grids = _depth_baseline_grid(rows, day, timestamps=ts)
    for event_index, event in enumerate(clustered, 1):
        state = _classify_event(rows, event, day=day, depth_grid=baseline_grids[int(event["session"])], timestamps=ts)
        for key in ("after_depletion", "after_refill", "after_persistence", "after_confirmation"):
            if state.get(key):
                pass
        if state.get("after_depletion"):
            attrition["after_depletion"] = attrition.get("after_depletion", 0) + 1
            attrition_by_direction[str(event["direction"])]["after_depletion"] = attrition_by_direction[str(event["direction"])].get("after_depletion", 0) + 1
        if state.get("after_refill"):
            attrition["after_refill"] = attrition.get("after_refill", 0) + 1
            attrition_by_direction[str(event["direction"])]["after_refill"] = attrition_by_direction[str(event["direction"])].get("after_refill", 0) + 1
        if state.get("after_persistence"):
            attrition["after_persistence"] = attrition.get("after_persistence", 0) + 1
            attrition_by_direction[str(event["direction"])]["after_persistence"] = attrition_by_direction[str(event["direction"])].get("after_persistence", 0) + 1
        if state.get("after_confirmation"):
            attrition["after_confirmation"] = attrition.get("after_confirmation", 0) + 1
            attrition_by_direction[str(event["direction"])]["after_confirmation"] = attrition_by_direction[str(event["direction"])].get("after_confirmation", 0) + 1
            passed.append({**event, **state})
        if event_index % 5_000 == 0:
            print(f"VACUUM_FILTER_PROGRESS date={day} processed={event_index}/{len(clustered)} seconds={time.perf_counter()-phase_start:.1f}", flush=True)
    trades: list[dict[str, Any]] = []
    busy_until = -10**30
    entry_rejections: dict[str, int] = {}
    for signal in passed:
        trade = _entry_and_exit(tape_events, signal, busy_until, event_timestamps, event_sessions,
                                event_session_bounds)
        if trade is None:
            reason = "OVERLAPPING_POSITION_OR_POST_EXIT_REFRACTORY" if int(signal["signal_anchor_ns"]) < busy_until else "NO_VALID_EXECUTABLE_PATH_OR_RISK_SIZE"
            entry_rejections[reason] = entry_rejections.get(reason, 0) + 1
            continue
        trades.append(trade)
        attrition_by_direction[str(signal["direction"])]["actual_entries"] = attrition_by_direction[str(signal["direction"])].get("actual_entries", 0) + 1
        busy_until = int(trade["exit_time_ns"]) + 2_000_000_000
    attrition["actual_entries"] = len(trades)
    attrition["entry_rejections"] = entry_rejections
    return {"date": day, "pressure_threshold": float(threshold), "pressure_observation_count": len(pressure),
            "pressure_history_sample": _systematic_sample(np.abs(pressure)),
            "raw_pressure_events": int(len(raw_indices)), "clustered_events": clustered,
            "flow_only": _flow_only_summary(tape_events, flow_rows), "trades": trades,
            "attrition": attrition, "attrition_by_direction": attrition_by_direction,
            "elapsed_seconds": None}


def _checkpoint_path(root: Path, day: str) -> Path:
    return root / "checkpoints" / f"{day}.json.gz"


def _read_checkpoint(path: Path, *, day: str, source_sha: str, tape_sha: str) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            item = json.load(stream)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    if (item.get("checkpoint_version") != CHECKPOINT_VERSION or item.get("date") != day
            or item.get("source_sha256") != source_sha or item.get("tape_sha256") != tape_sha
            or item.get("config_sha256") != CONFIG_SHA256 or item.get("study_sha256") != STUDY_SHA256
            or item.get("status") != "DATE_COMPLETE"):
        return None
    return item


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default).encode())
    os.replace(temp, path)


def _summarize_trades(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net = [float(row["net_R"]) for row in trades]
    gross = [float(row["gross_R"]) for row in trades]
    wins = sum(value > 0 for value in net); losses = sum(value < 0 for value in net)
    gross_profit = sum(value for value in net if value > 0)
    gross_loss = abs(sum(value for value in net if value < 0))
    running = peak = maxdd = 0.0; win_streak = loss_streak = max_win_streak = max_loss_streak = 0
    for row in trades:
        value = float(row["net_R"]); running += value; peak = max(peak, running); maxdd = max(maxdd, peak - running)
        if value > 0:
            win_streak += 1; loss_streak = 0
        elif value < 0:
            loss_streak += 1; win_streak = 0
        else:
            win_streak = loss_streak = 0
        max_win_streak = max(max_win_streak, win_streak); max_loss_streak = max(max_loss_streak, loss_streak)
    holds = [float(row["hold_seconds"]) for row in trades]
    counts = {reason: sum(row["exit_reason"] == reason for row in trades) for reason in ("TARGET", "STOP", "TIME_EXIT")}
    return {"trade_count": len(trades), "long_count": sum(row["entry_direction"] == "LONG" for row in trades),
            "short_count": sum(row["entry_direction"] == "SHORT" for row in trades), "wins": wins, "losses": losses,
            "time_exits": counts["TIME_EXIT"], "win_rate": wins / len(trades) if trades else None,
            "gross_R": float(sum(gross)), "net_R": float(sum(net)),
            "average_R": float(np.mean(net)) if net else None, "median_R": float(np.median(net)) if net else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else ("Infinity" if gross_profit else None),
            "max_drawdown_R": maxdd, "max_consecutive_losses": max_loss_streak,
            "max_consecutive_wins": max_win_streak, "average_hold_seconds": float(np.mean(holds)) if holds else None,
            "target_hit_rate": counts["TARGET"] / len(trades) if trades else None,
            "stop_hit_rate": counts["STOP"] / len(trades) if trades else None,
            "time_exit_rate": counts["TIME_EXIT"] / len(trades) if trades else None,
            "target_exits": counts["TARGET"], "stop_exits": counts["STOP"]}


def _series_results(trades: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output = {}
    for label in sorted({str(row[key]) for row in trades}):
        selected = [row for row in trades if str(row[key]) == label]
        output[label] = _summarize_trades(selected)
    return output


def _daily_weekly(trades: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datetime import datetime, timezone
    daily = []
    for day in dates:
        selected = [row for row in trades if row["date"] == day]
        s = _summarize_trades(selected)
        daily.append({"date": day, "trade_count": s["trade_count"], "wins": s["wins"], "losses": s["losses"],
                      "R": s["net_R"], "max_dd_R": s["max_drawdown_R"]})
    weeks: dict[str, list[Mapping[str, Any]]] = {}
    for day in dates:
        dt = datetime.fromisoformat(str(day)).replace(tzinfo=timezone.utc).date()
        iso = dt.isocalendar(); weeks.setdefault(f"{iso.year}-W{iso.week:02d}", [])
    for row in trades:
        dt = datetime.fromisoformat(str(row["date"])).replace(tzinfo=timezone.utc).date()
        iso = dt.isocalendar(); weeks.setdefault(f"{iso.year}-W{iso.week:02d}", []).append(row)
    weekly = []
    for week, selected in sorted(weeks.items()):
        s = _summarize_trades(selected)
        weekly.append({"week": week, "trades": s["trade_count"], "R": s["net_R"],
                       "profit_factor": s["profit_factor"], "max_dd": s["max_drawdown_R"]})
    return daily, weekly


def _markout_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS_MS:
        values = [float(row["markouts_ticks"][str(horizon)]) for row in rows
                  if row.get("markouts_ticks", {}).get(str(horizon)) is not None]
        output[str(horizon)] = {"n": len(values), "mean_ticks": float(np.mean(values)) if values else None}
    return output


def _diagnostics(trades: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    subsets = {"all": list(trades), "wins": [t for t in trades if t["net_R"] > 0],
               "losses": [t for t in trades if t["net_R"] < 0],
               "spring": [t for t in trades if t["date"] in SPRING_DATES],
               "october": [t for t in trades if t["date"] in OCTOBER_DATES],
               "long": [t for t in trades if t["entry_direction"] == "LONG"],
               "short": [t for t in trades if t["entry_direction"] == "SHORT"]}
    mfe_mae = {}
    for label, selected in subsets.items():
        mfe = [float(t["mfe_ticks"]) for t in selected]; mae = [float(t["mae_ticks"]) for t in selected]
        mfe_mae[label] = {"count": len(selected), "mean_mfe": float(np.mean(mfe)) if mfe else None,
                          "median_mfe": float(np.median(mfe)) if mfe else None,
                          "mean_mae": float(np.mean(mae)) if mae else None,
                          "median_mae": float(np.median(mae)) if mae else None}
    first_touch = {}
    for barrier in ("+1/-1", "+2/-2", "+4/-4", "+8/-4"):
        vals = [t["first_touch"][barrier] for t in trades]
        first_touch[barrier] = {name: vals.count(name) for name in ("UP", "DOWN", "TIE", "NO_TOUCH")}
    markouts = {label: _markout_aggregate(selected) for label, selected in subsets.items()}
    return markouts, mfe_mae, first_touch


def _aggregate(date_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trades = sorted([trade for payload in date_payloads for trade in payload.get("trades", [])],
                    key=lambda row: (int(row["entry_time_ns"]), row["date"], row["entry_direction"]))
    spring = [row for row in trades if row["date"] in SPRING_DATES]
    october = [row for row in trades if row["date"] in OCTOBER_DATES]
    long = [row for row in trades if row["entry_direction"] == "LONG"]
    short = [row for row in trades if row["entry_direction"] == "SHORT"]
    daily, weekly = _daily_weekly(trades, ALL_TARGET_DATES)
    markouts, mfe_mae, first_touch = _diagnostics(trades)
    day_r = [float(row["R"]) for row in daily]
    week_r = [float(row["R"]) for row in weekly]
    all_summary = _summarize_trades(trades)
    spring_summary, october_summary = _summarize_trades(spring), _summarize_trades(october)
    if spring_summary["net_R"] >= 0 and october_summary["net_R"] >= 0:
        period_class = "BOTH_POSITIVE"
    elif spring_summary["net_R"] >= 0 and october_summary["net_R"] < 0:
        period_class = "SPRING_ONLY"
    elif spring_summary["net_R"] < 0 and october_summary["net_R"] >= 0:
        period_class = "OCTOBER_ONLY"
    else:
        period_class = "BOTH_NEGATIVE"
    if not long or not short:
        direction_class = "INSUFFICIENT"
    elif sum(t["net_R"] for t in long) > 0 and sum(t["net_R"] for t in short) > 0:
        direction_class = "SYMMETRIC"
    elif sum(t["net_R"] for t in long) > 0 and sum(t["net_R"] for t in short) <= 0:
        direction_class = "LONG_DOMINANT"
    elif sum(t["net_R"] for t in short) > 0 and sum(t["net_R"] for t in long) <= 0:
        direction_class = "SHORT_DOMINANT"
    else:
        direction_class = "OPPOSITE"
    flow_only = {"SPRING": {}, "OCTOBER": {}}
    vacuum_flow = {"SPRING": {}, "OCTOBER": {}}
    for period, day_list in (("SPRING", SPRING_DATES), ("OCTOBER", OCTOBER_DATES)):
        payloads = [p for p in date_payloads if p["date"] in day_list]
        for horizon in HORIZONS_MS:
            sums = [p["flow_only"]["markouts"][str(horizon)] for p in payloads]
            n = sum(x["sample_count"] for x in sums)
            total = sum((x["mean_ticks"] or 0.0) * x["sample_count"] for x in sums)
            flow_only[period][str(horizon)] = {"event_count": sum(p["flow_only"]["event_count"] for p in payloads),
                                                "sample_count": n, "mean_ticks": total / n if n else None}
            selected = [t for t in trades if t["date"] in day_list]
            values = [t["markouts_ticks"].get(str(horizon)) for t in selected]
            values = [float(x) for x in values if x is not None]
            vacuum_flow[period][str(horizon)] = {"event_count": len(selected), "sample_count": len(values),
                                                 "mean_ticks": float(np.mean(values)) if values else None}
    attrition = {}
    for period, days in (("SPRING", SPRING_DATES), ("OCTOBER", OCTOBER_DATES)):
        attrition[period] = {}
        for direction, sign in (("LONG", 1), ("SHORT", -1)):
            rows = [p for p in date_payloads if p["date"] in days]
            attrition[period][direction] = {key: sum(int(p["attrition"].get(key, 0)) for p in rows)
                                             for key in ("raw_pressure_events", "clustered_events", "after_depletion", "after_refill", "after_persistence", "after_confirmation", "actual_entries")}
            # Side-specific stages are sourced from each date's explicit side counts.
            for key in attrition[period][direction]:
                attrition[period][direction][key] = sum(int(p.get("attrition_by_direction", {}).get(str(sign), {}).get(key, 0)) for p in rows)
    daily_positive = sum(x["R"] > 0 for x in daily); daily_negative = sum(x["R"] < 0 for x in daily)
    weekly_positive = sum(x["R"] > 0 for x in weekly); weekly_negative = sum(x["R"] < 0 for x in weekly)
    return {"trades": trades, "performance": all_summary,
            "period_results": {"SPRING_2025": spring_summary, "OCTOBER_2025": october_summary,
                               "classification": period_class},
            "direction_results": {"LONG": _summarize_trades(long), "SHORT": _summarize_trades(short),
                                  "classification": direction_class},
            "daily": daily, "weekly": weekly,
            "daily_summary": {"positive_days": daily_positive, "negative_days": daily_negative,
                              "flat_days": len(daily) - daily_positive - daily_negative,
                              "median_daily_R": float(np.median(day_r)) if day_r else 0.0,
                              "mean_daily_R": float(np.mean(day_r)) if day_r else 0.0,
                              "best_5_days": sorted(daily, key=lambda x: x["R"], reverse=True)[:5],
                              "worst_5_days": sorted(daily, key=lambda x: x["R"])[:5]},
            "weekly_summary": {"positive_weeks": weekly_positive, "negative_weeks": weekly_negative},
            "flow_only": flow_only, "vacuum_entry_markouts": markouts,
            "vacuum_incremental_over_flow_only": {period: {h: (vacuum_flow[period][h]["mean_ticks"] - flow_only[period][h]["mean_ticks"]
                                     if vacuum_flow[period][h]["mean_ticks"] is not None and flow_only[period][h]["mean_ticks"] is not None else None)
                                                    for h in flow_only[period]} for period in flow_only},
            "vacuum_entry_markouts_by_period": vacuum_flow,
            "mfe_mae": mfe_mae, "first_touch": first_touch,
            "component_attrition": attrition,
            "overlap_audit": {"raw_pressure_events": sum(p["raw_pressure_events"] for p in date_payloads),
                              "clustered_events": sum(p["attrition"].get("clustered_events", 0) for p in date_payloads),
                              "actual_entries": len(trades)}}


def _decision(aggregate: Mapping[str, Any]) -> tuple[str, str, bool]:
    perf = aggregate["performance"]; periods = aggregate["period_results"]; dirs = aggregate["direction_results"]
    weekly = aggregate["weekly"]
    flow = aggregate["flow_only"]; inc = aggregate["vacuum_incremental_over_flow_only"]
    long_r = dirs["LONG"]["net_R"]; short_r = dirs["SHORT"]["net_R"]
    later = [aggregate["vacuum_entry_markouts"]["all"][str(h)] for h in (500, 1000, 2000, 5000, 10000, 30000)]
    executable = any(x["mean_ticks"] is not None and x["mean_ticks"] > 0 for x in later)
    both_nonnegative = periods["SPRING_2025"]["net_R"] >= 0 and periods["OCTOBER_2025"]["net_R"] >= 0
    both_directions_nonnegative = long_r >= 0 and short_r >= 0
    weekly_coherent = aggregate["weekly_summary"]["positive_weeks"] > aggregate["weekly_summary"]["negative_weeks"]
    if perf["trade_count"] == 0:
        return "LIQUIDITY_VACUUM_V1_MIXED", "DIAGNOSE_WHICH_FIXED_COMPONENT_OR_REGIME_FAILS", executable
    incremental = [value for period in inc.values() for horizon, value in period.items()
                   if int(horizon) >= 500 and value is not None]
    vacuum_better = bool(incremental) and sum(incremental) / len(incremental) > 0
    reasonable_dd = perf["max_drawdown_R"] <= max(1.0, abs(perf["gross_R"]))
    promising = (perf["net_R"] > 0 and both_nonnegative and both_directions_nonnegative and weekly_coherent
                and reasonable_dd and vacuum_better and executable)
    if promising:
        return "LIQUIDITY_VACUUM_V1_PROMISING", "RUN_SMALL_ROBUSTNESS_GRID_BEFORE_OPTUNA", executable
    if perf["net_R"] > 0 or periods["SPRING_2025"]["net_R"] >= 0 or periods["OCTOBER_2025"]["net_R"] >= 0:
        return "LIQUIDITY_VACUUM_V1_MIXED", "DIAGNOSE_WHICH_FIXED_COMPONENT_OR_REGIME_FAILS", executable
    return "LIQUIDITY_VACUUM_V1_FAILED", "DO_NOT_OPTIMIZE_THIS_V1; evaluate whether FLOW_MOMENTUM_ONLY merits a separate simple branch", executable


def run(*, data_root: Path = DATA_ROOT, output_root: Path = OUT_ROOT, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    if TAPE_VERSION != "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE":
        raise VacuumStudyError("expected canonical candidate tape V2 is not available")
    paths, manifest_rows = _source_catalog(data_root)
    train_manifest = json.loads((TRAIN_TAPE_ROOT.parent / "train-tape-manifest.json").read_text())
    oct_manifest_path = OCT_TAPE_ROOT.parent / "october-tape-manifest.json"
    oct_manifest = json.loads(oct_manifest_path.read_text())
    if train_manifest.get("status") != "COMPLETE" or train_manifest.get("train_dates") != list(SPRING_DATES):
        raise VacuumStudyError("Spring sealed canonical tape manifest does not match 35 frozen target dates")
    if oct_manifest.get("status") != "COMPLETE" or oct_manifest.get("validation_dates") != list(OCTOBER_DATES):
        raise VacuumStudyError("October sealed canonical tape manifest does not match 19 frozen dates")
    tape_paths = {day: _tape_path(day) for day in ALL_TARGET_DATES}
    tape_hashes: dict[str, str] = {}
    for day in ALL_TARGET_DATES:
        tape = tape_paths[day]
        actual = _sha(tape) if tape.is_file() else ""
        declared = (train_manifest if day in SPRING_DATES else oct_manifest).get("source_sha256_by_date", {}).get(day)
        source_sha = str(manifest_rows[day]["sha256"])
        if actual == "" or declared != source_sha:
            raise VacuumStudyError(f"canonical tape/source manifest identity failure: {day}")
        tape_hashes[day] = actual
    if smoke:
        source_days = (DEPENDENCY_DATES[0], SPRING_DATES[0])
        target_days = (SPRING_DATES[0],)
    else:
        source_days = ALL_SOURCE_DATES
        target_days = ALL_TARGET_DATES
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    coverage = {"status": "PASS", "dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES",
                "spring_dates": list(SPRING_DATES), "october_dates": list(OCTOBER_DATES),
                "dependency_dates": list(DEPENDENCY_DATES), "no_mbo": True, "no_mes_market_data": True,
                "files": {day: {"path": str(paths[day]), "bytes": paths[day].stat().st_size,
                               "sha256": str(manifest_rows[day]["sha256"]), "symbol": manifest_rows[day]["symbol"],
                               "category": manifest_rows[day]["category"], "record_count": manifest_rows[day].get("verification", {}).get("record_count")}
                          for day in source_days},
                "canonical_event_tapes": {day: {"path": str(tape_paths[day]), "sha256": tape_hashes[day], "version": TAPE_VERSION}
                                          for day in target_days}}
    _write_json(output_root / "source-coverage.json", coverage)
    histories: list[float] = []
    payloads: dict[str, dict[str, Any]] = {}
    progress = {"completed_dates": [], "resumed_dates": [], "failed_dates": [], "target_dates": list(target_days)}
    for pos, day in enumerate(source_days, 1):
        is_target = day in target_days
        tape_path = tape_paths.get(day)
        tape_sha = tape_hashes.get(day, "dependency-only")
        source_sha = str(manifest_rows[day]["sha256"])
        cp_path = _checkpoint_path(output_root, day)
        cached = _read_checkpoint(cp_path, day=day, source_sha=source_sha, tape_sha=tape_sha) if not force else None
        if cached is not None:
            daily_payload = cached["payload"]
            histories.extend(daily_payload["pressure_history_sample"])
            if is_target:
                payloads[day] = daily_payload
                progress["resumed_dates"].append(day)
            print(f"VACUUM_DATE_RESUME={day}", flush=True)
            continue
        print(f"VACUUM_DATE_START={pos}/{len(source_days)} date={day} target={is_target}", flush=True)
        date_start = time.monotonic()
        compact, temp_path, comp = relative._extract_compact(day, paths[day], checkpoint_dir / "_work" / f"{day}.compact", source_sha)
        try:
            pressure = rolling_pressure(compact)
            threshold = float(np.quantile(np.asarray(histories, dtype=np.float64), 0.90)) if histories else 0.0
            if is_target:
                tape_events, tape_meta = _load_tape(day, tape_path, source_sha)
                result = evaluate_day(day, compact, tape_events, threshold, histories, pressure=pressure)
            else:
                result = {"date": day, "pressure_history_sample": _systematic_sample(np.abs(pressure)),
                          "pressure_observation_count": len(pressure), "dependency_only": True}
            # Target evaluation carries its own sample so date history is only
            # admitted after that date has been fully evaluated.
            if is_target:
                # evaluate_day recomputes pressure; keep the exact fixed sample.
                result["pressure_history_sample"] = _systematic_sample(np.abs(pressure))
                result["pressure_threshold"] = threshold
                result["tape_sha256"] = tape_sha
                result["source_sha256"] = source_sha
                result["elapsed_seconds"] = time.monotonic() - date_start
                payloads[day] = result
                progress["completed_dates"].append(day)
            else:
                result["source_sha256"] = source_sha
                result["tape_sha256"] = tape_sha
                result["elapsed_seconds"] = time.monotonic() - date_start
            checkpoint = {"checkpoint_version": CHECKPOINT_VERSION, "status": "DATE_COMPLETE", "date": day,
                          "source_sha256": source_sha, "tape_sha256": tape_sha,
                          "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256, "payload": result}
            _write_checkpoint(cp_path, checkpoint)
            histories.extend(result["pressure_history_sample"])
            print(f"VACUUM_DATE_COMPLETE={day} pressure_rows={len(pressure)} raw={result.get('raw_pressure_events',0)} clustered={result.get('attrition',{}).get('clustered_events',0)} entries={result.get('attrition',{}).get('actual_entries',0)} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
        finally:
            del compact
            temp_path.unlink(missing_ok=True)
        _write_json(checkpoint_dir / "progress.json", {**progress, "current_date": day,
                    "config_sha256": CONFIG_SHA256, "study_sha256": STUDY_SHA256})
    if set(payloads) != set(target_days):
        raise VacuumStudyError(f"incomplete target checkpoint set: {sorted(set(target_days)-set(payloads))}")
    if smoke:
        return {"status": "SMOKE_PASS", "dates": list(payloads), "entries": sum(len(p["trades"]) for p in payloads.values())}
    date_payloads = [payloads[d] for d in target_days]
    aggregate = _aggregate(date_payloads)
    all_trades = aggregate.pop("trades")
    summary, period_results, direction_results = aggregate["performance"], aggregate["period_results"], aggregate["direction_results"]
    decision, next_step, executable = _decision(aggregate)
    component_by = {period: {direction: stages for direction, stages in rows.items()}
                    for period, rows in aggregate["component_attrition"].items()}
    artifact = {"run_id": RUN_ID, "status": "COMPLETE", "primary_decision": decision, "next_step": next_step,
                "dataset": "SPRING_2025 + OCTOBER_2025", "spring_dates": list(SPRING_DATES),
                "october_dates": list(OCTOBER_DATES), "dependency_dates": list(DEPENDENCY_DATES),
                "native_es_only": True, "dataset_name": "GLBX.MDP3", "schema": "mbp-10", "MBO_used": False,
                "MES_market_data_used": False, "strategy_config": CONFIG,
                "execution_model": {"entry": "first canonical executable ES quote at/after signal anchor+2ms; buy fill ask+1tick, sell fill bid-1tick",
                                    "stop": "exit-side bid/ask crosses fixed stop; adverse one-tick execution",
                                    "target": "exit-side bid/ask crosses fixed target; adverse one-tick execution",
                                    "same_timestamp_precedence": "STOP before TARGET",
                                    "fees": "ES $3 per side per contract; MES fallback $1.25 per side only if ES risk sizing yields zero",
                                    "economics": "ES $50/point; existing RISK_BUDGET_USD and contract caps via size_for_instrument",
                                    "time_exit": "first executable canonical event at/after entry+30s; no trade if full hold cannot be observed within session",
                                    "R": "net PnL / initial risk including entry/stop adverse ticks and round-trip fees",
                                    "execution_policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS},
                "performance": summary, "period_results": period_results, "direction_results": direction_results,
                "daily_summary": aggregate["daily_summary"], "weekly_summary": aggregate["weekly_summary"],
                "flow_only_baseline": aggregate["flow_only"], "vacuum_entry_markouts": aggregate["vacuum_entry_markouts"],
                "vacuum_incremental_over_flow_only": aggregate["vacuum_incremental_over_flow_only"],
                "mfe_mae": aggregate["mfe_mae"], "first_touch": aggregate["first_touch"],
                "daily_results": aggregate["daily"], "weekly_results": aggregate["weekly"],
                "component_attrition": aggregate["component_attrition"], "overlap_audit": aggregate["overlap_audit"],
                "executable_at_500ms_plus": executable, "optimization_performed": False,
                "optuna_performed": False, "threshold_search_performed": False,
                "level_filters_used": False, "pnl_calculated": True,
                "final_oos_accessed": False, "data_downloaded": False,
                "source_sha256_by_date": {d: str(manifest_rows[d]["sha256"]) for d in source_days},
                "tape_sha256_by_date": tape_hashes, "config_sha256": CONFIG_SHA256,
                "study_version_sha256": STUDY_SHA256, "elapsed_seconds": time.monotonic() - started}
    _write_gzip_jsonl(output_root / "trades.jsonl.gz", all_trades)
    files = {
        "strategy-config.json": CONFIG,
        "summary.json": artifact,
        "performance-summary.json": summary,
        "period-results.json": period_results,
        "direction-results.json": direction_results,
        "daily-results.json": {"summary": aggregate["daily_summary"], "dates": aggregate["daily"]},
        "weekly-results.json": {"summary": aggregate["weekly_summary"], "weeks": aggregate["weekly"]},
        "flow-only-baseline.json": {"flow_only": aggregate["flow_only"], "vacuum_entries": aggregate["vacuum_entry_markouts_by_period"],
                                    "incremental": aggregate["vacuum_incremental_over_flow_only"]},
        "component-attrition.json": aggregate["component_attrition"],
        "markouts.json": {"flow_only": aggregate["flow_only"], "vacuum_entries": aggregate["vacuum_entry_markouts"]},
        "mfe-mae.json": aggregate["mfe_mae"],
        "first-touch.json": aggregate["first_touch"],
        "overlap-audit.json": aggregate["overlap_audit"],
        "run-manifest.json": {"run_id": RUN_ID, "status": "COMPLETE", "config_sha256": CONFIG_SHA256,
                              "study_version_sha256": STUDY_SHA256, "source_coverage_path": str(output_root / "source-coverage.json"),
                              "source_coverage_sha256": _sha(output_root / "source-coverage.json"),
                              "no_2026_accessed": True, "final_oos_accessed": False, "data_downloaded": False,
                              "source_sha256_by_date": artifact["source_sha256_by_date"], "tape_sha256_by_date": tape_hashes,
                              "execution_model": artifact["execution_model"]},
    }
    for name, content in files.items():
        _write_json(output_root / name, content)
    report = _render_report(artifact)
    (output_root / "report.md").write_text(report, encoding="utf-8")
    artifact_hashes = {p.name: _sha(p) for p in sorted(output_root.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    artifact_hashes["trades.jsonl.gz"] = _sha(output_root / "trades.jsonl.gz")
    _write_json(output_root / "artifact-hashes.json", {"status": "HASHED", "files": artifact_hashes})
    return artifact


def _render_report(a: Mapping[str, Any]) -> str:
    p = a["performance"]; periods = a["period_results"]; dirs = a["direction_results"]
    lines = [f"# {RUN_ID}", "", f"Decision: **{a['primary_decision']}**", "", "Frozen one-shot strategy run; no optimization or threshold search.", "",
             f"- Spring dates: {len(a['spring_dates'])}; October dates: {len(a['october_dates'])}; dependencies: {', '.join(a['dependency_dates'])}",
             f"- Trades: {p['trade_count']} (long {p['long_count']}, short {p['short_count']}); net R {p['net_R']:.4f}; gross R {p['gross_R']:.4f}",
             f"- Win rate: {p['win_rate']}; PF: {p['profit_factor']}; max DD R: {p['max_drawdown_R']:.4f}",
             f"- Spring: {periods['SPRING_2025']['trade_count']} trades, {periods['SPRING_2025']['net_R']:.4f} net R, PF {periods['SPRING_2025']['profit_factor']}",
             f"- October: {periods['OCTOBER_2025']['trade_count']} trades, {periods['OCTOBER_2025']['net_R']:.4f} net R, PF {periods['OCTOBER_2025']['profit_factor']}",
             f"- Direction classification: {dirs['classification']}", "", "## Execution convention", "",
             json.dumps(a["execution_model"], indent=2, sort_keys=True), "", "## Period results", "",
             "| Period | Trades | Win rate | Avg R | Net R | PF | Max DD R |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in periods.items():
        if isinstance(row, dict) and "trade_count" in row:
            lines.append(f"| {name} | {row['trade_count']} | {row['win_rate']} | {row['average_R']} | {row['net_R']:.4f} | {row['profit_factor']} | {row['max_drawdown_R']:.4f} |")
    lines += ["", "## Frozen strategy configuration", "", "```json", json.dumps(CONFIG, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true", help="dependency plus first Spring target date only")
    parser.add_argument("--force", action="store_true", help="recompute checkpoints; does not modify source data")
    args = parser.parse_args(argv)
    print(json.dumps(run(data_root=args.data_root, output_root=args.output_root, smoke=args.smoke, force=args.force), indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
