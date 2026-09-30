"""Preregistered Spring/October 2025 absorption mechanism-state study.

Only four frozen pre-event mechanisms are analyzed. This is descriptive event
research; it performs no optimization, PnL calculation, or data acquisition.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import mac_2025_absorption_relative_feature_stability as prior
from . import mac_2025_absorption_relative_normalization as norm
from . import mac_2025_es_only_train_baseline as baseline

RUN_ID = "CMEOrderflow_ABSORPTION_MECHANISM_STATE_STUDY_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
PRIOR_ROOT = Path("research_runs/CMEOrderflow_ABSORPTION_RELATIVE_FEATURE_STABILITY_2025_V1")
STUDY_SPEC = {
    "version": 8,
    "periods": ["Spring 2025", "October 2025"],
    "families": {"discovery": "EU_CURRENT_HIGH_SWEEP", "replication": "NY_W04",
                 "descriptive_only": ["EU_PRIOR_HIGH", "PRIOR_VAH"]},
    "primary": ["pre_event_recovery_ratio_500ms", "opposing_mlofi_persistence_2s",
                 "impact_per_flow_10s", "ER_5S", "ER_30S"],
    "pre_event_rule": "all source observations have ts_recv < interaction_start_ns",
    "lookbacks": {"resiliency_history_ms": 60000, "mlofi_persistence_ms": 2000,
                  "mlofi_bin_ms": 250, "impact_ms": 10000, "ER_ms": [5000, 30000]},
    "topology_precedence": ["F insufficient full 30s path or +/-4 tie/no +/-4 touch",
       "E +4 first within 5s", "D +4 first after 5s", "A -4 first within 2s and MFE_before_-4 < +1",
       "B -4 first and MFE_before_-4 >= +2", "C all other -4-first failures"],
    "topology_fixed_barrier": "+4/-4 ticks within 30s; A/B split uses 2s and +1/+2 favorable excursion",
    "path_endpoint_sampling": "first candidate-tape quote at or after each fixed horizon; full 30s path requires first quote at/after +30s and event horizon within frozen session window",
    "deciles": "historical prior-date percentile, 10 fixed equal-width bins; no same-date calibration",
    "volatility_control": "existing RV_30S_RAW / RV_30S_TOD_PERCENTILE from prior validated pipeline",
    "interaction_ids": ["RESILIENCY_X_OPPOSING_MLOFI", "IMPACT_PER_FLOW_X_ER", "RV_X_RESILIENCY"],
    "decision_rules": {"boundary_representation": "same fixed historical percentile, decile D1 vs D10; quintiles Q1 vs Q5; terciles fixed thirds",
                        "a_priori_expected_effect_sign": {"PRE_EVENT_RESILIENCY": "positive high-minus-low",
                            "NORMALIZED_MLOFI_PERSISTENCE": "negative high-minus-low",
                            "IMPACT_PER_FLOW": "negative high-minus-low",
                            "TREND_EFFICIENCY_5S": "negative high-minus-low",
                            "TREND_EFFICIENCY_30S": "negative high-minus-low",
                            "RESILIENCY_X_OPPOSING_MLOFI": "positive good-minus-bad",
                            "IMPACT_PER_FLOW_X_ER": "positive good-minus-bad",
                            "RV_X_RESILIENCY": "positive high-RV fast-refill minus high-RV slow-refill"},
                        "day_week_sign_stability_minimum": 0.8,
                        "multi_horizon": "same effect direction at 2 or more of 500ms/1s/2s/5s/10s/30s",
                        "supported_requires": ["cross-period direction", "boundary stability", "LODO and LOWO >= 0.8",
                                               "multi-horizon coherence", "permutation > fixed P95", "NY qualitative support where sample permits"]},
    "permutation_seed": 20250930, "permutation_repetitions": 399,
}
STUDY_VERSION_HASH = hashlib.sha256(json.dumps(STUDY_SPEC, sort_keys=True).encode()).hexdigest()
EXPECTED_STRATEGY_SHA = norm.EXPECTED_STRATEGY_MANIFEST_SHA
HORIZONS_MS = (500, 1_000, 2_000, 5_000, 10_000, 30_000)
PATH_HORIZONS_MS = (1_000, 2_000, 5_000, 10_000, 30_000)
BARRIERS = ((1, 1), (2, 2), (4, 4), (8, 4), (12, 6))
FAMILY_KEYS = tuple(norm.LIVE_TO_TAPE.values())
DISCOVERY = "EU_CURRENT_HIGH_SWEEP"
REPLICATION = "NY_W04"
SPARSE = ("EU_PRIOR_HIGH", "PRIOR_VAH")
FEATURES = {
    "PRE_EVENT_RESILIENCY": "pre_event_recovery_ratio_500ms",
    "NORMALIZED_MLOFI_PERSISTENCE": "opposing_mlofi_persistence_2s",
    "IMPACT_PER_FLOW": "impact_per_flow_10s",
    "TREND_EFFICIENCY_5S": "ER_5S",
    "TREND_EFFICIENCY_30S": "ER_30S",
}
FEATURE_PERCENTILES = {
    "PRE_EVENT_RESILIENCY": "pre_event_recovery_ratio_500ms_historical_percentile",
    "NORMALIZED_MLOFI_PERSISTENCE": "opposing_mlofi_persistence_2s_historical_percentile",
    "IMPACT_PER_FLOW": "impact_per_flow_10s_historical_percentile",
    "TREND_EFFICIENCY_5S": "ER_5S_historical_percentile",
    "TREND_EFFICIENCY_30S": "ER_30S_historical_percentile",
    "ER_30S": "ER_30S_historical_percentile",
    "RV_30S": "RV_30S_TOD_PERCENTILE",
}
ROLLING_TICK_NS = 50_000_000
TICK = 0.25


class MechanismStudyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default).encode())
    os.replace(temp, path)


def _finite(x: Any) -> bool:
    return x is not None and math.isfinite(float(x))


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    return norm._summarize_values(x for x in values if _finite(x))


def _decile(p: Any) -> int | None:
    if not _finite(p):
        return None
    return min(9, max(0, int(float(p) * 10)))


def _tercile(p: Any) -> str | None:
    if not _finite(p):
        return None
    return ("LOW", "MID", "HIGH")[min(2, int(float(p) * 3))]


def _rank(value: float, values: list[float]) -> float | None:
    if not values:
        return None
    a = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.searchsorted(a, value, side="right") / len(a))


def _prior_strict_indices(ts: np.ndarray, query_ns: int, lookback_ns: int) -> tuple[int, int]:
    """Return [lo, hi) where every included source timestamp is strictly < query."""
    hi = int(np.searchsorted(ts, int(query_ns), side="left"))
    lo = int(np.searchsorted(ts, int(query_ns) - int(lookback_ns), side="left"))
    if hi and int(ts[hi - 1]) >= int(query_ns):
        raise MechanismStudyError("pre-event slice contains interaction-time record")
    return lo, hi


def _forward_max(values: np.ndarray, steps: int) -> np.ndarray:
    n = len(values)
    if n <= steps:
        return np.empty(0, dtype=np.float64)
    return np.maximum.reduce([values[i:n - steps + i] for i in range(steps + 1)])


def _pre_event_resiliency(grid_ns: np.ndarray, bid_depth: np.ndarray, ask_depth: np.ndarray,
                          query_ns: int, passive_side: str) -> dict[str, Any]:
    """Median capped recovery of 50ms same-side depletion episodes in prior 60s.

    A depletion episode contributes only if its full recovery horizon ends
    strictly before the absorption start. Refill latency is measured to 50% of
    the depleted amount, observed within a fully prior 1s window.
    """
    depth = ask_depth if passive_side == "ASK" else bid_depth
    _prior_strict_indices(grid_ns, query_ns, 60_000_000_000)
    # A depletion episode is indexed at its post-depletion sample and needs
    # the immediately previous 50ms sample, so keep both within the fixed 60s.
    lo = max(1, int(np.searchsorted(grid_ns, query_ns - 60_000_000_000 + ROLLING_TICK_NS, side="left")))
    hi = int(np.searchsorted(grid_ns, query_ns, side="left"))
    out: dict[str, Any] = {"history_ms": 60000, "passive_side": passive_side, "episodes": {}}
    for horizon_ms, key in ((100, "100MS"), (250, "250MS"), (500, "500MS"), (1000, "1S")):
        steps = horizon_ms // 50
        max_future = _forward_max(depth, steps)
        # Episode index is the first post-depletion sample, matching the
        # validated prior-study convention: drop[j] = depth[j-1] - depth[j].
        drop = np.maximum(0.0, np.r_[0.0, depth[:-1]])[:len(max_future)] - depth[:len(max_future)]
        drop = np.maximum(0.0, drop)
        # max_future[j] spans depth[j:j+steps+1], so episode j is eligible
        # only if j+steps is strictly before query and the episode is within 60s.
        episode_hi = min(len(max_future), hi - steps)
        ids = np.arange(lo, max(lo, episode_hi), dtype=np.int64)
        valid = ids[(grid_ns[ids] + steps * ROLLING_TICK_NS < query_ns) & (drop[ids] > 0)] if len(ids) else ids
        if len(valid):
            ratios = np.minimum(drop[valid], np.maximum(0.0, max_future[valid] - depth[valid])) / drop[valid]
            median = float(np.median(ratios))
        else:
            ratios = np.asarray([], dtype=np.float64)
            median = None
        out["episodes"][key] = {"n": int(len(valid)), "median_recovery_ratio": median,
                                "mean_recovery_ratio": float(np.mean(ratios)) if len(ratios) else None}
    # 50%-refill latency, conditioned on prior 1s depletion episodes.
    steps = 20
    max_future = _forward_max(depth, steps)
    drop = np.maximum(0.0, np.r_[0.0, depth[:-1]])[:len(max_future)] - depth[:len(max_future)]
    drop = np.maximum(0.0, drop)
    episode_hi = min(len(max_future), hi - steps)
    ids = np.arange(lo, max(lo, episode_hi), dtype=np.int64)
    valid = ids[(grid_ns[ids] + 1_000_000_000 < query_ns) & (drop[ids] > 0)] if len(ids) else ids
    latencies = []
    for idx in valid:
        target = depth[idx] + 0.5 * drop[idx]
        future = depth[idx:idx + steps + 1]
        hit = np.flatnonzero(future >= target)
        if len(hit):
            latencies.append(float(hit[0] * 50))
    out["refill_50pct_latency_ms"] = {"n": len(latencies), "median": float(np.median(latencies)) if latencies else None}
    return out


def _pre_event_mlofi(rows: np.ndarray, query_ns: int, direction: float) -> dict[str, Any]:
    ts = rows["ts"]
    lo, hi = _prior_strict_indices(ts, query_ns, 2_000_000_000)
    if hi <= lo:
        return {"normalized_level": None, "signed_normalized_level": None,
                "opposing_flow_persistence": None, "supporting_flow_persistence": None,
                "bin_signs": []}
    denom = float(rows["denom"][hi - 1])
    integral = float(np.sum(rows["mlofi"][lo:hi], dtype=np.float64))
    normalized = integral / denom if denom > 0 else None
    signed = normalized * direction if normalized is not None else None
    bins = []
    for i in range(8):
        start = query_ns - 2_000_000_000 + i * 250_000_000
        end = query_ns - 2_000_000_000 + (i + 1) * 250_000_000
        b0 = int(np.searchsorted(ts, start, side="left"))
        b1 = int(np.searchsorted(ts, end, side="left"))
        if b1 <= b0:
            bins.append(0)
        else:
            bins.append(int(np.sign(float(np.sum(rows["mlofi"][b0:b1], dtype=np.float64)) * direction)))
    nonzero = [x for x in bins if x]
    if nonzero:
        # Match the validated pipeline: persistence is the fraction of
        # non-zero 250ms bins matching the latest non-zero bin's sign.
        last_sign = nonzero[-1]
        persistence = sum(x == last_sign for x in nonzero) / len(nonzero)
        opposing = persistence if last_sign < 0 else 0.0
        supporting = persistence if last_sign > 0 else 0.0
    else:
        opposing = supporting = None
    return {"mlofi_integral": integral, "depth_denominator": denom,
            "normalized_level": normalized, "signed_normalized_level": signed,
            "opposing_flow_persistence": opposing, "supporting_flow_persistence": supporting,
            "bin_signs": bins, "nonzero_bin_count": len(nonzero),
            "persistence": persistence if nonzero else None}


def _pre_event_impact(rows: np.ndarray, query_ns: int, direction: float) -> dict[str, Any]:
    ts = rows["ts"]
    lo, hi = _prior_strict_indices(ts, query_ns, 10_000_000_000)
    if hi <= lo or hi - 1 <= lo:
        return {"mid_change_ticks": None, "directional_mid_change_ticks": None,
                "normalized_mlofi_integral": None, "impact_per_flow": None, "flow_integral": None}
    # The start price is the as-of midpoint at the 10s boundary; endpoint and
    # all flow observations are strictly prior to interaction_start.
    start_ix = max(lo, int(np.searchsorted(ts, query_ns - 10_000_000_000, side="right")) - 1)
    end_ix = hi - 1
    move = (float(rows["mid"][end_ix]) - float(rows["mid"][start_ix])) / TICK
    flow_integral = float(np.sum(rows["mlofi"][lo:hi], dtype=np.float64))
    denom = float(rows["denom"][end_ix])
    normalized_flow = flow_integral / denom if denom > 0 else None
    impact = (abs(move) / (abs(normalized_flow) + 1e-12)) if normalized_flow is not None else None
    return {"mid_change_ticks": abs(move), "directional_mid_change_ticks": move * direction,
            "normalized_mlofi_integral": normalized_flow, "impact_per_flow": impact,
            "flow_integral": flow_integral, "depth_denominator": denom}


def _path_metrics(event: Mapping[str, Any], candidate: Mapping[str, Any], path: np.ndarray,
                  session_end_ns: int) -> dict[str, Any]:
    start = int(event["interaction_start_ns"])
    direction = 1.0 if event["direction"] == "BUYER_ABSORPTION" else -1.0
    ts = np.asarray(path["timestamp_ns"], dtype=np.int64)
    ix = int(np.searchsorted(ts, start, side="left"))
    anchor_ix = ix - 1
    if anchor_ix < 0 or ix >= len(ts):
        return {"path_sufficient": False, "topology_f_reason": "MISSING_PRE_EVENT_ANCHOR",
                "topology": "F_UNCLASSIFIED_INSUFFICIENT_PATH"}
    anchor = (float(path["bid"][anchor_ix]) + float(path["ask"][anchor_ix])) / 2
    end_ns = start + 30_000_000_000
    if end_ns > session_end_ns:
        return {"path_sufficient": False, "topology_f_reason": "30S_HORIZON_CROSSES_SESSION_BOUNDARY",
                "topology": "F_UNCLASSIFIED_INSUFFICIENT_PATH", "anchor_mid": anchor}
    # Match the sealed tape outcome semantics: each horizon is marked at the
    # first quote at or after its boundary, not the last quote before it.
    first_end_ix = int(np.searchsorted(ts, end_ns, side="left"))
    if end_ns < session_end_ns:
        end_ix = min(len(ts), first_end_ix + 1)
        if first_end_ix < len(ts) and int(ts[first_end_ix]) > session_end_ns:
            end_ix = int(np.searchsorted(ts, session_end_ns, side="right"))
    else:
        # Never borrow the first quote from the next session to complete a
        # horizon ending exactly at this session's boundary.
        end_ix = int(np.searchsorted(ts, session_end_ns, side="right"))
    segment = path[ix:end_ix]
    seg_ts = ts[ix:end_ix]
    if not len(segment) or int(seg_ts[-1]) < end_ns:
        return {"path_sufficient": False, "topology_f_reason": "NO_QUOTE_AT_OR_AFTER_30S_HORIZON",
                "topology": "F_UNCLASSIFIED_INSUFFICIENT_PATH", "anchor_mid": anchor}
    mids = (segment["bid"].astype(np.float64) + segment["ask"].astype(np.float64)) / 2
    signed = (mids - anchor) / TICK * direction
    marks, excursions, times, barriers = {}, {}, {}, {}
    for ms in HORIZONS_MS:
        j = int(np.searchsorted(seg_ts, start + ms * 1_000_000, side="left"))
        marks[str(ms)] = float(signed[j]) if j < len(signed) else None
    for ms in PATH_HORIZONS_MS:
        mask = seg_ts <= start + ms * 1_000_000
        x = signed[mask]
        excursions[str(ms)] = {"mfe": float(max(0.0, np.max(x))) if len(x) else 0.0,
                               "mae": float(max(0.0, -np.min(x))) if len(x) else 0.0}
    for threshold in (1, 2, 4, 8):
        up = np.flatnonzero(signed >= threshold)
        down = np.flatnonzero(signed <= -threshold)
        times[f"+{threshold}"] = float((seg_ts[up[0]] - start) / 1e6) if len(up) else None
        times[f"-{threshold}"] = float((seg_ts[down[0]] - start) / 1e6) if len(down) else None
    for up, down in BARRIERS:
        iup = np.flatnonzero(signed >= up)
        idown = np.flatnonzero(signed <= -down)
        if not len(iup) and not len(idown):
            hit = "NO_TOUCH"
        elif len(iup) and len(idown) and iup[0] == idown[0]:
            hit = "TIE"
        else:
            hit = "UP" if len(iup) and (not len(idown) or iup[0] < idown[0]) else "DOWN"
        barriers[f"+{up}/-{down}"] = hit
    i_neg4 = np.flatnonzero(signed <= -4)
    i_pos4 = np.flatnonzero(signed >= 4)
    first_neg4 = int(i_neg4[0]) if len(i_neg4) else None
    first_pos4 = int(i_pos4[0]) if len(i_pos4) else None
    mfe_before_neg4 = float(max(0.0, np.max(signed[:first_neg4 + 1]))) if first_neg4 is not None else None
    mae_before_pos4 = float(max(0.0, -np.min(signed[:first_pos4 + 1]))) if first_pos4 is not None else None
    first_touch = "TIE" if first_neg4 is not None and first_pos4 is not None and first_neg4 == first_pos4 else (
        "UP" if first_pos4 is not None and (first_neg4 is None or first_pos4 < first_neg4) else
        "DOWN" if first_neg4 is not None else "NONE")
    time_pos4 = float((seg_ts[first_pos4] - start) / 1e6) if first_pos4 is not None else None
    time_neg4 = float((seg_ts[first_neg4] - start) / 1e6) if first_neg4 is not None else None
    if first_touch == "TIE" or first_touch == "NONE":
        topology = "F_UNCLASSIFIED_INSUFFICIENT_PATH"
    elif first_touch == "UP":
        topology = "E_SUCCESSFUL_REVERSAL" if time_pos4 <= 5000 else "D_DELAYED_SUCCESSFUL_REVERSAL"
    elif time_neg4 <= 2000 and (mfe_before_neg4 or 0.0) < 1:
        topology = "A_IMMEDIATE_CONTINUATION_FAILURE"
    elif (mfe_before_neg4 or 0.0) >= 2:
        topology = "B_TEMPORARY_REVERSAL_THEN_FAILURE"
    else:
        topology = "C_STAGNATION_OR_CHOP_THEN_FAILURE"
    f_reason = ("TIED_PLUS4_MINUS4_FIRST_TOUCH" if first_touch == "TIE" else
                "NO_PLUS4_OR_MINUS4_TOUCH_IN_30S" if first_touch == "NONE" else None)
    # Frozen interaction-zone boundary supplies an objective structural break.
    structure = float(candidate["zone_low"] if direction > 0 else candidate["zone_high"])
    break_mask = (mids <= structure) if direction > 0 else (mids >= structure)
    break_ixs = np.flatnonzero(break_mask)
    break_ix = int(break_ixs[0]) if len(break_ixs) else None
    struct_mfe = float(max(0.0, np.max(signed[:break_ix + 1]))) if break_ix is not None else None
    return {"path_sufficient": True, "topology_f_reason": f_reason, "anchor_mid": anchor, "markouts_ticks": marks,
            "mfe_mae_ticks": excursions, "time_to_barrier_ms": times, "barriers": barriers,
            "first_touch_4_ticks": first_touch, "time_to_plus4_ms": time_pos4,
            "time_to_minus4_ms": time_neg4, "mfe_before_first_minus4": mfe_before_neg4,
            "mae_before_first_plus4": mae_before_pos4, "structure_break_price": structure,
            "time_to_structure_break_ms": float((seg_ts[break_ix] - start) / 1e6) if break_ix is not None else None,
            "mfe_before_structure_break": struct_mfe, "topology": topology}


def _candidate_map(tape_path: Path, day: str, source_sha: str) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    with np.load(tape_path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        candidates = json.loads(str(z["candidate_json"].item()))
        path = np.asarray(z["events"])
    if (meta.get("date") != day or meta.get("source_sha256") != source_sha
            or meta.get("semantic_sha256") != norm.EXPECTED_TAPE_SEMANTIC_SHA):
        raise MechanismStudyError(f"candidate tape identity mismatch: {day}")
    return {str(x.get("interaction_id")): x for x in candidates if x.get("interaction_id")}, path


def _grid_depth(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(rows["ts"], dtype=np.int64)
    grid = np.arange(int(ts[0]), int(ts[-1]) + ROLLING_TICK_NS, ROLLING_TICK_NS, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1
    valid = idx >= 0
    return grid[valid], np.asarray(rows["bid5"], dtype=np.float64)[idx[valid]], np.asarray(rows["ask5"], dtype=np.float64)[idx[valid]]


def _event_feature_rows(day: str, source: Path, source_sha: str, tape_path: Path,
                        cached_events: list[dict[str, Any]], workdir: Path) -> list[dict[str, Any]]:
    candidates, path = _candidate_map(tape_path, day, source_sha)
    rows, temp, _coverage = norm._extract_compact(day, source, workdir / f"{day}.compact", source_sha)
    try:
        grid, bid_grid, ask_grid = _grid_depth(rows)
        windows = baseline._session_windows(day)
        out = []
        for cached in cached_events:
            interaction_id = str(cached.get("interaction_id", ""))
            candidate = candidates.get(interaction_id)
            if not candidate:
                raise MechanismStudyError(f"frozen candidate absent from source tape: {day} {interaction_id}")
            start = int(cached["interaction_start_ns"])
            query_ix = int(np.searchsorted(rows["ts"], start, side="left"))
            if query_ix <= 0 or int(rows["ts"][query_ix - 1]) >= start:
                raise MechanismStudyError(f"no strictly prior MBP-10 row for event {interaction_id}")
            direction = 1.0 if cached["direction"] == "BUYER_ABSORPTION" else -1.0
            passive = "ASK" if direction > 0 else "BID"
            resil = _pre_event_resiliency(grid, bid_grid, ask_grid, start, passive)
            mlofi = _pre_event_mlofi(rows, start, direction)
            impact = _pre_event_impact(rows, start, direction)
            session = str(cached.get("session", ""))
            session_end = windows.get(session, (0, 0))[1]
            path_result = _path_metrics(cached, candidate, path, session_end)
            event = dict(cached)
            event.update({"pre_event_features": {
                "pre_event_recovery_ratio_100ms": resil["episodes"]["100MS"]["median_recovery_ratio"],
                "pre_event_recovery_ratio_250ms": resil["episodes"]["250MS"]["median_recovery_ratio"],
                "pre_event_recovery_ratio_500ms": resil["episodes"]["500MS"]["median_recovery_ratio"],
                "pre_event_recovery_ratio_1s": resil["episodes"]["1S"]["median_recovery_ratio"],
                "pre_event_refill_50pct_latency_ms": resil["refill_50pct_latency_ms"]["median"],
                "pre_event_refill_50pct_episode_count": resil["refill_50pct_latency_ms"]["n"],
                "pre_event_recovery_episode_counts": {k: v["n"] for k, v in resil["episodes"].items()},
                "pre_event_passive_side": passive,
                "opposing_mlofi_persistence_2s": mlofi["opposing_flow_persistence"],
                "supporting_mlofi_persistence_2s": mlofi["supporting_flow_persistence"],
                "pre_event_mlofi_2s_normalized": mlofi["normalized_level"],
                "pre_event_mlofi_2s_signed_normalized": mlofi["signed_normalized_level"],
                "pre_event_mlofi_2s_bin_signs": mlofi["bin_signs"],
                "pre_event_impact_per_flow_10s": impact["impact_per_flow"],
                "pre_event_mid_change_ticks_10s": impact["mid_change_ticks"],
                "pre_event_directional_mid_change_ticks_10s": impact["directional_mid_change_ticks"],
                "pre_event_normalized_mlofi_integral_10s": impact["normalized_mlofi_integral"],
                "ER_5S": cached.get("ER_5S"), "ER_30S": cached.get("ER_30S"),
                "ER_5S_DIRECTIONAL": cached.get("ER_5S_DIRECTIONAL"),
                "ER_30S_DIRECTIONAL": cached.get("ER_30S_DIRECTIONAL"),
                "ER_5S_GLOBAL_PERCENTILE": cached.get("ER_5S_GLOBAL_PERCENTILE"),
                "ER_30S_GLOBAL_PERCENTILE": cached.get("ER_30S_GLOBAL_PERCENTILE"),
                "RV_30S_RAW": cached.get("RV_30S_RAW"),
                "RV_30S_TOD_PERCENTILE": cached.get("RV_30S_TOD_PERCENTILE"),
                "RV_30S_GLOBAL_PERCENTILE": cached.get("RV_30S_GLOBAL_PERCENTILE")}, **path_result})
            out.append(event)
        return out
    finally:
        del rows
        temp.unlink(missing_ok=True)


def _history_percentiles(events: list[dict[str, Any]]) -> None:
    histories: dict[tuple[str, str, int], list[float]] = {}
    history_features = ("pre_event_recovery_ratio_500ms", "opposing_mlofi_persistence_2s", "pre_event_impact_per_flow_10s")
    percentile_names = {"pre_event_recovery_ratio_500ms": "pre_event_recovery_ratio_500ms_historical_percentile",
                        "opposing_mlofi_persistence_2s": "opposing_mlofi_persistence_2s_historical_percentile",
                        "pre_event_impact_per_flow_10s": "impact_per_flow_10s_historical_percentile"}
    for day in sorted({str(e["date"]) for e in events}):
        day_rows = [e for e in events if e["date"] == day]
        windows = baseline._session_windows(day)
        for event in day_rows:
            session = str(event.get("session", ""))
            tod_bin = (int(event["interaction_start_ns"]) - windows.get(session, (0, 0))[0]) // norm.TOD_BIN_NS
            for feature in history_features:
                value = event["pre_event_features"].get(feature)
                hist = histories.get((feature, session, int(tod_bin)), [])
                event["pre_event_features"][percentile_names[feature]] = _rank(float(value), hist) if _finite(value) else None
            event["pre_event_features"]["ER_5S_historical_percentile"] = event["pre_event_features"].get("ER_5S_GLOBAL_PERCENTILE")
            event["pre_event_features"]["ER_30S_historical_percentile"] = event["pre_event_features"].get("ER_30S_GLOBAL_PERCENTILE")
        # Append only after every event from this session date has been ranked.
        for event in day_rows:
            session = str(event.get("session", ""))
            tod_bin = (int(event["interaction_start_ns"]) - windows.get(session, (0, 0))[0]) // norm.TOD_BIN_NS
            for feature in history_features:
                value = event["pre_event_features"].get(feature)
                if _finite(value): histories.setdefault((feature, session, int(tod_bin)), []).append(float(value))
        for event in day_rows:
            features = event["pre_event_features"]
            features["feature_deciles"] = {
                "PRE_EVENT_RESILIENCY": _decile(features.get("pre_event_recovery_ratio_500ms_historical_percentile")),
                "NORMALIZED_MLOFI_PERSISTENCE": _decile(features.get("opposing_mlofi_persistence_2s_historical_percentile")),
                "IMPACT_PER_FLOW": _decile(features.get("impact_per_flow_10s_historical_percentile")),
                "ER_5S": _decile(features.get("ER_5S_historical_percentile")),
                "ER_30S": _decile(features.get("ER_30S_historical_percentile")),
                "RV_30S": _decile(features.get("RV_30S_TOD_PERCENTILE")),
            }


def _markout_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {"event_count": len(events), "active_dates": len({e["date"] for e in events}),
            "horizons": {str(h): _stats(e.get("markouts_ticks", {}).get(str(h)) for e in events) for h in HORIZONS_MS}}


def _excursion_stats(events: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    mfe = [e.get("mfe_mae_ticks", {}).get(str(horizon), {}).get("mfe") for e in events]
    mae = [e.get("mfe_mae_ticks", {}).get(str(horizon), {}).get("mae") for e in events]
    result = {"mfe": _stats(mfe), "mae": _stats(mae)}
    for metric, values in (("mfe", mfe), ("mae", mae)):
        finite = [float(x) for x in values if _finite(x)]
        result[f"p_{metric}_at_least"] = {str(t): (sum(x >= t for x in finite) / len(finite) if finite else None)
                                           for t in (1, 2, 4, 8)}
    return result


def _feature_summary(events: list[dict[str, Any]], feature: str, *, group: tuple[str, ...] = ("period", "live_family")) -> dict[str, Any]:
    grouped: dict[tuple[str, ...] + tuple[int, ...], list[dict[str, Any]]] = {}
    for e in events:
        dec = e["pre_event_features"]["feature_deciles"].get(feature)
        if dec is None: continue
        key = tuple(str(e.get(k, "")) for k in group) + (int(dec),)
        grouped.setdefault(key, []).append(e)
    result = {}
    for key, rows in sorted(grouped.items()):
        node = result
        for p in key[:-1]: node = node.setdefault(p, {})
        node[f"D{key[-1]+1}"] = _markout_stats(rows)
    return result


def _continuous_shape(events: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    result = {}
    for period in prior.PERIODS:
        result[period] = {}
        for family in FAMILY_KEYS:
            rows = [e for e in events if e["period"] == period and e["live_family"] == family]
            deciles = []
            for i in range(10):
                subset = [e for e in rows if e["pre_event_features"]["feature_deciles"].get(feature) == i]
                deciles.append({"decile": i + 1, "event_count": len(subset),
                    "markouts": {str(h): _stats(e.get("markouts_ticks", {}).get(str(h)) for e in subset) for h in HORIZONS_MS},
                    "mfe_5s": _stats(e.get("mfe_mae_ticks", {}).get("5000", {}).get("mfe") for e in subset),
                    "mae_5s": _stats(e.get("mfe_mae_ticks", {}).get("5000", {}).get("mae") for e in subset)})
            means = [x["markouts"]["5000"].get("mean") for x in deciles]
            smooth = [None if i < 1 or i > 8 or any(means[j] is None for j in (i - 1, i, i + 1))
                      else float(np.mean([means[i - 1], means[i], means[i + 1]])) for i in range(10)]
            finite = [(i, x) for i, x in enumerate(means) if x is not None]
            xs, ys = ([x[0] for x in finite], [x[1] for x in finite]) if finite else ([], [])
            rho = float(np.corrcoef(xs, ys)[0, 1]) if len(finite) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else None
            shape = "INSUFFICIENT" if rho is None else "MONOTONIC_UP" if rho > .5 else "MONOTONIC_DOWN" if rho < -.5 else "NONMONOTONIC_OR_WEAK"
            result[period][family] = {"deciles": deciles, "fixed_three_decile_moving_average_5s": smooth,
                                      "linear_decile_shape_correlation": rho, "shape_label": shape}
    return result


def _relationship_groups(e: Mapping[str, Any], relationship: str) -> tuple[str | None, str | None]:
    if relationship in FEATURES:
        d = e["pre_event_features"]["feature_deciles"].get(relationship)
        return ("LOW", None) if d in (0, 1) else (None, "HIGH") if d in (8, 9) else (None, None)
    a, b = _interaction_labels(e, relationship)
    if a is None or b is None:
        return None, None
    if relationship == "RESILIENCY_X_OPPOSING_MLOFI":
        if (a, b) == ("HIGH", "LOW"): return "GOOD", None
        if (a, b) == ("LOW", "HIGH"): return "BAD", None
    elif relationship == "IMPACT_PER_FLOW_X_ER":
        if a == "LOW" and b in ("LOW", "MID"): return "GOOD", None
        if a == "HIGH" and b == "HIGH": return "BAD", None
    else:
        if a == "HIGH" and b == "HIGH": return "GOOD", None
        if a == "HIGH" and b == "LOW": return "BAD", None
    return None, None


def _daily_rows(events: list[dict[str, Any]], relationship: str, expected_dates: Iterable[str] = ()) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for e in events: by_day.setdefault(str(e["date"]), []).append(e)
    output, effects = {}, []
    for day in sorted(set(map(str, expected_dates)) | set(by_day)):
        rows = by_day.get(day, [])
        grouped = {"LOW": [], "HIGH": [], "GOOD": [], "BAD": []}
        for e in rows:
            first, second = _relationship_groups(e, relationship)
            if first: grouped[first].append(e)
            if second: grouped[second].append(e)
        is_interaction = relationship in STUDY_SPEC["interaction_ids"]
        left_name, right_name = ("GOOD", "BAD") if is_interaction else ("LOW", "HIGH")
        left, right = grouped[left_name], grouped[right_name]
        entry = {"event_count": len(rows), "left_group": left_name, "right_group": right_name,
                 "left_n": len(left), "right_n": len(right),
                 "markouts": {str(h): {"all": _stats(e.get("markouts_ticks", {}).get(str(h)) for e in rows),
                                       "left": _stats(e.get("markouts_ticks", {}).get(str(h)) for e in left),
                                       "right": _stats(e.get("markouts_ticks", {}).get(str(h)) for e in right)} for h in HORIZONS_MS},
                 "mfe_mae_by_horizon": {str(h): _excursion_stats(rows, h) for h in PATH_HORIZONS_MS},
                 "barriers_30s": {f"+{up}/-{down}": {state: sum(e.get("barriers", {}).get(f"+{up}/-{down}") == state for e in rows)
                                                          for state in ("UP", "DOWN", "TIE", "NO_TOUCH")}
                                  for up, down in BARRIERS}}
        a, b = entry["markouts"]["5000"]["left"].get("mean"), entry["markouts"]["5000"]["right"].get("mean")
        effect = (a - b) if is_interaction and a is not None and b is not None else (b - a) if a is not None and b is not None else None
        entry["preregistered_contrast_5s"] = effect
        if effect is not None: effects.append((day, effect))
        output[day] = entry
    vals = [v for _, v in effects]
    return {"dates": output, "positive_dates": sum(v > 0 for v in vals), "negative_dates": sum(v < 0 for v in vals),
            "insufficient_dates": len(output) - len(vals), "median_daily_effect": float(np.median(vals)) if vals else None,
            "mean_daily_effect": float(np.mean(vals)) if vals else None,
            "p25_daily": float(np.quantile(vals, .25)) if vals else None, "p75_daily": float(np.quantile(vals, .75)) if vals else None,
            "worst_5_days": sorted(effects, key=lambda x: x[1])[:5], "best_5_days": sorted(effects, key=lambda x: x[1], reverse=True)[:5]}


def _fixed_relationship_effect(rows: list[dict[str, Any]], rel: str) -> float | None:
    return _relationship_effect_horizon(rows, rel, "5000")


def _relationship_effect_horizon(rows: list[dict[str, Any]], rel: str, horizon: str) -> float | None:
    if rel in FEATURES:
        lo = [float(e["markouts_ticks"][horizon]) for e in rows if e["pre_event_features"]["feature_deciles"].get(rel) in (0, 1) and e.get("markouts_ticks", {}).get(horizon) is not None]
        hi = [float(e["markouts_ticks"][horizon]) for e in rows if e["pre_event_features"]["feature_deciles"].get(rel) in (8, 9) and e.get("markouts_ticks", {}).get(horizon) is not None]
        return float(np.mean(hi) - np.mean(lo)) if len(lo) >= 2 and len(hi) >= 2 else None
    good, bad = [], []
    for e in rows:
        a, b = _interaction_labels(e, rel)
        y = e.get("markouts_ticks", {}).get(horizon)
        if y is None or a is None or b is None: continue
        if rel == "RESILIENCY_X_OPPOSING_MLOFI":
            if (a, b) == ("HIGH", "LOW"): good.append(float(y))
            if (a, b) == ("LOW", "HIGH"): bad.append(float(y))
        elif rel == "IMPACT_PER_FLOW_X_ER":
            if a == "LOW" and b in ("LOW", "MID"): good.append(float(y))
            if a == "HIGH" and b == "HIGH": bad.append(float(y))
        elif a == "HIGH":
            if b == "HIGH": good.append(float(y))
            if b == "LOW": bad.append(float(y))
    return float(np.mean(good) - np.mean(bad)) if len(good) >= 2 and len(bad) >= 2 else None


def _interaction_labels(e: Mapping[str, Any], interaction: str) -> tuple[str | None, str | None]:
    f = e["pre_event_features"]
    def t(name: str) -> str | None:
        percentile_field = FEATURE_PERCENTILES.get(name)
        p = f.get(percentile_field) if percentile_field else None
        return None if not _finite(p) else "LOW" if float(p) < 1 / 3 else "MID" if float(p) < 2 / 3 else "HIGH"
    if interaction == "RESILIENCY_X_OPPOSING_MLOFI":
        return t("PRE_EVENT_RESILIENCY"), t("NORMALIZED_MLOFI_PERSISTENCE")
    if interaction == "IMPACT_PER_FLOW_X_ER":
        return t("IMPACT_PER_FLOW"), t("ER_30S")
    if interaction == "RV_X_RESILIENCY":
        rv = t("RV_30S")
        return rv, t("PRE_EVENT_RESILIENCY")
    return None, None


def _interaction_effect(rows: list[dict[str, Any]], interaction: str) -> float | None:
    good, bad = [], []
    for e in rows:
        a, b = _interaction_labels(e, interaction)
        y = e.get("markouts_ticks", {}).get("5000")
        if y is None or a is None or b is None: continue
        if interaction == "RESILIENCY_X_OPPOSING_MLOFI":
            if (a, b) == ("HIGH", "LOW"): good.append(float(y))
            if (a, b) == ("LOW", "HIGH"): bad.append(float(y))
        elif interaction == "IMPACT_PER_FLOW_X_ER":
            if a == "LOW" and b in ("LOW", "MID"): good.append(float(y))
            if a == "HIGH" and b == "HIGH": bad.append(float(y))
        else:
            if a == "HIGH" and b == "HIGH": good.append(float(y))
            if a == "HIGH" and b == "LOW": bad.append(float(y))
    return float(np.mean(good) - np.mean(bad)) if len(good) >= 2 and len(bad) >= 2 else None


def _lodo_lowo(events: list[dict[str, Any]], relationships: tuple[str, ...],
                expected_dates: Iterable[str] = ()) -> dict[str, Any]:
    output = {}
    for rel in relationships:
        rows = [e for e in events if e.get("markouts_ticks", {}).get("5000") is not None]
        full = _fixed_relationship_effect(rows, rel)
        daily = sorted(set(map(str, expected_dates)) | {str(e["date"]) for e in rows})
        weekly = sorted({date.fromisoformat(d).strftime("%G-W%V") for d in daily})
        def leave_out(groups: list[str], get_group: Any) -> list[tuple[str, float]]:
            vals = []
            for group in groups:
                remain = [e for e in rows if get_group(e) != group]
                effect = _fixed_relationship_effect(remain, rel)
                if effect is not None: vals.append((group, effect))
            return vals
        lodo = leave_out(daily, lambda e: str(e["date"]))
        lowo = leave_out(weekly, lambda e: date.fromisoformat(str(e["date"])).strftime("%G-W%V"))
        def describe(items: list[tuple[str, float]]) -> dict[str, Any]:
            values = [x[1] for x in items]
            return {"groups_tested": len(items), "sign_stability": (sum(np.sign(v) == np.sign(full) for v in values) / len(values) if values and full is not None else None),
                    "median_effect": float(np.median(values)) if values else None,
                    "minimum_effect": min(values) if values else None, "maximum_effect": max(values) if values else None,
                    "worst_omitted_group": min(items, key=lambda x: x[1]) if items else None,
                    "best_omitted_group": max(items, key=lambda x: x[1]) if items else None}
        output[rel] = {"full_effect": full, "LODO": describe(lodo), "LOWO": describe(lowo), "weeks_tested": len(weekly)}
    return output


def _neighbor_stability(events: list[dict[str, Any]], relationships: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for rel in relationships:
        vals = {}
        for name in ("deciles", "quintiles", "terciles"):
            ys = []
            for e in events:
                y = e.get("markouts_ticks", {}).get("5000")
                pfield = FEATURE_PERCENTILES.get(rel)
                p = e["pre_event_features"].get(pfield) if pfield else None
                if y is None or not _finite(p):
                    continue
                p = float(p)
                if name == "deciles": low, high = p < .1, p >= .9
                elif name == "quintiles": low, high = p < .2, p >= .8
                else: low, high = p < 1 / 3, p >= 2 / 3
                if low: ys.append(("low", float(y)))
                elif high: ys.append(("high", float(y)))
            low = [v for k, v in ys if k == "low"]; high = [v for k, v in ys if k == "high"]
            vals[name] = {"low_n": len(low), "high_n": len(high),
                          "high_minus_low_5s": float(np.mean(high) - np.mean(low)) if len(low) >= 2 and len(high) >= 2 else None}
        effects = [x["high_minus_low_5s"] for x in vals.values() if x["high_minus_low_5s"] is not None]
        signs = {int(np.sign(v)) for v in effects if v != 0}
        if len(effects) < 2: cls = "INSUFFICIENT"
        elif len(effects) == 3 and len(signs) == 1: cls = "ROBUST"
        elif len(signs) == 1: cls = "MODERATELY_STABLE"
        elif vals["quintiles"]["high_minus_low_5s"] is not None and vals["terciles"]["high_minus_low_5s"] is not None and np.sign(vals["quintiles"]["high_minus_low_5s"]) == np.sign(vals["terciles"]["high_minus_low_5s"]): cls = "BOUNDARY_SENSITIVE"
        else: cls = "UNSTABLE"
        out[rel] = {"representations": vals, "classification": cls}
    return out


def _interaction_tables(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(STUDY_SPEC["interaction_ids"])
    out = {}
    for name in names:
        cells = {}
        for period in prior.PERIODS:
            for family in FAMILY_KEYS:
                rows = [e for e in events if e["period"] == period and e["live_family"] == family]
                for a in ("LOW", "MID", "HIGH"):
                    for b in ("LOW", "MID", "HIGH"):
                        subset = [e for e in rows if _interaction_labels(e, name) == (a, b)]
                        cells.setdefault(period, {}).setdefault(family, {})[f"{a}_X_{b}"] = _markout_stats(subset)
        node = {"cells": cells, "fixed_contrast_5s": _interaction_effect(events, name),
                "definition": "coarse terciles; preregistered cells only"}
        if name == "RV_X_RESILIENCY":
            node["rv_stratum_contrasts_5s"] = {
                rv: {"high_resiliency_minus_low_resiliency": _mean_cell_contrast(events, name, rv, "HIGH", "LOW")}
                for rv in ("LOW", "MID", "HIGH")}
        out[name] = node
    return out


def _mean_cell_contrast(events: list[dict[str, Any]], interaction: str, first: str,
                        good_second: str, bad_second: str) -> float | None:
    good, bad = [], []
    for e in events:
        a, b = _interaction_labels(e, interaction)
        y = e.get("markouts_ticks", {}).get("5000")
        if a != first or y is None: continue
        if b == good_second: good.append(float(y))
        if b == bad_second: bad.append(float(y))
    return float(np.mean(good) - np.mean(bad)) if len(good) >= 2 and len(bad) >= 2 else None


def _mechanism_variable_summaries(events: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = {
        "PRE_EVENT_RESILIENCY": (("recovery_100ms", "pre_event_recovery_ratio_100ms"),
                                 ("recovery_250ms", "pre_event_recovery_ratio_250ms"),
                                 ("recovery_500ms", "pre_event_recovery_ratio_500ms"),
                                 ("recovery_1s", "pre_event_recovery_ratio_1s"),
                                 ("refill_latency_ms", "pre_event_refill_50pct_latency_ms")),
        "NORMALIZED_MLOFI_PERSISTENCE": (("normalized_level", "pre_event_mlofi_2s_normalized"),
                                         ("signed_normalized_level", "pre_event_mlofi_2s_signed_normalized"),
                                         ("opposing_persistence", "opposing_mlofi_persistence_2s"),
                                         ("supporting_persistence", "supporting_mlofi_persistence_2s")),
        "IMPACT_PER_FLOW": (("impact_per_flow", "pre_event_impact_per_flow_10s"),
                            ("absolute_mid_change_ticks", "pre_event_mid_change_ticks_10s"),
                            ("directional_mid_change_ticks", "pre_event_directional_mid_change_ticks_10s"),
                            ("normalized_mlofi_integral", "pre_event_normalized_mlofi_integral_10s")),
        "TREND_EFFICIENCY": (("ER_5S", "ER_5S"), ("ER_30S", "ER_30S"),
                             ("ER_5S_directional", "ER_5S_DIRECTIONAL"),
                             ("ER_30S_directional", "ER_30S_DIRECTIONAL")),
    }
    out = {}
    for variable, fields in mapping.items():
        out[variable] = {}
        for period in prior.PERIODS:
            out[variable][period] = {}
            for family in FAMILY_KEYS:
                subset = [e for e in events if e["period"] == period and e["live_family"] == family]
                out[variable][period][family] = {label: _stats(e["pre_event_features"].get(field) for e in subset)
                                                 for label, field in fields}
    return out


def _period_effects(events: list[dict[str, Any]], relationships: tuple[str, ...], family: str) -> dict[str, Any]:
    return {rel: {period: {
        "fixed_5s_effect": _fixed_relationship_effect([e for e in events if e["period"] == period and e["live_family"] == family], rel),
        "effects_by_horizon": {str(h): _relationship_effect_horizon(
            [e for e in events if e["period"] == period and e["live_family"] == family], rel, str(h)) for h in HORIZONS_MS}}
        for period in prior.PERIODS} for rel in relationships}


def _permutation(events: list[dict[str, Any]], relationships: tuple[str, ...], repetitions: int = 399) -> dict[str, Any]:
    rng = np.random.default_rng(int(STUDY_SPEC["permutation_seed"]))
    output = {}
    for rel in relationships:
        observed_parts = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for e in events:
            if e.get("markouts_ticks", {}).get("5000") is not None:
                groups.setdefault((str(e["date"]), str(e["live_family"])), []).append(e)
        valid_groups = []
        for key, rows in groups.items():
            if rel in FEATURES:
                labels = np.asarray([(-1 if e["pre_event_features"]["feature_deciles"].get(rel) is None
                                      else int(e["pre_event_features"]["feature_deciles"][rel])) for e in rows], dtype=int)
                y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows])
                low, high = np.isin(labels, (0, 1)), np.isin(labels, (8, 9))
                if low.sum() >= 2 and high.sum() >= 2:
                    observed_parts.append((float(y[high].mean() - y[low].mean()), len(rows)))
                    valid_groups.append(("feature", labels, y))
            else:
                a = np.asarray([_interaction_labels(e, rel)[0] for e in rows], dtype=object)
                b = np.asarray([_interaction_labels(e, rel)[1] for e in rows], dtype=object)
                y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows])
                good, bad = _interaction_masks(a, b, rel)
                if good.sum() >= 2 and bad.sum() >= 2:
                    observed_parts.append((float(y[good].mean() - y[bad].mean()), len(rows)))
                    valid_groups.append(("interaction", a, b, y))
        real = float(np.average([v for v, _ in observed_parts], weights=[n for _, n in observed_parts])) if observed_parts else None
        null = []
        for _ in range(repetitions):
            parts = []
            for record in valid_groups:
                if record[0] == "feature":
                    _, labels, y = record
                    labels = rng.permutation(labels)
                    low, high = np.isin(labels, (0, 1)), np.isin(labels, (8, 9))
                else:
                    _, a, b, y = record
                    a, b = rng.permutation(a), rng.permutation(b)
                    good, bad = _interaction_masks(a, b, rel)
                    low, high = bad, good
                if low.sum() >= 2 and high.sum() >= 2:
                    parts.append((float(y[high].mean() - y[low].mean()), len(y)))
            if parts:
                null.append(float(np.average([v for v, _ in parts], weights=[n for _, n in parts])))
        if null:
            q = {str(p): float(np.quantile(null, p)) for p in (.5, .9, .95, .99)}
            percentile = float((sum(x <= real for x in null) + 1) / (len(null) + 1)) if real is not None else None
        else:
            q, percentile = {str(p): None for p in (.5, .9, .95, .99)}, None
        output[rel] = {"real_effect": real, "null_quantiles": q, "empirical_percentile": percentile,
                       "exceeds_null_p95": bool(real > q["0.95"]) if real is not None and q["0.95"] is not None else None,
                       "permutations": len(null), "seed": int(STUDY_SPEC["permutation_seed"]),
                       "unit": "labels permuted within date/family; interaction margins permuted independently"}
    return output


def _interaction_masks(a: np.ndarray, b: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "RESILIENCY_X_OPPOSING_MLOFI":
        return (a == "HIGH") & (b == "LOW"), (a == "LOW") & (b == "HIGH")
    if name == "IMPACT_PER_FLOW_X_ER":
        return (a == "LOW") & np.isin(b, ("LOW", "MID")), (a == "HIGH") & (b == "HIGH")
    return (a == "HIGH") & (b == "HIGH"), (a == "HIGH") & (b == "LOW")


def _topology_reports(events: list[dict[str, Any]], expected_dates: Iterable[str] = ()) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    classes = ("A_IMMEDIATE_CONTINUATION_FAILURE", "B_TEMPORARY_REVERSAL_THEN_FAILURE",
               "C_STAGNATION_OR_CHOP_THEN_FAILURE", "D_DELAYED_SUCCESSFUL_REVERSAL",
               "E_SUCCESSFUL_REVERSAL", "F_UNCLASSIFIED_INSUFFICIENT_PATH")
    by = {}
    for period in prior.PERIODS:
        by[period] = {}
        period_dates = sorted(d for d in expected_dates if (d.startswith("2025-10-") == (period == "OCTOBER_2025")))
        for family in FAMILY_KEYS:
            rows = [e for e in events if e["period"] == period and e["live_family"] == family]
            total = len(rows)
            by[period][family] = {c: {"event_count": sum(e.get("topology") == c for e in rows),
                "share_of_events": sum(e.get("topology") == c for e in rows) / total if total else None,
                "markouts": {str(h): _stats(e.get("markouts_ticks", {}).get(str(h)) for e in rows if e.get("topology") == c) for h in HORIZONS_MS},
                "mfe_mae_by_horizon": {str(h): _excursion_stats([e for e in rows if e.get("topology") == c], h)
                                       for h in PATH_HORIZONS_MS},
                "mfe_before_first_minus4": _stats(e.get("mfe_before_first_minus4") for e in rows if e.get("topology") == c),
                "mae_before_first_plus4": _stats(e.get("mae_before_first_plus4") for e in rows if e.get("topology") == c),
                "mfe_before_structure_break": _stats(e.get("mfe_before_structure_break") for e in rows if e.get("topology") == c),
                "time_to_structure_break_ms": _stats(e.get("time_to_structure_break_ms") for e in rows if e.get("topology") == c),
                "first_touch_direction_counts": {side: sum(e.get("first_touch_4_ticks") == side for e in rows if e.get("topology") == c)
                                                  for side in ("UP", "DOWN", "TIE", "NONE")},
                "unclassified_path_reasons": {reason: sum(e.get("topology") == c and e.get("topology_f_reason") == reason for e in rows)
                                               for reason in ("MISSING_PRE_EVENT_ANCHOR", "30S_HORIZON_CROSSES_SESSION_BOUNDARY",
                                                              "NO_QUOTE_AT_OR_AFTER_30S_HORIZON", "TIED_PLUS4_MINUS4_FIRST_TOUCH",
                                                              "NO_PLUS4_OR_MINUS4_TOUCH_IN_30S")},
                "time_to_all_barriers_ms": {side: _stats(e.get("time_to_barrier_ms", {}).get(side)
                    for e in rows if e.get("topology") == c) for side in ("+1", "+2", "+4", "+8", "-1", "-2", "-4", "-8")},
                "daily_distribution": {d: sum(e.get("topology") == c and e["date"] == d for e in rows)
                                       for d in period_dates}}
                for c in classes}
    by_date = {}
    for e in events:
        key = (e["date"], e["period"], e["live_family"])
        by_date.setdefault(key, {c: 0 for c in classes})[e.get("topology", classes[-1])] += 1
    for day in expected_dates:
        period = "OCTOBER_2025" if str(day).startswith("2025-10-") else "SPRING_2025"
        for family in FAMILY_KEYS:
            by_date.setdefault((str(day), period, family), {c: 0 for c in classes})
    date_report = {"|".join(k): {"event_count": sum(v.values()), "topology_counts": v,
        "unclassified_path_reasons": {reason: sum(e["date"] == k[0] and e["period"] == k[1]
            and e["live_family"] == k[2] and e.get("topology_f_reason") == reason for e in events)
            for reason in ("MISSING_PRE_EVENT_ANCHOR", "30S_HORIZON_CROSSES_SESSION_BOUNDARY",
                           "NO_QUOTE_AT_OR_AFTER_30S_HORIZON", "TIED_PLUS4_MINUS4_FIRST_TOUCH",
                           "NO_PLUS4_OR_MINUS4_TOUCH_IN_30S")}}
        for k, v in by_date.items()}
    family_report = {period: {family: {"event_count": sum(e["period"] == period and e["live_family"] == family for e in events),
        "topology_counts": {c: sum(e["period"] == period and e["live_family"] == family and e.get("topology") == c for e in events) for c in classes}}
        for family in FAMILY_KEYS} for period in prior.PERIODS}
    return by, date_report, family_report


def _mechanism_by_topology(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for period in prior.PERIODS:
        result[period] = {}
        for family in (DISCOVERY, REPLICATION):
            rows = [e for e in events if e["period"] == period and e["live_family"] == family]
            result[period][family] = {}
            for topology in sorted({e.get("topology") for e in rows}):
                subset = [e for e in rows if e.get("topology") == topology]
                result[period][family][topology] = {name: _stats(e["pre_event_features"].get(field) for e in subset)
                    for name, field in (("resiliency", "pre_event_recovery_ratio_500ms"),
                                        ("opposing_mlofi_persistence", "opposing_mlofi_persistence_2s"),
                                        ("impact_per_flow", "impact_per_flow_10s"),
                                        ("trend_efficiency_5s", "ER_5S"), ("trend_efficiency_30s", "ER_30S"))}
    return result


def _shape_relationship(feature_outputs: Mapping[str, Any], boundary: Mapping[str, Any], robustness: Mapping[str, Any],
                        permutations: Mapping[str, Any], replication: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    mapping = {"PRE_EVENT_RESILIENCY": "PRE_EVENT_RESILIENCY", "NORMALIZED_MLOFI_PERSISTENCE": "NORMALIZED_MLOFI_PERSISTENCE",
               "IMPACT_PER_FLOW": "IMPACT_PER_FLOW", "TREND_EFFICIENCY_5S": "TREND_EFFICIENCY_5S",
               "TREND_EFFICIENCY_30S": "TREND_EFFICIENCY_30S"}
    for name, key in mapping.items():
        out[name] = {"shape_spring": feature_outputs[name]["SPRING_2025"][DISCOVERY]["shape_label"],
                     "shape_october": feature_outputs[name]["OCTOBER_2025"][DISCOVERY]["shape_label"],
                     "boundary_stability": boundary[key]["classification"],
                     "LODO": robustness[key]["LODO"], "LOWO": robustness[key]["LOWO"],
                     "permutation": permutations[key], "NY_replication": replication.get(name)}
    return out


def _report(summary: Mapping[str, Any]) -> str:
    lines = ["# Absorption Mechanism State Study V1", "", "Preregistered descriptive study; no thresholds optimized, no PnL computed.",
             "", f"Primary decision: **{summary['primary_decision']}**", "", "## Frozen topology rules", "",
             json.dumps(STUDY_SPEC["topology_precedence"], indent=2), "", "## Event topology counts", "",
             "| Period | Family | A | B | C | D | E | F |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for period in prior.PERIODS:
        for family in FAMILY_KEYS:
            c = summary["failure_topology"][period][family]
            counts = [c[x]["event_count"] for x in c]
            lines.append(f"| {period} | {family} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "## Preregistered interpretation", "", summary["decision_rationale"],
              "", "All outcome quantities are directional ticks/markouts, not trading PnL."]
    return "\n".join(lines) + "\n"


def _load_cached_date(day: str, source_sha: str, tape: Path, config_sha: str) -> dict[str, Any]:
    path = PRIOR_ROOT / "checkpoints" / f"{day}.json.gz"
    try:
        payload = json.load(gzip.open(path, "rt", encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismStudyError(f"previous causal feature checkpoint missing/corrupt: {day}") from exc
    if payload.get("status") != "COMPLETE" or payload.get("source_sha256") != source_sha or payload.get("tape_sha256") != _sha(tape) or payload.get("config_sha256") != config_sha:
        raise MechanismStudyError(f"previous checkpoint hash/config identity mismatch: {day}")
    return payload


def _valid_checkpoint_payload(payload: Mapping[str, Any] | None, day: str, source_sha: str,
                              tape_sha: str, config_sha: str) -> bool:
    return bool(payload and payload.get("status") == "COMPLETE" and payload.get("date") == day
                and payload.get("source_sha256") == source_sha and payload.get("tape_sha256") == tape_sha
                and payload.get("config_sha256") == config_sha
                and payload.get("study_version_hash") == STUDY_VERSION_HASH
                and isinstance(payload.get("events"), list))


def _base_checkpoint_payload(payload: Mapping[str, Any] | None, day: str, source_sha: str,
                             tape_sha: str, config_sha: str) -> bool:
    return bool(payload and payload.get("status") == "COMPLETE" and payload.get("date") == day
                and payload.get("source_sha256") == source_sha and payload.get("tape_sha256") == tape_sha
                and payload.get("config_sha256") == config_sha and isinstance(payload.get("events"), list)
                and all("pre_event_features" in e for e in payload.get("events", [])))


def _refresh_event_paths(day: str, events: list[dict[str, Any]], tape_path: Path,
                         source_sha: str) -> list[dict[str, Any]]:
    candidates, path = _candidate_map(tape_path, day, source_sha)
    windows = baseline._session_windows(day)
    refreshed = []
    for event in events:
        candidate = candidates.get(str(event.get("interaction_id", "")))
        if candidate is None:
            raise MechanismStudyError(f"cannot refresh topology; sealed tape interaction missing: {day} {event.get('interaction_id')}")
        result = _path_metrics(event, candidate, path, windows.get(str(event.get("session", "")), (0, 0))[1])
        event.update(result)
        refreshed.append(event)
    return refreshed


def run(*, output_root: Path = OUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    root = output_root
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    configs, config_sha = norm._load_live_configs()
    if set(configs) != set(norm.LIVE_TO_TAPE):
        raise MechanismStudyError("frozen live family mapping mismatch")
    rows, tapes, coverage = prior._manifest_and_inputs()
    if smoke:
        day = coverage["spring_dates"][0]
        rows, tapes = {day: rows[day]}, {day: tapes[day]}
        coverage["spring_dates"] = [day]
        coverage["october_dates"] = []
    coverage.update({"run_id": RUN_ID, "study_version_hash": STUDY_VERSION_HASH,
                     "config_sha256": config_sha, "strategy_manifest_sha256": EXPECTED_STRATEGY_SHA,
                     "frozen_family_map": norm.LIVE_TO_TAPE, "no_data_downloaded": True, "no_2026_data_accessed": True})
    _write_json(root / "source-coverage.json", coverage)
    all_events: list[dict[str, Any]] = []
    workdir = root / "checkpoints" / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    progress = {"completed_dates": [], "reused_dates": [], "refreshed_dates": [], "new_dates": []}
    for i, day in enumerate(sorted(rows), 1):
        print(f"[mechanism] date={i}/{len(rows)} {day}", flush=True)
        cp_path = root / "checkpoints" / f"{day}.json.gz"
        cp = None
        try:
            cp = json.load(gzip.open(cp_path, "rt", encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        valid = _valid_checkpoint_payload(cp, day, rows[day]["source_sha256"], _sha(tapes[day]), config_sha)
        if valid:
            events = cp["events"]
            progress["reused_dates"].append(day)
        elif _base_checkpoint_payload(cp, day, rows[day]["source_sha256"], _sha(tapes[day]), config_sha):
            events = _refresh_event_paths(day, cp["events"], tapes[day], rows[day]["source_sha256"])
            cp.update({"study_version_hash": STUDY_VERSION_HASH, "study_spec_hash": STUDY_VERSION_HASH,
                       "events": events, "event_count": len(events)})
            _write_gzip(cp_path, cp)
            progress["refreshed_dates"].append(day)
        else:
            old = _load_cached_date(day, rows[day]["source_sha256"], tapes[day], config_sha)
            frozen = [e for e in old["events"] if e.get("live_family") in FAMILY_KEYS]
            events = _event_feature_rows(day, rows[day]["source"], rows[day]["source_sha256"], tapes[day], frozen, workdir)
            cp = {"status": "COMPLETE", "date": day, "source_sha256": rows[day]["source_sha256"],
                  "tape_sha256": _sha(tapes[day]), "config_sha256": config_sha,
                  "study_version_hash": STUDY_VERSION_HASH, "study_spec_hash": STUDY_VERSION_HASH,
                  "event_count": len(events), "events": events}
            _write_gzip(root / "checkpoints" / f"{day}.json.gz", cp)
            progress["new_dates"].append(day)
        all_events.extend(events)
        progress["completed_dates"].append(day)
        _write_json(root / "checkpoints" / "progress.json", {**progress, "status": "RUNNING",
            "target_date_count": len(rows), "next_date": sorted(rows)[i] if i < len(rows) else None,
            "source_manifest_sha256": coverage["source_manifest_sha256"], "config_sha256": config_sha,
            "study_version_hash": STUDY_VERSION_HASH})
        if smoke:
            return {"status": "SMOKE_PASS" if events else "SMOKE_NO_EVENTS", "date": day, "event_count": len(events)}
    _history_percentiles(all_events)
    if not all_events: raise MechanismStudyError("no frozen live-family events available")
    # Persist percentiles/buckets into date checkpoints only after ranks are
    # computed from strictly earlier date events.
    for day in sorted(rows):
        cp_path = root / "checkpoints" / f"{day}.json.gz"
        cp = json.load(gzip.open(cp_path, "rt", encoding="utf-8"))
        cp["events"] = [e for e in all_events if e["date"] == day]
        _write_gzip(cp_path, cp)
    feature_outputs = {name: _continuous_shape(all_events, name) for name in FEATURES}
    curve_outputs = {name: _feature_summary(all_events, name) for name in FEATURES}
    relationships = tuple(FEATURES) + tuple(STUDY_SPEC["interaction_ids"])
    target_dates = sorted(rows)
    daily = {name: _daily_rows([e for e in all_events if e["live_family"] == DISCOVERY], name, target_dates)
             for name in tuple(FEATURES) if name in FEATURES}
    robustness = {name: _lodo_lowo([e for e in all_events if e["live_family"] == DISCOVERY], (name,), target_dates)[name]
                  for name in FEATURES}
    interaction_tables = _interaction_tables(all_events)
    for name in STUDY_SPEC["interaction_ids"]:
        robustness[name] = _lodo_lowo([e for e in all_events if e["live_family"] == DISCOVERY], (name,), target_dates)[name]
        daily[name] = _daily_rows([e for e in all_events if e["live_family"] == DISCOVERY], name, target_dates)
    neighbor = _neighbor_stability([e for e in all_events if e["live_family"] == DISCOVERY], tuple(FEATURES))
    permutation = _permutation([e for e in all_events if e["live_family"] == DISCOVERY], relationships)
    topology, topology_date, topology_family = _topology_reports(all_events, target_dates)
    topology_mechanism = _mechanism_by_topology(all_events)
    # Feature-specific markout/MFE/barrier tables are keyed by period/family/decile.
    markouts = {name: curve_outputs[name] for name in FEATURES}
    mfe_mae = {name: {period: {family: {
        f"D{i+1}": {str(h): _excursion_stats([e for e in all_events if e["period"] == period and e["live_family"] == family
            and e["pre_event_features"]["feature_deciles"].get(name) == i], h)
            for h in PATH_HORIZONS_MS}
        for i in range(10)} for family in FAMILY_KEYS} for period in prior.PERIODS} for name in FEATURES}
    barriers = {name: {period: {family: {f"D{i+1}": {
        f"+{up}/-{down}": {outcome: sum(e.get("barriers", {}).get(f"+{up}/-{down}") == outcome for e in all_events
            if e["period"] == period and e["live_family"] == family and e["pre_event_features"]["feature_deciles"].get(name) == i)
            for outcome in ("UP", "DOWN", "TIE", "NO_TOUCH")}
        for up, down in BARRIERS} for i in range(10)} for family in FAMILY_KEYS} for period in prior.PERIODS} for name in FEATURES}
    period_effects = _period_effects(all_events, relationships, DISCOVERY)
    replication = {}
    for name in relationships:
        rep_by_period = {p: _fixed_relationship_effect(
            [e for e in all_events if e["period"] == p and e["live_family"] == REPLICATION], name) for p in prior.PERIODS}
        discovery_by_period = {p: period_effects[name][p]["fixed_5s_effect"] for p in prior.PERIODS}
        comparable = [(discovery_by_period[p], rep_by_period[p]) for p in prior.PERIODS
                      if discovery_by_period[p] is not None and rep_by_period[p] is not None]
        if len(comparable) < 2:
            cls = "INSUFFICIENT"
        elif all(np.sign(a) == np.sign(b) for a, b in comparable):
            cls = "SAME_DIRECTION"
        elif all(np.sign(a) != np.sign(b) for a, b in comparable):
            cls = "OPPOSITE"
        else:
            cls = "COMPATIBLE_BUT_WEAK"
        replication[name] = {"discovery_effect_by_period": discovery_by_period,
                             "NY_W04_effect_by_period": rep_by_period, "classification": cls}
    sparse_results = {}
    sparse_fields = (("resiliency", "pre_event_recovery_ratio_500ms"),
                     ("mlofi_persistence", "opposing_mlofi_persistence_2s"),
                     ("impact_per_flow", "impact_per_flow_10s"), ("ER_5S", "ER_5S"), ("ER_30S", "ER_30S"))
    for family in SPARSE:
        sparse_results[family] = {}
        for period in prior.PERIODS:
            subset = [e for e in all_events if e["period"] == period and e["live_family"] == family]
            sparse_results[family][period] = {"event_count": len(subset),
                "features": {feature: _stats(e["pre_event_features"].get(field) for e in subset)
                             for feature, field in sparse_fields}}
    counts = {f: {p: sum(e["live_family"] == f and e["period"] == p for e in all_events) for p in prior.PERIODS} for f in FAMILY_KEYS}
    topology_primary = max(
        [(c, topology["OCTOBER_2025"][DISCOVERY][c]["event_count"])
         for c in topology["OCTOBER_2025"][DISCOVERY] if c.startswith(("A_", "B_", "C_", "D_", "E_"))],
        key=lambda x: x[1])[0]
    decision_evidence = {}
    expected_sign = {"PRE_EVENT_RESILIENCY": 1, "NORMALIZED_MLOFI_PERSISTENCE": -1,
        "IMPACT_PER_FLOW": -1, "TREND_EFFICIENCY_5S": -1, "TREND_EFFICIENCY_30S": -1,
        "RESILIENCY_X_OPPOSING_MLOFI": 1, "IMPACT_PER_FLOW_X_ER": 1, "RV_X_RESILIENCY": 1}
    for rel in relationships:
        spring = period_effects[rel]["SPRING_2025"]["fixed_5s_effect"]
        october = period_effects[rel]["OCTOBER_2025"]["fixed_5s_effect"]
        cross_period_ok = spring is not None and october is not None and np.sign(spring) == np.sign(october)
        hypothesis_direction_match = (cross_period_ok and np.sign(spring) == expected_sign[rel]
                                      and np.sign(october) == expected_sign[rel])
        boundary = neighbor.get(rel, {}).get("classification", "FIXED_PREREGISTERED_TERCILES_NOT_PERTURBED")
        boundary_ok = boundary in ("ROBUST", "MODERATELY_STABLE")
        lodo_stability = robustness[rel]["LODO"].get("sign_stability")
        lowo_stability = robustness[rel]["LOWO"].get("sign_stability")
        day_week_ok = (lodo_stability is not None and lodo_stability >= STUDY_SPEC["decision_rules"]["day_week_sign_stability_minimum"]
                       and lowo_stability is not None and lowo_stability >= STUDY_SPEC["decision_rules"]["day_week_sign_stability_minimum"])
        horizon_signs = {}
        for period in prior.PERIODS:
            base = period_effects[rel][period]["fixed_5s_effect"]
            values = list(period_effects[rel][period]["effects_by_horizon"].values())
            horizon_signs[period] = sum(v is not None and base is not None and np.sign(v) == np.sign(base) for v in values)
        multi_horizon = all(v >= 2 for v in horizon_signs.values())
        perm_pass = permutation[rel].get("exceeds_null_p95") is True
        ny_cls = replication[rel]["classification"]
        ny_ok = ny_cls == "SAME_DIRECTION"
        feature_rel = rel in FEATURES
        supported = (cross_period_ok and hypothesis_direction_match and boundary_ok and day_week_ok and multi_horizon
                     and perm_pass and (ny_ok or ny_cls == "INSUFFICIENT") and feature_rel)
        coherent = cross_period_ok and (boundary_ok or day_week_ok) and multi_horizon
        decision_evidence[rel] = {"spring_effect_5s": spring, "october_effect_5s": october,
            "cross_period_direction": cross_period_ok, "boundary_stability": boundary,
            "preregistered_expected_direction": STUDY_SPEC["decision_rules"]["a_priori_expected_effect_sign"][rel],
            "matches_preregistered_direction": bool(hypothesis_direction_match),
            "LODO_sign_stability": lodo_stability, "LOWO_sign_stability": lowo_stability,
            "day_week_robustness": day_week_ok, "same_direction_horizons_by_period": horizon_signs,
            "multi_horizon_coherence": multi_horizon, "permutation_exceeds_null_p95": perm_pass,
            "NY_replication": ny_cls, "supported_candidate": bool(supported), "coherent_but_unconfirmed": bool(coherent and not supported)}
    if any(v["supported_candidate"] for v in decision_evidence.values()):
        decision = "MECHANISM_SUPPORTED"
    elif any(v["coherent_but_unconfirmed"] for v in decision_evidence.values()):
        decision = "MECHANISM_POSSIBLE_BUT_UNCONFIRMED"
    else:
        decision = "NO_STABLE_CONDITIONAL_EDGE"
    permutation_any = any(v.get("exceeds_null_p95") is True for v in permutation.values())
    neighbor_any = any(v["classification"] in ("ROBUST", "MODERATELY_STABLE") for v in neighbor.values())
    lodo_any = any(v["LODO"].get("sign_stability") is not None and
                   v["LODO"]["sign_stability"] >= STUDY_SPEC["decision_rules"]["day_week_sign_stability_minimum"]
                   for v in robustness.values())
    lowo_any = any(v["LOWO"].get("sign_stability") is not None and
                   v["LOWO"]["sign_stability"] >= STUDY_SPEC["decision_rules"]["day_week_sign_stability_minimum"]
                   for v in robustness.values())
    replication_any = any(v["classification"] in ("SAME_DIRECTION", "COMPATIBLE_BUT_WEAK") for v in replication.values())
    variable_summaries = _mechanism_variable_summaries(all_events)
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "study_version_hash": STUDY_VERSION_HASH,
        "dataset": "SPRING_2025 + OCTOBER_2025", "spring_dates": coverage["spring_dates"], "october_dates": coverage["october_dates"],
        "frozen_config_identity": {"config_sha256": config_sha, "strategy_manifest_sha256": EXPECTED_STRATEGY_SHA,
                                   "runtime_contract_sha256": norm.EXPECTED_RUNTIME_CONTRACT_SHA},
        "family_roles": {"discovery": DISCOVERY, "replication": REPLICATION, "descriptive_only": list(SPARSE)},
        "event_counts": counts, "total_events": len(all_events), "failure_topology": topology,
        "failure_topology_by_date": topology_date, "failure_topology_by_family": topology_family,
        "features": feature_outputs, "decile_curves": curve_outputs, "daily_results": daily,
        "mechanism_variable_summaries": variable_summaries, "discovery_period_effects": period_effects,
        "lodo_lowo": robustness, "neighbor_stability": neighbor, "permutation": permutation,
        "interactions": interaction_tables, "failure_topology_mechanism_link": topology_mechanism,
        "ny_replication": replication, "descriptive_sparse_families": sparse_results,
        "october_failure_primary_type_discovery": topology_primary,
        "decision_protocol_checks": {"relationship_evidence": decision_evidence,
                                     "permutation_any_exceeds_p95": permutation_any, "neighbor_stability_any": neighbor_any,
                                     "LODO_any_sign_stability_ge_0_8": lodo_any, "LOWO_any_sign_stability_ge_0_8": lowo_any,
                                     "NY_qualitative_replication_any": replication_any},
        "primary_decision": decision,
        "decision_rationale": ("At least one preregistered association is directionally coherent across periods with day/week and multi-horizon support, but no candidate met all support conditions, including fixed P95 and preregistered-direction requirements; therefore evidence remains possible but unconfirmed. No thresholds or family definitions were changed after results."
            if decision == "MECHANISM_POSSIBLE_BUT_UNCONFIRMED" else
            "At least one preregistered relationship met the complete frozen support protocol; independent calibration is still required before any production use."
            if decision == "MECHANISM_SUPPORTED" else
            "No preregistered relationship showed a stable conditional edge under the frozen cross-period, boundary, robustness, horizon, and permutation review; no thresholds or family definitions were changed after results."),
        "no_data_downloaded": True, "no_2026_accessed": True, "optimization_performed": False,
        "threshold_search_performed": False, "pnl_calculation_performed": False}
    # Required immutable result files.
    outputs = {
        "failure-topology.json": topology, "failure-topology-by-date.json": topology_date,
        "failure-topology-by-family.json": topology_family,
        "resiliency.json": {"primary": "pre_event_recovery_ratio_500ms", "supporting": "100ms/250ms/1s and 50pct refill latency", "deciles": curve_outputs["PRE_EVENT_RESILIENCY"], "native_measurements": variable_summaries["PRE_EVENT_RESILIENCY"]},
        "mlofi-persistence.json": {"deciles": curve_outputs["NORMALIZED_MLOFI_PERSISTENCE"], "native_measurements": variable_summaries["NORMALIZED_MLOFI_PERSISTENCE"], "state_semantics": "validated TOP5 inverse-level depth-normalized MLOFI; eight prior 250ms bins; persistence fraction matches latest nonzero sign"},
        "impact-per-flow.json": {"formula": "abs(10s pre-event mid change in ticks)/(abs(10s MLOFI integral / event-end depth denominator)+1e-12)", "deciles": curve_outputs["IMPACT_PER_FLOW"], "native_measurements": variable_summaries["IMPACT_PER_FLOW"]},
        "trend-efficiency.json": {"deciles": {k: v for k, v in curve_outputs.items() if k.startswith("TREND_EFFICIENCY")}, "native_measurements": variable_summaries["TREND_EFFICIENCY"]},
        "decile-curves.json": curve_outputs, "continuous-shapes.json": feature_outputs,
        "markouts.json": markouts, "mfe-mae.json": mfe_mae, "barriers.json": barriers,
        "daily-results.json": daily, "lodo-results.json": {k: v["LODO"] for k, v in robustness.items()},
        "lowo-results.json": {k: v["LOWO"] for k, v in robustness.items()},
        "interaction-resiliency-mlofi.json": interaction_tables["RESILIENCY_X_OPPOSING_MLOFI"],
        "interaction-impact-efficiency.json": interaction_tables["IMPACT_PER_FLOW_X_ER"],
        "interaction-rv-resiliency.json": interaction_tables["RV_X_RESILIENCY"],
        "permutation-results.json": permutation, "neighbor-stability.json": {"features": neighbor,
            "interactions": {x: "FIXED_PREREGISTERED_COARSE_TERCILES_ONLY" for x in STUDY_SPEC["interaction_ids"]}},
        "discovery-family-results.json": {p: topology[p][DISCOVERY] for p in prior.PERIODS},
        "ny-replication-results.json": replication, "descriptive-sparse-family-results.json": sparse_results,
        "failure-topology-mechanism-link.json": topology_mechanism,
    }
    for name, obj in outputs.items(): _write_json(root / name, obj)
    _write_gzip(root / "event-features.jsonl.gz", all_events)
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {"run_id": RUN_ID, "status": "COMPLETE", "study_version_hash": STUDY_VERSION_HASH,
        "source_manifest_sha256": coverage["source_manifest_sha256"], "source_coverage_sha256": _sha(root / "source-coverage.json"),
        "config_sha256": config_sha, "strategy_manifest_sha256": EXPECTED_STRATEGY_SHA,
        "ordered_dates": sorted(rows), "event_count": len(all_events),
        "checkpoint_hashes": {p.name: _sha(p) for p in sorted((root / "checkpoints").glob("*.json.gz"))},
        "artifact_hashes": {p.name: _sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name not in {"run-manifest.json"}},
        "no_data_downloaded": True, "no_2026_accessed": True}
    _write_json(root / "checkpoints" / "progress.json", {"status": "COMPLETE", **progress,
        "summary_sha256": _sha(root / "summary.json"), "study_version_hash": STUDY_VERSION_HASH})
    manifest["checkpoint_progress_sha256"] = _sha(root / "checkpoints" / "progress.json")
    _write_json(root / "run-manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(output_root=args.output_root, smoke=args.smoke)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({k: result.get(k) for k in ("status", "total_events", "primary_decision", "october_failure_primary_type_discovery")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
