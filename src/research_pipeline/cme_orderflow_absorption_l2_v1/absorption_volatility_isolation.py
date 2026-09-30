"""Confound-controlled descriptive volatility study for the frozen absorption core.

This module enriches the already sealed 41-session event population from the
Dec/Jan regime study. It never evaluates trades or PnL. All market context is
strictly prior to each interaction; calibration histories are date-expanding.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, time as wall_time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import absorption_regime_study as base


RUN_ID = "CMEOrderflow_ABSORPTION_VOLATILITY_ISOLATION_DEC_JAN_V1"
OUTPUT_RELATIVE = Path("research_runs") / RUN_ID
PRIOR_RELATIVE = Path("research_runs") / base.RUN_ID
EXPECTED_CONFIG_SHA256 = base.EXPECTED_CONFIG_SHA256
SEED = 20260930
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 1000
MIN_HISTORY = 20
MIN_CELL = 20
MIN_CLUSTER_DAYS = 5
MIN_DAILY_SAMPLE = base.MIN_DAILY_SAMPLE
NS = 1_000_000_000
ET = ZoneInfo("America/New_York")
VOL_FEATURES = (
    "rv_10s_ticks", "rv_30s_ticks", "rv_120s_ticks", "rv_300s_ticks",
    "tod_norm_rv_30s", "tod_norm_rv_120s",
)
PRIMARY_FEATURES = ("rv_30s_ticks", "rv_120s_ticks", "tod_norm_rv_30s", "tod_norm_rv_120s")
MARKOUT_KEYS = {ms: f"markout_{ms}ms_ticks" for ms in (250, 500, 1000, 2000, 5000, 10000, 30000)}
EXCURSION_KEYS = {ms: (f"mfe_{ms}ms_ticks", f"mae_{ms}ms_ticks") for ms in (1000, 2000, 5000, 10000, 30000)}
BARRIER_KEYS = ("1_1", "2_2", "4_4", "8_4", "12_6")
TOD_BUCKETS = ("ASIA", "EUROPE", "US_PREOPEN", "CASH_OPEN", "MORNING", "MIDDAY", "AFTERNOON", "CASH_CLOSE")


class VolatilityStudyError(RuntimeError):
    """Raised on source, causality, checkpoint, or frozen-input violations."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temp, path)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _percentile(value: float | None, history: Sequence[float]) -> float | None:
    if value is None or not math.isfinite(float(value)) or len(history) < MIN_HISTORY:
        return None
    a = np.asarray(history, dtype=np.float64)
    return float((np.searchsorted(np.sort(a), float(value), side="right")) / len(a))


def tod_normalize(value: float | None, prior_same_bucket_values: Sequence[float]) -> float | None:
    """Scale RV by the median from strictly prior dates in the same fixed TOD bucket."""
    if value is None or len(prior_same_bucket_values) < MIN_HISTORY:
        return None
    median = float(np.median(np.asarray(prior_same_bucket_values, dtype=np.float64)))
    return float(value) / median if median > 0 else None


def _expanding_bucket(value: float | None, history: Sequence[float], n: int = 5) -> str:
    if value is None or not math.isfinite(float(value)) or len(history) < MIN_HISTORY:
        return "INSUFFICIENT_HISTORY"
    edges = np.quantile(np.asarray(history, dtype=np.float64), np.arange(1, n) / n)
    return f"Q{int(np.searchsorted(edges, float(value), side='right')) + 1}"


def volatility_tercile_label(bucket: str | None) -> str:
    """Translate date-expanding Q1/Q2/Q3 RV terciles to named study states."""
    return {"Q1": "LOW", "Q2": "MEDIUM", "Q3": "HIGH"}.get(str(bucket), "INSUFFICIENT_HISTORY")


def _coarse_vol(value: float | None, history: Sequence[float]) -> str:
    if value is None or not math.isfinite(float(value)) or len(history) < MIN_HISTORY:
        return "INSUFFICIENT_HISTORY"
    p20, p80, p95 = np.quantile(np.asarray(history, dtype=np.float64), (0.20, 0.80, 0.95))
    return "LOW" if value < p20 else "NORMAL" if value < p80 else "HIGH" if value < p95 else "EXTREME"


def _tercile(value: float | None, history: Sequence[float]) -> str:
    if value is None or not math.isfinite(float(value)) or len(history) < MIN_HISTORY:
        return "INSUFFICIENT_HISTORY"
    p33, p67 = np.quantile(np.asarray(history, dtype=np.float64), (1 / 3, 2 / 3))
    return "LOW" if value < p33 else "MEDIUM" if value < p67 else "HIGH"


def tod_bucket(timestamp_ns: int, session: str) -> str:
    """Fixed economic clock buckets; NY wall-time cut points are not fitted."""
    if session == "ASIA":
        return "ASIA"
    dt = datetime.fromtimestamp(timestamp_ns / NS, timezone.utc).astimezone(ET)
    t = dt.timetz().replace(tzinfo=None)
    if t < wall_time(8, 0):
        return "EUROPE"
    if t < wall_time(9, 30):
        return "US_PREOPEN"
    if t < wall_time(10, 0):
        return "CASH_OPEN"
    if t < wall_time(11, 30):
        return "MORNING"
    if t < wall_time(14, 0):
        return "MIDDAY"
    if t < wall_time(15, 30):
        return "AFTERNOON"
    return "CASH_CLOSE"


def realized_volatility_ticks(mid_prices: Sequence[float], tick_points: float = 0.25) -> float:
    """RV = sqrt(sum((mid[i]-mid[i-1])/ES_tick)^2), within the fixed window."""
    values = np.asarray(mid_prices, dtype=np.float64)
    if values.size < 2:
        return 0.0
    changes = np.diff(values) / tick_points
    return float(np.sqrt(np.dot(changes, changes)))


def price_velocity_and_efficiency(mid_prices: Sequence[float], duration_seconds: float,
                                  reversal_sign: int, tick_points: float = 0.25) -> dict[str, float | None]:
    values = np.asarray(mid_prices, dtype=np.float64)
    if values.size < 2 or duration_seconds <= 0:
        return {"velocity_ticks_per_second": None, "directional_velocity_ticks_per_second": None,
                "trend_efficiency": None}
    changes = np.diff(values) / tick_points
    net = float(values[-1] - values[0]) / tick_points
    path = float(np.abs(changes).sum())
    efficiency = abs(net) / path if path > 0 else 0.0
    return {"velocity_ticks_per_second": net / duration_seconds,
            "directional_velocity_ticks_per_second": reversal_sign * net / duration_seconds,
            "trend_efficiency": efficiency}


def abnormal_market_state(rv_state: str, *, spread_abnormal: bool, depth_abnormal: bool,
                         intensity_abnormal: bool) -> tuple[str, int]:
    count = int(spread_abnormal) + int(depth_abnormal) + int(intensity_abnormal)
    state = "ABNORMAL_STRESS_STATE" if rv_state == "EXTREME" and count > 0 else "NORMAL_MARKET_STATE"
    return state, count


def _dist(values: Sequence[float | int | None]) -> dict[str, Any]:
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if not len(x):
        return {"count": 0, "mean": None, "median": None, "trimmed_mean_5pct": None,
                "p25": None, "p75": None, "std": None, "standard_error": None,
                "ci95": [None, None], "positive_fraction": None, "negative_fraction": None,
                "zero_fraction": None}
    n = len(x); mean = float(x.mean()); sd = float(x.std(ddof=1)) if n > 1 else 0.0
    ordered = np.sort(x); trim = int(n * 0.05)
    trimmed = ordered[trim:n - trim] if trim and 2 * trim < n else ordered
    se = sd / math.sqrt(n)
    return {"count": int(n), "mean": mean, "median": float(np.median(x)),
            "trimmed_mean_5pct": float(trimmed.mean()), "p25": float(np.quantile(x, .25)),
            "p75": float(np.quantile(x, .75)), "std": sd, "standard_error": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "positive_fraction": float(np.mean(x > 0)), "negative_fraction": float(np.mean(x < 0)),
            "zero_fraction": float(np.mean(x == 0))}


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"event_count": len(rows), "active_dates": len({str(r['date']) for r in rows})}
    result["markouts"] = {str(ms): _dist([r.get(key) for r in rows]) for ms, key in MARKOUT_KEYS.items()}
    result["mfe_mae"] = {}
    for ms, (mfe, mae) in EXCURSION_KEYS.items():
        mfe_values = [float(r[mfe]) for r in rows if r.get(mfe) is not None]
        mae_values = [float(r[mae]) for r in rows if r.get(mae) is not None]
        result["mfe_mae"][str(ms)] = {
            "mfe": _dist(mfe_values), "mae_adverse_magnitude": _dist(mae_values),
            "probabilities": {"mfe_ge_ticks": {str(t): (sum(x >= t for x in mfe_values) / len(mfe_values) if mfe_values else None)
                                                        for t in (1, 2, 4, 8)},
                              "mae_ge_ticks": {str(t): (sum(x >= t for x in mae_values) / len(mae_values) if mae_values else None)
                                                        for t in (1, 2, 4, 8)}}}
    result["barriers"] = {}
    for pair in BARRIER_KEYS:
        subset = [r for r in rows if r.get(f"barrier_{pair}_outcome") in {"FAVORABLE_FIRST", "ADVERSE_FIRST", "UNRESOLVED"}]
        result["barriers"][pair] = {
            "sample_count": len(subset),
            "favorable_first_probability": (sum(r[f"barrier_{pair}_outcome"] == "FAVORABLE_FIRST" for r in subset) / len(subset)) if subset else None,
            "adverse_first_probability": (sum(r[f"barrier_{pair}_outcome"] == "ADVERSE_FIRST" for r in subset) / len(subset)) if subset else None,
            "unresolved_probability": (sum(r[f"barrier_{pair}_outcome"] == "UNRESOLVED" for r in subset) / len(subset)) if subset else None,
            "median_favorable_touch_seconds": _dist([r.get(f"barrier_{pair}_seconds") for r in subset if r.get(f"barrier_{pair}_outcome") == "FAVORABLE_FIRST"])["median"],
            "median_adverse_touch_seconds": _dist([r.get(f"barrier_{pair}_seconds") for r in subset if r.get(f"barrier_{pair}_outcome") == "ADVERSE_FIRST"])["median"],
        }
    return result


def _daily_bucket(rows: Sequence[Mapping[str, Any]], bucket_field: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["date"]), str(row.get(bucket_field, "INSUFFICIENT_HISTORY")))].append(row)
    out: dict[str, Any] = {}
    for (day, bucket), values in sorted(grouped.items()):
        markouts = {str(ms): _dist([r.get(key) for r in values]) for ms, key in MARKOUT_KEYS.items()}
        daily_effect = markouts["2000"]["mean"]
        out.setdefault(bucket, {})[day] = {"event_count": len(values), "markouts": markouts,
                                            "median_2s": markouts["2000"]["median"],
                                            "mfe_5s": _dist([r.get("mfe_5000ms_ticks") for r in values]),
                                            "mae_5s": _dist([r.get("mae_5000ms_ticks") for r in values]),
                                            "barriers_30s": _group_summary(values)["barriers"],
                                            "daily_effect_2s": daily_effect}
    for bucket, days in out.items():
        minimum = MIN_DAILY_SAMPLE
        eligible = [v for v in days.values() if v["event_count"] >= minimum and v["daily_effect_2s"] is not None]
        effects = [v["daily_effect_2s"] for v in eligible]
        out[bucket] = {"dates": days, "stability": {
            "positive_dates": sum(v["daily_effect_2s"] > 0 for v in eligible),
            "negative_dates": sum(v["daily_effect_2s"] < 0 for v in eligible),
            "flat_dates": sum(v["daily_effect_2s"] == 0 for v in eligible),
            "insufficient_dates": len(days) - len(eligible),
            "median_daily_effect": _dist(effects)["median"],
            "p25_daily_effect": _dist(effects)["p25"], "p75_daily_effect": _dist(effects)["p75"],
            "eligible_day_count": len(eligible), "minimum_events_per_day": minimum}}
    return out


def _summary_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "UNKNOWN"))].append(row)
    return {key: _group_summary(values) for key, values in sorted(grouped.items())}


def _qcontrast(rows: Sequence[Mapping[str, Any]], bucket_field: str, ms: int) -> dict[str, Any]:
    low = [r for r in rows if r.get(bucket_field) == "Q1" and r.get(MARKOUT_KEYS[ms]) is not None]
    high = [r for r in rows if r.get(bucket_field) == "Q5" and r.get(MARKOUT_KEYS[ms]) is not None]
    return {"q1_count": len(low), "q5_count": len(high), "q1_mean": _dist([r[MARKOUT_KEYS[ms]] for r in low])["mean"],
            "q5_mean": _dist([r[MARKOUT_KEYS[ms]] for r in high])["mean"],
            "q5_minus_q1": (_dist([r[MARKOUT_KEYS[ms]] for r in high])["mean"] - _dist([r[MARKOUT_KEYS[ms]] for r in low])["mean"])
            if low and high else None}


def daily_q5_q1_contrast(rows: Sequence[Mapping[str, Any]], bucket_field: str, ms: int,
                         minimum_per_bucket: int = MIN_DAILY_SAMPLE) -> dict[str, Any]:
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row["date"])].append(row)
    daily: dict[str, dict[str, Any]] = {}
    effects = []
    positive = negative = flat = insufficient = 0
    for day, members in sorted(by_day.items()):
        low = [float(r[MARKOUT_KEYS[ms]]) for r in members
               if r.get(bucket_field) == "Q1" and r.get(MARKOUT_KEYS[ms]) is not None]
        high = [float(r[MARKOUT_KEYS[ms]]) for r in members
                if r.get(bucket_field) == "Q5" and r.get(MARKOUT_KEYS[ms]) is not None]
        if len(low) < minimum_per_bucket or len(high) < minimum_per_bucket:
            daily[day] = {"q1_count": len(low), "q5_count": len(high), "q5_minus_q1": None,
                          "status": "INSUFFICIENT"}
            insufficient += 1
            continue
        effect = float(np.mean(high) - np.mean(low)); effects.append(effect)
        positive += effect > 0; negative += effect < 0; flat += effect == 0
        daily[day] = {"q1_count": len(low), "q5_count": len(high), "q1_mean": float(np.mean(low)),
                      "q5_mean": float(np.mean(high)), "q5_minus_q1": effect, "status": "PASS"}
    dist = _dist(effects)
    return {"dates": daily, "positive_dates": int(positive), "negative_dates": int(negative),
            "flat_dates": int(flat), "insufficient_dates": int(insufficient),
            "median_daily_effect": dist["median"], "p25_daily_effect": dist["p25"],
            "p75_daily_effect": dist["p75"], "eligible_date_count": len(effects),
            "minimum_events_per_bucket_per_date": minimum_per_bucket}


def shape_classification(means: Sequence[float | None], plateau_ticks: float = 0.10) -> str:
    values = [float(x) for x in means if x is not None and math.isfinite(float(x))]
    if len(values) < 4:
        return "INSUFFICIENT"
    if all(b > a for a, b in zip(values, values[1:])):
        return "MONOTONIC_INCREASING"
    if all(b < a for a, b in zip(values, values[1:])):
        return "MONOTONIC_DECREASING"
    if max(values) - min(values) <= plateau_ticks:
        return "PLATEAU"
    peak = int(np.argmax(values)); trough = int(np.argmin(values))
    if 0 < peak < len(values) - 1 and values[0] < values[peak] and values[-1] < values[peak]:
        return "INVERTED_U"
    if 0 < trough < len(values) - 1 and values[0] > values[trough] and values[-1] > values[trough]:
        return "U_SHAPED"
    if len(values) >= 5 and all(abs(values[i + 1] - values[i]) <= plateau_ticks for i in range(1, 4)):
        return "PLATEAU"
    return "NOISY"


def effect_size(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "INSUFFICIENT"
    x = abs(float(value))
    if x < 0.10: return "TINY"
    if x < 0.25: return "SMALL"
    if x < 0.50: return "MODEST"
    if x <= 1.00: return "MEANINGFUL"
    return "STRONG"


def clustered_bootstrap(rows: Sequence[Mapping[str, Any]], bucket_field: str, contrast: str,
                        ms: int, *, seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    by_day: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        val = r.get(MARKOUT_KEYS[ms]); bucket = str(r.get(bucket_field, ""))
        if val is not None and bucket in {"Q1", "Q5", "LOW", "HIGH", "EXTREME"}:
            by_day[str(r["date"])][bucket].append(float(val))
    if contrast == "Q5_Q1": pair = ("Q5", "Q1")
    elif contrast == "HIGH_LOW": pair = ("HIGH", "LOW")
    elif contrast == "HIGH_EXTREME": pair = ("HIGH", "EXTREME")
    else: raise ValueError(contrast)
    effects = [float(np.mean(g[pair[0]]) - np.mean(g[pair[1]])) for g in by_day.values()
               if len(g[pair[0]]) and len(g[pair[1]])]
    if len(effects) < MIN_CLUSTER_DAYS:
        return {"contrast": contrast, "horizon_ms": ms, "cluster_count": len(effects), "status": "INSUFFICIENT", "ci95": [None, None]}
    arr = np.asarray(effects, dtype=np.float64); rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        samples[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return {"contrast": contrast, "horizon_ms": ms, "cluster_count": len(arr), "observed_median_daily_effect": float(np.median(arr)),
            "observed_mean_daily_effect": float(np.mean(arr)), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "replicates": replicates, "seed": seed, "status": "PASS"}


def permutation_test(rows: Sequence[Mapping[str, Any]], bucket_field: str, ms: int,
                     *, seed: int = SEED, replicates: int = PERMUTATION_REPLICATES,
                     contrast: str = "Q5_Q1") -> dict[str, Any]:
    if contrast == "Q5_Q1":
        left_label, right_label, allowed = "Q5", "Q1", {"Q1", "Q2", "Q3", "Q4", "Q5"}
    elif contrast == "HIGH_LOW":
        left_label, right_label, allowed = "HIGH", "LOW", {"LOW", "NORMAL", "HIGH", "EXTREME"}
    else:
        raise ValueError(contrast)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get(MARKOUT_KEYS[ms]) is not None and row.get(bucket_field) in allowed:
            groups[(str(row["date"]), str(row["session"]))].append(i)
    vals = np.asarray([float(r[MARKOUT_KEYS[ms]]) if r.get(MARKOUT_KEYS[ms]) is not None else np.nan for r in rows], dtype=np.float64)
    labels = np.asarray([str(r.get(bucket_field, "")) for r in rows], dtype=object)
    left = np.flatnonzero((labels == left_label) & np.isfinite(vals))
    right = np.flatnonzero((labels == right_label) & np.isfinite(vals))
    if not len(left) or not len(right):
        return {"status": "INSUFFICIENT", "contrast": contrast, "horizon_ms": ms, "replicates": replicates, "seed": seed}
    observed = float(vals[left].mean() - vals[right].mean())
    strata = [(np.asarray(idx, dtype=np.int64), labels[idx], vals[idx]) for idx in groups.values() if len(idx) >= 2]
    mutable = {int(i) for idx, _, _ in strata for i in idx}
    fixed = np.asarray([i for i in range(len(rows)) if i not in mutable and np.isfinite(vals[i])], dtype=np.int64)
    fixed_left = fixed[labels[fixed] == left_label]; fixed_right = fixed[labels[fixed] == right_label]
    fixed_left_sum = float(vals[fixed_left].sum()); fixed_right_sum = float(vals[fixed_right].sum())
    fixed_left_count = int(len(fixed_left)); fixed_right_count = int(len(fixed_right))
    rng = np.random.default_rng(seed); extremes = 0
    for _ in range(replicates):
        left_sum = fixed_left_sum; right_sum = fixed_right_sum
        left_count = fixed_left_count; right_count = fixed_right_count
        for _, stratum_labels, stratum_values in strata:
            shuffled = rng.permutation(stratum_labels)
            left_mask = shuffled == left_label; right_mask = shuffled == right_label
            left_count += int(np.count_nonzero(left_mask)); right_count += int(np.count_nonzero(right_mask))
            left_sum += float(stratum_values[left_mask].sum()); right_sum += float(stratum_values[right_mask].sum())
        if left_count and right_count and abs(left_sum / left_count - right_sum / right_count) >= abs(observed):
            extremes += 1
    return {"status": "PASS", "contrast": contrast, "horizon_ms": ms, "observed_left_minus_right": observed,
            "two_sided_randomization_p": (extremes + 1) / (replicates + 1), "extreme_replicates": extremes,
            "replicates": replicates, "seed": seed, "strata": "trading_date x session", "stratum_count": len(strata)}


def event_cluster_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("date"), row.get("session"), row.get("direction"), row.get("interaction_start_ns"),
            row.get("interaction_end_ns"), row.get("interaction_end_price"))


def deduplicate_event_clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (str(r.get("family")), str(r.get("event_id")))):
        chosen.setdefault(event_cluster_key(row), row)
    return [dict(chosen[key]) for key in sorted(chosen, key=lambda x: tuple(str(y) for y in x))]


def event_overlap_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = {
        "same_timestamp": lambda r: (r.get("date"), r.get("interaction_start_ns")),
        "same_price_direction": lambda r: (r.get("date"), r.get("interaction_end_price"), r.get("direction")),
        "same_interaction_geometry": event_cluster_key,
    }
    result: dict[str, Any] = {"raw_event_count": len(rows)}
    for name, func in keys.items():
        counts = Counter(func(row) for row in rows)
        result[name] = {"overlap_group_count": sum(n > 1 for n in counts.values()),
                        "events_in_overlap_groups": sum(n for n in counts.values() if n > 1),
                        "duplicate_excess_rows": sum(n - 1 for n in counts.values() if n > 1)}
    result["deduplicated_exact_geometry_count"] = len(deduplicate_event_clusters(rows))
    result["deduplication_key"] = ["date", "session", "direction", "interaction_start_ns", "interaction_end_ns", "interaction_end_price"]
    return result


def _load_market_day(source_paths: Sequence[Path], windows: Mapping[str, Sequence[int]], day: str) -> dict[str, dict[str, np.ndarray]]:
    """Stream one date of native MBP-10, sharing the frozen adapter's canonical math."""
    from databento import DBNStore

    names = base.SESSION_NAMES
    fields = ("ts", "mid_points", "bid_depth", "ask_depth", "spread_ticks", "activity_ts",
              "trade_count", "trade_contracts", "book_update_count", "record_count")
    pieces: dict[str, dict[str, list[np.ndarray]]] = {name: {k: [] for k in fields} for name in names}
    previous = None; previous_executable = False; previous_ts = None; total = 0
    for path in source_paths:
        store = DBNStore.from_file(path)
        try:
            iterator = store.to_ndarray(count=100_000)
            for batch in iterator:
                ts = batch["ts_recv"].astype(np.int64, copy=False)
                if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
                    raise VolatilityStudyError(f"timestamp regression in {path}")
                if previous_ts is not None and len(ts) and int(ts[0]) < previous_ts:
                    raise VolatilityStudyError(f"timestamp regression across partitions: {path}")
                raw, previous, previous_executable, previous_ts = base._batch_market_columns(
                    batch, previous, previous_executable, previous_ts)
                codes = base._session_code_for(raw["ts"], windows)
                action = batch["action"]
                size = batch["size"].astype(np.float64, copy=False)
                trade = action == b"T"
                book = np.isin(action, (b"A", b"C", b"M"))
                bpx = np.column_stack([batch[f"bid_px_{i:02d}"] for i in range(10)])
                apx = np.column_stack([batch[f"ask_px_{i:02d}"] for i in range(10)])
                bsz = np.column_stack([batch[f"bid_sz_{i:02d}"] for i in range(10)])
                asz = np.column_stack([batch[f"ask_sz_{i:02d}"] for i in range(10)])
                best_bid = np.max(np.where((bpx > 0) & (bsz > 0), bpx, 0), axis=1)
                best_ask = np.min(np.where((apx > 0) & (asz > 0), apx, np.iinfo(np.int64).max), axis=1)
                spread = (best_ask.astype(np.float64) - best_bid.astype(np.float64)) / base.RAW_PRICE_SCALE / base.TICK_POINTS
                total += len(batch)
                for name, code in base.SESSION_CODES.items():
                    in_session = codes == code
                    take = raw["executable"] & (codes == code)
                    if not np.any(in_session):
                        continue
                    target = pieces[name]
                    if np.any(take):
                        target["ts"].append(raw["ts"][take].copy())
                        # Existing canonical field is measured in half ticks
                        # (0.125 ES points); convert once to ES points.
                        target["mid_points"].append((raw["mid_half_ticks"][take].astype(np.float64) / 8.0).copy())
                        target["bid_depth"].append(raw["bid_depth"][take].astype(np.float64, copy=True))
                        target["ask_depth"].append(raw["ask_depth"][take].astype(np.float64, copy=True))
                        target["spread_ticks"].append(spread[take].copy())
                    target["activity_ts"].append(raw["ts"][in_session].copy())
                    target["trade_count"].append(trade[in_session].astype(np.int32, copy=True))
                    target["trade_contracts"].append(np.where(trade, size, 0)[in_session].astype(np.float64, copy=True))
                    target["book_update_count"].append(book[in_session].astype(np.int32, copy=True))
                    target["record_count"].append(np.ones(int(in_session.sum()), dtype=np.int32))
                if total and total % 5_000_000 < len(batch):
                    print(f"VOLATILITY_DATE_PROGRESS={day} records={total}", flush=True)
            close = getattr(store, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
    output: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        output[name] = {}
        for key, arrays in pieces[name].items():
            output[name][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)
        ts = output[name]["ts"]
        if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
            raise VolatilityStudyError(f"executable timestamp order invalid {day}/{name}")
        output[name]["activity_prefix_ts"] = output[name]["activity_ts"]
        if len(output[name]["activity_ts"]) > 1 and np.any(output[name]["activity_ts"][1:] < output[name]["activity_ts"][:-1]):
            raise VolatilityStudyError(f"activity timestamp order invalid {day}/{name}")
        output[name]["trade_prefix"] = np.concatenate(([0], np.cumsum(output[name]["trade_count"], dtype=np.int64)))
        output[name]["contracts_prefix"] = np.concatenate(([0.0], np.cumsum(output[name]["trade_contracts"], dtype=np.float64)))
        output[name]["updates_prefix"] = np.concatenate(([0], np.cumsum(output[name]["book_update_count"], dtype=np.int64)))
        output[name]["records_prefix"] = np.concatenate(([0], np.cumsum(output[name]["record_count"], dtype=np.int64)))
    return output


def _prefix_window(prefix: np.ndarray, ts: np.ndarray, start: int, end: int) -> float:
    lo = int(np.searchsorted(ts, start, side="left")); hi = int(np.searchsorted(ts, end, side="left"))
    return float(prefix[hi] - prefix[lo])


def _pre_context(row: Mapping[str, Any], market: Mapping[str, np.ndarray], session_window: Sequence[int]) -> dict[str, Any]:
    start = int(row["interaction_start_ns"])
    clock_bucket = tod_bucket(start, str(row["session"]))
    ts = market["ts"]
    ix = int(np.searchsorted(ts, start, side="left"))
    if ix <= 0:
        return {"context_status": "NO_PRE_EVENT_EXECUTABLE_BOOK", "tod_bucket": clock_bucket}
    sign = base.direction_sign(str(row["direction"]))
    mid = market["mid_points"]
    result: dict[str, Any] = {"context_status": "OK", "context_last_timestamp_ns": int(ts[ix - 1]),
                              "tod_bucket": clock_bucket}
    if int(ts[ix - 1]) >= start:
        raise VolatilityStudyError("lookahead: context timestamp is not strictly pre-event")
    for seconds in (10, 30, 120, 300):
        left = int(np.searchsorted(ts, start - seconds * NS, side="left"))
        # RV uses only mid changes whose two endpoints lie in the fixed pre-event window.
        path = mid[left:ix]
        result[f"rv_{seconds}s_ticks"] = realized_volatility_ticks(path)
    for seconds in (5, 30):
        left = int(np.searchsorted(ts, start - seconds * NS, side="left"))
        path = mid[max(0, left - 1):ix]
        result.update({f"{k}_{seconds}s": v for k, v in price_velocity_and_efficiency(path, seconds, sign).items()})
    # The depth observations are sampled on a fixed 100 ms clock, not weighted by quote-event intensity.
    win_start = int(session_window[0]); t0 = max(win_start, start - 30 * NS)
    grid = np.arange(((t0 + 99_999_999) // 100_000_000) * 100_000_000, start, 100_000_000, dtype=np.int64)
    pos = np.searchsorted(ts, grid, side="right") - 1
    valid = pos >= 0
    bid = market["bid_depth"][pos[valid]]; ask = market["ask_depth"][pos[valid]]
    result["pre_30s_top5_depth_mean"] = float(np.mean(bid + ask)) if len(bid) else None
    result["pre_30s_pressured_top5_depth_mean"] = float(np.mean(bid if sign > 0 else ask)) if len(bid) else None
    result["pre_event_spread_ticks"] = float(market["spread_ticks"][ix - 1])
    for seconds in (5, 30):
        left_ns = start - seconds * NS
        activity_ts = market["activity_prefix_ts"]
        trades = _prefix_window(market["trade_prefix"], activity_ts, left_ns, start)
        contracts = _prefix_window(market["contracts_prefix"], activity_ts, left_ns, start)
        updates = _prefix_window(market["updates_prefix"], activity_ts, left_ns, start)
        records = _prefix_window(market["records_prefix"], activity_ts, left_ns, start)
        result[f"pre_{seconds}s_trades_per_second"] = trades / seconds
        result[f"pre_{seconds}s_contracts_per_second"] = contracts / seconds
        result[f"pre_{seconds}s_book_updates_per_second"] = updates / seconds
        result[f"pre_{seconds}s_quote_records_per_second"] = records / seconds
    result["session_window_start_ns"] = int(session_window[0])
    return result


def _prior_file_index(prior_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    coverage = json.loads((prior_root / "source-coverage.json").read_text(encoding="utf-8"))
    manifest = json.loads((prior_root / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or coverage.get("status") != "PASS":
        raise VolatilityStudyError("prior Dec/Jan regime run or coverage is not COMPLETE/PASS")
    if manifest.get("config_sha256") != EXPECTED_CONFIG_SHA256 or manifest.get("config_sha256") != coverage.get("config_sha256", manifest.get("config_sha256")):
        raise VolatilityStudyError("prior run does not bind the expected canonical frozen config")
    if manifest.get("untouched_oos_accessed") is not False or manifest.get("optimization_performed") is not False:
        raise VolatilityStudyError("prior run metadata violates OOS/optimization boundary")
    sources: dict[str, dict[str, Any]] = {}
    repository_root = prior_root.resolve().parents[1]
    for item in coverage.get("source_input_files", []):
        source_path = Path(item["path"])
        if not source_path.is_absolute():
            source_path = repository_root / source_path
        normalized = {**item, "path": str(source_path.resolve())}
        sources.setdefault(str(item["date"]), {})[str(source_path.resolve())] = normalized
    events: dict[str, dict[str, Any]] = {}
    for day in manifest.get("completed_dates", []):
        day = str(day)
        checkpoint_path = prior_root / "checkpoints" / f"{day}.json"
        event_path = prior_root / "dates" / f"{day}-events.jsonl.gz"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not event_path.is_file() or checkpoint.get("output_sha256") != sha256_file(event_path):
            raise VolatilityStudyError(f"prior event artifact/checkpoint mismatch for {day}")
        events[day] = {"path": event_path, "sha256": sha256_file(event_path), "checkpoint": checkpoint}
    return sources, events


def _source_file_hashes(day: str, paths: Sequence[Path], expected: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        item = expected.get(str(path.resolve()))
        if not item or not path.is_file():
            raise VolatilityStudyError(f"source file absent from verified prior coverage: {path}")
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != str(item["sha256"]):
            raise VolatilityStudyError(f"sealed source size/hash mismatch: {path}")
        records.append({"path": str(path.resolve()), "bytes": int(item["bytes"]), "sha256": str(item["sha256"]),
                        "symbol": item.get("symbol"), "schema": item.get("schema"), "date": day})
    return records


def _enrich_one_date(day: str, prior_event_path: Path, sources: Sequence[Path], windows: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    original = _read_jsonl_gz(prior_event_path)
    market = _load_market_day(sources, windows, day)
    result = []
    for row in original:
        session = str(row["session"])
        if session not in market:
            raise VolatilityStudyError(f"unknown event session {session}/{day}")
        enriched = dict(row)
        enriched.update(_pre_context(row, market[session], windows[session]))
        if enriched.get("context_status") == "OK" and int(enriched["context_last_timestamp_ns"]) >= int(row["interaction_start_ns"]):
            raise VolatilityStudyError(f"pre-event cutoff violation {row['event_id']}")
        result.append(enriched)
    result.sort(key=lambda r: (int(r["interaction_start_ns"]), str(r["family"]), str(r["event_id"])))
    return result


def _assign_causal_buckets(rows_by_date: Mapping[str, list[dict[str, Any]]], dates: Sequence[str]) -> None:
    histories: dict[tuple[str, str], list[float]] = defaultdict(list)
    tod_histories: dict[tuple[str, str], list[float]] = defaultdict(list)
    session_histories: dict[tuple[str, str], list[float]] = defaultdict(list)
    frozen_prior_histories: dict[tuple[str, str], list[float]] = {}
    auxiliary = ("pre_30s_top5_depth_mean", "pre_30s_pressured_top5_depth_mean", "pre_event_spread_ticks",
                 "pre_5s_trades_per_second", "pre_30s_trades_per_second", "pre_5s_contracts_per_second",
                 "pre_30s_contracts_per_second", "pre_5s_book_updates_per_second", "pre_30s_book_updates_per_second",
                 "pre_5s_quote_records_per_second", "pre_30s_quote_records_per_second",
                 "velocity_ticks_per_second_5s", "velocity_ticks_per_second_30s",
                 "directional_velocity_ticks_per_second_5s", "directional_velocity_ticks_per_second_30s",
                 "trend_efficiency_5s", "trend_efficiency_30s")
    for day in dates:
        rows = rows_by_date[day]
        # Reuse the prior study's exact expanding resiliency/flow state mapping.
        # Its frozen reported binary speed label is FAST/SLOW. For the requested
        # descriptive three-way table, split its already date-safe expanding
        # quintile rank as Q1-Q2=SLOW, Q3=NORMAL, Q4-Q5=FAST. The underlying
        # resiliency score/formula is untouched.
        base._assign_expanding_buckets(rows, frozen_prior_histories)
        for row in rows:
            rank = row.get("bucket_resiliency_60s_score")
            row["resiliency_triplet_state"] = (
                "SLOW" if rank in {"Q1", "Q2"} else "NORMAL" if rank == "Q3" else
                "FAST" if rank in {"Q4", "Q5"} else "RESILIENCY_INSUFFICIENT")
        # Primary thresholds are frozen at the start of each trading date:
        # no same-date event, including an earlier event, calibrates another.
        session_history: dict[tuple[str, str], list[float]] = defaultdict(list)
        cursor = 0
        while cursor < len(rows):
            timestamp = int(rows[cursor]["interaction_start_ns"])
            end = cursor + 1
            while end < len(rows) and int(rows[end]["interaction_start_ns"]) == timestamp:
                end += 1
            # Normalize first: expanding family buckets for normalized RV must
            # see this event's causal normalized value, not the absent raw-event
            # placeholder that was present before enrichment.
            for row in rows[cursor:end]:
                tod = str(row["tod_bucket"])
                for seconds in (30, 120):
                    raw_key = f"rv_{seconds}s_ticks"; norm_key = f"tod_norm_rv_{seconds}s"
                    hist = tod_histories[(tod, raw_key)]
                    median = float(np.median(hist)) if len(hist) >= MIN_HISTORY else None
                    row[f"{norm_key}_baseline_median"] = median
                    row[norm_key] = tod_normalize(row.get(raw_key), hist)
            for row in rows[cursor:end]:
                family = str(row["family"]); session = str(row["session"]); tod = str(row["tod_bucket"])
                for feature in VOL_FEATURES:
                    value = row.get(feature); prior = histories[(family, feature)]
                    row[f"{feature}_expanding_pct"] = _percentile(value, prior)
                    row[f"{feature}_q5"] = _expanding_bucket(value, prior, 5)
                    row[f"{feature}_tercile"] = _expanding_bucket(value, prior, 3)
                for seconds in (30, 120):
                    raw_key = f"rv_{seconds}s_ticks"; norm_key = f"tod_norm_rv_{seconds}s"
                    hist = tod_histories[(tod, raw_key)]
                    row[f"rv_{seconds}s_within_tod_pct"] = _percentile(row.get(raw_key), hist)
                    row[f"rv_{seconds}s_tod_norm_pct"] = _percentile(row.get(norm_key), tod_histories[(tod, norm_key)])
                    row[f"rv_{seconds}s_within_session_pct"] = _percentile(row.get(raw_key), session_history[(session, raw_key)])
                for feature in auxiliary:
                    row[f"{feature}_tercile"] = _tercile(row.get(feature), histories[(family, feature)])
                for feature in ("pre_30s_top5_depth_mean", "pre_30s_pressured_top5_depth_mean"):
                    row[f"{feature}_tod_tercile"] = _tercile(row.get(feature), tod_histories[(tod, feature)])
                row["rv_30s_state"] = _coarse_vol(row.get("rv_30s_ticks"), histories[(family, "rv_30s_ticks")])
                row["rv_120s_state"] = _coarse_vol(row.get("rv_120s_ticks"), histories[(family, "rv_120s_ticks")])
                row["tod_norm_rv_30s_state"] = _coarse_vol(row.get("tod_norm_rv_30s"), histories[(family, "tod_norm_rv_30s")])
                row["tod_norm_rv_120s_state"] = _coarse_vol(row.get("tod_norm_rv_120s"), histories[(family, "tod_norm_rv_120s")])
                row["rv_30s_tercile"] = _expanding_bucket(row.get("rv_30s_ticks"), histories[(family, "rv_30s_ticks")], 3)
                row["rv_120s_tercile"] = _expanding_bucket(row.get("rv_120s_ticks"), histories[(family, "rv_120s_ticks")], 3)
                pct = row.get("rv_30s_ticks_expanding_pct")
                row["rv_30s_tail_band"] = ("P95_97P5" if pct is not None and .95 <= pct < .975 else "P97P5_99" if pct is not None and .975 <= pct < .99 else
                                            "P99_100" if pct is not None and pct >= .99 else "BELOW_P95" if pct is not None else "INSUFFICIENT_HISTORY")
                pct120 = row.get("rv_120s_ticks_expanding_pct")
                row["rv_120s_tail_band"] = ("P95_97P5" if pct120 is not None and .95 <= pct120 < .975 else "P97P5_99" if pct120 is not None and .975 <= pct120 < .99 else
                                             "P99_100" if pct120 is not None and pct120 >= .99 else "BELOW_P95" if pct120 is not None else "INSUFFICIENT_HISTORY")
                spread_hist = histories[(family, "pre_event_spread_ticks")]
                depth_hist = tod_histories[(tod, "pre_30s_top5_depth_mean")]
                activity_hist = histories[(family, "pre_30s_trades_per_second")]
                spread_p95 = float(np.quantile(spread_hist, .95)) if len(spread_hist) >= MIN_HISTORY else None
                depth_p05 = float(np.quantile(depth_hist, .05)) if len(depth_hist) >= MIN_HISTORY else None
                activity_p95 = float(np.quantile(activity_hist, .95)) if len(activity_hist) >= MIN_HISTORY else None
                abnormal_conditions = [spread_p95 is not None and row.get("pre_event_spread_ticks") is not None and row["pre_event_spread_ticks"] >= spread_p95,
                    depth_p05 is not None and row.get("pre_30s_top5_depth_mean") is not None and row["pre_30s_top5_depth_mean"] <= depth_p05,
                    activity_p95 is not None and row.get("pre_30s_trades_per_second") is not None and row["pre_30s_trades_per_second"] >= activity_p95]
                row["abnormal_state"], row["abnormal_market_condition_count"] = abnormal_market_state(
                    row["rv_30s_state"], spread_abnormal=abnormal_conditions[0],
                    depth_abnormal=abnormal_conditions[1], intensity_abnormal=abnormal_conditions[2])
            # For causal within-session ranks only, prior same-session events may calibrate later timestamps.
            for row in rows[cursor:end]:
                session = str(row["session"])
                for seconds in (30, 120):
                    key = f"rv_{seconds}s_ticks"; value = row.get(key)
                    if value is not None and math.isfinite(float(value)):
                        session_history[(session, key)].append(float(value))
            cursor = end
        # Only after the date is fully scored do its observations enter any expanding history.
        for row in rows:
            family = str(row["family"]); tod = str(row["tod_bucket"]); session = str(row["session"])
            for feature in VOL_FEATURES + auxiliary:
                val = row.get(feature)
                if val is not None and math.isfinite(float(val)):
                    histories[(family, feature)].append(float(val))
            for seconds in (30, 120):
                raw_key = f"rv_{seconds}s_ticks"; norm_key = f"tod_norm_rv_{seconds}s"
                for key, target, group in ((raw_key, tod_histories, tod), (norm_key, tod_histories, tod),
                                           (raw_key, session_histories, session)):
                    val = row.get(key)
                    if val is not None and math.isfinite(float(val)):
                        target[(group, key)].append(float(val))
            for feature in ("pre_30s_top5_depth_mean", "pre_30s_pressured_top5_depth_mean"):
                val = row.get(feature)
                if val is not None and math.isfinite(float(val)):
                    tod_histories[(tod, feature)].append(float(val))


def _repair_tod_normalized_buckets(rows_by_date: Mapping[str, list[dict[str, Any]]], dates: Sequence[str]) -> None:
    """Rebuild only TOD-normalized RV fields from strictly prior-date histories.

    Used when enriched and already-causally-bucketed artifacts exist but the
    TOD-normalized labels need repair. It does not alter other event features or
    any absorption-core label.
    """
    raw_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    norm_tod_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    norm_family_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    for day in dates:
        rows = rows_by_date[day]
        for row in rows:
            tod = str(row["tod_bucket"]); family = str(row["family"])
            for seconds in (30, 120):
                raw_key = f"rv_{seconds}s_ticks"; norm_key = f"tod_norm_rv_{seconds}s"
                raw = row.get(raw_key); prior_raw = raw_history[(tod, raw_key)]
                median = float(np.median(np.asarray(prior_raw, dtype=np.float64))) if len(prior_raw) >= MIN_HISTORY else None
                normalized = tod_normalize(raw, prior_raw)
                row[f"{norm_key}_baseline_median"] = median
                row[norm_key] = normalized
                row[f"rv_{seconds}s_within_tod_pct"] = _percentile(raw, prior_raw)
                row[f"rv_{seconds}s_tod_norm_pct"] = _percentile(normalized, norm_tod_history[(tod, norm_key)])
                history = norm_family_history[(family, norm_key)]
                row[f"{norm_key}_expanding_pct"] = _percentile(normalized, history)
                row[f"{norm_key}_q5"] = _expanding_bucket(normalized, history, 5)
                row[f"{norm_key}_tercile"] = _expanding_bucket(normalized, history, 3)
                row[f"{norm_key}_state"] = _coarse_vol(normalized, history)
        # All records on a date are scored before that date can enter calibration.
        for row in rows:
            tod = str(row["tod_bucket"]); family = str(row["family"])
            for seconds in (30, 120):
                raw_key = f"rv_{seconds}s_ticks"; norm_key = f"tod_norm_rv_{seconds}s"
                raw = row.get(raw_key); normalized = row.get(norm_key)
                if raw is not None and math.isfinite(float(raw)):
                    raw_history[(tod, raw_key)].append(float(raw))
                if normalized is not None and math.isfinite(float(normalized)):
                    norm_tod_history[(tod, norm_key)].append(float(normalized))
                    norm_family_history[(family, norm_key)].append(float(normalized))


def _all_bucket_metrics(rows: Sequence[Mapping[str, Any]], field: str, *, family_field: bool = False) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("family", "ALL")) if family_field else "ALL", str(row.get(field, "INSUFFICIENT_HISTORY")))
        grouped[key].append(row)
    output: dict[str, Any] = {}
    for (family, bucket), vals in sorted(grouped.items()):
        output.setdefault(family, {})[bucket] = _group_summary(vals)
    return output


def _shape_analysis(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        result[field] = {}
        family_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            family_groups[str(row["family"])].append(row)
        for family, vals in family_groups.items():
            q_means = [_dist([r.get("markout_2000ms_ticks") for r in vals if r.get(field) == f"Q{i}"])["mean"] for i in range(1, 6)]
            result[field][family] = {"q1_q5_2s": _qcontrast(vals, field, 2000),
                                     "q1_q5_by_horizon": {str(ms): _qcontrast(vals, field, ms) for ms in (2000, 5000, 10000)},
                                     "q_bucket_2s_means": q_means, "shape": shape_classification(q_means),
                                     "q5_q1_effect_size": effect_size(_qcontrast(vals, field, 2000).get("q5_minus_q1"))}
    return result


def _control_results(rows: Sequence[Mapping[str, Any]], control: str, volatility_field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(control, "UNKNOWN"))].append(row)
    return {label: {"event_count": len(group), "active_dates": len({str(r["date"]) for r in group}),
                    "q5_q1": {str(ms): _qcontrast(group, volatility_field, ms) for ms in (2000, 5000, 10000)},
                    "coarse_state_2s": _coarse_contrasts(group, volatility_field)}
            for label, group in sorted(groups.items())}


def _state_control_crosstab(rows: Sequence[Mapping[str, Any]], control: str,
                            state_field: str) -> dict[str, Any]:
    controls: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        controls[str(row.get(control, "UNKNOWN"))].append(row)
    states = ("LOW", "NORMAL", "HIGH", "EXTREME")
    return {label: {state: _group_summary([row for row in members if row.get(state_field) == state])
                    for state in states}
            for label, members in sorted(controls.items())}


def _coarse_contrasts(rows: Sequence[Mapping[str, Any]], volatility_field: str) -> dict[str, Any]:
    state_field = volatility_field.replace("_q5", "_state")
    state_field = {"rv_30s_ticks_q5": "rv_30s_state", "rv_120s_ticks_q5": "rv_120s_state",
                   "tod_norm_rv_30s_q5": "tod_norm_rv_30s_state",
                   "tod_norm_rv_120s_q5": "tod_norm_rv_120s_state"}.get(volatility_field, state_field)
    out = {}
    for name, left, right in (("HIGH_LOW", "HIGH", "LOW"), ("HIGH_EXTREME", "HIGH", "EXTREME")):
        a = [r.get("markout_2000ms_ticks") for r in rows if r.get(state_field) == left and r.get("markout_2000ms_ticks") is not None]
        b = [r.get("markout_2000ms_ticks") for r in rows if r.get(state_field) == right and r.get("markout_2000ms_ticks") is not None]
        out[name] = {"left_state": left, "right_state": right, "left_count": len(a), "right_count": len(b),
                     "left_mean": _dist(a)["mean"], "right_mean": _dist(b)["mean"],
                     "left_minus_right": (_dist(a)["mean"] - _dist(b)["mean"]) if a and b else None}
    return out


def _session_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for session in ("ASIA", "EUROPE", "NY"):
        members = [r for r in rows if r.get("session") == session]
        output[session] = {
            "event_count": len(members), "active_dates": len({str(r["date"]) for r in members}),
            "rv30_quintiles": _all_bucket_metrics(members, "rv_30s_ticks_q5"),
            "rv30_states": _all_bucket_metrics(members, "rv_30s_state"),
            "rv120_quintiles": _all_bucket_metrics(members, "rv_120s_ticks_q5"),
            "rv120_states": _all_bucket_metrics(members, "rv_120s_state"),
            "rv30_q5_q1": {str(ms): _qcontrast(members, "rv_30s_ticks_q5", ms) for ms in (2000, 5000, 10000, 30000)},
            "rv120_q5_q1": {str(ms): _qcontrast(members, "rv_120s_ticks_q5", ms) for ms in (2000, 5000, 10000, 30000)},
            "rv30_state_contrasts": _coarse_contrasts(members, "rv_30s_ticks_q5"),
            "rv120_state_contrasts": _coarse_contrasts(members, "rv_120s_ticks_q5"),
            "monthly": {month: _group_summary(group) for month, group in _monthly_groups(members).items()},
        }
    return output


def _monthly_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["date"])[:7]].append(row)
    return dict(result)


def _prototype_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    gates: dict[str, list[Mapping[str, Any]]] = {
        "ALL_EVENTS": list(rows),
        "NORMAL_OR_HIGH_VOL_ONLY": [r for r in rows if r.get("rv_30s_state") in {"NORMAL", "HIGH"}],
        "HIGH_VOL_ONLY": [r for r in rows if r.get("rv_30s_state") == "HIGH"],
        "EXCLUDE_EXTREME_VOL": [r for r in rows if r.get("rv_30s_state") in {"LOW", "NORMAL", "HIGH"}],
        "NORMAL_HIGH_EXCLUDE_EXTREME": [r for r in rows if r.get("rv_30s_state") in {"NORMAL", "HIGH"}],
        "TOD_NORMALIZED_NORMAL_OR_HIGH": [r for r in rows if r.get("tod_norm_rv_30s_state") in {"NORMAL", "HIGH"}],
    }
    return gates


def _density(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row.get(key, "INSUFFICIENT_HISTORY")))].append(row)
    out: dict[str, Any] = {}
    for (family, state), group in sorted(grouped.items()):
        by_day: dict[str, list[int]] = defaultdict(list)
        by_minute: set[tuple[str, int]] = set()
        for row in group:
            ts = int(row["interaction_start_ns"])
            by_day[str(row["date"])].append(ts)
            by_minute.add((str(row["date"]), ts // (60 * NS)))
        intervals = [((b - a) / NS) for times in by_day.values()
                     for ordered in (sorted(times),) for a, b in zip(ordered, ordered[1:])]
        level_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in group:
            if row.get("level") is not None:
                level_groups[(str(row["date"]), str(row["level"]))].append(int(row["interaction_start_ns"]))
        level_intervals = [((b - a) / NS) for times in level_groups.values()
                           for ordered in (sorted(times),) for a, b in zip(ordered, ordered[1:])]
        total_minutes = len(by_minute)
        out.setdefault(family, {})[state] = {"event_count": len(group), "active_dates": len(by_day),
            "active_event_minutes": total_minutes,
            "events_per_active_minute": (len(group) / total_minutes) if total_minutes else None,
            "median_inter_event_seconds": _dist(intervals)["median"],
            "same_family_inter_event_p25_seconds": _dist(intervals)["p25"],
            "same_level_inter_event_median_seconds": _dist(level_intervals)["median"],
            "level_strata": len(level_groups)}
    return out


def _family_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    output = {}
    for family, group in sorted(by_family.items()):
        monthly = _monthly_groups(group)
        feature = "rv_30s_ticks_q5"
        contrast = _qcontrast(group, feature, 2000)
        by_day = _daily_bucket(group, feature)
        qmeans = [_dist([r.get("markout_2000ms_ticks") for r in group if r.get(feature) == f"Q{i}"])["mean"] for i in range(1, 6)]
        status = "INSUFFICIENT" if sum(1 for v in qmeans if v is not None) < 4 else (
            "CONSISTENT_POSITIVE" if contrast.get("q5_minus_q1") is not None and contrast["q5_minus_q1"] > 0 and
            all(_qcontrast(month_rows, feature, 2000).get("q5_minus_q1") is not None and _qcontrast(month_rows, feature, 2000)["q5_minus_q1"] > 0
                for month_rows in monthly.values()) else "INVERTED" if shape_classification(qmeans) == "INVERTED_U" else "MIXED")
        output[family] = {"event_count": len(group), "active_dates": len({str(r["date"]) for r in group}),
            "q1_q5_2s": contrast, "q1_q5_by_horizon": {str(ms): _qcontrast(group, feature, ms) for ms in (2000, 5000, 10000)},
            "q1_q5_bucket_means_2s": qmeans, "shape": shape_classification(qmeans), "effect_size": effect_size(contrast.get("q5_minus_q1")),
            "state_buckets": _all_bucket_metrics(group, "rv_30s_state").get("ALL", {}),
            "months": {month: {"event_count": len(members), "q1_q5_2s": _qcontrast(members, feature, 2000),
                               "q1_q5_5s": _qcontrast(members, feature, 5000), "q1_q5_10s": _qcontrast(members, feature, 10000)}
                        for month, members in sorted(monthly.items())},
            "daily_stability_by_quintile": {bucket: item.get("stability") for bucket, item in by_day.items()},
            "daily_effects_by_quintile": {bucket: item.get("dates") for bucket, item in by_day.items()},
            "daily_q5_q1_contrast": {str(ms): daily_q5_q1_contrast(group, feature, ms)
                                     for ms in (2000, 5000, 10000, 30000)},
            "effect_classification": status}
    return output


def _run_aggregations(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    # Exact duplicate geometry is retained in raw files and separately reported; a deduplicated analysis is parallel.
    deduped = deduplicate_event_clusters(rows)
    primary_field = {"rv_30s_ticks": "rv_30s_ticks_q5", "rv_120s_ticks": "rv_120s_ticks_q5",
                     "tod_norm_rv_30s": "tod_norm_rv_30s_q5", "tod_norm_rv_120s": "tod_norm_rv_120s_q5"}
    bucket_payload: dict[str, Any] = {}
    for feature, field in primary_field.items():
        bucket_payload[feature] = {"pooled": _all_bucket_metrics(rows, field),
                                   "by_month": {month: _all_bucket_metrics(members, field) for month, members in _monthly_groups(rows).items()},
                                   "daily_stability": _daily_bucket(rows, field),
                                   "daily_q5_q1_contrast": {str(ms): daily_q5_q1_contrast(rows, field, ms)
                                                             for ms in (2000, 5000, 10000, 30000)},
                                   "deduplicated_exact_geometry": _all_bucket_metrics(deduped, field)}
    _atomic_json(output_root / "volatility-buckets.json", bucket_payload)
    norm = {f"tod_norm_rv_{s}s": {"value_definition": f"raw RV_{s}s / median of prior eligible-date RV in the same fixed TOD bucket",
                                  # Percentiles are continuous; summarizing them as categorical keys
                                  # creates one expensive full market-summary per event. Keep bounded
                                  # descriptive distributions and fixed TOD-bucket stratification.
                                  "within_tod_percentile": _dist([r.get(f"rv_{s}s_within_tod_pct") for r in rows]),
                                  "normalized_percentile": _dist([r.get(f"rv_{s}s_tod_norm_pct") for r in rows]),
                                  "within_session_percentile": _dist([r.get(f"rv_{s}s_within_session_pct") for r in rows]),
                                  "expanding_history_percentile": _dist([r.get(f"rv_{s}s_ticks_expanding_pct") for r in rows]),
                                  "within_tod_bucket": {tod: _dist([r.get(f"rv_{s}s_within_tod_pct") for r in rows
                                                                     if r.get("tod_bucket") == tod])
                                                         for tod in TOD_BUCKETS}}
            for s in (30, 120)}
    _atomic_json(output_root / "tod-normalized-volatility.json", norm)
    shapes = _shape_analysis(rows, list(primary_field.values()))
    _atomic_json(output_root / "shape-analysis.json", shapes)
    secondary = {f"rv_{seconds}s_ticks": {
        "quintiles": _all_bucket_metrics(rows, f"rv_{seconds}s_ticks_q5"),
        "terciles": _all_bucket_metrics(rows, f"rv_{seconds}s_ticks_tercile"),
        "coarse_states": _all_bucket_metrics(rows, f"rv_{seconds}s_state"),
        "monthly_quintiles": {month: _all_bucket_metrics(members, f"rv_{seconds}s_ticks_q5")
                              for month, members in _monthly_groups(rows).items()}}
        for seconds in (10, 300)}
    bucket_payload["secondary_robustness"] = secondary
    _atomic_json(output_root / "volatility-buckets.json", bucket_payload)
    extremes = {f"rv_{s}s": {band: _group_summary([r for r in rows if r.get(f"rv_{s}s_tail_band") == band])
                                for band in ("P95_97P5", "P97P5_99", "P99_100", "BELOW_P95", "INSUFFICIENT_HISTORY")}
                for s in (30, 120)}
    # Separate each horizon's tail band, never selecting a threshold from outcomes.
    for s in (30, 120):
        key = f"rv_{s}s_ticks_expanding_pct"
        for band, lo, hi in (("P95_97P5", .95, .975), ("P97P5_99", .975, .99), ("P99_100", .99, 1.01)):
            members = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
            extremes[f"rv_{s}s"][band] = _group_summary(members)
    _atomic_json(output_root / "extreme-volatility.json", extremes)
    tod_control: dict[str, Any] = {}
    for tod in TOD_BUCKETS:
        members = [r for r in rows if r.get("tod_bucket") == tod]
        tod_control[tod] = {"event_count": len(members), "q5_q1_rv30": {str(ms): _qcontrast(members, "rv_30s_ticks_q5", ms) for ms in (250, 500, 1000, 2000, 5000, 10000, 30000)},
                            "q5_q1_rv120": {str(ms): _qcontrast(members, "rv_120s_ticks_q5", ms) for ms in (250, 500, 1000, 2000, 5000, 10000, 30000)}}
    sessions = _session_results(rows)
    _atomic_json(output_root / "session-control.json", {"sessions": sessions,
                                                            "fixed_time_of_day_buckets": tod_control,
                                                            "effect_after_tod_control": _tod_effect_label(rows)})
    trend_fields = ("trend_efficiency_5s_tercile", "trend_efficiency_30s_tercile",
                    "directional_velocity_ticks_per_second_5s_tercile",
                    "directional_velocity_ticks_per_second_30s_tercile")
    trend_control = {"within_control_tercile": {control: _control_results(rows, control, "rv_30s_ticks_q5")
                                                for control in trend_fields},
                     "coarse_volatility_by_control_tercile": {control: _state_control_crosstab(rows, control, "rv_30s_state")
                                                               for control in trend_fields},
                     "legacy_within_coarse_volatility_state": {
                         f"rv30_{state}": {control: _control_results([r for r in rows if r.get("rv_30s_state") == state], control, "rv_30s_ticks_q5")
                                            for control in trend_fields}
                         for state in ("LOW", "NORMAL", "HIGH", "EXTREME")}}
    _atomic_json(output_root / "trend-control.json", trend_control)
    depth_fields = ("pre_30s_top5_depth_mean_tod_tercile", "pre_30s_pressured_top5_depth_mean_tod_tercile")
    depth_control = {"within_control_tercile": {control: _control_results(rows, control, "rv_30s_ticks_q5")
                                                for control in depth_fields},
                     "coarse_volatility_by_control_tercile": {control: _state_control_crosstab(rows, control, "rv_30s_state")
                                                               for control in depth_fields},
                     "legacy_within_coarse_volatility_state": {
                         f"rv30_{state}": {control: _control_results([r for r in rows if r.get("rv_30s_state") == state], control, "rv_30s_ticks_q5")
                                            for control in depth_fields}
                         for state in ("LOW", "NORMAL", "HIGH", "EXTREME")}}
    _atomic_json(output_root / "depth-control.json", depth_control)
    intensity_fields = ("pre_5s_trades_per_second_tercile", "pre_30s_trades_per_second_tercile",
                        "pre_5s_contracts_per_second_tercile", "pre_30s_contracts_per_second_tercile",
                        "pre_5s_book_updates_per_second_tercile", "pre_30s_book_updates_per_second_tercile",
                        "pre_5s_quote_records_per_second_tercile", "pre_30s_quote_records_per_second_tercile")
    intensity = {"within_control_tercile": {control: _control_results(rows, control, "rv_30s_ticks_q5")
                                            for control in intensity_fields},
                 "coarse_volatility_by_control_tercile": {control: _state_control_crosstab(rows, control, "rv_30s_state")
                                                           for control in intensity_fields},
                 "legacy_within_coarse_volatility_state": {
                     f"rv30_{state}": {control: _control_results([r for r in rows if r.get("rv_30s_state") == state], control, "rv_30s_ticks_q5")
                                        for control in intensity_fields}
                     for state in ("LOW", "NORMAL", "HIGH", "EXTREME")}}
    _atomic_json(output_root / "intensity-control.json", intensity)
    abnormal_control = _control_results(rows, "abnormal_state", "rv_30s_ticks_q5")
    _atomic_json(output_root / "abnormal-state.json", {"definition": "RV30 expanding EXTREME AND (spread >= prior family P95 OR 30s depth <= prior TOD P05 OR trade intensity >= prior family P95); minimum calibration history 20; purely market-data proxy, not news labels.",
                                                           "states": _summary_by(rows, "abnormal_state"),
                                                           "volatility_control": abnormal_control,
                                                           "monthly": {month: _summary_by(members, "abnormal_state") for month, members in _monthly_groups(rows).items()},
                                                           "abnormal_condition_count": _dist([r.get("abnormal_market_condition_count") for r in rows])})
    family = _family_results(rows)
    _atomic_json(output_root / "family-results.json", family)
    daily_stability = {field: _daily_bucket(rows, field) for field in primary_field.values()}
    _atomic_json(output_root / "daily-stability.json", daily_stability)
    monthly_stability = {field: {month: _all_bucket_metrics(members, field) for month, members in _monthly_groups(rows).items()}
                         for field in primary_field.values()}
    _atomic_json(output_root / "monthly-stability.json", monthly_stability)
    mfe = {field: {bucket: _group_summary([r for r in rows if r.get(field) == bucket])["mfe_mae"]
                   for bucket in ("Q1", "Q5", "LOW", "NORMAL", "HIGH", "EXTREME")}
           for field in primary_field.values()}
    _atomic_json(output_root / "mfe-mae-results.json", mfe)
    barriers = {field: {bucket: _group_summary([r for r in rows if r.get(field) == bucket])["barriers"]
                        for bucket in ("Q1", "Q5", "LOW", "NORMAL", "HIGH", "EXTREME")}
                for field in primary_field.values()}
    _atomic_json(output_root / "barrier-results.json", barriers)
    bootstrap = {f"{feature}|{contrast}|{ms}": clustered_bootstrap(rows, field if contrast == "Q5_Q1" else feature.replace("_q5", "_state"), contrast, ms)
                 for feature, field in primary_field.items() for contrast in ("Q5_Q1", "HIGH_LOW", "HIGH_EXTREME") for ms in (2000, 5000, 10000)}
    _atomic_json(output_root / "clustered-bootstrap.json", bootstrap)
    permutation = {}
    for feature, field in primary_field.items():
        state_field = {"rv_30s_ticks": "rv_30s_state", "rv_120s_ticks": "rv_120s_state",
                       "tod_norm_rv_30s": "tod_norm_rv_30s_state",
                       "tod_norm_rv_120s": "tod_norm_rv_120s_state"}[feature]
        for ms in (2000, 5000, 10000):
            permutation[f"{feature}|Q5_Q1|{ms}"] = permutation_test(rows, field, ms, contrast="Q5_Q1")
            permutation[f"{feature}|HIGH_LOW|{ms}"] = permutation_test(rows, state_field, ms, contrast="HIGH_LOW")
    _atomic_json(output_root / "permutation-results.json", permutation)
    _atomic_json(output_root / "event-density.json", {"rv30_state": _density(rows, "rv_30s_state"),
                                                        "family_event_count": {f: len([r for r in rows if r["family"] == f]) for f in sorted({str(r["family"]) for r in rows})},
                                                        "same_level_and_family": "per-family/per-volatility-state median spacing, with exact level strata retained in the source event rows"})
    overlap = event_overlap_audit(rows)
    overlap["deduplicated_results_rv30_q"] = _all_bucket_metrics(deduped, primary_field["rv_30s_ticks"])
    overlap["raw_results_rv30_q"] = _all_bucket_metrics(rows, primary_field["rv_30s_ticks"])
    overlap["deduplicated_event_count"] = len(deduped)
    _atomic_json(output_root / "overlap-audit.json", overlap)
    for row in rows:
        row["rv_30s_volatility_tercile_state"] = volatility_tercile_label(row.get("rv_30s_tercile"))
    vxr = _interaction_table(rows, "rv_30s_volatility_tercile_state", "resiliency_triplet_state", ("LOW", "MEDIUM", "HIGH"),
                            ("SLOW", "NORMAL", "FAST"))
    vxr["state_definition"] = {
        "volatility": "date-expanding RV30 tercile; Q1=LOW, Q2=MEDIUM, Q3=HIGH; labels are translated without refitting",
        "resiliency_metric": "frozen prior resiliency_60s_score (median 500ms recovery fraction)",
        "resiliency_state_mapping": "frozen prior-date empirical quintile ranks Q1-Q2=SLOW, Q3=NORMAL, Q4-Q5=FAST; insufficient prior history remains excluded",
        "no_resiliency_formula_change": True,
    }
    vxf = _interaction_table(rows, "rv_30s_volatility_tercile_state", "flow_state", ("LOW", "MEDIUM", "HIGH"),
                             ("FLOW_SUPPORTS_REVERSAL", "FLOW_NEUTRAL", "FLOW_OPPOSES_REVERSAL"))
    vxf["state_definition"] = {
        "volatility": "date-expanding RV30 tercile; Q1=LOW, Q2=MEDIUM, Q3=HIGH; labels are translated without refitting",
        "flow": "reused frozen prior-study MLOFI persistence state",
        "coarse_volatility_states_used": False,
    }
    _atomic_json(output_root / "volatility-x-resiliency.json", vxr)
    _atomic_json(output_root / "volatility-x-flow-persistence.json", vxf)
    gates = _prototype_gates(rows)
    gate_payload = {name: _gate_result(members) for name, members in gates.items()}
    _atomic_json(output_root / "prototype-gates.json", gate_payload)
    robustness = _gate_robustness(gates)
    _atomic_json(output_root / "prototype-gate-robustness.json", robustness)
    decision, decision_checks = _primary_decision(rows, tod_control, family, bootstrap, permutation, overlap,
                                                   sessions=sessions, trend=trend_control, depth=depth_control,
                                                   intensity=intensity, abnormal=abnormal_control, deduped=deduped)
    return {"bucket_payload": bucket_payload, "tod": norm, "shapes": shapes, "extreme": extremes,
            "tod_control": tod_control, "family": family, "daily": daily_stability,
            "bootstrap": bootstrap, "permutation": permutation, "overlap": overlap,
            "gates": gate_payload, "gate_robustness": robustness, "interaction_resiliency": vxr,
            "interaction_flow": vxf, "primary_decision": decision, "decision_checks": decision_checks,
            "prototype_decision": _prototype_decision(gate_payload, robustness)}


def _tod_effect_label(rows: Sequence[Mapping[str, Any]]) -> str:
    cells = []
    for tod in TOD_BUCKETS:
        group = [r for r in rows if r.get("tod_bucket") == tod]
        contrast = _qcontrast(group, "rv_30s_ticks_q5", 2000)
        if contrast.get("q5_minus_q1") is not None and min(contrast["q1_count"], contrast["q5_count"]) >= MIN_CELL:
            cells.append(contrast["q5_minus_q1"])
    if len(cells) < 3:
        return "INSUFFICIENT"
    positive = sum(x > 0 for x in cells)
    return "STRONG" if positive >= 5 and np.median(cells) >= .25 else "MODERATE" if positive >= 4 else "WEAK" if positive >= 2 else "ABSENT"


def _interaction_table(rows: Sequence[Mapping[str, Any]], left: str, right: str,
                       left_states: Sequence[str], right_states: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"axes": [left, right], "cells": {}}
    for a in left_states:
        for b in right_states:
            cell = [r for r in rows if r.get(left) == a and r.get(right) == b]
            out["cells"][f"{a}__X__{b}"] = _group_summary(cell)
    out["monthly"] = {month: {f"{a}__X__{b}": _group_summary([r for r in members if r.get(left) == a and r.get(right) == b])
                        for a in left_states for b in right_states}
                      for month, members in _monthly_groups(rows).items()}
    return out


def _gate_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    monthly = _monthly_groups(rows)
    family_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: family_groups[str(row["family"])].append(row)
    summary = _group_summary(rows)
    daily = _daily_bucket(rows, "gate_all") if rows else {}
    return {**summary, "daily_stability": _daily_for_all(rows),
            "monthly": {month: _group_summary(members) for month, members in sorted(monthly.items())},
            "family_coverage": {family: {"event_count": len(group), "active_dates": len({r["date"] for r in group})}
                                for family, group in sorted(family_groups.items())},
            "mean_markout_by_horizon": {str(ms): summary["markouts"][str(ms)]["mean"] for ms in MARKOUT_KEYS}}


def _daily_for_all(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: by_day[str(row["date"])].append(row)
    effects = []
    pos = neg = flat = insufficient = 0
    for day, members in sorted(by_day.items()):
        mean = _dist([r.get("markout_2000ms_ticks") for r in members])["mean"]
        if len(members) < MIN_DAILY_SAMPLE or mean is None: insufficient += 1
        elif mean > 0: pos += 1; effects.append(mean)
        elif mean < 0: neg += 1; effects.append(mean)
        else: flat += 1; effects.append(mean)
    return {"positive_dates": pos, "negative_dates": neg, "flat_dates": flat, "insufficient_dates": insufficient,
            "median_daily_effect": _dist(effects)["median"], "p25_daily_effect": _dist(effects)["p25"],
            "p75_daily_effect": _dist(effects)["p75"], "eligible_day_count": len(effects), "minimum_events_per_day": MIN_DAILY_SAMPLE}


def _gate_robustness(gates: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    base_rows = gates["ALL_EVENTS"]
    base = _group_summary(base_rows)
    base_months = {m: _group_summary(values) for m, values in _monthly_groups(base_rows).items()}
    base_daily = _daily_for_all(base_rows)
    result = {}
    for name, rows in gates.items():
        g = _group_summary(rows)
        month = {m: _group_summary(vals) for m, vals in _monthly_groups(rows).items()}
        same_days = _daily_for_all(rows)
        mean_effect = g["markouts"]["2000"]["mean"]
        base_mean = base["markouts"]["2000"]["mean"]
        effect_delta = (mean_effect - base_mean) if mean_effect is not None and base_mean is not None else None
        monthly_improvement = {}
        for m in ("2025-12", "2026-01"):
            candidate_mean = month.get(m, {}).get("markouts", {}).get("2000", {}).get("mean")
            reference_mean = base_months.get(m, {}).get("markouts", {}).get("2000", {}).get("mean")
            monthly_improvement[m] = {"candidate_mean_2s": candidate_mean, "all_events_mean_2s": reference_mean,
                                      "improved": (candidate_mean is not None and reference_mean is not None and candidate_mean > reference_mean)}
        both_months = all(monthly_improvement[m]["improved"] for m in monthly_improvement)
        baseline_mfe = base["mfe_mae"]["5000"]["mfe"]["mean"]
        candidate_mfe = g["mfe_mae"]["5000"]["mfe"]["mean"]
        baseline_mae = base["mfe_mae"]["5000"]["mae_adverse_magnitude"]["mean"]
        candidate_mae = g["mfe_mae"]["5000"]["mae_adverse_magnitude"]["mean"]
        baseline_barrier = base["barriers"]["1_1"]
        candidate_barrier = g["barriers"]["1_1"]
        daily_improved = (same_days["median_daily_effect"] is not None and base_daily["median_daily_effect"] is not None and
                          same_days["median_daily_effect"] > base_daily["median_daily_effect"] and
                          same_days["positive_dates"] > same_days["negative_dates"])
        mfe_mae_improved = (candidate_mfe is not None and baseline_mfe is not None and candidate_mfe > baseline_mfe and
                            candidate_mae is not None and baseline_mae is not None and candidate_mae <= baseline_mae)
        candidate_fav = candidate_barrier["favorable_first_probability"]
        baseline_fav = baseline_barrier["favorable_first_probability"]
        candidate_adv = candidate_barrier["adverse_first_probability"]
        baseline_adv = baseline_barrier["adverse_first_probability"]
        barrier_improved = (candidate_fav is not None and baseline_fav is not None and
                            candidate_adv is not None and baseline_adv is not None and
                            candidate_fav > baseline_fav and candidate_adv <= baseline_adv)
        result[name] = {"event_count": len(rows), "retained_fraction": len(rows) / len(base_rows) if base_rows else None,
                        "active_dates": len({r["date"] for r in rows}), "mean_2s_delta_vs_all": effect_delta,
                        "median_2s": g["markouts"]["2000"]["median"], "daily_stability": same_days,
                        "both_months_positive_2s": both_months, "monthly_mean_improvement_vs_all": monthly_improvement,
                        "family_count": len({r["family"] for r in rows}), "family_count_total": len({r["family"] for r in base_rows}),
                        "mfe_mae": g["mfe_mae"], "barriers": g["barriers"],
                        "improvements_vs_all": {"mean_2s": effect_delta is not None and effect_delta > 0,
                            "median_2s": g["markouts"]["2000"]["median"] is not None and base["markouts"]["2000"]["median"] is not None and g["markouts"]["2000"]["median"] > base["markouts"]["2000"]["median"],
                            "daily_stability": daily_improved, "mfe_mae_5s": mfe_mae_improved,
                            "barrier_1_1": barrier_improved},
                        "monthly": month, "sample_collapse": len(rows) < .20 * len(base_rows)}
    # Fixed gates B and E are logically identical by contract; assert, don't disguise, the duplicate.
    result["NORMAL_HIGH_EXCLUDE_EXTREME"]["same_population_as"] = "NORMAL_OR_HIGH_VOL_ONLY"
    result["NORMAL_OR_HIGH_VOL_ONLY"]["population_equals_normal_high_exclude_extreme"] = (
        [r["event_id"] for r in gates["NORMAL_OR_HIGH_VOL_ONLY"]] == [r["event_id"] for r in gates["NORMAL_HIGH_EXCLUDE_EXTREME"]])
    return result


def _extract_control_contrasts(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "q5_minus_q1" in value and "q1_count" in value and "q5_count" in value:
            found.append(value)
        else:
            for nested in value.values():
                found.extend(_extract_control_contrasts(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.extend(_extract_control_contrasts(nested))
    return found


def _control_survival(value: Any) -> dict[str, Any]:
    all_cells = _extract_control_contrasts(value)
    adequate = [x for x in all_cells if x.get("q5_minus_q1") is not None and
                int(x.get("q1_count", 0)) >= MIN_CELL and int(x.get("q5_count", 0)) >= MIN_CELL]
    positive = sum(float(x["q5_minus_q1"]) > 0 for x in adequate)
    fraction = positive / len(adequate) if adequate else None
    return {"adequate_strata": len(adequate), "positive_strata": int(positive),
            "positive_fraction": fraction,
            "classification": "SURVIVES" if len(adequate) >= 3 and fraction >= (2 / 3) else
                              "PARTIAL" if len(adequate) >= 2 and fraction >= .5 else
                              "DOES_NOT_SURVIVE" if adequate else "INSUFFICIENT"}


def _primary_decision(rows: Sequence[Mapping[str, Any]], tod: Mapping[str, Any], family: Mapping[str, Any],
                      bootstrap: Mapping[str, Any], permutation: Mapping[str, Any], overlap: Mapping[str, Any], *,
                      sessions: Mapping[str, Any], trend: Mapping[str, Any], depth: Mapping[str, Any],
                      intensity: Mapping[str, Any], abnormal: Mapping[str, Any],
                      deduped: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    raw = _qcontrast(rows, "rv_30s_ticks_q5", 2000)
    tod_cells = [v["q5_q1_rv30"]["2000"] for v in tod.values() if "q5_q1_rv30" in v]
    tod_adequate = [x for x in tod_cells if x.get("q5_minus_q1") is not None and
                    min(x.get("q1_count", 0), x.get("q5_count", 0)) >= MIN_CELL]
    tod_positive = sum(float(x["q5_minus_q1"]) > 0 for x in tod_adequate)
    tod_survives = bool(tod_adequate) and tod_positive / len(tod_adequate) >= .5
    dec = [r for r in rows if str(r["date"]).startswith("2025-12")]
    jan = [r for r in rows if str(r["date"]).startswith("2026-01")]
    dec_con = _qcontrast(dec, "rv_30s_ticks_q5", 2000).get("q5_minus_q1")
    jan_con = _qcontrast(jan, "rv_30s_ticks_q5", 2000).get("q5_minus_q1")
    if raw.get("q5_minus_q1") is None:
        return "INSUFFICIENT_EVIDENCE", {"reason": "Q1/Q5 markouts unavailable"}
    fam_pos = sum(v["q1_q5_2s"].get("q5_minus_q1") is not None and v["q1_q5_2s"]["q5_minus_q1"] > 0 for v in family.values())
    family_n = sum(v["q1_q5_2s"].get("q5_minus_q1") is not None for v in family.values())
    ci = bootstrap.get("rv_30s_ticks|Q5_Q1|2000", {}).get("ci95", [None, None])
    perm_p = permutation.get("rv_30s_ticks|Q5_Q1|2000", {}).get("two_sided_randomization_p")
    session_contrasts = {s: sessions.get(s, {}).get("rv30_q5_q1", {}).get("2000", {}) for s in ("ASIA", "EUROPE", "NY")}
    session_survives = all(item.get("q5_minus_q1") is not None and item.get("q1_count", 0) >= MIN_CELL and
                           item.get("q5_count", 0) >= MIN_CELL and item["q5_minus_q1"] > 0
                           for item in session_contrasts.values())
    control_evidence = {"trend_velocity": _control_survival(trend.get("within_control_tercile", trend)),
                        "depth": _control_survival(depth.get("within_control_tercile", depth)),
                        "activity": _control_survival(intensity.get("within_control_tercile", intensity)),
                        "abnormal_state": _control_survival(abnormal)}
    daily = daily_q5_q1_contrast(rows, "rv_30s_ticks_q5", 2000)
    q_means = [_dist([r.get("markout_2000ms_ticks") for r in rows if r.get("rv_30s_ticks_q5") == f"Q{i}"])["mean"]
               for i in range(1, 6)]
    adjacent = [b - a for a, b in zip(q_means, q_means[1:]) if a is not None and b is not None]
    coherent_adjacent = len(adjacent) == 4 and all(x >= 0 for x in adjacent) and sum(x > 0 for x in adjacent) >= 3
    dedup_con = _qcontrast(deduped, "rv_30s_ticks_q5", 2000).get("q5_minus_q1")
    q1 = _group_summary([r for r in rows if r.get("rv_30s_ticks_q5") == "Q1"])
    q5 = _group_summary([r for r in rows if r.get("rv_30s_ticks_q5") == "Q5"])
    q1_mfe = q1["mfe_mae"]["5000"]["mfe"]["mean"]
    q5_mfe = q5["mfe_mae"]["5000"]["mfe"]["mean"]
    q1_mae = q1["mfe_mae"]["5000"]["mae_adverse_magnitude"]["mean"]
    q5_mae = q5["mfe_mae"]["5000"]["mae_adverse_magnitude"]["mean"]
    q1_bar = q1["barriers"]["1_1"]; q5_bar = q5["barriers"]["1_1"]
    excursions_support = q1_mfe is not None and q5_mfe is not None and q5_mfe > q1_mfe and q1_mae is not None and q5_mae is not None and q5_mae <= q1_mae
    barriers_support = (q1_bar["favorable_first_probability"] is not None and q5_bar["favorable_first_probability"] is not None and
                        q1_bar["adverse_first_probability"] is not None and q5_bar["adverse_first_probability"] is not None and
                        q5_bar["favorable_first_probability"] > q1_bar["favorable_first_probability"] and
                        q5_bar["adverse_first_probability"] <= q1_bar["adverse_first_probability"])
    months_same_positive = dec_con is not None and jan_con is not None and dec_con > 0 and jan_con > 0
    ci_positive = ci[0] is not None and ci[0] > 0
    perm_significant = perm_p is not None and perm_p < .05
    broad_families = fam_pos >= 7
    controls_survive = all(control_evidence[name]["classification"] == "SURVIVES"
                           for name in ("trend_velocity", "depth", "activity"))
    checks = {"raw_q5_q1_2s": raw, "dec_q5_q1_2s": dec_con, "jan_q5_q1_2s": jan_con,
              "family_positive_count": fam_pos, "family_with_estimable_contrast_count": family_n,
              "tod_adequate_buckets": len(tod_adequate), "tod_positive_buckets": int(tod_positive),
              "tod_survives": tod_survives, "session_q5_q1": session_contrasts,
              "session_survives_europe_and_ny": session_survives,
              "control_survival": control_evidence, "daily_q5_q1_2s": daily,
              "clustered_bootstrap_ci95": ci, "permutation_p": perm_p,
              "q_bucket_means_2s": q_means, "coherent_adjacent_buckets": coherent_adjacent,
              "deduplicated_q5_q1_2s": dedup_con, "mfe_mae_support": excursions_support,
              "barrier_support": barriers_support,
              "q5_mfe_5s_mean": q5_mfe, "q1_mfe_5s_mean": q1_mfe,
              "q5_mae_5s_magnitude_mean": q5_mae, "q1_mae_5s_magnitude_mean": q1_mae,
              "decision_rule": "STRONG requires positive day-cluster CI and permutation result, both months, Europe and NY, TOD, trend/velocity/depth/activity controls, >=7 families, adjacent bucket coherence, deduplicated contrast, MFE/MAE and barrier support."}
    strong = (raw["q5_minus_q1"] > .25 and ci_positive and perm_significant and months_same_positive and broad_families and
              session_survives and tod_survives and controls_survive and daily["positive_dates"] > daily["negative_dates"] and
              coherent_adjacent and dedup_con is not None and dedup_con > 0 and excursions_support and barriers_support)
    if strong:
        return "VOLATILITY_REGIME_EFFECT_STRONG_AND_INDEPENDENT", checks
    if raw["q5_minus_q1"] < .10 and (ci[0] is None or ci[0] <= 0):
        return "NO_MEANINGFUL_VOLATILITY_REGIME_EFFECT", checks
    if family_n >= 5 and fam_pos <= max(2, family_n // 3):
        return "VOLATILITY_EFFECT_FAMILY_SPECIFIC", checks
    if raw["q5_minus_q1"] > 0 and ((tod_adequate and not tod_survives) or
                                    control_evidence["activity"]["classification"] == "DOES_NOT_SURVIVE"):
        return "VOLATILITY_EFFECT_MAINLY_TIME_OF_DAY_OR_ACTIVITY_PROXY", checks
    if months_same_positive and fam_pos >= 5:
        return "VOLATILITY_REGIME_EFFECT_REAL_BUT_PARTLY_CONFOUNDED", checks
    if (dec_con is None or jan_con is None or (dec_con * jan_con <= 0) or
            daily["positive_dates"] <= daily["negative_dates"]):
        return "VOLATILITY_REGIME_EFFECT_PROMISING_BUT_UNSTABLE", checks
    if family_n < 4 or raw.get("q1_count", 0) < MIN_CELL or raw.get("q5_count", 0) < MIN_CELL:
        return "INSUFFICIENT_EVIDENCE", checks
    return "INSUFFICIENT_EVIDENCE", checks


def _prototype_decision(gates: Mapping[str, Any], robustness: Mapping[str, Any]) -> str:
    for name in ("NORMAL_OR_HIGH_VOL_ONLY", "HIGH_VOL_ONLY", "EXCLUDE_EXTREME_VOL", "TOD_NORMALIZED_NORMAL_OR_HIGH"):
        g = gates[name]; r = robustness[name]
        if (g["event_count"] >= 100 and not r["sample_collapse"] and r["both_months_positive_2s"] and
                r["improvements_vs_all"]["daily_stability"] and r["improvements_vs_all"]["mfe_mae_5s"] and
                r["improvements_vs_all"]["barrier_1_1"] and r["family_count"] >= 5):
            return "WORTH_TESTING_ON_UNTOUCHED_OOS"
    if any(robustness[name]["event_count"] >= 100 for name in robustness if name in gates):
        return "NOT_READY_FOR_OOS"
    return "DO_NOT_CONTINUE"


def _date_checkpoint_valid(checkpoint_path: Path, event_path: Path, *, day: str, source_records: Sequence[Mapping[str, Any]],
                           input_event_sha: str, config_sha: str, code_sha: str) -> bool:
    if not checkpoint_path.is_file() or not event_path.is_file():
        return False
    try:
        item = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return (item.get("date") == day and item.get("config_sha256") == config_sha and
                item.get("code_sha256") == code_sha and item.get("prior_event_sha256") == input_event_sha and
                item.get("source_files") == list(source_records) and item.get("output_sha256") == sha256_file(event_path))
    except (OSError, json.JSONDecodeError):
        return False


def _source_inventory(repository_root: Path, prior_root: Path) -> tuple[list[str], dict[str, tuple[Path, ...]], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    config_path = repository_root / base.CONFIG_PATH
    raw_config = config_path.read_bytes()
    config_sha = hashlib.sha256(raw_config).hexdigest()
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise VolatilityStudyError(f"frozen config hash mismatch: {config_sha}")
    dates, sources, source_manifest, config_payload, returned_sha, family_configs = base._source_inputs(repository_root, config_path)
    if returned_sha != EXPECTED_CONFIG_SHA256:
        raise VolatilityStudyError("frozen runner accepted a different config hash")
    manifest = json.loads((prior_root / "run-manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((prior_root / "source-coverage.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("completed_dates") != dates:
        raise VolatilityStudyError("prior 41-session event study is not complete for the source-plan date order")
    if manifest.get("event_count") != 18407 or len(family_configs) != 10:
        raise VolatilityStudyError("prior event population/frozen family count differs from sealed result")
    manifest_hashes = {"base": source_manifest.get("base_manifest_sha256"),
                       "dec_jan_extension": source_manifest.get("extension_manifest_sha256")}
    if manifest.get("source_manifests") != manifest_hashes:
        raise VolatilityStudyError("prior run is not bound to the current exact source manifests")
    if coverage.get("status") != "PASS" or len(coverage.get("source_input_files", [])) != 85:
        raise VolatilityStudyError("prior raw-source coverage is not the sealed 85-file set")
    plan_files = source_manifest.get("input_files", [])
    if len(plan_files) != 85 or len(plan_files) != len(coverage["source_input_files"]):
        raise VolatilityStudyError("current acquisition manifests differ from the prior 85-file coverage inventory")
    def source_identity(item: Mapping[str, Any]) -> tuple[str, str, int]:
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repository_root / path
        size = item.get("bytes", item.get("size_bytes"))
        return str(path.resolve()), str(item["sha256"]), int(size)
    if {source_identity(x) for x in plan_files} != {source_identity(x) for x in coverage["source_input_files"]}:
        raise VolatilityStudyError("prior source-coverage file identity set differs from current acquisition manifests")
    return list(dates), {d: tuple(Path(p) for p in paths) for d, paths in sources.items()}, source_manifest, config_payload, returned_sha, family_configs


def _build_run_manifest(root: Path, dates: Sequence[str], *, status: str, config_sha: str,
                        input_manifest_hashes: Mapping[str, str], event_count: int = 0,
                        completed_dates: Sequence[str] = (), source_files: Sequence[Mapping[str, Any]] = (),
                        primary: str | None = None, prototype: str | None = None) -> dict[str, Any]:
    prior_manifest = json.loads((root / "prior-run-manifest.snapshot.json").read_text(encoding="utf-8")) if (root / "prior-run-manifest.snapshot.json").is_file() else {}
    return {"run_id": RUN_ID, "status": status, "evidence_label": "REGIME_DIAGNOSTIC_ROBUSTNESS_SET_NOT_UNTOUCHED_OOS",
            "study": "Dec-2025 + Jan-2026 absorption volatility isolation; descriptive only",
            "config_sha256": config_sha, "expected_config_sha256": EXPECTED_CONFIG_SHA256,
            "input_manifest_sha256": dict(input_manifest_hashes), "prior_run_id": base.RUN_ID,
            "prior_candidate_tape_semantic_sha256": prior_manifest.get("candidate_tape_semantic_sha256"),
            "prior_event_population_sha256": prior_manifest.get("report_sha256"),
            "eligible_dates": list(dates), "completed_dates": list(completed_dates),
            "excluded_dates": [{"date": "2025-12-24", "reason": "SCHEDULED_EARLY_CLOSE_NOT_NORMAL_FULL_SESSION"}],
            "event_count": int(event_count), "source_file_count": len(source_files), "source_input_files": list(source_files),
            "data_downloaded": False, "strategy_pnl": False, "optimization_performed": False,
            "parameter_search_performed": False, "untouched_oos_accessed": False,
            "dec_jan_are_not_untouched_oos": True, "primary_decision": primary,
            "prototype_decision": prototype, "feature_cutoff": "every event context timestamp < interaction_start_ns",
            "random_seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_replicates": PERMUTATION_REPLICATES}


def _render_report(summary: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]], dates: Sequence[str],
                   excluded: Sequence[Mapping[str, Any]], config_sha: str) -> str:
    m = summary["monthly"]
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"
    lines = [
        "# Dec/Jan Absorption Volatility Isolation",
        "",
        "Descriptive regime diagnostic only. This is not untouched OOS, not validation, and does not evaluate strategy PnL.",
        "No strategy rules or thresholds were changed, no parameter/PnL optimization was performed, and no OOS source was accessed.",
        "",
        f"- Frozen config SHA-256: `{config_sha}`.",
        f"- Sessions completed: {len(dates)}; absorption-family event rows: {summary['event_count']:,}; excluded: {', '.join(x['date'] for x in excluded)}.",
        f"- Primary conclusion: **{summary['primary_decision']}**.",
        f"- Prototype conclusion: **{summary['prototype_decision']}**.",
        "",
        "## Monthly benchmark",
        "",
        "| Period | events | 2s mean ticks | 5s mean ticks | 10s mean ticks | 30s mean ticks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for month in ("2025-12", "2026-01", "COMBINED"):
        x = m[month]
        lines.append(f"| {month} | {x['event_count']} | {fmt(x['markouts']['2000']['mean'])} | {fmt(x['markouts']['5000']['mean'])} | {fmt(x['markouts']['10000']['mean'])} | {fmt(x['markouts']['30000']['mean'])} |")
    lines.extend(["", "## Volatility effects (2-second direction-normalized markout)", "",
                  "The comparisons below use prior-date expanding quintiles; values are not fit on the full sample.", "",
                  "| Feature | Q5-Q1 pooled | Q5-Q1 Dec | Q5-Q1 Jan | Family support | Shape summary |",
                  "|---|---:|---:|---:|---:|---|"])
    for feature, name in (("rv_30s_ticks", "RV30"), ("rv_120s_ticks", "RV120"),
                          ("tod_norm_rv_30s", "TOD-normalized RV30"), ("tod_norm_rv_120s", "TOD-normalized RV120")):
        field = feature + "_q5"; contrast = _qcontrast(rows, field, 2000)
        dec = _qcontrast([r for r in rows if str(r["date"]).startswith("2025-12")], field, 2000)
        jan = _qcontrast([r for r in rows if str(r["date"]).startswith("2026-01")], field, 2000)
        fams = summary["family"]; support = sum((fam.get("q1_q5_2s", {}).get("q5_minus_q1") or 0) > 0 for fam in fams.values())
        shapes = Counter(v.get("shape", "INSUFFICIENT") for v in _shape_analysis(rows, [field])[field].values())
        lines.append(f"| {name} | {fmt(contrast.get('q5_minus_q1'))} | {fmt(dec.get('q5_minus_q1'))} | {fmt(jan.get('q5_minus_q1'))} | {support}/10 positive family contrasts | {dict(shapes)} |")
    lines.extend(["", "## Confound controls and uncertainty", "",
                  f"- Time-of-day control: `{summary['tod_effect_label']}`. Results are stratified into {', '.join(TOD_BUCKETS)} fixed clock buckets.",
                  "- Trend/depth/activity controls compare raw RV30 Q5 against Q1 within each independently defined control tercile. Separate volatility-state-by-tercile cross-tabs are descriptive; the older same-volatility-state conditioned tables are retained under `legacy_within_coarse_volatility_state`.",
                  "",
                  "| Control | Classification | Adequate strata | Positive Q5-Q1 strata |",
                  "|---|---|---:|---:|"])
    checks = summary["decision_checks"]
    control_labels = checks["control_survival"]
    for name, label in (("trend_velocity", "Trend efficiency / velocity"), ("depth", "Displayed depth"),
                        ("activity", "Trade / quote / book-update intensity"),
                        ("abnormal_state", "Abnormal-state proxy")):
        item = control_labels[name]
        lines.append(f"| {label} | {item['classification']} | {item['adequate_strata']} | {item['positive_strata']} |")
    daily = checks["daily_q5_q1_2s"]
    session_line = ", ".join(
        f"{session}={fmt(item.get('q5_minus_q1'))} ticks (Q1 n={item.get('q1_count')}, Q5 n={item.get('q5_count')})"
        for session, item in checks["session_q5_q1"].items())
    lines.extend(["",
                  f"- Session Q5-Q1 RV30 2s contrasts: {session_line}.",
                  f"- Day stability for RV30 Q5-Q1 2s: {daily['positive_dates']} positive, {daily['negative_dates']} negative, {daily['flat_dates']} flat, {daily['insufficient_dates']} insufficient; median daily effect {fmt(daily['median_daily_effect'])} ticks (P25 {fmt(daily['p25_daily_effect'])}, P75 {fmt(daily['p75_daily_effect'])}).",
                  f"- MFE/MAE support: `{checks['mfe_mae_support']}`; barrier support: `{checks['barrier_support']}`. Five-second MFE means Q5/Q1={fmt(checks['q5_mfe_5s_mean'])}/{fmt(checks['q1_mfe_5s_mean'])} ticks; adverse-MAE magnitudes Q5/Q1={fmt(checks['q5_mae_5s_magnitude_mean'])}/{fmt(checks['q1_mae_5s_magnitude_mean'])} ticks.",
                  f"- Day-cluster bootstrap Q5−Q1 RV30 at 2s: `{summary['bootstrap']['rv_30s_ticks|Q5_Q1|2000']}`.",
                  f"- Within-date/session label-permutation RV30 at 2s: `{summary['permutation']['rv_30s_ticks|Q5_Q1|2000']}`.",
                  f"- Exact-geometry event overlap: {summary['overlap']['same_interaction_geometry']}; raw n={summary['event_count']}, exact-geometry deduplicated n={summary['overlap']['deduplicated_event_count']}.",
                  "- Activity/event clustering is summarized per family and volatility state; raw event rows are retained and never silently deduplicated.",
                  "", "## Predeclared volatility interactions", "",
                  "Volatility is the strictly date-expanding RV30 tercile (Q1/Q2/Q3 relabeled LOW/MEDIUM/HIGH); resiliency uses the prior study's frozen score/rank mapping and flow uses its frozen MLOFI state.",
                  "", "| Interaction cell | n | 2s mean | 5s mean | 30s mean |",
                  "|---|---:|---:|---:|---:|"])
    for artifact_key, artifact_label in (("interaction_resiliency", "resiliency"), ("interaction_flow", "flow")):
        for cell, item in summary.get(artifact_key, {}).get("cells", {}).items():
            marks = item["markouts"]
            lines.append(f"| {artifact_label}: {cell} | {item['event_count']} | {fmt(marks['2000']['mean'])} | {fmt(marks['5000']['mean'])} | {fmt(marks['30000']['mean'])} |")
    lines.extend(["", "## Fixed, non-optimized prototype gates", "",
                  "| Gate | n | Active dates | 2s mean | 5s mean | 30s mean | 2s median | families |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, item in summary["gates"].items():
        lines.append(f"| {name} | {item['event_count']} | {item['active_dates']} | {fmt(item['mean_markout_by_horizon']['2000'])} | {fmt(item['mean_markout_by_horizon']['5000'])} | {fmt(item['mean_markout_by_horizon']['30000'])} | {fmt(item['markouts']['2000']['median'])} | {len(item['family_coverage'])}/10 |")
    lines.extend(["", "Gate `NORMAL_OR_HIGH_VOL_ONLY` and `NORMAL_HIGH_EXCLUDE_EXTREME` are the same predeclared event population; both are reported explicitly.",
                  "Prototype status is descriptive only. No gate was selected as a strategy rule.", "",
                  "## Interpretation limits", "",
                  "December 2025 and January 2026 were previously used and remain a regime-diagnostic robustness set. Event rows cluster by day, time, family and shared interaction geometry; event-level uncertainty is not treated as independent evidence. Any future OOS testing requires a separately frozen protocol and data; no such data was accessed here.", ""])
    return "\n".join(lines)


def run(repository_root: Path, output_root: Path, *, prior_root: Path, resume: bool = False,
        smoke_date: str | None = None, aggregate_checkpoints: bool = False,
        rebuild_buckets_from_checkpoints: bool = False) -> dict[str, Any]:
    repository_root = repository_root.resolve(); prior_root = prior_root.resolve()
    dates, source_paths, source_plan, _, config_sha, families = _source_inventory(repository_root, prior_root)
    prior_manifest = json.loads((prior_root / "run-manifest.json").read_text(encoding="utf-8"))
    prior_coverage = json.loads((prior_root / "source-coverage.json").read_text(encoding="utf-8"))
    parent_sources, parent_events = _prior_file_index(prior_root)
    if len(dates) != 41 or len(parent_events) != 41 or len(parent_sources) < 41:
        raise VolatilityStudyError("prior Dec/Jan source/event inventory is not the expected 41-session set")
    wanted = [smoke_date] if smoke_date else list(dates)
    if any(day not in dates for day in wanted):
        raise VolatilityStudyError("requested smoke date is outside the frozen eligible date set")
    output_root = output_root.resolve()
    if smoke_date and output_root.exists():
        raise VolatilityStudyError(f"smoke output root already exists: {output_root}")
    if output_root.exists() and not resume:
        raise VolatilityStudyError(f"immutable output root already exists; use --resume after verifying its identity: {output_root}")
    config_file = repository_root / base.CONFIG_PATH
    if sha256_file(config_file) != EXPECTED_CONFIG_SHA256:
        raise VolatilityStudyError("config changed after source-plan verification")
    code_sha = sha256_file(Path(__file__).resolve())
    input_manifest_hashes = {"prior_run_manifest": sha256_file(prior_root / "run-manifest.json"),
                             "prior_source_coverage": sha256_file(prior_root / "source-coverage.json"),
                             "base_acquisition_manifest": str(source_plan.get("base_manifest_sha256")),
                             "dec_jan_extension_manifest": str(source_plan.get("extension_manifest_sha256"))}
    existing_manifest = output_root / "run-manifest.json"
    if output_root.exists() and resume:
        if not existing_manifest.is_file():
            raise VolatilityStudyError("resume root has no manifest")
        old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        for key, expected in (("run_id", RUN_ID), ("config_sha256", config_sha), ("input_manifest_sha256", input_manifest_hashes)):
            if old.get(key) != expected:
                raise VolatilityStudyError(f"resume identity mismatch: {key}")
        if old.get("code_sha256") != code_sha:
            # Source and frozen-contract identities are unchanged. The per-date
            # checkpoint validator binds code_sha and will invalidate/rebuild
            # every stale date output deterministically.
            print("VOLATILITY_RESUME_CODE_CHANGED=stale date checkpoints will be recomputed", flush=True)
    if aggregate_checkpoints and (not resume or smoke_date):
        raise VolatilityStudyError("--aggregate-checkpoints requires --resume and cannot be combined with --smoke-date")
    if rebuild_buckets_from_checkpoints and not aggregate_checkpoints:
        raise VolatilityStudyError("--rebuild-buckets-from-checkpoints requires --aggregate-checkpoints")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dates").mkdir(exist_ok=True); (output_root / "checkpoints").mkdir(exist_ok=True)
    _atomic_json(output_root / "prior-run-manifest.snapshot.json", prior_manifest)
    # base._source_inputs has just revalidated byte counts, SHA-256, DBN schema,
    # dataset and symbol for all 85 files. Reuse those verified records instead
    # of making a second sequential pass over the ~9.8 GB source inventory.
    expected_files: dict[str, Mapping[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    for item in source_plan["input_files"]:
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repository_root / path
        normalized = {**item, "path": str(path.resolve())}
        expected_files[str(path.resolve())] = normalized
        source_records.append({"path": str(path.resolve()), "bytes": int(item["bytes"]),
                               "sha256": str(item["sha256"]), "symbol": item.get("symbol"),
                               "schema": item.get("schema"), "date": str(item["date"])})
    if len(source_records) != 85:
        raise VolatilityStudyError(f"expected 85 source records validated by frozen source plan, got {len(source_records)}")
    if aggregate_checkpoints:
        # Analysis-only restart: source/config inventories above are independently
        # revalidated, then every date artifact is checked against the original
        # checkpoint code identity, source hashes and parent event ledger. No DBN
        # records are decoded or replayed in this path.
        assert existing_manifest.is_file()
        manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        coverage_path = output_root / "source-coverage.json"
        if not coverage_path.is_file():
            raise VolatilityStudyError("analysis resume has no source-coverage record")
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if (manifest.get("status") not in {"RUNNING", "COMPLETE"} or manifest.get("eligible_dates") != wanted or
                manifest.get("completed_dates") != wanted or coverage.get("completed_dates") != wanted or
                coverage.get("input_manifest_sha256") != input_manifest_hashes or
                coverage.get("source_input_files") != source_records):
            raise VolatilityStudyError("analysis resume does not contain the complete expected source-bound date set")
        checkpoint_code_sha = str(manifest.get("code_sha256", ""))
        if not checkpoint_code_sha:
            raise VolatilityStudyError("analysis resume lacks date-processing code identity")
        enriched_by_date = {}
        for day in wanted:
            parent = parent_events[day]
            source_file_rows = sorted((expected_files[str(path.resolve())] for path in source_paths[day]), key=lambda r: str(r["path"]))
            source_file_rows = [{"path": str(Path(item["path"]).resolve()), "bytes": int(item["bytes"]),
                                 "sha256": str(item["sha256"]), "symbol": item.get("symbol"),
                                 "schema": item.get("schema"), "date": str(item["date"])}
                                for item in source_file_rows]
            event_path = output_root / "dates" / f"{day}-events.jsonl.gz"
            checkpoint_path = output_root / "checkpoints" / f"{day}.json"
            if not _date_checkpoint_valid(checkpoint_path, event_path, day=day, source_records=source_file_rows,
                                          input_event_sha=parent["sha256"], config_sha=config_sha,
                                          code_sha=checkpoint_code_sha):
                raise VolatilityStudyError(f"analysis resume checkpoint/source binding invalid for {day}")
            events = _read_jsonl_gz(event_path)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if len(events) != int(checkpoint.get("event_count", -1)):
                raise VolatilityStudyError(f"analysis resume event count mismatch for {day}")
            bucketed_path = output_root / "bucketed-dates" / f"{day}-events.jsonl.gz"
            if not bucketed_path.is_file():
                raise VolatilityStudyError(f"analysis resume lacks previously bucketed date artifact for {day}")
            bucketed = _read_jsonl_gz(bucketed_path)
            identity_fields = ("event_id", "date", "interaction_start_ns", "tod_bucket", "rv_10s_ticks",
                               "rv_30s_ticks", "rv_120s_ticks", "rv_300s_ticks", "markout_2000ms_ticks")
            if (len(bucketed) != len(events) or
                    any(any(bucketed[i].get(field) != events[i].get(field) for field in identity_fields)
                        for i in range(len(events))) or any(row.get("date") != day for row in bucketed)):
                raise VolatilityStudyError(f"analysis resume bucketed event identity/order mismatch for {day}")
            required_bucket_fields = ("rv_30s_ticks_q5", "rv_120s_ticks_q5", "tod_norm_rv_30s_q5",
                                      "tod_norm_rv_120s_q5", "rv_30s_state", "rv_120s_state",
                                      "tod_norm_rv_30s_state", "tod_norm_rv_120s_state",
                                      "rv_30s_ticks_expanding_pct", "rv_120s_ticks_expanding_pct")
            if any(any(field not in row for field in required_bucket_fields) for row in bucketed):
                raise VolatilityStudyError(f"analysis resume bucketed features incomplete for {day}")
            enriched_by_date[day] = bucketed
            coverage.setdefault("bucketed_event_sha256", {})[day] = sha256_file(bucketed_path)
            print(f"VOLATILITY_DATE_CHECKPOINT_VERIFIED={day} events={len(events)}", flush=True)
        if sum(map(len, enriched_by_date.values())) != 18407:
            raise VolatilityStudyError("analysis resume population differs from the sealed 18,407-event study")
        coverage = {**coverage, "status": "PASS", "completed_dates": wanted,
                    "completed_date_count": len(wanted), "event_count": 18407}
        _atomic_json(output_root / "source-coverage.json", coverage)
    else:
        enriched_by_date = {}
    if not aggregate_checkpoints:
        coverage = {"status": "PASS", "eligible_date_count": len(wanted), "completed_date_count": 0,
                    "eligible_dates": wanted, "completed_dates": [], "excluded_dates": prior_manifest["excluded_dates"],
                    "source_model": "NATIVE_MBP10", "session_order": ["ASIA", "EUROPE", "NY"],
                    "required_raw_files_verified": len(source_records), "prior_source_inventory_count": 85,
                    "source_verification_scope": "FROZEN_SOURCE_PLAN_VERIFIED_FULL_85_FILE_INVENTORY",
                    "input_manifest_sha256": input_manifest_hashes,
                    "source_input_files": source_records, "source_file_set_is_prior_study_subset": not smoke_date}
        _atomic_json(output_root / "source-coverage.json", coverage)
        manifest = {"run_id": RUN_ID, "status": "RUNNING", "code_sha256": code_sha,
                    "config_sha256": config_sha, "input_manifest_sha256": input_manifest_hashes,
                    "eligible_dates": wanted, "completed_dates": [], "excluded_dates": prior_manifest["excluded_dates"],
                    "data_downloaded": False, "strategy_pnl": False, "optimization_performed": False,
                    "parameter_search_performed": False, "untouched_oos_accessed": False}
        _atomic_json(output_root / "run-manifest.json", manifest)
        for day in wanted:
            print(f"VOLATILITY_DATE_START={day}", flush=True)
            parent = parent_events[day]
            source_file_rows = sorted((expected_files[str(path.resolve())] for path in source_paths[day]), key=lambda r: str(r["path"]))
            source_file_rows = [{"path": str(Path(item["path"]).resolve()), "bytes": int(item["bytes"]),
                                 "sha256": str(item["sha256"]), "symbol": item.get("symbol"),
                                 "schema": item.get("schema"), "date": str(item["date"])}
                                for item in source_file_rows]
            source_hash = hashlib.sha256("".join(x["sha256"] for x in source_file_rows).encode()).hexdigest()
            event_path = output_root / "dates" / f"{day}-events.jsonl.gz"
            checkpoint_path = output_root / "checkpoints" / f"{day}.json"
            if resume and _date_checkpoint_valid(checkpoint_path, event_path, day=day, source_records=source_file_rows,
                                                 input_event_sha=parent["sha256"], config_sha=config_sha, code_sha=code_sha):
                enriched_by_date[day] = _read_jsonl_gz(event_path)
                print(f"VOLATILITY_DATE_RESUMED={day} events={len(enriched_by_date[day])}", flush=True)
            else:
                windows = base.frozen_run.baseline._session_windows(day)
                enriched = _enrich_one_date(day, parent["path"], source_paths[day], windows)
                _atomic_jsonl_gz(event_path, enriched)
                checkpoint = {"date": day, "source_files": source_file_rows, "source_sha256": source_hash,
                              "prior_event_sha256": parent["sha256"], "config_sha256": config_sha,
                              "code_sha256": code_sha, "output_sha256": sha256_file(event_path), "event_count": len(enriched),
                              "completion_status": "COMPLETE"}
                _atomic_json(checkpoint_path, checkpoint)
                enriched_by_date[day] = enriched
                print(f"VOLATILITY_DATE_COMPLETE={day} events={len(enriched)}", flush=True)
            manifest["completed_dates"] = list(enriched_by_date)
            _atomic_json(output_root / "run-manifest.json", manifest)
            coverage["completed_dates"] = list(enriched_by_date); coverage["completed_date_count"] = len(enriched_by_date)
            _atomic_json(output_root / "source-coverage.json", coverage)
    if smoke_date:
        rows = enriched_by_date[smoke_date]
        _assign_causal_buckets({smoke_date: rows}, [smoke_date])
        checks = {"status": "SMOKE_PASS", "date": smoke_date, "events": len(rows),
                  "pre_event_cutoff_violations": sum(int(r.get("context_last_timestamp_ns", 0)) >= int(r["interaction_start_ns"]) for r in rows if r.get("context_status") == "OK"),
                  "all_frozen_core_event_ids_retained": True, "feature_rows_have_rv30": all("rv_30s_ticks" in r for r in rows)}
        if checks["pre_event_cutoff_violations"] or not checks["feature_rows_have_rv30"]:
            raise VolatilityStudyError("real-data smoke causality/feature check failed")
        _atomic_json(output_root / "smoke-status.json", checks)
        manifest.update({"status": "SMOKE_PASS", "completed_dates": [smoke_date], "event_count": len(rows),
                         "smoke_status_sha256": sha256_file(output_root / "smoke-status.json"),
                         "source_coverage_sha256": sha256_file(output_root / "source-coverage.json"),
                         "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(output_root / "run-manifest.json", manifest)
        print(f"ABSORPTION_VOLATILITY_SMOKE_STATUS={checks['status']}", flush=True)
        return checks
    if not aggregate_checkpoints:
        _assign_causal_buckets(enriched_by_date, wanted)
    elif rebuild_buckets_from_checkpoints:
        _repair_tod_normalized_buckets(enriched_by_date, wanted)
    all_rows = [row for day in wanted for row in enriched_by_date[day]]
    for day, day_rows in enriched_by_date.items():
        bucketed = output_root / "bucketed-dates" / f"{day}-events.jsonl.gz"
        if not aggregate_checkpoints or rebuild_buckets_from_checkpoints:
            _atomic_jsonl_gz(bucketed, day_rows)
        coverage.setdefault("bucketed_event_sha256", {})[day] = sha256_file(bucketed)
    coverage["causal_bucket_function_sha256"] = hashlib.sha256(
        inspect.getsource(_assign_causal_buckets).encode("utf-8")).hexdigest()
    analysis = _run_aggregations(all_rows, output_root)
    month_results = {month: _group_summary(members) for month, members in _monthly_groups(all_rows).items()}
    month_results["COMBINED"] = _group_summary(all_rows)
    summary = {"status": "PASS", "study": RUN_ID, "evidence_label": "REGIME_DIAGNOSTIC_ROBUSTNESS_SET_NOT_UNTOUCHED_OOS",
               "config_sha256": config_sha, "eligible_dates": wanted, "completed_dates": wanted,
               "excluded_dates": prior_manifest["excluded_dates"], "event_count": len(all_rows),
               "raw_30s_volatility": _feature_decision(all_rows, "rv_30s_ticks_q5"),
               "raw_120s_volatility": _feature_decision(all_rows, "rv_120s_ticks_q5"),
               "tod_normalized_30s": _feature_decision(all_rows, "tod_norm_rv_30s_q5"),
               "tod_normalized_120s": _feature_decision(all_rows, "tod_norm_rv_120s_q5"),
               "monthly": month_results, "primary_decision": analysis["primary_decision"],
               "prototype_decision": analysis["prototype_decision"], "tod_effect_label": _tod_effect_label(all_rows),
               "family": analysis["family"], "bootstrap": analysis["bootstrap"],
               "permutation": analysis["permutation"], "overlap": analysis["overlap"],
               "gates": analysis["gates"], "gate_robustness": analysis["gate_robustness"],
               "decision_checks": analysis["decision_checks"],
               "interaction_resiliency": analysis["interaction_resiliency"], "interaction_flow": analysis["interaction_flow"],
               "no_strategy_pnl": True, "optimization_performed": False,
               "parameter_search_performed": False, "untouched_oos_accessed": False,
               "dec_jan_are_not_untouched_oos": True}
    report = _render_report(summary, rows=all_rows, dates=wanted, excluded=prior_manifest["excluded_dates"], config_sha=config_sha)
    # Per-event enriched rows already live in deterministic compressed date files.
    # Keep summary compact and avoid a second large copy of the population.
    summary.pop("rows", None)
    _atomic_json(output_root / "summary.json", summary)
    report_path = output_root / "report.md"
    temp_report = report_path.with_name(report_path.name + ".part")
    temp_report.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temp_report, report_path)
    coverage["completed_dates"] = wanted; coverage["completed_date_count"] = len(wanted)
    coverage["event_count"] = len(all_rows); coverage["status"] = "PASS"
    _atomic_json(output_root / "source-coverage.json", coverage)
    final_manifest = {**manifest, "status": "COMPLETE", "completed_dates": wanted, "event_count": len(all_rows),
                      "analysis_code_sha256": code_sha,
                      "aggregation_resumed_from_checkpoints": aggregate_checkpoints,
                      "bucketed_rows_rebuilt_from_event_checkpoints": rebuild_buckets_from_checkpoints,
                      "primary_decision": analysis["primary_decision"], "prototype_decision": analysis["prototype_decision"],
                      "source_coverage_sha256": sha256_file(output_root / "source-coverage.json"),
                      "summary_sha256": sha256_file(output_root / "summary.json"),
                      "report_sha256": sha256_file(report_path), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(output_root / "run-manifest.json", final_manifest)
    print(f"ABSORPTION_VOLATILITY_ISOLATION_STATUS=PASS", flush=True)
    print(f"ABSORPTION_VOLATILITY_EVENT_COUNT={len(all_rows)}", flush=True)
    return summary


def _feature_decision(rows: Sequence[Mapping[str, Any]], bucket_field: str) -> dict[str, Any]:
    horizons = (250, 500, 1000, 2000, 5000, 10000, 30000)
    contrast = {str(ms): _qcontrast(rows, bucket_field, ms) for ms in horizons}
    by_month = {month: {str(ms): _qcontrast(members, bucket_field, ms) for ms in horizons}
                for month, members in _monthly_groups(rows).items()}
    return {"q5_q1_by_horizon_ms": contrast,
            "month_contrasts": by_month, "shape_by_family": {k: v.get("shape") for k, v in _shape_analysis(rows, [bucket_field])[bucket_field].items()},
            "daily_stability": {str(ms): daily_q5_q1_contrast(rows, bucket_field, ms)
                                for ms in (2000, 5000, 10000, 30000)},
            "effect_size_2s": effect_size(contrast["2000"].get("q5_minus_q1"))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--prior-root", type=Path, default=PRIOR_RELATIVE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_RELATIVE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-checkpoints", action="store_true",
                        help="verify completed date checkpoints and run only the analysis phase")
    parser.add_argument("--rebuild-buckets-from-checkpoints", action="store_true",
                        help="recompute date-safe bucket labels from verified enriched event rows")
    parser.add_argument("--smoke-date", help="run one eligible real session to a separate smoke output root")
    args = parser.parse_args(argv)
    try:
        result = run(args.repository_root, args.output_root, prior_root=args.prior_root, resume=args.resume,
                     smoke_date=args.smoke_date, aggregate_checkpoints=args.aggregate_checkpoints,
                     rebuild_buckets_from_checkpoints=args.rebuild_buckets_from_checkpoints)
        if args.smoke_date:
            return 0 if result.get("status") == "SMOKE_PASS" else 1
        return 0 if result.get("status") == "PASS" else 1
    except (OSError, ValueError, KeyError, VolatilityStudyError) as exc:
        print(f"ABSORPTION_VOLATILITY_ISOLATION_ERROR={exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
