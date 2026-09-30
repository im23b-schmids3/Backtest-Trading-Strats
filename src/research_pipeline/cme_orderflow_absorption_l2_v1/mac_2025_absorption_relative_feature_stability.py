"""Cross-period stability audit for four frozen live absorption families.

Descriptive only: this module performs no parameter search, PnL calculation,
data acquisition, or access outside local Spring/October 2025 ES MBP-10.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import mac_2025_absorption_relative_normalization as norm
from . import mac_2025_es_only_train_baseline as baseline

RUN_ID = "CMEOrderflow_ABSORPTION_RELATIVE_FEATURE_STABILITY_2025_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
OLD_ROOT = Path("research_runs/CMEOrderflow_ABSORPTION_RELATIVE_NORMALIZATION_APR_OCT_2025_V1")
SCHEME = "spring-march-april-then-october-prior-date-only-tod-v2"
HORIZONS = (500, 1_000, 2_000, 5_000, 10_000, 30_000)
DAILY_HORIZONS = (2_000, 5_000, 10_000, 30_000)
FEATURES: dict[str, dict[str, str]] = {
    "TREND_EFFICIENCY_5S": {"raw": "ER_5S", "relative": "ER_5S_GLOBAL_PERCENTILE", "kind": "global"},
    "TREND_EFFICIENCY_30S": {"raw": "ER_30S", "relative": "ER_30S_GLOBAL_PERCENTILE", "kind": "global"},
    "AGGRESSION_TO_DEPTH_1S": {"raw": "AGGRESSION_TO_DEPTH_1S", "relative": "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE", "kind": "tod"},
    "AGGRESSION_TO_DEPTH_2S": {"raw": "AGGRESSION_TO_DEPTH_2S", "relative": "AGGRESSION_TO_DEPTH_2S_TOD_PERCENTILE", "kind": "tod"},
    "RELATIVE_TRADE_INTENSITY_5S": {"raw": "TRADES_PER_SECOND_5S", "relative": "TRADE_INTENSITY_TOD_PERCENTILE", "kind": "tod"},
    "RELATIVE_TRADE_INTENSITY_30S": {"raw": "TRADES_PER_SECOND_30S", "relative": "TRADE_INTENSITY_30S_TOD_PERCENTILE", "kind": "tod_custom"},
    "DEPTH_NORMALIZED_MLOFI_1S": {"raw": "MLOFI_1S_DEPTH_NORMALIZED", "relative": "MLOFI_1S_TOD_PERCENTILE", "kind": "tod_abs"},
    "DEPTH_NORMALIZED_MLOFI_5S": {"raw": "MLOFI_5S_DEPTH_NORMALIZED", "relative": "MLOFI_PERCENTILE_TOD", "kind": "tod_abs"},
}
RV_REL = "RV_30S_TOD_PERCENTILE"
FAMILIES = tuple(norm.LIVE_TO_TAPE.values())  # live strategy aliases stored in event["live_family"]
PERIODS = ("SPRING_2025", "OCTOBER_2025")
GATE_NAMES = ("BASELINE", "A", "B", "C", "D", "E")


class StabilityError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    os.replace(temp, path)


def _manifest_and_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, Any]]:
    manifest_path = norm.DATA_ROOT / baseline.MANIFEST_NAME
    manifest, requests = baseline._manifest(norm.DATA_ROOT)
    if manifest.get("status") != "COMPLETE":
        raise StabilityError("native ES source manifest is not COMPLETE")
    train_dates = sorted(d for d in baseline.TRAIN_DATES if d.startswith(("2025-03-", "2025-04-")))
    oct_dates = sorted({str(r.get("session_date")) for r in requests.values()
                        if isinstance(r, dict) and r.get("category") == "VALIDATION"
                        and str(r.get("session_date", "")).startswith("2025-10-")})
    expected = train_dates + oct_dates
    if not train_dates or not oct_dates or len(set(expected)) != len(expected):
        raise StabilityError("Spring/October target date lists are empty or overlap")
    rows: dict[str, dict[str, Any]] = {}
    tapes: dict[str, Path] = {}
    missing: list[str] = []
    for day in expected:
        category = "TRAIN" if day in train_dates else "VALIDATION"
        candidates = [r for r in requests.values() if isinstance(r, dict) and r.get("session_date") == day
                      and r.get("category") == category and r.get("schema") == "mbp-10"]
        if len(candidates) != 1:
            missing.append(f"{day}: expected one {category} ES mbp-10 request, got {len(candidates)}")
            continue
        row = candidates[0]
        src = norm.DATA_ROOT / str(row.get("path", ""))
        if not src.is_file() or src.stat().st_size != int(row.get("bytes", -1)) or _sha(src) != row.get("sha256"):
            missing.append(f"{day}: native DBN missing/size/hash mismatch")
            continue
        contract = "ESH5" if day <= "2025-03-14" else "ESM5" if day <= "2025-04-21" else "ESZ5"
        from databento import DBNStore
        meta = DBNStore.from_file(src).metadata
        if meta.dataset != "GLBX.MDP3" or meta.schema != "mbp-10" or contract not in meta.symbols:
            missing.append(f"{day}: DBN identity mismatch, expected {contract}")
            continue
        tape_root = norm.TRAIN_TAPE_ROOT if category == "TRAIN" else norm.OCT_TAPE_ROOT
        tape = tape_root / f"{day}-candidate-tape.npz"
        if not tape.is_file():
            missing.append(f"{day}: sealed candidate tape missing")
            continue
        with np.load(tape, allow_pickle=False) as z:
            tm = json.loads(str(z["metadata_json"].item()))
        if tm.get("date") != day or tm.get("source_sha256") != row["sha256"] or tm.get("semantic_sha256") != norm.EXPECTED_TAPE_SEMANTIC_SHA:
            missing.append(f"{day}: candidate tape source/semantic mismatch")
            continue
        rows[day] = {"source": src, "source_sha256": row["sha256"], "category": category,
                     "schema": "mbp-10", "symbol": contract, "record_count": row.get("verification", {}).get("record_count")}
        tapes[day] = tape
    if missing:
        raise StabilityError("incomplete local coverage: " + "; ".join(missing))
    dependency = next((r for r in requests.values() if isinstance(r, dict) and r.get("session_date") == "2025-10-06"
                       and r.get("category") == "DEPENDENCY"), None)
    if not dependency or not (norm.DATA_ROOT / str(dependency.get("path", ""))).is_file():
        raise StabilityError("required October profile dependency 2025-10-06 is absent")
    coverage = {"source_manifest": str(manifest_path), "source_manifest_sha256": _sha(manifest_path),
                "spring_dates": train_dates, "october_dates": oct_dates, "missing_or_incomplete": [],
                "dependency_only": ["2025-10-06"], "dependency_used_as_target": False,
                "source_files": {d: {**{k: str(v) if isinstance(v, Path) else v for k, v in rows[d].items()},
                                     "bytes": rows[d]["source"].stat().st_size} for d in expected},
                "candidate_tapes": {d: {"path": str(tapes[d]), "sha256": _sha(tapes[d])} for d in expected},
                "data_downloaded": False, "oos_2026_accessed": False}
    return rows, tapes, coverage


def _tod_key(day: str, row: Mapping[str, Any]) -> str:
    windows = baseline._session_windows(day)
    session = str(row.get("session", ""))
    start = windows.get(session, (0, 0))[0]
    return f"{session}:{(int(row['timestamp_ns']) - start) // norm.TOD_BIN_NS}"


def _rank_custom(events: list[dict[str, Any]], samples: list[dict[str, Any]], history: dict[str, dict[str, list[float]]], day: str) -> None:
    # Extra date-safe ranks for 30s trade intensity and 1s normalized MLOFI.
    custom = (("TRADES_PER_SECOND_30S", "TRADE_INTENSITY_30S_TOD_PERCENTILE", False),
              ("MLOFI_1S_DEPTH_NORMALIZED", "MLOFI_1S_TOD_PERCENTILE", True))
    for raw, out, absolute in custom:
        sorted_hist = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in history.get(raw, {}).items()}
        for event in events:
            val = event.get(raw)
            if val is None or not math.isfinite(float(val)):
                event[out] = None
                continue
            key = _tod_key(day, {"timestamp_ns": event["interaction_start_ns"], "session": event.get("session")})
            dist = sorted_hist.get(key, np.empty(0))
            event[out] = norm._rank_sorted(abs(float(val)) if absolute else float(val), dist)
        windows = baseline._session_windows(day)
        for sample in samples:
            val = sample.get(raw)
            if val is None or not math.isfinite(float(val)):
                continue
            key = _tod_key(day, sample)
            history.setdefault(raw, {}).setdefault(key, []).append(abs(float(val)) if absolute else float(val))


def _clear_ranks(events: list[dict[str, Any]]) -> None:
    rank_fields = {out for spec in FEATURES.values() for out in (spec["relative"],)} | {
        "TRADE_INTENSITY_30S_TOD_PERCENTILE", "MLOFI_1S_TOD_PERCENTILE", RV_REL,
        "RV_120S_TOD_PERCENTILE", "RV_30S_GLOBAL_PERCENTILE", "RV_120S_GLOBAL_PERCENTILE",
        "AGGRESSIVE_VOLUME_1S_TOD_PERCENTILE", "AGGRESSIVE_VOLUME_2S_TOD_PERCENTILE",
        "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE", "AGGRESSION_TO_DEPTH_2S_TOD_PERCENTILE",
        "TOP5_DEPTH_TOD_PERCENTILE", "TRADE_INTENSITY_TOD_PERCENTILE", "VOLUME_INTENSITY_TOD_PERCENTILE",
        "MLOFI_PERCENTILE_TOD", "ER_5S_GLOBAL_PERCENTILE", "ER_30S_GLOBAL_PERCENTILE",
        "RESILIENCY_PERCENTILE"}
    for event in events:
        for field in rank_fields:
            event.pop(field, None)


def _derive_mlofi_states(events: list[dict[str, Any]]) -> None:
    """Orient signed normalized flow to the candidate's expected outcome side."""
    for event in events:
        direction = 1.0 if event.get("direction") == "BUYER_ABSORPTION" else -1.0
        for window in ("1S", "5S"):
            value = event.get(f"MLOFI_{window}_DEPTH_NORMALIZED")
            directional = None if value is None else float(value) * direction
            event[f"MLOFI_{window}_DIRECTIONAL"] = directional
        value = event.get("MLOFI_5S_DIRECTIONAL")
        event["MLOFI_REVERSAL_STATE"] = ("NEUTRAL" if value is None or float(value) == 0 else
                                          "SUPPORTS_REVERSAL" if float(value) > 0 else "OPPOSES_REVERSAL")


def _load_old_checkpoint(day: str, source_sha: str, tape: Path, config_sha: str) -> dict[str, Any] | None:
    path = OLD_ROOT / "checkpoints" / f"{day}.json.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if (payload.get("status") != "COMPLETE" or payload.get("source_sha256") != source_sha
            or payload.get("tape_sha256") != _sha(tape) or payload.get("config_sha256") != config_sha):
        return None
    norm._refresh_checkpoint_path_outcomes(payload, tape, day)
    payload["period"] = "SPRING_2025" if day.startswith(("2025-03-", "2025-04-")) else "OCTOBER_2025"
    return payload


def _load_or_extract(day: str, row: Mapping[str, Any], tape: Path, configs: Mapping[str, Any], config_sha: str,
                     root: Path) -> tuple[dict[str, Any], str, float]:
    cp_path = root / "checkpoints" / f"{day}.json.gz"
    try:
        with gzip.open(cp_path, "rt", encoding="utf-8") as f:
            cp = json.load(f)
        if (cp.get("status") == "COMPLETE" and cp.get("source_sha256") == row["source_sha256"]
                and cp.get("tape_sha256") == _sha(tape) and cp.get("config_sha256") == config_sha
                and cp.get("calibration_scheme") == SCHEME):
            norm._refresh_checkpoint_path_outcomes(cp, tape, day)
            return cp, "RESUMED", float(cp.get("elapsed_seconds", 0))
    except (OSError, json.JSONDecodeError):
        pass
    old = _load_old_checkpoint(day, row["source_sha256"], tape, config_sha)
    if old is not None:
        old.update({"status": "RAW_FEATURES_REUSED", "config_sha256": config_sha,
                    "calibration_scheme": SCHEME, "reused_from": str(OLD_ROOT / "checkpoints" / f"{day}.json.gz")})
        old["raw_feature_origin"] = "HASH_VERIFIED_PRIOR_STUDY_CHECKPOINT"
        return old, "REUSED_RAW_FEATURE_CHECKPOINT", 0.0
    payload, elapsed = norm._extract_date(day, row["source"], tape, row["source_sha256"], configs)
    payload["period"] = "SPRING_2025" if day.startswith(("2025-03-", "2025-04-")) else "OCTOBER_2025"
    payload.update({"status": "RAW_FEATURES_REUSED", "config_sha256": config_sha,
                    "calibration_scheme": SCHEME, "elapsed_seconds": elapsed,
                    "raw_feature_origin": "EXTRACTED_FROM_LOCAL_NATIVE_DBN"})
    return payload, "EXTRACTED_NATIVE_DBn", elapsed


def _finite(row: Mapping[str, Any], key: str) -> bool:
    x = row.get(key)
    return x is not None and math.isfinite(float(x))


def _bucket(value: float, representation: str) -> str:
    if representation == "quintile":
        return f"Q{min(5, int(value * 5) + 1)}"
    if representation == "tercile":
        return ("LOW", "MID", "HIGH")[min(2, int(value * 3))]
    return "LOW_0_20" if value < .2 else "MID_20_80" if value < .8 else "HIGH_80_100"


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    return norm._summarize_values(v for v in values if v is not None)


def _event_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"event_count": len(rows), "active_dates": len({x.get("date") for x in rows if x.get("date") is not None}),
            "markouts": {str(h): _stats(x.get("markouts_ticks", {}).get(str(h)) for x in rows) for h in HORIZONS}}


def _group_table(events: list[dict[str, Any]], feature: str, rep: str, *, group_keys: tuple[str, ...] = ("period", "family")) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for e in events:
        if not _finite(e, feature):
            continue
        key = tuple(str(e.get(k, "")) for k in group_keys) + (_bucket(float(e[feature]), rep),)
        grouped.setdefault(key, []).append(e)
    out: dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        node = out
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = _event_summary(rows)
    return out


def _daily_relationship(events: list[dict[str, Any]], feature: str, rep: str = "tercile") -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        if _finite(e, feature): by.setdefault(str(e["date"]), []).append(e)
    result = {}
    effects = []
    for day, rows in sorted(by.items()):
        bins: dict[str, list[dict[str, Any]]] = {}
        for e in rows: bins.setdefault(_bucket(float(e[feature]), rep), []).append(e)
        low, high = ("Q1", "Q5") if rep == "quintile" else (("LOW_0_20", "HIGH_80_100") if rep == "broad" else ("LOW", "HIGH"))
        entry = {"event_count": len(rows), "buckets": {k: {"event_count": len(v), "markouts": {
            str(h): _stats(x["markouts_ticks"].get(str(h)) for x in v) for h in DAILY_HORIZONS}
            } for k, v in sorted(bins.items())}}
        lo_m = entry["buckets"].get(low, {}).get("markouts", {}).get("5000", {}).get("mean")
        hi_m = entry["buckets"].get(high, {}).get("markouts", {}).get("5000", {}).get("mean")
        effect = None if lo_m is None or hi_m is None else hi_m - lo_m
        entry["high_minus_low_5s"] = effect
        result[day] = entry
        if effect is not None: effects.append(effect)
    return {"dates": result, "positive_dates": sum(x > 0 for x in effects),
            "negative_dates": sum(x < 0 for x in effects), "insufficient_dates": len(by) - len(effects),
            "median_daily_effect": float(np.median(effects)) if effects else None,
            "p25_daily": float(np.quantile(effects, .25)) if effects else None,
            "p75_daily": float(np.quantile(effects, .75)) if effects else None}


def _daily_feature_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, spec in FEATURES.items():
        key = spec["relative"]
        result[name] = {rep: _daily_relationship(events, key, rep) for rep in ("quintile", "tercile", "broad")}
    return result


def _sign_consistency(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, spec in FEATURES.items():
        feature = spec["relative"]
        result[name] = {}
        for family in FAMILIES:
            by_period = {}
            for period in PERIODS:
                rows = [e for e in events if e.get("live_family") == family and e["period"] == period and _finite(e, feature)]
                low = [e for e in rows if float(e[feature]) <= .2]
                high = [e for e in rows if float(e[feature]) >= .8]
                low5 = _stats(e["markouts_ticks"].get("5000") for e in low).get("mean")
                high5 = _stats(e["markouts_ticks"].get("5000") for e in high).get("mean")
                effect = None if low5 is None or high5 is None else high5 - low5
                adequately_sampled = len(rows) >= 20 and len(low) >= 5 and len(high) >= 5
                by_period[period] = {"event_count": len(rows), "low_state_n": len(low), "high_state_n": len(high),
                                     "adequately_sampled": adequately_sampled,
                                     "low_5s_mean": low5, "high_5s_mean": high5, "high_minus_low": effect}
            a, b = (by_period[p].get("high_minus_low") for p in PERIODS)
            enough = all(by_period[p]["adequately_sampled"] for p in PERIODS)
            classification = "INSUFFICIENT" if not enough or a is None or b is None else "CONSISTENT_SAME_DIRECTION" if a * b > 0 else "OPPOSITE" if a * b < 0 else "MIXED"
            result[name][family] = {"periods": by_period, "classification": classification}
    return result


def _mfe_mae(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, spec in FEATURES.items():
        key = spec["relative"]
        result[name] = {}
        for state in ("LOW", "MID", "HIGH"):
            rows = [e for e in events if _finite(e, key) and _bucket(float(e[key]), "tercile") == state]
            horizons = {}
            for h in (1_000, 2_000, 5_000, 10_000, 30_000):
                values = [e.get("mfe_mae_ticks", {}).get(str(h)) for e in rows]
                valid = [v for v in values if isinstance(v, dict)]
                horizons[str(h)] = {metric: _stats(v.get(metric) for v in valid) for metric in ("mfe", "mae")}
                for metric in ("mfe", "mae"):
                    vals = [float(v[metric]) for v in valid]
                    horizons[str(h)][metric]["probability_ge"] = {str(t): (sum(x >= t for x in vals) / len(vals) if vals else None)
                                                                     for t in (1, 2, 4, 8)}
            result[name][state] = {"event_count": len(rows), "horizons": horizons}
    return result


def _barriers(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, spec in FEATURES.items():
        key = spec["relative"]
        result[name] = {}
        for state in ("LOW", "MID", "HIGH"):
            rows = [e for e in events if _finite(e, key) and _bucket(float(e[key]), "tercile") == state]
            result[name][state] = {"event_count": len(rows), "barriers": {f"+{up}/-{down}": {
                "favorable_first_probability": (sum(e.get("barriers", {}).get(f"+{up}/-{down}") == "UP" for e in rows) / len(rows) if rows else None),
                "adverse_first_probability": (sum(e.get("barriers", {}).get(f"+{up}/-{down}") == "DOWN" for e in rows) / len(rows) if rows else None),
                "no_touch_or_tie_probability": (sum(e.get("barriers", {}).get(f"+{up}/-{down}") in {"NO_TOUCH", "TIE", None} for e in rows) / len(rows) if rows else None)
                } for up, down in norm.BARRIERS}}
    return result


def _volatility_control(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for feature_name, spec in FEATURES.items():
        key = spec["relative"]
        out[feature_name] = {}
        for vol_state in ("LOW", "MID", "HIGH"):
            subset = [e for e in events if _finite(e, RV_REL) and _bucket(float(e[RV_REL]), "tercile") == vol_state]
            out[feature_name][vol_state] = {"event_count": len(subset), "feature_buckets": _group_table(subset, key, "tercile", group_keys=("period", "family"))}
    return out


def _interactions(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    left = "ER_5S_GLOBAL_PERCENTILE"
    states = ("SUPPORTS_REVERSAL", "NEUTRAL", "OPPOSES_REVERSAL")
    cells_a = {}
    for tercile in ("LOW", "MID", "HIGH"):
        for state in states:
            subset = [e for e in events if _finite(e, left) and _bucket(float(e[left]), "tercile") == tercile
                      and e.get("MLOFI_REVERSAL_STATE") == state]
            cells_a[f"{tercile}_ER_X_{state}"] = _event_summary(subset)
    result["INTERACTION_A_TREND_X_MLOFI"] = {"features": [left, "MLOFI_REVERSAL_STATE"], "cells": cells_a,
                                              "classification": "DESCRIPTIVE_COARSE_TERCILES_AND_FIXED_STATES_ONLY"}
    left, right = "AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE", "ER_5S_GLOBAL_PERCENTILE"
    cells_b = {}
    for a in ("LOW", "MID", "HIGH"):
        for b in ("LOW", "MID", "HIGH"):
            subset = [e for e in events if _finite(e, left) and _finite(e, right)
                      and _bucket(float(e[left]), "tercile") == a and _bucket(float(e[right]), "tercile") == b]
            cells_b[f"{a}_AGGRESSION_X_{b}_ER"] = _event_summary(subset)
    result["INTERACTION_B_AGGRESSION_X_TREND"] = {"features": [left, right], "cells": cells_b,
                                                   "classification": "DESCRIPTIVE_COARSE_TERCILES_ONLY"}
    return result


def _overlap(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clusters: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for e in events:
        key = (int(e["interaction_start_ns"]), round(float(e.get("price") or 0.0) / norm.TICK), e.get("direction"))
        clusters.setdefault(key, []).append(e)
    duplicate = [rows for rows in clusters.values() if len(rows) > 1]
    dedup = [sorted(rows, key=lambda x: (x["family"], x["event_id"]))[0] for rows in clusters.values()]
    pooled = {}
    for label, subset in (("RAW_EVENTS", events), ("DEDUPLICATED_EVENT_CLUSTERS", dedup)):
        pooled[label] = {name: {rep: _group_table(subset, spec["relative"], rep, group_keys=())
                               for rep in ("quintile", "tercile", "broad")}
                         for name, spec in FEATURES.items()}
        pooled[label]["overall"] = _event_summary(subset)
    report = {"raw_event_count": len(events), "cluster_count": len(clusters), "overlap_cluster_count": len(duplicate),
              "events_in_overlap_clusters": sum(map(len, duplicate)), "duplicate_excess_events": len(events) - len(clusters),
              "cluster_definition": "exact interaction timestamp + tick-rounded reference price + direction",
              "overlap_examples": [{"timestamp_ns": k[0], "price_ticks": k[1], "direction": k[2],
                                    "families": sorted({x["family"] for x in v}), "event_ids": [x["event_id"] for x in v]}
                                   for k, v in list((k, v) for k, v in clusters.items() if len(v) > 1)[:100]],
              "raw_vs_deduplicated_5s_markout": {
                  "RAW_EVENTS": _stats(e["markouts_ticks"].get("5000") for e in events),
                  "DEDUPLICATED_EVENT_CLUSTERS": _stats(e["markouts_ticks"].get("5000") for e in dedup)},
              "pooled_feature_buckets": pooled}
    return report, dedup


def _permutation(events: list[dict[str, Any]], repetitions: int = 199) -> dict[str, Any]:
    rng = np.random.default_rng(20250930)
    result = {}
    for name, spec in FEATURES.items():
        feature = spec["relative"]
        observed, null = [], []
        for family in FAMILIES:
            for period in PERIODS:
                subset = [e for e in events if e.get("live_family") == family and e["period"] == period
                          and _finite(e, feature) and e["markouts_ticks"].get("5000") is not None]
                by_day: dict[str, list[dict[str, Any]]] = {}
                for e in subset: by_day.setdefault(e["date"], []).append(e)
                for rows in by_day.values():
                    if len(rows) < 5: continue
                    labels = np.asarray([min(4, int(float(e[feature]) * 5)) for e in rows], dtype=np.int8)
                    y = np.asarray([float(e["markouts_ticks"]["5000"]) for e in rows])
                    means = [float(y[labels == i].mean()) for i in (0, 4) if np.any(labels == i)]
                    if len(means) != 2: continue
                    observed.append(means[1] - means[0])
                    for _ in range(repetitions):
                        shuffled = rng.permutation(labels)
                        m = [float(y[shuffled == i].mean()) for i in (0, 4) if np.any(shuffled == i)]
                        if len(m) == 2: null.append(m[1] - m[0])
        n95 = float(np.quantile(np.abs(null), .95)) if null else None
        effect = float(np.mean(observed)) if observed else None
        result[name] = {"real_effect_mean_daily_high_minus_low_5s": effect, "daily_effect_count": len(observed),
                        "null_repetitions_per_date_family": repetitions, "null_absolute_p95": n95,
                        "exceeds_null_p95": bool(abs(effect) > n95) if effect is not None and n95 is not None else None,
                        "seed": 20250930, "permutation_unit": "labels shuffled within date/family, counts preserved"}
    return result


def _neighbor_stability(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, spec in FEATURES.items():
        key = spec["relative"]
        representations = {}
        for rep in ("quintile", "tercile", "broad"):
            rows = [e for e in events if _finite(e, key) and e["markouts_ticks"].get("5000") is not None]
            buckets: dict[str, list[float]] = {}
            for e in rows: buckets.setdefault(_bucket(float(e[key]), rep), []).append(float(e["markouts_ticks"]["5000"]))
            ordered = sorted(buckets, key=lambda x: x)
            effect = None if len(ordered) < 2 else float(np.mean(buckets[ordered[-1]]) - np.mean(buckets[ordered[0]]))
            representations[rep] = {"bucket_means": {k: float(np.mean(v)) for k, v in sorted(buckets.items())}, "high_minus_low": effect}
        effects = [x["high_minus_low"] for x in representations.values() if x["high_minus_low"] is not None]
        signs = {int(np.sign(x)) for x in effects if x != 0}
        spread = max(effects) - min(effects) if effects else 0
        signs_by_rep = {name: int(np.sign(item["high_minus_low"])) for name, item in representations.items()
                        if item["high_minus_low"] is not None and item["high_minus_low"] != 0}
        coarse_signs = {signs_by_rep[k] for k in ("broad", "tercile") if k in signs_by_rep}
        cls = ("INSUFFICIENT" if len(effects) < 2 else
               "ROBUST" if len(signs) == 1 and spread < max(1.0, abs(float(np.mean(effects)))) else
               "MODERATELY_STABLE" if len(signs) == 1 else
               "BOUNDARY_SENSITIVE" if len(coarse_signs) <= 1 and "quintile" in signs_by_rep else "UNSTABLE")
        out[name] = {"representations": representations, "classification": cls}
    return out


def _gate_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    def selected(e: Mapping[str, Any], gate: str) -> bool:
        er = e.get("ER_5S_GLOBAL_PERCENTILE")
        mlofi_state = e.get("MLOFI_REVERSAL_STATE")
        agg = e.get("AGGRESSION_TO_DEPTH_1S_TOD_PERCENTILE")
        intensity = e.get("TRADE_INTENSITY_TOD_PERCENTILE")
        high_er = er is not None and float(er) >= 2 / 3
        opposes = mlofi_state == "OPPOSES_REVERSAL"
        if gate == "BASELINE": return True
        if gate == "A": return not high_er
        if gate == "B": return not opposes
        if gate == "C": return not high_er and not opposes
        if gate == "D": return agg is not None and 1 / 3 <= float(agg) < 2 / 3
        if gate == "E": return not high_er and intensity is not None and float(intensity) >= 1 / 3
        raise AssertionError(gate)
    result = {}
    for gate in GATE_NAMES:
        kept = [e for e in events if selected(e, gate)]
        partitions = {p: [e for e in kept if e["period"] == p] for p in PERIODS}
        by_day: dict[str, list[dict[str, Any]]] = {}
        for event in kept:
            by_day.setdefault(str(event["date"]), []).append(event)
        daily_outcomes = {day: {"event_count": len(day_rows),
            "markouts": {str(h): _stats(e["markouts_ticks"].get(str(h)) for e in day_rows) for h in DAILY_HORIZONS}}
            for day, day_rows in sorted(by_day.items())}
        daily_5 = [v["markouts"]["5000"].get("mean") for v in daily_outcomes.values()
                   if v["markouts"]["5000"].get("mean") is not None]
        result[gate] = {"event_retention": len(kept) / len(events) if events else None,
                        "overall": _event_summary(kept), "periods": {p: _event_summary(rows) for p, rows in partitions.items()},
                        "families": {fam: _event_summary([e for e in kept if e.get("live_family") == fam]) for fam in FAMILIES},
                        "daily_stability_5s": _daily_relationship(kept, "ER_5S_GLOBAL_PERCENTILE", "tercile"),
                        "daily_outcome_stability": {"dates": daily_outcomes,
                            "positive_dates": sum(float(x) > 0 for x in daily_5),
                            "negative_dates": sum(float(x) < 0 for x in daily_5),
                            "median_daily_5s": float(np.median(daily_5)) if daily_5 else None,
                            "p25_daily_5s": float(np.quantile(daily_5, .25)) if daily_5 else None,
                            "p75_daily_5s": float(np.quantile(daily_5, .75)) if daily_5 else None},
                        "mfe_mae": {str(h): {m: _stats(e.get("mfe_mae_ticks", {}).get(str(h), {}).get(m) for e in kept)
                                                 for m in ("mfe", "mae")} for h in (1_000, 2_000, 5_000, 10_000, 30_000)},
                        "barriers": {f"+{u}/-{d}": {"up_first": sum(e.get("barriers", {}).get(f"+{u}/-{d}") == "UP" for e in kept),
                                                             "down_first": sum(e.get("barriers", {}).get(f"+{u}/-{d}") == "DOWN" for e in kept),
                                                             "no_touch_or_tie": sum(e.get("barriers", {}).get(f"+{u}/-{d}") in {"NO_TOUCH", "TIE", None} for e in kept)}
                                       for u, d in norm.BARRIERS}}
    # Fixed rules above deliberately treat missing measurements as not excluded except where a gate requires the value.
    return {"definitions": {"A": "exclude high ER_5S tercile", "B": "exclude negative MLOFI persistence directional state",
                             "C": "exclude both A and B", "D": "retain middle aggression/depth tercile only",
                             "E": "exclude high ER and retain middle/high 5s relative trade intensity"},
            "interpretation": "descriptive event-quality screens only; not production rules or PnL", "gates": result}


def _report_md(summary: Mapping[str, Any]) -> str:
    counts = summary["event_counts"]
    lines = ["# Spring + October 2025 Relative Feature Stability", "", "Descriptive frozen-event study; no PnL or parameter search.", "",
             f"- Spring dates: {len(summary['spring_dates'])}; October dates: {len(summary['october_dates'])}",
             f"- Event count: {summary['total_events']}; overlaps: {summary['overlap']['overlap_cluster_count']} clusters",
             f"- Primary decision: `{summary['primary_decision']}`",
             f"- Four-family gate decision: `{summary['gate_decision']}`", "", "## Family event counts", "",
             "| Family | Spring | October |", "|---|---:|---:|"]
    for fam in FAMILIES: lines.append(f"| {fam} | {counts[fam]['SPRING_2025']} | {counts[fam]['OCTOBER_2025']} |")
    lines += ["", "## Cross-period qualitative consistency", "", "| Feature | Consistent families | Mixed/opposite/insufficient |", "|---|---:|---:|"]
    for feature, fams in summary["family_results"].items():
        vals = [x["classification"] for x in fams.values()]
        lines.append(f"| {feature} | {vals.count('CONSISTENT_SAME_DIRECTION')} | {len(vals)-vals.count('CONSISTENT_SAME_DIRECTION')} |")
    lines += ["", "## Interpretation", "", summary["decision_rationale"], "",
              "No tested gate is promoted. Spring combines the locally available March and April 2025 sessions; the only other target period is October 2025."]
    return "\n".join(lines) + "\n"


def run(*, output_root: Path = OUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    root = output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    configs, config_sha = norm._load_live_configs()
    rows, tapes, coverage = _manifest_and_inputs()
    if smoke:
        first = coverage["spring_dates"][0]
        rows, tapes = {first: rows[first]}, {first: tapes[first]}
        coverage["spring_dates"] = [first]
        coverage["october_dates"] = []
    _write_json(root / "source-coverage.json", coverage)
    _write_json(root / "checkpoints" / "run-identity.json", {"run_id": RUN_ID, "calibration_scheme": SCHEME,
        "config_sha256": config_sha, "source_manifest_sha256": coverage["source_manifest_sha256"],
        "ordered_dates": list(rows), "resume_safe": True})
    norm.OUT_ROOT = root
    histories_global: dict[str, list[float]] = {}
    histories_tod: dict[str, dict[str, list[float]]] = {}
    custom_history: dict[str, dict[str, list[float]]] = {}
    events: list[dict[str, Any]] = []
    resume_stats = {"new_native_extractions": [], "reused_raw_checkpoints": [], "resumed_stability_checkpoints": []}
    timings: dict[str, float] = {}
    ordered = sorted(rows)
    for pos, day in enumerate(ordered, 1):
        print(f"[stability] {pos}/{len(ordered)} {day}", flush=True)
        cp, state, elapsed = _load_or_extract(day, rows[day], tapes[day], configs, config_sha, root)
        _clear_ranks(cp["events"])
        norm._add_percentiles(cp["events"], cp["samples"], histories_global, histories_tod, day)
        _rank_custom(cp["events"], cp["samples"], custom_history, day)
        for event in cp["events"]:
            event["period"] = "SPRING_2025" if day.startswith(("2025-03-", "2025-04-")) else "OCTOBER_2025"
        _derive_mlofi_states(cp["events"])
        cp.update({"status": "COMPLETE", "calibration_scheme": SCHEME, "config_sha256": config_sha,
                   "source_sha256": rows[day]["source_sha256"], "tape_sha256": _sha(tapes[day]),
                   "elapsed_seconds": elapsed, "ranked_after_date_extraction": True})
        if state == "EXTRACTED_NATIVE_DBn":
            cp["raw_feature_origin"] = "EXTRACTED_FROM_LOCAL_NATIVE_DBN"
        elif state == "REUSED_RAW_FEATURE_CHECKPOINT":
            cp["raw_feature_origin"] = "HASH_VERIFIED_PRIOR_STUDY_CHECKPOINT"
        elif state == "RESUMED" and cp.get("reused_from"):
            cp["raw_feature_origin"] = "HASH_VERIFIED_PRIOR_STUDY_CHECKPOINT"
        elif state == "RESUMED" and not cp.get("raw_feature_origin"):
            cp["raw_feature_origin"] = "EXTRACTED_FROM_LOCAL_NATIVE_DBN"
        _write_gzip(root / "checkpoints" / f"{day}.json.gz", cp)
        if state == "RESUMED":
            resume_stats["resumed_stability_checkpoints"].append(day)
            if cp.get("raw_feature_origin") == "HASH_VERIFIED_PRIOR_STUDY_CHECKPOINT":
                resume_stats["reused_raw_checkpoints"].append(day)
            else:
                resume_stats["new_native_extractions"].append(day)
        elif state == "EXTRACTED_NATIVE_DBn":
            resume_stats["new_native_extractions"].append(day)
        else:
            resume_stats["reused_raw_checkpoints"].append(day)
        timings[day] = elapsed
        events.extend(cp["events"])
        _write_json(root / "checkpoints" / "progress.json", {"status": "RUNNING", "completed_dates": ordered[:pos],
            "next_date": ordered[pos] if pos < len(ordered) else None, "target_date_count": len(ordered),
            "checkpoint_sha256": {d: _sha(root / "checkpoints" / f"{d}.json.gz") for d in ordered[:pos]},
            "source_manifest_sha256": coverage["source_manifest_sha256"], "config_sha256": config_sha,
            "calibration_scheme": SCHEME, "resume_stats": resume_stats})
        if smoke:
            return {"status": "SMOKE_PASS", "date": day, "events": len(cp["events"]), "checkpoint": str(root / "checkpoints" / f"{day}.json.gz")}
    if not events:
        raise StabilityError("no frozen live-family events found")
    for e in events:
        # Relative trade-intensity 30s is the date-safe normalized primary, and
        # RV remains a control only. Ensure the 1s MLOFI percentile is present.
        if "MLOFI_1S_TOD_PERCENTILE" not in e:
            raise StabilityError("missing causal 1s MLOFI percentile")
    counts = {fam: {period: sum(e.get("live_family") == fam and e["period"] == period for e in events) for period in PERIODS}
              for fam in FAMILIES}
    overlap, dedup = _overlap(events)
    family_results = _sign_consistency(events)
    daily = _daily_feature_report(events)
    feature_outputs: dict[str, Any] = {}
    for name, spec in FEATURES.items():
        raw, rel = spec["raw"], spec["relative"]
        feature_outputs[name] = {"raw_quintiles": _group_table(events, raw, "quintile"),
                                 "relative_quintiles": _group_table(events, rel, "quintile"),
                                 "relative_terciles": _group_table(events, rel, "tercile"),
                                 "relative_broad_states": _group_table(events, rel, "broad"),
                                 "family_cross_period_sign_consistency": family_results[name],
                                 "neighbor_stability": _neighbor_stability(events)}
    interactions = _interactions(events)
    mlofi_state_results = {period: {family: {state: _event_summary([e for e in events if e["period"] == period
        and e.get("live_family") == family and e.get("MLOFI_REVERSAL_STATE") == state])
        for state in ("SUPPORTS_REVERSAL", "NEUTRAL", "OPPOSES_REVERSAL")}
        for family in FAMILIES} for period in PERIODS}
    volatility = _volatility_control(events)
    mfe = _mfe_mae(events)
    barriers = _barriers(events)
    permutations = _permutation(events)
    neighbors = _neighbor_stability(events)
    gates = _gate_report(events)
    # Coarse qualitative decision: avoid treating event-count-weighted pooling as proof.
    cls = [x["classification"] for fs in family_results.values() for x in fs.values()]
    consistent = sum(x == "CONSISTENT_SAME_DIRECTION" for x in cls)
    sufficient = sum(x != "INSUFFICIENT" for x in cls)
    boundary = sum(v["classification"] in {"BOUNDARY_SENSITIVE", "UNSTABLE"} for v in neighbors.values())
    if sufficient < 8:
        primary = "INSUFFICIENT_EVIDENCE"
    elif boundary >= 5:
        primary = "RELATIVE_FEATURE_RELATIONSHIPS_BOUNDARY_SENSITIVE"
    elif consistent >= 20:
        primary = "RELATIVE_FEATURE_RELATIONSHIPS_CROSS_REGIME_ROBUST"
    elif consistent >= 10:
        primary = "RELATIVE_FEATURE_RELATIONSHIPS_PARTIALLY_ROBUST"
    elif len(set(cls) - {"INSUFFICIENT"}) > 1:
        primary = "RELATIVE_FEATURE_RELATIONSHIPS_FAMILY_SPECIFIC"
    else:
        primary = "NO_STABLE_RELATIVE_FEATURE_RELATIONSHIP"
    useful_gate = any((gates["gates"][g]["overall"]["markouts"]["5000"].get("mean") or 0) >
                      (gates["gates"]["BASELINE"]["overall"]["markouts"]["5000"].get("mean") or 0)
                      for g in GATE_NAMES[1:])
    gate_decision = "PROMISING_BUT_NOT_READY" if useful_gate else "NOT_USEFUL"
    # Keep pooled deduplicated results explicitly separate; family reports use raw events unchanged.
    dedup_summary = {"event_count": len(dedup), "markouts": {str(h): _stats(e["markouts_ticks"].get(str(h)) for e in dedup) for h in HORIZONS}}
    outputs = {
        "trend-efficiency.json": {k: v for k, v in feature_outputs.items() if k.startswith("TREND_EFFICIENCY")},
        "aggression-depth.json": {k: v for k, v in feature_outputs.items() if k.startswith("AGGRESSION_TO_DEPTH")},
        "relative-trade-intensity.json": {k: v for k, v in feature_outputs.items() if k.startswith("RELATIVE_TRADE_INTENSITY")},
        "normalized-mlofi.json": {"features": {k: v for k, v in feature_outputs.items() if k.startswith("DEPTH_NORMALIZED_MLOFI")},
                                  "reversal_states": mlofi_state_results},
        "volatility-control.json": volatility, "interaction-results.json": interactions,
        "family-results.json": family_results, "daily-stability.json": daily,
        "mfe-mae.json": mfe, "barriers.json": barriers, "overlap-audit.json": overlap,
        "permutation-results.json": permutations, "neighbor-stability.json": neighbors,
        "prototype-gates.json": gates,
    }
    for filename, value in outputs.items(): _write_json(root / filename, value)
    _write_gzip(root / "event-features.jsonl.gz", events)
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "data_limitation": "SPRING_2025 + OCTOBER_2025 ONLY",
        "spring_dates": coverage["spring_dates"], "october_dates": coverage["october_dates"],
        "frozen_config_identity": {"config_sha256": config_sha, "strategy_manifest_sha256": norm.EXPECTED_STRATEGY_MANIFEST_SHA,
                                   "runtime_contract_sha256": norm.EXPECTED_RUNTIME_CONTRACT_SHA},
        "total_events": len(events), "event_counts": counts, "trend_efficiency": feature_outputs,
        "aggression_depth": {k: v for k, v in feature_outputs.items() if k.startswith("AGGRESSION_TO_DEPTH")},
        "relative_trade_intensity": {k: v for k, v in feature_outputs.items() if k.startswith("RELATIVE_TRADE_INTENSITY")},
        "normalized_mlofi": {"features": {k: v for k, v in feature_outputs.items() if k.startswith("DEPTH_NORMALIZED_MLOFI")},
                             "reversal_states": mlofi_state_results},
        "volatility_control": volatility, "interactions": interactions, "family_results": family_results,
        "overlap": overlap, "deduplicated_result": dedup_summary, "prototype_gates": gates,
        "mfe_mae": mfe, "barriers": barriers, "daily_stability": daily, "permutation": permutations,
        "neighbor_stability": neighbors, "primary_decision": primary, "gate_decision": gate_decision,
        "decision_rationale": f"Family-period high/low directions consistent in {consistent}/{sufficient} adequately observed combinations; {boundary}/{len(neighbors)} variables were unstable across neighboring bucket representations. Fixed gates remain descriptive.",
        "resume_stats": resume_stats, "elapsed_extraction_seconds": sum(timings.values()),
        "optimization_performed": False, "parameter_search_performed": False, "pnl_optimization_performed": False,
        "data_downloaded": False, "oos_2026_accessed": False}
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(_report_md(summary), encoding="utf-8")
    manifest = {"run_id": RUN_ID, "status": "COMPLETE", "source_coverage_sha256": _sha(root / "source-coverage.json"),
                "frozen_config_sha256": config_sha, "calibration_scheme": SCHEME,
                "ordered_dates": ordered, "event_count": len(events), "family_counts": counts,
                "checkpoint_files": {p.name: _sha(p) for p in sorted((root / "checkpoints").glob("*.json.gz"))},
                "artifact_hashes": {p.name: _sha(p) for p in sorted(root.glob("*.json")) if p.name != "run-manifest.json"},
                "report_sha256": _sha(root / "report.md"), "event_features_sha256": _sha(root / "event-features.jsonl.gz"),
                "data_downloaded": False, "oos_2026_accessed": False}
    _write_json(root / "run-manifest.json", manifest)
    _write_json(root / "checkpoints" / "progress.json", {"status": "COMPLETE", "completed_dates": ordered,
        "target_date_count": len(ordered), "summary_sha256": _sha(root / "summary.json"), "resume_stats": resume_stats})
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
    print(json.dumps({k: result[k] for k in ("status", "primary_decision", "gate_decision", "total_events") if k in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
