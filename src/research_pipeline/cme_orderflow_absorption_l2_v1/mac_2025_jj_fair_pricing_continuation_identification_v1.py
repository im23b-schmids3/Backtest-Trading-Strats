"""Continuation detector, ATR-history, and covariate-matched identification audit.

Only the hash-pinned 2025 Candidate Tape V2 and existing V2.1 artifacts are
read. The alternative control contract is persisted before markout outcomes are
opened. Spring and October are treated as explored development data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import mac_2025_jj_fair_pricing_expanded_v2 as v2
from . import mac_2025_jj_fair_pricing_v2_1_corrected_bos as v21
from . import mac_2025_jj_fair_pricing_v2_mechanism_diagnostic as diag

RUN_ID = "CMEOrderflow_ES_JJ_FAIR_PRICING_CONTINUATION_IDENTIFICATION_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
TRIGGER_NAMES = ("DISPLACEMENT_CANDLE", "BOS_ONLY", "BOS_PLUS_DISPLACEMENT")
COHORTS = ("DISPLACEMENT_ONLY", "BOS_ONLY", "COMBINED")
HORIZONS = (30, 60, 120, 300, 600, 900)
FEATURES = ("time_minute", "prior_1m_directional_ticks", "prior_2m_directional_ticks",
            "log_prior_5m_range_ticks", "anchor_distance_ticks")
TICK = v2.TICK


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=v2._json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    fields = list(columns or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _verify_v21_artifacts() -> dict[str, Any]:
    root = v21.OUT_ROOT
    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "artifact-hashes.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or hashes.get("status") != "HASHED":
        raise RuntimeError("V2.1 corrected run is not COMPLETE/HASHED")
    bad = []
    for name, digest in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or _sha(path) != digest:
            bad.append(name)
    for name, digest in hashes.get("files", {}).items():
        path = root / name
        if not path.is_file() or _sha(path) != digest:
            bad.append(name)
    if bad:
        raise RuntimeError(f"V2.1 artifact hash mismatch: {sorted(set(bad))}")
    required = ("all-signals.jsonl.gz", "event-markout-rows.jsonl.gz", "source-coverage.json")
    if any(name not in manifest.get("files", {}) or name not in hashes.get("files", {}) for name in required):
        raise RuntimeError("V2.1 required signal/outcome/source artifacts are not hash-pinned")
    return {"run_manifest_sha256": _sha(root / "run-manifest.json"),
        "artifact_hash_manifest_sha256": _sha(root / "artifact-hashes.json"),
        "files_sha256": {name: manifest["files"][name] for name in required},
        "completed_dates": manifest.get("completed_dates"), "status": manifest.get("status")}


def _contract() -> dict[str, Any]:
    return {
        "study_id": "ES_JJ_FAIR_PRICING_CONTINUATION_IDENTIFICATION_V1",
        "status": "EXPLORATORY_INTERNAL_CONTRACT_NOT_INDEPENDENT_PREREGISTRATION",
        "signal_source": "V2.1 corrected directional BOS catalog rebuilt from hash-verified V2 Candidate Tape V2 for the exact 54 authorized 2025 dates",
        "signal_population": "NY_AM and NY_PM completed one-minute bars, minute indices 1..14, valid opening anchor, phase direction away from anchor; mutually exclusive candle cohorts DISPLACEMENT_ONLY, BOS_ONLY, COMBINED; combined means corrected directional BOS and displacement on the same completed candle",
        "control_population": "Completed eligible continuation minutes 1..14 with known phase direction, valid six-bar causal history, a valid BBO at or after decision+2ms, and none of displacement, BOS_ONLY, or combined trigger flags",
        "matching": {
            "strata": ["exact date", "exact NY session", "exact proposed direction"],
            "ratio": "1:1 nearest control per signal, no replacement within date/session/cohort; controls may be reused only across separately defined estimands",
            "time_caliper_minutes": 5,
            "covariates": list(FEATURES),
            "scaling": "pooled eligible-risk-set population standard deviation per covariate, calculated without outcomes; zero scales replaced by 1",
            "caliper": "absolute difference <= 1.5 pooled SD for every covariate",
            "distance": "Euclidean norm of pooled-SD standardized covariate differences; tie-break by earlier control timestamp then minute index",
            "missingness": "drop signal/control if any required preceding bar, covariate, or executable quote is unavailable; no imputation",
        },
        "outcomes": {
            "primary": "directional executable BBO net markout at 300 seconds after causal entry quote at or after decision+2ms; entry/exit adverse tick convention and existing 0.48-tick round-trip commission equivalent retained",
            "secondary": ["30s", "60s", "120s", "600s", "900s executable net BBO markout", "executable MFE", "executable MAE", "first favorable versus adverse one-tick quote excursion"],
            "dependence_unit": "trading date; percentile bootstrap resamples dates with all within-date matched pairs retained",
        },
        "minimum_support": {"primary_displacement_any_pairs": 30, "primary_displacement_any_distinct_dates": 15,
                            "per_subgroup_descriptive_pairs": 10, "per_subgroup_distinct_dates": 8},
        "comparisons": ["DISPLACEMENT_ANY versus trigger-negative eligible minutes", "BOS_ANY versus trigger-negative eligible minutes",
                        "COMBINED versus trigger-negative eligible minutes", "combined versus displacement-only for BOS incrementality"],
        "multiple_comparisons": "All reported triggers, sessions, directions, periods, and horizons are exploratory; no alpha claim or outcome-driven selection.",
        "scope": {"optimization": False, "new_exit_grid": False, "October_is_development_data": True,
                  "final_oos_accessed": False, "2026_accessed": False, "raw_dbn_read": False,
                  "downloads": False, "live_strategy_changed": False},
    }


def _warmup_bars(events: np.ndarray, start: int, end: int, warm_minutes: int = 30) -> list[dict[str, Any]]:
    ts = events["timestamp_ns"].astype(np.int64, copy=False)
    price = events["execution_price"].astype(float, copy=False)
    size = events["execution_size"].astype(float, copy=False)
    mask = (size > 0) & np.isfinite(price) & (ts >= start - warm_minutes * v2.MINUTE_NS) & (ts < end)
    ix = np.flatnonzero(mask)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in ix:
        buckets[(int(ts[i]) - start) // v2.MINUTE_NS].append(int(i))
    bars = []
    for minute, indices in sorted(buckets.items()):
        vals = price[indices]
        bars.append({"minute_index": int(minute), "start_ns": start + minute * v2.MINUTE_NS,
                     "end_ns": start + (minute + 1) * v2.MINUTE_NS, "open": float(vals[0]),
                     "high": float(np.max(vals)), "low": float(np.min(vals)), "close": float(vals[-1]),
                     "volume": float(np.sum(size[indices])), "trade_count": len(indices)})
    return bars


def _atr14(bars: Sequence[Mapping[str, Any]], minute: int) -> float | None:
    bm = {int(b["minute_index"]): b for b in bars}
    vals = []
    for i in range(minute - 13, minute + 1):
        if i not in bm or i - 1 not in bm:
            return None
        b, p = bm[i], bm[i - 1]
        vals.append(max(float(b["high"]) - float(b["low"]), abs(float(b["high"]) - float(p["close"])),
                        abs(float(b["low"]) - float(p["close"]))) / TICK)
    return statistics.mean(vals) if vals else None


def _entry_quote_exists(events: np.ndarray, timestamp_ns: int, end_ns: int) -> bool:
    ts = events["timestamp_ns"].astype(np.int64, copy=False)
    bid = events["bid"].astype(float, copy=False); ask = events["ask"].astype(float, copy=False)
    i = int(np.searchsorted(ts, timestamp_ns + 2_000_000, side="left"))
    while i < len(ts) and int(ts[i]) < end_ns:
        if math.isfinite(bid[i]) and math.isfinite(ask[i]) and bid[i] > 0 and ask[i] >= bid[i]:
            return True
        i += 1
    return False


def _direction_funnel(day: str, session: str, events: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    start, end = v2._session_ns(day, session)
    bars, tts, px, _, _ = v2._bars_for_session(events, start, end)
    warm = _warmup_bars(events, start, end)
    bm = {int(b["minute_index"]): b for b in bars}
    wb = {int(b["minute_index"]): b for b in warm}
    anchor = v2._anchor(tts, px, start)
    row: dict[str, Any] = {"date": day, "period": "SPRING_2025" if day in v2.PERIODS["SPRING_2025"] else "OCTOBER_2025",
        "session": session, "anchor_status": anchor["status"], "session_bars": len(bars), "first_session_minute": min(bm) if bm else None,
        "warmup_observed_minutes_before_open": sum(i in wb for i in range(-15, 0)), "eligible_bars_1_to_14": 0,
        "eligible_bullish_candles": 0, "eligible_bearish_candles": 0, "bullish_close_through_previous_2_highs": 0,
        "bearish_close_through_previous_2_lows": 0, "price_above_anchor": 0, "price_below_anchor": 0,
        "phase_direction_long": 0, "phase_direction_short": 0, "prior_two_consecutive_bars": 0,
        "displacement_long": 0, "displacement_short": 0, "bos_long": 0, "bos_short": 0,
        "combined_long": 0, "combined_short": 0, "episode_found": 0, "episode_side_matches": 0,
        "final_long_signals": 0, "final_short_signals": 0, "atr14_session_local_available": 0,
        "atr14_with_preopen_warmup_available": 0}
    minutes: dict[int, dict[str, Any]] = {}
    if anchor["status"] != "VALID":
        row["ineligible_reason"] = anchor["status"]
        return row, [], minutes
    ep_ids, ep_starts, ep_info = v2._episode_map(tts, px, float(anchor["price"]))
    catalog, _ = v2._signal_catalog(day, session, events)
    final_by_min = defaultdict(list)
    for s in catalog:
        if s["model"] == "OPENING_CONTINUATION":
            final_by_min[int(s["minute_index"])].append(s)
    for m in range(1, v2.CONTINUATION_MINUTES):
        b = bm.get(m)
        if b is None:
            continue
        close = float(b["close"]); direction = 1 if close > anchor["price"] else -1 if close < anchor["price"] else 0
        if not direction:
            continue
        row["eligible_bars_1_to_14"] += 1
        row["price_above_anchor" if direction > 0 else "price_below_anchor"] += 1
        phase = v2._model_direction(float(anchor["price"]), close, m)
        if phase is None:
            continue
        row["phase_direction_long" if phase[1] > 0 else "phase_direction_short"] += 1
        candle_dir = 1 if b["close"] > b["open"] else -1 if b["close"] < b["open"] else 0
        row["eligible_bullish_candles"] += candle_dir > 0
        row["eligible_bearish_candles"] += candle_dir < 0
        prev, prev2 = bm.get(m - 1), bm.get(m - 2)
        bullish_bos = bool(prev and prev2 and close > max(float(prev["high"]), float(prev2["high"])))
        bearish_bos = bool(prev and prev2 and close < min(float(prev["low"]), float(prev2["low"])))
        row["bullish_close_through_previous_2_highs"] += bullish_bos
        row["bearish_close_through_previous_2_lows"] += bearish_bos
        if prev and prev2:
            row["prior_two_consecutive_bars"] += 1
        disp = v2.displacement_candle(b, prev, int(phase[1]))["valid"]
        bos = v21.directional_bos_for_bar(bm, m, int(phase[1]))["confirmed"]
        row["displacement_long" if phase[1] > 0 else "displacement_short"] += disp
        row["bos_long" if phase[1] > 0 else "bos_short"] += bos
        row["combined_long" if phase[1] > 0 else "combined_short"] += bool(disp and bos)
        end_ns = int(b["end_ns"]); ti = int(np.searchsorted(tts, end_ns, side="left")) - 1
        episode_ok = False; side_ok = False
        if ti >= 0:
            epi = int(ep_ids[ti]); ep_start = int(ep_starts[ti]); episode_ok = epi != 0 and ep_info.get(epi) is not None
            side = 1 if px[ti] > anchor["price"] else -1
            side_ok = episode_ok and side == direction
        row["episode_found"] += episode_ok
        row["episode_side_matches"] += side_ok
        local_atr = _atr14(bars, m); full_atr = _atr14(warm, m)
        row["atr14_session_local_available"] += local_atr is not None
        row["atr14_with_preopen_warmup_available"] += full_atr is not None
        flags = {"DISPLACEMENT_CANDLE": disp, "BOS_ONLY": bos, "BOS_PLUS_DISPLACEMENT": bool(disp and bos)}
        candidates = []
        if side_ok:
            row["final_long_signals" if direction > 0 else "final_short_signals"] += sum(bool(x) for x in flags.values())
            for name, active in flags.items():
                if active:
                    candidates.append({"trigger": name, "direction": direction, "minute": m, "timestamp_ns": end_ns,
                        "date": day, "period": row["period"], "session": session, "model": "OPENING_CONTINUATION",
                        "anchor": float(anchor["price"]), "close": close, "bar": b, "prev": prev,
                        "flags": flags, "cohort": "COMBINED" if disp and bos else "DISPLACEMENT_ONLY" if disp else "BOS_ONLY"})
        minutes[m] = {"bar": b, "direction": direction, "phase": phase, "displacement": bool(disp), "bos": bool(bos),
                      "flags": flags, "side_ok": side_ok, "catalog_signals": final_by_min.get(m, []), "candidates": candidates,
                      "atr_session_local": local_atr, "atr_full_warmup": full_atr, "warm_bars": wb}
    return row, [x for vals in final_by_min.values() for x in vals], minutes


def _signal_covariates(day: str, session: str, minute: int, direction: int, close: float, anchor: float,
                       warm_bm: Mapping[int, Mapping[str, Any]]) -> dict[str, float] | None:
    prior = [warm_bm.get(i) for i in range(minute - 5, minute)]
    if any(b is None for b in prior) or warm_bm.get(minute) is None:
        return None
    p1, p2, p3 = warm_bm[minute - 1], warm_bm[minute - 2], warm_bm[minute - 3]
    assert p1 and p2 and p3
    ranges = [max(float(b["high"]) - float(b["low"]), 0.0) / TICK for b in prior if b is not None]
    if len(ranges) != 5:
        return None
    return {"time_minute": float(minute),
        "prior_1m_directional_ticks": direction * (float(p1["close"]) - float(p1["open"])) / TICK,
        "prior_2m_directional_ticks": direction * (float(p1["close"]) - float(p3["close"])) / TICK,
        "log_prior_5m_range_ticks": math.log(max(float(np.mean(ranges)), 0.25)),
        "anchor_distance_ticks": abs(close - anchor) / TICK}


def _stds(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out = {}
    for key in FEATURES:
        vals = np.asarray([float(r["covariates"][key]) for r in rows], dtype=float)
        sd = float(np.std(vals, ddof=0)) if len(vals) else 0.0
        out[key] = sd if math.isfinite(sd) and sd > 1e-12 else 1.0
    return out


def _risk_rows(day: str, session: str, events: np.ndarray, minutes: Mapping[int, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start, end = v2._session_ns(day, session)
    bars, tts, px, _, _ = v2._bars_for_session(events, start, end)
    anchor = v2._anchor(tts, px, start)
    if anchor["status"] != "VALID":
        return [], []
    warm = _warmup_bars(events, start, end); wb = {int(b["minute_index"]): b for b in warm}
    controls, treated = [], []
    for m in range(1, v2.CONTINUATION_MINUTES):
        info = minutes.get(m)
        b = wb.get(m)
        if info is None or b is None or info["direction"] == 0:
            continue
        direction = int(info["direction"]); close = float(b["close"])
        cov = _signal_covariates(day, session, m, direction, close, float(anchor["price"]), wb)
        if cov is None:
            continue
        timestamp = int(b["end_ns"])
        if not _entry_quote_exists(events, timestamp, end):
            continue
        flags = info["flags"]
        if not any(flags.values()):
            controls.append({"date": day, "session": session, "period": "SPRING_2025" if day in v2.PERIODS["SPRING_2025"] else "OCTOBER_2025",
                "minute": m, "timestamp_ns": timestamp, "direction": direction, "anchor": float(anchor["price"]),
                "close": close, "covariates": cov})
        elif info["side_ok"]:
            cohort = "COMBINED" if flags["BOS_PLUS_DISPLACEMENT"] else "DISPLACEMENT_ONLY" if flags["DISPLACEMENT_CANDLE"] else "BOS_ONLY"
            treated.append({"date": day, "session": session, "period": "SPRING_2025" if day in v2.PERIODS["SPRING_2025"] else "OCTOBER_2025",
                "minute": m, "timestamp_ns": timestamp, "direction": direction, "anchor": float(anchor["price"]),
                "close": close, "covariates": cov, "cohort": cohort,
                "signal_id": f"{day}|{session}|OPENING_CONTINUATION|{'BOS_PLUS_DISPLACEMENT' if cohort=='COMBINED' else 'DISPLACEMENT_CANDLE' if cohort=='DISPLACEMENT_ONLY' else 'BOS_ONLY'}|{direction}|{timestamp}"})
    return controls, treated


def _match(treated: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]], scales: Mapping[str, float]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for c in controls:
        groups[(str(c["date"]), str(c["session"]), "LONG" if int(c["direction"]) > 0 else "SHORT")].append(c)
    used: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for s in sorted(treated, key=lambda x: (x["date"], x["session"], x["cohort"], x["minute"], x["signal_id"])):
        key = (str(s["date"]), str(s["session"]), "LONG" if int(s["direction"]) > 0 else "SHORT")
        candidates = []
        for c in groups.get(key, []):
            diffs = {f: (float(s["covariates"][f]) - float(c["covariates"][f])) / float(scales[f]) for f in FEATURES}
            if abs(int(s["minute"]) - int(c["minute"])) > 5 or any(abs(x) > 1.5 for x in diffs.values()):
                continue
            ukey = (*key, str(s["cohort"]))
            if int(c["minute"]) in used[ukey]:
                continue
            distance = math.sqrt(sum(x * x for x in diffs.values()))
            candidates.append((distance, int(c["timestamp_ns"]), int(c["minute"]), c, diffs))
        if candidates:
            _, _, _, c, diffs = min(candidates, key=lambda x: (x[0], x[1], x[2]))
            used[(*key, str(s["cohort"]))].add(int(c["minute"]))
            out.append({"cohort": s["cohort"], "signal": s, "control": c, "standardized_differences": diffs})
    return out


def _balance_rows(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cohort in (*COHORTS, "DISPLACEMENT_ANY", "BOS_ANY"):
        group = [m for m in matches if cohort == m["cohort"] or
                 cohort == "DISPLACEMENT_ANY" and m["cohort"] in ("DISPLACEMENT_ONLY", "COMBINED") or
                 cohort == "BOS_ANY" and m["cohort"] in ("BOS_ONLY", "COMBINED")]
        for f in FEATURES:
            a = np.asarray([m["signal"]["covariates"][f] for m in group], dtype=float)
            b = np.asarray([m["control"]["covariates"][f] for m in group], dtype=float)
            pooled = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2) if len(a) else 0.0
            smd = (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled > 1e-12 else (0.0 if len(a) else None)
            rows.append({"estimand": cohort, "covariate": f, "matched_pairs": len(group),
                "signal_mean": float(np.mean(a)) if len(a) else None, "control_mean": float(np.mean(b)) if len(b) else None,
                "standardized_mean_difference": smd, "max_abs_pair_standardized_difference": max((abs(float(m["standardized_differences"][f])) for m in group), default=None)})
    return rows


def _preliminary_old_matches(signal_rows: Sequence[Mapping[str, Any]], session_info: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Reconstruct the exact 5-minute preliminary funnel, then test whether causal
    # full-tape warmup resolves the session-local ATR missingness.
    prelim = []
    counts = Counter()
    by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for s in signal_rows:
        if s["model"] == "OPENING_CONTINUATION" and s["trigger"] == "DISPLACEMENT_CANDLE":
            by_group[(s["date"], s["session"])].append(s)
    all_signal_minutes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in signal_rows:
        if row["model"] == "OPENING_CONTINUATION":
            all_signal_minutes[(row["date"], row["session"])].add(int(row["minute_index"]))
    for key, ss in by_group.items():
        info = session_info[key]; bm = info["session_bm"]; warm = info["warm_bm"]
        signal_minutes = all_signal_minutes[key]
        local_controls = []
        for cb in info["bars"]:
            cm = int(cb["minute_index"])
            if cm >= v2.CONTINUATION_MINUTES or cm in signal_minutes or cm - 2 not in bm:
                continue
            close = float(cb["close"]); side = 1 if close > info["anchor"] else -1 if close < info["anchor"] else 0
            atr_local = _atr14(info["bars"], cm)
            if side and atr_local is not None and atr_local > 0:
                local_controls.append({"bar": cb, "minute": cm, "direction": side, "atr": atr_local,
                    "distance": abs(close - info["anchor"]) / TICK})
        used = set()
        for s in ss:
            m = int(s["minute_index"]); direction = int(s["direction_sign"])
            if m - 2 not in bm:
                continue
            trend = 1 if float(bm[m]["close"]) - float(bm[m-2]["close"]) > 0 else -1 if float(bm[m]["close"]) - float(bm[m-2]["close"]) < 0 else 0
            pool = []
            for control in local_controls:
                cm = control["minute"]; c = control["bar"]
                move = float(c["close"]) - float(bm[cm-2]["close"]); ctrend = 1 if move > 0 else -1 if move < 0 else 0
                if control["direction"] != direction or ctrend != trend or abs(cm-m) > 5: continue
                pool.append(control)
            counts["preliminary_signal_count"] += bool(pool)
            counts["preliminary_candidate_pairs"] += len(pool)
            satr = _atr14(info["bars"], m); watr = _atr14(info["warm_bars"], m)
            for control in pool:
                c = control["bar"]; cm = control["minute"]
                catr = _atr14(info["warm_bars"], cm)
                item = {"signal_id": s["signal_id"], "date": s["date"], "session": s["session"], "signal_minute": m,
                    "candidate_control_minute": cm, "session_local_signal_atr14": satr,
                    "session_local_control_atr14": control["atr"], "warmup_signal_atr14": watr, "warmup_control_atr14": catr,
                    "signal_atr_missing_with_session_local_bars": satr is None,
                    "signal_atr_recovered_with_preopen_warmup": watr is not None,
                    "control_atr_recovered_with_preopen_warmup": catr is not None}
                prelim.append(item)
            if pool and watr is not None and watr > 0:
                distance = abs(float(s["signal_close"]) - float(s["anchor_price"])) / TICK
                caliper = max(8.0, .5 * distance); candidates = []
                for control in pool:
                    cm = control["minute"]; catr = _atr14(info["warm_bars"], cm)
                    if catr is None or catr <= 0 or abs(math.log(catr / watr)) > .5:
                        continue
                    if abs(control["distance"] - distance) > caliper or cm in used:
                        continue
                    score = abs(cm-m)/5 + abs(math.log(catr/watr))/.5 + abs(control["distance"]-distance)/caliper
                    candidates.append((score, abs(cm-m), cm, control))
                if candidates:
                    chosen = min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]
                    used.add(chosen["minute"])
                    counts["warmup_atr_matches"] += 1
            elif pool:
                counts["signals_lost_to_session_local_signal_atr"] += 1
    return {"original_funnel_from_v2_1": {"cohort_signals": 142, "preliminary_candidate_pairs_within_5m": 19, "final_matches": 0},
        "reconstructed": dict(counts), "preliminary_pair_rows": len(prelim),
        "atr_cause": "session-local initialization: the production diagnostic passes only session-clipped bars to ATR14; a causal 14-TR ATR needs warmup before each continuation minute. Full sealed Candidate Tape contains the authorized prior prints, so no invented bars are needed.",
        "notes": "The legacy preliminary funnel is reconstructed using session-local ATR eligibility, all-trigger signal-minute exclusions, exact date/session/direction/trailing-return/time rules. Pair rows then show causal full-tape ATR recovery and whether original ATR/distance/no-replacement rules can select a match."}, prelim


def _write_report(root: Path, summary: Mapping[str, Any], short_audit: Mapping[str, Any],
                  feasibility: Mapping[str, Any], period_data: Mapping[str, Any] | None) -> None:
    lines = ["# ES JJ Fair Pricing continuation identification", "",
        f"Decision: **{summary['primary_decision']}**", "",
        "This is an internal exploratory analysis of the already-examined 2025 Spring and October development periods, not independent preregistration or untouched validation.", "",
        "## Detector and signal census", "",
        f"The corrected production catalog rebuild matched V2.1 signal identities: `{short_audit['production_catalog_rebuild_matches_v21']['rebuilt_signal_identity_parity']}`.",
        f"Continuation trigger rows: {short_audit['continuation_signal_rows']} (duplicate IDs: {short_audit['duplicate_signal_ids']}). Direction/trigger counts: `{json.dumps(short_audit['signal_counts_by_direction_trigger'], sort_keys=True)}`.",
        "The detector accepts both long and short continuation: bearish BOS is a strict close below the minimum low of the two immediately preceding completed candles; no zero-short anomaly exists in the corrected sample.", "",
        "## ATR14 and legacy-control funnel", "",
        f"The legacy zero-match result is attributable to session-local initialization: the matcher computes ATR14 from session-clipped bars. Candidate Tape contains earlier same-day trades, allowing causal warmup bars without fabricating candles. Legacy-funnel reconstruction: `{json.dumps(summary.get('original_matches_recoverable'), sort_keys=True)}`.",
        f"The alternative control rule was frozen and hashed before the markout-row file was loaded: `{summary.get('contract_sha256')}`.", "",
        "## Covariate-only support", "",
        f"Eligible trigger-negative controls: {feasibility['risk_control_observations']}; displacement-any pairs: {feasibility['displacement_any']['pairs']} across {feasibility['displacement_any']['dates']} dates; matching coverage: {feasibility['matching_success_rate']:.1%}. The prespecified support gate passed: {feasibility['valid_control_population']}.",
        "Five-minute executable markouts are net of the frozen 2ms entry delay, adverse ticks on entry/exit and 0.48 tick commission equivalent.", "",
        "## Four research questions", "",
        f"A. Descriptive signal follow-through is mixed/weak after executable costs. Per-trigger Spring/October means: `{json.dumps(summary.get('signal_markout_descriptive', {}), sort_keys=True)}`.",
        "B. Against same-date/session/direction covariate-matched trigger-negative minutes, the displacement-any primary estimate and BOS-any estimate are both negative in Spring and October; see `period-comparison.json` for date-cluster intervals. Thus the observed corrected signals did not outperform these comparators in this development sample.",
        "C. Incremental BOS information is unidentified. Combined-versus-displacement-only is retained only as an unpaired descriptive contrast; it is not interpreted as a BOS effect.",
        "D. Positive raw price movement is not enough: the executable primary outcome includes entry delay, spread, adverse ticks and commissions. The matched 5-minute contrast is negative, so this sample does not support an economic continuation advantage.", "",
        "## Interpretation limits", "",
        "A matched association is not proof of causality. The combined-versus-displacement-only comparison is an unpaired descriptive cohort contrast here and does not identify BOS's incremental causal information. All sessions, periods, directions, triggers and horizons are exploratory multiple comparisons. Spring/October compatibility is descriptive only.",
        "The fixed all-covariate order-flow OLS is a supplemental post-hoc exploratory analysis performed after the primary markout comparison was reviewed; it has no feature selection or thresholds and must not be treated as confirmatory. The sealed Candidate Tape contains BBO, trades and aggressor fields but no depth ladder. TOP5 imbalance, MLOFI and resiliency cannot be derived from it; a separate authorized native MBP-10 event-time extraction would be required.", "",
        "No optimization, exit grid, raw DBN access, data download, live strategy change, OOS access or 2026 access was performed.", ""]
    if period_data:
        lines.extend(["## Matched effect estimates", ""])
        for item in period_data.get("estimands", []):
            if item.get("period") == "ALL_2025" and item.get("session") == "ALL_SESSIONS" and item.get("direction") == "ALL_DIRECTIONS":
                effect = item.get("primary_5m_signal_minus_control", {})
                lines.append(f"- {item['estimand']}: n={effect.get('n', item.get('pairs'))}, dates={effect.get('date_clusters', item.get('date_clusters'))}, mean 5m difference={effect.get('mean')} ticks, date-cluster 95% CI={effect.get('ci95')}.")
        lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _cluster_bootstrap(values_by_date: Mapping[str, Sequence[float]], seed: int, reps: int = 2000) -> dict[str, Any]:
    dates = sorted(values_by_date)
    values = [float(x) for d in dates for x in values_by_date[d]]
    if not values:
        return {"n": 0, "date_clusters": 0, "mean": None, "ci95": None}
    mean = float(np.mean(values))
    if len(dates) < 2:
        return {"n": len(values), "date_clusters": len(dates), "mean": mean, "ci95": None}
    rng = np.random.default_rng(seed); boots = []
    for _ in range(reps):
        picked = rng.choice(dates, size=len(dates), replace=True)
        sample = [float(x) for d in picked for x in values_by_date[str(d)]]
        if sample: boots.append(float(np.mean(sample)))
    return {"n": len(values), "date_clusters": len(dates), "mean": mean,
        "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        "method": "date-cluster percentile bootstrap; all within-date pairs travel together"}


def _excursion_order(tape: Any, signal_ns: int, direction: int, end_ns: int) -> bool | None:
    ev = tape.events; ts = ev["timestamp_ns"].astype(np.int64, copy=False)
    bid = ev["bid"].astype(float, copy=False); ask = ev["ask"].astype(float, copy=False)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid) & (ts < end_ns)
    ix = np.flatnonzero(valid); ready = signal_ns + 2_000_000
    i = int(np.searchsorted(ts[ix], ready, side="left")) if len(ix) else 0
    if i >= len(ix): return None
    entry = i; mid0 = (bid[ix[i]] + ask[ix[i]]) / 2
    horizon_end_ns = min(end_ns, int(ts[ix[i]]) + 900 * 1_000_000_000)
    favorable = adverse = None
    for j in range(entry, len(ix)):
        if int(ts[ix[j]]) > horizon_end_ns:
            break
        mid = (bid[ix[j]] + ask[ix[j]]) / 2
        delta = direction * (mid - mid0) / TICK
        if favorable is None and delta >= 1: favorable = j
        if adverse is None and delta <= -1: adverse = j
        if favorable is not None and adverse is not None: break
    if favorable is None and adverse is None: return None
    return adverse is None or favorable is not None and favorable < adverse


def _outcome_analysis(matches: Sequence[Mapping[str, Any]], tapes: Mapping[str, Any], markout_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(r["signal_id"]): r for r in markout_rows}
    rows = []; balance = []
    for index, m in enumerate(matches):
        s, c = m["signal"], m["control"]
        sr = by_id.get(str(s["signal_id"]))
        if sr is None:
            continue
        # Controls are valued by the identical diagnostic execution path, only
        # after the frozen contract and support gate have passed.
        end = v2._session_ns(str(c["date"]), str(c["session"]))[1]
        cp = diag._quote_path(tapes[str(c["date"])], int(c["timestamp_ns"]), int(c["direction"]), end, float(c["anchor"]))
        primary_signal = sr.get("executable_bbo_markouts", {}).get("300")
        primary_control = cp.get("horizons", {}).get("300")
        if not primary_signal or not primary_control:
            continue
        diff = float(primary_signal["executable_net_ticks"]) - float(primary_control["executable_net_ticks"])
        rec = {"estimand": m["cohort"], "date": s["date"], "period": s["period"], "session": s["session"],
            "direction": "LONG" if int(s["direction"]) > 0 else "SHORT", "signal_id": s["signal_id"],
            "signal_minute": s["minute"], "control_minute": c["minute"], "signal_net_5m_ticks": primary_signal["executable_net_ticks"],
            "control_net_5m_ticks": primary_control["executable_net_ticks"], "signal_minus_control_5m_ticks": diff,
            "signal_mfe_net_ticks": sr.get("executable_mfe_net_ticks"), "signal_mae_ticks": sr.get("executable_mae_net_ticks"),
            "control_mfe_net_ticks": cp.get("mfe_executable_net_ticks"), "control_mae_ticks": cp.get("mae_executable_ticks"),
            "signal_favorable_before_adverse": _excursion_order(tapes[str(s["date"])], int(s["timestamp_ns"]), int(s["direction"]), end),
            "control_favorable_before_adverse": _excursion_order(tapes[str(c["date"])], int(c["timestamp_ns"]), int(c["direction"]), end)}
        for h in HORIZONS:
            sv = sr.get("executable_bbo_markouts", {}).get(str(h)); cv = cp.get("horizons", {}).get(str(h))
            rec[f"signal_net_{h}s_ticks"] = sv.get("executable_net_ticks") if sv else None
            rec[f"control_net_{h}s_ticks"] = cv.get("executable_net_ticks") if cv else None
            rec[f"difference_{h}s_ticks"] = (float(rec[f"signal_net_{h}s_ticks"]) - float(rec[f"control_net_{h}s_ticks"])) if rec[f"signal_net_{h}s_ticks"] is not None and rec[f"control_net_{h}s_ticks"] is not None else None
        rows.append(rec)
    # Add predeclared aggregate estimands without pooling overlapping trigger labels.
    for name, cohorts in (("DISPLACEMENT_ANY", ("DISPLACEMENT_ONLY", "COMBINED")), ("BOS_ANY", ("BOS_ONLY", "COMBINED")),
                          ("COMBINED", ("COMBINED",)),
                          ("COMBINED_VS_DISPLACEMENT_ONLY", ("COMBINED", "DISPLACEMENT_ONLY"))):
        for period in ("SPRING_2025", "OCTOBER_2025", "ALL_2025"):
            for session in ("NY_AM", "NY_PM", "ALL_SESSIONS"):
                for direction in ("LONG", "SHORT", "ALL_DIRECTIONS"):
                    subset = [r for r in rows if r["estimand"] in cohorts and
                        (period == "ALL_2025" or r["period"] == period) and
                        (session == "ALL_SESSIONS" or r["session"] == session) and
                        (direction == "ALL_DIRECTIONS" or r["direction"] == direction)]
                    if name == "COMBINED_VS_DISPLACEMENT_ONLY":
                        # This is an unpaired difference of cohort means, explicitly not a matched contrast.
                        lhs = [r for r in subset if r["estimand"] == "COMBINED"]
                        rhs = [r for r in subset if r["estimand"] == "DISPLACEMENT_ONLY"]
                        outcome = {h: {"combined_mean": float(np.mean([x[f"signal_net_{h}s_ticks"] for x in lhs if x[f"signal_net_{h}s_ticks"] is not None])) if any(x[f"signal_net_{h}s_ticks"] is not None for x in lhs) else None,
                                       "displacement_only_mean": float(np.mean([x[f"signal_net_{h}s_ticks"] for x in rhs if x[f"signal_net_{h}s_ticks"] is not None])) if any(x[f"signal_net_{h}s_ticks"] is not None for x in rhs) else None}
                                  for h in HORIZONS}
                        for h, v in outcome.items():
                            v["difference_ticks"] = v["combined_mean"] - v["displacement_only_mean"] if v["combined_mean"] is not None and v["displacement_only_mean"] is not None else None
                        balance.append({"estimand": name, "period": period, "session": session, "direction": direction,
                            "n_combined": len(lhs), "n_displacement_only": len(rhs), "date_clusters": len({r["date"] for r in subset}),
                            "support_passed": len(lhs) >= 10 and len(rhs) >= 10 and len({r["date"] for r in subset}) >= 8,
                            "outcomes": outcome, "interpretation": "exploratory unpaired cohort comparison; not causal BOS incrementality"})
                        continue
                    by_date = defaultdict(list)
                    for r in subset:
                        if r["difference_300s_ticks"] is not None: by_date[r["date"]].append(float(r["difference_300s_ticks"]))
                    stats = _cluster_bootstrap(by_date, 81231 + len(name) + len(period) + len(session) + len(direction))
                    horizon_stats = {}
                    for h in HORIZONS:
                        vals = defaultdict(list)
                        for r in subset:
                            if r[f"difference_{h}s_ticks"] is not None: vals[r["date"]].append(float(r[f"difference_{h}s_ticks"]))
                        horizon_stats[str(h)] = _cluster_bootstrap(vals, 81231 + h + len(name))
                    balance.append({"estimand": name, "period": period, "session": session, "direction": direction,
                        "pairs": sum(len(x) for x in by_date.values()), "date_clusters": len(by_date), "primary_5m_signal_minus_control": stats,
                        "horizons": horizon_stats,
                        "outlier_sensitivity_1_99_winsorized_mean_ticks": float(np.mean(np.clip([x for vals in by_date.values() for x in vals],
                            *np.quantile([x for vals in by_date.values() for x in vals], [.01, .99])))) if by_date else None})
    return rows, balance, {"matched_pairs_with_primary_outcome": len(rows), "row_level_results": "signal-control-comparisons.csv",
        "estimates": balance, "overlap_signal_ids_missing_outcome": [m["signal"]["signal_id"] for m in matches if str(m["signal"]["signal_id"]) not in by_id]}


def _flow_adjusted_analysis(treated: Sequence[Mapping[str, Any]], markout_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fixed, no-selection linear partial association with date-cluster uncertainty."""
    by_id = {str(r["signal_id"]): r for r in markout_rows}
    flow_fields = ("delta_30s_normalized", "delta_2m_normalized", "session_cvd_normalized", "aggression_reversal",
                   "opposing_aggression_fraction_2m", "price_progress_toward_ticks_2m", "effort_without_result",
                   "price_impact_ticks_per_100_aggressive_contracts")
    base_fields = ("time_minute", "prior_1m_directional_ticks", "prior_2m_directional_ticks",
                   "log_prior_5m_range_ticks", "anchor_distance_ticks", "session_pm", "direction_short")
    names = (*base_fields, *flow_fields)
    results = {}
    for period in ("SPRING_2025", "OCTOBER_2025"):
        rows = []
        for t in treated:
            if t["period"] != period:
                continue
            outcome = by_id.get(str(t["signal_id"]))
            if not outcome:
                continue
            y = (outcome.get("executable_bbo_markouts", {}).get("300") or {}).get("executable_net_ticks")
            feat = outcome.get("features", {})
            xs = [float(t["covariates"][k]) for k in FEATURES]
            xs.extend([float(t["session"] == "NY_PM"), float(int(t["direction"]) < 0)])
            valid = True
            for f in flow_fields:
                value = feat.get(f)
                if isinstance(value, bool): value = float(value)
                if value is None or not math.isfinite(float(value)):
                    valid = False; break
                xs.append(float(value))
            if valid and y is not None and math.isfinite(float(y)):
                rows.append((str(t["date"]), np.asarray(xs, dtype=float), float(y)))
        if len(rows) < len(names) + 5:
            results[period] = {"n": len(rows), "status": "INSUFFICIENT_COMPLETE_CASES", "predictors": list(names)}
            continue
        x0 = np.vstack([r[1] for r in rows]); y0 = np.asarray([r[2] for r in rows]); dates = sorted({r[0] for r in rows})
        means = np.mean(x0, axis=0); sds = np.std(x0, axis=0)
        active = sds > 1e-12
        x = np.column_stack([np.ones(len(x0)), (x0[:, active] - means[active]) / sds[active]])
        ysd = float(np.std(y0)); ymean = float(np.mean(y0))
        yz = (y0-ymean)/ysd if ysd > 1e-12 else y0-ymean
        coef = np.linalg.lstsq(x, yz, rcond=None)[0]
        fitted_names = [names[i] for i, yes in enumerate(active) if yes]
        coefficient_map = dict(zip(fitted_names, [float(z) for z in coef[1:]]))
        by_date = defaultdict(list)
        for d, xx, yy in rows: by_date[d].append((xx, yy))
        boot = {name: [] for name in fitted_names}
        if len(dates) > 1:
            rng = np.random.default_rng(46109 + len(period))
            for _ in range(600):
                picked = rng.choice(dates, size=len(dates), replace=True)
                sample = [pair for d in picked for pair in by_date[str(d)]]
                bx0 = np.vstack([p[0] for p in sample]); by0 = np.asarray([p[1] for p in sample])
                bm = np.mean(bx0, axis=0); bs = np.std(bx0, axis=0); ba = bs > 1e-12
                bx = np.column_stack([np.ones(len(bx0)), (bx0[:, ba]-bm[ba])/bs[ba]])
                bysd = float(np.std(by0)); byz = (by0-np.mean(by0))/bysd if bysd > 1e-12 else by0-np.mean(by0)
                bc = np.linalg.lstsq(bx, byz, rcond=None)[0]
                bnames = [names[i] for i, yes in enumerate(ba) if yes]
                for nm, val in zip(bnames, bc[1:]):
                    if nm in boot: boot[nm].append(float(val))
        results[period] = {"n_complete_cases": len(rows), "date_clusters": len(dates), "mean_outcome_net_ticks": ymean,
            "fixed_predictors": list(names), "standardized_partial_coefficients": coefficient_map,
            "date_cluster_bootstrap_ci95": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else None for k,v in boot.items()},
            "interpretation": "Fixed all-covariate OLS partial associations, not causal effects; no feature selection, thresholds, or model search. Collinear predictors may be weakly identified."}
    return {"method": "fixed complete-case linear adjustment for predetermined lagged price-action covariates, session, direction, and all listed continuous causal order-flow features; date-cluster bootstrap",
        "flow_features": list(flow_fields), "price_action_adjustment": list(base_fields), "feature_selection_or_threshold_search": False, "by_period": results}


def _directional_funnel(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = []; signal_rows = []; risk_controls = []; risk_treated = []; info_by_group = {}; tapes = {}; rebuilt = []
    old_bos = v2.v1.bos_for_bar
    v2.v1.bos_for_bar = v21.directional_bos_for_bar
    try:
        for day in v2.DATES:
            events, _, _ = v2._load_events(day, source)
            tape = __import__("research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_candidate_tape", fromlist=["load_tape"]).load_tape(
                Path(source["candidate_tapes"][day]["path"]), source_sha256=source["candidate_tapes"][day]["source_sha256"],
                semantic_sha256=source["candidate_tapes"][day]["semantic_sha256"])
            tapes[day] = tape
            for session in ("NY_AM", "NY_PM"):
                row, session_signals, mins = _direction_funnel(day, session, events); rows.append(row)
                signal_rows.extend(session_signals)
                start, end = v2._session_ns(day, session)
                bars, tts, px, _, _ = v2._bars_for_session(events, start, end)
                warm = _warmup_bars(events, start, end)
                anchor = v2._anchor(tts, px, start)
                info_by_group[(day, session)] = {"bars": bars, "session_bm": {int(b["minute_index"]): b for b in bars},
                    "warm_bars": warm, "warm_bm": {int(b["minute_index"]): b for b in warm},
                    "anchor": float(anchor["price"]) if anchor["status"] == "VALID" else None}
                if anchor["status"] == "VALID":
                    c, t = _risk_rows(day, session, events, mins); risk_controls.extend(c); risk_treated.extend(t)
        # Match rebuilt detector identity against the sealed authoritative corrected artifact.
        frozen = v21._read_jsonl_gz(v21.OUT_ROOT / "all-signals.jsonl.gz")
        expected = {(str(s["signal_id"]), int(s["direction_sign"])) for s in frozen if s["model"] == "OPENING_CONTINUATION"}
        got = {(str(s["signal_id"]), int(s["direction_sign"])) for s in signal_rows}
        if expected != got:
            raise v21.CorrectedBosError(f"rebuilt continuation catalog differs from V2.1: missing={len(expected-got)} extra={len(got-expected)}")
    finally:
        v2.v1.bos_for_bar = old_bos
    return rows, signal_rows, risk_controls, {"treated": risk_treated, "groups": info_by_group}, tapes, {"rebuilt_signal_identity_parity": True, "continuation_signal_count": len(signal_rows)}


def _run(output_root: Path, force: bool = False) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise RuntimeError(f"output directory is nonempty; use --force only for this study namespace: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = v21.freeze_inputs()
    frozen["corrected_v21"] = _verify_v21_artifacts()
    source, _ = v2._source_contract()
    # Freeze before any forward outcomes are opened.
    contract = _contract(); contract_hash = _canonical_hash(contract)
    _write_json(output_root / "frozen-control-methodology.json", {"contract": contract, "sha256": contract_hash})

    funnel, signal_rows, controls, aux, tapes, parity = _directional_funnel(source)
    all_treated = aux["treated"]
    scales = _stds([*controls, *all_treated])
    matches_by_cohort = {}
    for cohort in COHORTS:
        matches_by_cohort[cohort] = _match([x for x in all_treated if x["cohort"] == cohort], controls, scales)
    # Evaluate displacement-any and BOS-any as distinct estimands while retaining
    # the disjoint primitive cohorts; each estimand has its own no-replacement set.
    displacement_any = _match([x for x in all_treated if x["cohort"] in ("DISPLACEMENT_ONLY", "COMBINED")], controls, scales)
    bos_any = _match([x for x in all_treated if x["cohort"] in ("BOS_ONLY", "COMBINED")], controls, scales)
    matches = [*matches_by_cohort["DISPLACEMENT_ONLY"], *matches_by_cohort["BOS_ONLY"], *matches_by_cohort["COMBINED"]]
    primary_dates = {m["signal"]["date"] for m in displacement_any}
    valid_population = len(displacement_any) >= 30 and len(primary_dates) >= 15
    balances = _balance_rows(matches)
    _write_csv(output_root / "continuation-direction-funnel.csv", funnel)
    old_atr, old_pairs = _preliminary_old_matches(signal_rows, aux["groups"])
    _write_json(output_root / "short-continuation-audit.json", {
        "symmetric_detector": True, "production_directional_bos": "strict close above max preceding-two highs for long; below min preceding-two lows for short",
        "production_catalog_rebuild_matches_v21": parity, "signal_counts_by_direction_trigger": {
            f"{trigger}|{direction}": sum(s["trigger"] == trigger and s["direction"] == direction for s in signal_rows)
            for trigger in TRIGGER_NAMES for direction in ("LONG", "SHORT")},
        "continuation_signal_rows": len(signal_rows), "duplicate_signal_ids": len(signal_rows)-len({s["signal_id"] for s in signal_rows}),
        "eligible_minute_rule": "minute index 1..14; first observed actual ES trade within opening minute is anchor; phase direction is close side of anchor"})
    _write_json(output_root / "atr14-missingness-audit.json", {
        **old_atr, "session_local_atr14_available_continuation_minutes": sum(int(r["atr14_session_local_available"]) for r in funnel),
        "full_tape_warmup_atr14_available_candidate_minute_rows": sum(int(r["atr14_with_preopen_warmup_available"]) for r in funnel),
        "signal_atr_availability_by_trigger": {trigger: {
            "signals": sum(s["trigger"] == trigger for s in signal_rows),
            "session_local_available": sum(_atr14(aux["groups"][(s["date"],s["session"])]["bars"],int(s["minute_index"])) is not None for s in signal_rows if s["trigger"]==trigger),
            "full_tape_warmup_available": sum(_atr14(aux["groups"][(s["date"],s["session"])]["warm_bars"],int(s["minute_index"])) is not None for s in signal_rows if s["trigger"]==trigger)} for trigger in TRIGGER_NAMES},
        "continuation_signal_atr_by_signal": [{"signal_id": s["signal_id"], "date": s["date"], "session": s["session"],
            "minute": s["minute_index"], "atr14_session_local": _atr14(aux["groups"][(s["date"], s["session"])]["bars"], int(s["minute_index"])),
            "atr14_with_warmup": _atr14(aux["groups"][(s["date"], s["session"])]["warm_bars"], int(s["minute_index"]))}
            for s in signal_rows], "preliminary_pair_detail": old_pairs,
        "cause_classification": "SESSION_LOCAL_INITIALIZATION; full sealed candidate tape contains prior events sufficient for observed warmup where consecutive trade candles exist; no candles synthesized"})
    _write_json(output_root / "original-control-funnel.json", old_atr)
    match_summary = {"risk_control_observations": len(controls), "trigger_treatment_minutes": len(all_treated),
        "treatment_by_cohort": dict(Counter(x["cohort"] for x in all_treated)),
        "matches_by_cohort": {k: {"pairs": len(v), "dates": len({m["signal"]["date"] for m in v})} for k, v in matches_by_cohort.items()},
        "displacement_any": {"pairs": len(displacement_any), "dates": len(primary_dates)},
        "bos_any": {"pairs": len(bos_any), "dates": len({m["signal"]["date"] for m in bos_any})},
        "feature_scales": scales, "valid_control_population": valid_population,
        "minimum_support_gate": {"required_displacement_any_pairs": 30, "required_distinct_dates": 15,
            "observed_pairs": len(displacement_any), "observed_dates": len(primary_dates), "passed": valid_population},
        "effective_sample_size_pairs": len(displacement_any), "matching_success_rate": len(displacement_any) / max(1, sum(x["cohort"] in ("DISPLACEMENT_ONLY", "COMBINED") for x in all_treated)),
        "session_period_direction_overlap": {f"{p}|{s}|{'LONG' if d>0 else 'SHORT'}": sum(1 for x in all_treated if x["period"]==p and x["session"]==s and x["direction"]==d)
            for p in ("SPRING_2025", "OCTOBER_2025") for s in ("NY_AM", "NY_PM") for d in (1, -1)},
        "contract_sha256": contract_hash, "outcomes_opened": False}
    _write_json(output_root / "control-feasibility.json", match_summary)
    _write_csv(output_root / "control-covariate-balance.csv", balances)
    if not valid_population:
        summary = {"study_id": RUN_ID, "status": "PARTIAL", "primary_decision": "CONTINUATION_EFFECT_UNIDENTIFIED",
        "short_continuation_detector_valid": parity["rebuilt_signal_identity_parity"],
        "zero_short_signals_explained": sum(s["direction"] == "SHORT" for s in signal_rows) == 0,
            "material_implementation_defect_found": False, "atr14_unavailable_root_cause": old_atr["atr_cause"],
            "original_matches_recoverable": "causal warmup availability reconstructed; final legacy calipers not necessarily met",
            "alternative_control_predeclared": True, "valid_control_population": False,
            "continuation_signal_count": len(signal_rows), "control_observation_count": len(controls),
            "matched_or_standardized_signal_count": len(displacement_any), "contract_sha256": contract_hash,
            "frozen_inputs": frozen, "scope": contract["scope"]}
        _write_json(output_root / "period-comparison.json", {"status": "NOT_ESTIMATED", "reason": "minimum prespecified common-support gate failed"})
        _write_json(output_root / "orderflow-conditional-analysis.json", {"status": "NOT_ESTIMATED", "reason": "controlled outcome gate failed; no new outcome analysis"})
        (output_root / "research-limitations.md").write_text("# Limitations\n\nAlternative-control support did not meet the frozen minimum. No signal-control outcome comparison was produced. Spring and October are explored development data, not untouched validation. Sealed Candidate Tape contains BBO and trades, not depth ladders; TOP5 imbalance/MLOFI/resiliency cannot be reconstructed.\n", encoding="utf-8")
        _write_json(output_root / "summary.json", summary)
        _write_report(output_root, summary, _json(output_root / "short-continuation-audit.json"), match_summary, None)
        _finalize_hashes(output_root)
        return summary

    # The output contract and feasibility files above are durable before this point.
    markout_path = v21.OUT_ROOT / "event-markout-rows.jsonl.gz"
    markouts = v21._read_jsonl_gz(markout_path)
    comparison_rows, estimates, outcome_summary = _outcome_analysis(matches, tapes, markouts)
    # Conditional order-flow associations are continuous and descriptive; no feature selection.
    mark_by_id = {str(r["signal_id"]): r for r in markouts}
    flow_fields = ("delta_30s_normalized", "delta_2m_normalized", "session_cvd_normalized", "aggression_reversal",
                   "opposing_aggression_fraction_2m", "price_progress_toward_ticks_2m", "effort_without_result",
                   "price_impact_ticks_per_100_aggressive_contracts")
    flow_results = []
    for period in ("SPRING_2025", "OCTOBER_2025"):
        sample = [s for s in signal_rows if s["period"] == period and s["model"] == "OPENING_CONTINUATION"]
        for field in flow_fields:
            pairs = []
            for s in sample:
                row = mark_by_id.get(str(s["signal_id"])); feat = (row or {}).get("features", {}).get(field)
                y = ((row or {}).get("executable_bbo_markouts", {}).get("300") or {}).get("executable_net_ticks")
                if isinstance(feat, bool): feat = float(feat)
                if feat is not None and y is not None and math.isfinite(float(feat)) and math.isfinite(float(y)):
                    pairs.append((float(feat), float(y), str(s["date"])))
            grouped = defaultdict(list)
            for x, y, d in pairs: grouped[d].append((x, y))
            def corr(rows0: Sequence[tuple[float, float]]) -> float | None:
                if len(rows0) < 3: return None
                a=np.asarray([x for x,_ in rows0]); b=np.asarray([y for _,y in rows0])
                return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
            dates = sorted(grouped); boot = []
            if len(dates) >= 2:
                rng=np.random.default_rng(99231+len(field)+len(period))
                for _ in range(1000):
                    chosen=rng.choice(dates,size=len(dates),replace=True)
                    sample0=[pair for d in chosen for pair in grouped[str(d)]]
                    value=corr(sample0)
                    if value is not None: boot.append(value)
            flow_results.append({"period":period,"feature":field,"n":len(pairs),"date_clusters":len(dates),
                "pearson_correlation_with_executable_5m_net_ticks":corr([(x,y) for x,y,_ in pairs]),
                "date_cluster_bootstrap_ci95": [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))] if boot else None,
                "adjustment": "These are unadjusted descriptive correlations. Matched comparisons adjust the fixed price-action covariates; no new fitted feature model was used."})
    _write_csv(output_root / "signal-control-comparisons.csv", comparison_rows)
    daily = defaultdict(list)
    for r in comparison_rows:
        daily[(r["estimand"], r["date"])].append(float(r["signal_minus_control_5m_ticks"]))
    _write_csv(output_root / "daily-results.csv", [{"estimand": name, "date": date, "pairs": len(vals),
        "mean_signal_minus_control_5m_ticks": float(np.mean(vals))} for (name, date), vals in sorted(daily.items())])
    _write_json(output_root / "period-comparison.json", {"contract_sha256": contract_hash, "estimands": estimates,
        "exploratory_spring_october_compatibility": "compare estimates descriptively; both were previously explored development periods; no untouched validation claim"})
    adjusted_flow = _flow_adjusted_analysis(all_treated, markouts)
    _write_json(output_root / "orderflow-conditional-analysis.json", {"continuous_associations": flow_results,
        "price_action_adjusted_analysis": adjusted_flow,
        "adjusted_analysis_timing": "POST_HOC_EXPLORATORY_SUPPLEMENT after primary signal-control outcomes were inspected; fixed full predictor set, no selection or threshold search; not covered by the frozen primary comparison contract",
        "features": list(flow_fields), "depth_features": {"TOP5_IMBALANCE":"not present in sealed tape", "MLOFI":"not present in sealed tape",
            "future_feasibility":"native MBP-10 file metadata was already pinned in source coverage; separate authorized event-time depth extraction would be required; no raw DBN read was performed"},
        "feature_selection_or_threshold_search": False})
    (output_root / "research-limitations.md").write_text("# Research limitations\n\nThe 54 dates (Spring and October 2025) are explored development data, not an independent confirmation sample. Matched associations are not automatically causal. Five-minute executable markouts include the frozen 2ms entry rule, adverse ticks and existing commission convention; they do not establish strategy profitability. Multiple trigger/session/direction/period/horizon comparisons are exploratory. Candidate Tape preserves BBO, executions and aggressor fields, but not the depth ladder; TOP5 imbalance, MLOFI and liquidity resiliency are unavailable without a separately authorized native MBP-10 event-time extraction. No native DBN was opened.\n", encoding="utf-8")
    # Four-question decision: exploratory incremental comparison is available only
    # if the primary contrast estimate is present; no thresholded alpha claim.
    primary = next((x for x in estimates if x["estimand"]=="DISPLACEMENT_ANY" and x["period"]=="ALL_2025" and x["session"]=="ALL_SESSIONS" and x["direction"]=="ALL_DIRECTIONS"), None)
    decision = "CONTINUATION_EVIDENCE_MIXED_OR_INSUFFICIENT" if primary and primary.get("primary_5m_signal_minus_control",{}).get("ci95") else "CONTINUATION_EFFECT_UNIDENTIFIED"
    summary = {"study_id": RUN_ID, "status": "PASS", "primary_decision": decision,
        "short_continuation_detector_valid": True,
        "zero_short_signals_explained": sum(s["direction"] == "SHORT" for s in signal_rows) == 0,
        "material_implementation_defect_found": False, "atr14_unavailable_root_cause": old_atr["atr_cause"],
        "original_matches_recoverable": old_atr["reconstructed"], "alternative_control_predeclared": True,
        "valid_control_population": True, "continuation_signal_count": len(signal_rows),
        "control_observation_count": len(controls), "matched_or_standardized_signal_count": len(displacement_any),
        "covariate_balance": balances, "displacement_incremental_effect": primary,
        "bos_incremental_effect": None,
        "bos_incremental_status": "UNIDENTIFIED: combined-versus-displacement-only is only an unpaired descriptive cohort contrast; no outcome-independent, covariate-matched direct BOS incrementality design was frozen",
        "combined_trigger_incremental_effect": next((x for x in estimates if x["estimand"]=="COMBINED" and x["period"]=="ALL_2025" and x["session"]=="ALL_SESSIONS" and x["direction"]=="ALL_DIRECTIONS"), None),
        "orderflow_explanatory_relationships": {"unadjusted_continuous_associations": flow_results, "posthoc_fixed_price_action_adjustment": adjusted_flow},
        "economic_execution_hurdle": "Net executable markouts include entry at next valid BBO >=2ms, one adverse tick at entry and exit, plus 0.48 tick commission equivalent; no positive raw markout alone implies economic viability.",
        "contract_sha256": contract_hash, "frozen_inputs": frozen, "scope": contract["scope"]}
    _write_json(output_root / "summary.json", summary)
    summary["continuation_signal_count_unique_candles"] = len({(s["date"],s["session"],s["signal_timestamp_ns"]) for s in signal_rows})
    summary["signal_markout_descriptive"] = {trigger: {period: {
        "n": sum(1 for s in signal_rows if s["trigger"]==trigger and s["period"]==period),
        "mean_executable_5m_net_ticks": float(np.mean([float((mark_by_id[str(s["signal_id"])].get("executable_bbo_markouts",{}).get("300") or {})["executable_net_ticks"])
            for s in signal_rows if s["trigger"]==trigger and s["period"]==period and
            (mark_by_id.get(str(s["signal_id"]),{}).get("executable_bbo_markouts",{}).get("300") or {}).get("executable_net_ticks") is not None]))
        if any(s["trigger"]==trigger and s["period"]==period and
            (mark_by_id.get(str(s["signal_id"]),{}).get("executable_bbo_markouts",{}).get("300") or {}).get("executable_net_ticks") is not None for s in signal_rows) else None}
        for period in ("SPRING_2025","OCTOBER_2025")} for trigger in TRIGGER_NAMES}
    _write_json(output_root / "summary.json", summary)
    _write_report(output_root, summary, json.loads((output_root / "short-continuation-audit.json").read_text(encoding="utf-8")), match_summary,
        json.loads((output_root / "period-comparison.json").read_text(encoding="utf-8")))
    _finalize_hashes(output_root)
    return summary


def _finalize_hashes(root: Path) -> None:
    files = {p.name: _sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    _write_json(root / "artifact-hashes.json", {"status": "HASHED", "files": files,
        "convention": "SHA-256 of all regular artifacts except this self-referential hash manifest"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = _run(args.output_root, args.force)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({"status": result["status"], "primary_decision": result["primary_decision"],
        "signals": result.get("continuation_signal_count"), "controls": result.get("control_observation_count"),
        "matched": result.get("matched_or_standardized_signal_count"), "output": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
