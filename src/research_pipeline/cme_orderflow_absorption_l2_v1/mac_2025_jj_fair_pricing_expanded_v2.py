"""Exploratory JJ fair-pricing event catalog and fixed-exit study, 2025 only.

This is an explicit operationalization of public descriptions, not an exact
replication of a private strategy. It consumes the sealed, source-bound
Candidate Tape V2 generated from native ES MBP-10/trade data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import date, datetime, time as wall_time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_jj_fair_pricing_reversion_bos_v1 as v1
from .mac_2025_candidate_tape import TAPE_VERSION, load_tape
from .model import ES_COMMISSION, ES_POINT_VALUE

STUDY_ID = "ES_JJ_FAIR_PRICING_EXPANDED_RESEARCH_V2"
RUN_ID = "CMEOrderflow_ES_JJ_FAIR_PRICING_EXPANDED_V2"
OUT_ROOT = Path("research_runs") / RUN_ID
SOURCE_AUDIT = Path("research_runs/CMEOrderflow_ES_JJ_FAIR_PRICING_REVERSION_BOS_V1/source-coverage.json")
NY = ZoneInfo("America/New_York")
NS = 1_000_000_000
MINUTE_NS = 60 * NS
TICK = 0.25
AM = (9, 30, 11, 0)
PM = (14, 0, 15, 0)
CONTINUATION_MINUTES = 15
STOP_FAMILIES = ("EPISODE_STRUCTURAL", "LOCAL_SWING", "ATR_BASED")
FIXED_R_TARGETS = (1.0, 1.5, 2.0)
TRIGGERS = ("DISPLACEMENT_CANDLE", "BOS_ONLY", "BOS_PLUS_DISPLACEMENT")
MODELS = ("OPENING_CONTINUATION", "FAIR_PRICE_REVERSION")
PERIODS = {"SPRING_2025": tuple(v1.SPRING_DATES), "OCTOBER_2025": tuple(v1.OCTOBER_DATES)}
DATES = PERIODS["SPRING_2025"] + PERIODS["OCTOBER_2025"]

CONFIG: dict[str, Any] = {
    "study_id": STUDY_ID, "dataset": "GLBX.MDP3", "schema": "mbp-10", "instrument": "ES",
    "date_scope": {k: list(v) for k, v in PERIODS.items()}, "data_source": "existing sealed MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE; no rebuild or download",
    "sessions_et": {"NY_AM": {"start": "09:30", "end": "11:00"}, "NY_PM": {"start": "14:00", "end": "15:00"}},
    "anchor": "first actual ES trade in [session open, open+1 minute); dates without it are session-ineligible",
    "phase": "first 15 minutes continuation; remaining session reversion",
    "displacement_candle": "completed 1m trade candle; body > preceding candle body; bullish close > previous high or bearish close < previous low; previous candle opposite color; candle direction must equal intended trade; body/range and wick ratios recorded",
    "bos": "completed close strictly above max(prior two completed candle highs) for long or strictly below min(prior two lows) for short; requires immediately preceding calendar-minute bars",
    "bos_plus_displacement": "same completed candle meets both the above displacement and directional BOS definitions",
    "direction": "continuation points away from anchor according to completed close side; reversion points toward anchor and requires anchor on profitable side of proposed entry",
    "episodes": "trade-price excursion begins on first trade away from anchor after open/anchor touch; terminates on actual trade touch/cross of anchor; structural extreme uses only trades from current excursion start through signal",
    "repeated_signals": "one unique DATE|SESSION|MODEL|TRIGGER|DIRECTION|TIMESTAMP event; no daily cap; every qualifying candle is retained; sequential execution is separately simulated per stop/target cell, later signals while a position is open are recorded and suppressed",
    "execution": {"delay_ms": 2, "entry": "first valid canonical BBO at/after signal+2ms; ask for long, bid for short, then one adverse tick", "exit": "executable-side BBO plus one adverse tick", "same-observation_precedence": "stop before target", "session_end": "force close at last valid executable quote strictly before the session end", "commission_usd_per_side": float(ES_COMMISSION), "point_value_usd": float(ES_POINT_VALUE)},
    "stops": {"EPISODE_STRUCTURAL": "current anchor-excursion adverse trade extreme +/- one tick, using only prints through signal", "LOCAL_SWING": "adverse extreme among immediately preceding two completed candles +/- one tick", "ATR_BASED": "2 x simple mean of 14 completed 1m true ranges ending at the signal candle, adverse to entry"},
    "targets": {"FAIR_PRICE_ANCHOR": "reversion only; session anchor", "FIXED_R": list(FIXED_R_TARGETS), "fixed_r_basis": "entry-to-stop-line distance; fee and adverse exit tick remain in realized initial-risk denominator"},
    "features": {"local_delta": "aggressive buy-sell volume normalized by total aggressive volume, preceding 30s and 120s", "session_cvd": "aggressive signed volume from the session anchor through signal, normalized by total aggressive volume", "aggression_reversal": "two nonoverlapping completed 30s windows, opposing then supportive normalized flow", "delta_price_divergence": "continuous opposing-flow share versus 120s signed executable-mid progress; fixed feature-only groups", "price_impact": "directional executable-mid ticks per 100 aggressive contracts", "top5_imbalance": "UNAVAILABLE: the sealed candidate tape has BBO, not depth snapshots", "normalized_mlofi_persistence": "UNAVAILABLE: no compatible event-time feature stream is present in the sealed tape; not approximated"},
    "V1_reference_only": ["32-tick anchor displacement", "16-tick remaining anchor distance", "two-candle BOS", "first trade per direction/day"],
    "exit_grid": {"stops": list(STOP_FAMILIES), "targets": ["FAIR_PRICE_ANCHOR", *[f"FIXED_{x:g}R" for x in FIXED_R_TARGETS]]},
    "selection": "descriptive fixed grid only; no optimization or production selection",
}


class StudyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,default=_json_default).encode()).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default).encode() + b"\n")
    os.replace(tmp, path)


def _session_ns(day: str, session: str) -> tuple[int, int]:
    spec = AM if session == "NY_AM" else PM
    start = datetime.combine(date.fromisoformat(day), wall_time(spec[0], spec[1]), tzinfo=NY)
    end = datetime.combine(date.fromisoformat(day), wall_time(spec[2], spec[3]), tzinfo=NY)
    return v1._ns(start), v1._ns(end)


def _source_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    if not SOURCE_AUDIT.is_file():
        raise StudyError(f"missing V1 source-coverage provenance: {SOURCE_AUDIT}")
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("schema") != "mbp-10" or audit.get("instrument") != "ES":
        raise StudyError("V1 source coverage does not establish complete native ES MBP-10 tapes")
    tapes = audit.get("candidate_tapes", {})
    sources = audit.get("source_files", {})
    if set(tapes) != set(DATES) or set(sources) != set(DATES):
        raise StudyError("source audit date set is not exactly the 54 requested 2025 dates")
    for day in DATES:
        row = tapes[day]
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size == 0 or _sha(path) != row["sha256"]:
            raise StudyError(f"candidate tape missing/hash mismatch for {day}")
        src = sources[day]
        source_path = Path(src["path"])
        if (src.get("schema") != "mbp-10" or src.get("dataset") != "GLBX.MDP3"
                or not source_path.is_file() or source_path.stat().st_size != int(src.get("bytes", -1))
                or not tapes[day].get("bbo_path_complete")):
            raise StudyError(f"native source identity/path failure for {day}")
    return audit, {"path": str(SOURCE_AUDIT), "sha256": _sha(SOURCE_AUDIT)}


def _load_events(day: str, coverage: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any], Path]:
    row = coverage["candidate_tapes"][day]
    path = Path(row["path"])
    tape = load_tape(path, source_sha256=row["source_sha256"], semantic_sha256=row["semantic_sha256"])
    if tape.metadata.get("date") != day or tape.metadata.get("tape_version") != TAPE_VERSION:
        raise StudyError(f"tape date/version mismatch for {day}")
    ev = tape.events
    if len(ev) and np.any(np.diff(ev["timestamp_ns"]) < 0):
        raise StudyError(f"unordered tape for {day}")
    return ev, tape.metadata, path


def _bars_for_session(ev: np.ndarray, start: int, end: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts = ev["timestamp_ns"].astype(np.int64, copy=False)
    is_trade = (ev["execution_size"] > 0) & np.isfinite(ev["execution_price"])
    ti = np.flatnonzero(is_trade & (ts >= start) & (ts < end))
    tts = ts[ti]
    px = ev["execution_price"][ti].astype(float, copy=False)
    sz = ev["execution_size"][ti].astype(float, copy=False)
    bars = v1.build_one_minute_bars(tts, px, sz, start, end)
    return bars, tts, px, sz, ti


def _anchor(tts: np.ndarray, px: np.ndarray, start: int) -> dict[str, Any]:
    lo = int(np.searchsorted(tts, start, side="left")); hi = int(np.searchsorted(tts, start + MINUTE_NS, side="left"))
    for i in range(lo, hi):
        if math.isfinite(float(px[i])) and px[i] > 0:
            return {"status": "VALID", "price": float(px[i]), "timestamp_ns": int(tts[i])}
    return {"status": "NO_OPENING_TRADE", "price": None, "timestamp_ns": None}


def _model_direction(anchor_price: float, close: float, minute: int) -> tuple[str, int] | None:
    """Return a phase-valid direction; reversion must point from entry toward anchor."""
    side_from_anchor = 1 if close > anchor_price else -1 if close < anchor_price else 0
    if side_from_anchor == 0:
        return None
    if minute < CONTINUATION_MINUTES:
        model, direction = "OPENING_CONTINUATION", side_from_anchor
        if direction * (close - anchor_price) <= 0:
            return None
    else:
        model, direction = "FAIR_PRICE_REVERSION", -side_from_anchor
        # The anchor is the reversion target, so it must lie ahead in trade direction.
        if direction * (anchor_price - close) <= 0:
            return None
    return model, direction


def displacement_candle(bar: Mapping[str, Any], previous: Mapping[str, Any] | None, direction: int) -> dict[str, Any]:
    if previous is None:
        return {"valid": False, "body_range_ratio": None, "upper_wick_ratio": None, "lower_wick_ratio": None}
    body = abs(float(bar["close"]) - float(bar["open"]))
    prev_body = abs(float(previous["close"]) - float(previous["open"]))
    span = float(bar["high"]) - float(bar["low"])
    prev_opposite = (float(previous["close"]) < float(previous["open"]) if direction > 0 else float(previous["close"]) > float(previous["open"]))
    candle_direction = 1 if bar["close"] > bar["open"] else -1 if bar["close"] < bar["open"] else 0
    beyond = float(bar["close"]) > float(previous["high"]) if direction > 0 else float(bar["close"]) < float(previous["low"])
    upper = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
    lower = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
    return {"valid": bool(span > 0 and body > prev_body and beyond and prev_opposite and candle_direction == direction),
            "body_range_ratio": body / span if span > 0 else None,
            "upper_wick_ratio": upper / span if span > 0 else None,
            "lower_wick_ratio": lower / span if span > 0 else None,
            "body_ticks": body / TICK, "previous_body_ticks": prev_body / TICK,
            "previous_opposite_color": prev_opposite, "close_beyond_previous_wick": beyond}


def _episode_map(tts: np.ndarray, prices: np.ndarray, anchor: float) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, float, float]]]:
    """Assign trade rows to anchor-to-anchor excursions and their causal extremes."""
    ids = np.full(len(tts), -1, dtype=np.int32); starts = np.zeros(len(tts), dtype=np.int32)
    episodes: dict[int, tuple[int, int, float, float]] = {}
    active: int | None = None; side = 0; next_id = 0; lo = hi = 0; mn = mx = float(anchor)
    for i, p0 in enumerate(prices):
        p = float(p0)
        if active is not None and ((side > 0 and p <= anchor) or (side < 0 and p >= anchor)):
            episodes[active] = (lo, i, mn, mx)
            active = None; side = 0
        if active is None and p != anchor:
            next_id += 1; active = next_id; side = 1 if p > anchor else -1; lo = i; mn = mx = p
        if active is not None:
            mn = min(mn, p); mx = max(mx, p); ids[i] = active; starts[i] = lo
    if active is not None:
        episodes[active] = (lo, len(tts), mn, mx)
    return ids, starts, episodes


def _feature_context(ev: np.ndarray, start: int, end: int) -> dict[str, Any]:
    ts_all=ev["timestamp_ns"].astype(np.int64,copy=False)
    lo=int(np.searchsorted(ts_all,start,side="left")); hi=int(np.searchsorted(ts_all,end,side="left"))
    ts=ts_all[lo:hi]
    size=ev["execution_size"][lo:hi].astype(np.float64,copy=False)
    aggressor=ev["aggressor"][lo:hi]
    buy=np.where((size>0)&(aggressor>0),size,0.0); sell=np.where((size>0)&(aggressor<0),size,0.0)
    return {"ts":ts,"bid":ev["bid"][lo:hi],"ask":ev["ask"][lo:hi],"buy_prefix":np.r_[0.0,np.cumsum(buy)],
            "sell_prefix":np.r_[0.0,np.cumsum(sell)],"start":start}


def _mid_asof(context: Mapping[str, Any], timestamp: int) -> float | None:
    ts=context["ts"]
    if not len(ts): return None
    i=int(np.searchsorted(ts,timestamp,side="right"))-1
    if i<0: return None
    bid, ask = float(context["bid"][i]), float(context["ask"][i])
    return (bid + ask) / 2 if v1._valid_quote(bid, ask) else None


def causal_features(ev: np.ndarray, signal_ns: int, session_start: int, session_end: int, direction: int,
                    context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    context=context or _feature_context(ev,session_start,session_end)
    ts=context["ts"]
    hi=int(np.searchsorted(ts,signal_ns,side="right"))
    def flow(a: int, b: int) -> tuple[float, float, float]:
        a=max(session_start,a); left=int(np.searchsorted(ts,a,side="left")); right=min(hi,int(np.searchsorted(ts,b,side="right")))
        if right<left: right=left
        bv=float(context["buy_prefix"][right]-context["buy_prefix"][left]); sv=float(context["sell_prefix"][right]-context["sell_prefix"][left])
        total = bv + sv
        return bv - sv, total, (bv - sv) / total if total else float("nan")
    d30, v30, n30 = flow(signal_ns - 30 * NS, signal_ns)
    d120, v120, n120 = flow(signal_ns - 120 * NS, signal_ns)
    cvd, cvdv, cvdn = flow(session_start, signal_ns)
    p0 = _mid_asof(context, signal_ns - 120 * NS); p1 = _mid_asof(context, signal_ns)
    progress = direction * (p1 - p0) / TICK if p0 is not None and p1 is not None else None
    old_delta, old_vol, old_norm = flow(signal_ns - 60 * NS, signal_ns - 30 * NS)
    new_delta, new_vol, new_norm = flow(signal_ns - 30 * NS, signal_ns)
    old_support = direction * old_norm if math.isfinite(old_norm) else None
    new_support = direction * new_norm if math.isfinite(new_norm) else None
    support30 = direction * n30 if math.isfinite(n30) else None
    support120 = direction * n120 if math.isfinite(n120) else None
    cvd_support = direction * cvdn if math.isfinite(cvdn) else None
    opposition = max(0.0, -support120) if support120 is not None else None
    impact = (progress * 100.0 / v120) if progress is not None and v120 > 0 else None
    # Midprice/depth are not synthesized from trades. Depth-specific fields are explicit nulls.
    return {"delta_30s_contracts": d30, "delta_30s_normalized": n30 if math.isfinite(n30) else None,
            "aggressive_volume_30s": v30, "delta_2m_contracts": d120, "delta_2m_normalized": n120 if math.isfinite(n120) else None,
            "aggressive_volume_2m": v120, "session_cvd_contracts": cvd, "session_cvd_normalized": cvdn if math.isfinite(cvdn) else None,
            "session_cvd_supportive_fraction": cvd_support, "local_delta_30s_supportive_fraction": support30,
            "local_delta_2m_supportive_fraction": support120,
            "aggression_reversal": bool(old_support is not None and new_support is not None and old_support < 0 < new_support),
            "prior_30s_opposed_fraction": max(0.0, -old_support) if old_support is not None else None,
            "recent_30s_support_fraction": new_support,
            "mid_2m_start": p0, "mid_signal": p1, "price_progress_toward_ticks_2m": progress,
            "opposing_aggression_fraction_2m": opposition,
            "effort_without_result": (opposition / (1.0 + max(0.0, progress))) if opposition is not None and progress is not None else None,
            "price_impact_ticks_per_100_aggressive_contracts": impact,
            "top5_depth_imbalance": None, "normalized_mlofi_persistence": None,
            "depth_features_unavailable_reason": "sealed candidate tape contains no depth ladder; no raw-book reconstruction or proxy used"}


def _signal_catalog(day: str, session: str, ev: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start, end = _session_ns(day, session)
    bars, tts, px, sizes, event_trade_ix = _bars_for_session(ev, start, end)
    anchor = _anchor(tts, px, start)
    if anchor["status"] != "VALID":
        return [], {"eligible": False, "reason": anchor["status"], "anchor": anchor, "bars": len(bars)}
    ep_ids, ep_starts, ep_info = _episode_map(tts, px, anchor["price"])
    bm = {int(b["minute_index"]): b for b in bars}
    sigs: list[dict[str, Any]] = []; seen: set[tuple[Any, ...]] = set()
    for b in bars:
        minute = int(b["minute_index"]); end_ns = int(b["end_ns"])
        if minute < 1 or end_ns > end:
            continue
        close = float(b["close"])
        phase = _model_direction(float(anchor["price"]), close, minute)
        if phase is None:
            continue
        model, direction = phase
        side_from_anchor = 1 if close > float(anchor["price"]) else -1
        prior = bm.get(minute - 1)
        disp = displacement_candle(b, prior, direction)
        bos = v1.bos_for_bar(bm, minute, direction)
        flags = {"DISPLACEMENT_CANDLE": disp["valid"], "BOS_ONLY": bool(bos["confirmed"]),
                 "BOS_PLUS_DISPLACEMENT": bool(disp["valid"] and bos["confirmed"])}
        trade_ix = int(np.searchsorted(tts, end_ns, side="left")) - 1
        if trade_ix < 0:
            continue
        epi = int(ep_ids[trade_ix]); ep_start = int(ep_starts[trade_ix])
        info = ep_info.get(epi)
        if info is None or epi == 0:
            continue
        # A signal's event-side excursion must agree with its intended direction's anchor side.
        exc_side = 1 if px[trade_ix] > anchor["price"] else -1
        if exc_side != side_from_anchor:
            continue
        ep_low = float(np.min(px[ep_start:trade_ix+1])); ep_high = float(np.max(px[ep_start:trade_ix+1]))
        episode_displacement_ticks = max(abs(ep_high-float(anchor["price"])),abs(ep_low-float(anchor["price"]))) / TICK
        distance = abs(close - float(anchor["price"])) / TICK
        for trigger in TRIGGERS:
            if not flags[trigger]:
                continue
            key = (day, session, model, trigger, direction, end_ns)
            if key in seen:
                continue
            seen.add(key)
            sigs.append({"signal_id": "|".join(map(str, key)), "date": day, "period": v1.period_for(day),
                         "session": session, "model": model, "trigger": trigger,
                         "direction": "LONG" if direction > 0 else "SHORT", "direction_sign": direction,
                         "signal_timestamp_ns": end_ns, "signal_close": close,
                         "anchor_price": float(anchor["price"]), "anchor_timestamp_ns": anchor["timestamp_ns"],
                         "minute_index": minute, "episode_id": epi, "episode_start_trade_index": ep_start,
                         "episode_adverse_extreme": ep_low if direction > 0 else ep_high,
                         "displacement_body_range_ratio": disp.get("body_range_ratio"),
                         "displacement_upper_wick_ratio": disp.get("upper_wick_ratio"),
                         "displacement_lower_wick_ratio": disp.get("lower_wick_ratio"),
                         "bos_reference_level": bos.get("reference_level"),
                         "bos_prior_minute_indices": bos.get("prior_minute_indices"),
                         "v1_reference_applicable": session == "NY_AM",
                         "v1_32_tick_displacement": bool(session == "NY_AM" and model == "FAIR_PRICE_REVERSION" and episode_displacement_ticks >= 32),
                         "v1_16_tick_remaining_distance": bool(session == "NY_AM" and model == "FAIR_PRICE_REVERSION" and distance >= 16),
                         "v1_two_candle_bos": bool(session == "NY_AM" and model == "FAIR_PRICE_REVERSION" and bos["confirmed"]),
                         "v1_episode_max_displacement_ticks": episode_displacement_ticks,
                         "v1_distance_ticks": distance,
                         "previous_candle": dict(prior) if prior else None,
                         "prior_two_candles": [dict(bm[minute - 2]), dict(bm[minute - 1])] if minute - 2 in bm and minute - 1 in bm else None})
    sigs.sort(key=lambda x: (x["signal_timestamp_ns"], x["model"], x["trigger"], x["direction"]))
    first: dict[int, int] = {}
    for row in sigs:
        sign = int(row["direction_sign"])
        qualifies_v1 = (row["model"] == "FAIR_PRICE_REVERSION" and row["v1_32_tick_displacement"]
                        and row["v1_16_tick_remaining_distance"] and row["v1_two_candle_bos"])
        first.setdefault(sign, int(row["signal_timestamp_ns"])) if qualifies_v1 else None
        row["v1_first_trade_per_direction_eligible"] = bool(qualifies_v1 and first.get(sign) == int(row["signal_timestamp_ns"]))
    # Causal order-flow diagnostics use only the sealed tape records at or before the completed candle.
    feature_context=_feature_context(ev,start,end)
    for row in sigs:
        row["features"] = causal_features(ev, int(row["signal_timestamp_ns"]), start, end, int(row["direction_sign"]),feature_context)
    return sigs, {"eligible": True, "reason": "PASS", "anchor": anchor, "bars": len(bars),
                  "trade_count": len(tts), "episode_count": len(ep_info), "start_ns": start, "end_ns": end}


def _atr(bars: Sequence[Mapping[str, Any]], minute: int, period: int = 14) -> float | None:
    need = [minute - j for j in range(period)]
    bm = {int(b["minute_index"]): b for b in bars}
    if any(i not in bm or i - 1 not in bm for i in need):
        return None
    vals = []
    for i in reversed(need):
        b, prev = bm[i], bm[i - 1]
        vals.append(max(float(b["high"])-float(b["low"]), abs(float(b["high"])-float(prev["close"])), abs(float(b["low"])-float(prev["close"]))))
    return float(statistics.mean(vals))


def _target_cells(model: str) -> tuple[str, ...]:
    fixed = tuple(f"FIXED_{x:g}R" for x in FIXED_R_TARGETS)
    return ("FAIR_PRICE_ANCHOR", *fixed) if model == "FAIR_PRICE_REVERSION" else fixed


def execute_sequential_cell(ev: np.ndarray, signals: Sequence[Mapping[str, Any]], bars: Sequence[Mapping[str, Any]],
                            start_ns: int, end_ns: int, stop_family: str, target_name: str) -> dict[str, Any]:
    """Execute one fixed strategy configuration chronologically, never overlapping positions."""
    blocked_until = -1
    trades: list[dict[str, Any]] = []
    statuses: dict[str, int] = defaultdict(int)
    overlap = 0
    for signal in signals:
        if int(signal["signal_timestamp_ns"]) <= blocked_until:
            overlap += 1
            continue
        result = _execution_cell(ev, signal, bars, start_ns, end_ns, stop_family, target_name)
        statuses[result["status"]] += 1
        if result["status"] == "EXECUTED":
            blocked_until = int(result["exit_timestamp_ns"])
            trades.append(result)
    return {"trades": trades, "statuses": dict(statuses), "overlap_suppressed_signals": overlap,
            "raw_signals": len(signals)}


def _execution_cell(ev: np.ndarray, signal: Mapping[str, Any], bars: Sequence[Mapping[str, Any]],
                    start_ns: int, end_ns: int, stop_family: str, target_name: str) -> dict[str, Any]:
    direction = int(signal["direction_sign"]); entry_ready = int(signal["signal_timestamp_ns"]) + 2_000_000
    ts = ev["timestamp_ns"].astype(np.int64, copy=False); bid = ev["bid"].astype(float, copy=False); ask = ev["ask"].astype(float, copy=False)
    ix = int(np.searchsorted(ts, entry_ready, side="left")); entry_ix = None
    while ix < len(ts) and ts[ix] < end_ns:
        if v1._valid_quote(float(bid[ix]), float(ask[ix])):
            entry_ix = ix; break
        ix += 1
    if entry_ix is None:
        return {"status": "NO_VALID_ENTRY_QUOTE"}
    entry_quote = float(ask[entry_ix] if direction > 0 else bid[entry_ix]); entry = entry_quote + direction*TICK
    minute = int(signal["minute_index"]); bm = {int(b["minute_index"]): b for b in bars}
    if stop_family == "EPISODE_STRUCTURAL":
        stop = float(signal["episode_adverse_extreme"]) - direction*TICK
    elif stop_family == "LOCAL_SWING":
        if minute-1 not in bm or minute-2 not in bm:
            return {"status": "INVALID_STOP_MISSING_TWO_PRIOR_CANDLES"}
        extreme = min(float(bm[minute-1]["low"]), float(bm[minute-2]["low"])) if direction > 0 else max(float(bm[minute-1]["high"]), float(bm[minute-2]["high"]))
        stop = extreme - direction*TICK
    else:
        atr = _atr(bars, minute)
        if atr is None or atr <= 0:
            return {"status": "INVALID_STOP_ATR_UNAVAILABLE"}
        stop = entry - direction*2.0*atr
    stop_ticks = direction*(entry-stop)/TICK
    if not math.isfinite(stop_ticks) or stop_ticks <= 0:
        return {"status": "INVALID_STOP_WRONG_SIDE", "entry_price": entry, "stop_price": stop}
    if target_name == "FAIR_PRICE_ANCHOR":
        target = float(signal["anchor_price"])
    else:
        mult = float(target_name.removeprefix("FIXED_").removesuffix("R"))
        target = entry + direction*(entry-stop)*mult
    target_ahead = direction*(target-entry) > 0
    if not target_ahead:
        return {"status": "INVALID_TARGET_NOT_PROFITABLE_SIDE", "entry_price": entry, "stop_price": stop, "target_price": target}
    # Reversion anchor target must remain beyond the entry in the intended direction.
    if signal["model"] == "FAIR_PRICE_REVERSION" and target_name == "FAIR_PRICE_ANCHOR" and direction*(target-entry) <= 0:
        return {"status": "INVALID_ANCHOR_TARGET_GEOMETRY", "entry_price": entry, "stop_price": stop, "target_price": target}
    outcome = None; exit_ix = None
    # First valid executable-side barrier. Session-end is an explicit forced flat using last valid quote before end.
    for j in range(entry_ix+1, len(ts)):
        if int(ts[j]) >= end_ns: break
        if not v1._valid_quote(float(bid[j]), float(ask[j])): continue
        ref = float(bid[j] if direction > 0 else ask[j])
        stop_hit = direction*(ref-stop) <= 0
        target_hit = (float(bid[j]) >= target) if direction > 0 else (float(ask[j]) <= target)
        outcome = v1.resolve_exit_trigger(stop_hit, target_hit)
        if outcome:
            exit_ix = j; break
    if exit_ix is None:
        candidates = np.flatnonzero((ts < end_ns) & (np.arange(len(ts)) > entry_ix) & np.isfinite(bid) & np.isfinite(ask) & (ask > bid))
        if not len(candidates): return {"status": "CENSORED_NO_SESSION_END_QUOTE"}
        exit_ix = int(candidates[-1]); outcome = "SESSION_END"
    exit_ref = float(bid[exit_ix] if direction > 0 else ask[exit_ix]); exit_fill = exit_ref-direction*TICK
    fee = 2*float(ES_COMMISSION); risk_usd = (stop_ticks*TICK + TICK)*float(ES_POINT_VALUE)+fee
    gross = direction*(exit_fill-entry)*float(ES_POINT_VALUE); net = gross-fee
    path = np.arange(entry_ix, exit_ix+1)
    valid = np.isfinite(bid[path]) & np.isfinite(ask[path]) & (ask[path] > bid[path])
    path = path[valid]
    refs = np.where(direction>0,bid[path],ask[path])-direction*TICK
    signed_ticks = direction*(refs-entry)/TICK
    return {"status": "EXECUTED", "outcome": outcome, "date": signal["date"], "period": signal["period"],
            "session": signal["session"], "model": signal["model"], "trigger": signal["trigger"],
            "signal_id": signal["signal_id"], "signal_timestamp_ns": signal["signal_timestamp_ns"],
            "direction": signal["direction"], "stop_family": stop_family, "target_model": target_name,
            "entry_timestamp_ns": int(ts[entry_ix]), "entry_quote": entry_quote, "entry_price": entry,
            "stop_price": stop, "target_price": target, "stop_distance_ticks": stop_ticks,
            "target_distance_ticks": direction*(target-entry)/TICK,
            "initial_risk_usd": risk_usd, "fees_usd": fee, "exit_timestamp_ns": int(ts[exit_ix]),
            "exit_quote": exit_ref, "exit_price": exit_fill, "gross_pnl_usd": gross, "net_pnl_usd": net,
            "gross_r": gross/risk_usd, "net_r": net/risk_usd,
            "mfe_ticks": float(max(0.0, np.max(signed_ticks))), "mae_ticks": float(max(0.0, -np.min(signed_ticks))),
            "mfe_r": float(max(0.0, np.max(signed_ticks)))*TICK*float(ES_POINT_VALUE)/risk_usd,
            "mae_r": -float(max(0.0, -np.min(signed_ticks)))*TICK*float(ES_POINT_VALUE)/risk_usd,
            "session_end_forced": outcome == "SESSION_END"}


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = [r for r in rows if float(r["net_pnl_usd"]) > 0]; losses = [r for r in rows if float(r["net_pnl_usd"]) < 0]
    net = [float(r["net_r"]) for r in rows]; dailies: dict[str,float] = defaultdict(float)
    for r in rows: dailies[str(r["date"])] += float(r["net_r"])
    targetfirst = sum(r["outcome"] not in ("STOP","SESSION_END") for r in rows)
    dd=0.0; peak=0.0; running=0.0
    for r in sorted(rows,key=lambda z:(z["date"],z["exit_timestamp_ns"])):
        running += float(r["net_r"]); peak=max(peak,running); dd=max(dd,peak-running)
    return {"trade_count":len(rows),"wins":len(wins),"losses":len(losses),"win_rate":len(wins)/len(rows) if rows else None,
            "average_net_r":float(np.mean(net)) if net else None,"median_net_r":float(np.median(net)) if net else None,
            "total_net_r":float(sum(net)),"profit_factor_net_usd":sum(float(r["net_pnl_usd"]) for r in wins)/abs(sum(float(r["net_pnl_usd"]) for r in losses)) if losses and sum(float(r["net_pnl_usd"]) for r in losses) else None,
            "max_drawdown_r":dd,"target_before_stop_rate":targetfirst/len(rows) if rows else None,
            "active_days":len(dailies),"mean_daily_net_r":float(np.mean(list(dailies.values()))) if dailies else None,
            "median_initial_risk_ticks":float(np.median([r["stop_distance_ticks"] for r in rows])) if rows else None,
            "median_mfe_ticks":float(np.median([r["mfe_ticks"] for r in rows])) if rows else None,
            "median_mae_ticks":float(np.median([r["mae_ticks"] for r in rows])) if rows else None}


def _combinations():
    for model in MODELS:
        for trigger in TRIGGERS:
            yield model, trigger


def run(*, output_root: Path = OUT_ROOT, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    coverage, coverage_ref = _source_contract()
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise StudyError(f"output already exists; pass --force only to replace this audit output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_public_comparison(output_root)
    _write_json(output_root/"study-config.json", {**CONFIG, "tape_version": TAPE_VERSION, "source_coverage_sha256": coverage_ref["sha256"]})
    source_rows=[]
    for day in DATES:
        tr=coverage["source_files"][day]; tape_row=coverage["candidate_tapes"][day]
        source_rows.append({"date":day,"period":v1.period_for(day),"dataset":tr["dataset"],"schema":tr["schema"],"symbol":tr["symbol"],
                            "native_source_path":tr["path"],"native_source_bytes":tr["bytes"],"native_source_sha256":tr["sha256"],
                            "candidate_tape_path":tape_row["path"],"candidate_tape_sha256":tape_row["sha256"],
                            "candidate_tape_source_sha256":tape_row["source_sha256"],"candidate_tape_semantic_sha256":tape_row["semantic_sha256"],
                            "bbo_path_complete":tape_row["bbo_path_complete"]})
    source_coverage={"status":"PASS","dataset":"GLBX.MDP3","schema":"mbp-10","instrument":"ES","target_dates":list(DATES),
                     "dates_processed":0,"period_roles":{"SPRING_2025":"development only","OCTOBER_2025":"development compatibility only"},
                     "source_coverage_manifest":coverage_ref,"files":source_rows,"no_2026":True,"no_oos":True,"data_downloaded":False}
    _write_json(output_root/"source-coverage.json",source_coverage)
    if smoke: days = DATES[:1]
    else: days = DATES
    all_signals=[]; all_trades=[]; daily_rows=[]; session_audits={}; cell_by_key=defaultdict(list); cell_diag=defaultdict(lambda:{"raw_signals":0,"overlap_suppressed_signals":0,"statuses":defaultdict(int)})
    for di, day in enumerate(days,1):
        print(f"JJ_FAIR_PRICE_V2_DATE={di}/{len(days)} {day}",flush=True)
        ev, meta, tape_path = _load_events(day,coverage)
        for session in ("NY_AM","NY_PM"):
            start,end=_session_ns(day,session); bars,_,_,_,_=_bars_for_session(ev,start,end)
            sigs, audit = _signal_catalog(day,session,ev); audit.update({"session":session,"date":day,"period":v1.period_for(day),"tape_path":str(tape_path),"tape_events":len(ev)})
            session_audits[f"{day}|{session}"]=audit
            if not audit["eligible"]:
                for model,trigger in _combinations():
                    daily_rows.append({"date":day,"period":v1.period_for(day),"session":session,"model":model,"trigger":trigger,"eligible":False,"raw_signals":0,"sequential_trades":0,"exit_cells_json":"{}"})
                continue
            group_signals={(m,t):[s for s in sigs if s["model"]==m and s["trigger"]==t] for m,t in _combinations()}
            for sig in sigs:
                all_signals.append(sig)
            for model,trigger in _combinations():
                raw=group_signals[(model,trigger)]; cell_stats={}; trade_before=len(all_trades)
                for stop_family in STOP_FAMILIES:
                    for target_name in _target_cells(model):
                        key=(day,session,model,trigger,stop_family,target_name)
                        selected=[s for s in raw if not (s["model"]=="OPENING_CONTINUATION" and target_name=="FAIR_PRICE_ANCHOR")]
                        run_cell=execute_sequential_cell(ev,selected,bars,start,end,stop_family,target_name)
                        cell_trades=run_cell["trades"]; statuses=run_cell["statuses"]; overlap=run_cell["overlap_suppressed_signals"]
                        all_trades.extend(cell_trades)
                        cell_stats[f"{stop_family}|{target_name}"]={**_metrics(cell_trades),"overlap_suppressed_signals":overlap,"nonexecution_statuses":statuses,"raw_signals":len(selected)}
                        cell_by_key[key].extend(cell_trades)
                        for period_key in ("ALL",v1.period_for(day)):
                            dkey=(period_key,session,model,trigger,stop_family,target_name)
                            cell_diag[dkey]["raw_signals"]+=len(selected)
                            cell_diag[dkey]["overlap_suppressed_signals"]+=overlap
                            for status,count in statuses.items(): cell_diag[dkey]["statuses"][status]+=count
                # Signal-level average outcome over fixed valid cells, for descriptive feature association only.
                group_trades=all_trades[trade_before:]
                daily_rows.append({"date":day,"period":v1.period_for(day),"session":session,"model":model,"trigger":trigger,
                    "eligible":True,"raw_signals":len(raw),"sequential_trades":sum(x["model"]==model and x["trigger"]==trigger for x in group_trades),
                    "gross_r_sum":sum((x["gross_pnl_usd"] / x["initial_risk_usd"]) for x in group_trades),"net_r_sum":sum(x["net_r"] for x in group_trades),
                    "winners":sum(x["net_pnl_usd"]>0 for x in group_trades),"losers":sum(x["net_pnl_usd"]<0 for x in group_trades),
                    "median_risk_ticks":float(np.median([x["stop_distance_ticks"] for x in group_trades])) if group_trades else None,
                    "median_mfe_ticks":float(np.median([x["mfe_ticks"] for x in group_trades])) if group_trades else None,
                    "median_mae_ticks":float(np.median([x["mae_ticks"] for x in group_trades])) if group_trades else None,
                    "exit_cells_json":json.dumps(cell_stats,sort_keys=True,separators=(",",":"))})
        source_coverage["dates_processed"]+=1
        # Drop the day's event array before loading the next date.
        del ev
        _write_json(output_root/"source-coverage.json",source_coverage)
    if smoke:
        return {"status":"SMOKE_PASS","dates_processed":len(days),"signals":len(all_signals),"trades":len(all_trades)}
    if len(days)!=54 or len(session_audits)!=108:
        raise StudyError(f"incomplete date/session set: dates={len(days)} sessions={len(session_audits)}")
    # V1 comparator flag is one first candidate per reversion direction/day, assigned after deterministic ordering.
    _write_jsonl_gz(output_root/"all-signals.jsonl.gz",all_signals)
    _write_jsonl_gz(output_root/"causal-features.jsonl.gz",({"signal_id":x["signal_id"],"date":x["date"],"session":x["session"],"model":x["model"],"trigger":x["trigger"],"direction":x["direction"],"timestamp_ns":x["signal_timestamp_ns"],**x["features"]} for x in all_signals))
    _write_jsonl_gz(output_root/"executed-trades.jsonl.gz",all_trades)
    # Exit surface by period and full sample, with each strategy configuration kept separate.
    surface=[]
    groups=defaultdict(list)
    for key,rows in cell_by_key.items():
        day,session,model,trigger,stop,target=key
        groups[("ALL",session,model,trigger,stop,target)].extend(rows)
        groups[(v1.period_for(day),session,model,trigger,stop,target)].extend(rows)
    for k,rows in sorted(groups.items()):
        period,session,model,trigger,stop,target=k
        diag=cell_diag[(period,session,model,trigger,stop,target)]
        surface.append({"period":period,"session":session,"model":model,"trigger":trigger,"stop_family":stop,"target_model":target,
                        "raw_signals":diag["raw_signals"],"overlap_suppressed_signals":diag["overlap_suppressed_signals"],
                        "nonexecution_status_counts":json.dumps(dict(diag["statuses"]),sort_keys=True),**_metrics(rows)})
    if surface:
        with (output_root/"exit-model-surface.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(surface[0]));w.writeheader();w.writerows(surface)
    with (output_root/"daily-results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(daily_rows[0]) if daily_rows else ["date","period","session","model","trigger","eligible","raw_signals","sequential_trades","exit_cells_json"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(daily_rows)
    combo_summary=[]
    for session in ("NY_AM","NY_PM"):
      for model,trigger in _combinations():
        sr=[x for x in all_signals if x["session"]==session and x["model"]==model and x["trigger"]==trigger]
        cells={}
        for stop in STOP_FAMILIES:
          for target in _target_cells(model):
            rows=[x for x in all_trades if x["session"]==session and x["model"]==model and x["trigger"]==trigger and x["stop_family"]==stop and x["target_model"]==target]
            diag=cell_diag[("ALL",session,model,trigger,stop,target)]
            cells[f"{stop}|{target}"]={**_metrics(rows),"raw_signals":diag["raw_signals"],
              "overlap_suppressed_signals":diag["overlap_suppressed_signals"],"nonexecution_status_counts":dict(diag["statuses"])}
        combo_summary.append({"session":session,"model":model,"trigger":trigger,"raw_qualified_signals":len(sr),"signal_dates":len({x["date"] for x in sr}),"exit_cells":cells})
    _write_json(output_root/"session-model-trigger-results.json",{"combinations":combo_summary,"session_eligibility":session_audits})
    # Feature relationships use event-average net R across the fixed grid; no cell/threshold selection.
    trade_per_signal=defaultdict(list)
    for tr in all_trades: trade_per_signal[tr["signal_id"]].append(float(tr["net_r"]))
    feat_rows=[]
    for s in all_signals:
        vals=trade_per_signal.get(s["signal_id"],[])
        feat_rows.append({**s["features"],"date":s["date"],"period":s["period"],"session":s["session"],"model":s["model"],"trigger":s["trigger"],"signal_id":s["signal_id"],"mean_net_r_across_executed_fixed_cells":float(np.mean(vals)) if vals else None})
    feature_analysis=_feature_analysis(feat_rows)
    _write_json(output_root/"orderflow-feature-analysis.json",feature_analysis)
    for period,name in (("SPRING_2025","spring-results.json"),("OCTOBER_2025","october-results.json")):
        _write_json(output_root/name,{"period":period,"metrics":[x for x in surface if x["period"]==period],"dates":list(PERIODS[period])})
    total_sig_by_day=defaultdict(int); total_tr_by_day=defaultdict(int)
    for s in all_signals: total_sig_by_day[s["date"]]+=1
    for t in all_trades: total_tr_by_day[t["date"]]+=1
    summary={"status":"PARTIAL" if feature_analysis["unavailable"] else "PASS","study_id":STUDY_ID,"dates_processed":len(days),"valid_ny_am_sessions":sum(x.get("eligible",False) for x in session_audits.values() if x["session"]=="NY_AM"),
      "valid_ny_pm_sessions":sum(x.get("eligible",False) for x in session_audits.values() if x["session"]=="NY_PM"),"total_raw_signals":len(all_signals),
      "total_sequential_trades_across_independent_fixed_exit_configurations":len(all_trades),"unique_signals_executed_at_least_once":len(trade_per_signal),
      "signals_per_day_distribution":_dist(list(total_sig_by_day.values()),expected=len(DATES)),"trades_per_day_distribution":_dist(list(total_tr_by_day.values()),expected=len(DATES)),
      "model_trigger_counts":combo_summary,"feature_availability":{"delta_cvd_aggression_reversal_divergence_price_impact":"COMPUTED_CAUSALLY_FROM_VALIDATED_TAPE_EXECUTIONS_AND_BBO","top5_imbalance":"UNAVAILABLE_NO_DEPTH_IN_SEALED_TAPE","normalized_mlofi_persistence":"UNAVAILABLE_NO_EVENT_TIME_FEATURE_CACHE_FOR_ALL_PERIODS"},
      "multiple_comparison_count":{"session_model_trigger_combinations":len(MODELS)*len(TRIGGERS)*2,
        "unique_fixed_exit_configurations":2*sum(len(_target_cells(m))*len(STOP_FAMILIES)*len(TRIGGERS) for m in MODELS),
        "period_specific_exit_cell_summaries":2*2*sum(len(_target_cells(m))*len(STOP_FAMILIES)*len(TRIGGERS) for m in MODELS),
        "available_feature_relationships":len(feature_analysis["features"])+1,
        "available_feature_relationships_per_period":(len(feature_analysis["features"])+1)*2,
        "unavailable_feature_relationships":len(feature_analysis["unavailable"]),
        "interpretation":"descriptive comparisons, not independent hypothesis tests; no p-value selection"},
      "spring_october_compatibility":_period_compatibility(surface),"candidate_mechanisms":"NONE_SELECTED_FROM_THIS_EXPLORATORY_RUN","economic_edge_validated":False,"production_strategy_approved":False,
      "data_downloaded":False,"2026_data_accessed":False,"final_oos_accessed":False,"production_code_changed":False,"runtime_seconds":time.monotonic()-started}
    _write_json(output_root/"summary.json",summary)
    report=_render_report(summary,surface,feature_analysis)
    (output_root/"report.md").write_text(report,encoding="utf-8")
    manifest={"status":"COMPLETE","study_id":STUDY_ID,"config_sha256":_sha_obj(CONFIG),"source_coverage_sha256":coverage_ref["sha256"],
              "files":{},"completed_dates":list(days),"runtime_seconds":summary["runtime_seconds"],"no_2026":True,"no_oos":True,"no_download":True}
    _write_json(output_root/"run-manifest.json",manifest)
    files=[p for p in output_root.iterdir() if p.is_file() and p.name!="artifact-hashes.json"]
    manifest["files"]={p.name:_sha(p) for p in sorted(files) if p.name!="run-manifest.json"}; _write_json(output_root/"run-manifest.json",manifest)
    hashes={p.name:_sha(p) for p in sorted(output_root.iterdir()) if p.is_file() and p.name!="artifact-hashes.json"}
    _write_json(output_root/"artifact-hashes.json",{"status":"HASHED","files":hashes})
    return summary


def _dist(vals: Sequence[float], expected: int | None = None) -> dict[str, Any]:
    a=np.asarray(vals,dtype=float)
    if expected is not None and len(a)<expected: a=np.pad(a,(0,expected-len(a)))
    if not len(a): return {"n":0}
    return {"n":len(a),"min":float(np.min(a)),"p10":float(np.quantile(a,.1)),"median":float(np.median(a)),"p90":float(np.quantile(a,.9)),"max":float(np.max(a)),"mean":float(np.mean(a))}


def _feature_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric=("local_delta_30s_supportive_fraction","local_delta_2m_supportive_fraction","session_cvd_supportive_fraction","price_progress_toward_ticks_2m","opposing_aggression_fraction_2m","effort_without_result","price_impact_ticks_per_100_aggressive_contracts")
    result={"method":"descriptive only; per-signal mean net R averaged over executed fixed exit cells; no threshold search or p-value selection",
            "features":{},"period_relationships":{},"aggression_reversal_outcomes":{},
            "reversion_opposing_aggression_vs_price_progress":{},"unavailable":{"TOP5_DEPTH_IMBALANCE":"candidate tape lacks depth ladder","NORMALIZED_MLOFI_PERSISTENCE":"no compatible event-time feature stream across both periods"}}
    for feature in numeric:
        xs=[]; ys=[]
        for row in rows:
            x=row.get(feature);y=row.get("mean_net_r_across_executed_fixed_cells")
            if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)):
                xs.append(float(x));ys.append(float(y))
        corr=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 and np.std(xs)>0 and np.std(ys)>0 else None
        result["features"][feature]={"n":len(xs),"pearson_descriptive_correlation_with_event_mean_net_r":corr,"feature_distribution":_dist(xs)}
        result["period_relationships"][feature]={}
        for period in ("SPRING_2025","OCTOBER_2025"):
            pairs=[(float(row[feature]),float(row["mean_net_r_across_executed_fixed_cells"])) for row in rows
                   if row.get("period")==period and row.get(feature) is not None and row.get("mean_net_r_across_executed_fixed_cells") is not None
                   and math.isfinite(float(row[feature])) and math.isfinite(float(row["mean_net_r_across_executed_fixed_cells"]))]
            px=[x for x,_ in pairs]; py=[y for _,y in pairs]
            pcorr=float(np.corrcoef(px,py)[0,1]) if len(pairs)>2 and np.std(px)>0 and np.std(py)>0 else None
            result["period_relationships"][feature][period]={"n":len(pairs),"pearson_descriptive_correlation_with_event_mean_net_r":pcorr}
    result["aggression_reversal_outcomes"]={}
    for period in ("ALL","SPRING_2025","OCTOBER_2025"):
        selected=[row for row in rows if row.get("mean_net_r_across_executed_fixed_cells") is not None
                  and (period=="ALL" or row.get("period")==period)]
        groups={}
        for flag in (True,False):
            values=[float(row["mean_net_r_across_executed_fixed_cells"]) for row in selected if row.get("aggression_reversal") is flag]
            groups["REVERSAL_TRUE" if flag else "REVERSAL_FALSE"]={"signals":len(values),"mean_event_net_r":float(np.mean(values)) if values else None}
        result["aggression_reversal_outcomes"][period]=groups
    rev=[r for r in rows if r["model"]=="FAIR_PRICE_REVERSION" and r.get("opposing_aggression_fraction_2m") is not None and r.get("price_progress_toward_ticks_2m") is not None]
    if rev:
        opp=np.asarray([float(r["opposing_aggression_fraction_2m"]) for r in rev]); progress=np.asarray([float(r["price_progress_toward_ticks_2m"]) for r in rev])
        high=float(np.quantile(opp,.75)); med=float(np.median(np.abs(progress)))
        for period in ("ALL","SPRING_2025","OCTOBER_2025"):
            subset=[r for r in rev if period=="ALL" or r["period"]==period]
            groups={"OPPOSING_AGGRESSION_HIGH_PRICE_PROGRESS_LOW":[],"OPPOSING_AGGRESSION_HIGH_PRICE_PROGRESS_HIGH":[],"OTHER":[]}
            for r in subset:
                ishigh=float(r["opposing_aggression_fraction_2m"])>=high
                low=abs(float(r["price_progress_toward_ticks_2m"]))<=med
                groups["OPPOSING_AGGRESSION_HIGH_PRICE_PROGRESS_LOW" if ishigh and low else "OPPOSING_AGGRESSION_HIGH_PRICE_PROGRESS_HIGH" if ishigh else "OTHER"].append(r)
            result["reversion_opposing_aggression_vs_price_progress"][period]={"feature_only_cutoffs":{"opposing_aggression_q75_all_reversion_signals":high,"absolute_price_progress_median_all_reversion_signals":med},
              "groups":{k:{"signals":len(v),"mean_event_net_r":float(np.mean([x["mean_net_r_across_executed_fixed_cells"] for x in v if x["mean_net_r_across_executed_fixed_cells"] is not None])) if any(x["mean_net_r_across_executed_fixed_cells"] is not None for x in v) else None} for k,v in groups.items()}}
    return result


def _period_compatibility(surface: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by={(r["session"],r["model"],r["trigger"],r["stop_family"],r["target_model"],r["period"]):r for r in surface}
    rows=[]
    for r in surface:
        if r["period"]!="SPRING_2025": continue
        other=by.get((r["session"],r["model"],r["trigger"],r["stop_family"],r["target_model"],"OCTOBER_2025"))
        if other:
            a,b=r.get("average_net_r"),other.get("average_net_r")
            rows.append({"session":r["session"],"model":r["model"],"trigger":r["trigger"],"stop_family":r["stop_family"],"target_model":r["target_model"],
                         "spring_avg_net_r":a,"october_avg_net_r":b,"same_sign":bool(a is not None and b is not None and (a==0 or b==0 or math.copysign(1,a)==math.copysign(1,b)))})
    return {"cell_count":len(rows),"same_sign_cells":sum(x["same_sign"] for x in rows),"rows":rows,"interpretation":"period compatibility is descriptive; neither sample is untouched validation/OOS"}


def _render_report(summary: Mapping[str, Any], surface: Sequence[Mapping[str, Any]], features: Mapping[str, Any]) -> str:
    lines=["# ES JJ Fair Pricing Expanded V2 — Exploratory Report","",f"Processed {summary['dates_processed']} dates; {summary['valid_ny_am_sessions']} eligible AM anchors and {summary['valid_ny_pm_sessions']} eligible PM anchors.",
      f"Raw qualified signals: {summary['total_raw_signals']}; sequential fills summed across independent fixed exit configurations: {summary['total_sequential_trades_across_independent_fixed_exit_configurations']}.",
      "No parameter optimization, production selection, downloads, 2026, or final OOS were used.","",
      "Public descriptions conflict on session window cutoffs and whether a valid entry requires displacement alone or BOS plus displacement. This run uses the frozen operationalization in `study-config.json`; see `public-rule-comparison.md`.",
      "TOP5 depth imbalance and normalized MLOFI persistence are unavailable in the sealed BBO/execution tape and are reported null, not approximated. Delta/CVD/flow-price diagnostics use only tape records at or before each completed-candle signal.","",
      "## Fixed exit surface","","See `exit-model-surface.csv` and period files for all fixed stop/target cells, including negative and invalid cells. No best cell is selected.","",
      "## Period separation","",json.dumps(summary["spring_october_compatibility"],sort_keys=True),"",
      "## Feature diagnostics","",json.dumps(features,sort_keys=True),"",
      "## Research limits","","All 54 dates are development-only. The comparisons are descriptive and subject to substantial multiple testing; no unadjusted significance claim or production filter is supported.",""]
    return "\n".join(lines)


def write_public_comparison(output_root: Path = OUT_ROOT) -> None:
    output_root.mkdir(parents=True,exist_ok=True)
    text="""# Public-rule comparison and V2 operationalizations

## Public descriptions found

- The user-supplied video is the primary reference: https://www.youtube.com/watch?v=KHEQ5g55dQ4 (direct page/transcript was not retrievable in this audit environment).
- JJ Simon-authored FX Replay breakdown, dated 2026-06-23: https://fxreplay-webflow.fxreplay.app/strategies/jj-simons-fair-value-theory-nq-strategy . It describes a 09:30 opening fair-value price and 14:00 afternoon fair-value price, first 10–15 minutes as continuation, then reversion; its entry description requires BOS/MSB with displacement, and it describes a small counter-wick criterion and ATR risk / fixed 1.5R target.
- Accessible Fair Pricing Theory breakdown: https://www.chartfanatics.com/strategies/fair-pricing-theory-strategy . It describes displacement candle entries and BOS as distinct alternatives; its candle operationalization is a larger body, close beyond previous wick, and displacement of an opposite-color candle. It also describes continuation followed by reversion and notes that anchors/targets can change with context/account rules.

These are public descriptions, not a source of a single exact private JJ configuration. They differ on whether displacement alone can trigger or BOS plus displacement is required, displacement wick-quality thresholds, and session cutoffs. Public examples include news/pre-news fair-price anchors and other sessions; V2 excludes those because this dataset contract covers only NY 09:30 and 14:00 anchors. The public theory is described on NQ; this study tests ES and cannot be assumed to transfer to NQ.

## Frozen V2 research choices

V2 separately catalogs continuation (first 15 minutes, away from anchor) and reversion (remaining session, toward anchor). It independently labels displacement-only, two-prior-candle close BOS-only, and same-candle BOS-plus-displacement. A displacement candle is completed 1-minute trade OHLC: body larger than prior body, close beyond prior relevant wick, opposite-color prior candle, and direction aligned with the proposed trade. No 8-point multi-candle move is substituted.

AM is fixed to 09:30–11:00 ET; PM to 14:00–15:00 ET for this study. A separate first-trade opening-minute anchor is used per session. These fixed windows are a research scope choice, not a claim that the public source defines exact cutoffs. Signals are candle-level with stable identity and no daily count cap. Executable trades are sequential within each independently evaluated exit cell.

Stops and targets are the preregistered 3-by-4 descriptive grid in `study-config.json`. Anchor targets apply only to reversion. Entries and exits use the validated Candidate Tape V2 BBO/execution path, 2ms delay, executable sides, adverse tick, project fees, and stop-first precedence. All cells are reported; none is selected as a production winner.
"""
    (output_root/"public-rule-comparison.md").write_text(text,encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=OUT_ROOT)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args(argv)
    result=run(output_root=args.output_root,smoke=args.smoke,force=args.force)
    print(json.dumps(result,sort_keys=True,default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
