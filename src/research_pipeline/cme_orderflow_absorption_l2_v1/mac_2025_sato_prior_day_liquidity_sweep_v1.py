"""Frozen public-model PDH/PDL sweep and reclaim event study for MAC 2025.

This is an event/mechanism study, not an optimizer or strategy PnL runner.
It uses only the repository's sealed native ES MBP-10 source and canonical
ES event tapes. Spring is primary DEV; October is secondary DEV compatibility.
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import mac_2025_es_flow_momentum_v1 as flow
from . import mac_2025_es_liquidity_vacuum_v1 as native
from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_native_mbp_quote as plan
from . import mac_2025_es_value_area_migration_event_study_v1 as prior_study

RUN_ID = "CMEOrderflow_SATO_ES_PRIOR_DAY_LIQUIDITY_SWEEP_RECLAIM_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
TICK = 0.25
BAR_NS = 300_000_000_000
ENTRY_DELAY_NS = 2_000_000
MAX_RECLAIM_BARS = 3
SWEEP_START_ET = (9, 30)
SWEEP_END_ET = (10, 0)
RTH_OPEN_ET = (9, 30)
RTH_CLOSE_ET = (16, 0)
MARKOUT_MS = (250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 120000, 300000, 600000, 1800000)
EXCURSION_MS = (5000, 10000, 30000, 60000, 120000, 300000, 600000, 1800000)
ACCEPTANCE_MS = (10000, 30000, 60000, 120000, 300000, 600000, 1800000)
STUDY_VERSION = "sato-prior-day-sweep-reclaim-v1.2"
CHECKPOINT_VERSION = "sato-pdh-pdl-per-date-v3"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
# No early-close session falls inside these target/prior-date windows. Dates
# must be added here only from an official, product-specific CME schedule.
OFFICIAL_RTH_EARLY_CLOSE_ET: dict[str, tuple[int, int]] = {}

SPRING_DATES = tuple(flow.SPRING_DATES)
OCTOBER_DATES = tuple(flow.OCTOBER_DATES)
TARGET_DATES = SPRING_DATES + OCTOBER_DATES
SOURCE_DATES = tuple(native.ALL_SOURCE_DATES)
DEPENDENCY_DATES = tuple(native.DEPENDENCY_DATES)

CONFIG: dict[str, Any] = {
    "study_version": STUDY_VERSION,
    "model": "public prior-day RTH high/low liquidity sweep + fast reclaim + public orderflow confirmation",
    "source": "native GLBX.MDP3 ES mbp-10 only; no MBO, MES, NQ, private Sato levels, OOS, 2026, downloads",
    "period_roles": {"SPRING_2025": "PRIMARY_DEV_DISCOVERY", "OCTOBER_2025": "SECONDARY_DEV_COMPATIBILITY"},
    "prior_rth": "previous valid RTH trading date; actual ES trades in [09:30 ET,16:00 ET); levels fixed before current RTH",
    "early_close_policy": "use deterministic official session close only; otherwise exclude dependent current session",
    "current_rth": "09:30 ET inclusive to 16:00 ET exclusive",
    "sweep_window": "09:30 ET inclusive to 10:00 ET exclusive; first strict trade through PDH and first strict trade through PDL, at most one each",
    "bars": "fixed 5m trade bars anchored to current RTH open; six completed chronological pre-breach reference bars; unavailable references stay unavailable",
    "reclaim": "first completed breach-containing or following 5m bar with close strictly back inside prior range, max 3 bars including breach bar",
    "volume_spike": "breach-bar total volume / mean total volume of previous six fully completed 5m bars >= 1.50; V1 convention, not a private Sato threshold",
    "delta_divergence": "at least one breach-through-reclaim bar has break-direction Delta and closes non-progressing (PDH Delta>0 close<=open; PDL Delta<0 close>=open)",
    "absorption": "MBP10_ABSORPTION_PROXY only: event break-direction aggression rate above median prior-six comparable bars and price impact per flow below their median",
    "stack": "reclaim mandatory plus at least two of volume spike, Delta divergence, absorption; no threshold search",
    "entry": "reclaim bar close; 2ms delay; first executable quote; long ask / short bid; actual-fill sensitivity adverse one ES tick",
    "exit_markout": "long liquidation bid; short liquidation ask; same elapsed horizon from raw signal and executable entry respectively",
    "structural_stop": "one tick beyond observed sweep extreme; diagnostic first-touch only",
    "targets": "freeze RTH VWAP at reclaim, prior-day midpoint, opposite prior-day extreme; first touch vs structural stop",
    "no_optimization": True, "no_strategy_pnl": True, "private_sato_levels_used": False, "true_mbo_used": False,
}
CONFIG_SHA256 = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SatoStudyError(RuntimeError):
    """Source, coverage, causal, or checkpoint contract failure."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_json), encoding="utf-8")
    os.replace(temp, path)


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            for row in rows:
                out.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json).encode() + b"\n")
    os.replace(temp, path)


def _period(day: str) -> str:
    return "SPRING_2025" if day in SPRING_DATES else "OCTOBER_2025"


def _ns(day: str, hh: int, mm: int = 0) -> int:
    dt = datetime.fromisoformat(day).replace(hour=hh, minute=mm, second=0, microsecond=0, tzinfo=ET)
    return int(dt.astimezone(UTC).timestamp() * 1e9)


def _rth(day: str) -> tuple[int, int]:
    return _ns(day, *RTH_OPEN_ET), _ns(day, *OFFICIAL_RTH_EARLY_CLOSE_ET.get(day, RTH_CLOSE_ET))


def _checkpoint_key(day: str, source_sha: str, prior_source_sha: str) -> str:
    payload={"day":day,"source_sha":source_sha,"prior_source_sha":prior_source_sha,
             "config_sha":CONFIG_SHA256,"study_version":STUDY_VERSION,"version":CHECKPOINT_VERSION}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def previous_valid_rth_map(dates: Sequence[str], valid_source_dates: Sequence[str]) -> dict[str, str]:
    valid = sorted(set(valid_source_dates))
    result = {}
    for day in dates:
        candidates = [x for x in valid if x < day]
        if not candidates:
            raise SatoStudyError(f"no previous valid RTH source for {day}")
        result[day] = candidates[-1]
    return result


def _trades(events: np.ndarray, start: int, end: int) -> np.ndarray:
    ts = events["timestamp_ns"].astype(np.int64, copy=False)
    return np.flatnonzero((ts >= start) & (ts < end) & (events["execution_size"] > 0)
                          & np.isfinite(events["execution_price"]))


def _bars(day: str, tape: np.ndarray, *, start_ns: int | None = None, end_ns: int | None = None,
          anchor_ns: int | None = None) -> list[dict[str, Any]]:
    """Fixed 5m actual-trade bars. Optional pre-RTH bins share a 5m UTC grid."""
    rth_open, rth_close = _rth(day)
    lo = rth_open if start_ns is None else start_ns
    hi = rth_close if end_ns is None else end_ns
    anchor = rth_open if anchor_ns is None else anchor_ns
    ids = _trades(tape, lo, hi)
    if not len(ids):
        return []
    ts = tape["timestamp_ns"][ids].astype(np.int64, copy=False)
    p = tape["execution_price"][ids].astype(np.float64, copy=False)
    q = tape["execution_size"][ids].astype(np.int64, copy=False)
    ag = tape["aggressor"][ids].astype(np.int8, copy=False)
    first = math.floor((int(ts[0]) - anchor) / BAR_NS)
    last = math.floor((int(ts[-1]) - anchor) / BAR_NS)
    out = []
    for index in range(first, last + 1):
        bstart = anchor + index * BAR_NS
        bend = bstart + BAR_NS
        a = int(np.searchsorted(ts, bstart, side="left")); b = int(np.searchsorted(ts, bend, side="left"))
        if b <= a:
            out.append({"bar_index": index, "start_ns": int(bstart), "close_ns": int(bend), "date": day,
                        "status": "NO_TRADES", "open": None, "high": None, "low": None, "close": None,
                        "total_volume": 0, "buy_volume": 0, "sell_volume": 0, "delta": 0, "trade_count": 0})
            continue
        pp, qq, aa = p[a:b], q[a:b], ag[a:b]
        out.append({"bar_index": index, "start_ns": int(bstart), "close_ns": int(bend), "status": "COMPLETE",
                    "open": float(pp[0]), "high": float(np.max(pp)), "low": float(np.min(pp)),
                    "close": float(pp[-1]), "total_volume": int(np.sum(qq)),
            "buy_volume": int(np.sum(qq[aa == 1])), "sell_volume": int(np.sum(qq[aa == -1])),
                    "delta": int(np.sum(qq[aa == 1]) - np.sum(qq[aa == -1])),
                    "aggressor_known_count": int(np.sum(aa != 0)),
                    "trade_count": int(b-a), "date": day})
    return out


def _prior_trade_bounds(day: str) -> tuple[int, int]:
    return _rth(day)


def _raw_dependency_trades(path: Path, day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded-memory extraction for the two dependency sessions lacking tape files."""
    try:
        from databento import DBNStore
    except ImportError as exc:
        raise SatoStudyError("databento is required to read the sealed dependency DBNs") from exc
    start, end = _prior_trade_bounds(day)
    tt: list[np.ndarray] = []; pp: list[np.ndarray] = []; qq: list[np.ndarray] = []
    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
        ts = batch["ts_recv"].astype(np.int64, copy=False)
        mask = (ts >= start) & (ts < end) & (batch["action"] == b"T") & (batch["size"] > 0)
        if np.any(mask):
            tt.append(ts[mask].copy()); pp.append(batch["price"][mask].astype(np.float64) / 1e9); qq.append(batch["size"][mask].astype(np.int64))
    if not tt:
        raise SatoStudyError(f"no actual trades in dependency prior RTH: {day}")
    return np.concatenate(tt), np.concatenate(pp), np.concatenate(qq)


def _prior_levels(day: str, prev: str, paths: Mapping[str, Path], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    start, end = _prior_trade_bounds(prev)
    request_start = datetime.fromisoformat(str(rows[prev]["start"]).replace("Z", "+00:00"))
    request_end = datetime.fromisoformat(str(rows[prev]["end"]).replace("Z", "+00:00"))
    if int(request_start.timestamp()*1e9) > start or int(request_end.timestamp()*1e9) < end:
        raise SatoStudyError(f"prior RTH request bounds incomplete for target={day}, prior={prev}")
    if prev in TARGET_DATES:
        ev, _ = native._load_tape(prev, native._tape_path(prev), str(rows[prev]["sha256"]))
        ix = _trades(ev, start, end)
        if not len(ix):
            raise SatoStudyError(f"no prior RTH trades for target={day}, prior={prev}")
        prices = ev["execution_price"][ix].astype(np.float64)
        count = int(len(ix))
        first_ns = int(ev["timestamp_ns"][ix[0]]); last_ns = int(ev["timestamp_ns"][ix[-1]])
    else:
        ts, prices, sizes = _raw_dependency_trades(paths[prev], prev)
        count = int(len(ts)); first_ns = int(ts[0]); last_ns = int(ts[-1])
    high = float(np.max(prices)); low = float(np.min(prices))
    if not np.isfinite(high + low) or high < low or abs(high / TICK - round(high / TICK)) > 1e-7 or abs(low / TICK - round(low / TICK)) > 1e-7:
        raise SatoStudyError(f"invalid prior trade levels for {day} from {prev}")
    return {"current_session": day, "previous_valid_rth_date": prev, "coverage_complete": True,
            "source_file": str(paths[prev]), "source_sha256": str(rows[prev]["sha256"]),
            "prior_rth_start_ns": start, "prior_rth_end_ns": end, "trade_count": count,
            "first_trade_ns": first_ns, "last_trade_ns": last_ns, "pdh": high, "pdl": low,
            "midpoint": (high + low) / 2, "range_ticks": (high - low) / TICK,
            "early_close": prev in OFFICIAL_RTH_EARLY_CLOSE_ET,
            "close_policy": (f"09:30-{OFFICIAL_RTH_EARLY_CLOSE_ET[prev][0]:02d}:{OFFICIAL_RTH_EARLY_CLOSE_ET[prev][1]:02d} ET official override"
                             if prev in OFFICIAL_RTH_EARLY_CLOSE_ET else "09:30-16:00 ET normal RTH; CME 2025-04-17 notice states other products normal times")}


def _quote_mid(tape: np.ndarray, at: int, *, side: str = "left", end: int | None = None) -> tuple[int, float] | None:
    ts = tape["timestamp_ns"].astype(np.int64, copy=False)
    i = int(np.searchsorted(ts, at, side=side))
    while i < len(tape) and (end is None or int(ts[i]) < end):
        b, a = float(tape["bid"][i]), float(tape["ask"][i])
        if math.isfinite(b) and math.isfinite(a) and a > b:
            return i, (b + a) / 2
        i += 1
    return None


def _make_candidate(day: str, side: str, level: float, tape: np.ndarray, bars: list[dict[str, Any]],
                    all_bars: list[dict[str, Any]], prior: dict[str, Any], trade_ids: np.ndarray,
                    trade_ts: np.ndarray, trade_px: np.ndarray, trade_sz: np.ndarray,
                    trade_ag: np.ndarray) -> dict[str, Any] | None:
    open_ns, close_ns = _rth(day)
    sweep_end = _ns(day, *SWEEP_END_ET)
    current_window = (trade_ts >= open_ns) & (trade_ts < sweep_end)
    first = np.flatnonzero(current_window & (trade_px > level if side == "PDH" else trade_px < level))
    if not len(first):
        return None
    k = int(first[0]); breach_ns = int(trade_ts[k]); breach_price = float(trade_px[k])
    bar_pos = next((i for i,b in enumerate(bars) if b["start_ns"] <= breach_ns < b["close_ns"]), None)
    if bar_pos is None:
        return None
    eligible = bars[bar_pos:bar_pos + MAX_RECLAIM_BARS]
    reclaim = None
    for rank, bar in enumerate(eligible, 1):
        if bar.get("status", "COMPLETE") != "COMPLETE":
            continue
        inside = bar["close"] < level if side == "PDH" else bar["close"] > level
        if inside:
            reclaim = (rank, bar)
            break
    horizon_bars = eligible[:(reclaim[0] if reclaim else min(MAX_RECLAIM_BARS, len(eligible)))]
    last_time = int((reclaim[1] if reclaim else (horizon_bars[-1] if horizon_bars else bars[bar_pos]))["close_ns"])
    stop_ts = trade_ts[k:]
    segment_mask = (trade_ts >= breach_ns) & (trade_ts < last_time)
    seg_ids = np.flatnonzero(segment_mask)
    if not len(seg_ids):
        return None
    seg_px = trade_px[seg_ids]; seg_sz = trade_sz[seg_ids]; seg_ag = trade_ag[seg_ids]
    inside_after = np.flatnonzero((trade_ts >= breach_ns) & (trade_ts < last_time) &
                                  (trade_px < level if side == "PDH" else trade_px > level))
    spent_until = int(trade_ts[inside_after[0]]) if len(inside_after) else last_time
    sign = -1 if side == "PDH" else 1
    extreme = float(np.max(seg_px) if side == "PDH" else np.min(seg_px))
    extension = abs(extreme - level) / TICK
    duration_s = max((last_time - breach_ns) / 1e9, 1e-9)
    break_aggr = int(np.sum(seg_sz[seg_ag == (1 if side == "PDH" else -1)]))
    classified_aggressor = any(np.any(trade_ag[(trade_ts >= b["start_ns"]) & (trade_ts < b["close_ns"])] != 0)
                               for b in horizon_bars if b.get("status", "COMPLETE") == "COMPLETE")
    delta_div = any((b["delta"] > 0 and b["close"] <= b["open"]) if side == "PDH" else
                    (b["delta"] < 0 and b["close"] >= b["open"]) for b in horizon_bars
                    if b.get("status", "COMPLETE") == "COMPLETE")
    # The six immediately preceding chronological 5m bins must all exist, complete, and be populated.
    by_index={int(b["bar_index"]):b for b in all_bars if b["close_ns"]<=bars[bar_pos]["start_ns"]}
    prior_six=[by_index.get(int(bars[bar_pos]["bar_index"])-offset) for offset in range(6,0,-1)]
    refs_valid=all(b is not None and b.get("status","COMPLETE")=="COMPLETE" and b["total_volume"]>0 for b in prior_six)
    ref_bars=[b for b in prior_six if b is not None] if refs_valid else []
    volume_ratio = None
    if len(ref_bars) == 6 and np.mean([b["total_volume"] for b in ref_bars]) > 0:
        volume_ratio = bars[bar_pos]["total_volume"] / float(np.mean([b["total_volume"] for b in ref_bars]))
    volume_spike = volume_ratio is not None and volume_ratio >= 1.50
    comp_bars = ref_bars
    ref_rate = None; ref_impact = None
    if len(comp_bars) == 6 and all(int(b.get("aggressor_known_count",0))>0 for b in comp_bars) and np.any(seg_ag != 0):
        rates=[]; impacts=[]
        for b in comp_bars:
            agvol = b["buy_volume"] if side == "PDH" else b["sell_volume"]
            rates.append(agvol / 300.0)
            impacts.append(abs(b["close"] - b["open"]) / TICK / max(agvol, 1e-12))
        ref_rate=float(np.median(rates)); ref_impact=float(np.median(impacts))
    flow_rate=break_aggr/duration_s
    price_impact=extension/max(break_aggr,1e-12)
    absorption = (ref_rate is not None and ref_impact is not None and flow_rate > ref_rate and price_impact < ref_impact)
    reclaim_rank = reclaim[0] if reclaim else None
    reclaim_bar = reclaim[1] if reclaim else None
    signal_ns = int(reclaim_bar["close_ns"]) if reclaim_bar else None
    signal_close = float(reclaim_bar["close"]) if reclaim_bar else None
    # Causal current RTH VWAP at signal from actual trades only.
    rth_mask = (trade_ts >= open_ns) & (trade_ts < (signal_ns if signal_ns is not None else last_time))
    if np.any(rth_mask):
        vwap = float(np.sum(trade_px[rth_mask] * trade_sz[rth_mask]) / np.sum(trade_sz[rth_mask]))
    else:
        vwap = None
    comps = int(bool(volume_spike)) + int(bool(delta_div)) + int(bool(absorption))
    qualified = bool(reclaim_bar is not None and comps >= 2)
    # CVD is causal and starts at RTH open; unknown aggressor rows contribute zero.
    cvd = np.cumsum(np.where(trade_ag == 1, trade_sz, np.where(trade_ag == -1, -trade_sz, 0)))
    def cvd_at(t: int) -> int:
        ix = int(np.searchsorted(trade_ts, t, side="right") - 1)
        return int(cvd[ix]) if ix >= 0 and trade_ts[ix] >= open_ns else 0
    extreme_ix = int(seg_ids[np.flatnonzero(seg_px == extreme)[0]])
    extreme_time = int(trade_ts[extreme_ix])
    return {"date": day, "period": _period(day), "side": side,
            "level_type": side, "level_price": level, "break_direction": "UP" if side == "PDH" else "DOWN",
            "reversal_direction": "SHORT" if side == "PDH" else "LONG", "breach_timestamp_ns": breach_ns,
            "breach_price": breach_price, "breach_bar_index": bars[bar_pos]["bar_index"],
            "breach_time_bucket": ("09:30-09:40" if breach_ns < _ns(day,9,40) else "09:40-09:50" if breach_ns < _ns(day,9,50) else "09:50-10:00"),
            "distance_beyond_level_ticks": abs(breach_price-level)/TICK, "sweep_extreme": extreme,
            "sweep_extension_ticks": float(extension), "extreme_timestamp_ns": extreme_time,
            "time_breach_to_extreme_ms": (extreme_time-breach_ns)/1e6,
            "time_breach_to_reclaim_ms": (signal_ns-breach_ns)/1e6 if signal_ns else None,
            "time_spent_beyond_level_ms": (spent_until-breach_ns)/1e6,
            "reclaim_speed": ("SAME_BAR" if reclaim_rank==1 else "NEXT_BAR" if reclaim_rank==2 else "THIRD_BAR" if reclaim_rank==3 else None),
            "reclaim_bar_rank": reclaim_rank, "reclaim_timestamp_ns": signal_ns,
            "reclaim_close": signal_close, "no_reclaim_acceptance_candidate": reclaim_bar is None,
            "acceptance_timestamp_ns": int(horizon_bars[-1]["close_ns"]) if not reclaim_bar and horizon_bars else None,
            "breach_bar_total_volume": int(bars[bar_pos]["total_volume"]),
            "reference_volume": float(np.mean([b["total_volume"] for b in ref_bars])) if len(ref_bars)==6 else None,
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "volume_spike": bool(volume_spike), "volume_spike_status": "AVAILABLE" if volume_ratio is not None else "INSUFFICIENT_REFERENCE",
            "delta_divergence": bool(delta_div), "delta_divergence_status": "AVAILABLE" if classified_aggressor else "INSUFFICIENT_AGGRESSOR_DATA",
            "rth_cvd_at_breach": cvd_at(breach_ns), "rth_cvd_at_extreme": cvd_at(extreme_time),
            "rth_cvd_at_reclaim": cvd_at(signal_ns) if signal_ns else None,
            "rth_cvd_change_breach_to_reclaim": cvd_at(signal_ns)-cvd_at(breach_ns) if signal_ns else None,
            "break_direction_aggressive_volume": break_aggr, "break_direction_aggressive_volume_rate": flow_rate,
            "break_direction_price_impact_per_flow": float(price_impact),
            "reference_aggressive_volume_rate_median": ref_rate,
            "reference_price_impact_per_flow_median": ref_impact,
            "absorption": bool(absorption), "absorption_status": "AVAILABLE" if ref_rate is not None else
                "INSUFFICIENT_REFERENCE" if len(comp_bars)<6 else "INSUFFICIENT_AGGRESSOR_DATA",
            "pre_reclaim_support_count": comps if reclaim_bar else None,
            "total_public_stack_count": comps+1 if reclaim_bar else None,
            "public_stack_qualified": qualified, "prior_day_high": prior["pdh"], "prior_day_low": prior["pdl"],
            "prior_day_midpoint": prior["midpoint"], "prior_day_range_ticks": prior["range_ticks"],
            "session_vwap_at_reclaim": vwap, "reclaim_only": reclaim_bar is not None}


def _path_event(event: dict[str, Any], tape: np.ndarray, rth_close: int) -> dict[str, Any]:
    t = event.get("reclaim_timestamp_ns")
    if t is None:
        return event
    sign = 1 if event["reversal_direction"] == "LONG" else -1
    signal = int(t)
    raw = _quote_mid(tape, signal, side="right", end=rth_close)
    entry = _quote_mid(tape, signal + ENTRY_DELAY_NS, side="left", end=rth_close)
    if raw is None or entry is None:
        event["path_status"] = "NO_EXECUTABLE_QUOTE"
        return event
    ri, rm = raw; ei, em = entry
    ts=tape["timestamp_ns"].astype(np.int64,copy=False)
    bid=tape["bid"].astype(float,copy=False); ask=tape["ask"].astype(float,copy=False)
    last_i=int(np.searchsorted(ts,signal,side="right")-1)
    while last_i>=0 and not (tape["execution_size"][last_i]>0 and np.isfinite(tape["execution_price"][last_i])):
        last_i-=1
    last_trade=float(tape["execution_price"][last_i]) if last_i>=0 else None
    entry_quote=float(ask[ei] if sign>0 else bid[ei]); actual_fill=entry_quote+sign*TICK
    paths={}
    for h in MARKOUT_MS:
        rawq=_quote_mid(tape,signal+h*1_000_000,side="left",end=rth_close)
        exq=_quote_mid(tape,int(ts[ei])+h*1_000_000,side="left",end=rth_close)
        raw_m=sign*(rawq[1]-rm)/TICK if rawq else None
        quote_m=sign*((float(bid[exq[0]]) if sign>0 else float(ask[exq[0]]))-entry_quote)/TICK if exq else None
        actual_m=sign*((float(bid[exq[0]]) if sign>0 else float(ask[exq[0]]))-actual_fill)/TICK if exq else None
        paths[str(h)]={"raw":raw_m,"quote":quote_m,"actual":actual_m,
                       "pre_entry_move_ticks":sign*(em-rm)/TICK,
                       "bid_ask_entry_effect_ticks":sign*((em-entry_quote))/TICK,
                       "adverse_entry_tick":-1.0,
                       "horizon_alignment_ticks":(sign*(exq[1]-rawq[1])/TICK if exq and rawq else None)}
    excursion={}
    for h in EXCURSION_MS:
        endq=_quote_mid(tape,signal+h*1_000_000,side="left",end=rth_close)
        if not endq: excursion[str(h)]={"raw":None,"quote":None,"actual":None};continue
        hi=endq[0]+1
        mid=sign*((bid[ri:hi]+ask[ri:hi])/2-rm)/TICK
        liq=bid[ei:hi] if sign>0 else ask[ei:hi]
        qv=sign*(liq-entry_quote)/TICK; av=sign*(liq-actual_fill)/TICK
        excursion[str(h)]={"raw":{"mfe":float(max(0,np.max(mid))),"mae":float(min(0,np.min(mid)))},
                           "quote":{"mfe":float(max(0,np.max(qv))),"mae":float(min(0,np.min(qv)))},
                           "actual":{"mfe":float(max(0,np.max(av))),"mae":float(min(0,np.min(av)))}}
    stop=float(event["sweep_extreme"] - sign*TICK)
    # For short sign=-1 this is extreme+tick; for long it is extreme-tick.
    def first_touch(target: float | None, until: int) -> dict[str, Any]:
        if target is None:return {"result":"UNAVAILABLE","touch_ms":None}
        j0=ei; j1=int(np.searchsorted(ts,until,side="left"))
        if j1<=j0:return {"result":"NEITHER","touch_ms":None}
        exit_side=bid if sign>0 else ask
        stop_hit=(exit_side[j0:j1] <= stop) if sign>0 else (exit_side[j0:j1] >= stop)
        target_hit=(exit_side[j0:j1] >= target) if sign>0 else (exit_side[j0:j1] <= target)
        si=np.flatnonzero(stop_hit); ti=np.flatnonzero(target_hit)
        if len(si)==0 and len(ti)==0:return {"result":"NEITHER","touch_ms":None}
        sidx=j0+int(si[0]) if len(si) else 10**30; tidx=j0+int(ti[0]) if len(ti) else 10**30
        ix=min(sidx,tidx)
        return {"result":"STOP_FIRST" if sidx<=tidx else "TARGET_FIRST","touch_ms":(int(ts[ix])-signal)/1e6}
    targets={"structural_stop":stop,"session_vwap":event.get("session_vwap_at_reclaim"),
             "prior_midpoint":event.get("prior_day_midpoint"),
             "opposite_prior_extreme":event.get("prior_day_low" if sign<0 else "prior_day_high")}
    touches={}
    for name,target in targets.items():
        if name=="structural_stop":continue
        tvalid=target
        if tvalid is not None and ((sign>0 and tvalid<=actual_fill) or (sign<0 and tvalid>=actual_fill)):
            touches[name]={"result":"MIDPOINT_TARGET_NOT_DIRECTIONALLY_VALID" if name=="prior_midpoint" else "NOT_DIRECTIONALLY_VALID","touch_ms":None}
        else:touches[name]=first_touch(float(tvalid) if tvalid is not None else None,rth_close)
    reward_risk={}
    risk=abs(actual_fill-stop)
    for name,target in targets.items():
        if name=="structural_stop" or target is None or risk<=0:
            continue
        favorable=sign*(float(target)-actual_fill)
        reward_risk[name]=(float(favorable/risk) if favorable>0 else None)
    event.update({"path_status":"AVAILABLE","raw_anchor_timestamp_ns":int(ts[ri]),"raw_signal_mid":rm,
                  "raw_signal_best_bid":float(bid[ri]),"raw_signal_best_ask":float(ask[ri]),"last_trade_at_signal":last_trade,
                  "executable_entry_timestamp_ns":int(ts[ei]),"executable_entry_quote":entry_quote,
                  "actual_fill_price":actual_fill,"entry_spread_ticks":(ask[ei]-bid[ei])/TICK,
                  "structural_stop_reference":stop,"stop_distance_ticks":abs(actual_fill-stop)/TICK,
                  "frozen_target_prices":targets,"implied_reward_risk":reward_risk,
                  "markouts":paths,"mfe_mae":excursion,"first_touch":touches})
    return event


def _stats(vals: Sequence[float | None]) -> dict[str, Any]:
    a=np.asarray([float(x) for x in vals if x is not None and math.isfinite(float(x))],dtype=float)
    if not len(a):return {"n":0,"mean":None,"median":None,"trimmed_mean":None,"p25":None,"p75":None,"positive_fraction":None,"negative_fraction":None}
    s=np.sort(a);k=int(.1*len(s));trim=s[k:len(s)-k] if len(s)-2*k else s
    return {"n":int(len(a)),"mean":float(np.mean(a)),"median":float(np.median(a)),"trimmed_mean":float(np.mean(trim)),
            "p25":float(np.quantile(a,.25)),"p75":float(np.quantile(a,.75)),
            "positive_fraction":float(np.mean(a>0)),"negative_fraction":float(np.mean(a<0))}


def _group_summary(events: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    result={}
    for h in MARKOUT_MS:
        result[str(h)]={kind:_stats([e.get("markouts",{}).get(str(h),{}).get(kind) for e in events]) for kind in ("raw","quote","actual")}
    return result


def _source_inventory() -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, str]]:
    paths,rows=native._source_catalog(native.DATA_ROOT)
    train=json.loads((baseline.OUTPUT_ROOT/"candidate-tapes"/"train-tape-manifest.json").read_text())
    train_reports={str(r.get("date")):r for r in train.get("reports",[]) if isinstance(r,dict)}
    tape_hashes={}
    for day in TARGET_DATES:
        declared=rows[day]["sha256"]
        tape_path=native._tape_path(day)
        if day in SPRING_DATES:
            manifest=train_reports.get(day)
        else:
            manifest_path=Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928/october-candidate-tapes/tapes")/f"{day}-candidate-tape-manifest.json"
            manifest=json.loads(manifest_path.read_text())
        if not isinstance(manifest,dict) or manifest.get("source_sha256")!=declared:
            raise SatoStudyError(f"tape/source manifest mismatch for {day}")
        if not manifest.get("bbo_path_complete", False):
            raise SatoStudyError(f"candidate tape BBO completeness is not proven for {day}")
        if not tape_path.is_file() or tape_path.stat().st_size<=0:
            raise SatoStudyError(f"candidate tape missing/empty for {day}")
        actual_tape_sha=_sha(tape_path)
        if manifest.get("tape_sha256") and actual_tape_sha!=manifest["tape_sha256"]:
            raise SatoStudyError(f"candidate tape hash mismatch for {day}")
        events,_=native._load_tape(day,tape_path,declared)
        if not len(events):raise SatoStudyError(f"empty canonical tape: {day}")
        tape_hashes[day]=actual_tape_sha
        del events
    return paths,rows,tape_hashes


def _bars_with_premarket(day: str, tape: np.ndarray) -> list[dict[str, Any]]:
    rth_open,rth_close=_rth(day)
    start=rth_open-6*BAR_NS
    # The bar grid is anchored at RTH open, giving six adjacent completed pre-RTH bars ending at the open.
    pre=_bars(day,tape,start_ns=start,end_ns=rth_open,anchor_ns=rth_open)
    rth=_bars(day,tape,start_ns=rth_open,end_ns=rth_close,anchor_ns=rth_open)
    return pre+rth


def _daily_run(day: str, prev: str, tape: np.ndarray, prior: dict[str,Any]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    start,close=_rth(day); bars=_bars_with_premarket(day,tape)
    rth_bars=[b for b in bars if start<=b["start_ns"]<close]
    if not rth_bars:raise SatoStudyError(f"no RTH bars for {day}")
    ix=_trades(tape,start,close); ts=tape["timestamp_ns"][ix].astype(np.int64); px=tape["execution_price"][ix].astype(float)
    sz=tape["execution_size"][ix].astype(np.int64); ag=tape["aggressor"][ix].astype(np.int8)
    events=[]
    for side,level in (("PDH",prior["pdh"]),("PDL",prior["pdl"])):
        ev=_make_candidate(day,side,float(level),tape,rth_bars,bars,prior,ix,ts,px,sz,ag)
        if ev:
            ev["previous_valid_rth_date"]=prev
            _path_event(ev,tape,close)
            events.append(ev)
    for ev in events:
        ev["both_prior_day_sides_swept"]=len(events)==2
        iso=date.fromisoformat(day).isocalendar()
        ev["iso_week"]=f"{iso.year}-W{iso.week:02d}"
    accepted=[dict(e) for e in events if e["no_reclaim_acceptance_candidate"]]
    for ev in accepted:
        ev["acceptance_markouts"]={}
        t=ev.get("acceptance_timestamp_ns")
        if t is None:continue
        breakout_sign=1 if ev["side"]=="PDH" else -1
        q0=_quote_mid(tape,int(t),side="left",end=close)
        for h in ACCEPTANCE_MS:
            q1=_quote_mid(tape,int(t)+h*1_000_000,side="left",end=close)
            ev["acceptance_markouts"][str(h)]=(breakout_sign*(q1[1]-q0[1])/TICK if q0 and q1 else None)
    bars_out=[{"date":day,"period":_period(day),"previous_valid_rth_date":prev,**b} for b in bars]
    return events,bars_out,accepted


def _flatten_components(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    reclaimed=[e for e in events if e["reclaim_only"]]
    qualified=[e for e in reclaimed if e["public_stack_qualified"]]
    nonreclaim=[e for e in events if e["no_reclaim_acceptance_candidate"]]
    comp={}
    for key in ("volume_spike","delta_divergence","absorption"):
        comp[key]={"true":sum(bool(e[key]) for e in reclaimed),"false":sum(not bool(e[key]) for e in reclaimed),
                   "unavailable":sum(e.get(key+"_status")!="AVAILABLE" for e in reclaimed)}
    groups={"ALL_LEVEL_BREACHES":events,"RECLAIMED_SWEEPS":reclaimed,
            "PUBLIC_STACK_QUALIFIED_SWEEPS":qualified,"NO_RECLAIM_ACCEPTANCE_CANDIDATES":nonreclaim}
    analyses={name:{"n":len(rows),"markouts":_group_summary(rows,"markouts"),
                    "speed":dict((s,sum(e.get("reclaim_speed")==s for e in rows)) for s in ("SAME_BAR","NEXT_BAR","THIRD_BAR"))}
              for name,rows in groups.items()}
    period={}
    for p in ("SPRING_2025","OCTOBER_2025"):
        period[p]={}
        for group,rows in groups.items():
            subset=[e for e in rows if e["period"]==p]
            period[p][group]={"n":len(subset),"markouts":_group_summary(subset,"markouts")}
        accepted=[e for e in nonreclaim if e["period"]==p]
        period[p]["NO_RECLAIM_ACCEPTANCE_BREAKOUT_DIRECTION"]={"n":len(accepted),"markouts":{
            str(h):_stats([e.get("acceptance_markouts",{}).get(str(h)) for e in accepted]) for h in ACCEPTANCE_MS}}
    side={}
    for s in ("PDH","PDL"):
        subset=[e for e in reclaimed if e["side"]==s]
        side[s]={"n":len(subset),"qualified_n":sum(e["public_stack_qualified"] for e in subset),
                 "markouts":_group_summary(subset,"markouts")}
        accepted=[e for e in nonreclaim if e["side"]==s]
        side[s]["no_reclaim_acceptance"]={"n":len(accepted),"breakout_direction_markouts":{
            str(h):_stats([e.get("acceptance_markouts",{}).get(str(h)) for e in accepted]) for h in ACCEPTANCE_MS}}
    by_day=defaultdict(list)
    for e in events:by_day[e["date"]].append(e)
    daily=[]
    for d in TARGET_DATES:
        rows=by_day[d]
        daily.append({"date":d,"period":_period(d),"previous_valid_rth_date":rows[0]["previous_valid_rth_date"] if rows else None,
                      "pdh_breach":any(e["side"]=="PDH" for e in rows),"pdl_breach":any(e["side"]=="PDL" for e in rows),
                      "reclaimed_count":sum(e["reclaim_only"] for e in rows),"qualified_count":sum(e["public_stack_qualified"] for e in rows),
                      "events":rows})
    weekly=defaultdict(list)
    for e in events:weekly[e["iso_week"]].append(e)
    weekly_out={w:{"breaches":len(rows),"reclaims":sum(e["reclaim_only"] for e in rows),
                   "qualified_sweeps":sum(e["public_stack_qualified"] for e in rows),
                   "actual_5m_ticks":_stats([e.get("markouts",{}).get("300000",{}).get("actual") for e in rows if e["reclaim_only"]])}
                for w,rows in sorted(weekly.items())}
    return {"groups":analyses,"components":comp,"period":period,"side":side,"daily":daily,"weekly":weekly_out,
            "event_counts":{"all_breaches":len(events),"reclaimed":len(reclaimed),"qualified":len(qualified),"no_reclaim":len(nonreclaim)},
            "orderflow_adds_incremental_information":"insufficient" if len(qualified)<20 else None}


def _terciles(events: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    rows=[e for e in events if e.get("reclaim_only") and e.get(feature) is not None
          and e.get("markouts",{}).get("300000",{}).get("actual") is not None]
    if len(rows)<30:
        return {"feature":feature,"n":len(rows),"status":"INSUFFICIENT_BUCKET_SAMPLE","terciles":[]}
    values=np.asarray([float(e[feature]) for e in rows]);cuts=np.quantile(values,[1/3,2/3])
    labels=np.digitize(values,cuts,right=True);out=[]
    for ix,name in enumerate(("LOW","MIDDLE","HIGH")):
        group=[e for e,j in zip(rows,labels) if int(j)==ix]
        out.append({"tercile":name,"n":len(group),"feature_min":min(float(e[feature]) for e in group),
                    "feature_max":max(float(e[feature]) for e in group),
                    "actual_5m_reversal_ticks":_stats([e["markouts"]["300000"]["actual"] for e in group])})
    means=[g["actual_5m_reversal_ticks"]["mean"] for g in out]
    if means[0]<means[1]<means[2]:shape="MONOTONIC_POSITIVE"
    elif means[0]>means[1]>means[2]:shape="MONOTONIC_NEGATIVE"
    elif means[1]>max(means[0],means[2]):shape="INVERTED_U"
    elif means[1]<min(means[0],means[2]):shape="U_SHAPED"
    else:shape="NO_CLEAR_SHAPE"
    return {"feature":feature,"n":len(rows),"status":"AVAILABLE","cutpoints":cuts.tolist(),"terciles":out,"shape":shape}


def _component_effects(events: Sequence[Mapping[str, Any]], component: str) -> dict[str, Any]:
    reclaimed=[e for e in events if e["reclaim_only"]]
    status=component+"_status"
    return {"component":component,"convention":"frozen public V1 operational proxy; no threshold search",
            "true":{"n":sum(bool(e[component]) for e in reclaimed),"markouts":_group_summary([e for e in reclaimed if e[component]],"markouts")},
            "false":{"n":sum(not bool(e[component]) for e in reclaimed if e.get(status)=="AVAILABLE"),
                      "markouts":_group_summary([e for e in reclaimed if not bool(e[component]) and e.get(status)=="AVAILABLE"],"markouts")},
            "unavailable_n":sum(e.get(status)!="AVAILABLE" for e in reclaimed)}


def _price_only_control(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    reclaimed=[e for e in events if e["reclaim_only"]]
    supported=[e for e in reclaimed if e["public_stack_qualified"]]
    controls=[e for e in reclaimed if not e["public_stack_qualified"]]
    # Preregistered coarse matching: exact period, side, breach-time bucket, then nearest normalized
    # extension/speed/range distance. No volume, Delta, CVD or depth feature enters matching.
    matches=[]; used=set()
    scales=(4.0,1.0,40.0)
    for e in supported:
        candidates=[c for c in controls if c["period"]==e["period"] and c["side"]==e["side"] and c["breach_time_bucket"]==e["breach_time_bucket"] and c["date"]!=e["date"] and c["date"] not in used]
        if not candidates:continue
        def distance(c):
            v=(abs(c["sweep_extension_ticks"]-e["sweep_extension_ticks"])/scales[0],
               abs((c["reclaim_bar_rank"] or 0)-(e["reclaim_bar_rank"] or 0))/scales[1],
               abs(c["prior_day_range_ticks"]-e["prior_day_range_ticks"])/scales[2])
            return sum(x*x for x in v),c["date"]
        c=min(candidates,key=distance);used.add(c["date"])
        matches.append({"qualified_date":e["date"],"control_date":c["date"],"side":e["side"],
                        "distance":distance(c)[0],"qualified_actual_5m_ticks":e.get("markouts",{}).get("300000",{}).get("actual"),
                        "control_actual_5m_ticks":c.get("markouts",{}).get("300000",{}).get("actual")})
    return {"method":"deterministic nearest price-shape match; same period/side/breach bucket; matching uses only extension, reclaim speed, prior-day range",
            "matches":matches,"matched_n":len(matches),"qualified_actual_5m":_stats([m["qualified_actual_5m_ticks"] for m in matches]),
            "control_actual_5m":_stats([m["control_actual_5m_ticks"] for m in matches])}


def _resampling(events: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    rng=np.random.default_rng(20251006)
    result={}
    def value(e):
        if e.get("reclaim_only"):
            return e.get("markouts",{}).get("300000",{}).get("actual")
        return e.get("acceptance_markouts",{}).get("300000")
    def compare(label, rows, predicate):
        strata=defaultdict(list)
        for e in rows:
            v=value(e)
            if v is None:continue
            strata[(e["period"],e["side"],e["breach_time_bucket"])].append((float(v),bool(predicate(e))))
        observed=[]; permuted=[]
        for group in strata.values():
            x=np.asarray([v for v,g in group if g]);y=np.asarray([v for v,g in group if not g])
            if len(x) and len(y):
                observed.append((len(x),float(np.mean(x)-np.mean(y)),np.asarray([v for v,_ in group],float),len(x)))
        nleft=sum(x[0] for x in observed); nstrata=len(observed)
        if nleft<2 or sum(len(g) for g in strata.values())-nleft<2 or nstrata==0:
            return {"n_qualified":nleft,"strata_with_both_groups":nstrata,"status":"INSUFFICIENT"}
        observed_delta=float(np.average([x[1] for x in observed],weights=[x[0] for x in observed]))
        hits=0
        for _ in range(2000):
            deltas=[];weights=[]
            for _,_,pool,n in observed:
                shuffled=rng.permutation(pool);deltas.append(float(np.mean(shuffled[:n])-np.mean(shuffled[n:])));weights.append(n)
            draw=float(np.average(deltas,weights=weights))
            hits += abs(draw)>=abs(observed_delta)
        return {"n_qualified":nleft,"strata_with_both_groups":nstrata,"mean_delta_qualified_minus_other":observed_delta,
                "permutation_p_two_sided":(hits+1)/2001,"seed":20251006,"permutations":2000,
                "strata":[{"period":k[0],"side":k[1],"breach_time_bucket":k[2],"n":len(v)} for k,v in strata.items()]}
    reclaimed=[e for e in events if e["reclaim_only"] and e.get("path_status")=="AVAILABLE"]
    acceptance=[e for e in events if e["no_reclaim_acceptance_candidate"] and e.get("acceptance_markouts",{}).get("300000") is not None]
    result["reclaim_vs_no_reclaim_acceptance"]=compare("reclaim_vs_no_reclaim_acceptance",[e for e in reclaimed+acceptance],lambda e:e["reclaim_only"])
    result["public_stack_vs_reclaim_only"]=compare("public_stack_vs_reclaim_only",reclaimed,lambda e:e["public_stack_qualified"])
    for name in ("volume_spike","delta_divergence","absorption"):
        result[name+"_relationship"]=compare(name+"_relationship",reclaimed,lambda e,n=name:bool(e[n]))
    return result


def _lodo_lowo(events: Sequence[Mapping[str,Any]]) -> tuple[dict[str,Any],dict[str,Any]]:
    def effect(rows):
        a=[e.get("markouts",{}).get("300000",{}).get("actual") for e in rows if e["reclaim_only"] and e["public_stack_qualified"]]
        b=[e.get("markouts",{}).get("300000",{}).get("actual") for e in rows if e["reclaim_only"] and not e["public_stack_qualified"]]
        aa=[x for x in a if x is not None];bb=[x for x in b if x is not None]
        return float(np.mean(aa)-np.mean(bb)) if aa and bb else None
    all_eff=effect(events);dates=sorted({e["date"] for e in events});weeks=sorted({e["iso_week"] for e in events})
    def omit(key,values):
        out=[]
        for v in values:
            rows=[e for e in events if e[key]!=v];out.append({"omitted":v,"effect":effect(rows)})
        vals=[r["effect"] for r in out if r["effect"] is not None]
        return {"full_effect":all_eff,"omissions":out,"sign_stability":float(np.mean(np.sign(vals)==np.sign(all_eff))) if vals and all_eff is not None else None,
                "median_effect":float(np.median(vals)) if vals else None,"minimum":float(min(vals)) if vals else None,"maximum":float(max(vals)) if vals else None,
                "best_omitted":max(out,key=lambda x:x["effect"] if x["effect"] is not None else -math.inf) if vals else None,
                "worst_omitted":min(out,key=lambda x:x["effect"] if x["effect"] is not None else math.inf) if vals else None}
    return omit("date",dates),omit("iso_week",weeks)


def _event_groups(events: Sequence[Mapping[str,Any]]) -> dict[str,list[dict[str,Any]]]:
    return {"all":list(events),"reclaim_only":[dict(e) for e in events if e["reclaim_only"]],
            "qualified":[dict(e) for e in events if e["reclaim_only"] and e["public_stack_qualified"]],
            "no_reclaim":[dict(e) for e in events if e["no_reclaim_acceptance_candidate"]],
            "one_signal_side_per_day":[dict(e) for e in events]}


def run(*, resume: bool = True, smoke: bool = False) -> dict[str,Any]:
    started=time.monotonic(); out=OUT_ROOT; out.mkdir(parents=True,exist_ok=True)
    paths,source_rows,tape_hashes=_source_inventory()
    prior_map=previous_valid_rth_map(TARGET_DATES,SOURCE_DATES)
    if smoke: days=(SPRING_DATES[0],)
    else: days=TARGET_DATES
    levels={d:_prior_levels(d,prior_map[d],paths,source_rows) for d in days}
    level_manifest={"eligible_dates":list(days),"excluded_dates":[],"mapping":prior_map,"levels":levels}
    _write_json(out/"prior-day-levels.json",level_manifest)
    coverage={"dataset":"GLBX.MDP3","schema":"mbp-10","instrument":"ES","target_dates":list(TARGET_DATES),
              "eligible_dates":list(days),"source_dates":list(SOURCE_DATES),"previous_valid_rth_mapping":prior_map,
              "sessions":[{"current_session":d,"previous_valid_rth_date":prior_map[d],"previous_rth_coverage_complete":True,
                           "source_file":str(paths[prior_map[d]]),"source_hash":source_rows[prior_map[d]]["sha256"],
                           "source_request_start":source_rows[prior_map[d]]["start"],"source_request_end":source_rows[prior_map[d]]["end"],
                           "actual_prior_rth_trade_count":levels[d]["trade_count"],"prior_pdh":levels[d]["pdh"],"prior_pdl":levels[d]["pdl"]} for d in days]}
    _write_json(out/"source-coverage.json",coverage)
    checkpoint_dir=out/"checkpoints";checkpoint_dir.mkdir(exist_ok=True)
    all_events=[];all_bars=[];acceptance=[];cp_hashes={}
    for day in days:
        current_tape=None
        cp=checkpoint_dir/f"{day}.json.gz";source_sha=source_rows[day]["sha256"]
        cp_key=_checkpoint_key(day,source_sha,levels[day]["source_sha256"])
        loaded_checkpoint = False
        if resume and cp.exists():
            try:
                with gzip.open(cp,"rt",encoding="utf-8") as f: payload=json.load(f)
                if payload.get("checkpoint_key")==cp_key:
                    ev=payload["events"]; bars=payload["bars"]; acc=payload["acceptance"]
                    loaded_checkpoint = True
                else:
                    current_tape,_=native._load_tape(day,native._tape_path(day),str(source_rows[day]["sha256"]))
                    ev,bars,acc=_daily_run(day,prior_map[day],current_tape,levels[day])
            except Exception:
                current_tape,_=native._load_tape(day,native._tape_path(day),str(source_rows[day]["sha256"]))
                ev,bars,acc=_daily_run(day,prior_map[day],current_tape,levels[day])
        else:
            current_tape,_=native._load_tape(day,native._tape_path(day),str(source_rows[day]["sha256"]))
            ev,bars,acc=_daily_run(day,prior_map[day],current_tape,levels[day])
        if current_tape is not None:
            del current_tape
        if not loaded_checkpoint:
            temp=cp.with_name(f".{cp.name}.{os.getpid()}.tmp")
            with temp.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                    gz.write(json.dumps({"checkpoint_key":cp_key,"events":ev,"bars":bars,"acceptance":acc},sort_keys=True,default=_json,allow_nan=False).encode())
            os.replace(temp,cp)
        all_events.extend(ev);all_bars.extend(bars);acceptance.extend(acc);cp_hashes[day]=_sha(cp)
        if smoke:break
    if not smoke:days=TARGET_DATES
    # Group and factor outputs are strictly descriptive; no thresholds are selected from outcomes.
    agg=_flatten_components(all_events)
    price_control=_price_only_control(all_events)
    lodo,lowo=_lodo_lowo(all_events)
    perms=_resampling(all_events)
    acceptance_out=[]
    for e in acceptance:
        x=dict(e)
        acceptance_out.append(x)
    groups=_event_groups(all_events)
    first_by_day={}
    for e in sorted(all_events,key=lambda x:x["breach_timestamp_ns"]):
        first_by_day.setdefault(e["date"],e)
    first_sweep_events=list(first_by_day.values())
    artifacts={
        "five-minute-bars.jsonl.gz":all_bars,"breach-events.jsonl.gz":all_events,
        "reclaim-events.jsonl.gz":[e for e in all_events if e["reclaim_only"]],
        "acceptance-controls.jsonl.gz":acceptance_out,
        "raw-markouts.json":{g:{str(h):_stats([e.get("markouts",{}).get(str(h),{}).get("raw") for e in es]) for h in MARKOUT_MS} for g,es in groups.items()},
        "executable-markouts.json":{g:{str(h):_stats([e.get("markouts",{}).get(str(h),{}).get("quote") for e in es]) for h in MARKOUT_MS} for g,es in groups.items()},
        "actual-fill-markouts.json":{g:{str(h):_stats([e.get("markouts",{}).get(str(h),{}).get("actual") for e in es]) for h in MARKOUT_MS} for g,es in groups.items()},
        "mfe-mae.json":{g:{str(h):{"raw_mfe":_stats([e.get("mfe_mae",{}).get(str(h),{}).get("raw",{}).get("mfe") for e in es]),
                                        "raw_mae":_stats([e.get("mfe_mae",{}).get(str(h),{}).get("raw",{}).get("mae") for e in es]),
                                        "actual_mfe":_stats([e.get("mfe_mae",{}).get(str(h),{}).get("actual",{}).get("mfe") for e in es]),
                                        "actual_mae":_stats([e.get("mfe_mae",{}).get(str(h),{}).get("actual",{}).get("mae") for e in es])} for h in EXCURSION_MS} for g,es in groups.items()},
        "vwap-first-touch.json":{e["date"]+"/"+e["side"]:e.get("first_touch",{}).get("session_vwap") for e in all_events if e["reclaim_only"]},
        "prior-day-midpoint-first-touch.json":{e["date"]+"/"+e["side"]:e.get("first_touch",{}).get("prior_midpoint") for e in all_events if e["reclaim_only"]},
        "opposite-prior-extreme-first-touch.json":{e["date"]+"/"+e["side"]:e.get("first_touch",{}).get("opposite_prior_extreme") for e in all_events if e["reclaim_only"]},
        "volume-spike-analysis.json":_component_effects(all_events,"volume_spike"),
        "delta-divergence-analysis.json":_component_effects(all_events,"delta_divergence"),
        "absorption-analysis.json":_component_effects(all_events,"absorption"),
        "public-stack-analysis.json":{"summary":agg["groups"]["PUBLIC_STACK_QUALIFIED_SWEEPS"],
            "support_component_count":{"1":sum(e.get("total_public_stack_count")==1 for e in all_events),
                "2":sum(e.get("total_public_stack_count")==2 for e in all_events),
                "3":sum(e.get("total_public_stack_count")==3 for e in all_events),
                "4":sum(e.get("total_public_stack_count")==4 for e in all_events)},
            "qualified_n":agg["event_counts"]["qualified"],"reclaim_only_n":agg["event_counts"]["reclaimed"],
            "unavailable_component_counts":agg["components"]},
        "acceptance-markouts.json":{"breakout_direction":{str(h):_stats([e.get("acceptance_markouts",{}).get(str(h)) for e in acceptance_out]) for h in ACCEPTANCE_MS},
            "by_period":{period:{str(h):_stats([e.get("acceptance_markouts",{}).get(str(h)) for e in acceptance_out if e["period"]==period]) for h in ACCEPTANCE_MS} for period in ("SPRING_2025","OCTOBER_2025")}},
        "execution-hurdle.json":{"entry_delay_ns":ENTRY_DELAY_NS,"one_adverse_tick_entry":True,
            "qualified_actual_5m_ticks":_stats([e.get("markouts",{}).get("300000",{}).get("actual") for e in all_events if e["public_stack_qualified"]]),
            "reclaim_actual_5m_ticks":_stats([e.get("markouts",{}).get("300000",{}).get("actual") for e in all_events if e["reclaim_only"]])},
        "reclaim-speed.json":{s:{"n":sum(e.get("reclaim_speed")==s for e in all_events),"actual_5m":_stats([e.get("markouts",{}).get("300000",{}).get("actual") for e in all_events if e.get("reclaim_speed")==s])} for s in ("SAME_BAR","NEXT_BAR","THIRD_BAR")},
        "sweep-extension.json":{"continuous_ticks":_stats([e["sweep_extension_ticks"] for e in all_events]),
            "terciles":_terciles(all_events,"sweep_extension_ticks"),
            "events":[{"date":e["date"],"side":e["side"],"extension_ticks":e["sweep_extension_ticks"],"actual_5m":e.get("markouts",{}).get("300000",{}).get("actual")} for e in all_events]},
        "prior-day-range-analysis.json":{"continuous_ticks":_stats([e["prior_day_range_ticks"] for e in all_events]),
            "terciles":_terciles(all_events,"prior_day_range_ticks"),
            "events":[{"date":e["date"],"range_ticks":e["prior_day_range_ticks"],"actual_5m":e.get("markouts",{}).get("300000",{}).get("actual")} for e in all_events]},
        "price-only-reclaim-control.json":price_control,"period-results.json":agg["period"],"side-results.json":agg["side"],
        "time-results.json":{b:{"n":sum(e["breach_time_bucket"]==b for e in all_events),"actual_5m":_stats([e.get("markouts",{}).get("300000",{}).get("actual") for e in all_events if e["breach_time_bucket"]==b])} for b in ("09:30-09:40","09:40-09:50","09:50-10:00")},
        "first-sweep-sensitivity.json":{"primary":agg["groups"],"first_sweep_of_session_only":{"n":len(first_sweep_events),"markouts":_group_summary(first_sweep_events,"markouts")}},
        "daily-results.json":agg["daily"],"weekly-results.json":agg["weekly"],"lodo-results.json":lodo,"lowo-results.json":lowo,
        "permutation-results.json":perms}
    for name,payload in artifacts.items():
        if name.endswith(".jsonl.gz"):_gzip_jsonl(out/name,payload)
        else:_write_json(out/name,payload)
    _gzip_jsonl(out/"reclaim-events.jsonl.gz",all_events)
    summary={"study_id":RUN_ID,"study_version":STUDY_VERSION,"status":"SMOKE_COMPLETE" if smoke else "COMPLETE",
             "period_roles":CONFIG["period_roles"],"target_dates":list(days),"source_scope":list(SOURCE_DATES),
             "event_counts":agg["event_counts"],"qualified_n":agg["event_counts"]["qualified"],
             "component_unavailable_counts":{k:v["unavailable"] for k,v in agg["components"].items()},
             "orderflow_adds_incremental_information":("insufficient" if agg["event_counts"]["qualified"]<20 else "requires_review"),
             "primary_decision":("INSUFFICIENT_SAMPLE" if agg["event_counts"]["reclaimed"]<20 or agg["event_counts"]["qualified"]<20 else "REQUIRES_REVIEW"),
             "candidate_hypothesis":"NONE",
             "next_step":"REQUIRE_ADDITIONAL_PREDECLARED_DATA" if agg["event_counts"]["reclaimed"]<20 or agg["event_counts"]["qualified"]<20 else "MANUAL_REVIEW_REQUIRED",
             "no_optimization":True,"no_strategy_pnl":True,"oos_accessed":False,"data_downloaded":False,
             "runtime_seconds":time.monotonic()-started}
    _write_json(out/"summary.json",summary)
    _write_json(out/"study-config.json",CONFIG)
    actual_5m=artifacts["actual-fill-markouts.json"]["reclaim_only"]["300000"]
    quote_5m=artifacts["executable-markouts.json"]["reclaim_only"]["300000"]
    qualified_actual_5m=artifacts["execution-hurdle.json"]["qualified_actual_5m_ticks"]
    report=(f"# {RUN_ID}\n\nStatus: {summary['status']}\n\nEligible sessions: {len(days)} ({len(SPRING_DATES)} Spring + {len(OCTOBER_DATES)} October requested).\n\n"
            f"Breach/reclaim/qualified/no-reclaim counts: {agg['event_counts']}\n\n"
            f"Reclaim-only 5m executable quote markout (ticks): {quote_5m}\n\n"
            f"Reclaim-only 5m conservative actual-fill markout (ticks): {actual_5m}\n\n"
            f"Qualified-stack N={agg['event_counts']['qualified']}; actual-fill 5m: {qualified_actual_5m}\n\n"
            f"Primary decision: {summary['primary_decision']}; candidate hypothesis: NONE.\n\n"
            "Public-model replication only, not Sato's private model. Spring and October are previously studied DEV periods, not OOS. No optimization or strategy PnL was performed. No private levels, MBO, MES, NQ, 2026, or downloads.\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    excluded_hashes={"artifact-hashes.json","run-manifest.json"}
    artifact_hashes={p.name:_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in excluded_hashes}
    _write_json(out/"artifact-hashes.json",artifact_hashes)
    manifest={"study_id":RUN_ID,"study_version":STUDY_VERSION,"status":summary["status"],"config_sha256":CONFIG_SHA256,
              "source_manifest_sha256":_sha(native.DATA_ROOT/baseline.MANIFEST_NAME),"source_sha256_by_date":{d:source_rows[d]["sha256"] for d in days},
              "candidate_tape_sha256_by_date":{d:tape_hashes[d] for d in days},
              "checkpoint_sha256_by_date":cp_hashes,"artifact_sha256":artifact_hashes,"hash_exclusions":sorted(excluded_hashes),"eligible_dates":list(days),
              "validation_is_dev_not_oos":True,"no_2026_oos_access":True}
    _write_json(out/"run-manifest.json",manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="run the frozen study")
    parser.add_argument("--smoke",action="store_true",help="process first Spring target only")
    parser.add_argument("--no-resume",action="store_true",help="ignore compatible date checkpoints")
    args=parser.parse_args(argv)
    if not args.run:parser.error("--run is required")
    result=run(resume=not args.no_resume,smoke=args.smoke)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
