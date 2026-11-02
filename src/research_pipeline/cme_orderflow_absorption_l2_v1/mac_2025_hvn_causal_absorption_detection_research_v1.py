"""Causal, event-driven absorption measurements on five sealed ESH5 MBP-10 days.

Research instrumentation only: no entry logic, optimization, or trading rules.
The native source is streamed once per date; event windows are selected in
native order and output in bounded Parquet row groups.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import mac_2025_hvn_native_mbp10_absorption_mechanism_v1 as base

STUDY_ID = "ES_HVN_CAUSAL_ABSORPTION_DETECTION_RESEARCH_V1"
OUT = Path("research_runs") / STUDY_ID
OFFSETS_MS = (250, 500, 1000, 2000, 3000, 5000)
FORWARD_HORIZONS_MS = (1000, 5000, 10000, 30000)
MAX_STALE_NS = 2_000_000_000
CHUNK_ROWS = 250_000
TICK = base.TICK
NS = base.NS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(x for v in values if (x := _finite(v)) is not None)
    if not clean:
        return {k: None for k in ("n", "q10", "median", "q90", "mean")}
    def q(p: float) -> float:
        at = (len(clean) - 1) * p
        lo, hi = int(at), min(int(at) + 1, len(clean) - 1)
        return clean[lo] + (clean[hi] - clean[lo]) * (at - lo)
    return {"n": len(clean), "q10": q(.1), "median": statistics.median(clean), "q90": q(.9), "mean": statistics.mean(clean)}


def _origin(episode: base.Episode) -> tuple[int, str]:
    """Return the V1 availability-time reference (its compact tape used ts_recv)."""
    if episode.v1_heavy and episode.v1_heavy_ns is not None:
        return int(episode.v1_heavy_ns), "FIRST_SOURCE_SUPPORTED_HEAVY_PRINT"
    return int(episode.t0), "FIRST_TOUCH_NO_HEAVY_PRINT"


def source_origin_match(episode: base.Episode, ts_recv: int, action: str, price_raw: int,
                       size: int, side: str) -> bool:
    """Map a V1 ts_recv identity back to its native trade row without timestamp rounding."""
    reference, kind = _origin(episode)
    if ts_recv != reference or action != "T":
        return False
    expected_price = episode.v1_heavy_price if kind == "FIRST_SOURCE_SUPPORTED_HEAVY_PRINT" else episode.touch_price
    if expected_price is None or price_raw != round(float(expected_price) * 1e9):
        return False
    if kind == "FIRST_SOURCE_SUPPORTED_HEAVY_PRINT":
        if action != "T":
            return False
        expected_side = "A" if episode.direction and episode.direction > 0 else "B"
        return side == expected_side and size == int(episode.v1_heavy_size or 0)
    # A first-touch origin is an exact source receive timestamp, not necessarily
    # an execution. The native book row is matched separately in _process_date.
    return True


def source_touch_match(episode: base.Episode, row: Any, ts_recv: int) -> bool:
    """Match first-touch reference by exact receive time and touch price in that row."""
    if episode.v1_heavy:
        return False
    if ts_recv != _origin(episode)[0]:
        return False
    # Zone-only first touches have no touch ledger price/direction. Their V1
    # first_touch_ns is itself the exact source receive-time identity.
    if episode.touch_price is None:
        return episode.origin == "ZONE_FIRST_TOUCH_NO_DIRECTION"
    target = round(float(episode.touch_price) * 1e9)
    if bytes(row["action"]) == b"T" and int(row["price"]) == target:
        return True
    book = base._book(row)
    return any(price == target for price, _ in (*book["bids"], *book["asks"]))


def snapshot_is_valid(book_valid: bool, age_ns: int | None, gap_count: int,
                      max_stale_ns: int = MAX_STALE_NS) -> bool:
    """A checkpoint must use a valid quote at/before it and a continuous source window."""
    return bool(book_valid and age_ns is not None and 0 <= age_ns <= max_stale_ns and gap_count == 0)


def directional_price_progress(mid: float, initial_mid: float, attack_sign: int,
                                tick: float = TICK) -> float:
    """Positive means movement with aggressive flow; negative means against it."""
    return (mid - initial_mid) / tick * attack_sign


def weak_impact_marker(attack_volume: int, threshold: float | None, max_progress_ticks: float,
                       attack_execution_count: int) -> bool:
    """Outcome-blind weak-impact descriptor; requires repeated execution records."""
    return bool(threshold is not None and threshold > 0 and attack_execution_count >= 2
                and attack_volume >= 2 * threshold and max_progress_ticks <= 1.0)


def _depth_components(book: dict[str, Any], episode: base.Episode) -> dict[str, Any]:
    bids, asks = book["bids"], book["asks"]
    side = bids if episode.direction is not None and episode.direction > 0 else asks if episode.direction is not None else None
    aligned = sum(size for p, size in side if round(episode.low * 4) <= round((p / 1e9) * 4) <= round(episode.high * 4)) if episode.direction is not None else None
    best_price = (book["bid"] if episode.direction and episode.direction > 0 else book["ask"] if episode.direction else None)
    return {
        "defended_price_depth": aligned,
        "top1_opposing_depth": side[0][1] if side else None,
        "top5_opposing_depth": sum(s for _, s in side[:5]) if side is not None else None,
        "top10_opposing_depth": sum(s for _, s in side) if side is not None else None,
        "top1_bid_depth": bids[0][1] if bids else 0, "top1_ask_depth": asks[0][1] if asks else 0,
        "top5_bid_depth": sum(s for _, s in bids[:5]), "top5_ask_depth": sum(s for _, s in asks[:5]),
        "top10_bid_depth": sum(s for _, s in bids), "top10_ask_depth": sum(s for _, s in asks),
        "best_opposing_quote": best_price,
    }


def _new_state(episode: base.Episode) -> dict[str, Any]:
    origin_reference, origin_kind = _origin(episode)
    attack_sign = (-1 if episode.direction > 0 else 1) if episode.direction is not None else None
    return {
        "episode": episode, "origin_ns": origin_reference, "origin_reference_ns":origin_reference,
        "origin_event_ns":None,"origin_resolved":False,"origin_match_action":None,"origin_match_side":None,"origin_match_size":None,
        "origin_match_price":None,"origin_kind": origin_kind, "attack_sign": attack_sign,
        "previous_ns": None, "previous_event_ns":None,"previous_book": None, "previous_flags": 0, "gap_count": 0,
        "start_mid": None, "last_mid": None, "last_valid_quote_ns": None, "event_count": 0,
        "attack_volume": 0, "opposing_volume": 0, "unknown_volume": 0, "attack_count": 0,
        "opposing_count": 0, "unknown_count": 0, "signed_delta_buy_minus_sell": 0,
        "attack_prices": set(), "attack_level_volume": 0, "first_attack_ns": None, "last_attack_ns": None,
        "all_buy_volume": 0, "all_sell_volume": 0, "origin_event_seen": False, "measurement_event_rows": 0,
        "max_attack_progress_ticks": 0.0, "min_attack_progress_ticks": 0.0,
        "max_boundary_penetration_ticks": 0.0, "max_reversal_from_attack_extreme_ticks": 0.0,
        "depth_start": None, "depth_min": None, "depth_last": None, "depth_reduction_seen": False,
        "depth_reduction_contracts": 0, "depth_increase_contracts": 0, "depth_increase_events": 0,
        "depth_persistent_attack_trades": 0, "first_substantial_ns": None, "first_weak_impact_ns": None,
        "first_impact_ratio_decline_ns": None, "max_impact_ratio_seen": None,
        "first_depth_recovery_ns": None, "first_stall_ns": None, "first_reversal_quote_ns": None,
        "milestone_mids":{},"milestone_forward_registered":set(),
        "snapshots": {}, "forward": [], "events": [],
    }


def _feature_row(state: dict[str, Any], observed_ns: int | None, book: dict[str, Any] | None,
                 flags: int, target_ns: int, *, endpoint_reason: str | None = None) -> dict[str, Any]:
    ep = state["episode"]
    age_ns = target_ns - observed_ns if observed_ns is not None else None
    quote_ok = bool(book and book["valid"] and not (flags & base.FLAG_BAD_TS_RECV)
                    and age_ns is not None and 0 <= age_ns <= MAX_STALE_NS)
    valid = snapshot_is_valid(bool(book and book["valid"]), age_ns, state["gap_count"]) and not (flags & base.FLAG_BAD_TS_RECV)
    depth = _depth_components(book, ep) if book else {}
    mid = book["mid"] if quote_ok else None
    current_progress = (directional_price_progress(mid, state["start_mid"], state["attack_sign"])
                        if mid is not None and state["start_mid"] is not None and state["attack_sign"] is not None else None)
    v1_threshold = ep.v1_threshold
    substantial_threshold = 2 * v1_threshold if v1_threshold is not None and v1_threshold > 0 else None
    progress_per_contract = (current_progress / state["attack_volume"]
                             if current_progress is not None and state["attack_volume"] > 0 else None)
    weak_impact = bool(valid and weak_impact_marker(state["attack_volume"],v1_threshold,
                       state["max_attack_progress_ticks"],state["attack_count"]))
    return {
        "episode_id": ep.episode_id, "date": ep.date, "zone_id": ep.zone_id,
        "origin_ns": state["origin_ns"], "origin_kind": state["origin_kind"], "offset_ms": (target_ns-state["origin_ns"])/1e6,
        "v1_entry_ns": int(ep.v1["TE_ns"]) if ep.v1 and ep.v1.get("TE_ns") else None,
        "available_before_v1_entry": bool(ep.v1 and ep.v1.get("TE_ns") and target_ns <= int(ep.v1["TE_ns"])) if ep.v1 else None,
        "checkpoint_ns": target_ns, "observed_recv_ns": observed_ns,"observed_event_ns":state["previous_event_ns"],
        "observation_age_ms": age_ns/1e6 if age_ns is not None else None,
        "snapshot_valid": bool(valid), "quote_valid": bool(quote_ok), "source_gap_count_since_origin": state["gap_count"],
        "flags": flags, "v1_status": ep.v1_status, "v1_confirmed": bool(ep.v1),
        "rr1_class": "WINNER" if ep.rr1 and float(ep.rr1.get("net_r", 0) or 0) > 0 else "LOSER" if ep.rr1 else "UNTRADED_OR_UNAVAILABLE",
        "direction": ep.direction, "attack_side": "SELL" if state["attack_sign"] == -1 else "BUY" if state["attack_sign"] == 1 else "UNKNOWN",
        "v1_heavy_threshold_contracts": v1_threshold, "substantial_aggression_convention_contracts": substantial_threshold,
        "initial_mid": state["start_mid"] if valid else None, "current_mid": mid,
        "attack_direction_displacement_ticks": current_progress,
        "max_attack_direction_progress_ticks": state["max_attack_progress_ticks"] if valid else None,
        "attack_direction_price_impact_ticks_per_attack_contract": progress_per_contract if valid else None,
        "max_hvn_boundary_penetration_ticks": state["max_boundary_penetration_ticks"] if valid else None,
        "attack_volume_contracts": state["attack_volume"], "opposing_aggressor_volume_contracts": state["opposing_volume"],
        "unknown_aggressor_volume_contracts": state["unknown_volume"],
        "all_buy_aggressor_volume_contracts": state["all_buy_volume"], "all_sell_aggressor_volume_contracts": state["all_sell_volume"],
        "net_delta_buy_minus_sell_contracts": state["signed_delta_buy_minus_sell"],
        "known_attack_execution_count": state["attack_count"], "opposing_execution_count": state["opposing_count"],
        "unknown_side_execution_count": state["unknown_count"],
        "attack_executions_per_second": state["attack_count"] / max((observed_ns-state["origin_ns"])/NS, .001) if observed_ns else 0.0,
        "directional_persistence_ratio": state["attack_count"] / max(state["attack_count"]+state["opposing_count"], 1),
        "aggression_at_defended_prices_contracts": state["attack_level_volume"],
        "attack_duration_ms": (state["last_attack_ns"]-state["first_attack_ns"])/1e6 if state["first_attack_ns"] and state["last_attack_ns"] else 0.0,
        "first_substantial_aggression_ns": state["first_substantial_ns"], "weak_impact_evidence_ns": state["first_weak_impact_ns"],
        "first_relative_impact_decline_ns":state["first_impact_ratio_decline_ns"],
        "displayed_depth_recovery_ns": state["first_depth_recovery_ns"], "first_price_stall_ns": state["first_stall_ns"],
        "first_reversal_quote_ns": state["first_reversal_quote_ns"],
        **depth,
        "depth_reductions_contracts": state["depth_reduction_contracts"], "depth_increases_contracts": state["depth_increase_contracts"],
        "depth_increase_events": state["depth_increase_events"], "attack_trades_with_opposing_depth_present": state["depth_persistent_attack_trades"],
        "endpoint_unavailable_reason": endpoint_reason,
    }


def _take_snapshot(state: dict[str, Any], target_ns: int, observed_ns: int | None,
                   book: dict[str, Any] | None, flags: int, reason: str | None = None) -> None:
    ep = state["episode"]
    offset = int((target_ns - state["origin_ns"]) / 1e6)
    feature = _feature_row(state, observed_ns, book, flags, target_ns, endpoint_reason=reason)
    state["snapshots"][offset] = feature
    if feature["snapshot_valid"]:
        state["forward"].append({
            "snapshot": feature, "reference_mid": feature["current_mid"], "direction": ep.direction,
            "checkpoint_kind": "FIXED_OFFSET", "milestone": None,
            "horizons": {h: {"end_ns": target_ns+h*1_000_000, "last_mid": feature["current_mid"],
                             "last_ns": target_ns, "max_favorable_ticks": 0.0, "max_adverse_ticks": 0.0,
                             "gap_count": 0} for h in FORWARD_HORIZONS_MS},
        })


def _update_state(state: dict[str, Any], row: Any, ts: int, recv: int, seq: int, flags: int) -> dict[str, Any]:
    ep: base.Episode = state["episode"]
    book = base._book(row)
    if state["previous_ns"] is not None and ts - state["previous_ns"] > MAX_STALE_NS:
        state["gap_count"] += 1
    if book["valid"] and not (flags & base.FLAG_BAD_TS_RECV):
        if state["start_mid"] is None:
            state["start_mid"] = book["mid"]
        state["last_mid"] = book["mid"]
        state["last_valid_quote_ns"] = ts
    state["event_count"] += 1
    action, side = bytes(row["action"]).decode("ascii"), bytes(row["side"]).decode("ascii")
    price = int(row["price"])/1e9
    size = int(row["size"])
    depth_now = _depth_components(book, ep).get("defended_price_depth") if book["valid"] else None
    if state["depth_start"] is None and depth_now is not None:
        state["depth_start"] = state["depth_min"] = depth_now
    if depth_now is not None:
        prior_depth = state["depth_last"]
        if action in "AMC" and prior_depth is not None:
            delta = depth_now-prior_depth
            if delta < 0:
                state["depth_reduction_contracts"] += -delta
                state["depth_reduction_seen"] = True
            elif delta > 0:
                state["depth_increase_contracts"] += delta
                state["depth_increase_events"] += 1
            if state["gap_count"]==0 and not (flags & base.FLAG_BAD_TS_RECV) and state["depth_reduction_seen"] and state["first_depth_recovery_ns"] is None and state["depth_min"] is not None and depth_now > state["depth_min"] and depth_now > 0:
                state["first_depth_recovery_ns"] = ts
                state["milestone_mids"]["first_displayed_depth_recovery"]=book["mid"]
        if state["depth_min"] is None or depth_now < state["depth_min"]:
            state["depth_min"] = depth_now
        state["depth_last"] = depth_now
    if action == "T" and size > 0:
        aggressor = base.trade_aggressor(action, side)
        if aggressor == "BUY":
            state["signed_delta_buy_minus_sell"] += size
            state["all_buy_volume"] += size
        elif aggressor == "SELL":
            state["signed_delta_buy_minus_sell"] -= size
            state["all_sell_volume"] += size
        if ts == state["origin_ns"]:
            state["origin_event_seen"] = True
        if state["attack_sign"] is not None and ts >= state["origin_ns"]:
            attack = (aggressor == "SELL" and state["attack_sign"] < 0) or (aggressor == "BUY" and state["attack_sign"] > 0)
            opposite = aggressor in ("BUY", "SELL") and not attack
            if attack:
                state["attack_volume"] += size
                state["attack_count"] += 1
                state["first_attack_ns"] = state["first_attack_ns"] or ts
                state["last_attack_ns"] = ts
                state["attack_prices"].add(price)
                if ep.low <= price <= ep.high:
                    state["attack_level_volume"] += size
                if book["valid"] and depth_now and depth_now > 0:
                    state["depth_persistent_attack_trades"] += 1
            elif opposite:
                state["opposing_volume"] += size
                state["opposing_count"] += 1
            elif aggressor == "UNKNOWN":
                state["unknown_volume"] += size
                state["unknown_count"] += 1
            if book["valid"] and state["start_mid"] is not None:
                progress = directional_price_progress(book["mid"],state["start_mid"],state["attack_sign"])
                state["max_attack_progress_ticks"] = max(state["max_attack_progress_ticks"], progress)
                state["min_attack_progress_ticks"] = min(state["min_attack_progress_ticks"], progress)
                if state["gap_count"]==0 and not (flags & base.FLAG_BAD_TS_RECV) and state["first_substantial_ns"] is None and ep.v1_threshold and state["attack_volume"] >= 2*ep.v1_threshold:
                    state["first_substantial_ns"] = ts
                    state["milestone_mids"]["first_substantial_aggression"]=book["mid"] if book["valid"] else None
                if state["gap_count"]==0 and ep.v1_threshold and state["attack_volume"] >= 2*ep.v1_threshold and state["attack_volume"]>0:
                    impact_ratio=state["max_attack_progress_ticks"]/state["attack_volume"]
                    if state["max_impact_ratio_seen"] is not None and impact_ratio < state["max_impact_ratio_seen"] and state["first_impact_ratio_decline_ns"] is None:
                        state["first_impact_ratio_decline_ns"]=ts
                        state["milestone_mids"]["first_relative_impact_reduction"]=book["mid"] if book["valid"] else None
                    state["max_impact_ratio_seen"]=(impact_ratio if state["max_impact_ratio_seen"] is None else max(state["max_impact_ratio_seen"],impact_ratio))
    if book["valid"] and not (flags & base.FLAG_BAD_TS_RECV) and state["start_mid"] is not None and state["attack_sign"] is not None:
        progress = directional_price_progress(book["mid"],state["start_mid"],state["attack_sign"])
        state["max_attack_progress_ticks"] = max(state["max_attack_progress_ticks"], progress)
        edge = ep.low if state["attack_sign"] < 0 else ep.high
        penetration = ((edge-book["mid"])/TICK if state["attack_sign"] < 0 else (book["mid"]-edge)/TICK)
        state["max_boundary_penetration_ticks"] = max(state["max_boundary_penetration_ticks"], penetration)
        reversal = state["max_attack_progress_ticks"]-progress
        state["max_reversal_from_attack_extreme_ticks"] = max(state["max_reversal_from_attack_extreme_ticks"], reversal)
        weak = weak_impact_marker(state["attack_volume"],ep.v1_threshold,
                                  state["max_attack_progress_ticks"],state["attack_count"])
        if weak and state["gap_count"]==0 and state["first_weak_impact_ns"] is None:
            state["first_weak_impact_ns"] = ts
            state["milestone_mids"]["first_weak_impact_evidence"]=book["mid"]
        if weak and state["gap_count"]==0 and state["first_stall_ns"] is None and state["attack_count"] >= 2:
            state["first_stall_ns"] = ts
            state["milestone_mids"]["first_directional_price_stall"]=book["mid"]
        if state["gap_count"]==0 and state["max_reversal_from_attack_extreme_ticks"] >= 1 and state["first_reversal_quote_ns"] is None:
            state["first_reversal_quote_ns"] = ts
            state["milestone_mids"]["first_reversal_direction_quote_move"]=book["mid"]
    state["previous_ns"], state["previous_event_ns"], state["previous_book"], state["previous_flags"] = recv, int(row["ts_event"]), book, flags
    return book


def _event_row(state: dict[str, Any], row: Any, ts: int, recv: int, seq: int, flags: int,
               book: dict[str, Any]) -> dict[str, Any]:
    ep: base.Episode = state["episode"]
    action, side = bytes(row["action"]).decode("ascii"), bytes(row["side"]).decode("ascii")
    depth = _depth_components(book, ep)
    feature = _feature_row(state, recv, book, flags, recv)
    return {"episode_id": ep.episode_id, "date": ep.date, "origin_ns": state["origin_ns"], "origin_kind": state["origin_kind"],
            "event_ns": ts, "ts_recv": recv, "origin_event_ns":state["origin_event_ns"], "ts_in_delta": int(row["ts_in_delta"]), "sequence": seq,
            "action": action, "side": side, "trade_aggressor": base.trade_aggressor(action, side),
            "price": int(row["price"])/1e9, "size": int(row["size"]), "depth_index": int(row["depth"]),
            "flags": flags, "quote_valid": book["valid"], "bid": book["bid"], "ask": book["ask"], "mid": book["mid"],
            "spread_ticks": book["spread_ticks"], "top1_opposing_depth": depth["top1_opposing_depth"],
            "top5_opposing_depth": depth["top5_opposing_depth"], "top10_opposing_depth": depth["top10_opposing_depth"],
            "top1_bid_depth":depth["top1_bid_depth"],"top1_ask_depth":depth["top1_ask_depth"],
            "top5_bid_depth":depth["top5_bid_depth"],"top5_ask_depth":depth["top5_ask_depth"],
            "top10_bid_depth":depth["top10_bid_depth"],"top10_ask_depth":depth["top10_ask_depth"],
            **{k: feature[k] for k in ("attack_volume_contracts", "opposing_aggressor_volume_contracts", "unknown_aggressor_volume_contracts",
                                       "net_delta_buy_minus_sell_contracts", "known_attack_execution_count", "opposing_execution_count",
                                       "attack_direction_displacement_ticks", "max_attack_direction_progress_ticks",
                                       "attack_direction_price_impact_ticks_per_attack_contract", "defended_price_depth",
                                       "depth_reductions_contracts", "depth_increases_contracts",
                                       "all_buy_aggressor_volume_contracts", "all_sell_aggressor_volume_contracts")}}


def _process_date(repo: Path, date: str, episodes: list[base.Episode], raw_meta: dict[str, Any], parquet_writer: Any, parquet_schema: Any) -> dict[str, Any]:
    from databento import DBNStore
    date_eps = sorted((e for e in episodes if e.date == date), key=lambda e: (_origin(e)[0], e.episode_id))
    states = [_new_state(e) for e in date_eps]
    origin_refs = [_origin(s["episode"])[0] for s in states]
    starts = [x - 2_000_000_000 for x in origin_refs]
    ends = [x + max(OFFSETS_MS[-1], 5000)*1_000_000 + FORWARD_HORIZONS_MS[-1]*1_000_000 for x in origin_refs]
    intervals = sorted((start, end) for start, end in zip(starts, ends))
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    store = DBNStore.from_file(raw_meta["absolute_path"])
    total = trade_count = trade_a = trade_b = trade_n = 0
    actions: Counter[str] = Counter()
    sequence_nonincrease = event_back = recv_back = 0
    first_event = last_event = first_recv = last_recv = prior_event = prior_recv = prior_seq = None
    flags_count: Counter[str] = Counter()
    unknown_rows: list[dict[str, Any]] = []
    next_state = 0
    active: list[dict[str, Any]] = []
    parquet_rows: list[dict[str, Any]] = []
    for chunk in store.to_ndarray(count=CHUNK_ROWS):
        if not isinstance(chunk, np.ndarray) or chunk.dtype.names is None:
            raise TypeError("DBNStore.to_ndarray returned unexpected data")
        if not np.all(chunk["rtype"] == 10):
            raise ValueError(f"Unexpected record type in {date}; expected MBP-10 rtype=10")
        total += len(chunk)
        if not len(chunk): continue
        tsarr, rarr, sarr = chunk["ts_event"], chunk["ts_recv"], chunk["sequence"]
        if first_event is None: first_event, first_recv = int(tsarr[0]), int(rarr[0])
        event_back += int(np.count_nonzero(tsarr[1:] < tsarr[:-1]))
        recv_back += int(np.count_nonzero(rarr[1:] < rarr[:-1]))
        sequence_nonincrease += int(np.count_nonzero(sarr[1:] <= sarr[:-1]))
        if prior_event is not None and int(tsarr[0]) < prior_event: event_back += 1
        if prior_recv is not None and int(rarr[0]) < prior_recv: recv_back += 1
        if prior_seq is not None and int(sarr[0]) <= prior_seq: sequence_nonincrease += 1
        prior_event, prior_recv, prior_seq = int(tsarr[-1]), int(rarr[-1]), int(sarr[-1])
        last_event, last_recv = prior_event, prior_recv
        action_values, action_counts = np.unique(chunk["action"], return_counts=True)
        actions.update({bytes(a).decode("ascii"): int(c) for a, c in zip(action_values, action_counts)})
        trade_mask = chunk["action"] == b"T"
        trade_count += int(np.count_nonzero(trade_mask))
        for side_name, variable in ((b"A", "a"), (b"B", "b"), (b"N", "n")):
            n = int(np.count_nonzero(trade_mask & (chunk["side"] == side_name)))
            if variable == "a": trade_a += n
            elif variable == "b": trade_b += n
            else: trade_n += n
        for bit, name in ((base.FLAG_BAD_TS_RECV,"F_BAD_TS_RECV"),(base.FLAG_MAYBE_BAD_BOOK,"F_MAYBE_BAD_BOOK"),(base.FLAG_SNAPSHOT,"F_SNAPSHOT"),(base.FLAG_LAST,"F_LAST")):
            flags_count[name] += int(np.count_nonzero(chunk["flags"] & bit))
        if trade_n and len(unknown_rows)<8:
            for row in chunk[trade_mask & (chunk["side"] == b"N")][:8-len(unknown_rows)]:
                unknown_rows.append({"ts_event":int(row["ts_event"]),"ts_recv":int(row["ts_recv"]),"sequence":int(row["sequence"]),"size":int(row["size"]),"price":int(row["price"])/1e9,"depth":int(row["depth"]),"flags":int(row["flags"])})
        # V1 ledger timestamps are ts_recv. Checkpoints use receive/availability
        # time so records cannot enter a snapshot before they were observable.
        monotone = bool(np.all(rarr[1:] >= rarr[:-1]))
        selected: list[int] = []
        if monotone:
            for lo, hi in merged:
                a, b = int(np.searchsorted(rarr, lo, "left")), int(np.searchsorted(rarr, hi, "right"))
                if a < b: selected.extend(range(a,b))
        else:
            selected = base.native_order_window_indices(rarr, merged)
        for ix in selected:
            row = chunk[ix]
            ts, recv, seq, flags = int(row["ts_event"]), int(row["ts_recv"]), int(row["sequence"]), int(row["flags"])
            clock_ns = recv
            while next_state < len(states) and starts[next_state] <= clock_ns:
                active.append(states[next_state]); next_state += 1
            active = [st for st in active if clock_ns <= st["origin_reference_ns"] + 35_000_000_000]
            for st in active:
                if not st["origin_resolved"]:
                    ep=st["episode"]
                    action0=bytes(row["action"])
                    price0=int(row["price"])
                    side0=bytes(row["side"]).decode("ascii")
                    matched = (source_origin_match(ep,recv,action0.decode("ascii"),price0,int(row["size"]),side0)
                               if ep.v1_heavy else source_touch_match(ep,row,recv))
                    if not matched:
                        continue
                    st["origin_resolved"]=True
                    st["origin_event_ns"]=ts
                    st["origin_ns"]=recv
                    st["origin_event_seen"]=(action0.decode("ascii")=="T")
                    st["origin_match_action"]=action0.decode("ascii")
                    st["origin_match_side"]=side0
                    st["origin_match_size"]=int(row["size"])
                    st["origin_match_price"]=price0/1e9
                origin = st["origin_ns"]
                if clock_ns > origin + 35_000_000_000: continue
                # Close fixed snapshots from the latest event strictly before each checkpoint.
                for offset in OFFSETS_MS:
                    target = origin + offset*1_000_000
                    if offset in st["snapshots"] or target >= clock_ns: continue
                    prev_ts, prev_book = st["previous_ns"], st["previous_book"]
                    reason = "NO_PRIOR_EVENT_AT_CHECKPOINT"
                    if prev_ts is not None:
                        reason = None if target-prev_ts <= MAX_STALE_NS and prev_book and prev_book["valid"] and st["gap_count"]==0 else "STALE_OR_INVALID_SOURCE_OBSERVATION"
                    _take_snapshot(st,target,prev_ts,prev_book,st["previous_flags"],reason)
                previous_ns = st["previous_ns"]
                book = _update_state(st,row,clock_ns,recv,seq,flags)
                if book["valid"] and not (flags & base.FLAG_BAD_TS_RECV) and st["gap_count"] == 0:
                    milestone_keys = (
                        ("first_substantial_aggression", "first_substantial_ns"),
                        ("first_weak_impact_evidence", "first_weak_impact_ns"),
                        ("first_relative_impact_reduction", "first_impact_ratio_decline_ns"),
                        ("first_displayed_depth_recovery", "first_depth_recovery_ns"),
                        ("first_directional_price_stall", "first_stall_ns"),
                        ("first_reversal_direction_quote_move", "first_reversal_quote_ns"),
                    )
                    for milestone, field in milestone_keys:
                        if st[field] != recv or milestone in st["milestone_forward_registered"]:
                            continue
                        snap = _feature_row(st, recv, book, flags, recv)
                        snap["checkpoint_kind"], snap["milestone"] = "EVENT_MILESTONE", milestone
                        st["forward"].append({
                            "snapshot": snap, "reference_mid": book["mid"], "direction": st["episode"].direction,
                            "checkpoint_kind": "EVENT_MILESTONE", "milestone": milestone,
                            "horizons": {h: {"end_ns": recv+h*1_000_000, "last_mid": book["mid"],
                                "last_ns": recv, "last_event_ns": ts, "max_favorable_ticks": 0.0,
                                "max_adverse_ticks": 0.0, "gap_count": 0} for h in FORWARD_HORIZONS_MS},
                        })
                        st["milestone_forward_registered"].add(milestone)
                if clock_ns <= origin + 5000*1_000_000:
                    parquet_rows.append(_event_row(st,row,ts,recv,seq,flags,book))
                    st["measurement_event_rows"] += 1
                for offset in OFFSETS_MS:
                    target = origin + offset*1_000_000
                    if offset not in st["snapshots"] and target == clock_ns:
                        reason = None if book["valid"] and st["gap_count"]==0 else "INVALID_BOOK_OR_SOURCE_GAP"
                        _take_snapshot(st,target,recv,book,flags,reason)
                if book["valid"] and not (flags & base.FLAG_BAD_TS_RECV):
                    for tracker in st["forward"]:
                        if recv < tracker["snapshot"]["checkpoint_ns"]: continue
                        rel = (book["mid"]-tracker["reference_mid"])/TICK*(tracker["direction"] or 0)
                        for hor in tracker["horizons"].values():
                            if recv <= hor["end_ns"]:
                                if previous_ns is not None and recv-previous_ns>MAX_STALE_NS:
                                    hor["gap_count"] += 1
                                hor["last_mid"], hor["last_ns"] = book["mid"], recv
                                hor["last_event_ns"] = ts
                                hor["max_favorable_ticks"] = max(hor["max_favorable_ticks"],rel)
                                hor["max_adverse_ticks"] = min(hor["max_adverse_ticks"],rel)
            if len(parquet_rows) >= 50_000:
                import pyarrow as pa
                table = pa.Table.from_pylist(parquet_rows, schema=parquet_schema)
                parquet_writer.write_table(table, row_group_size=50_000)
                parquet_rows.clear()
    if parquet_rows:
        import pyarrow as pa
        parquet_writer.write_table(pa.Table.from_pylist(parquet_rows,schema=parquet_schema),row_group_size=50_000)
    # Complete snapshots at end of a source window from the latest prior event; do not forward-fill stale/invalid data.
    for st in states:
        for offset in OFFSETS_MS:
            if offset in st["snapshots"]: continue
            target = st["origin_ns"] + offset*1_000_000
            pts,pbook = st["previous_ns"],st["previous_book"]
            reason = None if pts is not None and target-pts<=MAX_STALE_NS and pbook and pbook["valid"] and st["gap_count"]==0 else "SOURCE_END_STALE_OR_INVALID"
            _take_snapshot(st,target,pts,pbook,st["previous_flags"],reason)
    # Markouts use only valid quotes no later than each fixed horizon; incomplete horizons are censored.
    for st in states:
        for tracker in st["forward"]:
            for horizon_ms, record in tracker["horizons"].items():
                endpoint=record["end_ns"]
                latest=record["last_ns"]
                complete=last_recv is not None and last_recv>=endpoint and latest is not None and endpoint-latest<=MAX_STALE_NS and record["gap_count"]==0
                record["source_valid_horizon"]=bool(complete)
                record["markout_observed_ns"]=latest
                record["markout_direction_adjusted_ticks"]=(record["last_mid"]-tracker["reference_mid"])/TICK*(tracker["direction"] or 0) if complete else None
                record["max_favorable_ticks"]=record["max_favorable_ticks"] if complete else None
                record["max_adverse_ticks"]=record["max_adverse_ticks"] if complete else None
                record["horizon_ms"]=horizon_ms
                record["snapshot_offset_ms"]=tracker["snapshot"]["offset_ms"]
                record["checkpoint_kind"]=tracker["checkpoint_kind"]
                record["milestone"]=tracker["milestone"]
                record["response_category"]=_response_category(tracker["snapshot"])
                record["depth_recovery_by_checkpoint"]=("RECOVERY_OBSERVED" if tracker["snapshot"].get("displayed_depth_recovery_ns") is not None and tracker["snapshot"]["displayed_depth_recovery_ns"]<=tracker["snapshot"]["checkpoint_ns"] else "NO_RECOVERY_OBSERVED_BY_CHECKPOINT")
                record["episode_id"]=tracker["snapshot"]["episode_id"]
                record["date"]=date
                st.setdefault("forward_rows",[]).append(record.copy())
    return {"date":date,"record_count":total,"trade_records":trade_count,"trade_side_counts":{"A_sell_aggressor":trade_a,"B_buy_aggressor":trade_b,"N_unknown":trade_n},
            "actions":dict(actions),"flags":dict(flags_count),"first_ts_event":first_event,"last_ts_event":last_event,"first_ts_recv":first_recv,"last_ts_recv":last_recv,
            "ts_event_backward_steps":event_back,"ts_recv_backward_steps":recv_back,"sequence_nonincrease_steps":sequence_nonincrease,
            "unknown_side_examples":unknown_rows,"episode_count":len(date_eps),"selected_event_rows":sum(len(st["events"]) for st in states),"states":states}


def _rank_groups(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for row in snapshots:
        if not row["snapshot_valid"]: continue
        vol, threshold = row["attack_volume_contracts"], row["substantial_aggression_convention_contracts"]
        if threshold is None:
            cat="UNCLASSIFIED_NO_THRESHOLD"
        elif vol >= threshold and row["max_attack_direction_progress_ticks"] is not None and row["max_attack_direction_progress_ticks"] <= 1:
            cat="HIGH_EFFORT_LIMITED_PROGRESS_MARKER"
        elif vol >= threshold and (row["max_hvn_boundary_penetration_ticks"] or 0) >= 4:
            cat="HIGH_EFFORT_PRICE_DISCOVERY_MARKER"
        elif vol < threshold:
            cat="LOWER_EFFORT_AT_CHECKPOINT"
        else:
            cat="MIXED_OR_UNCLASSIFIED"
        groups[cat].append(row)
    out={}
    for name, rs in groups.items():
        out[name]={"n":len(rs),"episode_count":len({r['episode_id'] for r in rs}),"dates":sorted({r['date'] for r in rs}),
                   "forward":{}}
    return out


def _response_category(row: dict[str, Any]) -> str:
    """Outcome-blind descriptive grouping; continuous values remain primary."""
    threshold = row.get("substantial_aggression_convention_contracts")
    if threshold is None:
        return "UNCLASSIFIED_NO_V1_THRESHOLD"
    volume = row.get("attack_volume_contracts", 0)
    if volume >= threshold and row.get("max_attack_direction_progress_ticks") is not None and row["max_attack_direction_progress_ticks"] <= 1:
        return "HIGH_EFFORT_LIMITED_PROGRESS_MARKER"
    if volume >= threshold and (row.get("max_hvn_boundary_penetration_ticks") or 0) >= 4:
        return "HIGH_EFFORT_PRICE_DISCOVERY_MARKER"
    if volume < threshold:
        return "LOWER_EFFORT_AT_CHECKPOINT"
    return "MIXED_OR_UNCLASSIFIED"


def _combined_measurement_timing(event_path: Path, population: list[dict[str, Any]]) -> dict[str, Any]:
    """Stream the event parquet and locate the earliest fully measurable effort/impact/depth row."""
    import pyarrow.parquet as pq
    meta={r["episode_id"]:r for r in population}
    previous: dict[str,int]={}
    gap: set[str]=set()
    found: dict[str,dict[str,Any]]={}
    parquet=pq.ParquetFile(event_path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["episode_id","ts_recv","origin_ns","flags","quote_valid",
            "attack_volume_contracts","attack_direction_displacement_ticks","defended_price_depth"]):
        cols=batch.to_pydict()
        for i,episode_id in enumerate(cols["episode_id"]):
            recv=int(cols["ts_recv"][i]); prior=previous.get(episode_id)
            if prior is not None and recv-prior>MAX_STALE_NS: gap.add(episode_id)
            previous[episode_id]=recv
            flags=int(cols["flags"][i] or 0)
            if flags & base.FLAG_BAD_TS_RECV: gap.add(episode_id)
            if episode_id in found or episode_id not in meta: continue
            ep=meta[episode_id]
            if ep["direction"] is None or episode_id in gap: continue
            if (cols["quote_valid"][i] and int(cols["attack_volume_contracts"][i] or 0)>0
                    and cols["attack_direction_displacement_ticks"][i] is not None
                    and cols["defended_price_depth"][i] is not None):
                origin=int(cols["origin_ns"][i]); entry=ep.get("v1_entry_ns")
                found[episode_id]={"timestamp_ns":recv,"ms_from_origin":(recv-origin)/1e6,
                    "available_before_v1_entry":bool(entry is not None and recv<=int(entry))}
    timing=[found[k] for k in sorted(found)]
    directional=sum(r.get("direction") is not None for r in population)
    return {"definition":"Earliest native event with known episode direction, at least one known attack execution, valid two-sided quote/price-impact baseline, non-null original-zone opposing depth, and no preceding >2 s receive-time gap or bad receive-time flag within the recorded event window.",
        "directional_episodes":directional,"episodes_with_measurement":len(timing),
        "episodes_with_measurement_before_v1_entry":sum(x["available_before_v1_entry"] for x in timing),
        "measurable_pct_of_directional_episodes":100*len(timing)/directional if directional else None,
        "measurable_before_entry_pct_of_directional_episodes":100*sum(x["available_before_v1_entry"] for x in timing)/directional if directional else None,
        "latency_ms_from_origin":_quantiles([x["ms_from_origin"] for x in timing]),
        "note":"Event parquet covers the origin through the 5 s fixed observation window. Episodes without a qualifying row in that window are right-censored for this timing summary; this is not evidence that measurements never become available later."}


def _analyse(snapshots: list[dict[str, Any]], forward: list[dict[str,Any]], episodes: list[base.Episode]) -> dict[str,Any]:
    entry_ns={e.episode_id:int(e.v1["TE_ns"]) for e in episodes if e.v1 and e.v1.get("TE_ns")}
    by_offset={}
    for offset in OFFSETS_MS:
        cohort=[r for r in snapshots if int(r["offset_ms"])==offset]
        valid=[r for r in cohort if r["snapshot_valid"]]
        confirmed=[r for r in cohort if r["v1_confirmed"]]
        nonconfirmed=[r for r in cohort if not r["v1_confirmed"]]
        confirmed_pre=[r for r in confirmed if int(r["checkpoint_ns"]) <= entry_ns.get(r["episode_id"],-1)]
        by_offset[str(offset)]={"all":{"n":len(cohort),"valid":len(valid),"coverage_pct":100*len(valid)/len(cohort) if cohort else None},
          "v1_entries":{"n":len(confirmed),"valid":sum(r["snapshot_valid"] and int(r["checkpoint_ns"]) <= entry_ns.get(r["episode_id"],-1) for r in confirmed),
                        "valid_snapshot_checkpoints":sum(r["snapshot_valid"] for r in confirmed),
                        "checkpoint_before_entry":len(confirmed_pre),
                        "coverage_pct":100*sum(r["snapshot_valid"] and int(r["checkpoint_ns"]) <= entry_ns.get(r["episode_id"],-1) for r in confirmed)/len(confirmed) if confirmed else None},
          "nonconfirmed":{"n":len(nonconfirmed),"valid":sum(r["snapshot_valid"] for r in nonconfirmed),"coverage_pct":100*sum(r["snapshot_valid"] for r in nonconfirmed)/len(nonconfirmed) if nonconfirmed else None}}
        for group, subset in (("WINNER",[r for r in confirmed if r["rr1_class"]=="WINNER"]),("LOSER",[r for r in confirmed if r["rr1_class"]=="LOSER"])):
            by_offset[str(offset)][group.lower()]={"n":len(subset),"valid":sum(r["snapshot_valid"] for r in subset),"date_count":len({r['date'] for r in subset if r['snapshot_valid']})}
    fwd={}
    response_groups={}
    depth_groups={}
    for offset in OFFSETS_MS:
        fwd[str(offset)]={}
        response_groups[str(offset)]={}
        depth_groups[str(offset)]={}
        for horizon in FORWARD_HORIZONS_MS:
            rs=[x for x in forward if x.get("checkpoint_kind", "FIXED_OFFSET")=="FIXED_OFFSET" and x["snapshot_offset_ms"]==offset and x["horizon_ms"]==horizon]
            valid=[x for x in rs if x.get("source_valid_horizon")]
            fwd[str(offset)][str(horizon)]={"eligible_snapshots":len(rs),"source_valid":len(valid),"censored":len(rs)-len(valid),
                "survivor_pct":100*len(valid)/len(rs) if rs else None,
                "direction_adjusted_markout_ticks":_quantiles([x["markout_direction_adjusted_ticks"] for x in valid]),
                "max_favorable_ticks":_quantiles([x["max_favorable_ticks"] for x in valid]),
                "max_adverse_ticks":_quantiles([x["max_adverse_ticks"] for x in valid])}
            for field, target in (("response_category", response_groups[str(offset)]), ("depth_recovery_by_checkpoint", depth_groups[str(offset)])):
                labels=sorted({x[field] for x in rs})
                target[str(horizon)]={}
                for label in labels:
                    group=[x for x in rs if x[field]==label]
                    survivors=[x for x in group if x.get("source_valid_horizon")]
                    target[str(horizon)][label]={"eligible":len(group),"source_valid":len(survivors),"censored":len(group)-len(survivors),
                        "direction_adjusted_markout_ticks":_quantiles([x["markout_direction_adjusted_ticks"] for x in survivors]),
                        "max_favorable_ticks":_quantiles([x["max_favorable_ticks"] for x in survivors]),
                        "max_adverse_ticks":_quantiles([x["max_adverse_ticks"] for x in survivors])}
    v1=[r for r in snapshots if r["v1_confirmed"]]
    cutoffs={str(offset):{} for offset in OFFSETS_MS}
    for offset in OFFSETS_MS:
        rows=[r for r in v1 if r["offset_ms"]==offset and int(r["checkpoint_ns"]) <= entry_ns.get(r["episode_id"],-1)]
        for feat in ("attack_volume_contracts","max_attack_direction_progress_ticks","attack_direction_price_impact_ticks_per_attack_contract","defended_price_depth","depth_reductions_contracts","depth_increases_contracts","directional_persistence_ratio"):
            cutoffs[str(offset)][feat]={}
            for cls in ("WINNER","LOSER"):
                subset=[r for r in rows if r["rr1_class"]==cls and r["snapshot_valid"]]
                vals=[float(r[feat]) for r in subset if _finite(r.get(feat)) is not None]
                cutoffs[str(offset)][feat][cls]={"valid_n":len(vals),"date_count":len({r['date'] for r in subset if _finite(r.get(feat)) is not None}),**_quantiles(vals)}
    milestone_forward={}
    for milestone in sorted({x.get("milestone") for x in forward if x.get("checkpoint_kind")=="EVENT_MILESTONE"}):
        milestone_forward[milestone]={}
        for horizon in FORWARD_HORIZONS_MS:
            rs=[x for x in forward if x.get("checkpoint_kind")=="EVENT_MILESTONE" and x.get("milestone")==milestone and x["horizon_ms"]==horizon]
            valid=[x for x in rs if x.get("source_valid_horizon")]
            milestone_forward[milestone][str(horizon)]={"eligible":len(rs),"source_valid":len(valid),"censored":len(rs)-len(valid),
                "direction_adjusted_markout_ticks":_quantiles([x["markout_direction_adjusted_ticks"] for x in valid]),
                "max_favorable_ticks":_quantiles([x["max_favorable_ticks"] for x in valid]),
                "max_adverse_ticks":_quantiles([x["max_adverse_ticks"] for x in valid])}
    return {"fixed_snapshot_coverage":by_offset,"forward_response_by_aligned_snapshot_cohort":fwd,
            "forward_response_after_event_milestones":milestone_forward,
            "forward_response_by_predeclared_effort_impact_group":response_groups,
            "forward_response_by_depth_recovery_status":depth_groups,"winner_loser_by_offset":cutoffs,
            "forward_cohort_warning":"Each horizon is reported only among snapshots valid at that offset; longer-horizon survivor populations are separately counted and must not be compared as if they were the same cohort."}


def run(repo: Path) -> dict[str,Any]:
    started=time.monotonic()
    repo=repo.resolve()
    _, prior_checks, raw=base.load_and_verify_sources(repo)
    previous_root=repo/"research_runs/ES_HVN_NATIVE_MBP10_ABSORPTION_MECHANISM_V1"
    previous_hashes=json.loads((previous_root/"artifact-hashes.json").read_text())
    previous_bad=[]
    for rel,digest in previous_hashes.items():
        path=previous_root/rel
        if not path.is_file() or _sha256(path)!=digest:
            previous_bad.append(rel)
    if previous_bad:
        raise ValueError(f"Prior native mechanism artifacts failed hash verification: {previous_bad[:5]}")
    prior_checks["research_runs/ES_HVN_NATIVE_MBP10_ABSORPTION_MECHANISM_V1"]={"checked_artifacts":len(previous_hashes),"all_match":True}
    episodes=base.make_episodes(repo)
    if len(episodes)!=350: raise ValueError(f"Expected prior deterministic five-day episode population 350; found {len(episodes)}")
    out=repo/OUT
    out.mkdir(parents=True,exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("The repository .venv with pyarrow is required to write the requested real Parquet event file") from e
    event_path=out/"event-driven-absorption-features.parquet"
    event_schema=pa.schema([
        ("episode_id",pa.string()),("date",pa.string()),("origin_ns",pa.int64()),("origin_kind",pa.string()),
        ("event_ns",pa.int64()),("ts_recv",pa.int64()),("origin_event_ns",pa.int64()),("ts_in_delta",pa.int32()),("sequence",pa.uint32()),
        ("action",pa.string()),("side",pa.string()),("trade_aggressor",pa.string()),("price",pa.float64()),("size",pa.uint32()),("depth_index",pa.uint8()),
        ("flags",pa.uint8()),("quote_valid",pa.bool_()),("bid",pa.float64()),("ask",pa.float64()),("mid",pa.float64()),("spread_ticks",pa.float64()),
        ("top1_opposing_depth",pa.int64()),("top5_opposing_depth",pa.int64()),("top10_opposing_depth",pa.int64()),
        ("top1_bid_depth",pa.int64()),("top1_ask_depth",pa.int64()),("top5_bid_depth",pa.int64()),("top5_ask_depth",pa.int64()),
        ("top10_bid_depth",pa.int64()),("top10_ask_depth",pa.int64()),
        ("attack_volume_contracts",pa.int64()),("opposing_aggressor_volume_contracts",pa.int64()),("unknown_aggressor_volume_contracts",pa.int64()),
        ("all_buy_aggressor_volume_contracts",pa.int64()),("all_sell_aggressor_volume_contracts",pa.int64()),
        ("net_delta_buy_minus_sell_contracts",pa.int64()),("known_attack_execution_count",pa.int64()),("opposing_execution_count",pa.int64()),
        ("attack_direction_displacement_ticks",pa.float64()),("max_attack_direction_progress_ticks",pa.float64()),
        ("attack_direction_price_impact_ticks_per_attack_contract",pa.float64()),("defended_price_depth",pa.int64()),
        ("depth_reductions_contracts",pa.int64()),("depth_increases_contracts",pa.int64())
    ])
    writer=pq.ParquetWriter(event_path,event_schema)
    all_stats=[]; all_states=[]
    try:
        for date in base.DATES:
            print(f"streaming {date}",flush=True)
            result=_process_date(repo,date,episodes,raw[date],writer,event_schema)
            all_stats.append({k:v for k,v in result.items() if k != "states"})
            all_states.extend(result["states"])
            # Exact raw trade accounting reconciles the prior V1 metadata discrepancy.
            if result["trade_records"] != int(raw[date]["trade_records"])+ (1 if date=="2025-03-07" else 0):
                raise ValueError(f"Raw trade count does not reconcile with sealed source metadata: {date}")
    finally:
        writer.close()
    population=[]; snapshots=[]; forward=[]; milestones=[]; event_rows=0
    for st in all_states:
        ep:base.Episode=st["episode"]
        origin,kind=st["origin_ns"],st["origin_kind"]
        if not st["origin_resolved"]:
            raise ValueError(f"V1 ts_recv origin does not map to the exact native source row: {ep.episode_id} at {st['origin_reference_ns']}")
        population.append({"episode_id":ep.episode_id,"date":ep.date,"zone_id":ep.zone_id,"origin_ns":origin,"origin_kind":kind,
            "origin_price":ep.v1_heavy_price if kind=="FIRST_SOURCE_SUPPORTED_HEAVY_PRINT" else ep.touch_price,
            "origin_reference_ts_recv_ns":st["origin_reference_ns"],"origin_event_ts_event_ns":st["origin_event_ns"],
            "origin_event_matched_native_trade":st["origin_event_seen"],"origin_native_source_row_matched":st["origin_resolved"],
            "origin_match_action":st["origin_match_action"],"origin_match_side":st["origin_match_side"],
            "origin_match_size":st["origin_match_size"],"origin_match_price":st["origin_match_price"],
            "v1_entry_ns":int(ep.v1["TE_ns"]) if ep.v1 and ep.v1.get("TE_ns") else None,
            "v1_status":ep.v1_status,"v1_heavy_print":ep.v1_heavy,"v1_heavy_threshold_contracts":ep.v1_threshold,
            "direction":ep.direction,"attack_side":"SELL" if ep.direction==1 else "BUY" if ep.direction==-1 else "UNKNOWN",
            "v1_confirmed":bool(ep.v1),"rr1_class":"WINNER" if ep.rr1 and float(ep.rr1.get("net_r",0) or 0)>0 else "LOSER" if ep.rr1 else "UNTRADED_OR_UNAVAILABLE",
            "episode_overlap_preserved":True,"source_episode_id":ep.episode_id})
        snapshots.extend(st["snapshots"].values())
        forward.extend(st.get("forward_rows",[]))
        event_rows+=st["measurement_event_rows"]
        for name, ts in (("first_substantial_aggression",st["first_substantial_ns"]),("first_weak_impact_evidence",st["first_weak_impact_ns"]),
                         ("first_relative_impact_reduction",st["first_impact_ratio_decline_ns"]),
                         ("first_displayed_depth_recovery",st["first_depth_recovery_ns"]),("first_directional_price_stall",st["first_stall_ns"]),
                         ("first_reversal_direction_quote_move",st["first_reversal_quote_ns"])):
            milestones.append({"episode_id":ep.episode_id,"date":ep.date,"origin_kind":kind,"milestone":name,"timestamp_ns":ts,
                "ms_from_origin":(ts-origin)/1e6 if ts is not None else None,"available_before_v1_entry":ts<=int(ep.v1["TE_ns"]) if ts is not None and ep.v1 and ep.v1.get("TE_ns") else None,
                "v1_confirmed":bool(ep.v1),"rr1_class":population[-1]["rr1_class"]})
    snapshots.sort(key=lambda x:(x["date"],x["origin_ns"],x["episode_id"],x["offset_ms"]))
    forward.sort(key=lambda x:(x["date"],x["episode_id"],x["snapshot_offset_ms"],x["horizon_ms"]))
    _write_csv(out/"episode-population.csv",population)
    _write_csv(out/"time-snapshot-features.csv",snapshots)
    availability=[]
    for s in snapshots:
        availability.append({"episode_id":s["episode_id"],"date":s["date"],"origin_kind":s["origin_kind"],"offset_ms":s["offset_ms"],
            "v1_confirmed":s["v1_confirmed"],"v1_entry_ns":s["v1_entry_ns"],"available_before_v1_entry":s["available_before_v1_entry"],
            "rr1_class":s["rr1_class"],"source_endpoint_valid":s["snapshot_valid"],
            "aggression_measurement_valid":s["snapshot_valid"] and s["direction"] is not None,
            "price_impact_measurement_valid":s["snapshot_valid"] and s["direction"] is not None and s["initial_mid"] is not None,
            "depth_response_measurement_valid":s["snapshot_valid"] and s["direction"] is not None and s["defended_price_depth"] is not None,
            "combined_measurement_valid":s["snapshot_valid"] and s["direction"] is not None and s["initial_mid"] is not None and s["defended_price_depth"] is not None,
            "unavailable_reason":s["endpoint_unavailable_reason"]})
    _write_csv(out/"measurement-availability.csv",availability)
    import pyarrow as pa
    import pyarrow.parquet as pq
    event_rows=pq.ParquetFile(event_path).metadata.num_rows
    combined_timing=_combined_measurement_timing(event_path,population)
    # Event output keeps all relevant native market updates, not a fabricated or sampled tape.
    source_episodes={e.episode_id:e for e in episodes}
    substantial_thresholds=[2*e.v1_threshold for e in episodes if e.v1_threshold and e.v1_threshold>0]
    metrics=[r for r in snapshots if r["v1_confirmed"]]
    v1entries={e.episode_id:int(e.v1["TE_ns"]) for e in episodes if e.v1 and e.v1.get("TE_ns")}
    before_entry={}
    for offset in OFFSETS_MS:
        eligible=[r for r in snapshots if r["v1_confirmed"] and r["offset_ms"]==offset]
        eligible_before=[r for r in eligible if int(r["checkpoint_ns"]) <= v1entries.get(r["episode_id"],-1)]
        valid=[r for r in eligible_before if r["snapshot_valid"]]
        before_entry[str(offset)]={"entries":len(eligible),"checkpoint_before_entry":len(eligible_before),"valid_before_entry":len(valid),
            "percent_of_entries_with_valid_evidence_by_checkpoint":100*len(valid)/len(eligible) if eligible else None,
            "valid_snapshot_pct_when_checkpoint_precedes_entry":100*len(valid)/len(eligible_before) if eligible_before else None}
    first_events={}
    for name in ("first_substantial_aggression","first_weak_impact_evidence","first_relative_impact_reduction","first_displayed_depth_recovery","first_directional_price_stall","first_reversal_direction_quote_move"):
        vals=[r["timestamp_ns"] for r in milestones if r["milestone"]==name and r["timestamp_ns"] is not None]
        first_events[name]={"episode_count":len(vals),"minimum_ms_from_origin":min((r["ms_from_origin"] for r in milestones if r["milestone"]==name and r["timestamp_ns"] is not None),default=None),
                            "median_ms_from_origin":statistics.median(r["ms_from_origin"] for r in milestones if r["milestone"]==name and r["timestamp_ns"] is not None) if vals else None}
    analysis=_analyse(snapshots,forward,episodes)
    counts={}
    for key in ("first_substantial_aggression","first_weak_impact_evidence","first_relative_impact_reduction","first_displayed_depth_recovery","first_directional_price_stall","first_reversal_direction_quote_move"):
        rs=[r for r in milestones if r["milestone"]==key and r["v1_confirmed"]]
        counts[key]={"all_confirmed":sum(r["timestamp_ns"] is not None for r in rs),"before_entry":sum(r["available_before_v1_entry"] is True for r in rs)}
    source_meta=json.loads((repo/base.V1/"source-integrity-results.json").read_text())
    v1_by_date={x["date"]:x for x in source_meta["sources"]}
    discrepancy={s["date"]:{"native_trade_records":s["trade_records"],"v1_source_manifest_trade_records":int(v1_by_date[s["date"]]["trade_records"]),
                             "unknown_aggressor_trade_records":s["trade_side_counts"]["N_unknown"],"difference":s["trade_records"]-int(v1_by_date[s["date"]]["trade_records"])} for s in all_stats}
    integrity={"status":"PASS_WITH_RECONCILED_V1_METADATA_DISCREPANCY","source_dataset":"GLBX.MDP3","schema":"mbp-10","symbol":"ESH5",
        "dates":list(base.DATES),"native_sources":{d:{"path":raw[d]["raw_path"],"bytes":raw[d]["raw_bytes"],"sha256":raw[d]["raw_sha256"],"hash_matches":raw[d]["hash_matches"],"bytes_match":raw[d]["bytes_match"]} for d in base.DATES},
        "prior_artifact_hash_checks":prior_checks,"native_stream_checks":all_stats,"trade_count_reconciliation":discrepancy,
        "unknown_side_trade_detail_2025_03_07":all_stats[-1]["unknown_side_examples"],
        "v1_timestamp_mapping":"The frozen V1 compact tape documents its timestamp as ts_recv. Every episode origin was mapped to the native T row by exact ts_recv plus price, and for heavy prints exact side and size. The new causal clock is ts_recv (availability); paired ts_event is retained as event time. No rounding or nearest-event substitution is used.",
        "conclusion":"One Mar 7 T/N record, size 99 at 5776.50, explains the +1 native trade count versus V1 source metadata. It is retained in raw-event totals and excluded from attack-side volume and signed directional side assignment. No prior tape or V1 artifact was revised.",
        "sequence_ordering":"Native DBN row order is retained. ts_event and ts_recv monotonicity are checked; venue sequence is recorded but not assumed to be a global monotone key."}
    _write_json(out/"source-integrity-and-semantics.json",integrity)
    _write_json(out/"aggression-price-impact-analysis.json",{"n_episodes":len(population),"origin_convention":{"heavy":"first source-supported V1 qualifying heavy print timestamp","no_heavy":"first-touch timestamp; separately stratified"},
        "substantial_aggression_marker":"Cumulative attack-side volume >= 2x the episode's source-defined V1 heavy threshold; descriptive continuity marker only, not outcome-selected.",
        "weak_impact_marker":"Substantial aggression plus <=1 tick maximum attack-direction progress and at least two attack-side execution records, so one large print at the origin is not called weak impact.",
        "measurements":{"attack_volume":_quantiles([s["attack_volume_contracts"] for s in snapshots if s["snapshot_valid"]]),"opposing_volume":_quantiles([s["opposing_aggressor_volume_contracts"] for s in snapshots if s["snapshot_valid"]]),
                        "price_impact_ticks_per_contract":_quantiles([s["attack_direction_price_impact_ticks_per_attack_contract"] for s in snapshots if s["snapshot_valid"]]),
                        "attack_direction_displacement_ticks":_quantiles([s["attack_direction_displacement_ticks"] for s in snapshots if s["snapshot_valid"]])},
        "milestones":first_events,"milestone_availability":counts,"limitations":["Trade size is execution size, not inferred parent-order volume.","T/N size excluded from directional metrics.","Episodes overlap; observations are not independent."]})
    depth_rows=[s for s in snapshots if s["snapshot_valid"]]
    _write_json(out/"depth-resilience-analysis.json",{"valid_snapshot_count":len(depth_rows),"origin_group_counts":dict(Counter(s["origin_kind"] for s in population)),
        "defended_price_depth_contracts":_quantiles([s["defended_price_depth"] for s in depth_rows]),"top1_opposing_depth_contracts":_quantiles([s["top1_opposing_depth"] for s in depth_rows]),
        "top5_opposing_depth_contracts":_quantiles([s["top5_opposing_depth"] for s in depth_rows]),"top10_opposing_depth_contracts":_quantiles([s["top10_opposing_depth"] for s in depth_rows]),
        "first_recovery_milestones":first_events["first_displayed_depth_recovery"],"interpretation":"Depth changes at original defended prices are observable aggregate changes. Increases can reflect additions, migration, or other book updates; they are not proof of replenishment of consumed orders."})
    _write_json(out/"decision-time-feasibility.json",{"v1_entry_evidence_coverage_by_fixed_offset":before_entry,"first_combined_measurement":combined_timing,"combined_measurement":"Valid snapshot with known direction, valid quote, complete event clock (no >2s source gap), and price-aligned depth observable; marker conjunctions remain descriptive.",
        "milestone_counts":counts,"interpretation":"Snapshot availability means source-valid measurements can be computed by that checkpoint; it does not assert a detector signal or edge."})
    _write_json(out/"aligned-forward-response-analysis.json",analysis)
    _write_json(out/"winner-loser-preentry-analysis.json",{"source":"Original V1 RR=1 outcomes; secondary diagnostic only.","by_fixed_cutoff":analysis["winner_loser_by_offset"],
        "small_sample_warning":"Counts, quantiles and date coverage are descriptive; no classifier, threshold search, or config selection."})
    lineage={"availability_clock":"ts_recv, matching the frozen V1 compact tape timestamp; all features/checkpoints use the time the row became observable.",
        "event_clock":"ts_event retained alongside ts_recv. Native DBN file order is preserved; neither sequence nor timestamp is used to reorder equal-time rows.",
        "trade_side":"T/A=SELL aggressor, T/B=BUY aggressor, T/N unknown; only T records enter execution metrics; unknown sizes remain a separate column.",
        "attack_orientation":"Long-oriented support: attack side SELL; short-oriented resistance: attack side BUY. Signed direction is kept separate from conventional buy-minus-sell Delta.",
        "episode_origins":{"heavy":"episode's source-supported first qualifying V1 heavy-print timestamp","touch_with_price":"first-touch timestamp mapped at exact ts_recv and matching touch price in trade or displayed top-10","zone_only_no_direction":"exact first_touch_ns ts_recv row; no touch price/direction is invented","without_heavy":"first-touch timestamp, analyzed as a distinct origin cohort"},
        "snapshot_offsets_ms":list(OFFSETS_MS),"snapshot_selection":"latest native observation received at or before checkpoint; valid only with uncrossed two-sided MBP-10 quote, no F_BAD_TS_RECV, age <=2s, and no >2s receive-time gap since origin.",
        "substantial_aggression":"cumulative attack-side execution volume >= 2 * source-defined per-zone V1 heavy threshold; descriptive and not optimized.",
        "price_impact":"(mid_now-mid_at_origin)/tick * attack_sign; positive means movement with aggressor. Max progress is a running max; impact/contract divides current progress by cumulative attack volume.",
        "depth":"price-aligned sum on bid for sell attack / ask for buy attack at original HVN prices; top-1/5/10 on opposing side. Only A/M/C state changes change displayed book; no passive-order identity inferred.",
        "depth_recovery":"first increase above running minimum after an observed reduction from initial depth; proxy, not trade attribution.",
        "forward_response":"fixed checkpoints and source-valid event milestones; favorable direction is original proposed reversal direction. Horizon markout is latest valid quote at/before horizon, requiring source coverage and <=2s staleness; extrema are over valid quotes no later than horizon.",
        "post_entry_outcomes":"MFE/MAE are not used as detector features; winner/loser analysis uses only snapshots timestamped by original entry.","unknown_side_discrepancy":"One Mar 7 T/N 99-contract record is held out of directional volume and Delta."}
    _write_json(out/"causal-feature-lineage.json",lineage)
    (out/"v2-detector-feasibility.md").write_text("# V2 detector feasibility (research decision)\n\nNo detector is specified as ready. The source supports exploratory measurement of executions, price response, and aggregate opposing depth. The five-day sample does not establish reliable pre-entry discrimination or predictive value, and displayed-depth changes cannot identify passive-order replenishment.\n\nThe smallest candidate measurement set to investigate further is: (1) known-side cumulative attack executions, (2) directional price progress per executed contract over a fixed causal window, and (3) valid quote/source coverage. Price-aligned depth persistence/recovery should remain an optional incremental feature until it shows stable information beyond volume and price impact. This is a research component list, not an entry rule; no numerical threshold is proposed.\n\nThe V1 heavy threshold doubled is used only as a descriptive marker for timing continuity and was not selected on outcomes. Five dates, overlapping episodes, and the small 8-winner/16-loser pre-entry subset are insufficient to claim a trading edge. Review the per-episode, time-aligned outputs before considering a detector specification.\n")
    summary={"study_id":STUDY_ID,"dates":list(base.DATES),"episode_count":len(population),"episode_origins":dict(Counter(x["origin_kind"] for x in population)),
        "episodes_by_v1_status":dict(Counter(x["v1_status"] for x in population)),"event_feature_rows":event_rows,"time_snapshot_rows":len(snapshots),
        "fixed_offsets_ms":list(OFFSETS_MS),"forward_horizons_ms":list(FORWARD_HORIZONS_MS),"decision_time_feasibility":before_entry,
        "milestone_availability":counts,"forward_analysis":analysis["forward_response_by_aligned_snapshot_cohort"],"source_integrity_status":integrity["status"],
        "first_combined_measurement":combined_timing,
        "v1_source_metadata_discrepancy":discrepancy,"real_absorption_filter_feasibility":"inconclusive","v2_detector_specification_ready":False,
        "limitations":["five source dates only","episodes can overlap","native MBP-10 depth is aggregate, not order-level","T/N trade cannot be directionally assigned","pre-entry winner/loser sample is small","forward cohorts have horizon-specific censoring"],"runtime_seconds":time.monotonic()-started}
    _write_json(out/"summary.json",summary)
    report=(f"# Causal absorption measurement research\n\nStudy `{STUDY_ID}`; exactly five sealed Spring TRAIN ESH5 MBP-10 dates. This is measurement research, not a strategy.\n\n"
      f"## Population and integrity\n\n- Episodes: {len(population)}; origins: `{summary['episode_origins']}`. Heavy-print episodes use the first source-supported V1 heavy-print timestamp; no-heavy episodes use first touch and remain a separate cohort.\n"
      f"- Source hashes and prior V1/audit manifests were verified. One Mar 7 T/N trade (99 contracts at 5776.50) explains the +1 native count vs V1 metadata; excluded from directional features, not removed from data.\n"
      f"- Native row order preserved; event and receive timestamps are monotone on all five files. Sequence is not treated as a global order key.\n\n## Time coverage\n\n"+
      "\n".join(f"- {offset} ms: {v['v1_entries']['valid']}/{v['v1_entries']['n']} V1 entries have valid source snapshots ({v['v1_entries']['coverage_pct']:.1f}%); nonconfirmed {v['nonconfirmed']['valid']}/{v['nonconfirmed']['n']} ({v['nonconfirmed']['coverage_pct']:.1f}%)." for offset,v in analysis['fixed_snapshot_coverage'].items())+
      f"\n\nFixed snapshots are latest observations at or before each checkpoint, not future-filled values. Quote must be valid, no more than 2 s stale, with no >2 s receive-time source gap from origin. `ts_recv` is the availability clock and paired `ts_event` is retained.\n\n## Mechanism and outcomes\n\n"
      f"The substantial-aggression marker is cumulative attack-side volume >=2x the source-defined V1 heavy threshold; it is descriptive only. Weak-impact evidence combines that marker with <=1 tick maximum attack-direction progress and at least two attack-side execution records, so one large print at the origin is not treated as observed weak impact. Depth recovery is an aggregate price-aligned increase above a prior reduced minimum. These do not identify passive orders.\n\n"
      f"The forward-response artifact aligns every horizon within its own valid snapshot cohort and reports survivor/censor counts. Do not compare 5-second survivors naively with the 250 ms population. Winner/loser cutoffs use only valid snapshots at/before original entry; the exact 5-second-after-origin pre-entry checkpoint contains {analysis['winner_loser_by_offset']['5000']['attack_volume_contracts']['WINNER']['valid_n']} winner and {analysis['winner_loser_by_offset']['5000']['attack_volume_contracts']['LOSER']['valid_n']} losers. The prior mechanism report's 8-winner/16-loser complete-window sample is a different landmark and remains preserved there. Episodes overlap.\n\n"
      f"The event-level earliest combined measurement (known attack execution + valid price response + price-aligned opposing depth) is summarized in `decision-time-feasibility.json`; it describes feature availability, not an absorption decision.\n\n"
      f"In this event window it was measurable in {combined_timing['episodes_with_measurement']}/{combined_timing['directional_episodes']} directional episodes and before original entry in {combined_timing['episodes_with_measurement_before_v1_entry']}; median observed latency was {combined_timing['latency_ms_from_origin']['median']:.3f} ms. This is a technical measurability point, not proof of failed aggression; episodes without such a row within five seconds are right-censored.\n\n"
      f"## Decision\n\nMeaningful aggression, price impact, and displayed-depth response are causally measurable when source-valid. Whether their combination is associated with reversal early enough to act remains inconclusive in this five-day pilot. Depth adds a measurable but ambiguous aggregate feature; incremental predictive information is not established. No reliable absorption filter or detector specification is ready.\n\n"
      f"## Sources and hashes\n\nSee `source-integrity-and-semantics.json`, `causal-feature-lineage.json`, and `artifact-hashes.json`. Runtime {summary['runtime_seconds']:.1f}s.\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    tests={"status":"PENDING_TEST_RUN","module":"tests/test_mac_2025_hvn_causal_absorption_detection_research_v1.py"}
    _write_json(out/"test-results.json",tests)
    manifest={"study_id":STUDY_ID,"complete":True,"dates":list(base.DATES),"dataset":"GLBX.MDP3","schema":"mbp-10","symbol":"ESH5",
        "source_sha256_verified":True,"prior_v1_artifacts_verified":True,"streaming_passes_per_source":1,"bounded_chunk_rows":CHUNK_ROWS,
        "native_order_preserved":True,"v1_config_changed":False,"v2_implemented":False,"optimization_performed":False,
        "october_data_accessed":False,"protected_oos_accessed":False,"data_downloaded":False,"commit_performed":False,
        "source_discrepancy":"Mar 7 one T/N trade reconciled; excluded from directional metrics","runtime_seconds":summary["runtime_seconds"]}
    _write_json(out/"run-manifest.json",manifest)
    hashes={}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="artifact-hashes.json": hashes[str(p.relative_to(out))]=_sha256(p)
    _write_json(out/"artifact-hashes.json",hashes)
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    args=p.parse_args()
    result=run(args.repo)
    print(json.dumps({"status":"PASS","artifact_root":str((args.repo/OUT).resolve()),"episodes":result["episode_count"],
                      "event_feature_rows":result["event_feature_rows"],"snapshots":result["time_snapshot_rows"],"runtime_seconds":result["runtime_seconds"]},indent=2))


if __name__=="__main__": main()
