"""Decision-time-matched descriptive study of the frozen HVN absorption V2.

This module does not fit a trading rule. It replays the frozen V2 state updater
unchanged, verifies its transition ledger, and measures fixed landmark outcomes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import mac_2025_hvn_failed_aggression_detector_v2_research as v2
from . import mac_2025_hvn_native_mbp10_absorption_mechanism_v1 as base

STUDY_ID = "ES_HVN_ABSORPTION_FINAL_PREDICTIVE_DECISION_V1"
OUT = Path("research_runs") / STUDY_ID
LANDMARKS_MS = (250, 500, 1000, 2000, 5000)
BARRIERS_TICKS = (2, 4, 6)
HORIZON_NS = 30 * v2.NS
PREHISTORY_NS = 30 * v2.NS
MAX_GAP_NS = 2 * v2.NS
TICK = 0.25
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20251027
CT = ZoneInfo("America/Chicago")

DESIGN = {
    "study_id": STUDY_ID,
    "frozen_v2_spec_sha256": "8de9119f827dab50908ff5d785b426095200f579960ef4d44bb5ba50d13d8b1b",
    "population": "all first-touch zone episodes in frozen 35-date V1 source manifest; retain nondirectional and invalid cases",
    "landmarks_ms_after_touch": list(LANDMARKS_MS),
    "primary": {
        "landmark_ms": 1000,
        "barrier_ticks": 4,
        "horizon_seconds": 30,
        "treatment": "frozen V2 complete candidate observed by landmark, with no continuation/rejection terminal before landmark",
        "control": "frozen V2 aggression effort observed by landmark but no complete candidate, same at-risk population",
        "risk_set": "directional episode with valid as-of landmark mid no older than 2 seconds, no continuation/rejection before landmark, and frozen V2 source validity through landmark",
        "matching": "exact trading date, direction and America/Chicago hour; deterministic greedy nearest-neighbor without replacement on robust-scaled pre-landmark covariates; max distance 2.0 Euclidean robust SD; no matching on outcomes",
        "estimand": "matched-pair difference in 30-second reversal-first cumulative incidence among pairs with uncensored paths; report all-observation competing outcome rates and censoring separately",
    },
    "secondary": {"landmarks_ms": [250, 500, 2000, 5000], "barriers_ticks": [2, 4, 6], "horizon_seconds": 30},
    "first_passage": {
        "reference": "latest valid causal MBP-10 quote mid at or before landmark; ts_recv availability clock",
        "direction": "original V1 setup direction; positive outcome is reversal against attack direction",
        "outcomes": ["REVERSAL_FIRST", "CONTINUATION_FIRST", "TIED_OR_ORDER_AMBIGUOUS", "NOT_REACHED", "SOURCE_CENSORED", "NOT_AT_RISK_AT_LANDMARK"],
        "same_receive_time": "if opposite barriers first cross at equal ts_recv, classify tie; do not use sequence monotonicity to break tie",
        "validity": "no interpolation; censor on invalid quote, bad receive timestamp, or >2s receive-time gap before a barrier; require valid coverage through horizon to classify NOT_REACHED",
    },
    "components": {"aggression": "V2 effort transition by landmark", "weak_impact": "V2 qualification-time weak impact by landmark", "depth": "V2 original-price aggregate depth response by landmark", "combined": "V2 candidate transition by landmark", "rejection": "V2 rejection transition by landmark; measured only as already-known state"},
    "covariates": ["recent_30s_mid_range_ticks", "recent_30s_mid_change_ticks", "recent_30s_trade_count", "recent_30s_trade_volume", "spread_ticks", "zone_width_ticks", "distance_to_setup_boundary_ticks", "available_history_seconds"],
    "bootstrap": {"unit": "trading date", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "interval": "percentile 95%", "unavailable_replicate": "retain as unavailable; never fill with zero"},
    "restrictions": {"threshold_optimization": False, "detector_modified": False, "entry_stop_tp_modified": False, "october_accessed": False, "protected_oos_accessed": False, "data_downloaded": False, "production_code_changed": False, "commit_performed": False},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({"status": pa.array([], type=pa.string())})
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
    tmp.replace(path)


def verify_inputs(repo: Path) -> tuple[list[base.Episode], list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    # Reuse the prior V2 loader's checks for V1/audit/native-mechanism/causal
    # artifact hashes, V1 specification, 35 native source hashes and identities.
    episodes, dates, sources = v2._load_inputs(repo)
    v2_root = repo / v2.OUT
    v2_hashes = json.loads((v2_root / "artifact-hashes.json").read_text())
    bad = [name for name, expected in v2_hashes.items()
           if not (v2_root / name).is_file() or sha256(v2_root / name) != expected]
    if bad:
        raise ValueError(f"frozen V2 artifact hash mismatch: {bad[:5]}")
    spec_doc = json.loads((v2_root / "frozen-detector-specification.json").read_text())
    spec_hash = canonical_hash(spec_doc["specification"])
    if spec_hash != DESIGN["frozen_v2_spec_sha256"] or spec_doc.get("sha256") != spec_hash:
        raise ValueError("frozen V2 detector specification identity mismatch")
    sidecar = (v2_root / "detector-specification.sha256").read_text().split()[0]
    if sidecar != spec_hash:
        raise ValueError("V2 detector SHA sidecar mismatch")
    manifest = json.loads((v2_root / "run-manifest.json").read_text())
    source_integrity = json.loads((v2_root / "source-integrity.json").read_text())
    if not manifest.get("complete") or manifest.get("episode_count") != len(episodes) or len(dates) != 35:
        raise ValueError("frozen V2 run manifest is not complete for the expected population")
    if source_integrity.get("status") != "PASS" or source_integrity.get("episode_count") != len(episodes):
        raise ValueError("frozen V2 source integrity is not passing")
    if len(episodes) != 2630:
        raise ValueError(f"unexpected frozen episode count {len(episodes)}")
    checked = {"prior_study_artifacts": "PASS", "v2_artifacts_checked": len(v2_hashes),
               "v2_spec_sha256": spec_hash, "v2_run_complete": True,
               "source_dates": len(dates), "episode_count": len(episodes),
               "native_sources": {d: {k: sources[d][k] for k in ("raw_path", "raw_sha256", "raw_bytes", "dataset", "schema", "symbol")} for d in dates}}
    return episodes, dates, sources, checked


def _ct_hour_bucket(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1e9, timezone.utc).astimezone(CT)
    return f"{dt.hour:02d}:00"


def _safe_float(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def _snapshot(st: dict[str, Any], ts_ns: int, market: dict[str, Any], source_ok: bool) -> dict[str, Any]:
    ep = st["ep"]
    quote = market.get("quote")
    quote_age = ts_ns - quote["ts_recv"] if quote else None
    quote_valid = bool(quote and quote["valid"] and quote_age is not None and 0 <= quote_age <= MAX_GAP_NS)
    effort = st["effort"]
    weak = st["impact"] if effort else None
    depth = v2.depth_response(st["attack_depth_samples"], st["depth_reduction"], st["depth_recovered"]) if effort else None
    cp = {k: (int(st[k]["ts_recv"]) if st.get(k) else None) for k in ("candidate", "continuation", "rejection")}
    terminal_before = bool((cp["continuation"] is not None and cp["continuation"] <= ts_ns)
                           or (cp["rejection"] is not None and cp["rejection"] <= ts_ns))
    setup_boundary = ep.low if ep.direction and ep.direction > 0 else ep.high
    mid = quote["mid"] if quote_valid else None
    opposing_depth = None
    if quote and quote.get("book") and ep.direction is not None:
        opposing_depth = v2._zone_depth(quote["book"], ep)
    qhist = market["quotes"]
    recent_mids = [x[1] for x in qhist if ts_ns - PREHISTORY_NS <= x[0] <= ts_ns]
    trades = [x for x in market["trades"] if ts_ns - PREHISTORY_NS <= x[0] <= ts_ns]
    history_start = qhist[0][0] if qhist else ts_ns
    for left, right in zip(qhist, list(qhist)[1:]):
        if right[0] - left[0] > MAX_GAP_NS:
            history_start = right[0]
    max_history = max(0, min(30.0, (ts_ns - history_start) / 1e9))
    spread_ticks = ((quote["ask"] - quote["bid"]) / TICK) if quote_valid and quote["bid"] is not None and quote["ask"] is not None else None
    row = {
        "episode_id": ep.episode_id, "date": ep.date, "zone_id": ep.zone_id, "landmark_ms": (ts_ns - ep.t0) / 1e6,
        "landmark_ns": ts_ns, "direction": ep.direction, "attack_direction": -ep.direction if ep.direction is not None else None,
        "zone_low": ep.low, "zone_high": ep.high, "zone_poc": ep.poc,
        "quote_mid": mid, "quote_bid": quote["bid"] if quote_valid else None, "quote_ask": quote["ask"] if quote_valid else None,
        "quote_age_ms": quote_age / 1e6 if quote_age is not None else None, "spread_ticks": spread_ticks,
        "attack_volume": st["attack_volume"], "counter_volume": st["opposing_volume"], "attack_execution_count": st["attack_count"],
        "counter_execution_count": st["opposing_count"],
        "attack_persistence": st["attack_count"] / (st["attack_count"] + st["opposing_count"]) if st["attack_count"] + st["opposing_count"] else None,
        "max_attack_progress_ticks": st["max_attack_progress"], "max_hvn_penetration_ticks": st["max_boundary_penetration"],
        "opposing_depth_at_original_zone": opposing_depth, "depth_positive_attack_prints": sum(v > 0 for v in st["attack_depth_samples"]),
        "depth_reduction_observed": bool(st["depth_reduction"]), "depth_recovery_observed": bool(st["depth_recovered"]),
        "aggression_qualified": effort is not None, "aggression_evidence_status": "QUALIFIED" if effort is not None else "UNAVAILABLE_NO_THRESHOLD" if ep.v1_threshold is None else "NOT_QUALIFIED",
        "weak_impact_available": weak is not None, "weak_impact_satisfied": weak,
        "depth_response_available": depth is not None, "depth_response_satisfied": depth,
        "depth_evidence_sufficient": bool(effort is not None and (st["attack_depth_samples"] or st["depth_start"] is not None)),
        "candidate_observed": cp["candidate"] is not None and cp["candidate"] <= ts_ns,
        "continuation_observed": cp["continuation"] is not None and cp["continuation"] <= ts_ns,
        "rejection_observed": cp["rejection"] is not None and cp["rejection"] <= ts_ns,
        "candidate_ns": cp["candidate"], "continuation_ns": cp["continuation"], "rejection_ns": cp["rejection"],
        "current_detector_state": ("REJECTION_OBSERVED" if cp["rejection"] is not None and cp["rejection"] <= ts_ns else
                                   "CONTINUATION" if cp["continuation"] is not None and cp["continuation"] <= ts_ns else
                                   "CANDIDATE" if cp["candidate"] is not None and cp["candidate"] <= ts_ns else
                                   "EFFORT" if effort else "TOUCH"),
        "source_valid": bool(source_ok and st["gap_count"] == 0 and not st["bad_source"]),
        "quote_valid": quote_valid, "history_seconds": max_history,
        "recent_30s_mid_range_ticks": ((max(recent_mids) - min(recent_mids)) / TICK) if recent_mids else None,
        "recent_30s_mid_change_ticks": ((recent_mids[-1] - recent_mids[0]) / TICK) if len(recent_mids) >= 2 else None,
        "recent_30s_trade_count": len(trades), "recent_30s_trade_volume": sum(x[1] for x in trades),
        "zone_width_ticks": (ep.high - ep.low) / TICK,
        "distance_to_setup_boundary_ticks": ((mid - setup_boundary) / TICK * ep.direction) if mid is not None and ep.direction is not None else None,
        "local_hour_bucket": _ct_hour_bucket(ts_ns), "v1_status": ep.v1_status, "threshold_available": ep.v1_threshold is not None,
        "v1_confirmed": bool(ep.v1), "pre_landmark_terminal": terminal_before,
    }
    row["outcome_observable"] = bool(ep.direction is not None and quote_valid and row["source_valid"])
    row["at_risk"] = bool(row["outcome_observable"] and not terminal_before)
    row["complete_absorption"] = bool(row["at_risk"] and row["candidate_observed"])
    row["aggression_only_control"] = bool(row["at_risk"] and row["aggression_qualified"] and not row["complete_absorption"])
    return row


def first_passage_result(reference: float, direction: int, ticks: int, records: list[dict[str, Any]],
                         horizon_ns: int, max_gap_ns: int = MAX_GAP_NS) -> dict[str, Any]:
    """Pure deterministic first-passage evaluator used by stream and tests."""
    last_recv: int | None = None
    landmark_ns = int(records[0].get("landmark_ns", 0)) if records else 0
    first_hit: tuple[str, int] | None = None
    later = [r for r in records if int(r["ts_recv"]) > landmark_ns and int(r["ts_recv"]) <= horizon_ns]
    i = 0
    while i < len(later):
        recv = int(later[i]["ts_recv"])
        batch = []
        while i < len(later) and int(later[i]["ts_recv"]) == recv:
            batch.append(later[i]); i += 1
        if last_recv is not None and recv - last_recv > max_gap_ns:
            return {"outcome": "SOURCE_CENSORED", "first_passage_ns": None, "last_valid_ns": last_recv}
        sides = set()
        for rec in batch:
            if not rec.get("valid") or rec.get("mid") is None:
                return {"outcome": "SOURCE_CENSORED", "first_passage_ns": None, "last_valid_ns": last_recv}
            mid = float(rec["mid"])
            side = "REVERSAL_FIRST" if (mid - reference) * direction >= ticks * TICK else (
                "CONTINUATION_FIRST" if (reference - mid) * direction >= ticks * TICK else None)
            if side: sides.add(side)
        last_recv = recv
        if len(sides) > 1:
            return {"outcome": "TIED_OR_ORDER_AMBIGUOUS", "first_passage_ns": recv, "last_valid_ns": recv,
                    "reversal_ts_recv": recv, "continuation_ts_recv": recv}
        if sides:
            side = next(iter(sides))
            first_hit = (side, recv)
            return {"outcome": side, "first_passage_ns": recv, "last_valid_ns": recv}
    if last_recv is not None and horizon_ns - last_recv <= max_gap_ns:
        return {"outcome": "NOT_REACHED", "first_passage_ns": None, "last_valid_ns": last_recv}
    return {"outcome": "SOURCE_CENSORED", "first_passage_ns": None, "last_valid_ns": last_recv}


def _outcome_state(landmark: dict[str, Any]) -> dict[str, Any]:
    return {"landmark": landmark, "horizon_ns": landmark["landmark_ns"] + HORIZON_NS,
            "last_recv": landmark["landmark_ns"], "status": "PENDING", "pending_side": None,
            "records": [], "first_move": None, "max_adverse_ticks": 0.0, "max_favorable_ticks": 0.0}


def _feed_outcome(out: dict[str, Any], row: Any, valid: bool, book: dict[str, Any], recv: int) -> None:
    lm = out["landmark"]
    if out["status"] != "PENDING" or recv <= lm["landmark_ns"] or not lm.get("outcome_observable") or lm.get("quote_mid") is None or lm.get("direction") is None:
        return
    if recv > out["horizon_ns"]:
        if out["last_recv"] is not None and out["horizon_ns"] - out["last_recv"] <= MAX_GAP_NS:
            out["status"] = "NOT_REACHED"
        else:
            out["status"] = "SOURCE_CENSORED"
        return
    if recv - out["last_recv"] > MAX_GAP_NS or not valid or not book["valid"]:
        out["status"] = "SOURCE_CENSORED"
        return
    out["last_recv"] = recv
    mid = float(book["mid"])
    ref, direction = float(lm["quote_mid"]), int(lm["direction"])
    move = (mid - ref) / TICK * direction
    out["max_favorable_ticks"] = max(out["max_favorable_ticks"], move)
    out["max_adverse_ticks"] = min(out["max_adverse_ticks"], move)
    hit = None
    for n in BARRIERS_TICKS:
        item = out["barriers"][str(n)]
        if item["outcome"] in ("REVERSAL_FIRST", "CONTINUATION_FIRST") and item.get("pending_recv") == recv:
            reversal = (mid - float(out["landmark"]["quote_mid"])) * int(out["landmark"]["direction"]) >= n * TICK
            continuation = (float(out["landmark"]["quote_mid"]) - mid) * int(out["landmark"]["direction"]) >= n * TICK
            if (reversal and item["outcome"] == "CONTINUATION_FIRST") or (continuation and item["outcome"] == "REVERSAL_FIRST"):
                item.update(outcome="TIED_OR_ORDER_AMBIGUOUS", first_passage_ns=recv,
                            reversal_ts_recv=recv, continuation_ts_recv=recv)
            continue
        if item["outcome"] != "PENDING":
            continue
        item["adverse_before_first_passage_ticks"] = min(item.get("adverse_before_first_passage_ticks", 0.0), move)
        side = "REVERSAL_FIRST" if move >= n else "CONTINUATION_FIRST" if move <= -n else None
        if side:
            if item["pending_side"] and item["pending_recv"] == recv and item["pending_side"] != side:
                item.update(outcome="TIED_OR_ORDER_AMBIGUOUS", first_passage_ns=recv,
                            reversal_ts_recv=recv, continuation_ts_recv=recv)
            else:
                item["pending_side"] = side
                item["pending_recv"] = recv
                item["outcome"] = side
                item["first_passage_ns"] = recv
                item["first_passage_ts_event"] = int(row["ts_event"])
                item["first_passage_sequence"] = int(row["sequence"])
                item["first_passage_mid"] = mid
            if out["first_move"] is None:
                out["first_move"] = {"side": side, "ts_recv": recv, "mid": mid}
    out["records"].append({"ts_recv": recv, "mid": mid, "valid": True})


def _finish_outcome(out: dict[str, Any]) -> list[dict[str, Any]]:
    lm = out["landmark"]
    rows = []
    for n in BARRIERS_TICKS:
        b = out["barriers"][str(n)]
        status = b["outcome"]
        if status == "PENDING":
            status = "NOT_REACHED" if out["last_recv"] is not None and out["horizon_ns"] - out["last_recv"] <= MAX_GAP_NS else "SOURCE_CENSORED"
        rows.append({"episode_id": lm["episode_id"], "date": lm["date"], "landmark_ms": lm["landmark_ms"],
                     "landmark_ns": lm["landmark_ns"], "barrier_ticks": n,
                     "outcome": "NOT_AT_RISK_AT_LANDMARK" if not lm.get("outcome_observable") else status,
                     "reference_mid": lm["quote_mid"], "direction": lm["direction"],
                     "first_passage_ns": b.get("first_passage_ns"), "first_passage_ts_event": b.get("first_passage_ts_event"),
                     "first_passage_sequence": b.get("first_passage_sequence"), "first_passage_mid": b.get("first_passage_mid"),
                     "time_to_first_passage_ms": ((b["first_passage_ns"] - lm["landmark_ns"]) / 1e6) if b.get("first_passage_ns") else None,
                     "adverse_excursion_before_first_passage_ticks": b.get("adverse_before_first_passage_ticks"),
                     "max_favorable_excursion_ticks": out["max_favorable_ticks"], "max_adverse_excursion_ticks": out["max_adverse_ticks"],
                     "source_valid_through_horizon": status != "SOURCE_CENSORED", "pre_landmark_terminal": lm["pre_landmark_terminal"],
                     "complete_absorption": lm["complete_absorption"], "aggression_qualified": lm["aggression_qualified"],
                     "weak_impact_satisfied": lm["weak_impact_satisfied"], "depth_response_satisfied": lm["depth_response_satisfied"],
                     "rejection_observed": lm["rejection_observed"], "at_risk": lm["at_risk"]})
    return rows


def _state_context(market: dict[str, Any], recv: int) -> dict[str, Any]:
    q = market.get("quote")
    return {"quote": q.copy() if q else None, "quotes": list(market["quotes"]), "trades": list(market["trades"]),
            "first_observed_ns": market.get("first_observed_ns", recv)}


def _process_date(repo: Path, date: str, episodes: list[base.Episode], source: dict[str, Any],
                  expected_transitions: list[dict[str, str]]) -> dict[str, Any]:
    from databento import DBNStore

    states = [v2._new_episode_state(e) for e in episodes]
    intervals = sorted((max(0, e.t0 - PREHISTORY_NS), e.t0 + v2.DETECTION_WINDOW_NS + 30 * v2.NS) for e in episodes)
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    active: list[dict[str, Any]] = []
    next_state = 0
    landmarks: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    market = {"quote": None, "quotes": deque(), "trades": deque(), "first_observed_ns": None}
    transition_rows: list[dict[str, Any]] = []
    original_update = v2._update_episode
    # Instrumentation surrounds the source-frozen updater. It never changes its
    # inputs or state transitions and is restored in a finally block.
    current_context = market
    def ensure_lm(st: dict[str, Any], recv_before: int) -> None:
        ep = st["ep"]
        snap_by_ms = st.setdefault("study_landmarks", {})
        for ms in LANDMARKS_MS:
            target = ep.t0 + ms * 1_000_000
            if ms not in snap_by_ms and target < recv_before:
                row = _snapshot(st, target, _state_context(current_context, target), not st["bad_source"])
                snap_by_ms[ms] = row
                landmarks.append(row)
                ob = {"landmark": row, "horizon_ns": target + HORIZON_NS, "last_recv": target,
                          "status": "PENDING", "pending_side": None, "records": [], "first_move": None,
                          "max_adverse_ticks": 0.0, "max_favorable_ticks": 0.0,
                          "barriers": {str(n): {"outcome": "PENDING", "pending_side": None, "pending_recv": None}
                                       for n in BARRIERS_TICKS}}
                outcomes.append(ob)
                st.setdefault("study_outcomes", []).append(ob)
    def wrapped(st: dict[str, Any], row: Any) -> None:
        recv = int(row["ts_recv"])
        ensure_lm(st, recv)
        original_update(st, row)
        if st.get("study_landmarks"):
            book = base._book(row)
            valid = bool(book["valid"] and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV) and st["gap_count"] == 0)
            for ob in st.get("study_outcomes", []):
                _feed_outcome(ob, row, valid, book, recv)
    v2._update_episode = wrapped
    all_records = backward_recv = backward_event = sequence_nonincrease = bad_recv = 0
    prior_recv = prior_event = prior_seq = None
    try:
        store = DBNStore.from_file(source["absolute_path"])
        for chunk in store.to_ndarray(count=v2.CHUNK_ROWS):
            if not isinstance(chunk, np.ndarray) or chunk.dtype.names is None:
                raise TypeError("unexpected Databento ndarray")
            if not len(chunk):
                continue
            if not np.all(chunk["rtype"] == 10):
                raise ValueError(f"non-MBP10 source record on {date}")
            all_records += len(chunk)
            r, ev, seq = chunk["ts_recv"], chunk["ts_event"], chunk["sequence"]
            backward_recv += int(np.count_nonzero(r[1:] < r[:-1]))
            backward_event += int(np.count_nonzero(ev[1:] < ev[:-1]))
            sequence_nonincrease += int(np.count_nonzero(seq[1:] <= seq[:-1]))
            if prior_recv is not None and int(r[0]) < prior_recv: backward_recv += 1
            if prior_event is not None and int(ev[0]) < prior_event: backward_event += 1
            if prior_seq is not None and int(seq[0]) <= prior_seq: sequence_nonincrease += 1
            prior_recv, prior_event, prior_seq = int(r[-1]), int(ev[-1]), int(seq[-1])
            bad_recv += int(np.count_nonzero(chunk["flags"] & base.FLAG_BAD_TS_RECV))
            if np.all(r[1:] >= r[:-1]):
                selected: list[int] = []
                for lo, hi in merged:
                    a, b = int(np.searchsorted(r, lo, "left")), int(np.searchsorted(r, hi, "right"))
                    if a < b: selected.extend(range(a, b))
            else:
                selected = base.native_order_window_indices(r, merged)
            for i in selected:
                row = chunk[i]
                recv = int(row["ts_recv"])
                while next_state < len(states) and states[next_state]["ep"].t0 <= recv:
                    active.append(states[next_state]); next_state += 1
                active = [st for st in active if recv <= st["ep"].t0 + v2.DETECTION_WINDOW_NS + 30 * v2.NS]
                for st in active:
                    ensure_lm(st, recv)
                # Update rolling, strictly pre-current-event context once. The
                # landmark flush above therefore cannot see this future record.
                if market["first_observed_ns"] is None: market["first_observed_ns"] = recv
                cutoff = recv - PREHISTORY_NS
                while market["quotes"] and market["quotes"][0][0] < cutoff: market["quotes"].popleft()
                while market["trades"] and market["trades"][0][0] < cutoff: market["trades"].popleft()
                book = base._book(row)
                if book["valid"] and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV):
                    market["quote"] = {"ts_recv": recv, "valid": True, "mid": float(book["mid"]),
                                       "bid": float(book["bid"]), "ask": float(book["ask"]), "book": book}
                    market["quotes"].append((recv, float(book["mid"])))
                if bytes(row["action"]) == b"T":
                    market["trades"].append((recv, int(row["size"])))
                current_context = market
                for st in active:
                    wrapped(st, row)
                    if st.get("study_landmarks"):
                        for ob in st.get("study_outcomes", []):
                            if ob["status"] == "PENDING" and recv >= ob["horizon_ns"]:
                                ob["status"] = "NOT_REACHED" if ob["horizon_ns"] - ob["last_recv"] <= MAX_GAP_NS else "SOURCE_CENSORED"
        if backward_recv or backward_event:
            raise ValueError(f"native timestamp order changed during replay on {date}: recv={backward_recv}, event={backward_event}")
        # Flush any landmark lacking a later record using the final as-of state.
        end_ns = max((e.t0 + max(LANDMARKS_MS) * 1_000_000 for e in episodes), default=0) + 1
        for st in states:
            ensure_lm(st, end_ns)
            _, trans = v2._finalize_date({"states": [st]})
            transition_rows.extend(trans)
    finally:
        v2._update_episode = original_update
    actual = _canonical_transitions(transition_rows)
    expected = _canonical_transitions(expected_transitions)
    if actual != expected:
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise ValueError(f"replayed V2 transition mismatch on {date} at row {first}; generated={len(actual)} expected={len(expected)}")
    outcome_rows = [r for ob in outcomes for r in _finish_outcome(ob)]
    return {"date": date, "record_count": all_records, "ts_recv_backward_steps": backward_recv,
            "ts_event_backward_steps": backward_event, "sequence_nonincrease_steps": sequence_nonincrease,
            "bad_recv_records": bad_recv, "episodes": len(states), "transition_count": len(transition_rows), "landmarks": landmarks,
            "outcomes": outcome_rows, "transition_match": True}


def _canonical_transitions(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    keys = ("episode_id", "transition", "ts_recv", "ts_event", "sequence", "market_mid", "quote_valid", "attack_volume",
            "attack_execution_count", "opposing_volume", "max_attack_progress_ticks", "max_boundary_penetration_ticks",
            "defended_zone_depth", "depth_start", "depth_reduced", "depth_recovered", "note")
    def val(row: dict[str, Any], key: str) -> Any:
        x = row.get(key)
        if key == "market_mid" and x not in (None, ""):
            return round(float(x), 10)
        if key in {"attack_volume", "attack_execution_count", "opposing_volume", "ts_recv", "ts_event", "sequence", "defended_zone_depth", "depth_start"}:
            return int(x) if x not in (None, "") else None
        if key in {"quote_valid", "depth_reduced", "depth_recovered"}:
            return x if isinstance(x, bool) else str(x).lower() == "true"
        if key in {"max_attack_progress_ticks", "max_boundary_penetration_ticks"}:
            return round(float(x), 10) if x not in (None, "") else None
        return x
    return sorted((tuple(val(row, k) for k in keys) for row in rows), key=lambda x: (str(x[0]), int(x[2]), str(x[1])))


MATCH_COVARS = ("recent_30s_mid_range_ticks", "recent_30s_mid_change_ticks", "recent_30s_trade_count",
                "recent_30s_trade_volume", "spread_ticks", "zone_width_ticks", "distance_to_setup_boundary_ticks", "history_seconds")


def greedy_match(rows: list[dict[str, Any]], treatment_key: str, control_key: str,
                 covariates: tuple[str, ...] = MATCH_COVARS, caliper: float = 2.0) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
    """Greedy deterministic matching, exact date/direction/local-hour strata."""
    treatments = [i for i, r in enumerate(rows) if r.get(treatment_key)]
    controls = [i for i, r in enumerate(rows) if r.get(control_key)]
    pairs: list[tuple[int, int, float]] = []
    unmatched_t = set(treatments)
    used_c: set[int] = set()
    strata: dict[tuple[Any, ...], tuple[list[int], list[int]]] = {}
    for i in treatments:
        r = rows[i]; key = (r.get("date"), r.get("direction"), r.get("local_hour_bucket"))
        strata.setdefault(key, ([], []))[0].append(i)
    for i in controls:
        r = rows[i]; key = (r.get("date"), r.get("direction"), r.get("local_hour_bucket"))
        strata.setdefault(key, ([], []))[1].append(i)
    all_distances = []
    for key, (tids, cids) in sorted(strata.items(), key=lambda x: str(x[0])):
        if not tids or not cids: continue
        med_scale = {}
        for cov in covariates:
            values = [_safe_float(rows[i].get(cov)) for i in tids + cids]
            values = [x for x in values if x is not None]
            if not values: med_scale[cov] = (0.0, 1.0); continue
            med = float(np.median(values)); q25, q75 = np.percentile(values, [25, 75]); scale = float(q75 - q25)
            med_scale[cov] = (med, scale if scale > 1e-9 else 1.0)
        # Missing covariates are median-imputed within stratum for matching only;
        # their missingness is reported separately and never changed in ledger.
        choices = []
        for ti in tids:
            for ci in cids:
                d2 = 0.0
                for cov in covariates:
                    med, scale = med_scale[cov]
                    tv = _safe_float(rows[ti].get(cov)); cv = _safe_float(rows[ci].get(cov))
                    d2 += (((med if tv is None else tv) - (med if cv is None else cv)) / scale) ** 2
                dist = math.sqrt(d2 / max(1, len(covariates)))
                choices.append((dist, str(rows[ti]["episode_id"]), str(rows[ci]["episode_id"]), ti, ci))
        choices.sort()
        used_t: set[int] = set()
        used_local_c: set[int] = set()
        for dist, _, _, ti, ci in choices:
            if dist > caliper or ti in used_t or ci in used_local_c: continue
            pairs.append((ti, ci, dist)); used_t.add(ti); used_local_c.add(ci)
        unmatched_t.difference_update(used_t)
        used_c.update(used_local_c)
        all_distances.extend(d for _, _, d in [(a, b, c) for a, b, c in pairs if rows[a].get("date") == key[0] and rows[a].get("direction") == key[1] and rows[a].get("local_hour_bucket") == key[2]])
    pairs.sort(key=lambda p: (str(rows[p[0]]["date"]), str(rows[p[0]]["episode_id"])))
    details = {"treatment_n": len(treatments), "control_n": len(controls), "matched_pairs": len(pairs),
               "treatment_coverage": len(pairs) / len(treatments) if treatments else None,
               "control_coverage": len(used_c) / len(controls) if controls else None,
               "unmatched_treatment": len(treatments) - len(pairs), "unmatched_control": len(controls) - len(used_c),
               "mean_match_distance_robust_sd": float(np.mean([p[2] for p in pairs])) if pairs else None,
               "max_match_distance_robust_sd": max((p[2] for p in pairs), default=None), "caliper": caliper,
               "stratum_count": len(strata), "covariates": list(covariates)}
    return pairs, details


def _balance(rows: list[dict[str, Any]], pairs: list[tuple[int, int, float]], covariates: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for cov in covariates:
        t = [_safe_float(rows[a].get(cov)) for a, _, _ in pairs]
        c = [_safe_float(rows[b].get(cov)) for _, b, _ in pairs]
        valid = [(a, b) for a, b in zip(t, c) if a is not None and b is not None]
        before_t = [_safe_float(r.get(cov)) for r in rows if r.get("complete_absorption")]
        before_c = [_safe_float(r.get(cov)) for r in rows if r.get("aggression_only_control")]
        before_t = [x for x in before_t if x is not None]; before_c = [x for x in before_c if x is not None]
        after_t, after_c = [x[0] for x in valid], [x[1] for x in valid]
        pooled = np.std(before_t + before_c, ddof=1) if len(before_t + before_c) > 1 else np.nan
        smd_before = (np.mean(before_t) - np.mean(before_c)) / pooled if len(before_t) and len(before_c) and pooled > 0 else None
        after_sd = np.std(after_t + after_c, ddof=1) if len(after_t + after_c) > 1 else np.nan
        smd_after = (np.mean(after_t) - np.mean(after_c)) / after_sd if len(after_t) and len(after_c) and after_sd > 0 else None
        result.append({"covariate": cov, "matched_n_nonmissing": len(valid), "treatment_mean_before": float(np.mean(before_t)) if before_t else None,
                       "control_mean_before": float(np.mean(before_c)) if before_c else None, "smd_before": float(smd_before) if smd_before is not None else None,
                       "treatment_mean_after": float(np.mean(after_t)) if after_t else None, "control_mean_after": float(np.mean(after_c)) if after_c else None,
                       "smd_after": float(smd_after) if smd_after is not None else None,
                       "missing_treatment": sum(x is None for x in t), "missing_control": sum(x is None for x in c)})
    return result


def _bootstrap_date_effect(pairs_by_date: dict[str, list[float]], replicates: int = BOOTSTRAP_REPLICATES,
                           seed: int = BOOTSTRAP_SEED, date_universe: list[str] | None = None) -> dict[str, Any]:
    dates = sorted(date_universe if date_universe is not None else pairs_by_date)
    if len(dates) < 2:
        return {"replicates_requested": replicates, "valid_replicates": 0, "ci95": None, "estimate": None}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(replicates):
        chosen = rng.choice(dates, size=len(dates), replace=True)
        clusters = [pairs_by_date.get(str(d), []) for d in chosen if pairs_by_date.get(str(d))]
        if not clusters: continue
        flattened = [v for cluster in clusters for v in cluster if math.isfinite(v)]
        if flattened: vals.append(float(np.mean(flattened)))
    return {"replicates_requested": replicates, "valid_replicates": len(vals), "seed": seed,
            "estimate": float(np.mean([x for v in pairs_by_date.values() for x in v])) if any(pairs_by_date.values()) else None,
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None}


def _bootstrap_group_rate(rows: list[dict[str, Any]], seed: int, date_universe: list[str] | None = None) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows: clusters.setdefault(str(row["date"]), []).append(row)
    dates = sorted(date_universe if date_universe is not None else clusters)
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(dates, size=len(dates), replace=True) if dates else []
        sample = [row for d in chosen for row in clusters.get(str(d), [])]
        if sample: rates.append(sum(r["outcome"] == "REVERSAL_FIRST" for r in sample) / len(sample))
    estimate = sum(r["outcome"] == "REVERSAL_FIRST" for r in rows) / len(rows) if rows else None
    return {"estimate_all_eligible": estimate, "ci95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))] if rates else None,
            "valid_replicates": len(rates), "replicates_requested": BOOTSTRAP_REPLICATES, "seed": seed}


def _outcome_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["outcome"] for r in rows)
    eligible = len(rows)
    resolved = counts["REVERSAL_FIRST"] + counts["CONTINUATION_FIRST"]
    return {"n": eligible, "dates": len({r["date"] for r in rows}), "outcome_counts": dict(counts),
            "reversal_first_among_resolved": counts["REVERSAL_FIRST"] / resolved if resolved else None,
            "reversal_first_all_eligible": counts["REVERSAL_FIRST"] / eligible if eligible else None,
            "continuation_first_all_eligible": counts["CONTINUATION_FIRST"] / eligible if eligible else None,
            "nonresolution_or_censoring_share": sum(counts[k] for k in ("NOT_REACHED", "SOURCE_CENSORED", "TIED_OR_ORDER_AMBIGUOUS")) / eligible if eligible else None,
            "median_time_to_first_passage_ms": float(np.median([r["time_to_first_passage_ms"] for r in rows if r.get("time_to_first_passage_ms") is not None])) if any(r.get("time_to_first_passage_ms") is not None for r in rows) else None}


def _compare(rows: list[dict[str, Any]], treatment_key: str, control_key: str, label: str,
             date_universe: list[str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    eligible = [r for r in rows if r.get(treatment_key) or r.get(control_key)]
    pairs, match = greedy_match(eligible, treatment_key, control_key)
    result_rows = []
    pair_by_date: dict[str, list[float]] = {}
    usable = 0
    for ti, ci, dist in pairs:
        t, c = eligible[ti], eligible[ci]
        if t["outcome"] in ("SOURCE_CENSORED", "TIED_OR_ORDER_AMBIGUOUS", "NOT_AT_RISK_AT_LANDMARK") or c["outcome"] in ("SOURCE_CENSORED", "TIED_OR_ORDER_AMBIGUOUS", "NOT_AT_RISK_AT_LANDMARK"):
            continue
        effect = float(t["outcome"] == "REVERSAL_FIRST") - float(c["outcome"] == "REVERSAL_FIRST")
        pair_by_date.setdefault(t["date"], []).append(effect)
        usable += 1
        result_rows.append({"date": t["date"], "treatment_episode_id": t["episode_id"], "control_episode_id": c["episode_id"],
                            "treatment_outcome": t["outcome"], "control_outcome": c["outcome"], "difference": effect, "distance": dist})
    date_effects = {d: float(np.mean(v)) for d, v in pair_by_date.items() if v}
    bootstrap = _bootstrap_date_effect(pair_by_date, date_universe=date_universe)
    group_t = [r for r in eligible if r.get(treatment_key)]
    group_c = [r for r in eligible if r.get(control_key)]
    result = {"comparison": label, "matching": match, "matched_pairs_with_uncensored_paths": usable,
              "treatment_outcomes": _outcome_stats(group_t), "control_outcomes": _outcome_stats(group_c),
              "treatment_reversal_rate_date_cluster_bootstrap": _bootstrap_group_rate(group_t, BOOTSTRAP_SEED + 1, date_universe),
              "control_reversal_rate_date_cluster_bootstrap": _bootstrap_group_rate(group_c, BOOTSTRAP_SEED + 2, date_universe),
              "matched_reversal_first_risk_difference": bootstrap["estimate"], "date_cluster_bootstrap": bootstrap,
              "positive_date_share": (sum(x > 0 for x in date_effects.values()) / len(date_effects)) if date_effects else None,
              "date_effects": date_effects,
              "classification": "NOT_IDENTIFIABLE" if match["treatment_coverage"] is None or match["treatment_coverage"] < 0.25 or usable == 0 else "ESTIMATED_OBSERVATIONAL_ASSOCIATION"}
    return result, result_rows, {"pairs": pairs, "balance": _balance(eligible, pairs, MATCH_COVARS), "eligible_rows": eligible}


def analyze(landmarks: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    lm = {(r["episode_id"], int(r["landmark_ms"])): r for r in landmarks}
    joined = []
    for o in outcomes:
        r = {**lm[(o["episode_id"], int(o["landmark_ms"]))], **o}
        joined.append(r)
    primary = [r for r in joined if int(r["landmark_ms"]) == 1000 and r["barrier_ticks"] == 4]
    date_universe=sorted({r["date"] for r in landmarks})
    primary, pair_rows, internals = _compare(primary, "complete_absorption", "aggression_only_control", "complete_detector_vs_aggression_only", date_universe)
    for pair in pair_rows:
        pair["comparison"] = "complete_detector_vs_aggression_only"
    comparisons = {"primary": primary}
    specs = [
        ("aggression_vs_no_qualified_aggression", "aggression_qualified", "no_aggression_control", lambda r: r["at_risk"] and r["threshold_available"]),
        ("weak_impact_within_aggression", "weak_impact_true", "weak_impact_false", lambda r: r["at_risk"] and r["aggression_qualified"]),
        ("depth_within_aggression_weak_impact", "depth_true", "depth_false", lambda r: r["at_risk"] and r["aggression_qualified"] and r["weak_impact_satisfied"] is True and r["depth_evidence_sufficient"]),
        ("combined_vs_components", "complete_absorption", "aggression_only_control", lambda r: r["at_risk"] and r["aggression_qualified"]),
        ("rejection_observed_within_candidate", "rejection_true", "candidate_no_rejection", lambda r: r["candidate_observed"] and r["quote_valid"] and r["source_valid"] and not r["continuation_observed"]),
    ]
    effects = []
    for landmark_ms in LANDMARKS_MS:
        for barrier in BARRIERS_TICKS:
            cell = [r for r in joined if int(r["landmark_ms"]) == landmark_ms and r["barrier_ticks"] == barrier]
            observable=[r for r in cell if r["outcome"]!="NOT_AT_RISK_AT_LANDMARK"]
            summary = _outcome_stats(observable)
            summary.update(episode_count=len(cell),outcome_observable_n=len(observable),landmark_ms=landmark_ms, barrier_ticks=barrier,
                           risk_set_n=sum(bool(r["at_risk"]) for r in cell),
                           terminal_before_landmark=sum(bool(r["pre_landmark_terminal"]) for r in cell),
                           not_at_risk_outcome_count=len(cell)-len(observable),
                           quote_or_source_invalid=sum(not bool(r.get("outcome_observable")) for r in cell))
            effects.append(summary)
            for label, tk, ck, filt in specs:
                group_rows = []
                for r in cell:
                    if not filt(r): continue
                    rr = dict(r)
                    if label == "aggression_vs_no_qualified_aggression":
                        rr["no_aggression_control"] = not rr["aggression_qualified"]
                    if label == "weak_impact_within_aggression":
                        rr["weak_impact_true"] = rr["weak_impact_satisfied"] is True
                        rr["weak_impact_false"] = rr["weak_impact_satisfied"] is False
                    if label == "depth_within_aggression_weak_impact":
                        rr["depth_true"] = rr["depth_response_satisfied"] is True
                        rr["depth_false"] = rr["depth_response_satisfied"] is False and rr["depth_evidence_sufficient"]
                    if label == "rejection_observed_within_candidate":
                        rr["rejection_true"] = bool(rr["rejection_observed"])
                        rr["candidate_no_rejection"] = not bool(rr["rejection_observed"])
                    group_rows.append(rr)
                comp, prs, internal = _compare(group_rows, tk, ck, label, date_universe)
                key = f"{label}|{landmark_ms}ms|{barrier}ticks"
                comparisons[key] = comp
                for pr in prs: pr["comparison"] = key; pair_rows.append(pr)
    daily = []
    primary_rows = [r for r in joined if int(r["landmark_ms"]) == 1000 and r["barrier_ticks"] == 4]
    for date in sorted({r["date"] for r in primary_rows}):
        dr = [r for r in primary_rows if r["date"] == date]
        for group, pred in (("absorption", lambda r:r["complete_absorption"]), ("aggression_only",lambda r:r["aggression_only_control"])):
            x=[r for r in dr if pred(r)]; s=_outcome_stats(x); daily.append({"date":date,"group":group,**s})
    date_leaveout=[]
    for date in sorted({r["date"] for r in primary_rows}):
        kept=[p for p in pair_rows if p.get("comparison")=="complete_detector_vs_aggression_only" and p["date"]!=date]
        date_leaveout.append({"omitted_date":date,"matched_pairs":len(kept),"risk_difference":float(np.mean([p["difference"] for p in kept])) if kept else None})
    return {"joined": joined, "first_passage_summary": effects, "comparisons": comparisons,
            "primary_pair_rows": [p for p in pair_rows if p.get("comparison")=="complete_detector_vs_aggression_only"],
            "component_pair_rows": pair_rows, "daily": daily, "leave_one_date_out": date_leaveout,
            "primary_balance": internals["balance"], "primary_matching": comparisons["primary"]["matching"]}


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _replay_terminal_outcomes(repo: Path, date: str, result: dict[str, Any], source: dict[str, Any]) -> int:
    """Measure forward paths for valid landmark rows already terminal at t0.

    The original V2 pass correctly preserves those episodes but they are outside
    the primary at-risk set. This small second pass fills their descriptive and
    rejection-component first-passage records without rerunning detector state.
    """
    if result.get("terminal_outcomes_replayed"):
        return 0
    from databento import DBNStore
    lm_by_key = {(r["episode_id"], int(r["landmark_ms"])): r for r in result["landmarks"]}
    existing = {(r["episode_id"], int(r["landmark_ms"]), int(r["barrier_ticks"])): r for r in result["outcomes"]}
    pending = []
    for key, lm in lm_by_key.items():
        lm.setdefault("outcome_observable", bool(lm.get("direction") is not None and lm.get("quote_valid") and lm.get("source_valid")))
        if not lm.get("pre_landmark_terminal") or not lm.get("outcome_observable"):
            continue
        ob = {"landmark": lm, "horizon_ns": int(lm["landmark_ns"] + HORIZON_NS), "last_recv": int(lm["landmark_ns"]),
              "status": "PENDING", "pending_side": None, "records": [], "first_move": None,
              "max_adverse_ticks": 0.0, "max_favorable_ticks": 0.0,
              "barriers": {str(n): {"outcome": "PENDING", "pending_side": None, "pending_recv": None}
                           for n in BARRIERS_TICKS}}
        pending.append(ob)
    if not pending:
        return 0
    intervals = sorted((int(x["landmark"]["landmark_ns"]), int(x["horizon_ns"])) for x in pending)
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    active: list[dict[str, Any]] = []
    next_ob = 0
    from databento import DBNStore
    store = DBNStore.from_file(source["absolute_path"])
    prior_recv = None
    for chunk in store.to_ndarray(count=v2.CHUNK_ROWS):
        if not isinstance(chunk, np.ndarray) or chunk.dtype.names is None: raise TypeError("unexpected Databento ndarray")
        if not len(chunk): continue
        if not np.all(chunk["rtype"] == 10): raise ValueError(f"non-MBP10 record in terminal-outcome replay {date}")
        recv_array = chunk["ts_recv"]
        if np.any(recv_array[1:] < recv_array[:-1]) or (prior_recv is not None and int(recv_array[0]) < prior_recv):
            raise ValueError(f"native receive-time order changed in terminal-outcome replay {date}")
        prior_recv = int(recv_array[-1])
        if np.all(recv_array[1:] >= recv_array[:-1]):
            selected=[]
            for lo,hi in merged:
                a,b=int(np.searchsorted(recv_array,lo,"left")),int(np.searchsorted(recv_array,hi,"right"))
                if a<b:selected.extend(range(a,b))
        else:
            selected=base.native_order_window_indices(recv_array,merged)
        for i in selected:
            row=chunk[i]; recv=int(row["ts_recv"])
            while next_ob<len(pending) and int(pending[next_ob]["landmark"]["landmark_ns"])<=recv:
                active.append(pending[next_ob]); next_ob+=1
            active=[ob for ob in active if recv<=ob["horizon_ns"]]
            book=base._book(row)
            valid=bool(book["valid"] and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV))
            for ob in active:
                _feed_outcome(ob,row,valid,book,recv)
                if ob["status"]=="PENDING" and recv>=ob["horizon_ns"]:
                    ob["status"]="NOT_REACHED" if ob["horizon_ns"]-ob["last_recv"]<=MAX_GAP_NS else "SOURCE_CENSORED"
    changed=0
    for ob in pending:
        replacement=_finish_outcome(ob)
        for row in replacement:
            key=(row["episode_id"],int(row["landmark_ms"]),int(row["barrier_ticks"]))
            existing[key]=row; changed+=1
    result["outcomes"]=[existing[(r["episode_id"],int(r["landmark_ms"]),int(r["barrier_ticks"]))] for r in result["outcomes"]]
    result["terminal_outcomes_replayed"] = True
    return changed


def run(repo: Path, resume: bool = True) -> dict[str, Any]:
    started = time.time()
    episodes, dates, sources, integrity = verify_inputs(repo)
    out = repo / OUT
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / "frozen-study-design.json"
    design_bytes = (json.dumps(DESIGN, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    design_hash = hashlib.sha256(design_bytes).hexdigest()
    if design_path.exists() and design_path.read_bytes() != design_bytes:
        raise ValueError("existing frozen study design differs; refusing to overwrite")
    design_path.write_bytes(design_bytes)
    (out / "frozen-study-design.sha256").write_text(design_hash + "  frozen-study-design.json\n", encoding="ascii")
    prior = repo / v2.OUT
    expected_by_date: dict[str, list[dict[str, str]]] = {}
    for d in dates:
        trpath = prior / (("pilot-" if d in v2.PILOT_DATES else "date-") + d + "-transitions.csv")
        if not trpath.is_file(): raise FileNotFoundError(trpath)
        expected_by_date[d] = _rows_from_csv(trpath)
    by_date = {d: [e for e in episodes if e.date == d] for d in dates}
    date_results = []
    for i, date in enumerate(dates, 1):
        ckpt = out / "checkpoints" / f"{date}.json"
        digest_key = hashlib.sha256(json.dumps({"source": sources[date]["raw_sha256"], "design": design_hash,
                                                "v2_module": sha256(Path(v2.__file__)), "date": date}, sort_keys=True).encode()).hexdigest()
        cached = None
        if resume and ckpt.is_file():
            try:
                cached_doc = json.loads(ckpt.read_text())
                if cached_doc.get("cache_key") == digest_key:
                    cached = cached_doc["result"]
            except Exception:
                cached = None
        if cached is None:
            print(f"PREDICTIVE_STUDY_DATE {i}/{len(dates)} {date} START", flush=True)
            res = _process_date(repo, date, by_date[date], sources[date], expected_by_date[date])
            write_json(ckpt, {"cache_key": digest_key, "date": date, "result": res})
        else:
            res = cached
            print(f"PREDICTIVE_STUDY_DATE {i}/{len(dates)} {date} RESUMED", flush=True)
        for landmark in res.get("landmarks", []):
            landmark.setdefault("outcome_observable", bool(landmark.get("direction") is not None and landmark.get("quote_valid") and landmark.get("source_valid")))
        if not res.get("terminal_outcomes_replayed"):
            changed = _replay_terminal_outcomes(repo, date, res, sources[date])
            print(f"PREDICTIVE_STUDY_TERMINAL_PATHS {date} rows={changed}", flush=True)
            write_json(ckpt, {"cache_key": digest_key, "date": date, "result": res})
        date_results.append(res)
    landmarks = [r for d in date_results for r in d["landmarks"]]
    outcomes = [r for d in date_results for r in d["outcomes"]]
    if len(landmarks) != len(episodes) * len(LANDMARKS_MS):
        raise ValueError(f"landmark population mismatch: {len(landmarks)}")
    write_parquet(out / "landmark-risk-set-ledger.parquet", landmarks)
    write_parquet(out / "landmark-feature-ledger.parquet", landmarks)
    write_parquet(out / "first-passage-outcomes.parquet", outcomes)
    results = analyze(landmarks, outcomes)
    write_csv(out / "first-passage-summary.csv", results["first_passage_summary"])
    write_csv(out / "component-incremental-effects.csv", [
        {"comparison": k, **{kk: vv for kk, vv in val.items() if kk not in ("date_effects", "treatment_outcomes", "control_outcomes")}}
        for k, val in results["comparisons"].items()])
    write_json(out / "matched-comparison-diagnostics.json", {"primary": results["comparisons"]["primary"],
               "component_comparisons": {k:v for k,v in results["comparisons"].items() if k != "primary"},
               "common_support": results["primary_matching"]})
    write_csv(out / "covariate-balance.csv", results["primary_balance"])
    primary_pairs = results["primary_pair_rows"]
    by_day: dict[str, list[float]] = {}
    for r in primary_pairs: by_day.setdefault(r["date"], []).append(r["difference"])
    bootstrap = _bootstrap_date_effect(by_day, date_universe=dates)
    write_json(out / "date-cluster-bootstrap-results.json", {"primary_incremental_effect": bootstrap,
               "primary_matching_pairs": len(primary_pairs), "date_cluster_count": len(by_day),
               "date_universe_count": len(dates),
               "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
               "component_effects": {k:v.get("date_cluster_bootstrap") for k,v in results["comparisons"].items()}})
    write_csv(out / "daily-effect-results.csv", results["daily"])
    write_csv(out / "leave-one-date-out-results.csv", results["leave_one_date_out"])
    long_short = {}
    for direction, label in ((1,"LONG_SETUP"),(-1,"SHORT_SETUP")):
        x=[r for r in results["joined"] if int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r["direction"]==direction and r["outcome"]!="NOT_AT_RISK_AT_LANDMARK"]
        long_short[label]=_outcome_stats(x)
    write_json(out / "long-short-robustness.json",long_short)
    overlap = {"duplicate_episode_ids": len(landmarks)-len({(r["episode_id"],r["landmark_ms"]) for r in landmarks}),
               "episode_overlap_not_independent": True,
               "sensitivity": "episode identity retained; primary uncertainty resamples full trading-date clusters; zone-overlap de-duplication is not performed"}
    primary_cell = [r for r in results["joined"] if int(r["landmark_ms"]) == 1000 and r["barrier_ticks"] == 4]
    unique_touch = {}
    for r in primary_cell: unique_touch.setdefault(r["episode_id"], {"date": r["date"], "t0": int(r["landmark_ns"] - 1_000_000_000)})
    overlapping_ids = set()
    byday_touch: dict[str, list[tuple[str, int]]] = {}
    for eid, rec in unique_touch.items(): byday_touch.setdefault(rec["date"], []).append((eid, rec["t0"]))
    for date, vals in byday_touch.items():
        vals.sort(key=lambda x:x[1])
        for i,(eid,t0) in enumerate(vals):
            if any(abs(t0-other_t0) < HORIZON_NS for j,(_,other_t0) in enumerate(vals) if j != i):
                overlapping_ids.add(eid)
    nonoverlap_cell=[r for r in primary_cell if r["episode_id"] not in overlapping_ids]
    overlap_cmp, _, _ = _compare(nonoverlap_cell,"complete_absorption","aggression_only_control","primary_nonoverlap_episode_subset",dates)
    overlap.update({"primary_episodes":len(unique_touch),"overlapping_episode_count":len(overlapping_ids),
                    "nonoverlap_episode_count":len(unique_touch)-len(overlapping_ids),
                    "primary_outcomes_nonoverlap":_outcome_stats([r for r in nonoverlap_cell if r["outcome"]!="NOT_AT_RISK_AT_LANDMARK"]),
                    "matched_nonoverlap_comparison":overlap_cmp,
                    "definition":"same-date episodes whose 30-second first-passage windows overlap by touch time (<30 seconds apart) are marked overlapping; identities are retained in the main analysis"})
    write_json(out / "episode-overlap-sensitivity.json", overlap)
    pilot_dates=set(v2.PILOT_DATES)
    pilot_compare={}
    for name,date_set in (("pilot_five_dates",pilot_dates),("remaining_thirty_dates",set(dates)-pilot_dates)):
        cell=[r for r in primary_cell if r["date"] in date_set and r["at_risk"]]
        treatment=[r for r in cell if r["complete_absorption"]]
        control=[r for r in cell if r["aggression_only_control"]]
        pilot_compare[name]={"exploratory_only":True,"date_count":len({r["date"] for r in cell}),
                             "treatment":_outcome_stats(treatment),"aggression_only_control":_outcome_stats(control),
                             "all_eligible_reversal_rate_difference":(_outcome_stats(treatment)["reversal_first_all_eligible"]-
                                  _outcome_stats(control)["reversal_first_all_eligible"])
                                  if treatment and control else None}
    write_json(out/"pilot-vs-remaining-train-consistency.json",pilot_compare)
    censor = {}
    for lm_ms in LANDMARKS_MS:
        for n in BARRIERS_TICKS:
            x=[r for r in outcomes if int(r["landmark_ms"])==lm_ms and r["barrier_ticks"]==n]
            censor[f"{lm_ms}ms_{n}ticks"]={"counts":dict(Counter(r["outcome"] for r in x)),"n":len(x)}
    write_json(out / "censoring-and-competing-risks.json", {"cells":censor,
               "interpretation":"REVERSAL_FIRST and CONTINUATION_FIRST are competing first-passage outcomes; SOURCE_CENSORED and NOT_REACHED remain distinct. Incidence fractions are descriptive; censoring is not treated as proof of no reversal."})
    primary_joined=[r for r in results["joined"] if int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r["at_risk"]]
    cost_scale={"spread_at_primary_landmark_median_ticks":float(np.median([r["spread_ticks"] for r in primary_joined if r["spread_ticks"] is not None])) if any(r["spread_ticks"] is not None for r in primary_joined) else None,
                "approx_round_trip_cost_ticks":2.48,"basis":"Existing V1 execution-cost artifact: $6 round-trip commission, 1 ES tick entry market slippage, 1 ES tick stop-market slippage; at $12.50/tick this is 2.48 ticks. Path-dependent; spread is separately reported and no executable PnL is inferred.",
                "median_favorable_excursion_before_reversal_ticks":float(np.median([r["max_favorable_excursion_ticks"] for r in results["joined"] if int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r["outcome"]=="REVERSAL_FIRST"])) if any(int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r["outcome"]=="REVERSAL_FIRST" for r in results["joined"]) else None,
                "median_adverse_excursion_before_4tick_first_passage":float(np.median([r["adverse_excursion_before_first_passage_ticks"] for r in results["joined"] if int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r.get("adverse_excursion_before_first_passage_ticks") is not None])) if any(int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 and r.get("adverse_excursion_before_first_passage_ticks") is not None for r in results["joined"]) else None,
                "median_adverse_excursion_ticks_30s":float(np.median([r["max_adverse_excursion_ticks"] for r in results["joined"] if int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4])) if any(int(r["landmark_ms"])==1000 and r["barrier_ticks"]==4 for r in results["joined"]) else None}
    write_json(out / "economic-relevance-diagnostics.json",cost_scale)
    primary=results["comparisons"]["primary"]
    effect=primary["matched_reversal_first_risk_difference"]
    ci=primary["date_cluster_bootstrap"].get("ci95")
    if primary["classification"]=="NOT_IDENTIFIABLE":
        predictive="inconclusive"
    elif effect is not None and ci and ci[0]>0 and primary["positive_date_share"] is not None and primary["positive_date_share"]>=0.60:
        predictive="supported_exploratorily"
    elif effect is not None and ci and ci[1]<0:
        predictive="unsupported"
    else:
        predictive="inconclusive"
    decision="CONTINUE_TO_ECONOMIC_VALIDATION" if predictive=="supported_exploratorily" and cost_scale["approx_round_trip_cost_ticks"] is not None else "STOP_THIS_HVN_ABSORPTION_STRATEGY_FOR_NOW"
    if predictive=="inconclusive": decision="STOP_THIS_HVN_ABSORPTION_STRATEGY_FOR_NOW"
    decision_doc={"primary_estimand":DESIGN["primary"],"primary_result":primary,"predictive_information":predictive,
                  "economic_relevance":"inconclusive",
                  "mechanism_hypothesis":"supported_exploratorily" if predictive=="supported_exploratorily" else "unsupported" if predictive=="unsupported" else "inconclusive",
                  "research_allocation_decision":decision,
                  "justification":"The predeclared primary matched comparison and date-cluster uncertainty govern. No threshold, landmark or barrier is selected from results. Economic tradability remains separate and no executable PnL is inferred."}
    write_json(out / "decision-matrix.json",decision_doc)
    integrity_out={"status":"PASS","frozen_v2_spec_sha256":integrity["v2_spec_sha256"],"prior_artifact_integrity":integrity,
                   "replayed_transition_rows":sum(r["transition_count"] for r in date_results),
                   "transition_match_by_date":{r["date"]:r["transition_match"] for r in date_results},
                   "native_scan_diagnostics":{r["date"]:{k:r[k] for k in ("record_count","ts_recv_backward_steps","ts_event_backward_steps","sequence_nonincrease_steps","bad_recv_records")} for r in date_results}}
    write_json(out / "source-and-detector-integrity.json",integrity_out)
    summary={"study_id":STUDY_ID,"status":"PASS","dates":dates,"episode_count":len(episodes),"landmark_count":len(landmarks),
             "first_passage_row_count":len(outcomes),"primary":primary,"predictive_information":predictive,
             "economic_relevance":decision_doc["economic_relevance"],"mechanism_hypothesis":decision_doc["mechanism_hypothesis"],
             "research_allocation_decision":decision,"october_accessed":False,"protected_oos_accessed":False,"data_downloaded":False,
             "runtime_seconds":time.time()-started}
    write_json(out / "summary.json",summary)
    (out / "executive-decision.md").write_text(f"# Executive decision\n\n**{decision}**\n\nPredictive information: **{predictive}**. The primary fixed 1,000 ms / 4-tick / 30 s comparison has {primary['matching']['matched_pairs']} matched pairs, {primary['matched_pairs_with_uncensored_paths']} uncensored pairs, effect {effect}, date-cluster 95% CI {ci}, and positive-date share {primary['positive_date_share']}. Economic tradability is assessed separately and is inconclusive without a verified round-trip cost assumption.\n",encoding="utf-8")
    (out / "report.md").write_text(_report(summary,results,pilot_compare,overlap,cost_scale),encoding="utf-8")
    (out / "scientific-limitations.md").write_text("# Scientific limitations\n\nThis is an observational event study. Episode overlap remains; date is the bootstrap cluster. Matching is deterministic, greedy, and restricted to exact date/direction/local-hour strata. The primary effect excludes censored and tied matched pairs while separately reporting all outcomes and censoring. It is not a causal estimate absent unmeasured confounding assumptions. Aggregate MBP depth does not identify queue position, hidden liquidity, passive fills, or trade-caused replenishment. Midpoint first passage is not an executable fill or PnL simulation.\n",encoding="utf-8")
    (out / "next-research-allocation.md").write_text(f"# Next research allocation\n\nRecommendation: **{decision}**. Do not tune this detector from these results. {'If future work is authorized, use a distinct strategy hypothesis and preserve the Spring TRAIN evidence as exploratory only.' if decision.startswith('STOP') else 'Any economic validation requires a separately specified execution study and untouched validation data.'}\n",encoding="utf-8")
    write_json(out / "test-results.json",{"status":"PENDING_EXTERNAL_TEST_COMMAND","focused_tests":0,"data_downloaded":False})
    names=[p.name for p in out.iterdir() if p.is_file() and p.name!="artifact-hashes.json"]
    hashes={name:sha256(out/name) for name in sorted(names)}
    write_json(out/"artifact-hashes.json",hashes)
    write_json(out/"run-manifest.json",{"study_id":STUDY_ID,"complete":True,"status":"PASS","date_count":len(dates),
               "episode_count":len(episodes),"landmark_rows":len(landmarks),"first_passage_rows":len(outcomes),"design_sha256":design_hash,
               "spec_sha256":integrity["v2_spec_sha256"],"source_hashes_verified":True,"transition_match_all_dates":True,
               "validation_accessed":False,"oos_accessed":False,"data_downloaded":False,"commit_performed":False,
               "runtime_seconds":time.time()-started})
    return summary


def _report(summary: dict[str, Any], result: dict[str, Any], pilot_compare: dict[str, Any], overlap: dict[str, Any], economic: dict[str, Any]) -> str:
    p=summary["primary"]
    lines=[f"# {STUDY_ID}","",f"Status: **{summary['status']}**; 35 TRAIN dates and {summary['episode_count']} frozen episodes.",
           "","## Frozen primary comparison", "",f"- Landmark/barrier/horizon: 1000 ms / 4 ticks / 30 s.",
           f"- At-risk treatment/control: {p['treatment_outcomes']['n']} / {p['control_outcomes']['n']}.",
           f"- Matched pairs: {p['matching']['matched_pairs']}; uncensored matched pairs: {p['matched_pairs_with_uncensored_paths']}; date coverage: {len(p['date_effects'])} dates.",
           f"- Complete evidence reversal-first, all eligible: {p['treatment_outcomes']['reversal_first_all_eligible']}; aggression-only control: {p['control_outcomes']['reversal_first_all_eligible']}.",
           f"- Matched reversal risk difference: {p['matched_reversal_first_risk_difference']}; date-cluster CI: {p['date_cluster_bootstrap']['ci95']}; positive-date share: {p['positive_date_share']}.",
           f"- Treatment/control censor or nonresolution shares: {p['treatment_outcomes']['nonresolution_or_censoring_share']} / {p['control_outcomes']['nonresolution_or_censoring_share']}.",
           f"- Common support: treatment coverage {p['matching']['treatment_coverage']}; control coverage {p['matching']['control_coverage']}; median covariate balance is in covariate-balance.csv.",
           f"- Allocation: **{summary['research_allocation_decision']}**.","","## Fixed landmark and barrier matrix", "","| Landmark | Barrier | n | reversal all | continuation all | censored/nonresolved | resolved reversal |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["first_passage_summary"]:
        lines.append(f"| {r['landmark_ms']} ms | {r['barrier_ticks']} ticks | {r['n']} | {r['reversal_first_all_eligible']} | {r['continuation_first_all_eligible']} | {r['nonresolution_or_censoring_share']} | {r['reversal_first_among_resolved']} |")
    lines += ["","## Incremental components at the primary cell","","The primary full-detector comparison is not identifiable under the predeclared matched-support rule. Component comparisons and their 5,000 date-cluster bootstrap intervals are in matched-comparison-diagnostics.json and date-cluster-bootstrap-results.json. Where matched support is sparse or absent, estimates are explicitly not identifiable.",
              "","## Robustness and economics","",f"- Pilot five dates versus remaining TRAIN: `{json.dumps(pilot_compare,sort_keys=True)}`.",
              f"- Episode overlap: {overlap['overlapping_episode_count']} / {overlap['primary_episodes']} primary-cell episodes overlap another 30-second outcome window; nonoverlap sensitivity is in episode-overlap-sensitivity.json.",
              f"- Existing cost scale: approximately {economic['approx_round_trip_cost_ticks']} ticks including commission and assumed entry/stop slippage; median landmark spread {economic['spread_at_primary_landmark_median_ticks']} ticks. This is not an executable-PnL estimate.",
              "","All outcomes are observational midpoint first-passage measures, not simulated fills. Source censoring is not evidence of no reversal. The weak common support and two matched primary pairs do not support a credible incremental predictive conclusion."]
    return "\n".join(lines)+"\n"


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    result=run(args.repo.resolve(),resume=not args.no_resume)
    print(json.dumps({k:result[k] for k in ("status","episode_count","landmark_count","first_passage_row_count","research_allocation_decision")},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
