"""Outcome-blind, causal failed-aggression detector research on Spring 2025 ES.

This is descriptive event instrumentation, not a trading strategy.  Native
DBN order is retained and ts_recv is the availability clock; ts_event is kept
beside every recorded transition.
"""
from __future__ import annotations

import argparse
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

STUDY_ID = "ES_HVN_FAILED_AGGRESSION_DETECTOR_V2_RESEARCH"
OUT = Path("research_runs") / STUDY_ID
V1 = Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1")
CAUSAL = Path("research_runs/ES_HVN_CAUSAL_ABSORPTION_DETECTION_RESEARCH_V1")
PILOT_DATES = ("2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07")
HORIZONS_MS = (1000, 5000, 10000, 30000)
NS = 1_000_000_000
TICK = 0.25
CHUNK_ROWS = 250_000
DETECTION_WINDOW_NS = 35 * NS

# Fixed before detector/outcome processing.  Values are descriptive conventions
# anchored to the existing V1 heavy-print scale and ES tick, not fitted cutoffs.
SPEC = {
    "study_id": STUDY_ID,
    "source": {"dataset": "GLBX.MDP3", "schema": "mbp-10", "contract_by_date": "ESH5 through 2025-03-14; ESM5 from 2025-03-17", "native_order": "preserve DBN file order; never sort by sequence", "availability_clock": "ts_recv", "paired_event_clock": "ts_event", "bad_recv_flag_excluded_from_validity": True, "unknown_aggressor": "retained in raw counts; excluded from directional volume/count", "book_actions": "A/M/C aggregate MBP updates; not order IDs or proof of trade-caused replenishment"},
    "population": {"identity": "every V1 zone first-touch identity; V1 touch direction/status/threshold joined by (date, zone_id); otherwise nondirectional zone first touch", "pilot_dates": list(PILOT_DATES), "full_train_dates": "all 35 dates enumerated in V1 source-integrity-results.json", "per_episode_detection_censor_ns_after_touch": DETECTION_WINDOW_NS},
    "detector": {
        "attack_direction": "aggressor direction into the touched zone, opposite the V1 setup direction; unknown setup direction is unresolved",
        "AGGRESSION_EFFORT_ESTABLISHED": "at least 2 known attack-side T records and cumulative attack volume >= 2 * that episode's stored V1 heavy_threshold; no threshold means evidence unavailable, not a substitute value",
        "directional_persistence": "attack-side execution count / all known directional execution count >= 0.5 at qualification; descriptive balance, not evidence of intent",
        "PRICE_IMPACT_EVALUATED": "at qualification, maximum observed quote-mid movement in attack direction from the first valid post-touch mid <= 1 ES tick; continuous max progress retained",
        "CONTINUATION": "after effort qualification, maximum attack-direction penetration beyond the original zone boundary >= 4 ES ticks; fixed tick-scale description, not fitted",
        "DEPTH_RESPONSE_EVALUATED": "at least 2 attack prints while aggregate displayed depth remains positive at one or more original opposing-side zone prices, OR observed depth reduction followed by positive depth recovery at those same original prices; no best-price substitution",
        "FAILED_AGGRESSION_CANDIDATE": "effort + weak-impact convention + depth response, before any qualifying continuation; candidate is distinct from rejection",
        "REJECTION_OBSERVED": "after candidate, valid quote-mid reverses >= 1 ES tick from the most adverse attack-direction mid extreme; continuous reversal and timestamp retained",
        "depth_semantics": "aggregate MBP-10 persistence/recovery only; does not identify order IDs, passive fills, or causal replenishment",
        "UNKNOWN_OR_INVALID": "missing direction, missing V1 heavy threshold, invalid/stale quote, bad receive-time flag, or source gap prevents the corresponding claim; never coerced into a positive detector class",
        "final_status": "FAILED_AGGRESSION_REJECTED, FAILED_AGGRESSION_NO_REJECTION, CONTINUATION, or UNRESOLVED at right censor; no forced signal"
    },
    "forward": {"horizons_ms": list(HORIZONS_MS), "clock": "ts_recv from actual stage-specific classification timestamp", "metrics": ["direction_adjusted_mid_markout_ticks", "MFE_ticks", "MAE_ticks", "reversal_frequency", "continuation_frequency", "validity", "censoring"], "independent_unit": "date clusters; episodes overlap and are not independent"},
    "constraints": {"profitability_optimization": False, "threshold_search": False, "entry_or_stop_changes": False, "october_accessed": False, "protected_oos_accessed": False, "downloaded": False, "commit": False}
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def qualifies_effort(volume: int, threshold: float | None, count: int, persistence: float) -> bool:
    return bool(threshold is not None and math.isfinite(threshold) and threshold > 0 and count >= 2
                and volume >= 2 * threshold and persistence >= 0.5)


def weak_impact(max_progress_ticks: float) -> bool:
    return math.isfinite(max_progress_ticks) and max_progress_ticks <= 1.0


def final_status(candidate_ns: int | None, continuation_ns: int | None,
                 rejection_ns: int | None) -> str:
    """Resolve later transitions by first terminal evidence, not eventual path extrema.

    A continuation that occurs before a subsequent pullback supersedes a prior
    candidate; a rejection that is observed first confirms the failed-aggression
    path. Equal receive timestamps are conservatively assigned to continuation
    because timestamp alone cannot establish which same-time event came first.
    """
    if candidate_ns is not None and rejection_ns is not None and (
        continuation_ns is None or rejection_ns < continuation_ns
    ):
        return "FAILED_AGGRESSION_REJECTED"
    if continuation_ns is not None:
        return "CONTINUATION"
    if candidate_ns is not None:
        return "FAILED_AGGRESSION_NO_REJECTION"
    return "UNRESOLVED"


def depth_response(attack_depth_samples: list[int], had_reduction: bool, recovered: bool) -> bool:
    persistent = sum(v > 0 for v in attack_depth_samples) >= 2
    return persistent or (had_reduction and recovered)


def touch_row_matches(ep: base.Episode, row: Any, ts_recv: int) -> bool:
    """Match the frozen V1 first-touch identity exactly, including heavy episodes."""
    if ts_recv != ep.t0:
        return False
    if ep.touch_price is None:
        return True
    target = round(float(ep.touch_price) * 1e9)
    if bytes(row["action"]) == b"T" and int(row["price"]) == target:
        return True
    book = base._book(row)
    return any(price == target for price, _ in (*book["bids"], *book["asks"]))


def state_transition_invariants(rows: list[dict[str, Any]]) -> None:
    rank = {"HVN_TOUCH": 0, "AGGRESSION_OBSERVED": 1, "AGGRESSION_EFFORT_ESTABLISHED": 2,
            "PRICE_IMPACT_EVALUATED": 3, "DEPTH_RESPONSE_EVALUATED": 4,
            "FAILED_AGGRESSION_CANDIDATE": 5, "CONTINUATION": 5, "UNRESOLVED": 5, "REJECTION_OBSERVED": 6}
    prev_recv = -1
    seen: set[str] = set()
    for r in rows:
        kind = r["transition"]
        if kind not in rank:
            raise ValueError(f"unknown transition {kind}")
        recv = int(r["ts_recv"])
        # ts_event is retained as a paired exchange clock; availability/order
        # claims use ts_recv and native DBN order only.
        if recv < prev_recv:
            raise ValueError("causal transition receive-time/order violation")
        if kind == "REJECTION_OBSERVED" and "FAILED_AGGRESSION_CANDIDATE" not in seen:
            raise ValueError("rejection cannot precede candidate")
        if kind in {"FAILED_AGGRESSION_CANDIDATE", "CONTINUATION"} and "AGGRESSION_EFFORT_ESTABLISHED" not in seen:
            raise ValueError("terminal evidence state lacks qualified effort")
        if kind in seen:
            raise ValueError(f"duplicate transition {kind}")
        seen.add(kind)
        prev_recv = recv


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _load_inputs(repo: Path) -> tuple[list[base.Episode], list[str], dict[str, dict[str, Any]]]:
    prior_roots = [V1, Path("research_runs/ES_HVN_FAILED_AGGRESSION_DETECTION_AUDIT_V1"),
                   Path("research_runs/ES_HVN_NATIVE_MBP10_ABSORPTION_MECHANISM_V1"), CAUSAL]
    for prior in prior_roots:
        manifest_path = repo / prior / "artifact-hashes.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prior study hash manifest missing: {prior}")
        hashes = json.loads(manifest_path.read_text())
        for rel, expected in hashes.items():
            path = repo / prior / rel
            if not path.is_file() or _sha(path) != expected:
                raise ValueError(f"prior study artifact hash mismatch: {prior / rel}")
    frozen = json.loads((repo / V1 / "frozen-v1-specification.json").read_text())
    frozen_hash = canonical_hash(frozen["specification"])
    if frozen_hash != frozen.get("sha256"):
        raise ValueError("prior V1 frozen specification hash mismatch")
    causal_manifest = json.loads((repo / CAUSAL / "run-manifest.json").read_text())
    if not causal_manifest.get("complete") or causal_manifest.get("study_id") != "ES_HVN_CAUSAL_ABSORPTION_DETECTION_RESEARCH_V1":
        raise ValueError("prior causal measurement run is not complete")
    source_doc = json.loads((repo / base.V1 / "source-integrity-results.json").read_text())
    if source_doc.get("status") not in ("PASS", "VERIFIED", "COMPLETE") or len(source_doc.get("sources", [])) != 35:
        raise ValueError("V1 full Spring TRAIN source manifest is not complete")
    dates = sorted(x["date"] for x in source_doc["sources"])
    # Rebuild the episode population via V1's frozen identity construction, then
    # retain the full set rather than only the mechanism-study pilot dates.
    episodes = base.make_episodes(repo)
    # base.DATES is deliberately five-day pilot; reconstruct the same identity
    # join across all 35 ledger dates without changing V1 code or its artifacts.
    import csv as _csv
    def read(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as f: return list(_csv.DictReader(f))
    zones = read(repo / base.V1 / "hvn-zone-ledger.csv")
    touches = read(repo / base.V1 / "hvn-touch-episodes.csv")
    signals = read(repo / base.V1 / "confirmed-signal-ledger.csv")
    rrrows = [r for r in read(repo / base.V1 / "rr-grid-trade-results.csv") if float(r.get("rr", 0) or 0) == 1.0]
    touch_map = {(r["date"], r["zone_id"]): r for r in touches}
    signal_map = {(r["date"], r["zone_id"]): r for r in signals}
    rr_map = {r["signal_id"]: r for r in rrrows}
    full = []
    for z in zones:
        date, first = z["date"], z.get("first_touch_ns", "")
        if date not in dates or not first: continue
        t = touch_map.get((date, z["zone_id"]))
        t0 = int(t["T0_ns"]) if t else int(first)
        sig = signal_map.get((date, z["zone_id"]))
        full.append(base.Episode(
            episode_id=f"{date}|{z['zone_id']}|{t0}", date=date, zone_id=z["zone_id"], t0=t0,
            direction=int(t["direction"]) if t and t.get("direction") not in (None, "") else None,
            low=float(z["zone_low"]), high=float(z["zone_high"]), poc=float(z["zone_poc"]),
            origin="V1_TOUCH_EPISODE" if t else "ZONE_FIRST_TOUCH_NO_DIRECTION",
            v1_status=t.get("episode_status", "FIRST_TOUCH_NO_DIRECTION") if t else "FIRST_TOUCH_NO_DIRECTION",
            v1_heavy=bool(t and t.get("heavy_print", "False").lower() == "true"),
            v1_heavy_size=float(t["heavy_size"]) if t and t.get("heavy_size") else None,
            v1_threshold=float(t["heavy_threshold"]) if t and t.get("heavy_threshold") else None,
            v1_heavy_ns=int(t["TH_ns"]) if t and t.get("TH_ns") else None,
            v1_heavy_price=float(t["TH_price"]) if t and t.get("TH_price") else None,
            touch_price=float(t["T0_price"]) if t and t.get("T0_price") else None,
            approach=t.get("approach", "UNKNOWN") if t else "UNKNOWN", v1=sig,
            rr1=rr_map.get(sig["signal_id"]) if sig else None))
    full.sort(key=lambda e: (e.date, e.t0, e.zone_id, e.episode_id))
    expected_episodes = sum(bool(z.get("first_touch_ns")) and z.get("date") in dates for z in zones)
    if len(full) != expected_episodes or len({e.episode_id for e in full}) != len(full):
        raise ValueError(f"unexpected frozen 35-day episode population: {len(full)} != {expected_episodes}")
    # Identity regression: pilot's first 350 identities must match the V1 maker.
    pilot_ids = [e.episode_id for e in full if e.date in PILOT_DATES]
    if pilot_ids != [e.episode_id for e in episodes]:
        raise ValueError("pilot episode identities differ from V1 frozen identity builder")
    by_date = {}
    for rec in source_doc["sources"]:
        p = repo / rec["raw_path"]
        if not p.is_file() or p.stat().st_size != rec["raw_bytes"] or _sha(p) != rec["raw_sha256"]:
            raise ValueError(f"source hash/size mismatch: {rec['date']}")
        expected = "ESH5" if rec["date"] <= "2025-03-14" else "ESM5"
        if (rec.get("dataset"), rec.get("schema"), rec.get("symbol")) != ("GLBX.MDP3", "mbp-10", expected):
            raise ValueError(f"source identity mismatch for {rec['date']}")
        by_date[rec["date"]] = {**rec, "absolute_path": str(p.resolve()), "hash_verified": True}
    return full, dates, by_date


def _new_episode_state(ep: base.Episode) -> dict[str, Any]:
    return {"ep": ep, "touch": False, "first_aggr": None, "effort": None, "impact": None,
            "depth_eval": None, "candidate": None, "continuation": None, "unresolved": None, "rejection": None,
            "transitions": [], "start_mid": None, "last_mid": None, "max_attack_progress": 0.0,
            "max_boundary_penetration": 0.0, "attack_volume": 0, "attack_count": 0,
            "opposing_volume": 0, "opposing_count": 0, "unknown_volume": 0,
            "depth_start": None, "depth_last": None, "depth_min": None, "depth_reduction": False,
            "depth_recovered": False, "attack_depth_samples": [], "trackers": [],
            "unknown_dir": ep.direction is None, "threshold_missing": ep.v1_threshold is None,
            "bad_source": False, "gap_count": 0, "prev_recv": None, "v1_entry_ns": int(ep.v1["TE_ns"]) if ep.v1 and ep.v1.get("TE_ns") else None}


def _append_transition(st: dict[str, Any], kind: str, row: Any, book: dict[str, Any] | None, note: str = "") -> None:
    ep = st["ep"]
    recv, event = int(row["ts_recv"]), int(row["ts_event"])
    item = {"episode_id": ep.episode_id, "date": ep.date, "transition": kind, "ts_recv": recv,
            "ts_event": event, "sequence": int(row["sequence"]), "market_mid": book["mid"] if book and book["valid"] else None,
            "quote_valid": bool(book and book["valid"] and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV)),
            "direction": ep.direction, "attack_side": "SELL" if ep.direction == 1 else "BUY" if ep.direction == -1 else "UNKNOWN",
            "attack_volume": st["attack_volume"], "attack_execution_count": st["attack_count"],
            "opposing_volume": st["opposing_volume"], "max_attack_progress_ticks": st["max_attack_progress"],
            "max_boundary_penetration_ticks": st["max_boundary_penetration"],
            "defended_zone_depth": st["depth_last"], "depth_start": st["depth_start"],
            "depth_reduced": st["depth_reduction"], "depth_recovered": st["depth_recovered"],
            "v1_status": ep.v1_status, "v1_entry_ns": st["v1_entry_ns"], "note": note}
    st["transitions"].append(item)
    if book and book["valid"] and ep.direction is not None and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV):
        st["trackers"].append({"stage": kind, "ts_recv": recv, "ts_event": event, "reference_mid": book["mid"],
                               # Positive markout is reversal toward the original
                               # setup direction; continuation should be negative.
                               "direction": ep.direction, "horizons": {h: {"end": recv + h*1_000_000,
                               "last_mid": book["mid"], "mfe": 0.0, "mae": 0.0, "gap": False,
                               "last_recv": recv, "covered": False} for h in HORIZONS_MS}})


def _zone_depth(book: dict[str, Any], ep: base.Episode) -> int:
    if ep.direction is None: return 0
    opposing = book["bids"] if ep.direction > 0 else book["asks"]
    low, high = round(ep.low * 4), round(ep.high * 4)
    return sum(size for price, size in opposing if low <= round(price / 1e9 * 4) <= high)


def _update_episode(st: dict[str, Any], row: Any) -> None:
    ep = st["ep"]
    recv, event, flags = int(row["ts_recv"]), int(row["ts_event"]), int(row["flags"])
    if recv < ep.t0 or recv > ep.t0 + DETECTION_WINDOW_NS + 30 * NS: return
    if st["prev_recv"] is not None and recv - st["prev_recv"] > 2 * NS:
        st["gap_count"] += 1
    st["prev_recv"] = recv
    book = base._book(row)
    valid = bool(book["valid"] and not (flags & base.FLAG_BAD_TS_RECV) and st["gap_count"] == 0)
    if flags & base.FLAG_BAD_TS_RECV: st["bad_source"] = True
    if not st["touch"]:
        # V1's source-supported touch is exact ts_recv + trade/book price identity.
        matched = touch_row_matches(ep, row, recv)
        if not matched: return
        st["touch"] = True
        _append_transition(st, "HVN_TOUCH", row, book, "exact V1 receive-time/price identity")
        if ep.direction is None: return
    if ep.direction is None: return
    if (recv >= ep.t0 + DETECTION_WINDOW_NS and st["candidate"] is None
            and st["continuation"] is None and st["unresolved"] is None):
        st["unresolved"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"] if valid else None}
        _append_transition(st, "UNRESOLVED", row, book, "no candidate or continuation by the fixed right-censor")
    if recv <= ep.t0 + DETECTION_WINDOW_NS and valid:
        st["last_mid"] = book["mid"]
        if st["start_mid"] is None: st["start_mid"] = book["mid"]
        depth = _zone_depth(book, ep)
        if st["depth_start"] is None: st["depth_start"] = depth
        if st["depth_last"] is not None and depth < st["depth_last"]: st["depth_reduction"] = True
        if st["depth_min"] is None or depth < st["depth_min"]: st["depth_min"] = depth
        if st["depth_reduction"] and depth > (st["depth_min"] or 0) and depth > 0: st["depth_recovered"] = True
        st["depth_last"] = depth
        progress = ((book["mid"] - st["start_mid"]) / TICK) * (-ep.direction)
        st["max_attack_progress"] = max(st["max_attack_progress"], progress)
        boundary = ep.high if ep.direction > 0 else ep.low
        penetration = ((book["mid"] - boundary) / TICK) * (-ep.direction)
        st["max_boundary_penetration"] = max(st["max_boundary_penetration"], penetration)
    if recv <= ep.t0 + DETECTION_WINDOW_NS and bytes(row["action"]) == b"T":
        side = bytes(row["side"]).decode("ascii")
        aggressor = base.trade_aggressor("T", side)
        attack = "SELL" if ep.direction > 0 else "BUY"
        if aggressor == "UNKNOWN": st["unknown_volume"] += int(row["size"])
        elif aggressor == attack:
            if st["first_aggr"] is None:
                st["first_aggr"] = {"ts_recv": recv, "ts_event": event, "price": int(row["price"]) / 1e9}
                _append_transition(st, "AGGRESSION_OBSERVED", row, book)
            st["attack_volume"] += int(row["size"]); st["attack_count"] += 1
            depth = _zone_depth(book, ep) if book["valid"] else 0
            st["attack_depth_samples"].append(depth)
        else:
            st["opposing_volume"] += int(row["size"]); st["opposing_count"] += 1
        known = st["attack_count"] + st["opposing_count"]
        persistence = st["attack_count"] / known if known else 0.0
        if st["effort"] is None and valid and qualifies_effort(st["attack_volume"], ep.v1_threshold, st["attack_count"], persistence):
            st["effort"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"], "entry_progress": st["max_attack_progress"]}
            _append_transition(st, "AGGRESSION_EFFORT_ESTABLISHED", row, book)
            # Impact is evaluated on the same causal record as effort; no wait timer.
            st["impact"] = weak_impact(st["max_attack_progress"])
            _append_transition(st, "PRICE_IMPACT_EVALUATED", row, book,
                               "weak-impact convention satisfied" if st["impact"] else "impact not weak")
            depth_ok = depth_response(st["attack_depth_samples"], st["depth_reduction"], st["depth_recovered"])
            st["depth_eval"] = depth_ok
            _append_transition(st, "DEPTH_RESPONSE_EVALUATED", row, book,
                               "original-zone aggregate depth persisted/recovered" if depth_ok else "depth response absent")
            if st["max_boundary_penetration"] >= 4.0:
                st["continuation"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"]}
                _append_transition(st, "CONTINUATION", row, book)
            elif st["impact"] and depth_ok:
                st["candidate"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"]}
                _append_transition(st, "FAILED_AGGRESSION_CANDIDATE", row, book)
    # A continuation may become observable after effort qualification.
    if st["effort"] and st["continuation"] is None and recv <= ep.t0 + DETECTION_WINDOW_NS and valid and st["max_boundary_penetration"] >= 4.0:
        st["continuation"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"]}
        _append_transition(st, "CONTINUATION", row, book)
    if st["candidate"] and st["rejection"] is None and valid:
        attack_extreme = st["start_mid"] + ((st["max_attack_progress"] * TICK) * (-ep.direction))
        reverse_ticks = ((book["mid"] - attack_extreme) / TICK) * ep.direction
        if reverse_ticks >= 1.0:
            st["rejection"] = {"ts_recv": recv, "ts_event": event, "mid": book["mid"], "reverse_ticks": reverse_ticks}
            _append_transition(st, "REJECTION_OBSERVED", row, book)
    for tracker in st["trackers"]:
        if recv <= tracker["ts_recv"]: continue
        for horizon in tracker["horizons"].values():
            if recv > horizon["end"]:
                horizon["covered"] = horizon["covered"] or horizon["last_recv"] >= horizon["end"]
                continue
            if recv - horizon["last_recv"] > 2 * NS: horizon["gap"] = True
            horizon["last_recv"] = recv
            if valid:
                rel = ((book["mid"] - tracker["reference_mid"]) / TICK) * tracker["direction"]
                horizon["mfe"] = max(horizon["mfe"], rel); horizon["mae"] = min(horizon["mae"], rel)
                horizon["last_mid"] = book["mid"]
            if recv >= horizon["end"]: horizon["covered"] = True


def _process_date(repo: Path, date: str, eps: list[base.Episode], source: dict[str, Any]) -> dict[str, Any]:
    from databento import DBNStore
    states = [_new_episode_state(e) for e in eps]
    intervals = sorted((e.t0, e.t0 + DETECTION_WINDOW_NS + 30 * NS) for e in eps)
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    store = DBNStore.from_file(source["absolute_path"])
    all_records = backward_recv = backward_event = sequence_nonincrease = bad_recv = trade_n = 0
    prior_recv = prior_event = prior_seq = None
    active: list[dict[str, Any]] = []; next_state = 0
    for chunk in store.to_ndarray(count=CHUNK_ROWS):
        if not isinstance(chunk, np.ndarray) or chunk.dtype.names is None: raise TypeError("unexpected DBN ndarray")
        if not len(chunk): continue
        if not np.all(chunk["rtype"] == 10): raise ValueError(f"non-MBP10 record in {date}")
        all_records += len(chunk)
        r, ev, seq = chunk["ts_recv"], chunk["ts_event"], chunk["sequence"]
        backward_recv += int(np.count_nonzero(r[1:] < r[:-1])); backward_event += int(np.count_nonzero(ev[1:] < ev[:-1]))
        sequence_nonincrease += int(np.count_nonzero(seq[1:] <= seq[:-1]))
        if prior_recv is not None and int(r[0]) < prior_recv: backward_recv += 1
        if prior_event is not None and int(ev[0]) < prior_event: backward_event += 1
        if prior_seq is not None and int(seq[0]) <= prior_seq: sequence_nonincrease += 1
        prior_recv, prior_event, prior_seq = int(r[-1]), int(ev[-1]), int(seq[-1])
        bad_recv += int(np.count_nonzero(chunk["flags"] & base.FLAG_BAD_TS_RECV))
        tm = chunk["action"] == b"T"; trade_n += int(np.count_nonzero(tm & (chunk["side"] == b"N")))
        selected: list[int] = []
        if np.all(r[1:] >= r[:-1]):
            for lo, hi in merged:
                a, b = int(np.searchsorted(r, lo, "left")), int(np.searchsorted(r, hi, "right"))
                if a < b: selected.extend(range(a, b))
        else:
            selected = base.native_order_window_indices(r, merged)
        for i in selected:
            row = chunk[i]; recv = int(row["ts_recv"])
            while next_state < len(states) and states[next_state]["ep"].t0 <= recv:
                active.append(states[next_state]); next_state += 1
            active = [st for st in active if recv <= st["ep"].t0 + DETECTION_WINDOW_NS + 30 * NS]
            for st in active: _update_episode(st, row)
    return {"date": date, "source_path": source["raw_path"], "source_sha256": source["raw_sha256"],
            "record_count": all_records, "unknown_aggressor_records": trade_n, "bad_recv_records": bad_recv,
            "ts_recv_backward_steps": backward_recv, "ts_event_backward_steps": backward_event,
            "sequence_nonincrease_steps": sequence_nonincrease, "episode_count": len(states), "states": states}


def _finalize_date(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger, transitions = [], []
    for st in result["states"]:
        ep = st["ep"]
        rows = st["transitions"]
        state_transition_invariants(rows)
        final = final_status(st["candidate"]["ts_recv"] if st["candidate"] else None,
                             st["continuation"]["ts_recv"] if st["continuation"] else None,
                             st["rejection"]["ts_recv"] if st["rejection"] else None)
        transition_map = {x["transition"]: x for x in rows}
        record = {"episode_id": ep.episode_id, "date": ep.date, "zone_id": ep.zone_id, "v1_status": ep.v1_status,
                  "v1_confirmed": bool(ep.v1), "v1_entry_ns": st["v1_entry_ns"], "direction": ep.direction,
                  "touch_ns": ep.t0, "origin": ep.origin, "v1_heavy_threshold": ep.v1_threshold,
                  "first_aggression_ns": st["first_aggr"]["ts_recv"] if st["first_aggr"] else None,
                  "effort_ns": st["effort"]["ts_recv"] if st["effort"] else None,
                  "low_impact_ns": transition_map.get("PRICE_IMPACT_EVALUATED", {}).get("ts_recv") if st["impact"] else None,
                  "depth_response_ns": transition_map.get("DEPTH_RESPONSE_EVALUATED", {}).get("ts_recv") if st["depth_eval"] else None,
                  "candidate_ns": st["candidate"]["ts_recv"] if st["candidate"] else None,
                  "continuation_ns": st["continuation"]["ts_recv"] if st["continuation"] else None,
                  "unresolved_ns": st["unresolved"]["ts_recv"] if st["unresolved"] else None,
                  "rejection_ns": st["rejection"]["ts_recv"] if st["rejection"] else None,
                  "final_status": final, "source_validity": "VALID_WITHIN_OBSERVED_ROWS" if not st["bad_source"] and st["gap_count"] == 0 else "GAP_OR_BAD_RECV_FLAG",
                  "threshold_available": ep.v1_threshold is not None, "qualified_effort": st["effort"] is not None,
                  "weak_impact": bool(st["impact"]), "depth_response": bool(st["depth_eval"]),
                  "attack_volume": st["attack_volume"], "attack_count": st["attack_count"],
                  "opposing_volume": st["opposing_volume"], "opposing_count": st["opposing_count"],
                  "max_attack_progress_ticks": st["max_attack_progress"], "max_boundary_penetration_ticks": st["max_boundary_penetration"],
                  "depth_start": st["depth_start"], "depth_min": st["depth_min"], "depth_last": st["depth_last"],
                  "effort_before_v1_entry": bool(st["effort"] and st["v1_entry_ns"] and st["effort"]["ts_recv"] <= st["v1_entry_ns"]),
                  "candidate_before_v1_entry": bool(st["candidate"] and st["v1_entry_ns"] and st["candidate"]["ts_recv"] <= st["v1_entry_ns"]),
                  "rejection_before_v1_entry": bool(st["rejection"] and st["v1_entry_ns"] and st["rejection"]["ts_recv"] <= st["v1_entry_ns"])}
        # Keep full timestamped state transitions and milestone quote prices.
        for kind in ("HVN_TOUCH", "AGGRESSION_OBSERVED", "AGGRESSION_EFFORT_ESTABLISHED", "PRICE_IMPACT_EVALUATED", "DEPTH_RESPONSE_EVALUATED", "FAILED_AGGRESSION_CANDIDATE", "CONTINUATION", "UNRESOLVED", "REJECTION_OBSERVED"):
            tr = transition_map.get(kind, {})
            record[kind.lower() + "_event_ns"] = tr.get("ts_event")
            record[kind.lower() + "_mid"] = tr.get("market_mid")
        touch_mid = record.get("hvn_touch_mid")
        candidate_mid = record.get("failed_aggression_candidate_mid")
        rejection_mid = record.get("rejection_observed_mid")
        attack_sign = -ep.direction if ep.direction is not None else None
        record["attack_direction_mid_move_to_candidate_ticks"] = ((candidate_mid-touch_mid)/TICK*attack_sign
            if touch_mid is not None and candidate_mid is not None and attack_sign is not None else None)
        record["reversal_mid_move_candidate_to_rejection_ticks"] = ((rejection_mid-candidate_mid)/TICK*ep.direction
            if candidate_mid is not None and rejection_mid is not None and ep.direction is not None else None)
        record["elapsed_touch_to_aggression_ms"] = ((record["first_aggression_ns"]-ep.t0)/1e6 if record["first_aggression_ns"] else None)
        record["elapsed_heavy_to_effort_ms"] = ((record["effort_ns"]-ep.v1_heavy_ns)/1e6 if record["effort_ns"] and ep.v1_heavy_ns else None)
        record["elapsed_effort_to_rejection_ms"] = ((record["rejection_ns"]-record["effort_ns"])/1e6
            if record["rejection_ns"] and record["effort_ns"] else None)
        ledger.append(record); transitions.extend(rows)
    return ledger, transitions


def _forward_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for result in results:
        for st in result["states"]:
            ep = st["ep"]
            for tr in st["trackers"]:
                for h in HORIZONS_MS:
                    horizon = tr["horizons"][h]
                    out.append({"episode_id": ep.episode_id, "date": ep.date, "class": _class_at(st, tr["stage"]),
                                "stage": tr["stage"], "horizon_ms": h,
                                "classification_ts_recv": tr["ts_recv"], "classification_ts_event": tr["ts_event"],
                                "valid": bool(horizon["covered"] and not horizon["gap"]), "censored": not horizon["covered"],
                                "markout_ticks": ((horizon["last_mid"] - tr["reference_mid"]) / TICK) * tr["direction"],
                                "mfe_ticks": horizon["mfe"], "mae_ticks": horizon["mae"], "direction": tr["direction"]})
    return out


def _class_at(st: dict[str, Any], stage: str) -> str:
    if stage == "CONTINUATION": return "CONTINUATION"
    if stage == "UNRESOLVED": return "UNRESOLVED"
    if stage in ("FAILED_AGGRESSION_CANDIDATE", "REJECTION_OBSERVED"): return "FAILED_AGGRESSION"
    return "UNRESOLVED_OR_COMPONENT_ONLY"


def _aggregate_forward(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for r in rows: grouped[(r["class"], r["stage"], r["horizon_ms"])].append(r)
    out = []
    for (cls, stage, horizon), values in sorted(grouped.items()):
        valid = [r for r in values if r["valid"]]
        marks = [r["markout_ticks"] for r in valid]
        out.append({"class": cls, "stage": stage, "horizon_ms": horizon, "episodes": len(values), "valid": len(valid),
                    "censored_or_invalid": len(values)-len(valid), "mean_markout_ticks": statistics.mean(marks) if marks else None,
                    "median_markout_ticks": statistics.median(marks) if marks else None,
                    "positive_markout_fraction": sum(x > 0 for x in marks)/len(marks) if marks else None,
                    "reversal_fraction": sum(x > 0 for x in marks)/len(marks) if marks else None,
                    "continuation_fraction": sum(x < 0 for x in marks)/len(marks) if marks else None,
                    "mean_mfe_ticks": statistics.mean(r["mfe_ticks"] for r in valid) if valid else None,
                    "mean_mae_ticks": statistics.mean(r["mae_ticks"] for r in valid) if valid else None,
                    "independent_dates": len({r["date"] for r in valid})})
    return out


def _write_transitions(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def recompute_forward_only(repo: Path) -> dict[str, Any]:
    """Re-measure fixed horizons from frozen transitions; never recomputes classes."""
    import pyarrow.parquet as pq
    from databento import DBNStore
    out = repo / OUT
    source_doc = json.loads((out / "source-integrity.json").read_text())
    if source_doc.get("status") != "PASS" or len(source_doc.get("dates", [])) != 35:
        raise ValueError("forward-only pass requires completed 35-date source-integrity results")
    transition_rows = pq.read_table(out / "detector-state-transitions.parquet").to_pylist()
    eligibility = {"AGGRESSION_EFFORT_ESTABLISHED", "PRICE_IMPACT_EVALUATED", "DEPTH_RESPONSE_EVALUATED",
                   "FAILED_AGGRESSION_CANDIDATE", "CONTINUATION", "UNRESOLVED", "REJECTION_OBSERVED"}
    trackers_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transition_rows:
        if row["transition"] not in eligibility or not row.get("quote_valid") or row.get("market_mid") is None or row.get("direction") is None:
            continue
        if row["transition"] == "PRICE_IMPACT_EVALUATED" and row.get("note") != "weak-impact convention satisfied": continue
        if row["transition"] == "DEPTH_RESPONSE_EVALUATED" and row.get("note") != "original-zone aggregate depth persisted/recovered": continue
        start = int(row["ts_recv"])
        trackers_by_date[row["date"]].append({"episode_id": row["episode_id"], "date": row["date"],
            "stage": row["transition"], "class": _class_at({}, row["transition"]), "start": start,
            "event": int(row["ts_event"]), "mid": float(row["market_mid"]), "direction": int(row["direction"]),
            "horizons": {h: {"end": start+h*1_000_000, "mid": float(row["market_mid"]), "last_quote_recv": start,
                             "last_recv": start, "mfe": 0.0, "mae": 0.0, "gap": False,
                             "invalid_quote": False, "covered": False} for h in HORIZONS_MS}})
    sources = json.loads((repo / base.V1 / "source-integrity-results.json").read_text())
    source_by_date = {x["date"]: x for x in sources["sources"]}
    all_rows = []
    for date in sorted(trackers_by_date):
        trackers = sorted(trackers_by_date[date], key=lambda x: x["start"])
        windows = sorted((x["start"], max(y["end"] for y in x["horizons"].values())) for x in trackers)
        merged: list[list[int]] = []
        for lo, hi in windows:
            if merged and lo <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], hi)
            else: merged.append([lo, hi])
        rec = source_by_date[date]
        path = repo / rec["raw_path"]
        active: list[dict[str, Any]] = []; next_i = 0
        for chunk in DBNStore.from_file(path).to_ndarray(count=CHUNK_ROWS):
            recv_arr = chunk["ts_recv"]
            selected: list[int] = []
            if np.all(recv_arr[1:] >= recv_arr[:-1]):
                for lo, hi in merged:
                    a, b = int(np.searchsorted(recv_arr, lo, "left")), int(np.searchsorted(recv_arr, hi, "right"))
                    if a < b: selected.extend(range(a, b))
            else:
                selected = base.native_order_window_indices(recv_arr, merged)
            for ix in selected:
                row = chunk[ix]; recv = int(row["ts_recv"])
                while next_i < len(trackers) and trackers[next_i]["start"] <= recv:
                    active.append(trackers[next_i]); next_i += 1
                active = [t for t in active if recv <= max(h["end"] for h in t["horizons"].values())]
                book = base._book(row)
                valid = bool(book["valid"] and not (int(row["flags"]) & base.FLAG_BAD_TS_RECV))
                for tracker in active:
                    if recv <= tracker["start"]: continue
                    for horizon in tracker["horizons"].values():
                        if horizon["covered"]: continue
                        if recv > horizon["end"]:
                            age = horizon["end"] - horizon["last_quote_recv"]
                            horizon["covered"] = 0 <= age <= 2*NS and not horizon["gap"] and not horizon["invalid_quote"]
                            continue
                        if recv - horizon["last_recv"] > 2*NS: horizon["gap"] = True
                        horizon["last_recv"] = recv
                        if valid:
                            rel = ((book["mid"] - tracker["mid"])/TICK)*tracker["direction"]
                            horizon["mid"] = book["mid"]; horizon["last_quote_recv"] = recv
                            horizon["mfe"] = max(horizon["mfe"], rel); horizon["mae"] = min(horizon["mae"], rel)
                            horizon["invalid_quote"] = False
                        else:
                            horizon["invalid_quote"] = True
        for tracker in trackers:
            for h, horizon in tracker["horizons"].items():
                if not horizon["covered"]:
                    age = horizon["end"] - horizon["last_quote_recv"]
                    horizon["covered"] = 0 <= age <= 2*NS and not horizon["gap"] and not horizon["invalid_quote"]
                all_rows.append({"episode_id": tracker["episode_id"], "date": date, "class": tracker["class"],
                    "stage": tracker["stage"], "horizon_ms": h, "classification_ts_recv": tracker["start"],
                    "classification_ts_event": tracker["event"], "valid": bool(horizon["covered"]),
                    "censored": not horizon["covered"],
                    "markout_ticks": ((horizon["mid"]-tracker["mid"])/TICK)*tracker["direction"],
                    "mfe_ticks": horizon["mfe"], "mae_ticks": horizon["mae"], "direction": tracker["direction"]})
        print(f"FORWARD_DATE_COMPLETE {date} checkpoints={len(trackers)}", flush=True)
    forward_agg = _aggregate_forward(all_rows)
    _write_csv(out / "forward-response-by-class.csv", forward_agg)
    ablation = json.loads((out / "aggression-impact-depth-ablation.json").read_text())
    ablation["forward"] = forward_agg
    ablation["forward_pass"] = "RECOMPUTED_FROM_FROZEN_STAGE_TIMESTAMPS_LAST_VALID_QUOTE_AT_OR_BEFORE_HORIZON"
    _write_json(out / "aggression-impact-depth-ablation.json", ablation)
    return {"forward_rows": len(all_rows), "groups": len(forward_agg), "valid": sum(r["valid"] for r in all_rows)}


def run(repo: Path, pilot_only: bool = False) -> dict[str, Any]:
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    episodes, dates, sources = _load_inputs(repo)
    spec_hash = canonical_hash(SPEC)
    # Freeze and hash before any native event/outcome processing.
    _write_json(repo / OUT / "frozen-detector-specification.json", {"specification": SPEC, "sha256": spec_hash})
    (repo / OUT / "detector-specification.sha256").write_text(spec_hash + "  frozen-detector-specification.json\n", encoding="ascii")
    all_results: list[dict[str, Any]] = []
    date_results = []
    run_dates = list(PILOT_DATES)
    for date in run_dates:
        result = _process_date(repo, date, [e for e in episodes if e.date == date], sources[date])
        all_results.append(result)
        ledger, trans = _finalize_date(result)
        date_results.append({"date": date, "episode_count": len(ledger), "record_count": result["record_count"],
                             "ts_recv_backward_steps": result["ts_recv_backward_steps"], "ts_event_backward_steps": result["ts_event_backward_steps"],
                             "sequence_nonincrease_steps": result["sequence_nonincrease_steps"], "bad_recv_records": result["bad_recv_records"],
                             "unknown_aggressor_records": result["unknown_aggressor_records"], "source_sha256": result["source_sha256"]})
        _write_csv(repo / OUT / f"pilot-{date}-episodes.csv", ledger)
        _write_csv(repo / OUT / f"pilot-{date}-transitions.csv", trans)
        print(f"PILOT_DATE_COMPLETE {date} episodes={len(ledger)} records={result['record_count']}", flush=True)
    # Causal technical gate. Expansion depends only on integrity, identities and
    # invariant checks; no return/performance threshold controls this decision.
    pilot_ledger = [row for date in run_dates for row in _finalize_date(next(x for x in all_results if x["date"] == date))[0]]
    pilot_transitions = [tr for x in all_results for st in x["states"] for tr in st["transitions"]]
    pilot_pass = (len(pilot_ledger) == 350 and all(x["ts_recv_backward_steps"] == 0 and x["ts_event_backward_steps"] == 0 for x in date_results)
                  and all(st["touch"] for x in all_results for st in x["states"])
                  and all(len([e for e in episodes if e.date == d]) == next(x["episode_count"] for x in date_results if x["date"] == d) for d in PILOT_DATES))
    if not pilot_pass:
        raise RuntimeError("five-date technical pilot gate failed; full expansion stopped")
    full_run = not pilot_only
    if full_run:
        for date in dates:
            if date in PILOT_DATES: continue
            result = _process_date(repo, date, [e for e in episodes if e.date == date], sources[date])
            all_results.append(result)
            ledger, trans = _finalize_date(result)
            date_results.append({"date": date, "episode_count": len(ledger), "record_count": result["record_count"],
                                 "ts_recv_backward_steps": result["ts_recv_backward_steps"], "ts_event_backward_steps": result["ts_event_backward_steps"],
                                 "sequence_nonincrease_steps": result["sequence_nonincrease_steps"], "bad_recv_records": result["bad_recv_records"],
                                 "unknown_aggressor_records": result["unknown_aggressor_records"], "source_sha256": result["source_sha256"]})
            _write_csv(repo / OUT / f"date-{date}-episodes.csv", ledger)
            _write_csv(repo / OUT / f"date-{date}-transitions.csv", trans)
            print(f"TRAIN_DATE_COMPLETE {date} episodes={len(ledger)} records={result['record_count']}", flush=True)
    ledgers, transitions = [], []
    for x in all_results:
        l, t = _finalize_date(x); ledgers.extend(l); transitions.extend(t)
    forward = _forward_rows(all_results)
    # Join the secondary V1 RR=1 label only after detector results are fixed.
    rr = {e.episode_id: e.rr1 for e in episodes if e.rr1}
    for r in ledgers:
        x = rr.get(r["episode_id"])
        r["rr1_net_r_secondary"] = float(x["net_r"]) if x and x.get("net_r") not in (None, "") else None
        r["v1_outcome_secondary"] = x.get("outcome") if x else None
    _write_csv(repo / OUT / "detector-event-ledger.csv", ledgers)
    _write_transitions(repo / OUT / "detector-state-transitions.parquet", transitions)
    timing = []
    for r in ledgers:
        item = dict(r)
        for name in ("first_aggression", "effort", "candidate", "rejection"):
            ts = item.get(name + "_ns")
            item[name + "_ms_from_touch"] = (int(ts)-int(item["touch_ns"]))/1e6 if ts is not None else None
        effort_mid = item.get("aggression_effort_established_mid")
        item["touch_to_effort_mid_move_ticks"] = None if effort_mid is None else None
        timing.append(item)
    _write_csv(repo / OUT / "detection-timing-analysis.csv", timing)
    forward_agg = _aggregate_forward(forward)
    _write_csv(repo / OUT / "forward-response-by-class.csv", forward_agg)
    statuses = Counter(r["final_status"] for r in ledgers)
    funnel = {"episode_count": len(ledgers), "v1_confirmed": sum(bool(r["v1_confirmed"]) for r in ledgers),
              "first_aggression": sum(r["first_aggression_ns"] is not None for r in ledgers),
              "qualified_effort": sum(r["qualified_effort"] for r in ledgers), "weak_impact": sum(r["weak_impact"] for r in ledgers),
              "depth_response": sum(r["depth_response"] for r in ledgers), "candidate": statuses["FAILED_AGGRESSION_NO_REJECTION"] + statuses["FAILED_AGGRESSION_REJECTED"],
              "continuation": statuses["CONTINUATION"], "unresolved": statuses["UNRESOLVED"], "final_status_counts": dict(statuses)}
    _write_json(repo / OUT / "classification-funnel.json", funnel)
    stages = defaultdict(list)
    for r in ledgers:
        for label, key in (("AGGRESSION_ONLY", "effort_ns"), ("AGGRESSION_PLUS_WEAK_IMPACT", "low_impact_ns"), ("AGGRESSION_IMPACT_DEPTH", "candidate_ns"), ("REJECTION_ADDED", "rejection_ns")):
            if r[key] is not None: stages[label].append(r)
    ablation = {name: {"n": len(rows), "dates": len({r["date"] for r in rows}), "v1_confirmed": sum(r["v1_confirmed"] for r in rows),
                       "before_v1_entry": sum(bool(r["v1_entry_ns"] and r[key] <= r["v1_entry_ns"]) for r in rows)}
                for name, rows in stages.items() for key in [{"AGGRESSION_ONLY":"effort_ns","AGGRESSION_PLUS_WEAK_IMPACT":"low_impact_ns","AGGRESSION_IMPACT_DEPTH":"candidate_ns","REJECTION_ADDED":"rejection_ns"}[name]]}
    _write_json(repo / OUT / "aggression-impact-depth-ablation.json", {"stages": ablation, "forward": _aggregate_forward(forward),
                "interpretation": "Stage timestamps differ; unmatched later cohorts are not a causal incremental-value comparison."})
    comparability = {"status": "NOT_IDENTIFIED_WITHOUT_LANDMARK_MATCHING", "reason": "Stage populations have different event times and survivor/censoring structures; no naive later-rejection versus earlier-unresolved comparison is interpreted as incremental predictive value.", "same_source_observations": True}
    _write_json(repo / OUT / "decision-time-comparability.json", comparability)
    confirmed = [r for r in ledgers if r["v1_confirmed"]]
    winner_loser = {"sample": len(confirmed),
                    "candidate_before_entry_winners": sum(bool(r["candidate_before_v1_entry"] and r["rr1_net_r_secondary"] is not None and r["rr1_net_r_secondary"] > 0) for r in confirmed),
                    "candidate_before_entry_losers": sum(bool(r["candidate_before_v1_entry"] and r["rr1_net_r_secondary"] is not None and r["rr1_net_r_secondary"] <= 0) for r in confirmed),
                    "v1_confirmations_without_candidate_before_entry": sum(not r["candidate_before_v1_entry"] for r in confirmed),
                    "v1_nonconfirmed_with_candidate_before_censor": sum((not r["v1_confirmed"]) and r["candidate_ns"] is not None for r in ledgers),
                    "note": "Secondary only. Detector thresholds are frozen without using outcomes; unavailable entry/outcome labels are omitted."}
    _write_json(repo / OUT / "winner-loser-diagnostics.json", winner_loser)
    _write_csv(repo / OUT / "spring-date-results.csv", date_results)
    source_integrity = {"status": "PASS" if len(date_results) == (35 if full_run else 5) and all(x["ts_recv_backward_steps"] == x["ts_event_backward_steps"] == 0 for x in date_results) else "PARTIAL",
                       "dates": date_results, "episode_count": len(ledgers), "pilot_technical_gate_pass": pilot_pass,
                       "prior_artifacts_preserved": True, "all_source_hashes_verified_before_scan": True}
    _write_json(repo / OUT / "source-integrity.json", source_integrity)
    summary = {"study_id": STUDY_ID, "detector_spec_sha256": spec_hash, "pilot_episode_count": 350,
               "full_spring_episode_count": len(ledgers) if full_run else None, "dates_processed": [x["date"] for x in date_results],
               "classification_counts": dict(statuses), "pilot_technical_gate": "PASS" if pilot_pass else "FAIL",
               "full_spring_expansion": full_run, "predictive_information": "inconclusive", "economic_tradability": "not_yet_tested",
               "runtime_seconds": time.monotonic()-started, "october_accessed": False, "protected_oos_accessed": False,
               "data_downloaded": False, "commit_performed": False}
    _write_json(repo / OUT / "summary.json", summary)
    limitations = """# Scientific limitations\n\nThis is a descriptive detector study, not a trading strategy. Numeric conventions (2x stored V1 heavy threshold, two executions, one tick limited impact, four ticks through-zone continuation, one tick rejection) are frozen descriptors and are not validated optimal microstructure thresholds. MBP-10 shows aggregate displayed depth, not order identities, hidden liquidity, queue priority, passive fills, or proof that a trade caused a depth recovery. Episodes overlap in time and zones; 35 dates are the independent clustering unit, not thousands of independent samples. Stage-conditioned forward responses have different classification times and survivor/censor structures; the study explicitly does not identify incremental predictive value without defensible decision-time matching. A mechanically causal classification is not evidence of predictive advantage or economic tradability.\n"""
    (repo / OUT / "scientific-limitations.md").write_text(limitations, encoding="utf-8")
    decision = """# Next strategy-design decision\n\nNo trading strategy is justified by detector mechanics alone. Review date-clustered, source-valid forward outcomes and the explicit comparability limitation. If estimates remain inconclusive, keep this as measurement research; do not tune thresholds on these outcomes. Economic tradability has not been tested.\n"""
    (repo / OUT / "next-strategy-design-decision.md").write_text(decision, encoding="utf-8")
    report = (f"# {STUDY_ID}\n\nStatus: {'PASS' if source_integrity['status']=='PASS' else 'PARTIAL'}; detector spec SHA-256 `{spec_hash}`.\n\n"
              f"Episodes processed: {len(ledgers)}; pilot episodes: 350; processed dates: {len(date_results)}.\n\n"
              f"Classifications: `{dict(statuses)}`.\n\n"
              "The detector requires repeated aggressive prints, price-impact evaluation, and original-zone opposing-depth response. Candidate and later rejection are separate states. This output is exploratory; predictive information is inconclusive and economic tradability was not tested. See forward-response and comparability artifacts.\n")
    (repo / OUT / "report.md").write_text(report, encoding="utf-8")
    _write_json(repo / OUT / "test-results.json", {"status": "PENDING", "tests": None})
    run_manifest = {"study_id": STUDY_ID, "complete": full_run and source_integrity["status"] == "PASS", "pilot_gate": pilot_pass,
                    "spec_sha256": spec_hash, "dates": [x["date"] for x in date_results], "episode_count": len(ledgers),
                    "runtime_seconds": time.monotonic()-started, "october_accessed": False, "protected_oos_accessed": False,
                    "data_downloaded": False, "v1_changed": False, "commit_performed": False}
    _write_json(repo / OUT / "run-manifest.json", run_manifest)
    names = ["report.md", "summary.json", "frozen-detector-specification.json", "detector-specification.sha256",
             "detector-state-transitions.parquet", "detector-event-ledger.csv", "classification-funnel.json",
             "detection-timing-analysis.csv", "aggression-impact-depth-ablation.json", "forward-response-by-class.csv",
             "decision-time-comparability.json", "winner-loser-diagnostics.json", "source-integrity.json",
             "spring-date-results.csv", "test-results.json", "scientific-limitations.md", "next-strategy-design-decision.md", "run-manifest.json"]
    _write_json(repo / OUT / "artifact-hashes.json", {name: _sha(repo / OUT / name) for name in names if (repo / OUT / name).is_file()})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pilot-only", action="store_true", help="stop after the five-date technical pilot")
    parser.add_argument("--forward-only", action="store_true", help="recompute fixed-horizon outcomes from frozen transitions")
    args = parser.parse_args()
    summary = recompute_forward_only(args.repo.resolve()) if args.forward_only else run(args.repo.resolve(), args.pilot_only)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
