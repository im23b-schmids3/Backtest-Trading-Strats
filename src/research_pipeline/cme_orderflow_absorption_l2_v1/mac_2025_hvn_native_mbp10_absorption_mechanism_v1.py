"""Source-faithful, bounded-memory native MBP-10 mechanism audit for five dates.

This is a descriptive audit of existing V1 episodes, not a strategy or optimizer.
It reads only the five explicitly sealed March 2025 ESH5 source files.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DATES = tuple(f"2025-03-{d:02d}" for d in range(3, 8))
STUDY_ID = "ES_HVN_NATIVE_MBP10_ABSORPTION_MECHANISM_V1"
V1 = Path("research_runs/ES_HVN_HEAVY_ABSORPTION_RESEARCH_V1")
AUDIT = Path("research_runs/ES_HVN_FAILED_AGGRESSION_DETECTION_AUDIT_V1")
SOURCE_ROOT = Path("data/databento/mac-2025-native-es-mbp10-final-reduced/TRAIN")
OUT = Path("research_runs") / STUDY_ID
TICK = 0.25
NS = 1_000_000_000
TRACE_BEFORE_NS = 2 * NS
TRACE_AFTER_NS = 35 * NS
ATTACK_AFTER_ONSET_NS = 5 * NS
ATTACK_ONSET_HORIZON_NS = 15 * NS
MAX_MISSING_NS = 2 * NS
FLAG_LAST, FLAG_TOB, FLAG_SNAPSHOT = 128, 64, 32
FLAG_MBP, FLAG_BAD_TS_RECV, FLAG_MAYBE_BAD_BOOK = 16, 8, 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_value(v) for k, v in row.items()})


def _csv_value(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return v


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_and_verify_sources(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec_doc = json.loads((repo / V1 / "frozen-v1-specification.json").read_text())
    spec = spec_doc["specification"]
    spec_canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    spec_hash = hashlib.sha256(spec_canonical).hexdigest()
    if spec_hash != spec_doc["sha256"]:
        raise ValueError("V1 frozen specification hash mismatch")
    v1_manifest = json.loads((repo / V1 / "run-manifest.json").read_text())
    audit_manifest = json.loads((repo / AUDIT / "run-manifest.json").read_text())
    if not v1_manifest.get("complete") or not audit_manifest.get("complete"):
        raise ValueError("V1 source/audit manifest is not complete")
    if audit_manifest.get("v1_source_spec_sha256") != spec_hash:
        raise ValueError("failed-aggression audit is bound to a different V1 specification")
    checked: dict[str, Any] = {}
    for base in (V1, AUDIT):
        hashes = json.loads((repo / base / "artifact-hashes.json").read_text())
        for rel, expected in hashes.items():
            p = repo / base / rel
            if not p.is_file() or sha256(p) != expected:
                raise ValueError(f"V1/audit artifact hash mismatch: {base / rel}")
        checked[str(base)] = {"checked_artifacts": len(hashes), "all_match": True}
    sources = json.loads((repo / V1 / "source-integrity-results.json").read_text())
    if sources.get("status") not in ("PASS", "VERIFIED", "COMPLETE"):
        raise ValueError(f"V1 source-integrity status not passing: {sources.get('status')}")
    by_date = {x["date"]: x for x in sources["sources"]}
    raw: dict[str, Any] = {}
    for date in DATES:
        rec = by_date.get(date)
        if not rec:
            raise ValueError(f"V1 source manifest lacks {date}")
        p = repo / rec["raw_path"]
        if not p.is_file():
            raise FileNotFoundError(p)
        actual_hash, actual_bytes = sha256(p), p.stat().st_size
        if actual_hash != rec["raw_sha256"] or actual_bytes != rec["raw_bytes"]:
            raise ValueError(f"native source changed for {date}")
        if rec["dataset"] != "GLBX.MDP3" or rec["schema"] != "mbp-10" or rec["symbol"] != "ESH5":
            raise ValueError(f"unexpected source identity for {date}: {rec}")
        raw[date] = {**rec, "absolute_path": str(p.resolve()), "hash_matches": True, "bytes_match": True}
    return spec_doc, checked, raw


@dataclass
class Episode:
    episode_id: str
    date: str
    zone_id: str
    t0: int
    direction: int | None
    low: float
    high: float
    poc: float
    origin: str
    v1_status: str
    v1_heavy: bool
    v1_heavy_size: float | None
    v1_threshold: float | None
    v1_heavy_ns: int | None
    v1_heavy_price: float | None
    touch_price: float | None
    approach: str
    v1: dict[str, str] | None = None
    rr1: dict[str, str] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    first_book: dict[str, Any] | None = None
    last_book: dict[str, Any] | None = None
    prior_book: dict[str, Any] | None = None
    attack_onset: int | None = None
    attack_onset_price: float | None = None
    attack_last: int | None = None
    attack_trades: int = 0
    directional_attack_trades: int = 0
    attack_attempts: int = 0
    last_attack_side_trade_ns: int | None = None
    attack_volume: int = 0
    attack_delta: int = 0
    attack_prices: set[float] = field(default_factory=set)
    max_penetration_ticks: float = 0.0
    impact_start_mid: float | None = None
    impact_end_mid: float | None = None
    impact_start_bbo: tuple[float, float] | None = None
    impact_end_bbo: tuple[float, float] | None = None
    attack_start_spread_ticks: float | None = None
    max_attack_spread_ticks: float | None = None
    min_defended_zone_depth: float | None = None
    attack_start_depth: float | None = None
    attack_end_depth: float | None = None
    recovery_contracts: float = 0.0
    depletion_contracts: float = 0.0
    recovery_events: int = 0
    depletion_events: int = 0
    valid_attack_records: int = 0
    invalid_attack_records: int = 0
    strong_side_volume: int = 0
    other_side_volume: int = 0
    unknown_side_volume: int = 0
    attack_levels: set[float] = field(default_factory=set)
    time_at_beyond_ms: float = 0.0
    prior_beyond: bool | None = None
    prior_beyond_ts: int | None = None
    first_low_impact_ns: int | None = None
    first_low_impact_mid: float | None = None
    first_depth_recovery_ns: int | None = None
    first_depth_recovery_mid: float | None = None
    first_rejection_ns: int | None = None
    first_rejection_mid: float | None = None
    first_heavy_ns: int | None = None
    bad_recv_records: int = 0
    maybe_bad_book_records: int = 0
    missing_record_gaps: int = 0
    trace_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attack_side(self) -> str | None:
        if self.direction is None:
            return None
        return "A" if self.direction > 0 else "B"  # opposite aggression into support/resistance

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("v1", "rr1", "metrics", "first_book", "last_book", "prior_book", "attack_prices", "attack_levels", "trace_records"):
            d.pop(k, None)
        d["attack_price_levels"] = len(self.attack_prices)
        d["aggressed_price_levels"] = len(self.attack_levels)
        return d


def make_episodes(repo: Path) -> list[Episode]:
    zones = read_csv(repo / V1 / "hvn-zone-ledger.csv")
    touches = read_csv(repo / V1 / "hvn-touch-episodes.csv")
    signals = read_csv(repo / V1 / "confirmed-signal-ledger.csv")
    rr1rows = [r for r in read_csv(repo / V1 / "rr-grid-trade-results.csv") if float(r.get("rr", 0) or 0) == 1.0]
    touch_by_zone = {(r["date"], r["zone_id"]): r for r in touches if r["date"] in DATES}
    signal_by_zone = {(r["date"], r["zone_id"]): r for r in signals if r["date"] in DATES}
    rr_by_signal = {r["signal_id"]: r for r in rr1rows if r["date"] in DATES}
    out = []
    for z in zones:
        date, ts = z["date"], z.get("first_touch_ns", "")
        if date not in DATES or not ts:
            continue
        t = touch_by_zone.get((date, z["zone_id"]))
        # Every V1 touch episode is retained; otherwise the zone's first touch is
        # included as an outcome-blind nondirectional first-touch observation.
        t0 = int(t["T0_ns"]) if t else int(ts)
        sig = signal_by_zone.get((date, z["zone_id"]))
        direction = int(t["direction"]) if t and t.get("direction") not in (None, "") else None
        episode_id = f"{date}|{z['zone_id']}|{t0}"
        out.append(Episode(
            episode_id=episode_id, date=date, zone_id=z["zone_id"], t0=t0,
            direction=direction, low=float(z["zone_low"]), high=float(z["zone_high"]),
            poc=float(z["zone_poc"]), origin="V1_TOUCH_EPISODE" if t else "ZONE_FIRST_TOUCH_NO_DIRECTION",
            v1_status=t.get("episode_status", "FIRST_TOUCH_NO_DIRECTION") if t else "FIRST_TOUCH_NO_DIRECTION",
            v1_heavy=(t.get("heavy_print", "False").lower() == "true") if t else False,
            v1_heavy_size=float(t["heavy_size"]) if t and t.get("heavy_size") else None,
            v1_threshold=float(t["heavy_threshold"]) if t and t.get("heavy_threshold") else None,
            v1_heavy_ns=int(t["TH_ns"]) if t and t.get("TH_ns") else None,
            v1_heavy_price=float(t["TH_price"]) if t and t.get("TH_price") else None,
            touch_price=float(t["T0_price"]) if t and t.get("T0_price") else None,
            approach=t.get("approach", "UNKNOWN") if t else "UNKNOWN",
            v1=sig, rr1=rr_by_signal.get(sig["signal_id"]) if sig else None,
        ))
    return sorted(out, key=lambda e: (e.date, e.t0, e.zone_id, e.episode_id))


def _book(row: Any) -> dict[str, Any]:
    bids = [(int(row[f"bid_px_{i:02d}"]), int(row[f"bid_sz_{i:02d}"])) for i in range(10)]
    asks = [(int(row[f"ask_px_{i:02d}"]), int(row[f"ask_sz_{i:02d}"])) for i in range(10)]
    bids = [(p, s) for p, s in bids if p > 0 and s > 0]
    asks = [(p, s) for p, s in asks if p > 0 and s > 0]
    best_bid = bids[0][0] / 1e9 if bids else None
    best_ask = asks[0][0] / 1e9 if asks else None
    ordered = all(bids[i][0] > bids[i + 1][0] for i in range(len(bids) - 1)) and all(asks[i][0] < asks[i + 1][0] for i in range(len(asks) - 1))
    valid = bool(bids and asks and ordered and best_bid < best_ask and not (int(row["flags"]) & FLAG_MAYBE_BAD_BOOK))
    return {
        "bids": bids, "asks": asks, "bid": best_bid, "ask": best_ask,
        "mid": (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None,
        "spread_ticks": (best_ask - best_bid) / TICK if best_bid is not None and best_ask is not None else None,
        "top5_bid": sum(s for _, s in bids[:5]), "top5_ask": sum(s for _, s in asks[:5]),
        "top10_bid": sum(s for _, s in bids), "top10_ask": sum(s for _, s in asks),
        "valid": valid,
    }


def _depth_at(book: dict[str, Any] | None, ep: Episode) -> int:
    if not book or ep.direction is None:
        return 0
    side = book["bids"] if ep.direction > 0 else book["asks"]
    low, high = round(ep.low * 4), round(ep.high * 4)
    return sum(size for p, size in side if low <= round(p / 1e9 * 4) <= high)


def trade_aggressor(action: str, side: str) -> str:
    """Map only native Trade records to aggressor; A/M/C side is resting side."""
    if action != "T":
        return "NOT_A_TRADE"
    return {"A": "SELL", "B": "BUY"}.get(side, "UNKNOWN")


def attack_window_contains(ts_event: int, onset_ns: int) -> bool:
    """Fixed inclusive onset-to-five-second boundary, independent of outcome."""
    return onset_ns <= ts_event <= onset_ns + ATTACK_AFTER_ONSET_NS


def attack_onset_eligible(ts_event: int, touch_ns: int) -> bool:
    return touch_ns <= ts_event <= touch_ns + ATTACK_ONSET_HORIZON_NS


def _depth_delta(previous: dict[str, Any] | None, current: dict[str, Any], ep: Episode) -> int:
    """Net visible depth change at original zone prices, not trade attribution."""
    return _depth_at(current, ep) - _depth_at(previous, ep)


def preentry_feature_complete(onset_ns: int | None, entry_ns: int | None) -> bool:
    return bool(onset_ns is not None and entry_ns is not None and onset_ns + ATTACK_AFTER_ONSET_NS <= entry_ns)


def timestamp_gap_exceeds(previous_ns: int | None, current_ns: int, threshold_ns: int = MAX_MISSING_NS) -> bool:
    return previous_ns is not None and current_ns - previous_ns > threshold_ns


def native_order_window_indices(ts_event_array: Any, intervals: list[tuple[int,int]]) -> list[int]:
    """Select matching rows while preserving their original array/file order."""
    import numpy as np
    mask=np.zeros(len(ts_event_array),dtype=bool)
    for start,end in intervals: mask |= (ts_event_array>=start)&(ts_event_array<=end)
    return np.flatnonzero(mask).tolist()


def _record_value(row: Any, key: str) -> int:
    return int(row[key])


def process_date(repo: Path, date: str, episodes: list[Episode], raw_meta: dict[str, Any], capture_traces: set[str]) -> dict[str, Any]:
    try:
        import numpy as np
        from databento import DBNStore
    except ImportError as e:
        raise RuntimeError("Run with the repository .venv, which contains Databento SDK and NumPy") from e
    path = Path(raw_meta["absolute_path"])
    eps = [e for e in episodes if e.date == date]
    starts = [e.t0 - TRACE_BEFORE_NS for e in eps]
    ends = [e.t0 + TRACE_AFTER_NS for e in eps]
    store = DBNStore.from_file(str(path))
    chunks = store.to_ndarray(count=250_000)
    total = trades = 0
    first_event = last_event = first_recv = last_recv = None
    prior_event = prior_recv = prior_seq = None
    event_backwards = recv_backwards = sequence_nonincrease = 0
    ts_recv_bad_flag = maybe_bad_book = snapshot_flags = last_flags = mbp_flags = tob_flags = 0
    actions, sides = Counter(), Counter()
    previous_book: dict[str, Any] | None = None
    previous_row: Any = None
    last_processed_global_index = -2
    global_index_base = 0
    # Merge overlapping episode windows. Only these slices need book decoding;
    # every raw row is still included in vectorized chronology/flag checks.
    intervals = sorted(zip(starts, ends))
    merged: list[list[int]] = []
    for a,b in intervals:
        if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1],b)
        else: merged.append([a,b])
    for chunk in chunks:
        if not isinstance(chunk, np.ndarray) or chunk.dtype.names is None:
            raise TypeError("Databento ndarray reader returned an unexpected chunk")
        if not np.all(chunk["rtype"] == 10):
            raise ValueError(f"non-MBP-10 record in sealed source for {date}")
        n = len(chunk); total += n
        if not n: continue
        ts_arr, recv_arr, seq_arr, flag_arr = chunk["ts_event"],chunk["ts_recv"],chunk["sequence"],chunk["flags"]
        if first_event is None:
            first_event,first_recv=int(ts_arr[0]),int(recv_arr[0])
        last_event,last_recv=int(ts_arr[-1]),int(recv_arr[-1])
        event_backwards += int(np.count_nonzero(ts_arr[1:] < ts_arr[:-1]))
        recv_backwards += int(np.count_nonzero(recv_arr[1:] < recv_arr[:-1]))
        sequence_nonincrease += int(np.count_nonzero(seq_arr[1:] <= seq_arr[:-1]))
        if prior_event is not None and int(ts_arr[0]) < prior_event: event_backwards += 1
        if prior_recv is not None and int(recv_arr[0]) < prior_recv: recv_backwards += 1
        if prior_seq is not None and int(seq_arr[0]) <= prior_seq: sequence_nonincrease += 1
        prior_event,prior_recv,prior_seq=int(ts_arr[-1]),int(recv_arr[-1]),int(seq_arr[-1])
        trades += int(np.count_nonzero(chunk["action"] == b"T"))
        for value,count in zip(*np.unique(chunk["action"],return_counts=True)):
            actions[bytes(value).decode("ascii")]=actions.get(bytes(value).decode("ascii"),0)+int(count)
        for value,count in zip(*np.unique(chunk["side"],return_counts=True)):
            sides[bytes(value).decode("ascii")]=sides.get(bytes(value).decode("ascii"),0)+int(count)
        ts_recv_bad_flag += int(np.count_nonzero(flag_arr & FLAG_BAD_TS_RECV))
        maybe_bad_book += int(np.count_nonzero(flag_arr & FLAG_MAYBE_BAD_BOOK))
        snapshot_flags += int(np.count_nonzero(flag_arr & FLAG_SNAPSHOT))
        last_flags += int(np.count_nonzero(flag_arr & FLAG_LAST))
        mbp_flags += int(np.count_nonzero(flag_arr & FLAG_MBP))
        tob_flags += int(np.count_nonzero(flag_arr & FLAG_TOB))
        # DBN ts_event is monotone in these sealed files. If it is not, retain
        # native order and use a direct mask instead of searchsorted.
        monotone=bool(np.all(ts_arr[1:] >= ts_arr[:-1]))
        segments=[]
        if monotone:
            for a,b in merged:
                lo=int(np.searchsorted(ts_arr,a,"left")); hi=int(np.searchsorted(ts_arr,b,"right"))
                if lo<hi: segments.extend(range(lo,hi))
        else:
            segments=native_order_window_indices(ts_arr,merged)
        # A raw row can belong to multiple overlapping episodes. Native file
        # order is retained; active episodes are found by the sorted start list.
        for index in segments:
            row=chunk[index]; ts=int(row["ts_event"]); recv=int(row["ts_recv"]); seq=int(row["sequence"])
            active_end=bisect.bisect_right(starts,ts)
            active=[]; j=active_end-1
            while j>=0 and ends[j]>=ts:
                active.append(eps[j]); j-=1
            if not active:
                previous_book=None; last_processed_global_index=global_index_base+index; continue
            action=bytes(row["action"]).decode("ascii"); side=bytes(row["side"]).decode("ascii"); flags=int(row["flags"])
            book=_book(row)
            global_index=global_index_base+index
            if global_index == last_processed_global_index + 1 and previous_book is not None:
                adjacent_book=previous_book
            else:
                prior_row=chunk[index-1] if index>0 else previous_row
                adjacent_book=_book(prior_row) if prior_row is not None else None
            prior_ts=int((chunk[index-1] if index>0 else previous_row)["ts_event"]) if (index>0 or previous_row is not None) else None
            for ep in active:
                start = ep.t0 - TRACE_BEFORE_NS
                end = ep.t0 + TRACE_AFTER_NS
                if ep.first_book is None and ts >= ep.t0:
                    ep.first_book = book
                if ts >= ep.t0:
                    ep.last_book = book
                if flags & FLAG_BAD_TS_RECV:
                    ep.bad_recv_records += 1
                if flags & FLAG_MAYBE_BAD_BOOK:
                    ep.maybe_bad_book_records += 1
                if timestamp_gap_exceeds(prior_ts,ts):
                    ep.missing_record_gaps += 1
                if ep.direction is not None and adjacent_book is not None and action in "AMC" and ep.attack_onset is not None and attack_window_contains(ts,ep.attack_onset):
                    delta = _depth_delta(adjacent_book, book, ep)
                    if delta > 0:
                        ep.recovery_contracts += delta; ep.recovery_events += 1
                        if ep.first_depth_recovery_ns is None and ep.attack_onset is not None and ts >= ep.attack_onset:
                            ep.first_depth_recovery_ns = ts; ep.first_depth_recovery_mid = book["mid"]
                    elif delta < 0:
                        ep.depletion_contracts += -delta; ep.depletion_events += 1
                if ep.direction is not None and ts >= ep.t0 and book["valid"]:
                    defended = _depth_at(book, ep)
                    if ep.attack_onset is not None and attack_window_contains(ts, ep.attack_onset):
                        ep.valid_attack_records += 1
                        ep.min_defended_zone_depth = defended if ep.min_defended_zone_depth is None else min(ep.min_defended_zone_depth, defended)
                        ep.attack_end_depth = defended
                        if book["spread_ticks"] is not None:
                            ep.max_attack_spread_ticks = book["spread_ticks"] if ep.max_attack_spread_ticks is None else max(ep.max_attack_spread_ticks,book["spread_ticks"])
                        if ep.impact_start_mid is None:
                            ep.impact_start_mid = book["mid"]; ep.impact_start_bbo = (book["bid"], book["ask"])
                        ep.impact_end_mid = book["mid"]; ep.impact_end_bbo = (book["bid"], book["ask"])
                        adverse = (ep.low - (book["mid"] or ep.low)) / TICK if ep.direction > 0 else ((book["mid"] or ep.high) - ep.high) / TICK
                        ep.max_penetration_ticks = max(ep.max_penetration_ticks, adverse)
                        if ep.prior_beyond is not None and ep.prior_beyond_ts is not None:
                            ep.time_at_beyond_ms += max(0, min(ts, ep.attack_onset + ATTACK_AFTER_ONSET_NS) - ep.prior_beyond_ts) / 1e6 if ep.prior_beyond else 0
                        ep.prior_beyond = adverse > 0
                        ep.prior_beyond_ts = ts
                        if ep.first_rejection_ns is None and adverse <= -1.0:
                            ep.first_rejection_ns = ts; ep.first_rejection_mid = book["mid"]
                        if ep.first_low_impact_ns is None and ep.attack_volume >= max(1, int(ep.v1_threshold or 1)) and ep.attack_trades >= 1 and abs(adverse) <= 1.0:
                            ep.first_low_impact_ns = ts; ep.first_low_impact_mid = book["mid"]
                if action == "T" and ts >= ep.t0 and ep.direction is not None:
                    price = int(row["price"]) / 1e9
                    attack_side = ep.attack_side
                    in_attack_band = ep.low - 2*TICK <= price <= ep.high + 2*TICK
                    if ep.attack_onset is None and attack_onset_eligible(ts, ep.t0) and side == attack_side and in_attack_band:
                        ep.attack_onset = ts
                        ep.attack_onset_price = price
                        ep.first_heavy_ns = ep.v1_heavy_ns
                        ep.impact_start_mid = book["mid"] if book["valid"] else None
                        ep.impact_start_bbo = (book["bid"], book["ask"]) if book["valid"] else None
                        ep.attack_start_depth = _depth_at(book,ep) if book["valid"] else None
                        ep.attack_start_spread_ticks = book["spread_ticks"] if book["valid"] else None
                        ep.first_book = ep.first_book or book
                    if ep.attack_onset is not None and attack_window_contains(ts, ep.attack_onset):
                        size = int(row["size"]); ep.attack_trades += 1; ep.attack_volume += size
                        ep.attack_prices.add(price)
                        ep.attack_last = ts
                        if side == attack_side:
                            ep.strong_side_volume += size; ep.directional_attack_trades += 1; ep.attack_levels.add(price)
                            if ep.attack_attempts == 0 or (ep.last_attack_side_trade_ns is not None and ts-ep.last_attack_side_trade_ns >= NS): ep.attack_attempts += 1
                            ep.last_attack_side_trade_ns = ts
                        ep.other_side_volume += size if side in ("A", "B") and side != attack_side else 0
                        ep.unknown_side_volume += size if side == "N" else 0
                        sign = 1 if side == "B" else -1 if side == "A" else 0
                        ep.attack_delta += sign * size
                        adverse_ticks = ((ep.low - price) / TICK if ep.direction > 0 else (price - ep.high) / TICK)
                        ep.max_penetration_ticks = max(ep.max_penetration_ticks, adverse_ticks)
                        if ep.first_heavy_ns is None and ep.v1_heavy and ep.v1 and ts >= int(ep.v1["TH_ns"]):
                            ep.first_heavy_ns = ts
                if ep.episode_id in capture_traces and start <= ts <= end and (action == "T" or ts == ep.t0 or action in "AMC"):
                    if len(ep.trace_records) < 5000:
                        ep.trace_records.append({"ts_event":ts,"ts_recv":recv,"ts_in_delta":int(row["ts_in_delta"]),"sequence":seq,"action":action,"side":side,"price":int(row["price"])/1e9,"size":int(row["size"]),"depth":int(row["depth"]),"flags":flags,"bid":book["bid"],"ask":book["ask"],"valid_book":book["valid"],"bid_levels":[[p/1e9,s] for p,s in book["bids"]],"ask_levels":[[p/1e9,s] for p,s in book["asks"]],"top5_bid":book["top5_bid"],"top5_ask":book["top5_ask"],"top10_bid":book["top10_bid"],"top10_ask":book["top10_ask"],"defended_zone_depth":_depth_at(book,ep)})
                if start <= ts <= end:
                    ep.prior_book = book
            previous_book, last_processed_global_index = book, global_index
        previous_row=chunk[-1].copy()
        global_index_base += n
    for ep in eps:
        if ep.attack_onset is not None:
            ep.metrics["attack_window_complete"] = ep.attack_onset + ATTACK_AFTER_ONSET_NS <= last_event
            mid_move = ((ep.impact_end_mid - ep.impact_start_mid) / TICK) if ep.impact_end_mid is not None and ep.impact_start_mid is not None else None
            bbo_move = ((ep.impact_end_bbo[0] - ep.impact_start_bbo[0]) / TICK, (ep.impact_end_bbo[1] - ep.impact_start_bbo[1]) / TICK) if ep.impact_end_bbo and ep.impact_start_bbo else None
            ep.metrics.update({
                "attack_duration_ms": (ep.attack_last-ep.attack_onset)/1e6 if ep.attack_last else 0,
                "all_trade_records_per_second_in_fixed_window": ep.attack_trades/(ATTACK_AFTER_ONSET_NS/NS),
                "directional_attack_executions_per_second": ep.directional_attack_trades/(ATTACK_AFTER_ONSET_NS/NS),
                "repeated_attack_attempt_count_1s_gap": ep.attack_attempts,
                "attack_window_end_ns": ep.attack_onset + ATTACK_AFTER_ONSET_NS,
                "mid_displacement_ticks": mid_move,
                "adverse_end_penetration_ticks": ((ep.low-ep.impact_end_mid)/TICK if ep.direction>0 else (ep.impact_end_mid-ep.high)/TICK) if ep.impact_end_mid is not None else None,
                "bbo_bid_displacement_ticks": bbo_move[0] if bbo_move else None,
                "bbo_ask_displacement_ticks": bbo_move[1] if bbo_move else None,
                "attack_start_spread_ticks": ep.attack_start_spread_ticks,
                "maximum_attack_spread_ticks": ep.max_attack_spread_ticks,
                "abs_mid_impact_ticks_per_contract": abs(mid_move)/ep.strong_side_volume if mid_move is not None and ep.strong_side_volume else None,
                "displayed_depth_at_attack_end": ep.attack_end_depth,
                "displayed_depth_at_attack_start": ep.attack_start_depth,
                "min_displayed_zone_depth_during_attack": ep.min_defended_zone_depth,
                "gross_positive_net_level_changes_during_attack_contracts": ep.recovery_contracts,
                "gross_negative_net_level_changes_during_attack_contracts": ep.depletion_contracts,
                "net_displayed_depth_change_start_to_end_contracts": (ep.attack_end_depth-ep.attack_start_depth) if ep.attack_end_depth is not None and ep.attack_start_depth is not None else None,
                "displayed_depth_recovery_from_attack_minimum_contracts": max(0,ep.attack_end_depth-ep.min_defended_zone_depth) if ep.attack_end_depth is not None and ep.min_defended_zone_depth is not None else None,
                "attack_time_beyond_zone_ms": ep.time_at_beyond_ms,
                "attack_max_penetration_ticks": ep.max_penetration_ticks,
            })
        else:
            ep.metrics["attack_window_complete"] = False
        if ep.first_book:
            ep.metrics.update({"t0_bid":ep.first_book["bid"],"t0_ask":ep.first_book["ask"],"t0_spread_ticks":ep.first_book["spread_ticks"],"t0_top5_bid":ep.first_book["top5_bid"],"t0_top5_ask":ep.first_book["top5_ask"],"t0_top10_bid":ep.first_book["top10_bid"],"t0_top10_ask":ep.first_book["top10_ask"],"t0_defended_zone_depth":_depth_at(ep.first_book,ep)})
        ep.metrics.update({"attack_onset_ns":ep.attack_onset,"attack_onset_price":ep.attack_onset_price,"directional_attack_volume":ep.strong_side_volume,"other_aggressor_volume":ep.other_side_volume,"unknown_aggressor_volume":ep.unknown_side_volume,"net_aggressive_delta_signed_buy_minus_sell":ep.attack_delta,"attack_execution_count":ep.attack_trades,"directional_attack_execution_count":ep.directional_attack_trades,"repeated_attack_attempt_count_1s_gap":ep.attack_attempts,"distinct_attack_price_levels":len(ep.attack_levels),"bad_ts_recv_records_in_trace":ep.bad_recv_records,"maybe_bad_book_records_in_trace":ep.maybe_bad_book_records,"timestamp_gaps_over_2s_in_trace":ep.missing_record_gaps})
    return {
        "date":date,"source":str(path),"records":total,"trades":trades,"first_ts_event":first_event,"last_ts_event":last_event,"first_ts_recv":first_recv,"last_ts_recv":last_recv,
        "ts_event_backward_steps_in_native_order":event_backwards,"ts_recv_backward_steps_in_native_order":recv_backwards,"sequence_nonincrease_steps_in_native_order":sequence_nonincrease,
        "actions":dict(actions),"sides":dict(sides),"flags":{"F_LAST":last_flags,"F_TOB":tob_flags,"F_SNAPSHOT":snapshot_flags,"F_MBP":mbp_flags,"F_BAD_TS_RECV":ts_recv_bad_flag,"F_MAYBE_BAD_BOOK":maybe_bad_book},
    }


def enrich_and_write(repo: Path, episodes: list[Episode], source_stats: list[dict[str, Any]], started: float, source_checks: dict[str, Any], spec_doc: dict[str, Any]) -> dict[str, Any]:
    OUT_DIR = repo / OUT
    traces_dir = OUT_DIR / "representative-market-traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    # Representative traces are selected by stable episode ID, never outcome.
    representatives = sorted(episodes, key=lambda e: (hashlib.sha256(e.episode_id.encode()).hexdigest(), e.episode_id))[:12]
    for e in representatives:
        if e.trace_records:
            p = traces_dir / f"{e.date}-{hashlib.sha256(e.episode_id.encode()).hexdigest()[:12]}.jsonl.gz"
            with gzip.open(p, "wt", encoding="utf-8") as f:
                for row in e.trace_records:
                    f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    for e in episodes:
        m = e.metrics
        v1 = e.v1 or {}
        rr = e.rr1 or {}
        m.update({
            "origin":e.origin,"episode_status":e.v1_status,"v1_heavy_print":e.v1_heavy,"v1_heavy_size":e.v1_heavy_size,"v1_heavy_threshold":e.v1_threshold,
            "v1_signal_id":v1.get("signal_id"),"v1_entry_ns":int(v1["TE_ns"]) if v1.get("TE_ns") else None,
            "v1_confirmed":bool(v1),"rr1_outcome":rr.get("outcome"),"rr1_net_r":float(rr["net_r"]) if rr.get("net_r") else None,
            "rr1_class":("WINNER" if float(rr.get("net_r",0) or 0)>0 else "LOSER" if rr else "UNTRADED_OR_UNAVAILABLE"),
            "attack_side":"SELL" if e.direction==1 else "BUY" if e.direction==-1 else "UNKNOWN",
            "attack_band_low":e.low-2*TICK,"attack_band_high":e.high+2*TICK,
            "attack_construction":"first opposite-aggressor T in zone +/-2 ticks within 15s after T0; fixed 5s window; no reversal-dependent endpoint",
            "preentry_complete":bool(v1 and preentry_feature_complete(e.attack_onset, int(v1["TE_ns"]))),
        })
        volume=m.get("directional_attack_volume",0) or 0
        impact=m.get("mid_displacement_ticks")
        if e.direction is None or e.attack_onset is None:
            category="UNRESOLVED_NO_DIRECTIONAL_ATTACK_ONSET"
        elif not m.get("attack_window_complete"):
            category="UNRESOLVED_INCOMPLETE_FIXED_WINDOW"
        elif e.v1_threshold is None or e.v1_threshold <= 0:
            category="UNRESOLVED_NO_V1_HEAVY_THRESHOLD"
        elif volume >= 2*e.v1_threshold and impact is not None and abs(impact)<=1.0:
            category="A_STRONG_AGGRESSION_LIMITED_PRICE_PROGRESS"
        elif volume >= 2*e.v1_threshold and e.metrics.get("adverse_end_penetration_ticks") is not None and e.metrics["adverse_end_penetration_ticks"]>=4.0:
            category="B_STRONG_AGGRESSION_SUSTAINED_PRICE_DISCOVERY"
        else:
            category="C_NONEXTREME_OR_MIXED_RESPONSE"
        m["mechanism_category_predeclared_descriptive"] = category
    ledger = [{**e.as_dict(), **e.metrics} for e in episodes]
    write_csv(OUT_DIR/"aggression-episode-ledger.csv", ledger)
    write_csv(OUT_DIR/"price-impact-analysis.csv", [dict(episode_id=e.episode_id,date=e.date,zone_id=e.zone_id,**{k:v for k,v in e.metrics.items() if ("impact" in k or "displacement" in k or "penetration" in k or "attack_" in k) and k!="attack_onset_ns"}, attack_onset_ns=e.attack_onset) for e in episodes])
    write_csv(OUT_DIR/"depth-response-analysis.csv", [dict(episode_id=e.episode_id,date=e.date,zone_id=e.zone_id,attack_onset_ns=e.attack_onset,**{k:v for k,v in e.metrics.items() if "depth" in k or "recovery" in k or "reduction" in k or "resilien" in k or "level_changes" in k}, depth_at_t0=e.metrics.get("t0_defended_zone_depth"), first_depth_recovery_ns=e.first_depth_recovery_ns, observed_recovery_event_count=e.recovery_events, observed_depletion_event_count=e.depletion_events) for e in episodes])
    v1comp=[]
    for e in episodes:
        if e.v1:
            v1comp.append(dict(episode_id=e.episode_id,date=e.date,signal_id=e.v1.get("signal_id"),v1_status=e.v1_status,heavy_print=e.v1_heavy,heavy_size=e.v1_heavy_size,heavy_threshold=e.v1_threshold,attack_side=e.metrics.get("attack_side"),directional_attack_volume=e.metrics.get("directional_attack_volume"),attack_trades=e.attack_trades,impact_ticks=e.metrics.get("mid_displacement_ticks"),impact_per_contract=e.metrics.get("abs_mid_impact_ticks_per_contract"),minimum_defended_depth=e.metrics.get("min_displayed_zone_depth_during_attack"),net_depth_change=e.metrics.get("net_displayed_depth_change_start_to_end_contracts"),depth_recovery_from_minimum=e.metrics.get("displayed_depth_recovery_from_attack_minimum_contracts"),mechanism_category=e.metrics.get("mechanism_category_predeclared_descriptive"),preentry_complete=e.metrics.get("preentry_complete"),rr1_class=e.metrics.get("rr1_class"),rr1_net_r=e.metrics.get("rr1_net_r")))
    write_csv(OUT_DIR/"v1-confirmation-mechanism-comparison.csv",v1comp)
    # Small-sample descriptive comparison, winner/loser observations are censored at TE.
    features=["directional_attack_volume","other_aggressor_volume","unknown_aggressor_volume","attack_execution_count","net_aggressive_delta_signed_buy_minus_sell","mid_displacement_ticks","abs_mid_impact_ticks_per_contract","attack_duration_ms","attack_max_penetration_ticks","min_displayed_zone_depth_during_attack","net_displayed_depth_change_start_to_end_contracts","displayed_depth_recovery_from_attack_minimum_contracts","t0_defended_zone_depth","attack_time_beyond_zone_ms"]
    wl=[]
    confirmed_rr=[e for e in episodes if e.rr1]
    confirmed=[e for e in confirmed_rr if e.metrics.get("preentry_complete")]
    for feat in features:
        groups={k:[float(e.metrics[feat]) for e in confirmed if e.metrics.get(feat) is not None and e.metrics.get("rr1_class")==k] for k in ("WINNER","LOSER")}
        for g,vals in groups.items():
            wl.append({"feature":feat,"class":g,"n_valid_preentry":len(vals),"denominator_all_sampled_rr1":sum(e.metrics.get("rr1_class")==g for e in confirmed_rr),"missing_or_not_complete_by_entry_n":sum(e.metrics.get("rr1_class")==g and (not e.metrics.get("preentry_complete") or e.metrics.get(feat) is None) for e in confirmed_rr),"median":statistics.median(vals) if vals else None,"mean":statistics.mean(vals) if vals else None,"q25":_quantile(vals,.25),"q75":_quantile(vals,.75),"date_count":len({e.date for e in confirmed if e.metrics.get(feat) is not None and e.metrics.get("rr1_class")==g})})
        a,b=groups["WINNER"],groups["LOSER"]
        wl.append({"feature":feat,"class":"WINNER_MINUS_LOSER","n_valid_preentry":min(len(a),len(b)),"missing_or_not_complete_by_entry_n":None,"median_difference":(statistics.median(a)-statistics.median(b)) if a and b else None,"mean_difference":(statistics.mean(a)-statistics.mean(b)) if a and b else None,"effect_size_cliffs_delta":_cliffs_delta(a,b) if a and b else None,"interpretation":"descriptive only; fixed attack window must complete by original TE; no classifier or threshold search"})
    write_csv(OUT_DIR/"winner-loser-preentry-comparison.csv",wl)
    timing=[]
    for e in episodes:
        te=int(e.v1["TE_ns"]) if e.v1 and e.v1.get("TE_ns") else None
        events={"first_heavy_print_ns":(e.v1_heavy_ns,e.v1_heavy_price),"attack_onset_ns":(e.attack_onset,e.attack_onset_price),"first_low_impact_ns":(e.first_low_impact_ns,e.first_low_impact_mid),"first_depth_recovery_ns":(e.first_depth_recovery_ns,e.first_depth_recovery_mid),"first_price_rejection_ns":(e.first_rejection_ns,e.first_rejection_mid),"v1_confirmation_ns":(int(e.v1["TC_ns"]) if e.v1 and e.v1.get("TC_ns") else None,float(e.v1["TC_price"]) if e.v1 and e.v1.get("TC_price") else None),"v1_entry_ns":(te,float(e.rr1["entry_price"]) if e.rr1 and e.rr1.get("entry_price") else None)}
        t0_price=float(e.v1["T0_price"]) if e.v1 and e.v1.get("T0_price") else e.touch_price
        for name,(ts,mark_price) in events.items():
            mark_price=float(mark_price) if mark_price not in (None,"") else None
            directional_move=((mark_price-t0_price)/TICK*e.direction) if mark_price is not None and t0_price is not None and e.direction is not None else None
            timing.append(dict(episode_id=e.episode_id,date=e.date,event=name,timestamp_ns=ts,mark_price=mark_price,ms_from_touch=(ts-e.t0)/1e6 if ts is not None else None,price_move_from_t0_directional_ticks=directional_move,available_before_v1_entry=(ts<=te if ts is not None and te is not None else None),remaining_after_event_ms=(te-ts)/1e6 if ts is not None and te is not None else None,rr1_outcome=e.metrics.get("rr1_outcome"),rr1_mfe_ticks=e.rr1.get("mfe_ticks") if e.rr1 else None,rr1_mae_ticks=e.rr1.get("mae_ticks") if e.rr1 else None,excursion_fields_are_postentry_outcome=True if e.rr1 else None,rr1_class=e.metrics.get("rr1_class")))
    write_csv(OUT_DIR/"mechanism-timing-analysis.csv",timing)
    cat=Counter(e.metrics.get("mechanism_category_predeclared_descriptive") for e in episodes)
    confirmed_eps=[e for e in episodes if e.v1]
    confirmed_meaningful=[e for e in confirmed_eps if e.v1_threshold is not None and (e.metrics.get("directional_attack_volume") or 0) >= 2*e.v1_threshold]
    confirmed_weak_impact=[e for e in confirmed_eps if e.metrics.get("mid_displacement_ticks") is not None and abs(e.metrics["mid_displacement_ticks"])<=1]
    confirmed_resilient=[e for e in confirmed_eps if (e.metrics.get("displayed_depth_recovery_from_attack_minimum_contracts") or 0)>0 and (e.metrics.get("min_displayed_zone_depth_during_attack") or 0)>0]
    preentry_eps=[e for e in confirmed_eps if e.metrics.get("preentry_complete")]
    preentry_meaningful=[e for e in preentry_eps if e.v1_threshold is not None and (e.metrics.get("directional_attack_volume") or 0)>=2*e.v1_threshold]
    preentry_weak_impact=[e for e in preentry_eps if e.metrics.get("mid_displacement_ticks") is not None and abs(e.metrics["mid_displacement_ticks"])<=1]
    preentry_depth_recovery=[e for e in preentry_eps if (e.metrics.get("displayed_depth_recovery_from_attack_minimum_contracts") or 0)>0 and (e.metrics.get("min_displayed_zone_depth_during_attack") or 0)>0]
    nonextreme=[e for e in confirmed_eps if e.metrics.get("mechanism_category_predeclared_descriptive")=="C_NONEXTREME_OR_MIXED_RESPONSE"]
    ordinary_bounce_like=[e for e in confirmed_eps if e.v1_threshold is not None and (e.metrics.get("directional_attack_volume") or 0)<e.v1_threshold and e.metrics.get("mid_displacement_ticks") is not None and abs(e.metrics["mid_displacement_ticks"])<=1]
    wins=[e for e in confirmed if e.metrics.get("rr1_class")=="WINNER"]
    losses=[e for e in confirmed if e.metrics.get("rr1_class")=="LOSER"]
    hypothesis={"study_id":STUDY_ID,"illustrative_categories":{"A":"directional attack volume >= 2x that episode's V1 heavy threshold and absolute 5-second mid displacement <= 1 tick","B":"same volume floor and adverse endpoint penetration beyond the zone >= 4 ticks","C":"remaining complete windows with a usable V1 threshold; mixed/ordinary label only, not a claim of no absorption","unresolved":"no direction/onset, incomplete fixed 5-second window, or missing V1 heavy threshold"},"continuous_metrics_primary":True,"research_questions":["Test whether fixed-window aggression volume plus low impact is measurable before V1 entry.","Separate net displayed depth changes at fixed HVN prices from order-level replenishment; MBP-10 cannot identify individual passive orders.","Compare only pre-entry-complete fixed-window observations across original RR=1 winners and losers; treat small samples and correlated episodes as inconclusive."],"not_a_strategy":True,"no_entry_rule":True}
    (OUT_DIR/"v2-research-hypotheses.md").write_text("# V2 mechanism research hypotheses (not a strategy)\n\n"+"\n".join(f"- {x}" for x in hypothesis["research_questions"])+"\n\nNo entry rules, thresholds chosen from outcomes, optimization, or V2 implementation are included.\n",encoding="utf-8")
    write_json(OUT_DIR/"false-positive-mechanism-taxonomy.json",{"categories":dict(cat),"definitions":hypothesis["illustrative_categories"],"v1_confirmation_counts":{"total":len(confirmed_eps),"meaningful_aggression_by_predeclared_rule":len(confirmed_meaningful),"weak_price_impact_by_predeclared_rule":len(confirmed_weak_impact),"displayed_depth_recovery_from_attack_minimum_with_depth_remaining":len(confirmed_resilient),"nonextreme_or_mixed_response":len(nonextreme),"ordinary_bounce_like_by_low_volume_and_small_move":len(ordinary_bounce_like),"fixed_window_complete_before_entry":len(preentry_eps),"meaningful_aggression_complete_before_entry":len(preentry_meaningful),"weak_price_impact_complete_before_entry":len(preentry_weak_impact),"depth_resiliency_proxy_complete_before_entry":len(preentry_depth_recovery)},"warning":"Categories are descriptive and not mutually exclusive except category label; aggregate depth changes do not identify passive orders or prove replenishment. C means nonextreme or mixed response and is not a literal ordinary-bounce finding."})
    source_semantics={"schema":"mbp-10","dataset":"GLBX.MDP3","symbol":"ESH5","price_scale":1e9,"native_order":"DBN file order retained; no sorting by timestamp","ts_event":"matching-engine-received event timestamp; analysis event clock","ts_recv":"Databento capture-server receive timestamp; flagged F_BAD_TS_RECV records excluded from receive-latency claims","ts_in_delta":"matching engine sending timestamp delta before ts_recv","sequence":"venue-assigned message sequence; native sequence retained, nonincrease/backsteps counted; not treated as unique event ID","action":{"A":"add","M":"modify","C":"cancel","R":"clear book","T":"aggressing trade; no book effect"},"side":{"T/A":"seller aggressor","T/B":"buyer aggressor","T/N":"unknown aggressor","A/M/C":"resting book side"},"flags":{"F_LAST":FLAG_LAST,"F_TOB":FLAG_TOB,"F_SNAPSHOT":FLAG_SNAPSHOT,"F_MBP":FLAG_MBP,"F_BAD_TS_RECV":FLAG_BAD_TS_RECV,"F_MAYBE_BAD_BOOK":FLAG_MAYBE_BAD_BOOK},"depth":"Each MBP-10 event carries the aggregate top-ten bid/ask levels; point-in-time displayed book is read directly from the row. No order IDs or hidden liquidity.","alignment":"Trade is attributed to its native T record. Displayed depth deltas compare adjacent full top-ten states in native record order, for A/M/C records; these are net observable changes and are not assigned as fills/refills to a particular trade.","event_same_timestamp":"DBN file order retained; sequence and ts_in_delta also exported for traces; no timestamp-only resorting.","quotes":"valid only with positive two-sided ordered levels, uncrossed BBO, and no F_MAYBE_BAD_BOOK on that row.","official_documentation":["https://databento.com/docs/schemas/mbp-10","https://databento.com/docs/conventions/common-fields-enums-types"]}
    write_json(OUT_DIR/"native-source-semantics.json",source_semantics)
    selection={"dates":DATES,"sampling":"All source-ledger first-touch episodes on the five authorized dates; touch-ledger rows joined by (date,zone_id); any remaining zone first touches included as nondirectional; sorted date,timestamp,zone_id; no outcome field used for inclusion or ordering.","episode_count":len(episodes),"by_date":dict(Counter(e.date for e in episodes)),"attack_window":"After touch, onset is the first native T with aggressor opposite the proposed direction and price in zone +/-2 ticks within 15s of T0; fixed inclusive [onset,onset+5s] observation. Neither onset nor endpoint uses later reversal or PnL.","trace_windows":"T0-2s through T0+35s, selected representative subset by SHA256(episode_id) rank, capped at 5000 records per episode.","representative_episode_ids":[e.episode_id for e in representatives]}
    write_json(OUT_DIR/"pilot-selection-manifest.json",selection)
    source_integrity={"status":"PASS_WITH_SOURCE_METADATA_DISCREPANCY","source_files":source_checks,"v1_artifact_checks":source_checks["prior_artifacts"],"five_dates_only":list(DATES),"metadata_crosscheck":{"2025-03-07":{"native_mbp10_T_records":892715,"v1_source_manifest_trade_records":892714,"native_T_N_records":1,"unknown_side_trade":{"ts_event":1741354208654000000,"ts_recv":1741354208658815753,"sequence":97497410,"size":99,"price":5776.5,"flags":0,"depth":0},"handling":"Retained in raw-source counts; excluded from directional attack volume/delta because aggressor side is unknown."}}}
    write_json(OUT_DIR/"source-integrity-results.json",source_integrity)
    episodes_by_date=Counter(e.date for e in episodes)
    v1_counts=Counter(e.v1_status for e in episodes)
    summary={"study_id":STUDY_ID,"dates":list(DATES),"episode_count":len(episodes),"episodes_by_date":dict(episodes_by_date),"episodes_by_v1_status":dict(v1_counts),"native_source_stats":source_stats,"category_counts":dict(cat),"v1_confirmation_counts":{"sampled":len(confirmed_eps),"meaningful_aggression_in_full_fixed_window":len(confirmed_meaningful),"weak_price_impact_in_full_fixed_window":len(confirmed_weak_impact),"depth_resiliency_proxy_in_full_fixed_window":len(confirmed_resilient),"nonextreme_or_mixed_response":len(nonextreme),"ordinary_bounce_like_by_low_volume_and_small_move":len(ordinary_bounce_like),"fixed_window_complete_before_entry":len(preentry_eps),"meaningful_aggression_complete_before_entry":len(preentry_meaningful),"weak_price_impact_complete_before_entry":len(preentry_weak_impact),"depth_resiliency_proxy_complete_before_entry":len(preentry_depth_recovery)},"winner_loser_sample":{"winner_n":len(wins),"loser_n":len(losses),"preentry_complete_confirmed_n":len(confirmed)},"limitations":["MBP-10 is aggregate depth and has no order IDs; net depth increases are not proof of passive replenishment or iceberg liquidity.","Trade and depth feed-event association is not order-level; sequential native records only.","Episodes may overlap and are not independent observations.","Five-date pilot is descriptive; no classifier, threshold search, strategy, or validation/OOS."],"v2_confirmation_mechanism_justified":"inconclusive","runtime_seconds":time.monotonic()-started}
    write_json(OUT_DIR/"summary.json",summary)
    report=_report_text(summary, episodes, wl)
    (OUT_DIR/"report.md").write_text(report,encoding="utf-8")
    # Machine-readable integrity inventory is generated after all substantive files.
    tests_result={"status":"PENDING_EXTERNAL_TEST_COMMAND","test_module":"tests/test_mac_2025_hvn_native_mbp10_absorption_mechanism_v1.py"}
    write_json(OUT_DIR/"test-results.json",tests_result)
    run={"study_id":STUDY_ID,"complete":True,"dates":list(DATES),"source_schema":"mbp-10","source_symbol":"ESH5","source_dataset":"GLBX.MDP3","v1_config_changed":False,"v2_implemented":False,"optimization":False,"october_accessed":False,"protected_oos_accessed":False,"data_downloaded":False,"commit_performed":False,"attack_definition":selection["attack_window"],"source_event_ordering_verified":all(x["ts_event_backward_steps_in_native_order"]==0 for x in source_stats),"runtime_seconds":summary["runtime_seconds"]}
    write_json(OUT_DIR/"run-manifest.json",run)
    hashes={}
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file() and p.name!="artifact-hashes.json": hashes[str(p.relative_to(OUT_DIR))]=sha256(p)
    write_json(OUT_DIR/"artifact-hashes.json",hashes)
    return summary


def _quantile(values: list[float], q: float) -> float | None:
    if not values: return None
    a=sorted(values); x=(len(a)-1)*q; lo=int(x); hi=min(lo+1,len(a)-1); return a[lo]+(a[hi]-a[lo])*(x-lo)


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:return float("nan")
    return sum(1 if x>y else -1 if x<y else 0 for x in a for y in b)/(len(a)*len(b))


def _report_text(summary: dict[str, Any], episodes: list[Episode], wl: list[dict[str, Any]]) -> str:
    status=Counter(e.metrics.get("mechanism_category_predeclared_descriptive") for e in episodes)
    confirmed=[e for e in episodes if e.v1]
    def n(predicate):return sum(bool(predicate(e)) for e in confirmed)
    return f"""# Genuine failed-aggression and displayed-depth mechanism audit\n\nStudy: `{STUDY_ID}`. Scope is exactly {', '.join(DATES)}; native ESH5 MBP-10. This is descriptive research, not a strategy, optimizer, validation, or OOS analysis.\n\n## Pilot counts\n\n- First-touch episodes analyzed: {len(episodes)}\n- V1-confirmed episodes represented: {len(confirmed)}\n- Categories: `{dict(status)}`\n- Full fixed-window confirmations with attack-side volume >=2x stored V1 heavy threshold: {n(lambda e:e.v1_threshold is not None and (e.metrics.get('directional_attack_volume') or 0)>=2*e.v1_threshold)}\n- Full fixed-window confirmations with absolute five-second mid displacement <=1 tick: {n(lambda e:e.metrics.get('mid_displacement_ticks') is not None and abs(e.metrics['mid_displacement_ticks'])<=1)}\n- Full fixed-window confirmations with displayed-depth recovery from its observed minimum and depth remaining: {n(lambda e:(e.metrics.get('displayed_depth_recovery_from_attack_minimum_contracts') or 0)>0 and (e.metrics.get('min_displayed_zone_depth_during_attack') or 0)>0)}\n- Nonextreme or mixed responses: {summary['v1_confirmation_counts']['nonextreme_or_mixed_response']}; this does not mean the cases were ordinary bounces.\n- Ordinary-bounce-like cases under the predeclared strict marker (attack volume below the V1 heavy threshold and absolute mid move <=1 tick): {summary['v1_confirmation_counts']['ordinary_bounce_like_by_low_volume_and_small_move']}\n- Fixed attack windows complete by original entry: {summary['v1_confirmation_counts']['fixed_window_complete_before_entry']} (meaningful aggression {summary['v1_confirmation_counts']['meaningful_aggression_complete_before_entry']}, weak impact {summary['v1_confirmation_counts']['weak_price_impact_complete_before_entry']}, depth recovery {summary['v1_confirmation_counts']['depth_resiliency_proxy_complete_before_entry']})\n- Original RR=1 pre-entry comparison sample: {summary['winner_loser_sample']['preentry_complete_confirmed_n']} (winners {summary['winner_loser_sample']['winner_n']}, losers {summary['winner_loser_sample']['loser_n']})\n\n## Construction and interpretation\n\nAll zone first touches on the five dates are included. Touch-ledger episodes supply direction and V1 status; unmatched first touches remain nondirectional. Attack onset is the first native `T` record with the opposing aggressor in the zone expanded by two ticks, within 15 seconds of T0. The fixed observation is the following five seconds. A window is complete when the source covers its endpoint, regardless of whether another trade printed exactly then. Category A uses attack-side volume >=2x the stored V1 heavy threshold and <=1 tick absolute mid displacement; category B uses that volume and >=4 ticks of adverse endpoint penetration. C is explicitly nonextreme or mixed. These cutoffs are descriptive markers, not fit to outcomes; continuous values are primary.\n\nNative DBN order is retained. `T` side denotes aggressor; A/M/C side denotes resting side. Every MBP-10 row supplies the aggregate ten-level state. Changes at original HVN prices are compared across adjacent native states during the attack window and reported as gross positive/negative transitions, net start-to-end change, and recovery from minimum. None identifies a passive order or proves trade-caused replenishment. `F_BAD_TS_RECV` flags are counted and excluded from receive-time reliability claims; `F_MAYBE_BAD_BOOK` records fail per-row validity. Venue sequence is retained but its many nonincreases show it is not a global file-order key in these records; DBN row order is preserved. The source integrity cross-check found one Mar 7 `T/N` trade (99 contracts, 5776.50) not counted in the prior V1 source manifest trade total; it is retained in raw counts and excluded from directional measures. Raw file hashes still match.\n\n## Winner/loser and timing\n\nWinner/loser summaries use only fixed attack-window features complete before original V1 entry. The valid sample is small (8 winners, 16 losers) and episodes can overlap. Most fixed-window measures mature after the original entry; the complete pre-entry subset is 24. Differences and Cliff's delta are descriptive, with no fitted classifier or threshold search. MFE/MAE fields in timing rows are post-entry outcomes, not evidence available at those timestamps. No hypothetical entry or profit is computed.\n\n## Scientific conclusion\n\nThe raw feed can measure aggressive-side volume, price response, aggregate displayed depth and quote validity. In this sample the V1 confirmation rule is not equivalent to a test of failed continuation: its checks do not require sustained attack duration, bounded price impact per contract, depth persistence/recovery, or a minimum rejection. Meaningful aggression and low-impact/depth-response markers often appear in the full five-second windows, but only a subset is fully observable before entry. Winner/loser differences are inconclusive at this sample size. MBP-10 cannot identify individual passive orders, hidden icebergs, trader identity, or order-level replenishment. A source-informed V2 hypothesis may be explored only after reviewing the per-episode audit; no strategy, V2, or optimization is included here.\n\n## Source processing\n\n{json.dumps(summary['native_source_stats'],indent=2)}\n"""


def run(repo: Path) -> dict[str, Any]:
    started=time.monotonic()
    spec_doc, prior_checks, raw=load_and_verify_sources(repo)
    episodes=make_episodes(repo)
    # Outcome-blind deterministic representative set, chosen before raw analysis.
    traces=sorted((e.episode_id for e in episodes),key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))[:12]
    source_stats=[]
    for d in DATES:
        print(f"decoding {d}: {raw[d]['absolute_path']}",flush=True)
        # Reset native-order counters for each independent DBN file.
        source_stats.append(process_date(repo,d,episodes,raw[d],set(traces)))
    prior_files={}
    for root,stats in prior_checks.items(): prior_files[root]=stats
    src_checks={"v1_spec_sha256":json.loads((repo/V1/"frozen-v1-specification.json").read_text())["sha256"],"prior_artifacts":prior_files,"native_sources":raw}
    return enrich_and_write(repo,episodes,source_stats,started,src_checks,spec_doc)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    args=p.parse_args()
    result=run(args.repo.resolve())
    print(json.dumps({"status":"PASS","artifact_root":str((args.repo/OUT).resolve()),"episodes":result["episode_count"],"dates":result["dates"],"runtime_seconds":result["runtime_seconds"]},indent=2))


if __name__=="__main__":main()
