"""Native MBP-10 mechanism diagnostics for the frozen JJ continuation pairs.

Exploratory event study only: no strategy rules, thresholds, rematching, or
optimization.  Native records are streamed one authorized 2025 date at a time.
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

from . import mac_2025_es_only_train_baseline as baseline
from . import mac_2025_jj_fair_pricing_continuation_identification_v1 as frozen
from . import mac_2025_jj_fair_pricing_expanded_v2 as fair_v2
from . import mac_2025_jj_fair_pricing_v2_1_corrected_bos as v21
from . import mac_2025_jj_fair_pricing_v2_mechanism_diagnostic as fair_diag
from . import mac_2025_mlofi_event_study as mlofi
from .mac_2025_candidate_tape import load_tape

RUN_ID = "ES_JJ_CONTINUATION_NATIVE_MBP10_MECHANISM_V1"
OUT = Path("research_runs/CMEOrderflow_ES_JJ_CONTINUATION_NATIVE_MBP10_MECHANISM_V1")
SOURCE_COVERAGE = v21.OUT_ROOT / "source-coverage.json"
INPUT_ROOT = baseline.DATA_ROOT
TARGET_ESTIMANDS = {"DISPLACEMENT_ONLY", "COMBINED"}
UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
TICK_RAW = 250_000_000
TICK = 0.25
NS = 1_000_000_000
PRESIGNAL_WINDOWS = {"aggression_5s": 5, "aggression_30s": 30, "impact_5s": 5}
POST_WINDOWS = ((0, 5), (5, 30), (30, 120), (120, 300))


class MechanismError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    os.replace(tmp, path)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _input_contract() -> dict[str, Any]:
    # Frozen before any native MBP10 rows or feature/outcome relationships are opened.
    return {
        "study_id": RUN_ID,
        "status": "EXPLORATORY_INTERNAL_CONTRACT_NOT_PREREGISTRATION",
        "primary_pair_rule": "all rows of frozen signal-control-comparisons.csv with estimand DISPLACEMENT_ONLY or COMBINED; no rematching or replacement",
        "secondary_signal_rule": "all 142 frozen OPENING_CONTINUATION|DISPLACEMENT_CANDLE signal identities from the hash-verified V2.1 all-signals artifact; descriptive only and never substituted into primary paired estimates",
        "decision_time": "signal timestamp is frozen signal_id suffix; control minute is the zero-based completed-bar index, so control decision timestamp is session start plus (control_minute+1) minutes (the bar end)",
        "native_time": "DBN ts_recv, stable source order for ties, as used by the acquisition range audit and canonical reader",
        "policy_c": "A timestamp is feature-valid only if the latest source record at/before the query is executable; invalid/locked/crossed rows suspend the state until an executable full snapshot reopens it; no stale-state carry-through",
        "imbalance": "inverse-rank weighted TOP5 and TOP10 bid-minus-ask over total depth, multiplied by trade direction; weights 1/(rank+1)",
        "relative_opposing_depth": "inverse-rank opposing/directional depth ratio at decision divided by median ratio from 100ms as-of samples in [decision-60s, decision-5s]",
        "mlofi": "repository price-keyed accounting from mac_2025_mlofi_event_study.account_mbp_event; A/C/M explicit price-level size delta, T execution quantity on passive side, inverse-rank TOP5; no rank migration inferred as cancel",
        "mlofi_windows": {"instant": "(decision-500ms, decision]", "persistence": "eight consecutive 250ms bins ending at decision; fraction of nonzero bins matching latest nonzero bin sign"},
        "aggression": "MBP10 T records; side B=aggressive buy, A=aggressive sell; closed-right windows ending at decision; normalized delta=(buy-sell)/(buy+sell)*trade direction",
        "impact": "directional mid change over [decision-5s, decision] in ticks divided by directional aggressive contracts in same interval; absent for zero/low (<=1) contracts; prior adjacent 5s interval retained for change comparison",
        "resiliency": "opposing-side inverse-rank TOP5 displayed depth in [decision-5s, decision]; consumed=max(0, depth_at_start-min_depth), restored=max(0, depth_at_decision-min_depth), ratio=restored/consumed only when consumed>0",
        "postsig_windows_seconds": [list(x) for x in POST_WINDOWS],
        "post_signal_label": "explanatory outcome/path only; never an entry-time predictor",
        "decision_rule": "Support requires the prespecified directional feature pattern and feature/outcome relation to be directionally coherent in both Spring and October with date-cluster uncertainty reported; no threshold is selected. Otherwise report no stable mechanism or insufficient depth evidence.",
        "markout_execution": "same frozen 2ms next-valid BBO entry, ask long/bid short, one adverse tick each side, $6 round-trip commission converted at $12.50 per ES tick (=0.48 ticks); use quote at/before entry+300s",
        "inference": "paired descriptive associations; trading date is cluster; Spring and October reported independently; date-cluster bootstrap; no causal identification or untouched-validation claim",
        "threshold_search": False,
        "optimization": False,
        "new_exit_grid": False,
        "new_trading_filters": False,
    }


def _verify_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    hashes = _json(frozen.OUT_ROOT / "artifact-hashes.json")
    if hashes.get("status") != "HASHED":
        raise MechanismError("continuation-identification artifacts are not HASHED")
    bad = [name for name, digest in hashes.get("files", {}).items()
           if not (frozen.OUT_ROOT / name).is_file() or _sha(frozen.OUT_ROOT / name) != digest]
    if bad:
        raise MechanismError(f"frozen continuation artifact hash mismatch: {bad}")
    v21_manifest = _json(v21.OUT_ROOT / "run-manifest.json")
    v21_hashes = _json(v21.OUT_ROOT / "artifact-hashes.json")
    if v21_manifest.get("status") != "COMPLETE" or v21_hashes.get("status") != "HASHED":
        raise MechanismError("V2.1 corrected source is not COMPLETE/HASHED")
    bad = [name for name, digest in v21_manifest.get("files", {}).items()
           if not (v21.OUT_ROOT / name).is_file() or _sha(v21.OUT_ROOT / name) != digest]
    bad += [name for name, digest in v21_hashes.get("files", {}).items()
            if not (v21.OUT_ROOT / name).is_file() or _sha(v21.OUT_ROOT / name) != digest]
    if bad:
        raise MechanismError(f"V2.1 source artifact hash mismatch: {sorted(set(bad))}")
    cohort = list(csv.DictReader((frozen.OUT_ROOT / "signal-control-comparisons.csv").open(encoding="utf-8", newline="")))
    pairs = [r for r in cohort if r["estimand"] in TARGET_ESTIMANDS]
    if len(pairs) != 113 or len({r["signal_id"] for r in pairs}) != 113:
        raise MechanismError(f"frozen primary pair identity count changed: rows={len(pairs)} unique={len({r['signal_id'] for r in pairs})}")
    src = _json(SOURCE_COVERAGE)
    if src.get("dates_processed") != 54 or len(src.get("files", [])) != 54:
        raise MechanismError("native source coverage is not exactly 54 dates")
    dates = {r["date"] for r in pairs}
    if not dates.issubset({r["date"] for r in src["files"]}):
        raise MechanismError("frozen pairs refer to missing native source date")
    source_manifest, requests = baseline._manifest(INPUT_ROOT)
    if source_manifest.get("status") != "COMPLETE":
        raise MechanismError("native MBP10 acquisition manifest is not COMPLETE")
    src_by_date = {r["date"]: r for r in src["files"]}
    request_by_date = {str(r.get("session_date")): r for r in requests.values()
                       if isinstance(r, dict) and r.get("category") in {"TRAIN", "VALIDATION"}}
    if set(src_by_date) != set(request_by_date):
        raise MechanismError("source coverage and acquisition manifest date sets differ")
    verified = []
    for day, row in sorted(src_by_date.items()):
        req = request_by_date[day]
        rel = str(req.get("path", ""))
        path = INPUT_ROOT / rel
        if (req.get("status") != "VERIFIED" or req.get("schema") != "mbp-10"
                or req.get("symbol") != row["symbol"] or req.get("sha256") != row["native_source_sha256"]
                or not str(row["native_source_path"]).endswith("/" + rel) or not path.is_file()
                or path.stat().st_size != int(req.get("bytes", -1))):
            raise MechanismError(f"native source identity/status/size mismatch: {day}")
        digest = _sha(path)
        if digest != row["native_source_sha256"]:
            raise MechanismError(f"native source SHA-256 mismatch: {day}")
        verified.append({"date": day, "path": rel, "sha256_expected": row["native_source_sha256"],
                         "sha256_actual": digest, "bytes_expected": int(req["bytes"]), "schema": req["schema"], "symbol": req["symbol"],
                         "dataset": req.get("verification", {}).get("dataset"), "record_count_expected": req.get("verification", {}).get("record_count"),
                         "start": req["start"], "end": req["end"]})
    return {"continuation_artifact_hashes": hashes["files"], "v21_manifest": v21_manifest,
            "v21_hashes": v21_hashes}, src, pairs, {"acquisition_manifest_sha256": _sha(INPUT_ROOT / baseline.MANIFEST_NAME),
            "acquisition_status": source_manifest["status"], "authorized_dates": verified}


def _session_start_ns(day: str, session: str) -> int:
    start, _ = fair_v2._session_ns(day, session)
    return int(start)


def _targets(pairs: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        day, session = str(row["date"]), str(row["session"])
        signal_ns = int(str(row["signal_id"]).rsplit("|", 1)[1])
        control_ns = _session_start_ns(day, session) + (int(row["control_minute"]) + 1) * 60 * NS
        direction = 1 if row["direction"] == "LONG" else -1
        for role, timestamp in (("SIGNAL", signal_ns), ("CONTROL", control_ns)):
            result[day].append({"pair_id": f"{row['estimand']}|{row['signal_id']}", "estimand": row["estimand"],
                "role": role, "date": day, "period": row["period"], "session": session,
                "direction": row["direction"], "direction_sign": direction, "timestamp_ns": timestamp,
                "signal_id": row["signal_id"], "signal_minute": int(row["signal_minute"]),
                "control_minute": int(row["control_minute"]),
                "outcome_5m_ticks": float(row["signal_net_5m_ticks"] if role == "SIGNAL" else row["control_net_5m_ticks"]),
                "pair_outcome_delta_ticks": float(row["signal_minus_control_5m_ticks"]),
                "pair_mfe_ticks": float(row["signal_mfe_net_ticks"] if role == "SIGNAL" else row["control_mfe_net_ticks"]),
                "pair_mae_ticks": float(row["signal_mae_ticks"] if role == "SIGNAL" else row["control_mae_ticks"]),
                "favorable_before_adverse": (str(row.get("signal_favorable_before_adverse" if role == "SIGNAL" else "control_favorable_before_adverse", "")).lower() == "true") if row.get("signal_favorable_before_adverse" if role == "SIGNAL" else "control_favorable_before_adverse") is not None else None})
    for day in result:
        result[day].sort(key=lambda x: (x["timestamp_ns"], x["pair_id"], x["role"]))
    return dict(result)


def _secondary_targets(signals: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the frozen displacement-signal-only descriptive cohort."""
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected = [r for r in signals if r.get("model") == "OPENING_CONTINUATION"
                and r.get("trigger") == "DISPLACEMENT_CANDLE"]
    if len(selected) != 142 or len({str(r["signal_id"]) for r in selected}) != 142:
        raise MechanismError(f"secondary displacement cohort changed: {len(selected)} rows")
    for row in selected:
        day = str(row["date"])
        result[day].append({"pair_id": f"SECONDARY|{row['signal_id']}", "event_kind": "SECONDARY_DISPLACEMENT_SIGNAL",
            "estimand": "SECONDARY_DESCRIPTIVE", "role": "SECONDARY_SIGNAL", "date": day,
            "period": row["period"], "session": row["session"], "direction": row["direction"],
            "direction_sign": int(row["direction_sign"]), "timestamp_ns": int(row["signal_timestamp_ns"]),
            "signal_id": str(row["signal_id"])})
    for day in result:
        result[day].sort(key=lambda x: (x["timestamp_ns"], x["pair_id"]))
    return dict(result)


def _merged_intervals(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    spans = sorted((int(e["timestamp_ns"]) - 65 * NS, int(e["timestamp_ns"]) + 301 * NS) for e in events)
    merged: list[list[int]] = []
    for left, right in spans:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(x[0], x[1]) for x in merged]


def _decode_code(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values
    return np.asarray([x.encode() if isinstance(x, str) else str(x).encode() for x in values], dtype="S1")


def _stream_date(day: str, path: Path, source: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from databento import DBNStore
    from .mac_2025_es_only_train_baseline import _sha256

    # Compressed bytes were hash-verified for every authorized date before extraction.
    source_hash = str(source["native_source_sha256"])
    store = DBNStore.from_file(path)
    metadata = store.metadata
    if metadata.dataset != "GLBX.MDP3" or metadata.schema != "mbp-10" or source["symbol"] not in metadata.symbols:
        raise MechanismError(f"native DBN metadata mismatch for {day}: {metadata}")
    intervals = _merged_intervals(events)
    captured: dict[int, tuple[Any, ...]] = {}
    interval_cursor = 0
    raw_count = 0
    last_ts: int | None = None
    interval_rows = [0] * len(intervals)
    first_ts = last_ts_seen = None
    for batch in store.to_ndarray(count=250_000):
        ts = np.asarray(batch["ts_recv"], dtype=np.int64)
        if len(ts):
            if (last_ts is not None and int(ts[0]) < last_ts) or (len(ts) > 1 and bool(np.any(ts[1:] < ts[:-1]))):
                raise MechanismError(f"ts_recv order regression in {day}")
            last_ts = int(ts[-1]); raw_count += len(ts)
            first_ts = int(ts[0]) if first_ts is None else first_ts
            last_ts_seen = int(ts[-1])
        bid_px, bid_sz = mlofi._raw_book_arrays(batch, "bid")
        ask_px, ask_sz = mlofi._raw_book_arrays(batch, "ask")
        selected_parts: list[tuple[int, int]] = []
        for idx, (left, right) in enumerate(intervals):
            lo = int(np.searchsorted(ts, left, side="left"))
            hi = int(np.searchsorted(ts, right, side="left"))
            if lo < hi:
                selected_parts.append((lo, hi))
                interval_rows[idx] += hi - lo
        for lo, hi in selected_parts:
            # Keep the preceding source row for price-keyed deltas at segment entry.
            if lo > 0:
                indexes = range(lo - 1, hi)
            elif ts.size and int(ts[0]) >= intervals[max(0, interval_cursor)][0]:
                indexes = range(lo, hi)
            else:
                indexes = range(lo, hi)
            for i in indexes:
                ordinal = raw_count - len(ts) + i
                if ordinal in captured:
                    continue
                captured[ordinal] = (int(ts[i]), bid_px[i].copy(), bid_sz[i].copy(), ask_px[i].copy(), ask_sz[i].copy(),
                    bytes(batch["action"][i]), bytes(batch["side"][i]), int(batch["price"][i]), int(batch["size"][i]))
        while interval_cursor < len(intervals) and ts.size and int(ts[-1]) >= intervals[interval_cursor][1]:
            interval_cursor += 1
    expected = int(source.get("record_count_expected") or mlofi._source_record_count(day, path))
    if raw_count != expected:
        raise MechanismError(f"sealed raw-row count mismatch {day}: {raw_count} != {expected}")
    rows = [captured[k] for k in sorted(captured)]
    if not rows:
        raise MechanismError(f"no native MBP rows captured for {day}")
    ts = np.asarray([x[0] for x in rows], dtype=np.int64)
    bid_px = np.asarray([x[1] for x in rows], dtype=np.int64)
    bid_sz = np.asarray([x[2] for x in rows], dtype=np.int64)
    ask_px = np.asarray([x[3] for x in rows], dtype=np.int64)
    ask_sz = np.asarray([x[4] for x in rows], dtype=np.int64)
    action = np.asarray([x[5] for x in rows], dtype="S1")
    side = np.asarray([x[6] for x in rows], dtype="S1")
    price = np.asarray([x[7] for x in rows], dtype=np.int64)
    size = np.asarray([x[8] for x in rows], dtype=np.int64)
    bid_visible = mlofi._packed_book_or_raise(bid_px, bid_sz, "bid")
    ask_visible = mlofi._packed_book_or_raise(ask_px, ask_sz, "ask")
    executable = bid_visible[:, 0] & ask_visible[:, 0] & (ask_px[:, 0] > bid_px[:, 0])
    segment_start = np.zeros(len(rows), dtype=bool)
    segment_start[0] = True
    # A gap in selected source ordinals means the previous snapshot was not retained.
    ordinals = sorted(captured)
    segment_start[1:] = np.diff(np.asarray(ordinals, dtype=np.int64)) > 1
    prior_valid = np.zeros(len(rows), dtype=bool)
    prior_valid[1:] = executable[:-1] & ~segment_start[1:]
    contrib = mlofi._vector_price_keyed_contribution(
        bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
        previous_bid_px=np.vstack([bid_px[0], bid_px[:-1]]), previous_bid_sz=np.vstack([bid_sz[0], bid_sz[:-1]]),
        previous_ask_px=np.vstack([ask_px[0], ask_px[:-1]]), previous_ask_sz=np.vstack([ask_sz[0], ask_sz[:-1]]),
        previous_valid=prior_valid, action=action, side=side, price=price, size=size, current_valid=executable)
    event_rows = []
    path_rows = []
    weighted5 = np.asarray([1 / (i + 1) for i in range(5)], dtype=float)
    weighted10 = np.asarray([1 / (i + 1) for i in range(10)], dtype=float)

    def state_index(query: int) -> int | None:
        ix = int(np.searchsorted(ts, query, side="right")) - 1
        if ix < 0 or not executable[ix] or query - int(ts[ix]) > NS:
            return None
        return ix

    def sums_at(query: int, start: int, end: int) -> tuple[float, float, int]:
        # Current-price keyed contributions; exact same accounting as the repository MLOFI study.
        lo = int(np.searchsorted(ts, start, side="right"))
        hi = int(np.searchsorted(ts, end, side="right"))
        if hi <= lo:
            return 0.0, 0.0, 0
        vals = contrib[lo:hi, :5].astype(float) @ weighted5
        return float(vals.sum()), float(np.count_nonzero(vals)), hi - lo

    def trade_flow(start: int, end: int, sign: int) -> dict[str, float]:
        lo = int(np.searchsorted(ts, start, side="right")); hi = int(np.searchsorted(ts, end, side="right"))
        mask = (action[lo:hi] == b"T") & np.isin(side[lo:hi], (b"B", b"A")) & (size[lo:hi] > 0)
        sides = side[lo:hi][mask]; sizes = size[lo:hi][mask].astype(float)
        buys = float(sizes[sides == b"B"].sum()); sells = float(sizes[sides == b"A"].sum()); total = buys + sells
        return {"buy_contracts": buys, "sell_contracts": sells, "total_contracts": total,
                "directional_contracts": sign * (buys - sells),
                "normalized_delta_directional": sign * (buys - sells) / total if total else None,
                "trade_count": int(mask.sum())}

    def depth_at(ix: int, levels: int = 5) -> tuple[float, float]:
        n = min(levels, 10)
        wb, wa = weighted5[:n] if n <= 5 else weighted10[:n], weighted5[:n] if n <= 5 else weighted10[:n]
        return float(np.dot(bid_sz[ix, :n], wb)), float(np.dot(ask_sz[ix, :n], wa))

    by_time = sorted(events, key=lambda x: x["timestamp_ns"])
    for event in by_time:
        t, sign = int(event["timestamp_ns"]), int(event["direction_sign"])
        ix = state_index(t)
        row = dict(event)
        row.update({"book_valid_at_decision": ix is not None, "top5_imbalance_directional": None,
                    "top10_imbalance_directional": None, "spread_ticks": None,
                    "opposing_to_directional_depth_ratio": None, "opposing_depth_relative_to_causal_baseline": None,
                    "mlofi_top5_500ms_directional": None, "mlofi_top5_2s_persistence": None,
                    "aggressive_flow_5s": None, "aggressive_flow_30s": None,
                    "aggression_intensity_change_5s": None, "price_impact_ticks_per_directional_contract": None,
                    "price_impact_change": None, "resiliency_consumed_depth": None,
                    "resiliency_restored_depth": None, "resiliency_restore_to_consumption_ratio": None})
        if ix is not None:
            bd5, ad5 = depth_at(ix, 5); bd10, ad10 = depth_at(ix, 10)
            den5, den10 = bd5 + ad5, bd10 + ad10
            row["top5_imbalance_directional"] = sign * (bd5 - ad5) / den5 if den5 else None
            row["top10_imbalance_directional"] = sign * (bd10 - ad10) / den10 if den10 else None
            row["spread_ticks"] = (int(ask_px[ix, 0]) - int(bid_px[ix, 0])) / TICK_RAW
            directional_depth, opposing_depth = (bd5, ad5) if sign > 0 else (ad5, bd5)
            row["opposing_to_directional_depth_ratio"] = opposing_depth / directional_depth if directional_depth > 0 else None
            baseline_samples = []
            for q in range(t - 60 * NS, t - 5 * NS + 1, 100_000_000):
                j = state_index(q)
                if j is None:
                    continue
                b, a = depth_at(j, 5)
                own, opp = (b, a) if sign > 0 else (a, b)
                if own > 0:
                    baseline_samples.append(opp / own)
            if baseline_samples and row["opposing_to_directional_depth_ratio"] is not None:
                row["opposing_depth_relative_to_causal_baseline"] = row["opposing_to_directional_depth_ratio"] / float(np.median(baseline_samples)) if np.median(baseline_samples) else None
            total500, _, _ = sums_at(t, t - 500_000_000, t)
            # Normalize with the contemporaneous inverse-rank mean TOP5 resting depth.
            mean_depth = float(np.dot(bid_sz[ix, :5] + ask_sz[ix, :5], weighted5) / (2 * weighted5.sum()))
            row["mlofi_top5_500ms_directional"] = sign * total500 / mean_depth if mean_depth > 0 else None
            bins = []
            for b in range(8):
                end = t - b * 250_000_000
                start = end - 250_000_000
                value, _, _ = sums_at(end, start, end)
                bins.append(value)
            nonzero = [float(v) for v in bins if v != 0]
            if nonzero:
                latest_sign = math.copysign(1, nonzero[0])
                row["mlofi_top5_2s_persistence"] = sum(math.copysign(1, v) == latest_sign for v in nonzero) / len(nonzero)
                row["mlofi_top5_2s_latest_sign_directional"] = sign * latest_sign
            else:
                row["mlofi_top5_2s_persistence"] = None
                row["mlofi_top5_2s_latest_sign_directional"] = None
            flow5 = trade_flow(t - 5 * NS, t, sign); flow30 = trade_flow(t - 30 * NS, t, sign)
            row["aggressive_flow_5s"], row["aggressive_flow_30s"] = flow5, flow30
            prev5 = trade_flow(t - 10 * NS, t - 5 * NS, sign)
            row["aggression_intensity_change_5s"] = flow5["directional_contracts"] - prev5["directional_contracts"]
            m0 = (int(bid_px[ix, 0]) + int(ask_px[ix, 0])) / 2 / 1e9
            i5 = state_index(t - 5 * NS); i10 = state_index(t - 10 * NS)
            if i5 is not None:
                m5 = (int(bid_px[i5, 0]) + int(ask_px[i5, 0])) / 2 / 1e9
                directional_move = sign * (m0 - m5) / TICK
                volume = flow5["total_contracts"]
                row["price_impact_ticks_per_directional_contract"] = directional_move / flow5["directional_contracts"] if volume > 1 and abs(flow5["directional_contracts"]) > 1 else None
                if i10 is not None:
                    m10 = (int(bid_px[i10, 0]) + int(ask_px[i10, 0])) / 2 / 1e9
                    old_move = sign * (m5 - m10) / TICK
                    old_flow = trade_flow(t - 10 * NS, t - 5 * NS, sign)["directional_contracts"]
                    row["price_impact_change"] = row["price_impact_ticks_per_directional_contract"] - (old_move / old_flow if abs(old_flow) > 1 else 0.0) if row["price_impact_ticks_per_directional_contract"] is not None else None
            # Fixed completed pre-signal five-second aggregate-depth depletion/recovery proxy.
            start_ix = state_index(t - 5 * NS)
            if start_ix is not None:
                depths = []
                lo = int(np.searchsorted(ts, t - 5 * NS, side="left")); hi = int(np.searchsorted(ts, t, side="right"))
                for j in range(lo, hi):
                    if executable[j]:
                        b, a = depth_at(j, 5); depths.append(a if sign > 0 else b)
                if depths:
                    consumed = max(0.0, depths[0] - min(depths))
                    restored = max(0.0, depths[-1] - min(depths))
                    row["resiliency_consumed_depth"] = consumed
                    row["resiliency_restored_depth"] = restored
                    row["resiliency_restore_to_consumption_ratio"] = restored / consumed if consumed > 0 else None
                    row["resiliency_observation_supported"] = True
                else:
                    row["resiliency_observation_supported"] = False
            else:
                row["resiliency_observation_supported"] = False
        else:
            row["mlofi_top5_2s_latest_sign_directional"] = None
            row["resiliency_observation_supported"] = False
        event_rows.append(row)
        for start_s, end_s in POST_WINDOWS:
            lo_t, hi_t = t + start_s * NS, t + end_s * NS
            left_ix, right_ix = state_index(lo_t), state_index(hi_t)
            seg = {"pair_id": event["pair_id"], "role": event["role"], "date": day,
                   "period": event["period"], "session": event["session"], "direction": event["direction"],
                   "window_start_seconds": start_s, "window_end_seconds": end_s,
                   "complete": right_ix is not None and hi_t <= _session_end_ns(day, str(event["session"]))}
            if left_ix is not None and right_ix is not None and seg["complete"]:
                m0 = (int(bid_px[left_ix, 0]) + int(ask_px[left_ix, 0])) / 2 / 1e9
                m1 = (int(bid_px[right_ix, 0]) + int(ask_px[right_ix, 0])) / 2 / 1e9
                seg["directional_mid_change_ticks"] = sign * (m1 - m0) / TICK
                b0, a0 = depth_at(left_ix, 5); b1, a1 = depth_at(right_ix, 5)
                own0, opp0 = (b0, a0) if sign > 0 else (a0, b0)
                own1, opp1 = (b1, a1) if sign > 0 else (a1, b1)
                seg["directional_depth_change"] = own1 - own0
                seg["opposing_depth_change"] = opp1 - opp0
                seg["opposing_depth_ratio_at_end"] = opp1 / own1 if own1 > 0 else None
                f = trade_flow(lo_t, hi_t, sign)
                seg["directional_aggressive_contracts"] = f["directional_contracts"]
                seg["aggressive_contracts"] = f["total_contracts"]
                ofi, _, _ = sums_at(hi_t, lo_t, hi_t)
                seg["directional_top5_price_keyed_ofi"] = sign * ofi
            else:
                seg.update({"directional_mid_change_ticks": None, "directional_depth_change": None,
                            "opposing_depth_change": None, "opposing_depth_ratio_at_end": None,
                            "directional_aggressive_contracts": None, "aggressive_contracts": None,
                            "directional_top5_price_keyed_ofi": None})
            path_rows.append(seg)
    audit = {"date": day, "native_records": raw_count, "captured_event_window_records": len(rows),
             "captured_first_ts_recv_ns": int(ts[0]), "captured_last_ts_recv_ns": int(ts[-1]),
             "source_first_ts_recv_ns": first_ts, "source_last_ts_recv_ns": last_ts_seen,
             "event_window_interval_count": len(intervals), "event_window_rows_by_interval": interval_rows,
             "book_valid_event_observations": sum(bool(r["book_valid_at_decision"]) for r in event_rows),
             "mlofi_500ms_supported": sum(r["mlofi_top5_500ms_directional"] is not None for r in event_rows),
             "resiliency_supported": sum(bool(r.get("resiliency_observation_supported")) for r in event_rows)}
    return event_rows, path_rows, audit


def _session_end_ns(day: str, session: str) -> int:
    return int(fair_v2._session_ns(day, session)[1])


def _stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "sd": None, "q10": None, "q90": None}
    return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)), "sd": float(np.std(a, ddof=1)) if len(a)>1 else 0.0,
            "q10": float(np.quantile(a, .1)), "q90": float(np.quantile(a, .9))}


def _date_bootstrap(values_by_date: Mapping[str, Sequence[float]], seed: int, reps: int = 2000) -> dict[str, Any]:
    days = sorted(values_by_date)
    vals = [float(x) for d in days for x in values_by_date[d]]
    if not vals:
        return {"n": 0, "date_clusters": len(days), "mean": None, "ci95": None}
    mean = float(np.mean(vals))
    if len(days) < 2:
        return {"n": len(vals), "date_clusters": len(days), "mean": mean, "ci95": None}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        chosen = rng.choice(days, size=len(days), replace=True)
        sample = [x for d in chosen for x in values_by_date[str(d)]]
        if sample:
            boot.append(float(np.mean(sample)))
    return {"n": len(vals), "date_clusters": len(days), "mean": mean,
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "method": "date-cluster percentile bootstrap; all same-day observations travel together"}


def _corr(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return {"n": len(a), "pearson": None, "spearman": None}
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values), dtype=float)
        left = 0
        while left < len(values):
            right = left + 1
            while right < len(values) and values[order[right]] == values[order[left]]:
                right += 1
            out[order[left:right]] = (left + right - 1) / 2
            left = right
        return out
    ranks_a, ranks_b = ranks(a), ranks(b)
    return {"n": len(a), "pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(ranks_a, ranks_b)[0, 1])}


def _cluster_corr_ci(records: Sequence[Mapping[str, Any]], x_name: str, y_name: str, seed: int) -> list[float] | None:
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_day[str(row["date"])].append(row)
    days = sorted(by_day)
    if len(days) < 4:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(1500):
        chosen = rng.choice(days, size=len(days), replace=True)
        sample = [row for day in chosen for row in by_day[str(day)]]
        corr = _corr([float(r[x_name]) for r in sample], [float(r[y_name]) for r in sample])["pearson"]
        if corr is not None and math.isfinite(float(corr)):
            vals.append(float(corr))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else None


def _analyze(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["role"])] = row
    features = ("top5_imbalance_directional", "top10_imbalance_directional", "spread_ticks",
                "opposing_to_directional_depth_ratio", "opposing_depth_relative_to_causal_baseline",
                "mlofi_top5_500ms_directional", "mlofi_top5_2s_persistence", "mlofi_top5_2s_latest_sign_directional", "aggressive_flow_5s",
                "aggressive_flow_30s", "aggressive_delta_5s_normalized", "aggressive_delta_30s_normalized",
                "aggression_intensity_change_5s", "price_impact_ticks_per_directional_contract",
                "price_impact_change", "resiliency_consumed_depth", "resiliency_restored_depth",
                "resiliency_restore_to_consumption_ratio")
    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair_id, roles in by_pair.items():
        if "SIGNAL" not in roles or "CONTROL" not in roles:
            continue
        s, c = roles["SIGNAL"], roles["CONTROL"]
        for feature in features:
            sv, cv = s.get(feature), c.get(feature)
            if feature.startswith("aggressive_flow_"):
                sv = (sv or {}).get("directional_contracts")
                cv = (cv or {}).get("directional_contracts")
            elif feature.startswith("aggressive_delta_"):
                flow_key = "aggressive_flow_" + feature.removeprefix("aggressive_delta_").removesuffix("_normalized")
                sv = (s.get(flow_key) or {}).get("normalized_delta_directional")
                cv = (c.get(flow_key) or {}).get("normalized_delta_directional")
            if sv is not None and cv is not None and math.isfinite(float(sv)) and math.isfinite(float(cv)):
                paired[feature].append({"pair_id": pair_id, "date": s["date"], "period": s["period"],
                    "session": s["session"], "direction": s["direction"], "signal": float(sv), "control": float(cv),
                    "difference": float(sv)-float(cv), "signal_outcome": float(s["outcome_5m_ticks"]),
                    "pair_outcome_delta": float(s["pair_outcome_delta_ticks"]),
                    "signal_mfe": float(s["pair_mfe_ticks"]), "signal_mae": float(s["pair_mae_ticks"]),
                    "pair_mfe_delta": float(s["pair_mfe_ticks"])-float(c["pair_mfe_ticks"]),
                    "pair_mae_delta": float(s["pair_mae_ticks"])-float(c["pair_mae_ticks"])})
    output = {"features": {}, "periods": {}, "subgroups": {}, "multiple_comparisons": {
        "feature_relationships_reported": len(features), "all_estimates_exploratory": True, "no_threshold_selection": True}}
    for feature in features:
        records = paired[feature]
        output["features"][feature] = {"complete_pairs": len(records),
            "signal_distribution": _stats([r["signal"] for r in records]),
            "control_distribution": _stats([r["control"] for r in records]),
            "paired_difference_signal_minus_control": _date_bootstrap(_group(records, "difference"), 12017+len(feature)),
            "signal_feature_vs_signal_net_5m": _corr([r["signal"] for r in records], [r["signal_outcome"] for r in records]),
            "signal_feature_vs_signal_mfe": _corr([r["signal"] for r in records], [r["signal_mfe"] for r in records]),
            "signal_feature_vs_signal_mae": _corr([r["signal"] for r in records], [r["signal_mae"] for r in records]),
            "paired_feature_difference_vs_paired_mfe_difference": _corr([r["difference"] for r in records], [r["pair_mfe_delta"] for r in records]),
            "paired_feature_difference_vs_paired_mae_difference": _corr([r["difference"] for r in records], [r["pair_mae_delta"] for r in records]),
            "signal_feature_net_markout_date_cluster_ci95": _cluster_corr_ci(
                [{"date": r["date"], "x": r["signal"], "y": r["signal_outcome"]} for r in records], "x", "y", 29011+len(feature)),
            "paired_feature_difference_vs_paired_outcome_difference": _corr([r["difference"] for r in records], [r["pair_outcome_delta"] for r in records]),
            "paired_feature_outcome_correlation_date_cluster_ci95": _cluster_corr_ci(
                [{"date": r["date"], "x": r["difference"], "y": r["pair_outcome_delta"]} for r in records], "x", "y", 30011+len(feature))}
        output["periods"][feature] = {}
        for period in ("SPRING_2025", "OCTOBER_2025"):
            subset = [r for r in records if r["period"] == period]
            output["periods"][feature][period] = {"complete_pairs": len(subset),
                "paired_difference": _date_bootstrap(_group(subset, "difference"), 22031+len(feature)+len(period)),
                "signal_feature_vs_signal_net_5m": _corr([r["signal"] for r in subset], [r["signal_outcome"] for r in subset]),
                "signal_feature_net_markout_date_cluster_ci95": _cluster_corr_ci(
                    [{"date": r["date"], "x": r["signal"], "y": r["signal_outcome"]} for r in subset], "x", "y", 41017+len(feature)+len(period)),
                "paired_feature_difference_vs_markout_difference": _corr([r["difference"] for r in subset], [r["pair_outcome_delta"] for r in subset]),
                "paired_feature_outcome_correlation_date_cluster_ci95": _cluster_corr_ci(
                    [{"date": r["date"], "x": r["difference"], "y": r["pair_outcome_delta"]} for r in subset], "x", "y", 42017+len(feature)+len(period))}
        output["subgroups"][feature] = {}
        for key in ("direction", "session"):
            output["subgroups"][feature][key] = {}
            for value in sorted({str(r[key]) for r in records}):
                subset = [r for r in records if str(r[key]) == value]
                output["subgroups"][feature][key][value] = {"n": len(subset), "date_clusters": len({r["date"] for r in subset}),
                    "mean_signal_minus_control": float(np.mean([r["difference"] for r in subset])) if subset else None}
    return output


def _analyze_postsignal(paths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare post-event paths by frozen pair; these are outcomes, not predictors."""
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in paths:
        if row.get("role") not in {"SIGNAL", "CONTROL"}:
            continue
        by_key[(str(row["pair_id"]), int(row["window_end_seconds"]))][str(row["role"])] = row
    fields = ("directional_mid_change_ticks", "opposing_depth_change", "directional_aggressive_contracts",
              "directional_top5_price_keyed_ofi")
    output: dict[str, Any] = {"semantics": "post-signal explanatory outcomes; never entry-time predictors",
                              "windows": {}}
    for end_s in (5, 30, 120, 300):
        paired = []
        for (pair_id, window_end), roles in by_key.items():
            if window_end != end_s or not {"SIGNAL", "CONTROL"}.issubset(roles):
                continue
            s, c = roles["SIGNAL"], roles["CONTROL"]
            rec = {"pair_id": pair_id, "date": s["date"], "period": s["period"]}
            for field in fields:
                sv, cv = s.get(field), c.get(field)
                if sv is not None and cv is not None:
                    rec[field] = float(sv) - float(cv)
            paired.append(rec)
        per_period: dict[str, Any] = {}
        for period in ("ALL", "SPRING_2025", "OCTOBER_2025"):
            subset = [r for r in paired if period == "ALL" or r["period"] == period]
            per_period[period] = {field: _date_bootstrap(_group(subset, field), 57001 + end_s + len(field))
                                  for field in fields if all(field in r for r in subset)}
        output["windows"][f"{end_s}s"] = {"paired_signal_minus_control": per_period,
            "complete_matched_paths": sum(bool(by_key.get((pid, end_s), {}).get("SIGNAL", {}).get("complete"))
                                           and bool(by_key.get((pid, end_s), {}).get("CONTROL", {}).get("complete"))
                                           for pid, w in by_key if w == end_s)}
    return output


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        result[str(row["date"])].append(float(row[field]))
    return result


def _economics(pairs: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_date[str(pair["date"])].append(pair)
    source_by_date = {r["date"]: r for r in source["files"]}
    event_economics = []
    tape_hash_checks = []
    for day, rows in sorted(by_date.items()):
        source_row = source_by_date[day]
        tape_path = Path(source_row["candidate_tape_path"])
        if _sha(tape_path) != source_row["candidate_tape_sha256"]:
            raise MechanismError(f"Candidate Tape SHA mismatch for economics: {day}")
        tape = load_tape(tape_path, source_sha256=source_row["candidate_tape_source_sha256"],
                         semantic_sha256=source_row["candidate_tape_semantic_sha256"])
        tape_hash_checks.append({"date": day, "sha256": source_row["candidate_tape_sha256"], "verified": True})
        for pair in rows:
            signal_ns = int(str(pair["signal_id"]).rsplit("|", 1)[1])
            control_ns = _session_start_ns(day, str(pair["session"])) + (int(pair["control_minute"]) + 1) * 60 * NS
            for role, decision in (("SIGNAL", signal_ns), ("CONTROL", control_ns)):
                end_ns = _session_end_ns(day, str(pair["session"]))
                path = fair_diag._quote_path(tape, decision, 1 if pair["direction"] == "LONG" else -1, end_ns, 0.0)
                if not path.get("entry_available"):
                    event_economics.append({"pair_id": f"{pair['estimand']}|{pair['signal_id']}", "role": role,
                                            "date": day, "period": pair["period"], "entry_available": False})
                    continue
                direction = 1 if pair["direction"] == "LONG" else -1
                ev = tape.events; ts = ev["timestamp_ns"].astype(np.int64, copy=False)
                bid = ev["bid"].astype(float, copy=False); ask = ev["ask"].astype(float, copy=False)
                qts = int(path["entry_timestamp_ns"]); horizon_ns = qts + 300 * NS
                valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid) & (ts < end_ns)
                qix = np.flatnonzero(valid & (ts <= horizon_ns) & (ts >= qts))
                if not len(qix):
                    raise MechanismError(f"missing executable horizon quote {day} {role}")
                exit_i = int(qix[-1]); entry_mid = float(path["entry_mid"])
                exit_mid = (float(bid[exit_i]) + float(ask[exit_i])) / 2
                mid_ticks = direction * (exit_mid - entry_mid) / TICK
                entry_spread = (float(path["entry_ask"]) - float(path["entry_bid"])) / TICK
                exit_spread = (float(ask[exit_i]) - float(bid[exit_i])) / TICK
                entry_impact = entry_spread / 2 + 1.0
                exit_impact = exit_spread / 2 + 1.0
                fees = 6.0 / 12.5
                net = mid_ticks - entry_spread / 2 - exit_spread / 2 - 2.0 - fees
                raw_trade_mask = (ts >= qts) & (ts <= int(ts[exit_i])) & (ev["execution_size"] > 0)
                trade_ix = np.flatnonzero(raw_trade_mask)
                raw_move = None
                if len(trade_ix):
                    raw_last = float(ev["execution_price"][trade_ix[-1]])
                    raw_move = direction * (raw_last - entry_mid) / TICK
                if abs(net - float(path["horizons"]["300"]["executable_net_ticks"])) > 1e-7:
                    raise MechanismError(f"frozen markout arithmetic failed to reconcile {day} {role}: {net}")
                event_economics.append({"pair_id": f"{pair['estimand']}|{pair['signal_id']}", "role": role,
                    "date": day, "period": pair["period"], "session": pair["session"], "direction": pair["direction"],
                    "entry_timestamp_ns": qts, "exit_timestamp_ns": int(ts[exit_i]), "entry_spread_ticks": entry_spread,
                    "exit_spread_ticks": exit_spread, "entry_impact_ticks_halfspread_plus_adverse_tick": entry_impact,
                    "exit_impact_ticks_halfspread_plus_adverse_tick": exit_impact, "fees_ticks": fees,
                    "directional_mid_markout_ticks": mid_ticks, "last_trade_directional_move_from_entry_mid_ticks": raw_move,
                    "executable_net_markout_ticks": net, "modeled_total_execution_cost_ticks": entry_impact+exit_impact+fees,
                    "mfe_executable_ticks": path.get("mfe_executable_net_ticks"), "mae_executable_ticks": path.get("mae_executable_ticks"),
                    "decomposition_residual_ticks": net-(mid_ticks-entry_spread/2-exit_spread/2-2-fees)})
        del tape
    event_by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in event_economics:
        event_by_pair[str(row["pair_id"])][str(row["role"])] = row
    paired_rows = []
    for pair_id, roles in event_by_pair.items():
        if "SIGNAL" not in roles or "CONTROL" not in roles or not roles["SIGNAL"].get("entry_available", True) or not roles["CONTROL"].get("entry_available", True):
            continue
        s, c = roles["SIGNAL"], roles["CONTROL"]
        paired_rows.append({"pair_id": pair_id, "date": s["date"], "period": s["period"],
            "signal_minus_control_raw_trade_move_ticks": (s.get("last_trade_directional_move_from_entry_mid_ticks") or 0)-(c.get("last_trade_directional_move_from_entry_mid_ticks") or 0),
            "signal_minus_control_mid_move_ticks": s["directional_mid_markout_ticks"]-c["directional_mid_markout_ticks"],
            "signal_minus_control_execution_cost_ticks": s["modeled_total_execution_cost_ticks"]-c["modeled_total_execution_cost_ticks"],
            "signal_minus_control_net_ticks": s["executable_net_markout_ticks"]-c["executable_net_markout_ticks"],
            "decomposition_residual_ticks": (s["decomposition_residual_ticks"]+c["decomposition_residual_ticks"]),
            "frozen_original_signal_minus_control_ticks": next(float(p["signal_minus_control_5m_ticks"]) for p in pairs if f"{p['estimand']}|{p['signal_id']}" == pair_id)})
    def period_sum(field: str, period: str) -> dict[str, Any]:
        subset = [r for r in paired_rows if period == "ALL" or r["period"] == period]
        grouped = _group(subset, field)
        return {"pairs": len(subset), "date_clusters": len(grouped), "mean_signal_minus_control": float(np.mean([r[field] for r in subset])) if subset else None,
                "date_cluster_ci95": _date_bootstrap(grouped, 18473+len(field)+len(period))["ci95"]}
    recon = {"candidate_tapes_verified": tape_hash_checks, "complete_matched_economics": len(paired_rows),
        "periods": {period: {field: period_sum(field, period) for field in (
            "signal_minus_control_raw_trade_move_ticks", "signal_minus_control_mid_move_ticks",
            "signal_minus_control_execution_cost_ticks", "signal_minus_control_net_ticks")}
            for period in ("ALL", "SPRING_2025", "OCTOBER_2025")},
        "arithmetic_identity": "net directional mid movement - entry half-spread - exit half-spread - two adverse ticks - 0.48 tick fee",
        "fee_units": "0.48 ES ticks = $6 round-trip / $12.50 per ES tick per contract",
        "reconciles_frozen_pair_outcome": all(abs(r["signal_minus_control_net_ticks"]-r["frozen_original_signal_minus_control_ticks"]) <= 1e-6 for r in paired_rows),
        "pair_results": paired_rows}
    return event_economics, recon


def run(output: Path = OUT, *, force: bool = False) -> dict[str, Any]:
    resume_extraction = False
    if output.exists() and any(output.iterdir()) and not force:
        config_path = output / "study-config.json"
        features_path = output / "event-depth-features.jsonl.gz"
        paths_path = output / "postsignal-book-paths.jsonl.gz"
        if config_path.is_file() and features_path.is_file() and paths_path.is_file():
            prior = _json(config_path)
            contract = _input_contract()
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            prior_contract = prior.get("contract", {})
            # This additive protocol note only declares a separate secondary
            # cohort; it does not change the already-persisted primary extract.
            compatible_prior = dict(prior_contract)
            compatible_current = dict(contract)
            compatible_prior.pop("secondary_signal_rule", None)
            compatible_current.pop("secondary_signal_rule", None)
            prior_digest = hashlib.sha256(json.dumps(prior_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            resume_extraction = (prior.get("contract_sha256") == digest or
                (prior.get("contract_sha256") == prior_digest and compatible_prior == compatible_current))
        if not resume_extraction:
            raise MechanismError(f"refusing to overwrite non-empty output without matching extraction artifacts: {output}")
    started = time.perf_counter()
    # Freeze and persist protocol before native MBP10 event data/features/outcomes are analyzed.
    output.mkdir(parents=True, exist_ok=True)
    contract = _input_contract()
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_json(output / "study-config.json", {"contract": contract, "contract_sha256": contract_hash,
        "authorized_periods": {"SPRING_2025": "2025-03-03..2025-04-21 selected weekdays", "OCTOBER_2025": "2025-10-07..2025-10-31 weekdays"},
        "no_2026": True, "no_final_oos": True, "no_download": True})
    frozen_refs, source_coverage, pairs, source_audit = _verify_inputs()
    source_rows = source_coverage["files"]
    pair_dates = {str(p["date"]) for p in pairs}
    date_events = _targets(pairs)
    source_by_date = {r["date"]: r for r in source_rows}
    verified_dates: list[dict[str, Any]] = []
    if resume_extraction:
        all_event_features = _read_jsonl_gz(output / "event-depth-features.jsonl.gz")
        all_paths = _read_jsonl_gz(output / "postsignal-book-paths.jsonl.gz")
        expected_ids = {(e["pair_id"], e["role"]) for day in date_events.values() for e in day}
        actual_ids = {(str(e["pair_id"]), str(e["role"])) for e in all_event_features}
        if actual_ids != expected_ids or len(all_event_features) != len(expected_ids) or len(all_paths) != len(expected_ids)*len(POST_WINDOWS):
            raise MechanismError("persisted compact extraction does not exactly match frozen cohort identities")
        for day in sorted(pair_dates):
            source_row = source_by_date[day]
            dayrows = [r for r in all_event_features if r["date"] == day]
            verified_dates.append({"date": day, "source_path": source_row["native_source_path"],
                "source_sha256": source_row["native_source_sha256"], "source_bytes": source_row["native_source_bytes"],
                "symbol": source_row["symbol"], "schema": source_row["schema"], "audit": {
                    "date": day, "record_count_verified_during_extraction": mlofi._source_record_count(day, Path(source_row["native_source_path"])),
                    "book_valid_event_observations": sum(bool(x["book_valid_at_decision"]) for x in dayrows),
                    "mlofi_500ms_supported": sum(x.get("mlofi_top5_500ms_directional") is not None for x in dayrows),
                    "resiliency_supported": sum(bool(x.get("resiliency_observation_supported")) for x in dayrows),
                    "resumed_from_persisted_compact_extraction": True}})
        print(f"NATIVE_MBP10_EXTRACTION_RESUMED rows={len(all_event_features)} paths={len(all_paths)}", flush=True)
    else:
        all_event_features: list[dict[str, Any]] = []
        all_paths: list[dict[str, Any]] = []
        for index, day in enumerate(sorted(pair_dates), 1):
            row = source_by_date[day]
            path = Path(row["native_source_path"])
            print(f"NATIVE_MBP10_DATE_START={index}/{len(pair_dates)} {day}", flush=True)
            features, paths, audit = _stream_date(day, path, row, date_events[day])
            all_event_features.extend(features); all_paths.extend(paths)
            verified_dates.append({"date": day, "source_path": row["native_source_path"],
                "source_sha256": row["native_source_sha256"], "source_bytes": row["native_source_bytes"],
                "symbol": row["symbol"], "schema": row["schema"], "audit": audit})
            print(f"NATIVE_MBP10_DATE_COMPLETE={day} records={audit['native_records']} valid={audit['book_valid_event_observations']}/{len(features)}", flush=True)
        _write_jsonl_gz(output / "event-depth-features.jsonl.gz", all_event_features)
        _write_jsonl_gz(output / "postsignal-book-paths.jsonl.gz", all_paths)
    # The 142-signal secondary population is extracted separately so it cannot
    # alter primary matched-pair membership or paired estimates.
    secondary_path = output / "secondary-displacement-signal-features.jsonl.gz"
    all_signal_rows = [json.loads(line) for line in gzip.open(v21.OUT_ROOT / "all-signals.jsonl.gz", "rt", encoding="utf-8")]
    secondary_by_date = _secondary_targets(all_signal_rows)
    expected_secondary = {(e["pair_id"], e["role"]) for day in secondary_by_date.values() for e in day}
    secondary_rows = _read_jsonl_gz(secondary_path) if secondary_path.is_file() else []
    actual_secondary = {(str(r.get("pair_id")), str(r.get("role"))) for r in secondary_rows}
    if actual_secondary != expected_secondary or len(secondary_rows) != len(expected_secondary):
        secondary_rows = []
        for index, day in enumerate(sorted(secondary_by_date), 1):
            source_row = source_by_date[day]
            print(f"SECONDARY_NATIVE_MBP10_DATE_START={index}/{len(secondary_by_date)} {day}", flush=True)
            rows, _, audit = _stream_date(day, Path(source_row["native_source_path"]), source_row, secondary_by_date[day])
            secondary_rows.extend(rows)
            print(f"SECONDARY_NATIVE_MBP10_DATE_COMPLETE={day} valid={audit['book_valid_event_observations']}/{len(rows)}", flush=True)
        _write_jsonl_gz(secondary_path, secondary_rows)
    secondary_valid = sum(bool(r["book_valid_at_decision"]) for r in secondary_rows)
    secondary_coverage = {"cohort": "OPENING_CONTINUATION|DISPLACEMENT_CANDLE", "expected_signals": 142,
        "features_extracted": len(secondary_rows), "valid_books": secondary_valid,
        "invalid_books": len(secondary_rows) - secondary_valid, "dates": len({r["date"] for r in secondary_rows}),
        "period_counts": {period: sum(r["period"] == period for r in secondary_rows) for period in ("SPRING_2025", "OCTOBER_2025")},
        "selection_rule": "frozen signal identities only; separate descriptive file; no rematching"}
    economics, econ_summary = _economics(pairs, source_coverage)
    _write_jsonl_gz(output / "execution-event-economics.jsonl.gz", economics)
    feature_analysis = _analyze(all_event_features, pairs)
    postsignal_analysis = _analyze_postsignal(all_paths)
    _write_json(output / "execution-markout-reconciliation.json", econ_summary)
    _write_json(output / "matched-depth-comparison.json", feature_analysis)
    feature_fields = ("top5_imbalance_directional", "top10_imbalance_directional", "mlofi_top5_500ms_directional",
                      "mlofi_top5_2s_persistence", "mlofi_top5_2s_latest_sign_directional", "aggressive_flow_5s", "aggressive_flow_30s",
                      "price_impact_ticks_per_directional_contract", "resiliency_consumed_depth", "resiliency_restored_depth",
                      "resiliency_restore_to_consumption_ratio")
    def feature_period(name: str, periods: Sequence[str]) -> dict[str, Any]:
        out = {}
        for period in periods:
            selected = [r for r in all_event_features if r["period"] == period]
            by_role = {role: [r for r in selected if r["role"] == role] for role in ("SIGNAL", "CONTROL")}
            out[period] = {"events": len(selected), "feature_coverage": {
                field: {role: {"supported": sum(r.get(field) is not None for r in by_role[role]), "total": len(by_role[role])}
                        for role in by_role} for field in feature_fields}}
        return out
    periods = feature_period("all", ("SPRING_2025", "OCTOBER_2025"))
    analysis = {"contract_sha256": contract_hash, "primary_pairs_expected": 113,
        "primary_pairs_with_both_event_rows": len({r["pair_id"] for r in all_event_features if r["role"] == "SIGNAL" and any(c["pair_id"] == r["pair_id"] and c["role"] == "CONTROL" for c in all_event_features)}),
        "valid_source_dates": len(verified_dates), "distinct_independent_days": len({r["date"] for r in all_event_features}),
        "valid_book_observations": sum(bool(r["book_valid_at_decision"]) for r in all_event_features),
        "invalid_book_observations": sum(not bool(r["book_valid_at_decision"]) for r in all_event_features),
        "supported_mlofi_observations": sum(r.get("mlofi_top5_500ms_directional") is not None for r in all_event_features),
        "supported_resiliency_observations": sum(bool(r.get("resiliency_observation_supported")) for r in all_event_features),
        "period_feature_coverage": periods, "feature_relationships": feature_analysis,
        "post_signal_semantics": "all postsignal records are explanatory paths, not entry-time predictors",
        "coverage_bias": "reported event-level book validity by signal/control and date; no pair is replaced or dropped from frozen identity set",
        "max_absolute_smd_in_original_match": 0.417,
        "native_book_coverage_by_role": {role: {"valid": sum(r["role"] == role and bool(r["book_valid_at_decision"]) for r in all_event_features),
            "total": sum(r["role"] == role for r in all_event_features)} for role in ("SIGNAL", "CONTROL")},
        "feature_missingness": {feature: {"supported": sum(r.get(feature) is not None for r in all_event_features),
            "missing": sum(r.get(feature) is None for r in all_event_features), "total": len(all_event_features)}
            for feature in ("top5_imbalance_directional", "top10_imbalance_directional", "mlofi_top5_500ms_directional",
                "mlofi_top5_2s_persistence", "mlofi_top5_2s_latest_sign_directional", "aggressive_flow_5s", "aggressive_flow_30s",
                "price_impact_ticks_per_directional_contract", "resiliency_consumed_depth", "resiliency_restored_depth",
                "resiliency_restore_to_consumption_ratio")},
        "adverse_excursion_timing": "Exact event-time-to-MAE was not included in frozen cohort identities; the pre-existing favorable-before-adverse binary and MFE/MAE magnitudes are analyzed, but timing is not inferred.",
        "statistical_limitations": ["date is the independent cluster", "feature family comparisons are exploratory and multiplicity is not corrected into confirmatory claims", "Spring and October are previously explored development periods, not validation", "matched associations do not identify causal effects", "MBP-10 does not expose queue identity, exact cancellations, iceberg identity, or spoofing"]}
    _write_json(output / "spring-october-comparison.json", {"periods": periods, "economics": econ_summary["periods"],
        "feature_relationships_by_period": feature_analysis["periods"], "postsignal_paired_paths": postsignal_analysis,
        "interpretation": "both periods are previously explored development data; comparison is descriptive only"})
    import csv as csv_module
    with (output / "daily-results.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["date", "period", "pair_count", "signal_events", "control_events", "valid_books", "invalid_books", "mlofi_supported", "resiliency_supported", "mean_frozen_net_markout_delta_ticks", "mean_top5_imbalance_delta"]
        writer = csv_module.DictWriter(f, fieldnames=fields); writer.writeheader()
        for day in sorted(pair_dates):
            dayrows = [r for r in all_event_features if r["date"] == day]
            day_pairs = [r for r in econ_summary["pair_results"] if r["date"] == day]
            by_pair_roles: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
            for feature_row in dayrows:
                by_pair_roles[str(feature_row["pair_id"])][str(feature_row["role"])] = feature_row
            top5_diffs = [float(roles["SIGNAL"]["top5_imbalance_directional"]) - float(roles["CONTROL"]["top5_imbalance_directional"])
                          for roles in by_pair_roles.values() if {"SIGNAL", "CONTROL"}.issubset(roles)
                          and roles["SIGNAL"].get("top5_imbalance_directional") is not None
                          and roles["CONTROL"].get("top5_imbalance_directional") is not None]
            writer.writerow({"date": day, "period": dayrows[0]["period"], "pair_count": len({r["pair_id"] for r in dayrows}),
                "signal_events": sum(r["role"] == "SIGNAL" for r in dayrows), "control_events": sum(r["role"] == "CONTROL" for r in dayrows),
                "valid_books": sum(bool(r["book_valid_at_decision"]) for r in dayrows), "invalid_books": sum(not bool(r["book_valid_at_decision"]) for r in dayrows),
                "mlofi_supported": sum(r.get("mlofi_top5_500ms_directional") is not None for r in dayrows),
                "resiliency_supported": sum(bool(r.get("resiliency_observation_supported")) for r in dayrows),
                "mean_frozen_net_markout_delta_ticks": float(np.mean([r["signal_minus_control_net_ticks"] for r in day_pairs])) if day_pairs else "",
                "mean_top5_imbalance_delta": float(np.mean(top5_diffs)) if top5_diffs else ""})
    decisions = _scientific_decision(all_event_features, feature_analysis, econ_summary)
    analysis["secondary_displacement_signal_coverage"] = secondary_coverage
    analysis["postsignal_paired_paths"] = postsignal_analysis
    analysis["decision_diagnostics"] = decisions["hypothesis_diagnostics"]
    for filename, hypothesis, diagnostic_key, fields in (
        ("exhaustion-analysis.json", "H1_AGGRESSION_EXHAUSTION", "H1", ["aggressive_flow_5s", "aggressive_flow_30s", "aggressive_delta_5s_normalized", "aggressive_delta_30s_normalized", "aggression_intensity_change_5s", "price_impact_ticks_per_directional_contract", "price_impact_change"]),
        ("liquidity-opposition-analysis.json", "H2_LIQUIDITY_OPPOSITION", "H2", ["top5_imbalance_directional", "top10_imbalance_directional", "opposing_to_directional_depth_ratio", "opposing_depth_relative_to_causal_baseline", "mlofi_top5_500ms_directional", "mlofi_top5_2s_persistence", "mlofi_top5_2s_latest_sign_directional"]),
        ("resiliency-analysis.json", "H3_LIQUIDITY_REPLENISHMENT_FAILED_CONTINUATION", "H3", ["resiliency_consumed_depth", "resiliency_restored_depth", "resiliency_restore_to_consumption_ratio", "price_impact_ticks_per_directional_contract"])):
        _write_json(output / filename, {"hypothesis": hypothesis, "features": fields, "analysis": feature_analysis,
            "postsignal_paths": postsignal_analysis, "decision": decisions["hypothesis_diagnostics"].get(diagnostic_key)})
    summary = {"study_id": RUN_ID, "status": "PASS" if len(verified_dates) == len(pair_dates) and econ_summary["reconciles_frozen_pair_outcome"] else "PARTIAL",
        "primary_decision": decisions["decision"], "decision_reason": decisions["reason"],
        "authorized_source_dates_processed": len(verified_dates), "source_dates_hash_verified": len(source_audit["authorized_dates"]),
        "primary_source_dates_processed": len(verified_dates),
        "secondary_signal_source_dates_processed": len({r["date"] for r in secondary_rows}), "primary_matched_pairs": 113,
        "feature_event_rows": len(all_event_features), "postsignal_path_rows": len(all_paths),
        "matched_markout_reconciliation": econ_summary["reconciles_frozen_pair_outcome"],
        "economics_periods": econ_summary["periods"], "analysis": analysis,
        "secondary_displacement_signal_coverage": secondary_coverage,
        "source_audit": {"source_dates": verified_dates, "acquisition_manifest": source_audit,
                         "candidate_tapes": econ_summary["candidate_tapes_verified"]},
        "scope": {"optimization": False, "new_exit_grid": False, "new_trading_filters": False,
            "2026_data_accessed": False, "final_oos_accessed": False, "data_downloaded": False,
            "production_code_changed": False, "commit_performed": False}, "runtime_seconds": time.perf_counter()-started}
    _write_json(output / "source-coverage.json", {"status": "PASS" if len(verified_dates) == len(pair_dates) and len(source_audit["authorized_dates"]) == 54 else "PARTIAL",
        "authorized_manifest_dates": [r["date"] for r in source_audit["authorized_dates"]],
        "cohort_authorized_dates": sorted(pair_dates), "primary_event_dates_processed": verified_dates,
        "secondary_signal_dates_processed": sorted({r["date"] for r in secondary_rows}),
        "verified_source_dates": source_audit["authorized_dates"],
        "authorized_source_manifest_sha256": source_audit["acquisition_manifest_sha256"],
        "source_coverage_sha256": _sha(SOURCE_COVERAGE), "candidate_tape_checks": econ_summary["candidate_tapes_verified"]})
    _write_json(output / "summary.json", summary)
    report = _render_report(summary, feature_analysis, econ_summary, decisions)
    (output / "report.md").write_text(report, encoding="utf-8")
    run_manifest = {"status": summary["status"], "study_id": RUN_ID, "contract_sha256": contract_hash,
        "completed_dates": sorted(pair_dates | {r["date"] for r in secondary_rows}),
        "source_dates_verified": len(source_audit["authorized_dates"]),
        "primary_event_dates_processed": sorted(pair_dates),
        "secondary_signal_dates_processed": sorted({r["date"] for r in secondary_rows}), "files": {},
        "runtime_seconds": summary["runtime_seconds"], "2026_data_accessed": False, "final_oos_accessed": False,
        "data_downloaded": False, "production_code_changed": False, "commit_performed": False}
    _write_json(output / "run-manifest.json", run_manifest)
    # artifact-hashes.json includes this manifest, so it must not be included
    # back into this manifest (that would create a hash cycle).
    run_manifest["files"] = {p.name: _sha(p) for p in sorted(output.iterdir())
                              if p.is_file() and p.name not in {"run-manifest.json", "artifact-hashes.json"}}
    _write_json(output / "run-manifest.json", run_manifest)
    _write_json(output / "artifact-hashes.json", {"status": "HASHED", "files": {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}})
    return summary


def _scientific_decision(events: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], econ: Mapping[str, Any]) -> dict[str, Any]:
    # Prespecified signs. Require the paired feature pattern and relevant
    # continuous paired feature/outcome association to agree in both periods.
    specs = {
        "H1": {"label": "AGGRESSION_EXHAUSTION", "expected": {"aggressive_flow_30s": 1,
            "price_impact_ticks_per_directional_contract": -1}, "relationship_feature": "price_impact_ticks_per_directional_contract", "relationship_sign": 1},
        "H2": {"label": "OPPOSING_LIQUIDITY", "expected": {"top5_imbalance_directional": -1,
            "mlofi_top5_500ms_directional": -1}, "relationship_feature": "top5_imbalance_directional", "relationship_sign": 1},
        "H3": {"label": "REPLENISHMENT_FAILED_CONTINUATION", "expected": {"resiliency_restore_to_consumption_ratio": 1,
            "price_impact_ticks_per_directional_contract": -1}, "relationship_feature": "resiliency_restore_to_consumption_ratio", "relationship_sign": -1},
    }
    evidence = {}
    for hypothesis, spec in specs.items():
        per_period = {}
        coherent = True
        for period in ("SPRING_2025", "OCTOBER_2025"):
            details = {}
            for feature, expected_sign in spec["expected"].items():
                stats = analysis["periods"].get(feature, {}).get(period, {})
                diff = stats.get("paired_difference", {})
                val = diff.get("mean")
                details[feature] = {"paired_difference": val, "ci95": diff.get("ci95"), "expected_sign": expected_sign}
                if val is None or val * expected_sign <= 0:
                    coherent = False
            relationship = analysis["periods"].get(spec["relationship_feature"], {}).get(period, {})
            corr = (relationship.get("paired_feature_difference_vs_markout_difference") or {}).get("pearson")
            ci = relationship.get("paired_feature_outcome_correlation_date_cluster_ci95")
            details["feature_outcome_relationship"] = {"feature": spec["relationship_feature"], "pearson": corr,
                "date_cluster_ci95": ci, "expected_sign": spec["relationship_sign"]}
            if corr is None or corr * spec["relationship_sign"] <= 0 or not ci or ci[0] * spec["relationship_sign"] <= 0:
                coherent = False
            per_period[period] = details
        evidence[hypothesis] = {"label": spec["label"], "periods": per_period, "supported": coherent,
            "rule": "prespecified feature directions and date-cluster correlation interval for paired outcome difference agree in both Spring and October"}
    supported = [k for k, v in evidence.items() if v["supported"]]
    if len(supported) > 1:
        decision = "MULTIPLE_MECHANISMS_POSSIBLE"
    elif supported:
        decision = {"H1": "EXHAUSTION_MECHANISM_SUPPORTED_EXPLORATORILY", "H2": "OPPOSING_LIQUIDITY_MECHANISM_SUPPORTED_EXPLORATORILY", "H3": "REPLENISHMENT_MECHANISM_SUPPORTED_EXPLORATORILY"}[supported[0]]
    else:
        valid_fraction = sum(bool(r["book_valid_at_decision"]) for r in events) / max(len(events), 1)
        decision = "INSUFFICIENT_VALID_DEPTH_EVIDENCE" if valid_fraction < .8 else "NO_STABLE_EXPLANATORY_MECHANISM"
    return {"decision": decision, "reason": "Mechanism labels use a frozen directional consistency rule across the two previously explored periods, including date-cluster uncertainty on feature/outcome association; this remains descriptive and noncausal.", "hypothesis_diagnostics": evidence}


def _render_report(summary: Mapping[str, Any], analysis: Mapping[str, Any], economics: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [f"# {RUN_ID}", "", "Exploratory native ES MBP-10 mechanism study; frozen 113 signal/control pairs; no strategy optimization.", "",
        f"Status: **{summary['status']}**", f"Primary decision: **{decision['decision']}**", "",
        "## Scope and caution", "", "Spring and October 2025 were previously explored development periods, not validation. Pair membership was frozen before native depth extraction. Date is the dependence cluster. Results are descriptive, not causal. MBP-10 cannot identify individual queue orders, exact cancellations, icebergs, or spoofing.", "",
        "## Markout economics", "", "The exact decomposition is directional mid markout minus entry half-spread minus exit half-spread minus two adverse ticks minus 0.48 fee ticks. Fees are $6 round trip divided by $12.50 per ES tick per contract.", ""]
    secondary = summary.get("secondary_displacement_signal_coverage", {})
    lines += ["## Secondary displacement signal cohort", "",
        f"Extracted {secondary.get('features_extracted', 0)} of {secondary.get('expected_signals', 0)} frozen displacement signals; valid books={secondary.get('valid_books', 0)}, invalid={secondary.get('invalid_books', 0)}. This signal-only cohort is descriptive and is not used in primary matched-pair comparisons.", ""]
    for period, metrics in economics["periods"].items():
        lines.append(f"### {period}")
        for field, result in metrics.items():
            lines.append(f"- {field}: mean={result['mean_signal_minus_control']}; date-cluster 95% CI={result['date_cluster_ci95']}; pairs={result['pairs']}")
        lines.append("")
    lines += ["## Feature comparisons", "", "Signal-minus-control differences are direction-normalized; post-signal paths are separate explanatory outcomes.", ""]
    for field, result in analysis["features"].items():
        lines.append(f"- **{field}**: all-period paired mean={result['paired_difference_signal_minus_control']['mean']}, date-cluster CI={result['paired_difference_signal_minus_control']['ci95']}, pairs={result['complete_pairs']}; feature-vs-markout Pearson={result['signal_feature_vs_signal_net_5m']['pearson']}")
    lines += ["", "## Post-signal paths", "", "These windows are post-event explanatory outcomes, not entry predictors.", ""]
    for window, details in analysis.get("postsignal_paired_paths", {}).get("windows", {}).items():
        all_metrics = details["paired_signal_minus_control"].get("ALL", {})
        m = all_metrics.get("directional_mid_change_ticks", {})
        opp = all_metrics.get("opposing_depth_change", {})
        lines.append(f"- **{window}**: paired signal-control directional mid change={m.get('mean')} (date CI={m.get('ci95')}); opposing displayed-depth change={opp.get('mean')} (date CI={opp.get('ci95')})")
    lines += ["", "## Scientific interpretation", "", decision["reason"], "", "No thresholds, filters, exit grids, or parameter searches were evaluated.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true", help="replace only this study output directory")
    args = parser.parse_args(argv)
    result = run(args.output, force=args.force)
    print(json.dumps({"ES_JJ_CONTINUATION_NATIVE_MBP10_MECHANISM": result["status"],
                      "artifact_root": str(args.output), "primary_decision": result["primary_decision"],
                      "elapsed_seconds": result["runtime_seconds"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
