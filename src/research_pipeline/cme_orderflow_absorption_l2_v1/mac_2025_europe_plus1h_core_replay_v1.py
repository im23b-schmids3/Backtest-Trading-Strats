"""Execute the frozen 54-day Europe-session +1h core replay comparison.

Only the local 2025 ES MBP-10 package is read. Existing sealed baseline tapes
are re-evaluated under the frozen live snapshot; the shifted-window tapes are
rebuilt independently from the corresponding native DBNs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from . import mac_2025_absorption_relative_normalization as norm
from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_class_b_optuna as class_b_eval
from . import mac_2025_es_only_train_baseline as baseline
from .model import L2Config

RUN_ID = "CMEOrderflow_ES_LIVE_STRATEGY_EUROPE_SESSION_PLUS1H_CORE_V1"
OUT_ROOT = Path("research_runs") / RUN_ID
FAMILIES = tuple(norm.LIVE_TO_TAPE)
EXPECTED_CONFIG_SHA = "8d06fbd5f9eb8850a1660c5bde96d9bb3bbe8b0fcddcbfc1ca5b2f31530dd23b"
TRAIN_TAPES = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE/candidate-tapes/tapes")
OCT_TAPES = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928/october-candidate-tapes/tapes")
TRAIN_PROFILES = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE/candidate-tapes/_cache/profiles")
OCT_PROFILES = Path("research_runs/CMEOrderflowAbsorption.ES_L2_LIVE_CONFIG_UNDER_BACKTEST_EXECUTION_OCTOBER_20260928/october-candidate-tapes/profiles")
UTC = timezone.utc
FIRST_HOUR_COVERAGE_VERSION = 1


class ReplayError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns or ["empty"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _atomic_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                for row in rows:
                    gz.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _value(item: Any, default: Any = None) -> Any:
    return item.get("value", default) if isinstance(item, dict) else (default if item is None else item)


def _frozen_configs() -> tuple[dict[str, L2Config], dict[str, dict[str, Any]], dict[str, Any], str]:
    snapshot = norm.LIVE_CONFIG_PATH
    digest = _sha(snapshot)
    if digest != EXPECTED_CONFIG_SHA:
        raise ReplayError(f"frozen strategy snapshot SHA mismatch: {digest}")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    raw_by_family: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("strategies", []):
        key = _value(row.get("derived_parameters", {}).get("runtime_family_key"))
        family = norm.FAMILY_MAP.get(str(key))
        if family in FAMILIES:
            raw_by_family[family] = row
    if set(raw_by_family) != set(FAMILIES):
        raise ReplayError("snapshot does not contain all four requested frozen strategies")
    class_a: dict[str, L2Config] = {}
    class_b: dict[str, dict[str, Any]] = {}
    for family, row in raw_by_family.items():
        ca = row["class_a_parameters"]
        raw_weights = ca["feature_weights"]
        weights = {f"{name}_weight": float(_value(raw_weights[name])) for name in
                   ("aggression", "restoration", "price_resistance", "persistence", "multi_level_support")}
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
            raise ReplayError(f"frozen score weights do not normalize: {family}")
        class_a[family] = L2Config(**weights, min_quality_score=float(_value(ca["quality_threshold"])))
        cb, ex = row["class_b_parameters"], row["execution_parameters"]
        class_b[family] = {
            "min_confirmation_seconds": float(_value(cb.get("min_confirmation_seconds"), 5.0)),
            "max_confirmation_seconds": float(_value(cb.get("max_confirmation_seconds"), 15.0)),
            "favorable_confirmation_ticks": float(_value(cb.get("favorable_ticks"), 3.0)),
            "confirmation_execution_count": int(_value(cb.get("confirmation_execution_count"), 1)),
            "confirmation_volume_threshold": int(_value(cb.get("confirmation_cumulative_volume"), 0)),
            "stop_ticks": int(_value(ex.get("stop_ticks"), 5)),
            "target_r": float(_value(ex.get("target_r"), 3.0)),
        }
    return class_a, class_b, payload, digest


def _period_dates(requests: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    spring = sorted(set(baseline.TRAIN_DATES))
    october = sorted({str(r["session_date"]) for r in requests.values()
                      if isinstance(r, dict) and r.get("category") == "VALIDATION"
                      and str(r.get("session_date", "")).startswith("2025-10-")})
    if len(spring) != 35 or len(october) != 19:
        raise ReplayError(f"expected 35 Spring + 19 October dates, got {len(spring)} + {len(october)}")
    return spring, october


def _source_records(requests: Mapping[str, Any], root: Path, day: str) -> tuple[Path, str, int]:
    item = next((r for r in requests.values() if isinstance(r, dict) and r.get("session_date") == day
                 and r.get("category") in {"TRAIN", "VALIDATION", "DEPENDENCY"}), None)
    if not item:
        raise ReplayError(f"no sealed native source manifest row for {day}")
    path = root / item["path"]
    if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
        raise ReplayError(f"source absent or size mismatch: {path}")
    digest = _sha(path)
    if digest != item.get("sha256"):
        raise ReplayError(f"source hash mismatch: {path}")
    if not item.get("end"):
        raise ReplayError(f"source manifest lacks requested end bound for {day}")
    return path, digest, baseline._ns(str(item["end"]))


def _profile_cache(day: str, path: Path, digest: str, windows: Mapping[str, tuple[int, int]]) -> dict[str, baseline.Profile]:
    root = TRAIN_PROFILES if day in baseline.TRAIN_DATES or day == baseline.DEPENDENCY_DATE else OCT_PROFILES
    cache = root / f"{day}.json"
    if not cache.is_file():
        raise ReplayError(f"required existing causal profile cache missing: {cache}")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    # The Spring cache records status/date explicitly; the earlier October
    # cache format records only its filename, profile dates, and source hash.
    # In both formats the source hash and per-profile date are authoritative.
    if payload.get("source_sha256") != digest or (payload.get("date") not in (None, day)):
        raise ReplayError(f"profile cache source identity mismatch: {cache}")
    profiles = {str(r["session"]): baseline._profile_from_payload(r) for r in payload["profiles"]}
    if any(profile.day != day for profile in profiles.values()):
        raise ReplayError(f"profile cache date mismatch: {cache}")
    for session in ("ASIA", "NY"):
        if session not in profiles or (profiles[session].start_ns, profiles[session].end_ns) != windows[session]:
            raise ReplayError(f"baseline {session} profile cache window mismatch for {day}")
    return profiles


def _plus_windows(day: str) -> dict[str, tuple[int, int]]:
    windows = baseline._session_windows(day)
    start = int(datetime.combine(datetime.fromisoformat(day).date(), clock_time(9, 0), tzinfo=UTC).timestamp() * 1e9)
    if start >= windows["EUROPE"][1]:
        raise ReplayError(f"invalid shifted Europe window for {day}")
    windows["EUROPE"] = (start, windows["EUROPE"][1])
    return windows


def _variant_europe_profile(day: str, path: Path, windows: Mapping[str, tuple[int, int]], *, observe: bool) -> tuple[baseline.Profile, dict[str, Any] | None]:
    from databento import DBNStore
    profile = baseline.Profile.create(day, "EUROPE", *windows["EUROPE"])
    first_ns = baseline._day_ns(day, clock_time(8, 0))
    last_ns = baseline._day_ns(day, clock_time(9, 0))
    first_seen = last_seen = None
    event_count = trade_count = 0
    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
        ts = batch["ts_recv"]
        if observe:
            mask = (ts >= first_ns) & (ts < last_ns)
            if mask.any():
                values = ts[mask]
                first_seen = int(values[0]) if first_seen is None else first_seen
                last_seen = int(values[-1])
                event_count += int(mask.sum())
                trade_count += int((mask & (batch["action"] == b"T") & (batch["size"] > 0)).sum())
        mask = ((batch["action"] == b"T") & (batch["size"] > 0)
                & (ts >= profile.start_ns) & (ts < profile.end_ns))
        if mask.any():
            prices, sizes = batch["price"][mask].astype(np.int64, copy=False), batch["size"][mask].astype(np.int64, copy=False)
            unique, inverse = np.unique(prices, return_inverse=True)
            volume = np.zeros(len(unique), dtype=np.int64)
            np.add.at(volume, inverse, sizes)
            profile.volume_by_tick.update({int(p): int(v) for p, v in zip(unique, volume)})
    if not profile.volume_by_tick:
        raise ReplayError(f"empty shifted Europe executed-volume profile for {day}")
    coverage = None
    if observe:
        coverage = {"date": day, "raw_first_hour_present": event_count > 0,
                    "first_event_timestamp": baseline._iso(first_seen) if first_seen is not None else None,
                    "last_event_timestamp": baseline._iso(last_seen) if last_seen is not None else None,
                    "event_count": event_count, "trade_count": trade_count}
    return profile, coverage


def _iso(ns: Any) -> str | None:
    return baseline._iso(int(ns)) if ns is not None else None


def _trade_rows(trades: Sequence[Mapping[str, Any]], *, day: str, period: str, variant: str,
                tape: candidate_tape.CandidateTape, family_id: str) -> list[dict[str, Any]]:
    family_rows = [r for r in tape.candidates if str(r.get("level", r.get("family_id"))) == family_id]
    by_id = {str(r.get("interaction_id")): r for r in family_rows}
    out = []
    for trade in trades:
        c = by_id.get(str(trade.get("interaction_id")), {})
        out.append({"date": day, "period": period, "variant": variant, "family": family_id,
                    "setup_id": trade.get("setup_id"), "direction": trade.get("direction"),
                    "signal_timestamp": _iso(c.get("interaction_start_ns")),
                    "entry_timestamp": _iso(trade.get("entry_timestamp_ns")),
                    "entry_price": trade.get("entry"), "stop_price": trade.get("stop"),
                    "target_price": trade.get("target"), "realized_r": trade.get("r_multiple"),
                    "mfe_r": None, "mae_r": None,
                    "target_before_stop": trade.get("exit_reason") == "TARGET" if trade.get("exit_reason") else None,
                    "trade_id": trade.get("trade_id"), "exit_timestamp": _iso(trade.get("exit_timestamp_ns")),
                    "exit_price": trade.get("exit"), "exit_reason": trade.get("exit_reason"),
                    "instrument": trade.get("instrument"), "contracts": trade.get("contracts")})
    return out


def _evaluate_day(tape: candidate_tape.CandidateTape, day: str, period: str, variant: str,
                  configs: Mapping[str, L2Config], class_b: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trades_out: list[dict[str, Any]] = []
    level_values: dict[str, Any] = {"date": day, "period": period, "variant": variant}
    for family in FAMILIES:
        params = class_b_eval._tape_parameters(configs[family], class_b[family])
        ft = candidate_tape.CandidateTape(tape.metadata,
              tuple(r for r in tape.candidates if str(r.get("level", r.get("family_id"))) == family), tape.events)
        result = candidate_tape.evaluate_candidate_tape(ft, params, config=configs[family])
        trades_out.extend(_trade_rows(result["trades"], day=day, period=period, variant=variant,
                                      tape=tape, family_id=family))
    return trades_out, level_values


def _read_checkpoint(path: Path, *, source_sha: str, variant: str,
                     session_windows: Mapping[str, tuple[int, int]] | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") == "COMPLETE" and payload.get("source_sha256") == source_sha and payload.get("variant") == variant:
            if session_windows is not None:
                expected = {name: list(window) for name, window in session_windows.items()}
                saved = payload.get("session_windows")
                if saved is None and isinstance(payload.get("profiles"), list):
                    saved = {str(row["session"]): [int(row["start_ns"]), int(row["end_ns"])]
                             for row in payload["profiles"] if "session" in row and
                             "start_ns" in row and "end_ns" in row}
                if saved != expected:
                    return None
            return payload
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _aggregate(trades: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dayrows = []
    for day in dates:
        period = "SPRING_2025" if day in baseline.TRAIN_DATES else "OCTOBER_2025"
        for family in FAMILIES:
            rec = {"date": day, "period": period, "family": family}
            for variant in ("BASELINE_0800", "PLUS1H_0900"):
                rows = [r for r in trades if r["date"] == day and r["family"] == family and r["variant"] == variant]
                values = [float(r["realized_r"]) for r in rows if r.get("realized_r") is not None]
                prefix = "baseline" if variant == "BASELINE_0800" else "plus1h"
                rec[f"{prefix}_trade_count"] = len(rows)
                rec[f"{prefix}_net_r"] = float(sum(values))
                rec[f"{prefix}_avg_r_per_trade"] = float(np.mean(values)) if values else None
                rec[f"{prefix}_win_rate"] = float(sum(v > 0 for v in values) / len(values)) if values else None
            rec["delta_net_r"] = rec["plus1h_net_r"] - rec["baseline_net_r"]
            dayrows.append(rec)
    summaries = []
    for family in FAMILIES:
        for period in ("SPRING_2025", "OCTOBER_2025"):
            rows = [r for r in dayrows if r["family"] == family and r["period"] == period]
            summary: dict[str, Any] = {"family": family, "period": period}
            for variant, prefix in (("BASELINE_0800", "baseline"), ("PLUS1H_0900", "plus1h")):
                ts = [r for r in trades if r["family"] == family and r["period"] == period and r["variant"] == variant]
                vals = [float(t["realized_r"]) for t in ts if t.get("realized_r") is not None]
                summary[f"{prefix}_trades"] = len(ts)
                summary[f"{prefix}_net_r"] = float(sum(vals))
                summary[f"{prefix}_avg_r_per_trade"] = float(np.mean(vals)) if vals else None
            deltas = [r["delta_net_r"] for r in rows]
            summary.update(delta_net_r=round(sum(deltas), 12), improved_days=sum(x > 0 for x in deltas),
                           worsened_days=sum(x < 0 for x in deltas), unchanged_days=sum(x == 0 for x in deltas))
            summaries.append(summary)
    return dayrows, summaries


def run(*, output_root: Path = OUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    source_manifest, requests = baseline._manifest(norm.DATA_ROOT)
    spring, october = _period_dates(requests)
    dates = sorted([*spring, *october])
    configs, class_b, snapshot, config_sha = _frozen_configs()
    # Restrict the manifest/source list to the requested 54 target dates and
    # the two already-declared profile dependency dates. No later market data
    # is enumerated or read.
    dependencies = {baseline.DEPENDENCY_DATE, "2025-10-06"}
    source_map: dict[str, tuple[Path, str, int]] = {d: _source_records(requests, norm.DATA_ROOT, d) for d in [*dates, *sorted(dependencies)]}
    for day in dates:
        if source_map[day][2] < max(end for _, end in baseline._session_windows(day).values()):
            raise ReplayError(f"verified source request bound does not cover required session end for {day}")
    identity = {"run_id": RUN_ID, "dates": dates, "families": list(FAMILIES), "snapshot_sha256": config_sha,
                "session_windows": {"baseline_europe_start_utc": "08:00:00", "plus1h_europe_start_utc": "09:00:00",
                                    "europe_end": "09:30 America/New_York"},
                "source_manifest_sha256": _sha(norm.DATA_ROOT / baseline.MANIFEST_NAME),
                "source_hashes": {d: source_map[d][1] for d in source_map},
                "baseline_tape_semantic_sha256": norm.EXPECTED_TAPE_SEMANTIC_SHA,
                "data_downloaded": False, "oos_accessed": False}
    _atomic_json(output_root / "run-identity.json", identity)
    profiles: dict[str, dict[str, baseline.Profile]] = {}
    raw_coverage: list[dict[str, Any]] = []
    profile_checkpoint_dir = output_root / "checkpoints" / "plus1h-profiles"
    for index, day in enumerate(sorted(source_map), 1):
        path, sha, _coverage_end = source_map[day]
        win = _plus_windows(day)
        cp_path = profile_checkpoint_dir / f"{day}.json.gz"
        payload = _read_checkpoint(cp_path, source_sha=sha, variant="PLUS1H_PROFILE", session_windows=win)
        if payload is not None and payload.get("coverage_version") != FIRST_HOUR_COVERAGE_VERSION:
            payload = None
        if payload is None:
            baseline_profiles = _profile_cache(day, path, sha, baseline._session_windows(day))
            europe, coverage = _variant_europe_profile(day, path, win, observe=day in dates)
            merged = dict(baseline_profiles)
            merged["EUROPE"] = europe
            payload = {"status": "COMPLETE", "variant": "PLUS1H_PROFILE", "coverage_version": FIRST_HOUR_COVERAGE_VERSION, "date": day,
                       "source_sha256": sha, "profiles": [baseline._profile_payload(v) for v in merged.values()],
                       "coverage": coverage}
            _write_checkpoint(cp_path, payload)
        profiles[day] = {str(row["session"]): baseline._profile_from_payload(row) for row in payload["profiles"]}
        if day in dates:
            raw_coverage.append(payload["coverage"])
        coverage = payload.get("coverage")
        print(f"PROFILE {index}/{len(source_map)} {day} events={coverage['event_count'] if coverage else 'dependency'}", flush=True)
    _atomic_csv(output_root / "first-hour-raw-coverage.csv", raw_coverage)

    all_trades: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    # Re-evaluate the already sealed baseline candidate tapes under the exact
    # frozen live snapshot; this is a new replay evaluation, not copied output.
    for index, day in enumerate(dates, 1):
        period = "SPRING_2025" if day in spring else "OCTOBER_2025"
        tape_path = (TRAIN_TAPES if day in spring else OCT_TAPES) / f"{day}-candidate-tape.npz"
        if not tape_path.is_file():
            raise ReplayError(f"sealed baseline tape missing: {tape_path}")
        tape = candidate_tape.load_tape(tape_path)
        source_sha = source_map[day][1]
        expected = baseline._session_windows(day)
        if tape.metadata.get("date") != day or tape.metadata.get("source_sha256") != source_sha or tape.metadata.get("semantic_sha256") != norm.EXPECTED_TAPE_SEMANTIC_SHA:
            raise ReplayError(f"baseline candidate tape identity mismatch: {day}")
        if {k: list(v) for k, v in expected.items()} != tape.metadata.get("session_windows"):
            raise ReplayError(f"baseline window mismatch: {day}")
        outpath = output_root / "checkpoints" / "baseline" / f"{day}.json.gz"
        saved = _read_checkpoint(outpath, source_sha=source_sha, variant="BASELINE_0800")
        if saved is None:
            trades, _ = _evaluate_day(tape, day, period, "BASELINE_0800", configs, class_b)
            saved = {"status": "COMPLETE", "variant": "BASELINE_0800", "date": day,
                     "source_sha256": source_sha, "tape_sha256": _sha(tape_path), "trades": trades}
            _write_checkpoint(outpath, saved)
        all_trades.extend(saved["trades"])
        print(f"BASELINE {index}/{len(dates)} {day} trades={len(saved['trades'])}", flush=True)

    # Independently replay the entire raw day under shifted Europe boundaries.
    for index, day in enumerate(dates, 1):
        period = "SPRING_2025" if day in spring else "OCTOBER_2025"
        path, source_sha, _coverage_end = source_map[day]
        win = _plus_windows(day)
        profile_path = output_root / "checkpoints" / "plus1h-profiles" / f"{day}.json.gz"
        profile_payload = _read_checkpoint(profile_path, source_sha=source_sha, variant="PLUS1H_PROFILE",
                                           session_windows=win)
        assert profile_payload is not None
        current = profiles[day]
        prior_day = (baseline.DEPENDENCY_DATE if day == spring[0] else
                     "2025-10-06" if day == october[0] else
                     (spring[spring.index(day)-1] if day in spring else october[october.index(day)-1]))
        prior = profiles[prior_day]
        outpath = output_root / "checkpoints" / "plus1h" / f"{day}.json.gz"
        saved = _read_checkpoint(outpath, source_sha=source_sha, variant="PLUS1H_0900",
                                 session_windows=win)
        tape_path = output_root / "plus1h-tapes" / f"{day}-candidate-tape.npz"
        if saved is None:
            # Tape files are date-specific and isolated from both baseline tape
            # roots and the shifted-profile cache.
            if tape_path.is_file():
                tape = candidate_tape.load_tape(tape_path)
                if (tape.metadata.get("source_sha256") != source_sha or tape.metadata.get("date") != day
                        or tape.metadata.get("session_windows") != {k: list(v) for k, v in win.items()}
                        or set(tape.metadata.get("available_families", ())) != set(FAMILIES)):
                    tape_path.unlink()
            if not tape_path.is_file():
                tape, _raw_result = candidate_tape.build_candidate_tape(
                    day, path, prior, current, output_path=tape_path, source_sha256=source_sha,
                    semantic_sha256=hashlib.sha256((baseline._semantic_sha256() + candidate_tape._semantic_sha256()).encode()).hexdigest(),
                    config=L2Config(), session_windows=win, family_ids=frozenset(FAMILIES),
                    source_coverage_end_ns=source_map[day][2])
            else:
                tape = candidate_tape.load_tape(tape_path)
            trades, _ = _evaluate_day(tape, day, period, "PLUS1H_0900", configs, class_b)
            saved = {"status": "COMPLETE", "variant": "PLUS1H_0900", "date": day,
                     "source_sha256": source_sha, "tape_sha256": _sha(tape_path), "trades": trades,
                     "session_windows": tape.metadata.get("session_windows")}
            _write_checkpoint(outpath, saved)
        all_trades.extend(saved["trades"])
        print(f"PLUS1H {index}/{len(dates)} {day} trades={len(saved['trades'])}", flush=True)

    baseline_trades = [r for r in all_trades if r["variant"] == "BASELINE_0800"]
    plus_trades = [r for r in all_trades if r["variant"] == "PLUS1H_0900"]
    dayrows, summaries = _aggregate(all_trades, dates)
    _atomic_gzip_jsonl(output_root / "baseline-trades.jsonl.gz", baseline_trades)
    _atomic_gzip_jsonl(output_root / "plus1h-trades.jsonl.gz", plus_trades)
    _atomic_csv(output_root / "family-day-comparison.csv", dayrows)
    _atomic_json(output_root / "family-summary.json", {"families": summaries})

    for day in dates:
        period = "SPRING_2025" if day in spring else "OCTOBER_2025"
        prev = (baseline.DEPENDENCY_DATE if day == spring[0] else "2025-10-06" if day == october[0] else
                spring[spring.index(day)-1] if day in spring else october[october.index(day)-1])
        base_profiles = _profile_cache(day, source_map[day][0], source_map[day][1], baseline._session_windows(day))
        plus_profile = profiles[day]["EUROPE"].values()
        base_profile = base_profiles["EUROPE"].values()
        prev_base = _profile_cache(prev, source_map[prev][0], source_map[prev][1], baseline._session_windows(prev))["EUROPE"].values()
        prev_plus = profiles[prev]["EUROPE"].values()
        level_rows.append({"date": day, "period": period,
            "baseline_europe_high": base_profile["HIGH"], "plus1h_europe_high": plus_profile["HIGH"],
            "baseline_europe_poc": base_profile["POC"], "plus1h_europe_poc": plus_profile["POC"],
            "baseline_europe_vah": base_profile["VAH"], "plus1h_europe_vah": plus_profile["VAH"],
            "baseline_prior_europe_high": prev_base["HIGH"], "plus1h_prior_europe_high": prev_plus["HIGH"],
            "baseline_prior_europe_vah": prev_base["VAH"], "plus1h_prior_europe_vah": prev_plus["VAH"],
            "baseline_prior_europe_poc": prev_base["POC"], "plus1h_prior_europe_poc": prev_plus["POC"]})
    _atomic_csv(output_root / "level-comparison.csv", level_rows)
    summary = {"run_id": RUN_ID, "status": "COMPLETE", "dates_expected": 54,
        "baseline_dates_completed": 54, "plus1h_dates_completed": 54,
        "first_hour_raw_present_dates": sum(bool(r["raw_first_hour_present"]) for r in raw_coverage),
        "first_hour_raw_missing_dates": [r["date"] for r in raw_coverage if not r["raw_first_hour_present"]],
        "baseline_total_trades": len(baseline_trades), "plus1h_total_trades": len(plus_trades),
        "family_count": len(FAMILIES), "target_date_count": len(dates),
        "baseline_net_r": sum(float(r["realized_r"] or 0) for r in baseline_trades),
        "plus1h_net_r": sum(float(r["realized_r"] or 0) for r in plus_trades),
        "data_downloaded": False, "oos_accessed": False, "optimization_performed": False,
        "runtime_seconds": time.perf_counter() - started}
    _atomic_json(output_root / "summary.json", summary)
    _atomic_json(output_root / "run-manifest.json", {**identity, **summary,
        # A manifest cannot contain a stable digest of itself. Hash every
        # completed output artifact, excluding only this manifest.
        "outputs": {p.name: _sha(p) for p in output_root.iterdir()
                    if p.is_file() and p.name != "run-manifest.json"},
        "config_sha256": config_sha, "completed_at_utc": datetime.now(UTC).isoformat()})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)
    result = run(output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
