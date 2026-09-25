"""Build and evaluate the parameter-independent MAC 2025 candidate tape.

The tape is deliberately downstream of the existing causal MBP-10 route.  It
stores every completed level interaction produced by that route (including
interactions rejected by the current thresholds), primitive feature inputs,
the baseline component scores, and an ordered ES quote/execution path.  It
does not store fitted decisions and it never opens a DBN file during offline
evaluation.

The cache is invalidated by both the source SHA-256 and the semantic source
hash.  A tape is therefore a reproducible preparation artifact, not a second
strategy implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import struct
import tempfile
import time as wall_time
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import mac_2025_es_only_train_baseline as baseline
from .model import (
    ES_COMMISSION,
    ES_POINT_VALUE,
    ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    ENTRY_LATENCY_NS,
    L2Config,
    INACTIVITY_NS,
    EXIT_RESET_NS,
    MAX_CONFIRMATION_NS,
    MIN_CONFIRMATION_NS,
    initial_prices,
    size_for_instrument,
)


TAPE_VERSION = "MAC2025_CANDIDATE_TAPE_V2_BBO_COMPLETE"
TAPE_SEMANTIC_REVISION = "MAC2025_CANDIDATE_TAPE_SEMANTICS_V3_BBO_COMPLETE"
TAPE_FILENAME = "candidate-tape.npz"
TAPE_MANIFEST_FILENAME = "candidate-tape-manifest.json"
WEIGHT_NAMES = (
    "aggression_weight", "restoration_weight", "price_resistance_weight",
    "persistence_weight", "multi_level_support_weight",
)
SCORE_NAMES = (
    "aggression_score", "restoration_score", "price_resistance_score",
    "persistence_score", "multi_level_support_score", "false_refill_penalty",
)
RAW_FEATURE_NAMES = (
    "directional_aggressive_volume", "opposite_aggressive_volume",
    "aggressive_volume_imbalance", "executed_to_initial_displayed_ratio",
    "execution_count", "relevant_execution_count",
    "consume_restore_cycles", "maximum_through_level_progress_ticks",
    "interaction_rejection_ticks", "executed_volume_at_defended_price",
    "executed_volume_within_1_tick", "executed_volume_within_2_ticks",
    "initial_displayed_depth_at_price", "median_displayed_depth_at_price",
    "max_displayed_depth_at_price", "defended_price_present_fraction",
    "defended_depth_time_weighted_mean", "cumulative_consumed_volume",
    "cumulative_restored_volume", "restored_depth_volume",
    "mean_restoration_latency_ms", "median_restoration_latency_ms",
    "fastest_restoration_latency_ms", "restoration_to_consumption_ratio",
    "multi_level_ofi", "bid_depth_1", "ask_depth_1", "bid_depth_3",
    "ask_depth_3", "bid_depth_5", "ask_depth_5", "depth_imbalance_1",
    "depth_imbalance_3", "depth_imbalance_5", "depth_recovery_100ms",
    "depth_recovery_250ms", "depth_recovery_500ms", "depth_recovery_1s",
    "unexecuted_add_volume", "rapid_cancel_volume", "rapid_cancel_ratio",
    "restoration_supported_by_execution_ratio",
    "restoration_away_from_defended_price_volume", "adverse_progress_per_100_aggressive_contracts",
    "aggressive_contracts_per_adverse_tick", "final_through_level_progress_ticks",
)
FEATURE_NAMES = RAW_FEATURE_NAMES + SCORE_NAMES


EVENT_DTYPE = np.dtype([
    ("timestamp_ns", "<i8"),
    ("bid", "<f8"),
    ("ask", "<f8"),
    ("execution_price", "<f8"),
    ("execution_size", "<i8"),
    ("aggressor", "i1"),  # BUY=1, SELL=-1, unknown=0
    ("session", "i1"),    # ASIA=0, EUROPE=1, NY=2
])
EVENT_STRUCT = struct.Struct("<qdddqbb")


class CandidateTapeError(RuntimeError):
    """The candidate cache is incomplete, stale, or internally inconsistent."""


class EventSpool:
    """Bounded-memory sparse public path accumulator.

    Unchanged quotes cannot change a first stop/target hit.  We retain BBO
    changes, all executions, and the first public observation after each
    possible two-millisecond entry delay.  This preserves the exact lookup
    points needed by the evaluator without retaining a Python object per raw
    MBP row.
    """

    def __init__(self) -> None:
        fd, name = tempfile.mkstemp(prefix="mac2025-candidate-events-", suffix=".bin")
        os.close(fd)
        self.path = Path(name)
        self.handle = self.path.open("wb")
        self.count = 0
        self.bbo_transition_count = 0
        self.execution_event_count = 0
        self.previous: tuple[str, float, float] | None = None
        self.pending_entry_probes: list[int] = []

    def append(self, row: Mapping[str, Any]) -> None:
        timestamp = int(row["timestamp_ns"])
        bid, ask = float(row["bid"]), float(row["ask"])
        execution_size = int(row.get("execution_size", 0) or 0)
        execution_price = float(row["execution_price"]) if row.get("execution_price") is not None else float("nan")
        session = str(row.get("session", ""))
        changed = self.previous is None or self.previous != (session, bid, ask)
        probe = any(timestamp >= ready for ready in self.pending_entry_probes)
        self.pending_entry_probes = [ready for ready in self.pending_entry_probes if timestamp < ready]
        if changed or execution_size > 0 or probe:
            self.handle.write(EVENT_STRUCT.pack(
                timestamp, bid, ask, execution_price, execution_size,
                _aggressor_code(row.get("aggressor")), _session_code(session),
            ))
            self.count += 1
            if changed:
                self.bbo_transition_count += 1
            if execution_size > 0:
                self.execution_event_count += 1
        if execution_size > 0:
            self.pending_entry_probes.append(timestamp + 2_000_000)
        self.previous = (session, bid, ask)

    def to_array(self) -> np.ndarray:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        raw = self.path.read_bytes()
        if len(raw) % EVENT_STRUCT.size:
            raise CandidateTapeError("event spool has a partial record")
        return np.frombuffer(raw, dtype=EVENT_DTYPE).copy()

    def close(self) -> None:
        try:
            if not self.handle.closed:
                self.handle.close()
        finally:
            self.path.unlink(missing_ok=True)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _aggressor_code(value: Any) -> int:
    return 1 if value == "BUY" else -1 if value == "SELL" else 0


def _session_code(value: Any) -> int:
    return {"ASIA": 0, "EUROPE": 1, "NY": 2}.get(str(value), -1)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class CandidateTape:
    """In-memory representation used by the offline evaluator only."""

    metadata: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    events: np.ndarray

    @property
    def event_timestamps(self) -> np.ndarray:
        return self.events["timestamp_ns"]

    @property
    def feature_matrix(self) -> np.ndarray:
        return np.asarray([
            [_finite(row.get(name)) for name in FEATURE_NAMES]
            for row in self.candidates
        ], dtype=np.float64)


def _candidate_payload(rows: Iterable[Mapping[str, Any]]) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    candidates: list[dict[str, Any]] = []
    for ordinal, source in enumerate(rows):
        row = dict(source)
        row.setdefault("candidate_id", row.get("interaction_id"))
        row.setdefault("candidate_ordinal", ordinal)
        row["baseline_accepted"] = bool(row.get("accepted", False))
        row["baseline_setup_id"] = f"L2:{row['interaction_id']}" if row.get("accepted") else None
        # Preserve all identity and numeric fields needed for auditability;
        # nested diagnostic values are excluded from this compact cache.
        candidates.append({key: value for key, value in row.items()
                           if not isinstance(value, (dict, list, tuple))})
    return tuple(candidates), np.asarray([
        [_finite(row.get(name)) for name in FEATURE_NAMES] for row in candidates
    ], dtype=np.float64)


def _event_array(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    events = np.zeros(0, dtype=EVENT_DTYPE)
    materialized = list(rows)
    events = np.zeros(len(materialized), dtype=EVENT_DTYPE)
    for index, row in enumerate(materialized):
        events[index] = (
            int(row["timestamp_ns"]), float(row["bid"]), float(row["ask"]),
            float(row["execution_price"]) if row.get("execution_price") is not None else np.nan,
            int(row.get("execution_size", 0) or 0), _aggressor_code(row.get("aggressor")),
            _session_code(row.get("session")),
        )
    if len(events) and np.any(np.diff(events["timestamp_ns"]) < 0):
        raise CandidateTapeError("candidate public-event timestamps are not ordered")
    return events


def _atomic_npz(path: Path, *, candidates: Sequence[Mapping[str, Any]], matrix: np.ndarray,
                events: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(
            temporary,
            candidate_json=np.asarray(json.dumps(list(candidates), sort_keys=True, default=_json_default)),
            candidate_features=matrix,
            events=events,
            metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True, default=_json_default)),
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_tape(path: Path, tape: CandidateTape) -> Path:
    _atomic_npz(path, candidates=tape.candidates, matrix=tape.feature_matrix,
                events=tape.events, metadata=tape.metadata)
    return path


def _tape_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}-manifest.json")


def load_tape(path: Path, *, source_sha256: str | None = None,
              semantic_sha256: str | None = None) -> CandidateTape:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            candidates = tuple(json.loads(str(archive["candidate_json"].item())))
            events = np.asarray(archive["events"], dtype=EVENT_DTYPE)
            matrix = np.asarray(archive["candidate_features"], dtype=np.float64)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CandidateTapeError(f"unreadable candidate tape: {path}") from exc
    if metadata.get("tape_version") != TAPE_VERSION:
        raise CandidateTapeError("candidate tape version mismatch")
    if metadata.get("bbo_path_complete") is not True:
        raise CandidateTapeError("candidate tape lacks complete executable BBO path guarantee")
    if source_sha256 is not None and metadata.get("source_sha256") != source_sha256:
        raise CandidateTapeError("candidate tape source hash mismatch")
    if semantic_sha256 is not None and metadata.get("semantic_sha256") != semantic_sha256:
        raise CandidateTapeError("candidate tape semantic hash mismatch")
    if len(candidates) != len(matrix) or len(events) != int(metadata.get("event_count", -1)):
        raise CandidateTapeError("candidate tape array lengths do not match manifest")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise CandidateTapeError("candidate tape feature schema mismatch")
    if len(events) and np.any(np.diff(events["timestamp_ns"]) < 0):
        raise CandidateTapeError("candidate tape event ordering is invalid")
    return CandidateTape(dict(metadata), candidates, events)


def _component_scores(row: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, float]:
    def value(name: str, default: float = 0.0) -> float:
        return float(row.get(name, default) or default)

    def clamp(value_to_clamp: float) -> float:
        return max(0.0, min(1.0, value_to_clamp))

    directional = value("directional_aggressive_volume")
    imbalance = clamp((value("aggressive_volume_imbalance") + 1.0) / 2.0)
    aggression = (min(1.0, directional / float(parameters["aggressive_volume_saturation"])) +
                  min(1.0, value("relevant_execution_count") / float(parameters["execution_count_saturation"])) +
                  imbalance + min(1.0, value("executed_to_initial_displayed_ratio"))) / 4.0
    restoration = (min(1.0, value("consume_restore_cycles") / float(parameters["restore_cycle_saturation"])) +
                   min(1.0, value("restoration_to_consumption_ratio") / float(parameters["restoration_ratio_saturation"])) +
                   clamp(value("restoration_supported_by_execution_ratio")) +
                   (1.0 - min(1.0, value("mean_restoration_latency_ms") /
                              float(parameters["restoration_latency_saturation_ms"])))) / 4.0
    maximum = value("maximum_through_level_progress_ticks")
    rejection = value("interaction_rejection_ticks")
    resistance = ((1.0 - min(1.0, maximum / float(parameters["max_through_level_progress_ticks"]))) +
                  min(1.0, rejection / float(parameters["rejection_saturation_ticks"]))) / 2.0
    persistence = (clamp(value("defended_price_present_fraction")) +
                   min(1.0, value("defended_depth_time_weighted_mean") /
                       float(parameters["persistence_depth_saturation"]))) / 2.0
    direction = str(row.get("direction", ""))
    directional_book = value("depth_imbalance_5") if direction == "BUYER_ABSORPTION" else -value("depth_imbalance_5")
    directional_ofi = value("multi_level_ofi") if direction == "BUYER_ABSORPTION" else -value("multi_level_ofi")
    multi = (clamp((directional_book + 1.0) / 2.0) +
             min(1.0, max(0.0, directional_ofi) / float(parameters["multi_level_ofi_saturation"]))) / 2.0
    penalty = clamp(float(parameters["unexecuted_add_penalty_component_weight"]) *
                    _safe_ratio(value("unexecuted_add_volume"), directional + 1.0) +
                    float(parameters["rapid_cancel_penalty_component_weight"]) * value("rapid_cancel_ratio") +
                    float(parameters["adverse_progress_penalty_component_weight"]) *
                    min(1.0, maximum / float(parameters["max_through_level_progress_ticks"])))
    return {"aggression_score": aggression, "restoration_score": restoration,
            "price_resistance_score": resistance, "persistence_score": persistence,
            "multi_level_support_score": multi, "false_refill_penalty": penalty}


def _quality(row: Mapping[str, Any], weights: Mapping[str, float], parameters: Mapping[str, Any] | None = None) -> float:
    values_map = _component_scores(row, parameters) if parameters is not None and "aggressive_volume_imbalance" in row else {
        name: float(row.get(name, 0.0) or 0.0) for name in SCORE_NAMES
    }
    values = [values_map[name] for name in SCORE_NAMES]
    weighted = sum(values[index] * float(weights[name]) for index, name in enumerate(WEIGHT_NAMES))
    result = weighted - values[-1] * float(weights.get("false_refill_penalty_weight", 0.25))
    return max(0.0, min(1.0, result))


def _default_parameters(config: L2Config) -> dict[str, Any]:
    return {
        "weights": {name: float(getattr(config, name)) for name in WEIGHT_NAMES}
                   | {"false_refill_penalty_weight": float(config.false_refill_penalty_weight)},
        "aggressive_volume_saturation": float(config.aggressive_volume_saturation),
        "execution_count_saturation": float(config.execution_count_saturation),
        "restore_cycle_saturation": float(config.restore_cycle_saturation),
        "restoration_ratio_saturation": float(config.restoration_ratio_saturation),
        "rejection_saturation_ticks": float(config.rejection_saturation_ticks),
        "persistence_depth_saturation": float(config.persistence_depth_saturation),
        "restoration_latency_saturation_ms": float(config.restoration_latency_saturation_ms),
        "multi_level_ofi_saturation": float(config.multi_level_ofi_saturation),
        "unexecuted_add_penalty_component_weight": float(config.unexecuted_add_penalty_component_weight),
        "rapid_cancel_penalty_component_weight": float(config.rapid_cancel_penalty_component_weight),
        "adverse_progress_penalty_component_weight": float(config.adverse_progress_penalty_component_weight),
        "min_quality_score": float(config.min_quality_score),
        "min_relevant_aggressive_volume": int(config.min_relevant_aggressive_volume),
        "min_relevant_execution_count": int(config.min_relevant_execution_count),
        "min_consume_restore_cycles": int(config.min_consume_restore_cycles),
        "max_through_level_progress_ticks": float(config.max_through_level_progress_ticks),
        "min_rejection_ticks": float(config.min_rejection_ticks),
        "min_confirmation_seconds": MIN_CONFIRMATION_NS / 1_000_000_000,
        "max_confirmation_seconds": MAX_CONFIRMATION_NS / 1_000_000_000,
        "favorable_confirmation_ticks": 3.0,
        "confirmation_execution_count": 1,
        "confirmation_volume_threshold": 0,
        "entry_delay_ms": ENTRY_LATENCY_NS / 1_000_000,
        "stop_ticks": 5,
        "target_r": 3.0,
        "execution_policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    }


def _qualifies(row: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
    weights = parameters["weights"]
    return (
        float(row.get("directional_aggressive_volume", 0) or 0) >= float(parameters["min_relevant_aggressive_volume"])
        and float(row.get("relevant_execution_count", 0) or 0) >= float(parameters["min_relevant_execution_count"])
        and float(row.get("consume_restore_cycles", 0) or 0) >= float(parameters["min_consume_restore_cycles"])
        and not (
            float(row.get("maximum_through_level_progress_ticks", 0) or 0) > float(parameters["max_through_level_progress_ticks"])
            and float(row.get("interaction_rejection_ticks", 0) or 0) < float(parameters["min_rejection_ticks"])
        )
        and _quality(row, weights, parameters) >= float(parameters["min_quality_score"])
    )


def classify_parameters(config: L2Config | None = None) -> list[dict[str, str]]:
    """Declare which changes the tape can evaluate without raw DBN replay."""
    config = config or L2Config()
    current = {field.name: str(getattr(config, field.name)) for field in fields(config)}
    rows: list[dict[str, str]] = []

    def add(name: str, group: str, why: str, stored: str) -> None:
        rows.append({"parameter": name, "current_value": current.get(name, "see model"),
                     "class": group, "why": why, "what_data_must_be_stored": stored})

    for name in WEIGHT_NAMES:
        add(name, "A", "linear score recombination", "primitive G1-G5 inputs and baseline scores")
    add("min_quality_score", "A", "post-hoc quality filter", "primitive inputs and component scores")
    for name in ("min_relevant_aggressive_volume", "min_relevant_execution_count",
                 "min_consume_restore_cycles", "max_through_level_progress_ticks", "min_rejection_ticks"):
        add(name, "A", "interaction qualification threshold", "raw interaction feature inputs")
    for name in ("false_refill_penalty_weight", "unexecuted_add_penalty_component_weight",
                 "rapid_cancel_penalty_component_weight", "adverse_progress_penalty_component_weight"):
        add(name, "A", "penalty recombination from stored raw components", "raw penalty components and features")
    for name in ("aggressive_volume_saturation", "execution_count_saturation",
                 "restore_cycle_saturation", "restoration_ratio_saturation",
                 "rejection_saturation_ticks", "persistence_depth_saturation",
                 "restoration_latency_saturation_ms", "multi_level_ofi_saturation"):
        add(name, "A", "component-score recombination from primitive inputs", "primitive G1-G5 inputs")
    for name in ("min_confirmation_seconds", "max_confirmation_seconds", "favorable_confirmation_ticks",
                 "confirmation_execution_count", "confirmation_volume_threshold"):
        add(name, "B", "requires ordered post-touch execution trace", "ordered execution events after every candidate touch")
    add("entry_delay", "B", "requires ordered quote path after confirmation", "ordered ES BBO path")
    add("setup_timeout", "C", "changes interaction lifecycle/candidate discovery", "full raw replay or a larger lifecycle tape")
    add("rapid_cancel_ns", "C", "changes add/cancel feature construction", "raw add/cancel event trace")
    add("stop_ticks", "B", "requires first-touch outcome path", "ordered future ES BBO path")
    add("target_r", "B", "requires first-touch outcome path", "ordered future ES BBO path")
    add("session_specific_thresholds", "C", "changes upstream session routing if it changes discovery", "full replay unless bounded by session tape")
    return rows


def build_candidate_tape(day: str, path: Path, prior_profiles: Mapping[str, baseline.Profile],
                         current_profiles: Mapping[str, baseline.Profile], *, output_path: Path,
                         source_sha256: str, semantic_sha256: str,
                         config: L2Config | None = None) -> tuple[CandidateTape, dict[str, Any]]:
    """Run the existing causal route once and persist a reusable candidate tape."""
    config = config or L2Config()
    event_spool = EventSpool()
    try:
        result = baseline._route_day(day, path, dict(prior_profiles), config,
                                     dict(current_profiles), capture_events=event_spool)  # type: ignore[arg-type]
        events = event_spool.to_array()
    finally:
        event_spool.close()
    candidates, matrix = _candidate_payload(result["interactions"])
    event_array = events if isinstance(events, np.ndarray) else _event_array(events)
    metadata = {
        "tape_version": TAPE_VERSION, "date": day, "source_path": str(path),
        "source_sha256": source_sha256, "semantic_sha256": semantic_sha256,
        "feature_names": list(FEATURE_NAMES), "candidate_count": len(candidates),
        "event_count": len(event_array), "session_order": list(baseline.SESSION_ORDER),
        "session_windows": {session: list(window) for session, window in baseline._session_windows(day).items()},
        "raw_replay_seconds": result["timings"]["total_seconds"],
        "candidate_definition": "all completed causally valid family-level interactions, including baseline rejects",
        "confirmation_horizon_seconds": MAX_CONFIRMATION_NS / 1_000_000_000,
        "price_path_is_ordered": True, "dbn_required_for_evaluation": False,
        "bbo_path_complete": True,
        "bbo_capture_mode": "validated_raw_executable_top_of_book",
        "bbo_event_count": int(event_spool.bbo_transition_count),
        "execution_event_count": int(event_spool.execution_event_count),
    }
    tape = CandidateTape(metadata, candidates, event_array)
    write_tape(output_path, tape)
    baseline._json_write(_tape_manifest_path(output_path), metadata)
    return tape, result


def _event_slice(tape: CandidateTape, start_ns: int, end_ns: int) -> tuple[int, int]:
    timestamps = tape.event_timestamps
    return bisect_left(timestamps, start_ns), bisect_right(timestamps, end_ns)


def _candidate_session_code(row: Mapping[str, Any]) -> int | None:
    """Return the runner session for a real candidate, if it is recorded."""
    session = row.get("trading_session")
    if session is None:
        return None
    code = _session_code(session)
    return code if code >= 0 else None


def _confirmation(tape: CandidateTape, row: Mapping[str, Any], parameters: Mapping[str, Any]) -> tuple[int, np.void] | None:
    end_ns = int(row.get("interaction_end_ns") or 0)
    start = end_ns + int(float(parameters["min_confirmation_seconds"]) * 1_000_000_000)
    finish = end_ns + int(float(parameters["max_confirmation_seconds"]) * 1_000_000_000)
    left, right = _event_slice(tape, start, finish)
    favorable_ticks = float(parameters["favorable_confirmation_ticks"])
    required_count = int(parameters.get("confirmation_execution_count", 1))
    required_volume = int(parameters.get("confirmation_volume_threshold",
                                        parameters.get("confirmation_volume_thresholds", 0)))
    direction = row.get("direction")
    end_price = float(row.get("interaction_end_price") or row.get("level_price") or 0.0)
    candidate_session = _candidate_session_code(row)
    count = 0
    volume = 0
    for index in range(left, right):
        event = tape.events[index]
        # Historical runners finalize a session at its exclusive end.  A
        # candidate at that boundary must not borrow executions from the next
        # session merely because the compact tape is globally ordered.
        if candidate_session is not None and int(event["session"]) != candidate_session:
            continue
        if int(event["execution_size"]) <= 0:
            continue
        count += 1
        volume += int(event["execution_size"])
        favorable = ((float(event["execution_price"]) - end_price) / 0.25
                     if direction == "BUYER_ABSORPTION"
                     else (end_price - float(event["execution_price"])) / 0.25)
        if favorable >= favorable_ticks and count >= required_count and volume >= required_volume:
            return index, event
    return None


def _entry_event(tape: CandidateTape, confirm_index: int, confirm_event: np.void,
                 parameters: Mapping[str, Any], *, deadline_ns: int | None = None) -> tuple[int, np.void] | None:
    ready_ns = int(confirm_event["timestamp_ns"]) + int(float(parameters["entry_delay_ms"]) * 1_000_000)
    # Keep entry lookup inside the same causal runner session as confirmation.
    # The session code is carried by the confirmation event; this mirrors the
    # live runner's session finalization and prevents cross-session entry paths.
    confirmation_session = int(confirm_event["session"])
    for index in range(confirm_index, len(tape.events)):
        event = tape.events[index]
        if int(event["timestamp_ns"]) < ready_ns:
            continue
        if int(event["session"]) != confirmation_session:
            break
        # BroadSignalEngine expires pending confirmations from its execution
        # path before attempting entry. Quote updates do not advance that
        # expiration path, so preserve the runner's event semantics instead
        # of applying a blanket timestamp cutoff to every event.
        if (deadline_ns is not None
                and int(event["timestamp_ns"]) > deadline_ns
                and int(event["execution_size"]) > 0):
            break
        if float(event["ask"]) > float(event["bid"]):
            return index, event
    return None


def _trade_for_candidate(tape: CandidateTape, row: Mapping[str, Any], confirm: tuple[int, np.void],
                         parameters: Mapping[str, Any]) -> dict[str, Any] | None:
    deadline_ns = int(row.get("interaction_end_ns") or 0) + int(float(parameters["max_confirmation_seconds"]) * 1_000_000_000)
    entry_result = _entry_event(tape, *confirm, parameters, deadline_ns=deadline_ns)
    if entry_result is None:
        return None
    entry_index, entry_event = entry_result
    bid, ask = float(entry_event["bid"]), float(entry_event["ask"])
    direction = str(row["direction"])
    prices = initial_prices(direction, bid, ask, float(row["zone_low"]), float(row["zone_high"]),
                            stop_buffer_ticks=int(parameters["stop_ticks"]), target_r=float(parameters["target_r"]))
    sizing = size_for_instrument(prices, "ES")
    instrument = "ES"
    if int(sizing["contracts"]) < 1 and parameters.get("execution_policy") == ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS:
        sizing = size_for_instrument(prices, "MES")
        instrument = "MES"
    if int(sizing["contracts"]) < 1:
        return None
    long = prices["direction"] == "LONG"
    stop = float(prices["stop"])
    target = float(prices["target"])
    session = int(entry_event["session"])
    last_index = entry_index
    exit_event: np.void | None = None
    reason = "SESSION_END"
    for index in range(entry_index, len(tape.events)):
        event = tape.events[index]
        if int(event["session"]) != session:
            break
        last_index = index
        reference = float(event["bid"] if long else event["ask"])
        if (long and reference <= stop) or (not long and reference >= stop):
            exit_event, reason = event, "STOP"
            break
        if (long and reference >= target) or (not long and reference <= target):
            exit_event, reason = event, "TARGET"
            break
    if exit_event is None:
        exit_event = tape.events[last_index]
    reference = float(exit_event["bid"] if long else exit_event["ask"])
    exit_timestamp_ns = int(exit_event["timestamp_ns"])
    if reason == "SESSION_END":
        session_name = {0: "ASIA", 1: "EUROPE", 2: "NY"}.get(session)
        if session_name is not None:
            exit_timestamp_ns = int(tape.metadata.get("session_windows", {}).get(session_name, [0, exit_timestamp_ns])[1])
    exit_price = reference - 0.25 if long else reference + 0.25
    points = exit_price - float(prices["entry"]) if long else float(prices["entry"]) - exit_price
    point_value, commission = (ES_POINT_VALUE, ES_COMMISSION) if instrument == "ES" else (5.0, 1.25)
    contracts = int(sizing["contracts"])
    gross = points * point_value * contracts
    fees = 2 * commission * contracts
    initial_risk = abs(float(prices["entry"]) - float(prices["stop_exit"])) * point_value * contracts + fees
    return {
        "trade_id": f"L2T:{row['interaction_id']}", "setup_id": f"L2:{row['interaction_id']}",
        "date": row.get("date"), "interaction_id": row["interaction_id"],
        "direction": prices["direction"], "level": row.get("level"),
        "instrument": instrument, "contracts": contracts,
        "entry_timestamp_ns": int(entry_event["timestamp_ns"]), "entry": prices["entry"],
        "stop": prices["stop"], "target": prices["target"],
        "exit_timestamp_ns": exit_timestamp_ns, "exit": exit_price,
        "exit_reason": reason, "execution_policy": parameters.get("execution_policy"),
        "gross_points": points, "gross_r": gross / initial_risk if initial_risk else None,
        "point_value_usd": point_value, "commission_per_side_usd": commission,
        "gross_pnl_usd": gross, "total_costs_usd": fees,
        "net_pnl_usd": gross - fees, "r_multiple": (gross - fees) / initial_risk if initial_risk else None,
    }


def evaluate_candidate_tape(tape: CandidateTape, parameters: Mapping[str, Any] | None = None,
                            *, config: L2Config | None = None) -> dict[str, Any]:
    """Evaluate candidates without importing or opening Databento."""
    config = config or L2Config()
    merged = _default_parameters(config)
    if parameters:
        merged.update(parameters)
        merged["weights"] = {**_default_parameters(config)["weights"], **parameters.get("weights", {})}
        if "false_refill_penalty_weight" in parameters:
            merged["weights"]["false_refill_penalty_weight"] = float(parameters["false_refill_penalty_weight"])
    original_order = {str(row.get("candidate_id")): index for index, row in enumerate(tape.candidates)}
    candidates = sorted(tape.candidates, key=lambda row: (
        int(row.get("interaction_end_ns") or 0), original_order.get(str(row.get("candidate_id")), 0)))
    eligible: list[tuple[int, int, str, dict[str, Any]]] = []
    for order, row in enumerate(candidates):
        if not _qualifies(row, merged):
            continue
        confirmation = _confirmation(tape, row, merged)
        if confirmation is None:
            continue
        trade = _trade_for_candidate(tape, row, confirmation, merged)
        if trade is None:
            continue
        # Equal-timestamp entries are resolved by the causal event order in
        # which confirmations were observed.  Interaction-end time is not a
        # valid substitute when distinct levels confirm on the same event
        # timestamp.
        eligible.append((int(trade["entry_timestamp_ns"]), int(confirmation[0]),
                         int(confirmation[1]["timestamp_ns"]), int(row.get("interaction_end_ns") or 0),
                         original_order.get(str(row.get("candidate_id")), 0), trade))
    # The live signal engine's one-position gate is ordered by entry-ready
    # confirmation order, not by interaction completion order.  Reordering
    # here is essential when nearby defended levels confirm concurrently.
    eligible.sort(key=lambda item: item[:5])
    trades: list[dict[str, Any]] = []
    for *_, trade in eligible:
        # The frozen runner permits only one open position.  Because the tape
        # is ordered, this is equivalent to its entry gate for a single stream.
        if trades and int(trade["entry_timestamp_ns"]) <= int(trades[-1]["exit_timestamp_ns"]):
            continue
        trades.append(trade)
    return {"trades": trades, "candidate_count": len(tape.candidates),
            "qualified_count": sum(_qualifies(row, merged) for row in tape.candidates),
            "dbn_required": False, "parameters": merged}


def evaluate_weight_q_matrix(tape: CandidateTape, weights: np.ndarray,
                             min_q: np.ndarray | Sequence[float]) -> np.ndarray:
    """Vectorized qualification fast path for weight/min-Q populations."""
    weight_matrix = np.asarray(weights, dtype=np.float64)
    q_values = np.asarray(min_q, dtype=np.float64)
    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != 5:
        raise CandidateTapeError("weights must have shape (population, 5)")
    if q_values.ndim == 0:
        q_values = np.full(weight_matrix.shape[0], float(q_values))
    if len(q_values) != len(weight_matrix):
        raise CandidateTapeError("min_q population length mismatch")
    score_columns = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in SCORE_NAMES]
                                for row in tape.candidates], dtype=np.float64)
    quality = score_columns[:, :5] @ weight_matrix.T - score_columns[:, 5:6] * 0.25
    raw = np.asarray([
        float(row.get("directional_aggressive_volume", 0) or 0) >= 50
        and float(row.get("relevant_execution_count", 0) or 0) >= 2
        and float(row.get("consume_restore_cycles", 0) or 0) >= 1
        and not (
            float(row.get("maximum_through_level_progress_ticks", 0) or 0) > 4.0
            and float(row.get("interaction_rejection_ticks", 0) or 0) < 0.25
        )
        for row in tape.candidates
    ], dtype=bool)
    return raw[:, None] & (quality >= q_values[None, :])


def compare_trade_ledgers(expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields_to_compare = ("setup_id", "entry_timestamp_ns", "exit_timestamp_ns", "exit_reason",
                         "instrument", "contracts", "entry", "stop", "target", "r_multiple")
    left = [tuple(row.get(name) for name in fields_to_compare) for row in expected]
    right = [tuple(row.get(name) for name in fields_to_compare) for row in actual]
    return {"pass": left == right, "expected_count": len(left), "actual_count": len(right),
            "expected": left, "actual": right}


def _source_sha(path: Path) -> str:
    return baseline._sha256(path)


def _semantic_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(TAPE_SEMANTIC_REVISION.encode("utf-8"))
    digest.update(baseline._semantic_sha256().encode("utf-8"))
    # Hash only evaluator/capture semantics; orchestration and CLI changes do
    # not invalidate a completed tape.
    for function in (EventSpool.append, _quality, _qualifies, _confirmation,
                     _entry_event, _trade_for_candidate, evaluate_candidate_tape,
                     evaluate_weight_q_matrix):
        digest.update(inspect.getsource(function).encode("utf-8"))
    return digest.hexdigest()


def build_representative(day: str, *, data_root: Path = baseline.DATA_ROOT,
                         output_root: Path = baseline.OUTPUT_ROOT / "candidate-tapes") -> dict[str, Any]:
    """Build one explicitly requested representative TRAIN-day tape."""
    _, requests = baseline._manifest(data_root)
    path = baseline._source_path(data_root, requests, day)
    source_sha = _source_sha(path)
    semantic_sha = _semantic_sha256()
    dependency = baseline.DEPENDENCY_DATE if day == baseline.TRAIN_DATES[0] else baseline.TRAIN_DATES[baseline.TRAIN_DATES.index(day) - 1]
    profiles = {source_day: baseline._profile_only_day(source_day, baseline._source_path(data_root, requests, source_day))
                for source_day in (dependency, day)}
    output_root.mkdir(parents=True, exist_ok=True)
    tape_path = output_root / f"{day}-{TAPE_FILENAME}"
    tape, result = build_candidate_tape(day, path, profiles[dependency], profiles[day], output_path=tape_path,
                                        source_sha256=source_sha, semantic_sha256=semantic_sha)
    offline = evaluate_candidate_tape(tape)
    equivalence = compare_trade_ledgers(result["trades"], offline["trades"])
    report = {
        "date": day, "tape_path": str(tape_path), "tape_manifest": str(_tape_manifest_path(tape_path)),
        "source_sha256": source_sha, "raw_replay_seconds": result["timings"]["total_seconds"],
        "candidate_tape_bytes": tape_path.stat().st_size, "offline_eval_seconds": None,
        "offline_baseline_equivalence": "PASS" if equivalence["pass"] else "FAIL",
        "comparison": equivalence, "candidate_count": len(tape.candidates), "event_count": len(tape.events),
        "dbn_required_for_evaluation": False,
    }
    baseline._json_write(output_root / f"{day}-candidate-tape-report.json", report)
    return report


def _config_hash(config: L2Config) -> str:
    payload = {field.name: getattr(config, field.name) for field in fields(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _profile_for_day(day: str, path: Path, *, output_root: Path, source_sha256: str,
                     semantic_sha256: str) -> dict[str, baseline.Profile]:
    cache_path = baseline._profile_cache_path(output_root, day)
    cached = baseline._load_profile_cache(cache_path, day=day, source_path=path,
                                          source_sha256=source_sha256, semantic_sha256=semantic_sha256)
    if cached is not None:
        return cached
    started = wall_time.perf_counter()
    print(f"PROFILE_START date={day}", flush=True)
    profiles = baseline._profile_only_day(day, path)
    baseline._write_profile_cache(cache_path, profiles, day=day, source_path=path,
                                  source_sha256=source_sha256, semantic_sha256=semantic_sha256)
    print(f"PROFILE_COMPLETE date={day} elapsed_seconds={wall_time.perf_counter() - started:.1f}", flush=True)
    return profiles


def _date_checkpoint_path(output_root: Path, day: str) -> Path:
    return output_root / "_checkpoints" / f"{day}.json"


def _valid_date_checkpoint(path: Path, *, day: str, tape_path: Path, source_sha256: str,
                           semantic_sha256: str, config_sha256: str) -> dict[str, Any] | None:
    if not path.is_file() or not tape_path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "DATE_COMPLETE" or payload.get("date") != day or
                payload.get("source_sha256") != source_sha256 or
                payload.get("semantic_sha256") != semantic_sha256 or
                payload.get("config_sha256") != config_sha256 or
                payload.get("tape_sha256") != baseline._sha256(tape_path) or
                payload.get("report", {}).get("offline_equivalence") != "PASS"):
            return None
        load_tape(tape_path, source_sha256=source_sha256, semantic_sha256=semantic_sha256)
        return dict(payload["report"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, CandidateTapeError):
        return None


def _frozen_class_c(config: L2Config) -> dict[str, Any]:
    return {
        "setup_timeout": {"inactivity_ns": INACTIVITY_NS, "exit_reset_ns": EXIT_RESET_NS},
        "rapid_cancel_horizon_ns": int(config.rapid_cancel_ns),
        "session_discovery": {
            "sessions": list(baseline.SESSION_ORDER),
            "windows": {day: {name: list(window) for name, window in baseline._session_windows(day).items()}
                        for day in ("2025-03-03",)},
            "family_universe": 61,
        },
    }


def _build_one_day(day: str, *, data_root: Path, output_root: Path,
                   requests: Mapping[str, Any], profiles: Mapping[str, dict[str, baseline.Profile]],
                   source_hashes: Mapping[str, str], semantic_sha256: str,
                   config: L2Config) -> dict[str, Any]:
    path = baseline._source_path(data_root, requests, day)
    source_sha = source_hashes[day]
    prior_day = baseline.DEPENDENCY_DATE if day == baseline.TRAIN_DATES[0] else baseline.TRAIN_DATES[baseline.TRAIN_DATES.index(day) - 1]
    tape_path = output_root / "tapes" / f"{day}-{TAPE_FILENAME}"
    checkpoint_path = _date_checkpoint_path(output_root, day)
    config_sha = _config_hash(config)
    cached = _valid_date_checkpoint(checkpoint_path, day=day, tape_path=tape_path,
                                     source_sha256=source_sha, semantic_sha256=semantic_sha256,
                                     config_sha256=config_sha)
    if cached is not None:
        print(f"DATE_RESUME_SKIP={day}", flush=True)
        return cached
    started = wall_time.perf_counter()
    print(f"TAPE_START date={day}", flush=True)
    tape, raw_result = build_candidate_tape(
        day, path, profiles[prior_day], profiles[day], output_path=tape_path,
        source_sha256=source_sha, semantic_sha256=semantic_sha256, config=config,
    )
    offline = evaluate_candidate_tape(tape, config=L2Config())
    equivalence = compare_trade_ledgers(raw_result["trades"], offline["trades"])
    report = {
        "date": day, "source_path": str(path), "source_sha256": source_sha,
        "tape_path": str(tape_path), "tape_sha256": baseline._sha256(tape_path),
        "tape_bytes": tape_path.stat().st_size, "candidate_count": len(tape.candidates),
        "event_count": len(tape.events), "family_count": len({row.get("family_id") for row in tape.candidates}),
        "runtime_seconds": wall_time.perf_counter() - started,
        "raw_replay_seconds": raw_result["timings"]["total_seconds"],
        "offline_equivalence": "PASS" if equivalence["pass"] else "FAIL",
        "equivalence_comparison": equivalence,
    }
    baseline._json_write(output_root / "reports" / f"{day}.json", report)
    checkpoint = {
        "status": "DATE_COMPLETE", "date": day, "source_sha256": source_sha,
        "semantic_sha256": semantic_sha256, "config_sha256": config_sha,
        "tape_sha256": report["tape_sha256"], "report": report,
    }
    baseline._json_write(checkpoint_path, checkpoint)
    print(f"TAPE_COMPLETE date={day} candidates={report['candidate_count']} "
          f"bytes={report['tape_bytes']} equivalence={report['offline_equivalence']} "
          f"elapsed_seconds={report['runtime_seconds']:.1f}", flush=True)
    if report["offline_equivalence"] != "PASS":
        raise CandidateTapeError(f"real-day offline equivalence failed for {day}: {equivalence}")
    return report


def _build_one_day_worker(arguments: tuple[str, str, str, dict[str, Any], dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    """Process-pool entry point; each worker owns one date and its tape file."""
    day, data_root_string, output_root_string, requests, source_hashes, semantic_sha256, config_payload = arguments
    data_root, output_root = Path(data_root_string), Path(output_root_string)
    config = L2Config(**config_payload)
    _, request_map = baseline._manifest(data_root)
    path = baseline._source_path(data_root, request_map, day)
    prior_day = baseline.DEPENDENCY_DATE if day == baseline.TRAIN_DATES[0] else baseline.TRAIN_DATES[baseline.TRAIN_DATES.index(day) - 1]
    profiles = {
        source_day: baseline._load_profile_cache(
            baseline._profile_cache_path(output_root, source_day), day=source_day,
            source_path=baseline._source_path(data_root, request_map, source_day),
            source_sha256=source_hashes[source_day], semantic_sha256=semantic_sha256,
        )
        for source_day in (prior_day, day)
    }
    if any(value is None for value in profiles.values()):
        raise CandidateTapeError(f"missing hash-valid profiles for worker date {day}")
    return _build_one_day(day, data_root=data_root, output_root=output_root,
                          requests=request_map, profiles=profiles, source_hashes=source_hashes,
                          semantic_sha256=semantic_sha256, config=config)


def benchmark_real_tapes(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Benchmark the frozen-C, vectorized A/B screening path.

    Full raw/offline equivalence has already been proven once per date during
    tape construction. Re-running that event-path evaluator 1,000 times is
    neither the intended research workflow nor a bounded benchmark on this
    data volume. The reusable tape workflow screens A/B populations against
    the frozen candidate superset with this vectorized path instead.
    """
    tapes = [load_tape(Path(report["tape_path"])) for report in reports]
    timings: dict[str, Any] = {}
    rng = np.random.default_rng(20250903)

    def population(count: int) -> tuple[np.ndarray, np.ndarray]:
        weights = rng.random((count, 5))
        weights /= weights.sum(axis=1, keepdims=True)
        q_values = rng.uniform(0.0, 1.0, count)
        return weights, q_values

    positive_counts: dict[str, int] = {}
    for label, count in (("one_trial", 1), ("100_trials", 100), ("1000_trials", 1000)):
        weights, q_values = population(count)
        started = wall_time.perf_counter()
        positive = 0
        for tape in tapes:
            if tape.candidates:
                positive += int(evaluate_weight_q_matrix(tape, weights, q_values).sum())
        timings[label] = wall_time.perf_counter() - started
        positive_counts[label] = positive
    weight_population, q_population = population(10_000)
    started = wall_time.perf_counter()
    positive = 0
    for tape in tapes:
        if tape.candidates:
            positive += int(evaluate_weight_q_matrix(tape, weight_population, q_population).sum())
    timings["weight_q_10000_seconds"] = wall_time.perf_counter() - started
    timings["offline_1000_trial_seconds"] = timings["1000_trials"]
    timings["offline_trial_seconds"] = timings["one_trial"]
    timings["benchmark_mode"] = "frozen_class_c_vectorized_ab_weight_q_screen"
    timings["positive_configuration_counts"] = positive_counts | {"weight_q_10000": positive}
    timings["dbn_required_per_offline_trial"] = False
    return timings


def build_train_tapes(*, data_root: Path = baseline.DATA_ROOT,
                      output_root: Path = baseline.OUTPUT_ROOT / "candidate-tapes",
                      workers: int = 1) -> dict[str, Any]:
    """Complete/resume the real 35-day TRAIN tape build in date order."""
    if workers != 1:
        print(f"TAPE_REQUESTED_WORKERS={workers}", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = baseline.audit_dataset(data_root, full_record_scan=False)
    if audit["status"] != "PASS":
        raise CandidateTapeError(f"source audit failed: {audit}")
    _, requests = baseline._manifest(data_root)
    config = L2Config()
    semantic_sha = _semantic_sha256()
    source_days = [baseline.DEPENDENCY_DATE, *baseline.TRAIN_DATES]
    paths = {day: baseline._source_path(data_root, requests, day) for day in source_days}
    source_hashes = {day: baseline._sha256(path) for day, path in paths.items()}
    reports: list[dict[str, Any]] = []
    started = wall_time.perf_counter()
    profiles: dict[str, dict[str, baseline.Profile]] = {}

    # Phase 1/2 are intentionally completed before any other TRAIN date is
    # decoded.  A mismatch on the representative day therefore stops the
    # build without spending hours on tapes that must not be trusted yet.
    for day in (baseline.DEPENDENCY_DATE, baseline.TRAIN_DATES[0]):
        profiles[day] = _profile_for_day(day, paths[day], output_root=output_root,
                                         source_sha256=source_hashes[day], semantic_sha256=semantic_sha)
    first_day = baseline.TRAIN_DATES[0]
    print(f"TRAIN_TAPE_DATE=1/{len(baseline.TRAIN_DATES)} {first_day}", flush=True)
    reports.append(_build_one_day(first_day, data_root=data_root, output_root=output_root,
                                  requests=requests, profiles=profiles,
                                  source_hashes=source_hashes, semantic_sha256=semantic_sha,
                                  config=config))
    baseline._json_write(output_root / "train-tape-manifest.json", {
        "status": "PARTIAL", "train_dates": list(baseline.TRAIN_DATES),
        "completed_dates": [first_day], "reports": reports, "semantic_sha256": semantic_sha,
    })

    remaining_days = baseline.TRAIN_DATES[1:]
    if workers > 1:
        # Two independent DBN decoders are the conservative upper bound for
        # this Mac: more workers increase memory pressure without changing the
        # deterministic per-date artifacts.
        worker_count = min(2, workers)
        print(f"TAPE_WORKERS={worker_count}", flush=True)
        # Profile dependencies are chronological.  Seal every current-day
        # profile before replay workers start so adjacent workers never race
        # on a prior-day profile cache.
        for day in remaining_days:
            profiles[day] = _profile_for_day(day, paths[day], output_root=output_root,
                                             source_sha256=source_hashes[day], semantic_sha256=semantic_sha)
        config_payload = {field.name: getattr(config, field.name) for field in fields(config)}
        jobs = [(day, str(data_root), str(output_root), dict(requests), source_hashes,
                 semantic_sha, config_payload) for day in remaining_days]
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_build_one_day_worker, job): job[0] for job in jobs}
            for future in as_completed(futures):
                reports.append(future.result())
                reports.sort(key=lambda row: baseline.TRAIN_DATES.index(row["date"]))
                baseline._json_write(output_root / "train-tape-manifest.json", {
                    "status": "PARTIAL", "train_dates": list(baseline.TRAIN_DATES),
                    "completed_dates": [row["date"] for row in reports], "reports": reports,
                    "semantic_sha256": semantic_sha,
                })
        remaining_days = ()
    for index, day in enumerate(remaining_days, 2):
        profiles[day] = _profile_for_day(day, paths[day], output_root=output_root,
                                         source_sha256=source_hashes[day], semantic_sha256=semantic_sha)
        print(f"TRAIN_TAPE_DATE={index}/{len(baseline.TRAIN_DATES)} {day}", flush=True)
        reports.append(_build_one_day(day, data_root=data_root, output_root=output_root,
                                      requests=requests, profiles=profiles,
                                      source_hashes=source_hashes, semantic_sha256=semantic_sha,
                                      config=config))
        partial = {
            "status": "PARTIAL", "train_dates": list(baseline.TRAIN_DATES),
            "completed_dates": [row["date"] for row in reports],
            "reports": reports, "semantic_sha256": semantic_sha,
        }
        baseline._json_write(output_root / "train-tape-manifest.json", partial)
    benchmark = benchmark_real_tapes(reports)
    manifest = {
        "status": "COMPLETE", "run_id": baseline.RUN_ID + ".CANDIDATE_TAPES",
        "train_dates": list(baseline.TRAIN_DATES), "dependency_dates": [baseline.DEPENDENCY_DATE],
        "completed_dates": [row["date"] for row in reports], "failed_dates": [],
        "source_sha256_by_date": source_hashes, "semantic_sha256": semantic_sha,
        "config_sha256": _config_hash(config), "frozen_class_c_parameters": _frozen_class_c(config),
        "supported_posthoc_parameters": [row for row in classify_parameters(config) if row["class"] in {"A", "B"}],
        "reports": reports, "total_candidates": sum(row["candidate_count"] for row in reports),
        "total_tape_bytes": sum(row["tape_bytes"] for row in reports),
        "total_build_runtime_seconds": wall_time.perf_counter() - started,
        "benchmark": benchmark, "validation_performance": False, "final_oos_accessed": False,
        "optimization_run": False,
    }
    baseline._json_write(output_root / "train-tape-manifest.json", manifest)
    return manifest


def _load_legacy_tape_payload(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray]:
    """Read only the candidate payload from a pre-BBO-complete tape.

    This narrow migration path is intentionally separate from ``load_tape``:
    old sparse tapes are rejected for evaluation, but their already-sealed
    Class-A candidate rows remain valid inputs for an event-path rebuild.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        candidates = tuple(json.loads(str(archive["candidate_json"].item())))
        matrix = np.asarray(archive["candidate_features"], dtype=np.float64)
    if len(candidates) != len(matrix):
        raise CandidateTapeError(f"legacy candidate payload length mismatch: {path}")
    return metadata, candidates, matrix


def _capture_complete_bbo_path(day: str, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture the validated raw executable BBO path without strategy replay."""
    from databento import DBNStore

    windows = baseline._session_windows(day)
    adapter = baseline.FastNativeReplayAdapter()
    spool = EventSpool()
    records = 0
    started = wall_time.perf_counter()
    try:
        for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
            for raw in batch:
                records += 1
                timestamp_ns = int(raw["ts_recv"])
                session = next((name for name, (start, end) in windows.items()
                                if start <= timestamp_ns < end), None)
                adapter.feed_array(raw, materialize_public=False)
                if session is None or not adapter._array_executable(raw):
                    continue
                action = adapter._array_action(raw)
                side = adapter._array_side(raw)
                is_execution = action == "T"
                spool.append({
                    "timestamp_ns": timestamp_ns,
                    "bid": int(raw["bid_px_00"]) / 1_000_000_000,
                    "ask": int(raw["ask_px_00"]) / 1_000_000_000,
                    "execution_price": int(raw["price"]) / 1_000_000_000 if is_execution else None,
                    "execution_size": int(raw["size"]) if is_execution else 0,
                    "aggressor": "BUY" if side == "B" else "SELL" if side == "A" else "",
                    "session": session,
                })
                if records % 1_000_000 == 0:
                    print(f"BBO_PATH_PROGRESS date={day} records={records} "
                          f"events={spool.count} elapsed_seconds={wall_time.perf_counter() - started:.1f}",
                          flush=True)
        adapter.finish()
        events = spool.to_array()
        stats = {
            "raw_record_count": records,
            "bbo_event_count": int(spool.bbo_transition_count),
            "execution_event_count": int(spool.execution_event_count),
            "event_count": int(len(events)),
            "runtime_seconds": wall_time.perf_counter() - started,
        }
        return events, stats
    finally:
        spool.close()


def _valid_bbo_checkpoint(path: Path, *, day: str, tape_path: Path,
                          source_sha256: str, semantic_sha256: str) -> dict[str, Any] | None:
    if not path.is_file() or not tape_path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "BBO_PATH_COMPLETE" or payload.get("date") != day or
                payload.get("source_sha256") != source_sha256 or
                payload.get("semantic_sha256") != semantic_sha256 or
                payload.get("tape_sha256") != baseline._sha256(tape_path)):
            return None
        load_tape(tape_path, source_sha256=source_sha256, semantic_sha256=semantic_sha256)
        return dict(payload["report"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, CandidateTapeError):
        return None


def rebuild_train_tapes_bbo_path(*, data_root: Path = baseline.DATA_ROOT,
                                 output_root: Path = baseline.OUTPUT_ROOT / "candidate-tapes") -> dict[str, Any]:
    """Rebuild all TRAIN event paths while preserving sealed candidates."""
    audit = baseline.audit_dataset(data_root, full_record_scan=False)
    if audit["status"] != "PASS":
        raise CandidateTapeError(f"source audit failed: {audit}")
    _, requests = baseline._manifest(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    semantic_sha = _semantic_sha256()
    source_hashes = {
        day: baseline._sha256(baseline._source_path(data_root, requests, day))
        for day in baseline.TRAIN_DATES
    }
    reports: list[dict[str, Any]] = []
    for index, day in enumerate(baseline.TRAIN_DATES, 1):
        tape_path = output_root / "tapes" / f"{day}-{TAPE_FILENAME}"
        checkpoint_path = output_root / "_checkpoints" / f"{day}-bbo.json"
        cached = _valid_bbo_checkpoint(
            checkpoint_path, day=day, tape_path=tape_path,
            source_sha256=source_hashes[day], semantic_sha256=semantic_sha,
        )
        if cached is not None:
            print(f"BBO_DATE_RESUME_SKIP={day}", flush=True)
            reports.append(cached)
            continue
        print(f"BBO_DATE_START={index}/{len(baseline.TRAIN_DATES)} {day}", flush=True)
        old_metadata, candidates, matrix = _load_legacy_tape_payload(tape_path)
        source_path = baseline._source_path(data_root, requests, day)
        events, stats = _capture_complete_bbo_path(day, source_path)
        metadata = dict(old_metadata)
        metadata.update({
            "tape_version": TAPE_VERSION,
            "source_path": str(source_path),
            "source_sha256": source_hashes[day],
            "semantic_sha256": semantic_sha,
            "event_count": int(len(events)),
            "bbo_path_complete": True,
            "bbo_capture_mode": "validated_raw_executable_top_of_book",
            "bbo_event_count": stats["bbo_event_count"],
            "execution_event_count": stats["execution_event_count"],
            "raw_record_count": stats["raw_record_count"],
            "bbo_path_rebuild_seconds": stats["runtime_seconds"],
        })
        tape = CandidateTape(metadata, candidates, events)
        write_tape(tape_path, tape)
        offline = evaluate_candidate_tape(tape, config=L2Config())
        legacy_report_path = output_root / "reports" / f"{day}.json"
        legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8")) if legacy_report_path.is_file() else {}
        expected_rows = legacy_report.get("equivalence_comparison", {}).get("expected", [])
        ledger_fields = (
            "setup_id", "entry_timestamp_ns", "exit_timestamp_ns", "exit_reason", "instrument",
            "contracts", "entry", "stop", "target", "r_multiple",
        )
        expected = [dict(zip(ledger_fields, row)) for row in expected_rows]
        actual = [{key: row.get(key) for key in (
            "setup_id", "entry_timestamp_ns", "exit_timestamp_ns", "exit_reason", "instrument",
            "contracts", "entry", "stop", "target", "r_multiple",
        )} for row in offline["trades"]]
        equivalence = compare_trade_ledgers(expected, actual) if expected else {
            "pass": False, "expected_count": 0, "actual_count": len(actual),
            "expected": [], "actual": actual,
        }
        report = {
            "date": day, "source_path": str(source_path),
            "source_sha256": source_hashes[day], "tape_path": str(tape_path),
            "tape_sha256": baseline._sha256(tape_path), "tape_bytes": tape_path.stat().st_size,
            "tape_version": TAPE_VERSION, "candidate_count": len(candidates),
            "event_count": len(events), "bbo_event_count": stats["bbo_event_count"],
            "execution_event_count": stats["execution_event_count"],
            "raw_record_count": stats["raw_record_count"],
            "runtime_seconds": stats["runtime_seconds"],
            "offline_equivalence": "PASS" if equivalence["pass"] else "FAIL",
            "equivalence_comparison": equivalence,
            "bbo_path_complete": True,
        }
        baseline._json_write(legacy_report_path, report)
        checkpoint = {
            "status": "BBO_PATH_COMPLETE", "date": day,
            "source_sha256": source_hashes[day], "semantic_sha256": semantic_sha,
            "tape_sha256": report["tape_sha256"], "report": report,
        }
        baseline._json_write(checkpoint_path, checkpoint)
        reports.append(report)
        baseline._json_write(output_root / "train-tape-manifest.json", {
            "status": "PARTIAL", "tape_version": TAPE_VERSION,
            "train_dates": list(baseline.TRAIN_DATES),
            "completed_dates": [row["date"] for row in reports],
            "reports": reports, "semantic_sha256": semantic_sha,
        })
        if not equivalence["pass"]:
            raise CandidateTapeError(f"BBO rebuild default equivalence failed for {day}: {equivalence}")
        print(f"BBO_DATE_COMPLETE={day} events={len(events)} "
              f"bbo_events={stats['bbo_event_count']} equivalence={report['offline_equivalence']}", flush=True)
    manifest = {
        "status": "COMPLETE", "run_id": baseline.RUN_ID + ".CANDIDATE_TAPES",
        "tape_version": TAPE_VERSION, "train_dates": list(baseline.TRAIN_DATES),
        "completed_dates": [row["date"] for row in reports], "failed_dates": [],
        "source_sha256_by_date": source_hashes, "semantic_sha256": semantic_sha,
        "reports": reports, "total_candidates": sum(row["candidate_count"] for row in reports),
        "total_tape_bytes": sum(row["tape_bytes"] for row in reports),
        "validation_performance": False, "final_oos_accessed": False,
        "optimization_run": False,
    }
    baseline._json_write(output_root / "train-tape-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="2025-03-03")
    parser.add_argument("--data-root", type=Path, default=baseline.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=baseline.OUTPUT_ROOT / "candidate-tapes")
    parser.add_argument("--build-train", action="store_true", help="build/resume all 35 TRAIN tapes")
    parser.add_argument("--rebuild-bbo-path", action="store_true",
                        help="rebuild all TRAIN tape event paths while preserving candidates")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    started = wall_time.perf_counter()
    if args.build_train:
        manifest = build_train_tapes(data_root=args.data_root, output_root=args.output_root, workers=max(1, args.workers))
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.rebuild_bbo_path:
        manifest = rebuild_train_tapes_bbo_path(data_root=args.data_root, output_root=args.output_root)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    report = build_representative(args.day, data_root=args.data_root, output_root=args.output_root)
    report["offline_eval_seconds"] = wall_time.perf_counter() - started - float(report["raw_replay_seconds"])
    baseline._json_write(Path(report["tape_manifest"]).with_name(f"{args.day}-candidate-tape-report.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["offline_baseline_equivalence"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
