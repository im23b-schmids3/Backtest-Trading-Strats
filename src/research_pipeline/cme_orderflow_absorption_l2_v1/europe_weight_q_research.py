"""Causal Europe-session Weight x Q retrospective research.

Stage A is the only source replay. It feeds the seven frozen structural-level
engines in one pass and seals completed interactions, confirmation indexes and
a compact causal execution tape. Stage B opens only those Parquet artifacts;
every configuration receives independent pending-setup and position state.

This is retrospective research on already-seen data, never fresh OOS evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import causal_master_tape as master
from . import europe_w04_replay as europe
from . import europe_w04_structural_matrix as structural
from . import weight_q_research as common

EVIDENCE_LABEL = "EUROPE_WEIGHT_Q_RETROSPECTIVE_RESEARCH_NOT_OOS"
SOURCE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_EUROPE_STRUCTURAL_MATRIX")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_EUROPE_WEIGHT_Q_RESEARCH")
CACHE_DIRECTORY = "causal-cache-v2"
CACHE_VERSION = 2
FAMILIES = (
    "PRIOR_EUROPE_POC", "PRIOR_EUROPE_HIGH", "PRIOR_EUROPE_LOW",
    "PRIOR_EUROPE_VAH", "PRIOR_EUROPE_VAL",
    "CURRENT_EUROPE_HIGH_SWEEP", "CURRENT_EUROPE_LOW_SWEEP",
)
SOURCE_BY_FAMILY = {
    "PRIOR_EUROPE_POC": "PRIOR_EUROPE_SESSION_POC",
    "PRIOR_EUROPE_HIGH": "PRIOR_EUROPE_SESSION_HIGH",
    "PRIOR_EUROPE_LOW": "PRIOR_EUROPE_SESSION_LOW",
    "PRIOR_EUROPE_VAH": "PRIOR_EUROPE_SESSION_VAH",
    "PRIOR_EUROPE_VAL": "PRIOR_EUROPE_SESSION_VAL",
    "CURRENT_EUROPE_HIGH_SWEEP": "CURRENT_EUROPE_HIGH_SWEEP",
    "CURRENT_EUROPE_LOW_SWEEP": "CURRENT_EUROPE_LOW_SWEEP",
}
FAMILY_BY_SOURCE = {value: key for key, value in SOURCE_BY_FAMILY.items()}
QUALITY_THRESHOLDS = tuple(Decimal(value) for value in ("0.45", "0.50", "0.55", "0.60", "0.65"))
EXPECTED_WEIGHT_COUNT = 3_876
EXPECTED_LEVEL_CONFIGURATION_COUNT = 19_380
EXPECTED_TOTAL_COUNT = 135_660
EXPECTED_ELIGIBLE_SESSIONS = 46
W04_UNITS = (4, 2, 6, 4, 4)
MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
EXECUTION_MODELS = ("ES_NATIVE_SOURCE", "MES_PROXY_FROM_ES")
DIRECTIONS = ("LONG", "SHORT")
NO_AUTOMATIC_SELECTION = True

# The prompt calls bucket C's neighbor floor "not catastrophically negative".
# This literal is frozen before the corrected matrix is evaluated and remains
# a descriptive reporting threshold only.
ROBUSTNESS_C_NEIGHBOR_WORST_FLOOR_R = -10.0


class EuropeWeightQError(RuntimeError):
    """The sealed causal/cache/reconciliation contract was not preserved."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EuropeWeightQError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["config_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class CsvSink:
    """Append stable per-family tables into one staging CSV."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.part = path.with_suffix(path.suffix + ".part")
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None
        self.fields: list[str] | None = None

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.part.open("w", newline="", encoding="utf-8")
            self.fields = list(rows[0])
            self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
            self.writer.writeheader()
        assert self.writer is not None and self.fields is not None
        if any(set(row) != set(self.fields) for row in rows):
            raise EuropeWeightQError(f"unstable CSV schema: {self.path.name}")
        self.writer.writerows(rows)

    def close(self) -> None:
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.part.write_text("config_id\n", encoding="utf-8")
        else:
            self.handle.close()
        self.part.replace(self.path)


def generate_weight_grid() -> tuple[tuple[int, int, int, int, int], ...]:
    rows = tuple(
        (g1, g2, g3, g4, 20 - g1 - g2 - g3 - g4)
        for g1 in range(1, 17)
        for g2 in range(1, 18 - g1)
        for g3 in range(1, 19 - g1 - g2)
        for g4 in range(1, 20 - g1 - g2 - g3)
    )
    if len(rows) != EXPECTED_WEIGHT_COUNT or len(set(rows)) != EXPECTED_WEIGHT_COUNT:
        raise EuropeWeightQError("weight grid does not contain exactly 3,876 unique vectors")
    if any(sum(row) != 20 or min(row) < 1 for row in rows):
        raise EuropeWeightQError("weight grid violates the positive 0.05 simplex")
    return rows


def config_id(family: str, units: Sequence[int], threshold: Decimal | str | float) -> str:
    q = Decimal(str(threshold)).quantize(Decimal("0.00"))
    if q not in QUALITY_THRESHOLDS:
        raise EuropeWeightQError(f"unsupported Europe Q: {threshold}")
    return f"{family}|W{'-'.join(f'{int(value):02d}' for value in units)}|Q{int(q * 100):02d}"


def configuration_registry() -> tuple[tuple[str, tuple[int, int, int, int, int], Decimal], ...]:
    rows = tuple(
        (family, units, threshold)
        for family in FAMILIES for units in generate_weight_grid() for threshold in QUALITY_THRESHOLDS
    )
    if len(rows) != EXPECTED_TOTAL_COUNT or len({config_id(*row) for row in rows}) != EXPECTED_TOTAL_COUNT:
        raise EuropeWeightQError("Europe family x Weight x Q registry cardinality mismatch")
    return rows


def neighbors(
    units: Sequence[int], threshold: Decimal,
) -> tuple[tuple[tuple[int, int, int, int, int], Decimal], ...]:
    source = tuple(int(value) for value in units)
    try:
        threshold_index = QUALITY_THRESHOLDS.index(threshold)
    except ValueError as exc:
        raise EuropeWeightQError(f"unsupported Europe Q: {threshold}") from exc
    output = {(candidate, threshold) for candidate in common.weight_neighbors(source)}
    output.update((source, QUALITY_THRESHOLDS[index]) for index in (threshold_index - 1, threshold_index + 1) if 0 <= index < len(QUALITY_THRESHOLDS))
    return tuple(sorted(output))


def accepted(row: Mapping[str, Any], units: Sequence[int], threshold: Decimal) -> bool:
    if str(row.get("non_quality_rejection_reasons") or ""):
        return False
    weights = common.unit_weights(units)
    positive = sum(Decimal(str(row[name])) * weights[name] for name in master.SCORE_FIELDS)
    penalty = Decimal(str(row["false_refill_penalty"])) * Decimal(
        str(europe.W04_CONFIG.false_refill_penalty_weight)
    )
    score = max(Decimal("0"), min(Decimal("1"), positive - penalty))
    return score >= threshold


def _accepted_mask(
    rows: Sequence[Mapping[str, Any]], primitive: np.ndarray,
    approximate_scores: np.ndarray, units: Sequence[int], threshold: Decimal,
) -> np.ndarray:
    """Vectorize the grid and resolve floating threshold boundaries exactly."""
    mask = primitive & (approximate_scores >= float(threshold))
    for index in np.flatnonzero(np.abs(approximate_scores - float(threshold)) <= 1e-12):
        mask[int(index)] = accepted(rows[int(index)], units, threshold)
    return mask


def semantic_hashes() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    dependencies = (
        root / "europe_w04_replay.py", root / "europe_w04_structural_matrix.py",
        root / "causal_master_tape.py", root / "weight_q_research.py",
    )
    return {
        "europe_session_contract_sha256": sha256({
            "timezone": europe.EUROPE_TIMEZONE,
            "signal_window": "08:00:00_INCLUSIVE_TO_16:30:00_EXCLUSIVE",
            "position_cutoff": "16:55:00",
        }),
        "feature_semantics_sha256": sha256({
            "source": inspect.getsource(structural.interaction_row),
            "score_fields": master.SCORE_FIELDS,
            "false_refill_penalty_weight": str(europe.W04_CONFIG.false_refill_penalty_weight),
        }),
        "structural_level_semantics_sha256": sha256(structural.semantic_diff_document()),
        "execution_semantics_sha256": sha256({
            "tape": inspect.getsource(europe.EuropeProxySessionCausalTape),
            "simulation": inspect.getsource(common.simulate_independent_session),
        }),
        "dependency_file_sha256": {path.name: _file_sha256(path) for path in dependencies},
    }


def make_cache(rows: Iterable[Mapping[str, Any]], *, source_sessions: Sequence[str]) -> dict[str, Any]:
    """Small synthetic cache used only by unit tests; no execution outcomes."""
    payload = {
        "cache_version": CACHE_VERSION, "evidence_label": EVIDENCE_LABEL,
        "source_sessions": list(source_sessions), "semantic_hashes": semantic_hashes(),
        "rows": [dict(row) for row in rows],
    }
    payload["cache_sha256"] = sha256(payload)
    return payload


def validate_cache(cache: Mapping[str, Any]) -> None:
    body = {key: value for key, value in cache.items() if key != "cache_sha256"}
    if (
        cache.get("cache_version") != CACHE_VERSION
        or cache.get("evidence_label") != EVIDENCE_LABEL
        or cache.get("cache_sha256") != sha256(body)
        or cache.get("semantic_hashes") != semantic_hashes()
    ):
        raise EuropeWeightQError("cache version/hash/semantic dependency rejection")


def _source_reference(source_root: Path) -> dict[str, dict[str, str]]:
    return {str(row["level_family"]): row for row in _read_csv(source_root / "level-matrix.csv")}


def _baseline_row(
    sessions: Sequence[Mapping[str, Any]], family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_family = SOURCE_BY_FAMILY[family]
    setups, trades = structural._setup_trade_rows(sessions, source_family)
    return structural._funnel_row(sessions, source_family, setups, trades), trades


def reconcile_w04_baseline(
    sessions: Sequence[Mapping[str, Any]], *, source_root: Path,
) -> dict[str, Any]:
    reference = _source_reference(source_root)
    integer_fields = (
        "raw_interactions", "accepted_setups", "confirmation_passed", "confirmation_failed",
        "active_position_blocked", "trades", "wins", "losses", "es_native_trades",
        "mes_proxy_trades", "unresolved",
    )
    float_fields = ("total_r", "net_pnl_usd", "max_cumulative_drawdown_r", "max_cumulative_drawdown_usd")
    families: dict[str, Any] = {}
    for family in FAMILIES:
        source_family = SOURCE_BY_FAMILY[family]
        actual, trades = _baseline_row(sessions, family)
        expected = reference[source_family]
        mismatches: dict[str, Any] = {}
        for field_name in integer_fields:
            if int(actual[field_name]) != int(expected[field_name]):
                mismatches[field_name] = {"expected": expected[field_name], "actual": actual[field_name]}
        for field_name in float_fields:
            if not math.isclose(float(actual[field_name]), float(expected[field_name]), rel_tol=0.0, abs_tol=1e-9):
                mismatches[field_name] = {"expected": expected[field_name], "actual": actual[field_name]}
        if mismatches:
            raise EuropeWeightQError(f"W04 baseline mismatch {family}: {mismatches}")
        families[family] = {
            "status": "PASS", "source_level_family": source_family,
            "metrics": {name: actual[name] for name in (*integer_fields, *float_fields)},
            "trade_id_sha256": sha256(sorted(str(row["trade_id"]) for row in trades)),
        }
    return {
        "status": "PASS", "evidence_label": EVIDENCE_LABEL,
        "weights": ["0.20", "0.10", "0.30", "0.20", "0.20"],
        "quality_threshold": "0.45", "families": families,
    }


def _cache_descriptor(cache_root: Path, relative: str) -> dict[str, Any]:
    path = cache_root / relative
    if not path.is_file():
        raise EuropeWeightQError(f"cache artifact missing: {relative}")
    descriptor: dict[str, Any] = {
        "relative_path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size, "sha256": _file_sha256(path),
    }
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        try:
            descriptor["rows"] = parquet.metadata.num_rows
        finally:
            master._close_parquet(parquet)
    return descriptor


def _load_established_interactions(source_root: Path) -> list[dict[str, Any]]:
    """Load the already-published causal G rows; no source market file is opened."""
    output: list[dict[str, Any]] = []
    numeric = (
        "interaction_start_ns", "interaction_end_ns", "level_price", "interaction_end_price",
        "zone_low", "zone_high", *master.SCORE_FIELDS, "false_refill_penalty",
    )
    for source in _read_csv(source_root / "interaction-features.csv"):
        source_family = str(source["level_family"])
        if source_family not in FAMILY_BY_SOURCE:
            raise EuropeWeightQError(f"unexpected established Europe level: {source_family}")
        row: dict[str, Any] = {
            "research_family": FAMILY_BY_SOURCE[source_family],
            "level_family": source_family,
            "interaction_id": str(source["interaction_id"]),
            "source_interaction_id": str(source["source_interaction_id"]),
            "session_date": str(source["session_date"]),
            "direction": str(source["direction"]), "level": str(source["level"]),
            "termination": str(source["termination"]),
            "non_quality_rejection_reasons": str(source.get("non_quality_rejection_reasons") or ""),
        }
        for name in numeric:
            row[name] = int(source[name]) if name.endswith("_ns") else float(source[name])
        output.append(row)
    output.sort(key=lambda row: (
        str(row["session_date"]), str(row["research_family"]),
        int(row["interaction_end_ns"]), str(row["interaction_id"]),
    ))
    identities = [(str(row["research_family"]), str(row["interaction_id"])) for row in output]
    if len(identities) != len(set(identities)):
        raise EuropeWeightQError("established Europe interaction identities are not unique per family")
    return output


class _FeatureTapeDayState:
    """Re-index established interactions against one causal Europe source day."""

    def __init__(self, spec: europe.SessionSpec, rows: Sequence[Mapping[str, Any]], staging: Path) -> None:
        self.spec = spec
        self.adapter = structural.historical.HistoricalMBOToMBP10Adapter()
        self.rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            self.rows_by_family[str(row["research_family"])].append(dict(row))
        for values in self.rows_by_family.values():
            values.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["interaction_id"])))
        self.cursors = {family: 0 for family in FAMILIES}
        self.trackers = {family: master.CausalWindowTracker() for family in FAMILIES}
        self.latest_quote: tuple[float, float] | None = None
        self.latest_quote_ns: int | None = None
        self.prior_quote: tuple[float, float] | None = None
        self.source_index = 0
        self.ordinal = 0
        self.reached_cutoff = False
        self.closed = False
        self.writer = master.AtomicParquetStream(staging / "causal-event-tape" / f"{spec.day}.parquet")

    def _register_completed(self, timestamp_ns: int) -> None:
        for family in FAMILIES:
            values = self.rows_by_family.get(family, [])
            cursor = self.cursors[family]
            while cursor < len(values) and int(values[cursor]["interaction_end_ns"]) <= timestamp_ns:
                self.trackers[family].register(values[cursor])
                cursor += 1
            self.cursors[family] = cursor

    def _append(
        self, *, timestamp_ns: int, event_type: str,
        execution: Any | None = None, due: Mapping[str, Sequence[str]] | None = None,
        book_state: str = "EXECUTABLE", hard_flat_reason: str | None = None,
    ) -> None:
        due = due or {}
        build_spec = master.SessionBuildSpec(
            self.spec.day, str(self.spec.source_path), "MES_PROXY_FROM_ES", 0.0,
            self.spec.start_ns, self.spec.cutoff_ns, europe.EUROPE_HARD_FLAT_REASON,
            str(self.spec.staging_root), "EUROPE_LONDON",
        )
        self.writer.append(master._event_row(
            spec=build_spec, ordinal=self.ordinal, timestamp_ns=timestamp_ns,
            stream="CALENDAR" if event_type == "HARD_FLAT" else "ES",
            source_index=0 if event_type == "HARD_FLAT" else self.source_index,
            event_type=event_type, es_quote=self.latest_quote, mes_quote=None,
            execution=execution, book_state=book_state,
            entry_probe_count=sum(len(items) for items in due.values()),
            es_quote_timestamp_ns=self.latest_quote_ns, mes_quote_timestamp_ns=None,
            hard_flat_reason=hard_flat_reason,
        ))
        for family, identifiers in due.items():
            self.trackers[family].bind_entry_probe(identifiers, self.ordinal)
        self.ordinal += 1

    def observe(self, record: Any) -> None:
        if self.closed:
            raise EuropeWeightQError(f"event routed to closed feature tape: {self.spec.day}")
        self.source_index += 1
        if record.timestamp_ns >= self.spec.cutoff_ns:
            self.reached_cutoff = True
            return
        previous_state = self.adapter.state
        public = self.adapter.feed(record, materialize_public=True)
        if public is None:
            if (
                record.timestamp_ns >= self.spec.start_ns
                and self.adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}
                and (previous_state != self.adapter.state or self.latest_quote is not None)
            ):
                self.latest_quote = self.prior_quote = None
                self.latest_quote_ns = None
                self._append(
                    timestamp_ns=record.timestamp_ns, event_type="BOOK_NON_EXECUTABLE",
                    book_state=self.adapter.state,
                )
            return
        quote = structural.historical._quote(public.snapshot)
        if quote is None:
            raise EuropeWeightQError(f"feature tape received non-executable BBO: {self.spec.day}")
        self.latest_quote, self.latest_quote_ns = quote, public.timestamp_ns
        if public.timestamp_ns < self.spec.start_ns:
            return
        self._register_completed(public.timestamp_ns)
        if public.execution is not None:
            for tracker in self.trackers.values():
                tracker.observe_es_execution(public.execution)
        due = {
            family: identifiers for family, tracker in self.trackers.items()
            if (identifiers := tracker.due_entry_probes(public.timestamp_ns))
        }
        if quote != self.prior_quote or previous_state != "EXECUTABLE" or public.execution is not None or due:
            self._append(
                timestamp_ns=public.timestamp_ns,
                event_type="ES_EXECUTION" if public.execution is not None else "ES_BBO",
                execution=public.execution, due=due,
            )
            self.prior_quote = quote

    def finish(self, staging: Path) -> dict[str, Any]:
        if not self.reached_cutoff:
            raise EuropeWeightQError(f"source did not reach Europe cutoff: {self.spec.day}")
        self.adapter.finish()
        self._register_completed(self.spec.cutoff_ns)
        if any(self.cursors[family] != len(self.rows_by_family.get(family, [])) for family in FAMILIES):
            raise EuropeWeightQError(f"interaction completes outside sealed Europe day: {self.spec.day}")
        quote, quote_ns = master.liquidation_window_quote(
            cutoff_ns=self.spec.cutoff_ns, quote=self.latest_quote,
            quote_timestamp_ns=self.latest_quote_ns,
        )
        if quote is None or quote_ns is None:
            raise EuropeWeightQError(f"feature tape lacks pre-cutoff liquidation BBO: {self.spec.day}")
        self.latest_quote, self.latest_quote_ns = quote, quote_ns
        self._append(
            timestamp_ns=self.spec.cutoff_ns, event_type="HARD_FLAT",
            hard_flat_reason=europe.EUROPE_HARD_FLAT_REASON,
        )
        event = self.writer.close()
        indexes: list[dict[str, Any]] = []
        for family, tracker in self.trackers.items():
            indexes.extend({**row, "research_family": family} for row in tracker.index_rows(
                day=self.spec.day, first_event=0, last_event=self.ordinal - 1,
                cutoff_ns=self.spec.cutoff_ns,
            ))
        indexes.sort(key=lambda row: (str(row["research_family"]), str(row["interaction_id"])))
        index_path = staging / "session-index" / f"{self.spec.day}.parquet"
        index = master._write_small_parquet(index_path, indexes)
        checkpoint = {
            "session_date": self.spec.day, "event": event, "index": index,
            "interaction_count": sum(len(values) for values in self.rows_by_family.values()),
        }
        _write_json(staging / "feature-checkpoints" / f"{self.spec.day}.json", checkpoint)
        self.closed = True
        return checkpoint

    def abort(self) -> None:
        if not self.closed:
            self.writer.abort()


def _feature_tape_worker(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Process one sealed source partition once; safe for Windows spawn."""
    staging = Path(str(task["staging"]))
    path = Path(str(task["path"]))
    sessions = [dict(item) for item in task["sessions"]]
    rows_by_day = {str(day): list(rows) for day, rows in task["rows_by_day"].items()}
    states: dict[str, _FeatureTapeDayState] = {}
    for item in sessions:
        bounds = europe.europe_session_bounds(str(item["day"]))
        spec = europe.SessionSpec(
            str(item["day"]), str(item["period"]), str(item["source_model"]),
            str(item["prior_day"]) if item.get("prior_day") else None, True,
            bounds["start_ns"], bounds["signal_end_ns"], bounds["cutoff_ns"], path, staging,
        )
        states[spec.day] = _FeatureTapeDayState(spec, rows_by_day.get(spec.day, []), staging)
    completed: list[dict[str, Any]] = []
    try:
        for record in structural.historical._stream_private_mbo(path):
            day = europe._date_from_ns(record.timestamp_ns)
            state = states.get(day)
            if state is None:
                continue
            state.observe(record)
            if state.reached_cutoff:
                completed.append(state.finish(staging))
                states.pop(day)
                if not states:
                    break
        if states:
            raise EuropeWeightQError(f"source partition did not complete eligible days: {sorted(states)}")
        return completed
    except BaseException:
        for state in states.values():
            state.abort()
        raise


def _build_feature_tapes(
    *, staging: Path, bindings: Sequence[europe.SourceBinding],
    sessions: Sequence[europe.AuditSession], interactions: Sequence[Mapping[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    eligible = {item.day: item for item in sessions if item.eligible}
    rows_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        rows_by_day[str(row["session_date"])].append(dict(row))
    completed: list[dict[str, Any]] = []
    completed_days: set[str] = set()
    for day in sorted(eligible):
        checkpoint_path = staging / "feature-checkpoints" / f"{day}.json"
        if not checkpoint_path.is_file():
            continue
        checkpoint = _read_json(checkpoint_path)
        if (
            checkpoint.get("session_date") != day
            or int(checkpoint.get("interaction_count", -1)) != len(rows_by_day.get(day, []))
        ):
            raise EuropeWeightQError(f"feature checkpoint identity mismatch: {day}")
        for name in ("event", "index"):
            descriptor = checkpoint.get(name)
            if not isinstance(descriptor, dict):
                raise EuropeWeightQError(f"feature checkpoint descriptor missing: {day}/{name}")
            path = Path(str(descriptor.get("path")))
            if (
                not path.is_file() or path.stat().st_size != int(descriptor.get("bytes", -1))
                or _file_sha256(path) != str(descriptor.get("sha256"))
            ):
                raise EuropeWeightQError(f"feature checkpoint artifact mismatch: {day}/{name}")
        completed.append(checkpoint)
        completed_days.add(day)
    tasks: list[dict[str, Any]] = []
    for binding in bindings:
        days = [day for day in binding.days if day in eligible and day not in completed_days]
        if not days:
            continue
        tasks.append({
            "staging": str(staging), "path": str(binding.path),
            "sessions": [{
                "day": day, "period": eligible[day].period,
                "source_model": eligible[day].source_model, "prior_day": eligible[day].prior_day,
            } for day in days],
            "rows_by_day": {day: rows_by_day.get(day, []) for day in days},
        })
    if completed_days:
        print(
            f"EUROPE_WEIGHT_Q_STAGE_A_RESUME verified_sessions={len(completed_days):02d}/{len(eligible):02d}",
            flush=True,
        )
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_feature_tape_worker, task): tuple(item["day"] for item in task["sessions"]) for task in tasks}
        for future in as_completed(futures):
            days = futures[future]
            rows = future.result()
            completed.extend(rows)
            print(
                f"EUROPE_WEIGHT_Q_STAGE_A_TAPE completed={','.join(days)} "
                f"sessions={len(completed):02d}/{len(eligible):02d}", flush=True,
            )
    completed.sort(key=lambda row: str(row["session_date"]))
    if [str(row["session_date"]) for row in completed] != sorted(eligible):
        raise EuropeWeightQError("feature-tape completion does not match 46 eligible sessions")
    return completed


def build_causal_cache(
    *, repository_root: Path, output_root: Path, workers: int = 4,
) -> dict[str, Any]:
    """Run the one permitted source pass and atomically seal the causal cache."""
    started = time.monotonic()
    repository_root = repository_root.resolve()
    output_root = (output_root if output_root.is_absolute() else repository_root / output_root).resolve()
    cache_root = output_root / CACHE_DIRECTORY
    staging = output_root / f"{CACHE_DIRECTORY}.building"
    if cache_root.exists():
        manifest = validate_causal_cache(cache_root)
        return {**manifest, "reused": True}
    if staging.exists() and not (staging / "frozen-contract.json").is_file():
        raise EuropeWeightQError(f"unrecognized Stage-A staging root: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    audit_root = staging / "coverage-audit"
    if not (audit_root / "summary.json").is_file():
        europe.build_europe_coverage_audit(repository_root, audit_root)
    sessions = europe.load_audit_sessions(audit_root)
    eligible_days = [item.day for item in sessions if item.eligible]
    if len(eligible_days) != EXPECTED_ELIGIBLE_SESSIONS:
        raise EuropeWeightQError("Stage A requires exactly 46 eligible Europe sessions")
    bindings = europe.source_bindings(repository_root, sessions)
    source_verification = europe.verify_source_bindings(bindings)
    dependencies = semantic_hashes()
    contract = {
        "cache_version": CACHE_VERSION, "evidence_label": EVIDENCE_LABEL,
        "eligible_sessions": eligible_days, "source_sessions": [item.day for item in sessions],
        "level_families": list(FAMILIES), "source_level_families": list(structural.LEVEL_FAMILIES),
        "semantic_hashes": dependencies, "stage_b_source_reads": 0, "network_calls": 0, "downloads": 0,
    }
    contract["contract_sha256"] = sha256(contract)
    contract_path = staging / "frozen-contract.json"
    if contract_path.is_file() and _read_json(contract_path) != contract:
        raise EuropeWeightQError("Stage-A staging contract differs")
    _write_json(contract_path, contract)

    interactions = _load_established_interactions(repository_root / SOURCE_ROOT)
    if not {str(row["session_date"]) for row in interactions}.issubset(set(eligible_days)):
        raise EuropeWeightQError("established interactions escape the sealed 46-session population")
    checkpoints = _build_feature_tapes(
        staging=staging, bindings=bindings, sessions=sessions,
        interactions=interactions, workers=workers,
    )
    indexes = [
        row for checkpoint in checkpoints
        for row in master._read_parquet_rows(Path(str(checkpoint["index"]["path"])))
    ]
    indexes.sort(key=lambda row: (
        str(row["research_family"]), str(row["session_date"]), str(row["interaction_id"]),
    ))
    interaction_ids = [(str(row["research_family"]), str(row["interaction_id"])) for row in interactions]
    index_ids = [(str(row["research_family"]), str(row["interaction_id"])) for row in indexes]
    if len(interaction_ids) != len(set(interaction_ids)) or set(interaction_ids) != set(index_ids):
        raise EuropeWeightQError("Stage-A interaction/index identities do not reconcile")
    master._write_small_parquet(staging / "europe-interaction-cache.parquet", interactions)
    master._write_small_parquet(staging / "interaction-event-index.parquet", indexes)
    baseline = reconcile_w04_cached(staging, source_root=repository_root / SOURCE_ROOT)
    _write_json(staging / "baseline-reconciliation.json", baseline)

    artifacts = {
        "interactions": _cache_descriptor(staging, "europe-interaction-cache.parquet"),
        "indexes": _cache_descriptor(staging, "interaction-event-index.parquet"),
        "baseline": _cache_descriptor(staging, "baseline-reconciliation.json"),
        "event_tapes": {day: _cache_descriptor(staging, f"causal-event-tape/{day}.parquet") for day in eligible_days},
    }
    manifest = {
        "status": "CAUSAL_CACHE_COMPLETE_BASELINE_RECONCILED",
        "cache_version": CACHE_VERSION, "evidence_label": EVIDENCE_LABEL,
        "eligible_session_count": len(eligible_days), "eligible_sessions": eligible_days,
        "interaction_count": len(interactions), "index_count": len(indexes),
        "source_population_sha256": sha256({"sessions": eligible_days, "sources": source_verification}),
        "semantic_hashes": dependencies, "contract_sha256": contract["contract_sha256"],
        "source_verification": source_verification, "artifacts": artifacts,
        "baseline_reconciliation": "PASS", "stage_a_duration_seconds": time.monotonic() - started,
        "raw_market_values_serialized_to_reports": False, "network_calls": 0, "downloads": 0,
    }
    manifest["cache_sha256"] = sha256(manifest)
    _write_json(staging / "cache-manifest.json", manifest)
    for transient in (
        staging / "_checkpoints", staging / "_work", staging / "coverage-audit",
        staging / "feature-checkpoints", staging / "session-index",
    ):
        if transient.exists():
            shutil.rmtree(transient)
    os.rename(staging, cache_root)
    return {**manifest, "cache_root": str(cache_root), "reused": False}


def validate_causal_cache(cache_root: Path) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    manifest = _read_json(cache_root / "cache-manifest.json")
    claimed = manifest.get("cache_sha256")
    body = {key: value for key, value in manifest.items() if key != "cache_sha256"}
    if (
        manifest.get("cache_version") != CACHE_VERSION
        or manifest.get("evidence_label") != EVIDENCE_LABEL
        or claimed != sha256(body)
        or manifest.get("semantic_hashes") != semantic_hashes()
        or manifest.get("eligible_session_count") != EXPECTED_ELIGIBLE_SESSIONS
        or manifest.get("baseline_reconciliation") != "PASS"
    ):
        raise EuropeWeightQError("causal cache manifest/hash/semantics rejection")
    descriptors = [
        manifest["artifacts"]["interactions"], manifest["artifacts"]["indexes"],
        manifest["artifacts"]["baseline"], *manifest["artifacts"]["event_tapes"].values(),
    ]
    for descriptor in descriptors:
        path = cache_root / str(descriptor["relative_path"])
        if (
            not path.is_file() or path.stat().st_size != int(descriptor["bytes"])
            or _file_sha256(path) != str(descriptor["sha256"])
        ):
            raise EuropeWeightQError(f"causal cache artifact mismatch: {descriptor['relative_path']}")
    return manifest


@dataclass
class Segment:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    net_pnl_usd: float = 0.0
    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    equity_r: float = 0.0
    peak_r: float = 0.0
    max_drawdown_r: float = 0.0
    equity_usd: float = 0.0
    peak_usd: float = 0.0
    max_drawdown_usd: float = 0.0

    def add(self, trade: Mapping[str, Any]) -> None:
        r_value = float(trade["r_multiple"] or 0.0)
        net = float(trade["net_pnl_usd"])
        self.trades += 1
        self.wins += net > 0
        self.losses += net <= 0
        self.total_r += r_value
        self.net_pnl_usd += net
        self.gross_profit_usd += max(net, 0.0)
        self.gross_loss_usd += max(-net, 0.0)
        self.equity_r += r_value
        self.peak_r = max(self.peak_r, self.equity_r)
        self.max_drawdown_r = min(self.max_drawdown_r, self.equity_r - self.peak_r)
        self.equity_usd += net
        self.peak_usd = max(self.peak_usd, self.equity_usd)
        self.max_drawdown_usd = min(self.max_drawdown_usd, self.equity_usd - self.peak_usd)

    def row(self) -> dict[str, Any]:
        return {
            "trades": self.trades, "wins": self.wins, "losses": self.losses,
            "win_rate": self.wins / self.trades if self.trades else 0.0,
            "total_r": self.total_r, "average_r": self.total_r / self.trades if self.trades else 0.0,
            "net_pnl_usd": self.net_pnl_usd,
            "profit_factor": self.gross_profit_usd / self.gross_loss_usd if self.gross_loss_usd else None,
            "max_drawdown_r": self.max_drawdown_r, "max_drawdown_usd": self.max_drawdown_usd,
        }


@dataclass
class ResearchAccumulator:
    units: tuple[int, int, int, int, int]
    threshold: Decimal
    accepted_setups: int = 0
    confirmations: int = 0
    confirmation_expiries: int = 0
    active_position_blocks: int = 0
    unresolved: int = 0
    other_terminal: dict[str, int] = field(default_factory=dict)
    aggregate: Segment = field(default_factory=Segment)
    months: dict[str, Segment] = field(default_factory=dict)
    directions: dict[str, Segment] = field(default_factory=dict)
    executions: dict[str, Segment] = field(default_factory=dict)
    r_values: list[float] = field(default_factory=list)

    def add(self, day: str, session: common.SessionResult) -> None:
        self.accepted_setups += session.accepted_setups
        self.confirmations += session.confirmations
        self.confirmation_expiries += session.confirmation_expiries
        self.active_position_blocks += session.active_position_blocks
        self.unresolved += session.unresolved
        for reason, count in session.other_terminal.items():
            self.other_terminal[reason] = self.other_terminal.get(reason, 0) + count
        for trade in session.trades:
            self.aggregate.add(trade)
            self.months.setdefault(day[:7], Segment()).add(trade)
            self.directions.setdefault(str(trade["direction"]), Segment()).add(trade)
            execution = str(trade.get("execution_model") or trade.get("instrument"))
            self.executions.setdefault(execution, Segment()).add(trade)
            self.r_values.append(float(trade["r_multiple"] or 0.0))

    def result_row(self, family: str, *, raw_interactions: int, primitive_eligible: int) -> dict[str, Any]:
        terminal_count = (
            self.confirmation_expiries + self.active_position_blocks + self.aggregate.trades
            + self.unresolved + sum(self.other_terminal.values())
        )
        if terminal_count != self.accepted_setups:
            raise EuropeWeightQError(
                f"configuration terminal reconciliation failed: {terminal_count} != {self.accepted_setups}"
            )
        month_values = [self.months.get(month, Segment()).total_r for month in MONTHS]
        month_trades = [self.months.get(month, Segment()).trades for month in MONTHS]
        return {
            "evidence_label": EVIDENCE_LABEL, "not_fresh_oos_evidence": True,
            "level": family, "config_id": config_id(family, self.units, self.threshold),
            **{f"G{index}_weight": value / 20 for index, value in enumerate(self.units, 1)},
            "quality_threshold": float(self.threshold),
            "raw_interactions": raw_interactions, "primitive_eligible_interactions": primitive_eligible,
            "quality_accepted": self.accepted_setups, "confirmations_passed": self.confirmations,
            "confirmations_failed": self.confirmation_expiries, "blocked_setups": self.active_position_blocks,
            **self.aggregate.row(),
            "median_r": statistics.median(self.r_values) if self.r_values else None,
            "es_native_count": self.executions.get("ES_NATIVE_SOURCE", Segment()).trades,
            "mes_proxy_count": self.executions.get("MES_PROXY_FROM_ES", Segment()).trades,
            "unresolved": self.unresolved, "other_terminal_count": sum(self.other_terminal.values()),
            "other_terminal_outcomes": json.dumps(self.other_terminal, sort_keys=True, separators=(",", ":")),
            "positive_month_count": sum(value > 0 for value in month_values),
            "worst_month_r": min(month_values), "best_month_r": max(month_values),
            "minimum_monthly_trade_count": min(month_trades), "LOW_SAMPLE": self.aggregate.trades < 15,
            "sample_band": sample_band(self.aggregate.trades),
        }


def sample_band(trades: int) -> str:
    if trades < 10:
        return "LT_10"
    if trades < 15:
        return "10_TO_14"
    if trades < 25:
        return "15_TO_24"
    if trades < 40:
        return "25_TO_39"
    return "40_PLUS"


def _segment_rows(family: str, accumulator: ResearchAccumulator) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identifier = config_id(family, accumulator.units, accumulator.threshold)
    base = {"evidence_label": EVIDENCE_LABEL, "level": family, "config_id": identifier}
    months = [{**base, "month": month, **accumulator.months.get(month, Segment()).row()} for month in MONTHS]
    directions = [{**base, "direction": direction, **accumulator.directions.get(direction, Segment()).row()} for direction in DIRECTIONS]
    executions = [{**base, "execution_model": model, **accumulator.executions.get(model, Segment()).row()} for model in EXECUTION_MODELS]
    return months, directions, executions


def _load_family_cache(cache_root: Path, family: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    interactions = [row for row in master._read_parquet_rows(cache_root / "europe-interaction-cache.parquet") if str(row["research_family"]) == family]
    indexes = {
        str(row["interaction_id"]): row
        for row in master._read_parquet_rows(cache_root / "interaction-event-index.parquet")
        if str(row["research_family"]) == family
    }
    if len(interactions) != len(indexes) or {str(row["interaction_id"]) for row in interactions} != set(indexes):
        raise EuropeWeightQError(f"family interaction/index cache mismatch: {family}")
    return interactions, indexes


def _run_one_configuration(
    *, cache_root: Path, family: str, eligible_days: Sequence[str],
    units: tuple[int, int, int, int, int], threshold: Decimal,
) -> dict[str, Any]:
    interactions, indexes = _load_family_cache(cache_root, family)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    accumulator = ResearchAccumulator(units, threshold)
    for day in eligible_days:
        rows = sorted(
            by_day.get(day, []),
            key=lambda row: (int(row["interaction_end_ns"]), str(row["interaction_id"])),
        )
        tape = europe.EuropeProxySessionCausalTape.from_parquet(
            day, cache_root / "causal-event-tape" / f"{day}.parquet",
        )
        selected = [row for row in rows if accepted(row, units, threshold)]
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        simulation = common.simulate_independent_session(tape, selected, day_indexes)
        europe._classify_europe_entry_cutoff(simulation, tape=tape)
        accumulator.add(day, simulation)
    return accumulator.result_row(
        family, raw_interactions=len(interactions),
        primitive_eligible=sum(not str(row.get("non_quality_rejection_reasons") or "") for row in interactions),
    )


def reconcile_w04_cached(cache_root: Path, *, source_root: Path) -> dict[str, Any]:
    """Gate Stage B on exact published W04 metrics from the new causal cache."""
    reference = _source_reference(source_root)
    eligible_days = [str(row["session_date"]) for row in _read_csv(source_root / "eligible-sessions.csv")]
    if len(eligible_days) != EXPECTED_ELIGIBLE_SESSIONS or len(set(eligible_days)) != len(eligible_days):
        raise EuropeWeightQError("published W04 eligible-session chronology is not exactly 46 unique dates")
    mapping = {
        "raw_interactions": "raw_interactions", "accepted_setups": "quality_accepted",
        "confirmation_passed": "confirmations_passed", "confirmation_failed": "confirmations_failed",
        "active_position_blocked": "blocked_setups", "trades": "trades", "wins": "wins",
        "losses": "losses", "total_r": "total_r", "net_pnl_usd": "net_pnl_usd",
        "max_cumulative_drawdown_r": "max_drawdown_r",
        "max_cumulative_drawdown_usd": "max_drawdown_usd",
        "es_native_trades": "es_native_count", "mes_proxy_trades": "mes_proxy_count",
        "unresolved": "unresolved",
    }
    integer = {
        "raw_interactions", "accepted_setups", "confirmation_passed", "confirmation_failed",
        "active_position_blocked", "trades", "wins", "losses", "es_native_trades",
        "mes_proxy_trades", "unresolved",
    }
    families: dict[str, Any] = {}
    for family in FAMILIES:
        actual = _run_one_configuration(
            cache_root=cache_root, family=family, eligible_days=eligible_days,
            units=W04_UNITS, threshold=Decimal(".45"),
        )
        expected = reference[SOURCE_BY_FAMILY[family]]
        mismatches: dict[str, Any] = {}
        for expected_name, actual_name in mapping.items():
            if expected_name in integer:
                match = int(expected[expected_name]) == int(actual[actual_name])
            else:
                match = math.isclose(
                    float(expected[expected_name]), float(actual[actual_name]),
                    rel_tol=0.0, abs_tol=1e-9,
                )
            if not match:
                mismatches[expected_name] = {
                    "expected": expected[expected_name], "actual": actual[actual_name],
                }
        if mismatches:
            raise EuropeWeightQError(f"cached W04 baseline mismatch {family}: {mismatches}")
        families[family] = {"status": "PASS", "metrics": actual}
    return {
        "status": "PASS", "evidence_label": EVIDENCE_LABEL,
        "weights": ["0.20", "0.10", "0.30", "0.20", "0.20"],
        "quality_threshold": "0.45", "families": families,
    }


def _simulate_selection_groups(
    *, tape: europe.EuropeProxySessionCausalTape,
    rows: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[int, ...], Sequence[int]],
) -> list[tuple[Sequence[int], common.SessionResult]]:
    """Run the unchanged simulator once for each distinct accepted setup set."""
    simulations: list[tuple[Sequence[int], common.SessionResult]] = []
    for selected_indexes, accumulator_indexes in groups.items():
        selected = [rows[index] for index in selected_indexes]
        simulation = common.simulate_independent_session(tape, selected, indexes)
        europe._classify_europe_entry_cutoff(simulation, tape=tape)
        simulations.append((accumulator_indexes, simulation))
    return simulations


def _run_family(
    *, cache_root: Path, family: str, eligible_days: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one family from sealed Parquet only; no source reader exists here."""
    interactions, indexes = _load_family_cache(cache_root, family)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    grid = generate_weight_grid()
    weights_matrix = np.asarray(grid, dtype=np.float64) / 20.0
    registry = tuple((units, threshold) for units in grid for threshold in QUALITY_THRESHOLDS)
    accumulators = [ResearchAccumulator(units, threshold) for units, threshold in registry]
    started = time.monotonic()
    for session_number, day in enumerate(eligible_days, 1):
        rows = sorted(by_day.get(day, []), key=lambda row: (int(row["interaction_end_ns"]), str(row["interaction_id"])))
        tape = europe.EuropeProxySessionCausalTape.from_parquet(day, cache_root / "causal-event-tape" / f"{day}.parquet")
        components = np.asarray([[float(row[name]) for name in master.SCORE_FIELDS] for row in rows], dtype=np.float64).reshape((-1, 5))
        penalties = np.asarray([float(row["false_refill_penalty"]) for row in rows], dtype=np.float64)
        primitive = np.asarray([not str(row.get("non_quality_rejection_reasons") or "") for row in rows], dtype=np.bool_)
        scores = np.clip(components @ weights_matrix.T - penalties[:, None] * float(europe.W04_CONFIG.false_refill_penalty_weight), 0.0, 1.0)
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        selection_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for weight_index, units in enumerate(grid):
            base = weight_index * len(QUALITY_THRESHOLDS)
            for q_index, threshold in enumerate(QUALITY_THRESHOLDS):
                mask = _accepted_mask(rows, primitive, scores[:, weight_index], units, threshold)
                selection_groups[tuple(int(index) for index in np.flatnonzero(mask))].append(base + q_index)
        for accumulator_indexes, simulation in _simulate_selection_groups(
            tape=tape, rows=rows, indexes=day_indexes, groups=selection_groups,
        ):
            for accumulator_index in accumulator_indexes:
                accumulators[accumulator_index].add(day, simulation)
        print(
            f"EUROPE_WEIGHT_Q_STAGE_B level={family} session={session_number:02d}/{len(eligible_days):02d} "
            f"{day} unique_selection_sets={len(selection_groups):05d} "
            f"configuration_cells={len(registry):05d} elapsed={time.monotonic()-started:.1f}s", flush=True,
        )
    raw = len(interactions)
    primitive_count = sum(not str(row.get("non_quality_rejection_reasons") or "") for row in interactions)
    results: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for accumulator in accumulators:
        results.append(accumulator.result_row(family, raw_interactions=raw, primitive_eligible=primitive_count))
        month_rows, direction_rows, execution_rows = _segment_rows(family, accumulator)
        monthly.extend(month_rows)
        directions.extend(direction_rows)
        executions.extend(execution_rows)
    return results, monthly, directions, executions


def neighbor_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["config_id"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        family = str(row["level"])
        units = tuple(int(round(float(row[f"G{index}_weight"]) * 20)) for index in range(1, 6))
        threshold = Decimal(str(row["quality_threshold"]))
        identifiers = [config_id(family, item, q) for item, q in neighbors(units, threshold)]
        if any(identifier not in by_id for identifier in identifiers):
            raise EuropeWeightQError(f"neighbor missing from complete family registry: {row['config_id']}")
        adjacent = [by_id[identifier] for identifier in identifiers]
        values = [float(item["total_r"]) for item in adjacent]
        output.append({
            "evidence_label": EVIDENCE_LABEL, "level": family, "config_id": row["config_id"],
            "neighbor_count": len(adjacent), "neighbor_median_r": statistics.median(values),
            "neighbor_worst_r": min(values), "neighbor_mean_r": statistics.fmean(values),
            "profitable_neighbor_fraction": sum(value > 0 for value in values) / len(values),
            "positive_majority_month_neighbor_fraction": sum(int(item["positive_month_count"]) >= 3 for item in adjacent) / len(adjacent),
        })
    return output


def plateau_membership(
    rows: Sequence[Mapping[str, Any]], neighbor_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["config_id"]): row for row in rows}
    neighbor_by_id = {str(row["config_id"]): row for row in neighbor_rows}
    eligible = {
        str(row["config_id"]) for row in rows
        if int(row["trades"]) >= 15 and float(row["total_r"]) > 0
        and float(neighbor_by_id[str(row["config_id"])]["profitable_neighbor_fraction"]) >= 0.5
    }
    components: list[list[str]] = []
    remaining = set(eligible)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = [seed]
        component: list[str] = []
        while queue:
            identifier = queue.pop()
            component.append(identifier)
            row = by_id[identifier]
            units = tuple(int(round(float(row[f"G{index}_weight"]) * 20)) for index in range(1, 6))
            threshold = Decimal(str(row["quality_threshold"]))
            for other in (config_id(str(row["level"]), item, q) for item, q in neighbors(units, threshold)):
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        components.append(sorted(component))
    components.sort(key=lambda item: (-len(item), item[0]))
    memberships: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, component in enumerate(components, 1):
        values = [by_id[item] for item in component]
        summaries.append({
            "level": rows[0]["level"] if rows else None, "plateau_id": f"P{index:04d}",
            "size": len(component), "median_total_r": statistics.median(float(item["total_r"]) for item in values),
            "worst_total_r": min(float(item["total_r"]) for item in values),
            "best_total_r": max(float(item["total_r"]) for item in values),
        })
        memberships.extend({
            "evidence_label": EVIDENCE_LABEL, "level": row["level"], "config_id": row["config_id"],
            "plateau_id": f"P{index:04d}", "plateau_size": len(component),
        } for row in values)
    return memberships, summaries


def descriptive_buckets(
    rows: Sequence[Mapping[str, Any]], neighbor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    robust = {str(row["config_id"]): row for row in neighbor_rows}
    a = [dict(row) for row in rows if int(row["trades"]) >= 20 and float(row["total_r"]) > 0 and int(row["positive_month_count"]) >= 3]
    b = [dict(row) for row in rows if int(row["trades"]) >= 25 and float(row["total_r"]) > 0 and int(row["positive_month_count"]) >= 3 and float(robust[str(row["config_id"])]["neighbor_median_r"]) > 0]
    c = [dict(row) for row in rows if int(row["trades"]) >= 30 and float(row["total_r"]) > 0 and int(row["positive_month_count"]) >= 3 and float(row["worst_month_r"]) > -3.0 and float(robust[str(row["config_id"])]["neighbor_median_r"]) > 0 and float(robust[str(row["config_id"])]["neighbor_worst_r"]) >= ROBUSTNESS_C_NEIGHBOR_WORST_FLOOR_R]
    return {"A": a, "B": b, "C": c}


def _level_summary(
    family: str, rows: Sequence[Mapping[str, Any]], neighbors_rows: Sequence[Mapping[str, Any]],
    plateau_summaries: Sequence[Mapping[str, Any]], buckets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    robust = {str(row["config_id"]): row for row in neighbors_rows}
    ranked = sorted(rows, key=lambda row: (-float(row["total_r"]), str(row["config_id"])))
    robust_ranked = sorted(rows, key=lambda row: (-float(robust[str(row["config_id"])]["neighbor_median_r"]), -float(row["total_r"]), str(row["config_id"])))
    profitable = [row for row in rows if float(row["total_r"]) > 0]
    return {
        "evidence_label": EVIDENCE_LABEL, "level": family, "configuration_count": len(rows),
        "profitable_configuration_count": len(profitable),
        "low_sample_configuration_count": sum(bool(row["LOW_SAMPLE"]) for row in rows),
        "top_aggregate_config_id": ranked[0]["config_id"], "top_aggregate_total_r": ranked[0]["total_r"],
        "leading_robust_config_id": robust_ranked[0]["config_id"],
        "leading_neighbor_median_r": robust[str(robust_ranked[0]["config_id"])]["neighbor_median_r"],
        "leading_neighbor_worst_r": robust[str(robust_ranked[0]["config_id"])]["neighbor_worst_r"],
        "plateau_count": len(plateau_summaries),
        "broadest_plateau_size": max((int(item["size"]) for item in plateau_summaries), default=0),
        "robustness_a_count": len(buckets["A"]), "robustness_b_count": len(buckets["B"]),
        "robustness_c_count": len(buckets["C"]),
        **{f"profitable_median_G{index}_weight": statistics.median(float(row[f"G{index}_weight"]) for row in profitable) if profitable else None for index in range(1, 6)},
    }


def _q_summary(
    family: str, rows: Sequence[Mapping[str, Any]], plateau_memberships: Sequence[Mapping[str, Any]],
    buckets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    size_by_config = {str(row["config_id"]): int(row["plateau_size"]) for row in plateau_memberships}
    bucket_ids = {name: {str(row["config_id"]) for row in values} for name, values in buckets.items()}
    output: list[dict[str, Any]] = []
    for threshold in QUALITY_THRESHOLDS:
        part = [row for row in rows if Decimal(str(row["quality_threshold"])) == threshold]
        output.append({
            "evidence_label": EVIDENCE_LABEL, "level": family, "quality_threshold": float(threshold),
            "profitable_weight_count": sum(float(row["total_r"]) > 0 for row in part),
            "median_trades": statistics.median(int(row["trades"]) for row in part),
            "median_total_r": statistics.median(float(row["total_r"]) for row in part),
            "best_total_r": max(float(row["total_r"]) for row in part),
            "median_max_drawdown_r": statistics.median(float(row["max_drawdown_r"]) for row in part),
            "robustness_a_count": sum(str(row["config_id"]) in bucket_ids["A"] for row in part),
            "robustness_b_count": sum(str(row["config_id"]) in bucket_ids["B"] for row in part),
            "robustness_c_count": sum(str(row["config_id"]) in bucket_ids["C"] for row in part),
            "largest_connected_plateau_size": max((size_by_config.get(str(row["config_id"]), 0) for row in part), default=0),
        })
    return output


def _report(summary: Mapping[str, Any]) -> str:
    level_lines = "\n".join(
        f"- {row['level']}: profitable={row['profitable_configuration_count']:,}, "
        f"top={row['top_aggregate_config_id']} ({float(row['top_aggregate_total_r']):.6f}R), "
        f"broadest plateau={row['broadest_plateau_size']:,}"
        for row in summary["level_summary"]
    )
    q_lines = "\n".join(
        f"- {row['level']} Q={float(row['quality_threshold']):.2f}: "
        f"profitable={row['profitable_weight_count']:,}, median trades={float(row['median_trades']):.1f}, "
        f"median R={float(row['median_total_r']):.6f}, best R={float(row['best_total_r']):.6f}"
        for row in summary["q_summary"]
    )
    bands = summary["sample_bands"]
    return "\n".join([
        "# Europe Weight x Q retrospective research", "",
        "**THIS IS RETROSPECTIVE EUROPE WEIGHT x Q OPTIMIZATION ON ALREADY-SEEN DATA.**", "",
        "**IT IS NOT FRESH OOS EVIDENCE. Any candidate requires future untouched validation.**", "",
        f"Evidence label: `{EVIDENCE_LABEL}`", "", "## Integrity and architecture", "",
        f"- Cache interactions: {summary['cache_interactions']:,}",
        f"- Eligible sessions: {summary['eligible_sessions']}",
        f"- Legal weights: {summary['weight_count']:,}", f"- Q values: {summary['q_count']}",
        f"- Independent structural levels: {summary['level_count']}", f"- Matrix cells: {summary['matrix_cells']:,}",
        f"- W04 baseline reconciliation: {summary['baseline_reconciliation']}",
        "- Stage B source/DBN reads: 0", "- Databento/network calls: 0",
        "- Every cell used independent confirmation, pending-setup and one-position state.", "",
        "## Level results", "", level_lines, "", "## Q results", "", q_lines, "",
        "## Trade-count safety", "", f"- <10: {bands['LT_10']:,}", f"- 10-14: {bands['10_TO_14']:,}",
        f"- 15-24: {bands['15_TO_24']:,}", f"- 25-39: {bands['25_TO_39']:,}", f"- 40+: {bands['40_PLUS']:,}", "",
        "## Descriptive robustness", "", f"- A configurations: {summary['robustness_counts']['A']:,}",
        f"- B configurations: {summary['robustness_counts']['B']:,}", f"- C configurations: {summary['robustness_counts']['C']:,}",
        f"- Connected plateaus: {summary['plateau_count']:,}", "- Bucket C's predeclared neighbor-worst floor is -10R.", "",
        "## Interpretation", "", summary["g3_g4_conclusion"], "", summary["higher_q_conclusion"], "",
        summary["broad_region_conclusion"], "", "No combined multi-level strategy was constructed. No production winner was selected.", "",
    ])


def _publish_corrected(staging: Path, output_root: Path) -> None:
    archive = output_root / "superseded-derived-ledger-attempt-v1"
    existing_files = [path for path in output_root.iterdir() if path.is_file()]
    if existing_files:
        if archive.exists():
            raise EuropeWeightQError("superseded-attempt archive already exists while legacy files remain")
        archive.mkdir()
        for path in existing_files:
            path.replace(archive / path.name)
    for path in sorted(staging.iterdir()):
        path.replace(output_root / path.name)
    staging.rmdir()


def run_stage_b(*, cache_root: Path, output_root: Path) -> dict[str, Any]:
    """Run all 135,660 cells from the sealed cache and publish canonical artifacts."""
    started = time.monotonic()
    cache_root, output_root = cache_root.resolve(), output_root.resolve()
    manifest = validate_causal_cache(cache_root)
    eligible_days = [str(value) for value in manifest["eligible_sessions"]]
    staging = output_root / "_corrected-matrix-building"
    if staging.exists():
        raise EuropeWeightQError(f"corrected matrix staging already exists: {staging}")
    staging.mkdir(parents=True)
    sinks = {
        name: CsvSink(staging / filename) for name, filename in {
            "configuration": "configuration-results.csv", "monthly": "monthly-results.csv",
            "direction": "direction-results.csv", "execution": "execution-model-results.csv",
            "neighbor": "neighbor-robustness.csv", "plateau": "plateau-membership.csv",
            "a": "robustness-a.csv", "b": "robustness-b.csv", "c": "robustness-c.csv",
        }.items()
    }
    level_summaries: list[dict[str, Any]] = []
    q_summaries: list[dict[str, Any]] = []
    plateau_summaries: list[dict[str, Any]] = []
    sample_bands: dict[str, int] = {name: 0 for name in ("LT_10", "10_TO_14", "15_TO_24", "25_TO_39", "40_PLUS")}
    robust_counts = {"A": 0, "B": 0, "C": 0}
    profitable_weights: list[dict[str, Any]] = []
    cell_count = 0
    for family in FAMILIES:
        results, monthly, directions, executions = _run_family(cache_root=cache_root, family=family, eligible_days=eligible_days)
        if len(results) != EXPECTED_LEVEL_CONFIGURATION_COUNT:
            raise EuropeWeightQError(f"family matrix cardinality mismatch: {family}")
        neighbor_rows = neighbor_statistics(results)
        memberships, component_summaries = plateau_membership(results, neighbor_rows)
        buckets = descriptive_buckets(results, neighbor_rows)
        level_summaries.append(_level_summary(family, results, neighbor_rows, component_summaries, buckets))
        q_summaries.extend(_q_summary(family, results, memberships, buckets))
        plateau_summaries.extend({**row, "plateau_id": f"{family}|{row['plateau_id']}"} for row in component_summaries)
        for row in memberships:
            row["plateau_id"] = f"{family}|{row['plateau_id']}"
        for name in sample_bands:
            sample_bands[name] += sum(str(row["sample_band"]) == name for row in results)
        for name in robust_counts:
            robust_counts[name] += len(buckets[name])
        profitable_weights.extend(row for row in results if float(row["total_r"]) > 0)
        sinks["configuration"].append(results)
        sinks["monthly"].append(monthly)
        sinks["direction"].append(directions)
        sinks["execution"].append(executions)
        sinks["neighbor"].append(neighbor_rows)
        sinks["plateau"].append(memberships)
        sinks["a"].append(buckets["A"])
        sinks["b"].append(buckets["B"])
        sinks["c"].append(buckets["C"])
        cell_count += len(results)
    for sink in sinks.values():
        sink.close()
    if cell_count != EXPECTED_TOTAL_COUNT or sum(sample_bands.values()) != EXPECTED_TOTAL_COUNT:
        raise EuropeWeightQError("final matrix/sample-band cardinality mismatch")

    weights = [{
        "weight_id": "W" + "-".join(f"{value:02d}" for value in units),
        **{f"G{index}_weight": value / 20 for index, value in enumerate(units, 1)}, "sum": 1.0,
    } for units in generate_weight_grid()]
    _write_csv(staging / "weight-grid.csv", weights)
    _write_csv(staging / "level-summary.csv", level_summaries)
    _write_csv(staging / "q-summary.csv", q_summaries)
    _write_json(staging / "plateau-summary.json", {"components": plateau_summaries, "automatic_selection": False})
    shutil.copy2(cache_root / "europe-interaction-cache.parquet", staging / "europe-interaction-cache.parquet")
    shutil.copy2(cache_root / "cache-manifest.json", staging / "cache-manifest.json")
    shutil.copy2(cache_root / "baseline-reconciliation.json", staging / "baseline-reconciliation.json")
    contract = {
        "evidence_label": EVIDENCE_LABEL, "not_fresh_oos_evidence": True,
        "eligible_sessions": EXPECTED_ELIGIBLE_SESSIONS, "weight_count": EXPECTED_WEIGHT_COUNT,
        "q_values": [str(value) for value in QUALITY_THRESHOLDS], "level_families": list(FAMILIES),
        "matrix_cells": EXPECTED_TOTAL_COUNT, "cache_sha256": manifest["cache_sha256"],
        "semantic_hashes": semantic_hashes(), "independent_configuration_state": True,
        "frozen_confirmation": "+3 favorable ES ticks from 5 through 15 seconds; no early invalidation",
        "latency_ms": 2, "stop_buffer_ticks": 5, "target_r": 3.0, "maximum_nominal_risk_usd": 250.0,
        "execution": "ES first; MES_PROXY_FROM_ES fallback", "automatic_selection": False,
        "robustness_c_neighbor_worst_floor_r": ROBUSTNESS_C_NEIGHBOR_WORST_FLOOR_R,
    }
    _write_json(staging / "frozen-contract.json", contract)
    if profitable_weights:
        medians = {f"G{index}": statistics.median(float(row[f"G{index}_weight"]) for row in profitable_weights) for index in range(1, 6)}
        g3_g4 = f"Across profitable cells, median weights were {medians}. This is descriptive only; it does not alter W04 or select a new rule."
    else:
        g3_g4 = "No profitable cells existed, so no profitable-region G3/G4 pattern could be estimated."
    q_profitable: dict[float, int] = defaultdict(int)
    for row in q_summaries:
        q_profitable[float(row["quality_threshold"])] += int(row["profitable_weight_count"])
    highest = max(q_profitable, key=q_profitable.get)
    higher_q = f"Profitable-cell counts by Q were {dict(sorted(q_profitable.items()))}; Q={highest:.2f} had the most. This is retrospective and does not select Q."
    broad_levels = [row["level"] for row in level_summaries if int(row["broadest_plateau_size"]) > 1]
    broad = f"Levels with a connected profitable plateau larger than one cell: {broad_levels or 'none'}. Plateaus are descriptive, not validated evidence."
    summary = {
        "status": "MATRIX_COMPLETE_CAUSAL_OFFLINE", "evidence_label": EVIDENCE_LABEL,
        "not_fresh_oos_evidence": True, "cache_interactions": int(manifest["interaction_count"]),
        "cache_build_duration_seconds": float(manifest["stage_a_duration_seconds"]),
        "stage_b_duration_seconds": time.monotonic() - started,
        "eligible_sessions": EXPECTED_ELIGIBLE_SESSIONS, "weight_count": EXPECTED_WEIGHT_COUNT,
        "q_count": len(QUALITY_THRESHOLDS), "level_count": len(FAMILIES), "matrix_cells": cell_count,
        "baseline_reconciliation": "PASS", "stage_b_source_reads": 0, "network_calls": 0,
        "downloads": 0, "automatic_selection": False, "sample_bands": sample_bands,
        "robustness_counts": robust_counts, "plateau_count": len(plateau_summaries),
        "level_summary": level_summaries, "q_summary": q_summaries,
        "g3_g4_conclusion": g3_g4, "higher_q_conclusion": higher_q, "broad_region_conclusion": broad,
    }
    _write_json(staging / "summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_report(summary), encoding="utf-8")
    artifact_hashes = {path.name: _file_sha256(path) for path in sorted(staging.iterdir()) if path.is_file()}
    _write_json(staging / "run-manifest.json", {
        "status": summary["status"], "evidence_label": EVIDENCE_LABEL,
        "artifact_sha256": artifact_hashes, "network_calls": 0, "downloads": 0,
    })
    _publish_corrected(staging, output_root)
    return summary


def run_all(
    *, repository_root: Path, output_root: Path = OUTPUT_ROOT, stage_a_workers: int = 4,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    resolved_output = (output_root if output_root.is_absolute() else repository_root / output_root).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    build_causal_cache(
        repository_root=repository_root, output_root=resolved_output,
        workers=stage_a_workers,
    )
    return run_stage_b(cache_root=resolved_output / CACHE_DIRECTORY, output_root=resolved_output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage-a", "stage-b", "run"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="compatibility; immutable reference root")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache", type=Path, help="compatibility alias for --cache-root")
    parser.add_argument("--stage-a-workers", type=int, default=4)
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_root = (args.output_root if args.output_root.is_absolute() else repository_root / args.output_root).resolve()
    try:
        if args.command == "stage-a":
            output_root.mkdir(parents=True, exist_ok=True)
            result = build_causal_cache(
                repository_root=repository_root, output_root=output_root,
                workers=args.stage_a_workers,
            )
        elif args.command == "stage-b":
            cache_root = args.cache_root or args.cache or (output_root / CACHE_DIRECTORY)
            result = run_stage_b(cache_root=cache_root, output_root=output_root)
        else:
            result = run_all(
                repository_root=repository_root, output_root=output_root,
                stage_a_workers=args.stage_a_workers,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps({
        key: result[key] for key in ("status", "evidence_label", "matrix_cells", "cache_sha256", "cache_root", "reused")
        if key in result
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
