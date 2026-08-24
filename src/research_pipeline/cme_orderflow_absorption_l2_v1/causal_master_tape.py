"""One-pass causal master tape for the frozen Dec-2025/Jan-2026 L2 block.

The builder is intentionally local-only.  It consumes the already sealed
native ES MBP-10, native MES MBP-1, and prior-RTH trade files through the
existing V3 adapters.  It never contacts Databento and it does not select a
threshold or a score weight.

The compact tape retains every ES execution, every changed executable BBO,
the first ordinary source observation at/after each possible setup's frozen
two-millisecond entry latency, explicit book-state transitions, and the
effective hard-flat boundary.  Unchanged ordinary observations are omitted
unless they are an entry probe; they cannot alter confirmation, stop, target,
or hard-flat economics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import historical_runner as historical
from . import v3_poc_dec2025_jan2026_replay as parent
from . import v3_poc_fresh_august_replay as native
from .model import (
    ENTRY_LATENCY_NS,
    MAX_CONFIRMATION_NS,
    MIN_CONFIRMATION_NS,
    TICK,
    Execution,
    L2Interaction,
    L2InteractionEngine,
    L2Setup,
    StructuralLevel,
)
from .v2_quality050 import V2_CONFIG
from .v3_poc_only import STRATEGY_ID as V3_STRATEGY_ID
from .v3_poc_only import v3_contract_sha256
from .v4_poc_q45 import STRATEGY_ID as V4_STRATEGY_ID
from .v4_poc_q45 import V4_CONFIG, v4_contract_sha256


DATA_ROOT = parent.DATA_ROOT
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_CAUSAL_MASTER_DEC2025_JAN2026")
V3_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
V4_CONTRACT_SHA256 = "1e2e9248cb7661e3c075afbeac38e8ecdf9e10bb1eb58a939ff0c22235286cf9"
EXPECTED_INTERACTIONS = 1_482
EVENT_SCHEMA_VERSION = "L2_CAUSAL_EVENT_TAPE_V1"
ORDERING_RULE_VERSION = "TIMESTAMP_THEN_MES0_ES1_CALENDAR2_THEN_SOURCE_INDEX_V1"
CALENDAR_SEMANTICS_VERSION = "FINALIZED_WINTER_RTH_EFFECTIVE_HARD_FLAT_V1"
EVIDENCE_LABEL = "DEC2025_JAN2026_CAUSAL_MASTER_NO_PARAMETER_SELECTION"
ALLOWED_THRESHOLDS = tuple(Decimal(value) for value in ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60"))
DEFAULT_WEIGHTS = {
    "aggression_score": Decimal("0.28"),
    "restoration_score": Decimal("0.25"),
    "price_resistance_score": Decimal("0.22"),
    "persistence_score": Decimal("0.12"),
    "multi_level_support_score": Decimal("0.13"),
}
SCORE_FIELDS = tuple(DEFAULT_WEIGHTS)
STREAM_PRIORITY = {"MES": 0, "ES": 1, "CALENDAR": 2}
EVENT_COLUMNS = (
    "session_date", "event_ordinal", "timestamp_ns", "stream", "stream_priority",
    "source_index", "event_type", "es_bid", "es_ask", "mes_bid", "mes_ask",
    "execution_price", "execution_size", "execution_aggressor", "book_state",
    "entry_probe_count", "es_quote_timestamp_ns", "mes_quote_timestamp_ns",
    "hard_flat_reason",
)

EXPECTED_GATES: dict[str, dict[str, float | int | None]] = {
    "V3_DECEMBER": {
        "completed_trades": 38, "wins": 10, "losses": 28,
        "total_r": -3.790261594206621, "net_pnl_usd": -793.75,
        "profit_factor": 0.8745901963107793,
        "max_cumulative_drawdown_r": -12.946378636204262,
    },
    "V3_FULL": {
        "completed_trades": 65, "wins": 21, "losses": 44,
        "total_r": 9.383354585327483, "net_pnl_usd": 2136.50,
        "profit_factor": 1.2172233236744445,
        "max_cumulative_drawdown_r": -12.946378636204262,
    },
    "V4_FULL": {
        "completed_trades": 73, "wins": 22, "losses": 51,
        "total_r": 5.063008264981163, "net_pnl_usd": 1116.0,
        "profit_factor": 1.0972549019607842,
        "max_cumulative_drawdown_r": -17.946378636204265,
    },
}

REQUIRED_ROOT_ARTIFACTS = frozenset({
    "build-summary.json",
    "calendar.json",
    "diagnostic-report.md",
    "interaction-event-index.parquet",
    "interaction-master.parquet",
    "metadata.json",
    "reproduction-gates.json",
})
MASTER_REQUIRED_COLUMNS = frozenset({
    "interaction_id", "source_interaction_id", "session_date",
    "interaction_start_ns", "interaction_end_ns", "direction", "level",
    "level_price", "interaction_end_price", "zone_low", "zone_high",
    "termination", "aggression_score", "restoration_score",
    "price_resistance_score", "persistence_score",
    "multi_level_support_score", "false_refill_penalty",
    "original_v3_quality_score", "original_v3_rejection_reasons",
    "non_quality_rejection_reasons", "weights_label",
})
INDEX_REQUIRED_COLUMNS = frozenset({
    "interaction_id", "session_date", "event_partition",
    "session_first_event_ordinal", "session_last_event_ordinal",
    "confirmation_start_ns", "confirmation_end_ns_inclusive",
    "derived_first_confirmation_timestamp_ns",
    "derived_first_confirmation_price", "entry_ready_ns",
    "entry_observation_event_ordinal", "counterfactual_path_end_ns",
})


class CausalMasterTapeError(RuntimeError):
    """The master tape cannot be proven equivalent to the frozen source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _event_schema() -> Any:
    import pyarrow as pa

    return pa.schema([
        ("session_date", pa.string()), ("event_ordinal", pa.int64()),
        ("timestamp_ns", pa.int64()), ("stream", pa.string()),
        ("stream_priority", pa.int8()), ("source_index", pa.int64()),
        ("event_type", pa.string()), ("es_bid", pa.float64()),
        ("es_ask", pa.float64()), ("mes_bid", pa.float64()),
        ("mes_ask", pa.float64()), ("execution_price", pa.float64()),
        ("execution_size", pa.int64()), ("execution_aggressor", pa.string()),
        ("book_state", pa.string()), ("entry_probe_count", pa.int32()),
        ("es_quote_timestamp_ns", pa.int64()), ("mes_quote_timestamp_ns", pa.int64()),
        ("hard_flat_reason", pa.string()),
    ])


class AtomicParquetStream:
    """Bounded-memory Parquet writer with an atomic final rename."""

    def __init__(self, path: Path, *, batch_rows: int = 50_000) -> None:
        import pyarrow.parquet as pq

        self.path = path
        self.part = path.with_suffix(path.suffix + ".part")
        if path.exists() or self.part.exists():
            raise CausalMasterTapeError(f"refusing to overwrite causal tape partition: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = _event_schema()
        self.writer = pq.ParquetWriter(self.part, self.schema, compression="zstd", use_dictionary=True)
        self.batch_rows = batch_rows
        self.buffer: list[dict[str, Any]] = []
        self.row_count = 0

    def append(self, row: Mapping[str, Any]) -> None:
        self.buffer.append({name: row.get(name) for name in EVENT_COLUMNS})
        if len(self.buffer) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        import pyarrow as pa

        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table)
        self.row_count += len(self.buffer)
        self.buffer.clear()

    def close(self) -> dict[str, Any]:
        self.flush()
        self.writer.close()
        self.part.replace(self.path)
        return {"path": str(self.path), "rows": self.row_count, "bytes": self.path.stat().st_size,
                "sha256": _sha256(self.path)}

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            if self.part.exists():
                self.part.unlink()


def _write_small_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists():
        raise CausalMasterTapeError(f"refusing to overwrite immutable parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    temporary.replace(path)
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _close_parquet(parquet: Any) -> None:
    """Close a PyArrow reader deterministically, including memory maps on Windows."""
    try:
        parquet.close(force=True)
    except TypeError:  # pragma: no cover - compatibility with older PyArrow
        parquet.close()


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    try:
        rows: list[dict[str, Any]] = []
        for batch in parquet.iter_batches(batch_size=65_536):
            rows.extend(batch.to_pylist())
        return rows
    finally:
        _close_parquet(parquet)


def _read_parquet_rows_for_sessions(path: Path, sessions: Sequence[str]) -> list[dict[str, Any]]:
    """Predicate-filter a sealed table before exposing rows to a caller."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("session_date", "in", list(sessions))])
    try:
        rows = table.to_pylist()
    finally:
        del table
    allowed = set(sessions)
    if any(str(row["session_date"]) not in allowed for row in rows):
        raise CausalMasterTapeError("session predicate exposed an out-of-scope row")
    return rows


def _iter_parquet_rows(path: Path, *, batch_size: int = 65_536) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
    finally:
        _close_parquet(parquet)


def event_order_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["timestamp_ns"]), int(row["stream_priority"]), int(row["source_index"])


def validate_event_order(rows: Iterable[Mapping[str, Any]]) -> int:
    previous: tuple[int, int, int] | None = None
    count = 0
    for row in rows:
        key = event_order_key(row)
        if previous is not None and key <= previous:
            raise CausalMasterTapeError(f"causal event ordering is not strictly increasing: {previous} then {key}")
        previous = key
        count += 1
    return count


def normalize_threshold(value: float | str | Decimal) -> Decimal:
    threshold = Decimal(str(value)).quantize(Decimal("0.00"))
    if threshold not in ALLOWED_THRESHOLDS or Decimal(str(value)) != threshold:
        raise CausalMasterTapeError(f"unsupported predeclared threshold: {value}")
    return threshold


def validate_legal_weights(weights: Mapping[str, float | str | Decimal]) -> dict[str, Decimal]:
    if set(weights) != set(SCORE_FIELDS):
        raise CausalMasterTapeError("weight vector must contain exactly the five frozen positive components")
    normalized = {name: Decimal(str(weights[name])) for name in SCORE_FIELDS}
    # The frozen V3 vector predates the later legal 0.05 research grid and is
    # deliberately not snapped to that grid.  It remains an explicitly valid
    # reproduction vector; every future alternative must satisfy the grid.
    if normalized == DEFAULT_WEIGHTS:
        return normalized
    if any(value < Decimal("0.05") or value * 20 != (value * 20).to_integral_value() for value in normalized.values()):
        raise CausalMasterTapeError("weights must be at least 0.05 and use exact 0.05 increments")
    if sum(normalized.values()) != Decimal("1.00"):
        raise CausalMasterTapeError("weights must sum exactly to 1.00")
    return normalized


def recompute_quality(row: Mapping[str, Any], weights: Mapping[str, float | str | Decimal] = DEFAULT_WEIGHTS) -> float:
    normalized = validate_legal_weights(weights)
    positive = sum(Decimal(str(row[name])) * normalized[name] for name in SCORE_FIELDS)
    penalty = Decimal(str(row["false_refill_penalty"])) * Decimal(str(V2_CONFIG.false_refill_penalty_weight))
    return float(max(Decimal("0"), min(Decimal("1"), positive - penalty)))


def interaction_is_accepted(
    row: Mapping[str, Any], *, threshold: float | str | Decimal,
    weights: Mapping[str, float | str | Decimal] = DEFAULT_WEIGHTS,
) -> bool:
    required = normalize_threshold(threshold)
    primitive = str(row.get("non_quality_rejection_reasons") or "")
    return not primitive and Decimal(str(recompute_quality(row, weights))) >= required


def _interaction_row(interaction: L2Interaction, day: str) -> dict[str, Any]:
    if interaction.end_ns is None or interaction.end_price is None:
        raise CausalMasterTapeError("only completed interactions can enter the master")
    features = interaction.feature_inputs()
    components = interaction.component_scores()
    quality = interaction.quality()["l2_absorption_quality_score"]
    _accepted, reasons = interaction.qualification()
    primitive = tuple(reason for reason in reasons if reason != "L2_QUALITY_BELOW_THRESHOLD")
    master_id = f"{day}|{interaction.interaction_id}"
    row = {
        "interaction_id": master_id,
        "source_interaction_id": interaction.interaction_id,
        "session_date": day,
        "interaction_start_ns": interaction.start_ns,
        "interaction_end_ns": interaction.end_ns,
        "direction": interaction.direction,
        "level": interaction.level.name,
        "level_price": interaction.level.price,
        "interaction_end_price": interaction.end_price,
        "zone_low": interaction.zone_low,
        "zone_high": interaction.zone_high,
        "termination": interaction.termination,
        **features,
        **components,
        "original_v3_quality_score": quality,
        "original_v3_rejection_reasons": ";".join(reasons),
        "non_quality_rejection_reasons": ";".join(primitive),
        "weights_label": V2_CONFIG.weights_label,
    }
    recalculated = recompute_quality(row)
    if not math.isclose(recalculated, float(quality), rel_tol=0.0, abs_tol=1e-12):
        raise CausalMasterTapeError(f"stored score components do not reproduce V3 quality: {master_id}")
    return row


@dataclass
class _Window:
    interaction_id: str
    end_ns: int
    end_price: float
    direction: str
    confirmation_timestamp_ns: int | None = None
    confirmation_price: float | None = None
    entry_ready_ns: int | None = None
    entry_event_ordinal: int | None = None


class CausalWindowTracker:
    """Outcome-free confirmation/index tracker for every pre-quality interaction."""

    def __init__(self) -> None:
        self.windows: dict[str, _Window] = {}
        self._unconfirmed: set[str] = set()
        self._ready: list[tuple[int, str]] = []

    def register(self, row: Mapping[str, Any]) -> None:
        identifier = str(row["interaction_id"])
        if identifier in self.windows:
            raise CausalMasterTapeError(f"duplicate interaction id in causal tracker: {identifier}")
        self.windows[identifier] = _Window(
            identifier, int(row["interaction_end_ns"]), float(row["interaction_end_price"]), str(row["direction"]),
        )
        self._unconfirmed.add(identifier)

    def observe_es_execution(self, execution: Execution) -> list[str]:
        newly_confirmed: list[str] = []
        for identifier in tuple(self._unconfirmed):
            window = self.windows[identifier]
            age = execution.timestamp_ns - window.end_ns
            if age > MAX_CONFIRMATION_NS:
                self._unconfirmed.remove(identifier)
                continue
            if age < MIN_CONFIRMATION_NS:
                continue
            favorable = ((execution.price - window.end_price) / TICK if window.direction == "BUYER_ABSORPTION"
                         else (window.end_price - execution.price) / TICK)
            if favorable >= 3:
                window.confirmation_timestamp_ns = execution.timestamp_ns
                window.confirmation_price = execution.price
                window.entry_ready_ns = execution.timestamp_ns + ENTRY_LATENCY_NS
                heapq.heappush(self._ready, (window.entry_ready_ns, identifier))
                self._unconfirmed.remove(identifier)
                newly_confirmed.append(identifier)
        return newly_confirmed

    def due_entry_probes(self, timestamp_ns: int) -> list[str]:
        due: list[str] = []
        while self._ready and self._ready[0][0] <= timestamp_ns:
            _ready_ns, identifier = heapq.heappop(self._ready)
            window = self.windows[identifier]
            if window.entry_event_ordinal is None:
                due.append(identifier)
        return due

    def bind_entry_probe(self, identifiers: Iterable[str], event_ordinal: int) -> None:
        for identifier in identifiers:
            window = self.windows[identifier]
            if window.entry_event_ordinal is not None:
                raise CausalMasterTapeError(f"duplicate entry observation for interaction: {identifier}")
            window.entry_event_ordinal = event_ordinal

    def index_rows(self, *, day: str, first_event: int, last_event: int, cutoff_ns: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for identifier, window in sorted(self.windows.items(), key=lambda item: (item[1].end_ns, item[0])):
            rows.append({
                "interaction_id": identifier,
                "session_date": day,
                "event_partition": f"causal-event-tape/{day}.parquet",
                "session_first_event_ordinal": first_event,
                "session_last_event_ordinal": last_event,
                "confirmation_start_ns": window.end_ns + MIN_CONFIRMATION_NS,
                "confirmation_end_ns_inclusive": window.end_ns + MAX_CONFIRMATION_NS,
                "derived_first_confirmation_timestamp_ns": window.confirmation_timestamp_ns,
                "derived_first_confirmation_price": window.confirmation_price,
                "entry_ready_ns": window.entry_ready_ns,
                "entry_observation_event_ordinal": window.entry_event_ordinal,
                "counterfactual_path_end_ns": cutoff_ns,
            })
        return rows


@dataclass(frozen=True)
class SessionBuildSpec:
    day: str
    es_path: str
    mes_path: str
    level_price: float
    start_ns: int
    cutoff_ns: int
    hard_flat_reason: str
    staging_root: str
    maintenance_mode: str = "WINTER_CALENDAR"


def _event_row(
    *, spec: SessionBuildSpec, ordinal: int, timestamp_ns: int, stream: str,
    source_index: int, event_type: str, es_quote: tuple[float, float] | None,
    mes_quote: tuple[float, float] | None, execution: Execution | None = None,
    book_state: str = "EXECUTABLE", entry_probe_count: int = 0,
    es_quote_timestamp_ns: int | None = None, mes_quote_timestamp_ns: int | None = None,
    hard_flat_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "session_date": spec.day, "event_ordinal": ordinal, "timestamp_ns": timestamp_ns,
        "stream": stream, "stream_priority": STREAM_PRIORITY[stream], "source_index": source_index,
        "event_type": event_type,
        "es_bid": es_quote[0] if es_quote else None, "es_ask": es_quote[1] if es_quote else None,
        "mes_bid": mes_quote[0] if mes_quote else None, "mes_ask": mes_quote[1] if mes_quote else None,
        "execution_price": execution.price if execution else None,
        "execution_size": execution.size if execution else None,
        "execution_aggressor": execution.aggressor if execution else None,
        "book_state": book_state, "entry_probe_count": entry_probe_count,
        "es_quote_timestamp_ns": es_quote_timestamp_ns, "mes_quote_timestamp_ns": mes_quote_timestamp_ns,
        "hard_flat_reason": hard_flat_reason,
    }


def liquidation_window_quote(
    *, cutoff_ns: int, quote: tuple[float, float] | None, quote_timestamp_ns: int | None,
) -> tuple[tuple[float, float] | None, int | None]:
    """Return only an executable quote inside the frozen inclusive one-second window."""
    if quote is None or quote_timestamp_ns is None:
        return None, None
    if not cutoff_ns - historical.CUTOFF_QUOTE_LOOKBACK_NS <= quote_timestamp_ns <= cutoff_ns:
        return None, None
    bid, ask = quote
    if not (bid > 0.0 and ask > bid):
        raise CausalMasterTapeError("hard-flat boundary received a malformed executable quote")
    return quote, quote_timestamp_ns


def resolve_native_hard_flat_boundary(
    *, cutoff_ns: int,
    es_quote: tuple[float, float] | None,
    es_quote_timestamp_ns: int | None,
    mes_quote: tuple[float, float] | None,
    mes_quote_timestamp_ns: int | None,
) -> dict[str, Any]:
    """Resolve native cutoff evidence with the frozen Fresh-August semantics.

    The strategy exits only the selected instrument.  The compact tape retains
    every independently valid instrument quote and leaves an unavailable
    instrument absent so replay remains fail-closed if that instrument is ever
    selected.  A calendar boundary never manufactures a quote or timestamp.
    """
    valid_es, valid_es_ns = liquidation_window_quote(
        cutoff_ns=cutoff_ns, quote=es_quote, quote_timestamp_ns=es_quote_timestamp_ns,
    )
    valid_mes, valid_mes_ns = liquidation_window_quote(
        cutoff_ns=cutoff_ns, quote=mes_quote, quote_timestamp_ns=mes_quote_timestamp_ns,
    )
    if valid_es is None and valid_mes is None:
        raise CausalMasterTapeError(
            "frozen hard-flat completion lacks any executable BBO inside the inclusive liquidation window"
        )
    return {
        "es_quote": valid_es,
        "es_quote_timestamp_ns": valid_es_ns,
        "mes_quote": valid_mes,
        "mes_quote_timestamp_ns": valid_mes_ns,
        "liquidation_window_start_ns": cutoff_ns - historical.CUTOFF_QUOTE_LOOKBACK_NS,
        "liquidation_window_end_ns": cutoff_ns,
        "exact_hard_flat_record_required": False,
        "invented_quote_count": 0,
    }


def _session_checkpoint_valid(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for record in payload.get("artifacts", {}).values():
        target = Path(str(record.get("path")))
        if not target.is_file() or target.stat().st_size != record.get("bytes") or _sha256(target) != record.get("sha256"):
            return None
    return payload


def _build_session(spec: SessionBuildSpec) -> dict[str, Any]:
    """Stream one session once and atomically publish its staging partitions."""
    root = Path(spec.staging_root)
    checkpoint = root / "session-builds" / f"{spec.day}.json"
    existing = _session_checkpoint_valid(checkpoint)
    if existing is not None:
        return existing
    event_path = root / "causal-event-tape" / f"{spec.day}.parquet"
    interaction_path = root / "session-builds" / spec.day / "interactions.parquet"
    index_path = root / "session-builds" / spec.day / "interaction-index.parquet"
    if event_path.exists() or interaction_path.exists() or index_path.exists():
        raise CausalMasterTapeError(f"unverified partial session output exists: {spec.day}")

    adapter = native.NativeMBP10Adapter()
    engine = L2InteractionEngine([StructuralLevel("PRIOR_RTH_POC", spec.level_price)], V2_CONFIG)
    tracker = CausalWindowTracker()
    writer = AtomicParquetStream(event_path)
    es_iter = iter(native._stream_native_mbp10_records(Path(spec.es_path)))
    mes_iter = iter(historical._stream_mes_quotes(Path(spec.mes_path)))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    initialized = closed = False
    completed_seen = 0
    interaction_rows: list[dict[str, Any]] = []
    es_quote: tuple[float, float] | None = None
    mes_quote: tuple[float, float] | None = None
    prior_stored_es: tuple[float, float] | None = None
    prior_stored_mes: tuple[float, float] | None = None
    es_quote_ns = mes_quote_ns = None
    es_source_index = mes_source_index = 0
    ordinal = 0
    decoded_records = stored_events = 0
    next_progress = 5_000_000
    hard_flat_completion_mode: str | None = None

    if spec.maintenance_mode not in {"WINTER_CALENDAR", "SUMMER_NATIVE", "NONE"}:
        raise CausalMasterTapeError(f"unsupported maintenance mode: {spec.maintenance_mode}")

    def is_maintenance(timestamp_ns: int) -> bool:
        if spec.maintenance_mode == "WINTER_CALENDAR":
            return parent._in_maintenance(timestamp_ns, date.fromisoformat(spec.day))
        if spec.maintenance_mode == "SUMMER_NATIVE":
            return native._in_maintenance_pause(timestamp_ns, spec.day)
        return False

    def drain() -> None:
        nonlocal completed_seen
        for interaction in engine.completed[completed_seen:]:
            row = _interaction_row(interaction, spec.day)
            interaction_rows.append(row)
            tracker.register(row)
        completed_seen = len(engine.completed)

    def append_event(row: dict[str, Any], due: Sequence[str]) -> None:
        nonlocal ordinal, stored_events
        writer.append(row)
        tracker.bind_entry_probe(due, ordinal)
        ordinal += 1
        stored_events += 1

    def complete_hard_flat(mode: str) -> None:
        nonlocal closed, es_quote, mes_quote, es_quote_ns, mes_quote_ns, hard_flat_completion_mode
        if not initialized:
            raise CausalMasterTapeError(f"native MBP-10 initialization absent before RTH: {spec.day}")
        adapter.assert_executable_at_boundary()
        evidence = resolve_native_hard_flat_boundary(
            cutoff_ns=spec.cutoff_ns,
            es_quote=es_quote,
            es_quote_timestamp_ns=es_quote_ns,
            mes_quote=mes_quote,
            mes_quote_timestamp_ns=mes_quote_ns,
        )
        es_quote, es_quote_ns = evidence["es_quote"], evidence["es_quote_timestamp_ns"]
        mes_quote, mes_quote_ns = evidence["mes_quote"], evidence["mes_quote_timestamp_ns"]
        engine.finish_rth(spec.cutoff_ns)
        drain()
        append_event(_event_row(
            spec=spec, ordinal=ordinal, timestamp_ns=spec.cutoff_ns, stream="CALENDAR",
            source_index=0, event_type="HARD_FLAT", es_quote=es_quote, mes_quote=mes_quote,
            es_quote_timestamp_ns=es_quote_ns, mes_quote_timestamp_ns=mes_quote_ns,
            hard_flat_reason=spec.hard_flat_reason,
        ), ())
        hard_flat_completion_mode = mode
        closed = True

    try:
        while es is not None or mes is not None:
            es_timestamp = native._timestamp(es) if es is not None else None
            mes_timestamp = mes[0] if mes is not None else None
            timestamp_ns = min(value for value in (es_timestamp, mes_timestamp) if value is not None)
            if timestamp_ns >= spec.cutoff_ns:
                complete_hard_flat("SOURCE_REACHED_OR_PASSED_HARD_FLAT")
                break

            if is_maintenance(timestamp_ns):
                if es_quote is not None or mes_quote is not None:
                    append_event(_event_row(
                        spec=spec, ordinal=ordinal, timestamp_ns=timestamp_ns, stream="CALENDAR",
                        source_index=0, event_type="BOOK_NON_EXECUTABLE", es_quote=None, mes_quote=None,
                        book_state="MAINTENANCE", es_quote_timestamp_ns=None, mes_quote_timestamp_ns=None,
                    ), ())
                    es_quote = mes_quote = prior_stored_es = prior_stored_mes = None
                    es_quote_ns = mes_quote_ns = None
                if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
                    mes_source_index += 1; decoded_records += 1; mes = historical._next(mes_iter)
                else:
                    assert es is not None
                    adapter.feed(es, expected_non_executable=True)
                    es_source_index += 1; decoded_records += 1; es = historical._next(es_iter)
                continue

            if mes_timestamp is not None and mes_timestamp <= (es_timestamp if es_timestamp is not None else mes_timestamp):
                mes_source_index += 1; decoded_records += 1
                if adapter.state != "TEMPORARILY_NON_EXECUTABLE":
                    new_quote = (float(mes[1]), float(mes[2]))
                    mes_quote, mes_quote_ns = new_quote, mes_timestamp
                    due = tracker.due_entry_probes(mes_timestamp)
                    if new_quote != prior_stored_mes or due:
                        append_event(_event_row(
                            spec=spec, ordinal=ordinal, timestamp_ns=mes_timestamp, stream="MES",
                            source_index=mes_source_index, event_type="MES_BBO", es_quote=es_quote,
                            mes_quote=mes_quote, entry_probe_count=len(due),
                            es_quote_timestamp_ns=es_quote_ns, mes_quote_timestamp_ns=mes_quote_ns,
                        ), due)
                        prior_stored_mes = new_quote
                mes = historical._next(mes_iter)
            else:
                assert es is not None and es_timestamp is not None
                es_source_index += 1; decoded_records += 1
                previous_state = adapter.state
                public = adapter.feed(es)
                if es_timestamp < spec.start_ns:
                    initialized = adapter.first_valid_book_ns is not None and adapter.first_valid_book_ns < spec.start_ns
                    es = historical._next(es_iter)
                    continue
                if not initialized:
                    raise CausalMasterTapeError(f"native MBP-10 initialization absent before first RTH event: {spec.day}")
                if public is not None:
                    # Match HistoricalL2Runner exactly: an unexposed native
                    # non-executable row does not advance strategy time.  The
                    # interaction clock advances only on a public ES event.
                    engine.advance(es_timestamp)
                    drain()
                    quote = historical._quote(public.snapshot)
                    if quote is None:
                        raise CausalMasterTapeError("public executable MBP-10 event lacked a valid BBO")
                    es_quote, es_quote_ns = quote, es_timestamp
                    engine.observe_snapshot(public.snapshot, public.update)
                    if public.execution is not None:
                        engine.observe_execution(public.execution)
                        drain()
                        tracker.observe_es_execution(public.execution)
                    due = tracker.due_entry_probes(es_timestamp)
                    changed = quote != prior_stored_es or previous_state != "EXECUTABLE"
                    if changed or public.execution is not None or due:
                        event_type = "ES_EXECUTION" if public.execution is not None else "ES_BBO"
                        append_event(_event_row(
                            spec=spec, ordinal=ordinal, timestamp_ns=es_timestamp, stream="ES",
                            source_index=es_source_index, event_type=event_type, es_quote=es_quote,
                            mes_quote=mes_quote, execution=public.execution, entry_probe_count=len(due),
                            es_quote_timestamp_ns=es_quote_ns, mes_quote_timestamp_ns=mes_quote_ns,
                        ), due)
                        prior_stored_es = quote
                elif adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                    if es_quote is not None or mes_quote is not None:
                        append_event(_event_row(
                            spec=spec, ordinal=ordinal, timestamp_ns=es_timestamp, stream="ES",
                            source_index=es_source_index, event_type="BOOK_NON_EXECUTABLE",
                            es_quote=None, mes_quote=None, book_state=adapter.state,
                        ), ())
                    es_quote = mes_quote = prior_stored_es = prior_stored_mes = None
                    es_quote_ns = mes_quote_ns = None
                es = historical._next(es_iter)

            if decoded_records >= next_progress:
                print(
                    f"MASTER_TAPE {spec.day} records={decoded_records:,} "
                    f"interactions={len(interaction_rows):,} events={stored_events:,}", flush=True,
                )
                next_progress += 5_000_000

        if not closed:
            try:
                complete_hard_flat("SOURCE_ENDED_WITH_SUFFICIENT_LIQUIDATION_EVIDENCE")
            except (CausalMasterTapeError, native.V3FreshReplayError) as exc:
                raise CausalMasterTapeError(
                    f"source ended before frozen hard-flat completion: {spec.day}: {exc}"
                ) from exc
        event_artifact = writer.close()
    except BaseException:
        writer.abort()
        raise

    index_rows = tracker.index_rows(
        day=spec.day, first_event=0, last_event=ordinal - 1, cutoff_ns=spec.cutoff_ns,
    )
    if len(index_rows) != len(interaction_rows):
        raise CausalMasterTapeError("interaction/index cardinality mismatch")
    interaction_artifact = _write_small_parquet(interaction_path, interaction_rows)
    index_artifact = _write_small_parquet(index_path, index_rows)
    result = {
        "session_date": spec.day,
        "status": "SESSION_CAUSAL_TAPE_COMPLETE",
        "decoded_source_records": decoded_records,
        "completed_interactions": len(interaction_rows),
        "stored_events": stored_events,
        "confirmed_interactions": sum(row["derived_first_confirmation_timestamp_ns"] is not None for row in index_rows),
        "entry_probe_interactions": sum(row["entry_observation_event_ordinal"] is not None for row in index_rows),
        "hard_flat_completion_mode": hard_flat_completion_mode,
        "artifacts": {"events": event_artifact, "interactions": interaction_artifact, "index": index_artifact},
    }
    _write_json(checkpoint, result)
    return result


def _session_specs(preflight: Mapping[str, Any], data_root: Path, staging_root: Path) -> list[SessionBuildSpec]:
    specs: list[SessionBuildSpec] = []
    for day in preflight["base"]["target_sessions"]:
        level = parent._validated_merged_poc(preflight, data_root, day)
        es_path, mes_path = parent._execution_paths_for_day(preflight["base"], data_root, day)
        sufficiency = next(row for row in preflight["sufficiency_matrix"] if row["session_date"] == day)
        cutoff_ns = parent._ns(str(sufficiency["effective_hard_flat_utc"]))
        normal = cutoff_ns == historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS)
        specs.append(SessionBuildSpec(
            day=day, es_path=str(es_path), mes_path=str(mes_path), level_price=level.price,
            start_ns=parent._ns(str(sufficiency["required_mes_mbp1_utc"]["start_utc"])),
            cutoff_ns=cutoff_ns,
            hard_flat_reason="HARD_CUTOFF_2245" if normal else "HARD_FLAT_SCHEDULED_CLOSE",
            staging_root=str(staging_root),
        ))
    if len(specs) != parent.EXPECTED_SESSION_COUNT:
        raise CausalMasterTapeError("master tape requires exactly the frozen 42 sessions")
    return specs


def _combine_session_artifacts(staging: Path, sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    interactions: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    for session in sorted(sessions, key=lambda row: str(row["session_date"])):
        artifacts = session["artifacts"]
        interactions.extend(_read_parquet_rows(Path(str(artifacts["interactions"]["path"]))))
        indexes.extend(_read_parquet_rows(Path(str(artifacts["index"]["path"]))))
    interaction_ids = [str(row["interaction_id"]) for row in interactions]
    if len(interaction_ids) != len(set(interaction_ids)):
        raise CausalMasterTapeError("master interaction IDs are not globally unique")
    if set(interaction_ids) != {str(row["interaction_id"]) for row in indexes}:
        raise CausalMasterTapeError("master interaction/index identities do not reconcile")
    return {
        "interactions": _write_small_parquet(staging / "interaction-master.parquet", interactions),
        "index": _write_small_parquet(staging / "interaction-event-index.parquet", indexes),
        "interaction_rows": interactions,
        "index_rows": indexes,
    }


def _event_paths(master_root: Path, calendar: Mapping[str, Any] | None = None) -> list[Path]:
    days = None if calendar is None else calendar.get("target_sessions")
    if days:
        paths = [master_root / "causal-event-tape" / f"{day}.parquet" for day in days]
    else:
        paths = sorted((master_root / "causal-event-tape").glob("*.parquet"))
    if not paths or any(not path.is_file() for path in paths):
        raise CausalMasterTapeError("causal event tape partitions are missing")
    return paths


def _interaction_from_master(row: Mapping[str, Any], config: Any) -> L2Interaction:
    interaction = L2Interaction(
        str(row["source_interaction_id"]), StructuralLevel(str(row["level"]), float(row["level_price"])),
        int(row["interaction_start_ns"]), str(row["direction"]), config,
    )
    interaction.end_ns = int(row["interaction_end_ns"])
    interaction.end_price = float(row["interaction_end_price"])
    interaction.zone_low = float(row["zone_low"])
    interaction.zone_high = float(row["zone_high"])
    interaction.termination = str(row.get("termination") or "MASTER_TAPE")
    return interaction


def _register_master_setup(runner: historical.HistoricalL2Runner, row: Mapping[str, Any]) -> None:
    interaction = _interaction_from_master(row, runner.config)
    setup_id = f"L2:{interaction.interaction_id}"
    if setup_id in runner.signals.pending:
        raise CausalMasterTapeError(f"duplicate session setup identity: {setup_id}")
    setup = L2Setup(setup_id, interaction)
    runner.signals.pending[setup_id] = setup
    runner.setup_ledger.append({
        "interaction_id": interaction.interaction_id, "setup_id": setup_id,
        "date": runner.date, "accepted": True, "interaction_end_ns": interaction.end_ns,
    })


def _process_tape_event(runner: historical.HistoricalL2Runner, row: Mapping[str, Any]) -> None:
    timestamp_ns = int(row["timestamp_ns"])
    event_type = str(row["event_type"])
    if event_type == "BOOK_NON_EXECUTABLE":
        if runner.signals.position is not None:
            raise CausalMasterTapeError("non-executable tape boundary overlaps a counterfactual position")
        # The frozen native adapter distinguishes a bounded crossed-book
        # sequence from a durable source/maintenance boundary.  A transient
        # sequence suspends execution but does not invalidate confirmation;
        # other non-executable boundaries keep the existing terminal policy.
        if str(row.get("book_state")) != "TEMPORARILY_NON_EXECUTABLE":
            for setup in runner.signals.pending.values():
                if setup.state == "CONFIRMED" and setup.terminal_reason is None:
                    setup.state, setup.terminal_reason = "FAILED", "SOURCE_NON_EXECUTABLE_BEFORE_ENTRY"
        runner.es_quote = runner.mes_quote = None
        runner.es_quote_timestamp_ns = runner.mes_quote_timestamp_ns = None
        return
    if event_type == "HARD_FLAT":
        if row.get("es_bid") is not None and row.get("es_ask") is not None:
            runner.es_quote = (float(row["es_bid"]), float(row["es_ask"]))
            runner.es_quote_timestamp_ns = int(row["es_quote_timestamp_ns"])
        if row.get("mes_bid") is not None and row.get("mes_ask") is not None:
            runner.mes_quote = (float(row["mes_bid"]), float(row["mes_ask"]))
            runner.mes_quote_timestamp_ns = int(row["mes_quote_timestamp_ns"])
        if runner.signals.position is not None:
            runner.force_flat_from_last_causal_cutoff_quote(
                timestamp_ns, exit_reason=str(row.get("hard_flat_reason") or "HARD_CUTOFF_2245"),
            )
        runner.signals.advance(timestamp_ns)
        return
    if str(row["stream"]) == "MES":
        runner.observe_mes_quote(timestamp_ns, float(row["mes_bid"]), float(row["mes_ask"]))
        return
    if str(row["stream"]) != "ES":
        raise CausalMasterTapeError(f"unsupported causal event type: {event_type}")
    runner.es_quote = (float(row["es_bid"]), float(row["es_ask"]))
    runner.es_quote_timestamp_ns = timestamp_ns
    runner._manage_position(timestamp_ns, runner.es_quote, "ES")
    runner.signals.advance(timestamp_ns)
    if event_type == "ES_EXECUTION":
        runner.signals.observe_execution(Execution(
            timestamp_ns, float(row["execution_price"]), int(row["execution_size"]),
            str(row["execution_aggressor"]),
        ))
    runner._attempt_entry(timestamp_ns)


def replay_configuration(
    master_root: Path, *, threshold: float | str | Decimal,
    weights: Mapping[str, float | str | Decimal] = DEFAULT_WEIGHTS,
    session_prefix: str | None = None,
) -> dict[str, Any]:
    """Replay one independent chronological portfolio without opening DBNs."""
    normalized_threshold = normalize_threshold(threshold)
    normalized_weights = validate_legal_weights(weights)
    calendar = json.loads((master_root / "calendar.json").read_text(encoding="utf-8"))
    session_days = tuple(
        str(day) for day in calendar["target_sessions"]
        if session_prefix is None or str(day).startswith(session_prefix)
    )
    all_interactions = (
        _read_parquet_rows(master_root / "interaction-master.parquet")
        if session_prefix is None
        else _read_parquet_rows_for_sessions(master_root / "interaction-master.parquet", session_days)
    )
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_interactions:
        day = str(row["session_date"])
        if session_prefix is None or day.startswith(session_prefix):
            by_day[day].append(row)

    runners: list[historical.HistoricalL2Runner] = []
    for event_path in _event_paths(master_root, calendar):
        day = event_path.stem
        if session_prefix is not None and not day.startswith(session_prefix):
            continue
        config = replace(V2_CONFIG, min_quality_score=float(normalized_threshold))
        runner = historical.HistoricalL2Runner(
            date=day, evidence_label=EVIDENCE_LABEL, levels=[], config=config,
            strategy_id=f"MASTER_REPLAY_Q{normalized_threshold}", require_native_mes_for_fallback=True,
        )
        eligible = sorted(
            (row for row in by_day.get(day, ()) if interaction_is_accepted(
                row, threshold=normalized_threshold, weights=normalized_weights,
            )),
            key=lambda row: (int(row["interaction_end_ns"]), str(row["source_interaction_id"])),
        )
        cursor = 0
        previous_key: tuple[int, int, int] | None = None
        for event in _iter_parquet_rows(event_path):
            key = event_order_key(event)
            if previous_key is not None and key <= previous_key:
                raise CausalMasterTapeError(f"event partition is not causally ordered: {day}")
            previous_key = key
            timestamp_ns = int(event["timestamp_ns"])
            while cursor < len(eligible) and int(eligible[cursor]["interaction_end_ns"]) <= timestamp_ns:
                _register_master_setup(runner, eligible[cursor])
                cursor += 1
            _process_tape_event(runner, event)
        while cursor < len(eligible):
            _register_master_setup(runner, eligible[cursor]); cursor += 1
        runners.append(runner)

    trades = [row for runner in runners for row in runner.trade_ledger]
    metrics = historical._performance(trades)
    return {
        "threshold": str(normalized_threshold),
        "weights": {name: str(normalized_weights[name]) for name in SCORE_FIELDS},
        "independent_portfolio_state": True,
        "sessions": len(runners),
        "accepted_setups": sum(len(runner.setup_ledger) for runner in runners),
        "metrics": metrics,
        "trades": trades,
        "dbn_files_opened": 0,
        "network_calls": 0,
    }


def _assert_metrics(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if isinstance(expected_value, float):
            if actual_value is None or not math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise CausalMasterTapeError(
                    f"{name} reproduction mismatch for {field}: expected {expected_value!r}, got {actual_value!r}"
                )
        elif actual_value != expected_value:
            raise CausalMasterTapeError(
                f"{name} reproduction mismatch for {field}: expected {expected_value!r}, got {actual_value!r}"
            )


def reproduction_gates(master_root: Path) -> dict[str, Any]:
    if v3_contract_sha256() != V3_CONTRACT_SHA256 or v4_contract_sha256() != V4_CONTRACT_SHA256:
        raise CausalMasterTapeError("frozen V3/V4 contract hash changed")
    interactions = _read_parquet_rows(master_root / "interaction-master.parquet")
    if len(interactions) != EXPECTED_INTERACTIONS:
        raise CausalMasterTapeError(
            f"pre-quality interaction count mismatch: expected {EXPECTED_INTERACTIONS}, got {len(interactions)}"
        )
    v3_dec = replay_configuration(master_root, threshold="0.50", session_prefix="2025-12")
    v3_full = replay_configuration(master_root, threshold="0.50")
    v4_full = replay_configuration(master_root, threshold="0.45")
    _assert_metrics("V3_DECEMBER", v3_dec["metrics"], EXPECTED_GATES["V3_DECEMBER"])
    _assert_metrics("V3_FULL", v3_full["metrics"], EXPECTED_GATES["V3_FULL"])
    _assert_metrics("V4_FULL", v4_full["metrics"], EXPECTED_GATES["V4_FULL"])
    threshold_contracts = [
        {"threshold": str(value), "state_identity": f"INDEPENDENT_PORTFOLIO_Q{value}"}
        for value in ALLOWED_THRESHOLDS
    ]
    if len({row["state_identity"] for row in threshold_contracts}) != len(ALLOWED_THRESHOLDS):
        raise CausalMasterTapeError("threshold configurations do not have independent state identities")
    return {
        "status": "EXACT_REPRODUCTION_GATES_PASS",
        "v3_december": {"metrics": v3_dec["metrics"], "expected": EXPECTED_GATES["V3_DECEMBER"]},
        "v3_full": {"metrics": v3_full["metrics"], "expected": EXPECTED_GATES["V3_FULL"]},
        "v4_full": {"metrics": v4_full["metrics"], "expected": EXPECTED_GATES["V4_FULL"]},
        "threshold_engine_sanity": threshold_contracts,
        "threshold_matrix_executed": False,
        "weight_grid_generated_or_evaluated": False,
    }


def _source_metadata(preflight: Mapping[str, Any]) -> dict[str, Any]:
    base = preflight["base"]
    supplement = preflight["supplement"]
    return {
        "base_manifest_path": base.get("manifest_path"),
        "base_manifest_sha256": base.get("manifest_sha256"),
        "supplement_manifest_path": supplement.get("manifest_path"),
        "supplement_manifest_sha256": supplement.get("manifest_sha256"),
        "verified_input_files": base.get("files_verified", parent.EXPECTED_COMPONENT_COUNT) + supplement.get("files_verified", 0),
    }


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _expected_session_days() -> tuple[str, ...]:
    return tuple(str(item.session_date) for item in parent._expected_sessions())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalMasterTapeError(f"required JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CausalMasterTapeError(f"required JSON artifact is not an object: {path}")
    return value


def _parquet_descriptor(path: Path) -> dict[str, Any]:
    """Read only a Parquet footer and close its source before returning."""
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise CausalMasterTapeError(f"Parquet footer is unreadable: {path}") from exc
    try:
        return {
            "rows": int(parquet.metadata.num_rows),
            "row_groups": int(parquet.metadata.num_row_groups),
            "schema": parquet.schema_arrow,
        }
    finally:
        _close_parquet(parquet)


def _checkpoint_target(building_root: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    direct = (Path.cwd() / candidate).resolve()
    if direct.exists():
        return direct
    parts = candidate.parts
    markers = (building_root.name, building_root.name + ".building")
    marker = next((parts.index(name) for name in markers if name in parts), None)
    if marker is None:
        raise CausalMasterTapeError(f"checkpoint artifact is outside the building root: {candidate}")
    return building_root.joinpath(*parts[marker + 1:]).resolve()


def _validate_reproduction_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "EXACT_REPRODUCTION_GATES_PASS":
        raise CausalMasterTapeError("staged mandatory reproduction gates did not pass")
    if payload.get("threshold_matrix_executed") is not False:
        raise CausalMasterTapeError("staged artifact unexpectedly executed the threshold matrix")
    if payload.get("weight_grid_generated_or_evaluated") is not False:
        raise CausalMasterTapeError("staged artifact unexpectedly evaluated a weight grid")
    for key, gate_name in (
        ("v3_december", "V3_DECEMBER"),
        ("v3_full", "V3_FULL"),
        ("v4_full", "V4_FULL"),
    ):
        gate = payload.get(key)
        if not isinstance(gate, Mapping) or not isinstance(gate.get("metrics"), Mapping):
            raise CausalMasterTapeError(f"staged reproduction gate is incomplete: {key}")
        _assert_metrics(gate_name, gate["metrics"], EXPECTED_GATES[gate_name])


def validate_building_root(building_root: Path) -> dict[str, Any]:
    """Prove a completed staging tree is publishable without opening a DBN."""
    building_root = building_root.resolve()
    if not building_root.is_dir():
        raise CausalMasterTapeError(f"building root does not exist: {building_root}")
    unfinished = sorted(
        path for path in building_root.rglob("*")
        if path.is_file() and (path.name.endswith(".part") or path.name.endswith(".tmp"))
    )
    if unfinished:
        raise CausalMasterTapeError(f"unfinished staging artifact exists: {unfinished[0]}")

    days = _expected_session_days()
    if len(days) != parent.EXPECTED_SESSION_COUNT:
        raise CausalMasterTapeError("frozen calendar no longer contains exactly 42 sessions")
    expected_files = set(REQUIRED_ROOT_ARTIFACTS)
    for day in days:
        expected_files.update({
            f"causal-event-tape/{day}.parquet",
            f"session-builds/{day}.json",
            f"session-builds/{day}/interaction-index.parquet",
            f"session-builds/{day}/interactions.parquet",
        })
    actual_files = {
        path.relative_to(building_root).as_posix()
        for path in building_root.rglob("*") if path.is_file()
    }
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    if missing:
        raise CausalMasterTapeError(f"building root is incomplete; missing artifact: {missing[0]}")
    if extra:
        raise CausalMasterTapeError(f"building root contains an unexpected artifact: {extra[0]}")

    calendar = _read_json_object(building_root / "calendar.json")
    if tuple(calendar.get("target_sessions", ())) != days:
        raise CausalMasterTapeError("calendar does not contain the exact ordered 42-session block")
    if calendar.get("calendar_semantics_version") != CALENDAR_SEMANTICS_VERSION:
        raise CausalMasterTapeError("calendar semantics version changed")
    metadata = _read_json_object(building_root / "metadata.json")
    if (
        metadata.get("artifact_kind") != "CAUSAL_MASTER_TAPE"
        or metadata.get("event_schema_version") != EVENT_SCHEMA_VERSION
        or metadata.get("ordering_rule_version") != ORDERING_RULE_VERSION
        or metadata.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or metadata.get("v4_contract_sha256") != V4_CONTRACT_SHA256
        or metadata.get("network_calls") != 0
        or metadata.get("downloads") != 0
        or metadata.get("threshold_matrix_executed") is not False
        or metadata.get("weight_grid_generated_or_evaluated") is not False
    ):
        raise CausalMasterTapeError("staged metadata does not match the sealed causal-master contract")
    summary = _read_json_object(building_root / "build-summary.json")
    if (
        summary.get("status") != "CAUSAL_MASTER_TAPE_VALID"
        or summary.get("sessions") != parent.EXPECTED_SESSION_COUNT
        or summary.get("completed_interactions") != EXPECTED_INTERACTIONS
        or summary.get("indexed_interactions") != EXPECTED_INTERACTIONS
        or summary.get("heavy_dbn_passes") != 1
        or summary.get("threshold_performance_matrix_executed") is not False
        or summary.get("weight_grid_generated_or_evaluated") is not False
    ):
        raise CausalMasterTapeError("build summary does not describe a complete sealed master tape")
    gates = _read_json_object(building_root / "reproduction-gates.json")
    _validate_reproduction_artifact(gates)

    master_path = building_root / "interaction-master.parquet"
    index_path = building_root / "interaction-event-index.parquet"
    master_descriptor = _parquet_descriptor(master_path)
    index_descriptor = _parquet_descriptor(index_path)
    if master_descriptor["rows"] != EXPECTED_INTERACTIONS:
        raise CausalMasterTapeError(
            f"interaction population mismatch: expected {EXPECTED_INTERACTIONS}, got {master_descriptor['rows']}"
        )
    if index_descriptor["rows"] != EXPECTED_INTERACTIONS:
        raise CausalMasterTapeError("master interaction/index row counts do not reconcile")
    master_names = set(master_descriptor["schema"].names)
    index_names = set(index_descriptor["schema"].names)
    if not MASTER_REQUIRED_COLUMNS.issubset(master_names):
        raise CausalMasterTapeError("interaction master schema is incomplete")
    if not INDEX_REQUIRED_COLUMNS.issubset(index_names):
        raise CausalMasterTapeError("interaction event-index schema is incomplete")

    master_rows = _read_parquet_rows(master_path)
    index_rows = _read_parquet_rows(index_path)
    interaction_ids = [str(row["interaction_id"]) for row in master_rows]
    index_ids = [str(row["interaction_id"]) for row in index_rows]
    if len(interaction_ids) != len(set(interaction_ids)) or set(interaction_ids) != set(index_ids):
        raise CausalMasterTapeError("master interaction identities do not reconcile with the event index")
    if any(str(row["level"]) != "PRIOR_RTH_POC" for row in master_rows):
        raise CausalMasterTapeError("master contains a non-POC interaction")

    total_event_rows = 0
    total_session_interactions = 0
    total_session_index = 0
    exact_event_schema = _event_schema()
    for day in days:
        event_path = building_root / "causal-event-tape" / f"{day}.parquet"
        session_interactions = building_root / "session-builds" / day / "interactions.parquet"
        session_index = building_root / "session-builds" / day / "interaction-index.parquet"
        event_descriptor = _parquet_descriptor(event_path)
        interaction_descriptor = _parquet_descriptor(session_interactions)
        session_index_descriptor = _parquet_descriptor(session_index)
        if not event_descriptor["schema"].equals(exact_event_schema, check_metadata=False):
            raise CausalMasterTapeError(f"causal event schema mismatch: {day}")
        if interaction_descriptor["rows"] and set(interaction_descriptor["schema"].names) != master_names:
            raise CausalMasterTapeError(f"session interaction schema mismatch: {day}")
        if session_index_descriptor["rows"] and set(session_index_descriptor["schema"].names) != index_names:
            raise CausalMasterTapeError(f"session event-index schema mismatch: {day}")

        checkpoint_path = building_root / "session-builds" / f"{day}.json"
        checkpoint = _read_json_object(checkpoint_path)
        if checkpoint.get("status") != "SESSION_CAUSAL_TAPE_COMPLETE" or checkpoint.get("session_date") != day:
            raise CausalMasterTapeError(f"session checkpoint is incomplete: {day}")
        artifacts = checkpoint.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"events", "interactions", "index"}:
            raise CausalMasterTapeError(f"session checkpoint artifact registry is incomplete: {day}")
        expected_targets = {
            "events": event_path,
            "interactions": session_interactions,
            "index": session_index,
        }
        descriptors = {
            "events": event_descriptor,
            "interactions": interaction_descriptor,
            "index": session_index_descriptor,
        }
        for key, expected_target in expected_targets.items():
            record = artifacts[key]
            if not isinstance(record, Mapping):
                raise CausalMasterTapeError(f"invalid checkpoint artifact record: {day}/{key}")
            target = _checkpoint_target(building_root, record.get("path"))
            if target != expected_target.resolve():
                raise CausalMasterTapeError(f"checkpoint artifact path mismatch: {day}/{key}")
            if (
                record.get("rows") != descriptors[key]["rows"]
                or record.get("bytes") != expected_target.stat().st_size
                or record.get("sha256") != _sha256(expected_target)
            ):
                raise CausalMasterTapeError(f"checkpoint artifact integrity mismatch: {day}/{key}")
        if (
            checkpoint.get("completed_interactions") != interaction_descriptor["rows"]
            or interaction_descriptor["rows"] != session_index_descriptor["rows"]
            or checkpoint.get("stored_events") != event_descriptor["rows"]
        ):
            raise CausalMasterTapeError(f"session checkpoint row counts do not reconcile: {day}")
        total_event_rows += event_descriptor["rows"]
        total_session_interactions += interaction_descriptor["rows"]
        total_session_index += session_index_descriptor["rows"]

    if total_event_rows != summary.get("stored_events"):
        raise CausalMasterTapeError("stored event count does not reconcile across 42 partitions")
    if total_session_interactions != EXPECTED_INTERACTIONS or total_session_index != EXPECTED_INTERACTIONS:
        raise CausalMasterTapeError("session interaction populations do not reconcile to 1,482")

    return {
        "status": "COMPLETED_BUILDING_ROOT_VALID",
        "building_root": str(building_root),
        "file_count": len(actual_files),
        "total_bytes": _tree_size(building_root),
        "sessions": len(days),
        "event_partitions": len(days),
        "stored_events": total_event_rows,
        "completed_interactions": len(interaction_ids),
        "indexed_interactions": len(index_ids),
        "unfinished_artifacts": 0,
        "dbn_files_opened": 0,
        "network_calls": 0,
        "mandatory_reproduction_artifact_status": gates["status"],
    }


def _atomic_directory_publish(building_root: Path, output_root: Path) -> None:
    if building_root.parent.resolve() != output_root.parent.resolve():
        raise CausalMasterTapeError("atomic publish requires building and output roots in the same directory")
    os.rename(building_root, output_root)


def finalize_building_root(*, building_root: Path, output_root: Path) -> dict[str, Any]:
    """Validate and atomically publish an existing completed build, opening zero DBNs."""
    building_root = building_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable causal master output already exists: {output_root}")
    validation = validate_building_root(building_root)
    # Validation closes every Parquet reader explicitly.  Collection releases
    # any Arrow wrapper cycles before the Windows directory move.
    gc.collect()
    _atomic_directory_publish(building_root, output_root)
    if building_root.exists() or not output_root.is_dir():
        raise CausalMasterTapeError("atomic causal-master publish did not complete")
    return {
        **validation,
        "status": "CAUSAL_MASTER_TAPE_PUBLISHED",
        "output_root": str(output_root),
        "atomic_publish": True,
    }


def _diagnostic_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Dec 2025 / Jan 2026 L2 causal master tape", "",
        f"Status: `{summary['status']}`", "",
        "This artifact is infrastructure for later causal research. It did not rank a threshold, generate a weight grid, select a strategy, or contact Databento.", "",
        f"Pre-quality POC interactions: {summary['completed_interactions']:,}",
        f"Shared causal events: {summary['stored_events']:,}",
        f"Source sessions / heavy passes: {summary['sessions']} / 1", "",
        "## Event contract", "",
        "Ordering is timestamp, MES-before-ES for exact ties, calendar boundary last, then monotonically increasing source index.",
        "Stored events are ES executions, changed executable ES BBOs, changed native MES BBOs, first post-latency entry probes, non-executable book transitions, and effective hard-flat boundaries.", "",
        "Every pre-quality interaction has an indexed inclusive +5s/+15s confirmation window and a path through the session hard-flat. Rejected and originally blocked interactions use the same shared tape, so their status is not treated as permanent.", "",
        "## Frozen reproduction", "",
        f"Gate status: `{summary['reproduction_status']}`. Only V3 q0.50 December/full and V4 q0.45 full were executed as mandatory integrity reproductions.",
        "No six-threshold performance matrix and no legal-weight grid was evaluated.", "",
    ])


def _run_session_builds(specs: Sequence[SessionBuildSpec], workers: int) -> list[dict[str, Any]]:
    """Complete every worker and join the pool before returning any artifacts."""
    results: list[dict[str, Any]] = []
    if workers == 1:
        for index, spec in enumerate(specs, start=1):
            print(f"=== CAUSAL MASTER {index:02d}/{len(specs):02d} {spec.day} ===", flush=True)
            results.append(_build_session(spec))
        return results

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(_build_session, spec): spec.day for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"=== CAUSAL MASTER COMPLETE {len(results):02d}/{len(specs):02d} "
                f"{result['session_date']} ===",
                flush=True,
            )
    finally:
        # Explicit rather than context-manager implicit: no combination,
        # validation, or Windows directory rename can begin before every child
        # has exited and released its Parquet writers.
        executor.shutdown(wait=True, cancel_futures=False)
    return results


def build_master_tape(
    *, data_root: Path = DATA_ROOT, output_root: Path = OUTPUT_ROOT,
    workers: int = 1, resume: bool = False,
) -> dict[str, Any]:
    """Run the one authorized heavy pass; callers must invoke this manually."""
    if output_root.exists():
        raise FileExistsError(f"immutable causal master output already exists: {output_root}")
    if workers < 1:
        raise CausalMasterTapeError("workers must be positive")
    if v3_contract_sha256() != V3_CONTRACT_SHA256:
        raise CausalMasterTapeError("frozen V3 contract hash mismatch")
    if v4_contract_sha256() != V4_CONTRACT_SHA256:
        raise CausalMasterTapeError("frozen V4 contract hash mismatch")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists() and not resume:
        raise CausalMasterTapeError(f"staging directory already exists; pass --resume after inspection: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    preflight = parent.verify_data_preflight(data_root, verify_hashes=True)
    if preflight["status"] != "ALL_42_SESSIONS_DATA_SUFFICIENT" or preflight["sufficient_session_count"] != 42:
        raise CausalMasterTapeError("frozen 42-session data preflight failed")
    if not parent.audit_degraded_nov28_source(data_root, preflight).get("usable"):
        raise CausalMasterTapeError("November 28 prior-RTH source no longer passes its frozen source-only audit")
    specs = _session_specs(preflight, data_root, staging)
    results = _run_session_builds(specs, workers)
    results.sort(key=lambda row: str(row["session_date"]))
    combined = _combine_session_artifacts(staging, results)
    interactions = combined["interaction_rows"]
    indexes = combined["index_rows"]
    if len(interactions) != EXPECTED_INTERACTIONS:
        raise CausalMasterTapeError(
            f"expected exactly {EXPECTED_INTERACTIONS} pre-quality interactions, got {len(interactions)}"
        )
    calendar = {
        "calendar_semantics_version": CALENDAR_SEMANTICS_VERSION,
        "target_sessions": [spec.day for spec in specs],
        "canonical_rth": "09:30-16:00 America/New_York",
        "normal_winter_rth_utc": "14:30-21:00",
        "normal_winter_maintenance_utc": "22:00-23:00",
        "nominal_hard_flat_utc": "22:45",
        "normal_winter_effective_hard_flat_utc": "22:00",
        "dec_24_effective_hard_flat_utc": "18:15",
        "nov_28_source_classification": "USABLE_WITH_DEGRADED_SOURCE_WARNING",
        "sessions": [asdict(spec) | {"staging_root": None} for spec in specs],
    }
    _write_json(staging / "calendar.json", calendar)
    metadata = {
        "artifact_kind": "CAUSAL_MASTER_TAPE",
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "ordering_rule_version": ORDERING_RULE_VERSION,
        "ordering_rule": ["timestamp_ns ASC", "MES priority 0 before ES priority 1", "calendar priority 2", "source_index ASC"],
        "v3_strategy_id": V3_STRATEGY_ID, "v3_contract_sha256": V3_CONTRACT_SHA256,
        "v4_strategy_id": V4_STRATEGY_ID, "v4_contract_sha256": V4_CONTRACT_SHA256,
        "source_manifests": _source_metadata(preflight),
        "interaction_population": "EVERY_COMPLETED_PRIOR_RTH_POC_INTERACTION_BEFORE_QUALITY_GATE",
        "score_components": list(SCORE_FIELDS),
        "default_weights": {name: str(value) for name, value in DEFAULT_WEIGHTS.items()},
        "frozen_penalty_weight": V2_CONFIG.false_refill_penalty_weight,
        "supported_thresholds": [str(value) for value in ALLOWED_THRESHOLDS],
        "future_weight_grid_contract": "five components >=0.05, exact 0.05 increments, sum exactly 1.00",
        "threshold_matrix_executed": False, "weight_grid_generated_or_evaluated": False,
        "network_calls": 0, "downloads": 0, "existing_trade_rows_used_as_counterfactuals": False,
    }
    _write_json(staging / "metadata.json", metadata)
    gates = reproduction_gates(staging)
    _write_json(staging / "reproduction-gates.json", gates)
    elapsed = time.monotonic() - started
    summary = {
        "status": "CAUSAL_MASTER_TAPE_VALID",
        "completed_interactions": len(interactions),
        "indexed_interactions": len(indexes),
        "sessions": len(results),
        "decoded_source_records": sum(int(row["decoded_source_records"]) for row in results),
        "stored_events": sum(int(row["stored_events"]) for row in results),
        "confirmed_interactions": sum(int(row["confirmed_interactions"]) for row in results),
        "entry_probe_interactions": sum(int(row["entry_probe_interactions"]) for row in results),
        "heavy_dbn_passes": 1,
        "runtime_seconds": elapsed,
        "approx_peak_memory": "bounded by one adapter/book per worker plus 50,000 event rows per writer",
        "output_bytes_before_summary": _tree_size(staging),
        "reproduction_status": gates["status"],
        "threshold_performance_matrix_executed": False,
        "weight_grid_generated_or_evaluated": False,
    }
    _write_json(staging / "build-summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_diagnostic_report(summary), encoding="utf-8")
    published = finalize_building_root(building_root=staging, output_root=output_root)
    return summary | {"output_root": str(output_root), "publish_status": published["status"]}


def _cli_result(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="perform the one heavy local DBN pass")
    build.add_argument("--data-root", type=Path, default=DATA_ROOT)
    build.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    build.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    build.add_argument("--resume", action="store_true")
    reproduce = subparsers.add_parser("reproduce", help="run only frozen V3/V4 gates from the completed tape")
    reproduce.add_argument("--master-root", type=Path, default=OUTPUT_ROOT)
    finalize = subparsers.add_parser("finalize", help="validate and publish a completed .building tree without DBNs")
    finalize.add_argument("--building-root", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_master_tape(
                data_root=args.data_root, output_root=args.output_root, workers=args.workers, resume=args.resume,
            )
        elif args.command == "reproduce":
            result = reproduction_gates(args.master_root)
        else:
            result = finalize_building_root(building_root=args.building_root, output_root=args.output_root)
        _cli_result(result)
    except (CausalMasterTapeError, historical.HistoricalReplayError, parent.DecJanReplayError,
            native.V3FreshReplayError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
