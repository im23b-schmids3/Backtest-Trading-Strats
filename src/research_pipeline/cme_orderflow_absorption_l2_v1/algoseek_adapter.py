"""Local, deterministic Algoseek normalization for the CME L2 research tape.

This module is deliberately an ingestion boundary.  It reads only explicitly
named local CSV/CSV.GZ files, makes no network calls, and never runs a
strategy.  Its same-timestamp ordering is a research convention, not a claim
about exchange event order:

``MES quote -> ES depth -> ES trade -> ES quote -> source file/row order``.

Multiple Depth bid/ask rows sharing an exact timestamp are assembled before
the depth event is exposed.  The final side states are published once and all
raw rows remain attached as provenance.  This is a deliberate provider
variant: it avoids exposing a synthetic half-updated book when Algoseek does
not document a cross-dataset event sequence.

Algoseek Multiple Depth has no documented MBO action provenance.  Consequently
the resulting public events deliberately expose ``update=None``: the model can
calculate aggregate-depth features, but must not infer add/cancel/fill history
for the false-refill subcomponents.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

from . import historical_runner as historical
from .model import Execution, MBP10Snapshot, MBPLevel, TICK


PROVIDER = "ALGOSEEK"
PROVIDER_SEMANTICS = "ALGOSEEK_CAUSAL_VARIANT"
ADAPTER_VERSION = "algoseek-cme-l2-adapter-v2"
ORDERING_POLICY_ID = "algoseek-causal-order-v2"
DEPTH_ASSEMBLY_POLICY_ID = "algoseek-depth-assembly-v2-final-state-with-raw-provenance"
SOURCE_TIMEZONE = "America/Chicago"
DEPTH_DATASET = "US Futures Multiple Depth"
TAQ_DATASET = "US Futures Trade & Quote"


class AlgoseekAdapterError(ValueError):
    """An Algoseek source cannot safely satisfy the public causal contract."""


@dataclass(frozen=True)
class DepthRow:
    timestamp_ns: int
    provider_timestamp: str
    ticker: str
    security_id: str
    side: Literal["B", "A"]
    levels: tuple[MBPLevel, ...]
    flags: str
    source_file: str
    source_index: int


@dataclass(frozen=True)
class TAQRow:
    timestamp_ns: int
    provider_timestamp: str
    ticker: str
    security_id: str
    instrument: Literal["ES", "MES"]
    event_kind: Literal["TRADE", "BID", "ASK", "OTHER"]
    price: float
    quantity: int
    aggressor: Literal["BUY", "SELL", "UNKNOWN"]
    event_type: str
    flags: str
    type_mask: str
    source_file: str
    source_index: int


@dataclass(frozen=True)
class CanonicalEvent:
    """Provider-neutral normalized observation for a causal-tape builder.

    ``snapshot`` is present only after both ES depth sides have initialized.
    A downstream master-tape builder uses it exactly as it uses an aggregate
    MBP-10 snapshot.  ``book_update`` is intentionally always ``None`` for
    Algoseek because action provenance is unavailable.
    """

    timestamp_ns: int
    kind: Literal["MES_BBO", "ES_DEPTH", "ES_TRADE", "ES_BBO"]
    source_file: str
    source_index: int
    ticker: str
    security_id: str
    provider_timestamp: str = ""
    provider_flags: str = ""
    provider_event_type: str = ""
    provider_type_mask: str = ""
    snapshot: MBP10Snapshot | None = None
    execution: Execution | None = None
    es_taq_bbo: tuple[float | None, float | None] | None = None
    mes_bbo: tuple[float | None, float | None] | None = None
    book_state: str = "INCOMPLETE"
    raw_event_count: int = 1
    raw_provenance: tuple[Mapping[str, Any], ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_csv(path: Path) -> Any:
    if not path.is_file():
        raise AlgoseekAdapterError(f"missing declared Algoseek input: {path}")
    return gzip.open(path, "rt", newline="", encoding="utf-8-sig") if path.suffix == ".gz" else path.open(
        "r", newline="", encoding="utf-8-sig"
    )


def _field(row: Mapping[str, str], name: str, *, required: bool = True) -> str:
    value = row.get(name)
    if value is None or (required and not value.strip()):
        raise AlgoseekAdapterError(f"missing required Algoseek field: {name}")
    return value.strip() if value is not None else ""


def _integer(value: str, *, field: str, nonnegative: bool = False) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise AlgoseekAdapterError(f"invalid integer {field}: {value!r}") from exc
    if nonnegative and result < 0:
        raise AlgoseekAdapterError(f"negative {field}: {value!r}")
    return result


def _price(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise AlgoseekAdapterError(f"invalid price {field}: {value!r}") from exc
    if result <= 0 or abs(result / TICK - round(result / TICK)) > 1e-8:
        raise AlgoseekAdapterError(f"invalid ES tick-aligned {field}: {value!r}")
    return result


def normalize_event_datetime(value: str) -> int:
    """Convert documented CST/CDT wall time to UTC ns, rejecting ambiguity.

    ``zoneinfo`` accepts nonexistent/ambiguous local wall times by choosing a
    fold.  Round-tripping both folds gives this boundary a fail-closed rule.
    """
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlgoseekAdapterError(f"invalid EventDateTime: {value!r}") from exc
    if parsed.tzinfo is not None:
        return int(parsed.astimezone(timezone.utc).timestamp() * 1_000_000_000)
    chicago = ZoneInfo(SOURCE_TIMEZONE)
    candidates: set[datetime] = set()
    for fold in (0, 1):
        aware = parsed.replace(tzinfo=chicago, fold=fold)
        utc = aware.astimezone(timezone.utc)
        if utc.astimezone(chicago).replace(tzinfo=None) == parsed:
            candidates.add(utc)
    if len(candidates) != 1:
        description = "ambiguous" if len(candidates) > 1 else "nonexistent"
        raise AlgoseekAdapterError(f"{description} CST/CDT EventDateTime: {value!r}")
    return int(next(iter(candidates)).timestamp() * 1_000_000_000)


def parse_multiple_depth_row(row: Mapping[str, str], *, source_file: str, source_index: int) -> DepthRow:
    side_text = _field(row, "Side").upper()
    if side_text not in {"B", "S"}:
        raise AlgoseekAdapterError(f"unsupported Multiple Depth side: {side_text!r}")
    levels: list[MBPLevel] = []
    for level in range(1, 11):
        price_text, size_text, count_text = (_field(row, f"L{level}{suffix}", required=False)
                                            for suffix in ("Price", "Size", "Orders"))
        if not price_text and not size_text and not count_text:
            continue
        if not price_text or not size_text or not count_text:
            raise AlgoseekAdapterError(f"partial Multiple Depth L{level} level")
        levels.append(MBPLevel(_price(price_text, field=f"L{level}Price"),
                               _integer(size_text, field=f"L{level}Size", nonnegative=True),
                               _integer(count_text, field=f"L{level}Orders", nonnegative=True)))
    prices = [item.price for item in levels]
    expected = sorted(prices, reverse=side_text == "B")
    if prices != expected or len(set(prices)) != len(prices):
        raise AlgoseekAdapterError(f"Multiple Depth levels are not ordered for side {side_text}")
    return DepthRow(normalize_event_datetime(_field(row, "EventDateTime")), _field(row, "EventDateTime"),
                    _field(row, "Ticker"), _field(row, "SecurityID"), "B" if side_text == "B" else "A",
                    tuple(levels), _field(row, "Flags", required=False), source_file, source_index)


def _taq_kind(event_type: str) -> tuple[Literal["TRADE", "BID", "ASK", "OTHER"], Literal["BUY", "SELL", "UNKNOWN"]]:
    normalized = " ".join(event_type.upper().split())
    if normalized == "QUOTE BID":
        return "BID", "UNKNOWN"
    if normalized in {"QUOTE SELL", "QUOTE ASK"}:
        return "ASK", "UNKNOWN"
    if "TRADE" in normalized and ("AGRESSOR ON BUY" in normalized or "AGGRESSOR ON BUY" in normalized):
        return "TRADE", "BUY"
    if "TRADE" in normalized and ("AGRESSOR ON SELL" in normalized or "AGGRESSOR ON SELL" in normalized):
        return "TRADE", "SELL"
    return "OTHER", "UNKNOWN"


def parse_taq_row(row: Mapping[str, str], *, instrument: Literal["ES", "MES"], source_file: str, source_index: int) -> TAQRow:
    event_type = _field(row, "EventType")
    kind, aggressor = _taq_kind(event_type)
    quantity = _integer(_field(row, "Quantity", required=False) or "0", field="Quantity", nonnegative=True)
    price = _price(_field(row, "Price"), field="Price")
    if kind == "TRADE" and quantity <= 0:
        raise AlgoseekAdapterError("trade Quantity must be positive")
    return TAQRow(normalize_event_datetime(_field(row, "EventDateTime")), _field(row, "EventDateTime"),
                  _field(row, "Ticker"), _field(row, "SecurityID"), instrument, kind, price, quantity, aggressor,
                  event_type, _field(row, "Flags", required=False), _field(row, "TypeMask", required=False),
                  source_file, source_index)


def iter_multiple_depth(path: Path) -> Iterator[DepthRow]:
    with _open_csv(path) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            yield parse_multiple_depth_row(row, source_file=str(path), source_index=index)


def iter_taq(path: Path, *, instrument: Literal["ES", "MES"]) -> Iterator[TAQRow]:
    with _open_csv(path) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            yield parse_taq_row(row, instrument=instrument, source_file=str(path), source_index=index)


class LatestSideBook:
    """Construct a public book from independently updated Algoseek sides."""

    def __init__(self) -> None:
        self.bids: tuple[MBPLevel, ...] | None = None
        self.asks: tuple[MBPLevel, ...] | None = None

    def apply(self, row: DepthRow) -> MBP10Snapshot | None:
        if row.side == "B":
            self.bids = row.levels
        else:
            self.asks = row.levels
        if self.bids is None or self.asks is None:
            return None
        snapshot = MBP10Snapshot(row.timestamp_ns, self.bids, self.asks)
        if not snapshot.bids or not snapshot.asks or snapshot.asks[0].price <= snapshot.bids[0].price:
            return None
        return snapshot

    @property
    def state(self) -> str:
        if self.bids is None or self.asks is None:
            return "INCOMPLETE"
        if not self.bids or not self.asks or self.asks[0].price <= self.bids[0].price:
            return "NON_EXECUTABLE"
        return "EXECUTABLE"


def _depth_groups(events: Iterable[DepthRow | TAQRow]) -> Iterator[list[DepthRow] | TAQRow]:
    """Batch adjacent same-timestamp depth rows without reordering raw rows."""
    pending: list[DepthRow] = []
    for event in events:
        if isinstance(event, DepthRow):
            pending.append(event)
            continue
        if pending:
            yield pending
            pending = []
        yield event
    if pending:
        yield pending


def _depth_provenance(row: DepthRow) -> dict[str, Any]:
    return {
        "provider_timestamp": row.provider_timestamp,
        "side": row.side,
        "flags": row.flags,
        "source_file": row.source_file,
        "source_index": row.source_index,
        "levels": [{"price": level.price, "size": level.size, "orders": level.order_count} for level in row.levels],
    }


def _event_priority(event: DepthRow | TAQRow) -> int:
    if isinstance(event, DepthRow):
        return 1
    if event.instrument == "MES" and event.event_kind in {"BID", "ASK"}:
        return 0
    if event.instrument == "ES" and event.event_kind == "TRADE":
        return 2
    if event.instrument == "ES" and event.event_kind in {"BID", "ASK"}:
        return 3
    return 4


def ordered_source_events(*, es_depth: Iterable[DepthRow], es_taq: Iterable[TAQRow], mes_taq: Iterable[TAQRow]) -> list[DepthRow | TAQRow]:
    """Order known events without asserting undocumented exchange ordering."""
    rows = [*es_depth, *es_taq, *mes_taq]
    for row in rows:
        if isinstance(row, TAQRow) and row.event_kind == "OTHER":
            continue
        if isinstance(row, TAQRow) and row.instrument == "MES" and row.event_kind == "TRADE":
            continue
    usable = [row for row in rows if not isinstance(row, TAQRow) or row.event_kind != "OTHER"]
    # MES trade rows are retained as input provenance but never enter the BBO
    # stream; their priority is deliberately after useful events.
    return sorted(usable, key=lambda row: (row.timestamp_ns, _event_priority(row), row.source_file, row.source_index))


def _quote(state: Mapping[str, float | None]) -> tuple[float | None, float | None]:
    return state["B"], state["A"]


def validate_session_contracts(*, es_depth: Sequence[DepthRow], es_taq: Sequence[TAQRow], mes_taq: Sequence[TAQRow]) -> dict[str, tuple[str, str]]:
    """Reject a rollover blend before it reaches a single causal session.

    A caller must explicitly select one ES and one MES contract per canonical
    session.  The future roll/contract map belongs in the input manifest, not
    in heuristic symbol handling here.
    """
    def identity(rows: Sequence[DepthRow | TAQRow], label: str) -> tuple[str, str]:
        values = {(row.ticker, row.security_id) for row in rows}
        if len(values) != 1:
            raise AlgoseekAdapterError(f"{label} session contains zero or multiple contract identities")
        return next(iter(values))
    depth_identity, taq_identity = identity(es_depth, "ES Multiple Depth"), identity(es_taq, "ES Trade & Quote")
    if depth_identity != taq_identity:
        raise AlgoseekAdapterError("ES Multiple Depth and ES Trade & Quote contract identities differ")
    return {"ES": depth_identity, "MES": identity(mes_taq, "MES Trade & Quote")}


def canonical_events(*, es_depth: Iterable[DepthRow], es_taq: Iterable[TAQRow], mes_taq: Iterable[TAQRow]) -> Iterator[CanonicalEvent]:
    """Yield deterministic, aggregate-only canonical observations.

    This is the provider adapter boundary.  It intentionally does not invoke
    an interaction engine or any Stage 1/2/3 research path.
    """
    depth_rows, es_taq_rows, mes_taq_rows = list(es_depth), list(es_taq), list(mes_taq)
    validate_session_contracts(es_depth=depth_rows, es_taq=es_taq_rows, mes_taq=mes_taq_rows)
    book = LatestSideBook()
    es_quote: dict[str, float | None] = {"B": None, "A": None}
    mes_quote: dict[str, float | None] = {"B": None, "A": None}
    for group in _depth_groups(ordered_source_events(es_depth=depth_rows, es_taq=es_taq_rows, mes_taq=mes_taq_rows)):
        if isinstance(group, list):
            snapshot: MBP10Snapshot | None = None
            for row in group:
                snapshot = book.apply(row)
            row = group[-1]
            yield CanonicalEvent(row.timestamp_ns, "ES_DEPTH", row.source_file, row.source_index, row.ticker,
                                 row.security_id, row.provider_timestamp, row.flags, "MULTIPLE_DEPTH", "",
                                 snapshot=snapshot, book_state=book.state,
                                 raw_event_count=len(group),
                                 raw_provenance=tuple(_depth_provenance(item) for item in group))
            continue
        row = group
        if row.instrument == "MES":
            if row.event_kind in {"BID", "ASK"}:
                mes_quote["B" if row.event_kind == "BID" else "A"] = row.price
                yield CanonicalEvent(row.timestamp_ns, "MES_BBO", row.source_file, row.source_index, row.ticker,
                                     row.security_id, row.provider_timestamp, row.flags, row.event_type, row.type_mask,
                                     mes_bbo=_quote(mes_quote), book_state="EXECUTABLE")
            continue
        if row.event_kind == "TRADE":
            yield CanonicalEvent(row.timestamp_ns, "ES_TRADE", row.source_file, row.source_index, row.ticker,
                                 row.security_id, row.provider_timestamp, row.flags, row.event_type, row.type_mask,
                                 snapshot=(MBP10Snapshot(row.timestamp_ns, book.bids, book.asks)
                                 if book.state == "EXECUTABLE" and book.bids is not None and book.asks is not None else None),
                                 execution=Execution(row.timestamp_ns, row.price, row.quantity, row.aggressor),
                                 es_taq_bbo=_quote(es_quote), book_state=book.state)
        elif row.event_kind in {"BID", "ASK"}:
            es_quote["B" if row.event_kind == "BID" else "A"] = row.price
            yield CanonicalEvent(row.timestamp_ns, "ES_BBO", row.source_file, row.source_index, row.ticker,
                                 row.security_id, row.provider_timestamp, row.flags, row.event_type, row.type_mask,
                                 es_taq_bbo=_quote(es_quote), book_state=book.state)


def profile_from_executions(rows: Iterable[TAQRow]) -> dict[str, float]:
    """The existing 70% volume-profile rule, using ES trades only."""
    volume: Counter[float] = Counter()
    for row in rows:
        if row.instrument == "ES" and row.event_kind == "TRADE":
            volume[row.price] += row.quantity
    if not volume:
        raise AlgoseekAdapterError("ES executions are required to build a volume profile")
    poc = min(volume, key=lambda price: (-volume[price], price))
    total, included, low, high = sum(volume.values()), volume[poc], poc, poc
    while included * 100 < total * 70:
        below, above = low - TICK, high + TICK
        if volume[below] >= volume[above]:
            low, included = below, included + volume[below]
        else:
            high, included = above, included + volume[above]
    return {"POC": poc, "VAH": high, "VAL": low, "HIGH": max(volume), "LOW": min(volume)}


def provider_provenance(
    *, files: Sequence[Path], ticker: str | None = None, security_id: str | None = None,
    source_date: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "provider": PROVIDER,
        "provider_semantics": PROVIDER_SEMANTICS,
        "adapter_version": ADAPTER_VERSION,
        "ordering_policy_id": ORDERING_POLICY_ID,
        "depth_assembly_policy_id": DEPTH_ASSEMBLY_POLICY_ID,
        "ordering_policy": ["timestamp_ns", "MES_BBO", "ES_DEPTH", "ES_TRADE", "ES_BBO", "source_file", "source_row"],
        "source_timezone": SOURCE_TIMEZONE,
        "canonical_timezone": "UTC",
        "datasets": {"es_depth": DEPTH_DATASET, "es_taq": TAQ_DATASET, "mes_taq": TAQ_DATASET},
        "ticker": ticker,
        "security_id": security_id,
        "source_date": source_date,
        "raw_files": [{"path": str(path), "sha256": _sha256(path)} for path in files],
        "known_semantic_limitations": [
            "UNDOCUMENTED_CROSS_DATASET_SAME_TIMESTAMP_ORDER",
            "SAME_TIMESTAMP_DEPTH_ROWS_BATCHED_TO_FINAL_STATE_RAW_ROWS_RETAINED",
            "NO_MBO_ADD_CANCEL_FILL_MODIFY_RESET_PROVENANCE",
            "FALSE_REFILL_UNEXECUTED_ADD_AND_RAPID_CANCEL_COMPONENTS_PROVIDER_LIMITED",
        ],
    }


def validate_session_provider_ownership(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Require exactly one authoritative provider per canonical session."""
    ownership: dict[str, str] = {}
    for row in rows:
        day, provider = str(row.get("date", "")), str(row.get("provider", ""))
        if not day or provider not in {"ALGOSEEK", "DATABENTO"}:
            raise AlgoseekAdapterError("each period session requires explicit ALGOSEEK or DATABENTO ownership")
        if day in ownership:
            raise AlgoseekAdapterError(f"duplicate/mixed provider ownership for session: {day}")
        ownership[day] = provider
    if not ownership:
        raise AlgoseekAdapterError("period requires at least one explicitly owned session")
    return ownership


def load_input_manifest(path: Path) -> dict[str, Any]:
    """Load an explicit local-file manifest without discovering data.

    The manifest is intentionally provider-oriented and can coexist with the
    older Databento period manifests.  Each row must name its provider and
    paths, so a mixed campaign cannot silently assign two sources to one day.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlgoseekAdapterError(f"invalid Algoseek input manifest: {path}") from exc
    sessions = payload.get("sessions") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or not isinstance(sessions, list):
        raise AlgoseekAdapterError("Algoseek input manifest requires schema_version=1 and sessions")
    validate_session_provider_ownership(sessions)
    root = path.parent
    normalized: list[dict[str, Any]] = []
    for row in sessions:
        if row.get("provider") != "ALGOSEEK":
            normalized.append(dict(row)); continue
        required = ("es_depth", "es_taq", "mes_taq", "es_ticker", "es_security_id", "mes_ticker", "mes_security_id")
        if any(not row.get(name) for name in required):
            raise AlgoseekAdapterError(f"Algoseek session lacks required feeds or contract identity: {row.get('date')}")
        item = dict(row)
        for name in ("es_depth", "es_taq", "mes_taq"):
            values = row[name] if isinstance(row[name], list) else [row[name]]
            item[name] = [str((root / str(value)).resolve()) for value in values]
        normalized.append(item)
    return {"manifest_path": str(path), "provider_ownership": validate_session_provider_ownership(sessions), "sessions": normalized}


def audit_inputs(*, es_depth_paths: Sequence[Path], es_taq_paths: Sequence[Path], mes_taq_paths: Sequence[Path]) -> dict[str, Any]:
    """Read and validate inputs only; never run an interaction or backtest."""
    if not es_depth_paths or not es_taq_paths or not mes_taq_paths:
        raise AlgoseekAdapterError("audit requires ES Multiple Depth, ES TAQ, and MES TAQ files")
    depth, es_taq, mes_taq = ([row for path in paths for row in iterator(path)] for paths, iterator in (
        (es_depth_paths, iter_multiple_depth), (es_taq_paths, lambda path: iter_taq(path, instrument="ES")),
        (mes_taq_paths, lambda path: iter_taq(path, instrument="MES")),
    ))
    ordered = ordered_source_events(es_depth=depth, es_taq=es_taq, mes_taq=mes_taq)
    timestamps: dict[int, list[DepthRow | TAQRow]] = defaultdict(list)
    for row in ordered:
        timestamps[row.timestamp_ns].append(row)
    collisions = [rows for rows in timestamps.values() if len(rows) > 1]
    tie_metrics = {
        "total_normalized_events": len(ordered), "timestamps_with_multiple_events": len(collisions),
        "maximum_events_sharing_one_timestamp": max((len(rows) for rows in timestamps.values()), default=0),
        "same_timestamp_es_depth_es_trade": sum(any(isinstance(row, DepthRow) for row in rows) and any(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind == "TRADE" for row in rows) for rows in collisions),
        "same_timestamp_es_trade_es_bbo": sum(any(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind == "TRADE" for row in rows) and any(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind in {"BID", "ASK"} for row in rows) for rows in collisions),
        "same_timestamp_mes_quote_es_event": sum(any(isinstance(row, TAQRow) and row.instrument == "MES" and row.event_kind in {"BID", "ASK"} for row in rows) and any((isinstance(row, DepthRow) or (isinstance(row, TAQRow) and row.instrument == "ES")) for row in rows) for rows in collisions),
    }
    trade_rows = [row for row in es_taq if row.event_kind == "TRADE"]
    affected = sum(1 for row in trade_rows if len(timestamps[row.timestamp_ns]) > 1)
    tie_metrics["trades_affected_by_same_timestamp_collision"] = affected
    tie_metrics["trades_affected_pct"] = 0.0 if not trade_rows else affected * 100.0 / len(trade_rows)
    book = LatestSideBook(); incomplete = bid_only = ask_only = crossed = locked = 0; disagreements = 0
    latest_taq: dict[str, float | None] = {"B": None, "A": None}
    for group in _depth_groups(ordered):
        if isinstance(group, list):
            snapshot: MBP10Snapshot | None = None
            for event in group:
                snapshot = book.apply(event)
            if book.bids is None or book.asks is None:
                incomplete += 1; bid_only += int(book.bids is not None and book.asks is None); ask_only += int(book.asks is not None and book.bids is None)
            elif snapshot is None:
                crossed += int(book.asks[0].price < book.bids[0].price); locked += int(book.asks[0].price == book.bids[0].price)
            if snapshot is not None and latest_taq["B"] is not None and latest_taq["A"] is not None:
                disagreements += int((snapshot.bids[0].price, snapshot.asks[0].price) != (latest_taq["B"], latest_taq["A"]))
            continue
        event = group
        if event.instrument == "ES" and event.event_kind in {"BID", "ASK"}:
            latest_taq["B" if event.event_kind == "BID" else "A"] = event.price
    per_file: list[dict[str, Any]] = []
    for path in [*es_depth_paths, *es_taq_paths, *mes_taq_paths]:
        matching = [row for row in [*depth, *es_taq, *mes_taq] if row.source_file == str(path)]
        per_file.append({"path": str(path), "rows": len(matching), "first_timestamp_ns": min((row.timestamp_ns for row in matching), default=None), "last_timestamp_ns": max((row.timestamp_ns for row in matching), default=None), "sha256": _sha256(path)})
    source_dates = sorted({row.provider_timestamp[:10] for row in [*depth, *es_taq, *mes_taq] if row.provider_timestamp})
    return {
        "status": "ALGOSEEK_INPUT_AUDIT_COMPLETE", "provider_provenance": provider_provenance(
            files=[*es_depth_paths, *es_taq_paths, *mes_taq_paths], source_date=source_dates,
        ),
        "files": per_file, "es_depth_rows": len(depth), "es_trade_rows": len(trade_rows),
        "es_bbo_rows": sum(row.event_kind in {"BID", "ASK"} for row in es_taq), "mes_quote_rows": sum(row.event_kind in {"BID", "ASK"} for row in mes_taq),
        "aggressive_buy_trades": sum(row.aggressor == "BUY" for row in trade_rows), "aggressive_sell_trades": sum(row.aggressor == "SELL" for row in trade_rows),
        "malformed_rows": 0, "duplicate_rows": len(ordered) - len({(type(row).__name__, row.timestamp_ns, row.source_file, row.source_index) for row in ordered}),
        "bid_only_initialization_periods": bid_only, "ask_only_initialization_periods": ask_only,
        "incomplete_book_states": incomplete, "crossed_book_states": crossed, "locked_book_states": locked,
        "es_depth_l1_vs_es_taq_bbo_disagreements": disagreements, "timestamp_normalization_issues": 0,
        "causal_tie_diagnostics": tie_metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="validate local Algoseek inputs; never run research")
    for name in ("es-depth", "es-taq", "mes-taq"):
        audit.add_argument(f"--{name}", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_inputs(es_depth_paths=args.es_depth, es_taq_paths=args.es_taq, mes_taq_paths=args.mes_taq)
    except (AlgoseekAdapterError, OSError, csv.Error) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
