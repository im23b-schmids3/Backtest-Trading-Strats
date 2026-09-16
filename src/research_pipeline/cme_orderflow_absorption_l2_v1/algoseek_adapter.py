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
import os
import resource
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    aggression_eligible: bool = False


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


@dataclass
class StreamingMetrics:
    """Small, session-scoped counters for a bounded streaming pass."""

    es_depth_rows: int = 0
    es_taq_rows: int = 0
    mes_taq_rows: int = 0
    es_provider_trades: int = 0
    explicit_aggressor_trades: int = 0
    generic_trades: int = 0
    canonical_depth_states: int = 0
    canonical_events_emitted: int = 0
    max_depth_timestamp_rows: int = 0
    max_merge_pending_events: int = 0
    raw_first_timestamp_ns: int | None = None
    raw_last_timestamp_ns: int | None = None

    def as_dict(self) -> dict[str, int]:
        return {
            "es_depth_rows": self.es_depth_rows,
            "es_taq_rows": self.es_taq_rows,
            "mes_taq_rows": self.mes_taq_rows,
            "es_provider_trades": self.es_provider_trades,
            "explicit_aggressor_trades": self.explicit_aggressor_trades,
            "generic_trades": self.generic_trades,
            "canonical_depth_states": self.canonical_depth_states,
            "canonical_events_emitted": self.canonical_events_emitted,
            "max_depth_timestamp_rows": self.max_depth_timestamp_rows,
            "max_merge_pending_events": self.max_merge_pending_events,
            "raw_first_timestamp_ns": self.raw_first_timestamp_ns,
            "raw_last_timestamp_ns": self.raw_last_timestamp_ns,
        }


class StreamingProfile:
    """Incremental ES execution profile; generic trades remain profile-only."""

    def __init__(self) -> None:
        self.volume: Counter[float] = Counter()
        self.total_volume = 0
        self.high: float | None = None
        self.low: float | None = None

    def observe(self, row: TAQRow) -> None:
        if row.instrument != "ES" or row.event_kind != "TRADE" or row.quantity <= 0:
            return
        self.volume[row.price] += row.quantity
        self.total_volume += row.quantity
        self.high = row.price if self.high is None else max(self.high, row.price)
        self.low = row.price if self.low is None else min(self.low, row.price)

    def result(self) -> dict[str, float]:
        if not self.volume or self.high is None or self.low is None:
            raise AlgoseekAdapterError("ES executions are required to build a volume profile")
        poc = min(self.volume, key=lambda price: (-self.volume[price], price))
        included, low, high = self.volume[poc], poc, poc
        while included * 100 < self.total_volume * 70:
            below, above = low - TICK, high + TICK
            if self.volume[below] >= self.volume[above]:
                low, included = below, included + self.volume[below]
            else:
                high, included = above, included + self.volume[above]
        return {"POC": poc, "VAH": high, "VAL": low, "HIGH": self.high, "LOW": self.low}


class _RawBBOCoverage:
    """Time-weighted MES BBO availability from the provider's raw event stream."""

    def __init__(self) -> None:
        self.bid: float | None = None
        self.ask: float | None = None
        self.previous_timestamp: int | None = None
        self.covered_ns = 0

    def observe(self, row: TAQRow) -> None:
        if self.previous_timestamp is not None and self.bid is not None and self.ask is not None:
            self.covered_ns += row.timestamp_ns - self.previous_timestamp
        if row.event_kind == "BID":
            self.bid = row.price
        elif row.event_kind == "ASK":
            self.ask = row.price
        self.previous_timestamp = row.timestamp_ns

    def result(self, *, session_start: int | None, session_end: int | None) -> float:
        if session_start is None or session_end is None or session_end <= session_start:
            return 0.0
        if self.previous_timestamp is not None and self.bid is not None and self.ask is not None:
            self.covered_ns += max(0, session_end - self.previous_timestamp)
            self.previous_timestamp = session_end
        return self.covered_ns / (session_end - session_start)


def _logical_cme_session_bounds(timestamp_ns: int) -> tuple[int, int]:
    """Return the documented 17:00 CT to 16:00 CT logical session enclosing a row."""
    chicago = ZoneInfo(SOURCE_TIMEZONE)
    local = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).astimezone(chicago)
    session_day = local.date() if local.hour >= 17 else local.date() - timedelta(days=1)
    start = datetime.combine(session_day, datetime.min.time(), tzinfo=chicago).replace(hour=17)
    end = datetime.combine(session_day + timedelta(days=1), datetime.min.time(), tzinfo=chicago).replace(hour=16)
    return int(start.timestamp() * 1_000_000_000), int(end.timestamp() * 1_000_000_000)


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
        # Algoseek represents unused levels as a complete 0.0000/0/0
        # placeholder.  It is not a populated price level and must not be
        # passed through MBPLevel's positive-price validation.
        if price_text in {"0", "0.0", "0.00", "0.0000"} and size_text == "0" and count_text == "0":
            continue
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
    if normalized == "TRADE":
        # This is a real execution for profiles, but has no documented side.
        return "TRADE", "UNKNOWN"
    return "OTHER", "UNKNOWN"


def parse_taq_row(row: Mapping[str, str], *, instrument: Literal["ES", "MES"], source_file: str, source_index: int) -> TAQRow:
    event_type = _field(row, "EventType")
    kind, aggressor = _taq_kind(event_type)
    quantity = _integer(_field(row, "Quantity", required=False) or "0", field="Quantity", nonnegative=True)
    # Algoseek emits non-market control rows such as EMPTY BOOK FINAL with a
    # zero price.  They are retained for provenance but are not canonical
    # quote/trade events, so zero is valid only for OTHER rows.
    price_text = _field(row, "Price")
    if kind == "OTHER":
        try:
            price = float(price_text)
        except ValueError as exc:
            raise AlgoseekAdapterError(f"invalid price Price: {price_text!r}") from exc
        if price < 0:
            raise AlgoseekAdapterError(f"negative non-market Price: {price_text!r}")
    else:
        price = _price(price_text, field="Price")
    if kind == "TRADE" and aggressor in {"BUY", "SELL"} and quantity <= 0:
        raise AlgoseekAdapterError("explicit-aggressor trade Quantity must be positive")
    return TAQRow(normalize_event_datetime(_field(row, "EventDateTime")), _field(row, "EventDateTime"),
                  _field(row, "Ticker"), _field(row, "SecurityID"), instrument, kind, price, quantity, aggressor,
                  event_type, _field(row, "Flags", required=False), _field(row, "TypeMask", required=False),
                  source_file, source_index, aggressor in {"BUY", "SELL"})


def _iter_csv_rows(path: Path, *, required_fields: Sequence[str]) -> Iterator[tuple[int, Mapping[str, str]]]:
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required_fields):
            raise AlgoseekAdapterError(f"missing required Algoseek CSV header in {path}")
        for index, row in enumerate(reader, start=1):
            yield index, row


def iter_multiple_depth(path: Path) -> Iterator[DepthRow]:
    required = ("EventDateTime", "Ticker", "SecurityID", "Side", "Flags")
    required += tuple(f"L{level}{suffix}" for level in range(1, 11) for suffix in ("Price", "Size", "Orders"))
    for index, row in _iter_csv_rows(path, required_fields=required):
        yield parse_multiple_depth_row(row, source_file=str(path), source_index=index)


def iter_taq(path: Path, *, instrument: Literal["ES", "MES"]) -> Iterator[TAQRow]:
    for index, row in _iter_csv_rows(path, required_fields=("EventDateTime", "Ticker", "SecurityID", "EventType", "Price", "Quantity", "Flags", "TypeMask")):
        yield parse_taq_row(row, instrument=instrument, source_file=str(path), source_index=index)


def _iter_paths(
    paths: Sequence[Path], iterator: Any, *, label: str, metrics: StreamingMetrics | None = None,
    mes_bbo_coverage: _RawBBOCoverage | None = None,
) -> Iterator[DepthRow | TAQRow]:
    """Read declared chunks in caller order, rejecting time reversal at a boundary."""
    if not paths:
        raise AlgoseekAdapterError(f"{label} requires at least one declared input file")
    previous: int | None = None
    for path in paths:
        for row in iterator(path):
            if previous is not None and row.timestamp_ns < previous:
                raise AlgoseekAdapterError(f"{label} timestamps decrease across source/chunk boundary at {row.source_file}:{row.source_index}")
            previous = row.timestamp_ns
            if metrics is not None:
                metrics.raw_first_timestamp_ns = row.timestamp_ns if metrics.raw_first_timestamp_ns is None else min(metrics.raw_first_timestamp_ns, row.timestamp_ns)
                metrics.raw_last_timestamp_ns = row.timestamp_ns if metrics.raw_last_timestamp_ns is None else max(metrics.raw_last_timestamp_ns, row.timestamp_ns)
                if label == "ES Multiple Depth":
                    metrics.es_depth_rows += 1
                elif label == "ES Trade & Quote":
                    metrics.es_taq_rows += 1
                    if row.event_kind == "TRADE":
                        metrics.es_provider_trades += 1
                        if row.aggression_eligible:
                            metrics.explicit_aggressor_trades += 1
                        else:
                            metrics.generic_trades += 1
                else:
                    metrics.mes_taq_rows += 1
            if mes_bbo_coverage is not None:
                assert isinstance(row, TAQRow)
                mes_bbo_coverage.observe(row)
            yield row


def iter_multiple_depth_paths(paths: Sequence[Path], *, metrics: StreamingMetrics | None = None) -> Iterator[DepthRow]:
    yield from _iter_paths(paths, iter_multiple_depth, label="ES Multiple Depth", metrics=metrics)  # type: ignore[misc]


def iter_taq_paths(
    paths: Sequence[Path], *, instrument: Literal["ES", "MES"], metrics: StreamingMetrics | None = None,
    mes_bbo_coverage: _RawBBOCoverage | None = None,
) -> Iterator[TAQRow]:
    label = "ES Trade & Quote" if instrument == "ES" else "MES Trade & Quote"
    yield from _iter_paths(paths, lambda path: iter_taq(path, instrument=instrument), label=label, metrics=metrics,
                           mes_bbo_coverage=mes_bbo_coverage)  # type: ignore[misc]


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
    """Batch only adjacent depth rows sharing one exact normalized timestamp."""
    pending: list[DepthRow] = []
    for event in events:
        if isinstance(event, DepthRow):
            if pending and event.timestamp_ns != pending[0].timestamp_ns:
                yield pending
                pending = []
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
    session.  Algoseek SecurityIDs may be dataset-specific, so the future
    roll/contract map belongs in the input manifest, not in heuristic symbol
    handling here.
    """
    def identity(rows: Sequence[DepthRow | TAQRow], label: str) -> tuple[str, str]:
        values = {(row.ticker, row.security_id) for row in rows}
        if len(values) != 1:
            raise AlgoseekAdapterError(f"{label} session contains zero or multiple contract identities")
        return next(iter(values))
    depth_identity, taq_identity = identity(es_depth, "ES Multiple Depth"), identity(es_taq, "ES Trade & Quote")
    # Algoseek may assign dataset-specific SecurityIDs to the same contract.
    # The contract symbol must agree, while each feed's identity is retained.
    if depth_identity[0] != taq_identity[0]:
        raise AlgoseekAdapterError("ES Multiple Depth and ES Trade & Quote contract tickers differ")
    return {"ES_DEPTH": depth_identity, "ES_TAQ": taq_identity, "MES": identity(mes_taq, "MES Trade & Quote")}


class _IdentityGuard:
    """Validate stream identity without retaining a session's rows."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.value: tuple[str, str] | None = None

    def observe(self, row: DepthRow | TAQRow) -> None:
        candidate = (row.ticker, row.security_id)
        if self.value is None:
            self.value = candidate
        elif candidate != self.value:
            raise AlgoseekAdapterError(f"{self.label} session contains zero or multiple contract identities")

    def require(self) -> tuple[str, str]:
        if self.value is None:
            raise AlgoseekAdapterError(f"{self.label} session contains zero or multiple contract identities")
        return self.value


def _guarded(rows: Iterable[DepthRow | TAQRow], guard: _IdentityGuard) -> Iterator[DepthRow | TAQRow]:
    for row in rows:
        guard.observe(row)
        yield row


@dataclass(frozen=True)
class _DepthBatch:
    rows: tuple[DepthRow, ...]

    @property
    def timestamp_ns(self) -> int:
        return self.rows[-1].timestamp_ns


def _iter_depth_batches(rows: Iterable[DepthRow], *, metrics: StreamingMetrics | None = None) -> Iterator[_DepthBatch]:
    """Retain only one exact-time raw depth group, including an EOF flush."""
    group: list[DepthRow] = []
    current_timestamp: int | None = None
    for row in rows:
        if current_timestamp is None:
            current_timestamp = row.timestamp_ns
        if row.timestamp_ns != current_timestamp:
            ordered = tuple(sorted(group, key=lambda item: (item.source_file, item.source_index)))
            if metrics is not None:
                metrics.max_depth_timestamp_rows = max(metrics.max_depth_timestamp_rows, len(ordered))
            yield _DepthBatch(ordered)
            group.clear()
            current_timestamp = row.timestamp_ns
        group.append(row)
    if group:
        ordered = tuple(sorted(group, key=lambda item: (item.source_file, item.source_index)))
        if metrics is not None:
            metrics.max_depth_timestamp_rows = max(metrics.max_depth_timestamp_rows, len(ordered))
        yield _DepthBatch(ordered)


def _iter_taq_timestamp_rows(rows: Iterable[TAQRow]) -> Iterator[TAQRow]:
    """Order only one TAQ timestamp group by the published causal priority."""
    group: list[TAQRow] = []
    current_timestamp: int | None = None
    for row in rows:
        if current_timestamp is None:
            current_timestamp = row.timestamp_ns
        if row.timestamp_ns != current_timestamp:
            yield from sorted(group, key=lambda item: (_event_priority(item), item.source_file, item.source_index))
            group.clear()
            current_timestamp = row.timestamp_ns
        group.append(row)
    if group:
        yield from sorted(group, key=lambda item: (_event_priority(item), item.source_file, item.source_index))


def _raw_source_key(item: _DepthBatch | TAQRow) -> tuple[int, int, str, int]:
    if isinstance(item, _DepthBatch):
        final = item.rows[-1]
        return item.timestamp_ns, 1, final.source_file, final.source_index
    return item.timestamp_ns, _event_priority(item), item.source_file, item.source_index


def _merge_source_rows(
    sources: Sequence[Iterator[_DepthBatch | TAQRow]], *, metrics: StreamingMetrics | None = None,
) -> Iterator[_DepthBatch | TAQRow]:
    """Merge three sources with at most one lookahead from each source."""
    import heapq

    pending: list[tuple[tuple[int, int, str, int], int, _DepthBatch | TAQRow]] = []
    for index, source in enumerate(sources):
        try:
            item = next(source)
        except StopIteration:
            continue
        heapq.heappush(pending, (_raw_source_key(item), index, item))
    while pending:
        if metrics is not None:
            metrics.max_merge_pending_events = max(metrics.max_merge_pending_events, len(pending))
        _, index, item = heapq.heappop(pending)
        yield item
        try:
            following = next(sources[index])
        except StopIteration:
            continue
        heapq.heappush(pending, (_raw_source_key(following), index, following))


def iter_depth_events(
    rows: Iterable[DepthRow], *, metrics: StreamingMetrics | None = None,
) -> Iterator[CanonicalEvent]:
    """Batch only the current exact timestamp, flushing it at EOF as well."""
    book = LatestSideBook()
    group: list[DepthRow] = []
    current_timestamp: int | None = None

    def flush() -> CanonicalEvent:
        assert group
        snapshot: MBP10Snapshot | None = None
        for item in group:
            snapshot = book.apply(item)
        final = group[-1]
        if metrics is not None:
            metrics.canonical_depth_states += 1
            metrics.max_depth_timestamp_rows = max(metrics.max_depth_timestamp_rows, len(group))
        return CanonicalEvent(final.timestamp_ns, "ES_DEPTH", final.source_file, final.source_index, final.ticker,
                              final.security_id, final.provider_timestamp, final.flags, "MULTIPLE_DEPTH", "",
                              snapshot=snapshot, book_state=book.state, raw_event_count=len(group),
                              raw_provenance=tuple(_depth_provenance(item) for item in group))

    for row in rows:
        if current_timestamp is None:
            current_timestamp = row.timestamp_ns
        if row.timestamp_ns != current_timestamp:
            yield flush()
            group.clear()
            current_timestamp = row.timestamp_ns
        group.append(row)
    if group:
        yield flush()


def _source_key(event: CanonicalEvent) -> tuple[int, int, str, int]:
    priority = {"MES_BBO": 0, "ES_DEPTH": 1, "ES_TRADE": 2, "ES_BBO": 3}[event.kind]
    return event.timestamp_ns, priority, event.source_file, event.source_index


def _merge_canonical_sources(
    sources: Sequence[Iterator[CanonicalEvent]], *, metrics: StreamingMetrics | None = None,
) -> Iterator[CanonicalEvent]:
    """A three-lookahead merge; its pending state never grows with a session."""
    import heapq

    pending: list[tuple[tuple[int, int, str, int], int, CanonicalEvent]] = []
    for index, source in enumerate(sources):
        try:
            event = next(source)
        except StopIteration:
            continue
        heapq.heappush(pending, (_source_key(event), index, event))
    while pending:
        if metrics is not None:
            metrics.max_merge_pending_events = max(metrics.max_merge_pending_events, len(pending))
        _, index, event = heapq.heappop(pending)
        if metrics is not None:
            metrics.canonical_events_emitted += 1
        yield event
        try:
            following = next(sources[index])
        except StopIteration:
            continue
        heapq.heappush(pending, (_source_key(following), index, following))


def _iter_es_taq_events(
    rows: Iterable[TAQRow], *, book: LatestSideBook, profile: StreamingProfile | None,
) -> Iterator[CanonicalEvent]:
    es_quote: dict[str, float | None] = {"B": None, "A": None}
    for row in rows:
        if row.event_kind == "TRADE":
            if profile is not None:
                profile.observe(row)
            # Side-less provider trades are executions only for profile work;
            # sending them to the absorption engine would fabricate aggression.
            if not row.aggression_eligible:
                continue
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


def _iter_mes_bbo_events(rows: Iterable[TAQRow]) -> Iterator[CanonicalEvent]:
    mes_quote: dict[str, float | None] = {"B": None, "A": None}
    for row in rows:
        if row.event_kind not in {"BID", "ASK"}:
            continue
        mes_quote["B" if row.event_kind == "BID" else "A"] = row.price
        yield CanonicalEvent(row.timestamp_ns, "MES_BBO", row.source_file, row.source_index, row.ticker,
                             row.security_id, row.provider_timestamp, row.flags, row.event_type, row.type_mask,
                             mes_bbo=_quote(mes_quote), book_state="EXECUTABLE")


def iter_canonical_events(
    *, es_depth: Iterable[DepthRow], es_taq: Iterable[TAQRow], mes_taq: Iterable[TAQRow],
    profile: StreamingProfile | None = None, metrics: StreamingMetrics | None = None,
) -> Iterator[CanonicalEvent]:
    """Stream provider-neutral events with one depth group and three lookaheads.

    Contract errors can occur while consuming an iterator; callers writing an
    artifact should use :func:`write_canonical_jsonl` so partial output is not
    published.
    """
    depth_guard, es_guard, mes_guard = (_IdentityGuard(label) for label in (
        "ES Multiple Depth", "ES Trade & Quote", "MES Trade & Quote"))
    guarded_depth = _guarded(es_depth, depth_guard)
    guarded_es = _guarded(es_taq, es_guard)
    guarded_mes = _guarded(mes_taq, mes_guard)
    book = LatestSideBook()
    es_quote: dict[str, float | None] = {"B": None, "A": None}
    mes_quote: dict[str, float | None] = {"B": None, "A": None}
    merged = _merge_source_rows((
        _iter_taq_timestamp_rows(guarded_mes),
        _iter_depth_batches(guarded_depth, metrics=metrics),
        _iter_taq_timestamp_rows(guarded_es),
    ), metrics=metrics)
    for item in merged:
        event: CanonicalEvent | None = None
        if isinstance(item, _DepthBatch):
            snapshot: MBP10Snapshot | None = None
            for row in item.rows:
                snapshot = book.apply(row)
            final = item.rows[-1]
            if metrics is not None:
                metrics.canonical_depth_states += 1
            event = CanonicalEvent(final.timestamp_ns, "ES_DEPTH", final.source_file, final.source_index, final.ticker,
                                   final.security_id, final.provider_timestamp, final.flags, "MULTIPLE_DEPTH", "",
                                   snapshot=snapshot, book_state=book.state, raw_event_count=len(item.rows),
                                   raw_provenance=tuple(_depth_provenance(row) for row in item.rows))
        elif item.instrument == "MES":
            if item.event_kind in {"BID", "ASK"}:
                mes_quote["B" if item.event_kind == "BID" else "A"] = item.price
                event = CanonicalEvent(item.timestamp_ns, "MES_BBO", item.source_file, item.source_index, item.ticker,
                                       item.security_id, item.provider_timestamp, item.flags, item.event_type, item.type_mask,
                                       mes_bbo=_quote(mes_quote), book_state="EXECUTABLE")
        elif item.event_kind == "TRADE":
            if profile is not None:
                profile.observe(item)
            if item.aggression_eligible:
                event = CanonicalEvent(item.timestamp_ns, "ES_TRADE", item.source_file, item.source_index, item.ticker,
                                       item.security_id, item.provider_timestamp, item.flags, item.event_type, item.type_mask,
                                       snapshot=(MBP10Snapshot(item.timestamp_ns, book.bids, book.asks)
                                       if book.state == "EXECUTABLE" and book.bids is not None and book.asks is not None else None),
                                       execution=Execution(item.timestamp_ns, item.price, item.quantity, item.aggressor),
                                       es_taq_bbo=_quote(es_quote), book_state=book.state)
        elif item.event_kind in {"BID", "ASK"}:
            es_quote["B" if item.event_kind == "BID" else "A"] = item.price
            event = CanonicalEvent(item.timestamp_ns, "ES_BBO", item.source_file, item.source_index, item.ticker,
                                   item.security_id, item.provider_timestamp, item.flags, item.event_type, item.type_mask,
                                   es_taq_bbo=_quote(es_quote), book_state=book.state)
        if event is not None:
            if metrics is not None:
                metrics.canonical_events_emitted += 1
            yield event
    depth_identity, es_identity, _ = depth_guard.require(), es_guard.require(), mes_guard.require()
    if depth_identity[0] != es_identity[0]:
        raise AlgoseekAdapterError("ES Multiple Depth and ES Trade & Quote contract tickers differ")


def iter_canonical_events_from_paths(
    *, es_depth_paths: Sequence[Path], es_taq_paths: Sequence[Path], mes_taq_paths: Sequence[Path],
    profile: StreamingProfile | None = None, metrics: StreamingMetrics | None = None,
    mes_bbo_coverage: _RawBBOCoverage | None = None,
) -> Iterator[CanonicalEvent]:
    """The real-session API: stream declared raw chunks without concatenation."""
    yield from iter_canonical_events(
        es_depth=iter_multiple_depth_paths(es_depth_paths, metrics=metrics),
        es_taq=iter_taq_paths(es_taq_paths, instrument="ES", metrics=metrics),
        mes_taq=iter_taq_paths(mes_taq_paths, instrument="MES", metrics=metrics, mes_bbo_coverage=mes_bbo_coverage),
        profile=profile, metrics=metrics,
    )


def canonical_events(*, es_depth: Iterable[DepthRow], es_taq: Iterable[TAQRow], mes_taq: Iterable[TAQRow]) -> Iterator[CanonicalEvent]:
    """Compatibility wrapper for the prior API, now streaming its inputs."""
    return iter_canonical_events(es_depth=es_depth, es_taq=es_taq, mes_taq=mes_taq)


def profile_from_executions(rows: Iterable[TAQRow]) -> dict[str, float]:
    """Compatibility helper backed by the same bounded profile accumulator."""
    profile = StreamingProfile()
    for row in rows:
        profile.observe(row)
    return profile.result()


def write_canonical_jsonl(
    *, events: Iterable[CanonicalEvent], output_path: Path, completion_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Publish a fully streamed session artifact atomically after successful EOF.

    This writer intentionally has no append/resume mode: an interruption leaves
    a ``.tmp`` file that is never a valid session artifact, and a later run can
    safely replace it from the declared raw chunks.
    """
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for event in events:
                payload = {
                    "timestamp_ns": event.timestamp_ns, "kind": event.kind, "source_file": event.source_file,
                    "source_index": event.source_index, "ticker": event.ticker, "security_id": event.security_id,
                    "provider_timestamp": event.provider_timestamp, "provider_flags": event.provider_flags,
                    "provider_event_type": event.provider_event_type, "provider_type_mask": event.provider_type_mask,
                    "book_state": event.book_state, "raw_event_count": event.raw_event_count,
                    "raw_provenance": event.raw_provenance,
                    "snapshot": None if event.snapshot is None else {
                        "bids": [(level.price, level.size, level.order_count) for level in event.snapshot.bids],
                        "asks": [(level.price, level.size, level.order_count) for level in event.snapshot.asks],
                    },
                    "execution": None if event.execution is None else {
                        "price": event.execution.price, "size": event.execution.size, "aggressor": event.execution.aggressor,
                    },
                    "es_taq_bbo": event.es_taq_bbo, "mes_bbo": event.mes_bbo,
                }
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, output_path)
        if completion_manifest is not None:
            manifest_path = output_path.with_suffix(output_path.suffix + ".complete.json")
            manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
            manifest_temporary.write_text(json.dumps(dict(completion_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(manifest_temporary, manifest_path)
    except Exception:
        # Leave no artifact whose final name could be mistaken for completion.
        temporary.unlink(missing_ok=True)
        raise


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


def _merge_raw_rows(sources: Sequence[Iterator[DepthRow | TAQRow]]) -> Iterator[DepthRow | TAQRow]:
    """Bounded raw merge used only for streaming audit diagnostics."""
    import heapq

    pending: list[tuple[tuple[int, int, str, int], int, DepthRow | TAQRow]] = []
    for index, source in enumerate(sources):
        try:
            row = next(source)
        except StopIteration:
            continue
        heapq.heappush(pending, ((row.timestamp_ns, _event_priority(row), row.source_file, row.source_index), index, row))
    while pending:
        _, index, row = heapq.heappop(pending)
        yield row
        try:
            following = next(sources[index])
        except StopIteration:
            continue
        heapq.heappush(pending, ((following.timestamp_ns, _event_priority(following), following.source_file, following.source_index), index, following))


def _stream_audit_collisions(
    *, es_depth_paths: Sequence[Path], es_taq_paths: Sequence[Path], mes_taq_paths: Sequence[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], float]:
    """Collect timestamp and BBO-duration facts with one current-time buffer."""
    source_rows = _merge_raw_rows((
        iter_multiple_depth_paths(es_depth_paths),
        _iter_taq_timestamp_rows(iter_taq_paths(es_taq_paths, instrument="ES")),
        _iter_taq_timestamp_rows(iter_taq_paths(mes_taq_paths, instrument="MES")),
    ))
    current: list[DepthRow | TAQRow] = []
    current_timestamp: int | None = None
    total_events = collisions = max_events = depth_trade = trade_bbo = mes_es = affected_trades = provider_trades = 0
    es_bbo_rows = mes_quote_rows = aggressive_buys = aggressive_sells = 0
    mes_quote: dict[str, float | None] = {"B": None, "A": None}
    first_timestamp: int | None = None
    previous_timestamp: int | None = None
    mes_covered_ns = 0
    dates: set[str] = set()
    files: dict[str, dict[str, Any]] = {}

    def summarize(group: Sequence[DepthRow | TAQRow]) -> None:
        nonlocal total_events, collisions, max_events, depth_trade, trade_bbo, mes_es, affected_trades, provider_trades
        nonlocal es_bbo_rows, mes_quote_rows, aggressive_buys, aggressive_sells
        usable = [row for row in group if not isinstance(row, TAQRow) or row.event_kind != "OTHER"]
        total_events += len(usable)
        es_bbo_rows += sum(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind in {"BID", "ASK"} for row in usable)
        mes_quote_rows += sum(isinstance(row, TAQRow) and row.instrument == "MES" and row.event_kind in {"BID", "ASK"} for row in usable)
        aggressive_buys += sum(isinstance(row, TAQRow) and row.instrument == "ES" and row.aggressor == "BUY" for row in usable)
        aggressive_sells += sum(isinstance(row, TAQRow) and row.instrument == "ES" and row.aggressor == "SELL" for row in usable)
        if len(usable) < 2:
            provider_trades += sum(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind == "TRADE" for row in usable)
            return
        collisions += 1; max_events = max(max_events, len(usable))
        has_depth = any(isinstance(row, DepthRow) for row in usable)
        has_trade = any(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind == "TRADE" for row in usable)
        has_bbo = any(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind in {"BID", "ASK"} for row in usable)
        has_mes = any(isinstance(row, TAQRow) and row.instrument == "MES" and row.event_kind in {"BID", "ASK"} for row in usable)
        depth_trade += int(has_depth and has_trade); trade_bbo += int(has_trade and has_bbo)
        mes_es += int(has_mes and any(isinstance(row, DepthRow) or (isinstance(row, TAQRow) and row.instrument == "ES") for row in usable))
        trade_count = sum(isinstance(row, TAQRow) and row.instrument == "ES" and row.event_kind == "TRADE" for row in usable)
        provider_trades += trade_count; affected_trades += trade_count

    for row in source_rows:
        if current_timestamp is None:
            current_timestamp = row.timestamp_ns; first_timestamp = row.timestamp_ns
        if row.timestamp_ns != current_timestamp:
            assert previous_timestamp is not None
            if mes_quote["B"] is not None and mes_quote["A"] is not None:
                mes_covered_ns += row.timestamp_ns - previous_timestamp
            summarize(current)
            current.clear(); current_timestamp = row.timestamp_ns
        previous_timestamp = row.timestamp_ns
        current.append(row)
        dates.add(row.provider_timestamp[:10])
        entry = files.setdefault(row.source_file, {"path": row.source_file, "rows": 0, "first_timestamp_ns": row.timestamp_ns, "last_timestamp_ns": row.timestamp_ns})
        entry["rows"] += 1; entry["last_timestamp_ns"] = row.timestamp_ns
        if isinstance(row, TAQRow) and row.instrument == "MES" and row.event_kind in {"BID", "ASK"}:
            mes_quote["B" if row.event_kind == "BID" else "A"] = row.price
    if current:
        summarize(current)
    duration_ns = 0 if first_timestamp is None or previous_timestamp is None else previous_timestamp - first_timestamp
    coverage = 0.0 if not duration_ns else mes_covered_ns / duration_ns
    ties = {
        "total_normalized_events": total_events, "timestamps_with_multiple_events": collisions,
        "maximum_events_sharing_one_timestamp": max_events, "same_timestamp_es_depth_es_trade": depth_trade,
        "same_timestamp_es_trade_es_bbo": trade_bbo, "same_timestamp_mes_quote_es_event": mes_es,
        "trades_affected_by_same_timestamp_collision": affected_trades,
        "trades_affected_pct": 0.0 if not provider_trades else affected_trades * 100.0 / provider_trades,
    }
    ties["es_bbo_rows"] = es_bbo_rows
    ties["mes_quote_rows"] = mes_quote_rows
    ties["aggressive_buy_trades"] = aggressive_buys
    ties["aggressive_sell_trades"] = aggressive_sells
    return ties, list(files.values()), sorted(dates), coverage


def audit_inputs(*, es_depth_paths: Sequence[Path], es_taq_paths: Sequence[Path], mes_taq_paths: Sequence[Path]) -> dict[str, Any]:
    """Read and validate local feeds with bounded state; never run research."""
    metrics, profile = StreamingMetrics(), StreamingProfile()
    incomplete = bid_only = ask_only = crossed = locked = disagreements = 0
    latest_es_bbo: tuple[float | None, float | None] = (None, None)
    for event in iter_canonical_events_from_paths(es_depth_paths=es_depth_paths, es_taq_paths=es_taq_paths,
                                                   mes_taq_paths=mes_taq_paths, profile=profile, metrics=metrics):
        if event.kind == "ES_BBO":
            latest_es_bbo = event.es_taq_bbo or latest_es_bbo
        elif event.kind == "ES_DEPTH":
            if event.book_state == "INCOMPLETE":
                incomplete += 1
            elif event.book_state == "NON_EXECUTABLE":
                assert event.snapshot is None
                # LatestSideBook distinguishes these through state only after batching.
                locked += 1
            elif event.snapshot is not None and all(value is not None for value in latest_es_bbo):
                disagreements += int((event.snapshot.bids[0].price, event.snapshot.asks[0].price) != latest_es_bbo)
    ties, per_file, source_dates, mes_coverage = _stream_audit_collisions(
        es_depth_paths=es_depth_paths, es_taq_paths=es_taq_paths, mes_taq_paths=mes_taq_paths,
    )
    for entry in per_file:
        entry["sha256"] = _sha256(Path(entry["path"]))
    return {
        "status": "ALGOSEEK_INPUT_AUDIT_COMPLETE", "provider_provenance": provider_provenance(
            files=[*es_depth_paths, *es_taq_paths, *mes_taq_paths], source_date=source_dates,
        ),
        "files": per_file, "es_depth_rows": metrics.es_depth_rows, "canonical_es_depth_states": metrics.canonical_depth_states,
        "es_taq_rows": metrics.es_taq_rows, "mes_taq_rows": metrics.mes_taq_rows,
        "es_trade_rows": metrics.es_provider_trades,
        "es_bbo_rows": ties["es_bbo_rows"], "mes_quote_rows": ties["mes_quote_rows"],
        "aggressive_buy_trades": ties["aggressive_buy_trades"], "aggressive_sell_trades": ties["aggressive_sell_trades"],
        "generic_profile_trades": metrics.generic_trades, "profile": profile.result(),
        "malformed_rows": 0, "duplicate_rows": 0, "bid_only_initialization_periods": bid_only,
        "ask_only_initialization_periods": ask_only, "incomplete_book_states": incomplete,
        "crossed_book_states": crossed, "locked_book_states": locked,
        "es_depth_l1_vs_es_taq_bbo_disagreements": disagreements, "timestamp_normalization_issues": 0,
        "mes_bbo_coverage": mes_coverage, "streaming_metrics": metrics.as_dict(), "causal_tie_diagnostics": ties,
    }


def streaming_dry_build(
    *, es_depth_paths: Sequence[Path], es_taq_paths: Sequence[Path], mes_taq_paths: Sequence[Path],
) -> dict[str, Any]:
    """Exercise one session's canonical merge without retaining or writing events."""
    metrics, profile = StreamingMetrics(), StreamingProfile()
    started = time.monotonic()
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mes_coverage = _RawBBOCoverage()
    for event in iter_canonical_events_from_paths(es_depth_paths=es_depth_paths, es_taq_paths=es_taq_paths,
                                                  mes_taq_paths=mes_taq_paths, profile=profile, metrics=metrics,
                                                  mes_bbo_coverage=mes_coverage):
        pass
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.  The measurement is process-wide
    # high-water mark, so callers should run this in a fresh process for a
    # meaningful per-session peak.
    peak_bytes = end_rss if os.uname().sysname == "Darwin" else end_rss * 1024
    start_bytes = start_rss if os.uname().sysname == "Darwin" else start_rss * 1024
    session_start, session_end = (None, None) if metrics.raw_first_timestamp_ns is None else _logical_cme_session_bounds(metrics.raw_first_timestamp_ns)
    return {
        "status": "ALGOSEEK_STREAMING_DRY_BUILD_COMPLETE",
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_bytes,
        "peak_rss_increment_bytes": max(0, peak_bytes - start_bytes),
        "canonical_events_emitted": metrics.canonical_events_emitted,
        "maximum_depth_timestamp_rows": metrics.max_depth_timestamp_rows,
        "maximum_merge_pending_events": metrics.max_merge_pending_events,
        "mes_bbo_coverage": mes_coverage.result(session_start=session_start, session_end=session_end),
        "logical_session_start_ns": session_start, "logical_session_end_ns": session_end,
        "profile": profile.result(),
        "streaming_metrics": metrics.as_dict(),
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
