"""Canonical public-book adapters for L2 historical replay.

The strategy layer consumes only :class:`PublicBookEvent`.  This module keeps
source-specific details at the boundary: private MBO order identity remains in
the existing historical adapter, while native aggregate MBP-10 is translated
directly into the same public objects.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol

from . import historical_runner as historical
from .model import Execution, MBP10Snapshot, MBP10Update, MBPLevel


MBO_DERIVED_MBP10 = "MBO_DERIVED_MBP10"
NATIVE_MBP10 = "NATIVE_MBP10"
SUPPORTED_SOURCE_MODELS = frozenset({MBO_DERIVED_MBP10, NATIVE_MBP10})
RAW_PRICE_SCALE = historical.RAW_PRICE_SCALE


class PublicBookAdapter(Protocol):
    state: str
    first_valid_book_ns: int | None

    def feed(self, record: object, *, materialize_public: bool = True) -> historical.PublicBookEvent | None: ...
    def finish(self) -> None: ...

    def source_integrity_diagnostics(self) -> list[dict[str, Any]]: ...


def source_model_adapter(source_model: str) -> PublicBookAdapter:
    """Construct exactly one adapter named by the sealed session manifest."""
    if source_model == MBO_DERIVED_MBP10:
        return historical.HistoricalMBOToMBP10Adapter()
    if source_model == NATIVE_MBP10:
        return NativeDatabentoMBP10Adapter()
    raise historical.HistoricalReplayError(f"unsupported public-book source model: {source_model}")


def source_timestamp_ns(record: object) -> int:
    value = getattr(record, "timestamp_ns", None)
    if value is not None:
        return int(value)
    value = getattr(record, "ts_recv", None)
    if value is None:
        value = getattr(record, "ts_event", None)
    if value is None:
        raise historical.HistoricalReplayError("native MBP-10 record has no timestamp")
    return int(value)


def stream_source(path: Any, source_model: str, extra_paths: tuple[Any, ...] = ()) -> Iterator[object]:
    """Read only the explicitly declared local source using its model."""
    if source_model == MBO_DERIVED_MBP10:
        yield from historical._stream_private_mbo(path)
        return
    if source_model != NATIVE_MBP10:
        raise historical.HistoricalReplayError(f"unsupported public-book source model: {source_model}")
    from databento import DBNStore
    # The native Dec/Jan contract is explicitly two-part: pre-NY coverage
    # followed by the already sealed post-NY source.  The caller supplies both
    # paths from the session manifest; no globbing or discovery is performed.
    for declared_path in (path, *extra_paths):
        for record in DBNStore.from_file(declared_path):
            yield record


def _code(value: object) -> str:
    return str(getattr(value, "value", value)).rsplit(".", 1)[-1]


def _native_price(raw: object, context: str) -> float:
    value = int(raw)
    if value <= 0 or value >= historical.UNDEF_PRICE:
        raise historical.HistoricalReplayError(f"invalid native MBP-10 price: {context}")
    return value / RAW_PRICE_SCALE


def _levels(raw_levels: object, side: str) -> tuple[MBPLevel, ...]:
    rows: list[MBPLevel] = []
    for raw in tuple(raw_levels or ()):
        price = int(getattr(raw, "bid_px" if side == "B" else "ask_px", 0))
        size = int(getattr(raw, "bid_sz" if side == "B" else "ask_sz", 0))
        count = int(getattr(raw, "bid_ct" if side == "B" else "ask_ct", 0))
        if price == size == count == 0:
            continue
        if price <= 0 or size < 0 or count < 0:
            raise historical.HistoricalReplayError("malformed native MBP-10 aggregate level")
        rows.append(MBPLevel(_native_price(price, f"{side} level"), size, count))
    prices = [row.price for row in rows]
    if len(prices) != len(set(prices)):
        raise historical.HistoricalReplayError("native MBP-10 snapshot contains duplicate prices")
    return tuple(rows)


class NativeDatabentoMBP10Adapter:
    """Translate native Databento aggregate MBP-10 into canonical public events.

    Native trade ``side`` is the aggressor side.  A buy consumes the ask and a
    sell consumes the bid, so the accompanying aggregate FILL update is emitted
    on the opposite (resting/passive) side, matching the MBO adapter's public
    semantics.  No order identity is retained or exposed.
    """

    def __init__(self) -> None:
        self.previous: MBP10Snapshot | None = None
        self.first_valid_book_ns: int | None = None
        self.state = "UNINITIALIZED"
        self.temporary_non_executable_started_ns: int | None = None
        self.temporary_non_executable_records = 0
        self.last_transient: dict[str, int] | None = None
        self._last_timestamp_ns: int | None = None
        self._diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def _executable(snapshot: MBP10Snapshot) -> bool:
        return bool(snapshot.bids and snapshot.asks and snapshot.asks[0].price > snapshot.bids[0].price)

    def _suspend(self, timestamp_ns: int) -> None:
        if self.state != "TEMPORARILY_NON_EXECUTABLE":
            self.temporary_non_executable_started_ns = timestamp_ns
            self.temporary_non_executable_records = 0
        self.temporary_non_executable_records += 1
        self.state, self.previous = "TEMPORARILY_NON_EXECUTABLE", None

    def _open(self, timestamp_ns: int) -> None:
        if self.state == "TEMPORARILY_NON_EXECUTABLE":
            assert self.temporary_non_executable_started_ns is not None
            self.last_transient = {
                "start_timestamp_ns": self.temporary_non_executable_started_ns,
                "reopen_timestamp_ns": timestamp_ns,
                "non_executable_records": self.temporary_non_executable_records,
            }
        self.temporary_non_executable_started_ns = None
        self.temporary_non_executable_records = 0
        if self.first_valid_book_ns is None:
            self.first_valid_book_ns = timestamp_ns
        self.state = "EXECUTABLE"

    def feed(self, record: object, *, materialize_public: bool = True) -> historical.PublicBookEvent | None:
        timestamp = source_timestamp_ns(record)
        if self._last_timestamp_ns is not None and timestamp < self._last_timestamp_ns:
            raise historical.HistoricalReplayError("native MBP-10 timestamps decreased")
        self._last_timestamp_ns = timestamp
        action = _code(getattr(record, "action", ""))
        side = _code(getattr(record, "side", ""))
        if action not in {"A", "C", "M", "R", "T"} or side not in {"A", "B", "N"}:
            raise historical.HistoricalReplayError(f"unsupported native MBP-10 action/side: {action}/{side}")
        snapshot = MBP10Snapshot(timestamp, _levels(getattr(record, "levels", ()), "B"),
                                 _levels(getattr(record, "levels", ()), "A"))
        executable = self._executable(snapshot)
        if not executable:
            if action in {"T", "R"}:
                self.state, self.previous = "WAITING_FOR_REOPEN_BOOK", None
                return None
            if self.state == "UNINITIALIZED":
                return None
            if snapshot.bids and snapshot.asks:
                self._suspend(timestamp)
                return None
            raise historical.HistoricalReplayError("native MBP-10 book became unrecoverably non-executable")
        was_reopen = self.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}
        self._open(timestamp)
        if not materialize_public:
            return None
        update: MBP10Update | None = None
        if action in {"A", "C", "M"} and side in {"A", "B"}:
            raw_price = int(getattr(record, "price", 0))
            price = _native_price(raw_price, "aggregate update")
            before_rows = self.previous.bids if self.previous and side == "B" else self.previous.asks if self.previous else ()
            after_rows = snapshot.bids if side == "B" else snapshot.asks
            before = next((row for row in before_rows if row.price == price), None)
            after = next((row for row in after_rows if row.price == price), None)
            if before is not None or after is not None:
                update = MBP10Update(timestamp, side, price,
                                     (after.size if after else 0) - (before.size if before else 0),
                                     (after.order_count if after else 0) - (before.order_count if before else 0),
                                     {"A": "ADD", "C": "CANCEL", "M": "MODIFY"}[action])
        execution: Execution | None = None
        if action == "T":
            price = _native_price(getattr(record, "price", 0), "trade")
            size = int(getattr(record, "size", 0))
            if size <= 0:
                raise historical.HistoricalReplayError("native MBP-10 trade has non-positive size")
            aggressor = "BUY" if side == "B" else "SELL" if side == "A" else "UNKNOWN"
            execution = Execution(timestamp, price, size, aggressor)  # type: ignore[arg-type]
            passive = "A" if side == "B" else "B" if side == "A" else None
            if passive is not None:
                before_rows = self.previous.asks if self.previous and passive == "A" else self.previous.bids if self.previous else ()
                after_rows = snapshot.asks if passive == "A" else snapshot.bids
                before = next((row for row in before_rows if row.price == price), None)
                after = next((row for row in after_rows if row.price == price), None)
                delta = (after.size if after else 0) - (before.size if before else 0)
                if delta >= 0:
                    delta = -size
                update = MBP10Update(timestamp, passive, price, delta,
                                     (after.order_count if after else 0) - (before.order_count if before else 0), "FILL")
        self.previous = snapshot
        # A reopened book is an absolute snapshot; do not invent a delta from
        # the invalid predecessor.
        if was_reopen:
            update = None if update is not None and action in {"A", "C", "M"} else update
        return historical.PublicBookEvent(timestamp, snapshot, update, execution)

    def finish(self) -> None:
        if self.first_valid_book_ns is None or self.state != "EXECUTABLE":
            raise historical.HistoricalReplayError(f"incomplete native MBP-10 source: {self.state}")

    def source_integrity_diagnostics(self) -> list[dict[str, Any]]:
        return list(self._diagnostics)


def assert_supported_source_model(source_model: str) -> None:
    if source_model not in SUPPORTED_SOURCE_MODELS:
        raise historical.HistoricalReplayError(f"unsupported public-book source model: {source_model}")
