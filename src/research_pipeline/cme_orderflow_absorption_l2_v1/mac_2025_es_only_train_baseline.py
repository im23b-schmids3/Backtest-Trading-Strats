"""Causal ES-only TRAIN baseline for the reduced MAC 2025 MBP-10 package.

This is a deliberately small orchestration layer around the existing native
MBP-10 adapter and frozen :class:`HistoricalL2Runner`.  It does not discover
data, call a provider, inspect validation outcomes, or optimize parameters.
The only market-data input is native ES ``mbp-10``; trade records are the
``T`` records carried by that MBP-10 stream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time as wall_time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from . import historical_runner as historical
from . import mac_2025_native_mbp_quote as plan
from .model import (
    Execution, L2ClassBConfig, L2Config, L2SignalEngine, L2Setup,
    ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
)
from .public_book_adapters import NativeDatabentoMBP10Adapter, source_timestamp_ns


RUN_ID = "CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE"
DATASET = plan.DATASET
SOURCE_MODEL = "NATIVE_MBP10"
UTC = timezone.utc
SESSION_ORDER = ("ASIA", "EUROPE", "NY")
PROFILE_LEVELS = ("POC", "VAH", "VAL", "HIGH", "LOW")
TICK_RAW = 250_000_000
TRAIN_DATES = tuple(day.isoformat() for day in plan.train_dates())
DEPENDENCY_DATE = plan.DEPENDENCY_DATES[0].isoformat()
OUTPUT_ROOT = Path("research_runs") / RUN_ID
DATA_ROOT = Path("data/databento/mac-2025-native-es-mbp10-final-reduced")
MANIFEST_NAME = "mac-2025-native-mbp-download-manifest.json"
SEMANTIC_FILES = ("model.py", "historical_runner.py", "public_book_adapters.py", "mac_2025_es_only_train_baseline.py")


class BaselineError(RuntimeError):
    """The sealed source or causal baseline contract cannot be satisfied."""


class BroadSignalEngine(L2SignalEngine):
    """The frozen signal rules with bounded event-time bookkeeping.

    ``L2SignalEngine`` is intentionally simple and scans all pending setups on
    every public event.  A 50-family research matrix can have many pending
    confirmations, so this subclass keeps only the active 15-second window in
    a deque and only checks entry-ready setups after confirmation.  It does
    not alter any threshold, favorable-price, or sizing rule.
    """

    def __init__(self, config: L2Config, class_b: L2ClassBConfig = L2ClassBConfig()) -> None:
        super().__init__(config, class_b)
        self._pending_order: deque[str] = deque()
        self._entry_ready: deque[str] = deque()

    def register_completed(self, interaction: Any) -> L2Setup | None:
        setup = super().register_completed(interaction)
        if setup is not None:
            self._pending_order.append(setup.setup_id)
        return setup

    def _expire(self, timestamp_ns: int) -> None:
        while self._pending_order:
            setup = self.pending[self._pending_order[0]]
            max_ns = int(self.class_b.max_confirmation_seconds * 1_000_000_000)
            if setup.terminal_reason is not None or timestamp_ns > (setup.interaction.end_ns or 0) + max_ns:
                if setup.terminal_reason is None:
                    setup.state, setup.terminal_reason = "FAILED", "CONFIRMATION_WINDOW_EXPIRED"
                self._pending_order.popleft()
                continue
            break

    def advance(self, timestamp_ns: int) -> None:
        self._expire(timestamp_ns)

    def observe_execution(self, event: Any) -> None:
        self._expire(event.timestamp_ns)
        for setup_id in tuple(self._pending_order):
            setup = self.pending[setup_id]
            if setup.terminal_reason is not None or setup.state == "CONFIRMED":
                continue
            before = setup.state
            favorable = self._observe_confirmation(setup, event)
            if before != "CONFIRMED" and setup.state == "CONFIRMED":
                self._entry_ready.append(setup_id)


@dataclass(frozen=True)
class _FastLevel:
    price: float
    size: int
    order_count: int


@dataclass(frozen=True)
class _FastSnapshot:
    timestamp_ns: int
    bids: tuple[_FastLevel, ...]
    asks: tuple[_FastLevel, ...]

    def level_at(self, side: str, price: float) -> _FastLevel | None:
        rows = self.bids if side == "B" else self.asks
        return next((row for row in rows if row.price == price), None)

    def depth(self, side: str, levels: int) -> int:
        rows = self.bids if side == "B" else self.asks
        return sum(row.size for row in rows[:levels])


@dataclass(frozen=True)
class _FastUpdate:
    timestamp_ns: int
    side: str
    price: float
    size_delta: int
    order_count_delta: int
    kind: str


@dataclass(frozen=True)
class _FastPublic:
    timestamp_ns: int
    snapshot: _FastSnapshot
    update: _FastUpdate | None
    execution: Execution | None


class FastNativeReplayAdapter:
    """Allocation-light equivalent of ``NativeDatabentoMBP10Adapter``.

    The acquisition audit has already validated the DBN schema, dimensions,
    prices, and ordering.  This adapter therefore avoids repeating per-level
    dataclass validation on 600M+ source fields while preserving the generic
    adapter's public event semantics exactly at the strategy boundary.
    """

    def __init__(self) -> None:
        self.previous: _FastSnapshot | None = None
        self.previous_raw: object | None = None
        self.state = "UNINITIALIZED"
        self.first_valid_book_ns: int | None = None
        self._last_timestamp_ns: int | None = None

    @staticmethod
    def _code(value: object) -> str:
        return str(getattr(value, "value", value)).rsplit(".", 1)[-1]

    @staticmethod
    def _levels(raw_levels: object, side: str) -> tuple[_FastLevel, ...]:
        rows = []
        price_name, size_name, count_name = (
            ("bid_px", "bid_sz", "bid_ct") if side == "B" else ("ask_px", "ask_sz", "ask_ct")
        )
        for raw in raw_levels or ():
            price = int(getattr(raw, price_name, 0))
            size = int(getattr(raw, size_name, 0))
            count = int(getattr(raw, count_name, 0))
            if price == size == count == 0:
                continue
            rows.append(_FastLevel(price / 1_000_000_000, size, count))
        return tuple(rows)

    @staticmethod
    def _executable(snapshot: _FastSnapshot) -> bool:
        return bool(snapshot.bids and snapshot.asks and snapshot.asks[0].price > snapshot.bids[0].price)

    @staticmethod
    def _raw_executable(record: object) -> bool:
        levels = getattr(record, "levels", ())
        if not levels:
            return False
        first = levels[0]
        bid = int(getattr(first, "bid_px", 0))
        ask = int(getattr(first, "ask_px", 0))
        return bid > 0 and ask > bid

    @staticmethod
    def _array_levels(row: Any, side: str, interest_prices: frozenset[float], extra_price: float | None = None) -> tuple[_FastLevel, ...]:
        rows = []
        prefix = "bid" if side == "B" else "ask"
        for index in range(10):
            price = int(row[f"{prefix}_px_{index:02d}"])
            size = int(row[f"{prefix}_sz_{index:02d}"])
            count = int(row[f"{prefix}_ct_{index:02d}"])
            if price == size == count == 0:
                continue
            normalized = price / 1_000_000_000
            if index >= 5 and normalized not in interest_prices and normalized != extra_price:
                continue
            rows.append(_FastLevel(normalized, size, count))
        return tuple(rows)

    @staticmethod
    def _array_level_at(row: Any, side: str, target_price: float) -> _FastLevel | None:
        prefix = "bid" if side == "B" else "ask"
        for index in range(10):
            price = int(row[f"{prefix}_px_{index:02d}"])
            if price == 0:
                continue
            normalized = price / 1_000_000_000
            if normalized == target_price:
                return _FastLevel(normalized, int(row[f"{prefix}_sz_{index:02d}"]), int(row[f"{prefix}_ct_{index:02d}"]))
        return None

    @staticmethod
    def _array_action(row: Any) -> str:
        value = row["action"]
        return value.decode() if hasattr(value, "decode") else str(value)

    @staticmethod
    def _array_side(row: Any) -> str:
        value = row["side"]
        return value.decode() if hasattr(value, "decode") else str(value)

    @staticmethod
    def _array_executable(row: Any) -> bool:
        bid = int(row["bid_px_00"])
        ask = int(row["ask_px_00"])
        return bid > 0 and ask > bid

    def feed_array(self, row: Any, *, materialize_public: bool = True,
                   interest_prices: frozenset[float] = frozenset()) -> _FastPublic | None:
        """Feed one native DBN ndarray row using the same Policy-C rules."""
        timestamp = int(row["ts_recv"])
        if self._last_timestamp_ns is not None and timestamp < self._last_timestamp_ns:
            raise BaselineError("native MBP-10 timestamps decreased")
        self._last_timestamp_ns = timestamp
        action, side = self._array_action(row), self._array_side(row)
        if not materialize_public and action != "T":
            if not self._array_executable(row):
                if action in {"R", "T"}:
                    self.state, self.previous, self.previous_raw = "WAITING_FOR_REOPEN_BOOK", None, row
                    return None
                if self.state == "UNINITIALIZED":
                    self.previous_raw = row
                    return None
                self.state, self.previous, self.previous_raw = "TEMPORARILY_NON_EXECUTABLE", None, row
                return None
            self.state = "EXECUTABLE"
            if self.first_valid_book_ns is None:
                self.first_valid_book_ns = timestamp
            self.previous_raw = row
            self.previous = None
            return None
        raw_price = int(row["price"]) / 1_000_000_000
        snapshot = _FastSnapshot(timestamp, self._array_levels(row, "B", interest_prices, raw_price),
                                 self._array_levels(row, "A", interest_prices, raw_price))
        if not self._executable(snapshot):
            if action in {"T", "R"}:
                self.state, self.previous, self.previous_raw = "WAITING_FOR_REOPEN_BOOK", None, row
                return None
            if self.state == "UNINITIALIZED":
                self.previous_raw = row
                return None
            if snapshot.bids and snapshot.asks:
                self.state, self.previous, self.previous_raw = "TEMPORARILY_NON_EXECUTABLE", None, row
                return None
            raise BaselineError("native MBP-10 book became unrecoverably non-executable")
        was_reopen = self.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}
        self.state = "EXECUTABLE"
        if self.first_valid_book_ns is None:
            self.first_valid_book_ns = timestamp
        if self.previous is None and self.previous_raw is not None:
            prior = self.previous_raw
            self.previous = _FastSnapshot(int(prior["ts_recv"]), self._array_levels(prior, "B", interest_prices, raw_price),
                                          self._array_levels(prior, "A", interest_prices, raw_price))
        update = None
        if action in {"A", "C", "M"} and side in {"A", "B"}:
            price = raw_price
            before_rows = self.previous.bids if self.previous and side == "B" else self.previous.asks if self.previous else ()
            after_rows = snapshot.bids if side == "B" else snapshot.asks
            before = next((item for item in before_rows if item.price == price), None)
            if before is None and self.previous_raw is not None:
                before = self._array_level_at(self.previous_raw, side, price)
            after = next((item for item in after_rows if item.price == price), None)
            if before is not None or after is not None:
                update = _FastUpdate(timestamp, side, price, (after.size if after else 0) - (before.size if before else 0),
                                     (after.order_count if after else 0) - (before.order_count if before else 0),
                                     {"A": "ADD", "C": "CANCEL", "M": "MODIFY"}[action])
        execution = None
        if action == "T":
            price = raw_price
            size = int(row["size"])
            aggressor = "BUY" if side == "B" else "SELL" if side == "A" else "UNKNOWN"
            execution = Execution(timestamp, price, size, aggressor)  # type: ignore[arg-type]
            passive = "A" if side == "B" else "B" if side == "A" else None
            if passive is not None:
                before_rows = self.previous.asks if self.previous and passive == "A" else self.previous.bids if self.previous else ()
                after_rows = snapshot.asks if passive == "A" else snapshot.bids
                before = next((item for item in before_rows if item.price == price), None)
                if before is None and self.previous_raw is not None:
                    before = self._array_level_at(self.previous_raw, passive, price)
                after = next((item for item in after_rows if item.price == price), None)
                delta = (after.size if after else 0) - (before.size if before else 0)
                if delta >= 0:
                    delta = -size
                update = _FastUpdate(timestamp, passive, price, delta,
                                     (after.order_count if after else 0) - (before.order_count if before else 0), "FILL")
        self.previous, self.previous_raw = snapshot, row
        if was_reopen and update is not None and action in {"A", "C", "M"}:
            update = None
        return _FastPublic(timestamp, snapshot, update, execution)

    def feed(self, record: object, *, materialize_public: bool = True) -> _FastPublic | None:
        timestamp = int(getattr(record, "ts_recv", getattr(record, "ts_event", 0)))
        if self._last_timestamp_ns is not None and timestamp < self._last_timestamp_ns:
            raise BaselineError("native MBP-10 timestamps decreased")
        self._last_timestamp_ns = timestamp
        action = self._code(getattr(record, "action", ""))
        side = self._code(getattr(record, "side", ""))
        if not materialize_public and action != "T":
            executable = self._raw_executable(record)
            if not executable:
                if action in {"R", "T"}:
                    self.state, self.previous, self.previous_raw = "WAITING_FOR_REOPEN_BOOK", None, record
                    return None
                if self.state == "UNINITIALIZED":
                    self.previous_raw = record
                    return None
                if getattr(record, "levels", ()):
                    self.state, self.previous, self.previous_raw = "TEMPORARILY_NON_EXECUTABLE", None, record
                    return None
                raise BaselineError("native MBP-10 book became unrecoverably non-executable")
            self.state = "EXECUTABLE"
            if self.first_valid_book_ns is None:
                self.first_valid_book_ns = timestamp
            self.previous_raw = record
            self.previous = None
            return None
        snapshot = _FastSnapshot(timestamp, self._levels(getattr(record, "levels", ()), "B"),
                                 self._levels(getattr(record, "levels", ()), "A"))
        if not self._executable(snapshot):
            if action in {"T", "R"}:
                self.state, self.previous = "WAITING_FOR_REOPEN_BOOK", None
                return None
            if self.state == "UNINITIALIZED":
                return None
            if snapshot.bids and snapshot.asks:
                self.state, self.previous = "TEMPORARILY_NON_EXECUTABLE", None
                return None
            raise BaselineError("native MBP-10 book became unrecoverably non-executable")
        was_reopen = self.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}
        self.state = "EXECUTABLE"
        if self.first_valid_book_ns is None:
            self.first_valid_book_ns = timestamp
        if not materialize_public:
            self.previous = snapshot
            self.previous_raw = record
            return None
        if self.previous is None and self.previous_raw is not None:
            prior = self.previous_raw
            self.previous = _FastSnapshot(
                int(getattr(prior, "ts_recv", getattr(prior, "ts_event", 0))),
                self._levels(getattr(prior, "levels", ()), "B"), self._levels(getattr(prior, "levels", ()), "A"),
            )
        update = None
        if action in {"A", "C", "M"} and side in {"A", "B"}:
            price = int(getattr(record, "price", 0)) / 1_000_000_000
            before_rows = self.previous.bids if self.previous and side == "B" else self.previous.asks if self.previous else ()
            after_rows = snapshot.bids if side == "B" else snapshot.asks
            before = next((row for row in before_rows if row.price == price), None)
            after = next((row for row in after_rows if row.price == price), None)
            if before is not None or after is not None:
                update = _FastUpdate(timestamp, side, price, (after.size if after else 0) - (before.size if before else 0),
                                     (after.order_count if after else 0) - (before.order_count if before else 0),
                                     {"A": "ADD", "C": "CANCEL", "M": "MODIFY"}[action])
        execution = None
        if action == "T":
            price = int(getattr(record, "price", 0)) / 1_000_000_000
            size = int(getattr(record, "size", 0))
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
                update = _FastUpdate(timestamp, passive, price, delta,
                                     (after.order_count if after else 0) - (before.order_count if before else 0), "FILL")
        self.previous = snapshot
        self.previous_raw = record
        if was_reopen and update is not None and action in {"A", "C", "M"}:
            update = None
        return _FastPublic(timestamp, snapshot, update, execution)

    def finish(self) -> None:
        if self.first_valid_book_ns is None or self.state != "EXECUTABLE":
            raise BaselineError(f"incomplete native MBP-10 source: {self.state}")


class BroadHistoricalRunner(historical.HistoricalL2Runner):
    """Historical runner with bounded confirmation scans for the broad matrix."""

    def __init__(self, *, signal_engine_factory: Callable[[L2Config, L2ClassBConfig], L2SignalEngine] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.signals = (signal_engine_factory or BroadSignalEngine)(self.config, self.class_b)

    def _attempt_entry(self, timestamp_ns: int) -> None:
        if self.es_quote is None:
            return
        ready = self.signals._entry_ready
        for setup_id in tuple(ready):
            setup = self.signals.pending[setup_id]
            if setup.terminal_reason is not None or setup.state != "CONFIRMED":
                ready.remove(setup_id)
                continue
            if setup.entry_ready_ns is None or timestamp_ns < setup.entry_ready_ns:
                continue
            position = self.signals.try_enter(
                setup_id, timestamp_ns=timestamp_ns, es_bid=self.es_quote[0], es_ask=self.es_quote[1],
                execution_policy=self.execution_policy,
            )
            if position is not None:
                self.diagnostic_events.append({"event": "ENTRY", "setup_id": setup_id,
                                               "timestamp_ns": timestamp_ns, "instrument": position.instrument,
                                               "contracts": position.contracts})
                ready.remove(setup_id)
                break

    def observe_public(self, event: historical.PublicBookEvent) -> None:
        self.interactions.advance(event.timestamp_ns)
        self._new_completed()
        self.es_quote = historical._quote(event.snapshot)
        if self.es_quote is not None:
            self.es_quote_timestamp_ns = event.timestamp_ns
        self._manage_position(event.timestamp_ns, self.es_quote, "ES")
        # Before the first execution, no interaction can observe a depth
        # history.  The execution-bearing snapshot is still materialized
        # below, and thereafter every snapshot is preserved for active
        # interactions exactly as in HistoricalL2Runner.
        if self.interactions.active or event.execution is not None:
            self.interactions.observe_snapshot(event.snapshot, event.update)
        if event.execution is not None:
            self.signals.advance(event.timestamp_ns)
            self.interactions.observe_execution(event.execution)
            self._new_completed()
            self.signals.observe_execution(event.execution)
        self._attempt_entry(event.timestamp_ns)


@dataclass(frozen=True)
class BroadLevel:
    """Research-family level accepted by the existing engine boundary.

    ``StructuralLevel`` intentionally has a historical finite-name list.  The
    broad MAC matrix needs one identity per family, so this private boundary
    object keeps the same ``name``/``price`` contract without changing the
    frozen strategy model or its existing contract hash.
    """

    name: str
    price: float

    def __post_init__(self) -> None:
        value = float(self.price)
        if not 0.0 < value < 100_000.0:
            raise BaselineError(f"invalid ES structural price: {self.price}")
        object.__setattr__(self, "price", value)


@dataclass(frozen=True)
class Family:
    family_id: str
    trading_session: str
    reference_session: str
    reference_day: str
    reference_level: str
    causal_ready_timestamp: str
    dynamic: bool = False


@dataclass
class Profile:
    day: str
    session: str
    start_ns: int
    end_ns: int
    volume_by_tick: Counter[int]

    @classmethod
    def create(cls, day: str, session: str, start_ns: int, end_ns: int) -> "Profile":
        return cls(day, session, start_ns, end_ns, Counter())

    def add(self, price: float, size: int) -> None:
        if size > 0:
            self.volume_by_tick[int(round(float(price) * 1_000_000_000))] += int(size)

    def values(self) -> dict[str, float]:
        if not self.volume_by_tick:
            raise BaselineError(f"no executed volume for {self.day} {self.session} profile")
        poc = min(self.volume_by_tick, key=lambda tick: (-self.volume_by_tick[tick], tick))
        total = sum(self.volume_by_tick.values())
        included = self.volume_by_tick[poc]
        low = high = poc
        while included * 100 < total * 70:
            below, above = low - TICK_RAW, high + TICK_RAW
            below_volume = self.volume_by_tick.get(below, 0)
            above_volume = self.volume_by_tick.get(above, 0)
            if below_volume >= above_volume:
                low, included = below, included + below_volume
            else:
                high, included = above, included + above_volume
            if below_volume == 0 and above_volume == 0:
                break
        return {
            "POC": poc / 1_000_000_000,
            "VAH": high / 1_000_000_000,
            "VAL": low / 1_000_000_000,
            "HIGH": max(self.volume_by_tick) / 1_000_000_000,
            "LOW": min(self.volume_by_tick) / 1_000_000_000,
        }


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _csv_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).parent
    for name in SEMANTIC_FILES:
        path = package_root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise BaselineError(f"timestamp is not timezone-aware: {value}")
    return int(parsed.timestamp() * 1_000_000_000)


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def _day_ns(day: str, clock: time) -> int:
    return int(datetime.combine(date.fromisoformat(day), clock, tzinfo=UTC).timestamp() * 1_000_000_000)


def _session_windows(day: str) -> dict[str, tuple[int, int]]:
    calendar_day = date.fromisoformat(day)
    ny = datetime.combine(calendar_day, time(9, 30), tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    close = datetime.combine(calendar_day, time(16, 0), tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    return {
        "ASIA": (_day_ns(day, time(0, 0)), _day_ns(day, time(8, 0))),
        "EUROPE": (_day_ns(day, time(8, 0)), int(ny.timestamp() * 1_000_000_000)),
        "NY": (int(ny.timestamp() * 1_000_000_000), int(close.timestamp() * 1_000_000_000)),
    }


def _manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"missing or unreadable acquisition manifest: {path}") from exc
    requests = payload.get("requests")
    if not isinstance(requests, dict) or len(requests) != 56:
        raise BaselineError("ES-only acquisition manifest must contain exactly 56 requests")
    return payload, requests


def audit_dataset(root: Path = DATA_ROOT, *, full_record_scan: bool = True) -> dict[str, Any]:
    """Audit all 56 sealed requests before a baseline can start."""
    payload, requests = _manifest(root)
    expected = {item.request_id: item for item in plan.build_es_only_requests()}
    failures: list[dict[str, Any]] = []
    verified = 0
    records = 0
    actions: Counter[str] = Counter()
    actual_paths: set[str] = set()
    from databento import DBNStore

    for request_id, item in expected.items():
        row = requests.get(request_id)
        if not isinstance(row, dict):
            failures.append({"request_id": request_id, "reason": "missing_manifest_request"})
            continue
        relative = str(row.get("path", ""))
        actual_paths.add(relative)
        path = root / relative
        try:
            if row.get("status") != "VERIFIED" or row.get("schema") != "mbp-10" or row.get("symbol") != item.symbol:
                raise BaselineError("manifest identity/status mismatch")
            if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
                raise BaselineError("file missing or byte size mismatch")
            digest = _sha256(path)
            if digest != row.get("sha256"):
                raise BaselineError("SHA-256 mismatch")
            store = DBNStore.from_file(path)
            metadata = store.metadata
            if metadata.dataset != DATASET or metadata.schema != "mbp-10" or item.symbol not in metadata.symbols:
                raise BaselineError(f"DBN metadata mismatch: {metadata}")
            if full_record_scan:
                start, end = _ns(item.start), _ns(item.end)
                for record in store:
                    timestamp = int(getattr(record, "ts_recv", getattr(record, "ts_event", 0)))
                    if not start <= timestamp < end:
                        raise BaselineError(f"ts_recv outside [{item.start},{item.end})")
                    actions[str(getattr(getattr(record, "action", ""), "name", getattr(record, "action", "")))] += 1
                    records += 1
            verified += 1
        except Exception as exc:  # audit converts every source defect into a report
            failures.append({"request_id": request_id, "path": str(path), "reason": str(exc)})
    declared_paths = {str(row.get("path")) for row in requests.values() if isinstance(row, dict)}
    unexpected = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.dbn.zst") if path.relative_to(root).as_posix() not in declared_paths)
    result = {
        "status": "PASS" if not failures and not unexpected and verified == len(expected) else "FAIL",
        "dataset": DATASET,
        "expected_file_count": len(expected),
        "verified_file_count": verified,
        "failed_files": failures,
        "unexpected_files": unexpected,
        "source_integrity": "PASS" if not failures and not unexpected else "FAIL",
        "record_count": records,
        "actions": dict(actions),
        "manifest_sha256": _sha256(root / MANIFEST_NAME),
        "full_record_scan": full_record_scan,
    }
    return result


def _family_id(trading: str, reference: str, day_kind: str, level: str) -> str:
    return f"{trading}|{reference}|{day_kind}|{level}"


def _causal_ready(day: str, session: str, current_source: str | None, windows: Mapping[str, tuple[int, int]]) -> int:
    if current_source is None:
        return windows[session][0]
    return windows[current_source][1]


def build_families(day: str, prior_profiles: Mapping[str, Profile], current_profiles: Mapping[str, Profile]) -> tuple[Family, ...]:
    windows = _session_windows(day)
    specs: list[tuple[str, str, str, str, str | None]] = []
    # The same-session prior profile and prior RTH context are always causal.
    for level in PROFILE_LEVELS:
        specs.append(("ASIA", "ASIA", "PRIOR", level, None))
        specs.append(("ASIA", "RTH", "PRIOR", level, None))
        specs.append(("EUROPE", "EUROPE", "PRIOR", level, None))
        specs.append(("EUROPE", "RTH", "PRIOR", level, None))
        specs.append(("EUROPE", "ASIA", "CURRENT", level, "ASIA"))
        specs.append(("EUROPE", "ASIA", "PRIOR", level, None))
        specs.append(("NY", "NY", "PRIOR", level, None))
        specs.append(("NY", "EUROPE", "PRIOR", level, None))
        specs.append(("NY", "ASIA", "PRIOR", level, None))
        specs.append(("NY", "EUROPE", "CURRENT", level, "EUROPE"))
        specs.append(("NY", "ASIA", "CURRENT", level, "ASIA"))
    for trading in SESSION_ORDER:
        specs.extend(((trading, trading, "CURRENT", "HIGH", trading),
                      (trading, trading, "CURRENT", "LOW", trading)))
    families: list[Family] = []
    for trading, reference, day_kind, level, current_source in specs:
        source = current_profiles.get(current_source or "") if day_kind == "CURRENT" else prior_profiles.get(reference if reference != "RTH" else "NY")
        if source is None:
            continue
        ready = _causal_ready(day, trading, current_source, windows)
        dynamic = day_kind == "CURRENT" and level in {"HIGH", "LOW"} and current_source == trading
        families.append(Family(_family_id(trading, reference, day_kind, level), trading, reference, day_kind, level,
                               _iso(ready), dynamic))
    return tuple(families)


def _family_level_name(family: Family) -> str:
    # A distinct engine identity is required when two references happen to
    # print the same price.  The human-readable matrix fields remain separate.
    return family.family_id


def _static_level(family: Family, profile: Profile) -> BroadLevel:
    return BroadLevel(_family_level_name(family), profile.values()[family.reference_level])


def _profiles_for_phase(day: str, session: str, windows: Mapping[str, tuple[int, int]]) -> Profile:
    start, end = windows[session]
    return Profile.create(day, session, start, end)


def _source_path(root: Path, requests: Mapping[str, Any], day: str) -> Path:
    item = next((value for value in requests.values() if isinstance(value, dict) and value.get("category") == "TRAIN" and value.get("session_date") == day), None)
    if item is None:
        item = next((value for value in requests.values() if isinstance(value, dict) and value.get("category") == "DEPENDENCY" and value.get("session_date") == day), None)
    if not isinstance(item, dict):
        raise BaselineError(f"no acquired ES request for {day}")
    path = root / str(item["path"])
    if not path.is_file():
        raise BaselineError(f"missing acquired source: {path}")
    return path


def _new_runner(day: str, session: str, families: tuple[Family, ...], prior_profiles: Mapping[str, Profile], current_profiles: Mapping[str, Profile], config: L2Config,
                class_b: L2ClassBConfig = L2ClassBConfig()) -> BroadHistoricalRunner:
    levels = [_static_level(
                  family,
                  (current_profiles if family.reference_day == "CURRENT" else prior_profiles)[family.reference_session if family.reference_session != "RTH" else "NY"],
              )
              for family in families if not family.dynamic]
    return BroadHistoricalRunner(
        date=day, evidence_label="MAC_2025_ES_ONLY_TRAIN_BASELINE", levels=levels, config=config, class_b=class_b,
        strategy_id=f"{RUN_ID}.{session}", execution_policy=ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    )


def _profile_only_day(day: str, path: Path) -> dict[str, Profile]:
    """Build a dependency profile without running any strategy logic."""
    windows = _session_windows(day)
    profiles = {session: _profiles_for_phase(day, session, windows) for session in SESSION_ORDER}
    from databento import DBNStore
    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
        for row in batch:
            timestamp_ns = int(row["ts_recv"])
            session = next((name for name, (start, end) in windows.items() if start <= timestamp_ns < end), None)
            action = row["action"].decode() if hasattr(row["action"], "decode") else str(row["action"])
            if session is not None and action == "T":
                profiles[session].add(int(row["price"]) / 1_000_000_000, int(row["size"]))
    if any(not profile.volume_by_tick for profile in profiles.values()):
        raise BaselineError(f"empty dependency profile for {day}")
    return profiles


def _add_dynamic_levels(runner: historical.HistoricalL2Runner, families: tuple[Family, ...], extrema: Mapping[str, float]) -> None:
    static = [level for level in runner.interactions.levels if not any(level.name == family.family_id for family in families if family.dynamic)]
    dynamic = [BroadLevel(family.family_id, extrema[family.family_id]) for family in families
               if family.dynamic and family.family_id in extrema]
    runner.interactions.levels = tuple([*static, *dynamic])


def _route_day(day: str, path: Path, prior_profiles: dict[str, Profile], config: L2Config,
               known_current_profiles: dict[str, Profile] | None = None,
               capture_events: list[dict[str, Any]] | None = None,
               class_b: L2ClassBConfig = L2ClassBConfig()) -> dict[str, Any]:
    replay_started = wall_time.perf_counter()
    adapter_seconds = 0.0
    profile_seconds = 0.0
    strategy_seconds = 0.0
    windows = _session_windows(day)
    current_profiles = known_current_profiles or {session: _profiles_for_phase(day, session, windows) for session in SESSION_ORDER}
    profiles_are_sealed = known_current_profiles is not None
    families = build_families(day, prior_profiles, current_profiles)
    by_session = {session: tuple(family for family in families if family.trading_session == session) for session in SESSION_ORDER}
    runners: dict[str, historical.HistoricalL2Runner] = {}
    extrema: dict[str, float] = {}
    active_session: str | None = None
    adapter = FastNativeReplayAdapter()
    from databento import DBNStore
    records = 0

    def finish_session(session: str) -> None:
        if session in runners:
            runners[session].finish(windows[session][1])

    def ensure_session(timestamp_ns: int) -> str | None:
        nonlocal active_session
        for session in SESSION_ORDER:
            start, end = windows[session]
            if start <= timestamp_ns < end:
                if active_session != session:
                    if active_session is not None:
                        finish_session(active_session)
                    runners[session] = _new_runner(day, session, by_session[session], prior_profiles, current_profiles, config, class_b)
                    active_session = session
                return session
        return None

    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
      for raw in batch:
        records += 1
        timestamp_ns = int(raw["ts_recv"])
        if capture_events is not None and records % 1_000_000 == 0:
            elapsed = wall_time.perf_counter() - replay_started
            print(f"TAPE_PROGRESS date={day} records={records} market_time={_iso(timestamp_ns)} "
                  f"candidates={sum(len(runner.interaction_ledger) for runner in runners.values())} "
                  f"elapsed_seconds={elapsed:.1f}", flush=True)
        session = ensure_session(timestamp_ns)
        action = raw["action"].decode() if hasattr(raw["action"], "decode") else str(raw["action"])
        active = bool(session and (runners[session].interactions.active or runners[session].signals.position
                                   or getattr(runners[session].signals, "_entry_ready", ())))
        interest = frozenset(level.price for level in runners[session].interactions.levels) if session is not None else frozenset()
        adapter_started = wall_time.perf_counter()
        public = adapter.feed_array(raw, materialize_public=session is not None and
                                    (action == "T" or active),
                                    interest_prices=interest)
        adapter_seconds += wall_time.perf_counter() - adapter_started
        # Candidate-tape capture is deliberately independent of the baseline
        # runner's active-state materialization.  The validated raw top of
        # book is the authoritative executable path; strategy dispatch below
        # remains sparse and unchanged.  EventSpool retains only BBO changes,
        # executions, and entry-delay probes, so this does not retain every
        # unchanged raw row.
        if (capture_events is not None and session is not None
                and adapter._array_executable(raw)):
            raw_side = adapter._array_side(raw)
            is_execution = action == "T"
            capture_events.append({
                "timestamp_ns": timestamp_ns,
                "bid": int(raw["bid_px_00"]) / 1_000_000_000,
                "ask": int(raw["ask_px_00"]) / 1_000_000_000,
                "execution_price": int(raw["price"]) / 1_000_000_000 if is_execution else None,
                "execution_size": int(raw["size"]) if is_execution else 0,
                "aggressor": "BUY" if raw_side == "B" else "SELL" if raw_side == "A" else "",
                "session": session,
            })
        if public is None or session is None:
            continue
        if action != "T" and not active:
            continue
        if public.execution is not None:
            profile_started = wall_time.perf_counter()
            if not profiles_are_sealed:
                current_profiles[session].add(public.execution.price, public.execution.size)
            if session == "ASIA" or session == "EUROPE" or session == "NY":
                for family in by_session[session]:
                    if family.dynamic:
                        old = extrema.get(family.family_id)
                        if family.reference_level == "HIGH":
                            extrema[family.family_id] = public.execution.price if old is None else max(old, public.execution.price)
                        else:
                            extrema[family.family_id] = public.execution.price if old is None else min(old, public.execution.price)
                _add_dynamic_levels(runners[session], by_session[session], extrema)
            profile_seconds += wall_time.perf_counter() - profile_started
        strategy_started = wall_time.perf_counter()
        runners[session].observe_public(public)
        strategy_seconds += wall_time.perf_counter() - strategy_started
    if active_session is not None:
        finish_session(active_session)
    adapter.finish()
    for session in SESSION_ORDER:
        if not current_profiles[session].volume_by_tick:
            raise BaselineError(f"empty {day} {session} execution profile")
    if not profiles_are_sealed:
        prior_profiles.clear()
        prior_profiles.update(current_profiles)
    for runner in runners.values():
        runner.refresh_setup_ledger()

    interactions: list[dict[str, Any]] = []
    setups: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    session_metrics: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    family_trades: Counter[str] = Counter()
    for session, runner in runners.items():
        for row in runner.interaction_ledger:
            family = next((item for item in by_session[session] if item.family_id == row["level"]), None)
            if family is None:
                raise BaselineError(f"interaction family not in setup universe: {row['level']}")
            enriched = {**row, **family.__dict__, "source_date": day}
            interactions.append(enriched)
            family_counts[family.family_id] += 1
        for row in runner.setup_ledger:
            family = next((item for item in by_session[session] if item.family_id == row["level"]), None)
            enriched = {**row, **(family.__dict__ if family else {}), "source_date": day}
            setups.append(enriched)
        for row in runner.trade_ledger:
            family = next((item for item in by_session[session] if item.family_id == row["level"]), None)
            enriched = {**row, **(family.__dict__ if family else {}), "source_date": day}
            trades.append(enriched)
            if family:
                family_trades[family.family_id] += 1
        session_metrics.append({**runner.summary(), "records": records, "profile_volume": sum(current_profiles[session].volume_by_tick.values())})
    return {
        "date": day, "records": records, "profiles": current_profiles, "families": families,
        "interactions": interactions, "setups": setups, "trades": trades,
        "session_metrics": session_metrics, "family_counts": family_counts, "family_trades": family_trades,
        "market_events": capture_events or [],
        "timings": {"total_seconds": wall_time.perf_counter() - replay_started, "adapter_seconds": adapter_seconds,
                     "profile_seconds": profile_seconds, "strategy_seconds": strategy_seconds},
    }


def _run_target_job(arguments: tuple[Any, ...]) -> dict[str, Any]:
    if len(arguments) == 5:
        day, path_string, prior_profiles, current_profiles, config_payload = arguments
        class_b_payload = None
    else:
        day, path_string, prior_profiles, current_profiles, config_payload, class_b_payload = arguments
    config = L2Config(**config_payload)
    class_b = L2ClassBConfig(**class_b_payload) if class_b_payload is not None else L2ClassBConfig()
    return _route_day(day, Path(path_string), prior_profiles, config, current_profiles, class_b=class_b)


def _profile_job(arguments: tuple[str, str]) -> tuple[str, dict[str, Profile]]:
    day, path_string = arguments
    return day, _profile_only_day(day, Path(path_string))


def _profile_rows(profiles: Mapping[str, Profile]) -> list[dict[str, Any]]:
    rows = []
    for session, profile in profiles.items():
        rows.append({"date": profile.day, "session": session, "start": _iso(profile.start_ns), "end": _iso(profile.end_ns),
                     "volume": sum(profile.volume_by_tick.values()), **profile.values()})
    return rows


def _profile_payload(profile: Profile) -> dict[str, Any]:
    return {"day": profile.day, "session": profile.session, "start_ns": profile.start_ns, "end_ns": profile.end_ns,
            "volume_by_tick": [[int(key), int(value)] for key, value in sorted(profile.volume_by_tick.items())]}


def _profile_from_payload(payload: Mapping[str, Any]) -> Profile:
    profile = Profile.create(str(payload["day"]), str(payload["session"]), int(payload["start_ns"]), int(payload["end_ns"]))
    profile.volume_by_tick.update({int(key): int(value) for key, value in payload["volume_by_tick"]})
    return profile


def _profile_cache_path(output_root: Path, day: str) -> Path:
    return output_root / "_cache" / "profiles" / f"{day}.json"


def _checkpoint_path(output_root: Path, day: str) -> Path:
    return output_root / "_checkpoints" / f"{day}.json"


def _checkpoint_payload(result: Mapping[str, Any], *, source_path: Path, source_sha256: str,
                        config_sha256: str, semantic_sha256: str) -> dict[str, Any]:
    return {
        "status": "DATE_COMPLETE", "date": result["date"], "source_path": str(source_path),
        "source_sha256": source_sha256, "config_sha256": config_sha256, "semantic_sha256": semantic_sha256,
        "result": {"date": result["date"], "records": result["records"],
                   "profiles": [_profile_payload(profile) for profile in result["profiles"].values()],
                   "families": [item.__dict__ for item in result["families"]],
                   "interactions": result["interactions"], "setups": result["setups"], "trades": result["trades"],
                   "session_metrics": result["session_metrics"], "family_counts": dict(result["family_counts"]),
                   "family_trades": dict(result["family_trades"]), "timings": result.get("timings", {})},
    }


def _checkpoint_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(payload["result"])
    raw["profiles"] = {row["session"]: _profile_from_payload(row) for row in raw["profiles"]}
    raw["families"] = tuple(Family(**row) for row in raw["families"])
    raw["family_counts"] = Counter(raw["family_counts"])
    raw["family_trades"] = Counter(raw["family_trades"])
    return raw


def _load_valid_checkpoint(path: Path, *, day: str, source_path: Path, source_sha256: str,
                           config_sha256: str, semantic_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "DATE_COMPLETE" or payload.get("date") != day
                or payload.get("source_path") != str(source_path)
                or payload.get("source_sha256") != source_sha256
                or payload.get("config_sha256") != config_sha256
                or payload.get("semantic_sha256") != semantic_sha256):
            return None
        return _checkpoint_result(payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_profile_cache(path: Path, profiles: Mapping[str, Profile], *, day: str, source_path: Path,
                         source_sha256: str, semantic_sha256: str) -> None:
    _json_write(path, {"status": "PROFILE_CACHE_COMPLETE", "date": day, "source_path": str(source_path),
                       "source_sha256": source_sha256, "semantic_sha256": semantic_sha256,
                       "profiles": [_profile_payload(profile) for profile in profiles.values()]})


def _load_profile_cache(path: Path, *, day: str, source_path: Path, source_sha256: str,
                        semantic_sha256: str) -> dict[str, Profile] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "PROFILE_CACHE_COMPLETE" or payload.get("date") != day
                or payload.get("source_path") != str(source_path)
                or payload.get("source_sha256") != source_sha256
                or payload.get("semantic_sha256") != semantic_sha256):
            return None
        return {str(row["session"]): _profile_from_payload(row) for row in payload["profiles"]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _aggregate(setups: list[dict[str, Any]], trades: list[dict[str, Any]], session_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    r_values = [float(row["r_multiple"]) for row in trades if row.get("r_multiple") is not None]
    gross_r = [float(row["gross_r"]) for row in trades if row.get("gross_r") is not None]
    net = [float(row["net_pnl_usd"]) for row in trades if row.get("net_pnl_usd") is not None]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    winners = sum(value > 0 for value in r_values)
    losers = sum(value < 0 for value in r_values)
    gross_profit = sum(value for value in net if value > 0)
    gross_loss = -sum(value for value in net if value < 0)
    profitable_days = sum(any(float(row.get("net_pnl_usd", 0) or 0) > 0 for row in trades if row.get("date") == day)
                          for day in TRAIN_DATES)
    return {
        "total_setups": len(setups),
        "total_trades": len(trades), "winners": winners, "losers": losers,
        "net_r": sum(r_values), "net_pnl_usd": sum(net), "gross_r": sum(gross_r),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "max_drawdown_r": max_dd, "profitable_session_ratio": profitable_days / len(TRAIN_DATES),
        "sessions": len(session_metrics),
    }


def run_baseline(*, data_root: Path = DATA_ROOT, output_root: Path = OUTPUT_ROOT, full_audit: bool = True,
                 workers: int = 1, resume: bool = False) -> dict[str, Any]:
    started = wall_time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, full_record_scan=full_audit)
    if audit["status"] != "PASS":
        raise BaselineError(f"MAC_2025_DOWNLOAD_AUDIT failed: {audit['failed_files']}")
    _, requests = _manifest(data_root)
    config = L2Config()
    config_payload = {field.name: getattr(config, field.name) for field in fields(config)}
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    semantic_hash = _semantic_sha256()
    source_paths = {day: _source_path(data_root, requests, day) for day in [DEPENDENCY_DATE, *TRAIN_DATES]}
    source_hashes = {str(path): str(next(row["sha256"] for row in requests.values() if isinstance(row, dict) and row.get("path") == str(path.relative_to(data_root)))) for path in source_paths.values()}
    _json_write(output_root / "source-audit.json", audit)

    # Profile caches are immutable, source-hash-bound preparation artifacts.
    profile_by_day: dict[str, dict[str, Profile]] = {}
    profile_jobs: list[tuple[str, str]] = []
    for day, path in source_paths.items():
        cached = _load_profile_cache(_profile_cache_path(output_root, day), day=day, source_path=path,
                                     source_sha256=source_hashes[str(path)], semantic_sha256=semantic_hash)
        if cached is None:
            profile_jobs.append((day, str(path)))
        else:
            profile_by_day[day] = cached
    if profile_jobs:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                profile_futures = {pool.submit(_profile_job, job): job[0] for job in profile_jobs}
                for future in as_completed(profile_futures):
                    day, profiles = future.result()
                    profile_by_day[day] = profiles
                    _write_profile_cache(_profile_cache_path(output_root, day), profiles, day=day,
                                         source_path=source_paths[day], source_sha256=source_hashes[str(source_paths[day])], semantic_sha256=semantic_hash)
        else:
            for job in profile_jobs:
                day, profiles = _profile_job(job)
                profile_by_day[day] = profiles
                _write_profile_cache(_profile_cache_path(output_root, day), profiles, day=day,
                                     source_path=source_paths[day], source_sha256=source_hashes[str(source_paths[day])], semantic_sha256=semantic_hash)

    results_by_day: dict[str, dict[str, Any]] = {}
    for day in TRAIN_DATES:
        cached = _load_valid_checkpoint(_checkpoint_path(output_root, day), day=day, source_path=source_paths[day],
                                        source_sha256=source_hashes[str(source_paths[day])], config_sha256=config_hash,
                                        semantic_sha256=semantic_hash)
        if cached is not None:
            results_by_day[day] = cached
    remaining = [day for day in TRAIN_DATES if day not in results_by_day]
    print(f"COMPLETED_DATES={sorted(results_by_day)}", flush=True)
    print(f"REMAINING_DATES={remaining}", flush=True)
    jobs = []
    for day in remaining:
        prior_day = DEPENDENCY_DATE if day == TRAIN_DATES[0] else TRAIN_DATES[TRAIN_DATES.index(day) - 1]
        jobs.append((day, str(source_paths[day]), profile_by_day[prior_day], profile_by_day[day], config_payload))
    if workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_target_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                day = futures[future]
                result = future.result()
                results_by_day[day] = result
                _json_write(_checkpoint_path(output_root, day), _checkpoint_payload(
                    result, source_path=source_paths[day], source_sha256=source_hashes[str(source_paths[day])],
                    config_sha256=config_hash, semantic_sha256=semantic_hash,
                ))
                print(f"DATE_COMPLETE={day}", flush=True)
    else:
        for day, path_string, prior, current, payload in jobs:
            result = _run_target_job((day, path_string, prior, current, payload))
            results_by_day[day] = result
            _json_write(_checkpoint_path(output_root, day), _checkpoint_payload(
                result, source_path=source_paths[day], source_sha256=source_hashes[str(source_paths[day])],
                config_sha256=config_hash, semantic_sha256=semantic_hash,
            ))
            print(f"DATE_COMPLETE={day}", flush=True)
    if set(results_by_day) != set(TRAIN_DATES):
        raise BaselineError("not all TRAIN dates completed; aggregate was not published")

    all_interactions: list[dict[str, Any]] = []
    all_setups: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    family_specs: dict[str, Family] = {}
    family_counts: Counter[str] = Counter()
    family_trades: Counter[str] = Counter()
    for index, day in enumerate(TRAIN_DATES, 1):
        result = results_by_day[day]
        print(f"TRAIN {index}/{len(TRAIN_DATES)} {day}", flush=True)
        all_interactions.extend(result["interactions"])
        all_setups.extend(result["setups"])
        all_trades.extend(result["trades"])
        all_sessions.extend(result["session_metrics"])
        all_profiles.extend(_profile_rows(result["profiles"]))
        for family in result["families"]:
            family_specs[family.family_id] = family
        family_counts.update(result["family_counts"])
        family_trades.update(result["family_trades"])
    families = []
    for family in family_specs.values():
        families.append({**family.__dict__, "causally_valid": True, "target_sessions_eligible": len(TRAIN_DATES),
                         "setups_seen": family_counts[family.family_id], "trades_taken": family_trades[family.family_id]})
    zero_trade = [row["family_id"] for row in families if row["trades_taken"] == 0]
    aggregate = _aggregate(all_setups, all_trades, all_sessions)
    source_hashes = {str(_source_path(data_root, requests, day)): _sha256(_source_path(data_root, requests, day)) for day in [DEPENDENCY_DATE, *TRAIN_DATES]}
    run_manifest = {
        "run_id": RUN_ID, "status": "COMPLETE", "dataset": DATASET, "source_model": SOURCE_MODEL,
        "schema": "mbp-10", "market_data": "native ES only", "mbo_used": False, "mes_market_data_used": False,
        "synthetic_mes_used": False, "execution_policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
        "train_dates": list(TRAIN_DATES), "dependency_dates": [DEPENDENCY_DATE], "validation_performance": False,
        "final_oos_accessed": False, "source_hashes": source_hashes, "config_sha256": config_hash,
        "semantic_sha256": semantic_hash, "config": config_payload, "audit": audit,
        "profile_cache": True, "date_checkpoints": True, "workers": workers,
        "completed_dates": list(TRAIN_DATES), "runtime_seconds": wall_time.monotonic() - started,
        "outcome_parameter_selection": False, "optimization": False,
    }
    setup_universe = {
        "run_id": RUN_ID, "level_types": ["PRIOR_POC", "PRIOR_VAH", "PRIOR_VAL", "PRIOR_HIGH", "PRIOR_LOW", "CURRENT_HIGH", "CURRENT_LOW"],
        "trading_sessions": list(SESSION_ORDER), "cross_session_levels_enabled": True,
        "current_profile_poc_vah_val": "enabled only after the source session closes; no intraprofile lookahead",
        "families": families, "rejected_for_causality": [], "zero_trade_families": zero_trade,
    }
    _json_write(output_root / "source-audit.json", audit)
    _json_write(output_root / "run-manifest.json", run_manifest)
    _json_write(output_root / "train-date-manifest.json", {"train_dates": list(TRAIN_DATES), "dependency_dates": [DEPENDENCY_DATE]})
    _json_write(output_root / "config.json", {"config": config_payload, "sha256": config_hash})
    _json_write(output_root / "setup-universe.json", setup_universe)
    _csv_write(output_root / "trade-ledger.csv", all_trades)
    _csv_write(output_root / "setup-ledger.csv", all_setups)
    _csv_write(output_root / "interaction-features.csv", all_interactions)
    _csv_write(output_root / "per-session-metrics.csv", all_sessions)
    _csv_write(output_root / "per-setup-family-metrics.csv", families)
    _csv_write(output_root / "profiles.csv", all_profiles)
    _json_write(output_root / "aggregate-metrics.json", aggregate)
    _json_write(output_root / "zero-trade-families.json", {"families": zero_trade})
    _json_write(output_root / "runtime.json", {"runtime_seconds": run_manifest["runtime_seconds"], "records": audit["record_count"]})
    return {"audit": audit, "setup_universe": setup_universe, "aggregate": aggregate, "runtime_seconds": run_manifest["runtime_seconds"], "output_root": str(output_root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--metadata-only-audit", action="store_true", help="skip the second full record scan")
    parser.add_argument("--workers", type=int, default=1, help="parallel TRAIN date workers; results remain date-sorted")
    parser.add_argument("--resume", action="store_true", help="reuse hash-valid profile caches and date checkpoints")
    args = parser.parse_args(argv)
    if args.audit_only:
        result = audit_dataset(args.data_root, full_record_scan=not args.metadata_only_audit)
        _json_write(args.output_root / "source-audit.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    result = run_baseline(data_root=args.data_root, output_root=args.output_root, full_audit=not args.metadata_only_audit,
                          workers=max(1, args.workers), resume=args.resume)
    print(json.dumps({"MAC_2025_DOWNLOAD_AUDIT": result["audit"]["status"], "TRAIN_BASELINE_UNIVERSE": "PASS",
                      "TOTAL_SETUP_FAMILIES": len(result["setup_universe"]["families"]),
                      "CAUSALLY_VALID_SETUP_FAMILIES": sum(row["causally_valid"] for row in result["setup_universe"]["families"]),
                      "ZERO_TRADE_FAMILIES": len(result["setup_universe"]["zero_trade_families"]),
                      **result["aggregate"], "runtime_seconds": result["runtime_seconds"],
                      "READY_TO_DESIGN_FULL_OPTUNA_SEARCH": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
