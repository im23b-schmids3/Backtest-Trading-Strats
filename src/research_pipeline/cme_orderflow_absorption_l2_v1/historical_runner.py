"""Fail-closed historical replay plumbing for ``CMEOrderflowAbsorption.ES_L2_V1``.

This module deliberately separates three concerns:

* private MBO reconstruction, where order ids are necessary to calculate
  aggregate displayed depth;
* the public MBP-10/Execution stream consumed by the L2 model; and
* deterministic replay/accounting artifacts.

It has no network client and never discovers files.  A future real run must
name every session and every input in its session manifest.  The command is
provided for a later first frozen replay; this implementation task does not
invoke it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Literal

from .model import (
    ENTRY_LATENCY_NS, ES_COMMISSION, ES_POINT_VALUE, MAX_CONFIRMATION_NS, L2Config, L2Interaction,
    L2InteractionEngine, L2Position, L2Setup, L2SignalEngine, L2ValidationError,
    MBOEvent, MBOToMBP10View, MBP10Snapshot, MBP10Update, StructuralLevel,
    TICK, _point_price,
)


F_SNAPSHOT = 32
F_LAST = 128
RAW_PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = 9_223_372_036_854_775_807
MAX_TOLERATED_OFFBOOK_ANOMALIES_PER_SESSION = 10
STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V1"
WEIGHTS_LABEL = "L2_V1_PREDECLARED_RESEARCH_WEIGHTS"
MAY_LABEL = "UNSEEN_MAY_2026_RETROSPECTIVE_HOLDOUT"
RETRO_LABEL = "NOT_STRICT_CHRONOLOGICAL_OOS_RETROSPECTIVE"
AUGUST_LABEL = "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE"
MAY_REPLAY_LABEL = "FIRST_BROAD_HISTORICAL_L2_V1_REPLAY_MAY_2026"
MAY_DATES = (
    "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
)
MAY_PRIOR_RTH = {day: ("2026-05-01" if index == 0 else MAY_DATES[index - 1]) for index, day in enumerate(MAY_DATES)}
RTH_START_SECONDS = 13 * 3600 + 30 * 60
HARD_CUTOFF_SECONDS = 22 * 3600 + 45 * 60
CUTOFF_QUOTE_LOOKBACK_NS = 1_000_000_000


class HistoricalReplayError(RuntimeError):
    """A historical source or immutable replay contract is incomplete."""


def _parse_utc_ns(value: object, *, field: str) -> int:
    if not isinstance(value, str):
        raise HistoricalReplayError(f"sealed source coverage is unknown: {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalReplayError(f"sealed source coverage timestamp is invalid: {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise HistoricalReplayError(f"sealed source coverage timestamp is not UTC: {field}")
    return int(parsed.timestamp() * 1_000_000_000)


def validate_declared_session_coverage(day: str, inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fail closed on unknown or insufficient declared source coverage.

    Coverage is an acquisition/DBN contract, not an assertion that the market
    produced a book, quote, or trade at the hard-cutoff instant.
    """
    cutoff_ns = _clock_ns(day, HARD_CUTOFF_SECONDS)
    expected_starts = {"ES_MBO_L3": _clock_ns(day, 0), "MES_NATIVE_EXECUTION": _clock_ns(day, RTH_START_SECONDS)}
    result: dict[str, Any] = {}
    for label, start_ns in expected_starts.items():
        record = inputs.get(label)
        if not isinstance(record, dict):
            raise HistoricalReplayError(f"sealed source coverage is unknown: {label} {day}")
        declared_start = _parse_utc_ns(record.get("start"), field=f"{label}.start")
        declared_end = _parse_utc_ns(record.get("end"), field=f"{label}.end")
        if declared_start != start_ns or declared_end <= declared_start:
            raise HistoricalReplayError(f"sealed source coverage chronology mismatch: {label} {day}")
        if declared_end < cutoff_ns:
            raise HistoricalReplayError(f"sealed source coverage ends before frozen 22:45 UTC cutoff: {label} {day}")
        result[label] = {"declared_start_ns": declared_start, "declared_end_ns": declared_end}
    return result


@dataclass(frozen=True)
class PrivateMBORecord:
    """Source-only record.  ``order_id`` cannot escape this adapter boundary."""

    timestamp_ns: int
    action: Literal["A", "C", "M", "F", "R", "T"]
    side: Literal["B", "A"]
    price: float
    size: int
    order_id: int
    flags: int = 0
    raw_price: int | None = None


@dataclass
class SourceAnomaly:
    timestamp_ns: int
    action: str
    side: str
    raw_price: int
    normalized_price: float
    size: int
    flags: int
    order_id: int
    entered_top_ten: bool = False
    affected_bbo: bool = False
    terminal_action: str | None = None


@dataclass(frozen=True)
class PublicBookEvent:
    """The only book event handed to the L2 strategy-facing replay layer."""

    timestamp_ns: int
    snapshot: MBP10Snapshot
    update: MBP10Update | None
    execution: Any | None


def _aggressor(resting_side: str) -> Literal["BUY", "SELL", "UNKNOWN"]:
    return "BUY" if resting_side == "A" else "SELL" if resting_side == "B" else "UNKNOWN"


class HistoricalMBOToMBP10Adapter:
    """MBO snapshot gate plus private aggregate reconstruction.

    Databento historical MBO uses an initialization snapshot made of an ``R``
    reset and ``A`` records.  It is not safe to use any ordinary record until
    that snapshot ends with ``F_LAST``.  Snapshot records seed public L2 depth
    only; they can never create an interaction or an execution signal.
    """

    def __init__(self) -> None:
        self._view = MBOToMBP10View()
        self._snapshot_open = False
        self._book_valid = False
        self._anomalies: dict[int, SourceAnomaly] = {}
        self._completed_anomalies: list[SourceAnomaly] = []

    @staticmethod
    def _offbook_source_price(price: float) -> bool:
        try:
            _point_price(price)
        except L2ValidationError:
            return True
        return False

    def _audit_anomalies(self, record: PrivateMBORecord) -> None:
        existing = self._anomalies.get(record.order_id)
        if existing is not None and record.action in {"C", "F", "M"}:
            existing.terminal_action = record.action
            if record.action in {"C", "M"}:
                self._completed_anomalies.append(existing)
                del self._anomalies[record.order_id]
        if record.action in {"A", "M"} and self._offbook_source_price(record.price):
            raw_price = record.raw_price if record.raw_price is not None else int(record.price * RAW_PRICE_SCALE)
            anomaly = SourceAnomaly(record.timestamp_ns, record.action, record.side, raw_price, record.price,
                                    record.size, record.flags, record.order_id)
            self._anomalies[record.order_id] = anomaly
            if len(self._anomalies) + len(self._completed_anomalies) > MAX_TOLERATED_OFFBOOK_ANOMALIES_PER_SESSION:
                raise HistoricalReplayError("SOURCE_INTEGRITY_OFFBOOK_ANOMALY_LIMIT_EXCEEDED")
        for order_id, anomaly in list(self._anomalies.items()):
            order = self._view.order(order_id)
            if order is None:
                anomaly.terminal_action = anomaly.terminal_action or "REMOVED"
                self._completed_anomalies.append(anomaly)
                continue
            anomaly.entered_top_ten = self._view.is_price_in_top_ten(order.side, order.price)
            anomaly.affected_bbo = self._view.is_price_best(order.side, order.price)
            if anomaly.entered_top_ten or anomaly.affected_bbo:
                raise HistoricalReplayError(
                    f"SOURCE_INTEGRITY_OFFBOOK_ANOMALY_EXPOSED raw_price={anomaly.raw_price} "
                    f"normalized_price={anomaly.normalized_price} order_id={order_id}"
                )

    def _is_anomaly_record(self, record: PrivateMBORecord) -> bool:
        return record.order_id in self._anomalies or (
            record.action in {"A", "M"} and self._offbook_source_price(record.price)
        )

    def source_integrity_diagnostics(self) -> list[dict[str, Any]]:
        rows = [*self._completed_anomalies, *self._anomalies.values()]
        return [{"timestamp_ns": row.timestamp_ns, "action": row.action, "side": row.side,
                 "raw_price": row.raw_price, "normalized_price": row.normalized_price,
                 "size": row.size, "flags": row.flags, "entered_top_ten": row.entered_top_ten,
                 "affected_bbo": row.affected_bbo, "terminal_action": row.terminal_action}
                for row in rows]

    @property
    def book_valid(self) -> bool:
        return self._book_valid

    def feed(self, record: PrivateMBORecord, *, materialize_public: bool = True) -> PublicBookEvent | None:
        snapshot = bool(record.flags & F_SNAPSHOT)
        last = bool(record.flags & F_LAST)
        if snapshot:
            if not self._snapshot_open:
                if record.action != "R":
                    raise HistoricalReplayError("MBO snapshot must begin with reset before F_LAST")
                self._snapshot_open = True
                self._book_valid = False
            elif record.action not in {"A", "R"}:
                raise HistoricalReplayError("MBO snapshot contains non-initialization action")
            public_snapshot, update = self._view.apply(MBOEvent(
                record.timestamp_ns, record.action, record.side, record.price, record.size, record.order_id,
            ), materialize_snapshot=False, materialize_update=False)
            self._audit_anomalies(record)
            if last:
                if record.action != "A":
                    raise HistoricalReplayError("MBO snapshot F_LAST must complete an add sequence")
                self._snapshot_open = False
                self._book_valid = True
                if not materialize_public:
                    return None
                public_snapshot = self._view.snapshot(record.timestamp_ns)
                if public_snapshot is None:
                    raise HistoricalReplayError("MBO F_LAST did not materialize the initial MBP-10 snapshot")
                return PublicBookEvent(record.timestamp_ns, public_snapshot, update, None)
            return None
        if self._snapshot_open:
            raise HistoricalReplayError("ordinary MBO record precedes F_LAST snapshot completion")
        if not self._book_valid:
            raise HistoricalReplayError("missing causal MBO snapshot initialization")
        if record.action == "R":
            raise HistoricalReplayError("ordinary reset invalidates the book; a new F_SNAPSHOT is required")
        # Existing sealed MBO reconstruction treats F as execution evidence
        # without mutating displayed depth: a later C may remove the same
        # displayed order. Preserve that exact public-book convention.
        anomaly_record = self._is_anomaly_record(record)
        book_action = "T" if record.action == "F" else record.action
        public_snapshot, update = self._view.apply(MBOEvent(
            record.timestamp_ns, book_action, record.side, record.price, record.size, record.order_id,
        ), materialize_snapshot=False, materialize_update=not anomaly_record)
        self._audit_anomalies(record)
        if not materialize_public:
            return None
        public_snapshot = self._view.snapshot(record.timestamp_ns)
        if public_snapshot is None:
            raise HistoricalReplayError("public MBO record did not materialize an MBP-10 snapshot")
        execution = None
        if not anomaly_record and record.action in {"F", "T"} and record.size > 0:
            from .model import Execution  # Keeps public source types explicit at this boundary.
            execution = Execution(record.timestamp_ns, record.price, record.size, _aggressor(record.side))
        return PublicBookEvent(record.timestamp_ns, public_snapshot, None if anomaly_record else update, execution)

    def finish(self) -> None:
        if self._snapshot_open or not self._book_valid:
            raise HistoricalReplayError("incomplete MBO source: no completed F_LAST snapshot")


def assert_no_order_identity_in_strategy_layer() -> None:
    """Assert the adapter is the sole location permitted to hold MBO identity."""
    public = (PublicBookEvent, MBP10Snapshot, MBP10Update, L2Interaction, L2Setup, L2Position)
    for item in public:
        names = {field.name for field in fields(item)}
        if "order_id" in names or "resting_order_id" in names or "queue_id" in names:
            raise HistoricalReplayError(f"order identity leaked into strategy model: {item.__name__}")


def contract_for_config(*, strategy_id: str, config: L2Config, first_run_policy: str) -> dict[str, Any]:
    """Canonical contract for an explicitly supplied, immutable L2 configuration."""
    return {
        "strategy_id": strategy_id,
        "weights_label": config.weights_label,
        "weights": {
            "aggression_score": config.aggression_weight,
            "restoration_score": config.restoration_weight,
            "price_resistance_score": config.price_resistance_weight,
            "persistence_score": config.persistence_weight,
            "multi_level_support_score": config.multi_level_support_weight,
            "false_refill_penalty": config.false_refill_penalty_weight,
        },
        "interaction": {"vicinity_ticks": 4, "recovery_windows_ms": [100, 250, 500, 1000]},
        "execution": {
            "confirmation_window_seconds": [5.0, 15.0], "confirmation_favorable_ticks": 3,
            "entry_latency_ms": ENTRY_LATENCY_NS / 1_000_000, "stop_buffer_ticks": 5,
            "target_r": 3.0, "risk_budget_usd": 250.0, "es_first": True,
            "mes_fallback": True, "max_es_contracts": 6, "max_mes_contracts": 60,
        },
        "configuration": {field.name: getattr(config, field.name) for field in fields(config)},
        "development_only_parameters": [],
        "first_run_policy": first_run_policy,
    }


def frozen_contract() -> dict[str, Any]:
    """Canonical, pre-outcome L2 V1 freeze surfaced in every V1 report."""
    return contract_for_config(
        strategy_id=STRATEGY_ID,
        config=L2Config(),
        first_run_policy="FIRST_BROAD_HISTORICAL_L2_V1_REPLAY; NO_OUTCOME_BASED_PARAMETER_SELECTION_BEFORE_RUN",
    )


def _iso_day(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).date().isoformat()


def _quote(snapshot: MBP10Snapshot) -> tuple[float, float] | None:
    if not snapshot.bids or not snapshot.asks:
        return None
    bid, ask = snapshot.bids[0].price, snapshot.asks[0].price
    return (bid, ask) if ask > bid else None


class HistoricalL2Runner:
    """Incremental single-session runner over public MBP-10 and execution events."""

    def __init__(self, *, date: str, evidence_label: str, levels: Iterable[StructuralLevel], config: L2Config = L2Config(), strategy_id: str = STRATEGY_ID,
                 require_native_mes_for_fallback: bool = False) -> None:
        assert_no_order_identity_in_strategy_layer()
        self.date, self.evidence_label, self.config, self.strategy_id = date, evidence_label, config, strategy_id
        self.require_native_mes_for_fallback = require_native_mes_for_fallback
        self.interactions = L2InteractionEngine(list(levels), config)
        self.signals = L2SignalEngine(config)
        self.completed_seen = 0
        self.es_quote: tuple[float, float] | None = None
        self.mes_quote: tuple[float, float] | None = None
        self.es_quote_timestamp_ns: int | None = None
        self.mes_quote_timestamp_ns: int | None = None
        self.setup_ledger: list[dict[str, Any]] = []
        self.interaction_ledger: list[dict[str, Any]] = []
        self.trade_ledger: list[dict[str, Any]] = []
        self.diagnostic_events: list[dict[str, Any]] = []
        # This is opt-in evidence plumbing for a declared source boundary that
        # precedes the frozen 22:45 hard cutoff.  It never substitutes for the
        # normal hard-flat/session-end behavior used by complete inputs.
        self.source_end_unresolved: list[dict[str, Any]] = []
        self.mes_execution_unavailable: list[dict[str, Any]] = []
        # Source-layer audit data is intentionally separate from all strategy
        # ledgers: it cannot participate in features, scoring, or execution.
        self.source_integrity_diagnostics: list[dict[str, Any]] = []

    def _new_completed(self) -> None:
        """Register only newly closed interactions; never rebuild prior summaries."""
        for interaction in self.interactions.completed[self.completed_seen:]:
            features, scores, quality = interaction.feature_inputs(), interaction.component_scores(), interaction.quality()
            accepted, reasons = interaction.qualification()
            row = {
                "interaction_id": interaction.interaction_id, "date": self.date, "level": interaction.level.name,
                "direction": interaction.direction, "interaction_start_ns": interaction.start_ns,
                "interaction_end_ns": interaction.end_ns, "interaction_end_price": interaction.end_price,
                "zone_low": interaction.zone_low, "zone_high": interaction.zone_high, **features, **scores, **quality,
                "accepted": accepted, "rejection_reasons": ";".join(reasons), "weights_label": self.config.weights_label,
            }
            self.interaction_ledger.append(row)
            setup = self.signals.register_completed(interaction)
            self.setup_ledger.append({**row, "setup_id": setup.setup_id if setup else None,
                                      "confirmation_status": "PENDING" if setup else "REJECTED"})
        self.completed_seen = len(self.interactions.completed)

    def _attempt_entry(self, timestamp_ns: int) -> None:
        if self.es_quote is None:
            return
        for setup_id in sorted(self.signals.pending):
            setup = self.signals.pending[setup_id]
            if setup.state != "CONFIRMED" or setup.entry_ready_ns is None or timestamp_ns < setup.entry_ready_ns:
                continue
            # A missing native MES feed is a source limitation, not a reason to
            # invent an ES-derived MES fill.  ES remains executable whenever its
            # frozen risk sizing permits it.  The standard runner retains its
            # historical INSUFFICIENT_RISK_BUDGET classification unless this
            # explicit source-unavailable policy is enabled by a later runner.
            if getattr(self, "require_native_mes_for_fallback", False) and self.mes_quote is None:
                from .model import initial_prices, size_for_instrument
                es_prices = initial_prices(
                    setup.interaction.direction, self.es_quote[0], self.es_quote[1],
                    setup.interaction.zone_low, setup.interaction.zone_high,
                )
                if int(size_for_instrument(es_prices, "ES")["contracts"]) < 1:
                    setup.state, setup.terminal_reason = "FAILED", "MES_EXECUTION_UNAVAILABLE"
                    self.mes_execution_unavailable.append({"setup_id": setup_id, "timestamp_ns": timestamp_ns})
                    self.diagnostic_events.append({"event": "MES_EXECUTION_UNAVAILABLE", "setup_id": setup_id, "timestamp_ns": timestamp_ns})
                    continue
            position = self.signals.try_enter(
                setup_id, timestamp_ns=timestamp_ns, es_bid=self.es_quote[0], es_ask=self.es_quote[1],
                mes_bid=self.mes_quote[0] if self.mes_quote else None, mes_ask=self.mes_quote[1] if self.mes_quote else None,
            )
            if position is not None:
                self.diagnostic_events.append({"event": "ENTRY", "setup_id": setup_id, "timestamp_ns": timestamp_ns,
                                               "instrument": position.instrument, "contracts": position.contracts})
                break

    def _close_position(self, timestamp_ns: int, reference: float, reason: str) -> None:
        position = self.signals.position
        if position is None:
            return
        long = position.prices["direction"] == "LONG"
        exit_price = reference - TICK if long else reference + TICK
        points = exit_price - float(position.prices["entry"]) if long else float(position.prices["entry"]) - exit_price
        point_value, commission = (ES_POINT_VALUE, ES_COMMISSION) if position.instrument == "ES" else (5.0, 1.25)
        gross = points * point_value * position.contracts
        fees = 2 * commission * position.contracts
        initial_risk = float(position.prices["entry"]) - float(position.prices["stop_exit"])
        initial_risk = abs(initial_risk) * point_value * position.contracts + fees
        self.trade_ledger.append({
            "trade_id": f"L2T:{position.setup.setup_id}", "setup_id": position.setup.setup_id, "date": self.date,
            "interaction_id": position.setup.interaction.interaction_id, "direction": position.prices["direction"],
            "level": position.setup.interaction.level.name, "instrument": position.instrument, "contracts": position.contracts,
            "entry_timestamp_ns": position.entry_timestamp_ns, "entry": position.prices["entry"], "stop": position.prices["stop"],
            "target": position.prices["target"], "exit_timestamp_ns": timestamp_ns, "exit": exit_price,
            "exit_reason": reason, "gross_pnl_usd": gross, "total_costs_usd": fees, "net_pnl_usd": gross - fees,
            "r_multiple": (gross - fees) / initial_risk if initial_risk else None,
        })
        self.diagnostic_events.append({"event": "EXIT", "setup_id": position.setup.setup_id, "timestamp_ns": timestamp_ns, "reason": reason})
        self.signals.position = None

    def _manage_position(self, timestamp_ns: int, quote: tuple[float, float] | None, instrument: str) -> None:
        position = self.signals.position
        if position is None or position.instrument != instrument or quote is None:
            return
        long = position.prices["direction"] == "LONG"
        adverse, favorable = (quote[0], quote[0]) if long else (quote[1], quote[1])
        if (long and adverse <= float(position.prices["stop"])) or (not long and adverse >= float(position.prices["stop"])):
            self._close_position(timestamp_ns, adverse, "STOP")
        elif (long and favorable >= float(position.prices["target"])) or (not long and favorable <= float(position.prices["target"])):
            self._close_position(timestamp_ns, favorable, "TARGET")

    def observe_public(self, event: PublicBookEvent) -> None:
        self.interactions.advance(event.timestamp_ns)
        self._new_completed()
        self.es_quote = _quote(event.snapshot)
        if self.es_quote is not None:
            self.es_quote_timestamp_ns = event.timestamp_ns
        self._manage_position(event.timestamp_ns, self.es_quote, "ES")
        self.interactions.observe_snapshot(event.snapshot, event.update)
        self.signals.advance(event.timestamp_ns)
        if event.execution is not None:
            self.interactions.observe_execution(event.execution)
            self._new_completed()
            self.signals.observe_execution(event.execution)
        self._attempt_entry(event.timestamp_ns)

    def observe_mes_quote(self, timestamp_ns: int, bid: float, ask: float) -> None:
        self.mes_quote = (_point_price(bid), _point_price(ask))
        if self.mes_quote[1] <= self.mes_quote[0]:
            raise HistoricalReplayError("invalid native MES BBO")
        self.mes_quote_timestamp_ns = timestamp_ns
        self._manage_position(timestamp_ns, self.mes_quote, "MES")
        self._attempt_entry(timestamp_ns)

    def finish(self, timestamp_ns: int) -> None:
        self.interactions.finish_rth(timestamp_ns)
        self._new_completed()
        self.signals.advance(timestamp_ns)
        if self.signals.position is not None:
            quote = self.es_quote if self.signals.position.instrument == "ES" else self.mes_quote
            if quote is None:
                raise HistoricalReplayError("cannot close active position without native executable quote")
            self._close_position(timestamp_ns, quote[0] if self.signals.position.prices["direction"] == "LONG" else quote[1], "SESSION_END")

    def mark_source_end_incomplete(self, timestamp_ns: int) -> None:
        """Close no economics at a declared early source boundary.

        The frozen session remains open until 22:45 UTC.  When a local source
        stops earlier, a position or confirmation that cannot be observed to a
        terminal result is retained as an explicit incomplete observation.  In
        particular, this method never calls :meth:`finish` or :meth:`force_flat`.
        """
        self._new_completed()
        self.signals.advance(timestamp_ns)
        for setup in self.signals.pending.values():
            if setup.terminal_reason is not None:
                continue
            if setup.state == "CONFIRMED":
                setup.state, setup.terminal_reason = "SOURCE_INCOMPLETE", "EXECUTION_UNRESOLVED_SOURCE_INCOMPLETE"
                self.source_end_unresolved.append({"setup_id": setup.setup_id, "timestamp_ns": timestamp_ns,
                                                   "reason": setup.terminal_reason})
            elif timestamp_ns <= (setup.interaction.end_ns or 0) + MAX_CONFIRMATION_NS:
                setup.state, setup.terminal_reason = "SOURCE_INCOMPLETE", "CONFIRMATION_UNRESOLVED_SOURCE_INCOMPLETE"
                self.source_end_unresolved.append({"setup_id": setup.setup_id, "timestamp_ns": timestamp_ns,
                                                   "reason": setup.terminal_reason})
        if self.signals.position is not None:
            position = self.signals.position
            position.setup.state, position.setup.terminal_reason = "SOURCE_INCOMPLETE", "UNRESOLVED_SOURCE_END"
            self.source_end_unresolved.append({"setup_id": position.setup.setup_id, "timestamp_ns": timestamp_ns,
                                               "reason": "UNRESOLVED_SOURCE_END", "instrument": position.instrument,
                                               "contracts": position.contracts})
            self.diagnostic_events.append({"event": "UNRESOLVED_SOURCE_END", "setup_id": position.setup.setup_id,
                                           "timestamp_ns": timestamp_ns})
            self.signals.position = None

    def force_flat(self, timestamp_ns: int) -> None:
        """Frozen 22:45 UTC hard-flat handling using the current native quote."""
        if self.signals.position is None:
            return
        quote = self.es_quote if self.signals.position.instrument == "ES" else self.mes_quote
        if quote is None:
            raise HistoricalReplayError("cannot hard-flat active position without native executable quote")
        reference = quote[0] if self.signals.position.prices["direction"] == "LONG" else quote[1]
        self._close_position(timestamp_ns, reference, "HARD_CUTOFF_2245")

    def force_flat_from_last_causal_cutoff_quote(self, cutoff_ns: int, *, exit_reason: str = "HARD_CUTOFF_2245") -> None:
        """Apply the frozen L3 cutoff convention without inventing a quote."""
        position = self.signals.position
        if position is None:
            return
        quote_timestamp = self.es_quote_timestamp_ns if position.instrument == "ES" else self.mes_quote_timestamp_ns
        if quote_timestamp is None or not cutoff_ns - CUTOFF_QUOTE_LOOKBACK_NS <= quote_timestamp <= cutoff_ns:
            raise HistoricalReplayError("hard-cutoff execution requires a valid inclusive causal BBO observation")
        # Keep the frozen normal-session reason by default.  A separately
        # sealed execution-calendar clarification may name an earlier exchange
        # close without changing signal, pricing, or sizing semantics.
        if exit_reason == "HARD_CUTOFF_2245":
            self.force_flat(quote_timestamp)
        else:
            position = self.signals.position
            if position is None:
                return
            quote = self.es_quote if position.instrument == "ES" else self.mes_quote
            if quote is None:
                raise HistoricalReplayError("cannot hard-flat active position without native executable quote")
            reference = quote[0] if position.prices["direction"] == "LONG" else quote[1]
            self._close_position(quote_timestamp, reference, exit_reason)

    def refresh_setup_ledger(self) -> None:
        """Materialize terminal confirmation/entry fields after the causal pass."""
        trades = {str(row["setup_id"]): row for row in self.trade_ledger}
        for row in self.setup_ledger:
            setup_id = row.get("setup_id")
            if not setup_id:
                continue
            setup = self.signals.pending[str(setup_id)]
            row["confirmation_status"] = setup.state
            row["confirmation_timestamp_ns"] = setup.confirmation_timestamp_ns
            row["confirmation_price"] = setup.confirmation_price
            row["confirmation_seconds"] = ((setup.confirmation_timestamp_ns - int(row["interaction_end_ns"])) / 1_000_000_000
                                           if setup.confirmation_timestamp_ns is not None else None)
            row["terminal_reason"] = setup.terminal_reason
            trade = trades.get(str(setup_id))
            row["entry_displacement_points"] = (
                (float(trade["entry"]) - float(row["interaction_end_price"]))
                if trade and row["direction"] == "BUYER_ABSORPTION"
                else (float(row["interaction_end_price"]) - float(trade["entry"])) if trade else None
            )
            row["instrument"] = trade.get("instrument") if trade else None
            row["eventual_r"] = trade.get("r_multiple") if trade else None

    def summary(self) -> dict[str, Any]:
        accepted = sum(bool(row["accepted"]) for row in self.setup_ledger)
        rejected = len(self.setup_ledger) - accepted
        confirmed = sum(setup.state == "CONFIRMED" or setup.terminal_reason == "ENTRY" for setup in self.signals.pending.values())
        result = {"strategy_id": self.strategy_id, "date": self.date, "evidence_label": self.evidence_label,
                  "weights_label": self.config.weights_label, "interactions_completed": len(self.interaction_ledger),
                  "accepted_setups": accepted, "rejected_setups": rejected, "confirmed_setups": confirmed,
                  "trades": len(self.trade_ledger), "es_trades": sum(row["instrument"] == "ES" for row in self.trade_ledger),
                  "mes_trades": sum(row["instrument"] == "MES" for row in self.trade_ledger),
                  "incremental_completed_seen": self.completed_seen}
        if self.source_end_unresolved or self.mes_execution_unavailable or self.require_native_mes_for_fallback:
            result.update({"unresolved_source_end": len(self.source_end_unresolved),
                           "mes_execution_unavailable": len(self.mes_execution_unavailable)})
        return result


def _rows_write(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    names = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))}) or fallback
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader(); writer.writerows([{key: row.get(key) for key in names} for row in rows])


def write_future_artifacts(output_dir: Path, runners: list[HistoricalL2Runner], *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the immutable artifact shape used only by a future explicit replay."""
    if output_dir.exists():
        raise HistoricalReplayError("historical L2 output directory already exists")
    for runner in runners:
        runner.refresh_setup_ledger()
    contract = frozen_contract() if contract is None else contract
    summary = {"strategy_id": runners[0].strategy_id, "frozen_contract": contract, "period_results": [runner.summary() for runner in runners],
               "first_run_policy": "FIRST_BROAD_HISTORICAL_L2_V1_REPLAY", "outcome_parameter_selection": False}
    output_dir.mkdir(parents=True, exist_ok=False)
    _rows_write(output_dir / "daily-results.csv", [runner.summary() for runner in runners], ["date"])
    _rows_write(output_dir / "trade-ledger.csv", [row for runner in runners for row in runner.trade_ledger], ["trade_id"])
    _rows_write(output_dir / "setup-ledger.csv", [row for runner in runners for row in runner.setup_ledger], ["interaction_id"])
    _rows_write(output_dir / "interaction-features.csv", [row for runner in runners for row in runner.interaction_ledger], ["interaction_id"])
    source_integrity = {
        "policy": "RETAIN_PRIVATE_FAIL_CLOSED_IF_STRATEGY_VISIBLE",
        "max_tolerated_offbook_anomalies_per_session": MAX_TOLERATED_OFFBOOK_ANOMALIES_PER_SESSION,
        "sessions": [
            {"date": runner.date, "anomaly_count": len(runner.source_integrity_diagnostics),
             "anomalies": runner.source_integrity_diagnostics}
            for runner in runners
        ],
    }
    (output_dir / "source-integrity-diagnostics.json").write_text(
        json.dumps(source_integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text(
        "# Frozen L2 V1 historical replay\n\n"
        "`FIRST_BROAD_HISTORICAL_L2_V1_REPLAY`\n\n"
        "`NO_OUTCOME_BASED_PARAMETER_SELECTION_BEFORE_RUN`\n\n"
        "This report preserves period labels and never pools them as a single OOS result.\n",
        encoding="utf-8",
    )
    return summary


def inventory_existing_sessions(repository_root: Path) -> list[dict[str, Any]]:
    """Filesystem/manifest inventory only: it never opens a DBN or runs the model."""
    rows: list[dict[str, Any]] = []
    may = repository_root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
    manifest = may / "acquisition-manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for day in payload.get("request_identity", {}).get("target_rth_dates", []):
            es = may / "es_mbo" / f"ESM6_{day}_000000_224501_mbo.dbn.zst"
            mes = may / "mes_mbp1" / f"MESM6_{day}_133000_224501_mbp1.dbn.zst"
            rows.append({"date": day, "es_mbo_path": str(es), "mbo_start_end": "00:00:00-22:45:01 UTC",
                         "snapshot_initialization": "PRESENT_BUT_F_LAST_REPLAY_VALIDATION_REQUIRED",
                         "prior_rth_profile": "DECLARED_ES_PRIOR_RTH_TRADES", "mes_native_data": mes.is_file(),
                         "full_execution_replay": es.is_file() and mes.is_file(), "evidence_period_label": MAY_LABEL})
    retro = repository_root / "data/cme_orderflow_absorption_v2_holdout"
    for es in sorted((retro / "es_mbo").glob("ESU6_*_0000_1600_mbo.dbn.zst")) if (retro / "es_mbo").is_dir() else []:
        day = es.name.split("_")[1]
        mes = retro / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst"
        prior = retro / "es_rth_trades"
        rows.append({"date": day, "es_mbo_path": str(es), "mbo_start_end": "00:00:00-16:00:00 UTC",
                     "snapshot_initialization": "PRESENT_BUT_F_LAST_REPLAY_VALIDATION_REQUIRED",
                     "prior_rth_profile": any(prior.glob("*.dbn.zst")), "mes_native_data": mes.is_file(),
                     "full_execution_replay": False, "exclusion_reason": "ES_MBO_ENDS_16:00_UTC; cannot complete frozen session execution",
                     "evidence_period_label": RETRO_LABEL})
    august = repository_root / "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
    if august.is_file():
        rows.append({"date": "2026-08-03..2026-08-07", "es_mbo_path": str(august), "mbo_start_end": "2026-08-03..2026-08-08 UTC",
                     "snapshot_initialization": "PRESENT_BUT_F_LAST_REPLAY_VALIDATION_REQUIRED", "prior_rth_profile": True,
                     "mes_native_data": False, "full_execution_replay": False,
                     "exclusion_reason": "NO_NATIVE_MES_MBP1_INPUT", "evidence_period_label": AUGUST_LABEL})
    development = repository_root / "data/cme_orderflow_absorption_v1/ESU6/mbo/ESU6_2026-07-20_2026-08-01_mbo.dbn"
    if development.is_file():
        rows.append({"date": "2026-07-20..2026-07-31", "es_mbo_path": str(development), "mbo_start_end": "2026-07-20..2026-08-01 UTC",
                     "snapshot_initialization": "PRESENT_BUT_F_LAST_REPLAY_VALIDATION_REQUIRED", "prior_rth_profile": False,
                     "mes_native_data": False, "full_execution_replay": False,
                     "exclusion_reason": "DEVELOPMENT_SOURCE_HAS_NO_DECLARED_PRIOR_RTH_OR_NATIVE_MES_PACKAGE",
                     "evidence_period_label": "DEVELOPMENT_DESIGN_ONLY_NOT_HISTORICAL_L2_EVIDENCE"})
    return rows


def _code(value: object) -> str:
    text = str(getattr(value, "value", value))
    return text.rsplit(".", 1)[-1]


def normalize_source_mbo_price(raw_price: int, action: str) -> float:
    """Normalize a Databento fixed-point MBO price exactly once.

    Reset records have no price semantics and use a private typed placeholder.
    Every other supported book action must carry a finite fixed-point price.
    A finite off-book value is retained privately and can pass only when the
    aggregate reconstruction proves it never becomes strategy-visible.
    """
    if action == "R":
        return 5_000.0
    if raw_price == UNDEF_PRICE:
        raise HistoricalReplayError("MBO_UNDEFINED_PRICE_NON_RESET")
    price = raw_price / RAW_PRICE_SCALE
    if price <= 0.0 or price >= 1_000_000_000.0:
        raise HistoricalReplayError(
            f"MBO_INVALID_NORMALIZED_PRICE raw_price={raw_price} divisor={RAW_PRICE_SCALE} normalized_price={price} action={action}"
        )
    return price


def _stream_private_mbo(path: Path) -> Iterator[PrivateMBORecord]:
    """Local-only DBN reader, imported lazily so synthetic tests need no client."""
    from databento import DBNStore
    for record in DBNStore.from_file(path):
        action, side = _code(getattr(record, "action", "")), _code(getattr(record, "side", ""))
        raw_price = int(getattr(record, "price", 0))
        if action not in {"A", "C", "M", "F", "R", "T"} or side not in {"A", "B", "N"}:
            continue
        if action == "R":
            side = "B"  # Reset has no public side; its private adapter ignores it.
        if side not in {"A", "B"}:
            continue
        try:
            price = normalize_source_mbo_price(raw_price, action)
        except HistoricalReplayError as exc:
            raise HistoricalReplayError(
                f"{exc} ts_event={int(record.ts_event)} ts_recv={int(getattr(record, 'ts_recv', record.ts_event))} "
                f"side={side} size={int(record.size)} flags={int(getattr(record, 'flags', 0))}"
            ) from exc
        yield PrivateMBORecord(int(getattr(record, "ts_recv", record.ts_event)), action, side,
                               price, int(record.size), int(record.order_id), int(getattr(record, "flags", 0)), raw_price)


def _stream_mes_quotes(path: Path) -> Iterator[tuple[int, float, float]]:
    """Yield native MES MBP-1 top-of-book quotes from a declared local file."""
    from databento import DBNStore
    for record in DBNStore.from_file(path):
        levels = getattr(record, "levels", ())
        if not levels:
            continue
        level = levels[0]
        bid, ask = int(level.bid_px), int(level.ask_px)
        if bid > 0 and ask > bid and int(level.bid_sz) > 0 and int(level.ask_sz) > 0:
            yield int(getattr(record, "ts_event", record.ts_recv)), bid / RAW_PRICE_SCALE, ask / RAW_PRICE_SCALE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _may_paths(day: str) -> dict[str, str]:
    prior = MAY_PRIOR_RTH[day]
    return {
        "ES_MBO_L3": f"es_mbo/ESM6_{day}_000000_224501_mbo.dbn.zst",
        "MES_NATIVE_EXECUTION": f"mes_mbp1/MESM6_{day}_133000_224501_mbp1.dbn.zst",
        "ES_PRIOR_RTH_PROFILE": f"es_prior_rth_trades/ESM6_{prior}_133000_200000_trades.dbn.zst",
    }


def verify_may_acquisition_manifest(data_root: Path) -> dict[str, Any]:
    """Hash-verify only the 45 sealed May inputs before any replay work begins."""
    manifest_path = data_root / "acquisition-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalReplayError("missing or unreadable sealed May acquisition manifest") from exc
    if manifest.get("manifest_kind") != "MAY_2026_ES_MES_COST_PROXY_ACQUISITION" or manifest.get("data_acquired") is not True:
        raise HistoricalReplayError("May acquisition manifest is not an acquired sealed package")
    if manifest.get("request_identity", {}).get("target_rth_dates") != list(MAY_DATES):
        raise HistoricalReplayError("May manifest target chronology is not the sealed 15-session set")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 45:
        raise HistoricalReplayError("May acquisition manifest must declare exactly 45 inputs")
    verified: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for day in MAY_DATES:
        session_inputs: dict[str, dict[str, Any]] = {}
        for label, relative in _may_paths(day).items():
            record = files.get(relative)
            path = data_root / relative
            expected_day = day if label != "ES_PRIOR_RTH_PROFILE" else MAY_PRIOR_RTH[day]
            expected_schema = {"ES_MBO_L3": "mbo", "MES_NATIVE_EXECUTION": "mbp-1", "ES_PRIOR_RTH_PROFILE": "trades"}[label]
            if not isinstance(record, dict) or not path.is_file():
                raise HistoricalReplayError(f"required sealed May input missing: {relative}")
            if record.get("label") != label or record.get("schema") != expected_schema or record.get("session_date") != expected_day:
                raise HistoricalReplayError(f"sealed May input identity mismatch: {relative}")
            if record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
                raise HistoricalReplayError(f"sealed May input hash/size mismatch: {relative}")
            if label in {"ES_MBO_L3", "MES_NATIVE_EXECUTION"}:
                session_inputs[label] = record
            verified.append({"relative_path": relative, "label": label, "session_date": expected_day,
                             "bytes": path.stat().st_size, "sha256": record["sha256"]})
        coverage[day] = validate_declared_session_coverage(day, session_inputs)
    if set(files) != {relative for day in MAY_DATES for relative in _may_paths(day).values()}:
        raise HistoricalReplayError("May acquisition manifest has undeclared, missing, or extra inputs")
    return {"manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "files_verified": len(verified), "by_label": dict(Counter(row["label"] for row in verified)),
            "coverage": coverage, "files": verified}


def _profile_levels_from_declared_trades(path: Path) -> list[StructuralLevel]:
    """Build the frozen prior-RTH 70% profile without touching signal outcomes."""
    from databento import DBNStore
    by_price: Counter[int] = Counter()
    for record in DBNStore.from_file(path):
        price, size = int(getattr(record, "price", 0)), int(getattr(record, "size", 0))
        if price > 0 and size > 0:
            by_price[price] += size
    if not by_price:
        raise HistoricalReplayError("declared prior-RTH trades cannot build a causal structural profile")
    tick = int(TICK * RAW_PRICE_SCALE)
    poc = min(by_price, key=lambda price: (-by_price[price], price))
    total, included, low, high = sum(by_price.values()), by_price[poc], poc, poc
    while included * 100 < total * 70:
        below, above = low - tick, high + tick
        if by_price.get(below, 0) >= by_price.get(above, 0):
            low, included = below, included + by_price.get(below, 0)
        else:
            high, included = above, included + by_price.get(above, 0)
    return [StructuralLevel(name, value / RAW_PRICE_SCALE) for name, value in (
        ("PRIOR_RTH_HIGH", max(by_price)), ("PRIOR_RTH_LOW", min(by_price)), ("PRIOR_RTH_POC", poc),
        ("PRIOR_RTH_VAH", high), ("PRIOR_RTH_VAL", low),
    )]


def _clock_ns(day: str, seconds: int) -> int:
    midnight = datetime.fromisoformat(f"{day}T00:00:00+00:00")
    return int(midnight.timestamp() * 1_000_000_000) + seconds * 1_000_000_000


def _next(iterator: Iterator[Any]) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _run_may_session(day: str, data_root: Path, *, config: L2Config = L2Config(), strategy_id: str = STRATEGY_ID,
                     evidence_label: str = MAY_LABEL) -> HistoricalL2Runner:
    paths = _may_paths(day)
    levels = _profile_levels_from_declared_trades(data_root / paths["ES_PRIOR_RTH_PROFILE"])
    runner = HistoricalL2Runner(date=day, evidence_label=evidence_label, levels=levels, config=config, strategy_id=strategy_id)
    adapter = HistoricalMBOToMBP10Adapter()
    es_iter = iter(_stream_private_mbo(data_root / paths["ES_MBO_L3"]))
    mes_iter = iter(_stream_mes_quotes(data_root / paths["MES_NATIVE_EXECUTION"]))
    es, mes = _next(es_iter), _next(mes_iter)
    start_ns, cutoff_ns = _clock_ns(day, RTH_START_SECONDS), _clock_ns(day, HARD_CUTOFF_SECONDS)
    records = 0
    while es is not None or mes is not None:
        es_ts = es.timestamp_ns if es is not None else 2**63 - 1
        mes_ts = mes[0] if mes is not None else 2**63 - 1
        timestamp = min(es_ts, mes_ts)
        if timestamp >= cutoff_ns:
            runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
            runner.finish(cutoff_ns)
            break
        if mes_ts < es_ts:
            if mes_ts >= start_ns:
                runner.observe_mes_quote(*mes)
            mes = _next(mes_iter)
            continue
        record, es = es, _next(es_iter)
        records += 1
        try:
            public = adapter.feed(record, materialize_public=record.timestamp_ns >= start_ns)
        except L2ValidationError as exc:
            raise HistoricalReplayError(
                f"invalid normalized MBO record day={day} timestamp_ns={record.timestamp_ns} "
                f"action={record.action} side={record.side} price={record.price} size={record.size}"
            ) from exc
        if public is not None and public.timestamp_ns >= start_ns:
            runner.observe_public(public)
        if records % 5_000_000 == 0:
            print(f"  {day} records={records:,} completed={len(runner.interaction_ledger):,} accepted={sum(bool(row['accepted']) for row in runner.setup_ledger):,}", flush=True)
    else:
        # The sealed manifest/header validates coverage through the cutoff.
        # A quiet market can legitimately produce no source event at 22:45.
        runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
        runner.finish(cutoff_ns)
    adapter.finish()
    runner.source_integrity_diagnostics = adapter.source_integrity_diagnostics()
    return runner


def _session_levels(raw: dict[str, Any]) -> list[StructuralLevel]:
    levels = raw.get("structural_levels")
    if not isinstance(levels, dict) or not levels:
        raise HistoricalReplayError("session manifest requires explicit prior-RTH/current-RTH structural levels")
    return [StructuralLevel(str(name), float(price)) for name, price in sorted(levels.items())]


def run_manifest(session_manifest: Path, output_dir: Path) -> dict[str, Any]:
    """Execute a later explicit real replay from fully declared local inputs."""
    raw = json.loads(session_manifest.read_text(encoding="utf-8"))
    if raw.get("strategy_id") != STRATEGY_ID or raw.get("weights_label") != WEIGHTS_LABEL:
        raise HistoricalReplayError("session manifest does not bind the frozen L2 V1 contract")
    runners: list[HistoricalL2Runner] = []
    for session in raw.get("sessions", []):
        path = Path(session["es_mbo_path"])
        if not path.is_file():
            raise HistoricalReplayError("declared ES MBO input is missing")
        adapter = HistoricalMBOToMBP10Adapter()
        runner = HistoricalL2Runner(date=str(session["date"]), evidence_label=str(session["evidence_period_label"]), levels=_session_levels(session))
        for record in _stream_private_mbo(path):
            public = adapter.feed(record)
            if public is not None:
                runner.observe_public(public)
        adapter.finish()
        runner.source_integrity_diagnostics = adapter.source_integrity_diagnostics()
        runner.finish(int(session["session_end_ns"]))
        runners.append(runner)
    if not runners:
        raise HistoricalReplayError("session manifest contains no declared replay sessions")
    return write_future_artifacts(output_dir, runners)


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": sum(values) / len(values), "median": float(median(values)),
            "min": min(values), "max": max(values)}


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {name: {"trades": len(items), "total_r": sum(float(item["r_multiple"] or 0) for item in items),
                   "net_pnl_usd": sum(float(item["net_pnl_usd"]) for item in items)} for name, items in sorted(groups.items())}


def _performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    values = [float(row["r_multiple"] or 0.0) for row in ordered]
    net = [float(row["net_pnl_usd"]) for row in ordered]
    wins, losses = sum(value > 0 for value in net), sum(value < 0 for value in net)
    profit, loss = sum(value for value in net if value > 0), abs(sum(value for value in net if value < 0))
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value; peak = max(peak, equity); drawdown = min(drawdown, equity - peak)
    return {"completed_trades": len(ordered), "wins": wins, "losses": losses,
            "win_rate": wins / len(ordered) if ordered else 0.0, "total_r": sum(values),
            "average_r": sum(values) / len(values) if values else 0.0,
            "median_r": float(median(values)) if values else None, "net_pnl_usd": sum(net),
            "profit_factor": profit / loss if loss else None, "max_cumulative_drawdown_r": drawdown,
            "es_trades": sum(row["instrument"] == "ES" for row in ordered),
            "mes_trades": sum(row["instrument"] == "MES" for row in ordered),
            "target_exits": sum(row["exit_reason"] == "TARGET" for row in ordered),
            "stop_exits": sum(row["exit_reason"] == "STOP" for row in ordered),
            "hard_cutoff_exits": sum(row["exit_reason"] == "HARD_CUTOFF_2245" for row in ordered)}


def _may_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    return "\n".join([
        f"# {summary['strategy_id']} replay — May 2026", "",
        f"Run label: `{summary['run_label']}`", f"Evidence label: `{summary['evidence_label']}`", "",
        "This is not strict chronological OOS because later-July research exists elsewhere.",
        "No outcome-based parameter selection, rerun, filter, or optimization was performed.", "",
        f"Completed interactions: {metrics['completed_interactions']}",
        f"Accepted/rejected L2 setups: {metrics['accepted_l2_setups']}/{metrics['rejected_l2_setups']}",
        f"Completed trades: {metrics['completed_trades']}; total R: {metrics['total_r']:.6f}; net PnL: {metrics['net_pnl_usd']:.2f}",
        f"Unresolved trades: {metrics['unresolved_trades']}", "",
        "All per-day, direction, structural-level, instrument, and accepted-vs-rejected feature summaries are in `summary.json`.",
    ]) + "\n"


def run_first_broad_may_2026(
    *, data_root: Path, output_dir: Path, config: L2Config = L2Config(), strategy_id: str = STRATEGY_ID,
    evidence_label: str = MAY_LABEL, run_label: str = MAY_REPLAY_LABEL, contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one explicit immutable May replay using the supplied frozen configuration."""
    if output_dir.exists():
        raise HistoricalReplayError("first broad L2 May output directory already exists")
    started = time.monotonic()
    verification = verify_may_acquisition_manifest(data_root)
    runners: list[HistoricalL2Runner] = []
    for index, day in enumerate(MAY_DATES, start=1):
        print(f"=== L2 MAY {index:02d}/{len(MAY_DATES)} {day} ===", flush=True)
        runners.append(_run_may_session(day, data_root, config=config, strategy_id=strategy_id, evidence_label=evidence_label))
    contract = contract or contract_for_config(
        strategy_id=strategy_id, config=config,
        first_run_policy="EXPLICIT_IMMUTABLE_L2_MAY_REPLAY; NO_OUTCOME_BASED_PARAMETER_SELECTION_BEFORE_RUN",
    )
    base = write_future_artifacts(output_dir, runners, contract=contract)
    interactions = [row for runner in runners for row in runner.interaction_ledger]
    setups = [row for runner in runners for row in runner.setup_ledger]
    trades = [row for runner in runners for row in runner.trade_ledger]
    accepted, rejected = [row for row in setups if row["accepted"]], [row for row in setups if not row["accepted"]]
    pending = [setup for runner in runners for setup in runner.signals.pending.values()]
    score_fields = ("aggression_score", "restoration_score", "price_resistance_score", "persistence_score",
                    "multi_level_support_score", "false_refill_penalty", "l2_absorption_quality_score")
    feature_fields = ("directional_aggressive_volume", "relevant_execution_count", "cumulative_consumed_volume",
                      "cumulative_restored_volume", "depth_restoration_count", "consume_restore_cycles",
                      "mean_restoration_latency_ms", "executed_to_initial_displayed_ratio",
                      "maximum_through_level_progress_ticks", "interaction_rejection_ticks",
                      "defended_price_present_fraction", "order_count_restoration_cycles", "depth_imbalance_1",
                      "depth_imbalance_3", "depth_imbalance_5", "multi_level_ofi", "rapid_cancel_ratio")
    performance = _performance(trades)
    metrics = {"raw_interactions": len(interactions), "completed_interactions": len(interactions),
               "accepted_l2_setups": len(accepted), "rejected_l2_setups": len(rejected),
               "acceptance_rate": len(accepted) / len(interactions) if interactions else 0.0,
               "confirmations_passed": sum(setup.confirmation_timestamp_ns is not None for setup in pending),
               "confirmations_failed": sum(setup.terminal_reason == "CONFIRMATION_WINDOW_EXPIRED" for setup in pending),
               "blocked_setups": sum(setup.terminal_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for setup in pending),
               "unresolved_trades": sum(setup.state in {"WAIT_MIN_CONFIRMATION_TIME", "CONFIRMED"} and setup.terminal_reason is None for setup in pending),
               **performance}
    payload = {**base, "run_label": run_label, "evidence_label": evidence_label,
               "strict_chronological_oos": False, "manifest_verification": verification,
               "snapshot_contract": {"required": "R/A F_SNAPSHOT through F_LAST before ordinary MBO", "verified_per_session": True},
               "order_identity_boundary": "PRIVATE_MBO_ADAPTER_ONLY", "metrics": metrics,
               "breakdowns": {"day": _breakdown(trades, "date"), "direction": _breakdown(trades, "direction"),
                              "structural_level": _breakdown(trades, "level"), "instrument": _breakdown(trades, "instrument")},
               "accepted_vs_rejected_diagnostics": {
                   "accepted": {field: _distribution(accepted, field) for field in score_fields + feature_fields},
                   "rejected": {field: _distribution(rejected, field) for field in score_fields + feature_fields},
               }, "unresolved_setup_ids": [setup.setup_id for setup in pending if setup.state in {"WAIT_MIN_CONFIRMATION_TIME", "CONFIRMED"} and setup.terminal_reason is None],
               "first_run_policy": contract["first_run_policy"]}
    payload["frozen_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["runtime_seconds"] = time.monotonic() - started
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text(_may_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen L2 V1 on explicitly declared local MBO sessions")
    parser.add_argument("--session-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--may-2026", action="store_true", help="Run the one sealed 15-session May 2026 replay")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.inventory_only:
            print(json.dumps({"sessions": inventory_existing_sessions(args.repository_root)}, indent=2, sort_keys=True))
        elif args.may_2026:
            if args.session_manifest is not None:
                raise HistoricalReplayError("--may-2026 cannot be combined with --session-manifest")
            data_root = args.data_root or (args.repository_root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy")
            output_dir = args.output_dir or (args.repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V1_MAY_2026")
            print(json.dumps(run_first_broad_may_2026(data_root=data_root, output_dir=output_dir), indent=2, sort_keys=True))
        else:
            if args.session_manifest is None:
                raise HistoricalReplayError("--session-manifest is required unless --inventory-only is selected")
            if args.output_dir is None:
                raise HistoricalReplayError("--output-dir is required unless --inventory-only is selected")
            print(json.dumps(run_manifest(args.session_manifest, args.output_dir), indent=2, sort_keys=True))
    except (HistoricalReplayError, L2ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
