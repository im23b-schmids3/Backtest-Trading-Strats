"""Versioned L2 execution contract with a Europe/Berlin daily hard flat.

This module intentionally does not modify historical V3.  It reuses the
frozen signal, confirmation, price geometry, sizing, and cost functions while
giving entered positions a new, fully terminal execution lifecycle.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from . import causal_master_tape as master
from . import v3_poc_dec2025_jan2026_calendar_audit as exchange_calendar
from . import weight_q_research as historical_matrix
from .model import (
    ES_COMMISSION,
    ES_POINT_VALUE,
    MES_COMMISSION,
    MES_POINT_VALUE,
    TICK,
    initial_prices,
    size_for_instrument,
)


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT"
SEMANTIC_VERSION = "BERLIN_HARDFLAT_DATA_GAP_3S_SOURCE_END_V1"
HISTORICAL_V3_CONTRACT_SHA256 = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
BERLIN = ZoneInfo("Europe/Berlin")
UTC = timezone.utc
HARD_FLAT_LOCAL_TIME = time(22, 45)
MAX_EXECUTABLE_BBO_GAP_NS = 3_000_000_000
TERMINAL_EXIT_REASONS = (
    "TARGET",
    "STOP",
    "HARD_FLAT_BERLIN",
    "DATA_GAP_3S_FORCE_FLAT",
    "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
)

EXECUTION_CONTRACT: dict[str, Any] = {
    "strategy_id": STRATEGY_ID,
    "semantic_version": SEMANTIC_VERSION,
    "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
    "signal_contract": {
        "eligible_levels": ["PRIOR_RTH_POC"],
        "quality_components": list(master.SCORE_FIELDS),
        "confirmation_favorable_ticks": 3,
        "confirmation_horizon_seconds": 15,
        "entry_latency_ms": 2,
    },
    "execution_contract": {
        "hard_flat_timezone": "Europe/Berlin",
        "hard_flat_local_time": "22:45:00",
        "hard_flat_actions": [
            "CANCEL_ALL_WORKING_ORDERS", "LIQUIDATE_OPEN_POSITION",
            "BLOCK_NEW_ENTRIES_FOR_SESSION",
        ],
        "long_liquidation_reference": "FRESH_NATIVE_EXECUTABLE_BID",
        "short_liquidation_reference": "FRESH_NATIVE_EXECUTABLE_ASK",
        "maintenance_invariant": "OPEN_POSITION_AT_MAINTENANCE_IS_FATAL",
        "max_executable_bbo_gap_seconds": 3.0,
        "gap_at_exact_threshold": "RESUME_IF_FRESH_EXECUTABLE_BBO_RETURNS_WITHIN_OR_AT_3_SECONDS",
        "gap_force_exit": "LAST_VALID_EXECUTABLE_NATIVE_BBO_WITH_NORMAL_ADVERSE_EXIT_SLIPPAGE",
        "source_end_exit": "LAST_VALID_EXECUTABLE_NATIVE_BBO_WITH_NORMAL_ADVERSE_EXIT_SLIPPAGE",
        "stop": "COMPLETED_INTERACTION_ZONE_PLUS_5_TICKS",
        "target_r": 3.0,
        "fixed_risk_usd": 250.0,
        "instrument_preference": "ES_FIRST_MES_FALLBACK",
        "apex_caps": {"ES": 6, "MES": 60},
        "terminal_exit_reasons": list(TERMINAL_EXIT_REASONS),
        "unresolved_trade_policy": "FORBIDDEN",
    },
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


CONTRACT_SHA256 = _canonical_sha256(EXECUTION_CONTRACT)


class BerlinExecutionError(RuntimeError):
    pass


class UnpricedSourceIntegrityFailure(BerlinExecutionError):
    pass


class OpenPositionAtMaintenance(BerlinExecutionError):
    pass


def berlin_hard_flat_datetime(day: str | date) -> datetime:
    session_date = date.fromisoformat(day) if isinstance(day, str) else day
    return datetime.combine(session_date, HARD_FLAT_LOCAL_TIME, tzinfo=BERLIN)


def berlin_hard_flat_utc(day: str | date) -> datetime:
    return berlin_hard_flat_datetime(day).astimezone(UTC)


def berlin_hard_flat_ns(day: str | date) -> int:
    return int(berlin_hard_flat_utc(day).timestamp() * 1_000_000_000)


def maintenance_window_utc(day: str | date) -> tuple[datetime, datetime]:
    session_date = date.fromisoformat(day) if isinstance(day, str) else day
    return exchange_calendar.maintenance_window(session_date)


def maintenance_window_ns(day: str | date) -> tuple[int, int]:
    start, end = maintenance_window_utc(day)
    return int(start.timestamp() * 1e9), int(end.timestamp() * 1e9)


@dataclass(frozen=True)
class QuoteObservation:
    event_ordinal: int
    observation_timestamp_ns: int
    source_timestamp_ns: int
    bid: float
    ask: float


@dataclass(frozen=True)
class ExitObservation:
    event_ordinal: int
    exit_timestamp_ns: int
    reference_price: float
    reason: str
    price_source_timestamp_ns: int
    decision_timestamp_ns: int
    gap_timeout_timestamp_ns: int | None = None
    gap_start_timestamp_ns: int | None = None


def _valid_quote(bid: float, ask: float) -> bool:
    return math.isfinite(bid) and math.isfinite(ask) and bid < ask


class BerlinSessionCausalTape(historical_matrix.SessionCausalTape):
    """Historical compact tape interpreted under the new execution contract."""

    def __init__(self, day: str, rows: Iterable[Mapping[str, Any]]) -> None:
        super().__init__(day, rows)
        self.hard_flat_local = berlin_hard_flat_datetime(day)
        self.hard_flat_utc = self.hard_flat_local.astimezone(UTC)
        self.hard_flat_timestamp_ns = int(self.hard_flat_utc.timestamp() * 1e9)
        maintenance_start, maintenance_end = maintenance_window_ns(day)
        self.maintenance_start_ns = maintenance_start
        self.maintenance_end_ns = maintenance_end
        self.original_terminal_event = dict(self.hard_event or self.source_end_event or {})
        if not self.original_terminal_event:
            raise BerlinExecutionError(f"source terminal missing: {day}")
        self._quotes = {
            "ES": self._build_quote_observations("ES"),
            "MES": self._build_quote_observations("MES"),
        }
        self._quote_observation_timestamps = {
            instrument: tuple(row.observation_timestamp_ns for row in rows)
            for instrument, rows in self._quotes.items()
        }
        self._quote_event_ordinals = {
            instrument: tuple(row.event_ordinal for row in rows)
            for instrument, rows in self._quotes.items()
        }

        original_timestamp = int(self.original_terminal_event["timestamp_ns"])
        original_ordinal = int(self.original_terminal_event["event_ordinal"])
        if original_timestamp < self.hard_flat_timestamp_ns:
            # Early close or incomplete source: the new source-end rule owns
            # the terminal transition and no Berlin hard-flat event is forged.
            self.hard_event = None
            self.source_end_event = {
                **self.original_terminal_event,
                "event_type": "SOURCE_END",
                "hard_flat_reason": "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
            }
            self.contract_terminal_kind = "SOURCE_END"
            self.contract_terminal_timestamp_ns = original_timestamp
            self.contract_terminal_ordinal = original_ordinal
        else:
            regular_index = bisect.bisect_left(self.timestamps, self.hard_flat_timestamp_ns)
            terminal_ordinal = (
                int(self.ordinals[regular_index]) if regular_index < len(self.ordinals) else original_ordinal
            )
            self.source_end_event = None
            self.hard_event = {
                "session_date": day,
                "event_ordinal": terminal_ordinal,
                "timestamp_ns": self.hard_flat_timestamp_ns,
                "event_type": "HARD_FLAT",
                "hard_flat_reason": "HARD_FLAT_BERLIN",
                "hard_flat_local": self.hard_flat_local.isoformat(),
                "hard_flat_utc": self.hard_flat_utc.isoformat(),
            }
            self.contract_terminal_kind = "HARD_FLAT_BERLIN"
            self.contract_terminal_timestamp_ns = self.hard_flat_timestamp_ns
            self.contract_terminal_ordinal = terminal_ordinal

    @classmethod
    def from_parquet(cls, day: str, path: Path) -> "BerlinSessionCausalTape":
        return cls(day, master._iter_parquet_rows(path))

    def _build_quote_observations(self, instrument: str) -> tuple[QuoteObservation, ...]:
        stream_value = 1 if instrument == "ES" else 2
        bids = self.es_bid if instrument == "ES" else self.mes_bid
        asks = self.es_ask if instrument == "ES" else self.mes_ask
        sources = self.es_quote_timestamps if instrument == "ES" else self.mes_quote_timestamps
        output: list[QuoteObservation] = []
        for index, stream in enumerate(self.streams):
            bid, ask = float(bids[index]), float(asks[index])
            source_timestamp = int(sources[index])
            if stream != stream_value or source_timestamp < 0 or not _valid_quote(bid, ask):
                continue
            observation = QuoteObservation(
                int(self.ordinals[index]), int(self.timestamps[index]), source_timestamp, bid, ask,
            )
            if output and observation == output[-1]:
                continue
            output.append(observation)
        return tuple(output)

    def _entry_quote(self, instrument: str, regular_index: int) -> QuoteObservation | None:
        bids = self.es_bid if instrument == "ES" else self.mes_bid
        asks = self.es_ask if instrument == "ES" else self.mes_ask
        sources = self.es_quote_timestamps if instrument == "ES" else self.mes_quote_timestamps
        bid, ask = float(bids[regular_index]), float(asks[regular_index])
        if not _valid_quote(bid, ask):
            return None
        source = int(sources[regular_index])
        return QuoteObservation(
            int(self.ordinals[regular_index]), int(self.timestamps[regular_index]),
            source if source >= 0 else int(self.timestamps[regular_index]), bid, ask,
        )

    def _last_quote(
        self, instrument: str, timestamp_ns: int, *, entry_seed: QuoteObservation | None = None,
    ) -> QuoteObservation | None:
        rows = self._quotes[instrument]
        timestamps = self._quote_observation_timestamps[instrument]
        index = bisect.bisect_right(timestamps, timestamp_ns) - 1
        candidate = rows[index] if index >= 0 else None
        if entry_seed is not None and (
            candidate is None or entry_seed.observation_timestamp_ns > candidate.observation_timestamp_ns
        ):
            candidate = entry_seed
        return candidate

    def _first_quote(
        self, instrument: str, timestamp_ns: int, *, end_ns: int | None = None,
    ) -> QuoteObservation | None:
        rows = self._quotes[instrument]
        timestamps = self._quote_observation_timestamps[instrument]
        index = bisect.bisect_left(timestamps, timestamp_ns)
        if index >= len(rows):
            return None
        candidate = rows[index]
        if end_ns is not None and candidate.observation_timestamp_ns > end_ns:
            return None
        return candidate

    def _last_quote_before_boundary(
        self, instrument: str, boundary: historical_matrix.TapeBoundary,
        *, entry_seed: QuoteObservation,
    ) -> QuoteObservation | None:
        rows = self._quotes[instrument]
        index = bisect.bisect_left(
            self._quote_event_ordinals[instrument], boundary.start_ordinal,
        ) - 1
        candidate = rows[index] if index >= 0 else None
        if entry_seed.event_ordinal < boundary.start_ordinal and (
            candidate is None or entry_seed.event_ordinal > candidate.event_ordinal
        ):
            candidate = entry_seed
        return candidate

    def _first_quote_after_boundary(
        self, instrument: str, boundary: historical_matrix.TapeBoundary,
    ) -> QuoteObservation | None:
        rows = self._quotes[instrument]
        index = bisect.bisect_right(
            self._quote_event_ordinals[instrument], boundary.start_ordinal,
        )
        return rows[index] if index < len(rows) else None

    def _stop_target_candidate(
        self, *, instrument: str, direction: str, entry_ordinal: int,
        stop: float, target: float, terminal_instruction_ns: int,
    ) -> ExitObservation | None:
        start = bisect.bisect_right(self.ordinals, entry_ordinal)
        end = bisect.bisect_left(self.timestamps, terminal_instruction_ns)
        if direction == "LONG":
            series = self._series[(instrument, "bid")]
            stop_index = series.first_le(start, end, stop)
            target_index = series.first_ge(start, end, target)
        else:
            series = self._series[(instrument, "ask")]
            stop_index = series.first_ge(start, end, stop)
            target_index = series.first_le(start, end, target)
        candidates = [
            (index, reason) for index, reason in ((stop_index, "STOP"), (target_index, "TARGET"))
            if index is not None
        ]
        if not candidates:
            return None
        index, reason = min(
            candidates,
            key=lambda item: (int(self.timestamps[item[0]]), 0 if item[1] == "STOP" else 1),
        )
        reference = (
            self.es_bid[index] if instrument == "ES" and direction == "LONG"
            else self.es_ask[index] if instrument == "ES"
            else self.mes_bid[index] if direction == "LONG"
            else self.mes_ask[index]
        )
        timestamp = int(self.timestamps[index])
        source_timestamp = int(
            self.es_quote_timestamps[index] if instrument == "ES" else self.mes_quote_timestamps[index]
        )
        return ExitObservation(
            int(self.ordinals[index]), timestamp, float(reference), reason,
            source_timestamp if source_timestamp >= 0 else timestamp, timestamp,
        )

    def _decision_event_ordinal(self, decision_timestamp_ns: int) -> int:
        """Map a synthetic timeout to the first causal event at/after it.

        The trade price/time remains the explicit last-BBO timeout assumption;
        this ordinal exists only to keep pending/active-position chronology in
        the generic portfolio state machine correctly ordered.
        """
        index = bisect.bisect_left(self.timestamps, decision_timestamp_ns)
        if index < len(self.ordinals):
            return int(self.ordinals[index])
        return self.contract_terminal_ordinal

    def _gap_candidate(
        self, *, instrument: str, direction: str, entry_ordinal: int,
        entry_timestamp_ns: int, terminal_instruction_ns: int,
        entry_seed: QuoteObservation,
    ) -> ExitObservation | None:
        earliest: ExitObservation | None = None
        for boundary in self.non_executable_boundaries:
            if earliest is not None and boundary.start_timestamp_ns >= earliest.decision_timestamp_ns:
                break
            if boundary.start_ordinal <= entry_ordinal:
                continue
            if boundary.start_timestamp_ns >= terminal_instruction_ns:
                break
            if boundary.classification == "EXPECTED_SCHEDULED_MAINTENANCE":
                raise OpenPositionAtMaintenance(
                    f"OPEN_POSITION_AT_MAINTENANCE:{self.day}:{boundary.start_timestamp_ns}"
                )
            last = self._last_quote_before_boundary(
                instrument, boundary, entry_seed=entry_seed,
            )
            if last is None or last.observation_timestamp_ns < entry_timestamp_ns:
                raise UnpricedSourceIntegrityFailure(
                    f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:DATA_GAP"
                )
            timeout = last.observation_timestamp_ns + MAX_EXECUTABLE_BBO_GAP_NS
            decision = max(boundary.start_timestamp_ns, timeout)
            # TapeBoundary's historical reopen field can be satisfied by a
            # carried quote on the other instrument's stream.  The corrected
            # contract requires a fresh native observation, so consult the
            # native quote index instead.
            reopen_observation = self._first_quote_after_boundary(instrument, boundary)
            reopen = (
                reopen_observation.observation_timestamp_ns
                if reopen_observation is not None else None
            )
            if reopen is not None and reopen <= timeout:
                continue
            reference = last.bid if direction == "LONG" else last.ask
            candidate = ExitObservation(
                self._decision_event_ordinal(decision),
                decision,
                reference,
                "DATA_GAP_3S_FORCE_FLAT",
                last.source_timestamp_ns,
                decision,
                gap_timeout_timestamp_ns=timeout,
                gap_start_timestamp_ns=boundary.start_timestamp_ns,
            )
            if earliest is None or candidate.decision_timestamp_ns < earliest.decision_timestamp_ns:
                earliest = candidate
        return earliest

    def _terminal_candidate(
        self, *, instrument: str, direction: str, entry_seed: QuoteObservation,
    ) -> ExitObservation:
        if self.contract_terminal_kind == "SOURCE_END":
            last = self._last_quote(
                instrument, self.contract_terminal_timestamp_ns, entry_seed=entry_seed,
            )
            if last is None or last.observation_timestamp_ns < entry_seed.observation_timestamp_ns:
                raise UnpricedSourceIntegrityFailure(
                    f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:SOURCE_END"
                )
            return ExitObservation(
                self.contract_terminal_ordinal,
                self.contract_terminal_timestamp_ns,
                last.bid if direction == "LONG" else last.ask,
                "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
                last.source_timestamp_ns,
                self.contract_terminal_timestamp_ns,
            )

        for boundary in self.non_executable_boundaries:
            reopen_observation = self._first_quote_after_boundary(instrument, boundary)
            reopen = (
                reopen_observation.observation_timestamp_ns
                if reopen_observation is not None else None
            )
            if not (
                boundary.start_timestamp_ns <= self.hard_flat_timestamp_ns
                and (reopen is None or reopen > self.hard_flat_timestamp_ns)
            ):
                continue
            if boundary.classification == "EXPECTED_SCHEDULED_MAINTENANCE":
                raise OpenPositionAtMaintenance(
                    f"OPEN_POSITION_AT_MAINTENANCE:{self.day}:{boundary.start_timestamp_ns}"
                )
            last = self._last_quote_before_boundary(
                instrument, boundary, entry_seed=entry_seed,
            )
            if last is None:
                raise UnpricedSourceIntegrityFailure(
                    f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:HARD_FLAT_GAP"
                )
            timeout = last.observation_timestamp_ns + MAX_EXECUTABLE_BBO_GAP_NS
            decision = max(boundary.start_timestamp_ns, timeout)
            if reopen is None or reopen > timeout:
                return ExitObservation(
                    self._decision_event_ordinal(decision),
                    decision,
                    last.bid if direction == "LONG" else last.ask,
                    "DATA_GAP_3S_FORCE_FLAT",
                    last.source_timestamp_ns,
                    decision,
                    gap_timeout_timestamp_ns=timeout,
                    gap_start_timestamp_ns=boundary.start_timestamp_ns,
                )
            break

        first = self._first_quote(
            instrument,
            self.hard_flat_timestamp_ns,
            end_ns=int(self.original_terminal_event["timestamp_ns"]),
        )
        if first is None:
            # An executable state remains live until explicitly invalidated.
            # This fallback is permitted only when no non-executable interval
            # covers the instruction and the last quote is defensible.
            for boundary in self.non_executable_boundaries:
                reopen_observation = self._first_quote_after_boundary(instrument, boundary)
                reopen = (
                    reopen_observation.observation_timestamp_ns
                    if reopen_observation is not None else None
                )
                if boundary.start_timestamp_ns <= self.hard_flat_timestamp_ns and (
                    reopen is None or reopen > self.hard_flat_timestamp_ns
                ):
                    raise UnpricedSourceIntegrityFailure(
                        f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:HARD_FLAT"
                    )
            first = self._last_quote(instrument, self.hard_flat_timestamp_ns, entry_seed=entry_seed)
        if first is None:
            raise UnpricedSourceIntegrityFailure(
                f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:HARD_FLAT"
            )
        if first.observation_timestamp_ns >= self.maintenance_start_ns:
            raise OpenPositionAtMaintenance(
                f"OPEN_POSITION_AT_MAINTENANCE:{self.day}:{first.observation_timestamp_ns}"
            )
        return ExitObservation(
            first.event_ordinal,
            max(self.hard_flat_timestamp_ns, first.observation_timestamp_ns),
            first.bid if direction == "LONG" else first.ask,
            "HARD_FLAT_BERLIN",
            first.source_timestamp_ns,
            self.hard_flat_timestamp_ns,
        )

    def _trade_exit_berlin(
        self, *, instrument: str, direction: str, entry_ordinal: int,
        entry_timestamp_ns: int, stop: float, target: float,
        entry_seed: QuoteObservation,
    ) -> ExitObservation:
        gap = self._gap_candidate(
            instrument=instrument,
            direction=direction,
            entry_ordinal=entry_ordinal,
            entry_timestamp_ns=entry_timestamp_ns,
            terminal_instruction_ns=self.contract_terminal_timestamp_ns,
            entry_seed=entry_seed,
        )
        terminal = (
            None if gap is not None else self._terminal_candidate(
                instrument=instrument, direction=direction, entry_seed=entry_seed,
            )
        )
        forced_timestamp = (
            gap.decision_timestamp_ns if gap is not None
            else terminal.decision_timestamp_ns  # type: ignore[union-attr]
        )
        stop_target = self._stop_target_candidate(
            instrument=instrument,
            direction=direction,
            entry_ordinal=entry_ordinal,
            stop=stop,
            target=target,
            terminal_instruction_ns=forced_timestamp,
        )
        candidates = [terminal] if terminal is not None else []
        if gap is not None:
            candidates.append(gap)
        if stop_target is not None:
            candidates.append(stop_target)
        priority = {
            "DATA_GAP_3S_FORCE_FLAT": 0,
            "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO": 0,
            "HARD_FLAT_BERLIN": 0,
            "STOP": 1,
            "TARGET": 2,
        }
        return min(candidates, key=lambda item: (item.decision_timestamp_ns, priority[item.reason]))

    def entry_outcome(
        self, interaction: Mapping[str, Any], event_ordinal: int,
    ) -> historical_matrix.EntryOutcome:
        identifier = str(interaction["interaction_id"])
        cache_key = (identifier, event_ordinal)
        cached = self._outcome_cache.get(cache_key)
        if cached is not None:
            return cached
        index = self.regular_index(event_ordinal)
        entry_timestamp = int(self.timestamps[index])
        if entry_timestamp >= self.contract_terminal_timestamp_ns:
            outcome = historical_matrix.EntryOutcome("ENTRY_BLOCKED_AT_OR_AFTER_HARD_FLAT_BERLIN")
            self._outcome_cache[cache_key] = outcome
            return outcome
        if not _valid_quote(float(self.es_bid[index]), float(self.es_ask[index])):
            outcome = historical_matrix.EntryOutcome("WAIT_FOR_ES_EXECUTABLE_QUOTE")
            self._outcome_cache[cache_key] = outcome
            return outcome

        direction = str(interaction["direction"])
        prices = initial_prices(
            direction, self.es_bid[index], self.es_ask[index],
            float(interaction["zone_low"]), float(interaction["zone_high"]),
        )
        sizing = size_for_instrument(prices, "ES")
        instrument = "ES"
        if int(sizing["contracts"]) < 1:
            if not _valid_quote(float(self.mes_bid[index]), float(self.mes_ask[index])):
                outcome = historical_matrix.EntryOutcome("MES_EXECUTION_UNAVAILABLE")
                self._outcome_cache[cache_key] = outcome
                return outcome
            prices = initial_prices(
                direction, self.mes_bid[index], self.mes_ask[index],
                float(interaction["zone_low"]), float(interaction["zone_high"]),
            )
            sizing = size_for_instrument(prices, "MES")
            instrument = "MES"
        if int(sizing["contracts"]) < 1:
            outcome = historical_matrix.EntryOutcome("INSUFFICIENT_RISK_BUDGET_FOR_ONE_CONTRACT")
            self._outcome_cache[cache_key] = outcome
            return outcome

        entry_seed = self._entry_quote(instrument, index)
        if entry_seed is None:
            raise UnpricedSourceIntegrityFailure(
                f"UNPRICED_SOURCE_INTEGRITY_FAILURE:{self.day}:{instrument}:ENTRY"
            )
        exit_observation = self._trade_exit_berlin(
            instrument=instrument,
            direction=str(prices["direction"]),
            entry_ordinal=event_ordinal,
            entry_timestamp_ns=entry_timestamp,
            stop=float(prices["stop"]),
            target=float(prices["target"]),
            entry_seed=entry_seed,
        )
        long = prices["direction"] == "LONG"
        exit_price = exit_observation.reference_price - TICK if long else exit_observation.reference_price + TICK
        contracts = int(sizing["contracts"])
        point_value, commission = (
            (ES_POINT_VALUE, ES_COMMISSION) if instrument == "ES" else (MES_POINT_VALUE, MES_COMMISSION)
        )
        points = exit_price - float(prices["entry"]) if long else float(prices["entry"]) - exit_price
        gross = points * point_value * contracts
        fees = 2 * commission * contracts
        initial_risk = abs(float(prices["entry"]) - float(prices["stop_exit"])) * point_value * contracts + fees
        setup_id = f"L2:{interaction['source_interaction_id']}"
        trade = {
            "trade_id": f"L2T:{setup_id}",
            "setup_id": setup_id,
            "date": self.day,
            "interaction_id": str(interaction["source_interaction_id"]),
            "direction": str(prices["direction"]),
            "level": str(interaction["level"]),
            "instrument": instrument,
            "contracts": contracts,
            "entry_timestamp_ns": entry_timestamp,
            "entry": float(prices["entry"]),
            "stop": float(prices["stop"]),
            "target": float(prices["target"]),
            "exit_timestamp_ns": exit_observation.exit_timestamp_ns,
            "exit": exit_price,
            "exit_reason": exit_observation.reason,
            "liquidation_reference_price": exit_observation.reference_price,
            "price_source_timestamp_ns": exit_observation.price_source_timestamp_ns,
            "liquidation_decision_timestamp_ns": exit_observation.decision_timestamp_ns,
            "source_end_timestamp_ns": (
                self.contract_terminal_timestamp_ns
                if exit_observation.reason == "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO" else None
            ),
            "hard_flat_instruction_timestamp_ns": (
                self.hard_flat_timestamp_ns
                if exit_observation.reason == "HARD_FLAT_BERLIN" else None
            ),
            "gap_start_timestamp_ns": exit_observation.gap_start_timestamp_ns,
            "gap_timeout_timestamp_ns": exit_observation.gap_timeout_timestamp_ns,
            "last_valid_bbo_age_ns": (
                exit_observation.decision_timestamp_ns - exit_observation.price_source_timestamp_ns
            ),
            "working_orders_cancelled": exit_observation.reason in {
                "HARD_FLAT_BERLIN", "DATA_GAP_3S_FORCE_FLAT",
                "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
            },
            "new_entries_blocked_for_session": exit_observation.reason in {
                "HARD_FLAT_BERLIN", "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
            },
            "hard_flat_local": self.hard_flat_local.isoformat(),
            "hard_flat_utc": self.hard_flat_utc.isoformat(),
            "gross_pnl_usd": gross,
            "total_costs_usd": fees,
            "net_pnl_usd": gross - fees,
            "r_multiple": (gross - fees) / initial_risk if initial_risk else None,
        }
        outcome = historical_matrix.EntryOutcome(
            "ENTRY", trade, exit_observation.event_ordinal,
        )
        self._outcome_cache[cache_key] = outcome
        return outcome


NONTRADE_UNRESOLVED_CLASSIFICATION = {
    "CONFIRMATION_UNRESOLVED_SOURCE_INCOMPLETE": "SOURCE_END_CANCELLED_BEFORE_CONFIRMATION",
    "EXECUTION_UNRESOLVED_SOURCE_INCOMPLETE": "SOURCE_END_CANCELLED_BEFORE_ENTRY",
    "UNRESOLVED_NO_ENTRY_OBSERVATION": "NO_EXECUTABLE_ENTRY_BEFORE_SESSION_END",
    "UNRESOLVED_NO_LATER_EVENT": "SESSION_ENDED_BEFORE_ENTRY",
    "UNRESOLVED_AT_HARD_FLAT": "CANCELLED_AT_HARD_FLAT_BERLIN",
}


def simulate_berlin_session(
    tape: BerlinSessionCausalTape,
    accepted_interactions: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Mapping[str, Any]],
) -> historical_matrix.SessionResult:
    """Reuse frozen pending/position chronology and eliminate unresolved trades."""
    result = historical_matrix.simulate_independent_session(tape, accepted_interactions, indexes)
    replacements = 0
    for identifier, reason in tuple(result.terminal_outcomes.items()):
        replacement = NONTRADE_UNRESOLVED_CLASSIFICATION.get(reason)
        if replacement is None:
            continue
        result.terminal_outcomes[identifier] = replacement
        result.other_terminal[replacement] = result.other_terminal.get(replacement, 0) + 1
        replacements += 1
    result.unresolved -= replacements
    if result.unresolved != 0:
        raise BerlinExecutionError(f"UNRESOLVED_TRADE_POLICY_VIOLATION:{tape.day}:{result.unresolved}")
    for trade in result.trades:
        if trade["exit_reason"] not in TERMINAL_EXIT_REASONS:
            raise BerlinExecutionError(f"unsupported entered-trade terminal reason: {trade['exit_reason']}")
        if int(trade["entry_timestamp_ns"]) >= tape.hard_flat_timestamp_ns:
            raise BerlinExecutionError(f"entry at or after Berlin hard flat: {tape.day}")
        if int(trade["exit_timestamp_ns"]) >= tape.maintenance_start_ns:
            raise OpenPositionAtMaintenance(f"OPEN_POSITION_AT_MAINTENANCE:{tape.day}")
    return result
