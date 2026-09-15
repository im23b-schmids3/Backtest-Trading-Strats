"""December-only offline weight x quality research over the sealed causal tape.

This module never opens a DBN and has no network client.  It first runs the
frozen V3 December replay through the canonical causal-tape adapter.  Only
after that exact gate passes does it evaluate the predeclared 3,876 x 6 grid.

The event tape is indexed one December session at a time.  Each configuration
still owns its own pending-setup and one-position state; compact indexing only
answers the same causal entry/exit observations without scanning millions of
irrelevant quote changes once per configuration.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import statistics
import sys
import time
import tracemalloc
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import causal_master_tape as master
from . import historical_runner as historical
from .model import (
    ES_COMMISSION,
    ES_POINT_VALUE,
    MES_COMMISSION,
    MES_POINT_VALUE,
    STOP_BUFFER_TICKS,
    TARGET_R,
    TICK,
    initial_prices,
    size_for_instrument,
)


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_WEIGHT_Q_RESEARCH_DEC2025"
EVIDENCE_LABEL = "DECEMBER_2025_WEIGHT_Q_RESEARCH_NOT_OOS_EVIDENCE"
MASTER_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_CAUSAL_MASTER_DEC2025_JAN2026")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_WEIGHT_Q_RESEARCH_DEC2025")
DECEMBER_PREFIX = "2025-12"
QUALITY_THRESHOLDS = tuple(Decimal(value) for value in ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60"))
WEIGHT_UNIT = Decimal("0.05")
WEIGHT_TOTAL_UNITS = 20
EXPECTED_WEIGHT_COUNT = 3_876
EXPECTED_CONFIGURATION_COUNT = 23_256
MIN_RANKING_TRADES = 15
TOP_TABLE_LIMIT = 100
PROGRESS_INTERVAL = 500
NO_AUTOMATIC_V5_SELECTION = True

RESULT_COLUMNS = (
    "config_id", "G1", "G2", "G3", "G4", "G5", "quality_threshold",
    "accepted_setups", "confirmations", "confirmation_expiries",
    "active_position_blocks", "trades", "wins", "losses", "win_rate",
    "total_r", "average_r", "median_r", "net_pnl_usd", "profit_factor",
    "max_cumulative_drawdown_r", "es_trades", "mes_trades", "target_exits",
    "stop_exits", "hard_cutoff_exits", "unresolved",
    "other_terminal_count", "other_terminal_outcomes",
    "common_trades_with_v3", "unique_trades_vs_v3", "v3_trades_lost",
)


class WeightQResearchError(RuntimeError):
    """The offline matrix cannot preserve the sealed causal contract."""


def generate_weight_grid() -> tuple[tuple[int, int, int, int, int], ...]:
    """Return every positive five-part composition of 20 exact 0.05 units."""
    rows = tuple(
        (g1, g2, g3, g4, WEIGHT_TOTAL_UNITS - g1 - g2 - g3 - g4)
        for g1 in range(1, WEIGHT_TOTAL_UNITS - 3)
        for g2 in range(1, WEIGHT_TOTAL_UNITS - g1 - 2)
        for g3 in range(1, WEIGHT_TOTAL_UNITS - g1 - g2 - 1)
        for g4 in range(1, WEIGHT_TOTAL_UNITS - g1 - g2 - g3)
        if WEIGHT_TOTAL_UNITS - g1 - g2 - g3 - g4 >= 1
    )
    if (
        len(rows) != EXPECTED_WEIGHT_COUNT
        or len(set(rows)) != EXPECTED_WEIGHT_COUNT
        or any(sum(row) != WEIGHT_TOTAL_UNITS or min(row) < 1 for row in rows)
    ):
        raise WeightQResearchError("the sealed weight grid does not contain exactly 3,876 legal vectors")
    return rows


def unit_weights(units: Sequence[int]) -> dict[str, Decimal]:
    if len(units) != 5 or sum(units) != WEIGHT_TOTAL_UNITS or min(units) < 1:
        raise WeightQResearchError("illegal integer weight vector")
    return {name: Decimal(value) * WEIGHT_UNIT for name, value in zip(master.SCORE_FIELDS, units)}


def config_id(units: Sequence[int], threshold: Decimal | str | float) -> str:
    q = master.normalize_threshold(threshold)
    return "W" + "-".join(f"{value:02d}" for value in units) + f"-Q{int(q * 100):02d}"


def configuration_registry() -> tuple[tuple[tuple[int, int, int, int, int], Decimal], ...]:
    registry = tuple((weights, threshold) for weights in generate_weight_grid() for threshold in QUALITY_THRESHOLDS)
    identities = {config_id(weights, threshold) for weights, threshold in registry}
    if len(registry) != EXPECTED_CONFIGURATION_COUNT or len(identities) != EXPECTED_CONFIGURATION_COUNT:
        raise WeightQResearchError("weight x quality registry cardinality mismatch")
    return registry


def weight_neighbors(units: Sequence[int]) -> tuple[tuple[int, int, int, int, int], ...]:
    source = tuple(int(value) for value in units)
    neighbors: set[tuple[int, int, int, int, int]] = set()
    for donor in range(5):
        if source[donor] <= 1:
            continue
        for receiver in range(5):
            if receiver == donor:
                continue
            candidate = list(source)
            candidate[donor] -= 1
            candidate[receiver] += 1
            neighbors.add(tuple(candidate))
    return tuple(sorted(neighbors))


def quality_neighbors(threshold: Decimal | str | float) -> tuple[Decimal, ...]:
    q = master.normalize_threshold(threshold)
    index = QUALITY_THRESHOLDS.index(q)
    return tuple(QUALITY_THRESHOLDS[pos] for pos in (index - 1, index + 1) if 0 <= pos < len(QUALITY_THRESHOLDS))


def combined_neighbor_ids(units: Sequence[int], threshold: Decimal | str | float) -> tuple[str, ...]:
    q = master.normalize_threshold(threshold)
    identifiers = {config_id(item, q) for item in weight_neighbors(units)}
    identifiers.update(config_id(units, adjacent) for adjacent in quality_neighbors(q))
    return tuple(sorted(identifiers))


def recompute_grid_score(row: Mapping[str, Any], units: Sequence[int]) -> float:
    """Delegate to the frozen score implementation, including its penalty."""
    return master.recompute_quality(row, unit_weights(units))


def _accepted_mask(
    rows: Sequence[Mapping[str, Any]], primitive_ok: np.ndarray,
    approximate_scores: np.ndarray, units: Sequence[int], threshold: Decimal,
) -> np.ndarray:
    """Vectorize ordinary cases and resolve threshold-boundary rows exactly."""
    mask = primitive_ok & (approximate_scores >= float(threshold))
    near = np.flatnonzero(np.abs(approximate_scores - float(threshold)) <= 1e-12)
    if len(near):
        weights = unit_weights(units)
        for index in near:
            mask[index] = master.interaction_is_accepted(rows[int(index)], threshold=threshold, weights=weights)
    return mask


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _BlockExtrema:
    """First threshold crossing over one compact quote array.

    Block minima/maxima keep memory linear and make each path lookup bounded by
    two partial blocks plus the small block summary, avoiding a Python segment
    tree containing millions of objects.
    """

    def __init__(self, values: array, *, block_size: int = 1024) -> None:
        self.values = values
        self.block_size = block_size
        self.minimum: list[float] = []
        self.maximum: list[float] = []
        for start in range(0, len(values), block_size):
            finite = [value for value in values[start:start + block_size] if math.isfinite(value)]
            self.minimum.append(min(finite) if finite else math.inf)
            self.maximum.append(max(finite) if finite else -math.inf)

    def _first(self, start: int, end: int, threshold: float, *, lower: bool) -> int | None:
        if start >= end:
            return None
        block = start // self.block_size
        while start < end:
            block_end = min(end, (block + 1) * self.block_size)
            whole = start == block * self.block_size and block_end == min(len(self.values), (block + 1) * self.block_size)
            possible = (
                self.minimum[block] <= threshold if lower else self.maximum[block] >= threshold
            )
            if not whole or possible:
                for index in range(start, block_end):
                    value = self.values[index]
                    if math.isfinite(value) and ((value <= threshold) if lower else (value >= threshold)):
                        return index
            start = block_end
            block += 1
        return None

    def first_le(self, start: int, end: int, threshold: float) -> int | None:
        return self._first(start, end, threshold, lower=True)

    def first_ge(self, start: int, end: int, threshold: float) -> int | None:
        return self._first(start, end, threshold, lower=False)


@dataclass(frozen=True)
class EntryOutcome:
    terminal_reason: str
    trade: Mapping[str, Any] | None = None
    exit_event_ordinal: int | None = None
    open_until_source_end: bool = False
    interaction_id: str | None = None
    boundary: "TapeBoundary | None" = None


@dataclass(frozen=True)
class TapeBoundary:
    """One explicit non-executable interval in a compact causal tape.

    Reopen observations are evidence about tape completeness only.  They do
    not authorize an open counterfactual position to cross the interval: the
    frozen native runners fail closed for both scheduled maintenance and
    temporary book reconstruction when a position is open.
    """

    start_ordinal: int
    start_timestamp_ns: int
    book_state: str
    classification: str
    first_regular_ordinal: int | None
    first_regular_timestamp_ns: int | None
    es_reopen_ordinal: int | None
    es_reopen_timestamp_ns: int | None
    es_reopen_bid: float | None
    es_reopen_ask: float | None
    mes_reopen_ordinal: int | None
    mes_reopen_timestamp_ns: int | None
    mes_reopen_bid: float | None
    mes_reopen_ask: float | None

    def reopen_ordinal(self, instrument: str) -> int | None:
        return self.es_reopen_ordinal if instrument == "ES" else self.mes_reopen_ordinal

    def reopen_timestamp_ns(self, instrument: str) -> int | None:
        return self.es_reopen_timestamp_ns if instrument == "ES" else self.mes_reopen_timestamp_ns


class PositionBoundaryOverlap(WeightQResearchError):
    """An entered counterfactual reaches a frozen fail-closed source state."""

    def __init__(self, boundary: TapeBoundary, instrument: str) -> None:
        self.boundary = boundary
        self.instrument = instrument
        super().__init__(
            "counterfactual position crosses a classified non-executable tape boundary: "
            f"{boundary.classification}"
        )


def _boundary_classification(book_state: str) -> str:
    if book_state == "MAINTENANCE":
        return "EXPECTED_SCHEDULED_MAINTENANCE"
    if book_state == "TEMPORARILY_NON_EXECUTABLE":
        return "TEMPORARY_BOOK_RECONSTRUCTION"
    if book_state == "WAITING_FOR_REOPEN_BOOK":
        return "UNRESOLVED_INVALID_BOOK"
    return "OTHER_INTEGRITY_FAILURE"


class SessionCausalTape:
    """Compact, read-only execution index for one causal event partition."""

    def __init__(
        self, day: str, rows: Iterable[Mapping[str, Any]], *,
        stop_buffer_ticks: int = STOP_BUFFER_TICKS,
        target_r: float = TARGET_R,
    ) -> None:
        self.day = day
        self.stop_buffer_ticks = int(stop_buffer_ticks)
        self.target_r = float(target_r)
        if self.stop_buffer_ticks < 0 or self.target_r <= 0:
            raise WeightQResearchError("invalid execution geometry")
        self.ordinals = array("q")
        self.timestamps = array("q")
        self.streams = bytearray()
        self.es_bid = array("d")
        self.es_ask = array("d")
        self.mes_bid = array("d")
        self.mes_ask = array("d")
        # Retained for versioned execution contracts that must audit the exact
        # causal source timestamp of the last executable quote.  Historical V3
        # does not read these arrays, so its behavior remains unchanged.
        self.es_quote_timestamps = array("q")
        self.mes_quote_timestamps = array("q")
        self.durable_non_executable_ordinals: list[int] = []
        boundary_rows: list[dict[str, Any]] = []
        self.hard_event: dict[str, Any] | None = None
        self.source_end_event: dict[str, Any] | None = None
        self._outcome_cache: dict[tuple[str, int, int, float], EntryOutcome] = {}
        expected_ordinal = 0
        for row in rows:
            ordinal = int(row["event_ordinal"])
            if ordinal != expected_ordinal:
                raise WeightQResearchError(f"non-contiguous causal event ordinal in {day}: {ordinal}")
            expected_ordinal += 1
            event_type = str(row["event_type"])
            if event_type == "BOOK_NON_EXECUTABLE":
                book_state = str(row.get("book_state"))
                classification = _boundary_classification(book_state)
                if classification == "OTHER_INTEGRITY_FAILURE":
                    raise WeightQResearchError(
                        f"unsupported non-executable tape state in {day}: {book_state}"
                    )
                boundary_rows.append(dict(row))
                if book_state != "TEMPORARILY_NON_EXECUTABLE":
                    self.durable_non_executable_ordinals.append(ordinal)
                continue
            if event_type == "HARD_FLAT":
                if self.hard_event is not None:
                    raise WeightQResearchError(f"duplicate hard-flat event in {day}")
                self.hard_event = dict(row)
                continue
            if event_type == "SOURCE_END":
                if self.source_end_event is not None:
                    raise WeightQResearchError(f"duplicate source-end event in {day}")
                self.source_end_event = dict(row)
                continue
            stream = str(row["stream"])
            if stream not in {"ES", "MES"}:
                raise WeightQResearchError(f"unsupported regular stream in {day}: {stream}")
            self.ordinals.append(ordinal)
            self.timestamps.append(int(row["timestamp_ns"]))
            self.streams.append(1 if stream == "ES" else 2)
            self.es_bid.append(float(row["es_bid"]) if row.get("es_bid") is not None else math.nan)
            self.es_ask.append(float(row["es_ask"]) if row.get("es_ask") is not None else math.nan)
            self.mes_bid.append(float(row["mes_bid"]) if row.get("mes_bid") is not None else math.nan)
            self.mes_ask.append(float(row["mes_ask"]) if row.get("mes_ask") is not None else math.nan)
            self.es_quote_timestamps.append(
                int(row["es_quote_timestamp_ns"]) if row.get("es_quote_timestamp_ns") is not None else -1
            )
            self.mes_quote_timestamps.append(
                int(row["mes_quote_timestamp_ns"]) if row.get("mes_quote_timestamp_ns") is not None else -1
            )
        if (self.hard_event is None) == (self.source_end_event is None):
            raise WeightQResearchError(
                f"causal session must have exactly one hard-flat or source-end event: {day}"
            )
        if not self.ordinals:
            raise WeightQResearchError(f"causal session has no executable observations: {day}")

        executable_reopen_indexes: dict[str, tuple[array, array]] = {}
        for instrument, bid_values, ask_values in (
            ("ES", self.es_bid, self.es_ask), ("MES", self.mes_bid, self.mes_ask),
        ):
            executable_ordinals, regular_indexes = array("q"), array("q")
            for index, (bid, ask) in enumerate(zip(bid_values, ask_values)):
                if math.isfinite(bid) and math.isfinite(ask) and ask > bid:
                    executable_ordinals.append(int(self.ordinals[index]))
                    regular_indexes.append(index)
            executable_reopen_indexes[instrument] = executable_ordinals, regular_indexes

        def first_reopen(start_ordinal: int, instrument: str) -> tuple[int | None, int | None, float | None, float | None]:
            executable_ordinals, regular_indexes = executable_reopen_indexes[instrument]
            candidate = bisect.bisect_right(executable_ordinals, start_ordinal)
            if candidate < len(executable_ordinals):
                index = int(regular_indexes[candidate])
                bid_values, ask_values = (
                    (self.es_bid, self.es_ask) if instrument == "ES" else (self.mes_bid, self.mes_ask)
                )
                return (
                    int(self.ordinals[index]), int(self.timestamps[index]),
                    float(bid_values[index]), float(ask_values[index]),
                )
            return None, None, None, None

        materialized_boundaries: list[TapeBoundary] = []
        for row in boundary_rows:
            start_ordinal = int(row["event_ordinal"])
            regular_index = bisect.bisect_right(self.ordinals, start_ordinal)
            first_ordinal = int(self.ordinals[regular_index]) if regular_index < len(self.ordinals) else None
            first_timestamp = int(self.timestamps[regular_index]) if regular_index < len(self.timestamps) else None
            es_ordinal, es_timestamp, es_bid, es_ask = first_reopen(start_ordinal, "ES")
            mes_ordinal, mes_timestamp, mes_bid, mes_ask = first_reopen(start_ordinal, "MES")
            materialized_boundaries.append(TapeBoundary(
                start_ordinal=start_ordinal,
                start_timestamp_ns=int(row["timestamp_ns"]),
                book_state=str(row.get("book_state")),
                classification=_boundary_classification(str(row.get("book_state"))),
                first_regular_ordinal=first_ordinal,
                first_regular_timestamp_ns=first_timestamp,
                es_reopen_ordinal=es_ordinal,
                es_reopen_timestamp_ns=es_timestamp,
                es_reopen_bid=es_bid,
                es_reopen_ask=es_ask,
                mes_reopen_ordinal=mes_ordinal,
                mes_reopen_timestamp_ns=mes_timestamp,
                mes_reopen_bid=mes_bid,
                mes_reopen_ask=mes_ask,
            ))
        self.non_executable_boundaries = tuple(materialized_boundaries)
        self._position_boundary_ordinals = tuple(item.start_ordinal for item in self.non_executable_boundaries)

        # Stop/target monitoring only reacts to the position's native stream.
        es_bid = array("d", (value if stream == 1 else math.nan for value, stream in zip(self.es_bid, self.streams)))
        es_ask = array("d", (value if stream == 1 else math.nan for value, stream in zip(self.es_ask, self.streams)))
        mes_bid = array("d", (value if stream == 2 else math.nan for value, stream in zip(self.mes_bid, self.streams)))
        mes_ask = array("d", (value if stream == 2 else math.nan for value, stream in zip(self.mes_ask, self.streams)))
        self._series = {
            ("ES", "bid"): _BlockExtrema(es_bid), ("ES", "ask"): _BlockExtrema(es_ask),
            ("MES", "bid"): _BlockExtrema(mes_bid), ("MES", "ask"): _BlockExtrema(mes_ask),
        }

    @classmethod
    def from_parquet(
        cls, day: str, path: Path, *, stop_buffer_ticks: int = STOP_BUFFER_TICKS,
        target_r: float = TARGET_R,
    ) -> "SessionCausalTape":
        # The canonical iterator opens only Parquet, closes every reader, and
        # has no DBN or network dependency.
        return cls(
            day, master._iter_parquet_rows(path), stop_buffer_ticks=stop_buffer_ticks,
            target_r=target_r,
        )

    def regular_index(self, event_ordinal: int) -> int:
        index = bisect.bisect_left(self.ordinals, event_ordinal)
        if index >= len(self.ordinals) or self.ordinals[index] != event_ordinal:
            raise WeightQResearchError(f"entry probe does not reference a regular event: {self.day}/{event_ordinal}")
        return index

    def next_regular_ordinal(self, event_ordinal: int) -> int | None:
        index = bisect.bisect_right(self.ordinals, event_ordinal)
        return int(self.ordinals[index]) if index < len(self.ordinals) else None

    def _first_durable_boundary(self, start_exclusive: int, end_inclusive: int) -> int | None:
        index = bisect.bisect_right(self.durable_non_executable_ordinals, start_exclusive)
        if index < len(self.durable_non_executable_ordinals):
            value = self.durable_non_executable_ordinals[index]
            if value <= end_inclusive:
                return value
        return None

    def first_position_boundary(
        self, start_exclusive: int, end_inclusive: int,
    ) -> TapeBoundary | None:
        index = bisect.bisect_right(self._position_boundary_ordinals, start_exclusive)
        if index < len(self.non_executable_boundaries):
            boundary = self.non_executable_boundaries[index]
            if boundary.start_ordinal <= end_inclusive:
                return boundary
        return None

    def _trade_exit(
        self, *, instrument: str, direction: str, entry_ordinal: int,
        stop: float, target: float,
    ) -> tuple[int, int, float, str] | None:
        terminal = self.hard_event or self.source_end_event
        assert terminal is not None
        hard_ordinal = int(terminal["event_ordinal"])
        start = bisect.bisect_right(self.ordinals, entry_ordinal)
        end = bisect.bisect_left(self.ordinals, hard_ordinal)
        if direction == "LONG":
            series = self._series[(instrument, "bid")]
            stop_index = series.first_le(start, end, stop)
            target_index = series.first_ge(start, end, target)
        else:
            series = self._series[(instrument, "ask")]
            stop_index = series.first_ge(start, end, stop)
            target_index = series.first_le(start, end, target)
        indexes = [(index, reason) for index, reason in ((stop_index, "STOP"), (target_index, "TARGET")) if index is not None]
        if indexes:
            trigger_index, reason = min(indexes, key=lambda item: (int(self.ordinals[item[0]]), 0 if item[1] == "STOP" else 1))
            exit_ordinal = int(self.ordinals[trigger_index])
            boundary = self.first_position_boundary(entry_ordinal, exit_ordinal)
            if boundary is not None:
                raise PositionBoundaryOverlap(boundary, instrument)
            reference = (
                self.es_bid[trigger_index] if instrument == "ES" and direction == "LONG"
                else self.es_ask[trigger_index] if instrument == "ES"
                else self.mes_bid[trigger_index] if direction == "LONG"
                else self.mes_ask[trigger_index]
            )
            return exit_ordinal, int(self.timestamps[trigger_index]), float(reference), reason

        boundary = self.first_position_boundary(entry_ordinal, hard_ordinal)
        if boundary is not None:
            raise PositionBoundaryOverlap(boundary, instrument)
        if self.source_end_event is not None:
            return None
        quote_prefix = "es" if instrument == "ES" else "mes"
        quote_timestamp = self.hard_event.get(f"{quote_prefix}_quote_timestamp_ns")
        reference_name = f"{quote_prefix}_{'bid' if direction == 'LONG' else 'ask'}"
        reference = self.hard_event.get(reference_name)
        if quote_timestamp is None or reference is None:
            raise WeightQResearchError("hard-flat event lacks the frozen native execution quote")
        return hard_ordinal, int(quote_timestamp), float(reference), str(
            self.hard_event.get("hard_flat_reason") or "HARD_CUTOFF_2245"
        )

    def entry_outcome(self, interaction: Mapping[str, Any], event_ordinal: int) -> EntryOutcome:
        identifier = str(interaction["interaction_id"])
        cache_key = (identifier, event_ordinal, self.stop_buffer_ticks, self.target_r)
        cached = self._outcome_cache.get(cache_key)
        if cached is not None:
            return cached
        index = self.regular_index(event_ordinal)
        if not math.isfinite(self.es_bid[index]) or not math.isfinite(self.es_ask[index]):
            outcome = EntryOutcome("WAIT_FOR_ES_EXECUTABLE_QUOTE")
            self._outcome_cache[cache_key] = outcome
            return outcome
        direction = str(interaction["direction"])
        prices = initial_prices(
            direction, self.es_bid[index], self.es_ask[index],
            float(interaction["zone_low"]), float(interaction["zone_high"]),
            stop_buffer_ticks=self.stop_buffer_ticks, target_r=self.target_r,
        )
        sizing = size_for_instrument(prices, "ES")
        instrument = "ES"
        if int(sizing["contracts"]) < 1:
            if not math.isfinite(self.mes_bid[index]) or not math.isfinite(self.mes_ask[index]):
                outcome = EntryOutcome("MES_EXECUTION_UNAVAILABLE")
                self._outcome_cache[cache_key] = outcome
                return outcome
            prices = initial_prices(
                direction, self.mes_bid[index], self.mes_ask[index],
                float(interaction["zone_low"]), float(interaction["zone_high"]),
                stop_buffer_ticks=self.stop_buffer_ticks, target_r=self.target_r,
            )
            sizing = size_for_instrument(prices, "MES")
            instrument = "MES"
        if int(sizing["contracts"]) < 1:
            outcome = EntryOutcome("INSUFFICIENT_RISK_BUDGET_FOR_ONE_CONTRACT")
            self._outcome_cache[cache_key] = outcome
            return outcome

        try:
            exit_observation = self._trade_exit(
                instrument=instrument, direction=str(prices["direction"]), entry_ordinal=event_ordinal,
                stop=float(prices["stop"]), target=float(prices["target"]),
            )
        except PositionBoundaryOverlap as exc:
            outcome = EntryOutcome(
                f"POSITION_UNRESOLVED_{exc.boundary.classification}", None,
                exc.boundary.start_ordinal, True, identifier, exc.boundary,
            )
            self._outcome_cache[cache_key] = outcome
            return outcome
        if exit_observation is None:
            assert self.source_end_event is not None
            outcome = EntryOutcome(
                "ENTRY_UNRESOLVED_SOURCE_END", None,
                int(self.source_end_event["event_ordinal"]), True, identifier,
            )
            self._outcome_cache[cache_key] = outcome
            return outcome
        exit_ordinal, exit_timestamp, reference, reason = exit_observation
        long = prices["direction"] == "LONG"
        exit_price = reference - TICK if long else reference + TICK
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
            "trade_id": f"L2T:{setup_id}", "setup_id": setup_id, "date": self.day,
            "interaction_id": str(interaction["source_interaction_id"]),
            "direction": str(prices["direction"]), "level": str(interaction["level"]),
            "instrument": instrument, "contracts": contracts,
            "entry_timestamp_ns": int(self.timestamps[index]), "entry": float(prices["entry"]),
            "stop": float(prices["stop"]), "target": float(prices["target"]),
            "exit_timestamp_ns": exit_timestamp, "exit": exit_price, "exit_reason": reason,
            "gross_pnl_usd": gross, "total_costs_usd": fees, "net_pnl_usd": gross - fees,
            "r_multiple": (gross - fees) / initial_risk if initial_risk else None,
        }
        outcome = EntryOutcome("ENTRY", trade, exit_ordinal)
        self._outcome_cache[cache_key] = outcome
        return outcome


@dataclass
class SessionResult:
    accepted_setups: int = 0
    confirmations: int = 0
    confirmation_expiries: int = 0
    active_position_blocks: int = 0
    unresolved: int = 0
    other_terminal: dict[str, int] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    terminal_outcomes: dict[str, str] = field(default_factory=dict)


def simulate_independent_session(
    tape: SessionCausalTape,
    accepted_interactions: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Mapping[str, Any]],
) -> SessionResult:
    """Run one configuration's exact pending/position chronology for a session."""
    result = SessionResult(accepted_setups=len(accepted_interactions))

    def record_terminal(identifier: str, outcome: str) -> None:
        if identifier in result.terminal_outcomes:
            raise WeightQResearchError(f"duplicate setup terminal outcome: {identifier}")
        result.terminal_outcomes[identifier] = outcome

    due: list[tuple[int, str, Mapping[str, Any]]] = []
    terminal: set[str] = set()
    for row in accepted_interactions:
        identifier = str(row["interaction_id"])
        index = indexes[identifier]
        confirmation = index.get("derived_first_confirmation_timestamp_ns")
        if confirmation is None:
            path_end_ns = int(index.get("counterfactual_path_end_ns") or 0)
            if tape.source_end_event is not None and path_end_ns <= int(row["interaction_end_ns"]) + master.MAX_CONFIRMATION_NS:
                result.unresolved += 1
                record_terminal(identifier, "CONFIRMATION_UNRESOLVED_SOURCE_INCOMPLETE")
            else:
                result.confirmation_expiries += 1
                terminal.add(identifier)
                record_terminal(identifier, "CONFIRMATION_WINDOW_EXPIRED")
            continue
        result.confirmations += 1
        ordinal = index.get("entry_observation_event_ordinal")
        if ordinal is None:
            result.unresolved += 1
            record_terminal(
                identifier,
                "EXECUTION_UNRESOLVED_SOURCE_INCOMPLETE"
                if tape.source_end_event is not None else "UNRESOLVED_NO_ENTRY_OBSERVATION",
            )
            continue
        due_ordinal = int(ordinal)
        # A durable source boundary after confirmation but before the first
        # post-latency ordinary observation preserves the frozen failure path.
        confirm_ordinal = bisect.bisect_left(tape.timestamps, int(confirmation))
        confirm_event_ordinal = int(tape.ordinals[min(confirm_ordinal, len(tape.ordinals) - 1)])
        if tape._first_durable_boundary(confirm_event_ordinal, due_ordinal - 1) is not None:
            result.other_terminal["SOURCE_NON_EXECUTABLE_BEFORE_ENTRY"] = (
                result.other_terminal.get("SOURCE_NON_EXECUTABLE_BEFORE_ENTRY", 0) + 1
            )
            terminal.add(identifier)
            record_terminal(identifier, "SOURCE_NON_EXECUTABLE_BEFORE_ENTRY")
            continue
        due.append((due_ordinal, f"L2:{row['source_interaction_id']}", row))
    due.sort(key=lambda item: (item[0], item[1]))

    due_cursor = 0
    waiting: dict[str, Mapping[str, Any]] = {}
    active: EntryOutcome | None = None
    current_ordinal = -1
    while due_cursor < len(due) or waiting or active is not None:
        candidates: list[int] = []
        if due_cursor < len(due):
            candidates.append(due[due_cursor][0])
        if active is not None:
            assert active.exit_event_ordinal is not None
            candidates.append(active.exit_event_ordinal)
        if waiting:
            next_regular = tape.next_regular_ordinal(current_ordinal)
            if next_regular is not None:
                candidates.append(next_regular)
        if not candidates:
            result.unresolved += len(waiting)
            for row in waiting.values():
                record_terminal(str(row["interaction_id"]), "UNRESOLVED_NO_LATER_EVENT")
            break
        event_ordinal = min(candidates)
        current_ordinal = event_ordinal
        if active is not None and active.exit_event_ordinal == event_ordinal:
            if active.open_until_source_end:
                result.unresolved += 1
                if active.interaction_id is None:
                    raise WeightQResearchError("unresolved position lost its interaction identity")
                identifier = active.interaction_id
                terminal_reason = (
                    "UNRESOLVED_SOURCE_END"
                    if active.terminal_reason == "ENTRY_UNRESOLVED_SOURCE_END"
                    else active.terminal_reason
                )
                record_terminal(identifier, terminal_reason)
                if active.boundary is not None:
                    # The frozen real runner aborts this session at the source
                    # state.  Preserve that fail-closed portfolio chronology:
                    # no later setup may be interpreted after the unresolved
                    # open-position boundary.
                    for row in waiting.values():
                        result.unresolved += 1
                        record_terminal(
                            str(row["interaction_id"]),
                            "SESSION_UNRESOLVED_AFTER_NON_EXECUTABLE_BOUNDARY",
                        )
                    waiting.clear()
                    while due_cursor < len(due):
                        row = due[due_cursor][2]
                        if str(row["interaction_id"]) not in result.terminal_outcomes:
                            result.unresolved += 1
                            record_terminal(
                                str(row["interaction_id"]),
                                "SESSION_UNRESOLVED_AFTER_NON_EXECUTABLE_BOUNDARY",
                            )
                        due_cursor += 1
                    active = None
                    break
            else:
                assert active.trade is not None
                result.trades.append(dict(active.trade))
            active = None
        while due_cursor < len(due) and due[due_cursor][0] <= event_ordinal:
            _ordinal, setup_id, row = due[due_cursor]
            waiting[setup_id] = row
            due_cursor += 1

        terminal_event = tape.hard_event or tape.source_end_event
        assert terminal_event is not None
        hard_ordinal = int(terminal_event["event_ordinal"])
        if event_ordinal >= hard_ordinal:
            result.unresolved += len(waiting)
            for row in waiting.values():
                record_terminal(str(row["interaction_id"]), "UNRESOLVED_AT_HARD_FLAT")
            waiting.clear()
            continue
        if not waiting:
            continue
        if active is not None:
            result.active_position_blocks += len(waiting)
            for row in waiting.values():
                record_terminal(str(row["interaction_id"]), "COMPLIANCE_BLOCK_ACTIVE_POSITION")
            waiting.clear()
            continue

        for setup_id in sorted(tuple(waiting)):
            row = waiting[setup_id]
            outcome = tape.entry_outcome(row, event_ordinal)
            if outcome.terminal_reason == "WAIT_FOR_ES_EXECUTABLE_QUOTE":
                continue
            waiting.pop(setup_id)
            if outcome.trade is None:
                if outcome.open_until_source_end:
                    active = outcome
                    break
                result.other_terminal[outcome.terminal_reason] = result.other_terminal.get(outcome.terminal_reason, 0) + 1
                record_terminal(str(row["interaction_id"]), outcome.terminal_reason)
                continue
            active = outcome
            record_terminal(str(row["interaction_id"]), "TRADE_EXECUTED")
            # HistoricalL2Runner breaks immediately after the first successful
            # entry; later setup IDs remain confirmed until the next event.
            break

    result.trades.sort(key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    classified = (
        result.confirmation_expiries + result.active_position_blocks + result.unresolved
        + sum(result.other_terminal.values()) + len(result.trades)
    )
    if classified != result.accepted_setups:
        raise WeightQResearchError(
            f"session setup reconciliation failed for {tape.day}: {classified} != {result.accepted_setups}"
        )
    if len(result.terminal_outcomes) != result.accepted_setups:
        raise WeightQResearchError(
            f"session terminal audit failed for {tape.day}: "
            f"{len(result.terminal_outcomes)} != {result.accepted_setups}"
        )
    return result


@dataclass
class ConfigurationAccumulator:
    weights: tuple[int, int, int, int, int]
    threshold: Decimal
    accepted_setups: int = 0
    confirmations: int = 0
    confirmation_expiries: int = 0
    active_position_blocks: int = 0
    unresolved: int = 0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    net_pnl_usd: float = 0.0
    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    equity_r: float = 0.0
    peak_r: float = 0.0
    max_drawdown_r: float = 0.0
    es_trades: int = 0
    mes_trades: int = 0
    target_exits: int = 0
    stop_exits: int = 0
    hard_cutoff_exits: int = 0
    common_v3: int = 0
    other_terminal: dict[str, int] = field(default_factory=dict)
    r_values: list[float] = field(default_factory=list)

    def add(self, session: SessionResult, baseline_trade_ids: set[str]) -> None:
        self.accepted_setups += session.accepted_setups
        self.confirmations += session.confirmations
        self.confirmation_expiries += session.confirmation_expiries
        self.active_position_blocks += session.active_position_blocks
        self.unresolved += session.unresolved
        for reason, count in session.other_terminal.items():
            self.other_terminal[reason] = self.other_terminal.get(reason, 0) + count
        for trade in session.trades:
            r_value = float(trade["r_multiple"] or 0.0)
            net = float(trade["net_pnl_usd"])
            self.trade_count += 1
            self.wins += net > 0
            self.losses += net < 0
            self.total_r += r_value
            self.net_pnl_usd += net
            self.gross_profit_usd += max(net, 0.0)
            self.gross_loss_usd += max(-net, 0.0)
            self.equity_r += r_value
            self.peak_r = max(self.peak_r, self.equity_r)
            self.max_drawdown_r = min(self.max_drawdown_r, self.equity_r - self.peak_r)
            self.es_trades += trade["instrument"] == "ES"
            self.mes_trades += trade["instrument"] == "MES"
            self.target_exits += trade["exit_reason"] == "TARGET"
            self.stop_exits += trade["exit_reason"] == "STOP"
            self.hard_cutoff_exits += str(trade["exit_reason"]).startswith("HARD_")
            self.common_v3 += str(trade["trade_id"]) in baseline_trade_ids
            self.r_values.append(r_value)

    def row(self, baseline_count: int) -> dict[str, Any]:
        values = [float(value) * float(WEIGHT_UNIT) for value in self.weights]
        terminal_count = (
            self.confirmation_expiries + self.active_position_blocks + self.trade_count
            + self.unresolved + sum(self.other_terminal.values())
        )
        if terminal_count != self.accepted_setups:
            raise WeightQResearchError(
                f"configuration setup reconciliation failed: {terminal_count} != {self.accepted_setups}"
            )
        return {
            "config_id": config_id(self.weights, self.threshold),
            **{f"G{index}": value for index, value in enumerate(values, start=1)},
            "quality_threshold": float(self.threshold),
            "accepted_setups": self.accepted_setups,
            "confirmations": self.confirmations,
            "confirmation_expiries": self.confirmation_expiries,
            "active_position_blocks": self.active_position_blocks,
            "trades": self.trade_count,
            "wins": self.wins, "losses": self.losses,
            "win_rate": self.wins / self.trade_count if self.trade_count else 0.0,
            "total_r": self.total_r,
            "average_r": self.total_r / self.trade_count if self.trade_count else 0.0,
            "median_r": statistics.median(self.r_values) if self.r_values else None,
            "net_pnl_usd": self.net_pnl_usd,
            "profit_factor": self.gross_profit_usd / self.gross_loss_usd if self.gross_loss_usd else None,
            "max_cumulative_drawdown_r": self.max_drawdown_r,
            "es_trades": self.es_trades, "mes_trades": self.mes_trades,
            "target_exits": self.target_exits, "stop_exits": self.stop_exits,
            "hard_cutoff_exits": self.hard_cutoff_exits, "unresolved": self.unresolved,
            "other_terminal_count": sum(self.other_terminal.values()),
            "other_terminal_outcomes": json.dumps(self.other_terminal, sort_keys=True, separators=(",", ":")),
            "common_trades_with_v3": self.common_v3,
            "unique_trades_vs_v3": self.trade_count - self.common_v3,
            "v3_trades_lost": baseline_count - self.common_v3,
        }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _median(values: Iterable[object]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return statistics.median(finite) if finite else None


def _mean(values: Iterable[object]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return statistics.fmean(finite) if finite else None


def _neighbor_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_r = [float(row["total_r"]) for row in rows]
    trades = [int(row["trades"]) for row in rows]
    profit_factors = [value for row in rows if (value := _finite(row.get("profit_factor"))) is not None]
    drawdowns = [float(row["max_cumulative_drawdown_r"]) for row in rows]
    return {
        "neighbor_count": len(rows),
        "median_neighbor_total_r": statistics.median(total_r) if total_r else None,
        "mean_neighbor_total_r": statistics.fmean(total_r) if total_r else None,
        "worst_neighbor_total_r": min(total_r) if total_r else None,
        "median_neighbor_profit_factor": statistics.median(profit_factors) if profit_factors else None,
        "median_neighbor_max_drawdown_r": statistics.median(drawdowns) if drawdowns else None,
        "proportion_neighbors_profitable": sum(value > 0 for value in total_r) / len(total_r) if total_r else None,
        "proportion_neighbors_profit_factor_gt_1": (
            sum(value > 1 for value in profit_factors) / len(rows) if rows else None
        ),
        "trade_count_sensitivity": max(trades) - min(trades) if trades else None,
        "total_r_dispersion": max(total_r) - min(total_r) if total_r else None,
        "profit_factor_dispersion": max(profit_factors) - min(profit_factors) if profit_factors else None,
    }


def build_robustness_tables(
    result_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["config_id"]): row for row in result_rows}
    weight_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for row in result_rows:
        units = tuple(int(round(float(row[f"G{index}"]) / float(WEIGHT_UNIT))) for index in range(1, 6))
        threshold = Decimal(str(row["quality_threshold"]))
        weight_ids = [config_id(neighbor, threshold) for neighbor in weight_neighbors(units)]
        quality_ids = [config_id(units, neighbor) for neighbor in quality_neighbors(threshold)]
        missing = [identifier for identifier in weight_ids + quality_ids if identifier not in by_id]
        if missing:
            raise WeightQResearchError(f"neighbor configuration missing from complete grid: {missing[0]}")
        weight_stats = _neighbor_statistics([by_id[identifier] for identifier in weight_ids])
        quality_stats = _neighbor_statistics([by_id[identifier] for identifier in quality_ids])
        combined_ids = sorted(set(weight_ids + quality_ids))
        combined_stats = _neighbor_statistics([by_id[identifier] for identifier in combined_ids])
        key = {name: row[name] for name in ("config_id", "G1", "G2", "G3", "G4", "G5", "quality_threshold")}
        weight_rows.append({**key, "immediate_weight_neighbor_count": len(weight_ids), **weight_stats})
        quality_rows.append({**key, "adjacent_quality_neighbor_count": len(quality_ids), **quality_stats})
        combined_rows.append({
            **key,
            "immediate_weight_neighbor_count": len(weight_ids),
            "adjacent_quality_neighbor_count": len(quality_ids),
            "combined_neighbor_count": len(combined_ids),
            **combined_stats,
        })
    return weight_rows, quality_rows, combined_rows


def _descriptive_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_r = [float(row["total_r"]) for row in rows]
    trades = [int(row["trades"]) for row in rows]
    pfs = [value for row in rows if (value := _finite(row.get("profit_factor"))) is not None]
    drawdowns = [float(row["max_cumulative_drawdown_r"]) for row in rows]
    return {
        "configuration_count": len(rows),
        "median_trades": statistics.median(trades) if trades else None,
        "median_total_r": statistics.median(total_r) if total_r else None,
        "mean_total_r": statistics.fmean(total_r) if total_r else None,
        "p25_total_r": _percentile(total_r, 0.25),
        "p75_total_r": _percentile(total_r, 0.75),
        "median_profit_factor": statistics.median(pfs) if pfs else None,
        "median_max_drawdown_r": statistics.median(drawdowns) if drawdowns else None,
        "proportion_profitable": sum(value > 0 for value in total_r) / len(rows) if rows else None,
        "proportion_profit_factor_ge_1_20": sum(value >= 1.20 for value in pfs) / len(rows) if rows else None,
        "proportion_profit_factor_ge_1_30": sum(value >= 1.30 for value in pfs) / len(rows) if rows else None,
        "proportion_with_at_least_15_trades": sum(value >= MIN_RANKING_TRADES for value in trades) / len(rows) if rows else None,
    }


def quality_threshold_summary(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"quality_threshold": float(threshold), **_descriptive_summary([
            row for row in result_rows if Decimal(str(row["quality_threshold"])) == threshold
        ])}
        for threshold in QUALITY_THRESHOLDS
    ]


def weight_component_sensitivity(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for component in range(1, 6):
        field_name = f"G{component}"
        values = sorted({float(row[field_name]) for row in result_rows})
        for threshold_label, threshold in [("ALL", None), *[(str(item), item) for item in QUALITY_THRESHOLDS]]:
            population = result_rows if threshold is None else [
                row for row in result_rows if Decimal(str(row["quality_threshold"])) == threshold
            ]
            for value in values:
                rows = [row for row in population if math.isclose(float(row[field_name]), value, abs_tol=1e-12)]
                summary = _descriptive_summary(rows)
                output.append({
                    "component": field_name, "weight_value": value, "quality_threshold": threshold_label,
                    "count": summary["configuration_count"],
                    "median_total_r": summary["median_total_r"], "mean_total_r": summary["mean_total_r"],
                    "p25_total_r": summary["p25_total_r"], "p75_total_r": summary["p75_total_r"],
                    "median_profit_factor": summary["median_profit_factor"],
                    "median_max_drawdown_r": summary["median_max_drawdown_r"],
                    "median_trades": summary["median_trades"],
                    "proportion_profitable": summary["proportion_profitable"],
                })
    return output


def plateau_analysis(
    result_rows: Sequence[Mapping[str, Any]], combined_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["config_id"]): row for row in result_rows}
    robust = {str(row["config_id"]): row for row in combined_rows}
    eligible = {
        identifier for identifier, row in by_id.items()
        if int(row["trades"]) >= MIN_RANKING_TRADES
        and float(row["total_r"]) > 0
        and (_finite(row.get("profit_factor")) or 0.0) > 1.0
        and (robust[identifier]["proportion_neighbors_profitable"] or 0.0) > 0.5
    }
    components: list[list[str]] = []
    unseen = set(eligible)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component: list[str] = []
        stack = [seed]
        while stack:
            identifier = stack.pop()
            component.append(identifier)
            row = by_id[identifier]
            units = tuple(int(round(float(row[f"G{index}"]) / float(WEIGHT_UNIT))) for index in range(1, 6))
            adjacent = set(combined_neighbor_ids(units, Decimal(str(row["quality_threshold"])))) & unseen & eligible
            unseen.difference_update(adjacent)
            stack.extend(sorted(adjacent, reverse=True))
        components.append(sorted(component))
    components.sort(key=lambda item: (-len(item), item[0]))

    representatives: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    for component_id, members in enumerate(components, start=1):
        rows = [by_id[identifier] for identifier in members]
        coordinates = [
            tuple(float(row[f"G{index}"]) for index in range(1, 6)) + (float(row["quality_threshold"]),)
            for row in rows
        ]
        center = tuple(statistics.fmean(values) for values in zip(*coordinates))
        representative = min(
            rows,
            key=lambda row: (
                sum(abs(value - target) for value, target in zip(
                    tuple(float(row[f"G{index}"]) for index in range(1, 6)) + (float(row["quality_threshold"]),),
                    center,
                )),
                str(row["config_id"]),
            ),
        )
        record = {
            "plateau_id": f"PLATEAU-{component_id:04d}", "configuration_count": len(members),
            "member_config_ids": members,
            "median_total_r": _median(row["total_r"] for row in rows),
            "median_profit_factor": _median(row["profit_factor"] for row in rows),
            "median_max_drawdown_r": _median(row["max_cumulative_drawdown_r"] for row in rows),
            "representative_config_id": representative["config_id"],
        }
        component_records.append(record)
        representatives.append({
            "plateau_id": record["plateau_id"], "plateau_configuration_count": len(members),
            **dict(representative),
            "combined_median_neighbor_total_r": robust[str(representative["config_id"])]["median_neighbor_total_r"],
            "selection_status": "DESCRIPTIVE_REPRESENTATIVE_NOT_SELECTED",
        })

    ranked_population = [row for row in result_rows if int(row["trades"]) >= MIN_RANKING_TRADES]
    excellent_cutoff = _percentile([float(row["total_r"]) for row in ranked_population], 0.95)
    isolated = [
        {**dict(row), **{
            "combined_median_neighbor_total_r": robust[str(row["config_id"])]["median_neighbor_total_r"],
            "combined_proportion_neighbors_profitable": robust[str(row["config_id"])]["proportion_neighbors_profitable"],
            "classification": "DESCRIPTIVE_ISOLATED_PEAK_NOT_SELECTED",
        }}
        for row in ranked_population
        if excellent_cutoff is not None and float(row["total_r"]) >= excellent_cutoff
        and (
            (robust[str(row["config_id"])]["proportion_neighbors_profitable"] or 0.0) <= 0.5
            or (robust[str(row["config_id"])]["median_neighbor_total_r"] or 0.0) <= 0.0
        )
    ]
    isolated.sort(key=lambda row: (-float(row["total_r"]), str(row["config_id"])))

    descriptive_filters: dict[str, int] = {}
    for pf in (1.20, 1.30, 1.40):
        for drawdown in (10.0, 8.0, 6.0):
            key = f"pf_ge_{pf:.2f}_and_abs_drawdown_le_{drawdown:.0f}r"
            descriptive_filters[key] = sum(
                int(row["trades"]) >= MIN_RANKING_TRADES
                and (_finite(row.get("profit_factor")) or 0.0) >= pf
                and abs(float(row["max_cumulative_drawdown_r"])) <= drawdown
                for row in result_rows
            )
    payload = {
        "status": "DESCRIPTIVE_PLATEAU_ANALYSIS_COMPLETE_NO_SELECTION",
        "eligibility_rule": {
            "minimum_trades": MIN_RANKING_TRADES, "positive_total_r": True,
            "profit_factor_gt": 1.0, "majority_combined_neighbors_profitable": True,
        },
        "connected_plateau_count": len(component_records),
        "eligible_configuration_count": len(eligible),
        "components": component_records,
        "descriptive_filter_counts": descriptive_filters,
        "automatic_v5_selection": False,
        "selected_configuration": None,
    }
    return payload, representatives, isolated


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ranked_tables(
    result_rows: Sequence[Mapping[str, Any]], combined_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    robust = {str(row["config_id"]): row for row in combined_rows}
    population = [dict(row) for row in result_rows if int(row["trades"]) >= MIN_RANKING_TRADES]
    profitable = [row for row in population if float(row["total_r"]) > 0]
    top_total = sorted(population, key=lambda row: (-float(row["total_r"]), str(row["config_id"])))[:TOP_TABLE_LIMIT]
    top_pf = sorted(
        population, key=lambda row: (-(_finite(row.get("profit_factor")) or -math.inf), -float(row["total_r"]), str(row["config_id"])),
    )[:TOP_TABLE_LIMIT]
    top_dd = sorted(
        profitable,
        key=lambda row: (abs(float(row["max_cumulative_drawdown_r"])), -float(row["total_r"]), str(row["config_id"])),
    )[:TOP_TABLE_LIMIT]
    top_robustness = sorted(
        ({**row, **{
            "combined_median_neighbor_total_r": robust[str(row["config_id"])]["median_neighbor_total_r"],
            "combined_worst_neighbor_total_r": robust[str(row["config_id"])]["worst_neighbor_total_r"],
            "combined_proportion_neighbors_profitable": robust[str(row["config_id"])]["proportion_neighbors_profitable"],
        }} for row in population),
        key=lambda row: (
            -(row["combined_median_neighbor_total_r"] or -math.inf),
            -(row["combined_proportion_neighbors_profitable"] or 0.0),
            -float(row["total_r"]), str(row["config_id"]),
        ),
    )[:TOP_TABLE_LIMIT]
    return {
        "top-total-r.csv": top_total, "top-profit-factor.csv": top_pf,
        "top-drawdown.csv": top_dd, "top-robustness.csv": top_robustness,
    }


def _nearest_v3_grid(result_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    default = tuple(master.DEFAULT_WEIGHTS[name] for name in master.SCORE_FIELDS)
    distances: dict[tuple[int, ...], Decimal] = {}
    for units in generate_weight_grid():
        weights = tuple(Decimal(value) * WEIGHT_UNIT for value in units)
        distances[units] = sum(abs(value - baseline) for value, baseline in zip(weights, default))
    minimum = min(distances.values())
    nearest = {units for units, distance in distances.items() if distance == minimum}
    q50 = [
        row for row in result_rows
        if Decimal(str(row["quality_threshold"])) == Decimal("0.50")
        and tuple(int(round(float(row[f"G{index}"]) / float(WEIGHT_UNIT))) for index in range(1, 6)) in nearest
    ]
    return {
        "immutable_v3_weights": {name: str(value) for name, value in master.DEFAULT_WEIGHTS.items()},
        "v3_weights_are_on_grid": False,
        "minimum_l1_distance": str(minimum),
        "nearest_q50_grid_results": sorted(q50, key=lambda row: str(row["config_id"])),
    }


def _assert_v3_gate(result: Mapping[str, Any]) -> None:
    master._assert_metrics("V3_DECEMBER", result["metrics"], master.EXPECTED_GATES["V3_DECEMBER"])
    if result.get("dbn_files_opened") != 0 or result.get("network_calls") != 0:
        raise WeightQResearchError("V3 gate violated the offline-only contract")


def _assert_compact_v3(
    trades: Sequence[Mapping[str, Any]],
    exact: Mapping[str, Any],
    *,
    benchmark_label: str,
    expected_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile a compact V3 replay against one explicitly scoped benchmark."""
    normalized_label = benchmark_label.strip().upper()
    if not normalized_label:
        raise WeightQResearchError("compact V3 benchmark label is required")
    if not expected_metrics:
        raise WeightQResearchError("compact V3 expected metrics are required")
    metrics = historical._performance(list(trades))
    master._assert_metrics(f"COMPACT_V3_{normalized_label}", metrics, expected_metrics)
    exact_rows = sorted(exact["trades"], key=lambda row: str(row["trade_id"]))
    compact_rows = sorted(trades, key=lambda row: str(row["trade_id"]))
    if len(exact_rows) != len(compact_rows):
        raise WeightQResearchError("compact V3 trade count differs from canonical replay")
    fields = (
        "trade_id", "setup_id", "entry_timestamp_ns", "entry", "stop", "target",
        "exit_timestamp_ns", "exit", "exit_reason", "instrument", "contracts",
        "net_pnl_usd", "r_multiple",
    )
    for expected, actual in zip(exact_rows, compact_rows):
        for name in fields:
            left, right = expected.get(name), actual.get(name)
            if isinstance(left, float) or isinstance(right, float):
                if left is None or right is None or not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                    raise WeightQResearchError(f"compact V3 trade mismatch: {expected['trade_id']}/{name}")
            elif left != right:
                raise WeightQResearchError(f"compact V3 trade mismatch: {expected['trade_id']}/{name}")
    return metrics


def _load_filtered_parquet(path: Path, days: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("session_date", "in", list(days))])
    try:
        rows = table.to_pylist()
    finally:
        del table
    if any(str(row["session_date"]) not in days for row in rows):
        raise WeightQResearchError("Parquet predicate exposed a non-December row")
    return rows


def _matrix_report(summary: Mapping[str, Any]) -> str:
    gate = summary["v3_december_reproduction"]
    return "\n".join([
        "# December 2025 L2 weight x quality research", "",
        f"Status: `{summary['status']}`", "",
        f"Evidence label: `{EVIDENCE_LABEL}`. December 2025 is parameter research; January 2026 remains a strict internal holdout.", "",
        "No January strategy metric was calculated, no candidate or V5 rule was selected, and no DBN or network source was opened.", "",
        "## Frozen reproduction gate", "",
        f"V3 December reproduced {gate['completed_trades']} trades, {gate['wins']} wins, {gate['losses']} losses, "
        f"{gate['total_r']:.15f}R, ${gate['net_pnl_usd']:.2f}, PF {gate['profit_factor']:.15f}, "
        f"and {gate['max_cumulative_drawdown_r']:.15f}R maximum cumulative drawdown.", "",
        "## Matrix", "",
        f"The raw output retains all {summary['configuration_count']:,} configurations: "
        f"{summary['weight_count']:,} legal positive 0.05 weight vectors x six quality thresholds.", "",
        "Each session was indexed once, but each configuration maintained independent pending setup and one-position chronology. "
        "Ranked tables require at least 15 December trades.", "",
        "## Robustness", "",
        "Immediate weight neighbors transfer one 0.05 unit between components. Quality neighbors use adjacent predeclared thresholds. "
        "Connected plateaus require positive R, PF above one, at least 15 trades, and a majority of combined neighbors profitable.", "",
        "All plateaus, isolated peaks, rankings, and nearest-grid V3 comparisons are descriptive only.", "",
    ])


def _matrix_report_html(summary: Mapping[str, Any]) -> str:
    gate = summary["v3_december_reproduction"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>December 2025 L2 weight × quality research</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#18212b}}code{{background:#eef2f5;padding:.15rem .35rem;border-radius:4px}}.k{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.c{{border:1px solid #ccd5dd;border-radius:8px;padding:14px}}.n{{font-size:1.45rem;font-weight:700}}</style></head>
<body><h1>December 2025 L2 weight × quality research</h1><p>Status: <code>{summary['status']}</code></p>
<div class="k"><div class="c"><div class="n">{summary['weight_count']:,}</div>weights</div><div class="c"><div class="n">{summary['configuration_count']:,}</div>configurations</div><div class="c"><div class="n">{gate['completed_trades']}</div>V3 gate trades</div><div class="c"><div class="n">{gate['total_r']:.4f}R</div>V3 gate total</div></div>
<h2>Integrity</h2><p>The canonical V3 December result reproduced exactly before the matrix. The run opened zero DBNs, made zero network calls, and calculated no January strategy metrics.</p>
<h2>Interpretation</h2><p>December is research data. Rankings use a 15-trade minimum. Weight, quality, and combined-neighbor results distinguish connected plateaus from isolated peaks. No configuration or V5 rule is selected automatically.</p>
</body></html>"""


def _materialize_artifacts(
    root: Path, *, result_rows: list[dict[str, Any]], exact_v3: Mapping[str, Any],
    compact_v3_metrics: Mapping[str, Any], runtime: Mapping[str, Any], master_validation: Mapping[str, Any],
) -> dict[str, Any]:
    weight_rows = [
        {"weight_id": "W" + "-".join(f"{value:02d}" for value in units),
         **{f"G{index}": value * float(WEIGHT_UNIT) for index, value in enumerate(units, start=1)},
         "sum": 1.0}
        for units in generate_weight_grid()
    ]
    weight_robustness, quality_robustness, combined_robustness = build_robustness_tables(result_rows)
    q_summary = quality_threshold_summary(result_rows)
    component_summary = weight_component_sensitivity(result_rows)
    plateaus, representatives, isolated = plateau_analysis(result_rows, combined_robustness)
    ranked = _ranked_tables(result_rows, combined_robustness)
    nearest_v3 = _nearest_v3_grid(result_rows)

    _write_csv(root / "weight-grid.csv", weight_rows)
    _write_csv(root / "weight-q-results.csv", result_rows, RESULT_COLUMNS)
    _write_csv(root / "weight-neighbor-robustness.csv", weight_robustness)
    _write_csv(root / "quality-neighbor-robustness.csv", quality_robustness)
    _write_csv(root / "combined-neighbor-robustness.csv", combined_robustness)
    _write_csv(root / "quality-threshold-summary.csv", q_summary)
    _write_csv(root / "weight-component-sensitivity.csv", component_summary)
    for filename, rows in ranked.items():
        _write_csv(root / filename, rows)
    _write_csv(root / "plateau-representatives.csv", representatives)
    _write_csv(root / "top-isolated-peaks.csv", isolated)
    _write_json(root / "plateau-analysis.json", plateaus)

    gate = dict(exact_v3["metrics"])
    summary = {
        "status": "DECEMBER_WEIGHT_Q_RESEARCH_COMPLETE_NO_SELECTION",
        "strategy_id": STRATEGY_ID, "evidence_label": EVIDENCE_LABEL,
        "master_tape": str(master_validation["building_root"]),
        "master_validation": dict(master_validation),
        "research_period": {"start": "2025-12-01", "end_inclusive": "2025-12-31"},
        "january_internal_holdout": {
            "status": "UNTOUCHED_FOR_SELECTION", "strategy_metrics_calculated": False,
            "events_consulted_by_matrix": 0, "ranking_rows": 0,
        },
        "v3_december_reproduction": gate,
        "compact_engine_v3_reproduction": dict(compact_v3_metrics),
        "weight_count": EXPECTED_WEIGHT_COUNT,
        "quality_thresholds": [float(value) for value in QUALITY_THRESHOLDS],
        "configuration_count": EXPECTED_CONFIGURATION_COUNT,
        "minimum_ranking_trades": MIN_RANKING_TRADES,
        "raw_rows_retained": len(result_rows),
        "nearest_v3_grid": nearest_v3,
        "plateau_summary": {
            "eligible_configuration_count": plateaus["eligible_configuration_count"],
            "connected_plateau_count": plateaus["connected_plateau_count"],
            "automatic_v5_selection": False, "selected_configuration": None,
        },
        "runtime": dict(runtime),
        "dbn_files_opened": 0, "databento_calls": 0, "downloads": 0,
        "automatic_v5_selection": False, "strategy_parameters_changed": False,
    }
    _write_json(root / "summary.json", summary)
    (root / "diagnostic-report.md").write_text(_matrix_report(summary), encoding="utf-8")
    (root / "diagnostic-report.html").write_text(_matrix_report_html(summary), encoding="utf-8")
    return summary


def run_matrix(*, master_root: Path, output_root: Path) -> dict[str, Any]:
    run_started = time.monotonic()
    master_root = master_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable research output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists():
        raise FileExistsError(f"unverified matrix staging directory already exists: {staging}")

    registry = configuration_registry()
    # Full immutable-tape integrity plus the exact canonical replay are the
    # mandatory stop-before-matrix gates.
    master_validation = master.validate_building_root(master_root)
    exact_v3 = master.replay_configuration(master_root, threshold="0.50", session_prefix=DECEMBER_PREFIX)
    _assert_v3_gate(exact_v3)
    print("V3_DECEMBER_REPRODUCTION_GATE=PASS", flush=True)

    calendar = json.loads((master_root / "calendar.json").read_text(encoding="utf-8"))
    december_days = tuple(day for day in calendar["target_sessions"] if str(day).startswith(DECEMBER_PREFIX))
    if not december_days or any(not str(day).startswith(DECEMBER_PREFIX) for day in december_days):
        raise WeightQResearchError("December-only calendar isolation failed")
    interactions = _load_filtered_parquet(master_root / "interaction-master.parquet", december_days)
    index_rows = _load_filtered_parquet(master_root / "interaction-event-index.parquet", december_days)
    indexes = {str(row["interaction_id"]): row for row in index_rows}
    if len(indexes) != len(index_rows) or {str(row["interaction_id"]) for row in interactions} != set(indexes):
        raise WeightQResearchError("December interaction/index identities do not reconcile")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    for rows in by_day.values():
        rows.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["source_interaction_id"])))

    weight_grid = generate_weight_grid()
    weights_matrix = np.asarray(weight_grid, dtype=np.float64) * float(WEIGHT_UNIT)
    accumulators = [ConfigurationAccumulator(weights, threshold) for weights, threshold in registry]
    baseline_trade_ids = {str(row["trade_id"]) for row in exact_v3["trades"]}
    if len(baseline_trade_ids) != len(exact_v3["trades"]):
        raise WeightQResearchError("canonical V3 baseline contains duplicate trade identities")
    compact_v3_trades: list[dict[str, Any]] = []
    started = time.monotonic()
    tracemalloc.start()
    configuration_session_evaluations = 0
    next_progress = PROGRESS_INTERVAL

    for session_number, day in enumerate(december_days, start=1):
        print(f"MATRIX_SESSION {session_number:02d}/{len(december_days):02d} {day}", flush=True)
        day_rows = by_day.get(day, [])
        tape = SessionCausalTape.from_parquet(day, master_root / "causal-event-tape" / f"{day}.parquet")
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in day_rows}

        # Compact-engine identity check uses exact frozen V3 weights, not a
        # rounded grid approximation.
        v3_accepted = [
            row for row in day_rows if master.interaction_is_accepted(row, threshold="0.50")
        ]
        compact_v3_trades.extend(simulate_independent_session(tape, v3_accepted, day_indexes).trades)

        if day_rows:
            components = np.asarray([
                [float(row[name]) for name in master.SCORE_FIELDS] for row in day_rows
            ], dtype=np.float64)
            penalties = np.asarray([float(row["false_refill_penalty"]) for row in day_rows], dtype=np.float64)
            primitive_ok = np.asarray([not str(row.get("non_quality_rejection_reasons") or "") for row in day_rows])
            scores = np.clip(
                components @ weights_matrix.T - penalties[:, None] * float(master.V2_CONFIG.false_refill_penalty_weight),
                0.0, 1.0,
            )
        else:
            scores = np.empty((0, EXPECTED_WEIGHT_COUNT), dtype=np.float64)
            primitive_ok = np.empty((0,), dtype=np.bool_)

        for weight_index, units in enumerate(weight_grid):
            weight_scores = scores[:, weight_index]
            base_index = weight_index * len(QUALITY_THRESHOLDS)
            for q_index, threshold in enumerate(QUALITY_THRESHOLDS):
                accepted_mask = _accepted_mask(day_rows, primitive_ok, weight_scores, units, threshold)
                accepted = [row for row, keep in zip(day_rows, accepted_mask) if bool(keep)]
                session_result = simulate_independent_session(tape, accepted, day_indexes)
                accumulators[base_index + q_index].add(session_result, baseline_trade_ids)
                configuration_session_evaluations += 1
                equivalent = configuration_session_evaluations // len(december_days)
                if equivalent >= next_progress:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"MATRIX_PROGRESS equivalent_configs={equivalent:,}/{EXPECTED_CONFIGURATION_COUNT:,} "
                        f"config_session_evaluations={configuration_session_evaluations:,} "
                        f"rate={configuration_session_evaluations / elapsed:,.1f}_config_sessions_per_second",
                        flush=True,
                    )
                    next_progress += PROGRESS_INTERVAL
        del tape, scores

    compact_v3_metrics = _assert_compact_v3(
        compact_v3_trades,
        exact_v3,
        benchmark_label="DECEMBER",
        expected_metrics=master.EXPECTED_GATES["V3_DECEMBER"],
    )
    result_rows = [accumulator.row(len(baseline_trade_ids)) for accumulator in accumulators]
    if len(result_rows) != EXPECTED_CONFIGURATION_COUNT or len({row["config_id"] for row in result_rows}) != EXPECTED_CONFIGURATION_COUNT:
        raise WeightQResearchError("raw result does not retain exactly 23,256 unique configurations")
    if any(str(row["config_id"]).startswith("2026-01") for row in result_rows):
        raise WeightQResearchError("January identity leaked into matrix results")

    elapsed = time.monotonic() - started
    _current, peak = tracemalloc.get_traced_memory()
    runtime = {
        "matrix_simulation_seconds": elapsed,
        "configurations_per_second": EXPECTED_CONFIGURATION_COUNT / elapsed if elapsed else None,
        "configuration_session_evaluations": configuration_session_evaluations,
        "peak_python_tracemalloc_bytes": peak,
        "progress_interval_equivalent_configurations": PROGRESS_INTERVAL,
    }
    staging.mkdir(parents=True)
    summary = _materialize_artifacts(
        staging, result_rows=result_rows, exact_v3=exact_v3,
        compact_v3_metrics=compact_v3_metrics, runtime=runtime,
        master_validation=master_validation,
    )
    _current, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime["peak_python_tracemalloc_bytes"] = max(peak, final_peak)
    runtime["total_seconds_including_validation_and_reporting"] = time.monotonic() - run_started
    summary["runtime"] = runtime
    _write_json(staging / "summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_matrix_report(summary), encoding="utf-8")
    (staging / "diagnostic-report.html").write_text(_matrix_report_html(summary), encoding="utf-8")
    os.rename(staging, output_root)
    return {**summary, "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--master-root", type=Path, default=MASTER_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_matrix(master_root=args.master_root, output_root=args.output_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": result["status"], "output_root": result["output_root"],
        "weight_count": result["weight_count"], "configuration_count": result["configuration_count"],
        "january_internal_holdout": result["january_internal_holdout"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
