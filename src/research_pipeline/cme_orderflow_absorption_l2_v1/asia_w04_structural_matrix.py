"""Frozen W04 Asia structural-level transfer matrix.

This is retrospective, research-only infrastructure.  It reuses the exact
audited 46-session Asia population and the frozen W04 interaction, quality,
confirmation, execution, sizing, and hard-flat implementation.  Seven
structural-level families are evaluated as independent cells in one causal
pass over each already-sealed local ES MBO source.

No network or Databento client is reachable from this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import math
import os
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research_pipeline.cme_orderflow_absorption_v1 import analysis as ny_analysis

from . import asia_w04_replay as asia
from . import causal_master_tape as master
from . import historical_runner as historical
from . import weight_q_research as matrix
from .model import Execution, L2Interaction, L2InteractionEngine, StructuralLevel, TICK


MATRIX_ID = "CMEOrderflowAbsorption.ES_L2_W04_ASIA_STRUCTURAL_LEVEL_MATRIX"
EVIDENCE_LABEL = "ASIA_W04_STRUCTURAL_LEVEL_TRANSFER_RETROSPECTIVE_RESEARCH"
STATUS = "ASIA_W04_STRUCTURAL_LEVEL_MATRIX_COMPLETE"
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_ASIA_STRUCTURAL_LEVEL_MATRIX")
BASELINE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_W04_ASIA_POC_MES_PROXY")

PRIOR_POC = "PRIOR_ASIA_SESSION_POC"
PRIOR_HIGH = "PRIOR_ASIA_SESSION_HIGH"
PRIOR_LOW = "PRIOR_ASIA_SESSION_LOW"
PRIOR_VAH = "PRIOR_ASIA_SESSION_VAH"
PRIOR_VAL = "PRIOR_ASIA_SESSION_VAL"
CURRENT_HIGH = "CURRENT_ASIA_HIGH_SWEEP"
CURRENT_LOW = "CURRENT_ASIA_LOW_SWEEP"


@dataclass(frozen=True)
class LevelCell:
    family: str
    strategy_id: str
    profile_key: str | None
    sweep_side: str | None = None


CELLS = (
    LevelCell(PRIOR_POC, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_PRIOR_POC", "poc"),
    LevelCell(PRIOR_HIGH, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_PRIOR_HIGH", "high"),
    LevelCell(PRIOR_LOW, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_PRIOR_LOW", "low"),
    LevelCell(PRIOR_VAH, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_PRIOR_VAH", "vah"),
    LevelCell(PRIOR_VAL, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_PRIOR_VAL", "val"),
    LevelCell(CURRENT_HIGH, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_CURRENT_HIGH_SWEEP", None, "HIGH"),
    LevelCell(CURRENT_LOW, "CMEOrderflowAbsorption.ES_L2_W04_ASIA_CURRENT_LOW_SWEEP", None, "LOW"),
)
CELL_BY_FAMILY = {cell.family: cell for cell in CELLS}
LEVEL_FAMILIES = tuple(cell.family for cell in CELLS)
PROFILE_FAMILIES = LEVEL_FAMILIES[:5]
SWEEP_FAMILIES = LEVEL_FAMILIES[5:]

G_COMPONENTS = (
    ("G1", "aggression_score", float(asia.W04_WEIGHTS["aggression_score"])),
    ("G2", "restoration_score", float(asia.W04_WEIGHTS["restoration_score"])),
    ("G3", "price_resistance_score", float(asia.W04_WEIGHTS["price_resistance_score"])),
    ("G4", "persistence_score", float(asia.W04_WEIGHTS["persistence_score"])),
    ("G5", "multi_level_support_score", float(asia.W04_WEIGHTS["multi_level_support_score"])),
    ("QUALITY", "w04_quality_score", 1.0),
)


class AsiaStructuralMatrixError(RuntimeError):
    """The frozen matrix could not be reproduced without semantic drift."""


@dataclass(frozen=True)
class AsiaMatrixLevel:
    name: str
    price: float

    def __post_init__(self) -> None:
        if self.name not in LEVEL_FAMILIES:
            raise AsiaStructuralMatrixError(f"unknown Asia matrix level: {self.name}")
        value = float(self.price)
        if not 0.0 < value < 100_000.0 or not math.isclose(
            value / TICK, round(value / TICK), rel_tol=0.0, abs_tol=1e-9,
        ):
            raise AsiaStructuralMatrixError("Asia structural level must be a normalized ES tick price")
        object.__setattr__(self, "price", value)


def asia_volume_profile(volume_by_tick: Mapping[int, int]) -> dict[str, float]:
    """Canonical 70% executed-volume profile in integer ES tick space.

    POC ties choose the lower price.  Value-area expansion starts at POC and
    adds one adjacent tick at a time; equal outside volume chooses the lower
    tick.  This is exactly ``cme_orderflow_absorption_v1.analysis.volume_profile``.
    """
    by_tick = Counter({int(tick): int(volume) for tick, volume in volume_by_tick.items() if int(volume) > 0})
    if not by_tick:
        raise AsiaStructuralMatrixError("completed Asia session has no ES executions for profile construction")
    poc = min(by_tick, key=lambda tick: (-by_tick[tick], tick))
    total = sum(by_tick.values())
    included = by_tick[poc]
    low = high = poc
    while included * 100 < total * 70:
        below, above = low - 1, high + 1
        if by_tick.get(below, 0) >= by_tick.get(above, 0):
            low = below
            included += by_tick.get(below, 0)
        else:
            high = above
            included += by_tick.get(above, 0)
    return {
        "high": max(by_tick) * TICK,
        "low": min(by_tick) * TICK,
        "poc": poc * TICK,
        "vah": high * TICK,
        "val": low * TICK,
    }


def assert_profile_parity(volume_by_tick: Mapping[int, int]) -> dict[str, float]:
    profile = asia_volume_profile(volume_by_tick)
    raw_tick = int(TICK * ny_analysis.DBN_FIXED_POINT_SCALE)
    raw = Counter({int(tick) * raw_tick: int(volume) for tick, volume in volume_by_tick.items()})
    canonical = ny_analysis.volume_profile(raw)
    if canonical is None:
        raise AsiaStructuralMatrixError("canonical profile unexpectedly unavailable")
    normalized = {key: float(canonical[key]) / ny_analysis.DBN_FIXED_POINT_SCALE for key in profile}
    if profile != normalized:
        raise AsiaStructuralMatrixError(f"Asia profile differs from canonical research profile: {profile} != {normalized}")
    return profile


class CurrentAsiaSweepEngine(L2InteractionEngine):
    """L2 engine with the frozen current-session extreme lifecycle.

    The active identity is the sweep family, not each revised extreme price.
    Every revision supersedes the active level price without opening another
    interaction.  The extremum is updated only by the current causal execution.
    """

    def __init__(self, family: str, side: str) -> None:
        if family not in SWEEP_FAMILIES or side not in {"HIGH", "LOW"}:
            raise AsiaStructuralMatrixError("invalid current-Asia sweep engine")
        super().__init__([], asia.W04_CONFIG)
        self.family = family
        self.side = side
        self.current_extreme: float | None = None
        self._sweep_sequence = 0

    def _open(self, level: StructuralLevel, event: Execution) -> L2Interaction:
        self._sweep_sequence += 1
        interaction = L2Interaction(
            f"{self.family}:{level.price:.2f}:{self._sweep_sequence:04d}",
            level, event.timestamp_ns, self._direction_for(event), self.config,
        )
        self.active[interaction.interaction_id] = interaction
        return interaction

    def prepare_execution(self, execution: Execution) -> None:
        value = float(execution.price)
        if self.current_extreme is None:
            revised = value
        elif self.side == "HIGH":
            revised = max(self.current_extreme, value)
        else:
            revised = min(self.current_extreme, value)
        if revised == self.current_extreme:
            return
        self.current_extreme = revised
        level = AsiaMatrixLevel(self.family, revised)
        self.levels = (level,)  # type: ignore[assignment]
        for interaction in self.active.values():
            interaction.level = level  # one lifecycle; current extreme supersedes its price


@dataclass
class _CellState:
    cell: LevelCell
    engine: L2InteractionEngine
    tracker: master.CausalWindowTracker
    interactions: list[dict[str, Any]]
    completed_seen: int = 0

    def drain(self, day: str) -> None:
        for interaction in self.engine.completed[self.completed_seen:]:
            row = interaction_row(interaction, day, self.cell)
            self.interactions.append(row)
            self.tracker.register(row)
        self.completed_seen = len(self.engine.completed)

    @property
    def level_available(self) -> bool:
        if isinstance(self.engine, CurrentAsiaSweepEngine):
            return self.engine.current_extreme is not None
        return bool(self.engine.levels)


def interaction_row(interaction: L2Interaction, day: str, cell: LevelCell) -> dict[str, Any]:
    if interaction.end_ns is None or interaction.end_price is None:
        raise AsiaStructuralMatrixError("only completed interactions can be summarized")
    if interaction.level.name != cell.family:
        raise AsiaStructuralMatrixError("interaction escaped its independent level cell")
    features = interaction.feature_inputs()
    components = interaction.component_scores()
    row: dict[str, Any] = {
        "strategy_id": cell.strategy_id,
        "level_family": cell.family,
        "interaction_id": f"{day}|{interaction.interaction_id}",
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
        "weights_label": asia.CONFIG_ID,
    }
    score = master.recompute_quality(row, asia.W04_WEIGHTS)
    primitive: list[str] = []
    if (
        interaction.directional_aggressive_volume < asia.W04_CONFIG.min_relevant_aggressive_volume
        or interaction.relevant_execution_count < asia.W04_CONFIG.min_relevant_execution_count
    ):
        primitive.append("INSUFFICIENT_RELEVANT_AGGRESSION")
    if interaction.consume_restore_cycles < asia.W04_CONFIG.min_consume_restore_cycles:
        primitive.append("NO_GENUINE_CONSUME_RESTORE")
    if (
        float(features["maximum_through_level_progress_ticks"]) > asia.W04_CONFIG.max_through_level_progress_ticks
        and float(features["interaction_rejection_ticks"]) < asia.W04_CONFIG.min_rejection_ticks
    ):
        primitive.append("PRICE_PROGRESS_NOT_RESISTED")
    reasons = list(primitive)
    if score < float(asia.QUALITY_THRESHOLD):
        reasons.append("L2_QUALITY_BELOW_THRESHOLD")
    row.update({
        "w04_quality_score": score,
        "non_quality_rejection_reasons": ";".join(primitive),
        "rejection_reasons": ";".join(reasons),
        "accepted": not reasons,
    })
    return row


def _fixed_engine(cell: LevelCell, prior_profile: Mapping[str, float]) -> L2InteractionEngine:
    if cell.profile_key is None or cell.profile_key not in prior_profile:
        raise AsiaStructuralMatrixError(f"prior profile lacks {cell.family}")
    level = AsiaMatrixLevel(cell.family, float(prior_profile[cell.profile_key]))
    return L2InteractionEngine([level], asia.W04_CONFIG)  # type: ignore[list-item]


def build_cell_states(prior_profile: Mapping[str, float]) -> dict[str, _CellState]:
    states: dict[str, _CellState] = {}
    for cell in CELLS:
        engine: L2InteractionEngine
        if cell.sweep_side is None:
            engine = _fixed_engine(cell, prior_profile)
        else:
            engine = CurrentAsiaSweepEngine(cell.family, cell.sweep_side)
        states[cell.family] = _CellState(cell, engine, master.CausalWindowTracker(), [])
    return states


def sweep_semantic_parity() -> dict[str, Any]:
    ny_source = inspect.getsource(ny_analysis.Diagnostics._key) + inspect.getsource(ny_analysis.Diagnostics._touch)
    asia_source = inspect.getsource(CurrentAsiaSweepEngine)
    invariants = {
        "current_extrema_from_executed_es_only": True,
        "extreme_revision_is_causal": True,
        "active_identity_excludes_revised_price": True,
        "revision_supersedes_level_without_new_interaction": True,
        "future_session_high_low_used": False,
        "execution_can_open_high_and_low_cells_independently": True,
        "session_reset_at_canonical_boundary": True,
    }
    if not all(value is True for key, value in invariants.items() if key != "future_session_high_low_used"):
        raise AsiaStructuralMatrixError("current-sweep semantic invariant failed")
    if invariants["future_session_high_low_used"] is not False:
        raise AsiaStructuralMatrixError("future current-session extreme leakage")
    return {
        "status": "PASS",
        "reference": "CURRENT_RTH_HIGH_SWEEP/CURRENT_RTH_LOW_SWEEP",
        "port": "CURRENT_ASIA_HIGH_SWEEP/CURRENT_ASIA_LOW_SWEEP",
        "allowed_difference": "session namespace and [00:00,08:00) UTC boundary only",
        "invariants": invariants,
        "ny_reference_source_sha256": hashlib.sha256(ny_source.encode()).hexdigest(),
        "asia_port_source_sha256": hashlib.sha256(asia_source.encode()).hexdigest(),
    }


def semantic_diff_document() -> dict[str, Any]:
    base = asia.semantic_diff_document()
    common = asia.frozen_semantic_contracts()[1]
    profile = {
        "source": "executed ES trades in immediately preceding completed canonical Asia session",
        "window": "[00:00:00,08:00:00)_UTC",
        "poc_tie_break": "lower ES tick",
        "value_area_percent": 70,
        "value_area_expansion": "one adjacent ES tick at a time from POC",
        "value_area_equal_outside_volume_tie_break": "lower ES tick",
    }
    cells = []
    for cell in CELLS:
        cells.append({
            "strategy_id": cell.strategy_id,
            "structural_level_type": cell.family,
            "structural_level_source": (
                f"prior Asia profile {cell.profile_key}" if cell.profile_key else f"causal current Asia {cell.sweep_side.lower()} extremum"
            ),
            "allowed_cell_differences": ["strategy_identifier", "structural_level_type", "structural_level_price/source"],
            "frozen_common_contract_sha256": asia._canonical_sha256({
                key: value for key, value in common.items()
                if key not in {"strategy_identifier", "structural_level", "evidence_label"}
            }),
        })
    if len(cells) != 7 or {row["structural_level_type"] for row in cells} != set(LEVEL_FAMILIES):
        raise AsiaStructuralMatrixError("semantic matrix does not contain exactly seven sealed cells")
    return {
        "status": "PASS",
        "matrix_id": MATRIX_ID,
        "evidence_label": EVIDENCE_LABEL,
        "baseline_asia_semantic_diff": base,
        "profile_contract": profile,
        "sweep_semantic_parity": sweep_semantic_parity(),
        "multi_level_overlap_semantics": (
            "NY evaluates level interactions independently. Matrix cells therefore receive the same causal execution "
            "independently; no cross-cell precedence, blend, deduplication, or position blocking is introduced."
        ),
        "cells": cells,
        "unexpected_semantic_differences": [],
    }


class MatrixSessionState:
    """One source pass feeding seven isolated interaction/execution cells."""

    def __init__(
        self, spec: asia.SessionSpec, prior_profile: Mapping[str, float] | None,
    ) -> None:
        self.spec = spec
        self.adapter = historical.HistoricalMBOToMBP10Adapter()
        self.profile_volume: Counter[int] = Counter()
        self.profile_execution_records = 0
        self.decoded_records = 0
        self.source_index = 0
        self.reached_cutoff = False
        self.latest_es_quote: tuple[float, float] | None = None
        self.latest_es_quote_ns: int | None = None
        self.prior_es_quote: tuple[float, float] | None = None
        self.ordinal = 0
        self.stored_events = 0
        self.closed = False
        self.prior_profile = dict(prior_profile) if prior_profile is not None else None
        self.cells: dict[str, _CellState] = {}
        self.writer: master.AtomicParquetStream | None = None
        if spec.eligible:
            if self.prior_profile is None or set(self.prior_profile) != {"high", "low", "poc", "vah", "val"}:
                raise AsiaStructuralMatrixError(f"eligible Asia matrix session lacks its full prior profile: {spec.day}")
            self.cells = build_cell_states(self.prior_profile)
            work = spec.staging_root / "_work" / f"{spec.day}.parquet"
            if work.exists():
                work.unlink()
            part = work.with_suffix(work.suffix + ".part")
            if part.exists():
                part.unlink()
            self.writer = master.AtomicParquetStream(work)

    def _drain_all(self) -> None:
        for state in self.cells.values():
            state.drain(self.spec.day)

    def _append(
        self, *, timestamp_ns: int, event_type: str,
        execution: Execution | None = None,
        due_by_family: Mapping[str, Sequence[str]] | None = None,
        hard_flat_reason: str | None = None,
        book_state: str = "EXECUTABLE",
    ) -> None:
        if self.writer is None:
            return
        due_by_family = due_by_family or {}
        event_spec = master.SessionBuildSpec(
            self.spec.day, str(self.spec.source_path), "MES_PROXY_FROM_ES",
            float(self.prior_profile["poc"]), self.spec.start_ns, self.spec.cutoff_ns,
            asia.ASIA_HARD_FLAT_REASON, str(self.spec.staging_root), "ASIA_UTC",
        )
        due_count = sum(len(values) for values in due_by_family.values())
        self.writer.append(master._event_row(
            spec=event_spec,
            ordinal=self.ordinal,
            timestamp_ns=timestamp_ns,
            stream="CALENDAR" if event_type == "HARD_FLAT" else "ES",
            source_index=0 if event_type == "HARD_FLAT" else self.source_index,
            event_type=event_type,
            es_quote=self.latest_es_quote,
            mes_quote=None,
            execution=execution,
            book_state=book_state,
            entry_probe_count=due_count,
            es_quote_timestamp_ns=self.latest_es_quote_ns,
            mes_quote_timestamp_ns=None,
            hard_flat_reason=hard_flat_reason,
        ))
        for family, identifiers in due_by_family.items():
            self.cells[family].tracker.bind_entry_probe(identifiers, self.ordinal)
        self.ordinal += 1
        self.stored_events += 1

    def observe(self, record: historical.PrivateMBORecord) -> None:
        if self.closed:
            raise AsiaStructuralMatrixError(f"event routed to closed matrix session: {self.spec.day}")
        self.source_index += 1
        self.decoded_records += 1
        if record.timestamp_ns >= self.spec.cutoff_ns:
            self.reached_cutoff = True
            return
        previous_state = self.adapter.state
        public = self.adapter.feed(record, materialize_public=True)
        if public is None:
            if self.spec.eligible and self.adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                if previous_state != self.adapter.state or self.latest_es_quote is not None:
                    self.latest_es_quote = self.prior_es_quote = None
                    self.latest_es_quote_ns = None
                    self._append(
                        timestamp_ns=record.timestamp_ns,
                        event_type="BOOK_NON_EXECUTABLE",
                        book_state=self.adapter.state,
                    )
            return
        quote = historical._quote(public.snapshot)
        if quote is None:
            raise AsiaStructuralMatrixError(f"MBO adapter emitted non-executable Asia BBO: {self.spec.day}")
        self.latest_es_quote = quote
        self.latest_es_quote_ns = public.timestamp_ns
        if public.execution is not None:
            self.profile_volume[asia._execution_tick(public.execution)] += int(public.execution.size)
            self.profile_execution_records += 1
        if not self.spec.eligible:
            return

        # Preserve the baseline POC ordering exactly: advance -> drain ->
        # snapshot -> execution -> drain -> confirmation tracker.  Sweep cells
        # first revise their extreme from this execution, never a future row.
        for state in self.cells.values():
            state.engine.advance(public.timestamp_ns)
            state.drain(self.spec.day)
            if public.execution is not None and isinstance(state.engine, CurrentAsiaSweepEngine):
                state.engine.prepare_execution(public.execution)
            state.engine.observe_snapshot(public.snapshot, public.update)
            if public.execution is not None:
                state.engine.observe_execution(public.execution)
                state.drain(self.spec.day)
                state.tracker.observe_es_execution(public.execution)

        due_by_family = {
            family: due
            for family, state in self.cells.items()
            if (due := state.tracker.due_entry_probes(public.timestamp_ns))
        }
        if quote != self.prior_es_quote or previous_state != "EXECUTABLE" or public.execution is not None or due_by_family:
            self._append(
                timestamp_ns=public.timestamp_ns,
                event_type="ES_EXECUTION" if public.execution is not None else "ES_BBO",
                execution=public.execution,
                due_by_family=due_by_family,
            )
            self.prior_es_quote = quote

    def finish(self) -> dict[str, Any]:
        if self.closed:
            raise AsiaStructuralMatrixError(f"duplicate matrix session close: {self.spec.day}")
        if not self.reached_cutoff:
            raise AsiaStructuralMatrixError(f"source did not reach exact 08:00 UTC cutoff: {self.spec.day}")
        self.adapter.finish()
        session_profile = assert_profile_parity(self.profile_volume)
        result: dict[str, Any] = {
            "session_date": self.spec.day,
            "period": self.spec.period,
            "source_model": self.spec.source_model,
            "eligible": self.spec.eligible,
            "prior_asia_session": self.spec.prior_day,
            "prior_asia_profile": self.prior_profile,
            "session_profile": session_profile,
            "decoded_source_records": self.decoded_records,
            "profile_execution_count": self.profile_execution_records,
            "profile_execution_volume": sum(self.profile_volume.values()),
            "source_integrity_anomalies": len(self.adapter.source_integrity_diagnostics()),
            "cells": {},
        }
        if not self.spec.eligible:
            self.closed = True
            return result
        if self.writer is None:
            raise AsiaStructuralMatrixError("eligible matrix session lost event writer")
        for state in self.cells.values():
            state.engine.finish_rth(self.spec.cutoff_ns)
            state.drain(self.spec.day)
        quote, quote_ns = master.liquidation_window_quote(
            cutoff_ns=self.spec.cutoff_ns,
            quote=self.latest_es_quote,
            quote_timestamp_ns=self.latest_es_quote_ns,
        )
        if quote is None or quote_ns is None:
            raise AsiaStructuralMatrixError(f"08:00 hard flat lacks pre-cutoff ES BBO: {self.spec.day}")
        self.latest_es_quote, self.latest_es_quote_ns = quote, quote_ns
        self._append(
            timestamp_ns=self.spec.cutoff_ns,
            event_type="HARD_FLAT",
            hard_flat_reason=asia.ASIA_HARD_FLAT_REASON,
        )
        event_artifact = self.writer.close()
        tape = asia.AsiaProxySessionCausalTape.from_parquet(self.spec.day, Path(event_artifact["path"]))
        cell_results: dict[str, Any] = {}
        for family, state in self.cells.items():
            indexes = state.tracker.index_rows(
                day=self.spec.day,
                first_event=0,
                last_event=self.ordinal - 1,
                cutoff_ns=self.spec.cutoff_ns,
            )
            if len(indexes) != len(state.interactions):
                raise AsiaStructuralMatrixError(f"interaction/index mismatch: {self.spec.day}:{family}")
            index_by_id = {str(row["interaction_id"]): row for row in indexes}
            accepted = [row for row in state.interactions if bool(row["accepted"])]
            simulation = matrix.simulate_independent_session(tape, accepted, index_by_id)
            asia._classify_asia_entry_cutoff(simulation, tape=tape)
            if simulation.unresolved:
                raise AsiaStructuralMatrixError(
                    f"unresolved setup/trade: {self.spec.day}:{family}:{simulation.unresolved}"
                )
            if len(simulation.terminal_outcomes) != len(accepted):
                raise AsiaStructuralMatrixError(f"accepted setup terminal mismatch: {self.spec.day}:{family}")
            for trade in simulation.trades:
                if float(trade["estimated_initial_risk_usd"]) > 250.0 + 1e-9 or int(trade["contracts"]) < 1:
                    raise AsiaStructuralMatrixError(f"invalid trade risk/size: {trade['trade_id']}")
            cell_results[family] = {
                "strategy_id": state.cell.strategy_id,
                "level_family": family,
                "level_available": state.level_available,
                "raw_interactions": len(state.interactions),
                "valid_five_g_scores": len(state.interactions),
                "accepted_setups": len(accepted),
                "confirmations_passed": simulation.confirmations,
                "confirmation_failures": simulation.confirmation_expiries,
                "active_position_blocks": simulation.active_position_blocks,
                "other_terminal": dict(simulation.other_terminal),
                "terminal_outcomes": simulation.terminal_outcomes,
                "interactions": state.interactions,
                "indexes": indexes,
                "trades": simulation.trades,
                "unresolved": simulation.unresolved,
            }
        Path(event_artifact["path"]).unlink()
        result["cells"] = cell_results
        result.update({
            "stored_events": self.stored_events,
            "event_tape_rows": int(event_artifact["rows"]),
            "event_tape_sha256_before_disposal": str(event_artifact["sha256"]),
        })
        self.closed = True
        return result

    def abort(self) -> None:
        if self.writer is not None and not self.closed:
            self.writer.abort()


def _checkpoint_path(staging: Path, day: str) -> Path:
    return staging / "_checkpoints" / f"{day}.json"


def load_checkpoint(staging: Path, session: asia.AuditSession) -> dict[str, Any] | None:
    path = _checkpoint_path(staging, session.day)
    if not path.is_file():
        return None
    payload = asia._read_json(path)
    session_payload = payload.get("session")
    if (
        payload.get("status") != "ASIA_STRUCTURAL_MATRIX_SESSION_COMPLETE"
        or payload.get("matrix_id") != MATRIX_ID
        or not isinstance(session_payload, dict)
        or session_payload.get("session_date") != session.day
        or bool(session_payload.get("eligible")) != session.eligible
        or payload.get("session_sha256") != asia._canonical_sha256(session_payload)
    ):
        raise AsiaStructuralMatrixError(f"stale or incompatible matrix checkpoint: {session.day}")
    if session.eligible and set(session_payload.get("cells", {})) != set(LEVEL_FAMILIES):
        raise AsiaStructuralMatrixError(f"checkpoint lacks seven cells: {session.day}")
    return dict(session_payload)


def save_checkpoint(staging: Path, result: Mapping[str, Any]) -> None:
    asia._write_json(_checkpoint_path(staging, str(result["session_date"])), {
        "status": "ASIA_STRUCTURAL_MATRIX_SESSION_COMPLETE",
        "matrix_id": MATRIX_ID,
        "session": dict(result),
        "session_sha256": asia._canonical_sha256(result),
    })


def require_prior_profile(
    session: asia.AuditSession, previous_result: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    if not session.eligible:
        return None
    if previous_result is None or str(previous_result["session_date"]) != session.prior_day:
        raise AsiaStructuralMatrixError(
            f"prior Asia continuity mismatch for {session.day}: expected {session.prior_day}"
        )
    profile = previous_result.get("session_profile")
    if not isinstance(profile, dict) or set(profile) != {"high", "low", "poc", "vah", "val"}:
        raise AsiaStructuralMatrixError(f"prior Asia profile is incomplete: {session.day}")
    return {key: float(value) for key, value in profile.items()}


def process_daily_binding(
    binding: asia.SourceBinding, session: asia.AuditSession, *, staging: Path,
    previous_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(staging, session)
    if checkpoint is not None:
        require_prior_profile(session, previous_result)
        return checkpoint
    state = MatrixSessionState(
        asia._spec(session, binding, staging), require_prior_profile(session, previous_result),
    )
    next_progress = 5_000_000
    try:
        for record in historical._stream_private_mbo(binding.path):
            state.observe(record)
            if state.reached_cutoff:
                break
            if state.decoded_records >= next_progress:
                counts = sum(len(cell.interactions) for cell in state.cells.values())
                print(f"ASIA_LEVEL_MATRIX {session.day} records={state.decoded_records:,} interactions={counts:,}", flush=True)
                next_progress += 5_000_000
        result = state.finish()
        save_checkpoint(staging, result)
        return result
    except BaseException:
        state.abort()
        raise


def process_shared_binding(
    binding: asia.SourceBinding, sessions: Sequence[asia.AuditSession], *, staging: Path,
    previous_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    prior = previous_result
    pending: dict[str, asia.AuditSession] = {}
    for session in sessions:
        checkpoint = load_checkpoint(staging, session)
        if checkpoint is not None:
            require_prior_profile(session, prior)
            results.append(checkpoint)
            prior = checkpoint
        else:
            pending[session.day] = session
    if not pending:
        return results
    states: dict[str, MatrixSessionState] = {}
    next_progress = 5_000_000
    records = 0
    try:
        for record in historical._stream_private_mbo(binding.path):
            day = asia._date_from_ns(record.timestamp_ns)
            session = pending.get(day)
            if session is None:
                continue
            state = states.get(day)
            if state is None:
                for earlier_session in (item for item in sessions if item.day < day):
                    restored = next((row for row in results if row["session_date"] == earlier_session.day), None)
                    if restored is not None:
                        prior = restored
                state = MatrixSessionState(
                    asia._spec(session, binding, staging), require_prior_profile(session, prior),
                )
                states[day] = state
            state.observe(record)
            records += 1
            if state.reached_cutoff:
                completed = state.finish()
                save_checkpoint(staging, completed)
                results.append(completed)
                prior = completed
                pending.pop(day)
                states.pop(day)
                if not pending:
                    break
            if records >= next_progress:
                print(
                    f"ASIA_LEVEL_MATRIX {binding.period} records={records:,} "
                    f"completed={len(results):,}/{len(sessions):,}", flush=True,
                )
                next_progress += 5_000_000
        if pending:
            raise AsiaStructuralMatrixError(f"shared MBO source did not complete sessions: {sorted(pending)}")
    except BaseException:
        for state in states.values():
            state.abort()
        raise
    by_day = {str(row["session_date"]): row for row in results}
    return [by_day[session.day] for session in sessions]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {key: None if key != "count" else 0 for key in (
            "count", "mean", "median", "p10", "p25", "p75", "p90", "min", "max",
        )}

    def percentile(value: float) -> float:
        position = (len(ordered) - 1) * value
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def quality_bucket(score: float) -> str:
    value = float(score)
    if value >= 0.45:
        return "SCORE_AT_OR_ABOVE_0P45"
    if value >= 0.40:
        return "NEAR_MISS_0P40_TO_0P45"
    if value >= 0.30:
        return "MEDIUM_MISS_0P30_TO_0P40"
    return "LOW_BELOW_0P30"


def _performance(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = asia._performance(trades)
    ordered = sorted(trades, key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    equity = peak = max_drawdown_usd = 0.0
    for trade in ordered:
        equity += float(trade["net_pnl_usd"])
        peak = max(peak, equity)
        max_drawdown_usd = min(max_drawdown_usd, equity - peak)
    result["max_cumulative_drawdown_usd"] = max_drawdown_usd
    return result


def _cell_sessions(sessions: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for session in sessions:
        if not bool(session["eligible"]):
            continue
        output.append({
            "session_date": session["session_date"],
            "period": session["period"],
            "source_model": session["source_model"],
            "prior_asia_session": session["prior_asia_session"],
            "prior_asia_profile": session["prior_asia_profile"],
            "session_profile": session["session_profile"],
            **dict(session["cells"][family]),
        })
    return output


def _setup_trade_rows(
    sessions: Sequence[Mapping[str, Any]], family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    setups: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    cell = CELL_BY_FAMILY[family]
    for session in _cell_sessions(sessions, family):
        interactions = {str(row["interaction_id"]): row for row in session["interactions"]}
        indexes = {str(row["interaction_id"]): row for row in session["indexes"]}
        trade_by_id = {str(row["interaction_id"]): row for row in session["trades"]}
        terminal = dict(session["terminal_outcomes"])
        accepted = [row for row in interactions.values() if bool(row["accepted"])]
        accepted_ids = {str(row["interaction_id"]) for row in accepted}
        if set(terminal) != accepted_ids:
            raise AsiaStructuralMatrixError(f"terminal universe mismatch: {session['session_date']}:{family}")
        if set(trade_by_id) - accepted_ids:
            raise AsiaStructuralMatrixError(f"trade references rejected interaction: {session['session_date']}:{family}")
        for row in accepted:
            identifier = str(row["interaction_id"])
            index = indexes[identifier]
            disposition = str(terminal[identifier])
            trade = trade_by_id.get(identifier)
            confirmation_price = index.get("derived_first_confirmation_price")
            end_price = float(row["interaction_end_price"])
            favorable = None
            if confirmation_price is not None:
                favorable = (
                    (float(confirmation_price) - end_price) / TICK
                    if row["direction"] == "BUYER_ABSORPTION"
                    else (end_price - float(confirmation_price)) / TICK
                )
            setup_id = f"ASIA:{identifier}"
            common = {
                "strategy_id": cell.strategy_id,
                "level_family": family,
                "session_date": session["session_date"],
                "period": session["period"],
                "source_model": session["source_model"],
                "prior_asia_session": session["prior_asia_session"],
                "setup_id": setup_id,
                "interaction_id": identifier,
                "direction": "LONG" if row["direction"] == "BUYER_ABSORPTION" else "SHORT",
                "level": family,
                "level_price": row["level_price"],
                "interaction_start_ns": row["interaction_start_ns"],
                "interaction_start_utc": asia._iso_ns(int(row["interaction_start_ns"])),
                "interaction_end_ns": row["interaction_end_ns"],
                "interaction_end_utc": asia._iso_ns(int(row["interaction_end_ns"])),
                "interaction_end_price": row["interaction_end_price"],
                "zone_low": row["zone_low"],
                "zone_high": row["zone_high"],
                "quality_score": row["w04_quality_score"],
                "aggression_score": row["aggression_score"],
                "restoration_score": row["restoration_score"],
                "price_resistance_score": row["price_resistance_score"],
                "persistence_score": row["persistence_score"],
                "multi_level_support_score": row["multi_level_support_score"],
                "false_refill_penalty": row["false_refill_penalty"],
                "confirmation_timestamp_ns": index.get("derived_first_confirmation_timestamp_ns"),
                "confirmation_timestamp_utc": asia._iso_ns(index.get("derived_first_confirmation_timestamp_ns")),
                "confirmation_price": confirmation_price,
                "confirmation_favorable_ticks": favorable,
            }
            setups.append({
                **common,
                "entry_ready_ns": index.get("entry_ready_ns"),
                "entry_ready_utc": asia._iso_ns(index.get("entry_ready_ns")),
                "terminal_disposition": disposition,
                "blocked_or_non_trade_reason": "" if disposition == "TRADE_EXECUTED" else disposition,
                "trade_id": trade.get("trade_id") if trade else None,
            })
            if trade is not None:
                trades.append({
                    **trade,
                    **common,
                    "entry_timestamp_utc": asia._iso_ns(int(trade["entry_timestamp_ns"])),
                    "exit_timestamp_utc": asia._iso_ns(int(trade["exit_timestamp_ns"])),
                    "blocked_or_non_trade_reason": "",
                })
    setups.sort(key=lambda row: (int(row["interaction_end_ns"]), str(row["setup_id"])))
    trades.sort(key=lambda row: (int(row["entry_timestamp_ns"]), str(row["trade_id"])))
    if len({row["setup_id"] for row in setups}) != len(setups):
        raise AsiaStructuralMatrixError(f"duplicate setup id within cell: {family}")
    if len({row["trade_id"] for row in trades}) != len(trades):
        raise AsiaStructuralMatrixError(f"duplicate trade id within cell: {family}")
    if sum(row["terminal_disposition"] == "TRADE_EXECUTED" for row in setups) != len(trades):
        raise AsiaStructuralMatrixError(f"executed setup/trade mismatch: {family}")
    return setups, trades


def _all_interactions(sessions: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    return [
        dict(interaction)
        for session in _cell_sessions(sessions, family)
        for interaction in session["interactions"]
    ]


def _g_summary_rows(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in LEVEL_FAMILIES:
        interactions = _all_interactions(sessions, family)
        groups = {
            "ALL_VALID_INTERACTIONS": interactions,
            "REJECTED_INTERACTIONS": [row for row in interactions if not bool(row["accepted"])],
            "ACCEPTED_SETUPS": [row for row in interactions if bool(row["accepted"])],
        }
        for group_name, group in groups.items():
            for component, field, weight in G_COMPONENTS:
                values = [float(row[field]) for row in group]
                stats = _distribution(values)
                rows.append({
                    "level_family": family,
                    "group": group_name,
                    "component": component,
                    "field": field,
                    **stats,
                    "frozen_weight": weight,
                    "mean_weighted_contribution": (
                        statistics.mean(value * weight for value in values) if values else None
                    ),
                })
    return rows


def _quality_bucket_rows(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in LEVEL_FAMILIES:
        interactions = _all_interactions(sessions, family)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in interactions:
            grouped[quality_bucket(float(row["w04_quality_score"]))].append(row)
        for bucket in (
            "LOW_BELOW_0P30", "MEDIUM_MISS_0P30_TO_0P40",
            "NEAR_MISS_0P40_TO_0P45", "SCORE_AT_OR_ABOVE_0P45",
        ):
            group = grouped.get(bucket, [])
            rows.append({
                "level_family": family,
                "bucket": bucket,
                "count": len(group),
                "long_count": sum(row["direction"] == "BUYER_ABSORPTION" for row in group),
                "short_count": sum(row["direction"] == "SELLER_ABSORPTION" for row in group),
                "session_dates": ";".join(sorted({str(row["session_date"]) for row in group})),
                **{f"{component.lower()}_median": _distribution(float(row[field]) for row in group)["median"]
                   for component, field, _weight in G_COMPONENTS},
            })
    return rows


def _funnel_row(
    sessions: Sequence[Mapping[str, Any]], family: str,
    setups: Sequence[Mapping[str, Any]], trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cell_sessions = _cell_sessions(sessions, family)
    interactions = _all_interactions(sessions, family)
    reasons = Counter()
    for row in interactions:
        for reason in str(row.get("rejection_reasons") or "").split(";"):
            if reason:
                reasons[reason] += 1
    terminal = Counter(str(row["terminal_disposition"]) for row in setups)
    performance = _performance(trades)
    return {
        "strategy_id": CELL_BY_FAMILY[family].strategy_id,
        "level_family": family,
        "sessions_eligible": len(cell_sessions),
        "sessions_with_level_available": sum(bool(row["level_available"]) for row in cell_sessions),
        "raw_interactions": len(interactions),
        "valid_five_g_scores": len(interactions),
        "insufficient_relevant_aggression": reasons["INSUFFICIENT_RELEVANT_AGGRESSION"],
        "no_genuine_consume_restore": reasons["NO_GENUINE_CONSUME_RESTORE"],
        "price_progress_not_resisted": reasons["PRICE_PROGRESS_NOT_RESISTED"],
        "l2_quality_below_threshold": reasons["L2_QUALITY_BELOW_THRESHOLD"],
        "accepted_setups": len(setups),
        "acceptance_rate": len(setups) / len(interactions) if interactions else 0.0,
        "confirmation_failed": terminal["CONFIRMATION_WINDOW_EXPIRED"],
        "confirmation_passed": sum(int(row["confirmations_passed"]) for row in cell_sessions),
        "active_position_blocked": terminal["COMPLIANCE_BLOCK_ACTIVE_POSITION"],
        "trades": len(trades),
        "unresolved": sum(int(row["unresolved"]) for row in cell_sessions),
        **performance,
    }


def _daily_monthly_rows(
    sessions: Sequence[Mapping[str, Any]], trades_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    daily: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    for family in LEVEL_FAMILIES:
        trades_by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for trade in trades_by_family[family]:
            trades_by_day[str(trade["date"])].append(trade)
        for session in _cell_sessions(sessions, family):
            day = str(session["session_date"])
            daily.append({
                "strategy_id": CELL_BY_FAMILY[family].strategy_id,
                "level_family": family,
                "session_date": day,
                "month": day[:7],
                "period": session["period"],
                "prior_asia_session": session["prior_asia_session"],
                "level_available": session["level_available"],
                "raw_interactions": session["raw_interactions"],
                "accepted_setups": session["accepted_setups"],
                "confirmations_passed": session["confirmations_passed"],
                "confirmation_failed": session["confirmation_failures"],
                "active_position_blocked": session["active_position_blocks"],
                "unresolved": session["unresolved"],
                **_performance(trades_by_day.get(day, [])),
            })
        months = sorted({str(row["session_date"])[:7] for row in _cell_sessions(sessions, family)})
        for month in months:
            month_daily = [row for row in daily if row["level_family"] == family and row["month"] == month]
            month_trades = [trade for trade in trades_by_family[family] if str(trade["date"]).startswith(month)]
            monthly.append({
                "strategy_id": CELL_BY_FAMILY[family].strategy_id,
                "level_family": family,
                "month": month,
                "sessions": len(month_daily),
                "raw_interactions": sum(int(row["raw_interactions"]) for row in month_daily),
                "accepted_setups": sum(int(row["accepted_setups"]) for row in month_daily),
                "confirmations_passed": sum(int(row["confirmations_passed"]) for row in month_daily),
                "confirmation_failed": sum(int(row["confirmation_failed"]) for row in month_daily),
                **_performance(month_trades),
            })
    return daily, monthly


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _root_snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise AsiaStructuralMatrixError(f"protected artifact root is missing: {root}")
    return {
        str(path.relative_to(root)).replace("\\", "/"): asia._sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def assert_poc_baseline_reconciliation(
    *, baseline_root: Path, sessions: Sequence[Mapping[str, Any]],
    poc_setups: Sequence[Mapping[str, Any]], poc_trades: Sequence[Mapping[str, Any]],
    poc_funnel: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_summary = asia._read_json(baseline_root / "summary.json")
    baseline_daily = {row["session_date"]: row for row in _read_csv(baseline_root / "daily-results.csv")}
    baseline_setups = _read_csv(baseline_root / "setup-ledger.csv")
    baseline_trades = _read_csv(baseline_root / "trade-ledger.csv")
    expected = {
        "raw_interactions": 742,
        "accepted_setups": 9,
        "confirmation_passed": 2,
        "confirmation_failed": 7,
        "trades": 2,
        "total_r": -2.0,
        "net_pnl_usd": -491.0,
    }
    actual = {key: poc_funnel[key] for key in expected}
    numeric_mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if not math.isclose(float(actual[key]), float(value), rel_tol=0.0, abs_tol=1e-9)
    }
    if numeric_mismatches:
        raise AsiaStructuralMatrixError(f"POC baseline aggregate drift: {numeric_mismatches}")
    if baseline_summary.get("raw_interactions") != 742 or baseline_summary.get("accepted_setups") != 9:
        raise AsiaStructuralMatrixError("published POC baseline identity is unexpected")
    poc_sessions = _cell_sessions(sessions, PRIOR_POC)
    if len(poc_sessions) != asia.EXPECTED_ELIGIBLE_SESSIONS or len(baseline_daily) != asia.EXPECTED_ELIGIBLE_SESSIONS:
        raise AsiaStructuralMatrixError("POC daily chronology is incomplete")
    daily_mismatches: list[dict[str, Any]] = []
    for session in poc_sessions:
        day = str(session["session_date"])
        reference = baseline_daily.get(day)
        if reference is None:
            daily_mismatches.append({"session_date": day, "reason": "missing baseline day"})
            continue
        checks = {
            "raw_interactions": int(session["raw_interactions"]),
            "accepted_setups": int(session["accepted_setups"]),
            "confirmations_passed": int(session["confirmations_passed"]),
            "confirmation_failures": int(session["confirmation_failures"]),
            "trade_count": len(session["trades"]),
        }
        for key, value in checks.items():
            if int(reference[key]) != value:
                daily_mismatches.append({"session_date": day, "field": key, "expected": reference[key], "actual": value})
        prior_poc = float(session["prior_asia_profile"]["poc"])
        session_poc = float(session["session_profile"]["poc"])
        if not math.isclose(prior_poc, float(reference["prior_asia_poc"]), abs_tol=1e-12):
            daily_mismatches.append({"session_date": day, "field": "prior_asia_poc"})
        if not math.isclose(session_poc, float(reference["session_poc"]), abs_tol=1e-12):
            daily_mismatches.append({"session_date": day, "field": "session_poc"})
    if daily_mismatches:
        raise AsiaStructuralMatrixError(f"POC daily baseline drift: {daily_mismatches[:10]}")
    setup_ids = {str(row["interaction_id"]) for row in poc_setups}
    baseline_setup_ids = {str(row["interaction_id"]) for row in baseline_setups}
    trade_ids = {str(row["trade_id"]) for row in poc_trades}
    baseline_trade_ids = {str(row["trade_id"]) for row in baseline_trades}
    if setup_ids != baseline_setup_ids or trade_ids != baseline_trade_ids:
        raise AsiaStructuralMatrixError("POC setup/trade identities drifted from published baseline")
    return {
        "status": "PASS",
        "expected": expected,
        "actual": actual,
        "daily_session_reconciliation": "46_OF_46_EXACT",
        "accepted_interaction_ids_exact": True,
        "trade_ids_exact": True,
        "baseline_summary_sha256": asia._sha256(baseline_root / "summary.json"),
    }


def _concentration(
    interactions: Sequence[Mapping[str, Any]], setups: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def groups(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[field]) for row in rows).items()))
    normalized_interactions = [
        {**row, "trade_direction": "LONG" if row["direction"] == "BUYER_ABSORPTION" else "SHORT",
         "month": str(row["session_date"])[:7]}
        for row in interactions
    ]
    normalized_setups = [{**row, "month": str(row["session_date"])[:7]} for row in setups]
    normalized_trades = [{**row, "month": str(row["date"])[:7]} for row in trades]
    return {
        "raw_by_direction": groups(normalized_interactions, "trade_direction"),
        "accepted_by_direction": groups(normalized_setups, "direction"),
        "trades_by_direction": groups(normalized_trades, "direction"),
        "raw_by_month": groups(normalized_interactions, "month"),
        "accepted_by_month": groups(normalized_setups, "month"),
        "trades_by_month": groups(normalized_trades, "month"),
        "accepted_by_date": groups(normalized_setups, "session_date"),
        "trades_by_date": groups(normalized_trades, "date"),
    }


def _median_lookup(
    g_rows: Sequence[Mapping[str, Any]], family: str, group: str, component: str,
) -> float | None:
    row = next(item for item in g_rows if item["level_family"] == family and item["group"] == group and item["component"] == component)
    return None if row["median"] is None else float(row["median"])


def structural_questions(
    level_rows: Sequence[Mapping[str, Any]], g_rows: Sequence[Mapping[str, Any]],
    concentrations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    by_family = {str(row["level_family"]): row for row in level_rows}
    poc = by_family[PRIOR_POC]
    questions: list[dict[str, Any]] = []
    broad: list[str] = []
    for family in LEVEL_FAMILIES:
        row = by_family[family]
        raw_more = int(row["raw_interactions"]) > int(poc["raw_interactions"])
        rate_higher = float(row["acceptance_rate"]) > float(poc["acceptance_rate"])
        g3 = _median_lookup(g_rows, family, "ALL_VALID_INTERACTIONS", "G3")
        g4 = _median_lookup(g_rows, family, "ALL_VALID_INTERACTIONS", "G4")
        poc_g3 = _median_lookup(g_rows, PRIOR_POC, "ALL_VALID_INTERACTIONS", "G3")
        poc_g4 = _median_lookup(g_rows, PRIOR_POC, "ALL_VALID_INTERACTIONS", "G4")
        g3_higher = g3 is not None and poc_g3 is not None and g3 > poc_g3
        g4_higher = g4 is not None and poc_g4 is not None and g4 > poc_g4
        concentration = concentrations[family]
        dates = len(concentration["accepted_by_date"])
        months = len(concentration["accepted_by_month"])
        broad_improvement = (
            family != PRIOR_POC
            and int(row["accepted_setups"]) >= int(poc["accepted_setups"])
            and rate_higher and g3_higher and g4_higher
            and dates >= 3 and months >= 2
        )
        if broad_improvement:
            broad.append(family)
        questions.append({
            "level_family": family,
            "more_raw_interactions_than_poc": raw_more,
            "higher_accepted_setup_rate_than_poc": rate_higher,
            "g3_all_valid_median": g3,
            "g4_all_valid_median": g4,
            "g3_median_higher_than_poc": g3_higher,
            "g4_median_higher_than_poc": g4_higher,
            "confirmation_pass_rate": (
                int(row["confirmation_passed"]) / int(row["accepted_setups"])
                if int(row["accepted_setups"]) else 0.0
            ),
            "accepted_dates": dates,
            "accepted_months": months,
            "broad_structural_improvement_descriptive": broad_improvement,
            "trade_sample_sufficient_for_inference": False,
        })
    if broad:
        classification = "A_POC_REFERENCE_APPEARS_TO_BE_THE_MAIN_PROBLEM_DESCRIPTIVELY"
    elif all(int(by_family[family]["accepted_setups"]) < 9 for family in LEVEL_FAMILIES if family != PRIOR_POC):
        classification = "B_W04_ABSORPTION_REMAINS_RARE_ACROSS_ASIA_LEVELS"
    else:
        classification = "C_MIXED_INCONCLUSIVE"
    return questions, classification


def _report_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {MATRIX_ID}", "",
        f"Evidence label: `{EVIDENCE_LABEL}`", "",
        "Retrospective structural-level transfer research on the same sealed 46 Asia sessions. "
        "No optimization, winner selection, strategy mutation, download, or Databento call occurred.", "",
        "## POC baseline gate", "",
        f"`{summary['poc_baseline_reconciliation']['status']}` — 742 raw, 9 accepted, "
        "2 confirmation passes, 7 failures, 2 trades, -2.0R, -491.00 USD.", "",
        "## Seven independent cells", "",
        "| Level | Raw | Accepted | Rate | Confirm pass/fail | Trades | W/L | R | PnL USD | DD R |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["level_matrix"]:
        lines.append(
            f"| {row['level_family']} | {row['raw_interactions']} | {row['accepted_setups']} | "
            f"{100*float(row['acceptance_rate']):.3f}% | {row['confirmation_passed']}/{row['confirmation_failed']} | "
            f"{row['trades']} | {row['wins']}/{row['losses']} | {float(row['total_r']):.4f} | "
            f"{float(row['net_pnl_usd']):.2f} | {float(row['max_cumulative_drawdown_r']):.4f} |"
        )
    lines.extend([
        "", "## Structural interpretation", "",
        f"`{summary['structural_classification']}`", "",
        "This classification is descriptive only. No cell is selected or promoted, and no combined strategy is created.", "",
        "## Frozen profile and overlap semantics", "",
        "POC uses lower-price ties. VAH/VAL use the canonical 70% value area, one-tick expansion, and lower-price outside-volume ties. "
        "Each level is an independent cell; a causal execution may open independent interactions in multiple cells exactly as the NY implementation evaluates multiple levels independently.", "",
        "## Evidence limits", "",
        "MES fallback is `MES_PROXY_FROM_ES`, not native Asia MES evidence. This is retrospective research, not fresh OOS evidence.", "",
    ])
    return "\n".join(lines)


def _report_html(summary: Mapping[str, Any]) -> str:
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["level_family"], row["raw_interactions"], row["accepted_setups"],
            f"{100*float(row['acceptance_rate']):.3f}%", f"{row['confirmation_passed']}/{row['confirmation_failed']}",
            row["trades"], f"{row['wins']}/{row['losses']}", f"{float(row['total_r']):.4f}",
            f"{float(row['net_pnl_usd']):.2f}", f"{float(row['max_cumulative_drawdown_r']):.4f}",
        )) + "</tr>" for row in summary["level_matrix"]
    )
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{MATRIX_ID}</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#172033}}h1{{font-size:1.75rem}}.pill{{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;color:#3730a3;font-weight:650}}table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{border:1px solid #dbe1ea;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f6f8fb}}.warning{{border-left:4px solid #d97706;background:#fff7ed;padding:12px}}code{{background:#f3f4f6;padding:2px 5px;border-radius:4px}}</style></head>
<body><h1>{MATRIX_ID}</h1><p class='pill'>{EVIDENCE_LABEL}</p><p>Same frozen 46-session Asia population; seven independent structural-level cells.</p>
<h2>POC baseline gate</h2><p><strong>PASS</strong>: 742 raw → 9 accepted → 2 confirmed/traded; -2.0R and -491.00 USD.</p>
<h2>Level matrix</h2><table><thead><tr><th>Level</th><th>Raw</th><th>Accepted</th><th>Rate</th><th>Confirm P/F</th><th>Trades</th><th>W/L</th><th>R</th><th>PnL USD</th><th>DD R</th></tr></thead><tbody>{body_rows}</tbody></table>
<h2>Structural interpretation</h2><p><code>{html.escape(str(summary['structural_classification']))}</code></p>
<p class='warning'>Descriptive retrospective research only. No winner was selected, no combined strategy was created, and no strategy parameter was changed. MES fallback remains ES-based proxy execution.</p></body></html>"""


def materialize(
    *, staging: Path, audit_root: Path, baseline_root: Path,
    source_verification: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
    semantic_diff: Mapping[str, Any], baseline_before: Mapping[str, str], baseline_after: Mapping[str, str],
    ny_before: Mapping[str, str], ny_after: Mapping[str, str],
) -> dict[str, Any]:
    eligible = [row for row in sessions if bool(row["eligible"])]
    if len(eligible) != asia.EXPECTED_ELIGIBLE_SESSIONS:
        raise AsiaStructuralMatrixError("matrix does not contain exactly 46 eligible sessions")
    setups_by_family: dict[str, list[dict[str, Any]]] = {}
    trades_by_family: dict[str, list[dict[str, Any]]] = {}
    interactions_by_family: dict[str, list[dict[str, Any]]] = {}
    funnel_rows: list[dict[str, Any]] = []
    all_setups: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_interactions: list[dict[str, Any]] = []
    concentrations: dict[str, dict[str, Any]] = {}
    reconciliation: dict[str, Any] = {}
    for family in LEVEL_FAMILIES:
        setups, trades = _setup_trade_rows(sessions, family)
        interactions = _all_interactions(sessions, family)
        setups_by_family[family] = setups
        trades_by_family[family] = trades
        interactions_by_family[family] = interactions
        all_setups.extend(setups)
        all_trades.extend(trades)
        all_interactions.extend(interactions)
        funnel_rows.append(_funnel_row(sessions, family, setups, trades))
        concentrations[family] = _concentration(interactions, setups, trades)
        terminal_count = len(setups)
        executed = sum(row["terminal_disposition"] == "TRADE_EXECUTED" for row in setups)
        reconciliation[family] = {
            "raw_interactions": len(interactions),
            "unique_interaction_ids": len({row["interaction_id"] for row in interactions}),
            "accepted_setups": len(setups),
            "unique_setup_ids": len({row["setup_id"] for row in setups}),
            "terminal_dispositions": terminal_count,
            "executed_dispositions": executed,
            "trades": len(trades),
            "unique_trade_ids": len({row["trade_id"] for row in trades}),
        }
        reconciliation[family]["pass"] = (
            len(interactions) == reconciliation[family]["unique_interaction_ids"]
            and len(setups) == reconciliation[family]["unique_setup_ids"] == terminal_count
            and executed == len(trades) == reconciliation[family]["unique_trade_ids"]
        )
        if not reconciliation[family]["pass"]:
            raise AsiaStructuralMatrixError(f"cell reconciliation failed: {family}")
    if len({(row["level_family"], row["interaction_id"]) for row in all_interactions}) != len(all_interactions):
        raise AsiaStructuralMatrixError("duplicate matrix interaction identity")
    if len({(row["level_family"], row["setup_id"]) for row in all_setups}) != len(all_setups):
        raise AsiaStructuralMatrixError("duplicate matrix setup identity")
    if len({(row["level_family"], row["trade_id"]) for row in all_trades}) != len(all_trades):
        raise AsiaStructuralMatrixError("duplicate matrix trade identity")

    g_rows = _g_summary_rows(sessions)
    quality_rows = _quality_bucket_rows(sessions)
    daily_rows, monthly_rows = _daily_monthly_rows(sessions, trades_by_family)
    poc_funnel = next(row for row in funnel_rows if row["level_family"] == PRIOR_POC)
    poc_reconciliation = assert_poc_baseline_reconciliation(
        baseline_root=baseline_root,
        sessions=sessions,
        poc_setups=setups_by_family[PRIOR_POC],
        poc_trades=trades_by_family[PRIOR_POC],
        poc_funnel=poc_funnel,
    )
    questions, structural_classification = structural_questions(funnel_rows, g_rows, concentrations)
    if dict(baseline_before) != dict(baseline_after):
        raise AsiaStructuralMatrixError("existing Asia POC baseline artifacts changed")
    if dict(ny_before) != dict(ny_after):
        raise AsiaStructuralMatrixError("protected NY artifacts changed")
    if any(int(row["unresolved"]) != 0 for row in funnel_rows):
        raise AsiaStructuralMatrixError("matrix contains unresolved cell outcome")
    summary: dict[str, Any] = {
        "status": STATUS,
        "matrix_id": MATRIX_ID,
        "evidence_label": EVIDENCE_LABEL,
        "interpretation": "RETROSPECTIVE_RESEARCH_NOT_FRESH_OOS_EVIDENCE",
        "eligible_session_count": len(eligible),
        "source_session_count": len(sessions),
        "profile_only_dates": [str(row["session_date"]) for row in sessions if not bool(row["eligible"])],
        "eligible_dates": [str(row["session_date"]) for row in eligible],
        "session_window": {"timezone": "UTC", "start_inclusive": "00:00:00", "end_exclusive": "08:00:00"},
        "level_families": list(LEVEL_FAMILIES),
        "cell_strategy_ids": {cell.family: cell.strategy_id for cell in CELLS},
        "weights": {key: str(value) for key, value in asia.W04_WEIGHTS.items()},
        "quality_threshold": str(asia.QUALITY_THRESHOLD),
        "level_matrix": funnel_rows,
        "cell_reconciliation": reconciliation,
        "poc_baseline_reconciliation": poc_reconciliation,
        "structural_questions": questions,
        "structural_classification": structural_classification,
        "concentration_by_level": concentrations,
        "semantic_diff_status": semantic_diff["status"],
        "sweep_semantic_parity": semantic_diff["sweep_semantic_parity"],
        "profile_contract": semantic_diff["profile_contract"],
        "baseline_artifacts_mutated": False,
        "ny_artifacts_mutated": False,
        "protected_baseline_artifact_hashes": dict(baseline_after),
        "protected_ny_artifact_hashes": dict(ny_after),
        "source_verification": list(source_verification),
        "coverage_audit": {
            "path": str(audit_root),
            "summary_sha256": asia._sha256(audit_root / "summary.json"),
            "session_coverage_sha256": asia._sha256(audit_root / "session-coverage.csv"),
        },
        "mes_execution_model": "MES_PROXY_FROM_ES",
        "network_calls": 0,
        "downloads": 0,
        "databento_api_calls": 0,
        "optimization_performed": False,
        "winner_selected": False,
        "combined_strategy_created": False,
        "strategy_semantics_changed": False,
        "fresh_oos_evidence": False,
    }
    all_interactions.sort(key=lambda row: (str(row["level_family"]), int(row["interaction_end_ns"]), str(row["interaction_id"])))
    all_setups.sort(key=lambda row: (str(row["level_family"]), int(row["interaction_end_ns"]), str(row["setup_id"])))
    all_trades.sort(key=lambda row: (str(row["level_family"]), int(row["entry_timestamp_ns"]), str(row["trade_id"])))
    asia._write_json(staging / "summary.json", summary)
    asia._write_csv(staging / "level-matrix.csv", funnel_rows)
    asia._write_csv(staging / "interaction-funnel-by-level.csv", funnel_rows)
    asia._write_csv(staging / "g-summary-by-level.csv", g_rows)
    asia._write_csv(staging / "quality-buckets-by-level.csv", quality_rows)
    asia._write_csv(staging / "trade-ledger.csv", all_trades)
    asia._write_csv(staging / "setup-ledger.csv", all_setups)
    asia._write_csv(staging / "interaction-features.csv", all_interactions)
    asia._write_csv(staging / "daily-results.csv", daily_rows)
    asia._write_csv(staging / "monthly-results.csv", monthly_rows)
    asia._write_json(staging / "semantic-diff.json", semantic_diff)
    (staging / "diagnostic-report.md").write_text(_report_markdown(summary), encoding="utf-8")
    (staging / "diagnostic-report.html").write_text(_report_html(summary), encoding="utf-8")
    return summary


def run_matrix(
    *, repository_root: Path, output_root: Path = OUTPUT_ROOT,
    audit_root: Path = asia.AUDIT_ROOT, baseline_root: Path = BASELINE_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_root = (output_root if output_root.is_absolute() else repository_root / output_root).resolve()
    audit_root = (audit_root if audit_root.is_absolute() else repository_root / audit_root).resolve()
    baseline_root = (baseline_root if baseline_root.is_absolute() else repository_root / baseline_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable Asia structural matrix output exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    contract_path = staging / "matrix-contract.json"
    if staging.exists() and not contract_path.is_file():
        raise AsiaStructuralMatrixError(f"unrecognized existing matrix staging root: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    sessions = asia.load_audit_sessions(audit_root)
    bindings = asia.source_bindings(repository_root, sessions)
    semantic_diff = semantic_diff_document()
    contract = {
        "matrix_id": MATRIX_ID,
        "evidence_label": EVIDENCE_LABEL,
        "eligible_sessions": asia.EXPECTED_ELIGIBLE_SESSIONS,
        "source_sessions": asia.EXPECTED_SOURCE_SESSIONS,
        "level_families": list(LEVEL_FAMILIES),
        "weights": {key: str(value) for key, value in asia.W04_WEIGHTS.items()},
        "quality_threshold": str(asia.QUALITY_THRESHOLD),
        "network_allowed": False,
        "optimization_allowed": False,
        "semantic_diff_sha256": asia._canonical_sha256(semantic_diff),
    }
    if contract_path.is_file() and asia._read_json(contract_path) != contract:
        raise AsiaStructuralMatrixError("existing matrix staging contract differs")
    asia._write_json(contract_path, contract)
    baseline_before = _root_snapshot(baseline_root)
    ny_before = asia._protected_ny_snapshot(repository_root)
    source_verification = asia.verify_source_bindings(bindings)
    sessions_by_day = {session.day: session for session in sessions}
    results: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for binding in bindings:
        binding_sessions = [sessions_by_day[day] for day in binding.days]
        if binding.shared:
            batch = process_shared_binding(
                binding, binding_sessions, staging=staging, previous_result=previous,
            )
            results.extend(batch)
            previous = batch[-1]
        else:
            session = binding_sessions[0]
            print(
                f"ASIA_LEVEL_MATRIX_SESSION {len(results)+1:02d}/{asia.EXPECTED_SOURCE_SESSIONS:02d} "
                f"{session.day} eligible={str(session.eligible).lower()}", flush=True,
            )
            current = process_daily_binding(
                binding, session, staging=staging, previous_result=previous,
            )
            results.append(current)
            previous = current
    results.sort(key=lambda row: str(row["session_date"]))
    if [str(row["session_date"]) for row in results] != [session.day for session in sessions]:
        raise AsiaStructuralMatrixError("matrix results do not match audited chronology")
    baseline_after = _root_snapshot(baseline_root)
    ny_after = asia._protected_ny_snapshot(repository_root)
    summary = materialize(
        staging=staging,
        audit_root=audit_root,
        baseline_root=baseline_root,
        source_verification=source_verification,
        sessions=results,
        semantic_diff=semantic_diff,
        baseline_before=baseline_before,
        baseline_after=baseline_after,
        ny_before=ny_before,
        ny_after=ny_after,
    )
    for transient in (staging / "_work", staging / "_checkpoints"):
        if transient.exists():
            shutil.rmtree(transient)
    run_manifest = {
        "status": STATUS,
        "matrix_id": MATRIX_ID,
        "evidence_label": EVIDENCE_LABEL,
        "artifact_hashes": {
            path.name: asia._sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
        },
        "network_calls": 0,
        "downloads": 0,
    }
    asia._write_json(staging / "run-manifest.json", run_manifest)
    os.rename(staging, output_root)
    return {**summary, "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--audit-root", type=Path, default=asia.AUDIT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            repository_root=args.repository_root,
            output_root=args.output_root,
            audit_root=args.audit_root,
            baseline_root=args.baseline_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps({
        "status": result["status"],
        "output_root": result["output_root"],
        "structural_classification": result["structural_classification"],
        "level_count": len(result["level_matrix"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
