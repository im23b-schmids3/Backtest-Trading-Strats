"""Cross-period causal tapes and offline L2 POC-only weight x quality research.

The explicit tape-build command is the only code path in this module that can
open local DBN files.  The optimizer accepts only compact Parquet tapes, runs
the immutable V3 reproduction gates first, and then evaluates the predeclared
3,876 x 6 grid.  It never imports a Databento client or selects a production
configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import causal_master_tape as master
from . import cross_period_robust_stress as cross
from . import historical_runner as historical
from . import v2_august_seen_replay as august_seen
from . import v2_extended_existing_data as extended
from . import v3_poc_april_retro_replay as april
from . import v3_poc_fresh_august_replay as fresh_august
from . import weight_q_research as matrix
from .model import Execution, L2InteractionEngine, StructuralLevel, TICK
from .v2_quality050 import V2_CONFIG


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH"
EVIDENCE_LABEL = "ALL_PERIOD_RETROSPECTIVE_PARAMETER_RESEARCH_NOT_OOS_EVIDENCE"
TAPE_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_CAUSAL_TAPES")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH")
DEC_JAN_MASTER = master.OUTPUT_ROOT
V3_CONTRACT_SHA256 = cross.V3_HASH
EXPECTED_SESSION_COUNT = 87
EXPECTED_HEAVY_PERIOD_PASSES = 5
EXPECTED_CONFIGURATION_COUNT = matrix.EXPECTED_CONFIGURATION_COUNT
MINIMUM_RANKING_TRADES = 60
TOP_LIMIT = 100
NO_AUTOMATIC_SELECTION = True
REPRODUCTION_BASELINE_LABEL = "SOURCE_INTEGRITY_REPLAY_CLARIFICATION"
REPRODUCTION_BASELINE_REPORT = TAPE_ROOT / "mbo-v3-reproduction-reconciliation.json"
REPRODUCTION_PREFLIGHT_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_REPRODUCTION_PREFLIGHT"
)
EXPECTED_REPRODUCTION_BASELINE_SHA256 = (
    "420cba3cfea295d7623069a2dd95755124ca0322d50a19b8c676a46698a78e2f"
)
REPRODUCTION_METRIC_FIELDS = ("trades", "wins", "losses", "total_r", "net_pnl_usd")
MBO_REPRODUCTION_PERIOD_IDS = ("MAY_2026", "RETRO_JUNE_JULY_2026", "AUGUST_03_06_2026")
NATIVE_REPRODUCTION_PERIOD_IDS = ("APRIL_2026", "AUGUST_10_14_2026", "DECEMBER_2025", "JANUARY_2026")

EXPECTED_V3 = {
    "APRIL_2026": {"trades": 3, "wins": 1, "losses": 2, "total_r": 0.5714285714285716, "net_pnl_usd": 150.25},
    "MAY_2026": {"trades": 5, "wins": 2, "losses": 3, "total_r": 1.9545454545454546, "net_pnl_usd": 467.5},
    "RETRO_JUNE_JULY_2026": {"trades": 8, "wins": 3, "losses": 5, "total_r": 2.6755555555555555, "net_pnl_usd": 620.25},
    "AUGUST_03_06_2026": {"trades": 0, "wins": 0, "losses": 0, "total_r": 0.0, "net_pnl_usd": 0.0},
    "AUGUST_10_14_2026": {"trades": 9, "wins": 3, "losses": 6, "total_r": 1.8119712482950732, "net_pnl_usd": 529.0},
    "DECEMBER_2025": {"trades": 38, "wins": 10, "losses": 28, "total_r": -3.790261594206621, "net_pnl_usd": -793.75},
    "JANUARY_2026": {"trades": 27, "wins": 11, "losses": 16, "total_r": 13.173616179534104, "net_pnl_usd": 2930.25},
}
EXPECTED_V3_AGGREGATE = {
    "sessions": 87, "trades": 90, "wins": 30, "losses": 60,
    "total_r": 16.39685541515214, "net_pnl_usd": 3903.5,
}

EXPECTED_V3_SESSIONS = {
    "APRIL_2026": 3,
    "MAY_2026": 15,
    "RETRO_JUNE_JULY_2026": 18,
    "AUGUST_03_06_2026": 4,
    "AUGUST_10_14_2026": 5,
    "DECEMBER_2025": 22,
    "JANUARY_2026": 20,
}

EXPECTED_MBO_REPRODUCTION_DETAILS = {
    "MAY_2026": {
        "profit_factor": 1.6317567567567568,
        "max_cumulative_drawdown_r": -2.0,
        "es_trades": 0,
        "mes_trades": 5,
        "unresolved": 0,
    },
    "RETRO_JUNE_JULY_2026": {
        "profit_factor": 1.5175219023779725,
        "max_cumulative_drawdown_r": -2.0,
        "es_trades": 1,
        "mes_trades": 7,
        "unresolved": 0,
    },
    "AUGUST_03_06_2026": {
        "profit_factor": None,
        "max_cumulative_drawdown_r": 0.0,
        "es_trades": 0,
        "mes_trades": 0,
        "unresolved": 0,
    },
}

OLD_MBO_V3_BASELINES = {
    "MAY_2026": {"trades": 6, "wins": 3, "losses": 3, "total_r": 4.409090909090909, "net_pnl_usd": 1075.0},
    "RETRO_JUNE_JULY_2026": dict(EXPECTED_V3["RETRO_JUNE_JULY_2026"]),
    "AUGUST_03_06_2026": dict(EXPECTED_V3["AUGUST_03_06_2026"]),
}

SOURCE_GROUP = {
    "APRIL_2026": "NATIVE_MBP10",
    "MAY_2026": "MBO_DERIVED",
    "RETRO_JUNE_JULY_2026": "MBO_DERIVED",
    "AUGUST_03_06_2026": "MBO_DERIVED",
    "AUGUST_10_14_2026": "NATIVE_MBP10",
    "DECEMBER_2025": "NATIVE_MBP10",
    "JANUARY_2026": "NATIVE_MBP10",
}


class AllPeriodResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeriodDefinition:
    period_id: str
    dates: tuple[str, ...]
    source_model: str
    source_group: str
    source_tail_complete: bool
    reusable_master: Path | None = None


PERIODS: tuple[PeriodDefinition, ...] = (
    PeriodDefinition("APRIL_2026", april.TARGET_DATES, "NATIVE_DATABENTO_MBP10", "NATIVE_MBP10", True),
    PeriodDefinition("MAY_2026", historical.MAY_DATES, "MBO_DERIVED_SYNTHETIC_MBP10", "MBO_DERIVED", True),
    PeriodDefinition("RETRO_JUNE_JULY_2026", extended.RETRO_DATES, "MBO_DERIVED_SYNTHETIC_MBP10", "MBO_DERIVED", False),
    PeriodDefinition("AUGUST_03_06_2026", august_seen.TARGET_DATES, "MBO_DERIVED_SYNTHETIC_MBP10_SHARED_FILE", "MBO_DERIVED", True),
    PeriodDefinition("AUGUST_10_14_2026", fresh_august.TARGET_DATES, "NATIVE_DATABENTO_MBP10", "NATIVE_MBP10", True),
    PeriodDefinition("DECEMBER_2025", tuple(), "EXISTING_CAUSAL_MASTER_UNDERLYING_NATIVE_MBP10", "NATIVE_MBP10", True, DEC_JAN_MASTER),
    PeriodDefinition("JANUARY_2026", tuple(), "EXISTING_CAUSAL_MASTER_UNDERLYING_NATIVE_MBP10", "NATIVE_MBP10", True, DEC_JAN_MASTER),
)
PERIOD_BY_ID = {period.period_id: period for period in PERIODS}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllPeriodResearchError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AllPeriodResearchError(f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary, compression="zstd")
    temporary.replace(path)


def _calendar_days(repository_root: Path, period: PeriodDefinition) -> tuple[str, ...]:
    if period.dates:
        return period.dates
    calendar = _read_json(repository_root / DEC_JAN_MASTER / "calendar.json")
    prefix = "2025-12" if period.period_id == "DECEMBER_2025" else "2026-01"
    return tuple(str(day) for day in calendar.get("target_sessions", ()) if str(day).startswith(prefix))


def _period_tape_root(tape_root: Path, period_id: str) -> Path:
    return tape_root / period_id.lower()


def validate_period_tape(root: Path, period: PeriodDefinition) -> dict[str, Any]:
    manifest = _read_json(root / "period-tape-manifest.json")
    if (
        manifest.get("status") != "PERIOD_CAUSAL_TAPE_COMPLETE"
        or manifest.get("period_id") != period.period_id
        or manifest.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or manifest.get("source_model") != period.source_model
        or manifest.get("network_calls") != 0
        or manifest.get("downloads") != 0
    ):
        raise AllPeriodResearchError(f"period tape contract mismatch: {period.period_id}")
    days = tuple(str(day) for day in manifest.get("target_sessions", ()))
    if len(days) != len(set(days)) or tuple(sorted(days)) != days:
        raise AllPeriodResearchError(f"period tape chronology invalid: {period.period_id}")
    required = (root / "interaction-master.parquet", root / "interaction-event-index.parquet")
    if any(not path.is_file() for path in required):
        raise AllPeriodResearchError(f"period tape root is incomplete: {period.period_id}")
    for day in days:
        if not (root / "causal-event-tape" / f"{day}.parquet").is_file():
            raise AllPeriodResearchError(f"period event partition missing: {period.period_id}/{day}")
    interactions = master._read_parquet_rows(required[0])
    indexes = master._read_parquet_rows(required[1])
    ids = [str(row["interaction_id"]) for row in interactions]
    if len(ids) != len(set(ids)) or set(ids) != {str(row["interaction_id"]) for row in indexes}:
        raise AllPeriodResearchError(f"period interaction/index reconciliation failed: {period.period_id}")
    if any(str(row.get("level")) != "PRIOR_RTH_POC" for row in interactions):
        raise AllPeriodResearchError(f"non-POC interaction in period tape: {period.period_id}")
    return {**manifest, "completed_interactions": len(interactions), "indexed_interactions": len(indexes)}


def build_source_inventory(repository_root: Path, tape_root: Path = TAPE_ROOT) -> dict[str, Any]:
    """Inventory only: this function never opens DBNs or contacts a provider."""
    repository_root = repository_root.resolve()
    tape_root = (repository_root / tape_root).resolve() if not tape_root.is_absolute() else tape_root.resolve()
    declared = {row["period_id"]: row for row in cross.build_source_inventory(repository_root)["periods"]}
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        days = _calendar_days(repository_root, period)
        if period.reusable_master is not None:
            reusable = all((repository_root / period.reusable_master / name).is_file() for name in (
                "interaction-master.parquet", "interaction-event-index.parquet", "calendar.json",
            ))
            compact_root = repository_root / period.reusable_master
            status = "REUSABLE_EXISTING_DEC_JAN_MASTER" if reusable else "MISSING_EXISTING_MASTER"
        else:
            compact_root = _period_tape_root(tape_root, period.period_id)
            reusable = compact_root.is_dir()
            if reusable:
                validate_period_tape(compact_root, period)
            status = "REUSABLE_PERIOD_TAPE" if reusable else "ONE_LOCAL_HEAVY_PASS_REQUIRED"
        source = declared[period.period_id]
        rows.append({
            "period_id": period.period_id, "target_sessions": list(days), "session_count": len(days),
            "source_model": period.source_model, "source_group": period.source_group,
            "source_tail_complete": period.source_tail_complete,
            "source_files_available": source["status"] == "READY_FROM_EXISTING_LOCAL_DATA",
            "missing_source_paths": source["missing_paths"], "compact_root": str(compact_root),
            "compact_tape_reusable": reusable, "heavy_local_pass_required": not reusable,
            "status": status,
        })
    if sum(row["session_count"] for row in rows) != EXPECTED_SESSION_COUNT:
        raise AllPeriodResearchError("the seven-period inventory does not contain exactly 87 sessions")
    return {
        "strategy_id": STRATEGY_ID, "evidence_label": EVIDENCE_LABEL,
        "period_count": len(rows), "session_count": EXPECTED_SESSION_COUNT,
        "periods": rows, "network_calls": 0, "downloads": 0,
        "automatic_strategy_selection": False,
    }


@dataclass(frozen=True)
class MboSessionSpec:
    day: str
    es_path: Path
    mes_path: Path
    level_price: float
    start_ns: int
    terminal_ns: int
    complete_hard_flat: bool
    hard_flat_reason: str
    staging_root: Path


class _MboTapeState:
    """One MBO-derived session tape; fed by either a private or shared source pass."""

    def __init__(self, spec: MboSessionSpec) -> None:
        self.spec = spec
        self.adapter = historical.HistoricalMBOToMBP10Adapter()
        self.engine = L2InteractionEngine([StructuralLevel("PRIOR_RTH_POC", spec.level_price)], V2_CONFIG)
        self.tracker = master.CausalWindowTracker()
        self.writer = master.AtomicParquetStream(spec.staging_root / "causal-event-tape" / f"{spec.day}.parquet")
        self.interactions: list[dict[str, Any]] = []
        self.completed_seen = 0
        self.ordinal = 0
        self.stored_events = 0
        self.decoded_records = 0
        self.es_source_index = 0
        self.mes_source_index = 0
        self.es_quote: tuple[float, float] | None = None
        self.mes_quote: tuple[float, float] | None = None
        self.es_quote_ns: int | None = None
        self.mes_quote_ns: int | None = None
        self.prior_es: tuple[float, float] | None = None
        self.prior_mes: tuple[float, float] | None = None
        self.closed = False

    def drain(self) -> None:
        for interaction in self.engine.completed[self.completed_seen:]:
            row = master._interaction_row(interaction, self.spec.day)
            self.interactions.append(row)
            self.tracker.register(row)
        self.completed_seen = len(self.engine.completed)

    def append(self, *, timestamp_ns: int, stream: str, source_index: int, event_type: str,
               execution: Execution | None = None, due: Sequence[str] = (),
               hard_flat_reason: str | None = None, book_state: str | None = None) -> None:
        self.writer.append(master._event_row(
            spec=self.spec, ordinal=self.ordinal, timestamp_ns=timestamp_ns, stream=stream,
            source_index=source_index, event_type=event_type, es_quote=self.es_quote,
            mes_quote=self.mes_quote, execution=execution, entry_probe_count=len(due),
            es_quote_timestamp_ns=self.es_quote_ns, mes_quote_timestamp_ns=self.mes_quote_ns,
            hard_flat_reason=hard_flat_reason, book_state=book_state,
        ))
        self.tracker.bind_entry_probe(due, self.ordinal)
        self.ordinal += 1
        self.stored_events += 1

    def observe_mes(self, row: tuple[int, float, float]) -> None:
        timestamp_ns, bid, ask = row
        self.mes_source_index += 1
        self.decoded_records += 1
        if timestamp_ns < self.spec.start_ns:
            return
        if self.adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
            return
        self.mes_quote, self.mes_quote_ns = (float(bid), float(ask)), timestamp_ns
        due = self.tracker.due_entry_probes(timestamp_ns)
        if self.mes_quote != self.prior_mes or due:
            self.append(timestamp_ns=timestamp_ns, stream="MES", source_index=self.mes_source_index,
                        event_type="MES_BBO", due=due)
            self.prior_mes = self.mes_quote

    def observe_mbo(self, record: Any) -> None:
        self.es_source_index += 1
        self.decoded_records += 1
        previous_state = self.adapter.state
        public = self.adapter.feed(record, materialize_public=record.timestamp_ns >= self.spec.start_ns)
        if record.timestamp_ns < self.spec.start_ns:
            return
        if public is None:
            if self.adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                if previous_state != self.adapter.state or self.es_quote is not None or self.mes_quote is not None:
                    self.es_quote = self.mes_quote = self.prior_es = self.prior_mes = None
                    self.es_quote_ns = self.mes_quote_ns = None
                    self.append(
                        timestamp_ns=record.timestamp_ns, stream="ES", source_index=self.es_source_index,
                        event_type="BOOK_NON_EXECUTABLE", book_state=self.adapter.state,
                    )
            return
        self.engine.advance(public.timestamp_ns)
        self.drain()
        quote = historical._quote(public.snapshot)
        if quote is None:
            raise AllPeriodResearchError(f"MBO adapter emitted non-executable public BBO: {self.spec.day}")
        self.es_quote, self.es_quote_ns = quote, public.timestamp_ns
        self.engine.observe_snapshot(public.snapshot, public.update)
        if public.execution is not None:
            self.engine.observe_execution(public.execution)
            self.drain()
            self.tracker.observe_es_execution(public.execution)
        due = self.tracker.due_entry_probes(public.timestamp_ns)
        if quote != self.prior_es or previous_state != "EXECUTABLE" or public.execution is not None or due:
            self.append(
                timestamp_ns=public.timestamp_ns, stream="ES", source_index=self.es_source_index,
                event_type="ES_EXECUTION" if public.execution is not None else "ES_BBO",
                execution=public.execution, due=due,
            )
            self.prior_es = quote

    def finish(self) -> dict[str, Any]:
        if self.closed:
            raise AllPeriodResearchError(f"duplicate MBO tape close: {self.spec.day}")
        if self.spec.complete_hard_flat:
            self.engine.finish_rth(self.spec.terminal_ns)
            self.drain()
            event_type, reason = "HARD_FLAT", self.spec.hard_flat_reason
        else:
            self.drain()
            event_type, reason = "SOURCE_END", "SOURCE_END_INCOMPLETE_NO_FORCED_EXIT"
        self.append(
            timestamp_ns=self.spec.terminal_ns, stream="CALENDAR", source_index=0,
            event_type=event_type, hard_flat_reason=reason,
        )
        self.adapter.finish()
        event_artifact = self.writer.close()
        index_rows = self.tracker.index_rows(
            day=self.spec.day, first_event=0, last_event=self.ordinal - 1,
            cutoff_ns=self.spec.terminal_ns,
        )
        if len(index_rows) != len(self.interactions):
            raise AllPeriodResearchError(f"MBO interaction/index mismatch: {self.spec.day}")
        session = self.spec.staging_root / "session-builds" / self.spec.day
        interaction_artifact = master._write_small_parquet(session / "interactions.parquet", self.interactions)
        index_artifact = master._write_small_parquet(session / "interaction-index.parquet", index_rows)
        result = {
            "session_date": self.spec.day, "status": "SESSION_CAUSAL_TAPE_COMPLETE",
            "source_model": "MBO_DERIVED_SYNTHETIC_MBP10",
            "source_tail_complete": self.spec.complete_hard_flat,
            "decoded_source_records": self.decoded_records,
            "completed_interactions": len(self.interactions), "stored_events": self.stored_events,
            "confirmed_interactions": sum(row["derived_first_confirmation_timestamp_ns"] is not None for row in index_rows),
            "entry_probe_interactions": sum(row["entry_observation_event_ordinal"] is not None for row in index_rows),
            "source_integrity_anomalies": len(self.adapter.source_integrity_diagnostics()),
            "artifacts": {"events": event_artifact, "interactions": interaction_artifact, "index": index_artifact},
        }
        master._write_json(self.spec.staging_root / "session-builds" / f"{self.spec.day}.json", result)
        self.closed = True
        return result

    def abort(self) -> None:
        if not self.closed:
            self.writer.abort()


def _session_checkpoint(root: Path, day: str) -> dict[str, Any] | None:
    return master._session_checkpoint_valid(root / "session-builds" / f"{day}.json")


def _build_mbo_file_session(spec: MboSessionSpec) -> dict[str, Any]:
    existing = _session_checkpoint(spec.staging_root, spec.day)
    if existing is not None:
        return existing
    state = _MboTapeState(spec)
    es_iter = iter(historical._stream_private_mbo(spec.es_path))
    mes_iter = iter(historical._stream_mes_quotes(spec.mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    next_progress = 5_000_000
    try:
        while es is not None or mes is not None:
            es_ts = es.timestamp_ns if es is not None else 2**63 - 1
            mes_ts = mes[0] if mes is not None else 2**63 - 1
            timestamp_ns = min(es_ts, mes_ts)
            if timestamp_ns >= spec.terminal_ns:
                break
            # Match the frozen MBO runner: ES wins an exact timestamp tie.
            if mes_ts < es_ts:
                state.observe_mes(mes)
                mes = historical._next(mes_iter)
            else:
                state.observe_mbo(es)
                es = historical._next(es_iter)
            if state.decoded_records >= next_progress:
                print(
                    f"ALL_PERIOD_TAPE {spec.day} records={state.decoded_records:,} "
                    f"interactions={len(state.interactions):,} events={state.stored_events:,}",
                    flush=True,
                )
                next_progress += 5_000_000
        return state.finish()
    except BaseException:
        state.abort()
        raise


def _build_shared_august_sessions(specs: Sequence[MboSessionSpec]) -> list[dict[str, Any]]:
    if tuple(spec.day for spec in specs) != august_seen.TARGET_DATES:
        raise AllPeriodResearchError("shared August builder requires the exact four-session chronology")
    existing = [_session_checkpoint(spec.staging_root, spec.day) for spec in specs]
    if all(existing):
        return [dict(row) for row in existing if row is not None]
    if any(existing):
        raise AllPeriodResearchError("partial shared-August checkpoints would require a second shared-file pass")
    states = {spec.day: _MboTapeState(spec) for spec in specs}
    mes_iters = {spec.day: iter(historical._stream_mes_quotes(spec.mes_path)) for spec in specs}
    mes_next = {day: historical._next(iterator) for day, iterator in mes_iters.items()}
    closed: set[str] = set()
    records = 0
    shared_path = specs[0].es_path
    try:
        for record in historical._stream_private_mbo(shared_path):
            day = august_seen._date_from_ns(record.timestamp_ns)
            if day > specs[-1].day:
                break
            if day not in states or day in closed:
                continue
            state = states[day]
            while mes_next[day] is not None and mes_next[day][0] < record.timestamp_ns:
                state.observe_mes(mes_next[day])
                mes_next[day] = historical._next(mes_iters[day])
            if record.timestamp_ns >= state.spec.terminal_ns:
                state.finish()
                closed.add(day)
                continue
            state.observe_mbo(record)
            records += 1
            if records % 5_000_000 == 0:
                print(f"ALL_PERIOD_TAPE AUGUST_SHARED records={records:,} closed={len(closed)}/4", flush=True)
        if closed != set(states):
            raise AllPeriodResearchError("shared August source did not reach all four hard-flat boundaries")
        return [_session_checkpoint(spec.staging_root, spec.day) for spec in specs]  # type: ignore[return-value]
    except BaseException:
        for day, state in states.items():
            if day not in closed:
                state.abort()
        raise


def _publish_period_tape(
    *, staging: Path, output: Path, period: PeriodDefinition,
    days: Sequence[str], sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined = master._combine_session_artifacts(staging, sessions)
    manifest = {
        "status": "PERIOD_CAUSAL_TAPE_COMPLETE", "artifact_kind": "L2_PERIOD_CAUSAL_TAPE",
        "period_id": period.period_id, "target_sessions": list(days), "session_count": len(days),
        "source_model": period.source_model, "source_group": period.source_group,
        "source_tail_complete": period.source_tail_complete,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "completed_interactions": len(combined["interaction_rows"]),
        "indexed_interactions": len(combined["index_rows"]),
        "stored_events": sum(int(row["stored_events"]) for row in sessions),
        "heavy_source_passes": 1, "pre_quality_poc_population": True,
        "network_calls": 0, "downloads": 0, "strategy_outcomes_consulted": False,
        "automatic_strategy_selection": False,
    }
    _write_json(staging / "period-tape-manifest.json", manifest)
    _write_json(staging / "calendar.json", {
        "period_id": period.period_id, "target_sessions": list(days),
        "source_tail_complete": period.source_tail_complete,
    })
    if output.exists():
        raise FileExistsError(f"immutable period tape already exists: {output}")
    os.rename(staging, output)
    return {**manifest, "output_root": str(output)}


def _native_specs(repository_root: Path, staging: Path, period: PeriodDefinition) -> list[master.SessionBuildSpec]:
    specs: list[master.SessionBuildSpec] = []
    data_root = repository_root / (
        april.DATA_ROOT if period.period_id == "APRIL_2026" else fresh_august.DATA_ROOT
    )
    if period.period_id == "APRIL_2026":
        april.verify_acquisition_manifest(data_root)
    else:
        fresh_august.verify_acquisition_manifest(data_root)
    for day in period.dates:
        es_path, mes_path, profile_path = (
            april._paths(data_root, day) if period.period_id == "APRIL_2026"
            else fresh_august._paths(data_root, day)
        )
        level = (
            april._validated_prior_rth_poc(profile_path) if period.period_id == "APRIL_2026"
            else fresh_august._profile_poc(profile_path)
        )
        cutoff_seconds = (
            historical.HARD_CUTOFF_SECONDS if period.period_id == "APRIL_2026"
            else fresh_august.effective_hard_flat_seconds(day)
        )
        scheduled = cutoff_seconds != historical.HARD_CUTOFF_SECONDS
        specs.append(master.SessionBuildSpec(
            day=day, es_path=str(es_path), mes_path=str(mes_path), level_price=level.price,
            start_ns=historical._clock_ns(day, historical.RTH_START_SECONDS),
            cutoff_ns=historical._clock_ns(day, cutoff_seconds),
            hard_flat_reason="HARD_FLAT_SCHEDULED_CLOSE_2100" if scheduled else "HARD_CUTOFF_2245",
            staging_root=str(staging), maintenance_mode="SUMMER_NATIVE",
        ))
    return specs


def _mbo_specs(repository_root: Path, staging: Path, period: PeriodDefinition) -> list[MboSessionSpec]:
    specs: list[MboSessionSpec] = []
    if period.period_id == "MAY_2026":
        data_root = repository_root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy"
        historical.verify_may_acquisition_manifest(data_root)
        for day in period.dates:
            paths = historical._may_paths(day)
            level = cross._select_poc(historical._profile_levels_from_declared_trades(data_root / paths["ES_PRIOR_RTH_PROFILE"]))[0]
            specs.append(MboSessionSpec(
                day, data_root / paths["ES_MBO_L3"], data_root / paths["MES_NATIVE_EXECUTION"], level.price,
                historical._clock_ns(day, historical.RTH_START_SECONDS),
                historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS), True,
                "HARD_CUTOFF_2245", staging,
            ))
    elif period.period_id == "RETRO_JUNE_JULY_2026":
        data_root = repository_root / "data/cme_orderflow_absorption_v2_holdout"
        for day in period.dates:
            prior = extended.RETRO_PRIOR_RTH[day]
            level = cross._select_poc(historical._profile_levels_from_declared_trades(
                data_root / "es_rth_trades" / f"ESU6_{prior}_1330_2000_trades.dbn.zst"
            ))[0]
            specs.append(MboSessionSpec(
                day, data_root / "es_mbo" / f"ESU6_{day}_0000_1600_mbo.dbn.zst",
                data_root / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst", level.price,
                historical._clock_ns(day, historical.RTH_START_SECONDS),
                historical._clock_ns(day, 16 * 3600), False,
                "SOURCE_END_INCOMPLETE_NO_FORCED_EXIT", staging,
            ))
    elif period.period_id == "AUGUST_03_06_2026":
        data_root = repository_root / "data/cme_orderflow_absorption_l2_v2/august_completion"
        august_seen.verify_august_inputs(repository_root=repository_root, completion_root=data_root)
        shared = repository_root / august_seen.ES_MBO_RELATIVE
        for day in period.dates:
            profile, mes = august_seen._paths(data_root, day)
            level = cross._select_poc(historical._profile_levels_from_declared_trades(profile))[0]
            specs.append(MboSessionSpec(
                day, shared, mes, level.price,
                historical._clock_ns(day, historical.RTH_START_SECONDS),
                historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS), True,
                "HARD_CUTOFF_2245", staging,
            ))
    else:
        raise AllPeriodResearchError(f"unsupported MBO tape period: {period.period_id}")
    return specs


def build_period_tape(*, repository_root: Path, tape_root: Path, period_id: str) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = (repository_root / tape_root).resolve() if not tape_root.is_absolute() else tape_root.resolve()
    period = PERIOD_BY_ID.get(period_id)
    if period is None or period.reusable_master is not None:
        raise AllPeriodResearchError(f"period does not require a new tape: {period_id}")
    output = _period_tape_root(tape_root, period_id)
    if output.exists():
        return validate_period_tape(output, period)
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    if period.source_group == "NATIVE_MBP10":
        sessions = []
        for index, spec in enumerate(_native_specs(repository_root, staging, period), start=1):
            print(f"ALL_PERIOD_TAPE {period_id} {index:02d}/{len(period.dates):02d} {spec.day}", flush=True)
            sessions.append(master._build_session(spec))
    else:
        specs = _mbo_specs(repository_root, staging, period)
        if period_id == "AUGUST_03_06_2026":
            sessions = _build_shared_august_sessions(specs)
        else:
            sessions = []
            for index, spec in enumerate(specs, start=1):
                print(f"ALL_PERIOD_TAPE {period_id} {index:02d}/{len(specs):02d} {spec.day}", flush=True)
                sessions.append(_build_mbo_file_session(spec))
    if len(sessions) != len(period.dates):
        raise AllPeriodResearchError(f"period session build count mismatch: {period_id}")
    return _publish_period_tape(
        staging=staging, output=output, period=period, days=period.dates, sessions=sessions,
    )


def build_missing_tapes(*, repository_root: Path, tape_root: Path) -> dict[str, Any]:
    inventory = build_source_inventory(repository_root, tape_root)
    missing = [
        row["period_id"] for row in inventory["periods"]
        if row["heavy_local_pass_required"] and row["period_id"] not in {"DECEMBER_2025", "JANUARY_2026"}
    ]
    results = [build_period_tape(repository_root=repository_root, tape_root=tape_root, period_id=period_id) for period_id in missing]
    return {
        "status": "ALL_REQUIRED_PERIOD_TAPES_AVAILABLE", "built_periods": missing,
        "heavy_period_passes": len(missing), "period_results": results,
        "network_calls": 0, "downloads": 0,
    }


@dataclass(frozen=True)
class PeriodBundle:
    period: PeriodDefinition
    root: Path
    days: tuple[str, ...]


def _period_bundles(repository_root: Path, tape_root: Path) -> list[PeriodBundle]:
    bundles: list[PeriodBundle] = []
    for period in PERIODS:
        days = _calendar_days(repository_root, period)
        root = (
            repository_root / period.reusable_master
            if period.reusable_master is not None
            else _period_tape_root(tape_root, period.period_id)
        )
        if period.reusable_master is None:
            validate_period_tape(root, period)
        elif not root.is_dir():
            raise AllPeriodResearchError(f"existing Dec/Jan master missing: {root}")
        bundles.append(PeriodBundle(period, root, days))
    if sum(len(bundle.days) for bundle in bundles) != EXPECTED_SESSION_COUNT:
        raise AllPeriodResearchError("period bundle chronology does not contain exactly 87 sessions")
    return bundles


def _load_bundle_rows(bundle: PeriodBundle) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    interactions = matrix._load_filtered_parquet(bundle.root / "interaction-master.parquet", bundle.days)
    index_rows = matrix._load_filtered_parquet(bundle.root / "interaction-event-index.parquet", bundle.days)
    indexes = {str(row["interaction_id"]): row for row in index_rows}
    if len(indexes) != len(index_rows) or {str(row["interaction_id"]) for row in interactions} != set(indexes):
        raise AllPeriodResearchError(f"bundle interaction/index mismatch: {bundle.period.period_id}")
    interactions.sort(key=lambda row: (
        str(row["session_date"]), int(row["interaction_end_ns"]), str(row["source_interaction_id"]),
    ))
    return interactions, indexes


def validate_corrected_baseline_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen seven-period V3 reproduction source of truth."""
    if payload.get("status") != "MBO_V3_REPRODUCTION_RECONCILIATION_COMPLETE":
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_STATUS_INVALID")
    if payload.get("classification") != REPRODUCTION_BASELINE_LABEL:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_CLASSIFICATION_INVALID")
    if payload.get("v3_contract_sha256") != V3_CONTRACT_SHA256:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_STRATEGY_HASH_INVALID")
    if payload.get("v3_parameter_changes") is not False or payload.get("weight_changes") is not False:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_PARAMETER_CHANGE_FORBIDDEN")
    if payload.get("quality_threshold_changes") is not False:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_THRESHOLD_CHANGE_FORBIDDEN")
    if payload.get("optimizer_executed") is not False or int(payload.get("weight_q_configurations_evaluated", -1)) != 0:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_OPTIMIZER_CONTAMINATION")
    if int(payload.get("network_calls", -1)) != 0 or int(payload.get("downloads", -1)) != 0:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_NETWORK_CONTAMINATION")
    updates = payload.get("baseline_updates")
    if not isinstance(updates, Mapping) or set(updates) != {"MAY_2026"}:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_SCOPE_INVALID")
    update = updates["MAY_2026"]
    if not isinstance(update, Mapping) or update.get("label") != REPRODUCTION_BASELINE_LABEL:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_UPDATE_INVALID")
    old, corrected = update.get("old"), update.get("corrected")
    if not isinstance(old, Mapping) or not isinstance(corrected, Mapping):
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_METRICS_MISSING")
    for field in REPRODUCTION_METRIC_FIELDS:
        expected_old, expected_new = OLD_MBO_V3_BASELINES["MAY_2026"][field], EXPECTED_V3["MAY_2026"][field]
        if not math.isclose(float(old[field]), float(expected_old), rel_tol=0.0, abs_tol=1e-12):
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_OLD_METRIC_INVALID:{field}")
        if not math.isclose(float(corrected[field]), float(expected_new), rel_tol=0.0, abs_tol=1e-12):
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_NEW_METRIC_INVALID:{field}")
    canonical = {
        "classification": payload["classification"],
        "v3_contract_sha256": payload["v3_contract_sha256"],
        "baseline_updates": updates,
    }
    expected_hash = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    documented_hash = payload.get("canonical_baseline_sha256")
    if documented_hash != EXPECTED_REPRODUCTION_BASELINE_SHA256:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_CANONICAL_HASH_INVALID:FROZEN_EXPECTATION")
    if documented_hash != expected_hash:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_CANONICAL_HASH_INVALID")
    if payload.get("optimizer_gate_update_required") is not True:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_GATE_UPDATE_INVALID")
    if payload.get("optimizer_permitted_after_documented_gate_validation") is not True:
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_OPTIMIZER_PERMISSION_INVALID")

    periods = payload.get("periods")
    native_periods = payload.get("native_periods")
    if not isinstance(periods, Mapping) or set(periods) != set(MBO_REPRODUCTION_PERIOD_IDS):
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_MBO_PERIOD_SCOPE_INVALID")
    if not isinstance(native_periods, Mapping) or set(native_periods) != set(NATIVE_REPRODUCTION_PERIOD_IDS):
        raise AllPeriodResearchError("V3_CORRECTED_BASELINE_NATIVE_PERIOD_SCOPE_INVALID")

    canonical_rows: list[dict[str, Any]] = []
    for period_id in MBO_REPRODUCTION_PERIOD_IDS:
        period = periods[period_id]
        if not isinstance(period, Mapping):
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_PERIOD_INVALID:{period_id}")
        decision = period.get("canonical_decision")
        expected_decision = (
            "SUPERSEDED_BY_MBO_PUBLIC_BOOK_INTEGRITY_CLARIFICATION"
            if period_id == "MAY_2026" else "OLD_PUBLISHED_BASELINE_REMAINS_EXACT"
        )
        if decision != expected_decision:
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_DECISION_INVALID:{period_id}")
        metrics = period.get("corrected_semantics")
        if not isinstance(metrics, Mapping):
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_METRICS_MISSING:{period_id}")
        canonical_rows.append(_validated_reproduction_metric_row(
            period_id, metrics, gate_source="HASH_BOUND_CORRECTED_MBO_RECONCILIATION",
        ))
        for field, expected in EXPECTED_MBO_REPRODUCTION_DETAILS[period_id].items():
            _require_same_metric(period_id, field, metrics.get(field), expected)

    for period_id in NATIVE_REPRODUCTION_PERIOD_IDS:
        period = native_periods[period_id]
        if not isinstance(period, Mapping) or period.get("status") != "UNCHANGED":
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_NATIVE_STATUS_INVALID:{period_id}")
        metrics = period.get("published")
        if not isinstance(metrics, Mapping):
            raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_NATIVE_METRICS_MISSING:{period_id}")
        canonical_rows.append(_validated_reproduction_metric_row(
            period_id, metrics, gate_source="HASH_BOUND_UNCHANGED_NATIVE_EVIDENCE",
        ))

    gate = _evaluate_reproduction_gates(canonical_rows)
    if not gate["optimizer_permitted"]:
        failure = next(item for item in gate["period_gates"] if item["status"] != "PASS")
        raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_PERIOD_MISMATCH:{failure['period_id']}")
    return dict(payload)


def _same_metric(actual: object, expected: object, *, field: str) -> bool:
    if expected is None:
        return actual is None
    if field in {"sessions", "trades", "wins", "losses", "es_trades", "mes_trades", "unresolved"}:
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return False
    tolerance = 1e-9 if field == "net_pnl_usd" else 1e-12
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _require_same_metric(period_id: str, field: str, actual: object, expected: object) -> None:
    if not _same_metric(actual, expected, field=field):
        raise AllPeriodResearchError(f"V3_CORRECTED_BASELINE_METRIC_INVALID:{period_id}:{field}")


def _validated_reproduction_metric_row(
    period_id: str,
    metrics: Mapping[str, Any],
    *,
    gate_source: str,
) -> dict[str, Any]:
    expected = EXPECTED_V3[period_id]
    _require_same_metric(period_id, "sessions", metrics.get("sessions"), EXPECTED_V3_SESSIONS[period_id])
    for field in REPRODUCTION_METRIC_FIELDS:
        _require_same_metric(period_id, field, metrics.get(field), expected[field])
    return {
        "period_id": period_id,
        "sessions": int(metrics["sessions"]),
        **{field: metrics[field] for field in REPRODUCTION_METRIC_FIELDS},
        "source_group": SOURCE_GROUP[period_id],
        "gate_source": gate_source,
    }


def reproduction_rows_from_baseline_document(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select each period's actual gate row from the frozen reconciliation artifact."""
    document = validate_corrected_baseline_document(payload)
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        period_id = period.period_id
        if period_id in MBO_REPRODUCTION_PERIOD_IDS:
            metrics = document["periods"][period_id]["corrected_semantics"]
            source = "HASH_BOUND_CORRECTED_MBO_RECONCILIATION"
        else:
            metrics = document["native_periods"][period_id]["published"]
            source = "HASH_BOUND_UNCHANGED_NATIVE_EVIDENCE"
        rows.append(_validated_reproduction_metric_row(period_id, metrics, gate_source=source))
    return rows


def load_corrected_baseline_document(repository_root: Path) -> dict[str, Any]:
    path = repository_root / REPRODUCTION_BASELINE_REPORT
    return validate_corrected_baseline_document(_read_json(path))


def _simulate_v3_bundle(bundle: PeriodBundle) -> dict[str, Any]:
    interactions, indexes = _load_bundle_rows(bundle)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    accumulator = matrix.ConfigurationAccumulator((1, 1, 1, 1, 16), Decimal("0.50"))
    all_trades: list[dict[str, Any]] = []
    for day in bundle.days:
        tape = matrix.SessionCausalTape.from_parquet(day, bundle.root / "causal-event-tape" / f"{day}.parquet")
        rows = [row for row in by_day.get(day, ()) if master.interaction_is_accepted(row, threshold="0.50")]
        session = matrix.simulate_independent_session(
            tape, rows, {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows},
        )
        accumulator.add(session, set())
        all_trades.extend(session.trades)
    performance = historical._performance(all_trades)
    return {
        "period_id": bundle.period.period_id, "sessions": len(bundle.days),
        "completed_interactions": len(interactions), "trades": performance["completed_trades"],
        "wins": performance["wins"], "losses": performance["losses"],
        "total_r": performance["total_r"], "net_pnl_usd": performance["net_pnl_usd"],
        "profit_factor": performance["profit_factor"],
        "max_cumulative_drawdown_r": performance["max_cumulative_drawdown_r"],
        "unresolved": accumulator.unresolved,
        "source_group": bundle.period.source_group,
    }


def _evaluate_reproduction_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["period_id"]): row for row in rows}
    exact_scope = len(by_id) == len(rows) and set(by_id) == set(EXPECTED_V3)
    gates: list[dict[str, Any]] = []
    for period_id, expected in EXPECTED_V3.items():
        actual = by_id.get(period_id)
        passed = actual is not None and all(
            _same_metric(actual.get(field), expected[field], field=field)
            for field in REPRODUCTION_METRIC_FIELDS
        ) and _same_metric(
            actual.get("sessions"), EXPECTED_V3_SESSIONS[period_id], field="sessions",
        )
        gates.append({
            "period_id": period_id,
            "status": "PASS" if passed else "FAIL",
            "expected": {"sessions": EXPECTED_V3_SESSIONS[period_id], **expected},
            "actual": None if actual is None else dict(actual),
        })
    aggregate = {
        "sessions": sum(int(row.get("sessions", 0)) for row in rows),
        "trades": sum(int(row.get("trades", 0)) for row in rows),
        "wins": sum(int(row.get("wins", 0)) for row in rows),
        "losses": sum(int(row.get("losses", 0)) for row in rows),
        "total_r": sum(float(row.get("total_r", 0.0)) for row in rows),
        "net_pnl_usd": sum(float(row.get("net_pnl_usd", 0.0)) for row in rows),
    }
    aggregate_fields = {
        name: _same_metric(aggregate[name], expected, field=name)
        for name, expected in EXPECTED_V3_AGGREGATE.items()
    }
    period_pass = exact_scope and all(item["status"] == "PASS" for item in gates)
    aggregate_pass = all(aggregate_fields.values())
    permitted = period_pass and aggregate_pass
    return {
        "status": (
            "ALL_SEVEN_V3_REPRODUCTION_GATES_PASS"
            if permitted else "V3_REPRODUCTION_PREFLIGHT_FAILED"
        ),
        "baseline_label": REPRODUCTION_BASELINE_LABEL,
        "reconciliation_hash": EXPECTED_REPRODUCTION_BASELINE_SHA256,
        "period_scope_exact": exact_scope,
        "period_gates": gates,
        "aggregate_gate": {
            "status": "PASS" if aggregate_pass else "FAIL",
            "expected": dict(EXPECTED_V3_AGGREGATE),
            "actual": aggregate,
            "field_status": {name: "PASS" if passed else "FAIL" for name, passed in aggregate_fields.items()},
        },
        "aggregate": aggregate,
        "optimizer_permitted": permitted,
    }


def assert_reproduction_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _evaluate_reproduction_gates(rows)
    if not result["period_scope_exact"]:
        raise AllPeriodResearchError("V3 reproduction did not cover exactly seven periods")
    for gate in result["period_gates"]:
        if gate["status"] != "PASS":
            raise AllPeriodResearchError(f"V3_REPRODUCTION_GATE_FAILED:{gate['period_id']}")
    if result["aggregate_gate"]["status"] != "PASS":
        failed = next(
            name for name, status in result["aggregate_gate"]["field_status"].items()
            if status != "PASS"
        )
        raise AllPeriodResearchError(f"V3_AGGREGATE_REPRODUCTION_FAILED:{failed}")
    return result


def _evaluate_bundle(
    bundle: PeriodBundle,
    *, progress_interval: int = matrix.PROGRESS_INTERVAL,
) -> list[dict[str, Any]]:
    interactions, indexes = _load_bundle_rows(bundle)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    registry = matrix.configuration_registry()
    weight_grid = matrix.generate_weight_grid()
    weights_matrix = np.asarray(weight_grid, dtype=np.float64) * float(matrix.WEIGHT_UNIT)
    accumulators = [matrix.ConfigurationAccumulator(weights, threshold) for weights, threshold in registry]
    evaluations = 0
    next_progress = progress_interval
    started = time.monotonic()
    for session_number, day in enumerate(bundle.days, start=1):
        print(
            f"ALL_PERIOD_MATRIX {bundle.period.period_id} "
            f"session={session_number:02d}/{len(bundle.days):02d} {day}", flush=True,
        )
        rows = by_day.get(day, [])
        tape = matrix.SessionCausalTape.from_parquet(day, bundle.root / "causal-event-tape" / f"{day}.parquet")
        if rows:
            components = np.asarray([[float(row[name]) for name in master.SCORE_FIELDS] for row in rows], dtype=np.float64)
            penalties = np.asarray([float(row["false_refill_penalty"]) for row in rows], dtype=np.float64)
            primitive_ok = np.asarray([not str(row.get("non_quality_rejection_reasons") or "") for row in rows])
            scores = np.clip(
                components @ weights_matrix.T
                - penalties[:, None] * float(master.V2_CONFIG.false_refill_penalty_weight), 0.0, 1.0,
            )
        else:
            scores = np.empty((0, matrix.EXPECTED_WEIGHT_COUNT), dtype=np.float64)
            primitive_ok = np.empty((0,), dtype=np.bool_)
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        for weight_index, units in enumerate(weight_grid):
            base = weight_index * len(matrix.QUALITY_THRESHOLDS)
            for q_index, threshold in enumerate(matrix.QUALITY_THRESHOLDS):
                accepted_mask = matrix._accepted_mask(rows, primitive_ok, scores[:, weight_index], units, threshold)
                accepted = [row for row, keep in zip(rows, accepted_mask) if bool(keep)]
                session = matrix.simulate_independent_session(tape, accepted, day_indexes)
                accumulators[base + q_index].add(session, set())
                evaluations += 1
        equivalent = evaluations // max(1, len(bundle.days))
        if equivalent >= next_progress:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"ALL_PERIOD_MATRIX_PROGRESS period={bundle.period.period_id} "
                f"equivalent_configs={equivalent:,}/{EXPECTED_CONFIGURATION_COUNT:,} "
                f"rate={evaluations / elapsed:,.1f}_config_sessions_per_second",
                flush=True,
            )
            while next_progress <= equivalent:
                next_progress += progress_interval
        del tape, scores
    output: list[dict[str, Any]] = []
    for accumulator in accumulators:
        row = accumulator.row(0)
        v3 = EXPECTED_V3[bundle.period.period_id]
        output.append({
            "period_id": bundle.period.period_id, "source_model": bundle.period.source_model,
            "source_group": bundle.period.source_group, "sessions": len(bundle.days),
            "completed_interactions": len(interactions),
            **row,
            "gross_profit_usd": accumulator.gross_profit_usd,
            "gross_loss_usd": accumulator.gross_loss_usd,
            "v3_reference_trades": v3["trades"],
            "v3_reference_total_r": v3["total_r"],
            "v3_reference_net_pnl_usd": v3["net_pnl_usd"],
            "trade_delta_vs_v3": int(row["trades"]) - int(v3["trades"]),
            "total_r_delta_vs_v3": float(row["total_r"]) - float(v3["total_r"]),
            "net_pnl_delta_vs_v3": float(row["net_pnl_usd"]) - float(v3["net_pnl_usd"]),
        })
    return output


def _finite(value: object) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_configuration_periods(period_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(period_rows) != len(PERIODS):
        raise AllPeriodResearchError("aggregate configuration requires exactly seven period rows")
    config_ids = {str(row["config_id"]) for row in period_rows}
    if len(config_ids) != 1:
        raise AllPeriodResearchError("aggregate rows do not belong to one configuration")
    period_r = [float(row["total_r"]) for row in period_rows]
    trades = sum(int(row["trades"]) for row in period_rows)
    total_r = sum(period_r)
    gross_profit = sum(float(row["gross_profit_usd"]) for row in period_rows)
    gross_loss = sum(float(row["gross_loss_usd"]) for row in period_rows)
    pfs = [value for row in period_rows if (value := _finite(row.get("profit_factor"))) is not None]
    native = [row for row in period_rows if row["source_group"] == "NATIVE_MBP10"]
    mbo = [row for row in period_rows if row["source_group"] == "MBO_DERIVED"]
    first = period_rows[0]
    native_r, mbo_r = sum(float(row["total_r"]) for row in native), sum(float(row["total_r"]) for row in mbo)
    if native_r > 0 and mbo_r > 0:
        source_class = "POSITIVE_BOTH_SOURCE_MODELS"
    elif native_r > 0:
        source_class = "POSITIVE_NATIVE_ONLY"
    elif mbo_r > 0:
        source_class = "POSITIVE_MBO_DERIVED_ONLY"
    else:
        source_class = "NEGATIVE_BOTH_SOURCE_MODELS"
    return {
        "config_id": first["config_id"],
        **{f"G{index}": first[f"G{index}"] for index in range(1, 6)},
        "quality_threshold": first["quality_threshold"],
        "total_trades": trades, "wins": sum(int(row["wins"]) for row in period_rows),
        "losses": sum(int(row["losses"]) for row in period_rows),
        "total_r": total_r, "total_net_pnl_usd": sum(float(row["net_pnl_usd"]) for row in period_rows),
        "weighted_average_r": total_r / trades if trades else 0.0,
        "aggregate_profit_factor": gross_profit / gross_loss if gross_loss else None,
        "positive_periods": sum(value > 0 for value in period_r),
        "negative_periods": sum(value < 0 for value in period_r),
        "periods_with_trades": sum(int(row["trades"]) > 0 for row in period_rows),
        "median_period_r": statistics.median(period_r), "worst_period_r": min(period_r),
        "best_period_r": max(period_r), "median_period_profit_factor": statistics.median(pfs) if pfs else None,
        "worst_finite_period_profit_factor": min(pfs) if pfs else None,
        "worst_period_max_drawdown_r": min(float(row["max_cumulative_drawdown_r"]) for row in period_rows),
        "total_r_excluding_best_period": total_r - max(period_r),
        "total_r_excluding_worst_period": total_r - min(period_r),
        "period_r_standard_deviation": statistics.pstdev(period_r),
        "period_r_dispersion": max(period_r) - min(period_r),
        "proportion_periods_positive": sum(value > 0 for value in period_r) / len(period_r),
        "native_mbp10_total_r": native_r, "native_mbp10_trade_count": sum(int(row["trades"]) for row in native),
        "mbo_derived_total_r": mbo_r, "mbo_derived_trade_count": sum(int(row["trades"]) for row in mbo),
        "source_model_classification": source_class,
        "source_balance_min_total_r": min(native_r, mbo_r),
        "source_balance_absolute_r_gap": abs(native_r - mbo_r),
        "unresolved": sum(int(row["unresolved"]) for row in period_rows),
        "es_trades": sum(int(row["es_trades"]) for row in period_rows),
        "mes_trades": sum(int(row["mes_trades"]) for row in period_rows),
        "target_exits": sum(int(row["target_exits"]) for row in period_rows),
        "stop_exits": sum(int(row["stop_exits"]) for row in period_rows),
        "cutoff_exits": sum(int(row["hard_cutoff_exits"]) for row in period_rows),
        "v3_reference_trades": EXPECTED_V3_AGGREGATE["trades"],
        "v3_reference_total_r": EXPECTED_V3_AGGREGATE["total_r"],
        "v3_reference_net_pnl_usd": EXPECTED_V3_AGGREGATE["net_pnl_usd"],
        "trade_delta_vs_v3": trades - int(EXPECTED_V3_AGGREGATE["trades"]),
        "total_r_delta_vs_v3": total_r - float(EXPECTED_V3_AGGREGATE["total_r"]),
        "net_pnl_delta_vs_v3": sum(float(row["net_pnl_usd"]) for row in period_rows)
        - float(EXPECTED_V3_AGGREGATE["net_pnl_usd"]),
        "guard_60": trades >= 60 and sum(int(row["trades"]) > 0 for row in period_rows) >= 5 and not sum(int(row["unresolved"]) for row in period_rows),
        "guard_80": trades >= 80 and sum(int(row["trades"]) > 0 for row in period_rows) >= 5 and not sum(int(row["unresolved"]) for row in period_rows),
        "guard_100": trades >= 100 and sum(int(row["trades"]) > 0 for row in period_rows) >= 5 and not sum(int(row["unresolved"]) for row in period_rows),
    }


def build_neighbor_robustness(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["config_id"]): row for row in aggregate_rows}
    output: list[dict[str, Any]] = []
    for row in aggregate_rows:
        units = tuple(int(round(float(row[f"G{index}"]) / float(matrix.WEIGHT_UNIT))) for index in range(1, 6))
        q = Decimal(str(row["quality_threshold"]))
        weight_ids = [matrix.config_id(item, q) for item in matrix.weight_neighbors(units)]
        quality_ids = [matrix.config_id(units, item) for item in matrix.quality_neighbors(q)]
        ids = sorted(set(weight_ids + quality_ids))
        if any(identifier not in by_id for identifier in ids):
            raise AllPeriodResearchError(f"neighbor absent from complete registry: {row['config_id']}")
        neighbors = [by_id[identifier] for identifier in ids]
        total_r = [float(item["total_r"]) for item in neighbors]
        output.append({
            "config_id": row["config_id"],
            **{f"G{index}": row[f"G{index}"] for index in range(1, 6)},
            "quality_threshold": row["quality_threshold"],
            "weight_neighbor_count": len(weight_ids), "quality_neighbor_count": len(quality_ids),
            "combined_neighbor_count": len(ids),
            "median_neighbor_total_r": statistics.median(total_r),
            "worst_neighbor_total_r": min(total_r),
            "median_neighbor_positive_periods": statistics.median(int(item["positive_periods"]) for item in neighbors),
            "median_neighbor_worst_period_r": statistics.median(float(item["worst_period_r"]) for item in neighbors),
            "median_neighbor_worst_period_drawdown_r": statistics.median(
                float(item["worst_period_max_drawdown_r"]) for item in neighbors
            ),
            "proportion_neighbors_aggregate_positive": sum(value > 0 for value in total_r) / len(total_r),
            "proportion_neighbors_at_least_five_positive_periods": sum(
                int(item["positive_periods"]) >= 5 for item in neighbors
            ) / len(neighbors),
        })
    return output


def plateau_analysis(
    aggregate_rows: Sequence[Mapping[str, Any]], neighbor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    neighbors = {str(row["config_id"]): row for row in neighbor_rows}
    by_id = {str(row["config_id"]): row for row in aggregate_rows}
    eligible = {
        str(row["config_id"]) for row in aggregate_rows
        if bool(row["guard_60"]) and float(row["total_r"]) > 0
        and int(row["positive_periods"]) >= 5 and float(row["worst_period_r"]) > -5.0
        and float(neighbors[str(row["config_id"])]["proportion_neighbors_aggregate_positive"]) > 0.5
    }
    components: list[list[str]] = []
    unseen = set(eligible)
    while unseen:
        seed = min(unseen)
        stack, component = [seed], []
        unseen.remove(seed)
        while stack:
            identifier = stack.pop()
            component.append(identifier)
            row = by_id[identifier]
            units = tuple(int(round(float(row[f"G{index}"]) / 0.05)) for index in range(1, 6))
            adjacent = set(matrix.combined_neighbor_ids(units, Decimal(str(row["quality_threshold"])))) & unseen & eligible
            unseen.difference_update(adjacent)
            stack.extend(sorted(adjacent, reverse=True))
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values[0]))

    def strict(minimum_r: float, positives: int, worst: float) -> list[str]:
        return sorted(
            str(row["config_id"]) for row in aggregate_rows
            if bool(row["guard_60"]) and float(row["total_r"]) >= minimum_r
            and int(row["positive_periods"]) >= positives and float(row["worst_period_r"]) >= worst
        )

    strict_a, strict_b, strict_c = strict(15, 5, -5), strict(20, 5, -4), strict(25, 6, -4)
    return {
        "status": "DESCRIPTIVE_ALL_PERIOD_PLATEAU_ANALYSIS_NO_SELECTION",
        "base_conditions": {
            "minimum_trades": 60, "positive_aggregate_r": True,
            "minimum_positive_periods": 5, "worst_period_r_strictly_greater_than": -5,
            "majority_neighbors_aggregate_positive": True,
        },
        "eligible_configuration_count": len(eligible), "connected_plateau_count": len(components),
        "components": [
            {"plateau_id": f"PLATEAU-{index:04d}", "configuration_count": len(ids), "config_ids": ids}
            for index, ids in enumerate(components, start=1)
        ],
        "strict_views": {
            "A": {"conditions": "R>=15; positive_periods>=5; worst_period>=-5", "count": len(strict_a), "config_ids": strict_a},
            "B": {"conditions": "R>=20; positive_periods>=5; worst_period>=-4", "count": len(strict_b), "config_ids": strict_b},
            "C": {"conditions": "R>=25; positive_periods>=6; worst_period>=-4", "count": len(strict_c), "config_ids": strict_c},
        },
        "automatic_strategy_selection": False, "selected_configuration": None,
    }


def _rankings(
    aggregate_rows: Sequence[Mapping[str, Any]], neighbor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    robust = {str(row["config_id"]): row for row in neighbor_rows}
    eligible = [dict(row) for row in aggregate_rows if bool(row["guard_60"])]
    profitable = [row for row in eligible if float(row["total_r"]) > 0]

    def limited(rows: Iterable[Mapping[str, Any]], key: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in sorted(rows, key=key)[:TOP_LIMIT]]

    with_neighbor = [{**row, **{key: value for key, value in robust[str(row["config_id"])].items() if key not in row}} for row in eligible]
    return {
        "top-total-r.csv": limited(eligible, lambda row: (-float(row["total_r"]), str(row["config_id"]))),
        "top-positive-periods.csv": limited(eligible, lambda row: (-int(row["positive_periods"]), -float(row["total_r"]), str(row["config_id"]))),
        "top-worst-period.csv": limited(eligible, lambda row: (-float(row["worst_period_r"]), -float(row["total_r"]), str(row["config_id"]))),
        "top-median-period.csv": limited(eligible, lambda row: (-float(row["median_period_r"]), -float(row["total_r"]), str(row["config_id"]))),
        "top-pf.csv": limited(eligible, lambda row: (-(_finite(row["aggregate_profit_factor"]) or -math.inf), -float(row["total_r"]), str(row["config_id"]))),
        "top-dd.csv": limited(profitable, lambda row: (abs(float(row["worst_period_max_drawdown_r"])), -float(row["total_r"]), str(row["config_id"]))),
        "top-source-balanced.csv": limited(
            eligible,
            lambda row: (-float(row["source_balance_min_total_r"]), float(row["source_balance_absolute_r_gap"]), -float(row["total_r"]), str(row["config_id"])),
        ),
        "top-neighbor-robustness.csv": limited(
            with_neighbor,
            lambda row: (-float(row["median_neighbor_total_r"]), -float(row["proportion_neighbors_aggregate_positive"]), -float(row["total_r"]), str(row["config_id"])),
        ),
    }


def _weight_grid_rows() -> list[dict[str, Any]]:
    return [
        {
            "weight_id": "W" + "-".join(f"{value:02d}" for value in units),
            **{f"G{index}": value * 0.05 for index, value in enumerate(units, start=1)},
            "sum": 1.0,
        }
        for units in matrix.generate_weight_grid()
    ]


def _report(summary: Mapping[str, Any]) -> str:
    gate = summary["reproduction_gates"]["aggregate"]
    return "\n".join([
        "# All-period L2 POC-only weight x quality research", "",
        f"Status: `{summary['status']}`", "",
        f"Evidence: `{EVIDENCE_LABEL}`. Every period is already-seen retrospective research; no result is OOS.", "",
        "## Integrity", "",
        f"The immutable off-grid V3 reference reproduced {gate['sessions']} sessions, {gate['trades']} trades, "
        f"{gate['wins']} wins / {gate['losses']} losses, {gate['total_r']:.15f}R, and ${gate['net_pnl_usd']:.2f} "
        "before the grid was evaluated.", "",
        "The optimizer used compact Parquet only, maintained independent portfolio chronology inside every period, "
        "and did not concatenate periods into one account.", "",
        "## Research scope", "",
        f"All {summary['configuration_count']:,} predeclared configurations are retained. Ranked views apply the "
        "60-trade / five-period-with-trades / zero-unresolved guard; 80- and 100-trade guards remain descriptive columns.", "",
        "Native-MBP10 and MBO-derived source groups are reported separately. Plateau and neighbor views are descriptive. "
        "No V5 or production configuration was selected.", "",
    ])


def _preflight_html(result: Mapping[str, Any]) -> str:
    rows = []
    for gate in result["period_gates"]:
        expected, actual = gate["expected"], gate["actual"] or {}
        rows.append("<tr>" + "".join(
            f"<td>{html.escape(str(value))}</td>" for value in (
                gate["period_id"], actual.get("gate_source"), expected["trades"],
                actual.get("trades"), expected["total_r"], actual.get("total_r"),
                expected["net_pnl_usd"], actual.get("net_pnl_usd"), gate["status"],
            )
        ) + "</tr>")
    aggregate = result["aggregate_gate"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-period V3 reproduction preflight</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#17202a}}
h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd4dd;padding:7px;text-align:left}}
.pass{{color:#087f5b}}.fail{{color:#b42318}}code{{background:#f3f5f7;padding:2px 4px}}.card{{border:1px solid #d9e0e7;border-radius:8px;padding:16px;margin:16px 0}}
</style></head><body><h1>All-period V3 reproduction preflight</h1>
<p class="{'pass' if result['optimizer_permitted'] else 'fail'}"><strong>{html.escape(str(result['status']))}</strong></p>
<p>Canonical reconciliation hash: <code>{html.escape(str(result['reconciliation_hash']))}</code>.</p>
<table><thead><tr><th>Period</th><th>Gate source</th><th>Expected trades</th><th>Actual trades</th><th>Expected R</th><th>Actual R</th><th>Expected PnL</th><th>Actual PnL</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="card"><h2>Aggregate</h2><p>Status: <strong>{html.escape(str(aggregate['status']))}</strong></p>
<p>Expected: <code>{html.escape(json.dumps(aggregate['expected'], sort_keys=True))}</code></p>
<p>Actual: <code>{html.escape(json.dumps(aggregate['actual'], sort_keys=True))}</code></p></div>
<p>Grid configurations evaluated: <strong>0</strong>. Network calls: <strong>0</strong>. Downloads: <strong>0</strong>.</p>
</body></html>"""


def run_reproduction_preflight(
    *,
    repository_root: Path,
    report_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the hash-bound seven-period baseline without evaluating the grid."""
    repository_root = repository_root.resolve()
    document = load_corrected_baseline_document(repository_root)
    rows = reproduction_rows_from_baseline_document(document)
    result = _evaluate_reproduction_gates(rows)
    result.update({
        "strategy_id": STRATEGY_ID,
        "reconciliation_artifact": str(repository_root / REPRODUCTION_BASELINE_REPORT),
        "reproduction_actual_source": "FROZEN_SOURCE_MODEL_RECONCILIATION_ARTIFACT",
        "configuration_count_evaluated": 0,
        "network_calls": 0,
        "downloads": 0,
    })
    if report_root is not None:
        resolved = report_root if report_root.is_absolute() else repository_root / report_root
        resolved.mkdir(parents=True, exist_ok=True)
        _write_json(resolved / "preflight-summary.json", result)
        temporary = resolved / "preflight-report.html.part"
        temporary.write_text(_preflight_html(result), encoding="utf-8")
        temporary.replace(resolved / "preflight-report.html")
        result["report_root"] = str(resolved)
    return result


def require_reproduction_preflight(*, repository_root: Path) -> dict[str, Any]:
    result = run_reproduction_preflight(repository_root=repository_root)
    if not result["optimizer_permitted"]:
        for gate in result["period_gates"]:
            if gate["status"] != "PASS":
                raise AllPeriodResearchError(f"V3_REPRODUCTION_GATE_FAILED:{gate['period_id']}")
        raise AllPeriodResearchError("V3_AGGREGATE_REPRODUCTION_FAILED")
    return result


def run_optimizer(*, repository_root: Path, tape_root: Path, output_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = (repository_root / tape_root).resolve() if not tape_root.is_absolute() else tape_root.resolve()
    output_root = (repository_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    reproduction = require_reproduction_preflight(repository_root=repository_root)
    if output_root.exists():
        raise FileExistsError(f"immutable all-period output exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists():
        raise FileExistsError(f"unverified all-period staging exists: {staging}")
    inventory = build_source_inventory(repository_root, tape_root)
    missing = [row["period_id"] for row in inventory["periods"] if not row["compact_tape_reusable"]]
    if missing:
        raise AllPeriodResearchError(f"MISSING_CAUSAL_PERIOD_TAPES:{missing}")
    dec_jan_validation = master.validate_building_root(repository_root / DEC_JAN_MASTER)
    bundles = _period_bundles(repository_root, tape_root)
    print("ALL_SEVEN_V3_REPRODUCTION_GATES=PASS", flush=True)

    started = time.monotonic()
    period_rows: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles, start=1):
        print(f"ALL_PERIOD_OPTIMIZER {index}/7 {bundle.period.period_id}", flush=True)
        period_rows.extend(_evaluate_bundle(bundle))
    if len(period_rows) != EXPECTED_CONFIGURATION_COUNT * len(PERIODS):
        raise AllPeriodResearchError("configuration x period output cardinality mismatch")
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in period_rows:
        by_config[str(row["config_id"])].append(row)
    aggregate_rows = [aggregate_configuration_periods(by_config[identifier]) for identifier in sorted(by_config)]
    if len(aggregate_rows) != EXPECTED_CONFIGURATION_COUNT:
        raise AllPeriodResearchError("aggregate output does not retain exactly 23,256 configurations")
    neighbor_rows = build_neighbor_robustness(aggregate_rows)
    source_rows = [{
        key: row[key] for key in (
            "config_id", "G1", "G2", "G3", "G4", "G5", "quality_threshold",
            "native_mbp10_total_r", "native_mbp10_trade_count", "mbo_derived_total_r",
            "mbo_derived_trade_count", "source_model_classification", "source_balance_min_total_r",
            "source_balance_absolute_r_gap",
        )
    } for row in aggregate_rows]
    plateau = plateau_analysis(aggregate_rows, neighbor_rows)
    rankings = _rankings(aggregate_rows, neighbor_rows)

    staging.mkdir(parents=True)
    _write_json(staging / "source-inventory.json", inventory)
    _write_json(staging / "reproduction-gates.json", reproduction)
    _write_json(staging / "period-tape-summary.json", {
        "periods": [{"period_id": bundle.period.period_id, "root": str(bundle.root), "sessions": len(bundle.days),
                     "source_model": bundle.period.source_model} for bundle in bundles],
        "session_count": EXPECTED_SESSION_COUNT,
        "existing_dec_jan_master_validation": dec_jan_validation,
    })
    _write_csv(staging / "weight-grid.csv", _weight_grid_rows())
    _write_parquet(staging / "weight-q-period-results.parquet", period_rows)
    _write_csv(staging / "weight-q-aggregate-results.csv", aggregate_rows)
    _write_csv(staging / "neighbor-robustness.csv", neighbor_rows)
    _write_csv(staging / "source-model-robustness.csv", source_rows)
    for filename, rows in rankings.items():
        _write_csv(staging / filename, rows)
    _write_json(staging / "plateau-analysis.json", plateau)
    summary = {
        "status": "ALL_PERIOD_WEIGHT_Q_RESEARCH_COMPLETE_NO_SELECTION",
        "strategy_id": STRATEGY_ID, "evidence_label": EVIDENCE_LABEL,
        "period_count": len(PERIODS), "session_count": EXPECTED_SESSION_COUNT,
        "weight_count": matrix.EXPECTED_WEIGHT_COUNT,
        "quality_thresholds": [float(value) for value in matrix.QUALITY_THRESHOLDS],
        "configuration_count": EXPECTED_CONFIGURATION_COUNT,
        "configuration_period_rows": len(period_rows),
        "minimum_sample_guards": {"ranked": 60, "strict_descriptive": [80, 100], "periods_with_trades": 5, "unresolved": 0},
        "reproduction_gates": reproduction,
        "immutable_v3_reference": EXPECTED_V3_AGGREGATE,
        "v3_weights_on_grid": False,
        "runtime_seconds": time.monotonic() - started,
        "dbn_files_opened_by_optimizer": 0, "network_calls": 0, "downloads": 0,
        "automatic_strategy_selection": False, "selected_configuration": None,
    }
    _write_json(staging / "summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_report(summary), encoding="utf-8")
    os.rename(staging, output_root)
    return {**summary, "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "inventory", "build-period", "build-missing-tapes", "preflight-reproduction", "optimize",
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--tape-root", type=Path, default=TAPE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-output-root", type=Path, default=REPRODUCTION_PREFLIGHT_ROOT)
    parser.add_argument("--period", choices=tuple(
        period.period_id for period in PERIODS if period.reusable_master is None
    ))
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = build_source_inventory(args.repository_root, args.tape_root)
        elif args.command == "build-period":
            if args.period is None:
                raise AllPeriodResearchError("--period is required for build-period")
            result = build_period_tape(
                repository_root=args.repository_root, tape_root=args.tape_root, period_id=args.period,
            )
        elif args.command == "build-missing-tapes":
            result = build_missing_tapes(repository_root=args.repository_root, tape_root=args.tape_root)
        elif args.command == "preflight-reproduction":
            result = run_reproduction_preflight(
                repository_root=args.repository_root,
                report_root=args.preflight_output_root,
            )
        else:
            result = run_optimizer(
                repository_root=args.repository_root, tape_root=args.tape_root,
                output_root=args.output_root,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("optimizer_permitted", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
