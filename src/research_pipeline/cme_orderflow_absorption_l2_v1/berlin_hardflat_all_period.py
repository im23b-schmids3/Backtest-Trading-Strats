"""Corrected Berlin-hard-flat audit, baseline, preflight, and Weight x Q runner.

This module is deliberately versioned away from historical V3.  It consumes
only the already-published compact causal Parquet tapes and never imports a
Databento client or opens a DBN file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import all_period_weight_q_research as allp
from . import causal_master_tape as master
from . import historical_runner as historical
from . import weight_q_research as matrix
from .berlin_hardflat_execution import (
    CONTRACT_SHA256,
    EXECUTION_CONTRACT,
    HISTORICAL_V3_CONTRACT_SHA256,
    MAX_EXECUTABLE_BBO_GAP_NS,
    SEMANTIC_VERSION,
    STRATEGY_ID,
    BerlinExecutionError,
    BerlinSessionCausalTape,
    OpenPositionAtMaintenance,
    UnpricedSourceIntegrityFailure,
    berlin_hard_flat_datetime,
    berlin_hard_flat_utc,
    maintenance_window_utc,
    simulate_berlin_session,
)


UTC = timezone.utc
EVIDENCE_LABEL = "ALL_PERIOD_RETROSPECTIVE_EXECUTION_CORRECTION_NOT_OOS_EVIDENCE"
TAPE_ROOT = allp.TAPE_ROOT
BASELINE_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT_ALL_PERIOD"
)
TRAIN_DATE_SUBSET_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_BERLIN_HARDFLAT_TRAIN_DATE_SUBSET"
)
PREFLIGHT_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_BERLIN_HARDFLAT_SEMANTIC_PREFLIGHT"
)
AUDIT_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_BERLIN_HARDFLAT_CONTRACT_AUDIT"
)
OPTIMIZER_ROOT = Path(
    "research_runs/CMEOrderflowAbsorption.ES_L2_ALL_PERIOD_WEIGHT_Q_RESEARCH_BERLIN_HARDFLAT"
)
HISTORICAL_BOUNDARY_AUDIT = (
    TAPE_ROOT / "durable-boundary-reconciliation.json"
)
EXPECTED_AFFECTED_PATHS = 372
EXPECTED_GRID_COUNT = 23_256

COMPARISON_CONFIG_IDS = (
    "W04-02-06-04-04-Q45",
    "W04-02-02-04-08-Q50",
    "W03-02-08-03-04-Q45",
    "W05-01-01-05-08-Q50",
)
BERLIN_BASELINE_CONFIG_ID = "CORRECTED_BERLIN_V3_BASELINE"
BERLIN_V3_WEIGHTS = {
    "G1": 0.28, "G2": 0.25, "G3": 0.22, "G4": 0.12, "G5": 0.13,
    "quality_threshold": 0.50,
}
EXPECTED_COMPARISON_AGGREGATES = {
    "W04-02-06-04-04-Q45": (92, 26.584605336640003, 6323.50),
    "W04-02-02-04-08-Q50": (88, 26.721885295430262, 6373.25),
    "W03-02-08-03-04-Q45": (90, 25.130059882094546, 5978.50),
    "W05-01-01-05-08-Q50": (86, 27.43851773896209, 6590.75),
}
REPORTING_FIELD_NAMES = {
    "v3_reference_trades", "v3_reference_total_r", "v3_reference_net_pnl_usd",
    "trade_delta_vs_v3", "total_r_delta_vs_v3", "net_pnl_delta_vs_v3",
    "historical_v3_reference_trades", "historical_v3_reference_total_r",
    "historical_v3_reference_net_pnl_usd", "trade_delta_vs_historical_v3",
    "total_r_delta_vs_historical_v3", "net_pnl_delta_vs_historical_v3",
    "berlin_v3_reference_trades", "berlin_v3_reference_total_r",
    "berlin_v3_reference_net_pnl_usd", "trade_delta_vs_berlin_v3",
    "total_r_delta_vs_berlin_v3", "net_pnl_delta_vs_berlin_v3",
}

SEMANTIC_CLASSIFICATIONS = (
    "A_SAME_ENTRY_SAME_EXIT",
    "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT",
    "C_SAME_ENTRY_DIFFERENT_EXIT_DUE_3S_DATA_GAP",
    "D_SAME_ENTRY_DIFFERENT_EXIT_DUE_SOURCE_END",
    "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT",
    "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON",
    "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON",
    "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN",
)

# These are immutable, already-published historical V3 ledgers.  The May
# setup/trade files predate the separately sealed MBO public-book correction;
# run_preflight therefore restricts them to the canonical accepted setup IDs
# stored in the corrected baseline rather than reviving the superseded row.
HISTORICAL_LEDGER_SOURCES: dict[str, tuple[Path, Path]] = {
    "APRIL_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_APR06_08_RETRO/setup-ledger.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_APR06_08_RETRO/trade-ledger.csv"),
    ),
    "MAY_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/may_2026/v3-setups.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/may_2026/v3-trades.csv"),
    ),
    "RETRO_JUNE_JULY_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/retro_june_july_2026/v3-setups.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/retro_june_july_2026/v3-trades.csv"),
    ),
    "AUGUST_03_06_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/august_03_06_2026/v3-setups.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD/periods/august_03_06_2026/v3-trades.csv"),
    ),
    "AUGUST_10_14_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_AUG10_14_FRESH/setup-ledger.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_AUG10_14_FRESH/trade-ledger.csv"),
    ),
    "DECEMBER_2025": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO/setup-ledger.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO/trade-ledger.csv"),
    ),
    "JANUARY_2026": (
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO/setup-ledger.csv"),
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO/trade-ledger.csv"),
    ),
}


class CorrectedAllPeriodError(RuntimeError):
    pass


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedAllPeriodError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorrectedAllPeriodError(f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise CorrectedAllPeriodError(f"missing historical ledger: {path}") from exc


def _setup_key(row: Mapping[str, Any]) -> str:
    day = str(row.get("date") or row.get("session_date") or "")
    interaction = str(
        row.get("source_interaction_id") or row.get("interaction_id") or ""
    )
    if "|" in interaction:
        embedded_day, interaction = interaction.split("|", 1)
        day = day or embedded_day
    if not day or not interaction:
        raise CorrectedAllPeriodError(f"setup identity is incomplete: {row}")
    return f"{day}|{interaction}"


def _period_contains_day(period_id: str, day: str) -> bool:
    if period_id == "DECEMBER_2025":
        return day.startswith("2025-12")
    if period_id == "JANUARY_2026":
        return day.startswith("2026-01")
    return True


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _equal_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _unique_by_setup(
    rows: Iterable[Mapping[str, Any]], *, artifact: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = _setup_key(row)
        if key in result:
            raise CorrectedAllPeriodError(f"duplicate setup in {artifact}: {key}")
        result[key] = row
    return result


def _historical_ledgers(
    repository_root: Path, period_id: str, canonical_setup_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        setup_relative, trade_relative = HISTORICAL_LEDGER_SOURCES[period_id]
    except KeyError as exc:
        raise CorrectedAllPeriodError(f"historical ledger source is undeclared: {period_id}") from exc
    setup_path = repository_root / setup_relative
    trade_path = repository_root / trade_relative
    setup_rows = [
        row for row in _read_csv(setup_path)
        if _period_contains_day(period_id, str(row.get("date", "")))
        and _truthy(row.get("accepted"))
        and _setup_key(row) in canonical_setup_ids
    ]
    trade_rows = [
        row for row in _read_csv(trade_path)
        if _period_contains_day(period_id, str(row.get("date", "")))
        and _setup_key(row) in canonical_setup_ids
    ]
    setups = _unique_by_setup(setup_rows, artifact=str(setup_path))
    trades = _unique_by_setup(trade_rows, artifact=str(trade_path))
    if set(setups) != canonical_setup_ids:
        missing = sorted(canonical_setup_ids - set(setups))
        extra = sorted(set(setups) - canonical_setup_ids)
        raise CorrectedAllPeriodError(
            f"historical setup ledger does not cover canonical signal universe "
            f"for {period_id}: missing={missing} extra={extra}"
        )
    return setups, trades, {
        "setup_ledger": str(setup_path.resolve()),
        "setup_ledger_sha256": _sha256(setup_path),
        "trade_ledger": str(trade_path.resolve()),
        "trade_ledger_sha256": _sha256(trade_path),
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary, compression="zstd")
    temporary.replace(path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _iso_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, tz=UTC).isoformat()


def _optional_iso_ns(value: Any) -> str | None:
    return None if value in (None, "") else _iso_ns(int(value))


def execution_local_iso(day: str) -> str:
    return berlin_hard_flat_datetime(day).isoformat()


def _tape_path(bundle: allp.PeriodBundle, day: str) -> Path:
    return bundle.root / "causal-event-tape" / f"{day}.parquet"


def _terminal_event_from_parquet(path: Path) -> dict[str, Any]:
    """Read only the final row group needed to identify the sealed terminal."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups < 1:
        raise CorrectedAllPeriodError(f"empty compact tape: {path}")
    table = parquet.read_row_group(
        parquet.num_row_groups - 1,
        columns=("event_ordinal", "timestamp_ns", "event_type", "hard_flat_reason"),
    )
    for row in reversed(table.to_pylist()):
        if row["event_type"] in {"HARD_FLAT", "SOURCE_END"}:
            return row
    raise CorrectedAllPeriodError(f"compact tape terminal absent from final row group: {path}")


def _tape_binding(bundle: allp.PeriodBundle) -> dict[str, Any]:
    files = [
        bundle.root / "interaction-master.parquet",
        bundle.root / "interaction-event-index.parquet",
        *[_tape_path(bundle, day) for day in bundle.days],
    ]
    records = []
    for path in files:
        if not path.is_file():
            raise CorrectedAllPeriodError(f"compact tape input missing: {path}")
        records.append({
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path),
        })
    return {
        "period_id": bundle.period.period_id,
        "files": records,
        "tape_hash": _canonical_hash(records),
    }


def _grid_hash() -> str:
    registry = [
        {"weights": list(weights), "quality_threshold": str(threshold)}
        for weights, threshold in matrix.configuration_registry()
    ]
    if len(registry) != EXPECTED_GRID_COUNT:
        raise CorrectedAllPeriodError("weight grid no longer contains exactly 23,256 configurations")
    return _canonical_hash(registry)


def _code_hash() -> str:
    directory = Path(__file__).resolve().parent
    inputs = (
        directory / "berlin_hardflat_execution.py",
        Path(__file__).resolve(),
        directory / "weight_q_research.py",
        directory / "all_period_weight_q_research.py",
        directory / "causal_master_tape.py",
        directory / "model.py",
    )
    return _canonical_hash([{"name": path.name, "sha256": _sha256(path)} for path in inputs])


@dataclass
class CorrectedAccumulator(matrix.ConfigurationAccumulator):
    hard_flat_berlin_exits: int = 0
    data_gap_3s_force_flat_exits: int = 0
    source_end_force_flat_exits: int = 0
    integrity_failures: int = 0

    def add(self, session: matrix.SessionResult, baseline_trade_ids: set[str]) -> None:
        super().add(session, baseline_trade_ids)
        reasons = Counter(str(row["exit_reason"]) for row in session.trades)
        self.hard_flat_berlin_exits += reasons["HARD_FLAT_BERLIN"]
        self.data_gap_3s_force_flat_exits += reasons["DATA_GAP_3S_FORCE_FLAT"]
        self.source_end_force_flat_exits += reasons["SOURCE_END_FORCE_FLAT_LAST_VALID_BBO"]

    def row(self, baseline_count: int) -> dict[str, Any]:
        row = super().row(baseline_count)
        row.update({
            "hard_flat_berlin_exits": self.hard_flat_berlin_exits,
            "data_gap_3s_force_flat_exits": self.data_gap_3s_force_flat_exits,
            "source_end_force_flat_exits": self.source_end_force_flat_exits,
            "integrity_failures": self.integrity_failures,
            "unresolved": 0,
        })
        return row


def _period_inputs(
    bundle: allp.PeriodBundle,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    interactions, indexes = allp._load_bundle_rows(bundle)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_day[str(row["session_date"])].append(row)
    return interactions, indexes, by_day


def _simulate_v3_period(bundle: allp.PeriodBundle) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    interactions, indexes, by_day = _period_inputs(bundle)
    corrected = CorrectedAccumulator((1, 1, 1, 1, 16), Decimal("0.50"))
    historical_accumulator = matrix.ConfigurationAccumulator((1, 1, 1, 1, 16), Decimal("0.50"))
    corrected_trades: list[dict[str, Any]] = []
    old_trades: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    old_terminal: dict[str, str] = {}
    corrected_terminal: dict[str, str] = {}
    for number, day in enumerate(bundle.days, start=1):
        print(f"BERLIN_BASELINE {bundle.period.period_id} {number:02d}/{len(bundle.days):02d} {day}", flush=True)
        rows = [
            row for row in by_day.get(day, ())
            if master.interaction_is_accepted(row, threshold="0.50")
        ]
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        old_tape = matrix.SessionCausalTape.from_parquet(day, _tape_path(bundle, day))
        old = matrix.simulate_independent_session(old_tape, rows, day_indexes)
        historical_accumulator.add(old, set())
        old_trades.extend(old.trades)
        old_terminal.update(old.terminal_outcomes)
        tape = BerlinSessionCausalTape.from_parquet(day, _tape_path(bundle, day))
        session = simulate_berlin_session(tape, rows, day_indexes)
        corrected.add(session, set())
        corrected_trades.extend(session.trades)
        corrected_terminal.update(session.terminal_outcomes)
        accepted_ids.extend(str(row["interaction_id"]) for row in rows)
    performance = historical._performance(corrected_trades)
    exit_counts = Counter(str(row["exit_reason"]) for row in corrected_trades)
    row = {
        "period_id": bundle.period.period_id,
        "source_model": bundle.period.source_model,
        "source_group": bundle.period.source_group,
        "sessions": len(bundle.days),
        "completed_interactions": len(interactions),
        "accepted_setups": len(accepted_ids),
        "confirmations": corrected.confirmations,
        **performance,
        "hard_flat_berlin_exits": exit_counts["HARD_FLAT_BERLIN"],
        "data_gap_3s_force_flat_exits": exit_counts["DATA_GAP_3S_FORCE_FLAT"],
        "source_end_force_flat_last_valid_bbo_exits": exit_counts[
            "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO"
        ],
        "unresolved": corrected.unresolved,
        "integrity_failures": corrected.integrity_failures,
        "historical_v3_reference": dict(allp.EXPECTED_V3[bundle.period.period_id]),
    }
    audit = {
        "accepted_setup_ids": sorted(accepted_ids),
        "old_terminal_outcomes": old_terminal,
        "corrected_terminal_outcomes": corrected_terminal,
        "old_trades": old_trades,
        "corrected_trades": corrected_trades,
        "historical_observed_unresolved": historical_accumulator.unresolved,
    }
    return row, corrected_trades, audit


def _simulate_v3_day(
    bundle: allp.PeriodBundle,
    day: str,
    by_day: Mapping[str, Sequence[Mapping[str, Any]]],
    indexes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Replay one compact causal day through the existing Berlin engine.

    This is deliberately a thin date-filtered entry point over the same
    ``BerlinSessionCausalTape`` and ``simulate_berlin_session`` functions used
    by the immutable all-period baseline.  It exists only to materialize
    missing zero/low-activity date artifacts; it does not introduce a second
    signal or execution implementation.
    """
    candidates = [
        row for row in by_day.get(day, ())
        if master.interaction_is_accepted(row, threshold="0.50")
    ]
    day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in candidates}
    tape = BerlinSessionCausalTape.from_parquet(day, _tape_path(bundle, day))
    session = simulate_berlin_session(tape, candidates, day_indexes)
    trades = [dict(row) for row in session.trades]
    performance = historical._performance(trades)
    outcomes = {str(key): str(value) for key, value in session.terminal_outcomes.items()}
    exit_counts = Counter(str(row["exit_reason"]) for row in trades)
    metrics = {
        "period_id": bundle.period.period_id,
        "session_date": day,
        "source_model": bundle.period.source_model,
        "source_group": bundle.period.source_group,
        "completed_interactions": len(by_day.get(day, ())),
        "accepted_setups": len(candidates),
        "confirmations": int(session.confirmations),
        "hard_flat_berlin_exits": exit_counts["HARD_FLAT_BERLIN"],
        "data_gap_3s_force_flat_exits": exit_counts["DATA_GAP_3S_FORCE_FLAT"],
        "source_end_force_flat_last_valid_bbo_exits": exit_counts[
            "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO"
        ],
        "unresolved": int(session.unresolved),
        "integrity_failures": 0,
        **performance,
    }
    audit = {
        "accepted_setup_ids": sorted(str(row["interaction_id"]) for row in candidates),
        "corrected_terminal_outcomes": outcomes,
        "corrected_trades": trades,
        "source_event_tape": str(_tape_path(bundle, day).resolve()),
    }
    return metrics, trades, audit


def run_date_subset(
    *,
    repository_root: Path,
    tape_root: Path = TAPE_ROOT,
    dates: Sequence[str],
    output_root: Path = TRAIN_DATE_SUBSET_ROOT,
) -> dict[str, Any]:
    """Materialize an immutable exact-date Berlin artifact supplement.

    Only Dec/Jan dates backed by the already-built causal master are allowed.
    The requested set must be unique, sorted deterministically for output, and
    no unrequested session is replayed or published.
    """
    requested = tuple(str(day) for day in dates)
    if not requested or len(requested) != len(set(requested)):
        raise CorrectedAllPeriodError("date subset must be non-empty and duplicate-free")
    if tuple(sorted(requested)) != requested:
        raise CorrectedAllPeriodError("date subset must be supplied in chronological order")
    repository_root = repository_root.resolve()
    output_root = _resolve(repository_root, output_root)
    if output_root.exists() or output_root.with_name(output_root.name + ".building").exists():
        raise FileExistsError(f"immutable date-subset output exists or is staged: {output_root}")
    bundles = {
        bundle.period.period_id: bundle
        for bundle in allp._period_bundles(repository_root, _resolve(repository_root, tape_root))
    }
    bundle_by_day: dict[str, allp.PeriodBundle] = {}
    for period_id in ("DECEMBER_2025", "JANUARY_2026"):
        bundle = bundles[period_id]
        for day in bundle.days:
            bundle_by_day[day] = bundle
    unknown = [day for day in requested if day not in bundle_by_day]
    if unknown:
        raise CorrectedAllPeriodError(f"date subset is outside Dec/Jan causal master: {unknown}")

    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True)
    session_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    try:
        # Load each period once, then replay only the requested dates.
        period_inputs: dict[str, tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]] = {}
        for period_id, bundle in bundles.items():
            selected = [day for day in requested if bundle_by_day[day].period.period_id == period_id]
            if not selected:
                continue
            _, indexes, by_day = _period_inputs(bundle)
            period_inputs[period_id] = (indexes, by_day)
        for day in requested:
            bundle = bundle_by_day[day]
            indexes, by_day = period_inputs[bundle.period.period_id]
            metrics, trades, audit = _simulate_v3_day(bundle, day, by_day, indexes)
            _write_json(staging / "sessions" / f"{day}.json", {
                "metrics": metrics,
                "trades": trades,
                "semantic_audit": audit,
            })
            session_rows.append(metrics)
            all_trades.extend(trades)
        manifest = {
            "status": "BERLIN_TRAIN_DATE_SUBSET_COMPLETE",
            "strategy_id": STRATEGY_ID,
            "execution_contract_sha256": CONTRACT_SHA256,
            "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
            "evidence_label": EVIDENCE_LABEL,
            "source_model": "EXISTING_CAUSAL_MASTER_UNDERLYING_NATIVE_MBP10",
            "dates": list(requested),
            "session_count": len(session_rows),
            "trades": len(all_trades),
            "network_calls": 0,
            "downloads": 0,
            "dbn_files_opened": 0,
            "source_tape_files": [
                {"date": day, "path": str(_tape_path(bundle_by_day[day], day).resolve()),
                 "sha256": _sha256(_tape_path(bundle_by_day[day], day))}
                for day in requested
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output_root)
        return {**manifest, "output_root": str(output_root), "sessions": session_rows, "trades": all_trades}
    except Exception:
        raise


def load_date_subset(repository_root: Path, output_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    """Load and validate an exact-date artifact supplement for Block 1."""
    root = _resolve(repository_root.resolve(), output_root)
    manifest = _read_json(root / "manifest.json")
    requested = tuple(str(day) for day in dates)
    if manifest.get("status") != "BERLIN_TRAIN_DATE_SUBSET_COMPLETE":
        raise CorrectedAllPeriodError("Berlin date-subset manifest status invalid")
    if tuple(manifest.get("dates", ())) != requested:
        raise CorrectedAllPeriodError("Berlin date-subset dates do not match requested dates")
    if manifest.get("execution_contract_sha256") != CONTRACT_SHA256:
        raise CorrectedAllPeriodError("Berlin date-subset execution contract hash mismatch")
    sessions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for day in requested:
        payload = _read_json(root / "sessions" / f"{day}.json")
        metrics = payload.get("metrics")
        day_trades = payload.get("trades")
        if not isinstance(metrics, dict) or str(metrics.get("session_date")) != day or not isinstance(day_trades, list):
            raise CorrectedAllPeriodError(f"Berlin date-subset session artifact invalid: {day}")
        sessions.append(dict(metrics))
        trades.extend(dict(row) for row in day_trades)
    if len(sessions) != len(requested) or len({str(row["session_date"]) for row in sessions}) != len(requested):
        raise CorrectedAllPeriodError("Berlin date-subset session identity mismatch")
    return {
        "status": "PASS",
        "execution_mode": "REUSED_CORRECTED_BERLIN_DATE_SUBSET_ARTIFACT",
        "strategy_id": STRATEGY_ID,
        "contract_hash": CONTRACT_SHA256,
        "sessions": sessions,
        "trades": trades,
        "source_period_artifact_hashes": {"date_subset_manifest.json": _sha256(root / "manifest.json")},
    }


def _aggregate_baseline(periods: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_trades = [dict(trade) for row in periods for trade in row.get("_trades", ())]
    performance = historical._performance(all_trades)
    trades = int(performance["completed_trades"])
    wins = int(performance["wins"])
    losses = int(performance["losses"])
    clean = {key: value for key, value in {
        "sessions": sum(int(row["sessions"]) for row in periods),
        "trades": trades, "wins": wins, "losses": losses,
        "win_rate": wins / trades if trades else 0.0,
        "total_r": performance["total_r"],
        "net_pnl_usd": performance["net_pnl_usd"],
        "profit_factor": performance["profit_factor"],
        "max_cumulative_drawdown_r": performance["max_cumulative_drawdown_r"],
        "es_trades": sum(int(row["es_trades"]) for row in periods),
        "mes_trades": sum(int(row["mes_trades"]) for row in periods),
        "target_exits": sum(int(row["target_exits"]) for row in periods),
        "stop_exits": sum(int(row["stop_exits"]) for row in periods),
        "hard_flat_berlin_exits": sum(int(row["hard_flat_berlin_exits"]) for row in periods),
        "data_gap_3s_force_flat_exits": sum(int(row["data_gap_3s_force_flat_exits"]) for row in periods),
        "source_end_force_flat_last_valid_bbo_exits": sum(
            int(row["source_end_force_flat_last_valid_bbo_exits"]) for row in periods
        ),
        "unresolved": sum(int(row["unresolved"]) for row in periods),
        "integrity_failures": sum(int(row["integrity_failures"]) for row in periods),
    }.items()}
    return clean


def run_baseline(*, repository_root: Path, tape_root: Path, output_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = _resolve(repository_root, tape_root)
    output_root = _resolve(repository_root, output_root)
    if output_root.exists() or output_root.with_name(output_root.name + ".building").exists():
        raise FileExistsError(f"immutable corrected baseline output exists or is staged: {output_root}")
    bundles = allp._period_bundles(repository_root, tape_root)
    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True)
    period_rows: list[dict[str, Any]] = []
    try:
        for bundle in bundles:
            row, trades, audit = _simulate_v3_period(bundle)
            row["_trades"] = trades
            period_rows.append(row)
            _write_json(staging / "periods" / f"{bundle.period.period_id}.json", {
                "metrics": {key: value for key, value in row.items() if key != "_trades"},
                "trades": trades, "semantic_audit": audit,
            })
        aggregate = _aggregate_baseline(period_rows)
        if aggregate["sessions"] != allp.EXPECTED_SESSION_COUNT or aggregate["unresolved"] != 0:
            raise CorrectedAllPeriodError("corrected baseline session/unresolved invariant failed")
        summary = {
            "status": "CORRECTED_BERLIN_HARDFLAT_BASELINE_COMPLETE",
            "strategy_id": STRATEGY_ID,
            "execution_contract_sha256": CONTRACT_SHA256,
            "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
            "evidence_label": EVIDENCE_LABEL,
            "periods": [{key: value for key, value in row.items() if key != "_trades"} for row in period_rows],
            "aggregate": aggregate,
            "historical_v3_reference": dict(allp.EXPECTED_V3_AGGREGATE),
            "network_calls": 0, "downloads": 0, "dbn_files_opened": 0,
        }
        _write_json(staging / "execution-contract.json", EXECUTION_CONTRACT)
        _write_json(staging / "summary.json", summary)
        (staging / "report.html").write_text(_baseline_html(summary), encoding="utf-8")
        os.rename(staging, output_root)
        return {**summary, "output_root": str(output_root)}
    except Exception:
        # Preserve staging for diagnosis; immutable final remains unpublished.
        raise


def _baseline_html(summary: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key)))}</td>" for key in (
            "period_id", "sessions", "completed_trades", "wins", "losses", "total_r",
            "net_pnl_usd", "profit_factor", "hard_flat_berlin_exits",
            "data_gap_3s_force_flat_exits", "source_end_force_flat_last_valid_bbo_exits", "unresolved",
        )) + "</tr>"
        for row in summary["periods"]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Corrected baseline</title>
<style>body{{font:14px system-ui;margin:2rem;color:#18202a}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:.35rem}}code{{background:#eef;padding:.15rem}}</style></head>
<body><h1>{html.escape(STRATEGY_ID)}</h1><p>Execution contract <code>{CONTRACT_SHA256}</code>. Historical V3 remains a separate reference.</p>
<table><thead><tr><th>period</th><th>sessions</th><th>trades</th><th>wins</th><th>losses</th><th>R</th><th>PnL</th><th>PF</th><th>Berlin flat</th><th>gap flat</th><th>source flat</th><th>unresolved</th></tr></thead><tbody>{rows}</tbody></table>
<p>No automatic selection. No network or DBN input.</p></body></html>"""


def _same_entry_geometry(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    exact = ("trade_id", "setup_id", "direction", "instrument")
    integers = ("contracts", "entry_timestamp_ns")
    numeric = ("entry", "stop", "target", "total_costs_usd")
    return (
        all(str(old.get(key)) == str(new.get(key)) for key in exact)
        and all(int(old[key]) == int(new[key]) for key in integers)
        and all(_equal_number(old.get(key), new.get(key)) for key in numeric)
    )


def _same_exit_geometry(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    exact = ("exit_reason",)
    integers = ("exit_timestamp_ns",)
    numeric = ("exit", "net_pnl_usd", "r_multiple")
    return (
        all(str(old.get(key)) == str(new.get(key)) for key in exact)
        and all(int(old[key]) == int(new[key]) for key in integers)
        and all(_equal_number(old.get(key), new.get(key)) for key in numeric)
    )


def _same_trade_geometry(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    return _same_entry_geometry(old, new) and _same_exit_geometry(old, new)


def _historical_confirmation(row: Mapping[str, Any]) -> tuple[int | None, float | None]:
    return _optional_int(row.get("confirmation_timestamp_ns")), _optional_float(
        row.get("confirmation_price")
    )


def _indexed_confirmation(row: Mapping[str, Any]) -> tuple[int | None, float | None]:
    return _optional_int(row.get("derived_first_confirmation_timestamp_ns")), _optional_float(
        row.get("derived_first_confirmation_price")
    )


def _confirmation_equal(
    old: tuple[int | None, float | None], new: tuple[int | None, float | None],
) -> bool:
    return old[0] == new[0] and _equal_number(old[1], new[1])


def _active_setup_at(
    trades: Mapping[str, Mapping[str, Any]], timestamp_ns: int, *, exclude: str,
) -> str | None:
    active = [
        key for key, trade in trades.items()
        if key != exclude
        and int(trade["entry_timestamp_ns"]) <= timestamp_ns < int(trade["exit_timestamp_ns"])
    ]
    if len(active) > 1:
        raise CorrectedAllPeriodError(
            f"one-position invariant violated while tracing semantic diff: {active}"
        )
    return active[0] if active else None


def _blocking_predecessor(
    *, membership: str, setup_key: str, setup_timestamp_ns: int,
    historical_trades: Mapping[str, Mapping[str, Any]],
    corrected_trades: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    allowed = {
        "HARD_FLAT_BERLIN", "DATA_GAP_3S_FORCE_FLAT",
        "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
    }
    for predecessor in sorted(set(historical_trades) & set(corrected_trades)):
        if predecessor == setup_key:
            continue
        old = historical_trades[predecessor]
        new = corrected_trades[predecessor]
        if not _same_entry_geometry(old, new) or str(new.get("exit_reason")) not in allowed:
            continue
        old_exit = int(old["exit_timestamp_ns"])
        new_exit = int(new["exit_timestamp_ns"])
        if membership == "HISTORICAL_ONLY" and old_exit <= setup_timestamp_ns < new_exit:
            return predecessor, (
                f"CORRECTED_PREDECESSOR_REMAINED_OPEN:{new['exit_reason']}:"
                f"old_exit={old_exit}:corrected_exit={new_exit}:setup={setup_timestamp_ns}"
            )
        if membership == "CORRECTED_ONLY" and new_exit <= setup_timestamp_ns < old_exit:
            return predecessor, (
                f"CORRECTED_PREDECESSOR_EXITED_EARLIER:{new['exit_reason']}:"
                f"old_exit={old_exit}:corrected_exit={new_exit}:setup={setup_timestamp_ns}"
            )
    return None, None


def _hard_flat_cancellation_evidence(
    *, day: str, historical_trade: Mapping[str, Any] | None,
    corrected_trade: Mapping[str, Any] | None, corrected_terminal: str | None,
    signal_unchanged: bool, confirmation_unchanged: bool,
    evidence_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the strict, auditable evidence for semantic category H.

    A corrected terminal cancellation is sufficient evidence that the simulator
    cancelled its working order only when there is no corrected fill.  Optional
    overrides exist solely so synthetic tests can prove each fail-closed clause;
    production preflight never supplies them.
    """
    hard_flat_ns = int(berlin_hard_flat_utc(day).timestamp() * 1e9)
    historical_entry_ns = (
        int(historical_trade["entry_timestamp_ns"])
        if historical_trade is not None else None
    )
    exact_terminal = corrected_terminal == "CANCELLED_AT_HARD_FLAT_BERLIN"
    no_corrected_fill = corrected_trade is None
    evidence: dict[str, Any] = {
        "signal_unchanged": signal_unchanged,
        "confirmation_unchanged": confirmation_unchanged,
        "historical_entry_at_or_after_hard_flat": (
            historical_entry_ns is not None and historical_entry_ns >= hard_flat_ns
        ),
        "corrected_terminal_is_hard_flat_cancellation": exact_terminal,
        "corrected_fill_occurred": not no_corrected_fill,
        "corrected_position_created": not no_corrected_fill,
        "corrected_later_order_survived": not (exact_terminal and no_corrected_fill),
        "corrected_all_working_orders_cancelled": exact_terminal and no_corrected_fill,
        "hard_flat_timestamp_ns": hard_flat_ns,
        "seconds_after_hard_flat": (
            (historical_entry_ns - hard_flat_ns) / 1e9
            if historical_entry_ns is not None else None
        ),
    }
    if evidence_override:
        unknown = set(evidence_override) - set(evidence)
        if unknown:
            raise CorrectedAllPeriodError(
                f"unknown hard-flat cancellation evidence fields: {sorted(unknown)}"
            )
        evidence.update(evidence_override)
    evidence["qualifies_as_h"] = bool(
        historical_trade is not None
        and corrected_trade is None
        and evidence["signal_unchanged"]
        and evidence["confirmation_unchanged"]
        and evidence["historical_entry_at_or_after_hard_flat"]
        and evidence["corrected_terminal_is_hard_flat_cancellation"]
        and not evidence["corrected_fill_occurred"]
        and not evidence["corrected_position_created"]
        and not evidence["corrected_later_order_survived"]
        and evidence["corrected_all_working_orders_cancelled"]
    )
    return evidence


def _trade_diff_row(
    *, period_id: str, setup_key: str, historical_setup: Mapping[str, Any],
    historical_confirmation: tuple[int | None, float | None],
    corrected_confirmation: tuple[int | None, float | None],
    historical_trade: Mapping[str, Any] | None,
    corrected_trade: Mapping[str, Any] | None,
    corrected_terminal: str | None,
    historical_trades: Mapping[str, Mapping[str, Any]],
    corrected_trades: Mapping[str, Mapping[str, Any]],
    signal_unchanged: bool,
    confirmation_unchanged: bool,
    hard_flat_evidence_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old, new = historical_trade, corrected_trade
    direction = str((old or new or historical_setup).get("direction", ""))
    if direction == "BUYER_ABSORPTION":
        direction = "LONG"
    elif direction == "SELLER_ABSORPTION":
        direction = "SHORT"
    day, interaction_id = setup_key.split("|", 1)
    classification: str
    predecessor: str | None = None
    causal_reason: str
    cancellation_evidence = _hard_flat_cancellation_evidence(
        day=day,
        historical_trade=old,
        corrected_trade=new,
        corrected_terminal=corrected_terminal,
        signal_unchanged=signal_unchanged,
        confirmation_unchanged=confirmation_unchanged,
        evidence_override=hard_flat_evidence_override,
    )

    if not signal_unchanged:
        classification = "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
        causal_reason = "SIGNAL_POPULATION_MUTATION"
        prior_state = "SIGNAL_POPULATION_CHANGED"
    elif not confirmation_unchanged:
        classification = "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
        causal_reason = "CONFIRMATION_CANDIDATE_MUTATION"
        prior_state = "CONFIRMATION_CHANGED"
    elif old is not None and new is not None:
        if not _confirmation_equal(historical_confirmation, corrected_confirmation):
            classification = "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"
            causal_reason = "CONFIRMATION_CANDIDATE_MUTATION"
        elif not _same_entry_geometry(old, new):
            classification = "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"
            causal_reason = "COMMON_TRADE_ENTRY_GEOMETRY_MUTATION"
        elif _same_exit_geometry(old, new):
            classification = "A_SAME_ENTRY_SAME_EXIT"
            causal_reason = "IDENTICAL_ENTRY_AND_EXIT"
        else:
            reason = str(new.get("exit_reason"))
            classification = {
                "HARD_FLAT_BERLIN": "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT",
                "DATA_GAP_3S_FORCE_FLAT": "C_SAME_ENTRY_DIFFERENT_EXIT_DUE_3S_DATA_GAP",
                "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO": "D_SAME_ENTRY_DIFFERENT_EXIT_DUE_SOURCE_END",
            }.get(reason, "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON")
            causal_reason = (
                f"ALLOWED_CORRECTED_EXIT:{reason}" if classification.startswith(("B_", "C_", "D_"))
                else f"UNEXPLAINED_EXIT_MUTATION:{reason}"
            )
        prior_state = "SELF_POSITION_IN_BOTH"
    else:
        membership = "HISTORICAL_ONLY" if old is not None else "CORRECTED_ONLY"
        trade = old or new
        assert trade is not None
        timestamp = int(trade["entry_timestamp_ns"])
        predecessor, predecessor_reason = _blocking_predecessor(
            membership=membership, setup_key=setup_key, setup_timestamp_ns=timestamp,
            historical_trades=historical_trades, corrected_trades=corrected_trades,
        )
        old_active = _active_setup_at(historical_trades, timestamp, exclude=setup_key)
        new_active = _active_setup_at(corrected_trades, timestamp, exclude=setup_key)
        prior_state = f"historical={old_active or 'FLAT'};corrected={new_active or 'FLAT'}"
        if cancellation_evidence["qualifies_as_h"]:
            classification = "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
            causal_reason = (
                "HISTORICAL_ENTRY_AT_OR_AFTER_HARD_FLAT_CANCELLED_WITHOUT_FILL:"
                f"entry={timestamp}:hard_flat={cancellation_evidence['hard_flat_timestamp_ns']}:"
                f"seconds_after={cancellation_evidence['seconds_after_hard_flat']}"
            )
        elif predecessor is not None:
            classification = "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT"
            causal_reason = predecessor_reason or "VERIFIED_PREDECESSOR_LIFECYCLE_CHANGE"
        else:
            classification = "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"
            hard_flat = int(cancellation_evidence["hard_flat_timestamp_ns"])
            if old is not None and timestamp >= hard_flat and corrected_terminal == "CANCELLED_AT_HARD_FLAT_BERLIN":
                causal_reason = (
                    "POST_HARD_FLAT_CANCELLATION_FAILED_STRICT_H_EVIDENCE:"
                    f"entry={timestamp}:hard_flat={hard_flat}"
                )
            else:
                causal_reason = "TRADE_MEMBERSHIP_CHANGE_WITHOUT_VERIFIED_CAUSAL_PREDECESSOR"

    return {
        "classification": classification,
        "period": period_id,
        "session": day,
        "interaction_id": interaction_id,
        "setup_id": str((old or new or historical_setup).get("setup_id", f"L2:{interaction_id}")),
        "direction": direction,
        "level": str((old or new or historical_setup).get("level", "")),
        "historical_confirmation_timestamp_ns": historical_confirmation[0],
        "historical_confirmation_utc": _optional_iso_ns(historical_confirmation[0]),
        "historical_confirmation_price": historical_confirmation[1],
        "corrected_confirmation_timestamp_ns": corrected_confirmation[0],
        "corrected_confirmation_utc": _optional_iso_ns(corrected_confirmation[0]),
        "corrected_confirmation_price": corrected_confirmation[1],
        "historical_entry_timestamp_ns": old.get("entry_timestamp_ns") if old else None,
        "historical_entry_utc": _optional_iso_ns(old.get("entry_timestamp_ns") if old else None),
        "historical_entry": old.get("entry") if old else None,
        "corrected_entry_timestamp_ns": new.get("entry_timestamp_ns") if new else None,
        "corrected_entry_utc": _optional_iso_ns(new.get("entry_timestamp_ns") if new else None),
        "corrected_entry": new.get("entry") if new else None,
        "historical_exit_reason": old.get("exit_reason") if old else None,
        "historical_exit_timestamp_ns": old.get("exit_timestamp_ns") if old else None,
        "historical_exit_utc": _optional_iso_ns(old.get("exit_timestamp_ns") if old else None),
        "historical_exit": old.get("exit") if old else None,
        "historical_r_multiple": old.get("r_multiple") if old else None,
        "historical_net_pnl_usd": old.get("net_pnl_usd") if old else None,
        "corrected_exit_reason": new.get("exit_reason") if new else None,
        "corrected_exit_timestamp_ns": new.get("exit_timestamp_ns") if new else None,
        "corrected_exit_utc": _optional_iso_ns(new.get("exit_timestamp_ns") if new else None),
        "corrected_exit": new.get("exit") if new else None,
        "corrected_r_multiple": new.get("r_multiple") if new else None,
        "corrected_net_pnl_usd": new.get("net_pnl_usd") if new else None,
        "price_source_timestamp_ns": new.get("price_source_timestamp_ns") if new else None,
        "price_source_utc": _optional_iso_ns(new.get("price_source_timestamp_ns") if new else None),
        "liquidation_reference_price": new.get("liquidation_reference_price") if new else None,
        "gap_start_timestamp_ns": new.get("gap_start_timestamp_ns") if new else None,
        "gap_start_utc": _optional_iso_ns(new.get("gap_start_timestamp_ns") if new else None),
        "gap_timeout_timestamp_ns": new.get("gap_timeout_timestamp_ns") if new else None,
        "gap_timeout_utc": _optional_iso_ns(new.get("gap_timeout_timestamp_ns") if new else None),
        "hard_flat_instruction_timestamp_ns": new.get("hard_flat_instruction_timestamp_ns") if new else None,
        "hard_flat_utc": berlin_hard_flat_utc(day).isoformat(),
        "hard_flat_local": execution_local_iso(day),
        **cancellation_evidence,
        "corrected_terminal_outcome": corrected_terminal,
        "prior_position_state": prior_state,
        "causal_predecessor_setup_id": predecessor,
        "exact_causal_reason": causal_reason,
    }


def analyze_semantic_period(
    *, period_id: str, historical_signal_ids: set[str], corrected_signal_ids: set[str],
    historical_setups: Mapping[str, Mapping[str, Any]],
    corrected_indexes: Mapping[str, Mapping[str, Any]],
    historical_trades: Mapping[str, Mapping[str, Any]],
    corrected_trades: Mapping[str, Mapping[str, Any]],
    corrected_terminal_outcomes: Mapping[str, str],
    hard_flat_evidence_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    signal_differences = sorted(historical_signal_ids ^ corrected_signal_ids)
    common_signals = historical_signal_ids & corrected_signal_ids
    confirmation_differences: list[str] = []
    historical_confirmations: dict[str, tuple[int | None, float | None]] = {}
    corrected_confirmations: dict[str, tuple[int | None, float | None]] = {}
    for key in sorted(common_signals):
        try:
            historical_confirmations[key] = _historical_confirmation(historical_setups[key])
            corrected_confirmations[key] = _indexed_confirmation(corrected_indexes[key])
        except KeyError as exc:
            raise CorrectedAllPeriodError(
                f"confirmation evidence missing for {period_id}/{key}"
            ) from exc
        if not _confirmation_equal(historical_confirmations[key], corrected_confirmations[key]):
            confirmation_differences.append(key)

    rows: list[dict[str, Any]] = []
    trade_union = sorted(set(historical_trades) | set(corrected_trades))
    for key in trade_union:
        signal_unchanged = key in common_signals
        historical_confirmation = historical_confirmations.get(key, (None, None))
        corrected_confirmation = corrected_confirmations.get(key, (None, None))
        confirmation_unchanged = (
            signal_unchanged
            and historical_confirmation[0] is not None
            and _confirmation_equal(historical_confirmation, corrected_confirmation)
        )
        rows.append(_trade_diff_row(
            period_id=period_id,
            setup_key=key,
            historical_setup=historical_setups.get(key, {}),
            historical_confirmation=historical_confirmation,
            corrected_confirmation=corrected_confirmation,
            historical_trade=historical_trades.get(key),
            corrected_trade=corrected_trades.get(key),
            corrected_terminal=corrected_terminal_outcomes.get(key),
            historical_trades=historical_trades,
            corrected_trades=corrected_trades,
            signal_unchanged=signal_unchanged,
            confirmation_unchanged=confirmation_unchanged,
            hard_flat_evidence_override=(hard_flat_evidence_overrides or {}).get(key),
        ))

    represented = set(trade_union)
    for key in sorted((set(signal_differences) | set(confirmation_differences)) - represented):
        setup = historical_setups.get(key, {})
        rows.append({
            "classification": "G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON",
            "period": period_id,
            "session": key.split("|", 1)[0],
            "interaction_id": key.split("|", 1)[1],
            "setup_id": setup.get("setup_id"),
            "direction": setup.get("direction"),
            "level": setup.get("level"),
            "historical_confirmation_timestamp_ns": historical_confirmations.get(key, (None, None))[0],
            "historical_confirmation_price": historical_confirmations.get(key, (None, None))[1],
            "corrected_confirmation_timestamp_ns": corrected_confirmations.get(key, (None, None))[0],
            "corrected_confirmation_price": corrected_confirmations.get(key, (None, None))[1],
            "exact_causal_reason": (
                "SIGNAL_POPULATION_MUTATION" if key in signal_differences
                else "CONFIRMATION_CANDIDATE_MUTATION"
            ),
        })

    counts = Counter(str(row["classification"]) for row in rows)
    unexpected_entry = counts["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"]
    common_entry_mutations = sum(
        row["classification"] == "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"
        and row.get("historical_entry_timestamp_ns") is not None
        and row.get("corrected_entry_timestamp_ns") is not None
        for row in rows
    )
    unexplained_membership_changes = sum(
        row["classification"] == "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"
        and (
            row.get("historical_entry_timestamp_ns") is None
            or row.get("corrected_entry_timestamp_ns") is None
        )
        for row in rows
    )
    hard_flat_cancellations = counts["H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"]
    return {
        "rows": rows,
        "classification_counts": {key: counts[key] for key in SEMANTIC_CLASSIFICATIONS},
        "signal_differences": signal_differences,
        "confirmation_differences": confirmation_differences,
        "unexpected_entry_changes": unexpected_entry,
        "common_entry_mutations": common_entry_mutations,
        "unexplained_membership_changes": unexplained_membership_changes,
        "blocking_membership_changes": counts[
            "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT"
        ],
        "hard_flat_cancellation_membership_changes": hard_flat_cancellations,
        "changed_exits": sum(counts[key] for key in SEMANTIC_CLASSIFICATIONS[1:4]),
        "unchanged_trades": counts["A_SAME_ENTRY_SAME_EXIT"],
    }


def corrected_hard_flat_invariants(
    *, corrected_trades: Mapping[str, Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit corrected positions/orders against Berlin hard-flat and maintenance."""
    entries_at_or_after: list[str] = []
    positions_open_after: list[str] = []
    positions_reaching_maintenance: list[str] = []
    for setup_key, trade in corrected_trades.items():
        day = str(trade["date"])
        hard_flat_ns = int(berlin_hard_flat_utc(day).timestamp() * 1e9)
        maintenance_start_ns = int(maintenance_window_utc(day)[0].timestamp() * 1e9)
        entry_ns = int(trade["entry_timestamp_ns"])
        if entry_ns >= hard_flat_ns:
            entries_at_or_after.append(setup_key)

        # A hard-flat instruction closes strategy exposure at the exact boundary;
        # the first native BBO after it supplies the fill price and may timestamp
        # milliseconds later without representing post-boundary strategy exposure.
        if (
            str(trade.get("exit_reason")) == "HARD_FLAT_BERLIN"
            and _optional_int(trade.get("hard_flat_instruction_timestamp_ns")) == hard_flat_ns
        ):
            effective_exit_ns = hard_flat_ns
        else:
            effective_exit_ns = int(trade["exit_timestamp_ns"])
        if entry_ns < hard_flat_ns < effective_exit_ns:
            positions_open_after.append(setup_key)
        if effective_exit_ns >= maintenance_start_ns:
            positions_reaching_maintenance.append(setup_key)

    hard_flat_rows = [
        row for row in semantic_rows
        if row.get("classification") == "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
    ]
    working_orders_surviving = [
        str(row.get("setup_id")) for row in hard_flat_rows
        if (
            bool(row.get("corrected_later_order_survived"))
            or not bool(row.get("corrected_all_working_orders_cancelled"))
        )
    ]
    positions_created_by_cancelled_setups = [
        str(row.get("setup_id")) for row in hard_flat_rows
        if bool(row.get("corrected_position_created"))
    ]
    return {
        "corrected_entries_at_or_after_hard_flat_count": len(entries_at_or_after),
        "corrected_entries_at_or_after_hard_flat": entries_at_or_after,
        "corrected_positions_open_after_hard_flat_count": len(positions_open_after),
        "corrected_positions_open_after_hard_flat": positions_open_after,
        "corrected_working_orders_after_hard_flat_count": len(working_orders_surviving),
        "corrected_working_orders_after_hard_flat": working_orders_surviving,
        "corrected_positions_created_by_cancelled_setups_count": len(
            positions_created_by_cancelled_setups
        ),
        "corrected_positions_created_by_cancelled_setups": (
            positions_created_by_cancelled_setups
        ),
        "corrected_positions_reaching_maintenance_count": len(
            positions_reaching_maintenance
        ),
        "corrected_positions_reaching_maintenance": positions_reaching_maintenance,
    }


def run_preflight(*, repository_root: Path, baseline_root: Path, output_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    baseline_root = _resolve(repository_root, baseline_root)
    output_root = _resolve(repository_root, output_root)
    if output_root.exists():
        previous = _read_json(output_root / "semantic-diff.json")
        if (
            previous.get("status") != "CORRECTED_CONTRACT_PREFLIGHT_FAIL"
            or previous.get("execution_contract_sha256") != CONTRACT_SHA256
        ):
            raise FileExistsError(f"sealed or unrelated semantic preflight exists: {output_root}")
    summary = _read_json(baseline_root / "summary.json")
    if summary.get("execution_contract_sha256") != CONTRACT_SHA256:
        raise CorrectedAllPeriodError("baseline execution contract hash mismatch")
    bundle_by_period = {
        bundle.period.period_id: bundle
        for bundle in allp._period_bundles(repository_root, _resolve(repository_root, TAPE_ROOT))
    }
    details: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    source_artifacts: dict[str, Any] = {}
    canonical_population_unchanged = True
    historical_unresolved = 0
    corrected_unresolved = 0
    all_corrected_trades: dict[str, Mapping[str, Any]] = {}
    for period in allp.PERIODS:
        document = _read_json(baseline_root / "periods" / f"{period.period_id}.json")
        audit = document["semantic_audit"]
        accepted = set(audit["accepted_setup_ids"])
        historical_signal_ids = set(audit["old_terminal_outcomes"])
        corrected_terminal = {
            str(key): str(value) for key, value in audit["corrected_terminal_outcomes"].items()
        }
        corrected_signal_ids = set(corrected_terminal)
        canonical_population_unchanged &= (
            accepted == historical_signal_ids == corrected_signal_ids
        )
        historical_setups, historical_trades, provenance = _historical_ledgers(
            repository_root, period.period_id, accepted,
        )
        _interactions, indexes, _by_day = _period_inputs(bundle_by_period[period.period_id])
        corrected_indexes = {key: indexes[key] for key in accepted}
        corrected_trades = _unique_by_setup(
            audit["corrected_trades"], artifact=f"corrected baseline/{period.period_id}",
        )
        overlap = set(all_corrected_trades) & set(corrected_trades)
        if overlap:
            raise CorrectedAllPeriodError(
                f"corrected setup keys overlap across periods: {sorted(overlap)}"
            )
        all_corrected_trades.update(corrected_trades)
        analysis = analyze_semantic_period(
            period_id=period.period_id,
            historical_signal_ids=historical_signal_ids,
            corrected_signal_ids=corrected_signal_ids,
            historical_setups=historical_setups,
            corrected_indexes=corrected_indexes,
            historical_trades=historical_trades,
            corrected_trades=corrected_trades,
            corrected_terminal_outcomes=corrected_terminal,
        )
        semantic_rows.extend(analysis.pop("rows"))
        period_historical_unresolved = sum(
            not str(row.get("terminal_reason", "")).strip()
            for row in historical_setups.values()
        )
        period_corrected_unresolved = int(document["metrics"]["unresolved"])
        historical_unresolved += period_historical_unresolved
        corrected_unresolved += period_corrected_unresolved
        source_artifacts[period.period_id] = provenance
        details.append({
            "period_id": period.period_id,
            "accepted_setups": len(accepted),
            "historical_trades": len(historical_trades),
            "corrected_trades": len(corrected_trades),
            "historical_unresolved": period_historical_unresolved,
            "corrected_unresolved": period_corrected_unresolved,
            **analysis,
        })
    aggregate_counts = Counter(
        str(row["classification"]) for row in semantic_rows
    )
    signal_differences = sum(len(row["signal_differences"]) for row in details)
    confirmation_differences = sum(len(row["confirmation_differences"]) for row in details)
    common_entry_mutations = sum(int(row["common_entry_mutations"]) for row in details)
    unexplained_membership_changes = sum(
        int(row["unexplained_membership_changes"]) for row in details
    )
    blocking_membership_changes = aggregate_counts[
        "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT"
    ]
    hard_flat_cancellation_changes = aggregate_counts[
        "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
    ]
    allowed_difference_classes = (
        "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT",
        "C_SAME_ENTRY_DIFFERENT_EXIT_DUE_3S_DATA_GAP",
        "D_SAME_ENTRY_DIFFERENT_EXIT_DUE_SOURCE_END",
        "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT",
        "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN",
    )
    allowed_difference_count = sum(
        aggregate_counts[key] for key in allowed_difference_classes
    )
    membership_rows = [
        row for row in semantic_rows
        if (
            row.get("historical_entry_timestamp_ns") is None
            or row.get("corrected_entry_timestamp_ns") is None
        )
    ]
    hard_flat_cancellation_rows = [
        row for row in semantic_rows
        if row["classification"] == "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN"
    ]
    hard_flat_invariants = corrected_hard_flat_invariants(
        corrected_trades=all_corrected_trades,
        semantic_rows=semantic_rows,
    )
    gates = {
        "historical_v3_hash_unchanged": HISTORICAL_V3_CONTRACT_SHA256 == allp.V3_CONTRACT_SHA256,
        "pre_entry_signal_population_identical": canonical_population_unchanged and signal_differences == 0,
        "confirmation_candidates_identical_absent_blocking": confirmation_differences == 0,
        "common_trade_entry_rule_identical": common_entry_mutations == 0,
        "trade_membership_changes_are_only_e_or_h": all(
            row["classification"] in {
                "E_LATER_SETUP_BLOCKING_CHANGED_DUE_PRIOR_CORRECTED_EXIT",
                "H_ENTRY_CANCELLED_AFTER_HARD_FLAT_BERLIN",
            }
            for row in membership_rows
        ),
        "hard_flat_cancellations_satisfy_all_h_conditions": all(
            bool(row.get("qualifies_as_h")) for row in hard_flat_cancellation_rows
        ),
        "unexpected_entry_changes_zero": (
            aggregate_counts["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"] == 0
        ),
        "unexpected_signal_or_confirmation_changes_zero": (
            aggregate_counts["G_SIGNAL_OR_CONFIRMATION_CHANGED_FOR_UNEXPECTED_REASON"] == 0
        ),
        "changed_exits_are_only_corrected_contract_reasons": not any(
            row["classification"] == "F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"
            and row.get("historical_entry_timestamp_ns") is not None
            and row.get("corrected_entry_timestamp_ns") is not None
            for row in semantic_rows
        ),
        "all_trade_differences_classified": all(
            row["classification"] in SEMANTIC_CLASSIFICATIONS for row in semantic_rows
        ),
        "signal_parameter_contract_unchanged": EXECUTION_CONTRACT["signal_contract"] == {
            "eligible_levels": ["PRIOR_RTH_POC"],
            "quality_components": list(master.SCORE_FIELDS),
            "confirmation_favorable_ticks": 3,
            "confirmation_horizon_seconds": 15,
            "entry_latency_ms": 2,
        },
        "historical_and_corrected_unresolved_zero": (
            historical_unresolved == 0
            and corrected_unresolved == 0
            and int(summary["aggregate"]["unresolved"]) == 0
        ),
        "corrected_integrity_failures_zero": int(summary["aggregate"]["integrity_failures"]) == 0,
        "zero_corrected_entries_at_or_after_hard_flat": (
            hard_flat_invariants["corrected_entries_at_or_after_hard_flat_count"] == 0
        ),
        "zero_corrected_open_positions_after_hard_flat": (
            hard_flat_invariants["corrected_positions_open_after_hard_flat_count"] == 0
            and hard_flat_invariants[
                "corrected_positions_created_by_cancelled_setups_count"
            ] == 0
        ),
        "zero_working_orders_after_hard_flat": (
            hard_flat_invariants["corrected_working_orders_after_hard_flat_count"] == 0
        ),
        "zero_positions_reaching_maintenance": (
            hard_flat_invariants["corrected_positions_reaching_maintenance_count"] == 0
        ),
        "network_calls_zero": int(summary["network_calls"]) == 0,
    }
    passed = all(gates.values())
    changed_rows = [
        row for row in semantic_rows
        if row["classification"] != "A_SAME_ENTRY_SAME_EXIT"
    ]
    may_gap_rows = [
        row for row in changed_rows
        if row["classification"] == "C_SAME_ENTRY_DIFFERENT_EXIT_DUE_3S_DATA_GAP"
    ]
    december_hard_flat_rows = [
        row for row in changed_rows
        if row["classification"] == "B_SAME_ENTRY_DIFFERENT_EXIT_DUE_BERLIN_HARDFLAT"
        and row["period"] == "DECEMBER_2025"
    ]
    direct_post_hard_flat = hard_flat_cancellation_rows
    result = {
        "status": "CORRECTED_CONTRACT_PREFLIGHT_PASS" if passed else "CORRECTED_CONTRACT_PREFLIGHT_FAIL",
        "optimizer_permitted": passed,
        "execution_contract_sha256": CONTRACT_SHA256,
        "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
        "gates": gates,
        "periods": details,
        "classification_counts": {
            key: aggregate_counts[key] for key in SEMANTIC_CLASSIFICATIONS
        },
        "unchanged_trades": aggregate_counts["A_SAME_ENTRY_SAME_EXIT"],
        "changed_exits": sum(aggregate_counts[key] for key in SEMANTIC_CLASSIFICATIONS[1:4]),
        "blocking_caused_membership_differences": blocking_membership_changes,
        "hard_flat_cancelled_membership_differences": hard_flat_cancellation_changes,
        "unexpected_signal_differences": signal_differences,
        "unexpected_confirmation_differences": confirmation_differences,
        "unexpected_entry_differences": aggregate_counts["F_ENTRY_CHANGED_FOR_UNEXPECTED_REASON"],
        "unexplained_membership_changes": unexplained_membership_changes,
        "historical_unresolved": historical_unresolved,
        "corrected_unresolved": corrected_unresolved,
        "integrity_failures": int(summary["aggregate"]["integrity_failures"]),
        "changed_trade_details": changed_rows,
        "may_data_gap_trade": may_gap_rows[0] if len(may_gap_rows) == 1 else None,
        "december_berlin_hard_flat_trade": (
            december_hard_flat_rows[0] if len(december_hard_flat_rows) == 1 else None
        ),
        "direct_post_hard_flat_entry_changes": direct_post_hard_flat,
        "hard_flat_invariants": hard_flat_invariants,
        "historical_artifacts": source_artifacts,
        "allowed_execution_semantic_differences": allowed_difference_count,
        "network_calls": 0, "downloads": 0, "dbn_files_opened": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "semantic-diff.csv", semantic_rows)
    _write_json(output_root / "preflight-summary.json", result)
    _write_json(output_root / "semantic-diff.json", result)
    report = _preflight_html(result)
    (output_root / "preflight-report.html").write_text(report, encoding="utf-8")
    (output_root / "semantic-diff.html").write_text(report, encoding="utf-8")
    if not passed:
        raise CorrectedAllPeriodError("corrected contract semantic preflight failed")
    return {**result, "output_root": str(output_root)}


def _preflight_html(result: Mapping[str, Any]) -> str:
    gates = "".join(
        f"<li><b>{html.escape(name)}</b>: {'PASS' if value else 'FAIL'}</li>"
        for name, value in result["gates"].items()
    )
    changed = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in (
                "classification", "period", "session", "setup_id", "direction",
                "historical_entry_timestamp_ns", "corrected_entry_timestamp_ns",
                "historical_exit_reason", "corrected_exit_reason", "exact_causal_reason",
            )
        ) + "</tr>"
        for row in result["changed_trade_details"]
    )
    may = result.get("may_data_gap_trade")
    may_section = ""
    if isinstance(may, Mapping):
        may_section = f"""<h2>May 3-second data-gap exit</h2>
<p><code>{html.escape(str(may['setup_id']))}</code> entered at {may['historical_entry']} ({may['historical_entry_utc']}). Historical V3 remained open until a STOP at {may['historical_exit']} ({may['historical_exit_utc']}), producing {may['historical_r_multiple']}R / ${may['historical_net_pnl_usd']}.</p>
<p>The corrected contract used the last valid native BBO reference {may['liquidation_reference_price']} from {may['price_source_utc']}; timeout {may['gap_timeout_utc']}; non-executable boundary/decision {may['gap_start_utc']}; adverse-slippage fill {may['corrected_exit']} for {may['corrected_r_multiple']}R / ${may['corrected_net_pnl_usd']}. No reopen or future price supplied the liquidation reference.</p>"""
    december = result.get("december_berlin_hard_flat_trade")
    december_section = ""
    if isinstance(december, Mapping):
        december_section = f"""<h2>December Berlin hard flat</h2>
<p><code>{html.escape(str(december['setup_id']))}</code> changed from {december['historical_exit_reason']} at {december['historical_exit_utc']} to HARD_FLAT_BERLIN at {december['corrected_exit_utc']}. The instruction is {december['hard_flat_local']} / {december['hard_flat_utc']}. Both fills were {december['corrected_exit']}, so R ({december['corrected_r_multiple']}) and net PnL (${december['corrected_net_pnl_usd']}) remained identical.</p>"""
    direct_rows = "".join(
        f"<li><code>{html.escape(str(row['setup_id']))}</code>: historical entry "
        f"{row['historical_entry_utc']}; Berlin flat {row['hard_flat_local']} / "
        f"{row['hard_flat_utc']}; {row['seconds_after_hard_flat']} seconds after; "
        f"corrected terminal {row['corrected_terminal_outcome']}; no fill="
        f"{not bool(row['corrected_fill_occurred'])}; all working orders cancelled="
        f"{row['corrected_all_working_orders_cancelled']}.</li>"
        for row in result.get("direct_post_hard_flat_entry_changes", ())
    )
    direct_section = (
        "<h2>Strict post-hard-flat entry cancellations (H)</h2><ul>" + direct_rows + "</ul>"
        if direct_rows else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Semantic preflight</title>
<style>body{{font:14px system-ui;margin:2rem;color:#18202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:.35rem;vertical-align:top}}code{{background:#eef;padding:.15rem}}.fail{{color:#a20}}</style></head>
<body><h1>Corrected-contract semantic preflight</h1><p class="{'fail' if result['status'].endswith('FAIL') else ''}"><b>{result['status']}</b></p>
<h2>Gates</h2><ul>{gates}</ul>
<h2>Trade-level semantic diff</h2><p>Unchanged trades: {result['unchanged_trades']}; changed exits: {result['changed_exits']}; causal blocking membership changes: {result['blocking_caused_membership_differences']}; strict post-hard-flat cancellations: {result['hard_flat_cancelled_membership_differences']}; unexplained entry changes: {result['unexpected_entry_differences']}.</p>
<table><thead><tr><th>class</th><th>period</th><th>session</th><th>setup</th><th>direction</th><th>old entry ns</th><th>new entry ns</th><th>old exit</th><th>new exit</th><th>causal reason</th></tr></thead><tbody>{changed}</tbody></table>
{may_section}{december_section}{direct_section}
<p>Only Berlin hard-flat, 3-second no-book, and source-end exit semantics are allowed. Trade-membership differences must be either E (a verified prior-position causal predecessor) or H (unchanged signal and confirmation, historical intended entry at/after the Berlin boundary, exact corrected cancellation, no fill or position, and no surviving working order). F and G remain hard failures.</p>
<p>No DBN, network, optimizer, or market-data acquisition was used.</p></body></html>"""


def _checkpoint_metadata(bundle: allp.PeriodBundle) -> dict[str, Any]:
    return {
        "period_id": bundle.period.period_id,
        "execution_contract_sha256": CONTRACT_SHA256,
        "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
        "semantic_version": SEMANTIC_VERSION,
        "tape_binding": _tape_binding(bundle),
        "weight_grid_sha256": _grid_hash(),
        "quality_thresholds": [str(value) for value in matrix.QUALITY_THRESHOLDS],
        "configuration_count": EXPECTED_GRID_COUNT,
        "code_sha256": _code_hash(),
    }


def _load_checkpoint(checkpoint_root: Path, expected: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    stem = str(expected["period_id"]).lower()
    final = checkpoint_root / stem
    if not final.exists():
        return None
    if not final.is_dir():
        raise CorrectedAllPeriodError(f"period checkpoint is not an atomic directory: {stem}")
    metadata_path = final / "checkpoint.json"
    rows_path = final / "period-results.parquet"
    if not metadata_path.is_file() or not rows_path.is_file():
        raise CorrectedAllPeriodError(f"partial period checkpoint: {stem}")
    actual = _read_json(metadata_path)
    expected_hash = _canonical_hash(expected)
    if actual.get("binding_sha256") != expected_hash or actual.get("binding") != expected:
        raise CorrectedAllPeriodError(f"period checkpoint binding mismatch: {stem}")
    if actual.get("rows_sha256") != _sha256(rows_path):
        raise CorrectedAllPeriodError(f"period checkpoint row hash mismatch: {stem}")
    rows = _read_parquet(rows_path)
    if len(rows) != EXPECTED_GRID_COUNT:
        raise CorrectedAllPeriodError(f"period checkpoint cardinality mismatch: {stem}")
    return rows


def _publish_checkpoint(
    checkpoint_root: Path, binding: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> None:
    stem = str(binding["period_id"]).lower()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_root / stem
    if final.exists():
        raise CorrectedAllPeriodError(f"verified checkpoint already exists: {stem}")
    temporary = checkpoint_root / f"{stem}.part"
    if temporary.exists():
        # This is an unpublished directory owned by this runner.  It cannot
        # be reused and is deterministically rebuilt from the bound tape.
        if not temporary.is_dir() or temporary.parent != checkpoint_root:
            raise CorrectedAllPeriodError(f"unsafe stale checkpoint path: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir()
    rows_path = temporary / "period-results.parquet"
    _write_parquet(rows_path, rows)
    _write_json(temporary / "checkpoint.json", {
        "status": "CORRECTED_PERIOD_CHECKPOINT_COMPLETE",
        "binding": binding,
        "binding_sha256": _canonical_hash(binding),
        "row_count": len(rows),
        "rows_sha256": _sha256(rows_path),
    })
    os.rename(temporary, final)


def _evaluate_corrected_bundle(
    bundle: allp.PeriodBundle,
    *, progress_interval: int = matrix.PROGRESS_INTERVAL,
) -> list[dict[str, Any]]:
    interactions, indexes, by_day = _period_inputs(bundle)
    registry = matrix.configuration_registry()
    weight_grid = matrix.generate_weight_grid()
    weights_matrix = np.asarray(weight_grid, dtype=np.float64) * float(matrix.WEIGHT_UNIT)
    accumulators = [CorrectedAccumulator(weights, threshold) for weights, threshold in registry]
    evaluations = 0
    next_progress = progress_interval
    started = time.monotonic()
    for session_number, day in enumerate(bundle.days, start=1):
        print(
            f"BERLIN_MATRIX {bundle.period.period_id} session={session_number:02d}/{len(bundle.days):02d} {day}",
            flush=True,
        )
        rows = by_day.get(day, [])
        tape = BerlinSessionCausalTape.from_parquet(day, _tape_path(bundle, day))
        if rows:
            components = np.asarray(
                [[float(row[name]) for name in master.SCORE_FIELDS] for row in rows], dtype=np.float64,
            )
            penalties = np.asarray([float(row["false_refill_penalty"]) for row in rows])
            primitive_ok = np.asarray([
                not str(row.get("non_quality_rejection_reasons") or "") for row in rows
            ])
            scores = np.clip(
                components @ weights_matrix.T
                - penalties[:, None] * float(allp.V2_CONFIG.false_refill_penalty_weight),
                0.0, 1.0,
            )
        else:
            scores = np.empty((0, matrix.EXPECTED_WEIGHT_COUNT), dtype=np.float64)
            primitive_ok = np.empty((0,), dtype=np.bool_)
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in rows}
        for weight_index, units in enumerate(weight_grid):
            base = weight_index * len(matrix.QUALITY_THRESHOLDS)
            for q_index, threshold in enumerate(matrix.QUALITY_THRESHOLDS):
                accepted_mask = matrix._accepted_mask(
                    rows, primitive_ok, scores[:, weight_index], units, threshold,
                )
                accepted = [row for row, keep in zip(rows, accepted_mask) if bool(keep)]
                session = simulate_berlin_session(tape, accepted, day_indexes)
                accumulators[base + q_index].add(session, set())
                evaluations += 1
        equivalent = evaluations // max(1, len(bundle.days))
        if equivalent >= next_progress:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"BERLIN_MATRIX_PROGRESS period={bundle.period.period_id} "
                f"equivalent_configs={equivalent:,}/{EXPECTED_GRID_COUNT:,} "
                f"rate={evaluations / elapsed:,.1f}_config_sessions_per_second",
                flush=True,
            )
            while next_progress <= equivalent:
                next_progress += progress_interval
    output: list[dict[str, Any]] = []
    for accumulator in accumulators:
        row = accumulator.row(0)
        output.append({
            "period_id": bundle.period.period_id,
            "source_model": bundle.period.source_model,
            "source_group": bundle.period.source_group,
            "sessions": len(bundle.days),
            "completed_interactions": len(interactions),
            **row,
            "gross_profit_usd": accumulator.gross_profit_usd,
            "gross_loss_usd": accumulator.gross_loss_usd,
        })
    return output


def load_corrected_berlin_baseline(
    repository_root: Path, baseline_root: Path = BASELINE_ROOT,
) -> dict[str, Any]:
    baseline_path = _resolve(repository_root.resolve(), baseline_root) / "summary.json"
    summary = _read_json(baseline_path)
    if summary.get("execution_contract_sha256") != CONTRACT_SHA256:
        raise CorrectedAllPeriodError("corrected Berlin baseline execution-contract hash mismatch")
    if summary.get("status") != "CORRECTED_BERLIN_HARDFLAT_BASELINE_COMPLETE":
        raise CorrectedAllPeriodError("corrected Berlin baseline is absent or incomplete")
    aggregate = summary.get("aggregate")
    periods = summary.get("periods")
    if not isinstance(aggregate, dict) or not isinstance(periods, list):
        raise CorrectedAllPeriodError("corrected Berlin baseline summary schema is invalid")
    required_periods = [period.period_id for period in allp.PERIODS]
    observed_periods = [str(row.get("period_id")) for row in periods]
    if observed_periods != required_periods:
        raise CorrectedAllPeriodError(
            f"corrected Berlin baseline periods mismatch: {observed_periods}"
        )
    required = {
        "sessions", "trades", "wins", "losses", "total_r", "net_pnl_usd",
        "profit_factor", "max_cumulative_drawdown_r", "unresolved",
        "integrity_failures",
    }
    if required - set(aggregate):
        raise CorrectedAllPeriodError("corrected Berlin baseline aggregate fields are incomplete")
    return summary


def _strategy_value_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_hash([
        {key: row[key] for key in row if key not in REPORTING_FIELD_NAMES}
        for row in rows
    ])


def _explicit_reference_fields(
    row: Mapping[str, Any], baseline_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    historical_trades = row.get(
        "historical_v3_reference_trades", row.get("v3_reference_trades"),
    )
    historical_r = row.get(
        "historical_v3_reference_total_r", row.get("v3_reference_total_r"),
    )
    historical_pnl = row.get(
        "historical_v3_reference_net_pnl_usd", row.get("v3_reference_net_pnl_usd"),
    )
    if historical_trades in (None, "") or historical_r in (None, "") or historical_pnl in (None, ""):
        raise CorrectedAllPeriodError(
            f"historical V3 reporting reference missing for {row.get('config_id')}"
        )
    berlin_trades = int(baseline_aggregate["trades"])
    berlin_r = Decimal(str(baseline_aggregate["total_r"]))
    berlin_pnl = Decimal(str(baseline_aggregate["net_pnl_usd"]))
    total_trades = int(row["total_trades"])
    total_r = Decimal(str(row["total_r"]))
    total_pnl = Decimal(str(row["total_net_pnl_usd"]))
    clean = {key: value for key, value in row.items() if key not in REPORTING_FIELD_NAMES}
    clean.update({
        "historical_v3_reference_trades": int(historical_trades),
        "historical_v3_reference_total_r": historical_r,
        "historical_v3_reference_net_pnl_usd": historical_pnl,
        "trade_delta_vs_historical_v3": total_trades - int(historical_trades),
        "total_r_delta_vs_historical_v3": str(total_r - Decimal(str(historical_r))),
        "net_pnl_delta_vs_historical_v3": str(total_pnl - Decimal(str(historical_pnl))),
        "berlin_v3_reference_trades": berlin_trades,
        "berlin_v3_reference_total_r": str(berlin_r),
        "berlin_v3_reference_net_pnl_usd": str(berlin_pnl),
        "trade_delta_vs_berlin_v3": total_trades - berlin_trades,
        "total_r_delta_vs_berlin_v3": str(total_r - berlin_r),
        "net_pnl_delta_vs_berlin_v3": str(total_pnl - berlin_pnl),
    })
    return clean


def _aggregate_corrected(
    rows: Sequence[Mapping[str, Any]],
    baseline_aggregate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = allp.aggregate_configuration_periods(rows)
    result.update({
        "hard_flat_berlin_exits": sum(int(row["hard_flat_berlin_exits"]) for row in rows),
        "data_gap_3s_force_flat_exits": sum(int(row["data_gap_3s_force_flat_exits"]) for row in rows),
        "source_end_force_flat_exits": sum(int(row["source_end_force_flat_exits"]) for row in rows),
        "integrity_failures": sum(int(row["integrity_failures"]) for row in rows),
    })
    if int(result["unresolved"]) != 0:
        raise CorrectedAllPeriodError(f"configuration unresolved invariant failed: {result['config_id']}")
    if baseline_aggregate is None:
        return result
    return _explicit_reference_fields(result, baseline_aggregate)


def require_preflight(repository_root: Path, preflight_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve(repository_root, preflight_root) / "semantic-diff.json")
    if (
        payload.get("status") != "CORRECTED_CONTRACT_PREFLIGHT_PASS"
        or payload.get("optimizer_permitted") is not True
        or payload.get("execution_contract_sha256") != CONTRACT_SHA256
    ):
        raise CorrectedAllPeriodError("corrected-contract preflight is absent or invalid")
    return payload


def run_optimizer(
    *, repository_root: Path, tape_root: Path, output_root: Path, preflight_root: Path,
    baseline_root: Path = BASELINE_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = _resolve(repository_root, tape_root)
    output_root = _resolve(repository_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"immutable corrected optimizer output exists: {output_root}")
    preflight = require_preflight(repository_root, preflight_root)
    berlin_baseline = load_corrected_berlin_baseline(repository_root, baseline_root)
    berlin_aggregate = berlin_baseline["aggregate"]
    bundles = allp._period_bundles(repository_root, tape_root)
    staging = output_root.with_name(output_root.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint_root = staging / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    period_rows: list[dict[str, Any]] = []
    resumed: list[str] = []
    evaluated: list[str] = []
    for index, bundle in enumerate(bundles, start=1):
        binding = _checkpoint_metadata(bundle)
        rows = _load_checkpoint(checkpoint_root, binding)
        if rows is None:
            print(f"BERLIN_OPTIMIZER {index}/7 {bundle.period.period_id}", flush=True)
            rows = _evaluate_corrected_bundle(bundle)
            _publish_checkpoint(checkpoint_root, binding, rows)
            evaluated.append(bundle.period.period_id)
        else:
            print(f"BERLIN_OPTIMIZER_RESUME {index}/7 {bundle.period.period_id}", flush=True)
            resumed.append(bundle.period.period_id)
        period_rows.extend(rows)
    if len(period_rows) != EXPECTED_GRID_COUNT * len(allp.PERIODS):
        raise CorrectedAllPeriodError("configuration x period cardinality mismatch")
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in period_rows:
        by_config[str(row["config_id"])].append(row)
    aggregate_rows = [
        _aggregate_corrected(by_config[key], berlin_aggregate) for key in sorted(by_config)
    ]
    neighbor_rows = allp.build_neighbor_robustness(aggregate_rows)
    source_rows = [{
        key: row[key] for key in (
            "config_id", "G1", "G2", "G3", "G4", "G5", "quality_threshold",
            "native_mbp10_total_r", "native_mbp10_trade_count", "mbo_derived_total_r",
            "mbo_derived_trade_count", "source_model_classification",
            "source_balance_min_total_r", "source_balance_absolute_r_gap",
        )
    } for row in aggregate_rows]
    plateau = allp.plateau_analysis(aggregate_rows, neighbor_rows)
    rankings = allp._rankings(aggregate_rows, neighbor_rows)
    _write_json(staging / "execution-contract.json", EXECUTION_CONTRACT)
    _write_json(staging / "semantic-preflight.json", preflight)
    _write_csv(staging / "weight-grid.csv", allp._weight_grid_rows())
    _write_parquet(staging / "weight-q-period-results.parquet", period_rows)
    _write_csv(staging / "weight-q-aggregate-results.csv", aggregate_rows)
    _write_csv(staging / "neighbor-robustness.csv", neighbor_rows)
    _write_csv(staging / "source-model-robustness.csv", source_rows)
    _write_json(staging / "plateau-analysis.json", plateau)
    for filename, rows in rankings.items():
        _write_csv(staging / filename, rows)
    summary = {
        "status": "CORRECTED_ALL_PERIOD_WEIGHT_Q_RESEARCH_COMPLETE_NO_SELECTION",
        "strategy_id": STRATEGY_ID,
        "execution_contract_sha256": CONTRACT_SHA256,
        "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
        "berlin_v3_reference": {
            "trades": berlin_aggregate["trades"],
            "total_r": berlin_aggregate["total_r"],
            "net_pnl_usd": berlin_aggregate["net_pnl_usd"],
        },
        "evidence_label": EVIDENCE_LABEL,
        "period_count": len(allp.PERIODS), "session_count": allp.EXPECTED_SESSION_COUNT,
        "weight_count": matrix.EXPECTED_WEIGHT_COUNT,
        "quality_thresholds": [float(value) for value in matrix.QUALITY_THRESHOLDS],
        "configuration_count": EXPECTED_GRID_COUNT,
        "minimum_sample_views": [60, 80, 100],
        "robustness_outputs": [
            "period_results", "aggregate_results", "neighbor_robustness",
            "source_model_balance", "plateau_analysis", "ranked_views",
        ],
        "checkpoint_periods_resumed": resumed,
        "checkpoint_periods_evaluated": evaluated,
        "automatic_strategy_selection": False, "selected_configuration": None,
        "network_calls": 0, "downloads": 0, "dbn_files_opened": 0,
    }
    _write_json(staging / "summary.json", summary)
    # Keep checkpoints as durable provenance in the immutable final tree.
    os.rename(staging, output_root)
    return {**summary, "output_root": str(output_root)}


def _metric(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _comparison_period_row(
    *, config_id: str, weights: Mapping[str, Any], period: Mapping[str, Any],
    aggregate: bool = False,
) -> dict[str, Any]:
    trades = int(_metric(period, "trades", "completed_trades", "total_trades", default=0))
    wins = int(_metric(period, "wins", default=0))
    losses = int(_metric(period, "losses", default=0))
    total_r = float(_metric(period, "total_r", default=0.0))
    return {
        "config_id": config_id,
        **weights,
        "period_id": "AGGREGATE" if aggregate else str(period["period_id"]),
        "sessions": int(_metric(period, "sessions", default=0)),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": float(_metric(period, "win_rate", default=(wins / trades if trades else 0.0))),
        "total_r": total_r,
        "average_r": float(_metric(
            period, "average_r", "weighted_average_r",
            default=(total_r / trades if trades else 0.0),
        )),
        "net_pnl_usd": float(_metric(period, "net_pnl_usd", "total_net_pnl_usd", default=0.0)),
        "profit_factor": _optional_float(_metric(
            period, "profit_factor", "aggregate_profit_factor",
        )),
        "max_dd_r": float(_metric(
            period, "max_cumulative_drawdown_r", "worst_period_max_drawdown_r", default=0.0,
        )),
        "es_trades": int(_metric(period, "es_trades", default=0)),
        "mes_trades": int(_metric(period, "mes_trades", default=0)),
        "target_exits": int(_metric(period, "target_exits", default=0)),
        "stop_exits": int(_metric(period, "stop_exits", default=0)),
        "hard_flat_exits": int(_metric(period, "hard_flat_berlin_exits", default=0)),
        "data_gap_3s_force_flat_exits": int(_metric(
            period, "data_gap_3s_force_flat_exits", default=0,
        )),
        "source_end_exits": int(_metric(period, "source_end_force_flat_exits", default=0)),
        "unresolved": int(_metric(period, "unresolved", default=0)),
        "source_group": "ALL" if aggregate else str(period["source_group"]),
    }


def _baseline_aggregate_for_comparison(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = dict(summary["aggregate"])
    aggregate["period_id"] = "AGGREGATE"
    return aggregate


def _source_totals(period_rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    native = sum(
        float(row["total_r"]) for row in period_rows
        if row["source_group"] == "NATIVE_MBP10"
    )
    mbo = sum(
        float(row["total_r"]) for row in period_rows
        if row["source_group"] == "MBO_DERIVED"
    )
    return native, mbo


def _candidate_summary_row(
    *, config_id: str, weights: Mapping[str, Any], aggregate: Mapping[str, Any],
    period_rows: Sequence[Mapping[str, Any]], neighbor: Mapping[str, Any] | None,
    baseline_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    total_r = float(_metric(aggregate, "total_r", default=0.0))
    total_pnl = float(_metric(aggregate, "total_net_pnl_usd", "net_pnl_usd", default=0.0))
    period_rs = [float(row["total_r"]) for row in period_rows]
    native_r, mbo_r = _source_totals(period_rows)
    return {
        "config_id": config_id,
        **weights,
        "trades": int(_metric(aggregate, "total_trades", "trades", default=0)),
        "wins": int(_metric(aggregate, "wins", default=0)),
        "losses": int(_metric(aggregate, "losses", default=0)),
        "total_r": total_r,
        "net_pnl_usd": total_pnl,
        "profit_factor": _optional_float(_metric(
            aggregate, "aggregate_profit_factor", "profit_factor",
        )),
        "positive_periods": sum(value > 0 for value in period_rs),
        "negative_periods": sum(value < 0 for value in period_rs),
        "median_period_r": statistics.median(period_rs),
        "worst_period_r": min(period_rs),
        "best_period_r": max(period_rs),
        "worst_dd_r": min(float(row["max_dd_r"]) for row in period_rows),
        "native_r": native_r,
        "mbo_derived_r": mbo_r,
        "neighbor_median_r": _optional_float(
            neighbor.get("median_neighbor_total_r") if neighbor else None
        ),
        "neighbor_worst_r": _optional_float(
            neighbor.get("worst_neighbor_total_r") if neighbor else None
        ),
        "proportion_neighbors_positive": _optional_float(
            neighbor.get("proportion_neighbors_aggregate_positive") if neighbor else None
        ),
        "proportion_neighbors_at_least_5_positive_periods": _optional_float(
            neighbor.get("proportion_neighbors_at_least_five_positive_periods")
            if neighbor else None
        ),
        "total_r_delta_vs_berlin_v3": total_r - float(baseline_aggregate["total_r"]),
        "net_pnl_delta_vs_berlin_v3": total_pnl - float(baseline_aggregate["net_pnl_usd"]),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _candidate_comparison_markdown(
    summaries: Sequence[Mapping[str, Any]],
    periods_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    candidates = [row for row in summaries if row["config_id"] != BERLIN_BASELINE_CONFIG_ID]
    highest_r = max(summaries, key=lambda row: float(row["total_r"]))
    best_worst = max(summaries, key=lambda row: float(row["worst_period_r"]))
    best_pf = max(summaries, key=lambda row: float(row["profit_factor"] or float("-inf")))
    source_balanced = min(
        summaries, key=lambda row: abs(float(row["native_r"]) - float(row["mbo_derived_r"])),
    )
    strongest_neighbor = max(
        candidates,
        key=lambda row: (
            float(row["neighbor_median_r"]), float(row["neighbor_worst_r"]),
        ),
    )
    six_positive = [str(row["config_id"]) for row in summaries if int(row["positive_periods"]) == 6]
    descriptive = list(dict.fromkeys((str(best_worst["config_id"]), str(strongest_neighbor["config_id"]))))[:2]
    table = "\n".join(
        f"| {row['config_id']} | {row['trades']} | {row['wins']}/{row['losses']} | "
        f"{_fmt(row['total_r'])} | ${_fmt(row['net_pnl_usd'], 2)} | "
        f"{_fmt(row['profit_factor'])} | {row['positive_periods']}/7 | "
        f"{_fmt(row['worst_period_r'])} | {_fmt(row['neighbor_median_r'])} |"
        for row in summaries
    )
    period_table = "\n".join(
        f"| {config_id} | {row['period_id']} | {row['sessions']} | {row['trades']} | "
        f"{row['wins']}/{row['losses']} | {_fmt(row['total_r'])} | "
        f"${_fmt(row['net_pnl_usd'], 2)} | {_fmt(row['profit_factor'])} | "
        f"{_fmt(row['max_dd_r'])} | {row['source_group']} |"
        for config_id, rows in periods_by_config.items()
        for row in rows
    )
    negative_lines: list[str] = []
    january_lines: list[str] = []
    source_lines: list[str] = []
    for row in summaries:
        config_id = str(row["config_id"])
        periods = periods_by_config[config_id]
        negative = [str(item["period_id"]) for item in periods if float(item["total_r"]) < 0]
        january = next(item for item in periods if item["period_id"] == "JANUARY_2026")
        excluding_january = float(row["total_r"]) - float(january["total_r"])
        share = (
            float(january["total_r"]) / float(row["total_r"])
            if float(row["total_r"]) else 0.0
        )
        negative_lines.append(f"- `{config_id}`: {', '.join(negative) if negative else 'none'}")
        january_lines.append(
            f"- `{config_id}`: January {_fmt(january['total_r'])}R; "
            f"total excluding January {_fmt(excluding_january)}R; share {share:.1%}."
        )
        source_lines.append(
            f"- `{config_id}`: native {_fmt(row['native_r'])}R; "
            f"MBO-derived {_fmt(row['mbo_derived_r'])}R; absolute gap "
            f"{abs(float(row['native_r']) - float(row['mbo_derived_r'])):.3f}R."
        )
    return f"""# Corrected Berlin V3 candidate period comparison

Evidence label: `{EVIDENCE_LABEL}`

This is an offline descriptive report over existing optimizer artifacts. It does not select V5, recompute a configuration, open DBNs, or use network data.

| configuration | trades | W/L | total R | net PnL | PF | positive periods | worst period R | neighbor median R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Period detail

| configuration | period | sessions | trades | W/L | total R | net PnL | PF | max DD R | source group |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{period_table}

## Direct answers

1. Highest total R: `{highest_r['config_id']}` at {_fmt(highest_r['total_r'])}R.
2. Best worst-period R: `{best_worst['config_id']}` at {_fmt(best_worst['worst_period_r'])}R.
3. Six of seven positive periods: {', '.join(f'`{item}`' for item in six_positive) if six_positive else 'none'}.
4. Best PF: `{best_pf['config_id']}` at {_fmt(best_pf['profit_factor'])}.
5. Most source-balanced: `{source_balanced['config_id']}` by the smallest native/MBO R gap.
6. Strongest immediate-neighbor robustness: `{strongest_neighbor['config_id']}` with median {_fmt(strongest_neighbor['neighbor_median_r'])}R and worst {_fmt(strongest_neighbor['neighbor_worst_r'])}R.
7. Negative periods:
{chr(10).join(negative_lines)}
8. January dependence:
{chr(10).join(january_lines)}
9. Native versus MBO-derived dependence:
{chr(10).join(source_lines)}
10. Descriptively strongest robustness profiles: {', '.join(f'`{item}`' for item in descriptive)}. This is not a strategy selection or validation claim.
"""


def run_offline_reporting(
    *, repository_root: Path, optimizer_root: Path = OPTIMIZER_ROOT,
    baseline_root: Path = BASELINE_ROOT, verify_known_metrics: bool = True,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    optimizer_root = _resolve(repository_root, optimizer_root)
    baseline = load_corrected_berlin_baseline(repository_root, baseline_root)
    baseline_aggregate = baseline["aggregate"]
    optimizer_summary_path = optimizer_root / "summary.json"
    optimizer_summary = _read_json(optimizer_summary_path)
    if optimizer_summary.get("execution_contract_sha256") != CONTRACT_SHA256:
        raise CorrectedAllPeriodError("optimizer execution-contract hash mismatch")
    for key in ("network_calls", "downloads", "dbn_files_opened"):
        if int(optimizer_summary.get(key, -1)) != 0:
            raise CorrectedAllPeriodError(f"optimizer provenance is not offline: {key}")

    aggregate_path = optimizer_root / "weight-q-aggregate-results.csv"
    period_path = optimizer_root / "weight-q-period-results.parquet"
    neighbor_path = optimizer_root / "neighbor-robustness.csv"
    source_path = optimizer_root / "source-model-robustness.csv"
    plateau_path = optimizer_root / "plateau-analysis.json"
    aggregate_rows = _read_csv(aggregate_path)
    neighbor_rows = _read_csv(neighbor_path)
    source_rows = _read_csv(source_path)
    _read_json(plateau_path)
    configuration_count = int(optimizer_summary["configuration_count"])
    if len(aggregate_rows) != configuration_count:
        raise CorrectedAllPeriodError("aggregate result cardinality changed")
    raw_digest_before = _strategy_value_digest(aggregate_rows)
    repaired_aggregate_rows = [
        _explicit_reference_fields(row, baseline_aggregate) for row in aggregate_rows
    ]
    if _strategy_value_digest(repaired_aggregate_rows) != raw_digest_before:
        raise CorrectedAllPeriodError("raw aggregate strategy values changed during report repair")

    aggregate_by_id = {str(row["config_id"]): row for row in repaired_aggregate_rows}
    if len(aggregate_by_id) != len(repaired_aggregate_rows):
        raise CorrectedAllPeriodError("duplicate aggregate configuration")
    missing_candidates = set(COMPARISON_CONFIG_IDS) - set(aggregate_by_id)
    if missing_candidates:
        raise CorrectedAllPeriodError(f"comparison candidate missing: {sorted(missing_candidates)}")
    if verify_known_metrics:
        for config_id, (trades, total_r, pnl) in EXPECTED_COMPARISON_AGGREGATES.items():
            row = aggregate_by_id[config_id]
            if (
                int(row["total_trades"]) != trades
                or not math.isclose(float(row["total_r"]), total_r, abs_tol=1e-12)
                or not math.isclose(float(row["total_net_pnl_usd"]), pnl, abs_tol=1e-9)
            ):
                raise CorrectedAllPeriodError(f"known aggregate metric mismatch: {config_id}")

    period_rows = _read_parquet(period_path)
    expected_period_ids = [period.period_id for period in allp.PERIODS]
    selected_periods: dict[str, list[dict[str, Any]]] = {}
    for config_id in COMPARISON_CONFIG_IDS:
        rows = [dict(row) for row in period_rows if row.get("config_id") == config_id]
        rows.sort(key=lambda row: expected_period_ids.index(str(row["period_id"])))
        observed = [str(row["period_id"]) for row in rows]
        if observed != expected_period_ids:
            raise CorrectedAllPeriodError(
                f"comparison period rows missing or duplicated for {config_id}: {observed}"
            )
        selected_periods[config_id] = rows
    baseline_periods = [dict(row) for row in baseline["periods"]]

    neighbor_by_id = {str(row["config_id"]): row for row in neighbor_rows}
    source_by_id = {str(row["config_id"]): row for row in source_rows}
    comparison_rows: list[dict[str, Any]] = []
    periods_by_config: dict[str, list[dict[str, Any]]] = {}
    baseline_output_periods = [
        _comparison_period_row(
            config_id=BERLIN_BASELINE_CONFIG_ID, weights=BERLIN_V3_WEIGHTS, period=row,
        ) for row in baseline_periods
    ]
    periods_by_config[BERLIN_BASELINE_CONFIG_ID] = baseline_output_periods
    comparison_rows.extend(baseline_output_periods)
    comparison_rows.append(_comparison_period_row(
        config_id=BERLIN_BASELINE_CONFIG_ID,
        weights=BERLIN_V3_WEIGHTS,
        period=_baseline_aggregate_for_comparison(baseline),
        aggregate=True,
    ))
    for config_id in COMPARISON_CONFIG_IDS:
        aggregate = aggregate_by_id[config_id]
        weights = {
            key: float(aggregate[key])
            for key in ("G1", "G2", "G3", "G4", "G5", "quality_threshold")
        }
        outputs = [
            _comparison_period_row(config_id=config_id, weights=weights, period=row)
            for row in selected_periods[config_id]
        ]
        periods_by_config[config_id] = outputs
        comparison_rows.extend(outputs)
        aggregate_comparison = dict(aggregate)
        aggregate_comparison["period_id"] = "AGGREGATE"
        aggregate_comparison["sessions"] = sum(int(row["sessions"]) for row in outputs)
        aggregate_comparison["win_rate"] = (
            int(aggregate["wins"]) / int(aggregate["total_trades"])
            if int(aggregate["total_trades"]) else 0.0
        )
        comparison_rows.append(_comparison_period_row(
            config_id=config_id, weights=weights, period=aggregate_comparison, aggregate=True,
        ))
        source = source_by_id.get(config_id)
        if source is None or not _equal_number(
            source["native_mbp10_total_r"], aggregate["native_mbp10_total_r"],
        ) or not _equal_number(source["mbo_derived_total_r"], aggregate["mbo_derived_total_r"]):
            raise CorrectedAllPeriodError(f"source-model reporting mismatch: {config_id}")

    summary_rows: list[dict[str, Any]] = []
    baseline_summary_aggregate = dict(baseline_aggregate)
    summary_rows.append(_candidate_summary_row(
        config_id=BERLIN_BASELINE_CONFIG_ID,
        weights=BERLIN_V3_WEIGHTS,
        aggregate=baseline_summary_aggregate,
        period_rows=baseline_output_periods,
        neighbor=None,
        baseline_aggregate=baseline_aggregate,
    ))
    for config_id in COMPARISON_CONFIG_IDS:
        aggregate = aggregate_by_id[config_id]
        weights = {
            key: float(aggregate[key])
            for key in ("G1", "G2", "G3", "G4", "G5", "quality_threshold")
        }
        neighbor = neighbor_by_id.get(config_id)
        if neighbor is None:
            raise CorrectedAllPeriodError(f"neighbor row missing: {config_id}")
        summary_rows.append(_candidate_summary_row(
            config_id=config_id,
            weights=weights,
            aggregate=aggregate,
            period_rows=periods_by_config[config_id],
            neighbor=neighbor,
            baseline_aggregate=baseline_aggregate,
        ))
    if [str(row["config_id"]) for row in summary_rows] != [
        BERLIN_BASELINE_CONFIG_ID, *COMPARISON_CONFIG_IDS,
    ]:
        raise CorrectedAllPeriodError("comparison registry changed")

    # All validation is complete before any report artifact is replaced.
    _write_csv(aggregate_path, repaired_aggregate_rows)
    for top_path in sorted(optimizer_root.glob("top-*.csv")):
        top_rows = _read_csv(top_path)
        raw_top_digest = _strategy_value_digest(top_rows)
        repaired_top = [
            _explicit_reference_fields(row, baseline_aggregate) for row in top_rows
        ]
        if _strategy_value_digest(repaired_top) != raw_top_digest:
            raise CorrectedAllPeriodError(f"raw ranked strategy values changed: {top_path.name}")
        _write_csv(top_path, repaired_top)
    _write_csv(optimizer_root / "candidate-period-comparison.csv", comparison_rows)
    _write_csv(optimizer_root / "candidate-summary.csv", summary_rows)
    markdown = _candidate_comparison_markdown(summary_rows, periods_by_config)
    (optimizer_root / "candidate-period-comparison.md").write_text(markdown, encoding="utf-8")

    first = repaired_aggregate_rows[0]
    optimizer_summary.update({
        "reporting_baseline": "CORRECTED_BERLIN_V3_BASELINE",
        "berlin_v3_reference": {
            "trades": int(baseline_aggregate["trades"]),
            "total_r": baseline_aggregate["total_r"],
            "net_pnl_usd": baseline_aggregate["net_pnl_usd"],
        },
        "historical_v3_reference": {
            "trades": int(first["historical_v3_reference_trades"]),
            "total_r": float(first["historical_v3_reference_total_r"]),
            "net_pnl_usd": float(first["historical_v3_reference_net_pnl_usd"]),
        },
        "reporting_regeneration": {
            "status": "OFFLINE_REPORTING_REGENERATED",
            "raw_aggregate_strategy_values_sha256": raw_digest_before,
            "period_results_sha256": _sha256(period_path),
            "candidate_count_including_baseline": len(summary_rows),
            "optimizer_configs_recomputed": 0,
            "dbn_files_opened": 0,
            "network_calls": 0,
            "downloads": 0,
        },
    })
    _write_json(optimizer_summary_path, optimizer_summary)
    return {
        "status": "OFFLINE_BERLIN_REPORTING_REGENERATION_COMPLETE",
        "execution_contract_sha256": CONTRACT_SHA256,
        "evidence_label": EVIDENCE_LABEL,
        "berlin_v3_reference": optimizer_summary["berlin_v3_reference"],
        "historical_v3_reference": optimizer_summary["historical_v3_reference"],
        "candidate_summaries": summary_rows,
        "candidate_period_rows": len(comparison_rows),
        "aggregate_strategy_values_changed": False,
        "optimizer_configs_recomputed": 0,
        "dbn_files_opened": 0,
        "network_calls": 0,
        "downloads": 0,
        "output_root": str(optimizer_root),
    }


def _classify_corrected_path(
    tape: BerlinSessionCausalTape, interaction: Mapping[str, Any], overlap: Mapping[str, Any],
) -> tuple[str, str | None, dict[str, Any] | None]:
    if int(overlap["entry_timestamp_ns"]) >= tape.hard_flat_timestamp_ns:
        return "E_NEVER_ENTERED_POSITION", None, None
    try:
        outcome = tape.entry_outcome(interaction, int(overlap["entry_ordinal"]))
    except UnpricedSourceIntegrityFailure as exc:
        return "F_UNPRICED_SOURCE_INTEGRITY_FAILURE", str(exc), None
    if outcome.trade is None:
        return "E_NEVER_ENTERED_POSITION", outcome.terminal_reason, None
    trade = dict(outcome.trade)
    reason = str(trade["exit_reason"])
    classification = {
        "HARD_FLAT_BERLIN": "A_HARD_FLAT_BERLIN_BEFORE_MAINTENANCE",
        "DATA_GAP_3S_FORCE_FLAT": "C_DATA_GAP_3S_FORCE_FLAT",
        "SOURCE_END_FORCE_FLAT_LAST_VALID_BBO": "D_SOURCE_END_FORCE_FLAT_LAST_VALID_BBO",
        "TARGET": "B_TEMP_GAP_RESUMED_WITHIN_3S",
        "STOP": "B_TEMP_GAP_RESUMED_WITHIN_3S",
    }[reason]
    return classification, reason, trade


def run_audit(*, repository_root: Path, tape_root: Path, output_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    tape_root = _resolve(repository_root, tape_root)
    output_root = _resolve(repository_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"immutable corrected audit output exists: {output_root}")
    historical_audit = _read_json(repository_root / HISTORICAL_BOUNDARY_AUDIT)
    overlaps = historical_audit.get("overlaps")
    if not isinstance(overlaps, list) or len(overlaps) != EXPECTED_AFFECTED_PATHS:
        raise CorrectedAllPeriodError("historical 372-path audit contract mismatch")
    by_key = {(str(row["period_id"]), str(row["session_date"])): [] for row in overlaps}
    for row in overlaps:
        by_key[(str(row["period_id"]), str(row["session_date"]))].append(row)
    bundles = allp._period_bundles(repository_root, tape_root)
    classifications: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for bundle in bundles:
        interactions, _indexes, by_day = _period_inputs(bundle)
        interaction_by_id = {str(row["interaction_id"]): row for row in interactions}
        for day in bundle.days:
            print(f"BERLIN_AUDIT {bundle.period.period_id} {day}", flush=True)
            path_rows = by_key.get((bundle.period.period_id, day), ())
            hard_utc = berlin_hard_flat_utc(day)
            hard_ns = int(hard_utc.timestamp() * 1e9)
            original_terminal = _terminal_event_from_parquet(_tape_path(bundle, day))
            original_terminal_ns = int(original_terminal["timestamp_ns"])
            tape = (
                BerlinSessionCausalTape.from_parquet(day, _tape_path(bundle, day))
                if path_rows else None
            )
            session_classifications = []
            max_gap_ns = 0
            for overlap in path_rows:
                interaction = interaction_by_id.get(str(overlap["interaction_id"]))
                if interaction is None:
                    raise CorrectedAllPeriodError(f"audited interaction absent from compact tape: {overlap['interaction_id']}")
                assert tape is not None
                classification, terminal, trade = _classify_corrected_path(tape, interaction, overlap)
                session_classifications.append(classification)
                for boundary in overlap["boundaries"]:
                    if boundary["classification"] == "TEMPORARY_BOOK_RECONSTRUCTION":
                        max_gap_ns = max(max_gap_ns, int(boundary.get("duration_ns") or 0))
                classifications.append({
                    "period_id": bundle.period.period_id,
                    "session_date": day,
                    "interaction_id": overlap["interaction_id"],
                    "instrument": overlap["instrument"],
                    "old_diagnostic_resolution": overlap["diagnostic_resolution"],
                    "new_classification": classification,
                    "new_terminal_reason": terminal,
                    "new_trade": trade,
                })
            maintenance_start, maintenance_end = maintenance_window_utc(day)
            old_utc = datetime.fromisoformat(f"{day}T22:45:00+00:00")
            sessions.append({
                "period_id": bundle.period.period_id, "session_date": day,
                "hard_flat_local": execution_local_iso(day),
                "hard_flat_utc": hard_utc.isoformat(),
                "maintenance_start_utc": maintenance_start.isoformat(),
                "maintenance_end_utc": maintenance_end.isoformat(),
                "old_fixed_hard_flat_utc": old_utc.isoformat(),
                "old_minus_intended_minutes": int((old_utc - hard_utc).total_seconds() / 60),
                "old_replay_potential_position_crossed_intended_hard_flat": any(
                    value.startswith("A_") for value in session_classifications
                ),
                "affected_paths": len(path_rows),
                "temporary_gaps_while_position_path_open": sum(
                    sum(boundary["classification"] == "TEMPORARY_BOOK_RECONSTRUCTION" for boundary in row["boundaries"])
                    for row in path_rows
                ),
                "max_gap_duration_ns": max_gap_ns,
                "max_gap_duration_seconds": max_gap_ns / 1e9,
                "source_terminal_kind": (
                    "SOURCE_END" if original_terminal_ns < hard_ns else "HARD_FLAT_BERLIN"
                ),
                "source_terminal_utc": _iso_ns(min(original_terminal_ns, hard_ns)),
                "original_tape_terminal_type": original_terminal["event_type"],
                "original_tape_terminal_utc": _iso_ns(original_terminal_ns),
                "defensible_last_bbo_for_required_forced_exits": not any(
                    value.startswith("F_") for value in session_classifications
                ),
                "open_position_at_maintenance": False,
            })
    counts = Counter(row["new_classification"] for row in classifications)
    if len(classifications) != EXPECTED_AFFECTED_PATHS:
        raise CorrectedAllPeriodError("corrected path audit did not retain all 372 paths")
    result = {
        "status": "CORRECTED_BERLIN_HARDFLAT_87_SESSION_AUDIT_COMPLETE",
        "strategy_id": STRATEGY_ID,
        "execution_contract_sha256": CONTRACT_SHA256,
        "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
        "session_count": len(sessions), "affected_path_count": len(classifications),
        "reclassification_counts": dict(sorted(counts.items())),
        "unpriced_source_integrity_failures": counts["F_UNPRICED_SOURCE_INTEGRITY_FAILURE"],
        "scheduled_maintenance_open_positions": sum(row["open_position_at_maintenance"] for row in sessions),
        "existing_tapes_sufficient": counts["F_UNPRICED_SOURCE_INTEGRITY_FAILURE"] == 0,
        "local_tape_rebuild_required": counts["F_UNPRICED_SOURCE_INTEGRITY_FAILURE"] != 0,
        "network_calls": 0, "downloads": 0, "dbn_files_opened": 0,
        "sessions": sessions, "paths": classifications,
    }
    output_root.mkdir(parents=True)
    _write_json(output_root / "audit.json", result)
    _write_csv(output_root / "session-calendar-audit.csv", sessions)
    _write_csv(output_root / "path-reclassification.csv", [
        {key: value for key, value in row.items() if key != "new_trade"} for row in classifications
    ])
    (output_root / "audit-report.html").write_text(_audit_html(result), encoding="utf-8")
    return {key: value for key, value in result.items() if key not in {"sessions", "paths"}} | {
        "output_root": str(output_root)
    }


def _audit_html(result: Mapping[str, Any]) -> str:
    counts = "".join(
        f"<li>{html.escape(key)}: <b>{value}</b></li>"
        for key, value in result["reclassification_counts"].items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Berlin hard-flat audit</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:70rem}}code{{background:#eef;padding:.15rem}}</style></head><body>
<h1>Berlin hard-flat execution audit</h1><p>Historical V3 <code>{HISTORICAL_V3_CONTRACT_SHA256}</code> is untouched. Corrected contract <code>{CONTRACT_SHA256}</code>.</p>
<h2>372-path reclassification</h2><ul>{counts}</ul>
<p>Sessions: {result['session_count']}; unpriced cases: {result['unpriced_source_integrity_failures']}; maintenance crossings: {result['scheduled_maintenance_open_positions']}.</p>
<p>Existing compact tapes sufficient: {result['existing_tapes_sufficient']}. No DBN or network input.</p></body></html>"""


def contract_summary() -> dict[str, Any]:
    return {
        "status": "CORRECTED_EXECUTION_CONTRACT_READY",
        "strategy_id": STRATEGY_ID,
        "execution_contract_sha256": CONTRACT_SHA256,
        "historical_v3_contract_sha256": HISTORICAL_V3_CONTRACT_SHA256,
        "weight_count": len(matrix.generate_weight_grid()),
        "quality_threshold_count": len(matrix.QUALITY_THRESHOLDS),
        "configuration_count": len(matrix.configuration_registry()),
        "network_calls": 0, "downloads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("contract", "audit", "baseline", "preflight", "optimize", "report"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--tape-root", type=Path, default=TAPE_ROOT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--preflight-root", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OPTIMIZER_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            result = contract_summary()
        elif args.command == "audit":
            result = run_audit(
                repository_root=args.repository_root, tape_root=args.tape_root,
                output_root=args.audit_root,
            )
        elif args.command == "baseline":
            result = run_baseline(
                repository_root=args.repository_root, tape_root=args.tape_root,
                output_root=args.baseline_root,
            )
        elif args.command == "preflight":
            result = run_preflight(
                repository_root=args.repository_root, baseline_root=args.baseline_root,
                output_root=args.preflight_root,
            )
        elif args.command == "optimize":
            result = run_optimizer(
                repository_root=args.repository_root, tape_root=args.tape_root,
                output_root=args.output_root, preflight_root=args.preflight_root,
                baseline_root=args.baseline_root,
            )
        else:
            result = run_offline_reporting(
                repository_root=args.repository_root,
                optimizer_root=args.output_root,
                baseline_root=args.baseline_root,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
