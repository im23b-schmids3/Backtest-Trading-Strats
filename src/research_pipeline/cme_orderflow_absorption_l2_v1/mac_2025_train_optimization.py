"""Fail-closed TRAIN-only optimization preparation for the MAC 2025 tapes.

This module deliberately stops before Optuna.  It seals the exact TRAIN tape
set, creates an immutable Stage-0 baseline/constraint snapshot, runs the
deterministic vectorized weight/Q exploratory kernel, and freezes robust
objective scaling.  Validation, OOS data, raw DBN, and provider access are
outside this module's scope.

The Stage-1A kernel is an exploratory opportunity screen.  It uses the
existing vectorized qualification kernel and deterministic candidate-level
trade-path summaries prepared once from the already sealed tapes.  It does
not claim to replace exact one-position event-path replay; later shortlist
evaluation must use ``evaluate_candidate_tape`` before any Optuna result is
accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_es_only_train_baseline as baseline
from .model import L2Config


RUN_ID = "CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE"
TAPE_MANIFEST_NAME = "train-tape-manifest.json"
DEFAULT_TAPE_ROOT = Path("research_runs") / RUN_ID / "candidate-tapes"
DEFAULT_OUTPUT_ROOT = Path("research_runs") / RUN_ID / "train-optimization"
STAGE1A_SEED = 20250922
STAGE1A_TARGET = 1_000_000
STAGE1A_CHUNK_SIZE = 512
PROFIT_FACTOR_CAP = 10.0
ROBUST_SCALE_FLOOR = 1.0
OBJECTIVE_VERSION = "MAC2025_TRAIN_ROBUST_OBJECTIVE_V2"
OBJECTIVE_DEFINITION = {
    "median_date_r": {"weight": 0.10, "direction": "maximize", "description": "equal-date central result"},
    "lower_quartile_date_r": {"weight": 0.15, "direction": "maximize", "description": "date lower-tail consistency"},
    "median_session_date_r": {"weight": 0.10, "direction": "maximize", "description": "session diversity and consistency"},
    "profit_factor": {"weight": 0.15, "direction": "maximize", "description": "bounded profit-factor transform"},
    "profitable_date_ratio": {"weight": 0.10, "direction": "maximize", "description": "breadth of profitable dates"},
    "max_drawdown_r": {"weight": 0.15, "direction": "maximize", "description": "less negative drawdown"},
    "date_concentration": {"weight": 0.05, "direction": "minimize", "description": "top-five absolute date contribution"},
    "family_concentration": {"weight": 0.05, "direction": "minimize", "description": "top-three family absolute contribution"},
    "downside_tail": {"weight": 0.05, "direction": "maximize", "description": "10th-percentile date result"},
    "session_diversity": {"weight": 0.05, "direction": "maximize", "description": "number of active research sessions"},
    "trade_activity": {"weight": 0.10, "direction": "maximize", "description": "finite log-scaled activity; zero trade remains observable but ranks poorly"},
}


class TrainOptimizationError(RuntimeError):
    """The sealed TRAIN optimization contract cannot be satisfied."""


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _repo_relative(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainOptimizationError(f"artifact path escapes repository: {path}") from exc
    parts = {part.casefold() for part in resolved.relative_to(root).parts}
    if parts & {"validation", "oos", "holdout"}:
        raise TrainOptimizationError(f"Validation/OOS artifact rejected: {path}")
    return resolved


EXPECTED_TRAIN_DATES = tuple(baseline.TRAIN_DATES)


def validate_train_manifest_metadata(payload: Mapping[str, Any]) -> None:
    """Validate manifest identity without opening any tape files."""
    if payload.get("status") != "COMPLETE":
        raise TrainOptimizationError("TRAIN tape manifest is not COMPLETE")
    if tuple(payload.get("train_dates", ())) != EXPECTED_TRAIN_DATES:
        raise TrainOptimizationError("TRAIN tape manifest date allowlist mismatch")
    if payload.get("failed_dates") != []:
        raise TrainOptimizationError("TRAIN tape manifest contains failed dates")
    for flag in ("validation_performance", "final_oos_accessed", "optimization_run"):
        if payload.get(flag) is not False:
            raise TrainOptimizationError(f"sealed TRAIN manifest has invalid {flag} flag")
    reports = payload.get("reports")
    if not isinstance(reports, list) or len(reports) != len(EXPECTED_TRAIN_DATES):
        raise TrainOptimizationError("TRAIN tape manifest report count mismatch")
    dates = [row.get("date") for row in reports if isinstance(row, Mapping)]
    if len(dates) != len(set(dates)) or set(dates) != set(EXPECTED_TRAIN_DATES):
        raise TrainOptimizationError("TRAIN tape manifest has missing or duplicate dates")


@dataclass(frozen=True)
class TrainTapeBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    tapes: tuple[candidate_tape.CandidateTape, ...]
    reports: tuple[dict[str, Any], ...]

    @property
    def dates(self) -> tuple[str, ...]:
        return EXPECTED_TRAIN_DATES


def load_train_tapes(*, manifest_path: Path = DEFAULT_TAPE_ROOT / TAPE_MANIFEST_NAME,
                     repo_root: Path = Path(".")) -> TrainTapeBundle:
    """Load exactly the sealed 35 tapes through the manifest allowlist."""
    repo_root = repo_root.resolve()
    manifest_path = _repo_relative(manifest_path, repo_root)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainOptimizationError(f"unreadable TRAIN tape manifest: {manifest_path}") from exc
    validate_train_manifest_metadata(payload)
    manifest_root = manifest_path.parent
    current_semantic = candidate_tape._semantic_sha256()
    if payload.get("semantic_sha256") != current_semantic:
        raise TrainOptimizationError("TRAIN manifest semantic hash does not match current evaluator")
    reports_by_date = {row["date"]: dict(row) for row in payload["reports"]}
    tapes: list[candidate_tape.CandidateTape] = []
    ordered_reports: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for day in EXPECTED_TRAIN_DATES:
        report = reports_by_date[day]
        tape_path = _repo_relative(report.get("tape_path", ""), repo_root)
        if tape_path in seen_paths:
            raise TrainOptimizationError(f"duplicate tape path for {day}")
        seen_paths.add(tape_path)
        source_path = _repo_relative(report.get("source_path", ""), repo_root)
        expected_source = payload.get("source_sha256_by_date", {}).get(day)
        if not expected_source or report.get("source_sha256") != expected_source:
            raise TrainOptimizationError(f"source hash identity mismatch for {day}")
        if not source_path.is_file() or _sha256(source_path) != expected_source:
            raise TrainOptimizationError(f"source hash verification failed for {day}")
        if not tape_path.is_file() or _sha256(tape_path) != report.get("tape_sha256"):
            raise TrainOptimizationError(f"tape hash verification failed for {day}")
        tape_manifest_path = tape_path.with_name(f"{day}-candidate-tape-manifest.json")
        report_path = manifest_root / "reports" / f"{day}.json"
        # The BBO-complete rebuild writes a versioned checkpoint alongside the
        # older sparse-tape checkpoint.  Prefer it when present; otherwise
        # retain compatibility with the original sealed tape layout.
        checkpoint_candidates = (
            manifest_root / "_checkpoints" / f"{day}-bbo.json",
            manifest_root / "_checkpoints" / f"{day}.json",
        )
        checkpoint_path = next((path for path in checkpoint_candidates if path.is_file()), checkpoint_candidates[-1])
        for identity_path in (tape_manifest_path, report_path, checkpoint_path):
            if not identity_path.is_file():
                raise TrainOptimizationError(f"missing identity artifact for {day}: {identity_path}")
        try:
            tape_meta = json.loads(tape_manifest_path.read_text(encoding="utf-8"))
            date_report = json.loads(report_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainOptimizationError(f"unreadable identity artifact for {day}") from exc
        if tape_meta.get("date") != day or tape_meta.get("source_sha256") != expected_source:
            raise TrainOptimizationError(f"tape manifest identity mismatch for {day}")
        if date_report != report:
            raise TrainOptimizationError(f"date report identity mismatch for {day}")
        checkpoint_status = checkpoint.get("status")
        if (checkpoint_status not in {"DATE_COMPLETE", "BBO_PATH_COMPLETE"} or
                checkpoint.get("date") != day or
                checkpoint.get("source_sha256") != expected_source or
                checkpoint.get("semantic_sha256") != current_semantic or
                checkpoint.get("tape_sha256") != report.get("tape_sha256") or
                checkpoint.get("report") != report or
                (checkpoint_status == "DATE_COMPLETE" and
                 payload.get("config_sha256") is not None and
                 checkpoint.get("config_sha256") != payload.get("config_sha256"))):
            raise TrainOptimizationError(f"checkpoint identity mismatch for {day}")
        tape = candidate_tape.load_tape(tape_path, source_sha256=expected_source,
                                        semantic_sha256=current_semantic)
        if tape.metadata.get("date") != day or int(tape.metadata.get("candidate_count", -1)) != len(tape.candidates):
            raise TrainOptimizationError(f"tape date/count mismatch for {day}")
        tapes.append(tape)
        ordered_reports.append(report)
    return TrainTapeBundle(manifest_path, dict(payload), tuple(tapes), tuple(ordered_reports))


def _max_drawdown(values: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return float(drawdown)


def _profit_factor(values: Sequence[float]) -> tuple[float | None, float]:
    positive = sum(value for value in values if value > 0)
    negative = -sum(value for value in values if value < 0)
    if negative == 0:
        return None, PROFIT_FACTOR_CAP if positive > 0 else 0.0
    return positive / negative, min(PROFIT_FACTOR_CAP, positive / negative)


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q)) if values else 0.0


def _group_trade_metrics(trades: Sequence[Mapping[str, Any]], keys: Sequence[str], expected_keys: Sequence[Any]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {
        (key,) if len(keys) == 1 else tuple(key): [] for key in expected_keys
    }
    for trade in trades:
        group = tuple(trade.get(key) for key in keys)
        groups.setdefault(group, []).append(_finite(trade.get("r_multiple"), default=0.0))
    result: dict[str, dict[str, Any]] = {}
    for group, values in groups.items():
        raw_pf, capped_pf = _profit_factor(values)
        result["|".join(str(item) for item in group)] = {
            "trade_count": len(values), "net_r": sum(values),
            "profit_factor": raw_pf, "profit_factor_capped": capped_pf,
            "max_drawdown_r": _max_drawdown(values),
            "active": bool(values), "profitable": bool(values and sum(values) > 0),
        }
    return result


def _concentration(values: Mapping[str, float], top_n: int) -> float:
    magnitudes = sorted((abs(float(value)) for value in values.values()), reverse=True)
    total = sum(magnitudes)
    return float(sum(magnitudes[:top_n]) / total) if total else 0.0


def _snapshot_metrics(*, trades: Sequence[Mapping[str, Any]], dates: Sequence[str], sessions: Sequence[str], exact_trade_dd: bool = True) -> dict[str, Any]:
    date_values = {day: [] for day in dates}
    session_values = {(day, session): [] for day in dates for session in sessions}
    family_values: dict[str, list[float]] = {}
    for trade in trades:
        value = _finite(trade.get("r_multiple"), default=0.0)
        day = str(trade.get("date"))
        date_values.setdefault(day, []).append(value)
        session = str(trade.get("trading_session", trade.get("session", "UNKNOWN")))
        session_values.setdefault((day, session), []).append(value)
        family = str(trade.get("family_id", trade.get("level", "UNKNOWN")))
        family_values.setdefault(family, []).append(value)
    date_r = {day: sum(values) for day, values in date_values.items()}
    session_date_r = {f"{day}|{session}": sum(session_values[(day, session)])
                      for day in dates for session in sessions}
    values = [_finite(trade.get("r_multiple"), default=0.0) for trade in trades]
    raw_pf, capped_pf = _profit_factor(values)
    active_dates = [day for day, rows in date_values.items() if rows]
    profitable_dates = [day for day, result in date_r.items() if result > 0]
    session_active_dates = {session: sum(bool(session_values[(day, session)]) for day in dates) for session in sessions}
    return {
        "total_trades": len(values), "net_r": sum(values),
        "gross_pnl_usd": sum(_finite(trade.get("gross_pnl_usd"), default=0.0) for trade in trades),
        "profit_factor": raw_pf, "profit_factor_capped": capped_pf,
        "max_drawdown_r": _max_drawdown(values) if exact_trade_dd else _max_drawdown(list(date_r.values())),
        "active_dates": len(active_dates), "profitable_date_ratio": len(profitable_dates) / len(dates),
        "date_r": date_r, "session_date_r": session_date_r,
        "session_active_dates": session_active_dates,
        "active_sessions": sum(value > 0 for value in session_active_dates.values()),
        "date_concentration_top1": _concentration(date_r, 1),
        "date_concentration_top3": _concentration(date_r, 3),
        "date_concentration_top5": _concentration(date_r, 5),
        "family_concentration_top1": _concentration({key: sum(value) for key, value in family_values.items()}, 1),
        "family_concentration_top3": _concentration({key: sum(value) for key, value in family_values.items()}, 3),
        "family_concentration_top5": _concentration({key: sum(value) for key, value in family_values.items()}, 5),
        "lower_quartile_date_r": _quantile(list(date_r.values()), 0.25),
        "p10_date_r": _quantile(list(date_r.values()), 0.10),
        "median_date_r": _quantile(list(date_r.values()), 0.50),
        "lower_quartile_session_date_r": _quantile(list(session_date_r.values()), 0.25),
        "median_session_date_r": _quantile(list(session_date_r.values()), 0.50),
        "per_date": _group_trade_metrics(trades, ("date",), dates),
        "per_session_date": _group_trade_metrics(
            trades, ("date", "trading_session"), [(day, session) for day in dates for session in sessions]
        ),
    }


def stage0_baseline(bundle: TrainTapeBundle) -> dict[str, Any]:
    """Evaluate the frozen baseline once, entirely from candidate tapes."""
    all_trades: list[dict[str, Any]] = []
    per_date: dict[str, dict[str, Any]] = {}
    per_session: dict[str, dict[str, Any]] = {}
    for day, tape in zip(bundle.dates, bundle.tapes):
        result = candidate_tape.evaluate_candidate_tape(tape, config=L2Config())
        candidate_by_id = {str(row.get("interaction_id")): row for row in tape.candidates}
        enriched = []
        for trade in result["trades"]:
            row = candidate_by_id.get(str(trade.get("interaction_id")), {})
            enriched.append({**trade, "date": day, "trading_session": row.get("trading_session"),
                             "family_id": row.get("family_id")})
        all_trades.extend(enriched)
        per_date[day] = _snapshot_metrics(trades=enriched, dates=(day,), sessions=("ASIA", "EUROPE", "NY"))
        for session in ("ASIA", "EUROPE", "NY"):
            session_trades = [row for row in enriched if row.get("trading_session") == session]
            per_session[f"{day}|{session}"] = _snapshot_metrics(
                trades=session_trades, dates=(day,), sessions=(session,)
            )
    aggregate = _snapshot_metrics(trades=all_trades, dates=bundle.dates, sessions=("ASIA", "EUROPE", "NY"))
    session_summary: dict[str, dict[str, Any]] = {}
    for session in ("ASIA", "EUROPE", "NY"):
        summary = _snapshot_metrics(
            trades=[row for row in all_trades if row.get("trading_session") == session],
            dates=bundle.dates, sessions=(session,),
        )
        summary["active_date_count"] = summary["active_dates"]
        summary["profitable_date_ratio"] = sum(
            value > 0 for value in summary["date_r"].values()
        ) / len(bundle.dates)
        session_summary[session] = summary
    return {
        "status": "COMPLETE", "run_id": RUN_ID, "stage": "STAGE0_BASELINE",
        "baseline_parameters": candidate_tape._default_parameters(L2Config()),
        "aggregate": aggregate, "per_date": per_date, "per_session_date": per_session,
        "per_session": session_summary,
        "frozen_class_c_parameters": bundle.manifest.get("frozen_class_c_parameters", {}),
        "source_manifest": str(bundle.manifest_path), "train_dates": list(bundle.dates),
        "validation_performance": False, "final_oos_accessed": False, "optimization_run": False,
    }


def derive_constraints(stage0: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return technical validity rules only.

    Stage-0 performance is retained as descriptive context, never as a
    rejection rule.  In particular, this function intentionally contains no
    trade, date, profitability, drawdown, concentration, or session floor.
    """
    return {
        "version": "MAC2025_TRAIN_TECHNICAL_VALIDITY_V2",
        "source_stage": "STAGE1A_PARAMETER_SCHEMA",
        "performance_constraints_removed": True,
        "rules": {
            "finite_parameter_vector": True,
            "weight_dimension": 5,
            "weights_strictly_positive": True,
            "weights_sum": 1.0,
            "weights_sum_absolute_tolerance": 1.0e-12,
            "min_quality_range": [0.0, 1.0],
            "finite_metric_vectors": True,
            "evaluator_failure_is_invalid": True,
            "manifest_hash_integrity_required": True,
        },
    }


@dataclass(frozen=True)
class FastCandidatePool:
    scores: np.ndarray
    raw_mask: np.ndarray
    trade_r: np.ndarray
    has_trade: np.ndarray
    date_index: np.ndarray
    session_index: np.ndarray
    family_index: np.ndarray
    family_count: int


def _candidate_raw_mask(tape: candidate_tape.CandidateTape) -> np.ndarray:
    return np.asarray([
        float(row.get("directional_aggressive_volume", 0) or 0) >= 50
        and float(row.get("relevant_execution_count", 0) or 0) >= 2
        and float(row.get("consume_restore_cycles", 0) or 0) >= 1
        and not (float(row.get("maximum_through_level_progress_ticks", 0) or 0) > 4.0
                 and float(row.get("interaction_rejection_ticks", 0) or 0) < 0.25)
        for row in tape.candidates
    ], dtype=bool)


def _prepare_fast_pool(bundle: TrainTapeBundle) -> FastCandidatePool:
    """Prepare candidate-level outcomes once; no DBN access occurs here."""
    score_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    trade_rows: list[np.ndarray] = []
    has_rows: list[np.ndarray] = []
    date_rows: list[np.ndarray] = []
    session_rows: list[np.ndarray] = []
    family_rows: list[np.ndarray] = []
    family_names: dict[str, int] = {}
    params = candidate_tape._default_parameters(L2Config())
    for date_index, (day, tape) in enumerate(zip(bundle.dates, bundle.tapes)):
        # The raw qualification predicate is fixed by the supported fast
        # kernel.  Discarding rows that can never enter Stage-1A is therefore
        # semantics-preserving and keeps the one-time outcome preparation
        # bounded on the real 35-day set.
        candidate_indices = np.flatnonzero(_candidate_raw_mask(tape))
        score_rows.append(np.asarray([[float(tape.candidates[index].get(name, 0.0) or 0.0)
                                       for name in candidate_tape.SCORE_NAMES]
                                      for index in candidate_indices], dtype=np.float64))
        raw_rows.append(np.ones(len(candidate_indices), dtype=bool))
        trade_values = np.zeros(len(candidate_indices), dtype=np.float64)
        has = np.zeros(len(candidate_indices), dtype=bool)
        session = np.full(len(candidate_indices), -1, dtype=np.int16)
        family = np.zeros(len(candidate_indices), dtype=np.int32)
        for position, index in enumerate(candidate_indices):
            row = tape.candidates[index]
            confirmation = candidate_tape._confirmation(tape, row, params)
            if confirmation is not None:
                trade = candidate_tape._trade_for_candidate(tape, row, confirmation, params)
                if trade is not None:
                    trade_values[position] = _finite(trade.get("r_multiple"), default=0.0)
                    has[position] = True
            session[position] = {"ASIA": 0, "EUROPE": 1, "NY": 2}.get(str(row.get("trading_session")), -1)
            family_name = str(row.get("family_id", row.get("level", "UNKNOWN")))
            if family_name not in family_names:
                family_names[family_name] = len(family_names)
            family[position] = family_names[family_name]
        trade_rows.append(trade_values)
        has_rows.append(has)
        date_rows.append(np.full(len(candidate_indices), date_index, dtype=np.int16))
        session_rows.append(session)
        family_rows.append(family)
    return FastCandidatePool(
        scores=np.concatenate(score_rows), raw_mask=np.concatenate(raw_rows),
        trade_r=np.concatenate(trade_rows), has_trade=np.concatenate(has_rows),
        date_index=np.concatenate(date_rows), session_index=np.concatenate(session_rows),
        family_index=np.concatenate(family_rows), family_count=len(family_names),
    )


def _sample_population(count: int, seed: int = STAGE1A_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.random((count, 5), dtype=np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    q_values = rng.uniform(0.0, 1.0, count)
    if count:
        config = L2Config()
        weights[0] = [getattr(config, name) for name in candidate_tape.WEIGHT_NAMES]
        weights[0] /= weights[0].sum()
        q_values[0] = float(config.min_quality_score)
    return weights, q_values


def _metrics_from_selected(selected: np.ndarray, pool: FastCandidatePool, config_count: int) -> dict[str, np.ndarray]:
    selected = selected & pool.has_trade[:, None]
    selected_float = selected.astype(np.float64, copy=False)
    date_r = np.zeros((config_count, len(EXPECTED_TRAIN_DATES)), dtype=np.float64)
    session_date_r = np.zeros((config_count, len(EXPECTED_TRAIN_DATES) * 3), dtype=np.float64)
    for day in range(len(EXPECTED_TRAIN_DATES)):
        mask = pool.date_index == day
        date_r[:, day] = selected_float[mask].T @ pool.trade_r[mask]
        for session in range(3):
            slot = day * 3 + session
            session_mask = mask & (pool.session_index == session)
            session_date_r[:, slot] = selected_float[session_mask].T @ pool.trade_r[session_mask]
    family_counts = np.zeros((config_count, pool.family_count), dtype=np.float64)
    for family in range(pool.family_count):
        family_counts[:, family] = selected[pool.family_index == family, :].sum(axis=0)
    positive = np.maximum(pool.trade_r, 0.0)
    negative = np.maximum(-pool.trade_r, 0.0)
    gross_profit = selected_float.T @ positive
    gross_loss = selected_float.T @ negative
    pf = np.divide(gross_profit, gross_loss, out=np.full(config_count, PROFIT_FACTOR_CAP), where=gross_loss > 0)
    # A configuration with no trades is valid exploratory data, but it is not
    # a zero-loss winner.  Keep the explicit cap only for positive gross
    # profit with no losses; zero profit and zero loss maps to PF=0.
    pf = np.where((gross_loss == 0) & (gross_profit == 0), 0.0, pf)
    pf = np.minimum(PROFIT_FACTOR_CAP, pf)
    abs_date = np.abs(date_r)
    abs_total = abs_date.sum(axis=1)
    sorted_dates = np.sort(abs_date, axis=1)[:, ::-1]
    abs_family = family_counts
    family_total = abs_family.sum(axis=1)
    sorted_family = np.sort(abs_family, axis=1)[:, ::-1]
    equity = np.cumsum(date_r, axis=1)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0), axis=1)
    return {
        "total_trades": selected.sum(axis=0).astype(np.float64),
        "net_r": selected_float.T @ pool.trade_r,
        "profit_factor": pf,
        "max_drawdown_r": np.min(equity - peak, axis=1),
        "active_dates": (np.abs(date_r) > 0).sum(axis=1).astype(np.float64),
        "profitable_date_ratio": (date_r > 0).sum(axis=1) / len(EXPECTED_TRAIN_DATES),
        "median_date_r": np.quantile(date_r, 0.50, axis=1),
        "lower_quartile_date_r": np.quantile(date_r, 0.25, axis=1),
        "downside_tail": np.quantile(date_r, 0.10, axis=1),
        "median_session_date_r": np.quantile(session_date_r, 0.50, axis=1),
        "date_concentration": np.divide(sorted_dates[:, :5].sum(axis=1), abs_total, out=np.zeros(config_count), where=abs_total > 0),
        "family_concentration": np.divide(sorted_family[:, :3].sum(axis=1), family_total, out=np.zeros(config_count), where=family_total > 0),
        "active_sessions": (np.abs(session_date_r.reshape(config_count, len(EXPECTED_TRAIN_DATES), 3)).sum(axis=1) > 0).sum(axis=1).astype(np.float64),
    }


def _technically_valid(metrics: Mapping[str, float], *, weights: Sequence[float], min_q: float) -> bool:
    """Reject only malformed/non-finite parameter or metric states."""
    try:
        vector = np.asarray(weights, dtype=np.float64)
        q_value = float(min_q)
    except (TypeError, ValueError):
        return False
    if vector.shape != (5,) or not np.all(np.isfinite(vector)) or not math.isfinite(q_value):
        return False
    if np.any(vector <= 0.0) or not 0.0 <= q_value <= 1.0:
        return False
    if not math.isclose(float(vector.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        return False
    return all(math.isfinite(float(value)) for value in metrics.values())


def _metrics_row(metrics: Mapping[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in metrics.items()}


def run_stage1a(*, bundle: TrainTapeBundle, constraints: Mapping[str, Any], target: int = STAGE1A_TARGET,
                seed: int = STAGE1A_SEED, chunk_size: int = STAGE1A_CHUNK_SIZE,
                output_path: Path | None = None) -> dict[str, Any]:
    if target <= 0 or chunk_size <= 0:
        raise ValueError("target and chunk_size must be positive")
    pool = _prepare_fast_pool(bundle)
    weights, q_values = _sample_population(target, seed)
    metric_names = ("total_trades", "net_r", "profit_factor", "max_drawdown_r", "active_dates",
                    "profitable_date_ratio", "median_date_r", "lower_quartile_date_r",
                    "downside_tail", "median_session_date_r", "date_concentration",
                    "family_concentration", "active_sessions")
    arrays = {name: np.zeros(target, dtype=np.float64) for name in metric_names}
    technically_valid = np.zeros(target, dtype=bool)
    score_columns = pool.scores[:, :5]
    penalty = pool.scores[:, 5:6] * 0.25
    started = time.perf_counter()
    for start in range(0, target, chunk_size):
        stop = min(target, start + chunk_size)
        quality = score_columns @ weights[start:stop].T - penalty
        selected = pool.raw_mask[:, None] & (quality >= q_values[start:stop][None, :])
        chunk_metrics = _metrics_from_selected(selected, pool, stop - start)
        for name in metric_names:
            arrays[name][start:stop] = chunk_metrics[name]
        for offset in range(stop - start):
            technically_valid[start + offset] = _technically_valid(
                {name: float(chunk_metrics[name][offset]) for name in metric_names},
                weights=weights[start + offset], min_q=q_values[start + offset],
            )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, weights=weights, min_q=q_values,
                            technically_valid=technically_valid, feasible=technically_valid,
                            **arrays)
    feasible_indices = np.flatnonzero(technically_valid)
    return {
        "status": "COMPLETE", "version": "MAC2025_STAGE1A_V1", "seed": seed,
        "target_configurations": target, "configurations_evaluated": target,
        "feasible_configurations": int(technically_valid.sum()),
        "technically_valid_configurations": int(technically_valid.sum()), "chunk_size": chunk_size,
        "runtime_seconds": time.perf_counter() - started,
        "output_path": str(output_path) if output_path else None,
        "screen_semantics": "vectorized_weight_q_candidate_trade_opportunity_screen",
        "exact_event_path_replay_required_before_acceptance": True,
        "feasible_indices": feasible_indices.tolist(),
        "weights": weights, "min_q": q_values, "feasible": technically_valid,
        "technically_valid": technically_valid, "metrics": arrays,
    }


def _scaling_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise TrainOptimizationError("cannot scale empty/non-finite objective population")
    median = float(np.quantile(values, 0.50))
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median, "mad": mad, "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "scale": max(mad, ROBUST_SCALE_FLOOR),
    }


def build_objective_scaling(stage1a: Mapping[str, Any], *, valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    technically_valid = np.asarray(
        valid_mask if valid_mask is not None else stage1a.get("technically_valid", stage1a["feasible"]),
        dtype=bool,
    )
    if not technically_valid.any():
        raise TrainOptimizationError("Stage-1A produced no technically valid configurations")
    metrics = stage1a["metrics"]
    scaling = {}
    for name in OBJECTIVE_DEFINITION:
        values = (np.asarray(metrics["active_sessions"], dtype=np.float64) / 3.0
                  if name == "session_diversity" else
                  np.log1p(np.asarray(metrics["total_trades"], dtype=np.float64))
                  if name == "trade_activity" else
                  np.asarray(metrics[name], dtype=np.float64))
        if name == "profit_factor":
            values = np.log1p(np.minimum(PROFIT_FACTOR_CAP, np.maximum(0.0, values))) / math.log1p(PROFIT_FACTOR_CAP)
        scaling[name] = _scaling_stats(values[technically_valid])
    return {
        "status": "COMPLETE", "version": "MAC2025_OBJECTIVE_SCALING_V2",
        "objective_version": OBJECTIVE_VERSION, "source": "ALL_TECHNICALLY_VALID_STAGE1A",
        "technically_valid_configurations": int(technically_valid.sum()),
        "performance_constraints": "NONE",
        "profit_factor_cap": PROFIT_FACTOR_CAP,
        "robust_scale_floor": ROBUST_SCALE_FLOOR, "scaling": scaling,
    }


def robust_objective(metrics: Mapping[str, float], scaling: Mapping[str, Any],
                     definition: Mapping[str, Any] = OBJECTIVE_DEFINITION) -> dict[str, float]:
    """Return finite component scores and the explicitly weighted robust score."""
    components: dict[str, float] = {}
    for name, spec in definition.items():
        value = _finite(
            metrics.get(name,
                        _finite(metrics.get("active_sessions"), default=0.0) / 3.0
                        if name == "session_diversity" else
                        math.log1p(max(0.0, _finite(metrics.get("total_trades"), default=0.0)))
                        if name == "trade_activity" else 0.0),
            default=0.0,
        )
        if name == "profit_factor":
            value = math.log1p(min(PROFIT_FACTOR_CAP, max(0.0, value))) / math.log1p(PROFIT_FACTOR_CAP)
        if spec["direction"] == "minimize":
            value = -value
        stats = scaling["scaling"][name]
        score = (value - float(stats["median"])) / max(float(stats["scale"]), ROBUST_SCALE_FLOOR)
        components[name] = float(score)
    total = sum(float(definition[name]["weight"]) * components[name] for name in definition)
    components["robust_score"] = float(total)
    if not all(math.isfinite(value) for value in components.values()):
        raise TrainOptimizationError("robust objective produced non-finite score")
    return components


def _population_from_npz(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load and technically validate the existing Stage-1A result only."""
    required = {
        "weights", "min_q", "net_r", "total_trades", "profit_factor", "max_drawdown_r",
        "active_dates", "profitable_date_ratio", "median_date_r", "lower_quartile_date_r",
        "downside_tail", "median_session_date_r", "date_concentration", "family_concentration",
        "active_sessions",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise TrainOptimizationError(f"Stage-1A result lacks required metrics: {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files
                      if name in required or name == "technically_valid"}
    except (OSError, ValueError, KeyError) as exc:
        raise TrainOptimizationError(f"unreadable Stage-1A result: {path}") from exc
    count = len(arrays["min_q"])
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    q_values = np.asarray(arrays["min_q"], dtype=np.float64)
    if weights.shape != (count, 5) or q_values.shape != (count,):
        raise TrainOptimizationError("malformed Stage-1A parameter arrays")
    mask = (np.all(np.isfinite(weights), axis=1) & np.isfinite(q_values) &
            (weights > 0.0).all(axis=1) & (q_values >= 0.0) & (q_values <= 1.0) &
            (np.abs(weights.sum(axis=1) - 1.0) <= 1.0e-12))
    for name in required - {"weights", "min_q"}:
        values = np.asarray(arrays[name], dtype=np.float64)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise TrainOptimizationError(f"non-finite or malformed Stage-1A metric: {name}")
        arrays[name] = values
    arrays["weights"] = weights
    arrays["min_q"] = q_values
    arrays["profit_factor"] = np.where(arrays["total_trades"] == 0, 0.0, arrays["profit_factor"])
    arrays["technically_valid"] = mask
    return arrays, mask


def _population_objective_scores(arrays: Mapping[str, np.ndarray], scaling: Mapping[str, Any]) -> np.ndarray:
    scores = np.zeros(len(arrays["min_q"]), dtype=np.float64)
    for name, spec in OBJECTIVE_DEFINITION.items():
        values = (np.asarray(arrays["active_sessions"], dtype=np.float64) / 3.0
                  if name == "session_diversity" else
                  np.log1p(np.asarray(arrays["total_trades"], dtype=np.float64))
                  if name == "trade_activity" else np.asarray(arrays[name], dtype=np.float64))
        if name == "profit_factor":
            values = np.log1p(np.minimum(PROFIT_FACTOR_CAP, np.maximum(0.0, values))) / math.log1p(PROFIT_FACTOR_CAP)
        if spec["direction"] == "minimize":
            values = -values
        stat = scaling["scaling"][name]
        scores += float(spec["weight"]) * (values - float(stat["median"])) / max(float(stat["scale"]), ROBUST_SCALE_FLOOR)
    if not np.all(np.isfinite(scores)):
        raise TrainOptimizationError("population robust scores are non-finite")
    return scores


def _distribution(values: np.ndarray, *, direction: str) -> dict[str, float | str]:
    values = np.asarray(values, dtype=np.float64)
    order = "minimize" if direction == "minimize" else "maximize"
    return {
        "direction": order,
        "best": float(np.min(values) if order == "minimize" else np.max(values)),
        "q99_9": float(np.quantile(values, 0.999)), "q99": float(np.quantile(values, 0.99)),
        "q95": float(np.quantile(values, 0.95)), "q90": float(np.quantile(values, 0.90)),
        "median": float(np.quantile(values, 0.50)),
        "worst": float(np.max(values) if order == "minimize" else np.min(values)),
    }


def _config_row(index: int, arrays: Mapping[str, np.ndarray], robust_scores: np.ndarray) -> dict[str, Any]:
    return {
        "index": int(index),
        "aggression_weight": float(arrays["weights"][index, 0]),
        "restoration_weight": float(arrays["weights"][index, 1]),
        "price_resistance_weight": float(arrays["weights"][index, 2]),
        "persistence_weight": float(arrays["weights"][index, 3]),
        "multi_level_support_weight": float(arrays["weights"][index, 4]),
        "min_quality_score": float(arrays["min_q"][index]),
        "net_r": float(arrays["net_r"][index]), "trades": float(arrays["total_trades"][index]),
        "profit_factor": float(arrays["profit_factor"][index]),
        "max_drawdown_r": float(arrays["max_drawdown_r"][index]),
        "active_dates": float(arrays["active_dates"][index]),
        "profitable_date_ratio": float(arrays["profitable_date_ratio"][index]),
        "date_concentration_top5": float(arrays["date_concentration"][index]),
        "family_concentration_top3": float(arrays["family_concentration"][index]),
        "robust_score": float(robust_scores[index]),
    }


def _ranked_indices(values: np.ndarray, *, descending: bool = True) -> np.ndarray:
    return np.argsort(-values if descending else values, kind="mergesort")


def _parameter_regions(arrays: Mapping[str, np.ndarray], robust_scores: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    params = ("aggression_weight", "restoration_weight", "price_resistance_weight",
              "persistence_weight", "multi_level_support_weight", "min_quality_score")
    population = {
        "top_100": _ranked_indices(robust_scores)[valid[_ranked_indices(robust_scores)]][:100],
        "top_500": _ranked_indices(robust_scores)[valid[_ranked_indices(robust_scores)]][:500],
        "top_1000": _ranked_indices(robust_scores)[valid[_ranked_indices(robust_scores)]][:1000],
        "top_1_percent": _ranked_indices(robust_scores)[valid[_ranked_indices(robust_scores)]][:max(1, int(valid.sum() * 0.01))],
    }
    output: dict[str, Any] = {}
    for label, indices in population.items():
        matrix = np.column_stack([arrays["weights"][indices, i] for i in range(5)] + [arrays["min_q"][indices]])
        summary = {}
        for column, name in enumerate(params):
            values = matrix[:, column]
            bins = np.floor(values * 10.0).astype(int)
            counts = {str(int(key)): int(value) for key, value in zip(*np.unique(bins, return_counts=True))}
            summary[name] = {
                "min": float(values.min()), "q10": float(np.quantile(values, 0.10)),
                "median": float(np.quantile(values, 0.50)), "q90": float(np.quantile(values, 0.90)),
                "max": float(values.max()), "dominant_decile": int(max(counts, key=counts.get)),
                "decile_counts": counts,
            }
        output[label] = {"count": int(len(indices)), "parameters": summary}
    return output


def _dedup_append(selected: list[int], index: int, arrays: Mapping[str, np.ndarray], tolerance: float = 1.0e-8) -> bool:
    vector = np.r_[arrays["weights"][index], arrays["min_q"][index]]
    for existing in selected:
        other = np.r_[arrays["weights"][existing], arrays["min_q"][existing]]
        if np.max(np.abs(vector - other)) <= tolerance:
            return False
    selected.append(int(index))
    return True


def _build_shortlist(arrays: Mapping[str, np.ndarray], robust_scores: np.ndarray, valid: np.ndarray,
                     *, seed: int = STAGE1A_SEED) -> tuple[list[int], dict[str, list[int]]]:
    valid_indices = np.flatnonzero(valid)
    by_robust = _ranked_indices(robust_scores)
    categories: dict[str, list[int]] = {"A_robust": [], "B_net_r": [], "C_diverse": [],
                                        "D_low_frequency": [], "E_medium_frequency": [], "F_random_control": []}
    selected: list[int] = []
    def add_from(label: str, candidates: Iterable[int], limit: int) -> None:
        for index in candidates:
            if len(categories[label]) >= limit:
                break
            if bool(valid[index]) and _dedup_append(selected, int(index), arrays):
                categories[label].append(int(index))
    add_from("A_robust", by_robust, 75)
    add_from("B_net_r", _ranked_indices(arrays["net_r"]), 50)
    # Quantized parameter cells provide deterministic diversity across weight/Q
    # regions without importing a clustering dependency or fitting to outcomes.
    diverse = []
    diverse_keys: set[tuple[int, ...]] = set()
    for index in by_robust:
        if not valid[index]:
            continue
        key = tuple(np.floor(np.r_[arrays["weights"][index], arrays["min_q"][index]] * 10.0).astype(int))
        if key not in diverse_keys:
            diverse_keys.add(key)
            diverse.append(int(index))
        if len(diverse) >= 500:
            break
    add_from("C_diverse", diverse, 50)
    add_from("D_low_frequency", by_robust[np.asarray(arrays["total_trades"])[by_robust] <= 100], 25)
    add_from("E_medium_frequency", by_robust[(np.asarray(arrays["total_trades"])[by_robust] >= 101) &
                                              (np.asarray(arrays["total_trades"])[by_robust] <= 250)], 25)
    remaining = [int(index) for index in valid_indices if index not in set(selected)]
    rng = np.random.default_rng(seed)
    if remaining:
        random_indices = rng.choice(np.asarray(remaining), size=min(25, len(remaining)), replace=False)
        add_from("F_random_control", random_indices.tolist(), 25)
    return selected, categories


@dataclass(frozen=True)
class ExactOpportunityPool:
    rows: tuple[dict[str, Any], ...]
    family_ids: tuple[str, ...]


def _build_exact_opportunity_pool(bundle: TrainTapeBundle) -> ExactOpportunityPool:
    rows: list[dict[str, Any]] = []
    family_ids = sorted({str(row.get("family_id", row.get("level", "UNKNOWN")))
                         for tape in bundle.tapes for row in tape.candidates})
    params = candidate_tape._default_parameters(L2Config())
    for day, tape in zip(bundle.dates, bundle.tapes):
        raw_mask = _candidate_raw_mask(tape)
        for original_index in np.flatnonzero(raw_mask):
            row = tape.candidates[int(original_index)]
            confirmation = candidate_tape._confirmation(tape, row, params)
            if confirmation is None:
                continue
            trade = candidate_tape._trade_for_candidate(tape, row, confirmation, params)
            if trade is None:
                continue
            rows.append({
                **trade, "date": day, "trading_session": row.get("trading_session"),
                "family_id": str(row.get("family_id", row.get("level", "UNKNOWN"))),
                "score_values": tuple(float(row.get(name, 0.0) or 0.0) for name in candidate_tape.SCORE_NAMES),
                "confirmation_index": int(confirmation[0]),
                "confirmation_timestamp_ns": int(confirmation[1]["timestamp_ns"]),
                "interaction_end_ns": int(row.get("interaction_end_ns") or 0),
                "candidate_order": int(original_index),
            })
    return ExactOpportunityPool(tuple(rows), tuple(family_ids))


def _exact_config_trades(pool: ExactOpportunityPool, weights: np.ndarray, min_q: float) -> list[dict[str, Any]]:
    eligible_by_date: dict[str, list[dict[str, Any]]] = {}
    for row in pool.rows:
        scores = row["score_values"]
        quality = sum(scores[index] * float(weights[index]) for index in range(5)) - scores[5] * 0.25
        if quality < float(min_q):
            continue
        eligible_by_date.setdefault(str(row["date"]), []).append(row)
    output: list[dict[str, Any]] = []
    for day in EXPECTED_TRAIN_DATES:
        eligible = eligible_by_date.get(day, [])
        eligible.sort(key=lambda row: (int(row["entry_timestamp_ns"]), int(row["confirmation_index"]),
                                       int(row["confirmation_timestamp_ns"]), int(row["interaction_end_ns"]),
                                       int(row["candidate_order"])))
        for row in eligible:
            if output and str(output[-1]["date"]) == day and int(row["entry_timestamp_ns"]) <= int(output[-1]["exit_timestamp_ns"]):
                continue
            output.append(dict(row))
    return output


def _family_report(trades: Sequence[Mapping[str, Any]], family_ids: Sequence[str], dates: Sequence[str]) -> dict[str, Any]:
    total_abs = sum(abs(_finite(row.get("r_multiple"))) for row in trades)
    report = {}
    for family in family_ids:
        rows = [row for row in trades if row.get("family_id") == family]
        date_r = {day: sum(_finite(row.get("r_multiple")) for row in rows if row.get("date") == day) for day in dates}
        values = [_finite(row.get("r_multiple")) for row in rows]
        raw_pf, capped_pf = _profit_factor(values)
        report[family] = {
            "trades": len(rows), "net_r": sum(values), "profit_factor": raw_pf,
            "profit_factor_capped": capped_pf, "active_dates": sum(bool(value) for value in date_r.values()),
            "profitable_dates": sum(value > 0 for value in date_r.values()),
            "contribution_share": abs(sum(values)) / total_abs if total_abs else 0.0,
        }
    return report


def _exact_metrics(trades: Sequence[Mapping[str, Any]], family_ids: Sequence[str]) -> dict[str, Any]:
    return _snapshot_metrics(trades=trades, dates=EXPECTED_TRAIN_DATES,
                             sessions=("ASIA", "EUROPE", "NY"))


def _exact_population_result(pool: ExactOpportunityPool, arrays: Mapping[str, np.ndarray], indices: Sequence[int],
                             fast_scores: np.ndarray, scaling: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for index in indices:
        trades = _exact_config_trades(pool, arrays["weights"][index], float(arrays["min_q"][index]))
        metrics = _exact_metrics(trades, pool.family_ids)
        objective_metrics = {
            "median_date_r": metrics["median_date_r"], "lower_quartile_date_r": metrics["lower_quartile_date_r"],
            "median_session_date_r": metrics["median_session_date_r"], "profit_factor": metrics["profit_factor_capped"],
            "profitable_date_ratio": metrics["profitable_date_ratio"], "max_drawdown_r": metrics["max_drawdown_r"],
            "date_concentration": metrics["date_concentration_top5"],
            "family_concentration": metrics["family_concentration_top3"], "downside_tail": metrics["p10_date_r"],
            "session_diversity": metrics["active_sessions"] / 3.0,
            "trade_activity": math.log1p(max(0, metrics["total_trades"])),
        }
        exact_objective = robust_objective(objective_metrics, scaling)
        base = _config_row(index, arrays, fast_scores)
        base.update({
            "fast_trades": base.pop("trades"), "fast_net_r": base.pop("net_r"),
            "fast_profit_factor": base.pop("profit_factor"), "fast_max_drawdown_r": base.pop("max_drawdown_r"),
            "fast_robust_score": base.pop("robust_score"),
            "exact_trades": metrics["total_trades"], "exact_net_r": metrics["net_r"],
            "exact_profit_factor": metrics["profit_factor"], "exact_max_drawdown_r": metrics["max_drawdown_r"],
            "exact_robust_score": exact_objective["robust_score"],
            "exact_objective_components": exact_objective,
            "exact_per_date": metrics["per_date"],
            "exact_family_metrics": _family_report(trades, pool.family_ids, EXPECTED_TRAIN_DATES),
        })
        results.append(base)
    return results


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)
    br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(ar, br)[0, 1])


def reanalyze_existing_stage1a(*, screen_path: Path = DEFAULT_OUTPUT_ROOT / "train-stage1a-screen.npz",
                               manifest_path: Path = DEFAULT_TAPE_ROOT / TAPE_MANIFEST_NAME,
                               output_root: Path = DEFAULT_OUTPUT_ROOT, seed: int = STAGE1A_SEED) -> dict[str, Any]:
    """Re-rank the stored million-row result and exact-evaluate its shortlist."""
    arrays, valid = _population_from_npz(screen_path)
    bundle = load_train_tapes(manifest_path=manifest_path)
    scaling = build_objective_scaling({"metrics": arrays, "technically_valid": valid}, valid_mask=valid)
    robust_scores = _population_objective_scores(arrays, scaling)
    distributions = {}
    distribution_specs = {
        "net_r": "maximize", "total_trades": "maximize", "profit_factor": "maximize",
        "max_drawdown_r": "maximize", "active_dates": "maximize", "profitable_date_ratio": "maximize",
        "date_concentration": "minimize", "family_concentration": "minimize", "robust_score": "maximize",
        "median_date_r": "maximize", "lower_quartile_date_r": "maximize", "downside_tail": "maximize",
        "median_session_date_r": "maximize", "active_sessions": "maximize",
    }
    for name, direction in distribution_specs.items():
        values = robust_scores if name == "robust_score" else arrays[name]
        distributions[name] = _distribution(values[valid], direction=direction)
    top = {}
    for label, values, descending in (
        ("net_r", arrays["net_r"], True), ("profit_factor", arrays["profit_factor"], True),
        ("lowest_drawdown_among_profitable", arrays["max_drawdown_r"], True),
        ("robust_score", robust_scores, True),
    ):
        ordered = _ranked_indices(values, descending=descending)
        if label == "lowest_drawdown_among_profitable":
            ordered = np.asarray([i for i in ordered if valid[i] and arrays["net_r"][i] > 0])
        top[label] = [_config_row(int(i), arrays, robust_scores) for i in ordered if valid[i]][:25]
    for label, low, high in (("net_r_le_100", None, 100), ("net_r_101_250", 101, 250),
                             ("net_r_251_500", 251, 500), ("net_r_gt_500", 501, None)):
        mask = valid & (arrays["total_trades"] >= (0 if low is None else low)) & (arrays["total_trades"] <= (np.inf if high is None else high))
        if low is None:
            mask = valid & (arrays["total_trades"] <= high)
        top[label] = [_config_row(int(i), arrays, robust_scores) for i in _ranked_indices(arrays["net_r"]) if mask[i]][:25]
    regions = _parameter_regions(arrays, robust_scores, valid)
    selected, categories = _build_shortlist(arrays, robust_scores, valid, seed=seed)
    shortlist_rows = []
    for label, indexes in categories.items():
        for index in indexes:
            row = _config_row(index, arrays, robust_scores)
            row["shortlist_category"] = label
            shortlist_rows.append(row)
    _json_write(output_root / "train-objective-scaling.json", scaling)
    _json_write(output_root / "train-optimization-constraints.json", derive_constraints())
    _json_write(output_root / "train-stage1a-screen.json", {
        "status": "COMPLETE", "version": "MAC2025_STAGE1A_V1",
        "configurations_evaluated": int(len(valid)),
        "technically_valid_configurations": int(valid.sum()),
        "performance_constraints": "NONE",
        "legacy_performance_feasible_field_ignored": True,
        "source_path": str(screen_path), "seed": STAGE1A_SEED,
        "exact_event_path_replay_required_before_acceptance": True,
    })
    _json_write(output_root / "train-stage1a-unconstrained-analysis.json", {
        "status": "COMPLETE", "technically_valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()), "performance_constraints": "NONE",
        "distributions": distributions, "top_configurations": top,
        "parameter_regions": regions, "seed": seed,
        "source_screen": str(screen_path), "validation_accessed": False, "oos_accessed": False,
    })
    _json_write(output_root / "train-exact-shortlist.json", {
        "status": "SELECTED", "count": len(shortlist_rows), "category_target_counts": {
            key: len(value) for key, value in categories.items()
        }, "configurations": shortlist_rows,
    })
    pool = _build_exact_opportunity_pool(bundle)
    exact_results = _exact_population_result(pool, arrays, selected, robust_scores, scaling)
    fast_by_index = {row["index"]: row for row in shortlist_rows}
    exact_robust = [row["exact_robust_score"] for row in exact_results]
    fast_robust = [fast_by_index[row["index"]]["robust_score"] for row in exact_results]
    exact_net = [row["exact_net_r"] for row in exact_results]
    fast_net = [fast_by_index[row["index"]]["net_r"] for row in exact_results]
    direct_checks = []
    for index in selected[:3]:
        direct_trades: list[dict[str, Any]] = []
        for day, tape in zip(bundle.dates, bundle.tapes):
            direct = candidate_tape.evaluate_candidate_tape(
                tape, parameters={"weights": dict(zip(candidate_tape.WEIGHT_NAMES, arrays["weights"][index])),
                                  "min_quality_score": float(arrays["min_q"][index])},
            )
            direct_trades.extend(direct["trades"])
        optimized_trades = _exact_config_trades(pool, arrays["weights"][index], float(arrays["min_q"][index]))
        comparison = candidate_tape.compare_trade_ledgers(direct_trades, optimized_trades)
        direct_checks.append({"index": int(index), "pass": bool(comparison["pass"]),
                              "expected_count": comparison["expected_count"],
                              "actual_count": comparison["actual_count"]})
    correlation = {
        "net_r": _rank_correlation(fast_net, exact_net),
        "robust_score": _rank_correlation(fast_robust, exact_robust),
        "trades": _rank_correlation([fast_by_index[row["index"]]["trades"] for row in exact_results], [row["exact_trades"] for row in exact_results]),
        "profit_factor": _rank_correlation([fast_by_index[row["index"]]["profit_factor"] for row in exact_results], [row["exact_profit_factor"] if row["exact_profit_factor"] is not None else 0.0 for row in exact_results]),
        "max_drawdown_r": _rank_correlation([fast_by_index[row["index"]]["max_drawdown_r"] for row in exact_results], [row["exact_max_drawdown_r"] for row in exact_results]),
    }
    robust_correlation = correlation["robust_score"]
    classification = ("STRONG" if robust_correlation >= 0.80 else
                      "ACCEPTABLE" if robust_correlation >= 0.60 else
                      "WEAK" if robust_correlation >= 0.30 else "INVALID")
    exact_report = {
        "status": "COMPLETE", "count": len(exact_results), "family_count": len(pool.family_ids),
        "family_ids": list(pool.family_ids), "results": exact_results,
        "fast_exact_rank_correlation": correlation, "fast_kernel_exact_proxy": classification,
        "direct_event_path_checks": direct_checks,
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _json_write(output_root / "train-exact-shortlist-results.json", exact_report)
    return {
        "technically_valid_count": int(valid.sum()), "best_net_r_config": top["net_r"][0],
        "best_robust_config": top["robust_score"][0],
        "best_low_frequency_config": top["net_r_le_100"][0] if top["net_r_le_100"] else None,
        "best_medium_frequency_config": top["net_r_101_250"][0] if top["net_r_101_250"] else None,
        "exact_shortlist_count": len(exact_results), "fast_kernel_exact_proxy": classification,
        "fast_exact_rank_correlation": correlation, "robust_parameter_regions": regions,
        "family_report_ready": len(pool.family_ids) == 61 and all(len(row["exact_family_metrics"]) == 61 for row in exact_results),
        "ready_to_run_optuna": bool(all(row["pass"] for row in direct_checks)),
    }


def prepare_train_optimization(*, manifest_path: Path = DEFAULT_TAPE_ROOT / TAPE_MANIFEST_NAME,
                                output_root: Path = DEFAULT_OUTPUT_ROOT, target: int = STAGE1A_TARGET,
                                seed: int = STAGE1A_SEED, chunk_size: int = STAGE1A_CHUNK_SIZE,
                                repo_root: Path = Path(".")) -> dict[str, Any]:
    """Generate all TRAIN-only preparation artifacts; never accesses DBN."""
    bundle = load_train_tapes(manifest_path=manifest_path, repo_root=repo_root)
    output_root = _repo_relative(output_root, repo_root.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    loader_report = {
        "status": "PASS", "manifest": str(bundle.manifest_path), "train_date_count": len(bundle.dates),
        "train_dates": list(bundle.dates), "validation_accessed": False, "oos_accessed": False,
        "dbn_accessed_during_optimization": False,
    }
    _json_write(output_root / "train-loader-verification.json", loader_report)
    stage0 = stage0_baseline(bundle)
    _json_write(output_root / "train-stage0-baseline.json", stage0)
    constraints = derive_constraints(stage0)
    _json_write(output_root / "train-optimization-constraints.json", constraints)
    screen = run_stage1a(bundle=bundle, constraints=constraints, target=target, seed=seed,
                         chunk_size=chunk_size, output_path=output_root / "train-stage1a-screen.npz")
    screen_report = {key: value for key, value in screen.items() if key not in {"weights", "min_q", "feasible", "metrics", "feasible_indices"}}
    _json_write(output_root / "train-stage1a-screen.json", screen_report)
    scaling = build_objective_scaling(screen)
    _json_write(output_root / "train-objective-scaling.json", scaling)
    definition = {
        "objective_version": OBJECTIVE_VERSION, "coefficients": OBJECTIVE_DEFINITION,
        "profit_factor_cap": PROFIT_FACTOR_CAP, "robust_scale_floor": ROBUST_SCALE_FLOOR,
        "screen_semantics": screen["screen_semantics"],
        "exact_event_path_replay_required_before_acceptance": True,
    }
    _json_write(output_root / "train-objective-definition.json", definition)
    return {"loader": loader_report, "stage0": stage0, "constraints": constraints,
            "stage1a": screen_report, "objective_scaling": scaling,
            "objective_version": OBJECTIVE_VERSION, "output_root": str(output_root)}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TAPE_ROOT / TAPE_MANIFEST_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target", type=int, default=STAGE1A_TARGET)
    parser.add_argument("--seed", type=int, default=STAGE1A_SEED)
    parser.add_argument("--chunk-size", type=int, default=STAGE1A_CHUNK_SIZE)
    parser.add_argument("--reanalyze-existing", action="store_true",
                        help="reuse the stored Stage-1A NPZ; do not rerun the million-row screen")
    args = parser.parse_args(argv)
    if args.reanalyze_existing:
        result = reanalyze_existing_stage1a(
            screen_path=args.output_root / "train-stage1a-screen.npz",
            manifest_path=args.manifest, output_root=args.output_root, seed=args.seed,
        )
        print(json.dumps({
            "STAGE1A_TECHNICALLY_VALID_COUNT": result["technically_valid_count"],
            "EXACT_SHORTLIST_COUNT": result["exact_shortlist_count"],
            "FAST_KERNEL_EXACT_PROXY": result["fast_kernel_exact_proxy"],
            "FAST_EXACT_RANK_CORRELATION": result["fast_exact_rank_correlation"],
            "FAMILY_REPORT_READY": result["family_report_ready"],
            "READY_TO_RUN_OPTUNA": result["ready_to_run_optuna"],
            "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False,
        }, indent=2, sort_keys=True))
        return 0
    result = prepare_train_optimization(manifest_path=args.manifest, output_root=args.output_root,
                                        target=args.target, seed=args.seed, chunk_size=args.chunk_size)
    print(json.dumps({"READY_TO_RUN_OPTUNA": True,
                      "TRAIN_ONLY_LOADER": result["loader"]["status"],
                      "TRAIN_DATE_COUNT": result["loader"]["train_date_count"],
                      "STAGE1A_CONFIGURATIONS_EVALUATED": result["stage1a"]["configurations_evaluated"],
                      "STAGE1A_FEASIBLE": result["stage1a"]["feasible_configurations"],
                      "OBJECTIVE_VERSION": result["objective_version"],
                      "DBN_ACCESSED_DURING_OPTIMIZATION": False,
                      "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False,
                      "OUTPUT_ROOT": result["output_root"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
