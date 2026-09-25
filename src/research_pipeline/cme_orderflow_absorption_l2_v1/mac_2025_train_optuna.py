"""TRAIN-only Optuna Stage-2A search over the sealed MAC 2025 candidate tapes.

This module is deliberately downstream of the candidate-tape proof.  It never
opens DBN or market-data files and it never loads a validation/OOS artifact.
All Class-A parameters are evaluated from stored primitive interaction fields;
confirmation, entry, and exit semantics remain frozen to the candidate-tape
contract.  The SQLite study is the resumable source of truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_train_optimization as preparation
from .model import L2Config


STAGE2A_TRIALS = 50_000
STAGE2A_SEED = 20250922
STAGE2A_OBJECTIVE_VERSION = "MAC2025_TRAIN_ROBUST_OBJECTIVE_V1"
STAGE2A_STUDY_NAME = "MAC2025_TRAIN_STAGE2A_CLASS_A_TPE_JOURNAL_V1"
STAGE2A_DB_NAME = "train-stage2a-optuna-final-journal.log"
STAGE2A_SEARCH_SPACE_NAME = "train-stage2a-search-space.json"
STAGE2A_TRIALS_NAME = "train-stage2a-trials.jsonl"
STAGE2A_SUMMARY_NAME = "train-stage2a-summary.json"
STAGE2A_EXACT_NAME = "train-stage2a-exact-shortlist.json"
STAGE2A_POOL_NPZ_NAME = "train-stage2a-opportunity-pool.npz"
STAGE2A_POOL_JSON_NAME = "train-stage2a-opportunity-pool.json"

THRESHOLD_NAMES = (
    "min_relevant_aggressive_volume", "min_relevant_execution_count",
    "min_consume_restore_cycles", "max_through_level_progress_ticks",
    "min_rejection_ticks",
)
PENALTY_NAMES = (
    "false_refill_penalty_weight", "unexecuted_add_penalty_component_weight",
    "rapid_cancel_penalty_component_weight", "adverse_progress_penalty_component_weight",
)
SATURATION_NAMES = (
    "aggressive_volume_saturation", "execution_count_saturation",
    "restore_cycle_saturation", "restoration_ratio_saturation",
    "rejection_saturation_ticks", "persistence_depth_saturation",
    "restoration_latency_saturation_ms", "multi_level_ofi_saturation",
)
ALL_A_NAMES = tuple(candidate_tape.WEIGHT_NAMES) + ("min_quality_score",) + THRESHOLD_NAMES + PENALTY_NAMES + SATURATION_NAMES


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _search_space() -> dict[str, Any]:
    return {
        "version": "MAC2025_STAGE2A_SEARCH_SPACE_V1",
        "objective_version": STAGE2A_OBJECTIVE_VERSION,
        "scope": "TRAIN_ONLY",
        "trial_count": STAGE2A_TRIALS,
        "seed": STAGE2A_SEED,
        "weights": {f"raw_{name}": {"distribution": "log_float", "low": 0.01, "high": 5.0}
                    for name in candidate_tape.WEIGHT_NAMES},
        "min_quality_score": {"distribution": "float", "low": 0.50, "high": 0.80},
        "thresholds": {
            "min_relevant_aggressive_volume": {"distribution": "int", "low": 0, "high": 300},
            "min_relevant_execution_count": {"distribution": "int", "low": 0, "high": 12},
            "min_consume_restore_cycles": {"distribution": "int", "low": 0, "high": 6},
            "max_through_level_progress_ticks": {"distribution": "float", "low": 0.25, "high": 12.0},
            "min_rejection_ticks": {"distribution": "float", "low": 0.0, "high": 4.0},
        },
        "penalty_weights": {name: {"distribution": "float", "low": 0.0, "high": 1.0} for name in PENALTY_NAMES},
        "saturations": {
            "aggressive_volume_saturation": {"distribution": "log_float", "low": 25.0, "high": 1000.0},
            "execution_count_saturation": {"distribution": "log_float", "low": 1.0, "high": 32.0},
            "restore_cycle_saturation": {"distribution": "log_float", "low": 0.5, "high": 10.0},
            "restoration_ratio_saturation": {"distribution": "log_float", "low": 0.25, "high": 4.0},
            "rejection_saturation_ticks": {"distribution": "log_float", "low": 0.5, "high": 12.0},
            "persistence_depth_saturation": {"distribution": "log_float", "low": 10.0, "high": 500.0},
            "restoration_latency_saturation_ms": {"distribution": "log_float", "low": 100.0, "high": 5000.0},
            "multi_level_ofi_saturation": {"distribution": "log_float", "low": 10.0, "high": 500.0},
        },
    }


@dataclass(frozen=True)
class AllAOpportunityPool:
    features: dict[str, np.ndarray]
    direction_sign: np.ndarray
    has_trade: np.ndarray
    trade_r: np.ndarray
    date_index: np.ndarray
    session_index: np.ndarray
    family_index: np.ndarray
    family_ids: tuple[str, ...]
    trade_rows: tuple[dict[str, Any] | None, ...]

    @property
    def count(self) -> int:
        return len(self.trade_r)


def _row_value(row: Mapping[str, Any], name: str) -> float:
    value = _finite(row.get(name, 0.0), default=float("nan"))
    if not math.isfinite(value):
        raise preparation.TrainOptimizationError(f"non-finite Class-A candidate feature: {name}")
    return value


def build_all_a_pool(bundle: preparation.TrainTapeBundle) -> AllAOpportunityPool:
    """Build all-candidate fixed-B opportunities once, including rejects."""
    family_ids = tuple(sorted({str(row.get("family_id", row.get("level", "UNKNOWN")))
                               for tape in bundle.tapes for row in tape.candidates}))
    family_index = {name: index for index, name in enumerate(family_ids)}
    feature_names = (
        "directional_aggressive_volume", "relevant_execution_count", "consume_restore_cycles",
        "maximum_through_level_progress_ticks", "interaction_rejection_ticks",
        "aggressive_volume_imbalance", "executed_to_initial_displayed_ratio",
        "restoration_to_consumption_ratio", "restoration_supported_by_execution_ratio",
        "mean_restoration_latency_ms", "defended_price_present_fraction",
        "defended_depth_time_weighted_mean", "depth_imbalance_5", "multi_level_ofi",
        "unexecuted_add_volume", "rapid_cancel_ratio",
    )
    values = {name: [] for name in feature_names}
    directions: list[float] = []
    dates: list[int] = []
    sessions: list[int] = []
    families: list[int] = []
    trade_values: list[float] = []
    has_trade: list[bool] = []
    trade_rows: list[dict[str, Any] | None] = []
    fixed_parameters = candidate_tape._default_parameters(L2Config())
    session_codes = {"ASIA": 0, "EUROPE": 1, "NY": 2}
    for date_index, (day, tape) in enumerate(zip(bundle.dates, bundle.tapes)):
        for candidate_order, row in enumerate(tape.candidates):
            for name in feature_names:
                values[name].append(_row_value(row, name))
            directions.append(1.0 if row.get("direction") == "BUYER_ABSORPTION" else -1.0)
            dates.append(date_index)
            sessions.append(session_codes.get(str(row.get("trading_session")), -1))
            family_name = str(row.get("family_id", row.get("level", "UNKNOWN")))
            families.append(family_index[family_name])
            confirmation = candidate_tape._confirmation(tape, row, fixed_parameters)
            trade = None if confirmation is None else candidate_tape._trade_for_candidate(
                tape, row, confirmation, fixed_parameters
            )
            if trade is None:
                trade_values.append(0.0)
                has_trade.append(False)
                trade_rows.append(None)
            else:
                enriched = {**trade, "date": day, "trading_session": row.get("trading_session"),
                            "family_id": family_name, "candidate_order": candidate_order,
                            "confirmation_index": int(confirmation[0]),
                            "confirmation_timestamp_ns": int(confirmation[1]["timestamp_ns"]),
                            "interaction_end_ns": int(row.get("interaction_end_ns") or 0)}
                trade_values.append(_finite(trade.get("r_multiple")))
                has_trade.append(True)
                trade_rows.append(enriched)
    return AllAOpportunityPool(
        features={name: np.asarray(values[name], dtype=np.float64) for name in feature_names},
        direction_sign=np.asarray(directions, dtype=np.float64), has_trade=np.asarray(has_trade, dtype=bool),
        trade_r=np.asarray(trade_values, dtype=np.float64), date_index=np.asarray(dates, dtype=np.int16),
        session_index=np.asarray(sessions, dtype=np.int8), family_index=np.asarray(families, dtype=np.int16),
        family_ids=family_ids, trade_rows=tuple(trade_rows),
    )


def save_all_a_pool(pool: AllAOpportunityPool, *, output_root: Path, train_dates: Sequence[str]) -> None:
    """Persist the expensive fixed-B preparation for clean study resume."""
    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / STAGE2A_POOL_NPZ_NAME
    fd, name = tempfile.mkstemp(prefix=f".{npz_path.name}.", suffix=".npz", dir=output_root)
    os.close(fd)
    try:
        np.savez_compressed(name,
                            **pool.features, direction_sign=pool.direction_sign,
                            has_trade=pool.has_trade, trade_r=pool.trade_r,
                            date_index=pool.date_index, session_index=pool.session_index,
                            family_index=pool.family_index)
        os.replace(name, npz_path)
    finally:
        Path(name).unlink(missing_ok=True)
    _atomic_json(output_root / STAGE2A_POOL_JSON_NAME, {
        "version": "MAC2025_STAGE2A_OPPORTUNITY_POOL_V1", "scope": "TRAIN_ONLY",
        "train_dates": list(train_dates), "candidate_count": pool.count,
        "family_ids": list(pool.family_ids),
        "trade_rows": list(pool.trade_rows),
    })


def load_all_a_pool(*, output_root: Path, train_dates: Sequence[str]) -> AllAOpportunityPool | None:
    npz_path = output_root / STAGE2A_POOL_NPZ_NAME
    json_path = output_root / STAGE2A_POOL_JSON_NAME
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("version") != "MAC2025_STAGE2A_OPPORTUNITY_POOL_V1" or \
                metadata.get("train_dates") != list(train_dates):
            return None
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        trade_rows = tuple(metadata.get("trade_rows", ()))
        count = int(metadata.get("candidate_count", -1))
        if count <= 0 or len(trade_rows) != count or len(arrays["trade_r"]) != count:
            return None
        feature_names = (
            "directional_aggressive_volume", "relevant_execution_count", "consume_restore_cycles",
            "maximum_through_level_progress_ticks", "interaction_rejection_ticks",
            "aggressive_volume_imbalance", "executed_to_initial_displayed_ratio",
            "restoration_to_consumption_ratio", "restoration_supported_by_execution_ratio",
            "mean_restoration_latency_ms", "defended_price_present_fraction",
            "defended_depth_time_weighted_mean", "depth_imbalance_5", "multi_level_ofi",
            "unexecuted_add_volume", "rapid_cancel_ratio",
        )
        if any(name not in arrays or len(arrays[name]) != count for name in feature_names):
            return None
        return AllAOpportunityPool(
            features={name: arrays[name] for name in feature_names}, direction_sign=arrays["direction_sign"],
            has_trade=arrays["has_trade"].astype(bool), trade_r=arrays["trade_r"],
            date_index=arrays["date_index"].astype(np.int16), session_index=arrays["session_index"].astype(np.int8),
            family_index=arrays["family_index"].astype(np.int16),
            family_ids=tuple(metadata.get("family_ids", ())), trade_rows=trade_rows,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _quality(pool: AllAOpportunityPool, params: Mapping[str, Any]) -> np.ndarray:
    f = pool.features
    directional = f["directional_aggressive_volume"]
    aggression = (
        np.minimum(1.0, directional / float(params["aggressive_volume_saturation"]))
        + np.minimum(1.0, f["relevant_execution_count"] / float(params["execution_count_saturation"]))
        + _clamp01((f["aggressive_volume_imbalance"] + 1.0) / 2.0)
        + np.minimum(1.0, f["executed_to_initial_displayed_ratio"])
    ) / 4.0
    restoration = (
        np.minimum(1.0, f["consume_restore_cycles"] / float(params["restore_cycle_saturation"]))
        + np.minimum(1.0, f["restoration_to_consumption_ratio"] / float(params["restoration_ratio_saturation"]))
        + _clamp01(f["restoration_supported_by_execution_ratio"])
        + (1.0 - np.minimum(1.0, f["mean_restoration_latency_ms"] /
                             float(params["restoration_latency_saturation_ms"])))
    ) / 4.0
    maximum = f["maximum_through_level_progress_ticks"]
    rejection = f["interaction_rejection_ticks"]
    resistance = (
        1.0 - np.minimum(1.0, maximum / float(params["max_through_level_progress_ticks"]))
        + np.minimum(1.0, rejection / float(params["rejection_saturation_ticks"]))
    ) / 2.0
    persistence = (
        _clamp01(f["defended_price_present_fraction"])
        + np.minimum(1.0, f["defended_depth_time_weighted_mean"] /
                     float(params["persistence_depth_saturation"]))
    ) / 2.0
    directional_book = pool.direction_sign * f["depth_imbalance_5"]
    directional_ofi = pool.direction_sign * f["multi_level_ofi"]
    multi = (
        _clamp01((directional_book + 1.0) / 2.0)
        + np.minimum(1.0, np.maximum(0.0, directional_ofi) /
                     float(params["multi_level_ofi_saturation"]))
    ) / 2.0
    penalty = np.clip(
        float(params["unexecuted_add_penalty_component_weight"]) *
        f["unexecuted_add_volume"] / (directional + 1.0)
        + float(params["rapid_cancel_penalty_component_weight"]) * f["rapid_cancel_ratio"]
        + float(params["adverse_progress_penalty_component_weight"]) *
        np.minimum(1.0, maximum / float(params["max_through_level_progress_ticks"])),
        0.0, 1.0,
    )
    weights = np.asarray([float(params[name]) for name in candidate_tape.WEIGHT_NAMES], dtype=np.float64)
    return np.clip(
        aggression * weights[0] + restoration * weights[1] + resistance * weights[2]
        + persistence * weights[3] + multi * weights[4]
        - penalty * float(params["false_refill_penalty_weight"]), 0.0, 1.0
    )


def qualified_mask(pool: AllAOpportunityPool, params: Mapping[str, Any]) -> np.ndarray:
    f = pool.features
    raw = (
        (f["directional_aggressive_volume"] >= float(params["min_relevant_aggressive_volume"]))
        & (f["relevant_execution_count"] >= float(params["min_relevant_execution_count"]))
        & (f["consume_restore_cycles"] >= float(params["min_consume_restore_cycles"]))
        & ~((f["maximum_through_level_progress_ticks"] > float(params["max_through_level_progress_ticks"]))
            & (f["interaction_rejection_ticks"] < float(params["min_rejection_ticks"])))
    )
    return raw & (_quality(pool, params) >= float(params["min_quality_score"]))


def _max_drawdown(values: np.ndarray) -> float:
    equity = np.cumsum(values, dtype=np.float64)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    return float(np.min(equity - peak)) if len(equity) else 0.0


def _concentration(values: np.ndarray, top_n: int) -> float:
    magnitudes = np.sort(np.abs(values))[::-1]
    total = float(magnitudes.sum())
    return float(magnitudes[:top_n].sum() / total) if total else 0.0


def fast_metrics(pool: AllAOpportunityPool, qualified: np.ndarray, date_count: int) -> dict[str, Any]:
    selected = qualified & pool.has_trade
    r = pool.trade_r[selected]
    date_r = np.bincount(pool.date_index[selected], weights=r, minlength=date_count).astype(np.float64)
    session_mask = selected & (pool.session_index >= 0)
    session_slots = pool.date_index[session_mask] * 3 + pool.session_index[session_mask]
    session_r = np.bincount(session_slots, weights=pool.trade_r[session_mask], minlength=date_count * 3).astype(np.float64)
    family_r = np.bincount(pool.family_index[selected], weights=r, minlength=len(pool.family_ids)).astype(np.float64)
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    profit_factor = preparation.PROFIT_FACTOR_CAP if gross_loss == 0 and gross_profit > 0 else (
        gross_profit / gross_loss if gross_loss else 0.0
    )
    profit_factor = min(preparation.PROFIT_FACTOR_CAP, profit_factor)
    return {
        "total_trades": float(selected.sum()), "net_r": float(r.sum()),
        "profit_factor": float(profit_factor), "max_drawdown_r": _max_drawdown(date_r),
        "active_dates": float((np.abs(date_r) > 0).sum()),
        "profitable_date_ratio": float((date_r > 0).sum() / date_count),
        "median_date_r": float(np.quantile(date_r, 0.50)),
        "lower_quartile_date_r": float(np.quantile(date_r, 0.25)),
        "downside_tail": float(np.quantile(date_r, 0.10)),
        "median_session_date_r": float(np.quantile(session_r, 0.50)),
        "date_concentration": _concentration(date_r, 5),
        "family_concentration": _concentration(family_r, 3),
        "active_sessions": float((np.abs(session_r) > 0).sum()),
        "date_r": date_r.tolist(), "session_date_r": session_r.tolist(),
    }


def _objective_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "median_date_r": float(metrics["median_date_r"]),
        "lower_quartile_date_r": float(metrics["lower_quartile_date_r"]),
        "median_session_date_r": float(metrics["median_session_date_r"]),
        "profit_factor": float(metrics.get("profit_factor_capped", metrics.get("profit_factor") or 0.0)),
        "profitable_date_ratio": float(metrics["profitable_date_ratio"]),
        "max_drawdown_r": float(metrics["max_drawdown_r"]),
        "date_concentration": float(metrics.get("date_concentration", metrics.get("date_concentration_top5", 0.0))),
        "family_concentration": float(metrics.get("family_concentration", metrics.get("family_concentration_top3", 0.0))),
        "downside_tail": float(metrics.get("downside_tail", metrics.get("p10_date_r", 0.0))),
        "session_diversity": float(metrics["active_sessions"]) / 3.0,
        "trade_activity": math.log1p(max(0.0, float(metrics["total_trades"]))),
    }


def _trial_params(trial: optuna.Trial) -> dict[str, Any]:
    raw = [trial.suggest_float(f"raw_{name}", 0.01, 5.0, log=True) for name in candidate_tape.WEIGHT_NAMES]
    total = sum(raw)
    weights = {name: value / total for name, value in zip(candidate_tape.WEIGHT_NAMES, raw)}
    params: dict[str, Any] = dict(weights)
    params["min_quality_score"] = trial.suggest_float("min_quality_score", 0.50, 0.80)
    params["min_relevant_aggressive_volume"] = trial.suggest_int("min_relevant_aggressive_volume", 0, 300)
    params["min_relevant_execution_count"] = trial.suggest_int("min_relevant_execution_count", 0, 12)
    params["min_consume_restore_cycles"] = trial.suggest_int("min_consume_restore_cycles", 0, 6)
    params["max_through_level_progress_ticks"] = trial.suggest_float("max_through_level_progress_ticks", 0.25, 12.0)
    params["min_rejection_ticks"] = trial.suggest_float("min_rejection_ticks", 0.0, 4.0)
    for name in PENALTY_NAMES:
        params[name] = trial.suggest_float(name, 0.0, 1.0)
    bounds = {
        "aggressive_volume_saturation": (25.0, 1000.0), "execution_count_saturation": (1.0, 32.0),
        "restore_cycle_saturation": (0.5, 10.0), "restoration_ratio_saturation": (0.25, 4.0),
        "rejection_saturation_ticks": (0.5, 12.0), "persistence_depth_saturation": (10.0, 500.0),
        "restoration_latency_saturation_ms": (100.0, 5000.0), "multi_level_ofi_saturation": (10.0, 500.0),
    }
    for name, (low, high) in bounds.items():
        params[name] = trial.suggest_float(name, low, high, log=True)
    return params


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = dict(trial.user_attrs)
    return {
        "number": int(trial.number), "state": trial.state.name,
        "params": dict(trial.params), "normalized_weights": attrs.get("normalized_weights", {}),
        "parameters": attrs.get("parameters", {}), "objective": float(trial.value) if trial.value is not None else None,
        "objective_components": attrs.get("objective_components", {}),
        "metrics": attrs.get("metrics", {}), "technical_valid": bool(attrs.get("technical_valid", False)),
    }


def _write_trial_jsonl(path: Path, trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "w", encoding="utf-8") as handle:
            for trial in sorted(trials, key=lambda row: row.number):
                handle.write(json.dumps(_trial_record(trial), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _load_scaling(output_root: Path) -> dict[str, Any]:
    path = output_root / "train-objective-scaling.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    raise preparation.TrainOptimizationError(f"missing frozen objective scaling: {path}")


def _exact_trades(pool: AllAOpportunityPool, qualified: np.ndarray, dates: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for date_index, day in enumerate(dates):
        indices = np.flatnonzero(qualified & pool.has_trade & (pool.date_index == date_index))
        indices = sorted(indices.tolist(), key=lambda index: (
            int(pool.trade_rows[index]["entry_timestamp_ns"]), int(pool.trade_rows[index]["confirmation_index"]),
            int(pool.trade_rows[index]["confirmation_timestamp_ns"]), int(pool.trade_rows[index]["interaction_end_ns"]),
            int(pool.trade_rows[index]["candidate_order"]),
        ))
        last_exit = None
        for index in indices:
            row = dict(pool.trade_rows[index])  # type: ignore[arg-type]
            if last_exit is not None and int(row["entry_timestamp_ns"]) <= last_exit:
                continue
            result.append(row)
            last_exit = int(row["exit_timestamp_ns"])
    return result


def _exact_family_report(trades: Sequence[Mapping[str, Any]], family_ids: Sequence[str], dates: Sequence[str]) -> dict[str, Any]:
    total_abs = sum(abs(_finite(row.get("r_multiple"))) for row in trades)
    report: dict[str, Any] = {}
    for family in family_ids:
        rows = [row for row in trades if row.get("family_id") == family]
        per_date = [sum(_finite(row.get("r_multiple")) for row in rows if row.get("date") == day) for day in dates]
        values = [_finite(row.get("r_multiple")) for row in rows]
        positive = sum(value for value in values if value > 0)
        negative = -sum(value for value in values if value < 0)
        report[family] = {
            "trades": len(rows), "net_r": sum(values),
            "profit_factor": positive / negative if negative else (preparation.PROFIT_FACTOR_CAP if positive else 0.0),
            "active_dates": sum(value != 0 for value in per_date),
            "profitable_dates": sum(value > 0 for value in per_date),
            "contribution_share": abs(sum(values)) / total_abs if total_abs else 0.0,
        }
    return report


def _exact_result(pool: AllAOpportunityPool, params: Mapping[str, Any], scaling: Mapping[str, Any], dates: Sequence[str]) -> dict[str, Any]:
    qualified = qualified_mask(pool, params)
    trades = _exact_trades(pool, qualified, dates)
    metrics = preparation._snapshot_metrics(trades=trades, dates=dates, sessions=("ASIA", "EUROPE", "NY"))
    objective = preparation.robust_objective(_objective_metrics(metrics), scaling)
    return {
        "parameters": dict(params), "qualified_count": int(qualified.sum()), "trades": trades,
        "metrics": {key: value for key, value in metrics.items() if key not in {"per_date", "per_session_date"}},
        "objective_components": objective,
        "family_metrics": _exact_family_report(trades, pool.family_ids, dates),
    }


def _rank_corr(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


def _representative_trials(records: list[dict[str, Any]], count: int = 100) -> list[dict[str, Any]]:
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        p = row["parameters"]
        return tuple(round(float(p[name]), 10) for name in ALL_A_NAMES)

    def add(rows: Sequence[dict[str, Any]], limit: int) -> None:
        for row in rows:
            if len(selected) >= count or limit <= 0:
                return
            key = vector(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            limit -= 1

    by_objective = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    by_net = sorted(complete, key=lambda row: (-float(row["metrics"]["net_r"]), int(row["number"])))
    by_trades = sorted(complete, key=lambda row: (float(row["metrics"]["total_trades"]), -float(row["objective"]), int(row["number"])))
    by_medium = [row for row in by_objective if 101 <= float(row["metrics"]["total_trades"]) <= 250]
    add(by_objective, 25); add(by_net, 20)
    clusters: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in by_objective:
        p = row["parameters"]
        key = tuple(int(float(p[name]) * 10) for name in candidate_tape.WEIGHT_NAMES + ("min_quality_score",))
        clusters.setdefault(key, row)
    add(list(clusters.values()), 15)
    add([row for row in by_objective if float(row["metrics"]["total_trades"]) <= 100], 10)
    add(by_medium, 10)
    add([row for row in by_objective if float(row["metrics"]["total_trades"]) > 250], 10)
    add(sorted(complete, key=lambda row: int(row["number"])), 10)
    return selected[:count]


def _parameter_regions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    output: dict[str, Any] = {}
    for label, rows in (
        ("top_100", sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))[:100]),
        ("top_500", sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))[:500]),
        ("top_1000", sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))[:1000]),
    ):
        summary = {}
        for name in ALL_A_NAMES:
            values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=np.float64)
            summary[name] = {"min": float(values.min()), "q10": float(np.quantile(values, .10)),
                             "median": float(np.quantile(values, .50)), "q90": float(np.quantile(values, .90)),
                             "max": float(values.max())}
        output[label] = {"count": len(rows), "parameters": summary}
    return output


def run_stage2a(*, manifest_path: Path = preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME,
                output_root: Path = preparation.DEFAULT_OUTPUT_ROOT, trials: int = STAGE2A_TRIALS,
                seed: int = STAGE2A_SEED) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    scaling = _load_scaling(output_root)
    search_space = _search_space()
    _atomic_json(output_root / STAGE2A_SEARCH_SPACE_NAME, {
        **search_space, "tape_manifest": str(bundle.manifest_path),
        "tape_manifest_sha256": _sha256(bundle.manifest_path),
    })
    pool = load_all_a_pool(output_root=output_root, train_dates=bundle.dates)
    if pool is None:
        pool = build_all_a_pool(bundle)
        save_all_a_pool(pool, output_root=output_root, train_dates=bundle.dates)
    db_path = output_root / STAGE2A_DB_NAME
    # Standard seeded TPE keeps suggestion ordering deterministic while the
    # append-only journal avoids SQLite transaction contention at 50k trials.
    sampler = optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4)
    journal = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(db_path))
    )
    study = optuna.create_study(direction="maximize", study_name=STAGE2A_STUDY_NAME,
                                sampler=sampler, storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial)
        weights = np.asarray([params[name] for name in candidate_tape.WEIGHT_NAMES], dtype=float)
        if (not np.all(np.isfinite(weights)) or np.any(weights <= 0.0)
                or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)):
            trial.set_user_attr("technical_valid", False)
            return -1.0e12
        metrics = fast_metrics(pool, qualified_mask(pool, params), len(bundle.dates))
        if not all(math.isfinite(float(metrics[name])) for name in (
                "total_trades", "net_r", "profit_factor", "max_drawdown_r", "active_dates",
                "profitable_date_ratio", "median_date_r", "lower_quartile_date_r", "downside_tail",
                "median_session_date_r", "date_concentration", "family_concentration", "active_sessions")):
            trial.set_user_attr("technical_valid", False)
            return -1.0e12
        components = preparation.robust_objective(_objective_metrics(metrics), scaling)
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("parameters", params)
        trial.set_user_attr("normalized_weights", {name: float(params[name]) for name in candidate_tape.WEIGHT_NAMES})
        trial.set_user_attr("metrics", {key: value for key, value in metrics.items() if key not in {"date_r", "session_date_r"}})
        trial.set_user_attr("objective_components", components)
        return float(components["robust_score"])

    completed_before = len(study.trials)
    remaining = max(0, int(trials) - completed_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False,
                       show_progress_bar=False)
    records = [_trial_record(row) for row in study.trials]
    _write_trial_jsonl(output_root / STAGE2A_TRIALS_NAME, study.trials)
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    if len(complete) < int(trials):
        raise preparation.TrainOptimizationError(f"Stage2A did not complete {trials} technical trials")
    representatives = _representative_trials(records, count=min(100, len(complete)))
    exact_results = []
    for row in representatives:
        exact = _exact_result(pool, row["parameters"], scaling, bundle.dates)
        exact_results.append({"trial_number": row["number"], "fast": row, "exact": exact})
    fast_scores = [float(row["fast"]["objective"]) for row in exact_results]
    exact_scores = [float(row["exact"]["objective_components"]["robust_score"]) for row in exact_results]
    exact_nets = [float(row["exact"]["metrics"]["net_r"]) for row in exact_results]
    fast_nets = [float(row["fast"]["metrics"]["net_r"]) for row in exact_results]
    exact_report = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "count": len(exact_results),
        "results": exact_results,
        "fast_exact_rank_correlation": {"robust_score": _rank_corr(fast_scores, exact_scores),
                                         "net_r": _rank_corr(fast_nets, exact_nets)},
        "fast_kernel_exact_proxy": (
            "STRONG" if _rank_corr(fast_scores, exact_scores) >= .80 else
            "ACCEPTABLE" if _rank_corr(fast_scores, exact_scores) >= .60 else
            "WEAK" if _rank_corr(fast_scores, exact_scores) >= .30 else "INVALID"
        ),
    }
    _atomic_json(output_root / STAGE2A_EXACT_NAME, exact_report)
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    summary = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": STAGE2A_OBJECTIVE_VERSION,
        "study_name": STAGE2A_STUDY_NAME, "seed": seed, "requested_trials": int(trials),
        "completed_trials": len(complete), "technical_valid_trials": len(complete),
        "candidate_count": pool.count, "train_dates": list(bundle.dates),
        "best_objective": ranked[0], "top_100": ranked[:100], "top_500": ranked[:500],
        "top_1000": ranked[:1000], "parameter_regions": _parameter_regions(records),
        "frequency_bands": {
            "low": max((row for row in complete if float(row["metrics"]["total_trades"]) <= 100),
                       key=lambda row: (float(row["metrics"]["net_r"]), -int(row["number"])), default=None),
            "medium": max((row for row in complete if 101 <= float(row["metrics"]["total_trades"]) <= 250),
                          key=lambda row: (float(row["metrics"]["net_r"]), -int(row["number"])), default=None),
            "high": max((row for row in complete if float(row["metrics"]["total_trades"]) > 250),
                        key=lambda row: (float(row["metrics"]["net_r"]), -int(row["number"])), default=None),
        },
        "exact_shortlist_count": len(exact_results), "fast_kernel_exact_proxy": exact_report["fast_kernel_exact_proxy"],
        "fast_exact_rank_correlation": exact_report["fast_exact_rank_correlation"],
        "ready_for_stage2b": exact_report["fast_kernel_exact_proxy"] in {"STRONG", "ACCEPTABLE"},
        "stage2b_recommendation": "Run exact event-path Stage-2B only after reviewing Stage-2A top regions and exact shortlist.",
        "artifacts": {"study_db": str(db_path), "trials": str(output_root / STAGE2A_TRIALS_NAME),
                      "exact_shortlist": str(output_root / STAGE2A_EXACT_NAME)},
    }
    _atomic_json(output_root / STAGE2A_SUMMARY_NAME, summary)
    return summary


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME)
    parser.add_argument("--output-root", type=Path, default=preparation.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trials", type=int, default=STAGE2A_TRIALS)
    parser.add_argument("--seed", type=int, default=STAGE2A_SEED)
    args = parser.parse_args(argv)
    result = run_stage2a(manifest_path=args.manifest, output_root=args.output_root,
                         trials=args.trials, seed=args.seed)
    best = result["best_objective"]
    print(json.dumps({
        "OPTUNA_STAGE2A_TRIALS": result["completed_trials"],
        "BEST_OBJECTIVE": best["objective"], "BEST_NET_R": best["metrics"]["net_r"],
        "FAST_EXACT_RANK_CORRELATION": result["fast_exact_rank_correlation"],
        "OPTUNA_FAST_EXACT_PROXY": result["fast_kernel_exact_proxy"],
        "EXACT_SHORTLIST_COUNT": result["exact_shortlist_count"],
        "READY_FOR_STAGE2B": result["ready_for_stage2b"],
        "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
