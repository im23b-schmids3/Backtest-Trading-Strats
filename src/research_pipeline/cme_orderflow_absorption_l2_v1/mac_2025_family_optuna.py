"""TRAIN-only family-specific Class-A Optuna pilot.

This is intentionally separate from the completed global Stage2A study.  Each
family has its own JournalStorage file and objective, and the study evaluates
only that family's sealed candidate opportunities.  The module never opens
Validation, OOS, or raw DBN artifacts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna

from . import mac_2025_train_optuna as stage2a
from . import mac_2025_train_optimization as preparation


PILOT_TRIALS = 10_000
PILOT_SEED = 20250923
PILOT_OBJECTIVE_VERSION = "MAC2025_TRAIN_FAMILY_NET_R_OBJECTIVE_V1"
PILOT_ROOT_NAME = "family-optuna-pilot"
PILOT_SUMMARY_NAME = "family-optuna-pilot-summary.json"

SELECTED_FAMILIES = (
    "NY|EUROPE|CURRENT|POC",       # consistently strong, 15 median trades
    "ASIA|ASIA|CURRENT|LOW",        # parameter-sensitive, 49 median trades
    "NY|ASIA|PRIOR|HIGH",           # strong and different session/context
)

_CORE_REGION_NAMES = tuple(stage2a.candidate_tape.WEIGHT_NAMES) + ("min_quality_score",)
_BOUNDARIES: dict[str, tuple[float, float]] = {
    "min_quality_score": (0.50, 0.80),
    "min_relevant_aggressive_volume": (0.0, 300.0),
    "min_relevant_execution_count": (0.0, 12.0),
    "min_consume_restore_cycles": (0.0, 6.0),
    "max_through_level_progress_ticks": (0.25, 12.0),
    "min_rejection_ticks": (0.0, 4.0),
    "false_refill_penalty_weight": (0.0, 1.0),
    "unexecuted_add_penalty_component_weight": (0.0, 1.0),
    "rapid_cancel_penalty_component_weight": (0.0, 1.0),
    "adverse_progress_penalty_component_weight": (0.0, 1.0),
    "aggressive_volume_saturation": (25.0, 1000.0),
    "execution_count_saturation": (1.0, 32.0),
    "restore_cycle_saturation": (0.5, 10.0),
    "restoration_ratio_saturation": (0.25, 4.0),
    "rejection_saturation_ticks": (0.5, 12.0),
    "persistence_depth_saturation": (10.0, 500.0),
    "restoration_latency_saturation_ms": (100.0, 5000.0),
    "multi_level_ofi_saturation": (10.0, 500.0),
}


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


def _slug(family_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", family_id).strip("-").lower()


def family_output_root(output_root: Path, family_id: str) -> Path:
    return output_root / PILOT_ROOT_NAME / _slug(family_id)


def _family_mask(pool: stage2a.AllAOpportunityPool, family_id: str) -> np.ndarray:
    try:
        index = pool.family_ids.index(family_id)
    except ValueError as exc:
        raise preparation.TrainOptimizationError(f"unknown family: {family_id}") from exc
    return pool.family_index == index


def _family_metrics(pool: stage2a.AllAOpportunityPool, qualified: np.ndarray,
                    family_id: str, date_count: int) -> dict[str, Any]:
    return stage2a.fast_metrics(pool, qualified & _family_mask(pool, family_id), date_count)


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return stage2a._trial_record(trial)


def _write_trials(path: Path, trials: Sequence[optuna.trial.FrozenTrial]) -> None:
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


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise preparation.TrainOptimizationError("non-finite family metric")
    return number


def _metric_distribution(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, float]:
    values = np.asarray([_finite(row["metrics"][name]) for row in rows], dtype=np.float64)
    return {
        "best": float(values.max()), "q99_9": float(np.quantile(values, .999)),
        "q99": float(np.quantile(values, .99)), "q95": float(np.quantile(values, .95)),
        "q90": float(np.quantile(values, .90)), "median": float(np.quantile(values, .50)),
        "worst": float(values.min()),
    }


def _parameter_bands(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in stage2a.ALL_A_NAMES:
        values = np.asarray([_finite(row["parameters"][name]) for row in rows], dtype=np.float64)
        entry: dict[str, Any] = {
            "min": float(values.min()), "q10": float(np.quantile(values, .10)),
            "median": float(np.quantile(values, .50)), "q90": float(np.quantile(values, .90)),
            "max": float(values.max()),
        }
        if name in _BOUNDARIES:
            low, high = _BOUNDARIES[name]
            span = high - low or 1.0
            entry["near_lower_boundary_ratio"] = float(np.mean(values <= low + .05 * span))
            entry["near_upper_boundary_ratio"] = float(np.mean(values >= high - .05 * span))
            entry["boundary"] = [low, high]
        output[name] = entry
    return output


def _row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_number": int(row["number"]), "objective": float(row["objective"]),
        "net_r": float(row["metrics"]["net_r"]),
        "profit_factor": float(row["metrics"]["profit_factor"]),
        "max_drawdown_r": float(row["metrics"]["max_drawdown_r"]),
        "total_trades": float(row["metrics"]["total_trades"]),
        "active_dates": float(row["metrics"]["active_dates"]),
        "profitable_date_ratio": float(row["metrics"]["profitable_date_ratio"]),
        "parameters": dict(row["parameters"]),
    }


def _region_clusters(rows: Sequence[Mapping[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(int(round(float(row["parameters"][name]) * 10.0)) for name in _CORE_REGION_NAMES)
        groups.setdefault(key, []).append(row)
    clusters = []
    for key, members in groups.items():
        best = max(members, key=lambda row: (float(row["objective"]), -int(row["number"])))
        clusters.append({
            "count": len(members), "best": _row_summary(best),
            "median_objective": float(np.median([float(row["objective"]) for row in members])),
            "parameter_median": {
                name: float(np.median([float(row["parameters"][name]) for row in members]))
                for name in _CORE_REGION_NAMES
            },
        })
    return sorted(clusters, key=lambda row: (-row["best"]["objective"], -row["count"]))[:limit]


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (-float(row["objective"]), int(row["number"])))
    metrics = ("net_r", "profit_factor", "max_drawdown_r", "total_trades",
               "active_dates", "profitable_date_ratio", "date_concentration", "downside_tail")
    return {
        "count": len(rows), "best": _row_summary(ranked[0]),
        "metric_distributions": {name: _metric_distribution(rows, name) for name in metrics},
        "parameter_bands": _parameter_bands(rows),
        "region_clusters": _region_clusters(rows),
    }


def _representatives(rows: Sequence[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(round(float(row["parameters"][name]), 10) for name in stage2a.ALL_A_NAMES)

    def add(candidates: Sequence[dict[str, Any]], limit: int) -> None:
        for row in candidates:
            if len(selected) >= count or limit <= 0:
                return
            key = vector(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            limit -= 1

    by_net = sorted(rows, key=lambda row: (-float(row["metrics"]["net_r"]), int(row["number"])))
    by_objective = sorted(rows, key=lambda row: (-float(row["objective"]), int(row["number"])))
    clusters: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in by_objective:
        key = tuple(int(round(float(row["parameters"][name]) * 10.0)) for name in _CORE_REGION_NAMES)
        clusters.setdefault(key, row)
    add(by_net, 6)
    add(by_objective, 4)
    add(list(clusters.values()), 5)
    add([row for row in by_net if float(row["metrics"]["total_trades"]) <= 100], 3)
    add(sorted(rows, key=lambda row: int(row["number"])), 2)
    return selected[:count]


def _exact_family_result(pool: stage2a.AllAOpportunityPool, params: Mapping[str, Any],
                         family_id: str, dates: Sequence[str]) -> dict[str, Any]:
    qualified = stage2a.qualified_mask(pool, params) & _family_mask(pool, family_id)
    trades = stage2a._exact_trades(pool, qualified, dates)
    metrics = preparation._snapshot_metrics(trades=trades, dates=dates, sessions=("ASIA", "EUROPE", "NY"))
    return {
        "qualified_count": int(qualified.sum()),
        "net_r": float(metrics["net_r"]), "profit_factor": float(metrics["profit_factor_capped"]),
        "max_drawdown_r": float(metrics["max_drawdown_r"]), "trades": float(metrics["total_trades"]),
        "active_dates": float(metrics["active_dates"]),
        "profitable_date_ratio": float(metrics["profitable_date_ratio"]),
        "date_r": list(metrics["date_r"].values()),
    }


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return stage2a._rank_corr(left, right)


def _exact_report(pool: stage2a.AllAOpportunityPool, family_id: str,
                  rows: Sequence[dict[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    representatives = _representatives(list(rows), count=20)
    results = []
    for row in representatives:
        fast = row["metrics"]
        exact = _exact_family_result(pool, row["parameters"], family_id, dates)
        results.append({
            "trial_number": int(row["number"]),
            "fast": {name: float(fast[name]) for name in ("net_r", "profit_factor", "max_drawdown_r", "total_trades")},
            "exact": {name: exact[name] for name in ("net_r", "profit_factor", "max_drawdown_r", "trades", "active_dates", "profitable_date_ratio")},
            "parameters": dict(row["parameters"]),
        })
    metrics = ("net_r", "profit_factor", "max_drawdown_r", "trades")
    rank = {}
    errors = {}
    for name in metrics:
        fast_key = "total_trades" if name == "trades" else name
        rank[name] = _rank_correlation([row["fast"][fast_key] for row in results],
                                        [row["exact"][name] for row in results])
        errors[name] = float(np.median([abs(row["fast"][fast_key] - row["exact"][name]) for row in results]))
    net_top = lambda row, side: (row[side]["net_r"])
    top10_fast = {row["trial_number"] for row in sorted(results, key=lambda row: -net_top(row, "fast"))[:10]}
    top10_exact = {row["trial_number"] for row in sorted(results, key=lambda row: -net_top(row, "exact"))[:10]}
    proxy = "STRONG" if rank["net_r"] >= .80 else "ACCEPTABLE" if rank["net_r"] >= .60 else "WEAK" if rank["net_r"] >= .30 else "INVALID"
    return {
        "count": len(results), "results": results, "rank_correlation": rank,
        "median_absolute_error": errors, "top10_overlap": len(top10_fast & top10_exact),
        "fast_exact_proxy": proxy,
    }


def _validate_selected_families(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    available = set(report.get("families", {}))
    if not set(SELECTED_FAMILIES) <= available:
        raise preparation.TrainOptimizationError("selected family missing from existing report")
    strong = set(report.get("consistently_strong", []))
    sensitive = set(report.get("parameter_sensitive", []))
    if SELECTED_FAMILIES[0] not in strong or SELECTED_FAMILIES[1] not in sensitive or SELECTED_FAMILIES[2] not in strong:
        raise preparation.TrainOptimizationError("selected family classification changed")


def run_family(family_id: str, *, manifest_path: Path, output_root: Path,
               trials: int = PILOT_TRIALS, seed: int = PILOT_SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    pool = stage2a.load_all_a_pool(output_root=output_root, train_dates=bundle.dates)
    if pool is None:
        raise preparation.TrainOptimizationError("sealed Stage2A opportunity pool is missing or invalid")
    _family_mask(pool, family_id)
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    study_name = f"MAC2025_TRAIN_FAMILY_{_slug(family_id).upper()}_TPE_JOURNAL_V1"
    db_path = root / "family-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(db_path)))
    sampler = optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4)
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=sampler, storage=journal, load_if_exists=True)
    family_index = pool.family_ids.index(family_id)

    def objective(trial: optuna.Trial) -> float:
        params = stage2a._trial_params(trial)
        weights = np.asarray([params[name] for name in stage2a.candidate_tape.WEIGHT_NAMES], dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
            trial.set_user_attr("technical_valid", False)
            return -1.0e12
        mask = stage2a.qualified_mask(pool, params) & (pool.family_index == family_index)
        metrics = stage2a.fast_metrics(pool, mask, len(bundle.dates))
        if not all(math.isfinite(float(metrics[name])) for name in ("total_trades", "net_r", "profit_factor", "max_drawdown_r", "active_dates", "profitable_date_ratio")):
            trial.set_user_attr("technical_valid", False)
            return -1.0e12
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", params)
        trial.set_user_attr("normalized_weights", {name: float(params[name]) for name in stage2a.candidate_tape.WEIGHT_NAMES})
        trial.set_user_attr("metrics", {key: value for key, value in metrics.items() if key not in {"date_r", "session_date_r"}})
        trial.set_user_attr("objective_components", {"net_r": float(metrics["net_r"]), "objective_version": PILOT_OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_trial_record(row) for row in study.trials]
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    failed = sum(row["state"] == "FAIL" for row in records)
    if len(complete) < int(trials):
        raise preparation.TrainOptimizationError(f"family study incomplete: {family_id}: {len(complete)}/{trials}, failed={failed}")
    _write_trials(root / "family-trials.jsonl", study.trials)
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    summary = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": PILOT_OBJECTIVE_VERSION, "study_name": study_name,
        "seed": seed, "requested_trials": int(trials), "complete_trials": len(complete),
        "failed_trials": failed, "best": _row_summary(ranked[0]),
        "top_50": _group_summary(ranked[:50]), "top_100": _group_summary(ranked[:100]),
        "top_500": _group_summary(ranked[:500]),
        "exact": _exact_report(pool, family_id, complete, bundle.dates),
        "artifacts": {"journal": str(db_path), "trials": str(root / "family-trials.jsonl")},
    }
    _atomic_json(root / "family-summary.json", summary)
    return summary


def run_pilot(*, manifest_path: Path = preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME,
              output_root: Path = preparation.DEFAULT_OUTPUT_ROOT, trials: int = PILOT_TRIALS,
              seed: int = PILOT_SEED, workers: int = 3) -> dict[str, Any]:
    report_path = output_root / "train-stage2a-family-contribution.json"
    _validate_selected_families(report_path)
    print(json.dumps({"selected_families": list(SELECTED_FAMILIES), "objective": PILOT_OBJECTIVE_VERSION}, sort_keys=True))
    jobs = [(family, manifest_path, output_root, int(trials), int(seed) + index) for index, family in enumerate(SELECTED_FAMILIES)]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = [executor.submit(run_family, family, manifest_path=manifest, output_root=root, trials=count, seed=job_seed)
                   for family, manifest, root, count, job_seed in jobs]
        for future in futures:
            results.append(future.result())
    results.sort(key=lambda row: SELECTED_FAMILIES.index(row["family_id"]))
    payload = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": PILOT_OBJECTIVE_VERSION,
        "selected_families": list(SELECTED_FAMILIES), "requested_trials_per_family": int(trials),
        "studies": results, "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_json(output_root / PILOT_SUMMARY_NAME, payload)
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME)
    parser.add_argument("--output-root", type=Path, default=preparation.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trials", type=int, default=PILOT_TRIALS)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_pilot(manifest_path=args.manifest, output_root=args.output_root, trials=args.trials,
                       seed=args.seed, workers=args.workers)
    print(json.dumps({
        "FAMILY_OPTUNA_PILOT_COMPLETE": result["status"] == "COMPLETE",
        "FAMILIES": result["selected_families"], "TRIALS_PER_FAMILY": result["requested_trials_per_family"],
        "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
