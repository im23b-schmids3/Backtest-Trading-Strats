"""Post-run TRAIN-only analysis for the completed Stage-2A study.

The analysis consumes only the persisted Optuna JournalStorage study and the
sealed candidate-tape opportunity cache.  It creates no trials and does not
open market-data files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna

from . import mac_2025_train_optuna as stage2a


OUTPUT_NAME = "train-stage2a-complete-analysis.json"
EXACT_NAME = "train-stage2a-exact-shortlist-v2.json"
FAMILY_NAME = "train-stage2a-family-contribution.json"

METRICS = (
    "robust_objective", "net_r", "profit_factor", "max_drawdown_r", "total_trades",
    "active_dates", "profitable_date_ratio", "median_date_r", "lower_quartile_date_r",
    "downside_tail", "date_concentration", "family_concentration",
)
PARAMETERS = stage2a.ALL_A_NAMES


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in (_finite(v) for v in values) if math.isfinite(value)], dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)), "best": float(np.max(array)),
        "q99_9": float(np.quantile(array, .999)), "q99": float(np.quantile(array, .99)),
        "q95": float(np.quantile(array, .95)), "q90": float(np.quantile(array, .90)),
        "median": float(np.quantile(array, .50)), "worst": float(np.min(array)),
    }


def _metric_value(row: Mapping[str, Any], name: str) -> float:
    if name == "robust_objective":
        return _finite(row.get("objective"))
    return _finite(row.get("metrics", {}).get(name))


def _row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_number": int(row["number"]), "objective": float(row["objective"]),
        "net_r": float(row["metrics"]["net_r"]), "profit_factor": float(row["metrics"]["profit_factor"]),
        "max_drawdown_r": float(row["metrics"]["max_drawdown_r"]),
        "trades": float(row["metrics"]["total_trades"]),
        "parameters": {name: float(row["parameters"][name]) for name in PARAMETERS},
    }


def _ranked(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row["objective"]), int(row["number"])))


def _population_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "metric_distributions": {
            name: _distribution([_metric_value(row, name) for row in rows]) for name in METRICS
        },
        "best_objective": _row_summary(_ranked(rows)[0]) if rows else None,
        "best_net_r": _row_summary(sorted(rows, key=lambda row: (-float(row["metrics"]["net_r"]), int(row["number"])))[0]) if rows else None,
    }


def _bounds() -> dict[str, tuple[float, float]]:
    return {
        "min_quality_score": (.50, .80), "min_relevant_aggressive_volume": (0, 300),
        "min_relevant_execution_count": (0, 12), "min_consume_restore_cycles": (0, 6),
        "max_through_level_progress_ticks": (.25, 12), "min_rejection_ticks": (0, 4),
        **{name: (0, 1) for name in stage2a.PENALTY_NAMES},
        "aggressive_volume_saturation": (25, 1000), "execution_count_saturation": (1, 32),
        "restore_cycle_saturation": (.5, 10), "restoration_ratio_saturation": (.25, 4),
        "rejection_saturation_ticks": (.5, 12), "persistence_depth_saturation": (10, 500),
        "restoration_latency_saturation_ms": (100, 5000), "multi_level_ofi_saturation": (10, 500),
    }


def _parameter_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "parameters": {}, "pairwise_dependencies": []}
    objective = np.asarray([float(row["objective"]) for row in rows], dtype=float)
    bounds = _bounds()
    values: dict[str, np.ndarray] = {
        name: np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        for name in PARAMETERS
    }
    parameters: dict[str, Any] = {}
    for name, array in values.items():
        entry: dict[str, Any] = {
            "min": float(np.min(array)), "q10": float(np.quantile(array, .10)),
            "median": float(np.quantile(array, .50)), "q90": float(np.quantile(array, .90)),
            "max": float(np.max(array)),
        }
        if name in bounds:
            low, high = bounds[name]
            width = high - low
            entry["near_lower_boundary_ratio"] = float((array <= low + .01 * width).mean())
            entry["near_upper_boundary_ratio"] = float((array >= high - .01 * width).mean())
            entry["boundary"] = [low, high]
        if np.std(array) and np.std(objective):
            entry["objective_correlation"] = float(np.corrcoef(array, objective)[0, 1])
        else:
            entry["objective_correlation"] = 0.0
        parameters[name] = entry
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(PARAMETERS):
        for right in PARAMETERS[left_index + 1:]:
            a, b = values[left], values[right]
            if np.std(a) and np.std(b):
                correlation = float(np.corrcoef(a, b)[0, 1])
                pairwise.append({"left": left, "right": right, "correlation": correlation,
                                 "absolute_correlation": abs(correlation)})
    pairwise.sort(key=lambda row: (-row["absolute_correlation"], row["left"], row["right"]))
    return {"count": len(rows), "parameters": parameters, "pairwise_dependencies": pairwise[:25]}


def _cluster(rows: Sequence[dict[str, Any]], cluster_count: int = 8) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = _ranked(rows)
    matrix = np.asarray([[float(row["parameters"][name]) for name in PARAMETERS] for row in ordered], dtype=float)
    center = matrix.mean(axis=0); scale = matrix.std(axis=0); scale[scale == 0] = 1.0
    standardized = (matrix - center) / scale
    count = min(cluster_count, len(ordered))
    seed_positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    centers = standardized[seed_positions].copy()
    labels = np.zeros(len(ordered), dtype=int)
    for _ in range(40):
        distances = ((standardized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for index in range(count):
            members = standardized[new_labels == index]
            if len(members):
                new_centers[index] = members.mean(axis=0)
        if np.array_equal(labels, new_labels):
            break
        labels, centers = new_labels, new_centers
    clusters = []
    for index in range(count):
        members = [ordered[position] for position in np.flatnonzero(labels == index)]
        if not members:
            continue
        clusters.append({
            "cluster": index, "count": len(members),
            "best": _row_summary(_ranked(members)[0]),
            "median_objective": float(np.median([row["objective"] for row in members])),
            "median_net_r": float(np.median([row["metrics"]["net_r"] for row in members])),
            "parameter_median": {name: float(np.median([row["parameters"][name] for row in members])) for name in PARAMETERS},
        })
    return sorted(clusters, key=lambda row: (-row["best"]["objective"], row["cluster"]))


def _representatives(rows: Sequence[dict[str, Any]], count: int = 100) -> list[dict[str, Any]]:
    selected = stage2a._representative_trials(list(rows), count=count)
    seen = {tuple(round(float(row["parameters"][name]), 10) for name in PARAMETERS) for row in selected}
    rng = np.random.default_rng(stage2a.STAGE2A_SEED)
    ordered = _ranked(rows)
    categories = [
        [row for row in ordered if row["metrics"]["total_trades"] <= 100],
        [row for row in ordered if 101 <= row["metrics"]["total_trades"] <= 250],
        [row for row in ordered if row["metrics"]["total_trades"] > 250],
        sorted(rows, key=lambda row: int(row["number"])),
    ]
    for category in categories:
        for row in category:
            if len(selected) >= count:
                break
            key = tuple(round(float(row["parameters"][name]), 10) for name in PARAMETERS)
            if key not in seen:
                seen.add(key); selected.append(row)
        if len(selected) >= count:
            break
    if len(selected) < count:
        indices = rng.permutation(len(rows))
        for index in indices:
            row = rows[int(index)]
            key = tuple(round(float(row["parameters"][name]), 10) for name in PARAMETERS)
            if key not in seen:
                seen.add(key); selected.append(row)
            if len(selected) >= count:
                break
    return selected[:count]


def _compact_fast(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"objective": float(row["objective"]), "net_r": float(metrics["net_r"]),
            "profit_factor": float(metrics["profit_factor"]), "max_drawdown_r": float(metrics["max_drawdown_r"]),
            "trades": float(metrics["total_trades"]), "date_r": list(metrics["date_r"]),
            "parameters": {name: float(row["parameters"][name]) for name in PARAMETERS}}


def _compact_exact(exact: Mapping[str, Any]) -> dict[str, Any]:
    metrics = exact["metrics"]
    return {"objective": float(exact["objective_components"]["robust_score"]),
            "net_r": float(metrics["net_r"]), "profit_factor": float(metrics["profit_factor"] or 0.0),
            "max_drawdown_r": float(metrics["max_drawdown_r"]), "trades": float(metrics["total_trades"]),
            "date_r": dict(metrics["date_r"]), "family_metrics": exact["family_metrics"],
            "objective_components": exact["objective_components"], "qualified_count": exact["qualified_count"]}


def _correlation(rows: Sequence[Mapping[str, Any]], fast_key: str, exact_key: str) -> float:
    fast = np.asarray([float(row["fast"][fast_key]) for row in rows], dtype=float)
    exact = np.asarray([float(row["exact"][exact_key]) for row in rows], dtype=float)
    return stage2a._rank_corr(fast, exact)


def run_analysis(*, output_root: Path = stage2a.preparation.DEFAULT_OUTPUT_ROOT,
                 journal_path: Path | None = None, shortlist_count: int = 100) -> dict[str, Any]:
    journal_path = journal_path or output_root / stage2a.STAGE2A_DB_NAME
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    study = optuna.load_study(study_name=stage2a.STAGE2A_STUDY_NAME, storage=storage)
    trials = list(study.trials)
    complete = [stage2a._trial_record(trial) for trial in trials
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("technical_valid")]
    if len(complete) != 50_000:
        raise RuntimeError(f"expected 50000 technically valid trials, found {len(complete)}")
    ranked = _ranked(complete)
    top_groups = {f"top_{size}": ranked[:size] for size in (100, 500, 1000)}
    top_groups["top_5_percent"] = ranked[:max(1, len(ranked) // 20)]
    scaling = stage2a._load_scaling(output_root)
    pool = stage2a.load_all_a_pool(output_root=output_root, train_dates=stage2a.preparation.EXPECTED_TRAIN_DATES)
    if pool is None:
        raise RuntimeError("sealed Stage2A opportunity pool is unavailable")
    representatives = _representatives(complete, shortlist_count)
    existing: dict[int, dict[str, Any]] = {}
    old_path = output_root / stage2a.STAGE2A_EXACT_NAME
    if old_path.is_file():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        existing = {int(row["trial_number"]): row for row in old.get("results", [])}
    exact_rows: list[dict[str, Any]] = []
    for row in representatives:
        fast_metrics = stage2a.fast_metrics(
            pool, stage2a.qualified_mask(pool, row["parameters"]), len(stage2a.preparation.EXPECTED_TRAIN_DATES)
        )
        exact = existing.get(int(row["number"]))
        if exact is None:
            exact_payload = stage2a._exact_result(pool, row["parameters"], scaling, stage2a.preparation.EXPECTED_TRAIN_DATES)
            exact_compact = _compact_exact(exact_payload)
        else:
            exact_compact = _compact_exact({
                "objective_components": exact["exact"]["objective_components"],
                "metrics": exact["exact"]["metrics"], "family_metrics": exact["exact"]["exact"]["family_metrics"]
                if "exact" in exact["exact"] else exact["exact"].get("family_metrics", {}),
                "qualified_count": exact["exact"].get("qualified_count", 0),
            })
        exact_rows.append({"trial_number": int(row["number"]), "fast": _compact_fast(row, fast_metrics), "exact": exact_compact})
    exact_rows.sort(key=lambda row: row["trial_number"])
    exact_report = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "count": len(exact_rows), "results": exact_rows,
        "fast_exact_rank_correlation": {
            "objective": _correlation(exact_rows, "objective", "objective"),
            "net_r": _correlation(exact_rows, "net_r", "net_r"),
            "profit_factor": _correlation(exact_rows, "profit_factor", "profit_factor"),
            "max_drawdown_r": _correlation(exact_rows, "max_drawdown_r", "max_drawdown_r"),
            "trades": _correlation(exact_rows, "trades", "trades"),
        },
    }
    robust_corr = exact_report["fast_exact_rank_correlation"]["objective"]
    exact_report["fast_kernel_exact_proxy"] = "STRONG" if robust_corr >= .80 else "ACCEPTABLE" if robust_corr >= .60 else "WEAK" if robust_corr >= .30 else "INVALID"
    # Use the preparation writer directly to avoid any study mutation.
    stage2a.preparation._json_write(output_root / EXACT_NAME, exact_report)
    family_report: dict[str, Any] = {}
    family_ids = sorted(pool.family_ids)
    for family in family_ids:
        rows = [item["exact"]["family_metrics"][family] for item in exact_rows]
        family_report[family] = {
            "configs": len(rows), "median_trades": float(np.median([r["trades"] for r in rows])),
            "median_net_r": float(np.median([r["net_r"] for r in rows])),
            "median_profit_factor": float(np.median([r["profit_factor"] for r in rows])),
            "median_active_dates": float(np.median([r["active_dates"] for r in rows])),
            "median_profitable_dates": float(np.median([r["profitable_dates"] for r in rows])),
            "positive_config_ratio": float(np.mean([r["net_r"] > 0 for r in rows])),
            "contribution_share_median": float(np.median([r["contribution_share"] for r in rows])),
        }
    strong = [name for name, row in family_report.items() if row["median_net_r"] > 0 and row["positive_config_ratio"] >= .60]
    weak = [name for name, row in family_report.items() if row["median_net_r"] < 0 and row["positive_config_ratio"] <= .25]
    sensitive = [name for name, row in family_report.items() if row["positive_config_ratio"] >= .25 and row["positive_config_ratio"] <= .75]
    family_payload = {"status": "COMPLETE", "scope": "TRAIN_ONLY", "family_count": len(family_ids),
                      "families": family_report, "consistently_strong": strong,
                      "parameter_sensitive": sensitive, "consistently_weak": weak}
    stage2a.preparation._json_write(output_root / FAMILY_NAME, family_payload)
    frequency: dict[str, Any] = {}
    for name, predicate in (("low", lambda r: r["metrics"]["total_trades"] <= 100),
                            ("medium", lambda r: 101 <= r["metrics"]["total_trades"] <= 250),
                            ("high", lambda r: r["metrics"]["total_trades"] > 250)):
        subset = [row for row in complete if predicate(row)]
        frequency[name] = {"count": len(subset),
                           "best_objective": _row_summary(_ranked(subset)[0]) if subset else None,
                           "best_net_r": _row_summary(sorted(subset, key=lambda r: (-r["metrics"]["net_r"], r["number"]))[0]) if subset else None}
    analysis = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": stage2a.STAGE2A_OBJECTIVE_VERSION,
        "trial_count": len(complete), "best_objective": float(study.best_value),
        "full_distributions": {name: _distribution([_metric_value(row, name) for row in complete]) for name in METRICS},
        "top_groups": {name: _population_report(rows) for name, rows in top_groups.items()},
        "parameter_analysis": {name: _parameter_report(rows) for name, rows in top_groups.items()},
        "robust_global_regions": _cluster(ranked[:1000]),
        "frequency_buckets": frequency,
        "exact_shortlist_count": len(exact_rows),
        "fast_exact": exact_report,
        "family_summary": family_payload,
        "stage2b_decision": "STAGE2B_GLOBAL_REFINEMENT" if exact_report["fast_kernel_exact_proxy"] in {"STRONG", "ACCEPTABLE"} else "NO_STAGE2B_NEEDED",
        "stage2b_recommendation": "Run a narrowed exact event-path refinement after reviewing the 100-trial shortlist; do not use validation/OOS for selection.",
    }
    stage2a.preparation._json_write(output_root / OUTPUT_NAME, analysis)
    return analysis


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=stage2a.preparation.DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run_analysis(output_root=args.output_root)
    print(json.dumps({
        "STAGE2A_COMPLETE": True, "BEST_OBJECTIVE": result["best_objective"],
        "EXACT_SHORTLIST_COUNT": result["exact_shortlist_count"],
        "OPTUNA_FAST_EXACT_PROXY": result["fast_exact"]["fast_kernel_exact_proxy"],
        "FAST_EXACT_RANK_CORRELATION": result["fast_exact"]["fast_exact_rank_correlation"],
        "STAGE2B_DECISION": result["stage2b_decision"],
        "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
