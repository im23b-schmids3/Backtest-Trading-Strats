"""TRAIN-only reduced Class-A V2 calibration.

This module is intentionally isolated from the completed Class-A V1 studies.
It evaluates exactly five calibration families against the sealed TRAIN
candidate tapes using the exact candidate-tape evaluator.  Class-B remains at
its repository default configuration and all non-weight Class-A parameters
are fixed to :class:`L2Config` defaults except for ``min_quality_score``.
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

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_class_b_optuna as class_b_v1
from . import mac_2025_class_b_v2_calibration as exact_eval
from . import mac_2025_train_optimization as preparation
from .model import L2ClassBConfig, L2Config


TRIALS = 3_000
SEED = 20250926
ROOT_NAME = "class-a-v2-reduced-calibration"
SUMMARY_NAME = "class-a-v2-reduced-calibration-summary.json"
OBJECTIVE_VERSION = "MAC2025_TRAIN_CLASS_A_EXACT_NET_R_V2_REDUCED"
MIN_QUALITY_SCORE_RANGE = (0.45, 0.65)

CALIBRATION_FAMILIES = (
    "ASIA|ASIA|CURRENT|HIGH",
    "EUROPE|ASIA|CURRENT|LOW",
    "EUROPE|ASIA|CURRENT|VAH",
    "EUROPE|EUROPE|CURRENT|HIGH",
    "EUROPE|RTH|PRIOR|VAL",
)

WEIGHT_NAMES = tuple(candidate_tape.WEIGHT_NAMES)
FREE_NAMES = WEIGHT_NAMES + ("min_quality_score",)
CLASS_B_NAMES = tuple(exact_eval.SEARCH_SPACE)


def _slug(family_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", family_id).strip("-").lower()


def family_output_root(output_root: Path, family_id: str) -> Path:
    return output_root / ROOT_NAME / _slug(family_id)


def _atomic_json(path: Path, payload: Any) -> None:
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


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = dict(trial.user_attrs)
    return {
        "number": int(trial.number),
        "state": trial.state.name,
        "params": dict(trial.params),
        "parameters": attrs.get("parameters", {}),
        "normalized_weights": attrs.get("normalized_weights", {}),
        "objective": float(trial.value) if trial.value is not None else None,
        "metrics": attrs.get("metrics", {}),
        "technical_valid": bool(attrs.get("technical_valid", False)),
    }


def _write_trials(path: Path, trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    lines = [json.dumps(_trial_record(trial), sort_keys=True, allow_nan=False)
             for trial in sorted(trials, key=lambda row: row.number)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _default_class_b() -> dict[str, Any]:
    config = L2ClassBConfig()
    return {name: getattr(config, name) for name in CLASS_B_NAMES}


def _sample_parameters(trial: optuna.Trial) -> dict[str, Any]:
    raw = [trial.suggest_float(f"raw_{name}", 0.01, 5.0, log=True) for name in WEIGHT_NAMES]
    total = sum(raw)
    weights = {name: value / total for name, value in zip(WEIGHT_NAMES, raw)}
    return weights | {
        "min_quality_score": trial.suggest_float(
            "min_quality_score", MIN_QUALITY_SCORE_RANGE[0], MIN_QUALITY_SCORE_RANGE[1]
        )
    }


def _class_a_config(parameters: Mapping[str, Any]) -> L2Config:
    defaults = L2Config()
    values = {name: getattr(defaults, name) for name in L2Config.__dataclass_fields__}
    values.update({name: float(parameters[name]) for name in WEIGHT_NAMES})
    values["min_quality_score"] = float(parameters["min_quality_score"])
    return L2Config(**values)


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    names = ("net_r", "profit_factor", "max_drawdown_r", "total_trades", "active_dates",
             "profitable_date_ratio")
    return all(math.isfinite(float(metrics[name])) for name in names)


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "net_r": float(metrics["net_r"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_r": float(metrics["max_drawdown_r"]),
        "total_trades": int(metrics["total_trades"]),
        "active_dates": int(metrics["active_dates"]),
        "profitable_date_ratio": float(metrics["profitable_date_ratio"]),
        "date_r": dict(metrics.get("date_r", {})),
        "winners": int(metrics.get("winners", 0)),
        "losers": int(metrics.get("losers", 0)),
    }


def _q(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"q10": float(np.quantile(array, .10)),
            "q50": float(np.quantile(array, .50)),
            "q90": float(np.quantile(array, .90))}


def _group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (-float(row["objective"]), int(row["number"])))
    distributions = {}
    for name in FREE_NAMES:
        values = [float(row["parameters"][name]) for row in rows]
        distributions[name] = _q(values)
    return {
        "count": len(rows),
        "best": {
            "trial_number": int(ranked[0]["number"]),
            "objective": float(ranked[0]["objective"]),
            "metrics": _metric_summary(ranked[0]["metrics"]),
            "parameters": dict(ranked[0]["parameters"]),
        },
        "parameter_q10_q50_q90": distributions,
        "trade_count_q10_q50_q90": _q([float(row["metrics"]["total_trades"]) for row in rows]),
        "active_date_q10_q50_q90": _q([float(row["metrics"]["active_dates"]) for row in rows]),
    }


def _clusters(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(int(round(float(row["parameters"][name]) * 10.0)) for name in FREE_NAMES)
        groups.setdefault(key, []).append(row)
    result = []
    for key, members in groups.items():
        ranked = sorted(members, key=lambda row: (-float(row["objective"]), int(row["number"])))
        result.append({
            "key": list(key),
            "count": len(members),
            "median_objective": float(np.median([float(row["objective"]) for row in members])),
            "best": {
                "trial_number": int(ranked[0]["number"]),
                "objective": float(ranked[0]["objective"]),
                "metrics": _metric_summary(ranked[0]["metrics"]),
                "parameters": dict(ranked[0]["parameters"]),
            },
        })
    return sorted(result, key=lambda row: (-row["count"], -row["best"]["objective"]))


def _representative(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Select the best member of the largest top-300 normalized region."""
    clusters = _clusters(rows)
    if not clusters:
        raise RuntimeError("no rows available for representative selection")
    largest = clusters[0]
    key = tuple(largest["key"])
    members = [row for row in rows if tuple(int(round(float(row["parameters"][name]) * 10.0)) for name in FREE_NAMES) == key]
    selected = sorted(members, key=lambda row: (-float(row["objective"]), int(row["number"])))[0]
    flags = ["LARGEST_TOP_300_PARAMETER_REGION"]
    if largest["count"] >= 10:
        flags.append("BROAD_REGION")
    else:
        flags.append("NARROW_REGION")
    if any(float(selected["parameters"][name]) <= MIN_QUALITY_SCORE_RANGE[0] + .05 * (MIN_QUALITY_SCORE_RANGE[1] - MIN_QUALITY_SCORE_RANGE[0]) or
           float(selected["parameters"][name]) >= MIN_QUALITY_SCORE_RANGE[1] - .05 * (MIN_QUALITY_SCORE_RANGE[1] - MIN_QUALITY_SCORE_RANGE[0])
           for name in ("min_quality_score",)):
        flags.append("MIN_QUALITY_BOUNDARY_SENSITIVE")
    return selected, flags


def _class_a_v1_baseline(master_path: Path, family_id: str) -> dict[str, Any]:
    payload = json.loads(master_path.read_text(encoding="utf-8"))
    for row in payload["families"]:
        if row["family"] == family_id:
            return {
                "net_r": float(row["best_exact_net_r"]),
                "trades": int(row["trades"]),
                "active_dates": int(row["active_dates"]),
            }
    raise ValueError(f"family missing from Class-A master: {family_id}")


def run_family(family_id: str, *, manifest_path: Path, output_root: Path,
               master_path: Path, trials: int = TRIALS, seed: int = SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    tapes = tuple(bundle.tapes)
    class_b = _default_class_b()
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    study_name = f"MAC2025_TRAIN_CLASS_A_V2_{_slug(family_id).upper()}_TPE_JOURNAL_V1"
    journal_path = root / "class-a-v2-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    sampler = optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4)
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=sampler, storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        parameters = _sample_parameters(trial)
        weights = np.asarray([float(parameters[name]) for name in WEIGHT_NAMES], dtype=float)
        technical_valid = bool(np.all(np.isfinite(weights)) and np.all(weights > 0.0) and
                               math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12) and
                               math.isfinite(float(parameters["min_quality_score"])))
        if not technical_valid:
            trial.set_user_attr("technical_valid", False)
            raise optuna.TrialPruned("invalid normalized Class-A parameters")
        class_a = _class_a_config(parameters)
        metrics, _ = exact_eval._evaluate_family_full(tapes, bundle.dates, family_id, class_a, class_b)
        if not _finite_metrics(metrics):
            trial.set_user_attr("technical_valid", False)
            raise optuna.TrialPruned("non-finite exact metrics")
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("normalized_weights", {name: float(parameters[name]) for name in WEIGHT_NAMES})
        trial.set_user_attr("metrics", _metric_summary(metrics))
        trial.set_user_attr("objective_components", {"net_r": float(metrics["net_r"]),
                                                       "objective_version": OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(trial.state == optuna.trial.TrialState.COMPLETE and
                          bool(trial.user_attrs.get("technical_valid", False)) for trial in study.trials)
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_trial_record(trial) for trial in study.trials]
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    failed = sum(row.state == optuna.trial.TrialState.FAIL for row in study.trials)
    pruned = sum(row.state == optuna.trial.TrialState.PRUNED for row in study.trials)
    if len(complete) != int(trials) or failed or pruned:
        raise preparation.TrainOptimizationError(
            f"Class-A V2 study incomplete or invalid: {family_id}: "
            f"complete={len(complete)}/{trials}, failed={failed}, pruned={pruned}"
        )
    _write_trials(root / "class-a-v2-trials.jsonl", study.trials)
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    representative, flags = _representative(ranked[:300])
    selected_metrics = representative["metrics"]
    payload = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": OBJECTIVE_VERSION, "study_name": study_name,
        "seed": int(seed), "requested_trials": int(trials), "complete_trials": len(complete),
        "failed_trials": failed, "pruned_trials": pruned,
        "class_b_baseline": class_b,
        "v1_class_a_baseline": _class_a_v1_baseline(master_path, family_id),
        "fixed_class_a_parameters": {
            name: getattr(L2Config(), name) for name in L2Config.__dataclass_fields__ if name not in FREE_NAMES
        },
        "search_space": {
            "raw_weights": {name: {"distribution": "log_float", "low": .01, "high": 5.0} for name in WEIGHT_NAMES},
            "min_quality_score": {"distribution": "float", "low": .45, "high": .65},
        },
        "best_search": {
            "trial_number": int(ranked[0]["number"]), "metrics": _metric_summary(ranked[0]["metrics"]),
            "parameters": dict(ranked[0]["parameters"]),
        },
        "top_50": _group_summary(ranked[:50]),
        "top_100": _group_summary(ranked[:100]),
        "top_300": _group_summary(ranked[:300]),
        "parameter_cluster_count_top_300": len(_clusters(ranked[:300])),
        "selected_representative": {
            "trial_number": int(representative["number"]),
            "parameters": dict(representative["parameters"]),
            "metrics": _metric_summary(selected_metrics),
            "robustness_flags": flags,
            "selection_rule": "largest deterministic normalized parameter region in top 300; highest exact net R in region",
        },
        "artifacts": {"journal": str(journal_path), "trials": str(root / "class-a-v2-trials.jsonl")},
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_json(root / "class-a-v2-family-summary.json", payload)
    return payload


def run_campaign(*, manifest_path: Path = preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME,
                 output_root: Path = preparation.DEFAULT_OUTPUT_ROOT,
                 master_path: Path = preparation.DEFAULT_OUTPUT_ROOT / "class-a-final" / "class-a-master-selection.json",
                 trials: int = TRIALS, seed: int = SEED, workers: int = 3) -> dict[str, Any]:
    print(json.dumps({"BATCH_FAMILIES": list(CALIBRATION_FAMILIES),
                      "TRIALS_PER_FAMILY": int(trials), "OBJECTIVE": OBJECTIVE_VERSION,
                      "MIN_QUALITY_SCORE_RANGE": list(MIN_QUALITY_SCORE_RANGE),
                      "CLASS_B": "repository defaults", "NAMESPACE": ROOT_NAME}, sort_keys=True))
    jobs = [(family, manifest_path, output_root, master_path, int(trials), int(seed) + index)
            for index, family in enumerate(CALIBRATION_FAMILIES)]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = [executor.submit(run_family, family, manifest_path=manifest, output_root=root,
                                    master_path=master, trials=count, seed=job_seed)
                   for family, manifest, root, master, count, job_seed in jobs]
        for future in futures:
            results.append(future.result())
    results.sort(key=lambda row: CALIBRATION_FAMILIES.index(row["family_id"]))
    payload = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": OBJECTIVE_VERSION,
        "namespace": ROOT_NAME, "selected_families": list(CALIBRATION_FAMILIES),
        "requested_trials_per_family": int(trials), "studies": results,
        "class_a_v2_free_parameters": list(FREE_NAMES),
        "class_a_v2_effective_degrees_of_freedom": 5,
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_json(output_root / SUMMARY_NAME, payload)
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME)
    parser.add_argument("--output-root", type=Path, default=preparation.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--master", type=Path, default=preparation.DEFAULT_OUTPUT_ROOT / "class-a-final" / "class-a-master-selection.json")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_campaign(manifest_path=args.manifest, output_root=args.output_root,
                          master_path=args.master, trials=args.trials, seed=args.seed,
                          workers=args.workers)
    print(json.dumps({
        "CLASS_A_V2_CALIBRATION_TRAIN_COMPLETE": result["status"] == "COMPLETE",
        "FAMILIES": result["selected_families"],
        "TRIALS_PER_FAMILY": result["requested_trials_per_family"],
        "OCTOBER_ACCESSED": False, "OOS_ACCESSED": False, "DATA_DOWNLOADED": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
