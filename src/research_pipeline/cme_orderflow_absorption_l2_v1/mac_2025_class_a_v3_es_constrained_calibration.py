"""TRAIN-only ES-constrained Class-A V3 calibration.

This campaign is isolated from the V1 and reduced V2 namespaces.  It uses
the exact Candidate Tape V2 evaluator, keeps Class-B at repository defaults,
and varies only the predeclared Class-A microstructure ranges.
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
from . import mac_2025_class_a_v2_reduced_calibration as v2
from . import mac_2025_class_b_optuna as class_b_v1
from . import mac_2025_class_b_v2_calibration as exact_eval
from . import mac_2025_train_optimization as preparation
from .model import L2ClassBConfig, L2Config, TICK


TRIALS = 5_000
SEED = 20250927
ROOT_NAME = "class-a-v3-es-constrained-calibration"
SUMMARY_NAME = "class-a-v3-es-constrained-calibration-summary.json"
OBJECTIVE_VERSION = "MAC2025_TRAIN_CLASS_A_EXACT_NET_R_V3_ES_CONSTRAINED"
ENTRY_DELAY_MS = 2.0
MIN_REJECTION_UNIT = "ES ticks"

CALIBRATION_FAMILIES = v2.CALIBRATION_FAMILIES
WEIGHT_NAMES = tuple(candidate_tape.WEIGHT_NAMES)

RANGES: dict[str, tuple[float, float, str]] = {
    "min_quality_score": (.45, .65, "float"),
    "min_relevant_aggressive_volume": (40, 120, "int"),
    "min_relevant_execution_count": (2, 5, "int"),
    "min_consume_restore_cycles": (1, 3, "int"),
    "max_through_level_progress_ticks": (2.0, 5.0, "float"),
    "min_rejection_ticks": (.25, 1.0, "float"),
    "false_refill_penalty_weight": (.15, .35, "float"),
    "unexecuted_add_penalty_component_weight": (.45, .60, "float"),
    "rapid_cancel_penalty_component_weight": (.20, .35, "float"),
    "aggressive_volume_saturation": (150, 350, "float"),
    "execution_count_saturation": (5, 12, "float"),
    "restore_cycle_saturation": (2, 5, "float"),
    "restoration_ratio_saturation": (.75, 1.50, "float"),
    "rejection_saturation_ticks": (2, 5, "float"),
    "persistence_depth_saturation": (75, 150, "float"),
    "restoration_latency_saturation_ms": (250, 1000, "float"),
    "multi_level_ofi_saturation": (60, 140, "float"),
}

FREE_NAMES = WEIGHT_NAMES + tuple(RANGES)
NUMERIC_BOUNDS = {name: (0.0, 1.0) for name in WEIGHT_NAMES} | {
    name: (low, high) for name, (low, high, _) in RANGES.items()
}


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
    text = "\n".join(json.dumps(_trial_record(row), sort_keys=True, allow_nan=False)
                     for row in sorted(trials, key=lambda row: row.number)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _default_class_b() -> dict[str, Any]:
    config = L2ClassBConfig()
    return {name: getattr(config, name) for name in exact_eval.SEARCH_SPACE}


def _suggest(trial: optuna.Trial, name: str, low: float, high: float, kind: str) -> float | int:
    if kind == "int":
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, low, high)


def _sample_parameters(trial: optuna.Trial) -> dict[str, Any]:
    raw = [trial.suggest_float(f"raw_{name}", .01, 5.0, log=True) for name in WEIGHT_NAMES]
    total = sum(raw)
    parameters: dict[str, Any] = {
        name: value / total for name, value in zip(WEIGHT_NAMES, raw)
    }
    for name, (low, high, kind) in RANGES.items():
        parameters[name] = _suggest(trial, name, low, high, kind)
    parameters["adverse_progress_penalty_component_weight"] = (
        1.0
        - float(parameters["unexecuted_add_penalty_component_weight"])
        - float(parameters["rapid_cancel_penalty_component_weight"])
    )
    return parameters


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    weights = np.asarray([float(parameters[name]) for name in WEIGHT_NAMES], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("invalid normalized score weights")
    adverse = float(parameters["adverse_progress_penalty_component_weight"])
    if not math.isfinite(adverse) or not 0.0 <= adverse <= 1.0:
        raise ValueError("derived adverse-progress penalty is outside [0, 1]")
    for name, (low, high, kind) in RANGES.items():
        value = float(parameters[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} outside requested range")
        if kind == "int" and int(value) != value:
            raise ValueError(f"{name} is not integral")


def _class_a_config(parameters: Mapping[str, Any]) -> L2Config:
    defaults = L2Config()
    values = {name: getattr(defaults, name) for name in L2Config.__dataclass_fields__}
    for name in WEIGHT_NAMES:
        values[name] = float(parameters[name])
    for name in RANGES:
        values[name] = int(parameters[name]) if RANGES[name][2] == "int" else float(parameters[name])
    values["adverse_progress_penalty_component_weight"] = float(parameters["adverse_progress_penalty_component_weight"])
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
    return {
        "count": len(rows),
        "best": {
            "trial_number": int(ranked[0]["number"]),
            "objective": float(ranked[0]["objective"]),
            "metrics": _metric_summary(ranked[0]["metrics"]),
            "parameters": dict(ranked[0]["parameters"]),
        },
        "parameter_q10_q50_q90": {
            name: _q([float(row["parameters"][name]) for row in rows])
            for name in FREE_NAMES
        },
        "trade_count_q10_q50_q90": _q([float(row["metrics"]["total_trades"]) for row in rows]),
        "active_date_q10_q50_q90": _q([float(row["metrics"]["active_dates"]) for row in rows]),
    }


def _cell(row: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        min(3, max(0, int((float(row["parameters"][name]) - low) / (high - low) * 4)))
        for name, (low, high) in NUMERIC_BOUNDS.items()
    )


def _clusters(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_cell(row), []).append(row)
    result = []
    for key, members in groups.items():
        ranked = sorted(members, key=lambda row: (-float(row["objective"]), int(row["number"])))
        result.append({
            "cell": list(key),
            "count": len(members),
            "best": {
                "trial_number": int(ranked[0]["number"]),
                "objective": float(ranked[0]["objective"]),
                "metrics": _metric_summary(ranked[0]["metrics"]),
                "parameters": dict(ranked[0]["parameters"]),
            },
            "parameter_median": {
                name: float(np.median([float(row["parameters"][name]) for row in members]))
                for name in FREE_NAMES
            },
        })
    return sorted(result, key=lambda row: (-row["count"], -row["best"]["objective"], row["cell"]))


def _boundary_pinned(rows: Sequence[dict[str, Any]]) -> list[str]:
    pinned = []
    for name, (low, high) in NUMERIC_BOUNDS.items():
        values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        near_low = float(np.mean(values <= low + .05 * (high - low)))
        near_high = float(np.mean(values >= high - .05 * (high - low)))
        if max(near_low, near_high) >= .75:
            pinned.append(f"{name}:{'LOWER' if near_low >= near_high else 'UPPER'}")
    return pinned


def _representative(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    clusters = _clusters(rows)
    if not clusters:
        raise preparation.TrainOptimizationError("V3 study has no top-region clusters")
    selected_cell = tuple(clusters[0]["cell"])
    members = [row for row in rows if _cell(row) == selected_cell]
    selected = sorted(members, key=lambda row: (-float(row["objective"]), int(row["number"])))[0]
    flags = ["BROAD_REGION" if clusters[0]["count"] >= 15 else "NARROW_REGION"]
    if int(selected["metrics"]["total_trades"]) < 10:
        flags.append("LOW_SAMPLE")
    if _boundary_pinned(rows):
        flags.append("BOUNDARY_SENSITIVE")
    if float(rows[0]["objective"]) <= 0:
        flags.append("NO_USEFUL_REGION")
    return selected, flags, clusters


def _v1_baseline(master_path: Path, family_id: str) -> dict[str, Any]:
    payload = json.loads(master_path.read_text(encoding="utf-8"))
    for row in payload["families"]:
        if row["family"] == family_id:
            return {"net_r": float(row["best_exact_net_r"]), "trades": int(row["trades"]),
                    "active_dates": int(row["active_dates"])}
    raise ValueError(f"family missing from Class-A master: {family_id}")


def _v2_baseline(output_root: Path, family_id: str) -> dict[str, Any]:
    path = output_root / v2.ROOT_NAME / _slug(family_id) / "class-a-v2-family-summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["selected_representative"]["metrics"]
    return {"net_r": float(metrics["net_r"]), "trades": int(metrics["total_trades"]),
            "active_dates": int(metrics["active_dates"])}


def run_family(family_id: str, *, manifest_path: Path, output_root: Path,
               master_path: Path, trials: int = TRIALS, seed: int = SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    tapes = tuple(bundle.tapes)
    class_b = _default_class_b()
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    study_name = f"MAC2025_TRAIN_CLASS_A_V3_{_slug(family_id).upper()}_TPE_JOURNAL_V1"
    journal_path = root / "class-a-v3-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4),
                                storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        parameters = _sample_parameters(trial)
        _validate_parameters(parameters)
        class_a = _class_a_config(parameters)
        metrics, _ = exact_eval._evaluate_family_full(tapes, bundle.dates, family_id, class_a, class_b)
        if not _finite_metrics(metrics):
            raise ValueError("non-finite exact V3 metrics")
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("normalized_weights", {name: float(parameters[name]) for name in WEIGHT_NAMES})
        trial.set_user_attr("metrics", _metric_summary(metrics))
        trial.set_user_attr("objective_components", {"net_r": float(metrics["net_r"]),
                                                       "objective_version": OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE and
        bool(trial.user_attrs.get("technical_valid", False))
        for trial in study.trials
    )
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_trial_record(trial) for trial in study.trials]
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    failed = sum(row.state == optuna.trial.TrialState.FAIL for row in study.trials)
    pruned = sum(row.state == optuna.trial.TrialState.PRUNED for row in study.trials)
    if len(complete) != int(trials) or failed or pruned:
        raise preparation.TrainOptimizationError(
            f"Class-A V3 study incomplete or invalid: {family_id}: "
            f"complete={len(complete)}/{trials}, failed={failed}, pruned={pruned}"
        )
    _write_trials(root / "class-a-v3-trials.jsonl", study.trials)
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    selected, flags, clusters = _representative(ranked[:500])
    payload = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": OBJECTIVE_VERSION, "study_name": study_name,
        "seed": int(seed), "requested_trials": int(trials), "complete_trials": len(complete),
        "failed_trials": failed, "pruned_trials": pruned,
        "min_rejection_unit": MIN_REJECTION_UNIT,
        "entry_delay_ms": ENTRY_DELAY_MS,
        "class_b_baseline": class_b,
        "v1_class_a_baseline": _v1_baseline(master_path, family_id),
        "v2_reduced_class_a_baseline": _v2_baseline(output_root, family_id),
        "fixed_class_b": True,
        "fixed_class_a_derivation": "adverse_progress_penalty_component_weight = 1 - unexecuted - rapid_cancel",
        "search_space": {
            "raw_weights": {name: {"distribution": "log_float", "low": .01, "high": 5.0} for name in WEIGHT_NAMES},
            **{name: {"distribution": kind, "low": low, "high": high} for name, (low, high, kind) in RANGES.items()},
        },
        "best_search": {"trial_number": int(ranked[0]["number"]), "metrics": _metric_summary(ranked[0]["metrics"]),
                        "parameters": dict(ranked[0]["parameters"])},
        "top_50": _group_summary(ranked[:50]),
        "top_100": _group_summary(ranked[:100]),
        "top_500": _group_summary(ranked[:500]),
        "parameter_cluster_count_top_500": len(clusters),
        "boundary_pinned_parameters_top_500": _boundary_pinned(ranked[:500]),
        "selected_representative": {
            "trial_number": int(selected["number"]), "parameters": dict(selected["parameters"]),
            "metrics": _metric_summary(selected["metrics"]), "robustness_flags": flags,
            "selection_rule": "largest deterministic 4-bin normalized parameter region in top 500; highest exact net R in region",
        },
        "artifacts": {"journal": str(journal_path), "trials": str(root / "class-a-v3-trials.jsonl")},
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_json(root / "class-a-v3-family-summary.json", payload)
    return payload


def run_campaign(*, manifest_path: Path = preparation.DEFAULT_TAPE_ROOT / preparation.TAPE_MANIFEST_NAME,
                 output_root: Path = preparation.DEFAULT_OUTPUT_ROOT,
                 master_path: Path = preparation.DEFAULT_OUTPUT_ROOT / "class-a-final" / "class-a-master-selection.json",
                 trials: int = TRIALS, seed: int = SEED, workers: int = 3) -> dict[str, Any]:
    print(json.dumps({"BATCH_FAMILIES": list(CALIBRATION_FAMILIES), "TRIALS_PER_FAMILY": int(trials),
                      "OBJECTIVE": OBJECTIVE_VERSION, "MIN_REJECTION_UNIT": MIN_REJECTION_UNIT,
                      "TICK_SIZE": TICK, "CLASS_B": "repository defaults", "NAMESPACE": ROOT_NAME}, sort_keys=True))
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
        "class_a_v3_free_parameters": list(FREE_NAMES) + ["adverse_progress_penalty_component_weight (derived)"],
        "min_rejection_unit": MIN_REJECTION_UNIT, "tick_size": TICK, "entry_delay_ms": ENTRY_DELAY_MS,
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
        "CLASS_A_V3_CALIBRATION_TRAIN_COMPLETE": result["status"] == "COMPLETE",
        "FAMILIES": result["selected_families"], "TRIALS_PER_FAMILY": result["requested_trials_per_family"],
        "MIN_REJECTION_UNIT": MIN_REJECTION_UNIT, "ENTRY_DELAY_MS": ENTRY_DELAY_MS,
        "OCTOBER_ACCESSED": False, "OOS_ACCESSED": False, "DATA_DOWNLOADED": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
