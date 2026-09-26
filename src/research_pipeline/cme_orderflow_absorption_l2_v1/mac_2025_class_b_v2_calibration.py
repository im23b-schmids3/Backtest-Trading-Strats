"""TRAIN-only Class-B V2 calibration over the sealed MAC 2025 candidate tapes.

This is intentionally a separate campaign namespace.  It never opens raw DBN,
Validation, or OOS data and never mutates the completed Class-B V1 studies.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
from . import mac_2025_class_b_optuna as v1
from . import mac_2025_train_optimization as preparation
from .model import ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS, L2ClassBConfig


TRIALS = 3_000
SEED = 20250925
ROOT_NAME = "class-b-v2-calibration-rr1p5-3-sl8-window45"
SUMMARY_NAME = "class-b-v2-calibration-summary.json"
OBJECTIVE_VERSION = "MAC2025_TRAIN_CLASS_B_EXACT_NET_R_V2_CALIBRATION"
ENTRY_DELAY_MS = 2.0

SEARCH_SPACE: dict[str, Any] = {
    "min_confirmation_seconds": {"distribution": "float", "low": 0.0, "high": 15.0},
    "max_confirmation_seconds": {"distribution": "dependent_float", "low": "min_confirmation_seconds + 0.001", "high": 45.0},
    "favorable_confirmation_ticks": {"distribution": "float", "low": 0.25, "high": 12.0},
    "confirmation_execution_count": {"distribution": "int", "low": 1, "high": 12},
    "confirmation_volume_threshold": {"distribution": "int", "low": 0, "high": 500},
    "stop_ticks": {"distribution": "int", "low": 0, "high": 8},
    "target_r": {"distribution": "float", "low": 1.5, "high": 3.0},
}
NUMERIC_BOUNDS = {
    "min_confirmation_seconds": (0.0, 15.0), "max_confirmation_seconds": (0.0, 45.0),
    "favorable_confirmation_ticks": (0.25, 12.0), "confirmation_execution_count": (1.0, 12.0),
    "confirmation_volume_threshold": (0.0, 500.0), "stop_ticks": (0.0, 8.0),
    "target_r": (1.5, 3.0),
}
CALIBRATION_FAMILIES = (
    "ASIA|ASIA|CURRENT|HIGH",
    "EUROPE|ASIA|CURRENT|LOW",
    "EUROPE|ASIA|CURRENT|VAH",
    "EUROPE|EUROPE|CURRENT|HIGH",
    "EUROPE|RTH|PRIOR|VAL",
)


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


def _atomic_text(path: Path, text: str) -> None:
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


def _slug(family_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", family_id).strip("-").lower()


def family_output_root(output_root: Path, family_id: str) -> Path:
    return output_root / ROOT_NAME / _slug(family_id)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sample_class_b(trial: optuna.Trial) -> dict[str, Any]:
    minimum = trial.suggest_float("min_confirmation_seconds", 0.0, 15.0)
    maximum = trial.suggest_float("max_confirmation_seconds", minimum + 0.001, 45.0)
    return {
        "min_confirmation_seconds": minimum,
        "max_confirmation_seconds": maximum,
        "favorable_confirmation_ticks": trial.suggest_float("favorable_confirmation_ticks", 0.25, 12.0),
        "confirmation_execution_count": trial.suggest_int("confirmation_execution_count", 1, 12),
        "confirmation_volume_threshold": trial.suggest_int("confirmation_volume_threshold", 0, 500),
        "stop_ticks": trial.suggest_int("stop_ticks", 0, 8),
        "target_r": trial.suggest_float("target_r", 1.5, 3.0),
    }


def _validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    validated = L2ClassBConfig(**dict(parameters))
    result = {name: getattr(validated, name) for name in SEARCH_SPACE}
    if not 1.5 <= float(result["target_r"]) <= 3.0:
        raise ValueError("V2 target_r outside [1.5, 3.0]")
    if not 0 <= int(result["stop_ticks"]) <= 8:
        raise ValueError("V2 stop_ticks outside [0, 8]")
    if float(result["max_confirmation_seconds"]) > 45.0:
        raise ValueError("V2 max confirmation exceeds 45 seconds")
    return result


def _evaluate_family_full(tapes: Sequence[candidate_tape.CandidateTape], dates: Sequence[str], family_id: str,
                          class_a: Any, class_b: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameters = v1._tape_parameters(class_a, class_b)
    all_trades: list[dict[str, Any]] = []
    qualified_count = 0
    for day, tape in zip(dates, tapes):
        family_tape = candidate_tape.CandidateTape(
            tape.metadata,
            tuple(row for row in tape.candidates if str(row.get("level", row.get("family_id"))) == family_id),
            tape.events,
        )
        result = candidate_tape.evaluate_candidate_tape(family_tape, parameters, config=class_a)
        qualified_count += int(result["qualified_count"])
        rows_by_id = {str(row.get("interaction_id")): row for row in family_tape.candidates}
        for trade in result["trades"]:
            source = rows_by_id.get(str(trade.get("interaction_id")), {})
            all_trades.append({**trade, "date": day, "trading_session": source.get("trading_session"), "family_id": family_id})
    metrics = v1._metrics(all_trades, dates)
    metrics["qualified_count"] = qualified_count
    metrics["winners"] = sum(float(row.get("r_multiple", row.get("r", 0.0))) > 0 for row in all_trades)
    metrics["losers"] = sum(float(row.get("r_multiple", row.get("r", 0.0))) < 0 for row in all_trades)
    return metrics, all_trades


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = dict(trial.user_attrs)
    return {
        "number": int(trial.number), "state": trial.state.name, "params": dict(trial.params),
        "parameters": attrs.get("parameters", {}), "objective": float(trial.value) if trial.value is not None else None,
        "metrics": attrs.get("metrics", {}), "technical_valid": bool(attrs.get("technical_valid", False)),
    }


def _summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"trial_number": int(row["number"]), "objective": float(row["objective"]),
            "parameters": dict(row["parameters"]), **dict(row["metrics"])}


def _bands(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in SEARCH_SPACE:
        values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        result[name] = {"min": float(values.min()), "q10": float(np.quantile(values, .10)),
                        "median": float(np.quantile(values, .50)), "q90": float(np.quantile(values, .90)),
                        "max": float(values.max())}
    return result


def _clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(min(3, max(0, int((float(row["parameters"][name]) - low) / (high - low) * 4)))
                    for name, (low, high) in NUMERIC_BOUNDS.items())
        groups.setdefault(key, []).append(row)
    result = []
    for key, members in groups.items():
        best = max(members, key=lambda row: (float(row["objective"]), -int(row["number"])))
        result.append({"cell": list(key), "count": len(members), "best": _summary_row(best),
                       "parameter_median": {name: float(np.median([float(x["parameters"][name]) for x in members]))
                                             for name in SEARCH_SPACE}})
    return sorted(result, key=lambda row: (-row["count"], -row["best"]["objective"], row["cell"]))


def _representative(ranked: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    top = list(ranked[:300])
    clusters = _clusters(top)
    if not clusters:
        raise preparation.TrainOptimizationError("V2 study has no top-region clusters")
    selected_cell = tuple(clusters[0]["cell"])
    members = [row for row in top if tuple(
        min(3, max(0, int((float(row["parameters"][name]) - low) / (high - low) * 4)))
        for name, (low, high) in NUMERIC_BOUNDS.items()) == selected_cell]
    chosen = max(members, key=lambda row: (float(row["objective"]), -int(row["number"])))
    return chosen, clusters


def _boundary_pinned(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    pinned = []
    for name, (low, high) in NUMERIC_BOUNDS.items():
        values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        near_low = float(np.mean(values <= low + .05 * (high - low)))
        near_high = float(np.mean(values >= high - .05 * (high - low)))
        if max(near_low, near_high) >= .75:
            pinned.append(f"{name}:{'LOWER' if near_low >= near_high else 'UPPER'}")
    return pinned


def _study_name(family_id: str) -> str:
    return f"MAC2025_TRAIN_CLASS_B_V2_CALIBRATION_{_slug(family_id).upper()}_TPE_JOURNAL_V2"


def run_family(family_id: str, *, manifest_path: Path, master_path: Path, output_root: Path,
               trials: int = TRIALS, seed: int = SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    master = v1._master_rows(master_path)
    class_a = v1._class_a_config(master[family_id])
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "class-b-v2-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    study = optuna.create_study(direction="maximize", study_name=_study_name(family_id),
                                sampler=optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4),
                                storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        try:
            parameters = _validate_parameters(_sample_class_b(trial))
            metrics, _ = _evaluate_family_full(bundle.tapes, bundle.dates, family_id, class_a, parameters)
            if not all(math.isfinite(float(metrics[name])) for name in ("net_r", "profit_factor", "max_drawdown_r", "profitable_date_ratio")):
                raise ValueError("non-finite V2 metric")
        except Exception:
            trial.set_user_attr("technical_valid", False)
            raise
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("objective_components", {"net_r": metrics["net_r"], "objective_version": OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_trial_record(t) for t in study.trials]
    complete = [r for r in records if r["state"] == "COMPLETE" and r["technical_valid"]]
    if len(complete) != int(trials):
        raise preparation.TrainOptimizationError(f"V2 study incomplete: {family_id}: {len(complete)}/{trials}")
    failed = sum(r["state"] == "FAIL" for r in records)
    if failed:
        raise preparation.TrainOptimizationError(f"V2 study has failed trials: {family_id}: {failed}")
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    top_regions = {}
    for count in (50, 100, 300):
        top = ranked[:count]
        top_regions[str(count)] = {"count": len(top), "best": _summary_row(top[0]),
                                   "parameter_bands": _bands(top), "cluster_count": len(_clusters(top)),
                                   "boundary_pinned_params": _boundary_pinned(top), "clusters": _clusters(top)[:20]}
    selected, clusters = _representative(ranked)
    selected_metrics, selected_trades = _evaluate_family_full(bundle.tapes, bundle.dates, family_id, class_a, selected["parameters"])
    baseline_params = {name: getattr(L2ClassBConfig(), name) for name in SEARCH_SPACE}
    baseline_metrics, _ = _evaluate_family_full(bundle.tapes, bundle.dates, family_id, class_a, baseline_params)
    v1_summary_path = output_root / v1.ROOT_NAME / _slug(family_id) / "class-b-family-summary.json"
    v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
    v1_best = v1_summary["best_search"]
    top300 = ranked[:300]
    largest = clusters[0]
    flags = []
    if largest["count"] >= 15 and float(np.median([float(r["objective"]) for r in top300[:50]])) > 0:
        flags.append("BROAD_STABLE_REGION")
    if largest["count"] <= 5:
        flags.append("NARROW_REGION")
    if selected_metrics["total_trades"] < 10 or selected_metrics["active_dates"] < 5:
        flags.append("LOW_SAMPLE")
    pinned = _boundary_pinned(ranked[:50])
    if pinned:
        flags.append("BOUNDARY_SENSITIVE")
    if float(selected_metrics["net_r"]) <= 0:
        flags.append("NO_USEFUL_REGION")
    summary = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": OBJECTIVE_VERSION, "study_name": _study_name(family_id), "seed": int(seed),
        "requested_trials": int(trials), "complete_trials": len(complete), "failed_trials": failed,
        "calibration_namespace": ROOT_NAME, "search_space": SEARCH_SPACE,
        "class_a_frozen": True, "class_a_frozen_config": master[family_id]["class_a_frozen_config"],
        "class_a_frozen_config_hash": _sha256_json(master[family_id]["class_a_frozen_config"]),
        "entry_delay_ms": ENTRY_DELAY_MS, "entry_delay_frozen": True,
        "tape_version": candidate_tape.TAPE_VERSION, "dates": list(bundle.dates),
        "best_search": _summary_row(ranked[0]), "top_regions": top_regions,
        "selected_v2": {"trial_number": selected["number"], "parameters": selected["parameters"], "metrics": selected_metrics,
                        "selection_rule": "largest deterministic top-300 normalized parameter cell; highest objective within that cell",
                        "cluster": largest, "robustness_flags": flags, "boundary_pinned_params": pinned},
        "class_a_baseline": baseline_metrics,
        "v1_comparison": {"net_r": v1_best["net_r"], "trades": v1_best["total_trades"],
                          "target_r": v1_best["parameters"]["target_r"], "stop_ticks": v1_best["parameters"]["stop_ticks"]},
        "delta_v2_vs_v1": {"net_r": selected_metrics["net_r"] - v1_best["net_r"],
                           "trades": selected_metrics["total_trades"] - v1_best["total_trades"]},
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
        "artifacts": {"journal": str(journal_path), "trials": str(root / "class-b-v2-trials.jsonl")},
    }
    _atomic_text(root / "class-b-v2-trials.jsonl", "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in records) + "\n")
    _atomic_json(root / "class-b-v2-family-summary.json", summary)
    return summary


def run_batch(*, manifest_path: Path, master_path: Path, output_root: Path,
              trials: int = TRIALS, seed: int = SEED, workers: int = 3) -> dict[str, Any]:
    selected = list(CALIBRATION_FAMILIES)
    master = v1._master_rows(master_path)
    if any(f not in master for f in selected):
        raise preparation.TrainOptimizationError("calibration family missing from Class-A master")
    print(json.dumps({"BATCH_FAMILIES": selected, "CLASS_B_V2_SEARCH_SPACE": SEARCH_SPACE,
                      "TRIALS_PER_FAMILY": int(trials), "SEED": int(seed),
                      "NAMESPACE": ROOT_NAME}, sort_keys=True), flush=True)
    jobs = [(family, manifest_path, master_path, output_root, int(trials), int(seed) + i) for i, family in enumerate(selected)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = [executor.submit(run_family, family, manifest_path=manifest, master_path=master,
                                   output_root=root, trials=count, seed=job_seed)
                   for family, manifest, master, root, count, job_seed in jobs]
        for future in futures:
            results.append(future.result())
    results.sort(key=lambda row: selected.index(row["family_id"]))
    payload = {"status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": OBJECTIVE_VERSION,
               "selected_families": selected, "requested_trials_per_family": int(trials),
               "studies": results, "search_space": SEARCH_SPACE, "entry_delay_ms": ENTRY_DELAY_MS,
               "entry_delay_frozen": True, "class_a_frozen": True, "tape_version": candidate_tape.TAPE_VERSION,
               "calibration_namespace": ROOT_NAME, "validation_accessed": False, "oos_accessed": False,
               "dbn_accessed": False}
    _atomic_json(output_root / SUMMARY_NAME, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE")
    parser.add_argument("--manifest", type=Path, default=default_root / "candidate-tapes/train-tape-manifest.json")
    parser.add_argument("--master", type=Path, default=default_root / "train-optimization/class-a-final/class-a-master-selection.json")
    parser.add_argument("--output-root", type=Path, default=default_root / "train-optimization")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    payload = run_batch(manifest_path=args.manifest, master_path=args.master, output_root=args.output_root,
                        trials=args.trials, seed=args.seed, workers=args.workers)
    print(json.dumps({"CLASS_B_V2_CALIBRATION_TRAIN_COMPLETE": payload["status"] == "COMPLETE",
                      "CALIBRATION_FAMILIES": payload["selected_families"], "TRIALS_PER_FAMILY": payload["requested_trials_per_family"],
                      "ENTRY_DELAY_MS": ENTRY_DELAY_MS, "ENTRY_DELAY_FROZEN": True, "CLASS_A_FROZEN": True,
                      "TARGET_R_RANGE": [1.5, 3.0], "STOP_TICKS_MAX": 8, "MAX_CONFIRMATION_SECONDS_MAX": 45.0,
                      "TAPE_VERSION": candidate_tape.TAPE_VERSION, "VALIDATION_ACCESSED": False,
                      "OOS_ACCESSED": False, "DBN_ACCESSED": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
