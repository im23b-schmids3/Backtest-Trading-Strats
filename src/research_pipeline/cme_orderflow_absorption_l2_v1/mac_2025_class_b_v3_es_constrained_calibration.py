"""TRAIN-only Class-B V3 calibration on frozen Class-A V3 configurations.

This campaign is deliberately isolated from all prior Class-B namespaces.  It
loads only the sealed TRAIN candidate tapes, freezes the authoritative Class-A
V3 representative for each requested family, and varies the semantically
validated Class-B confirmation/exit parameters.
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
from . import mac_2025_class_a_v3_es_constrained_calibration as class_a_v3
from . import mac_2025_class_b_optuna as class_b_v1
from . import mac_2025_class_b_v2_calibration as exact_eval
from . import mac_2025_train_optimization as preparation
from .model import L2ClassBConfig


TRIALS = 5_000
SEED = 20250928
ROOT_NAME = "class-b-v3-es-constrained-calibration"
SUMMARY_NAME = "class-b-v3-es-constrained-calibration-summary.json"
OBJECTIVE_VERSION = "MAC2025_TRAIN_CLASS_B_EXACT_NET_R_V3_ES_CONSTRAINED"
ENTRY_DELAY_MS = 2.0

CALIBRATION_FAMILIES = (
    "ASIA|ASIA|CURRENT|HIGH",
    "EUROPE|ASIA|CURRENT|LOW",
    "EUROPE|ASIA|CURRENT|VAH",
    "EUROPE|EUROPE|CURRENT|HIGH",
    "EUROPE|RTH|PRIOR|VAL",
)

TARGET_CHOICES = (1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00)
SEARCH_SPACE: dict[str, Any] = {
    "min_confirmation_seconds": {"distribution": "int", "low": 2, "high": 8},
    "max_confirmation_seconds": {"distribution": "int", "low": 10, "high": 30},
    "favorable_confirmation_ticks": {"distribution": "int", "low": 1, "high": 5},
    "confirmation_execution_count": {"distribution": "int", "low": 2, "high": 6},
    "confirmation_volume_threshold": {"distribution": "int", "low": 25, "high": 125, "unit": "ES contracts"},
    "stop_ticks": {"distribution": "int", "low": 1, "high": 4},
    "target_r": {"distribution": "categorical", "choices": list(TARGET_CHOICES)},
}
NUMERIC_BOUNDS = {
    "min_confirmation_seconds": (2.0, 8.0),
    "max_confirmation_seconds": (10.0, 30.0),
    "favorable_confirmation_ticks": (1.0, 5.0),
    "confirmation_execution_count": (2.0, 6.0),
    "confirmation_volume_threshold": (25.0, 125.0),
    "stop_ticks": (1.0, 4.0),
    "target_r": (1.50, 3.00),
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


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_v3_class_a(output_root: Path, family_id: str) -> tuple[Any, dict[str, Any], str]:
    path = output_root / class_a_v3.ROOT_NAME / _slug(family_id) / "class-a-v3-family-summary.json"
    if not path.is_file():
        raise preparation.TrainOptimizationError(f"missing authoritative Class-A V3 summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("scope") != "TRAIN_ONLY":
        raise preparation.TrainOptimizationError(f"Class-A V3 summary is not complete TRAIN-only: {path}")
    if int(payload.get("complete_trials", 0)) != int(payload.get("requested_trials", -1)):
        raise preparation.TrainOptimizationError(f"Class-A V3 trial count mismatch: {path}")
    params = dict(payload.get("selected_representative", {}).get("parameters", {}))
    if not params:
        raise preparation.TrainOptimizationError(f"Class-A V3 representative missing: {path}")
    config = class_a_v3._class_a_config(params)
    return config, params, _sha256_json(params)


def _load_v2_parameters(output_root: Path, family_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output_root / "class-b-v2-calibration-rr1p5-3-sl8-window45" / _slug(family_id) / "class-b-v2-family-summary.json"
    if not path.is_file():
        raise preparation.TrainOptimizationError(f"missing prior Class-B V2 summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise preparation.TrainOptimizationError(f"Class-B V2 summary is not complete: {path}")
    selected = payload.get("selected_v2", {})
    parameters = dict(selected.get("parameters", {}))
    if not parameters:
        raise preparation.TrainOptimizationError(f"Class-B V2 representative missing: {path}")
    return parameters, dict(selected.get("metrics", {}))


def _sample_class_b(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "min_confirmation_seconds": trial.suggest_int("min_confirmation_seconds", 2, 8),
        "max_confirmation_seconds": trial.suggest_int("max_confirmation_seconds", 10, 30),
        "favorable_confirmation_ticks": trial.suggest_int("favorable_confirmation_ticks", 1, 5),
        "confirmation_execution_count": trial.suggest_int("confirmation_execution_count", 2, 6),
        "confirmation_volume_threshold": trial.suggest_int("confirmation_volume_threshold", 25, 125),
        "stop_ticks": trial.suggest_int("stop_ticks", 1, 4),
        "target_r": trial.suggest_categorical("target_r", list(TARGET_CHOICES)),
    }


def _validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(parameters)
    if int(values["max_confirmation_seconds"]) <= int(values["min_confirmation_seconds"]):
        raise ValueError("max_confirmation_seconds must exceed min_confirmation_seconds")
    config = L2ClassBConfig(**values)
    result = {name: getattr(config, name) for name in SEARCH_SPACE}
    if result["target_r"] not in TARGET_CHOICES:
        raise ValueError("target_r is outside the discrete V3 choices")
    if not 1 <= int(result["stop_ticks"]) <= 4:
        raise ValueError("stop_ticks is outside the V3 range")
    return result


def _default_class_b() -> dict[str, Any]:
    config = L2ClassBConfig()
    return {name: getattr(config, name) for name in SEARCH_SPACE}


def _metrics(trades: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    snapshot = preparation._snapshot_metrics(trades=trades, dates=dates, sessions=("ASIA", "EUROPE", "NY"))
    return {
        "net_r": float(snapshot["net_r"]),
        "profit_factor": float(snapshot["profit_factor_capped"]),
        "max_drawdown_r": float(snapshot["max_drawdown_r"]),
        "total_trades": int(snapshot["total_trades"]),
        "active_dates": int(snapshot["active_dates"]),
        "profitable_date_ratio": float(snapshot["profitable_date_ratio"]),
        "date_r": dict(snapshot.get("date_r", {})),
        "winners": int(sum(float(row.get("r_multiple", row.get("r", 0.0))) > 0 for row in trades)),
        "losers": int(sum(float(row.get("r_multiple", row.get("r", 0.0))) < 0 for row in trades)),
    }


def _evaluate(tapes: Sequence[candidate_tape.CandidateTape], dates: Sequence[str], family_id: str,
              class_a: Any, class_b: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return exact_eval._evaluate_family_full(tapes, dates, family_id, class_a, class_b)


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    return all(math.isfinite(float(metrics[name])) for name in
               ("net_r", "profit_factor", "max_drawdown_r", "total_trades", "active_dates", "profitable_date_ratio"))


def _record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = dict(trial.user_attrs)
    return {
        "number": int(trial.number), "state": trial.state.name, "params": dict(trial.params),
        "parameters": attrs.get("parameters", {}),
        "objective": float(trial.value) if trial.value is not None else None,
        "metrics": attrs.get("metrics", {}), "technical_valid": bool(attrs.get("technical_valid", False)),
    }


def _summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"trial_number": int(row["number"]), "objective": float(row["objective"]),
            "parameters": dict(row["parameters"]), **dict(row["metrics"])}


def _quantiles(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, float]:
    values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
    return {"q10": float(np.quantile(values, .10)), "q50": float(np.quantile(values, .50)),
            "q90": float(np.quantile(values, .90))}


def _cell(row: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(min(3, max(0, int((float(row["parameters"][name]) - low) / (high - low) * 4)))
                 for name, (low, high) in NUMERIC_BOUNDS.items())


def _clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_cell(row), []).append(row)
    result = []
    for cell, members in groups.items():
        best = max(members, key=lambda row: (float(row["objective"]), -int(row["number"])))
        result.append({
            "cell": list(cell), "count": len(members), "best": _summary_row(best),
            "parameter_median": {name: float(np.median([float(x["parameters"][name]) for x in members]))
                                  for name in SEARCH_SPACE},
        })
    return sorted(result, key=lambda row: (-row["count"], -row["best"]["objective"], row["cell"]))


def _boundary_pinned(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result = []
    for name, (low, high) in NUMERIC_BOUNDS.items():
        values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        near_low = float(np.mean(values <= low + .05 * (high - low)))
        near_high = float(np.mean(values >= high - .05 * (high - low)))
        if max(near_low, near_high) >= .75:
            result.append(f"{name}:{'LOWER' if near_low >= near_high else 'UPPER'}")
    return result


def _top_summary(ranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top = list(ranked)
    return {
        "count": len(top), "best": _summary_row(top[0]),
        "parameter_bands": {name: _quantiles(top, name) for name in SEARCH_SPACE},
        "trade_count_q10_q50_q90": {key: value for key, value in zip(
            ("q10", "q50", "q90"), np.quantile([float(row["metrics"]["total_trades"]) for row in top], [.1, .5, .9]))},
        "active_date_q10_q50_q90": {key: value for key, value in zip(
            ("q10", "q50", "q90"), np.quantile([float(row["metrics"]["active_dates"]) for row in top], [.1, .5, .9]))},
        "cluster_count": len(_clusters(top)), "boundary_pinned_params": _boundary_pinned(top),
        "clusters": _clusters(top)[:20],
    }


def _study_name(family_id: str) -> str:
    return f"MAC2025_TRAIN_CLASS_B_V3_{_slug(family_id).upper()}_TPE_JOURNAL_V1"


def run_family(family_id: str, *, manifest_path: Path, output_root: Path,
               trials: int = TRIALS, seed: int = SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    class_a, class_a_params, class_a_hash = _load_v3_class_a(output_root, family_id)
    v2_params, v2_artifact_metrics = _load_v2_parameters(output_root, family_id)
    print(json.dumps({"FAMILY": family_id, "CLASS_A_V3_CONFIG_HASH": class_a_hash,
                      "CLASS_A_V3_CONFIG": class_a_params, "ENTRY_DELAY_MS": ENTRY_DELAY_MS},
                     sort_keys=True), flush=True)
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "class-b-v3-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    study = optuna.create_study(direction="maximize", study_name=_study_name(family_id),
                                sampler=optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4),
                                storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        parameters = _validate_parameters(_sample_class_b(trial))
        metrics, _ = _evaluate(bundle.tapes, bundle.dates, family_id, class_a, parameters)
        if not _finite_metrics(metrics):
            raise ValueError("non-finite exact V3 Class-B metrics")
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("objective_components", {"net_r": metrics["net_r"], "objective_version": OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(t.state == optuna.trial.TrialState.COMPLETE and bool(t.user_attrs.get("technical_valid", False))
                          for t in study.trials)
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_record(t) for t in study.trials]
    complete = [row for row in records if row["state"] == "COMPLETE" and row["technical_valid"]]
    failed = sum(row["state"] == "FAIL" for row in records)
    pruned = sum(row["state"] == "PRUNED" for row in records)
    if len(complete) != int(trials) or failed or pruned:
        raise preparation.TrainOptimizationError(
            f"Class-B V3 study incomplete or invalid: {family_id}: complete={len(complete)}/{trials}, failed={failed}, pruned={pruned}")
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    top_regions = {str(count): _top_summary(ranked[:count]) for count in (50, 100, 500)}
    top500 = ranked[:500]
    clusters = _clusters(top500)
    selected = max([row for row in top500 if _cell(row) == tuple(clusters[0]["cell"])],
                   key=lambda row: (float(row["objective"]), -int(row["number"])))
    selected_metrics, _ = _evaluate(bundle.tapes, bundle.dates, family_id, class_a, selected["parameters"])
    default_params = _default_class_b()
    baseline_metrics, _ = _evaluate(bundle.tapes, bundle.dates, family_id, class_a, default_params)
    v2_metrics, _ = _evaluate(bundle.tapes, bundle.dates, family_id, class_a, v2_params)
    pinned = _boundary_pinned(top500)
    flags = ["BROAD_REGION" if clusters[0]["count"] >= 15 else "NARROW_REGION"]
    if pinned:
        flags.append("BOUNDARY_SENSITIVE")
    if selected_metrics["total_trades"] < 10 or selected_metrics["active_dates"] < 5:
        flags.append("LOW_SAMPLE")
    if selected_metrics["net_r"] <= 0:
        flags.append("NO_USEFUL_REGION")
    summary = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": OBJECTIVE_VERSION, "study_name": _study_name(family_id), "seed": int(seed),
        "requested_trials": int(trials), "complete_trials": len(complete), "failed_trials": failed, "pruned_trials": pruned,
        "calibration_namespace": ROOT_NAME, "search_space": SEARCH_SPACE,
        "class_a_v3_frozen": True, "class_a_v3_config": class_a_params, "class_a_v3_config_hash": class_a_hash,
        "entry_delay_ms": ENTRY_DELAY_MS, "entry_delay_frozen": True,
        "tape_version": candidate_tape.TAPE_VERSION, "dates": list(bundle.dates),
        "best_search": _summary_row(ranked[0]), "top_regions": top_regions,
        "selected_v3": {
            "trial_number": selected["number"], "parameters": selected["parameters"], "metrics": selected_metrics,
            "selection_rule": "largest deterministic 4-bin normalized top-500 Class-B region; highest net R within that region",
            "cluster": clusters[0], "robustness_flags": flags, "boundary_pinned_params": pinned,
        },
        "class_a_v3_train_metrics": baseline_metrics,
        "class_b_v2_train_metrics_on_same_class_a_v3": v2_metrics,
        "class_b_v2_artifact_metrics": v2_artifact_metrics,
        "deltas": {
            "v3_vs_v2": {"net_r": selected_metrics["net_r"] - v2_metrics["net_r"],
                          "profit_factor": selected_metrics["profit_factor"] - v2_metrics["profit_factor"],
                          "max_drawdown_r": selected_metrics["max_drawdown_r"] - v2_metrics["max_drawdown_r"],
                          "total_trades": selected_metrics["total_trades"] - v2_metrics["total_trades"]},
            "v3_vs_class_a_v3": {"net_r": selected_metrics["net_r"] - baseline_metrics["net_r"],
                                  "profit_factor": selected_metrics["profit_factor"] - baseline_metrics["profit_factor"],
                                  "max_drawdown_r": selected_metrics["max_drawdown_r"] - baseline_metrics["max_drawdown_r"],
                                  "total_trades": selected_metrics["total_trades"] - baseline_metrics["total_trades"]},
        },
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
        "artifacts": {"journal": str(journal_path), "trials": str(root / "class-b-v3-trials.jsonl")},
    }
    _atomic_text(root / "class-b-v3-trials.jsonl", "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in records) + "\n")
    _atomic_json(root / "class-b-v3-family-summary.json", summary)
    return summary


def run_batch(*, manifest_path: Path, output_root: Path, trials: int = TRIALS,
              seed: int = SEED, workers: int = 3) -> dict[str, Any]:
    print(json.dumps({"BATCH_FAMILIES": list(CALIBRATION_FAMILIES), "TRIALS_PER_FAMILY": int(trials),
                      "CLASS_A_NAMESPACE": class_a_v3.ROOT_NAME, "NAMESPACE": ROOT_NAME,
                      "SEARCH_SPACE": SEARCH_SPACE, "ENTRY_DELAY_MS": ENTRY_DELAY_MS,
                      "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False},
                     sort_keys=True), flush=True)
    jobs = [(family, manifest_path, output_root, int(trials), int(seed) + index)
            for index, family in enumerate(CALIBRATION_FAMILIES)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = [executor.submit(run_family, family, manifest_path=manifest, output_root=root,
                                    trials=count, seed=job_seed)
                   for family, manifest, root, count, job_seed in jobs]
        for future in futures:
            results.append(future.result())
    results.sort(key=lambda row: CALIBRATION_FAMILIES.index(row["family_id"]))
    payload = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "objective_version": OBJECTIVE_VERSION,
        "selected_families": list(CALIBRATION_FAMILIES), "requested_trials_per_family": int(trials),
        "studies": results, "search_space": SEARCH_SPACE, "class_a_v3_frozen": True,
        "class_a_namespace": class_a_v3.ROOT_NAME, "calibration_namespace": ROOT_NAME,
        "entry_delay_ms": ENTRY_DELAY_MS, "entry_delay_frozen": True,
        "tape_version": candidate_tape.TAPE_VERSION, "validation_accessed": False,
        "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_json(output_root / SUMMARY_NAME, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE")
    parser.add_argument("--manifest", type=Path, default=default_root / "candidate-tapes/train-tape-manifest.json")
    parser.add_argument("--output-root", type=Path, default=default_root / "train-optimization")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    payload = run_batch(manifest_path=args.manifest, output_root=args.output_root,
                        trials=args.trials, seed=args.seed, workers=args.workers)
    print(json.dumps({"CLASS_B_V3_CALIBRATION_TRAIN_COMPLETE": payload["status"] == "COMPLETE",
                      "BATCH_FAMILIES": payload["selected_families"],
                      "TRIALS_PER_FAMILY": payload["requested_trials_per_family"],
                      "CLASS_A_V3_FROZEN": True, "ENTRY_DELAY_MS": ENTRY_DELAY_MS,
                      "VALIDATION_ACCESSED": False, "OOS_ACCESSED": False, "DBN_ACCESSED": False},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
