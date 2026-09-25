"""TRAIN-only Class-B family optimization over sealed Candidate Tape V2 files.

This campaign deliberately loads only the sealed 35-date candidate tapes.  It
does not open raw DBN, Validation, or OOS artifacts.  Each family owns an
independent JournalStorage file so an interrupted worker can be resumed
without recreating completed trials.
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
from . import mac_2025_train_optimization as preparation
from .mac_2025_class_b_proof import _class_a_config
from .model import ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS, L2ClassBConfig, L2Config


TRIALS = 5_000
SEED = 20250924
ROOT_NAME = "class-b-family-optuna"
SUMMARY_NAME = "class-b-family-optuna-summary.json"
OBJECTIVE_VERSION = "MAC2025_TRAIN_CLASS_B_EXACT_NET_R_V1"
ENTRY_DELAY_MS = 2.0

# L2ClassBConfig contains the semantic validity checks.  The repository had
# no prior Class-B search bounds; these are therefore explicit campaign
# bounds, kept in the machine-readable summary and printed before execution.
SEARCH_SPACE: dict[str, Any] = {
    "min_confirmation_seconds": {"distribution": "float", "low": 0.0, "high": 15.0},
    "max_confirmation_seconds": {"distribution": "dependent_float", "low": "min_confirmation_seconds + 0.001", "high": 30.0},
    "favorable_confirmation_ticks": {"distribution": "float", "low": 0.25, "high": 12.0},
    "confirmation_execution_count": {"distribution": "int", "low": 1, "high": 12},
    "confirmation_volume_threshold": {"distribution": "int", "low": 0, "high": 500},
    "stop_ticks": {"distribution": "int", "low": 0, "high": 12},
    "target_r": {"distribution": "float", "low": 0.25, "high": 8.0},
}
_NUMERIC_BOUNDS = {
    "min_confirmation_seconds": (0.0, 15.0), "max_confirmation_seconds": (0.0, 30.0),
    "favorable_confirmation_ticks": (0.25, 12.0), "confirmation_execution_count": (1.0, 12.0),
    "confirmation_volume_threshold": (0.0, 500.0), "stop_ticks": (0.0, 12.0),
    "target_r": (0.25, 8.0),
}


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


def _master_rows(master_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(master_path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("class_a_frozen") is not True:
        raise preparation.TrainOptimizationError("Class-A master selection is not sealed")
    if payload.get("validation_accessed") or payload.get("oos_accessed") or payload.get("dbn_accessed"):
        raise preparation.TrainOptimizationError("sealed Class-A master has forbidden access flags")
    rows = {str(row["family"]): dict(row) for row in payload.get("families", [])}
    if len(rows) != 61:
        raise preparation.TrainOptimizationError("Class-A master does not contain all 61 families")
    return rows


def _existing_completed(output_root: Path) -> set[str]:
    result: set[str] = set()
    base = output_root / ROOT_NAME
    if not base.is_dir():
        return result
    for summary in tuple(base.glob("*/family-summary.json")) + tuple(base.glob("*/class-b-family-summary.json")):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "COMPLETE" and int(payload.get("complete_trials", 0)) >= TRIALS:
            result.add(str(payload.get("family_id")))
    return result


def select_next_families(master_path: Path, output_root: Path, count: int = 5) -> list[str]:
    rows = _master_rows(master_path)
    completed = _existing_completed(output_root)
    return [family for family in rows if rows[family].get("class_b_candidate") == "YES" and family not in completed][:count]


def _sample_class_b(trial: optuna.Trial) -> dict[str, Any]:
    minimum = trial.suggest_float("min_confirmation_seconds", 0.0, 15.0)
    maximum = trial.suggest_float("max_confirmation_seconds", minimum + 0.001, 30.0)
    return {
        "min_confirmation_seconds": minimum,
        "max_confirmation_seconds": maximum,
        "favorable_confirmation_ticks": trial.suggest_float("favorable_confirmation_ticks", 0.25, 12.0),
        "confirmation_execution_count": trial.suggest_int("confirmation_execution_count", 1, 12),
        "confirmation_volume_threshold": trial.suggest_int("confirmation_volume_threshold", 0, 500),
        "stop_ticks": trial.suggest_int("stop_ticks", 0, 12),
        "target_r": trial.suggest_float("target_r", 0.25, 8.0),
    }


def _tape_parameters(class_a: L2Config, class_b: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "weights": {name: float(getattr(class_a, name)) for name in candidate_tape.WEIGHT_NAMES}
                   | {"false_refill_penalty_weight": float(class_a.false_refill_penalty_weight)},
        "aggressive_volume_saturation": class_a.aggressive_volume_saturation,
        "execution_count_saturation": class_a.execution_count_saturation,
        "restore_cycle_saturation": class_a.restore_cycle_saturation,
        "restoration_ratio_saturation": class_a.restoration_ratio_saturation,
        "rejection_saturation_ticks": class_a.rejection_saturation_ticks,
        "persistence_depth_saturation": class_a.persistence_depth_saturation,
        "restoration_latency_saturation_ms": class_a.restoration_latency_saturation_ms,
        "multi_level_ofi_saturation": class_a.multi_level_ofi_saturation,
        "unexecuted_add_penalty_component_weight": class_a.unexecuted_add_penalty_component_weight,
        "rapid_cancel_penalty_component_weight": class_a.rapid_cancel_penalty_component_weight,
        "adverse_progress_penalty_component_weight": class_a.adverse_progress_penalty_component_weight,
        "min_quality_score": class_a.min_quality_score,
        "min_relevant_aggressive_volume": class_a.min_relevant_aggressive_volume,
        "min_relevant_execution_count": class_a.min_relevant_execution_count,
        "min_consume_restore_cycles": class_a.min_consume_restore_cycles,
        "max_through_level_progress_ticks": class_a.max_through_level_progress_ticks,
        "min_rejection_ticks": class_a.min_rejection_ticks,
        **dict(class_b),
        "entry_delay_ms": ENTRY_DELAY_MS,
        "execution_policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    }


def _metrics(trades: Sequence[Mapping[str, Any]], dates: Sequence[str]) -> dict[str, Any]:
    snapshot = preparation._snapshot_metrics(trades=trades, dates=dates, sessions=("ASIA", "EUROPE", "NY"))
    return {
        "net_r": float(snapshot["net_r"]),
        "profit_factor": float(snapshot["profit_factor_capped"]),
        "max_drawdown_r": float(snapshot["max_drawdown_r"]),
        "total_trades": int(snapshot["total_trades"]),
        "active_dates": int(snapshot["active_dates"]),
        "profitable_date_ratio": float(snapshot["profitable_date_ratio"]),
        "date_r": {str(k): float(v) for k, v in snapshot["date_r"].items()},
    }


def _evaluate_family(tapes: Sequence[candidate_tape.CandidateTape], dates: Sequence[str],
                     family_id: str, class_a: L2Config, class_b: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _tape_parameters(class_a, class_b)
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
            all_trades.append({**trade, "date": day, "trading_session": source.get("trading_session"),
                               "family_id": family_id})
    metrics = _metrics(all_trades, dates)
    metrics["qualified_count"] = qualified_count
    return metrics


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = dict(trial.user_attrs)
    return {
        "number": int(trial.number), "state": trial.state.name,
        "params": dict(trial.params), "parameters": attrs.get("parameters", {}),
        "objective": float(trial.value) if trial.value is not None else None,
        "metrics": attrs.get("metrics", {}), "technical_valid": bool(attrs.get("technical_valid", False)),
    }


def _summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"trial_number": int(row["number"]), "objective": float(row["objective"]),
            "parameters": dict(row["parameters"]), **dict(row["metrics"])}


def _bands(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in SEARCH_SPACE:
        values = np.asarray([float(row["parameters"][name]) for row in rows], dtype=float)
        result[name] = {"min": float(values.min()), "q10": float(np.quantile(values, .10)),
                        "median": float(np.quantile(values, .50)), "q90": float(np.quantile(values, .90)),
                        "max": float(values.max())}
    return result


def _clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    names = tuple(SEARCH_SPACE)
    groups: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(
            min(3, max(0, int((float(row["parameters"][name]) - low) / (high - low) * 4)))
            for name in names for low, high in [_NUMERIC_BOUNDS[name]]
        )
        groups.setdefault(key, []).append(row)
    result = []
    for members in groups.values():
        best = max(members, key=lambda row: (float(row["objective"]), -int(row["number"])))
        result.append({"count": len(members), "best": _summary_row(best),
                       "parameter_median": {name: float(np.median([float(x["parameters"][name]) for x in members])) for name in names}})
    return sorted(result, key=lambda row: (-row["best"]["objective"], -row["count"]))


def _representatives(rows: Sequence[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    names = tuple(SEARCH_SPACE)

    def add(candidates: Sequence[dict[str, Any]], limit: int) -> None:
        for row in candidates:
            if len(chosen) >= count or limit <= 0:
                return
            key = tuple(round(float(row["parameters"][name]), 10) for name in names)
            if key in seen:
                continue
            seen.add(key); chosen.append(row); limit -= 1

    by_objective = sorted(rows, key=lambda row: (-float(row["objective"]), int(row["number"])))
    by_net = sorted(rows, key=lambda row: (-float(row["metrics"]["net_r"]), int(row["number"])))
    clusters = _clusters(by_objective)
    cluster_rows = []
    by_number = {int(row["number"]): row for row in by_objective}
    for cluster in clusters:
        trial_number = int(cluster["best"]["trial_number"])
        if trial_number in by_number:
            cluster_rows.append(by_number[trial_number])
    add(by_objective, 6); add(by_net, 5); add(cluster_rows, 5)
    add([row for row in by_objective if int(row["metrics"]["total_trades"]) <= 100], 2)
    add(sorted(rows, key=lambda row: int(row["number"])), 2)
    return chosen


def refresh_completed_analysis(summary_path: Path) -> dict[str, Any]:
    """Refresh post-run region analysis without creating or changing trials."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trial_path = Path(summary["artifacts"]["trials"])
    rows = [json.loads(line) for line in trial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    complete = [row for row in rows if row.get("state") == "COMPLETE" and row.get("technical_valid")]
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    top_regions: dict[str, Any] = {}
    for count in (50, 100, 500):
        top = ranked[:count]
        pinned = []
        for name, (low, high) in _NUMERIC_BOUNDS.items():
            values = np.asarray([float(row["parameters"][name]) for row in top], dtype=float)
            if float(np.mean(values <= low + .05 * (high - low))) >= .75 or float(np.mean(values >= high - .05 * (high - low))) >= .75:
                pinned.append(name)
        clusters = _clusters(top)
        top_regions[str(count)] = {"count": len(top), "best": _summary_row(top[0]),
                                   "parameter_bands": _bands(top), "cluster_count": len(clusters),
                                   "boundary_pinned_params": pinned, "clusters": clusters[:12]}
    summary["top_regions"] = top_regions
    summary["analysis_revision"] = "CLASS_B_REGION_ANALYSIS_V2"
    _atomic_json(summary_path, summary)
    return summary


def _study_name(family_id: str) -> str:
    return f"MAC2025_TRAIN_CLASS_B_FAMILY_{_slug(family_id).upper()}_TPE_JOURNAL_V1"


def run_family(family_id: str, *, manifest_path: Path, master_path: Path, output_root: Path,
               trials: int = TRIALS, seed: int = SEED) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    master = _master_rows(master_path)
    master_row = master[family_id]
    class_a = _class_a_config(master_row)
    bundle = preparation.load_train_tapes(manifest_path=manifest_path, repo_root=Path("."))
    tapes = bundle.tapes
    root = family_output_root(output_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "class-b-study-journal.log"
    journal = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(journal_path)))
    study = optuna.create_study(direction="maximize", study_name=_study_name(family_id),
                                sampler=optuna.samplers.TPESampler(seed=seed, n_ei_candidates=4),
                                storage=journal, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        class_b = _sample_class_b(trial)
        try:
            validated = L2ClassBConfig(**class_b)
            class_b = {name: getattr(validated, name) for name in SEARCH_SPACE}
            metrics = _evaluate_family(tapes, bundle.dates, family_id, class_a, class_b)
            if not all(math.isfinite(float(metrics[name])) for name in ("net_r", "profit_factor", "max_drawdown_r", "profitable_date_ratio")):
                raise ValueError("non-finite Class-B metric")
        except Exception:
            trial.set_user_attr("technical_valid", False)
            raise
        trial.set_user_attr("technical_valid", True)
        trial.set_user_attr("family_id", family_id)
        trial.set_user_attr("parameters", class_b)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("objective_components", {"net_r": metrics["net_r"], "objective_version": OBJECTIVE_VERSION})
        return float(metrics["net_r"])

    complete_before = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, int(trials) - complete_before)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=False, show_progress_bar=False)
    records = [_trial_record(t) for t in study.trials]
    complete = [r for r in records if r["state"] == "COMPLETE" and r["technical_valid"]]
    if len(complete) < int(trials):
        raise preparation.TrainOptimizationError(f"Class-B study incomplete: {family_id}: {len(complete)}/{trials}")
    ranked = sorted(complete, key=lambda row: (-float(row["objective"]), int(row["number"])))
    reps = _representatives(complete, 20)
    exact = []
    for row in reps:
        metrics = _evaluate_family(tapes, bundle.dates, family_id, class_a, row["parameters"])
        exact.append({"trial_number": int(row["number"]), "parameters": dict(row["parameters"]), "fast": dict(row["metrics"]), "exact": metrics})
    baseline_metrics = _evaluate_family(tapes, bundle.dates, family_id, class_a, {name: getattr(L2ClassBConfig(), name) for name in SEARCH_SPACE})
    top_groups = {}
    for count in (50, 100, 500):
        top = ranked[:count]
        top_groups[str(count)] = {"count": len(top), "best": _summary_row(top[0]), "parameter_bands": _bands(top),
                                  "cluster_count": len(_clusters(top)), "clusters": _clusters(top)[:12]}
    best_exact = max(exact, key=lambda row: (float(row["exact"]["net_r"]), -int(row["trial_number"])))
    summary = {
        "status": "COMPLETE", "scope": "TRAIN_ONLY", "family_id": family_id,
        "objective_version": OBJECTIVE_VERSION, "study_name": _study_name(family_id),
        "seed": int(seed), "requested_trials": int(trials), "complete_trials": len(complete),
        "failed_trials": sum(r["state"] == "FAIL" for r in records),
        "class_a_final_classification": master_row["class_a_final_classification"],
        "class_a_study_depth": master_row["study_depth"],
        "class_a_frozen_config": master_row["class_a_frozen_config"],
        "class_a_frozen_config_hash": _sha256_json(master_row["class_a_frozen_config"]),
        "class_a_baseline": baseline_metrics, "best_search": _summary_row(ranked[0]),
        "top_regions": top_groups, "representatives": exact,
        "best_exact": best_exact["exact"], "fast_exact_proxy": "STRONG",
        "search_space": SEARCH_SPACE, "entry_delay_ms": ENTRY_DELAY_MS,
        "entry_delay_frozen": True, "class_a_frozen": True,
        "tape_version": candidate_tape.TAPE_VERSION,
        "artifacts": {"journal": str(journal_path), "trials": str(root / "class-b-trials.jsonl")},
        "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False,
    }
    _atomic_text(root / "class-b-trials.jsonl",
                 "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in records) + "\n")
    _atomic_json(root / "class-b-family-summary.json", summary)
    return summary


def run_batch(*, manifest_path: Path, master_path: Path, output_root: Path,
              families: Sequence[str] | None = None, trials: int = TRIALS,
              seed: int = SEED, workers: int = 3) -> dict[str, Any]:
    selected = list(families or select_next_families(master_path, output_root, 5))
    master = _master_rows(master_path)
    completed = _existing_completed(output_root)
    if len(selected) != 5:
        raise preparation.TrainOptimizationError(f"expected exactly 5 selected Class-B-YES families, got {len(selected)}")
    if any(master[f].get("class_b_candidate") != "YES" for f in selected):
        raise preparation.TrainOptimizationError("Class-B batch contains a non-YES family")
    if any(f in completed for f in selected):
        raise preparation.TrainOptimizationError("selected family already has a completed Class-B study")
    print(json.dumps({"BATCH_FAMILIES": selected,
                      "CLASS_B_SEARCH_SPACE": SEARCH_SPACE,
                      "TRIALS_PER_FAMILY": int(trials), "SEED": int(seed)}, sort_keys=True), flush=True)
    jobs = [(family, manifest_path, master_path, output_root, int(trials), int(seed) + i) for i, family in enumerate(selected)]
    results: list[dict[str, Any]] = []
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
               "validation_accessed": False, "oos_accessed": False, "dbn_accessed": False}
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
    parser.add_argument("--family", action="append", dest="families")
    args = parser.parse_args(argv)
    payload = run_batch(manifest_path=args.manifest, master_path=args.master, output_root=args.output_root,
                        families=args.families, trials=args.trials, seed=args.seed, workers=args.workers)
    print(json.dumps({"CLASS_B_BATCH_COMPLETE": payload["status"] == "COMPLETE",
                      "BATCH_FAMILIES": payload["selected_families"], "TRIALS_PER_FAMILY": payload["requested_trials_per_family"],
                      "ENTRY_DELAY_MS": ENTRY_DELAY_MS, "ENTRY_DELAY_FROZEN": True, "CLASS_A_FROZEN": True,
                      "TAPE_VERSION": candidate_tape.TAPE_VERSION, "VALIDATION_ACCESSED": False,
                      "OOS_ACCESSED": False, "DBN_ACCESSED": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
