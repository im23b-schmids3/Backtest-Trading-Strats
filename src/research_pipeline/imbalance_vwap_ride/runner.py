from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .adapter import ImbalanceVWAPRideAdapter
from .alpha_proxy import run_alpha_proxy
from .artifacts import ArtifactContext, code_hash, sha256_file, sha256_value, utc_now
from .footprint import load_footprint_dataset
from .models import (
    DATASET_HASH,
    EVIDENCE_LABEL,
    PARAMETER_REGISTRY,
    SOURCE_MANIFEST_HASH,
    SPEC_VERSION,
    STRATEGY_ID,
    ImbalanceVWAPRideConfig,
    development_gate,
    freeze_development_candidates,
    locked_test_gate,
    preregistered_variants,
    select_validation_candidate,
    validation_gate,
)
from .strategy import coarsen_footprints

DEVELOPMENT_MONTHS = {"2024-01", "2024-02", "2024-03", "2024-04"}
VALIDATION_MONTHS = {"2024-05", "2024-06"}
LOCKED_MONTHS = {"2024-07"}
FOCUSED_TEST = "tests/research_pipeline/test_imbalance_vwap_ride.py"


def _common_claims() -> dict[str, Any]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "confirmation_evidence": False,
        "external_holdout_required": True,
        "optimization_claimed": False,
        "selection_method": "PRE_REGISTERED_ROBUSTNESS_FILTER",
        "raw_trades_transmitted_externally": False,
        "live_orders_used": False,
        "network_used": False,
        "renormalization_performed": False,
    }


def _run_command(command: list[str], root: Path, *, timeout: int) -> dict[str, Any]:
    started = utc_now()
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def execute_preflight(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    commands = [
        ([sys.executable, "-m", "pytest", "-q", "--basetemp", ".tmp/pytest-imbalance-focused", FOCUSED_TEST], 900),
        ([sys.executable, "-m", "pytest", "-q", "--basetemp", ".tmp/pytest-imbalance-full", "tests/research_pipeline"], 1800),
        ([sys.executable, "-m", "compileall", "-q", "src/research_pipeline", "tests/research_pipeline"], 600),
        (["git", "diff", "--check"], 120),
    ]
    results = [_run_command(command, root, timeout=timeout) for command, timeout in commands]
    return {
        "preflight_version": "imbalance-vwap-ride-preflight-1",
        "timestamp": utc_now(),
        "repository_root": str(root),
        "code_hash": code_hash(root),
        "focused_tests_passed": results[0]["passed"],
        "full_research_pipeline_tests_passed": results[1]["passed"],
        "compileall_passed": results[2]["passed"],
        "diff_check_passed": results[3]["passed"],
        "tests_passed": all(item["passed"] for item in results),
        "commands": results,
    }


def _validate_preflight(preflight: dict[str, Any], root: Path) -> None:
    required = (
        "focused_tests_passed",
        "full_research_pipeline_tests_passed",
        "compileall_passed",
        "diff_check_passed",
        "tests_passed",
    )
    if any(preflight.get(name) is not True for name in required):
        raise RuntimeError("sealed real study is fail-closed until all preflight checks pass")
    if preflight.get("code_hash") != code_hash(root):
        raise RuntimeError("preflight code hash no longer matches the implementation")


def _variant_context(base: ArtifactContext, config: ImbalanceVWAPRideConfig) -> ArtifactContext:
    return ArtifactContext(
        run_id=base.run_id,
        dataset_hash=base.dataset_hash,
        source_manifest_hash=base.source_manifest_hash,
        specification_hash=base.specification_hash,
        parameter_hash=sha256_value(config.parameter_payload()),
        code_hash=base.code_hash,
        evidence_label=base.evidence_label,
        timestamp=base.timestamp,
    )


def _persist_result(
    base_context: ArtifactContext,
    target: Path,
    config: ImbalanceVWAPRideConfig,
    result: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    context = _variant_context(base_context, config)
    context.write_json(target / "specification.json", {**_common_claims(), "phase": phase, "variant_id": config.variant_id, "parameters": config.parameter_payload()})
    context.write_parquet(target / "events.parquet", result["events"])
    context.write_parquet(target / "zones.parquet", result["zones"])
    context.write_parquet(target / "trades.parquet", result["trades"])
    context.write_json(target / "months.json", {"phase": phase, "variant_id": config.variant_id, "months": result["metrics"]["months"]})
    context.write_json(target / "funnel.json", {"phase": phase, "variant_id": config.variant_id, **result["funnel"]})
    report = {**_common_claims(), "phase": phase, "variant_id": config.variant_id, "parameters": config.parameter_payload(), "metrics": result["metrics"]}
    context.write_json(target / "report.json", report)
    context.write_json(
        target / "diagnostics.json",
        {
            "phase": phase,
            "variant_id": config.variant_id,
            "event_count": len(result["events"]),
            "zone_record_count": len(result["zones"]),
            "trade_count": len(result["trades"]),
            "funnel_reconciles": result["funnel"]["reconciles"],
            "same_bar_policy": "STOP_FIRST",
            "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
        },
    )
    return report


def _existing_final(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path, final_path = root / "study-manifest.json", root / "final_report.json"
    if not manifest_path.exists() and not final_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError(f"immutable study artifact collision: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"immutable study identity collision for {key}: {root}")
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        return {
            "status": final["status"],
            "summary": final["summary"],
            "finalReportPath": str(final_path),
            "testsPassed": bool(final.get("tests_passed")),
            "studyExecuted": bool(final.get("study_executed")),
        }
    return None


def run_sealed_study(
    *,
    data_manifest: str | Path,
    artifact_root: str | Path = "research_runs",
    repository_root: str | Path = ".",
    footprint_cache_root: str | Path = "data/imbalance_vwap_ride/footprints",
    batch_size: int = 1_000_000,
    preflight_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    _validate_preflight(preflight_evidence or {}, repository)
    spec_path = repository / ".smithers" / "specs" / "imbalance-vwap-ride-btc-exploratory.md"
    specification_hash = sha256_file(spec_path)
    registry = [item.model_dump(mode="json") for item in preregistered_variants()]
    parameter_hash = sha256_value(registry)
    implementation_hash = code_hash(repository)
    identity = {
        "strategy_id": STRATEGY_ID,
        "dataset_hash": DATASET_HASH,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "specification_hash": specification_hash,
        "parameter_hash": parameter_hash,
        "code_hash": implementation_hash,
        "specification_version": SPEC_VERSION,
        "evidence_label": EVIDENCE_LABEL,
    }
    run_id = sha256_value(identity)[:24]
    root = Path(artifact_root).resolve() / STRATEGY_ID / run_id
    existing = _existing_final(root, identity)
    if existing is not None:
        return existing
    manifest_path = root / "study-manifest.json"
    manifest_already_exists = manifest_path.exists()
    timestamp = utc_now()
    if manifest_path.exists():
        timestamp = json.loads(manifest_path.read_text(encoding="utf-8"))["artifact_timestamp"]
    context = ArtifactContext(
        run_id=run_id,
        dataset_hash=DATASET_HASH,
        source_manifest_hash=SOURCE_MANIFEST_HASH,
        specification_hash=specification_hash,
        parameter_hash=parameter_hash,
        code_hash=implementation_hash,
        evidence_label=EVIDENCE_LABEL,
        timestamp=timestamp,
    )
    if not manifest_already_exists:
        context.write_json(
            manifest_path,
            {
                **identity,
                **_common_claims(),
                "study_run_id": run_id,
                "source_manifest_path": str(Path(data_manifest).resolve()),
                "development_months": sorted(DEVELOPMENT_MONTHS),
                "validation_months": sorted(VALIDATION_MONTHS),
                "locked_test_months": sorted(LOCKED_MONTHS),
                "preflight": preflight_evidence,
            },
        )
    context.write_json(root / "specification.json", {**_common_claims(), "specification_version": SPEC_VERSION, "sealed_specification_path": str(spec_path)})
    context.write_json(
        root / "parameter_registry.json",
        {
            **_common_claims(),
            "search_type": "UNIQUE_ONE_FACTOR_ONLY",
            "cartesian_search": False,
            "bayesian_search": False,
            "genetic_search": False,
            "random_search": False,
            "registry": registry,
            "families": [{"name": name, "values": [str(value) for value in values]} for name, values in PARAMETER_REGISTRY],
        },
    )
    if not (root / "preflight_validation.json").exists():
        context.write_json(root / "preflight_validation.json", preflight_evidence)

    adapter = ImbalanceVWAPRideAdapter()
    source_validation = adapter.validate_source(data_manifest, verify_hashes=True)
    context.write_json(root / "data_validation_report.json", {**_common_claims(), **{key: value for key, value in source_validation.items() if key != "manifest"}})
    if not source_validation["valid"]:
        raise RuntimeError("immutable source validation failed")
    footprint_manifest = adapter.materialize_footprint(
        data_manifest,
        repository / footprint_cache_root,
        batch_size=batch_size,
        verify_source_hashes=False,
    )
    footprint_validation = adapter.validate_footprint(footprint_manifest["footprint_root"])
    context.write_json(root / "footprint_validation_report.json", {**_common_claims(), **footprint_validation, "footprint_manifest": footprint_manifest})
    if not footprint_validation["valid"]:
        raise RuntimeError("content-addressed footprint validation failed")

    development_footprints, development_bars = load_footprint_dataset(footprint_manifest, months=DEVELOPMENT_MONTHS)
    development_coarsened = {
        size: coarsen_footprints(development_footprints, size)
        for size in {config.bin_size_usd for config in preregistered_variants()}
    }
    del development_footprints
    development_runs: list[tuple[ImbalanceVWAPRideConfig, dict[str, Any]]] = []
    development_reports: list[dict[str, Any]] = []
    for config in preregistered_variants():
        result = adapter.run_loaded(bars=development_bars, footprints=development_coarsened[config.bin_size_usd], config=config)
        target = root / "development" / config.variant_id
        report = _persist_result(context, target, config, result, phase="DEVELOPMENT_2024_01_TO_04")
        gate = development_gate(result["metrics"])
        _variant_context(context, config).write_json(target / "gate.json", {"phase": "DEVELOPMENT", "variant_id": config.variant_id, **gate})
        development_runs.append((config, result["metrics"]))
        development_reports.append({"variant_id": config.variant_id, "metrics": result["metrics"], "gate": gate, "report_path": str(target / "report.json")})
        if config.variant_id == "baseline":
            context.write_json(root / "baseline_report.json", report)
    context.write_json(
        root / "ablation_report.json",
        {
            **_common_claims(),
            "diagnostic_only": True,
            "selection_by_highest_pnl_prohibited": True,
            "runs": development_reports,
        },
    )
    context.write_json(root / "development_report.json", {**_common_claims(), "runs": development_reports})
    frozen = freeze_development_candidates(development_runs)
    frozen_payload = [
        {"variant_id": config.variant_id, "parameters": config.parameter_payload(), "development_metrics": metrics}
        for config, metrics in frozen
    ]
    context.write_json(
        root / "selection_report.json",
        {
            **_common_claims(),
            "phase": "DEVELOPMENT",
            "candidate_count": len(frozen),
            "candidates": frozen_payload,
            "selection_basis": "development gates followed by baseline-nearest and lexicographic stability ordering; never highest PnL",
        },
    )
    context.write_json(root / "frozen_candidates.json", {**_common_claims(), "maximum_candidates": 3, "candidates": frozen_payload})

    validation_reports: list[dict[str, Any]] = []
    selected: tuple[ImbalanceVWAPRideConfig, dict[str, Any]] | None = None
    locked_result: dict[str, Any] | None = None
    locked_gate: dict[str, Any] | None = None
    if frozen:
        validation_footprints, validation_bars = load_footprint_dataset(footprint_manifest, months=VALIDATION_MONTHS)
        validation_coarsened = {
            size: coarsen_footprints(validation_footprints, size)
            for size in {config.bin_size_usd for config, _ in frozen}
        }
        del validation_footprints
        for config, _ in frozen:
            result = adapter.run_loaded(bars=validation_bars, footprints=validation_coarsened[config.bin_size_usd], config=config)
            target = root / "validation" / config.variant_id
            _persist_result(context, target, config, result, phase="VALIDATION_2024_05_TO_06")
            gate = validation_gate(result["metrics"])
            _variant_context(context, config).write_json(target / "gate.json", {"phase": "VALIDATION", "variant_id": config.variant_id, **gate})
            validation_reports.append({"variant_id": config.variant_id, "parameters": config.parameter_payload(), "metrics": result["metrics"], "gate": gate, "report_path": str(target / "report.json")})
        selected = select_validation_candidate([(config, next(item["metrics"] for item in validation_reports if item["variant_id"] == config.variant_id)) for config, _ in frozen])
    context.write_json(
        root / "validation_report.json",
        {
            **_common_claims(),
            "status": "COMPLETED" if frozen else "NOT_EXECUTED_DEVELOPMENT_EDGE_NOT_FOUND",
            "runs": validation_reports,
            "selected_variant_id": selected[0].variant_id if selected else None,
            "selection_tiebreakers": ["highest_profit_factor", "lower_drawdown", "higher_trade_count", "baseline_nearest", "lexicographic_variant_id"],
        },
    )
    context.write_json(
        root / "final_frozen_strategy.json",
        {
            **_common_claims(),
            "status": "FROZEN" if selected else "NO_VALIDATION_CANDIDATE",
            "variant_id": selected[0].variant_id if selected else None,
            "parameters": selected[0].parameter_payload() if selected else None,
            "validation_metrics": selected[1] if selected else None,
        },
    )

    if selected:
        lock_path = root / "locked_test_execution_lock.json"
        locked_report_path = root / "locked_test_report.json"
        if lock_path.exists() and not locked_report_path.exists():
            raise RuntimeError("locked test was already opened but did not complete; automatic re-execution is forbidden")
        locked_footprints, locked_bars = load_footprint_dataset(footprint_manifest, months=LOCKED_MONTHS)
        if locked_report_path.exists():
            prior_locked = json.loads(locked_report_path.read_text(encoding="utf-8"))
            locked_gate = prior_locked["gate"]
            locked_result = {
                "metrics": prior_locked["metrics"],
                "trades": pq.read_table(root / "locked_test" / "trades.parquet").to_pylist(),
            }
        else:
            context.write_json(lock_path, {**_common_claims(), "state": "OPENED_ONCE", "variant_id": selected[0].variant_id, "months": sorted(LOCKED_MONTHS)})
            locked_result = adapter.run_loaded(
                bars=locked_bars,
                footprints=coarsen_footprints(locked_footprints, selected[0].bin_size_usd),
                config=selected[0],
            )
            _persist_result(context, root / "locked_test", selected[0], locked_result, phase="LOCKED_TEST_2024_07")
            locked_gate = locked_test_gate(locked_result["metrics"])
            context.write_json(
                locked_report_path,
                {
                    **_common_claims(),
                    "status": "PASS" if locked_gate["passed"] else "FAIL",
                    "variant_id": selected[0].variant_id,
                    "metrics": locked_result["metrics"],
                    "gate": locked_gate,
                    "opened_exactly_once": True,
                },
            )
    else:
        context.write_json(
            root / "locked_test_report.json",
            {
                **_common_claims(),
                "status": "NOT_EXECUTED_NO_FROZEN_VALIDATION_CANDIDATE" if frozen else "NOT_EXECUTED_DEVELOPMENT_EDGE_NOT_FOUND",
                "opened_exactly_once": False,
            },
        )

    if locked_result is not None and locked_gate is not None and locked_gate["passed"]:
        alpha = run_alpha_proxy(locked_result["trades"], locked_bars, paths=10_000)
        context.write_json(root / "alpha_proxy_rules_report.json", {**_common_claims(), "status": "EXECUTED", "rules": alpha["rules"]})
        context.write_json(root / "alpha_proxy_evaluation_report.json", {**_common_claims(), "status": "EXECUTED", "mapping": alpha["mapping"], "daily_pnl": alpha["daily_pnl"], "evaluation": alpha["evaluation"], "sensitivities": alpha["sensitivities"]})
        context.write_json(root / "alpha_proxy_qualified_report.json", {**_common_claims(), "status": "EXECUTED", "qualified": alpha["qualified"]})
        alpha_status = "EXECUTED"
    else:
        reason = "JULY_GATE_NOT_PASSED" if locked_result is not None else "JULY_NOT_EXECUTED"
        context.write_json(root / "alpha_proxy_rules_report.json", {**_common_claims(), "status": "NOT_EXECUTED", "reason": reason})
        context.write_json(root / "alpha_proxy_evaluation_report.json", {**_common_claims(), "status": "NOT_EXECUTED", "reason": reason})
        context.write_json(root / "alpha_proxy_qualified_report.json", {**_common_claims(), "status": "NOT_EXECUTED", "reason": reason})
        alpha_status = "NOT_EXECUTED"

    status = "DEVELOPMENT_EDGE_NOT_FOUND" if not frozen else "COMPLETED"
    if not frozen:
        summary = "All pre-registered development diagnostics completed; no variant passed the development candidate gate, so validation, the July locked test, and Alpha proxy were not opened."
    elif selected is None:
        summary = "Development candidates were frozen and validated, but none passed validation; the July locked test and Alpha proxy were not opened."
    else:
        summary = f"The sealed study completed through the July locked test ({'PASS' if locked_gate and locked_gate['passed'] else 'FAIL'}); Alpha proxy status: {alpha_status}."
    final = {
        **_common_claims(),
        "status": status,
        "summary": summary,
        "tests_passed": True,
        "study_executed": True,
        "development_candidate_count": len(frozen),
        "validation_selected_variant_id": selected[0].variant_id if selected else None,
        "locked_test_status": "PASS" if locked_gate and locked_gate["passed"] else "FAIL" if locked_gate else "NOT_EXECUTED",
        "alpha_proxy_status": alpha_status,
        "artifact_index": sorted({str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()} | {"final_report.json"}),
    }
    context.write_json(root / "final_report.json", final)
    return {
        "status": status,
        "summary": summary,
        "finalReportPath": str(root / "final_report.json"),
        "testsPassed": True,
        "studyExecuted": True,
    }


def verify_and_run_sealed_study(**kwargs: Any) -> dict[str, Any]:
    repository = Path(kwargs.get("repository_root", ".")).resolve()
    preflight = execute_preflight(repository)
    if not preflight["tests_passed"]:
        return {
            "status": "FAILED",
            "summary": "Pre-execution focused/full tests, compileall, or diff check failed; the sealed study was not executed.",
            "finalReportPath": None,
            "testsPassed": False,
            "studyExecuted": False,
            "preflight": preflight,
        }
    return run_sealed_study(**kwargs, preflight_evidence=preflight)
