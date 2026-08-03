from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .alpha_proxy import alpha_proxy_rules, run_alpha_proxy
from .artifacts import ArtifactContext, code_hash, sha256_file, sha256_value, utc_now
from .footprint import load_footprint_dataset
from .v2_adapter import ImbalanceVWAPRideV2Adapter
from .v2_models import (
    DATASET_HASH,
    EVIDENCE_LABEL,
    PARAMETER_REGISTRY,
    SOURCE_MANIFEST_HASH,
    SPEC_VERSION,
    STRATEGY_ID,
    ImbalanceVWAPRideV2Config,
    development_gate,
    freeze_development_candidates,
    locked_test_gate,
    preregistered_variants,
    select_validation_candidate,
    validation_gate,
)
from .v2_strategy import coarsen_footprints

DEVELOPMENT_MONTHS = {"2024-01", "2024-02", "2024-03", "2024-04"}
VALIDATION_MONTHS = {"2024-05", "2024-06"}
LOCKED_MONTHS = {"2024-07"}
FOCUSED_TESTS = (
    "tests/research_pipeline/test_imbalance_vwap_ride.py",
    "tests/research_pipeline/test_imbalance_vwap_ride_v2.py",
)
V1_RUN_RELATIVE = Path("research_runs/ImbalanceVWAPRide.BTC_EXPLORATORY/ff0afc85ef4b46c0bf671cfb")
V1_FILE_COUNT = 171
V1_TREE_DIGEST = "726c383e2ffde79bd4504889ca222871bf052345dc7cd3c7f93a7eb3448ef182"
SOURCE_MANIFEST_FILE_SHA256 = "286f3388629b68c474af056c2588e61298bfdf25a6390a1ff7677016c8f26365"


def _common_claims() -> dict[str, Any]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "confirmation_evidence": False,
        "external_holdout_required": True,
        "optimization_claimed": False,
        "selection_method": "UNIQUE_ONE_FACTOR_ROBUSTNESS_FILTER",
        "raw_trades_transmitted_externally": False,
        "live_orders_used": False,
        "network_used": False,
        "downloads_performed": False,
        "renormalization_performed": False,
    }


@dataclass(frozen=True)
class V2ArtifactContext(ArtifactContext):
    """Artifact context that seals the V2 labels into JSON and Parquet metadata."""

    def metadata(self) -> dict[str, str]:
        return {
            **ArtifactContext.metadata(self),
            "confirmation_evidence": "false",
            "optimization_claimed": "false",
            "external_holdout_required": "true",
        }

    def envelope(self, payload: Any) -> dict[str, Any]:
        base = {**ArtifactContext.metadata(self), **_common_claims()}
        if isinstance(payload, dict):
            return {**base, **payload}
        return {**base, "payload": payload}


def tree_digest(path: str | Path) -> tuple[int, str]:
    root = Path(path).resolve()
    records: list[str] = []
    for item in sorted((candidate for candidate in root.rglob("*") if candidate.is_file())):
        relative = item.relative_to(root).as_posix()
        records.append(f"{relative}\t{item.stat().st_size}\t{sha256_file(item)}")
    return len(records), hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def preservation_snapshot(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    count, digest = tree_digest(root / V1_RUN_RELATIVE)
    manifest_path = (
        root
        / "data"
        / "value_area_trap"
        / "normalized"
        / "BTCUSDT"
        / DATASET_HASH
        / "manifest.json"
    )
    return {
        "v1_file_count": count,
        "v1_tree_digest": digest,
        "v1_preserved": count == V1_FILE_COUNT and digest == V1_TREE_DIGEST,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_preserved": sha256_file(manifest_path) == SOURCE_MANIFEST_FILE_SHA256,
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
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                ".tmp/pytest-imbalance-v2-focused",
                *FOCUSED_TESTS,
            ],
            900,
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                ".tmp/pytest-imbalance-v2-full",
                "tests/research_pipeline",
            ],
            1800,
        ),
        ([sys.executable, "-m", "compileall", "-q", "src/research_pipeline", "tests/research_pipeline"], 600),
        (["git", "diff", "--check"], 120),
    ]
    results = [_run_command(command, root, timeout=timeout) for command, timeout in commands]
    preservation = preservation_snapshot(root)
    tests_passed = all(result["passed"] for result in results) and all(
        preservation[key] for key in ("v1_preserved", "source_manifest_preserved")
    )
    return {
        "preflight_version": "imbalance-vwap-ride-btc-macro-bins-v2-preflight-1",
        "timestamp": utc_now(),
        "repository_root": str(root),
        "code_hash": code_hash(root),
        "focused_tests_passed": results[0]["passed"],
        "full_research_pipeline_tests_passed": results[1]["passed"],
        "compileall_passed": results[2]["passed"],
        "diff_check_passed": results[3]["passed"],
        "preservation": preservation,
        "tests_passed": tests_passed,
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
        raise RuntimeError("V2 real study is fail-closed until every preflight check passes")
    preservation = preflight.get("preservation", {})
    if not preservation.get("v1_preserved") or not preservation.get("source_manifest_preserved"):
        raise RuntimeError("V1 or immutable source preservation check failed")
    if preflight.get("code_hash") != code_hash(root):
        raise RuntimeError("preflight code hash no longer matches the V2 implementation")


def _variant_context(base: ArtifactContext, config: ImbalanceVWAPRideV2Config) -> V2ArtifactContext:
    return V2ArtifactContext(
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
    config: ImbalanceVWAPRideV2Config,
    result: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    context = _variant_context(base_context, config)
    metrics = result["metrics"]
    context.write_json(
        target / "specification.json",
        {
            **_common_claims(),
            "phase": phase,
            "variant_id": config.variant_id,
            "parameters": config.parameter_payload(),
        },
    )
    context.write_parquet(target / "events.parquet", result["events"])
    context.write_parquet(target / "zones.parquet", result["zones"])
    context.write_parquet(target / "trades.parquet", result["trades"])
    context.write_json(target / "months.json", {"phase": phase, "variant_id": config.variant_id, "months": metrics["months"]})
    context.write_json(target / "funnel.json", {"phase": phase, "variant_id": config.variant_id, **result["funnel"]})
    context.write_json(
        target / "gross_net.json",
        {
            "phase": phase,
            "variant_id": config.variant_id,
            **{
                name: metrics[name]
                for name in (
                    "gross_pnl",
                    "net_pnl",
                    "gross_profit_factor",
                    "net_profit_factor",
                    "average_gross_r",
                    "average_net_r",
                    "median_gross_r",
                    "median_net_r",
                )
            },
        },
    )
    context.write_json(
        target / "cost_diagnostics.json",
        {
            "phase": phase,
            "variant_id": config.variant_id,
            **{
                name: metrics[name]
                for name in (
                    "median_initial_risk_usd",
                    "gross_risk_usd",
                    "fees",
                    "slippage_cost",
                    "total_costs",
                    "median_cost_to_risk",
                    "cost_to_risk_share_over_10_percent",
                    "cost_to_risk_share_over_25_percent",
                    "cost_to_risk_share_over_50_percent",
                )
            },
        },
    )
    context.write_json(
        target / "concentration.json",
        {
            "phase": phase,
            "variant_id": config.variant_id,
            "maximum_positive_month_contribution": metrics["maximum_positive_month_contribution"],
            "best_five_positive_pnl_contribution": metrics["best_five_positive_pnl_contribution"],
        },
    )
    context.write_json(
        target / "long_short.json",
        {
            "phase": phase,
            "variant_id": config.variant_id,
            "long_trades": metrics["long_trades"],
            "short_trades": metrics["short_trades"],
            "long_short_metrics": metrics["long_short_metrics"],
            "long_short_reconciliation": metrics["long_short_reconciliation"],
        },
    )
    report = {
        **_common_claims(),
        "phase": phase,
        "variant_id": config.variant_id,
        "parameters": config.parameter_payload(),
        "metrics": metrics,
    }
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
            "long_short_reconciles": metrics["long_short_reconciliation"]["reconciles"],
            "terminal_states": ["EXPIRED", "INVALIDATED", "TRADED", "SUPERSEDED"],
            "same_bar_policy": "STOP_FIRST",
            "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
            "risk_basis": "ADVERSELY_SLIPPED_QUANTIZED_NEXT_BAR_ACTUAL_ENTRY",
        },
    )
    return report


def _existing_final(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / "study-manifest.json"
    final_path = root / "final_report.json"
    if not manifest_path.exists() and not final_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError(f"immutable V2 study artifact collision: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"immutable V2 study identity collision for {key}: {root}")
    if not final_path.exists():
        return None
    final = json.loads(final_path.read_text(encoding="utf-8"))
    return {
        "status": final["status"],
        "summary": final["summary"],
        "finalReportPath": str(final_path),
        "testsPassed": bool(final.get("tests_passed")),
        "studyExecuted": bool(final.get("study_executed")),
    }


def _alpha_contract() -> dict[str, Any]:
    rules = alpha_proxy_rules()
    return {
        "rules_version": rules["rules_version"],
        "instrument_mapping": "BINANCE_BTCUSDT_TO_ONE_MBT_PROXY_ONLY",
        "mbt_contracts": 1,
        "btc_equivalent": "0.1",
        "risk_basis": "SOURCE_TRADES_USE_ACTUAL_QUANTIZED_ENTRY_RISK",
        "eligibility_claimed": False,
        "limitations": rules["limitations"],
    }


def run_sealed_v2_study(
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
    preservation_before = preservation_snapshot(repository)
    spec_path = repository / ".smithers" / "specs" / "imbalance-vwap-ride-btc-macro-bins-v2.md"
    registry = [item.model_dump(mode="json") for item in preregistered_variants()]
    identity = {
        "strategy_id": STRATEGY_ID,
        "dataset_hash": DATASET_HASH,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "specification_hash": sha256_file(spec_path),
        "parameter_hash": sha256_value(registry),
        "code_hash": code_hash(repository),
        "specification_version": SPEC_VERSION,
        "evidence_label": EVIDENCE_LABEL,
    }
    run_id = sha256_value(identity)[:24]
    root = Path(artifact_root).resolve() / STRATEGY_ID / run_id
    existing = _existing_final(root, identity)
    if existing is not None:
        if preservation_snapshot(repository) != preservation_before:
            raise RuntimeError("preservation changed while validating deterministic V2 rerun")
        return existing

    manifest_path = root / "study-manifest.json"
    timestamp = utc_now()
    if manifest_path.exists():
        timestamp = json.loads(manifest_path.read_text(encoding="utf-8"))["artifact_timestamp"]
    context = V2ArtifactContext(
        run_id=run_id,
        dataset_hash=DATASET_HASH,
        source_manifest_hash=SOURCE_MANIFEST_HASH,
        specification_hash=identity["specification_hash"],
        parameter_hash=identity["parameter_hash"],
        code_hash=identity["code_hash"],
        evidence_label=EVIDENCE_LABEL,
        timestamp=timestamp,
    )
    if not manifest_path.exists():
        context.write_json(
            manifest_path,
            {
                **identity,
                **_common_claims(),
                "study_run_id": run_id,
                "source_manifest_path": str(Path(data_manifest).resolve()),
                "development_months": sorted(DEVELOPMENT_MONTHS),
                "validation_months_conditional": sorted(VALIDATION_MONTHS),
                "locked_test_months_conditional": sorted(LOCKED_MONTHS),
                "preflight": preflight_evidence,
                "preservation_before": preservation_before,
            },
        )
    context.write_json(
        root / "specification.json",
        {
            **_common_claims(),
            "specification_version": SPEC_VERSION,
            "sealed_specification_path": str(spec_path),
        },
    )
    context.write_json(
        root / "parameter_registry.json",
        {
            **_common_claims(),
            "search_type": "EXACT_UNIQUE_ONE_FACTOR_ONLY",
            "stable_order": True,
            "cartesian_search": False,
            "bayesian_search": False,
            "genetic_search": False,
            "random_search": False,
            "hidden_optimization": False,
            "registry": registry,
            "families": [
                {"name": name, "values": [str(value) for value in values]}
                for name, values in PARAMETER_REGISTRY
            ],
        },
    )
    if not (root / "preflight_validation.json").exists():
        context.write_json(root / "preflight_validation.json", preflight_evidence)

    adapter = ImbalanceVWAPRideV2Adapter()
    source_validation = adapter.validate_source(data_manifest, verify_hashes=True)
    context.write_json(
        root / "data_validation_report.json",
        {
            **_common_claims(),
            **{key: value for key, value in source_validation.items() if key != "manifest"},
        },
    )
    if not source_validation["valid"]:
        raise RuntimeError("immutable source validation failed")
    footprint_manifest = adapter.materialize_footprint(
        data_manifest,
        repository / footprint_cache_root,
        batch_size=batch_size,
        verify_source_hashes=False,
    )
    footprint_validation = adapter.validate_footprint(footprint_manifest["footprint_root"])
    context.write_json(
        root / "footprint_validation_report.json",
        {
            **_common_claims(),
            **footprint_validation,
            "footprint_manifest": footprint_manifest,
            "bounded_batch_size": batch_size,
            "existing_content_addressed_footprint_reused": True,
        },
    )
    if not footprint_validation["valid"]:
        raise RuntimeError("content-addressed footprint validation failed")

    development_footprints, development_bars = load_footprint_dataset(
        footprint_manifest,
        months=DEVELOPMENT_MONTHS,
    )
    variants = preregistered_variants()
    development_coarsened = {
        size: coarsen_footprints(development_footprints, size)
        for size in {config.bin_size_usd for config in variants}
    }
    del development_footprints
    development_runs: list[tuple[ImbalanceVWAPRideV2Config, dict[str, Any]]] = []
    development_reports: list[dict[str, Any]] = []
    for config in variants:
        result = adapter.run_loaded(
            bars=development_bars,
            footprints=development_coarsened[config.bin_size_usd],
            config=config,
        )
        target = root / "development" / config.variant_id
        report = _persist_result(context, target, config, result, phase="DEVELOPMENT_2024_01_TO_04")
        gate = development_gate(result["metrics"])
        _variant_context(context, config).write_json(
            target / "gate.json",
            {"phase": "DEVELOPMENT", "variant_id": config.variant_id, **gate},
        )
        development_runs.append((config, result["metrics"]))
        development_reports.append(
            {
                "variant_id": config.variant_id,
                "metrics": result["metrics"],
                "gate": gate,
                "report_path": str(target / "report.json"),
            }
        )
        if config.variant_id == "baseline":
            context.write_json(root / "baseline_report.json", report)
    context.write_json(
        root / "development_report.json",
        {**_common_claims(), "runs": development_reports},
    )
    context.write_json(
        root / "ablation_report.json",
        {
            **_common_claims(),
            "diagnostic_only": True,
            "selection_by_highest_pnl_prohibited": True,
            "runs": development_reports,
        },
    )
    diagnosis_counts = Counter(
        item["gate"]["edge_diagnosis"]["classification"] for item in development_reports
    )
    context.write_json(
        root / "edge_diagnosis_report.json",
        {
            **_common_claims(),
            "classifications": dict(sorted(diagnosis_counts.items())),
            "definitions": {
                "PRE_COST_EDGE_FAILURE": "At least 40 trades but gross PF/R did not clear the development edge threshold.",
                "COST_DESTROYED_EDGE": "Gross PF/R cleared, but net PF/R failed after measured fees and slippage.",
                "RESTRICTIVE_THRESHOLD_SAMPLE_INSUFFICIENCY": "Fewer than 40 executed trades under the registered setting.",
            },
        },
    )
    frozen = freeze_development_candidates(development_runs)
    frozen_payload = [
        {
            "variant_id": config.variant_id,
            "parameters": config.parameter_payload(),
            "development_metrics": metrics,
        }
        for config, metrics in frozen
    ]
    context.write_json(
        root / "selection_report.json",
        {
            **_common_claims(),
            "phase": "DEVELOPMENT",
            "candidate_count": len(frozen),
            "candidates": frozen_payload,
            "selection_basis": "development gates, then baseline-nearest and stable lexicographic order; maximum three",
        },
    )
    context.write_json(
        root / "frozen_candidates.json",
        {**_common_claims(), "maximum_candidates": 3, "candidates": frozen_payload},
    )

    validation_reports: list[dict[str, Any]] = []
    selected: tuple[ImbalanceVWAPRideV2Config, dict[str, Any]] | None = None
    locked_result: dict[str, Any] | None = None
    locked_gate: dict[str, Any] | None = None
    locked_bars: list[dict[str, Any]] = []
    if frozen:
        validation_footprints, validation_bars = load_footprint_dataset(
            footprint_manifest,
            months=VALIDATION_MONTHS,
        )
        validation_coarsened = {
            size: coarsen_footprints(validation_footprints, size)
            for size in {config.bin_size_usd for config, _ in frozen}
        }
        del validation_footprints
        validation_candidates: list[tuple[ImbalanceVWAPRideV2Config, dict[str, Any]]] = []
        for config, _ in frozen:
            result = adapter.run_loaded(
                bars=validation_bars,
                footprints=validation_coarsened[config.bin_size_usd],
                config=config,
            )
            target = root / "validation" / config.variant_id
            _persist_result(context, target, config, result, phase="VALIDATION_2024_05_TO_06")
            gate = validation_gate(result["metrics"])
            _variant_context(context, config).write_json(
                target / "gate.json",
                {"phase": "VALIDATION", "variant_id": config.variant_id, **gate},
            )
            validation_candidates.append((config, result["metrics"]))
            validation_reports.append(
                {
                    "variant_id": config.variant_id,
                    "parameters": config.parameter_payload(),
                    "metrics": result["metrics"],
                    "gate": gate,
                    "report_path": str(target / "report.json"),
                }
            )
        selected = select_validation_candidate(validation_candidates)
    context.write_json(
        root / "validation_report.json",
        {
            **_common_claims(),
            "status": "COMPLETED" if frozen else "NOT_EXECUTED_DEVELOPMENT_EDGE_NOT_FOUND",
            "runs": validation_reports,
            "selected_variant_id": selected[0].variant_id if selected else None,
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
            raise RuntimeError("V2 locked test was opened but did not complete; automatic re-execution is forbidden")
        locked_footprints, locked_bars = load_footprint_dataset(footprint_manifest, months=LOCKED_MONTHS)
        if locked_report_path.exists():
            prior = json.loads(locked_report_path.read_text(encoding="utf-8"))
            locked_gate = prior["gate"]
            locked_result = {
                "metrics": prior["metrics"],
                "trades": pq.read_table(root / "locked_test" / "trades.parquet").to_pylist(),
            }
        else:
            context.write_json(
                lock_path,
                {
                    **_common_claims(),
                    "state": "OPENED_ONCE",
                    "variant_id": selected[0].variant_id,
                    "months": sorted(LOCKED_MONTHS),
                },
            )
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
                "status": (
                    "NOT_EXECUTED_NO_FROZEN_VALIDATION_CANDIDATE"
                    if frozen
                    else "NOT_EXECUTED_DEVELOPMENT_EDGE_NOT_FOUND"
                ),
                "opened_exactly_once": False,
            },
        )

    alpha_contract = _alpha_contract()
    if locked_result is not None and locked_gate is not None and locked_gate["passed"]:
        alpha = run_alpha_proxy(locked_result["trades"], locked_bars, paths=10_000)
        context.write_json(
            root / "alpha_proxy_rules_report.json",
            {**_common_claims(), "status": "EXECUTED", **alpha_contract, "rules": alpha["rules"]},
        )
        context.write_json(
            root / "alpha_proxy_evaluation_report.json",
            {
                **_common_claims(),
                "status": "EXECUTED",
                **alpha_contract,
                "mapping": alpha["mapping"],
                "daily_pnl": alpha["daily_pnl"],
                "evaluation": alpha["evaluation"],
                "sensitivities": alpha["sensitivities"],
            },
        )
        context.write_json(
            root / "alpha_proxy_qualified_report.json",
            {**_common_claims(), "status": "EXECUTED", **alpha_contract, "qualified": alpha["qualified"]},
        )
        alpha_status = "EXECUTED"
    else:
        reason = "JULY_GATE_NOT_PASSED" if locked_result is not None else "JULY_NOT_EXECUTED"
        for name in (
            "alpha_proxy_rules_report.json",
            "alpha_proxy_evaluation_report.json",
            "alpha_proxy_qualified_report.json",
        ):
            context.write_json(
                root / name,
                {**_common_claims(), "status": "NOT_EXECUTED", "reason": reason, **alpha_contract},
            )
        alpha_status = "NOT_EXECUTED"

    status = "DEVELOPMENT_EDGE_NOT_FOUND" if not frozen else "COMPLETED"
    if not frozen:
        summary = (
            "All 18 registered Jan-Apr 2024 V2 development configurations completed; none passed the "
            "development gate, so validation, July, and Alpha were not opened."
        )
    elif selected is None:
        summary = (
            "At most three V2 development candidates were frozen and validated, but none passed validation; "
            "July and Alpha were not opened."
        )
    else:
        summary = (
            f"The conditional V2 study completed through July ({'PASS' if locked_gate and locked_gate['passed'] else 'FAIL'}); "
            f"Alpha status: {alpha_status}."
        )
    preservation_after = preservation_snapshot(repository)
    if preservation_after != preservation_before:
        raise RuntimeError("V1 or immutable source changed during V2 execution")
    context.write_json(
        root / "preservation_report.json",
        {
            **_common_claims(),
            "before": preservation_before,
            "after": preservation_after,
            "preserved": True,
        },
    )
    final = {
        **_common_claims(),
        "status": status,
        "summary": summary,
        "tests_passed": True,
        "study_executed": True,
        "development_configuration_count": len(development_reports),
        "development_candidate_count": len(frozen),
        "development_failure_classifications": dict(sorted(diagnosis_counts.items())),
        "validation_selected_variant_id": selected[0].variant_id if selected else None,
        "locked_test_status": (
            "PASS" if locked_gate and locked_gate["passed"] else "FAIL" if locked_gate else "NOT_EXECUTED"
        ),
        "alpha_proxy_status": alpha_status,
        "preservation": preservation_after,
        "artifact_index": sorted(
            {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()}
            | {"final_report.json"}
        ),
    }
    context.write_json(root / "final_report.json", final)
    return {
        "status": status,
        "summary": summary,
        "finalReportPath": str(root / "final_report.json"),
        "testsPassed": True,
        "studyExecuted": True,
    }


def verify_and_run_sealed_v2_study(**kwargs: Any) -> dict[str, Any]:
    repository = Path(kwargs.get("repository_root", ".")).resolve()
    preflight = execute_preflight(repository)
    if not preflight["tests_passed"]:
        return {
            "status": "FAILED",
            "summary": "V2 focused/full tests, compileall, diff check, or preservation checks failed; the real study was not executed.",
            "finalReportPath": None,
            "testsPassed": False,
            "studyExecuted": False,
            "preflight": preflight,
        }
    return run_sealed_v2_study(**kwargs, preflight_evidence=preflight)
