from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactContext, code_hash, sha256_file, sha256_value, utc_now, write_bytes_once
from .v3_adapter import ImbalanceVWAPRideV3Adapter
from .v3_data import (
    acquire_and_normalize_v3_data,
    build_v3_footprint_dataset,
    load_v3_bars,
    load_v3_footprints,
    validate_v3_footprint_dataset,
    validate_v3_source_manifest,
)
from .v3_models import (
    ADAPTER_ID,
    AUTHORIZED_MONTHS,
    BASELINE,
    EVIDENCE_LABEL,
    PARAMETER_REGISTRY,
    PERIOD_LABEL,
    SELECTION_METHOD,
    SPEC_VERSION,
    STRATEGY_ID,
    ImbalanceVWAPRideV3Config,
    freeze_passing_candidates,
    preregistered_variants,
    promotion_gate,
)

FOCUSED_TESTS = (
    "tests/research_pipeline/test_imbalance_vwap_ride.py",
    "tests/research_pipeline/test_imbalance_vwap_ride_v2.py",
    "tests/research_pipeline/test_imbalance_vwap_ride_v3.py",
)
V1_RUN_RELATIVE = Path("research_runs/ImbalanceVWAPRide.BTC_EXPLORATORY/ff0afc85ef4b46c0bf671cfb")
V1_FILE_COUNT = 171
V1_TREE_DIGEST = "726c383e2ffde79bd4504889ca222871bf052345dc7cd3c7f93a7eb3448ef182"
V2_RUN_RELATIVE = Path(
    "research_runs/ImbalanceVWAPRide.BTC_MACRO_BINS_V2_EXPLORATORY/a77a41379b37cad665f3b721"
)
V2_FILE_COUNT = 254
V2_TREE_DIGEST = "62bdaa844cf153ca6aadd917bc7e8041531a16b899833eec60aa7f49e7042ca5"
JAN_JUL_SOURCE_RELATIVE = Path(
    "data/value_area_trap/normalized/BTCUSDT/"
    "c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc"
)
JAN_JUL_SOURCE_FILE_COUNT = 8
JAN_JUL_SOURCE_TREE_DIGEST = "d61d65a589f1d950b57e3b0d3fdb4d635c2f10a5b098465816e98d498af133ad"


def _common_claims() -> dict[str, Any]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "period_label": PERIOD_LABEL,
        "confirmation_evidence": False,
        "optimization_claimed": False,
        "external_confirmation_required": True,
        "selection_method": SELECTION_METHOD,
        "direction": "LONG_ONLY",
        "raw_trades_transmitted_externally": False,
        "live_orders_used": False,
        "alpha_executed_from_development_data": False,
    }


@dataclass(frozen=True)
class V3ArtifactContext(ArtifactContext):
    def metadata(self) -> dict[str, str]:
        return {
            **ArtifactContext.metadata(self),
            "period_label": PERIOD_LABEL,
            "confirmation_evidence": "false",
            "optimization_claimed": "false",
            "external_confirmation_required": "true",
            "selection_method": SELECTION_METHOD,
            "direction": "LONG_ONLY",
        }

    def envelope(self, payload: Any) -> dict[str, Any]:
        base = {**ArtifactContext.metadata(self), **_common_claims()}
        if isinstance(payload, dict):
            return {**base, **payload}
        return {**base, "payload": payload}


def tree_digest(path: str | Path) -> tuple[int, str]:
    root = Path(path).resolve()
    records: list[str] = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        records.append(
            f"{item.relative_to(root).as_posix()}\t{item.stat().st_size}\t{sha256_file(item)}"
        )
    return len(records), hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def preservation_snapshot(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    v1_count, v1_digest = tree_digest(root / V1_RUN_RELATIVE)
    v2_count, v2_digest = tree_digest(root / V2_RUN_RELATIVE)
    source_count, source_digest = tree_digest(root / JAN_JUL_SOURCE_RELATIVE)
    return {
        "v1_file_count": v1_count,
        "v1_tree_digest": v1_digest,
        "v1_preserved": v1_count == V1_FILE_COUNT and v1_digest == V1_TREE_DIGEST,
        "v2_file_count": v2_count,
        "v2_tree_digest": v2_digest,
        "v2_preserved": v2_count == V2_FILE_COUNT and v2_digest == V2_TREE_DIGEST,
        "jan_jul_source_file_count": source_count,
        "jan_jul_source_tree_digest": source_digest,
        "jan_jul_source_preserved": (
            source_count == JAN_JUL_SOURCE_FILE_COUNT
            and source_digest == JAN_JUL_SOURCE_TREE_DIGEST
        ),
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
                ".tmp/pytest-imbalance-v3-focused",
                *FOCUSED_TESTS,
            ],
            1200,
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                ".tmp/pytest-imbalance-v3-full",
                "tests/research_pipeline",
            ],
            2400,
        ),
        ([sys.executable, "-m", "compileall", "-q", "src/research_pipeline", "tests/research_pipeline"], 600),
        (["git", "diff", "--check"], 120),
    ]
    results = [_run_command(command, root, timeout=timeout) for command, timeout in commands]
    preservation = preservation_snapshot(root)
    tests_passed = all(item["passed"] for item in results) and all(
        preservation[name]
        for name in ("v1_preserved", "v2_preserved", "jan_jul_source_preserved")
    )
    return {
        "preflight_version": "imbalance-vwap-ride-btc-long-only-v3-preflight-1",
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
        raise RuntimeError("V3 execution is fail-closed until every preflight check passes")
    preservation = preflight.get("preservation", {})
    if any(
        preservation.get(name) is not True
        for name in ("v1_preserved", "v2_preserved", "jan_jul_source_preserved")
    ):
        raise RuntimeError("V1, V2, or the Jan-Jul source was not preserved")
    if preflight.get("code_hash") != code_hash(root):
        raise RuntimeError("preflight code hash no longer matches the V3 implementation")


def _variant_context(base: ArtifactContext, config: ImbalanceVWAPRideV3Config) -> V3ArtifactContext:
    return V3ArtifactContext(
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
    config: ImbalanceVWAPRideV3Config,
    result: dict[str, Any],
) -> dict[str, Any]:
    context = _variant_context(base_context, config)
    metrics = result["metrics"]
    gate = promotion_gate(metrics)
    context.write_json(
        target / "configuration.json",
        {
            "variant_id": config.variant_id,
            "parameters": config.parameter_payload(),
            "configuration_hash": sha256_value(config.parameter_payload()),
            "registry_position": [item.variant_id for item in preregistered_variants()].index(config.variant_id),
        },
    )
    context.write_parquet(target / "events.parquet", result["events"])
    context.write_parquet(target / "zones.parquet", result["zones"])
    context.write_parquet(target / "trades.parquet", result["trades"])
    context.write_json(target / "monthly_report.json", {"variant_id": config.variant_id, "months": metrics["months"]})
    context.write_json(target / "subperiod_report.json", {"variant_id": config.variant_id, "subperiods": result["subperiods"]})
    context.write_json(target / "funnel_report.json", {"variant_id": config.variant_id, **result["funnel"]})
    context.write_json(
        target / "zone_funnel_report.json",
        {
            "variant_id": config.variant_id,
            **{
                name: metrics[name]
                for name in (
                    "imbalance_sequences",
                    "zones_created",
                    "vwap_qualified_zones",
                    "move_away_confirmed_zones",
                    "retest_triggers",
                    "terminal_expired",
                    "terminal_invalidated",
                    "terminal_traded",
                    "terminal_superseded",
                    "active_zones_at_end",
                )
            },
        },
    )
    context.write_json(
        target / "gross_net_report.json",
        {
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
        target / "cost_risk_report.json",
        {
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
        target / "long_only_report.json",
        {
            "variant_id": config.variant_id,
            **metrics["long_only_reconciliation"],
            "all_zone_directions_long": all(zone["direction"] == "LONG" for zone in result["zones"]),
            "all_trade_directions_long": all(trade["direction"] == "LONG" for trade in result["trades"]),
            "all_event_directions_long_or_absent": all(
                event.get("direction") in {None, "LONG"} for event in result["events"]
            ),
        },
    )
    context.write_json(target / "promotion_gate.json", {"variant_id": config.variant_id, **gate})
    report = {
        **_common_claims(),
        "variant_id": config.variant_id,
        "parameters": config.parameter_payload(),
        "metrics": metrics,
        "subperiods": result["subperiods"],
        "promotion_gate": gate,
    }
    context.write_json(target / "report.json", report)
    return report


def _existing_final(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / "study-manifest.json"
    final_path = root / "final_report.json"
    if not manifest_path.exists() and not final_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError(f"immutable V3 study artifact collision: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"immutable V3 study identity collision for {key}: {root}")
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


def _content_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.name != "final_report.json"
    }


def run_sealed_v3_study(
    *,
    artifact_root: str | Path = "research_runs",
    repository_root: str | Path = ".",
    data_cache_root: str | Path = "data/value_area_trap",
    footprint_cache_root: str | Path = "data/imbalance_vwap_ride/v3/footprints",
    batch_size: int = 1_000_000,
    allow_authorized_downloads: bool = True,
    preflight_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    _validate_preflight(preflight_evidence or {}, repository)
    preservation_before = preservation_snapshot(repository)
    source_path, source_manifest, download_manifest = acquire_and_normalize_v3_data(
        repository / data_cache_root,
        allow_authorized_downloads=allow_authorized_downloads,
    )
    source_validation = validate_v3_source_manifest(source_path, verify_archives=False)
    if not source_validation["valid"]:
        raise RuntimeError("V3 normalized source failed validation")
    footprint_manifest = build_v3_footprint_dataset(
        source_path,
        repository / footprint_cache_root,
        batch_size=batch_size,
    )
    footprint_validation = validate_v3_footprint_dataset(footprint_manifest["footprint_root"])
    if not footprint_validation["valid"]:
        raise RuntimeError("V3 exact footprint dataset failed validation")

    variants = preregistered_variants()
    registry = [item.model_dump(mode="json") for item in variants]
    spec_path = repository / ".smithers" / "specs" / "imbalance-vwap-ride-btc-long-only-v3.md"
    identity = {
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "dataset_hash": source_manifest.normalized_dataset_hash,
        "source_manifest_hash": source_manifest.manifest_hash,
        "source_manifest_file_sha256": sha256_file(source_path),
        "footprint_dataset_hash": footprint_manifest["footprint_dataset_hash"],
        "specification_hash": sha256_file(spec_path),
        "parameter_hash": sha256_value(registry),
        "code_hash": code_hash(repository),
        "specification_version": SPEC_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "period_label": PERIOD_LABEL,
        "selection_method": SELECTION_METHOD,
    }
    run_id = sha256_value(identity)[:24]
    root = Path(artifact_root).resolve() / STRATEGY_ID / run_id
    existing = _existing_final(root, identity)
    if existing is not None:
        if preservation_snapshot(repository) != preservation_before:
            raise RuntimeError("preservation changed during deterministic V3 rerun validation")
        return existing

    manifest_path = root / "study-manifest.json"
    timestamp = utc_now()
    if manifest_path.exists():
        timestamp = json.loads(manifest_path.read_text(encoding="utf-8"))["artifact_timestamp"]
    context = V3ArtifactContext(
        run_id=run_id,
        dataset_hash=source_manifest.normalized_dataset_hash,
        source_manifest_hash=source_manifest.manifest_hash,
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
                "authorized_months": list(AUTHORIZED_MONTHS),
                "source_manifest_path": str(source_path),
                "footprint_manifest_path": str(Path(footprint_manifest["footprint_root"]) / "manifest.json"),
                "preflight": preflight_evidence,
                "preservation_before": preservation_before,
            },
        )
    context.write_json(root / "source_download_manifest.json", download_manifest)
    context.write_json(
        root / "aggregate_trade_manifest.json",
        {
            "manifest_path": str(source_path),
            "manifest_file_sha256": sha256_file(source_path),
            "manifest": source_manifest.model_dump(mode="json"),
        },
    )
    context.write_json(
        root / "normalized_dataset_manifest.json",
        {
            **{key: value for key, value in source_validation.items() if key != "manifest"},
            "content_addressed": True,
            "normalization_local_only": True,
        },
    )
    context.write_json(root / "footprint_manifest.json", footprint_manifest)
    context.write_json(root / "footprint_validation_report.json", footprint_validation)
    context.write_json(root / "preflight_validation.json", preflight_evidence)
    context.write_json(
        root / "specification.json",
        {
            "specification_version": SPEC_VERSION,
            "sealed_specification_path": str(spec_path),
            "sealed_specification_sha256": sha256_file(spec_path),
        },
    )
    baseline_payload = {
        **_common_claims(),
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "variant_id": BASELINE.variant_id,
        "parameters": BASELINE.parameter_payload(),
        "configuration_hash": sha256_value(BASELINE.parameter_payload()),
    }
    write_bytes_once(
        root / "baseline.yaml",
        yaml.safe_dump(baseline_payload, sort_keys=True).encode("utf-8"),
    )
    context.write_json(
        root / "parameter_registry.json",
        {
            "search_type": "EXACT_SEVEN_PRE_REGISTERED_LONG_ONLY_OAT",
            "configuration_count": 7,
            "stable_order": True,
            "cartesian_search": False,
            "random_search": False,
            "hidden_variations": False,
            "registry": registry,
            "families": [
                {"name": name, "values": [str(value) for value in values]}
                for name, values in PARAMETER_REGISTRY
            ],
        },
    )

    adapter = ImbalanceVWAPRideV3Adapter()
    context.write_json(root / "adapter_capabilities.json", adapter.capabilities())
    bars = load_v3_bars(footprint_manifest)
    if tuple(sorted({str(item["month"]) for item in bars})) != AUTHORIZED_MONTHS:
        raise RuntimeError("V3 footprint bars do not cover the exact authorized six months")
    results_by_id: dict[str, tuple[ImbalanceVWAPRideV3Config, dict[str, Any], dict[str, Any]]] = {}
    for size in (BASELINE.bin_size_usd, ImbalanceVWAPRideV3Config(bin_size_usd="30").bin_size_usd, ImbalanceVWAPRideV3Config(bin_size_usd="75").bin_size_usd):
        footprints = load_v3_footprints(footprint_manifest, size)
        for config in [item for item in variants if item.bin_size_usd == size]:
            result = adapter.run_loaded(bars=bars, footprints=footprints, config=config)
            target = root / "configurations" / config.variant_id
            report = _persist_result(context, target, config, result)
            results_by_id[config.variant_id] = (config, result, report)
        del footprints
    if list(results_by_id) != [
        "baseline",
        "min_bin_volume_btc=20",
        "min_bin_volume_btc=50",
        "vwap_slope_bars=18",
        "vwap_slope_bars=36",
        "bin_size_usd=30",
        "bin_size_usd=75",
    ]:
        raise AssertionError("V3 execution groups were not deterministic")
    ordered = [results_by_id[config.variant_id] for config in variants]
    if len(ordered) != 7:
        raise AssertionError("V3 did not execute exactly seven configurations")
    runs = [
        {
            "variant_id": config.variant_id,
            "parameters": config.parameter_payload(),
            "metrics": result["metrics"],
            "subperiods": result["subperiods"],
            "promotion_gate": report["promotion_gate"],
            "report_path": str(root / "configurations" / config.variant_id / "report.json"),
        }
        for config, result, report in ordered
    ]
    context.write_json(root / "configuration_report.json", {"configuration_count": 7, "runs": runs})
    context.write_json(
        root / "comparison_report.json",
        {
            "stable_registry_order": [config.variant_id for config in variants],
            "baseline_variant_id": "baseline",
            "selection_by_highest_pnl_alone_prohibited": True,
            "runs": runs,
        },
    )
    context.write_json(
        root / "monthly_report.json",
        {"configurations": {item["variant_id"]: item["metrics"]["months"] for item in runs}},
    )
    context.write_json(
        root / "subperiod_report.json",
        {"configurations": {item["variant_id"]: item["subperiods"] for item in runs}},
    )
    context.write_json(
        root / "funnel_report.json",
        {
            "all_reconciled": all(item["metrics"]["funnel_reconciliation"]["reconciles"] for item in runs),
            "configurations": {
                item["variant_id"]: item["metrics"]["funnel_reconciliation"] for item in runs
            },
        },
    )
    context.write_json(
        root / "cost_risk_report.json",
        {
            "configurations": {
                item["variant_id"]: {
                    name: item["metrics"][name]
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
                }
                for item in runs
            }
        },
    )
    context.write_json(
        root / "promotion_gates.json",
        {
            "minimum_trade_count": 48,
            "informative_non_promotable_range": "36-47",
            "configurations": {item["variant_id"]: item["promotion_gate"] for item in runs},
        },
    )
    context.write_json(
        root / "long_only_report.json",
        {
            "all_configurations_reconciled": all(
                item["metrics"]["long_only_reconciliation"]["reconciles"] for item in runs
            ),
            "total_long_trades": sum(item["metrics"]["long_trades"] for item in runs),
            "total_short_trades": 0,
            "short_setups": 0,
            "short_orders": 0,
            "short_fills": 0,
            "short_candidates": 0,
            "short_pnl": "0",
            "configurations": {
                item["variant_id"]: item["metrics"]["long_only_reconciliation"] for item in runs
            },
        },
    )

    candidate_inputs = [(config, result["metrics"]) for config, result, _ in ordered]
    frozen = freeze_passing_candidates(candidate_inputs)
    frozen_payload = [
        {
            "rank": rank,
            "variant_id": config.variant_id,
            "parameters": config.parameter_payload(),
            "development_metrics": metrics,
            "promotion_gate": promotion_gate(metrics),
        }
        for rank, (config, metrics) in enumerate(frozen, start=1)
    ]
    context.write_json(
        root / "frozen_candidates.json",
        {
            "maximum_candidates": 2,
            "candidate_count": len(frozen_payload),
            "deterministic_order": "NET_PF_DESC_DD_ASC_ACTIVITY_DESC_COUNT_DESC_BASELINE_DISTANCE_ASC_LEXICAL",
            "candidates": frozen_payload,
        },
    )
    if frozen:
        validation_preparation = {
            "status": "PREPARED_NOT_EXECUTED",
            "external_validation_required": True,
            "development_months_forbidden_as_confirmation": list(AUTHORIZED_MONTHS),
            "candidate_count": len(frozen_payload),
            "candidates": frozen_payload,
            "alpha_status": "NOT_EXECUTED_EXTERNAL_VALIDATION_REQUIRED",
        }
        status = "COMPLETED"
        summary = (
            f"All seven V3 long-only configurations completed over six development months; "
            f"{len(frozen_payload)} candidate(s) passed and were frozen for external validation only."
        )
        alpha_status = "NOT_EXECUTED_EXTERNAL_VALIDATION_REQUIRED"
    else:
        validation_preparation = {
            "status": "NOT_PREPARED_DEVELOPMENT_EDGE_NOT_FOUND",
            "external_validation_required": True,
            "candidate_count": 0,
            "candidates": [],
            "alpha_status": "NOT_ELIGIBLE_DEVELOPMENT_FAILED",
        }
        status = "DEVELOPMENT_EDGE_NOT_FOUND"
        summary = (
            "All seven registered V3 long-only OAT configurations completed over 2024-08 through "
            "2025-01; none passed every promotion gate, so no candidate was promoted and Alpha is "
            "NOT_ELIGIBLE_DEVELOPMENT_FAILED."
        )
        alpha_status = "NOT_ELIGIBLE_DEVELOPMENT_FAILED"
    context.write_json(root / "validation_preparation.json", validation_preparation)
    context.write_json(
        root / "alpha_classification.json",
        {
            "status": alpha_status,
            "alpha_executed": False,
            "development_data_used_for_alpha": False,
            "reason": "EXTERNAL_CONFIRMATION_REQUIRED" if frozen else "DEVELOPMENT_FAILED_PROMOTION_GATES",
        },
    )

    preservation_after = preservation_snapshot(repository)
    if preservation_after != preservation_before:
        raise RuntimeError("V1, V2, or Jan-Jul data changed during V3 execution")
    context.write_json(
        root / "preservation_report.json",
        {"before": preservation_before, "after": preservation_after, "preserved": True},
    )
    all_funnels = all(item["metrics"]["funnel_reconciliation"]["reconciles"] for item in runs)
    all_long_only = all(item["metrics"]["long_only_reconciliation"]["reconciles"] for item in runs)
    if not all_funnels or not all_long_only:
        raise AssertionError("V3 final reconciliations failed")
    artifact_hashes = _content_hashes(root)
    final = {
        **_common_claims(),
        **identity,
        "status": status,
        "summary": summary,
        "tests_passed": True,
        "study_executed": True,
        "configuration_count": len(runs),
        "authorized_months": list(AUTHORIZED_MONTHS),
        "development_candidate_count": len(frozen_payload),
        "alpha_status": alpha_status,
        "all_funnels_reconciled": all_funnels,
        "all_long_only_reconciled": all_long_only,
        "preservation": preservation_after,
        "deterministic_rerun_verified": True,
        "artifact_content_hashes": artifact_hashes,
        "artifact_index": sorted([*artifact_hashes, "final_report.json"]),
    }
    context.write_json(root / "final_report.json", final)
    expected = {
        "status": status,
        "summary": summary,
        "finalReportPath": str(root / "final_report.json"),
        "testsPassed": True,
        "studyExecuted": True,
    }
    if _existing_final(root, identity) != expected or _existing_final(root, identity) != expected:
        raise AssertionError("V3 deterministic final-report rerun validation failed")
    return expected


def verify_and_run_sealed_v3_study(**kwargs: Any) -> dict[str, Any]:
    repository = Path(kwargs.get("repository_root", ".")).resolve()
    preflight = execute_preflight(repository)
    if not preflight["tests_passed"]:
        return {
            "status": "FAILED",
            "summary": "V3 focused/full tests, compileall, diff check, or V1/V2/data preservation checks failed; the real study was not executed.",
            "finalReportPath": None,
            "testsPassed": False,
            "studyExecuted": False,
            "preflight": preflight,
        }
    return run_sealed_v3_study(**kwargs, preflight_evidence=preflight)
