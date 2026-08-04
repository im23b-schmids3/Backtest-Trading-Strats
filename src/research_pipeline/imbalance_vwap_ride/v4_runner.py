from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import code_hash, sha256_file, sha256_value
from .v4_adapter import ImbalanceVWAPRideV4Adapter
from .v4_alpha import run_v4_alpha_proxy
from .v4_artifacts import ImmutableV4ArtifactStore, V4ArtifactContext
from .v4_data import (
    acquire_and_normalize_v4_phase,
    build_v4_phase_footprint_dataset,
    load_v4_bars,
    load_v4_footprints,
)
from .v4_models import (
    ADAPTER_ID,
    PHASE_A_MONTHS,
    PHASE_B_MONTHS,
    SPEC_VERSION,
    STRATEGY_ID,
    candidate_registry_hash,
    candidate_registry_payload,
    phase_a_gate,
    phase_b_gate,
    preregistered_candidates,
    rank_phase_a_candidates,
)

FOCUSED_TESTS = ("tests/research_pipeline/test_imbalance_vwap_ride_v4.py",)


def _existing_final(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / "study-manifest.json"
    final_path = root / "final_report.json"
    if not manifest_path.exists() and not final_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError(f"immutable V4 study artifact collision: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"immutable V4 study identity collision for {key}: {root}")
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


def _run_command(command: list[str], root: Path, *, timeout: int) -> dict[str, Any]:
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
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
        "passed": completed.returncode == 0,
    }


def execute_v4_preflight(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    checks = [
        _run_command([sys.executable, "-m", "pytest", *FOCUSED_TESTS, "-q"], root, timeout=600),
        _run_command(
            [sys.executable, "-m", "pytest", "tests/research_pipeline", "-q"],
            root,
            timeout=1_800,
        ),
        _run_command(
            [sys.executable, "-m", "compileall", "src/research_pipeline"], root, timeout=300
        ),
        _run_command(["git", "diff", "--check"], root, timeout=300),
    ]
    return {
        "preflight_version": "imbalance-vwap-ride-btc-long-only-v4-preflight-1",
        "checks": checks,
        "tests_passed": all(item["passed"] for item in checks),
        "real_study_executed": False,
    }


def _tree_digest(paths: list[Path], root: Path) -> dict[str, Any]:
    records: list[str] = []
    for base in paths:
        if base.is_file():
            candidates = [base]
        elif base.is_dir():
            candidates = sorted(item for item in base.rglob("*") if item.is_file())
        else:
            candidates = []
        for path in candidates:
            records.append(
                f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{sha256_file(path)}"
            )
    return {
        "file_count": len(records),
        "tree_digest": hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(),
    }


def preservation_snapshot(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = root / "src" / "research_pipeline" / "imbalance_vwap_ride"
    protected_source = [
        path
        for path in sorted(package.glob("*.py"))
        if not path.name.startswith("v4_") and path.name != "__init__.py"
    ]
    protected_runs = [
        root / "research_runs" / "ImbalanceVWAPRide.BTC_EXPLORATORY",
        root / "research_runs" / "ImbalanceVWAPRide.BTC_MACRO_BINS_V2_EXPLORATORY",
        root / "research_runs" / "ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY",
    ]
    return {
        "protected_source": _tree_digest(protected_source, root),
        "v1_v2_v3_runs": _tree_digest(protected_runs, root),
    }


def _persist_candidate_result(
    store: ImmutableV4ArtifactStore,
    candidate_id: str,
    result: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    prefix = f"{phase.lower()}/candidates/{candidate_id}"
    store.write_parquet(f"{prefix}/events.parquet", result["events"], phase=phase)
    store.write_parquet(f"{prefix}/zones.parquet", result["zones"], phase=phase)
    store.write_parquet(f"{prefix}/trades.parquet", result["trades"], phase=phase)
    report = {
        "candidate_id": candidate_id,
        "phase": phase,
        "parameters": result["parameters"],
        "metrics": result["metrics"],
        "subperiods": result["subperiods"],
        "funnel": result["funnel"],
        "events_hash": sha256_value(result["events"]),
        "zones_hash": sha256_value(result["zones"]),
        "trades_hash": sha256_value(result["trades"]),
        "result_hash": sha256_value(
            {
                "parameters": result["parameters"],
                "metrics": result["metrics"],
                "subperiods": result["subperiods"],
                "funnel": result["funnel"],
            }
        ),
    }
    store.write_json(f"{prefix}/report.json", report, phase=phase)
    return report


def run_sealed_v4_study(
    *,
    artifact_root: str | Path = "research_runs",
    repository_root: str | Path = ".",
    data_cache_root: str | Path = "data/value_area_trap",
    footprint_cache_root: str | Path = "data/imbalance_vwap_ride/v4/footprints",
    batch_size: int = 1_000_000,
    allow_authorized_downloads: bool = True,
    alpha_rules_artifact: str | Path | None = None,
    preflight_evidence: dict[str, Any] | None = None,
    phase_a_only: bool = False,
) -> dict[str, Any]:
    """Execute V4 lazily: Phase B is not acquired unless Phase A selects one candidate."""

    repository = Path(repository_root).resolve()
    if preflight_evidence is not None and not preflight_evidence.get("tests_passed"):
        raise ValueError("V4 real study requires a passing preflight")
    preservation_before = preservation_snapshot(repository)
    spec_path = repository / ".smithers" / "specs" / "imbalance-vwap-ride-btc-long-only-v4.md"
    phase_a_source, phase_a_manifest, phase_a_download = acquire_and_normalize_v4_phase(
        repository / data_cache_root,
        phase="PHASE_A",
        allow_authorized_downloads=allow_authorized_downloads,
    )
    phase_a_footprint = build_v4_phase_footprint_dataset(
        phase_a_source,
        repository / footprint_cache_root,
        phase="PHASE_A",
        batch_size=batch_size,
    )
    # This sealed scope hash is known before opening Phase B and therefore keeps
    # the study ID deterministic without violating Phase-A early termination.
    phase_b_scope_hash = sha256_value(
        {"phase": "PHASE_B", "symbol": "BTCUSDT", "months": list(PHASE_B_MONTHS)}
    )
    identity = {
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "specification_version": SPEC_VERSION,
        "specification_hash": sha256_file(spec_path),
        "candidate_registry_hash": candidate_registry_hash(),
        "code_hash": code_hash(repository),
        "phase_a_dataset_hash": phase_a_footprint["footprint_dataset_hash"],
        "phase_a_source_manifest_hash": phase_a_manifest.manifest_hash,
        "phase_b_dataset_hash": phase_b_scope_hash,
        "phase_b_source_manifest_hash": phase_b_scope_hash,
    }
    store = ImmutableV4ArtifactStore(artifact_root, identity)
    final_path = store.root / "final_report.json"
    if final_path.exists():
        health = store.validate_health()
        if not health["valid"]:
            raise ValueError("existing deterministic V4 run is not healthy")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if phase_a_only:
            return {
                "status": final["status"],
                "summary": final["summary"],
                "selectedCandidateId": final.get("selected_candidate_id"),
                "frozenCandidateHash": final.get("frozen_candidate_hash"),
                "phaseAComplete": bool(final.get("phase_a_complete")),
            }
        return {
            "status": final["status"],
            "summary": final["summary"],
            "finalReportPath": str(final_path),
            "testsPassed": bool(final.get("tests_passed")),
            "studyExecuted": bool(final.get("study_executed")),
        }
    store.write_json(
        "candidate_registry.json",
        {
            "sealed_before_phase_a_results": True,
            "configuration_count": 4,
            "cartesian_search": False,
            "registry_hash": candidate_registry_hash(),
            "registry": candidate_registry_payload(),
        },
        phase="PHASE_A",
    )
    store.write_json("phase_a/source_acquisition.json", phase_a_download, phase="PHASE_A")
    store.write_json("phase_a/footprint_manifest.json", phase_a_footprint, phase="PHASE_A")
    adapter = ImbalanceVWAPRideV4Adapter()
    store.write_json("adapter_capabilities.json", adapter.capabilities(), phase="PHASE_A")
    bars_a = load_v4_bars(phase_a_footprint)
    footprints_a = load_v4_footprints(phase_a_footprint)
    candidate_runs: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for config in preregistered_candidates():
        result = adapter.run_loaded(
            bars=bars_a,
            footprints=footprints_a,
            config=config,
            phase="PHASE_A",
        )
        report = _persist_candidate_result(store, config.candidate_id, result, phase="PHASE_A")
        candidate_runs.append((config, result, report))
    if len(candidate_runs) != 4 or [item[0].candidate_id for item in candidate_runs] != [
        item["candidate_id"] for item in candidate_registry_payload()
    ]:
        raise AssertionError("V4 Phase A did not execute the sealed registry exactly once")
    ranked = rank_phase_a_candidates(
        [(config, result["metrics"]) for config, result, _ in candidate_runs]
    )
    serializable_ranking = [
        {key: value for key, value in item.items() if key not in {"config", "metrics"}}
        | {"metrics": item["metrics"]}
        for item in ranked
    ]
    store.write_json(
        "phase_a/selection_report.json",
        {
            "selection_method": "PRE_REGISTERED_ROBUSTNESS_RANKING",
            "selection_by_pnl_only": False,
            "candidate_gates": {
                config.candidate_id: phase_a_gate(result["metrics"])
                for config, result, _ in candidate_runs
            },
            "rank_trace": serializable_ranking,
        },
        phase="PHASE_A",
    )
    selected_candidate_id: str | None = None
    frozen_candidate_hash: str | None = None
    phase_b_execution_count = 0
    if not ranked:
        status = "PHASE_A_NO_ROBUST_CANDIDATE"
        summary = "All four sealed Phase A candidates completed; none passed every robustness gate, so Phase B and Alpha were not opened."
        store.write_json(
            "phase_b/status.json",
            {"status": "NOT_OPENED", "reason": status, "execution_count": 0},
            phase="PHASE_B",
        )
        alpha = {"status": "NOT_EXECUTED", "reason": status, "alpha_executed": False}
    else:
        selected = ranked[0]
        selected_config = selected["config"]
        selected_report = next(
            report for config, _, report in candidate_runs if config.candidate_id == selected_config.candidate_id
        )
        frozen = store.freeze_candidate(
            selected_config,
            selected["metrics"],
            selected["rank_trace"],
            phase_a_result_hash=selected_report["result_hash"],
        )
        selected_candidate_id = selected_config.candidate_id
        frozen_candidate_hash = frozen["frozen_candidate_hash"]
        if phase_a_only:
            status = "PHASE_A_CANDIDATE_FROZEN"
            summary = (
                "All four sealed Phase A candidates completed; the deterministic "
                "robustness ranking selected and immutably froze one candidate. "
                "Phase B and Alpha were not opened."
            )
            store.write_json(
                "phase_b/status.json",
                {
                    "status": "NOT_OPENED",
                    "reason": "PHASE_A_ONLY_REQUEST",
                    "execution_count": 0,
                    "frozen_candidate_hash": frozen_candidate_hash,
                },
                phase="PHASE_B",
            )
            alpha = {
                "status": "NOT_EXECUTED",
                "reason": "PHASE_A_ONLY_REQUEST",
                "alpha_executed": False,
            }
        else:
            phase_b_source, phase_b_manifest, phase_b_download = acquire_and_normalize_v4_phase(
                repository / data_cache_root,
                phase="PHASE_B",
                allow_authorized_downloads=allow_authorized_downloads,
            )
            phase_b_footprint = build_v4_phase_footprint_dataset(
                phase_b_source,
                repository / footprint_cache_root,
                phase="PHASE_B",
                batch_size=batch_size,
            )
            store.write_json("phase_b/source_acquisition.json", phase_b_download, phase="PHASE_B")
            store.write_json("phase_b/footprint_manifest.json", phase_b_footprint, phase="PHASE_B")
            store.begin_phase_b(frozen["frozen_candidate_hash"])
            phase_b_execution_count = 1
            bars_b = load_v4_bars(phase_b_footprint)
            footprints_b = load_v4_footprints(phase_b_footprint)
            result_b = adapter.run_loaded(
                bars=bars_b,
                footprints=footprints_b,
                config=selected_config,
                phase="PHASE_B",
            )
            result_b["metrics"]["hashes_valid"] = bool(
                phase_b_manifest.manifest_hash and phase_b_footprint["footprint_dataset_hash"]
            )
            result_b["metrics"]["costs_valid"] = all(
                all(name in trade for name in ("fees", "slippage_cost", "total_costs", "net_pnl"))
                for trade in result_b["trades"]
            )
            _persist_candidate_result(store, selected_config.candidate_id, result_b, phase="PHASE_B")
            locked = phase_b_gate(result_b["metrics"])
            store.write_json("phase_b/locked_test_report.json", locked, phase="PHASE_B")
            if locked["status"] != "LOCKED_TEST_PASSED":
                status = "LOCKED_TEST_FAILED"
                summary = "The single frozen Phase B candidate completed exactly once and failed one or more locked-test gates."
                alpha = {"status": "NOT_EXECUTED", "reason": status, "alpha_executed": False}
            else:
                rules = None
                if alpha_rules_artifact is not None:
                    rules = json.loads(Path(alpha_rules_artifact).read_text(encoding="utf-8"))
                alpha = run_v4_alpha_proxy(
                    result_b["trades"],
                    bars_b,
                    locked_test_status=locked["status"],
                    phase_b_execution_count=1,
                    frozen_candidate_valid=True,
                    rules_artifact=rules,
                )
                status = "LOCKED_TEST_PASSED"
                summary = "The single frozen candidate passed the strategy-specific locked test; this remains non-confirmatory evidence."
    store.write_json("alpha/status.json", alpha, phase="ALPHA")
    preservation_after = preservation_snapshot(repository)
    if preservation_after != preservation_before:
        raise RuntimeError("V1, V2, or V3 source/artifacts changed during V4")
    final = {
        "status": status,
        "summary": summary,
        "tests_passed": bool(preflight_evidence is None or preflight_evidence.get("tests_passed")),
        "study_executed": True,
        "strategy_id": STRATEGY_ID,
        "adapter_id": ADAPTER_ID,
        "phase_a_candidate_count": 4,
        "phase_a_complete": True,
        "selected_candidate_id": selected_candidate_id,
        "frozen_candidate_hash": frozen_candidate_hash,
        "phase_b_execution_count": phase_b_execution_count,
        "confirmation_evidence": False,
        "optimization_claimed": False,
        "alpha_status": alpha["status"],
        "preservation": preservation_after,
    }
    store.write_json("final_report.json", final, phase="FINAL")
    store.seal_integrity_manifest()
    if phase_a_only:
        return {
            "status": status,
            "summary": summary,
            "selectedCandidateId": selected_candidate_id,
            "frozenCandidateHash": frozen_candidate_hash,
            "phaseAComplete": True,
        }
    return {
        "status": status,
        "summary": summary,
        "finalReportPath": str(final_path),
        "testsPassed": bool(final["tests_passed"]),
        "studyExecuted": True,
    }


def verify_and_run_sealed_v4_study(**kwargs: Any) -> dict[str, Any]:
    repository = Path(kwargs.get("repository_root", ".")).resolve()
    preflight = execute_v4_preflight(repository)
    if not preflight["tests_passed"]:
        return {
            "status": "FAILED",
            "summary": "V4 focused/full tests, compileall, or diff check failed; the real study was not executed.",
            "finalReportPath": None,
            "testsPassed": False,
            "studyExecuted": False,
            "preflight": preflight,
        }
    return run_sealed_v4_study(**kwargs, preflight_evidence=preflight)


__all__ = [
    "V4ArtifactContext",
    "_existing_final",
    "execute_v4_preflight",
    "preservation_snapshot",
    "run_sealed_v4_study",
    "verify_and_run_sealed_v4_study",
]
