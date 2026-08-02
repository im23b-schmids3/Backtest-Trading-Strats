from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research_pipeline.phase_f1.models import MasterStep
from research_pipeline.phase_f1.service import MasterPipelineService
from research_pipeline.research.models import ResearchArtifact
from research_pipeline.research.services import PhaseCService
from research_pipeline.value_area_trap.data import AggregateTradeManifest


ROOT = Path(__file__).parents[2]
SPEC = ROOT / "research_registry/spec_drafts/ValueAreaTrap_vphase-b-1.yaml"
DATASET_HASH = "908a22b85825a2c58cdf60d748500d403c16e57b52648a2376290547088f2b10"


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "normalized" / "BTCUSDT" / DATASET_HASH
    root.mkdir(parents=True)
    manifest = AggregateTradeManifest.model_construct(
        date_start=date(2026, 4, 1), date_end=date(2026, 4, 30),
        retrieved_at=datetime(2026, 5, 1, tzinfo=timezone.utc), source_files=["archive.zip"],
        source_file_hashes={"archive.zip": "a" * 64}, normalized_dataset_hash=DATASET_HASH,
        row_count=41_544_041, duplicate_count=0, manifest_hash="pending",
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = MasterPipelineService._manifest_integrity_hash(
        AggregateTradeManifest.model_validate(payload)
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    # Creation-time persistence intentionally needs only the immutable paths;
    # real adapter/schema validation remains separately tested.
    (root / "aggregate_trades.parquet").write_bytes(b"fixture-parquet-placeholder")
    return path


def _intake(tmp_path: Path) -> Path:
    path = tmp_path / "intake.json"
    path.write_text(json.dumps({
        "strategy_name": "ValueAreaTrap", "description": "Validate durable real-data provenance for the reference strategy.",
        "markets": ["BTCUSDT"], "timeframes": ["5m"], "entry_logic": ["documented"], "exit_logic": ["documented"],
    }), encoding="utf-8")
    return path


def _started(tmp_path: Path):
    manifest = _manifest(tmp_path)
    service = MasterPipelineService(tmp_path / "registry.sqlite3", tmp_path)
    options = service.input_model(_intake(tmp_path), tmp_path, registry_path=tmp_path / "registry.sqlite3", data_manifest_path=manifest, allow_proxy_data=True)
    started = service.start(options.model_copy(update={"prebuilt_spec_path": str(SPEC)}))
    return service, started, manifest


def test_manifest_backed_run_forces_and_persists_real_data_context(tmp_path: Path):
    service, started, manifest = _started(tmp_path)
    run = service.registry.get_master_run(started["run_id"])
    context = run["resume_state_json"]["real_data_context"]
    assert run["resume_state_json"]["options"]["mode"] == "real_run"
    assert run["resume_state_json"]["options"]["dry_run"] is False
    assert context["execution_mode"] == "REAL_DATA"
    assert context["manifest_path"] == str(manifest.resolve())
    assert context["dataset_hash"] == DATASET_HASH
    assert context["normalized_artifact_paths"]["aggregate_trades_parquet"].endswith("aggregate_trades.parquet")
    # Both resume-time provenance reads use durable state, not caller options.
    assert service._persisted_real_data_context(started["run_id"]).dataset_hash == DATASET_HASH
    assert service._persisted_real_data_context(started["run_id"]).dataset_hash == DATASET_HASH


def test_manifest_context_fails_closed_when_missing_or_changed(tmp_path: Path):
    service, started, manifest = _started(tmp_path)
    manifest.unlink()
    with pytest.raises(Exception, match="manifest is unavailable"):
        service._persisted_real_data_context(started["run_id"])

    service, started, manifest = _started(tmp_path / "changed")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["provider"] = "changed provider"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="file hash changed"):
        service._persisted_real_data_context(started["run_id"])


def test_real_cache_rejects_fixture_baseline_and_uses_run_specific_root(tmp_path: Path):
    input_path = tmp_path / "input.json"
    cache_context = {"run_id": "f1-ValueAreaTrap-test", "execution_mode": "REAL_DATA", "dataset_hash": DATASET_HASH,
                     "specification_hash": "spec", "adapter_identity": "adapter"}
    input_path.write_text(json.dumps({"pipeline_cache_context": cache_context}), encoding="utf-8")
    artifact = ResearchArtifact(
        experiment_id="fixture", strategy_id="ValueAreaTrap", strategy_version="phase-b-1", phase="baseline",
        experiment_dir=str(tmp_path / "research_runs" / "ValueAreaTrap" / "phase-b-1" / "baseline"),
        input_path=str(input_path), metrics_path=str(tmp_path / "metrics.json"), dataset_hash="dataset-phase-c",
        split_hash="split", status="COMPLETED", command=["synthetic-fixture", "strong-stable", "baseline"],
        metrics={"execution_mode": "FIXTURE", "dataset_hash": "dataset-phase-c"},
    )
    phase_c = PhaseCService(tmp_path / "registry.sqlite3", repository_root=tmp_path, master_run_id=cache_context["run_id"], cache_context=cache_context)
    phase_c._strategy = lambda _: {"version": "phase-b-1"}  # type: ignore[method-assign]
    assert not phase_c._cached_baseline_matches_context(artifact)
    assert phase_c._root("ValueAreaTrap", "baseline") == (tmp_path / "research_runs" / "ValueAreaTrap" / cache_context["run_id"] / "research" / "baseline")
