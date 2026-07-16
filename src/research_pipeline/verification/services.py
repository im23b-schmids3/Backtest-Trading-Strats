from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..controller.pipeline_controller import PipelineController
from ..enums import PipelineState
from ..registry.database import Database
from ..registry.repositories import Registry, now_iso
from .models import VerificationManifest, VerificationOutcome
from .verifier import VerificationRunner, artifact_hash


class VerificationService:
    def __init__(self, registry_path: str | Path | None = None):
        import os
        self.registry_path = Path(registry_path or os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3"))
        self.registry = Registry(Database(self.registry_path))
        self.controller = PipelineController(self.registry)

    def create_manifest(self, strategy_id: str, diagnostic_dir: str | Path | None = None, verification_run_id: str | None = None) -> VerificationManifest:
        strategy = self.registry.get_strategy(strategy_id)
        target_dir = Path(diagnostic_dir or self.registry_path.parent / "verification" / strategy_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest = VerificationManifest(strategy_id=strategy["strategy_id"], strategy_version=strategy["version"],
            implementation_commit=None, verification_run_id=verification_run_id or str(uuid.uuid4()),
            diagnostic_files=[str(target_dir / "diagnostics.json")], approved_invariants_hash=strategy["specification_hash"])
        manifest_path = target_dir / "manifest.yaml"
        raw = manifest.model_dump(mode="json")
        raw["manifest_path"] = str(manifest_path)
        manifest.save(manifest_path)
        return manifest

    def run(self, strategy_id: str, manifest_path: str | Path) -> dict:
        manifest = VerificationManifest.load(manifest_path)
        if manifest.strategy_id != strategy_id:
            raise ValueError("manifest strategy_id does not match requested strategy")
        strategy = self.registry.get_strategy(strategy_id, manifest.strategy_version)
        current = PipelineState(strategy["current_phase"])
        if current == PipelineState.IMPLEMENTATION_VERIFICATION:
            self.controller.transition(strategy_id, PipelineState.TECHNICAL_INTEGRITY_VERIFICATION, "start Phase B.5 technical integrity verification")
        elif current == PipelineState.TECHNICAL_REPAIR_REQUIRED:
            self.controller.transition(strategy_id, PipelineState.TECHNICAL_INTEGRITY_VERIFICATION, "rerun Phase B.5 after bounded repair")
        elif current not in {PipelineState.TECHNICAL_INTEGRITY_VERIFICATION, PipelineState.BASELINE_BACKTEST}:
            raise ValueError(f"verification requires IMPLEMENTATION_VERIFICATION or TECHNICAL_INTEGRITY_VERIFICATION, got {current}")
        existing = self.registry.get_verification(strategy_id, manifest.verification_run_id)
        if existing:
            return existing
        result = VerificationRunner(manifest).run()
        manifest_path = Path(manifest_path).resolve()
        result_path = manifest_path.parent / f"result-{manifest.verification_run_id}.json"
        result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        artifacts = []
        for file_path in manifest.diagnostic_files:
            path = Path(file_path)
            if path.is_file():
                try:
                    row_count = len(json.loads(path.read_text(encoding="utf-8")).get("trades", [])) if path.suffix.lower() == ".json" else 0
                except (OSError, ValueError):
                    row_count = 0
                artifacts.append({"file_path": str(path.resolve()), "file_hash": artifact_hash(path), "row_count": row_count, "created_at": now_iso()})
        payload = result.model_dump(mode="json")
        self.registry.record_verification(payload, {**manifest.model_dump(mode="json"), "manifest_path": str(manifest_path)}, artifacts)
        outcome_to_state = {
            VerificationOutcome.VERIFIED: PipelineState.BASELINE_BACKTEST,
            VerificationOutcome.TECHNICAL_REPAIR_REQUIRED: PipelineState.TECHNICAL_REPAIR_REQUIRED,
            VerificationOutcome.MANUAL_REVIEW_REQUIRED: PipelineState.MANUAL_REVIEW_REQUIRED,
            VerificationOutcome.INSUFFICIENT_DIAGNOSTIC_DATA: PipelineState.INSUFFICIENT_DIAGNOSTIC_DATA,
            VerificationOutcome.TECHNICAL_FAILURE: PipelineState.TECHNICAL_FAILURE,
        }
        next_state = outcome_to_state[result.outcome]
        current = PipelineState(self.registry.get_strategy(strategy_id)["current_phase"])
        if current == PipelineState.TECHNICAL_INTEGRITY_VERIFICATION and next_state != PipelineState.TECHNICAL_REPAIR_REQUIRED or current == PipelineState.TECHNICAL_INTEGRITY_VERIFICATION and next_state == PipelineState.TECHNICAL_REPAIR_REQUIRED:
            self.controller.transition(strategy_id, next_state, f"Phase B.5 outcome {result.outcome}")
        return payload

    def status(self, strategy_id: str, verification_run_id: str | None = None) -> dict | None:
        return self.registry.get_verification(strategy_id, verification_run_id)
