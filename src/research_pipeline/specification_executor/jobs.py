from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..errors import ExternalSpecificationRequired, RegistryError
from ..phase_b.models import WorkflowInput
from ..phase_b.redaction import redact_secrets
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.strategy_spec import StrategySpec
from .models import SpecificationJobRequest, SpecificationJobType


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class SpecificationJobService:
    """Creates and validates immutable external specification job contracts."""

    schema_contract_version = "StrategySpec-v1"

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry = Registry(Database(self.registry_path))

    @staticmethod
    def _root(workflow: WorkflowInput, strategy_id: str, run_id: str) -> Path:
        root = Path(workflow.repository_root).resolve() / "research_runs" / strategy_id / run_id / "specification"
        for name in ("intake", "attempts", "canonical", "failure", "jobs"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _job_status(job_type: SpecificationJobType) -> str:
        return "WAITING_EXTERNAL_SPECIFICATION_GENERATION" if job_type == SpecificationJobType.GENERATE_SPECIFICATION else "WAITING_EXTERNAL_SPECIFICATION_REPAIR"

    @staticmethod
    def _event_name(job_type: SpecificationJobType) -> str:
        return "external.codex.specification.completed"

    def _existing(self, run_id: str, attempt: int) -> dict | None:
        for job in self.registry.specification_jobs(run_id):
            if int(job["attempt"]) == attempt:
                return job
        return None

    def create(
        self,
        workflow: WorkflowInput,
        *,
        strategy_id: str,
        strategy_version: str,
        run_id: str,
        attempt: int,
        job_type: SpecificationJobType,
        prompt: str,
        prior_invalid_draft_path: str | None = None,
        validation_report_path: str | None = None,
        semantic_validation_path: str | None = None,
    ) -> dict:
        existing = self._existing(run_id, attempt)
        if existing:
            request_path = Path(existing["job_path"]) / "request.json"
            request = SpecificationJobRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
            return {"created": False, "idempotent_reuse": True, "request": request.model_dump(mode="json"), "job_path": str(request_path.parent), "next_command": f"py -m research_pipeline specification-executor run {run_id}"}

        root = self._root(workflow, strategy_id, run_id)
        job_dir = root / "jobs" / f"job-{attempt:03d}"
        job_dir.mkdir(parents=True, exist_ok=False)
        intake_path = job_dir / "intake.yaml"
        intake_path.write_text(redact_secrets(yaml.safe_dump(workflow.model_dump(mode="json"), sort_keys=False)), encoding="utf-8")
        prompt_path = job_dir / "prompt.md"
        prompt_path.write_text(redact_secrets(prompt), encoding="utf-8")
        contract_path = job_dir / "schema_contract.json"
        contract_path.write_text(json.dumps({"contract_version": self.schema_contract_version, "model": "StrategySpec", "schema": StrategySpec.model_json_schema()}, indent=2, sort_keys=True), encoding="utf-8")

        prior_path: Path | None = None
        if prior_invalid_draft_path:
            source = Path(prior_invalid_draft_path)
            if not source.is_file():
                raise RegistryError(f"prior invalid draft not found: {source}")
            prior_path = job_dir / "invalid_draft.yaml"
            shutil.copyfile(source, prior_path)
        validation_path: Path | None = None
        if validation_report_path:
            source = Path(validation_report_path)
            if source.is_file():
                validation_path = job_dir / "validation.json"
                shutil.copyfile(source, validation_path)
        semantic_path: Path | None = None
        if semantic_validation_path:
            source = Path(semantic_validation_path)
            if source.is_file():
                semantic_path = job_dir / "semantic_validation.json"
                shutil.copyfile(source, semantic_path)

        files = {"intake.yaml": intake_path, "prompt.md": prompt_path, "schema_contract.json": contract_path}
        if prior_path:
            files["invalid_draft.yaml"] = prior_path
        if validation_path:
            files["validation.json"] = validation_path
        if semantic_path:
            files["semantic_validation.json"] = semantic_path
        input_hashes = {name: sha256_file(path) for name, path in files.items()}
        input_manifest = job_dir / "input_hash_manifest.json"
        input_manifest.write_text(json.dumps(input_hashes, indent=2, sort_keys=True), encoding="utf-8")
        input_hashes["input_hash_manifest.json"] = sha256_file(input_manifest)

        request = SpecificationJobRequest(
            run_id=run_id, smithers_run_id=os.environ.get("SMITHERS_RUN_ID"), job_id=job_dir.name,
            strategy_id=strategy_id, strategy_version=strategy_version, attempt=attempt, job_type=job_type,
            repository_root=str(Path(workflow.repository_root).resolve()), intake_path=str(intake_path.resolve()), intake_hash=sha256_file(intake_path),
            prior_invalid_draft_path=str(prior_path.resolve()) if prior_path else None,
            prior_invalid_draft_hash=sha256_file(prior_path) if prior_path else None,
            validation_report_path=str(validation_path.resolve()) if validation_path else None,
            validation_report_hash=sha256_file(validation_path) if validation_path else None,
            semantic_validation_path=str(semantic_path.resolve()) if semantic_path else None,
            semantic_validation_hash=sha256_file(semantic_path) if semantic_path else None,
            schema_contract_version=self.schema_contract_version, timeout_seconds=900,
            codex_config_requirements={"sandbox": "read-only", "prompt_transport": "stdin", "network": "policy-controlled"},
            expected_output_paths=["result/raw_output.txt", "result/extracted_draft.yaml", "result/extraction_report.json", "result/codex_invocation.json", "result/output_hash_manifest.json", "result/completion.json"],
            provenance={"intake_path": str(intake_path.resolve()), "smithers_run_id": os.environ.get("SMITHERS_RUN_ID"), "event_name": self._event_name(job_type)},
            input_hash_manifest=input_hashes, created_at=datetime.now(timezone.utc),
        )
        (job_dir / "request.json").write_text(request.model_dump_json(indent=2), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"status": self._job_status(job_type), "run_id": run_id, "job_id": request.job_id, "attempt": attempt, "job_type": job_type.value, "next_command": f"py -m research_pipeline specification-executor run {run_id}"}, indent=2, sort_keys=True), encoding="utf-8")
        status = self._job_status(job_type)
        self.registry.save_specification_job(request.model_dump(mode="json"), str(job_dir), status)
        self.registry.save_specification_pause_signal(run_id, request.job_id, request.smithers_run_id, self._event_name(job_type), run_id, "WAITING")
        return {"created": True, "idempotent_reuse": False, "request": request.model_dump(mode="json"), "job_path": str(job_dir), "next_command": f"py -m research_pipeline specification-executor run {run_id}", "pause_reason": status}

    def load_request(self, run_id: str, job_id: str | None = None) -> tuple[SpecificationJobRequest, Path, dict]:
        record = self.registry.get_specification_job(run_id, job_id) if job_id else self.registry.get_latest_specification_job(run_id)
        if not record:
            raise RegistryError(f"specification job not found: {run_id}")
        job_dir = Path(record["job_path"])
        request = SpecificationJobRequest.model_validate_json((job_dir / "request.json").read_text(encoding="utf-8"))
        observed = canonical_hash(request.model_dump(mode="json"))
        if observed != record["request_hash"]:
            raise RegistryError("FAILED_INPUT_INTEGRITY: specification job request hash mismatch")
        return request, job_dir, record

    def status(self, run_id: str) -> dict:
        record = self.registry.get_latest_specification_job(run_id)
        if not record:
            return {"run_id": run_id, "status": "NO_JOB", "next_command": None}
        request, _, _ = self.load_request(run_id, record["job_id"])
        return {"run_id": run_id, "job_id": request.job_id, "attempt": request.attempt, "job_type": request.job_type.value, "status": record["status"], "result_path": record.get("result_path"), "error": record.get("error"), "next_command": f"py -m research_pipeline specification-executor run {run_id}"}

    def inspect(self, run_id: str) -> dict:
        status = self.status(run_id)
        jobs = []
        for record in self.registry.specification_jobs(run_id):
            request, job_dir, _ = self.load_request(run_id, record["job_id"])
            jobs.append({"request": request.model_dump(mode="json"), "registry": record, "job_path": str(job_dir), "files": sorted(str(path.relative_to(job_dir)) for path in job_dir.rglob("*") if path.is_file())})
        return {"status": status, "jobs": jobs, "repair_budget": {"max_total_attempts": 3, "candidate_attempts": len(jobs)}}

    def pending(self, run_id: str) -> dict | None:
        record = self.registry.get_latest_specification_job(run_id)
        if record and record["status"] in {"WAITING_EXTERNAL_SPECIFICATION_GENERATION", "WAITING_EXTERNAL_SPECIFICATION_REPAIR", "FAILED_CODEX_EXECUTION", "FAILED_OUTPUT_EXTRACTION", "FAILED_INPUT_INTEGRITY", "FAILED_ARTIFACT_INTEGRITY"}:
            return record
        return None

    def require_external(self, created: dict) -> ExternalSpecificationRequired:
        request = created["request"]
        reason = "external specification generation required" if request["job_type"] == SpecificationJobType.GENERATE_SPECIFICATION.value else "external specification repair required"
        return ExternalSpecificationRequired(f"{reason}: {created['next_command']}", classification=created["pause_reason"], run_id=request["run_id"], job_id=request["job_id"], command=created["next_command"])

    def require_existing(self, run_id: str, job_id: str | None = None) -> ExternalSpecificationRequired:
        request, _, record = self.load_request(run_id, job_id)
        reason = "external specification generation required" if request.job_type == SpecificationJobType.GENERATE_SPECIFICATION else "external specification repair required"
        status = record["status"] if record["status"].startswith("WAITING_EXTERNAL") else ("WAITING_EXTERNAL_SPECIFICATION_GENERATION" if request.job_type == SpecificationJobType.GENERATE_SPECIFICATION else "WAITING_EXTERNAL_SPECIFICATION_REPAIR")
        command = f"py -m research_pipeline specification-executor run {run_id}"
        return ExternalSpecificationRequired(f"{reason}: {command}", classification=status, run_id=run_id, job_id=request.job_id, command=command)
