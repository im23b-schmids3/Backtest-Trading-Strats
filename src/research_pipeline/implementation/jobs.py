from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..errors import RegistryError, SpecificationValidationError
from ..phase_b.services import PhaseBService
from ..phase_b.redaction import redact_secrets
from ..phase_f1.models import MasterRunInput, MasterRunStatus, MasterStep
from ..repository.worktree_preflight import run_worktree_preflight
from ..registry.database import Database
from ..registry.repositories import Registry
from .models import CodexCompletionStatus, ImplementationCompletion, ImplementationJobRequest, ImplementationJobStatus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImplementationJobService:
    """Creates immutable implementation jobs and ingests only matching results."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry = Registry(Database(self.registry_path))

    def _run_context(self, run_id: str) -> tuple[dict, dict, dict, Path]:
        run = self.registry.get_master_run(run_id)
        state = run["resume_state_json"]
        if run["approval_status"] != "APPROVED":
            raise SpecificationValidationError("implementation job requires an approved specification")
        strategy_id = state.get("strategy_id") or run["strategy_id"]
        spec = self.registry.get_specification(strategy_id)
        options = MasterRunInput.model_validate(state["options"] | {"intake_path": state["intake_path"]})
        root = Path(run["root_path"]).resolve()
        return run, state, {"spec": spec, "options": options}, root

    def _job_dir(self, run: dict) -> Path:
        root = Path(run["root_path"]).resolve()
        jobs = root / "implementation" / "jobs"
        existing = sorted(path for path in jobs.glob("job-*") if path.is_dir())
        number = len(existing) + 1
        while (jobs / f"job-{number:03d}").exists():
            number += 1
        return jobs / f"job-{number:03d}"

    def _existing_job(self, run: dict) -> Path | None:
        jobs = Path(run["root_path"]).resolve() / "implementation" / "jobs"
        candidates = sorted(jobs.glob("job-*/request.json"))
        return candidates[-1].parent if candidates else None

    def create(self, run_id: str, *, probe: bool = False) -> dict:
        run, state, context, root = self._run_context(run_id)
        existing = self._existing_job(run)
        if existing:
            request = ImplementationJobRequest.model_validate_json((existing / "request.json").read_text(encoding="utf-8"))
            return {"created": False, "idempotent_reuse": True, "job": request.model_dump(mode="json"), "job_path": str(existing), "preflight": request.provenance.get("preflight_report")}
        spec = context["spec"]
        options: MasterRunInput = context["options"]
        repository_root = Path(options.repository_root).resolve()
        preflight = run_worktree_preflight(repository_root, probe=probe, persist=True)
        root.mkdir(parents=True, exist_ok=True)
        (root / "implementation" / "preflight.json").write_text(preflight.model_dump_json(indent=2), encoding="utf-8")
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "WORKTREE_PREFLIGHT", preflight.model_dump(mode="json"))
        if not preflight.safe_for_isolated_worktree:
            self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "WORKTREE_PREFLIGHT_FAILED", {"report_path": preflight.report_path, "error_code": "WORKTREE_PREFLIGHT_FAILED"})
            raise SpecificationValidationError("WORKTREE_PREFLIGHT_FAILED: isolated implementation worktree is unsafe")
        phase_b = PhaseBService(self.registry_path)
        plan = phase_b.implementation_plan(spec.strategy_id, str(repository_root), dry_run=True, worktree_suffix=run_id)
        job_dir = self._job_dir(run)
        job_dir.mkdir(parents=True, exist_ok=False)
        spec_path = Path(state.get("draft_copy") or state.get("specification_path") or "")
        if not spec_path.is_file():
            spec_path = Path(self.registry.get_strategy(spec.strategy_id)["specification_path"])
        specification_path = job_dir / "specification.yaml"
        shutil.copyfile(spec_path, specification_path)
        prompt = phase_b.build_implementation_prompt(spec.strategy_id, plan)
        (job_dir / "prompt.md").write_text(redact_secrets(prompt), encoding="utf-8")
        manifest_payload = {
            "repository_root": str(repository_root),
            "base_commit": plan.base_commit,
            "allowed_paths": plan.allowed_files,
            "forbidden_paths": ["data/", "research_runs/*/holdout/", "src/fib_backtester/strategy/fibonacci.py"],
            "required_tests": plan.required_tests,
        }
        (job_dir / "repository_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
        input_hashes = {name: _sha256(path) for name, path in (("specification.yaml", specification_path), ("prompt.md", job_dir / "prompt.md"), ("repository_manifest.json", job_dir / "repository_manifest.json"))}
        (job_dir / "input_hash_manifest.json").write_text(json.dumps(input_hashes, indent=2, sort_keys=True), encoding="utf-8")
        request = ImplementationJobRequest(
            job_id=job_dir.name, run_id=run_id, strategy_id=spec.strategy_id, strategy_version=spec.version,
            specification_hash=spec.specification_hash, repository_root=str(repository_root),
            approved_worktree_root=plan.worktree_path, base_commit=plan.base_commit, branch=plan.branch,
            approved_implementation_scope="Implement only the approved strategy adapter and its tests.",
            allowed_paths=plan.allowed_files, forbidden_paths=manifest_payload["forbidden_paths"], required_tests=plan.required_tests,
            expected_output_contract="implementation_manifest.json, changed_files.json, scope_validation.json, test_results.json, codex_invocation.json, output_hash_manifest.json, completion.json",
            timeout_seconds=900, max_repair_attempts=plan.max_repair_attempts, codex_model=os.environ.get("CODEX_MODEL"), codex_config_requirements={"sandbox": "workspace-write", "prompt_transport": "stdin"},
            provenance={"base_commit": plan.base_commit, "preflight_report": preflight.report_path, "repository_root": str(repository_root), "smithers_run_id": os.environ.get("SMITHERS_RUN_ID")},
            input_hash_manifest=input_hashes, created_at=datetime.now(timezone.utc),
        )
        (job_dir / "request.json").write_text(request.model_dump_json(indent=2), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"status": "WAITING_EXTERNAL_CODEX", "job_id": request.job_id, "run_id": run_id}, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "WAITING_EXTERNAL_CODEX")
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "EXTERNAL_CODEX_EXECUTION_REQUIRED", {"job_id": request.job_id, "job_path": str(job_dir), "command": f"py -m research_pipeline codex-executor run {run_id}"})
        return {"created": True, "idempotent_reuse": False, "job": request.model_dump(mode="json"), "job_path": str(job_dir), "preflight": preflight.model_dump(mode="json"), "next_command": f"py -m research_pipeline codex-executor run {run_id}"}

    def status(self, run_id: str) -> dict:
        record = self.registry.get_implementation_job(run_id)
        if not record:
            return {"run_id": run_id, "status": "NO_JOB", "external_executor_required": False, "next_command": f"py -m research_pipeline implementation job {run_id}"}
        return ImplementationJobStatus(job_id=record["job_id"], run_id=run_id, status=record["status"], next_command=f"py -m research_pipeline codex-executor run {run_id}", updated_at=datetime.fromisoformat(record["updated_at"]), error=record.get("error")).model_dump(mode="json")

    def load_request(self, run_id: str) -> tuple[ImplementationJobRequest, Path]:
        record = self.registry.get_implementation_job(run_id)
        if not record:
            raise RegistryError(f"implementation job not found: {run_id}")
        job_dir = Path(record["job_path"])
        request = ImplementationJobRequest.model_validate_json((job_dir / "request.json").read_text(encoding="utf-8"))
        request_hash = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if request_hash != record["request_hash"]:
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: immutable implementation request hash mismatch")
        return request, job_dir

    def ingest(self, run_id: str) -> dict:
        request, job_dir = self.load_request(run_id)
        completion_path = job_dir / "result" / "completion.json"
        if not completion_path.is_file():
            raise RegistryError(f"implementation result not found: {completion_path}")
        completion = ImplementationCompletion.model_validate_json(completion_path.read_text(encoding="utf-8"))
        if completion.job_id != request.job_id or completion.run_id != request.run_id or completion.specification_hash != request.specification_hash or completion.repository_root != request.repository_root:
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: implementation result identity does not match the immutable job")
        canonical = json.dumps(completion.model_copy(update={"result_hash": None}).model_dump(mode="json", exclude={"result_hash"}), sort_keys=True, separators=(",", ":"))
        if completion.result_hash != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: implementation completion hash mismatch")
        for name, expected in completion.artifact_hashes.items():
            path = job_dir / "result" / name
            if not path.is_file() or _sha256(path) != expected:
                raise RegistryError(f"FAILED_ARTIFACT_INTEGRITY: result artifact hash mismatch for {name}")
        if completion.status != CodexCompletionStatus.SUCCEEDED:
            raise RegistryError(f"implementation result is not successful: {completion.status.value}")
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "INGESTED", result_path=str(completion_path), result_hash=completion.result_hash)
        run = self.registry.get_master_run(run_id)
        state = dict(run["resume_state_json"])
        state.update({"implementation_job_id": request.job_id, "implementation_completion_path": str(completion_path.resolve()), "implementation_completion_hash": completion.result_hash})
        self.registry.update_master_run(run_id, resume_state=state)
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "IMPLEMENTATION_RESULT_INGESTED", {"job_id": request.job_id, "result_hash": completion.result_hash})
        return completion.model_dump(mode="json")
