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
from ..runners.isolated_environment import validate_required_fixture_sources
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

    ACTIVE_STATUSES = {"WAITING_EXTERNAL_CODEX", "RUNNING"}
    TERMINAL_STATUSES = {
        "INGESTED", "INTERRUPTED", "ABORTED_BY_USER",
        *(item.value for item in CodexCompletionStatus),
    }
    SUCCESSFUL_COMPLETION_STATUSES = {
        CodexCompletionStatus.SUCCEEDED,
        CodexCompletionStatus.COMPLETED,
    }
    RETRYABLE_TERMINAL_STATUSES = {
        "INTERRUPTED",
        CodexCompletionStatus.TIMED_OUT.value,
    }

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

    @staticmethod
    def _result_artifacts(job_dir: Path) -> dict[str, str]:
        result = job_dir / "result"
        return {
            str(path.relative_to(job_dir)): _sha256(path)
            for path in sorted(result.rglob("*"))
            if path.is_file() and path.name != "completion.json"
        } if result.is_dir() else {}

    @staticmethod
    def _status_file(job_dir: Path) -> dict[str, Any]:
        path = job_dir / "status.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _process_is_live(pid: Any) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except (ProcessLookupError, OSError):
            return False
        else:
            return True

    @staticmethod
    def _age_seconds(value: str | None) -> float:
        if not value:
            return float("inf")
        try:
            created = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
        return max(0.0, (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds())

    def stale_interruption(self, run_id: str, *, stale_after_seconds: int = 300) -> dict[str, Any]:
        """Return structural evidence that a job may be retried, never resumed."""

        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be positive")
        record = self.registry.get_implementation_job(run_id)
        if not record:
            raise RegistryError(f"implementation job not found: {run_id}")
        job_dir = Path(record["job_path"])
        status = self._status_file(job_dir)
        completion = job_dir / "result" / "completion.json"
        artifacts = self._result_artifacts(job_dir)
        owning_pid = status.get("owner_pid")
        age = self._age_seconds(status.get("heartbeat_at") or status.get("started_at") or record.get("updated_at"))
        active_status = record["status"] in self.ACTIVE_STATUSES or status.get("status") in self.ACTIVE_STATUSES
        return {
            "run_id": run_id,
            "job_id": record["job_id"],
            "job_path": str(job_dir),
            "status": record["status"],
            "status_file_status": status.get("status"),
            "completion_exists": completion.is_file(),
            "partial_artifacts": artifacts,
            "owner_pid": owning_pid,
            "owner_live": self._process_is_live(owning_pid),
            "age_seconds": age,
            "stale_after_seconds": stale_after_seconds,
            "stale": bool(active_status and not completion.is_file() and artifacts and not self._process_is_live(owning_pid) and age >= stale_after_seconds),
        }

    def mark_interrupted(self, run_id: str, *, stale_after_seconds: int = 300, reason: str = "stale external executor") -> dict[str, Any]:
        evidence = self.stale_interruption(run_id, stale_after_seconds=stale_after_seconds)
        if not evidence["stale"]:
            raise RegistryError("implementation job is not a stale interrupted job and cannot be replaced")
        request, job_dir = self.load_request(run_id)
        snapshot = {
            "job_id": request.job_id,
            "run_id": run_id,
            "classification": "INTERRUPTED",
            "reason": reason,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "stale_evidence": evidence,
            "preserved_artifact_hashes": evidence["partial_artifacts"],
        }
        interruption_path = job_dir / "result" / "interruption.json"
        interruption_path.parent.mkdir(parents=True, exist_ok=True)
        interruption_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({
            "status": "INTERRUPTED", "job_id": request.job_id, "run_id": run_id,
            "reason": reason, "interruption_path": str(interruption_path),
        }, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.archive_implementation_job(self.registry.get_implementation_job(run_id) or {})
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "INTERRUPTED", error=reason)
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "IMPLEMENTATION_JOB_INTERRUPTED", snapshot)
        return {**snapshot, "interruption_path": str(interruption_path), "interruption_hash": _sha256(interruption_path)}

    def mark_running(self, run_id: str) -> None:
        request, job_dir = self.load_request(run_id)
        record = self.registry.get_implementation_job(run_id)
        if not record or record["status"] != "WAITING_EXTERNAL_CODEX":
            raise RegistryError("only a fresh WAITING_EXTERNAL_CODEX job may begin execution")
        now = datetime.now(timezone.utc).isoformat()
        (job_dir / "status.json").write_text(json.dumps({
            "status": "RUNNING", "job_id": request.job_id, "run_id": run_id,
            "owner_pid": os.getpid(), "started_at": now, "heartbeat_at": now,
        }, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "RUNNING")

    def abort_by_user(self, run_id: str, *, reason: str = "external executor interrupted") -> dict[str, Any]:
        request, job_dir = self.load_request(run_id)
        record = self.registry.get_implementation_job(run_id)
        if not record or record["status"] not in self.ACTIVE_STATUSES:
            raise RegistryError("only an active implementation job may be marked ABORTED_BY_USER")
        snapshot = {
            "job_id": request.job_id, "run_id": run_id, "classification": "ABORTED_BY_USER",
            "reason": reason, "detected_at": datetime.now(timezone.utc).isoformat(),
            "preserved_artifact_hashes": self._result_artifacts(job_dir),
        }
        path = job_dir / "result" / "interruption.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"status": "ABORTED_BY_USER", "job_id": request.job_id, "run_id": run_id, "reason": reason, "interruption_path": str(path)}, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.archive_implementation_job(record)
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "ABORTED_BY_USER", error=reason)
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "IMPLEMENTATION_JOB_ABORTED_BY_USER", snapshot)
        return {**snapshot, "interruption_path": str(path), "interruption_hash": _sha256(path)}

    def create(self, run_id: str, *, probe: bool = False, retry: bool = False) -> dict:
        run, state, context, root = self._run_context(run_id)
        existing = self._existing_job(run)
        if existing and not retry:
            request = ImplementationJobRequest.model_validate_json((existing / "request.json").read_text(encoding="utf-8"))
            return {"created": False, "idempotent_reuse": True, "job": request.model_dump(mode="json"), "job_path": str(existing), "preflight": request.provenance.get("preflight_report")}
        if retry:
            current = self.registry.get_implementation_job(run_id)
            if not current or current["status"] not in self.RETRYABLE_TERMINAL_STATUSES:
                raise RegistryError("implementation retry requires an immutable INTERRUPTED or TIMED_OUT job")
        spec = context["spec"]
        options: MasterRunInput = context["options"]
        repository_root = Path(options.repository_root).resolve()
        fixture_preflight = validate_required_fixture_sources(repository_root)
        preflight = run_worktree_preflight(repository_root, probe=probe, persist=True)
        root.mkdir(parents=True, exist_ok=True)
        (root / "implementation" / "preflight.json").write_text(preflight.model_dump_json(indent=2), encoding="utf-8")
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "WORKTREE_PREFLIGHT", preflight.model_dump(mode="json"))
        if not preflight.safe_for_isolated_worktree:
            self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "WORKTREE_PREFLIGHT_FAILED", {"report_path": preflight.report_path, "error_code": "WORKTREE_PREFLIGHT_FAILED"})
            raise SpecificationValidationError("WORKTREE_PREFLIGHT_FAILED: isolated implementation worktree is unsafe")
        job_dir = self._job_dir(run)
        phase_b = PhaseBService(self.registry_path)
        # The attempt identity participates in both branch and worktree names.
        # A retry therefore cannot reuse mutable execution state from job-001.
        plan = phase_b.implementation_plan(spec.strategy_id, str(repository_root), dry_run=True, worktree_suffix=f"{run_id}-{job_dir.name}", worktree_parent=options.worktree_parent)
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
            timeout_seconds=options.implementation_timeout_seconds, max_repair_attempts=plan.max_repair_attempts, codex_model=os.environ.get("CODEX_MODEL"), codex_config_requirements={"sandbox": "workspace-write", "prompt_transport": "stdin"},
            provenance={"base_commit": plan.base_commit, "preflight_report": preflight.report_path, "fixture_preflight": fixture_preflight, "repository_root": str(repository_root), "worktree_parent": str(Path(plan.worktree_path).parent), "smithers_run_id": os.environ.get("SMITHERS_RUN_ID")},
            input_hash_manifest=input_hashes, created_at=datetime.now(timezone.utc),
        )
        (job_dir / "request.json").write_text(request.model_dump_json(indent=2), encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"status": "WAITING_EXTERNAL_CODEX", "job_id": request.job_id, "run_id": run_id}, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "WAITING_EXTERNAL_CODEX")
        latest = self.registry.get_master_run(run_id)
        resume = dict(latest["resume_state_json"])
        resume.update({"implementation_job_id": request.job_id, "implementation_job_path": str(job_dir), "previous_implementation_job_id": state.get("implementation_job_id") if retry else None})
        self.registry.update_master_run(run_id, resume_state=resume)
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "EXTERNAL_CODEX_EXECUTION_REQUIRED", {"job_id": request.job_id, "job_path": str(job_dir), "command": f"py -m research_pipeline codex-executor run {run_id}"})
        return {"created": True, "idempotent_reuse": False, "job": request.model_dump(mode="json"), "job_path": str(job_dir), "preflight": preflight.model_dump(mode="json"), "next_command": f"py -m research_pipeline codex-executor run {run_id}"}

    def retry(self, run_id: str, *, stale_after_seconds: int = 300) -> dict:
        current = self.registry.get_implementation_job(run_id)
        if not current:
            raise RegistryError(f"implementation job not found: {run_id}")
        if current["status"] == CodexCompletionStatus.TIMED_OUT.value:
            interrupted = {
                "run_id": run_id,
                "job_id": current["job_id"],
                "classification": "TIMED_OUT",
                "reason": "immutable timed-out job replaced by fresh retry",
            }
            self.registry.archive_implementation_job(current)
        elif current["status"] in self.TERMINAL_STATUSES:
            raise RegistryError(f"completed or failed implementation job cannot be retried in place: {current['status']}")
        else:
            interrupted = self.mark_interrupted(run_id, stale_after_seconds=stale_after_seconds)
        created = self.create(run_id, retry=True)
        current = self.registry.get_master_run(run_id)
        state = dict(current["resume_state_json"])
        state.update({
            "implementation_job_id": created["job"]["job_id"],
            "implementation_job_path": created["job_path"],
            "external_executor_required": True,
            "next_command": created["next_command"],
        })
        self.registry.update_master_run(
            run_id,
            current_step=MasterStep.IMPLEMENTATION.value,
            outcome=MasterRunStatus.WAITING_EXTERNAL_CODEX.value,
            resume_state=state,
        )
        return {"interrupted": interrupted, "retry": created}

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

    def _load_and_validate_completion(
        self, run_id: str
    ) -> tuple[ImplementationJobRequest, Path, ImplementationCompletion, Path]:
        """Read and verify a terminal completion without mutating job artifacts."""

        request, job_dir = self.load_request(run_id)
        completion_path = job_dir / "result" / "completion.json"
        if not completion_path.is_file():
            raise RegistryError(f"implementation result not found: {completion_path}")
        raw_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion = ImplementationCompletion.model_validate(raw_completion)
        if (
            completion.job_id != request.job_id
            or completion.run_id != request.run_id
            or completion.specification_hash != request.specification_hash
            or completion.repository_root != request.repository_root
        ):
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: implementation result identity does not match the immutable job")
        # Hash the serialized schema that was actually written.  New optional
        # fields must not invalidate completion records produced by an older
        # executor version.
        canonical_payload = dict(raw_completion)
        canonical_payload.pop("result_hash", None)
        canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if completion.result_hash != calculated_hash:
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: implementation completion hash mismatch")
        for name, expected in completion.artifact_hashes.items():
            path = job_dir / "result" / name
            if not path.is_file() or _sha256(path) != expected:
                raise RegistryError(f"FAILED_ARTIFACT_INTEGRITY: result artifact hash mismatch for {name}")
        record = self.registry.get_implementation_job(run_id)
        if record and record.get("result_hash") and record["result_hash"] != completion.result_hash:
            raise RegistryError("FAILED_ARTIFACT_INTEGRITY: recorded implementation completion hash mismatch")
        return request, job_dir, completion, completion_path

    def reclassify_legacy_timeout(self, run_id: str) -> dict[str, Any]:
        """Correct a pre-timeout-contract result using immutable evidence only."""

        record = self.registry.get_implementation_job(run_id)
        if not record:
            raise RegistryError(f"implementation job not found: {run_id}")
        existing = self.registry.get_implementation_job_correction(
            run_id, record["job_id"], "LEGACY_TIMEOUT_MISCLASSIFICATION"
        )
        if existing:
            return {"run_id": run_id, "job_id": record["job_id"], "idempotent": True, "correction": existing}
        if record["status"] != CodexCompletionStatus.FAILED_CODEX_EXECUTION.value:
            raise RegistryError("legacy timeout reclassification requires current FAILED_CODEX_EXECUTION status")
        request, job_dir, completion, completion_path = self._load_and_validate_completion(run_id)
        invocation_path = job_dir / "result" / "codex_invocation.json"
        if not invocation_path.is_file():
            raise RegistryError("legacy timeout reclassification requires codex_invocation.json")
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        if completion.status != CodexCompletionStatus.FAILED_CODEX_EXECUTION:
            raise RegistryError("legacy timeout reclassification requires FAILED_CODEX_EXECUTION completion")
        if completion.exit_code is not None:
            raise RegistryError("legacy timeout reclassification requires null completion exit_code")
        if invocation.get("timed_out") is not True:
            raise RegistryError("legacy timeout reclassification requires invocation timed_out=true")
        if not isinstance(request.timeout_seconds, int) or request.timeout_seconds < 1:
            raise RegistryError("legacy timeout reclassification requires configured timeout_seconds")
        evidence = {
            "completion_path": str(completion_path.resolve()),
            "completion_hash": _sha256(completion_path),
            "invocation_path": str(invocation_path.resolve()),
            "invocation_hash": _sha256(invocation_path),
            "request_timeout_seconds": request.timeout_seconds,
            "completion_result_hash": completion.result_hash,
            "invocation_timed_out": True,
        }
        correction = self.registry.save_implementation_job_correction(
            run_id,
            request.job_id,
            "LEGACY_TIMEOUT_MISCLASSIFICATION",
            original_status=CodexCompletionStatus.FAILED_CODEX_EXECUTION.value,
            corrected_status=CodexCompletionStatus.TIMED_OUT.value,
            reason="LEGACY_TIMEOUT_MISCLASSIFICATION",
            evidence=evidence,
        )
        self.registry.set_current_implementation_job_status(
            run_id,
            CodexCompletionStatus.TIMED_OUT.value,
            error="LEGACY_TIMEOUT_MISCLASSIFICATION",
        )
        self.registry.add_master_journal(
            run_id,
            MasterStep.IMPLEMENTATION.value,
            "LEGACY_TIMEOUT_RECLASSIFIED",
            {"job_id": request.job_id, "original_status": CodexCompletionStatus.FAILED_CODEX_EXECUTION.value,
             "corrected_status": CodexCompletionStatus.TIMED_OUT.value, "reason": "LEGACY_TIMEOUT_MISCLASSIFICATION",
             "evidence": evidence},
        )
        return {"run_id": run_id, "job_id": request.job_id, "idempotent": False, "correction": correction}

    @staticmethod
    def _failure_reason(completion: ImplementationCompletion | None, fallback: str) -> str:
        if completion:
            return completion.stderr_summary.strip() or completion.status.value
        return fallback

    def _record_reconciliation_event(self, run_id: str, payload: dict[str, Any]) -> None:
        """Write the durable reconciliation event once for a job/result pair."""

        for entry in self.registry.master_journal(run_id):
            if (
                entry["event"] == "IMPLEMENTATION_COMPLETION_RECONCILED"
                and entry["payload_json"].get("job_id") == payload["job_id"]
                and entry["payload_json"].get("result_hash") == payload.get("result_hash")
                and entry["payload_json"].get("status") == payload["status"]
            ):
                return
        self.registry.add_master_journal(
            run_id, MasterStep.IMPLEMENTATION.value, "IMPLEMENTATION_COMPLETION_RECONCILED", payload
        )

    def reconcile(self, run_id: str) -> dict[str, Any]:
        """Reconcile a terminal immutable external job into the master run.

        This intentionally performs no Codex execution and never modifies job
        files.  It only verifies durable artifacts and updates SQLite/master
        state, making repeated calls safe.
        """

        record = self.registry.get_implementation_job(run_id)
        if not record:
            raise RegistryError(f"implementation job not found: {run_id}")
        completion_path = Path(record["job_path"]) / "result" / "completion.json"
        completion: ImplementationCompletion | None = None
        correction: dict[str, Any] | None = None
        if completion_path.is_file():
            request, job_dir, completion, completion_path = self._load_and_validate_completion(run_id)
            terminal_status = completion.status.value
            correction = self.registry.get_implementation_job_correction(
                run_id, request.job_id, "LEGACY_TIMEOUT_MISCLASSIFICATION"
            )
            if correction:
                terminal_status = correction["corrected_status"]
            if terminal_status not in self.TERMINAL_STATUSES:
                raise RegistryError(f"implementation completion is not terminal: {terminal_status}")
        elif record["status"] in {"INTERRUPTED", "ABORTED_BY_USER"}:
            request, job_dir = self.load_request(run_id)
            terminal_status = record["status"]
        else:
            raise RegistryError("implementation job has no terminal completion to reconcile")

        run = self.registry.get_master_run(run_id)
        state = dict(run["resume_state_json"])
        successful = completion is not None and completion.status in self.SUCCESSFUL_COMPLETION_STATUSES
        if successful:
            job_status = "INGESTED"
            outcome = MasterRunStatus.IMPLEMENTATION_VERIFICATION_REQUIRED.value
            step = MasterStep.IMPLEMENTATION_VERIFICATION.value
            final_reason = "external implementation completion ingested"
            test_status = "PASSED" if (job_dir / "result" / "test_results.json").is_file() else None
            next_command = f"py -m research_pipeline resume {run_id}"
        else:
            job_status = terminal_status
            outcome = MasterRunStatus.IMPLEMENTATION_FAILURE.value
            step = MasterStep.IMPLEMENTATION.value
            final_reason = self._failure_reason(completion, terminal_status)
            test_status = "FAILED_REQUIRED_TESTS" if terminal_status == CodexCompletionStatus.FAILED_REQUIRED_TESTS.value else None
            next_command = None

        result_hash = completion.result_hash if completion else None
        result_path = str(completion_path.resolve()) if completion else record.get("result_path")
        if correction:
            # Preserve the original immutable attempt status; this is only a
            # current-state projection of an audited legacy correction.
            self.registry.set_current_implementation_job_status(
                run_id, job_status, error=None if successful else final_reason
            )
        else:
            self.registry.save_implementation_job(
                run_id,
                request.model_dump(mode="json"),
                str(job_dir),
                job_status,
                result_path=result_path,
                result_hash=result_hash,
                error=None if successful else final_reason,
            )
        state.update(
            {
                "implementation_job_id": request.job_id,
                "implementation_completion_path": result_path,
                "implementation_completion_hash": result_hash,
                "external_executor_required": False,
                "final_reason": final_reason,
                "implementation_test_status": test_status,
                "next_command": next_command,
            }
        )
        self.registry.update_master_run(
            run_id,
            current_step=step,
            outcome=outcome,
            resume_state=state,
        )
        payload = {
            "job_id": request.job_id,
            "status": terminal_status,
            "result_hash": result_hash,
            "outcome": outcome,
            "final_reason": final_reason,
        }
        self._record_reconciliation_event(run_id, payload)
        return {
            "run_id": run_id,
            "job_id": request.job_id,
            "job_status": job_status,
            "completion_status": terminal_status,
            "current_step": step,
            "outcome": outcome,
            "pipeline_status": "IMPLEMENTATION_VERIFICATION_REQUIRED" if successful else "IMPLEMENTATION_FAILED",
            "final_reason": final_reason,
            "implementation_test_status": test_status,
            "b5_available": False,
            "idempotent": True,
        }

    def ingest(self, run_id: str) -> dict:
        request, job_dir, completion, completion_path = self._load_and_validate_completion(run_id)
        if completion.status not in self.SUCCESSFUL_COMPLETION_STATUSES:
            raise RegistryError(f"implementation result is not successful: {completion.status.value}")
        self.registry.save_implementation_job(run_id, request.model_dump(mode="json"), str(job_dir), "INGESTED", result_path=str(completion_path), result_hash=completion.result_hash)
        run = self.registry.get_master_run(run_id)
        state = dict(run["resume_state_json"])
        state.update({"implementation_job_id": request.job_id, "implementation_completion_path": str(completion_path.resolve()), "implementation_completion_hash": completion.result_hash})
        self.registry.update_master_run(run_id, resume_state=state)
        self.registry.add_master_journal(run_id, MasterStep.IMPLEMENTATION.value, "IMPLEMENTATION_RESULT_INGESTED", {"job_id": request.job_id, "result_hash": completion.result_hash})
        return completion.model_dump(mode="json")
