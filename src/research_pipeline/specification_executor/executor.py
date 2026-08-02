from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..phase_b.models import CodexExecutionResult
from ..phase_b.redaction import redact_secrets
from ..phase_b.services import PhaseBService
from ..registry.database import Database
from ..registry.repositories import Registry
from ..runners.codex_runner import CodexRunner
from .jobs import SpecificationJobService, canonical_hash, sha256_file
from .models import SpecificationCompletion, SpecificationCompletionStatus, SpecificationJobRequest


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return sha256_file(path)


def _git_snapshot(root: Path) -> list[str] | None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "-z"], capture_output=True, shell=False, creationflags=flags)
    if result.returncode != 0:
        return None
    return sorted(item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\x00") if item)


class ExternalSpecificationExecutor:
    """Runs only the approved read-only specification Codex job externally."""

    def __init__(self, registry_path: str | Path, *, codex_runner: CodexRunner | None = None):
        self.registry_path = Path(registry_path)
        self.registry = Registry(Database(self.registry_path))
        self.jobs = SpecificationJobService(self.registry_path)
        self.codex = codex_runner or CodexRunner()

    @staticmethod
    def _result_paths(job_dir: Path) -> dict[str, Path]:
        result = job_dir / "result"
        return {
            "raw_output": result / "raw_output.txt",
            "extracted_draft": result / "extracted_draft.yaml",
            "extraction_report": result / "extraction_report.json",
            "codex_invocation": result / "codex_invocation.json",
            "output_hash_manifest": result / "output_hash_manifest.json",
            "completion": result / "completion.json",
        }

    def _verify_inputs(self, request: SpecificationJobRequest, job_dir: Path) -> None:
        for name, expected in request.input_hash_manifest.items():
            path = job_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"FAILED_INPUT_INTEGRITY: input hash mismatch for {name}")

    def _existing_completion(self, job_dir: Path) -> SpecificationCompletion | None:
        path = self._result_paths(job_dir)["completion"]
        if not path.is_file():
            return None
        return SpecificationCompletion.model_validate_json(path.read_text(encoding="utf-8"))

    def _finish(self, request: SpecificationJobRequest, job_dir: Path, status: SpecificationCompletionStatus, **values: Any) -> SpecificationCompletion:
        paths = self._result_paths(job_dir)
        artifact_hashes = {path.name: sha256_file(path) for key, path in paths.items() if key != "completion" and path.is_file()}
        completion = SpecificationCompletion(
            run_id=request.run_id, smithers_run_id=request.smithers_run_id, job_id=request.job_id,
            strategy_id=request.strategy_id, strategy_version=request.strategy_version, attempt=request.attempt,
            job_type=request.job_type, status=status, repository_root=request.repository_root,
            input_hash_manifest=request.input_hash_manifest, artifact_hashes=artifact_hashes,
            created_at=datetime.now(timezone.utc), **values,
        )
        result_hash = hashlib.sha256(json.dumps(completion.model_copy(update={"result_hash": None}).model_dump(mode="json", exclude={"result_hash"}), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        completion = completion.model_copy(update={"result_hash": result_hash})
        _write_json(paths["completion"], completion.model_dump(mode="json"))
        (job_dir / "status.json").write_text(json.dumps({"status": status.value, "run_id": request.run_id, "job_id": request.job_id, "attempt": request.attempt, "result_path": str(paths["completion"])}, indent=2, sort_keys=True), encoding="utf-8")
        self.registry.save_specification_job(request.model_dump(mode="json"), str(job_dir), status.value, result_path=str(paths["completion"]), result_hash=result_hash, error=completion.stderr_summary or None)
        self.registry.save_specification_pause_signal(request.run_id, request.job_id, request.smithers_run_id, "external.codex.specification.completed", request.run_id, "COMPLETED")
        self._notify_smithers(request, completion)
        return completion

    @staticmethod
    def _notify_smithers(request: SpecificationJobRequest, completion: SpecificationCompletion) -> None:
        smithers_run_id = request.smithers_run_id
        script = Path(request.repository_root) / ".smithers" / "node_modules" / "smthrs" / "src" / "bin" / "smithers.js"
        bun = shutil.which("bun")
        if not smithers_run_id or not bun or not script.is_file():
            return
        payload = json.dumps({"source_run_id": request.run_id, "job_id": request.job_id, "status": completion.status.value, "job_type": request.job_type.value}, sort_keys=True)
        options = {"cwd": request.repository_root, "capture_output": True, "text": True, "timeout": 30, "shell": False}
        if os.name == "nt":
            options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run([bun, str(script), "signal", str(smithers_run_id), "external.codex.specification.completed", "--correlation", request.run_id, "--data", payload], **options)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _verify_completion(self, request: SpecificationJobRequest, job_dir: Path) -> SpecificationCompletion:
        completion_path = self._result_paths(job_dir)["completion"]
        completion = SpecificationCompletion.model_validate_json(completion_path.read_text(encoding="utf-8"))
        if completion.run_id != request.run_id or completion.job_id != request.job_id or completion.attempt != request.attempt or completion.job_type != request.job_type or completion.strategy_id != request.strategy_id:
            raise ValueError("FAILED_ARTIFACT_INTEGRITY: specification completion identity mismatch")
        canonical = json.dumps(completion.model_copy(update={"result_hash": None}).model_dump(mode="json", exclude={"result_hash"}), sort_keys=True, separators=(",", ":"))
        if completion.result_hash != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise ValueError("FAILED_ARTIFACT_INTEGRITY: specification completion hash mismatch")
        for name, expected in completion.artifact_hashes.items():
            path = job_dir / "result" / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"FAILED_ARTIFACT_INTEGRITY: result hash mismatch for {name}")
        return completion

    def run(self, run_id: str, job_id: str | None = None) -> dict:
        request, job_dir, _ = self.jobs.load_request(run_id, job_id)
        existing = self._existing_completion(job_dir)
        if existing and existing.status == SpecificationCompletionStatus.SUCCEEDED:
            self._verify_completion(request, job_dir)
            return existing.model_dump(mode="json")
        try:
            self._verify_inputs(request, job_dir)
        except ValueError as exc:
            completion = self._finish(request, job_dir, SpecificationCompletionStatus.FAILED_INPUT_INTEGRITY, stderr_summary=str(exc))
            return completion.model_dump(mode="json")

        root = Path(request.repository_root).resolve()
        before = _git_snapshot(root)
        started = time.monotonic()
        prompt = (job_dir / "prompt.md").read_text(encoding="utf-8")
        result: CodexExecutionResult = self.codex.run(prompt, root, sandbox="read-only", timeout_seconds=request.timeout_seconds, dry_run=False)
        after = _git_snapshot(root)
        mutations = sorted(set(after or []) - set(before or [])) if before is not None and after is not None else []
        paths = self._result_paths(job_dir)
        invocation_hash = _write_json(paths["codex_invocation"], result.model_dump(mode="json"))
        self.registry.save_specification_invocation(request.run_id, request.job_id, str(paths["codex_invocation"]), invocation_hash, result.exit_code, "EXECUTED")
        duration = int((time.monotonic() - started) * 1000)
        if mutations:
            completion = self._finish(request, job_dir, SpecificationCompletionStatus.FAILED_ARTIFACT_INTEGRITY, exit_code=result.exit_code, duration_ms=duration, codex_invocation_path=str(paths["codex_invocation"]), repository_mutations=mutations, stderr_summary="repository mutations detected during read-only specification execution")
            return completion.model_dump(mode="json")
        if not result.success:
            completion = self._finish(request, job_dir, SpecificationCompletionStatus.FAILED_CODEX_EXECUTION, exit_code=result.exit_code, duration_ms=duration, codex_invocation_path=str(paths["codex_invocation"]), stdout_summary=result.stdout[-4000:], stderr_summary=result.stderr[-4000:])
            return completion.model_dump(mode="json")

        raw_output = result.stdout or ""
        paths["raw_output"].parent.mkdir(parents=True, exist_ok=True)
        paths["raw_output"].write_text(raw_output, encoding="utf-8")
        try:
            payload = PhaseBService._extract_structured_payload(raw_output)
        except (ValueError, yaml.YAMLError) as exc:
            _write_json(paths["extraction_report"], {"valid": False, "error": redact_secrets(str(exc)), "payload_count": 0})
            output_hashes = {path.name: sha256_file(path) for key, path in paths.items() if key != "completion" and key != "output_hash_manifest" and path.is_file()}
            _write_json(paths["output_hash_manifest"], output_hashes)
            completion = self._finish(request, job_dir, SpecificationCompletionStatus.FAILED_OUTPUT_EXTRACTION, exit_code=result.exit_code, duration_ms=duration, raw_output_path=str(paths["raw_output"]), extraction_report_path=str(paths["extraction_report"]), codex_invocation_path=str(paths["codex_invocation"]), output_hash_manifest_path=str(paths["output_hash_manifest"]), extraction_outcome="INVALID", stdout_summary=result.stdout[-4000:], stderr_summary=redact_secrets(str(exc)))
            return completion.model_dump(mode="json")

        paths["extracted_draft"].write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        _write_json(paths["extraction_report"], {"valid": True, "payload_count": 1, "payload_hash": canonical_hash(payload), "format": "yaml"})
        output_hashes = {path.name: sha256_file(path) for key, path in paths.items() if key != "completion" and key != "output_hash_manifest" and path.is_file()}
        _write_json(paths["output_hash_manifest"], output_hashes)
        completion = self._finish(request, job_dir, SpecificationCompletionStatus.SUCCEEDED, exit_code=result.exit_code, duration_ms=duration, raw_output_path=str(paths["raw_output"]), extracted_draft_path=str(paths["extracted_draft"]), extraction_report_path=str(paths["extraction_report"]), codex_invocation_path=str(paths["codex_invocation"]), output_hash_manifest_path=str(paths["output_hash_manifest"]), extraction_outcome="VALID_MAPPING", stdout_summary=result.stdout[-4000:])
        self.registry.save_specification_result_manifest(request.run_id, request.job_id, str(paths["output_hash_manifest"]), sha256_file(paths["output_hash_manifest"]), str(paths["extracted_draft"]), sha256_file(paths["extracted_draft"]))
        return completion.model_dump(mode="json")

    def verify_and_load_success(self, run_id: str, job_id: str | None = None) -> tuple[SpecificationCompletion, Path, Path]:
        request, job_dir, _ = self.jobs.load_request(run_id, job_id)
        completion = self._verify_completion(request, job_dir)
        if completion.status != SpecificationCompletionStatus.SUCCEEDED or not completion.extracted_draft_path:
            raise ValueError(f"specification external result is not a successful draft: {completion.status.value}")
        return completion, job_dir, Path(completion.extracted_draft_path)

    def load_completion(self, run_id: str, job_id: str | None = None) -> tuple[SpecificationCompletion, Path, Path]:
        request, job_dir, _ = self.jobs.load_request(run_id, job_id)
        completion = self._verify_completion(request, job_dir)
        raw_path = Path(completion.raw_output_path) if completion.raw_output_path else Path()
        return completion, job_dir, raw_path

    def status(self, run_id: str) -> dict:
        return self.jobs.status(run_id)
