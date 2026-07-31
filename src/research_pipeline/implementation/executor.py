from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.compatibility import verify_implementation_scope
from ..errors import RegistryError
from ..phase_b.models import ImplementationPlan
from ..phase_b.redaction import redact_secrets
from ..registry.database import Database
from ..registry.repositories import Registry
from ..repository.worktree_preflight import run_worktree_preflight
from ..runners.codex_runner import CodexRunner
from ..runners.isolated_environment import (
    IsolationError,
    build_isolated_environment,
    import_origin_preflight,
    prepare_required_fixtures,
    sanitized_environment_report,
)
from ..runners.test_runner import DeterministicTestRunner
from ..runners.worktree_manager import WorktreeError, WorktreeManager
from .jobs import ImplementationJobService
from .models import CodexCompletionStatus, ImplementationCompletion, ImplementationJobRequest


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return _hash(path)


class ExternalCodexExecutor:
    """Runs the approved local Codex job outside the Smithers sandbox."""

    def __init__(self, registry_path: str | Path, *, codex_runner: CodexRunner | None = None, test_runner: DeterministicTestRunner | None = None):
        self.registry_path = Path(registry_path)
        self.registry = Registry(Database(self.registry_path))
        self.jobs = ImplementationJobService(self.registry_path)
        self.codex = codex_runner or CodexRunner()
        self.tests = test_runner or DeterministicTestRunner()

    def _paths(self, job_dir: Path) -> dict[str, Path]:
        result = job_dir / "result"
        return {
            "implementation_manifest": result / "implementation_manifest.json",
            "changed_files": result / "changed_files.json",
            "scope_validation": result / "scope_validation.json",
            "test_results": result / "test_results.json",
            "codex_invocation": result / "codex_invocation.json",
            "environment_preflight": result / "environment_preflight.json",
            "import_origin": result / "import_origin.json",
            "fixture_preflight": result / "fixture_preflight.json",
            "output_hash_manifest": result / "output_hash_manifest.json",
            "completion": result / "completion.json",
        }

    def _verify_inputs(self, request: ImplementationJobRequest, job_dir: Path) -> None:
        for name, expected in request.input_hash_manifest.items():
            path = job_dir / name
            if not path.is_file() or _hash(path) != expected:
                raise ValueError(f"FAILED_ARTIFACT_INTEGRITY: input hash mismatch for {name}")

    @staticmethod
    def _verify_worktree_path(request: ImplementationJobRequest) -> None:
        if not request.approved_worktree_root:
            raise ValueError("approved worktree root is missing")
        repository = Path(request.repository_root).resolve()
        worktree = Path(request.approved_worktree_root).resolve()
        configured_parent = request.provenance.get("worktree_parent")
        expected_parent = Path(configured_parent).resolve() if configured_parent else (repository.parent / ".research_worktrees").resolve()
        if worktree.parent != expected_parent or repository == worktree or expected_parent == repository:
            raise ValueError("unsafe isolated worktree path")

    def _existing_completion(self, job_dir: Path) -> ImplementationCompletion | None:
        path = self._paths(job_dir)["completion"]
        if not path.is_file():
            return None
        return ImplementationCompletion.model_validate_json(path.read_text(encoding="utf-8"))

    def _finish(self, request: ImplementationJobRequest, job_dir: Path, status: CodexCompletionStatus, **values: object) -> ImplementationCompletion:
        paths = self._paths(job_dir)
        artifact_hashes: dict[str, str] = {}
        for key, path in paths.items():
            if key != "completion" and path.is_file():
                artifact_hashes[path.name] = _hash(path)
        completion = ImplementationCompletion(
            job_id=request.job_id, run_id=request.run_id, strategy_id=request.strategy_id,
            strategy_version=request.strategy_version, specification_hash=request.specification_hash,
            repository_root=request.repository_root, status=status, input_hash_manifest=request.input_hash_manifest,
            created_at=datetime.now(timezone.utc), artifact_hashes=artifact_hashes, **values,
        )
        canonical = json.dumps(completion.model_copy(update={"result_hash": None}).model_dump(mode="json", exclude={"result_hash"}), sort_keys=True, separators=(",", ":"))
        completion = completion.model_copy(update={"result_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()})
        _write(paths["completion"], completion.model_dump(mode="json"))
        (job_dir / "status.json").write_text(json.dumps({"status": completion.status.value, "job_id": request.job_id, "run_id": request.run_id, "result_path": str(paths["completion"])}, indent=2, sort_keys=True), encoding="utf-8")
        self._notify_smithers(request, completion)
        self.registry.save_implementation_job(request.run_id, request.model_dump(mode="json"), str(job_dir), status.value, result_path=str(paths["completion"]), result_hash=completion.result_hash, error=completion.stderr_summary or None)
        return completion

    @staticmethod
    def _notify_smithers(request: ImplementationJobRequest, completion: ImplementationCompletion) -> None:
        smithers_run_id = request.provenance.get("smithers_run_id")
        script = Path(request.repository_root) / ".smithers" / "node_modules" / "smithers-orchestrator" / "src" / "bin" / "smithers.js"
        bun = shutil.which("bun")
        if not smithers_run_id or not bun or not script.is_file():
            return
        payload = json.dumps({"source_run_id": request.run_id, "job_id": request.job_id, "status": completion.status.value}, sort_keys=True)
        try:
            subprocess.run([bun, str(script), "signal", str(smithers_run_id), "external.codex.completed", "--correlation", request.run_id, "--data", payload], cwd=request.repository_root, capture_output=True, text=True, timeout=30, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def run(self, run_id: str) -> dict:
        request, job_dir = self.jobs.load_request(run_id)
        existing = self._existing_completion(job_dir)
        if existing and existing.status == CodexCompletionStatus.SUCCEEDED:
            return existing.model_dump(mode="json")
        try:
            self._verify_inputs(request, job_dir)
        except ValueError as exc:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_ARTIFACT_INTEGRITY, stderr_summary=str(exc))
            return completion.model_dump(mode="json")
        try:
            self._verify_worktree_path(request)
        except ValueError as exc:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_WORKTREE_CREATION, stderr_summary=redact_secrets(str(exc)))
            return completion.model_dump(mode="json")
        preflight = run_worktree_preflight(request.repository_root, persist=True)
        _write(job_dir / "preflight.json", preflight.model_dump(mode="json"))
        if not preflight.safe_for_isolated_worktree:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_WORKTREE_PREFLIGHT, stderr_summary="WORKTREE_PREFLIGHT_FAILED", artifact_paths={"preflight": str(preflight.report_path)})
            return completion.model_dump(mode="json")
        if not request.approved_worktree_root:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_WORKTREE_CREATION, stderr_summary="approved worktree root is missing")
            return completion.model_dump(mode="json")
        plan = ImplementationPlan(strategy_id=request.strategy_id, version=request.strategy_version, base_commit=request.base_commit, branch=request.branch, worktree_path=request.approved_worktree_root, allowed_files=request.allowed_paths, required_tests=request.required_tests, invariants=[], prohibited_actions=request.forbidden_paths, max_repair_attempts=0)
        manager = WorktreeManager(request.repository_root, Path(request.approved_worktree_root).parent)
        try:
            manager.create(plan, dry_run=False)
        except WorktreeError as exc:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_WORKTREE_CREATION, stderr_summary=redact_secrets(str(exc)))
            return completion.model_dump(mode="json")
        try:
            environment, environment_report = build_isolated_environment(
                plan.worktree_path,
                repository_root=request.repository_root,
            )
            _write(self._paths(job_dir)["environment_preflight"], sanitized_environment_report(environment_report))
            fixture_report = prepare_required_fixtures(request.repository_root, plan.worktree_path)
            _write(self._paths(job_dir)["fixture_preflight"], fixture_report)
            origin_report = import_origin_preflight(plan.worktree_path, environment=environment)
            _write(self._paths(job_dir)["import_origin"], origin_report)
        except (IsolationError, OSError, ValueError) as exc:
            completion = self._finish(
                request,
                job_dir,
                CodexCompletionStatus.FAILED_WORKTREE_PREFLIGHT,
                stderr_summary=redact_secrets(str(exc)),
                worktree_path=plan.worktree_path,
                base_commit=request.base_commit,
                resulting_commit=manager.current_commit(plan.worktree_path),
                artifact_paths={key: str(path) for key, path in self._paths(job_dir).items() if key in {"environment_preflight", "fixture_preflight", "import_origin"} and path.is_file()},
            )
            return completion.model_dump(mode="json")
        started = time.monotonic()
        prompt = (job_dir / "prompt.md").read_text(encoding="utf-8")
        codex = self.codex.run(
            prompt,
            plan.worktree_path,
            sandbox="workspace-write",
            timeout_seconds=request.timeout_seconds,
            dry_run=False,
            environment=environment,
            source_repository_root=request.repository_root,
        )
        invocation_path = self._paths(job_dir)["codex_invocation"]
        _write(invocation_path, codex.model_dump(mode="json"))
        if not codex.success:
            completion = self._finish(request, job_dir, CodexCompletionStatus.FAILED_CODEX_EXECUTION, exit_code=codex.exit_code, stdout_summary=codex.stdout[-4000:], stderr_summary=codex.stderr[-4000:], duration_ms=int((time.monotonic() - started) * 1000), worktree_path=plan.worktree_path, base_commit=request.base_commit, resulting_commit=manager.current_commit(plan.worktree_path), artifact_paths={"codex_invocation": str(invocation_path)})
            return completion.model_dump(mode="json")
        changed = manager.changed_files(plan)
        _write(self._paths(job_dir)["changed_files"], changed)
        try:
            verify_implementation_scope(changed, request.allowed_paths)
            scope = {"valid": True, "changed_files": changed}
            scope_status = CodexCompletionStatus.SUCCEEDED
        except Exception as exc:
            scope = {"valid": False, "changed_files": changed, "error": redact_secrets(str(exc))}
            scope_status = CodexCompletionStatus.FAILED_SCOPE_VALIDATION
        _write(self._paths(job_dir)["scope_validation"], scope)
        if scope_status != CodexCompletionStatus.SUCCEEDED:
            completion = self._finish(request, job_dir, scope_status, exit_code=codex.exit_code, stdout_summary=codex.stdout[-4000:], stderr_summary=scope.get("error", ""), duration_ms=int((time.monotonic() - started) * 1000), worktree_path=plan.worktree_path, base_commit=request.base_commit, resulting_commit=manager.current_commit(plan.worktree_path), changed_files=changed)
            return completion.model_dump(mode="json")
        test_results = []
        for index, command in enumerate(request.required_tests):
            # Keep pytest's temporary Git fixtures outside the long immutable
            # research-run path.  On Windows, nested temporary repositories
            # can otherwise make Git reject the synthetic test repo with
            # '$GIT_DIR' too big even though the worktree itself is valid.
            run_token = hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()[:12]
            basetemp = Path(tempfile.gettempdir()) / "research_pipeline_external" / f"{run_token}-{request.job_id}" / f"test-{index}"
            test = self.tests.run(
                plan.worktree_path,
                command,
                dry_run=False,
                report_path=job_dir / "result" / f"test-{index}.txt",
                source_repository_root=request.repository_root,
                environment=environment,
                basetemp=basetemp,
            )
            test_results.append(test.model_dump(mode="json"))
        _write(self._paths(job_dir)["test_results"], test_results)
        passed = all(item["passed"] for item in test_results)
        status = CodexCompletionStatus.SUCCEEDED if passed else CodexCompletionStatus.FAILED_REQUIRED_TESTS
        manifest = {"job_id": request.job_id, "run_id": request.run_id, "base_commit": request.base_commit, "resulting_commit": manager.current_commit(plan.worktree_path), "worktree_path": plan.worktree_path, "changed_files": changed}
        _write(self._paths(job_dir)["implementation_manifest"], manifest)
        output_hashes = {path.name: _hash(path) for path in self._paths(job_dir).values() if path.name != "completion.json" and path.is_file()}
        _write(self._paths(job_dir)["output_hash_manifest"], output_hashes)
        completion = self._finish(request, job_dir, status, exit_code=codex.exit_code, stdout_summary=codex.stdout[-4000:], stderr_summary="" if passed else "required implementation tests failed", duration_ms=int((time.monotonic() - started) * 1000), worktree_path=plan.worktree_path, base_commit=request.base_commit, resulting_commit=manager.current_commit(plan.worktree_path), changed_files=changed, artifact_paths={key: str(path) for key, path in self._paths(job_dir).items() if key != "completion"})
        return completion.model_dump(mode="json")

    def status(self, run_id: str) -> dict:
        return self.jobs.status(run_id)
