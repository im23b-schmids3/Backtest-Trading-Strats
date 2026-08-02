from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from research_pipeline.phase_b.models import CodexExecutionResult
from research_pipeline.phase_b.services import PhaseBService
from research_pipeline.phase_b.models import WorkflowInput
from research_pipeline.errors import ExternalSpecificationRequired
from research_pipeline.runners.codex_runner import CodexRunner, is_restricted_execution_failure
from research_pipeline.specification_executor.executor import ExternalSpecificationExecutor
from research_pipeline.specification_executor.jobs import SpecificationJobService
from research_pipeline.specification_executor.models import SpecificationJobType
from research_pipeline.phase_f1.models import MasterRunStatus
from research_pipeline.phase_f1.service import MasterPipelineService


def _git_repo(root: Path) -> None:
    root.mkdir()
    for command in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        ["git", "-C", str(root), "config", "user.name", "Specification Executor Test"],
    ):
        subprocess.run(command, check=True, shell=False)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, shell=False)


def test_restricted_codex_failure_is_classified_without_being_retried() -> None:
    result = CodexExecutionResult(success=False, executed=True, command=["codex", "exec"], cwd=".", sandbox="read-only", exit_code=1, stdout="", stderr="network access is disabled by tenant policy (os error 10013)", duration_ms=1, timed_out=False)
    assert is_restricted_execution_failure(result)


def test_codex_runner_uses_stdin_and_redacts_denial_output() -> None:
    calls: list[dict] = []

    def fake_process(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 1, "", "Authorization: Bearer secret-value; os error 10013")

    runner = CodexRunner(executable="codex", run_process=fake_process)
    result = runner.run("structured prompt", ".", sandbox="read-only", dry_run=False)
    assert calls[0]["input"] == "structured prompt"
    assert "secret-value" not in result.stderr
    assert is_restricted_execution_failure(result)


def test_spec_generation_policy_denial_does_not_consume_spec_attempt(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"

    def denied(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "network access is disabled by tenant policy (os error 10013)")

    service = PhaseBService(registry, CodexRunner(executable="codex", run_process=denied))
    workflow = WorkflowInput(strategy_name="Policy Denial Fixture", natural_language_description="A sufficiently detailed policy-denial fixture strategy.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(tmp_path), registry_path=str(registry), dry_run=False, max_generation_attempts=3, max_repair_attempts=2)
    with pytest.raises(ExternalSpecificationRequired, match="external specification generation required") as raised:
        service.generate_spec(workflow)
    run_id = service._intake_run_id(workflow, service._strategy_id(workflow.strategy_name))
    assert service.registry.specification_attempts(run_id) == []
    job = service.registry.get_latest_specification_job(run_id)
    assert job is not None and job["status"] == "WAITING_EXTERNAL_SPECIFICATION_GENERATION"
    assert raised.value.command.endswith(run_id)


def test_external_spec_executor_valid_result_is_idempotent_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    workflow = WorkflowInput(strategy_name="Executor Fixture", natural_language_description="A sufficiently detailed specification executor fixture.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(registry), dry_run=False, run_id="run-executor-valid")
    phase_b = PhaseBService(registry)
    spec = phase_b._dry_spec(workflow, "Executor-Fixture", "phase-b-1", [])
    raw = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
    created = SpecificationJobService(registry).create(workflow, strategy_id="Executor-Fixture", strategy_version="phase-b-1", run_id=workflow.run_id or "", attempt=1, job_type=SpecificationJobType.GENERATE_SPECIFICATION, prompt="Return one mapping.")
    calls: list[dict] = []

    def fake_process(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, raw, "")

    executor = ExternalSpecificationExecutor(registry, codex_runner=CodexRunner(executable="codex", run_process=fake_process))
    result = executor.run(workflow.run_id or "")
    assert result["status"] == "SUCCEEDED"
    assert "--sandbox" in calls[0]["command"] and calls[0]["command"][calls[0]["command"].index("--sandbox") + 1] == "read-only"
    assert calls[0]["input"] == "Return one mapping."
    executor.codex = CodexRunner(executable="codex", run_process=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("successful job reran Codex")))
    assert executor.run(workflow.run_id or "")["result_hash"] == result["result_hash"]
    assert created["idempotent_reuse"] is False


def test_external_spec_executor_rejects_multiple_payloads_and_detects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    workflow = WorkflowInput(strategy_name="Executor Failure Fixture", natural_language_description="A sufficiently detailed specification executor failure fixture.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(registry), dry_run=False, run_id="run-executor-failure")
    SpecificationJobService(registry).create(workflow, strategy_id="Executor-Failure-Fixture", strategy_version="phase-b-1", run_id=workflow.run_id or "", attempt=1, job_type=SpecificationJobType.GENERATE_SPECIFICATION, prompt="Return one mapping.")

    def malformed(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "a: 1\n---\nb: 2\n", "")

    result = ExternalSpecificationExecutor(registry, codex_runner=CodexRunner(executable="codex", run_process=malformed)).run(workflow.run_id or "")
    assert result["status"] == "FAILED_OUTPUT_EXTRACTION"
    raw_path = Path(result["raw_output_path"])
    assert raw_path.is_file()

    workflow2 = workflow.model_copy(update={"run_id": "run-executor-mutation"})
    SpecificationJobService(registry).create(workflow2, strategy_id="Executor-Failure-Fixture", strategy_version="phase-b-1", run_id=workflow2.run_id or "", attempt=1, job_type=SpecificationJobType.GENERATE_SPECIFICATION, prompt="Return one mapping.")

    def mutating(command, **kwargs):
        Path(kwargs["cwd"], "README.md").write_text("mutated\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "a: 1\n", "")

    result2 = ExternalSpecificationExecutor(registry, codex_runner=CodexRunner(executable="codex", run_process=mutating)).run(workflow2.run_id or "")
    assert result2["status"] == "FAILED_ARTIFACT_INTEGRITY"
    assert any("README.md" in item for item in result2["repository_mutations"])


def test_f1_restricted_intake_pauses_then_resumes_to_single_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    intake = tmp_path / "intake.yaml"
    intake.write_text(yaml.safe_dump({"strategy_name": "External Intake Fixture", "description": "A sufficiently detailed external specification intake fixture.", "markets": ["TEST"], "timeframes": ["1h"], "confirmed_facts": ["fixture only"], "assumptions": [], "missing_information": [], "ambiguities": []}, sort_keys=False), encoding="utf-8")

    denied = lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "network access is disabled by tenant policy (os error 10013)")
    import research_pipeline.phase_f1.service as f1_module
    real_phase_b = PhaseBService
    monkeypatch.setattr(f1_module, "PhaseBService", lambda path: real_phase_b(path, CodexRunner(executable="codex", run_process=denied)))
    service = MasterPipelineService(registry, root)
    options = service.input_model(intake, root, registry_path=registry, dry_run=False, mode="real_run", implementation_enabled=True)
    paused = service.start(options)
    assert paused["outcome"] == MasterRunStatus.WAITING_EXTERNAL_SPECIFICATION_GENERATION.value
    assert paused["next_command"].endswith(paused["run_id"])
    job = service.registry.get_latest_specification_job(paused["run_id"])
    assert job is not None and job["status"] == "WAITING_EXTERNAL_SPECIFICATION_GENERATION"

    request = SpecificationJobService(registry).load_request(paused["run_id"])[0]
    workflow = WorkflowInput(strategy_name="External Intake Fixture", natural_language_description="A sufficiently detailed external specification intake fixture.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(registry), dry_run=False, run_id=paused["run_id"])
    spec = real_phase_b(registry)._dry_spec(workflow, request.strategy_id, request.strategy_version, [])
    raw = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
    executor = ExternalSpecificationExecutor(registry, codex_runner=CodexRunner(executable="codex", run_process=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, raw, "")))
    assert executor.run(paused["run_id"])["status"] == "SUCCEEDED"
    resumed = service.resume(paused["run_id"])
    assert resumed["outcome"] == MasterRunStatus.WAITING_FOR_APPROVAL.value
    assert resumed["approval_status"] == "PENDING"


def test_invalid_external_draft_creates_repair_and_valid_repair_reaches_approval_ready(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    workflow = WorkflowInput(strategy_name="Repair Fixture", natural_language_description="A sufficiently detailed external specification repair fixture.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(registry), dry_run=False, run_id="run-repair")
    denied = lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "network access is disabled by tenant policy (os error 10013)")
    phase_b = PhaseBService(registry, CodexRunner(executable="codex", run_process=denied))
    with pytest.raises(Exception):
        phase_b.generate_spec(workflow)
    executor = ExternalSpecificationExecutor(registry, codex_runner=CodexRunner(executable="codex", run_process=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "strategy_id: Repair Fixture\nversion: phase-b-1\n", "")))
    assert executor.run(workflow.run_id or "")["status"] == "SUCCEEDED"
    with pytest.raises(Exception, match="external specification repair required"):
        phase_b.generate_spec(workflow)
    jobs = SpecificationJobService(registry)
    repair_job = jobs.registry.get_latest_specification_job(workflow.run_id or "")
    assert repair_job is not None and repair_job["attempt"] == 2 and repair_job["job_type"] == "REPAIR_SPECIFICATION"
    repair_request = jobs.load_request(workflow.run_id or "")[0]
    valid = phase_b._dry_spec(workflow, repair_request.strategy_id, repair_request.strategy_version, [])
    valid_raw = yaml.safe_dump(valid.model_dump(mode="json"), sort_keys=False)
    executor.codex = CodexRunner(executable="codex", run_process=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, valid_raw, ""))
    assert executor.run(workflow.run_id or "")["status"] == "SUCCEEDED"
    generated = phase_b.generate_spec(workflow)
    assert generated.attempt == 2
    assert phase_b.validate_spec(generated).valid


def test_external_repair_budget_exhaustion_is_terminal(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    workflow = WorkflowInput(strategy_name="Exhausted Repair Fixture", natural_language_description="A sufficiently detailed exhausted external specification repair fixture.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root), registry_path=str(registry), dry_run=False, run_id="run-exhausted", max_generation_attempts=3, max_repair_attempts=2)
    denied = lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "network access is disabled by tenant policy (os error 10013)")
    phase_b = PhaseBService(registry, CodexRunner(executable="codex", run_process=denied))
    with pytest.raises(Exception):
        phase_b.generate_spec(workflow)
    invalid_runner = CodexRunner(executable="codex", run_process=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "strategy_id: only-one-field\n", ""))
    executor = ExternalSpecificationExecutor(registry, codex_runner=invalid_runner)
    for _ in range(3):
        assert executor.run(workflow.run_id or "")["status"] == "SUCCEEDED"
        try:
            phase_b.generate_spec(workflow)
        except Exception as exc:
            if "SPECIFICATION_GENERATION_FAILURE" in str(exc):
                break
    failure = phase_b.registry.specification_failure(workflow.run_id or "")
    assert failure is not None and failure["classification"] == "SPECIFICATION_GENERATION_FAILURE"
    assert len(phase_b.registry.specification_attempts(workflow.run_id or "")) == 3
