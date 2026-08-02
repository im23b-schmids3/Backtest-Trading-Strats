from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research_pipeline.errors import RegistryError
from research_pipeline.implementation.executor import ExternalCodexExecutor
from research_pipeline.implementation.jobs import ImplementationJobService
from research_pipeline.implementation.models import CodexCompletionStatus
from research_pipeline.phase_f1.models import MasterRunStatus, MasterStep
from research_pipeline.phase_f1.service import MasterPipelineService
from research_pipeline.repository.worktree_preflight import WorktreePreflightReport
from research_pipeline.value_area_trap.data import AggregateTradeManifest


ROOT = Path(__file__).parents[2]
SPEC = ROOT / "research_registry/spec_drafts/F2-real-breakout-demo_vphase-b-1.yaml"
DATASET_HASH = "908a22b85825a2c58cdf60d748500d403c16e57b52648a2376290547088f2b10"


def _git_repo(root: Path) -> None:
    root.mkdir(parents=True)
    for command in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        ["git", "-C", str(root), "config", "user.name", "Retry Test"],
    ):
        subprocess.run(command, check=True, shell=False)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    (root / "research_registry" / "spec_drafts").mkdir(parents=True)
    (root / "tests" / "fixtures").mkdir(parents=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, shell=False)


def _manifest(root: Path) -> Path:
    directory = root / "data" / DATASET_HASH
    directory.mkdir(parents=True)
    model = AggregateTradeManifest.model_construct(
        date_start=date(2026, 4, 1), date_end=date(2026, 4, 30),
        retrieved_at=datetime(2026, 5, 1, tzinfo=timezone.utc), source_files=["archive.zip"],
        source_file_hashes={"archive.zip": "a" * 64}, normalized_dataset_hash=DATASET_HASH,
        row_count=41_544_041, duplicate_count=0, manifest_hash="pending",
    )
    payload = model.model_dump(mode="json")
    payload["manifest_hash"] = MasterPipelineService._manifest_integrity_hash(AggregateTradeManifest.model_validate(payload))
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    (directory / "aggregate_trades.parquet").write_bytes(b"placeholder")
    return path


def _prepared_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    _git_repo(root)
    registry = tmp_path / "registry.sqlite3"
    intake = tmp_path / "intake.json"
    intake.write_text(json.dumps({"strategy_name": "F2-real-breakout-demo", "description": "A durable retry fixture for external implementation jobs.", "markets": ["SPX"], "timeframes": ["1h"]}), encoding="utf-8")
    service = MasterPipelineService(registry, root)
    options = service.input_model(intake, root, registry_path=registry, data_manifest_path=_manifest(root), allow_proxy_data=True)
    started = service.start(options.model_copy(update={"prebuilt_spec_path": str(SPEC)}))
    service.approve(started["run_id"], "APPROVE", "retry fixture")
    safe = WorktreePreflightReport(repository_root=str(root), git_head="fixture", checked_at=datetime.now(timezone.utc), max_path_length=240, tracked_path_count=1, safe_for_isolated_worktree=True, report_path=str(tmp_path / "preflight.json"))
    monkeypatch.setattr("research_pipeline.implementation.jobs.run_worktree_preflight", lambda *args, **kwargs: safe)
    jobs = ImplementationJobService(registry)
    created = jobs.create(started["run_id"])
    return service, jobs, started["run_id"], created


def _make_stale(jobs: ImplementationJobService, run_id: str) -> tuple[Path, dict]:
    request, job_dir = jobs.load_request(run_id)
    result = job_dir / "result"
    result.mkdir()
    (result / "changed_files.json").write_text("[]", encoding="utf-8")
    (result / "codex_invocation.json").write_text(json.dumps({"exit_code": None}), encoding="utf-8")
    (job_dir / "status.json").write_text(json.dumps({"status": "WAITING_EXTERNAL_CODEX", "job_id": request.job_id, "run_id": run_id, "owner_pid": 999999, "heartbeat_at": "2000-01-01T00:00:00+00:00"}), encoding="utf-8")
    return job_dir, request.model_dump(mode="json")


def test_stale_interrupted_job_is_immutable_and_retry_uses_fresh_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, jobs, run_id, first = _prepared_job(tmp_path, monkeypatch)
    old_dir, old_request = _make_stale(jobs, run_id)
    evidence = jobs.stale_interruption(run_id, stale_after_seconds=1)
    assert evidence["stale"] and evidence["owner_live"] is False

    retried = jobs.retry(run_id, stale_after_seconds=1)
    second = retried["retry"]
    assert retried["interrupted"]["classification"] == "INTERRUPTED"
    assert (old_dir / "request.json").is_file()
    assert json.loads((old_dir / "request.json").read_text(encoding="utf-8")) == old_request
    assert (old_dir / "result" / "completion.json").exists() is False
    assert (old_dir / "result" / "interruption.json").is_file()
    assert second["job"]["job_id"] == "job-002"
    assert second["job"]["approved_worktree_root"] != first["job"]["approved_worktree_root"]
    assert Path(second["job_path"]) != old_dir
    assert service.registry.get_implementation_job(run_id)["job_id"] == "job-002"
    assert service.registry.get_master_run(run_id)["resume_state_json"]["implementation_job_id"] == "job-002"
    resumed_state = service.registry.get_master_run(run_id)["resume_state_json"]
    assert resumed_state["real_data_context"]["dataset_hash"] == DATASET_HASH
    assert resumed_state["real_data_context"]["execution_mode"] == "REAL_DATA"
    assert "FIXTURE" not in json.dumps(resumed_state, sort_keys=True)
    attempts = service.registry.list_implementation_jobs(run_id)
    assert [item["job_id"] for item in attempts] == ["job-001", "job-002"]
    assert attempts[0]["status"] == "INTERRUPTED"


def test_active_job_can_be_marked_aborted_without_completion_and_terminal_jobs_are_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, jobs, run_id, _ = _prepared_job(tmp_path, monkeypatch)
    jobs.mark_running(run_id)
    aborted = jobs.abort_by_user(run_id, reason="Ctrl+C")
    assert aborted["classification"] == "ABORTED_BY_USER"
    assert not (Path(service.registry.get_implementation_job(run_id)["job_path"]) / "result" / "completion.json").exists()
    with pytest.raises(RegistryError, match="cannot be retried in place"):
        jobs.retry(run_id, stale_after_seconds=1)


def _finish_terminal(
    jobs: ImplementationJobService,
    run_id: str,
    status: CodexCompletionStatus,
    *,
    stderr_summary: str = "",
) -> None:
    request, job_dir = jobs.load_request(run_id)
    ExternalCodexExecutor(jobs.registry_path)._finish(
        request,
        job_dir,
        status,
        stderr_summary=stderr_summary,
        worktree_path=str(Path(request.repository_root)),
        base_commit=request.base_commit,
        resulting_commit=request.base_commit,
        changed_files=[],
        artifact_paths={},
    )


def test_failed_codex_completion_reconciles_without_test_results_and_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, jobs, run_id, _ = _prepared_job(tmp_path, monkeypatch)
    _finish_terminal(jobs, run_id, CodexCompletionStatus.FAILED_CODEX_EXECUTION, stderr_summary="Codex unavailable")
    job_dir = Path(service.registry.get_implementation_job(run_id)["job_path"])
    assert (job_dir / "result" / "completion.json").is_file()
    assert not (job_dir / "result" / "test_results.json").exists()

    first = jobs.reconcile(run_id)
    second = jobs.reconcile(run_id)
    status = service.status(run_id)
    master = service.registry.get_master_run(run_id)
    assert first == second
    assert first["completion_status"] == "FAILED_CODEX_EXECUTION"
    assert status["current_step"] == MasterStep.IMPLEMENTATION.value
    assert status["outcome"] == MasterRunStatus.IMPLEMENTATION_FAILURE.value
    assert status["pipeline_status"] == "IMPLEMENTATION_FAILED"
    assert status["b5_available"] is False
    assert master["resume_state_json"]["final_reason"] == "Codex unavailable"
    assert master["resume_state_json"]["implementation_test_status"] is None
    assert master["resume_state_json"]["real_data_context"]["dataset_hash"] == DATASET_HASH
    assert master["resume_state_json"]["real_data_context"]["execution_mode"] == "REAL_DATA"
    assert "FIXTURE" not in json.dumps(master["resume_state_json"], sort_keys=True)
    events = [item for item in service.registry.master_journal(run_id) if item["event"] == "IMPLEMENTATION_COMPLETION_RECONCILED"]
    assert len(events) == 1
    assert service.resume(run_id) == status


def test_failed_required_tests_and_completed_results_reconcile_to_their_required_next_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    failed_service, failed_jobs, failed_run, _ = _prepared_job(tmp_path / "failed", monkeypatch)
    _finish_terminal(failed_jobs, failed_run, CodexCompletionStatus.FAILED_REQUIRED_TESTS, stderr_summary="required tests failed")
    failed = failed_jobs.reconcile(failed_run)
    assert failed["outcome"] == MasterRunStatus.IMPLEMENTATION_FAILURE.value
    assert failed["implementation_test_status"] == "FAILED_REQUIRED_TESTS"
    assert failed_service.status(failed_run)["b5_available"] is False

    complete_service, complete_jobs, complete_run, _ = _prepared_job(tmp_path / "complete", monkeypatch)
    _finish_terminal(complete_jobs, complete_run, CodexCompletionStatus.COMPLETED)
    completed = complete_jobs.reconcile(complete_run)
    assert completed["job_status"] == "INGESTED"
    assert completed["current_step"] == MasterStep.IMPLEMENTATION_VERIFICATION.value
    assert completed["outcome"] == MasterRunStatus.IMPLEMENTATION_VERIFICATION_REQUIRED.value
    assert complete_service.status(complete_run)["b5_available"] is False


def test_interrupted_terminal_job_reconciles_without_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, jobs, run_id, _ = _prepared_job(tmp_path, monkeypatch)
    jobs.mark_running(run_id)
    jobs.abort_by_user(run_id, reason="Ctrl+C")
    reconciled = jobs.reconcile(run_id)
    status = service.status(run_id)
    assert reconciled["completion_status"] == "ABORTED_BY_USER"
    assert status["outcome"] == MasterRunStatus.IMPLEMENTATION_FAILURE.value
    assert status["current_step"] == MasterStep.IMPLEMENTATION.value
    assert status["b5_available"] is False


def test_proven_legacy_timeout_reclassification_preserves_artifacts_and_retry_creates_job_003(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, jobs, run_id, _ = _prepared_job(tmp_path, monkeypatch)
    old_dir, _ = _make_stale(jobs, run_id)
    jobs.retry(run_id, stale_after_seconds=1)
    request, job_dir = jobs.load_request(run_id)
    invocation_path = job_dir / "result" / "codex_invocation.json"
    invocation_path.parent.mkdir(exist_ok=True)
    invocation_path.write_text(json.dumps({"timed_out": True, "configured_timeout_seconds": 900}), encoding="utf-8")
    ExternalCodexExecutor(jobs.registry_path)._finish(
        request, job_dir, CodexCompletionStatus.FAILED_CODEX_EXECUTION,
        exit_code=None, stderr_summary="timeout", worktree_path=str(Path(request.repository_root)),
        base_commit=request.base_commit, resulting_commit=request.base_commit, changed_files=[], artifact_paths={},
    )
    second = service.registry.get_implementation_job(run_id)
    second_dir = Path(second["job_path"])
    request_before = (second_dir / "request.json").read_bytes()
    completion_before = (second_dir / "result" / "completion.json").read_bytes()
    old_worktree = json.loads(request_before)["approved_worktree_root"]

    reclassified = jobs.reclassify_legacy_timeout(run_id)
    repeated = jobs.reclassify_legacy_timeout(run_id)
    assert reclassified["idempotent"] is False and repeated["idempotent"] is True
    assert reclassified["correction"]["original_status"] == "FAILED_CODEX_EXECUTION"
    assert reclassified["correction"]["corrected_status"] == "TIMED_OUT"
    assert service.registry.get_implementation_job(run_id)["status"] == "TIMED_OUT"
    assert service.registry.list_implementation_jobs(run_id)[1]["status"] == "FAILED_CODEX_EXECUTION"

    retried = jobs.retry(run_id)
    third = retried["retry"]
    state = service.registry.get_master_run(run_id)["resume_state_json"]
    assert third["job"]["job_id"] == "job-003"
    assert third["job"]["approved_worktree_root"] != old_worktree
    assert Path(third["job_path"]) != old_dir
    assert (second_dir / "request.json").read_bytes() == request_before
    assert (second_dir / "result" / "completion.json").read_bytes() == completion_before
    assert state["implementation_job_id"] == "job-003"
    assert state["real_data_context"]["dataset_hash"] == DATASET_HASH
    assert state["real_data_context"]["execution_mode"] == "REAL_DATA"
    attempts = service.registry.list_implementation_jobs(run_id)
    assert [(item["job_id"], item["status"]) for item in attempts] == [
        ("job-001", "INTERRUPTED"),
        ("job-002", "FAILED_CODEX_EXECUTION"),
        ("job-003", "WAITING_EXTERNAL_CODEX"),
    ]
    assert service.status(run_id)["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value


@pytest.mark.parametrize(
    ("exit_code", "timed_out", "message"),
    [(None, False, "timed_out=true"), (1, True, "null completion exit_code")],
)
def test_legacy_timeout_reclassification_requires_complete_immutable_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int | None, timed_out: bool, message: str):
    _, jobs, run_id, _ = _prepared_job(tmp_path, monkeypatch)
    request, job_dir = jobs.load_request(run_id)
    invocation_path = job_dir / "result" / "codex_invocation.json"
    invocation_path.parent.mkdir(exist_ok=True)
    invocation_path.write_text(json.dumps({"timed_out": timed_out}), encoding="utf-8")
    ExternalCodexExecutor(jobs.registry_path)._finish(
        request, job_dir, CodexCompletionStatus.FAILED_CODEX_EXECUTION,
        exit_code=exit_code, stderr_summary="failure", worktree_path=str(Path(request.repository_root)),
        base_commit=request.base_commit, resulting_commit=request.base_commit, changed_files=[], artifact_paths={},
    )
    completion_before = (job_dir / "result" / "completion.json").read_bytes()
    with pytest.raises(RegistryError, match=message):
        jobs.reclassify_legacy_timeout(run_id)
    assert (job_dir / "result" / "completion.json").read_bytes() == completion_before
