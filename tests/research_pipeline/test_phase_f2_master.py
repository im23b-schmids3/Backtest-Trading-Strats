from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_pipeline.errors import SpecificationValidationError
from research_pipeline.implementation.executor import ExternalCodexExecutor
from research_pipeline.implementation.jobs import ImplementationJobService
from research_pipeline.implementation.models import CodexCompletionStatus
from research_pipeline.phase_f1.service import MasterPipelineService
from research_pipeline.phase_f1.models import FinalClassification, MasterRunStatus, MasterStep
from research_pipeline.repository.worktree_preflight import WorktreePreflightReport
from research_pipeline.runners.codex_runner import CodexRunner


ROOT = Path(__file__).parents[2]
INTAKE = ROOT / "configs/research_pipeline/phase_f2_real_demo_intake.yaml"
SPEC = ROOT / "research_registry/spec_drafts/F2-real-breakout-demo_vphase-b-1.yaml"


def _use_isolated_run_root(service: MasterPipelineService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic test run IDs away from existing repository artifacts."""

    def isolated_root(strategy_id: str, run_id: str) -> Path:
        root = tmp_path / "research_runs" / strategy_id / run_id
        for name in ("run", "specification", "implementation", "verification", "research", "prop", "portfolio", "report", "archive"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        service,
        "_root",
        isolated_root,
    )


def test_real_mode_is_explicit_and_approval_gated(tmp_path):
    service = MasterPipelineService(tmp_path / "registry.sqlite3", ROOT)
    options = service.input_model(INTAKE, ROOT, registry_path=tmp_path / "registry.sqlite3", dry_run=False, mode="real_run", allow_proxy_data=True)
    options = options.model_copy(update={"prebuilt_spec_path": str(SPEC)})
    started = service.start(options)
    assert started["current_step"] == "APPROVAL"
    assert started["approval_status"] == "PENDING"
    assert service.resume(started["run_id"])["current_step"] == "APPROVAL"


def test_real_demo_uses_external_implementation_job_before_b5(tmp_path, monkeypatch: pytest.MonkeyPatch):
    service = MasterPipelineService(tmp_path / "registry.sqlite3", ROOT)
    _use_isolated_run_root(service, tmp_path, monkeypatch)
    options = service.input_model(INTAKE, ROOT, registry_path=tmp_path / "registry.sqlite3", dry_run=False, mode="real_run", allow_proxy_data=True)
    options = options.model_copy(update={"prebuilt_spec_path": str(SPEC)})
    started = service.start(options); run_id = started["run_id"]
    service.approve(run_id, "APPROVE", "bounded deterministic real-data fixture")

    # This fixture repository intentionally contains tracked historical
    # runtime outputs. Keep the production preflight gate intact and replace
    # only its result here so this test can reach the external-job contract.
    safe_preflight = WorktreePreflightReport(
        repository_root=str(ROOT.resolve()),
        git_head="fixture-head",
        checked_at=datetime.now(timezone.utc),
        max_path_length=240,
        tracked_path_count=0,
        safe_for_isolated_worktree=True,
        report_path=str(tmp_path / "preflight.json"),
    )
    monkeypatch.setattr("research_pipeline.implementation.jobs.run_worktree_preflight", lambda *args, **kwargs: safe_preflight)

    codex_calls: list[tuple[tuple, dict]] = []

    def unexpected_codex_call(*args, **kwargs):
        codex_calls.append((args, kwargs))
        raise AssertionError("real-mode implementation must not invoke Codex inside the controller")

    monkeypatch.setattr(CodexRunner, "run", unexpected_codex_call)
    paused = service.resume(run_id)

    assert paused["current_step"] == MasterStep.IMPLEMENTATION.value
    assert paused["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value
    assert paused["pipeline_status"] == "WAITING_EXTERNAL_CODEX"
    assert paused["next_command"] == f"py -m research_pipeline codex-executor run {run_id}"
    assert codex_calls == []

    journal = service.registry.master_journal(run_id)
    assert any(item["event"] == "EXTERNAL_CODEX_EXECUTION_REQUIRED" for item in journal)

    jobs = ImplementationJobService(tmp_path / "registry.sqlite3")
    request, job_dir = jobs.load_request(run_id)
    job = service.registry.get_implementation_job(run_id)
    spec = service.registry.get_specification(started["strategy_id"])
    assert job is not None
    assert job["status"] == "WAITING_EXTERNAL_CODEX"
    assert job["result_path"] is None
    assert request.specification_hash == spec.specification_hash
    assert request.base_commit
    assert request.provenance["repository_root"] == str(ROOT.resolve())
    assert request.provenance["preflight_report"]
    assert service.registry.get_master_phase_result(run_id, MasterStep.IMPLEMENTATION.value) is None
    assert jobs.create(run_id)["idempotent_reuse"] is True

    # Build a valid, hash-checked completion through the external executor's
    # durable completion writer. This is a mocked external result; no Codex
    # process and no worktree mutation occur in this test.
    external = ExternalCodexExecutor(tmp_path / "registry.sqlite3")
    completion = external._finish(
        request,
        job_dir,
        CodexCompletionStatus.SUCCEEDED,
        worktree_path=str(ROOT.resolve()),
        base_commit=request.base_commit,
        resulting_commit=request.base_commit,
        changed_files=[],
        artifact_paths={},
    )
    assert completion.status == CodexCompletionStatus.SUCCEEDED
    assert service.registry.get_implementation_job(run_id)["status"] == "SUCCEEDED"

    first = service.resume(run_id)
    assert first["current_step"] == MasterStep.IMPLEMENTATION_VERIFICATION.value
    assert first["outcome"] == MasterRunStatus.IMPLEMENTATION_VERIFICATION_REQUIRED.value
    assert first["pipeline_status"] == "IMPLEMENTATION_VERIFICATION_REQUIRED"
    assert first["next_command"] == f"py -m research_pipeline resume {run_id}"
    assert first["b5_available"] is False
    assert service.registry.get_master_phase_result(run_id, MasterStep.IMPLEMENTATION.value) is not None
    assert service.registry.get_master_phase_result(run_id, MasterStep.IMPLEMENTATION_VERIFICATION.value) is None
    assert service.registry.get_master_phase_result(run_id, MasterStep.TECHNICAL_VERIFICATION.value) is None
    assert service.registry.get_implementation_job(run_id)["status"] == "INGESTED"

    # Resume crosses the verification boundary deterministically.  B.5 and
    # the remaining pipeline are unreachable until implementation verification
    # succeeds.
    verified = service.resume(run_id)
    assert verified["current_step"] == MasterStep.COMPLETED.value
    assert verified["outcome"] == MasterRunStatus.SUCCESS.value
    assert verified["b5_available"] is True

    # A completed external result is cached; neither the executor nor the
    # controller starts a second Codex implementation attempt.
    assert external.run(run_id)["status"] == CodexCompletionStatus.SUCCEEDED.value
    final = service.resume(run_id)
    assert final == verified
    assert codex_calls == []
    assert final["current_step"] == "COMPLETED"
    assert final["outcome"] == "SUCCESS"
    report = service.report(run_id)
    assert report["mode"] == "real_run"
    assert report["classification"] in {item.value for item in FinalClassification}
    assert report["confidence"] == "REAL_LOCAL_DATA_DEMONSTRATION"


def test_real_implementation_verification_failure_does_not_enter_b5(tmp_path, monkeypatch: pytest.MonkeyPatch):
    service = MasterPipelineService(tmp_path / "registry.sqlite3", ROOT)
    _use_isolated_run_root(service, tmp_path, monkeypatch)
    options = service.input_model(INTAKE, ROOT, registry_path=tmp_path / "registry.sqlite3", dry_run=False, mode="real_run", allow_proxy_data=True)
    options = options.model_copy(update={"prebuilt_spec_path": str(SPEC)})
    started = service.start(options)
    service.approve(started["run_id"], "APPROVE", "verification failure fixture")
    run_id = started["run_id"]
    monkeypatch.setattr("research_pipeline.implementation.jobs.run_worktree_preflight", lambda *args, **kwargs: WorktreePreflightReport(
        repository_root=str(ROOT.resolve()), git_head="fixture-head", checked_at=datetime.now(timezone.utc),
        max_path_length=240, tracked_path_count=0, safe_for_isolated_worktree=True,
        report_path=str(tmp_path / "preflight.json")))
    service.resume(run_id)
    request, job_dir = ImplementationJobService(tmp_path / "registry.sqlite3").load_request(run_id)
    ExternalCodexExecutor(tmp_path / "registry.sqlite3")._finish(
        request, job_dir, CodexCompletionStatus.SUCCEEDED, worktree_path=str(ROOT.resolve()),
        base_commit=request.base_commit, resulting_commit=request.base_commit, changed_files=[], artifact_paths={})
    assert service.resume(run_id)["current_step"] == MasterStep.IMPLEMENTATION_VERIFICATION.value
    monkeypatch.setattr(service, "_technical_verification", lambda *args, **kwargs: (_ for _ in ()).throw(SpecificationValidationError("verification failed")))
    failed = service.resume(run_id)
    assert failed["current_step"] == MasterStep.IMPLEMENTATION_VERIFICATION.value
    assert failed["outcome"] == MasterRunStatus.TECHNICAL_FAILURE.value
    assert failed["b5_available"] is False


def test_portfolio_deferral_does_not_override_standalone_precedence():
    assert MasterPipelineService._classification("ACCEPTED_STANDALONE", "PROP_ACCEPTED_STANDALONE", "INSUFFICIENT_EVIDENCE", real_mode=True) == FinalClassification.ACCEPTED_STANDALONE
