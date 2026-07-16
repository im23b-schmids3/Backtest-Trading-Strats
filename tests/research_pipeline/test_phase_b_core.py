from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from research_pipeline.enums import PipelineState
from research_pipeline.errors import ImmutableSpecificationError, InvalidTransitionError
from research_pipeline.phase_b.models import GeneratedStrategySpec, WorkflowInput
from research_pipeline.phase_b.prompt_builder import build_implementation_prompt
from research_pipeline.phase_b.services import PhaseBService


def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    commands = [
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "phase-b@example.invalid"],
        ["git", "-C", str(root), "config", "user.name", "Phase B Tests"],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True, shell=False)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True, capture_output=True, text=True, shell=False)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, capture_output=True, text=True, shell=False)
    return root


def workflow_input(root: Path, **overrides: object) -> WorkflowInput:
    data: dict[str, object] = {
        "strategy_name": "fictional-phase-b-fixture",
        "natural_language_description": "A fictional deterministic entry and exit rule for a workflow fixture.",
        "requested_markets": ["TEST"],
        "requested_timeframes": ["1h"],
        "repository_root": str(root),
        "dry_run": True,
        "implementation_enabled": False,
    }
    data.update(overrides)
    return WorkflowInput.model_validate(data)


def registered_service(tmp_path: Path) -> tuple[PhaseBService, WorkflowInput]:
    root = git_repo(tmp_path)
    service = PhaseBService(tmp_path / "registry.sqlite3")
    workflow = workflow_input(root)
    generated = service.generate_spec(workflow)
    validation = service.validate_spec(generated)
    assert validation.valid
    registration = service.register_generated(validation)
    assert registration.current_phase is PipelineState.WAITING_FOR_SPEC_APPROVAL
    return service, workflow


def test_workflow_input_and_generated_spec_contracts(tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    value = workflow_input(root)
    assert value.dry_run is True
    with pytest.raises(ValidationError):
        WorkflowInput.model_validate({**value.model_dump(), "strategy_name": "../unsafe"})

    service = PhaseBService(tmp_path / "registry.sqlite3")
    generated = service.generate_spec(value)
    assert generated.specification_path.endswith("phase-b-1.yaml")
    assert generated.manual_review_required is False
    assert GeneratedStrategySpec.model_validate(generated.model_dump()) == generated
    assert "invariants" in generated.approval_summary


def test_ambiguous_spec_requires_manual_review(tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    service = PhaseBService(tmp_path / "registry.sqlite3")
    generated = service.generate_spec(workflow_input(root, natural_language_description="This is ambiguous; maybe use a filter."))
    assert generated.manual_review_required is True
    assert generated.ambiguities


def test_registration_approval_and_immutability_are_idempotent(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    validation = service.validate_spec(generated)
    first = service.register_generated(validation)
    second = service.register_generated(validation)
    assert first.idempotent_reuse is True
    assert second.idempotent_reuse is True

    with pytest.raises(InvalidTransitionError):
        service.implementation_plan(generated.strategy_id, workflow.repository_root)

    approved = service.approve(generated.strategy_id, "APPROVE", "fixture approval")
    assert approved.approved and approved.immutable_verified
    assert approved.current_phase is PipelineState.IMPLEMENTATION
    with pytest.raises(ImmutableSpecificationError):
        service.registry.approve_specification(generated.strategy_id, generated.version, service.registry.get_specification(generated.strategy_id))


def test_rejection_prevents_implementation(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    result = service.approve(generated.strategy_id, "REJECT", "fixture rejected")
    assert result.current_phase is PipelineState.REJECTED
    with pytest.raises(InvalidTransitionError):
        service.implementation_plan(generated.strategy_id, workflow.repository_root)


def test_dry_run_reaches_verification_and_resume_is_idempotent(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    service.approve(generated.strategy_id, "APPROVE")
    plan = service.implementation_plan(generated.strategy_id, workflow.repository_root, dry_run=True)
    prompt = service.build_implementation_prompt(generated.strategy_id, plan)
    assert "This fictional dry run" in prompt
    assert "holdout results" in prompt.lower()
    execution = service.execute_codex(generated.strategy_id, workflow.repository_root, plan, prompt, dry_run=True, task_name="implementation")
    assert execution.executed is False
    tests = service.run_tests(workflow.repository_root, dry_run=True, worktree_path=plan.worktree_path)
    summary = service.final_status(generated.strategy_id, tests, implementation_executed=False, repair_attempts=0)
    assert summary.final_state is PipelineState.IMPLEMENTATION_VERIFICATION
    service.final_status(generated.strategy_id, tests, implementation_executed=False, repair_attempts=0)
    assert len(service.registry.history(generated.strategy_id)["experiments"]) == 1


def test_implementation_prompt_is_structured_and_protective(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    service.approve(generated.strategy_id, "APPROVE")
    plan = service.implementation_plan(generated.strategy_id, workflow.repository_root)
    prompt = build_implementation_prompt(service.registry.get_specification(generated.strategy_id), plan.allowed_files, plan.required_tests, plan.max_repair_attempts)
    assert "invariants" in prompt.lower()
    assert "do not run backtests" in prompt.lower()
    assert "holdout results" in prompt.lower()
    assert "holdout trades" not in prompt.lower()


def test_structured_bridge_payload_is_json_serializable(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    payload = json.dumps(generated.model_dump(mode="json"))
    assert json.loads(payload)["strategy_id"] == generated.strategy_id


def test_required_test_suites_are_bounded_and_aggregated(tmp_path: Path) -> None:
    service, workflow = registered_service(tmp_path)
    generated = service.generate_spec(workflow)
    service.approve(generated.strategy_id, "APPROVE")
    plan = service.implementation_plan(generated.strategy_id, workflow.repository_root)
    result = service.run_required_tests(workflow.repository_root, plan.required_tests, dry_run=True, worktree_path=plan.worktree_path)
    assert result.passed and result.executed is False
    assert len(plan.required_tests) == 3
    assert plan.max_repair_attempts == 3


def test_workflow_has_one_approval_and_explicit_repair_bound() -> None:
    source = Path(".smithers/workflows/trading-research-phase-b.tsx").read_text(encoding="utf-8")
    assert source.count('id="approve-spec"') == 1
    assert "maxIterations={plan.max_repair_attempts}" in source
    assert "baseline backtesting" not in source.lower()
