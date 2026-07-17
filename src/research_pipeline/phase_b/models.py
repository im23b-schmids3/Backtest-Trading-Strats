from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from ..enums import PipelineState
from ..schemas.strategy_spec import StrictModel
from ..validation.specification_semantics import SpecificationProvenance, SpecificationValidationIssue, SpecificationValidationReport


class WorkflowInput(StrictModel):
    strategy_name: str = Field(min_length=1, max_length=128)
    natural_language_description: str = Field(min_length=10)
    requested_markets: list[str] = Field(min_length=1)
    requested_timeframes: list[str] = Field(min_length=1)
    optional_notes: str | None = None
    repository_root: str
    registry_path: str | None = None
    dry_run: bool = True
    implementation_enabled: bool = False
    confirmed_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    run_id: str | None = None
    max_generation_attempts: int = Field(default=3, ge=1, le=3)
    max_repair_attempts: int = Field(default=2, ge=0, le=2)

    @field_validator("strategy_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if any(character in value for character in ("/", "\\", "\x00")):
            raise ValueError("strategy_name may not contain path separators or NUL")
        return value

    @field_validator("repository_root")
    @classmethod
    def valid_root(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("repository_root is required")
        return str(Path(value).expanduser())


class GeneratedStrategySpec(StrictModel):
    strategy_id: str
    version: str
    specification_path: str
    specification_hash: str
    assumptions: list[str]
    ambiguities: list[str]
    fields_requiring_confirmation: list[str]
    manual_review_required: bool
    approval_summary: str
    provenance: SpecificationProvenance = Field(default_factory=SpecificationProvenance)
    validation_report_path: str | None = None
    semantic_validation_report_path: str | None = None
    attempt: int = Field(default=1, ge=1)


class SpecificationValidationResult(StrictModel):
    valid: bool
    strategy_id: str
    version: str
    specification_path: str
    specification_hash: str
    errors: list[str]
    manual_review_required: bool = False
    structured_errors: list[SpecificationValidationIssue] = Field(default_factory=list)
    semantic_report: SpecificationValidationReport | None = None
    canonical_path: str | None = None
    approval_ready: bool = False
    provenance: SpecificationProvenance = Field(default_factory=SpecificationProvenance)


class SpecificationGenerationFailure(StrictModel):
    classification: str = "SPECIFICATION_GENERATION_FAILURE"
    strategy_id: str
    run_id: str
    attempts: int
    repair_attempts: int
    final_reason: str
    validation_report_paths: list[str] = Field(default_factory=list)
    draft_paths: list[str] = Field(default_factory=list)
    repair_prompt_paths: list[str] = Field(default_factory=list)
    codex_invocation_paths: list[str] = Field(default_factory=list)


class RegistrationResult(StrictModel):
    registered: bool
    idempotent_reuse: bool
    strategy_id: str
    version: str
    current_phase: PipelineState
    specification_hash: str


class ApprovalResult(StrictModel):
    decision: str
    approved: bool
    note: str | None = None
    strategy_id: str
    version: str
    current_phase: PipelineState
    immutable_verified: bool


class ImplementationPlan(StrictModel):
    strategy_id: str
    version: str
    base_commit: str
    branch: str
    worktree_path: str
    allowed_files: list[str]
    required_tests: list[list[str]]
    invariants: list[str]
    prohibited_actions: list[str]
    max_repair_attempts: int


class CodexExecutionResult(StrictModel):
    success: bool
    executed: bool
    command: list[str]
    cwd: str
    sandbox: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    error_type: str | None = None
    session_id: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    resulting_commit: str | None = None


class TestResult(StrictModel):
    passed: bool
    command: list[str]
    exit_code: int | None
    parsed_passed: int
    parsed_failed: int
    parsed_skipped: int
    duration_ms: int
    report_path: str | None
    failure_summary: str
    executed: bool


class RepairResult(StrictModel):
    attempt: int
    budget_remaining: int
    codex_result: CodexExecutionResult
    test_result: TestResult | None
    material_change_detected: bool
    stopped: bool
    reason: str


class RepairRequest(StrictModel):
    strategy_id: str
    repository_root: str
    worktree_path: str
    prompt: str
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=0)
    dry_run: bool = True


class FinalPhaseBSummary(StrictModel):
    strategy_id: str
    version: str
    final_state: PipelineState
    approval: str
    manual_review_required: bool
    implementation_executed: bool
    tests_passed: bool
    repair_attempts: int
    registry_reconciled: bool
    worktree_path: str | None
    outputs: list[str]
    limitation: str


class BridgeRequest(StrictModel):
    input: dict[str, Any] = Field(default_factory=dict)
