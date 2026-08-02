from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class CodexCompletionStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED_CODEX_EXECUTION = "FAILED_CODEX_EXECUTION"
    FAILED_WORKTREE_PREFLIGHT = "FAILED_WORKTREE_PREFLIGHT"
    FAILED_WORKTREE_CREATION = "FAILED_WORKTREE_CREATION"
    FAILED_SCOPE_VALIDATION = "FAILED_SCOPE_VALIDATION"
    FAILED_REQUIRED_TESTS = "FAILED_REQUIRED_TESTS"
    FAILED_ARTIFACT_INTEGRITY = "FAILED_ARTIFACT_INTEGRITY"
    CANCELLED = "CANCELLED"


class ImplementationJobRequest(StrictModel):
    job_id: str
    run_id: str
    strategy_id: str
    strategy_version: str
    specification_hash: str
    repository_root: str
    approved_worktree_root: str | None = None
    base_commit: str
    branch: str
    approved_implementation_scope: str
    allowed_paths: list[str]
    forbidden_paths: list[str]
    required_tests: list[list[str]]
    expected_output_contract: str
    timeout_seconds: int = Field(ge=1)
    max_repair_attempts: int = Field(default=0, ge=0)
    codex_model: str | None = None
    codex_config_requirements: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    input_hash_manifest: dict[str, str]
    created_at: datetime


class ImplementationJobStatus(StrictModel):
    job_id: str
    run_id: str
    status: str
    external_executor_required: bool = True
    next_command: str
    updated_at: datetime
    error: str | None = None


class ImplementationCompletion(StrictModel):
    job_id: str
    run_id: str
    strategy_id: str
    strategy_version: str
    specification_hash: str
    repository_root: str
    worktree_path: str | None = None
    base_commit: str | None = None
    resulting_commit: str | None = None
    status: CodexCompletionStatus
    exit_code: int | None = None
    stdout_summary: str = ""
    stderr_summary: str = ""
    duration_ms: int = Field(default=0, ge=0)
    changed_files: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    input_hash_manifest: dict[str, str]
    result_hash: str | None = None
    created_at: datetime
