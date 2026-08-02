from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class SpecificationJobType(StrEnum):
    GENERATE_SPECIFICATION = "GENERATE_SPECIFICATION"
    REPAIR_SPECIFICATION = "REPAIR_SPECIFICATION"


class SpecificationCompletionStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED_CODEX_EXECUTION = "FAILED_CODEX_EXECUTION"
    FAILED_OUTPUT_EXTRACTION = "FAILED_OUTPUT_EXTRACTION"
    FAILED_INPUT_INTEGRITY = "FAILED_INPUT_INTEGRITY"
    FAILED_ARTIFACT_INTEGRITY = "FAILED_ARTIFACT_INTEGRITY"
    CANCELLED = "CANCELLED"


class SpecificationJobRequest(StrictModel):
    run_id: str
    smithers_run_id: str | None = None
    job_id: str
    strategy_id: str
    strategy_version: str
    attempt: int = Field(ge=1)
    job_type: SpecificationJobType
    repository_root: str
    intake_path: str
    intake_hash: str
    prior_invalid_draft_path: str | None = None
    prior_invalid_draft_hash: str | None = None
    validation_report_path: str | None = None
    validation_report_hash: str | None = None
    semantic_validation_path: str | None = None
    semantic_validation_hash: str | None = None
    schema_contract_version: str
    allowed_output_format: str = "one YAML or JSON mapping"
    timeout_seconds: int = Field(ge=1)
    codex_config_requirements: dict[str, Any] = Field(default_factory=dict)
    expected_output_paths: list[str]
    provenance: dict[str, Any] = Field(default_factory=dict)
    input_hash_manifest: dict[str, str]
    created_at: datetime


class SpecificationJobStatus(StrictModel):
    run_id: str
    job_id: str
    attempt: int
    job_type: SpecificationJobType
    status: str
    next_command: str
    updated_at: datetime
    result_path: str | None = None
    error: str | None = None


class SpecificationCompletion(StrictModel):
    run_id: str
    smithers_run_id: str | None = None
    job_id: str
    strategy_id: str
    strategy_version: str
    attempt: int = Field(ge=1)
    job_type: SpecificationJobType
    status: SpecificationCompletionStatus
    repository_root: str
    exit_code: int | None = None
    duration_ms: int = Field(default=0, ge=0)
    raw_output_path: str | None = None
    extracted_draft_path: str | None = None
    extraction_report_path: str | None = None
    codex_invocation_path: str | None = None
    output_hash_manifest_path: str | None = None
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    input_hash_manifest: dict[str, str]
    extraction_outcome: str | None = None
    stdout_summary: str = ""
    stderr_summary: str = ""
    repository_mutations: list[str] = Field(default_factory=list)
    result_hash: str | None = None
    created_at: datetime
