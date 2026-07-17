from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel


class MasterRunStatus(StrEnum):
    SUCCESS = "SUCCESS"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
    RESEARCH_FAILURE = "RESEARCH_FAILURE"
    PROP_FAILURE = "PROP_FAILURE"
    PORTFOLIO_FAILURE = "PORTFOLIO_FAILURE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


MasterRunOutcome = MasterRunStatus


class MasterStep(StrEnum):
    INTAKE = "INTAKE"
    SPECIFICATION = "SPECIFICATION"
    APPROVAL = "APPROVAL"
    IMPLEMENTATION = "IMPLEMENTATION"
    IMPLEMENTATION_VERIFICATION = "IMPLEMENTATION_VERIFICATION"
    TECHNICAL_VERIFICATION = "TECHNICAL_VERIFICATION"
    BASELINE = "BASELINE"
    RESEARCH = "RESEARCH"
    WALK_FORWARD = "WALK_FORWARD"
    HOLDOUT = "HOLDOUT"
    STRESS = "STRESS"
    PROP = "PROP"
    PORTFOLIO = "PORTFOLIO"
    FINAL_REPORT = "FINAL_REPORT"
    ARCHIVE = "ARCHIVE"
    COMPLETED = "COMPLETED"


class FinalClassification(StrEnum):
    ACCEPTED_STANDALONE = "ACCEPTED_STANDALONE"
    ACCEPTED_PORTFOLIO_COMPONENT = "ACCEPTED_PORTFOLIO_COMPONENT"
    OWN_CAPITAL_ONLY = "OWN_CAPITAL_ONLY"
    PORTFOLIO_ACCEPTED = "PORTFOLIO_ACCEPTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class IntakeSpec(StrictModel):
    strategy_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=10)
    markets: list[str] = Field(min_length=1)
    timeframes: list[str] = Field(min_length=1)
    entry_logic: list[str] = Field(default_factory=list)
    exit_logic: list[str] = Field(default_factory=list)
    risk_model: str | None = None
    position_sizing: str | None = None
    filters: list[str] = Field(default_factory=list)
    optional_notes: str | None = None
    unknown_fields: dict[str, Any] = Field(default_factory=dict)
    confidence_flags: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_evidence(self) -> "IntakeSpec":
        if not self.confirmed_facts:
            self.confirmed_facts = [f"markets={','.join(self.markets)}", f"timeframes={','.join(self.timeframes)}"]
        return self


class MasterRunInput(StrictModel):
    intake_path: str
    repository_root: str
    registry_path: str | None = None
    dry_run: bool = True
    implementation_enabled: bool = False
    research_scenario: str = "strong-stable"
    prop_scenario: str = "profitable"
    portfolio_scenario: str = "complementary"
    prop_product: str = "Alpha Futures Zero 25K"


class ArtifactReference(StrictModel):
    phase: str
    path: str
    artifact_type: str
    sha256: str


class PhaseTiming(StrictModel):
    phase: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int = Field(ge=0)
    result_hash: str
    artifact_paths: list[str] = Field(default_factory=list)


class ApprovalRecord(StrictModel):
    decision: str
    approved: bool
    note: str | None = None
    decided_at: datetime | None = None


class FinalReport(StrictModel):
    run_id: str
    strategy_id: str
    strategy_version: str
    classification: FinalClassification
    specification: dict[str, Any]
    implementation_summary: dict[str, Any]
    verification_summary: dict[str, Any]
    research_summary: dict[str, Any]
    prop_summary: dict[str, Any]
    portfolio_summary: dict[str, Any]
    final_recommendation: str
    known_limitations: list[str]
    confidence: str
    artifacts: list[ArtifactReference]
    hashes: dict[str, str]
    phase_timings: list[PhaseTiming]
    generated_at: datetime


class MasterStatus(StrictModel):
    run_id: str
    strategy_id: str
    strategy_version: str | None
    current_step: MasterStep
    outcome: MasterRunStatus
    approval_status: str
    root_path: str
    phase_results: list[dict[str, Any]]
    journal_entries: int
    artifacts: list[ArtifactReference]
    report: dict[str, Any] | None = None
