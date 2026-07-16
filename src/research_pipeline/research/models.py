from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from ..enums import PipelineState
from ..schemas.strategy_spec import StrictModel


class ResearchClassification(StrEnum):
    ACCEPTED_STANDALONE = "ACCEPTED_STANDALONE"
    ACCEPTED_PORTFOLIO_COMPONENT = "ACCEPTED_PORTFOLIO_COMPONENT"
    REJECTED_NO_EDGE = "REJECTED_NO_EDGE"
    REJECTED_UNSTABLE = "REJECTED_UNSTABLE"
    REJECTED_EXECUTION_SENSITIVE = "REJECTED_EXECUTION_SENSITIVE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class MetricCitation(StrictModel):
    metric_name: str = Field(min_length=1)
    value: float
    source_file: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)


class AnalystDecision(StrictModel):
    strategy_id: str
    strategy_version: str
    current_phase: PipelineState
    decision: str
    confidence: float = Field(ge=0, le=1)
    evidence_strength: str
    primary_bottleneck: str
    selected_parameter_family: str | None = None
    current_value: Any = None
    proposed_values: list[Any] = Field(default_factory=list)
    proposal_method: str
    parameter_hypothesis: str
    expected_behavior: str
    files_inspected: list[str]
    metrics_cited: list[MetricCitation]
    risks: list[str]
    overfitting_risk: float = Field(ge=0, le=1)
    stop_reason: str | None = None
    next_phase: PipelineState | None = None
    rationale: str = Field(min_length=1)


class StatisticalReview(StrictModel):
    strategy_id: str
    strategy_version: str
    round_id: str
    decision: str
    stable_region: list[Any] = Field(default_factory=list)
    selected_value: Any = None
    isolated_maximum_risk: bool = False
    evidence_strength: str
    metrics_cited: list[MetricCitation] = Field(default_factory=list)
    veto_reason: str | None = None
    rationale: str = Field(min_length=1)


class ResearchArtifact(StrictModel):
    experiment_id: str
    strategy_id: str
    strategy_version: str
    phase: str
    experiment_dir: str
    input_path: str
    metrics_path: str
    diagnostic_manifest_path: str | None = None
    report_hashes: dict[str, str] = Field(default_factory=dict)
    dataset_hash: str
    split_hash: str
    code_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    diagnostic_manifest: dict[str, Any] = Field(default_factory=dict)


class BaselineResult(StrictModel):
    artifact: ResearchArtifact
    verification_outcome: str
    gate_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    edge_decision: str | None = None


class ParameterProposal(StrictModel):
    strategy_id: str
    strategy_version: str
    family: str
    current_value: Any
    proposed_values: list[Any] = Field(min_length=1)
    round_number: int = Field(ge=1)
    hypothesis: str
    reason: str


class ParameterExperiment(StrictModel):
    value: Any
    artifact: ResearchArtifact
    robustness_score: float
    score_components: dict[str, float] = Field(default_factory=dict)


class ParameterRoundResult(StrictModel):
    round_id: str
    family: str
    experiments: list[ParameterExperiment]
    review: StatisticalReview
    selected_value: Any = None
    stable_region: list[Any] = Field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None


class CandidateManifest(StrictModel):
    strategy_id: str
    strategy_version: str
    approved_specification_hash: str
    split_hash: str
    code_commit: str | None = None
    selected_parameters: dict[str, Any]
    frozen_families: list[str]
    research_decisions: list[str]
    total_selection_count: int = Field(ge=0)
    budget_usage: dict[str, Any]
    candidate_hash: str
    manifest_path: str


class WalkForwardResult(StrictModel):
    status: str
    folds: list[dict[str, Any]] = Field(min_length=1)
    aggregate_metrics: dict[str, Any]
    verification_outcome: str
    reason: str
    gate_outcomes: list[dict[str, Any]] = Field(default_factory=list)


class HoldoutResult(StrictModel):
    status: str
    untouched: bool = True
    access_count: int
    dataset_hash: str
    metrics: dict[str, Any]
    verification_outcome: str
    reason: str
    gate_outcomes: list[dict[str, Any]] = Field(default_factory=list)


class StressResult(StrictModel):
    classification: str
    scenarios: list[dict[str, Any]] = Field(min_length=1)
    profitable_scenario_ratio: float = Field(ge=0, le=1)
    expectancy_range: list[float] = Field(min_length=2, max_length=2)
    worst_drawdown: float
    break_even_fee_level: float | None = None
    reason: str


class ThroughputResult(StrictModel):
    classification: str
    candidate_setups: int
    unique_setups: int
    filled_positions: int
    completed_positions: int
    trades_per_month: float
    median_days_between_trades: float | None
    longest_no_trade_period_days: float | None
    zero_trade_month_percentage: float
    trades_by_market: dict[str, int]
    trades_by_timeframe: dict[str, int]
    accumulation_days: dict[str, float | None]
    reason: str
    gate_outcomes: list[dict[str, Any]] = Field(default_factory=list)


class FinalResearchReview(StrictModel):
    strategy_id: str
    strategy_version: str
    classification: ResearchClassification
    current_phase: PipelineState
    evidence_strength: str
    evidence: dict[str, Any]
    metrics_cited: list[MetricCitation]
    risks: list[str]
    rationale: str
    next_phase: PipelineState | None = None
