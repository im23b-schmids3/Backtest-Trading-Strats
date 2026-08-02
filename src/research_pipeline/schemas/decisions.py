from __future__ import annotations

from typing import Any

from pydantic import Field

from ..enums import DecisionType, PipelineState
from .strategy_spec import StrictModel


class DecisionRecord(StrictModel):
    phase: PipelineState
    strategy_id: str
    decision: DecisionType
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    blocking_issues: list[str]
    selected_parameter_family: str | None = None
    proposed_values: dict[str, Any] = Field(default_factory=dict)
    rejected_values: list[Any] = Field(default_factory=list)
    overfitting_risk: float = Field(ge=0, le=1)
    next_phase: PipelineState | None = None
    files_inspected: list[str]
    metrics_cited: list[str]
    rationale: str = Field(min_length=1)

