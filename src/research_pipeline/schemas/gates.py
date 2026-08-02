from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from .strategy_spec import StrictModel
from ..enums import GateOutcomeStatus


class Comparison(StrEnum):
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    GREATER = ">"
    LESS = "<"
    EQUAL = "=="


class GateDefinition(StrictModel):
    name: str
    category: str
    metric: str
    threshold: float
    comparison: Comparison
    source_file: str = ""
    required: bool = True


class GateOutcome(StrictModel):
    gate: str
    status: GateOutcomeStatus
    metric: str
    observed_value: Any = None
    threshold: float
    comparison: Comparison
    source_file: str
    reason: str


class GateSet(StrictModel):
    gates: list[GateDefinition] = Field(default_factory=list)

