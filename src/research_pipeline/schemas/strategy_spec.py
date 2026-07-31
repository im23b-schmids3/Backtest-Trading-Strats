from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..enums import ApprovalStatus

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ParameterFamily(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    description: str
    baseline_value: Any
    value_type: str = Field(min_length=1)
    allowed_min: float | int | None = None
    allowed_max: float | int | None = None
    allowed_values: list[Any] | None = None
    optimization_order: int = Field(ge=0)
    maximum_rounds: int = Field(ge=0)
    mutable: bool
    hypothesis_relevance: str

    @model_validator(mode="after")
    def validate_bounds(self) -> "ParameterFamily":
        if self.allowed_min is not None and self.allowed_max is not None and self.allowed_min > self.allowed_max:
            raise ValueError("allowed_min cannot exceed allowed_max")
        if self.allowed_values is not None and not self.allowed_values:
            raise ValueError("allowed_values cannot be empty when provided")
        if not self.mutable and self.maximum_rounds != 0:
            raise ValueError("immutable parameter families must have maximum_rounds=0")
        return self


class StrategySpec(StrictModel):
    strategy_id: str
    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    hypothesis: str
    strategy_family: str = Field(min_length=1)
    markets: list[str] = Field(min_length=1)
    timeframes: list[str] = Field(min_length=1)
    long_rules: list[str]
    short_rules: list[str]
    entry_logic: str
    initial_stop_logic: str
    exit_logic: str
    session_assumptions: list[str]
    baseline_parameters: dict[str, Any]
    parameter_families: list[ParameterFamily]
    invariants: list[str]
    required_data: list[str]
    known_limitations: list[str]
    status: ApprovalStatus = ApprovalStatus.DRAFT
    created_at: datetime
    approved_at: datetime | None = None
    specification_hash: str

    @field_validator("strategy_id")
    @classmethod
    def safe_strategy_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError(
                "strategy_id must be filesystem-safe "
                "(letters, numbers, _, -, . only)"
            )
        return value

    @model_validator(mode="after")
    def validate_specification(
        self,
        info: ValidationInfo,
    ) -> "StrategySpec":
        names = [family.name for family in self.parameter_families]
        if len(names) != len(set(names)):
            raise ValueError("parameter family names must be unique")

        if self.status == ApprovalStatus.APPROVED and self.approved_at is None:
            raise ValueError("approved specifications require approved_at")

        if self.status != ApprovalStatus.APPROVED and self.approved_at is not None:
            raise ValueError("only approved specifications may have approved_at")

        if (info.context or {}).get("skip_specification_hash_validation"):
            return self

        expected = calculate_specification_hash(self)
        if self.specification_hash != expected:
            raise ValueError(
                "specification_hash does not match the canonical specification"
            )

        return self

    def canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("specification_hash", None)
        # Approval is an audit event, not a material strategy change.
        payload.pop("approved_at", None)
        payload.pop("status", None)
        return payload

    def approved_copy(self, when: datetime) -> "StrategySpec":
        return self.model_copy(
            update={
                "status": ApprovalStatus.APPROVED,
                "approved_at": when,
            }
        )


def calculate_specification_hash(
    specification: StrategySpec | dict[str, Any],
) -> str:
    if isinstance(specification, StrategySpec):
        normalized = specification
    else:
        data = dict(specification)
        data["specification_hash"] = str(data.get("specification_hash") or "pending")
        normalized = StrategySpec.model_validate(
            data,
            context={"skip_specification_hash_validation": True},
        )

    payload = normalized.model_dump(mode="json")

    payload.pop("specification_hash", None)
    payload.pop("approved_at", None)
    payload.pop("status", None)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()

    return hashlib.sha256(encoded).hexdigest()


def load_strategy_spec(path: str) -> StrategySpec:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("strategy YAML must contain a mapping")

    return StrategySpec.model_validate(raw)


def save_strategy_spec(
    specification: StrategySpec,
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            specification.model_dump(mode="json"),
            handle,
            sort_keys=False,
        )
