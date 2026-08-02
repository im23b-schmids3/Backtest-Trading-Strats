from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ResearchBudget(StrictModel):
    max_parameter_families: int = Field(default=6, ge=0)
    max_rounds_per_family: int = Field(default=3, ge=0)
    max_values_per_round: int = Field(default=5, ge=0)
    max_total_backtests: int = Field(default=500, ge=0)
    max_codex_repair_attempts: int = Field(default=3, ge=0)
    max_runtime_minutes_per_phase: int = Field(default=180, ge=0)
    max_report_size_mb: int = Field(default=100, ge=0)
    max_holdout_accesses: int = Field(default=1, ge=0)


class BudgetUsage(StrictModel):
    parameter_families: list[str] = Field(default_factory=list)
    rounds_per_family: dict[str, int] = Field(default_factory=dict)
    values_per_round: dict[str, int] = Field(default_factory=dict)
    family_by_round: dict[str, str] = Field(default_factory=dict)
    values_by_round: dict[str, int] = Field(default_factory=dict)
    total_backtests: int = Field(default=0, ge=0)
    codex_repair_attempts: int = Field(default=0, ge=0)
    runtime_minutes_by_phase: dict[str, int] = Field(default_factory=dict)
    report_size_mb: float = Field(default=0, ge=0)
    holdout_accesses: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def normalize_families(self) -> "BudgetUsage":
        object.__setattr__(self, "parameter_families", sorted(set(self.parameter_families)))
        return self
