from typing import Any

from pydantic import Field

from .strategy_spec import StrictModel


class ExperimentRecord(StrictModel):
    experiment_id: str
    strategy_id: str
    phase: str
    parameter_family: str | None = None
    parameter_values: dict[str, Any] = Field(default_factory=dict)
    dataset_hash: str | None = None
    code_commit: str | None = None
    start_time: str
    end_time: str | None = None
    status: str
    report_paths: list[str] = Field(default_factory=list)
