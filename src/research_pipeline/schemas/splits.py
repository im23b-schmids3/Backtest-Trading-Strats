from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import Field, model_validator

from .strategy_spec import StrictModel


class SplitWindow(StrictModel):
    start_timestamp: datetime
    end_timestamp: datetime

    @model_validator(mode="after")
    def valid_window(self) -> "SplitWindow":
        if self.start_timestamp >= self.end_timestamp:
            raise ValueError("split window start must precede end")
        return self


class SplitDefinition(StrictModel):
    dataset_identifier: str = Field(min_length=1)
    source_data_hash: str = Field(min_length=1)
    start_timestamp: datetime
    end_timestamp: datetime
    training_boundaries: SplitWindow
    validation_boundaries: SplitWindow
    holdout_boundaries: SplitWindow
    created_timestamp: datetime
    split_hash: str

    @model_validator(mode="after")
    def chronological(self) -> "SplitDefinition":
        windows = [self.training_boundaries, self.validation_boundaries, self.holdout_boundaries]
        if self.start_timestamp > windows[0].start_timestamp or self.end_timestamp < windows[-1].end_timestamp:
            raise ValueError("split windows must fit within the dataset bounds")
        for previous, current in zip(windows, windows[1:]):
            if previous.end_timestamp > current.start_timestamp:
                raise ValueError("training, validation and holdout windows must be chronological")
        if self.split_hash != calculate_split_hash(self):
            raise ValueError("split_hash does not match split boundaries and source hash")
        return self

    def deterministic_payload(self) -> dict:
        payload = self.model_dump(mode="json")
        payload.pop("split_hash", None)
        payload.pop("created_timestamp", None)
        return payload


def calculate_split_hash(split: SplitDefinition | dict) -> str:
    payload = split.model_dump(mode="json") if isinstance(split, SplitDefinition) else dict(split)
    payload.pop("split_hash", None)
    payload.pop("created_timestamp", None)
    if not isinstance(split, SplitDefinition):
        payload = _normalize_timestamps(payload)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalize_timestamps(value):
    if isinstance(value, dict):
        return {key: _normalize_timestamps(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_timestamps(item) for item in value]
    if isinstance(value, str) and "T" in value:
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.isoformat().replace("+00:00", "Z") if parsed.tzinfo else parsed.isoformat()
        except ValueError:
            pass
    return value


def load_split_definition(path: str) -> SplitDefinition:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return SplitDefinition.model_validate(yaml.safe_load(handle))
