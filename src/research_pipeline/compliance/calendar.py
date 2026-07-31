from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import Field, field_validator

from ..schemas.strategy_spec import StrictModel


class EconomicEvent(StrictModel):
    event_id: str
    title: str
    timestamp: datetime
    event_timezone: str = "UTC"
    impact_level: str = "UNKNOWN"
    affected_currencies: list[str] = Field(default_factory=list)
    affected_instruments: list[str] = Field(default_factory=list)
    source: str
    retrieved_at: datetime
    source_data_hash: str

    @field_validator("timestamp", "retrieved_at")
    @classmethod
    def aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("economic-event datetimes must be timezone-aware")
        return value


class EconomicCalendarProvider(Protocol):
    def events(self, start: datetime, end: datetime) -> list[EconomicEvent]: ...


class CalendarArtifact(StrictModel):
    start: datetime
    end: datetime
    events: list[EconomicEvent] = Field(default_factory=list)
    retrieved_at: datetime
    source: str
    source_data_hash: str
    artifact_hash: str = "pending"

    @field_validator("start", "end", "retrieved_at")
    @classmethod
    def aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calendar artifact datetimes must be timezone-aware")
        return value


class FixtureEconomicCalendarProvider:
    """Deterministic provider for tests and offline research."""

    def __init__(self, events: list[EconomicEvent] | None = None, *, unavailable: bool = False):
        self._events = list(events or [])
        self.unavailable = unavailable

    def events(self, start: datetime, end: datetime) -> list[EconomicEvent]:
        if self.unavailable:
            raise RuntimeError("fixture calendar unavailable")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("calendar query datetimes must be timezone-aware")
        return [event for event in self._events if start <= event.timestamp <= end]


def calendar_data_hash(events: list[EconomicEvent]) -> str:
    payload = [event.model_dump(mode="json") for event in sorted(events, key=lambda item: item.event_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _artifact_hash(artifact: CalendarArtifact) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("artifact_hash", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def save_calendar_artifact(
    start: datetime,
    end: datetime,
    events: list[EconomicEvent],
    path: str | Path,
    *,
    source: str = "fixture",
    retrieved_at: datetime | None = None,
) -> CalendarArtifact:
    artifact = CalendarArtifact(
        start=start,
        end=end,
        events=events,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        source=source,
        source_data_hash=calendar_data_hash(events),
        artifact_hash="pending",
    )
    payload = artifact.model_dump(mode="python")
    payload["artifact_hash"] = _artifact_hash(artifact)
    artifact = CalendarArtifact.model_validate(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def load_calendar_artifact(path: str | Path) -> CalendarArtifact:
    artifact = CalendarArtifact.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    if artifact.source_data_hash != calendar_data_hash(artifact.events):
        raise ValueError("calendar source-data hash mismatch")
    if artifact.artifact_hash != _artifact_hash(artifact):
        raise ValueError("calendar artifact hash mismatch")
    return artifact


from .models import MarketState  # noqa: E402  (keeps the model module independent)

MarketState.model_rebuild(_types_namespace={"EconomicEvent": EconomicEvent})
