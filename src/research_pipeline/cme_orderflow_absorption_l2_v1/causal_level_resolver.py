"""Artifact-backed, causal structural-level resolution for multi-session L2 research.

The resolver deliberately has no market-data, DBN, or profile-construction
path.  A causal-tape build supplies an explicit session relationship map and
level observations.  This keeps calendar policy in the canonical builder and
lets the research workflow reject missing or future information rather than
guessing a prior date or using a completed-session value too early.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


CATALOG_SCHEMA_VERSION = 1
PRIOR_PROFILE = "PRIOR_COMPLETED_SOURCE_SESSION_PROFILE"
CURRENT_EXTREMUM = "CURRENT_SOURCE_EXTREMUM"
COMPLETED_CURRENT_PROFILE = "COMPLETED_CURRENT_DATE_SOURCE_PROFILE"
SUPPORTED_SEMANTICS = frozenset({PRIOR_PROFILE, CURRENT_EXTREMUM, COMPLETED_CURRENT_PROFILE})
PROFILE_MODE = "COMPLETED_PROFILE"
DYNAMIC_MODE = "DYNAMIC_EXTREMUM"
LEVEL_FAMILIES = frozenset({"HIGH", "LOW", "POC", "VAH", "VAL"})


class StrategyLike(Protocol):
    strategy_id: str
    session: str
    source_session: str
    reference_level: str
    reference_semantics: str


class CausalLevelResolutionError(RuntimeError):
    """A level catalog cannot be used without breaking its causal contract."""


@dataclass(frozen=True)
class ResolutionResult:
    available: bool
    reason: str | None
    strategy_id: str
    trading_date: str
    target_session: str
    source_session: str
    source_date: str | None = None
    level_family: str | None = None
    level_value: float | None = None
    source_artifact: str | None = None
    source_artifact_sha256: str | None = None
    causal_availability_timestamp_ns: int | None = None
    prior_current_semantics: str | None = None
    observation_mode: str | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trading_date": self.trading_date,
            "target_session": self.target_session,
            "source_session": self.source_session,
            "source_date": self.source_date,
            "level_type": self.level_family,
            "level_value": self.level_value,
            "source_artifact": self.source_artifact,
            "source_artifact_sha256": self.source_artifact_sha256,
            "causal_availability_timestamp_ns": self.causal_availability_timestamp_ns,
            "prior_current_semantics": self.prior_current_semantics,
            "observation_mode": self.observation_mode,
            "available": self.available,
            "unavailable_reason": self.reason,
        }

    def structural_level(self, reference_level: str):
        """Return a validated shared-engine level only when it is causal."""
        if not self.available or self.level_value is None:
            raise CausalLevelResolutionError(f"cannot create structural level from unavailable resolution: {self.reason}")
        from .model import StructuralLevel
        return StructuralLevel(reference_level, self.level_value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def level_family(reference_level: str) -> str:
    """Derive the five canonical structural families from a manifest level name."""
    normalized = str(reference_level)
    if normalized.endswith("_HIGH_SWEEP") or normalized.endswith("_HIGH"):
        return "HIGH"
    if normalized.endswith("_LOW_SWEEP") or normalized.endswith("_LOW"):
        return "LOW"
    for family in ("POC", "VAH", "VAL"):
        if normalized.endswith(f"_{family}"):
            return family
    raise CausalLevelResolutionError(f"unsupported structural level family: {reference_level}")


def expected_observation_mode(reference_semantics: str) -> str:
    if reference_semantics == CURRENT_EXTREMUM:
        return DYNAMIC_MODE
    if reference_semantics in {PRIOR_PROFILE, COMPLETED_CURRENT_PROFILE}:
        return PROFILE_MODE
    raise CausalLevelResolutionError(f"unsupported reference semantics: {reference_semantics}")


def supports_strategy(strategy: StrategyLike) -> tuple[bool, str | None]:
    try:
        if strategy.reference_semantics not in SUPPORTED_SEMANTICS:
            return False, f"UNSUPPORTED_REFERENCE_SEMANTICS:{strategy.reference_semantics}"
        if level_family(strategy.reference_level) not in LEVEL_FAMILIES:
            return False, f"UNSUPPORTED_LEVEL_FAMILY:{strategy.reference_level}"
        expected_observation_mode(strategy.reference_semantics)
    except CausalLevelResolutionError as exc:
        return False, str(exc)
    return True, None


class CausalLevelResolver:
    """Resolve a manifest strategy at one target interaction timestamp.

    Catalog rows are intentionally generic, shared inputs:

    ``session_relationships`` maps a target session/date and source session to
    the exact source session date for one semantic relationship.  It replaces
    date-minus-one logic and therefore handles weekends and sparse research
    periods. ``level_observations`` contains either completed-profile values or
    causally timestamped dynamic extrema. No strategy-specific file is needed.
    """

    def __init__(self, payload: Mapping[str, Any], *, catalog_path: Path) -> None:
        if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise CausalLevelResolutionError("level catalog requires schema_version=1")
        relationships = payload.get("session_relationships")
        observations = payload.get("level_observations")
        if not isinstance(relationships, list) or not isinstance(observations, list):
            raise CausalLevelResolutionError("level catalog requires session_relationships and level_observations lists")
        self.catalog_path = catalog_path
        self.catalog_sha256 = _sha256(catalog_path)
        self.relationships = tuple(self._relationship(row) for row in relationships)
        self.observations = tuple(self._observation(row) for row in observations)
        self._validate_dynamic_extrema()

    @classmethod
    def from_path(cls, path: Path) -> "CausalLevelResolver":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CausalLevelResolutionError(f"invalid level catalog: {path}") from exc
        if not isinstance(payload, Mapping):
            raise CausalLevelResolutionError("level catalog root must be an object")
        return cls(payload, catalog_path=path)

    @staticmethod
    def _relationship(row: object) -> dict[str, str]:
        if not isinstance(row, Mapping):
            raise CausalLevelResolutionError("session relationship must be an object")
        required = ("target_session", "trading_date", "source_session", "source_date", "reference_semantics")
        if any(not row.get(key) for key in required):
            raise CausalLevelResolutionError("session relationship has missing fields")
        semantic = str(row["reference_semantics"])
        if semantic not in SUPPORTED_SEMANTICS:
            raise CausalLevelResolutionError(f"unsupported relationship semantics: {semantic}")
        return {key: str(row[key]) for key in required}

    @staticmethod
    def _observation(row: object) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise CausalLevelResolutionError("level observation must be an object")
        required = ("source_session", "source_date", "level_family", "level_value", "available_at_ns", "mode", "source_artifact")
        if any(key not in row or row[key] in (None, "") for key in required):
            raise CausalLevelResolutionError("level observation has missing fields")
        family, mode = str(row["level_family"]), str(row["mode"])
        if family not in LEVEL_FAMILIES or mode not in {PROFILE_MODE, DYNAMIC_MODE}:
            raise CausalLevelResolutionError("level observation has unsupported family or mode")
        try:
            value, available_at = float(row["level_value"]), int(row["available_at_ns"])
        except (TypeError, ValueError) as exc:
            raise CausalLevelResolutionError("level observation has invalid value or availability timestamp") from exc
        if not math.isfinite(value) or value <= 0.0 or available_at < 0:
            raise CausalLevelResolutionError("level observation has non-causal value or timestamp")
        return {
            "source_session": str(row["source_session"]), "source_date": str(row["source_date"]),
            "level_family": family, "level_value": value, "available_at_ns": available_at, "mode": mode,
            "source_artifact": str(row["source_artifact"]),
            "source_artifact_sha256": None if row.get("source_artifact_sha256") in (None, "") else str(row["source_artifact_sha256"]),
        }

    def _source_date(self, strategy: StrategyLike, trading_date: str) -> str | None:
        matches = [row for row in self.relationships if row["target_session"] == strategy.session
                   and row["trading_date"] == trading_date and row["source_session"] == strategy.source_session
                   and row["reference_semantics"] == strategy.reference_semantics]
        if len(matches) > 1:
            raise CausalLevelResolutionError(f"ambiguous session mapping: {strategy.strategy_id}/{trading_date}")
        return matches[0]["source_date"] if matches else None

    def _validate_dynamic_extrema(self) -> None:
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        for row in self.observations:
            if row["mode"] == DYNAMIC_MODE:
                grouped.setdefault((str(row["source_session"]), str(row["source_date"]), str(row["level_family"])), []).append(row)
        for (_session, _date, family), rows in grouped.items():
            ordered = sorted(rows, key=lambda row: int(row["available_at_ns"]))
            timestamps = [int(row["available_at_ns"]) for row in ordered]
            if len(timestamps) != len(set(timestamps)):
                raise CausalLevelResolutionError("dynamic level observations must have unique availability timestamps")
            values = [float(row["level_value"]) for row in ordered]
            if family == "HIGH" and any(later < earlier for earlier, later in zip(values, values[1:])):
                raise CausalLevelResolutionError("dynamic high observations must be non-decreasing")
            if family == "LOW" and any(later > earlier for earlier, later in zip(values, values[1:])):
                raise CausalLevelResolutionError("dynamic low observations must be non-increasing")

    def resolve(self, strategy: StrategyLike, *, trading_date: str, signal_timestamp_ns: int) -> ResolutionResult:
        supported, reason = supports_strategy(strategy)
        if not supported:
            return ResolutionResult(False, reason, strategy.strategy_id, trading_date, strategy.session, strategy.source_session)
        source_date = self._source_date(strategy, trading_date)
        if source_date is None:
            return ResolutionResult(False, "SOURCE_SESSION_MAPPING_MISSING", strategy.strategy_id, trading_date, strategy.session, strategy.source_session,
                                    level_family=level_family(strategy.reference_level), prior_current_semantics=strategy.reference_semantics)
        family, mode = level_family(strategy.reference_level), expected_observation_mode(strategy.reference_semantics)
        available = [row for row in self.observations if row["source_session"] == strategy.source_session
                     and row["source_date"] == source_date and row["level_family"] == family and row["mode"] == mode
                     and int(row["available_at_ns"]) <= int(signal_timestamp_ns)]
        if not available:
            return ResolutionResult(False, "LEVEL_UNAVAILABLE_AT_SIGNAL_TIMESTAMP", strategy.strategy_id, trading_date, strategy.session,
                                    strategy.source_session, source_date, family, prior_current_semantics=strategy.reference_semantics,
                                    observation_mode=mode)
        chosen = max(available, key=lambda row: int(row["available_at_ns"]))
        return ResolutionResult(True, None, strategy.strategy_id, trading_date, strategy.session, strategy.source_session,
                                source_date, family, float(chosen["level_value"]), str(chosen["source_artifact"]),
                                chosen["source_artifact_sha256"], int(chosen["available_at_ns"]),
                                strategy.reference_semantics, mode)


def candidate_executability_audit(strategies: Sequence[StrategyLike]) -> list[dict[str, Any]]:
    """Offline capability audit; it does not assert that historical data exists."""
    rows: list[dict[str, Any]] = []
    for strategy in sorted(strategies, key=lambda item: item.strategy_id):
        supported, reason = supports_strategy(strategy)
        rows.append({"strategy_id": strategy.strategy_id, "target_session": strategy.session,
                     "source_session": strategy.source_session, "reference_level": strategy.reference_level,
                     "reference_semantics": strategy.reference_semantics,
                     "classification": "EXECUTABLE_WITH_SUPPORTED_INPUTS" if supported else "NOT_EXECUTABLE",
                     "missing_capability": reason})
    return rows
