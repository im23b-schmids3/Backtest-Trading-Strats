from __future__ import annotations

import json
from pathlib import Path

from ..registry.repositories import Registry
from .calendar import CalendarArtifact
from .costs import ExecutionCostConfig
from .models import ComplianceDecision, PropFirmPolicy


class ComplianceStore:
    """Small persistence facade for reproducible policy evidence."""

    def __init__(self, registry: Registry):
        self.registry = registry

    def save_policy(self, strategy_id: str, version: str, policy: PropFirmPolicy) -> None:
        self.registry.save_compliance_policy(strategy_id, version, policy.model_dump(mode="json"))

    def save_calendar(self, strategy_id: str, version: str, artifact: CalendarArtifact, path: str | Path | None = None) -> None:
        self.registry.save_compliance_artifact(strategy_id, version, f"calendar:{artifact.artifact_hash}", "ECONOMIC_CALENDAR", artifact.artifact_hash, artifact.model_dump(mode="json"), str(path) if path else None)

    def save_cost_config(self, strategy_id: str, version: str, config: ExecutionCostConfig) -> None:
        self.registry.save_compliance_artifact(strategy_id, version, f"execution-costs:{config.configuration_hash}", "EXECUTION_COST_CONFIGURATION", config.configuration_hash, config.model_dump(mode="json"))

    def save_decision(self, strategy_id: str, version: str, decision: ComplianceDecision) -> int:
        return self.registry.save_compliance_decision(strategy_id, version, decision.model_dump(mode="json"))

    def save_event(self, strategy_id: str, version: str, event_type: str, timestamp: str, result: dict) -> int:
        return self.registry.add_compliance_event(strategy_id, version, event_type, timestamp, result)
