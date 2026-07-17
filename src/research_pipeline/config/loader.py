from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .defaults import DEFAULT_BUDGETS, DEFAULT_GATES
from ..schemas.budgets import ResearchBudget
from ..schemas.gates import GateSet


def load_pipeline_config(path: str | Path | None = None, strategy_id: str | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    strategy_overrides = (raw.get("strategies", {}) or {}).get(strategy_id, {}) if strategy_id else {}
    budget_values = dict(raw.get("budgets", DEFAULT_BUDGETS.model_dump()))
    budget_values.update(strategy_overrides.get("budgets", {}))
    gates_raw = strategy_overrides.get("gates", raw.get("gates", DEFAULT_GATES.model_dump()))
    if isinstance(gates_raw, list):
        gates_raw = {"gates": gates_raw}
    return {
        "registry_path": raw.get("registry_path", "research_registry/research_pipeline.sqlite3"),
        "budgets": ResearchBudget.model_validate(budget_values),
        "gates": GateSet.model_validate(gates_raw),
        "report_size_limit_mb": raw.get("report_size_limit_mb", DEFAULT_BUDGETS.max_report_size_mb),
        "specification_generation": raw.get("specification_generation", {"max_generation_attempts": 3, "max_repair_attempts": 2}),
        "allowed_timeframes": raw.get("allowed_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]),
        "allowed_terminal_states": raw.get("allowed_terminal_states", ["ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE", "TECHNICAL_FAILURE", "MANUAL_REVIEW_REQUIRED"]),
        "logging": raw.get("logging", {"path": "research_registry/research_pipeline.log", "level": "INFO"}),
    }
