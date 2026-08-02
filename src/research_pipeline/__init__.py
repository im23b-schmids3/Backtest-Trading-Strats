"""Deterministic research-pipeline foundations.

Phase A deliberately contains no market-data, backtesting, or AI execution
code.  It provides schemas, process controls, and an auditable SQLite registry.
"""

from .enums import DecisionType, GateOutcomeStatus, PipelineState

__all__ = ["DecisionType", "GateOutcomeStatus", "PipelineState"]

