"""Provider-independent prop-firm compliance and execution controls.

The package is deliberately policy-driven.  It ships no verified firm rules;
callers must supply a policy profile and its evidence before enabling checks.
"""

from .calendar import (
    EconomicCalendarProvider,
    EconomicEvent,
    FixtureEconomicCalendarProvider,
    CalendarArtifact,
    save_calendar_artifact,
    load_calendar_artifact,
)
from .costs import (
    ExecutionCostConfig,
    ExecutionCostEngine,
    ExecutionCostResult,
    InstrumentCostConfig,
    OrderType,
    calculate_cost_config_hash,
)
from .daily_loss import DailyLossGuard, DailyLossResult, DailyLossState
from .diagnostics import ActivityDiagnostics, calculate_activity_diagnostics
from .evaluator import ComplianceEvaluator, evaluate_alert_action, evaluate_backtest_action
from .models import (
    AccountState,
    ActionType,
    AutomationPolicy,
    ComplianceClassification,
    ComplianceDecision,
    ComplianceViolation,
    DailyLossPolicy,
    ExecutionMode,
    HoldingTimePolicy,
    ImpactLevel,
    MarketState,
    NewsTradingPolicy,
    PolicyEvidence,
    PositionLimitPolicy,
    PropFirmPolicy,
    ProposedAction,
    SessionPolicy,
    SessionOverride,
    calculate_decision_hash,
)
from .policy import calculate_policy_hash, load_policy, save_policy, unconfigured_policy
from .session import SessionDecision, SessionDecisionResult, SessionManager
from .store import ComplianceStore

__all__ = [
    "AccountState",
    "ActionType",
    "ActivityDiagnostics",
    "AutomationPolicy",
    "CalendarArtifact",
    "ComplianceClassification",
    "ComplianceDecision",
    "ComplianceEvaluator",
    "ComplianceStore",
    "ComplianceViolation",
    "DailyLossPolicy",
    "EconomicCalendarProvider",
    "EconomicEvent",
    "ExecutionCostConfig",
    "ExecutionCostEngine",
    "ExecutionCostResult",
    "ExecutionMode",
    "evaluate_alert_action",
    "evaluate_backtest_action",
    "FixtureEconomicCalendarProvider",
    "HoldingTimePolicy",
    "ImpactLevel",
    "InstrumentCostConfig",
    "MarketState",
    "NewsTradingPolicy",
    "OrderType",
    "PositionLimitPolicy",
    "PolicyEvidence",
    "PropFirmPolicy",
    "ProposedAction",
    "SessionDecision",
    "SessionDecisionResult",
    "SessionManager",
    "SessionPolicy",
    "SessionOverride",
    "calculate_activity_diagnostics",
    "calculate_decision_hash",
    "calculate_cost_config_hash",
    "calculate_policy_hash",
    "load_calendar_artifact",
    "load_policy",
    "save_calendar_artifact",
    "save_policy",
    "unconfigured_policy",
]
