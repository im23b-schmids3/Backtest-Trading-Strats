from __future__ import annotations

from datetime import datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

from .models import ActionType, ProposedAction, SessionPolicy
from ..schemas.strategy_spec import StrictModel


class SessionDecision(StrEnum):
    ENTRY_ALLOWED = "ENTRY_ALLOWED"
    ENTRY_BLOCKED_SESSION_CUTOFF = "ENTRY_BLOCKED_SESSION_CUTOFF"
    CANCEL_PENDING_ORDERS = "CANCEL_PENDING_ORDERS"
    FORCE_FLATTEN = "FORCE_FLATTEN"
    SESSION_CLOSED = "SESSION_CLOSED"
    ALLOW = "ALLOW"


class SessionDecisionResult(StrictModel):
    decision: SessionDecision
    allowed: bool
    evaluated_timestamp: datetime
    local_timestamp: datetime
    reason: str
    required_actions: list[str] = []


class SessionManager:
    ENTRY_ACTIONS = {ActionType.NEW_ENTRY_ALERT, ActionType.ORDER_SUBMISSION, ActionType.PENDING_ORDER, ActionType.STRATEGY_REENTRY}

    def evaluate(self, timestamp: datetime, action: ProposedAction, policy: SessionPolicy) -> SessionDecisionResult:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("session evaluation requires a timezone-aware timestamp")
        local = timestamp.astimezone(ZoneInfo(policy.timezone))
        if not policy.enabled:
            return SessionDecisionResult(decision=SessionDecision.ALLOW, allowed=True, evaluated_timestamp=timestamp, local_timestamp=local, reason="session policy disabled")

        override = policy.holiday_overrides.get(local.date().isoformat())
        if override and override.closed:
            return SessionDecisionResult(decision=SessionDecision.SESSION_CLOSED, allowed=False, evaluated_timestamp=timestamp, local_timestamp=local, reason="configured holiday/session override is closed")
        if local.weekday() >= 5:
            if policy.weekend_flattening and action.action == ActionType.EXIT:
                return SessionDecisionResult(decision=SessionDecision.FORCE_FLATTEN, allowed=True, evaluated_timestamp=timestamp, local_timestamp=local, reason="weekend flattening is configured", required_actions=["FORCE_FLATTEN"])
            return SessionDecisionResult(decision=SessionDecision.SESSION_CLOSED, allowed=False, evaluated_timestamp=timestamp, local_timestamp=local, reason="weekend session is closed")

        no_entry = override.no_new_entry_time if override else policy.no_new_entry_time
        cancel = override.pending_order_cancellation_time if override else policy.pending_order_cancellation_time
        forced = override.forced_flat_time if override else policy.forced_flat_time
        forced = forced or policy.official_deadline
        current = local.timetz().replace(tzinfo=None)

        if forced is not None and current >= forced and action.action != ActionType.EXIT:
            return SessionDecisionResult(decision=SessionDecision.FORCE_FLATTEN, allowed=False, evaluated_timestamp=timestamp, local_timestamp=local, reason="forced-flat/deadline boundary reached", required_actions=["FORCE_FLATTEN"])
        if cancel is not None and current >= cancel and action.action == ActionType.PENDING_ORDER:
            return SessionDecisionResult(decision=SessionDecision.CANCEL_PENDING_ORDERS, allowed=False, evaluated_timestamp=timestamp, local_timestamp=local, reason="pending-order cancellation boundary reached", required_actions=["CANCEL_PENDING_ORDERS"])
        if no_entry is not None and current >= no_entry and action.action in self.ENTRY_ACTIONS:
            return SessionDecisionResult(decision=SessionDecision.ENTRY_BLOCKED_SESSION_CUTOFF, allowed=False, evaluated_timestamp=timestamp, local_timestamp=local, reason="new-entry cutoff boundary reached")
        if forced is not None and current >= forced and action.action == ActionType.EXIT:
            return SessionDecisionResult(decision=SessionDecision.FORCE_FLATTEN, allowed=True, evaluated_timestamp=timestamp, local_timestamp=local, reason="forced-flat/deadline boundary reached", required_actions=["FORCE_FLATTEN"])
        return SessionDecisionResult(decision=SessionDecision.ENTRY_ALLOWED if action.action in self.ENTRY_ACTIONS else SessionDecision.ALLOW, allowed=True, evaluated_timestamp=timestamp, local_timestamp=local, reason="session action allowed")
