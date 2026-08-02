from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .calendar import EconomicEvent
from .daily_loss import DailyLossGuard
from .models import (
    AccountState,
    ActionType,
    ComplianceClassification,
    ComplianceDecision,
    ComplianceViolation,
    ExecutionMode,
    MarketState,
    PropFirmPolicy,
    ProposedAction,
    calculate_decision_hash,
)
from .session import SessionManager


class ComplianceEvaluator:
    """Single deterministic decision boundary for backtest and alert callers."""

    def __init__(self, *, daily_loss_guard: DailyLossGuard | None = None, session_manager: SessionManager | None = None):
        self.daily_loss_guard = daily_loss_guard or DailyLossGuard()
        self.session_manager = session_manager or SessionManager()

    @staticmethod
    def _news_match(event: EconomicEvent, action: ProposedAction, policy: PropFirmPolicy) -> bool:
        if policy.news.impact_levels and event.impact_level.upper() not in {item.value for item in policy.news.impact_levels}:
            return False
        currencies = {item.upper() for item in policy.news.applicable_currencies}
        instruments = set(policy.news.applicable_instruments)
        event_currencies = {item.upper() for item in event.affected_currencies}
        event_instruments = set(event.affected_instruments)
        currency_match = not currencies or bool(currencies.intersection(event_currencies)) or (not event_currencies and (action.currency or "").upper() in currencies)
        instrument_match = not instruments or bool(instruments.intersection(event_instruments)) or (not event_instruments and action.instrument in instruments)
        return currency_match and instrument_match

    def _news_violations(self, timestamp: datetime, action: ProposedAction, market: MarketState, policy: PropFirmPolicy) -> tuple[list[ComplianceViolation], list[str], list[str], ComplianceClassification | None]:
        if not policy.news.enabled:
            return [], [], [], None
        if not market.calendar_available:
            classification = ComplianceClassification.DATA_UNAVAILABLE if policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY else ComplianceClassification.MANUAL_REVIEW_REQUIRED
            return [], ["economic calendar data unavailable; no event absence was assumed"], ["RESEARCH_DATA_GAP"], classification
        if policy.news.max_calendar_age_minutes is not None:
            if market.calendar_retrieved_at is None:
                classification = ComplianceClassification.DATA_UNAVAILABLE if policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY else ComplianceClassification.MANUAL_REVIEW_REQUIRED
                return [], ["economic calendar retrieval timestamp unavailable"], ["CALENDAR_DATA_REQUIRED"], classification
            age = (timestamp - market.calendar_retrieved_at).total_seconds() / 60
            if age > policy.news.max_calendar_age_minutes:
                classification = ComplianceClassification.DATA_STALE if policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY else ComplianceClassification.MANUAL_REVIEW_REQUIRED
                return [], [f"economic calendar data is stale by {age:.1f} minutes"], ["CALENDAR_DATA_STALE"], classification
        if not market.calendar_source_hash:
            classification = ComplianceClassification.DATA_UNAVAILABLE if policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY else ComplianceClassification.MANUAL_REVIEW_REQUIRED
            return [], ["economic calendar artifact hash unavailable; no event absence was assumed"], ["CALENDAR_ARTIFACT_HASH_REQUIRED"], classification
        violations: list[ComplianceViolation] = []
        warnings: list[str] = []
        actions: list[str] = []
        for event in market.calendar_events:
            if not self._news_match(event, action, policy):
                continue
            delta = (timestamp - event.timestamp).total_seconds() / 60
            if -policy.news.minutes_before <= delta <= policy.news.minutes_after:
                block = ((action.action in {ActionType.NEW_ENTRY_ALERT, ActionType.ORDER_SUBMISSION, ActionType.PENDING_ORDER, ActionType.STRATEGY_REENTRY} and policy.news.block_new_entries) or (action.action == ActionType.EXIT and policy.news.block_exits) or (action.action == ActionType.ORDER_MODIFICATION and policy.news.block_order_modifications))
                if block:
                    violations.append(ComplianceViolation(code="NEWS_WINDOW", message=f"action is within configured window for {event.title}", classification=ComplianceClassification.POLICY_VIOLATION, action=action.action, evidence_references=[event.event_id, event.source]))
                if policy.news.cancel_pending_entries and action.action == ActionType.PENDING_ORDER:
                    actions.append("CANCEL_PENDING_ENTRIES")
                if policy.news.force_flatten:
                    actions.append("FORCE_FLATTEN")
                if policy.news.allow_existing_positions_open:
                    warnings.append("existing positions may remain open under this news policy")
        return violations, warnings, actions, ComplianceClassification.POLICY_VIOLATION if violations else None

    def evaluate(self, *, timestamp: datetime, instrument: str, account_state: AccountState, market_state: MarketState, proposed_action: ProposedAction, policy: PropFirmPolicy) -> ComplianceDecision:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("compliance evaluation requires a timezone-aware timestamp")
        violations: list[ComplianceViolation] = []
        warnings: list[str] = []
        actions: list[str] = []
        classifications: list[ComplianceClassification] = []
        if policy.account_type == "UNKNOWN":
            if policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY:
                warnings.append("account type is unknown; firm-specific compliance evidence is unavailable")
            else:
                violations.append(ComplianceViolation(code="UNKNOWN_ACCOUNT_TYPE", message="account type must be configured before alert or order evaluation", classification=ComplianceClassification.MANUAL_REVIEW_REQUIRED))
        session = self.session_manager.evaluate(timestamp, proposed_action, policy.session)
        if not session.allowed:
            violations.append(ComplianceViolation(code=session.decision.value, message=session.reason, classification=ComplianceClassification.POLICY_VIOLATION, action=proposed_action.action))
        actions.extend(session.required_actions)

        daily = self.daily_loss_guard.evaluate(timestamp, account_state, policy.daily_loss)
        actions.extend(daily.required_actions)
        if daily.state.value in {"SOFT_LOCK", "FIRM_LOCK"} and proposed_action.action in {ActionType.NEW_ENTRY_ALERT, ActionType.ORDER_SUBMISSION, ActionType.PENDING_ORDER, ActionType.STRATEGY_REENTRY}:
            violations.append(ComplianceViolation(code=f"DAILY_LOSS_{daily.state.value}", message=daily.reason, classification=ComplianceClassification.POLICY_VIOLATION, action=proposed_action.action))
        elif daily.state.value == "WARNING":
            warnings.append(daily.reason)

        if policy.position_limits.enabled:
            if policy.position_limits.maximum_positions is not None and account_state.open_positions + (1 if proposed_action.action in {ActionType.ORDER_SUBMISSION, ActionType.NEW_ENTRY_ALERT, ActionType.PENDING_ORDER} else 0) > policy.position_limits.maximum_positions:
                violations.append(ComplianceViolation(code="POSITION_LIMIT", message="maximum position count exceeded", classification=ComplianceClassification.POLICY_VIOLATION, action=proposed_action.action))
            if policy.position_limits.maximum_quantity is not None and account_state.open_quantity + proposed_action.quantity > policy.position_limits.maximum_quantity:
                violations.append(ComplianceViolation(code="QUANTITY_LIMIT", message="maximum position quantity exceeded", classification=ComplianceClassification.POLICY_VIOLATION, action=proposed_action.action))

        news_violations, news_warnings, news_actions, news_classification = self._news_violations(timestamp, proposed_action, market_state, policy)
        violations.extend(news_violations); warnings.extend(news_warnings); actions.extend(news_actions)
        if news_classification is not None:
            classifications.append(news_classification)

        if proposed_action.action == ActionType.ORDER_SUBMISSION and not policy.automation.allow_order_submission and policy.automation.execution_mode != ExecutionMode.RESEARCH_ONLY:
            violations.append(ComplianceViolation(code="AUTOMATION_DISABLED", message="order submission is not enabled by the policy profile", classification=ComplianceClassification.MANUAL_REVIEW_REQUIRED, action=proposed_action.action))
        if proposed_action.action == ActionType.ORDER_MODIFICATION and not policy.automation.allow_order_modification and policy.automation.execution_mode != ExecutionMode.RESEARCH_ONLY:
            violations.append(ComplianceViolation(code="AUTOMATION_MODIFICATION_DISABLED", message="order modification is not enabled by the policy profile", classification=ComplianceClassification.MANUAL_REVIEW_REQUIRED, action=proposed_action.action))
        if proposed_action.action == ActionType.PENDING_ORDER and not policy.automation.allow_pending_orders and policy.automation.execution_mode != ExecutionMode.RESEARCH_ONLY:
            violations.append(ComplianceViolation(code="AUTOMATION_PENDING_DISABLED", message="pending orders are not enabled by the policy profile", classification=ComplianceClassification.MANUAL_REVIEW_REQUIRED, action=proposed_action.action))

        if violations:
            classification = ComplianceClassification.BLOCK if any(item.classification == ComplianceClassification.POLICY_VIOLATION for item in violations) else ComplianceClassification.MANUAL_REVIEW_REQUIRED
            allowed = False
        elif classifications:
            classification = classifications[0]
            allowed = policy.automation.execution_mode == ExecutionMode.RESEARCH_ONLY
        elif warnings:
            classification = ComplianceClassification.INFORMATIONAL_WARNING
            allowed = True
        else:
            classification = ComplianceClassification.ALLOW
            allowed = True
        payload = {
            "allowed": allowed,
            "classification": classification,
            "violations": violations,
            "warnings": warnings,
            "required_actions": sorted(set(actions)),
            "evaluated_timestamp": timestamp,
            "policy_version": policy.policy_version,
            "evidence_references": sorted(set(market_state.calendar_source_hash and [market_state.calendar_source_hash] or [])),
            "decision_hash": "pending",
        }
        decision = ComplianceDecision.model_validate(payload, context={"skip_decision_hash_validation": True})
        payload = decision.model_dump(mode="python")
        payload["decision_hash"] = calculate_decision_hash(decision)
        return ComplianceDecision.model_validate(payload)

    def evaluate_backtest(self, **kwargs: Any) -> ComplianceDecision:
        """Evaluate a proposed backtest action through the shared boundary."""
        return self.evaluate(**kwargs)

    def evaluate_alert(self, **kwargs: Any) -> ComplianceDecision:
        """Evaluate a proposed alert through the same boundary as backtests."""
        return self.evaluate(**kwargs)


def evaluate_backtest_action(**kwargs: Any) -> ComplianceDecision:
    return ComplianceEvaluator().evaluate(**kwargs)


def evaluate_alert_action(**kwargs: Any) -> ComplianceDecision:
    return ComplianceEvaluator().evaluate(**kwargs)
