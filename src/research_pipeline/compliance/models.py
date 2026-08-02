from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..schemas.strategy_spec import StrictModel


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return value


class ExecutionMode(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    ALERT_ONLY = "ALERT_ONLY"
    SEMI_AUTOMATED = "SEMI_AUTOMATED"
    AUTOMATED = "AUTOMATED"


class ActionType(StrEnum):
    NEW_ENTRY_ALERT = "NEW_ENTRY_ALERT"
    ORDER_SUBMISSION = "ORDER_SUBMISSION"
    ORDER_MODIFICATION = "ORDER_MODIFICATION"
    PENDING_ORDER = "PENDING_ORDER"
    STRATEGY_REENTRY = "STRATEGY_REENTRY"
    EXIT = "EXIT"


class ComplianceClassification(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DATA_STALE = "DATA_STALE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INFORMATIONAL_WARNING = "INFORMATIONAL_WARNING"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class ImpactLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class PolicyEvidence(StrictModel):
    reference: str
    retrieved_at: datetime
    effective_date: date | None = None
    applies_to_account_types: list[str] = Field(default_factory=list)
    first_party: bool = False

    _check_retrieved_at = field_validator("retrieved_at")(_aware)


class NewsTradingPolicy(StrictModel):
    enabled: bool = False
    impact_levels: list[ImpactLevel] = Field(default_factory=list)
    minutes_before: int = Field(default=0, ge=0)
    minutes_after: int = Field(default=0, ge=0)
    applicable_currencies: list[str] = Field(default_factory=list)
    applicable_instruments: list[str] = Field(default_factory=list)
    block_new_entries: bool = True
    block_exits: bool = False
    block_order_modifications: bool = True
    cancel_pending_entries: bool = False
    allow_existing_positions_open: bool = True
    force_flatten: bool = False
    max_calendar_age_minutes: int | None = Field(default=None, ge=0)


class SessionOverride(StrictModel):
    closed: bool = False
    no_new_entry_time: time | None = None
    pending_order_cancellation_time: time | None = None
    forced_flat_time: time | None = None


class SessionPolicy(StrictModel):
    enabled: bool = False
    timezone: str = "UTC"
    no_new_entry_time: time | None = None
    pending_order_cancellation_time: time | None = None
    forced_flat_time: time | None = None
    official_deadline: time | None = None
    weekend_flattening: bool = False
    holiday_overrides: dict[str, SessionOverride] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class DailyLossPolicy(StrictModel):
    enabled: bool = False
    daily_loss_limit: float | None = Field(default=None, ge=0)
    internal_safety_fraction: float | None = Field(default=None, ge=0, le=1)
    soft_lock_fraction: float | None = Field(default=None, ge=0, le=1)
    reset_timezone: str = "UTC"
    reset_time: time = time(0, 0)
    block_new_entries: bool = True
    suppress_new_alerts: bool = True
    cancel_pending_orders: bool = False
    force_flatten: bool = False
    remain_locked_until_reset: bool = True

    @field_validator("reset_timezone")
    @classmethod
    def valid_reset_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class PositionLimitPolicy(StrictModel):
    enabled: bool = False
    maximum_positions: int | None = Field(default=None, ge=0)
    maximum_quantity: float | None = Field(default=None, ge=0)


class HoldingTimePolicy(StrictModel):
    enabled: bool = False
    short_duration_threshold_minutes: list[int] = Field(default_factory=list)
    small_price_movement_threshold: float | None = Field(default=None, ge=0)


class AutomationPolicy(StrictModel):
    execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY
    allow_order_submission: bool = False
    allow_order_modification: bool = False
    allow_pending_orders: bool = False
    allow_reentry: bool = False

    @model_validator(mode="after")
    def safe_default(self) -> "AutomationPolicy":
        if self.execution_mode == ExecutionMode.AUTOMATED and not self.allow_order_submission:
            return self
        return self


class PropFirmPolicy(StrictModel):
    policy_id: str = "unconfigured"
    firm_name: str = "UNCONFIGURED"
    account_type: str = "UNKNOWN"
    account_state: str = "UNCONFIGURED"
    policy_version: str = "unconfigured-1"
    effective_date: date | None = None
    policy_timezone: str = "UTC"
    source_references: list[PolicyEvidence] = Field(default_factory=list)
    news: NewsTradingPolicy = Field(default_factory=NewsTradingPolicy)
    session: SessionPolicy = Field(default_factory=SessionPolicy)
    daily_loss: DailyLossPolicy = Field(default_factory=DailyLossPolicy)
    position_limits: PositionLimitPolicy = Field(default_factory=PositionLimitPolicy)
    holding_time: HoldingTimePolicy = Field(default_factory=HoldingTimePolicy)
    automation: AutomationPolicy = Field(default_factory=AutomationPolicy)
    policy_hash: str = "pending"

    @field_validator("policy_timezone")
    @classmethod
    def valid_policy_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def validate_hash(self, info: ValidationInfo) -> "PropFirmPolicy":
        if (info.context or {}).get("skip_policy_hash_validation"):
            return self
        expected = calculate_policy_hash(self)
        if self.policy_hash != expected:
            raise ValueError("policy_hash does not match canonical policy")
        return self


class ProposedAction(StrictModel):
    action: ActionType
    instrument: str
    currency: str | None = None
    order_type: str | None = None
    quantity: float = Field(default=0, ge=0)
    direction: str | None = None


class AccountState(StrictModel):
    account_id: str
    current_equity: float
    daily_start_equity: float | None = None
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    commissions: float = Field(default=0, ge=0)
    exchange_fees: float = Field(default=0, ge=0)
    slippage: float = Field(default=0, ge=0)
    other_costs: float = Field(default=0, ge=0)
    open_positions: int = Field(default=0, ge=0)
    open_quantity: float = Field(default=0, ge=0)


class MarketState(StrictModel):
    calendar_available: bool = True
    calendar_retrieved_at: datetime | None = None
    calendar_source_hash: str | None = None
    calendar_events: list["EconomicEvent"] = Field(default_factory=list)

    @field_validator("calendar_retrieved_at")
    @classmethod
    def aware_retrieval(cls, value: datetime | None) -> datetime | None:
        return _aware(value) if value is not None else None


class ComplianceViolation(StrictModel):
    code: str
    message: str
    classification: ComplianceClassification
    action: ActionType | None = None
    evidence_references: list[str] = Field(default_factory=list)


class ComplianceDecision(StrictModel):
    allowed: bool
    classification: ComplianceClassification
    violations: list[ComplianceViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    evaluated_timestamp: datetime
    policy_version: str
    evidence_references: list[str] = Field(default_factory=list)
    decision_hash: str = "pending"

    _check_timestamp = field_validator("evaluated_timestamp")(_aware)

    @model_validator(mode="after")
    def validate_decision_hash(self, info: ValidationInfo) -> "ComplianceDecision":
        if (info.context or {}).get("skip_decision_hash_validation"):
            return self
        expected = calculate_decision_hash(self)
        if self.decision_hash != expected:
            raise ValueError("decision_hash does not match canonical decision")
        return self


def _canonical_payload(model: StrictModel, excluded: set[str]) -> dict[str, Any]:
    payload = model.model_dump(mode="json")
    for key in excluded:
        payload.pop(key, None)
    return payload


def calculate_policy_hash(policy: PropFirmPolicy | dict[str, Any]) -> str:
    if not isinstance(policy, PropFirmPolicy):
        data = dict(policy)
        data["policy_hash"] = str(data.get("policy_hash") or "pending")
        policy = PropFirmPolicy.model_validate(data, context={"skip_policy_hash_validation": True})
    encoded = json.dumps(_canonical_payload(policy, {"policy_hash"}), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def calculate_decision_hash(decision: ComplianceDecision | dict[str, Any]) -> str:
    if not isinstance(decision, ComplianceDecision):
        data = dict(decision)
        data["decision_hash"] = str(data.get("decision_hash") or "pending")
        decision = ComplianceDecision.model_validate(data, context={"skip_decision_hash_validation": True})
    encoded = json.dumps(_canonical_payload(decision, {"decision_hash"}), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
