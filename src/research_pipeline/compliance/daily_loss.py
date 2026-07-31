from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from .models import AccountState, DailyLossPolicy
from ..schemas.strategy_spec import StrictModel


class DailyLossState(StrEnum):
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    SOFT_LOCK = "SOFT_LOCK"
    FIRM_LOCK = "FIRM_LOCK"
    RESET_PENDING = "RESET_PENDING"


class DailyLossResult(StrictModel):
    state: DailyLossState
    loss: float
    net_pnl: float
    current_equity: float
    daily_start_equity: float | None = None
    limit: float | None
    evaluated_timestamp: datetime
    reset_boundary: datetime
    transition: str | None = None
    required_actions: list[str] = []
    reason: str


class DailyLossGuard:
    def __init__(self) -> None:
        self._period_key: str | None = None
        self._state = DailyLossState.ACTIVE

    @staticmethod
    def _boundary(timestamp: datetime, policy: DailyLossPolicy) -> tuple[str, datetime]:
        local = timestamp.astimezone(ZoneInfo(policy.reset_timezone))
        day = local.date()
        if local.timetz().replace(tzinfo=None) < policy.reset_time:
            day -= timedelta(days=1)
        boundary = datetime.combine(day, policy.reset_time, tzinfo=ZoneInfo(policy.reset_timezone))
        return boundary.date().isoformat(), boundary

    def evaluate(self, timestamp: datetime, account: AccountState, policy: DailyLossPolicy) -> DailyLossResult:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("daily-loss evaluation requires a timezone-aware timestamp")
        key, boundary = self._boundary(timestamp, policy)
        if key != self._period_key:
            self._period_key = key
            self._state = DailyLossState.ACTIVE
        net_pnl = account.realized_pnl + account.unrealized_pnl - account.commissions - account.exchange_fees - account.slippage - account.other_costs
        loss = max(0.0, -net_pnl)
        if not policy.enabled or policy.daily_loss_limit is None:
            return DailyLossResult(state=self._state, loss=loss, net_pnl=net_pnl, current_equity=account.current_equity, daily_start_equity=account.daily_start_equity, limit=policy.daily_loss_limit, evaluated_timestamp=timestamp, reset_boundary=boundary, reason="daily-loss policy disabled or limit unavailable")
        limit = policy.daily_loss_limit
        if loss >= limit:
            target = DailyLossState.FIRM_LOCK
        elif policy.soft_lock_fraction is not None and loss >= limit * policy.soft_lock_fraction:
            target = DailyLossState.SOFT_LOCK
        elif policy.internal_safety_fraction is not None and loss >= limit * policy.internal_safety_fraction:
            target = DailyLossState.WARNING
        else:
            target = DailyLossState.ACTIVE
        transition = None if target == self._state else f"{self._state.value}->{target.value}"
        self._state = target
        actions: list[str] = []
        if target in {DailyLossState.WARNING, DailyLossState.SOFT_LOCK, DailyLossState.FIRM_LOCK} and policy.suppress_new_alerts:
            actions.append("SUPPRESS_NEW_ALERTS")
        if target in {DailyLossState.SOFT_LOCK, DailyLossState.FIRM_LOCK} and policy.block_new_entries:
            actions.append("BLOCK_NEW_ENTRIES")
        if target in {DailyLossState.SOFT_LOCK, DailyLossState.FIRM_LOCK} and policy.cancel_pending_orders:
            actions.append("CANCEL_PENDING_ORDERS")
        if target == DailyLossState.FIRM_LOCK and policy.force_flatten:
            actions.append("FORCE_FLATTEN")
        return DailyLossResult(state=target, loss=loss, net_pnl=net_pnl, current_equity=account.current_equity, daily_start_equity=account.daily_start_equity, limit=limit, evaluated_timestamp=timestamp, reset_boundary=boundary, transition=transition, required_actions=actions, reason=f"daily loss {loss:.2f} against limit {limit:.2f}")
