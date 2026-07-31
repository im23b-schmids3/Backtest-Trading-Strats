from __future__ import annotations

import hashlib
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field, field_validator

from ..compliance import (
    AccountState,
    ActionType,
    ComplianceDecision,
    ComplianceEvaluator,
    ExecutionCostConfig,
    ExecutionCostEngine,
    InstrumentCostConfig,
    MarketState,
    OrderType,
    PropFirmPolicy,
    ProposedAction,
    SessionPolicy,
    calculate_cost_config_hash,
    calculate_policy_hash,
    unconfigured_policy,
)
from ..compliance.diagnostics import calculate_activity_diagnostics
from ..schemas.strategy_spec import StrictModel


class RandomOpenTestConfig(StrictModel):
    """Explicit, bounded parameters for the reference integration strategy."""

    instrument: str = "SPY"
    seed: str = "RandomOpenTest"
    timezone: str = "America/New_York"
    session_open: time = time(9, 30)
    forced_flat_time: time = time(16, 0)
    quantity: float = Field(default=1, gt=0)
    initial_capital: float = Field(default=10_000, gt=0)
    initial_stop_ticks: int = Field(default=4, ge=1)
    profit_target_ticks: int = Field(default=8, ge=1)
    tick_size: float = Field(default=0.01, gt=0)
    test_start_date: date | None = None
    test_end_date: date | None = None
    allow_reentry: bool = False
    allow_pyramiding: bool = False

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class RandomOpenTestSignal(StrictModel):
    instrument: str
    trading_date: date
    entry_timestamp: datetime
    direction: str
    seed_material: str
    quantity: float
    entry_reference_price: float
    initial_stop_price: float
    profit_target_price: float


class RandomOpenTestRun(StrictModel):
    strategy_id: str = "RandomOpenTest"
    seed: str
    direction_inputs: list[str] = Field(default_factory=list)
    proposed_entries: int
    accepted_entries: int
    blocked_entries: int
    blocked_signals: list[dict[str, Any]] = Field(default_factory=list)
    compliance_decisions: list[ComplianceDecision] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    forced_flat_trade_count: int = 0
    gross_pnl: float = 0
    commissions: float = 0
    fees: float = 0
    slippage_cost: float = 0
    net_pnl: float = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    policy_hash: str
    execution_cost_configuration_hash: str


def stable_random_direction(seed: str, instrument: str, trading_date: date | str) -> str:
    material = f"{seed}|{instrument}|{trading_date}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return "LONG" if digest[0] % 2 == 0 else "SHORT"


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def generate_random_open_signals(bars: pd.DataFrame, config: RandomOpenTestConfig) -> list[RandomOpenTestSignal]:
    """Generate at most one deterministic proposal for each local trading day."""
    if bars.empty:
        return []
    ordered = bars.sort_index().copy()
    ordered.index = pd.DatetimeIndex([_timestamp(item) for item in ordered.index])
    local = ordered.index.tz_convert(config.timezone)
    frame = ordered.assign(_local_date=[item.date() for item in local], _local_time=[item.timetz().replace(tzinfo=None) for item in local])
    signals: list[RandomOpenTestSignal] = []
    for trading_day, day_rows in frame.groupby("_local_date", sort=True):
        if config.test_start_date and trading_day < config.test_start_date:
            continue
        if config.test_end_date and trading_day >= config.test_end_date:
            continue
        eligible = day_rows[day_rows["_local_time"] >= config.session_open]
        if eligible.empty:
            continue
        row = eligible.iloc[0]
        timestamp = _timestamp(eligible.index[0]).to_pydatetime()
        reference = float(row["open"])
        direction = stable_random_direction(config.seed, config.instrument, trading_day)
        sign = 1 if direction == "LONG" else -1
        signals.append(RandomOpenTestSignal(instrument=config.instrument, trading_date=trading_day, entry_timestamp=timestamp, direction=direction, seed_material=f"{config.seed}|{config.instrument}|{trading_day}", quantity=config.quantity, entry_reference_price=reference, initial_stop_price=reference - sign * config.initial_stop_ticks * config.tick_size, profit_target_price=reference + sign * config.profit_target_ticks * config.tick_size))
    return signals


def default_random_open_cost_config(config: RandomOpenTestConfig) -> ExecutionCostConfig:
    raw = {"instruments": {config.instrument: InstrumentCostConfig(tick_size=config.tick_size, tick_value=1.0, commission_per_side=0.01, exchange_fee_per_side=0.0, regulatory_fee_per_side=0.0, market_slippage_ticks=1, stop_slippage_ticks=1, limit_slippage_ticks=0).model_dump()}, "configuration_hash": "pending"}
    candidate = ExecutionCostConfig.model_validate(raw, context={"skip_configuration_hash_validation": True})
    raw["configuration_hash"] = calculate_cost_config_hash(candidate)
    return ExecutionCostConfig.model_validate(raw)


def default_random_open_policy(config: RandomOpenTestConfig) -> PropFirmPolicy:
    base = unconfigured_policy().model_dump(mode="python")
    base["policy_id"] = "random-open-test-research"
    base["account_type"] = "RESEARCH_FIXTURE"
    base["policy_timezone"] = config.timezone
    base["session"] = SessionPolicy(enabled=True, timezone=config.timezone, forced_flat_time=config.forced_flat_time).model_dump(mode="python")
    base["policy_hash"] = "pending"
    candidate = PropFirmPolicy.model_validate(base, context={"skip_policy_hash_validation": True})
    base["policy_hash"] = calculate_policy_hash(candidate)
    return PropFirmPolicy.model_validate(base)


def run_random_open_test(
    bars: pd.DataFrame,
    config: RandomOpenTestConfig,
    *,
    policy: PropFirmPolicy | None = None,
    cost_config: ExecutionCostConfig | None = None,
    evaluator: ComplianceEvaluator | None = None,
    market_state: MarketState | None = None,
) -> RandomOpenTestRun:
    """Run the reference adapter in memory without invoking a broker."""
    policy = policy or default_random_open_policy(config)
    cost_config = cost_config or default_random_open_cost_config(config)
    evaluator = evaluator or ComplianceEvaluator()
    market_state = market_state or MarketState(calendar_source_hash="random-open-empty-calendar")
    signals = generate_random_open_signals(bars, config)
    ordered = bars.sort_index().copy()
    ordered.index = pd.DatetimeIndex([_timestamp(item) for item in ordered.index])
    local = ordered.index.tz_convert(config.timezone)
    decisions: list[ComplianceDecision] = []
    blocked: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity = config.initial_capital
    cost_engine = ExecutionCostEngine(cost_config)
    forced_flat_time = policy.session.forced_flat_time or config.forced_flat_time
    for signal in signals:
        account = AccountState(account_id="random-open-test", current_equity=equity, open_positions=0, open_quantity=0)
        decision = evaluator.evaluate_backtest(timestamp=signal.entry_timestamp, instrument=signal.instrument, account_state=account, market_state=market_state, proposed_action=ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument=signal.instrument, quantity=signal.quantity, direction=signal.direction), policy=policy)
        decisions.append(decision)
        if not decision.allowed:
            blocked.append({"signal": signal.model_dump(mode="json"), "reason": decision.classification.value, "required_actions": decision.required_actions, "decision_hash": decision.decision_hash})
            continue
        day_mask = [item.date() == signal.trading_date for item in local]
        day_indices = [index for index, selected in enumerate(day_mask) if selected]
        entry_index = next(index for index in day_indices if ordered.index[index] == pd.Timestamp(signal.entry_timestamp))
        exit_index = day_indices[-1]
        exit_reason = "forced_flat"
        exit_price = float(ordered.iloc[exit_index]["close"])
        sign = 1 if signal.direction == "LONG" else -1
        stop = signal.initial_stop_price
        target = signal.profit_target_price
        entry_position = day_indices.index(entry_index)
        for index in day_indices[entry_position:]:
            bar = ordered.iloc[index]
            if sign == 1 and float(bar["low"]) <= stop:
                exit_index, exit_price, exit_reason = index, stop - cost_config.instruments[config.instrument].stop_slippage_ticks * config.tick_size, "stop"
                break
            if sign == -1 and float(bar["high"]) >= stop:
                exit_index, exit_price, exit_reason = index, stop + cost_config.instruments[config.instrument].stop_slippage_ticks * config.tick_size, "stop"
                break
            if sign == 1 and float(bar["high"]) >= target:
                exit_index, exit_price, exit_reason = index, target, "target"
                break
            if sign == -1 and float(bar["low"]) <= target:
                exit_index, exit_price, exit_reason = index, target, "target"
                break
            current_local = local[index].timetz().replace(tzinfo=None)
            if current_local >= forced_flat_time:
                exit_index, exit_price, exit_reason = index, float(bar["close"]), "forced_flat"
                break
        entry_price = signal.entry_reference_price + sign * cost_config.instruments[config.instrument].market_slippage_ticks * config.tick_size
        exit_type = OrderType.STOP if exit_reason == "stop" else OrderType.LIMIT if exit_reason == "target" else OrderType.MARKET
        costs = cost_engine.calculate(config.instrument, signal.quantity, order_types=(OrderType.MARKET, exit_type))
        gross = sign * (exit_price - entry_price) * signal.quantity
        net = gross - costs.total_cost
        equity += net
        exit_timestamp = ordered.index[exit_index].to_pydatetime()
        trades.append({"trade_id": f"{signal.instrument}-{signal.trading_date}-random-open", "instrument": signal.instrument, "direction": signal.direction, "seed_material": signal.seed_material, "entry_timestamp": signal.entry_timestamp.isoformat(), "exit_timestamp": exit_timestamp.isoformat(), "entry_price": entry_price, "exit_price": exit_price, "quantity": signal.quantity, "initial_stop_price": signal.initial_stop_price, "profit_target_price": signal.profit_target_price, "exit_reason": exit_reason, "gross_pnl": gross, "commissions": costs.commissions, "fees": costs.exchange_fees + costs.regulatory_fees, "slippage_cost": costs.slippage_cost, "net_pnl": net, "compliance_decision_hash": decision.decision_hash, "execution_cost_configuration_hash": costs.configuration_hash})
    diagnostics = calculate_activity_diagnostics([{"entry_time": datetime.fromisoformat(item["entry_timestamp"]), "exit_time": datetime.fromisoformat(item["exit_timestamp"]), "net_pnl": item["net_pnl"], "gross_pnl": item["gross_pnl"], "entry": item["entry_price"]} for item in trades])
    return RandomOpenTestRun(seed=config.seed, direction_inputs=[item.seed_material for item in signals], proposed_entries=len(signals), accepted_entries=len(trades), blocked_entries=len(blocked), blocked_signals=blocked, compliance_decisions=decisions, trades=trades, forced_flat_trade_count=sum(item["exit_reason"] == "forced_flat" for item in trades), gross_pnl=sum(item["gross_pnl"] for item in trades), commissions=sum(item["commissions"] for item in trades), fees=sum(item["fees"] for item in trades), slippage_cost=sum(item["slippage_cost"] for item in trades), net_pnl=sum(item["net_pnl"] for item in trades), diagnostics=diagnostics.model_dump(mode="json"), policy_hash=policy.policy_hash, execution_cost_configuration_hash=cost_config.configuration_hash)
