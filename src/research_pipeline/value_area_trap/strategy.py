from __future__ import annotations

import hashlib
import json
from datetime import datetime, time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Literal

from pydantic import Field

from ..compliance import AccountState, ActionType, ComplianceDecision, ComplianceEvaluator, ExecutionCostConfig, ExecutionCostEngine, InstrumentCostConfig, MarketState, OrderType, PropFirmPolicy, ProposedAction, SessionPolicy, calculate_cost_config_hash, calculate_policy_hash, unconfigured_policy
from ..schemas.strategy_spec import StrictModel
from .profile import FiveMinuteBar, SessionProfile


class ValueAreaTrapConfig(StrictModel):
    symbol: str = "BTCUSDT"
    session_timezone: str = "America/New_York"
    value_area_fraction: Decimal = Decimal("0.70")
    breakout_volume_multiplier: Decimal = Decimal("1.5")
    breakout_volume_lookback_bars: int = 10
    minimum_breakout_buckets: int = 1
    swing_left_bars: int = 2
    swing_right_bars: int = 2
    stop_buffer_buckets: int = 1
    maximum_trades_per_day: int = 1
    same_bar_stop_target_policy: Literal["stop_first"] = "stop_first"
    quantity: Decimal = Decimal("0.001")
    minimum_quantity: Decimal = Decimal("0.001")
    quantity_step: Decimal = Decimal("0.001")
    price_tick: Decimal = Decimal("0.10")
    forced_flat_time: time = time(16, 0)
    variant: Literal["VALUE_AREA_RETURN_ONLY", "VALUE_AREA_STOP_RUN", "VALUE_AREA_CVD_DIVERGENCE", "FULL"] = "FULL"


class ValueAreaTrapResult(StrictModel):
    strategy_id: str = "ValueAreaTrap"
    strategy_family: str = "value_area_trap_reference"
    proposed_setups: int = 0
    significant_stop_runs: int = 0
    confirmed_divergences: int = 0
    return_triggers: int = 0
    compliance_blocks: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    setup_events: list[dict[str, Any]] = Field(default_factory=list)
    gross_pnl: Decimal = Decimal()
    fees: Decimal = Decimal()
    slippage_cost: Decimal = Decimal()
    net_pnl: Decimal = Decimal()
    forced_flat_count: int = 0
    same_bar_ambiguity_count: int = 0
    policy_hash: str
    cost_model_hash: str


def default_value_area_costs(config: ValueAreaTrapConfig) -> ExecutionCostConfig:
    raw = {"model_version": "binance-btcusdt-research-assumption-1", "instruments": {config.symbol: InstrumentCostConfig(tick_size=float(config.price_tick), tick_value=float(config.price_tick), commission_per_side=0.0, exchange_fee_per_side=0.0005, regulatory_fee_per_side=0.0, market_slippage_ticks=1, stop_slippage_ticks=2, limit_slippage_ticks=0, limit_fill_assumption="RESEARCH_ASSUMPTION").model_dump()}, "configuration_hash": "pending"}
    candidate = ExecutionCostConfig.model_validate(raw, context={"skip_configuration_hash_validation": True})
    raw["configuration_hash"] = calculate_cost_config_hash(candidate)
    return ExecutionCostConfig.model_validate(raw)


def default_value_area_policy(config: ValueAreaTrapConfig) -> PropFirmPolicy:
    raw = unconfigured_policy().model_dump(mode="python")
    raw.update({"policy_id": "value-area-trap-research-only", "account_type": "RESEARCH_FIXTURE", "policy_timezone": config.session_timezone, "session": SessionPolicy(enabled=True, timezone=config.session_timezone, forced_flat_time=config.forced_flat_time).model_dump(mode="python"), "policy_hash": "pending"})
    candidate = PropFirmPolicy.model_validate(raw, context={"skip_policy_hash_validation": True})
    raw["policy_hash"] = calculate_policy_hash(candidate)
    return PropFirmPolicy.model_validate(raw)


def _swings(bars: list[FiveMinuteBar], current: int, *, side: str) -> tuple[int, FiveMinuteBar] | None:
    i = current - 2
    if i < 2 or current >= len(bars):
        return None
    centre = bars[i]
    left, right = bars[i - 2:i], bars[i + 1:i + 3]
    if len(left) != 2 or len(right) != 2:
        return None
    if side == "SHORT" and all(centre.high > item.high for item in left) and all(centre.high >= item.high for item in right):
        return i, centre
    if side == "LONG" and all(centre.low < item.low for item in left) and all(centre.low <= item.low for item in right):
        return i, centre
    return None


def _eligible_quantity(config: ValueAreaTrapConfig, value: Decimal | None = None) -> Decimal | None:
    quantity = value if value is not None else config.quantity
    rounded = (quantity / config.quantity_step).to_integral_value(rounding=ROUND_DOWN) * config.quantity_step
    return rounded if rounded >= config.minimum_quantity else None


def _previous_profile(day, profiles: dict) -> SessionProfile | None:
    candidates = [item for item in profiles if item < day]
    return profiles[max(candidates)] if candidates else None


def run_value_area_trap(
    bars: list[FiveMinuteBar],
    profiles: dict,
    config: ValueAreaTrapConfig = ValueAreaTrapConfig(),
    *,
    policy: PropFirmPolicy | None = None,
    cost_config: ExecutionCostConfig | None = None,
    evaluator: ComplianceEvaluator | None = None,
    market_state: MarketState | None = None,
) -> ValueAreaTrapResult:
    """Deterministic, completed-bar-only ValueAreaTrap baseline.

    A swing at i is introduced only while processing i+2; entries are always
    proposed for the next available bar, never the return-trigger close.
    """
    policy = policy or default_value_area_policy(config)
    costs = cost_config or default_value_area_costs(config)
    evaluator = evaluator or ComplianceEvaluator()
    market_state = market_state or MarketState(calendar_source_hash="value-area-trap-no-calendar")
    by_day: dict[Any, list[FiveMinuteBar]] = {}
    for bar in sorted(bars, key=lambda item: item.start_utc):
        by_day.setdefault(bar.session_date, []).append(bar)
    events: list[dict[str, Any]] = []; trades: list[dict[str, Any]] = []; blocked: list[dict[str, Any]] = []
    proposed = stop_runs = divergences = returns = force_flat = ambiguities = 0
    equity = Decimal("10000"); engine = ExecutionCostEngine(costs)
    for day, day_bars in sorted(by_day.items()):
        profile = _previous_profile(day, profiles)
        if profile is None:
            events.append({"session_date": str(day), "state": "SESSION_EXPIRED", "reason": "MISSING_PREVIOUS_PROFILE"}); continue
        setups = {"SHORT": {"state": "IDLE", "swings": [], "extreme": None}, "LONG": {"state": "IDLE", "swings": [], "extreme": None}}
        traded = False
        for index, bar in enumerate(day_bars):
            history = day_bars[max(0, index - config.breakout_volume_lookback_bars):index]
            median = sorted((item.total_volume for item in history))[len(history) // 2] if len(history) == config.breakout_volume_lookback_bars else None
            for side in ("SHORT", "LONG"):
                setup = setups[side]; outside = bar.high >= profile.vah + profile.bucket_size * config.minimum_breakout_buckets if side == "SHORT" else bar.low <= profile.val - profile.bucket_size * config.minimum_breakout_buckets
                if setup["state"] == "IDLE" and outside and median is not None and bar.total_volume > config.breakout_volume_multiplier * median:
                    setup.update({"state": "STOP_RUN_CONFIRMED", "extreme": bar.high if side == "SHORT" else bar.low}); stop_runs += 1
                    events.append({"session_date": str(day), "timestamp": bar.end_utc.isoformat(), "side": side, "state": "STOP_RUN_CONFIRMED", "median_excludes_current": str(median), "volume": str(bar.total_volume)})
                if setup["state"] in {"STOP_RUN_CONFIRMED", "DIVERGENCE_CONFIRMED"}:
                    setup["extreme"] = max(setup["extreme"], bar.high) if side == "SHORT" else min(setup["extreme"], bar.low)
                    swing = _swings(day_bars, index, side=side)
                    if swing:
                        _, swing_bar = swing
                        outside_swing = swing_bar.high >= profile.vah if side == "SHORT" else swing_bar.low <= profile.val
                        if outside_swing and all(existing["timestamp"] != swing_bar.start_utc for existing in setup["swings"]):
                            setup["swings"].append({"timestamp": swing_bar.start_utc, "price": swing_bar.high if side == "SHORT" else swing_bar.low, "cvd": swing_bar.cumulative_volume_delta})
                    if len(setup["swings"]) >= 2 and setup["state"] == "STOP_RUN_CONFIRMED":
                        first, second = setup["swings"][-2:]
                        divergent = second["price"] > first["price"] and second["cvd"] < first["cvd"] if side == "SHORT" else second["price"] < first["price"] and second["cvd"] > first["cvd"]
                        if divergent:
                            setup["state"] = "DIVERGENCE_CONFIRMED"; divergences += 1
                            events.append({"session_date": str(day), "side": side, "state": "DIVERGENCE_CONFIRMED", "first": {key: str(value) for key, value in first.items()}, "second": {key: str(value) for key, value in second.items()}, "magnitude": str(second["cvd"] - first["cvd"])})
                return_inside = profile.val <= bar.close < profile.vah if side == "SHORT" else profile.val < bar.close <= profile.vah
                required = setup["state"] == "DIVERGENCE_CONFIRMED" if config.variant == "FULL" else setup["state"] in {"STOP_RUN_CONFIRMED", "DIVERGENCE_CONFIRMED"} if config.variant == "VALUE_AREA_STOP_RUN" else setup["state"] == "DIVERGENCE_CONFIRMED" if config.variant == "VALUE_AREA_CVD_DIVERGENCE" else True
                if return_inside and required and not traded and index + 1 < len(day_bars):
                    returns += 1; proposed += 1
                    next_bar = day_bars[index + 1]
                    if next_bar.session_date != day or next_bar.start_new_york.timetz().replace(tzinfo=None) >= config.forced_flat_time:
                        events.append({"session_date": str(day), "side": side, "state": "NO_EXECUTABLE_ENTRY"}); continue
                    quantity = _eligible_quantity(config)
                    stop = setup["extreme"] + profile.bucket_size * config.stop_buffer_buckets if side == "SHORT" else setup["extreme"] - profile.bucket_size * config.stop_buffer_buckets
                    target = profile.poc; entry = next_bar.open
                    valid = quantity is not None and ((target < entry < stop) if side == "SHORT" else (stop < entry < target))
                    if not valid:
                        events.append({"session_date": str(day), "side": side, "state": "INVALIDATED", "reason": "INVALID_TARGET_OR_STOP"}); continue
                    decision = evaluator.evaluate_backtest(timestamp=next_bar.start_utc, instrument=config.symbol, account_state=AccountState(account_id="value-area-trap", current_equity=float(equity)), market_state=market_state, proposed_action=ProposedAction(action=ActionType.ORDER_SUBMISSION, instrument=config.symbol, quantity=float(quantity), direction=side), policy=policy)
                    if not decision.allowed:
                        blocked.append({"session_date": str(day), "side": side, "decision_hash": decision.decision_hash, "classification": decision.classification.value, "required_actions": decision.required_actions}); continue
                    exit_bar = day_bars[-1]; reason = "SESSION_FORCE_FLAT"; exit_price = exit_bar.close; ambiguity = False
                    for candidate in day_bars[index + 1:]:
                        hit_stop = candidate.high >= stop if side == "SHORT" else candidate.low <= stop
                        hit_target = candidate.low <= target if side == "SHORT" else candidate.high >= target
                        if hit_stop and hit_target:
                            exit_bar, exit_price, reason, ambiguity = candidate, stop, "STOP_FIRST_AMBIGUITY", True; break
                        if hit_stop:
                            exit_bar, exit_price, reason = candidate, stop, "STOP"; break
                        if hit_target:
                            exit_bar, exit_price, reason = candidate, target, "TARGET"; break
                    sign = Decimal("1") if side == "LONG" else Decimal("-1")
                    entry_price = entry + sign * config.price_tick
                    exit_type = OrderType.STOP if reason.startswith("STOP") else OrderType.LIMIT if reason == "TARGET" else OrderType.MARKET
                    result = engine.calculate(config.symbol, float(quantity), order_types=(OrderType.MARKET, exit_type))
                    gross = sign * (exit_price - entry_price) * quantity; net = gross - Decimal(str(result.total_cost)); equity += net
                    trades.append({"trade_id": hashlib.sha256(f"{day}|{side}|{next_bar.start_utc}".encode()).hexdigest()[:16], "session_date": str(day), "direction": side, "signal_timestamp": bar.start_utc.isoformat(), "signal_bar_close_timestamp": bar.end_utc.isoformat(), "entry_timestamp": next_bar.start_utc.isoformat(), "exit_timestamp": exit_bar.end_utc.isoformat(), "entry_price": str(entry_price), "exit_price": str(exit_price), "initial_stop_price": str(stop), "target_price": str(target), "quantity": str(quantity), "notional": str(entry_price * quantity), "gross_pnl": str(gross), "entry_fees": str(Decimal(str(result.total_cost)) / 2), "exit_fees": str(Decimal(str(result.total_cost)) / 2), "fees": str(Decimal(str(result.commissions + result.exchange_fees + result.regulatory_fees))), "slippage_cost": str(result.slippage_cost), "net_pnl": str(net), "exit_reason": reason, "same_bar_ambiguity": ambiguity, "compliance_decision_hash": decision.decision_hash, "cost_model_hash": result.configuration_hash})
                    traded = True; force_flat += int(reason == "SESSION_FORCE_FLAT"); ambiguities += int(ambiguity); break
    gross = sum((Decimal(item["gross_pnl"]) for item in trades), Decimal()); fees = sum((Decimal(item["fees"]) for item in trades), Decimal()); slip = sum((Decimal(item["slippage_cost"]) for item in trades), Decimal()); net = sum((Decimal(item["net_pnl"]) for item in trades), Decimal())
    return ValueAreaTrapResult(proposed_setups=proposed, significant_stop_runs=stop_runs, confirmed_divergences=divergences, return_triggers=returns, compliance_blocks=blocked, trades=trades, setup_events=events, gross_pnl=gross, fees=fees, slippage_cost=slip, net_pnl=net, forced_flat_count=force_flat, same_bar_ambiguity_count=ambiguities, policy_hash=policy.policy_hash, cost_model_hash=costs.configuration_hash)
