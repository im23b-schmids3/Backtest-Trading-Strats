"""Fixed Alpha Futures Zero lifecycle replay for V12.

This module deliberately reuses the frozen V12/V7 strategy and Binance cache.
Only the Alpha account lifecycle, billing, payout, and shared-account layer is
reimplemented.  The previous V12 implementation and outputs are left intact.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fib_backtester.research import v12_binance_proxy_prop_simulation as legacy
from fib_backtester.research.v12_contract_registry import CONTRACTS, PROXY_SYMBOLS, PROXY_TO_CONTRACT, build_synthetic_context, mapped_price, round_to_tick


ROOT = Path("reports/v12_fixed")
TIMEFRAMES = ("1h", "4h")
POSITION_SIZES = legacy.POSITION_SIZES
PRIMARY_BILLING_SCENARIO = "B_REBILL_AFTER_BREACH"
BILLING_SCENARIOS = ("A_CANCEL_ON_BREACH", "B_REBILL_AFTER_BREACH", "C_EXPLICIT_EVALUATION_RESET")
OFFICIAL_VERIFICATION_DATE = "2026-07-15"


@dataclass(frozen=True)
class FixedAccountSpec:
    name: str
    account_size: float
    target: float
    mll_amount: float
    daily_loss_guard: float
    subscription: float
    evaluation_reset: float
    qualified_reset: float
    payout_max: float
    max_micros_evaluation: int
    max_micros_qualified_initial: int


ACCOUNT_SPECS = {
    "25K Zero": FixedAccountSpec("25K Zero", 25_000.0, 1_500.0, 1_000.0, 500.0, 79.0, 69.0, 399.0, 1_000.0, 10, 10),
    "50K Zero": FixedAccountSpec("50K Zero", 50_000.0, 3_000.0, 2_000.0, 1_000.0, 119.0, 109.0, 499.0, 1_500.0, 30, 10),
}


@dataclass(frozen=True)
class CanonicalProxy:
    market: str
    alpha_product: str
    multiplier: float
    tick_size: float
    proxy_type: str
    source_status: str

    @property
    def tick_value(self) -> float:
        return CONTRACTS[self.alpha_product].tick_value


# One proxy per Alpha exposure.  Alpha's current trading-hours documentation
# confirms MBT and MET are both available, so BTC maps to MBT rather than MET.
CANONICAL_PROXIES = {
    market: CanonicalProxy(market, product, CONTRACTS[product].multiplier, CONTRACTS[product].tick_size, f"Binance {PROXY_SYMBOLS[market]}", "canonical")
    for market, product in PROXY_TO_CONTRACT.items()
}

EXCLUDED_PROXY_REASONS = {
    "SPY": "SPYUSDT duplicates MES exposure already represented by the older SPXUSDT proxy.",
}

PORTFOLIO_BASE = {
    "Portfolio A - ETH only": ["ETH"],
    "Portfolio B - ETH + Gold": ["ETH", "Gold"],
    "Portfolio C - BTC + ETH + Gold": ["BTC", "ETH", "Gold"],
}

ALLOWED_TRANSITIONS = {
    ("EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"),
    ("EVALUATION_DAILY_LOCKED", "EVALUATION_ACTIVE"),
    ("EVALUATION_ACTIVE", "EVALUATION_PASSED"),
    ("EVALUATION_ACTIVE", "EVALUATION_FAILED"),
    ("EVALUATION_DAILY_LOCKED", "EVALUATION_PASSED"),
    ("EVALUATION_DAILY_LOCKED", "EVALUATION_FAILED"),
    ("EVALUATION_PASSED", "CLOSED"),
    ("EVALUATION_FAILED", "CLOSED"),
    ("QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"),
    ("QUALIFIED_DAILY_LOCKED", "QUALIFIED_ACTIVE"),
    ("QUALIFIED_ACTIVE", "QUALIFIED_PAYOUT_ELIGIBLE"),
    ("QUALIFIED_DAILY_LOCKED", "QUALIFIED_PAYOUT_ELIGIBLE"),
    ("QUALIFIED_PAYOUT_ELIGIBLE", "QUALIFIED_ACTIVE"),
    ("QUALIFIED_ACTIVE", "QUALIFIED_FAILED"),
    ("QUALIFIED_DAILY_LOCKED", "QUALIFIED_FAILED"),
    ("QUALIFIED_FAILED", "CLOSED"),
    ("QUALIFIED_ACTIVE", "CENSORED_END_OF_DATA"),
    ("QUALIFIED_DAILY_LOCKED", "CENSORED_END_OF_DATA"),
    ("EVALUATION_ACTIVE", "CENSORED_END_OF_DATA"),
    ("EVALUATION_DAILY_LOCKED", "CENSORED_END_OF_DATA"),
}


def _session(timestamp: pd.Timestamp) -> str:
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    return str((local - pd.Timedelta(days=1)).date() if local.hour < 18 else local.date())


def _next_session(session: str) -> str:
    return str(pd.Timestamp(session) + pd.Timedelta(days=1))[:10]


def _transition(transitions: list[dict], lifecycle_id: str, timestamp, before: str, after: str, reason: str) -> str:
    valid = (before, after) in ALLOWED_TRANSITIONS
    if not valid:
        raise ValueError(f"invalid Alpha state transition: {before} -> {after}")
    transitions.append({"lifecycle_id": lifecycle_id, "timestamp": str(timestamp), "state_before": before, "state_after": after, "reason": reason, "valid_transition": True})
    return after


def _rebill_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    result = []
    current = pd.Timestamp(start)
    while current <= end:
        result.append(current)
        current = current + pd.DateOffset(months=1)
    return result


def _billing_count(start: pd.Timestamp, stop: pd.Timestamp, rate: float) -> tuple[int, float]:
    if stop < start:
        return 0, 0.0
    count = len(_rebill_dates(start, stop))
    return count, count * rate


def _next_rebill(start: pd.Timestamp, failure: pd.Timestamp) -> pd.Timestamp:
    current = pd.Timestamp(start)
    while current <= failure:
        current = current + pd.DateOffset(months=1)
    return current


def _session_close_events(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    local_start = pd.Timestamp(start).tz_convert("America/New_York").normalize()
    local_end = pd.Timestamp(end).tz_convert("America/New_York").normalize()
    days = pd.date_range(local_start, local_end, freq="D", tz="America/New_York")
    return [day + pd.Timedelta(hours=16, minutes=20) for day in days]


def _next_session_close_after(timestamp: pd.Timestamp) -> pd.Timestamp:
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    candidate = pd.Timestamp(local.date(), tz="America/New_York") + pd.Timedelta(hours=16, minutes=20)
    return candidate if candidate > timestamp else candidate + pd.Timedelta(days=1)


def _allocate(total: int) -> list[int]:
    fractions = np.asarray((0.30, 0.25, 0.20, 0.15, 0.10)) * total
    allocation = np.floor(fractions).astype(int)
    for index in np.argsort(-(fractions - allocation))[: total - int(allocation.sum())]:
        allocation[index] += 1
    return allocation.tolist()


def _prepare_trade(raw: dict, market: str, size: int, conversion_context: dict | None = None) -> dict:
    """Convert unchanged V12 strategy fills to canonical proxy dollar legs."""
    spec = CANONICAL_PROXIES[market]
    fee_rate = 0.001 if market in {"BTC", "ETH"} else 0.0005
    entry = round_to_tick(mapped_price(float(raw["entry_price"]), market, conversion_context), spec.tick_size)
    allocation = _allocate(size)
    remaining = size
    legs = []
    for event in legacy._events(raw.get("exit_events")):
        reason = event.get("reason", "end_of_test")
        quantity = min(allocation[int(reason[2:]) - 1], remaining) if reason.startswith("tp") and reason[2:].isdigit() else remaining
        if quantity <= 0:
            continue
        proxy_price = float(event.get("fill_price", event.get("raw_price")))
        price = round_to_tick(mapped_price(proxy_price, market, conversion_context), spec.tick_size)
        direction = 1 if raw["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * quantity
        fee = abs(price * spec.multiplier * quantity) * fee_rate
        legs.append({"timestamp": str(event.get("timestamp", raw["exit_timestamp"])), "reason": reason, "price": price, "quantity": quantity, "gross": gross, "fee": fee, "net": gross - fee})
        remaining -= quantity
        if remaining <= 0:
            break
    if remaining > 0:
        price = round_to_tick(mapped_price(float(raw.get("average_exit_price", raw["entry_price"])), market, conversion_context), spec.tick_size)
        direction = 1 if raw["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * remaining
        fee = abs(price * spec.multiplier * remaining) * fee_rate
        legs.append({"timestamp": str(raw["exit_timestamp"]), "reason": raw.get("exit_reason", "end_of_test"), "price": price, "quantity": remaining, "gross": gross, "fee": fee, "net": gross - fee})
    entry_fee = abs(entry * spec.multiplier * size) * fee_rate
    return {"market": market, "alpha_product": spec.alpha_product, "conversion_mode": (conversion_context or {}).get("mode", "DIRECT_PRICE_LEVEL"), "proxy_entry": float(raw["entry_price"]), "setup_id": raw.get("setup_id", ""), "entry_timestamp": pd.Timestamp(raw["fill_timestamp"]), "exit_timestamp": pd.Timestamp(legs[-1]["timestamp"]), "side": raw["side"], "contracts": size, "entry": entry, "entry_fee": entry_fee, "legs": legs, "gross_pnl": sum(leg["gross"] for leg in legs), "net_pnl": sum(leg["net"] for leg in legs) - entry_fee, "fees": sum(leg["fee"] for leg in legs) + entry_fee, "slippage": 0.0, "holding_hours": (pd.Timestamp(legs[-1]["timestamp"]) - pd.Timestamp(raw["fill_timestamp"])).total_seconds() / 3600}


def _flatten_leg(trade: dict, price: float, timestamp, quantity: int, reason: str) -> dict:
    spec = CANONICAL_PROXIES[trade["market"]]
    direction = 1 if trade["side"] == "long" else -1
    price = round_to_tick(float(price), spec.tick_size)
    gross = direction * (price - trade["entry"]) * spec.multiplier * quantity
    fee_rate = 0.001 if trade["market"] in {"BTC", "ETH"} else 0.0005
    fee = abs(price * spec.multiplier * quantity) * fee_rate
    return {"timestamp": str(timestamp), "reason": reason, "price": price, "quantity": quantity, "gross": gross, "fee": fee, "net": gross - fee}


def _event_row(ctx: dict, timestamp, event_type: str, trade: dict | None, leg: dict | None, before: str, after: str, balance_before: float, balance_after: float, reason: str, subscription_charge: float = 0.0, payout_request: float = 0.0, trader_payout: float = 0.0) -> dict:
    return {"run_id": ctx["run_id"], "lifecycle_id": ctx["lifecycle_id"], "timestamp": str(timestamp), "lifecycle_stage": ctx["stage"], "account_state_before": before, "account_state_after": after, "event_type": event_type, "market": trade["market"] if trade else "", "timeframe": ctx["timeframe"], "contracts": int(leg["quantity"]) if leg else (int(trade["contracts"]) if trade else 0), "entry": trade["entry"] if trade else "", "exit": leg["price"] if leg else "", "gross_pnl": float(leg["gross"]) if leg else 0.0, "fees": float((trade["entry_fee"] if event_type == "ENTRY_FILLED" and trade else 0.0) + (leg["fee"] if leg else 0.0)), "slippage": 0.0, "net_pnl": float((-(trade["entry_fee"]) if event_type == "ENTRY_FILLED" and trade else 0.0) + (leg["net"] if leg else 0.0)), "balance_before": balance_before, "balance_after": balance_after, "daily_loss_guard_value": ctx["daily_profit"], "mll_threshold": ctx["mll"], "distance_to_target": max(ctx["target"] - (balance_after - ctx["account_size"]), 0.0), "distance_to_mll": balance_after - ctx["mll"], "winning_day_count": len(ctx["winning_days"]), "consistency_percentage": ctx["consistency"], "payout_cycle_profit": ctx["cycle_profit"], "subscription_charge": subscription_charge, "payout_request": payout_request, "trader_payout_after_split": trader_payout, "transition_reason": reason}


def _subscription_event(run_id: str, lifecycle_id: str, timestamp: pd.Timestamp, timeframe: str, balance: float, amount: float) -> dict:
    """Record an external Evaluation rebill without changing trading equity."""
    return {"run_id": run_id, "lifecycle_id": lifecycle_id, "timestamp": str(timestamp), "lifecycle_stage": "EVALUATION", "account_state_before": "EVALUATION_ACTIVE", "account_state_after": "EVALUATION_ACTIVE", "event_type": "SUBSCRIPTION_CHARGED", "market": "", "timeframe": timeframe, "contracts": 0, "entry": "", "exit": "", "gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "net_pnl": 0.0, "balance_before": balance, "balance_after": balance, "daily_loss_guard_value": 0.0, "mll_threshold": "", "distance_to_target": "", "distance_to_mll": "", "winning_day_count": 0, "consistency_percentage": 0.0, "payout_cycle_profit": 0.0, "subscription_charge": float(amount), "payout_request": 0.0, "trader_payout_after_split": 0.0, "transition_reason": "calendar-month Evaluation subscription rebill"}


def _append_subscription_events(event_rows: list[dict], run_id: str, result: dict, start: pd.Timestamp, stop: pd.Timestamp, timeframe: str, amount: float) -> None:
    for timestamp in _rebill_dates(start, stop):
        event_rows.append(_subscription_event(run_id, result["lifecycle_id"], timestamp, timeframe, float(result["starting_balance"]), amount))


def _run_stage(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, account: FixedAccountSpec, stage: str, lifecycle_id: str, run_id: str, initial_balance: float | None = None, collect_events: bool = True) -> dict:
    account_size = float(initial_balance if initial_balance is not None else account.account_size)
    target = account.target if stage == "EVALUATION" else 0.0
    balance = account_size
    mll = account_size - account.mll_amount
    state = "EVALUATION_ACTIVE" if stage == "EVALUATION" else "QUALIFIED_ACTIVE"
    transitions, events_log, active, event_list = [], [], {}, []
    for number, trade in enumerate(trades):
        if not (start <= trade["entry_timestamp"] <= end):
            continue
        event_list.append((trade["entry_timestamp"], 1, number, "entry", None, trade))
        for leg_index, leg in enumerate(trade["legs"]):
            timestamp = pd.Timestamp(leg["timestamp"])
            if start <= timestamp <= end:
                event_list.append((timestamp, 0, number, "exit", leg_index, trade))
    event_list.sort(key=lambda row: (row[0], row[1], row[2]))
    current_session = None
    daily_profit = 0.0
    winning_days: set[str] = set()
    cycle_days: dict[str, float] = {}
    cycle_profit = 0.0
    payouts = []
    gross_trading = 0.0
    net_trading = 0.0
    skipped = 0
    conflicts = 0
    daily_locks = 0
    pass_time = fail_time = None
    failure_reason = ""
    payout_months: dict[str, int] = {}

    ctx = {"run_id": run_id, "lifecycle_id": lifecycle_id, "stage": stage, "timeframe": "", "daily_profit": daily_profit, "mll": mll, "target": target, "account_size": account_size, "winning_days": winning_days, "consistency": 0.0, "cycle_profit": cycle_profit}

    def log(timestamp, event_type, trade=None, leg=None, before=state, after=state, reason="", subscription_charge=0.0, payout_request=0.0, trader_payout=0.0, balance_before=None, balance_after=None):
        ctx["daily_profit"] = daily_profit
        ctx["mll"] = mll
        ctx["consistency"] = (max(cycle_days.values()) / cycle_profit) if cycle_profit > 0 and cycle_days else 0.0
        ctx["cycle_profit"] = cycle_profit
        if collect_events:
            events_log.append(_event_row(ctx, timestamp, event_type, trade, leg, before, after, balance if balance_before is None else balance_before, balance if balance_after is None else balance_after, reason, subscription_charge, payout_request, trader_payout))

    def finish_day(session: str, timestamp):
        nonlocal state, daily_profit, mll, cycle_profit, balance
        if stage == "QUALIFIED" and daily_profit > 0:
            cycle_days[session] = daily_profit
        if stage == "QUALIFIED" and daily_profit >= legacy.WINNING_DAY_MINIMUM:
            winning_days.add(session)
        if stage == "QUALIFIED" and len(winning_days) >= legacy.WINNING_DAYS_REQUIRED and cycle_profit > 0:
            consistency = max(cycle_days.values(), default=0.0) / cycle_profit if cycle_profit else 0.0
            if consistency <= legacy.CONSISTENCY_LIMIT:
                state = _transition(transitions, lifecycle_id, timestamp, state, "QUALIFIED_PAYOUT_ELIGIBLE", "five winning days and 40% consistency satisfied")
                month_key = str(pd.Timestamp(timestamp).date())[:7]
                if payout_months.get(month_key, 0) < 4:
                    request = min(0.50 * cycle_profit, account.payout_max)
                    if request >= legacy.WINNING_DAY_MINIMUM and balance - request > mll:
                        payout_months[month_key] = payout_months.get(month_key, 0) + 1
                        balance_before = balance
                        balance -= request
                        payouts.append({"timestamp": str(timestamp), "gross": request, "received": request * legacy.PAYOUT_SPLIT})
                        log(timestamp, "PAYOUT_REQUEST_FILLED", None, None, "QUALIFIED_PAYOUT_ELIGIBLE", "QUALIFIED_ACTIVE", "payout cycle completed", payout_request=request, trader_payout=request * legacy.PAYOUT_SPLIT, balance_before=balance_before, balance_after=balance)
                        state = _transition(transitions, lifecycle_id, timestamp, "QUALIFIED_PAYOUT_ELIGIBLE", "QUALIFIED_ACTIVE", "payout filled and counters reset")
                        cycle_profit = 0.0
                        winning_days.clear()
                        cycle_days.clear()
        mll = min(account_size, max(mll, balance - account.mll_amount))
        daily_profit = 0.0
        if state.endswith("DAILY_LOCKED"):
            after = "EVALUATION_ACTIVE" if stage == "EVALUATION" else "QUALIFIED_ACTIVE"
            state = _transition(transitions, lifecycle_id, timestamp, state, after, "next official trading day opened")

    def force_session_close(timestamp):
        nonlocal balance, gross_trading, net_trading, daily_profit, cycle_profit
        for other_number, other in list(active.items()):
            flatten = _flatten_leg(other["trade"], other["last_price"], timestamp, other["remaining"], "session_forced_liquidation")
            balance_before = balance
            balance += flatten["net"]
            gross_trading += flatten["gross"]
            net_trading += flatten["net"]
            daily_profit += flatten["net"]
            if stage == "QUALIFIED":
                cycle_profit += flatten["net"]
            log(timestamp, "SESSION_FORCED_LIQUIDATION", other["trade"], flatten, state, state, "all sim-account positions closed before 4:20PM ET", balance_before=balance_before, balance_after=balance)
            active.pop(other_number, None)

    last_event_timestamp = start
    for timestamp, kind, number, event_type, leg_index, trade in event_list:
        if state in {"EVALUATION_PASSED", "EVALUATION_FAILED", "QUALIFIED_FAILED", "CLOSED", "CENSORED_END_OF_DATA"}:
            break
        if active:
            cutoff = _next_session_close_after(last_event_timestamp)
            if cutoff < timestamp:
                force_session_close(cutoff)
        # Advance the event cursor for every processed event, including entries
        # that take the early ``continue`` branches.  Without this, an open
        # position entered after the first event could be force-closed against
        # the first historical session cutoff repeatedly.
        last_event_timestamp = timestamp
        session = _session(timestamp)
        if current_session is not None and session != current_session:
            finish_day(current_session, timestamp)
        current_session = session
        if kind == 1:
            if state.endswith("DAILY_LOCKED"):
                skipped += 1
                log(timestamp, "ORDER_SKIPPED_DAILY_LOCK", trade, None, state, state, "Daily Loss Guard lock")
                continue
            local_entry = pd.Timestamp(timestamp).tz_convert("America/New_York")
            if local_entry.hour == 17 or (local_entry.hour == 16 and local_entry.minute >= 20):
                skipped += 1
                log(timestamp, "ORDER_SKIPPED_SESSION_CLOSE", trade, None, state, state, "new entries are not permitted at or after 4:20PM ET")
                continue
            max_micros = account.max_micros_evaluation if stage == "EVALUATION" else account.max_micros_qualified_initial
            current_contracts = sum(item["remaining"] for item in active.values())
            same_market = any(item["trade"]["market"] == trade["market"] for item in active.values())
            if same_market or current_contracts + trade["contracts"] > max_micros:
                conflicts += 1
                skipped += 1
                log(timestamp, "ORDER_SKIPPED_POSITION_CONFLICT", trade, None, state, state, "shared account position limit or same-market active trade")
                continue
            balance_before = balance
            balance -= trade["entry_fee"]
            daily_profit -= trade["entry_fee"]
            net_trading -= trade["entry_fee"]
            active[number] = {"trade": trade, "remaining": trade["contracts"], "last_price": trade["entry"]}
            log(timestamp, "ENTRY_FILLED", trade, None, state, state, "strategy entry accepted by shared account", balance_before=balance_before, balance_after=balance)
            if balance <= mll:
                fail_time, failure_reason = timestamp, "Maximum Loss Limit"
                state = _transition(transitions, lifecycle_id, timestamp, state, "EVALUATION_FAILED" if stage == "EVALUATION" else "QUALIFIED_FAILED", failure_reason)
            continue
        if number not in active:
            continue
        leg = trade["legs"][leg_index]
        active[number]["last_price"] = leg["price"]
        balance_before = balance
        balance += leg["net"]
        gross_trading += leg["gross"]
        net_trading += leg["net"]
        daily_profit += leg["net"]
        active[number]["remaining"] -= leg["quantity"]
        if stage == "QUALIFIED":
            cycle_profit += leg["net"]
        log(timestamp, "EXIT_FILLED", trade, leg, state, state, leg["reason"], balance_before=balance_before, balance_after=balance)
        if daily_profit <= -account.daily_loss_guard:
            daily_locks += 1
            prior = state
            state = _transition(transitions, lifecycle_id, timestamp, state, "EVALUATION_DAILY_LOCKED" if stage == "EVALUATION" else "QUALIFIED_DAILY_LOCKED", "Daily Loss Guard reached; flatten and cancel pending orders")
            for other_number, other in list(active.items()):
                remaining = other["remaining"]
                if other_number == number:
                    remaining -= leg["quantity"]
                if remaining > 0:
                    flatten = _flatten_leg(other["trade"], other["last_price"], timestamp, remaining, "daily_loss_guard_flatten")
                    balance_before = balance
                    balance += flatten["net"]
                    gross_trading += flatten["gross"]
                    net_trading += flatten["net"]
                    daily_profit += flatten["net"]
                    log(timestamp, "DLG_FORCED_FLATTEN", other["trade"], flatten, state, state, "Daily Loss Guard liquidation", balance_before=balance_before, balance_after=balance)
                active.pop(other_number, None)
        if balance <= mll:
            fail_time, failure_reason = timestamp, "Maximum Loss Limit"
            prior = state
            state = _transition(transitions, lifecycle_id, timestamp, state, "EVALUATION_FAILED" if stage == "EVALUATION" else "QUALIFIED_FAILED", failure_reason)
            log(timestamp, "ACCOUNT_BREACH", None, None, prior, state, failure_reason, balance_before=balance, balance_after=balance)
        if stage == "EVALUATION" and balance >= account.account_size + account.target:
            pass_time = timestamp
            prior = state
            state = _transition(transitions, lifecycle_id, timestamp, state, "EVALUATION_PASSED", "official Zero Evaluation profit target reached")
            log(timestamp, "EVALUATION_PASSED", None, None, prior, state, "Evaluation closed at profit target", balance_before=balance, balance_after=balance)
            for other_number, other in list(active.items()):
                remaining = other["remaining"] - (leg["quantity"] if other_number == number else 0)
                if remaining > 0:
                    flatten = _flatten_leg(other["trade"], other["last_price"], timestamp, remaining, "evaluation_pass_flatten")
                    balance_before = balance
                    balance += flatten["net"]
                    gross_trading += flatten["gross"]
                    net_trading += flatten["net"]
                    log(timestamp, "EVALUATION_ORDER_CANCELLED_AND_FLATTENED", other["trade"], flatten, state, state, "all pending Evaluation orders cancelled at pass", balance_before=balance_before, balance_after=balance)
                active.pop(other_number, None)
            state = _transition(transitions, lifecycle_id, timestamp, state, "CLOSED", "Evaluation lifecycle ended after pass")
            break
        if leg_index == len(trade["legs"]) - 1:
            active.pop(number, None)
        last_event_timestamp = timestamp
    if active and state not in {"CLOSED", "EVALUATION_FAILED", "QUALIFIED_FAILED"}:
        cutoff = _next_session_close_after(last_event_timestamp)
        if cutoff <= end:
            force_session_close(cutoff)
    if current_session is not None and state not in {"CLOSED", "EVALUATION_FAILED", "QUALIFIED_FAILED"}:
        finish_day(current_session, end)
    if fail_time is not None:
        terminal = fail_time
        if state != "CLOSED":
            state = _transition(transitions, lifecycle_id, fail_time, state, "CLOSED", "account lifecycle ended after breach")
    elif pass_time is not None:
        terminal = pass_time
    else:
        terminal = end
        if state not in {"CLOSED", "EVALUATION_FAILED", "QUALIFIED_FAILED"}:
            state = _transition(transitions, lifecycle_id, end, state, "CENSORED_END_OF_DATA", "historical data ended before terminal outcome")
    failed = fail_time is not None
    passed = pass_time is not None
    status = "PASSED" if passed else ("FAILED" if failed else "CENSORED_END_OF_DATA")
    return {"lifecycle_id": lifecycle_id, "stage": stage, "status": status, "passed": passed, "failed": failed, "censored": status == "CENSORED_END_OF_DATA", "start": str(start), "end": str(terminal), "history_end": str(end), "pass_timestamp": str(pass_time) if pass_time else "", "failure_timestamp": str(fail_time) if fail_time else "", "failure_reason": failure_reason, "lifetime_days": (terminal - start).total_seconds() / 86400, "gross_trading_pnl": gross_trading, "net_trading_pnl": net_trading, "fees": sum(float(t["entry_fee"]) for t in trades if start <= t["entry_timestamp"] <= terminal) + max(gross_trading - net_trading - sum(float(t["entry_fee"]) for t in trades if start <= t["entry_timestamp"] <= terminal), 0.0), "slippage": 0.0, "withdrawal_requested": sum(item["gross"] for item in payouts), "trader_payout": sum(item["received"] for item in payouts), "payout_count": len(payouts), "first_payout_timestamp": payouts[0]["timestamp"] if payouts else "", "second_payout_timestamp": payouts[1]["timestamp"] if len(payouts) > 1 else "", "third_payout_timestamp": payouts[2]["timestamp"] if len(payouts) > 2 else "", "ending_balance": balance, "account_size": account.account_size, "starting_balance": account_size, "mll_threshold": mll, "daily_lock_count": daily_locks, "position_conflicts": conflicts, "skipped_trades": skipped, "winning_day_count": len(winning_days), "consistency_percentage": max(cycle_days.values()) / cycle_profit if cycle_profit > 0 and cycle_days else 0.0, "transitions": transitions, "events": events_log, "active_open_positions_at_end": len(active)}


def _portfolio_members(validation: pd.DataFrame, streams: dict) -> dict[str, list[str]]:
    members = dict(PORTFOLIO_BASE)
    canonical = sorted(set(CANONICAL_PROXIES).intersection({market for market, _ in streams}))
    members["Portfolio D - All canonical Alpha exposures"] = canonical
    members["Portfolio E - Profitable validated canonical markets"] = {
        timeframe: [market for market in canonical if not validation[(validation.market == market) & (validation.timeframe == timeframe) & (validation.stage == "validation") & (validation.history_class == "LONG_HISTORY")].empty and float(validation[(validation.market == market) & (validation.timeframe == timeframe) & (validation.stage == "validation") & (validation.history_class == "LONG_HISTORY")].net_return.iloc[0]) > 0]
        for timeframe in TIMEFRAMES
    }
    return members


def _members_for(portfolios: dict, portfolio: str, timeframe: str) -> list[str]:
    value = portfolios[portfolio]
    return value.get(timeframe, []) if isinstance(value, dict) else value


def _run_path(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, account: FixedAccountSpec, scenario: str, portfolio: str, timeframe: str, position_size: int, run_id: str, collect_events: bool = False) -> dict:
    cursor = start
    evaluation_rows, qualified_rows, transitions, event_rows = [], [], [], []
    total_eval_subscription = 0.0
    total_eval_reset = 0.0
    replacement_evaluations = 0
    safety = 0
    final_result = None
    while cursor < end and safety < 100:
        safety += 1
        eval_id = f"{run_id}:evaluation:{safety}"
        result = _run_stage(trades, cursor, end, account, "EVALUATION", eval_id, run_id, collect_events=collect_events)
        for row in result["transitions"]:
            transitions.append(row)
        if collect_events:
            for event in result["events"]:
                event["timeframe"] = timeframe
            event_rows.extend(result["events"])
        pass_time = pd.Timestamp(result["pass_timestamp"]) if result["pass_timestamp"] else None
        failure_time = pd.Timestamp(result["failure_timestamp"]) if result["failure_timestamp"] else None
        if pass_time:
            _, sub_cost = _billing_count(cursor, pass_time, account.subscription)
            result["evaluation_subscription_cost"] = sub_cost
            result["evaluation_reset_cost"] = 0.0
            result["replacement_evaluation_cost"] = 0.0
            total_eval_subscription += sub_cost
            evaluation_rows.append(result)
            if collect_events:
                _append_subscription_events(event_rows, run_id, result, cursor, pass_time, timeframe, account.subscription)
            qualified_id = f"{run_id}:qualified:{safety}"
            q_start = pass_time + pd.Timedelta(nanoseconds=1)
            q = _run_stage(trades, q_start, end, account, "QUALIFIED", qualified_id, run_id, initial_balance=result["ending_balance"], collect_events=collect_events)
            for row in q["transitions"]:
                transitions.append(row)
            if collect_events:
                for event in q["events"]:
                    event["timeframe"] = timeframe
                event_rows.extend(q["events"])
            q["evaluation_id"] = eval_id
            q["qualified_id"] = qualified_id
            q["qualified_subscription_cost"] = 0.0
            q["qualified_reset_cost"] = 0.0
            q["activation_fee"] = 0.0
            qualified_rows.append(q)
            final_result = {"evaluations": evaluation_rows, "qualified": qualified_rows, "transitions": transitions, "events": event_rows, "replacement_evaluations": replacement_evaluations, "scenario": scenario}
            break
        if failure_time:
            if scenario == "A_CANCEL_ON_BREACH":
                _, sub_cost = _billing_count(cursor, failure_time, account.subscription)
                reset_cost = 0.0
                next_cursor = None
            elif scenario == "B_REBILL_AFTER_BREACH":
                next_cursor = _next_rebill(cursor, failure_time)
                # The rebill timestamp starts the replacement Evaluation.  It
                # must not be charged once to the failed Evaluation and again
                # to the fresh Evaluation.
                _, sub_cost = _billing_count(cursor, failure_time, account.subscription)
                reset_cost = 0.0
            else:
                next_cursor = failure_time + pd.Timedelta(nanoseconds=1)
                _, sub_cost = _billing_count(cursor, failure_time, account.subscription)
                reset_cost = account.evaluation_reset
            result["evaluation_subscription_cost"] = sub_cost
            result["evaluation_reset_cost"] = reset_cost
            result["replacement_evaluation_cost"] = 0.0
            total_eval_subscription += sub_cost
            total_eval_reset += reset_cost
            evaluation_rows.append(result)
            if collect_events:
                _append_subscription_events(event_rows, run_id, result, cursor, failure_time, timeframe, account.subscription)
            if next_cursor is None or next_cursor >= end:
                final_result = {"evaluations": evaluation_rows, "qualified": qualified_rows, "transitions": transitions, "events": event_rows, "replacement_evaluations": replacement_evaluations, "scenario": scenario}
                break
            replacement_evaluations += 1
            cursor = next_cursor
            continue
        result["evaluation_subscription_cost"] = _billing_count(cursor, end, account.subscription)[1]
        result["evaluation_reset_cost"] = 0.0
        result["replacement_evaluation_cost"] = 0.0
        total_eval_subscription += result["evaluation_subscription_cost"]
        evaluation_rows.append(result)
        if collect_events:
            _append_subscription_events(event_rows, run_id, result, cursor, end, timeframe, account.subscription)
        final_result = {"evaluations": evaluation_rows, "qualified": qualified_rows, "transitions": transitions, "events": event_rows, "replacement_evaluations": replacement_evaluations, "scenario": scenario}
        break
    if final_result is None:
        final_result = {"evaluations": evaluation_rows, "qualified": qualified_rows, "transitions": transitions, "events": event_rows, "replacement_evaluations": replacement_evaluations, "scenario": scenario}
    if collect_events:
        event_rows.sort(key=lambda event: (pd.Timestamp(event["timestamp"]), event["lifecycle_id"], event["event_type"]))
        final_result["events"] = event_rows
    final_result.update({"portfolio": portfolio, "timeframe": timeframe, "position_size": position_size, "account": account.name, "run_id": run_id, "evaluation_subscription_cost_total": total_eval_subscription, "evaluation_reset_cost_total": total_eval_reset, "qualified_reset_cost_total": 0.0, "activation_fee_total": 0.0, "net_trader_cashflow": sum(q["trader_payout"] for q in qualified_rows) - total_eval_subscription - total_eval_reset})
    return final_result


def _flatten_stage_rows(paths: list[dict]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for result in path["evaluations"]:
            rows.append(_lifecycle_row(path, result, "EVALUATION"))
        for result in path["qualified"]:
            rows.append(_lifecycle_row(path, result, "QUALIFIED"))
        if path["qualified"]:
            q = path["qualified"][-1]
            rows.append({"run_id": path["run_id"], "lifecycle_id": path["run_id"], "lifecycle_stage": "FULL_LIFECYCLE", "portfolio": path["portfolio"], "timeframe": path["timeframe"], "account": path["account"], "position_size": path["position_size"], "status": q["status"], "censored": q["censored"], "start": path["evaluations"][0]["start"] if path["evaluations"] else "", "end": q["end"], "lifetime_days": (pd.Timestamp(q["end"]) - pd.Timestamp(path["evaluations"][0]["start"])).total_seconds() / 86400 if path["evaluations"] else 0.0, "gross_trading_pnl": sum(r["gross_trading_pnl"] for r in path["evaluations"] + path["qualified"]), "net_trading_pnl": sum(r["net_trading_pnl"] for r in path["evaluations"] + path["qualified"]), "gross_withdrawal_requested": sum(r["withdrawal_requested"] for r in path["qualified"]), "trader_payout_after_split": sum(r["trader_payout"] for r in path["qualified"]), "evaluation_subscription_cost": path["evaluation_subscription_cost_total"], "evaluation_reset_cost": path["evaluation_reset_cost_total"], "qualified_reset_cost": 0.0, "activation_fee": 0.0, "replacement_evaluation_cost": 0.0, "net_trader_cashflow": path["net_trader_cashflow"], "ending_balance": q["ending_balance"], "passed": True, "failed": q["failed"], "qualified_failed": q["failed"], "evaluation_count": len(path["evaluations"]), "qualified_count": len(path["qualified"]), "payout_count": sum(r["payout_count"] for r in path["qualified"]), "position_conflicts": sum(r["position_conflicts"] for r in path["evaluations"] + path["qualified"]), "skipped_trades": sum(r["skipped_trades"] for r in path["evaluations"] + path["qualified"]), "confidence": "PROXY-BASED EXPLORATORY RESEARCH - Binance proxy and lifecycle assumptions"})
    return pd.DataFrame(rows)


def _lifecycle_row(path: dict, result: dict, stage: str) -> dict:
    return {"run_id": path["run_id"], "lifecycle_id": result["lifecycle_id"], "lifecycle_stage": stage, "portfolio": path["portfolio"], "timeframe": path["timeframe"], "account": path["account"], "position_size": path["position_size"], "status": result["status"], "censored": result["censored"], "start": result["start"], "end": result["end"], "history_end": result["history_end"], "lifetime_days": result["lifetime_days"], "pass_timestamp": result["pass_timestamp"], "failure_timestamp": result["failure_timestamp"], "failure_reason": result["failure_reason"], "gross_trading_pnl": result["gross_trading_pnl"], "net_trading_pnl": result["net_trading_pnl"], "gross_withdrawal_requested": result["withdrawal_requested"], "trader_payout_after_split": result["trader_payout"], "evaluation_subscription_cost": result.get("evaluation_subscription_cost", 0.0), "evaluation_reset_cost": result.get("evaluation_reset_cost", 0.0), "qualified_reset_cost": result.get("qualified_reset_cost", 0.0), "activation_fee": result.get("activation_fee", 0.0), "replacement_evaluation_cost": result.get("replacement_evaluation_cost", 0.0), "net_trader_cashflow": result["trader_payout"] - result.get("evaluation_subscription_cost", 0.0) - result.get("evaluation_reset_cost", 0.0), "starting_balance": result["starting_balance"], "ending_balance": result["ending_balance"], "payout_count": result["payout_count"], "first_payout_timestamp": result["first_payout_timestamp"], "second_payout_timestamp": result["second_payout_timestamp"], "third_payout_timestamp": result["third_payout_timestamp"], "daily_lock_count": result["daily_lock_count"], "position_conflicts": result["position_conflicts"], "skipped_trades": result["skipped_trades"], "winning_day_count": result["winning_day_count"], "consistency_percentage": result["consistency_percentage"], "passed": result["passed"], "failed": result["failed"], "qualified_failed": result["failed"] if stage == "QUALIFIED" else False, "confidence": "PROXY-BASED EXPLORATORY RESEARCH - Binance proxy and lifecycle assumptions"}


def _summary_rows(paths: list[dict]) -> pd.DataFrame:
    rows = []
    for key, group in pd.DataFrame([{"portfolio": p["portfolio"], "timeframe": p["timeframe"], "account": p["account"], "position_size": p["position_size"], "path": p} for p in paths]).groupby(["portfolio", "timeframe", "account", "position_size"], sort=False):
        path_rows = [row.path for row in group.itertuples()]
        evals = [e for p in path_rows for e in p["evaluations"]]
        quals = [q for p in path_rows for q in p["qualified"]]
        rows.append({"portfolio": key[0], "timeframe": key[1], "account": key[2], "position_size": key[3], "evaluations": len(evals), "evaluation_passed": sum(e["passed"] for e in evals), "evaluation_failed": sum(e["failed"] for e in evals), "evaluation_censored": sum(e["censored"] for e in evals), "qualified_accounts": len(quals), "qualified_failures": sum(q["failed"] for q in quals), "qualified_censored": sum(q["censored"] for q in quals), "first_payout_accounts": sum(q["payout_count"] >= 1 for q in quals), "second_payout_accounts": sum(q["payout_count"] >= 2 for q in quals), "third_payout_accounts": sum(q["payout_count"] >= 3 for q in quals), "pass_rate": sum(e["passed"] for e in evals) / len(evals) if evals else 0.0, "first_payout_rate_among_qualified": sum(q["payout_count"] >= 1 for q in quals) / len(quals) if quals else 0.0, "median_days_to_pass": np.median([(pd.Timestamp(e["pass_timestamp"]) - pd.Timestamp(e["start"])).total_seconds() / 86400 for e in evals if e["pass_timestamp"]]) if any(e["pass_timestamp"] for e in evals) else np.nan, "median_days_to_first_payout": np.median([(pd.Timestamp(q["first_payout_timestamp"]) - pd.Timestamp(q["start"])).total_seconds() / 86400 for q in quals if q["first_payout_timestamp"]]) if any(q["first_payout_timestamp"] for q in quals) else np.nan, "median_uncensored_evaluation_lifetime": np.median([e["lifetime_days"] for e in evals if not e["censored"]]) if any(not e["censored"] for e in evals) else np.nan, "median_uncensored_qualified_lifetime": np.median([q["lifetime_days"] for q in quals if not q["censored"]]) if any(not q["censored"] for q in quals) else np.nan, "evaluation_subscription_cost": sum(p["evaluation_subscription_cost_total"] for p in path_rows), "evaluation_reset_cost": sum(p["evaluation_reset_cost_total"] for p in path_rows), "qualified_reset_cost": 0.0, "gross_withdrawal_requested": sum(q["withdrawal_requested"] for q in quals), "trader_payout_after_split": sum(q["trader_payout"] for q in quals), "net_trader_cashflow": sum(p["net_trader_cashflow"] for p in path_rows), "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _market_metrics(market: str, timeframe: str, bars: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    return legacy._market_metrics(market, timeframe, bars, trades, equity) | {"alpha_exposure_status": "descriptive_only" if market not in CANONICAL_PROXIES else "canonical_shared_account", "proxy_selection_reason": EXCLUDED_PROXY_REASONS.get(market, "one proxy selected per Alpha exposure")}


def _shared_portfolio_rows(paths: list[dict], portfolios: dict[str, list[str]], streams: dict) -> pd.DataFrame:
    rows = []
    for (portfolio, timeframe, account, position), group in pd.DataFrame([{"portfolio": p["portfolio"], "timeframe": p["timeframe"], "account": p["account"], "position": p["position_size"], "path": p} for p in paths]).groupby(["portfolio", "timeframe", "account", "position"], sort=False):
        ps = [row.path for row in group.itertuples()]
        evals = [e for p in ps for e in p["evaluations"]]
        quals = [q for p in ps for q in p["qualified"]]
        rows.append({"record_type": "shared_account_summary", "portfolio": portfolio, "timeframe": timeframe, "markets": ",".join(_members_for(portfolios, portfolio, timeframe)), "account": account, "position_size": position, "evaluation_instances": len(evals), "qualified_instances": len(quals), "evaluation_pass_rate": sum(e["passed"] for e in evals) / len(evals) if evals else 0.0, "evaluation_failure_rate": sum(e["failed"] for e in evals) / len(evals) if evals else 0.0, "evaluation_censor_rate": sum(e["censored"] for e in evals) / len(evals) if evals else 0.0, "qualified_failure_rate": sum(q["failed"] for q in quals) / len(quals) if quals else 0.0, "first_payout_rate": sum(q["payout_count"] >= 1 for q in quals) / len(quals) if quals else 0.0, "second_payout_rate": sum(q["payout_count"] >= 2 for q in quals) / len(quals) if quals else 0.0, "third_payout_rate": sum(q["payout_count"] >= 3 for q in quals) / len(quals) if quals else 0.0, "gross_trading_pnl": sum(e["gross_trading_pnl"] for e in evals + quals), "net_trading_pnl": sum(e["net_trading_pnl"] for e in evals + quals), "gross_withdrawal_requested": sum(q["withdrawal_requested"] for q in quals), "trader_payout_after_split": sum(q["trader_payout"] for q in quals), "evaluation_subscription_cost": sum(p["evaluation_subscription_cost_total"] for p in ps), "evaluation_reset_cost": sum(p["evaluation_reset_cost_total"] for p in ps), "qualified_reset_cost": 0.0, "net_trader_cashflow": sum(p["net_trader_cashflow"] for p in ps), "position_conflicts": sum(e["position_conflicts"] for p in ps for e in p["evaluations"] + p["qualified"]), "skipped_trades": sum(e["skipped_trades"] for p in ps for e in p["evaluations"] + p["qualified"]), "aggregation_model": "one shared account; chronological merged trades; shared balance/DLG/MLL/position limit", "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _payout_economics(paths: list[dict]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for scenario in BILLING_SCENARIOS:
            evals = path["evaluations"]
            quals = path["qualified"]
            eval_subscription = 0.0
            eval_reset = 0.0
            for result in evals:
                start = pd.Timestamp(result["start"])
                stop = pd.Timestamp(result["pass_timestamp"] or result["failure_timestamp"] or result["history_end"])
                if scenario == "B_REBILL_AFTER_BREACH" and result["failure_timestamp"]:
                    stop = min(_next_rebill(start, pd.Timestamp(result["failure_timestamp"])), pd.Timestamp(result["history_end"]))
                eval_subscription += _billing_count(start, stop, ACCOUNT_SPECS[path["account"]].subscription)[1]
                if scenario == "C_EXPLICIT_EVALUATION_RESET" and result["failure_timestamp"]:
                    eval_reset += ACCOUNT_SPECS[path["account"]].evaluation_reset
            payout = sum(r["trader_payout"] for r in quals)
            rows.append({"run_id": path["run_id"], "portfolio": path["portfolio"], "timeframe": path["timeframe"], "account": path["account"], "position_size": path["position_size"], "billing_scenario": scenario, "scenario_scope": "primary replay" if scenario == PRIMARY_BILLING_SCENARIO else "cost-only repricing; trading path unchanged", "gross_trading_pnl": sum(r["gross_trading_pnl"] for r in evals + quals), "net_trading_pnl": sum(r["net_trading_pnl"] for r in evals + quals), "gross_withdrawal_requested": sum(r["withdrawal_requested"] for r in quals), "trader_payout_after_split": payout, "evaluation_subscription_cost": eval_subscription, "evaluation_reset_cost": eval_reset, "qualified_reset_cost": 0.0, "activation_fee": 0.0, "replacement_evaluation_cost": 0.0, "net_trader_cashflow": payout - eval_subscription - eval_reset, "equation": "trader payout - evaluation subscriptions - explicit resets - qualified reset - activation - replacement evaluation", "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _censoring_summary(lifetimes: pd.DataFrame) -> pd.DataFrame:
    if lifetimes.empty:
        return lifetimes
    rows = []
    for key, group in lifetimes[lifetimes.lifecycle_stage.isin(["EVALUATION", "QUALIFIED"])].groupby(["lifecycle_stage", "account", "position_size"], sort=False):
        stage, account, position = key
        uncensored = pd.to_numeric(group.loc[group.censored == False, "lifetime_days"], errors="coerce").dropna()
        rows.append({"lifecycle_stage": stage, "account": account, "position_size": position, "total_instances": len(group), "passed_evaluations": int((group.status == "PASSED").sum()) if stage == "EVALUATION" else 0, "failed_evaluations": int((group.status == "FAILED").sum()) if stage == "EVALUATION" else 0, "censored_evaluations": int((group.status == "CENSORED_END_OF_DATA").sum()) if stage == "EVALUATION" else 0, "qualified_failures": int((group.status == "FAILED").sum()) if stage == "QUALIFIED" else 0, "qualified_first_payouts": int((pd.to_numeric(group.payout_count, errors="coerce") >= 1).sum()) if stage == "QUALIFIED" else 0, "qualified_second_payouts": int((pd.to_numeric(group.payout_count, errors="coerce") >= 2).sum()) if stage == "QUALIFIED" else 0, "qualified_third_payouts": int((pd.to_numeric(group.payout_count, errors="coerce") >= 3).sum()) if stage == "QUALIFIED" else 0, "censored_instances": int(group.censored.sum()), "median_days_to_pass": np.median([(pd.Timestamp(x) - pd.Timestamp(s)).total_seconds() / 86400 for x, s in zip(group.pass_timestamp[group.pass_timestamp != ""], group.start[group.pass_timestamp != ""])]) if stage == "EVALUATION" and (group.pass_timestamp != "").any() else np.nan, "median_days_to_first_payout": np.median([(pd.Timestamp(x) - pd.Timestamp(s)).total_seconds() / 86400 for x, s in zip(group.first_payout_timestamp[group.first_payout_timestamp != ""], group.start[group.first_payout_timestamp != ""])]) if stage == "QUALIFIED" and (group.first_payout_timestamp != "").any() else np.nan, "median_uncensored_lifetime_days": float(uncensored.median()) if not uncensored.empty else np.nan, "mean_uncensored_lifetime_days": float(uncensored.mean()) if not uncensored.empty else np.nan, "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _confidence_rows(streams: dict, portfolios: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {"severity": "HIGH", "category": "proxy_data", "finding": "Binance assets are exploratory proxies, not native CME futures; no native contract equivalence is claimed."},
        {"severity": "HIGH", "category": "unrealized_pnl", "finding": "The frozen V12 trade objects do not retain complete per-candle account marks; DLG/MLL floating-equity checks use realized strategy legs plus available last prices, so exact intrabar liquidation cannot be guaranteed."},
        {"severity": "MEDIUM", "category": "qualified_start_balance", "finding": "Qualified lifecycle starts from the evaluation ending balance in a separate ledger because the current public Zero documentation does not state a separate reset balance at activation."},
        {"severity": "MEDIUM", "category": "news", "finding": "No historical Forex Factory high-impact news dataset was retained; Zero Qualified news execution restrictions are not independently screened."},
        {"severity": "MEDIUM", "category": "billing", "finding": "Scenario B models official automatic rebill after breach as a fresh Evaluation; A and C are explicit sensitivity scenarios, not optimization."},
        {"severity": "INFO", "category": "duplicate_exposure", "finding": "SPYUSDT excluded because SPXUSDT already represents MES; BTCUSDT maps to MBT and ETHUSDT maps to MET."},
        {"severity": "HIGH", "category": "invalidated_outputs", "finding": "Previous V12 Prop and economics outputs are invalidated for lifecycle/economics decisions by this repair; the old files are preserved and not overwritten."},
    ]
    return pd.DataFrame(rows)


def _representative_logs(paths: list[dict]) -> pd.DataFrame:
    wanted = [("25K Zero", "pass_eval"), ("25K Zero", "fail_eval"), ("25K Zero", "qualified_payout"), ("50K Zero", "pass_eval"), ("50K Zero", "fail_eval"), ("50K Zero", "qualified_payout")]
    selected = {}
    for path in paths:
        account = path["account"]
        evals = path["evaluations"]
        quals = path["qualified"]
        if (account, "pass_eval") not in selected and any(e["passed"] for e in evals):
            selected[(account, "pass_eval")] = path
        if (account, "fail_eval") not in selected and any(e["failed"] for e in evals):
            selected[(account, "fail_eval")] = path
        if (account, "qualified_payout") not in selected and any(q["payout_count"] >= 1 for q in quals):
            selected[(account, "qualified_payout")] = path
    rows = []
    for account, label in wanted:
        path = selected.get((account, label))
        if path is None:
            rows.append({"run_id": "MISSING", "representative_type": label, "account": account, "event_type": "REPRESENTATIVE_NOT_FOUND", "note": "No qualifying path in cached data for this requested representative."})
            continue
        for event in path["events"]:
            event = dict(event)
            event["representative_type"] = label
            event["account"] = account
            rows.append(event)
    return pd.DataFrame(rows)


def _rules_markdown() -> str:
    return f"""# V12 Fixed Alpha Futures Zero Rules

Verification date: **{OFFICIAL_VERIFICATION_DATE}** (Europe/Zurich). Sources were rechecked against the official Alpha Futures Help Center only.

## Verified official rules

| Rule | 25K Zero | 50K Zero | Source |
|---|---:|---:|---|
| Evaluation subscription | $79/month | $119/month | [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription) |
| Profit target | $1,500 | $3,000 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Maximum Loss Limit | $1,000 | $2,000 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [MLL](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll) |
| Daily Loss Guard | $500 | $1,000 | [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard) |
| Evaluation max position | 10 micros | 30 micros | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Evaluation consistency | None | None | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Qualified consistency | 40% | 40% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy), [Consistency Rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule) |
| Winning days | 5 days of >= $200 | 5 days of >= $200 | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Maximum withdrawal request | $1,000 | $1,500 | [Maximum Withdrawal](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request) |
| Minimum withdrawal request | $200 | $200 | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Maximum profit withdrawn per request | 50% | 50% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Trader split | 90% | 90% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Qualified monthly subscription | $0 | $0 | [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription) |
| Zero activation fee | $0 | $0 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [Activation Fee](https://help.alpha-futures.com/en/articles/9492083-activation-fee) |
| Qualified reset | $399, explicit only | $499, explicit only | [Reset](https://help.alpha-futures.com/en/articles/9492077-reset) |
| Trading session | 6PM ET to 5PM ET; close positions before 4:20PM ET | 6PM ET to 5PM ET; close positions before 4:20PM ET | [What and When You Can Trade](https://help.alpha-futures.com/en/articles/9492096-what-and-when-you-can-trade) |

## Lifecycle and billing interpretation

The official activation page says the Evaluation account is closed after reaching the profit target: [Activating your Qualified Account](https://help.alpha-futures.com/en/articles/9820801-activating-your-qualified-account). The fixed simulator therefore records `EVALUATION_PASSED`, cancels remaining Evaluation exposure, closes that ledger, and starts a separate Qualified ledger.

Alpha's current trading-hours page confirms both MBT and MET are available. CME contract metadata used for the proxy conversion is documented by [CME Micro Bitcoin](https://www.cmegroup.com/trading/files/micro-bitcoin-futures-fact-card-retail-us.pdf) and [CME Micro Ether](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html). BTC maps to MBT; ETH maps to MET; SPY is excluded because SPX already occupies MES.

The subscription page says billing continues after an Evaluation breach until the trader cancels or the next rebill resets the failed Evaluation. The replay therefore reports three explicit scenarios:

- `A_CANCEL_ON_BREACH`: cancel at breach; no future subscription or reset charge.
- `B_REBILL_AFTER_BREACH`: keep billing until the next calendar rebill and start a fresh Evaluation with no separate reset charge.
- `C_EXPLICIT_EVALUATION_RESET`: cancel at breach and charge the documented Evaluation Reset fee for an explicitly purchased reset.

Qualified monthly subscription and Zero activation fees are always zero. Qualified resets are not automatic; they are reported as zero in the primary replay and only represent a cost in an explicitly purchased scenario.

## Important rule difference from the old simulator

The old V12 model applied the same account ledger after pass, charged economics from payout-only fields, and labeled passed histories censored. The fixed simulator separates Evaluation and Qualified stages, records state transitions, separates trading PnL from withdrawals, and excludes censored stages from uncensored lifetime medians.

## Known data limitations

The frozen strategy produces Binance proxy fills rather than native CME futures. The retained trade objects do not contain a complete account-level intrabar mark stream, so floating DLG/MLL liquidation is represented conservatively from available trade-leg prices and last prices. News restrictions require a historical high-impact calendar, which was not retained; the relevant Qualified rule is documented in [News Trading Policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy).
"""


def run_v12_fixed(root: str | Path = ROOT, seed: int = 42) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    markets = legacy._load_cached_markets()
    streams, market_rows, validation_rows = {}, [], []
    skipped = []
    for market in markets:
        for timeframe in TIMEFRAMES:
            try:
                bars = legacy._read_cached_bars(market, timeframe)
                trades, equity = legacy._run_frozen(market, timeframe, bars)
                streams[(market, timeframe)] = {"bars": bars, "trades": trades, "equity": equity}
                market_rows.append(_market_metrics(market, timeframe, bars, trades, equity))
                validation_rows.extend(legacy._validation_metrics(market, timeframe, bars, trades, equity))
            except Exception as exc:
                skipped.append({"market": market, "timeframe": timeframe, "reason": f"{type(exc).__name__}: {exc}"})
    validation = pd.DataFrame(validation_rows)
    portfolios = _portfolio_members(validation, streams)
    first_proxy_prices = {}
    for (market, _timeframe), stream in streams.items():
        if market in CANONICAL_PROXIES and not stream["trades"].empty:
            first_proxy_prices.setdefault(market, float(stream["trades"].sort_values("fill_timestamp").iloc[0]["entry_price"]))
    conversion_context = build_synthetic_context(first_proxy_prices)
    prepared = {(market, timeframe, size): [_prepare_trade(row, market, size, conversion_context.get(market)) for row in stream["trades"].to_dict("records")] for (market, timeframe), stream in streams.items() if market in CANONICAL_PROXIES for size in POSITION_SIZES}
    paths = []
    for portfolio, value in portfolios.items():
        for timeframe in TIMEFRAMES:
            members = _members_for(portfolios, portfolio, timeframe)
            keys = [(market, timeframe) for market in members if (market, timeframe) in streams]
            if not keys:
                continue
            union_index = sorted(set().union(*[set(streams[key]["bars"].index) for key in keys]))
            starts = pd.date_range(min(union_index) + pd.Timedelta(days=legacy.START_WARMUP_DAYS), max(union_index), freq="MS", tz="UTC")
            starts = [next((t for t in pd.DatetimeIndex(union_index) if t >= month), None) for month in starts]
            starts = [t for t in starts if t is not None]
            data_end = max(union_index)
            for account_name, account in ACCOUNT_SPECS.items():
                for size in POSITION_SIZES:
                    trades = sorted([trade for market in members for trade in prepared.get((market, timeframe, size), [])], key=lambda row: (row["entry_timestamp"], row["market"], row["setup_id"]))
                    for start in starts:
                        run_id = f"fixed-{portfolio}-{timeframe}-{account_name}-{size}-{start.date()}".replace(" ", "_").replace("/", "-")
                        paths.append(_run_path(trades, start, data_end, account, PRIMARY_BILLING_SCENARIO, portfolio, timeframe, size, run_id, collect_events=False))
    lifetimes = _flatten_stage_rows(paths)
    prop = _summary_rows(paths)
    portfolio_results = pd.concat([pd.DataFrame(market_rows), _shared_portfolio_rows(paths, portfolios, streams)], ignore_index=True, sort=False)
    economics = _payout_economics(paths)
    transitions = pd.DataFrame([transition for path in paths for transition in path["transitions"]])
    if transitions.empty:
        transitions = pd.DataFrame(columns=["lifecycle_id", "timestamp", "state_before", "state_after", "reason", "valid_transition"])
    # Select representatives across the complete path set before rerunning
    # only the small set with detailed event retention.  Looking only at the
    # first N paths can miss failures or payouts that occur in later portfolio
    # or start-date partitions.
    representative_paths = {}
    for path in paths:
        account = path["account"]
        if (account, "pass_eval") not in representative_paths and any(e["passed"] for e in path["evaluations"]):
            representative_paths[(account, "pass_eval")] = path
        if (account, "fail_eval") not in representative_paths and any(e["failed"] for e in path["evaluations"]):
            representative_paths[(account, "fail_eval")] = path
        if (account, "qualified_payout") not in representative_paths and any(q["payout_count"] >= 1 for q in path["qualified"]):
            representative_paths[(account, "qualified_payout")] = path
    event_paths = []
    for path in representative_paths.values():
        trades_for_path = sorted([trade for market in _members_for(portfolios, path["portfolio"], path["timeframe"]) for trade in prepared.get((market, path["timeframe"], int(path["position_size"])), [])], key=lambda row: (row["entry_timestamp"], row["market"], row["setup_id"]))
        event_paths.append(_run_path(trades_for_path, pd.Timestamp(path["evaluations"][0]["start"]) if path["evaluations"] else pd.Timestamp(path["run_id"].split("-")[-1]), pd.Timestamp(path["evaluations"][0]["history_end"]) if path["evaluations"] else pd.Timestamp.max.tz_localize("UTC"), ACCOUNT_SPECS[path["account"]], PRIMARY_BILLING_SCENARIO, path["portfolio"], path["timeframe"], int(path["position_size"]), path["run_id"], collect_events=True))
    representative_events = _representative_logs(event_paths)
    censor = _censoring_summary(lifetimes)
    pnl_rows = []
    for _, r in pd.DataFrame(market_rows).iterrows():
        pnl_rows.append({"scope": "descriptive_market", "market": r.market, "timeframe": r.timeframe, "strategy_gross_pnl": r.gross_pnl, "strategy_net_pnl": r.net_pnl, "fees": r.fees, "slippage": r.slippage, "shared_account_pnl": "not directly comparable", "reconciliation_status": "Binance proxy strategy output"})
    for _, row in portfolio_results[portfolio_results.record_type == "shared_account_summary"].iterrows():
        pnl_rows.append({"scope": "shared_account", "market": "MULTI", "timeframe": row.timeframe, "portfolio": row.portfolio, "strategy_gross_pnl": "", "strategy_net_pnl": "", "fees": "", "slippage": "", "shared_account_pnl": row.net_trading_pnl, "gross_withdrawal_requested": row.gross_withdrawal_requested, "trader_payout_after_split": row.trader_payout_after_split, "evaluation_subscription_cost": row.evaluation_subscription_cost, "evaluation_reset_cost": row.evaluation_reset_cost, "net_trader_cashflow": row.net_trader_cashflow, "reconciliation_status": "shared account event-ledger aggregation"})
    transitions.to_csv(root / "v12_fixed_state_transitions.csv", index=False)
    prop.to_csv(root / "v12_fixed_prop_results.csv", index=False)
    portfolio_results.to_csv(root / "v12_fixed_portfolio_results.csv", index=False)
    lifetimes.to_csv(root / "v12_fixed_account_lifetimes.csv", index=False)
    economics.to_csv(root / "v12_fixed_payout_economics.csv", index=False)
    censor.to_csv(root / "v12_fixed_censoring_summary.csv", index=False)
    representative_events.to_csv(root / "v12_fixed_representative_account_events.csv", index=False)
    pd.DataFrame(pnl_rows).to_csv(root / "v12_fixed_pnl_reconciliation.csv", index=False)
    _confidence_rows(streams, portfolios).to_csv(root / "v12_fixed_confidence_warnings.csv", index=False)
    (root / "v12_fixed_verified_rules.md").write_text(_rules_markdown(), encoding="utf-8")
    _write_html(root / "v12_fixed_final_report.html", prop, portfolio_results, economics, censor, skipped, len(paths), len(transitions), len(representative_events))
    return {"markets": len(markets), "streams": len(streams), "paths": len(paths), "skipped": skipped, "root": str(root)}


def _write_html(path: Path, prop: pd.DataFrame, portfolio: pd.DataFrame, economics: pd.DataFrame, censor: pd.DataFrame, skipped: list[dict], paths: int, transitions: int, representative_events: int) -> None:
    def table(frame):
        return frame.to_html(index=False, classes="data", border=0) if frame is not None and not frame.empty else "<p>None</p>"
    text = f"""<!doctype html><html><head><meta charset='utf-8'><title>V12 Fixed Alpha Lifecycle</title><style>body{{font-family:Arial;margin:2rem;max-width:1500px}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}</style></head><body><h1>V12 Fixed Alpha Futures Zero Replay</h1><div class='warn'><b>PROXY-BASED EXPLORATORY RESEARCH.</b> Strategy rules were unchanged. Previous V12 Prop/economics outputs are invalidated for lifecycle decisions. Cached Binance data only. Paths: {paths}; transitions: {transitions}; representative event rows: {representative_events}; skipped streams: {len(skipped)}.</div><h2>Verified lifecycle changes</h2><ul><li>Evaluation stops at pass and creates a separate Qualified lifecycle.</li><li>Qualified monthly subscription and Zero activation fee are zero.</li><li>Failures use explicit billing scenarios A/B/C.</li><li>Portfolio results use one shared chronological account.</li></ul><h2>Prop results</h2>{table(prop)}<h2>Shared portfolio results</h2>{table(portfolio[portfolio.record_type == 'shared_account_summary'] if 'record_type' in portfolio else portfolio)}<h2>Payout economics</h2>{table(economics.groupby(['billing_scenario'], as_index=False)[['gross_trading_pnl','net_trading_pnl','gross_withdrawal_requested','trader_payout_after_split','evaluation_subscription_cost','evaluation_reset_cost','net_trader_cashflow']].sum(numeric_only=True) if not economics.empty else economics)}<h2>Censoring</h2>{table(censor)}<p>Rules and source URLs: v12_fixed_verified_rules.md. Limitations: floating DLG/MLL and news-event restrictions are explicitly documented in v12_fixed_confidence_warnings.csv.</p></body></html>"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    print(run_v12_fixed())
