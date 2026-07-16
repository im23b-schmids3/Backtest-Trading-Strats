"""Deterministic multi-account Alpha Zero 25K risk-management replay.

The signal stream, frozen strategy, cached Binance data, proxy mappings,
contract mappings, execution assumptions, and Alpha rules are imported from
the existing V12 research.  This module changes only account purchase timing,
per-trade initial-risk sizing, and the additional two-loss daily stop rule.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fib_backtester.research import v12_binance_proxy_prop_simulation as legacy
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed
from fib_backtester.research.v12_contract_registry import CONTRACTS, PROXY_SYMBOLS, build_synthetic_context, mapped_price, round_to_tick


ROOT = Path("reports/v13_fixed")
START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
PORTFOLIO = "Portfolio C - BTC + ETH + Gold"
MEMBERS = ["BTC", "ETH", "Gold"]
TIMEFRAME = "4h"
MAX_INITIAL_RISK = 250.0
MAX_MICROS = 10
WINNING_DAY_MINIMUM = legacy.WINNING_DAY_MINIMUM


@dataclass
class Position:
    trade: dict
    signal_id: int
    remaining: int
    last_price: float
    gross: float = 0.0
    fees: float = 0.0
    net: float = 0.0
    entry_fee: float = 0.0
    legs: list[dict] = field(default_factory=list)
    forced: bool = False


@dataclass
class Account:
    account_id: str
    kind: str
    purchase_timestamp: pd.Timestamp
    balance: float
    initial_balance: float
    mll: float
    state: str
    billing_next: pd.Timestamp | None = None
    subscription_paid: float = 0.0
    daily_profit: float = 0.0
    daily_losses: int = 0
    daily_wins: int = 0
    daily_stop: bool = False
    current_session: str | None = None
    positions: dict[int, Position] = field(default_factory=dict)
    trades_taken: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    pass_timestamp: pd.Timestamp | None = None
    failure_timestamp: pd.Timestamp | None = None
    qualified_created_timestamp: pd.Timestamp | None = None
    cycle_profit: float = 0.0
    cycle_days: dict[str, float] = field(default_factory=dict)
    winning_days: set[str] = field(default_factory=set)
    payout_count: int = 0
    gross_payout: float = 0.0
    trader_payout: float = 0.0
    first_payout_timestamp: pd.Timestamp | None = None
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    marked_equity: float = 0.0
    peak_equity: float = 0.0
    max_equity_drawdown: float = 0.0
    max_dlg_usage: float = 0.0
    max_mll_usage: float = 0.0
    worst_equity_timestamp: pd.Timestamp | None = None
    mll_breach_timestamp: pd.Timestamp | None = None
    dlg_breach_count: int = 0


def _session(timestamp: pd.Timestamp) -> str:
    return fixed._session(pd.Timestamp(timestamp))


def _month(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m")


def _utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _load_signal_stream():
    frames: dict[str, pd.DataFrame] = {}
    bar_times: set[pd.Timestamp] = set()
    latest = None
    first_prices = {}
    for market in MEMBERS:
        bars = legacy._read_cached_bars(market, TIMEFRAME)
        frames[market] = bars
        latest = bars.index.max() if latest is None else max(latest, bars.index.max())
        for timestamp in bars.index:
            timestamp = _utc(timestamp)
            if START <= timestamp <= latest:
                bar_times.add(timestamp)
        if not bars.empty:
            first_prices[market] = float(bars.iloc[0]["close"])
    end_exclusive = latest + pd.Timedelta(hours=4)
    contexts = build_synthetic_context(first_prices)
    mark_prices: dict[str, dict[pd.Timestamp, float]] = {}
    for market, bars in frames.items():
        spec = fixed.CANONICAL_PROXIES[market]
        mark_prices[market] = {_utc(timestamp): round_to_tick(mapped_price(float(row["close"]), market, contexts.get(market)), spec.tick_size) for timestamp, row in bars.iterrows() if START <= _utc(timestamp) < end_exclusive}
    session_closes = [_utc(timestamp) for timestamp in fixed._session_close_events(START, end_exclusive) if START <= _utc(timestamp) < end_exclusive]
    timeline = sorted(set(bar_times) | set(session_closes))
    raw_rows = []
    for market, bars in frames.items():
        frozen, _ = legacy._run_frozen(market, TIMEFRAME, bars)
        for raw in frozen.to_dict("records"):
            entry_time = _utc(raw["fill_timestamp"])
            if not (START <= entry_time < end_exclusive):
                continue
            spec = fixed.CANONICAL_PROXIES[market]
            entry = round_to_tick(mapped_price(float(raw["entry_price"]), market, contexts.get(market)), spec.tick_size)
            stop = round_to_tick(mapped_price(float(raw["initial_stop"]), market, contexts.get(market)), spec.tick_size)
            stop_distance = abs(entry - stop)
            stop_ticks = stop_distance / spec.tick_size if spec.tick_size else math.inf
            tick_value = CONTRACTS[spec.alpha_product].tick_value
            risk_per_contract = stop_ticks * tick_value
            contracts = min(MAX_MICROS, math.floor(MAX_INITIAL_RISK / risk_per_contract)) if risk_per_contract > 0 else 0
            trade = None
            if contracts > 0:
                trade = fixed._prepare_trade(raw, market, contracts, contexts.get(market))
                trade.update({"signal_timestamp": _utc(raw["signal_timestamp"]), "setup_timestamp": str(raw.get("signal_timestamp", "")), "initial_stop": stop, "stop_distance": stop_distance, "stop_ticks": stop_ticks, "tick_size": spec.tick_size, "tick_value": tick_value, "dollar_risk_per_contract": risk_per_contract, "initial_risk": risk_per_contract * contracts, "raw_setup_id": raw.get("setup_id", "")})
            raw_rows.append({"market": market, "entry_timestamp": entry_time, "raw": raw, "entry": entry, "stop": stop, "stop_distance": stop_distance, "stop_ticks": stop_ticks, "tick_size": spec.tick_size, "tick_value": tick_value, "dollar_risk_per_contract": risk_per_contract, "contracts": contracts, "trade": trade})
    raw_rows.sort(key=lambda row: (row["entry_timestamp"], row["market"], str(row["raw"].get("setup_id", ""))))
    for signal_id, row in enumerate(raw_rows, 1):
        row["signal_id"] = signal_id
        if row["trade"] is not None:
            row["trade"]["signal_id"] = signal_id
    return raw_rows, timeline, sorted(bar_times), session_closes, latest, end_exclusive, contexts, mark_prices


class Replay:
    def __init__(self, signals, timeline, end_exclusive, last_prices, mark_prices=None, trading_times=None, session_closes=None):
        self.signals = signals
        self.timeline = timeline
        self.end_exclusive = end_exclusive
        self.last_prices = last_prices
        self.mark_prices = mark_prices or {market: {} for market in MEMBERS}
        self.trading_times = trading_times or list(timeline)
        self.session_closes = set(session_closes or [])
        self.accounts: list[Account] = []
        self.events: list[dict] = []
        self.trades: list[dict] = []
        self.skipped: list[dict] = []
        self.payouts: list[dict] = []
        self.month_first: dict[str, pd.Timestamp] = {}
        self.replacements: dict[pd.Timestamp, list[str]] = {}
        self.next_eval = 0
        self.next_qualified = 0
        self.trade_number = 0
        self._signal_by_timestamp: dict[pd.Timestamp, list[dict]] = {}
        for signal in signals:
            self._signal_by_timestamp.setdefault(signal["entry_timestamp"], []).append(signal)
        for timestamp in timeline:
            if timestamp in self.trading_times:
                self.month_first.setdefault(_month(timestamp), timestamp)

    def _emit(self, timestamp, account, reason, old=None, new=None, subscription=0.0, gross=0.0, trader=0.0, before=None, after=None):
        self.events.append({"timestamp": str(timestamp), "account_id": account.account_id if account else "", "account_type": account.kind if account else "", "old_state": old if old is not None else (account.state if account else ""), "new_state": new if new is not None else (account.state if account else ""), "event_reason": reason, "balance_before": float(account.balance if before is None else before) if account else float(after or 0.0), "balance_after": float(account.balance if after is None else after) if account else float(after or 0.0), "marked_equity": float(account.marked_equity) if account else float(after or 0.0), "related_subscription_cost": float(subscription), "related_gross_withdrawal": float(gross), "related_trader_payout": float(trader), "winning_days_progress": f"{len(account.winning_days)} / 5" if account and account.kind == "QUALIFIED" else "", "consistency_percentage": self._consistency(account) * 100 if account and account.kind == "QUALIFIED" else 0.0, "payout_cycle_profit": float(account.cycle_profit) if account else 0.0, "daily_loss_value": float(account.daily_profit) if account else 0.0, "mll_threshold": float(account.mll) if account else ""})

    @staticmethod
    def _consistency(account):
        return max(account.cycle_days.values(), default=0.0) / account.cycle_profit if account and account.cycle_profit > 0 else 0.0

    def _buy_evaluation(self, timestamp, reason):
        self.next_eval += 1
        spec = fixed.ACCOUNT_SPECS["25K Zero"]
        account = Account(f"EVAL-{self.next_eval:03d}", "EVALUATION", timestamp, spec.account_size, spec.account_size, spec.account_size - spec.mll_amount, "EVALUATION_ACTIVE", billing_next=timestamp + pd.DateOffset(months=1), peak_balance=spec.account_size)
        self.accounts.append(account)
        self._emit(timestamp, account, "Evaluation purchased", old="NONE", new=account.state, before=account.balance, after=account.balance)
        account.subscription_paid += spec.subscription
        self._emit(timestamp, account, "subscription charged", subscription=spec.subscription, before=account.balance, after=account.balance)
        return account

    def _buy_qualified(self, timestamp, balance):
        self.next_qualified += 1
        account = Account(f"QUAL-{self.next_qualified:03d}", "QUALIFIED", timestamp, balance, balance, balance - fixed.ACCOUNT_SPECS["25K Zero"].mll_amount, "QUALIFIED_ACTIVE", qualified_created_timestamp=timestamp, peak_balance=balance)
        self.accounts.append(account)
        self._emit(timestamp, account, "Qualified account created", old="NONE", new=account.state, before=balance, after=balance)
        return account

    def _charge_subscriptions(self, account, timestamp):
        if account.kind != "EVALUATION" or account.billing_next is None or account.state not in {"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"}:
            return
        spec = fixed.ACCOUNT_SPECS["25K Zero"]
        while account.billing_next <= timestamp:
            charge_time = account.billing_next
            account.subscription_paid += spec.subscription
            self._emit(charge_time, account, "subscription charged", subscription=spec.subscription, before=account.balance, after=account.balance)
            account.billing_next = charge_time + pd.DateOffset(months=1)

    def _price_for(self, market, timestamp):
        timestamp = _utc(timestamp)
        available = [key for key in self.mark_prices.get(market, {}) if key <= timestamp]
        return self.mark_prices.get(market, {}).get(max(available), self.last_prices[market]) if available else self.last_prices[market]

    @staticmethod
    def _fee_rate(market):
        return 0.001 if market in {"BTC", "ETH"} else 0.0005

    def _mark_account(self, account, timestamp):
        equity = float(account.balance)
        unrealized = 0.0
        estimated_exit_fees = 0.0
        for position in account.positions.values():
            price = self._price_for(position.trade["market"], timestamp)
            position.last_price = price
            spec = fixed.CANONICAL_PROXIES[position.trade["market"]]
            direction = 1.0 if position.trade["side"] == "long" else -1.0
            unrealized += direction * (price - position.trade["entry"]) * spec.multiplier * position.remaining
            estimated_exit_fees += abs(price * spec.multiplier * position.remaining) * self._fee_rate(position.trade["market"])
        equity += unrealized - estimated_exit_fees
        daily_marked = account.daily_profit + unrealized - estimated_exit_fees
        account.marked_equity = equity
        account.peak_balance = max(account.peak_balance, account.balance)
        account.max_drawdown = max(account.max_drawdown, account.peak_balance - account.balance)
        if account.peak_equity == 0.0:
            account.peak_equity = equity
        account.peak_equity = max(account.peak_equity, equity)
        drawdown = account.peak_equity - equity
        if drawdown >= account.max_equity_drawdown:
            account.max_equity_drawdown = drawdown
            account.worst_equity_timestamp = _utc(timestamp)
        account.max_dlg_usage = max(account.max_dlg_usage, max(0.0, -daily_marked))
        account.max_mll_usage = max(account.max_mll_usage, max(0.0, account.mll - equity))
        return equity, daily_marked

    def _check_lifecycle(self, account, timestamp, trigger="checkpoint"):
        """Canonical per-account target, DLG, MLL, and state checker."""
        if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED", "CENSORED_END_OF_DATA"}:
            return
        equity, daily_marked = self._mark_account(account, timestamp)
        if equity <= account.mll + 1e-9:
            self._fail(account, timestamp, "Maximum Loss Limit; marked-equity breach")
            return
        if daily_marked <= -fixed.ACCOUNT_SPECS["25K Zero"].daily_loss_guard and not account.state.endswith("DAILY_LOCKED"):
            old = account.state
            account.state = "EVALUATION_DAILY_LOCKED" if account.kind == "EVALUATION" else "QUALIFIED_DAILY_LOCKED"
            account.dlg_breach_count += 1
            self._emit(timestamp, account, "Daily Loss Guard activated", old=old, new=account.state)
            self._flatten(account, timestamp, "daily_loss_guard")
            equity, _ = self._mark_account(account, timestamp)
            if equity <= account.mll + 1e-9:
                self._fail(account, timestamp, "Maximum Loss Limit; marked-equity breach after DLG")
            return
        if account.kind == "EVALUATION" and account.state == "EVALUATION_ACTIVE" and account.balance >= account.initial_balance + fixed.ACCOUNT_SPECS["25K Zero"].target:
            self._pass(account, timestamp)

    def _refresh_trade_state(self, account, timestamp):
        """Keep the state on a completed trade row aligned with its checkpoint."""
        stamp = str(timestamp)
        for row in reversed(self.trades):
            if row.get("account_id") == account.account_id and row.get("exit_timestamp") == stamp:
                row["account_state_after_trade"] = account.state
                return

    def _finish_day(self, account, timestamp):
        if account.current_session is None:
            return
        old_session = account.current_session
        self._check_lifecycle(account, timestamp, "daily_rollover")
        if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED"}:
            return
        if account.kind == "QUALIFIED":
            if account.daily_profit > 0:
                account.cycle_days[old_session] = account.daily_profit
            if account.daily_profit >= WINNING_DAY_MINIMUM:
                account.winning_days.add(old_session)
                self._emit(timestamp, account, "winning day recorded")
            self._emit(timestamp, account, "consistency changed")
            self._maybe_payout(account, timestamp)
        account.mll = min(account.initial_balance, max(account.mll, account.balance - fixed.ACCOUNT_SPECS["25K Zero"].mll_amount))
        account.daily_profit = 0.0
        account.daily_losses = 0
        account.daily_wins = 0
        account.daily_stop = False

    def _maybe_payout(self, account, timestamp):
        spec = fixed.ACCOUNT_SPECS["25K Zero"]
        if len(account.winning_days) < 5 or account.cycle_profit <= 0 or self._consistency(account) > legacy.CONSISTENCY_LIMIT:
            return
        request = min(0.50 * account.cycle_profit, spec.payout_max)
        if request < WINNING_DAY_MINIMUM or account.balance - request <= account.mll:
            return
        before = account.balance
        account.balance -= request
        account.payout_count += 1
        account.gross_payout += request
        account.trader_payout += request * legacy.PAYOUT_SPLIT
        if account.first_payout_timestamp is None:
            account.first_payout_timestamp = timestamp
        self.payouts.append({"timestamp": str(timestamp), "account_id": account.account_id, "gross_payout": request, "trader_payout_after_split": request * legacy.PAYOUT_SPLIT, "balance_after_payout": account.balance, "winning_days_used": len(account.winning_days), "payout_cycle_profit": account.cycle_profit, "consistency_percentage": self._consistency(account) * 100})
        self._emit(timestamp, account, "payout eligibility reached")
        self._emit(timestamp, account, "payout requested", gross=request)
        self._emit(timestamp, account, "payout received", gross=request, trader=request * legacy.PAYOUT_SPLIT, before=before, after=account.balance)
        account.cycle_profit = 0.0
        account.cycle_days.clear()
        account.winning_days.clear()

    def _advance(self, timestamp):
        session = _session(timestamp)
        for account in self.accounts:
            if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED"}:
                continue
            self._charge_subscriptions(account, timestamp)
            if account.current_session is None:
                account.current_session = session
            elif account.current_session != session:
                self._finish_day(account, timestamp)
                if account.state.endswith("DAILY_LOCKED"):
                    old = account.state
                    account.state = "EVALUATION_ACTIVE" if account.kind == "EVALUATION" else "QUALIFIED_ACTIVE"
                    self._emit(timestamp, account, "next trading day opened", old=old, new=account.state)
                account.current_session = session
            self._check_lifecycle(account, timestamp, "mark_to_market")

    def _session_cutoff(self, timestamp):
        local = timestamp.tz_convert("America/New_York")
        return local.hour == 17 or (local.hour == 16 and local.minute >= 20)

    def _record_skip(self, account, signal, reason):
        self.skipped.append({"timestamp": str(signal["entry_timestamp"]), "account_id": account.account_id, "account_type": account.kind, "signal_id": signal["signal_id"], "market": signal["market"], "proxy_symbol": PROXY_SYMBOLS[signal["market"]], "timeframe": TIMEFRAME, "direction": signal["raw"]["side"], "contracts_would_be_used": signal["contracts"], "stop_distance": signal["stop_distance"], "dollar_risk_per_contract": signal["dollar_risk_per_contract"], "reason": reason})

    def _apply_leg(self, account, signal_id, leg, timestamp, forced=False, check=True):
        position = account.positions.get(signal_id)
        if position is None:
            return
        before = account.balance
        account.balance += leg["net"]
        account.gross_pnl += leg["gross"]
        account.net_pnl += leg["net"]
        account.fees += leg["fee"]
        account.daily_profit += leg["net"]
        account.cycle_profit += leg["net"] if account.kind == "QUALIFIED" else 0.0
        position.gross += leg["gross"]
        position.fees += leg["fee"]
        position.net += leg["net"]
        position.remaining -= int(leg["quantity"])
        position.last_price = float(leg["price"])
        position.legs.append(dict(leg, timestamp=str(timestamp), forced=forced))
        position.forced = position.forced or forced
        account.peak_balance = max(account.peak_balance, account.balance)
        account.max_drawdown = max(account.max_drawdown, account.peak_balance - account.balance)
        if position.remaining <= 0:
            self._finalize_trade(account, signal_id, timestamp)
        if not check:
            return
        if account.state.endswith("FAILED") or account.state in {"EVALUATION_PASSED", "CLOSED"}:
            return
        self._check_lifecycle(account, timestamp, "realized_leg")
        self._refresh_trade_state(account, timestamp)
        if account.state in {"EVALUATION_ACTIVE", "QUALIFIED_ACTIVE", "EVALUATION_DAILY_LOCKED", "QUALIFIED_DAILY_LOCKED"} and account.daily_losses >= 2 and account.daily_wins == 0 and not account.daily_stop:
            account.daily_stop = True
            self._emit(timestamp, account, "two losing trades and no winning trades; daily stop activated")

    def _finalize_trade(self, account, signal_id, timestamp):
        position = account.positions.pop(signal_id, None)
        if position is None:
            return
        net = position.net - position.entry_fee
        if net < 0:
            account.daily_losses += 1
        elif net > 0:
            account.daily_wins += 1
        self.trade_number += 1
        trade = position.trade
        self.trades.append({"trade_number": self.trade_number, "signal_id": signal_id, "timestamp": str(position.legs[-1]["timestamp"]), "account_id": account.account_id, "account_type": account.kind, "market": trade["market"], "proxy_symbol": PROXY_SYMBOLS[trade["market"]], "mapped_futures_contract": trade["alpha_product"], "timeframe": TIMEFRAME, "direction": trade["side"], "setup_timestamp": trade.get("setup_timestamp", ""), "entry_timestamp": str(trade["entry_timestamp"]), "exit_timestamp": str(position.legs[-1]["timestamp"]), "entry_price": trade["entry"], "initial_stop_price": trade["initial_stop"], "final_exit_price": position.legs[-1]["price"], "total_contracts": trade["contracts"], "stop_distance": trade["stop_distance"], "stop_ticks": trade["stop_ticks"], "tick_size": trade["tick_size"], "tick_value": trade["tick_value"], "dollar_risk_per_contract": trade["dollar_risk_per_contract"], "total_initial_risk": trade["initial_risk"], "exit_reason": position.legs[-1]["reason"], "tp_levels_reached": ",".join(str(i) for i in range(1, 6) if any(leg["reason"] == f"tp{i}" for leg in position.legs)), "gross_pnl": position.gross, "entry_fees": position.entry_fee, "exit_fees": position.fees, "fees": position.entry_fee + position.fees, "slippage": 0.0, "net_pnl": net, "balance_before": position.trade["balance_before"], "balance_after": account.balance, "account_state_after_trade": account.state, "forced_exit": position.forced, "daily_losses_after_trade": account.daily_losses, "daily_wins_after_trade": account.daily_wins, "quantity_reconciles": sum(leg["quantity"] for leg in position.legs) == trade["contracts"]})

    def _accept(self, account, signal, timestamp):
        if account.state not in {"EVALUATION_ACTIVE", "QUALIFIED_ACTIVE"}:
            self._record_skip(account, signal, "inactive account")
            return
        if account.daily_stop:
            self._record_skip(account, signal, "daily stop rule")
            return
        if self._session_cutoff(timestamp):
            self._record_skip(account, signal, "session cutoff")
            return
        if signal["contracts"] <= 0:
            self._record_skip(account, signal, "risk above 250 USD")
            return
        current_contracts = sum(position.remaining for position in account.positions.values())
        same_market = any(position.trade["market"] == signal["market"] for position in account.positions.values())
        max_contracts = fixed.ACCOUNT_SPECS["25K Zero"].max_micros_evaluation if account.kind == "EVALUATION" else fixed.ACCOUNT_SPECS["25K Zero"].max_micros_qualified_initial
        if same_market or current_contracts + signal["contracts"] > max_contracts:
            self._record_skip(account, signal, "contract limit")
            return
        trade = dict(signal["trade"])
        trade["balance_before"] = account.balance
        account.balance -= trade["entry_fee"]
        account.daily_profit -= trade["entry_fee"]
        account.net_pnl -= trade["entry_fee"]
        account.fees += trade["entry_fee"]
        position = Position(trade, signal["signal_id"], trade["contracts"], trade["entry"], entry_fee=trade["entry_fee"])
        account.positions[signal["signal_id"]] = position
        account.trades_taken += 1
        self._check_lifecycle(account, timestamp, "entry_fee_booking")

    def _flatten(self, account, timestamp, reason):
        for signal_id, position in list(account.positions.items()):
            if position.remaining <= 0:
                continue
            price = self._price_for(position.trade["market"], timestamp)
            leg = fixed._flatten_leg(position.trade, price, timestamp, position.remaining, reason)
            self._apply_leg(account, signal_id, leg, timestamp, forced=True, check=False)

    def _fail(self, account, timestamp, reason):
        if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED"}:
            return
        old = account.state
        if "Maximum Loss Limit" in reason and account.mll_breach_timestamp is None:
            account.mll_breach_timestamp = timestamp
        account.state = "EVALUATION_FAILED" if account.kind == "EVALUATION" else "QUALIFIED_FAILED"
        account.failure_timestamp = timestamp
        self._flatten(account, timestamp, "account_failure_flatten")
        self._emit(timestamp, account, "Evaluation failed" if account.kind == "EVALUATION" else "Qualified failed", old=old, new=account.state)
        if account.kind == "EVALUATION":
            next_index = bisect_right(self.trading_times, timestamp)
            if next_index < len(self.trading_times):
                self.replacements.setdefault(self.trading_times[next_index], []).append("replacement Evaluation after failure")

    def _pass(self, account, timestamp):
        if account.kind != "EVALUATION" or account.state.endswith("FAILED"):
            return
        old = account.state
        account.state = "EVALUATION_PASSED"
        account.pass_timestamp = timestamp
        self._flatten(account, timestamp, "evaluation_pass_flatten")
        self._emit(timestamp, account, "Evaluation passed", old=old, new=account.state)
        account.state = "CLOSED"
        self._emit(timestamp, account, "Evaluation account closed after pass", old="EVALUATION_PASSED", new="CLOSED")
        qualified = self._buy_qualified(timestamp + pd.Timedelta(nanoseconds=1), account.balance)
        account.qualified_created_timestamp = qualified.purchase_timestamp

    def _process_exits(self, timestamp):
        for account in list(self.accounts):
            if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED"}:
                continue
            for signal_id, position in list(account.positions.items()):
                for leg in position.trade["legs"]:
                    if _utc(leg["timestamp"]) == timestamp and position.remaining > 0:
                        self._apply_leg(account, signal_id, leg, timestamp)
                        if account.state.endswith("FAILED") or account.state in {"CLOSED", "EVALUATION_PASSED"}:
                            break

    def run(self):
        month_purchase_times = set(self.month_first.values())
        all_times = sorted(set(self.timeline) | set(self._signal_by_timestamp) | set(self.replacements) | {self.end_exclusive})
        for timestamp in all_times:
            if timestamp >= self.end_exclusive:
                break
            self._advance(timestamp)
            if timestamp in month_purchase_times:
                self._buy_evaluation(timestamp, "first trading day of calendar month")
            for reason in self.replacements.pop(timestamp, []):
                self._buy_evaluation(timestamp, reason)
            self._process_exits(timestamp)
            if timestamp in self.session_closes:
                for account in list(self.accounts):
                    if account.state in {"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED", "QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"} and account.positions:
                        self._emit(timestamp, account, "normal session close; positions flattened")
                        self._flatten(account, timestamp, "session_forced_liquidation")
                        self._check_lifecycle(account, timestamp, "forced_session_close")
            for signal in self._signal_by_timestamp.get(timestamp, []):
                for account in list(self.accounts):
                    if account.purchase_timestamp <= timestamp:
                        self._accept(account, signal, timestamp)
        self._advance(self.end_exclusive)
        for account in list(self.accounts):
            if account.state in {"QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED", "EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"}:
                if account.positions:
                    self._flatten(account, self.end_exclusive, "end_of_data_flatten")
                    self._check_lifecycle(account, self.end_exclusive, "terminal_mark_to_market")
                if account.state in {"QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED", "EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"}:
                    self._emit(self.end_exclusive, account, "account censored at end of data", old=account.state, new="CENSORED_END_OF_DATA")
                    account.state = "CENSORED_END_OF_DATA"
        return self


def _build_outputs(replay: Replay, latest, end_exclusive, root: Path):
    trades = pd.DataFrame(replay.trades)
    skipped = pd.DataFrame(replay.skipped)
    events = pd.DataFrame(replay.events)
    if not events.empty:
        events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
        cumulative = 0.0
        for index, row in events.iterrows():
            cumulative += float(row.related_trader_payout) - float(row.related_subscription_cost)
            events.loc[index, "cumulative_external_cashflow"] = cumulative
    payouts = pd.DataFrame(replay.payouts)
    account_rows = []
    for account in replay.accounts:
        account_rows.append({"account_id": account.account_id, "account_type": account.kind, "purchase_timestamp": str(account.purchase_timestamp), "final_state": account.state, "subscription_paid": account.subscription_paid, "pass_timestamp": str(account.pass_timestamp) if account.pass_timestamp else "", "failure_timestamp": str(account.failure_timestamp) if account.failure_timestamp else "", "qualified_created_timestamp": str(account.qualified_created_timestamp) if account.qualified_created_timestamp else "", "trades_taken": account.trades_taken, "gross_pnl": account.gross_pnl, "net_pnl": account.net_pnl, "fees": account.fees, "payout_count": account.payout_count, "gross_payout": account.gross_payout, "trader_payout_after_split": account.trader_payout, "external_cashflow": account.trader_payout - account.subscription_paid, "first_payout_timestamp": str(account.first_payout_timestamp) if account.first_payout_timestamp else "", "max_cash_balance_drawdown": account.max_drawdown, "max_marked_equity_drawdown": account.max_equity_drawdown, "maximum_dlg_usage": account.max_dlg_usage, "maximum_mll_usage": account.max_mll_usage, "worst_marked_equity_timestamp": str(account.worst_equity_timestamp) if account.worst_equity_timestamp else "", "mll_breach_timestamp": str(account.mll_breach_timestamp) if account.mll_breach_timestamp else "", "dlg_breach_count": account.dlg_breach_count, "marked_equity": account.marked_equity, "winning_days_remaining_cycle": len(account.winning_days), "current_balance": account.balance})
    account_summary = pd.DataFrame(account_rows)
    months = [str(period) for period in pd.period_range(START.strftime("%Y-%m"), latest.strftime("%Y-%m"), freq="M")]
    monthly_rows = []
    event_times = pd.to_datetime(events.timestamp, utc=True, format="mixed") if not events.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for month in months:
        month_end = pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        em = events[event_times <= month_end] if not events.empty else pd.DataFrame()
        month_events = events[event_times.dt.strftime("%Y-%m") == month] if not events.empty else pd.DataFrame()
        tm = trades[pd.to_datetime(trades.exit_timestamp, utc=True, format="mixed").dt.strftime("%Y-%m") == month] if not trades.empty else pd.DataFrame()
        active_eval = em[(em.account_type == "EVALUATION") & em.new_state.isin(["EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"])] if not em.empty else pd.DataFrame()
        active_qualified = em[(em.account_type == "QUALIFIED") & em.new_state.isin(["QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"])] if not em.empty else pd.DataFrame()
        latest_by_account = em.sort_values("timestamp").drop_duplicates("account_id", keep="last") if not em.empty else pd.DataFrame()
        active_eval = latest_by_account[(latest_by_account.account_type == "EVALUATION") & latest_by_account.new_state.isin(["EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"])] if not latest_by_account.empty else pd.DataFrame()
        active_qualified = latest_by_account[(latest_by_account.account_type == "QUALIFIED") & latest_by_account.new_state.isin(["QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"])] if not latest_by_account.empty else pd.DataFrame()
        monthly_rows.append({"month": month, "active_evaluations": ",".join(active_eval.account_id) if not active_eval.empty else "", "active_qualified": ",".join(active_qualified.account_id) if not active_qualified.empty else "", "new_evaluations": int((month_events.event_reason == "Evaluation purchased").sum()) if not month_events.empty else 0, "failed_evaluations": int(((month_events.account_type == "EVALUATION") & (month_events.event_reason == "Evaluation failed")).sum()) if not month_events.empty else 0, "passed_evaluations": int(((month_events.account_type == "EVALUATION") & (month_events.event_reason == "Evaluation passed")).sum()) if not month_events.empty else 0, "subscriptions": float(month_events.related_subscription_cost.sum()) if not month_events.empty else 0.0, "payouts": int(len(payouts[pd.to_datetime(payouts.timestamp, utc=True, format="mixed").dt.strftime("%Y-%m") == month])) if not payouts.empty else 0, "gross_payout": float(payouts.loc[pd.to_datetime(payouts.timestamp, utc=True, format="mixed").dt.strftime("%Y-%m") == month, "gross_payout"].sum()) if not payouts.empty else 0.0, "trader_payout": float(payouts.loc[pd.to_datetime(payouts.timestamp, utc=True, format="mixed").dt.strftime("%Y-%m") == month, "trader_payout_after_split"].sum()) if not payouts.empty else 0.0, "gross_trading_pnl": float(tm.gross_pnl.sum()) if not tm.empty else 0.0, "net_trading_pnl": float(tm.net_pnl.sum()) if not tm.empty else 0.0, "external_cashflow": float(month_events.related_trader_payout.sum() - month_events.related_subscription_cost.sum()) if not month_events.empty else 0.0})
    monthly = pd.DataFrame(monthly_rows)
    monthly["cumulative_external_cashflow"] = monthly.external_cashflow.cumsum()
    total_subscriptions = float(account_summary.subscription_paid.sum()) if not account_summary.empty else 0.0
    total_payout = float(account_summary.gross_payout.sum()) if not account_summary.empty else 0.0
    total_trader_payout = float(account_summary.trader_payout_after_split.sum()) if not account_summary.empty else 0.0
    def active_end(account):
        terminal = end_exclusive
        if account.kind == "EVALUATION":
            terminal = min([value for value in (account.pass_timestamp, account.failure_timestamp, end_exclusive) if value is not None])
        elif account.failure_timestamp is not None:
            terminal = min(account.failure_timestamp, end_exclusive)
        return terminal

    account_days = sum(max((active_end(account) - account.purchase_timestamp).total_seconds() / 86400, 0.0) for account in replay.accounts)
    active_avg = account_days / max((end_exclusive - START).total_seconds() / 86400, 1.0)
    def active_at(account, timestamp):
        if account.purchase_timestamp > timestamp:
            return False
        if account.kind == "EVALUATION":
            return not ((account.pass_timestamp is not None and timestamp >= account.pass_timestamp) or (account.failure_timestamp is not None and timestamp >= account.failure_timestamp))
        return account.failure_timestamp is None or timestamp < account.failure_timestamp

    final = {"simulation_start": str(START), "latest_completed_cached_candle": str(latest), "simulation_end_exclusive": str(end_exclusive), "selected_portfolio": PORTFOLIO, "selected_timeframe": TIMEFRAME, "account_type": "Alpha Futures Zero 25K", "maximum_account_micros": MAX_MICROS, "maximum_initial_risk_per_trade": MAX_INITIAL_RISK, "evaluations_purchased": int((account_summary.account_type == "EVALUATION").sum()) if not account_summary.empty else 0, "subscriptions_paid": total_subscriptions, "evaluations_passed": int(account_summary.pass_timestamp.astype(bool).sum()) if not account_summary.empty else 0, "evaluations_failed": int(account_summary.failure_timestamp.astype(bool).sum()) if not account_summary.empty else 0, "qualified_accounts_created": int((account_summary.account_type == "QUALIFIED").sum()) if not account_summary.empty else 0, "total_payouts": int(account_summary.payout_count.sum()) if not account_summary.empty else len(payouts), "gross_payout_amount": total_payout, "trader_payout_after_split": total_trader_payout, "net_external_cashflow": total_trader_payout - total_subscriptions, "average_active_accounts": active_avg, "maximum_simultaneously_active_accounts": max(sum(active_at(account, timestamp) for account in replay.accounts) for timestamp in replay.timeline) if replay.timeline else 0, "average_contracts_traded": float(trades.total_contracts.mean()) if not trades.empty else 0.0, "average_initial_risk_per_trade": float(trades.total_initial_risk.mean()) if not trades.empty else 0.0, "risk_skipped_trades": int((skipped.reason == "risk above 250 USD").sum()) if not skipped.empty else 0, "daily_stop_skipped_trades": int((skipped.reason == "daily stop rule").sum()) if not skipped.empty else 0, "contract_limit_skipped_trades": int((skipped.reason == "contract limit").sum()) if not skipped.empty else 0, "pass_rate": (int(account_summary.pass_timestamp.astype(bool).sum()) / int((account_summary.account_type == "EVALUATION").sum()) * 100) if not account_summary.empty and int((account_summary.account_type == "EVALUATION").sum()) else 0.0, "first_payout_rate": (int(account_summary.first_payout_timestamp.astype(bool).sum()) / int((account_summary.account_type == "QUALIFIED").sum()) * 100) if not account_summary.empty and int((account_summary.account_type == "QUALIFIED").sum()) else 0.0, "average_days_to_pass": float(pd.Series([(pd.Timestamp(row.pass_timestamp) - pd.Timestamp(row.purchase_timestamp)).total_seconds() / 86400 for _, row in account_summary[account_summary.pass_timestamp.astype(bool)].iterrows()]).mean()) if not account_summary.empty and account_summary.pass_timestamp.astype(bool).any() else 0.0, "average_days_to_first_payout": float(pd.Series([(pd.Timestamp(row.first_payout_timestamp) - pd.Timestamp(row.qualified_created_timestamp)).total_seconds() / 86400 for _, row in account_summary[account_summary.first_payout_timestamp.astype(bool)].iterrows()]).mean()) if not account_summary.empty and account_summary.first_payout_timestamp.astype(bool).any() else 0.0, "total_executed_trades": len(trades), "trades_per_month": len(trades) / max(len(months), 1), "maximum_account_drawdown": float(account_summary.max_marked_equity_drawdown.max()) if not account_summary.empty else 0.0, "profit_by_market": trades.groupby("market").net_pnl.sum().to_dict() if not trades.empty else {}, "largest_winning_trade": float(trades.net_pnl.max()) if not trades.empty else 0.0, "largest_losing_trade": float(trades.net_pnl.min()) if not trades.empty else 0.0, "payouts_by_market": {}, "risk_model_note": "Contracts are floor(250 / initial-stop dollar risk per micro), clipped at 10; MLL, DLG, and balance do not size positions.", "comparison_note": "This is a deterministic money-management replay, not a strategy optimization. The prior V12 fixed-contract journal used one active Evaluation and one active Qualified account; this model allows monthly/replacement Evaluations and multiple Qualified accounts."}
    final.update({"maximum_cash_drawdown": float(account_summary.max_cash_balance_drawdown.max()) if not account_summary.empty else 0.0, "maximum_marked_equity_drawdown": float(account_summary.max_marked_equity_drawdown.max()) if not account_summary.empty else 0.0, "dlg_breached_accounts": int((account_summary.dlg_breach_count > 0).sum()) if not account_summary.empty else 0, "mll_breached_accounts": int(account_summary.mll_breach_timestamp.astype(bool).sum()) if not account_summary.empty else 0, "active_after_mll_breach": int(((account_summary.mll_breach_timestamp.astype(bool)) & account_summary.final_state.isin({"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED", "QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED", "CENSORED_END_OF_DATA"})).sum()) if not account_summary.empty else 0, "preterminal_passes": int(((account_summary.pass_timestamp != "") & (pd.to_datetime(account_summary.pass_timestamp, utc=True, format="mixed") < end_exclusive)).sum()) if not account_summary.empty else 0, "preterminal_failures": int(((account_summary.failure_timestamp != "") & (pd.to_datetime(account_summary.failure_timestamp, utc=True, format="mixed") < end_exclusive)).sum()) if not account_summary.empty else 0, "censored_or_active_at_end": int(account_summary.final_state.isin({"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED", "QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED", "CENSORED_END_OF_DATA"}).sum()) if not account_summary.empty else 0})
    if not payouts.empty and not trades.empty:
        payout_accounts = set(payouts.account_id)
        final["payouts_by_market"] = trades[(trades.account_type == "QUALIFIED") & trades.account_id.isin(payout_accounts)].groupby("market").net_pnl.sum().to_dict()
    pd.DataFrame([final]).to_csv(root / "final_report.csv", index=False)
    # The requested output list does not include final_report.csv; remove it
    # after using it as an intermediate in-memory-compatible serialization.
    (root / "final_report.csv").unlink(missing_ok=True)
    trades.to_csv(root / "trades.csv", index=False)
    skipped.to_csv(root / "skipped_trades.csv", index=False)
    events.to_csv(root / "account_events.csv", index=False)
    account_summary.to_csv(root / "account_summary.csv", index=False)
    monthly.to_csv(root / "monthly_summary.csv", index=False)
    drawdown_rows = []
    for account in replay.accounts:
        drawdown_rows.append({"account_id": account.account_id, "account_type": account.kind, "maximum_cash_balance_drawdown": account.max_drawdown, "maximum_marked_equity_drawdown": account.max_equity_drawdown, "maximum_dlg_usage": account.max_dlg_usage, "maximum_mll_usage": account.max_mll_usage, "worst_marked_equity_timestamp": str(account.worst_equity_timestamp) if account.worst_equity_timestamp else "", "dlg_triggered": account.dlg_breach_count > 0, "dlg_breach_count": account.dlg_breach_count, "mll_triggered": account.mll_breach_timestamp is not None, "mll_breach_timestamp": str(account.mll_breach_timestamp) if account.mll_breach_timestamp else "", "final_state": account.state})
    drawdown = pd.DataFrame(drawdown_rows)
    drawdown.to_csv(root / "drawdown_summary.csv", index=False)
    lifecycle_rows = []
    for account in replay.accounts:
        account_events = events[events.account_id == account.account_id] if not events.empty else pd.DataFrame()
        lifecycle_rows.append({"account_id": account.account_id, "account_type": account.kind, "purchase_timestamp": str(account.purchase_timestamp), "pass_timestamp": str(account.pass_timestamp) if account.pass_timestamp else "", "failure_timestamp": str(account.failure_timestamp) if account.failure_timestamp else "", "qualified_created_timestamp": str(account.qualified_created_timestamp) if account.qualified_created_timestamp else "", "final_state": account.state, "subscription_paid": account.subscription_paid, "subscription_charge_events": int((account_events.event_reason == "subscription charged").sum()) if not account_events.empty else 0, "trade_count": account.trades_taken, "completed_trade_rows": int((trades.account_id == account.account_id).sum()) if not trades.empty else 0, "session_forced_closures": int(trades[(trades.account_id == account.account_id) & (trades.exit_reason == "session_forced_liquidation")].shape[0]) if not trades.empty else 0, "terminal_flatten_count": int(trades[(trades.account_id == account.account_id) & (trades.exit_reason == "end_of_data_flatten")].shape[0]) if not trades.empty else 0, "terminal_transition": bool((account_events.event_reason.str.contains("terminal|censored", case=False, na=False)).any()) if not account_events.empty else False, "censored_at_end": account.state == "CENSORED_END_OF_DATA", "dlg_breaches": account.dlg_breach_count, "mll_breached": account.mll_breach_timestamp is not None})
    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(root / "lifecycle_reconciliation.csv", index=False)
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>V13 risk-managed Alpha Zero 25K replay</title><style>body{{font-family:Arial;margin:2rem;max-width:1800px}}table{{border-collapse:collapse;font-size:10px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}code{{background:#f3f3f3;padding:2px}}</style></head><body><h1>V13 Risk-Managed Alpha Zero 25K Replay</h1><div class='warn'>Frozen strategy and cached Binance data. Selected {PORTFOLIO}, {TIMEFRAME}. Only monthly/replacement account purchases, fixed $250 initial-risk sizing, and the two-loss daily stop rule changed.</div><h2>Final answers</h2><ul><li>Evaluations purchased: {final['evaluations_purchased']}; passed: {final['evaluations_passed']}; failed: {final['evaluations_failed']}.</li><li>Qualified accounts: {final['qualified_accounts_created']}; payouts: {final['total_payouts']}; gross payout: ${final['gross_payout_amount']:,.2f}; trader payout: ${final['trader_payout_after_split']:,.2f}.</li><li>Subscriptions: ${final['subscriptions_paid']:,.2f}; external net cashflow: ${final['net_external_cashflow']:,.2f}.</li><li>Average active accounts: {final['average_active_accounts']:.2f}; maximum simultaneous accounts: {final['maximum_simultaneously_active_accounts']}.</li><li>Average contracts/trade: {final['average_contracts_traded']:.2f}; average initial risk: ${final['average_initial_risk_per_trade']:,.2f}.</li><li>Risk skips: {final['risk_skipped_trades']}; daily-stop skips: {final['daily_stop_skipped_trades']}; pass rate: {final['pass_rate']:.2f}%; first-payout rate: {final['first_payout_rate']:.2f}%.</li><li>Average days to pass: {final['average_days_to_pass']:.2f}; average days from Qualified creation to first payout: {final['average_days_to_first_payout']:.2f}.</li><li>Executed positions: {final['total_executed_trades']}; trades/month: {final['trades_per_month']:.2f}; maximum account drawdown: ${final['maximum_account_drawdown']:,.2f}.</li><li>Profit by market: {json.dumps(final['profit_by_market'], sort_keys=True)}.</li><li>Largest winner: ${final['largest_winning_trade']:,.2f}; largest loser: ${final['largest_losing_trade']:,.2f}.</li><li>The model is more conservative on per-trade loss than fixed-contract sizing, but it is not automatically “better”: performance must be compared on identical frozen signals, account rules, and lifecycle assumptions.</li></ul><h2>Account summary</h2>{account_summary.to_html(index=False, border=0)}<h2>Monthly summary</h2>{monthly.to_html(index=False, border=0)}<h2>Account events</h2>{events.to_html(index=False, border=0) if not events.empty else '<p>None</p>'}<h2>Executed trade journal</h2><p>All rows are in <code>trades.csv</code>; it includes signal/account IDs, sizing math, PnL, fees, balances, and exit reasons.</p><h2>Skipped trade journal</h2><p>All rows are in <code>skipped_trades.csv</code>; reasons are risk above $250, daily stop, contract limit, inactive account, or session cutoff.</p><h2>Limitations</h2><p>Results use the existing Binance proxy-to-Alpha mapping, cached history, frozen strategy signals, and the existing conservative execution assumptions. They are not live-trading evidence. The selected Portfolio C fallback excludes unresolved/short-history exposures.</p></body></html>"""
    report = report.replace("</body></html>", f"<h2>Repaired lifecycle checkpoints</h2><ul><li>Normal session-forced position closures: {int((trades.exit_reason == 'session_forced_liquidation').sum()) if not trades.empty else 0}; terminal flatten rows: {int((trades.exit_reason == 'end_of_data_flatten').sum()) if not trades.empty else 0}.</li><li>DLG-breached accounts: {final['dlg_breached_accounts']}; MLL-breached accounts: {final['mll_breached_accounts']}; active after MLL breach: {final['active_after_mll_breach']}.</li><li>Preterminal passes: {final['preterminal_passes']}; preterminal failures: {final['preterminal_failures']}; censored at end: {final['censored_or_active_at_end']}.</li><li>Maximum marked-equity drawdown across one account: ${final['maximum_marked_equity_drawdown']:,.2f}; corrected subscriptions: ${final['subscriptions_paid']:,.2f}; corrected external cashflow: ${final['net_external_cashflow']:,.2f}.</li></ul></body></html>")
    (root / "final_report.html").write_text(report, encoding="utf-8")
    return {"trades": len(trades), "skipped": len(skipped), "events": len(events), "accounts": len(account_summary), "payouts": len(payouts), "latest_candle": str(latest), "root": str(root)}


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, timeline, trading_times, session_closes, latest, end_exclusive, contexts, mark_prices = _load_signal_stream()
    last_prices = {market: mark_prices[market][max(mark_prices[market])] for market in MEMBERS}
    replay = Replay(rows, timeline, end_exclusive, last_prices, mark_prices, trading_times, session_closes).run()
    return _build_outputs(replay, latest, end_exclusive, root)


if __name__ == "__main__":
    print(run())
