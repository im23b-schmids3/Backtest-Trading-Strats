"""One deterministic 2025 Alpha Zero trader economics simulation.

Trading strategy and market data are imported unchanged.  This module only
models account replacement, one active Evaluation, at most one Qualified
account, billing, and external cashflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.research import v12_binance_proxy_prop_simulation as legacy
from fib_backtester.research import v12_economics_fixed as economics
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed
from fib_backtester.research.v12_contract_registry import build_synthetic_context


ROOT = Path("reports/v12_single_trader")
START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
TIMEFRAMES = fixed.TIMEFRAMES
MODES = ("MIRRORED", "QUALIFIED_PRIORITY", "ONE_ACCOUNT_ONLY_BASELINE")
PORTFOLIOS = {
    "ETH only": ["ETH"],
    "Portfolio C - BTC + ETH + Gold": ["BTC", "ETH", "Gold"],
    "All canonical Alpha exposures": sorted(fixed.CANONICAL_PROXIES),
}
POSITION_SIZES = (2, 5, 7, 10)


def _month(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m")


def _session(timestamp: pd.Timestamp) -> str:
    return fixed._session(timestamp)


def _scenario_label(account: str, portfolio: str, timeframe: str, size: int, mode: str, kind: str) -> str:
    return f"{account}|{portfolio}|{timeframe}|{size} micros|{mode}|{kind}"


def _transition(rows, timestamp, account_type, old, new, reason, amount=0.0):
    rows.append({"timestamp": str(timestamp), "account_type": account_type, "old_state": old, "new_state": new, "reason": reason, "related_cost_or_payout": float(amount)})


@dataclass
class Account:
    kind: str
    account_name: str
    spec: object
    account_type: str
    state: str
    balance: float
    initial_balance: float
    mll: float
    daily_profit: float = 0.0
    current_session: str | None = None
    last_timestamp: pd.Timestamp | None = None
    cycle_profit: float = 0.0
    cycle_days: dict[str, float] = field(default_factory=dict)
    winning_days: set[str] = field(default_factory=set)
    positions: dict[int, dict] = field(default_factory=dict)
    billing_next: pd.Timestamp | None = None
    billing_start: pd.Timestamp | None = None
    subscription_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    trades_taken: int = 0
    failed_count: int = 0
    passed_count: int = 0
    qualified_payouts: int = 0
    gross_payout: float = 0.0
    trader_payout: float = 0.0
    first_payouts: int = 0
    first_payout_timestamp: pd.Timestamp | None = None


class SingleTrader:
    def __init__(self, account_name: str, portfolio: str, timeframe: str, size: int, mode: str, trades: list[dict]):
        self.account_name = account_name
        self.portfolio = portfolio
        self.timeframe = timeframe
        self.size = size
        self.mode = mode
        self.spec = fixed.ACCOUNT_SPECS[account_name]
        self.trades = trades
        self.transitions = []
        self.months = {_month(START): self._blank_month()}
        for month in pd.period_range("2025-01", "2025-12", freq="M").astype(str):
            self.months[month] = self._blank_month()
        self.eval: Account | None = None
        self.qualified: Account | None = None
        self.pending_passes = 0
        self.total_evaluations_purchased = 0
        self.total_evaluations_failed = 0
        self.total_evaluations_passed = 0
        self.total_qualified_started = 0
        self.total_qualified_failed = 0
        self.total_trades = 0
        self.trade_timestamps: list[pd.Timestamp] = []
        self.eval_pass_timestamps: list[pd.Timestamp] = []
        self.payout_timestamps: list[pd.Timestamp] = []

    @staticmethod
    def _blank_month():
        return {"evaluations_purchased": 0, "evaluations_failed": 0, "evaluations_passed": 0, "qualified_started": 0, "qualified_failed": 0, "trades_taken": 0, "evaluation_pnl": 0.0, "qualified_pnl": 0.0, "subscription_cost": 0.0, "gross_payout": 0.0, "trader_payout": 0.0}

    def _add(self, timestamp, field, value=1.0):
        month = _month(timestamp)
        if month in self.months:
            self.months[month][field] += value

    def _state(self, kind):
        account = self.eval if kind == "EVALUATION" else self.qualified
        return account.state if account is not None else "NONE"

    def _buy_evaluation(self, timestamp, reason):
        if self.eval is not None:
            raise AssertionError("attempted to create a second active Evaluation")
        label = _scenario_label(self.account_name, self.portfolio, self.timeframe, self.size, self.mode, "Evaluation")
        self.eval = Account("EVALUATION", self.account_name, self.spec, label, "EVALUATION_ACTIVE", self.spec.account_size, self.spec.account_size, self.spec.account_size - self.spec.mll_amount, last_timestamp=timestamp, billing_start=timestamp, billing_next=timestamp)
        self.total_evaluations_purchased += 1
        self._add(timestamp, "evaluations_purchased")
        _transition(self.transitions, timestamp, label, "NONE", "EVALUATION_ACTIVE", reason, self.spec.subscription)
        self._charge_subscription(timestamp)

    def _charge_subscription(self, timestamp):
        if self.eval is None or self.eval.billing_next is None:
            return
        while self.eval.billing_next <= timestamp and self.eval.state in {"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"}:
            charge_time = self.eval.billing_next
            self.eval.subscription_cost += self.spec.subscription
            self._add(charge_time, "subscription_cost", self.spec.subscription)
            _transition(self.transitions, charge_time, self.eval.account_type, self.eval.state, self.eval.state, "Evaluation monthly subscription rebill", self.spec.subscription)
            self.eval.billing_next = charge_time + pd.DateOffset(months=1)

    def _close_eval(self, timestamp, state, reason):
        if self.eval is None:
            return
        old = self.eval.state
        self.eval.state = state
        _transition(self.transitions, timestamp, self.eval.account_type, old, state, reason)

    def _start_qualified(self, timestamp, balance):
        if self.qualified is not None and self.qualified.state not in {"QUALIFIED_FAILED", "QUALIFIED_CLOSED"}:
            self.pending_passes += 1
            return False
        label = _scenario_label(self.account_name, self.portfolio, self.timeframe, self.size, self.mode, "Qualified")
        self.qualified = Account("QUALIFIED", self.account_name, self.spec, label, "QUALIFIED_ACTIVE", balance, balance, balance - self.spec.mll_amount, last_timestamp=timestamp)
        self.total_qualified_started += 1
        self._add(timestamp, "qualified_started")
        _transition(self.transitions, timestamp, label, "NONE", "QUALIFIED_ACTIVE", "passed Evaluation became Qualified; Zero activation fee", 0.0)
        return True

    def _handle_eval_pass(self, timestamp):
        if self.eval is None:
            return
        evaluation = self.eval
        self.total_evaluations_passed += 1
        self.eval_pass_timestamps.append(timestamp)
        self._add(timestamp, "evaluations_passed")
        evaluation.passed_count += 1
        self._close_eval(timestamp, "EVALUATION_PASSED", "profit target reached; Evaluation billing stopped")
        self._flatten(evaluation, timestamp, "evaluation_pass_flatten")
        balance = evaluation.balance
        _transition(self.transitions, timestamp, evaluation.account_type, "EVALUATION_PASSED", "CLOSED", "Evaluation lifecycle ended after pass")
        self.eval = None
        if self.qualified is None or self.qualified.state in {"QUALIFIED_FAILED", "QUALIFIED_CLOSED"}:
            self._start_qualified(timestamp + pd.Timedelta(nanoseconds=1), balance)
            if self.mode != "ONE_ACCOUNT_ONLY_BASELINE":
                self._buy_evaluation(timestamp + pd.Timedelta(nanoseconds=2), "new Evaluation after pass")
        else:
            self.pending_passes += 1
            _transition(self.transitions, timestamp, evaluation.account_type, "EVALUATION_PASSED", "CLOSED", "Qualified already active; conservative no second Qualified", 0.0)

    def _handle_eval_failure(self, timestamp, reason):
        if self.eval is None:
            return
        self.total_evaluations_failed += 1
        self._add(timestamp, "evaluations_failed")
        self.eval.failed_count += 1
        self._flatten(self.eval, timestamp, reason)
        self._close_eval(timestamp, "EVALUATION_FAILED", reason + "; billing stopped")
        self.eval = None
        self._buy_evaluation(timestamp + pd.Timedelta(nanoseconds=1), "replacement Evaluation after failure")

    def _handle_qualified_failure(self, timestamp):
        if self.qualified is None:
            return
        self.total_qualified_failed += 1
        self._add(timestamp, "qualified_failed")
        self._flatten(self.qualified, timestamp, "qualified_failure_flatten")
        old = self.qualified.state
        self.qualified.state = "QUALIFIED_FAILED"
        _transition(self.transitions, timestamp, self.qualified.account_type, old, "QUALIFIED_FAILED", "Maximum Loss Limit; Qualified closed")
        _transition(self.transitions, timestamp, self.qualified.account_type, "QUALIFIED_FAILED", "CLOSED", "Qualified lifecycle ended after breach")
        self.qualified = None
        if self.eval is None:
            self._buy_evaluation(timestamp + pd.Timedelta(nanoseconds=1), "new Evaluation after Qualified failure")

    def _flatten(self, account: Account, timestamp, reason):
        for signal_id, position in list(account.positions.items()):
            if position["remaining"] <= 0:
                account.positions.pop(signal_id, None)
                continue
            leg = fixed._flatten_leg(position["trade"], position["last_price"], timestamp, position["remaining"], reason)
            self._apply_exit(account, position["trade"], leg, timestamp, forced=True)
            account.positions.pop(signal_id, None)

    def _apply_exit(self, account: Account, trade: dict, leg: dict, timestamp, forced=False):
        account.balance += leg["net"]
        account.gross_pnl += leg["gross"]
        account.net_pnl += leg["net"]
        account.fees += leg["fee"]
        account.daily_profit += leg["net"]
        self._add(timestamp, "evaluation_pnl" if account.kind == "EVALUATION" else "qualified_pnl", leg["net"])
        if account.kind == "QUALIFIED":
            account.cycle_profit += leg["net"]

    def _finish_session(self, account: Account, timestamp, new_session: str):
        if account.current_session is None or account.current_session == new_session:
            account.current_session = new_session
            return
        if account.kind == "QUALIFIED":
            if account.daily_profit > 0:
                account.cycle_days[account.current_session] = account.daily_profit
            if account.daily_profit >= legacy.WINNING_DAY_MINIMUM:
                account.winning_days.add(account.current_session)
            if len(account.winning_days) >= legacy.WINNING_DAYS_REQUIRED and account.cycle_profit > 0:
                consistency = max(account.cycle_days.values(), default=0.0) / account.cycle_profit
                request = min(0.50 * account.cycle_profit, self.spec.payout_max)
                if consistency <= legacy.CONSISTENCY_LIMIT and request >= legacy.WINNING_DAY_MINIMUM and account.balance - request > account.mll:
                    account.balance -= request
                    account.qualified_payouts += 1
                    account.gross_payout += request
                    account.trader_payout += request * legacy.PAYOUT_SPLIT
                    self._add(timestamp, "gross_payout", request)
                    self._add(timestamp, "trader_payout", request * legacy.PAYOUT_SPLIT)
                    if account.first_payouts == 0:
                        account.first_payouts = 1
                        account.first_payout_timestamp = timestamp
                        self.payout_timestamps.append(timestamp)
                    _transition(self.transitions, timestamp, account.account_type, account.state, account.state, "Qualified payout request filled; account remains active", request * legacy.PAYOUT_SPLIT)
                    account.cycle_profit = 0.0
                    account.cycle_days.clear()
                    account.winning_days.clear()
        account.mll = min(account.initial_balance, max(account.mll, account.balance - self.spec.mll_amount))
        account.daily_profit = 0.0
        if account.state.endswith("DAILY_LOCKED"):
            old = account.state
            account.state = "EVALUATION_ACTIVE" if account.kind == "EVALUATION" else "QUALIFIED_ACTIVE"
            _transition(self.transitions, timestamp, account.account_type, old, account.state, "next official trading day opened")
        account.current_session = new_session

    def _before_event(self, account: Account | None, timestamp):
        if account is None:
            return
        self._charge_subscription(timestamp)
        if account.last_timestamp is not None and account.positions:
            cutoff = fixed._next_session_close_after(account.last_timestamp)
            if cutoff < timestamp:
                self._flatten(account, cutoff, "session_forced_liquidation")
        session = _session(timestamp)
        self._finish_session(account, timestamp, session)
        account.last_timestamp = timestamp

    def _eligible_accounts(self):
        if self.mode == "MIRRORED":
            return [account for account in (self.eval, self.qualified) if account is not None]
        if self.qualified is not None:
            return [self.qualified]
        return [self.eval] if self.eval is not None else []

    def _accept(self, account: Account, signal_id: int, trade: dict, timestamp):
        if account.state not in {"EVALUATION_ACTIVE", "QUALIFIED_ACTIVE"}:
            return
        local = pd.Timestamp(timestamp).tz_convert("America/New_York")
        if local.hour == 17 or (local.hour == 16 and local.minute >= 20):
            return
        current_contracts = sum(position["remaining"] for position in account.positions.values())
        same_market = any(position["trade"]["market"] == trade["market"] for position in account.positions.values())
        max_contracts = self.spec.max_micros_evaluation if account.kind == "EVALUATION" else self.spec.max_micros_qualified_initial
        if same_market or current_contracts + trade["contracts"] > max_contracts:
            return
        account.balance -= trade["entry_fee"]
        account.daily_profit -= trade["entry_fee"]
        account.net_pnl -= trade["entry_fee"]
        account.fees += trade["entry_fee"]
        self._add(timestamp, "evaluation_pnl" if account.kind == "EVALUATION" else "qualified_pnl", -trade["entry_fee"])
        account.positions[signal_id] = {"trade": trade, "remaining": trade["contracts"], "last_price": trade["entry"]}
        account.trades_taken += 1
        self.total_trades += 1
        self.trade_timestamps.append(timestamp)
        self._add(timestamp, "trades_taken")

    def _exit_for_account(self, account: Account, signal_id: int, leg_index: int, trade: dict, timestamp):
        position = account.positions.get(signal_id)
        if position is None:
            return
        leg = trade["legs"][leg_index]
        position["last_price"] = leg["price"]
        self._apply_exit(account, trade, leg, timestamp)
        position["remaining"] -= leg["quantity"]
        if position["remaining"] <= 0:
            account.positions.pop(signal_id, None)
        if account.daily_profit <= -self.spec.daily_loss_guard:
            old = account.state
            account.state = "EVALUATION_DAILY_LOCKED" if account.kind == "EVALUATION" else "QUALIFIED_DAILY_LOCKED"
            _transition(self.transitions, timestamp, account.account_type, old, account.state, "Daily Loss Guard; positions flattened")
            self._flatten(account, timestamp, "daily_loss_guard")
        if account.balance <= account.mll:
            if account.kind == "EVALUATION":
                self._handle_eval_failure(timestamp, "Maximum Loss Limit")
            else:
                self._handle_qualified_failure(timestamp)
        elif account.kind == "EVALUATION" and account.balance >= self.spec.account_size + self.spec.target:
            self._handle_eval_pass(timestamp)

    def run(self):
        self._buy_evaluation(START, "initial Evaluation purchase")
        events = []
        for signal_id, trade in enumerate(self.trades):
            if not (START <= trade["entry_timestamp"] < END):
                continue
            events.append((trade["entry_timestamp"], 1, signal_id, "entry", None, trade))
            for leg_index, leg in enumerate(trade["legs"]):
                timestamp = pd.Timestamp(leg["timestamp"])
                if START <= timestamp < END:
                    events.append((timestamp, 0, signal_id, "exit", leg_index, trade))
        events.sort(key=lambda row: (row[0], row[1], row[2]))
        for timestamp, kind, signal_id, event_type, leg_index, trade in events:
            self._before_event(self.eval, timestamp)
            self._before_event(self.qualified, timestamp)
            if event_type == "entry":
                for account in self._eligible_accounts():
                    self._accept(account, signal_id, trade, timestamp)
            else:
                for account in list((self.eval, self.qualified)):
                    if account is not None:
                        self._exit_for_account(account, signal_id, leg_index, trade, timestamp)
        for account in list((self.eval, self.qualified)):
            if account is None:
                continue
            self._charge_subscription(END - pd.Timedelta(nanoseconds=1))
            if account.positions:
                self._flatten(account, END, "end_of_period_flatten")
            self._finish_session(account, END, _session(END))
        return self._outputs()

    def _outputs(self):
        # Fill monthly account-state snapshots from the single chronological
        # transition stream.  No independent historical paths are combined.
        current_eval = "ACTIVE" if self.eval is not None else "NONE"
        current_qualified = "ACTIVE" if self.qualified is not None else "NONE"
        monthly = []
        cumulative = 0.0
        for month in [f"2025-{i:02d}" for i in range(1, 13)]:
            data = self.months[month]
            month_end = pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            eval_status = "NONE"
            qualified_status = "NONE"
            for transition in sorted(self.transitions, key=lambda row: pd.Timestamp(row["timestamp"])):
                if pd.Timestamp(transition["timestamp"]) > month_end:
                    break
                if transition["account_type"].endswith("|Evaluation"):
                    eval_status = transition["new_state"]
                elif transition["account_type"].endswith("|Qualified"):
                    qualified_status = transition["new_state"]
            if eval_status in {"EVALUATION_PASSED", "EVALUATION_FAILED", "CLOSED"}:
                eval_status = "NONE"
            if qualified_status in {"QUALIFIED_FAILED", "CLOSED"}:
                qualified_status = "NONE"
            cumulative += data["trader_payout"] - data["subscription_cost"]
            monthly.append({"account": self.account_name, "portfolio": self.portfolio, "timeframe": self.timeframe, "position_size": self.size, "signal_allocation_mode": self.mode, "month": month, "active_evaluation_status": eval_status, "active_qualified_status": qualified_status, **data, "net_monthly_cashflow": data["trader_payout"] - data["subscription_cost"], "cumulative_cashflow": cumulative})
        total_sub = sum(account.subscription_cost for account in (self.eval, self.qualified) if account is not None)
        # Closed account objects are not retained, so reconstruct costs and
        # PnL from monthly ledgers for the scenario-level summary.
        total_sub = sum(self.months[month]["subscription_cost"] for month in self.months)
        gross_payout = sum(self.months[month]["gross_payout"] for month in self.months)
        trader_payout = sum(self.months[month]["trader_payout"] for month in self.months)
        eval_pnl = sum(self.months[month]["evaluation_pnl"] for month in self.months)
        qualified_pnl = sum(self.months[month]["qualified_pnl"] for month in self.months)
        transitions = pd.DataFrame(self.transitions)
        pass_times = self.eval_pass_timestamps
        payout_times = self.payout_timestamps
        gaps = [b - a for a, b in zip(self.trade_timestamps, self.trade_timestamps[1:])]
        longest_trade_gap = max(gaps, default=pd.Timedelta(0)).total_seconds() / 86400
        longest_pass_gap = max([b - a for a, b in zip(pass_times, pass_times[1:])], default=pd.Timedelta(days=365)).total_seconds() / 86400
        longest_payout_gap = max([b - a for a, b in zip(payout_times, payout_times[1:])], default=pd.Timedelta(days=365)).total_seconds() / 86400
        yearly = {"account": self.account_name, "portfolio": self.portfolio, "timeframe": self.timeframe, "position_size": self.size, "signal_allocation_mode": self.mode, "period_start": str(START), "period_end": str(END), "evaluations_purchased": self.total_evaluations_purchased, "evaluation_subscription_cost": total_sub, "evaluations_passed": self.total_evaluations_passed, "evaluations_failed": self.total_evaluations_failed, "evaluations_active_at_year_end": int(self.eval is not None), "qualified_accounts_created": self.total_qualified_started, "qualified_failures": self.total_qualified_failed, "first_payouts_received": len(payout_times), "gross_payout_requested": gross_payout, "trader_payout_received": trader_payout, "net_external_cashflow": trader_payout - total_sub, "ending_evaluation_state": self.eval.state if self.eval else "NONE", "ending_qualified_state": self.qualified.state if self.qualified else "NONE", "total_trades": self.total_trades, "average_trades_per_month": self.total_trades / 12, "longest_period_without_trade_days": longest_trade_gap, "longest_period_without_evaluation_pass_days": longest_pass_gap, "longest_period_without_payout_days": longest_payout_gap, "cost_per_evaluation_started": total_sub / self.total_evaluations_purchased if self.total_evaluations_purchased else np.nan, "cost_per_evaluation_passed": total_sub / self.total_evaluations_passed if self.total_evaluations_passed else np.nan, "payout_per_qualified_account": trader_payout / self.total_qualified_started if self.total_qualified_started else 0.0, "payout_to_cost_ratio": trader_payout / total_sub if total_sub else np.nan, "annual_roi": (trader_payout - total_sub) / total_sub if total_sub else np.nan, "evaluation_trading_pnl": eval_pnl, "qualified_trading_pnl": qualified_pnl, "months_profitable": sum(self.months[m]["trader_payout"] - self.months[m]["subscription_cost"] > 0 for m in self.months), "months_unprofitable": sum(self.months[m]["trader_payout"] - self.months[m]["subscription_cost"] < 0 for m in self.months), "break_even_month": next((m for m in self.months if sum(self.months[x]["trader_payout"] - self.months[x]["subscription_cost"] for x in self.months if x <= m) >= 0), ""), "continuous_farming_improved": False}
        return pd.DataFrame(monthly), pd.DataFrame([yearly]), transitions


def _recent_trades() -> dict[tuple[str, str], list[dict]]:
    streams, _, _ = economics._prepare_recent_streams()
    result = {}
    for key, stream in streams.items():
        market, timeframe = key
        if market not in fixed.CANONICAL_PROXIES:
            continue
        result[key] = stream["trades"]
    return result


def _scenario_trades(market_trades, members, timeframe, size, conversion_context=None):
    trades = []
    for market in members:
        raw = market_trades.get((market, timeframe))
        if raw is None:
            continue
        for row in raw.to_dict("records"):
            trade = fixed._prepare_trade(row, market, size, (conversion_context or {}).get(market))
            if START <= trade["entry_timestamp"] < END:
                trades.append(trade)
    return sorted(trades, key=lambda trade: (trade["entry_timestamp"], trade["market"], trade["setup_id"]))


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    market_trades = _recent_trades()
    first_proxy_prices = {}
    for (market, _timeframe), frame in market_trades.items():
        if frame is not None and not frame.empty:
            ordered = frame.sort_values("fill_timestamp")
            first_proxy_prices.setdefault(market, float(ordered.iloc[0]["entry_price"]))
    conversion_context = build_synthetic_context(first_proxy_prices)
    monthly_rows, yearly_rows, transition_rows = [], [], []
    for account_name in ("25K Zero", "50K Zero"):
        for portfolio, members in PORTFOLIOS.items():
            for timeframe in TIMEFRAMES:
                for size in POSITION_SIZES:
                    trades = _scenario_trades(market_trades, members, timeframe, size, conversion_context)
                    for mode in MODES:
                        simulation = SingleTrader(account_name, portfolio, timeframe, size, mode, trades)
                        monthly, yearly, transitions = simulation.run()
                        monthly_rows.extend(monthly.to_dict("records"))
                        yearly_rows.extend(yearly.to_dict("records"))
                        if not transitions.empty:
                            transition_rows.extend(transitions.to_dict("records"))
    monthly = pd.DataFrame(monthly_rows)
    yearly = pd.DataFrame(yearly_rows)
    transitions = pd.DataFrame(transition_rows, columns=["timestamp", "account_type", "old_state", "new_state", "reason", "related_cost_or_payout"])
    # Compare modes without pooling unrelated timelines: each row remains one
    # deterministic scenario; deltas are only against its matching baseline.
    keys = ["account", "portfolio", "timeframe", "position_size"]
    baseline = yearly[yearly.signal_allocation_mode == "ONE_ACCOUNT_ONLY_BASELINE"][keys + ["net_external_cashflow", "first_payouts_received"]].rename(columns={"net_external_cashflow": "baseline_net_cashflow", "first_payouts_received": "baseline_first_payouts"})
    comparison = yearly.merge(baseline, on=keys, how="left")
    comparison["net_cashflow_delta_vs_one_account_only"] = comparison["net_external_cashflow"] - comparison["baseline_net_cashflow"]
    comparison["first_payout_delta_vs_one_account_only"] = comparison["first_payouts_received"] - comparison["baseline_first_payouts"]
    comparison.to_csv(root / "single_trader_scenario_comparison.csv", index=False)
    monthly.to_csv(root / "single_trader_monthly.csv", index=False)
    yearly.to_csv(root / "single_trader_year_summary.csv", index=False)
    transitions.to_csv(root / "single_trader_account_transitions.csv", index=False)
    _write_report(root, monthly, yearly, comparison)
    return {"scenarios": len(yearly), "months": len(monthly), "transitions": len(transitions), "root": str(root)}


def _write_report(root: Path, monthly: pd.DataFrame, yearly: pd.DataFrame, comparison: pd.DataFrame):
    def table(frame):
        return frame.to_html(index=False, border=0) if not frame.empty else "<p>None</p>"
    text = f"""<!doctype html><html><head><meta charset='utf-8'><title>Single Alpha Zero Trader 2025</title><style>body{{font-family:Arial;margin:2rem;max-width:1700px}}table{{border-collapse:collapse;font-size:10px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}</style></head><body><h1>One Deterministic Alpha Zero Trader — 2025</h1><div class='warn'><b>No multi-path aggregation.</b> Each scenario is one chronological trader from {START} through {END}. Strategy, data, proxies, execution assumptions, and official rules were unchanged. Mirrored, Qualified-priority, and one-account-only baseline modes are separate scenarios. Qualified accounts remain active after payout.</div><h2>Scenario comparison</h2>{table(comparison)}<h2>Year summaries</h2>{table(yearly)}<h2>Monthly cashflow</h2>{table(monthly.head(100))}<p>Verified official sources: Alpha <a href='https://help.alpha-futures.com/en/articles/11771813-zero-account-overview'>Zero Account Overview</a>, <a href='https://help.alpha-futures.com/en/articles/9492068-monthly-subscription'>Monthly Subscription</a>, <a href='https://help.alpha-futures.com/en/articles/9492051-payout-policy'>Payout Policy</a>, <a href='https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard'>Daily Loss Guard</a>, and <a href='https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll'>MLL</a>.</p></body></html>"""
    (root / "single_trader_final_report.html").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    print(run())
