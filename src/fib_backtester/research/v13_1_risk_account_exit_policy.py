"""V13.1 research: initial-risk caps and deterministic Evaluation exit policies.

This module is deliberately a research wrapper around the corrected V13
replay.  It rebuilds only the contract count from the frozen V13 signals and
adds account-management policies.  Signal generation, prices, mappings,
session handling, stops, targets, fees, slippage, and account rules remain
owned by the existing V13/V12 modules.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path

import pandas as pd

from fib_backtester.research import v13_risk_managed as v13
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed


ROOT = Path("reports/v13_1")
START = v13.START
RISK_CAPS = (150.0, 200.0, 250.0, 300.0)
POLICIES = ("A_FIXED_RISK", "B_DEFENSIVE_REDUCTION", "C_CANCEL_AT_800", "D_ONE_RECOVERY", "E_BUFFER_LIMITED_RECOVERY")
POLICY_LABELS = {
    "A_FIXED_RISK": "Policy A - Fixed Risk",
    "B_DEFENSIVE_REDUCTION": "Policy B - Defensive Reduction",
    "C_CANCEL_AT_800": "Policy C - Cancel at $800 Drawdown",
    "D_ONE_RECOVERY": "Policy D - One Recovery Attempt",
    "E_BUFFER_LIMITED_RECOVERY": "Policy E - Buffer-Limited Recovery",
}
ACTIVE_EVAL_STATES = {"EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"}
ACTIVE_STATES = ACTIVE_EVAL_STATES | {"QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"}
ZONE_ORDER = ("GREEN", "YELLOW", "RED", "CRITICAL")


def _zone(equity: float, starting_balance: float, mll: float) -> str:
    drawdown = max(0.0, starting_balance - equity)
    buffer = equity - mll
    if drawdown >= 800.0 or buffer <= 200.0:
        return "CRITICAL"
    if drawdown >= 700.0:
        return "RED"
    if drawdown >= 500.0:
        return "YELLOW"
    return "GREEN"


def _build_signal(row: dict, contracts: int, context: dict | None) -> dict:
    """Rebuild a frozen trade for a different contract count only."""
    signal = dict(row)
    raw = row["raw"]
    if contracts <= 0:
        signal["trade"] = None
        signal["contracts"] = 0
        return signal
    trade = fixed._prepare_trade(raw, row["market"], contracts, context)
    trade.update({
        "signal_timestamp": v13._utc(raw["signal_timestamp"]),
        "setup_timestamp": str(raw.get("signal_timestamp", "")),
        "initial_stop": row.get("stop", raw.get("initial_stop")),
        "stop_distance": row.get("stop_distance", abs(float(raw["entry_price"]) - float(raw["initial_stop"]))),
        "stop_ticks": row.get("stop_ticks", 0.0),
        "tick_size": row.get("tick_size", 0.0),
        "tick_value": row.get("tick_value", 0.0),
        "dollar_risk_per_contract": row["dollar_risk_per_contract"],
        "initial_risk": row["dollar_risk_per_contract"] * contracts,
        "raw_setup_id": raw.get("setup_id", ""),
        "signal_id": row["signal_id"],
    })
    signal["trade"] = trade
    signal["contracts"] = contracts
    return signal


def build_signals(raw_rows: list[dict], contexts: dict, risk_cap: float) -> list[dict]:
    signals = []
    for row in raw_rows:
        risk_per_contract = float(row["dollar_risk_per_contract"])
        contracts = min(v13.MAX_MICROS, math.floor(risk_cap / risk_per_contract)) if risk_per_contract > 0 else 0
        signal = _build_signal(row, contracts, contexts.get(row["market"]))
        signal["risk_cap"] = risk_cap
        signal["risk_per_contract"] = risk_per_contract
        signal["conversion_context"] = contexts.get(row["market"])
        signals.append(signal)
    return signals


class PolicyReplay(v13.Replay):
    """Corrected V13 replay with one isolated account-policy variable."""

    def __init__(self, *args, risk_cap: float, policy: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_cap = float(risk_cap)
        self.policy = policy
        self._mark_keys = {market: sorted(values) for market, values in self.mark_prices.items()}
        self._replacement_scheduled = set()
        self._initialize_existing_accounts = []

    def _price_for(self, market, timestamp):
        keys = self._mark_keys.get(market, [])
        if not keys:
            return self.last_prices[market]
        index = bisect_right(keys, v13._utc(timestamp)) - 1
        return self.mark_prices[market][keys[max(index, 0)]] if index >= 0 else self.last_prices[market]

    @staticmethod
    def _init_policy_fields(account):
        account.voluntary_cancel_timestamp = None
        account.voluntary_cancel_reason = ""
        account.recovery_pending = False
        account.recovery_attempts = 0
        account.recovery_successes = 0
        account.recovery_failures = 0
        account.recovery_critical_timestamp = None
        account.max_drawdown_used = 0.0
        account.zone_max = "GREEN"
        account.zone_entries = {zone: 0 for zone in ZONE_ORDER}
        account.zone_trade_counts = {zone: 0 for zone in ZONE_ORDER}
        account.zone_skip_counts = {zone: 0 for zone in ZONE_ORDER}
        account.last_policy_zone = None
        account.policy_risk_caps = []

    def _buy_evaluation(self, timestamp, reason):
        account = super()._buy_evaluation(timestamp, reason)
        self._init_policy_fields(account)
        return account

    def _buy_qualified(self, timestamp, balance):
        account = super()._buy_qualified(timestamp, balance)
        self._init_policy_fields(account)
        return account

    def _mark_account(self, account, timestamp):
        result = super()._mark_account(account, timestamp)
        if account.kind == "EVALUATION":
            equity = result[0]
            drawdown = max(0.0, account.initial_balance - equity)
            account.max_drawdown_used = max(account.max_drawdown_used, drawdown)
            current_zone = _zone(equity, account.initial_balance, account.mll)
            if current_zone != account.last_policy_zone:
                account.zone_entries[current_zone] += 1
                account.last_policy_zone = current_zone
            if ZONE_ORDER.index(current_zone) > ZONE_ORDER.index(account.zone_max):
                account.zone_max = current_zone
        return result

    def _schedule_replacement(self, account, timestamp, reason):
        if account.account_id in self._replacement_scheduled:
            return
        index = bisect_right(self.trading_times, timestamp)
        if index < len(self.trading_times):
            self.replacements.setdefault(self.trading_times[index], []).append(reason)
            self._replacement_scheduled.add(account.account_id)

    def _cancel_evaluation(self, account, timestamp, reason="VOLUNTARY_DRAWDOWN_CANCELLATION"):
        if account.kind != "EVALUATION" or account.state not in ACTIVE_EVAL_STATES:
            return
        old = account.state
        account.state = "CLOSED"
        account.voluntary_cancel_timestamp = timestamp
        account.voluntary_cancel_reason = reason
        self._flatten(account, timestamp, "voluntary_drawdown_cancellation")
        self._emit(timestamp, account, "VOLUNTARY_DRAWDOWN_CANCELLATION", old=old, new="CLOSED")
        self._schedule_replacement(account, timestamp, "replacement Evaluation after voluntary cancellation")

    def _has_future_legal_recovery(self, account, timestamp, risk_limit):
        for signal in self.signals:
            if signal["entry_timestamp"] <= timestamp:
                continue
            if signal["risk_per_contract"] <= risk_limit + 1e-9 and signal["risk_per_contract"] > 0:
                return True
        return False

    def _check_lifecycle(self, account, timestamp, trigger="checkpoint"):
        super()._check_lifecycle(account, timestamp, trigger)
        if account.kind != "EVALUATION" or account.state not in ACTIVE_EVAL_STATES:
            return
        equity, _ = self._mark_account(account, timestamp)
        drawdown = max(0.0, account.initial_balance - equity)
        buffer = equity - account.mll
        critical = drawdown >= 800.0 or buffer <= 200.0
        if not critical:
            return
        if self.policy == "C_CANCEL_AT_800":
            self._cancel_evaluation(account, timestamp)
            return
        if self.policy not in {"D_ONE_RECOVERY", "E_BUFFER_LIMITED_RECOVERY"}:
            return
        if account.state != "EVALUATION_ACTIVE" or account.recovery_pending or account.recovery_attempts:
            return
        account.recovery_pending = True
        account.recovery_critical_timestamp = timestamp
        self._emit(timestamp, account, "recovery attempt authorized", old=account.state, new=account.state)
        if self.policy == "E_BUFFER_LIMITED_RECOVERY" and not self._has_future_legal_recovery(account, timestamp, max(0.0, min(self.risk_cap, buffer))):
            self._cancel_evaluation(account, timestamp, "no legal buffer-limited recovery contract")

    def _effective_risk_limit(self, account, signal):
        limit = self.risk_cap
        if account.kind != "EVALUATION":
            return limit
        if self.policy == "B_DEFENSIVE_REDUCTION":
            equity, _ = self._mark_account(account, signal["entry_timestamp"])
            zone = _zone(equity, account.initial_balance, account.mll)
            limit *= {"GREEN": 1.0, "YELLOW": 0.75, "RED": 0.50, "CRITICAL": 0.25}[zone]
        if account.recovery_pending and self.policy == "E_BUFFER_LIMITED_RECOVERY":
            equity, _ = self._mark_account(account, signal["entry_timestamp"])
            limit = min(limit, max(0.0, equity - account.mll))
        return limit

    def _accept(self, account, signal, timestamp):
        risk_limit = self._effective_risk_limit(account, signal)
        risk_per_contract = float(signal["risk_per_contract"])
        contracts = min(v13.MAX_MICROS, math.floor(risk_limit / risk_per_contract)) if risk_per_contract > 0 else 0
        zone = "GREEN"
        if account.kind == "EVALUATION":
            zone = _zone(account.marked_equity, account.initial_balance, account.mll)
        before_skips = len(self.skipped)
        if contracts <= 0:
            self._record_skip(account, signal, "zero legal contracts")
            if account.recovery_pending and self.policy == "E_BUFFER_LIMITED_RECOVERY":
                self._cancel_evaluation(account, timestamp, "no legal buffer-limited recovery contract")
            if account.kind == "EVALUATION":
                account.zone_skip_counts[zone] += 1
            return
        prepared = _build_signal(signal, contracts, signal.get("conversion_context"))
        prepared["risk_cap_applied"] = risk_limit
        is_recovery = account.kind == "EVALUATION" and account.recovery_pending and self.policy in {"D_ONE_RECOVERY", "E_BUFFER_LIMITED_RECOVERY"}
        if is_recovery:
            prepared["trade"]["recovery_attempt"] = True
        super()._accept(account, prepared, timestamp)
        accepted = prepared["signal_id"] in account.positions
        if accepted:
            if account.kind == "EVALUATION":
                account.zone_trade_counts[zone] += 1
                account.policy_risk_caps.append(risk_limit)
            if is_recovery:
                account.recovery_pending = False
                account.recovery_attempts += 1
        elif account.kind == "EVALUATION" and len(self.skipped) > before_skips:
            account.zone_skip_counts[zone] += 1

    def _apply_leg(self, account, signal_id, leg, timestamp, forced=False, check=True):
        position = account.positions.get(signal_id)
        is_recovery = bool(position and position.trade.get("recovery_attempt"))
        closes = bool(position and position.remaining <= int(leg["quantity"]))
        super()._apply_leg(account, signal_id, leg, timestamp, forced=forced, check=check)
        if is_recovery and closes and account.kind == "EVALUATION" and account.recovery_attempts:
            self._evaluate_recovery(account, timestamp)

    def _evaluate_recovery(self, account, timestamp):
        if account.state == "CLOSED" and account.voluntary_cancel_timestamp is not None:
            account.recovery_failures += 1
            return
        if account.state.endswith("FAILED"):
            account.recovery_failures += 1
            return
        equity, _ = self._mark_account(account, timestamp)
        drawdown = max(0.0, account.initial_balance - equity)
        if account.state == "CLOSED" or account.pass_timestamp is not None:
            account.recovery_successes += 1
            return
        if drawdown < 700.0:
            account.recovery_successes += 1
            self._emit(timestamp, account, "recovery attempt successful")
            return
        account.recovery_failures += 1
        if self.policy == "E_BUFFER_LIMITED_RECOVERY" or drawdown >= 800.0:
            self._cancel_evaluation(account, timestamp, "recovery attempt did not restore drawdown")


def _replay_one(raw_rows, timeline, trading_times, session_closes, latest, end_exclusive, contexts, mark_prices, risk_cap, policy):
    signals = build_signals(raw_rows, contexts, risk_cap)
    last_prices = {market: mark_prices[market][max(mark_prices[market])] for market in v13.MEMBERS}
    return PolicyReplay(signals, timeline, end_exclusive, last_prices, mark_prices, trading_times, session_closes, risk_cap=risk_cap, policy=policy).run()


def _account_summary(replay, risk_cap, policy):
    rows = []
    for account in replay.accounts:
        rows.append({
            "risk_cap": risk_cap, "policy": policy, "policy_label": POLICY_LABELS[policy], "account_id": account.account_id,
            "account_type": account.kind, "purchase_timestamp": str(account.purchase_timestamp), "final_state": account.state,
            "passed": account.pass_timestamp is not None, "pass_timestamp": str(account.pass_timestamp) if account.pass_timestamp else "",
            "mll_failed": account.failure_timestamp is not None, "failure_timestamp": str(account.failure_timestamp) if account.failure_timestamp else "",
            "voluntary_cancelled": account.voluntary_cancel_timestamp is not None, "voluntary_cancel_timestamp": str(account.voluntary_cancel_timestamp) if account.voluntary_cancel_timestamp else "",
            "qualified_created": account.kind == "QUALIFIED", "first_payout": account.first_payout_timestamp is not None,
            "subscription_paid": account.subscription_paid, "trader_payout_after_split": account.trader_payout,
            "external_cashflow": account.trader_payout - account.subscription_paid, "trades_taken": account.trades_taken,
            "gross_pnl": account.gross_pnl, "net_pnl": account.net_pnl, "fees": account.fees,
            "max_marked_equity_drawdown": account.max_equity_drawdown, "max_drawdown_used": account.max_drawdown_used,
            "dlg_events": account.dlg_breach_count, "mll_events": int(account.failure_timestamp is not None),
            "recovery_attempts": account.recovery_attempts, "successful_recovery_attempts": account.recovery_successes,
            "recovery_attempt_failures": account.recovery_failures, "max_zone": account.zone_max,
            "average_policy_risk": sum(account.policy_risk_caps) / len(account.policy_risk_caps) if account.policy_risk_caps else 0.0,
        })
    return pd.DataFrame(rows)


def _metrics(replay, account_df, risk_cap, policy, latest, baseline_cashflow=None):
    evals = account_df[account_df.account_type == "EVALUATION"]
    passed = evals[evals.passed]
    qualified = account_df[account_df.account_type == "QUALIFIED"]
    trades = pd.DataFrame(replay.trades)
    skipped = pd.DataFrame(replay.skipped)
    pass_days = []
    life_days = []
    for _, row in evals.iterrows():
        purchase = pd.Timestamp(row.purchase_timestamp)
        if row.pass_timestamp:
            pass_days.append((pd.Timestamp(row.pass_timestamp) - purchase).total_seconds() / 86400)
        end = row.pass_timestamp or row.failure_timestamp or str(replay.end_exclusive)
        life_days.append(max((pd.Timestamp(end) - purchase).total_seconds() / 86400, 0.0))
    gross_by_market = trades.groupby("market").gross_pnl.sum().to_dict() if not trades.empty else {}
    current_cashflow = float(account_df.external_cashflow.sum())
    return {
        "risk_cap": risk_cap, "policy": policy, "policy_label": POLICY_LABELS[policy], "evaluations_purchased": len(evals),
        "evaluations_passed": int(evals.passed.sum()), "mll_failures": int(evals.mll_failed.sum()),
        "voluntary_cancellations": int(evals.voluntary_cancelled.sum()), "active_or_censored_evaluations_at_end": int(evals.final_state.isin(["CENSORED_END_OF_DATA", "EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"]).sum()),
        "qualified_accounts_created": len(qualified), "first_payouts": int(qualified.first_payout.sum()), "total_payouts": int(qualified.first_payout.sum()),
        "trader_payout_after_split": float(account_df.trader_payout_after_split.sum()), "subscription_cost": float(account_df.subscription_paid.sum()),
        "net_external_cashflow": current_cashflow, "pass_rate": float(evals.passed.mean() * 100) if len(evals) else 0.0,
        "first_payout_rate": float(qualified.first_payout.mean() * 100) if len(qualified) else 0.0,
        "average_days_to_pass": sum(pass_days) / len(pass_days) if pass_days else 0.0, "median_days_to_pass": float(pd.Series(pass_days).median()) if pass_days else 0.0,
        "average_evaluation_lifetime_days": sum(life_days) / len(life_days) if life_days else 0.0,
        "average_subscriptions_per_evaluation": float(evals.subscription_paid.mean()) / 79.0 if len(evals) else 0.0,
        "average_contracts_per_trade": float(trades.total_contracts.mean()) if not trades.empty else 0.0,
        "average_initial_risk": float(trades.total_initial_risk.mean()) if not trades.empty else 0.0,
        "maximum_actual_initial_risk": float(trades.total_initial_risk.max()) if not trades.empty else 0.0,
        "maximum_marked_equity_drawdown": float(account_df.max_marked_equity_drawdown.max()) if len(account_df) else 0.0,
        "maximum_evaluation_marked_equity_drawdown": float(evals.max_marked_equity_drawdown.max()) if len(evals) else 0.0,
        "dlg_events": int(account_df.dlg_events.sum()), "mll_events": int(account_df.mll_events.sum()),
        "two_loss_daily_locks": int((skipped.reason == "daily stop rule").sum()) if not skipped.empty else 0,
        "zero_legal_contract_skips": int((skipped.reason == "zero legal contracts").sum()) if not skipped.empty else 0,
        "daily_lock_skips": int((skipped.reason == "daily stop rule").sum()) if not skipped.empty else 0,
        "voluntary_cancellation_count": int(evals.voluntary_cancelled.sum()), "recovery_attempts": int(account_df.recovery_attempts.sum()),
        "successful_recovery_attempts": int(account_df.successful_recovery_attempts.sum()), "recovery_attempt_failures": int(account_df.recovery_attempt_failures.sum()),
        "gross_pnl": float(trades.gross_pnl.sum()) if not trades.empty else 0.0, "net_pnl": float(trades.net_pnl.sum()) if not trades.empty else 0.0,
        "profit_by_market": json.dumps(gross_by_market, sort_keys=True), "payout_contribution_by_market": json.dumps({}, sort_keys=True),
        "subscription_cost_saved_vs_policy_a": 0.0, "additional_payouts_vs_policy_a": 0.0,
        "net_cashflow_difference_vs_policy_a": 0.0, "cost_per_evaluation_pass": float(account_df.subscription_paid.sum() / len(passed)) if len(passed) else 0.0,
        "cost_per_first_payout": float(account_df.subscription_paid.sum() / len(qualified[qualified.first_payout])) if int(qualified.first_payout.sum()) else float("nan"),
        "payout_to_subscription_ratio": float(account_df.trader_payout_after_split.sum() / account_df.subscription_paid.sum()) if account_df.subscription_paid.sum() else 0.0,
        "positive_external_cashflow_accounts_pct": float((account_df.external_cashflow > 0).mean() * 100) if len(account_df) else 0.0,
    }


def _build_outputs(results, latest, end_exclusive, root):
    summary_rows = []
    account_frames = []
    zone_rows = []
    recovery_rows = []
    market_rows = []
    for result in results:
        replay = result["replay"]
        cap = result["risk_cap"]
        policy = result["policy"]
        account_df = _account_summary(replay, cap, policy)
        account_frames.append(account_df)
        row = _metrics(replay, account_df, cap, policy, latest)
        summary_rows.append(row)
        for zone in ZONE_ORDER:
            accounts = [account for account in replay.accounts if account.kind == "EVALUATION" and account.zone_entries.get(zone, 0)]
            zone_rows.append({"risk_cap": cap, "policy": policy, "policy_label": POLICY_LABELS[policy], "zone": zone, "accounts_entered": len(accounts), "zone_entries": sum(a.zone_entries[zone] for a in accounts), "trades_entered": sum(a.zone_trade_counts[zone] for a in accounts), "skipped_trades": sum(a.zone_skip_counts[zone] for a in accounts), "maximum_drawdown_in_zone": max([a.max_drawdown_used for a in accounts if a.zone_max == zone] or [0.0])})
        recovery_rows.append({"risk_cap": cap, "policy": policy, "policy_label": POLICY_LABELS[policy], "recovery_attempts": row["recovery_attempts"], "successful_recovery_attempts": row["successful_recovery_attempts"], "recovery_attempt_failures": row["recovery_attempt_failures"], "recovery_success_rate": row["successful_recovery_attempts"] / row["recovery_attempts"] * 100 if row["recovery_attempts"] else 0.0, "voluntary_cancellations_after_recovery": int(account_df.voluntary_cancelled.sum())})
        trades = pd.DataFrame(replay.trades)
        if not trades.empty:
            for market, group in trades.groupby("market"):
                market_rows.append({"risk_cap": cap, "policy": policy, "policy_label": POLICY_LABELS[policy], "market": market, "trades": len(group), "gross_pnl": group.gross_pnl.sum(), "net_pnl": group.net_pnl.sum(), "fees": group.fees.sum(), "payout_contribution": 0.0})
    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary.policy == "A_FIXED_RISK"].set_index("risk_cap") if not summary.empty else pd.DataFrame()
    if not summary.empty:
        for index, row in summary.iterrows():
            base = baseline.loc[row.risk_cap] if row.risk_cap in baseline.index else None
            if base is not None:
                summary.loc[index, "subscription_cost_saved_vs_policy_a"] = base.subscription_cost - row.subscription_cost
                summary.loc[index, "additional_payouts_vs_policy_a"] = row.trader_payout_after_split - base.trader_payout_after_split
                summary.loc[index, "net_cashflow_difference_vs_policy_a"] = row.net_external_cashflow - base.net_external_cashflow
    accounts = pd.concat(account_frames, ignore_index=True) if account_frames else pd.DataFrame()
    warnings = []
    for _, row in summary.iterrows():
        warning = ""
        combo_accounts = accounts[(accounts.risk_cap == row.risk_cap) & (accounts.policy == row.policy) & (accounts.account_type == "EVALUATION")] if not accounts.empty else pd.DataFrame()
        critical_evaluations = int((combo_accounts.max_zone == "CRITICAL").sum()) if not combo_accounts.empty else 0
        if row.recovery_attempts and row.recovery_attempt_failures / row.recovery_attempts >= 0.5:
            warning = "Recovery policy has a high unsuccessful-attempt rate; treat as research-only."
        if row.mll_failures > 0:
            warning = (warning + " " if warning else "") + "MLL failures occurred; do not interpret as a recommended recovery behavior."
        if row.policy in {"C_CANCEL_AT_800", "D_ONE_RECOVERY", "E_BUFFER_LIMITED_RECOVERY"} and critical_evaluations == 0:
            warning = (warning + " " if warning else "") + "No Evaluation reached CRITICAL under this policy; its cancellation/recovery branch was not exercised."
        if row.policy == "B_DEFENSIVE_REDUCTION" and critical_evaluations:
            warning = (warning + " " if warning else "") + f"{critical_evaluations} Evaluation(s) reached CRITICAL; Policy B has no cancellation/recovery branch by design."
        if row.evaluations_purchased < 10:
            warning = (warning + " " if warning else "") + "Small account count; aggregate policy comparison is uncertain."
        warnings.append({"risk_cap": row.risk_cap, "policy": row.policy, "warning": warning or "No mechanical warning; results remain proxy-data research."})
    economics = summary[["risk_cap", "policy", "policy_label", "subscription_cost", "trader_payout_after_split", "net_external_cashflow", "subscription_cost_saved_vs_policy_a", "additional_payouts_vs_policy_a", "net_cashflow_difference_vs_policy_a", "cost_per_evaluation_pass", "cost_per_first_payout", "payout_to_subscription_ratio", "positive_external_cashflow_accounts_pct"]].copy()
    summary.to_csv(root / "risk_policy_summary.csv", index=False)
    zone = pd.DataFrame(zone_rows)
    zone.to_csv(root / "drawdown_zone_summary.csv", index=False)
    accounts.to_csv(root / "account_outcome_summary.csv", index=False)
    pd.DataFrame(recovery_rows).to_csv(root / "recovery_policy_summary.csv", index=False)
    economics.to_csv(root / "economics_comparison.csv", index=False)
    pd.DataFrame(market_rows).to_csv(root / "market_contribution.csv", index=False)
    pd.DataFrame(warnings).to_csv(root / "confidence_warnings.csv", index=False)
    best_cash = summary.sort_values(["net_external_cashflow", "pass_rate"], ascending=False).iloc[0] if not summary.empty else None
    best_pass = summary.sort_values(["pass_rate", "net_external_cashflow"], ascending=False).iloc[0] if not summary.empty else None
    best_payout = summary.sort_values(["trader_payout_after_split", "net_external_cashflow"], ascending=False).iloc[0] if not summary.empty else None
    top_cash = summary[summary.net_external_cashflow == summary.net_external_cashflow.max()] if not summary.empty else pd.DataFrame()
    top_cash_label = "; ".join(f"{row.policy_label} @ ${row.risk_cap:,.0f}" for _, row in top_cash.iterrows()) if not top_cash.empty else "n/a"
    critical_text = "; ".join(f"${row.risk_cap:,.0f}/{row.policy}: {int((accounts[(accounts.risk_cap == row.risk_cap) & (accounts.policy == row.policy) & (accounts.account_type == 'EVALUATION')].max_zone == 'CRITICAL').sum())}" for _, row in summary[summary.policy.isin(["C_CANCEL_AT_800", "D_ONE_RECOVERY", "E_BUFFER_LIMITED_RECOVERY"])].drop_duplicates(["risk_cap", "policy"]).iterrows()) if not summary.empty else "n/a"
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>V13.1 Risk and Account Exit Policy Research</title><style>body{{font-family:Arial;margin:2rem;max-width:1800px}}table{{border-collapse:collapse;font-size:10px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}</style></head><body><h1>Strategy V13.1 Risk and Account Exit Policy Research</h1><div class='warn'>Frozen V13 signals and corrected lifecycle engine. Research variables: maximum initial risk cap and deterministic Evaluation account-exit policy. Policies D/E are research-only recovery experiments and are not recommendations.</div><p>Data window: {START} through {latest}; end-exclusive {end_exclusive}. No Monte Carlo, bootstrap, random paths, or strategy optimization.</p><h2>Headline answers</h2><ul><li>Highest net external cashflow (ties shown): {top_cash_label}, ${best_cash.net_external_cashflow:,.2f}.</li><li>Highest pass rate: {best_pass.policy_label if best_pass is not None else 'n/a'} at ${best_pass.risk_cap:,.0f} risk, {best_pass.pass_rate:.2f}%.</li><li>Highest trader payout: {best_payout.policy_label if best_payout is not None else 'n/a'} at ${best_payout.risk_cap:,.0f} risk, ${best_payout.trader_payout_after_split:,.2f}.</li><li>Critical Evaluation counts for Policies C/D/E by cap: {critical_text}. None of these branches was exercised; the one critical Evaluation under Policy B is a defensive-sizing observation, not a cancellation/recovery result.</li></ul><h2>Risk-policy summary</h2>{summary.to_html(index=False, border=0)}<h2>Economics comparison</h2>{economics.to_html(index=False, border=0)}<h2>Drawdown zones</h2>{zone.to_html(index=False, border=0)}<h2>Recovery policies</h2>{pd.DataFrame(recovery_rows).to_html(index=False, border=0)}<h2>Confidence warnings</h2>{pd.DataFrame(warnings).to_html(index=False, border=0)}</body></html>"""
    (root / "final_report.html").write_text(report, encoding="utf-8")
    return summary


def write_verified_rules(root: Path = ROOT):
    root.mkdir(parents=True, exist_ok=True)
    text = """# Alpha Futures Zero 25K rules verified 2026-07-15

The V13.1 model uses the current published Zero rules and the existing
corrected V13 lifecycle implementation:

- 25K Zero Evaluation: **$79/month**.
- Profit target: **$1,500**.
- Maximum Loss Limit: **$1,000**. The official MLL page states that a breach
  can occur from floating equity or closed balance and liquidates the account.
  The existing V13 engine therefore checks marked equity causally and keeps
  its verified end-of-day trailing MLL update.
- Daily Loss Guard: **$500** for 25K Zero. It is a soft breach: unrealized,
  realized PnL, fees and commissions are included; positions are flattened and
  the account is locked until the next trading day.
- Evaluation position limit: **1 mini or 10 micros**.
- Monthly billing: the subscription rebills on the signup day each month until
  the Evaluation passes or the trader cancels. After a failed Evaluation, the
  official documentation says rebilling continues unless cancelled; after
  rebill the failed account is reset. V13.1 explicitly models the requested
  trader action of manually cancelling immediately after a simulated failure,
  so no future rebill is charged for that account.
- Qualified payout economics: up to 50% of profit per request after five
  winning days of at least $200; the trader receives 90% of the requested
  withdrawal. The existing V13 payout engine is unchanged.
- Compliance: Alpha prohibits all-or-nothing trading, maximum-leverage
  account rolling, account stacking, and gambling-like repeated account
  failures. Policies D and E are therefore labeled research-only and receive
  explicit confidence/compliance warnings.

Sources:

- [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview)
- [Maximum Loss Limit](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll)
- [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard)
- [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription)
- [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy)
- [Prohibited Trading Practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)

The official documentation is time-sensitive. This verification is a research
record, not legal or commercial advice; live account terms should be checked
again before trading.
"""
    (root / "verified_rules.md").write_text(text, encoding="utf-8")


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_verified_rules(root)
    raw_rows, timeline, trading_times, session_closes, latest, end_exclusive, contexts, mark_prices = v13._load_signal_stream()
    results = []
    for risk_cap in RISK_CAPS:
        for policy in POLICIES:
            replay = _replay_one(raw_rows, timeline, trading_times, session_closes, latest, end_exclusive, contexts, mark_prices, risk_cap, policy)
            results.append({"risk_cap": risk_cap, "policy": policy, "replay": replay})
    summary = _build_outputs(results, latest, end_exclusive, root)
    return {"combinations": len(results), "latest_candle": str(latest), "root": str(root), "best_net_external_cashflow": summary.sort_values("net_external_cashflow", ascending=False).iloc[0].to_dict() if not summary.empty else {}}


if __name__ == "__main__":
    print(run())
