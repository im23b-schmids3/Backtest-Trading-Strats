from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Iterable

from .models import AccountEvent, AccountSummary, BillingEvent, PayoutRecord, PropRuleSet, PropScenarioConfig, PropScenarioMetrics, RiskSizingResult, TradeSignal
from .reconcile import reconcile_trade
from .sizing import SharedExposure, size_trade


def _day(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).date().isoformat()


def simulate_scenario(rule: PropRuleSet, scenario: PropScenarioConfig, trades: Iterable[TradeSignal], mappings: dict[str, object], contracts: dict[str, object], limitations, b5_verified: bool = True) -> tuple[PropScenarioMetrics, list[AccountSummary], list[PayoutRecord], list[BillingEvent], list[RiskSizingResult], list[object]]:
    ordered = sorted(list(trades), key=lambda item: (item.timestamp, item.trade_id))
    account_id = f"{scenario.scenario_id}-evaluation-1"
    started = ordered[0].timestamp if ordered else datetime.now(timezone.utc)
    account = AccountSummary(account_id=account_id, account_type="EVALUATION", status="ACTIVE", started_at=started)
    billing = [BillingEvent(account_id=account_id, timestamp=started, event_type="EVALUATION_SUBSCRIPTION", amount=rule.monthly_subscription, reason="evaluation started")]
    account.billing_events = billing.copy()
    qualified: AccountSummary | None = None
    payouts: list[PayoutRecord] = []
    sizing: list[RiskSizingResult] = []
    reconciliations: list[object] = []
    exposure = SharedExposure(rule.contract_limits.get("micro", 0))
    balance = rule.account_size
    eval_start_balance = balance
    qualified_start = balance
    day_pnl: dict[str, float] = defaultdict(float)
    day_wins: dict[str, float] = defaultdict(float)
    cycle_profit = 0.0
    stopped_days: set[str] = set()
    peak_balance = balance
    maximum_drawdown = 0.0
    for trade in ordered:
        if account.status != "ACTIVE" and qualified is None: break
        mapping = mappings.get(trade.source_market)
        if mapping is None: continue
        if _day(trade.timestamp) in stopped_days: continue
        active_account = account if qualified is None else qualified
        marked = balance
        risk = size_trade(active_account.account_id, trade, contracts[mapping.target_futures_contract], rule, scenario.risk_policy, exposure, marked, max(0.0, balance - (eval_start_balance - rule.maximum_loss_limit)))
        sizing.append(risk)
        if risk.legal_contracts <= 0:
            if risk.skipped_reason == "ACCOUNT_CONTRACT_LIMIT": continue
            continue
        reconciliation = reconcile_trade(trade, mapping, risk.legal_contracts, contracts)
        reconciliations.append(reconciliation)
        pnl = reconciliation.net_pnl
        balance += pnl
        peak_balance = max(peak_balance, balance)
        maximum_drawdown = max(maximum_drawdown, peak_balance - balance)
        d = _day(trade.exit_timestamp)
        day_pnl[d] += pnl
        if pnl > 0: day_wins[d] += pnl
        active_account.trades += 1
        active_account.total_pnl += pnl
        active_account.total_fees += reconciliation.fees + reconciliation.slippage
        active_account.events.append(AccountEvent(account_id=active_account.account_id, timestamp=trade.exit_timestamp, event_type="TRADE_SETTLED", balance=balance, marked_equity=balance, realized_pnl=pnl, unrealized_pnl=0, daily_realized_pnl=day_pnl[d], daily_unrealized_pnl=0, fees=reconciliation.fees, dlg_used=max(0, -day_pnl[d]), mll_threshold=eval_start_balance - rule.maximum_loss_limit, remaining_mll_buffer=balance - (eval_start_balance - rule.maximum_loss_limit), reason="deterministic trade settlement"))
        # The synthetic signal is fully settled at exit; do not carry closed
        # contracts into the next timestamp. SharedExposure still protects
        # simultaneous/open exposure in adapters that retain positions.
        exposure.release(active_account.account_id, risk.legal_contracts)
        if rule.daily_loss_guard and day_pnl[d] <= -rule.daily_loss_guard:
            stopped_days.add(d)
            active_account.events.append(AccountEvent(account_id=active_account.account_id, timestamp=trade.exit_timestamp, event_type="DLG_LOCK", balance=balance, marked_equity=balance, realized_pnl=0, unrealized_pnl=0, daily_realized_pnl=day_pnl[d], daily_unrealized_pnl=0, fees=0, dlg_used=-day_pnl[d], mll_threshold=eval_start_balance - rule.maximum_loss_limit, remaining_mll_buffer=balance - (eval_start_balance - rule.maximum_loss_limit), reason="daily loss guard reached"))
        if balance <= eval_start_balance - rule.maximum_loss_limit:
            account.status = "FAILED" if qualified is None else "QUALIFIED_FAILED"
            active_account.failure_timestamp = trade.exit_timestamp
            active_account.ended_at = trade.exit_timestamp
            active_account.events.append(AccountEvent(account_id=active_account.account_id, timestamp=trade.exit_timestamp, event_type="MLL_BREACH", balance=balance, marked_equity=balance, realized_pnl=0, unrealized_pnl=0, daily_realized_pnl=day_pnl[d], daily_unrealized_pnl=0, fees=0, dlg_used=max(0, -day_pnl[d]), mll_threshold=eval_start_balance - rule.maximum_loss_limit, remaining_mll_buffer=0, reason="maximum loss limit breached"))
            if qualified is None:
                account.cancellation_timestamp = trade.exit_timestamp
                cancellation = BillingEvent(account_id=account.account_id, timestamp=trade.exit_timestamp, event_type="EVALUATION_CANCELLED", amount=0, reason="immediate cancellation after failure under scenario policy")
                account.billing_events.append(cancellation)
                billing.append(cancellation)
            break
        if qualified is None and balance >= eval_start_balance + rule.profit_target:
            account.status = "PASSED"
            account.pass_timestamp = trade.exit_timestamp
            account.ended_at = trade.exit_timestamp
            qualified = AccountSummary(account_id=f"{scenario.scenario_id}-qualified-1", account_type="QUALIFIED", status="ACTIVE", started_at=trade.exit_timestamp)
            qualified_start = balance
            eval_start_balance = balance
            # Evaluation winning days cannot satisfy a qualified payout.
            day_wins.clear()
            day_pnl.clear()
        elif qualified is not None:
            qualified.qualified_trades += 1
            cycle_profit += pnl
            winning_days = [date for date, value in day_wins.items() if value >= rule.winning_day_requirements.get("minimum_profit_per_winning_day", 200)]
            largest = max(day_wins.values(), default=0)
            consistency = largest / cycle_profit if cycle_profit > 0 else 1.0
            if len(winning_days) >= rule.winning_day_requirements.get("minimum_winning_days", 5) and cycle_profit >= rule.payout_minimum and (rule.consistency_rule is None or consistency <= rule.consistency_rule):
                gross = min(cycle_profit * .5, rule.payout_maximum)
                if gross >= rule.payout_minimum:
                    payout = PayoutRecord(account_id=qualified.account_id, payout_number=len(payouts) + 1, eligibility_timestamp=trade.exit_timestamp, winning_day_count=len(winning_days), winning_day_dates=winning_days, largest_winning_day=largest, payout_cycle_profit=cycle_profit, consistency_percentage=consistency, maximum_legal_request=rule.payout_maximum, gross_payout_requested=gross, provider_share=gross * (1 - rule.payout_split), trader_share=gross * rule.payout_split, payout_date=trade.exit_timestamp, balance_before=balance, balance_after=balance - gross, cycle_reset_behavior=rule.payout_cycle_reset_behavior)
                    payouts.append(payout)
                    qualified.payouts.append(payout)
                    balance -= gross
                    cycle_profit = 0
                    day_wins.clear()
    if qualified is not None and qualified.status == "ACTIVE": qualified.ended_at = ordered[-1].exit_timestamp if ordered else qualified.started_at; account.status = "PASSED"
    if account.status == "ACTIVE":
        if ordered and (ordered[-1].exit_timestamp - account.started_at).days + 1 >= scenario.max_days:
            account.status = "CENSORED"; account.ended_at = account.started_at + timedelta(days=scenario.max_days)
    for event in account.billing_events:
        if account.status in {"FAILED", "PASSED"} and rule.billing_after_failure == "stop_on_management_action": pass
    evals = 1
    passed = 1 if account.status == "PASSED" else 0
    failed = 1 if account.status == "FAILED" else 0
    qualified_created = 1 if qualified is not None else 0
    all_accounts = [account] + ([qualified] if qualified else [])
    gross_pnl = sum(item.gross_pnl for item in reconciliations)
    fees = sum(item.fees for item in reconciliations)
    slippage = sum(item.slippage for item in reconciliations)
    net_pnl = sum(item.net_pnl for item in reconciliations)
    subscriptions = sum(item.amount for item in billing if item.event_type == "EVALUATION_SUBSCRIPTION")
    trader_payouts = sum(item.trader_share for item in payouts)
    net_external = trader_payouts - subscriptions
    period_days = max(1, ((ordered[-1].exit_timestamp - started).days + 1) if ordered else scenario.max_days)
    pass_days = ((account.pass_timestamp - account.started_at).total_seconds() / 86400) if account.pass_timestamp else None
    first_payout_days = ((payouts[0].payout_date - qualified.started_at).total_seconds() / 86400) if payouts and qualified else None
    evaluation_lifetime = ((account.ended_at - account.started_at).total_seconds() / 86400) if account.ended_at else period_days
    qualified_lifetime = ((qualified.ended_at - qualified.started_at).total_seconds() / 86400) if qualified and qualified.ended_at else None
    annual_factor = 365 / period_days
    metrics = PropScenarioMetrics(scenario_id=scenario.scenario_id, evaluations_purchased=evals, evaluations_passed=passed, evaluations_failed=failed, voluntarily_cancelled_evaluations=1 if account.cancellation_timestamp else 0, censored_evaluations=1 if account.status == "CENSORED" else 0, qualified_accounts_created=qualified_created, qualified_failures=1 if qualified and qualified.status == "QUALIFIED_FAILED" else 0, first_payouts=sum(1 for item in payouts if item.payout_number == 1), second_payouts=sum(1 for item in payouts if item.payout_number == 2), third_payouts=sum(1 for item in payouts if item.payout_number == 3), total_payouts=len(payouts), pass_rate=passed / evals, first_payout_rate_per_started_evaluation=(1 if payouts else 0) / evals, first_payout_rate_per_passed_evaluation=(1 if payouts else 0) / passed if passed else 0, median_days_to_pass=pass_days, median_days_to_first_payout=first_payout_days, average_evaluation_lifetime=evaluation_lifetime, average_qualified_lifetime=qualified_lifetime, average_trades_per_evaluation=account.trades, average_trades_per_qualified=qualified.qualified_trades if qualified else 0, average_contracts=sum(item.legal_contracts for item in sizing) / max(1, len(sizing)), average_initial_risk=sum(item.requested_risk for item in sizing) / max(1, len(sizing)), maximum_marked_equity_drawdown=maximum_drawdown, dlg_events=sum(1 for item in account.events if item.event_type == "DLG_LOCK"), mll_events=sum(1 for item in account.events if item.event_type == "MLL_BREACH"), contract_limit_skips=sum(1 for item in sizing if item.skipped_reason == "ACCOUNT_CONTRACT_LIMIT"), risk_cap_skips=sum(1 for item in sizing if item.skipped_reason == "RISK_CAP"), zero_legal_contract_skips=sum(1 for item in sizing if item.skipped_reason == "ZERO_LEGAL_CONTRACTS"), gross_trading_pnl=gross_pnl, fees=fees, slippage=slippage, net_trading_pnl=net_pnl, evaluation_subscriptions=subscriptions, gross_payout_requests=sum(item.gross_payout_requested for item in payouts), trader_payouts=trader_payouts, net_external_cashflow=net_external, cost_per_pass=subscriptions / passed if passed else None, cost_per_first_payout=subscriptions / 1 if payouts else None, payout_to_subscription_ratio=trader_payouts / subscriptions if subscriptions else None, roi_on_external_costs=net_external / subscriptions if subscriptions else None, profitable_external_path_percentage=100 if net_external > 0 else 0, annualized_evaluation_purchases=evals * annual_factor, annualized_payouts=len(payouts) * annual_factor, annualized_net_cashflow=net_external * annual_factor, maximum_capital_outlay=subscriptions)
    return metrics, all_accounts, payouts, billing, sizing, reconciliations
