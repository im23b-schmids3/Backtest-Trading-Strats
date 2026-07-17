from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import floor
from statistics import median
from typing import Sequence

from ..prop.contracts import default_contract_registry, get_contract
from ..prop.rule_registry import verified_rule_registry
from .models import PortfolioExecution, PortfolioMember, PortfolioPropMetrics, PortfolioRiskResult, PortfolioSignalEvent, RiskAllocationPolicy


def replay_shared_account(candidate_id: str, events: Sequence[PortfolioSignalEvent], members: Sequence[PortfolioMember], *, policy: RiskAllocationPolicy, maximum_total_contracts: int = 10, maximum_simultaneous_positions: int = 10, product: str = "Alpha Futures Zero 25K", scenario: str = "complementary", account_start: float = 25000) -> tuple[PortfolioPropMetrics, PortfolioRiskResult]:
    rule = verified_rule_registry()[f"Alpha Futures {product}" if not product.startswith("Alpha Futures") else product]
    contracts = default_contract_registry()
    member_ids = [member.strategy_id for member in members]
    ordered = sorted(events, key=lambda item: (item.entry_timestamp, item.strategy_id, item.signal_id))
    balance = account_start; peak = balance; maximum_dd = 0.0; day_pnl: dict[str, float] = defaultdict(float); winning_days: dict[str, float] = defaultdict(float); cycle_profit = 0.0; locked_days: set[str] = set(); failed = False; passed = False; qualified = False; pass_time = None; payout_times: list[datetime] = []; payouts = 0; gross_payouts = 0.0; trader_payouts = 0.0
    subscriptions = rule.monthly_subscription; executions: list[PortfolioExecution] = []; active: list[tuple[PortfolioSignalEvent, int]] = []; strategy_pnl = defaultdict(float); strategy_trades = defaultdict(int); unique_days: set[str] = set()
    for event in ordered:
        active = [(open_event, quantity) for open_event, quantity in active if open_event.exit_timestamp > event.entry_timestamp]
        day = event.exit_timestamp.astimezone(timezone.utc).date().isoformat()
        requested = max(1, event.quantity_intent); denied = 0; reasons: list[str] = []
        if failed: reasons.append("INACTIVE_ACCOUNT")
        if day in locked_days: reasons.append("DLG_LOCK")
        if len(active) >= maximum_simultaneous_positions: reasons.append("MAX_SIMULTANEOUS_POSITIONS")
        if any(open_event.duplicate_exposure_group == event.duplicate_exposure_group for open_event, _ in active): reasons.append("DUPLICATE_EXPOSURE")
        contract = get_contract("MBT" if event.market.upper().startswith("BTC") else "MET" if event.market.upper().startswith("ETH") else "MES" if event.market.upper() in {"SPX", "QQQ"} else "MBT", contracts)
        risk_per_contract = abs(event.entry_price - event.stop) / contract.minimum_tick * contract.tick_value
        risk_budget = 200 / max(1, len(members)) if policy == RiskAllocationPolicy.EQUAL_RISK else 200
        if policy == RiskAllocationPolicy.PRIORITY_BASED:
            risk_budget = 250 if next((item.priority for item in members if item.strategy_id == event.strategy_id), 0) == 0 else 100
        if policy == RiskAllocationPolicy.DRAWDOWN_AWARE:
            risk_budget = max(50, 200 * max(0.25, 1 - maximum_dd / max(1, account_start)))
        requested = min(requested, max(0, floor(risk_budget / max(risk_per_contract, 0.01))))
        open_contracts = sum(quantity for _, quantity in active)
        granted = max(0, min(requested, maximum_total_contracts - open_contracts)) if not reasons else 0
        if granted < requested:
            denied += requested - granted
            if open_contracts + requested > maximum_total_contracts: reasons.append("ACCOUNT_CONTRACT_LIMIT")
        if requested == 0: reasons.append("ZERO_LEGAL_CONTRACTS")
        direction = 1 if event.direction.upper() == "LONG" else -1
        gross = ((event.exit_price - event.entry_price) / contract.minimum_tick) * contract.tick_value * granted * direction
        fees = event.fees * granted; slippage = event.slippage * granted; net = gross - fees - slippage
        accepted = granted > 0 and not reasons
        if accepted:
            active.append((event, granted)); balance += net; day_pnl[day] += net; strategy_pnl[event.strategy_id] += net; strategy_trades[event.strategy_id] += 1; unique_days.add(day); peak = max(peak, balance); maximum_dd = max(maximum_dd, peak - balance)
            if net > 0: winning_days[day] += net
            if day_pnl[day] <= -rule.daily_loss_guard: locked_days.add(day)
            if balance <= account_start - rule.maximum_loss_limit: failed = True
            if not passed and balance >= account_start + rule.profit_target:
                passed = True; qualified = True; pass_time = event.exit_timestamp; winning_days.clear(); cycle_profit = 0
            elif qualified:
                cycle_profit += net
                qualified_days = [key for key, value in winning_days.items() if value >= rule.winning_day_requirements.get("minimum_profit_per_winning_day", 200)]
                if len(qualified_days) >= rule.winning_day_requirements.get("minimum_winning_days", 5) and cycle_profit >= rule.payout_minimum:
                    payout = min(cycle_profit * 0.5, rule.payout_maximum)
                    if payout >= rule.payout_minimum:
                        payouts += 1; gross_payouts += payout; trader_payouts += payout * rule.payout_split; payout_times.append(event.exit_timestamp); balance -= payout; cycle_profit = 0; winning_days.clear()
        else:
            denied = max(denied, requested)
        if "DLG_LOCK" in reasons: locked_days.add(day)
        executions.append(PortfolioExecution(signal_id=event.signal_id, strategy_id=event.strategy_id, market=event.market, direction=event.direction, entry_timestamp=event.entry_timestamp, exit_timestamp=event.exit_timestamp, requested_contracts=max(0, requested), granted_contracts=granted, denied_quantity=denied, denial_reasons=sorted(set(reasons)), gross_pnl=gross, fees=fees, slippage=slippage, net_pnl=net, accepted=accepted))
    completed = [item for item in executions if item.accepted]
    start = ordered[0].entry_timestamp if ordered else datetime.now(timezone.utc); end = ordered[-1].exit_timestamp if ordered else start; months = max(1 / 30, (end - start).days / 30)
    metrics = PortfolioPropMetrics(candidate_id=candidate_id, evaluations_passed=1 if passed else 0, evaluations_failed=1 if failed else 0, pass_rate=1 if passed else 0, qualified_accounts=1 if qualified else 0, first_payouts=1 if payouts else 0, total_payouts=payouts, payout_rate=1 if payouts else 0, median_days_to_pass=(pass_time - start).total_seconds() / 86400 if pass_time else None, median_days_to_first_payout=(payout_times[0] - pass_time).total_seconds() / 86400 if payout_times and pass_time else None, dlg_events=sum("DLG_LOCK" in item.denial_reasons for item in executions), mll_events=1 if failed else 0, gross_pnl=sum(item.gross_pnl for item in completed), fees=sum(item.fees for item in completed), slippage=sum(item.slippage for item in completed), net_pnl=sum(item.net_pnl for item in completed), subscriptions=subscriptions, gross_payouts=gross_payouts, trader_payouts=trader_payouts, net_external_cashflow=trader_payouts - subscriptions, roi=(trader_payouts - subscriptions) / subscriptions if subscriptions else None, cost_per_pass=subscriptions if passed else None, cost_per_payout=subscriptions if payouts else None, executable_trades_per_month=len(completed) / months, unique_completed_trades=len(completed), zero_trade_months=0 if completed else 1, winning_days_per_month=len(winning_days) / months, payout_qualifying_days_per_month=(5 if payouts else 0) / months, maximum_marked_equity_drawdown=maximum_dd, strategy_pnl=dict(strategy_pnl), strategy_trades=dict(strategy_trades))
    risk = PortfolioRiskResult(candidate_id=candidate_id, policy=policy, executions=executions, total_requested_contracts=sum(item.requested_contracts for item in executions), total_granted_contracts=sum(item.granted_contracts for item in executions), contract_limit_skips=sum("ACCOUNT_CONTRACT_LIMIT" in item.denial_reasons for item in executions), risk_allocation_skips=sum("ZERO_LEGAL_CONTRACTS" in item.denial_reasons for item in executions), duplicate_exposure_skips=sum("DUPLICATE_EXPOSURE" in item.denial_reasons for item in executions), conflict_skips=0, mll_buffer_skips=0, dlg_skips=sum("DLG_LOCK" in item.denial_reasons for item in executions), session_skips=0, inactive_account_skips=sum("INACTIVE_ACCOUNT" in item.denial_reasons for item in executions), zero_legal_contract_skips=sum("ZERO_LEGAL_CONTRACTS" in item.denial_reasons for item in executions), account_maximum_drawdown=maximum_dd, account_minimum_balance=min([account_start] + [account_start + sum(item.net_pnl for item in executions[:index + 1]) for index in range(len(executions))]), shared_account=True)
    return metrics, risk
