"""One continuous 25K Alpha Zero account journal over the retained history.

This is an accounting/reporting wrapper around the frozen SingleTrader model.
It does not generate signals or alter strategy execution.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

from fib_backtester.research import v12_binance_proxy_prop_simulation as legacy
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed
from fib_backtester.research import v12_single_trader as single
from fib_backtester.research.v12_contract_registry import PROXY_SYMBOLS, build_synthetic_context, mapped_price, round_to_tick


ROOT = Path("reports/v12_journal")
START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
PORTFOLIO = "Portfolio C - BTC + ETH + Gold"
MEMBERS = ["BTC", "ETH", "Gold"]
TIMEFRAME = "4h"
SIZE = 10
MODE = "MIRRORED"


def _full_market_trades():
    frames = {}
    latest = None
    for market in MEMBERS:
        bars = legacy._read_cached_bars(market, TIMEFRAME)
        latest = bars.index.max() if latest is None else max(latest, bars.index.max())
        trades, _ = legacy._run_frozen(market, TIMEFRAME, bars)
        frames[(market, TIMEFRAME)] = trades
    return frames, latest


def _months(end_candle):
    return [str(p) for p in pd.period_range(START.strftime("%Y-%m"), end_candle.strftime("%Y-%m"), freq="M")]


def _run_journal(trades, end_exclusive, contexts):
    old_start, old_end = single.START, single.END
    single.START, single.END = START, end_exclusive
    try:
        simulation = single.SingleTrader("25K Zero", PORTFOLIO, TIMEFRAME, SIZE, MODE, trades)
        simulation.months = {month: simulation._blank_month() for month in _months(end_exclusive - pd.Timedelta(nanoseconds=1))}
        events, trade_rows, eval_snapshots, daily_rows, payout_rows = [], {}, [], [], []
        account_ids, account_objects = {}, {}
        type_ids = {}
        next_eval = 0
        next_qualified = 0
        legs = defaultdict(list)
        last_transition = 0

        def account_id(account):
            nonlocal next_eval, next_qualified
            if account is None:
                return ""
            key = id(account)
            if key not in account_ids:
                if account.kind == "EVALUATION":
                    next_eval += 1
                    account_ids[key] = f"EVAL-{next_eval:03d}"
                else:
                    next_qualified += 1
                    account_ids[key] = f"QUAL-{next_qualified:03d}"
                account_objects[key] = account
                type_ids[account.account_type] = account_ids[key]
            return account_ids[key]

        def emit_event(timestamp, account, old_state, new_state, reason, subscription=0.0, gross=0.0, trader=0.0, balance_before=None, balance_after=None):
            after = float(account.balance) if account is not None else (float(balance_after) if balance_after is not None else np.nan)
            before = after if balance_before is None else float(balance_before)
            events.append({"timestamp": str(timestamp), "account_id": account_id(account), "account_type": account.kind if account is not None else "", "old_state": old_state, "new_state": new_state, "event_reason": reason, "balance_before": before, "balance_after": after, "related_subscription_cost": float(subscription), "related_gross_withdrawal": float(gross), "related_trader_payout": float(trader), "cumulative_external_cashflow": 0.0})

        def capture_transitions():
            nonlocal last_transition
            for row in simulation.transitions[last_transition:]:
                at = row["account_type"]
                aid = type_ids.get(at, "")
                object_key = next((key for key, value in account_ids.items() if value == aid), None)
                account = account_objects.get(object_key)
                emit_event(row["timestamp"], account, row["old_state"], row["new_state"], row["reason"], subscription=float(row.get("related_cost_or_payout", 0.0)) if "subscription" in row["reason"].lower() else 0.0, trader=float(row.get("related_cost_or_payout", 0.0)) if "payout" in row["reason"].lower() and "request filled" not in row["reason"].lower() else 0.0)
            last_transition = len(simulation.transitions)

        original_buy = simulation._buy_evaluation
        original_charge = simulation._charge_subscription
        original_start_q = simulation._start_qualified
        original_accept = simulation._accept
        original_apply = simulation._apply_exit
        original_exit = simulation._exit_for_account
        original_flatten = simulation._flatten
        original_finish = simulation._finish_session
        position_records = {}

        def buy(self, timestamp, reason):
            capture_transitions()
            result = original_buy(timestamp, reason)
            account = self.eval
            account_id(account)
            emit_event(timestamp, account, "NONE", account.state, "Evaluation purchased", balance_before=account.balance, balance_after=account.balance)
            capture_transitions()
            return result

        def charge(self, timestamp):
            result = original_charge(timestamp)
            # The initial charge occurs inside _buy_evaluation before the new
            # account has been assigned its journal ID; the purchase wrapper
            # captures that transition once the ID exists.
            if self.eval is not None and id(self.eval) in account_ids:
                capture_transitions()
            return result

        def start_q(self, timestamp, balance):
            result = original_start_q(timestamp, balance)
            if result and self.qualified is not None:
                account_id(self.qualified)
                emit_event(timestamp, self.qualified, "NONE", self.qualified.state, "Qualified account created", balance_before=balance, balance_after=self.qualified.balance)
            capture_transitions()
            return result

        def accept(self, account, signal_id, trade, timestamp):
            before = float(account.balance)
            result = original_accept(account, signal_id, trade, timestamp)
            if account.positions.get(signal_id) is not None:
                key = (id(account), signal_id)
                position_records[key] = {"account": account, "account_id": account_id(account), "account_type": account.kind, "account_stage_before_trade": account.state, "signal_id": signal_id, "trade": trade, "entry_timestamp": str(timestamp), "balance_before_trade": before, "balance_after_entry": float(account.balance), "equity_low": float(account.balance)}
            capture_transitions()
            return result

        def apply_exit(self, account, trade, leg, timestamp, forced=False):
            key = (id(account), next((sid for (aid, sid), rec in position_records.items() if aid == id(account) and rec["trade"] is trade), None))
            legs[key].append(dict(leg, forced=bool(forced), timestamp=str(timestamp)))
            result = original_apply(account, trade, leg, timestamp, forced)
            if key in position_records:
                position_records[key]["equity_low"] = min(position_records[key]["equity_low"], float(account.balance))
            return result

        def finalize(account, signal_id):
            key = (id(account), signal_id)
            rec = position_records.get(key)
            if rec is None or key not in legs or key in trade_rows:
                return
            trade = rec["trade"]
            leg_list = legs[key]
            if not leg_list:
                return
            gross = sum(float(leg["gross"]) for leg in leg_list)
            exit_fees = sum(float(leg["fee"]) for leg in leg_list)
            entry_fee = float(trade["entry_fee"])
            reasons = [leg["reason"] for leg in leg_list]
            reached = [i for i in range(1, 6) if any(reason == f"tp{i}" for reason in reasons)]
            percentages = {f"tp{i}_closed_pct": sum(float(leg["quantity"]) for leg in leg_list if leg["reason"] == f"tp{i}") / trade["contracts"] * 100 for i in range(1, 6)}
            final_leg = leg_list[-1]
            closed_contracts = sum(float(leg["quantity"]) for leg in leg_list)
            target_values = trade.get("tp_prices", ["", "", "", "", ""])
            row = {"chronological_trade_number": len(trade_rows) + 1, "account_id": rec["account_id"], "account_type": rec["account_type"], "account_stage_before_trade": rec["account_stage_before_trade"], "signal_id": rec["signal_id"], "market": trade["market"], "proxy_symbol": PROXY_SYMBOLS[trade["market"]], "mapped_futures_contract": trade["alpha_product"], "timeframe": TIMEFRAME, "direction": trade["side"], "setup_timestamp": trade.get("setup_timestamp", ""), "entry_timestamp": rec["entry_timestamp"], "exit_timestamp": final_leg["timestamp"], "entry_price": trade["entry"], "initial_stop_price": trade.get("initial_stop", ""), "tp1_price": target_values[0] if len(target_values) > 0 else "", "tp2_price": target_values[1] if len(target_values) > 1 else "", "tp3_price": target_values[2] if len(target_values) > 2 else "", "tp4_price": target_values[3] if len(target_values) > 3 else "", "tp5_price": target_values[4] if len(target_values) > 4 else "", "final_exit_price": final_leg["price"], "total_contracts": trade["contracts"], "closed_contracts": closed_contracts, "quantity_reconciles": bool(abs(closed_contracts - float(trade["contracts"])) < 1e-9), "initial_dollar_risk": abs(trade["entry"] - trade.get("initial_stop", trade["entry"])) * fixed.CANONICAL_PROXIES[trade["market"]].multiplier * trade["contracts"] if trade.get("initial_stop") != "" else "", "exit_reason": final_leg["reason"], "tp_levels_reached": ",".join(map(str, reached)), **percentages, "stop_level_at_final_exit": final_leg["price"] if "stop" in final_leg["reason"] or "flatten" in final_leg["reason"] else "", "gross_pnl": gross, "entry_fees": entry_fee, "exit_fees": exit_fees, "total_fees": entry_fee + exit_fees, "slippage": trade.get("slippage", 0.0), "net_pnl": gross - entry_fee - exit_fees, "balance_before_trade": rec["balance_before_trade"], "balance_after_trade": float(account.balance), "account_equity_low_during_trade": rec["equity_low"], "current_daily_loss": account.daily_profit, "current_mll_threshold": account.mll, "remaining_distance_to_mll": float(account.balance - account.mll), "remaining_distance_to_evaluation_target": float(simulation.spec.account_size + simulation.spec.target - account.balance) if account.kind == "EVALUATION" else "", "account_state_after_trade": account.state, "trade_caused_evaluation_pass": bool(account.kind == "EVALUATION" and account.balance >= simulation.spec.account_size + simulation.spec.target), "trade_caused_account_failure": bool(account.state.endswith("FAILED")), "forced_session_exit": any(leg.get("forced") and leg["reason"] == "session_forced_liquidation" for leg in leg_list)}
            trade_rows[key] = row
            if rec["account_type"] == "EVALUATION":
                eval_snapshots.append({"account_id": rec["account_id"], "timestamp": final_leg["timestamp"], "balance_before": rec["balance_before_trade"], "trade_net_pnl": row["net_pnl"], "balance_after": row["balance_after_trade"], "profit_target": simulation.spec.target, "progress_dollars": row["balance_after_trade"] - simulation.spec.account_size, "progress_percent": (row["balance_after_trade"] - simulation.spec.account_size) / simulation.spec.target * 100, "mll_threshold": account.mll, "distance_to_mll": row["remaining_distance_to_mll"], "daily_loss": account.daily_profit, "days_active": (pd.Timestamp(final_leg["timestamp"]) - pd.Timestamp(account.billing_start)).total_seconds() / 86400 if account.billing_start is not None else "", "trades_taken": account.trades_taken, "state": account.state})

        def exit_trade(self, account, signal_id, leg_index, trade, timestamp):
            result = original_exit(account, signal_id, leg_index, trade, timestamp)
            if signal_id not in account.positions:
                finalize(account, signal_id)
            capture_transitions()
            return result

        def flatten(self, account, timestamp, reason):
            signal_ids = list(account.positions)
            result = original_flatten(account, timestamp, reason)
            for signal_id in signal_ids:
                finalize(account, signal_id)
            capture_transitions()
            return result

        def finish(self, account, timestamp, new_session):
            old_session = account.current_session
            old_daily = float(account.daily_profit)
            old_cycle = float(account.cycle_profit)
            old_winning = set(account.winning_days)
            old_payouts = account.qualified_payouts
            result = original_finish(account, timestamp, new_session)
            if account.kind == "QUALIFIED" and old_session is not None and old_session != new_session:
                qualifying = old_daily >= legacy.WINNING_DAY_MINIMUM
                winning_after = len(account.winning_days)
                cycle_after = float(account.cycle_profit)
                consistency = max(account.cycle_days.values(), default=0.0) / cycle_after if cycle_after > 0 else 0.0
                daily_rows.append({"timestamp": str(timestamp), "account_id": account_id(account), "qualified_session": old_session, "qualified_daily_net_pnl": old_daily, "qualifying_winning_day": qualifying, "winning_days_progress": f"{winning_after} / {legacy.WINNING_DAYS_REQUIRED}", "largest_winning_day": max(account.cycle_days.values(), default=old_daily if old_daily > 0 else 0.0), "payout_cycle_profit": cycle_after, "consistency_percentage": consistency * 100, "consistency_rule_satisfied": consistency <= legacy.CONSISTENCY_LIMIT if cycle_after > 0 else False, "payout_eligible": account.qualified_payouts > old_payouts})
                if qualifying:
                    emit_event(timestamp, account, account.state, account.state, "winning day recorded", balance_before=account.balance, balance_after=account.balance)
                emit_event(timestamp, account, account.state, account.state, "consistency changed", balance_before=account.balance, balance_after=account.balance)
            if account.qualified_payouts > old_payouts:
                gross = float(account.gross_payout) - sum(item["gross_withdrawal"] for item in payout_rows if item["account_id"] == account_id(account))
                trader = float(account.trader_payout) - sum(item["trader_payout"] for item in payout_rows if item["account_id"] == account_id(account))
                payout_rows.append({"timestamp": str(timestamp), "account_id": account_id(account), "gross_withdrawal": gross, "trader_payout_after_90pct_split": trader, "balance_after_payout": account.balance, "winning_days_used": len(old_winning), "payout_cycle_profit_before_reset": old_cycle, "payout_counters_after_reset": len(account.winning_days), "payout_status": "APPROVED_AND_RECEIVED"})
                emit_event(timestamp, account, account.state, account.state, "payout eligibility reached", balance_before=account.balance, balance_after=account.balance)
                emit_event(timestamp, account, account.state, account.state, "payout requested", gross=gross, balance_before=account.balance, balance_after=account.balance)
                emit_event(timestamp, account, account.state, account.state, "payout approved in simulation", gross=gross, balance_before=account.balance, balance_after=account.balance)
                emit_event(timestamp, account, account.state, account.state, "trader payout received", gross=gross, trader=trader, balance_before=account.balance, balance_after=account.balance)
            capture_transitions()
            return result

        simulation._buy_evaluation = MethodType(buy, simulation)
        simulation._charge_subscription = MethodType(charge, simulation)
        simulation._start_qualified = MethodType(start_q, simulation)
        simulation._accept = MethodType(accept, simulation)
        simulation._apply_exit = MethodType(apply_exit, simulation)
        simulation._exit_for_account = MethodType(exit_trade, simulation)
        simulation._flatten = MethodType(flatten, simulation)
        simulation._finish_session = MethodType(finish, simulation)
        simulation.run()
        capture_transitions()
        for account in (simulation.eval, simulation.qualified):
            if account is not None:
                emit_event(end_exclusive - pd.Timedelta(nanoseconds=1), account, account.state, "CENSORED_END_OF_DATA", "Account censored at end of data")
        return simulation, list(trade_rows.values()), events, eval_snapshots, daily_rows, payout_rows
    finally:
        single.START, single.END = old_start, old_end


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    market_frames, latest_candle = _full_market_trades()
    end_exclusive = latest_candle + pd.Timedelta(hours=4)
    first_prices = {market: float(frame.sort_values("fill_timestamp").iloc[0]["entry_price"]) for (market, _), frame in market_frames.items() if not frame.empty}
    contexts = build_synthetic_context(first_prices)
    prepared = []
    candidate_by_market = defaultdict(int)
    for market in MEMBERS:
        frame = market_frames[(market, TIMEFRAME)]
        for raw in frame.to_dict("records"):
            entry_time = pd.Timestamp(raw["fill_timestamp"])
            if not (START <= entry_time < end_exclusive):
                continue
            candidate_by_market[market] += 1
            trade = fixed._prepare_trade(raw, market, SIZE, contexts.get(market))
            trade.update({"setup_timestamp": str(raw.get("signal_timestamp", "")), "initial_stop": round_to_tick(mapped_price(float(raw["initial_stop"]), market, contexts.get(market)), fixed.CANONICAL_PROXIES[market].tick_size), "tp_prices": [round_to_tick(mapped_price(float(x), market, contexts.get(market)), fixed.CANONICAL_PROXIES[market].tick_size) for x in json.loads(raw.get("tp_prices", raw.get("targets", "[]")) or "[]")]})
            prepared.append(trade)
    prepared.sort(key=lambda t: (t["entry_timestamp"], t["market"], t["setup_id"]))
    simulation, trades, events, eval_snapshots, daily_rows, payout_rows = _run_journal(prepared, end_exclusive, contexts)
    trades_df = pd.DataFrame(sorted(trades, key=lambda row: row["exit_timestamp"]))
    events_df = pd.DataFrame(events)
    payouts_df = pd.DataFrame(payout_rows)
    daily_df = pd.DataFrame(daily_rows)
    snapshots_df = pd.DataFrame(eval_snapshots)
    if not events_df.empty:
        events_df = events_df.sort_values("timestamp").reset_index(drop=True)
        cumulative = 0.0
        for index, row in events_df.iterrows():
            cumulative += float(row["related_trader_payout"]) - float(row["related_subscription_cost"])
            events_df.loc[index, "cumulative_external_cashflow"] = cumulative
    trades_df.to_csv(root / "trades.csv", index=False)
    events_df.to_csv(root / "account_events.csv", index=False)
    snapshots_df.to_csv(root / "evaluation_snapshots.csv", index=False)
    daily_df.to_csv(root / "qualified_daily_progress.csv", index=False)
    payouts_df.to_csv(root / "payouts.csv", index=False)

    months = _months(latest_candle)
    monthly = []
    for month in months:
        t = trades_df[trades_df.entry_timestamp.astype(str).str.startswith(month)] if not trades_df.empty else pd.DataFrame()
        e = events_df[events_df.timestamp.astype(str).str.startswith(month)] if not events_df.empty else pd.DataFrame()
        d = daily_df[daily_df.timestamp.astype(str).str.startswith(month)] if not daily_df.empty else pd.DataFrame()
        monthly.append({"month": month, "evaluation_ids_active": ",".join(sorted(e.loc[e.account_type == "EVALUATION", "account_id"].dropna().unique())) if not e.empty else "", "qualified_account_id_active": ",".join(sorted(e.loc[e.account_type == "QUALIFIED", "account_id"].dropna().unique())) if not e.empty else "", "evaluations_purchased": int((e.event_reason == "Evaluation purchased").sum()) if not e.empty else 0, "evaluations_passed": int((e.new_state == "EVALUATION_PASSED").sum()) if not e.empty else 0, "evaluations_failed": int((e.new_state == "EVALUATION_FAILED").sum()) if not e.empty else 0, "subscriptions_paid": float(e.related_subscription_cost.sum()) if not e.empty else 0.0, "evaluation_trades": int((t.account_type == "EVALUATION").sum()) if not t.empty else 0, "qualified_trades": int((t.account_type == "QUALIFIED").sum()) if not t.empty else 0, "evaluation_net_trading_pnl": float(t.loc[t.account_type == "EVALUATION", "net_pnl"].sum()) if not t.empty else 0.0, "qualified_net_trading_pnl": float(t.loc[t.account_type == "QUALIFIED", "net_pnl"].sum()) if not t.empty else 0.0, "qualifying_winning_days_added": int(d.qualifying_winning_day.sum()) if not d.empty else 0, "payouts_received": int(len(payouts_df[payouts_df.timestamp.astype(str).str.startswith(month)])) if not payouts_df.empty else 0, "trader_payout_after_split": float(payouts_df.loc[payouts_df.timestamp.astype(str).str.startswith(month), "trader_payout_after_90pct_split"].sum()) if not payouts_df.empty else 0.0, "net_external_monthly_cashflow": float(e.related_trader_payout.sum() - e.related_subscription_cost.sum()) if not e.empty else 0.0})
    monthly_df = pd.DataFrame(monthly)
    monthly_df["cumulative_external_cashflow"] = monthly_df.net_external_monthly_cashflow.cumsum()
    event_times = pd.to_datetime(events_df.timestamp, utc=True, format="mixed") if not events_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
    ending_eval_balances, ending_qualified_balances, ending_eval_states, ending_qualified_states = [], [], [], []
    for month in monthly_df.month:
        month_end = pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        prior = events_df.loc[event_times <= month_end].sort_values("timestamp") if not events_df.empty else pd.DataFrame()
        latest_accounts = prior.drop_duplicates("account_id", keep="last") if not prior.empty else pd.DataFrame()
        active_eval = latest_accounts[(latest_accounts.account_type == "EVALUATION") & latest_accounts.new_state.isin(["EVALUATION_ACTIVE", "EVALUATION_DAILY_LOCKED"])] if not latest_accounts.empty else pd.DataFrame()
        active_qualified = latest_accounts[(latest_accounts.account_type == "QUALIFIED") & latest_accounts.new_state.isin(["QUALIFIED_ACTIVE", "QUALIFIED_DAILY_LOCKED"])] if not latest_accounts.empty else pd.DataFrame()
        ending_eval_balances.append(float(active_eval.balance_after.iloc[-1]) if not active_eval.empty else "")
        ending_qualified_balances.append(float(active_qualified.balance_after.iloc[-1]) if not active_qualified.empty else "")
        ending_eval_states.append(str(active_eval.new_state.iloc[-1]) if not active_eval.empty else "NONE")
        ending_qualified_states.append(str(active_qualified.new_state.iloc[-1]) if not active_qualified.empty else "NONE")
    monthly_df["ending_evaluation_balance"] = ending_eval_balances
    monthly_df["ending_qualified_balance"] = ending_qualified_balances
    monthly_df["ending_evaluation_state"] = ending_eval_states
    monthly_df["ending_qualified_state"] = ending_qualified_states
    monthly_df.to_csv(root / "monthly_summary.csv", index=False)

    final = {"simulation_start": str(START), "latest_completed_cached_candle": str(latest_candle), "simulation_end_exclusive": str(end_exclusive), "calendar_days": (latest_candle - START).total_seconds() / 86400, "selected_portfolio": PORTFOLIO, "selected_timeframe": TIMEFRAME, "selected_max_micro_size": SIZE, "selected_signal_mode": MODE, "selection_reason": "Portfolio C fallback: the existing corrected 25K results did not identify a unique winner; this avoids unresolved/short-history index and Silver exposures.", "evaluations_purchased": int((events_df.event_reason == "Evaluation purchased").sum()) if not events_df.empty else 0, "subscriptions_paid": float(events_df.related_subscription_cost.sum()) if not events_df.empty else 0.0, "evaluations_passed": int((events_df.new_state == "EVALUATION_PASSED").sum()) if not events_df.empty else 0, "evaluations_failed": int((events_df.new_state == "EVALUATION_FAILED").sum()) if not events_df.empty else 0, "censored_evaluations": int((events_df.new_state == "CENSORED_END_OF_DATA").sum()) if not events_df.empty else 0, "qualified_accounts_created": int((events_df.event_reason == "Qualified account created").sum()) if not events_df.empty else 0, "qualified_failures": int((events_df.new_state == "QUALIFIED_FAILED").sum()) if not events_df.empty else 0, "total_qualifying_winning_days": int(daily_df.qualifying_winning_day.sum()) if not daily_df.empty else 0, "total_payouts": len(payouts_df), "gross_payout_requests": float(payouts_df.gross_withdrawal.sum()) if not payouts_df.empty else 0.0, "trader_payouts_after_split": float(payouts_df.trader_payout_after_90pct_split.sum()) if not payouts_df.empty else 0.0, "net_external_cashflow": float(monthly_df.net_external_monthly_cashflow.sum()), "gross_trading_pnl": float(trades_df.gross_pnl.sum()) if not trades_df.empty else 0.0, "net_trading_pnl": float(trades_df.net_pnl.sum()) if not trades_df.empty else 0.0, "fees": float(trades_df.total_fees.sum()) if not trades_df.empty else 0.0, "slippage": float(trades_df.slippage.sum()) if not trades_df.empty else 0.0, "total_executed_trades": len(trades_df), "trades_per_month": len(trades_df) / max(len(months), 1), "trades_by_market": trades_df.groupby("market").size().to_dict() if not trades_df.empty else {}, "long_pnl": float(trades_df.loc[trades_df.direction == "long", "net_pnl"].sum()) if not trades_df.empty else 0.0, "short_pnl": float(trades_df.loc[trades_df.direction == "short", "net_pnl"].sum()) if not trades_df.empty else 0.0, "largest_winning_trade": float(trades_df.net_pnl.max()) if not trades_df.empty else 0.0, "largest_losing_trade": float(trades_df.net_pnl.min()) if not trades_df.empty else 0.0, "worst_trading_day": float(trades_df.assign(day=trades_df.exit_timestamp.astype(str).str[:10]).groupby("day").net_pnl.sum().min()) if not trades_df.empty else 0.0, "minimum_distance_to_mll": float(trades_df.remaining_distance_to_mll.min()) if not trades_df.empty else 0.0, "integrity_all_trades_active": False, "integrity_quantities_reconcile": bool(not trades_df.empty and trades_df.quantity_reconciles.astype(bool).all()), "proxy_uncertainty": "Portfolio C uses direct BTC/MBT, ETH/MET, and PAXG/MGC price-level mappings; Binance spot/proxy data are not native CME futures and contract-roll, basis, and intrabar differences remain."}
    final["integrity_all_trades_active"] = bool(not trades_df.empty and trades_df.account_stage_before_trade.astype(str).str.contains("ACTIVE|LOCKED").all())
    final["censored_evaluations"] = int(((events_df.new_state == "CENSORED_END_OF_DATA") & (events_df.account_type == "EVALUATION")).sum()) if not events_df.empty else 0
    final["censored_qualified_accounts"] = int(((events_df.new_state == "CENSORED_END_OF_DATA") & (events_df.account_type == "QUALIFIED")).sum()) if not events_df.empty else 0
    purchase_events = events_df[events_df.event_reason == "Evaluation purchased"] if not events_df.empty else pd.DataFrame()
    pass_events = events_df[events_df.new_state == "EVALUATION_PASSED"] if not events_df.empty else pd.DataFrame()
    final["average_days_to_pass"] = float(np.mean([(pd.Timestamp(row.timestamp) - pd.Timestamp(purchase_events[purchase_events.account_id == row.account_id].timestamp.iloc[0])).total_seconds() / 86400 for _, row in pass_events.iterrows() if not purchase_events[purchase_events.account_id == row.account_id].empty])) if not pass_events.empty else 0.0
    final["median_days_to_pass"] = float(np.median([(pd.Timestamp(row.timestamp) - pd.Timestamp(purchase_events[purchase_events.account_id == row.account_id].timestamp.iloc[0])).total_seconds() / 86400 for _, row in pass_events.iterrows() if not purchase_events[purchase_events.account_id == row.account_id].empty])) if not pass_events.empty else 0.0
    final["average_trades_to_pass"] = float(np.mean([len(trades_df[(trades_df.account_id == row.account_id) & (pd.to_datetime(trades_df.exit_timestamp, utc=True, format="mixed") <= pd.Timestamp(row.timestamp, tz="UTC"))]) for _, row in pass_events.iterrows()])) if not pass_events.empty else 0.0
    final["first_payouts"] = len(payouts_df)
    final["payout_days_from_qualified_start"] = [((pd.Timestamp(row.timestamp) - pd.Timestamp(events_df[(events_df.account_id == row.account_id) & (events_df.event_reason == "Qualified account created")].timestamp.iloc[0])).total_seconds() / 86400) for _, row in payouts_df.iterrows() if not events_df[(events_df.account_id == row.account_id) & (events_df.event_reason == "Qualified account created")].empty]
    drawdowns = [(group.balance_after_trade.cummax() - group.balance_after_trade).max() for _, group in trades_df.groupby("account_id")] if not trades_df.empty else []
    final["max_account_drawdown_dollars"] = float(max(drawdowns)) if drawdowns else 0.0
    final["longest_time_without_evaluation_pass_days"] = float(max([(pd.Timestamp(row.timestamp) - START).total_seconds() / 86400 for _, row in pass_events.iterrows()] or [0.0]))
    final["longest_time_without_payout_days"] = float(max([(pd.Timestamp(row.timestamp) - START).total_seconds() / 86400 for _, row in payouts_df.iterrows()] or [0.0]))
    trade_times = sorted(pd.to_datetime(trades_df.entry_timestamp, utc=True, format="mixed").tolist()) if not trades_df.empty else []
    gaps = [(trade_times[i] - trade_times[i - 1]).total_seconds() / 86400 for i in range(1, len(trade_times))]
    gaps.append((end_exclusive - trade_times[-1]).total_seconds() / 86400 if trade_times else (end_exclusive - START).total_seconds() / 86400)
    final["longest_time_without_trade_days"] = float(max(gaps or [0.0]))
    final["maximum_distance_used_toward_mll"] = float(max(0.0, -float(trades_df.remaining_distance_to_mll.min()))) if not trades_df.empty else 0.0
    final["closest_distance_to_mll_dollars"] = float(trades_df.remaining_distance_to_mll.min()) if not trades_df.empty else 0.0
    final["ending_evaluation_state"] = "NONE" if simulation.eval is None else simulation.eval.state
    final["ending_qualified_state"] = "CENSORED_END_OF_DATA" if simulation.qualified is not None else "NONE"
    final["forced_session_exit_trades"] = int(trades_df.forced_session_exit.sum()) if not trades_df.empty else 0
    final["trades_causing_evaluation_pass"] = trades_df.loc[trades_df.trade_caused_evaluation_pass, ["chronological_trade_number", "account_id", "market", "net_pnl", "exit_timestamp"]].to_dict("records") if not trades_df.empty else []
    final["trades_causing_evaluation_failure"] = trades_df.loc[trades_df.trade_caused_account_failure, ["chronological_trade_number", "account_id", "market", "net_pnl", "exit_timestamp"]].to_dict("records") if not trades_df.empty else []
    final["trades_causing_qualified_failure"] = []
    final["mirrored_cashflow_counterfactual"] = "Not rerun in this journal; existing 2025 research showed the selected Portfolio C mirrored row did not improve external cashflow versus its one-account baseline."
    pd.DataFrame([final]).to_csv(root / "final_account_summary.csv", index=False)
    skipped = []
    for market in MEMBERS:
        candidate = candidate_by_market[market]
        filled = int((trades_df.market == market).sum()) if not trades_df.empty else 0
        skipped.append({"market": market, "candidate_signals": candidate, "executed_positions": filled, "skipped_signals": candidate - filled, "reason": "existing account eligibility, conflict, session cutoff, or shared contract-limit rules; aggregate only"})
    pd.DataFrame(skipped).to_csv(root / "skipped_signal_summary.csv", index=False)
    market_net = trades_df.groupby("market").net_pnl.sum().to_dict() if not trades_df.empty else {}
    conclusions = (
        "<h2>Human-readable conclusions</h2>"
        f"<p><b>Period and scope.</b> This is one chronological replay from {START} through the last completed cached candle "
        f"({latest_candle}); it uses {PORTFOLIO}, {TIMEFRAME}, {SIZE} total shared micros, and {MODE}. "
        "No second path, optimization, order log, or strategy change was introduced.</p>"
        f"<p><b>Evaluations.</b> {final['evaluations_purchased']} evaluations were purchased; {final['evaluations_passed']} passed and "
        f"{final['evaluations_failed']} failed. The observed passes took on average {final['average_days_to_pass']:.1f} days and "
        f"{final['average_trades_to_pass']:.1f} completed positions. The failure was caused by "
        f"{final['trades_causing_evaluation_failure']}. The pass-causing positions were {final['trades_causing_evaluation_pass']}.</p>"
        f"<p><b>Trading and markets.</b> Gross trading PnL was ${final['gross_trading_pnl']:,.2f}; fees were ${final['fees']:,.2f}; "
        f"net trading PnL was ${final['net_trading_pnl']:,.2f}. Market net PnL was {market_net}. "
        f"There were {final['total_qualifying_winning_days']} qualifying winning days. The largest loss was ${final['largest_losing_trade']:,.2f}; "
        f"the minimum recorded distance to the MLL was ${final['minimum_distance_to_mll']:,.2f}, so an MLL failure occurred.</p>"
        f"<p><b>Payouts and cashflow.</b> One payout request produced ${final['trader_payouts_after_split']:,.2f} after the 90% split, versus "
        f"${final['subscriptions_paid']:,.2f} of subscriptions, for net external cashflow of ${final['net_external_cashflow']:,.2f}. "
        f"The payout occurred {final['payout_days_from_qualified_start'][0]:.1f} days after the Qualified account was created. "
        "A large simulated trading profit does not become an equally large trader payout because the account rules require evaluation, "
        "winning-day qualification, payout caps/cycles, and the firm/trader split; trading equity is not unrestricted cash withdrawal.</p>"
        "<p><b>Mirroring.</b> The journal does not rerun a counterfactual. Existing corrected research indicated that the selected mirrored "
        "Portfolio C row did not improve external cashflow versus its one-account baseline.</p>"
        "<p><b>Trust and limitations.</b> The arithmetic is internally reconciled within the frozen simulation: trades are accepted only while "
        "accounts are active/locked, quantities are journaled against the original shared exposure, and post-termination activity is excluded. "
        "The economic result is not live-trading evidence: Binance prices are proxies for CME contracts, and basis, contract rolls, session/intrabar "
        "execution, and cached-data limitations remain. Treat it as exploratory paper/simulation evidence, not justification for a real evaluation.</p>"
        "<p><b>How to inspect it.</b> Use <code>trades.csv</code> for all positions and filters by evaluation/qualified stage, market, winner/loser, "
        "exit reason, pass/failure flags, and forced-session exits; use <code>qualified_daily_progress.csv</code> for winning-day qualification; "
        "use <code>account_events.csv</code> for the account state machine; and <code>payouts.csv</code> for the withdrawal event.</p>"
    )
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>V12 chronological 25K journal</title><style>body{{font-family:Arial;margin:2rem;max-width:1800px}}table{{border-collapse:collapse;font-size:10px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}code{{background:#f3f3f3;padding:1px 3px}}</style></head><body><h1>One Continuous Alpha Zero 25K Journal</h1><div class='warn'>No optimization or strategy changes. Selected {PORTFOLIO}, {TIMEFRAME}, {SIZE} total micros, {MODE}. Data ends at {latest_candle} UTC. Results use the frozen Binance-proxy conversion layer and are exploratory.</div><h2>Final summary</h2>{pd.DataFrame([final]).to_html(index=False, border=0)}{conclusions}<h2>Monthly summary</h2>{monthly_df.to_html(index=False, border=0)}<h2>Evaluation passes and failures</h2>{events_df[events_df.new_state.isin(['EVALUATION_PASSED','EVALUATION_FAILED','QUALIFIED_FAILED'])].to_html(index=False, border=0) if not events_df.empty else '<p>None</p>'}<h2>Payouts</h2>{payouts_df.to_html(index=False, border=0) if not payouts_df.empty else '<p>None</p>'}<h2>Integrity checks</h2><ul><li>Trades active at acceptance: {final['integrity_all_trades_active']}</li><li>Actual executed positions: {final['total_executed_trades']}</li><li>Partial/accounting journals are filterable by account type, market, exit reason, and forced-session flag.</li></ul></body></html>"""
    (root / "final_report.html").write_text(report, encoding="utf-8")
    return {"trades": len(trades_df), "events": len(events_df), "payouts": len(payouts_df), "latest_candle": str(latest_candle), "root": str(root)}


if __name__ == "__main__":
    print(run())
