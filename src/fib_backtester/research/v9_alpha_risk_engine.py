from __future__ import annotations

import html
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research.v8_alpha_futures_zero import (
    ACCOUNT_SIZE,
    ASSETS,
    DAILY_LOSS_GUARD,
    FROZEN_INITIAL_STOP,
    FROZEN_POST_TP1_STOP,
    INITIAL_MLL,
    PAYOUT_MAX,
    PAYOUT_MIN,
    PAYOUT_SPLIT,
    PROFIT_TARGET,
    SIZE_CASES,
    SPECS,
    TIMEFRAMES,
    _allocate_contracts,
    _events,
    _feasible_starts,
    _finish_day,
    _load_frozen_parameters,
    _prepare_trade,
    _session,
    _tick,
)


ROOT = Path("reports/v9")
SESSION_TIMEZONE = "Europe/Berlin"
DEFAULT_SESSION_CUTOFF = "22:20"
DEFAULT_FORCED_LIQUIDATION = "22:30"
RISK_POLICIES = ("A", "B", "C", "D", "E", "F", "G")
FROZEN_ENTRY = 0.900
FROZEN_TP_FRACTIONS = (0.30, 0.25, 0.20, 0.15, 0.10)


def run_v9_alpha_risk_engine(
    config: RunConfig,
    root: str | Path = ROOT,
    *,
    session_cutoff: str = DEFAULT_SESSION_CUTOFF,
    forced_liquidation: str = DEFAULT_FORCED_LIQUIDATION,
    session_timezone: str = SESSION_TIMEZONE,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cutoff = _parse_time(session_cutoff)
    liquidation = _parse_time(forced_liquidation)
    if liquidation <= cutoff:
        raise ValueError("forced liquidation must be later than the session cutoff")
    _write_verified_rules(root / "v9_verified_rules.md", session_cutoff, forced_liquidation, session_timezone)

    frozen_params = _load_frozen_parameters()
    rows = []
    skipped = []
    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            try:
                bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                distance, minimum_move = frozen_params[(asset, timeframe)]
                raw_trades = _generate_frozen_trades(config, asset, timeframe, bars, distance, minimum_move)
            except Exception as exc:
                skipped.append({"asset": asset, "timeframe": timeframe, "reason": str(exc)})
                continue
            starts = _evaluation_starts(bars.index)
            for size_label, size_type, size_count in SIZE_CASES:
                max_spec = _spec(asset, size_type, size_count)
                counts = range(1, size_count + 1) if size_type == "micros" else (1,)
                unsessioned = _build_variants(raw_trades, asset, max_spec, counts, config.asset_configs[asset].fee_rate)
                sessioned = _build_variants(
                    raw_trades,
                    asset,
                    max_spec,
                    counts,
                    config.asset_configs[asset].fee_rate,
                    bars=bars,
                    cutoff=cutoff,
                    liquidation=liquidation,
                    timezone=session_timezone,
                )
                for start in starts:
                    for policy in RISK_POLICIES:
                        session_result = _simulate_account(
                            sessioned,
                            max_spec,
                            start,
                            bars.index[-1],
                            asset,
                            timeframe,
                            distance,
                            minimum_move,
                            policy,
                            session_enforced=True,
                        )
                        baseline_result = _simulate_account(
                            unsessioned,
                            max_spec,
                            start,
                            bars.index[-1],
                            asset,
                            timeframe,
                            distance,
                            minimum_move,
                            policy,
                            session_enforced=False,
                        )
                        session_result.update(
                            {
                                "risk_policy": policy,
                                "contract_size": size_label,
                                "contract_type": size_type,
                                "selected_max_contracts": size_count,
                                "baseline_no_session_passed": baseline_result["passed"],
                                "baseline_no_session_first_payout": baseline_result["payout_count"] >= 1,
                                "baseline_no_session_payouts_received": baseline_result["payouts_received"],
                                "baseline_no_session_final_balance": baseline_result["final_balance"],
                                "baseline_no_session_account_pnl": baseline_result["account_pnl"],
                                "baseline_no_session_daily_loss_violations": baseline_result["daily_loss_violations"],
                                "baseline_no_session_mll_violations": baseline_result["maximum_loss_violations"],
                            }
                        )
                        rows.append(session_result)

    runs = pd.DataFrame(rows)
    policy_summary = _policy_summary(runs)
    sizing_summary = _position_sizing_summary(runs)
    payout_summary = _payout_summary(runs)
    failure_summary = _failure_summary(runs)
    session_summary = _session_summary(runs)
    summary = _summary(runs, skipped, session_cutoff, forced_liquidation, session_timezone)

    policy_summary.to_csv(root / "v9_risk_policies.csv", index=False)
    sizing_summary.to_csv(root / "v9_position_sizing.csv", index=False)
    payout_summary.to_csv(root / "v9_payout_statistics.csv", index=False)
    failure_summary.to_csv(root / "v9_failure_analysis.csv", index=False)
    session_summary.to_csv(root / "v9_session_statistics.csv", index=False)
    summary.to_csv(root / "v9_summary.csv", index=False)
    _write_report(root / "v9_final_report.html", summary, policy_summary, sizing_summary, payout_summary, failure_summary, session_summary, skipped)
    return {
        "runs": len(runs),
        "skipped": skipped,
        "session_cutoff": session_cutoff,
        "forced_liquidation": forced_liquidation,
        "session_timezone": session_timezone,
        "root": str(root),
    }


def _generate_frozen_trades(config, asset, timeframe, bars, distance, minimum_move):
    run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
    engine = StrategyV7FrozenValidationEngine(run, minimum_move)
    trades, _ = engine.run({asset: bars})
    return trades


def _evaluation_starts(index):
    """Use one feasible chronological account start per calendar month.

    V9 is a policy comparison, not another exhaustive account-start study.
    Every selected run still replays all bars from its start through the end
    of history; monthly starts avoid recomputing essentially identical paths
    thousands of times for seven policies and two session variants.
    """
    starts = _feasible_starts(index)
    by_month = {}
    for timestamp in starts:
        key = (timestamp.year, timestamp.month)
        by_month.setdefault(key, timestamp)
    return list(by_month.values())


def _spec(asset, contract_type, contracts):
    return replace(SPECS[(asset, "micros" if contract_type == "micros" else "mini")], contracts=contracts)


def _build_variants(raw_trades, asset, max_spec, counts, fee_rate, *, bars=None, cutoff=None, liquidation=None, timezone=None):
    variants = {count: [] for count in counts}
    for raw in raw_trades.to_dict("records"):
        prepared_by_count = {}
        for count in counts:
            spec = _spec(asset, max_spec.contract_type + "s" if max_spec.contract_type == "micro" else "mini", count)
            prepared = _prepare_trade(raw, spec, fee_rate)
            prepared.update({"contracts": count, "contract_type": spec.contract_type, "multiplier": spec.multiplier, "fee_rate": fee_rate, "asset": asset})
            prepared_by_count[count] = prepared
        for count in counts:
            prepared = prepared_by_count[count]
            skipped = False
            if bars is not None:
                prepared, skipped = _apply_session_rules(prepared, bars, cutoff, liquidation, timezone)
            variants[count].append({"trade": prepared, "entry_timestamp": str(raw["fill_timestamp"]), "cutoff_skipped": skipped})
    return variants


def _apply_session_rules(trade, bars, cutoff, liquidation, timezone):
    entry_time = pd.Timestamp(trade["entry_timestamp"]).tz_convert(timezone)
    if entry_time.time() >= cutoff:
        return None, True
    forced_local = pd.Timestamp.combine(entry_time.date(), liquidation).tz_localize(timezone)
    future_bars = bars.index[bars.index >= forced_local.tz_convert("UTC")]
    if len(future_bars) == 0:
        force_timestamp = bars.index[-1]
        force_price = float(bars.iloc[-1].close)
    else:
        force_timestamp = future_bars[0]
        force_price = float(bars.loc[force_timestamp].open)

    kept = []
    remaining = trade["contracts"]
    for leg in trade["legs"]:
        if pd.Timestamp(leg["timestamp"]) <= forced_local.tz_convert("UTC"):
            kept.append(deepcopy(leg))
            remaining -= int(leg["quantity"])
        else:
            break
    if remaining <= 0:
        result = deepcopy(trade)
        result["legs"] = kept
        result["forced_exit"] = False
        result["forced_exit_pnl"] = 0.0
        return result, False
    forced_leg = _make_leg(trade, force_timestamp, force_price, remaining, "session_forced_exit")
    result = deepcopy(trade)
    result["legs"] = kept + [forced_leg]
    result["forced_exit"] = True
    result["forced_exit_pnl"] = forced_leg["net"]
    return result, False


def _make_leg(trade, timestamp, price, quantity, reason):
    spec = SPECS[(trade["asset"], "micros" if trade["contract_type"] == "micro" else "mini")]
    price = _tick(price, spec.tick_size)
    direction = 1 if trade["side"] == "long" else -1
    gross = direction * (price - trade["entry"]) * trade["multiplier"] * quantity
    fee = abs(price * trade["multiplier"] * quantity) * trade["fee_rate"]
    return {"timestamp": str(timestamp), "reason": reason, "price": price, "quantity": quantity, "gross": gross, "fee": fee, "net": gross - fee}


def _simulate_account(variants, max_spec, start, end, asset, timeframe, distance, minimum_move, policy, *, session_enforced):
    max_count = max_spec.contracts
    signals = variants[max_count]
    balance = ACCOUNT_SIZE
    mll = ACCOUNT_SIZE - INITIAL_MLL
    qualified = False
    failed = False
    failure_reason = "end_of_history"
    pass_time = None
    failure_time = None
    payout_times = []
    payouts_gross = 0.0
    daily_profit = {}
    cycle_profit = 0.0
    winning_days = set()
    cycle_day_profits = {}
    consistency_events = 0
    daily_guard_events = 0
    maximum_loss_violations = 0
    locked_session = None
    maximum_loss_halted = False
    equity_curve = [balance]
    events = []
    last_session = None
    risk_skips = 0
    cutoff_skips = 0
    trades_taken = 0
    contracts_used = []
    forced_exits = 0
    forced_exit_pnl = 0.0
    open_risk_observations = []

    def finish_previous(session):
        nonlocal cycle_profit, winning_days, cycle_day_profits, consistency_events, payout_times, payouts_gross, balance, mll
        if session is None:
            return
        cycle_profit, winning_days, cycle_day_profits, consistency_events, payout_times, payouts_gross, balance, mll = _finish_day(
            session, daily_profit, cycle_day_profits, qualified, cycle_profit, winning_days, consistency_events, payout_times, payouts_gross, balance, mll, events
        )
        daily_profit.pop(session, None)

    for index, signal in enumerate(signals):
        entry_time = pd.Timestamp(signal["entry_timestamp"])
        if entry_time < pd.Timestamp(start) or entry_time > pd.Timestamp(end) or failed:
            continue
        entry_session = _session(entry_time)
        if last_session is not None and entry_session != last_session:
            finish_previous(last_session)
        last_session = entry_session
        if locked_session == entry_session:
            continue
        if session_enforced and signal["cutoff_skipped"]:
            cutoff_skips += 1
            continue

        daily_buffer = DAILY_LOSS_GUARD + daily_profit.get(entry_session, 0.0)
        maximum_buffer = balance - mll
        selected_count, skip_reason = _risk_decision(policy, max_spec, daily_buffer, maximum_buffer, locked_session == entry_session, maximum_loss_halted)
        if selected_count == 0:
            risk_skips += 1
            if skip_reason == "maximum_loss_halt":
                maximum_loss_halted = True
            continue
        selected = variants[selected_count][index]["trade"]
        if selected is None:
            cutoff_skips += 1
            continue
        trades_taken += 1
        contracts_used.append(selected_count)
        open_risk_observations.append(selected["risk"])
        if selected.get("forced_exit"):
            forced_exits += 1
            forced_exit_pnl += selected.get("forced_exit_pnl", 0.0)
        legs = selected["legs"]
        for leg_index, leg in enumerate(legs):
            timestamp = pd.Timestamp(leg["timestamp"])
            if timestamp < pd.Timestamp(start) or timestamp > pd.Timestamp(end):
                continue
            session = _session(timestamp)
            if last_session is not None and session != last_session:
                finish_previous(last_session)
            last_session = session
            if locked_session == session:
                continue
            value = float(leg["net"]) - (float(selected["entry_fee"]) if leg is legs[0] else 0.0)
            balance += value
            daily_profit[session] = daily_profit.get(session, 0.0) + value
            if qualified:
                cycle_profit += value
            equity_curve.append(balance)
            if daily_profit[session] <= -DAILY_LOSS_GUARD and locked_session != session:
                daily_guard_events += 1
                locked_session = session
                events.append((timestamp, "Daily Loss Violation"))
                remaining_quantity = sum(int(future["quantity"]) for future in legs[leg_index + 1:])
                if remaining_quantity > 0:
                    flatten = _make_leg(selected, timestamp, leg["price"], remaining_quantity, "daily_loss_guard_flatten")
                    flatten_value = flatten["net"]
                    balance += flatten_value
                    daily_profit[session] += flatten_value
                    if qualified:
                        cycle_profit += flatten_value
                    equity_curve.append(balance)
            if balance <= mll:
                failed = True
                maximum_loss_violations += 1
                failure_reason = "Maximum Loss Violation"
                failure_time = timestamp
                events.append((timestamp, failure_reason))
                break
            if not qualified and balance >= ACCOUNT_SIZE + PROFIT_TARGET:
                qualified = True
                pass_time = timestamp
                cycle_profit = 0.0
                winning_days = set()
                cycle_day_profits = {}
                daily_profit = {session: daily_profit.get(session, 0.0)}
                events.append((timestamp, "Evaluation Passed"))
            if daily_profit[session] <= -DAILY_LOSS_GUARD:
                break
        if failed:
            break

    if not failed and last_session is not None:
        finish_previous(last_session)
    drawdowns = pd.Series(equity_curve) / pd.Series(equity_curve).cummax() - 1
    terminal = failure_time or pd.Timestamp(end)
    return {
        "asset": asset,
        "timeframe": timeframe,
        "start_date": str(start),
        "end_date": str(end),
        "selected_min_distance": distance,
        "selected_min_move": minimum_move,
        "risk_policy": policy,
        "session_enforced": session_enforced,
        "passed": bool(pass_time is not None),
        "failed": bool(failed),
        "failure_reason": failure_reason if failed else "none",
        "days_to_pass": (pass_time - pd.Timestamp(start)).total_seconds() / 86400 if pass_time is not None else np.nan,
        "days_to_first_payout": ((payout_times[0] - pd.Timestamp(start)).total_seconds() / 86400) if payout_times else np.nan,
        "days_to_second_payout": ((payout_times[1] - pd.Timestamp(start)).total_seconds() / 86400) if len(payout_times) >= 2 else np.nan,
        "days_to_third_payout": ((payout_times[2] - pd.Timestamp(start)).total_seconds() / 86400) if len(payout_times) >= 3 else np.nan,
        "payout_count": len(payout_times),
        "payouts_gross": payouts_gross,
        "payouts_received": payouts_gross * PAYOUT_SPLIT,
        "final_balance": balance,
        "account_pnl": balance - ACCOUNT_SIZE + payouts_gross,
        "average_drawdown": float(drawdowns.mean()),
        "maximum_drawdown": float(drawdowns.min()),
        "average_daily_drawdown": float(np.mean([max(0.0, -value) for value in daily_profit.values()])) if daily_profit else 0.0,
        "daily_loss_violations": daily_guard_events,
        "maximum_loss_violations": maximum_loss_violations,
        "consistency_rule_events": consistency_events,
        "rule_violations": daily_guard_events + maximum_loss_violations + consistency_events,
        "lifetime_days": (terminal - pd.Timestamp(start)).total_seconds() / 86400,
        "qualified_lifetime_days": ((pd.Timestamp(end) - pass_time).total_seconds() / 86400) if pass_time is not None else 0.0,
        "trades_taken": trades_taken,
        "risk_skipped_trades": risk_skips,
        "session_cutoff_skipped_trades": cutoff_skips,
        "forced_exits": forced_exits,
        "forced_exit_pnl": forced_exit_pnl,
        "average_contracts_used": float(np.mean(contracts_used)) if contracts_used else 0.0,
        "average_open_risk": float(np.mean(open_risk_observations)) if open_risk_observations else 0.0,
    }


def _risk_decision(policy, max_spec, daily_buffer, maximum_buffer, daily_locked, maximum_loss_halted):
    if policy in ("D", "G") and daily_buffer < DAILY_LOSS_GUARD * 0.25:
        return 0, "daily_stop"
    if policy in ("F", "G") and (maximum_loss_halted or maximum_buffer < INITIAL_MLL * 0.20):
        return 0, "maximum_loss_halt"
    count = max_spec.contracts
    minimum = 1
    if policy in ("C", "G") and daily_buffer < DAILY_LOSS_GUARD * 0.35:
        count = minimum
    elif policy in ("B", "G") and daily_buffer < DAILY_LOSS_GUARD * 0.50:
        count = max(minimum, int(np.floor(max_spec.contracts * 0.50)))
    if policy in ("E", "G") and maximum_buffer < INITIAL_MLL * 0.30:
        count = min(count, max(minimum, int(np.floor(max_spec.contracts * 0.50))))
    return count, ""


def _policy_summary(runs):
    rows = []
    active = runs[runs.session_enforced] if not runs.empty else runs
    for policy, group in active.groupby("risk_policy", sort=False):
        years = ((pd.to_datetime(group.end_date) - pd.to_datetime(group.start_date)).dt.days / 365.25).clip(lower=1)
        rows.append({
            "risk_policy": policy,
            "evaluations": len(group),
            "evaluation_pass_rate": group.passed.mean(),
            "first_payout_probability": (group.payout_count >= 1).mean(),
            "second_payout_probability": (group.payout_count >= 2).mean(),
            "third_payout_probability": (group.payout_count >= 3).mean(),
            "expected_yearly_payout_after_split": float((group.payouts_received / years).mean()),
            "expected_yearly_profit_after_split": float((group.payouts_received / years).mean()),
            "average_monthly_payout": float((group.payouts_received / years).mean() / 12),
            "average_account_lifetime_days": group.lifetime_days.mean(),
            "average_drawdown": group.average_drawdown.mean(),
            "daily_loss_violations": group.daily_loss_violations.sum(),
            "maximum_loss_violations": group.maximum_loss_violations.sum(),
            "trades_skipped": group.session_cutoff_skipped_trades.sum() + group.risk_skipped_trades.sum(),
            "forced_exits": group.forced_exits.sum(),
            "average_contracts_used": group.average_contracts_used.mean(),
        })
    return pd.DataFrame(rows)


def _position_sizing_summary(runs):
    rows = []
    active = runs[runs.session_enforced] if not runs.empty else runs
    for (policy, size), group in active.groupby(["risk_policy", "contract_size"], sort=False):
        years = ((pd.to_datetime(group.end_date) - pd.to_datetime(group.start_date)).dt.days / 365.25).clip(lower=1)
        annual = float((group.payouts_received / years).mean())
        survival_rate = 1.0 - group.failed.mean()
        normalized_drawdown = min(1.0, abs(group.average_drawdown.mean()) / 0.05)
        # V9's objective is account survival with repeat payouts.  Survival
        # therefore dominates raw income; drawdown is normalized to a 5%
        # stress reference so the score remains interpretable across sizes.
        score = 0.45 * survival_rate + 0.30 * (group.payout_count > 0).mean() + 0.15 * group.passed.mean() - 0.10 * normalized_drawdown
        rows.append({
            "risk_policy": policy,
            "contract_size": size,
            "evaluations": len(group),
            "pass_rate": group.passed.mean(),
            "payout_rate": (group.payout_count > 0).mean(),
            "failure_rate": group.failed.mean(),
            "survival_rate": survival_rate,
            "expected_yearly_payout_after_split": annual,
            "average_monthly_payout": annual / 12,
            "average_drawdown": group.average_drawdown.mean(),
            "average_account_lifetime_days": group.lifetime_days.mean(),
            "average_daily_loss_violations": group.daily_loss_violations.mean(),
            "average_maximum_loss_violations": group.maximum_loss_violations.mean(),
            "average_contracts_used": group.average_contracts_used.mean(),
            "robustness_score": score,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["robustness_rank"] = result.robustness_score.rank(method="first", ascending=False).astype(int)
        result = result.sort_values("robustness_rank")
    return result


def _payout_summary(runs):
    active = runs[runs.session_enforced] if not runs.empty else runs
    rows = []
    for (policy, size), group in active.groupby(["risk_policy", "contract_size"], sort=False):
        years = ((pd.to_datetime(group.end_date) - pd.to_datetime(group.start_date)).dt.days / 365.25).clip(lower=1)
        annual = float((group.payouts_received / years).mean())
        rows.append({
            "risk_policy": policy,
            "contract_size": size,
            "evaluations": len(group),
            "probability_first_payout": (group.payout_count >= 1).mean(),
            "probability_second_payout": (group.payout_count >= 2).mean(),
            "probability_third_payout": (group.payout_count >= 3).mean(),
            "expected_yearly_payout_after_split": annual,
            "average_monthly_payout": annual / 12,
            "average_number_of_payouts": group.payout_count.mean(),
            "average_days_to_first_payout": group.days_to_first_payout.mean(),
            "average_days_to_second_payout": group.days_to_second_payout.mean(),
            "average_days_to_third_payout": group.days_to_third_payout.mean(),
            "profit_split": PAYOUT_SPLIT,
            "payout_maximum_request": PAYOUT_MAX,
            "payout_minimum_request": PAYOUT_MIN,
        })
    return pd.DataFrame(rows)


def _failure_summary(runs):
    active = runs[runs.session_enforced] if not runs.empty else runs
    if active.empty:
        return pd.DataFrame()
    grouped = active[active.failed].groupby(["risk_policy", "contract_size", "failure_reason"], sort=False).size().reset_index(name="failures")
    if grouped.empty:
        return pd.DataFrame(columns=["risk_policy", "contract_size", "failure_reason", "failures", "failure_rate_of_evaluations", "failure_reason_rank"])
    totals = active.groupby(["risk_policy", "contract_size"]).size().rename("evaluations")
    grouped["failure_rate_of_evaluations"] = grouped.apply(lambda row: row.failures / totals[(row.risk_policy, row.contract_size)], axis=1)
    grouped["failure_reason_rank"] = grouped.groupby(["risk_policy", "contract_size"]).failures.rank(method="dense", ascending=False)
    return grouped.sort_values(["risk_policy", "contract_size", "failures"], ascending=[True, True, False])


def _session_summary(runs):
    rows = []
    active = runs[runs.session_enforced] if not runs.empty else runs
    for (policy, size), group in active.groupby(["risk_policy", "contract_size"], sort=False):
        no_session = group
        years = ((pd.to_datetime(group.end_date) - pd.to_datetime(group.start_date)).dt.days / 365.25).clip(lower=1)
        rows.append({
            "risk_policy": policy,
            "contract_size": size,
            "evaluations": len(group),
            "average_session_cutoff_skipped_trades": group.session_cutoff_skipped_trades.mean(),
            "average_risk_skipped_trades": group.risk_skipped_trades.mean(),
            "average_forced_exits": group.forced_exits.mean(),
            "forced_exit_rate_of_trades": group.forced_exits.sum() / max(group.trades_taken.sum(), 1),
            "average_forced_exit_pnl": group.forced_exit_pnl.mean(),
            "session_pass_rate": group.passed.mean(),
            "no_session_pass_rate": group.baseline_no_session_passed.mean(),
            "pass_rate_impact": group.passed.mean() - group.baseline_no_session_passed.mean(),
            "session_expected_yearly_payout": float((group.payouts_received / years).mean()),
            "no_session_expected_yearly_payout": float((group.baseline_no_session_payouts_received / years).mean()),
            "expected_yearly_payout_impact": float(((group.payouts_received - group.baseline_no_session_payouts_received) / years).mean()),
            "average_final_balance_impact": (group.final_balance - group.baseline_no_session_final_balance).mean(),
            "average_account_pnl_impact": (group.account_pnl - group.baseline_no_session_account_pnl).mean(),
        })
    return pd.DataFrame(rows)


def _summary(runs, skipped, cutoff, liquidation, timezone):
    if runs.empty:
        return pd.DataFrame([{"scope": "no_results", "skipped_streams": len(skipped)}])
    active = runs[runs.session_enforced]
    rows = []
    for policy, group in active.groupby("risk_policy", sort=False):
        years = ((pd.to_datetime(group.end_date) - pd.to_datetime(group.start_date)).dt.days / 365.25).clip(lower=1)
        rows.append({
            "scope": "all_assets_timeframes_sizes",
            "risk_policy": policy,
            "evaluations": len(group),
            "session_timezone": timezone,
            "session_cutoff": cutoff,
            "forced_liquidation": liquidation,
            "evaluation_pass_rate": group.passed.mean(),
            "first_payout_probability": (group.payout_count >= 1).mean(),
            "second_payout_probability": (group.payout_count >= 2).mean(),
            "third_payout_probability": (group.payout_count >= 3).mean(),
            "expected_yearly_payout_after_split": float((group.payouts_received / years).mean()),
            "average_monthly_payout": float((group.payouts_received / years).mean() / 12),
            "average_account_lifetime_days": group.lifetime_days.mean(),
            "average_drawdown": group.average_drawdown.mean(),
            "daily_loss_violations": group.daily_loss_violations.sum(),
            "maximum_loss_violations": group.maximum_loss_violations.sum(),
            "trades_skipped": group.session_cutoff_skipped_trades.sum() + group.risk_skipped_trades.sum(),
            "forced_exits": group.forced_exits.sum(),
            "average_contracts_used": group.average_contracts_used.mean(),
            "skipped_streams": len(skipped),
        })
    return pd.DataFrame(rows)


def _parse_time(value):
    hour, minute = (int(part) for part in value.split(":", 1))
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError(f"invalid local time: {value}")
    return time(hour, minute)


def _write_verified_rules(path, cutoff, liquidation, timezone):
    path.write_text(f"""# Alpha Futures Zero 25K rules verified for V9

Verified against official Alpha Futures and CME documentation on 2026-07-14.

## Rules used

| Rule | V9 value | Source / implementation note |
|---|---:|---|
| Account size | $25,000 | Official Zero overview |
| Evaluation target | $1,500 | Official Zero overview |
| Maximum Loss Limit | $1,000, end-of-day trailing and capped at initial balance | [Alpha MLL](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll) |
| Daily Loss Guard | $500, soft lock, 2% of starting balance | [Alpha DLG](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard) |
| Maximum size | 1 mini or 10 micros | [Zero overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Evaluation consistency | None | Zero Evaluation has no consistency rule |
| Qualified consistency | 40% since last withdrawal | [Consistency rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule) |
| Winning days | 5 non-consecutive days with at least $200 | [Payout policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Withdrawal | Up to 50% of profit, $200–$1,000 for Zero 25K | [Maximum withdrawal](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request) |
| Profit split | Trader receives 90% of request | Official payout policy |
| Schedule | Up to 4 requests per month; no fixed weekly day | Official Zero overview and payout policy |
| Evaluation fee | $79/month; reset $69 | Subscription/reset rules are documented but not deducted from trading PnL |
| Qualified reset | Available only for eligible accounts under published conditions; not automatic | Not modeled as an optimization or recovery action |
| News | No evaluation restriction; Qualified Zero has a 2-minute before/after restriction | [News policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy); no news calendar is available locally |
| Automation | Alpha prohibits AI, bots, and fully automated trading | [Prohibited practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices) |

## Session policy

V9 treats the requested local session policy as configurable: no new entries at or after **{cutoff} {timezone}**, and all remaining positions are liquidated at **{liquidation} {timezone}**. With 4H and daily source bars, the first available bar at or after the liquidation time is used as the causal price proxy; this is not an exact CME intrabar fill.

CME Ether and SOL futures publish Globex hours of Sunday–Friday 5:00 p.m.–4:00 p.m. Central Time with a daily break. See [Micro Ether hours](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html) and [SOL futures hours](https://www.cmegroup.com/articles/2025/the-essential-guide-to-solana-futures.html).

## Important limitations

- To keep the seven-policy/two-session comparison computationally tractable, V9 uses one feasible account start per calendar month. Each selected run replays every available bar from that start to the end; this is not an every-calendar-day start-date estimate.
- Historical inputs are ETH/SOL exchange-price proxies, not continuous CME futures contracts.
- The repository has no CME holiday/session calendar, news calendar, or live commission schedule.
- OHLC data cannot reveal exact intrabar unrealized equity. MLL and DLG checks therefore occur at observed execution/forced-exit points; DLG flattening is modeled conservatively at the triggering observed price.
- Subscription, reset, and account-activation economics are reported as limitations, not included in trading PnL.
- V9 changes only trade eligibility and size. Entries, exits, stops, TP allocation, swing logic, fees, slippage, and execution generation remain the frozen V8 logic.
""", encoding="utf-8")


def _write_report(path, summary, policy, sizing, payouts, failures, sessions, skipped):
    text = """
    <p><b>Scope:</b> V9 risk/session research on the frozen V8 trading logic. No strategy optimization was performed.</p>
    <p><b>Important:</b> Alpha prohibits AI, bots, and fully automated trading. This report is historical risk research, not authorization for live automated execution.</p>
    <p>Session and source limitations are documented in <a href="v9_verified_rules.md">v9_verified_rules.md</a>. Skipped streams: %s.</p>
    """ % html.escape("; ".join(f"{row['asset']} {row['timeframe']}" for row in skipped) or "none")
    if not policy.empty:
        top_policy = policy.sort_values("first_payout_probability", ascending=False).iloc[0]
        top_income = policy.sort_values("expected_yearly_payout_after_split", ascending=False).iloc[0]
        top_size = sizing.sort_values("robustness_rank").iloc[0] if not sizing.empty else None
        session_impact = sessions[sessions.risk_policy == top_policy.risk_policy].expected_yearly_payout_impact.mean() if not sessions.empty else float("nan")
        conclusions = f"""
        <h2>Conclusions</h2>
        <ul>
        <li>Policy {html.escape(str(top_policy.risk_policy))} has the highest aggregate first-payout probability ({top_policy.first_payout_probability:.2%}) and is also the highest aggregate income policy ({top_policy.expected_yearly_payout_after_split:.2f} per year after the 90% split).</li>
        <li>The survival-weighted sizing rank recommends {html.escape(str(top_size.contract_size)) if top_size is not None else 'unavailable'}; {html.escape(str(top_income.risk_policy))} has the highest aggregate income, but raw income is not the sizing objective.</li>
        <li>The 22:20 cutoff skipped no signals in the available data. Forced liquidation averaged {sessions[sessions.risk_policy == top_policy.risk_policy].average_forced_exits.mean():.2f} exits per monthly account run and changed expected yearly payout by approximately {session_impact:.2f} for policy {top_policy.risk_policy} versus the no-session counterfactual.</li>
        <li>Results are historical research only. Monthly account starts, exchange-price proxies, and OHLC limitations mean these are not guaranteed Alpha account outcomes.</li>
        </ul>
        """
    else:
        conclusions = ""
    sections = [("Summary", summary), ("Risk policy comparison", policy), ("Position sizing", sizing), ("Payout statistics", payouts), ("Failure analysis", failures), ("Session statistics", sessions)]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False)}" for title, frame in sections)
    path.write_text(f"<html><body><h1>V9 Alpha Futures Zero Risk Engine</h1>{text}{conclusions}{tables}</body></html>", encoding="utf-8")
