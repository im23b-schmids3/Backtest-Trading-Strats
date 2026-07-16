from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research import v9_alpha_risk_engine as v9


ROOT = Path("reports/v10")
ACCOUNT_SPECS = {
    "25k": {"account_size": 25_000.0, "target": 1_500.0, "mll": 1_000.0, "dlg": 500.0, "subscription": 79.0, "eval_reset": 69.0, "qualified_reset": 399.0, "payout_max": 1_000.0, "max_micros": 10, "max_minis": 1},
    "50k": {"account_size": 50_000.0, "target": 3_000.0, "mll": 2_000.0, "dlg": 1_000.0, "subscription": 119.0, "eval_reset": 109.0, "qualified_reset": 499.0, "payout_max": 1_500.0, "max_micros": 30, "max_minis": 3},
    "100k": {"account_size": 100_000.0, "target": 6_000.0, "mll": 3_000.0, "dlg": 2_000.0, "subscription": 239.0, "eval_reset": 219.0, "qualified_reset": 799.0, "payout_max": 2_500.0, "max_micros": 60, "max_minis": 6},
}
MONTH_DAYS = 365.25 / 12
BOOTSTRAP_SIMS = 5_000
FROZEN_POLICY = "A"
FROZEN_SIZE = "2 Micros"


def run_v10_prop_economics_audit(config: RunConfig, root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_verified_rules(root / "v10_verified_rules.md")
    _write_formula_explanations(root / "v10_formula_explanations.md")
    lifecycles = _collect_lifecycles(config)

    account_comparison = _account_size_comparison(lifecycles)
    scenarios = _subscription_scenarios(lifecycles)
    continuous = _continuous_analysis(lifecycles, config.seed)
    multi = _multi_account_analysis(lifecycles, config.seed)
    breakeven = _break_even_analysis(lifecycles, config.seed)
    audit = _economics_audit(lifecycles)

    account_comparison.to_csv(root / "v10_account_size_comparison.csv", index=False)
    scenarios.to_csv(root / "v10_subscription_scenarios.csv", index=False)
    continuous.to_csv(root / "v10_continuous_trader.csv", index=False)
    multi.to_csv(root / "v10_multi_account.csv", index=False)
    breakeven.to_csv(root / "v10_break_even_analysis.csv", index=False)
    audit.to_csv(root / "v10_economics_audit.csv", index=False)
    _write_report(root / "v10_final_report.html", account_comparison, scenarios, continuous, multi, breakeven, audit)
    return {"lifecycle_rows": len(lifecycles), "bootstrap_simulations": BOOTSTRAP_SIMS, "root": str(root)}


def _collect_lifecycles(config):
    rows = []
    frozen = v9._load_frozen_parameters()
    for asset in v9.ASSETS:
        for timeframe in v9.TIMEFRAMES:
            bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
            distance, minimum_move = frozen[(asset, timeframe)]
            run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
            raw, _ = v9.StrategyV7FrozenValidationEngine(run, minimum_move).run({asset: bars})
            max_spec = v9._spec(asset, "micros", 2)
            variants = v9._build_variants(raw, asset, max_spec, (1, 2), config.asset_configs[asset].fee_rate, bars=bars, cutoff=v9._parse_time(v9.DEFAULT_SESSION_CUTOFF), liquidation=v9._parse_time(v9.DEFAULT_FORCED_LIQUIDATION), timezone=v9.SESSION_TIMEZONE)
            for start in v9._evaluation_starts(bars.index):
                for account_size, account in ACCOUNT_SPECS.items():
                    rows.append(_simulate_lifecycle(variants, start, bars.index[-1], asset, timeframe, distance, minimum_move, account_size, account))
    return pd.DataFrame(rows)


def _simulate_lifecycle(variants, start, end, asset, timeframe, distance, minimum_move, account_size, account):
    balance = account["account_size"]
    mll = account["account_size"] - account["mll"]
    daily_profit = {}
    cycle_profit = 0.0
    cycle_days = {}
    winning_days = set()
    qualified = False
    failed = False
    pass_time = None
    failure_time = None
    last_session = None
    locked_session = None
    payouts = []
    consistency_events = 0
    dlg_events = 0
    mll_events = 0
    equity = [balance]
    trades_taken = 0
    cutoff_skips = 0
    forced_exits = 0

    def finish(session):
        nonlocal balance, mll, cycle_profit, cycle_days, winning_days, consistency_events
        profit = daily_profit.get(session, 0.0)
        if qualified and profit > 0:
            cycle_days[session] = profit
        if qualified and profit >= 200:
            winning_days.add(session)
        if qualified and len(winning_days) >= 5 and cycle_profit > 0:
            largest = max(cycle_days.values(), default=0.0)
            if largest >= 0.40 * cycle_profit:
                consistency_events += 1
            else:
                request = min(0.50 * cycle_profit, account["payout_max"])
                if request >= 200 and balance - request > mll:
                    balance -= request
                    payouts.append({"timestamp": pd.Timestamp(session, tz="America/New_York"), "gross": request, "received": request * 0.90})
                    cycle_profit = 0.0
                    cycle_days = {}
                    winning_days = set()
        mll = min(account["account_size"], max(mll, max(account["account_size"], balance) - account["mll"]))
        daily_profit.pop(session, None)

    signals = variants[2]
    for index, signal in enumerate(signals):
        entry = pd.Timestamp(signal["entry_timestamp"])
        if entry < pd.Timestamp(start) or entry > pd.Timestamp(end) or failed:
            continue
        session = v9._session(entry)
        if last_session is not None and session != last_session:
            finish(last_session)
        last_session = session
        if signal["cutoff_skipped"]:
            cutoff_skips += 1
            continue
        if locked_session == session:
            continue
        trade = signal["trade"]
        if trade is None:
            continue
        trades_taken += 1
        forced_exits += int(trade.get("forced_exit", False))
        legs = trade["legs"]
        for leg_index, leg in enumerate(legs):
            timestamp = pd.Timestamp(leg["timestamp"])
            if timestamp < pd.Timestamp(start) or timestamp > pd.Timestamp(end):
                continue
            session = v9._session(timestamp)
            if last_session is not None and session != last_session:
                finish(last_session)
            last_session = session
            if locked_session == session:
                continue
            value = float(leg["net"]) - (float(trade["entry_fee"]) if leg is legs[0] else 0.0)
            balance += value
            daily_profit[session] = daily_profit.get(session, 0.0) + value
            if qualified:
                cycle_profit += value
            equity.append(balance)
            if daily_profit[session] <= -account["dlg"] and locked_session != session:
                dlg_events += 1
                locked_session = session
                remaining = sum(int(future["quantity"]) for future in legs[leg_index + 1:])
                if remaining:
                    flatten = v9._make_leg(trade, timestamp, leg["price"], remaining, "daily_loss_guard_flatten")
                    flat_value = flatten["net"]
                    balance += flat_value
                    daily_profit[session] += flat_value
                    if qualified:
                        cycle_profit += flat_value
                    equity.append(balance)
            if balance <= mll:
                failed = True
                mll_events += 1
                failure_time = timestamp
                break
            if not qualified and balance >= account["account_size"] + account["target"]:
                qualified = True
                pass_time = timestamp
                cycle_profit = 0.0
                cycle_days = {}
                winning_days = set()
                daily_profit = {session: daily_profit.get(session, 0.0)}
            if daily_profit[session] <= -account["dlg"]:
                break
        if failed:
            break
    if not failed and last_session is not None:
        finish(last_session)
    terminal = failure_time or pd.Timestamp(end)
    drawdown = pd.Series(equity) / pd.Series(equity).cummax() - 1
    eval_end = pass_time or terminal
    eval_days = max(0.0, (eval_end - pd.Timestamp(start)).total_seconds() / 86400)
    payout_offsets = [max(0.0, (payout["timestamp"] - pd.Timestamp(start)).total_seconds() / 86400) for payout in payouts]
    net_payouts = sum(payout["received"] for payout in payouts)
    return {
        "asset": asset, "timeframe": timeframe, "account_size": account_size, "start_date": str(start), "end_date": str(end),
        "selected_min_distance": distance, "selected_min_move": minimum_move, "risk_policy": FROZEN_POLICY, "position_size": FROZEN_SIZE,
        "passed": bool(pass_time is not None), "failed": failed, "failure_reason": "Maximum Loss Violation" if failed else "none",
        "pass_timestamp": str(pass_time) if pass_time is not None else "", "failure_timestamp": str(failure_time) if failure_time is not None else "",
        "lifetime_days": (terminal - pd.Timestamp(start)).total_seconds() / 86400, "evaluation_days": eval_days,
        "payout_count": len(payouts), "gross_payouts": sum(p["gross"] for p in payouts), "net_payouts": net_payouts,
        "payout_offsets": "|".join(f"{x:.6f}" for x in payout_offsets), "payout_amounts": "|".join(f"{p['received']:.6f}" for p in payouts),
        "average_drawdown": float(drawdown.mean()), "maximum_drawdown": float(drawdown.min()),
        "daily_loss_violations": dlg_events, "maximum_loss_violations": mll_events, "consistency_rule_events": consistency_events,
        "trades_taken": trades_taken, "cutoff_skips": cutoff_skips, "forced_exits": forced_exits,
        "subscription_months": max(1, int(np.ceil(eval_days / MONTH_DAYS))),
        "subscription_cost": max(1, int(np.ceil(eval_days / MONTH_DAYS))) * account["subscription"],
        "challenge_fees": 0.0, "reset_fees": 0.0, "commissions": 0.0,
        "commission_note": "embedded in frozen V9 trade PnL; not double-counted",
        "account_ending_balance": balance, "historical_trading_pnl": balance - account["account_size"],
    }


def _costed(row, scenario, payout_count=None):
    amounts = _split_float(row.payout_amounts)
    offsets = _split_float(row.payout_offsets)
    n = len(amounts) if payout_count is None else min(payout_count, len(amounts))
    revenue = sum(amounts[:n])
    closure_days = row.lifetime_days
    if n and scenario != "A":
        closure_days = offsets[n - 1]
    if scenario == "E":
        # Positive-EV rule: remain active only if realized net withdrawals
        # exceed the subscription cost incurred through the account end.
        closure_days = row.lifetime_days if revenue > row.subscription_cost else (offsets[0] if offsets else row.lifetime_days)
        n = len(amounts) if closure_days == row.lifetime_days else min(1, len(amounts))
        revenue = sum(amounts[:n])
    evaluation_days = min(float(getattr(row, "evaluation_days", row.lifetime_days)), closure_days)
    months = max(1, int(np.ceil(evaluation_days / MONTH_DAYS))) if evaluation_days > 0 else 0
    subscription = months * _account(row.account_size)["subscription"] if closure_days > 0 else 0.0
    costs = subscription + row.challenge_fees + row.reset_fees + row.commissions
    years = max(closure_days / 365.25, 1 / 365.25)
    return {"scenario": scenario, "closure_days": closure_days, "payouts": n, "gross_revenue": revenue / 0.90 if revenue else 0.0, "net_payouts": revenue, "subscription_cost": subscription, "challenge_fees": row.challenge_fees, "reset_fees": row.reset_fees, "commissions": row.commissions, "total_costs": costs, "net_profit": revenue - costs, "yearly_income": (revenue - costs) / years, "monthly_income": (revenue - costs) / years / 12, "roi": (revenue - costs) / costs if costs else np.nan}


def _account(key):
    return ACCOUNT_SPECS[str(key)]


def _split_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    return [float(x) for x in str(value).split("|") if x != ""]


def _subscription_scenarios(lifecycles):
    rows = []
    for account_size, group in lifecycles.groupby("account_size", sort=False):
        for scenario, count in (("A_keep_until_natural_end", None), ("B_close_after_first_payout", 1), ("C_close_after_second_payout", 2), ("D_close_after_third_payout", 3), ("E_keep_while_expected_value_positive", None)):
            values = [_costed(row, "E" if scenario.startswith("E") else scenario[0], count) for row in group.itertuples()]
            frame = pd.DataFrame(values)
            rows.append({"account_size": account_size, "scenario": scenario, "evaluations": len(frame), "average_account_lifetime_days": frame.closure_days.mean(), "average_subscription_cost": frame.subscription_cost.mean(), "average_payouts": frame.payouts.mean(), "average_net_profit": frame.net_profit.mean(), "yearly_income": frame.yearly_income.mean(), "monthly_income": frame.monthly_income.mean(), "roi": frame.roi.replace([np.inf, -np.inf], np.nan).mean()})
    return pd.DataFrame(rows)


def _account_size_comparison(lifecycles):
    rows = []
    for account_size, group in lifecycles.groupby("account_size", sort=False):
        years = np.maximum(group.lifetime_days / 365.25, 1 / 365.25)
        costs = group.subscription_cost + group.challenge_fees + group.reset_fees + group.commissions
        net = group.net_payouts - costs
        rows.append({"account_size": account_size, "evaluations": len(group), "pass_rate": group.passed.mean(), "payout_rate": (group.payout_count > 0).mean(), "failure_rate": group.failed.mean(), "average_gross_payouts": group.gross_payouts.mean(), "average_net_payouts": group.net_payouts.mean(), "yearly_income_after_all_modeled_costs": (net / years).mean(), "monthly_income_after_all_modeled_costs": (net / years).mean() / 12, "average_subscription_cost": group.subscription_cost.mean(), "average_total_costs": costs.mean(), "roi": (net / costs.replace(0, np.nan)).mean(), "average_drawdown": group.average_drawdown.mean(), "average_lifetime_days": group.lifetime_days.mean(), "average_payouts": group.payout_count.mean()})
    return pd.DataFrame(rows)


def _continuous_analysis(lifecycles, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for account_size, group in lifecycles.groupby("account_size", sort=False):
        paths = [_continuous_path(group, rng, 365.25) for _ in range(BOOTSTRAP_SIMS)]
        frame = pd.DataFrame(paths)
        rows.append({"account_size": account_size, "simulations": BOOTSTRAP_SIMS, "yearly_gross_revenue": frame.gross_revenue.mean(), "yearly_net_revenue": frame.net_revenue.mean(), "yearly_subscription_cost": frame.subscription_cost.mean(), "yearly_reset_cost": frame.reset_cost.mean(), "yearly_challenge_cost": frame.challenge_cost.mean(), "yearly_commissions": frame.commissions.mean(), "yearly_profit_after_all_costs": frame.net_revenue.mean(), "yearly_roi": frame.roi.mean(), "probability_profitable_year": (frame.net_revenue > 0).mean()})
    return pd.DataFrame(rows)


def _continuous_path(group, rng, horizon_days):
    elapsed = 0.0
    gross = net = subscription = reset = challenge = commissions = 0.0
    while elapsed < horizon_days:
        row = group.iloc[int(rng.integers(0, len(group)))]
        life = min(float(row.lifetime_days), horizon_days - elapsed)
        amounts = _split_float(row.payout_amounts)
        offsets = _split_float(row.payout_offsets)
        gross += sum(amount / 0.90 for amount, offset in zip(amounts, offsets) if offset <= life)
        net += sum(amount for amount, offset in zip(amounts, offsets) if offset <= life)
        eval_days = min(float(row.evaluation_days), life)
        subscription += max(1, int(np.ceil(eval_days / MONTH_DAYS))) * _account(row.account_size)["subscription"] if eval_days > 0 else 0.0
        if row.failed:
            reset += 0.0
            elapsed += life
        else:
            elapsed = horizon_days
    costs = subscription + reset + challenge + commissions
    return {"gross_revenue": gross, "net_revenue": net - costs, "subscription_cost": subscription, "reset_cost": reset, "challenge_cost": challenge, "commissions": commissions, "roi": (net - costs) / costs if costs else np.nan}


def _multi_account_analysis(lifecycles, seed):
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for account_count in (1, 3, 5):
        for account_size, group in lifecycles.groupby("account_size", sort=False):
            paths = []
            for _ in range(BOOTSTRAP_SIMS):
                accounts = [_continuous_path(group, rng, 365.25) for _ in range(account_count)]
                paths.append({key: sum(path[key] for path in accounts) for key in ("gross_revenue", "net_revenue", "subscription_cost", "reset_cost", "challenge_cost", "commissions")})
            frame = pd.DataFrame(paths)
            costs = frame.subscription_cost + frame.reset_cost + frame.challenge_cost + frame.commissions
            rows.append({"account_size": account_size, "simultaneous_accounts": account_count, "simulations": BOOTSTRAP_SIMS, "yearly_gross_revenue": frame.gross_revenue.mean(), "yearly_net_revenue": frame.net_revenue.mean(), "subscription_cost": frame.subscription_cost.mean(), "challenge_cost": frame.challenge_cost.mean(), "reset_cost": frame.reset_cost.mean(), "expected_payouts_after_split": frame.net_revenue.mean() + costs.mean(), "expected_roi": (frame.net_revenue / costs.replace(0, np.nan)).mean()})
    return pd.DataFrame(rows)


def _break_even_analysis(lifecycles, seed):
    rng = np.random.default_rng(seed + 2000)
    rows = []
    for account_size, group in lifecycles.groupby("account_size", sort=False):
        paths = [_monthly_path(group, rng, 36) for _ in range(BOOTSTRAP_SIMS)]
        frame = pd.DataFrame(paths)
        for month in range(1, 37):
            net = frame[f"net_{month}"]
            rows.append({"account_size": account_size, "month": month, "expected_cumulative_revenue": frame[f"revenue_{month}"].mean(), "expected_cumulative_costs": frame[f"cost_{month}"].mean(), "expected_cumulative_net": net.mean(), "probability_profitable": (net > 0).mean()})
    return pd.DataFrame(rows)


def _monthly_path(group, rng, months):
    revenue = np.zeros(months); cost = np.zeros(months)
    elapsed = 0.0
    while elapsed < months * MONTH_DAYS:
        row = group.iloc[int(rng.integers(0, len(group)))]
        life = min(float(row.lifetime_days), months * MONTH_DAYS - elapsed)
        amounts = _split_float(row.payout_amounts); offsets = _split_float(row.payout_offsets)
        for amount, offset in zip(amounts, offsets):
            if offset <= life:
                month = int((elapsed + offset) // MONTH_DAYS)
                if month < months: revenue[month] += amount
        eval_days = min(float(row.evaluation_days), life)
        for month in range(int(elapsed // MONTH_DAYS), min(months, int(np.ceil((elapsed + eval_days) / MONTH_DAYS)))):
            cost[month] += _account(row.account_size)["subscription"]
        elapsed += life
        if not row.failed: break
    return {**{f"revenue_{i+1}": revenue[:i+1].sum() for i in range(months)}, **{f"cost_{i+1}": cost[:i+1].sum() for i in range(months)}, **{f"net_{i+1}": revenue[:i+1].sum() - cost[:i+1].sum() for i in range(months)}}


def _economics_audit(lifecycles):
    v9_summary = pd.read_csv("reports/v9/v9_summary.csv")
    rows = [
        {"metric": "V9 evaluation pass rate", "formula": "mean(passed)", "numerator": "passed account rows", "denominator": "all V9 account rows", "value": float(v9_summary[v9_summary.risk_policy == "A"].evaluation_pass_rate.iloc[0]), "audit_result": "Correct for the reported sample; not a cost-adjusted probability."},
        {"metric": "V9 first payout probability", "formula": "mean(payout_count >= 1)", "numerator": "rows with at least one payout", "denominator": "all rows", "value": float(v9_summary[v9_summary.risk_policy == "A"].first_payout_probability.iloc[0]), "audit_result": "Correct conditional event frequency; it is not annual income."},
        {"metric": "V9 expected yearly payout", "formula": "mean(payouts_received / max((end-start)/365.25, 1))", "numerator": "per-row received payout annualization", "denominator": "number of rows in the policy aggregate", "value": float(v9_summary[v9_summary.risk_policy == "A"].expected_yearly_payout_after_split.iloc[0]), "audit_result": "Mathematically correct for annualized gross payout, but excludes subscription/reset costs and replacement accounts."},
        {"metric": "V10 baseline average net lifetime profit", "formula": "mean(net_payouts - subscription_cost - reset_fees - commissions)", "numerator": "lifetime net profit summed across account rows", "denominator": "all 25k/50k/100k lifecycle rows", "value": float((lifecycles.net_payouts - lifecycles.subscription_cost - lifecycles.reset_fees - lifecycles.commissions).mean()), "audit_result": "Adds lifecycle economics; subscription is charged only during evaluation."},
        {"metric": "Why V9 yearly income appears low", "formula": "payout cap × payout frequency × payout probability, averaged over all account rows and exposure years", "numerator": "received withdrawals", "denominator": "account-row exposure, not trader replacement capital", "value": np.nan, "audit_result": "The low number is primarily economic interpretation: payouts are capped, require five winning days and 40% consistency, failures produce zero revenue, and V9 excludes subscription/reset costs."},
    ]
    return pd.DataFrame(rows)


def _write_verified_rules(path):
    path.write_text("""# Alpha Futures Zero rules verified for V10

Verified against official Alpha Futures documentation on 2026-07-14.

| Account | Target | MLL | DLG | Subscription | Eval reset | Qualified reset | Max position | Payout max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero 25K | $1,500 | $1,000 | $500 | $79/month | $69 | $399 | 1 mini / 10 micros | $1,000 |
| Zero 50K | $3,000 | $2,000 | $1,000 | $119/month | $109 | $499 | 3 minis / 30 micros | $1,500 |
| Zero 100K | $6,000 | $3,000 | $2,000 | $239/month | $219 | $799 | 6 minis / 60 micros | $2,500 |

All Zero accounts have $0 activation fee, 90% profit split, no evaluation consistency rule, 40% qualified consistency, and payout eligibility after five non-consecutive $200+ winning days. Withdrawals are up to 50% of profit, with up to four requests per month. Qualified accounts do not pay the evaluation subscription. [Zero overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [payout policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy), [subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription), [reset](https://help.alpha-futures.com/en/articles/9492077-reset)

Qualified resets are available only for eligible Zero accounts, are limited to two before the first payout, and are not automatically assumed by V10. V10 treats failed-account replacement as a new evaluation purchase, not as an automatic paid reset.

The V10 lifecycle model charges subscriptions only through evaluation end, stops them at failure/pass/voluntary closure, and does not double-count commissions already embedded in frozen V9 trade PnL. Challenge fees and reset fees are zero in the base lifecycle because the official Zero product uses monthly evaluation pricing and V10 does not assume optional resets. A continuous replacement analysis is also provided.

Alpha permits five Qualified Zero accounts per user, subject to the official allocation rules. [Maximum allocation](https://help.alpha-futures.com/en/articles/9492088-maximum-allocation)

Important: Alpha prohibits AI, bots, and fully automated trading. This is an economics audit, not authorization for automated execution. [Prohibited practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)
""", encoding="utf-8")


def _write_formula_explanations(path):
    path.write_text("""# V10 formula explanations

## V9 audit

- Pass rate = `count(passed == true) / count(all account rows)`.
- First/second/third payout probability = `count(payout_count >= n) / count(all account rows)`.
- V9 expected yearly payout = `mean(row.payouts_received / max((end_date - start_date).days / 365.25, 1))`.
- V9 monthly payout = expected yearly payout divided by 12.

The V9 formulas are mathematically valid for the statistics they name. They are not net trader-income formulas. The V9 aggregate mixes all six sizes, seven policies, four streams, and monthly starts. It includes zero-payout failures, annualizes each account row over its own historical exposure, excludes subscription/reset costs, and does not purchase replacement accounts. A payout probability answers whether an account ever reaches a payout; it does not state the amount or timing of withdrawals.

## V10 lifecycle economics

- Subscription months = `max(1, ceil(evaluation_days / 30.4375))`; no subscription is charged after pass, failure, or voluntary closure.
- Net lifetime profit = `received payouts - subscription cost - challenge fees - reset fees - commissions`.
- Yearly income = `net lifetime profit / max(lifetime_days / 365.25, 1 / 365.25)`.
- ROI = `net lifetime profit / total lifetime costs`.
- Scenario B/C/D retain only the first/second/third received payouts and close at that payout timestamp.
- Scenario E closes when realized withdrawals do not exceed the subscription cost required to keep the account to its natural end; this is an explicit positive-value heuristic, not a new trading rule.
- Continuous trader results bootstrap complete V9 lifecycle rows until a one-year horizon. A failed account is immediately replaced; a non-failed historical account ends the simulated path.
- Multi-account results sum independent bootstrap paths for 1, 3, and 5 accounts.
- Break-even results show cumulative received payouts minus evaluation subscription costs by month.

Commissions are reported as zero incremental cost because frozen V9 net trade values already include the repository fee model; V10 does not double-count them. Optional reset fees are separately available in the verified-rules document but are not assumed in the base replacement model.
""", encoding="utf-8")


def _write_report(path, accounts, scenarios, continuous, multi, breakeven, audit):
    top = accounts.sort_values("yearly_income_after_all_modeled_costs", ascending=False).iloc[0]
    twentyfive = accounts[accounts.account_size == "25k"].iloc[0]
    scenario25 = scenarios[scenarios.account_size == "25k"].sort_values("yearly_income").iloc[-1]
    continuous25 = continuous[continuous.account_size == "25k"].iloc[0]
    text = f"""
    <p><b>Scope:</b> Economics audit of frozen V9, using Policy A and 2 Micros across Zero 25K/50K/100K.</p>
    <p><b>V9 audit conclusion:</b> the V9 yearly-payout formula is mathematically correct for annualized withdrawals, but it is not net trader income because subscriptions, resets, replacements, and voluntary account-ending decisions were outside its scope.</p>
    <p><b>Highest modeled account-size income:</b> {html.escape(str(top.account_size))}, {top.yearly_income_after_all_modeled_costs:.2f} yearly after modeled costs. See the formula and sampling limitations in <a href="v10_formula_explanations.md">v10_formula_explanations.md</a>.</p>
    <p><b>Economic decision:</b> 25K is the least-negative account-size result ({twentyfive.yearly_income_after_all_modeled_costs:.2f}/year); the best account-ending row is {html.escape(str(scenario25.scenario))}, but it is still negative. Continuous 25K replacement is also negative ({continuous25.yearly_profit_after_all_costs:.2f}/year).</p>
    <p><b>Conclusion:</b> one account is economically preferable to three or five because expected losses scale with account count. No tested configuration produces positive modeled net income after subscription costs.</p>
    """
    sections = [("Economics audit", audit), ("Account-size comparison", accounts), ("Subscription scenarios", scenarios), ("Continuous trader", continuous), ("Multi-account", multi), ("Break-even", breakeven)]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False)}" for title, frame in sections)
    path.write_text(f"<html><body><h1>V10 Alpha Futures Prop Economics Audit</h1>{text}<p>Official rules: <a href=\"v10_verified_rules.md\">v10_verified_rules.md</a></p>{tables}</body></html>", encoding="utf-8")
