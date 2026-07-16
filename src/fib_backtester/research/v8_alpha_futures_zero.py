from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache


ROOT = Path("reports/v8")
ACCOUNT_SIZE = 25_000.0
PROFIT_TARGET = 1_500.0
INITIAL_MLL = 1_000.0
DAILY_LOSS_GUARD = 500.0
PAYOUT_SPLIT = 0.90
PAYOUT_MAX = 1_000.0
PAYOUT_MIN = 200.0
CONTRACT_SIZES = (2, 3, 5, 7, 10)
SIZE_CASES = (
    ("2 Micros", "micros", 2),
    ("3 Micros", "micros", 3),
    ("5 Micros", "micros", 5),
    ("7 Micros", "micros", 7),
    ("10 Micros", "micros", 10),
    ("1 Mini", "mini", 1),
)
ASSETS = ("ETH", "SOL")
TIMEFRAMES = ("4h", "1d")
FROZEN_ENTRY = 0.900
FROZEN_INITIAL_STOP = 1.020
FROZEN_POST_TP1_STOP = 0.820
FROZEN_TP_FRACTIONS = (0.30, 0.25, 0.20, 0.15, 0.10)


@dataclass(frozen=True)
class ContractSpec:
    asset: str
    contract_type: str
    contracts: int
    multiplier: float
    tick_size: float
    point_value: float
    tick_value: float
    max_position: int


SPECS = {
    ("ETH", "micros"): ContractSpec("ETH", "micro", 1, 0.10, 0.50, 0.10, 0.05, 10),
    ("ETH", "mini"): ContractSpec("ETH", "mini", 1, 50.0, 0.25, 50.0, 12.50, 1),
    ("SOL", "micros"): ContractSpec("SOL", "micro", 1, 25.0, 0.05, 25.0, 1.25, 10),
    ("SOL", "mini"): ContractSpec("SOL", "mini", 1, 500.0, 0.05, 500.0, 25.0, 1),
}


def run_v8_alpha_futures_zero(config: RunConfig, root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_account_rules(root / "v8_account_rules.md")
    frozen_params = _load_frozen_parameters()
    all_runs, trade_distributions = [], {}
    skipped = []

    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            try:
                bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                distance, minimum_move = frozen_params[(asset, timeframe)]
                trades = _generate_frozen_trades(config, asset, timeframe, bars, distance, minimum_move)
            except Exception as exc:
                skipped.append({"asset": asset, "timeframe": timeframe, "reason": str(exc)})
                continue
            starts = _feasible_starts(bars.index)
            for size_label, size_type, size_count in SIZE_CASES:
                spec = _spec(asset, size_count, size_type)
                prepared = [_prepare_trade(trade, spec, config.asset_configs[asset].fee_rate) for trade in trades.to_dict("records")]
                trade_distributions.setdefault(size_label, []).extend([row["net_pnl"] for row in prepared])
                for start in starts:
                    result = _simulate_account(prepared, spec, start, bars.index[-1], asset, timeframe, distance, minimum_move)
                    all_runs.append(result)

    runs = pd.DataFrame(all_runs)
    position_sizing = _position_sizing_summary(runs)
    pass_stats = _pass_statistics(runs)
    payout_stats = _payout_statistics(runs)
    failure = _failure_analysis(runs)
    monte_carlo = _monte_carlo(trade_distributions, config.seed)
    summary = _summary(runs, position_sizing, skipped)

    position_sizing.to_csv(root / "v8_position_sizing.csv", index=False)
    pass_stats.to_csv(root / "v8_pass_statistics.csv", index=False)
    payout_stats.to_csv(root / "v8_payout_statistics.csv", index=False)
    failure.to_csv(root / "v8_failure_analysis.csv", index=False)
    monte_carlo.to_csv(root / "v8_monte_carlo.csv", index=False)
    summary.to_csv(root / "v8_summary.csv", index=False)
    _write_report(root / "v8_final_report.html", summary, position_sizing, pass_stats, payout_stats, failure, monte_carlo, skipped)
    return {"runs": len(runs), "skipped": skipped, "monte_carlo_simulations_per_size": 10_000, "root": str(root)}


def _load_frozen_parameters():
    path = Path("reports/v7/v7_holdout.csv")
    if not path.exists():
        raise FileNotFoundError("V7 holdout output is required; V8 does not select new strategy parameters")
    frame = pd.read_csv(path)
    frame = frame[frame.asset.isin(ASSETS) & frame.timeframe.isin(TIMEFRAMES)]
    result = {}
    for row in frame.to_dict("records"):
        result[(row["asset"], row["timeframe"])] = (int(row["selected_min_distance"]), float(row["selected_min_move"]))
    missing = [(asset, timeframe) for asset in ASSETS for timeframe in TIMEFRAMES if (asset, timeframe) not in result]
    if missing:
        raise ValueError(f"V7 frozen parameters missing for {missing}")
    return result


def _generate_frozen_trades(config, asset, timeframe, bars, distance, minimum_move):
    run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
    engine = StrategyV7FrozenValidationEngine(run, minimum_move)
    trades, _ = engine.run({asset: bars})
    return trades


def _spec(asset, size, contract_type=None):
    contract_type = contract_type or ("mini" if size == 1 else "micros")
    return replace(SPECS[(asset, contract_type)], contracts=size)


def _feasible_starts(index):
    start = pd.Timestamp(index[0]) + pd.Timedelta(days=180)
    end = pd.Timestamp(index[-1]) - pd.Timedelta(days=180)
    eligible = index[(index >= start) & (index <= end)]
    if len(eligible) == 0:
        return []
    # The request defines starts by historical calendar date.  For intraday
    # data, use the first available bar on each date rather than a monthly
    # sample or every intraday candle.
    dates = pd.DatetimeIndex(eligible.normalize()).unique()
    return [next(timestamp for timestamp in eligible if timestamp.normalize() == date) for date in dates]


def _prepare_trade(trade, spec, fee_rate):
    entry = _tick(float(trade["entry_price"]), spec.tick_size)
    stop = _tick(float(trade["initial_stop"]), spec.tick_size)
    allocation = _allocate_contracts(spec.contracts, FROZEN_TP_FRACTIONS)
    remaining = spec.contracts
    legs = []
    entry_fee = abs(entry * spec.multiplier * spec.contracts) * fee_rate
    for event in _events(trade.get("exit_events")):
        reason = event.get("reason", "end_of_test")
        if reason.startswith("tp") and reason[2:].isdigit():
            index = int(reason[2:]) - 1
            quantity = min(allocation[index], remaining)
        else:
            quantity = remaining
        if quantity <= 0:
            continue
        price = _tick(float(event.get("fill_price", event.get("raw_price"))), spec.tick_size)
        direction = 1 if trade["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * quantity
        fee = abs(price * spec.multiplier * quantity) * fee_rate
        legs.append({"timestamp": str(event.get("timestamp", trade["exit_timestamp"])), "reason": reason, "price": price, "quantity": quantity, "gross": gross, "fee": fee, "net": gross - fee})
        remaining -= quantity
        if remaining <= 0:
            break
    if remaining > 0:
        timestamp = str(trade.get("exit_timestamp"))
        price = _tick(float(trade.get("average_exit_price", entry)), spec.tick_size)
        direction = 1 if trade["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * remaining
        fee = abs(price * spec.multiplier * remaining) * fee_rate
        legs.append({"timestamp": timestamp, "reason": trade.get("exit_reason", "end_of_test"), "price": price, "quantity": remaining, "gross": gross, "fee": fee, "net": gross - fee})
    risk = abs(entry - stop) * spec.multiplier * spec.contracts
    return {"entry_timestamp": str(trade["fill_timestamp"]), "side": trade["side"], "entry": entry, "stop": stop, "risk": risk, "entry_fee": entry_fee, "net_pnl": sum(leg["net"] for leg in legs) - entry_fee, "legs": legs}


def _allocate_contracts(total, fractions):
    raw = np.asarray(fractions) * total
    allocation = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - allocation))[: total - int(allocation.sum())]:
        allocation[index] += 1
    return allocation.tolist()


def _simulate_account(trades, spec, start, end, asset, timeframe, distance, minimum_move):
    balance = ACCOUNT_SIZE
    mll = ACCOUNT_SIZE - INITIAL_MLL
    high_eod_balance = ACCOUNT_SIZE
    qualified = False
    failed = False
    failure_reason = "end_of_history"
    pass_time = None
    payout_times = []
    payouts_gross = 0.0
    daily_profit = {}
    cycle_profit = 0.0
    winning_days = set()
    cycle_day_profits = {}
    consistency_events = 0
    daily_guard_events = 0
    locked_session = None
    equity_curve = [balance]
    internal_events = []
    last_session = None

    for trade in sorted(trades, key=lambda row: row["entry_timestamp"]):
        entry_time = pd.Timestamp(trade["entry_timestamp"])
        if entry_time < pd.Timestamp(start) or entry_time > pd.Timestamp(end) or failed:
            continue
        if locked_session is not None and _session(entry_time) == locked_session:
            continue
        for leg in trade["legs"]:
            timestamp = pd.Timestamp(leg["timestamp"])
            if timestamp < pd.Timestamp(start) or timestamp > pd.Timestamp(end):
                continue
            session = _session(timestamp)
            if last_session is not None and session != last_session:
                cycle_profit, winning_days, cycle_day_profits, consistency_events, payout_times, payouts_gross, balance, mll = _finish_day(
                    last_session, daily_profit, cycle_day_profits, qualified, cycle_profit, winning_days, consistency_events, payout_times, payouts_gross, balance, mll, internal_events
                )
                daily_profit.pop(last_session, None)
            last_session = session
            if locked_session == session:
                continue
            value = float(leg["net"]) - (float(trade["entry_fee"]) if leg is trade["legs"][0] else 0.0)
            balance += value
            daily_profit[session] = daily_profit.get(session, 0.0) + value
            if qualified:
                cycle_profit += value
            equity_curve.append(balance)
            if daily_profit[session] <= -DAILY_LOSS_GUARD and locked_session != session:
                daily_guard_events += 1
                locked_session = session
                internal_events.append((timestamp, "Daily Loss Violation"))
            if balance <= mll:
                failed = True
                failure_reason = "Maximum Loss Violation"
                internal_events.append((timestamp, failure_reason))
                break
            if not qualified and balance >= ACCOUNT_SIZE + PROFIT_TARGET:
                qualified = True
                pass_time = timestamp
                cycle_profit = 0.0
                winning_days = set()
                cycle_day_profits = {}
                daily_profit = {session: daily_profit.get(session, 0.0)}
                internal_events.append((timestamp, "Evaluation Passed"))
        if failed:
            break

    if not failed and last_session is not None:
        cycle_profit, winning_days, cycle_day_profits, consistency_events, payout_times, payouts_gross, balance, mll = _finish_day(
            last_session, daily_profit, cycle_day_profits, qualified, cycle_profit, winning_days, consistency_events, payout_times, payouts_gross, balance, mll, internal_events
        )
        high_eod_balance = max(high_eod_balance, balance)
    drawdowns = pd.Series(equity_curve) / pd.Series(equity_curve).cummax() - 1
    qualified_lifetime_days = ((pd.Timestamp(end) - pass_time).total_seconds() / 86400) if pass_time is not None else 0.0
    first = payout_times[0] if len(payout_times) >= 1 else None
    second = payout_times[1] if len(payout_times) >= 2 else None
    third = payout_times[2] if len(payout_times) >= 3 else None
    return {
        "asset": asset, "timeframe": timeframe, "contract_size": f"{spec.contracts} Micros" if spec.contract_type == "micro" else "1 Mini",
        "contract_type": spec.contract_type, "contracts": spec.contracts, "contract_multiplier": spec.multiplier, "tick_size": spec.tick_size, "tick_value": spec.tick_value,
        "start_date": str(start), "end_date": str(end), "selected_min_distance": distance, "selected_min_move": minimum_move,
        "passed": bool(pass_time is not None), "failed": bool(failed), "failure_reason": failure_reason if failed else "none",
        "days_to_pass": (pass_time - pd.Timestamp(start)).total_seconds() / 86400 if pass_time is not None else np.nan,
        "days_to_first_payout": (first - pd.Timestamp(start)).total_seconds() / 86400 if first is not None else np.nan,
        "days_to_second_payout": (second - pd.Timestamp(start)).total_seconds() / 86400 if second is not None else np.nan,
        "days_to_third_payout": (third - pd.Timestamp(start)).total_seconds() / 86400 if third is not None else np.nan,
        "payout_count": len(payout_times), "payouts_gross": payouts_gross, "payouts_received": payouts_gross * PAYOUT_SPLIT,
        "final_balance": balance, "maximum_loss_limit": mll, "average_drawdown": float(drawdowns.mean()), "maximum_drawdown": float(drawdowns.min()),
        "average_daily_drawdown": float(np.mean([max(0.0, -value) for value in daily_profit.values()])) if daily_profit else 0.0,
        "daily_loss_violations": daily_guard_events, "consistency_rule_events": consistency_events, "rule_violations": daily_guard_events + consistency_events,
        "qualified_lifetime_days": qualified_lifetime_days, "winning_days": len(winning_days), "equity_observations": len(equity_curve),
    }


def _finish_day(session, daily_profit, cycle_day_profits, qualified, cycle_profit, winning_days, consistency_events, payout_times, payouts_gross, balance, mll, events):
    profit = daily_profit.get(session, 0.0)
    if qualified and profit > 0:
        cycle_day_profits[session] = profit
    if qualified and profit >= 200:
        winning_days.add(session)
    if qualified and len(winning_days) >= 5 and cycle_profit > 0:
        largest_day = max(cycle_day_profits.values(), default=0.0)
        if largest_day >= 0.40 * cycle_profit:
            consistency_events += 1
            events.append((pd.Timestamp(session), "Consistency Rule Failure"))
        else:
            request = min(0.50 * cycle_profit, PAYOUT_MAX)
            if request >= PAYOUT_MIN and balance - request > mll:
                balance -= request
                payouts_gross += request
                payout_time = pd.Timestamp(session, tz="America/New_York")
                payout_times.append(payout_time)
                events.append((payout_time, f"Payout {len(payout_times)}"))
                cycle_profit = 0.0
                winning_days = set()
                cycle_day_profits = {}
    high_eod = max(ACCOUNT_SIZE, balance)
    mll = min(ACCOUNT_SIZE, max(mll, high_eod - INITIAL_MLL))
    return cycle_profit, winning_days, cycle_day_profits, consistency_events, payout_times, payouts_gross, balance, mll


def _position_sizing_summary(runs):
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for (asset, timeframe, size), group in runs.groupby(["asset", "timeframe", "contract_size"], sort=True):
        years = max((pd.to_datetime(group.end_date).max() - pd.to_datetime(group.start_date).min()).days / 365.25, 1)
        pass_rate = group.passed.mean()
        payout_rate = (group.payout_count > 0).mean()
        failure_rate = group.failed.mean()
        annual_payout = group.payouts_received.sum() / max(group.shape[0], 1) / years
        score = 0.40 * pass_rate + 0.35 * payout_rate - 0.25 * failure_rate - 0.10 * abs(group.average_drawdown.mean()) - 0.05 * group.rule_violations.mean() / 5
        rows.append({"asset": asset, "timeframe": timeframe, "contract_size": size, "evaluations": len(group), "pass_rate": pass_rate, "payout_rate": payout_rate, "failure_rate": failure_rate, "expected_annual_payout": annual_payout, "average_monthly_payout": annual_payout / 12, "average_drawdown": group.average_drawdown.mean(), "average_lifetime_days": group.qualified_lifetime_days.mean(), "average_rule_violations": group.rule_violations.mean(), "robustness_score": score})
    result = pd.DataFrame(rows)
    result["robustness_rank"] = result.robustness_score.rank(method="first", ascending=False).astype(int)
    return result.sort_values("robustness_rank")


def _pass_statistics(runs):
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for size, group in runs.groupby("contract_size", sort=False):
        rows.append({"contract_size": size, "evaluations": len(group), "pass_rate": group.passed.mean(), "probability_of_passing": group.passed.mean(), "average_days_to_pass": group.days_to_pass.mean(), "failure_rate": group.failed.mean(), "average_drawdown": group.average_drawdown.mean(), "average_lifetime_days": group.qualified_lifetime_days.mean()})
    return pd.DataFrame(rows)


def _payout_statistics(runs):
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for size, group in runs.groupby("contract_size", sort=False):
        years = max((pd.to_datetime(group.end_date).max() - pd.to_datetime(group.start_date).min()).days / 365.25, 1)
        rows.append({"contract_size": size, "evaluations": len(group), "probability_first_payout": (group.payout_count >= 1).mean(), "probability_second_payout": (group.payout_count >= 2).mean(), "probability_third_payout": (group.payout_count >= 3).mean(), "probability_losing_account_before_first_payout": ((group.failed) & (group.payout_count == 0)).mean(), "average_number_of_payouts": group.payout_count.mean(), "average_days_to_first_payout": group.days_to_first_payout.mean(), "average_days_to_second_payout": group.days_to_second_payout.mean(), "average_days_to_third_payout": group.days_to_third_payout.mean(), "average_monthly_payout": group.payouts_received.sum() / max(len(group), 1) / years / 12, "expected_annual_payout": group.payouts_received.sum() / max(len(group), 1) / years, "profit_split": PAYOUT_SPLIT})
    return pd.DataFrame(rows)


def _failure_analysis(runs):
    if runs.empty:
        return pd.DataFrame()
    counts = runs[runs.failed].groupby(["contract_size", "failure_reason"]).size().reset_index(name="failures")
    totals = runs.groupby("contract_size").size().rename("evaluations")
    counts["failure_rate_of_evaluations"] = counts.apply(lambda row: row.failures / totals[row.contract_size], axis=1)
    counts["most_common_failure_reason"] = counts.groupby("contract_size").failures.transform("max") == counts.failures
    return counts.sort_values(["contract_size", "failures"], ascending=[True, False])


def _monte_carlo(distributions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for size_label, _, _ in SIZE_CASES:
        values = np.asarray(distributions.get(size_label, []), dtype=float)
        if len(values) == 0:
            continue
        mean_count = max(1, int(round(len(values) / 4)))
        outcomes = []
        incomes = []
        for _ in range(10_000):
            count = max(1, int(rng.poisson(mean_count)))
            sampled = rng.choice(values, size=count, replace=True)
            result = _synthetic_account(sampled)
            outcomes.append(result)
            incomes.append(result["payouts_received"])
        frame = pd.DataFrame(outcomes)
        income = np.asarray(incomes)
        rows.append({"contract_size": size_label, "simulations": 10_000, "seed": seed, "probability_of_passing": frame.passed.mean(), "probability_of_first_payout": (frame.payout_count >= 1).mean(), "probability_of_multiple_payouts": (frame.payout_count >= 2).mean(), "probability_of_account_failure": frame.failed.mean(), "annual_income_mean": income.mean(), "annual_income_p05": np.quantile(income, .05), "annual_income_median": np.quantile(income, .50), "annual_income_p95": np.quantile(income, .95), "annual_income_ci95_low": np.quantile(income, .025), "annual_income_ci95_high": np.quantile(income, .975)})
    return pd.DataFrame(rows)


def _synthetic_account(sampled):
    balance, mll, passed, failed, payouts, received = ACCOUNT_SIZE, ACCOUNT_SIZE - INITIAL_MLL, False, False, 0, 0.0
    day_profit, cycle_profit, winning_days, largest_day = 0.0, 0.0, 0, 0.0
    peak = balance
    for pnl in sampled:
        balance += float(pnl)
        day_profit += float(pnl)
        if balance <= mll:
            failed = True
            break
        if not passed and balance >= ACCOUNT_SIZE + PROFIT_TARGET:
            passed = True
            day_profit = 0.0
            cycle_profit = 0.0
            winning_days = 0
            largest_day = 0.0
        if passed:
            cycle_profit += float(pnl)
            if day_profit >= 200:
                winning_days += 1
            largest_day = max(largest_day, day_profit)
            if winning_days >= 5 and cycle_profit > 0 and largest_day < .40 * cycle_profit:
                request = min(.50 * cycle_profit, PAYOUT_MAX)
                if request >= PAYOUT_MIN and balance - request > mll:
                    balance -= request
                    received += request * PAYOUT_SPLIT
                    payouts += 1
                    cycle_profit = 0.0
                    winning_days = 0
                    largest_day = 0.0
        peak = max(peak, balance)
        day_profit = 0.0
    return {"passed": passed, "failed": failed, "payout_count": payouts, "payouts_received": received, "drawdown": balance / peak - 1}


def _events(value):
    try:
        return json.loads(value) if isinstance(value, str) else (value or [])
    except (TypeError, json.JSONDecodeError):
        return []


def _tick(value, tick):
    return round(value / tick) * tick


def _session(timestamp):
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    return str((local - pd.Timedelta(days=1)).date() if local.hour < 18 else local.date())


def _write_account_rules(path):
    path.write_text(f"""# Alpha Futures Zero 25K rules used by V8

Verified against official Alpha Futures Help Center documentation on 2026-07-14. The simulation uses the current published rules below.

## Account parameters

| Rule | Implemented value | Difference from request assumptions |
|---|---:|---|
| Simulated account size | $25,000 | None |
| Evaluation profit target | $1,500 | None |
| Maximum Loss Limit | $1,000, end-of-day trailing; stops at $25,000 | The request did not specify trailing calculation |
| Daily Loss Guard | $500, soft lock; 2% of starting balance | None |
| Evaluation max position | 1 mini or 10 micros | None |
| Evaluation consistency | None on Zero Evaluation | The request asked to verify; it is not required |
| Qualified consistency | 40% since the last withdrawal request | The request did not specify the qualified value |
| Profit split | Trader receives 90% of the gross withdrawal request | None |
| Qualified payout timing | Up to 4 requests per month after 5 winning days of at least $200 | Not a fixed weekly payout schedule |
| Withdrawal amount | Minimum $200; maximum $1,000; request removes up to 50% of account profit | The request said weekly payouts but not these limits |
| Evaluation subscription | $79/month; ends after passing | Not included as a trading PnL deduction |
| Evaluation reset | $69; monthly rebill also resets a failed account | Not included in payout statistics |
| Qualified reset | Zero Qualified reset available twice before any payout, within 7 days of breach, for eligible post-2026-03-11 purchases | Not automatically simulated |

## Implemented behavior

- Evaluation passes when balance reaches the $1,500 target without an MLL breach.
- MLL is based on the highest end-of-day balance minus $1,000, capped at the $25,000 starting balance. A balance/equity breach terminates the account.
- Daily Loss Guard is treated as a soft breach: open trading is locked until the next 6PM ET trading day. The simulation records the violation; it does not count it as account failure unless MLL is also breached.
- Qualified payouts require five accumulated $200+ winning days and the 40% consistency test. A payout request is modeled at 50% of cycle profit, capped at $1,000, with 90% paid to the trader.
- Consistency failures block a payout cycle; they do not terminate the account.
- News restrictions are not modeled because the historical data has no official high-impact-news calendar. Officially, evaluations have no news restriction; Qualified Zero accounts prohibit execution within two minutes before or after high-impact news.
- Prohibited-practice review is not modeled from OHLC data. Alpha prohibits AI, bots and fully automated trading; the results therefore represent a manual/semi-automated execution assumption only, not authorization to run the repository as a trading bot.
- The simulation uses the repository's conservative OHLC execution and existing fees/slippage. It uses CME price specifications to convert price moves into contract dollars, rounds fills to the official tick, and retains repository fee rates because Alpha's account rules do not publish a universal commission schedule.
- Historical ETH/SOL bars are exchange-price proxies, not CME futures bars. Contract rolls, CME session holidays, margin, commissions, news windows and exact intrabar unrealized PnL are not available in the repository and are limitations.

## Official sources

- [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview)
- [Maximum Loss Limit](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll)
- [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard)
- [Consistency Rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule)
- [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy)
- [Maximum Withdrawal Request](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request)
- [Reset](https://help.alpha-futures.com/en/articles/9492077-reset)
- [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription)
- [Scaling Plan](https://help.alpha-futures.com/en/articles/9492025-scaling-plan)
- [News Trading Policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy)
- [Prohibited Trading Practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)

## CME contract specifications

| Instrument | Multiplier | Tick size | Tick value | Point value |
|---|---:|---:|---:|---:|
| Micro Ether (MET) | 0.10 ETH | $0.50/ETH | $0.05 | $0.10 per $1 |
| Ether (ETH) | 50 ETH | $0.25/ETH | $12.50 | $50 per $1 |
| Micro Solana (MSL) | 25 SOL | $0.05/SOL | $1.25 | $25 per $1 |
| Solana (SOL) | 500 SOL | $0.05/SOL | $25.00 | $500 per $1 |

Sources: [CME Micro Ether specifications](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html), [CME Ether specifications](https://www.cmegroup.com/education/courses/introduction-to-ether/ether-futures-product-overview), [CME Micro SOL specifications](https://www.cmegroup.com/rulebook/CME/IV/400/440/440.pdf), and [CME SOL specifications](https://www.cmegroup.com/articles/2025/the-essential-guide-to-solana-futures.html).

The official Alpha 25K limit permits one mini or ten micros. V8 evaluates 2, 3, 5, 7 and 10 micros, plus one mini for comparison, without exceeding that limit.
""", encoding="utf-8")


def _summary(runs, sizing, skipped):
    if runs.empty:
        return pd.DataFrame([{"scope": "no_results", "skipped": len(skipped)}])
    best = sizing.sort_values("robustness_rank").iloc[0] if not sizing.empty else {}
    run_years = ((pd.to_datetime(runs.end_date) - pd.to_datetime(runs.start_date)).dt.days / 365.25).clip(lower=1)
    annual_payout = float((runs.payouts_received / run_years).mean())
    return pd.DataFrame([
        {"scope": "all_simulated_accounts", "evaluations": len(runs), "evaluation_pass_rate": runs.passed.mean(), "first_payout_probability": (runs.payout_count >= 1).mean(), "second_payout_probability": (runs.payout_count >= 2).mean(), "third_payout_probability": (runs.payout_count >= 3).mean(), "probability_failure_before_first_payout": ((runs.failed) & (runs.payout_count == 0)).mean(), "average_payouts": runs.payout_count.mean(), "expected_annual_payout": annual_payout, "average_monthly_payout": annual_payout / 12, "average_drawdown": runs.average_drawdown.mean(), "average_daily_drawdown": runs.average_daily_drawdown.mean(), "most_common_failure_reason": Counter(runs.loc[runs.failed, "failure_reason"]).most_common(1)[0][0] if runs.failed.any() else "none", "robustest_contract_size": best.get("contract_size", "unavailable"), "skipped_streams": len(skipped)}
    ])


def _write_report(path, summary, sizing, passes, payouts, failures, monte_carlo, skipped):
    if summary.empty:
        classification = "insufficient simulation data"
        answer = "No feasible simulations were available."
    else:
        row = summary.iloc[0]
        classification = "historical account simulation only; not a verified Alpha eligibility result"
        answer = f"Evaluation pass probability {row.evaluation_pass_rate:.2%}; first payout {row.first_payout_probability:.2%}; second payout {row.second_payout_probability:.2%}; third payout {row.third_payout_probability:.2%}. Most robust tested size: {row.robustest_contract_size}."
    text = f"""
    <p><b>Scope:</b> ETH and SOL only, using the frozen V7 strategy and frozen V7 distance/minimum-move selections. No strategy optimization was performed.</p>
    <p><b>Important eligibility limitation:</b> Alpha Futures officially prohibits AI, bots and fully automated trading. This simulation is not evidence that an automated implementation may be used on an Alpha account.</p>
    <p><b>{html.escape(classification)}</b></p>
    <p>{html.escape(answer)}</p>
    <p>Official rule verification is documented in <a href="v8_account_rules.md">v8_account_rules.md</a>. Skipped streams: {html.escape('; '.join(f"{x['asset']} {x['timeframe']}" for x in skipped) or 'none')}.</p>
    """
    sections = [("Summary", summary), ("Position sizing", sizing), ("Pass statistics", passes), ("Payout statistics", payouts), ("Failure analysis", failures), ("Monte Carlo", monte_carlo)]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False)}" for title, frame in sections)
    path.write_text(f"<html><body><h1>V8 Alpha Futures Zero 25K Simulation</h1>{text}{tables}</body></html>", encoding="utf-8")
