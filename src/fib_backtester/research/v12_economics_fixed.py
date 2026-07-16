"""Recent-window, one-account Alpha Futures Zero economics study.

This module does not alter a strategy, signal, execution, portfolio, or data
layer.  It reuses the frozen V12 trade generator and the canonical proxy map,
then models one trader turning over one Alpha account at a time.
"""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.research import v12_binance_proxy_prop_simulation as legacy
from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed
from fib_backtester.research.v12_contract_registry import build_synthetic_context


ROOT = Path("reports/v12_economics_fixed")
RECENT_DAYS = 730
HORIZONS_MONTHS = (12, 24)
POLICIES = ("MAX_30_DAYS", "MAX_60_DAYS", "MAX_90_DAYS", "DYNAMIC_PROGRESS")
OFFICIAL_VERIFICATION_DATE = "2026-07-15"

# The official Zero page currently lists 25K, 50K, and 100K.  The prior fixed
# replay covered 25K/50K; the 100K row is included here as a current-rule
# comparison, with an explicit warning that the frozen sizes stop at 10 micros.
ACCOUNT_SPECS = {
    "25K Zero": fixed.ACCOUNT_SPECS["25K Zero"],
    "50K Zero": fixed.ACCOUNT_SPECS["50K Zero"],
    "100K Zero": replace(
        fixed.ACCOUNT_SPECS["50K Zero"],
        name="100K Zero",
        account_size=100_000.0,
        target=6_000.0,
        mll_amount=3_000.0,
        daily_loss_guard=2_000.0,
        subscription=239.0,
        evaluation_reset=219.0,
        qualified_reset=799.0,
        payout_max=2_500.0,
        max_micros_evaluation=60,
        max_micros_qualified_initial=10,
    ),
}

PORTFOLIOS = {
    "Portfolio A - ETH only": ["ETH"],
    "Portfolio B - ETH + Gold": ["ETH", "Gold"],
    "Portfolio C - BTC + ETH + Gold": ["BTC", "ETH", "Gold"],
    "Portfolio D - All canonical Alpha exposures": sorted(fixed.CANONICAL_PROXIES),
}


def _fee_rate(market: str) -> float:
    return 0.001 if market in {"BTC", "ETH"} else 0.0005


def _billing_months(start: pd.Timestamp, stop: pd.Timestamp) -> int:
    if stop < start:
        return 0
    return len(fixed._rebill_dates(start, stop))


def _event_list(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
    events = []
    for number, trade in enumerate(trades):
        if not (start <= trade["entry_timestamp"] <= end):
            continue
        events.append((trade["entry_timestamp"], 1, number, "entry", None, trade))
        for leg_index, leg in enumerate(trade["legs"]):
            timestamp = pd.Timestamp(leg["timestamp"])
            if start <= timestamp <= end:
                events.append((timestamp, 0, number, "exit", leg_index, trade))
    return sorted(events, key=lambda row: (row[0], row[1], row[2]))


def _result(start, end, stage, account, balance, status, terminal, gross, net, fees, payout_gross=0.0, payout_received=0.0, payout_timestamp="", trades=0, conflicts=0, daily_locks=0, winning_days=0, reason=""):
    return {
        "stage": stage,
        "status": status,
        "start": start,
        "end": end,
        "terminal": terminal,
        "starting_balance": float(account.account_size if stage == "EVALUATION" else balance["initial"]),
        "ending_balance": float(balance["value"]),
        "gross_trading_pnl": float(gross),
        "net_trading_pnl": float(net),
        "fees": float(fees),
        "payout_gross": float(payout_gross),
        "payout_received": float(payout_received),
        "payout_timestamp": payout_timestamp,
        "trades": int(trades),
        "conflicts": int(conflicts),
        "daily_locks": int(daily_locks),
        "winning_days": int(winning_days),
        "reason": reason,
    }


def _run_stage(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, account, stage: str, cancellation_policy: str = "", initial_balance: float | None = None) -> dict:
    initial = float(account.account_size if initial_balance is None else initial_balance)
    balance = {"initial": initial, "value": initial}
    mll = initial - account.mll_amount
    active = {}
    current_session = None
    daily_profit = 0.0
    cycle_profit = 0.0
    cycle_days: dict[str, float] = {}
    winning_days: set[str] = set()
    payout = None
    gross = net = fees = 0.0
    conflicts = daily_locks = processed = 0
    last_timestamp = start
    checkpoints = [(start + pd.Timedelta(days=30), 0.25), (start + pd.Timedelta(days=60), 0.50), (start + pd.Timedelta(days=90), 0.75)] if cancellation_policy == "DYNAMIC_PROGRESS" and stage == "EVALUATION" else []

    def force_close(timestamp, reason):
        nonlocal gross, net, fees, daily_profit, cycle_profit
        for number, item in list(active.items()):
            flatten = fixed._flatten_leg(item["trade"], item["last_price"], timestamp, item["remaining"], reason)
            balance["value"] += flatten["net"]
            gross += flatten["gross"]
            net += flatten["net"]
            fees += flatten["fee"]
            daily_profit += flatten["net"]
            if stage == "QUALIFIED":
                cycle_profit += flatten["net"]
            active.pop(number, None)

    def finish_day(session, timestamp):
        nonlocal daily_profit, cycle_profit, mll, payout
        if stage == "QUALIFIED" and daily_profit > 0:
            cycle_days[session] = daily_profit
        if stage == "QUALIFIED" and daily_profit >= legacy.WINNING_DAY_MINIMUM:
            winning_days.add(session)
        if stage == "QUALIFIED" and len(winning_days) >= legacy.WINNING_DAYS_REQUIRED and cycle_profit > 0:
            consistency = max(cycle_days.values(), default=0.0) / cycle_profit if cycle_profit else 0.0
            request = min(0.50 * cycle_profit, account.payout_max)
            if consistency <= legacy.CONSISTENCY_LIMIT and request >= legacy.WINNING_DAY_MINIMUM and balance["value"] - request > mll:
                balance["value"] -= request
                payout = (request, request * legacy.PAYOUT_SPLIT, timestamp)
                return True
        mll = min(initial, max(mll, balance["value"] - account.mll_amount))
        daily_profit = 0.0
        return False

    def cancellation_at(timestamp):
        if stage != "EVALUATION" or not checkpoints:
            return None
        while checkpoints and checkpoints[0][0] <= timestamp:
            checkpoint, required = checkpoints.pop(0)
            progress = balance["value"] - initial
            if progress < account.target * required:
                force_close(checkpoint, "voluntary_cancellation")
                return checkpoint
        return None

    events = _event_list(trades, start, end)
    for timestamp, kind, number, event_type, leg_index, trade in events:
        cancellation_time = cancellation_at(timestamp)
        if cancellation_time is not None:
            return _result(start, end, stage, account, balance, "VOLUNTARY_CANCEL", cancellation_time, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="cancellation policy threshold")
        if active:
            cutoff = fixed._next_session_close_after(last_timestamp)
            if cutoff < timestamp:
                force_close(cutoff, "session_forced_liquidation")
        last_timestamp = timestamp
        session = fixed._session(timestamp)
        if current_session is not None and session != current_session:
            if finish_day(current_session, timestamp):
                return _result(start, end, stage, account, balance, "FIRST_PAYOUT", timestamp, gross, net, fees, payout[0], payout[1], str(payout[2]), processed, conflicts, daily_locks, len(winning_days), "first successful qualified payout")
            if stage == "EVALUATION" and balance["value"] <= mll:
                return _result(start, end, stage, account, balance, "FAILED", timestamp, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="Maximum Loss Limit")
        current_session = session
        if kind == 1:
            cancellation_time = cancellation_at(timestamp)
            if stage == "EVALUATION" and cancellation_time is not None:
                return _result(start, end, stage, account, balance, "VOLUNTARY_CANCEL", cancellation_time, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="cancellation policy threshold")
            local = pd.Timestamp(timestamp).tz_convert("America/New_York")
            if local.hour == 17 or (local.hour == 16 and local.minute >= 20):
                continue
            max_contracts = account.max_micros_evaluation if stage == "EVALUATION" else account.max_micros_qualified_initial
            current_contracts = sum(item["remaining"] for item in active.values())
            same_market = any(item["trade"]["market"] == trade["market"] for item in active.values())
            if same_market or current_contracts + trade["contracts"] > max_contracts:
                conflicts += 1
                continue
            entry_fee = float(trade["entry_fee"])
            balance["value"] -= entry_fee
            daily_profit -= entry_fee
            net -= entry_fee
            fees += entry_fee
            active[number] = {"trade": trade, "remaining": trade["contracts"], "last_price": trade["entry"]}
            continue
        if number not in active:
            continue
        leg = trade["legs"][leg_index]
        active[number]["last_price"] = leg["price"]
        balance["value"] += leg["net"]
        gross += leg["gross"]
        net += leg["net"]
        fees += leg["fee"]
        daily_profit += leg["net"]
        if stage == "QUALIFIED":
            cycle_profit += leg["net"]
        active[number]["remaining"] -= leg["quantity"]
        processed += 1
        if daily_profit <= -account.daily_loss_guard:
            daily_locks += 1
            for item in list(active.values()):
                if item["remaining"] > 0:
                    force_close(timestamp, "daily_loss_guard")
                    break
        if balance["value"] <= mll:
            force_close(timestamp, "maximum_loss_limit")
            return _result(start, end, stage, account, balance, "FAILED", timestamp, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="Maximum Loss Limit")
        if stage == "EVALUATION" and balance["value"] >= account.account_size + account.target:
            force_close(timestamp, "evaluation_pass_flatten")
            return _result(start, end, stage, account, balance, "PASSED", timestamp, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="profit target reached")
        if leg_index == len(trade["legs"]) - 1:
            active.pop(number, None)

    cancellation_time = cancellation_at(end)
    if cancellation_time is not None:
        return _result(start, end, stage, account, balance, "VOLUNTARY_CANCEL", cancellation_time, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="cancellation policy threshold")
    if active:
        force_close(fixed._next_session_close_after(last_timestamp), "session_forced_liquidation")
    if current_session is not None and finish_day(current_session, end):
        return _result(start, end, stage, account, balance, "FIRST_PAYOUT", end, gross, net, fees, payout[0], payout[1], str(payout[2]), processed, conflicts, daily_locks, len(winning_days), "first successful qualified payout")
    if stage == "EVALUATION":
        return _result(start, end, stage, account, balance, "VOLUNTARY_CANCEL" if cancellation_policy.startswith("MAX_") else "CENSORED_END_OF_DATA", end, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="maximum evaluation age" if cancellation_policy.startswith("MAX_") else "historical data ended")
    if balance["value"] <= mll:
        return _result(start, end, stage, account, balance, "FAILED", end, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="Maximum Loss Limit")
    return _result(start, end, stage, account, balance, "CENSORED_END_OF_DATA", end, gross, net, fees, trades=processed, conflicts=conflicts, daily_locks=daily_locks, winning_days=len(winning_days), reason="historical data ended")


def _prepare_recent_streams() -> tuple[dict, list[dict], pd.Timestamp]:
    streams = {}
    skipped = []
    for market in legacy._load_cached_markets():
        for timeframe in fixed.TIMEFRAMES:
            try:
                bars = legacy._read_cached_bars(market, timeframe)
                data_end = bars.index.max()
                recent_start = data_end - pd.Timedelta(days=RECENT_DAYS)
                recent_bars = bars.loc[bars.index >= recent_start].copy()
                trades, _ = legacy._run_frozen(market, timeframe, recent_bars)
                streams[(market, timeframe)] = {"bars": recent_bars, "trades": trades}
            except Exception as exc:
                skipped.append({"market": market, "timeframe": timeframe, "reason": f"{type(exc).__name__}: {exc}"})
    return streams, skipped, max((stream["bars"].index.max() for stream in streams.values()), default=pd.Timestamp.now(tz="UTC"))


def _path_starts(members: list[str], timeframe: str, streams: dict, horizon: int) -> list[pd.Timestamp]:
    keys = [(market, timeframe) for market in members if (market, timeframe) in streams]
    if len(keys) != len(members):
        return []
    common_start = max(streams[key]["bars"].index.min() for key in keys)
    common_end = min(streams[key]["bars"].index.max() for key in keys)
    lower = max(common_start, common_end - pd.Timedelta(days=RECENT_DAYS))
    latest_start = common_end - pd.DateOffset(months=horizon)
    if latest_start < lower:
        return []
    month_starts = list(pd.date_range(lower.normalize(), latest_start.normalize(), freq="MS", tz="UTC"))
    # A two-year window often has exactly one valid 24-month endpoint whose
    # day is not the first of a month.  Retain that endpoint instead of
    # incorrectly reporting that the horizon is unavailable.
    if not month_starts and latest_start >= lower:
        return [pd.Timestamp(latest_start)]
    return [pd.Timestamp(t) for t in month_starts]


def _merge_trades(members: list[str], timeframe: str, streams: dict, size: int) -> list[dict]:
    result = []
    first_proxy_prices = {}
    for market in members:
        stream = streams.get((market, timeframe))
        if stream is not None and market in fixed.CANONICAL_PROXIES and not stream["trades"].empty:
            first_proxy_prices[market] = float(stream["trades"].sort_values("fill_timestamp").iloc[0]["entry_price"])
    conversion_context = build_synthetic_context(first_proxy_prices)
    for market in members:
        stream = streams.get((market, timeframe))
        if stream is None or market not in fixed.CANONICAL_PROXIES:
            continue
        for raw in stream["trades"].to_dict("records"):
            result.append(fixed._prepare_trade(raw, market, size, conversion_context.get(market)))
    return sorted(result, key=lambda trade: (trade["entry_timestamp"], trade["market"], trade["setup_id"]))


def _simulate_path(trades: list[dict], start: pd.Timestamp, horizon_end: pd.Timestamp, account, policy: str, horizon: int, portfolio: str, timeframe: str, size: int, run_id: str) -> tuple[list[dict], dict]:
    cursor = start
    started = []
    total_payout = total_subscription = total_reset = 0.0
    passes = failures = cancellations = first_payouts = 0
    while cursor < horizon_end and len(started) < 200:
        if policy == "MAX_30_DAYS":
            eval_end = min(horizon_end, cursor + pd.Timedelta(days=30))
        elif policy == "MAX_60_DAYS":
            eval_end = min(horizon_end, cursor + pd.Timedelta(days=60))
        elif policy == "MAX_90_DAYS":
            eval_end = min(horizon_end, cursor + pd.Timedelta(days=90))
        else:
            eval_end = horizon_end
        evaluation = _run_stage(trades, cursor, eval_end, account, "EVALUATION", policy)
        subscription = _billing_months(cursor, evaluation["terminal"]) * account.subscription
        total_subscription += subscription
        account_row = {"run_id": run_id, "portfolio": portfolio, "timeframe": timeframe, "account": account.name, "position_size": size, "horizon_months": horizon, "cancellation_policy": policy, "evaluation_start": cursor, "evaluation_status": evaluation["status"], "evaluation_days": (evaluation["terminal"] - cursor).total_seconds() / 86400, "evaluation_net_pnl": evaluation["net_trading_pnl"], "evaluation_gross_pnl": evaluation["gross_trading_pnl"], "evaluation_fees": evaluation["fees"], "subscription_cost": subscription, "reset_cost": 0.0, "payout_gross": 0.0, "payout_received": 0.0, "net_cashflow": -subscription, "days_to_pass": np.nan, "days_to_first_payout": np.nan, "first_payout": False, "cancellation": evaluation["status"] == "VOLUNTARY_CANCEL", "position_conflicts": evaluation["conflicts"]}
        if evaluation["status"] == "PASSED":
            passes += 1
            account_row["days_to_pass"] = account_row["evaluation_days"]
            q_start = evaluation["terminal"] + pd.Timedelta(nanoseconds=1)
            qualified = _run_stage(trades, q_start, horizon_end, account, "QUALIFIED", initial_balance=evaluation["ending_balance"])
            account_row["qualified_net_pnl"] = qualified["net_trading_pnl"]
            account_row["qualified_fees"] = qualified["fees"]
            if qualified["status"] == "FIRST_PAYOUT":
                first_payouts += 1
                total_payout += qualified["payout_received"]
                account_row["payout_gross"] = qualified["payout_gross"]
                account_row["payout_received"] = qualified["payout_received"]
                account_row["net_cashflow"] = qualified["payout_received"] - subscription
                account_row["first_payout"] = True
                account_row["days_to_first_payout"] = (qualified["terminal"] - cursor).total_seconds() / 86400
                cursor = qualified["terminal"] + pd.Timedelta(nanoseconds=1)
            elif qualified["status"] == "FAILED":
                failures += 1
                cursor = qualified["terminal"] + pd.Timedelta(nanoseconds=1)
            else:
                cursor = horizon_end
        else:
            if evaluation["status"] == "FAILED":
                failures += 1
            elif evaluation["status"] == "VOLUNTARY_CANCEL":
                cancellations += 1
            else:
                cursor = horizon_end
            if cursor != horizon_end:
                cursor = evaluation["terminal"] + pd.Timedelta(nanoseconds=1)
        started.append(account_row)
    frame = pd.DataFrame(started)
    pass_days = frame["days_to_pass"].dropna() if not frame.empty else pd.Series(dtype=float)
    payout_days = frame["days_to_first_payout"].dropna() if not frame.empty else pd.Series(dtype=float)
    path = {"run_id": run_id, "portfolio": portfolio, "timeframe": timeframe, "account": account.name, "position_size": size, "horizon_months": horizon, "cancellation_policy": policy, "start": start, "end": horizon_end, "evaluations_started": len(frame), "passes": passes, "failures": failures, "voluntary_cancellations": cancellations, "first_payouts": first_payouts, "subscription_cost": total_subscription, "payouts": total_payout, "reset_cost": total_reset, "net_cashflow": total_payout - total_subscription - total_reset, "positive_net_cashflow": total_payout - total_subscription - total_reset > 0, "days_to_pass_median": pass_days.median() if not pass_days.empty else np.nan, "days_to_first_payout_median": payout_days.median() if not payout_days.empty else np.nan}
    return started, path


def _summary(accounts: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["cancellation_policy", "horizon_months", "portfolio", "timeframe", "account", "position_size"]
    rows = []
    for key, group in accounts.groupby(group_cols, dropna=False):
        path_group = paths
        for column, value in zip(group_cols, key):
            path_group = path_group[path_group[column] == value]
        started = len(group)
        pass_days = group["days_to_pass"].dropna()
        payout_days = group["days_to_first_payout"].dropna()
        rows.append({**dict(zip(group_cols, key)), "started_accounts": started, "pass_probability": (group.evaluation_status == "PASSED").mean(), "failure_probability": (group.evaluation_status == "FAILED").mean(), "voluntary_cancellation_probability": group.cancellation.mean(), "first_payout_probability": group.first_payout.mean(), "median_days_to_pass": pass_days.median() if not pass_days.empty else np.nan, "median_days_to_first_payout": payout_days.median() if not payout_days.empty else np.nan, "average_subscription_cost": group.subscription_cost.mean(), "median_subscription_cost": group.subscription_cost.median(), "average_payout": group.payout_received.mean(), "median_payout": group.payout_received.median(), "average_evaluation_profit": group.evaluation_net_pnl.mean(), "median_evaluation_profit": group.evaluation_net_pnl.median(), "average_net_cashflow": group.net_cashflow.mean(), "median_net_cashflow": group.net_cashflow.median(), "roi_on_external_cost": group.net_cashflow.sum() / group.subscription_cost.sum() if group.subscription_cost.sum() else np.nan, "positive_net_cashflow_probability": (group.net_cashflow > 0).mean(), "qualified_first_payouts": int(group.first_payout.sum()), "path_count": len(path_group), "average_yearly_net_cashflow": path_group.net_cashflow.mean() * (12 / key[1]) if not path_group.empty else np.nan, "positive_year_probability": path_group.positive_net_cashflow.mean() if not path_group.empty else np.nan, "average_yearly_subscription_cost": path_group.subscription_cost.mean() * (12 / key[1]) if not path_group.empty else np.nan, "average_yearly_payouts": path_group.payouts.mean() * (12 / key[1]) if not path_group.empty else np.nan})
    return pd.DataFrame(rows)


def _rank(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    aggregate = frame.groupby(group_cols, dropna=False).agg(expected_yearly_net_cashflow=("average_yearly_net_cashflow", "mean"), probability_positive_year=("positive_year_probability", "mean"), median_days_to_first_payout=("median_days_to_first_payout", "median"), average_yearly_subscription_cost=("average_yearly_subscription_cost", "mean"), paths=("path_count", "sum"), configurations=("account", "nunique")).reset_index()
    aggregate["rank_yearly_cashflow"] = aggregate["expected_yearly_net_cashflow"].rank(method="min", ascending=False)
    aggregate["rank_positive_year"] = aggregate["probability_positive_year"].rank(method="min", ascending=False)
    aggregate["rank_payout_speed"] = aggregate["median_days_to_first_payout"].rank(method="min", ascending=True)
    aggregate["rank_subscription_cost"] = aggregate["average_yearly_subscription_cost"].rank(method="min", ascending=True)
    aggregate["rank_total"] = aggregate[["rank_yearly_cashflow", "rank_positive_year", "rank_payout_speed", "rank_subscription_cost"]].sum(axis=1)
    return aggregate.sort_values(["rank_total", "expected_yearly_net_cashflow"], ascending=[True, False]).reset_index(drop=True)


def _load_and_run() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    streams, skipped, _ = _prepare_recent_streams()
    account_rows, path_rows = [], []
    for portfolio, members in PORTFOLIOS.items():
        for timeframe in fixed.TIMEFRAMES:
            for horizon in HORIZONS_MONTHS:
                starts = _path_starts(members, timeframe, streams, horizon)
                if not starts:
                    skipped.append({"portfolio": portfolio, "timeframe": timeframe, "horizon_months": horizon, "reason": "insufficient common recent history for a complete horizon"})
                for account_name, account in ACCOUNT_SPECS.items():
                    for size in legacy.POSITION_SIZES:
                        trades = _merge_trades(members, timeframe, streams, size)
                        for start in starts:
                            horizon_end = start + pd.DateOffset(months=horizon)
                            for policy in POLICIES:
                                run_id = f"econ-{portfolio}-{timeframe}-{account_name}-{size}-{horizon}-{start.date()}-{policy}".replace(" ", "_").replace("/", "-")
                                rows, path = _simulate_path(trades, start, horizon_end, account, policy, horizon, portfolio, timeframe, size, run_id)
                                account_rows.extend(rows)
                                path_rows.append(path)
    return pd.DataFrame(account_rows), pd.DataFrame(path_rows), skipped


def _write_report(root: Path, summary: pd.DataFrame, policy: pd.DataFrame, continuous: pd.DataFrame, comparison: pd.DataFrame, skipped: list[dict]) -> None:
    def table(frame):
        return frame.to_html(index=False, border=0) if frame is not None and not frame.empty else "<p>None</p>"
    best = policy.iloc[0].to_dict() if not policy.empty else {}
    text = f"""<!doctype html><html><head><meta charset='utf-8'><title>V12 Economics Fixed</title><style>body{{font-family:Arial;margin:2rem;max-width:1600px}}table{{border-collapse:collapse;font-size:11px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}</style></head><body><h1>V12 Alpha Zero Continuous-Trader Economics</h1><div class='warn'><b>Economics-only study.</b> Frozen V12 strategy and Binance data layer were unchanged. Only the most recent {RECENT_DAYS} days of cached data were used. One account was alive at a time; no logs or trade exports were generated. Skipped partitions: {len(skipped)}.</div><h2>Rules and modeling</h2><p>Verified {OFFICIAL_VERIFICATION_DATE}. Sources: <a href='https://help.alpha-futures.com/en/articles/11771813-zero-account-overview'>Zero Account Overview</a>, <a href='https://help.alpha-futures.com/en/articles/9492068-monthly-subscription'>Monthly Subscription</a>, <a href='https://help.alpha-futures.com/en/articles/9492051-payout-policy'>Payout Policy</a>, <a href='https://help.alpha-futures.com/en/articles/9492077-reset'>Reset</a>, <a href='https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard'>Daily Loss Guard</a>, and <a href='https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll'>MLL</a>. Evaluation subscriptions rebill until pass/cancel, Qualified Zero has no monthly subscription, one first payout ends the Qualified lifecycle, and one new Evaluation then begins. BTC maps MBT, ETH maps MET, SPY is excluded because SPX occupies MES. The current 100K Zero option is included, but frozen position sizes only reach 10 micros.</p><h2>Best ranked policy aggregate</h2>{table(policy.head(10))}<h2>Cancellation policy comparison</h2>{table(policy)}<h2>Account comparison</h2>{table(comparison.head(20))}<h2>Continuous trader paths</h2>{table(continuous.head(30))}<h2>Limitations</h2><p>Long-history portfolios have many rolling 12-month starts but only one feasible 24-month endpoint in this 730-day window. The all-canonical portfolio was skipped because Silver and QQQ do not provide a complete recent 12-month common history. DLG/MLL floating equity and Qualified news restrictions are limited by retained trade objects.</p></body></html>"""
    (root / "final_report.html").write_text(text, encoding="utf-8")


def run(root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    accounts, paths, skipped = _load_and_run()
    summary = _summary(accounts, paths)
    policy = _rank(summary, ["cancellation_policy", "horizon_months"])
    comparison = _rank(summary, ["account", "horizon_months"])
    if not paths.empty:
        continuous = paths.copy()
        continuous["yearly_net_cashflow"] = continuous["net_cashflow"] * 12 / continuous["horizon_months"]
        continuous["yearly_subscription_cost"] = continuous["subscription_cost"] * 12 / continuous["horizon_months"]
        continuous["yearly_payouts"] = continuous["payouts"] * 12 / continuous["horizon_months"]
    else:
        continuous = paths
    summary.to_csv(root / "account_summary.csv", index=False)
    policy.to_csv(root / "cancellation_policy.csv", index=False)
    continuous.to_csv(root / "continuous_trader.csv", index=False)
    comparison.to_csv(root / "account_comparison.csv", index=False)
    _write_report(root, summary, policy, continuous, comparison, skipped)
    return {"started_accounts": len(accounts), "continuous_paths": len(paths), "skipped": skipped, "root": str(root)}


if __name__ == "__main__":
    print(run())
