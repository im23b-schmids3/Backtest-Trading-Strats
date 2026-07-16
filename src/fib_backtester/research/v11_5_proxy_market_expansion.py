from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.research.v9_alpha_risk_engine import _session


ROOT = Path("reports/v11_5")
DATA_ROOT = Path("data/v11_5_proxy_raw")
TIMEFRAMES = ("1h", "4h")
PROXY_START = pd.Timestamp("2024-07-15T00:00:00Z")
FROZEN_DISTANCE = 4
FROZEN_MIN_MOVE = 0.0025
FROZEN_ENTRY = 0.900
FROZEN_INITIAL_STOP = 1.020
FROZEN_POST_TP1_STOP = 0.820
FROZEN_TP_ALLOCATION = "30/25/20/15/10"
PROXY_SOURCE = "Yahoo Finance historical OHLC proxy; proxy validation only"
INITIAL_CAPITAL = 10_000.0

PROXIES = {
    "ETH": {"alpha_market": "MET", "ticker": "ETH-USD", "description": "ETHUSDT spot proxy for Micro Ether", "proxy_type": "crypto spot", "source_url": "https://finance.yahoo.com/quote/ETH-USD/history/"},
    "QQQ": {"alpha_market": "MNQ/NQ", "ticker": "QQQ", "description": "QQQ ETF proxy for Nasdaq futures", "proxy_type": "equity ETF", "source_url": "https://finance.yahoo.com/quote/QQQ/history/"},
    "SPY": {"alpha_market": "MES/ES", "ticker": "SPY", "description": "SPY ETF proxy for S&P 500 futures", "proxy_type": "equity ETF", "source_url": "https://finance.yahoo.com/quote/SPY/history/"},
    "XAUUSD": {"alpha_market": "MGC/GC", "ticker": "GLD", "description": "GLD ETF proxy for gold", "proxy_type": "precious-metal ETF", "source_url": "https://finance.yahoo.com/quote/GLD/history/"},
    "XAGUSD": {"alpha_market": "SIL/SI", "ticker": "SLV", "description": "SLV ETF proxy for silver", "proxy_type": "precious-metal ETF", "source_url": "https://finance.yahoo.com/quote/SLV/history/"},
    "WTI": {"alpha_market": "MCL/CL", "ticker": "USO", "description": "USO ETF proxy for WTI crude oil", "proxy_type": "energy ETF", "source_url": "https://finance.yahoo.com/quote/USO/history/"},
    "NATGAS": {"alpha_market": "NG/MNG", "ticker": "UNG", "description": "UNG ETF proxy for natural gas", "proxy_type": "energy ETF", "source_url": "https://finance.yahoo.com/quote/UNG/history/"},
    "EURUSD": {"alpha_market": "M6E/6E", "ticker": "EURUSD=X", "description": "EUR/USD spot proxy for Euro FX futures", "proxy_type": "FX spot", "source_url": "https://finance.yahoo.com/quote/EURUSD=X/history/"},
    "IWM": {"alpha_market": "M2K/RTY", "ticker": "IWM", "description": "IWM ETF proxy for Russell 2000 futures", "proxy_type": "equity ETF", "source_url": "https://finance.yahoo.com/quote/IWM/history/"},
}
PORTFOLIOS = {
    "Portfolio A - ETH only": ["ETH"],
    "Portfolio B - ETH + QQQ + SPY": ["ETH", "QQQ", "SPY"],
    "Portfolio C - all admitted proxy markets": list(PROXIES),
}
ACCOUNT_SPECS = {
    "25K Zero": {"account_size": 25_000.0, "target": 1_500.0, "mll": 1_000.0, "dlg": 500.0, "payout_max": 1_000.0},
    "50K Zero": {"account_size": 50_000.0, "target": 3_000.0, "mll": 2_000.0, "dlg": 1_000.0, "payout_max": 1_500.0},
}


def run_v11_5_proxy_market_expansion(config: RunConfig, root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results, trades_by_key, curves_by_key, inventory = [], {}, {}, []
    skipped = []
    for asset, metadata in PROXIES.items():
        hourly = _read_proxy(asset)
        if hourly is None:
            skipped.append({"market": asset, "reason": "retained proxy parquet is missing or invalid"})
            continue
        start = max(PROXY_START, hourly.index.min())
        end = hourly.index.max()
        for timeframe in TIMEFRAMES:
            bars = _to_timeframe(hourly, timeframe).loc[start:end]
            if len(bars) < 300:
                skipped.append({"market": asset, "timeframe": timeframe, "reason": "fewer than 300 retained bars after common-window restriction"})
                continue
            pair_config = _proxy_config(config, asset, timeframe)
            try:
                trades, equity = _run_frozen(pair_config, asset, timeframe, bars)
            except Exception as exc:
                skipped.append({"market": asset, "timeframe": timeframe, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            split = _splits(bars.index[0], bars.index[-1])
            for stage, stage_start, stage_end in split:
                metrics = _metrics(trades, equity, stage_start, stage_end)
                results.append({
                    **_market_fields(asset, metadata, timeframe, bars, stage, stage_start, stage_end),
                    **metrics,
                    "status": "proxy_validation_result",
                    "proxy_validation": True,
                    "native_futures_validation": False,
                    "parameter_policy": "frozen global distance=4, minimum_move=0.0025; no optimization",
                })
            trades_by_key[(asset, timeframe)] = trades
            curves_by_key[(asset, timeframe)] = equity
            inventory.append(_market_fields(asset, metadata, timeframe, bars, "data", bars.index[0], bars.index[-1]))

    market_frame = pd.DataFrame(results)
    rankings = _rankings(market_frame)
    portfolio_frame = _portfolio_analysis(trades_by_key, curves_by_key)
    contributions = _contributions(market_frame, portfolio_frame, trades_by_key)
    prop = _prop_results(trades_by_key, config.seed)
    market_frame.to_csv(root / "v11_5_proxy_markets.csv", index=False)
    rankings.to_csv(root / "v11_5_market_rankings.csv", index=False)
    portfolio_frame.to_csv(root / "v11_5_portfolio_analysis.csv", index=False)
    prop.to_csv(root / "v11_5_prop_results.csv", index=False)
    contributions.to_csv(root / "v11_5_market_contributions.csv", index=False)
    _write_report(root / "v11_5_final_report.html", market_frame, rankings, portfolio_frame, prop, contributions, skipped)
    return {"markets": len(PROXIES), "market_timeframes_evaluated": len(trades_by_key), "skipped": skipped, "root": str(root), "status": "proxy_validation_only"}


def _read_proxy(asset: str) -> pd.DataFrame | None:
    path = DATA_ROOT / f"{asset}_1h.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()[["open", "high", "low", "close", "volume"]].dropna()
        frame = frame[~frame.index.duplicated(keep="last")]
        valid = (frame.high >= frame[["open", "close", "low"]].max(axis=1)) & (frame.low <= frame[["open", "close", "high"]].min(axis=1))
        return frame.loc[valid]
    except Exception:
        return None


def _to_timeframe(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1h":
        return frame
    return frame.resample("4h", origin="epoch", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _proxy_config(config: RunConfig, asset: str, timeframe: str) -> RunConfig:
    asset_config = AssetConfig(PROXIES[asset]["ticker"], "yfinance", 0.001, 0.0002)
    return replace(config, assets=[asset], timeframes=[timeframe], initial_cash=INITIAL_CAPITAL, min_pivot_distance=FROZEN_DISTANCE, max_positions=1, asset_configs={asset: asset_config})


def _run_frozen(config: RunConfig, asset: str, timeframe: str, bars: pd.DataFrame):
    engine = StrategyV7FrozenValidationEngine(config, FROZEN_MIN_MOVE)
    return engine.run({asset: bars})


def _splits(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = end - start
    train_end = start + span * 0.60
    validation_end = start + span * 0.80
    return [("training", start, train_end), ("validation", train_end + pd.Timedelta(nanoseconds=1), validation_end), ("holdout", validation_end + pd.Timedelta(nanoseconds=1), end)]


def _metrics(trades, equity, start, end):
    trades = trades.copy() if trades is not None else pd.DataFrame()
    equity = equity.copy() if equity is not None else pd.DataFrame()
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity = equity.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        equity = equity[(equity.timestamp >= pd.Timestamp(start)) & (equity.timestamp <= pd.Timestamp(end))]
    if not trades.empty:
        fills = pd.to_datetime(trades.fill_timestamp, utc=True)
        trades = trades[(fills >= pd.Timestamp(start)) & (fills <= pd.Timestamp(end))]
    base = float(equity.equity.iloc[0]) if not equity.empty else INITIAL_CAPITAL
    final = float(equity.equity.iloc[-1]) if not equity.empty else base
    n = len(trades)
    net = float(trades.net_pnl.sum()) if n else 0.0
    gross = float(trades.gross_pnl.sum()) if n else 0.0
    fees = float(trades.fees.sum()) if n and "fees" in trades else 0.0
    slip = float(trades.slippage_cost.sum()) if n and "slippage_cost" in trades else 0.0
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"] if n else pd.Series(dtype=float)
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"] if n else pd.Series(dtype=float)
    curve = equity.equity.astype(float) if not equity.empty else pd.Series(dtype=float)
    returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0]
    periods = 365.25 * 24 if len(equity) < 2 else 365.25 * 86400 / max(equity.timestamp.diff().dt.total_seconds().median(), 1)
    dd = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400, 1 / 24)
    total = final / base - 1 if base else 0.0
    annual = (1 + total) ** (365.25 / days) - 1 if 1 + total > 0 else -1.0
    return {
        "number_of_trades": n, "trades_per_month": n * 30.4375 / days, "trades_per_year": n * 365.25 / days,
        "initial_capital": base, "final_equity": final, "total_return": total, "cagr": annual,
        "net_pnl": net, "gross_pnl": gross, "fees": fees, "slippage": slip,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "win_rate": float((trades.net_pnl > 0).mean()) if n else 0.0,
        "expectancy": float(trades.net_pnl.mean()) if n else 0.0,
        "sharpe_ratio": float(periods**0.5 * returns.mean() / returns.std(ddof=0)) if len(returns) and returns.std(ddof=0) else 0.0,
        "sortino_ratio": float(periods**0.5 * returns.mean() / downside.std(ddof=0)) if len(downside) and downside.std(ddof=0) else 0.0,
        "maximum_drawdown": float(dd.min()) if len(dd) else 0.0,
        "average_r": float(pd.to_numeric(trades.r_multiple, errors="coerce").mean()) if n and "r_multiple" in trades else 0.0,
        "average_holding_hours": float(trades.holding_hours.mean()) if n and "holding_hours" in trades else 0.0,
        "long_trades": int((trades.side == "long").sum()) if n else 0, "short_trades": int((trades.side == "short").sum()) if n else 0,
    }


def _market_fields(asset, metadata, timeframe, bars, stage, stage_start, stage_end):
    return {"market": asset, "alpha_market": metadata["alpha_market"], "proxy_ticker": metadata["ticker"], "proxy_description": metadata["description"], "proxy_type": metadata["proxy_type"], "proxy_source": PROXY_SOURCE, "proxy_source_url": metadata["source_url"], "timeframe": timeframe, "stage": stage, "first_timestamp": str(bars.index[0]), "last_timestamp": str(bars.index[-1]), "candle_count": len(bars), "stage_start": str(stage_start), "stage_end": str(stage_end), "native_futures": False, "proxy_validation_only": True}


def _rankings(frame):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for market, group in frame.groupby("market", sort=True):
        val = group[group.stage == "validation"]
        hold = group[group.stage == "holdout"]
        validation_return = float(val.total_return.mean()) if not val.empty else np.nan
        holdout_return = float(hold.total_return.mean()) if not hold.empty else np.nan
        dd = float(abs(pd.concat([val.maximum_drawdown, hold.maximum_drawdown]).mean())) if not val.empty and not hold.empty else np.nan
        trades = int(group[group.stage == "holdout"].number_of_trades.sum()) if not hold.empty else 0
        validation_trades = int(val.number_of_trades.sum()) if not val.empty else 0
        sample_penalty = 0.08 * max(0, 5 - min(validation_trades, trades)) / 5
        score = 0.45 * np.nan_to_num(validation_return) + 0.45 * np.nan_to_num(holdout_return) - 0.10 * np.nan_to_num(dd) - sample_penalty
        if validation_trades >= 5 and trades >= 5 and validation_return > 0 and holdout_return > 0:
            evidence = "strong proxy evidence; native futures validation still required"
        elif trades >= 5 and holdout_return > 0:
            evidence = "mixed proxy evidence; native futures validation still required"
        elif trades >= 5:
            evidence = "negative or unstable proxy evidence; native futures validation still required"
        else:
            evidence = "insufficient proxy trade evidence; native futures validation still required"
        rows.append({"market": market, "alpha_market": group.alpha_market.iloc[0], "proxy_ticker": group.proxy_ticker.iloc[0], "validation_return": validation_return, "holdout_return": holdout_return, "validation_trades": validation_trades, "holdout_trades": trades, "validation_sharpe": float(val.sharpe_ratio.mean()) if not val.empty else np.nan, "holdout_sharpe": float(hold.sharpe_ratio.mean()) if not hold.empty else np.nan, "validation_drawdown": float(val.maximum_drawdown.mean()) if not val.empty else np.nan, "holdout_drawdown": float(hold.maximum_drawdown.mean()) if not hold.empty else np.nan, "robustness_score": score, "evidence": evidence})
    result = pd.DataFrame(rows).sort_values("robustness_score", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def _portfolio_analysis(trades_by_key, curves_by_key):
    rows = []
    for portfolio, markets in PORTFOLIOS.items():
        for timeframe in TIMEFRAMES:
            keys = [(m, timeframe) for m in markets if (m, timeframe) in curves_by_key]
            if not keys:
                continue
            for stage, frac_start, frac_end in (("training", 0.0, 0.6), ("validation", 0.6, 0.8), ("holdout", 0.8, 1.0)):
                series = []
                all_trades = []
                for key in keys:
                    curve = curves_by_key[key].copy()
                    curve["timestamp"] = pd.to_datetime(curve.timestamp, utc=True)
                    curve = curve.sort_values("timestamp").set_index("timestamp").equity.astype(float)
                    norm = curve / curve.iloc[0]
                    series.append(norm.rename(key[0]))
                    all_trades.append(trades_by_key[key])
                combined = pd.concat(series, axis=1).sort_index().ffill().dropna()
                normalized = combined.mean(axis=1)
                start = normalized.index[0] + (normalized.index[-1] - normalized.index[0]) * frac_start
                end = normalized.index[0] + (normalized.index[-1] - normalized.index[0]) * frac_end
                window = normalized.loc[start:end]
                returns = window.pct_change().dropna()
                dd = window / window.cummax() - 1
                joined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
                if not joined.empty:
                    fills = pd.to_datetime(joined.fill_timestamp, utc=True)
                    joined = joined[(fills >= start) & (fills <= end)]
                n = len(joined)
                wins = joined.loc[joined.net_pnl > 0, "net_pnl"] if n else pd.Series(dtype=float)
                losses = joined.loc[joined.net_pnl < 0, "net_pnl"] if n else pd.Series(dtype=float)
                rows.append({"portfolio": portfolio, "timeframe": timeframe, "stage": stage, "markets": ",".join(m for m, _ in keys), "trades": n, "trades_per_month": n * 30.4375 / max((end - start).total_seconds() / 86400, 1 / 24), "trades_per_year": n * 365.25 / max((end - start).total_seconds() / 86400, 1 / 24), "return": float(window.iloc[-1] / window.iloc[0] - 1) if len(window) else 0.0, "maximum_drawdown": float(dd.min()) if len(dd) else 0.0, "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0, "sharpe_ratio": float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if len(returns) and returns.std(ddof=0) else 0.0, "status": "proxy-based research result"})
    return pd.DataFrame(rows)


def _contributions(market_frame, portfolio_frame, trades_by_key):
    rows = []
    for portfolio, markets in PORTFOLIOS.items():
        for timeframe in TIMEFRAMES:
            p = portfolio_frame[(portfolio_frame.portfolio == portfolio) & (portfolio_frame.timeframe == timeframe)]
            for market in markets:
                m = market_frame[(market_frame.market == market) & (market_frame.timeframe == timeframe)]
                if m.empty or p.empty:
                    continue
                hold = m[m.stage == "holdout"].iloc[0]
                total_return = float(p[p.stage == "holdout"]["return"].iloc[0]) if not p[p.stage == "holdout"].empty else np.nan
                base_events = _events_for_markets(markets, timeframe, trades_by_key)
                for account, spec in ACCOUNT_SPECS.items():
                    base_prop = _aggregate_prop(base_events, spec)
                    reduced_prop = _aggregate_prop(_events_for_markets([m for m in markets if m != market], timeframe, trades_by_key), spec)
                    rows.append({"portfolio": portfolio, "market": market, "timeframe": timeframe, "account": account, "holdout_market_return": hold.total_return, "holdout_market_drawdown": hold.maximum_drawdown, "contribution_to_portfolio_return": hold.total_return / max(len(markets), 1), "contribution_to_portfolio_drawdown": hold.maximum_drawdown / max(len(markets), 1), "contribution_to_payout_probability": base_prop["first_payout_probability"] - reduced_prop["first_payout_probability"], "rank_basis": "holdout proxy return/drawdown and marginal normalized proxy payout probability", "status": "proxy-based research result"})
    return pd.DataFrame(rows)


def _events_for_markets(markets, timeframe, trades_by_key):
    frames = []
    for market in markets:
        frame = trades_by_key.get((market, timeframe))
        if frame is None or frame.empty:
            continue
        item = frame.copy()
        item["event_timestamp"] = pd.to_datetime(item.exit_timestamp, utc=True)
        item["normalized_return"] = item.net_pnl.astype(float) / INITIAL_CAPITAL
        frames.append(item[["event_timestamp", "normalized_return"]])
    return pd.concat(frames, ignore_index=True).sort_values("event_timestamp") if frames else pd.DataFrame(columns=["event_timestamp", "normalized_return"])


def _aggregate_prop(events, spec):
    if events.empty:
        return {"pass_probability": 0.0, "first_payout_probability": 0.0}
    dates = pd.DatetimeIndex(events.event_timestamp.dt.normalize().unique())
    starts = dates[::max(1, len(dates) // 12)]
    results = [_simulate_prop(events[events.event_timestamp >= start], start, spec) for start in starts]
    return {"pass_probability": float(np.mean([r["passed"] for r in results])), "first_payout_probability": float(np.mean([r["payout_count"] >= 1 for r in results]))}


def _prop_results(trades_by_key, seed):
    rows = []
    for portfolio, markets in PORTFOLIOS.items():
        for timeframe in TIMEFRAMES:
            trades = []
            for market in markets:
                if (market, timeframe) not in trades_by_key:
                    continue
                frame = trades_by_key[(market, timeframe)].copy()
                if frame.empty:
                    continue
                frame["event_timestamp"] = pd.to_datetime(frame.exit_timestamp, utc=True)
                frame["normalized_return"] = frame.net_pnl.astype(float) / INITIAL_CAPITAL
                trades.append(frame[["event_timestamp", "normalized_return"]])
            events = pd.concat(trades, ignore_index=True).sort_values("event_timestamp") if trades else pd.DataFrame(columns=["event_timestamp", "normalized_return"])
            if events.empty:
                continue
            starts = pd.DatetimeIndex(events.event_timestamp.dt.normalize().unique())[::max(1, len(pd.DatetimeIndex(events.event_timestamp.dt.normalize().unique())) // 12)]
            for account_name, spec in ACCOUNT_SPECS.items():
                for start in starts:
                    result = _simulate_prop(events[events.event_timestamp >= start], start, spec)
                    rows.append({"portfolio": portfolio, "timeframe": timeframe, "account": account_name, "evaluation_scope": "historical_start_run", "start": str(start), "pass_probability": float(result["passed"]), "first_payout_probability": float(result["payout_count"] >= 1), "second_payout_probability": float(result["payout_count"] >= 2), "average_account_lifetime_days": result["lifetime_days"], "average_days_to_pass": result["days_to_pass"], "average_days_to_first_payout": result["days_to_first_payout"], "average_payouts": result["payout_count"], "maximum_drawdown": result["maximum_drawdown"], "most_common_failure_reason": result["failure_reason"], "seed": seed, "status": "proxy-based research result; not native contract replay"})
                runs = pd.DataFrame([row for row in rows if row["portfolio"] == portfolio and row["timeframe"] == timeframe and row["account"] == account_name and row["evaluation_scope"] == "historical_start_run"])
                if not runs.empty:
                    rows.append({"portfolio": portfolio, "timeframe": timeframe, "account": account_name, "evaluation_scope": "aggregate_historical_starts", "start": "multiple", "pass_probability": runs.pass_probability.mean(), "first_payout_probability": runs.first_payout_probability.mean(), "second_payout_probability": runs.second_payout_probability.mean(), "average_account_lifetime_days": runs.average_account_lifetime_days.mean(), "average_days_to_pass": runs.average_days_to_pass.mean(), "average_days_to_first_payout": runs.average_days_to_first_payout.mean(), "average_payouts": runs.average_payouts.mean(), "maximum_drawdown": runs.maximum_drawdown.mean(), "most_common_failure_reason": runs.most_common_failure_reason.mode().iloc[0] if not runs.most_common_failure_reason.mode().empty else "none", "seed": seed, "status": "proxy-based research result; aggregate probability estimate"})
    return pd.DataFrame(rows)


def _simulate_prop(events, start, spec):
    balance = spec["account_size"]
    mll = balance - spec["mll"]
    high_eod = balance
    daily_profit, cycle_days, last_session, locked = {}, {}, None, None
    passed, failed, pass_time, first_payout = False, False, None, None
    payout_count, cycle_profit, winning_days = 0, 0.0, set()
    failure_reason = "end_of_history"
    equity = [balance]

    def finish_day(session):
        nonlocal balance, mll, cycle_profit, cycle_days, winning_days, payout_count, first_payout
        profit = daily_profit.get(session, 0.0)
        if passed and profit > 0:
            cycle_days[session] = profit
            winning_days.add(session)
        if passed and len(winning_days) >= 5 and cycle_profit > 0:
            request = min(0.50 * cycle_profit, spec["payout_max"])
            # With normalized proxy returns there is no contract-level daily
            # leg data, so this is an illustrative application of the verified
            # qualified payout rule, not a native account replay.
            largest_day = max(cycle_days.values(), default=0.0)
            if request >= 200 and largest_day < 0.40 * cycle_profit and balance - request > mll:
                balance -= request
                payout_count += 1
                first_payout = first_payout or pd.Timestamp(session, tz="America/New_York")
                cycle_profit = 0.0
                cycle_days = {}
                winning_days = set()
        mll = min(spec["account_size"], max(mll, high_eod - spec["mll"]))

    for row in events.itertuples(index=False):
        ts = pd.Timestamp(row.event_timestamp)
        session = _session(ts)
        if last_session is not None and session != last_session:
            finish_day(last_session)
            daily_profit.pop(last_session, None)
            locked = None
        last_session = session
        if locked == session or failed:
            continue
        pnl = float(row.normalized_return) * balance
        balance += pnl
        daily_profit[session] = daily_profit.get(session, 0.0) + pnl
        cycle_profit += pnl if passed else 0.0
        high_eod = max(high_eod, balance)
        equity.append(balance)
        if daily_profit[session] <= -spec["dlg"]:
            locked = session
        if balance <= mll:
            failed, failure_reason = True, "Maximum Loss Violation"
            break
        if not passed and balance >= spec["account_size"] + spec["target"]:
            passed, pass_time, cycle_profit, cycle_days, winning_days = True, ts, 0.0, {}, set()
    if last_session is not None and not failed:
        finish_day(last_session)
    lifetime = (ts - pd.Timestamp(start)).total_seconds() / 86400 if "ts" in locals() else 0.0
    dd = pd.Series(equity) / pd.Series(equity).cummax() - 1
    return {"passed": passed, "payout_count": payout_count, "lifetime_days": lifetime, "days_to_pass": (pass_time - pd.Timestamp(start)).total_seconds() / 86400 if pass_time else np.nan, "days_to_first_payout": (first_payout - pd.Timestamp(start)).total_seconds() / 86400 if first_payout else np.nan, "maximum_drawdown": float(dd.min()), "failure_reason": failure_reason}


def _write_report(path, market_frame, rankings, portfolio, prop, contributions, skipped):
    best = rankings.iloc[0].market if not rankings.empty else "none"
    strongest_frame = rankings[rankings.evidence.str.startswith("strong")] if not rankings.empty else pd.DataFrame()
    strongest = ", ".join(strongest_frame.market.astype(str).tolist()) if not strongest_frame.empty else (", ".join(rankings.head(3).market.astype(str).tolist()) if not rankings.empty else "none")
    excluded_frame = rankings[(rankings.holdout_trades < 5) | (rankings.holdout_return <= 0)] if not rankings.empty else pd.DataFrame()
    excluded = ", ".join(excluded_frame.market.astype(str).tolist()) if not excluded_frame.empty else "none"
    best_portfolio = "none"
    best_trade_rate = np.nan
    recommended = "none"
    if not portfolio.empty:
        hold = portfolio[portfolio.stage == "holdout"]
        if not hold.empty:
            best_row = hold.loc[hold.trades_per_month.idxmax()]
            best_portfolio, best_trade_rate = best_row.portfolio, best_row.trades_per_month
            positive = hold[hold["return"] > 0]
            recommended = positive.loc[positive.sharpe_ratio.idxmax(), "portfolio"] + " / " + positive.loc[positive.sharpe_ratio.idxmax(), "timeframe"] if not positive.empty else best_portfolio
    aggregate = prop[prop.evaluation_scope == "aggregate_historical_starts"] if not prop.empty else pd.DataFrame()
    prop_lines = ""
    if not aggregate.empty:
        for account in ACCOUNT_SPECS:
            account_rows = aggregate[aggregate.account == account]
            if not account_rows.empty:
                best_account = account_rows.loc[account_rows.pass_probability.idxmax()]
                prop_lines += f"<li><b>Proxy-based research result.</b> {html.escape(account)} highest pass estimate: {best_account.pass_probability:.1%} ({html.escape(best_account.portfolio)}, {best_account.timeframe}); average payouts in that row: {best_account.average_payouts:.2f}; average lifetime: {best_account.average_account_lifetime_days:.1f} days.</li>"
    summary = f"<p><b>Proxy-based research result.</b> The frozen strategy was replayed unchanged at 1H and 4H using nine documented price proxies over the common recent window. This is not native futures validation and cannot establish executable Alpha Futures performance.</p><p><b>Proxy-based research result.</b> Strongest descriptive markets: {html.escape(strongest)}. Markets with no holdout trades and therefore insufficient proxy evidence: {html.escape(excluded)}. Highest holdout trade frequency: {html.escape(str(best_portfolio))} at {best_trade_rate:.2f} trades/month. A descriptive proxy portfolio candidate is {html.escape(recommended)}; it is not a production recommendation.</p><p><b>Proxy-based research result.</b> Native futures markets admitted: 0. ETF/spot prices do not reproduce CME contract rolls, tick values, session holidays, commissions, liquidity, slippage, or intrabar order paths. Prop rows use normalized proxy trade returns and are illustrative only.</p><ul>{prop_lines}</ul><p><b>Proxy-based research result.</b> Future native validation should prioritize the top-ranked proxy families only after acquiring contract-level data: {html.escape(strongest)}.</p>"
    sections = [("Market proxy results", market_frame), ("Market rankings", rankings), ("Portfolio analysis", portfolio), ("Prop results", prop), ("Market contributions", contributions)]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, border=0)}" for title, frame in sections)
    skipped_html = html.escape("; ".join(f"{x.get('market')} {x.get('timeframe', '')}: {x['reason']}" for x in skipped) or "none")
    path.write_text(f"<html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:2em}}table{{border-collapse:collapse;font-size:11px}}th,td{{padding:4px 6px;border:1px solid #ddd;white-space:nowrap}}th{{background:#eee}}</style></head><body><h1>Strategy V11.5 Proxy Market Expansion</h1>{summary}<p><b>Skipped:</b> {skipped_html}</p>{tables}</body></html>", encoding="utf-8")
