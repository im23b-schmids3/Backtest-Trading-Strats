"""Strategy V12 Binance-proxy multi-market and Alpha Zero simulation.

This is deliberately an analysis harness.  It reuses the frozen V7 engine and
does not alter any strategy or backtesting implementation.  Binance candles
are loaded only from the completed Infrastructure V3 cache.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.research.v12_contract_registry import CONTRACTS, PROXY_SYMBOLS, PROXY_TO_CONTRACT, mapped_price, round_to_tick


ROOT = Path("reports/v12")
BINANCE_CACHE = Path("data/binance_cache")
BINANCE_HISTORY = Path("reports/binance/binance_history.csv")
TIMEFRAMES = ("1h", "4h")
FROZEN_DISTANCE = 4
FROZEN_MIN_MOVE = 0.0025
FROZEN_ENTRY = 0.900
FROZEN_INITIAL_STOP = 1.020
FROZEN_POST_TP1_STOP = 0.820
FROZEN_TP_PROFILE = "B:30/25/20/15/10"
INITIAL_CAPITAL = 10_000.0
LONG_HISTORY_YEARS = 2.0
START_WARMUP_DAYS = 30
POSITION_SIZES = (2, 3, 5, 7, 10)
PAYOUT_SPLIT = 0.90
WINNING_DAYS_REQUIRED = 5
WINNING_DAY_MINIMUM = 200.0
CONSISTENCY_LIMIT = 0.40


@dataclass(frozen=True)
class AccountSpec:
    name: str
    account_size: float
    target: float
    mll: float
    daily_loss_guard: float
    subscription: float
    reset: float
    payout_max: float
    max_micros: int


ACCOUNT_SPECS = {
    "25K Zero": AccountSpec("25K Zero", 25_000.0, 1_500.0, 1_000.0, 500.0, 79.0, 69.0, 1_000.0, 10),
    "50K Zero": AccountSpec("50K Zero", 50_000.0, 3_000.0, 2_000.0, 1_000.0, 119.0, 109.0, 1_500.0, 30),
}


@dataclass(frozen=True)
class ProxySpec:
    alpha_product: str
    multiplier: float
    tick_size: float
    tick_value: float


# These are CME contract specifications used only to translate Binance proxy
# price paths into illustrative Alpha-sized dollar paths.  They do not make a
# Binance instrument equivalent to the mapped CME product.
PROXY_SPECS = {
    market: ProxySpec(product, CONTRACTS[product].multiplier, CONTRACTS[product].tick_size, CONTRACTS[product].tick_value)
    for market, product in PROXY_TO_CONTRACT.items()
}


_PROXY_TYPES = {
    "BTC": ("Binance spot BTCUSDT", "crypto"),
    "ETH": ("Binance spot ETHUSDT", "crypto"),
    "Gold": ("Binance spot PAXGUSDT tokenized-gold proxy", "metal"),
    "Silver": ("Binance USD-M XAGUSDT proxy perpetual", "metal"),
    "QQQ": ("Binance USD-M QQQUSDT proxy perpetual", "equity"),
    "S&P proxy": ("Binance USD-M SPXUSDT proxy perpetual", "equity"),
}
PROXY_METADATA = {
    market: {"alpha_market": product, "proxy_type": _PROXY_TYPES[market][0], "history_group": _PROXY_TYPES[market][1]}
    for market, product in PROXY_TO_CONTRACT.items()
}


PORTFOLIO_BASE = {
    "Portfolio A - ETH only": ["ETH"],
    "Portfolio B - ETH + Gold": ["ETH", "Gold"],
    "Portfolio C - BTC + ETH + Gold": ["BTC", "ETH", "Gold"],
}


def run_v12_binance_proxy_prop_simulation(root: str | Path = ROOT, seed: int = 42) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    markets = _load_cached_markets()
    streams, market_rows, validation_rows, skipped = {}, [], [], []

    for market in markets:
        for timeframe in TIMEFRAMES:
            try:
                bars = _read_cached_bars(market, timeframe)
                trades, equity = _run_frozen(market, timeframe, bars)
                streams[(market, timeframe)] = {"bars": bars, "trades": trades, "equity": equity}
                market_rows.append(_market_metrics(market, timeframe, bars, trades, equity))
                validation_rows.extend(_validation_metrics(market, timeframe, bars, trades, equity))
            except Exception as exc:
                skipped.append({"market": market, "timeframe": timeframe, "reason": f"{type(exc).__name__}: {exc}"})

    market_frame = pd.DataFrame(market_rows)
    validation_frame = pd.DataFrame(validation_rows)
    portfolios = _portfolio_members(validation_frame, streams)
    portfolio_frame = _portfolio_metrics(portfolios, streams)
    prepared = _prepare_all_trades(streams)
    lifetimes = _account_lifecycles(portfolios, prepared, streams)
    prop_frame = _prop_summary(lifetimes)
    position_frame = _position_summary(lifetimes)
    economics_frame = _economics_summary(lifetimes)
    realistic_frame = _realistic_trader_summary(portfolios, prepared, streams, seed)
    multi_frame = _multi_account_summary(realistic_frame)

    market_frame.to_csv(root / "v12_market_results.csv", index=False)
    validation_frame.to_csv(root / "v12_market_validation.csv", index=False)
    portfolio_frame.to_csv(root / "v12_portfolios.csv", index=False)
    prop_frame.to_csv(root / "v12_prop_results.csv", index=False)
    lifetimes.to_csv(root / "v12_account_lifetimes.csv", index=False)
    position_frame.to_csv(root / "v12_position_sizes.csv", index=False)
    economics_frame.to_csv(root / "v12_economics.csv", index=False)
    realistic_frame.to_csv(root / "v12_realistic_trader.csv", index=False)
    multi_frame.to_csv(root / "v12_multi_account.csv", index=False)
    _write_report(root / "v12_final_report.html", market_frame, validation_frame, portfolio_frame, prop_frame, position_frame, economics_frame, realistic_frame, multi_frame, skipped)
    return {"markets": len(markets), "streams": len(streams), "skipped": skipped, "root": str(root)}


def _load_cached_markets() -> list[str]:
    if not BINANCE_HISTORY.exists():
        raise FileNotFoundError("Infrastructure V3 report is missing: reports/binance/binance_history.csv")
    history = pd.read_csv(BINANCE_HISTORY)
    selected = history[(history.timeframe == "1h") & history.research_market.isin(PROXY_SPECS)]
    markets = sorted(selected.research_market.dropna().unique().tolist())
    if not markets:
        raise ValueError("No successfully downloaded Binance 1H markets are available")
    return markets


def _cache_file(market: str, timeframe: str) -> Path:
    history = pd.read_csv(BINANCE_HISTORY)
    row = history[(history.research_market == market) & (history.timeframe == timeframe)].iloc[0]
    safe = market.replace(" ", "_")
    return BINANCE_CACHE / f"{safe}_{row.symbol}_{row.market_type}_{timeframe}.parquet"


def _read_cached_bars(market: str, timeframe: str) -> pd.DataFrame:
    path = _cache_file(market, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"cached Binance dataset missing: {path}")
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()[["open", "high", "low", "close", "volume"]].dropna()
    frame = frame[~frame.index.duplicated(keep="last")]
    valid = (frame.high >= frame[["open", "close", "low"]].max(axis=1)) & (frame.low <= frame[["open", "close", "high"]].min(axis=1))
    return frame.loc[valid]


def _run_frozen(market: str, timeframe: str, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee = 0.001 if market in {"BTC", "ETH"} else 0.0005
    config = RunConfig(
        run_name="v12_binance_proxy_frozen",
        seed=42,
        start=str(bars.index[0]),
        initial_cash=INITIAL_CAPITAL,
        assets=[market],
        timeframes=[timeframe],
        swing_n=3,
        min_pivot_distance=FROZEN_DISTANCE,
        max_anchor_age_days={"1h": 30.0, "4h": 60.0, "1d": 180.0},
        entry_max_age_bars=None,
        reentry=False,
        execution_policy="conservative",
        max_positions=1,
        max_total_risk_fraction=0.10,
        leverage=1.0,
        asset_configs={market: AssetConfig(market, "binance", fee, 0.0002)},
    )
    engine = StrategyV7FrozenValidationEngine(config, FROZEN_MIN_MOVE)
    return engine.run({market: bars})


def _stage_windows(index: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    start, end = pd.Timestamp(index[0]), pd.Timestamp(index[-1])
    span = end - start
    first = start + span * 0.60
    second = start + span * 0.80
    return [("training", start, first), ("validation", first + pd.Timedelta(nanoseconds=1), second), ("holdout", second + pd.Timedelta(nanoseconds=1), end)]


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> dict:
    trades = trades.copy() if trades is not None else pd.DataFrame()
    equity = equity.copy() if equity is not None else pd.DataFrame()
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity.timestamp, utc=True)
        equity = equity.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        equity = equity[(equity.timestamp >= start) & (equity.timestamp <= end)]
    if not trades.empty:
        fills = pd.to_datetime(trades.fill_timestamp, utc=True)
        trades = trades[(fills >= start) & (fills <= end)]
    base = float(equity.equity.iloc[0]) if not equity.empty else INITIAL_CAPITAL
    final = float(equity.equity.iloc[-1]) if not equity.empty else base
    days = max((end - start).total_seconds() / 86400, 1 / 24)
    net = float(trades.net_pnl.sum()) if not trades.empty else 0.0
    gross = float(trades.gross_pnl.sum()) if not trades.empty else 0.0
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    curve = equity.equity.astype(float) if not equity.empty else pd.Series(dtype=float)
    returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0]
    periods = 365.25 * (24 if timeframe == "1h" else 6)
    total_return = final / base - 1 if base else 0.0
    annualized = (1 + total_return) ** (365.25 / days) - 1 if 1 + total_return > 0 else -1.0
    drawdown = curve / curve.cummax() - 1 if not curve.empty else pd.Series(dtype=float)
    long = trades[trades.side == "long"] if not trades.empty else trades
    short = trades[trades.side == "short"] if not trades.empty else trades
    return {
        "number_of_trades": len(trades),
        "trades_per_month": len(trades) * 30.4375 / days,
        "trades_per_year": len(trades) * 365.25 / days,
        "initial_capital": base,
        "final_equity": final,
        "gross_return": gross / base if base else 0.0,
        "net_return": net / base if base else 0.0,
        "total_return": total_return,
        "cagr": annualized,
        "annualized_return": annualized,
        "gross_pnl": gross,
        "net_pnl": net,
        "fees": float(trades.fees.sum()) if not trades.empty and "fees" in trades else 0.0,
        "slippage": float(trades.slippage_cost.sum()) if not trades.empty and "slippage_cost" in trades else 0.0,
        "win_rate": float((trades.net_pnl > 0).mean()) if not trades.empty else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() else 0.0,
        "expectancy": float(trades.net_pnl.mean()) if not trades.empty else 0.0,
        "sharpe": float(np.sqrt(periods) * returns.mean() / returns.std(ddof=0)) if len(returns) and returns.std(ddof=0) else 0.0,
        "sortino": float(np.sqrt(periods) * returns.mean() / downside.std(ddof=0)) if len(downside) and downside.std(ddof=0) else 0.0,
        "calmar": float(annualized / abs(drawdown.min())) if len(drawdown) and drawdown.min() < 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "average_trade": float(trades.net_pnl.mean()) if not trades.empty else 0.0,
        "median_trade": float(trades.net_pnl.median()) if not trades.empty else 0.0,
        "average_r": float(pd.to_numeric(trades.r_multiple, errors="coerce").mean()) if not trades.empty and "r_multiple" in trades else 0.0,
        "median_r": float(pd.to_numeric(trades.r_multiple, errors="coerce").median()) if not trades.empty and "r_multiple" in trades else 0.0,
        "long_trade_count": len(long),
        "short_trade_count": len(short),
        "long_net_pnl": float(long.net_pnl.sum()) if not long.empty else 0.0,
        "short_net_pnl": float(short.net_pnl.sum()) if not short.empty else 0.0,
        "long_performance": float(long.net_pnl.sum() / base) if not long.empty and base else 0.0,
        "short_performance": float(short.net_pnl.sum() / base) if not short.empty and base else 0.0,
        "average_holding_hours": float(trades.holding_hours.mean()) if not trades.empty else 0.0,
    }


def _market_metrics(market: str, timeframe: str, bars: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    metrics = _metrics(trades, equity, bars.index[0], bars.index[-1], timeframe)
    years = (bars.index[-1] - bars.index[0]).total_seconds() / (365.25 * 86400)
    history_class = "LONG_HISTORY" if years >= LONG_HISTORY_YEARS else "SHORT_HISTORY_EXPLORATORY"
    return {"record_type": "market", "market": market, "alpha_market": PROXY_METADATA[market]["alpha_market"], "proxy_type": PROXY_METADATA[market]["proxy_type"], "timeframe": timeframe, "first_timestamp": str(bars.index[0]), "last_timestamp": str(bars.index[-1]), "history_years": round(years, 4), "history_class": history_class, "confidence": "proxy-based higher-confidence descriptive result" if history_class == "LONG_HISTORY" else "PROXY-BASED EXPLORATORY RESEARCH - short history", **metrics}


def _validation_metrics(market: str, timeframe: str, bars: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> list[dict]:
    rows = []
    years = (bars.index[-1] - bars.index[0]).total_seconds() / (365.25 * 86400)
    history_class = "LONG_HISTORY" if years >= LONG_HISTORY_YEARS else "SHORT_HISTORY_EXPLORATORY"
    for stage, start, end in _stage_windows(bars.index):
        rows.append({"record_type": "validation", "market": market, "alpha_market": PROXY_METADATA[market]["alpha_market"], "timeframe": timeframe, "stage": stage, "history_years": round(years, 4), "history_class": history_class, "confidence": "proxy-based higher-confidence descriptive result" if history_class == "LONG_HISTORY" else "PROXY-BASED EXPLORATORY RESEARCH - short history", "validation_eligible": "yes" if stage == "validation" and history_class == "LONG_HISTORY" else "no", **_metrics(trades, equity, start, end, timeframe)})
    return rows


def _portfolio_members(validation: pd.DataFrame, streams: dict) -> dict[str, list[str]]:
    members = dict(PORTFOLIO_BASE)
    all_markets = sorted({market for market, _ in streams})
    members["Portfolio D - All Binance markets"] = all_markets
    members["Portfolio E - Profitable validated markets"] = {
        timeframe: [
            market
            for market in all_markets
            if not validation[(validation.market == market) & (validation.timeframe == timeframe) & (validation.stage == "validation") & (validation.history_class == "LONG_HISTORY")].empty
            and float(validation[(validation.market == market) & (validation.timeframe == timeframe) & (validation.stage == "validation") & (validation.history_class == "LONG_HISTORY")].net_return.iloc[0]) > 0
        ]
        for timeframe in TIMEFRAMES
    }
    return members


def _members_for(portfolios: dict, portfolio: str, timeframe: str) -> list[str]:
    value = portfolios[portfolio]
    return value.get(timeframe, []) if isinstance(value, dict) else value


def _portfolio_metrics(portfolios: dict[str, list[str]], streams: dict) -> pd.DataFrame:
    rows = []
    for portfolio in portfolios:
        for timeframe in TIMEFRAMES:
            markets = _members_for(portfolios, portfolio, timeframe)
            keys = [(market, timeframe) for market in markets if (market, timeframe) in streams]
            if not keys:
                rows.append({"record_type": "portfolio_summary", "portfolio": portfolio, "timeframe": timeframe, "markets": ",".join(markets), "status": "NO_ADMITTED_MARKETS", "confidence": "PROXY-BASED EXPLORATORY RESEARCH - no eligible market"})
                continue
            union = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(streams[key]["equity"].timestamp, utc=True)) for key in keys])))
            curves = []
            contributions = {}
            all_trades = []
            for market, _ in keys:
                curve = streams[(market, timeframe)]["equity"].copy()
                curve["timestamp"] = pd.to_datetime(curve.timestamp, utc=True)
                series = curve.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp").equity.astype(float)
                normalized = series.reindex(union).ffill().fillna(1.0) / float(series.iloc[0])
                curves.append(normalized.rename(market))
                # Contribution is measured from the market's own first bar,
                # not from the portfolio's earlier cash placeholder when a
                # shorter-history market joins later.
                contributions[market] = float(series.iloc[-1] / series.iloc[0] - 1) / len(keys)
                all_trades.append(streams[(market, timeframe)]["trades"])
            combined = pd.concat(curves, axis=1).mean(axis=1)
            for stage, start, end in _stage_windows(union):
                window = combined.loc[start:end]
                metrics = _curve_metrics(window, pd.concat(all_trades, ignore_index=True), start, end, timeframe)
                conflict, skipped = _conflict_count(all_trades, max_contracts=10, contracts=2)
                rows.append({"record_type": "portfolio_summary", "portfolio": portfolio, "timeframe": timeframe, "stage": stage, "markets": ",".join(markets), "status": "proxy_based_research", "confidence": "PROXY-BASED EXPLORATORY RESEARCH - native CME equivalence not established", "skipped_trades": skipped, "position_conflicts": conflict, "market_contributions_json": json.dumps(contributions, sort_keys=True), **metrics})
            rows.append({"record_type": "portfolio_full", "portfolio": portfolio, "timeframe": timeframe, "stage": "full_history", "markets": ",".join(markets), "status": "proxy_based_research", "confidence": "PROXY-BASED EXPLORATORY RESEARCH - native CME equivalence not established", "skipped_trades": _conflict_count(all_trades, 10, 2)[1], "position_conflicts": _conflict_count(all_trades, 10, 2)[0], "market_contributions_json": json.dumps(contributions, sort_keys=True), **_curve_metrics(combined, pd.concat(all_trades, ignore_index=True), union[0], union[-1], timeframe)})
            for market, contribution in contributions.items():
                rows.append({"record_type": "market_contribution", "portfolio": portfolio, "timeframe": timeframe, "stage": "full_history", "market": market, "market_contribution_return": contribution, "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _curve_metrics(curve: pd.Series, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> dict:
    curve = curve.dropna().astype(float)
    days = max((end - start).total_seconds() / 86400, 1 / 24)
    returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    down = returns[returns < 0]
    periods = 365.25 * (24 if timeframe == "1h" else 6)
    total = float(curve.iloc[-1] / curve.iloc[0] - 1) if len(curve) else 0.0
    annual = (1 + total) ** (365.25 / days) - 1 if 1 + total > 0 else -1.0
    dd = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)
    fills = trades.copy()
    if not fills.empty:
        timestamps = pd.to_datetime(fills.fill_timestamp, utc=True)
        fills = fills[(timestamps >= start) & (timestamps <= end)]
    wins = fills.loc[fills.net_pnl > 0, "net_pnl"] if not fills.empty else pd.Series(dtype=float)
    losses = fills.loc[fills.net_pnl < 0, "net_pnl"] if not fills.empty else pd.Series(dtype=float)
    return {"trades": len(fills), "trades_per_month": len(fills) * 30.4375 / days, "trades_per_year": len(fills) * 365.25 / days, "return": total, "cagr": annual, "annualized_return": annual, "sharpe": float(np.sqrt(periods) * returns.mean() / returns.std(ddof=0)) if len(returns) and returns.std(ddof=0) else 0.0, "sortino": float(np.sqrt(periods) * returns.mean() / down.std(ddof=0)) if len(down) and down.std(ddof=0) else 0.0, "calmar": float(annual / abs(dd.min())) if len(dd) and dd.min() < 0 else 0.0, "max_drawdown": float(dd.min()) if len(dd) else 0.0, "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0, "expectancy": float(fills.net_pnl.mean()) if not fills.empty else 0.0, "average_trade": float(fills.net_pnl.mean()) if not fills.empty else 0.0, "median_trade": float(fills.net_pnl.median()) if not fills.empty else 0.0, "net_pnl": float(fills.net_pnl.sum()) if not fills.empty else 0.0}


def _prepare_all_trades(streams: dict) -> dict[tuple[str, str, int], list[dict]]:
    prepared = {}
    for (market, timeframe), stream in streams.items():
        for size in POSITION_SIZES:
            prepared[(market, timeframe, size)] = [_prepare_trade(row, market, size) for row in stream["trades"].to_dict("records")]
    return prepared


def _round_tick(value: float, tick: float) -> float:
    return round(value / tick) * tick


def _allocate(total: int) -> list[int]:
    fractions = np.asarray((0.30, 0.25, 0.20, 0.15, 0.10)) * total
    allocation = np.floor(fractions).astype(int)
    for index in np.argsort(-(fractions - allocation))[: total - int(allocation.sum())]:
        allocation[index] += 1
    return allocation.tolist()


def _events(value) -> list[dict]:
    try:
        return json.loads(value) if isinstance(value, str) else (value or [])
    except (TypeError, json.JSONDecodeError):
        return []


def _prepare_trade(raw: dict, market: str, size: int) -> dict:
    spec = PROXY_SPECS[market]
    fee_rate = 0.001 if market in {"BTC", "ETH"} else 0.0005
    entry = _round_tick(float(raw["entry_price"]), spec.tick_size)
    stop = _round_tick(float(raw["initial_stop"]), spec.tick_size)
    allocation = _allocate(size)
    remaining = size
    legs = []
    for event in _events(raw.get("exit_events")):
        reason = event.get("reason", "end_of_test")
        if reason.startswith("tp") and reason[2:].isdigit():
            quantity = min(allocation[int(reason[2:]) - 1], remaining)
        else:
            quantity = remaining
        if quantity <= 0:
            continue
        price = _round_tick(float(event.get("fill_price", event.get("raw_price"))), spec.tick_size)
        direction = 1 if raw["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * quantity
        fee = abs(price * spec.multiplier * quantity) * fee_rate
        legs.append({"timestamp": str(event.get("timestamp", raw["exit_timestamp"])), "reason": reason, "price": price, "quantity": quantity, "gross": gross, "fee": fee, "net": gross - fee})
        remaining -= quantity
        if remaining <= 0:
            break
    if remaining > 0:
        price = _round_tick(float(raw.get("average_exit_price", entry)), spec.tick_size)
        direction = 1 if raw["side"] == "long" else -1
        gross = direction * (price - entry) * spec.multiplier * remaining
        fee = abs(price * spec.multiplier * remaining) * fee_rate
        legs.append({"timestamp": str(raw["exit_timestamp"]), "reason": raw.get("exit_reason", "end_of_test"), "price": price, "quantity": remaining, "gross": gross, "fee": fee, "net": gross - fee})
    entry_fee = abs(entry * spec.multiplier * size) * fee_rate
    exit_time = pd.Timestamp(legs[-1]["timestamp"]) if legs else pd.Timestamp(raw["exit_timestamp"])
    return {"market": market, "alpha_product": spec.alpha_product, "setup_id": raw.get("setup_id", ""), "entry_timestamp": pd.Timestamp(raw["fill_timestamp"]), "exit_timestamp": exit_time, "side": raw["side"], "contracts": size, "entry": entry, "stop": stop, "risk": abs(entry - stop) * spec.multiplier * size, "entry_fee": entry_fee, "legs": legs, "gross_pnl": sum(leg["gross"] for leg in legs), "net_pnl": sum(leg["net"] for leg in legs) - entry_fee, "fees": sum(leg["fee"] for leg in legs) + entry_fee, "holding_hours": (exit_time - pd.Timestamp(raw["fill_timestamp"])).total_seconds() / 3600}


def _portfolio_trades(portfolio: str, timeframe: str, members: list[str], prepared: dict, size: int) -> list[dict]:
    trades = []
    for market in members:
        trades.extend(prepared.get((market, timeframe, size), []))
    return sorted(trades, key=lambda row: (row["entry_timestamp"], row["market"], row["setup_id"]))


def _starts(streams: dict, members: list[str], timeframe: str) -> list[pd.Timestamp]:
    indexes = [streams[(market, timeframe)]["bars"].index for market in members if (market, timeframe) in streams]
    if not indexes:
        return []
    start = min(index[0] for index in indexes) + pd.Timedelta(days=START_WARMUP_DAYS)
    end = max(index[-1] for index in indexes)
    eligible = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz="UTC")
    return [next((ts for ts in index if ts >= month), None) for month in eligible for index in [pd.DatetimeIndex(sorted(set().union(*indexes)))] if next((ts for ts in index if ts >= month), None) is not None]


def _session(timestamp: pd.Timestamp) -> str:
    local = pd.Timestamp(timestamp).tz_convert("America/New_York")
    return str((local - pd.Timedelta(days=1)).date() if local.hour < 18 else local.date())


def _simulate_account(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, account: AccountSpec) -> dict:
    balance = account.account_size
    mll = account.account_size - account.mll
    passed = False
    failed = False
    pass_time = None
    failure_time = None
    failure_reason = "CENSORED"
    active: dict[int, dict] = {}
    events = []
    for number, trade in enumerate(trades):
        if trade["entry_timestamp"] < start or trade["entry_timestamp"] > end:
            continue
        events.append((trade["entry_timestamp"], 1, number, trade, None))
        for leg_index, leg in enumerate(trade["legs"]):
            timestamp = pd.Timestamp(leg["timestamp"])
            if start <= timestamp <= end:
                events.append((timestamp, 0, number, trade, leg_index))
    events.sort(key=lambda row: (row[0], row[1]))
    current_session = None
    daily_profit: dict[str, float] = {}
    locked_session = None
    cycle_profit = 0.0
    winning_days: set[str] = set()
    cycle_days: dict[str, float] = {}
    payouts = []
    consistency_events = 0
    daily_guard_events = 0
    position_conflicts = 0
    skipped_trades = 0
    equity = [balance]

    def finish_day(session: str) -> None:
        nonlocal balance, mll, cycle_profit, consistency_events
        profit = daily_profit.get(session, 0.0)
        if passed and profit > 0:
            cycle_days[session] = profit
        if passed and profit >= WINNING_DAY_MINIMUM:
            winning_days.add(session)
        if passed and len(winning_days) >= WINNING_DAYS_REQUIRED and cycle_profit > 0:
            largest = max(cycle_days.values(), default=0.0)
            if largest >= CONSISTENCY_LIMIT * cycle_profit:
                consistency_events += 1
            else:
                request = min(0.50 * cycle_profit, account.payout_max)
                if request >= WINNING_DAY_MINIMUM and balance - request > mll:
                    balance -= request
                    payouts.append({"timestamp": session, "gross": request, "received": request * PAYOUT_SPLIT})
                    cycle_profit = 0.0
                    winning_days.clear()
                    cycle_days.clear()
        mll = min(account.account_size, max(mll, balance - account.mll))
        daily_profit.pop(session, None)

    for timestamp, kind, number, trade, leg_index in events:
        if failed:
            break
        session = _session(timestamp)
        if current_session is not None and session != current_session:
            finish_day(current_session)
            locked_session = None
        current_session = session
        if kind == 1:
            if locked_session == session:
                skipped_trades += 1
                continue
            current_contracts = sum(row["contracts"] for row in active.values())
            same_market = any(row["market"] == trade["market"] for row in active.values())
            if same_market or current_contracts + trade["contracts"] > account.max_micros:
                position_conflicts += 1
                skipped_trades += 1
                continue
            balance -= trade["entry_fee"]
            daily_profit[session] = daily_profit.get(session, 0.0) - trade["entry_fee"]
            active[number] = trade
            equity.append(balance)
            if balance <= mll:
                failed, failure_time, failure_reason = True, timestamp, "Maximum Loss Violation"
            continue
        if number not in active:
            continue
        leg = trade["legs"][leg_index]
        value = leg["net"]
        balance += value
        daily_profit[session] = daily_profit.get(session, 0.0) + value
        if passed:
            cycle_profit += value
        equity.append(balance)
        if daily_profit[session] <= -account.daily_loss_guard:
            daily_guard_events += 1
            locked_session = session
        if balance <= mll:
            failed, failure_time, failure_reason = True, timestamp, "Maximum Loss Violation"
        if not passed and balance >= account.account_size + account.target:
            passed = True
            pass_time = timestamp
            cycle_profit = 0.0
            winning_days.clear()
            cycle_days.clear()
        if leg_index == len(trade["legs"]) - 1:
            active.pop(number, None)
    if current_session is not None and not failed:
        finish_day(current_session)
    terminal = failure_time or end
    status = "FAILED" if failed else "CENSORED"
    return {"status": status, "censored": status == "CENSORED", "passed": passed, "failed": failed, "failure_reason": failure_reason if failed else "CENSORED", "start_date": str(start), "end_date": str(end), "failure_timestamp": str(failure_time) if failure_time is not None else "", "pass_timestamp": str(pass_time) if pass_time is not None else "", "lifetime_days": (terminal - start).total_seconds() / 86400, "days_to_pass": (pass_time - start).total_seconds() / 86400 if pass_time is not None else np.nan, "days_to_first_payout": (pd.Timestamp(payouts[0]["timestamp"], tz="America/New_York") - start).total_seconds() / 86400 if payouts else np.nan, "days_to_second_payout": (pd.Timestamp(payouts[1]["timestamp"], tz="America/New_York") - start).total_seconds() / 86400 if len(payouts) >= 2 else np.nan, "days_to_third_payout": (pd.Timestamp(payouts[2]["timestamp"], tz="America/New_York") - start).total_seconds() / 86400 if len(payouts) >= 3 else np.nan, "payout_count": len(payouts), "gross_payout": sum(item["gross"] for item in payouts), "net_payout": sum(item["received"] for item in payouts), "average_payout": float(np.mean([item["received"] for item in payouts])) if payouts else 0.0, "trades_per_account": sum(1 for trade in trades if start <= trade["entry_timestamp"] <= terminal), "trades_per_month": sum(1 for trade in trades if start <= trade["entry_timestamp"] <= terminal) * 30.4375 / max((terminal - start).total_seconds() / 86400, 1), "subscription_cost": max(1, int(np.ceil(((pass_time or terminal) - start).total_seconds() / 86400 / (365.25 / 12)))) * account.subscription if (pass_time or terminal) > start else 0.0, "reset_cost": account.reset if failed else 0.0, "account_purchases": 1, "daily_loss_guard_events": daily_guard_events, "position_conflicts": position_conflicts, "skipped_trades": skipped_trades, "consistency_events": consistency_events, "ending_balance": balance, "maximum_drawdown": float(pd.Series(equity).div(pd.Series(equity).cummax()).sub(1).min()) if equity else 0.0}


def _account_lifecycles(portfolios: dict[str, list[str]], prepared: dict, streams: dict) -> pd.DataFrame:
    rows = []
    for portfolio in portfolios:
        for timeframe in TIMEFRAMES:
            members = _members_for(portfolios, portfolio, timeframe)
            if not members or not any((market, timeframe) in streams for market in members):
                continue
            starts = _starts(streams, members, timeframe)
            end = max(streams[(market, timeframe)]["bars"].index[-1] for market in members if (market, timeframe) in streams)
            for account_name, account in ACCOUNT_SPECS.items():
                for size in POSITION_SIZES:
                    trades = _portfolio_trades(portfolio, timeframe, members, prepared, size)
                    for start in starts:
                        result = _simulate_account(trades, start, end, account)
                        rows.append({"portfolio": portfolio, "timeframe": timeframe, "markets": ",".join(members), "account": account_name, "position_size": f"{size} micros", "position_contracts": size, "confidence": "PROXY-BASED EXPLORATORY RESEARCH", **result})
    return pd.DataFrame(rows)


def _prop_summary(lifetimes: pd.DataFrame) -> pd.DataFrame:
    if lifetimes.empty:
        return lifetimes
    rows = []
    group_cols = ["portfolio", "timeframe", "account", "position_size"]
    for key, group in lifetimes.groupby(group_cols, sort=False):
        portfolio, timeframe, account, position = key
        rows.append({"portfolio": portfolio, "timeframe": timeframe, "account": account, "position_size": position, "evaluations": len(group), "pass_probability": group.passed.mean(), "first_payout_probability": (group.payout_count >= 1).mean(), "second_payout_probability": (group.payout_count >= 2).mean(), "third_payout_probability": (group.payout_count >= 3).mean(), "failure_rate": group.failed.mean(), "average_days_to_pass": group.days_to_pass.mean(), "median_days_to_pass": group.days_to_pass.median(), "average_days_to_first_payout": group.days_to_first_payout.mean(), "median_days_to_first_payout": group.days_to_first_payout.median(), "average_non_censored_account_lifetime": group.loc[~group.censored, "lifetime_days"].mean(), "censored_account_count": int(group.censored.sum()), "trades_per_account": group.trades_per_account.mean(), "trades_per_month": group.trades_per_month.mean(), "average_payouts": group.payout_count.mean(), "gross_payout": group.gross_payout.mean(), "net_payout_after_90pct_split": group.net_payout.mean(), "average_monthly_payout": (group.net_payout / (group.lifetime_days / (365.25 / 12)).replace(0, np.nan)).mean(), "average_yearly_payout": (group.net_payout / (group.lifetime_days / 365.25).replace(0, np.nan)).mean(), "average_subscription_cost": group.subscription_cost.mean(), "average_reset_cost": group.reset_cost.mean(), "most_common_failure_reason": group.loc[group.failed, "failure_reason"].mode().iloc[0] if group.failed.any() else "none", "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _position_summary(lifetimes: pd.DataFrame) -> pd.DataFrame:
    if lifetimes.empty:
        return lifetimes
    rows = []
    for (account, position), group in lifetimes.groupby(["account", "position_size"], sort=False):
        rows.append({"scope": "all portfolios", "account": account, "position_size": position, "evaluations": len(group), "pass_probability": group.passed.mean(), "first_payout_probability": (group.payout_count >= 1).mean(), "failure_rate": group.failed.mean(), "average_net_payout": group.net_payout.mean(), "average_drawdown": group.maximum_drawdown.mean(), "robustness_score": 0.45 * group.passed.mean() + 0.35 * (group.payout_count > 0).mean() - 0.20 * group.failed.mean() - 0.05 * abs(group.maximum_drawdown.mean()), "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    result = pd.DataFrame(rows)
    result["robustness_rank"] = result.robustness_score.rank(method="first", ascending=False).astype(int)
    return result.sort_values("robustness_rank")


def _economics_summary(lifetimes: pd.DataFrame) -> pd.DataFrame:
    if lifetimes.empty:
        return lifetimes
    rows = []
    cols = ["portfolio", "timeframe", "account", "position_size"]
    for key, group in lifetimes.groupby(cols, sort=False):
        portfolio, timeframe, account, position = key
        net = group.net_payout - group.subscription_cost - group.reset_cost
        costs = group.subscription_cost + group.reset_cost
        months = group.lifetime_days / (365.25 / 12)
        monthly = net / months.replace(0, np.nan)
        gross = group.gross_payout
        break_even = (costs / monthly.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        rows.append({"portfolio": portfolio, "timeframe": timeframe, "account": account, "position_size": position, "evaluations": len(group), "average_gross_revenue": gross.mean(), "average_subscription_cost": group.subscription_cost.mean(), "average_reset_cost": group.reset_cost.mean(), "average_net_revenue": net.mean(), "average_roi": (net / costs.replace(0, np.nan)).mean(), "average_roi_per_month": (net / costs.replace(0, np.nan) / months.replace(0, np.nan)).mean(), "average_roi_per_year": (net / costs.replace(0, np.nan) / (group.lifetime_days / 365.25).replace(0, np.nan)).mean(), "break_even_months": break_even.mean(), "profitable_evaluation_rate": (net > 0).mean(), "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _realistic_trader_summary(portfolios: dict[str, list[str]], prepared: dict, streams: dict, seed: int) -> pd.DataFrame:
    rows = []
    for portfolio in portfolios:
        for timeframe in TIMEFRAMES:
            members = _members_for(portfolios, portfolio, timeframe)
            if not members or not any((market, timeframe) in streams for market in members):
                continue
            starts = _starts(streams, members, timeframe)
            data_end = max(streams[(market, timeframe)]["bars"].index[-1] for market in members if (market, timeframe) in streams)
            for account_name, account in ACCOUNT_SPECS.items():
                for size in POSITION_SIZES:
                    trades = _portfolio_trades(portfolio, timeframe, members, prepared, size)
                    for horizon_years in (1, 3, 5):
                        horizon_days = horizon_years * 365.25
                        simulations = []
                        for start in starts:
                            end = min(start + pd.Timedelta(days=horizon_days), data_end)
                            simulations.append(_simulate_trader_path(trades, start, end, account))
                        if not simulations:
                            continue
                        frame = pd.DataFrame(simulations)
                        rows.append({"portfolio": portfolio, "timeframe": timeframe, "account": account_name, "position_size": f"{size} micros", "horizon_years": horizon_years, "simulations": len(frame), "complete_window_count": int((~frame.history_censored).sum()), "censored_window_count": int(frame.history_censored.sum()), "average_revenue": frame.net_payout.mean(), "average_gross_revenue": frame.gross_payout.mean(), "average_costs": (frame.subscription_cost + frame.reset_cost).mean(), "average_profit": frame.net_profit.mean(), "average_account_purchases": frame.account_purchases.mean(), "average_subscriptions": frame.subscription_cost.mean(), "average_payouts": frame.payout_count.mean(), "average_roi": (frame.net_profit / (frame.subscription_cost + frame.reset_cost).replace(0, np.nan)).mean(), "average_monthly_profit": (frame.net_profit / horizon_years / 12).mean(), "average_yearly_profit": (frame.net_profit / horizon_years).mean(), "confidence": "PROXY-BASED EXPLORATORY RESEARCH - censored horizons are not complete estimates"})
    return pd.DataFrame(rows)


def _simulate_trader_path(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, account: AccountSpec) -> dict:
    cursor = start
    total = {"gross_payout": 0.0, "net_payout": 0.0, "subscription_cost": 0.0, "reset_cost": 0.0, "payout_count": 0, "account_purchases": 0}
    safety = 0
    while cursor < end and safety < 100:
        result = _simulate_account(trades, cursor, end, account)
        safety += 1
        for key in ("gross_payout", "net_payout", "subscription_cost", "reset_cost"):
            total[key] += float(result[key])
        total["payout_count"] += int(result["payout_count"])
        total["account_purchases"] += 1
        if not result["failed"] or not result["failure_timestamp"]:
            break
        cursor = pd.Timestamp(result["failure_timestamp"]) + pd.Timedelta(nanoseconds=1)
    total["net_profit"] = total["net_payout"] - total["subscription_cost"] - total["reset_cost"]
    max_exit = max((trade["exit_timestamp"] for trade in trades), default=end)
    total["history_censored"] = end >= max_exit and not (locals().get("result") or {}).get("failed", False) and safety < 100
    return total


def _conflict_count(trade_frames: list[pd.DataFrame], max_contracts: int, contracts: int) -> tuple[int, int]:
    trades = []
    for frame in trade_frames:
        if frame is not None and not frame.empty:
            trades.extend(frame.to_dict("records"))
    active = []
    conflicts = 0
    for trade in sorted(trades, key=lambda row: pd.Timestamp(row["fill_timestamp"])):
        entry = pd.Timestamp(trade["fill_timestamp"])
        active = [item for item in active if item[0] > entry]
        if any(item[1] == trade["asset"] for item in active) or len(active) * contracts + contracts > max_contracts:
            conflicts += 1
        else:
            active.append((pd.Timestamp(trade["exit_timestamp"]), trade["asset"], contracts))
    return conflicts, conflicts


def _multi_account_summary(realistic: pd.DataFrame) -> pd.DataFrame:
    if realistic.empty:
        return realistic
    rows = []
    for row in realistic.itertuples(index=False):
        for count in (1, 3, 5):
            rows.append({"portfolio": row.portfolio, "timeframe": row.timeframe, "account": row.account, "position_size": row.position_size, "horizon_years": row.horizon_years, "simultaneous_accounts": count, "simulations": row.simulations, "average_gross_revenue": row.average_gross_revenue * count, "average_costs": row.average_costs * count, "average_profit": row.average_profit * count, "average_account_purchases": row.average_account_purchases * count, "average_subscriptions": row.average_subscriptions * count, "average_payouts": row.average_payouts * count, "average_roi": row.average_roi, "loss_correlation": 1.0, "highly_correlated_losses": "yes - identical strategy and synchronized historical path", "confidence": "PROXY-BASED EXPLORATORY RESEARCH"})
    return pd.DataFrame(rows)


def _write_report(path: Path, market: pd.DataFrame, validation: pd.DataFrame, portfolios: pd.DataFrame, prop: pd.DataFrame, position: pd.DataFrame, economics: pd.DataFrame, realistic: pd.DataFrame, multi: pd.DataFrame, skipped: list[dict]) -> None:
    profitable_1h = market[(market.timeframe == "1h") & (market.net_return > 0)].market.tolist() if not market.empty else []
    profitable_4h = market[(market.timeframe == "4h") & (market.net_return > 0)].market.tolist() if not market.empty else []
    long_markets = market[market.history_class == "LONG_HISTORY"].market.drop_duplicates().tolist() if not market.empty else []
    short_markets = market[market.history_class == "SHORT_HISTORY_EXPLORATORY"].market.drop_duplicates().tolist() if not market.empty else []
    full = portfolios[portfolios.record_type == "portfolio_full"] if not portfolios.empty else pd.DataFrame()
    best_trades = full.loc[full.trades.idxmax()] if not full.empty and "trades" in full else None
    best_return = full.loc[full["return"].idxmax()] if not full.empty and "return" in full else None
    best_risk = full.loc[full.sharpe.idxmax()] if not full.empty and "sharpe" in full else None
    position_best = position.iloc[0].position_size if not position.empty else "unavailable"
    text = f"""
    <p><b>PROXY-BASED EXPLORATORY RESEARCH.</b> Strategy V12 replays the frozen V7/V11.5 strategy unchanged on Binance cached data. Binance instruments are not native CME futures and no Alpha Futures equivalence is claimed.</p>
    <p><b>Frozen parameters:</b> entry Fib {FROZEN_ENTRY:.3f}; initial stop Fib {FROZEN_INITIAL_STOP:.3f}; TP Profile {FROZEN_TP_PROFILE}; post-TP1 stop Fib {FROZEN_POST_TP1_STOP:.3f}; minimum distance {FROZEN_DISTANCE}; minimum move {FROZEN_MIN_MOVE:.2%}; conservative execution; existing fees and slippage. No optimization was performed.</p>
    <p><b>PROXY-BASED EXPLORATORY RESEARCH.</b> Profitable markets by full-history net return: 1H {html.escape(', '.join(profitable_1h) or 'none')}; 4H {html.escape(', '.join(profitable_4h) or 'none')}.</p>
    <p><b>PROXY-BASED EXPLORATORY RESEARCH.</b> Long-history markets (at least {LONG_HISTORY_YEARS:.0f} years): {html.escape(', '.join(long_markets) or 'none')}. Short-history exploratory markets: {html.escape(', '.join(short_markets) or 'none')}.</p>
    <p><b>PROXY-BASED EXPLORATORY RESEARCH.</b> Highest trade-count portfolio: {html.escape(str(best_trades.portfolio) if best_trades is not None else 'unavailable')}. Highest return: {html.escape(str(best_return.portfolio) if best_return is not None else 'unavailable')}. Highest Sharpe: {html.escape(str(best_risk.portfolio) if best_risk is not None else 'unavailable')}. Descriptive position-size leader: {html.escape(str(position_best))}.</p>
    <p><b>Account rules modeled:</b> Alpha Zero 25K and 50K target/MLL/Daily Loss Guard, maximum micros, 90% split, five $200+ winning days, 40% qualified consistency, payout caps, monthly subscriptions and evaluation reset fees. Daily-loss handling uses realized OHLC leg PnL; cross-market open-equity liquidation is a limitation.</p>
    <p><b>Important:</b> Alpha’s published rules prohibit automated/bot trading in applicable policies. This report is research only and is not permission to deploy the repository as an Alpha trading bot.</p>
    <p>Skipped streams: {html.escape('; '.join(f"{row['market']} {row.get('timeframe', '')}: {row['reason']}" for row in skipped) or 'none')}.</p>
    """
    sections = [("Market results", market), ("60/20/20 market validation", validation), ("Portfolio comparison", portfolios), ("Alpha prop results", prop), ("Position sizes", position), ("Economics", economics), ("Realistic trader", realistic), ("Multi-account scaling", multi)]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, border=0)}" for title, frame in sections)
    path.write_text("<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:2em}table{border-collapse:collapse;font-size:10px}th,td{padding:4px 6px;border:1px solid #ddd;white-space:nowrap}th{background:#eee}</style></head><body><h1>Strategy V12 Binance Proxy Prop Simulation</h1>" + text + tables + "</body></html>", encoding="utf-8")


if __name__ == "__main__":
    result = run_v12_binance_proxy_prop_simulation()
    print(json.dumps(result, default=str))
