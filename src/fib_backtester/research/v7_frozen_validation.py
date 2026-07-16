from __future__ import annotations

import html
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.v7_frozen_validation_engine import StrategyV7FrozenValidationEngine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.strategy.v7_frozen_validation import (
    ADVERSE_FILL_EXTRA_SLIPPAGE,
    FROZEN_ENTRY,
    FROZEN_INITIAL_STOP,
    FROZEN_POST_TP1_STOP,
    FROZEN_TP_FRACTIONS,
    FROZEN_TP_RATIOS,
    STABILITY_ENTRY_LEVELS,
    STABILITY_INITIAL_STOPS,
    STABILITY_POST_TP1_STOPS,
    STABILITY_TP_ALLOCATIONS,
    STRESS_SCENARIOS,
    VALIDATION_CANDIDATES,
)


HOLDOUT_START = pd.Timestamp("2025-07-01T00:00:00Z")
EXPANDING_FOLDS = (
    ("2022-01-01T00:00:00Z", "2023-12-31T23:59:59Z", "2024-01-01T00:00:00Z", "2024-09-30T23:59:59Z"),
    ("2022-01-01T00:00:00Z", "2024-09-30T23:59:59Z", "2024-10-01T00:00:00Z", "2025-06-30T23:59:59Z"),
)
ROLLING_FOLDS = (
    ("2022-01-01T00:00:00Z", "2023-12-31T23:59:59Z", "2024-01-01T00:00:00Z", "2024-09-30T23:59:59Z"),
    ("2023-01-01T00:00:00Z", "2024-09-30T23:59:59Z", "2024-10-01T00:00:00Z", "2025-06-30T23:59:59Z"),
)


def run_v7_frozen_validation(config: RunConfig, root: str | Path = "reports/v7") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    available, skipped = _load_data(config)
    payloads = [
        (config, asset, timeframe, bars)
        for (asset, timeframe), bars in available.items()
    ]
    workers = min(max(len(payloads), 1), 8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        bundles = list(pool.map(_run_pair_bundle, payloads)) if payloads else []

    bundles = [bundle for bundle in bundles if bundle.get("error") is None]
    validation = pd.concat([pd.DataFrame(b["walk_forward"]) for b in bundles], ignore_index=True) if bundles else pd.DataFrame()
    holdout = pd.concat([pd.DataFrame(b["holdout"]) for b in bundles], ignore_index=True) if bundles else pd.DataFrame()
    stress = pd.concat([pd.DataFrame(b["stress"]) for b in bundles], ignore_index=True) if bundles else pd.DataFrame()
    stability = pd.concat([pd.DataFrame(b["stability"]) for b in bundles], ignore_index=True) if bundles else pd.DataFrame()
    holdout_trades = [
        (bundle["asset"], bundle["timeframe"], pd.DataFrame(bundle["holdout_trades"]), pd.DataFrame(bundle["bars"]))
        for bundle in bundles
    ]

    cross_asset = _cross_asset_summary(holdout)
    regimes = _regime_summary(holdout_trades, config)
    monte_carlo = _monte_carlo(holdout_trades, config.seed)
    summary = _validation_summary(validation, holdout, stress, skipped, len(payloads))
    warnings = _confidence_warnings(validation, holdout, stress, skipped)

    validation.to_csv(root / "v7_walk_forward.csv", index=False)
    holdout.to_csv(root / "v7_holdout.csv", index=False)
    stress.to_csv(root / "v7_stress_tests.csv", index=False)
    stability.to_csv(root / "v7_parameter_stability.csv", index=False)
    cross_asset.to_csv(root / "v7_cross_asset.csv", index=False)
    regimes.to_csv(root / "v7_regime_analysis.csv", index=False)
    monte_carlo.to_csv(root / "v7_monte_carlo.csv", index=False)
    summary.to_csv(root / "v7_validation_summary.csv", index=False)
    _write_report(root / "v7_final_report.html", summary, validation, holdout, stress, stability, cross_asset, regimes, monte_carlo, skipped)
    return {
        "available_pairs": len(payloads),
        "skipped_pairs": skipped,
        "walk_forward_rows": len(validation),
        "holdout_rows": len(holdout),
        "stress_rows": len(stress),
        "stability_rows": len(stability),
        "monte_carlo_simulations": 10000,
        "root": str(root),
    }


def _load_data(config):
    available, skipped = {}, []
    for asset in config.assets:
        for timeframe in config.timeframes:
            try:
                available[(asset, timeframe)] = Cache().read(
                    asset, timeframe, config.asset_configs[asset].source == "yfinance"
                )
            except Exception as exc:
                skipped.append({"asset": asset, "timeframe": timeframe, "reason": str(exc)})
    return available, skipped


def _pair_config(config, asset, timeframe, distance):
    return replace(
        config,
        assets=[asset],
        timeframes=[timeframe],
        min_pivot_distance=int(distance),
        max_positions=1,
    )


def _run_engine(bars, config, distance, minimum_move, **kwargs):
    run_config = _pair_config(config, config.assets[0], config.timeframes[0], distance)
    engine = StrategyV7FrozenValidationEngine(run_config, float(minimum_move), **kwargs)
    return engine.run({config.assets[0]: bars})


def _run_pair_bundle(payload):
    config, asset, timeframe, bars = payload
    try:
        index = pd.DatetimeIndex(bars.index).tz_convert("UTC")
        bars = bars.copy()
        bars.index = index
        pair_config = _pair_config(config, asset, timeframe, config.min_pivot_distance)
        training_cache = {}
        walk_rows = []

        def select(train_start, train_end):
            key = (str(train_start), str(train_end))
            if key in training_cache:
                return training_cache[key]
            train_bars = _slice(bars, train_start, train_end)
            candidates = []
            for distance, minimum_move in VALIDATION_CANDIDATES:
                trades, equity = _run_engine(train_bars, pair_config, distance, minimum_move)
                metrics = _portfolio_metrics(trades, equity, config.initial_cash)
                candidates.append({
                    "min_distance": distance,
                    "min_move": minimum_move,
                    **metrics,
                    "selection_score": _selection_score(metrics),
                })
            selected = sorted(
                candidates,
                key=lambda row: (row["selection_score"], row["number_of_trades"], row["min_distance"], -row["min_move"]),
                reverse=True,
            )[0]
            training_cache[key] = selected
            return selected

        for stage, folds in (("expanding", EXPANDING_FOLDS), ("rolling", ROLLING_FOLDS)):
            for fold, (train_start, train_end, validation_start, validation_end) in enumerate(folds, 1):
                if _slice(bars, validation_start, validation_end).empty:
                    continue
                selected = select(train_start, train_end)
                evaluation_bars = _slice(bars, train_start, validation_end)
                trades, equity = _run_engine(
                    evaluation_bars, pair_config, selected["min_distance"], selected["min_move"]
                )
                metrics = _period_metrics(trades, equity, config.initial_cash, validation_start, validation_end)
                walk_rows.append({
                    "strategy_version": "Strategy_V7_FrozenValidation",
                    "stage": stage,
                    "fold": fold,
                    "asset": asset,
                    "timeframe": timeframe,
                    "train_start": train_start,
                    "train_end": train_end,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "selected_min_distance": selected["min_distance"],
                    "selected_min_move": selected["min_move"],
                    "training_selection_score": selected["selection_score"],
                    "training_trades": selected["number_of_trades"],
                    **metrics,
                })

        final_training_end = HOLDOUT_START - pd.Timedelta(nanoseconds=1)
        final_selected = select(bars.index[0], final_training_end)
        full_bars = _slice(bars, bars.index[0], bars.index[-1])
        baseline_trades, baseline_equity = _run_engine(
            full_bars, pair_config, final_selected["min_distance"], final_selected["min_move"]
        )
        holdout_metrics = _period_metrics(
            baseline_trades, baseline_equity, config.initial_cash, HOLDOUT_START, bars.index[-1]
        )
        holdout_row = {
            "strategy_version": "Strategy_V7_FrozenValidation",
            "asset": asset,
            "timeframe": timeframe,
            "holdout_start": str(HOLDOUT_START),
            "holdout_end": str(bars.index[-1]),
            "selected_min_distance": final_selected["min_distance"],
            "selected_min_move": final_selected["min_move"],
            "training_selection_score": final_selected["selection_score"],
            "training_trades": final_selected["number_of_trades"],
            **holdout_metrics,
        }

        stress_rows = _stress_rows(
            bars, pair_config, final_selected, config, HOLDOUT_START, asset, timeframe
        )
        stability_rows = _stability_rows(
            bars, pair_config, final_selected, config, HOLDOUT_START, asset, timeframe
        )
        holdout_trade_records = _holdout_trade_records(baseline_trades, HOLDOUT_START)
        return {
            "asset": asset,
            "timeframe": timeframe,
            "bars": bars.reset_index().to_dict("records"),
            "holdout_trades": holdout_trade_records,
            "walk_forward": walk_rows,
            "holdout": [holdout_row],
            "stress": stress_rows,
            "stability": stability_rows,
            "error": None,
        }
    except Exception as exc:
        return {"asset": asset, "timeframe": timeframe, "error": f"{type(exc).__name__}: {exc}"}


def _stress_rows(bars, pair_config, selected, config, holdout_start, asset, timeframe):
    scenarios = {
        "baseline": [{"fee_multiplier": 1.0, "slippage_multiplier": 1.0}],
        "2x_fees": [{"fee_multiplier": 2.0, "slippage_multiplier": 1.0}],
        "3x_fees": [{"fee_multiplier": 3.0, "slippage_multiplier": 1.0}],
        "2x_slippage": [{"fee_multiplier": 1.0, "slippage_multiplier": 2.0}],
        "3x_slippage": [{"fee_multiplier": 1.0, "slippage_multiplier": 3.0}],
        "delayed_execution": [{"delay_bars": 1}],
        "adverse_fills": [{"adverse_fill_extra_slippage": ADVERSE_FILL_EXTRA_SLIPPAGE}],
    }
    rows = []
    for scenario, variants in scenarios.items():
        for variant in variants:
            trades, equity = _run_engine(
                bars, pair_config, selected["min_distance"], selected["min_move"], **variant
            )
            metrics = _period_metrics(trades, equity, config.initial_cash, holdout_start, bars.index[-1])
            rows.append({
                "strategy_version": "Strategy_V7_FrozenValidation",
                "asset": asset,
                "timeframe": timeframe,
                "stress_scenario": scenario,
                "replications": 1,
                "selected_min_distance": selected["min_distance"],
                "selected_min_move": selected["min_move"],
                **metrics,
            })

    for scenario, probability in (("missed_fills_5pct", .05), ("missed_fills_10pct", .10)):
        repetitions = []
        for replication in range(10):
            seed = config.seed + _stable_seed_offset(asset, timeframe, scenario) + replication
            trades, equity = _run_engine(
                bars,
                pair_config,
                selected["min_distance"],
                selected["min_move"],
                missed_fill_probability=probability,
                random_seed=seed,
            )
            repetitions.append(_period_metrics(trades, equity, config.initial_cash, holdout_start, bars.index[-1]))
        rows.append({
            "strategy_version": "Strategy_V7_FrozenValidation",
            "asset": asset,
            "timeframe": timeframe,
            "stress_scenario": scenario,
            "replications": len(repetitions),
            "selected_min_distance": selected["min_distance"],
            "selected_min_move": selected["min_move"],
            **_average_metric_rows(repetitions),
        })
    return rows


def _stability_rows(bars, pair_config, selected, config, holdout_start, asset, timeframe):
    variants = []
    for value in STABILITY_ENTRY_LEVELS:
        variants.append(("entry", f"{value:.2f}", {"entry_level": value}))
    for value in STABILITY_POST_TP1_STOPS:
        variants.append(("post_tp1_stop", f"{value:.2f}", {"post_tp1_stop": value}))
    for label, fractions in STABILITY_TP_ALLOCATIONS.items():
        variants.append(("tp_allocation", label, {"tp_fractions": fractions}))
    for value in STABILITY_INITIAL_STOPS:
        variants.append(("initial_stop", f"{value:.2f}", {"initial_stop": value}))

    rows = []
    for factor, value, kwargs in variants:
        trades, equity = _run_engine(
            bars, pair_config, selected["min_distance"], selected["min_move"], **kwargs
        )
        metrics = _period_metrics(trades, equity, config.initial_cash, holdout_start, bars.index[-1])
        rows.append({
            "strategy_version": "Strategy_V7_FrozenValidation",
            "asset": asset,
            "timeframe": timeframe,
            "factor": factor,
            "tested_value": value,
            "one_factor_at_a_time": True,
            "frozen_entry": FROZEN_ENTRY,
            "frozen_post_tp1_stop": FROZEN_POST_TP1_STOP,
            "frozen_initial_stop": FROZEN_INITIAL_STOP,
            "frozen_tp_allocation": "/".join(str(int(v * 100)) for v in FROZEN_TP_FRACTIONS),
            "selected_min_distance": selected["min_distance"],
            "selected_min_move": selected["min_move"],
            **metrics,
        })
    return rows


def _selection_score(metrics):
    # Training-only selection score.  It is deliberately conservative and is
    # never computed from validation or holdout observations.
    trades = metrics["number_of_trades"]
    return (
        float(metrics["total_return"])
        - 0.75 * abs(float(metrics["maximum_drawdown"]))
        + 0.05 * np.clip(float(metrics["sharpe_ratio"]), -3, 3)
        + 0.03 * np.clip(float(metrics["profit_factor"] - 1.0), -2, 2)
        - 0.50 * max(0, 20 - trades) / 20
    )


def _portfolio_metrics(trades, equity, initial_cash):
    return _period_metrics(trades, equity, initial_cash, None, None)


def _period_metrics(trades, equity, initial_cash, start, end):
    trades = trades.copy() if trades is not None else pd.DataFrame()
    equity = equity.copy() if equity is not None else pd.DataFrame()
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity = equity.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if start is not None and not equity.empty:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        equity = equity[(equity.timestamp >= start) & (equity.timestamp <= end)]
    if not equity.empty:
        base_equity = float(equity.equity.iloc[0])
        final_equity = float(equity.equity.iloc[-1])
        curve = equity.equity.astype(float)
    else:
        base_equity = float(initial_cash)
        final_equity = base_equity
        curve = pd.Series(dtype=float)

    if not trades.empty and "fill_timestamp" in trades:
        fills = pd.to_datetime(trades.fill_timestamp, utc=True)
        if start is not None:
            trades = trades[(fills >= pd.Timestamp(start)) & (fills <= pd.Timestamp(end))].copy()
    n = len(trades)
    net = float(trades.net_pnl.sum()) if n and "net_pnl" in trades else 0.0
    gross = float(trades.gross_pnl.sum()) if n and "gross_pnl" in trades else 0.0
    fees = float(trades.fees.sum()) if n and "fees" in trades else 0.0
    slippage = float(trades.slippage_cost.sum()) if n and "slippage_cost" in trades else 0.0
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"] if n else pd.Series(dtype=float)
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"] if n else pd.Series(dtype=float)
    returns = curve.pct_change().dropna() if len(curve) > 1 else pd.Series(dtype=float)
    downside = returns[returns < 0]
    periods = _periods_per_year(equity)
    sharpe = periods**0.5 * returns.mean() / returns.std(ddof=0) if len(returns) and returns.std(ddof=0) > 0 else 0.0
    sortino = periods**0.5 * returns.mean() / downside.std(ddof=0) if len(downside) and downside.std(ddof=0) > 0 else 0.0
    drawdown = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    if len(equity) > 1:
        days = max((equity.timestamp.iloc[-1] - equity.timestamp.iloc[0]).total_seconds() / 86400, 1 / 24)
    else:
        days = 1.0
    total_return = final_equity / base_equity - 1 if base_equity else 0.0
    annualized = (1 + total_return) ** (365.25 / days) - 1 if 1 + total_return > 0 else -1.0
    r_values = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "number_of_trades": int(n),
        "trades_per_year": float(n * 365.25 / days),
        "initial_capital": base_equity,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "net_pnl": net,
        "gross_pnl": gross,
        "fees": fees,
        "slippage": slippage,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "expectancy": float(trades.net_pnl.mean()) if n else 0.0,
        "average_r": float(r_values.mean()) if len(r_values) else 0.0,
        "median_r": float(r_values.median()) if len(r_values) else 0.0,
        "win_rate": float((trades.net_pnl > 0).mean()) if n else 0.0,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(annualized / abs(max_dd)) if max_dd < 0 else 0.0,
        "maximum_drawdown": max_dd,
        "average_holding_hours": float(trades.holding_hours.mean()) if n and "holding_hours" in trades else 0.0,
        "long_trades": int((trades.side == "long").sum()) if n else 0,
        "short_trades": int((trades.side == "short").sum()) if n else 0,
        "profitable": bool(total_return > 0),
    }


def _periods_per_year(equity):
    if len(equity) < 2:
        return 365.25
    step = pd.Series(equity.timestamp).diff().dropna().dt.total_seconds().median()
    return 365.25 * 86400 / step if step and step > 0 else 365.25


def _slice(bars, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return bars[(bars.index >= start) & (bars.index <= end)]


def _holdout_trade_records(trades, start):
    if trades is None or trades.empty:
        return []
    fills = pd.to_datetime(trades.fill_timestamp, utc=True)
    return trades.loc[fills >= pd.Timestamp(start)].to_dict("records")


def _average_metric_rows(rows):
    if not rows:
        return {}
    numeric = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float, np.number))]
        if values:
            numeric[key] = float(np.mean(values))
    return numeric


def _cross_asset_summary(holdout):
    if holdout.empty:
        return pd.DataFrame()
    metrics = ["number_of_trades", "total_return", "annualized_return", "net_pnl", "profit_factor", "win_rate", "sharpe_ratio", "sortino_ratio", "maximum_drawdown", "average_r"]
    rows = []
    for (asset, timeframe), group in holdout.groupby(["asset", "timeframe"], sort=True):
        row = {"asset": asset, "timeframe": timeframe, "configurations": len(group)}
        row.update({f"mean_{metric}": float(group[metric].mean()) for metric in metrics})
        row["profitable"] = bool((group.total_return > 0).all())
        rows.append(row)
    for asset, group in holdout.groupby("asset", sort=True):
        row = {"asset": asset, "timeframe": "all_available", "configurations": len(group)}
        row.update({f"mean_{metric}": float(group[metric].mean()) for metric in metrics})
        row["profitable"] = bool(group.total_return.mean() > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _regime_summary(holdout_trades, config):
    records = []
    for asset, timeframe, trades, bars_records in holdout_trades:
        if trades.empty:
            continue
        bars = pd.DataFrame(bars_records)
        index_column = "index" if "index" in bars.columns else next(
            column for column in bars.columns if column not in {"open", "high", "low", "close", "volume"}
        )
        bars["timestamp"] = pd.to_datetime(bars[index_column], utc=True)
        bars = bars.set_index("timestamp").sort_index()
        trend, volatility = _regime_series(bars.close)
        fills = pd.to_datetime(trades.fill_timestamp, utc=True)
        for record, fill in zip(trades.to_dict("records"), fills):
            position = bars.index.searchsorted(fill, side="right") - 1
            if position < 0:
                continue
            timestamp = bars.index[position]
            record = dict(record)
            record["asset"] = asset
            record["timeframe"] = timeframe
            record["trend_regime"] = trend.iloc[position]
            record["volatility_regime"] = volatility.iloc[position]
            records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    rows = []
    for label_column, labels in (("trend_regime", ("bullish", "bearish", "sideways")), ("volatility_regime", ("high_volatility", "low_volatility"))):
        for label in labels:
            group = frame[frame[label_column] == label]
            rows.append({
                "regime_dimension": label_column,
                "regime": label,
                "asset": "all",
                "timeframe": "all",
                "trades": len(group),
                "net_pnl": float(group.net_pnl.sum()) if not group.empty else 0.0,
                "average_r": float(pd.to_numeric(group.r_multiple, errors="coerce").mean()) if not group.empty else 0.0,
                "win_rate": float((group.net_pnl > 0).mean()) if not group.empty else 0.0,
                "profitable": bool(group.net_pnl.sum() > 0) if not group.empty else False,
                "definition": "trend: close versus 200-bar SMA and its 20-bar slope; volatility: 20-bar return standard deviation split at each series median",
            })
            for (asset, timeframe), subgroup in group.groupby(["asset", "timeframe"], sort=True):
                rows.append({
                    "regime_dimension": label_column,
                    "regime": label,
                    "asset": asset,
                    "timeframe": timeframe,
                    "trades": len(subgroup),
                    "net_pnl": float(subgroup.net_pnl.sum()),
                    "average_r": float(pd.to_numeric(subgroup.r_multiple, errors="coerce").mean()),
                    "win_rate": float((subgroup.net_pnl > 0).mean()),
                    "profitable": bool(subgroup.net_pnl.sum() > 0),
                    "definition": "trend: close versus 200-bar SMA and its 20-bar slope; volatility: 20-bar return standard deviation split at each series median",
                })
    return pd.DataFrame(rows)


def _regime_series(close):
    sma = close.rolling(200, min_periods=100).mean()
    slope = sma.diff(20)
    trend = pd.Series("sideways", index=close.index)
    trend[(close > sma) & (slope > 0)] = "bullish"
    trend[(close < sma) & (slope < 0)] = "bearish"
    volatility = close.pct_change().rolling(20, min_periods=10).std()
    threshold = volatility.median()
    vol_label = pd.Series(np.where(volatility >= threshold, "high_volatility", "low_volatility"), index=close.index)
    return trend, vol_label


def _monte_carlo(holdout_trades, seed):
    r_values = []
    for _, _, trades, _ in holdout_trades:
        if not trades.empty:
            r_values.extend(pd.to_numeric(trades.get("r_multiple"), errors="coerce").dropna().tolist())
    rng = np.random.default_rng(seed)
    simulations = 10_000
    if not r_values:
        return pd.DataFrame([{"metric": "unavailable", "estimate": np.nan, "simulations": simulations, "seed": seed, "source": "holdout trade R multiples"}])
    r_values = np.asarray(r_values, dtype=float)
    mean_trades_year = max(1, int(round(len(r_values) / max(_holdout_years(holdout_trades), 1))))
    annual_returns = np.empty(simulations)
    drawdowns = np.empty(simulations)
    for i in range(simulations):
        count = max(1, int(rng.poisson(mean_trades_year)))
        sampled = rng.choice(r_values, size=count, replace=True)
        path = np.cumprod(1 + 0.02 * sampled)
        annual_returns[i] = path[-1] - 1
        drawdowns[i] = np.min(path / np.maximum.accumulate(path) - 1)
    rows = []
    for metric, values, estimate in (
        ("annual_return", annual_returns, float(annual_returns.mean())),
        ("maximum_drawdown", drawdowns, float(drawdowns.mean())),
    ):
        rows.append({
            "metric": metric,
            "estimate": estimate,
            "p025": float(np.quantile(values, .025)),
            "p05": float(np.quantile(values, .05)),
            "p50": float(np.quantile(values, .50)),
            "p95": float(np.quantile(values, .95)),
            "p975": float(np.quantile(values, .975)),
            "simulations": simulations,
            "seed": seed,
            "source": "bootstrap of holdout trade R multiples; 2% risk per trade; Poisson trade count",
        })
    for threshold in (.10, .20, .30):
        rows.append({
            "metric": f"probability_drawdown_over_{int(threshold * 100)}pct",
            "estimate": float((drawdowns <= -threshold).mean()),
            "p025": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p975": np.nan,
            "simulations": simulations,
            "seed": seed,
            "source": "bootstrap of holdout trade R multiples; 2% risk per trade; Poisson trade count",
        })
    rows.append({
        "metric": "probability_losing_year",
        "estimate": float((annual_returns < 0).mean()),
        "p025": np.nan,
        "p05": np.nan,
        "p50": np.nan,
        "p95": np.nan,
        "p975": np.nan,
        "simulations": simulations,
        "seed": seed,
        "source": "bootstrap of holdout trade R multiples; 2% risk per trade; Poisson trade count",
    })
    return pd.DataFrame(rows)


def _holdout_years(holdout_trades):
    dates = []
    for _, _, trades, _ in holdout_trades:
        if not trades.empty:
            dates.extend(pd.to_datetime(trades.fill_timestamp, utc=True).tolist())
    if len(dates) < 2:
        return 1.0
    return max((max(dates) - min(dates)).total_seconds() / (365.25 * 86400), 1.0)


def _validation_summary(validation, holdout, stress, skipped, available_pairs):
    rows = []
    for scope, frame in (("expanding_walk_forward", validation[validation.stage == "expanding"] if not validation.empty else pd.DataFrame()), ("rolling_walk_forward", validation[validation.stage == "rolling"] if not validation.empty else pd.DataFrame()), ("untouched_holdout", holdout), ("stress_tests", stress)):
        if frame.empty:
            rows.append({"scope": scope, "rows": 0, "profitable_fraction": 0.0, "status": "no_results"})
            continue
        rows.append({
            "scope": scope,
            "rows": len(frame),
            "profitable_fraction": float(frame.profitable.mean()),
            "mean_total_return": float(frame.total_return.mean()),
            "median_total_return": float(frame.total_return.median()),
            "mean_annualized_return": float(frame.annualized_return.mean()),
            "mean_net_pnl": float(frame.net_pnl.mean()),
            "mean_profit_factor": float(frame.profit_factor.mean()),
            "mean_sharpe_ratio": float(frame.sharpe_ratio.mean()),
            "mean_sortino_ratio": float(frame.sortino_ratio.mean()),
            "mean_maximum_drawdown": float(frame.maximum_drawdown.mean()),
            "mean_trades_per_year": float(frame.trades_per_year.mean()),
            "total_trades": int(frame.number_of_trades.sum()),
            "status": "profitable" if frame.total_return.mean() > 0 else "unprofitable",
        })
    rows.append({
        "scope": "data_availability",
        "rows": available_pairs,
        "skipped_pairs": len(skipped),
        "status": "complete" if not skipped else "partial_cached_universe",
        "limitations": "; ".join(f"{x['asset']} {x['timeframe']}: {x['reason']}" for x in skipped),
    })
    return pd.DataFrame(rows)


def _confidence_warnings(validation, holdout, stress, skipped):
    warnings = []
    if skipped:
        warnings.append({"category": "data", "warning": "Unavailable cached pairs were not substituted", "detail": "; ".join(f"{x['asset']} {x['timeframe']}" for x in skipped)})
    for name, frame in (("walk_forward", validation), ("holdout", holdout), ("stress", stress)):
        if frame.empty:
            continue
        if frame.number_of_trades.min() < 20:
            warnings.append({"category": "sample_size", "warning": f"{name} contains cells with fewer than 20 trades", "detail": str(int(frame.number_of_trades.min()))})
        if frame.total_return.std() > .50:
            warnings.append({"category": "instability", "warning": f"{name} has high cross-cell return dispersion", "detail": f"std={frame.total_return.std():.4f}"})
    warnings.extend([
        {"category": "execution", "warning": "OHLC data cannot identify exact intrabar order; conservative stop precedence remains in force", "detail": "See docs/assumptions.md"},
        {"category": "monte_carlo", "warning": "Monte Carlo is a bootstrap scenario, not an independent market simulation", "detail": "It resamples holdout trade R multiples"},
    ])
    return pd.DataFrame(warnings)


def _write_report(path, summary, validation, holdout, stress, stability, cross_asset, regimes, monte_carlo, skipped):
    classification = _classification(summary, holdout, stress, stability, cross_asset)
    holdout_return = float(holdout.total_return.mean()) if not holdout.empty else np.nan
    holdout_profitable = bool(holdout.total_return.mean() > 0) if not holdout.empty else False
    losing_year = monte_carlo.loc[monte_carlo.metric == "probability_losing_year", "estimate"]
    annual = monte_carlo[monte_carlo.metric == "annual_return"]
    dd = monte_carlo[monte_carlo.metric == "maximum_drawdown"]
    questions = f"""
    <ol>
      <li>Expanding walk-forward: {_answer(summary, 'expanding_walk_forward')}.</li>
      <li>Rolling walk-forward: {_answer(summary, 'rolling_walk_forward')}.</li>
      <li>Untouched holdout: {'profitable' if holdout_profitable else 'not profitable'}; mean return {holdout_return:.2%}.</li>
      <li>Worse execution: {_stress_answer(stress)}.</li>
      <li>Nearby-parameter stability: {_stability_answer(stability)}.</li>
      <li>Asset concentration: {_asset_answer(cross_asset)}.</li>
      <li>Monte Carlo annual return: {_mc_range(annual)}.</li>
      <li>Monte Carlo drawdown: {_mc_range(dd)}; losing-year probability {_mc_value(losing_year)}.</li>
      <li>Paper trading readiness: {'not yet; evidence remains insufficient' if classification != 'reasonably robust historical evidence' else 'yes, with paper-only monitoring first'}.</li>
    </ol>"""
    sections = [
        ("Validation summary", summary),
        ("Walk-forward results", validation),
        ("Untouched holdout", holdout),
        ("Stress tests", stress),
        ("Parameter stability", stability),
        ("Cross asset", cross_asset),
        ("Regime analysis", regimes),
        ("Monte Carlo", monte_carlo),
    ]
    tables = "".join(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False)}" for title, frame in sections)
    skipped_text = "; ".join(f"{x['asset']} {x['timeframe']}: {x['reason']}" for x in skipped) or "None"
    text = f"""
    <p>Strategy_V7_FrozenValidation uses the unchanged V2 lifecycle, Fib 0.900 entry, Profile B, Fib 1.02 initial stop, Fib 0.820 post-TP1 stop, conservative OHLC execution, existing fees/slippage, anchor-age rules, and risk model. No production trading rule was optimized.</p>
    <p><b>Final classification: {html.escape(classification)}</b></p>
    <p>Distance and minimum-move selection occurred only inside each training window. The final holdout began {HOLDOUT_START} and was not used for selection. Stress definitions, regime definitions, and Monte Carlo limitations are recorded in the tables below.</p>
    <h2>Decision questions</h2>{questions}
    <p>Skipped cached pairs: {html.escape(skipped_text)}.</p>
    """
    path.write_text(f"<html><body><h1>V7 Frozen Validation</h1>{text}{tables}</body></html>", encoding="utf-8")


def _classification(summary, holdout, stress, stability, cross_asset):
    if holdout.empty or stress.empty or stability.empty:
        return "no evidence of an edge"
    expanding = _summary_value(summary, "expanding_walk_forward", "profitable_fraction")
    rolling = _summary_value(summary, "rolling_walk_forward", "profitable_fraction")
    holdout_return = float(holdout.total_return.mean())
    adverse = stress[stress.stress_scenario.isin(["2x_fees", "3x_fees", "2x_slippage", "3x_slippage", "missed_fills_10pct", "delayed_execution", "adverse_fills"])]
    adverse_profitable = float(adverse.profitable.mean()) if not adverse.empty else 0.0
    stability_return = float(stability.total_return.std()) if not stability.empty else np.inf
    asset_returns = cross_asset[cross_asset.timeframe == "all_available"].mean_total_return if not cross_asset.empty else pd.Series(dtype=float)
    concentration = bool(len(asset_returns) and (asset_returns > 0).sum() <= max(1, len(asset_returns) // 2))
    if expanding >= .75 and rolling >= .75 and holdout_return > 0 and adverse_profitable >= .50 and stability_return < .50 and not concentration:
        return "reasonably robust historical evidence"
    if holdout_return > 0 and (expanding > .40 or rolling > .40):
        return "promising but insufficient evidence"
    if expanding > .25 or rolling > .25 or holdout_return > 0:
        return "weak and unstable evidence"
    return "no evidence of an edge"


def _summary_value(summary, scope, column):
    rows = summary[summary.scope == scope]
    return float(rows.iloc[0].get(column, 0.0)) if not rows.empty else 0.0


def _answer(summary, scope):
    value = _summary_value(summary, scope, "profitable_fraction")
    return f"{value:.1%} of reported folds/cells were profitable"


def _stress_answer(stress):
    if stress.empty:
        return "not evaluated"
    adverse = stress[~stress.stress_scenario.isin(["baseline"])]
    return f"{adverse.profitable.mean():.1%} of stress cells remained profitable"


def _stability_answer(stability):
    if stability.empty:
        return "not evaluated"
    return f"return standard deviation across one-factor perturbations was {stability.total_return.std():.2%}"


def _asset_answer(cross_asset):
    if cross_asset.empty:
        return "not evaluated"
    aggregate = cross_asset[cross_asset.timeframe == "all_available"]
    return f"{int((aggregate.mean_total_return > 0).sum())}/{len(aggregate)} assets had positive mean holdout return"


def _mc_range(frame):
    if frame.empty:
        return "unavailable"
    row = frame.iloc[0]
    return f"mean {row.estimate:.2%}, 5th–95th percentile {row.p05:.2%} to {row.p95:.2%}"


def _mc_value(series):
    return f"{float(series.iloc[0]):.2%}" if len(series) else "unavailable"


def _stable_seed_offset(asset, timeframe, scenario):
    return sum(ord(char) for char in f"{asset}:{timeframe}:{scenario}")
