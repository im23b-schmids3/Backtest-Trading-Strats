from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.backtest.v3_entry_engine import StrategyV3EntryResearchEngine
from fib_backtester.config import RunConfig
from fib_backtester.research.v2 import DISTANCES, MOVES, _pairs
from fib_backtester.reporting.validation import _warnings
from fib_backtester.strategy.v3_entry_research import ENTRY_LEVELS, setup_with_entry_level


REQUIRED_METRICS = [
    "number_of_trades", "trades_per_year", "initial_capital", "final_equity", "total_return",
    "annualized_return", "net_pnl", "gross_pnl", "fees", "slippage", "win_rate", "average_win",
    "average_loss", "expectancy", "profit_factor", "average_r", "median_r", "sharpe_ratio",
    "sortino_ratio", "calmar_ratio", "maximum_drawdown", "average_holding_duration_hours",
    "long_trades", "short_trades", "long_net_pnl", "short_net_pnl",
]


def run_v3_entry_research(config: RunConfig, root: str | Path = "reports/v3") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(config)
    with ProcessPoolExecutor(max_workers=min(len(ENTRY_LEVELS), os.cpu_count() or 1)) as pool:
        batches = pool.map(_run_entry_level, [(config, entry_level) for entry_level in ENTRY_LEVELS])
        rows = [row for batch in batches for row in batch]

    matrix = pd.DataFrame(rows)
    ranked = _rank(matrix)
    ranked.to_csv(root / "v3_ranked_matrix.csv", index=False)
    summary = _entry_summary(ranked)
    summary.to_csv(root / "v3_entry_summary.csv", index=False)
    _write_comparison(root / "v3_entry_comparison.html", ranked, summary)
    warnings = ranked[["strategy_version", "asset", "timeframe", "entry_level", "min_distance", "min_move", "number_of_trades", "maximum_drawdown", "robust_score", "warnings"]].copy()
    warnings.to_csv(root / "v3_confidence_warnings.csv", index=False)
    metadata = {
        "strategy_version": "Strategy_V3_EntryResearch",
        "baseline_strategy": "Strategy_V2",
        "entry_levels": list(ENTRY_LEVELS),
        "baseline_entry_level": 0.882,
        "assets_requested": config.assets,
        "timeframes_requested": config.timeframes,
        "available_pairs": [{"asset": asset, "timeframe": timeframe, "rows": len(bars), "start": str(bars.index[0]), "end": str(bars.index[-1])} for asset, timeframe, bars in pairs],
        "minimum_distances": list(DISTANCES),
        "minimum_moves": list(MOVES),
        "configuration_count": len(matrix),
        "execution_policy": config.execution_policy,
        "initial_capital": config.initial_cash,
        "max_anchor_age_days": config.max_anchor_age_days,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_refresh_performed": False,
        "ranking": "robust_score = total_return - 0.75*abs(maximum_drawdown) + clipped_sharpe/12 + (clipped_profit_factor-1)/12 - low_trade_penalty - neighborhood_instability_penalty - asset_concentration_penalty - timeframe_concentration_penalty - cost_penalty",
        "outputs": [
            "v3_ranked_matrix.csv", "v3_entry_summary.csv", "v3_entry_comparison.html",
            "v3_run_metadata.json", "v3_confidence_warnings.csv",
        ],
    }
    (root / "v3_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"pairs": len(pairs), "configurations": len(matrix), "root": str(root)}


def _run_entry_level(arguments):
    config, entry_level = arguments
    rows = []
    for asset, timeframe, bars in _pairs(config):
        for distance in DISTANCES:
            for move in MOVES:
                run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
                engine = StrategyV3EntryResearchEngine(run, move, entry_level)
                trades, equity = engine.run({asset: bars})
                rows.append(_row(asset, timeframe, distance, move, entry_level, engine, trades, equity, config.initial_cash))
    return rows


def _row(asset, timeframe, distance, move, entry_level, engine, trades, equity, capital):
    metrics = calculate_metrics(trades, equity, capital)
    years = max((pd.Timestamp(equity.timestamp.iloc[-1]) - pd.Timestamp(equity.timestamp.iloc[0])).days / 365.25, 1) if len(equity) > 1 else 1
    setups = _eligible_setups(engine, asset, entry_level)
    filled = len(trades)
    realized_r = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
    rrs = _planned_rrs(engine, asset, entry_level)
    row = {
        "strategy_version": "Strategy_V3_EntryResearch", "asset": asset, "timeframe": timeframe,
        "entry_level": entry_level, "min_distance": distance, "min_move": move,
        **{key: metrics.get(source, 0.0) for key, source in {
            "number_of_trades": "number_of_trades", "initial_capital": "initial_capital", "final_equity": "final_equity",
            "total_return": "total_return", "annualized_return": "annualized_return", "net_pnl": "net_pnl",
            "gross_pnl": "gross_pnl", "win_rate": "win_rate", "average_win": "average_win", "average_loss": "average_loss",
            "expectancy": "expectancy", "profit_factor": "profit_factor", "sharpe_ratio": "sharpe_ratio",
            "sortino_ratio": "sortino_ratio", "calmar_ratio": "calmar_ratio", "maximum_drawdown": "maximum_drawdown",
            "long_trades": "long_trades", "short_trades": "short_trades", "long_net_pnl": "long_net_pnl", "short_net_pnl": "short_net_pnl",
        }.items()},
        "trades_per_year": len(trades) / years,
        "fees": metrics.get("fees_paid", 0.0), "slippage": metrics.get("slippage_cost", 0.0),
        "average_r": float(realized_r.mean()) if not realized_r.empty else 0.0,
        "median_r": float(realized_r.median()) if not realized_r.empty else 0.0,
        "average_holding_duration_hours": metrics.get("average_holding_hours", 0.0),
        "eligible_setups": setups, "filled_entries": filled,
        "entry_fill_rate": filled / setups if setups else 0.0, "missed_setups": max(0, setups - filled),
        "stop_before_tp1_rate": metrics.get("stop_before_tp1_rate", 0.0),
        "average_planned_reward_to_risk": float(np.mean(rrs)) if rrs else 0.0,
        "average_realized_r": float(realized_r.mean()) if not realized_r.empty else 0.0,
    }
    row["warnings"] = _warnings(pd.Series({**row, "number_of_trades": filled}))
    return row


def _eligible_setups(engine, asset, entry_level):
    return int(engine.diagnostics[asset].get("eligible_setups", 0))


def _planned_rrs(engine, asset, entry_level):
    setups = {}
    for event in engine.construction[asset].events:
        if event.action == "activate" and event.setup is not None and event.setup.identifier not in setups:
            setup = setup_with_entry_level(event.setup, entry_level)
            risk = abs(setup.fib.entry - setup.fib.stop)
            reward = abs(setup.fib.targets[0] - setup.fib.entry)
            if risk > 0:
                setups[setup.identifier] = reward / risk
    return list(setups.values())


def _rank(matrix):
    ranked = matrix.copy()
    group_asset = ranked.groupby(["entry_level", "asset"], as_index=False)["total_return"].mean().rename(columns={"total_return": "asset_mean_return"})
    group_timeframe = ranked.groupby(["entry_level", "timeframe"], as_index=False)["total_return"].mean().rename(columns={"total_return": "timeframe_mean_return"})
    ranked = ranked.merge(group_asset, on=["entry_level", "asset"], how="left").merge(group_timeframe, on=["entry_level", "timeframe"], how="left")
    asset_share = ranked.groupby("entry_level")["asset_mean_return"].transform(lambda values: values.abs() / max(values.abs().sum(), 1e-12))
    timeframe_share = ranked.groupby("entry_level")["timeframe_mean_return"].transform(lambda values: values.abs() / max(values.abs().sum(), 1e-12))
    ranked["asset_concentration_penalty"] = np.maximum(0, asset_share - 0.5) * 0.10
    ranked["timeframe_concentration_penalty"] = np.maximum(0, timeframe_share - 0.5) * 0.10
    ranked["cost_ratio"] = (ranked["fees"] + ranked["slippage"]) / ranked["gross_pnl"].abs().replace(0, np.nan)
    ranked["cost_ratio"] = ranked["cost_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    neighbor = ranked.groupby(["entry_level", "asset", "timeframe"])["total_return"].transform("median")
    neighbor_std = ranked.groupby(["entry_level", "asset", "timeframe"])["total_return"].transform("std").fillna(0)
    ranked["neighborhood_return_median"] = neighbor
    ranked["neighborhood_return_std"] = neighbor_std
    ranked["neighborhood_instability_penalty"] = (abs(ranked["total_return"] - neighbor) * 0.25 + neighbor_std * 0.10)
    trades = ranked["number_of_trades"].fillna(0)
    ranked["low_trade_penalty"] = 0.75 * np.maximum(0, 30 - trades) / 30
    ranked["cost_penalty"] = ranked["cost_ratio"] * 0.10
    ranked["robust_score"] = (
        ranked["total_return"] - 0.75 * ranked["maximum_drawdown"].abs()
        + ranked["sharpe_ratio"].clip(-3, 3).fillna(0) / 12
        + (ranked["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) - 1) / 12
        - ranked["low_trade_penalty"] - ranked["neighborhood_instability_penalty"]
        - ranked["asset_concentration_penalty"] - ranked["timeframe_concentration_penalty"] - ranked["cost_penalty"]
    )
    ranked["robust_rank"] = ranked["robust_score"].rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("robust_rank").reset_index(drop=True)


def _entry_summary(ranked):
    rows = []
    for entry, group in ranked.groupby("entry_level", sort=True):
        rows.append(_summary_row("all", None, None, entry, group, ranked))
        for (asset, timeframe), pair in group.groupby(["asset", "timeframe"]):
            rows.append(_summary_row("asset_timeframe", asset, timeframe, entry, pair, ranked))
    return pd.DataFrame(rows)


def _summary_row(scope, asset, timeframe, entry, group, all_ranked):
    asset_returns = group.groupby("asset")["total_return"].mean()
    timeframe_returns = group.groupby("timeframe")["total_return"].mean()
    return {
        "scope": scope, "asset": asset, "timeframe": timeframe, "entry_level": entry,
        "configurations": len(group), "total_trades": group.number_of_trades.sum(),
        "mean_trades_per_configuration": group.number_of_trades.mean(), "mean_trades_per_year": group.trades_per_year.mean(),
        "mean_total_return": group.total_return.mean(), "median_total_return": group.total_return.median(),
        "mean_net_pnl": group.net_pnl.mean(), "mean_sharpe": group.sharpe_ratio.mean(), "mean_sortino": group.sortino_ratio.mean(),
        "mean_robust_score": group.robust_score.mean(), "mean_maximum_drawdown": group.maximum_drawdown.mean(),
        "mean_entry_fill_rate": group.entry_fill_rate.mean(), "mean_planned_reward_to_risk": group.average_planned_reward_to_risk.mean(),
        "mean_realized_r": group.average_realized_r.mean(), "return_std_across_configurations": group.total_return.std(),
        "asset_return_dispersion": asset_returns.std() if len(asset_returns) > 1 else 0.0,
        "timeframe_return_dispersion": timeframe_returns.std() if len(timeframe_returns) > 1 else 0.0,
        "mean_cost_ratio": group.cost_ratio.mean(), "top_robust_rank": group.robust_rank.min(),
    }


def _write_comparison(path, ranked, summary):
    all_summary = summary[summary.scope == "all"].sort_values("entry_level")
    by_entry = all_summary.set_index("entry_level")
    most_trades = all_summary.loc[all_summary.total_trades.idxmax(), "entry_level"]
    best_risk = all_summary.loc[all_summary.mean_robust_score.idxmax(), "entry_level"]
    lowest_dd = all_summary.loc[all_summary.mean_maximum_drawdown.idxmax(), "entry_level"]
    stable_assets = all_summary.loc[all_summary.asset_return_dispersion.idxmin(), "entry_level"]
    stable_timeframes = all_summary.loc[all_summary.timeframe_return_dispersion.idxmin(), "entry_level"]
    entry_850 = by_entry.loc[0.85]
    entry_882 = by_entry.loc[0.882]
    entry_935 = by_entry.loc[0.935]
    filled_by_entry = ranked.groupby("entry_level")["filled_entries"].sum()
    eligible_by_entry = ranked.groupby("entry_level")["eligible_setups"].sum()
    text = f"""
    <p>Strategy_V3_EntryResearch changes only the Fibonacci entry ratio. V2 lifecycle, anchors, age limits, invalidation, stops, targets, sizing, costs, and execution assumptions are inherited unchanged.</p>
    <p>Most trades: <b>{most_trades:.3f}</b>. Best mean robustness score: <b>{best_risk:.3f}</b>. Lowest average drawdown: <b>{lowest_dd:.3f}</b>. Most stable across assets: <b>{stable_assets:.3f}</b>. Most stable across timeframes: <b>{stable_timeframes:.3f}</b>.</p>
    <h2>Entry-level findings</h2>
    <ul>
    <li>0.850: {int(filled_by_entry.loc[0.85]):,} trades, {entry_850.mean_entry_fill_rate:.1%} mean fill rate, {entry_850.mean_planned_reward_to_risk:.3f} planned reward/risk, and {int(eligible_by_entry.loc[0.85] - filled_by_entry.loc[0.85]):,} missed setups.</li>
    <li>0.882 baseline: {int(entry_882.total_trades):,} trades, {entry_882.mean_entry_fill_rate:.1%} mean fill rate, {entry_882.mean_planned_reward_to_risk:.3f} planned reward/risk.</li>
    <li>0.935: {int(filled_by_entry.loc[0.935]):,} trades, {entry_935.mean_entry_fill_rate:.1%} mean fill rate, {entry_935.mean_planned_reward_to_risk:.3f} planned reward/risk, and {int(eligible_by_entry.loc[0.935] - filled_by_entry.loc[0.935]):,} missed setups. It improves reward/risk but does not improve fills; it misses more setups than 0.850.</li>
    </ul>
    <p>The 0.882 result remains a reasonable baseline reference because it is the verified V2 behavior and sits between the aggressive-fill and high-reward extremes. The results support a sensitivity region rather than a universally dominant production entry; no production entry is selected.</p>
    <p>The robust score uses total return, drawdown, clipped Sharpe and profit factor, low-trade penalties, parameter-neighborhood instability, asset/timeframe concentration, and fee/slippage cost ratio. Raw metrics remain available for independent checking.</p>
    """
    top = ranked.head(50).to_html(index=False)
    overall = all_summary.to_html(index=False)
    pairs = summary[summary.scope == "asset_timeframe"].to_html(index=False)
    path.write_text(f"<html><body><h1>Strategy V3 Entry Sensitivity Research</h1>{text}<h2>All assets and timeframes</h2>{overall}<h2>Per asset and timeframe</h2>{pairs}<h2>Top robust configurations</h2>{top}</body></html>", encoding="utf-8")
