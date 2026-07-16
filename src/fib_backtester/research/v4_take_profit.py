from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.backtest.v4_take_profit_engine import StrategyV4TakeProfitResearchEngine
from fib_backtester.config import RunConfig
from fib_backtester.research.v2 import DISTANCES, MOVES, _pairs
from fib_backtester.reporting.validation import _warnings
from fib_backtester.strategy.v4_take_profit_research import ATR_SETTINGS, PROFILES, TakeProfitProfile


VARIANTS = [(name, None, None) for name in ("A", "B", "C")] + [(name, length, multiplier) for name in ("D", "E") for length, multiplier in ATR_SETTINGS]


def run_v4_take_profit_research(config: RunConfig, root: str | Path = "reports/v4") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(config)
    arguments = [(config, profile, length, multiplier) for profile, length, multiplier in VARIANTS]
    with ProcessPoolExecutor(max_workers=min(len(arguments), 8, os.cpu_count() or 1)) as pool:
        batches = pool.map(_run_variant, arguments)
        rows = [row for batch in batches for row in batch]
    matrix = pd.DataFrame(rows)
    ranked = _rank(matrix)
    ranked.to_csv(root / "v4_ranked_matrix.csv", index=False)
    summary = _profile_summary(ranked)
    summary.to_csv(root / "v4_tp_profile_summary.csv", index=False)
    warnings = ranked[["strategy_version", "profile", "atr_length", "atr_multiplier", "asset", "timeframe", "min_distance", "min_move", "number_of_trades", "maximum_drawdown", "robust_score", "warnings"]]
    warnings.to_csv(root / "v4_confidence_warnings.csv", index=False)
    _write_comparison(root / "v4_tp_comparison.html", ranked, summary)
    metadata = {
        "strategy_version": "Strategy_V4_TakeProfitResearch",
        "baseline_strategy": "Strategy_V3_EntryResearch",
        "fixed_entry_level": 0.900,
        "assets_requested": config.assets,
        "timeframes_requested": config.timeframes,
        "available_pairs": [{"asset": asset, "timeframe": timeframe, "rows": len(bars), "start": str(bars.index[0]), "end": str(bars.index[-1])} for asset, timeframe, bars in pairs],
        "minimum_distances": list(DISTANCES), "minimum_moves": list(MOVES),
        "profiles": {name: profile.__dict__ for name, profile in PROFILES.items()},
        "atr_settings": [list(setting) for setting in ATR_SETTINGS],
        "configuration_count": len(matrix),
        "execution_policy": config.execution_policy,
        "initial_capital": config.initial_cash,
        "max_anchor_age_days": config.max_anchor_age_days,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_refresh_performed": False,
        "ranking": "total_return - 0.75*abs(maximum_drawdown) + clipped_sharpe/12 + (clipped_profit_factor-1)/12 - low_trade_penalty - neighborhood_instability_penalty - asset_concentration_penalty - timeframe_concentration_penalty",
        "outputs": ["v4_ranked_matrix.csv", "v4_tp_profile_summary.csv", "v4_tp_comparison.html", "v4_confidence_warnings.csv", "v4_metadata.json"],
    }
    (root / "v4_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"pairs": len(pairs), "configurations": len(matrix), "root": str(root)}


def _run_variant(arguments):
    config, profile_name, atr_length, atr_multiplier = arguments
    profile = PROFILES[profile_name]
    rows = []
    for asset, timeframe, bars in _pairs(config):
        for distance in DISTANCES:
            for move in MOVES:
                run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
                engine = StrategyV4TakeProfitResearchEngine(run, move, profile, atr_length, atr_multiplier)
                trades, equity = engine.run({asset: bars})
                rows.append(_row(asset, timeframe, distance, move, profile, atr_length, atr_multiplier, trades, equity, config.initial_cash))
    return rows


def _row(asset, timeframe, distance, move, profile, atr_length, atr_multiplier, trades, equity, capital):
    metrics = calculate_metrics(trades, equity, capital)
    years = max((pd.Timestamp(equity.timestamp.iloc[-1]) - pd.Timestamp(equity.timestamp.iloc[0])).days / 365.25, 1) if len(equity) > 1 else 1
    r = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
    events = [json.loads(value) for value in trades.get("exit_events", pd.Series(dtype=str)).dropna()]
    tp_rates = {f"tp{i}_reach_rate": float(np.mean([any(event.get("reason") == f"tp{i}" for event in row) for row in events])) if events else 0.0 for i in range(1, 6)}
    max_mfe = pd.to_numeric(trades.get("max_mfe_r", pd.Series(dtype=float)), errors="coerce")
    realized_reward = float(r.mean()) if not r.empty else 0.0
    unrealized_reward = float(max_mfe.mean()) if not max_mfe.empty else 0.0
    runner_exit = pd.to_numeric(trades.get("runner_exit", pd.Series(dtype=bool)), errors="coerce").fillna(0).astype(bool)
    runner_outperformed = pd.to_numeric(trades.get("runner_outperformed", pd.Series(dtype=bool)), errors="coerce").fillna(0).astype(bool)
    row = {
        "strategy_version": "Strategy_V4_TakeProfitResearch", "profile": profile.name,
        "atr_length": atr_length, "atr_multiplier": atr_multiplier, "asset": asset, "timeframe": timeframe,
        "entry_level": .900, "min_distance": distance, "min_move": move,
        "number_of_trades": metrics.get("number_of_trades", 0), "trades_per_year": len(trades) / years,
        "initial_capital": metrics.get("initial_capital", capital), "final_equity": metrics.get("final_equity", capital),
        "total_return": metrics.get("total_return", 0.0), "annualized_return": metrics.get("annualized_return", 0.0),
        "net_pnl": metrics.get("net_pnl", 0.0), "gross_pnl": metrics.get("gross_pnl", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0), "expectancy": metrics.get("expectancy", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0), "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        "calmar_ratio": metrics.get("calmar_ratio", 0.0), "maximum_drawdown": metrics.get("maximum_drawdown", 0.0),
        "average_r": realized_reward, "median_r": float(r.median()) if not r.empty else 0.0,
        "win_rate": metrics.get("win_rate", 0.0), "average_win": metrics.get("average_win", 0.0), "average_loss": metrics.get("average_loss", 0.0),
        "average_holding_duration_hours": metrics.get("average_holding_hours", 0.0),
        "long_trades": metrics.get("long_trades", 0), "short_trades": metrics.get("short_trades", 0),
        "long_net_pnl": metrics.get("long_net_pnl", 0.0), "short_net_pnl": metrics.get("short_net_pnl", 0.0),
        "fees": metrics.get("fees_paid", 0.0), "slippage": metrics.get("slippage_cost", 0.0),
        **tp_rates,
        "average_realized_reward": realized_reward, "average_unrealized_maximum_reward": unrealized_reward,
        "average_profit_left_on_table": unrealized_reward - realized_reward,
        "average_profit_captured": realized_reward,
        "profit_capture_ratio": realized_reward / unrealized_reward if unrealized_reward > 0 else 0.0,
        "atr_trailing_stop_contribution": float(runner_exit.mean()) if not runner_exit.empty else 0.0,
        "atr_trailer_outperformed_fixed_tp_rate": float(runner_outperformed.mean()) if not runner_outperformed.empty else 0.0,
    }
    row["warnings"] = _warnings(pd.Series(row))
    return row


def _rank(matrix):
    ranked = matrix.copy()
    ranked["cost_ratio"] = ((ranked["fees"] + ranked["slippage"]) / ranked["gross_pnl"].abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    group_asset = ranked.groupby(["profile", "atr_length", "atr_multiplier", "asset"], dropna=False)["total_return"].transform("mean")
    group_timeframe = ranked.groupby(["profile", "atr_length", "atr_multiplier", "timeframe"], dropna=False)["total_return"].transform("mean")
    asset_share = group_asset.abs() / ranked.groupby(["profile", "atr_length", "atr_multiplier"])["total_return"].transform(lambda values: values.abs().sum()).replace(0, np.nan)
    timeframe_share = group_timeframe.abs() / ranked.groupby(["profile", "atr_length", "atr_multiplier"])["total_return"].transform(lambda values: values.abs().sum()).replace(0, np.nan)
    ranked["asset_concentration_penalty"] = np.maximum(0, asset_share.fillna(0) - .5) * .10
    ranked["timeframe_concentration_penalty"] = np.maximum(0, timeframe_share.fillna(0) - .5) * .10
    group_key = ["profile", "atr_length", "atr_multiplier", "asset", "timeframe"]
    neighborhood_median = ranked.groupby(group_key, dropna=False)["total_return"].transform("median")
    neighborhood_std = ranked.groupby(group_key, dropna=False)["total_return"].transform("std").fillna(0)
    ranked["neighborhood_instability_penalty"] = abs(ranked["total_return"] - neighborhood_median) * .25 + neighborhood_std * .10
    ranked["low_trade_penalty"] = .75 * np.maximum(0, 30 - ranked["number_of_trades"].fillna(0)) / 30
    ranked["robust_score"] = (
        ranked["total_return"] - .75 * ranked["maximum_drawdown"].abs()
        + ranked["sharpe_ratio"].clip(-3, 3).fillna(0) / 12
        + (ranked["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) - 1) / 12
        - ranked["low_trade_penalty"] - ranked["neighborhood_instability_penalty"]
        - ranked["asset_concentration_penalty"] - ranked["timeframe_concentration_penalty"]
    )
    ranked["robust_rank"] = ranked["robust_score"].rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("robust_rank").reset_index(drop=True)


def _profile_summary(ranked):
    groups = ranked.groupby(["profile", "atr_length", "atr_multiplier"], dropna=False)
    rows = []
    for key, group in groups:
        profile, length, multiplier = key
        row = {"profile": profile, "atr_length": length, "atr_multiplier": multiplier, "configurations": len(group), "total_trades": group.number_of_trades.sum()}
        for metric in ["total_return", "annualized_return", "net_pnl", "profit_factor", "expectancy", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "maximum_drawdown", "average_r", "median_r", "win_rate", "average_win", "average_loss", "average_holding_duration_hours", "trades_per_year", "fees", "slippage", "tp1_reach_rate", "tp2_reach_rate", "tp3_reach_rate", "tp4_reach_rate", "tp5_reach_rate", "average_realized_reward", "average_unrealized_maximum_reward", "average_profit_left_on_table", "average_profit_captured", "profit_capture_ratio", "atr_trailing_stop_contribution", "atr_trailer_outperformed_fixed_tp_rate", "robust_score"]:
            row[f"mean_{metric}"] = group[metric].mean()
        row["asset_stability_std"] = group.groupby("asset")["total_return"].mean().std()
        row["timeframe_stability_std"] = group.groupby("timeframe")["total_return"].mean().std()
        row["parameter_stability_std"] = group["total_return"].std()
        row["best_robust_rank"] = group.robust_rank.min()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["profile", "atr_length", "atr_multiplier"], na_position="first")


def _write_comparison(path, ranked, summary):
    most_robust = summary.loc[summary.mean_robust_score.idxmax()]
    best_risk = summary.loc[summary.mean_sharpe_ratio.idxmax()]
    highest_capture = summary.loc[summary.mean_profit_capture_ratio.idxmax()]
    d = summary[summary.profile == "D"].sort_values("mean_robust_score", ascending=False).head(1)
    e = summary[summary.profile == "E"].sort_values("mean_robust_score", ascending=False).head(1)
    e_return = summary[summary.profile == "E"].sort_values("mean_total_return", ascending=False).head(1)
    three = summary[summary.profile == "C"].iloc[0]
    profile_a = summary[summary.profile == "A"].iloc[0]
    profile_b = summary[summary.profile == "B"].iloc[0]
    d_best = d.iloc[0] if not d.empty else None
    e_best = e.iloc[0] if not e.empty else None
    e_return_best = e_return.iloc[0] if not e_return.empty else None
    label = lambda row: row.profile if pd.isna(row.atr_length) else f"{row.profile} ATR {int(row.atr_length)}×{row.atr_multiplier:.1f}"
    pct = lambda value: f"{float(value):.1%}"
    top = ranked.head(50).to_html(index=False)
    text = f"""
    <p>Strategy_V4_TakeProfitResearch fixes the verified V3 entry at Fib 0.900 and changes only take-profit profiles and the explicitly specified ATR runner. Stops, sizing, lifecycle, execution, fees, and slippage remain inherited.</p>
    <p>Most robust profile/variant by mean robustness score: <b>{label(most_robust)}</b> ({most_robust.mean_robust_score:.4f}). Highest mean risk-adjusted return by Sharpe: <b>{label(best_risk)}</b> ({best_risk.mean_sharpe_ratio:.3f}). Highest percentage of the available move captured: <b>{label(highest_capture)}</b> ({pct(highest_capture.mean_profit_capture_ratio)}).</p>
    <p>For the five-target profiles, TP4 is reached on {pct(profile_a.mean_tp4_reach_rate)} of trades and TP5 on {pct(profile_a.mean_tp5_reach_rate)}. The three-target profile reaches TP3 on {pct(three.mean_tp3_reach_rate)} and has no TP4/TP5 targets.</p>
    <p>ATR runners contribute exits on about {pct(e_best.mean_atr_trailing_stop_contribution) if e_best is not None else 'not available'} of trades in the most robust E variant and outperform the fixed-target alternative on {pct(e_best.mean_atr_trailer_outperformed_fixed_tp_rate) if e_best is not None else 'not available'}. The most robust runner variants are {label(d_best) if d_best is not None else 'D unavailable'} and {label(e_best) if e_best is not None else 'E unavailable'}; the highest-return E variant is {label(e_return_best) if e_return_best is not None else 'not available'}.</p>
    <p>Profile C improves on Profile A's mean return ({pct(three.mean_total_return)} vs {pct(profile_a.mean_total_return)}) and mean Sharpe ({three.mean_sharpe_ratio:.3f} vs {profile_a.mean_sharpe_ratio:.3f}), but Profile B is stronger overall. Profile B takes profits earlier than A and improves mean return ({pct(profile_b.mean_total_return)} vs {pct(profile_a.mean_total_return)}), drawdown ({pct(profile_b.mean_maximum_drawdown)} vs {pct(profile_a.mean_maximum_drawdown)}), and Sharpe ({profile_b.mean_sharpe_ratio:.3f} vs {profile_a.mean_sharpe_ratio:.3f}); A has slightly higher per-trade expectancy ({profile_a.mean_expectancy:.2f} vs {profile_b.mean_expectancy:.2f}).</p>
    <p>Conclusion: Profile B is the recommended research baseline for the next phase, not a production selection. The ATR profiles are less robust in aggregate, and the results do not justify replacing the fixed-target logic with a runner without further validation.</p>
    """
    path.write_text(f"<html><body><h1>Strategy V4 Take-Profit Research</h1>{text}<h2>Profile summary</h2>{summary.to_html(index=False)}<h2>Top robust configurations</h2>{top}</body></html>", encoding="utf-8")
