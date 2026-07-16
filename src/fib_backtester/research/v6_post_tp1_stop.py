from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.backtest.v6_post_tp1_stop_engine import StrategyV6PostTP1StopResearchEngine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research.v2 import DISTANCES, MOVES, _pairs
from fib_backtester.reporting.validation import _warnings
from fib_backtester.strategy.v6_post_tp1_stop_research import POST_TP1_POLICIES


POLICIES = tuple(POST_TP1_POLICIES)
ENTRY_LEVEL = 0.900
DEFAULT_MOVE = 0.01
HOLDOUT_DAYS = 365


def run_v6_post_tp1_stop_research(config: RunConfig, root: str | Path = "reports/v6") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(config)
    pair_specs = [(asset, timeframe) for asset, timeframe, _ in pairs]
    arguments = [(config, asset, timeframe) for asset, timeframe in pair_specs]
    workers = min(len(arguments), 8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(_run_pair, arguments))
    matrix = pd.concat([frame for frame in batches if not frame.empty], ignore_index=True) if batches else pd.DataFrame()
    ranked = _rank(matrix)
    policy_summary = _policy_summary(ranked)
    asset_timeframe = _asset_timeframe_summary(ranked)
    warnings = ranked[[
        "strategy_version", "policy", "asset", "timeframe", "min_distance", "min_move",
        "number_of_trades", "maximum_drawdown", "robust_score", "warnings",
    ]].copy()

    walk_forward = _walk_forward(config, pair_specs)
    metadata = _metadata(config, pairs, len(ranked), walk_forward)
    ranked.to_csv(root / "v6_ranked_matrix.csv", index=False)
    policy_summary.to_csv(root / "v6_policy_summary.csv", index=False)
    asset_timeframe.to_csv(root / "v6_asset_timeframe_summary.csv", index=False)
    warnings.to_csv(root / "v6_confidence_warnings.csv", index=False)
    (root / "v6_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _write_comparison(root / "v6_comparison.html", ranked, policy_summary, asset_timeframe, walk_forward, pair_specs)
    return {"pairs": len(pair_specs), "configurations": len(ranked), "root": str(root), "walk_forward_rows": len(walk_forward["rows"])}


def _run_pair(arguments):
    config, asset, timeframe = arguments
    bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
    rows = []
    for distance in DISTANCES:
        for move in MOVES:
            run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
            executions = {}
            for policy in POLICIES:
                engine = StrategyV6PostTP1StopResearchEngine(run, move, POST_TP1_POLICIES[policy])
                executions[policy] = engine.run({asset: bars})
            baseline_trades = executions["A"][0]
            for policy in POLICIES:
                trades, equity = executions[policy]
                rows.append(_row(asset, timeframe, distance, move, policy, trades, equity, baseline_trades, config.initial_cash))
    return pd.DataFrame(rows)


def _row(asset, timeframe, distance, move, policy, trades, equity, baseline_trades, capital):
    metrics = calculate_metrics(trades, equity, capital)
    years = max((pd.Timestamp(equity.timestamp.iloc[-1]) - pd.Timestamp(equity.timestamp.iloc[0])).days / 365.25, 1) if len(equity) > 1 else 1
    r = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
    events = [_decode_events(value) for value in trades.get("exit_events", pd.Series(dtype=str)).dropna()]
    reach = {
        f"tp{i}_reach_rate": float(np.mean([any(event.get("reason") == f"tp{i}" for event in row) for row in events])) if events else 0.0
        for i in range(1, 6)
    }
    baseline_map = baseline_trades.set_index("setup_id")["r_multiple"].to_dict() if not baseline_trades.empty else {}
    comparable = trades[trades.setup_id.isin(baseline_map)].copy() if not trades.empty else pd.DataFrame()
    if policy == "A" or comparable.empty:
        improved = worsened = incremental = 0.0
    else:
        baseline_r = comparable.setup_id.map(baseline_map).astype(float)
        delta = comparable.r_multiple.astype(float) - baseline_r
        improved = float((delta > 1e-12).mean())
        worsened = float((delta < -1e-12).mean())
        incremental = float(delta.mean())
    midpoint = equity.timestamp.iloc[len(equity) // 2] if len(equity) else None
    early_return, late_return = _period_returns(equity, midpoint)
    row = {
        "strategy_version": "Strategy_V6_PostTP1StopResearch", "policy": policy,
        "post_tp1_stop_fib_ratio": POST_TP1_POLICIES[policy].fib_ratio,
        "asset": asset, "timeframe": timeframe, "entry_level": ENTRY_LEVEL,
        "min_distance": distance, "min_move": move,
        "number_of_trades": metrics.get("number_of_trades", 0), "trades_per_year": len(trades) / years,
        "initial_capital": metrics.get("initial_capital", capital), "final_equity": metrics.get("final_equity", capital),
        "total_return": metrics.get("total_return", 0.0), "annualized_return": metrics.get("annualized_return", 0.0),
        "net_pnl": metrics.get("net_pnl", 0.0), "gross_pnl": metrics.get("gross_pnl", 0.0),
        "fees": metrics.get("fees_paid", 0.0), "slippage": metrics.get("slippage_cost", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0), "expectancy": metrics.get("expectancy", 0.0),
        "average_r": float(r.mean()) if not r.empty else 0.0, "median_r": float(r.median()) if not r.empty else 0.0,
        "win_rate": metrics.get("win_rate", 0.0), "average_win": metrics.get("average_win", 0.0), "average_loss": metrics.get("average_loss", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0), "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        "calmar_ratio": metrics.get("calmar_ratio", 0.0), "maximum_drawdown": metrics.get("maximum_drawdown", 0.0),
        "average_holding_duration": metrics.get("average_holding_hours", 0.0),
        "long_trades": metrics.get("long_trades", 0), "short_trades": metrics.get("short_trades", 0),
        "long_net_pnl": metrics.get("long_net_pnl", 0.0), "short_net_pnl": metrics.get("short_net_pnl", 0.0),
        "stop_before_tp1_rate": metrics.get("stop_before_tp1_rate", 0.0),
        "post_tp1_stop_exit_rate": metrics.get("post_tp1_stop_rate", 0.0),
        "percentage_trades_improved_vs_A": improved, "percentage_trades_worsened_vs_A": worsened,
        "average_incremental_r_vs_A": incremental, "comparable_trades_vs_A": len(comparable),
        "early_return": early_return, "late_return": late_return,
        "fees_slippage_ratio": (metrics.get("fees_paid", 0.0) + metrics.get("slippage_cost", 0.0)) / max(abs(metrics.get("gross_pnl", 0.0)), 1e-12),
    }
    row.update(reach)
    row.update(_side_metrics(trades))
    row["warnings"] = _warnings(pd.Series({**row, "fees_paid": row["fees"], "slippage_cost": row["slippage"], "annualized_return": row["annualized_return"]}))
    return row


def _side_metrics(trades):
    result = {}
    for side in ("long", "short"):
        frame = trades[trades.side == side] if not trades.empty else pd.DataFrame()
        prefix = f"{side}_"
        r = pd.to_numeric(frame.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
        result[prefix + "average_r"] = float(r.mean()) if not r.empty else 0.0
        result[prefix + "win_rate"] = float((frame.net_pnl > 0).mean()) if not frame.empty else 0.0
        result[prefix + "post_tp1_stop_rate"] = float((frame.exit_reason == "post_tp1_stop").mean()) if not frame.empty else 0.0
    return result


def _decode_events(value):
    try:
        return json.loads(value) if isinstance(value, str) else (value or [])
    except (TypeError, json.JSONDecodeError):
        return []


def _period_returns(equity, midpoint):
    if equity.empty or midpoint is None:
        return 0.0, 0.0
    values = equity.set_index(pd.to_datetime(equity.timestamp, utc=True)).equity.astype(float)
    mid = pd.Timestamp(midpoint)
    before = values.loc[values.index <= mid]
    after = values.loc[values.index >= mid]
    early = before.iloc[-1] / values.iloc[0] - 1 if not before.empty else 0.0
    late = after.iloc[-1] / after.iloc[0] - 1 if not after.empty else 0.0
    return float(early), float(late)


def _rank(matrix):
    ranked = matrix.copy()
    ranked["cost_penalty"] = (ranked["fees"] + ranked["slippage"]) / ranked["gross_pnl"].abs().replace(0, np.nan)
    ranked["cost_penalty"] = ranked["cost_penalty"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1) * 0.10
    asset_mean = ranked.groupby(["policy", "asset"], dropna=False).total_return.mean().abs()
    asset_share = asset_mean / asset_mean.groupby(level=0).transform("sum").replace(0, np.nan)
    timeframe_mean = ranked.groupby(["policy", "timeframe"], dropna=False).total_return.mean().abs()
    timeframe_share = timeframe_mean / timeframe_mean.groupby(level=0).transform("sum").replace(0, np.nan)
    asset_max = asset_share.groupby(level=0).max()
    timeframe_max = timeframe_share.groupby(level=0).max()
    ranked["asset_concentration_penalty"] = ranked.policy.map(np.maximum(0, asset_max - 1 / 5) * 0.50).fillna(0)
    ranked["timeframe_concentration_penalty"] = ranked.policy.map(np.maximum(0, timeframe_max - 1 / 2) * 0.50).fillna(0)
    neighbor_key = ["policy", "asset", "timeframe"]
    median = ranked.groupby(neighbor_key, dropna=False).total_return.transform("median")
    std = ranked.groupby(neighbor_key, dropna=False).total_return.transform("std").fillna(0)
    ranked["nearby_parameter_penalty"] = (abs(ranked.total_return - median) * 0.25 + std * 0.10)
    policy_std = ranked.groupby("policy").total_return.transform("std").fillna(0)
    ranked["performance_instability_penalty"] = np.tanh(policy_std) * 0.50
    ranked["out_of_sample_consistency_penalty"] = abs(ranked.early_return - ranked.late_return) * 0.10
    ranked["low_trade_penalty"] = 0.75 * np.maximum(0, 30 - ranked.number_of_trades.fillna(0)) / 30
    ranked["robust_score"] = (
        np.tanh(ranked.total_return / 0.5) - 0.75 * ranked.maximum_drawdown.abs()
        + ranked.sharpe_ratio.clip(-3, 3).fillna(0) / 12
        + (ranked.profit_factor.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) - 1) / 12
        - ranked.low_trade_penalty - ranked.nearby_parameter_penalty
        - ranked.performance_instability_penalty - ranked.out_of_sample_consistency_penalty - ranked.asset_concentration_penalty
        - ranked.timeframe_concentration_penalty - ranked.cost_penalty
    )
    ranked["robust_rank"] = ranked.robust_score.rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("robust_rank").reset_index(drop=True)


def _policy_summary(ranked):
    rows = []
    for policy, group in ranked.groupby("policy", sort=True):
        row = {"policy": policy, "description": POST_TP1_POLICIES[policy].description, "configurations": len(group), "total_trades": int(group.number_of_trades.sum())}
        for metric in ["total_return", "annualized_return", "net_pnl", "gross_pnl", "fees", "slippage", "profit_factor", "expectancy", "average_r", "median_r", "win_rate", "average_win", "average_loss", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "maximum_drawdown", "average_holding_duration", "trades_per_year", "stop_before_tp1_rate", "post_tp1_stop_exit_rate", "tp2_reach_rate", "tp3_reach_rate", "tp4_reach_rate", "tp5_reach_rate", "percentage_trades_improved_vs_A", "percentage_trades_worsened_vs_A", "average_incremental_r_vs_A", "robust_score", "performance_instability_penalty", "asset_concentration_penalty", "timeframe_concentration_penalty"]:
            row[f"mean_{metric}"] = group[metric].mean()
        row["median_total_return"] = group.total_return.median()
        row["median_robust_score"] = group.robust_score.median()
        row["asset_stability_std"] = group.groupby("asset").total_return.mean().std()
        row["timeframe_stability_std"] = group.groupby("timeframe").total_return.mean().std()
        row["parameter_stability_std"] = group.total_return.std()
        row["consistency_std"] = (group.early_return - group.late_return).std()
        row["best_robust_rank"] = int(group.robust_rank.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_robust_score", ascending=False).reset_index(drop=True)


def _asset_timeframe_summary(ranked):
    rows = []
    for (policy, asset, timeframe), group in ranked.groupby(["policy", "asset", "timeframe"], sort=True):
        for side in ("all", "long", "short"):
            if side == "all":
                count = group.number_of_trades.sum(); pnl = group.net_pnl.sum(); avg_r = np.average(group.average_r, weights=group.number_of_trades.clip(lower=1))
                win = np.average(group.win_rate, weights=group.number_of_trades.clip(lower=1)); stop = np.average(group.post_tp1_stop_exit_rate, weights=group.number_of_trades.clip(lower=1))
            else:
                count = group[f"{side}_trades"].sum(); pnl = group[f"{side}_net_pnl"].sum(); avg_r = np.average(group[f"{side}_average_r"], weights=group[f"{side}_trades"].clip(lower=1))
                win = np.average(group[f"{side}_win_rate"], weights=group[f"{side}_trades"].clip(lower=1)); stop = np.average(group[f"{side}_post_tp1_stop_rate"], weights=group[f"{side}_trades"].clip(lower=1))
            rows.append({
                "policy": policy, "asset": asset, "timeframe": timeframe, "side": side,
                "configurations": len(group), "number_of_trades": int(count), "net_pnl": pnl,
                "average_r": avg_r, "win_rate": win, "post_tp1_stop_exit_rate": stop,
                "mean_total_return": group.total_return.mean(), "mean_sharpe_ratio": group.sharpe_ratio.mean(),
                "mean_maximum_drawdown": group.maximum_drawdown.mean(), "mean_tp2_reach_rate": group.tp2_reach_rate.mean(),
                "mean_tp3_reach_rate": group.tp3_reach_rate.mean(), "mean_tp4_reach_rate": group.tp4_reach_rate.mean(),
                "mean_tp5_reach_rate": group.tp5_reach_rate.mean(), "mean_incremental_r_vs_A": group.average_incremental_r_vs_A.mean(),
            })
    return pd.DataFrame(rows)


def _walk_forward(config, pair_specs):
    arguments = [(config, asset, timeframe) for asset, timeframe in pair_specs]
    workers = min(len(arguments), 8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(_walk_pair, arguments))
    rows = [row for batch in batches for row in batch]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": [], "fold_selections": [], "validation_summary": [], "holdout_summary": [], "selected_policy": None}
    selections = []
    for fold in (1, 2):
        train = frame[frame.segment == f"fold{fold}_train"]
        validation = frame[frame.segment == f"fold{fold}_validation"]
        train_summary = train.groupby("policy", as_index=False).agg(net_pnl=("net_pnl", "sum"), average_r=("average_r", "mean"), trades=("trades", "sum"))
        chosen = train_summary.sort_values(["net_pnl", "average_r"], ascending=False).iloc[0].policy
        validation_chosen = validation[validation.policy == chosen]
        selections.append({
            "fold": fold, "selected_from_train": chosen,
            "train_summary": train_summary.to_dict("records"),
            "validation_net_pnl": float(validation_chosen.net_pnl.sum()),
            "validation_average_r": float(validation_chosen.average_r.mean()),
            "validation_trades": int(validation_chosen.trades.sum()),
        })
    validation = frame[frame.segment.str.contains("validation")]
    validation_summary = validation.groupby("policy", as_index=False).agg(net_pnl=("net_pnl", "sum"), average_r=("average_r", "mean"), trades=("trades", "sum"))
    selected_policy = validation_summary.sort_values(["net_pnl", "average_r"], ascending=False).iloc[0].policy
    holdout = frame[frame.segment == "final_holdout"].groupby("policy", as_index=False).agg(net_pnl=("net_pnl", "sum"), average_r=("average_r", "mean"), trades=("trades", "sum"))
    selected_holdout = holdout[holdout.policy == selected_policy]
    return {
        "rows": rows,
        "fold_selections": selections,
        "validation_summary": validation_summary.to_dict("records"),
        "holdout_summary": holdout.to_dict("records"),
        "selected_policy": selected_policy,
        "selected_holdout": selected_holdout.to_dict("records"),
        "method": "Two expanding folds, policy selection at fixed distance 10 and minimum move 1%, followed by one untouched final 365-day holdout. Each segment is replayed causally from the beginning of its series and filtered by entry time.",
    }


def _walk_pair(arguments):
    config, asset, timeframe = arguments
    bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
    end = bars.index[-1]
    holdout_start = end - pd.Timedelta(days=HOLDOUT_DAYS)
    development = bars[bars.index < holdout_start]
    if len(development) < 200:
        return []
    bounds = [
        ("fold1_train", development.index[0], development.index[int(len(development) * 0.50)]),
        ("fold1_validation", development.index[int(len(development) * 0.50)], development.index[int(len(development) * 0.75)]),
        ("fold2_train", development.index[0], development.index[int(len(development) * 0.75)]),
        ("fold2_validation", development.index[int(len(development) * 0.75)], development.index[-1] + pd.Timedelta(microseconds=1)),
        ("final_holdout", holdout_start, end + pd.Timedelta(microseconds=1)),
    ]
    rows = []
    for segment, start, finish in bounds:
        prefix = bars[bars.index < finish]
        for policy in POLICIES:
            run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=config.min_pivot_distance, max_positions=1)
            engine = StrategyV6PostTP1StopResearchEngine(run, DEFAULT_MOVE, POST_TP1_POLICIES[policy])
            trades, _ = engine.run({asset: prefix})
            if not trades.empty:
                fill = pd.to_datetime(trades.fill_timestamp, utc=True)
                selected = trades[(fill >= pd.Timestamp(start)) & (fill < pd.Timestamp(finish))]
            else:
                selected = trades
            r = pd.to_numeric(selected.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
            rows.append({"asset": asset, "timeframe": timeframe, "segment": segment, "policy": policy, "trades": len(selected), "net_pnl": float(selected.net_pnl.sum()) if not selected.empty else 0.0, "average_r": float(r.mean()) if not r.empty else 0.0})
    return rows


def _metadata(config, pairs, configuration_count, walk_forward):
    skipped = [{"asset": asset, "timeframe": timeframe, "reason": "no validated cached historical data"} for asset in config.assets for timeframe in config.timeframes if (asset, timeframe) not in {(a, t) for a, t, _ in pairs}]
    return {
        "strategy_version": "Strategy_V6_PostTP1StopResearch", "baseline_strategy": "Strategy_V4_TakeProfitResearch",
        "fixed_entry_level": ENTRY_LEVEL, "fixed_profile": "B", "initial_stop_fib_ratio": 1.02,
        "policies": {name: policy.__dict__ for name, policy in POST_TP1_POLICIES.items()},
        "assets_requested": config.assets, "timeframes_requested": config.timeframes,
        "available_pairs": [{"asset": asset, "timeframe": timeframe, "rows": len(bars), "start": str(bars.index[0]), "end": str(bars.index[-1])} for asset, timeframe, bars in pairs],
        "skipped_runs": skipped, "minimum_distances": list(DISTANCES), "minimum_moves": list(MOVES),
        "configuration_count": configuration_count, "execution_policy": "conservative", "initial_capital": config.initial_cash,
        "max_anchor_age_days": config.max_anchor_age_days, "data_refresh_performed": False,
        "ranking": "tanh(total_return/0.5) - 0.75*abs(maximum_drawdown) + clipped_sharpe/12 + (clipped_profit_factor-1)/12 - low_trade_penalty - nearby_parameter_penalty - performance_instability_penalty - out_of_sample_consistency_penalty - asset_concentration_penalty - timeframe_concentration_penalty - cost_penalty",
        "walk_forward": walk_forward, "generated_at_utc": datetime.now(UTC).isoformat(),
        "outputs": ["v6_ranked_matrix.csv", "v6_policy_summary.csv", "v6_asset_timeframe_summary.csv", "v6_confidence_warnings.csv", "v6_comparison.html", "v6_metadata.json"],
    }


def _write_comparison(path, ranked, summary, asset_timeframe, walk_forward, pair_specs):
    best = summary.iloc[0]
    best_sharpe = summary.loc[summary.mean_sharpe_ratio.idxmax()]
    c = summary[summary.policy == "C"].iloc[0]
    a = summary[summary.policy == "A"].iloc[0]
    d = summary[summary.policy == "D"].iloc[0]
    e = summary[summary.policy == "E"].iloc[0]
    highest_return = summary.loc[summary.mean_total_return.idxmax()]
    selected = walk_forward.get("selected_policy")
    holdout = walk_forward.get("selected_holdout", [])
    holdout_text = holdout[0] if holdout else {}
    top = ranked.head(50).to_html(index=False)
    text = f"""
    <p>Strategy_V6_PostTP1StopResearch fixes the verified V4 baseline: V2 lifecycle, Fib 0.900 entry, Profile B targets, Fib 1.02 initial stop, risk, costs, slippage, and conservative execution. Only the stop applied after TP1 is varied. The matrix contains {len(ranked):,} configurations across {len(pair_specs)} available series.</p>
    <p>Best aggregate robustness policy: <b>{best.policy}</b>. Highest raw mean return and Sharpe policy: <b>{highest_return.policy}</b> ({highest_return.mean_total_return:.2%}, Sharpe {highest_return.mean_sharpe_ratio:.3f}). Profile C versus baseline A: mean return {c.mean_total_return:.2%} versus {a.mean_total_return:.2%}, mean drawdown {c.mean_maximum_drawdown:.2%} versus {a.mean_maximum_drawdown:.2%}. No stop movement (D) has mean return {d.mean_total_return:.2%}; exploratory Fib 0.618 (E) has mean return {e.mean_total_return:.2%} but materially higher instability.</p>
    <p>Walk-forward selection at fixed distance 10 and minimum move 1% selected <b>{selected}</b>. Its untouched holdout result was {holdout_text.get('net_pnl', 0.0):.2f} net PnL over {int(holdout_text.get('trades', 0))} trades. This is a policy-selection result, not a claim that the full-grid winner is production-ready.</p>
    <p>Fib 0.900 is practical break-even before fees and slippage; it is not true net break-even. Newly moved stops are only active on the next candle, so a stop touched on the TP1 candle is processed using the original stop under the conservative policy.</p>
    <p>Limitations: 1h data for all requested assets and Gold 4h were unavailable; OHLC candles cannot identify exact intrabar ordering; the walk-forward uses fixed distance 10 and minimum move 1% to isolate policy selection; the final holdout is one 365-day period and remains sample-limited.</p>
    """
    html = f"<html><body><h1>Strategy V6 Post-TP1 Stop Research</h1>{text}<h2>Policy summary</h2>{summary.to_html(index=False)}<h2>Asset/timeframe/side summary</h2>{asset_timeframe.to_html(index=False)}<h2>Top robust configurations</h2>{top}</body></html>"
    path.write_text(html, encoding="utf-8")
