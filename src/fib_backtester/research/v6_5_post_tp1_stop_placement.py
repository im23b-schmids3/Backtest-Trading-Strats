from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.backtest.v6_5_post_tp1_stop_placement_engine import StrategyV65PostTP1StopPlacementEngine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.reporting.validation import _warnings
from fib_backtester.strategy.v6_5_post_tp1_stop_placement import STOP_PLACEMENT_ORDER, STOP_PLACEMENT_POLICIES


MIN_TRADES_FOR_SELECTION = 20
SELECTION_POLICY = "C"
BASELINE_POLICY = "Fib 0.880"


def run_v6_5_post_tp1_stop_placement(config: RunConfig, root: str | Path = "reports/v6_5") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selected = _select_v6_configurations(Path("reports/v6/v6_ranked_matrix.csv"))
    arguments = [(config, row) for row in selected]
    workers = min(len(arguments), 8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(_run_selected_configuration, arguments))
    matrix = pd.concat([frame for frame in batches if not frame.empty], ignore_index=True) if batches else pd.DataFrame()
    ranked = _rank(matrix)
    summary = _summary(ranked)
    ranked.to_csv(root / "v6_5_ranked_matrix.csv", index=False)
    summary.to_csv(root / "v6_5_stop_summary.csv", index=False)
    _write_comparison(root / "v6_5_comparison.html", ranked, summary, selected)
    return {"selected_configurations": len(selected), "configurations": len(ranked), "policies": len(STOP_PLACEMENT_ORDER), "root": str(root)}


def _select_v6_configurations(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"V6 ranked matrix is required for selection: {path}")
    v6 = pd.read_csv(path)
    candidates = v6[(v6.policy == SELECTION_POLICY) & (v6.number_of_trades >= MIN_TRADES_FOR_SELECTION) & (v6.asset != "GOLD")].copy()
    if candidates.empty:
        raise ValueError("V6 has no robust candidates meeting the minimum trade threshold")
    candidates = candidates.sort_values(["asset", "timeframe", "robust_score", "robust_rank"], ascending=[True, True, False, True])
    selected = candidates.groupby(["asset", "timeframe"], as_index=False, sort=True).head(1).copy()
    selected["selection_policy"] = SELECTION_POLICY
    selected["selection_minimum_trade_count"] = MIN_TRADES_FOR_SELECTION
    return selected.to_dict("records")


def _run_selected_configuration(arguments):
    config, selected = arguments
    asset, timeframe = selected["asset"], selected["timeframe"]
    bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
    run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=int(selected["min_distance"]), max_positions=1)
    executions = {}
    for policy_name in STOP_PLACEMENT_ORDER:
        engine = StrategyV65PostTP1StopPlacementEngine(run, float(selected["min_move"]), STOP_PLACEMENT_POLICIES[policy_name])
        executions[policy_name] = engine.run({asset: bars})
    baseline = executions[BASELINE_POLICY][0]
    rows = []
    for policy_name in STOP_PLACEMENT_ORDER:
        trades, equity = executions[policy_name]
        rows.append(_row(selected, policy_name, trades, equity, baseline, config.initial_cash))
    return pd.DataFrame(rows)


def _row(selected, policy_name, trades, equity, baseline, capital):
    metrics = calculate_metrics(trades, equity, capital)
    years = max((pd.Timestamp(equity.timestamp.iloc[-1]) - pd.Timestamp(equity.timestamp.iloc[0])).days / 365.25, 1) if len(equity) > 1 else 1
    r = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
    events = [_events(value) for value in trades.get("exit_events", pd.Series(dtype=str)).dropna()]
    reach = {f"tp{i}_reach_rate": float(np.mean([any(event.get("reason") == f"tp{i}" for event in row) for row in events])) if events else 0.0 for i in range(1, 6)}
    baseline_map = baseline.set_index("setup_id")["r_multiple"].to_dict() if not baseline.empty else {}
    comparable = trades[trades.setup_id.isin(baseline_map)].copy() if not trades.empty else pd.DataFrame()
    if policy_name == BASELINE_POLICY or comparable.empty:
        improved = worsened = incremental = 0.0
    else:
        delta = comparable.r_multiple.astype(float) - comparable.setup_id.map(baseline_map).astype(float)
        improved = float((delta > 1e-12).mean())
        worsened = float((delta < -1e-12).mean())
        incremental = float(delta.mean())
    midpoint = equity.timestamp.iloc[len(equity) // 2] if len(equity) else None
    early, late = _period_returns(equity, midpoint)
    row = {
        "strategy_version": "Strategy_V6_5_PostTP1StopPlacement", "stop_policy": policy_name,
        "post_tp1_stop_fib_ratio": STOP_PLACEMENT_POLICIES[policy_name].fib_ratio,
        "asset": selected["asset"], "timeframe": selected["timeframe"], "entry_level": .900,
        "min_distance": int(selected["min_distance"]), "min_move": float(selected["min_move"]),
        "v6_selection_policy": selected["selection_policy"], "v6_selection_robust_rank": selected["robust_rank"],
        "v6_selection_robust_score": selected["robust_score"], "number_of_trades": metrics.get("number_of_trades", 0),
        "trades_per_year": len(trades) / years, "initial_capital": metrics.get("initial_capital", capital),
        "final_equity": metrics.get("final_equity", capital), "total_return": metrics.get("total_return", 0.0),
        "annualized_return": metrics.get("annualized_return", 0.0), "net_pnl": metrics.get("net_pnl", 0.0),
        "gross_pnl": metrics.get("gross_pnl", 0.0), "profit_factor": metrics.get("profit_factor", 0.0),
        "expectancy": metrics.get("expectancy", 0.0), "average_r": float(r.mean()) if not r.empty else 0.0,
        "median_r": float(r.median()) if not r.empty else 0.0, "win_rate": metrics.get("win_rate", 0.0),
        "average_win": metrics.get("average_win", 0.0), "average_loss": metrics.get("average_loss", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0), "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        "calmar_ratio": metrics.get("calmar_ratio", 0.0), "maximum_drawdown": metrics.get("maximum_drawdown", 0.0),
        "average_holding_duration": metrics.get("average_holding_hours", 0.0), "fees": metrics.get("fees_paid", 0.0),
        "slippage": metrics.get("slippage_cost", 0.0), "stop_before_tp1_rate": metrics.get("stop_before_tp1_rate", 0.0),
        "post_tp1_stop_exit_rate": metrics.get("post_tp1_stop_rate", 0.0),
        "percentage_trades_improved_vs_0880": improved, "percentage_trades_worsened_vs_0880": worsened,
        "average_incremental_r_vs_0880": incremental, "comparable_trades_vs_0880": len(comparable),
        "early_return": early, "late_return": late,
    }
    row.update(reach)
    row["warnings"] = _warnings(pd.Series({**row, "fees_paid": row["fees"], "slippage_cost": row["slippage"]}))
    return row


def _events(value):
    try:
        return json.loads(value) if isinstance(value, str) else (value or [])
    except (TypeError, json.JSONDecodeError):
        return []


def _period_returns(equity, midpoint):
    if equity.empty or midpoint is None:
        return 0.0, 0.0
    values = equity.set_index(pd.to_datetime(equity.timestamp, utc=True)).equity.astype(float)
    mid = pd.Timestamp(midpoint)
    early = values.loc[values.index <= mid]
    late = values.loc[values.index >= mid]
    return (float(early.iloc[-1] / values.iloc[0] - 1) if not early.empty else 0.0,
            float(late.iloc[-1] / late.iloc[0] - 1) if not late.empty else 0.0)


def _rank(matrix):
    ranked = matrix.copy()
    config_key = ["asset", "timeframe", "min_distance", "min_move"]
    stop_median = ranked.groupby(config_key, dropna=False).total_return.transform("median")
    stop_std = ranked.groupby(config_key, dropna=False).total_return.transform("std").fillna(0)
    ranked["stop_neighborhood_penalty"] = abs(ranked.total_return - stop_median) * .20 + stop_std * .10
    policy_std = ranked.groupby("stop_policy").total_return.transform("std").fillna(0)
    ranked["placement_instability_penalty"] = np.tanh(policy_std) * .35
    ranked["consistency_penalty"] = np.tanh(abs(ranked.early_return - ranked.late_return)) * .20
    ranked["cost_penalty"] = ((ranked.fees + ranked.slippage) / ranked.gross_pnl.abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1) * .10
    ranked["low_trade_penalty"] = .75 * np.maximum(0, 20 - ranked.number_of_trades.fillna(0)) / 20
    ranked["robust_score"] = (
        np.tanh(ranked.total_return / .5) - .75 * ranked.maximum_drawdown.abs()
        + ranked.sharpe_ratio.clip(-3, 3).fillna(0) / 12
        + (ranked.profit_factor.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) - 1) / 12
        - ranked.stop_neighborhood_penalty - ranked.placement_instability_penalty
        - ranked.consistency_penalty - ranked.cost_penalty - ranked.low_trade_penalty
    )
    ranked["robust_rank"] = ranked.robust_score.rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("robust_rank").reset_index(drop=True)


def _summary(ranked):
    rows = []
    for policy, group in ranked.groupby("stop_policy", sort=False):
        row = {"stop_policy": policy, "post_tp1_stop_fib_ratio": STOP_PLACEMENT_POLICIES[policy].fib_ratio, "description": STOP_PLACEMENT_POLICIES[policy].description, "configurations": len(group), "total_trades": int(group.number_of_trades.sum())}
        metrics = ["total_return", "annualized_return", "net_pnl", "gross_pnl", "profit_factor", "expectancy", "average_r", "median_r", "win_rate", "average_win", "average_loss", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "maximum_drawdown", "average_holding_duration", "trades_per_year", "fees", "slippage", "stop_before_tp1_rate", "post_tp1_stop_exit_rate", "tp2_reach_rate", "tp3_reach_rate", "tp4_reach_rate", "tp5_reach_rate", "percentage_trades_improved_vs_0880", "percentage_trades_worsened_vs_0880", "average_incremental_r_vs_0880", "robust_score"]
        for metric in metrics:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"median_{metric}"] = group[metric].median()
        row["asset_stability_std"] = group.groupby("asset").total_return.mean().std()
        row["timeframe_stability_std"] = group.groupby("timeframe").total_return.mean().std()
        row["stop_level_stability_std"] = group.total_return.std()
        row["mean_stop_neighborhood_penalty"] = group.stop_neighborhood_penalty.mean()
        row["best_robust_rank"] = int(group.robust_rank.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_robust_score", ascending=False).reset_index(drop=True)


def _write_comparison(path, ranked, summary, selected):
    best_robust = summary.iloc[0]
    best_return = summary.loc[summary.mean_total_return.idxmax()]
    baseline = summary[summary.stop_policy == BASELINE_POLICY].iloc[0]
    tp1 = summary[summary.stop_policy == "Fib 0.786"].iloc[0]
    no_move = summary[summary.stop_policy == "no_stop_movement"].iloc[0]
    near = summary[summary.stop_policy.isin(["Fib 0.820", "Fib 0.830", "Fib 0.840", "Fib 0.850", "Fib 0.860", "Fib 0.870", "Fib 0.880", "Fib 0.890"])].sort_values("mean_robust_score", ascending=False)
    region = ", ".join(near.head(3).stop_policy.tolist())
    selection_text = "; ".join(f"{row['asset']} {row['timeframe']} d{int(row['min_distance'])} m{row['min_move']:.4g}" for row in selected)
    text = f"""
    <p>Strategy_V6_5_PostTP1StopPlacement fixes the V6 lifecycle, Fib 0.900 entry, Profile B targets, Fib 1.02 initial stop, sizing, costs, slippage, and conservative execution. Only the stop placement activated after TP1 is varied. The study uses {len(selected)} preselected V6 robust parameter combinations and {len(ranked)} policy/configuration rows.</p>
    <p>Selected V6 combinations: {selection_text}. Selection used V6 policy C, a minimum of 20 trades, one strongest robust cell per available non-Gold asset/timeframe, and excluded Gold because every cached Gold daily cell had only one trade.</p>
    <p>Best robust placement: <b>{best_robust.stop_policy}</b>. Highest mean return: <b>{best_return.stop_policy}</b>. Current Fib 0.880 mean return is {baseline.mean_total_return:.2%}; Fib 0.786 mean return is {tp1.mean_total_return:.2%}; no movement is {no_move.mean_total_return:.2%}. The strongest nearby robust levels are {region}, indicating whether the result forms a region rather than a single point.</p>
    <p>The Fib 0.786 stop is the TP1 price and is therefore the most aggressive lock. Fib 0.900 is only practical break-even before fees and slippage. All moved stops become active only after TP1 has filled; the newly moved stop is not checked again on the TP1 candle.</p>
    <p>This is sensitivity research only. The selected subset is not a replacement for a full-grid revalidation or an independent out-of-sample test.</p>
    """
    html = f"<html><body><h1>V6.5 Post-TP1 Stop Placement Research</h1>{text}<h2>Stop summary</h2>{summary.to_html(index=False)}<h2>Ranked configurations</h2>{ranked.head(100).to_html(index=False)}</body></html>"
    path.write_text(html, encoding="utf-8")
