from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_matrix_tables(rows: list[dict], output: str | Path = "reports") -> pd.DataFrame:
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    if results.empty:
        results.to_csv(root / "result_matrix.csv", index=False)
        return results
    results["warnings"] = results.apply(_warnings, axis=1)
    results["composite_score"] = results.apply(_score, axis=1)
    results = results.sort_values("composite_score", ascending=False)
    results.to_csv(root / "result_matrix.csv", index=False)
    for column, name in (("asset", "by_asset"), ("timeframe", "by_timeframe"), ("swing_n", "by_swing_n"), ("min_pivot_distance", "by_min_distance")):
        if column in results:
            _group(results, column).to_csv(root / f"{name}.csv", index=False)
    results[[c for c in ("run_id", "asset", "timeframe", "long_net_pnl", "short_net_pnl", "long_trades", "short_trades") if c in results]].to_csv(root / "long_vs_short.csv", index=False)
    results[[c for c in ("run_id", "asset", "timeframe", "gross_pnl", "net_pnl", "fees_paid", "slippage_cost") if c in results]].to_csv(root / "gross_vs_net.csv", index=False)
    results[[c for c in ("run_id", "asset", "timeframe", "total_return", "buy_and_hold_return", "trend_filter_return", "composite_score", "warnings") if c in results]].to_csv(root / "strategy_vs_benchmarks.csv", index=False)
    results[[c for c in ("run_id", "asset", "timeframe", "number_of_trades", "sharpe_ratio", "maximum_drawdown", "warnings") if c in results]].to_csv(root / "confidence_warnings.csv", index=False)
    return results


def _group(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    values = [column for column in ("total_return", "maximum_drawdown", "sharpe_ratio", "profit_factor", "number_of_trades", "composite_score") if column in frame]
    return frame.groupby(key, dropna=False)[values].mean().reset_index().sort_values("composite_score", ascending=False)


def _score(row: pd.Series) -> float:
    net = float(row.get("total_return", 0) or 0)
    drawdown = abs(float(row.get("maximum_drawdown", 0) or 0))
    sharpe = min(max(float(row.get("sharpe_ratio", 0) or 0), -3), 3) / 10
    factor = min(float(row.get("profit_factor", 0) or 0), 3) / 10
    trade_count = float(row.get("number_of_trades", 0) or 0)
    penalty = max(0, 20 - trade_count) / 20 * 0.5
    return net - 0.5 * drawdown + sharpe + factor - penalty


def _warnings(row: pd.Series) -> str:
    flags: list[str] = []
    trades = float(row.get("number_of_trades", 0) or 0)
    if trades < 20: flags.append("fewer_than_20_trades")
    if float(row.get("sharpe_ratio", 0) or 0) > 3: flags.append("suspicious_sharpe")
    if trades and float(row.get("win_rate", 0) or 0) == 1: flags.append("zero_losing_trades")
    if trades >= 20 and abs(float(row.get("maximum_drawdown", 0) or 0)) < .01: flags.append("implausibly_low_drawdown")
    if float(row.get("annualized_return", 0) or 0) > 2: flags.append("unusually_high_annualized_return")
    if float(row.get("fees_paid", 0) or 0) > max(float(row.get("gross_pnl", 0) or 0), 0): flags.append("fees_exceed_gross_profit")
    return ";".join(flags)
