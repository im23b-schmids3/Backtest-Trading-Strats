from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_cash: float) -> dict:
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else initial_cash
    returns = equity.set_index("timestamp")["equity"].pct_change().dropna() if not equity.empty else pd.Series(dtype=float)
    drawdown = equity["equity"] / equity["equity"].cummax() - 1 if not equity.empty else pd.Series(dtype=float)
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    span_days = (pd.Timestamp(equity.timestamp.iloc[-1]) - pd.Timestamp(equity.timestamp.iloc[0])).days if len(equity) > 1 else 0
    annualized = (final_equity / initial_cash) ** (365 / span_days) - 1 if span_days >= 30 and final_equity > 0 else None
    downside = returns[returns < 0]
    periods_per_year = _periods_per_year(equity)
    sharpe = np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0) if returns.std(ddof=0) > 0 else None
    sortino = np.sqrt(periods_per_year) * returns.mean() / downside.std(ddof=0) if downside.std(ddof=0) > 0 else None
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    max_dd_duration = _longest_drawdown(drawdown)
    return {
        "initial_capital": initial_cash, "final_equity": final_equity, "total_return": final_equity / initial_cash - 1,
        "annualized_return": annualized, "number_of_trades": int(len(trades)),
        "long_trades": int((trades.side == "long").sum()) if not trades.empty else 0,
        "short_trades": int((trades.side == "short").sum()) if not trades.empty else 0,
        "win_rate": float((trades.net_pnl > 0).mean()) if not trades.empty else None,
        "average_win": float(wins.mean()) if not wins.empty else None, "average_loss": float(losses.mean()) if not losses.empty else None,
        "expectancy": float(trades.net_pnl.mean()) if not trades.empty else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else None,
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "calmar_ratio": annualized / abs(max_dd) if annualized is not None and max_dd < 0 else None,
        "maximum_drawdown": max_dd, "average_drawdown": float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0,
        "longest_drawdown_bars": max_dd_duration,
        "exposure_time": float(trades.holding_hours.sum() / max(span_days * 24, 1)) if not trades.empty else 0.0,
        "average_holding_hours": float(trades.holding_hours.mean()) if not trades.empty else 0.0,
        "fees_paid": float(trades.fees.sum()) if not trades.empty else 0.0,
        "slippage_cost": float(trades.slippage_cost.sum()) if not trades.empty else 0.0,
        "gross_pnl": float(trades.gross_pnl.sum()) if not trades.empty else 0.0,
        "net_pnl": float(trades.net_pnl.sum()) if not trades.empty else 0.0,
        "long_net_pnl": float(trades.loc[trades.side == "long", "net_pnl"].sum()) if not trades.empty else 0.0,
        "short_net_pnl": float(trades.loc[trades.side == "short", "net_pnl"].sum()) if not trades.empty else 0.0,
        "return_by_asset": (trades.groupby("asset").net_pnl.sum() / initial_cash).to_dict() if not trades.empty else {},
        "tp_hit_distribution": trades.targets_hit.value_counts().sort_index().to_dict() if not trades.empty else {},
        "stop_before_tp1_rate": float((trades.exit_reason == "stop_before_tp1").mean()) if not trades.empty else 0.0,
        "post_tp1_stop_rate": float((trades.exit_reason == "post_tp1_stop").mean()) if not trades.empty else 0.0,
    }


def _longest_drawdown(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def _periods_per_year(equity: pd.DataFrame) -> float:
    if len(equity) < 2:
        return 365.0
    seconds = pd.Series(pd.to_datetime(equity["timestamp"], utc=True)).diff().dropna().dt.total_seconds().median()
    return 365.25 * 24 * 3600 / seconds if seconds and seconds > 0 else 365.0
