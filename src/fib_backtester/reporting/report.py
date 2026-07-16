from __future__ import annotations

import html
import importlib.metadata
import json
from pathlib import Path

import pandas as pd


def write_run(run_dir: Path, config: dict, trades: pd.DataFrame, equity: pd.DataFrame, metrics: dict, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(run_dir / "trades.csv", index=False)
    equity.to_csv(run_dir / "equity.csv", index=False)
    package_versions = {}
    for package in ("fib-backtester", "pandas", "numpy", "ccxt", "PyYAML", "pyarrow", "yfinance"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    reproducibility = {"strategy_config": config, "dependency_versions": package_versions}
    (run_dir / "config.json").write_text(json.dumps(reproducibility, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    monthly = _monthly_returns(equity)
    monthly.to_csv(run_dir / "monthly_returns.csv")
    report = _html(metrics, metadata, trades, equity, monthly)
    (run_dir / "report.html").write_text(report, encoding="utf-8")


def _monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["return"])
    values = equity.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], utc=True)
    return values.set_index("timestamp")["equity"].resample("ME").last().pct_change().to_frame("return")


def _svg(values: pd.Series, color: str) -> str:
    if values.empty:
        return "<svg width='800' height='180'></svg>"
    lo, hi = float(values.min()), float(values.max())
    width, height = 800, 180
    scale = hi - lo or 1
    points = " ".join(f"{i * width / max(len(values)-1, 1):.1f},{height - (v-lo)/scale*height:.1f}" for i, v in enumerate(values))
    return f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}'><polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/></svg>"


def _html(metrics: dict, metadata: dict, trades: pd.DataFrame, equity: pd.DataFrame, monthly: pd.DataFrame) -> str:
    curve = equity["equity"] if not equity.empty else pd.Series(dtype=float)
    dd = curve / curve.cummax() - 1 if not curve.empty else curve
    metrics_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in metrics.items())
    info = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>" for k, v in metadata.items())
    setup = _setup_svg(trades.iloc[0]) if not trades.empty else "<p>No setup was filled.</p>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Fibonacci backtest</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:auto}}table{{border-collapse:collapse}}th,td{{padding:5px;border:1px solid #ddd;text-align:left}}svg{{border:1px solid #ddd}}</style></head><body>
<h1>Fibonacci retracement backtest</h1><ul>{info}</ul><h2>Equity curve</h2>{_svg(curve, '#1769aa')}<h2>Drawdown</h2>{_svg(dd, '#b71c1c')}<h2>Example executed Fibonacci setup</h2>{setup}
<h2>Metrics</h2><table>{metrics_rows}</table><h2>Monthly returns</h2>{monthly.to_html()}<h2>Trades</h2>{trades.to_html(index=False) if not trades.empty else '<p>No completed trades.</p>'}</body></html>"""


def _setup_svg(trade: pd.Series) -> str:
    levels = [("swing high", float(trade.fib_high), "#555"), ("entry 0.882", float(trade.entry_price), "#1565c0"),
              ("stop 1.02", float(trade.initial_stop), "#b71c1c")]
    for number, price in enumerate(json.loads(trade.targets), 1):
        levels.append((f"TP{number}", float(price), "#2e7d32"))
    levels.append(("swing low", float(trade.fib_low), "#555"))
    low, high = min(value for _, value, _ in levels), max(value for _, value, _ in levels)
    scale = high - low or 1
    lines = []
    for label, price, colour in levels:
        y = 20 + (high - price) / scale * 220
        lines.append(f"<line x1='90' x2='700' y1='{y:.1f}' y2='{y:.1f}' stroke='{colour}'/><text x='8' y='{y+4:.1f}'>{html.escape(label)} {price:.4f}</text>")
    pivots = f"{html.escape(str(trade.first_pivot_timestamp))} → {html.escape(str(trade.second_pivot_timestamp))}"
    return f"<p>Pivots: {pivots}; filled: {html.escape(str(trade.fill_timestamp))}</p><svg viewBox='0 0 800 260' width='100%' height='260'>{''.join(lines)}</svg>"
