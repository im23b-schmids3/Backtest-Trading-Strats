from __future__ import annotations

import html
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.backtest.v2_engine import StrategyV2Engine
from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.reporting.validation import write_matrix_tables

DISTANCES = (4, 7, 10, 13, 16)
MOVES = (.0025, .005, .01, .02, .03, .05)


def run_v2_research(config: RunConfig, root: str | Path = "reports/v2") -> dict:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); np.random.seed(config.seed)
    previous = _read_previous_outputs(root)
    pairs = _pairs(config); v1_rows, v2_rows, diagnostics, samples, v2_trades, order_logs, equity_curves = [], [], [], [], [], [], []
    for asset, timeframe, bars in pairs:
        # V1 remains unchanged and is rerun on precisely the same bars for comparison.
        for n in range(2, 11):
            for distance in (5, 10, 15):
                run = replace(config, assets=[asset], timeframes=[timeframe], swing_n=n, min_pivot_distance=distance, max_positions=1)
                trades, equity = BacktestEngine(run).run({asset: bars})
                v1_rows.append(_row("Strategy_V1", asset, timeframe, distance, None, trades, equity, config.initial_cash, n))
        for distance in DISTANCES:
            for move in MOVES:
                run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
                engine = StrategyV2Engine(run, move); trades, equity = engine.run({asset: bars})
                v2_rows.append(_row("Strategy_V2", asset, timeframe, distance, move, trades, equity, config.initial_cash, None))
                if not trades.empty:
                    v2_trades.append(trades.assign(timeframe=timeframe, strategy="Strategy_V2", distance_parameter=distance, minimum_move_parameter=move))
                order_logs.append(_order_log(engine.lifecycle_history, asset, timeframe, distance, move))
                if not equity.empty:
                    equity_curves.append(equity.assign(asset=asset, timeframe=timeframe, min_distance=distance, min_move=move, configuration_id=f"{asset}-{timeframe}-d{distance}-m{move}"))
                diagnostics.append({"asset": asset, "timeframe": timeframe, "min_distance": distance, "min_move": move, **engine.diagnostics[asset]})
                if distance == 7 and move == .01:
                    samples.extend((asset, timeframe, bars, event.setup) for event in engine.construction[asset].events if event.setup is not None and event.action in {"activate", "update"})
    v1 = pd.DataFrame(v1_rows); v2 = pd.DataFrame(v2_rows); diag = pd.DataFrame(diagnostics)
    v1.to_csv(root / "v1_same_data_matrix.csv", index=False); v2.to_csv(root / "v2_full_matrix.csv", index=False); diag.to_csv(root / "v2_diagnostics_by_configuration.csv", index=False)
    trade_log = pd.concat(v2_trades, ignore_index=True) if v2_trades else pd.DataFrame()
    trade_log = _attach_config_metrics(trade_log, v2)
    trade_log.to_csv(root / "v2_executed_trades.csv", index=False)
    trade_log.to_parquet(root / "v2_executed_trades.parquet", index=False)
    order_log = pd.concat([frame for frame in order_logs if not frame.empty], ignore_index=True) if order_logs else pd.DataFrame()
    order_log = _attach_config_metrics(order_log, v2)
    order_log.to_csv(root / "v2_order_log.csv", index=False)
    equity_log = pd.concat(equity_curves, ignore_index=True) if equity_curves else pd.DataFrame()
    equity_log = _attach_config_metrics(equity_log, v2)
    equity_log.to_csv(root / "v2_equity_curves.csv", index=False)
    diag.melt(id_vars=["asset", "timeframe", "min_distance", "min_move"], value_vars=["rejected_distance", "rejected_move", "cancellations_extreme_updated", "expired_setups", "anchor_break_invalidations", "max_anchor_age_invalidations", "max_anchor_age_invalidations_before_entry"], var_name="rejection_reason", value_name="count").to_csv(root / "v2_rejection_reasons.csv", index=False)
    funnel = diag.groupby(["asset", "timeframe"], as_index=False)[[c for c in diag.columns if c not in {"asset", "timeframe", "min_distance", "min_move"}]].mean()
    funnel.to_csv(root / "v2_funnel_by_asset_timeframe.csv", index=False)
    ranked = write_matrix_tables(v2.rename(columns={"min_distance": "min_pivot_distance"}).assign(swing_n="V2"), root)
    robust_ranked = _robust_rank(v2)
    robust_ranked.to_csv(root / "v2_ranked_matrix.csv", index=False)
    _best_configurations(robust_ranked).to_csv(root / "v2_best_configurations.csv", index=False)
    _performance_summary(v2).to_csv(root / "v2_performance_summary.csv", index=False)
    comparison = _comparison(v1, v2); comparison.to_csv(root / "v1_v2_comparison.csv", index=False)
    behavior = _behavioral_comparison(v2, diag, trade_log, previous)
    behavior.to_csv(root / "v2_behavioral_comparison.csv", index=False)
    _write_samples(root / "visual_samples.html", samples, config.seed)
    _write_report(root, comparison, funnel, behavior, v1, v2)
    return {"pairs": len(pairs), "v1_runs": len(v1), "v2_runs": len(v2), "samples": min(100, len(samples)), "root": str(root)}


def _order_log(history, asset, timeframe, distance, move):
    versions = {}
    active = {}
    rows = []
    for event in history:
        setup_id = event.get("setup_id")
        if event["action"] in {"activate", "update"}:
            version = versions.get(setup_id, 0) + 1
            versions[setup_id] = version
            record = {
                "asset": asset, "timeframe": timeframe, "min_distance": distance, "min_move": move,
                "setup_id": setup_id, "side": event.get("side"), "trend_id": event.get("trend_id"),
                "order_version": version, "lifecycle_action": event["action"], "status": "pending",
                "created_timestamp": event.get("timestamp"), "order_submission_timestamp": event.get("order_submission"),
                "entry_price": event.get("entry"), "stop_price": event.get("stop"),
                "cancel_timestamp": None, "cancel_reason": None, "fill_timestamp": None,
            }
            rows.append(record)
            active[setup_id] = record
        elif event["action"] == "cancelled":
            record = active.pop(setup_id, None)
            if record is not None:
                record["status"] = "cancelled"
                record["cancel_timestamp"] = event.get("timestamp")
                record["cancel_reason"] = event.get("reason")
        elif event["action"] == "filled":
            record = active.pop(setup_id, None)
            if record is not None:
                record["status"] = "filled"
                record["fill_timestamp"] = event.get("timestamp")
        elif event["action"] == "invalidated":
            record = active.pop(setup_id, None)
            if record is not None:
                record["status"] = "invalidated"
                record["cancel_timestamp"] = event.get("timestamp")
                record["cancel_reason"] = event.get("reason")
    for record in active.values():
        record["status"] = "open_at_test_end"
    return pd.DataFrame(rows)


def _attach_config_metrics(frame, v2):
    if frame.empty:
        return frame
    result = frame.copy()
    if "min_distance" not in result:
        result["min_distance"] = result["distance_parameter"]
    if "min_move" not in result:
        result["min_move"] = result["minimum_move_parameter"]
    metrics = ["number_of_trades", "trades_per_year", "net_pnl", "total_return", "profit_factor", "win_rate", "expectancy", "sharpe_ratio", "sortino_ratio", "maximum_drawdown"]
    new_metrics = [metric for metric in metrics if metric not in result.columns]
    if not new_metrics:
        return result
    lookup = v2[["asset", "timeframe", "min_distance", "min_move", *new_metrics]].drop_duplicates(["asset", "timeframe", "min_distance", "min_move"])
    return result.merge(lookup, on=["asset", "timeframe", "min_distance", "min_move"], how="left", validate="many_to_one")


def _robust_rank(v2):
    ranked = v2.copy()
    trades = pd.to_numeric(ranked["number_of_trades"], errors="coerce").fillna(0)
    drawdown = ranked["maximum_drawdown"].abs().fillna(0)
    sharpe = ranked["sharpe_ratio"].clip(-3, 3).fillna(0)
    profit_factor = ranked["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3)
    low_trade_penalty = 0.75 * (np.maximum(0, 30 - trades) / 30)
    instability_penalty = np.maximum(0, ranked["total_return"].abs() - 0.05) * (np.maximum(0, 30 - trades) / 30)
    ranked["robust_score"] = ranked["total_return"] - 0.75 * drawdown + sharpe / 12 + (profit_factor - 1) / 12 - low_trade_penalty - instability_penalty
    ranked["robust_rank"] = ranked["robust_score"].rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("robust_rank").reset_index(drop=True)


def _best_configurations(ranked, per_group=3):
    best = ranked.sort_values(["asset", "timeframe", "robust_rank"]).groupby(["asset", "timeframe"], group_keys=False).head(per_group).copy()
    best["rank_within_asset_timeframe"] = best.groupby(["asset", "timeframe"]).cumcount() + 1
    return best


def _performance_summary(v2):
    metrics = ["number_of_trades", "trades_per_year", "net_pnl", "total_return", "profit_factor", "win_rate", "expectancy", "sharpe_ratio", "sortino_ratio", "maximum_drawdown"]
    summary = v2.groupby(["asset", "timeframe"], as_index=False)[metrics].median()
    summary.insert(2, "min_distance", np.nan)
    summary.insert(3, "min_move", np.nan)
    summary.insert(0, "aggregation", "median across parameter combinations")
    return summary


def _pairs(config):
    pairs=[]
    for asset in config.assets:
        for timeframe in config.timeframes:
            try: pairs.append((asset,timeframe,Cache().read(asset,timeframe,config.asset_configs[asset].source == "yfinance")))
            except (FileNotFoundError, ValueError): pass
    return pairs


def _row(strategy, asset, timeframe, distance, move, trades, equity, capital, n):
    metrics=calculate_metrics(trades,equity,capital)
    average_r=float((trades.net_pnl/trades.risk_budget).mean()) if not trades.empty else 0.0
    years=max((pd.Timestamp(equity.timestamp.iloc[-1])-pd.Timestamp(equity.timestamp.iloc[0])).days/365.25, 1) if len(equity)>1 else 1
    return {"strategy":strategy,"asset":asset,"timeframe":timeframe,"min_distance":distance,"min_move":move,"swing_n":n,"average_r":average_r,"trades_per_year":len(trades)/years,**metrics}


def _comparison(v1,v2):
    fields=["number_of_trades","trades_per_year","win_rate","profit_factor","sharpe_ratio","maximum_drawdown","total_return","expectancy","average_r"]
    rows=[]
    for strategy, frame in (("Strategy_V1",v1),("Strategy_V2",v2)):
        for (asset,timeframe), group in frame.groupby(["asset","timeframe"]):
            rows.append({"strategy":strategy,"asset":asset,"timeframe":timeframe,**{f"median_{field}":group[field].median() for field in fields},"configurations":len(group)})
    return pd.DataFrame(rows)


def _read_previous_outputs(root: Path) -> dict[str, pd.DataFrame]:
    report = root / "v2_report.html"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        if "only distance and percentage-move filters" in text:
            tables = _read_html_tables(text)
            if len(tables) >= 2:
                comparison = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                matrix = comparison[comparison.strategy == "Strategy_V2"][["asset", "timeframe", "median_trades_per_year"]].rename(columns={"median_trades_per_year": "trades_per_year"})
                diagnostics = pd.DataFrame(tables[1][1:], columns=tables[1][0])
                return {"v2_full_matrix.csv": matrix, "v2_diagnostics_by_configuration.csv": diagnostics, "v2_executed_trades.csv": pd.DataFrame()}
    outputs = {}
    for name in ("v2_full_matrix.csv", "v2_diagnostics_by_configuration.csv", "v2_executed_trades.csv"):
        path = root / name
        if path.exists():
            outputs[name] = pd.read_csv(path)
    return outputs


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.cells = [], [], []
        self.in_table = False
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table, self.rows = True, []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cells.append("")
        elif self.in_table and tag == "tr":
            self.cells = []

    def handle_endtag(self, tag):
        if self.in_table and tag in {"th", "td"}:
            self.in_cell = False
        elif self.in_table and tag == "tr":
            if self.cells:
                self.rows.append(self.cells)
        elif tag == "table":
            self.tables.append(self.rows)
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cells[-1] += data


def _read_html_tables(text):
    parser = _TableParser()
    parser.feed(text)
    return [[[cell.strip() for cell in row] for row in table] for table in parser.tables]


def _behavioral_comparison(v2, diag, trades, previous):
    current = _behavioral_summary(v2, diag, trades, "updated_v2")
    old_matrix = previous.get("v2_full_matrix.csv", pd.DataFrame())
    old_diag = previous.get("v2_diagnostics_by_configuration.csv", pd.DataFrame())
    old_trades = previous.get("v2_executed_trades.csv", pd.DataFrame())
    old = _behavioral_summary(old_matrix, old_diag, old_trades, "previous_v2")
    keys = ["asset", "timeframe"]
    merged = current.merge(old, on=keys, how="outer")
    prefix = "updated_v2_"
    metrics = [column.removeprefix(prefix) for column in current.columns if column not in keys]
    for metric in metrics:
        merged[f"change_{metric}"] = merged[f"updated_v2_{metric}"] - merged[f"previous_v2_{metric}"]
    return merged.sort_values(keys).reset_index(drop=True)


def _behavioral_summary(matrix, diag, trades, prefix):
    if matrix.empty:
        return pd.DataFrame(columns=["asset", "timeframe", f"{prefix}_trades_per_year", f"{prefix}_average_anchor_age_days", f"{prefix}_anchors_created", f"{prefix}_average_setups_per_trend", f"{prefix}_average_trades_per_trend", f"{prefix}_pct_setups_invalidated_max_anchor_age"])
    pairs = matrix[["asset", "timeframe"]].drop_duplicates()
    rows = []
    for asset, timeframe in pairs.itertuples(index=False):
        m = matrix[(matrix.asset == asset) & (matrix.timeframe == timeframe)]
        d = diag[(diag.asset == asset) & (diag.timeframe == timeframe)] if not diag.empty else pd.DataFrame()
        t = trades[(trades.asset == asset) & (trades.timeframe == timeframe)] if not trades.empty and "timeframe" in trades.columns else pd.DataFrame()
        if not t.empty and {"anchor_timestamp", "fill_timestamp"}.issubset(t.columns):
            anchor_age = (pd.to_datetime(t.fill_timestamp, utc=True) - pd.to_datetime(t.anchor_timestamp, utc=True)).dt.total_seconds().div(86400).mean()
        else:
            anchor_age = float("nan")
        anchors = _mean_or_zero(d, "anchors_created", "anchor_candidates")
        setups_per_trend = _mean_or_zero(d, "average_setups_per_trend")
        if setups_per_trend == 0 and not d.empty:
            setups_per_trend = _safe_divide(_mean_or_zero(d, "unique_active_setups"), _mean_or_zero(d, "anchor_candidates"))
        trades_per_trend = _mean_or_zero(d, "average_trades_per_trend")
        if trades_per_trend == 0 and not d.empty:
            trades_per_trend = _safe_divide(_mean_or_zero(d, "executed_trades"), _mean_or_zero(d, "anchor_candidates"))
        invalidated = _mean_or_zero(d, "max_anchor_age_invalidations_before_entry", "max_anchor_age_invalidations")
        setup_count = _mean_or_zero(d, "unique_active_setups")
        rows.append({
            "asset": asset, "timeframe": timeframe,
            f"{prefix}_trades_per_year": float(pd.to_numeric(m.trades_per_year, errors="coerce").median()) if "trades_per_year" in m else 0.0,
            f"{prefix}_average_anchor_age_days": float(anchor_age) if pd.notna(anchor_age) else float("nan"),
            f"{prefix}_anchors_created": anchors,
            f"{prefix}_average_setups_per_trend": setups_per_trend,
            f"{prefix}_average_trades_per_trend": trades_per_trend,
            f"{prefix}_pct_setups_invalidated_max_anchor_age": 100.0 * _safe_divide(invalidated, setup_count),
        })
    return pd.DataFrame(rows)


def _mean_or_zero(frame, *columns):
    for column in columns:
        if column in frame:
            return float(pd.to_numeric(frame[column], errors="coerce").mean())
    return 0.0


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _write_samples(path, samples, seed):
    grouped={}
    for asset,timeframe,bars,setup in samples:
        grouped.setdefault((asset,timeframe,setup.identifier),(bars,[]))[1].append(setup)
    choices=list(grouped); rng=np.random.default_rng(seed); selected=[(choices[i], grouped[choices[i]]) for i in rng.choice(len(choices),size=min(100,len(choices)),replace=False)] if choices else []
    charts=[]
    for (asset,timeframe,_), (bars,setups) in selected:
        setup=setups[-1]
        center=setup.second.pivot_index; frame=bars.iloc[max(0,center-20):min(len(bars),center+21)]; lo,hi=float(frame.low.min()),float(frame.high.max()); scale=hi-lo or 1
        candles=[]
        for j,(_,bar) in enumerate(frame.iterrows()):
            x=45+j*10; y1=220-(bar.high-lo)/scale*190; y2=220-(bar.low-lo)/scale*190; yo=220-(bar.open-lo)/scale*190; yc=220-(bar.close-lo)/scale*190
            candles.append(f"<line x1='{x}' x2='{x}' y1='{y1:.1f}' y2='{y2:.1f}' stroke='#333'/><rect x='{x-3}' y='{min(yo,yc):.1f}' width='6' height='{max(1,abs(yc-yo)):.1f}' fill='#1565c0'/>")
        lines=[]
        for version,old in enumerate(setups[:-1],1):
            y=220-(old.fib.entry-lo)/scale*190; lines.append(f"<line x1='35' x2='465' y1='{y:.1f}' y2='{y:.1f}' stroke='#999' stroke-dasharray='2 3'/><text x='470' y='{y+3:.1f}'>cancelled v{version}</text>")
        for label,price,color in [("final entry",setup.fib.entry,"#1565c0"),("stop",setup.fib.stop,"#c62828")]+[(f"TP{i+1}",p,"#2e7d32") for i,p in enumerate(setup.fib.targets)]:
            y=220-(price-lo)/scale*190; lines.append(f"<line x1='35' x2='465' y1='{y:.1f}' y2='{y:.1f}' stroke='{color}' stroke-dasharray='4 2'/><text x='470' y='{y+3:.1f}'>{label}</text>")
        charts.append(f"<h3>{html.escape(asset)} {timeframe} {setup.side} — {html.escape(setup.identifier)}</h3><p>{len(setups)} Fib/order versions; grey lines are cancelled entries and blue is final active entry.</p><svg width='580' height='240' viewBox='0 0 580 240'>{''.join(candles+lines)}</svg>")
    path.write_text("<html><body><h1>Strategy V2 active-swing visual samples</h1><p>100 deterministic random stable-anchor samples (seed 42) from distance=7, move=1% runs. Wicks, Fib versions, cancelled entries, and final levels are overlaid.</p>"+"".join(charts)+"</body></html>",encoding="utf-8")


def _write_report(root, comparison, funnel, behavior, v1, v2):
    root.joinpath("v2_report.html").write_text("<html><body><h1>Strategy V1 vs Strategy V2</h1><p>V2 uses independent forward-only Fibonacci generations. Execution, sizing, costs, and exits are inherited unchanged.</p><h2>V1/V2 performance comparison</h2>"+comparison.to_html(index=False)+"<h2>Previous V2 vs updated V2 lifecycle comparison</h2>"+behavior.to_html(index=False)+"<h2>V2 funnel</h2>"+funnel.to_html(index=False)+"<h2>V2 top ranked configurations</h2>"+v2.sort_values("total_return",ascending=False).head(30).to_html(index=False)+"</body></html>",encoding="utf-8")
