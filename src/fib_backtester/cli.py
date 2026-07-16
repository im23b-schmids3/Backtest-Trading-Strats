from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.engine import BacktestEngine
from .backtest.metrics import calculate_metrics
from .config import RunConfig, load_config
from .data.cache import Cache
from .data.downloader import download
from .data.validation import validate_ohlcv
from .reporting.report import write_run
from .reporting.validation import write_matrix_tables
from .research.pipeline import run_research
from .research.v2 import run_v2_research
from .research.v3_entry import run_v3_entry_research
from .research.v4_take_profit import run_v4_take_profit_research
from .research.v5_trade_path import run_v5_trade_path_analysis
from .research.v6_post_tp1_stop import run_v6_post_tp1_stop_research
from .research.v6_5_post_tp1_stop_placement import run_v6_5_post_tp1_stop_placement
from .research.v7_frozen_validation import run_v7_frozen_validation
from .research.v8_alpha_futures_zero import run_v8_alpha_futures_zero
from .research.v9_alpha_risk_engine import run_v9_alpha_risk_engine
from .research.v10_prop_economics_audit import run_v10_prop_economics_audit
from .research.v10_1_throughput_audit import run_v10_1_throughput_audit
from .research.v10_2_order_lifecycle_audit import run_v10_2_order_lifecycle_audit
from .research.v11_market_expansion import run_v11_market_expansion
from .research.v11_5_proxy_market_expansion import run_v11_5_proxy_market_expansion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fib_backtester.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/default.yaml")
    get = sub.add_parser("download", parents=[common]); get.add_argument("--assets", nargs="+"); get.add_argument("--timeframes", nargs="+"); get.add_argument("--refresh", action="store_true")
    sub.add_parser("validate", parents=[common])
    sub.add_parser("backtest", parents=[common]); sub.add_parser("grid", parents=[common]); sub.add_parser("research", parents=[common]); sub.add_parser("v2-research", parents=[common]); sub.add_parser("v3-entry-research", parents=[common]); sub.add_parser("v4-take-profit-research", parents=[common]); sub.add_parser("v5-trade-path-analysis", parents=[common]); sub.add_parser("v6-post-tp1-stop-research", parents=[common]); sub.add_parser("v6-5-post-tp1-stop-placement", parents=[common]); sub.add_parser("v7-frozen-validation", parents=[common]); sub.add_parser("v8-alpha-futures-zero", parents=[common])
    v9 = sub.add_parser("v9-alpha-risk-engine", parents=[common]); v9.add_argument("--session-cutoff", default="22:20"); v9.add_argument("--forced-liquidation", default="22:30"); v9.add_argument("--session-timezone", default="Europe/Berlin")
    sub.add_parser("v10-prop-economics-audit", parents=[common])
    sub.add_parser("v10-1-throughput-audit", parents=[common])
    sub.add_parser("v10-2-order-lifecycle-audit", parents=[common])
    sub.add_parser("v11-market-expansion", parents=[common])
    sub.add_parser("v11-5-proxy-market-expansion", parents=[common])
    report = sub.add_parser("report"); report.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        path = Path("reports/runs") / args.run_id / "report.html"
        if not path.exists():
            raise SystemExit(f"report not found: {path}")
        print(path.resolve()); return 0
    config = load_config(args.config)
    if args.command == "download":
        availability = []
        for asset in args.assets or config.assets:
            for timeframe in args.timeframes or config.timeframes:
                try:
                    frame = download(asset, timeframe, config.asset_configs[asset], config.start, Cache(), args.refresh)
                    availability.append({"asset": asset, "timeframe": timeframe, "status": "available", "rows": len(frame), "detail": ""})
                    print(f"cached {asset} {timeframe}: {len(frame)} rows")
                except Exception as exc:
                    availability.append({"asset": asset, "timeframe": timeframe, "status": "skipped", "rows": 0, "detail": str(exc)})
                    print(f"skipped {asset} {timeframe}: {exc}")
        _write_availability(availability)
        return 0
    if args.command == "validate":
        availability = []
        for asset in config.assets:
            for timeframe in config.timeframes:
                try:
                    frame = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                    validate_ohlcv(frame, timeframe, config.asset_configs[asset].source == "yfinance")
                    availability.append({"asset": asset, "timeframe": timeframe, "status": "valid", "rows": len(frame), "detail": ""})
                    print(f"valid {asset} {timeframe}: {len(frame)} rows")
                except Exception as exc:
                    availability.append({"asset": asset, "timeframe": timeframe, "status": "unavailable", "rows": 0, "detail": str(exc)})
                    print(f"unavailable {asset} {timeframe}: {exc}")
        _write_availability(availability)
        return 0
    if args.command == "grid":
        summaries = []
        skipped = []
        pairs = _available_pairs(config, skipped)
        for n in range(2, 11):
            for distance in (5, 10, 15):
                for asset, timeframe in pairs:
                    run = RunConfig(run_name=f"grid-{asset}-{timeframe}-n{n}-d{distance}", seed=config.seed, start=config.start,
                        initial_cash=config.initial_cash, assets=[asset], timeframes=[timeframe], swing_n=n, min_pivot_distance=distance,
                        entry_max_age_bars=config.entry_max_age_bars, reentry=config.reentry, execution_policy="conservative",
                        max_positions=1, max_total_risk_fraction=config.max_total_risk_fraction, leverage=config.leverage, asset_configs=config.asset_configs)
                    summaries.extend(_run(run, run.run_name))
        ranked = write_matrix_tables(summaries)
        pd.DataFrame(skipped).to_csv("reports/skipped_combinations.csv", index=False)
        print(f"reports/result_matrix.csv ({len(ranked)} runs)"); return 0
    if args.command == "research":
        result = run_research(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v2-research":
        result = run_v2_research(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v3-entry-research":
        result = run_v3_entry_research(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v4-take-profit-research":
        result = run_v4_take_profit_research(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v5-trade-path-analysis":
        result = run_v5_trade_path_analysis(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v6-post-tp1-stop-research":
        result = run_v6_post_tp1_stop_research(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v6-5-post-tp1-stop-placement":
        result = run_v6_5_post_tp1_stop_placement(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v7-frozen-validation":
        result = run_v7_frozen_validation(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v8-alpha-futures-zero":
        result = run_v8_alpha_futures_zero(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v9-alpha-risk-engine":
        result = run_v9_alpha_risk_engine(config, session_cutoff=args.session_cutoff, forced_liquidation=args.forced_liquidation, session_timezone=args.session_timezone)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v10-prop-economics-audit":
        result = run_v10_prop_economics_audit(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v10-1-throughput-audit":
        result = run_v10_1_throughput_audit(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v10-2-order-lifecycle-audit":
        result = run_v10_2_order_lifecycle_audit(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v11-market-expansion":
        result = run_v11_market_expansion(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "v11-5-proxy-market-expansion":
        result = run_v11_5_proxy_market_expansion(config)
        print(json.dumps(result, indent=2, default=str)); return 0
    summaries = _run(config, config.run_name)
    Path("reports").mkdir(exist_ok=True)
    write_matrix_tables(summaries)
    pd.DataFrame(summaries).to_csv("reports/summary.csv", index=False)
    print(json.dumps(summaries, indent=2, default=str)); return 0


def _run(config: RunConfig, prefix: str) -> list[dict]:
    np.random.seed(config.seed)
    run_id = f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    all_summaries = []
    for timeframe in config.timeframes:
        data = {asset: Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance") for asset in config.assets}
        replay = None
        if config.execution_policy == "lower_timeframe_replay":
            if timeframe == "1h":
                raise ValueError("lower_timeframe_replay needs data below 1h; no lower source is configured")
            replay = {asset: Cache().read(asset, "1h", config.asset_configs[asset].source == "yfinance") for asset in config.assets}
        trades, equity = BacktestEngine(config).run(data, replay)
        if not trades.empty:
            trades = trades.assign(timeframe=timeframe, execution_policy=config.execution_policy)
        metrics = calculate_metrics(trades, equity, config.initial_cash)
        # Buy-and-hold is equal-weighted source-price return, deliberately labelled separately from portfolio result.
        metrics["buy_and_hold_return"] = float(pd.Series([bars.close.iloc[-1] / bars.close.iloc[0] - 1 for bars in data.values()]).mean())
        metrics["trend_filter_return"] = float(pd.Series([_trend_filter_return(bars.close) for bars in data.values()]).mean())
        metadata = {"assets": ", ".join(config.assets), "timeframe": timeframe, "swing_confirmation": config.swing_n,
                    "minimum_pivot_distance": config.min_pivot_distance, "sources": ", ".join(f"{a}:{config.asset_configs[a].source}/{config.asset_configs[a].symbol}" for a in config.assets),
                    "test_start": config.start, "test_end": max(b.index[-1] for b in data.values()), "execution_policy": config.execution_policy,
                    "costs": {a: {"fee": config.asset_configs[a].fee_rate, "slippage": config.asset_configs[a].slippage_rate} for a in config.assets},
                    "version": _version()}
        output = Path("reports/runs") / f"{run_id}-{timeframe}"
        write_run(output, config.to_dict(), trades, equity, metrics, metadata)
        all_summaries.append({"run_id": output.name, "asset": ",".join(config.assets), "timeframe": timeframe,
                              "swing_n": config.swing_n, "min_pivot_distance": config.min_pivot_distance,
                              "execution_policy": config.execution_policy, **metrics})
    return all_summaries


def _version() -> str:
    try:
        return importlib.metadata.version("fib-backtester")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _available_pairs(config: RunConfig, skipped: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for asset in config.assets:
        for timeframe in config.timeframes:
            try:
                Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                pairs.append((asset, timeframe))
            except Exception as exc:
                skipped.append({"asset": asset, "timeframe": timeframe, "reason": str(exc)})
    return pairs


def _trend_filter_return(close: pd.Series, window: int = 200) -> float:
    returns = close.pct_change().fillna(0)
    signal = close.shift(1) > close.rolling(window).mean().shift(1)
    return float((1 + returns.where(signal, 0)).prod() - 1)


def _write_availability(rows: list[dict]) -> None:
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/data_availability.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
