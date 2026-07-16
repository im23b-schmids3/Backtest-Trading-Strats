from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.config import RunConfig


ROOT = Path("reports/v11")
INTRADAY_ROOT = Path("data/v11_intraday_raw")
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION = 0.20
FROZEN_DISTANCE = 4
FROZEN_MIN_MOVE = 0.0025
OFFICIAL_SOURCE = "https://alpha-futures.com/assets"
INTRADAY_PROXY_SOURCE = "Yahoo Finance continuous futures 1H proxy"
TIMEFRAMES = ("1h", "4h")


def _catalog() -> list[dict]:
    groups = {
        "Equity Indices": ("CME", ("ES", "MES", "NKD", "NQ", "MNQ", "RTY", "M2K", "YM", "MYM")),
        "Currencies": ("CME", ("6A", "M6A", "6B", "M6B", "6C", "6E", "E7", "M6E", "6S", "6M", "6N")),
        "Energy": ("NYMEX", ("CL", "MCL", "QM", "NG", "QG", "MNG", "RB", "HO")),
        "Agriculture": ("CBOT", ("ZC", "ZW", "ZS", "ZM", "ZL", "LE", "HE")),
        "Metals": ("COMEX", ("GC", "MGC", "SI", "SIL", "HG", "MHG", "PL")),
        "Interest Rates": ("CBOT", ("ZT", "ZF", "ZN", "ZB", "UB", "TN")),
        "Crypto": ("CME", ("MBT", "MET")),
    }
    micro = {"MES", "MNQ", "M2K", "MYM", "M6A", "M6B", "M6E", "MGC", "SIL", "MCL", "MNG", "MHG", "MBT", "MET"}
    e_mini = {"E7", "QM", "QG"}
    rows = []
    for asset_class, (exchange, symbols) in groups.items():
        for symbol in symbols:
            rows.append({
                "market": symbol,
                "symbol": symbol,
                "asset_class": asset_class,
                "exchange": exchange,
                "contract_type": "micro" if symbol in micro else ("e-mini" if symbol in e_mini else "mini_or_standard"),
                "official_alpha_supported": True,
                "official_source": OFFICIAL_SOURCE,
                "historical_ticker": f"{symbol}=F",
            })
    # Requested by the study, but not present on the current official list.
    for symbol in ("6J", "M6J"):
        rows.append({
            "market": symbol,
            "symbol": symbol,
            "asset_class": "Currencies",
            "exchange": "CME",
            "contract_type": "micro" if symbol == "M6J" else "mini_or_standard",
            "official_alpha_supported": False,
            "official_source": OFFICIAL_SOURCE,
            "historical_ticker": f"{symbol}=F",
            "official_status_reason": "Not listed on the current official Alpha market page.",
        })
    return rows


def _portfolio_sets(admitted: list[str], ranking: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Keep the prior helper contract while making verified Alpha admission explicit."""
    # Keep a non-empty logical baseline for callers that inspect the portfolio
    # definitions, while making clear that it is not an admitted/tradable
    # market.  The corrected study must not silently substitute proxy data.
    baseline = ["NO_VERIFIED_ALPHA_BASELINE"]
    ordered = admitted
    if ranking is not None and not ranking.empty and "admitted" in ranking:
        ordered = ranking[ranking.admitted & ranking.official_alpha_supported].sort_values("robustness_score", ascending=False).market.tolist()
    portfolios = {
        "Portfolio A - MET or verified ETH futures baseline": baseline,
        "Portfolio B - current validated Alpha baseline": baseline,
        "Portfolio C - baseline plus MNQ and MES": baseline + [m for m in ("MNQ", "MES") if m in ordered],
        "Portfolio D - baseline plus admitted micro contracts": baseline + [m for m in ordered if m.startswith("M")],
        "Portfolio E - all admitted markets": baseline + list(ordered),
    }
    # Backward-compatible names used by the existing helper test and prior
    # research tooling.  They are aliases, not additional research runs.
    portfolios.update({
        "Portfolio A - current baseline": baseline,
        "Portfolio C - top 3 admitted": baseline + list(ordered[:3]),
        "Portfolio D - top 5 admitted": baseline + list(ordered[:5]),
        "Portfolio E - all admitted": baseline + list(ordered),
    })
    return portfolios


def run_v11_market_expansion(config: RunConfig, root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    inventory = _inventory()
    quality = _data_quality(inventory)
    validation = _not_admissible_validation(quality)
    rankings = _rankings(inventory, quality)
    admitted: list[str] = []
    portfolios = _portfolio_sets(admitted, rankings)
    portfolio = _portfolio_analysis(portfolios)
    prop = _prop_impact(portfolios)

    inventory.to_csv(root / "v11_intraday_market_inventory.csv", index=False)
    quality.to_csv(root / "v11_intraday_data_quality.csv", index=False)
    validation.to_csv(root / "v11_intraday_market_validation.csv", index=False)
    rankings.to_csv(root / "v11_intraday_market_rankings.csv", index=False)
    portfolio.to_csv(root / "v11_intraday_portfolio_analysis.csv", index=False)
    prop.to_csv(root / "v11_intraday_prop_impact.csv", index=False)
    _write_report(root / "v11_intraday_final_report.html", inventory, quality, validation, rankings, portfolio, prop)
    return {
        "inventory_rows": len(inventory),
        "quality_rows": len(quality),
        "validated_markets": 0,
        "admitted_markets": [],
        "root": str(root),
        "status": "no_native_or_causally_stitched_intraday_source_available",
    }


def _inventory() -> pd.DataFrame:
    rows = []
    for base in _catalog():
        for timeframe in TIMEFRAMES:
            rows.append({
                **base,
                "timeframe": timeframe,
                "data_path": str(INTRADAY_ROOT / f"{base['symbol']}_1h.parquet"),
                "data_source": INTRADAY_PROXY_SOURCE,
                "native_continuous": False,
                "causally_stitched": False,
                "adjusted_for_contract_rolls": "unknown_vendor_adjustment",
                "admission_status": "not_official" if not base["official_alpha_supported"] else "data_quality_review_required",
            })
    return pd.DataFrame(rows)


def _read_hourly(symbol: str) -> pd.DataFrame | None:
    path = INTRADAY_ROOT / f"{symbol}_1h.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()[["open", "high", "low", "close", "volume"]]
        frame = frame[~frame.index.duplicated(keep="last")].dropna()
        if frame.empty or (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
            return None
        return frame
    except Exception:
        return None


def _data_quality(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in inventory.to_dict("records"):
        frame = _read_hourly(record["symbol"])
        if frame is None:
            rows.append({**record, "status": "no_reliable_intraday_history", "first_timestamp": "", "last_timestamp": "", "candle_count": 0, "missing_intervals": np.nan, "session_gaps": np.nan, "rollover_gaps": "not determinable", "minimum_history_months": 0.0, "proxy_only_not_admissible": True, "admission_eligible": False, "reason": "No validated retained 1H data."})
            continue
        delta = frame.index.to_series().diff().dropna()
        missing = int(sum(max(int(round(diff.total_seconds() / 3600)) - 1, 0) for diff in delta if diff > pd.Timedelta(hours=1)))
        session_gaps = int((delta > pd.Timedelta(hours=2)).sum())
        months = (frame.index[-1] - frame.index[0]).total_seconds() / (86400 * 30.4375)
        timeframe = record["timeframe"]
        rows.append({
                **record,
                "status": "proxy_only_not_admissible",
                "first_timestamp": str(frame.index[0]),
                "last_timestamp": str(frame.index[-1]),
                "candle_count": len(frame) if timeframe == "1h" else int(len(frame.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna())),
                "missing_intervals": missing,
                "session_gaps": session_gaps,
                "rollover_gaps": "unknown_vendor_continuous_series",
                "minimum_history_months": months,
                "proxy_only_not_admissible": True,
                "admission_eligible": False,
                "reason": "Intraday Yahoo continuous-futures proxy is available, but no native or causally stitched contract history is retained; excluded from final admission.",
        })
    return pd.DataFrame(rows)


def _not_admissible_validation(quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in quality.to_dict("records"):
        rows.append({
            "market": record["market"],
            "timeframe": record["timeframe"],
            "stage": "not_run",
            "data_status": record["status"],
            "admission_eligible": False,
            "parameter_policy": f"global frozen baseline: distance={FROZEN_DISTANCE}, minimum_move={FROZEN_MIN_MOVE}; no market-specific selection",
            "training_trade_count": np.nan,
            "validation_trade_count": np.nan,
            "holdout_trade_count": np.nan,
            "validation_return": np.nan,
            "holdout_return": np.nan,
            "validation_expectancy": np.nan,
            "holdout_expectancy": np.nan,
            "profit_factor": np.nan,
            "sharpe_ratio": np.nan,
            "maximum_drawdown": np.nan,
            "trades_per_month": np.nan,
            "long_trades": np.nan,
            "short_trades": np.nan,
            "fees": np.nan,
            "slippage": np.nan,
            "reason": record["reason"],
        })
    return pd.DataFrame(rows)


def _rankings(inventory: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, group in inventory.groupby("market", sort=True):
        q = quality[quality.market == market]
        official = bool(group.official_alpha_supported.iloc[0])
        reliable = bool((q.status == "native_or_causally_stitched").any())
        rows.append({
            "market": market,
            "asset_class": group.asset_class.iloc[0],
            "official_alpha_supported": official,
            "native_or_stitched_data_available": reliable,
            "one_hour_data_available": bool((q.timeframe == "1h").any() and (q.candle_count > 0).any()),
            "four_hour_data_available": bool((q.timeframe == "4h").any() and (q.candle_count > 0).any()),
            "validation_performed": False,
            "positive_validation": False,
            "positive_holdout": False,
            "meaningful_holdout_evidence": False,
            "robustness_score": np.nan,
            "admitted": False,
            "rejection_reason": "No native or causally stitched 1H/4H futures data available; proxy-only data is not admissible.",
        })
    return pd.DataFrame(rows)


def _portfolio_analysis(portfolios: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    metrics = ("raw_signals_per_month", "executable_trades_per_month", "concurrent_position_conflicts", "skipped_trades", "return", "profit_factor", "sharpe_ratio", "maximum_drawdown", "worst_day", "average_time_between_trades_days")
    for portfolio, markets in portfolios.items():
        for metric in metrics:
            rows.append({"portfolio": portfolio, "metric_type": "portfolio", "market": "", "metric": metric, "value": 0.0, "status": "not_evaluated_no_admitted_market"})
        rows.append({"portfolio": portfolio, "metric_type": "correlation", "market": "", "metric": "daily_pnl_correlation", "value": np.nan, "status": "not_evaluated_no_admitted_market"})
        for market in markets:
            rows.append({"portfolio": portfolio, "metric_type": "market_contribution", "market": market, "metric": "net_pnl_contribution", "value": 0.0, "status": "not_evaluated_no_admitted_market"})
    return pd.DataFrame(rows)


def _prop_impact(portfolios: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for portfolio in portfolios:
        rows.append({
            "portfolio": portfolio,
            "account_type": "Alpha Zero 25K",
            "evaluation_pass_probability": np.nan,
            "first_payout_probability": np.nan,
            "second_payout_probability": np.nan,
            "average_days_to_pass": np.nan,
            "average_days_to_first_payout": np.nan,
            "average_trades_per_month": 0.0,
            "maximum_loss_failures": np.nan,
            "daily_loss_failures": np.nan,
            "expected_net_payout_after_costs": np.nan,
            "status": "not_run_no_admitted_market",
            "reason": "Prop replay is not meaningful without at least one admitted market supported by native or causally stitched intraday data.",
        })
    return pd.DataFrame(rows)


def _write_report(path, inventory, quality, validation, rankings, portfolio, prop):
    official_count = int(inventory.official_alpha_supported.sum())
    native_count = int(rankings.native_or_stitched_data_available.sum())
    proxy_count = int((quality.status == "proxy_only_not_admissible").sum())
    skipped = int((quality.status == "no_reliable_intraday_history").sum())
    tables = "".join(f"<h2>{title}</h2>{frame.to_html(index=False, border=0, classes='data')}" for title, frame in (("Market inventory", inventory), ("Intraday data quality", quality), ("Market validation", validation), ("Market rankings", rankings), ("Portfolio analysis", portfolio), ("Prop impact", prop)))
    conclusion = f"""
    <h2>Corrected conclusion</h2>
    <p><b>Strategy status:</b> unchanged. Entry, stops, TP Profile B, post-TP1 stop, execution assumptions, costs, risk model, and portfolio rules were not modified.</p>
    <p><b>Official market scope:</b> {official_count} listed Alpha symbols were inventoried. No retained source qualified as native or causally stitched intraday futures data ({native_count} markets qualified). {proxy_count} market/timeframe rows were proxy-only and {skipped} had no validated retained intraday history.</p>
    <p><b>Admission:</b> no market was admitted. Positive metrics from daily data or Yahoo continuous intraday proxies would not answer the intended Alpha Futures question and were therefore excluded from final validation and prop replay.</p>
    <p><b>Portfolio:</b> no corrected Alpha portfolio can be recommended from this repository state. All portfolio and prop rows are explicitly marked not evaluated rather than presenting proxy results as futures evidence.</p>
    <p><b>Data limitation:</b> the available intraday Yahoo series is a continuous-futures proxy with unknown roll construction and no independently verified contract stitching or official contract specifications. The required next data source is native or contract-level intraday futures history with documented rollover, tick/multiplier metadata, and session timestamps.</p>
    """
    path.write_text("<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:2em}table{border-collapse:collapse;font-size:11px}th,td{padding:4px 6px;border:1px solid #ddd;white-space:nowrap}th{background:#eee}</style></head><body><h1>Strategy V11 Corrected Intraday Market Expansion</h1>" + conclusion + tables + "</body></html>", encoding="utf-8")
