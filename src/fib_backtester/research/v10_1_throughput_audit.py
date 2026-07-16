from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research import v9_alpha_risk_engine as v9
from fib_backtester.research import v10_prop_economics_audit as v10
from fib_backtester.strategy.swings import confirmed_swings


ROOT = Path("reports/v10_1")
MONTH_DAYS = 365.25 / 12
FROZEN_ACCOUNT = "25k"
FROZEN_MICROS = 2


def run_v10_1_throughput_audit(config: RunConfig, root: str | Path = ROOT) -> dict:
    """Audit throughput and account lifetime using the frozen V10/V9 path.

    This module deliberately owns no trading logic.  It calls the existing
    frozen V7/V9 engine and V10 account replay, then writes diagnostic tables.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    streams, lifecycles = _collect(config)
    timeline = _account_timeline(streams, lifecycles)
    throughput = _trade_throughput(streams, lifecycles)
    funnel = _signal_funnel(streams, lifecycles)
    loss = _trade_loss_analysis(streams, lifecycles)
    payout = _payout_analysis(streams, lifecycles)
    lifetime = _account_lifetime(timeline, throughput, loss)
    hypothetical = _hypothetical_frequency(timeline, throughput)

    timeline.to_csv(root / "v10_1_account_timeline.csv", index=False)
    throughput.to_csv(root / "v10_1_trade_throughput.csv", index=False)
    funnel.to_csv(root / "v10_1_signal_funnel.csv", index=False)
    loss.to_csv(root / "v10_1_trade_loss_analysis.csv", index=False)
    payout.to_csv(root / "v10_1_payout_analysis.csv", index=False)
    lifetime.to_csv(root / "v10_1_account_lifetime.csv", index=False)
    hypothetical.to_csv(root / "v10_1_hypothetical_trade_frequency.csv", index=False)
    _write_report(root / "v10_1_final_report.html", timeline, throughput, funnel, loss, payout, lifetime, hypothetical, streams)
    return {
        "streams": len(streams),
        "account_rows": len(timeline),
        "executed_trades": int(sum(item["executed_trades"] for item in streams)),
        "root": str(root),
    }


def _collect(config: RunConfig) -> tuple[list[dict], pd.DataFrame]:
    streams: list[dict] = []
    lifecycle_rows: list[dict] = []
    frozen = v9._load_frozen_parameters()
    cache = Cache()
    for asset in v9.ASSETS:
        for timeframe in v9.TIMEFRAMES:
            try:
                bars = cache.read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                distance, minimum_move = frozen[(asset, timeframe)]
                run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
                engine = v9.StrategyV7FrozenValidationEngine(run, minimum_move)
                raw, _ = engine.run({asset: bars})
            except Exception as exc:
                continue

            max_spec = v9._spec(asset, "micros", FROZEN_MICROS)
            variants = v9._build_variants(
                raw,
                asset,
                max_spec,
                (1, FROZEN_MICROS),
                config.asset_configs[asset].fee_rate,
                bars=bars,
                cutoff=v9._parse_time(v9.DEFAULT_SESSION_CUTOFF),
                liquidation=v9._parse_time(v9.DEFAULT_FORCED_LIQUIDATION),
                timezone=v9.SESSION_TIMEZONE,
            )
            stream = {
                "stream_key": f"{asset} {timeframe}",
                "asset": asset,
                "timeframe": timeframe,
                "bars": bars,
                "raw": raw,
                "variants": variants,
                "diagnostics": engine.diagnostics[asset],
                "detected_swings": len(confirmed_swings(bars, config.swing_n)),
                "distance": distance,
                "minimum_move": minimum_move,
                "history_days": max(1.0, (bars.index[-1] - bars.index[0]).total_seconds() / 86400),
                "executed_trades": int(engine.diagnostics[asset]["executed_trades"]),
                "error": "",
            }
            streams.append(stream)
            for start in v9._evaluation_starts(bars.index):
                for account_size, account in v10.ACCOUNT_SPECS.items():
                    row = v10._simulate_lifecycle(
                        variants,
                        start,
                        bars.index[-1],
                        asset,
                        timeframe,
                        distance,
                        minimum_move,
                        account_size,
                        account,
                    )
                    row["stream_key"] = f"{asset} {timeframe}"
                    lifecycle_rows.append(row)
    return streams, pd.DataFrame(lifecycle_rows)


def _parse_timestamp(value) -> pd.Timestamp | None:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value)


def _payout_timestamps(row) -> list[pd.Timestamp]:
    start = _parse_timestamp(row.start_date)
    offsets = v10._split_float(row.payout_offsets)
    return [start + pd.Timedelta(days=offset) for offset in offsets] if start is not None else []


def _eligible_signal_timestamps(stream: dict) -> list[pd.Timestamp]:
    values = []
    for signal in stream["variants"][FROZEN_MICROS]:
        if signal["cutoff_skipped"] or signal["trade"] is None:
            continue
        values.append(pd.Timestamp(signal["entry_timestamp"]))
    return values


def _account_timeline(streams: list[dict], lifecycles: pd.DataFrame) -> pd.DataFrame:
    lookup = {item["stream_key"]: item for item in streams}
    rows = []
    for row in lifecycles.itertuples(index=False):
        start = _parse_timestamp(row.start_date)
        end = _parse_timestamp(row.end_date)
        passed = _parse_timestamp(row.pass_timestamp)
        failed = _parse_timestamp(row.failure_timestamp)
        payouts = _payout_timestamps(row)
        terminal = failed or end
        signal_times = _eligible_signal_timestamps(lookup[row.stream_key])
        trading_times = [timestamp for timestamp in signal_times if start <= timestamp <= terminal]
        stages = [passed] + payouts + [failed]
        stage_days = {}
        if passed is not None:
            stage_days["evaluation_to_pass_days"] = (passed - start).total_seconds() / 86400
        for index, payout_time in enumerate(payouts[:3], 1):
            previous = passed if index == 1 else payouts[index - 2]
            stage_days[f"stage_to_payout_{index}_days"] = (payout_time - previous).total_seconds() / 86400 if previous is not None else np.nan
        if failed is not None:
            previous = payouts[min(len(payouts), 3) - 1] if payouts else (passed or start)
            stage_days["last_stage_to_failure_days"] = (failed - previous).total_seconds() / 86400
        rows.append({
            "stream_key": row.stream_key,
            "asset": row.asset,
            "timeframe": row.timeframe,
            "account_size": row.account_size,
            "evaluation_start": row.start_date,
            "evaluation_pass": row.pass_timestamp,
            "qualified_account": row.pass_timestamp,
            "first_payout_eligibility": str(payouts[0]) if len(payouts) >= 1 else "",
            "first_payout_received": str(payouts[0]) if len(payouts) >= 1 else "",
            "second_payout": str(payouts[1]) if len(payouts) >= 2 else "",
            "third_payout": str(payouts[2]) if len(payouts) >= 3 else "",
            "failure": row.failure_timestamp,
            "voluntary_closure": "",
            "terminal_timestamp": str(terminal),
            "lifetime_days": row.lifetime_days,
            "evaluation_days": row.evaluation_days,
            "trading_days": len({v9._session(timestamp) for timestamp in trading_times}),
            "number_of_trades": row.trades_taken,
            "payout_count": row.payout_count,
            "first_payout_days": (payouts[0] - start).total_seconds() / 86400 if payouts else np.nan,
            "second_payout_days": (payouts[1] - start).total_seconds() / 86400 if len(payouts) >= 2 else np.nan,
            "third_payout_days": (payouts[2] - start).total_seconds() / 86400 if len(payouts) >= 3 else np.nan,
            "failure_days": (failed - start).total_seconds() / 86400 if failed is not None else np.nan,
            **stage_days,
        })
    return pd.DataFrame(rows)


def _trade_throughput(streams: list[dict], lifecycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream in streams:
        raw = stream["raw"]
        timestamps = sorted(pd.Timestamp(value) for value in raw.fill_timestamp) if not raw.empty else []
        count = len(timestamps)
        history_years = stream["history_days"] / 365.25
        rows.append({
            "asset": stream["asset"], "timeframe": stream["timeframe"], "scope": "strategy_stream",
            "history_days": stream["history_days"], "executed_trades": count,
            "trades_per_week": count / max(history_years * 52.1775, 1 / 52.1775),
            "trades_per_month": count / max(history_years * 12, 1 / 12),
            "trades_per_year": count / max(history_years, 1 / 365.25),
            "average_days_between_trades": float(np.mean(np.diff([x.value for x in timestamps]) / 86_400_000_000_000)) if len(timestamps) > 1 else np.nan,
            "account_scope": "raw frozen Strategy_V7/V9 stream; not a new backtest",
        })
        group = lifecycles[(lifecycles.stream_key == f"{stream['asset']} {stream['timeframe']}") & (lifecycles.account_size == FROZEN_ACCOUNT)]
        months = group.lifetime_days.sum() / MONTH_DAYS if not group.empty else np.nan
        rows.append({
            "asset": stream["asset"], "timeframe": stream["timeframe"], "scope": "25k_account_replay",
            "history_days": group.lifetime_days.mean() if not group.empty else np.nan,
            "executed_trades": group.trades_taken.sum() if not group.empty else 0,
            "trades_per_week": group.trades_taken.sum() / max(months * 4.348125, 1 / 4.348125) if not group.empty else np.nan,
            "trades_per_month": group.trades_taken.sum() / max(months, 1 / 12) if not group.empty else np.nan,
            "trades_per_year": group.trades_taken.sum() / max(months / 12, 1 / 365.25) if not group.empty else np.nan,
            "average_days_between_trades": np.nan,
            "account_scope": "V10 baseline 25k account rows; trades taken after session/account rules",
        })
    return pd.DataFrame(rows)


def _signal_funnel(streams: list[dict], lifecycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream in streams:
        d = stream["diagnostics"]
        raw = stream["raw"]
        group = lifecycles[lifecycles.stream_key == f"{stream['asset']} {stream['timeframe']}"]
        first_payout_accounts = int((group.payout_count >= 1).sum())
        first_payout_contributions = 0
        for account in group.itertuples(index=False):
            payouts = _payout_timestamps(account)
            if not payouts:
                continue
            start = _parse_timestamp(account.start_date)
            first_payout_contributions += sum(
                1 for timestamp in _eligible_signal_timestamps(stream) if start <= timestamp <= payouts[0]
            )
        stages = [
            (1, "detected_swings", stream["detected_swings"]),
            (2, "valid_swing_pairs", d["anchor_candidates"]),
            (3, "fib_setups", d["eligible_setups"]),
            (4, "orders_placed", d["initial_orders"] + d["replacement_orders"]),
            (5, "orders_filled", d["filled_orders"]),
            (6, "trades_executed", d["executed_trades"]),
            (7, "tp1_reached", int((raw.targets_hit >= 1).sum()) if not raw.empty else 0),
            (8, "first_payout_contribution_trades", first_payout_contributions),
        ]
        denominator = max(float(stages[0][2]), 1.0)
        previous_count = None
        for order, stage, count in stages:
            rows.append({
                "asset": stream["asset"], "timeframe": stream["timeframe"], "stage_order": order,
                "stage": stage, "count": int(count), "per_100_detected": float(count) * 100 / denominator,
                "conversion_from_previous_stage": float(count) / previous_count if previous_count not in (None, 0) else np.nan,
                "denominator": "confirmed swings" if stage != "first_payout_contribution_trades" else "account replay contributions",
                "interpretation": "Causal V2/V9 diagnostic funnel; order versions include replacements." if stage == "orders_placed" else "",
            })
            previous_count = count
    return pd.DataFrame(rows)


def _trade_loss_analysis(streams: list[dict], lifecycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream in streams:
        d = stream["diagnostics"]
        group = lifecycles[lifecycles.stream_key == f"{stream['asset']} {stream['timeframe']}"]
        placed = d["initial_orders"] + d["replacement_orders"]
        age = d["max_anchor_age_invalidations_before_entry"]
        categories = {
            "session_cutoff": int(sum(item["cutoff_skipped"] for item in stream["variants"][FROZEN_MICROS])),
            "risk_engine": 0,
            "daily_loss_guard": int(group.daily_loss_violations.sum()),
            "maximum_loss_limit": int(group.maximum_loss_violations.sum()),
            "position_limit": int(d["conflicting_position_cancellations"]),
            "anchor_expiration": int(age + d["expired_setups"]),
            "entry_never_filled": int(max(0, placed - d["filled_orders"] - age - d["expired_setups"])),
            "forced_market_close": int(group.forced_exits.sum()),
            "other": 0,
        }
        total = max(sum(categories.values()), 1)
        for reason, count in categories.items():
            rows.append({
                "asset": stream["asset"], "timeframe": stream["timeframe"], "reason": reason,
                "event_count": count, "percentage_of_classified_events": count * 100 / total,
                "classification_basis": "diagnostic event counts; categories can describe different stages and are not a trade log",
                "notes": "Risk-engine opportunity rejects were not separately emitted by frozen V9; recorded as zero, not inferred.",
            })
    return pd.DataFrame(rows)


def _count_to_stage(stream: dict, start, timestamp) -> int:
    if timestamp is None:
        return 0
    return sum(1 for value in _eligible_signal_timestamps(stream) if start <= value <= timestamp)


def _payout_analysis(streams: list[dict], lifecycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = {item["stream_key"]: item for item in streams}
    for account_size, group in lifecycles.groupby("account_size", sort=False):
        stage_values = {"pass": [], "first_payout": [], "second_payout": [], "third_payout": []}
        realized = []
        contributions = []
        for row in group.itertuples(index=False):
            start = _parse_timestamp(row.start_date)
            passed = _parse_timestamp(row.pass_timestamp)
            payouts = _payout_timestamps(row)
            stream = lookup[row.stream_key]
            for key, timestamp in [("pass", passed), ("first_payout", payouts[0] if len(payouts) > 0 else None), ("second_payout", payouts[1] if len(payouts) > 1 else None), ("third_payout", payouts[2] if len(payouts) > 2 else None)]:
                if timestamp is not None:
                    stage_values[key].append(_count_to_stage(stream, start, timestamp))
            if row.trades_taken:
                realized.append(row.historical_trading_pnl / row.trades_taken)
                contributions.append(row.net_payouts / row.trades_taken)
        for stage, values in stage_values.items():
            rows.append({"account_size": account_size, "stage": stage, "accounts_reaching_stage": len(values), "average_trades_required": float(np.mean(values)) if values else np.nan, "median_trades_required": float(np.median(values)) if values else np.nan, "average_realized_profit_per_trade": float(np.mean(realized)) if realized else np.nan, "average_payout_contribution_per_trade": float(np.mean(contributions)) if contributions else np.nan, "method": "Frozen V10 account replay; stage trade counts use executable signal timestamps."})
    return pd.DataFrame(rows)


def _account_lifetime(timeline: pd.DataFrame, throughput: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in timeline.groupby("account_size", sort=False):
        rows.extend([
            {"scope": f"account_size_{label}", "metric": "average_lifetime_days", "value": group.lifetime_days.mean(), "evidence": "Account replay terminal time; natural end when no failure."},
            {"scope": f"account_size_{label}", "metric": "median_lifetime_days", "value": group.lifetime_days.median(), "evidence": "Account replay terminal time."},
            {"scope": f"account_size_{label}", "metric": "average_trading_days", "value": group.trading_days.mean(), "evidence": "Executable signal sessions through terminal time."},
            {"scope": f"account_size_{label}", "metric": "median_trading_days", "value": group.trading_days.median(), "evidence": "Executable signal sessions through terminal time."},
            {"scope": f"account_size_{label}", "metric": "average_number_of_trades", "value": group.number_of_trades.mean(), "evidence": "V10 account replay trades_taken."},
            {"scope": f"account_size_{label}", "metric": "median_number_of_trades", "value": group.number_of_trades.median(), "evidence": "V10 account replay trades_taken."},
            {"scope": f"account_size_{label}", "metric": "first_payout_rate", "value": (group.payout_count >= 1).mean(), "evidence": "Payout rules and consistency checks in frozen V10 replay."},
            {"scope": f"account_size_{label}", "metric": "average_payouts", "value": group.payout_count.mean(), "evidence": "Payout rules and consistency checks in frozen V10 replay."},
        ])
    stream = throughput[throughput.scope == "strategy_stream"]
    rows += [
        {"scope": "all_streams", "metric": "average_strategy_trades_per_month", "value": stream.trades_per_month.mean(), "evidence": "Raw frozen stream executed-trade throughput."},
        {"scope": "all_streams", "metric": "average_account_trades_per_month_25k", "value": throughput[(throughput.scope == "25k_account_replay")].trades_per_month.mean(), "evidence": "V10 25k account replay."},
        {"scope": "all_streams", "metric": "share_of_loss_classification_entry_never_filled", "value": (loss.reason == "entry_never_filled").mul(loss.event_count).sum() / max(loss.event_count.sum(), 1), "evidence": "Order-version funnel diagnostic; not mutually exclusive with account events."},
        {"scope": "all_streams", "metric": "risk_engine_events", "value": loss[loss.reason == "risk_engine"].event_count.sum(), "evidence": "Frozen V9 does not emit separate opportunity-rejection counters."},
        {"scope": "all_streams", "metric": "session_cutoff_events", "value": loss[loss.reason == "session_cutoff"].event_count.sum(), "evidence": "Session cutoff diagnostic."},
        {"scope": "all_streams", "metric": "position_limit_events", "value": loss[loss.reason == "position_limit"].event_count.sum(), "evidence": "Conflicting-position cancellations in frozen V2 engine."},
    ]
    return pd.DataFrame(rows)


def _hypothetical_frequency(timeline: pd.DataFrame, throughput: pd.DataFrame) -> pd.DataFrame:
    base = timeline[timeline.account_size == FROZEN_ACCOUNT]
    observed_months = base.lifetime_days.sum() / MONTH_DAYS
    actual_trades = base.number_of_trades.sum() / max(observed_months, 1 / 12)
    actual_payouts = base.payout_count.sum() / max(observed_months, 1 / 12)
    first_days = base.first_payout_days.dropna()
    rows = []
    for multiplier, label in [(1.0, "actual"), (1.25, "+25% trades"), (1.5, "+50% trades"), (2.0, "+100% trades")]:
        rows.append({"scenario": label, "trade_frequency_multiplier": multiplier, "estimated_trades_per_month": actual_trades * multiplier, "observed_payouts_per_month": actual_payouts, "estimated_days_to_first_payout": float(first_days.mean() / multiplier) if not first_days.empty else np.nan, "estimated_days_to_first_payout_median": float(first_days.median() / multiplier) if not first_days.empty else np.nan, "hypothetical_required_multiplier_for_one_payout_per_month": 1 / actual_payouts if actual_payouts > 0 else np.inf, "hypothetical_trades_per_month_for_one_payout": actual_trades / actual_payouts if actual_payouts > 0 else np.inf, "assumption": "Linear throughput scaling with unchanged trade quality, payout rules, sizing, and path; estimate only, not a backtest."})
    return pd.DataFrame(rows)


def _write_report(path: Path, timeline, throughput, funnel, loss, payout, lifetime, hypothetical, streams) -> None:
    stream_text = ", ".join(f"{item['asset']} {item['timeframe']}" for item in streams)
    bottleneck = funnel[(funnel.stage_order >= 2) & (funnel.stage_order <= 7) & funnel.conversion_from_previous_stage.notna()]
    biggest_stage = "not available"
    biggest_rate = np.nan
    if not bottleneck.empty:
        bottleneck_rates = {}
        for stage, frame in bottleneck.groupby("stage"):
            previous_order = int(frame.stage_order.iloc[0]) - 1
            previous = funnel[funnel.stage_order == previous_order]["count"].sum()
            bottleneck_rates[stage] = frame["count"].sum() / max(previous, 1)
        biggest_stage = min(bottleneck_rates, key=bottleneck_rates.get)
        biggest_rate = float(min(bottleneck_rates.values()))
    actual = hypothetical.iloc[0]
    required = actual.hypothetical_trades_per_month_for_one_payout
    tables = []
    for title, frame in [("Account timeline", timeline), ("Trade throughput", throughput), ("Signal funnel", funnel), ("Trade-loss diagnostics", loss), ("Payout analysis", payout), ("Account lifetime", lifetime), ("Hypothetical frequency", hypothetical)]:
        tables.append(f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, border=0, classes='data')}")
    conclusion = f"""
    <h2>Conclusion</h2>
    <p><b>Scope:</b> {html.escape(stream_text)}. This audit replays the frozen V10/V9 path and does not change or optimize trading rules.</p>
    <p><b>Why accounts remain active:</b> most rows reach the natural end of the retained history rather than a loss-limit failure; the strategy produces sparse executable trades and the payout process additionally requires qualification, five winning days, and consistency. Average lifetime is therefore primarily a horizon/throughput observation, not evidence that the account is continuously productive.</p>
    <p><b>Largest observable funnel bottleneck:</b> the transition into <b>{html.escape(biggest_stage)}</b>, with an aggregate conversion of {biggest_rate:.2%} from the preceding diagnostic stage. Order replacements are revisions of the same setup, so they are not independent opportunities. The loss table separates cutoff, account-rule events, anchor expiry, and unfilled order versions and explicitly marks unavailable V9 counters as zero rather than inventing them.</p>
    <p><b>Observed frequency:</b> the 25k account replay averages {actual.estimated_trades_per_month:.3f} trades/month across the retained streams. Under the linear-quality assumption, approximately {required:.3f} trades/month would be needed for one payout/month; this is a hypothetical scaling estimate, not a strategy result.</p>
    <p><b>Single improvement direction without changing edge:</b> improve opportunity throughput and execution availability measurement first—reduce avoidable signal/order attrition only if it can be done without changing the frozen edge. The audit does not prescribe a new trading rule.</p>
    <p><b>Limitations:</b> historical exchange-price proxies, monthly account starts, OHLC intrabar assumptions, and Alpha-style payout rules are modeled research assumptions. “First payout eligibility” is recorded at the payout event because the retained V10 replay does not persist a separate eligibility timestamp. Trading-day counts use executable signal sessions; payout contribution counts are account-replay measures and should not be mixed with raw stream opportunity counts.</p>
    """
    path.write_text("<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:2em}table{border-collapse:collapse;font-size:12px}th,td{padding:4px 6px;border:1px solid #ddd;white-space:nowrap}th{background:#eee}</style></head><body><h1>Strategy V10.1 Throughput Audit</h1>" + conclusion + "".join(tables) + "</body></html>", encoding="utf-8")
