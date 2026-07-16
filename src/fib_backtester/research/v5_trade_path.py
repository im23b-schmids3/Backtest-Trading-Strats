from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research.v2 import DISTANCES, MOVES, _pairs
from fib_backtester.strategy.v4_take_profit_research import PROFILES


PROFILE = PROFILES["B"]
ENTRY_LEVEL = 0.900
CURRENT_POST_TP1 = 0.880
STOP_LEVELS = (1.00, 1.01, 1.02, 1.03, 1.05)
BREAK_EVEN_POLICIES = (
    ("no_stop_move", 1.02),
    ("break_even", 0.900),
    ("fib_0.90", 0.900),
    ("fib_0.786", 0.786),
    ("current_fib_0.88", 0.880),
)


def run_v5_trade_path_analysis(config: RunConfig, root: str | Path = "reports/v5") -> dict:
    """Analyze the fixed V4/Profile-B baseline; no parameter optimization is performed."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pair_specs = [(asset, timeframe) for asset, timeframe, _ in _pairs(config)]
    arguments = [(config, asset, timeframe) for asset, timeframe in pair_specs]
    workers = min(len(arguments), 8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(_run_pair, arguments))
    trades = pd.concat([frame for frame in batches if not frame.empty], ignore_index=True) if batches else pd.DataFrame()
    trades = trades.sort_values(["asset", "timeframe", "fill_timestamp", "setup_id"]).reset_index(drop=True)
    _BAR_REGISTRY.clear()
    for asset, timeframe, bars in _pairs(config):
        _BAR_REGISTRY[(asset, timeframe)] = bars

    conditional = _conditional_probabilities(trades)
    exit_efficiency = _exit_efficiency(trades)
    stop_analysis = _stop_analysis(trades)
    break_even = _break_even_analysis(trades)
    importance = _improvement_ranking(trades, stop_analysis, break_even, exit_efficiency)

    trades.to_csv(root / "v5_trade_path_analysis.csv", index=False)
    conditional.to_csv(root / "v5_conditional_probabilities.csv", index=False)
    exit_efficiency.to_csv(root / "v5_exit_efficiency.csv", index=False)
    stop_analysis.to_csv(root / "v5_stop_analysis.csv", index=False)
    break_even.to_csv(root / "v5_break_even_analysis.csv", index=False)
    importance.to_csv(root / "v5_improvement_ranking.csv", index=False)
    _write_report(root / "v5_report.html", trades, conditional, exit_efficiency, stop_analysis, break_even, importance, pair_specs)
    return {"pairs": len(pair_specs), "trades": len(trades), "root": str(root), "baseline": "Strategy_V4 / entry 0.900 / Profile B"}


def _run_pair(arguments):
    config, asset, timeframe = arguments
    bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
    rows = []
    for distance in DISTANCES:
        for move in MOVES:
            run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
            from fib_backtester.backtest.v4_take_profit_engine import StrategyV4TakeProfitResearchEngine

            engine = StrategyV4TakeProfitResearchEngine(run, move, PROFILE)
            executed, _ = engine.run({asset: bars})
            for trade in executed.to_dict("records"):
                rows.append(_analyze_trade(trade, bars, asset, timeframe, distance, move))
    return pd.DataFrame(rows)


def _analyze_trade(trade, bars, asset, timeframe, distance, move):
    entry_index = _trade_index(trade, bars, "entry_candle_index", "fill_timestamp")
    exit_index = _trade_index(trade, bars, "exit_candle_index", "exit_timestamp")
    path = bars.iloc[entry_index : exit_index + 1]
    side = trade["side"]
    entry = float(trade["entry_price"])
    stop = float(trade["initial_stop"])
    risk = abs(entry - stop)
    direction = 1.0 if side == "long" else -1.0
    favorable = (path.high.astype(float) - entry) if side == "long" else (entry - path.low.astype(float))
    adverse = (entry - path.low.astype(float)) if side == "long" else (path.high.astype(float) - entry)
    mfe_index = int(np.argmax(favorable.to_numpy()))
    mae_index = int(np.argmax(adverse.to_numpy()))
    events = _events(trade)
    targets = [float(value) for value in json.loads(trade["targets"])]
    target_events = {}
    for event in events:
        reason = str(event.get("reason", ""))
        if reason.startswith("tp") and reason[2:].isdigit():
            target_events.setdefault(int(reason[2:]), event)
    stop_event = next((event for event in events if "stop" in str(event.get("reason", ""))), None)
    result = {
        "strategy_version": "Strategy_V5_TradePathAnalysis",
        "baseline_strategy": "Strategy_V4_TakeProfitResearch",
        "profile": "B",
        "entry_level": ENTRY_LEVEL,
        "asset": asset,
        "timeframe": timeframe,
        "min_distance": distance,
        "min_move": move,
        "setup_id": trade["setup_id"],
        "side": side,
        "fill_timestamp": str(trade["fill_timestamp"]),
        "exit_timestamp": str(trade["exit_timestamp"]),
        "entry_price": entry,
        "fib_low": float(trade["fib_low"]),
        "fib_high": float(trade["fib_high"]),
        "targets": trade["targets"],
        "initial_stop": stop,
        "risk_per_unit": risk,
        "mfe_price": float(favorable.iloc[mfe_index]),
        "mae_price": float(adverse.iloc[mae_index]),
        "mfe_timestamp": str(path.index[mfe_index]),
        "mae_timestamp": str(path.index[mae_index]),
        "mfe_r": float(favorable.max() / risk) if risk else 0.0,
        "mae_r": float(adverse.max() / risk) if risk else 0.0,
        "r_multiple_at_mfe": float(favorable.max() / risk) if risk else 0.0,
        "time_until_stop_hours": _hours_between(trade["fill_timestamp"], stop_event.get("timestamp")) if stop_event else np.nan,
        "stop_reached": bool(stop_event),
        "stop_before_tp1": bool(stop_event and 1 not in target_events),
        "stop_after_tp1": bool(stop_event and 1 in target_events),
        "stop_after_tp2": bool(stop_event and 2 in target_events),
        "stop_after_tp3": bool(stop_event and 3 in target_events),
        "realized_profit": float(trade.get("net_pnl", 0.0)),
        "realized_r": float(trade.get("r_multiple", 0.0)),
        "gross_profit": float(trade.get("gross_pnl", 0.0)),
        "fees": float(trade.get("fees", 0.0)),
        "slippage_cost": float(trade.get("slippage_cost", 0.0)),
        "average_exit_price": float(trade.get("average_exit_price", entry)),
        "risk_budget": float(trade.get("risk_budget", 0.0)),
        "raw_realized_r": float(trade.get("gross_pnl", 0.0)) / float(trade.get("risk_budget", 1.0)) if trade.get("risk_budget") else 0.0,
        "exit_reason": trade.get("exit_reason"),
    }
    post_stop = _fib_price(side, float(trade["fib_low"]), float(trade["fib_high"]), CURRENT_POST_TP1)
    for i, target in enumerate(targets, start=1):
        event = target_events.get(i)
        result[f"tp{i}_price"] = target if np.isfinite(target) else np.nan
        result[f"tp{i}_r_multiple"] = direction * (target - entry) / risk if risk and np.isfinite(target) else np.nan
        result[f"tp{i}_reached"] = bool(event)
        result[f"time_until_tp{i}_hours"] = _hours_between(trade["fill_timestamp"], event.get("timestamp")) if event else np.nan
        result[f"tp{i}_timestamp"] = str(event.get("timestamp")) if event else ""
        result[f"tp{i}_mfe_r"] = _mfe_until(path, entry, risk, side, event.get("timestamp")) if event else np.nan
        result[f"tp{i}_after_previous"] = bool(event and (i == 1 or all(j in target_events for j in range(1, i))))
        result[f"price_returned_to_stop_after_tp{i}"] = _returned_to_stop(path, bars, entry_index, exit_index, event, post_stop, side) if i <= 3 else False
    result["average_unrealized_profit"] = result["mfe_r"] * result["risk_budget"]
    result["average_profit_left_on_table"] = result["average_unrealized_profit"] - result["realized_profit"]
    return result


def _trade_index(trade, bars, field, timestamp_field):
    value = trade.get(field)
    if value is not None and not pd.isna(value):
        return int(value)
    return int(bars.index.get_loc(pd.Timestamp(trade[timestamp_field])))


def _events(trade):
    value = trade.get("exit_events", "[]")
    return json.loads(value) if isinstance(value, str) else (value or [])


def _hours_between(start, end):
    if end is None or end == "":
        return np.nan
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600


def _fib_price(side, low, high, ratio):
    distance = high - low
    return high - ratio * distance if side == "long" else low + ratio * distance


def _mfe_until(path, entry, risk, side, timestamp):
    if not risk:
        return 0.0
    subset = path.loc[path.index <= pd.Timestamp(timestamp)]
    if subset.empty:
        return np.nan
    favorable = subset.high.astype(float) - entry if side == "long" else entry - subset.low.astype(float)
    return float(favorable.max() / risk)


def _returned_to_stop(path, bars, entry_index, exit_index, event, stop, side):
    if not event:
        return False
    event_index = int(bars.index.get_loc(pd.Timestamp(event["timestamp"])))
    later = bars.iloc[max(entry_index, event_index + 1) : exit_index + 1]
    if later.empty:
        return False
    return bool((later.low.astype(float) <= stop).any()) if side == "long" else bool((later.high.astype(float) >= stop).any())


def _conditional_probabilities(trades):
    rows = []
    total = len(trades)
    for label, numerator, denominator, description in [
        ("P(TP2 | TP1)", "tp2_after_previous", "tp1_reached", "TP2 reached after TP1"),
        ("P(TP3 | TP2)", "tp3_after_previous", "tp2_reached", "TP3 reached after TP2"),
        ("P(TP4 | TP3)", "tp4_after_previous", "tp3_reached", "TP4 reached after TP3"),
        ("P(TP5 | TP4)", "tp5_after_previous", "tp4_reached", "TP5 reached after TP4"),
        ("P(Stop before TP1)", "stop_before_tp1", None, "Stop before TP1"),
        ("P(Stop after TP1)", "stop_after_tp1", "tp1_reached", "Stop after TP1, conditional on TP1"),
        ("P(Stop after TP2)", "stop_after_tp2", "tp2_reached", "Stop after TP2, conditional on TP2"),
        ("P(Stop after TP3)", "stop_after_tp3", "tp3_reached", "Stop after TP3, conditional on TP3"),
    ]:
        denominator_mask = pd.Series(True, index=trades.index) if denominator is None else trades[denominator].astype(bool)
        numerator_mask = trades[numerator].astype(bool) & denominator_mask
        denom = int(denominator_mask.sum())
        num = int(numerator_mask.sum())
        rows.append({
            "metric": label, "numerator": num, "denominator": denom,
            "probability": num / denom if denom else np.nan,
            "unconditional_probability": num / total if total else np.nan,
            "denominator_definition": "all executed trades" if denominator is None else denominator,
            "description": description,
        })
    return pd.DataFrame(rows)


def _exit_efficiency(trades):
    if trades.empty:
        return pd.DataFrame()
    rows = []
    groups = [("all", "all", trades)] + [(asset, timeframe, group) for (asset, timeframe), group in trades.groupby(["asset", "timeframe"])]
    for asset, timeframe, group in groups:
        realized = group["realized_profit"]
        unrealized = group["average_unrealized_profit"]
        rows.append({
            "scope": "all" if asset == "all" else "asset_timeframe",
            "asset": asset, "timeframe": timeframe, "trades": len(group),
            "average_unrealized_profit": unrealized.mean(),
            "average_realized_profit": realized.mean(),
            "average_profit_left_on_table": group["average_profit_left_on_table"].mean(),
            "average_mfe_r": group["mfe_r"].mean(),
            "average_realized_r": group["realized_r"].mean(),
            "median_profit_left_on_table": group["average_profit_left_on_table"].median(),
        })
    target_rows = []
    for i in range(1, 6):
        reached = trades[f"tp{i}_reached"].astype(bool)
        target_r = trades.loc[reached, f"tp{i}_r_multiple"].mean()
        fallback = trades["raw_realized_r"].where(~reached, trades[f"tp{i}_r_multiple"])
        target_rows.append({
            "scope": "mathematical_target_estimate", "asset": "all", "timeframe": "all", "target": f"TP{i}",
            "reach_probability": reached.mean(), "target_r_multiple": target_r,
            "estimated_expected_r_per_allocated_unit": fallback.mean(),
            "suggested_allocation_weight": np.nan,
            "method": "Observed target payoff when reached; observed terminal R fallback otherwise. No new profile was backtested.",
        })
    estimates = pd.DataFrame(target_rows)
    positive = estimates["estimated_expected_r_per_allocated_unit"].clip(lower=0)
    estimates["suggested_allocation_weight"] = positive / positive.sum() if positive.sum() else 1 / len(estimates)
    estimates["current_profile_weight"] = list(PROFILE.fractions)
    estimates["current_profile_contribution"] = estimates["current_profile_weight"] * estimates["estimated_expected_r_per_allocated_unit"]
    current_ev = float(estimates["current_profile_contribution"].sum())
    best_ev = float(estimates["estimated_expected_r_per_allocated_unit"].max())
    rows.append({
        "scope": "mathematical_target_summary", "asset": "all", "timeframe": "all", "trades": len(trades),
        "current_profile_expected_r": current_ev, "best_single_target_expected_r": best_ev,
        "estimated_tp_allocation_improvement_r": best_ev - current_ev,
        "best_single_target": estimates.loc[estimates["estimated_expected_r_per_allocated_unit"].idxmax(), "target"],
        "method": "Linear empirical target-EV estimate; allocation is mathematical only and was not backtested.",
    })
    summary = pd.DataFrame(rows)
    return pd.concat([summary, estimates], ignore_index=True, sort=False)


def _simulate_policy(trade, bars, initial_stop_ratio=1.02, post_tp1_ratio=CURRENT_POST_TP1):
    entry_index = _trade_index(trade, bars, "entry_candle_index", "fill_timestamp")
    side = trade["side"]
    entry = float(trade["entry_price"])
    low, high = float(trade["fib_low"]), float(trade["fib_high"])
    initial_stop = _fib_price(side, low, high, initial_stop_ratio)
    post_stop = _fib_price(side, low, high, post_tp1_ratio)
    targets = [float(x) for x in json.loads(trade["targets"])]
    direction = 1.0 if side == "long" else -1.0
    risk = abs(entry - initial_stop)
    fractions = PROFILE.fractions
    done = [False] * 5
    remaining = 1.0
    realized_r = 0.0
    events = []
    current_stop = initial_stop
    for index in range(entry_index, len(bars)):
        bar = bars.iloc[index]
        stop_hit = float(bar.low) <= current_stop if side == "long" else float(bar.high) >= current_stop
        target_hits = [(float(bar.high) >= target if side == "long" else float(bar.low) <= target) for target in targets]
        if stop_hit:
            realized_r += remaining * direction * (current_stop - entry) / risk if risk else 0.0
            events.append({"reason": "stop_before_tp1" if not any(done) else "post_tp1_stop", "timestamp": str(bars.index[index])})
            return {"reason": events[-1]["reason"], "exit_timestamp": str(bars.index[index]), "realized_r": realized_r, "target_count": sum(done), "stop_ratio": initial_stop_ratio}
        for target_index, hit in enumerate(target_hits):
            if hit and not done[target_index] and remaining > 1e-12:
                quantity = min(fractions[target_index], remaining)
                realized_r += quantity * direction * (targets[target_index] - entry) / risk if risk else 0.0
                remaining -= quantity
                done[target_index] = True
                events.append({"reason": f"tp{target_index + 1}", "timestamp": str(bars.index[index])})
                if target_index == 0:
                    current_stop = post_stop
        if remaining <= 1e-12:
            return {"reason": "targets_complete", "exit_timestamp": str(bars.index[index]), "realized_r": realized_r, "target_count": sum(done), "stop_ratio": initial_stop_ratio}
    final = float(bars.close.iloc[-1])
    realized_r += remaining * direction * (final - entry) / risk if risk else 0.0
    return {"reason": "end_of_data", "exit_timestamp": str(bars.index[-1]), "realized_r": realized_r, "target_count": sum(done), "stop_ratio": initial_stop_ratio}


def _stop_analysis(trades):
    rows = []
    # Counterfactuals follow each isolated trade from its observed entry through the end
    # of the retained series. They deliberately do not create post-exit setups or replay
    # portfolio interactions, so these are path estimates rather than backtest results.
    baseline_outcomes = pd.DataFrame([_simulate_policy(trade, _bars_for_trade(trade), 1.02, CURRENT_POST_TP1) for _, trade in trades.iterrows()])
    current = baseline_outcomes["realized_r"]
    for ratio in STOP_LEVELS:
        frame = pd.DataFrame([_simulate_policy(trade, _bars_for_trade(trade), ratio, CURRENT_POST_TP1) for _, trade in trades.iterrows()])
        hit = frame.reason.str.contains("stop")
        premature = hit & (frame.target_count == 0)
        rows.append({
            "stop_fib_ratio": ratio, "trades": len(frame),
            "surviving_trades_pct": float((~hit).mean()),
            "stopped_trades_pct": float(hit.mean()),
            "prematurely_stopped_pct": float(premature.mean()),
            "estimated_average_r": frame.realized_r.mean(),
            "current_average_r": current.mean(),
            "expected_rr_difference": frame.realized_r.mean() - current.mean(),
            "method": "Isolated path counterfactual through the end of the retained series; no post-exit setups, portfolio interactions, or research matrix rerun.",
        })
    return pd.DataFrame(rows)


def _bars_for_trade(trade):
    # The replay collector stores the bars lazily in the process-local registry.
    bars = _BAR_REGISTRY.get((trade["asset"], trade["timeframe"]))
    if bars is None:
        raise KeyError(f"bars not registered for {trade['asset']} {trade['timeframe']}")
    return bars


_BAR_REGISTRY: dict[tuple[str, str], pd.DataFrame] = {}


def _break_even_analysis(trades):
    rows = []
    baseline = pd.DataFrame([_simulate_policy(trade, _bars_for_trade(trade), 1.02, CURRENT_POST_TP1) for _, trade in trades.iterrows()])
    for name, ratio in BREAK_EVEN_POLICIES:
        frame = pd.DataFrame([_simulate_policy(trade, _bars_for_trade(trade), 1.02, ratio) for _, trade in trades.iterrows()])
        actual = baseline["realized_r"].reset_index(drop=True)
        estimated = frame["realized_r"].reset_index(drop=True)
        changed = (abs(estimated - actual) > 1e-9)
        rows.append({
            "policy": name, "post_tp1_fib_ratio": ratio, "trades": len(frame),
            "changed_outcome_count": int(changed.sum()), "changed_outcome_pct": float(changed.mean()),
            "improved_outcome_count": int((estimated > actual + 1e-9).sum()),
            "worsened_outcome_count": int((estimated < actual - 1e-9).sum()),
            "estimated_average_r": estimated.mean(), "current_average_r": actual.mean(),
            "estimated_average_r_difference": (estimated - actual).mean(),
            "method": "Isolated path counterfactual through the end of the retained series using fixed Profile-B targets; no post-exit setups or new TP profile backtest.",
        })
    return pd.DataFrame(rows)


def _improvement_ranking(trades, stop_analysis, break_even, exit_efficiency):
    current = stop_analysis.loc[stop_analysis.stop_fib_ratio == 1.02, "current_average_r"].iloc[0] if not stop_analysis.empty else np.nan
    stop_best = stop_analysis.iloc[stop_analysis["estimated_average_r"].argmax()] if not stop_analysis.empty else None
    be_best = break_even.iloc[break_even["estimated_average_r_difference"].argmax()] if not break_even.empty else None
    tp_summary = exit_efficiency[exit_efficiency.scope == "mathematical_target_summary"]
    tp_gain = float(tp_summary["estimated_tp_allocation_improvement_r"].iloc[0]) if not tp_summary.empty else np.nan
    rows = [
        {"opportunity": "Break-even / post-TP1 stop logic", "estimated_improvement": float(be_best["estimated_average_r_difference"]) if be_best is not None else np.nan, "evidence": "Path-only comparison of no move, break-even, Fib 0.786 and current Fib 0.88."},
        {"opportunity": "TP allocation", "estimated_improvement": tp_gain, "evidence": "Mathematical target-EV allocation estimate; no TP profile was backtested."},
        {"opportunity": "Stop placement", "estimated_improvement": float(stop_best["expected_rr_difference"]) if stop_best is not None else np.nan, "evidence": "Path-only comparison among Fib 1.00–1.05; requires a proper frozen replay before implementation."},
        {"opportunity": "Trend filter", "estimated_improvement": np.nan, "evidence": "No controlled V5 estimate exists; prior research evidence is insufficient for a numeric claim."},
        {"opportunity": "ATR filter", "estimated_improvement": np.nan, "evidence": "V4 runner profiles were weaker in aggregate and no new ATR optimization was run."},
        {"opportunity": "Entry", "estimated_improvement": np.nan, "evidence": "V3 entry sensitivity is already complete; further entry optimization is not the current bottleneck."},
    ]
    rows.sort(key=lambda row: -(row["estimated_improvement"] if pd.notna(row["estimated_improvement"]) else -np.inf))
    for row in rows:
        row["baseline_average_realized_r"] = current
        row["method"] = "Descriptive ranking, not a promise of out-of-sample performance."
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return pd.DataFrame(rows)


def _write_report(path, trades, conditional, efficiency, stops, break_even, importance, pairs):
    conditional_html = conditional.to_html(index=False)
    efficiency_html = efficiency.to_html(index=False)
    stops_html = stops.to_html(index=False)
    break_even_html = break_even.to_html(index=False)
    importance_html = importance.to_html(index=False)
    p_tp1 = conditional.loc[conditional.metric == "P(TP2 | TP1)", "probability"].iloc[0]
    stop_before = conditional.loc[conditional.metric == "P(Stop before TP1)", "probability"].iloc[0]
    best = importance.iloc[0]
    text = f"""
    <p>This is a fixed-baseline path analysis of Strategy_V4, entry Fib 0.900, TP Profile B. It covers {len(trades):,} executed trades across {len(pairs)} retained asset/timeframe series and the existing distance/move configurations. No new parameter optimization or TP profile backtest was run.</p>
    <p>TP2 followed TP1 on {p_tp1:.1%} of TP1-reaching trades. Stop-before-TP1 occurred on {stop_before:.1%} of all executed trades. Average realized and unrealized profit, target-level mathematical EVs, and profit left on the table are in the exit-efficiency table.</p>
    <p>The largest remaining measured opportunity is {best.opportunity}. Its path-only estimate is {best.estimated_improvement:.4f} R where available; this is a prioritization signal, not an out-of-sample forecast.</p>
    <p>Recommendation: V6 should focus on one frozen break-even research direction—test the post-TP1 stop policy while keeping the initial Fib 1.02 stop, entry, lifecycle, TP Profile B, costs, and execution assumptions frozen.</p>
    """
    html = f"<html><body><h1>V5 Trade Path Analysis</h1>{text}<h2>Conditional probabilities</h2>{conditional_html}<h2>Exit efficiency</h2>{efficiency_html}<h2>Stop analysis</h2>{stops_html}<h2>Break-even analysis</h2>{break_even_html}<h2>Improvement ranking</h2>{importance_html}</body></html>"
    path.write_text(html, encoding="utf-8")
