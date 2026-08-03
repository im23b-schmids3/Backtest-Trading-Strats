from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .artifacts import sha256_value

NEW_YORK = ZoneInfo("America/New_York")


def alpha_proxy_rules() -> dict[str, Any]:
    return {
        "rules_version": "alpha-mbt-proxy-sealed-1",
        "retrieval_date": "2026-08-03",
        "sources": [
            {
                "source": "sealed ImbalanceVWAPRide exploratory execution contract",
                "covers": ["account thresholds", "consistency", "payout mechanics", "session constraints"],
            },
            {
                "source": "local verified BTCUSDT research cost convention",
                "covers": ["conservative fee and slippage assumptions"],
            },
        ],
        "instrument_mapping": "BTCUSDT_TO_MBT_PROXY_ONLY",
        "mbt_contract_btc": "0.1",
        "mbt_tick_points": "5",
        "mbt_tick_value_usd": "0.50",
        "contracts": 1,
        "starting_balance_usd": "25000",
        "evaluation_target_usd": "1500",
        "maximum_loss_limit_usd": "1000",
        "daily_loss_guard_usd": "500",
        "evaluation_consistency_rule": None,
        "qualified_consistency_fraction": "0.40",
        "qualified_minimum_200_days": 5,
        "withdrawal_profit_fraction": "0.50",
        "withdrawal_minimum_usd": "200",
        "withdrawal_maximum_usd": "1000",
        "trader_share_fraction": "0.90",
        "commission_per_side_usd": "1.25",
        "slippage_ticks_per_side": 1,
        "flat_time_et": "16:20",
        "trading_day_start_et": "18:00",
        "bootstrap_paths": 10_000,
        "bootstrap_block_days": 5,
        "maximum_path_days": 250,
        "limitations": [
            "This is a BTCUSDT-to-MBT economic proxy, not CME execution evidence.",
            "This is not Alpha Futures compliance proof and is not a rule attestation.",
            "Fees are a versioned conservative local assumption because the sealed source does not provide a fee schedule.",
            "No weekend or maintenance-window entries are admitted; all positions are flat by 16:20 America/New_York.",
        ],
        "confirmation_evidence": False,
        "external_holdout_required": True,
        "optimization_claimed": False,
    }


def _dt(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def is_mbt_entry_available(timestamp: datetime | str) -> bool:
    local = _dt(timestamp).astimezone(NEW_YORK)
    weekday, local_time = local.weekday(), local.timetz().replace(tzinfo=None)
    if weekday == 5:
        return False
    if weekday == 6:
        return local_time >= time(18, 0)
    if weekday == 4:
        return local_time <= time(16, 20)
    return local_time <= time(16, 20) or local_time >= time(18, 0)


def mbt_session_close(timestamp: datetime | str) -> datetime:
    local = _dt(timestamp).astimezone(NEW_YORK)
    close_date = local.date() + (timedelta(days=1) if local.timetz().replace(tzinfo=None) >= time(18, 0) else timedelta())
    return datetime.combine(close_date, time(16, 20), tzinfo=NEW_YORK)


def _round_points(value: Decimal, *, up: bool) -> Decimal:
    tick = Decimal("5")
    return (value / tick).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR) * tick


def map_btc_trades_to_mbt(
    trades: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    *,
    commission_multiplier: Decimal = Decimal("1"),
    extra_slippage_ticks: int = 0,
    degradation: Decimal = Decimal("0"),
) -> dict[str, Any]:
    rules = alpha_proxy_rules()
    bar_rows = sorted(bars, key=lambda item: _dt(item["bar_start_utc"]))
    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda item: item["entry_timestamp"]):
        entry_time = _dt(trade["entry_timestamp"])
        if not is_mbt_entry_available(entry_time):
            excluded.append({"trade_id": trade["trade_id"], "reason": "OUTSIDE_MBT_AVAILABILITY"})
            continue
        direction = trade["direction"]
        sign = Decimal("1") if direction == "LONG" else Decimal("-1")
        reference_entry = Decimal(str(trade.get("reference_entry_price", trade["entry_price"])))
        reference_exit = Decimal(str(trade.get("reference_exit_price", trade["exit_price"])))
        exit_time = _dt(trade["exit_timestamp"])
        deadline = mbt_session_close(entry_time)
        if exit_time.astimezone(NEW_YORK) > deadline:
            candidates = [
                bar
                for bar in bar_rows
                if entry_time <= _dt(bar["bar_start_utc"]) and _dt(bar["bar_end_utc"]).astimezone(NEW_YORK) <= deadline
            ]
            if not candidates:
                excluded.append({"trade_id": trade["trade_id"], "reason": "NO_1620_ET_FLAT_BAR"})
                continue
            reference_exit = Decimal(str(candidates[-1]["close"]))
            exit_time = _dt(candidates[-1]["bar_end_utc"])
        slip_ticks = int(rules["slippage_ticks_per_side"]) + extra_slippage_ticks
        entry = _round_points(reference_entry, up=direction == "LONG") + sign * Decimal("5") * slip_ticks
        exit_price = _round_points(reference_exit, up=direction == "SHORT") - sign * Decimal("5") * slip_ticks
        gross = sign * (reference_exit - reference_entry) * Decimal("0.1")
        slippage = (abs(entry - reference_entry) + abs(exit_price - reference_exit)) * Decimal("0.1")
        commission = Decimal(rules["commission_per_side_usd"]) * Decimal("2") * commission_multiplier
        net = gross - slippage - commission
        if degradation:
            net = net * (Decimal("1") - degradation) if net > 0 else net * (Decimal("1") + degradation)
        mapped.append(
            {
                "proxy_trade_id": f"mbt-{trade['trade_id']}",
                "source_trade_id": trade["trade_id"],
                "direction": direction,
                "entry_timestamp": entry_time.isoformat(),
                "exit_timestamp": exit_time.isoformat(),
                "trading_day": mbt_session_close(entry_time).date().isoformat(),
                "reference_entry_points": str(reference_entry),
                "entry_points": str(entry),
                "reference_exit_points": str(reference_exit),
                "exit_points": str(exit_price),
                "gross_pnl_usd": str(gross),
                "slippage_usd": str(slippage),
                "commission_usd": str(commission),
                "net_pnl_usd": str(net),
                "contract": "MBT",
                "contracts": 1,
            }
        )
    return {"trades": mapped, "excluded": excluded, "rules_version": rules["rules_version"]}


def daily_proxy_pnl(trades: list[dict[str, Any]]) -> list[tuple[str, float]]:
    values: dict[str, Decimal] = defaultdict(Decimal)
    for trade in trades:
        values[str(trade["trading_day"])] += Decimal(str(trade["net_pnl_usd"]))
    return [(day, float(value)) for day, value in sorted(values.items())]


def evaluation_outcome(sequence: list[float], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or alpha_proxy_rules()
    starting = float(rules["starting_balance_usd"])
    target = starting + float(rules["evaluation_target_usd"])
    mll = float(rules["maximum_loss_limit_usd"])
    dlg = float(rules["daily_loss_guard_usd"])
    equity = peak = starting
    floor = starting - mll
    for day, pnl in enumerate(sequence, 1):
        if pnl < -dlg:
            return {"outcome": "BREACH", "breach_reason": "DAILY_LOSS_GUARD", "days": day, "ending_balance": equity + pnl, "mll_floor": floor}
        equity += pnl
        if equity < floor:
            return {"outcome": "BREACH", "breach_reason": "MAXIMUM_LOSS_LIMIT", "days": day, "ending_balance": equity, "mll_floor": floor}
        peak = max(peak, equity)
        floor = max(floor, peak - mll)
        if equity >= target:
            return {"outcome": "PASS", "breach_reason": None, "days": day, "ending_balance": equity, "mll_floor": floor}
    return {"outcome": "UNFINISHED", "breach_reason": None, "days": len(sequence), "ending_balance": equity, "mll_floor": floor}


def block_bootstrap(
    values: list[float], *, paths: int, path_days: int, block_days: int, seed: int
) -> np.ndarray:
    if not values:
        return np.zeros((paths, path_days), dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty((paths, path_days), dtype=np.float64)
    for path in range(paths):
        cursor = 0
        while cursor < path_days:
            start = int(rng.integers(0, len(source)))
            take = min(block_days, path_days - cursor)
            indices = (start + np.arange(take)) % len(source)
            output[path, cursor : cursor + take] = source[indices]
            cursor += take
    return output


def qualified_outcome(sequence: list[float], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or alpha_proxy_rules()
    starting = float(rules["starting_balance_usd"])
    equity = peak = starting
    floor = starting - float(rules["maximum_loss_limit_usd"])
    dlg = float(rules["daily_loss_guard_usd"])
    consistency = float(rules["qualified_consistency_fraction"])
    qualifying_days = 0
    positive_total = 0.0
    best_day = 0.0
    withdrawals: list[dict[str, Any]] = []
    blocked_consistency_days = 0
    for day, pnl in enumerate(sequence, 1):
        if pnl < -dlg:
            return {"outcome": "BREACH", "breach_reason": "DAILY_LOSS_GUARD", "days": day, "withdrawals": withdrawals, "second_payout_achieved": len(withdrawals) >= 2, "ending_balance": equity + pnl, "mll_floor": floor}
        equity += pnl
        if equity < floor:
            return {"outcome": "BREACH", "breach_reason": "MAXIMUM_LOSS_LIMIT", "days": day, "withdrawals": withdrawals, "second_payout_achieved": len(withdrawals) >= 2, "ending_balance": equity, "mll_floor": floor}
        peak = max(peak, equity)
        floor = max(floor, peak - float(rules["maximum_loss_limit_usd"]))
        if pnl > 0:
            positive_total += pnl
            best_day = max(best_day, pnl)
        if pnl >= 200:
            qualifying_days += 1
        profit = equity - starting
        consistency_ok = positive_total > 0 and best_day <= consistency * positive_total
        if qualifying_days >= int(rules["qualified_minimum_200_days"]) and profit >= float(rules["withdrawal_minimum_usd"]):
            if not consistency_ok:
                blocked_consistency_days += 1
            else:
                gross = min(float(rules["withdrawal_maximum_usd"]), max(float(rules["withdrawal_minimum_usd"]), profit * float(rules["withdrawal_profit_fraction"])))
                share = gross * float(rules["trader_share_fraction"])
                equity -= gross
                withdrawals.append({"day": day, "gross_withdrawal": gross, "trader_share": share, "balance_after": equity, "buffer_above_mll": equity - floor})
                qualifying_days = 0
                positive_total = best_day = 0.0
                if len(withdrawals) >= 2:
                    return {"outcome": "SECOND_PAYOUT", "breach_reason": None, "days": day, "withdrawals": withdrawals, "second_payout_achieved": True, "ending_balance": equity, "mll_floor": floor, "blocked_consistency_days": blocked_consistency_days}
    return {"outcome": "SURVIVED_UNFINISHED", "breach_reason": None, "days": len(sequence), "withdrawals": withdrawals, "second_payout_achieved": len(withdrawals) >= 2, "ending_balance": equity, "mll_floor": floor, "blocked_consistency_days": blocked_consistency_days}


def _evaluation_distribution(daily_values: list[float], *, paths: int, seed: int, rules: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    samples = block_bootstrap(
        daily_values,
        paths=paths,
        path_days=int(rules["maximum_path_days"]),
        block_days=int(rules["bootstrap_block_days"]),
        seed=seed,
    )
    outcomes = [evaluation_outcome(list(row), rules) for row in samples]
    counts = Counter(item["outcome"] for item in outcomes)
    breaches = Counter(item["breach_reason"] for item in outcomes if item["outcome"] == "BREACH")
    passed_days = [item["days"] for item in outcomes if item["outcome"] == "PASS"]
    report = {
        "paths": paths,
        "pass_paths": counts["PASS"],
        "breach_paths": counts["BREACH"],
        "unfinished_paths": counts["UNFINISHED"],
        "pass_probability": counts["PASS"] / paths,
        "breach_probability": counts["BREACH"] / paths,
        "unfinished_probability": counts["UNFINISHED"] / paths,
        "mll_breaches": breaches["MAXIMUM_LOSS_LIMIT"],
        "dlg_breaches": breaches["DAILY_LOSS_GUARD"],
        "median_days_to_pass": float(np.median(passed_days)) if passed_days else None,
        "rebills": counts["BREACH"],
        "resets": counts["BREACH"],
        "modeled_rebill_cost_usd": 0,
        "cost_limitation": "No evaluation/rebill fee was supplied by the sealed contract; account fees are not invented.",
    }
    return report, samples, outcomes


def run_alpha_proxy(
    trades: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    *,
    paths: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    if paths < 10_000:
        raise ValueError("sealed Alpha proxy requires at least 10,000 paths")
    rules = alpha_proxy_rules()
    mapped = map_btc_trades_to_mbt(trades, bars)
    daily = daily_proxy_pnl(mapped["trades"])
    values = [value for _, value in daily]
    resolved_seed = seed if seed is not None else int(sha256_value({"trade_ids": [item["source_trade_id"] for item in mapped["trades"]], "rules": rules["rules_version"]})[:16], 16)
    chronological = evaluation_outcome(values, rules)
    evaluation, samples, outcomes = _evaluation_distribution(values, paths=paths, seed=resolved_seed, rules=rules)
    evaluation.update(
        {
            "chronological": chronological,
            "source_trade_count": len(trades),
            "mapped_trade_count": len(mapped["trades"]),
            "excluded_trade_count": len(mapped["excluded"]),
            "source_day_count": len(daily),
            "pass_probability_insufficient_sample": len(mapped["trades"]) < 30,
            "deterministic_seed": resolved_seed,
            "block_days": rules["bootstrap_block_days"],
        }
    )
    qualified_outcomes: list[dict[str, Any]] = []
    for row, outcome in zip(samples, outcomes, strict=True):
        if outcome["outcome"] == "PASS":
            start = min(int(outcome["days"]), len(row))
            continuation = list(row[start:]) + list(row[:start])
            qualified_outcomes.append(qualified_outcome(continuation, rules))
    qcounts = Counter(item["outcome"] for item in qualified_outcomes)
    withdrawals = [withdrawal for item in qualified_outcomes for withdrawal in item["withdrawals"]]
    qualified = {
        "simulated_paths": len(qualified_outcomes),
        "survival_paths": sum(item["outcome"] != "BREACH" for item in qualified_outcomes),
        "breach_paths": qcounts["BREACH"],
        "second_payout_paths": sum(item["second_payout_achieved"] for item in qualified_outcomes),
        "survival_probability": (sum(item["outcome"] != "BREACH" for item in qualified_outcomes) / len(qualified_outcomes)) if qualified_outcomes else None,
        "second_payout_probability": (sum(item["second_payout_achieved"] for item in qualified_outcomes) / len(qualified_outcomes)) if qualified_outcomes else None,
        "total_gross_withdrawals": sum(item["gross_withdrawal"] for item in withdrawals),
        "total_trader_share": sum(item["trader_share"] for item in withdrawals),
        "median_days_to_second_payout": float(np.median([item["days"] for item in qualified_outcomes if item["second_payout_achieved"]])) if any(item["second_payout_achieved"] for item in qualified_outcomes) else None,
        "minimum_payout_buffer": min((item["buffer_above_mll"] for item in withdrawals), default=None),
        "consistency_blocking_days": sum(int(item.get("blocked_consistency_days", 0)) for item in qualified_outcomes),
        "blocking_or_breach_paths": sum(item["outcome"] == "BREACH" or int(item.get("blocked_consistency_days", 0)) > 0 for item in qualified_outcomes),
    }
    sensitivities: dict[str, Any] = {}
    scenarios = {
        "one_extra_tick_each_side": {"extra_slippage_ticks": 1},
        "double_commission": {"commission_multiplier": Decimal("2")},
        "degradation_20_percent": {"degradation": Decimal("0.20")},
        "degradation_30_percent": {"degradation": Decimal("0.30")},
        "degradation_40_percent": {"degradation": Decimal("0.40")},
    }
    for name, options in scenarios.items():
        scenario = map_btc_trades_to_mbt(trades, bars, **options)
        scenario_values = [value for _, value in daily_proxy_pnl(scenario["trades"])]
        distribution, _, _ = _evaluation_distribution(scenario_values, paths=paths, seed=resolved_seed, rules=rules)
        sensitivities[name] = distribution
    return {
        "rules": rules,
        "mapping": mapped,
        "daily_pnl": [{"trading_day": day, "net_pnl_usd": value} for day, value in daily],
        "evaluation": evaluation,
        "qualified": qualified,
        "sensitivities": sensitivities,
    }
