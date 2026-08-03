from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from statistics import median
from typing import Any, Callable

from .strategy import (
    _d,
    _overlaps,
    _quantize,
    _stable_id,
    _ts,
    coarsen_footprints,
    compute_completed_bar_regimes,
    maximal_imbalance_sequences,
)
from .v3_models import (
    AUTHORIZED_MONTHS,
    COST_MODEL_VERSION,
    EARLY_SUBPERIOD_MONTHS,
    LATE_SUBPERIOD_MONTHS,
    ImbalanceVWAPRideV3Config,
)

ACTIVE_STATES = {"ACTIVE", "ARMED"}
TERMINAL_STATES = {"EXPIRED", "INVALIDATED", "TRADED", "SUPERSEDED"}


def simulate_long_trade(
    *,
    zone: dict[str, Any],
    signal_bar: dict[str, Any],
    entry_index: int,
    bars: list[dict[str, Any]],
    config: ImbalanceVWAPRideV3Config,
) -> tuple[str, dict[str, Any] | None]:
    """Execute only a long next-bar-open trade using actual-entry risk geometry."""

    if zone.get("direction") != "LONG":
        raise ValueError("V3 rejects every non-long zone before order simulation")
    if entry_index >= len(bars):
        return "NO_EXECUTABLE_ENTRY", None
    entry_bar = bars[entry_index]
    signal_start = _ts(signal_bar["bar_start_utc"])
    entry_start = _ts(entry_bar["bar_start_utc"])
    if entry_start != signal_start + timedelta(minutes=5) or entry_start.date() != signal_start.date():
        return "NO_EXECUTABLE_ENTRY", None

    quantity = (
        (config.quantity_btc / config.quantity_step).to_integral_value(rounding=ROUND_FLOOR)
        * config.quantity_step
    )
    if quantity < config.minimum_quantity:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None
    reference_entry = _d(entry_bar["open"])
    entry_price = _quantize(
        reference_entry + config.price_tick * config.market_slippage_ticks,
        config.price_tick,
        ROUND_CEILING,
    )
    zone_edge = _d(zone["top"])
    stop_price = _quantize(
        _d(zone["bottom"]) - config.stop_buffer_bins * config.bin_size_usd,
        config.price_tick,
        ROUND_FLOOR,
    )
    theoretical_risk_distance = zone_edge - stop_price
    actual_risk_distance = entry_price - stop_price
    entry_gap_distance = entry_price - zone_edge
    if actual_risk_distance <= 0:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None
    target_price = _quantize(
        entry_price + config.target_r_multiple * actual_risk_distance,
        config.price_tick,
        ROUND_CEILING,
    )
    target_distance = target_price - entry_price
    if target_distance <= 0:
        return "INVALID_UNPROFITABLE_TARGET", None

    day = entry_start.date()
    reference_exit = _d(entry_bar["close"])
    exit_price = reference_exit - config.price_tick * config.market_slippage_ticks
    exit_bar = entry_bar
    exit_reason = "UTC_DAY_FORCE_FLAT"
    exit_slippage_ticks = config.market_slippage_ticks
    same_bar_ambiguity = False
    for candidate in bars[entry_index:]:
        if _ts(candidate["bar_start_utc"]).date() != day:
            break
        hit_stop = _d(candidate["low"]) <= stop_price
        hit_target = _d(candidate["high"]) >= target_price
        if hit_stop:
            reference_exit = stop_price
            exit_price = stop_price - config.price_tick * config.stop_slippage_ticks
            exit_bar = candidate
            exit_reason = "STOP_FIRST_AMBIGUITY" if hit_target else "STOP"
            exit_slippage_ticks = config.stop_slippage_ticks
            same_bar_ambiguity = hit_target
            break
        if hit_target:
            reference_exit = exit_price = target_price
            exit_bar = candidate
            exit_reason = "TARGET"
            exit_slippage_ticks = 0
            break
        reference_exit = _d(candidate["close"])
        exit_price = reference_exit - config.price_tick * config.market_slippage_ticks
        exit_bar = candidate

    entry_notional = entry_price * quantity
    exit_notional = exit_price * quantity
    entry_fee = config.taker_fee_rate * entry_notional
    exit_fee = config.taker_fee_rate * exit_notional
    fees = entry_fee + exit_fee
    entry_slippage = abs(entry_price - reference_entry) * quantity
    exit_slippage = abs(exit_price - reference_exit) * quantity
    slippage = entry_slippage + exit_slippage
    total_costs = fees + slippage
    gross_pnl = (reference_exit - reference_entry) * quantity
    net_pnl = gross_pnl - total_costs
    theoretical_risk = theoretical_risk_distance * quantity
    actual_risk = actual_risk_distance * quantity
    payload = {
        "zone_id": zone["zone_id"],
        "sequence_lineage": list(zone["sequence_lineage"]),
        "direction": "LONG",
        "session_date": day.isoformat(),
        "signal_bar_start_timestamp": signal_start.isoformat(),
        "signal_timestamp": _ts(signal_bar["bar_end_utc"]).isoformat(),
        "entry_timestamp": entry_start.isoformat(),
        "exit_timestamp": _ts(exit_bar["bar_end_utc"]).isoformat(),
        "reference_entry_price": str(reference_entry),
        "entry_price": str(entry_price),
        "zone_edge_entry_price": str(zone_edge),
        "entry_gap_distance": str(entry_gap_distance),
        "entry_gap_usd": str(entry_gap_distance * quantity),
        "initial_stop_price": str(stop_price),
        "target_price": str(target_price),
        "target_distance": str(target_distance),
        "target_distance_usd": str(target_distance * quantity),
        "reference_exit_price": str(reference_exit),
        "exit_price": str(exit_price),
        "quantity_btc": str(quantity),
        "entry_notional_usd": str(entry_notional),
        "exit_notional_usd": str(exit_notional),
        "round_trip_notional_usd": str(entry_notional + exit_notional),
        "theoretical_zone_edge_risk_distance": str(theoretical_risk_distance),
        "theoretical_zone_edge_risk_usd": str(theoretical_risk),
        "actual_risk_distance": str(actual_risk_distance),
        "actual_risk_usd": str(actual_risk),
        "initial_risk_usd": str(actual_risk),
        "gross_risk_usd": str(actual_risk),
        "gross_pnl": str(gross_pnl),
        "entry_fee": str(entry_fee),
        "exit_fee": str(exit_fee),
        "fees": str(fees),
        "entry_slippage_cost": str(entry_slippage),
        "exit_slippage_cost": str(exit_slippage),
        "slippage_cost": str(slippage),
        "total_costs": str(total_costs),
        "cost_to_risk": str(total_costs / actual_risk),
        "net_pnl": str(net_pnl),
        "gross_r": str(gross_pnl / actual_risk),
        "net_r": str(net_pnl / actual_risk),
        "exit_reason": exit_reason,
        "same_bar_ambiguity": same_bar_ambiguity,
        "entry_slippage_ticks": config.market_slippage_ticks,
        "exit_slippage_ticks": exit_slippage_ticks,
        "cost_model_version": COST_MODEL_VERSION,
    }
    payload["trade_id"] = _stable_id(
        "trade-v3-long",
        {
            "zone_id": zone["zone_id"],
            "entry_timestamp": entry_start.isoformat(),
            "parameters": config.parameter_payload(),
        },
    )
    return "TRADE_EXECUTED", payload


def _terminal_zone(
    zone: dict[str, Any],
    state: str,
    bar: dict[str, Any],
    index: int,
    reason: str,
    *,
    timestamp_field: str = "bar_end_utc",
    superseded_by: str | None = None,
) -> None:
    if state not in TERMINAL_STATES:
        raise ValueError(f"unsupported V3 terminal state: {state}")
    zone["state"] = state
    zone["terminal_state"] = state
    zone["terminal_reason"] = reason
    zone["terminal_timestamp"] = _ts(bar[timestamp_field]).isoformat()
    zone["lifetime_bars"] = index - int(zone["created_index"])
    zone["superseded_by_zone_id"] = superseded_by


def run_imbalance_vwap_ride_v3(
    bars: list[dict[str, Any]],
    footprints: list[dict[str, Any]] | dict[datetime, list[dict[str, Any]]],
    config: ImbalanceVWAPRideV3Config = ImbalanceVWAPRideV3Config(),
    *,
    compliance_check: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]] | None = None,
) -> dict[str, Any]:
    enriched = compute_completed_bar_regimes(bars, config.vwap_slope_bars)
    footprint_by_bar = (
        footprints if isinstance(footprints, dict) else coarsen_footprints(footprints, config.bin_size_usd)
    )
    events: list[dict[str, Any]] = []
    zones: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    used_days: set[str] = set()
    proposed = invalid = non_executable = compliance_blocks = 0
    sequence_count = zones_created = vwap_qualified = move_away_confirmed = retest_triggers = 0

    def event(bar: dict[str, Any], state: str, **extra: Any) -> None:
        if extra.get("direction") not in {None, "LONG"}:
            raise AssertionError("V3 event attempted to emit a non-long direction")
        events.append(
            {
                "event_id": _stable_id(
                    "event-v3-long",
                    {
                        "variant": config.variant_id,
                        "timestamp": _ts(bar["bar_end_utc"]).isoformat(),
                        "state": state,
                        "ordinal": len(events),
                        **extra,
                    },
                ),
                "variant_id": config.variant_id,
                "timestamp": _ts(bar["bar_end_utc"]).isoformat(),
                "state": state,
                **extra,
            }
        )

    def active_zones() -> list[dict[str, Any]]:
        return [zone for zone in zones if zone["state"] in ACTIVE_STATES]

    for index, bar in enumerate(enriched):
        start = _ts(bar["bar_start_utc"])
        day = start.date().isoformat()
        for zone in list(active_zones()):
            if zone["direction"] != "LONG":
                raise AssertionError("V3 active-zone collection contains a non-long zone")
            if index >= int(zone["expiry_index"]):
                _terminal_zone(zone, "EXPIRED", bar, index, "TIME_EXPIRY")
                event(bar, "ZONE_EXPIRED", zone_id=zone["zone_id"], reason="TIME_EXPIRY")
                continue
            if _d(bar["close"]) < _d(zone["bottom"]):
                _terminal_zone(zone, "INVALIDATED", bar, index, "CLOSE_BELOW_IZ_BOTTOM")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="CLOSE_BELOW_IZ_BOTTOM")
                continue
            regime = bool(bar["long_vwap_regime"])
            if zone["vwap_qualified"] and not regime:
                _terminal_zone(zone, "INVALIDATED", bar, index, "VWAP_REGIME_LOSS")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="VWAP_REGIME_LOSS")
                continue
            if not zone["vwap_qualified"] and regime:
                zone["vwap_qualified"] = True
                zone["vwap_qualified_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                vwap_qualified += 1
                event(bar, "VWAP_QUALIFIED", zone_id=zone["zone_id"], direction="LONG")
            if not zone["vwap_qualified"] or index == int(zone["created_index"]):
                continue
            moved = _d(bar["low"]) > _d(zone["top"])
            if zone["state"] == "ACTIVE":
                zone["move_away_count"] = int(zone["move_away_count"]) + 1 if moved else 0
                if int(zone["move_away_count"]) >= config.move_away_bars:
                    zone["state"] = "ARMED"
                    zone["move_away_confirmed"] = True
                    zone["move_away_confirmed_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                    move_away_confirmed += 1
                    event(bar, "MOVE_AWAY_CONFIRMED", zone_id=zone["zone_id"], direction="LONG")

        sequences = maximal_imbalance_sequences(footprint_by_bar.get(start, []), config, "LONG")
        for sequence in sequences:
            sequence_count += 1
            event(bar, "IMBALANCE_SEQUENCE", sequence_id=sequence["sequence_id"], direction="LONG")
            qualified = bool(bar["long_vwap_regime"])
            zone = {
                "variant_id": config.variant_id,
                "direction": "LONG",
                "created_index": index,
                "created_timestamp": _ts(bar["bar_end_utc"]).isoformat(),
                "expiry_index": index + config.zone_expiry_bars,
                "bottom": sequence["bottom"],
                "top": sequence["top"],
                "buy_volume_btc": sequence["buy_volume_btc"],
                "sell_volume_btc": sequence["sell_volume_btc"],
                "total_volume_btc": sequence["total_volume_btc"],
                "delta_btc": sequence["delta_btc"],
                "sequence_lineage": [sequence["sequence_id"]],
                "state": "ACTIVE",
                "move_away_count": 0,
                "move_away_confirmed": False,
                "move_away_confirmed_timestamp": None,
                "retest_triggered": False,
                "retest_timestamp": None,
                "vwap_qualified": qualified,
                "vwap_qualified_timestamp": _ts(bar["bar_end_utc"]).isoformat() if qualified else None,
                "trade_count": 0,
                "terminal_state": None,
                "terminal_timestamp": None,
                "terminal_reason": None,
                "lifetime_bars": None,
                "superseded_by_zone_id": None,
            }
            zone["zone_id"] = _stable_id(
                "zone-v3-long",
                {
                    key: zone[key]
                    for key in (
                        "variant_id",
                        "direction",
                        "created_timestamp",
                        "bottom",
                        "top",
                        "sequence_lineage",
                    )
                },
            )
            zones_created += 1
            vwap_qualified += int(qualified)
            event(bar, "ZONE_CREATED", zone_id=zone["zone_id"], direction="LONG", vwap_qualified=qualified)
            overlaps = [candidate for candidate in active_zones() if _overlaps(candidate, zone)]
            zones.append(zone)
            if overlaps:
                survivor = sorted(overlaps, key=lambda item: (int(item["created_index"]), item["zone_id"]))[0]
                survivor["bottom"] = str(min(_d(survivor["bottom"]), _d(zone["bottom"])))
                survivor["top"] = str(max(_d(survivor["top"]), _d(zone["top"])))
                survivor["sequence_lineage"] = sorted(set(survivor["sequence_lineage"] + zone["sequence_lineage"]))
                survivor["buy_volume_btc"] = str(_d(survivor["buy_volume_btc"]) + _d(zone["buy_volume_btc"]))
                survivor["sell_volume_btc"] = str(_d(survivor["sell_volume_btc"]) + _d(zone["sell_volume_btc"]))
                survivor["total_volume_btc"] = str(_d(survivor["buy_volume_btc"]) + _d(survivor["sell_volume_btc"]))
                survivor["delta_btc"] = str(_d(survivor["buy_volume_btc"]) - _d(survivor["sell_volume_btc"]))
                survivor["expiry_index"] = max(int(survivor["expiry_index"]), int(zone["expiry_index"]))
                survivor["vwap_qualified"] = bool(survivor["vwap_qualified"] or qualified)
                _terminal_zone(
                    zone,
                    "SUPERSEDED",
                    bar,
                    index,
                    "SAME_DIRECTION_MERGE",
                    superseded_by=survivor["zone_id"],
                )
                event(
                    bar,
                    "ZONE_SUPERSEDED",
                    zone_id=zone["zone_id"],
                    reason="SAME_DIRECTION_MERGE",
                    superseded_by=survivor["zone_id"],
                )

        candidates = sorted(
            active_zones(),
            key=lambda item: (int(item["created_index"]), item["zone_id"]),
            reverse=True,
        )
        for overflow in candidates[config.maximum_active_zones :]:
            _terminal_zone(overflow, "SUPERSEDED", bar, index, "ACTIVE_ZONE_CAP")
            event(bar, "ZONE_SUPERSEDED", zone_id=overflow["zone_id"], reason="ACTIVE_ZONE_CAP")

        triggered = [
            zone
            for zone in active_zones()
            if zone["state"] == "ARMED"
            and bool(bar["long_vwap_regime"])
            and _d(bar["low"]) <= _d(zone["top"]) <= _d(bar["high"])
            and _d(bar["close"]) >= _d(zone["top"])
        ]
        if triggered:
            zone = sorted(
                triggered,
                key=lambda item: (int(item["created_index"]), item["zone_id"]),
                reverse=True,
            )[0]
            zone["retest_triggered"] = True
            zone["retest_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
            retest_triggers += 1
            proposed += 1
            event(bar, "RETEST_TRIGGER", zone_id=zone["zone_id"], direction="LONG")
            event(bar, "PROPOSED_SETUP", zone_id=zone["zone_id"], direction="LONG")
            if day in used_days:
                compliance_blocks += 1
                _terminal_zone(zone, "INVALIDATED", bar, index, "DAILY_TRADE_CAP")
                event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason="DAILY_TRADE_CAP")
            else:
                allowed, reason = compliance_check(zone, bar) if compliance_check else (True, None)
                if not allowed:
                    compliance_blocks += 1
                    resolved = reason or "COMPLIANCE_BLOCK"
                    _terminal_zone(zone, "INVALIDATED", bar, index, resolved)
                    event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason=resolved)
                else:
                    state, trade = simulate_long_trade(
                        zone=zone,
                        signal_bar=bar,
                        entry_index=index + 1,
                        bars=enriched,
                        config=config,
                    )
                    if state == "NO_EXECUTABLE_ENTRY":
                        non_executable += 1
                        _terminal_zone(zone, "INVALIDATED", bar, index, state)
                        event(bar, state, zone_id=zone["zone_id"])
                    elif state != "TRADE_EXECUTED":
                        invalid += 1
                        _terminal_zone(zone, "INVALIDATED", bar, index, state)
                        event(bar, "INVALIDATED_SETUP", zone_id=zone["zone_id"], reason=state)
                    else:
                        assert trade is not None
                        zone["trade_count"] = int(zone["trade_count"]) + 1
                        if zone["trade_count"] > config.maximum_trades_per_zone:
                            raise AssertionError("V3 maximum trades per zone was exceeded")
                        trades.append(trade)
                        used_days.add(day)
                        entry_bar = enriched[index + 1]
                        _terminal_zone(
                            zone,
                            "TRADED",
                            entry_bar,
                            index + 1,
                            "POST_EXECUTION",
                            timestamp_field="bar_start_utc",
                        )
                        event(
                            bar,
                            "TRADE_EXECUTED",
                            zone_id=zone["zone_id"],
                            trade_id=trade["trade_id"],
                            direction="LONG",
                        )

    funnel = {
        "formula": "proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades",
        "proposed_setups": proposed,
        "invalid_setups": invalid,
        "non_executable_setups": non_executable,
        "compliance_blocks": compliance_blocks,
        "executed_trades": len(trades),
    }
    funnel["components_total"] = invalid + non_executable + compliance_blocks + len(trades)
    funnel["reconciles"] = proposed == funnel["components_total"]
    if not funnel["reconciles"]:
        raise AssertionError("V3 funnel failed exact reconciliation")
    if any(item.get("direction") != "LONG" for item in trades + zones):
        raise AssertionError("V3 emitted a non-long trade or zone")
    if any(item.get("direction") not in {None, "LONG"} for item in events):
        raise AssertionError("V3 emitted a short event")
    metrics = summarize_strategy_result(
        enriched,
        trades,
        funnel,
        {
            "imbalance_sequences": sequence_count,
            "zones_created": zones_created,
            "vwap_qualified_zones": vwap_qualified,
            "move_away_confirmed_zones": move_away_confirmed,
            "retest_triggers": retest_triggers,
            "terminal_expired": sum(zone["state"] == "EXPIRED" for zone in zones),
            "terminal_invalidated": sum(zone["state"] == "INVALIDATED" for zone in zones),
            "terminal_traded": sum(zone["state"] == "TRADED" for zone in zones),
            "terminal_superseded": sum(zone["state"] == "SUPERSEDED" for zone in zones),
            "active_zones_at_end": sum(zone["state"] in ACTIVE_STATES for zone in zones),
        },
    )
    return {
        "variant_id": config.variant_id,
        "parameters": config.parameter_payload(),
        "events": events,
        "zones": zones,
        "trades": trades,
        "funnel": funnel,
        "metrics": metrics,
        "subperiods": summarize_subperiods(trades, enriched),
    }


def _profit_factor(values: list[Decimal]) -> str | None:
    gains = sum((value for value in values if value > 0), Decimal())
    losses = -sum((value for value in values if value < 0), Decimal())
    if losses > 0:
        return str(gains / losses)
    return "Infinity" if gains > 0 else None


def _drawdown(values: list[Decimal]) -> Decimal:
    equity = peak = maximum = Decimal()
    for value in values:
        equity += value
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    gross = [_d(item["gross_pnl"]) for item in trades]
    net = [_d(item["net_pnl"]) for item in trades]
    gross_rs = [_d(item["gross_r"]) for item in trades]
    net_rs = [_d(item["net_r"]) for item in trades]
    return {
        "executed_trades": len(trades),
        "long_trades": len(trades),
        "short_trades": 0,
        "gross_pnl": str(sum(gross, Decimal())),
        "net_pnl": str(sum(net, Decimal())),
        "gross_profit_factor": _profit_factor(gross),
        "net_profit_factor": _profit_factor(net),
        "average_gross_r": str(sum(gross_rs, Decimal()) / len(gross_rs)) if gross_rs else "0",
        "average_net_r": str(sum(net_rs, Decimal()) / len(net_rs)) if net_rs else "0",
        "fees": str(sum((_d(item["fees"]) for item in trades), Decimal())),
        "slippage_cost": str(sum((_d(item["slippage_cost"]) for item in trades), Decimal())),
        "total_costs": str(sum((_d(item["total_costs"]) for item in trades), Decimal())),
        "maximum_drawdown": str(_drawdown(net)),
    }


def summarize_subperiods(
    trades: list[dict[str, Any]],
    bars: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, months in (
        ("AUG_TO_OCT_2024", EARLY_SUBPERIOD_MONTHS),
        ("NOV_2024_TO_JAN_2025", LATE_SUBPERIOD_MONTHS),
    ):
        subset = [item for item in trades if str(item["entry_timestamp"])[:7] in months]
        output[name] = {
            "months": list(months),
            "five_minute_bar_count": sum(
                str(bar.get("month") or _ts(bar["bar_start_utc"]).strftime("%Y-%m")) in months
                for bar in bars
            ),
            **_trade_summary(subset),
        }
    return output


def summarize_strategy_result(
    bars: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    funnel: dict[str, Any],
    funnel_stages: dict[str, int],
) -> dict[str, Any]:
    summary = _trade_summary(trades)
    gross_rs = [_d(item["gross_r"]) for item in trades]
    net_rs = [_d(item["net_r"]) for item in trades]
    risks = [_d(item["actual_risk_usd"]) for item in trades]
    cost_ratios = [_d(item["cost_to_risk"]) for item in trades]
    net_values = [_d(item["net_pnl"]) for item in trades]
    current_losing = longest_losing = 0
    for value in net_values:
        current_losing = current_losing + 1 if value <= 0 else 0
        longest_losing = max(longest_losing, current_losing)

    months_present = sorted(
        {str(bar.get("month") or _ts(bar["bar_start_utc"]).strftime("%Y-%m")) for bar in bars}
    )
    monthly: dict[str, dict[str, Any]] = {}
    for month in months_present:
        subset = [item for item in trades if str(item["entry_timestamp"]).startswith(month)]
        monthly[month] = {
            "five_minute_bar_count": sum(
                str(bar.get("month") or _ts(bar["bar_start_utc"]).strftime("%Y-%m")) == month
                for bar in bars
            ),
            **_trade_summary(subset),
        }
    positive_months = [max(_d(item["net_pnl"]), Decimal()) for item in monthly.values()]
    positive_month_total = sum(positive_months, Decimal())
    maximum_month_contribution = (
        max(positive_months, default=Decimal()) / positive_month_total
        if positive_month_total
        else Decimal("1")
    )
    positive_trades = sorted((value for value in net_values if value > 0), reverse=True)
    positive_total = sum(positive_trades, Decimal())
    best_five = sum(positive_trades[:5], Decimal()) / positive_total if positive_total else Decimal("1")
    return {
        **funnel_stages,
        **{key: funnel[key] for key in (
            "proposed_setups",
            "invalid_setups",
            "non_executable_setups",
            "compliance_blocks",
            "executed_trades",
        )},
        **summary,
        "long_only_reconciliation": {
            "executed_trades": len(trades),
            "long_trades": len(trades),
            "short_trades": 0,
            "short_setups": 0,
            "short_orders": 0,
            "short_fills": 0,
            "short_pnl": "0",
            "reconciles": all(item["direction"] == "LONG" for item in trades),
        },
        "median_gross_r": str(median(gross_rs)) if gross_rs else "0",
        "median_net_r": str(median(net_rs)) if net_rs else "0",
        "median_initial_risk_usd": str(median(risks)) if risks else "0",
        "gross_risk_usd": str(sum(risks, Decimal())),
        "median_cost_to_risk": str(median(cost_ratios)) if cost_ratios else "0",
        "cost_to_risk_share_over_10_percent": str(
            Decimal(sum(value > Decimal("0.10") for value in cost_ratios)) / len(cost_ratios)
        ) if cost_ratios else "0",
        "cost_to_risk_share_over_25_percent": str(
            Decimal(sum(value > Decimal("0.25") for value in cost_ratios)) / len(cost_ratios)
        ) if cost_ratios else "0",
        "cost_to_risk_share_over_50_percent": str(
            Decimal(sum(value > Decimal("0.50") for value in cost_ratios)) / len(cost_ratios)
        ) if cost_ratios else "0",
        "win_rate": str(Decimal(sum(value > 0 for value in net_values)) / len(net_values)) if net_values else "0",
        "longest_losing_streak": longest_losing,
        "months": monthly,
        "maximum_positive_month_contribution": str(maximum_month_contribution),
        "best_five_positive_pnl_contribution": str(best_five),
        "funnel_reconciliation": funnel,
        "same_bar_stop_first_count": sum(bool(item["same_bar_ambiguity"]) for item in trades),
        "forced_flat_count": sum(item["exit_reason"] == "UTC_DAY_FORCE_FLAT" for item in trades),
        "cost_model_version": COST_MODEL_VERSION,
    }
