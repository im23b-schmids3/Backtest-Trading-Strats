from __future__ import annotations

from collections import defaultdict
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
from .v2_models import COST_MODEL_VERSION, ImbalanceVWAPRideV2Config

ACTIVE_STATES = {"ACTIVE", "ARMED"}
TERMINAL_STATES = {"EXPIRED", "INVALIDATED", "TRADED", "SUPERSEDED"}


def simulate_trade(
    *,
    zone: dict[str, Any],
    signal_bar: dict[str, Any],
    entry_index: int,
    bars: list[dict[str, Any]],
    config: ImbalanceVWAPRideV2Config,
) -> tuple[str, dict[str, Any] | None]:
    """Simulate one V2 trade using the adverse quantized next-bar actual entry."""

    if entry_index >= len(bars):
        return "NO_EXECUTABLE_ENTRY", None
    entry_bar = bars[entry_index]
    signal_start = _ts(signal_bar["bar_start_utc"])
    entry_start = _ts(entry_bar["bar_start_utc"])
    if entry_start != signal_start + timedelta(minutes=5) or entry_start.date() != signal_start.date():
        return "NO_EXECUTABLE_ENTRY", None

    direction = str(zone["direction"])
    sign = Decimal("1") if direction == "LONG" else Decimal("-1")
    quantity = (
        (config.quantity_btc / config.quantity_step).to_integral_value(rounding=ROUND_FLOOR)
        * config.quantity_step
    )
    if quantity < config.minimum_quantity:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None

    reference_entry = _d(entry_bar["open"])
    slipped_entry = reference_entry + sign * config.price_tick * config.market_slippage_ticks
    entry_price = _quantize(
        slipped_entry,
        config.price_tick,
        ROUND_CEILING if direction == "LONG" else ROUND_FLOOR,
    )
    zone_edge = _d(zone["top"] if direction == "LONG" else zone["bottom"])
    raw_stop = (
        _d(zone["bottom"]) - config.stop_buffer_bins * config.bin_size_usd
        if direction == "LONG"
        else _d(zone["top"]) + config.stop_buffer_bins * config.bin_size_usd
    )
    stop_price = _quantize(
        raw_stop,
        config.price_tick,
        ROUND_FLOOR if direction == "LONG" else ROUND_CEILING,
    )
    theoretical_risk_distance = sign * (zone_edge - stop_price)
    actual_risk_distance = sign * (entry_price - stop_price)
    entry_gap_distance = sign * (entry_price - zone_edge)
    if actual_risk_distance <= 0:
        return "INVALID_ENTRY_GEOMETRY_OR_QUANTITY", None

    raw_target = entry_price + sign * config.target_r_multiple * actual_risk_distance
    target_price = _quantize(
        raw_target,
        config.price_tick,
        ROUND_CEILING if direction == "LONG" else ROUND_FLOOR,
    )
    target_distance = sign * (target_price - entry_price)
    if target_distance <= 0:
        return "INVALID_UNPROFITABLE_TARGET", None

    day = entry_start.date()
    reference_exit = _d(entry_bar["close"])
    exit_price = reference_exit - sign * config.price_tick * config.market_slippage_ticks
    exit_bar = entry_bar
    exit_reason = "UTC_DAY_FORCE_FLAT"
    exit_slippage_ticks = config.market_slippage_ticks
    same_bar_ambiguity = False
    for candidate in bars[entry_index:]:
        if _ts(candidate["bar_start_utc"]).date() != day:
            break
        hit_stop = (
            _d(candidate["low"]) <= stop_price
            if direction == "LONG"
            else _d(candidate["high"]) >= stop_price
        )
        hit_target = (
            _d(candidate["high"]) >= target_price
            if direction == "LONG"
            else _d(candidate["low"]) <= target_price
        )
        if hit_stop:
            reference_exit = stop_price
            exit_price = stop_price - sign * config.price_tick * config.stop_slippage_ticks
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
        exit_price = reference_exit - sign * config.price_tick * config.market_slippage_ticks
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
    gross_pnl = sign * (reference_exit - reference_entry) * quantity
    net_pnl = gross_pnl - total_costs
    theoretical_risk = theoretical_risk_distance * quantity
    actual_risk = actual_risk_distance * quantity
    target_value = target_distance * quantity
    gross_r = gross_pnl / actual_risk
    net_r = net_pnl / actual_risk
    cost_to_risk = total_costs / actual_risk
    payload = {
        "zone_id": zone["zone_id"],
        "sequence_lineage": list(zone["sequence_lineage"]),
        "direction": direction,
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
        "target_distance_usd": str(target_value),
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
        "cost_to_risk": str(cost_to_risk),
        "net_pnl": str(net_pnl),
        "gross_r": str(gross_r),
        "net_r": str(net_r),
        "exit_reason": exit_reason,
        "same_bar_ambiguity": same_bar_ambiguity,
        "entry_slippage_ticks": config.market_slippage_ticks,
        "exit_slippage_ticks": exit_slippage_ticks,
        "cost_model_version": COST_MODEL_VERSION,
    }
    payload["trade_id"] = _stable_id(
        "trade-v2",
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
        raise ValueError(f"unsupported terminal state: {state}")
    zone["state"] = state
    zone["terminal_state"] = state
    zone["terminal_reason"] = reason
    zone["terminal_timestamp"] = _ts(bar[timestamp_field]).isoformat()
    zone["lifetime_bars"] = index - int(zone["created_index"])
    zone["superseded_by_zone_id"] = superseded_by


def run_imbalance_vwap_ride_v2(
    bars: list[dict[str, Any]],
    footprints: list[dict[str, Any]] | dict[datetime, list[dict[str, Any]]],
    config: ImbalanceVWAPRideV2Config = ImbalanceVWAPRideV2Config(),
    *,
    compliance_check: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]] | None = None,
) -> dict[str, Any]:
    enriched = compute_completed_bar_regimes(bars, config.vwap_slope_bars)
    footprint_by_bar = (
        footprints
        if isinstance(footprints, dict)
        else coarsen_footprints(footprints, config.bin_size_usd)
    )
    events: list[dict[str, Any]] = []
    zones: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    used_days: set[str] = set()
    proposed = invalid = non_executable = compliance_blocks = 0
    sequence_count = zones_created = vwap_qualified = move_away_confirmed = retest_triggers = 0

    def event(bar: dict[str, Any], state: str, **extra: Any) -> None:
        events.append(
            {
                "event_id": _stable_id(
                    "event-v2",
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

    def active_zones(direction: str | None = None) -> list[dict[str, Any]]:
        return [
            zone
            for zone in zones
            if zone["state"] in ACTIVE_STATES and (direction is None or zone["direction"] == direction)
        ]

    for index, bar in enumerate(enriched):
        start = _ts(bar["bar_start_utc"])
        day = start.date().isoformat()
        for zone in list(active_zones()):
            direction = zone["direction"]
            if index >= int(zone["expiry_index"]):
                _terminal_zone(zone, "EXPIRED", bar, index, "TIME_EXPIRY")
                event(bar, "ZONE_EXPIRED", zone_id=zone["zone_id"], reason="TIME_EXPIRY")
                continue
            adverse = (
                _d(bar["close"]) < _d(zone["bottom"])
                if direction == "LONG"
                else _d(bar["close"]) > _d(zone["top"])
            )
            if adverse:
                _terminal_zone(zone, "INVALIDATED", bar, index, "ADVERSE_ZONE_BOUNDARY_CLOSE")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="ADVERSE_ZONE_BOUNDARY_CLOSE")
                continue
            regime = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
            if zone["vwap_qualified"] and not regime:
                _terminal_zone(zone, "INVALIDATED", bar, index, "VWAP_REGIME_LOSS")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="VWAP_REGIME_LOSS")
                continue
            if not zone["vwap_qualified"] and regime:
                zone["vwap_qualified"] = True
                zone["vwap_qualified_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                vwap_qualified += 1
                event(bar, "VWAP_QUALIFIED", zone_id=zone["zone_id"], direction=direction)
            if not zone["vwap_qualified"] or index == int(zone["created_index"]):
                continue
            moved = (
                _d(bar["low"]) > _d(zone["top"])
                if direction == "LONG"
                else _d(bar["high"]) < _d(zone["bottom"])
            )
            if zone["state"] == "ACTIVE":
                zone["move_away_count"] = int(zone["move_away_count"]) + 1 if moved else 0
                if int(zone["move_away_count"]) >= config.move_away_bars:
                    zone["state"] = "ARMED"
                    zone["move_away_confirmed"] = True
                    zone["move_away_confirmed_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                    move_away_confirmed += 1
                    event(bar, "MOVE_AWAY_CONFIRMED", zone_id=zone["zone_id"], direction=direction)

        for direction in ("LONG", "SHORT"):
            for sequence in maximal_imbalance_sequences(footprint_by_bar.get(start, []), config, direction):
                sequence_count += 1
                event(bar, "IMBALANCE_SEQUENCE", sequence_id=sequence["sequence_id"], direction=direction)
                qualified = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
                zone = {
                    "variant_id": config.variant_id,
                    "direction": direction,
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
                    "terminal_state": None,
                    "terminal_timestamp": None,
                    "terminal_reason": None,
                    "lifetime_bars": None,
                    "superseded_by_zone_id": None,
                }
                zone["zone_id"] = _stable_id(
                    "zone-v2",
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
                event(bar, "ZONE_CREATED", zone_id=zone["zone_id"], direction=direction, vwap_qualified=qualified)
                overlaps = [candidate for candidate in active_zones(direction) if _overlaps(candidate, zone)]
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

        opposing = active_zones()
        for left_index, left in enumerate(opposing):
            if left["state"] not in ACTIVE_STATES:
                continue
            for right in opposing[left_index + 1 :]:
                if right["state"] not in ACTIVE_STATES or right["direction"] == left["direction"]:
                    continue
                if not _overlaps(left, right):
                    continue
                left_rank = (
                    int(left["created_index"]),
                    _d(left["total_volume_btc"]),
                    abs(_d(left["delta_btc"])),
                    left["direction"],
                )
                right_rank = (
                    int(right["created_index"]),
                    _d(right["total_volume_btc"]),
                    abs(_d(right["delta_btc"])),
                    right["direction"],
                )
                loser, winner = (left, right) if left_rank < right_rank else (right, left)
                _terminal_zone(
                    loser,
                    "SUPERSEDED",
                    bar,
                    index,
                    "OPPOSITE_ZONE_SUPERSESSION",
                    superseded_by=winner["zone_id"],
                )
                event(
                    bar,
                    "ZONE_SUPERSEDED",
                    zone_id=loser["zone_id"],
                    reason="OPPOSITE_ZONE_SUPERSESSION",
                    superseded_by=winner["zone_id"],
                )

        for direction in ("LONG", "SHORT"):
            candidates = sorted(
                active_zones(direction),
                key=lambda item: (int(item["created_index"]), item["zone_id"]),
                reverse=True,
            )
            for overflow in candidates[config.maximum_active_zones_per_direction :]:
                _terminal_zone(overflow, "SUPERSEDED", bar, index, "ACTIVE_ZONE_CAP")
                event(bar, "ZONE_SUPERSEDED", zone_id=overflow["zone_id"], reason="ACTIVE_ZONE_CAP")

        triggered: list[dict[str, Any]] = []
        for zone in active_zones():
            if zone["state"] != "ARMED":
                continue
            direction = zone["direction"]
            regime = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
            retest = (
                _d(bar["low"]) <= _d(zone["top"]) <= _d(bar["high"])
                and _d(bar["close"]) >= _d(zone["top"])
                if direction == "LONG"
                else _d(bar["low"]) <= _d(zone["bottom"]) <= _d(bar["high"])
                and _d(bar["close"]) <= _d(zone["bottom"])
            )
            if regime and retest:
                triggered.append(zone)
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
            event(bar, "RETEST_TRIGGER", zone_id=zone["zone_id"], direction=zone["direction"])
            event(bar, "PROPOSED_SETUP", zone_id=zone["zone_id"], direction=zone["direction"])
            if day in used_days:
                compliance_blocks += 1
                _terminal_zone(zone, "INVALIDATED", bar, index, "DAILY_TRADE_CAP")
                event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason="DAILY_TRADE_CAP")
            else:
                allowed, reason = compliance_check(zone, bar) if compliance_check else (True, None)
                if not allowed:
                    compliance_blocks += 1
                    resolved_reason = reason or "COMPLIANCE_BLOCK"
                    _terminal_zone(zone, "INVALIDATED", bar, index, resolved_reason)
                    event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason=resolved_reason)
                else:
                    state, trade = simulate_trade(
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
                            direction=zone["direction"],
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
        raise AssertionError("V2 strategy funnel failed exact reconciliation")
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
    }


def _profit_factor(values: list[Decimal]) -> str | None:
    gains = sum((value for value in values if value > 0), Decimal())
    losses = -sum((value for value in values if value < 0), Decimal())
    if losses > 0:
        return str(gains / losses)
    return "Infinity" if gains > 0 else None


def _direction_metrics(trades: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    subset = [trade for trade in trades if trade["direction"] == direction]
    gross = [_d(trade["gross_pnl"]) for trade in subset]
    net = [_d(trade["net_pnl"]) for trade in subset]
    gross_rs = [_d(trade["gross_r"]) for trade in subset]
    net_rs = [_d(trade["net_r"]) for trade in subset]
    return {
        "direction": direction,
        "executed_trades": len(subset),
        "gross_pnl": str(sum(gross, Decimal())),
        "net_pnl": str(sum(net, Decimal())),
        "gross_profit_factor": _profit_factor(gross),
        "net_profit_factor": _profit_factor(net),
        "average_gross_r": str(sum(gross_rs, Decimal()) / len(gross_rs)) if gross_rs else "0",
        "average_net_r": str(sum(net_rs, Decimal()) / len(net_rs)) if net_rs else "0",
    }


def summarize_strategy_result(
    bars: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    funnel: dict[str, Any],
    funnel_stages: dict[str, int],
) -> dict[str, Any]:
    gross_values = [_d(item["gross_pnl"]) for item in trades]
    net_values = [_d(item["net_pnl"]) for item in trades]
    gross_rs = [_d(item["gross_r"]) for item in trades]
    net_rs = [_d(item["net_r"]) for item in trades]
    risks = [_d(item["actual_risk_usd"]) for item in trades]
    cost_ratios = [_d(item["cost_to_risk"]) for item in trades]
    equity = peak = maximum_drawdown = Decimal()
    current_losing = longest_losing = 0
    for value in net_values:
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
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
            "executed_trades": len(subset),
            "gross_pnl": str(sum((_d(item["gross_pnl"]) for item in subset), Decimal())),
            "net_pnl": str(sum((_d(item["net_pnl"]) for item in subset), Decimal())),
            "fees": str(sum((_d(item["fees"]) for item in subset), Decimal())),
            "slippage_cost": str(sum((_d(item["slippage_cost"]) for item in subset), Decimal())),
            "total_costs": str(sum((_d(item["total_costs"]) for item in subset), Decimal())),
            "long_trades": sum(item["direction"] == "LONG" for item in subset),
            "short_trades": sum(item["direction"] == "SHORT" for item in subset),
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
    fees = sum((_d(item["fees"]) for item in trades), Decimal())
    slippage = sum((_d(item["slippage_cost"]) for item in trades), Decimal())
    total_costs = fees + slippage
    long_count = sum(item["direction"] == "LONG" for item in trades)
    short_count = sum(item["direction"] == "SHORT" for item in trades)
    directions = {
        "LONG": _direction_metrics(trades, "LONG"),
        "SHORT": _direction_metrics(trades, "SHORT"),
    }
    return {
        **funnel_stages,
        **{
            key: funnel[key]
            for key in (
                "proposed_setups",
                "invalid_setups",
                "non_executable_setups",
                "compliance_blocks",
                "executed_trades",
            )
        },
        "long_trades": long_count,
        "short_trades": short_count,
        "long_short_metrics": directions,
        "long_short_reconciliation": {
            "executed_trades": len(trades),
            "long_plus_short": long_count + short_count,
            "reconciles": long_count + short_count == len(trades),
        },
        "gross_pnl": str(sum(gross_values, Decimal())),
        "net_pnl": str(sum(net_values, Decimal())),
        "gross_profit_factor": _profit_factor(gross_values),
        "net_profit_factor": _profit_factor(net_values),
        "profit_factor": _profit_factor(net_values),
        "average_gross_r": str(sum(gross_rs, Decimal()) / len(gross_rs)) if gross_rs else "0",
        "average_net_r": str(sum(net_rs, Decimal()) / len(net_rs)) if net_rs else "0",
        "median_gross_r": str(median(gross_rs)) if gross_rs else "0",
        "median_net_r": str(median(net_rs)) if net_rs else "0",
        "median_initial_risk_usd": str(median(risks)) if risks else "0",
        "gross_risk_usd": str(sum(risks, Decimal())),
        "fees": str(fees),
        "slippage_cost": str(slippage),
        "total_costs": str(total_costs),
        "median_cost_to_risk": str(median(cost_ratios)) if cost_ratios else "0",
        "cost_to_risk_share_over_10_percent": str(
            Decimal(sum(value > Decimal("0.10") for value in cost_ratios)) / len(cost_ratios)
        )
        if cost_ratios
        else "0",
        "cost_to_risk_share_over_25_percent": str(
            Decimal(sum(value > Decimal("0.25") for value in cost_ratios)) / len(cost_ratios)
        )
        if cost_ratios
        else "0",
        "cost_to_risk_share_over_50_percent": str(
            Decimal(sum(value > Decimal("0.50") for value in cost_ratios)) / len(cost_ratios)
        )
        if cost_ratios
        else "0",
        "win_rate": str(Decimal(sum(value > 0 for value in net_values)) / len(net_values)) if net_values else "0",
        "maximum_drawdown": str(maximum_drawdown),
        "longest_losing_streak": longest_losing,
        "months": monthly,
        "maximum_positive_month_contribution": str(maximum_month_contribution),
        "best_five_positive_pnl_contribution": str(best_five),
        "funnel_reconciliation": funnel,
        "same_bar_stop_first_count": sum(bool(item["same_bar_ambiguity"]) for item in trades),
        "forced_flat_count": sum(item["exit_reason"] == "UTC_DAY_FORCE_FLAT" for item in trades),
        "cost_model_version": COST_MODEL_VERSION,
    }
