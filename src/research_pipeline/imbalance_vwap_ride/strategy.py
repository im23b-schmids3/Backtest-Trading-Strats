from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from statistics import median
from typing import Any, Callable

from .artifacts import sha256_value
from .models import COST_MODEL_VERSION, ImbalanceVWAPRideConfig


def _d(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _stable_id(prefix: str, payload: Any) -> str:
    return f"{prefix}-{hashlib.sha256(str(sha256_value(payload)).encode()).hexdigest()[:20]}"


def compute_completed_bar_regimes(bars: list[dict[str, Any]], slope_bars: int) -> list[dict[str, Any]]:
    """Compute daily VWAP regimes using only the current and prior closed bars."""

    ordered = sorted((dict(item) for item in bars), key=lambda item: _ts(item["bar_start_utc"]))
    history: dict[str, list[Decimal]] = defaultdict(list)
    cumulative: dict[str, tuple[Decimal, Decimal]] = {}
    for position, bar in enumerate(ordered):
        start = _ts(bar["bar_start_utc"])
        day = str(bar.get("session_date") or start.date().isoformat())
        volume = _d(bar["volume"])
        notional = _d(bar.get("notional", _d(bar["close"]) * volume))
        prior_volume, prior_notional = cumulative.get(day, (Decimal(), Decimal()))
        cumulative[day] = prior_volume + volume, prior_notional + notional
        vwap = cumulative[day][1] / cumulative[day][0] if cumulative[day][0] else _d(bar["close"])
        earlier = history[day][-slope_bars] if len(history[day]) >= slope_bars else None
        close = _d(bar["close"])
        bar["daily_vwap"] = vwap
        bar["long_vwap_regime"] = bool(earlier is not None and close > vwap and vwap > earlier)
        bar["short_vwap_regime"] = bool(earlier is not None and close < vwap and vwap < earlier)
        bar["vwap_comparison_bar_start_utc"] = (
            _ts(ordered[position - slope_bars]["bar_start_utc"]).isoformat()
            if earlier is not None
            else None
        )
        history[day].append(vwap)
    return ordered


def coarsen_footprints(
    rows: list[dict[str, Any]], bin_size_usd: Decimal
) -> dict[datetime, list[dict[str, Any]]]:
    size = _d(bin_size_usd)
    grouped: dict[tuple[datetime, Decimal], list[Any]] = {}
    for row in rows:
        start = _ts(row["bar_start_utc"])
        floor = (_d(row["bin_floor"]) / size).to_integral_value(rounding=ROUND_FLOOR) * size
        key = (start, floor)
        current = grouped.setdefault(key, [Decimal(), Decimal(), 0])
        current[0] += _d(row["buy_volume_btc"])
        current[1] += _d(row["sell_volume_btc"])
        current[2] += int(row.get("trade_count", 0))
    output: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for (start, floor), (buy, sell, count) in sorted(grouped.items()):
        output[start].append(
            {
                "bar_start_utc": start,
                "bin_floor": floor,
                "bin_upper_exclusive": floor + size,
                "buy_volume_btc": buy,
                "sell_volume_btc": sell,
                "total_volume_btc": buy + sell,
                "delta_btc": buy - sell,
                "trade_count": count,
            }
        )
    return output


def qualifying_imbalance(row: dict[str, Any], config: ImbalanceVWAPRideConfig, direction: str) -> bool:
    buy, sell = _d(row["buy_volume_btc"]), _d(row["sell_volume_btc"])
    total = buy + sell
    if total < config.min_bin_volume_btc:
        return False
    if direction == "LONG":
        return (sell == 0 and buy >= config.min_bin_volume_btc) or (
            sell > 0 and buy >= config.min_imbalance_ratio * sell
        )
    return (buy == 0 and sell >= config.min_bin_volume_btc) or (
        buy > 0 and sell >= config.min_imbalance_ratio * buy
    )


def maximal_imbalance_sequences(
    rows: list[dict[str, Any]], config: ImbalanceVWAPRideConfig, direction: str
) -> list[dict[str, Any]]:
    """Return one record for every maximal adjacent qualifying run."""

    ordered = sorted(rows, key=lambda row: _d(row["bin_floor"]))
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        qualifies = qualifying_imbalance(row, config, direction)
        adjacent = bool(current and _d(row["bin_floor"]) == _d(current[-1]["bin_floor"]) + config.bin_size_usd)
        if qualifies and (not current or adjacent):
            current.append(row)
        elif qualifies:
            if len(current) >= config.stacked_bins:
                runs.append(current)
            current = [row]
        else:
            if len(current) >= config.stacked_bins:
                runs.append(current)
            current = []
    if len(current) >= config.stacked_bins:
        runs.append(current)
    sequences: list[dict[str, Any]] = []
    for run in runs:
        payload = {
            "direction": direction,
            "bar_start_utc": _ts(run[0]["bar_start_utc"]).isoformat(),
            "bottom": str(_d(run[0]["bin_floor"])),
            "top": str(_d(run[-1]["bin_floor"]) + config.bin_size_usd),
            "bin_floors": [str(_d(item["bin_floor"])) for item in run],
            "buy_volume_btc": str(sum((_d(item["buy_volume_btc"]) for item in run), Decimal())),
            "sell_volume_btc": str(sum((_d(item["sell_volume_btc"]) for item in run), Decimal())),
        }
        buy, sell = _d(payload["buy_volume_btc"]), _d(payload["sell_volume_btc"])
        payload.update(
            {
                "sequence_id": _stable_id("seq", payload),
                "total_volume_btc": str(buy + sell),
                "delta_btc": str(buy - sell),
                "bin_count": len(run),
            }
        )
        sequences.append(payload)
    return sequences


def _overlaps(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return _d(first["bottom"]) < _d(second["top"]) and _d(second["bottom"]) < _d(first["top"])


def _quantize(value: Decimal, tick: Decimal, rounding: str) -> Decimal:
    return (value / tick).to_integral_value(rounding=rounding) * tick


def _simulate_trade(
    *,
    zone: dict[str, Any],
    signal_bar: dict[str, Any],
    entry_index: int,
    bars: list[dict[str, Any]],
    config: ImbalanceVWAPRideConfig,
) -> tuple[str, dict[str, Any] | None]:
    if entry_index >= len(bars):
        return "NO_EXECUTABLE_ENTRY", None
    entry_bar = bars[entry_index]
    signal_start, entry_start = _ts(signal_bar["bar_start_utc"]), _ts(entry_bar["bar_start_utc"])
    if entry_start != signal_start + timedelta(minutes=5) or entry_start.date() != signal_start.date():
        return "NO_EXECUTABLE_ENTRY", None
    direction = zone["direction"]
    sign = Decimal("1") if direction == "LONG" else Decimal("-1")
    quantity = (config.quantity_btc / config.quantity_step).to_integral_value(rounding=ROUND_FLOOR) * config.quantity_step
    if quantity < config.minimum_quantity:
        return "INVALIDATED", None
    reference_entry = _d(entry_bar["open"])
    slipped_entry = reference_entry + sign * config.price_tick * config.market_slippage_ticks
    entry_price = _quantize(
        slipped_entry,
        config.price_tick,
        ROUND_CEILING if direction == "LONG" else ROUND_FLOOR,
    )
    raw_stop = (
        _d(zone["bottom"]) - config.stop_buffer_bins * config.bin_size_usd
        if direction == "LONG"
        else _d(zone["top"]) + config.stop_buffer_bins * config.bin_size_usd
    )
    stop = _quantize(raw_stop, config.price_tick, ROUND_FLOOR if direction == "LONG" else ROUND_CEILING)
    risk_distance = sign * (entry_price - stop)
    if risk_distance <= 0:
        return "INVALIDATED", None
    raw_target = entry_price + sign * config.target_r_multiple * risk_distance
    target = _quantize(raw_target, config.price_tick, ROUND_CEILING if direction == "LONG" else ROUND_FLOOR)
    day = entry_start.date()
    reference_exit = _d(entry_bar["close"])
    exit_price = reference_exit
    exit_bar = entry_bar
    exit_reason = "UTC_DAY_FORCE_FLAT"
    exit_slippage_ticks = config.market_slippage_ticks
    same_bar_ambiguity = False
    for candidate in bars[entry_index:]:
        if _ts(candidate["bar_start_utc"]).date() != day:
            break
        hit_stop = _d(candidate["low"]) <= stop if direction == "LONG" else _d(candidate["high"]) >= stop
        hit_target = _d(candidate["high"]) >= target if direction == "LONG" else _d(candidate["low"]) <= target
        if hit_stop:
            reference_exit = stop
            exit_price = stop - sign * config.price_tick * config.stop_slippage_ticks
            exit_bar = candidate
            exit_reason = "STOP_FIRST_AMBIGUITY" if hit_target else "STOP"
            exit_slippage_ticks = config.stop_slippage_ticks
            same_bar_ambiguity = hit_target
            break
        if hit_target:
            reference_exit = exit_price = target
            exit_bar = candidate
            exit_reason = "TARGET"
            exit_slippage_ticks = 0
            break
        reference_exit = _d(candidate["close"])
        exit_price = reference_exit - sign * config.price_tick * config.market_slippage_ticks
        exit_bar = candidate
    entry_slippage = abs(entry_price - reference_entry) * quantity
    exit_slippage = abs(exit_price - reference_exit) * quantity
    slippage = entry_slippage + exit_slippage
    gross = sign * (reference_exit - reference_entry) * quantity
    fees = config.taker_fee_rate * (entry_price + exit_price) * quantity
    net = gross - fees - slippage
    initial_risk = risk_distance * quantity
    trade_payload = {
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
        "initial_stop_price": str(stop),
        "target_price": str(target),
        "reference_exit_price": str(reference_exit),
        "exit_price": str(exit_price),
        "quantity_btc": str(quantity),
        "initial_risk_usd": str(initial_risk),
        "gross_pnl": str(gross),
        "fees": str(fees),
        "slippage_cost": str(slippage),
        "net_pnl": str(net),
        "net_r": str(net / initial_risk),
        "exit_reason": exit_reason,
        "same_bar_ambiguity": same_bar_ambiguity,
        "entry_slippage_ticks": config.market_slippage_ticks,
        "exit_slippage_ticks": exit_slippage_ticks,
        "cost_model_version": COST_MODEL_VERSION,
    }
    trade_payload["trade_id"] = _stable_id(
        "trade",
        {
            "zone_id": zone["zone_id"],
            "session_date": day.isoformat(),
            "entry_timestamp": entry_start.isoformat(),
            "parameters": config.parameter_payload(),
        },
    )
    return "TRADE_EXECUTED", trade_payload


def _terminal_zone(zone: dict[str, Any], state: str, bar: dict[str, Any], reason: str) -> dict[str, Any]:
    zone["state"] = state
    zone["terminal_reason"] = reason
    zone["terminal_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
    return zone


def run_imbalance_vwap_ride(
    bars: list[dict[str, Any]],
    footprints: list[dict[str, Any]] | dict[datetime, list[dict[str, Any]]],
    config: ImbalanceVWAPRideConfig = ImbalanceVWAPRideConfig(),
    *,
    compliance_check: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str | None]] | None = None,
) -> dict[str, Any]:
    """Run the completed-bar-only, next-bar-entry standalone strategy."""

    enriched = compute_completed_bar_regimes(bars, config.vwap_slope_bars)
    footprint_by_bar = footprints if isinstance(footprints, dict) else coarsen_footprints(footprints, config.bin_size_usd)
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
                    "event",
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

    for index, bar in enumerate(enriched):
        start = _ts(bar["bar_start_utc"])
        day = start.date().isoformat()
        active = [zone for zone in zones if zone["state"] in {"ACTIVE", "ARMED"}]
        for zone in active:
            direction = zone["direction"]
            if index - int(zone["created_index"]) > config.zone_expiry_bars:
                _terminal_zone(zone, "INVALIDATED", bar, "ZONE_EXPIRED")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="ZONE_EXPIRED")
                continue
            adverse = _d(bar["close"]) < _d(zone["bottom"]) if direction == "LONG" else _d(bar["close"]) > _d(zone["top"])
            if adverse:
                _terminal_zone(zone, "INVALIDATED", bar, "ADVERSE_ZONE_BOUNDARY_CLOSE")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="ADVERSE_ZONE_BOUNDARY_CLOSE")
                continue
            regime = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
            if zone["vwap_qualified"] and not regime:
                _terminal_zone(zone, "INVALIDATED", bar, "VWAP_LOSS_PRE_RETEST")
                event(bar, "ZONE_INVALIDATED", zone_id=zone["zone_id"], reason="VWAP_LOSS_PRE_RETEST")
                continue
            if not zone["vwap_qualified"] and regime:
                zone["vwap_qualified"] = True
                zone["vwap_qualified_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                vwap_qualified += 1
                event(bar, "VWAP_QUALIFIED", zone_id=zone["zone_id"], direction=direction)
            if not zone["vwap_qualified"] or index == int(zone["created_index"]):
                continue
            moved = _d(bar["low"]) > _d(zone["top"]) if direction == "LONG" else _d(bar["high"]) < _d(zone["bottom"])
            if zone["state"] == "ACTIVE":
                zone["move_away_count"] = int(zone["move_away_count"]) + 1 if moved else 0
                if int(zone["move_away_count"]) >= config.move_away_bars:
                    zone["state"] = "ARMED"
                    zone["armed_timestamp"] = _ts(bar["bar_end_utc"]).isoformat()
                    move_away_confirmed += 1
                    event(bar, "MOVE_AWAY_CONFIRMED", zone_id=zone["zone_id"], direction=direction)

        new_zones: list[dict[str, Any]] = []
        for direction in ("LONG", "SHORT"):
            sequences = maximal_imbalance_sequences(footprint_by_bar.get(start, []), config, direction)
            for sequence in sequences:
                sequence_count += 1
                event(bar, "IMBALANCE_SEQUENCE", sequence_id=sequence["sequence_id"], direction=direction)
                qualified = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
                zone_payload = {
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
                    "vwap_qualified": qualified,
                    "vwap_qualified_timestamp": _ts(bar["bar_end_utc"]).isoformat() if qualified else None,
                    "armed_timestamp": None,
                    "terminal_timestamp": None,
                    "terminal_reason": None,
                }
                zone_payload["zone_id"] = _stable_id(
                    "zone", {key: zone_payload[key] for key in ("variant_id", "direction", "created_timestamp", "bottom", "top", "sequence_lineage")}
                )
                zones_created += 1
                vwap_qualified += int(qualified)
                event(bar, "ZONE_CREATED", zone_id=zone_payload["zone_id"], direction=direction, vwap_qualified=qualified)
                overlaps = [
                    zone
                    for zone in zones
                    if zone["direction"] == direction and zone["state"] in {"ACTIVE", "ARMED"} and _overlaps(zone, zone_payload)
                ]
                if overlaps:
                    survivor = sorted(overlaps, key=lambda item: (int(item["created_index"]), item["zone_id"]))[0]
                    survivor["bottom"] = str(min(_d(survivor["bottom"]), _d(zone_payload["bottom"])))
                    survivor["top"] = str(max(_d(survivor["top"]), _d(zone_payload["top"])))
                    survivor["sequence_lineage"] = sorted(set(survivor["sequence_lineage"] + zone_payload["sequence_lineage"]))
                    survivor["buy_volume_btc"] = str(_d(survivor["buy_volume_btc"]) + _d(zone_payload["buy_volume_btc"]))
                    survivor["sell_volume_btc"] = str(_d(survivor["sell_volume_btc"]) + _d(zone_payload["sell_volume_btc"]))
                    survivor["total_volume_btc"] = str(_d(survivor["buy_volume_btc"]) + _d(survivor["sell_volume_btc"]))
                    survivor["delta_btc"] = str(_d(survivor["buy_volume_btc"]) - _d(survivor["sell_volume_btc"]))
                    survivor["expiry_index"] = max(int(survivor["expiry_index"]), int(zone_payload["expiry_index"]))
                    survivor["vwap_qualified"] = bool(survivor["vwap_qualified"] or qualified)
                    event(bar, "ZONE_MERGED", zone_id=survivor["zone_id"], merged_sequence_id=sequence["sequence_id"])
                    new_zones.append(survivor)
                else:
                    zones.append(zone_payload)
                    new_zones.append(zone_payload)

        # A newly created overlapping opposite zone supersedes the older one.
        # Same-bar ties are volume, absolute delta, then direction, all explicit.
        for left in [zone for zone in zones if zone["state"] in {"ACTIVE", "ARMED"}]:
            for right in [zone for zone in zones if zone["state"] in {"ACTIVE", "ARMED"} and zone["direction"] != left["direction"]]:
                if left is right or not _overlaps(left, right):
                    continue
                left_rank = (int(left["created_index"]), _d(left["total_volume_btc"]), abs(_d(left["delta_btc"])), left["direction"])
                right_rank = (int(right["created_index"]), _d(right["total_volume_btc"]), abs(_d(right["delta_btc"])), right["direction"])
                loser, winner = (left, right) if left_rank < right_rank else (right, left)
                if loser["state"] in {"ACTIVE", "ARMED"}:
                    _terminal_zone(loser, "INVALIDATED", bar, "OPPOSITE_ZONE_SUPERSESSION")
                    event(bar, "ZONE_INVALIDATED", zone_id=loser["zone_id"], reason="OPPOSITE_ZONE_SUPERSESSION", superseded_by=winner["zone_id"])

        for direction in ("LONG", "SHORT"):
            candidates = sorted(
                [zone for zone in zones if zone["direction"] == direction and zone["state"] in {"ACTIVE", "ARMED"}],
                key=lambda item: (int(item["created_index"]), item["zone_id"]),
                reverse=True,
            )
            for overflow in candidates[config.maximum_active_zones_per_direction :]:
                _terminal_zone(overflow, "INVALIDATED", bar, "ACTIVE_ZONE_CAP")
                event(bar, "ZONE_INVALIDATED", zone_id=overflow["zone_id"], reason="ACTIVE_ZONE_CAP")

        triggered = []
        for zone in zones:
            if zone["state"] != "ARMED":
                continue
            direction = zone["direction"]
            regime = bool(bar["long_vwap_regime"] if direction == "LONG" else bar["short_vwap_regime"])
            retest = (
                _d(bar["low"]) <= _d(zone["top"]) <= _d(bar["high"]) and _d(bar["close"]) >= _d(zone["top"])
                if direction == "LONG"
                else _d(bar["low"]) <= _d(zone["bottom"]) <= _d(bar["high"]) and _d(bar["close"]) <= _d(zone["bottom"])
            )
            if regime and retest:
                triggered.append(zone)
        if triggered:
            zone = sorted(triggered, key=lambda item: (int(item["created_index"]), item["zone_id"]), reverse=True)[0]
            retest_triggers += 1
            proposed += 1
            event(bar, "RETEST_TRIGGER", zone_id=zone["zone_id"], direction=zone["direction"])
            event(bar, "PROPOSED_SETUP", zone_id=zone["zone_id"], direction=zone["direction"])
            if day in used_days:
                compliance_blocks += 1
                _terminal_zone(zone, "INVALIDATED", bar, "DAILY_TRADE_CAP")
                event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason="DAILY_TRADE_CAP")
            else:
                allowed, reason = compliance_check(zone, bar) if compliance_check else (True, None)
                if not allowed:
                    compliance_blocks += 1
                    _terminal_zone(zone, "INVALIDATED", bar, reason or "COMPLIANCE_BLOCK")
                    event(bar, "COMPLIANCE_BLOCKED", zone_id=zone["zone_id"], reason=reason or "COMPLIANCE_BLOCK")
                else:
                    state, trade = _simulate_trade(zone=zone, signal_bar=bar, entry_index=index + 1, bars=enriched, config=config)
                    if state == "NO_EXECUTABLE_ENTRY":
                        non_executable += 1
                        _terminal_zone(zone, "INVALIDATED", bar, state)
                        event(bar, state, zone_id=zone["zone_id"])
                    elif state == "INVALIDATED":
                        invalid += 1
                        _terminal_zone(zone, "INVALIDATED", bar, "INVALID_ENTRY_GEOMETRY_OR_QUANTITY")
                        event(bar, "INVALIDATED_SETUP", zone_id=zone["zone_id"], reason="INVALID_ENTRY_GEOMETRY_OR_QUANTITY")
                    else:
                        assert trade is not None
                        trades.append(trade)
                        used_days.add(day)
                        _terminal_zone(zone, "EXECUTED", bar, "POST_EXECUTION")
                        event(bar, "TRADE_EXECUTED", zone_id=zone["zone_id"], trade_id=trade["trade_id"], direction=zone["direction"])

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
        raise AssertionError("strategy funnel failed exact reconciliation")
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


def summarize_strategy_result(
    bars: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    funnel: dict[str, Any],
    funnel_stages: dict[str, int],
) -> dict[str, Any]:
    values = [_d(item["net_pnl"]) for item in trades]
    rs = [_d(item["net_r"]) for item in trades]
    gains = sum((value for value in values if value > 0), Decimal())
    losses = -sum((value for value in values if value < 0), Decimal())
    equity = peak = maximum_drawdown = Decimal()
    current_losing = longest_losing = 0
    for value in values:
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
        current_losing = current_losing + 1 if value <= 0 else 0
        longest_losing = max(longest_losing, current_losing)
    months_present = sorted({str(bar.get("month") or _ts(bar["bar_start_utc"]).strftime("%Y-%m")) for bar in bars})
    monthly: dict[str, dict[str, Any]] = {}
    for month in months_present:
        month_trades = [item for item in trades if str(item["entry_timestamp"]).startswith(month)]
        month_values = [_d(item["net_pnl"]) for item in month_trades]
        monthly[month] = {
            "five_minute_bar_count": sum(str(bar.get("month") or _ts(bar["bar_start_utc"]).strftime("%Y-%m")) == month for bar in bars),
            "executed_trades": len(month_trades),
            "net_pnl": str(sum(month_values, Decimal())),
            "long_trades": sum(item["direction"] == "LONG" for item in month_trades),
            "short_trades": sum(item["direction"] == "SHORT" for item in month_trades),
        }
    positive_months = [max(_d(item["net_pnl"]), Decimal()) for item in monthly.values()]
    positive_month_total = sum(positive_months, Decimal())
    maximum_month_contribution = max(positive_months, default=Decimal()) / positive_month_total if positive_month_total else Decimal("1")
    positive_trades = sorted((value for value in values if value > 0), reverse=True)
    positive_total = sum(positive_trades, Decimal())
    best_five = sum(positive_trades[:5], Decimal()) / positive_total if positive_total else Decimal("1")
    costs = sum((_d(item["fees"]) + _d(item["slippage_cost"]) for item in trades), Decimal())
    days: dict[str, Decimal] = defaultdict(Decimal)
    for trade in trades:
        days[str(trade["session_date"])] += _d(trade["net_pnl"])
    positive_days = sorted((value for value in days.values() if value > 0), reverse=True)
    positive_day_total = sum(positive_days, Decimal())
    return {
        **funnel_stages,
        **{key: funnel[key] for key in ("proposed_setups", "invalid_setups", "non_executable_setups", "compliance_blocks", "executed_trades")},
        "long_trades": sum(item["direction"] == "LONG" for item in trades),
        "short_trades": sum(item["direction"] == "SHORT" for item in trades),
        "gross_pnl": str(sum((_d(item["gross_pnl"]) for item in trades), Decimal())),
        "fees": str(sum((_d(item["fees"]) for item in trades), Decimal())),
        "slippage_cost": str(sum((_d(item["slippage_cost"]) for item in trades), Decimal())),
        "total_costs": str(costs),
        "net_pnl": str(sum(values, Decimal())),
        "profit_factor": str(gains / losses) if losses > 0 else None,
        "average_net_r": str(sum(rs, Decimal()) / len(rs)) if rs else "0",
        "median_net_r": str(median(rs)) if rs else "0",
        "win_rate": str(Decimal(sum(value > 0 for value in values)) / len(values)) if values else "0",
        "maximum_drawdown": str(maximum_drawdown),
        "longest_losing_streak": longest_losing,
        "months": monthly,
        "maximum_positive_month_contribution": str(maximum_month_contribution),
        "best_day_positive_pnl_contribution": str(positive_days[0] / positive_day_total) if positive_day_total else "1",
        "best_five_positive_pnl_contribution": str(best_five),
        "funnel_reconciliation": funnel,
        "same_bar_stop_first_count": sum(bool(item["same_bar_ambiguity"]) for item in trades),
        "forced_flat_count": sum(item["exit_reason"] == "UTC_DAY_FORCE_FLAT" for item in trades),
        "cost_model_version": COST_MODEL_VERSION,
    }
