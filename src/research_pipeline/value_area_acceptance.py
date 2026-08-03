"""Standalone sealed BTC 2024 ValueAreaAcceptance study.

This module is intentionally independent of the repository's legacy trap
adapter.  It
contains the deterministic bar builder, sequential candidate machine and
artifact writer for the four pre-registered exploratory variants.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schemas.strategy_spec import StrictModel
from .value_area_trap.data import AggregateTradeImporter

STRATEGY_ID = "ValueAreaAcceptance.BTC_2024_EXPLORATORY"
ADAPTER_ID = "value-area-acceptance-btc-2024-1"
DATASET_HASH = "c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc"
STUDY_LABEL = "EXPLORATORY_IN_SAMPLE"
ENGINE_CONTRACT_VERSION = "value-area-acceptance-btc-2024-2"
PRICE_TICK = Decimal("0.10")
QUANTITY_STEP = Decimal("0.001")
QUANTITY = Decimal("0.001")
FEE_RATE = Decimal("0.0005")
SLIPPAGE_TICKS = Decimal("1")

SEALED_VARIANTS = (
    # Keep this tuple literal and ordered: it is part of the study contract.
    # A: 1.50 / 2 / 1.50R; B: 1.25 / 2 / 1.50R;
    # C: 1.50 / 2 / 2.00R; D: 1.25 / 1 / 2.00R.
)


class AcceptanceVariant(StrictModel):
    variant_id: str
    breakout_volume_multiplier: Decimal
    acceptance_bars: int
    target_r_multiple: Decimal


SEALED_VARIANTS = (
    AcceptanceVariant(variant_id="A", breakout_volume_multiplier=Decimal("1.50"), acceptance_bars=2, target_r_multiple=Decimal("1.50")),
    AcceptanceVariant(variant_id="B", breakout_volume_multiplier=Decimal("1.25"), acceptance_bars=2, target_r_multiple=Decimal("1.50")),
    AcceptanceVariant(variant_id="C", breakout_volume_multiplier=Decimal("1.50"), acceptance_bars=2, target_r_multiple=Decimal("2.00")),
    AcceptanceVariant(variant_id="D", breakout_volume_multiplier=Decimal("1.25"), acceptance_bars=1, target_r_multiple=Decimal("2.00")),
)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: Any) -> None:
    raw = json.dumps(payload, sort_keys=True, indent=2, default=str).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable acceptance artifact differs: {path}")
        return
    path.write_bytes(raw)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _bar_timestamp(bar: dict[str, Any]) -> str:
    return str(bar.get("timestamp_utc", bar.get("timestamp", "")))


class ValueAreaAcceptanceAdapter:
    """Standalone strategy identity and deterministic execution primitives."""

    strategy_id = STRATEGY_ID
    adapter_identity = ADAPTER_ID
    strategy_family = "value_area_acceptance_breakout"

    @staticmethod
    def deterministic_run_id(dataset_hash: str = DATASET_HASH) -> str:
        registry = [(v.variant_id, str(v.breakout_volume_multiplier), v.acceptance_bars, str(v.target_r_multiple)) for v in SEALED_VARIANTS]
        return "btc-acceptance-2024-" + _hash({"strategy_id": STRATEGY_ID, "adapter_identity": ADAPTER_ID, "engine_contract_version": ENGINE_CONTRACT_VERSION, "dataset_hash": dataset_hash, "variants": registry})[:16]

    @staticmethod
    def acceptance_confirmed(closes: list[Decimal], boundary: Decimal, *, side: str, count: int) -> bool:
        values = closes[-count:]
        return len(values) == count and all(value > boundary for value in values) if side == "LONG" else len(values) == count and all(value < boundary for value in values)

    @staticmethod
    def cvd_confirmed(cvd_start: Decimal, cvd_end: Decimal, *, side: str) -> bool:
        return cvd_end > cvd_start if side == "LONG" else cvd_end < cvd_start

    @staticmethod
    def pullback_confirmed(low: Decimal, high: Decimal, close: Decimal, boundary: Decimal, *, side: str) -> bool:
        return (low <= boundary and close >= boundary) if side == "LONG" else (high >= boundary and close <= boundary)

    @staticmethod
    def volume_qualified(volume: Decimal, history: list[Decimal], multiplier: Decimal) -> bool:
        if len(history) < 10:
            return False
        ordered = sorted(history[-10:])
        median = (ordered[4] + ordered[5]) / Decimal("2")
        return volume >= multiplier * median

    @staticmethod
    def fixed_r_exit(entry: Decimal, stop: Decimal, multiple: Decimal, *, side: str) -> Decimal:
        risk = abs(entry - stop)
        return entry + risk * multiple if side == "LONG" else entry - risk * multiple

    @staticmethod
    def deterministic_trade_id(day: str, side: str, entry_timestamp: str, variant_id: str) -> str:
        return _hash(f"{STRATEGY_ID}|{variant_id}|{day}|{side}|{entry_timestamp}")[:20]

    @staticmethod
    def _price(value: Decimal) -> Decimal:
        return (value / PRICE_TICK).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * PRICE_TICK

    @staticmethod
    def close_setup(*, day: str, side: str, pullback: dict[str, Any], next_bar: dict[str, Any] | None,
                    later_bars: list[dict[str, Any]], variant: AcceptanceVariant,
                    price_tick: Decimal = PRICE_TICK, quantity: Decimal = QUANTITY,
                    fee_rate: Decimal = FEE_RATE, slippage_ticks: Decimal = SLIPPAGE_TICKS) -> dict[str, Any]:
        if next_bar is None:
            return {"state": "NO_EXECUTABLE_ENTRY", "trade": None}
        sign = Decimal(1) if side == "LONG" else Decimal(-1)
        entry = ValueAreaAcceptanceAdapter._price(_decimal(next_bar["open"]) + sign * price_tick * slippage_ticks)
        stop = ValueAreaAcceptanceAdapter._price(_decimal(pullback["low"]) - price_tick if side == "LONG" else _decimal(pullback["high"]) + price_tick)
        target = ValueAreaAcceptanceAdapter._price(ValueAreaAcceptanceAdapter.fixed_r_exit(entry, stop, variant.target_r_multiple, side=side))
        if not ((stop < entry < target) if side == "LONG" else (target < entry < stop)):
            return {"state": "INVALIDATED", "trade": None}
        exit_price = _decimal(next_bar.get("close", next_bar["open"]))
        reason = "UTC_FORCE_FLAT"
        exit_timestamp = _bar_timestamp(next_bar)
        for bar in later_bars:
            hit_stop = _decimal(bar["low"]) <= stop if side == "LONG" else _decimal(bar["high"]) >= stop
            hit_target = _decimal(bar["high"]) >= target if side == "LONG" else _decimal(bar["low"]) <= target
            if hit_stop:
                exit_price = stop - sign * price_tick * slippage_ticks
                reason = "STOP_FIRST" if hit_target else "STOP"
                exit_timestamp = _bar_timestamp(bar)
                break
            if hit_target:
                exit_price = target - sign * price_tick * slippage_ticks
                reason = "TARGET"
                exit_timestamp = _bar_timestamp(bar)
                break
            exit_price = _decimal(bar.get("close", bar["open"]))
            exit_timestamp = _bar_timestamp(bar)
        exit_price = ValueAreaAcceptanceAdapter._price(exit_price)
        qty = (quantity / QUANTITY_STEP).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * QUANTITY_STEP
        gross = sign * (exit_price - entry) * qty
        fees = (abs(entry) + abs(exit_price)) * qty * fee_rate
        slippage_cost = price_tick * slippage_ticks * qty * Decimal("2")
        initial_risk = abs(entry - stop) * qty
        net_pnl = gross - fees - slippage_cost
        return {"state": "TRADE_EXECUTED", "trade": {"trade_id": ValueAreaAcceptanceAdapter.deterministic_trade_id(day, side, _bar_timestamp(next_bar), variant.variant_id), "day": day, "side": side, "entry_timestamp": _bar_timestamp(next_bar), "exit_timestamp": exit_timestamp, "entry_price": str(entry), "stop": str(stop), "target": str(target), "exit_price": str(exit_price), "exit_reason": reason, "quantity": str(qty), "initial_risk": str(initial_risk), "gross_pnl": str(gross), "fees": str(fees), "slippage_cost": str(slippage_cost), "net_pnl": str(net_pnl), "net_r": str(net_pnl / initial_risk) if initial_risk else "0"}}


BAR_SCHEMA = pa.schema([
    ("timestamp_utc", pa.timestamp("us", tz="UTC")), ("open", pa.string()), ("high", pa.string()), ("low", pa.string()),
    ("close", pa.string()), ("volume", pa.string()), ("buy_volume", pa.string()), ("sell_volume", pa.string()),
    ("cvd_delta", pa.string()), ("session_date", pa.string()),
])


def _arrow_bar(row: dict[str, Any]) -> dict[str, Any]:
    """Convert the in-memory Decimal bar into the pinned Arrow representation.

    The strategy keeps prices and quantities as ``Decimal`` until this narrow
    serialization boundary.  The immutable Parquet contract deliberately uses
    strings for exact decimal round-tripping, so Arrow must receive strings --
    not Decimal instances -- for those fields.
    """
    return {
        "timestamp_utc": row["timestamp_utc"],
        "open": str(row["open"]),
        "high": str(row["high"]),
        "low": str(row["low"]),
        "close": str(row["close"]),
        "volume": str(row["volume"]),
        "buy_volume": str(row["buy_volume"]),
        "sell_volume": str(row["sell_volume"]),
        "cvd_delta": str(row["cvd_delta"]),
        "session_date": row["session_date"],
    }


def _bucket(timestamp: datetime) -> datetime:
    return timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)


def _aggregate_batch(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    """Aggregate one ordered trade batch without materialising every trade.

    Prices are read through Arrow's ordered first/last aggregators.  BTCUSDT's
    pinned archive price increment is decimal-compatible, and converting the
    resulting double through ``str`` preserves that exchange representation;
    traded quantity and CVD remain Arrow decimals end-to-end.
    """
    timestamp = pc.strptime(
        pc.utf8_slice_codeunits(batch.column("trade_time_utc"), start=0, stop=19),
        format="%Y-%m-%dT%H:%M:%S",
        unit="us",
    )
    bucket = pc.floor_temporal(timestamp, multiple=5, unit="minute")
    amount_type = pa.decimal128(28, 18)
    quantity = pc.cast(batch.column("quantity_base"), amount_type)
    price = pc.cast(batch.column("price"), pa.float64())
    zero = pa.scalar(Decimal("0"), type=amount_type)
    buy_volume = pc.if_else(batch.column("buyer_is_maker"), zero, quantity)
    sell_volume = pc.if_else(batch.column("buyer_is_maker"), quantity, zero)
    delta = pc.subtract(buy_volume, sell_volume)
    grouped = pa.table({
        "bucket": bucket,
        "price": price,
        "quantity": quantity,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "delta": delta,
    }).group_by("bucket", use_threads=False).aggregate([
        ("price", "first"), ("price", "max"), ("price", "min"), ("price", "last"),
        ("quantity", "sum"), ("buy_volume", "sum"), ("sell_volume", "sum"), ("delta", "sum"),
    ])
    rows = sorted(grouped.to_pylist(), key=lambda row: row["bucket"])
    return [
        {
            "timestamp_utc": row["bucket"].replace(tzinfo=timezone.utc),
            "open": _decimal(row["price_first"]),
            "high": _decimal(row["price_max"]),
            "low": _decimal(row["price_min"]),
            "close": _decimal(row["price_last"]),
            "volume": _decimal(row["quantity_sum"]),
            "buy_volume": _decimal(row["buy_volume_sum"]),
            "sell_volume": _decimal(row["sell_volume_sum"]),
            "cvd_delta": _decimal(row["delta_sum"]),
            "session_date": row["bucket"].date().isoformat(),
        }
        for row in rows
    ]


def _merge_bar(current: dict[str, Any] | None, incoming: dict[str, Any], output: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge a bucket which spans a Parquet batch, retaining ordered OHLC."""
    if current is None:
        return incoming
    if incoming["timestamp_utc"] != current["timestamp_utc"]:
        output.append(current)
        return incoming
    current["high"] = max(current["high"], incoming["high"])
    current["low"] = min(current["low"], incoming["low"])
    current["close"] = incoming["close"]
    for field in ("volume", "buy_volume", "sell_volume", "cvd_delta"):
        current[field] += incoming[field]
    return current


def materialize_bars(manifest_path: str, cache_root: str) -> dict[str, Any]:
    """Build bars from verified monthly partitions using bounded Arrow batches."""
    manifest = AggregateTradeImporter(".").validate_monthly_manifest(manifest_path)
    if manifest.normalized_dataset_hash != DATASET_HASH or manifest.symbol != "BTCUSDT":
        raise ValueError("ValueAreaAcceptance requires the pinned BTCUSDT Jan-Jul 2024 immutable dataset")
    cache_base = Path(cache_root).resolve() / "value-area-acceptance-bars"
    for existing_manifest in cache_base.glob("*/bar-manifest.json") if cache_base.exists() else ():
        payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        existing_path = Path(payload.get("bar_data_path", ""))
        if payload.get("aggregate_trade_dataset_hash") == DATASET_HASH and existing_path.is_file() and _sha(existing_path) == payload.get("parquet_sha256"):
            return {**payload, "bar_path": str(existing_path)}
    bars: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for partition in manifest.partitions:
        parquet = Path(manifest_path).resolve().parent / partition.file_name
        if _sha(parquet) != partition.parquet_hash:
            raise ValueError(f"partition hash mismatch: {parquet}")
        parquet_file = pq.ParquetFile(parquet)
        for batch in parquet_file.iter_batches(batch_size=250_000, columns=["trade_time_utc", "price", "quantity_base", "buyer_is_maker"], use_threads=True):
            for incoming in _aggregate_batch(batch):
                current = _merge_bar(current, incoming, bars)
    if current is not None:
        bars.append(current)
    identity = [{key: str(value) if isinstance(value, Decimal) else value.isoformat() if isinstance(value, datetime) else value for key, value in row.items()} for row in bars]
    bar_hash = _hash(identity)
    root = cache_base / bar_hash
    path = root / "five-minute-bars.parquet"
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pq.write_table(
            pa.Table.from_pylist([_arrow_bar(row) for row in bars], schema=BAR_SCHEMA),
            path,
            compression="zstd",
            use_dictionary=False,
        )
    payload = {"aggregate_trade_dataset_hash": DATASET_HASH, "bar_dataset_hash": bar_hash, "bar_data_path": str(path), "bar_count": len(bars), "first_timestamp": bars[0]["timestamp_utc"].isoformat(), "last_timestamp": bars[-1]["timestamp_utc"].isoformat(), "parquet_sha256": _sha(path)}
    _write_once(root / "bar-manifest.json", payload)
    return {**payload, "bar_path": str(path)}


def _profiles(bars: list[dict[str, Any]]) -> dict[str, dict[str, Decimal]]:
    profiles: dict[str, dict[Decimal, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for bar in bars:
        price = _decimal(bar["close"]); profiles[str(bar["session_date"])][(price // Decimal("10")) * Decimal("10")] += _decimal(bar["volume"])
    result: dict[str, dict[str, Decimal]] = {}
    for day, volumes in profiles.items():
        poc = max(volumes, key=lambda value: (volumes[value], -value)); total = sum(volumes.values(), Decimal()); selected = {poc}; covered = volumes[poc]; lower, upper = poc - Decimal("10"), poc + Decimal("10")
        while total and covered / total < Decimal("0.70") and (lower in volumes or upper in volumes):
            if volumes.get(lower, Decimal("-1")) >= volumes.get(upper, Decimal("-1")): selected.add(lower); covered += volumes[lower]; lower -= Decimal("10")
            else: selected.add(upper); covered += volumes[upper]; upper += Decimal("10")
        result[day] = {"val": min(selected), "vah": max(selected) + Decimal("10"), "poc": poc}
    return result


def _event(day: str, variant: AcceptanceVariant, index: int, state: str, bar: dict[str, Any], **extra: Any) -> dict[str, Any]:
    side = extra.pop("side", None)
    event = {"event_id": _hash(f"{STRATEGY_ID}|{variant.variant_id}|{day}|{index}|{state}|{extra}")[:20], "timestamp": _bar_timestamp(bar), "session_date": day, "state": state, "side": side, "variant_id": variant.variant_id, **extra}
    if side is not None: event["side"] = side
    return event


def _scan(bars: list[dict[str, Any]], variant: AcceptanceVariant) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(bars, key=_bar_timestamp); profiles = _profiles(ordered); by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bar in ordered: by_day[str(bar["session_date"])].append(bar)
    days = sorted(by_day); events: list[dict[str, Any]] = []; trades: list[dict[str, Any]] = []; prior_volumes: list[Decimal] = []; used_days: set[str] = set()
    terminal = {"INVALIDATED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "TRADE_EXECUTED"}
    for day_index, day in enumerate(days):
        day_bars = by_day[day]
        if day_index == 0 or days[day_index - 1] not in profiles:
            prior_volumes.extend(_decimal(b["volume"]) for b in day_bars); continue
        prior = profiles[days[day_index - 1]]; candidate: dict[str, Any] | None = None
        for index, bar in enumerate(day_bars):
            close = _decimal(bar["close"])
            if candidate is None:
                side = "LONG" if close > prior["vah"] else "SHORT" if close < prior["val"] else None
                if side:
                    boundary = prior["vah"] if side == "LONG" else prior["val"]
                    events.append(_event(day, variant, index, "BREAKOUT_DETECTED", bar, side=side, boundary=str(boundary)))
                    if ValueAreaAcceptanceAdapter.volume_qualified(_decimal(bar["volume"]), prior_volumes, variant.breakout_volume_multiplier):
                        events.append(_event(day, variant, index, "BREAKOUT_VOLUME_QUALIFIED", bar, side=side, boundary=str(boundary))); candidate = {"side": side, "boundary": boundary, "accepted": [bar], "stage": "ACCEPTANCE"}
                    else: events.append(_event(day, variant, index, "INVALIDATED", bar, side=side, reason="BREAKOUT_VOLUME_NOT_QUALIFIED"))
                prior_volumes.append(_decimal(bar["volume"])); continue
            side, boundary = candidate["side"], candidate["boundary"]
            outside = close > boundary if side == "LONG" else close < boundary
            if candidate["stage"] == "ACCEPTANCE":
                if not outside:
                    events.append(_event(day, variant, index, "INVALIDATED", bar, side=side, reason="ACCEPTANCE_BROKEN")); candidate = None
                else:
                    candidate["accepted"].append(bar)
                    if len(candidate["accepted"]) >= variant.acceptance_bars:
                        accepted = candidate["accepted"][-variant.acceptance_bars:]; events.append(_event(day, variant, index, "ACCEPTANCE_CONFIRMED", bar, side=side))
                        cvd = sum((_decimal(x.get("cvd_delta", x.get("delta", "0"))) for x in accepted), Decimal())
                        if (side == "LONG" and cvd <= 0) or (side == "SHORT" and cvd >= 0):
                            events.append(_event(day, variant, index, "INVALIDATED", bar, side=side, reason="CVD_NOT_CONFIRMED")); candidate = None
                        else:
                            events.append(_event(day, variant, index, "CVD_CONFIRMED", bar, side=side)); candidate["stage"] = "PULLBACK"
                prior_volumes.append(_decimal(bar["volume"])); continue
            # A valid boundary pullback may close exactly on the boundary;
            # equality is therefore tested by pullback_confirmed before the
            # strict outside invalidation check.
            if ValueAreaAcceptanceAdapter.pullback_confirmed(_decimal(bar["low"]), _decimal(bar["high"]), close, boundary, side=side):
                pass
            elif not outside:
                events.append(_event(day, variant, index, "INVALIDATED", bar, side=side, reason="PULLBACK_INVALIDATED")); candidate = None; prior_volumes.append(_decimal(bar["volume"])); continue
            else:
                prior_volumes.append(_decimal(bar["volume"])); continue
            setup_id = _hash(f"{STRATEGY_ID}|{variant.variant_id}|{day}|{index}|SETUP")[:20]
            events.append(_event(day, variant, index, "PULLBACK_CONFIRMED", bar, side=side)); events.append(_event(day, variant, index, "PROPOSED_SETUP", bar, side=side, setup_id=setup_id))
            if day in used_days:
                outcome = {"state": "COMPLIANCE_BLOCKED", "trade": None}
            else:
                outcome = ValueAreaAcceptanceAdapter.close_setup(day=day, side=side, pullback=bar, next_bar=day_bars[index + 1] if index + 1 < len(day_bars) else None, later_bars=day_bars[index + 2:], variant=variant)
            terminal_state = outcome["state"]
            if terminal_state not in terminal: raise AssertionError(f"unknown setup terminal state: {terminal_state}")
            if outcome["trade"] is not None: trades.append(outcome["trade"]); used_days.add(day)
            events.append(_event(day, variant, index, terminal_state, bar, side=side, setup_id=setup_id))
            candidate = None
            prior_volumes.append(_decimal(bar["volume"]))
    return events, trades


def _summary(events: list[dict[str, Any]], trades: list[dict[str, Any]], bars: list[dict[str, Any]]) -> dict[str, Any]:
    proposed_events = [event for event in events if event["state"] == "PROPOSED_SETUP"]
    setup_ids = {event["setup_id"] for event in proposed_events}
    terminal_events = [event for event in events if event.get("setup_id") in setup_ids and event["state"] in {"INVALIDATED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "TRADE_EXECUTED"}]
    count = Counter(event["state"] for event in events); terminal_count = Counter(event["state"] for event in terminal_events); values = [_decimal(item["net_pnl"]) for item in trades]
    gains = sum((value for value in values if value > 0), Decimal()); losses = -sum((value for value in values if value < 0), Decimal()); running = peak = drawdown = Decimal(); losing = longest = 0
    for value in values:
        running += value; peak = max(peak, running); drawdown = max(drawdown, peak - running); losing = losing + 1 if value <= 0 else 0; longest = max(longest, losing)
    monthly: dict[str, Any] = {}
    for month in [f"2024-{number:02d}" for number in range(1, 8)]:
        mt = [item for item in trades if item["day"].startswith(month)]; mv = [_decimal(item["net_pnl"]) for item in mt]
        monthly[month] = {"five_minute_bar_count": sum(str(b["session_date"]).startswith(month) for b in bars), "executed_trades": len(mt), "wins": sum(v > 0 for v in mv), "losses": sum(v <= 0 for v in mv), "net_pnl": str(sum(mv, Decimal()))}
    proposed = len(proposed_events); components = sum(terminal_count[state] for state in ("INVALIDATED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "TRADE_EXECUTED"))
    net_rs = [_decimal(item["net_r"]) for item in trades]
    return {"session_count": len({str(b["session_date"]) for b in bars}), "five_minute_bar_count": len(bars), "breakouts": count["BREAKOUT_DETECTED"], "volume_qualified_breakouts": count["BREAKOUT_VOLUME_QUALIFIED"], "accepted_breakouts": count["ACCEPTANCE_CONFIRMED"], "cvd_confirmed_breakouts": count["CVD_CONFIRMED"], "pullback_triggers": count["PULLBACK_CONFIRMED"], "proposed_setups": proposed, "invalid_setups": terminal_count["INVALIDATED"], "non_executable_setups": terminal_count["NO_EXECUTABLE_ENTRY"], "compliance_blocks": terminal_count["COMPLIANCE_BLOCKED"], "executed_trades": len(trades), "wins": sum(v > 0 for v in values), "losses": sum(v <= 0 for v in values), "win_rate": str(Decimal(sum(v > 0 for v in values)) / Decimal(len(values))) if values else "0", "gross_pnl": str(sum((_decimal(item["gross_pnl"]) for item in trades), Decimal())), "net_pnl": str(sum(values, Decimal())), "total_costs": str(sum((_decimal(item["fees"]) + _decimal(item["slippage_cost"]) for item in trades), Decimal())), "profit_factor": str(gains / losses) if losses else None, "average_trade": str(sum(values, Decimal()) / Decimal(len(values))) if values else "0", "average_r": str(sum(net_rs, Decimal()) / Decimal(len(net_rs))) if net_rs else "0", "average_r_basis": "net_pnl_divided_by_initial_risk_after_fees_and_slippage", "maximum_drawdown": str(drawdown), "longest_losing_streak": longest, "zero_trade_reason": "NO_PROPOSED_SETUP" if not values and not proposed else None, "monthly": monthly, "funnel_reconciliation": {"formula": "proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades", "proposed_setups": proposed, "components_total": components, "reconciles": proposed == components}}


def run_btc_2024_study(*, data_manifest: str, artifact_root: str, repository_root: str, non_interactive: bool = False) -> dict[str, Any]:
    manifest_path = Path(data_manifest).resolve(); manifest = AggregateTradeImporter(".").validate_monthly_manifest(manifest_path)
    if manifest.normalized_dataset_hash != DATASET_HASH or manifest.symbol != "BTCUSDT" or str(manifest.date_start) != "2024-01-01" or str(manifest.date_end) != "2024-07-31":
        raise ValueError("ValueAreaAcceptance requires the pinned BTCUSDT Jan-Jul 2024 immutable dataset")
    expected = [("A", "1.50", 2, "1.50"), ("B", "1.25", 2, "1.50"), ("C", "1.50", 2, "2.00"), ("D", "1.25", 1, "2.00")]
    actual = [(v.variant_id, str(v.breakout_volume_multiplier), v.acceptance_bars, str(v.target_r_multiple)) for v in SEALED_VARIANTS]
    if actual != expected: raise ValueError("sealed ValueAreaAcceptance A-D registry mismatch")
    run_id = ValueAreaAcceptanceAdapter.deterministic_run_id(DATASET_HASH); root = Path(artifact_root).resolve() / STRATEGY_ID / run_id
    registry = {"strategy_id": STRATEGY_ID, "study_label": STUDY_LABEL, "adapter_identity": ADAPTER_ID, "dataset_hash": DATASET_HASH, "manifest_path": str(manifest_path), "variants": [v.model_dump(mode="json") for v in SEALED_VARIANTS], "confirmation_evidence": False, "optimization_claimed": False, "selection_prohibited": True, "requires_future_holdout": True, "best_variant": None, "recommendation": None}
    _write_once(root / "study-manifest.json", registry)
    bar_info = materialize_bars(str(manifest_path), str(Path(repository_root).resolve() / ".tmp")); bars = pq.ParquetFile(bar_info["bar_path"]).read().to_pylist()
    results = []
    for variant in SEALED_VARIANTS:
        events, trades = _scan(bars, variant); summary = _summary(events, trades, bars); variant_run_id = _hash({"run_id": run_id, "variant": variant.variant_id})[:16]; variant_root = root / variant.variant_id
        specification = {**registry, "variant": variant.model_dump(mode="json"), "variant_run_id": variant_run_id, "parameters": {"entry_execution": "next_bar_open", "value_area_percentage": "0.70", "volume_lookback_bars": 10, "same_bar_stop_target_policy": "stop_first", "maximum_trades_per_day": 1}}
        _write_once(variant_root / "specification.json", specification); _write_once(variant_root / "immutable-parameter-manifest.json", specification); _write_once(variant_root / "trades.json", trades); _write_once(variant_root / "diagnostics.json", {"status": "COMPLETED", "source_manifest": str(manifest_path), "source_dataset_hash": DATASET_HASH, "bar_manifest": bar_info, "event_count": len(events), "trade_count": len(trades)}); _write_once(variant_root / "monthly-results.json", summary["monthly"])
        event_path = variant_root / "strategy-events.parquet"; variant_root.mkdir(parents=True, exist_ok=True)
        if not event_path.exists(): pq.write_table(pa.Table.from_pylist(events), event_path, compression="zstd")
        report = {"status": STUDY_LABEL, "strategy_id": STRATEGY_ID, "variant_id": variant.variant_id, "run_id": run_id, "variant_run_id": variant_run_id, "dataset_hash": DATASET_HASH, **summary, "selection_prohibited": True, "optimization_claimed": False, "confirmation_evidence": False, "requires_future_holdout": True}
        _write_once(variant_root / "report.json", report); results.append({"variant_id": variant.variant_id, "variant_run_id": variant_run_id, "report_path": str(variant_root / "report.json"), "summary": summary})
    comparison = {**registry, "status": "EXPLORATORY_IN_SAMPLE_COMPLETE_NO_SELECTION", "run_id": run_id, "comparison_report": str(root / "comparison-report.json"), "results": results}; _write_once(root / "comparison-report.json", comparison)
    return {"implementation_status": "COMPLETED_INDEPENDENT_EVENT_DRIVEN_RUNNER", "status": comparison["status"], "run_id": run_id, "dataset_hash": DATASET_HASH, "comparison_report_path": str(root / "comparison-report.json"), "results": [{"variant_id": item["variant_id"], **{key: item["summary"][key] for key in ("executed_trades", "wins", "losses", "gross_pnl", "net_pnl", "maximum_drawdown")}} for item in results], "warnings": ["EXPLORATORY_IN_SAMPLE only; no variant selection or recommendation.", "Binance BTCUSDT perpetual proxy evidence; not CME MBT or Alpha Futures performance."], "rerun_command": f"python -m research_pipeline value-area-acceptance run-btc-2024-study --data-manifest {manifest_path} --artifact-root {Path(artifact_root).resolve()} --repository-root {Path(repository_root).resolve()} --non-interactive"}
