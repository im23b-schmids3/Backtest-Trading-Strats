from decimal import Decimal

from datetime import datetime, timezone

import pyarrow as pa

from research_pipeline.value_area_acceptance import BAR_SCHEMA, SEALED_VARIANTS, STRATEGY_ID, ValueAreaAcceptanceAdapter, _aggregate_batch, _arrow_bar, _merge_bar, _scan, _summary


def _bar(day: str, index: int, price: str, *, low: str | None = None, high: str | None = None, volume: str = "2", delta: str = "1") -> dict:
    return {
        "timestamp_utc": f"{day}T00:{index:02d}:00+00:00", "session_date": day,
        "open": price, "high": high or price, "low": low or price, "close": price,
        "volume": volume, "cvd_delta": delta,
    }


def test_exact_sealed_registry_and_identity() -> None:
    assert [(v.variant_id, str(v.breakout_volume_multiplier), v.acceptance_bars, str(v.target_r_multiple)) for v in SEALED_VARIANTS] == [("A", "1.50", 2, "1.50"), ("B", "1.25", 2, "1.50"), ("C", "1.50", 2, "2.00"), ("D", "1.25", 1, "2.00")]
    assert ValueAreaAcceptanceAdapter.strategy_id == STRATEGY_ID


def test_completed_bar_acceptance_cvd_and_boundary_pullback() -> None:
    assert ValueAreaAcceptanceAdapter.acceptance_confirmed([Decimal("101"), Decimal("102")], Decimal("100"), side="LONG", count=2)
    assert ValueAreaAcceptanceAdapter.cvd_confirmed(Decimal("1"), Decimal("2"), side="LONG")
    assert ValueAreaAcceptanceAdapter.pullback_confirmed(Decimal("99"), Decimal("102"), Decimal("100"), Decimal("100"), side="LONG")
    assert ValueAreaAcceptanceAdapter.volume_qualified(Decimal("15"), [Decimal("10")]*10, Decimal("1.5"))
    assert ValueAreaAcceptanceAdapter.fixed_r_exit(Decimal("100"), Decimal("98"), Decimal("2"), side="LONG") == Decimal("104")
    assert ValueAreaAcceptanceAdapter.deterministic_trade_id("2024-01-01", "LONG", "2024-01-01T00:05:00Z", "A") == ValueAreaAcceptanceAdapter.deterministic_trade_id("2024-01-01", "LONG", "2024-01-01T00:05:00Z", "A")


def test_long_lifecycle_enters_next_open_and_stops_first() -> None:
    result=ValueAreaAcceptanceAdapter.close_setup(day="2024-01-01",side="LONG",pullback={"low":"99","high":"101"},next_bar={"timestamp":"2024-01-01T00:10:00Z","open":"100"},later_bars=[{"low":"98","high":"104","close":"101"}],variant=SEALED_VARIANTS[0])
    assert result["state"] == "TRADE_EXECUTED"
    assert result["trade"]["exit_reason"] == "STOP_FIRST"
    assert Decimal(result["trade"]["net_r"]) < 0


def test_sequential_long_setup_has_one_terminal_classification() -> None:
    previous = [_bar("2024-01-01", i, "100", volume="1", delta="0") for i in range(10)]
    current = [
        _bar("2024-01-02", 0, "120", high="121", volume="20", delta="2"),
        _bar("2024-01-02", 1, "121", high="122", volume="2", delta="2"),
        _bar("2024-01-02", 2, "110", low="109", high="111", volume="2", delta="2"),
        _bar("2024-01-02", 3, "111", high="115", volume="2", delta="2"),
    ]
    events, trades = _scan(previous + current, SEALED_VARIANTS[0])
    proposed = [event for event in events if event["state"] == "PROPOSED_SETUP"]
    terminal = [event for event in events if event.get("setup_id") and event["state"] in {"INVALIDATED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "TRADE_EXECUTED"}]
    assert len(proposed) == len(terminal) == 1
    assert len(trades) == 1
    assert [event["state"] for event in events[:7]] == ["BREAKOUT_DETECTED", "BREAKOUT_VOLUME_QUALIFIED", "ACCEPTANCE_CONFIRMED", "CVD_CONFIRMED", "PULLBACK_CONFIRMED", "PROPOSED_SETUP", "TRADE_EXECUTED"]


def test_short_lifecycle_and_next_bar_open() -> None:
    previous = [_bar("2024-01-01", i, "100", volume="1", delta="0") for i in range(10)]
    current = [
        _bar("2024-01-02", 0, "80", low="79", volume="20", delta="-2"),
        _bar("2024-01-02", 1, "79", low="78", volume="2", delta="-2"),
        _bar("2024-01-02", 2, "90", low="89", high="100", volume="2", delta="-2"),
        _bar("2024-01-02", 3, "89", low="85", volume="2", delta="-2"),
    ]
    events, trades = _scan(previous + current, SEALED_VARIANTS[3])
    assert trades and trades[0]["side"] == "SHORT"
    assert trades[0]["entry_timestamp"].endswith("00:03:00+00:00")
    assert any(event["state"] == "TRADE_EXECUTED" for event in events)


def test_missing_next_bar_is_terminal_and_funnel_reconciles() -> None:
    previous = [_bar("2024-01-01", i, "100", volume="1", delta="0") for i in range(10)]
    current = [_bar("2024-01-02", 0, "120", volume="20", delta="2"), _bar("2024-01-02", 1, "121", volume="2", delta="2"), _bar("2024-01-02", 2, "110", low="109", volume="2", delta="2")]
    events, trades = _scan(previous + current, SEALED_VARIANTS[0])
    summary = _summary(events, trades, previous + current)
    assert summary["non_executable_setups"] == 1
    assert summary["funnel_reconciliation"]["reconciles"] is True


def test_second_valid_setup_is_explicitly_compliance_blocked_after_trade() -> None:
    previous = [_bar("2024-01-01", i, "100", volume="1", delta="0") for i in range(10)]
    current = [
        _bar("2024-01-02", 0, "120", high="121", volume="20", delta="2"),
        _bar("2024-01-02", 1, "121", high="122", volume="2", delta="2"),
        _bar("2024-01-02", 2, "110", low="109", high="111", volume="2", delta="2"),
        _bar("2024-01-02", 3, "111", high="115", volume="2", delta="2"),
        _bar("2024-01-02", 4, "120", high="121", volume="20", delta="2"),
        _bar("2024-01-02", 5, "121", high="122", volume="2", delta="2"),
        _bar("2024-01-02", 6, "110", low="109", high="111", volume="2", delta="2"),
        _bar("2024-01-02", 7, "111", high="115", volume="2", delta="2"),
    ]
    events, trades = _scan(previous + current, SEALED_VARIANTS[0])
    summary = _summary(events, trades, previous + current)
    assert len(trades) == 1
    assert summary["compliance_blocks"] == 1
    assert summary["funnel_reconciliation"]["reconciles"] is True


def test_profile_isolation_and_deterministic_replay() -> None:
    previous = [_bar("2024-01-01", i, "100", volume="1", delta="0") for i in range(10)]
    current = [_bar("2024-01-02", i, "100", volume="1", delta="0") for i in range(12)]
    first = _scan(previous + current, SEALED_VARIANTS[0])
    second = _scan(previous + current, SEALED_VARIANTS[0])
    assert first == second
    assert not any(event["state"] == "BREAKOUT_DETECTED" for event in first[0])


def test_bar_arrow_serialization_preserves_decimal_values() -> None:
    row = _bar("2024-01-01", 0, "100.25", volume="3.125", delta="1.125")
    row["timestamp_utc"] = datetime(2024, 1, 1, tzinfo=timezone.utc)
    row["open"] = row["high"] = row["low"] = row["close"] = Decimal("100.25")
    row["volume"] = Decimal("3.125")
    row["buy_volume"] = Decimal("2.125")
    row["sell_volume"] = Decimal("1")
    row["cvd_delta"] = Decimal("1.125")
    table = pa.Table.from_pylist([_arrow_bar(row)], schema=BAR_SCHEMA)
    assert table.to_pylist()[0]["volume"] == "3.125"


def test_batch_aggregation_and_cross_batch_merge_preserve_ohlcv_cvd() -> None:
    batch = pa.RecordBatch.from_pylist([
        {"trade_time_utc": "2024-01-01T00:00:00.000000Z", "price": "100.0", "quantity_base": "1", "buyer_is_maker": False},
        {"trade_time_utc": "2024-01-01T00:04:00.000000Z", "price": "101.0", "quantity_base": "2", "buyer_is_maker": True},
    ])
    bar = _aggregate_batch(batch)[0]
    assert (bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"], bar["cvd_delta"]) == (Decimal("100.0"), Decimal("101.0"), Decimal("100.0"), Decimal("101.0"), Decimal("3"), Decimal("-1"))
    follow = dict(bar); follow.update({"open": Decimal("102"), "high": Decimal("103"), "low": Decimal("99"), "close": Decimal("102"), "volume": Decimal("4"), "buy_volume": Decimal("4"), "sell_volume": Decimal("0"), "cvd_delta": Decimal("4")})
    output: list[dict] = []
    merged = _merge_bar(bar, follow, output)
    assert not output and merged["high"] == Decimal("103") and merged["low"] == Decimal("99") and merged["close"] == Decimal("102") and merged["volume"] == Decimal("7") and merged["cvd_delta"] == Decimal("3")
