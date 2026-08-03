from datetime import datetime, timezone
from decimal import Decimal

from research_pipeline.value_area_trap.es_tick_rule import LABELS, retained_duplicate_counts, tick_rule_direction
from research_pipeline.value_area_trap.es_tick_rule import BAR_SCHEMA, _bar_table
from research_pipeline.value_area_trap.thesnowguru import _column_map, _timestamp_from_row


def test_tick_rule_carries_direction_and_starts_neutral() -> None:
    assert tick_rule_direction(Decimal("10"), None, 0) == 0
    assert tick_rule_direction(Decimal("11"), Decimal("10"), 0) == 1
    assert tick_rule_direction(Decimal("11"), Decimal("11"), 1) == 1
    assert tick_rule_direction(Decimal("10"), Decimal("11"), 1) == -1


def test_tick_rule_pilot_is_never_exact_confirmation_evidence() -> None:
    assert LABELS["exact_cvd"] is False
    assert LABELS["tick_rule_approximated_cvd"] is True
    assert LABELS["confirmation_evidence"] is False
    assert LABELS["optimization_claimed"] is False


def test_retained_duplicate_counts_do_not_use_sparse_source_indexes() -> None:
    stamp = datetime(2018, 1, 2, tzinfo=timezone.utc)
    rows = [(stamp, 900_000, Decimal("10"), Decimal("1")), (stamp, 900_017, Decimal("10"), Decimal("1"))]
    assert retained_duplicate_counts(rows) == (1, 1)


def test_retained_order_is_deterministic_with_invalid_rows_skipped_elsewhere() -> None:
    first = datetime(2018, 1, 2, tzinfo=timezone.utc)
    later = datetime(2018, 1, 2, 0, 5, tzinfo=timezone.utc)
    rows = [(later, 50_000, Decimal("11"), Decimal("1")), (first, 50_123, Decimal("10"), Decimal("1"))]
    assert retained_duplicate_counts(rows) == retained_duplicate_counts(list(reversed(rows))) == (0, 0)


def test_thesnowguru_sp_dates_are_us_month_day_year() -> None:
    row = {"date": "12/31/2018", "time": "17:00:00.000"}
    parsed = _timestamp_from_row(row, _column_map(["date", "time"]))
    assert parsed is not None and parsed[0] == datetime(2018, 12, 31, 17, tzinfo=timezone.utc).replace(tzinfo=None)


def test_neutral_volume_is_preserved_without_affecting_delta() -> None:
    stamp = datetime(2018, 1, 2, tzinfo=timezone.utc)
    bars = {stamp: [Decimal("10"), Decimal("11"), Decimal("9"), Decimal("10"), Decimal("6"), Decimal("2"), Decimal("3"), Decimal("1"), 3]}
    row = _bar_table(bars).to_pylist()[0]
    assert BAR_SCHEMA.names[5:9] == ["total_volume", "buy_volume", "sell_volume", "neutral_volume"]
    assert row["total_volume"] == row["buy_volume"] + row["sell_volume"] + row["neutral_volume"]
    assert row["delta"] == Decimal("-1")
