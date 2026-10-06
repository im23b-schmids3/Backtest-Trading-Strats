from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_sato_overnight_liquidity_sweep_reclaim_v1 as sato


def test_timezone_derived_overnight_start_covers_standard_and_daylight_time() -> None:
    assert sato.expected_overnight_start_utc("2025-03-07") == datetime(2025, 3, 6, 23, tzinfo=timezone.utc)
    assert sato.expected_overnight_start_utc("2025-03-10") == datetime(2025, 3, 9, 22, tzinfo=timezone.utc)
    assert sato.expected_overnight_start_utc("2025-10-07") == datetime(2025, 10, 6, 22, tzinfo=timezone.utc)
    assert sato.expected_overnight_start_utc("2025-11-03") == datetime(2025, 11, 2, 23, tzinfo=timezone.utc)


def test_all_downloaded_patch_and_native_boundaries_validate() -> None:
    rows, summary = sato.validate_patch_and_reconstruct(write_artifacts=False)
    assert len(rows) == 54
    assert sum(row["PATCH_BAR_COUNT"] for row in rows) == 103
    assert summary["patch_invalid_requests"] == 0
    assert summary["boundary_valid_sessions"] == 54
    assert all(row["STATUS"] == "VERIFIED" for row in rows)
    assert all(row["PATCH_END_UTC"] == row["NATIVE_START_UTC"] for row in rows)
    assert all(row["RAW_ES_SYMBOL"] for row in rows)
    assert {row["PATCH_BAR_COUNT"] for row in rows} == {1, 2}


@pytest.mark.parametrize("patch_highs,patch_lows,native_high,native_low,high,low,high_source,low_source", [
    ([101.0], [99.0], 100.0, 100.0, 101.0, 99.0, "PATCH", "PATCH"),
    ([99.0], [98.0], 101.0, 97.0, 101.0, 97.0, "NATIVE", "NATIVE"),
    ([101.0, 102.0], [98.0, 99.0], 102.0, 98.0, 102.0, 98.0, "BOTH_EQUAL", "BOTH_EQUAL"),
])
def test_patch_and_native_extremes_and_source_attribution(
    patch_highs, patch_lows, native_high, native_low, high, low, high_source, low_source
) -> None:
    result = sato.combine_overnight_extremes(patch_highs, patch_lows, native_high, native_low)
    assert result["overnight_high"] == high
    assert result["overnight_low"] == low
    assert result["overnight_high_source"] == high_source
    assert result["overnight_low_source"] == low_source


def test_empty_or_malformed_patch_fails_closed() -> None:
    with pytest.raises(sato.OvernightStudyError):
        sato.combine_overnight_extremes([], [], 100.0, 99.0)
    with pytest.raises(sato.OvernightStudyError):
        sato.combine_overnight_extremes([100.0], [], 100.0, 99.0)


def test_overnight_level_adapter_uses_onh_onl_for_frozen_event_engine() -> None:
    adapted = sato._level_adapter({"OVERNIGHT_HIGH": 101.0, "OVERNIGHT_LOW": 99.0, "OVERNIGHT_RANGE_TICKS": 8.0})
    assert adapted["pdh"] == adapted["overnight_high"] == 101.0
    assert adapted["pdl"] == adapted["overnight_low"] == 99.0
    assert adapted["midpoint"] == 100.0
    assert adapted["range_ticks"] == 8.0
    assert adapted["level_source"] == "OHLCV1H_PATCH_PLUS_NATIVE_MBP10_TRADES"


def test_exported_event_labels_are_overnight_not_prior_day() -> None:
    event = {"side": "PDH", "level_type": "PDH", "prior_day_high": 101.0,
             "prior_day_low": 99.0, "prior_day_midpoint": 100.0, "prior_day_range_ticks": 8.0,
             "first_touch": {"prior_midpoint": {"result": "STOP_FIRST"},
                             "opposite_prior_extreme": {"result": "TARGET_FIRST"}}}
    result = sato._rename_event_level_fields(event)
    assert result["side"] == "ONH"
    assert result["level_type"] == "ONH"
    assert result["level_family"] == "OVERNIGHT_HIGH_LOW"
    assert result["overnight_high"] == 101.0
    assert result["overnight_low"] == 99.0
    assert "prior_day_high" not in result
    assert result["first_touch"]["overnight_midpoint"]["result"] == "STOP_FIRST"
    assert result["first_touch"]["opposite_overnight_extreme"]["result"] == "TARGET2_FIRST"


def test_period_first_touch_counts_only_reclaimed_events() -> None:
    events = [
        {"period": "SPRING_2025", "reclaim_only": True,
         "first_touch": {"session_vwap": {"result": "VWAP_FIRST"},
                         "overnight_midpoint": {"result": "STOP_FIRST"},
                         "opposite_overnight_extreme": {"result": "TARGET2_FIRST"}}},
        {"period": "SPRING_2025", "reclaim_only": False, "first_touch": {}},
    ]
    result = sato._period_first_touch(events)
    assert result["SPRING_2025"]["VWAP"]["results"] == {"VWAP_FIRST": 1}
    assert result["SPRING_2025"]["OPPOSITE_OVERNIGHT_EXTREME"]["results"] == {"TARGET2_FIRST": 1}
    assert result["OCTOBER_2025"]["VWAP"]["n"] == 0
