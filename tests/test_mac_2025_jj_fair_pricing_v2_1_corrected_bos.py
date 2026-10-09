from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_jj_fair_pricing_v2_1_corrected_bos as corrected,
)


def test_all_frozen_bos_candle_semantics_have_synthetic_examples():
    result = corrected._candle_tests()
    assert result["all_pass"] is True
    assert result["tests"]["valid_bullish_bos"]
    assert result["tests"]["valid_bearish_bos"]
    assert result["tests"]["same_candle_bullish_combined"]
    assert result["tests"]["same_candle_bearish_combined"]
    assert result["tests"]["missing_prior_candle_fails_closed"]


def test_actual_v2_signal_catalog_accepts_both_combined_directions_and_phase_directions():
    result = corrected._synthetic_production_catalog()
    assert result["production_signal_catalog_all_direction_cases_pass"]
    assert result["entry_ordering_pass"]
    for name in ("long_continuation", "short_continuation", "long_reversion", "short_reversion"):
        assert result["cases"][name]["combined_accepted"]
        assert result["cases"][name]["timestamp_equals_completed_bar_end"]


def test_directional_bos_is_strict_and_requires_both_completed_prior_bars():
    bars = {
        1: {"high": 101, "low": 99, "close": 100},
        2: {"high": 102, "low": 98, "close": 100},
        3: {"high": 103, "low": 97, "close": 102},
    }
    assert corrected.directional_bos_for_bar(bars, 3, 1)["confirmed"] is False
    bars[3]["close"] = 102.0001
    assert corrected.directional_bos_for_bar(bars, 3, 1)["confirmed"] is True
    assert corrected.directional_bos_for_bar({2: bars[2], 3: bars[3]}, 3, 1)["status"] == "MISSING_REFERENCE_CANDLE"
