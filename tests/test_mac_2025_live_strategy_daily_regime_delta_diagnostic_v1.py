import numpy as np
import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_live_strategy_daily_regime_delta_diagnostic_v1 as diagnostic,
)
from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_absorption_relative_normalization as norm


def test_frozen_live_family_handoff_loads_and_hashes():
    configs, digest = norm._load_live_configs()
    assert set(configs) == set(norm.LIVE_TO_TAPE)
    assert len(digest) == 64
    assert len({v["runtime_strategy_identity"] for v in configs.values()}) == 4


def test_trade_identity_is_scoped_to_strategy_and_date():
    base = {"family_id": "EUROPE|EUROPE|CURRENT|HIGH", "trade_id": "same", "date": "2025-03-03"}
    diagnostic._validate_unique_trade_identities([base, {**base, "date": "2025-03-04"}])
    with pytest.raises(diagnostic.DiagnosticError, match="duplicate live trade identity"):
        diagnostic._validate_unique_trade_identities([base, dict(base)])


def test_aggressor_delta_and_volume_use_half_open_interval():
    dtype = [("timestamp_ns", "i8"), ("execution_size", "i4"), ("aggressor", "i1")]
    events = np.array([(10, 2, 1), (20, 3, -1), (30, 5, 1)], dtype=dtype)
    assert diagnostic._window_delta(events, 10, 30) == (-1.0, 5.0)


def test_confirmation_delta_stops_at_confirming_tape_record_for_equal_timestamps():
    dtype = [("timestamp_ns", "i8"), ("execution_size", "i4"), ("aggressor", "i1"), ("execution_price", "f8")]
    events = np.array([(10, 2, 1, 100.25), (10, 50, -1, 99.75)], dtype=dtype)
    candidate = {"interaction_end_ns": 0, "interaction_end_price": 100.0}
    trade = {"direction": "LONG", "trade_id": "x", "entry_timestamp_ns": 2_000_010}
    ts, index, delta, volume = diagnostic._confirm(candidate, trade, events,
        {"min_confirmation_seconds": 0, "max_confirmation_seconds": 1, "favorable_ticks": 1, "entry_delay_ms": 0})
    assert (ts, index, delta, volume) == (10, 0, 2.0, 2.0)


def test_zero_distance_stop_gap_uses_frozen_gross_risk_denominator():
    events = np.array([(1, 100.0, 100.25), (2, 100.25, 100.5)], dtype=[
        ("timestamp_ns", "i8"), ("bid", "f8"), ("ask", "f8")
    ])
    trade = {"entry_timestamp_ns": 1, "exit_timestamp_ns": 2, "direction": "LONG",
             "entry": 100.0, "stop": 100.0, "gross_pnl_usd": -50.0, "gross_r": -1.0,
             "contracts": 1, "point_value_usd": 50.0, "exit_reason": "STOP"}
    result = diagnostic._path(trade, events)
    assert result["MFE_R"] == pytest.approx(0.25)
    assert result["MAE_R"] == pytest.approx(0.0)
    assert result["stop_before_target"] is True


def test_daily_metrics_group_by_family_and_keep_first_vs_later_trade():
    rows = [
        {"family_id": "A", "trade_id": "1", "entry_timestamp_ns": 1, "r_multiple": -1.0,
         "setup_id": "x", "MFE_R": 0.5, "MAE_R": 1.0, "target_before_stop": False,
         "stop_before_target": True, "context": {"DIRECTIONAL_SESSION_CVD_RATIO": -0.2}},
        {"family_id": "A", "trade_id": "2", "entry_timestamp_ns": 2, "r_multiple": 2.0,
         "setup_id": "x", "MFE_R": 2.0, "MAE_R": 0.2, "target_before_stop": True,
         "stop_before_target": False, "context": {"DIRECTIONAL_SESSION_CVD_RATIO": 0.3}},
    ]
    day = diagnostic._strategy_days(rows, "2025-03-03", "SPRING_2025")[0]
    assert day["trade_count"] == 2
    assert day["net_r"] == pytest.approx(1.0)
    assert day["first_trade_r"] == -1.0
    assert day["later_trades_net_r"] == 2.0
    assert day["cumulative_r_after_trade_1"] == -1.0
    assert day["cumulative_r_after_trade_2"] == 1.0


def test_calendar_includes_zero_trade_dates_and_uses_pooled_drawdown():
    days = [{"date": "2025-03-03", "period": "SPRING_2025", "family": "A", "net_r": 1.0,
             "trade_count": 2, "win_count": 1, "loss_count": 1, "max_intraday_drawdown_r": -1.0}]
    trades = [
        {"date": "2025-03-03", "family_id": "A", "trade_id": "1", "entry_timestamp_ns": 1, "r_multiple": 1.0},
        {"date": "2025-03-03", "family_id": "A", "trade_id": "2", "entry_timestamp_ns": 2, "r_multiple": -2.0},
    ]
    cal = diagnostic._calendar(days, trades, ["2025-03-03", "2025-03-04"])
    assert len(cal) == 2
    assert cal[0]["total_max_intraday_dd_r"] == -2.0
    assert cal[1]["classification"] == "FLAT"
    assert cal[1]["total_trades"] == 0


def test_spring_cutpoints_are_family_specific_and_reusable_for_october():
    rows = []
    families = list(diagnostic.FAMILIES)
    for family in families:
        family_offset = families.index(family) * 30
        for period, period_offset in (("SPRING_2025", 0), ("OCTOBER_2025", 100)):
            for i in range(9):
                rows.append({"family": family, "period": period, "date": f"2025-03-{i + 1:02d}",
                    "trade_count": 1, "net_r": float(i % 3 - 1), "avg_r_per_trade": float(i % 3 - 1),
                    "median_mfe_r": 1.0, "median_mae_r": 0.5, "target_before_stop_count": 0,
                    "first_signal_context": {f: float(i + family_offset + period_offset) for f in diagnostic.FEATURES}})
    result = diagnostic._analyze(rows)
    assert result["spring_cutpoints"][families[0]][diagnostic.FEATURES[0]] != result["spring_cutpoints"][families[1]][diagnostic.FEATURES[0]]
    assert result["october_compatibility"][diagnostic.FEATURES[0]]["spring_cutpoints_by_family"][families[0]] == result["spring_cutpoints"][families[0]][diagnostic.FEATURES[0]]
