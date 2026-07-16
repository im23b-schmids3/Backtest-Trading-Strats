import pandas as pd

from fib_backtester.research.v5_trade_path import _conditional_probabilities, _fib_price


def test_counterfactual_fibonacci_prices_match_long_and_short_orientation():
    assert _fib_price("long", 100.0, 120.0, 0.9) == 102.0
    assert _fib_price("short", 100.0, 120.0, 0.9) == 118.0


def test_conditional_probabilities_use_stage_reached_denominators():
    trades = pd.DataFrame([
        {"tp1_reached": True, "tp2_reached": True, "tp3_reached": False, "tp4_reached": False, "tp5_reached": False,
         "tp2_after_previous": True, "tp3_after_previous": False, "tp4_after_previous": False, "tp5_after_previous": False,
         "stop_before_tp1": False, "stop_after_tp1": True, "stop_after_tp2": True, "stop_after_tp3": False},
        {"tp1_reached": True, "tp2_reached": False, "tp3_reached": False, "tp4_reached": False, "tp5_reached": False,
         "tp2_after_previous": False, "tp3_after_previous": False, "tp4_after_previous": False, "tp5_after_previous": False,
         "stop_before_tp1": False, "stop_after_tp1": True, "stop_after_tp2": False, "stop_after_tp3": False},
        {"tp1_reached": False, "tp2_reached": False, "tp3_reached": False, "tp4_reached": False, "tp5_reached": False,
         "tp2_after_previous": False, "tp3_after_previous": False, "tp4_after_previous": False, "tp5_after_previous": False,
         "stop_before_tp1": True, "stop_after_tp1": False, "stop_after_tp2": False, "stop_after_tp3": False},
    ])
    result = _conditional_probabilities(trades).set_index("metric")
    assert result.loc["P(TP2 | TP1)", "numerator"] == 1
    assert result.loc["P(TP2 | TP1)", "denominator"] == 2
    assert result.loc["P(Stop before TP1)", "probability"] == 1 / 3
