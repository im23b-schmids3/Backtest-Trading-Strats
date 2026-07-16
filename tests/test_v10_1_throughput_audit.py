import pandas as pd

from fib_backtester.research.v10_1_throughput_audit import _hypothetical_frequency, _parse_timestamp


def test_timestamp_parser_handles_empty_and_timezone_values():
    assert _parse_timestamp("") is None
    assert _parse_timestamp("2024-01-01T00:00:00Z") == pd.Timestamp("2024-01-01T00:00:00Z")


def test_hypothetical_frequency_is_explicitly_linear_and_monotonic():
    timeline = pd.DataFrame({
        "account_size": ["25k", "25k"],
        "lifetime_days": [365.25, 365.25],
        "number_of_trades": [12, 24],
        "payout_count": [1, 2],
        "first_payout_days": [120.0, 240.0],
    })
    throughput = pd.DataFrame()
    result = _hypothetical_frequency(timeline, throughput)
    assert result.iloc[1].estimated_trades_per_month > result.iloc[0].estimated_trades_per_month
    assert result.iloc[1].estimated_days_to_first_payout < result.iloc[0].estimated_days_to_first_payout
    assert "not a backtest" in result.iloc[0].assumption
