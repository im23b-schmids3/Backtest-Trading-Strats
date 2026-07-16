import pandas as pd
import pytest

from fib_backtester.data.validation import validate_ohlcv


def test_duplicate_timestamps_are_rejected():
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")] * 2)
    frame = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [0.5, 0.5], "close": [1, 1], "volume": [1, 1]}, index=index)
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)


def test_gap_diagnostic_is_a_clean_validation_error():
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-01 02:00", tz="UTC")])
    frame = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [.5, .5], "close": [1, 1], "volume": [1, 1]}, index=index)
    with pytest.raises(ValueError, match="missing candle"):
        validate_ohlcv(frame, "1h")
