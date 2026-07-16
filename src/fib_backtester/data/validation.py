from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(frame: pd.DataFrame, timeframe: str | None = None, allow_expected_gaps: bool = False) -> pd.DataFrame:
    """Return normalized UTC OHLCV or raise a clear data-quality error."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" not in data:
            raise ValueError("OHLCV requires a DatetimeIndex or timestamp column")
        data.index = pd.to_datetime(data.pop("timestamp"), utc=True)
    elif data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError("duplicate candle timestamps detected")
    if data.empty:
        raise ValueError("no OHLCV candles received")
    if (data[["high", "low"]].min(axis=1) <= 0).any() or (data["high"] < data["low"]).any():
        raise ValueError("invalid high/low prices")
    if ((data["open"] > data["high"]) | (data["open"] < data["low"]) | (data["close"] > data["high"]) | (data["close"] < data["low"])).any():
        raise ValueError("OHLC open/close lies outside high-low range")
    if timeframe and not allow_expected_gaps:
        expected = pd.Timedelta({"1h": "1h", "4h": "4h", "1d": "1d"}[timeframe])
        gaps = data.index.to_series().diff().dropna() > expected
        if gaps.any():
            examples = gaps[gaps].index[:3].tolist()
            raise ValueError(f"missing candle intervals detected before {examples}")
    return data.astype(float)
