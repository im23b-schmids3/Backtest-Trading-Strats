from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd

from fib_backtester.config import AssetConfig
from .cache import Cache
from .validation import validate_ohlcv

_TIMEDELTAS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def download(asset: str, timeframe: str, asset_config: AssetConfig, start: str, cache: Cache, refresh: bool = False) -> pd.DataFrame:
    if cache.path(asset, timeframe).exists() and not refresh:
        cached = cache.read(asset, timeframe, asset_config.source == "yfinance")
        if cached.index[0] <= pd.Timestamp(start) and cached.index[-1] >= _completed_cutoff(timeframe):
            return cached
    if asset_config.source == "binance":
        frame = _download_ccxt(asset_config.symbol, timeframe, start)
    elif asset_config.source == "yfinance":
        frame = _download_yahoo(asset_config.symbol, timeframe, start)
    else:
        raise ValueError(f"unsupported source {asset_config.source}")
    frame = _completed_only(frame, timeframe)
    if frame.empty:
        raise RuntimeError(f"{asset} {timeframe}: source supplied no completed candles")
    requested_start = pd.Timestamp(start)
    permitted_start = requested_start + (pd.Timedelta(days=7) if asset_config.source == "yfinance" and timeframe == "1d" else pd.Timedelta(0))
    if frame.index[0] > permitted_start:
        raise RuntimeError(f"{asset} {timeframe}: source history begins {frame.index[0]}, after requested start {requested_start}; data is unavailable")
    cache.write(asset, timeframe, frame, asset_config.source == "yfinance")
    return frame


def _download_ccxt(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("CCXT is required for crypto data; install project dependencies") from exc
    exchange = ccxt.binance({"enableRateLimit": True})
    since = int(pd.Timestamp(start).timestamp() * 1000)
    rows: list[list[float]] = []
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    for _ in range(20_000):
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
                break
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                if attempt == 2:
                    raise RuntimeError(f"temporary CCXT data failure for {symbol}: {exc}") from exc
                time.sleep(2 ** attempt)
        assert batch is not None
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + 1
        if next_since <= since or batch[-1][0] >= now_ms:
            break
        since = next_since
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if frame.empty:
        raise RuntimeError(f"CCXT returned no OHLCV data for {symbol}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.set_index("timestamp")


def _download_yahoo(symbol: str, timeframe: str, start: str) -> pd.DataFrame:
    if timeframe == "4h":
        raise RuntimeError("Yahoo Finance does not provide a reliable 4-hour GC=F interval; choose a configured alternative source")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for gold proxy data; install project dependencies") from exc
    frame = yf.download(symbol, start=pd.Timestamp(start).date(), interval=timeframe, auto_adjust=False, progress=False)
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} {timeframe}; 1h history is availability-limited")
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [frame.columns.get_level_values(level).astype(str).str.lower() for level in range(frame.columns.nlevels)]
        ohlc_level = next((level for level, values in enumerate(levels) if {"open", "high", "low", "close"}.issubset(set(values))), None)
        if ohlc_level is None:
            raise RuntimeError(f"Yahoo returned an unrecognized column layout for {symbol}")
        frame.columns = frame.columns.get_level_values(ohlc_level)
    frame = frame.rename(columns=str.lower)
    if "close" not in frame and "adj close" in frame:
        frame = frame.rename(columns={"adj close": "close"})
    elif "adj close" in frame:
        frame = frame.drop(columns="adj close")
    frame["volume"] = frame.get("volume", 0.0)
    return frame[["open", "high", "low", "close", "volume"]]


def _completed_only(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    data = validate_ohlcv(frame)
    cutoff = _completed_cutoff(timeframe)
    return data[data.index <= cutoff]


def _completed_cutoff(timeframe: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    step = _TIMEDELTAS[timeframe]
    return now.floor(step) - step
