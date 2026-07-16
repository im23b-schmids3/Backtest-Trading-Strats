"""Official Binance exploratory research data layer.

This module deliberately does not depend on the strategy or backtest engines.
It discovers relevant Binance symbols, downloads completed 1H candles through
the public REST market-data endpoints, incrementally updates a local Parquet
cache, and derives deterministic UTC 4H candles from the cached 1H data.
Credentials are optional for these public endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


SPOT_BASE = "https://data-api.binance.vision"
FUTURES_BASE = "https://fapi.binance.com"
INTERVAL = "1h"
INTERVAL_MS = 60 * 60 * 1000
QUOTE_PRIORITY = {"USDT": 0, "USDC": 1, "BUSD": 2, "USD": 3}


@dataclass(frozen=True)
class ResearchMarket:
    name: str
    asset_class: str
    aliases: tuple[str, ...]


MARKETS: tuple[ResearchMarket, ...] = (
    ResearchMarket("BTC", "Crypto", ("BTC",)),
    ResearchMarket("ETH", "Crypto", ("ETH",)),
    ResearchMarket("Gold", "Metal proxy", ("PAXG", "XAUT", "GOLD", "XAU")),
    ResearchMarket("Silver", "Metal proxy", ("XAG", "SILVER", "KAG")),
    ResearchMarket("Oil", "Energy proxy", ("WTI", "BRENT", "OIL", "USOIL")),
    ResearchMarket("QQQ", "Equity proxy", ("QQQ",)),
    ResearchMarket("SPY", "Equity proxy", ("SPY",)),
    ResearchMarket("Nasdaq proxy", "Equity index proxy", ("NQ", "NAS100", "NDX", "USTEC")),
    ResearchMarket("S&P proxy", "Equity index proxy", ("ES", "SPX", "SP500", "US500")),
)


@dataclass
class SymbolCandidate:
    research_market: str
    asset_class: str
    market_type: str
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    contract_type: str
    onboard_date: pd.Timestamp | None
    earliest_candle: pd.Timestamp | None = None
    history_probe_status: str = "not_probed"
    selected: bool = False
    selection_reason: str = ""
    error: str = ""


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overwriting process variables."""

    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class BinancePublicClient:
    """Small stdlib-only client for Binance public market-data REST APIs."""

    def __init__(self, timeout: int = 20, min_request_interval: float = 0.05) -> None:
        _load_dotenv()
        self.timeout = timeout
        self.min_request_interval = min_request_interval
        self.last_request = 0.0
        self.api_key = os.getenv("BINANCE_API_KEY", "")

    def get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        wait = self.min_request_interval - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        query = f"?{urlencode(params)}" if params else ""
        headers = {"User-Agent": "fib-backtester-binance-research/1.0"}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        request = Request(f"{base_url}{path}{query}", headers=headers, method="GET")
        for attempt in range(5):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    self.last_request = time.monotonic()
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                self.last_request = time.monotonic()
                if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == 4:
                    body = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"Binance HTTP {exc.code}: {body}") from exc
                retry_after = float(exc.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, 2**attempt))
            except (URLError, TimeoutError) as exc:
                self.last_request = time.monotonic()
                if attempt == 4:
                    raise RuntimeError(f"Binance network failure: {exc}") from exc
                time.sleep(2**attempt)
        raise RuntimeError("Binance request failed after retries")

    def exchange_info(self, market_type: str) -> dict[str, Any]:
        if market_type == "spot":
            return self.get(SPOT_BASE, "/api/v3/exchangeInfo")
        return self.get(FUTURES_BASE, "/fapi/v1/exchangeInfo")

    def klines(
        self,
        market_type: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        path = "/api/v3/klines" if market_type == "spot" else "/fapi/v1/klines"
        params: dict[str, Any] = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
        if start_ms is not None:
            params["startTime"] = max(0, start_ms)
        if end_ms is not None:
            params["endTime"] = max(0, end_ms)
        return self.get(SPOT_BASE if market_type == "spot" else FUTURES_BASE, path, params)


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0, "0"):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True)


def _candidate_rows(info: dict[str, Any], market_type: str) -> Iterable[SymbolCandidate]:
    by_alias: dict[str, list[ResearchMarket]] = {}
    for market in MARKETS:
        for alias in market.aliases:
            by_alias.setdefault(alias.upper(), []).append(market)
    for item in info.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        symbol = str(item.get("symbol", "")).upper()
        base = str(item.get("baseAsset", "")).upper()
        quote = str(item.get("quoteAsset", "")).upper()
        if quote not in QUOTE_PRIORITY:
            continue
        markets = by_alias.get(base, [])
        if not markets:
            continue
        for market in markets:
            yield SymbolCandidate(
                research_market=market.name,
                asset_class=market.asset_class,
                market_type=market_type,
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                status=str(item.get("status", "")),
                contract_type=str(item.get("contractType", "SPOT")),
                onboard_date=_to_timestamp(item.get("onboardDate")),
            )


def discover_symbols(client: BinancePublicClient) -> list[SymbolCandidate]:
    candidates: list[SymbolCandidate] = []
    errors: list[str] = []
    for market_type in ("spot", "usd_m_futures"):
        try:
            info = client.exchange_info("spot" if market_type == "spot" else "usd_m_futures")
            candidates.extend(_candidate_rows(info, market_type))
        except RuntimeError as exc:
            errors.append(f"{market_type}: {exc}")
    if not candidates and errors:
        raise RuntimeError("Binance symbol discovery failed: " + " | ".join(errors))
    discovered_markets = {candidate.research_market for candidate in candidates}
    for market in MARKETS:
        if market.name not in discovered_markets:
            candidates.append(
                SymbolCandidate(
                    research_market=market.name,
                    asset_class=market.asset_class,
                    market_type="",
                    symbol="",
                    base_asset="",
                    quote_asset="",
                    status="NOT_FOUND",
                    contract_type="",
                    onboard_date=None,
                    history_probe_status="not_found",
                    selection_reason="no matching TRADING Binance symbol discovered",
                )
            )
    # Retain the most useful quote for each venue/symbol while preserving all
    # distinct spot/futures instruments that actually exist.
    return sorted(candidates, key=lambda c: (c.research_market, c.market_type, QUOTE_PRIORITY.get(c.quote_asset, 99), c.symbol))


def probe_history(client: BinancePublicClient, candidate: SymbolCandidate) -> SymbolCandidate:
    try:
        rows = client.klines(candidate.market_type, candidate.symbol, start_ms=0, limit=1)
        if rows:
            candidate.earliest_candle = _to_timestamp(rows[0][0])
            candidate.history_probe_status = "ok"
        else:
            candidate.history_probe_status = "empty"
    except RuntimeError as exc:
        candidate.history_probe_status = "error"
        candidate.error = str(exc)
    return candidate


def select_symbols(client: BinancePublicClient, candidates: list[SymbolCandidate]) -> list[SymbolCandidate]:
    grouped: dict[str, list[SymbolCandidate]] = {}
    for candidate in candidates:
        if not candidate.symbol:
            continue
        grouped.setdefault(candidate.research_market, []).append(probe_history(client, candidate))
    selected: list[SymbolCandidate] = []
    for market_name, rows in grouped.items():
        usable = [r for r in rows if r.earliest_candle is not None]
        if not usable:
            continue
        usable.sort(key=lambda r: (r.earliest_candle, -int(r.market_type == "spot"), QUOTE_PRIORITY[r.quote_asset], r.symbol))
        winner = usable[0]
        winner.selected = True
        winner.selection_reason = (
            "oldest successful Binance 1H history; spot preferred only when history starts equally"
        )
        selected.append(winner)
        for row in rows:
            if row is not winner and not row.selection_reason:
                row.selection_reason = "available candidate not selected because another Binance instrument has older history"
    return sorted(selected, key=lambda r: r.research_market)


def _rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["open", "high", "low", "close", "volume"]].dropna()


def _completed_cutoff() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=1)


def _cache_path(cache_dir: Path, candidate: SymbolCandidate) -> Path:
    return cache_dir / f"{candidate.research_market.replace(' ', '_')}_{candidate.symbol}_{candidate.market_type}_1h.parquet"


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()[["open", "high", "low", "close", "volume"]]


def _fetch_range(client: BinancePublicClient, candidate: SymbolCandidate, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if start > end:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows: list[list[Any]] = []
    cursor = max(0, int(start.timestamp() * 1000))
    end_ms = int(end.timestamp() * 1000)
    while cursor <= end_ms:
        batch = client.klines(candidate.market_type, candidate.symbol, start_ms=cursor, end_ms=end_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + INTERVAL_MS
        if next_cursor <= cursor or len(batch) < 1000:
            break
        cursor = next_cursor
    return _rows_to_frame(rows)


def update_1h_cache(client: BinancePublicClient, candidate: SymbolCandidate, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, candidate)
    cached = _read_cache(path)
    cutoff = _completed_cutoff()
    pieces = [cached]
    if cached.empty:
        pieces.append(_fetch_range(client, candidate, pd.Timestamp("2017-01-01", tz="UTC"), cutoff))
    else:
        pieces.append(_fetch_range(client, candidate, pd.Timestamp("2017-01-01", tz="UTC"), cached.index[0] - pd.Timedelta(hours=1)))
        pieces.append(_fetch_range(client, candidate, cached.index[-1] + pd.Timedelta(hours=1), cutoff))
    frame = pd.concat([piece for piece in pieces if not piece.empty]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[(frame.index <= cutoff)]
    if frame.empty:
        raise RuntimeError(f"{candidate.symbol}: Binance returned no completed 1H candles")
    frame.to_parquet(path)
    return frame


def build_4h(frame_1h: pd.DataFrame) -> pd.DataFrame:
    """Build UTC 4H candles and retain only complete four-candle buckets."""

    grouped = frame_1h.resample("4h", origin="epoch", label="left", closed="left")
    frame = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = grouped["close"].count()
    return frame[counts == 4].dropna()


def _quality_row(
    candidate: SymbolCandidate,
    timeframe: str,
    frame: pd.DataFrame,
    provider: str,
    source_note: str = "",
) -> dict[str, Any]:
    step = pd.Timedelta(hours=1 if timeframe == "1h" else 4)
    first = frame.index.min()
    last = frame.index.max()
    expected = int(((last - first) / step)) + 1
    missing = max(0, expected - len(frame))
    months = (last - first).total_seconds() / (30.4375 * 86400)
    return {
        "research_market": candidate.research_market,
        "symbol": candidate.symbol,
        "market_type": candidate.market_type,
        "timeframe": timeframe,
        "first_candle_utc": first.isoformat(),
        "last_candle_utc": last.isoformat(),
        "months_of_history": round(months, 2),
        "years_of_history": round(months / 12, 2),
        "timeframe_available": "yes",
        "candle_count": len(frame),
        "expected_candle_count": expected,
        "missing_candles": missing,
        "missing_ratio": round(missing / expected, 8) if expected else 0.0,
        "provider": provider,
        "source_note": source_note,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _symbol_dict(candidate: SymbolCandidate) -> dict[str, Any]:
    return {
        "research_market": candidate.research_market,
        "asset_class": candidate.asset_class,
        "market_type": candidate.market_type,
        "symbol": candidate.symbol,
        "base_asset": candidate.base_asset,
        "quote_asset": candidate.quote_asset,
        "status": candidate.status,
        "contract_type": candidate.contract_type,
        "onboard_date_utc": candidate.onboard_date.isoformat() if candidate.onboard_date is not None else "",
        "earliest_1h_candle_utc": candidate.earliest_candle.isoformat() if candidate.earliest_candle is not None else "",
        "history_probe_status": candidate.history_probe_status,
        "selected_for_download": "yes" if candidate.selected else "no",
        "selection_reason": candidate.selection_reason,
        "error": candidate.error,
        "provider": "Binance official REST API",
    }


def write_setup(path: Path, candidates: list[SymbolCandidate], selected: list[SymbolCandidate], quality_rows: list[dict[str, Any]], errors: list[str]) -> None:
    quality_by_market = {row["research_market"]: row for row in quality_rows if row["timeframe"] == "1h"}
    lines = [
        "# Binance exploratory data layer",
        "",
        "Generated by `fib_backtester.infrastructure.binance_data_layer`.",
        "This is an exploratory Binance spot/USD-M data source and is not CME futures data.",
        "No strategy or backtesting code is invoked by this layer.",
        "",
        "## Source and authentication",
        "",
        "The layer uses Binance's public REST market-data endpoints for Spot and USD-M Futures. API keys are optional for these public endpoints; if present, `BINANCE_API_KEY` is sent only as a public-data header. `BINANCE_API_SECRET` is never needed for this read-only acquisition.",
        "",
        "## Selection rule",
        "",
        "Relevant symbols are discovered from exchange metadata rather than assumed. For each research market, the selected instrument is the available TRADING symbol with the oldest successful 1H history. Spot wins only when history starts at the same time. Unavailable requested markets remain recorded in `binance_symbols.csv` and are not fabricated.",
        "",
        "## Cache and timeframes",
        "",
        "1H candles are stored under `data/binance_cache/`. Existing candles are deduplicated and only missing history before/after the cached range is requested. The final incomplete candle is excluded. 4H candles are deterministically aggregated in UTC from complete groups of four cached 1H candles; no separate 4H download is required.",
        "",
        "## Current acquisition",
        "",
        f"Discovered candidates: {len(candidates)}.",
        f"Selected markets downloaded: {len(selected)}.",
        "",
        "| Market | Symbol | Venue | 1H history |",
        "|---|---|---|---:|",
    ]
    for candidate in selected:
        row = quality_by_market.get(candidate.research_market, {})
        lines.append(f"| {candidate.research_market} | {candidate.symbol} | {candidate.market_type} | {row.get('years_of_history', 'n/a')} years |")
    if errors:
        lines.extend(["", "## Acquisition limitations", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "## Intended use",
            "",
            "Use BTC and ETH as the longest-history exploratory crypto series. Use any discovered tokenized or proxy instruments only with their explicitly reported history and venue. Do not interpret Binance proxies as identical to Alpha Futures CME contracts.",
            "",
            "Official documentation: https://developers.binance.com/en/docs/products/spot/rest-api and https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/Introduction",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cache_dir: Path = Path("data/binance_cache"), reports_dir: Path = Path("reports/binance")) -> dict[str, Any]:
    client = BinancePublicClient()
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    candidates = discover_symbols(client)
    selected = select_symbols(client, candidates)
    print(f"Discovered {len(candidates)} relevant symbol rows; selected {len(selected)} instruments", flush=True)
    quality_rows: list[dict[str, Any]] = []
    for candidate in selected:
        try:
            print(f"Downloading/caching {candidate.research_market} {candidate.symbol} ({candidate.market_type})", flush=True)
            frame_1h = update_1h_cache(client, candidate, cache_dir)
            frame_4h = build_4h(frame_1h)
            four_hour_path = cache_dir / f"{candidate.research_market.replace(' ', '_')}_{candidate.symbol}_{candidate.market_type}_4h.parquet"
            frame_4h.to_parquet(four_hour_path)
            provider = f"Binance official REST API ({candidate.market_type})"
            quality_rows.append(_quality_row(candidate, "1h", frame_1h, provider, "downloaded/cached Binance 1H OHLCV"))
            if not frame_4h.empty:
                quality_rows.append(_quality_row(candidate, "4h", frame_4h, provider, "deterministic UTC aggregation from complete 1H candles"))
            else:
                errors.append(f"{candidate.research_market}: no complete 4H buckets could be built")
            print(f"Completed {candidate.research_market}: {len(frame_1h)} 1H / {len(frame_4h)} 4H candles", flush=True)
        except (RuntimeError, OSError, ValueError) as exc:
            candidate.error = str(exc)
            errors.append(f"{candidate.research_market} {candidate.symbol}: {exc}")
    symbol_fields = [
        "research_market", "asset_class", "market_type", "symbol", "base_asset", "quote_asset", "status", "contract_type",
        "onboard_date_utc", "earliest_1h_candle_utc", "history_probe_status", "selected_for_download", "selection_reason", "error", "provider",
    ]
    quality_fields = [
        "research_market", "symbol", "market_type", "timeframe", "first_candle_utc", "last_candle_utc", "months_of_history", "years_of_history",
        "timeframe_available", "candle_count", "expected_candle_count", "missing_candles", "missing_ratio", "provider", "source_note",
    ]
    _write_csv(reports_dir / "binance_symbols.csv", [_symbol_dict(row) for row in candidates], symbol_fields)
    _write_csv(reports_dir / "binance_data_quality.csv", quality_rows, quality_fields)
    history_rows = []
    for row in quality_rows:
        history_rows.append({
            "research_market": row["research_market"], "symbol": row["symbol"], "market_type": row["market_type"], "timeframe": row["timeframe"],
            "first_candle_utc": row["first_candle_utc"], "last_candle_utc": row["last_candle_utc"], "months_of_history": row["months_of_history"],
            "years_of_history": row["years_of_history"], "candle_count": row["candle_count"], "missing_candles": row["missing_candles"],
            "provider": row["provider"],
        })
    _write_csv(reports_dir / "binance_history.csv", history_rows, [
        "research_market", "symbol", "market_type", "timeframe", "first_candle_utc", "last_candle_utc", "months_of_history", "years_of_history", "candle_count", "missing_candles", "provider",
    ])
    write_setup(reports_dir / "setup.md", candidates, selected, quality_rows, errors)
    return {"candidates": candidates, "selected": selected, "quality_rows": quality_rows, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and cache Binance exploratory research candles")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/binance_cache"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/binance"))
    args = parser.parse_args()
    result = run(args.cache_dir, args.reports_dir)
    print(f"Discovered {len(result['candidates'])} relevant Binance candidates")
    print(f"Downloaded/cached {len(result['selected'])} selected markets")
    print(f"Generated {len(result['quality_rows'])} quality rows")
    for error in result["errors"]:
        print(f"WARNING: {error}")


if __name__ == "__main__":
    main()
