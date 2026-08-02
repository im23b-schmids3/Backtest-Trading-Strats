from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .errors import DataAvailabilityError
from .models import DataAvailability, DataClassification


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataAvailabilityGate:
    """Resolve only explicitly declared local data; never substitutes silently."""

    def __init__(self, repository_root: str | Path):
        self.root = Path(repository_root).resolve()
        self.search_roots = [
            self.root / "data" / "v11_intraday_raw",
            self.root / "data" / "v11_5_proxy_raw",
            self.root / "data" / "raw",
        ]

    def check(self, market: str, timeframe: str, *, source_symbol: str | None = None,
              allow_proxy: bool = False) -> DataAvailability:
        symbol = source_symbol or market
        native_candidates = [root / f"{symbol.upper()}_{timeframe}.parquet" for root in self.search_roots]
        path = next((item for item in native_candidates if item.is_file()), None)
        if path is None:
            return DataAvailability(market=market, timeframe=timeframe, classification=DataClassification.UNAVAILABLE,
                                    source_symbol=symbol, warnings=[f"no local parquet data for {symbol}_{timeframe}"])
        frame = pd.read_parquet(path)
        if frame.empty or len(frame) < 20:
            return DataAvailability(market=market, timeframe=timeframe, classification=DataClassification.PARTIAL_HISTORY,
                                    provider="local_parquet", source_symbol=symbol, path=str(path.resolve()),
                                    dataset_hash=file_hash(path), rows=len(frame), warnings=["fewer than 20 rows"])
        classification = DataClassification.AVAILABLE_NATIVE if symbol.upper() == market.upper() else DataClassification.AVAILABLE_PROXY
        if classification == DataClassification.AVAILABLE_PROXY and not allow_proxy:
            return DataAvailability(market=market, timeframe=timeframe, classification=DataClassification.MANUAL_MAPPING_REQUIRED,
                                    provider="local_parquet", source_symbol=symbol, path=str(path.resolve()), dataset_hash=file_hash(path),
                                    rows=len(frame), warnings=["source symbol differs from requested market and proxy use is not approved"])
        index = pd.to_datetime(frame.index, utc=True)
        return DataAvailability(market=market, timeframe=timeframe, classification=classification, provider="local_parquet",
                                source_symbol=symbol, path=str(path.resolve()), dataset_hash=file_hash(path),
                                start_timestamp=index.min().to_pydatetime(), end_timestamp=index.max().to_pydatetime(), rows=len(frame),
                                declared_substitution=f"{market}->{symbol}" if classification == DataClassification.AVAILABLE_PROXY else None)

    def require(self, markets: list[str], timeframes: list[str], *, source_symbols: dict[str, str] | None = None,
                allow_proxy: bool = False) -> list[DataAvailability]:
        results = [self.check(market, timeframe, source_symbol=(source_symbols or {}).get(market), allow_proxy=allow_proxy)
                   for market in markets for timeframe in timeframes]
        invalid = [item for item in results if item.classification in {DataClassification.UNAVAILABLE, DataClassification.MANUAL_MAPPING_REQUIRED, DataClassification.PARTIAL_HISTORY}]
        if invalid:
            detail = "; ".join(f"{item.market}/{item.timeframe}: {item.classification.value}" for item in invalid)
            raise DataAvailabilityError(f"{DataAvailabilityError.code}: {detail}")
        return results


def chronological_split(availability: DataAvailability, *, train_ratio: float = .6, validation_ratio: float = .2) -> tuple[datetime, datetime, datetime, datetime, datetime, datetime]:
    if not availability.start_timestamp or not availability.end_timestamp:
        raise DataAvailabilityError("data availability has no chronological bounds")
    start, end = availability.start_timestamp, availability.end_timestamp
    total = (end - start).total_seconds()
    train_end = datetime.fromtimestamp(start.timestamp() + total * train_ratio, tz=timezone.utc)
    validation_end = datetime.fromtimestamp(start.timestamp() + total * (train_ratio + validation_ratio), tz=timezone.utc)
    return start, train_end, train_end, validation_end, validation_end, end
