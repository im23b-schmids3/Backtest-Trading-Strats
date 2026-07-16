from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import validate_ohlcv


class Cache:
    def __init__(self, root: str | Path = "data/raw") -> None:
        self.root = Path(root)

    def path(self, asset: str, timeframe: str) -> Path:
        return self.root / f"{asset.upper()}_{timeframe}.parquet"

    def read(self, asset: str, timeframe: str, allow_expected_gaps: bool = False) -> pd.DataFrame:
        path = self.path(asset, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"cached data not found: {path}; run the download command")
        try:
            return validate_ohlcv(pd.read_parquet(path), timeframe, allow_expected_gaps)
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow; install project dependencies") from exc

    def write(self, asset: str, timeframe: str, frame: pd.DataFrame, allow_expected_gaps: bool = False) -> Path:
        data = validate_ohlcv(frame, timeframe, allow_expected_gaps)
        target = self.path(asset, timeframe)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(target)
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow; install project dependencies") from exc
        return target
