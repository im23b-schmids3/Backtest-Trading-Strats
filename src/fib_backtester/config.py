from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal



@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    source: Literal["binance", "yfinance"]
    fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "baseline"
    seed: int = 42
    start: str = "2025-01-01T00:00:00Z"
    initial_cash: float = 10_000.0
    assets: list[str] = field(default_factory=lambda: ["BTC"])
    timeframes: list[str] = field(default_factory=lambda: ["1h"])
    swing_n: int = 3
    min_pivot_distance: int = 10
    max_anchor_age_days: dict[str, float] = field(default_factory=lambda: {"1h": 30.0, "4h": 60.0, "1d": 180.0})
    entry_max_age_bars: int | None = None
    reentry: bool = False
    execution_policy: Literal["conservative", "optimistic", "lower_timeframe_replay"] = "conservative"
    max_positions: int = 5
    max_total_risk_fraction: float = 0.10
    leverage: float = 1.0
    asset_configs: dict[str, AssetConfig] = field(default_factory=dict)

    def validate(self) -> None:
        if not 2 <= self.swing_n <= 10:
            raise ValueError("swing_n must be between 2 and 10")
        if self.min_pivot_distance < 1 or self.initial_cash <= 0:
            raise ValueError("min_pivot_distance and initial_cash must be positive")
        for timeframe, age in self.max_anchor_age_days.items():
            if age <= 0:
                raise ValueError(f"max_anchor_age_days[{timeframe}] must be positive")
        for asset in self.assets:
            if asset not in self.asset_configs:
                raise ValueError(f"missing asset config for {asset}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration; install project dependencies") from exc
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw["asset_configs"] = {key: AssetConfig(**value) for key, value in raw.get("asset_configs", {}).items()}
    config = RunConfig(**raw)
    config.validate()
    return config
