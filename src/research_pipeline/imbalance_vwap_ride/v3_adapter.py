from __future__ import annotations

from typing import Any

from .v3_models import ADAPTER_ID, STRATEGY_ID, ImbalanceVWAPRideV3Config
from .v3_strategy import run_imbalance_vwap_ride_v3


class ImbalanceVWAPRideV3Adapter:
    """Sealed local-only adapter for the V3 long-only development study."""

    adapter_id = ADAPTER_ID
    strategy_id = STRATEGY_ID
    live_orders_supported = False
    raw_trade_transmission_supported = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "strategy_id": self.strategy_id,
            "direction": "LONG_ONLY",
            "local_parquet_processing": True,
            "bounded_streaming": True,
            "authorized_archive_download_scope": "BTCUSDT_2024_08_TO_2025_01_ONLY",
            "live_orders": False,
            "external_raw_trade_transmission": False,
            "alpha_execution_from_development_data": False,
        }

    def run_loaded(
        self,
        *,
        bars: list[dict[str, Any]],
        footprints: list[dict[str, Any]] | dict[Any, list[dict[str, Any]]],
        config: ImbalanceVWAPRideV3Config,
    ) -> dict[str, Any]:
        return run_imbalance_vwap_ride_v3(bars, footprints, config)
