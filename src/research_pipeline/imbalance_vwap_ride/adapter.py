from __future__ import annotations

from pathlib import Path
from typing import Any

from .footprint import build_footprint_dataset, load_footprint_dataset, validate_footprint_dataset, validate_source_manifest
from .models import ADAPTER_ID, DATASET_HASH, STRATEGY_ID, ImbalanceVWAPRideConfig
from .strategy import run_imbalance_vwap_ride


class ImbalanceVWAPRideAdapter:
    """Standalone local-only adapter for the sealed BTC footprint study.

    It intentionally does not inherit from, import, resolve, or delegate to any
    ValueAreaTrap or ValueAreaAcceptance adapter.
    """

    adapter_id = ADAPTER_ID
    strategy_id = STRATEGY_ID
    dataset_hash = DATASET_HASH
    live_orders_supported = False
    network_supported = False
    raw_trade_transmission_supported = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "strategy_id": self.strategy_id,
            "local_parquet_only": True,
            "bounded_streaming": True,
            "exact_buyer_is_maker_aggressor": True,
            "five_minute_utc_bars": True,
            "live_orders": False,
            "downloads": False,
            "renormalization": False,
            "external_raw_trade_transmission": False,
        }

    def validate_source(self, manifest_path: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
        return validate_source_manifest(manifest_path, require_pinned=True, verify_parquet_hashes=verify_hashes)

    def materialize_footprint(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        *,
        batch_size: int = 1_000_000,
        verify_source_hashes: bool = True,
    ) -> dict[str, Any]:
        return build_footprint_dataset(
            manifest_path,
            cache_root,
            batch_size=batch_size,
            require_pinned=True,
            verify_source_hashes=verify_source_hashes,
        )

    def validate_footprint(self, footprint_root: str | Path) -> dict[str, Any]:
        return validate_footprint_dataset(footprint_root)

    def run_split(
        self,
        footprint_manifest: dict[str, Any] | str | Path,
        *,
        months: set[str],
        config: ImbalanceVWAPRideConfig,
    ) -> dict[str, Any]:
        footprints, bars = load_footprint_dataset(footprint_manifest, months=months)
        return run_imbalance_vwap_ride(bars, footprints, config)

    def run_loaded(
        self,
        *,
        bars: list[dict[str, Any]],
        footprints: list[dict[str, Any]] | dict[Any, list[dict[str, Any]]],
        config: ImbalanceVWAPRideConfig,
    ) -> dict[str, Any]:
        return run_imbalance_vwap_ride(bars, footprints, config)
