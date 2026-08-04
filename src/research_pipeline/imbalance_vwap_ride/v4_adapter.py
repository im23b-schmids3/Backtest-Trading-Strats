from __future__ import annotations

from typing import Any

from .v4_models import (
    ADAPTER_ID,
    PHASE_A_MONTHS,
    PHASE_B_MONTHS,
    STRATEGY_ID,
    ImbalanceVWAPRideV4Config,
)
from .v4_strategy import run_imbalance_vwap_ride_v4


class ImbalanceVWAPRideV4Adapter:
    """Local-only V4 candidate-selection and locked-test adapter."""

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
            "phase_a_scope": list(PHASE_A_MONTHS),
            "phase_b_scope": list(PHASE_B_MONTHS),
            "official_archive_origin": "https://data.binance.vision",
            "live_orders": False,
            "external_raw_trade_transmission": False,
            "secrets_required": False,
            "phase_b_isolated_by_frozen_hash": True,
            "confirmation_evidence": False,
        }

    @staticmethod
    def validate_phase_scope(bars: list[dict[str, Any]], phase: str) -> None:
        expected = PHASE_A_MONTHS if phase == "PHASE_A" else PHASE_B_MONTHS if phase == "PHASE_B" else None
        if expected is None:
            raise ValueError(f"unsupported V4 phase: {phase}")
        actual = tuple(sorted({str(item.get("month") or item["bar_start_utc"])[:7] for item in bars}))
        if actual != expected:
            raise ValueError(f"{phase} bars must cover exactly {expected}; got {actual}")

    def run_loaded(
        self,
        *,
        bars: list[dict[str, Any]],
        footprints: list[dict[str, Any]] | dict[Any, list[dict[str, Any]]],
        config: ImbalanceVWAPRideV4Config,
        phase: str,
        require_exact_scope: bool = True,
    ) -> dict[str, Any]:
        if require_exact_scope:
            self.validate_phase_scope(bars, phase)
        return run_imbalance_vwap_ride_v4(bars, footprints, config, phase=phase)
