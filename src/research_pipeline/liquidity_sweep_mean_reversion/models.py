from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import model_validator

from ..imbalance_vwap_ride.artifacts import sha256_value
from ..schemas.strategy_spec import StrictModel

STRATEGY_ID = "LiquiditySweepMeanReversion.BTC_LONG_SHORT_V1_SPECIFICATION"
EVIDENCE = "EXPLORATORY_IN_SAMPLE_SELECTION"
PHASE_A_MONTHS = tuple([f"2023-{month:02d}" for month in range(1, 13)] + ["2024-01"])
PHASE_B_MONTHS = tuple(f"2024-{month:02d}" for month in range(2, 8))
CANDIDATE_REGISTRY = (("LSMR-V1-1P5R", Decimal("1.5")), ("LSMR-V1-2P0R", Decimal("2.0")), ("LSMR-V1-2P5R", Decimal("2.5")))
TERMINAL_DISPOSITIONS = frozenset({"EXECUTED", "RECLAIM_WINDOW_EXPIRED", "REGIME_REJECTED", "STOP_DISTANCE_REJECTED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "SESSION_ENDED"})

class LSMRConfig(StrictModel):
    candidate_id: str = CANDIDATE_REGISTRY[0][0]
    target_r_multiple: Decimal = Decimal("1.5")
    symbol: str = "BTCUSDT"
    timeframe_minutes: int = 5
    reference_bars: int = 12
    reclaim_window_bars: int = 3
    body_fraction: Decimal = Decimal("0.50")
    volume_multiple: Decimal = Decimal("1.20")
    vwap_slope_bars: int = 24
    maximum_vwap_slope_fraction: Decimal = Decimal("0.003")
    penetration_fraction: Decimal = Decimal("0.0015")
    price_tick: Decimal = Decimal("0.1")
    stop_buffer_ticks: int = 2
    minimum_stop_fraction: Decimal = Decimal("0.002")
    maximum_stop_fraction: Decimal = Decimal("0.0125")
    time_stop_bars: int = 36
    utc_force_flat_hour: int = 23
    utc_force_flat_minute: int = 55
    taker_fee_rate: Decimal = Decimal("0.0005")
    market_slippage_ticks: int = 1
    stop_slippage_ticks: int = 2
    same_bar_policy: str = "STOP_FIRST"

    @model_validator(mode="after")
    def sealed(self):
        target = dict(CANDIDATE_REGISTRY).get(self.candidate_id)
        if target != self.target_r_multiple:
            raise ValueError("sealed LSMR candidate parameters do not match candidate_id")
        sealed = {
            "symbol": "BTCUSDT", "timeframe_minutes": 5, "reference_bars": 12,
            "reclaim_window_bars": 3, "body_fraction": Decimal("0.50"),
            "volume_multiple": Decimal("1.20"), "vwap_slope_bars": 24,
            "maximum_vwap_slope_fraction": Decimal("0.003"),
            "penetration_fraction": Decimal("0.0015"), "price_tick": Decimal("0.1"),
            "stop_buffer_ticks": 2, "minimum_stop_fraction": Decimal("0.002"),
            "maximum_stop_fraction": Decimal("0.0125"), "time_stop_bars": 36,
            "utc_force_flat_hour": 23, "utc_force_flat_minute": 55,
            "taker_fee_rate": Decimal("0.0005"), "market_slippage_ticks": 1,
            "stop_slippage_ticks": 2, "same_bar_policy": "STOP_FIRST",
        }
        if any(getattr(self, name) != value for name, value in sealed.items()):
            raise ValueError("sealed LSMR invariant violation")
        return self

    def parameter_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def frozen_payload(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "parameters": self.parameter_payload(), "phase_a_months": list(PHASE_A_MONTHS), "phase_b_months": list(PHASE_B_MONTHS)}

def preregistered_candidates() -> list[LSMRConfig]:
    return [LSMRConfig(candidate_id=candidate_id, target_r_multiple=target) for candidate_id, target in CANDIDATE_REGISTRY]

def candidate_registry_payload() -> list[dict[str, str]]:
    return [{"candidate_id": candidate_id, "target_r_multiple": str(target)} for candidate_id, target in CANDIDATE_REGISTRY]

def candidate_registry_hash() -> str:
    return sha256_value(candidate_registry_payload())

def candidate_configuration_hash(config: LSMRConfig) -> str:
    return sha256_value(config.frozen_payload())
