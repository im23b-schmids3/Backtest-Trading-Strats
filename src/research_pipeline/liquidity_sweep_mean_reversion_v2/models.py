from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import model_validator

from ..imbalance_vwap_ride.artifacts import sha256_value
from ..schemas.strategy_spec import StrictModel

STRATEGY_ID = "LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION"
EVIDENCE = "EXPLORATORY_IN_SAMPLE_SELECTION"
PHASE_A_MONTHS = tuple([f"2023-{month:02d}" for month in range(1, 13)] + ["2024-01"])
CANDIDATE_REGISTRY = (("LSMR-V2-2P0R", Decimal("2.0")), ("LSMR-V2-2P5R", Decimal("2.5")), ("LSMR-V2-3P0R", Decimal("3.0")))
TERMINAL_DISPOSITIONS = frozenset({"TRADE_EXECUTED", "RECLAIM_WINDOW_EXPIRED", "VOLUME_REJECTED", "CANDLE_REJECTED", "REGIME_REJECTED", "VWAP_PROXIMITY_REJECTED", "SESSION_CONTEXT_UNAVAILABLE", "STOP_DISTANCE_REJECTED", "DUPLICATE_REFERENCE_SUPPRESSED", "COOLDOWN_BLOCKED", "NO_EXECUTABLE_ENTRY", "COMPLIANCE_BLOCKED", "SESSION_ENDED"})


class LSMRV2Config(StrictModel):
    candidate_id: str = CANDIDATE_REGISTRY[0][0]
    target_r_multiple: Decimal = Decimal("2.0")
    symbol: str = "BTCUSDT"
    timeframe_minutes: int = 5
    reference_bars: int = 24
    reclaim_window_bars: int = 2
    body_fraction: Decimal = Decimal("0.65")
    volume_lookback_bars: int = 20
    volume_multiple: Decimal = Decimal("1.50")
    vwap_slope_bars: int = 24
    maximum_vwap_slope_fraction: Decimal = Decimal("0.0015")
    maximum_vwap_proximity_fraction: Decimal = Decimal("0.0075")
    penetration_fraction: Decimal = Decimal("0.0025")
    price_tick: Decimal = Decimal("0.1")
    stop_buffer_ticks: int = 2
    minimum_stop_fraction: Decimal = Decimal("0.002")
    maximum_stop_fraction: Decimal = Decimal("0.0125")
    cooldown_bars: int = 12
    time_stop_bars: int = 36
    utc_force_flat_hour: int = 23
    utc_force_flat_minute: int = 55
    taker_fee_rate: Decimal = Decimal("0.0005")
    market_slippage_ticks: int = 1
    stop_slippage_ticks: int = 2
    same_bar_policy: str = "STOP_FIRST"

    @model_validator(mode="after")
    def sealed(self):
        if dict(CANDIDATE_REGISTRY).get(self.candidate_id) != self.target_r_multiple:
            raise ValueError("sealed LSMR V2 candidate parameters do not match candidate_id")
        sealed = {"symbol": "BTCUSDT", "timeframe_minutes": 5, "reference_bars": 24, "reclaim_window_bars": 2, "body_fraction": Decimal("0.65"), "volume_lookback_bars": 20, "volume_multiple": Decimal("1.50"), "vwap_slope_bars": 24, "maximum_vwap_slope_fraction": Decimal("0.0015"), "maximum_vwap_proximity_fraction": Decimal("0.0075"), "penetration_fraction": Decimal("0.0025"), "price_tick": Decimal("0.1"), "stop_buffer_ticks": 2, "minimum_stop_fraction": Decimal("0.002"), "maximum_stop_fraction": Decimal("0.0125"), "cooldown_bars": 12, "time_stop_bars": 36, "utc_force_flat_hour": 23, "utc_force_flat_minute": 55, "taker_fee_rate": Decimal("0.0005"), "market_slippage_ticks": 1, "stop_slippage_ticks": 2, "same_bar_policy": "STOP_FIRST"}
        if any(getattr(self, name) != value for name, value in sealed.items()):
            raise ValueError("sealed LSMR V2 invariant violation")
        return self

    def parameter_payload(self) -> dict[str, Any]: return self.model_dump(mode="json")
    def frozen_payload(self) -> dict[str, Any]: return {"candidate_id": self.candidate_id, "parameters": self.parameter_payload(), "phase_a_months": list(PHASE_A_MONTHS)}


def preregistered_candidates() -> list[LSMRV2Config]: return [LSMRV2Config(candidate_id=name, target_r_multiple=target) for name, target in CANDIDATE_REGISTRY]
def candidate_registry_payload() -> list[dict[str, str]]: return [{"candidate_id": name, "target_r_multiple": str(target)} for name, target in CANDIDATE_REGISTRY]
def candidate_registry_hash() -> str: return sha256_value(candidate_registry_payload())
def candidate_configuration_hash(config: LSMRV2Config) -> str: return sha256_value(config.frozen_payload())
