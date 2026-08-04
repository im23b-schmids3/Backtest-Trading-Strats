"""Standalone sealed ImbalanceVWAPRide BTC exploratory research study."""

from .adapter import ImbalanceVWAPRideAdapter
from .models import (
    ADAPTER_ID,
    DATASET_HASH,
    EVIDENCE_LABEL,
    STRATEGY_ID,
    ImbalanceVWAPRideConfig,
    preregistered_variants,
)
from .runner import run_sealed_study
from .strategy import run_imbalance_vwap_ride
from .v2_adapter import ImbalanceVWAPRideV2Adapter
from .v2_models import ImbalanceVWAPRideV2Config
from .v2_runner import run_sealed_v2_study
from .v2_strategy import run_imbalance_vwap_ride_v2
from .v3_adapter import ImbalanceVWAPRideV3Adapter
from .v3_models import ImbalanceVWAPRideV3Config
from .v3_runner import run_sealed_v3_study
from .v3_strategy import run_imbalance_vwap_ride_v3
from .v4_adapter import ImbalanceVWAPRideV4Adapter
from .v4_models import ImbalanceVWAPRideV4Config, preregistered_candidates
from .v4_runner import run_sealed_v4_study
from .v4_strategy import run_imbalance_vwap_ride_v4

__all__ = [
    "ADAPTER_ID",
    "DATASET_HASH",
    "EVIDENCE_LABEL",
    "STRATEGY_ID",
    "ImbalanceVWAPRideAdapter",
    "ImbalanceVWAPRideConfig",
    "preregistered_variants",
    "run_imbalance_vwap_ride",
    "run_sealed_study",
    "ImbalanceVWAPRideV2Adapter",
    "ImbalanceVWAPRideV2Config",
    "run_imbalance_vwap_ride_v2",
    "run_sealed_v2_study",
    "ImbalanceVWAPRideV3Adapter",
    "ImbalanceVWAPRideV3Config",
    "run_imbalance_vwap_ride_v3",
    "run_sealed_v3_study",
    "ImbalanceVWAPRideV4Adapter",
    "ImbalanceVWAPRideV4Config",
    "preregistered_candidates",
    "run_imbalance_vwap_ride_v4",
    "run_sealed_v4_study",
]
