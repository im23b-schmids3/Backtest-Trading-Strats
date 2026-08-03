from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel

STRATEGY_ID = "ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY"
ADAPTER_ID = "imbalance-vwap-ride-btc-long-only-v3-1"
EVIDENCE_LABEL = "POST_HOC_V3_LONG_ONLY"
PERIOD_LABEL = "NEW_TEMPORAL_V3_DEVELOPMENT_PERIOD"
SELECTION_METHOD = "PRE_REGISTERED_LONG_ONLY_OAT"
SPEC_VERSION = "imbalance-vwap-ride-btc-long-only-v3-1"
COST_MODEL_VERSION = "binance-btcusdt-verified-research-costs-v3"

AUTHORIZED_MONTHS = (
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-01",
)
EARLY_SUBPERIOD_MONTHS = ("2024-08", "2024-09", "2024-10")
LATE_SUBPERIOD_MONTHS = ("2024-11", "2024-12", "2025-01")


class ImbalanceVWAPRideV3Config(StrictModel):
    variant_id: str = "baseline"
    bin_size_usd: Decimal = Field(default=Decimal("50"), gt=0)
    min_bin_volume_btc: Decimal = Field(default=Decimal("35"), gt=0)
    vwap_slope_bars: int = Field(default=24, ge=1)
    min_imbalance_ratio: Decimal = Decimal("3")
    stacked_bins: int = 3
    move_away_bars: int = 1
    zone_expiry_bars: int = 36
    stop_buffer_bins: int = 2
    target_r_multiple: Decimal = Decimal("2.5")
    maximum_active_zones: int = 3
    maximum_trades_per_utc_day: int = 1
    maximum_trades_per_zone: int = 1
    entry_execution: str = "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST"
    direction: str = "LONG_ONLY"
    symbol: str = "BTCUSDT"
    quantity_btc: Decimal = Decimal("0.001")
    price_tick: Decimal = Decimal("0.1")
    quantity_step: Decimal = Decimal("0.001")
    minimum_quantity: Decimal = Decimal("0.001")
    taker_fee_rate: Decimal = Decimal("0.0005")
    market_slippage_ticks: int = 1
    stop_slippage_ticks: int = 2
    same_bar_policy: str = "STOP_FIRST"

    @model_validator(mode="after")
    def sealed_invariants(self) -> "ImbalanceVWAPRideV3Config":
        expected = {
            "min_imbalance_ratio": Decimal("3"),
            "stacked_bins": 3,
            "move_away_bars": 1,
            "zone_expiry_bars": 36,
            "stop_buffer_bins": 2,
            "target_r_multiple": Decimal("2.5"),
            "maximum_active_zones": 3,
            "maximum_trades_per_utc_day": 1,
            "maximum_trades_per_zone": 1,
            "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
            "direction": "LONG_ONLY",
            "same_bar_policy": "STOP_FIRST",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"sealed V3 invariant {name} must equal {value}")
        return self

    def parameter_payload(self) -> dict[str, Any]:
        names = (
            "bin_size_usd",
            "min_bin_volume_btc",
            "vwap_slope_bars",
            "min_imbalance_ratio",
            "stacked_bins",
            "move_away_bars",
            "zone_expiry_bars",
            "stop_buffer_bins",
            "target_r_multiple",
            "maximum_active_zones",
            "maximum_trades_per_utc_day",
            "maximum_trades_per_zone",
            "entry_execution",
            "direction",
        )
        payload = self.model_dump(mode="json")
        return {name: payload[name] for name in names}


BASELINE = ImbalanceVWAPRideV3Config()

PARAMETER_REGISTRY: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("bin_size_usd", (Decimal("30"), Decimal("50"), Decimal("75"))),
    ("min_bin_volume_btc", (Decimal("20"), Decimal("35"), Decimal("50"))),
    ("vwap_slope_bars", (18, 24, 36)),
)


def _value_id(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def preregistered_variants() -> list[ImbalanceVWAPRideV3Config]:
    """Return exactly the seven sealed V3 long-only OAT configurations."""

    variants = [BASELINE]
    baseline = BASELINE.model_dump(mode="python")
    for name, values in PARAMETER_REGISTRY:
        for value in values:
            if value == baseline[name]:
                continue
            variants.append(
                BASELINE.model_copy(
                    update={"variant_id": f"{name}={_value_id(value)}", name: value}
                )
            )
    identities = [tuple(sorted(item.parameter_payload().items())) for item in variants]
    if len(variants) != 7 or len(set(identities)) != 7:
        raise AssertionError("V3 registry must contain exactly seven unique configurations")
    for variant in variants[1:]:
        changed = sum(
            variant.parameter_payload()[name] != BASELINE.parameter_payload()[name]
            for name, _ in PARAMETER_REGISTRY
        )
        if changed != 1:
            raise AssertionError(f"V3 variant {variant.variant_id} is not one-factor-at-a-time")
    return variants


def baseline_distance(config: ImbalanceVWAPRideV3Config) -> int:
    return sum(
        config.parameter_payload()[name] != BASELINE.parameter_payload()[name]
        for name, _ in PARAMETER_REGISTRY
    )


def _finite(metrics: dict[str, Any], name: str) -> bool:
    try:
        return math.isfinite(float(metrics[name]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _greater(metrics: dict[str, Any], name: str, threshold: float) -> bool:
    try:
        return float(metrics[name]) > threshold
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def sample_classification(metrics: dict[str, Any]) -> str:
    trades = int(metrics.get("executed_trades", 0))
    if trades >= 48:
        return "PROMOTION_SAMPLE_ELIGIBLE"
    if trades >= 36:
        return "INFORMATIVE_36_TO_47_NOT_PROMOTABLE"
    return "SAMPLE_INSUFFICIENT_BELOW_36"


def promotion_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    months = metrics.get("months", {}) if isinstance(metrics.get("months"), dict) else {}
    active_months = sum(int(item.get("executed_trades", 0)) > 0 for item in months.values())
    months_with_four = sum(int(item.get("executed_trades", 0)) >= 4 for item in months.values())
    gross_net_names = (
        "gross_pnl",
        "net_pnl",
        "gross_profit_factor",
        "net_profit_factor",
        "average_gross_r",
        "average_net_r",
    )
    gross_net_reported = all(name in metrics and metrics[name] is not None for name in gross_net_names)
    checks = {
        "minimum_48_trades": int(metrics.get("executed_trades", 0)) >= 48,
        "five_active_months": active_months >= 5,
        "four_months_with_at_least_four_trades": months_with_four >= 4,
        "net_profit_factor_above_1_10": _greater(metrics, "net_profit_factor", 1.10),
        "average_net_r_positive": _greater(metrics, "average_net_r", 0),
        "net_pnl_positive": Decimal(str(metrics.get("net_pnl", "0"))) > 0,
        "finite_drawdown": _finite(metrics, "maximum_drawdown"),
        "funnel_reconciled": bool(metrics.get("funnel_reconciliation", {}).get("reconciles")),
        "long_only_reconciled": bool(metrics.get("long_only_reconciliation", {}).get("reconciles")),
        "monthly_concentration_at_most_60_percent": float(
            metrics.get("maximum_positive_month_contribution", 1)
        )
        <= 0.60,
        "best_five_contribution_below_70_percent": float(
            metrics.get("best_five_positive_pnl_contribution", 1)
        )
        < 0.70,
        "gross_and_net_reporting_present": gross_net_reported,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_classification": sample_classification(metrics),
        "active_month_count": active_months,
        "months_with_at_least_four_trades": months_with_four,
    }


def freeze_passing_candidates(
    candidates: list[tuple[ImbalanceVWAPRideV3Config, dict[str, Any]]],
) -> list[tuple[ImbalanceVWAPRideV3Config, dict[str, Any]]]:
    passing = [item for item in candidates if promotion_gate(item[1])["passed"]]

    def key(item: tuple[ImbalanceVWAPRideV3Config, dict[str, Any]]) -> tuple[Any, ...]:
        config, metrics = item
        gate = promotion_gate(metrics)
        return (
            -float(metrics["net_profit_factor"]),
            float(metrics["maximum_drawdown"]),
            -int(gate["active_month_count"]),
            -int(metrics["executed_trades"]),
            baseline_distance(config),
            config.variant_id,
        )

    return sorted(passing, key=key)[:2]
