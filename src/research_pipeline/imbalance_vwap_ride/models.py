from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel

STRATEGY_ID = "ImbalanceVWAPRide.BTC_EXPLORATORY"
ADAPTER_ID = "imbalance-vwap-ride-btc-1"
DATASET_HASH = "c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc"
EVIDENCE_LABEL = "INTERNAL_LOCKED_TEST_NOT_PRISTINE_HOLDOUT"
SOURCE_MANIFEST_HASH = "f3abab68e0364d9e12e0e1ace8eb35957f8423ba5a2a7e0ac584321607c30d88"
SPEC_VERSION = "imbalance-vwap-ride-btc-exploratory-1"
FOOTPRINT_VERSION = "imbalance-vwap-footprint-1"
COST_MODEL_VERSION = "binance-btcusdt-verified-research-costs-1"


class ImbalanceVWAPRideConfig(StrictModel):
    variant_id: str = "baseline"
    bin_size_usd: Decimal = Field(default=Decimal("10.0"), gt=0)
    min_imbalance_ratio: Decimal = Field(default=Decimal("3.0"), gt=1)
    stacked_bins: int = Field(default=3, ge=1)
    min_bin_volume_btc: Decimal = Field(default=Decimal("5.0"), gt=0)
    vwap_slope_bars: int = Field(default=10, ge=1)
    move_away_bars: int = Field(default=2, ge=1)
    zone_expiry_bars: int = Field(default=20, ge=1)
    stop_buffer_bins: int = Field(default=2, ge=0)
    target_r_multiple: Decimal = Field(default=Decimal("2.0"), gt=0)
    maximum_active_zones_per_direction: int = 3
    maximum_trades_per_utc_day: int = 1
    entry_execution: str = "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST"
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
    def sealed_invariants(self) -> "ImbalanceVWAPRideConfig":
        if self.maximum_active_zones_per_direction != 3:
            raise ValueError("sealed zone cap must equal three")
        if self.maximum_trades_per_utc_day != 1:
            raise ValueError("sealed daily trade cap must equal one")
        if self.entry_execution != "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST":
            raise ValueError("same-bar entry is forbidden")
        if self.same_bar_policy != "STOP_FIRST":
            raise ValueError("sealed same-bar policy is STOP_FIRST")
        return self

    def parameter_payload(self) -> dict[str, Any]:
        names = (
            "bin_size_usd",
            "min_imbalance_ratio",
            "stacked_bins",
            "min_bin_volume_btc",
            "vwap_slope_bars",
            "move_away_bars",
            "zone_expiry_bars",
            "stop_buffer_bins",
            "target_r_multiple",
            "maximum_active_zones_per_direction",
            "maximum_trades_per_utc_day",
            "entry_execution",
        )
        payload = self.model_dump(mode="json")
        return {name: payload[name] for name in names}


BASELINE = ImbalanceVWAPRideConfig()

PARAMETER_REGISTRY: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("bin_size_usd", (Decimal("10"), Decimal("20"))),
    ("min_imbalance_ratio", (Decimal("2.5"), Decimal("3"), Decimal("4"))),
    ("stacked_bins", (2, 3, 4)),
    ("min_bin_volume_btc", (Decimal("2.5"), Decimal("5"), Decimal("10"))),
    ("vwap_slope_bars", (6, 10, 20)),
    ("move_away_bars", (1, 2)),
    ("zone_expiry_bars", (12, 20, 36)),
    ("stop_buffer_bins", (1, 2, 3)),
    ("target_r_multiple", (Decimal("1.5"), Decimal("2"), Decimal("2.5"))),
)


def _value_id(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def preregistered_variants() -> list[ImbalanceVWAPRideConfig]:
    """Return baseline plus the exact unique one-factor registry (17 runs)."""

    variants = [BASELINE]
    baseline_payload = BASELINE.model_dump(mode="python")
    for name, values in PARAMETER_REGISTRY:
        baseline_value = baseline_payload[name]
        for value in values:
            if value == baseline_value:
                continue
            variants.append(BASELINE.model_copy(update={"variant_id": f"{name}={_value_id(value)}", name: value}))
    identities = [tuple(sorted(item.parameter_payload().items())) for item in variants]
    if len(variants) != 17 or len(set(identities)) != len(identities):
        raise AssertionError("pre-registered one-factor registry is not exact and unique")
    for variant in variants[1:]:
        changed = sum(
            variant.parameter_payload()[name] != BASELINE.parameter_payload()[name]
            for name, _ in PARAMETER_REGISTRY
        )
        if changed != 1:
            raise AssertionError(f"variant {variant.variant_id} is not one-factor")
    return variants


def _finite_metric(metrics: dict[str, Any], name: str) -> bool:
    try:
        return math.isfinite(float(metrics[name]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def development_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_trades": int(metrics.get("executed_trades", 0)) >= 40,
        "profit_factor": _finite_metric(metrics, "profit_factor") and float(metrics["profit_factor"]) > 1.05,
        "average_net_r": _finite_metric(metrics, "average_net_r") and float(metrics["average_net_r"]) > 0,
        "monthly_concentration": float(metrics.get("maximum_positive_month_contribution", 1)) <= 0.60,
        "best_five_contribution": float(metrics.get("best_five_positive_pnl_contribution", 1)) < 0.70,
        "finite_drawdown": _finite_metric(metrics, "maximum_drawdown"),
        "funnel": bool(metrics.get("funnel_reconciliation", {}).get("reconciles")),
        "both_directions": int(metrics.get("long_trades", 0)) > 0 and int(metrics.get("short_trades", 0)) > 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def validation_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_trades": int(metrics.get("executed_trades", 0)) >= 15,
        "profit_factor": _finite_metric(metrics, "profit_factor") and float(metrics["profit_factor"]) > 1,
        "average_net_r": _finite_metric(metrics, "average_net_r") and float(metrics["average_net_r"]) > 0,
        "positive_pnl": Decimal(str(metrics.get("net_pnl", "0"))) > 0,
        "funnel": bool(metrics.get("funnel_reconciliation", {}).get("reconciles")),
    }
    return {"passed": all(checks.values()), "checks": checks}


def locked_test_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_trades": int(metrics.get("executed_trades", 0)) >= 8,
        "profit_factor": _finite_metric(metrics, "profit_factor") and float(metrics["profit_factor"]) > 1,
        "average_net_r": _finite_metric(metrics, "average_net_r") and float(metrics["average_net_r"]) > 0,
        "positive_pnl": Decimal(str(metrics.get("net_pnl", "0"))) > 0,
        "funnel": bool(metrics.get("funnel_reconciliation", {}).get("reconciles")),
    }
    return {"passed": all(checks.values()), "checks": checks}


def baseline_distance(config: ImbalanceVWAPRideConfig) -> int:
    return sum(
        config.parameter_payload()[name] != BASELINE.parameter_payload()[name]
        for name, _ in PARAMETER_REGISTRY
    )


def freeze_development_candidates(
    candidates: list[tuple[ImbalanceVWAPRideConfig, dict[str, Any]]],
) -> list[tuple[ImbalanceVWAPRideConfig, dict[str, Any]]]:
    passing = [item for item in candidates if development_gate(item[1])["passed"]]
    return sorted(passing, key=lambda item: (baseline_distance(item[0]), item[0].variant_id))[:3]


def select_validation_candidate(
    candidates: list[tuple[ImbalanceVWAPRideConfig, dict[str, Any]]],
) -> tuple[ImbalanceVWAPRideConfig, dict[str, Any]] | None:
    passing = [item for item in candidates if validation_gate(item[1])["passed"]]
    if not passing:
        return None

    def key(item: tuple[ImbalanceVWAPRideConfig, dict[str, Any]]) -> tuple[Any, ...]:
        config, metrics = item
        return (
            -float(metrics["profit_factor"]),
            float(metrics["maximum_drawdown"]),
            -int(metrics["executed_trades"]),
            baseline_distance(config),
            config.variant_id,
        )

    return sorted(passing, key=key)[0]
