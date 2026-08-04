from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel
from .artifacts import sha256_value

STRATEGY_ID = "ImbalanceVWAPRide.BTC_LONG_ONLY_V4_CANDIDATE_SELECTION"
ADAPTER_ID = "imbalance-vwap-ride-btc-long-only-v4-1"
SELECTION_EVIDENCE = "POST_HOC_V4_RETROSPECTIVE_CANDIDATE_SELECTION"
LOCKED_EVIDENCE = "STRATEGY_SPECIFIC_TEMPORAL_LOCKED_TEST"
EVIDENCE_LABEL = SELECTION_EVIDENCE
LOCKED_EVIDENCE_LABEL = LOCKED_EVIDENCE
SELECTION_METHOD = "PRE_REGISTERED_ROBUSTNESS_RANKING"
SPEC_VERSION = "imbalance-vwap-ride-btc-long-only-v4-1"
COST_MODEL_VERSION = "binance-btcusdt-verified-research-costs-v3"

PHASE_A_MONTHS = tuple(f"2023-{month:02d}" for month in range(1, 13))
PHASE_B_MONTHS = tuple(f"2025-{month:02d}" for month in range(2, 8))
AUTHORIZED_MONTHS = PHASE_A_MONTHS + PHASE_B_MONTHS
AUTHORIZED_PHASE_A_MONTHS = PHASE_A_MONTHS
AUTHORIZED_PHASE_B_MONTHS = PHASE_B_MONTHS
PHASE_A_QUARTERS = (
    ("Q1", PHASE_A_MONTHS[0:3]),
    ("Q2", PHASE_A_MONTHS[3:6]),
    ("Q3", PHASE_A_MONTHS[6:9]),
    ("Q4", PHASE_A_MONTHS[9:12]),
)
PHASE_A_HALVES = (("H1", PHASE_A_MONTHS[:6]), ("H2", PHASE_A_MONTHS[6:]))
PHASE_B_SUBPERIODS = (("FEB_APR", PHASE_B_MONTHS[:3]), ("MAY_JUL", PHASE_B_MONTHS[3:]))

# The tuple form is intentional: it is a small, stable, non-Cartesian registry
# that can be hashed before any result is opened.
CANDIDATE_REGISTRY: tuple[tuple[str, int, Decimal], ...] = (
    ("V4-A-BASELINE-2P5R", 24, Decimal("2.5")),
    ("V4-B-BASELINE-3P0R", 24, Decimal("3.0")),
    ("V4-C-BASELINE-3P5R", 24, Decimal("3.5")),
    ("V4-D-SLOW-VWAP-2P5R", 36, Decimal("2.5")),
)
SEALED_CANDIDATE_REGISTRY = CANDIDATE_REGISTRY
_CANDIDATES_BY_ID = {item[0]: item[1:] for item in CANDIDATE_REGISTRY}


class ImbalanceVWAPRideV4Config(StrictModel):
    candidate_id: str = CANDIDATE_REGISTRY[0][0]
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
    def sealed_invariants(self) -> "ImbalanceVWAPRideV4Config":
        expected = {
            "bin_size_usd": Decimal("50"),
            "min_bin_volume_btc": Decimal("35"),
            "min_imbalance_ratio": Decimal("3"),
            "stacked_bins": 3,
            "move_away_bars": 1,
            "zone_expiry_bars": 36,
            "stop_buffer_bins": 2,
            "maximum_active_zones": 3,
            "maximum_trades_per_utc_day": 1,
            "maximum_trades_per_zone": 1,
            "entry_execution": "NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST",
            "direction": "LONG_ONLY",
            "symbol": "BTCUSDT",
            "quantity_btc": Decimal("0.001"),
            "price_tick": Decimal("0.1"),
            "quantity_step": Decimal("0.001"),
            "minimum_quantity": Decimal("0.001"),
            "taker_fee_rate": Decimal("0.0005"),
            "market_slippage_ticks": 1,
            "stop_slippage_ticks": 2,
            "same_bar_policy": "STOP_FIRST",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"sealed V4 invariant {name} must equal {value}")
        registered = _CANDIDATES_BY_ID.get(self.candidate_id)
        if registered is None:
            raise ValueError(f"candidate_id is outside the sealed V4 registry: {self.candidate_id}")
        if (self.vwap_slope_bars, self.target_r_multiple) != registered:
            raise ValueError(
                "sealed V4 candidate parameters do not match candidate_id "
                f"{self.candidate_id}"
            )
        return self

    @property
    def variant_id(self) -> str:
        """Compatibility name used by the local strategy engine."""

        return self.candidate_id

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
            "symbol",
            "quantity_btc",
            "price_tick",
            "quantity_step",
            "minimum_quantity",
            "taker_fee_rate",
            "market_slippage_ticks",
            "stop_slippage_ticks",
            "same_bar_policy",
        )
        payload = self.model_dump(mode="json")
        return {name: payload[name] for name in names}

    def frozen_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameters": self.parameter_payload(),
            "cost_model_version": COST_MODEL_VERSION,
            "phase_a_months": list(PHASE_A_MONTHS),
            "phase_b_months": list(PHASE_B_MONTHS),
            "entry_timing": "NEXT_BAR_OPEN_NO_LOOKAHEAD",
            "exit_ambiguity": "STOP_FIRST",
            "force_flat": "UTC_DAY_END",
        }


def preregistered_candidates() -> list[ImbalanceVWAPRideV4Config]:
    return [
        ImbalanceVWAPRideV4Config(
            candidate_id=candidate_id,
            vwap_slope_bars=vwap_slope_bars,
            target_r_multiple=target_r_multiple,
        )
        for candidate_id, vwap_slope_bars, target_r_multiple in CANDIDATE_REGISTRY
    ]


def candidate_registry_payload() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "vwap_slope_bars": candidate.vwap_slope_bars,
            "target_r_multiple": str(candidate.target_r_multiple),
        }
        for candidate in preregistered_candidates()
    ]


def candidate_registry_hash() -> str:
    return sha256_value(candidate_registry_payload())


def _decimal(value: Any, default: Decimal | None = None) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        if default is not None:
            return default
        raise


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _reported_numeric(value: Any, *, allow_infinity: bool = False) -> bool:
    try:
        parsed = _decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return False
    return not parsed.is_nan() and (allow_infinity or parsed.is_finite())


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _greater(value: Any, threshold: Decimal) -> bool:
    try:
        return _decimal(value) > threshold
    except (InvalidOperation, TypeError, ValueError):
        return False


def _monthly(metrics: dict[str, Any], expected: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw = metrics.get("months")
    if not isinstance(raw, dict):
        return {}
    return {month: raw.get(month, {}) for month in expected}


def _period_pnl(months: dict[str, dict[str, Any]], selected: Iterable[str]) -> Decimal:
    return sum((_decimal(months.get(month, {}).get("net_pnl", 0), Decimal()) for month in selected), Decimal())


def _monthly_outputs_valid(months: dict[str, dict[str, Any]], expected: tuple[str, ...]) -> bool:
    required = ("executed_trades", "gross_pnl", "net_pnl", "total_costs")
    return set(months) == set(expected) and all(
        isinstance(months[month], dict)
        and all(name in months[month] for name in required)
        and all(_finite(months[month][name]) for name in required)
        and _integer(months[month]["executed_trades"], -1) >= 0
        and _decimal(months[month]["total_costs"], Decimal("-1")) >= 0
        and _decimal(months[month]["gross_pnl"], Decimal())
        - _decimal(months[month]["total_costs"], Decimal())
        == _decimal(months[month]["net_pnl"], Decimal("Infinity"))
        for month in expected
    )


def _concentrations(metrics: dict[str, Any], months: dict[str, dict[str, Any]]) -> tuple[Decimal, Decimal]:
    positive = sorted(
        (max(_decimal(item.get("net_pnl", 0), Decimal()), Decimal()) for item in months.values()),
        reverse=True,
    )
    total = sum(positive, Decimal())
    derived_best = positive[0] / total if total else Decimal("1")
    derived_best_three = sum(positive[:3], Decimal()) / total if total else Decimal("1")
    return (
        _decimal(metrics.get("maximum_positive_month_contribution", derived_best), derived_best),
        _decimal(metrics.get("best_three_positive_month_contribution", derived_best_three), derived_best_three),
    )


def _common_reconciliations(metrics: dict[str, Any]) -> tuple[bool, bool]:
    funnel = metrics.get("funnel_reconciliation", metrics.get("funnel", {}))
    long_only = metrics.get("long_only_reconciliation", {})
    funnel_names = (
        "proposed_setups",
        "invalid_setups",
        "non_executable_setups",
        "compliance_blocks",
        "executed_trades",
    )
    funnel_ok = bool(
        isinstance(funnel, dict)
        and funnel.get("reconciles")
        and all(name in funnel for name in funnel_names)
        and _integer(funnel.get("proposed_setups"), -1)
        == sum(
            _integer(funnel.get(name), -1)
            for name in (
                "invalid_setups",
                "non_executable_setups",
                "compliance_blocks",
                "executed_trades",
            )
        )
        and _integer(funnel.get("executed_trades"), -1)
        == _integer(metrics.get("executed_trades"), -2)
    )
    long_only_ok = bool(
        isinstance(long_only, dict)
        and long_only.get("reconciles")
        and _integer(long_only.get("executed_trades"), -1)
        == _integer(metrics.get("executed_trades"), -2)
        and _integer(long_only.get("long_trades"), -1)
        == _integer(metrics.get("executed_trades"), -2)
        and _integer(long_only.get("short_trades", 0), -1) == 0
        and _integer(long_only.get("short_setups", 0), -1) == 0
        and _decimal(long_only.get("short_pnl", 0), Decimal()) == 0
    )
    return funnel_ok, long_only_ok


def phase_a_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    months = _monthly(metrics, PHASE_A_MONTHS)
    active_months = sum(_integer(item.get("executed_trades", 0)) > 0 for item in months.values())
    months_with_four = sum(_integer(item.get("executed_trades", 0)) >= 4 for item in months.values())
    zero_months = sum(_integer(item.get("executed_trades", 0), -1) == 0 for item in months.values())
    quarter_pnl = {name: _period_pnl(months, period) for name, period in PHASE_A_QUARTERS}
    half_pnl = {name: _period_pnl(months, period) for name, period in PHASE_A_HALVES}
    nonnegative_quarters = sum(value >= 0 for value in quarter_pnl.values())
    positive_quarters = sum(value > 0 for value in quarter_pnl.values())
    both_halves_positive = len(half_pnl) == 2 and all(value > 0 for value in half_pnl.values())
    best_month, best_three = _concentrations(metrics, months)
    best_five = _decimal(metrics.get("best_five_positive_pnl_contribution", 1), Decimal("1"))
    funnel_ok, long_only_ok = _common_reconciliations(metrics)
    executed_trades = _integer(metrics.get("executed_trades", 0))
    monthly_trade_total = sum(
        _integer(item.get("executed_trades", 0)) for item in months.values()
    )
    gross = _decimal(metrics.get("gross_pnl", 0), Decimal("Infinity"))
    net = _decimal(metrics.get("net_pnl", 0), Decimal("Infinity"))
    fees = _decimal(metrics.get("fees", 0), Decimal("Infinity"))
    slippage = _decimal(metrics.get("slippage_cost", 0), Decimal("Infinity"))
    total_costs = _decimal(metrics.get("total_costs", 0), Decimal("Infinity"))
    required_finite_global = (
        "gross_pnl",
        "net_pnl",
        "average_gross_r",
        "average_net_r",
        "fees",
        "slippage_cost",
        "total_costs",
        "gross_risk_usd",
    )
    checks = {
        "minimum_72_trades": executed_trades >= 72,
        "monthly_trade_count_reconciles": monthly_trade_total == executed_trades,
        "ten_active_months": active_months >= 10,
        "eight_months_with_at_least_four_trades": months_with_four >= 8,
        "at_most_two_zero_months": zero_months <= 2,
        "net_pnl_positive": _greater(metrics.get("net_pnl"), Decimal()),
        "net_profit_factor_above_1_10": _greater(metrics.get("net_profit_factor"), Decimal("1.10")),
        "average_net_r_positive": _greater(metrics.get("average_net_r"), Decimal()),
        "finite_reported_drawdown": "maximum_drawdown" in metrics
        and _finite(metrics.get("maximum_drawdown"))
        and _decimal(metrics.get("maximum_drawdown"), Decimal("-1")) >= 0,
        "funnel_reconciled": funnel_ok,
        "long_only_reconciled": long_only_ok,
        "best_month_at_most_60_percent": Decimal() <= best_month <= Decimal("0.60"),
        "best_three_months_at_most_85_percent": Decimal() <= best_three <= Decimal("0.85"),
        "best_five_trades_below_65_percent": Decimal() <= best_five < Decimal("0.65"),
        "three_nonnegative_quarters": nonnegative_quarters >= 3,
        "both_half_years_positive": both_halves_positive,
        "monthly_outputs_valid": _monthly_outputs_valid(months, PHASE_A_MONTHS),
        "gross_net_cost_risk_outputs_valid": all(
            name in metrics and _finite(metrics.get(name)) for name in required_finite_global
        )
        and all(
            name in metrics and _reported_numeric(metrics.get(name), allow_infinity=True)
            for name in ("gross_profit_factor", "net_profit_factor")
        )
        and fees >= 0
        and slippage >= 0
        and total_costs >= 0
        and fees + slippage == total_costs
        and gross - total_costs == net,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_month_count": active_months,
        "months_with_at_least_four_trades": months_with_four,
        "zero_month_count": zero_months,
        "nonnegative_quarter_count": nonnegative_quarters,
        "positive_quarter_count": positive_quarters,
        "quarter_net_pnl": {name: str(value) for name, value in quarter_pnl.items()},
        "half_year_net_pnl": {name: str(value) for name, value in half_pnl.items()},
        "both_halves_positive": both_halves_positive,
        "best_month_concentration": str(best_month),
        "best_three_month_concentration": str(best_three),
    }


def _baseline_distance(config: ImbalanceVWAPRideV4Config) -> tuple[int, Decimal]:
    return (abs(config.vwap_slope_bars - 24), abs(config.target_r_multiple - Decimal("2.5")))


def rank_phase_a_candidates(
    candidates: list[tuple[ImbalanceVWAPRideV4Config, dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len({config.candidate_id for config, _ in candidates}) != len(candidates):
        raise ValueError("duplicate candidate_id in Phase A results")
    passing = [(config, metrics, phase_a_gate(metrics)) for config, metrics in candidates]
    passing = [item for item in passing if item[2]["passed"]]

    def key(item: tuple[ImbalanceVWAPRideV4Config, dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        config, metrics, gate = item
        distance = _baseline_distance(config)
        return (
            -int(gate["positive_quarter_count"]),
            -int(gate["both_halves_positive"]),
            _decimal(gate["best_month_concentration"]),
            -_decimal(metrics["net_profit_factor"]),
            _decimal(metrics["maximum_drawdown"]),
            -_decimal(metrics["average_net_r"]),
            -_integer(metrics["executed_trades"]),
            -_decimal(metrics["net_pnl"]),
            distance,
            config.candidate_id,
        )

    ranked = sorted(passing, key=key)
    output: list[dict[str, Any]] = []
    for rank, (config, metrics, gate) in enumerate(ranked, start=1):
        output.append(
            {
                "rank": rank,
                "candidate_id": config.candidate_id,
                "config": config,
                "metrics": metrics,
                "gate": gate,
                "selection_method": SELECTION_METHOD,
                "rank_trace": {
                    "positive_quarters": gate["positive_quarter_count"],
                    "both_halves_positive": gate["both_halves_positive"],
                    "best_month_concentration": gate["best_month_concentration"],
                    "net_profit_factor": str(metrics["net_profit_factor"]),
                    "maximum_drawdown": str(metrics["maximum_drawdown"]),
                    "average_net_r": str(metrics["average_net_r"]),
                    "executed_trades": _integer(metrics["executed_trades"]),
                    "net_pnl": str(metrics["net_pnl"]),
                    "baseline_distance": [str(value) for value in _baseline_distance(config)],
                    "lexical_candidate_id": config.candidate_id,
                },
            }
        )
    return output


def select_phase_a_candidate(
    candidates: list[tuple[ImbalanceVWAPRideV4Config, dict[str, Any]]],
) -> dict[str, Any] | None:
    ranked = rank_phase_a_candidates(candidates)
    return ranked[0] if ranked else None


def phase_b_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    months = _monthly(metrics, PHASE_B_MONTHS)
    active_months = sum(_integer(item.get("executed_trades", 0)) > 0 for item in months.values())
    months_with_four = sum(_integer(item.get("executed_trades", 0)) >= 4 for item in months.values())
    subperiod_pnl = {name: _period_pnl(months, period) for name, period in PHASE_B_SUBPERIODS}
    full_pnl = _decimal(metrics.get("net_pnl", 0), Decimal())
    one_positive = any(value > 0 for value in subperiod_pnl.values())
    other_floor = -(full_pnl * Decimal("0.5"))
    balanced_subperiods = full_pnl > 0 and one_positive and all(value >= other_floor for value in subperiod_pnl.values())
    best_month, _ = _concentrations(metrics, months)
    best_five = _decimal(metrics.get("best_five_positive_pnl_contribution", 1), Decimal("1"))
    funnel_ok, long_only_ok = _common_reconciliations(metrics)
    executed_trades = _integer(metrics.get("executed_trades", 0))
    monthly_trade_total = sum(
        _integer(item.get("executed_trades", 0)) for item in months.values()
    )
    gross = _decimal(metrics.get("gross_pnl", 0), Decimal("Infinity"))
    net = _decimal(metrics.get("net_pnl", 0), Decimal("Infinity"))
    fees = _decimal(metrics.get("fees", 0), Decimal("Infinity"))
    slippage = _decimal(metrics.get("slippage_cost", 0), Decimal("Infinity"))
    total_costs = _decimal(metrics.get("total_costs", 0), Decimal("Infinity"))
    checks = {
        "minimum_30_trades": executed_trades >= 30,
        "monthly_trade_count_reconciles": monthly_trade_total == executed_trades,
        "five_active_months": active_months >= 5,
        "four_months_with_at_least_four_trades": months_with_four >= 4,
        "net_pnl_positive": _greater(metrics.get("net_pnl"), Decimal()),
        "net_profit_factor_above_1_05": _greater(metrics.get("net_profit_factor"), Decimal("1.05")),
        "average_net_r_positive": _greater(metrics.get("average_net_r"), Decimal()),
        "finite_drawdown": "maximum_drawdown" in metrics
        and _finite(metrics.get("maximum_drawdown"))
        and _decimal(metrics.get("maximum_drawdown"), Decimal("-1")) >= 0,
        "funnel_reconciled": funnel_ok,
        "long_only_reconciled": long_only_ok,
        "monthly_outputs_valid": _monthly_outputs_valid(months, PHASE_B_MONTHS),
        "best_month_at_most_70_percent": Decimal() <= best_month <= Decimal("0.70"),
        "best_five_trades_below_75_percent": Decimal() <= best_five < Decimal("0.75"),
        "fixed_subperiod_diagnostics_pass": balanced_subperiods,
        "hashes_and_costs_valid": bool(metrics.get("hashes_valid", False))
        and bool(metrics.get("costs_valid", False))
        and all(
            name in metrics and _finite(metrics.get(name))
            for name in (
                "gross_pnl",
                "net_pnl",
                "average_net_r",
                "fees",
                "slippage_cost",
                "total_costs",
                "gross_risk_usd",
            )
        )
        and fees >= 0
        and slippage >= 0
        and total_costs >= 0
        and fees + slippage == total_costs
        and gross - total_costs == net,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "status": "LOCKED_TEST_PASSED" if passed else "LOCKED_TEST_FAILED",
        "evidence_label": LOCKED_EVIDENCE,
        "confirmation_evidence": False,
        "checks": checks,
        "active_month_count": active_months,
        "months_with_at_least_four_trades": months_with_four,
        "subperiod_net_pnl": {name: str(value) for name, value in subperiod_pnl.items()},
        "subperiod_floor": str(other_floor),
        "best_month_concentration": str(best_month),
    }


def alpha_eligibility(
    *,
    locked_test_status: str,
    phase_b_execution_count: int,
    frozen_candidate_valid: bool,
    proxy_eligible_trade_count: int,
    rules_artifact: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rules = rules_artifact if isinstance(rules_artifact, dict) else {}
    try:
        retrieved = datetime.fromisoformat(str(rules.get("retrieved_at", "")).replace("Z", "+00:00"))
        if retrieved.tzinfo is None:
            raise ValueError
        age_hours = (current - retrieved.astimezone(timezone.utc)).total_seconds() / 3600
    except (TypeError, ValueError):
        age_hours = math.inf
    source = str(rules.get("official_source", "")).lower()
    product = str(rules.get("product", ""))
    checks = {
        "locked_test_passed": locked_test_status == "LOCKED_TEST_PASSED",
        "exactly_one_phase_b_execution": phase_b_execution_count == 1,
        "frozen_candidate_valid": frozen_candidate_valid,
        "minimum_30_proxy_eligible_trades": proxy_eligible_trade_count >= 30,
        "official_alpha_futures_rules": bool(rules.get("official", False))
        and "alpha" in source
        and "25k" in product.lower()
        and "zero" in product.lower(),
        "rules_fresh": 0 <= age_hours <= 24,
        "rules_consistent": bool(rules.get("consistent", False))
        and bool(rules.get("content_hash")),
    }
    eligible = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "eligible": eligible,
        "status": "ELIGIBLE" if eligible else "NOT_EXECUTED",
        "reason": None if eligible else failed[0].upper(),
        "checks": checks,
        "minimum_bootstrap_paths": 20_000,
        "confirmation_evidence": False,
        "contract_accurate_fills_claimed": False,
    }


phase_a_promotion_gate = phase_a_gate
locked_test_gate = phase_b_gate
