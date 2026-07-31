from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, ValidationInfo, model_validator

from ..schemas.strategy_spec import StrictModel


class OrderType(StrEnum):
    MARKET = "MARKET"
    STOP = "STOP"
    LIMIT = "LIMIT"


class InstrumentCostConfig(StrictModel):
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    commission_per_side: float = Field(default=0, ge=0)
    exchange_fee_per_side: float = Field(default=0, ge=0)
    regulatory_fee_per_side: float = Field(default=0, ge=0)
    contract_multiplier: float = Field(default=1, gt=0)
    market_slippage_ticks: float = Field(default=0, ge=0)
    stop_slippage_ticks: float = Field(default=0, ge=0)
    limit_slippage_ticks: float = Field(default=0, ge=0)
    limit_fill_assumption: str = "touch_requires_declared_model"


class ExecutionCostConfig(StrictModel):
    model_version: str = "execution-costs-1"
    instruments: dict[str, InstrumentCostConfig] = Field(min_length=1)
    volatility_multiplier: float = Field(default=1, gt=0)
    session_open_multiplier: float = Field(default=1, gt=0)
    news_window_multiplier: float = Field(default=1, gt=0)
    configuration_hash: str = "pending"

    @model_validator(mode="after")
    def validate_hash(self, info: ValidationInfo) -> "ExecutionCostConfig":
        if (info.context or {}).get("skip_configuration_hash_validation"):
            return self
        expected = calculate_cost_config_hash(self)
        if self.configuration_hash != expected:
            raise ValueError("configuration_hash does not match canonical execution-cost configuration")
        return self


class ExecutionCostResult(StrictModel):
    instrument: str
    quantity: float
    sides: int
    commissions: float
    exchange_fees: float
    regulatory_fees: float
    slippage_cost: float
    total_cost: float
    configuration_hash: str


def calculate_cost_config_hash(config: ExecutionCostConfig | dict) -> str:
    if not isinstance(config, ExecutionCostConfig):
        data = dict(config)
        data["configuration_hash"] = str(data.get("configuration_hash") or "pending")
        config = ExecutionCostConfig.model_validate(data, context={"skip_configuration_hash_validation": True})
    payload = config.model_dump(mode="json")
    payload.pop("configuration_hash", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


class ExecutionCostEngine:
    def __init__(self, config: ExecutionCostConfig):
        self.config = config

    def calculate(
        self,
        instrument: str,
        quantity: float,
        *,
        order_types: tuple[OrderType, OrderType] = (OrderType.MARKET, OrderType.MARKET),
        sides: int = 2,
        volatility_multiplier: float | None = None,
        session_open: bool = False,
        news_window: bool = False,
    ) -> ExecutionCostResult:
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if sides not in {1, 2}:
            raise ValueError("sides must be 1 or 2")
        if len(order_types) != sides:
            raise ValueError("one order type is required for each execution side")
        try:
            item = self.config.instruments[instrument]
        except KeyError as exc:
            raise ValueError(f"unsupported instrument cost configuration: {instrument}") from exc
        multiplier = volatility_multiplier or self.config.volatility_multiplier
        if session_open:
            multiplier *= self.config.session_open_multiplier
        if news_window:
            multiplier *= self.config.news_window_multiplier
        billable_quantity = quantity * item.contract_multiplier
        commissions = item.commission_per_side * billable_quantity * sides
        exchange = item.exchange_fee_per_side * billable_quantity * sides
        regulatory = item.regulatory_fee_per_side * billable_quantity * sides
        ticks_by_type = {OrderType.MARKET: item.market_slippage_ticks, OrderType.STOP: item.stop_slippage_ticks, OrderType.LIMIT: item.limit_slippage_ticks}
        slippage_ticks = sum(ticks_by_type[order_type] for order_type in order_types) * multiplier
        slippage = slippage_ticks * item.tick_value * billable_quantity
        return ExecutionCostResult(instrument=instrument, quantity=quantity, sides=sides, commissions=commissions, exchange_fees=exchange, regulatory_fees=regulatory, slippage_cost=slippage, total_cost=commissions + exchange + regulatory + slippage, configuration_hash=self.config.configuration_hash)

    def costed_pnl(self, gross_pnl: float, result: ExecutionCostResult) -> dict[str, float | str]:
        return {"gross_pnl": gross_pnl, "commissions": result.commissions, "fees": result.exchange_fees + result.regulatory_fees, "slippage_cost": result.slippage_cost, "net_pnl": gross_pnl - result.total_cost, "execution_cost_configuration_hash": result.configuration_hash}
