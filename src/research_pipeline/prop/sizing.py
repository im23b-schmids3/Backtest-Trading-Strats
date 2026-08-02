from __future__ import annotations

import math

from .models import ContractSpec, PropRuleSet, RiskPolicy, RiskSizingResult, TradeSignal


class SharedExposure:
    def __init__(self, max_micros: int):
        self.max_micros = max_micros
        self.open_by_account: dict[str, int] = {}

    def current(self, account_id: str) -> int:
        return self.open_by_account.get(account_id, 0)

    def reserve(self, account_id: str, requested: int) -> tuple[int, str | None, int, int]:
        before = self.current(account_id)
        legal = max(0, min(requested, self.max_micros - before))
        reason = None if legal == requested else "ACCOUNT_CONTRACT_LIMIT"
        self.open_by_account[account_id] = before + legal
        return legal, reason, before, before + legal

    def release(self, account_id: str, quantity: int) -> None:
        self.open_by_account[account_id] = max(0, self.current(account_id) - quantity)


def risk_per_contract(trade: TradeSignal, contract: ContractSpec) -> float:
    stop_distance = abs(trade.entry_price - trade.initial_stop_price)
    return (stop_distance / contract.minimum_tick) * contract.tick_value


def requested_contracts(trade: TradeSignal, contract: ContractSpec, rule: PropRuleSet, policy: RiskPolicy, marked_equity: float, mll_buffer: float) -> tuple[int, float]:
    per_contract = risk_per_contract(trade, contract)
    if per_contract <= 0: return 0, 0.0
    if policy.kind == "FIXED_CONTRACTS": requested = policy.fixed_contracts or 0; risk = requested * per_contract
    elif policy.kind == "FIXED_INITIAL_DOLLAR_RISK": risk = policy.dollar_risk or 0; requested = math.floor(risk / per_contract)
    elif policy.kind == "MLL_PERCENTAGE_RISK": risk = rule.maximum_loss_limit * (policy.mll_percentage or 0); requested = math.floor(risk / per_contract)
    elif policy.kind == "VOLATILITY_CAPPED_RISK": risk = policy.dollar_risk or 0; requested = min(math.floor(risk / per_contract), math.floor((policy.volatility_cap or risk) / per_contract))
    elif policy.kind == "ACCOUNT_BUFFER_AWARE":
        risk = min(policy.dollar_risk or 0, max(0.0, mll_buffer * max(0.0, 1 - policy.buffer_floor)))
        requested = math.floor(risk / per_contract)
    else: raise ValueError(f"unsupported risk policy: {policy.kind}")
    return max(0, requested), per_contract


def size_trade(account_id: str, trade: TradeSignal, contract: ContractSpec, rule: PropRuleSet, policy: RiskPolicy, exposure: SharedExposure, marked_equity: float, mll_buffer: float) -> RiskSizingResult:
    requested, per_contract = requested_contracts(trade, contract, rule, policy, marked_equity, mll_buffer)
    max_contracts = policy.max_contracts_override if policy.max_contracts_override is not None else rule.contract_limits.get("micro", 0)
    if requested > max_contracts: requested = max_contracts
    legal, reason, before, after = exposure.reserve(account_id, requested)
    if legal == 0:
        reason = reason or ("ZERO_LEGAL_CONTRACTS" if requested == 0 else "ACCOUNT_CONTRACT_LIMIT")
    elif legal < requested:
        reason = "ACCOUNT_CONTRACT_LIMIT"
    return RiskSizingResult(trade_id=trade.trade_id, account_id=account_id, policy=policy.name, requested_risk=requested * per_contract, risk_per_contract=per_contract, legal_contracts=legal, skipped_reason=reason, shared_exposure_before=before, shared_exposure_after=after)
