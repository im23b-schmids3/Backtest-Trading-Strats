from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class PropPhase(StrEnum):
    ENTRY_VERIFICATION = "ENTRY_VERIFICATION"
    RULE_VERIFICATION = "RULE_VERIFICATION"
    CONTRACT_VERIFICATION = "CONTRACT_VERIFICATION"
    RECONCILIATION = "RECONCILIATION"
    RISK_SIZING = "RISK_SIZING"
    PROP_SIMULATION = "PROP_SIMULATION"
    PROP_ECONOMICS_REVIEW = "PROP_ECONOMICS_REVIEW"
    COMPLETE = "COMPLETE"


class PropBudget(StrictModel):
    max_scenarios: int = Field(default=20, ge=0)
    max_accounts_per_scenario: int = Field(default=5, ge=0)
    max_replay_duration_days: int = Field(default=365, ge=0)
    max_artifact_size_mb: float = Field(default=100, ge=0)
    max_policy_variants: int = Field(default=5, ge=0)
    max_concurrent_evaluations: int = Field(default=5, ge=0)


class PropBudgetUsage(StrictModel):
    scenarios: int = Field(default=0, ge=0)
    accounts: int = Field(default=0, ge=0)
    replay_days: int = Field(default=0, ge=0)
    artifact_size_mb: float = Field(default=0, ge=0)
    policy_variants: int = Field(default=0, ge=0)
    concurrent_evaluations: int = Field(default=0, ge=0)


class PropClassification(StrEnum):
    PROP_ACCEPTED_STANDALONE = "PROP_ACCEPTED_STANDALONE"
    PROP_ACCEPTED_PORTFOLIO_COMPONENT = "PROP_ACCEPTED_PORTFOLIO_COMPONENT"
    OWN_CAPITAL_ONLY = "OWN_CAPITAL_ONLY"
    REJECTED_PROP_INCOMPATIBLE = "REJECTED_PROP_INCOMPATIBLE"
    REJECTED_NEGATIVE_ECONOMICS = "REJECTED_NEGATIVE_ECONOMICS"
    INSUFFICIENT_FUTURES_DATA = "INSUFFICIENT_FUTURES_DATA"
    INSUFFICIENT_PROP_EVIDENCE = "INSUFFICIENT_PROP_EVIDENCE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    TECHNICAL_REPAIR_REQUIRED = "TECHNICAL_REPAIR_REQUIRED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class ConfidenceClass(StrEnum):
    NATIVE_FUTURES_SUPPORTED = "NATIVE_FUTURES_SUPPORTED"
    PROXY_EXPLORATORY = "PROXY_EXPLORATORY"
    SYNTHETIC_PROXY_HIGH_UNCERTAINTY = "SYNTHETIC_PROXY_HIGH_UNCERTAINTY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CRYPTO_PERPETUAL_PROXY = "CRYPTO_PERPETUAL_PROXY"


class PropRuleSet(StrictModel):
    provider: str
    product: str
    version: str
    account_size: float = Field(gt=0)
    evaluation_price: float = Field(ge=0)
    monthly_subscription: float = Field(ge=0)
    activation_fee: float = Field(ge=0)
    reset_fee: float = Field(ge=0)
    profit_target: float = Field(gt=0)
    maximum_loss_limit: float = Field(gt=0)
    drawdown_behavior: str
    drawdown_type: str
    intraday_or_end_of_day: str
    daily_loss_guard: float | None = Field(default=None, ge=0)
    contract_limits: dict[str, int]
    consistency_rule: float | None = Field(default=None, ge=0, le=1)
    winning_day_requirements: dict[str, Any]
    minimum_trading_days: int = Field(default=0, ge=0)
    payout_frequency: str
    payout_minimum: float = Field(ge=0)
    payout_maximum: float = Field(gt=0)
    payout_split: float = Field(gt=0, le=1)
    payout_cycle_reset_behavior: str
    account_cancellation_behavior: str
    billing_after_failure: str
    billing_after_pass: str
    session_close_requirements: str
    prohibited_practices: list[str]
    automation_restrictions: list[str]
    maximum_account_allocation: int = Field(gt=0)
    rule_source_urls: list[str] = Field(min_length=1)
    verification_date: datetime
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    official_verified: bool = False
    source_hash: str


class ContractSpec(StrictModel):
    exchange: str
    symbol: str
    product_name: str
    contract_unit: str
    minimum_tick: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    point_value: float = Field(gt=0)
    currency: str
    session_definition: str
    micro_mini_relationship: str
    official_source: str
    verification_date: datetime
    data_quality: str
    active: bool = True


class MarketMapping(StrictModel):
    strategy_market: str
    source_symbol: str
    source_provider: str
    target_futures_contract: str
    mapping_method: str
    native_or_proxy: str
    synthetic_transformation: str | None = None
    reference_price_method: str
    duplicate_exposure_group: str
    date_range: dict[str, str]
    confidence_level: ConfidenceClass
    limitations: list[str] = Field(default_factory=list)


class TradeSignal(StrictModel):
    trade_id: str
    timestamp: datetime
    exit_timestamp: datetime
    source_market: str
    timeframe: str
    direction: str
    entry_price: float
    initial_stop_price: float
    exit_price: float
    source_return: float
    fees: float = Field(default=0, ge=0)
    slippage: float = Field(default=0, ge=0)
    trade_legs: list[dict[str, Any]] = Field(default_factory=list)


class FuturesTradeReconciliation(StrictModel):
    trade_id: str
    source_market: str
    source_entry: float
    source_exit: float
    source_return: float
    mapped_entry: float
    mapped_exit: float
    direction: str
    contract: str
    tick_size: float
    tick_value: float
    point_value: float
    quantity: int = Field(ge=0)
    gross_pnl: float
    fees: float = Field(ge=0)
    slippage: float = Field(ge=0)
    net_pnl: float
    mapping_hash: str
    contract_registry_hash: str


class RiskPolicy(StrictModel):
    name: str
    kind: str
    fixed_contracts: int | None = Field(default=None, ge=0)
    dollar_risk: float | None = Field(default=None, gt=0)
    mll_percentage: float | None = Field(default=None, gt=0, le=1)
    volatility_cap: float | None = Field(default=None, gt=0)
    buffer_floor: float = Field(default=0.25, ge=0, le=1)
    max_contracts_override: int | None = Field(default=None, ge=0)


class PropScenarioConfig(StrictModel):
    scenario_id: str
    account_product: str
    risk_policy: RiskPolicy
    operating_model: str
    evaluation_exit_policy: str
    daily_risk_policy: str
    market_portfolio: list[str] = Field(min_length=1)
    timeframe_portfolio: list[str] = Field(min_length=1)
    simulation_period: str = "full_history"
    max_accounts: int = Field(default=1, gt=0)
    max_days: int = Field(default=365, gt=0)
    enabled: bool = True


class RiskSizingResult(StrictModel):
    trade_id: str
    account_id: str
    policy: str
    requested_risk: float
    risk_per_contract: float
    legal_contracts: int
    skipped_reason: str | None = None
    shared_exposure_before: int = 0
    shared_exposure_after: int = 0


class AccountEvent(StrictModel):
    account_id: str
    timestamp: datetime
    event_type: str
    balance: float
    marked_equity: float
    realized_pnl: float
    unrealized_pnl: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    fees: float
    dlg_used: float
    mll_threshold: float
    remaining_mll_buffer: float
    reason: str


class PayoutRecord(StrictModel):
    account_id: str
    payout_number: int = Field(gt=0)
    eligibility_timestamp: datetime
    winning_day_count: int = Field(ge=0)
    winning_day_dates: list[str]
    largest_winning_day: float
    payout_cycle_profit: float
    consistency_percentage: float | None
    maximum_legal_request: float
    gross_payout_requested: float
    provider_share: float
    trader_share: float
    payout_date: datetime
    balance_before: float
    balance_after: float
    cycle_reset_behavior: str


class PropScenarioMetrics(StrictModel):
    scenario_id: str
    evaluations_purchased: int = 0
    evaluations_passed: int = 0
    evaluations_failed: int = 0
    voluntarily_cancelled_evaluations: int = 0
    censored_evaluations: int = 0
    qualified_accounts_created: int = 0
    qualified_failures: int = 0
    first_payouts: int = 0
    second_payouts: int = 0
    third_payouts: int = 0
    total_payouts: int = 0
    pass_rate: float = 0
    first_payout_rate_per_started_evaluation: float = 0
    first_payout_rate_per_passed_evaluation: float = 0
    median_days_to_pass: float | None = None
    median_days_to_first_payout: float | None = None
    average_evaluation_lifetime: float | None = None
    average_qualified_lifetime: float | None = None
    average_trades_per_evaluation: float = 0
    average_trades_per_qualified: float = 0
    average_contracts: float = 0
    average_initial_risk: float = 0
    maximum_marked_equity_drawdown: float = 0
    dlg_events: int = 0
    mll_events: int = 0
    contract_limit_skips: int = 0
    risk_cap_skips: int = 0
    zero_legal_contract_skips: int = 0
    session_forced_exits: int = 0
    gross_trading_pnl: float = 0
    fees: float = 0
    slippage: float = 0
    net_trading_pnl: float = 0
    evaluation_subscriptions: float = 0
    reset_costs: float = 0
    activation_fees: float = 0
    qualified_fees: float = 0
    gross_payout_requests: float = 0
    trader_payouts: float = 0
    net_external_cashflow: float = 0
    cost_per_pass: float | None = None
    cost_per_first_payout: float | None = None
    payout_to_subscription_ratio: float | None = None
    roi_on_external_costs: float | None = None
    profitable_external_path_percentage: float = 0
    annualized_evaluation_purchases: float = 0
    annualized_payouts: float = 0
    annualized_net_cashflow: float = 0
    maximum_capital_outlay: float = 0
    break_even_month: float | None = None


class PropDataLimitations(StrictModel):
    confidence: ConfidenceClass
    native_futures_data: bool
    proxy_data: bool
    synthetic_return_mapped_proxy: bool
    short_history: bool
    incomplete_rollover_handling: bool
    incomplete_intrabar_equity: bool
    missing_news_calendar: bool
    missing_live_fill_information: bool
    warnings: list[str] = Field(default_factory=list)


class ComplianceResult(StrictModel):
    compliant: bool
    status: str
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rule_hash: str
    checked_at: datetime


class BillingEvent(StrictModel):
    account_id: str
    timestamp: datetime
    event_type: str
    amount: float = Field(ge=0)
    reason: str


class AccountSummary(StrictModel):
    account_id: str
    account_type: str
    status: str
    started_at: datetime
    ended_at: datetime | None = None
    pass_timestamp: datetime | None = None
    failure_timestamp: datetime | None = None
    cancellation_timestamp: datetime | None = None
    trades: int = 0
    qualified_trades: int = 0
    total_fees: float = 0
    total_pnl: float = 0
    payouts: list[PayoutRecord] = Field(default_factory=list)
    events: list[AccountEvent] = Field(default_factory=list)
    billing_events: list[BillingEvent] = Field(default_factory=list)


class SimulationResult(StrictModel):
    scenario: PropScenarioConfig
    metrics: PropScenarioMetrics
    accounts: list[AccountSummary]
    payouts: list[PayoutRecord] = Field(default_factory=list)
    billing_events: list[BillingEvent] = Field(default_factory=list)
    reconciliations: list[FuturesTradeReconciliation] = Field(default_factory=list)
    risk_sizing: list[RiskSizingResult] = Field(default_factory=list)
    data_limitations: PropDataLimitations
    compliance: ComplianceResult
    b5_verified: bool = False


class PropEconomicsReview(StrictModel):
    strategy_id: str
    strategy_version: str
    classification: PropClassification
    scenario_id: str
    metrics: PropScenarioMetrics
    compliance: ComplianceResult
    data_limitations: PropDataLimitations
    metrics_cited: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str
    next_phase: str | None = None
