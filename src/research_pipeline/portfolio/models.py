from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel


class PortfolioPhase(StrEnum):
    MULTI_STRATEGY_PORTFOLIO = "MULTI_STRATEGY_PORTFOLIO"
    PORTFOLIO_SIGNAL_ANALYSIS = "PORTFOLIO_SIGNAL_ANALYSIS"
    PORTFOLIO_RISK_ANALYSIS = "PORTFOLIO_RISK_ANALYSIS"
    PORTFOLIO_PROP_SIMULATION = "PORTFOLIO_PROP_SIMULATION"
    PORTFOLIO_FINAL_REVIEW = "PORTFOLIO_FINAL_REVIEW"
    COMPLETE = "COMPLETE"


class PortfolioClassification(StrEnum):
    PORTFOLIO_ACCEPTED = "PORTFOLIO_ACCEPTED"
    PORTFOLIO_ACCEPTED_EXPLORATORY = "PORTFOLIO_ACCEPTED_EXPLORATORY"
    PORTFOLIO_REJECTED_REDUNDANT = "PORTFOLIO_REJECTED_REDUNDANT"
    PORTFOLIO_REJECTED_CORRELATED = "PORTFOLIO_REJECTED_CORRELATED"
    PORTFOLIO_REJECTED_NEGATIVE_ECONOMICS = "PORTFOLIO_REJECTED_NEGATIVE_ECONOMICS"
    PORTFOLIO_REJECTED_PROP_INCOMPATIBLE = "PORTFOLIO_REJECTED_PROP_INCOMPATIBLE"
    PORTFOLIO_INSUFFICIENT_EVIDENCE = "PORTFOLIO_INSUFFICIENT_EVIDENCE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    TECHNICAL_REPAIR_REQUIRED = "TECHNICAL_REPAIR_REQUIRED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class PortfolioMemberRole(StrEnum):
    CORE = "CORE"
    DIVERSIFIER = "DIVERSIFIER"
    FREQUENCY_COMPLEMENT = "FREQUENCY_COMPLEMENT"
    REGIME_COMPLEMENT = "REGIME_COMPLEMENT"
    EXPLORATORY = "EXPLORATORY"


class ConflictPolicy(StrEnum):
    FIRST_SIGNAL_WINS = "FIRST_SIGNAL_WINS"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    STRATEGY_PRIORITY = "STRATEGY_PRIORITY"
    SKIP_CONFLICT = "SKIP_CONFLICT"
    NET_EXPOSURE = "NET_EXPOSURE"
    ALLOW_INDEPENDENT = "ALLOW_INDEPENDENT"


class RiskAllocationPolicy(StrEnum):
    EQUAL_RISK = "EQUAL_RISK"
    FIXED_STRATEGY_RISK_BUDGET = "FIXED_STRATEGY_RISK_BUDGET"
    VOLATILITY_WEIGHTED = "VOLATILITY_WEIGHTED"
    DRAWDOWN_AWARE = "DRAWDOWN_AWARE"
    MARGINAL_RISK_CONTRIBUTION = "MARGINAL_RISK_CONTRIBUTION"
    PRIORITY_BASED = "PRIORITY_BASED"


class ContributionClassification(StrEnum):
    STRONGLY_POSITIVE = "STRONGLY_POSITIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    STRONGLY_NEGATIVE = "STRONGLY_NEGATIVE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PortfolioMember(StrictModel):
    strategy_id: str
    strategy_version: str
    candidate_hash: str
    phase_c_classification: str
    phase_d_classification: str
    markets: list[str] = Field(min_length=1)
    timeframes: list[str] = Field(min_length=1)
    expected_trades_per_month: float = Field(ge=0)
    data_source_classification: str
    confidence_level: str
    role: PortfolioMemberRole
    priority: int = Field(default=0, ge=0)
    confidence_score: float = Field(default=0.5, ge=0, le=1)


class PortfolioSpec(StrictModel):
    portfolio_id: str
    version: str
    name: str
    description: str
    strategy_members: list[PortfolioMember] = Field(min_length=2)
    strategy_candidate_hashes: dict[str, str]
    strategy_code_commits: dict[str, str | None] = Field(default_factory=dict)
    dataset_hashes: dict[str, str]
    target_markets: list[str] = Field(min_length=1)
    target_timeframes: list[str] = Field(min_length=1)
    target_account_products: list[str] = Field(min_length=1)
    signal_combination_policy: str = "chronological causal merge"
    conflict_policy: ConflictPolicy = ConflictPolicy.FIRST_SIGNAL_WINS
    exposure_allocation_policy: str = "aggregate duplicate economic exposure"
    risk_budget_policy: RiskAllocationPolicy = RiskAllocationPolicy.EQUAL_RISK
    duplicate_exposure_rules: dict[str, Any] = Field(default_factory=dict)
    maximum_simultaneous_positions: int = Field(gt=0)
    maximum_total_contracts: int = Field(gt=0)
    maximum_strategy_risk_contribution: float = Field(gt=0)
    session_assumptions: list[str] = Field(min_length=1)
    prop_operating_model: str
    known_limitations: list[str] = Field(default_factory=list)
    budget: "PortfolioBudget" = Field(default_factory=lambda: PortfolioBudget())
    creation_timestamp: datetime
    specification_hash: str
    frozen: bool = False

    @model_validator(mode="after")
    def member_hashes_match(self) -> "PortfolioSpec":
        ids = [member.strategy_id for member in self.strategy_members]
        if len(ids) != len(set(ids)):
            raise ValueError("portfolio strategy members must be unique")
        for member in self.strategy_members:
            if self.strategy_candidate_hashes.get(member.strategy_id) != member.candidate_hash:
                raise ValueError(f"candidate hash mismatch for {member.strategy_id}")
        return self


class PortfolioCandidate(StrictModel):
    candidate_id: str
    portfolio_id: str
    member_strategy_ids: list[str] = Field(min_length=2)
    member_candidate_hashes: dict[str, str]
    candidate_hash: str
    eligible: bool = True
    rejection_reasons: list[str] = Field(default_factory=list)


class PortfolioSignalEvent(StrictModel):
    signal_id: str
    strategy_id: str
    market: str
    timeframe: str
    direction: str
    setup_timestamp: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime
    stop: float
    targets: list[float] = Field(default_factory=list)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    position_intent: str = "OPEN_AND_CLOSE"
    quantity_intent: int = Field(default=1, ge=1)
    fees: float = Field(default=1, ge=0)
    slippage: float = Field(default=0.5, ge=0)
    candidate_hash: str
    source_data_classification: str
    duplicate_exposure_group: str
    regime: str | None = None


class PortfolioExecution(StrictModel):
    signal_id: str
    strategy_id: str
    market: str
    direction: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    requested_contracts: int
    granted_contracts: int
    denied_quantity: int
    denial_reasons: list[str] = Field(default_factory=list)
    gross_pnl: float = 0
    fees: float = 0
    slippage: float = 0
    net_pnl: float = 0
    accepted: bool = False


class PortfolioOverlapMetrics(StrictModel):
    candidate_id: str
    total_signals_by_strategy: dict[str, int]
    unique_portfolio_signals: int
    exact_duplicates: int
    same_direction_overlaps: int
    opposite_signal_conflicts: int
    exposure_skips: int
    unique_contribution_rate: dict[str, float]
    signal_overlap_rate: float
    duplicate_exposure_rate: float
    opposite_signal_conflict_rate: float
    simultaneous_position_rate: float


class PortfolioCorrelationMetrics(StrictModel):
    candidate_id: str
    aligned_daily_periods: int
    aligned_weekly_periods: int
    aligned_monthly_periods: int
    minimum_required_periods: int
    daily_pnl_correlation: dict[str, float | None]
    weekly_pnl_correlation: dict[str, float | None]
    monthly_pnl_correlation: dict[str, float | None]
    trade_outcome_correlation: dict[str, float | None]
    drawdown_correlation: dict[str, float | None]
    worst_day_overlap: float
    worst_week_overlap: float
    losing_streak_overlap: float
    simultaneous_adverse_excursion_rate: float
    sufficient_evidence: bool
    reason: str


class PortfolioRiskResult(StrictModel):
    candidate_id: str
    policy: RiskAllocationPolicy
    executions: list[PortfolioExecution]
    total_requested_contracts: int
    total_granted_contracts: int
    contract_limit_skips: int
    risk_allocation_skips: int
    duplicate_exposure_skips: int
    conflict_skips: int
    mll_buffer_skips: int
    dlg_skips: int
    session_skips: int
    inactive_account_skips: int
    zero_legal_contract_skips: int
    account_maximum_drawdown: float
    account_minimum_balance: float
    shared_account: bool = True


class PortfolioPropMetrics(StrictModel):
    candidate_id: str
    evaluations_purchased: int = 1
    evaluations_passed: int = 0
    evaluations_failed: int = 0
    pass_rate: float = 0
    qualified_accounts: int = 0
    qualified_failures: int = 0
    first_payouts: int = 0
    total_payouts: int = 0
    payout_rate: float = 0
    median_days_to_pass: float | None = None
    median_days_to_first_payout: float | None = None
    dlg_events: int = 0
    mll_events: int = 0
    gross_pnl: float = 0
    fees: float = 0
    slippage: float = 0
    net_pnl: float = 0
    subscriptions: float = 0
    reset_costs: float = 0
    activation_fees: float = 0
    gross_payouts: float = 0
    trader_payouts: float = 0
    net_external_cashflow: float = 0
    roi: float | None = None
    cost_per_pass: float | None = None
    cost_per_payout: float | None = None
    break_even_month: float | None = None
    executable_trades_per_month: float = 0
    unique_completed_trades: int = 0
    zero_trade_months: float = 0
    winning_days_per_month: float = 0
    payout_qualifying_days_per_month: float = 0
    maximum_marked_equity_drawdown: float = 0
    strategy_pnl: dict[str, float] = Field(default_factory=dict)
    strategy_trades: dict[str, int] = Field(default_factory=dict)


class PortfolioAblationResult(StrictModel):
    candidate_id: str
    removed_strategy_id: str
    full_metrics: dict[str, Any]
    without_metrics: dict[str, Any]
    deltas: dict[str, float]
    contribution: ContributionClassification
    reason: str


class PortfolioStressResult(StrictModel):
    candidate_id: str
    scenario: str
    seed: int
    metrics: dict[str, Any]
    classification: str
    membership_unchanged: bool = True
    reason: str


class PortfolioReview(StrictModel):
    portfolio_id: str
    portfolio_version: str
    selected_candidate_id: str | None
    classification: PortfolioClassification
    best_portfolio: list[str]
    member_roles: dict[str, PortfolioMemberRole]
    unique_contribution: dict[str, float]
    excluded_strategies: dict[str, str]
    preferred_conflict_policy: ConflictPolicy
    preferred_risk_allocation: RiskAllocationPolicy
    preferred_account_product: str
    preferred_operating_model: str
    expected_trades_per_month: float
    expected_payout_frequency: float
    subscription_efficiency: float | None
    expected_external_cashflow: float
    confidence_classification: str
    primary_limitations: list[str]
    metric_citations: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str
    next_phase: str | None = None


class PortfolioBudget(StrictModel):
    minimum_strategies: int = Field(default=2, ge=2)
    maximum_strategies: int = Field(default=4, ge=2)
    maximum_candidate_portfolios: int = Field(default=50, gt=0)
    maximum_scenarios: int = Field(default=20, gt=0)
    maximum_stress_scenarios: int = Field(default=50, gt=0)
    maximum_ablation_runs: int = Field(default=50, gt=0)
    minimum_correlation_periods: int = Field(default=20, gt=1)


class PortfolioBudgetUsage(StrictModel):
    candidates: int = 0
    scenarios: int = 0
    stress_scenarios: int = 0
    ablation_runs: int = 0


PortfolioSpec.model_rebuild()
