from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class DataClassification(StrEnum):
    AVAILABLE_NATIVE = "AVAILABLE_NATIVE"
    AVAILABLE_PROXY = "AVAILABLE_PROXY"
    AVAILABLE_SYNTHETIC_PROXY = "AVAILABLE_SYNTHETIC_PROXY"
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    UNAVAILABLE = "UNAVAILABLE"
    MANUAL_MAPPING_REQUIRED = "MANUAL_MAPPING_REQUIRED"


class AdapterIdentity(StrictModel):
    strategy_id: str
    strategy_version: str
    implementation_module: str
    entry_point: str
    specification_hash: str
    code_commit: str | None = None
    worktree_path: str | None = None
    adapter_version: str = "phase-f2-1"
    schema_version: str = "1"


class AdapterCapabilities(StrictModel):
    baseline: bool = True
    parameter_experiment: bool = True
    frozen_candidate: bool = True
    walk_forward: bool = True
    holdout: bool = True
    stress: bool = True
    throughput: bool = True
    phase_d_export: bool = True
    phase_e_export: bool = True
    trade_diagnostics: bool = True
    supported_markets: list[str] = Field(default_factory=list)
    supported_timeframes: list[str] = Field(default_factory=list)
    parameter_families: list[str] = Field(default_factory=list)
    data_providers: list[str] = Field(default_factory=list)


class ResearchParameterFamily(StrictModel):
    family_name: str
    parameters: list[str] = Field(min_length=1)
    value_type: str
    legal_range: dict[str, float | int] | None = None
    bounded_candidate_values: list[Any] = Field(min_length=1)
    baseline_value: Any
    optimization_direction: str
    constraints: list[str] = Field(default_factory=list)
    immutable_dependencies: list[str] = Field(default_factory=list)
    maximum_evaluations: int = Field(ge=1)
    enabled: bool = True


class AdapterHealth(StrictModel):
    identity: AdapterIdentity
    capabilities: AdapterCapabilities
    importable: bool
    compatible: bool
    healthy: bool
    errors: list[str] = Field(default_factory=list)
    checked_at: datetime


class DataAvailability(StrictModel):
    market: str
    timeframe: str
    classification: DataClassification
    provider: str | None = None
    source_symbol: str | None = None
    path: str | None = None
    dataset_hash: str | None = None
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    rows: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    declared_substitution: str | None = None


class NormalizedTrade(StrictModel):
    trade_id: str
    signal_id: str
    market: str
    timeframe: str
    direction: str
    setup_time: datetime | None = None
    entry_time: datetime
    exit_time: datetime
    entry: float
    stop: float | None = None
    targets: list[float] = Field(default_factory=list)
    legs: list[dict[str, Any]] = Field(default_factory=list)
    quantity: float
    fees: float = 0.0
    slippage: float = 0.0
    gross_pnl: float
    net_pnl: float
    exit_reason: str
    source_classification: DataClassification


class BacktestRun(StrictModel):
    run_id: str
    strategy_id: str
    strategy_version: str
    candidate_hash: str
    dataset_hashes: list[str]
    code_commit: str | None = None
    configuration_hash: str
    phase: str
    parameters: dict[str, Any]
    starting_capital: float
    ending_capital: float
    gross_pnl: float
    net_pnl: float
    fees: float
    slippage: float
    trade_count: int = Field(ge=0)
    win_rate: float | None = None
    expectancy: float
    profit_factor: float | None = None
    maximum_drawdown: float
    risk_adjusted_metric: float | None = None
    trades: list[NormalizedTrade] = Field(default_factory=list)
    data: list[DataAvailability] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)


class PhaseDEvent(StrictModel):
    event_id: str
    strategy_id: str
    strategy_version: str
    candidate_hash: str
    market: str
    source_symbol: str
    futures_mapping_candidate: str | None = None
    entry: float
    stop: float | None = None
    exit: float
    position_intent: str
    timestamp: datetime
    direction: str
    source_classification: DataClassification


class PhaseEEligibility(StrictModel):
    strategy_id: str
    strategy_version: str
    candidate_hash: str
    phase_c_classification: str
    phase_d_classification: str | None = None
    data_confidence: str
    expected_trade_frequency: float | None = None
    eligible_markets: list[str] = Field(default_factory=list)
    eligible_timeframes: list[str] = Field(default_factory=list)
    outcome: str
    reasons: list[str] = Field(default_factory=list)


class ImplementationManifest(StrictModel):
    master_run_id: str
    strategy_id: str
    strategy_version: str
    specification_hash: str
    base_commit: str
    implementation_commit: str | None = None
    worktree_path: str
    branch: str
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    adapter_registration: str
    strategy_entry_point: str
    tests_added: list[str] = Field(default_factory=list)
    verification_command: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    code_hash: str | None = None
    adapter_version: str
