from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from ..schemas.strategy_spec import StrictModel


class VerificationOutcome(StrEnum):
    VERIFIED = "VERIFIED"
    TECHNICAL_REPAIR_REQUIRED = "TECHNICAL_REPAIR_REQUIRED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    INSUFFICIENT_DIAGNOSTIC_DATA = "INSUFFICIENT_DIAGNOSTIC_DATA"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"


class VerificationManifest(StrictModel):
    strategy_id: str
    strategy_version: str
    implementation_commit: str | None = None
    verification_run_id: str
    applicable_capabilities: list[str] = Field(default_factory=list)
    diagnostic_files: list[str] = Field(min_length=1)
    tolerance_settings: dict[str, float] = Field(default_factory=lambda: {
        "absolute_pnl": 1e-9, "relative_pnl": 1e-9, "quantity": 1e-9,
        "fees": 1e-9, "timestamp": 0.0, "hash": 0.0, "report_metric": 1e-9,
    })
    required_checks: list[str] = Field(default_factory=lambda: [
        "trade_pnl", "partial_exits", "position_scaling", "fees", "trade_counts",
        "causality", "session_boundary", "report_reconciliation", "data_sources", "determinism",
    ])
    optional_checks: list[str] = Field(default_factory=list)
    data_sources: list[dict[str, Any]] = Field(default_factory=list)
    expected_contracts: list[dict[str, Any]] = Field(default_factory=list)
    expected_sessions: list[dict[str, Any]] = Field(default_factory=list)
    known_exemptions: list[str] = Field(default_factory=list)
    approved_invariants_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_hash: str = ""

    @model_validator(mode="after")
    def hash_manifest(self) -> "VerificationManifest":
        if not self.manifest_hash:
            raw = self.model_dump(mode="json", exclude={"manifest_hash"})
            self.manifest_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self

    @classmethod
    def load(cls, path: str | Path) -> "VerificationManifest":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        manifest = cls.model_validate(raw)
        if manifest.manifest_hash != cls.model_validate({**raw, "manifest_hash": ""}).manifest_hash and raw.get("manifest_hash"):
            raise ValueError("verification manifest hash mismatch")
        return manifest

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return target


class TradeDiagnostic(StrictModel):
    trade_id: str
    signal_id: str | None = None
    strategy_id: str | None = None
    market: str
    timeframe: str | None = None
    direction: str
    entry_timestamp: str
    exit_timestamp: str | None = None
    entry_price: float
    exit_price: float | None = None
    quantity: float
    gross_pnl: float
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: float
    exit_reason: str | None = None
    account_id: str | None = None
    data_source: str | None = None
    is_proxy: bool = False
    mapping_id: str | None = None
    reconciliation_status: str | None = None
    contract_multiplier: float = 1.0
    tick_size: float | None = None
    tick_value: float | None = None
    fee_basis: str = "fixed"
    expected_gross_pnl: float | None = None
    expected_fees: float | None = None
    expected_slippage: float | None = None


class ExitLegDiagnostic(StrictModel):
    trade_id: str
    leg_number: int = Field(ge=1)
    leg_type: str
    leg_quantity: float
    price: float
    gross_pnl: float
    fees: float = 0.0
    net_pnl: float
    remaining_quantity: float
    initial_quantity: float | None = None
    is_open: bool = False


class ContractDiagnostic(StrictModel):
    source_symbol: str
    target_contract: str
    contract_multiplier: float
    tick_size: float
    tick_value: float
    point_value: float
    specification_source: str
    verification_date: str
    synthetic_or_native: str


class LifecycleDiagnostic(StrictModel):
    account_id: str
    account_type: str
    state_before: str
    state_after: str
    timestamp: str
    event_type: str
    balance: float
    marked_equity: float
    DLG_value: float | None = None
    MLL_threshold: float | None = None
    transition_reason: str


class ReportReconciliation(StrictModel):
    metric: str
    source_report: str
    source_rows: int
    recomputed_value: float
    reported_value: float
    absolute_difference: float | None = None
    relative_difference: float | None = None
    tolerance: float = 1e-9
    status: str | None = None

    @model_validator(mode="after")
    def calculate_differences(self) -> "ReportReconciliation":
        absolute_difference = abs(self.recomputed_value - self.reported_value)
        relative_difference = absolute_difference / max(abs(self.recomputed_value), 1e-12)
        object.__setattr__(self, "absolute_difference", absolute_difference)
        object.__setattr__(self, "relative_difference", relative_difference)
        object.__setattr__(self, "status", "PASS" if absolute_difference <= self.tolerance or relative_difference <= self.tolerance else "FAIL")
        return self


class CheckResult(StrictModel):
    check_name: str
    applicability: str = "mandatory"
    status: CheckStatus
    severity: str = "blocking"
    observed_value: Any = None
    expected_value: Any = None
    tolerance: float | None = None
    evidence_path: str | None = None
    evidence_rows: list[dict[str, Any]] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repair_eligible: bool = False


class VerificationResult(StrictModel):
    strategy_id: str
    strategy_version: str
    verification_run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: VerificationOutcome
    mandatory_checks_passed: list[str] = Field(default_factory=list)
    mandatory_checks_failed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    files_inspected: list[str] = Field(default_factory=list)
    evidence_rows: list[dict[str, Any]] = Field(default_factory=list)
    repair_eligibility: bool = False
    recommended_next_state: str
    checks: list[CheckResult] = Field(default_factory=list)
