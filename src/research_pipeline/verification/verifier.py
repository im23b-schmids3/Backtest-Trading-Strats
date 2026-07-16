from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .models import CheckResult, CheckStatus, ReportReconciliation, TradeDiagnostic, VerificationManifest, VerificationOutcome, VerificationResult


class VerificationRunner:
    """Deterministic B.5 checks over compact JSON/CSV diagnostic artifacts."""

    def __init__(self, manifest: VerificationManifest):
        self.manifest = manifest
        self.root = Path.cwd()
        self.artifacts: dict[str, Any] = {}
        self.files: list[str] = []

    def load(self) -> None:
        for raw_path in self.manifest.diagnostic_files:
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.root / path
            if not path.is_file():
                raise FileNotFoundError(str(path))
            self.files.append(str(path.resolve()))
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    self.artifacts[path.stem] = list(csv.DictReader(handle))
            elif path.suffix.lower() in {".json", ".yaml", ".yml"}:
                raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError(f"diagnostic artifact must be a mapping: {path}")
                self.artifacts.update(raw)
            else:
                raise ValueError(f"unsupported diagnostic format: {path.suffix}")

    def run(self) -> VerificationResult:
        try:
            self.load()
        except FileNotFoundError as exc:
            return self._result(VerificationOutcome.INSUFFICIENT_DIAGNOSTIC_DATA, [str(exc)])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._result(VerificationOutcome.TECHNICAL_FAILURE, [str(exc)])

        checks: list[CheckResult] = []
        methods = {
            "trade_pnl": self._trade_pnl, "partial_exits": self._partial_exits,
            "position_scaling": self._scaling, "fees": self._fees, "trade_counts": self._counts,
            "causality": self._causality, "session_boundary": self._sessions,
            "report_reconciliation": self._reports, "data_sources": self._data_sources,
            "determinism": self._determinism,
        }
        for name in self.manifest.required_checks:
            method = methods.get(name)
            if method is None:
                checks.append(CheckResult(check_name=name, status=CheckStatus.NOT_APPLICABLE, severity="warning", warnings=["unknown check is not executable"]))
                continue
            checks.append(method())
        if "prop" in self.manifest.applicable_capabilities:
            checks.append(self._prop_lifecycle())

        failed = [check for check in checks if check.status in {CheckStatus.FAIL, CheckStatus.MISSING}]
        missing = [check for check in failed if check.status == CheckStatus.MISSING]
        manual = [check for check in failed if check.severity == "manual"]
        if missing:
            outcome = VerificationOutcome.INSUFFICIENT_DIAGNOSTIC_DATA
        elif manual:
            outcome = VerificationOutcome.MANUAL_REVIEW_REQUIRED
        elif failed:
            outcome = VerificationOutcome.TECHNICAL_REPAIR_REQUIRED if all(check.repair_eligible for check in failed) else VerificationOutcome.MANUAL_REVIEW_REQUIRED
        else:
            outcome = VerificationOutcome.VERIFIED
        return self._result(outcome, [issue for check in failed for issue in check.blocking_issues], checks)

    def _result(self, outcome: VerificationOutcome, issues: list[str], checks: list[CheckResult] | None = None) -> VerificationResult:
        checks = checks or []
        passed = [check.check_name for check in checks if check.status == CheckStatus.PASS]
        failed = [check.check_name for check in checks if check.status in {CheckStatus.FAIL, CheckStatus.MISSING}]
        return VerificationResult(strategy_id=self.manifest.strategy_id, strategy_version=self.manifest.strategy_version,
            verification_run_id=self.manifest.verification_run_id, outcome=outcome,
            mandatory_checks_passed=passed, mandatory_checks_failed=failed,
            warnings=[warning for check in checks for warning in check.warnings], blocking_issues=issues,
            files_inspected=self.files, evidence_rows=[row for check in checks for row in check.evidence_rows],
            repair_eligibility=outcome == VerificationOutcome.TECHNICAL_REPAIR_REQUIRED,
            recommended_next_state={VerificationOutcome.VERIFIED: "BASELINE_BACKTEST", VerificationOutcome.TECHNICAL_REPAIR_REQUIRED: "TECHNICAL_REPAIR_REQUIRED", VerificationOutcome.MANUAL_REVIEW_REQUIRED: "MANUAL_REVIEW_REQUIRED", VerificationOutcome.INSUFFICIENT_DIAGNOSTIC_DATA: "INSUFFICIENT_DIAGNOSTIC_DATA", VerificationOutcome.TECHNICAL_FAILURE: "TECHNICAL_FAILURE"}[outcome], checks=checks)

    def _trades(self) -> list[TradeDiagnostic]:
        raw = self.artifacts.get("trades")
        if not isinstance(raw, list) or not raw:
            raise KeyError("trades")
        return [TradeDiagnostic.model_validate(row) for row in raw]

    def _trade_pnl(self) -> CheckResult:
        try:
            trades = self._trades()
        except KeyError as exc:
            return CheckResult(check_name="trade_pnl", status=CheckStatus.MISSING, blocking_issues=[f"missing diagnostic section: {exc.args[0]}"], repair_eligible=False)
        failures = []
        evidence = []
        tol = self.manifest.tolerance_settings["absolute_pnl"]
        for trade in trades:
            movement = trade.exit_price - trade.entry_price if trade.direction.lower() == "long" else trade.entry_price - trade.exit_price
            calculated = movement * trade.quantity * trade.contract_multiplier
            if trade.tick_size and trade.tick_value:
                calculated = movement / trade.tick_size * trade.tick_value * trade.quantity
            expected_gross = trade.expected_gross_pnl if trade.expected_gross_pnl is not None else calculated
            expected_fees = trade.expected_fees if trade.expected_fees is not None else trade.fees
            expected_slippage = trade.expected_slippage if trade.expected_slippage is not None else trade.slippage
            expected_net = expected_gross - expected_fees - expected_slippage
            evidence.append({"trade_id": trade.trade_id, "calculated_gross": calculated, "reported_gross": trade.gross_pnl})
            if abs(calculated - expected_gross) > tol or abs(trade.gross_pnl - expected_gross) > tol or abs(trade.net_pnl - expected_net) > tol:
                failures.append(f"{trade.trade_id}: PnL reconciliation failed")
        return CheckResult(check_name="trade_pnl", status=CheckStatus.FAIL if failures else CheckStatus.PASS, observed_value=len(failures), expected_value=0, tolerance=tol, evidence_rows=evidence, blocking_issues=failures, repair_eligible=True)

    def _partial_exits(self) -> CheckResult:
        rows = self.artifacts.get("exit_legs")
        if not isinstance(rows, list) or not rows:
            return CheckResult(check_name="partial_exits", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: exit_legs"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        failures = []
        for raw in rows:
            grouped[str(raw.get("trade_id"))].append(raw)
        for trade_id, raw_rows in grouped.items():
            legs = [self._leg(row) for row in raw_rows]
            initial = legs[0].initial_quantity if legs[0].initial_quantity is not None else sum(leg.leg_quantity for leg in legs) + max(0.0, legs[-1].remaining_quantity)
            if any(leg.remaining_quantity < -self.manifest.tolerance_settings["quantity"] for leg in legs): failures.append(f"{trade_id}: negative remaining quantity")
            if not legs[-1].is_open and abs(legs[-1].remaining_quantity) > self.manifest.tolerance_settings["quantity"]: failures.append(f"{trade_id}: fully closed trade retains quantity")
            if sum(leg.leg_quantity for leg in legs) > initial + self.manifest.tolerance_settings["quantity"]: failures.append(f"{trade_id}: exit quantities exceed initial position")
            if len(legs) > 1 and any(leg.leg_quantity == initial for leg in legs[1:]): failures.append(f"{trade_id}: later leg reuses full original position")
        return CheckResult(check_name="partial_exits", status=CheckStatus.FAIL if failures else CheckStatus.PASS, blocking_issues=failures, repair_eligible=True)

    @staticmethod
    def _leg(row: dict[str, Any]):
        from .models import ExitLegDiagnostic
        return ExitLegDiagnostic.model_validate(row)

    def _scaling(self) -> CheckResult:
        rows = self.artifacts.get("scaling_samples")
        if not isinstance(rows, list) or not rows:
            return CheckResult(check_name="position_scaling", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: scaling_samples"])
        base = next((float(row["pnl"]) / float(row["quantity"]) for row in rows if float(row["quantity"]) == 1), None)
        failures = [] if base is None else [f"quantity {row['quantity']} does not scale linearly" for row in rows if abs(float(row["pnl"]) - base * float(row["quantity"])) > self.manifest.tolerance_settings["absolute_pnl"]]
        return CheckResult(check_name="position_scaling", status=CheckStatus.FAIL if base is None or failures else CheckStatus.PASS, blocking_issues=failures or ([] if base is not None else ["quantity 1 scaling sample missing"]), repair_eligible=True)

    def _fees(self) -> CheckResult:
        rows = self.artifacts.get("fee_reconciliation")
        if not isinstance(rows, list) or not rows:
            return CheckResult(check_name="fees", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: fee_reconciliation"])
        failures = [f"{row.get('trade_id', 'unknown')}: fee reconciliation failed" for row in rows if abs(float(row.get("expected_fees", row.get("fees", 0))) - float(row.get("fees", 0))) > self.manifest.tolerance_settings["fees"]]
        return CheckResult(check_name="fees", status=CheckStatus.FAIL if failures else CheckStatus.PASS, blocking_issues=failures, repair_eligible=True)

    def _counts(self) -> CheckResult:
        row = self.artifacts.get("trade_counts")
        if not isinstance(row, dict): return CheckResult(check_name="trade_counts", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: trade_counts"])
        definition = str(row.get("total_trades_definition", "")).strip()
        if not definition: return CheckResult(check_name="trade_counts", status=CheckStatus.FAIL, severity="manual", blocking_issues=["ambiguous total_trades definition"])
        if int(row.get("order_versions", 0)) != int(row.get("completed_positions", row.get("total_trades", 0))) and row.get("total_trades") == row.get("order_versions"):
            return CheckResult(check_name="trade_counts", status=CheckStatus.FAIL, blocking_issues=["order versions reported as completed trades"], repair_eligible=True)
        return CheckResult(check_name="trade_counts", status=CheckStatus.PASS)

    def _causality(self) -> CheckResult:
        row = self.artifacts.get("causality")
        if not isinstance(row, dict): return CheckResult(check_name="causality", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: causality"])
        if row.get("lookahead_detected") is True: return CheckResult(check_name="causality", status=CheckStatus.FAIL, blocking_issues=["look-ahead detected"], repair_eligible=True)
        return CheckResult(check_name="causality", status=CheckStatus.PASS)

    def _sessions(self) -> CheckResult:
        row = self.artifacts.get("session_boundary")
        if not isinstance(row, dict): return CheckResult(check_name="session_boundary", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: session_boundary"])
        if row.get("terminal_flatten_cluster") is True: return CheckResult(check_name="session_boundary", status=CheckStatus.FAIL, blocking_issues=["terminal flatten cluster detected"], repair_eligible=True)
        return CheckResult(check_name="session_boundary", status=CheckStatus.PASS)

    def _reports(self) -> CheckResult:
        rows = self.artifacts.get("report_reconciliation")
        if not isinstance(rows, list) or not rows: return CheckResult(check_name="report_reconciliation", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: report_reconciliation"])
        failures = [f"{row.get('metric', 'unknown')}: report mismatch" for row in rows if ReportReconciliation.model_validate(row).status != "PASS"]
        return CheckResult(check_name="report_reconciliation", status=CheckStatus.FAIL if failures else CheckStatus.PASS, blocking_issues=failures, repair_eligible=True)

    def _data_sources(self) -> CheckResult:
        rows = self.artifacts.get("data_sources")
        if not isinstance(rows, list) or not rows: return CheckResult(check_name="data_sources", status=CheckStatus.MISSING, blocking_issues=["missing diagnostic section: data_sources"])
        failures = ["provider or synthetic transformation is undocumented" for row in rows if not row.get("provider") or (row.get("native_or_proxy") in {"proxy", "synthetic"} and not row.get("synthetic_transformation"))]
        return CheckResult(check_name="data_sources", status=CheckStatus.FAIL if failures else CheckStatus.PASS, severity="manual" if failures else "blocking", blocking_issues=failures, repair_eligible=False)

    def _determinism(self) -> CheckResult:
        rows = self.artifacts.get("replay_hashes")
        if not isinstance(rows, list) or len(rows) < 2: return CheckResult(check_name="determinism", status=CheckStatus.MISSING, blocking_issues=["two replay hashes are required"])
        if len(set(map(str, rows))) != 1: return CheckResult(check_name="determinism", status=CheckStatus.FAIL, blocking_issues=["replay hashes differ"], repair_eligible=True)
        return CheckResult(check_name="determinism", status=CheckStatus.PASS)

    def _prop_lifecycle(self) -> CheckResult:
        rows = self.artifacts.get("lifecycle")
        if not isinstance(rows, list) or not rows: return CheckResult(check_name="lifecycle", status=CheckStatus.MISSING, blocking_issues=["prop capability requires lifecycle diagnostics"])
        return CheckResult(check_name="lifecycle", status=CheckStatus.PASS)


def artifact_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
