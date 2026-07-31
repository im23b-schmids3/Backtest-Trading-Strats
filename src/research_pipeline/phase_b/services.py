from __future__ import annotations


import json
import hashlib
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError


from ..controller.pipeline_controller import PipelineController
from ..enums import ApprovalStatus, PipelineState
from ..errors import ExternalSpecificationRequired, ImmutableSpecificationError, InvalidTransitionError, RegistryError
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.strategy_spec import ParameterFamily, StrategySpec, calculate_specification_hash, save_strategy_spec
from ..validation.specification_semantics import (
    SpecificationProvenance, SpecificationValidationIssue, SpecificationValidationReport,
    semantic_validate,
)
from ..runners.codex_runner import CodexRunner, is_restricted_execution_failure
from ..runners.test_runner import DeterministicTestRunner
from ..runners.worktree_manager import WorktreeManager
from .models import (
    ApprovalResult, CodexExecutionResult, FinalPhaseBSummary, GeneratedStrategySpec,
    ImplementationPlan, RegistrationResult, SpecificationValidationResult, TestResult,
    WorkflowInput,
)
from .prompt_builder import build_implementation_prompt, build_spec_agent_prompt
from ..specification_executor.jobs import SpecificationJobService
from ..specification_executor.models import SpecificationCompletion, SpecificationCompletionStatus, SpecificationJobType


class PhaseBService:
    def __init__(self, registry_path: str | Path | None = None, codex_runner: CodexRunner | None = None):
        self.registry_path = Path(registry_path or os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3"))
        self.registry = Registry(Database(self.registry_path))
        self.controller = PipelineController(self.registry)
        self.codex = codex_runner or CodexRunner()
        self.tests = DeterministicTestRunner()

    @staticmethod
    def _strategy_id(name: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
        return value or "strategy"

    @staticmethod
    def _hash_payload(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def _intake_run_id(self, workflow: WorkflowInput, strategy_id: str) -> str:
        if workflow.run_id:
            return workflow.run_id
        payload = workflow.model_dump(mode="json", exclude={"run_id", "max_generation_attempts", "max_repair_attempts"})
        return f"spec-{strategy_id}-{self._hash_payload(payload)[:12]}"

    def _specification_root(self, workflow: WorkflowInput, strategy_id: str, run_id: str) -> Path:
        root = Path(workflow.repository_root).resolve() / "research_runs" / strategy_id / run_id / "specification"
        (root / "intake").mkdir(parents=True, exist_ok=True)
        (root / "attempts").mkdir(parents=True, exist_ok=True)
        (root / "canonical").mkdir(parents=True, exist_ok=True)
        (root / "failure").mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _provenance(workflow: WorkflowInput) -> SpecificationProvenance:
        text = f"{workflow.natural_language_description} {workflow.optional_notes or ''}".lower()
        technical: list[str] = []
        if "market open" in text or "cash-session open" in text:
            technical.append("TECHNICAL_TRANSLATION: US market open resolves to the declared local regular-session time, not a fixed UTC timestamp.")
        return SpecificationProvenance(confirmed=[*workflow.confirmed_facts, *workflow.requested_markets, *workflow.requested_timeframes],
            technical_translations=technical, assumptions=[*workflow.assumptions], missing_information=[*workflow.missing_information],
            blocking_ambiguities=[*workflow.ambiguities, *PhaseBService._ambiguities(workflow)], source_intake_hash=PhaseBService._hash_payload(workflow.model_dump(mode="json")))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _extract_structured_payload(output: str) -> dict[str, Any]:
        candidate = output.strip()
        if not candidate:
            raise ValueError("Codex returned an empty specification payload")
        fenced = re.findall(r"```(?:yaml|yml|json)?\s*(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
        if len(fenced) > 1:
            raise ValueError("Codex returned multiple competing structured payloads")
        if fenced:
            candidate = fenced[0].strip()
        try:
            documents = list(yaml.safe_load_all(candidate))
        except yaml.YAMLError as exc:
            raise ValueError(f"Codex returned malformed YAML/JSON: {exc}") from exc
        if len(documents) != 1 or not isinstance(documents[0], dict):
            raise ValueError("Codex output must be exactly one YAML/JSON mapping with no commentary")
        return documents[0]

    @staticmethod
    def _pydantic_issue(exc: ValidationError) -> list[SpecificationValidationIssue]:
        return [SpecificationValidationIssue(error_code="PYDANTIC_VALIDATION_ERROR", field_path=".".join(str(item) for item in error.get("loc", ())) or "specification",
            received_value=error.get("input"), expected_constraint=str(error.get("type", "valid value")), explanation=error.get("msg", "schema validation failed"),
            repair_hint="Return one complete specification matching the canonical StrategySpec fields and types.") for error in exc.errors()]

    def _validate_payload(
        self,
        raw: dict[str, Any],
        workflow: WorkflowInput,
        provenance: SpecificationProvenance,
    ) -> tuple[
        StrategySpec | None,
        SpecificationValidationReport,
        SpecificationProvenance,
        dict[str, Any],
    ]:
        normalized, semantic, provenance = semantic_validate(
            raw,
            provenance=provenance,
        )
        issues = list(semantic.errors)
        spec: StrategySpec | None = None
        try:
            data = dict(normalized)
            data["specification_hash"] = "pending"

            # Validate and normalize all field types before calculating the
            # canonical hash.
            candidate = StrategySpec.model_validate(
                data,
                context={"skip_specification_hash_validation": True},
            )

            expected_hash = calculate_specification_hash(candidate)

            # Use the normalized representation for final validation.
            validated_data = candidate.model_dump(mode="python")
            validated_data["specification_hash"] = expected_hash

            spec = StrategySpec.model_validate(validated_data)

        except ValidationError as exc:
            issues.extend(self._pydantic_issue(exc))

        except (TypeError, ValueError, KeyError) as exc:
            issues.append(
                SpecificationValidationIssue(
                    error_code="SPECIFICATION_PAYLOAD_INVALID",
                    field_path="specification",
                    received_value=raw,
                    expected_constraint="one complete StrategySpec mapping",
                    explanation=str(exc),
                    repair_hint="Return all required fields with valid values.",
                )
            )

        report = SpecificationValidationReport(
            valid=not issues,
            pydantic_valid=(
                spec is not None
                and not any(
                    item.error_code.startswith("PYDANTIC")
                    for item in issues
                )
            ),
            semantic_valid=not semantic.errors,
            errors=issues,
            blocking_ambiguities=provenance.blocking_ambiguities,
            normalized_payload_hash=semantic.normalized_payload_hash,
        )

        approval_blocked_only = (
            bool(spec)
            and bool(issues)
            and all(
                item.error_code == "BLOCKING_AMBIGUITY"
                for item in issues
            )
        )

        return (
            spec if (not issues or approval_blocked_only) else None,
            report,
            provenance,
            normalized,
        )

    def _approval_path(self, workflow: WorkflowInput, strategy_id: str, version: str, run_id: str) -> Path:
        draft_dir = Path(workflow.repository_root).resolve() / "research_registry" / "spec_drafts"
        draft_dir.mkdir(parents=True, exist_ok=True)
        base = draft_dir / f"{strategy_id}_v{version}.yaml"
        if not base.exists():
            return base
        return draft_dir / f"{strategy_id}_v{version}-{run_id}.yaml"

    def _persist_attempt(self, root: Path, run_id: str, strategy_id: str, attempt: int, *, raw_output: str,
                         report: SpecificationValidationReport, provenance: SpecificationProvenance,
                         invocation: dict[str, Any], prompt: str | None, status: str) -> None:
        attempt_root = root / "attempts" / f"attempt-{attempt:03d}"
        attempt_root.mkdir(parents=True, exist_ok=True)
        draft_path = attempt_root / "draft.yaml"
        draft_path.write_text(raw_output, encoding="utf-8")
        invocation_path = attempt_root / "codex_invocation.json"
        invocation_hash = self._write_json(invocation_path, invocation)
        validation_path = attempt_root / "validation.json"
        validation_hash = self._write_json(validation_path, report.model_dump(mode="json"))
        semantic_report = report.model_copy(update={"pydantic_valid": report.pydantic_valid, "semantic_valid": report.semantic_valid})
        semantic_path = attempt_root / "semantic_validation.json"
        self._write_json(semantic_path, semantic_report.model_dump(mode="json"))
        repair_path: Path | None = None
        if prompt:
            repair_path = attempt_root / "repair_prompt.md"
            repair_path.write_text(prompt, encoding="utf-8")
        for issue in report.errors:
            self.registry.save_specification_ambiguity(run_id, strategy_id, kind=issue.error_code, field_path=issue.field_path,
                message=issue.explanation, blocking=issue.error_code == "BLOCKING_AMBIGUITY")
        self.registry.save_specification_attempt(run_id, strategy_id, attempt, status=status, draft_path=str(draft_path.resolve()),
            validation_path=str(validation_path.resolve()), semantic_validation_path=str(semantic_path.resolve()),
            repair_prompt_path=str(repair_path.resolve()) if repair_path else None, codex_invocation_path=str(invocation_path.resolve()),
            draft_hash=hashlib.sha256(draft_path.read_bytes()).hexdigest(), validation_hash=validation_hash,
            error_summary="; ".join(f"{item.error_code}: {item.explanation}" for item in report.errors)[-4000:])

    def _repair_prompt(self, workflow: WorkflowInput, raw_output: str, report: SpecificationValidationReport) -> str:
        errors = "\n".join(f"- {item.error_code} | {item.field_path} | received={item.received_value!r} | expected={item.expected_constraint} | {item.explanation} | repair={item.repair_hint}" for item in report.errors)
        return f"""Repair exactly one strategy specification payload.
Original confirmed intake:
{workflow.natural_language_description}
Requested markets: {workflow.requested_markets}
Requested timeframes: {workflow.requested_timeframes}
Notes: {workflow.optional_notes or '(none)'}

Invalid draft (do not preserve invented rules):
{raw_output[:24000]}

Structured validation errors:
{errors}

Return exactly one YAML or JSON mapping matching StrategySpec. Do not emit prose,
multiple candidates, Markdown commentary, code, backtests, optimization, or
invented defaults. Preserve confirmed user intent. Preserve unresolved material
ambiguity instead of guessing; such ambiguity must stop before approval.
For RandomOpenTest, use strategy_family "f2_random_open_test", include
equity_fraction: 0.05 and initial_cash: 10000 in baseline_parameters, describe
allocation from current equity rather than risk-per-trade sizing, and retain
the repository-compatible 1-hour same-bar exit.
"""

    def generate_spec(self, workflow: WorkflowInput) -> GeneratedStrategySpec:
        strategy_id = self._strategy_id(workflow.strategy_name)
        version = "phase-b-1"
        run_id = self._intake_run_id(workflow, strategy_id)
        root = self._specification_root(workflow, strategy_id, run_id)
        intake_path = root / "intake" / "natural_language.json"
        self._write_json(intake_path, workflow.model_dump(mode="json"))
        provenance = self._provenance(workflow)
        attempts = self.registry.specification_attempts(run_id)
        for existing in attempts:
            if existing["status"] in {"VALID", "BLOCKED"}:
                canonical = root / "canonical" / "specification.yaml"
                if canonical.exists():
                    spec = self._load_spec(canonical)
                    return self._generated_metadata(spec, canonical, workflow, provenance=provenance, attempt=int(existing["attempt"]), root=root)
        last_raw = ""
        last_report: SpecificationValidationReport | None = None
        max_attempts = min(workflow.max_generation_attempts, 1 + workflow.max_repair_attempts)
        for attempt in range(1, max_attempts + 1):
            existing = next((item for item in attempts if item["attempt"] == attempt), None)
            if existing and existing["status"] in {"INVALID", "CODEX_FAILED"}:
                draft = Path(existing["draft_path"])
                last_raw = draft.read_text(encoding="utf-8") if draft.exists() else ""
                validation_path = Path(existing["validation_path"]) if existing.get("validation_path") else None
                if validation_path and validation_path.exists():
                    last_report = SpecificationValidationReport.model_validate(json.loads(validation_path.read_text(encoding="utf-8")))
                continue
            if attempt == 1:
                prompt = build_spec_agent_prompt(strategy_id, workflow.natural_language_description, workflow.requested_markets, workflow.requested_timeframes, workflow.optional_notes)
            else:
                if last_report is None:
                    break
                prompt = self._repair_prompt(workflow, last_raw, last_report)
            invocation: dict[str, Any]
            external_candidate = False
            external_job = self.registry.get_latest_specification_job(run_id)
            if external_job and int(external_job["attempt"]) == attempt:
                from ..specification_executor.executor import ExternalSpecificationExecutor
                completion_path = Path(external_job["job_path"]) / "result" / "completion.json"
                if not completion_path.exists():
                    raise self.registry_external_required(run_id, external_job["job_id"])
                completion, _, _ = ExternalSpecificationExecutor(self.registry_path).load_completion(run_id, external_job["job_id"])
                if completion.status in {SpecificationCompletionStatus.SUCCEEDED, SpecificationCompletionStatus.FAILED_OUTPUT_EXTRACTION}:
                    if not completion.raw_output_path:
                        raise self.registry_external_required(run_id, external_job["job_id"])
                    raw_output = Path(completion.raw_output_path).read_text(encoding="utf-8")
                    invocation = {"external_job_id": external_job["job_id"], "status": completion.status.value, "completion_path": str(Path(external_job["job_path"]) / "result" / "completion.json")}
                    external_candidate = True
                else:
                    raise self.registry_external_required(run_id, external_job["job_id"])
            elif workflow.dry_run:
                spec = self._dry_spec(workflow, strategy_id, version, [*self._ambiguities(workflow), *workflow.ambiguities])
                raw_output = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
                invocation = {"executed": False, "mode": "dry_run", "attempt": attempt, "source": "deterministic dry-run generator"}
            else:
                result = self.codex.run(prompt, workflow.repository_root, sandbox="read-only", dry_run=False)
                invocation = result.model_dump(mode="json")
                raw_output = result.stdout or result.stderr or ""
                if not result.success:
                    if is_restricted_execution_failure(result):
                        jobs = SpecificationJobService(self.registry_path)
                        created = jobs.create(workflow, strategy_id=strategy_id, strategy_version=version, run_id=run_id, attempt=attempt, job_type=SpecificationJobType.GENERATE_SPECIFICATION if attempt == 1 else SpecificationJobType.REPAIR_SPECIFICATION, prompt=prompt,
                                              prior_invalid_draft_path=str(root / "attempts" / f"attempt-{attempt - 1:03d}" / "draft.yaml") if attempt > 1 else None,
                                              validation_report_path=str(root / "attempts" / f"attempt-{attempt - 1:03d}" / "validation.json") if attempt > 1 else None,
                                              semantic_validation_path=str(root / "attempts" / f"attempt-{attempt - 1:03d}" / "semantic_validation.json") if attempt > 1 else None)
                        raise jobs.require_external(created)
                    report = SpecificationValidationReport(valid=False, pydantic_valid=False, semantic_valid=False, errors=[SpecificationValidationIssue(error_code="CODEX_EXECUTION_FAILURE", field_path="codex", received_value=result.exit_code, expected_constraint="successful Codex invocation", explanation=result.stderr or result.stdout or "Codex invocation failed", repair_hint="Retry with the structured output contract and preserve the original intake." )])
                    self._persist_attempt(root, run_id, strategy_id, attempt, raw_output=raw_output, report=report, provenance=provenance, invocation=invocation, prompt=prompt if attempt > 1 else None, status="CODEX_FAILED")
                    last_raw, last_report = raw_output, report
                    attempts = self.registry.specification_attempts(run_id)
                    continue
            try:
                raw = self._extract_structured_payload(raw_output)
                spec, report, provenance, normalized = self._validate_payload(raw, workflow, provenance)
            except (ValueError, yaml.YAMLError) as exc:
                report = SpecificationValidationReport(valid=False, pydantic_valid=False, semantic_valid=False, errors=[SpecificationValidationIssue(error_code="STRUCTURED_OUTPUT_INVALID", field_path="payload", received_value=raw_output[:4000], expected_constraint="exactly one YAML/JSON mapping", explanation=str(exc), repair_hint="Return exactly one complete structured payload with no prose.")])
                spec, normalized = None, {}
            approval_blocked_only = bool(spec) and bool(report.errors) and all(item.error_code == "BLOCKING_AMBIGUITY" for item in report.errors)
            self._persist_attempt(root, run_id, strategy_id, attempt, raw_output=raw_output, report=report, provenance=provenance, invocation=invocation, prompt=prompt if attempt > 1 else None, status="VALID" if spec and report.valid else "BLOCKED" if approval_blocked_only else "INVALID")
            last_raw, last_report = raw_output, report
            if spec and (report.valid or approval_blocked_only):
                canonical = root / "canonical" / "specification.yaml"
                save_strategy_spec(spec, str(canonical))
                manifest = {"specification_hash": spec.specification_hash, "normalized_payload_hash": report.normalized_payload_hash, "provenance": provenance.model_dump(mode="json"), "source_intake": str(intake_path.resolve())}
                self._write_json(root / "canonical" / "hash_manifest.json", manifest)
                approval_path = self._approval_path(workflow, strategy_id, version, run_id)
                save_strategy_spec(spec, str(approval_path))
                # A prior interrupted/failed attempt for this deterministic run
                # is resolved by the validated canonical artifact.  Keeping the
                # failure row would incorrectly suppress approval on resume.
                self.registry.clear_specification_failure(run_id)
                return self._generated_metadata(spec, approval_path, workflow, provenance=provenance, attempt=attempt, root=root)
            if external_candidate:
                if attempt >= max_attempts:
                    break
                jobs = SpecificationJobService(self.registry_path)
                created = jobs.create(workflow, strategy_id=strategy_id, strategy_version=version, run_id=run_id, attempt=attempt + 1, job_type=SpecificationJobType.REPAIR_SPECIFICATION,
                                      prompt=self._repair_prompt(workflow, raw_output, report),
                                      prior_invalid_draft_path=str(root / "attempts" / f"attempt-{attempt:03d}" / "draft.yaml"),
                                      validation_report_path=str(root / "attempts" / f"attempt-{attempt:03d}" / "validation.json"),
                                      semantic_validation_path=str(root / "attempts" / f"attempt-{attempt:03d}" / "semantic_validation.json"))
                raise jobs.require_external(created)
            attempts = self.registry.specification_attempts(run_id)
        failure = {"classification": "SPECIFICATION_GENERATION_FAILURE", "strategy_id": strategy_id, "run_id": run_id,
                   "attempts": len(self.registry.specification_attempts(run_id)), "repair_attempts": max(0, len(self.registry.specification_attempts(run_id)) - 1),
                   "final_reason": "; ".join(item["error_summary"] for item in self.registry.specification_attempts(run_id)[-1:]) or "no valid specification produced"}
        failure_path = root / "failure" / "final_failure.json"
        self._write_json(failure_path, failure)
        self.registry.save_specification_failure(run_id, strategy_id, failure, failure["final_reason"])
        raise RuntimeError(f"SPECIFICATION_GENERATION_FAILURE: {failure['final_reason']}")

    def registry_external_required(self, run_id: str, job_id: str) -> ExternalSpecificationRequired:
        return SpecificationJobService(self.registry_path).require_existing(run_id, job_id)

    @staticmethod
    def _load_spec(path: Path) -> StrategySpec:
        return StrategySpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def specification_status(self, run_id: str) -> dict[str, Any]:
        attempts = self.registry.specification_attempts(run_id)
        latest = attempts[-1] if attempts else None
        failure = self.registry.specification_failure(run_id)
        approval_available = bool(latest and latest["status"] == "VALID" and not failure)
        job = self.registry.get_latest_specification_job(run_id)
        return {"run_id": run_id, "attempt_count": len(attempts), "candidate_attempt_count": len(attempts), "repair_attempt_count": max(0, len(attempts) - 1),
                "latest_attempt": latest, "latest_validation_outcome": latest["status"] if latest else "NOT_STARTED",
                "latest_schema_validation_outcome": "VALID" if latest and latest.get("validation_path") and Path(latest["validation_path"]).is_file() and not latest.get("error_summary") else ("INVALID" if latest else "NOT_STARTED"),
                "latest_semantic_validation_outcome": "VALID" if latest and latest.get("semantic_validation_path") and Path(latest["semantic_validation_path"]).is_file() and not latest.get("error_summary") else ("INVALID" if latest else "NOT_STARTED"),
                "blocking_ambiguities": self.registry.specification_ambiguities(run_id), "approval_available": approval_available,
                "failure": failure, "current_specification_job": job, "specification_jobs": self.registry.specification_jobs(run_id),
                "repair_budget_remaining": max(0, 2 - max(0, len(attempts) - 1)),
                "next_command": f"py -m research_pipeline specification-executor run {run_id}" if job and job["status"].startswith("WAITING_EXTERNAL") else None}

    def specification_attempts(self, run_id: str) -> list[dict[str, Any]]:
        return self.registry.specification_attempts(run_id)

    def specification_errors(self, run_id: str) -> list[dict[str, Any]]:
        attempts = self.registry.specification_attempts(run_id)
        errors: list[dict[str, Any]] = []
        for item in attempts:
            path = item.get("validation_path")
            if path and Path(path).exists():
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                errors.extend({"attempt": item["attempt"], **error} for error in payload.get("errors", []))
        return errors

    def specification_latest(self, run_id: str) -> dict[str, Any] | None:
        attempts = self.registry.specification_attempts(run_id)
        return attempts[-1] if attempts else None

    def _dry_spec(self, workflow: WorkflowInput, strategy_id: str, version: str, ambiguities: list[str]) -> StrategySpec:
        if strategy_id == "RandomOpenTest":
            raw: dict[str, Any] = {
                "strategy_id": strategy_id, "version": version, "name": "RandomOpenTest",
                "description": "Deterministic one-hour repository-compatible pipeline integration test.",
                "hypothesis": "This intentionally meaningless strategy is not expected to have an edge.",
                "strategy_family": "f2_random_open_test", "markets": ["SPY"], "timeframes": ["1h"],
                "long_rules": ["At the first 09:30 America/New_York hourly bar, deterministic daily coin flip may select LONG."],
                "short_rules": ["At the first 09:30 America/New_York hourly bar, deterministic daily coin flip may select SHORT."],
                "entry_logic": "Resolve timestamps in America/New_York; enter at the first regular-session 09:30 hourly bar open. Seed the daily coin flip from strategy name plus trading date.",
                "initial_stop_logic": "No stop.", "exit_logic": "Exit at the close of the same hourly bar; no exact 10-minute exit is claimed.",
                "session_assumptions": ["TECHNICAL_TRANSLATION: US cash-session open is 09:30 America/New_York and is resolved with IANA timezone rules.", "Existing SPY_1h parquet is the repository-compatible proxy dataset."],
                "baseline_parameters": {"initial_cash": 10000, "equity_fraction": 0.05, "session_timezone": "America/New_York", "session_open_local": "09:30", "holding_period_hours": 1, "test_start_date": "2025-01-01", "test_end_date": "2026-01-01", "fee_rate": 0.0, "slippage_rate": 0.0},
                "parameter_families": [{"name": "execution_variant", "description": "Fixed one-hour repository-compatible test variant.", "baseline_value": "1-hour repository-compatible test variant", "value_type": "string", "allowed_min": None, "allowed_max": None, "allowed_values": ["1-hour repository-compatible test variant"], "optimization_order": 0, "maximum_rounds": 0, "mutable": False, "hypothesis_relevance": "Fixed integration-test rule; never optimized."}],
                "invariants": ["Exactly one trade per valid US trading day.", "Direction is deterministic from strategy name plus trading date.", "Allocate 5% of current equity per trade.", "No stop, target, filter, or optimization.", "This is a 1-hour repository-compatible variant, not an exact 10-minute strategy."],
                "required_data": ["Existing data/v11_5_proxy_raw/SPY_1h.parquet"],
                "known_limitations": ["SPY is an explicit repository proxy; no new dataset is introduced.", "Phase D has no existing futures mapping for SPY and is expected to be unsupported.", "Integration test only; not expected to be profitable.", *ambiguities],
                "status": ApprovalStatus.DRAFT, "created_at": datetime.now(timezone.utc), "approved_at": None,
            }
            return self._validated_with_hash(raw)
        raw: dict[str, Any] = {
            "strategy_id": strategy_id, "version": version, "name": workflow.strategy_name,
            "description": workflow.natural_language_description, "hypothesis": workflow.natural_language_description,
            "strategy_family": "phase_b_fictional", "markets": workflow.requested_markets,
            "timeframes": workflow.requested_timeframes, "long_rules": ["Use the described long condition."],
            "short_rules": ["Use the described short condition."], "entry_logic": "Use the described entry condition.",
            "initial_stop_logic": "Use a fixed initial stop described by the specification.",
            "exit_logic": "Use the described exit condition.", "session_assumptions": ["Chronological timestamps."],
            "baseline_parameters": {"fictional_baseline": 1},
            "parameter_families": [{"name": "fictional_baseline", "description": "Immutable dry-run baseline.", "baseline_value": 1,
                "value_type": "integer", "allowed_min": 1, "allowed_max": 1, "allowed_values": [1],
                "optimization_order": 0, "maximum_rounds": 0, "mutable": False, "hypothesis_relevance": "Fixture only."}],
            "invariants": ["This fictional dry run must not alter existing trading behavior."],
            "required_data": ["OHLCV candles"], "known_limitations": [*ambiguities, "Phase B dry-run fixture; no trading research."],
            "status": ApprovalStatus.DRAFT, "created_at": datetime.now(timezone.utc), "approved_at": None,
        }
        return self._validated_with_hash(raw)

    def _parse_codex_spec(self, output: str, strategy_id: str, version: str, workflow: WorkflowInput, ambiguities: list[str]) -> StrategySpec:
        raw = self._extract_structured_payload(output)
        if raw.get("strategy_id") not in {None, strategy_id}:
            raise ValueError(f"Codex returned strategy_id {raw.get('strategy_id')!r}; expected {strategy_id!r}")
        raw.setdefault("strategy_id", strategy_id); raw.setdefault("version", version)
        raw.setdefault("status", ApprovalStatus.DRAFT); raw.setdefault("approved_at", None)
        raw.setdefault("created_at", datetime.now(timezone.utc))
        raw.setdefault("known_limitations", [])
        raw["known_limitations"] = [*raw["known_limitations"], *ambiguities]
        return self._validated_with_hash(raw)

    @staticmethod
    def _validated_with_hash(
        raw: dict[str, Any],
    ) -> StrategySpec:
        data = dict(raw)

        # StrategySpec benötigt das Feld bereits bei der ersten Validierung.
        # Der Wert darf hier vorläufig sein, da die Hashprüfung per Context
        # übersprungen wird.
        data["specification_hash"] = str(
            data.get("specification_hash") or "pending"
        )

        # Erst das komplette Modell validieren und alle verschachtelten Werte
        # in ihre kanonischen Pydantic-Typen überführen.
        candidate = StrategySpec.model_validate(
            data,
            context={
                "skip_specification_hash_validation": True,
            },
        )

        # Den Hash aus dem vollständig validierten Modell berechnen.
        expected_hash = calculate_specification_hash(candidate)

        # Nicht candidate.specification_hash direkt ändern:
        # validate_assignment=True würde dabei sofort erneut validieren.
        validated_data = candidate.model_dump(mode="python")
        validated_data["specification_hash"] = expected_hash

        # Abschließende normale Validierung einschließlich Hashprüfung.
        return StrategySpec.model_validate(validated_data)

    @staticmethod
    def _ambiguities(workflow: WorkflowInput) -> list[str]:
        text = f"{workflow.natural_language_description} {workflow.optional_notes or ''}".lower()
        if any(word in text for word in ("ambiguous", "unclear", "maybe", "not sure", "unspecified")):
            return ["The strategy description contains unresolved material ambiguity."]
        return []

    def _generated_metadata(self, spec: StrategySpec, path: Path, workflow: WorkflowInput, *, provenance: SpecificationProvenance | None = None, attempt: int = 1, root: Path | None = None) -> GeneratedStrategySpec:
        ambiguities = [*self._ambiguities(workflow), *workflow.ambiguities]
        summary = json.dumps({"hypothesis": spec.hypothesis, "markets": spec.markets, "timeframes": spec.timeframes,
            "long_rules": spec.long_rules, "short_rules": spec.short_rules, "entry": spec.entry_logic,
            "initial_stop": spec.initial_stop_logic, "exits": spec.exit_logic, "baseline_parameters": spec.baseline_parameters,
            "mutable_parameter_families": [item.name for item in spec.parameter_families if item.mutable],
            "invariants": spec.invariants, "assumptions": spec.session_assumptions, "ambiguities": ambiguities,
            "provenance": (provenance or SpecificationProvenance()).model_dump(mode="json"),
            "implementation_variant": "1-hour repository-compatible test variant" if spec.strategy_family == "f2_random_open_test" else None,
            "specification_path": str(path), "specification_hash": spec.specification_hash}, indent=2, sort_keys=True)
        return GeneratedStrategySpec(strategy_id=spec.strategy_id, version=spec.version, specification_path=str(path), specification_hash=spec.specification_hash,
            assumptions=list(spec.session_assumptions), ambiguities=ambiguities, fields_requiring_confirmation=["entry_logic"] if ambiguities else [], manual_review_required=bool(ambiguities), approval_summary=summary,
            provenance=provenance or SpecificationProvenance(), attempt=attempt,
            validation_report_path=str((root / "attempts" / f"attempt-{attempt:03d}" / "validation.json").resolve()) if root else None,
            semantic_validation_report_path=str((root / "attempts" / f"attempt-{attempt:03d}" / "semantic_validation.json").resolve()) if root else None)

    def validate_spec(self, generated: GeneratedStrategySpec) -> SpecificationValidationResult:
        structured: list[SpecificationValidationIssue] = []
        semantic_report: SpecificationValidationReport | None = None
        try:
            raw = yaml.safe_load(Path(generated.specification_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("specification YAML must contain one mapping")
            normalized, semantic_report, provenance = semantic_validate(raw, provenance=generated.provenance)
            candidate = StrategySpec.model_validate(
                {**normalized, "specification_hash": "pending"},
                context={"skip_specification_hash_validation": True},
            )
            expected_hash = calculate_specification_hash(candidate)
            validated_data = candidate.model_dump(mode="python")
            validated_data["specification_hash"] = expected_hash
            spec = StrategySpec.model_validate(validated_data)
            if raw.get("specification_hash") != expected_hash:
                structured.append(SpecificationValidationIssue(error_code="CANONICAL_NORMALIZATION_MISMATCH", field_path="specification_hash", received_value=raw.get("specification_hash"),
                    expected_constraint="hash must be calculated after canonical normalization", explanation="The persisted specification hash does not represent the normalized payload.", repair_hint="Regenerate the canonical specification through the intake validator."))
            structured.extend(semantic_report.errors)
            if spec.specification_hash != generated.specification_hash:
                structured.append(SpecificationValidationIssue(error_code="METADATA_HASH_MISMATCH", field_path="specification_hash", received_value=spec.specification_hash,
                    expected_constraint="metadata hash equals persisted specification hash", explanation="Generated metadata and the draft disagree.", repair_hint="Persist metadata and specification atomically from one validated payload."))
            valid = not structured and not generated.manual_review_required
            errors = [f"{item.error_code}: {item.explanation}" for item in structured]
            semantic_report = semantic_report.model_copy(update={"valid": not bool(semantic_report.errors), "pydantic_valid": True, "semantic_valid": not bool(semantic_report.errors)})
            return SpecificationValidationResult(valid=valid, strategy_id=spec.strategy_id, version=spec.version, specification_path=generated.specification_path, specification_hash=spec.specification_hash, errors=errors, manual_review_required=generated.manual_review_required, structured_errors=structured, semantic_report=semantic_report, canonical_path=generated.specification_path, approval_ready=valid, provenance=provenance)
        except (OSError, ValidationError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                structured.extend(self._pydantic_issue(exc))
            else:
                structured.append(SpecificationValidationIssue(error_code="SPECIFICATION_FILE_INVALID", field_path="specification", received_value=str(exc), expected_constraint="readable canonical StrategySpec YAML", explanation=str(exc), repair_hint="Regenerate one complete structured specification."))
            return SpecificationValidationResult(valid=False, strategy_id=generated.strategy_id, version=generated.version, specification_path=generated.specification_path, specification_hash=generated.specification_hash, errors=[f"{item.error_code}: {item.explanation}" for item in structured], manual_review_required=generated.manual_review_required, structured_errors=structured, semantic_report=semantic_report, approval_ready=False, provenance=generated.provenance)

    def register_generated(self, validation: SpecificationValidationResult) -> RegistrationResult:
        if not validation.valid or not validation.approval_ready or validation.manual_review_required:
            raise ValueError("cannot register a specification that is not approval-ready")
        spec = StrategySpec.model_validate(yaml.safe_load(Path(validation.specification_path).read_text(encoding="utf-8")))
        try:
            existing = self.registry.get_strategy(spec.strategy_id, spec.version)
        except RegistryError:
            existing = None
        if existing is not None:
            if existing["specification_hash"] != spec.specification_hash:
                raise RegistryError("existing strategy version has a different specification hash")
            return RegistrationResult(registered=True, idempotent_reuse=True, strategy_id=spec.strategy_id, version=spec.version, current_phase=PipelineState(existing["current_phase"]), specification_hash=spec.specification_hash)
        self.controller.register_strategy(spec, validation.specification_path)
        self.controller.submit_specification(spec.strategy_id)
        current = self.registry.get_strategy(spec.strategy_id, spec.version)
        return RegistrationResult(registered=True, idempotent_reuse=False, strategy_id=spec.strategy_id, version=spec.version, current_phase=PipelineState(current["current_phase"]), specification_hash=spec.specification_hash)

    def approve(self, strategy_id: str, decision: str, note: str | None = None) -> ApprovalResult:
        strategy = self.registry.get_strategy(strategy_id)
        if decision == "REJECT":
            if strategy["current_phase"] == PipelineState.WAITING_FOR_SPEC_APPROVAL.value:
                self.controller.transition(strategy_id, PipelineState.REJECTED, note or "specification rejected")
            current = self.registry.get_strategy(strategy_id)
            return ApprovalResult(decision=decision, approved=False, note=note, strategy_id=strategy_id, version=current["version"], current_phase=PipelineState(current["current_phase"]), immutable_verified=False)
        if decision != "APPROVE":
            raise ValueError("approval decision must be APPROVE or REJECT")
        if strategy["current_phase"] == PipelineState.WAITING_FOR_SPEC_APPROVAL.value:
            self.controller.approve_specification(strategy_id)
        current = self.registry.get_strategy(strategy_id)
        spec = self.registry.get_specification(strategy_id)
        try:
            immutable_verified = self.registry.approve_specification(strategy_id, current["version"], spec) is None
        except ImmutableSpecificationError:
            immutable_verified = True
        return ApprovalResult(decision=decision, approved=True, note=note, strategy_id=strategy_id, version=current["version"], current_phase=PipelineState(current["current_phase"]), immutable_verified=immutable_verified)

    def implementation_plan(self, strategy_id: str, repository_root: str, *, dry_run: bool = True, worktree_suffix: str | None = None) -> ImplementationPlan:
        current = self.registry.get_strategy(strategy_id)
        if current["current_phase"] != PipelineState.IMPLEMENTATION.value:
            raise InvalidTransitionError(
                f"implementation planning requires IMPLEMENTATION, got {current['current_phase']}"
            )
        spec = self.registry.get_specification(strategy_id)
        manager = WorktreeManager(repository_root)
        plan = manager.plan(spec.strategy_id, spec.version, dry_run=dry_run, worktree_suffix=worktree_suffix)
        return plan.model_copy(update={"invariants": spec.invariants, "required_tests":[
            ["python", "-m", "pytest", "-q", "tests/research_pipeline"],
            ["python", "-m", "pytest", "-q", "tests/test_no_lookahead.py", "tests/test_replay.py"],
            ["python", "-m", "pytest", "-q"],
        ], "max_repair_attempts": self.registry.get_budget(strategy_id)["limits"]["max_codex_repair_attempts"]})

    def record_codex(self, strategy_id: str, result: CodexExecutionResult, *, task_name: str = "implementation") -> CodexExecutionResult:
        self._record_experiment(strategy_id, f"phase-b-{strategy_id}-{task_name}", "COMPLETED" if result.success else "FAILED", {"result": result.model_dump(mode="json")}, result.stdout[-4000:])
        return result

    def execute_codex(self, strategy_id: str, repository_root: str, plan: ImplementationPlan, prompt: str, *, dry_run: bool, task_name: str) -> CodexExecutionResult:
        manager = WorktreeManager(repository_root)
        manager.create(plan, dry_run=dry_run)
        result = self.codex.run(prompt, plan.worktree_path, sandbox="workspace-write", dry_run=dry_run)
        if not dry_run and result.success:
            changed = manager.changed_files(plan)
            from ..adapters.compatibility import verify_implementation_scope

            result = result.model_copy(update={"files_changed": changed, "resulting_commit": manager.current_commit(plan.worktree_path)})
            try:
                verify_implementation_scope(changed, plan.allowed_files)
            except Exception as exc:
                result = result.model_copy(update={"success": False, "error_type": getattr(exc, "code", "IMPLEMENTATION_SCOPE_VIOLATION"), "stderr": str(exc)})
        return self.record_codex(strategy_id, result, task_name=task_name)

    def build_implementation_prompt(self, strategy_id: str, plan: ImplementationPlan) -> str:
        return build_implementation_prompt(self.registry.get_specification(strategy_id), plan.allowed_files, plan.required_tests, plan.max_repair_attempts)

    def run_tests(self, repository_root: str, *, dry_run: bool = True, worktree_path: str | None = None) -> TestResult:
        cwd = worktree_path or repository_root
        return self.tests.run(cwd, ["python", "-m", "pytest", "-q"], dry_run=dry_run, report_path=Path(repository_root) / "research_registry" / "phase_b" / "test-results.txt")

    def run_required_tests(self, repository_root: str, required_tests: list[list[str]], *, dry_run: bool = True, worktree_path: str | None = None) -> TestResult:
        """Run every deterministic technical suite and aggregate process evidence."""
        cwd = worktree_path or repository_root
        results = [
            self.tests.run(cwd, command, dry_run=dry_run, report_path=Path(repository_root) / "research_registry" / "phase_b" / f"test-results-{index}.txt")
            for index, command in enumerate(required_tests)
        ]
        if not results:
            return TestResult(passed=False, command=["no-test-suites"], exit_code=None, parsed_passed=0, parsed_failed=1,
                parsed_skipped=0, duration_ms=0, report_path=None, failure_summary="no required test suites configured", executed=False)
        failures = [result.failure_summary for result in results if result.failure_summary]
        passed = all(result.passed for result in results)
        return TestResult(passed=passed, command=["required-test-suites"], exit_code=0 if passed else 1,
            parsed_passed=sum(result.parsed_passed for result in results), parsed_failed=sum(result.parsed_failed for result in results),
            parsed_skipped=sum(result.parsed_skipped for result in results), duration_ms=sum(result.duration_ms for result in results),
            report_path=str(Path(repository_root) / "research_registry" / "phase_b" / "test-results-aggregate.txt"),
            failure_summary="\n".join(failures)[-4000:], executed=any(result.executed for result in results))

    def technical_verification(self, strategy_id: str, tests: TestResult, *, implementation_executed: bool, repair_attempts: int, worktree_path: str | None = None) -> FinalPhaseBSummary:
        current = self.registry.get_strategy(strategy_id)
        if current["current_phase"] == PipelineState.IMPLEMENTATION.value:
            next_state = PipelineState.IMPLEMENTATION_VERIFICATION if tests.passed else PipelineState.TECHNICAL_FAILURE
            self.controller.transition(strategy_id, next_state, "Phase B technical verification")
        current = self.registry.get_strategy(strategy_id)
        return FinalPhaseBSummary(strategy_id=strategy_id, version=current["version"], final_state=PipelineState(current["current_phase"]), approval="APPROVED", manual_review_required=False,
            implementation_executed=implementation_executed, tests_passed=tests.passed, repair_attempts=repair_attempts, registry_reconciled=True,
            worktree_path=worktree_path, outputs=[str(self.registry_path)], limitation="Phase B stops at implementation verification and does not run baseline research or optimization.")

    def final_status(self, strategy_id: str, tests: TestResult, *, implementation_executed: bool, repair_attempts: int, worktree_path: str | None = None) -> FinalPhaseBSummary:
        """Compatibility API; the technical-verification node owns the transition."""
        return self.technical_verification(strategy_id, tests, implementation_executed=implementation_executed,
                                            repair_attempts=repair_attempts, worktree_path=worktree_path)

    def _record_experiment(self, strategy_id: str, experiment_id: str, status: str, values: dict, report: str) -> None:
        history = self.registry.history(strategy_id)["experiments"]
        if any(row["experiment_id"] == experiment_id for row in history):
            return
        self.controller.store_experiment(strategy_id, experiment_id=experiment_id, phase=self.registry.get_strategy(strategy_id)["current_phase"], parameter_values=values, status=status, report_paths=[report] if report else [])
