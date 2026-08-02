from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..controller.pipeline_controller import PipelineController
from ..enums import ApprovalStatus, PipelineState
from ..errors import ImmutableSpecificationError, InvalidTransitionError, RegistryError
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.strategy_spec import ParameterFamily, StrategySpec, calculate_specification_hash, save_strategy_spec
from ..runners.codex_runner import CodexRunner
from ..runners.test_runner import DeterministicTestRunner
from ..runners.worktree_manager import WorktreeManager
from .models import (
    ApprovalResult, CodexExecutionResult, FinalPhaseBSummary, GeneratedStrategySpec,
    ImplementationPlan, RegistrationResult, SpecificationValidationResult, TestResult,
    WorkflowInput,
)
from .prompt_builder import build_implementation_prompt, build_spec_agent_prompt


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

    def generate_spec(self, workflow: WorkflowInput) -> GeneratedStrategySpec:
        strategy_id = self._strategy_id(workflow.strategy_name)
        version = "phase-b-1"
        draft_dir = Path(workflow.repository_root).resolve() / "research_registry" / "spec_drafts"
        draft_dir.mkdir(parents=True, exist_ok=True)
        path = draft_dir / f"{strategy_id}_v{version}.yaml"
        if path.exists():
            existing = StrategySpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            return self._generated_metadata(existing, path, workflow)
        ambiguities = self._ambiguities(workflow)
        if workflow.dry_run:
            spec = self._dry_spec(workflow, strategy_id, version, ambiguities)
        else:
            prompt = build_spec_agent_prompt(strategy_id, workflow.natural_language_description, workflow.requested_markets, workflow.requested_timeframes, workflow.optional_notes)
            result = self.codex.run(prompt, workflow.repository_root, sandbox="read-only", dry_run=False)
            if not result.success:
                raise RuntimeError(f"Codex spec generation failed: {result.stderr or result.stdout}")
            spec = self._parse_codex_spec(result.stdout, strategy_id, version, workflow, ambiguities)
        save_strategy_spec(spec, str(path))
        return self._generated_metadata(spec, path, workflow)

    def _dry_spec(self, workflow: WorkflowInput, strategy_id: str, version: str, ambiguities: list[str]) -> StrategySpec:
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
        candidate = output.strip()
        fenced = re.search(r"```(?:yaml|yml)?\s*(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        start = candidate.find("strategy_id:")
        if start >= 0:
            candidate = candidate[start:]
        raw = yaml.safe_load(candidate)
        if not isinstance(raw, dict):
            raise ValueError("Codex did not return a YAML mapping")
        raw.setdefault("strategy_id", strategy_id); raw.setdefault("version", version)
        raw.setdefault("status", ApprovalStatus.DRAFT); raw.setdefault("approved_at", None)
        raw.setdefault("created_at", datetime.now(timezone.utc))
        raw.setdefault("known_limitations", [])
        raw["known_limitations"] = [*raw["known_limitations"], *ambiguities]
        return self._validated_with_hash(raw)

    @staticmethod
    def _validated_with_hash(raw: dict[str, Any]) -> StrategySpec:
        data = dict(raw)
        data["parameter_families"] = [ParameterFamily.model_validate(item) for item in data["parameter_families"]]
        data["status"] = ApprovalStatus(data.get("status", ApprovalStatus.DRAFT))
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        data["specification_hash"] = "pending"
        candidate = StrategySpec.model_construct(**data)
        data["specification_hash"] = calculate_specification_hash(candidate)
        return StrategySpec.model_validate(data)

    @staticmethod
    def _ambiguities(workflow: WorkflowInput) -> list[str]:
        text = f"{workflow.natural_language_description} {workflow.optional_notes or ''}".lower()
        if any(word in text for word in ("ambiguous", "unclear", "maybe", "not sure", "unspecified")):
            return ["The strategy description contains unresolved material ambiguity."]
        return []

    def _generated_metadata(self, spec: StrategySpec, path: Path, workflow: WorkflowInput) -> GeneratedStrategySpec:
        ambiguities = self._ambiguities(workflow)
        summary = json.dumps({"hypothesis": spec.hypothesis, "markets": spec.markets, "timeframes": spec.timeframes,
            "long_rules": spec.long_rules, "short_rules": spec.short_rules, "entry": spec.entry_logic,
            "initial_stop": spec.initial_stop_logic, "exits": spec.exit_logic, "baseline_parameters": spec.baseline_parameters,
            "mutable_parameter_families": [item.name for item in spec.parameter_families if item.mutable],
            "invariants": spec.invariants, "assumptions": spec.session_assumptions, "ambiguities": ambiguities,
            "specification_path": str(path), "specification_hash": spec.specification_hash}, indent=2, sort_keys=True)
        return GeneratedStrategySpec(strategy_id=spec.strategy_id, version=spec.version, specification_path=str(path), specification_hash=spec.specification_hash,
            assumptions=list(spec.session_assumptions), ambiguities=ambiguities, fields_requiring_confirmation=["entry_logic"] if ambiguities else [], manual_review_required=bool(ambiguities), approval_summary=summary)

    def validate_spec(self, generated: GeneratedStrategySpec) -> SpecificationValidationResult:
        try:
            spec = StrategySpec.model_validate(yaml.safe_load(Path(generated.specification_path).read_text(encoding="utf-8")))
            valid = spec.specification_hash == generated.specification_hash
            errors = [] if valid else ["generated metadata hash does not match YAML hash"]
            return SpecificationValidationResult(valid=valid, strategy_id=spec.strategy_id, version=spec.version, specification_path=generated.specification_path, specification_hash=spec.specification_hash, errors=errors, manual_review_required=generated.manual_review_required)
        except (OSError, ValidationError, ValueError) as exc:
            return SpecificationValidationResult(valid=False, strategy_id=generated.strategy_id, version=generated.version, specification_path=generated.specification_path, specification_hash=generated.specification_hash, errors=[str(exc)], manual_review_required=False)

    def register_generated(self, validation: SpecificationValidationResult) -> RegistrationResult:
        if not validation.valid:
            raise ValueError("cannot register invalid specification")
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

    def implementation_plan(self, strategy_id: str, repository_root: str, *, dry_run: bool = True) -> ImplementationPlan:
        current = self.registry.get_strategy(strategy_id)
        if current["current_phase"] != PipelineState.IMPLEMENTATION.value:
            raise InvalidTransitionError(
                f"implementation planning requires IMPLEMENTATION, got {current['current_phase']}"
            )
        spec = self.registry.get_specification(strategy_id)
        manager = WorktreeManager(repository_root)
        plan = manager.plan(spec.strategy_id, spec.version, dry_run=dry_run)
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
            protected = [item for item in changed if item.startswith(("src/fib_backtester/", "data/", "reports/"))]
            result = result.model_copy(update={"files_changed": changed, "resulting_commit": manager.current_commit(plan.worktree_path)})
            if protected:
                result = result.model_copy(update={"success": False, "error_type": "MATERIAL_CHANGE", "stderr": f"protected files changed: {protected}"})
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
