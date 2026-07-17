from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..enums import PipelineState
from ..errors import InvalidTransitionError, RegistryError, SpecificationValidationError
from ..phase_b.models import WorkflowInput
from ..phase_b.services import PhaseBService
from ..prop.services import PropResearchService
from ..registry.database import Database
from ..registry.repositories import Registry
from ..research.fixtures import make_phase_c_split
from ..research.services import PhaseCService
from ..verification.fixtures import make_fixture
from ..verification.services import VerificationService
from .models import (ApprovalRecord, ArtifactReference, FinalClassification, FinalReport, IntakeSpec,
    MasterRunInput, MasterRunOutcome, MasterRunStatus, MasterStatus, MasterStep, PhaseTiming)
from .utils import file_hash, safe_strategy_id, stable_hash


class MasterPipelineService:
    """Durable F1 composition layer over the already-tested phase services."""

    def __init__(self, registry_path: str | Path | None = None, repository_root: str | Path = "."):
        self.registry_path = Path(registry_path or "research_registry/research_pipeline.sqlite3")
        self.registry = Registry(Database(self.registry_path))
        self.repository_root = Path(repository_root).resolve()

    @staticmethod
    def load_intake(path: str | Path) -> IntakeSpec:
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8")) if source.suffix.lower() == ".json" else yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SpecificationValidationError("intake file must contain a mapping")
        return IntakeSpec.model_validate(raw)

    @staticmethod
    def input_model(intake_path: str | Path, repository_root: str | Path, *, registry_path: str | Path | None = None,
                    dry_run: bool = True, implementation_enabled: bool = False, research_scenario: str = "strong-stable",
                    prop_scenario: str = "profitable", portfolio_scenario: str = "complementary",
                    prop_product: str = "Alpha Futures Zero 25K") -> MasterRunInput:
        return MasterRunInput(intake_path=str(Path(intake_path).resolve()), repository_root=str(Path(repository_root).resolve()), registry_path=str(Path(registry_path).resolve()) if registry_path else None, dry_run=dry_run, implementation_enabled=implementation_enabled, research_scenario=research_scenario, prop_scenario=prop_scenario, portfolio_scenario=portfolio_scenario, prop_product=prop_product)

    def _input_hash(self, intake: IntakeSpec, options: MasterRunInput) -> str:
        return stable_hash({"intake": intake.model_dump(mode="json"), "options": options.model_dump(mode="json", exclude={"intake_path"})})

    def _root(self, strategy_id: str, run_id: str) -> Path:
        root = self.repository_root / "research_runs" / strategy_id / run_id
        for name in ("run", "specification", "implementation", "verification", "research", "prop", "portfolio", "report", "archive"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    def _write_json(self, path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return file_hash(path)

    def _record_artifact(self, run_id: str, phase: MasterStep | str, path: Path, artifact_type: str) -> ArtifactReference:
        reference = ArtifactReference(phase=str(phase), path=str(path.resolve()), artifact_type=artifact_type, sha256=file_hash(path))
        self.registry.add_master_artifact(run_id, reference.phase, reference.path, reference.sha256, reference.artifact_type)
        return reference

    def _record_phase(self, run_id: str, phase: MasterStep, result: Any, artifact_path: Path, status: str = "SUCCESS") -> dict:
        started = datetime.now(timezone.utc); began = time.monotonic()
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        digest = self._write_json(artifact_path, payload)
        ended = datetime.now(timezone.utc)
        self.registry.save_master_phase_result(run_id, phase.value, status, payload, [str(artifact_path.resolve())], digest, started.isoformat(), ended.isoformat(), int((time.monotonic() - began) * 1000))
        self.registry.add_master_journal(run_id, phase.value, "PHASE_COMPLETED" if status == "SUCCESS" else "PHASE_FAILED", {"status": status, "result_hash": digest, "artifact_path": str(artifact_path.resolve())})
        self._record_artifact(run_id, phase, artifact_path, "phase-result")
        return payload

    def _success_phase(self, run_id: str, phase: MasterStep) -> dict | None:
        record = self.registry.get_master_phase_result(run_id, phase.value)
        return record["result_json"] if record and record["status"] == "SUCCESS" else None

    def _set_step(self, run_id: str, step: MasterStep, outcome: MasterRunStatus | str, **state: Any) -> None:
        current = self.registry.get_master_run(run_id)
        resume = dict(current["resume_state_json"]); resume.update(state)
        self.registry.update_master_run(run_id, current_step=step.value, outcome=str(outcome), resume_state=resume)

    def start(self, options: MasterRunInput) -> dict:
        intake = self.load_intake(options.intake_path)
        strategy_id = safe_strategy_id(intake.strategy_name)
        input_hash = self._input_hash(intake, options)
        run_id = f"f1-{strategy_id}-{input_hash[:12]}"
        root = self._root(strategy_id, run_id)
        try:
            existing = self.registry.get_master_run(run_id)
            return self.status(run_id)
        except RegistryError:
            pass
        intake_path = root / "run" / "intake.json"
        self._write_json(intake_path, intake.model_dump(mode="json"))
        self.registry.save_master_run(run_id, strategy_id, None, input_hash, MasterStep.SPECIFICATION.value, MasterRunStatus.WAITING_FOR_APPROVAL.value, "PENDING" if not intake.ambiguities and not intake.missing_information else "MANUAL_REVIEW_REQUIRED", str(root), {"options": options.model_dump(mode="json"), "intake_path": str(intake_path), "intake": intake.model_dump(mode="json")})
        self.registry.add_master_journal(run_id, MasterStep.INTAKE.value, "INTAKE_ACCEPTED", {"input_hash": input_hash, "intake_artifact": str(intake_path.resolve()), "manual_review_required": bool(intake.ambiguities or intake.missing_information)})
        if intake.ambiguities or intake.missing_information:
            self._set_step(run_id, MasterStep.APPROVAL, MasterRunStatus.MANUAL_REVIEW_REQUIRED, manual_review_required=True)
            return self.status(run_id)
        return self._generate_and_register(run_id, options, intake, root)

    def _generate_and_register(self, run_id: str, options: MasterRunInput, intake: IntakeSpec, root: Path) -> dict:
        generation_name = intake.strategy_name
        probe = self.repository_root / "research_registry" / "spec_drafts" / f"{safe_strategy_id(generation_name)}_vphase-b-1.yaml"
        if probe.is_file():
            try:
                existing = yaml.safe_load(probe.read_text(encoding="utf-8")) or {}
                if existing.get("description") != self._description(intake) or existing.get("markets") != intake.markets or existing.get("timeframes") != intake.timeframes:
                    generation_name = f"{intake.strategy_name}-{self.registry.get_master_run(run_id)['input_hash'][:8]}"
            except (OSError, ValueError, TypeError):
                generation_name = f"{intake.strategy_name}-{self.registry.get_master_run(run_id)['input_hash'][:8]}"
        workflow = WorkflowInput(strategy_name=generation_name, natural_language_description=self._description(intake), requested_markets=intake.markets, requested_timeframes=intake.timeframes, optional_notes=intake.optional_notes, repository_root=str(self.repository_root), registry_path=str(self.registry_path), dry_run=options.dry_run, implementation_enabled=options.implementation_enabled)
        phase_b = PhaseBService(self.registry_path)
        generated = phase_b.generate_spec(workflow)
        draft_copy = root / "specification" / Path(generated.specification_path).name
        draft_copy.write_text(Path(generated.specification_path).read_text(encoding="utf-8"), encoding="utf-8")
        validation = phase_b.validate_spec(generated)
        registration = phase_b.register_generated(validation) if validation.valid else None
        payload = {"generated": generated.model_dump(mode="json"), "validation": validation.model_dump(mode="json"), "registration": registration.model_dump(mode="json") if registration else None, "draft_copy": str(draft_copy.resolve()), "manual_review_required": bool(generated.manual_review_required or validation.manual_review_required)}
        self._record_phase(run_id, MasterStep.SPECIFICATION, payload, root / "specification" / "result.json")
        self.registry.update_master_run(run_id, strategy_id=generated.strategy_id)
        current = self.registry.get_master_run(run_id)
        self.registry.update_master_run(run_id, strategy_version=generated.version, current_step=MasterStep.APPROVAL.value, outcome=MasterRunStatus.WAITING_FOR_APPROVAL.value, approval_status="MANUAL_REVIEW_REQUIRED" if payload["manual_review_required"] else "PENDING", resume_state={**current["resume_state_json"], "strategy_id": generated.strategy_id, "version": generated.version, "specification_path": generated.specification_path, "draft_copy": str(draft_copy.resolve())})
        self.registry.add_master_journal(run_id, MasterStep.APPROVAL.value, "APPROVAL_REQUIRED", {"strategy_id": generated.strategy_id, "version": generated.version, "specification_hash": generated.specification_hash, "manual_review_required": payload["manual_review_required"]})
        return self.status(run_id)

    @staticmethod
    def _description(intake: IntakeSpec) -> str:
        sections = [intake.description]
        for title, values in (("Confirmed facts", intake.confirmed_facts), ("Entry logic", intake.entry_logic), ("Exit logic", intake.exit_logic), ("Filters", intake.filters), ("Assumptions", intake.assumptions), ("Missing information", intake.missing_information)):
            if values: sections.append(f"{title}: " + "; ".join(values))
        if intake.risk_model: sections.append(f"Risk model: {intake.risk_model}")
        if intake.position_sizing: sections.append(f"Position sizing: {intake.position_sizing}")
        return "\n".join(sections)

    def approve(self, run_id: str, decision: str = "APPROVE", note: str | None = None) -> dict:
        run = self.registry.get_master_run(run_id); state = run["resume_state_json"]
        if run["approval_status"] == "MANUAL_REVIEW_REQUIRED":
            raise SpecificationValidationError("material intake ambiguity requires a clarified intake and a new run")
        if run["current_step"] not in {MasterStep.APPROVAL.value, MasterStep.SPECIFICATION.value}:
            return self.status(run_id)
        strategy_id = state.get("strategy_id") or run["strategy_id"]
        if decision == "REJECT":
            phase_b = PhaseBService(self.registry_path); result = phase_b.approve(strategy_id, "REJECT", note)
            self.registry.add_master_journal(run_id, MasterStep.APPROVAL.value, "APPROVAL_REJECTED", {"note": note, "result": result.model_dump(mode="json")})
            self.registry.update_master_run(run_id, approval_status="REJECTED")
            self._set_step(run_id, MasterStep.COMPLETED, MasterRunStatus.ABORTED, approval="REJECTED")
            return self.status(run_id)
        if decision != "APPROVE": raise ValueError("approval decision must be APPROVE or REJECT")
        phase_b = PhaseBService(self.registry_path); result = phase_b.approve(strategy_id, "APPROVE", note)
        self.registry.add_master_journal(run_id, MasterStep.APPROVAL.value, "APPROVAL_ACCEPTED", result.model_dump(mode="json"))
        self.registry.update_master_run(run_id, approval_status="APPROVED")
        self._set_step(run_id, MasterStep.IMPLEMENTATION, MasterRunStatus.WAITING_FOR_APPROVAL, approval="APPROVED")
        return self.status(run_id)

    def resume(self, run_id: str) -> dict:
        run = self.registry.get_master_run(run_id)
        if run["approval_status"] != "APPROVED": return self.status(run_id)
        if run["current_step"] == MasterStep.COMPLETED.value: return self.status(run_id)
        options = MasterRunInput.model_validate(run["resume_state_json"]["options"] | {"intake_path": run["resume_state_json"]["intake_path"]})
        try:
            self._implementation(run_id, options)
            self._technical_verification(run_id, options)
            self._research(run_id, options)
            self._prop(run_id, options)
            self._portfolio(run_id, options)
            self._report_and_archive(run_id, options)
            self._set_step(run_id, MasterStep.COMPLETED, MasterRunStatus.SUCCESS)
        except SpecificationValidationError as exc:
            phase = self.registry.get_master_run(run_id)["current_step"]
            outcome = MasterRunStatus.TECHNICAL_FAILURE if phase in {MasterStep.IMPLEMENTATION.value, MasterStep.IMPLEMENTATION_VERIFICATION.value, MasterStep.TECHNICAL_VERIFICATION.value} else MasterRunStatus.RESEARCH_FAILURE if phase in {MasterStep.BASELINE.value, MasterStep.RESEARCH.value, MasterStep.WALK_FORWARD.value, MasterStep.HOLDOUT.value, MasterStep.STRESS.value} else MasterRunStatus.PROP_FAILURE if phase == MasterStep.PROP.value else MasterRunStatus.PORTFOLIO_FAILURE
            self.registry.add_master_journal(run_id, phase, "FAILURE", {"error_type": type(exc).__name__, "message": str(exc)})
            self._set_step(run_id, MasterStep(phase) if phase in {item.value for item in MasterStep} else MasterStep.FINAL_REPORT, outcome, error=str(exc))
        except Exception as exc:
            phase = self.registry.get_master_run(run_id)["current_step"]
            self.registry.add_master_journal(run_id, phase, "FAILURE", {"error_type": type(exc).__name__, "message": str(exc)})
            self._set_step(run_id, MasterStep(phase) if phase in {item.value for item in MasterStep} else MasterStep.FINAL_REPORT, MasterRunStatus.FAILED, error=str(exc))
        return self.status(run_id)

    def _implementation(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.IMPLEMENTATION_VERIFICATION): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"]); phase_b = PhaseBService(self.registry_path)
        if self._success_phase(run_id, MasterStep.IMPLEMENTATION): return
        plan = phase_b.implementation_plan(strategy_id, str(self.repository_root), dry_run=options.dry_run)
        prompt = phase_b.build_implementation_prompt(strategy_id, plan)
        result = phase_b.execute_codex(strategy_id, str(self.repository_root), plan, prompt, dry_run=options.dry_run or not options.implementation_enabled, task_name=f"master-{run_id}")
        if not result.success: raise SpecificationValidationError(result.stderr or result.error_type or "implementation failed")
        self._record_phase(run_id, MasterStep.IMPLEMENTATION, {"plan": plan.model_dump(mode="json"), "codex": result.model_dump(mode="json")}, root / "implementation" / "result.json")
        self._set_step(run_id, MasterStep.IMPLEMENTATION_VERIFICATION, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _technical_verification(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.TECHNICAL_VERIFICATION): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"]); phase_b = PhaseBService(self.registry_path)
        implementation = self.registry.get_master_phase_result(run_id, MasterStep.IMPLEMENTATION.value)
        plan = implementation["result_json"]["plan"]
        test = phase_b.run_required_tests(str(self.repository_root), plan["required_tests"], dry_run=options.dry_run or not options.implementation_enabled, worktree_path=plan["worktree_path"])
        repair_results = []
        if not test.passed:
            max_attempts = int(plan.get("max_repair_attempts", 0))
            for attempt in range(1, max_attempts + 1):
                phase_b.controller.consume_budget(strategy_id, codex_repairs=1)
                repair_prompt = f"Repair only the concrete implementation test failures for {strategy_id}. Failure output: {test.failure_summary}. Preserve approved invariants and do not change strategy rules, run backtests, or optimize."
                codex = phase_b.execute_codex(strategy_id, str(self.repository_root), phase_b.implementation_plan(strategy_id, str(self.repository_root), dry_run=options.dry_run), repair_prompt, dry_run=options.dry_run or not options.implementation_enabled, task_name=f"master-repair-{attempt}")
                test = phase_b.run_required_tests(str(self.repository_root), plan["required_tests"], dry_run=options.dry_run or not options.implementation_enabled, worktree_path=plan["worktree_path"])
                repair_results.append({"attempt": attempt, "codex": codex.model_dump(mode="json"), "tests": test.model_dump(mode="json")})
                repair_path = root / "verification" / f"repair-{attempt}.json"
                self._write_json(repair_path, repair_results[-1]); self._record_artifact(run_id, MasterStep.IMPLEMENTATION_VERIFICATION, repair_path, "repair-result")
                if test.passed: break
        summary = phase_b.technical_verification(strategy_id, test, implementation_executed=bool(implementation["result_json"]["codex"].get("executed")), repair_attempts=0, worktree_path=plan["worktree_path"])
        self._record_phase(run_id, MasterStep.IMPLEMENTATION_VERIFICATION, {"tests": test.model_dump(mode="json"), "summary": summary.model_dump(mode="json"), "repair_attempts": len(repair_results), "repairs": repair_results}, root / "verification" / "implementation.json")
        if not test.passed: raise SpecificationValidationError("technical tests failed")
        manifest = make_fixture(root / "verification" / "b5", strategy_id, self.registry.get_strategy(strategy_id)["version"])
        result = VerificationService(self.registry_path).run(strategy_id, manifest)
        self._record_phase(run_id, MasterStep.TECHNICAL_VERIFICATION, result, root / "verification" / "technical.json")
        if result.get("outcome") != "VERIFIED": raise SpecificationValidationError(f"technical integrity verification failed: {result.get('outcome')}")
        self._set_step(run_id, MasterStep.RESEARCH, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _research(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.RESEARCH): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"])
        service = PhaseCService(self.registry_path, repository_root=self.repository_root, scenario=options.research_scenario)
        if self.registry.get_split(strategy_id) is None:
            if options.dry_run:
                service.controller.create_split(strategy_id, make_phase_c_split())
            else:
                raise SpecificationValidationError("non-dry F1 runs require an approved chronological split")
        service.start(strategy_id, f"master-{run_id}"); result = self._drive_research(service, strategy_id)
        self._record_phase(run_id, MasterStep.RESEARCH, result, root / "research" / "result.json")
        state = self.registry.get_strategy(strategy_id)["current_phase"]
        if state not in {PipelineState.ACCEPTED.value, PipelineState.INSUFFICIENT_EVIDENCE.value, PipelineState.REJECTED.value}: raise SpecificationValidationError(f"research did not reach a terminal classification: {state}")
        self._set_step(run_id, MasterStep.PROP, MasterRunStatus.WAITING_FOR_APPROVAL)

    @staticmethod
    def _drive_research(service: PhaseCService, strategy_id: str) -> dict:
        service.run_baseline(strategy_id); edge = service.evaluate_edge(strategy_id)
        if edge["decision"] != "CONTINUE": return service.status(strategy_id)
        for _ in range(6):
            decision = service.analyze(strategy_id)
            if decision.decision == "FREEZE_CANDIDATE": break
            proposal = service.propose_round(decision); result = service.run_round(strategy_id, proposal); service.review_round(strategy_id, result.round_id)
            if result.selected_value is None: break
            service.freeze_family(strategy_id, result.round_id)
        if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.PARAMETER_RESEARCH.value:
            service.freeze_candidate(strategy_id); service.run_walk_forward(strategy_id); service.run_holdout(strategy_id); service.run_stress(strategy_id); service.run_throughput(strategy_id); service.final_review(strategy_id)
        return service.status(strategy_id)

    def _prop(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.PROP): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"]); service = PropResearchService(self.registry_path, repository_root=self.repository_root, scenario=options.prop_scenario)
        service.start(strategy_id, f"master-prop-{run_id}"); rules = service.verify_rules(strategy_id, options.prop_product); contracts = service.verify_contracts(strategy_id)
        if rules.get("status") != "VERIFIED" or contracts.get("errors"): raise SpecificationValidationError("prop entry verification failed")
        service.reconcile(strategy_id); service.run_risk(strategy_id, options.prop_product); service.run_scenarios(strategy_id, options.prop_product); review = service.economics(strategy_id)
        self._record_phase(run_id, MasterStep.PROP, {"review": review.model_dump(mode="json"), "status": service.status(strategy_id)}, root / "prop" / "result.json")
        self._set_step(run_id, MasterStep.PORTFOLIO, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _portfolio(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.PORTFOLIO): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"])
        eligible = self.registry.list_strategies(); candidates = []
        for item in eligible:
            if item["strategy_id"] == strategy_id: candidates.append(item["strategy_id"])
        result = {"classification": FinalClassification.INSUFFICIENT_EVIDENCE.value, "eligible_strategy_ids": candidates, "reason": "Phase F1 requires at least two independently eligible frozen strategies for portfolio evaluation; no portfolio was fabricated.", "scenario": options.portfolio_scenario}
        self._record_phase(run_id, MasterStep.PORTFOLIO, result, root / "portfolio" / "result.json")
        self._set_step(run_id, MasterStep.FINAL_REPORT, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _report_and_archive(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.ARCHIVE): return
        run = self.registry.get_master_run(run_id); state = run["resume_state_json"]; strategy_id = state["strategy_id"]; root = Path(run["root_path"]); strategy = self.registry.get_strategy(strategy_id); spec = self.registry.get_specification(strategy_id)
        phase_results = {item["phase"]: item["result_json"] for item in self.registry.master_phase_results(run_id)}
        research = self.registry.get_research_json("research_final_reviews", strategy_id) or {}; prop = self.registry.get_prop_record("prop_final_reviews", strategy_id) or {}; prop_review = prop.get("result_json", {}) if prop else {}; portfolio = phase_results.get(MasterStep.PORTFOLIO.value, {})
        classification = self._classification(research.get("classification"), prop_review.get("classification"), portfolio.get("classification"))
        artifacts = [ArtifactReference(phase=item["phase"], path=item["artifact_path"], artifact_type=item["artifact_type"], sha256=item["artifact_hash"]) for item in self.registry.master_artifacts(run_id)]
        report = FinalReport(run_id=run_id, strategy_id=strategy_id, strategy_version=strategy["version"], classification=classification, specification={"specification": spec.model_dump(mode="json"), "source": state.get("specification_path")}, implementation_summary=phase_results.get(MasterStep.IMPLEMENTATION.value, {}), verification_summary={"implementation": phase_results.get(MasterStep.IMPLEMENTATION_VERIFICATION.value, {}), "technical": phase_results.get(MasterStep.TECHNICAL_VERIFICATION.value, {})}, research_summary=phase_results.get(MasterStep.RESEARCH.value, {}), prop_summary=phase_results.get(MasterStep.PROP.value, {}), portfolio_summary=portfolio, final_recommendation=self._recommendation(classification), known_limitations=spec.known_limitations + ["Phase F1 portfolio evaluation requires two independently eligible strategies."], confidence="SYNTHETIC_FIXTURE" if options.dry_run else "DETERMINISTIC_PIPELINE_EVIDENCE", artifacts=artifacts, hashes={"specification_hash": spec.specification_hash, "input_hash": run["input_hash"]}, phase_timings=[PhaseTiming(phase=item["phase"], status=item["status"], started_at=item["started_at"], ended_at=item["ended_at"], duration_ms=item["duration_ms"], result_hash=item["result_hash"], artifact_paths=item["artifact_paths_json"]) for item in self.registry.master_phase_results(run_id)], generated_at=datetime.now(timezone.utc))
        report_path = root / "report" / "final_report.json"; report_hash = self._write_json(report_path, report.model_dump(mode="json")); self.registry.save_master_report(run_id, str(report_path.resolve()), report_hash, report.model_dump(mode="json")); self._record_artifact(run_id, MasterStep.FINAL_REPORT, report_path, "final-report")
        archive_manifest = root / "archive" / "manifest.json"; self._write_json(archive_manifest, {"run_id": run_id, "report_path": str(report_path.resolve()), "report_hash": report_hash, "artifact_count": len(artifacts)}); self._record_phase(run_id, MasterStep.ARCHIVE, {"report_path": str(report_path.resolve()), "report_hash": report_hash, "archive_manifest": str(archive_manifest.resolve())}, root / "archive" / "result.json")

    @staticmethod
    def _classification(research: str | None, prop: str | None, portfolio: str | None) -> FinalClassification:
        if portfolio == "PORTFOLIO_ACCEPTED": return FinalClassification.PORTFOLIO_ACCEPTED
        if prop in {"REJECTED_PROP_INCOMPATIBLE", "REJECTED_NEGATIVE_ECONOMICS", "TECHNICAL_FAILURE"}: return FinalClassification.REJECTED
        if prop == "OWN_CAPITAL_ONLY": return FinalClassification.OWN_CAPITAL_ONLY
        if portfolio in {"PORTFOLIO_INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"} or prop in {"INSUFFICIENT_PROP_EVIDENCE", "INSUFFICIENT_FUTURES_DATA"}: return FinalClassification.INSUFFICIENT_EVIDENCE
        if research == "ACCEPTED_PORTFOLIO_COMPONENT" or prop == "PROP_ACCEPTED_PORTFOLIO_COMPONENT": return FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT
        if research == "ACCEPTED_STANDALONE" and prop == "PROP_ACCEPTED_STANDALONE": return FinalClassification.ACCEPTED_STANDALONE
        if research and research.startswith("REJECTED"): return FinalClassification.REJECTED
        return FinalClassification.INSUFFICIENT_EVIDENCE

    @staticmethod
    def _recommendation(classification: FinalClassification) -> str:
        return {FinalClassification.ACCEPTED_STANDALONE: "Evidence supports standalone research acceptance; operational authorization is outside F1.", FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT: "Evidence supports use as a portfolio component; Phase E portfolio review remains required.", FinalClassification.OWN_CAPITAL_ONLY: "Do not use in the evaluated prop model; own-capital suitability is a separate decision.", FinalClassification.PORTFOLIO_ACCEPTED: "Portfolio evidence supports the evaluated composition; this is not trading authorization.", FinalClassification.REJECTED: "Reject the strategy for this pipeline run.", FinalClassification.INSUFFICIENT_EVIDENCE: "Do not advance to deployment; evidence is insufficient for the requested classification.", FinalClassification.MANUAL_REVIEW_REQUIRED: "Pause for human review of unresolved material ambiguity."}[classification]

    def status(self, run_id: str) -> dict:
        run = self.registry.get_master_run(run_id); artifacts = [ArtifactReference(phase=item["phase"], path=item["artifact_path"], artifact_type=item["artifact_type"], sha256=item["artifact_hash"]) for item in self.registry.master_artifacts(run_id)]
        return MasterStatus(run_id=run_id, strategy_id=run["strategy_id"], strategy_version=run["strategy_version"], current_step=MasterStep(run["current_step"]), outcome=MasterRunStatus(run["outcome"]), approval_status=run["approval_status"], root_path=run["root_path"], phase_results=self.registry.master_phase_results(run_id), journal_entries=len(self.registry.master_journal(run_id)), artifacts=artifacts, report=self.registry.master_report(run_id)).model_dump(mode="json")

    def report(self, run_id: str) -> dict:
        report = self.registry.master_report(run_id)
        if not report: raise RegistryError(f"final report not found: {run_id}")
        return report["report_json"]

    def artifacts(self, run_id: str) -> list[dict]:
        return self.registry.master_artifacts(run_id)

    def journal(self, run_id: str) -> list[dict]:
        return self.registry.master_journal(run_id)

    def cancel(self, run_id: str, reason: str = "cancelled by operator") -> dict:
        self.registry.add_master_journal(run_id, self.registry.get_master_run(run_id)["current_step"], "CANCELLED", {"reason": reason})
        self._set_step(run_id, MasterStep.COMPLETED, MasterRunStatus.ABORTED, cancel_reason=reason)
        return self.status(run_id)
