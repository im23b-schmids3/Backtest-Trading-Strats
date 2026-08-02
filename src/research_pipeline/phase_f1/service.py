from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import pandas as pd

from ..enums import PipelineState
from ..errors import ExternalSpecificationRequired, InvalidTransitionError, RegistryError, SpecificationValidationError
from ..phase_b.models import GeneratedStrategySpec, WorkflowInput
from ..phase_b.services import PhaseBService
from ..adapters.compatibility import verify_artifact
from ..adapters.data import chronological_split
from ..adapters.errors import AdapterError, DataAvailabilityError, RealAdapterRequired
from ..adapters.models import ImplementationManifest
from ..adapters.native_backtest import NativeTradeSignalAdapter
from ..adapters.registry import default_adapter_registry
from ..prop.services import PropResearchService
from ..prop.mappings import default_market_mappings
from ..registry.database import Database
from ..registry.repositories import Registry
from ..research.fixtures import make_phase_c_split
from ..schemas.splits import SplitDefinition, SplitWindow, calculate_split_hash
from ..research.services import PhaseCService
from ..verification.fixtures import make_fixture
from ..verification.services import VerificationService
from ..implementation.jobs import ImplementationJobService
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
                    prop_product: str = "Alpha Futures Zero 25K", mode: str | None = None, allow_proxy_data: bool = False) -> MasterRunInput:
        selected_mode = mode or ("dry_run" if dry_run else "real_run")
        return MasterRunInput(intake_path=str(Path(intake_path).resolve()), repository_root=str(Path(repository_root).resolve()), registry_path=str(Path(registry_path).resolve()) if registry_path else None, dry_run=selected_mode == "dry_run", implementation_enabled=implementation_enabled, research_scenario=research_scenario, prop_scenario=prop_scenario, portfolio_scenario=portfolio_scenario, prop_product=prop_product, mode=selected_mode, allow_proxy_data=allow_proxy_data)

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

    def _real_adapter(self, run_id: str, repository_root: str | Path | None = None):
        run = self.registry.get_master_run(run_id); state = run["resume_state_json"]; spec = self.registry.get_specification(state["strategy_id"])
        options = MasterRunInput.model_validate(state["options"] | {"intake_path": state["intake_path"]})
        root = Path(repository_root).resolve() if repository_root else self.repository_root
        adapter = default_adapter_registry().resolve(spec, root)
        health = adapter.health(spec)
        self.registry.save_strategy_adapter(adapter.identity.model_dump(mode="json"), adapter.capabilities.model_dump(mode="json"), health.model_dump(mode="json"))
        if not health.healthy: raise RealAdapterRequired("REAL_ADAPTER_REQUIRED: adapter health check failed")
        availability = adapter.data_availability(spec)
        if any(item.classification.value == "AVAILABLE_PROXY" for item in availability) and not options.allow_proxy_data:
            raise DataAvailabilityError("INSUFFICIENT_MARKET_DATA: proxy data requires --allow-proxy-data")
        return adapter, spec

    @staticmethod
    def _real_split(adapter, spec) -> SplitDefinition:
        availability = adapter.require_data(spec)
        first = availability[0]
        requested_start = spec.baseline_parameters.get("test_start_date")
        requested_end = spec.baseline_parameters.get("test_end_date")
        if requested_start or requested_end:
            bounds = {}
            if requested_start:
                bounds["start_timestamp"] = pd.Timestamp(str(requested_start), tz="UTC").to_pydatetime()
            if requested_end:
                bounds["end_timestamp"] = pd.Timestamp(str(requested_end), tz="UTC").to_pydatetime()
            first = first.model_copy(update=bounds)
        start, train_end, validation_start, validation_end, holdout_start, end = chronological_split(first)
        raw = {"dataset_identifier": f"{first.provider}:{first.source_symbol}:{spec.timeframes[0]}", "source_data_hash": first.dataset_hash or "missing", "start_timestamp": start, "end_timestamp": end,
               "training_boundaries": SplitWindow(start_timestamp=start, end_timestamp=train_end),
               "validation_boundaries": SplitWindow(start_timestamp=validation_start, end_timestamp=validation_end), "holdout_boundaries": SplitWindow(start_timestamp=holdout_start, end_timestamp=end),
               "created_timestamp": datetime.now(timezone.utc), "split_hash": "pending"}
        candidate = SplitDefinition.model_construct(**raw); raw["split_hash"] = calculate_split_hash(candidate)
        return SplitDefinition.model_validate(raw)

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
        try:
            return self._generate_and_register(run_id, options, intake, root)
        except ExternalSpecificationRequired as exc:
            return self._pause_for_external_specification(run_id, exc, root)
        except RuntimeError as exc:
            if str(exc).startswith("CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR"):
                failure_path = root / "specification" / "external-execution-required.json"
                payload = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {"classification": "CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR", "final_reason": str(exc)}
                self._record_phase(run_id, MasterStep.SPECIFICATION, payload, root / "specification" / "result.json", status="WAITING_EXTERNAL")
                self.registry.add_master_journal(run_id, MasterStep.SPECIFICATION.value, "CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR", payload)
                self._set_step(run_id, MasterStep.SPECIFICATION, MasterRunStatus.WAITING_EXTERNAL_CODEX, specification_failure=payload, external_executor_required=True, next_command="External authenticated Codex execution is required; Smithers restricted mode must not invoke it.")
                return self.status(run_id)
            if not str(exc).startswith("SPECIFICATION_GENERATION_FAILURE"):
                raise
            failure_path = root / "specification" / "failure" / "final_failure.json"
            payload = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {"classification": "SPECIFICATION_GENERATION_FAILURE", "final_reason": str(exc)}
            self._record_phase(run_id, MasterStep.SPECIFICATION, payload, root / "specification" / "result.json", status="FAILED")
            self.registry.add_master_journal(run_id, MasterStep.SPECIFICATION.value, "SPECIFICATION_GENERATION_FAILURE", payload)
            self._set_step(run_id, MasterStep.SPECIFICATION, MasterRunStatus.SPECIFICATION_GENERATION_FAILURE, specification_failure=payload)
            return self.status(run_id)

    def _pause_for_external_specification(self, run_id: str, exc: ExternalSpecificationRequired, root: Path | None = None) -> dict:
        current = self.registry.get_master_run(run_id)
        root = root or Path(current["root_path"])
        payload = {"classification": exc.classification, "run_id": exc.run_id, "job_id": exc.job_id, "next_command": exc.command, "reason": str(exc)}
        self._record_phase(run_id, MasterStep.SPECIFICATION, payload, root / "specification" / "external-job-required.json", status="WAITING_EXTERNAL")
        self.registry.add_master_journal(run_id, MasterStep.SPECIFICATION.value, exc.classification, payload)
        self._set_step(run_id, MasterStep.SPECIFICATION, exc.classification, specification_job_id=exc.job_id, external_executor_required=True, next_command=exc.command, specification_pause_reason=exc.classification)
        return self.status(run_id)

    def _generate_and_register(self, run_id: str, options: MasterRunInput, intake: IntakeSpec, root: Path) -> dict:
        generation_name = intake.strategy_name
        phase_b = PhaseBService(self.registry_path)
        if options.prebuilt_spec_path:
            prebuilt = Path(options.prebuilt_spec_path).resolve()
            spec = yaml.safe_load(prebuilt.read_text(encoding="utf-8"))
            validated = phase_b._validated_with_hash(spec)
            generated = GeneratedStrategySpec(strategy_id=validated.strategy_id, version=validated.version, specification_path=str(prebuilt), specification_hash=validated.specification_hash,
                                              assumptions=list(validated.session_assumptions), ambiguities=[], fields_requiring_confirmation=[], manual_review_required=False,
                                              approval_summary=json.dumps({"strategy_id": validated.strategy_id, "version": validated.version, "specification_hash": validated.specification_hash}, sort_keys=True))
        else:
            workflow = WorkflowInput(strategy_name=generation_name, natural_language_description=self._description(intake), requested_markets=intake.markets, requested_timeframes=intake.timeframes, optional_notes=intake.optional_notes, repository_root=str(self.repository_root), registry_path=str(self.registry_path), dry_run=options.dry_run, implementation_enabled=options.implementation_enabled, run_id=run_id, confirmed_facts=intake.confirmed_facts, assumptions=intake.assumptions, missing_information=intake.missing_information, ambiguities=intake.ambiguities)
            generated = phase_b.generate_spec(workflow)
        draft_copy = root / "specification" / Path(generated.specification_path).name
        draft_copy.write_text(Path(generated.specification_path).read_text(encoding="utf-8"), encoding="utf-8")
        validation = phase_b.validate_spec(generated)
        if not validation.valid and not (generated.manual_review_required or validation.manual_review_required):
            details = " | ".join(validation.errors) or "canonical specification failed validation"
            raise RuntimeError(f"SPECIFICATION_GENERATION_FAILURE: {details}")
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
        if run["current_step"] == MasterStep.SPECIFICATION.value and run["outcome"] in {MasterRunStatus.WAITING_EXTERNAL_SPECIFICATION_GENERATION.value, MasterRunStatus.WAITING_EXTERNAL_SPECIFICATION_REPAIR.value}:
            options = MasterRunInput.model_validate(run["resume_state_json"]["options"] | {"intake_path": run["resume_state_json"]["intake_path"]})
            intake = IntakeSpec.model_validate(run["resume_state_json"]["intake"])
            try:
                return self._generate_and_register(run_id, options, intake, Path(run["root_path"]))
            except ExternalSpecificationRequired as exc:
                return self._pause_for_external_specification(run_id, exc)
            except RuntimeError as exc:
                if str(exc).startswith("SPECIFICATION_GENERATION_FAILURE"):
                    self._set_step(run_id, MasterStep.SPECIFICATION, MasterRunStatus.SPECIFICATION_GENERATION_FAILURE, specification_failure=str(exc))
                    return self.status(run_id)
                raise
        if run["approval_status"] != "APPROVED": return self.status(run_id)
        if run["current_step"] == MasterStep.COMPLETED.value: return self.status(run_id)
        options = MasterRunInput.model_validate(run["resume_state_json"]["options"] | {"intake_path": run["resume_state_json"]["intake_path"]})
        try:
            self._implementation(run_id, options)
            current = self.registry.get_master_run(run_id)
            if current["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value:
                return self.status(run_id)
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
            outcome = MasterRunStatus.IMPLEMENTATION_FAILURE if phase == MasterStep.IMPLEMENTATION.value else MasterRunStatus.FAILED
            self._set_step(run_id, MasterStep(phase) if phase in {item.value for item in MasterStep} else MasterStep.FINAL_REPORT, outcome, error=str(exc))
        return self.status(run_id)

    def retry_specification(self, run_id: str) -> dict:
        run = self.registry.get_master_run(run_id)
        if run["current_step"] != MasterStep.SPECIFICATION.value:
            return self.status(run_id)
        state = run["resume_state_json"]
        options = MasterRunInput.model_validate(state["options"] | {"intake_path": state["intake_path"]})
        intake = IntakeSpec.model_validate(state["intake"])
        try:
            return self._generate_and_register(run_id, options, intake, Path(run["root_path"]))
        except ExternalSpecificationRequired as exc:
            return self._pause_for_external_specification(run_id, exc)
        except RuntimeError as exc:
            if str(exc).startswith("CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR"):
                payload = {"classification": "CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR", "final_reason": str(exc)}
                self.registry.add_master_journal(run_id, MasterStep.SPECIFICATION.value, "CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR", payload)
                self._set_step(run_id, MasterStep.SPECIFICATION, MasterRunStatus.WAITING_EXTERNAL_CODEX, specification_failure=payload, external_executor_required=True, next_command="External authenticated Codex execution is required; Smithers restricted mode must not invoke it.")
                return self.status(run_id)
            if not str(exc).startswith("SPECIFICATION_GENERATION_FAILURE"):
                raise
            self._set_step(run_id, MasterStep.SPECIFICATION, MasterRunStatus.SPECIFICATION_GENERATION_FAILURE, specification_failure=str(exc))
            return self.status(run_id)

    def _implementation(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.IMPLEMENTATION_VERIFICATION): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"]); phase_b = PhaseBService(self.registry_path)
        if self._success_phase(run_id, MasterStep.IMPLEMENTATION): return
        if options.mode == "real_run":
            jobs = ImplementationJobService(self.registry_path)
            existing = self.registry.get_implementation_job(run_id)
            if not existing or existing["status"] == "WAITING_EXTERNAL_CODEX":
                created = jobs.create(run_id)
                state = self.registry.get_master_run(run_id)["resume_state_json"]
                self._set_step(run_id, MasterStep.IMPLEMENTATION, MasterRunStatus.WAITING_EXTERNAL_CODEX,
                                implementation_job_id=created["job"]["job_id"],
                                implementation_job_path=created["job_path"],
                                worktree_preflight=created.get("preflight") or {},
                                external_executor_required=True,
                                next_command=created.get("next_command", f"py -m research_pipeline codex-executor run {run_id}"))
                return
            if existing["status"] != "INGESTED":
                completion_path = Path(existing.get("result_path") or "")
                if not completion_path.is_file():
                    self._set_step(run_id, MasterStep.IMPLEMENTATION, MasterRunStatus.WAITING_EXTERNAL_CODEX,
                                   implementation_job_id=existing["job_id"], external_executor_required=True,
                                   next_command=f"py -m research_pipeline codex-executor run {run_id}")
                    return
                completion = jobs.ingest(run_id)
            else:
                completion_path = Path(existing["result_path"])
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
            spec = self.registry.get_specification(strategy_id)
            worktree_path = completion["worktree_path"]
            adapter, spec = self._real_adapter(run_id, worktree_path)
            manifest = ImplementationManifest(master_run_id=run_id, strategy_id=spec.strategy_id, strategy_version=spec.version, specification_hash=spec.specification_hash,
                                               base_commit=completion.get("base_commit") or "unknown", implementation_commit=completion.get("resulting_commit"),
                                               worktree_path=worktree_path, branch=json.loads((Path(self.registry.get_implementation_job(run_id)["job_path"]) / "request.json").read_text(encoding="utf-8"))["branch"],
                                               files_modified=completion.get("changed_files", []),
                                               adapter_registration=f"{adapter.identity.implementation_module}:{adapter.identity.entry_point}", strategy_entry_point=adapter.identity.entry_point,
                                               verification_command=[sys.executable, "-m", "pytest", "-q", "tests/research_pipeline/test_phase_f2_adapters.py"],
                                               known_limitations=["External Codex implementation is isolated and is not automatically merged into the primary branch"], adapter_version=adapter.identity.adapter_version)
            root = Path(self.registry.get_master_run(run_id)["root_path"])
            manifest_path = root / "implementation" / "manifest" / "implementation.json"; manifest_hash = self._write_json(manifest_path, manifest.model_dump(mode="json"))
            self.registry.save_implementation_manifest(run_id, str(manifest_path.resolve()), manifest_hash, manifest.model_dump(mode="json"))
            self.registry.save_worktree_metadata(run_id, {"strategy_id": spec.strategy_id, "strategy_version": spec.version, "base_commit": manifest.base_commit,
                                                           "implementation_commit": manifest.implementation_commit, "branch": manifest.branch, "worktree_path": manifest.worktree_path, "status": "EXTERNAL_VALIDATED"})
            self._record_phase(run_id, MasterStep.IMPLEMENTATION, {"mode": "external_codex", "adapter": adapter.identity.model_dump(mode="json"), "manifest": manifest.model_dump(mode="json"), "completion": completion}, root / "implementation" / "result.json")
            self._set_step(run_id, MasterStep.IMPLEMENTATION_VERIFICATION, MasterRunStatus.WAITING_FOR_APPROVAL,
                           implementation_job_id=self.registry.get_implementation_job(run_id)["job_id"], external_executor_required=False)
            return
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
        if options.mode == "real_run":
            worktree_path = implementation["result_json"].get("manifest", {}).get("worktree_path") or str(self.repository_root)
            adapter, spec = self._real_adapter(run_id, worktree_path)
            if self.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.IMPLEMENTATION.value:
                phase_b.controller.transition(strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "real adapter implementation discovered")
            test = phase_b.tests.run(worktree_path, [sys.executable, "-m", "pytest", "-q", "tests/research_pipeline/test_phase_f2_adapters.py"], dry_run=False, report_path=root / "implementation" / "tests" / "adapter-tests.txt")
            if not test.passed: raise SpecificationValidationError("real adapter implementation verification failed")
            artifact = adapter.run_baseline(spec, self._real_split(adapter, spec), root / "technical_verification" / "baseline")
            verification = VerificationService(self.registry_path).run(strategy_id, artifact.diagnostic_manifest_path or "")
            self._record_phase(run_id, MasterStep.IMPLEMENTATION_VERIFICATION, {"tests": test.model_dump(mode="json"), "adapter": adapter.identity.model_dump(mode="json"), "artifact": artifact.model_dump(mode="json")}, root / "verification" / "implementation.json")
            verification = {**verification, "data": [item.model_dump(mode="json") for item in adapter.data_availability(spec)], "adapter": adapter.identity.model_dump(mode="json")}
            self._record_phase(run_id, MasterStep.TECHNICAL_VERIFICATION, verification, root / "verification" / "technical.json")
            if verification.get("outcome") != "VERIFIED": raise SpecificationValidationError(f"real B.5 verification failed: {verification.get('outcome')}")
            self.registry.save_baseline(strategy_id, spec.version, artifact.experiment_id, artifact.model_dump(mode="json"), verification["outcome"], [])
            self._set_step(run_id, MasterStep.BASELINE, MasterRunStatus.WAITING_FOR_APPROVAL)
            return
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
        adapter = None
        if options.mode == "real_run":
            adapter, spec = self._real_adapter(run_id)
            if self.registry.get_split(strategy_id) is None: self.registry.create_split(strategy_id, None, self._real_split(adapter, spec))
        service = PhaseCService(self.registry_path, adapter=adapter, repository_root=self.repository_root, scenario=options.research_scenario)
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
            service.freeze_candidate(strategy_id); service.run_walk_forward(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] != PipelineState.HOLDOUT.value:
                return service.status(strategy_id)
            service.run_holdout(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] != PipelineState.STRESS_TESTS.value:
                return service.status(strategy_id)
            service.run_stress(strategy_id); service.run_throughput(strategy_id); service.final_review(strategy_id)
        return service.status(strategy_id)

    def _prop(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.PROP): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"])
        if self.registry.get_strategy(strategy_id)["current_phase"] != PipelineState.ACCEPTED.value:
            self._record_phase(run_id, MasterStep.PROP, {"status": "SKIPPED", "reason": "Phase C did not accept the candidate"}, root / "prop" / "result.json")
            self._set_step(run_id, MasterStep.FINAL_REPORT, MasterRunStatus.WAITING_FOR_APPROVAL)
            return
        spec = self.registry.get_specification(strategy_id)
        if not any(item.strategy_market in spec.markets for item in default_market_mappings()):
            result = {"status": "SKIPPED", "classification": "INSUFFICIENT_FUTURES_DATA",
                      "reason": "Phase D is not applicable: no existing futures mapping supports the strategy market.",
                      "markets": spec.markets, "mapping_source": "existing repository futures mappings"}
            self._record_phase(run_id, MasterStep.PROP, result, root / "prop" / "result.json")
            self._set_step(run_id, MasterStep.PORTFOLIO, MasterRunStatus.WAITING_FOR_APPROVAL)
            return
        trade_adapter = None
        real_adapter = None
        if options.mode == "real_run":
            real_adapter, _ = self._real_adapter(run_id); trade_adapter = NativeTradeSignalAdapter(real_adapter)
        service = PropResearchService(self.registry_path, repository_root=self.repository_root, scenario=options.prop_scenario, trade_adapter=trade_adapter)
        service.start(strategy_id, f"master-prop-{run_id}"); rules = service.verify_rules(strategy_id, options.prop_product); contracts = service.verify_contracts(strategy_id)
        if rules.get("status") != "VERIFIED" or contracts.get("errors"): raise SpecificationValidationError("prop entry verification failed")
        service.reconcile(strategy_id); service.run_risk(strategy_id, options.prop_product); service.run_scenarios(strategy_id, options.prop_product); review = service.economics(strategy_id)
        result_payload = {"review": review.model_dump(mode="json"), "status": service.status(strategy_id)}
        if options.mode == "real_run" and real_adapter is not None:
            candidate = self.registry.get_candidate(strategy_id); spec = self.registry.get_specification(strategy_id); candidate_hash = candidate["candidate_hash"]
            events = [item.model_dump(mode="json") for item in real_adapter.phase_d_export(spec, candidate_hash)]
            export_path = root / "prop" / "futures_export.json"; export_hash = self._write_json(export_path, {"strategy_id": strategy_id, "strategy_version": spec.version, "candidate_hash": candidate_hash, "events": events})
            self.registry.save_phase_d_export(run_id, {"strategy_id": strategy_id, "strategy_version": spec.version, "candidate_hash": candidate_hash, "events": events}, str(export_path.resolve()), export_hash)
            self._record_artifact(run_id, MasterStep.PROP, export_path, "phase-d-export")
            result_payload["phase_d_export"] = {"path": str(export_path.resolve()), "hash": export_hash, "event_count": len(events), "candidate_hash": candidate_hash}
        self._record_phase(run_id, MasterStep.PROP, result_payload, root / "prop" / "result.json")
        self._set_step(run_id, MasterStep.PORTFOLIO, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _portfolio(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.PORTFOLIO): return
        run = self.registry.get_master_run(run_id); strategy_id = run["resume_state_json"]["strategy_id"]; root = Path(run["root_path"])
        if self.registry.get_strategy(strategy_id)["current_phase"] != PipelineState.ACCEPTED.value:
            result = {"classification": "NOT_ELIGIBLE", "eligible_strategy_ids": [], "reason": "Phase C did not produce an accepted candidate; portfolio evaluation was not attempted.", "scenario": options.portfolio_scenario}
            self._record_phase(run_id, MasterStep.PORTFOLIO, result, root / "portfolio" / "result.json")
            self._set_step(run_id, MasterStep.FINAL_REPORT, MasterRunStatus.WAITING_FOR_APPROVAL)
            return
        eligible = self.registry.list_strategies(); candidates = []
        for item in eligible:
            if item["strategy_id"] == strategy_id: candidates.append(item["strategy_id"])
        if options.mode == "real_run":
            from ..portfolio.eligibility import eligibility
            result = eligibility(self.registry, strategy_id, exploratory_prop=False, non_prop=False)
            result.update({"classification": "ELIGIBLE_STANDALONE" if result.get("eligible") else "NOT_ELIGIBLE", "portfolio_deferred": "PORTFOLIO_DEFERRED_INSUFFICIENT_MEMBERS", "scenario": options.portfolio_scenario})
            adapter, spec = self._real_adapter(run_id); candidate = self.registry.get_candidate(strategy_id); candidate_hash = candidate["candidate_hash"]
            eligibility_result = adapter.phase_e_export(spec, candidate_hash, result.get("phase_c_classification") or "UNKNOWN", result.get("phase_d_classification"))
            result["adapter_eligibility"] = eligibility_result.model_dump(mode="json")
            self.registry.save_phase_e_eligibility(run_id, eligibility_result.model_dump(mode="json"))
        else:
            result = {"classification": FinalClassification.INSUFFICIENT_EVIDENCE.value, "eligible_strategy_ids": candidates, "reason": "Phase F1 requires at least two independently eligible frozen strategies for portfolio evaluation; no portfolio was fabricated.", "scenario": options.portfolio_scenario}
        self._record_phase(run_id, MasterStep.PORTFOLIO, result, root / "portfolio" / "result.json")
        self._set_step(run_id, MasterStep.FINAL_REPORT, MasterRunStatus.WAITING_FOR_APPROVAL)

    def _report_and_archive(self, run_id: str, options: MasterRunInput) -> None:
        if self._success_phase(run_id, MasterStep.ARCHIVE): return
        run = self.registry.get_master_run(run_id); state = run["resume_state_json"]; strategy_id = state["strategy_id"]; root = Path(run["root_path"]); strategy = self.registry.get_strategy(strategy_id); spec = self.registry.get_specification(strategy_id)
        phase_results = {item["phase"]: item["result_json"] for item in self.registry.master_phase_results(run_id)}
        research = self.registry.get_research_json("research_final_reviews", strategy_id) or {}; prop = self.registry.get_prop_record("prop_final_reviews", strategy_id) or {}; prop_review = prop.get("result_json", {}) if prop else {}; portfolio = phase_results.get(MasterStep.PORTFOLIO.value, {})
        prop_summary = phase_results.get(MasterStep.PROP.value, {}) or {}
        prop_classification = prop_review.get("classification") or prop_summary.get("classification")
        classification = FinalClassification.REJECTED if strategy["current_phase"] == PipelineState.REJECTED.value else self._classification(research.get("classification"), prop_classification, portfolio.get("classification"), real_mode=options.mode == "real_run")
        artifacts = [ArtifactReference(phase=item["phase"], path=item["artifact_path"], artifact_type=item["artifact_type"], sha256=item["artifact_hash"]) for item in self.registry.master_artifacts(run_id)]
        research_phase = phase_results.get(MasterStep.RESEARCH.value, {}) or {}
        report = FinalReport(run_id=run_id, strategy_id=strategy_id, strategy_version=strategy["version"], classification=classification, specification={"specification": spec.model_dump(mode="json"), "source": state.get("specification_path")}, implementation_summary=phase_results.get(MasterStep.IMPLEMENTATION.value, {}) or {}, verification_summary={"implementation": phase_results.get(MasterStep.IMPLEMENTATION_VERIFICATION.value, {}) or {}, "technical": phase_results.get(MasterStep.TECHNICAL_VERIFICATION.value, {}) or {}}, research_summary=research_phase, prop_summary=phase_results.get(MasterStep.PROP.value, {}) or {}, portfolio_summary=portfolio or {}, final_recommendation=self._recommendation(classification), known_limitations=spec.known_limitations + (["Phase F1 portfolio evaluation requires two independently eligible strategies."] if options.dry_run else []), implementation_variant="1-hour repository-compatible test variant" if spec.strategy_family == "f2_random_open_test" else None, confidence="SYNTHETIC_FIXTURE" if options.dry_run else "REAL_LOCAL_DATA_DEMONSTRATION", artifacts=artifacts, hashes={"specification_hash": spec.specification_hash, "input_hash": run["input_hash"]}, phase_timings=[PhaseTiming(phase=item["phase"], status=item["status"], started_at=item["started_at"], ended_at=item["ended_at"], duration_ms=item["duration_ms"], result_hash=item["result_hash"], artifact_paths=item["artifact_paths_json"]) for item in self.registry.master_phase_results(run_id)], generated_at=datetime.now(timezone.utc), mode=options.mode, intake_summary=state.get("intake", {}), adapter_validation=phase_results.get(MasterStep.IMPLEMENTATION.value, {}).get("adapter", {}) if phase_results.get(MasterStep.IMPLEMENTATION.value) else {}, data_availability=phase_results.get(MasterStep.TECHNICAL_VERIFICATION.value, {}).get("data", []) if phase_results.get(MasterStep.TECHNICAL_VERIFICATION.value) else [], baseline_summary=research_phase.get("baseline", {}) or {}, parameter_research_summary=research_phase, frozen_candidate_summary=research_phase.get("candidate", {}) or {}, walk_forward_summary=research_phase.get("walk_forward", {}) or {}, holdout_summary=research_phase.get("holdout", {}) or {}, stress_summary=research_phase.get("stress", {}) or {}, throughput_summary=research_phase.get("throughput", {}) or {}, futures_prop_summary=phase_results.get(MasterStep.PROP.value, {}) or {}, portfolio_eligibility=portfolio or {}, git_review=self.registry.get_worktree_metadata(run_id) or {})
        report_path = root / "report" / "final_report.json"; report_hash = self._write_json(report_path, report.model_dump(mode="json")); self.registry.save_master_report(run_id, str(report_path.resolve()), report_hash, report.model_dump(mode="json")); self._record_artifact(run_id, MasterStep.FINAL_REPORT, report_path, "final-report")
        archive_manifest = root / "archive" / "manifest.json"; self._write_json(archive_manifest, {"run_id": run_id, "report_path": str(report_path.resolve()), "report_hash": report_hash, "artifact_count": len(artifacts)}); self._record_phase(run_id, MasterStep.ARCHIVE, {"report_path": str(report_path.resolve()), "report_hash": report_hash, "archive_manifest": str(archive_manifest.resolve())}, root / "archive" / "result.json")

    @staticmethod
    def _classification(research: str | None, prop: str | None, portfolio: str | None, *, real_mode: bool = False) -> FinalClassification:
        if portfolio == "PORTFOLIO_ACCEPTED": return FinalClassification.PORTFOLIO_ACCEPTED
        if prop in {"REJECTED_PROP_INCOMPATIBLE", "REJECTED_NEGATIVE_ECONOMICS", "TECHNICAL_FAILURE"}: return FinalClassification.REJECTED
        if prop == "OWN_CAPITAL_ONLY": return FinalClassification.OWN_CAPITAL_ONLY
        if prop == "INSUFFICIENT_FUTURES_DATA": return FinalClassification.INSUFFICIENT_EVIDENCE
        if not real_mode:
            if portfolio in {"PORTFOLIO_INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"} or prop in {"INSUFFICIENT_PROP_EVIDENCE", "INSUFFICIENT_FUTURES_DATA"}: return FinalClassification.INSUFFICIENT_EVIDENCE
            if research == "ACCEPTED_PORTFOLIO_COMPONENT" or prop == "PROP_ACCEPTED_PORTFOLIO_COMPONENT": return FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT
            if research == "ACCEPTED_STANDALONE" and prop == "PROP_ACCEPTED_STANDALONE": return FinalClassification.ACCEPTED_STANDALONE
            if research and research.startswith("REJECTED"): return FinalClassification.REJECTED
            return FinalClassification.INSUFFICIENT_EVIDENCE
        if research == "ACCEPTED_PORTFOLIO_COMPONENT" or prop == "PROP_ACCEPTED_PORTFOLIO_COMPONENT": return FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT
        if research == "ACCEPTED_STANDALONE" and prop == "PROP_ACCEPTED_STANDALONE": return FinalClassification.ACCEPTED_STANDALONE
        if research in {"ACCEPTED_STANDALONE", "ACCEPTED_PORTFOLIO_COMPONENT"} and portfolio in {"PORTFOLIO_DEFERRED_INSUFFICIENT_MEMBERS", "INSUFFICIENT_EVIDENCE", "ELIGIBLE_STANDALONE", "NOT_ELIGIBLE", None}:
            return FinalClassification.ACCEPTED_STANDALONE if research == "ACCEPTED_STANDALONE" else FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT
        if portfolio in {"PORTFOLIO_INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"} or prop in {"INSUFFICIENT_PROP_EVIDENCE", "INSUFFICIENT_FUTURES_DATA"}: return FinalClassification.INSUFFICIENT_EVIDENCE
        if research and research.startswith("REJECTED"): return FinalClassification.REJECTED
        return FinalClassification.INSUFFICIENT_EVIDENCE

    @staticmethod
    def _recommendation(classification: FinalClassification) -> str:
        return {FinalClassification.ACCEPTED_STANDALONE: "Evidence supports standalone research acceptance; operational authorization is outside F1.", FinalClassification.ACCEPTED_PORTFOLIO_COMPONENT: "Evidence supports use as a portfolio component; Phase E portfolio review remains required.", FinalClassification.OWN_CAPITAL_ONLY: "Do not use in the evaluated prop model; own-capital suitability is a separate decision.", FinalClassification.PORTFOLIO_ACCEPTED: "Portfolio evidence supports the evaluated composition; this is not trading authorization.", FinalClassification.REJECTED: "Reject the strategy for this pipeline run.", FinalClassification.INSUFFICIENT_EVIDENCE: "Do not advance to deployment; evidence is insufficient for the requested classification.", FinalClassification.MANUAL_REVIEW_REQUIRED: "Pause for human review of unresolved material ambiguity."}[classification]

    def status(self, run_id: str) -> dict:
        run = self.registry.get_master_run(run_id); artifacts = [ArtifactReference(phase=item["phase"], path=item["artifact_path"], artifact_type=item["artifact_type"], sha256=item["artifact_hash"]) for item in self.registry.master_artifacts(run_id)]
        mode = run["resume_state_json"].get("options", {}).get("mode", "dry_run")
        job = self.registry.get_implementation_job(run_id)
        state = run["resume_state_json"]
        job_status = job["status"] if job else None
        waiting_spec = run["outcome"] in {MasterRunStatus.WAITING_EXTERNAL_SPECIFICATION_GENERATION.value, MasterRunStatus.WAITING_EXTERNAL_SPECIFICATION_REPAIR.value}
        pipeline_status = "PIPELINE_COMPLETED" if run["current_step"] == MasterStep.COMPLETED.value and run["outcome"] == MasterRunStatus.SUCCESS.value else (run["outcome"] if waiting_spec else ("WAITING_EXTERNAL_CODEX" if run["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value else ("IMPLEMENTATION_FAILED" if run["outcome"] == MasterRunStatus.IMPLEMENTATION_FAILURE.value else "ORCHESTRATOR_COMPLETED")))
        return MasterStatus(run_id=run_id, strategy_id=run["strategy_id"], strategy_version=run["strategy_version"], current_step=MasterStep(run["current_step"]), outcome=MasterRunStatus(run["outcome"]), approval_status=run["approval_status"], root_path=run["root_path"], phase_results=self.registry.master_phase_results(run_id), journal_entries=len(self.registry.master_journal(run_id)), artifacts=artifacts, report=self.registry.master_report(run_id), mode=mode, pipeline_status=pipeline_status, implementation_job_id=job["job_id"] if job else state.get("implementation_job_id"), external_executor_required=waiting_spec or run["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value, worktree_preflight=state.get("worktree_preflight", {}), codex_execution_status=job_status, implementation_test_status=job_status if job_status in {"INGESTED", "FAILED_REQUIRED_TESTS"} else None, b5_available=bool(self.registry.get_master_phase_result(run_id, MasterStep.TECHNICAL_VERIFICATION.value)), next_command=state.get("next_command") or (f"py -m research_pipeline specification-executor run {run_id}" if waiting_spec else (f"py -m research_pipeline codex-executor run {run_id}" if run["outcome"] == MasterRunStatus.WAITING_EXTERNAL_CODEX.value else None))).model_dump(mode="json")

    def report(self, run_id: str) -> dict:
        report = self.registry.master_report(run_id)
        if not report: raise RegistryError(f"final report not found: {run_id}")
        return report["report_json"]

    def artifacts(self, run_id: str) -> list[dict]:
        return self.registry.master_artifacts(run_id)

    def implementation(self, run_id: str) -> dict:
        manifest = self.registry.get_implementation_manifest(run_id)
        if not manifest: raise RegistryError(f"implementation manifest not found: {run_id}")
        verify_artifact(manifest["manifest_path"], manifest["manifest_hash"])
        return manifest

    def worktree(self, run_id: str) -> dict:
        result = self.registry.get_worktree_metadata(run_id)
        if not result: raise RegistryError(f"worktree metadata not found: {run_id}")
        return result

    def verify_data(self, run_id: str) -> list[dict]:
        run = self.registry.get_master_run(run_id); state = run["resume_state_json"]; spec = self.registry.get_specification(state["strategy_id"])
        adapter, spec = self._real_adapter(run_id)
        return [item.model_dump(mode="json") for item in adapter.data_availability(spec)]

    def journal(self, run_id: str) -> list[dict]:
        return self.registry.master_journal(run_id)

    def cancel(self, run_id: str, reason: str = "cancelled by operator") -> dict:
        self.registry.add_master_journal(run_id, self.registry.get_master_run(run_id)["current_step"], "CANCELLED", {"reason": reason})
        self._set_step(run_id, MasterStep.COMPLETED, MasterRunStatus.ABORTED, cancel_reason=reason)
        return self.status(run_id)
