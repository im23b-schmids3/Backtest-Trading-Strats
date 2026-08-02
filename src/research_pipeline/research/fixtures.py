from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config.defaults import DEFAULT_BUDGETS
from ..controller.pipeline_controller import PipelineController
from ..enums import PipelineState
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.splits import SplitDefinition, SplitWindow, calculate_split_hash
from ..schemas.strategy_spec import ParameterFamily, StrategySpec, calculate_specification_hash
from ..verification.fixtures import make_fixture
from ..verification.services import VerificationService


def make_phase_c_spec(strategy_id: str = "phase-c-fixture", version: str = "phase-c-1", markets: list[str] | None = None) -> StrategySpec:
    raw = {"strategy_id": strategy_id, "version": version, "name": "Phase C synthetic fixture", "description": "Deterministic synthetic Phase C fixture.", "hypothesis": "A local parameter neighborhood is stable.", "strategy_family": "synthetic", "markets": markets or ["TEST"], "timeframes": ["1h"], "long_rules": ["fixture long"], "short_rules": ["fixture short"], "entry_logic": "fixture entry", "initial_stop_logic": "fixture stop", "exit_logic": "fixture exit", "session_assumptions": ["UTC"], "baseline_parameters": {"entry_depth": 5, "stop_distance": 3}, "parameter_families": [ParameterFamily(name="entry_depth", description="entry neighborhood", baseline_value=5, value_type="integer", allowed_min=1, allowed_max=9, optimization_order=1, maximum_rounds=3, mutable=True, hypothesis_relevance="entry sensitivity"), ParameterFamily(name="stop_distance", description="stop neighborhood", baseline_value=3, value_type="integer", allowed_min=1, allowed_max=5, optimization_order=2, maximum_rounds=3, mutable=True, hypothesis_relevance="loss containment")], "invariants": ["synthetic only"], "required_data": ["synthetic candles"], "known_limitations": ["fixture only"], "status": "DRAFT", "created_at": datetime.now(timezone.utc), "approved_at": None, "specification_hash": "pending"}
    candidate = StrategySpec.model_construct(**raw)
    raw["specification_hash"] = calculate_specification_hash(candidate)
    return StrategySpec.model_validate(raw)


def make_phase_c_split() -> SplitDefinition:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    raw = {"dataset_identifier": "synthetic-phase-c", "source_data_hash": "dataset-phase-c", "start_timestamp": start, "end_timestamp": start + timedelta(days=1000), "training_boundaries": SplitWindow(start_timestamp=start, end_timestamp=start + timedelta(days=500)), "validation_boundaries": SplitWindow(start_timestamp=start + timedelta(days=500), end_timestamp=start + timedelta(days=750)), "holdout_boundaries": SplitWindow(start_timestamp=start + timedelta(days=750), end_timestamp=start + timedelta(days=1000)), "created_timestamp": datetime.now(timezone.utc), "split_hash": "pending"}
    candidate = SplitDefinition.model_construct(**raw)
    raw["split_hash"] = calculate_split_hash(candidate)
    return SplitDefinition.model_validate(raw)


def prepare_phase_c_fixture(registry_path: str | Path, repository_root: str | Path, strategy_id: str = "phase-c-fixture", scenario: str = "strong-stable", markets: list[str] | None = None) -> dict:
    registry = Registry(Database(registry_path)); controller = PipelineController(registry); spec = make_phase_c_spec(strategy_id, markets=markets)
    try: controller.register_strategy(spec, str(Path(repository_root) / "phase-c-fixture.yaml"), DEFAULT_BUDGETS)
    except Exception:
        pass
    strategy = registry.get_strategy(strategy_id)
    if strategy["current_phase"] == PipelineState.STRATEGY_DRAFT.value: controller.submit_specification(strategy_id)
    if registry.get_strategy(strategy_id)["current_phase"] == PipelineState.WAITING_FOR_SPEC_APPROVAL.value: controller.approve_specification(strategy_id)
    if registry.get_strategy(strategy_id)["current_phase"] == PipelineState.IMPLEMENTATION.value: controller.transition(strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "synthetic fixture implementation verified")
    if not registry.has_verified_verification(strategy_id):
        manifest = make_fixture(Path(repository_root) / "research_runs" / strategy_id / "fixture-b5", strategy_id, spec.version)
        VerificationService(registry_path).run(strategy_id, manifest)
    if registry.get_split(strategy_id) is None: controller.create_split(strategy_id, make_phase_c_split())
    return {"strategy_id": strategy_id, "version": spec.version, "scenario": scenario, "registry_path": str(Path(registry_path).resolve()), "repository_root": str(Path(repository_root).resolve()), "current_phase": registry.get_strategy(strategy_id)["current_phase"]}


def run_phase_c_dry_run(registry_path: str | Path, repository_root: str | Path, strategy_id: str, scenario: str = "strong-stable", markets: list[str] | None = None) -> dict:
    """Run only the deterministic synthetic fixture; never touches strategy code."""
    from .services import PhaseCService

    prepare_phase_c_fixture(registry_path, repository_root, strategy_id, scenario, markets=markets)
    service = PhaseCService(registry_path, repository_root=repository_root, scenario=scenario)
    service.start(strategy_id, f"dry-run-{strategy_id}-{scenario}")
    service.run_baseline(strategy_id)
    edge = service.evaluate_edge(strategy_id)
    if edge["decision"] == "CONTINUE":
        for _ in range(6):
            decision = service.analyze(strategy_id)
            if decision.decision == "FREEZE_CANDIDATE":
                break
            proposal = service.propose_round(decision)
            round_result = service.run_round(strategy_id, proposal)
            service.review_round(strategy_id, round_result.round_id)
            if round_result.selected_value is None:
                break
            service.freeze_family(strategy_id, round_result.round_id)
        if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.PARAMETER_RESEARCH.value:
            service.freeze_candidate(strategy_id)
            service.run_walk_forward(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.HOLDOUT.value:
                service.run_holdout(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.STRESS_TESTS.value:
                service.run_stress(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.THROUGHPUT.value:
                service.run_throughput(strategy_id)
            if service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.FINAL_REVIEW.value:
                service.final_review(strategy_id)
    status = service.status(strategy_id)
    state = status["strategy"]["current_phase"]
    return {"strategy_id": strategy_id, "scenario": scenario, "final_state": state, "holdout_accesses": service.registry.count_holdout_accesses(strategy_id), "journal_entries": len(service.journal(strategy_id)), "no_optimization_after_holdout": True, "status": status}
