from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_pipeline.config.defaults import DEFAULT_BUDGETS
from research_pipeline.controller.pipeline_controller import PipelineController
from research_pipeline.enums import PipelineState
from research_pipeline.errors import HoldoutAccessError, SpecificationValidationError
from research_pipeline.research.models import AnalystDecision, ResearchClassification
from research_pipeline.research.services import PhaseCService
from research_pipeline.research.synthetic_adapter import SyntheticFixtureAdapter
from research_pipeline.registry.database import Database
from research_pipeline.registry.repositories import Registry
from research_pipeline.schemas.splits import SplitDefinition, SplitWindow, calculate_split_hash
from research_pipeline.schemas.strategy_spec import ParameterFamily, StrategySpec, calculate_specification_hash
from research_pipeline.verification.fixtures import make_fixture
from research_pipeline.verification.services import VerificationService


def make_spec() -> StrategySpec:
    raw = {
        "strategy_id": "phase-c-fixture", "version": "phase-c-1", "name": "Phase C fixture",
        "description": "Synthetic deterministic Phase C strategy fixture.", "hypothesis": "A local parameter neighborhood is stable.",
        "strategy_family": "synthetic", "markets": ["TEST"], "timeframes": ["1h"],
        "long_rules": ["fixture long"], "short_rules": ["fixture short"], "entry_logic": "fixture entry",
        "initial_stop_logic": "fixture stop", "exit_logic": "fixture exit", "session_assumptions": ["UTC"],
        "baseline_parameters": {"entry_depth": 5, "stop_distance": 3},
        "parameter_families": [
            ParameterFamily(name="entry_depth", description="entry neighborhood", baseline_value=5, value_type="integer", allowed_min=1, allowed_max=9, optimization_order=1, maximum_rounds=3, mutable=True, hypothesis_relevance="entry sensitivity"),
            ParameterFamily(name="stop_distance", description="stop neighborhood", baseline_value=3, value_type="integer", allowed_min=1, allowed_max=5, optimization_order=2, maximum_rounds=3, mutable=True, hypothesis_relevance="loss containment"),
        ],
        "invariants": ["synthetic only"], "required_data": ["synthetic candles"], "known_limitations": ["fixture"],
        "created_at": datetime.now(timezone.utc), "approved_at": None, "status": "DRAFT", "specification_hash": "pending",
    }
    candidate = StrategySpec.model_construct(**raw)
    raw["specification_hash"] = calculate_specification_hash(candidate)
    return StrategySpec.model_validate(raw)


def make_split() -> SplitDefinition:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    raw = {"dataset_identifier": "synthetic-phase-c", "source_data_hash": "dataset-phase-c", "start_timestamp": start, "end_timestamp": start + timedelta(days=1000),
        "training_boundaries": {"start_timestamp": start, "end_timestamp": start + timedelta(days=500)},
        "validation_boundaries": {"start_timestamp": start + timedelta(days=500), "end_timestamp": start + timedelta(days=750)},
        "holdout_boundaries": {"start_timestamp": start + timedelta(days=750), "end_timestamp": start + timedelta(days=1000)},
        "created_timestamp": datetime.now(timezone.utc), "split_hash": "pending"}
    candidate = SplitDefinition.model_construct(**raw)
    raw["split_hash"] = calculate_split_hash(candidate)
    return SplitDefinition.model_validate(raw)


def setup_phase_c(tmp_path: Path, scenario: str = "strong-stable") -> tuple[PhaseCService, str]:
    registry = Registry(Database(tmp_path / "registry.sqlite3"))
    controller = PipelineController(registry)
    spec = make_spec()
    controller.register_strategy(spec, str(tmp_path / "spec.yaml"), DEFAULT_BUDGETS)
    controller.submit_specification(spec.strategy_id)
    controller.approve_specification(spec.strategy_id)
    controller.transition(spec.strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "fixture implementation verified")
    manifest = make_fixture(tmp_path / "implementation-b5", spec.strategy_id, spec.version)
    VerificationService(tmp_path / "registry.sqlite3").run(spec.strategy_id, manifest)
    controller.create_split(spec.strategy_id, make_split())
    service = PhaseCService(tmp_path / "registry.sqlite3", adapter=SyntheticFixtureAdapter(scenario), repository_root=tmp_path, scenario=scenario)
    return service, spec.strategy_id


def prepare_research(service: PhaseCService, strategy_id: str) -> None:
    service.start(strategy_id)
    service.run_baseline(strategy_id)
    service.evaluate_edge(strategy_id)


def run_one_round(service: PhaseCService, strategy_id: str):
    decision = service.analyze(strategy_id)
    assert decision.decision == "CONTINUE_PARAMETER_RESEARCH"
    proposal = service.propose_round(decision)
    return decision, proposal, service.run_round(strategy_id, proposal)


def test_baseline_requires_b5_and_uses_approved_baseline(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    service.start(strategy_id)
    result = service.run_baseline(strategy_id)
    assert result.verification_outcome == "VERIFIED"
    assert result.artifact.metrics["completed_trades"] == 60
    assert service.registry.get_baseline(strategy_id) is not None


def test_edge_gate_continue_reject_and_insufficient(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path, "strong-stable")
    prepare_research(service, strategy_id)
    assert service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.PARAMETER_RESEARCH.value

    no_edge, no_edge_id = setup_phase_c(tmp_path / "no-edge", "no-edge")
    no_edge.start(no_edge_id); no_edge.run_baseline(no_edge_id)
    result = no_edge.evaluate_edge(no_edge_id)
    assert result["decision"] == "REJECT"
    assert no_edge.registry.get_strategy(no_edge_id)["current_phase"] == PipelineState.REJECTED.value
    assert no_edge.registry.get_research_json("research_final_reviews", no_edge_id)["classification"] == ResearchClassification.REJECTED_NO_EDGE.value

    insufficient, insufficient_id = setup_phase_c(tmp_path / "insufficient", "insufficient-trade")
    insufficient.start(insufficient_id); insufficient.run_baseline(insufficient_id)
    assert insufficient.evaluate_edge(insufficient_id)["decision"] == "INSUFFICIENT_EVIDENCE"


def test_only_mutable_family_and_one_family_per_round(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    prepare_research(service, strategy_id)
    decision = service.analyze(strategy_id)
    assert decision.selected_parameter_family == "entry_depth"
    with pytest.raises(SpecificationValidationError):
        service.propose_round(decision.model_copy(update={"proposed_values": [1, 2], "selected_parameter_family": "not-declared"}))
    _, proposal, result = run_one_round(service, strategy_id)
    assert len({result.family}) == 1
    assert all(1 <= value <= 9 for value in proposal.proposed_values)


def test_citations_are_verified_independently(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    prepare_research(service, strategy_id)
    decision = service.analyze(strategy_id)
    bad = decision.metrics_cited[0].model_copy(update={"value": 999})
    with pytest.raises(SpecificationValidationError):
        service.validate_decision(decision.model_copy(update={"metrics_cited": [bad, *decision.metrics_cited[1:]]}))


def test_plateau_is_selected_and_isolated_maximum_is_vetoed(tmp_path: Path) -> None:
    plateau, plateau_id = setup_phase_c(tmp_path / "plateau", "stable-plateau")
    prepare_research(plateau, plateau_id)
    _, _, plateau_result = run_one_round(plateau, plateau_id)
    assert plateau_result.review.selected_value is not None
    assert plateau_result.review.isolated_maximum_risk is False

    isolated, isolated_id = setup_phase_c(tmp_path / "isolated", "isolated-maximum")
    prepare_research(isolated, isolated_id)
    _, _, isolated_result = run_one_round(isolated, isolated_id)
    assert isolated_result.review.isolated_maximum_risk is True
    assert isolated_result.selected_value is None


def test_family_and_candidate_freeze_block_later_parameter_changes(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    prepare_research(service, strategy_id)
    _, proposal, result = run_one_round(service, strategy_id)
    service.freeze_family(strategy_id, result.round_id)
    candidate = service.freeze_candidate(strategy_id)
    assert candidate.candidate_hash
    with pytest.raises(Exception):
        service.analyze(strategy_id)


def test_full_strong_fixture_reaches_accepted_final_review_and_journal(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    prepare_research(service, strategy_id)
    _, _, result = run_one_round(service, strategy_id)
    service.freeze_family(strategy_id, result.round_id)
    service.freeze_candidate(strategy_id)
    assert service.run_walk_forward(strategy_id).status == "PASS"
    assert service.run_holdout(strategy_id).access_count == 1
    service.run_stress(strategy_id)
    service.run_throughput(strategy_id)
    final = service.final_review(strategy_id)
    assert final.classification == ResearchClassification.ACCEPTED_STANDALONE
    assert service.registry.get_strategy(strategy_id)["current_phase"] == PipelineState.ACCEPTED.value
    assert len(service.journal(strategy_id)) >= 8


def test_holdout_and_stress_rules_are_structural(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path)
    prepare_research(service, strategy_id)
    _, _, result = run_one_round(service, strategy_id); service.freeze_family(strategy_id, result.round_id); service.freeze_candidate(strategy_id); service.run_walk_forward(strategy_id)
    service.run_holdout(strategy_id)
    with pytest.raises(HoldoutAccessError):
        service.controller.open_holdout(strategy_id, "reuse")


def test_low_frequency_fixture_is_portfolio_component(tmp_path: Path) -> None:
    service, strategy_id = setup_phase_c(tmp_path, "portfolio-component")
    prepare_research(service, strategy_id)
    _, _, result = run_one_round(service, strategy_id); service.freeze_family(strategy_id, result.round_id); service.freeze_candidate(strategy_id); service.run_walk_forward(strategy_id); service.run_holdout(strategy_id); service.run_stress(strategy_id); throughput = service.run_throughput(strategy_id)
    assert throughput.classification == "PORTFOLIO_COMPONENT_ONLY"
    assert service.final_review(strategy_id).classification == ResearchClassification.ACCEPTED_PORTFOLIO_COMPONENT

