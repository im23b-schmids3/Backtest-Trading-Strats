from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from research_pipeline.controller.gate_evaluator import GateEvaluator
from research_pipeline.controller.budget_enforcer import BudgetEnforcer, BudgetRequest
from research_pipeline.controller.state_machine import StateMachine
from research_pipeline.enums import GateOutcomeStatus, PipelineState
from research_pipeline.errors import (
    BudgetExceededError, HoldoutAccessError, ImmutableSpecificationError,
    InvalidTransitionError, SplitConflictError, TerminalStateError,
)
from research_pipeline.registry.database import Database
from research_pipeline.registry.repositories import Registry
from research_pipeline.schemas.budgets import BudgetUsage, ResearchBudget
from research_pipeline.schemas.decisions import DecisionRecord
from research_pipeline.schemas.gates import Comparison, GateDefinition
from research_pipeline.schemas.splits import SplitDefinition, calculate_split_hash
from research_pipeline.schemas.strategy_spec import StrategySpec, load_strategy_spec


ROOT = Path(__file__).parents[2]


@pytest.fixture
def specification():
    return load_strategy_spec(ROOT / "examples/research_pipeline/fibonacci_compatibility.yaml")


@pytest.fixture
def controller(tmp_path):
    from research_pipeline.controller.pipeline_controller import PipelineController
    from research_pipeline.registry.database import Database
    from research_pipeline.registry.repositories import Registry
    return PipelineController(Registry(Database(tmp_path / "registry.sqlite3")))


@pytest.fixture
def registered(controller, specification):
    controller.register_strategy(specification, "fixture.yaml")
    return controller


def split_definition():
    raw = {
        "dataset_identifier": "mock-dataset",
        "source_data_hash": "source-hash-1",
        "start_timestamp": "2020-01-01T00:00:00+00:00",
        "end_timestamp": "2024-01-01T00:00:00+00:00",
        "training_boundaries": {"start_timestamp": "2020-01-01T00:00:00+00:00", "end_timestamp": "2022-01-01T00:00:00+00:00"},
        "validation_boundaries": {"start_timestamp": "2022-01-01T00:00:00+00:00", "end_timestamp": "2023-01-01T00:00:00+00:00"},
        "holdout_boundaries": {"start_timestamp": "2023-01-01T00:00:00+00:00", "end_timestamp": "2024-01-01T00:00:00+00:00"},
        "created_timestamp": "2026-07-16T00:00:00+00:00",
    }
    raw["split_hash"] = calculate_split_hash(raw)
    return SplitDefinition.model_validate(raw)


def reach_parameter_research(controller):
    controller.submit_specification("fibonacci-compatibility")
    controller.approve_specification("fibonacci-compatibility")
    for state in (PipelineState.IMPLEMENTATION_VERIFICATION, PipelineState.BASELINE_BACKTEST, PipelineState.EDGE_GATE):
        controller.transition("fibonacci-compatibility", state, "mocked phase")
    controller.create_split("fibonacci-compatibility", split_definition())
    controller.transition("fibonacci-compatibility", PipelineState.PARAMETER_RESEARCH, "split locked")


def test_valid_strategy_yaml_loads_correctly(specification):
    assert specification.strategy_id == "fibonacci-compatibility"
    assert specification.version == "phase-a-1"
    assert specification.parameter_families[0].mutable is True


def test_invalid_strategy_yaml_is_rejected(tmp_path, specification):
    raw = specification.model_dump(mode="json")
    raw["strategy_id"] = "not safe/id"
    raw["specification_hash"] = "bad"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_strategy_spec(path)


def test_approved_specification_is_immutable(registered):
    registered.submit_specification("fibonacci-compatibility")
    registered.approve_specification("fibonacci-compatibility")
    spec = registered.registry.get_specification("fibonacci-compatibility")
    with pytest.raises(ImmutableSpecificationError):
        registered.registry.approve_specification("fibonacci-compatibility", spec.version, spec)


def test_legal_state_transitions_succeed(registered):
    registered.submit_specification("fibonacci-compatibility")
    assert registered.status("fibonacci-compatibility")["current_phase"] == PipelineState.WAITING_FOR_SPEC_APPROVAL
    registered.approve_specification("fibonacci-compatibility")
    assert registered.status("fibonacci-compatibility")["current_phase"] == PipelineState.IMPLEMENTATION


def test_illegal_transition_fails(registered):
    with pytest.raises(InvalidTransitionError):
        registered.transition("fibonacci-compatibility", PipelineState.HOLDOUT, "skip")


def test_terminal_states_cannot_transition():
    with pytest.raises(TerminalStateError):
        StateMachine.validate_transition(PipelineState.REJECTED, PipelineState.IMPLEMENTATION)


def test_budget_usage_is_recorded(registered):
    registered.consume_budget("fibonacci-compatibility", backtests=5)
    assert registered.registry.get_budget("fibonacci-compatibility")["usage"]["total_backtests"] == 5


def test_budget_overrun_is_blocked_before_execution(registered):
    before = registered.registry.get_budget("fibonacci-compatibility")["usage"]
    with pytest.raises(BudgetExceededError):
        registered.consume_budget("fibonacci-compatibility", backtests=501)
    assert registered.registry.get_budget("fibonacci-compatibility")["usage"] == before


def test_one_parameter_family_per_round_is_enforced(registered):
    enforcer = BudgetEnforcer()
    usage = enforcer.check(ResearchBudget(), BudgetUsage(), BudgetRequest(family="entry", rounds=1, values=1, round_id=1))
    with pytest.raises(BudgetExceededError):
        enforcer.check(ResearchBudget(), usage, BudgetRequest(family="exit", rounds=1, values=1, round_id=1))


def test_split_definitions_are_deterministic():
    first = split_definition()
    second = split_definition()
    assert first.split_hash == second.split_hash
    assert first.deterministic_payload() == second.deterministic_payload()


def test_split_changes_are_detected(registered):
    first = split_definition()
    registered.create_split("fibonacci-compatibility", first)
    changed = first.model_copy(update={"source_data_hash": "different"}, deep=True)
    changed = changed.model_copy(update={"split_hash": calculate_split_hash(changed)})
    with pytest.raises(SplitConflictError):
        registered.create_split("fibonacci-compatibility", changed)


def test_holdout_cannot_open_before_holdout_phase(registered):
    with pytest.raises(HoldoutAccessError):
        registered.open_holdout("fibonacci-compatibility", "too early")


def test_holdout_opens_only_once(registered):
    reach_parameter_research(registered)
    registered.transition("fibonacci-compatibility", PipelineState.CANDIDATE_FREEZE, "freeze")
    registered.transition("fibonacci-compatibility", PipelineState.WALK_FORWARD, "mock")
    registered.transition("fibonacci-compatibility", PipelineState.HOLDOUT, "mock")
    registered.open_holdout("fibonacci-compatibility", "final validation", "holdout-hash")
    with pytest.raises(HoldoutAccessError):
        registered.open_holdout("fibonacci-compatibility", "repeat", "holdout-hash")
    with pytest.raises(HoldoutAccessError):
        registered.transition("fibonacci-compatibility", PipelineState.PARAMETER_RESEARCH, "resume")


def test_sqlite_initialization_is_idempotent(tmp_path):
    path = tmp_path / "registry.sqlite3"
    Registry(Database(path)); Registry(Database(path))
    assert path.exists()


def test_registry_state_survives_process_restart(registered, tmp_path, specification):
    registered.submit_specification("fibonacci-compatibility")
    reopened = Registry(Database(tmp_path / "registry.sqlite3"))
    assert reopened.get_strategy("fibonacci-compatibility")["current_phase"] == PipelineState.WAITING_FOR_SPEC_APPROVAL


def test_structured_decision_json_validates(registered):
    decision = DecisionRecord(phase=PipelineState.EDGE_GATE, strategy_id="fibonacci-compatibility", decision="CONTINUE", confidence=.8,
        evidence=["mock evidence"], blocking_issues=[], overfitting_risk=.2, files_inspected=["mock.json"], metrics_cited=["profit_factor"], rationale="Evidence is sufficient for the next deterministic phase.")
    assert registered.record_decision("fibonacci-compatibility", decision) > 0


def test_unsupported_decision_values_fail():
    with pytest.raises(ValidationError):
        DecisionRecord(phase=PipelineState.EDGE_GATE, strategy_id="x", decision="MAYBE", confidence=.5, evidence=[], blocking_issues=[], overfitting_risk=.5, files_inspected=[], metrics_cited=[], rationale="x")


def test_gate_pass_fail_and_insufficient_evidence():
    gate = GateDefinition(name="pf", category="baseline", metric="profit_factor", threshold=1.2, comparison=Comparison.GREATER_EQUAL)
    evaluator = GateEvaluator()
    assert evaluator.evaluate(gate, {"profit_factor": 1.3}).status == GateOutcomeStatus.PASS
    assert evaluator.evaluate(gate, {"profit_factor": 1.0}).status == GateOutcomeStatus.FAIL
    assert evaluator.evaluate(gate, {}).status == GateOutcomeStatus.INSUFFICIENT_EVIDENCE


def test_fibonacci_fixture_registers_successfully(registered):
    assert registered.status("fibonacci-compatibility")["specification_hash"]


def test_invalid_transition_history_is_rejected(registered):
    strategy = registered.registry.get_strategy("fibonacci-compatibility")
    with pytest.raises(InvalidTransitionError):
        registered.registry.transition("fibonacci-compatibility", strategy["version"], strategy["current_phase"], PipelineState.HOLDOUT.value, "corrupt history")
