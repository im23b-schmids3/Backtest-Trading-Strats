from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ..enums import ApprovalStatus, PipelineState
from ..errors import HoldoutAccessError, SpecificationValidationError
from ..registry.repositories import Registry
from ..schemas.budgets import BudgetUsage
from ..schemas.decisions import DecisionRecord
from ..schemas.gates import GateDefinition, GateOutcome, GateSet
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from .budget_enforcer import BudgetEnforcer, BudgetRequest
from .gate_evaluator import GateEvaluator
from .holdout_lock import HoldoutLock
from .state_machine import StateMachine
from .transition_rules import requires_split_for_transition


class PipelineController:
    """Stateful orchestration around deterministic registry operations.

    Future runners can be attached at phase boundaries.  This class never
    invokes a backtester, market-data provider, Codex, or an autonomous agent.
    """

    def __init__(self, registry: Registry):
        self.registry = registry
        self.budget_enforcer = BudgetEnforcer()
        self.gate_evaluator = GateEvaluator()
        self.holdout_lock = HoldoutLock(registry)

    def register_strategy(self, specification: StrategySpec, specification_path: str, budget=None) -> dict:
        from ..config.defaults import DEFAULT_BUDGETS

        return self.registry.register_strategy(specification, specification_path, budget or DEFAULT_BUDGETS)

    def validate_specification(self, strategy_id: str) -> dict:
        try:
            spec = self.registry.get_specification(strategy_id)
            return {"valid": True, "strategy_id": spec.strategy_id, "specification_hash": spec.specification_hash, "errors": []}
        except (ValidationError, ValueError) as exc:
            raise SpecificationValidationError(str(exc)) from exc

    def submit_specification(self, strategy_id: str) -> dict:
        strategy = self.registry.get_strategy(strategy_id)
        self._transition(strategy, PipelineState.WAITING_FOR_SPEC_APPROVAL, "specification submitted for approval")
        self.registry.set_approval(strategy["strategy_id"], strategy["version"], ApprovalStatus.SUBMITTED)
        return self.registry.get_strategy(strategy_id)

    def approve_specification(self, strategy_id: str) -> dict:
        strategy = self.registry.get_strategy(strategy_id)
        if strategy["current_phase"] != PipelineState.WAITING_FOR_SPEC_APPROVAL.value:
            raise SpecificationValidationError("only a submitted specification can be approved")
        specification = self.registry.get_specification(strategy_id)
        approved = specification.approved_copy(datetime.now(timezone.utc))
        self.registry.approve_specification(strategy["strategy_id"], strategy["version"], approved)
        self._transition(strategy, PipelineState.IMPLEMENTATION, "specification approved")
        return self.registry.get_strategy(strategy_id)

    def transition(self, strategy_id: str, new_state: PipelineState | str, reason: str) -> dict:
        strategy = self.registry.get_strategy(strategy_id)
        state = PipelineState(new_state)
        if state == PipelineState.BASELINE_BACKTEST and strategy["current_phase"] == PipelineState.TECHNICAL_INTEGRITY_VERIFICATION.value and not self.registry.has_verified_verification(strategy_id, strategy["version"]):
            raise SpecificationValidationError("BASELINE_BACKTEST requires a persisted VERIFIED technical-integrity result")
        # Preserve the Phase A test fixture's historical direct path while all
        # Phase B+ strategy versions use the mandatory B.5 gate.
        if state == PipelineState.BASELINE_BACKTEST and strategy["current_phase"] == PipelineState.IMPLEMENTATION_VERIFICATION.value and not strategy["version"].startswith("phase-a-"):
            raise SpecificationValidationError("new implementations must enter TECHNICAL_INTEGRITY_VERIFICATION before baseline research")
        if requires_split_for_transition(state) and self.registry.get_split(strategy_id) is None:
            raise SpecificationValidationError("a split definition must exist before parameter research starts")
        if state == PipelineState.PARAMETER_RESEARCH and strategy["parameters_frozen"]:
            raise HoldoutAccessError("parameter research is frozen after holdout access")
        self._transition(strategy, state, reason)
        if state == PipelineState.CANDIDATE_FREEZE:
            self.registry.freeze_parameter_families(strategy_id, strategy["version"])
        return self.registry.get_strategy(strategy_id)

    def _transition(self, strategy: dict, state: PipelineState, reason: str) -> None:
        StateMachine.validate_transition(PipelineState(strategy["current_phase"]), state)
        self.registry.transition(strategy["strategy_id"], strategy["version"], strategy["current_phase"], state.value, reason)

    def evaluate_gates(self, strategy_id: str, gates: GateSet | list[GateDefinition], metrics: dict[str, Any], source_file: str = "") -> list[GateOutcome]:
        return self.gate_evaluator.evaluate_set(gates, metrics, source_file)

    def consume_budget(self, strategy_id: str, *, backtests: int = 0, family: str | None = None, rounds: int = 0, values: int = 0,
                       codex_repairs: int = 0, runtime_minutes: int = 0, report_size_mb: float = 0.0, research_round: int | None = None) -> BudgetUsage:
        strategy = self.registry.get_strategy(strategy_id)
        spec = self.registry.get_specification(strategy_id)
        if family:
            families = {item.name: item for item in spec.parameter_families}
            if family not in families:
                raise SpecificationValidationError(f"unknown parameter family: {family}")
            if not families[family].mutable:
                raise SpecificationValidationError(f"parameter family is immutable: {family}")
            if strategy["parameters_frozen"]:
                raise HoldoutAccessError("parameter families are frozen")
        budget = self.registry.get_budget(strategy_id, strategy["version"])
        usage = BudgetUsage.model_validate(budget["usage"])
        next_usage = self.budget_enforcer.check(budget_from_dict(budget["limits"]), usage, BudgetRequest(backtests=backtests, family=family, rounds=rounds, values=values, codex_repairs=codex_repairs, runtime_minutes=runtime_minutes, report_size_mb=report_size_mb, round_id=research_round))
        self.registry.update_budget_usage(strategy["strategy_id"], strategy["version"], next_usage)
        return next_usage

    def create_split(self, strategy_id: str, split: SplitDefinition) -> None:
        self.registry.create_split(strategy_id, None, split)

    def invalidate_split(self, strategy_id: str, reason: str) -> None:
        self.registry.invalidate_split(strategy_id, reason)

    def open_holdout(self, strategy_id: str, reason: str, dataset_hash: str | None = None) -> dict:
        return self.holdout_lock.open(strategy_id, reason, dataset_hash)

    def holdout_status(self, strategy_id: str) -> dict:
        return self.holdout_lock.status(strategy_id)

    def record_decision(self, strategy_id: str, decision: DecisionRecord) -> int:
        return self.registry.record_decision(strategy_id, None, decision)

    def store_experiment(self, strategy_id: str, **values) -> str:
        return self.registry.add_experiment(strategy_id, None, **values)

    def status(self, strategy_id: str) -> dict:
        return self.registry.get_strategy(strategy_id)


def budget_from_dict(values: dict) -> Any:
    from ..schemas.budgets import ResearchBudget

    return ResearchBudget.model_validate(values)
