from __future__ import annotations

from ..enums import PipelineState, TERMINAL_STATES
from ..errors import InvalidTransitionError, TerminalStateError


# Kept as data so it is easy to audit and test.  No transition is inferred at runtime.
LEGAL_TRANSITIONS: dict[PipelineState, frozenset[PipelineState]] = {
    PipelineState.STRATEGY_DRAFT: frozenset({PipelineState.WAITING_FOR_SPEC_APPROVAL}),
    PipelineState.WAITING_FOR_SPEC_APPROVAL: frozenset({PipelineState.IMPLEMENTATION, PipelineState.REJECTED, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.IMPLEMENTATION: frozenset({PipelineState.IMPLEMENTATION_VERIFICATION, PipelineState.TECHNICAL_FAILURE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.IMPLEMENTATION_VERIFICATION: frozenset({PipelineState.TECHNICAL_INTEGRITY_VERIFICATION, PipelineState.BASELINE_BACKTEST, PipelineState.TECHNICAL_FAILURE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.TECHNICAL_INTEGRITY_VERIFICATION: frozenset({PipelineState.BASELINE_BACKTEST, PipelineState.TECHNICAL_REPAIR_REQUIRED, PipelineState.MANUAL_REVIEW_REQUIRED, PipelineState.INSUFFICIENT_DIAGNOSTIC_DATA, PipelineState.TECHNICAL_FAILURE}),
    PipelineState.TECHNICAL_REPAIR_REQUIRED: frozenset({PipelineState.TECHNICAL_INTEGRITY_VERIFICATION, PipelineState.MANUAL_REVIEW_REQUIRED, PipelineState.TECHNICAL_FAILURE}),
    PipelineState.BASELINE_BACKTEST: frozenset({PipelineState.EDGE_GATE, PipelineState.TECHNICAL_FAILURE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.EDGE_GATE: frozenset({PipelineState.PARAMETER_RESEARCH, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.PARAMETER_RESEARCH: frozenset({PipelineState.CANDIDATE_FREEZE, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.TECHNICAL_FAILURE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.CANDIDATE_FREEZE: frozenset({PipelineState.WALK_FORWARD, PipelineState.TECHNICAL_FAILURE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.WALK_FORWARD: frozenset({PipelineState.HOLDOUT, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.HOLDOUT: frozenset({PipelineState.STRESS_TESTS, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.STRESS_TESTS: frozenset({PipelineState.THROUGHPUT, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.THROUGHPUT: frozenset({PipelineState.RISK_SIZING, PipelineState.FINAL_REVIEW, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.RISK_SIZING: frozenset({PipelineState.PROP_SIMULATION, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.PROP_SIMULATION: frozenset({PipelineState.MULTI_STRATEGY_PORTFOLIO, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.MULTI_STRATEGY_PORTFOLIO: frozenset({PipelineState.FINAL_REVIEW, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.FINAL_REVIEW: frozenset({PipelineState.ACCEPTED, PipelineState.REJECTED, PipelineState.INSUFFICIENT_EVIDENCE, PipelineState.MANUAL_REVIEW_REQUIRED}),
    PipelineState.ACCEPTED: frozenset(),
    PipelineState.REJECTED: frozenset(),
    PipelineState.INSUFFICIENT_EVIDENCE: frozenset(),
    PipelineState.TECHNICAL_FAILURE: frozenset(),
    PipelineState.MANUAL_REVIEW_REQUIRED: frozenset(),
}


class StateMachine:
    @staticmethod
    def validate_transition(old_state: PipelineState, new_state: PipelineState) -> None:
        if old_state in TERMINAL_STATES:
            raise TerminalStateError(f"terminal state {old_state} cannot transition")
        if new_state not in LEGAL_TRANSITIONS.get(old_state, frozenset()):
            raise InvalidTransitionError(f"illegal transition: {old_state} -> {new_state}")
