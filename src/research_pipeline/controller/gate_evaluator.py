from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..enums import GateOutcomeStatus
from ..schemas.gates import Comparison, GateDefinition, GateOutcome, GateSet


def _compare(value: float, comparison: Comparison, threshold: float) -> bool:
    return {
        Comparison.GREATER_EQUAL: value >= threshold,
        Comparison.LESS_EQUAL: value <= threshold,
        Comparison.GREATER: value > threshold,
        Comparison.LESS: value < threshold,
        Comparison.EQUAL: value == threshold,
    }[comparison]


class GateEvaluator:
    def evaluate(self, gate: GateDefinition, metrics: Mapping[str, Any], source_file: str | None = None) -> GateOutcome:
        observed = metrics.get(gate.metric)
        source = source_file or gate.source_file
        if observed is None or not isinstance(observed, (int, float)) or isinstance(observed, bool):
            return GateOutcome(gate=gate.name, status=GateOutcomeStatus.INSUFFICIENT_EVIDENCE,
                metric=gate.metric, observed_value=observed, threshold=gate.threshold,
                comparison=gate.comparison, source_file=source,
                reason=f"metric {gate.metric!r} is absent or non-numeric")
        passed = _compare(float(observed), gate.comparison, gate.threshold)
        return GateOutcome(gate=gate.name, status=GateOutcomeStatus.PASS if passed else GateOutcomeStatus.FAIL,
            metric=gate.metric, observed_value=observed, threshold=gate.threshold,
            comparison=gate.comparison, source_file=source,
            reason=f"observed {observed} {gate.comparison} threshold {gate.threshold}" if passed else f"observed {observed} does not satisfy {gate.comparison} {gate.threshold}")

    def evaluate_set(self, gate_set: GateSet | Sequence[GateDefinition], metrics: Mapping[str, Any], source_file: str | None = None) -> list[GateOutcome]:
        gates = gate_set.gates if isinstance(gate_set, GateSet) else gate_set
        return [self.evaluate(gate, metrics, source_file) for gate in gates]

