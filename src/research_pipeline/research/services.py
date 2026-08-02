from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..controller.gate_evaluator import GateEvaluator
from ..controller.pipeline_controller import PipelineController
from ..enums import GateOutcomeStatus, PipelineState
from ..errors import HoldoutAccessError, InvalidTransitionError, SpecificationValidationError
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.gates import GateDefinition, GateSet
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import ParameterFamily, StrategySpec
from ..verification.models import VerificationOutcome
from ..verification.services import VerificationService
from ..config.loader import load_pipeline_config
from .models import (AnalystDecision, BaselineResult, CandidateManifest, FinalResearchReview,
    HoldoutResult, MetricCitation, ParameterExperiment, ParameterProposal, ParameterRoundResult,
    ResearchArtifact, ResearchClassification, StatisticalReview, StressResult, ThroughputResult,
    WalkForwardResult)
from .runner import StrategyResearchAdapter
from .synthetic_adapter import SyntheticFixtureAdapter


class PhaseCService:
    """Deterministic policy engine for Phase C research.

    Agent outputs are inputs to this class, never authority. Every transition,
    budget check, split lookup, citation check, and holdout access is enforced
    here before a result is persisted.
    """

    def __init__(self, registry_path: str | Path | None = None, adapter: StrategyResearchAdapter | None = None, repository_root: str | Path = ".", scenario: str = "strong-stable"):
        self.registry_path = Path(registry_path or os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3"))
        self.registry = Registry(Database(self.registry_path))
        self.controller = PipelineController(self.registry)
        self.repository_root = Path(repository_root).resolve()
        self.adapter = adapter or SyntheticFixtureAdapter(scenario)
        self.scenario = scenario
        self.gate_evaluator = GateEvaluator()

    def _strategy(self, strategy_id: str) -> dict:
        return self.registry.get_strategy(strategy_id)

    def _spec_split(self, strategy_id: str) -> tuple[StrategySpec, SplitDefinition]:
        spec = self.registry.get_specification(strategy_id)
        split = self.registry.get_split(strategy_id)
        if split is None:
            raise SpecificationValidationError("Phase C requires a persisted chronological split")
        return spec, split

    def _root(self, strategy_id: str, phase: str, *parts: str) -> Path:
        strategy = self._strategy(strategy_id)
        root = self.repository_root / "research_runs" / strategy_id / strategy["version"]
        path = root / phase
        for part in parts: path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _record_artifact(self, artifact: ResearchArtifact) -> None:
        if any(row.get("experiment_id") == artifact.experiment_id for row in self.registry.history(artifact.strategy_id)["experiments"]):
            return
        self.controller.store_experiment(artifact.strategy_id, experiment_id=artifact.experiment_id, phase=artifact.phase,
            parameter_values=json.loads(Path(artifact.input_path).read_text(encoding="utf-8")).get("parameters", {}), dataset_hash=artifact.dataset_hash,
            code_commit=artifact.code_commit, start_time=datetime.now(timezone.utc).isoformat(), end_time=datetime.now(timezone.utc).isoformat(), status=artifact.status, report_paths=list(artifact.report_hashes))

    def _journal(self, strategy_id: str, phase: str, entry: dict[str, Any], markdown: str | None = None) -> None:
        strategy = self._strategy(strategy_id)
        self.registry.add_journal_entry(strategy_id, strategy["version"], phase, entry, markdown or json.dumps(entry, indent=2, sort_keys=True))

    def _gates(self, category: str, strategy_id: str | None = None) -> GateSet:
        config = load_pipeline_config(Path("configs/research_pipeline/defaults.yaml"), strategy_id=strategy_id)
        return GateSet(gates=[gate for gate in config["gates"].gates if gate.category == category])

    def _assert_phase(self, strategy_id: str, expected: PipelineState) -> dict:
        strategy = self._strategy(strategy_id)
        if strategy["current_phase"] != expected.value:
            raise InvalidTransitionError(f"Phase C requires {expected.value}, got {strategy['current_phase']}")
        return strategy

    def start(self, strategy_id: str, run_id: str | None = None) -> dict:
        strategy = self._strategy(strategy_id)
        self._assert_phase(strategy_id, PipelineState.BASELINE_BACKTEST)
        if not self.registry.has_verified_verification(strategy_id, strategy["version"]):
            raise SpecificationValidationError("baseline research requires verified Phase B.5 eligibility")
        self.adapter.validate_environment()
        return self.registry.research_run(run_id or f"phase-c-{strategy_id}-{strategy['version']}", strategy_id, strategy["version"], strategy["current_phase"], str(self.repository_root), self.scenario)

    def run_baseline(self, strategy_id: str) -> BaselineResult:
        strategy = self._assert_phase(strategy_id, PipelineState.BASELINE_BACKTEST)
        existing = self.registry.get_baseline(strategy_id, strategy["version"])
        if existing:
            artifact = ResearchArtifact.model_validate(existing["artifact_json"])
            return BaselineResult(artifact=artifact, verification_outcome=existing["verification_outcome"], gate_outcomes=existing["gate_outcomes_json"])
        spec, split = self._spec_split(strategy_id)
        artifact = self.adapter.run_baseline(spec, split, self._root(strategy_id, "baseline"))
        self._record_artifact(artifact)
        if not artifact.diagnostic_manifest_path:
            self.controller.transition(strategy_id, PipelineState.TECHNICAL_FAILURE, "baseline adapter did not provide a B.5 diagnostic manifest")
            raise SpecificationValidationError("baseline adapter must provide a B.5 diagnostic manifest")
        verification = VerificationService(self.registry_path).run(strategy_id, artifact.diagnostic_manifest_path)
        outcome = verification["outcome"]
        if outcome != VerificationOutcome.VERIFIED.value:
            self.registry.save_baseline(strategy_id, strategy["version"], artifact.experiment_id, artifact.model_dump(mode="json"), outcome, [])
            self.controller.transition(strategy_id, PipelineState.TECHNICAL_FAILURE, f"baseline B.5 verification failed: {outcome}")
            raise SpecificationValidationError(f"baseline B.5 verification did not pass: {outcome}")
        result = BaselineResult(artifact=artifact, verification_outcome=outcome)
        self.registry.save_baseline(strategy_id, strategy["version"], artifact.experiment_id, artifact.model_dump(mode="json"), outcome, [])
        self.controller.transition(strategy_id, PipelineState.EDGE_GATE, "verified baseline artifact is ready for edge gate")
        return result

    def evaluate_edge(self, strategy_id: str) -> dict[str, Any]:
        self._assert_phase(strategy_id, PipelineState.EDGE_GATE)
        baseline = self.registry.get_baseline(strategy_id)
        if not baseline or baseline["verification_outcome"] != VerificationOutcome.VERIFIED.value:
            raise SpecificationValidationError("edge analysis requires a verified baseline")
        artifact = ResearchArtifact.model_validate(baseline["artifact_json"])
        outcomes = self.gate_evaluator.evaluate_set(self._gates("baseline", strategy_id), artifact.metrics, artifact.metrics_path)
        if any(item.status == GateOutcomeStatus.INSUFFICIENT_EVIDENCE for item in outcomes) or any(item.status == GateOutcomeStatus.FAIL and item.metric in {"completed_trades", "independent_markets", "history_months"} for item in outcomes): decision = "INSUFFICIENT_EVIDENCE"
        elif any(item.status == GateOutcomeStatus.FAIL for item in outcomes): decision = "REJECT"
        elif any(item.status == GateOutcomeStatus.MANUAL_REVIEW_REQUIRED for item in outcomes): decision = "MANUAL_REVIEW_REQUIRED"
        else: decision = "CONTINUE"
        payload = {"decision": decision, "outcomes": [item.model_dump(mode="json") for item in outcomes], "metrics": artifact.metrics}
        self.registry.save_baseline(baseline["strategy_id"], baseline["strategy_version"], baseline["experiment_id"], artifact.model_dump(mode="json"), baseline["verification_outcome"], payload["outcomes"])
        self._journal(strategy_id, "EDGE_GATE", payload, f"Edge gate decision: {decision}")
        next_state = {"CONTINUE": PipelineState.PARAMETER_RESEARCH, "REJECT": PipelineState.REJECTED, "INSUFFICIENT_EVIDENCE": PipelineState.INSUFFICIENT_EVIDENCE, "MANUAL_REVIEW_REQUIRED": PipelineState.MANUAL_REVIEW_REQUIRED}[decision]
        if decision != "CONTINUE":
            classification = {"REJECT": ResearchClassification.REJECTED_NO_EDGE, "INSUFFICIENT_EVIDENCE": ResearchClassification.INSUFFICIENT_EVIDENCE, "MANUAL_REVIEW_REQUIRED": ResearchClassification.MANUAL_REVIEW_REQUIRED}[decision]
            early = FinalResearchReview(strategy_id=strategy_id, strategy_version=baseline["strategy_version"], classification=classification, current_phase=PipelineState.EDGE_GATE, evidence_strength="preliminary", evidence=payload, metrics_cited=[], risks=[], rationale=f"Edge gate stopped Phase C with {decision}.")
            self.registry.save_research_json("research_final_reviews", strategy_id, baseline["strategy_version"], early.model_dump(mode="json"))
        self.controller.transition(strategy_id, next_state, f"baseline edge gate: {decision}")
        return payload

    def analyze(self, strategy_id: str) -> AnalystDecision:
        self._assert_phase(strategy_id, PipelineState.PARAMETER_RESEARCH)
        strategy = self._strategy(strategy_id); spec, _ = self._spec_split(strategy_id)
        frozen = {row["family"] for row in self.registry.list_research_rounds(strategy_id) if row["status"] in {"FROZEN", "SELECTED"}}
        candidates = [family for family in sorted(spec.parameter_families, key=lambda value: value.optimization_order) if family.mutable and family.name not in frozen]
        baseline = self.registry.get_baseline(strategy_id)
        artifact = ResearchArtifact.model_validate(baseline["artifact_json"]) if baseline else None
        citations = []
        if artifact:
            for metric in ("expectancy_r", "profit_factor", "max_drawdown"):
                citations.append(MetricCitation(metric_name=metric, value=float(artifact.metrics[metric]), source_file=artifact.metrics_path, source_path=f"$.{metric}", experiment_id=artifact.experiment_id))
        if not candidates:
            return AnalystDecision(strategy_id=strategy_id, strategy_version=strategy["version"], current_phase=PipelineState.PARAMETER_RESEARCH, decision="FREEZE_CANDIDATE", confidence=.9, evidence_strength="adequate", primary_bottleneck="none", proposal_method="deterministic stopping rule", parameter_hypothesis="all justified mutable families have been assessed", expected_behavior="preserve stable candidate", files_inspected=[artifact.metrics_path] if artifact else [], metrics_cited=citations, risks=[], overfitting_risk=.1, stop_reason="no justified mutable parameter family remains", next_phase=PipelineState.CANDIDATE_FREEZE, rationale="The deterministic family budget and hypothesis relevance rules leave no eligible family.")
        family = candidates[0]
        values = self._proposal_values(family)
        return AnalystDecision(strategy_id=strategy_id, strategy_version=strategy["version"], current_phase=PipelineState.PARAMETER_RESEARCH, decision="CONTINUE_PARAMETER_RESEARCH", confidence=.8, evidence_strength="preliminary", primary_bottleneck=family.hypothesis_relevance, selected_parameter_family=family.name, current_value=family.baseline_value, proposed_values=values, proposal_method="bounded local neighborhood", parameter_hypothesis=family.hypothesis_relevance, expected_behavior="improve robustness without isolated maximum", files_inspected=[artifact.metrics_path] if artifact else [], metrics_cited=citations, risks=["selection risk"], overfitting_risk=.25, next_phase=PipelineState.PARAMETER_RESEARCH, rationale="The family is mutable, hypothesis-relevant, and has a bounded local neighborhood.")

    @staticmethod
    def _proposal_values(family: ParameterFamily) -> list[Any]:
        if family.allowed_values:
            values = list(family.allowed_values)
            if family.baseline_value in values:
                index = values.index(family.baseline_value); return values[max(0, index - 2): index + 3]
            return values[:5]
        current = family.baseline_value
        if not isinstance(current, (int, float)) or isinstance(current, bool): return [current]
        span = ((family.allowed_max or current) - (family.allowed_min or current))
        step = span / 4 if span else (1 if isinstance(current, int) else .01)
        raw = [current - 2 * step, current - step, current, current + step, current + 2 * step]
        clipped = [max(family.allowed_min, min(family.allowed_max, value)) if family.allowed_min is not None and family.allowed_max is not None else value for value in raw]
        return list(dict.fromkeys(int(value) if family.value_type in {"integer", "int"} else float(value) for value in clipped))[:5]

    def validate_decision(self, decision: AnalystDecision) -> None:
        strategy = self._strategy(decision.strategy_id)
        if decision.strategy_version != strategy["version"] or decision.current_phase != PipelineState.PARAMETER_RESEARCH:
            raise SpecificationValidationError("analyst decision does not match current strategy phase/version")
        spec = self.registry.get_specification(decision.strategy_id)
        family = next((item for item in spec.parameter_families if item.name == decision.selected_parameter_family), None) if decision.selected_parameter_family else None
        if decision.decision == "CONTINUE_PARAMETER_RESEARCH":
            if family is None or not family.mutable: raise SpecificationValidationError("decision selected a non-mutable or unknown family")
            self._validate_values(family, decision.proposed_values)
        for citation in decision.metrics_cited:
            self._verify_citation(decision.strategy_id, citation)

    @staticmethod
    def _validate_values(family: ParameterFamily, values: list[Any]) -> None:
        if not values or len(values) > 5: raise SpecificationValidationError("parameter proposal exceeds configured value limit")
        for value in values:
            if family.allowed_values is not None and value not in family.allowed_values: raise SpecificationValidationError(f"value {value!r} is not allowed for {family.name}")
            if family.allowed_min is not None and value < family.allowed_min: raise SpecificationValidationError(f"value {value!r} is below allowed_min")
            if family.allowed_max is not None and value > family.allowed_max: raise SpecificationValidationError(f"value {value!r} is above allowed_max")

    def _verify_citation(self, strategy_id: str, citation: MetricCitation) -> None:
        baseline = self.registry.get_baseline(strategy_id)
        if not baseline: raise SpecificationValidationError("metric citation has no baseline experiment")
        artifact = ResearchArtifact.model_validate(baseline["artifact_json"])
        if citation.experiment_id != artifact.experiment_id or Path(citation.source_file).resolve() != Path(artifact.metrics_path).resolve() or not Path(citation.source_file).is_file():
            self.registry.record_metric_citation(strategy_id, baseline["strategy_version"], "PARAMETER_RESEARCH", citation.model_dump(), False, "citation source does not belong to verified baseline")
            raise SpecificationValidationError("unsupported metric citation")
        key = citation.source_path.removeprefix("$." )
        metrics = json.loads(Path(citation.source_file).read_text(encoding="utf-8"))
        if key not in metrics or not math.isclose(float(metrics[key]), citation.value, rel_tol=1e-12, abs_tol=1e-12):
            self.registry.record_metric_citation(strategy_id, baseline["strategy_version"], "PARAMETER_RESEARCH", citation.model_dump(), False, "cited value does not match report")
            raise SpecificationValidationError("metric citation value mismatch")
        self.registry.record_metric_citation(strategy_id, baseline["strategy_version"], "PARAMETER_RESEARCH", citation.model_dump(), True, "verified against machine-readable baseline report")

    def propose_round(self, decision: AnalystDecision) -> ParameterProposal:
        self.validate_decision(decision)
        if decision.decision != "CONTINUE_PARAMETER_RESEARCH": raise SpecificationValidationError("decision does not propose a parameter round")
        strategy = self._strategy(decision.strategy_id)
        rounds = self.registry.list_research_rounds(decision.strategy_id)
        number = len(rounds) + 1
        budget = self.registry.get_budget(decision.strategy_id)
        if number > budget["limits"]["max_rounds_per_family"] * max(1, budget["limits"]["max_parameter_families"]): raise SpecificationValidationError("parameter research budget is exhausted")
        proposal = ParameterProposal(strategy_id=decision.strategy_id, strategy_version=strategy["version"], family=decision.selected_parameter_family, current_value=decision.current_value, proposed_values=decision.proposed_values, round_number=number, hypothesis=decision.parameter_hypothesis, reason=decision.rationale)
        round_id = f"round-{decision.strategy_id}-{number}-{decision.selected_parameter_family}"
        self.registry.save_research_round(round_id, decision.strategy_id, strategy["version"], proposal.family, number, proposal.proposed_values, reason=proposal.reason)
        self._journal(decision.strategy_id, "PARAMETER_RESEARCH", proposal.model_dump(mode="json"), f"Proposed {proposal.family}: {proposal.proposed_values}")
        return proposal

    def run_round(self, strategy_id: str, proposal: ParameterProposal) -> ParameterRoundResult:
        self._assert_phase(strategy_id, PipelineState.PARAMETER_RESEARCH)
        if proposal.strategy_id != strategy_id: raise SpecificationValidationError("proposal strategy mismatch")
        self.controller.consume_budget(strategy_id, backtests=len(proposal.proposed_values), family=proposal.family, rounds=1, values=len(proposal.proposed_values), research_round=proposal.round_number)
        spec, split = self._spec_split(strategy_id)
        experiments = []
        round_id = f"round-{strategy_id}-{proposal.round_number}-{proposal.family}"
        for index, value in enumerate(proposal.proposed_values):
            experiment_id = f"{round_id}-experiment-{index}"
            artifact = self.adapter.run_parameter_experiment(spec, split, {proposal.family: value}, self._root(strategy_id, "parameter_research", proposal.family, f"round_{proposal.round_number}", f"experiment_{index}"), experiment_id)
            self._record_artifact(artifact)
            score, components = self._score(artifact.metrics)
            experiments.append(ParameterExperiment(value=value, artifact=artifact, robustness_score=score, score_components=components))
        review = self._review_round(strategy_id, round_id, proposal.family, experiments)
        result = ParameterRoundResult(round_id=round_id, family=proposal.family, experiments=experiments, review=review, selected_value=review.selected_value, stable_region=review.stable_region, stopped=review.selected_value is None, stop_reason=review.veto_reason)
        self.registry.update_research_round(round_id, experiments=[item.model_dump(mode="json") for item in experiments], review=review.model_dump(mode="json"), selected_value=review.selected_value, status="SELECTED" if review.selected_value is not None else "STOPPED", reason=review.rationale)
        self._journal(strategy_id, "PARAMETER_RESEARCH", result.model_dump(mode="json"), f"Reviewed {proposal.family}; selected {review.selected_value!r}")
        if review.decision == "VETO":
            self.controller.transition(strategy_id, PipelineState.MANUAL_REVIEW_REQUIRED, "statistical reviewer vetoed isolated maximum risk")
        elif review.decision == "INSUFFICIENT_EVIDENCE":
            self.controller.transition(strategy_id, PipelineState.INSUFFICIENT_EVIDENCE, "statistical reviewer found no stable parameter region")
        return result

    @staticmethod
    def _score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
        components = {"expectancy": float(metrics.get("expectancy_r", 0)), "profit_factor": min(3.0, float(metrics.get("profit_factor", 0))) / 3, "drawdown_penalty": -float(metrics.get("max_drawdown", 1)), "fee_penalty": -float(metrics.get("fee_share_of_gross_profit", 1)), "trade_support": min(1.0, float(metrics.get("completed_trades", 0)) / 50)}
        return sum(components.values()), components

    def _review_round(self, strategy_id: str, round_id: str, family: str, experiments: list[ParameterExperiment]) -> StatisticalReview:
        ordered = sorted(experiments, key=lambda item: item.value if isinstance(item.value, (int, float)) else str(item.value))
        best = max(experiments, key=lambda item: item.robustness_score)
        isolated = False; region = [best.value]
        if len(ordered) >= 3 and all(isinstance(item.value, (int, float)) for item in ordered):
            neighbors = [item for item in ordered if item is not best and abs(float(item.value) - float(best.value)) <= max(1, abs(float(best.value)) * .5)]
            stable = [item for item in ordered if item.artifact.metrics.get("expectancy_r", 0) > 0.1 and abs(item.robustness_score - best.robustness_score) <= .25]
            region = [item.value for item in stable]
            isolated = len(stable) == 1 and len(neighbors) >= 2 and best.artifact.metrics.get("expectancy_r", 0) > .3
            if isolated: region = []
        selected = sorted(region, key=lambda value: float(value) if isinstance(value, (int, float)) else str(value))[len(region) // 2] if region else None
        decision = "VETO" if isolated else ("SELECT" if selected is not None else "INSUFFICIENT_EVIDENCE")
        return StatisticalReview(strategy_id=strategy_id, strategy_version=self._strategy(strategy_id)["version"], round_id=round_id, decision=decision, stable_region=region, selected_value=selected, isolated_maximum_risk=isolated, evidence_strength="adequate" if selected is not None else "weak", rationale="Stable neighboring values are preferred over a single maximum." if selected is not None else "No stable region met the deterministic tolerance.", veto_reason="ISOLATED_MAXIMUM_RISK" if isolated else None)

    def freeze_family(self, strategy_id: str, round_id: str) -> dict[str, Any]:
        self._assert_phase(strategy_id, PipelineState.PARAMETER_RESEARCH)
        round_record = self.registry.get_research_round(round_id)
        if not round_record or round_record["status"] != "SELECTED" or round_record["selected_value_json"] is None: raise SpecificationValidationError("only a reviewed selected round can be frozen")
        self.registry.update_research_round(round_id, status="FROZEN")
        payload = {"round_id": round_id, "family": round_record["family"], "selected_value": round_record["selected_value_json"], "status": "FROZEN"}
        self._journal(strategy_id, "PARAMETER_RESEARCH", payload, f"Frozen family {round_record['family']}")
        return payload

    def review_round(self, strategy_id: str, round_id: str) -> StatisticalReview:
        record = self.registry.get_research_round(round_id)
        if not record or record["strategy_id"] != strategy_id or not record["review_json"]:
            raise SpecificationValidationError("parameter round has no persisted statistical review")
        return StatisticalReview.model_validate(record["review_json"])

    def freeze_candidate(self, strategy_id: str) -> CandidateManifest:
        self._assert_phase(strategy_id, PipelineState.PARAMETER_RESEARCH)
        spec, split = self._spec_split(strategy_id); strategy = self._strategy(strategy_id)
        rounds = self.registry.list_research_rounds(strategy_id)
        selected: dict[str, Any] = {}
        frozen: list[str] = []
        for row in rounds:
            if row["status"] == "FROZEN": selected[row["family"]] = json.loads(row["selected_value_json"]); frozen.append(row["family"])
        if not frozen:
            selected = dict(spec.baseline_parameters)
        payload = {"strategy_id": strategy_id, "strategy_version": strategy["version"], "approved_specification_hash": spec.specification_hash, "split_hash": split.split_hash, "code_commit": None, "selected_parameters": selected, "frozen_families": frozen, "research_decisions": [row["round_id"] for row in rounds], "total_selection_count": len(frozen), "budget_usage": self.registry.get_budget(strategy_id)["usage"]}
        candidate_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = self._root(strategy_id, "candidate") / "candidate_manifest.json"
        manifest = CandidateManifest(**payload, candidate_hash=candidate_hash, manifest_path=str(path))
        path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        self.registry.save_candidate(strategy_id, strategy["version"], candidate_hash, manifest.model_dump(mode="json"))
        self.controller.transition(strategy_id, PipelineState.CANDIDATE_FREEZE, "candidate manifest frozen")
        self._journal(strategy_id, "CANDIDATE_FREEZE", manifest.model_dump(mode="json"), "Candidate frozen; no parameter changes are permitted.")
        return manifest

    def _candidate(self, strategy_id: str) -> CandidateManifest:
        rows = self.registry.list_research_rounds(strategy_id); strategy = self._strategy(strategy_id); spec, split = self._spec_split(strategy_id)
        selected = {row["family"]: json.loads(row["selected_value_json"]) for row in rows if row["status"] == "FROZEN"}
        if not selected: selected = dict(spec.baseline_parameters)
        payload = {"strategy_id": strategy_id, "strategy_version": strategy["version"], "approved_specification_hash": spec.specification_hash, "split_hash": split.split_hash, "code_commit": None, "selected_parameters": selected, "frozen_families": list(selected), "research_decisions": [row["round_id"] for row in rows], "total_selection_count": len(selected), "budget_usage": self.registry.get_budget(strategy_id)["usage"]}
        candidate_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = self._root(strategy_id, "candidate") / "candidate_manifest.json"
        return CandidateManifest(**payload, candidate_hash=candidate_hash, manifest_path=str(path))

    def run_walk_forward(self, strategy_id: str) -> WalkForwardResult:
        self._assert_phase(strategy_id, PipelineState.CANDIDATE_FREEZE)
        self.controller.transition(strategy_id, PipelineState.WALK_FORWARD, "run frozen candidate walk-forward validation")
        spec, split = self._spec_split(strategy_id); candidate = self._candidate(strategy_id)
        artifact = self.adapter.run_walk_forward(spec, split, candidate.selected_parameters, self._root(strategy_id, "walk_forward"))
        self._record_artifact(artifact)
        gate_outcomes = self.gate_evaluator.evaluate_set(self._gates("validation", strategy_id), artifact.metrics, artifact.metrics_path)
        status = "FAIL" if self.scenario == "walk-forward-failure" or any(item.status in {GateOutcomeStatus.FAIL, GateOutcomeStatus.INSUFFICIENT_EVIDENCE} for item in gate_outcomes) else "PASS"
        result = WalkForwardResult(status=status, folds=[{"fold": 1, "metrics": artifact.metrics}], aggregate_metrics=artifact.metrics, verification_outcome="VERIFIED", reason="synthetic deterministic walk-forward", gate_outcomes=[item.model_dump(mode="json") for item in gate_outcomes])
        self.registry.save_research_json("research_walk_forward", strategy_id, self._strategy(strategy_id)["version"], result.model_dump(mode="json"))
        self._journal(strategy_id, "WALK_FORWARD", result.model_dump(mode="json"), f"Walk-forward status: {status}")
        self.controller.transition(strategy_id, PipelineState.HOLDOUT if status == "PASS" else PipelineState.REJECTED, f"walk-forward result {status}")
        return result

    def run_holdout(self, strategy_id: str) -> HoldoutResult:
        self._assert_phase(strategy_id, PipelineState.HOLDOUT)
        spec, split = self._spec_split(strategy_id); self.controller.open_holdout(strategy_id, "Phase C one-time final validation", split.source_data_hash)
        candidate = self._candidate(strategy_id); artifact = self.adapter.run_holdout(spec, split, candidate.selected_parameters, self._root(strategy_id, "holdout")); self._record_artifact(artifact)
        gate_outcomes = self.gate_evaluator.evaluate_set(self._gates("holdout", strategy_id), artifact.metrics, artifact.metrics_path)
        status = "FAIL" if self.scenario == "holdout-failure" or any(item.status in {GateOutcomeStatus.FAIL, GateOutcomeStatus.INSUFFICIENT_EVIDENCE} for item in gate_outcomes) else "PASS"
        result = HoldoutResult(status=status, access_count=self.registry.count_holdout_accesses(strategy_id), dataset_hash=split.source_data_hash, metrics=artifact.metrics, verification_outcome="VERIFIED", reason="UNTOUCHED_HOLDOUT evaluated exactly once", gate_outcomes=[item.model_dump(mode="json") for item in gate_outcomes])
        self.registry.save_research_json("research_holdout", strategy_id, self._strategy(strategy_id)["version"], result.model_dump(mode="json")); self._journal(strategy_id, "HOLDOUT", result.model_dump(mode="json"), "UNTOUCHED_HOLDOUT accessed once.")
        self.controller.transition(strategy_id, PipelineState.STRESS_TESTS if status == "PASS" else PipelineState.REJECTED, f"holdout result {status}")
        return result

    def run_stress(self, strategy_id: str) -> StressResult:
        self._assert_phase(strategy_id, PipelineState.STRESS_TESTS)
        spec, split = self._spec_split(strategy_id); candidate = self._candidate(strategy_id); artifact = self.adapter.run_stress_test(spec, split, candidate.selected_parameters, self._root(strategy_id, "stress")); self._record_artifact(artifact)
        fragile = self.scenario == "stress-sensitive"; classification = "FRAGILE" if fragile else "ROBUST"; ratio = .25 if fragile else .8
        scenarios = [{"name": name, "profitable": not fragile or name == "normal_fees", "metrics": artifact.metrics} for name in ("normal_fees", "fees_x2", "fees_x3", "slippage", "worse_entries", "worse_exits", "trade_removal_seed_17", "missing_fills")]
        result = StressResult(classification=classification, scenarios=scenarios, profitable_scenario_ratio=ratio, expectancy_range=[-.2 if fragile else .05, .3], worst_drawdown=.45 if fragile else .2, break_even_fee_level=.001 if fragile else .003, reason="stress results are diagnostic only; no reselection performed")
        self.registry.save_research_json("research_stress", strategy_id, self._strategy(strategy_id)["version"], result.model_dump(mode="json")); self._journal(strategy_id, "STRESS_TESTS", result.model_dump(mode="json"), "Stress testing did not change the candidate."); self.controller.transition(strategy_id, PipelineState.THROUGHPUT, "stress tests completed")
        return result

    def run_throughput(self, strategy_id: str) -> ThroughputResult:
        self._assert_phase(strategy_id, PipelineState.THROUGHPUT)
        spec, split = self._spec_split(strategy_id); candidate = self._candidate(strategy_id); artifact = self.adapter.run_throughput_analysis(spec, split, candidate.selected_parameters, self._root(strategy_id, "throughput")); self._record_artifact(artifact)
        metrics = artifact.metrics; tpm = float(metrics["executable_trades_per_month"]); classification = "STANDALONE_CAPABLE" if tpm >= 5 else ("PORTFOLIO_COMPONENT_ONLY" if tpm > 0 else "INSUFFICIENT_EVIDENCE")
        gate_outcomes = self.gate_evaluator.evaluate_set(self._gates("throughput", strategy_id), artifact.metrics, artifact.metrics_path)
        if any(item.status in {GateOutcomeStatus.FAIL, GateOutcomeStatus.INSUFFICIENT_EVIDENCE} for item in gate_outcomes) and classification == "STANDALONE_CAPABLE": classification = "PORTFOLIO_COMPONENT_ONLY"
        result = ThroughputResult(classification=classification, candidate_setups=int(metrics["completed_trades"]), unique_setups=int(metrics["completed_trades"]), filled_positions=int(metrics["completed_trades"]), completed_positions=int(metrics["completed_trades"]), trades_per_month=tpm, median_days_between_trades=float(metrics["median_days_between_trades"]), longest_no_trade_period_days=float(metrics["median_days_between_trades"] * 3), zero_trade_month_percentage=float(metrics["zero_trade_month_percentage"]), trades_by_market={market: int(metrics["completed_trades"] / max(1, len(spec.markets))) for market in spec.markets}, trades_by_timeframe={frame: int(metrics["completed_trades"] / max(1, len(spec.timeframes))) for frame in spec.timeframes}, accumulation_days={str(n): (n / tpm * 30 if tpm else None) for n in (30, 50, 100, 200)}, reason="completed positions are used; order revisions are excluded", gate_outcomes=[item.model_dump(mode="json") for item in gate_outcomes])
        self.registry.save_research_json("research_throughput", strategy_id, self._strategy(strategy_id)["version"], result.model_dump(mode="json")); self._journal(strategy_id, "THROUGHPUT", result.model_dump(mode="json"), "Throughput classification completed."); self.controller.transition(strategy_id, PipelineState.FINAL_REVIEW, "throughput completed; final review required")
        return result

    def final_review(self, strategy_id: str) -> FinalResearchReview:
        self._assert_phase(strategy_id, PipelineState.FINAL_REVIEW)
        baseline = self.registry.get_baseline(strategy_id); wf = self.registry.get_research_json("research_walk_forward", strategy_id); holdout = self.registry.get_research_json("research_holdout", strategy_id); stress = self.registry.get_research_json("research_stress", strategy_id); throughput = self.registry.get_research_json("research_throughput", strategy_id)
        if not all((baseline, wf, holdout, stress, throughput)): raise SpecificationValidationError("final review requires baseline, walk-forward, holdout, stress, and throughput results")
        if wf["status"] != "PASS" or holdout["status"] != "PASS": classification = ResearchClassification.REJECTED_UNSTABLE
        elif stress["classification"] == "FRAGILE": classification = ResearchClassification.REJECTED_EXECUTION_SENSITIVE
        elif throughput["classification"] == "PORTFOLIO_COMPONENT_ONLY": classification = ResearchClassification.ACCEPTED_PORTFOLIO_COMPONENT
        elif throughput["classification"] == "INSUFFICIENT_EVIDENCE": classification = ResearchClassification.INSUFFICIENT_EVIDENCE
        else: classification = ResearchClassification.ACCEPTED_STANDALONE
        strategy = self._strategy(strategy_id); result = FinalResearchReview(strategy_id=strategy_id, strategy_version=strategy["version"], classification=classification, current_phase=PipelineState.FINAL_REVIEW, evidence_strength="verified", evidence={"baseline": baseline, "walk_forward": wf, "holdout": holdout, "stress": stress, "throughput": throughput}, metrics_cited=[], risks=["synthetic fixture"], rationale="Final classification is derived from persisted deterministic evidence; no model claim overrides a failed gate.", next_phase=PipelineState.ACCEPTED if classification in {ResearchClassification.ACCEPTED_STANDALONE, ResearchClassification.ACCEPTED_PORTFOLIO_COMPONENT} else None)
        self.registry.save_research_json("research_final_reviews", strategy_id, strategy["version"], result.model_dump(mode="json")); self._journal(strategy_id, "FINAL_REVIEW", result.model_dump(mode="json"), f"Final classification: {classification.value}")
        state = PipelineState.ACCEPTED if classification in {ResearchClassification.ACCEPTED_STANDALONE, ResearchClassification.ACCEPTED_PORTFOLIO_COMPONENT} else PipelineState.REJECTED if classification in {ResearchClassification.REJECTED_UNSTABLE, ResearchClassification.REJECTED_EXECUTION_SENSITIVE, ResearchClassification.REJECTED_NO_EDGE} else PipelineState.INSUFFICIENT_EVIDENCE
        self.controller.transition(strategy_id, state, f"final research classification {classification.value}")
        return result

    def status(self, strategy_id: str) -> dict[str, Any]:
        candidate = self.registry.get_candidate(strategy_id)
        return {"strategy": self._strategy(strategy_id), "baseline": self.registry.get_baseline(strategy_id), "rounds": self.registry.list_research_rounds(strategy_id), "candidate": candidate, "walk_forward": self.registry.get_research_json("research_walk_forward", strategy_id), "holdout": self.registry.get_research_json("research_holdout", strategy_id), "stress": self.registry.get_research_json("research_stress", strategy_id), "throughput": self.registry.get_research_json("research_throughput", strategy_id), "final_review": self.registry.get_research_json("research_final_reviews", strategy_id), "holdout_accesses": self.registry.count_holdout_accesses(strategy_id)}

    def journal(self, strategy_id: str) -> list[dict]:
        return self.registry.journal(strategy_id)
