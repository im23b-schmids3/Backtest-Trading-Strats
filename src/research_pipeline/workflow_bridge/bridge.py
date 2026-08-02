from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..phase_b.models import GeneratedStrategySpec, SpecificationValidationResult, WorkflowInput
from ..phase_b.services import PhaseBService


class PhaseBBridge:
    """JSON-in/JSON-out boundary used by Smithers tasks."""

    def __init__(self, service: PhaseBService | None = None):
        self.service = service or PhaseBService()

    def dispatch(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Registry routing is transport metadata, not part of node output schemas.
        # The Smithers bridge sets RESEARCH_PIPELINE_REGISTRY before dispatch.
        payload = {key: value for key, value in payload.items() if key != "registry_path"}
        if command == "generate-spec":
            # Smithers adds run metadata to the workflow input at execution time;
            # it is orchestration metadata, not part of the user contract.
            user_input = {key: value for key, value in payload.items() if key not in {"runId", "workflowId"}}
            return self.service.generate_spec(WorkflowInput.model_validate(user_input)).model_dump(mode="json")
        if command == "validate-spec":
            return self.service.validate_spec(GeneratedStrategySpec.model_validate(payload)).model_dump(mode="json")
        if command == "register-generated-spec":
            return self.service.register_generated(SpecificationValidationResult.model_validate(payload)).model_dump(mode="json")
        if command == "approve":
            return self.service.approve(str(payload["strategy_id"]), str(payload["decision"]), payload.get("note")).model_dump(mode="json")
        if command == "implementation-plan":
            return self.service.implementation_plan(str(payload["strategy_id"]), str(payload["repository_root"]), dry_run=bool(payload.get("dry_run", True))).model_dump(mode="json")
        if command == "record-codex-result":
            from ..phase_b.models import CodexExecutionResult
            return self.service.record_codex(str(payload["strategy_id"]), CodexExecutionResult.model_validate(payload["result"]), task_name=str(payload.get("task_name", "implementation"))).model_dump(mode="json")
        if command == "execute-codex":
            from ..phase_b.models import ImplementationPlan
            plan = ImplementationPlan.model_validate(payload["plan"])
            return self.service.execute_codex(str(payload["strategy_id"]), str(payload["repository_root"]), plan, str(payload["prompt"]), dry_run=bool(payload.get("dry_run", True)), task_name=str(payload.get("task_name", "implementation"))).model_dump(mode="json")
        if command == "run-tests":
            return self.service.run_tests(str(payload["repository_root"]), dry_run=bool(payload.get("dry_run", True)), worktree_path=payload.get("worktree_path")).model_dump(mode="json")
        if command == "run-required-tests":
            return self.service.run_required_tests(str(payload["repository_root"]), [list(item) for item in payload["required_tests"]], dry_run=bool(payload.get("dry_run", True)), worktree_path=payload.get("worktree_path")).model_dump(mode="json")
        if command.startswith("research-"):
            from ..research.models import AnalystDecision, ParameterProposal
            from ..research.services import PhaseCService
            research = PhaseCService(self.service.registry_path, repository_root=payload.get("repository_root", "."), scenario=payload.get("scenario", "strong-stable"))
            strategy_id = str(payload.get("strategy_id", ""))
            if command == "research-start": return research.start(strategy_id, payload.get("run_id"))
            if command == "research-run-baseline": return research.run_baseline(strategy_id).model_dump(mode="json")
            if command == "research-edge-gate": return research.evaluate_edge(strategy_id)
            if command == "research-analyze": return research.analyze(strategy_id).model_dump(mode="json")
            if command == "research-propose-round": return research.propose_round(AnalystDecision.model_validate(payload["decision"])).model_dump(mode="json")
            if command == "research-run-round": return research.run_round(strategy_id, ParameterProposal.model_validate(payload["proposal"])).model_dump(mode="json")
            if command == "research-review-round": return research.review_round(strategy_id, str(payload["round_id"])).model_dump(mode="json")
            if command == "research-freeze-family": return research.freeze_family(strategy_id, str(payload["round_id"]))
            if command == "research-freeze-candidate": return research.freeze_candidate(strategy_id).model_dump(mode="json")
            if command == "research-walk-forward": return research.run_walk_forward(strategy_id).model_dump(mode="json")
            if command == "research-holdout": return research.run_holdout(strategy_id).model_dump(mode="json")
            if command == "research-stress": return research.run_stress(strategy_id).model_dump(mode="json")
            if command == "research-throughput": return research.run_throughput(strategy_id).model_dump(mode="json")
            if command == "research-final-review": return research.final_review(strategy_id).model_dump(mode="json")
            if command == "research-status": return research.status(strategy_id)
            if command == "research-journal": return {"entries": research.journal(strategy_id)}
            raise ValueError(f"unsupported research command: {command}")
        if command.startswith("prop-"):
            from ..prop.services import PropResearchService
            prop = PropResearchService(self.service.registry_path, repository_root=payload.get("repository_root", "."), scenario=payload.get("scenario", "profitable"))
            strategy_id = str(payload.get("strategy_id", "")); product = str(payload.get("product", "Alpha Futures Zero 25K"))
            if command == "prop-start": return prop.start(strategy_id, payload.get("run_id"))
            if command == "prop-verify-rules": return prop.verify_rules(strategy_id, product)
            if command == "prop-verify-contracts": return prop.verify_contracts(strategy_id)
            if command == "prop-reconcile": return prop.reconcile(strategy_id)
            if command == "prop-run-risk": return prop.run_risk(strategy_id, product)
            if command == "prop-run-scenarios": return prop.run_scenarios(strategy_id, product)
            if command == "prop-economics" or command == "prop-final-review": return prop.economics(strategy_id).model_dump(mode="json")
            if command == "prop-status": return prop.status(strategy_id)
            if command == "prop-journal": return {"entries": prop.journal(strategy_id)}
            raise ValueError(f"unsupported prop command: {command}")
        if command.startswith("portfolio-"):
            from ..portfolio.models import PortfolioSpec
            from ..portfolio.service import PortfolioService
            portfolio = PortfolioService(self.service.registry_path, repository_root=payload.get("repository_root", "."), scenario=payload.get("scenario", "complementary"))
            if command == "portfolio-create":
                raw = payload.get("spec") or payload.get("portfolio_spec")
                if raw is None and payload.get("spec_path"):
                    import yaml
                    raw = yaml.safe_load(Path(str(payload["spec_path"])).read_text(encoding="utf-8"))
                if not isinstance(raw, dict): raise ValueError("portfolio-create requires a structured spec object or spec_path")
                return portfolio.create(PortfolioSpec.model_validate(raw))
            if command == "portfolio-eligible-strategies": return portfolio.eligible(exploratory_prop=bool(payload.get("exploratory_prop", False)), non_prop=bool(payload.get("non_prop", False)))
            portfolio_id = str(payload["portfolio_id"])
            commands = {
                "portfolio-generate-candidates": portfolio.generate_candidates,
                "portfolio-merge-signals": portfolio.merge_signals,
                "portfolio-analyze-overlap": portfolio.analyze_overlap,
                "portfolio-analyze-correlation": portfolio.analyze_correlation,
                "portfolio-run-risk": portfolio.run_risk,
                "portfolio-run-prop": portfolio.run_prop,
                "portfolio-run-ablation": portfolio.run_ablation,
                "portfolio-run-stress": portfolio.run_stress,
                "portfolio-final-review": portfolio.final_review,
                "portfolio-status": portfolio.status,
                "portfolio-journal": lambda value: {"entries": portfolio.journal(value)},
            }
            if command not in commands: raise ValueError(f"unsupported portfolio command: {command}")
            result = commands[command](portfolio_id)
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        if command.startswith("master-"):
            from ..phase_f1.models import MasterRunInput
            from ..phase_f1.service import MasterPipelineService
            service = MasterPipelineService(self.service.registry_path, payload.get("repository_root", "."))
            if command == "master-start":
                options = MasterRunInput.model_validate(payload)
                return service.start(options)
            run_id = str(payload["run_id"])
            if command == "master-approve": return service.approve(run_id, str(payload.get("decision", "APPROVE")), payload.get("note"))
            if command == "master-resume": return service.resume(run_id)
            if command == "master-status": return service.status(run_id)
            if command == "master-report": return service.report(run_id)
            if command == "master-artifacts": return service.artifacts(run_id)
            if command == "master-cancel": return service.cancel(run_id, str(payload.get("reason", "cancelled by operator")))
            raise ValueError(f"unsupported master command: {command}")
        if command == "verification-create-manifest":
            from ..verification.services import VerificationService
            if payload.get("manifest_path") and Path(str(payload["manifest_path"])).is_file():
                from ..verification.models import VerificationManifest
                return {**VerificationManifest.load(str(payload["manifest_path"])).model_dump(mode="json"), "manifest_path": str(Path(str(payload["manifest_path"])).resolve())}
            manifest = VerificationService(self.service.registry_path).create_manifest(str(payload["strategy_id"]), payload.get("diagnostic_dir"), payload.get("verification_run_id"))
            return {**manifest.model_dump(mode="json"), "manifest_path": str((Path(payload.get("diagnostic_dir") or self.service.registry_path.parent / "verification" / str(payload["strategy_id"])) / "manifest.yaml").resolve())}
        if command == "verification-run":
            from ..verification.services import VerificationService
            return VerificationService(self.service.registry_path).run(str(payload["strategy_id"]), str(payload["manifest_path"]))
        if command in {"technical-verification", "final-status"}:
            from ..phase_b.models import TestResult
            return self.service.technical_verification(str(payload["strategy_id"]), TestResult.model_validate(payload["test_result"]), implementation_executed=bool(payload.get("implementation_executed", False)), repair_attempts=int(payload.get("repair_attempts", 0)), worktree_path=payload.get("worktree_path")).model_dump(mode="json")
        raise ValueError(f"unsupported workflow bridge command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research_pipeline workflow")
    parser.add_argument("command", choices=["generate-spec", "validate-spec", "register-generated-spec", "approve", "implementation-plan", "execute-codex", "record-codex-result", "run-tests", "run-required-tests", "research-start", "research-run-baseline", "research-edge-gate", "research-analyze", "research-propose-round", "research-run-round", "research-review-round", "research-freeze-family", "research-freeze-candidate", "research-walk-forward", "research-holdout", "research-stress", "research-throughput", "research-final-review", "research-status", "research-journal", "prop-start", "prop-verify-rules", "prop-verify-contracts", "prop-reconcile", "prop-run-risk", "prop-run-scenarios", "prop-economics", "prop-final-review", "prop-status", "prop-journal", "portfolio-create", "portfolio-eligible-strategies", "portfolio-generate-candidates", "portfolio-merge-signals", "portfolio-analyze-overlap", "portfolio-analyze-correlation", "portfolio-run-risk", "portfolio-run-prop", "portfolio-run-ablation", "portfolio-run-stress", "portfolio-final-review", "portfolio-status", "portfolio-journal", "master-start", "master-approve", "master-resume", "master-status", "master-report", "master-artifacts", "master-cancel", "technical-verification", "final-status", "verification-create-manifest", "verification-run"])
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args(argv)
    try:
        source = Path(args.input_json)
        payload = json.loads(source.read_text(encoding="utf-8")) if source.exists() else json.loads(args.input_json)
        print(json.dumps(PhaseBBridge().dispatch(args.command, payload), sort_keys=True))
        return 0
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
