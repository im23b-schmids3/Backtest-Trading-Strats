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
        if command in {"technical-verification", "final-status"}:
            from ..phase_b.models import TestResult
            return self.service.technical_verification(str(payload["strategy_id"]), TestResult.model_validate(payload["test_result"]), implementation_executed=bool(payload.get("implementation_executed", False)), repair_attempts=int(payload.get("repair_attempts", 0)), worktree_path=payload.get("worktree_path")).model_dump(mode="json")
        raise ValueError(f"unsupported workflow bridge command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research_pipeline workflow")
    parser.add_argument("command", choices=["generate-spec", "validate-spec", "register-generated-spec", "approve", "implementation-plan", "execute-codex", "record-codex-result", "run-tests", "run-required-tests", "technical-verification", "final-status"])
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
