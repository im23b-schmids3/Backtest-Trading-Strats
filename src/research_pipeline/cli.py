from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import ValidationError

from .config.loader import load_pipeline_config
from .config.logging_setup import configure_logging
from .controller.pipeline_controller import PipelineController
from .enums import PipelineState
from .errors import ResearchPipelineError
from .registry.database import Database
from .registry.repositories import Registry
from .schemas.decisions import DecisionRecord
from .schemas.splits import SplitDefinition, calculate_split_hash
from .schemas.strategy_spec import load_strategy_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_pipeline", description="Deterministic Phase A research-pipeline registry")
    parser.add_argument("--registry", default=os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    command = sub.add_parser("new-strategy"); command.add_argument("path")
    sub.add_parser("list-strategies")
    command = sub.add_parser("status"); command.add_argument("strategy_id")
    command = sub.add_parser("validate-spec"); command.add_argument("strategy_id")
    command = sub.add_parser("submit-spec"); command.add_argument("strategy_id")
    command = sub.add_parser("approve-spec"); command.add_argument("strategy_id")
    command = sub.add_parser("transition"); command.add_argument("strategy_id"); command.add_argument("new_state", choices=[state.value for state in PipelineState]); command.add_argument("--reason", required=True)
    command = sub.add_parser("show-budget"); command.add_argument("strategy_id")
    command = sub.add_parser("consume-budget"); command.add_argument("strategy_id"); command.add_argument("--backtests", type=int, default=0); command.add_argument("--family"); command.add_argument("--rounds", type=int, default=0); command.add_argument("--values", type=int, default=0); command.add_argument("--research-round", type=int, default=None); command.add_argument("--codex-repairs", type=int, default=0); command.add_argument("--runtime-minutes", type=int, default=0); command.add_argument("--report-size-mb", type=float, default=0.0)
    command = sub.add_parser("create-split"); command.add_argument("strategy_id"); command.add_argument("split_config")
    command = sub.add_parser("holdout-status"); command.add_argument("strategy_id")
    command = sub.add_parser("open-holdout"); command.add_argument("strategy_id"); command.add_argument("--reason", required=True); command.add_argument("--dataset-hash", default=None)
    command = sub.add_parser("record-decision"); command.add_argument("strategy_id"); command.add_argument("decision_json")
    command = sub.add_parser("history"); command.add_argument("strategy_id")
    workflow = sub.add_parser("workflow", help="typed Smithers bridge commands")
    workflow.add_argument("workflow_command", choices=["generate-spec", "validate-spec", "register-generated-spec", "approve", "implementation-plan", "execute-codex", "record-codex-result", "run-tests", "run-required-tests", "research-start", "research-run-baseline", "research-edge-gate", "research-analyze", "research-propose-round", "research-run-round", "research-review-round", "research-freeze-family", "research-freeze-candidate", "research-walk-forward", "research-holdout", "research-stress", "research-throughput", "research-final-review", "research-status", "research-journal", "technical-verification", "final-status", "verification-create-manifest", "verification-run", "diagnose-tools"])
    workflow.add_argument("--input-json")
    workflow.add_argument("--repository-root", default=".")
    research = sub.add_parser("research", help="deterministic Phase C research commands")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    command = research_sub.add_parser("dry-run"); command.add_argument("--strategy-id", default="phase-c-dry-run"); command.add_argument("--scenario", default="strong-stable"); command.add_argument("--repository-root", default="."); command.add_argument("--registry-path")
    for name in ("fixture", "run-baseline", "baseline-status", "analyze", "propose-round", "run-round", "review-round", "freeze-family", "freeze-candidate", "run-walk-forward", "run-holdout", "run-stress", "run-throughput", "final-review", "journal", "status"):
        command = research_sub.add_parser(name); command.add_argument("strategy_id")
        command.add_argument("--scenario", default=argparse.SUPPRESS)
        command.add_argument("--repository-root", default=argparse.SUPPRESS)
        command.add_argument("--decision-json", default=argparse.SUPPRESS)
        command.add_argument("--proposal-json", default=argparse.SUPPRESS)
        command.add_argument("--round-id", default=argparse.SUPPRESS)
        command.add_argument("--registry-path", default=argparse.SUPPRESS)
    research.add_argument("--scenario", default="strong-stable")
    research.add_argument("--repository-root", default=".")
    research.add_argument("--decision-json")
    research.add_argument("--proposal-json")
    research.add_argument("--round-id")
    research.add_argument("--run-id")
    research.add_argument("--registry-path")
    verification = sub.add_parser("verification", help="Phase B.5 technical integrity verification")
    verification_sub = verification.add_subparsers(dest="verification_command", required=True)
    command = verification_sub.add_parser("create-manifest"); command.add_argument("strategy_id"); command.add_argument("--diagnostic-dir"); command.add_argument("--output")
    command = verification_sub.add_parser("run"); command.add_argument("strategy_id"); command.add_argument("--manifest", required=True)
    command = verification_sub.add_parser("status"); command.add_argument("strategy_id"); command.add_argument("--run-id")
    command = verification_sub.add_parser("show-failures"); command.add_argument("strategy_id"); command.add_argument("--run-id")
    command = verification_sub.add_parser("reconcile-report"); command.add_argument("strategy_id"); command.add_argument("--manifest", required=True)
    command = verification_sub.add_parser("rerun-check"); command.add_argument("strategy_id"); command.add_argument("check_name"); command.add_argument("--manifest", required=True)
    command = verification_sub.add_parser("export-defect-prompt"); command.add_argument("strategy_id"); command.add_argument("--manifest", required=True)
    command = verification_sub.add_parser("fixture"); command.add_argument("strategy_id"); command.add_argument("--kind", default="correct"); command.add_argument("--output", required=True); command.add_argument("--version", default="phase-b-1")
    command = verification_sub.add_parser("dry-run"); command.add_argument("--kind", default="correct")
    return parser


def _controller(registry_path: str) -> PipelineController:
    configure_logging(Path(registry_path).with_suffix(".log"))
    return PipelineController(Registry(Database(registry_path)))


def _load_split_config(path: str) -> SplitDefinition:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.setdefault("created_timestamp", datetime.now(timezone.utc).isoformat())
    raw.setdefault("split_hash", calculate_split_hash(raw))
    return SplitDefinition.model_validate(raw)


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        controller = _controller(args.registry)
        registry = controller.registry
        if args.command == "init":
            print(f"initialized registry: {Path(args.registry)}")
        elif args.command == "new-strategy":
            spec = load_strategy_spec(args.path)
            config = load_pipeline_config(Path("configs/research_pipeline/defaults.yaml"), strategy_id=spec.strategy_id)
            _print(controller.register_strategy(spec, str(Path(args.path).resolve()), config["budgets"]))
        elif args.command == "list-strategies":
            _print(registry.list_strategies())
        elif args.command == "status":
            _print(controller.status(args.strategy_id))
        elif args.command == "validate-spec":
            _print(controller.validate_specification(args.strategy_id))
        elif args.command == "submit-spec":
            _print(controller.submit_specification(args.strategy_id))
        elif args.command == "approve-spec":
            _print(controller.approve_specification(args.strategy_id))
        elif args.command == "transition":
            _print(controller.transition(args.strategy_id, args.new_state, args.reason))
        elif args.command == "show-budget":
            _print(registry.get_budget(args.strategy_id))
        elif args.command == "consume-budget":
            _print(controller.consume_budget(args.strategy_id, backtests=args.backtests, family=args.family, rounds=args.rounds, values=args.values, research_round=args.research_round, codex_repairs=args.codex_repairs, runtime_minutes=args.runtime_minutes, report_size_mb=args.report_size_mb).model_dump())
        elif args.command == "create-split":
            split = _load_split_config(args.split_config)
            controller.create_split(args.strategy_id, split)
            _print(split.model_dump(mode="json"))
        elif args.command == "holdout-status":
            _print(controller.holdout_status(args.strategy_id))
        elif args.command == "open-holdout":
            _print(controller.open_holdout(args.strategy_id, args.reason, args.dataset_hash))
        elif args.command == "record-decision":
            source = Path(args.decision_json)
            raw = json.loads(source.read_text(encoding="utf-8")) if source.exists() else json.loads(args.decision_json)
            _print({"decision_id": controller.record_decision(args.strategy_id, DecisionRecord.model_validate(raw))})
        elif args.command == "history":
            _print(registry.history(args.strategy_id))
        elif args.command == "workflow":
            if args.workflow_command == "diagnose-tools":
                from .tools import print_diagnostics
                return print_diagnostics(args.repository_root)
            if not args.input_json:
                raise ValueError("--input-json is required for workflow bridge commands")
            from .workflow_bridge.bridge import PhaseBBridge
            source = Path(args.input_json)
            payload = json.loads(source.read_text(encoding="utf-8")) if source.exists() else json.loads(args.input_json)
            _print(PhaseBBridge().dispatch(args.workflow_command, payload))
        elif args.command == "research":
            from .research.models import AnalystDecision, ParameterProposal
            from .research.services import PhaseCService
            if args.research_command == "dry-run":
                from .research.fixtures import run_phase_c_dry_run
                import tempfile
                if args.registry_path:
                    result = run_phase_c_dry_run(args.registry_path, args.repository_root, args.strategy_id, args.scenario)
                else:
                    with tempfile.TemporaryDirectory(prefix="research-pipeline-phase-c-") as temp:
                        root = Path(temp)
                        result = run_phase_c_dry_run(root / "research_registry.sqlite3", root, args.strategy_id, args.scenario)
                _print(result)
                return 0
            service = PhaseCService(args.registry, repository_root=args.repository_root, scenario=args.scenario)
            command = args.research_command
            if command == "fixture":
                from .research.fixtures import prepare_phase_c_fixture
                result = prepare_phase_c_fixture(args.registry_path or args.registry, args.repository_root, args.strategy_id, args.scenario)
            elif command == "run-baseline": result = service.run_baseline(args.strategy_id)
            elif command == "baseline-status": result = service.registry.get_baseline(args.strategy_id)
            elif command == "analyze": result = service.analyze(args.strategy_id)
            elif command == "propose-round":
                source = Path(args.decision_json); result = service.propose_round(AnalystDecision.model_validate(json.loads(source.read_text(encoding="utf-8") if source.exists() else args.decision_json)))
            elif command == "run-round":
                source = Path(args.proposal_json); result = service.run_round(args.strategy_id, ParameterProposal.model_validate(json.loads(source.read_text(encoding="utf-8") if source.exists() else args.proposal_json)))
            elif command == "review-round": result = service.review_round(args.strategy_id, args.round_id)
            elif command == "freeze-family": result = service.freeze_family(args.strategy_id, args.round_id)
            elif command == "freeze-candidate": result = service.freeze_candidate(args.strategy_id)
            elif command == "run-walk-forward": result = service.run_walk_forward(args.strategy_id)
            elif command == "run-holdout": result = service.run_holdout(args.strategy_id)
            elif command == "run-stress": result = service.run_stress(args.strategy_id)
            elif command == "run-throughput": result = service.run_throughput(args.strategy_id)
            elif command == "final-review": result = service.final_review(args.strategy_id)
            elif command == "journal": result = {"entries": service.journal(args.strategy_id)}
            elif command == "status": result = service.status(args.strategy_id)
            else: raise ValueError(f"unsupported research command: {command}")
            _print(result.model_dump(mode="json") if hasattr(result, "model_dump") else result)
        elif args.command == "verification":
            from .verification.fixtures import make_fixture
            from .verification.services import VerificationService
            service = VerificationService(args.registry)
            if args.verification_command == "create-manifest":
                manifest = service.create_manifest(args.strategy_id, args.diagnostic_dir)
                target = Path(args.output) if args.output else Path(args.diagnostic_dir or service.registry_path.parent / "verification" / args.strategy_id) / "manifest.yaml"
                manifest.save(target); _print(manifest.model_dump(mode="json"))
            elif args.verification_command == "fixture":
                _print({"manifest": str(make_fixture(args.output, args.strategy_id, args.version, args.kind))})
            elif args.verification_command == "dry-run":
                import tempfile
                from .phase_b.models import WorkflowInput
                with tempfile.TemporaryDirectory(prefix="research-pipeline-b5-") as temp:
                    root = Path(temp)
                    registry_path = root / "registry.sqlite3"
                    phase_b = __import__("research_pipeline.phase_b.services", fromlist=["PhaseBService"]).PhaseBService(registry_path)
                    strategy_name = "b5-dry-run"
                    generated = phase_b.generate_spec(WorkflowInput(strategy_name=strategy_name, natural_language_description="A deterministic B.5 verification fixture strategy.", requested_markets=["TEST"], requested_timeframes=["1h"], repository_root=str(root)))
                    phase_b.register_generated(phase_b.validate_spec(generated)); phase_b.approve(generated.strategy_id, "APPROVE")
                    phase_b.controller.transition(generated.strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "fixture implementation verified")
                    manifest_path = make_fixture(root / "diagnostics", generated.strategy_id, kind=args.kind)
                    result = VerificationService(registry_path).run(generated.strategy_id, manifest_path)
                    _print({"kind": args.kind, "strategy_id": generated.strategy_id, "registry_path": str(registry_path), "result": result})
            elif args.verification_command == "run":
                _print(service.run(args.strategy_id, args.manifest))
            elif args.verification_command in {"status", "show-failures"}:
                result = service.status(args.strategy_id, args.run_id)
                if args.verification_command == "show-failures" and result:
                    result = {"verification_run_id": result["verification_run_id"], "outcome": result["outcome"], "failed_checks": result.get("mandatory_checks_failed", []), "blocking_issues": result.get("blocking_issues", [])}
                _print(result or {})
            elif args.verification_command in {"reconcile-report", "rerun-check"}:
                result = service.run(args.strategy_id, args.manifest)
                if args.verification_command == "rerun-check":
                    result = {"check_name": args.check_name, "check": next((check for check in result.get("checks", []) if check["check_name"] == args.check_name), None), "verification_run_id": result["verification_run_id"]}
                else:
                    result = {"verification_run_id": result["verification_run_id"], "check": next((check for check in result.get("checks", []) if check["check_name"] == "report_reconciliation"), None)}
                _print(result)
            elif args.verification_command == "export-defect-prompt":
                result = service.status(args.strategy_id)
                if not result: raise ValueError("no verification result found")
                _print({"strategy_id": args.strategy_id, "prompt": "Repair only proven Phase B.5 defects. Failed checks: " + ", ".join(result.get("mandatory_checks_failed", [])) + ". Evidence: " + json.dumps(result.get("blocking_issues", []))})
        return 0
    except (ResearchPipelineError, ValidationError, ValueError, OSError, json.JSONDecodeError) as exc:
        logging.getLogger("research_pipeline").warning("command_error type=%s message=%s", type(exc).__name__, str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 2
