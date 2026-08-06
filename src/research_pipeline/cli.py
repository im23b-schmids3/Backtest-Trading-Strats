from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pyarrow.parquet as pq
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


def _symbol_manifest_map(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        try:
            symbol, path = item.split("=", 1)
        except ValueError as exc:
            raise ValueError(f"--manifest must use SYMBOL=PATH, got {item!r}") from exc
        symbol = symbol.upper()
        if not symbol or not path or symbol in result:
            raise ValueError(f"invalid or duplicate --manifest value: {item!r}")
        result[symbol] = str(Path(path).resolve())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_pipeline", description="Deterministic Phase A research-pipeline registry")
    parser.add_argument("--registry", default=os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    value_area = sub.add_parser("value-area-trap", help="public BTCUSDT aggregate-trade research data utilities")
    value_area_sub = value_area.add_subparsers(dest="value_area_command", required=True)
    acceptance = sub.add_parser("value-area-acceptance", help="standalone sealed BTC breakout-acceptance exploratory study")
    acceptance_sub = acceptance.add_subparsers(dest="acceptance_command", required=True)
    command = acceptance_sub.add_parser("run-btc-2024-study"); command.add_argument("--data-manifest", default="data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json"); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--non-interactive", action="store_true")
    imbalance = sub.add_parser("imbalance-vwap-ride", help="standalone sealed BTC footprint/VWAP exploratory study")
    imbalance_sub = imbalance.add_subparsers(dest="imbalance_command", required=True)
    command = imbalance_sub.add_parser("validate-source", help="validate the pinned immutable BTCUSDT manifest and Parquet files")
    command.add_argument("--data-manifest", default="data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json")
    command = imbalance_sub.add_parser("build-footprint", help="stream the pinned raw Parquet into a content-addressed local footprint dataset")
    command.add_argument("--data-manifest", default="data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json"); command.add_argument("--cache-root", default="data/imbalance_vwap_ride/footprints"); command.add_argument("--batch-size", type=int, default=1_000_000)
    command = imbalance_sub.add_parser("validate-footprint", help="validate content hashes, schemas, and row counts for a materialized footprint")
    command.add_argument("footprint_root")
    command = imbalance_sub.add_parser("run-btc-exploratory-study", help="run tests, compile/diff checks, then execute the sealed real-data study")
    command.add_argument("--data-manifest", default="data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json"); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--footprint-cache-root", default="data/imbalance_vwap_ride/footprints"); command.add_argument("--batch-size", type=int, default=1_000_000); command.add_argument("--non-interactive", action="store_true")
    command = imbalance_sub.add_parser("run-btc-macro-bins-v2-study", help="run V2 preflight, then execute the separate post-hoc macro-bin study")
    command.add_argument("--data-manifest", default="data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json"); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--footprint-cache-root", default="data/imbalance_vwap_ride/footprints"); command.add_argument("--batch-size", type=int, default=1_000_000); command.add_argument("--non-interactive", action="store_true")
    command = imbalance_sub.add_parser("run-btc-long-only-v3-study", help="download only the sealed six Binance months, validate/normalize, and execute V3")
    command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--data-cache-root", default="data/value_area_trap"); command.add_argument("--footprint-cache-root", default="data/imbalance_vwap_ride/v3/footprints"); command.add_argument("--batch-size", type=int, default=1_000_000); command.add_argument("--non-interactive", action="store_true")
    command = imbalance_sub.add_parser("run-btc-long-only-v4-study", help="execute the sealed V4 Phase-A selection and conditional locked test")
    command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--data-cache-root", default="data/value_area_trap"); command.add_argument("--footprint-cache-root", default="data/imbalance_vwap_ride/v4/footprints"); command.add_argument("--batch-size", type=int, default=1_000_000); command.add_argument("--alpha-rules-artifact"); command.add_argument("--non-interactive", action="store_true")
    command = imbalance_sub.add_parser("validate-btc-long-only-v4-source", help="validate one exact V4 phase manifest without exposing aggregate rows")
    command.add_argument("manifest"); command.add_argument("--phase", choices=["PHASE_A", "PHASE_B"], required=True); command.add_argument("--skip-archive-verification", action="store_true")
    command = imbalance_sub.add_parser("validate-btc-long-only-v4-artifacts", help="verify immutable V4 artifact identities and content hashes")
    command.add_argument("artifact_root")
    command = imbalance_sub.add_parser("run-btc-long-only-v5-study", help="execute V5 local-only price-scaled-bin preflight and artifacts")
    command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--non-interactive", action="store_true")
    command = sub.add_parser("v5-candidate-run", help="execute the sealed V5 Phase-A candidates exactly once")
    command.add_argument("--phase-a-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command = sub.add_parser("lsmr-v1-materialize", help="materialize the sealed LSMR V1 synthetic-only contract; never executes candidates")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", default=".")
    command = sub.add_parser("lsmr-v1-phase-a", help="execute the sealed LSMR V1 Phase-A candidates exactly once")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", default=".")
    command = sub.add_parser("lsmr-v2-strict-materialize", help="materialize the sealed LSMR V2 strict synthetic-only contract; never executes candidates")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", default=".")
    command = sub.add_parser("lsmr-v2-phase-a", help="execute the sealed LSMR V2 Phase-A candidates against an explicit validated bars manifest")
    command.add_argument("--phase-a-bars-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("vbtc-v1-synthetic-materialize", help="materialize only the sealed VBTC V1 synthetic contract")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("vbtc-v1-phase-a", help="deterministic sealed VBTC V1 Phase-A interface")
    command.add_argument("--phase-a-bars-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("vbtc-v2-synthetic-materialize", help="materialize only the sealed VBTC V2 synthetic contract")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("vbtc-v2-phase-a", help="deterministic sealed VBTC V2 Phase-A interface")
    command.add_argument("--phase-a-bars-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("htf-lfvg-v1-synthetic-materialize", help="materialize only the sealed HTF LFVG V1 synthetic contract")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("htf-lfvg-v1-phase-a", help="reserved deterministic sealed HTF LFVG V1 Phase-A interface")
    command.add_argument("--phase-a-bars-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("htf-lfvg-v1-phase-a-funnel-diagnostic", help="read-only HTF LFVG V1 Phase-A event funnel diagnostic")
    command = sub.add_parser("htf-lfvg-v2-synthetic-materialize", help="materialize only the sealed HTF LFVG V2 synthetic contract")
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("htf-lfvg-v2-phase-a", help="execute the sealed deterministic HTF LFVG V2 Phase-A candidates once")
    command.add_argument("--phase-a-bars-manifest", required=True)
    command.add_argument("--artifact-root", required=True)
    command.add_argument("--repository-root", required=True)
    command = sub.add_parser("htf-lfvg-v2-funnel-diagnostic", help="read-only HTF LFVG V2 synthetic funnel diagnostic")
    command.add_argument("--synthetic-manifest", required=True)
    command = sub.add_parser("fib09-v1-synthetic-materialize", help="materialize the sealed Fib09 V1 synthetic-only contract")
    command.add_argument("--artifact-root", required=True); command.add_argument("--repository-root", required=True)
    command = sub.add_parser("fib09-v1-synthetic-run", help="run deterministic synthetic Fib09 V1 fixtures only")
    command.add_argument("--synthetic-bars", required=True); command.add_argument("--artifact-root", required=True); command.add_argument("--repository-root", required=True)
    command = sub.add_parser("fib09-v1-development-diagnostic", help="verify explicit Fib09 V1 manifests")
    command.add_argument("--eth-manifest", required=True); command.add_argument("--btc-manifest", required=True)
    command = sub.add_parser("fib09-v1-development", help="future deterministic Fib09 V1 development runner")
    command.add_argument("--eth-manifest", required=True); command.add_argument("--btc-manifest", required=True); command.add_argument("--artifact-root", required=True); command.add_argument("--repository-root", required=True)
    command = sub.add_parser("fib09-v1-holdout", help="locked holdout refusal")
    command = imbalance_sub.add_parser("validate-btc-long-only-v5-artifacts", help="verify immutable V5 artifact identities and content hashes")
    command.add_argument("artifact_root")
    command = value_area_sub.add_parser("download"); command.add_argument("month", help="YYYY-MM"); command.add_argument("--symbol", default="BTCUSDT"); command.add_argument("--cache-root", default="data/value_area_trap"); command.add_argument("--allow-network", action="store_true")
    command = value_area_sub.add_parser("import-archive"); command.add_argument("archive"); command.add_argument("--symbol", default="BTCUSDT"); command.add_argument("--cache-root", default="data/value_area_trap")
    command = value_area_sub.add_parser("import-calendar"); command.add_argument("source"); command.add_argument("output")
    command = value_area_sub.add_parser("validate-data"); command.add_argument("parquet")
    command = value_area_sub.add_parser("audit-thesnowguru-data", help="read-only inventory and quality audit of local TheSnowGuru S&P/Nasdaq candidates")
    command.add_argument("--source-root", default="external_data/thesnowguru"); command.add_argument("--staging-root", default="data/value_area_trap/staging"); command.add_argument("--repository-root", default=".")
    command = value_area_sub.add_parser("import-thesnowguru-es", help="import only an audit-proven exact-CVD S&P futures tick candidate")
    command.add_argument("--audit-path", required=True); command.add_argument("--cache-root", default="data/value_area_trap")
    command = value_area_sub.add_parser("import-thesnowguru-es-tick-rule", help="import the explicitly exploratory 2018 ES tick-rule-CVD pilot")
    command.add_argument("--audit-path", required=True); command.add_argument("--cache-root", default="data/value_area_trap")
    command = value_area_sub.add_parser("run-thesnowguru-es-tick-rule-pilot", help="reserved runner for the explicitly exploratory ES pilot")
    command.add_argument("--dataset-root", required=True)
    command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default=".")
    command = value_area_sub.add_parser("validate-thesnowguru-es-tick-rule-dataset", help="validate immutable exploratory ES tick-rule five-minute bars")
    command.add_argument("--dataset-root", required=True)
    command = value_area_sub.add_parser("ingest-range", help="resumable download and normalization of immutable monthly BTCUSDT aggregate-trade partitions")
    command.add_argument("--start-month", required=True); command.add_argument("--end-month", required=True); command.add_argument("--symbol", default="BTCUSDT"); command.add_argument("--cache-root", default="data/value_area_trap"); command.add_argument("--allow-network", action="store_true"); command.add_argument("--allow-gap-repair", action="store_true", help="opt in to fetching only proven missing aggregate-trade IDs")
    command = value_area_sub.add_parser("validate-manifest", help="validate an immutable combined monthly aggregate-trade manifest")
    command.add_argument("manifest")
    command = value_area_sub.add_parser("run-frozen", help="run the packaged, non-optimizing UTC_24H_SESSION strategy")
    command.add_argument("--variant", required=True); command.add_argument("--data-manifest", required=True); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default="."); command.add_argument("--registry", default=argparse.SUPPRESS); command.add_argument("--auto-approve", action="store_true"); command.add_argument("--reuse-verified-implementation", action="store_true")
    command = value_area_sub.add_parser("ingest-cross-market", help="resumably ingest frozen XAUUSDT, QQQUSDT, and SPYUSDT proxy evidence")
    command.add_argument("--symbols", nargs="+", default=["XAUUSDT", "QQQUSDT", "SPYUSDT"]); command.add_argument("--start-month", default="2026-05"); command.add_argument("--end-month", default="2026-07"); command.add_argument("--cache-root", default="data/value_area_trap"); command.add_argument("--metadata-artifact"); command.add_argument("--allow-network", action="store_true"); command.add_argument("--allow-gap-repair", action="store_true")
    command = value_area_sub.add_parser("validate-cross-market", help="validate three pinned cross-market aggregate-trade manifests")
    command.add_argument("--manifest", action="append", required=True, metavar="SYMBOL=PATH")
    command = value_area_sub.add_parser("run-frozen-cross-market", help="run descriptive frozen cross-market robustness evaluation")
    command.add_argument("--manifest", action="append", required=True, metavar="SYMBOL=PATH"); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default=".")
    command = value_area_sub.add_parser("validate-equity-variant-study", help="validate the sealed QQQUSDT/SPYUSDT exploratory in-sample study inputs")
    command.add_argument("--manifest", action="append", required=True, metavar="SYMBOL=PATH")
    command = value_area_sub.add_parser("run-equity-variant-study", help="run only the six pre-registered QQQUSDT/SPYUSDT exploratory variants")
    command.add_argument("--manifest", action="append", required=True, metavar="SYMBOL=PATH"); command.add_argument("--artifact-root", default="research_runs"); command.add_argument("--repository-root", default=".")
    command = value_area_sub.add_parser("materialize-variants", help="write immutable, unexecuted ValueAreaTrap variant specifications")
    command.add_argument("--data-manifest", required=True)
    command.add_argument("--artifact-root", default="research_runs/ValueAreaTrapVariants")
    command.add_argument("--repository-root", default=".")
    repository = sub.add_parser("repository", help="repository safety checks")
    repository_sub = repository.add_subparsers(dest="repository_command", required=True)
    command = repository_sub.add_parser("worktree-preflight")
    command.add_argument("--repository-root", default=".")
    command.add_argument("--max-path-length", type=int, default=240)
    command.add_argument("--probe", action="store_true")
    command.add_argument("--format", choices=["json", "text"], default="json")
    implementation = sub.add_parser("implementation", help="durable external implementation jobs")
    implementation_sub = implementation.add_subparsers(dest="implementation_command", required=True)
    for name in ("job", "status", "ingest"):
        command = implementation_sub.add_parser(name); command.add_argument("run_id")
    executor = sub.add_parser("codex-executor", help="run an approved implementation job outside Smithers")
    executor_sub = executor.add_subparsers(dest="executor_command", required=True)
    for name in ("run", "status", "resume", "reconcile", "reclassify-legacy-timeout"):
        command = executor_sub.add_parser(name); command.add_argument("run_id")
    command = executor_sub.add_parser("retry", help="replace a stale interrupted implementation job with a new immutable job")
    command.add_argument("run_id")
    command.add_argument("--stale-after-seconds", type=int, default=300)
    specification_executor = sub.add_parser("specification-executor", help="run an external read-only specification job")
    specification_executor_sub = specification_executor.add_subparsers(dest="specification_executor_command", required=True)
    for name in ("run", "status", "resume", "inspect"):
        command = specification_executor_sub.add_parser(name); command.add_argument("run_id")
    command = specification_executor_sub.add_parser("run-job"); command.add_argument("run_id"); command.add_argument("job_id")
    command = sub.add_parser("new-strategy"); command.add_argument("path")
    sub.add_parser("list-strategies")
    command = sub.add_parser("status"); command.add_argument("strategy_id")
    command = sub.add_parser("run", help="Phase F1/F2 end-to-end master pipeline"); command.add_argument("strategy_file"); command.add_argument("--repository-root", default="."); command.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None); command.add_argument("--mode", choices=["dry_run", "real_run"]); command.add_argument("--prebuilt-spec"); command.add_argument("--run-id", dest="run_id_override"); command.add_argument("--implementation-enabled", action="store_true"); command.add_argument("--allow-proxy-data", action="store_true"); command.add_argument("--data-manifest", dest="data_manifest_path"); command.add_argument("--worktree-parent", dest="worktree_parent"); command.add_argument("--implementation-timeout-seconds", type=int, default=1800); command.add_argument("--research-scenario", default="strong-stable"); command.add_argument("--prop-scenario", default="profitable"); command.add_argument("--portfolio-scenario", default="complementary"); command.add_argument("--product", default="Alpha Futures Zero 25K")
    command = sub.add_parser("resume", help="resume a Phase F1 master run"); command.add_argument("run_id"); command.add_argument("--repository-root", default=".")
    command = sub.add_parser("approve", help="approve or reject a Phase F1 generated specification"); command.add_argument("run_id"); command.add_argument("--decision", choices=["APPROVE", "REJECT"], default="APPROVE"); command.add_argument("--note")
    command = sub.add_parser("report", help="show a Phase F1 final report"); command.add_argument("run_id")
    command = sub.add_parser("artifacts", help="list Phase F1 artifacts"); command.add_argument("run_id")
    command = sub.add_parser("cancel", help="cancel a Phase F1 master run"); command.add_argument("run_id"); command.add_argument("--reason", default="cancelled by operator")
    command = sub.add_parser("worktree", help="show persisted worktree metadata"); command.add_argument("run_id")
    command = sub.add_parser("verify-data", help="check real-mode data availability"); command.add_argument("run_id")
    adapters = sub.add_parser("adapters", help="inspect registered real strategy adapters")
    adapters_sub = adapters.add_subparsers(dest="adapters_command", required=True)
    adapters_sub.add_parser("list")
    for name in ("inspect", "validate", "capabilities"):
        command = adapters_sub.add_parser(name); command.add_argument("strategy_id")
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
    specification = sub.add_parser("specification", help="inspect durable natural-language specification intake")
    specification_sub = specification.add_subparsers(dest="specification_command", required=True)
    for name in ("status", "attempts", "validate", "errors", "latest"):
        command = specification_sub.add_parser(name); command.add_argument("run_id")
    workflow = sub.add_parser("workflow", help="typed Smithers bridge commands")
    workflow.add_argument("workflow_command", choices=["generate-spec", "validate-spec", "specification-status", "specification-attempts", "specification-errors", "specification-latest", "register-generated-spec", "approve", "implementation-plan", "execute-codex", "record-codex-result", "run-tests", "run-required-tests", "research-start", "research-run-baseline", "research-edge-gate", "research-analyze", "research-propose-round", "research-run-round", "research-review-round", "research-freeze-family", "research-freeze-candidate", "research-walk-forward", "research-holdout", "research-stress", "research-throughput", "research-final-review", "research-status", "research-journal", "prop-start", "prop-verify-rules", "prop-verify-contracts", "prop-reconcile", "prop-run-risk", "prop-run-scenarios", "prop-economics", "prop-final-review", "prop-status", "prop-journal", "portfolio-create", "portfolio-eligible-strategies", "portfolio-generate-candidates", "portfolio-merge-signals", "portfolio-analyze-overlap", "portfolio-analyze-correlation", "portfolio-run-risk", "portfolio-run-prop", "portfolio-run-ablation", "portfolio-run-stress", "portfolio-final-review", "portfolio-status", "portfolio-journal", "master-start", "master-specification-status", "master-specification-retry", "master-approve", "master-resume", "master-status", "master-implementation", "master-worktree", "master-verify-data", "master-report", "master-artifacts", "master-cancel", "technical-verification", "final-status", "verification-create-manifest", "verification-run", "diagnose-tools"])
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
    prop = sub.add_parser("prop", help="deterministic Phase D futures and prop-account research")
    prop_sub = prop.add_subparsers(dest="prop_command", required=True)
    for name in ("dry-run", "start", "verify-rules", "verify-contracts", "reconcile", "run-risk", "run-scenarios", "scenario-status", "economics", "final-review", "journal", "status"):
        command = prop_sub.add_parser(name); command.add_argument("strategy_id"); command.add_argument("--scenario", default="profitable"); command.add_argument("--repository-root", default="."); command.add_argument("--product", default="Alpha Futures Zero 25K")
    portfolio = sub.add_parser("portfolio", help="deterministic Phase E multi-strategy portfolio research")
    portfolio_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    command = portfolio_sub.add_parser("create"); command.add_argument("portfolio_config"); command.add_argument("--repository-root", default=".")
    command = portfolio_sub.add_parser("eligible-strategies"); command.add_argument("--exploratory-prop", action="store_true"); command.add_argument("--non-prop", action="store_true")
    command = portfolio_sub.add_parser("dry-run"); command.add_argument("--portfolio-id", default="phase-e-dry-run"); command.add_argument("--scenario", default="complementary"); command.add_argument("--repository-root", default=".")
    for name in ("generate-candidates", "merge-signals", "analyze-overlap", "analyze-correlation", "run-risk", "run-prop", "run-ablation", "run-stress", "final-review", "status", "journal"):
        command = portfolio_sub.add_parser(name); command.add_argument("portfolio_id"); command.add_argument("--repository-root", default="."); command.add_argument("--scenario", default="complementary")
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
        # Data-only and repository-preflight commands must not create an
        # unrelated SQLite registry as a side effect.  Their durable artifacts
        # are explicitly controlled by their own output arguments.
        controller = None
        registry = None
        if args.command not in {"value-area-trap", "value-area-acceptance", "imbalance-vwap-ride", "repository", "v5-candidate-run", "lsmr-v1-materialize", "lsmr-v1-phase-a", "lsmr-v2-strict-materialize", "lsmr-v2-phase-a", "vbtc-v1-synthetic-materialize", "vbtc-v1-phase-a", "vbtc-v2-synthetic-materialize", "vbtc-v2-phase-a", "htf-lfvg-v1-synthetic-materialize", "htf-lfvg-v1-phase-a", "htf-lfvg-v1-phase-a-funnel-diagnostic", "htf-lfvg-v2-synthetic-materialize", "htf-lfvg-v2-phase-a", "htf-lfvg-v2-funnel-diagnostic", "fib09-v1-synthetic-materialize", "fib09-v1-synthetic-run", "fib09-v1-development-diagnostic", "fib09-v1-development", "fib09-v1-holdout"}:
            controller = _controller(args.registry)
            registry = controller.registry
        if args.command == "v5-candidate-run":
            from .imbalance_vwap_ride.v5_runner import run_v5_candidate_cli
            _print(run_v5_candidate_cli(phase_a_manifest=args.phase_a_manifest, artifact_root=args.artifact_root))
        elif args.command == "fib09-v1-synthetic-materialize":
            from .fib_retracement_continuation_v1.runner import materialize_synthetic
            _print(materialize_synthetic(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "fib09-v1-synthetic-run":
            from .fib_retracement_continuation_v1.runner import run_synthetic
            from .fib_retracement_continuation_v1.models import Bar
            raw=json.loads(Path(args.synthetic_bars).read_text(encoding="utf-8")); bars={key:[Bar(**row) for row in rows] for key,rows in raw.items()}
            _print(run_synthetic(bars_by_candidate=bars, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "fib09-v1-development-diagnostic":
            from .fib_retracement_continuation_v1.runner import development_diagnostic
            _print(development_diagnostic(eth_manifest=args.eth_manifest, btc_manifest=args.btc_manifest))
        elif args.command == "fib09-v1-development":
            from .fib_retracement_continuation_v1.runner import run_development
            _print(run_development(eth_manifest=args.eth_manifest, btc_manifest=args.btc_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "fib09-v1-holdout":
            from .fib_retracement_continuation_v1.runner import run_holdout
            _print(run_holdout())
        elif args.command == "lsmr-v1-materialize":
            from .liquidity_sweep_mean_reversion.runner import materialize_lsmr_v1_contract
            _print(materialize_lsmr_v1_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "lsmr-v1-phase-a":
            from .liquidity_sweep_mean_reversion.runner import run_lsmr_v1_phase_a
            _print(run_lsmr_v1_phase_a(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "lsmr-v2-strict-materialize":
            from .liquidity_sweep_mean_reversion_v2.runner import materialize_lsmr_v2_strict_contract
            _print(materialize_lsmr_v2_strict_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "lsmr-v2-phase-a":
            from .liquidity_sweep_mean_reversion_v2.runner import run_lsmr_v2_phase_a
            _print(run_lsmr_v2_phase_a(phase_a_bars_manifest=args.phase_a_bars_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "vbtc-v1-synthetic-materialize":
            from .volatility_breakout_trend_continuation.runner import materialize_synthetic_contract
            _print(materialize_synthetic_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "vbtc-v1-phase-a":
            from .volatility_breakout_trend_continuation.runner import run_phase_a
            _print(run_phase_a(phase_a_bars_manifest=args.phase_a_bars_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "vbtc-v2-synthetic-materialize":
            from .volatility_breakout_trend_continuation_v2.runner import materialize_synthetic_contract
            _print(materialize_synthetic_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "vbtc-v2-phase-a":
            from .volatility_breakout_trend_continuation_v2.runner import run_phase_a
            _print(run_phase_a(phase_a_bars_manifest=args.phase_a_bars_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "htf-lfvg-v1-synthetic-materialize":
            from .htf_level_liquidity_fvg.runner import materialize_htf_lfvg_v1_contract
            _print(materialize_htf_lfvg_v1_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "htf-lfvg-v1-phase-a":
            from .htf_level_liquidity_fvg.runner import run_htf_lfvg_v1_phase_a
            _print(run_htf_lfvg_v1_phase_a(phase_a_bars_manifest=args.phase_a_bars_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "htf-lfvg-v1-phase-a-funnel-diagnostic":
            from .htf_level_liquidity_fvg.runner import htf_lfvg_v1_phase_a_funnel_diagnostic
            _print(htf_lfvg_v1_phase_a_funnel_diagnostic())
        elif args.command == "htf-lfvg-v2-synthetic-materialize":
            from .htf_level_liquidity_fvg_v2.runner import materialize_htf_lfvg_v2_contract
            _print(materialize_htf_lfvg_v2_contract(artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "htf-lfvg-v2-phase-a":
            from .htf_level_liquidity_fvg_v2.runner import run_htf_lfvg_v2_phase_a
            _print(run_htf_lfvg_v2_phase_a(phase_a_bars_manifest=args.phase_a_bars_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root))
        elif args.command == "htf-lfvg-v2-funnel-diagnostic":
            from .htf_level_liquidity_fvg_v2.runner import synthetic_funnel_diagnostic
            _print(synthetic_funnel_diagnostic(args.synthetic_manifest))
        elif args.command == "init":
            print(f"initialized registry: {Path(args.registry)}")
        elif args.command == "value-area-trap":
            if args.value_area_command == "download":
                from .value_area_trap.data import AggregateTradeImporter
                importer = AggregateTradeImporter(args.cache_root)
                path = importer.download_month(args.symbol, args.month, allow_network=args.allow_network)
                _print({"archive": str(path.resolve()), "source": "Binance USD-M Futures public historical archive", "network_used": True})
            elif args.value_area_command == "import-archive":
                from .value_area_trap.data import AggregateTradeImporter
                importer = AggregateTradeImporter(args.cache_root)
                records = importer.records_from_archive(args.archive)
                parquet, manifest = importer.ingest_records(records, symbol=args.symbol, source_files=[str(Path(args.archive).resolve())])
                _print({"parquet": str(parquet.resolve()), "manifest": manifest.model_dump(mode="json")})
            elif args.value_area_command == "import-calendar":
                from .value_area_trap.alpha_zero import import_usd_calendar
                _print(import_usd_calendar(args.source, args.output).model_dump(mode="json"))
            elif args.value_area_command == "audit-thesnowguru-data":
                from .value_area_trap.thesnowguru import audit_thesnowguru_data
                _print(audit_thesnowguru_data(repository_root=args.repository_root, source_root=args.source_root, staging_root=args.staging_root).model_dump(mode="json"))
            elif args.value_area_command == "import-thesnowguru-es":
                from .value_area_trap.thesnowguru import import_thesnowguru_es
                _print(import_thesnowguru_es(audit_path=args.audit_path, cache_root=args.cache_root))
            elif args.value_area_command == "import-thesnowguru-es-tick-rule":
                from .value_area_trap.es_tick_rule import import_es_tick_rule
                _print(import_es_tick_rule(audit_path=args.audit_path, cache_root=args.cache_root))
            elif args.value_area_command == "run-thesnowguru-es-tick-rule-pilot":
                from .value_area_trap.es_tick_rule import run_es_tick_rule_pilot
                _print(run_es_tick_rule_pilot(dataset_root=args.dataset_root, artifact_root=args.artifact_root, repository_root=args.repository_root))
            elif args.value_area_command == "validate-thesnowguru-es-tick-rule-dataset":
                from .value_area_trap.es_tick_rule import validate_es_tick_rule_dataset
                _print(validate_es_tick_rule_dataset(args.dataset_root))
            elif args.value_area_command == "ingest-range":
                from .value_area_trap.data import AggregateTradeImporter
                importer = AggregateTradeImporter(args.cache_root)
                manifest_path, manifest = importer.ingest_monthly_range(symbol=args.symbol, start_month=args.start_month, end_month=args.end_month, allow_network=args.allow_network, allow_gap_repair=args.allow_gap_repair)
                _print({"manifest_path": str(manifest_path.resolve()), "manifest": manifest.model_dump(mode="json"), "network_used": args.allow_network, "gap_repair_enabled": args.allow_gap_repair, "months": importer.last_ingestion_diagnostics})
            elif args.value_area_command == "validate-manifest":
                from .value_area_trap.data import AggregateTradeImporter
                manifest = AggregateTradeImporter(".").validate_monthly_manifest(args.manifest)
                _print({"path": str(Path(args.manifest).resolve()), "valid": True, "dataset_hash": manifest.normalized_dataset_hash, "row_count": manifest.row_count, "partition_count": len(manifest.partitions), "errors": []})
            elif args.value_area_command == "run-frozen":
                from .value_area_trap.frozen import FrozenRunRequest, FrozenValueAreaTrapService
                _print(FrozenValueAreaTrapService().run(FrozenRunRequest(variant=args.variant, data_manifest=str(Path(args.data_manifest).resolve()), artifact_root=str(Path(args.artifact_root).resolve()), registry_path=str(Path(args.registry).resolve()), repository_root=str(Path(args.repository_root).resolve()), auto_approve=args.auto_approve, reuse_verified_implementation=args.reuse_verified_implementation)).model_dump(mode="json"))
            elif args.value_area_command == "ingest-cross-market":
                from .value_area_trap.cross_market import CrossMarketIngestRequest, ingest_cross_market
                _print(ingest_cross_market(CrossMarketIngestRequest(symbols=args.symbols, cache_root=str(Path(args.cache_root).resolve()), start_month=args.start_month, end_month=args.end_month, metadata_artifact=str(Path(args.metadata_artifact).resolve()) if args.metadata_artifact else None, allow_network=args.allow_network, allow_gap_repair=args.allow_gap_repair)))
            elif args.value_area_command == "validate-cross-market":
                from .value_area_trap.cross_market import validate_cross_market
                _print(validate_cross_market(_symbol_manifest_map(args.manifest)))
            elif args.value_area_command == "run-frozen-cross-market":
                from .value_area_trap.cross_market import FrozenCrossMarketRequest, FrozenCrossMarketService
                _print(FrozenCrossMarketService().run(FrozenCrossMarketRequest(manifests=_symbol_manifest_map(args.manifest), artifact_root=str(Path(args.artifact_root).resolve()), repository_root=str(Path(args.repository_root).resolve()))).model_dump(mode="json"))
            elif args.value_area_command == "validate-equity-variant-study":
                from .value_area_trap.equity_variants import validate_equity_variant_study
                _print(validate_equity_variant_study(_symbol_manifest_map(args.manifest)))
            elif args.value_area_command == "run-equity-variant-study":
                from .value_area_trap.equity_variants import EquityVariantStudyRequest, EquityVariantStudyService
                _print(EquityVariantStudyService().run(EquityVariantStudyRequest(manifests=_symbol_manifest_map(args.manifest), artifact_root=str(Path(args.artifact_root).resolve()), repository_root=str(Path(args.repository_root).resolve()))).model_dump(mode="json"))
            elif args.value_area_command == "materialize-variants":
                from .value_area_trap.variants import materialize_variants
                _print(materialize_variants(
                    repository_root=args.repository_root,
                    data_manifest_path=args.data_manifest,
                    artifact_root=args.artifact_root,
                ))
            else:
                from .value_area_trap.data import AggregateTradeManifest, PARQUET_SCHEMA
                parquet = Path(args.parquet).resolve(); manifest_path = parquet.with_name("manifest.json")
                report = {"path": str(parquet), "valid": False, "row_count": None, "dataset_hash": None, "schema_version": None, "errors": [], "warnings": []}
                try:
                    if not parquet.is_file():
                        raise ValueError(f"aggregate-trade parquet does not exist: {parquet}")
                    if not manifest_path.is_file():
                        raise ValueError(f"adjacent aggregate-trade manifest does not exist: {manifest_path}")
                    manifest = AggregateTradeManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
                    report.update({"row_count": manifest.row_count, "dataset_hash": manifest.normalized_dataset_hash, "schema_version": manifest.schema_version})
                    parquet_file = pq.ParquetFile(parquet)
                    actual_schema = parquet_file.schema_arrow
                    if actual_schema.names != list(PARQUET_SCHEMA.names):
                        raise ValueError(f"malformed aggregate-trade schema: expected columns {list(PARQUET_SCHEMA.names)}, detected {actual_schema.names}")
                    actual_rows = parquet_file.metadata.num_rows
                    if actual_rows != manifest.row_count:
                        raise ValueError(f"row count mismatch: manifest={manifest.row_count}, parquet={actual_rows}")
                    report["valid"] = True
                except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                    if isinstance(exc, ValidationError):
                        details = "; ".join(
                            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                            for item in exc.errors()
                        )
                        report["errors"] = [f"manifest validation error: {details}"]
                    else:
                        report["errors"] = [str(exc)]
                    _print(report)
                    return 2
                _print(report)
        elif args.command == "value-area-acceptance":
            from .value_area_acceptance import run_btc_2024_study
            _print(run_btc_2024_study(data_manifest=args.data_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root, non_interactive=args.non_interactive))
        elif args.command == "imbalance-vwap-ride":
            if args.imbalance_command == "validate-source":
                from .imbalance_vwap_ride.footprint import validate_source_manifest
                report = validate_source_manifest(args.data_manifest, require_pinned=True, verify_parquet_hashes=True)
                _print({key: value for key, value in report.items() if key != "manifest"})
                return 0 if report["valid"] else 2
            if args.imbalance_command == "build-footprint":
                from .imbalance_vwap_ride.footprint import build_footprint_dataset
                _print(build_footprint_dataset(args.data_manifest, args.cache_root, batch_size=args.batch_size, require_pinned=True, verify_source_hashes=True))
            elif args.imbalance_command == "validate-footprint":
                from .imbalance_vwap_ride.footprint import validate_footprint_dataset
                report = validate_footprint_dataset(args.footprint_root)
                _print(report)
                return 0 if report["valid"] else 2
            elif args.imbalance_command == "run-btc-exploratory-study":
                from .imbalance_vwap_ride.runner import verify_and_run_sealed_study
                result = verify_and_run_sealed_study(data_manifest=args.data_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root, footprint_cache_root=args.footprint_cache_root, batch_size=args.batch_size)
                _print(result)
                return 0 if result["status"] in {"COMPLETED", "DEVELOPMENT_EDGE_NOT_FOUND"} else 2
            elif args.imbalance_command == "run-btc-macro-bins-v2-study":
                from .imbalance_vwap_ride.v2_runner import verify_and_run_sealed_v2_study
                result = verify_and_run_sealed_v2_study(data_manifest=args.data_manifest, artifact_root=args.artifact_root, repository_root=args.repository_root, footprint_cache_root=args.footprint_cache_root, batch_size=args.batch_size)
                _print(result)
                return 0 if result["status"] in {"COMPLETED", "DEVELOPMENT_EDGE_NOT_FOUND"} else 2
            elif args.imbalance_command == "run-btc-long-only-v3-study":
                from .imbalance_vwap_ride.v3_runner import verify_and_run_sealed_v3_study
                result = verify_and_run_sealed_v3_study(artifact_root=args.artifact_root, repository_root=args.repository_root, data_cache_root=args.data_cache_root, footprint_cache_root=args.footprint_cache_root, batch_size=args.batch_size, allow_authorized_downloads=True)
                _print(result)
                return 0 if result["status"] in {"COMPLETED", "DEVELOPMENT_EDGE_NOT_FOUND"} else 2
            elif args.imbalance_command == "run-btc-long-only-v4-study":
                from .imbalance_vwap_ride.v4_runner import verify_and_run_sealed_v4_study
                result = verify_and_run_sealed_v4_study(artifact_root=args.artifact_root, repository_root=args.repository_root, data_cache_root=args.data_cache_root, footprint_cache_root=args.footprint_cache_root, batch_size=args.batch_size, allow_authorized_downloads=True, alpha_rules_artifact=args.alpha_rules_artifact)
                _print(result)
                return 0 if result["status"] in {"PHASE_A_NO_ROBUST_CANDIDATE", "LOCKED_TEST_FAILED", "LOCKED_TEST_PASSED"} else 2
            elif args.imbalance_command == "validate-btc-long-only-v4-source":
                from .imbalance_vwap_ride.v4_data import validate_v4_source_manifest
                report = validate_v4_source_manifest(args.manifest, phase=args.phase, verify_archives=not args.skip_archive_verification)
                _print({key: value for key, value in report.items() if key != "manifest"})
                return 0 if report["valid"] else 2
            elif args.imbalance_command == "validate-btc-long-only-v4-artifacts":
                from .imbalance_vwap_ride.v4_artifacts import validate_v4_artifact_tree
                report = validate_v4_artifact_tree(args.artifact_root)
                _print(report)
                return 0 if report["valid"] else 2
            elif args.imbalance_command == "run-btc-long-only-v5-study":
                from .imbalance_vwap_ride.v5_runner import verify_and_run_sealed_v5_study
                result = verify_and_run_sealed_v5_study(artifact_root=args.artifact_root, repository_root=args.repository_root)
                _print(result)
                return 0 if result["status"] == "PHASE_A_NO_ROBUST_CANDIDATE" else 2
            elif args.imbalance_command == "validate-btc-long-only-v5-artifacts":
                from .imbalance_vwap_ride.v5_artifacts import validate_v5_artifact_tree
                _print(validate_v5_artifact_tree(args.artifact_root))
                return 0
        elif args.command == "repository":
            from .repository.worktree_preflight import run_worktree_preflight
            report = run_worktree_preflight(args.repository_root, max_path_length=args.max_path_length, probe=args.probe)
            if args.format == "text":
                print(f"safe_for_isolated_worktree={report.safe_for_isolated_worktree} tracked_paths={report.tracked_path_count} issues={len(report.issues)} probe={report.probe_status}")
                for issue in report.issues:
                    print(f"{issue.error_code}: {issue.tracked_path} - {issue.explanation}")
            else:
                _print(report.model_dump(mode="json"))
            return 0 if report.safe_for_isolated_worktree else 3
        elif args.command == "new-strategy":
            spec = load_strategy_spec(args.path)
            config = load_pipeline_config(Path("configs/research_pipeline/defaults.yaml"), strategy_id=spec.strategy_id)
            _print(controller.register_strategy(spec, str(Path(args.path).resolve()), config["budgets"]))
        elif args.command == "list-strategies":
            _print(registry.list_strategies())
        elif args.command == "status":
            try:
                _print(controller.status(args.strategy_id))
            except ResearchPipelineError:
                from .phase_f1.service import MasterPipelineService
                registry_path = args.registry
                default_registry = os.environ.get("RESEARCH_PIPELINE_REGISTRY", "research_registry/research_pipeline.sqlite3")
                if Path(registry_path).resolve() == Path(default_registry).resolve():
                    registry_path = MasterPipelineService.discover_registry_path(args.strategy_id) or registry_path
                _print(MasterPipelineService(registry_path).status(args.strategy_id))
        elif args.command == "run":
            from .phase_f1.service import MasterPipelineService
            service = MasterPipelineService(args.registry, args.repository_root)
            if args.data_manifest_path:
                if args.mode == "dry_run" or args.dry_run is True:
                    raise ValueError("--data-manifest requires real_run; remove --dry-run or use --mode real_run")
                selected_mode = "real_run"
            else:
                selected_mode = args.mode or ("real_run" if args.dry_run is False else "dry_run")
            options = service.input_model(args.strategy_file, args.repository_root, registry_path=args.registry, dry_run=selected_mode == "dry_run", mode=selected_mode, allow_proxy_data=args.allow_proxy_data, data_manifest_path=args.data_manifest_path, worktree_parent=args.worktree_parent, implementation_enabled=args.implementation_enabled, implementation_timeout_seconds=args.implementation_timeout_seconds, research_scenario=args.research_scenario, prop_scenario=args.prop_scenario, portfolio_scenario=args.portfolio_scenario, prop_product=args.product, run_id_override=args.run_id_override)
            if args.prebuilt_spec: options = options.model_copy(update={"prebuilt_spec_path": str(Path(args.prebuilt_spec).resolve())})
            _print(service.start(options))
        elif args.command == "resume":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry, args.repository_root).resume(args.run_id))
        elif args.command == "approve":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).approve(args.run_id, args.decision, args.note))
        elif args.command == "report":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).report(args.run_id))
        elif args.command == "artifacts":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).artifacts(args.run_id))
        elif args.command == "cancel":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).cancel(args.run_id, args.reason))
        elif args.command == "implementation":
            from .implementation.jobs import ImplementationJobService
            jobs = ImplementationJobService(args.registry)
            if args.implementation_command == "job": _print(jobs.create(args.run_id))
            elif args.implementation_command == "status": _print(jobs.status(args.run_id))
            elif args.implementation_command == "ingest": _print(jobs.ingest(args.run_id))
        elif args.command == "codex-executor":
            from .implementation.executor import ExternalCodexExecutor
            from .implementation.jobs import ImplementationJobService
            service = ExternalCodexExecutor(args.registry)
            if args.executor_command == "status": _print(service.status(args.run_id))
            elif args.executor_command == "retry":
                _print(ImplementationJobService(args.registry).retry(args.run_id, stale_after_seconds=args.stale_after_seconds))
            elif args.executor_command == "reconcile":
                _print(ImplementationJobService(args.registry).reconcile(args.run_id))
            elif args.executor_command == "reclassify-legacy-timeout":
                _print(ImplementationJobService(args.registry).reclassify_legacy_timeout(args.run_id))
            else:
                result = service.run(args.run_id)
                _print(result)
                if result.get("status") not in {"SUCCEEDED", "COMPLETED"}: return 3
        elif args.command == "specification-executor":
            from .specification_executor.executor import ExternalSpecificationExecutor
            from .specification_executor.jobs import SpecificationJobService
            if args.specification_executor_command in {"status", "inspect"}:
                service = SpecificationJobService(args.registry)
                _print(service.status(args.run_id) if args.specification_executor_command == "status" else service.inspect(args.run_id))
            else:
                service = ExternalSpecificationExecutor(args.registry)
                result = service.run(args.run_id, args.job_id if args.specification_executor_command == "run-job" else None)
                _print(result)
                if result.get("status") != "SUCCEEDED": return 3
        elif args.command == "worktree":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).worktree(args.run_id))
        elif args.command == "verify-data":
            from .phase_f1.service import MasterPipelineService
            _print(MasterPipelineService(args.registry).verify_data(args.run_id))
        elif args.command == "adapters":
            from .adapters.registry import default_adapter_registry
            if args.adapters_command == "list":
                _print({"schema_version": default_adapter_registry().schema_version, "adapters": default_adapter_registry().list()})
            else:
                from .adapters.registry import default_adapter_registry
                spec = registry.get_specification(args.strategy_id)
                health = default_adapter_registry().inspect(spec, Path(args.repository_root if hasattr(args, "repository_root") else "."))
                if args.adapters_command == "capabilities": _print(health.capabilities.model_dump(mode="json"))
                else: _print(health.model_dump(mode="json"))
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
        elif args.command == "specification":
            from .phase_b.services import PhaseBService
            service = PhaseBService(args.registry)
            if args.specification_command == "attempts": result = service.specification_attempts(args.run_id)
            elif args.specification_command == "errors": result = service.specification_errors(args.run_id)
            elif args.specification_command == "latest": result = service.specification_latest(args.run_id)
            else: result = service.specification_status(args.run_id)
            _print(result)
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
        elif args.command == "prop":
            from .prop.services import PropResearchService
            if args.prop_command == "dry-run":
                from .prop.fixtures import run_prop_dry_run
                _print(run_prop_dry_run(args.registry, args.repository_root, args.strategy_id, args.scenario, args.product))
                return 0
            service = PropResearchService(args.registry, repository_root=args.repository_root, scenario=args.scenario)
            command = args.prop_command
            if command == "start": result = service.start(args.strategy_id)
            elif command == "verify-rules": result = service.verify_rules(args.strategy_id, args.product)
            elif command == "verify-contracts": result = service.verify_contracts(args.strategy_id)
            elif command == "reconcile": result = service.reconcile(args.strategy_id)
            elif command == "run-risk": result = service.run_risk(args.strategy_id, args.product)
            elif command in {"run-scenarios", "scenario-status"}: result = service.run_scenarios(args.strategy_id, args.product) if command == "run-scenarios" else service.status(args.strategy_id).get("scenarios", [])
            elif command in {"economics", "final-review"}: result = service.economics(args.strategy_id)
            elif command == "journal": result = {"entries": service.journal(args.strategy_id)}
            elif command == "status": result = service.status(args.strategy_id)
            else: raise ValueError(f"unsupported prop command: {command}")
            _print(result.model_dump(mode="json") if hasattr(result, "model_dump") else result)
        elif args.command == "portfolio":
            from .portfolio.models import PortfolioSpec
            from .portfolio.service import PortfolioService
            if args.portfolio_command == "dry-run":
                from .portfolio.fixtures import run_portfolio_dry_run
                _print(run_portfolio_dry_run(args.registry, args.repository_root, args.portfolio_id, args.scenario))
            elif args.portfolio_command == "create":
                raw = yaml.safe_load(Path(args.portfolio_config).read_text(encoding="utf-8")) or {}
                spec = PortfolioSpec.model_validate(raw)
                _print(PortfolioService(args.registry, args.repository_root).create(spec))
            elif args.portfolio_command == "eligible-strategies":
                _print(PortfolioService(args.registry).eligible(exploratory_prop=args.exploratory_prop, non_prop=args.non_prop))
            else:
                service = PortfolioService(args.registry, args.repository_root, args.scenario)
                command = args.portfolio_command
                if command == "generate-candidates": result = service.generate_candidates(args.portfolio_id)
                elif command == "merge-signals": result = service.merge_signals(args.portfolio_id)
                elif command == "analyze-overlap": result = service.analyze_overlap(args.portfolio_id)
                elif command == "analyze-correlation": result = service.analyze_correlation(args.portfolio_id)
                elif command == "run-risk": result = service.run_risk(args.portfolio_id)
                elif command == "run-prop": result = service.run_prop(args.portfolio_id)
                elif command == "run-ablation": result = service.run_ablation(args.portfolio_id)
                elif command == "run-stress": result = service.run_stress(args.portfolio_id)
                elif command == "final-review": result = service.final_review(args.portfolio_id)
                elif command == "status": result = service.status(args.portfolio_id)
                elif command == "journal": result = {"entries": service.journal(args.portfolio_id)}
                else: raise ValueError(f"unsupported portfolio command: {command}")
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
    except (ResearchPipelineError, ValidationError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        logging.getLogger("research_pipeline").warning("command_error type=%s message=%s", type(exc).__name__, str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
