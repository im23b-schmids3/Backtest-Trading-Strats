"""One-command, fail-closed execution for the verified UTC frozen variant."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from ..adapters.value_area_trap import ValueAreaTrapAdapter
from ..controller.pipeline_controller import PipelineController
from ..enums import PipelineState
from ..errors import RegistryError, SpecificationValidationError
from ..registry.database import Database
from ..registry.repositories import Registry
from ..schemas.splits import SplitDefinition, SplitWindow, calculate_split_hash
from ..schemas.strategy_spec import StrategySpec, StrictModel, calculate_specification_hash, save_strategy_spec
from ..verification.services import VerificationService
from .data import AggregateTradeImporter, MonthlyAggregateTradeManifest
from .variants import PREDECLARED_VARIANTS, build_variant_specification


FROZEN_VARIANT = "UTC_24H_SESSION"
EVIDENCE_LABEL = "Binance BTCUSDT perpetual proxy evidence; not CME MBT or Alpha Futures performance."
PRIMARY_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
PRIMARY_END = datetime(2026, 3, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
APRIL_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
APRIL_END = datetime(2026, 4, 30, 23, 59, 59, 999999, tzinfo=timezone.utc)


class FrozenRunRequest(StrictModel):
    variant: str
    data_manifest: str
    artifact_root: str
    registry_path: str
    repository_root: str
    auto_approve: bool = False
    reuse_verified_implementation: bool = False


class FrozenRunResult(StrictModel):
    run_id: str
    strategy_id: str
    strategy_version: str
    specification_hash: str
    dataset_hash: str
    status: str
    report_path: str
    comparison_path: str
    reused_verified_implementation: bool
    external_executor_required: bool = False


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: Any) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"immutable frozen-run artifact differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def _split(dataset_hash: str, start: datetime, end: datetime, *, symbol: str = "BTCUSDT") -> SplitDefinition:
    # These windows are reporting partitions, not optimization train/validation
    # partitions. The split object only carries deterministic boundaries into
    # the existing adapter contract.
    from datetime import timedelta
    audit_start = end - timedelta(minutes=2)
    raw = {
        "dataset_identifier": f"Binance USD-M Futures:{symbol} perpetual:5m",
        "source_data_hash": dataset_hash,
        "start_timestamp": start,
        "end_timestamp": end,
        "training_boundaries": SplitWindow(start_timestamp=start, end_timestamp=audit_start),
        "validation_boundaries": SplitWindow(start_timestamp=audit_start, end_timestamp=end - timedelta(minutes=1)),
        "holdout_boundaries": SplitWindow(start_timestamp=end - timedelta(minutes=1), end_timestamp=end),
        "created_timestamp": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "split_hash": "pending",
    }
    # SplitDefinition requires non-empty chronological audit windows. Adapter
    # execution uses the enclosing full start/end bounds above.
    candidate = SplitDefinition.model_construct(**raw)
    raw["split_hash"] = calculate_split_hash(candidate)
    return SplitDefinition.model_validate(raw)


def _frozen_spec(manifest: MonthlyAggregateTradeManifest, manifest_path: Path, repository_root: Path) -> StrategySpec:
    variant = next(item for item in PREDECLARED_VARIANTS if item.variant_id == FROZEN_VARIANT)
    configured = str(manifest_path.relative_to(repository_root)) if manifest_path.is_relative_to(repository_root) else str(manifest_path)
    base = build_variant_specification(variant, manifest_path=configured)
    payload = base.model_dump(mode="python")
    payload["version"] = f"utc-24h-session-frozen-{manifest.normalized_dataset_hash[:12]}"
    payload["baseline_parameters"]["dataset_hash"] = manifest.normalized_dataset_hash
    payload["specification_hash"] = "pending"
    candidate = StrategySpec.model_validate(payload, context={"skip_specification_hash_validation": True})
    return candidate.model_copy(update={"specification_hash": calculate_specification_hash(candidate)})


def validate_frozen_specification(spec: StrategySpec, manifest: MonthlyAggregateTradeManifest) -> None:
    expected = {
        "strategy_id": "ValueAreaTrap.UTC_24H_SESSION",
        "session_definition": "UTC_24H_SESSION",
        "session_timezone": "UTC",
        "swing_left_bars": 2,
        "swing_right_bars": 2,
        "breakout_volume_multiplier": "1.5",
        "breakout_volume_lookback_bars": 10,
        "value_area_fraction": "0.70",
        "price_bucket_size": "10",
        "maximum_trades_per_day": 1,
        "same_bar_stop_target_policy": "stop_first",
        "optimization_allowed": False,
        "dataset_hash": manifest.normalized_dataset_hash,
    }
    if spec.strategy_id != expected.pop("strategy_id"):
        raise SpecificationValidationError("frozen strategy_id differs from UTC_24H_SESSION")
    expected_version = f"utc-24h-session-frozen-{manifest.normalized_dataset_hash[:12]}"
    if spec.version != expected_version:
        raise SpecificationValidationError("frozen strategy version differs from its pinned dataset version")
    values = spec.baseline_parameters
    for key, value in expected.items():
        if str(values.get(key)).lower() != str(value).lower():
            raise SpecificationValidationError(f"frozen parameter mismatch: {key}")
    required_contract = {
        "symbol": "BTCUSDT",
        "minimum_breakout_buckets": 1,
        "stop_buffer_buckets": 1,
        "entry_execution": "next_bar_open",
        "quantity": "0.001",
        "minimum_quantity": "0.001",
        "quantity_step": "0.001",
        "price_tick": "0.10",
        "variant": "FULL",
    }
    for key, value in required_contract.items():
        if str(values.get(key)).lower() != str(value).lower():
            raise SpecificationValidationError(f"frozen execution/cost contract mismatch: {key}")
    if any(family.mutable or family.maximum_rounds != 0 for family in spec.parameter_families):
        raise SpecificationValidationError("frozen specification contains mutable parameter research")
    if "next completed 5-minute bar open" not in spec.entry_logic.lower():
        raise SpecificationValidationError("frozen entry execution is not next_bar_open")
    if "existing stop" not in spec.initial_stop_logic.lower() or "existing poc target" not in spec.exit_logic.lower():
        raise SpecificationValidationError("frozen stop or target contract differs")


def _period_summary(artifact: dict[str, Any], root: Path) -> dict[str, Any]:
    metrics = artifact["metrics"]
    trades = json.loads(Path(artifact["input_path"]).with_name("trades.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(root / "strategy_events.parquet") if (root / "strategy_events.parquet").is_file() else pd.DataFrame()
    bars = pd.read_parquet(root / "5m_bars.parquet") if (root / "5m_bars.parquet").is_file() else pd.DataFrame()
    by_month: dict[str, Any] = {}
    months = sorted(set(bars.get("session_date", pd.Series(dtype=str)).astype(str)))
    for month in sorted({value[:7] for value in months}):
        month_trades = [item for item in trades if item["entry_timestamp"][:7] == month]
        month_events = events[events.get("session_date", pd.Series(dtype=str)).astype(str).str.startswith(month)] if not events.empty else events
        month_net = [float(item["net_pnl"]) for item in month_trades]
        gains = sum(item for item in month_net if item > 0)
        losses = -sum(item for item in month_net if item < 0)
        running = peak = maximum_drawdown = 0.0
        for value in month_net:
            running += value; peak = max(peak, running); maximum_drawdown = max(maximum_drawdown, peak - running)
        by_month[month] = {
            "five_minute_bar_count": int(sum(bars.get("session_date", pd.Series(dtype=str)).astype(str).str.startswith(month))),
            "significant_stop_runs": int(sum(month_events.get("state", pd.Series(dtype=str)) == "STOP_RUN_CONFIRMED")) if not month_events.empty else 0,
            "volume_qualified_stop_runs": int(sum(month_events.get("state", pd.Series(dtype=str)) == "STOP_RUN_CONFIRMED")) if not month_events.empty else 0,
            "confirmed_divergences": int(sum(month_events.get("state", pd.Series(dtype=str)) == "DIVERGENCE_CONFIRMED")) if not month_events.empty else 0,
            "return_triggers": int(sum(month_events.get("state", pd.Series(dtype=str)) == "RETURN_TRIGGER")) if not month_events.empty else 0,
            "proposed_setups": int(sum(month_events.get("state", pd.Series(dtype=str)) == "PROPOSED_SETUP")) if not month_events.empty else 0,
            "invalid_setup_count": int(sum(month_events.get("state", pd.Series(dtype=str)) == "INVALIDATED")) if not month_events.empty else 0,
            "executed_trades": len(month_trades),
            "wins": sum(item > 0 for item in month_net),
            "losses": sum(item <= 0 for item in month_net),
            "gross_pnl": sum(float(item["gross_pnl"]) for item in month_trades),
            "net_pnl": sum(month_net),
            "total_costs": sum(float(item["fees"]) + float(item["slippage_cost"]) for item in month_trades),
            "average_trade": sum(month_net) / len(month_net) if month_net else 0.0,
            "profit_factor": gains / losses if losses else None,
            "maximum_drawdown": maximum_drawdown,
        }
    net = [float(item["net_pnl"]) for item in trades]
    longest_loss = current_loss = 0
    for value in net:
        current_loss = current_loss + 1 if value < 0 else 0
        longest_loss = max(longest_loss, current_loss)
    return {
        "metrics": metrics,
        "monthly": by_month,
        "trades_by_month": {month: sum(1 for item in trades if item["entry_timestamp"][:7] == month) for month in by_month},
        "compliance_blocks": metrics.get("compliance_blocks", 0),
        "longest_losing_streak": longest_loss,
        "unique_setup_sequences": len({"|".join(group["state"].astype(str)) for _, group in events.groupby("session_date")}) if not events.empty else 0,
        "exposure_minutes": sum((datetime.fromisoformat(item["exit_timestamp"]) - datetime.fromisoformat(item["entry_timestamp"])).total_seconds() / 60 for item in trades),
        "trade_frequency_per_session": len(trades) / max(1, metrics.get("session_count", 0)),
    }


class FrozenValueAreaTrapService:
    def run(self, request: FrozenRunRequest) -> FrozenRunResult:
        if request.variant != FROZEN_VARIANT or not request.auto_approve or not request.reuse_verified_implementation:
            raise SpecificationValidationError("frozen runs require UTC_24H_SESSION, --auto-approve, and --reuse-verified-implementation")
        root = Path(request.repository_root).resolve()
        importer = AggregateTradeImporter(root / "data" / "value_area_trap")
        manifest_path = Path(request.data_manifest).resolve()
        manifest = importer.validate_monthly_manifest(manifest_path)
        if manifest.date_start > PRIMARY_START.date() or manifest.date_end < APRIL_END.date():
            raise SpecificationValidationError(
                "frozen UTC evaluation requires a pinned manifest covering 2025-01-01 through 2026-04-30"
            )
        spec = _frozen_spec(manifest, manifest_path, root)
        validate_frozen_specification(spec, manifest)
        run_id = f"frozen-{spec.strategy_id}-{_hash({'specification_hash': spec.specification_hash, 'dataset_hash': manifest.normalized_dataset_hash})[:16]}"
        run_root = Path(request.artifact_root).resolve() / spec.strategy_id / run_id
        report_path = run_root / "report" / "frozen-final-report.json"
        comparison_path = run_root / "report" / "frozen-comparison-report.json"
        if report_path.is_file() and comparison_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("specification_hash") != spec.specification_hash or report.get("dataset_hash") != manifest.normalized_dataset_hash:
                raise ValueError("existing frozen run root belongs to different immutable inputs")
            return FrozenRunResult(run_id=run_id, strategy_id=spec.strategy_id, strategy_version=spec.version, specification_hash=spec.specification_hash, dataset_hash=manifest.normalized_dataset_hash, status=report["status"], report_path=str(report_path), comparison_path=str(comparison_path), reused_verified_implementation=True)
        registry = Registry(Database(request.registry_path))
        controller = PipelineController(registry)
        spec_path = run_root / "specification" / "specification.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        if not spec_path.exists():
            save_strategy_spec(spec, str(spec_path))
        try:
            existing = registry.get_strategy(spec.strategy_id, spec.version)
            if existing["specification_hash"] != spec.specification_hash:
                raise ValueError("frozen strategy version already has a different specification hash")
        except RegistryError:
            controller.register_strategy(spec, str(spec_path))
            controller.submit_specification(spec.strategy_id)
            controller.approve_specification(spec.strategy_id)
        adapter = ValueAreaTrapAdapter(spec, root, manifest_path=manifest_path)
        health = adapter.health(spec)
        if not health.healthy or adapter.identity.adapter_version != "value-area-trap-3":
            raise SpecificationValidationError("verified frozen ValueAreaTrap adapter is not compatible")
        controller.transition(spec.strategy_id, PipelineState.IMPLEMENTATION_VERIFICATION, "reuse verified packaged ValueAreaTrap UTC adapter")
        _write_once(run_root / "implementation" / "verification.json", {"status": "VERIFIED_PACKAGED_IMPLEMENTATION", "adapter": adapter.identity.model_dump(mode="json"), "external_executor_required": False})
        controller.consume_budget(spec.strategy_id, backtests=3)
        full_artifact = adapter._run(spec, _split(manifest.normalized_dataset_hash, PRIMARY_START, APRIL_END), "frozen_full_period", spec.baseline_parameters, run_root / "research" / "full_period", f"{run_id}-full")
        verification = VerificationService(request.registry_path).run(spec.strategy_id, full_artifact.diagnostic_manifest_path or "")
        if verification.get("outcome") != "VERIFIED":
            raise SpecificationValidationError(f"frozen technical verification failed: {verification.get('outcome')}")
        primary_artifact = adapter._run(spec, _split(manifest.normalized_dataset_hash, PRIMARY_START, PRIMARY_END), "frozen_primary_holdout", spec.baseline_parameters, run_root / "research" / "primary_holdout", f"{run_id}-primary")
        april_artifact = adapter._run(spec, _split(manifest.normalized_dataset_hash, APRIL_START, APRIL_END), "frozen_previously_observed_april", spec.baseline_parameters, run_root / "research" / "previously_observed_selection_month", f"{run_id}-april")
        controller.transition(spec.strategy_id, PipelineState.EDGE_GATE, "frozen fixed-window reporting complete")
        controller.transition(spec.strategy_id, PipelineState.INSUFFICIENT_EVIDENCE, "frozen strategy has no permitted parameter selection phase")
        comparison = {
            "evidence_label": EVIDENCE_LABEL,
            "run_id": run_id,
            "strategy_id": spec.strategy_id,
            "strategy_version": spec.version,
            "specification_hash": spec.specification_hash,
            "dataset_hash": manifest.normalized_dataset_hash,
            "primary_holdout": _period_summary(primary_artifact.model_dump(mode="json"), Path(primary_artifact.experiment_dir)),
            "previously_observed_selection_month": _period_summary(april_artifact.model_dump(mode="json"), Path(april_artifact.experiment_dir)),
            "full_period_summary": _period_summary(full_artifact.model_dump(mode="json"), Path(full_artifact.experiment_dir)),
        }
        _write_once(comparison_path, comparison)
        report = {"status": "FROZEN_EVALUATION_COMPLETE_NO_SELECTION", "evidence_label": EVIDENCE_LABEL, "run_id": run_id, "strategy_id": spec.strategy_id, "strategy_version": spec.version, "specification_hash": spec.specification_hash, "dataset_hash": manifest.normalized_dataset_hash, "implementation": "VERIFIED_PACKAGED_IMPLEMENTATION", "technical_verification": verification, "comparison_report": str(comparison_path), "pipeline_state": PipelineState.INSUFFICIENT_EVIDENCE.value}
        _write_once(report_path, report)
        return FrozenRunResult(run_id=run_id, strategy_id=spec.strategy_id, strategy_version=spec.version, specification_hash=spec.specification_hash, dataset_hash=manifest.normalized_dataset_hash, status=report["status"], report_path=str(report_path), comparison_path=str(comparison_path), reused_verified_implementation=True)
