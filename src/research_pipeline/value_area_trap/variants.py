"""Predeclared, immutable ValueAreaTrap research variants.

This module intentionally has no runner.  It materializes auditable strategy
specifications and parameter manifests; executing any of them remains an
explicit Phase F1/F2 operation after the normal approval boundary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from ..schemas.strategy_spec import (
    ParameterFamily,
    StrategySpec,
    StrictModel,
    calculate_specification_hash,
    save_strategy_spec,
)
from .data import AggregateTradeManifest


REAL_DATASET_HASH = "908a22b85825a2c58cdf60d748500d403c16e57b52648a2376290547088f2b10"
REAL_MANIFEST_PATH = (
    "data/value_area_trap/normalized/BTCUSDT/"
    f"{REAL_DATASET_HASH}/manifest.json"
)
VARIANT_CREATED_AT = datetime(2026, 8, 1, tzinfo=timezone.utc)
COMPARISON_METRICS = [
    "session_count",
    "five_minute_bar_count",
    "significant_stop_runs",
    "volume_qualified_stop_runs",
    "confirmed_divergences",
    "return_triggers",
    "proposed_setups",
    "executed_trades",
    "blocked_trades",
    "wins",
    "losses",
    "gross_pnl",
    "net_pnl",
    "total_costs",
    "average_trade",
    "profit_factor",
    "maximum_drawdown",
    "alpha_mll_distance",
    "alpha_mll_breaches",
]


class ValueAreaTrapVariant(StrictModel):
    variant_id: str
    strategy_id: str
    version: str
    session_definition: str
    session_timezone: str
    swing_right_bars: int = Field(ge=1)
    breakout_volume_multiplier: str
    inherited_from: str | None = None


class VariantParameterManifest(StrictModel):
    schema_version: int = 1
    variant_id: str
    strategy_id: str
    strategy_version: str
    specification_hash: str
    planned_run_id: str
    artifact_root: str
    execution_mode: str = "REAL_DATA"
    provider: str = "Binance USD-M Futures"
    dataset_hash: str = REAL_DATASET_HASH
    manifest_path: str
    evidence_label: str = "Binance BTCUSDT proxy evidence; not CME or Alpha Futures performance."
    parameter_manifest: dict[str, Any]
    inherited_from: str | None = None
    manifest_hash: str

    def canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash", None)
        return payload


PREDECLARED_VARIANTS: tuple[ValueAreaTrapVariant, ...] = (
    ValueAreaTrapVariant(
        variant_id="UTC_24H_SESSION",
        strategy_id="ValueAreaTrap.UTC_24H_SESSION",
        version="utc-24h-session-v1",
        session_definition="UTC_24H_SESSION",
        session_timezone="UTC",
        swing_right_bars=2,
        breakout_volume_multiplier="1.5",
    ),
    ValueAreaTrapVariant(
        variant_id="UTC_24H_FAST_SWING",
        strategy_id="ValueAreaTrap.UTC_24H_FAST_SWING",
        version="utc-24h-fast-swing-v1",
        session_definition="UTC_24H_SESSION",
        session_timezone="UTC",
        swing_right_bars=1,
        breakout_volume_multiplier="1.5",
        inherited_from="UTC_24H_SESSION",
    ),
    ValueAreaTrapVariant(
        variant_id="UTC_24H_FAST_SWING_VOLUME_125",
        strategy_id="ValueAreaTrap.UTC_24H_FAST_SWING_VOLUME_125",
        version="utc-24h-fast-swing-volume-125-v1",
        session_definition="UTC_24H_SESSION",
        session_timezone="UTC",
        swing_right_bars=1,
        breakout_volume_multiplier="1.25",
        inherited_from="UTC_24H_FAST_SWING",
    ),
)


def _immutable_family(name: str, value: Any, order: int, description: str) -> ParameterFamily:
    return ParameterFamily(
        name=name,
        description=description,
        baseline_value=value,
        value_type=type(value).__name__,
        allowed_values=[value],
        optimization_order=order,
        maximum_rounds=0,
        mutable=False,
        hypothesis_relevance="Predeclared immutable variant definition; it is not a research freedom.",
    )


def build_variant_specification(
    variant: ValueAreaTrapVariant,
    *,
    manifest_path: str = REAL_MANIFEST_PATH,
) -> StrategySpec:
    """Build one complete non-optimizing canonical Phase A specification."""

    parameters: dict[str, Any] = {
        "aggregate_trade_manifest_path": manifest_path,
        "dataset_hash": REAL_DATASET_HASH,
        "symbol": "BTCUSDT",
        "session_definition": variant.session_definition,
        "session_timezone": variant.session_timezone,
        "value_area_fraction": "0.70",
        "price_bucket_size": "10",
        "breakout_volume_multiplier": variant.breakout_volume_multiplier,
        "breakout_volume_lookback_bars": 10,
        "minimum_breakout_buckets": 1,
        "swing_left_bars": 2,
        "swing_right_bars": variant.swing_right_bars,
        "stop_buffer_buckets": 1,
        "maximum_trades_per_day": 1,
        "entry_execution": "next_bar_open",
        "quantity": "0.001",
        "minimum_quantity": "0.001",
        "quantity_step": "0.001",
        "price_tick": "0.10",
        "same_bar_stop_target_policy": "stop_first",
        "variant": "FULL",
        "optimization_allowed": False,
    }
    payload: dict[str, Any] = {
        "strategy_id": variant.strategy_id,
        "version": variant.version,
        "name": f"ValueAreaTrap {variant.variant_id}",
        "description": (
            f"Predeclared immutable {variant.variant_id} session variant of the "
            "ValueAreaTrap reference strategy. This is Binance BTCUSDT proxy "
            "evidence only, not CME or Alpha Futures performance."
        ),
        "hypothesis": (
            "Changing only the declared session boundary and explicitly named "
            "fixed parameters may be evaluated as separate, non-optimized "
            "research variants using the same immutable aggregate-trade dataset."
        ),
        "strategy_family": "value_area_trap_reference",
        "markets": ["BTCUSDT"],
        "timeframes": ["5m"],
        "long_rules": [
            "Use only the previous fully completed UTC calendar-day volume profile.",
            "Require the existing completed-bar stop-run, divergence, and return-to-value-area conditions.",
        ],
        "short_rules": [
            "Use only the previous fully completed UTC calendar-day volume profile.",
            "Require the existing completed-bar stop-run, divergence, and return-to-value-area conditions.",
        ],
        "entry_logic": "Enter no earlier than the next completed 5-minute bar open after a confirmed return trigger, with the existing adverse slippage model.",
        "initial_stop_logic": "Preserve the existing stop beyond the recorded stop-run extreme plus one profile bucket.",
        "exit_logic": "Preserve the existing POC target, session force-flat behavior, fees, slippage, sizing, and Alpha scenario rules.",
        "session_assumptions": [
            "UTC_24H_SESSION means 00:00:00 through 23:59:59.999999 UTC, represented internally as an end-exclusive next-midnight boundary.",
            "The current UTC day never contributes to its own value-area profile.",
            "All signals use completed 5-minute bars only.",
        ],
        "baseline_parameters": parameters,
        "parameter_families": [
            _immutable_family("session_definition", variant.session_definition, 1, "UTC calendar-day session boundary."),
            _immutable_family("swing_right_bars", variant.swing_right_bars, 2, "Right-side confirmation count before a swing is available."),
            _immutable_family("breakout_volume_multiplier", variant.breakout_volume_multiplier, 3, "Fixed breakout-volume multiplier."),
        ],
        "invariants": [
            "No grid search, optimization, or automatic parameter selection.",
            "One executed trade at most per UTC session/day.",
            "A divergence becomes available only after all configured right confirmation bars have closed.",
            "Never backdate a signal to the swing bar; an entry is no earlier than the next bar open.",
            "The volume median uses exactly the previous 10 completed 5-minute bars and excludes the breakout bar.",
            "Use only the immutable REAL_DATA dataset hash 908a22b85825a2c58cdf60d748500d403c16e57b52648a2376290547088f2b10.",
            "Do not modify the completed baseline run f1-ValueAreaTrap-aa3ec3068f16 or its artifacts.",
        ],
        "required_data": [
            "Binance USD-M Futures BTCUSDT aggregate-trade manifest at the declared content-addressed path.",
            "Normalized aggregate-trade Parquet matching the declared dataset hash.",
        ],
        "known_limitations": [
            "Binance BTCUSDT perpetual aggregate trades are proxy evidence, not CME market data.",
            "No result establishes Alpha Futures performance, eligibility, or compliance.",
            "This predeclared variant is not an optimized parameter selection.",
        ],
        "status": "DRAFT",
        "created_at": VARIANT_CREATED_AT,
        "approved_at": None,
        "specification_hash": "pending",
    }
    candidate = StrategySpec.model_validate(
        payload,
        context={"skip_specification_hash_validation": True},
    )
    return candidate.model_copy(
        update={"specification_hash": calculate_specification_hash(candidate)}
    )


def build_parameter_manifest(
    variant: ValueAreaTrapVariant,
    specification: StrategySpec,
    artifact_root: Path,
    manifest_path: Path,
) -> VariantParameterManifest:
    planned_run_id = f"f1-{variant.strategy_id}-{specification.specification_hash[:12]}"
    payload: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "strategy_id": variant.strategy_id,
        "strategy_version": variant.version,
        "specification_hash": specification.specification_hash,
        "planned_run_id": planned_run_id,
        "artifact_root": str((artifact_root / variant.strategy_id / planned_run_id).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "parameter_manifest": specification.baseline_parameters,
        "inherited_from": variant.inherited_from,
        "manifest_hash": "pending",
    }
    unsigned = VariantParameterManifest.model_validate(payload, context={"skip": True})
    digest = hashlib.sha256(
        json.dumps(unsigned.canonical_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return unsigned.model_copy(update={"manifest_hash": digest})


def _write_immutable(path: Path, content: bytes) -> None:
    """Write once, allowing only an identical idempotent replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable variant artifact already differs: {path}")
        return
    path.write_bytes(content)


def materialize_variants(
    *,
    repository_root: str | Path,
    data_manifest_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Create immutable specs/manifests and an unexecuted comparison template."""

    root = Path(repository_root).resolve()
    manifest_path = Path(data_manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"ValueAreaTrap real-data manifest is missing: {manifest_path}")
    manifest = AggregateTradeManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload.pop("manifest_hash", None)
    expected_manifest_hash = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.manifest_hash != expected_manifest_hash:
        raise ValueError(f"ValueAreaTrap real-data manifest hash mismatch: {manifest_path}")
    if manifest.normalized_dataset_hash != REAL_DATASET_HASH:
        raise ValueError(
            "variant dataset hash mismatch: "
            f"expected {REAL_DATASET_HASH}, got {manifest.normalized_dataset_hash}"
        )
    destination = Path(artifact_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    output: list[dict[str, Any]] = []
    for variant in PREDECLARED_VARIANTS:
        spec = build_variant_specification(
            variant,
            manifest_path=str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
        )
        parameter_manifest = build_parameter_manifest(variant, spec, destination, manifest_path)
        variant_root = Path(parameter_manifest.artifact_root)
        spec_path = variant_root / "specification" / "specification.yaml"
        params_path = variant_root / "run" / "immutable-parameter-manifest.json"
        intake_path = variant_root / "run" / "variant-intake.json"
        import yaml
        _write_immutable(spec_path, yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False).encode("utf-8"))
        _write_immutable(params_path, json.dumps(parameter_manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8"))
        intake = {
            "strategy_name": variant.strategy_id,
            "description": spec.description,
            "markets": spec.markets,
            "timeframes": spec.timeframes,
            "entry_logic": [spec.entry_logic],
            "exit_logic": [spec.exit_logic],
            "confirmed_facts": spec.invariants,
            "assumptions": spec.session_assumptions,
        }
        _write_immutable(intake_path, json.dumps(intake, indent=2, sort_keys=True).encode("utf-8"))
        output.append({
            "variant_id": variant.variant_id,
            "strategy_id": variant.strategy_id,
            "version": variant.version,
            "specification_hash": spec.specification_hash,
            "planned_run_id": parameter_manifest.planned_run_id,
            "artifact_root": str(variant_root),
            "specification_path": str(spec_path),
            "parameter_manifest_path": str(params_path),
            "intake_path": str(intake_path),
        })
    comparison = {
        "schema_version": 1,
        "status": "NOT_EXECUTED",
        "evidence_label": "Binance BTCUSDT proxy evidence; not CME or Alpha Futures performance.",
        "dataset_hash": REAL_DATASET_HASH,
        "metrics": COMPARISON_METRICS,
        "variants": [{**item, "results": {metric: None for metric in COMPARISON_METRICS}} for item in output],
    }
    comparison_path = destination / "value-area-trap-variant-comparison.json"
    _write_immutable(comparison_path, json.dumps(comparison, indent=2, sort_keys=True).encode("utf-8"))
    return {"status": "MATERIALIZED_NOT_EXECUTED", "comparison_path": str(comparison_path), "variants": output}
