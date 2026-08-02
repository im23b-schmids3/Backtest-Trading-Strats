"""Frozen, descriptive-only cross-market robustness evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters.value_area_trap import ValueAreaTrapAdapter
from ..errors import SpecificationValidationError
from ..schemas.strategy_spec import StrategySpec, StrictModel, calculate_specification_hash, save_strategy_spec
from .data import (
    AggregateTradeImporter,
    MonthlyAggregateTradeManifest,
    validate_cross_market_symbol_eligibility,
)
from .frozen import _hash, _period_summary, _split, _write_once
from .variants import PREDECLARED_VARIANTS, build_variant_specification


CROSS_MARKET_SYMBOLS = ("XAUUSDT", "QQQUSDT", "SPYUSDT")
CROSS_START = datetime(2026, 5, 1, tzinfo=timezone.utc)
CROSS_END = datetime(2026, 7, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
CROSS_MONTHS = ("2026-05", "2026-06", "2026-07")
EVIDENCE_LABEL = (
    "Binance synthetic/TradFi perpetual proxy evidence; not ownership of QQQ or SPY ETF shares; "
    "not native COMEX gold; not CME MBT; not Alpha Futures performance."
)


class CrossMarketIngestRequest(StrictModel):
    symbols: list[str] = list(CROSS_MARKET_SYMBOLS)
    cache_root: str
    start_month: str = "2026-05"
    end_month: str = "2026-07"
    metadata_artifact: str | None = None
    allow_network: bool = False
    allow_gap_repair: bool = False


class FrozenCrossMarketRequest(StrictModel):
    manifests: dict[str, str]
    artifact_root: str
    repository_root: str


class FrozenCrossMarketResult(StrictModel):
    run_id: str
    comparison_path: str
    symbol_runs: dict[str, str]
    status: str
    external_executor_required: bool = False
    selection_prohibited: bool = True


def _normal_symbols(symbols: list[str]) -> list[str]:
    result = [item.upper() for item in symbols]
    if tuple(sorted(result)) != tuple(sorted(CROSS_MARKET_SYMBOLS)) or len(result) != len(CROSS_MARKET_SYMBOLS):
        raise SpecificationValidationError(f"cross-market evaluation accepts exactly {list(CROSS_MARKET_SYMBOLS)}")
    return result


def ingest_cross_market(request: CrossMarketIngestRequest) -> dict[str, Any]:
    symbols = _normal_symbols(request.symbols)
    if request.start_month != "2026-05" or request.end_month != "2026-07":
        raise SpecificationValidationError("frozen cross-market evaluation is fixed to complete common months 2026-05 through 2026-07")
    importer = AggregateTradeImporter(request.cache_root)
    metadata_path, artifact = importer.resolve_symbol_metadata(
        symbols, artifact_path=request.metadata_artifact, allow_network=request.allow_network
    )
    metadata = {item.symbol: item for item in artifact.symbols}
    output: dict[str, Any] = {"metadata_artifact": str(metadata_path), "metadata_artifact_hash": artifact.artifact_hash, "symbols": {}}
    for symbol in symbols:
        manifest_path, manifest = importer.ingest_monthly_range(
            symbol=symbol,
            start_month=request.start_month,
            end_month=request.end_month,
            allow_network=request.allow_network,
            allow_gap_repair=request.allow_gap_repair,
            symbol_metadata=metadata[symbol],
            metadata_artifact_path=metadata_path,
            metadata_artifact_hash=artifact.artifact_hash,
        )
        boundaries = importer.validate_complete_calendar_months(
            manifest, start_month=request.start_month, end_month=request.end_month
        )
        output["symbols"][symbol] = {
            "manifest_path": str(manifest_path),
            "dataset_hash": manifest.normalized_dataset_hash,
            "month_boundaries": boundaries,
            "months": importer.last_ingestion_diagnostics,
        }
    return output


def validate_cross_market(manifests: dict[str, str]) -> dict[str, Any]:
    symbols = _normal_symbols(list(manifests))
    importer = AggregateTradeImporter(".")
    output: dict[str, Any] = {"valid": True, "symbols": {}}
    for symbol in symbols:
        manifest_path = Path(manifests[symbol]).resolve()
        manifest = importer.validate_monthly_manifest(manifest_path)
        if manifest.symbol != symbol or manifest.symbol_metadata is None:
            raise ValueError(f"cross-market manifest does not have pinned metadata for {symbol}")
        validate_cross_market_symbol_eligibility(manifest.symbol_metadata)
        boundaries = importer.validate_complete_calendar_months(manifest, start_month="2026-05", end_month="2026-07")
        output["symbols"][symbol] = {
            "manifest_path": str(manifest_path), "dataset_hash": manifest.normalized_dataset_hash,
            "metadata": manifest.symbol_metadata.model_dump(mode="json"), "month_boundaries": boundaries,
        }
    return output


def _cross_spec(
    *, symbol: str, manifest: MonthlyAggregateTradeManifest, manifest_path: Path, root: Path
) -> StrategySpec:
    metadata = manifest.symbol_metadata
    if metadata is None or metadata.symbol != symbol:
        raise SpecificationValidationError(f"cross-market manifest lacks valid perpetual metadata for {symbol}")
    try:
        validate_cross_market_symbol_eligibility(metadata)
    except ValueError as exc:
        raise SpecificationValidationError(str(exc)) from exc
    variant = next(item for item in PREDECLARED_VARIANTS if item.variant_id == "UTC_24H_SESSION")
    configured = str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path)
    base = build_variant_specification(variant, manifest_path=configured)
    payload = base.model_dump(mode="python")
    payload["strategy_id"] = f"ValueAreaTrap.UTC_24H_SESSION.{symbol}"
    payload["version"] = f"utc-24h-session-cross-market-{symbol.lower()}-{manifest.normalized_dataset_hash[:12]}"
    payload["name"] = f"ValueAreaTrap UTC 24H frozen cross-market {symbol}"
    payload["description"] = EVIDENCE_LABEL
    payload["baseline_parameters"].update({
        "symbol": symbol,
        "dataset_hash": manifest.normalized_dataset_hash,
        "price_tick": str(metadata.tick_size),
        "quantity_step": str(metadata.quantity_step),
        "minimum_quantity": str(metadata.minimum_quantity),
        "quantity": str(metadata.minimum_quantity),
        "enforce_symbol_filters": True,
        "symbol_metadata_hash": manifest.metadata_artifact_hash,
        "cross_market_descriptive_only": True,
    })
    payload["invariants"] = [
        *payload["invariants"],
        "Cross-market evidence is descriptive only and must not select, rank, approve, or promote a symbol.",
        "Only exchange metadata (tick size, quantity step, and minimum quantity) may differ by symbol.",
    ]
    payload["specification_hash"] = "pending"
    candidate = StrategySpec.model_validate(payload, context={"skip_specification_hash_validation": True})
    return candidate.model_copy(update={"specification_hash": calculate_specification_hash(candidate)})


def validate_cross_specification(spec: StrategySpec, manifest: MonthlyAggregateTradeManifest) -> None:
    values = spec.baseline_parameters
    expected = {
        "session_definition": "UTC_24H_SESSION", "session_timezone": "UTC", "swing_left_bars": 2,
        "swing_right_bars": 2, "breakout_volume_multiplier": "1.5", "breakout_volume_lookback_bars": 10,
        "value_area_fraction": "0.70", "price_bucket_size": "10", "maximum_trades_per_day": 1,
        "entry_execution": "next_bar_open", "same_bar_stop_target_policy": "stop_first", "optimization_allowed": False,
        "dataset_hash": manifest.normalized_dataset_hash, "enforce_symbol_filters": True,
        "cross_market_descriptive_only": True,
    }
    for key, value in expected.items():
        if str(values.get(key)).lower() != str(value).lower():
            raise SpecificationValidationError(f"cross-market frozen parameter mismatch: {key}")
    if manifest.symbol_metadata is None:
        raise SpecificationValidationError("cross-market manifest has no pinned symbol metadata")
    metadata = manifest.symbol_metadata
    for key, value in {"price_tick": metadata.tick_size, "quantity_step": metadata.quantity_step, "minimum_quantity": metadata.minimum_quantity, "quantity": metadata.minimum_quantity}.items():
        if str(values.get(key)) != str(value):
            raise SpecificationValidationError(f"cross-market exchange filter mismatch: {key}")
    if any(item.mutable or item.maximum_rounds != 0 for item in spec.parameter_families):
        raise SpecificationValidationError("cross-market specification permits parameter research")


def _symbol_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    executed = metrics["executed_trades"]
    monthly = summary["monthly"]
    month_nets = [float(monthly[month]["net_pnl"]) for month in CROSS_MONTHS]
    return {
        "executed_trades": executed, "wins": metrics["wins"], "losses": metrics["losses"],
        "win_rate": metrics["wins"] / executed if executed else 0.0,
        "gross_pnl": metrics["gross_pnl"], "net_pnl": metrics["net_pnl"], "total_costs": metrics.get("total_costs", metrics["fees"] + metrics["slippage"]),
        "profit_factor": metrics.get("profit_factor"), "average_trade": metrics.get("average_trade", 0.0),
        "maximum_drawdown": metrics.get("maximum_drawdown", 0.0), "longest_losing_streak": summary["longest_losing_streak"],
        "positive_months": sum(value > 0 for value in month_nets), "negative_months": sum(value < 0 for value in month_nets),
        "flat_months": sum(value == 0 for value in month_nets), "trade_frequency": summary["trade_frequency_per_session"],
        "significant_stop_runs": metrics["significant_stop_runs"], "confirmed_divergences": metrics["confirmed_divergences"],
        "proposed_setups": metrics["proposed_setups"], "compliance_blocks": metrics["compliance_blocks"], "monthly": {month: monthly[month] for month in CROSS_MONTHS},
        "depends_on_one_exceptional_month": max((abs(value) for value in month_nets), default=0.0) > 0.5 * sum(abs(value) for value in month_nets),
    }


class FrozenCrossMarketService:
    def run(self, request: FrozenCrossMarketRequest) -> FrozenCrossMarketResult:
        validated = validate_cross_market(request.manifests)
        root = Path(request.repository_root).resolve()
        run_identity = _hash({"manifests": {symbol: validated["symbols"][symbol]["dataset_hash"] for symbol in sorted(validated["symbols"])}, "months": CROSS_MONTHS})
        run_id = f"frozen-cross-market-{run_identity[:16]}"
        run_root = Path(request.artifact_root).resolve() / "ValueAreaTrap.UTC_24H_SESSION.cross_market" / run_id
        comparison_path = run_root / "cross-market-comparison.json"
        if comparison_path.exists():
            payload = json.loads(comparison_path.read_text(encoding="utf-8"))
            if payload.get("run_id") != run_id:
                raise ValueError("immutable cross-market comparison collision")
            return FrozenCrossMarketResult(run_id=run_id, comparison_path=str(comparison_path), symbol_runs=payload["symbol_runs"], status=payload["status"])
        summaries: dict[str, Any] = {}
        symbol_runs: dict[str, str] = {}
        frozen_contract: dict[str, dict[str, Any]] = {}
        importer = AggregateTradeImporter(".")
        for symbol in CROSS_MARKET_SYMBOLS:
            manifest_path = Path(request.manifests[symbol]).resolve()
            manifest = importer.validate_monthly_manifest(manifest_path)
            spec = _cross_spec(symbol=symbol, manifest=manifest, manifest_path=manifest_path, root=root)
            validate_cross_specification(spec, manifest)
            symbol_root = run_root / symbol / f"frozen-{symbol}-{_hash({'specification_hash': spec.specification_hash, 'dataset_hash': manifest.normalized_dataset_hash})[:16]}"
            spec_path = symbol_root / "specification.yaml"
            if not spec_path.exists():
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                save_strategy_spec(spec, str(spec_path))
            adapter = ValueAreaTrapAdapter(spec, root, manifest_path=manifest_path)
            if not adapter.health(spec).healthy or adapter.identity.adapter_version != "value-area-trap-3":
                raise SpecificationValidationError(f"verified packaged adapter is incompatible for {symbol}")
            artifact = adapter._run(spec, _split(manifest.normalized_dataset_hash, CROSS_START, CROSS_END, symbol=symbol), "frozen_cross_market", spec.baseline_parameters, symbol_root / "research", f"{run_id}-{symbol}")
            summary = _symbol_summary(_period_summary(artifact.model_dump(mode="json"), Path(artifact.experiment_dir)))
            report = {"status": "FROZEN_CROSS_MARKET_DESCRIPTIVE_COMPLETE", "evidence_label": EVIDENCE_LABEL, "symbol": symbol, "run_id": symbol_root.name, "specification_hash": spec.specification_hash, "dataset_hash": manifest.normalized_dataset_hash, "implementation": "VERIFIED_PACKAGED_IMPLEMENTATION", "external_executor_required": False, "selection_prohibited": True, "summary": summary}
            _write_once(symbol_root / "report.json", report)
            summaries[symbol] = summary; symbol_runs[symbol] = str(symbol_root)
            frozen_contract[symbol] = {key: spec.baseline_parameters[key] for key in ("session_definition", "session_timezone", "swing_left_bars", "swing_right_bars", "breakout_volume_multiplier", "breakout_volume_lookback_bars", "value_area_fraction", "price_bucket_size", "entry_execution", "same_bar_stop_target_policy", "optimization_allowed")}
        if len({json.dumps(item, sort_keys=True, default=str) for item in frozen_contract.values()}) != 1:
            raise SpecificationValidationError("cross-market strategy parameters differ; selection comparison is forbidden")
        nets = [float(summaries[symbol]["net_pnl"]) for symbol in CROSS_MARKET_SYMBOLS]
        pooled = {key: sum(float(summaries[symbol][key]) for symbol in CROSS_MARKET_SYMBOLS) for key in ("executed_trades", "wins", "losses", "gross_pnl", "net_pnl", "total_costs", "significant_stop_runs", "confirmed_divergences", "proposed_setups", "compliance_blocks")}
        comparison = {"status": "FROZEN_CROSS_MARKET_DESCRIPTIVE_COMPLETE", "run_id": run_id, "evidence_label": EVIDENCE_LABEL, "evaluation_months": list(CROSS_MONTHS), "symbol_runs": symbol_runs, "symbols": summaries, "frozen_strategy_parameters": next(iter(frozen_contract.values())), "net_pnl_sign_agrees_across_all_symbols": len({(value > 0) - (value < 0) for value in nets}) == 1, "profit_factor_above_one": {symbol: (summaries[symbol]["profit_factor"] or 0) > 1 for symbol in CROSS_MARKET_SYMBOLS}, "depends_on_one_exceptional_month": {symbol: summaries[symbol]["depends_on_one_exceptional_month"] for symbol in CROSS_MARKET_SYMBOLS}, "pooled_descriptive_totals": pooled, "selection_prohibited": True, "best_symbol": None, "ranking": None, "promotion": None}
        _write_once(comparison_path, comparison)
        return FrozenCrossMarketResult(run_id=run_id, comparison_path=str(comparison_path), symbol_runs=symbol_runs, status=comparison["status"])
