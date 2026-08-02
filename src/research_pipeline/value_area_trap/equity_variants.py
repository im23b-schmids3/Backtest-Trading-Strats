"""Sealed exploratory in-sample study for the QQQUSDT and SPYUSDT proxies.

This is deliberately not a parameter-research engine.  The six variants below
are the entire pre-registered registry and every generated result is labelled
as exploratory, in-sample evidence that requires a future holdout.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import Field

from ..adapters.value_area_trap import ValueAreaTrapAdapter
from ..errors import SpecificationValidationError
from ..schemas.strategy_spec import ParameterFamily, StrategySpec, StrictModel, calculate_specification_hash, save_strategy_spec
from .data import AggregateTradeImporter, MonthlyAggregateTradeManifest
from .frozen import _hash, _split, _write_once
from .profile import US_CASH_SESSION_LABEL


EQUITY_STUDY_SYMBOLS = ("QQQUSDT", "SPYUSDT")
EQUITY_STUDY_MONTHS = ("2026-05", "2026-06", "2026-07")
EQUITY_STUDY_START = datetime(2026, 5, 1, tzinfo=timezone.utc)
EQUITY_STUDY_END = datetime(2026, 7, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
EQUITY_STUDY_LABEL = "EXPLORATORY_IN_SAMPLE_VARIANT_STUDY"
EQUITY_EVIDENCE_LABEL = (
    "EXPLORATORY_IN_SAMPLE_VARIANT_STUDY; Binance synthetic/TradFi perpetual proxy evidence; "
    "not ownership of QQQ or SPY ETF shares; not CME or Alpha Futures performance."
)


class EquityVariant(StrictModel):
    variant_id: str = Field(pattern="^[A-F]$")
    session_definition: str
    session_timezone: str
    session_open: str | None = None
    session_close: str | None = None
    breakout_volume_multiplier: str
    swing_right_bars: int = Field(ge=1)


class EquityVariantStudyRequest(StrictModel):
    manifests: dict[str, str]
    artifact_root: str
    repository_root: str


class EquityVariantStudyResult(StrictModel):
    run_id: str
    comparison_path: str
    status: str
    variant_count: int
    result_count: int
    selection_prohibited: bool = True
    optimization_claimed: bool = False
    confirmation_evidence: bool = False
    requires_future_holdout: bool = True


EQUITY_PRE_REGISTERED_VARIANTS: tuple[EquityVariant, ...] = (
    EquityVariant(variant_id="A", session_definition="UTC_24H_SESSION", session_timezone="UTC", breakout_volume_multiplier="1.50", swing_right_bars=2),
    EquityVariant(variant_id="B", session_definition="UTC_24H_SESSION", session_timezone="UTC", breakout_volume_multiplier="1.25", swing_right_bars=2),
    EquityVariant(variant_id="C", session_definition="UTC_24H_SESSION", session_timezone="UTC", breakout_volume_multiplier="1.00", swing_right_bars=2),
    EquityVariant(variant_id="D", session_definition="UTC_24H_SESSION", session_timezone="UTC", breakout_volume_multiplier="1.50", swing_right_bars=1),
    EquityVariant(variant_id="E", session_definition=US_CASH_SESSION_LABEL, session_timezone="America/New_York", session_open="09:30", session_close="16:00", breakout_volume_multiplier="1.50", swing_right_bars=2),
    EquityVariant(variant_id="F", session_definition=US_CASH_SESSION_LABEL, session_timezone="America/New_York", session_open="09:30", session_close="16:00", breakout_volume_multiplier="1.25", swing_right_bars=1),
)


def _immutable_family(name: str, value: Any, order: int) -> ParameterFamily:
    return ParameterFamily(
        name=name,
        description="Pre-registered exploratory-study value; no optimization is permitted.",
        baseline_value=value,
        value_type=type(value).__name__,
        allowed_values=[value],
        optimization_order=order,
        maximum_rounds=0,
        mutable=False,
        hypothesis_relevance="Fixed before any study result is produced.",
    )


def _require_exact_registry() -> None:
    expected = {
        "A": ("UTC_24H_SESSION", "1.50", 2),
        "B": ("UTC_24H_SESSION", "1.25", 2),
        "C": ("UTC_24H_SESSION", "1.00", 2),
        "D": ("UTC_24H_SESSION", "1.50", 1),
        "E": (US_CASH_SESSION_LABEL, "1.50", 2),
        "F": (US_CASH_SESSION_LABEL, "1.25", 1),
    }
    actual = {
        item.variant_id: (item.session_definition, item.breakout_volume_multiplier, item.swing_right_bars)
        for item in EQUITY_PRE_REGISTERED_VARIANTS
    }
    if actual != expected:
        raise SpecificationValidationError("equity exploratory study registry differs from its six pre-registered variants")


def _normal_manifests(manifests: dict[str, str]) -> dict[str, str]:
    normalized = {symbol.upper(): path for symbol, path in manifests.items()}
    if set(normalized) != set(EQUITY_STUDY_SYMBOLS):
        raise SpecificationValidationError(f"equity variant study accepts exactly {list(EQUITY_STUDY_SYMBOLS)}")
    return normalized


def validate_equity_variant_study(manifests: dict[str, str]) -> dict[str, Any]:
    """Verify existing content-addressed manifests without downloading data."""

    _require_exact_registry()
    normalized = _normal_manifests(manifests)
    importer = AggregateTradeImporter(".")
    output: dict[str, Any] = {"valid": True, "study_label": EQUITY_STUDY_LABEL, "symbols": {}, "variant_registry": [item.model_dump(mode="json") for item in EQUITY_PRE_REGISTERED_VARIANTS]}
    for symbol in EQUITY_STUDY_SYMBOLS:
        manifest_path = Path(normalized[symbol]).resolve()
        manifest = importer.validate_monthly_manifest(manifest_path)
        if manifest.symbol != symbol:
            raise SpecificationValidationError(f"equity study manifest symbol mismatch: expected {symbol}, got {manifest.symbol}")
        boundaries = importer.validate_complete_calendar_months(manifest, start_month="2026-05", end_month="2026-07")
        output["symbols"][symbol] = {
            "manifest_path": str(manifest_path),
            "dataset_hash": manifest.normalized_dataset_hash,
            "row_count": manifest.row_count,
            "partition_count": len(manifest.partitions),
            "month_boundaries": boundaries,
            "reused_immutable_partitions": True,
        }
    return output


def _specification(
    symbol: str,
    variant: EquityVariant,
    manifest: MonthlyAggregateTradeManifest,
    manifest_path: Path,
    repository_root: Path,
) -> StrategySpec:
    metadata = manifest.symbol_metadata
    if metadata is None:
        raise SpecificationValidationError(f"equity study manifest has no pinned exchange metadata for {symbol}")
    configured_manifest = str(manifest_path.relative_to(repository_root)) if manifest_path.is_relative_to(repository_root) else str(manifest_path)
    parameters: dict[str, Any] = {
        "aggregate_trade_manifest_path": configured_manifest,
        "dataset_hash": manifest.normalized_dataset_hash,
        "symbol": symbol,
        "session_definition": variant.session_definition,
        "session_timezone": variant.session_timezone,
        "session_open": variant.session_open,
        "session_close": variant.session_close,
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
        "quantity": str(metadata.minimum_quantity),
        "minimum_quantity": str(metadata.minimum_quantity),
        "quantity_step": str(metadata.quantity_step),
        "price_tick": str(metadata.tick_size),
        "enforce_symbol_filters": True,
        "record_compliance_events": True,
        "same_bar_stop_target_policy": "stop_first",
        "variant": "FULL",
        "optimization_allowed": False,
        "study_label": EQUITY_STUDY_LABEL,
        "selection_prohibited": True,
        "optimization_claimed": False,
        "confirmation_evidence": False,
        "requires_future_holdout": True,
    }
    if variant.session_definition == US_CASH_SESSION_LABEL:
        parameters["us_cash_holiday_calendar"] = "NYSE_FULL_DAY_HOLIDAYS_2026_MAY_JUL_PINNED"
    payload: dict[str, Any] = {
        "strategy_id": f"ValueAreaTrap.EquityExploratory.{symbol}.{variant.variant_id}",
        "version": f"equity-exploratory-2026-may-jul-{symbol.lower()}-{variant.variant_id.lower()}-{manifest.normalized_dataset_hash[:12]}",
        "name": f"ValueAreaTrap {symbol} exploratory variant {variant.variant_id}",
        "description": EQUITY_EVIDENCE_LABEL,
        "hypothesis": "This pre-registered in-sample comparison describes six fixed implementation variants only; it cannot select or confirm a strategy.",
        "strategy_family": "value_area_trap_reference",
        "markets": [symbol],
        "timeframes": ["5m"],
        "long_rules": ["Use only the previous fully completed session profile and confirmed completed-bar CVD divergence."],
        "short_rules": ["Use only the previous fully completed session profile and confirmed completed-bar CVD divergence."],
        "entry_logic": "Enter no earlier than the next 5-minute bar open after the completed return trigger, with existing adverse slippage.",
        "initial_stop_logic": "Preserve the existing stop beyond the stop-run extreme plus one profile bucket.",
        "exit_logic": "Preserve the existing POC target, session force-flat behavior, fees, slippage, sizing, and symbol quantization.",
        "session_assumptions": [
            "UTC variants use completed UTC calendar-day sessions.",
            "US_CASH_SESSION uses America/New_York 09:30-16:00 boundaries via timezone rules, never a fixed UTC offset.",
            "US_CASH_SESSION excludes weekends and the pinned full-day NYSE holidays Memorial Day, Juneteenth, and Independence Day observed in May-July 2026.",
            "Only complete US cash sessions with both the 09:30 and 16:00 five-minute boundaries are included.",
        ],
        "baseline_parameters": parameters,
        "parameter_families": [
            _immutable_family("session_definition", variant.session_definition, 1),
            _immutable_family("breakout_volume_multiplier", variant.breakout_volume_multiplier, 2),
            _immutable_family("swing_right_bars", variant.swing_right_bars, 3),
        ],
        "invariants": [
            "EXPLORATORY_IN_SAMPLE_VARIANT_STUDY only; it is not confirmation evidence.",
            "Exactly six pre-registered variants A through F may be reported; no variants may be added, removed, or tuned after results are seen.",
            "No grid search, automatic parameter selection, ranking, promotion, or best-variant recommendation.",
            "The value area uses the previous completed session profile at 70 percent.",
            "The volume median uses exactly the previous 10 completed five-minute bars and excludes the breakout bar.",
            "Divergences are confirmed only after configured right bars close; entries are no earlier than the next bar open.",
            "At most one trade is executed per session/day.",
            "Do not modify completed BTC frozen or cross-market robustness artifacts.",
        ],
        "required_data": [f"Pinned immutable Binance USD-M {symbol} aggregate-trade manifest: {configured_manifest}"],
        "known_limitations": [
            "Binance TradFi perpetual data is proxy evidence and is not ownership of QQQ or SPY ETF shares.",
            "The study is in-sample exploratory evidence and requires future untouched holdout evidence.",
            "No result is Alpha Futures performance, approval, or compliance evidence.",
        ],
        "status": "DRAFT",
        "created_at": datetime(2026, 8, 2, tzinfo=timezone.utc),
        "approved_at": None,
        "specification_hash": "pending",
    }
    candidate = StrategySpec.model_validate(payload, context={"skip_specification_hash_validation": True})
    return candidate.model_copy(update={"specification_hash": calculate_specification_hash(candidate)})


def _monthly_summary(root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    trades = json.loads((root / "trades.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(root / "strategy_events.parquet")
    bars = pd.read_parquet(root / "5m_bars.parquet")
    monthly: dict[str, Any] = {}
    for month in EQUITY_STUDY_MONTHS:
        month_trades = [item for item in trades if item["entry_timestamp"][:7] == month]
        month_events = events[events.get("session_date", pd.Series(dtype=str)).astype(str).str.startswith(month)] if not events.empty else events
        values = [float(item["net_pnl"]) for item in month_trades]
        gains, losses = sum(value for value in values if value > 0), -sum(value for value in values if value < 0)
        running = peak = drawdown = 0.0
        for value in values:
            running += value; peak = max(peak, running); drawdown = max(drawdown, peak - running)
        state_count = lambda state: int(sum(month_events.get("state", pd.Series(dtype=str)) == state)) if not month_events.empty else 0
        monthly[month] = {
            "session_count": int(bars.loc[bars.get("session_date", pd.Series(dtype=str)).astype(str).str.startswith(month), "session_date"].astype(str).nunique()) if not bars.empty else 0,
            "five_minute_bar_count": int(bars.get("session_date", pd.Series(dtype=str)).astype(str).str.startswith(month).sum()),
            "significant_stop_runs": state_count("STOP_RUN_CONFIRMED"),
            "volume_qualified_stop_runs": state_count("STOP_RUN_CONFIRMED"),
            "confirmed_divergences": state_count("DIVERGENCE_CONFIRMED"),
            "return_triggers": state_count("RETURN_TRIGGER"),
            "proposed_setups": state_count("PROPOSED_SETUP"),
            "invalid_setups": state_count("INVALIDATED"),
            "non_executable_setups": state_count("NO_EXECUTABLE_ENTRY"),
            "compliance_blocks": state_count("COMPLIANCE_BLOCKED"),
            "executed_trades": len(month_trades),
            "wins": sum(value > 0 for value in values),
            "losses": sum(value <= 0 for value in values),
            "gross_pnl": sum(float(item["gross_pnl"]) for item in month_trades),
            "net_pnl": sum(values),
            "total_costs": sum(float(item["fees"]) + float(item["slippage_cost"]) for item in month_trades),
            "profit_factor": gains / losses if losses else None,
            "average_trade": sum(values) / len(values) if values else 0.0,
            "maximum_drawdown": drawdown,
        }
    event_count = lambda state: int(sum(events.get("state", pd.Series(dtype=str)) == state)) if not events.empty else 0
    net_values = [float(item["net_pnl"]) for item in trades]
    current_loss = longest_loss = 0
    for value in net_values:
        current_loss = current_loss + 1 if value < 0 else 0
        longest_loss = max(longest_loss, current_loss)
    invalid = event_count("INVALIDATED")
    non_executable = event_count("NO_EXECUTABLE_ENTRY")
    blocks = event_count("COMPLIANCE_BLOCKED")
    executed = len(trades)
    proposed = int(metrics["proposed_setups"])
    funnel_components = invalid + non_executable + blocks + executed
    return {
        "session_count": int(metrics["session_count"]),
        "five_minute_bar_count": int(metrics["five_minute_bar_count"]),
        "significant_stop_runs": int(metrics["significant_stop_runs"]),
        "volume_qualified_stop_runs": int(metrics["volume_qualified_stop_runs"]),
        "confirmed_divergences": int(metrics["confirmed_divergences"]),
        "return_triggers": int(metrics["return_triggers"]),
        "proposed_setups": proposed,
        "invalid_setups": invalid,
        "non_executable_setups": non_executable,
        "compliance_blocks": blocks,
        "executed_trades": executed,
        "wins": int(metrics["wins"]),
        "losses": int(metrics["losses"]),
        "win_rate": float(metrics["wins"]) / executed if executed else 0.0,
        "gross_pnl": metrics["gross_pnl"],
        "net_pnl": metrics["net_pnl"],
        "total_costs": metrics["total_costs"],
        "profit_factor": metrics["profit_factor"],
        "average_trade": metrics["average_trade"],
        "maximum_drawdown": metrics["maximum_drawdown"],
        "longest_losing_streak": longest_loss,
        "monthly": monthly,
        "zero_trade_reason": metrics.get("zero_trade_reason"),
        "funnel_reconciliation": {
            "formula": "proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades",
            "proposed_setups": proposed,
            "components_total": funnel_components,
            "reconciles": proposed == funnel_components,
        },
        "us_cash_session_diagnostics": metrics.get("us_cash_session_diagnostics"),
    }


class EquityVariantStudyService:
    def run(self, request: EquityVariantStudyRequest) -> EquityVariantStudyResult:
        validated = validate_equity_variant_study(request.manifests)
        root = Path(request.repository_root).resolve()
        identity = _hash({
            "study_label": EQUITY_STUDY_LABEL,
            "variants": [item.model_dump(mode="json") for item in EQUITY_PRE_REGISTERED_VARIANTS],
            "datasets": {symbol: validated["symbols"][symbol]["dataset_hash"] for symbol in EQUITY_STUDY_SYMBOLS},
        })
        run_id = f"equity-exploratory-{identity[:16]}"
        run_root = Path(request.artifact_root).resolve() / "ValueAreaTrap.EquityExploratory" / run_id
        comparison_path = run_root / "equity-variant-study-comparison.json"
        if comparison_path.exists():
            existing = json.loads(comparison_path.read_text(encoding="utf-8"))
            if existing.get("run_id") != run_id or existing.get("pre_registered_variant_registry") != [item.model_dump(mode="json") for item in EQUITY_PRE_REGISTERED_VARIANTS]:
                raise ValueError("immutable equity variant-study artifact collision")
            return EquityVariantStudyResult(run_id=run_id, comparison_path=str(comparison_path), status=existing["status"], variant_count=6, result_count=12)
        importer = AggregateTradeImporter(".")
        results: list[dict[str, Any]] = []
        for symbol in EQUITY_STUDY_SYMBOLS:
            manifest_path = Path(request.manifests[symbol]).resolve()
            manifest = importer.validate_monthly_manifest(manifest_path)
            for variant in EQUITY_PRE_REGISTERED_VARIANTS:
                spec = _specification(symbol, variant, manifest, manifest_path, root)
                run_identity = _hash({"symbol": symbol, "variant_id": variant.variant_id, "specification_hash": spec.specification_hash, "dataset_hash": manifest.normalized_dataset_hash})
                variant_run_id = f"equity-{symbol.lower()}-{variant.variant_id.lower()}-{run_identity[:16]}"
                variant_root = run_root / symbol / variant.variant_id / variant_run_id
                spec_path = variant_root / "specification.yaml"
                if not spec_path.exists():
                    spec_path.parent.mkdir(parents=True, exist_ok=True)
                    save_strategy_spec(spec, str(spec_path))
                else:
                    saved = StrategySpec.model_validate(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
                    if saved.specification_hash != spec.specification_hash:
                        raise ValueError("immutable equity study specification collision")
                parameter_manifest = {
                    "study_label": EQUITY_STUDY_LABEL,
                    "symbol": symbol,
                    "variant": variant.model_dump(mode="json"),
                    "specification_hash": spec.specification_hash,
                    "dataset_hash": manifest.normalized_dataset_hash,
                    "manifest_path": str(manifest_path),
                    "parameters": spec.baseline_parameters,
                    "selection_prohibited": True,
                    "optimization_claimed": False,
                    "confirmation_evidence": False,
                    "requires_future_holdout": True,
                }
                parameter_manifest["manifest_hash"] = _hash(parameter_manifest)
                _write_once(variant_root / "immutable-parameter-manifest.json", parameter_manifest)
                adapter = ValueAreaTrapAdapter(spec, root, manifest_path=manifest_path)
                if not adapter.health(spec).healthy or adapter.identity.adapter_version != "value-area-trap-3":
                    raise SpecificationValidationError(f"verified ValueAreaTrap adapter is incompatible for {symbol} variant {variant.variant_id}")
                artifact = adapter._run(
                    spec,
                    _split(manifest.normalized_dataset_hash, EQUITY_STUDY_START, EQUITY_STUDY_END, symbol=symbol),
                    "exploratory_in_sample_variant_study",
                    spec.baseline_parameters,
                    variant_root / "research",
                    variant_run_id,
                )
                summary = _monthly_summary(Path(artifact.experiment_dir), artifact.metrics)
                report = {
                    "status": EQUITY_STUDY_LABEL,
                    "evidence_label": EQUITY_EVIDENCE_LABEL,
                    "run_id": variant_run_id,
                    "symbol": symbol,
                    "variant_id": variant.variant_id,
                    "specification_hash": spec.specification_hash,
                    "dataset_hash": manifest.normalized_dataset_hash,
                    "selection_prohibited": True,
                    "optimization_claimed": False,
                    "confirmation_evidence": False,
                    "requires_future_holdout": True,
                    "summary": summary,
                }
                _write_once(variant_root / "report.json", report)
                results.append({
                    "symbol": symbol,
                    "variant_id": variant.variant_id,
                    "run_id": variant_run_id,
                    "artifact_root": str(variant_root),
                    "specification_path": str(spec_path),
                    "parameter_manifest_path": str(variant_root / "immutable-parameter-manifest.json"),
                    "report_path": str(variant_root / "report.json"),
                    "summary": summary,
                })
        comparison = {
            "status": f"{EQUITY_STUDY_LABEL}_COMPLETE",
            "study_label": EQUITY_STUDY_LABEL,
            "evidence_label": EQUITY_EVIDENCE_LABEL,
            "run_id": run_id,
            "evaluation_months": list(EQUITY_STUDY_MONTHS),
            "pre_registered_variant_registry": [item.model_dump(mode="json") for item in EQUITY_PRE_REGISTERED_VARIANTS],
            "results": results,
            "selection_prohibited": True,
            "optimization_claimed": False,
            "confirmation_evidence": False,
            "requires_future_holdout": True,
            "best_variant": None,
            "recommendation": None,
            "promotion": None,
        }
        _write_once(comparison_path, comparison)
        return EquityVariantStudyResult(run_id=run_id, comparison_path=str(comparison_path), status=comparison["status"], variant_count=6, result_count=len(results))
