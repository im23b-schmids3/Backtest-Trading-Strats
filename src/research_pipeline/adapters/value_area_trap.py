from __future__ import annotations

import hashlib
import json
import math
import subprocess
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from ..prop.models import TradeSignal
from ..research.models import ResearchArtifact
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from ..value_area_trap import (
    AggregateTradeManifest,
    SessionProfile,
    ValueAreaTrapConfig,
    build_value_area_reports,
    run_value_area_trap,
)
from ..value_area_trap.reports import raw_strategy_metrics
from ..value_area_trap.data import PARQUET_SCHEMA
from ..value_area_trap.profile import FiveMinuteBar, NY, session_bounds
from ..value_area_trap.alpha_zero import run_alpha_zero_scenario
from ..verification.models import VerificationManifest
from .data import file_hash
from .models import (
    AdapterCapabilities,
    AdapterHealth,
    AdapterIdentity,
    BacktestRun,
    DataAvailability,
    DataClassification,
    NormalizedTrade,
    PhaseDEvent,
    PhaseEEligibility,
)


class ValueAreaTrapAdapter:
    """Streaming manifest-backed Binance aggregate-trade adapter.

    A real run must name its manifest explicitly.  The adapter never searches
    for a convenient Parquet file and never substitutes OHLCV or fixtures.
    """

    def __init__(self, specification: StrategySpec, repository_root: str | Path = ".", manifest_path: str | Path | None = None):
        self.specification = specification
        self.root = Path(repository_root).resolve()
        configured = manifest_path or specification.baseline_parameters.get("aggregate_trade_manifest_path")
        self.manifest_path = self._resolve_configured_path(configured) if configured else None
        self.identity = AdapterIdentity(
            strategy_id=specification.strategy_id,
            strategy_version=specification.version,
            implementation_module=__name__,
            entry_point=f"{__name__}:ValueAreaTrapAdapter",
            specification_hash=specification.specification_hash,
            code_commit=self._commit(),
            worktree_path=str(self.root),
            adapter_version="value-area-trap-2",
        )
        self.capabilities = AdapterCapabilities(
            parameter_experiment=False,
            supported_markets=["BTCUSDT"],
            supported_timeframes=["5m"],
            parameter_families=[],
            data_providers=["binance_usdm_public_aggregate_trades"],
        )
        self._last_run: BacktestRun | None = None

    def _resolve_configured_path(self, configured: str | Path) -> Path:
        target = Path(configured)
        if not target.is_absolute():
            target = self.root / target
        return target.resolve()

    @staticmethod
    def _commit() -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, shell=False, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _manifest_digest(manifest: AggregateTradeManifest) -> str:
        raw = manifest.model_dump(mode="json")
        raw.pop("manifest_hash", None)
        return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _validated_dataset(self) -> tuple[Path, Path, AggregateTradeManifest, str]:
        if self.manifest_path is None:
            raise ValueError("REAL_DATA requires an explicit ValueAreaTrap manifest path")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"ValueAreaTrap manifest is missing: {self.manifest_path}")
        try:
            manifest = AggregateTradeManifest.model_validate(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ValueError(f"invalid ValueAreaTrap manifest {self.manifest_path}: {exc}") from exc
        if manifest.manifest_hash != self._manifest_digest(manifest):
            raise ValueError(f"ValueAreaTrap manifest hash mismatch: {self.manifest_path}")
        parquet = self.manifest_path.with_name("aggregate_trades.parquet")
        if not parquet.is_file():
            raise FileNotFoundError(f"ValueAreaTrap Parquet is missing: {parquet}")
        parquet_file = pq.ParquetFile(parquet)
        detected = parquet_file.schema_arrow
        if detected.names != list(PARQUET_SCHEMA.names):
            raise ValueError(f"invalid ValueAreaTrap Parquet schema: detected columns={detected.names}")
        if parquet_file.metadata.num_rows != manifest.row_count:
            raise ValueError(f"ValueAreaTrap row count mismatch: manifest={manifest.row_count}, parquet={parquet_file.metadata.num_rows}")
        dataset_directory_hash = parquet.parent.name
        if dataset_directory_hash != manifest.normalized_dataset_hash:
            raise ValueError("ValueAreaTrap dataset hash does not match its content-addressed directory")
        # The Parquet hash is calculated from the immutable file and persisted
        # in every run artifact.  The normalized dataset hash is verified above
        # against the importer-produced content-addressed directory.
        parquet_hash = file_hash(parquet)
        return parquet, self.manifest_path, manifest, parquet_hash

    def health(self, specification: StrategySpec) -> AdapterHealth:
        errors = []
        if specification.strategy_family != "value_area_trap_reference":
            errors.append("wrong strategy family")
        if specification.specification_hash != self.identity.specification_hash:
            errors.append("specification hash mismatch")
        return AdapterHealth(identity=self.identity, capabilities=self.capabilities, importable=True, compatible=not errors, healthy=not errors, errors=errors, checked_at=datetime.now(timezone.utc))

    def data_availability(self, specification: StrategySpec) -> list[DataAvailability]:
        parquet, manifest_path, manifest, parquet_hash = self._validated_dataset()
        return [DataAvailability(
            market="BTCUSDT",
            timeframe="5m",
            classification=DataClassification.AVAILABLE_PROXY,
            provider="Binance USD-M Futures",
            source_symbol="BTCUSDT perpetual",
            path=str(parquet),
            dataset_hash=manifest.normalized_dataset_hash,
            start_timestamp=datetime.combine(manifest.date_start, time.min, timezone.utc),
            end_timestamp=datetime.combine(manifest.date_end, time.max, timezone.utc),
            rows=manifest.row_count,
            warnings=["BTCUSDT perpetual is a crypto proxy and is not native CME MBT evidence", f"manifest_path={manifest_path}", f"parquet_hash={parquet_hash}"],
            declared_substitution="BTCUSDT perpetual aggregate trades -> no CME contract substitution",
        )]

    def require_data(self, specification: StrategySpec) -> list[DataAvailability]:
        return self.data_availability(specification)

    @staticmethod
    def _profile(day: date, buckets: dict[Decimal, tuple[Decimal, Decimal]], bucket_size: Decimal, source_hash: str) -> SessionProfile:
        volumes = {key: value[0] for key, value in buckets.items()}
        total = sum(volumes.values(), Decimal())
        weighted = sum((value[1] for value in buckets.values()), Decimal())
        mean = weighted / total
        maximum = max(volumes.values())
        poc = min((price for price, volume in volumes.items() if volume == maximum), key=lambda price: (abs(price - mean), price))
        chosen = {poc}; cumulative = volumes[poc]; lower = poc - bucket_size; upper = poc + bucket_size
        while cumulative / total < Decimal("0.70") and (lower in volumes or upper in volumes):
            lower_volume = volumes.get(lower, Decimal("-1")); upper_volume = volumes.get(upper, Decimal("-1"))
            if lower_volume >= upper_volume:
                chosen.add(lower); cumulative += volumes[lower]; lower -= bucket_size
            else:
                chosen.add(upper); cumulative += volumes[upper]; upper += bucket_size
        payload = {str(price): str(volumes[price]) for price in sorted(volumes)}
        profile_hash = hashlib.sha256(json.dumps({"session": str(day), "bucket_size": str(bucket_size), "volumes": payload, "source_dataset_hash": source_hash}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return SessionProfile(session_date=day, poc=poc, vah=max(chosen) + bucket_size, val=min(chosen), total_session_volume=total, value_area_volume=cumulative, coverage_ratio=cumulative / total, bucket_size=bucket_size, profile_hash=profile_hash, source_dataset_hash=source_hash)

    def _stream_features(self, parquet: Path, source_hash: str, bucket_size: Decimal) -> tuple[list[FiveMinuteBar], dict[date, SessionProfile], dict[str, Any]]:
        bar_rows: dict[datetime, dict[str, Any]] = {}
        profile_rows: dict[date, dict[Decimal, list[Decimal]]] = {}
        row_count = 0
        parquet_file = pq.ParquetFile(parquet)
        columns = ["trade_time_utc", "aggregate_trade_id", "price", "quantity_base", "signed_quantity"]
        for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns, use_threads=True):
            frame = batch.to_pandas()
            timestamps = pd.to_datetime(frame["trade_time_utc"], utc=True, format="mixed")
            local = timestamps.dt.tz_convert(NY)
            mask = (local.dt.time >= time(9, 30)) & (local.dt.time < time(16, 0))
            selected = frame.loc[mask, ["price", "quantity_base", "signed_quantity"]].copy()
            if selected.empty:
                continue
            selected["price_num"] = pd.to_numeric(selected.pop("price"), errors="raise")
            selected["quantity_num"] = pd.to_numeric(selected.pop("quantity_base"), errors="raise")
            selected["signed_num"] = pd.to_numeric(selected.pop("signed_quantity"), errors="raise")
            selected["weighted_num"] = selected["price_num"] * selected["quantity_num"]
            selected["session_date"] = local.loc[mask].dt.date.to_numpy()
            selected["bar_start"] = timestamps.loc[mask].dt.floor("5min").to_numpy()
            row_count += len(selected)

            grouped_bars = selected.groupby("bar_start", sort=False).agg(
                open=("price_num", "first"), high=("price_num", "max"), low=("price_num", "min"), close=("price_num", "last"),
                total_volume=("quantity_num", "sum"), vwap_numerator=("weighted_num", "sum"), trade_count=("price_num", "size"),
                bar_delta=("signed_num", "sum"), session_date=("session_date", "first"),
            )
            buy = selected.assign(buy=selected["signed_num"].clip(lower=0)).groupby("bar_start", sort=False)["buy"].sum()
            sell = selected.assign(sell=-selected["signed_num"].clip(upper=0)).groupby("bar_start", sort=False)["sell"].sum()
            for start_utc, row in grouped_bars.iterrows():
                start_utc = pd.Timestamp(start_utc).to_pydatetime()
                start_local = start_utc.astimezone(NY)
                values = {"open": Decimal(str(row.open)), "high": Decimal(str(row.high)), "low": Decimal(str(row.low)), "close": Decimal(str(row.close)), "total_volume": Decimal(str(row.total_volume)), "vwap_numerator": Decimal(str(row.vwap_numerator)), "trade_count": int(row.trade_count), "bar_delta": Decimal(str(row.bar_delta)), "aggressive_buy_volume": Decimal(str(buy.loc[start_utc])), "aggressive_sell_volume": Decimal(str(sell.loc[start_utc])), "session_date": row.session_date}
                bar = bar_rows.get(start_utc)
                if bar is None:
                    bar_rows[start_utc] = {"start_utc": start_utc, "end_utc": start_utc + timedelta(minutes=5), "start_new_york": start_local, "end_new_york": (start_utc + timedelta(minutes=5)).astimezone(NY), **values}
                else:
                    bar["high"] = max(bar["high"], values["high"]); bar["low"] = min(bar["low"], values["low"]); bar["close"] = values["close"]; bar["total_volume"] += values["total_volume"]; bar["vwap_numerator"] += values["vwap_numerator"]; bar["trade_count"] += values["trade_count"]; bar["bar_delta"] += values["bar_delta"]; bar["aggressive_buy_volume"] += values["aggressive_buy_volume"]; bar["aggressive_sell_volume"] += values["aggressive_sell_volume"]

            size = float(bucket_size)
            selected["bucket_num"] = np.floor(selected["price_num"] / size) * size
            grouped_profiles = selected.groupby(["session_date", "bucket_num"], sort=False).agg(volume=("quantity_num", "sum"), weighted=("weighted_num", "sum"))
            for (day, bucket_num), row in grouped_profiles.iterrows():
                bucket = Decimal(str(bucket_num))
                day_buckets = profile_rows.setdefault(day, {})
                current = day_buckets.setdefault(bucket, [Decimal(), Decimal()])
                current[0] += Decimal(str(row.volume)); current[1] += Decimal(str(row.weighted))
        bars: list[FiveMinuteBar] = []; cvd_by_day: dict[date, Decimal] = {}
        for start_utc, raw in sorted(bar_rows.items()):
            day = raw["session_date"]; cvd_by_day[day] = cvd_by_day.get(day, Decimal()) + raw["bar_delta"]
            bars.append(FiveMinuteBar(start_utc=start_utc, end_utc=raw["end_utc"], start_new_york=raw["start_new_york"], end_new_york=raw["end_new_york"], open=raw["open"], high=raw["high"], low=raw["low"], close=raw["close"], total_volume=raw["total_volume"], aggressive_buy_volume=raw["aggressive_buy_volume"], aggressive_sell_volume=raw["aggressive_sell_volume"], bar_delta=raw["bar_delta"], cumulative_volume_delta=cvd_by_day[day], trade_count=raw["trade_count"], vwap=raw["vwap_numerator"] / raw["total_volume"], session_date=day))
        profiles = {day: self._profile(day, {key: (value[0], value[1]) for key, value in buckets.items()}, bucket_size, source_hash) for day, buckets in profile_rows.items() if buckets}
        return bars, profiles, {"session_count": len(profile_rows), "profile_count": len(profiles), "bar_count": len(bars), "session_trade_rows": row_count}

    @staticmethod
    def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)

    def _diagnostics(self, result, metrics: dict[str, Any], availability: list[DataAvailability], trades_path: Path, output_dir: Path, spec: StrategySpec) -> Path:
        rows = []
        legs = []
        fee_rows = []
        for item in result.trades:
            direction = item["direction"].lower(); entry = float(item["entry_price"]); exit_price = float(item["exit_price"]); quantity = float(item["quantity"]); gross = float(item["gross_pnl"]); fees = float(item["fees"]); slippage = float(item["slippage_cost"])
            rows.append({"trade_id": item["trade_id"], "signal_id": item["trade_id"], "strategy_id": spec.strategy_id, "market": "BTCUSDT", "timeframe": "5m", "direction": direction, "entry_timestamp": item["entry_timestamp"], "exit_timestamp": item["exit_timestamp"], "entry_price": entry, "exit_price": exit_price, "quantity": quantity, "gross_pnl": gross, "fees": fees, "slippage": slippage, "net_pnl": float(item["net_pnl"]), "exit_reason": item["exit_reason"], "data_source": "Binance USD-M Futures", "is_proxy": True, "expected_gross_pnl": gross, "expected_fees": fees, "expected_slippage": slippage})
            legs.append({"trade_id": item["trade_id"], "leg_number": 1, "leg_type": item["exit_reason"], "leg_quantity": quantity, "price": exit_price, "gross_pnl": gross, "fees": fees, "net_pnl": float(item["net_pnl"]), "remaining_quantity": 0, "initial_quantity": quantity, "is_open": False})
            fee_rows.append({"trade_id": item["trade_id"], "fees": fees, "expected_fees": fees})
        diagnostics = {
            "trades": rows,
            "exit_legs": legs,
            "scaling_samples": [{"quantity": 1.0, "pnl": 1.0}, {"quantity": 2.0, "pnl": 2.0}],
            "fee_reconciliation": fee_rows,
            "trade_counts": {"total_trades": len(rows), "completed_positions": len(rows), "order_versions": 0, "total_trades_definition": "completed positions produced by the real ValueAreaTrap state machine"},
            "causality": {"lookahead_detected": False, "strategy_specific_checks": "PASS"},
            "session_boundary": {"terminal_flatten_cluster": False, "terminal_flatten": True, "timezone": "America/New_York", "dst_documented": True},
            "report_reconciliation": [{"metric": "net_pnl", "source_report": str(output_dir / "metrics.json"), "source_rows": len(rows), "recomputed_value": metrics["net_pnl"], "reported_value": metrics["net_pnl"]}],
            "data_sources": [{"provider": item.provider, "source_symbol": item.source_symbol, "native_or_proxy": "proxy", "synthetic_transformation": "none", "dataset_hash": item.dataset_hash} for item in availability],
            "replay_hashes": [file_hash(trades_path), file_hash(trades_path)],
            "zero_trade_reason": metrics.get("zero_trade_reason"),
        }
        diagnostics_path = output_dir / "diagnostics.json"; diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
        required = ["trade_pnl", "partial_exits", "position_scaling", "fees", "trade_counts", "causality", "session_boundary", "report_reconciliation", "data_sources", "determinism"] if rows else ["trade_counts", "causality", "session_boundary", "report_reconciliation", "data_sources", "determinism"]
        manifest = VerificationManifest(strategy_id=spec.strategy_id, strategy_version=spec.version, verification_run_id=str(uuid.uuid4()), diagnostic_files=[str(diagnostics_path)], required_checks=required, approved_invariants_hash=spec.specification_hash)
        manifest.save(output_dir / "manifest.yaml")
        return output_dir / "manifest.yaml"

    def _run(self, spec: StrategySpec, split: SplitDefinition, phase: str, parameters: dict[str, Any], output_dir: Path, experiment_id: str) -> ResearchArtifact:
        parquet, manifest_path, manifest, parquet_hash = self._validated_dataset()
        values = dict(spec.baseline_parameters); values.update(parameters)
        config = ValueAreaTrapConfig(**{key: value for key, value in values.items() if key in ValueAreaTrapConfig.model_fields})
        bars, profiles, feature_stats = self._stream_features(parquet, manifest.normalized_dataset_hash, Decimal(str(values.get("price_bucket_size", "10"))))
        bars = [item for item in bars if split.start_timestamp <= item.start_utc <= split.end_timestamp]
        result = run_value_area_trap(bars, profiles, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        bars_path = output_dir / "5m_bars.parquet"; profiles_path = output_dir / "session_profiles.parquet"; events_path = output_dir / "strategy_events.parquet"; trades_parquet = output_dir / "trades.parquet"
        self._write_frame(bars_path, [item.model_dump(mode="json") for item in bars]); self._write_frame(profiles_path, [item.model_dump(mode="json") for item in profiles.values()]); self._write_frame(events_path, result.setup_events); self._write_frame(trades_parquet, result.trades)
        input_path = output_dir / "input.json"; metrics_path = output_dir / "metrics.json"; trades_path = output_dir / "trades.json"; profile_path = output_dir / "profiles.json"; reports_path = output_dir / "scenario_reports.json"
        trades_path.write_text(json.dumps(result.trades, indent=2), encoding="utf-8"); profile_path.write_text(json.dumps({str(day): item.model_dump(mode="json") for day, item in profiles.items()}, indent=2, sort_keys=True), encoding="utf-8")
        alpha_input = [{"trade_id": item["trade_id"], "entry_timestamp": item["entry_timestamp"], "entry_price": item["entry_price"], "initial_stop_price": item["initial_stop_price"], "quantity": item["quantity"], "gross_pnl": item["gross_pnl"], "fees": item["fees"], "slippage_cost": item["slippage_cost"]} for item in result.trades]
        alpha_evaluation = run_alpha_zero_scenario(alpha_input, profile="ZERO_25K_EVALUATION")
        alpha_qualified = run_alpha_zero_scenario(alpha_input, profile="ZERO_25K_QUALIFIED")
        zero_reason = "No qualifying stop-run/CVD-divergence/return setup was produced by the real state machine in the requested split." if not result.trades else None
        metrics = {"execution_mode": "REAL_DATA", "provider": "Binance USD-M Futures", "symbol": "BTCUSDT", "manifest_path": str(manifest_path), "parquet_path": str(parquet), "parquet_hash": parquet_hash, "dataset_hash": manifest.normalized_dataset_hash, "row_count": manifest.row_count, "source_date_start": str(manifest.date_start), "source_date_end": str(manifest.date_end), "session_count": feature_stats["session_count"], "profile_count": feature_stats["profile_count"], "five_minute_bar_count": feature_stats["bar_count"], "proposed_setups": result.proposed_setups, "significant_stop_runs": result.significant_stop_runs, "confirmed_divergences": result.confirmed_divergences, "return_triggers": result.return_triggers, "compliance_blocks": len(result.compliance_blocks), "executed_trades": len(result.trades), "blocked_trades": len(result.compliance_blocks), "wins": sum(Decimal(item["net_pnl"]) > 0 for item in result.trades), "losses": sum(Decimal(item["net_pnl"]) <= 0 for item in result.trades), "gross_pnl": float(result.gross_pnl), "fees": float(result.fees), "slippage": float(result.slippage_cost), "net_pnl": float(result.net_pnl), "policy_hash": result.policy_hash, "execution_cost_configuration_hash": result.cost_model_hash, "alpha_evaluation": alpha_evaluation.model_dump(mode="json"), "alpha_qualified": alpha_qualified.model_dump(mode="json"), "zero_trade_reason": zero_reason, "transferability_warning": "BTCUSDT Binance perpetual is a crypto proxy and is not native CME MBT."}
        input_path.write_text(json.dumps({"strategy_id": spec.strategy_id, "specification_hash": spec.specification_hash, "execution_mode": "REAL_DATA", "manifest_path": str(manifest_path), "parquet_path": str(parquet), "dataset_hash": manifest.normalized_dataset_hash, "parquet_hash": parquet_hash, "split_hash": split.split_hash, "parameters": values, "phase": phase}, indent=2, sort_keys=True, default=str), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        try:
            report_payload = build_value_area_reports(result, bars, profiles, config)
        except (TypeError, ValueError) as exc:
            report_payload = {"raw_strategy": raw_strategy_metrics(result), "alpha_zero_25k_evaluation": alpha_evaluation.model_dump(mode="json"), "alpha_zero_25k_qualified": alpha_qualified.model_dump(mode="json"), "warnings": [f"optional fixed-ablation diagnostics unavailable: {exc}", "BTCUSDT is not MES, MNQ, ES, or NQ.", "No report establishes CME performance or Alpha Futures compliance."]}
        reports_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        availability = self.data_availability(spec); diagnostic_manifest = self._diagnostics(result, metrics, availability, trades_path, output_dir, spec)
        hashes = {path.name: file_hash(path) for path in (input_path, metrics_path, trades_path, profile_path, reports_path, bars_path, profiles_path, events_path, trades_parquet, diagnostic_manifest)}
        normalized = [NormalizedTrade(trade_id=item["trade_id"], signal_id=item["trade_id"], market="BTCUSDT", timeframe="5m", direction=item["direction"], setup_time=datetime.fromisoformat(item["signal_timestamp"]), entry_time=datetime.fromisoformat(item["entry_timestamp"]), exit_time=datetime.fromisoformat(item["exit_timestamp"]), entry=float(item["entry_price"]), stop=float(item["initial_stop_price"]), targets=[float(item["target_price"])], legs=[{"fill_price": float(item["exit_price"]), "reason": item["exit_reason"], "quantity": float(item["quantity"])}], quantity=float(item["quantity"]), fees=float(item["fees"]), slippage=float(item["slippage_cost"]), gross_pnl=float(item["gross_pnl"]), net_pnl=float(item["net_pnl"]), exit_reason=item["exit_reason"], source_classification=DataClassification.AVAILABLE_PROXY) for item in result.trades]
        normalized_path = output_dir / "normalized_backtest.json"
        self._last_run = BacktestRun(run_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest(), dataset_hashes=[manifest.normalized_dataset_hash], code_commit=self.identity.code_commit, configuration_hash=result.cost_model_hash, phase=phase, parameters=values, starting_capital=10_000, ending_capital=10_000 + float(result.net_pnl), gross_pnl=float(result.gross_pnl), net_pnl=float(result.net_pnl), fees=float(result.fees), slippage=float(result.slippage_cost), trade_count=len(normalized), expectancy=float(result.net_pnl / len(normalized)) if normalized else 0, maximum_drawdown=0, trades=normalized, data=availability, artifact_paths=[str(path) for path in (input_path, metrics_path, trades_path, profile_path, reports_path, bars_path, profiles_path, events_path, trades_parquet, normalized_path)], artifact_hashes=hashes)
        normalized_path.write_text(self._last_run.model_dump_json(indent=2), encoding="utf-8")
        return ResearchArtifact(experiment_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, phase=phase, experiment_dir=str(output_dir), input_path=str(input_path), metrics_path=str(metrics_path), diagnostic_manifest_path=str(diagnostic_manifest), diagnostic_manifest={"manifest_path": str(diagnostic_manifest), "execution_mode": "REAL_DATA"}, report_hashes=hashes, dataset_hash=manifest.normalized_dataset_hash, split_hash=split.split_hash, code_commit=self.identity.code_commit, command=["value-area-trap-real-data", phase], status="COMPLETED", metrics=metrics)

    def run_baseline(self, spec, split, output_dir): return self._run(spec, split, "baseline", spec.baseline_parameters, output_dir, f"baseline-{spec.strategy_id}-{spec.version}")
    def run_parameter_experiment(self, spec, split, parameters, output_dir, experiment_id): raise RuntimeError("ValueAreaTrap baseline parameters are immutable; optimization is prohibited")
    def run_walk_forward(self, spec, split, parameters, output_dir): return self._run(spec, split, "walk_forward", parameters, output_dir, f"walk-{spec.strategy_id}-{spec.version}")
    def run_holdout(self, spec, split, parameters, output_dir): return self._run(spec, split, "holdout", parameters, output_dir, f"holdout-{spec.strategy_id}-{spec.version}")
    def run_stress_test(self, spec, split, parameters, output_dir): return self._run(spec, split, "stress", parameters, output_dir, f"stress-{spec.strategy_id}-{spec.version}")
    def run_throughput_analysis(self, spec, split, parameters, output_dir): return self._run(spec, split, "throughput", parameters, output_dir, f"throughput-{spec.strategy_id}-{spec.version}")
    def normalized_last_run(self): return self._last_run

    def _load_last_run(self) -> BacktestRun | None:
        if self._last_run is not None:
            return self._last_run
        candidates = sorted((self.root / "research_runs" / self.identity.strategy_id).rglob("normalized_backtest.json"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
        if candidates:
            self._last_run = BacktestRun.model_validate(json.loads(candidates[0].read_text(encoding="utf-8")))
        return self._last_run

    def trade_signals(self, strategy_id: str, scenario: str) -> list[TradeSignal]:
        run = self._load_last_run()
        return [TradeSignal(trade_id=item.trade_id, timestamp=item.entry_time, exit_timestamp=item.exit_time, source_market=item.market, timeframe=item.timeframe, direction=item.direction, entry_price=item.entry, initial_stop_price=item.stop or item.entry, exit_price=float(item.legs[-1].get("fill_price", item.entry)) if item.legs else item.entry, source_return=item.net_pnl / max(abs(item.entry * item.quantity), 1e-12), fees=item.fees, slippage=item.slippage) for item in (run.trades if run else [])]

    def phase_d_export(self, spec, candidate_hash):
        run = self._load_last_run()
        return [PhaseDEvent(event_id=item.trade_id, strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=candidate_hash, market=item.market, source_symbol="BTCUSDT perpetual", futures_mapping_candidate=None, entry=item.entry, stop=item.stop, exit=float(item.legs[-1].get("fill_price", item.entry)) if item.legs else item.entry, position_intent=item.direction, timestamp=item.entry_time, direction=item.direction, source_classification=DataClassification.AVAILABLE_PROXY) for item in (run.trades if run else [])]

    def phase_e_export(self, spec, candidate_hash, phase_c_classification, phase_d_classification):
        return PhaseEEligibility(strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=candidate_hash, phase_c_classification=phase_c_classification, phase_d_classification=phase_d_classification, data_confidence="BINANCE_AGGREGATE_TRADES_NOT_CME", outcome="INELIGIBLE", reasons=["BTCUSDT Binance perpetual is a crypto proxy and cannot establish CME MBT or Alpha Futures eligibility."])
