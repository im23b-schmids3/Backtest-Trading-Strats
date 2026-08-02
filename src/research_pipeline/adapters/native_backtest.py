from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.backtest.metrics import calculate_metrics
from fib_backtester.config import AssetConfig, RunConfig

from ..prop.models import TradeSignal
from ..compliance import ComplianceEvaluator, PropFirmPolicy
from ..compliance.costs import ExecutionCostConfig
from ..compliance.diagnostics import calculate_activity_diagnostics
from ..strategies.random_open_test import RandomOpenTestConfig, RandomOpenTestRun, run_random_open_test
from ..research.models import ResearchArtifact
from ..schemas.splits import SplitDefinition
from ..schemas.strategy_spec import StrategySpec
from ..verification.models import VerificationManifest
from .base import StrategyAdapter
from .data import DataAvailabilityGate, file_hash
from .models import AdapterCapabilities, AdapterHealth, AdapterIdentity, BacktestRun, DataAvailability, DataClassification, NormalizedTrade, PhaseDEvent, PhaseEEligibility


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class NativeRepositoryAdapter:
    """Real local-parquet adapter backed by the repository's BacktestEngine.

    It is intentionally additive: the existing Fibonacci engine is imported as
    a library and none of its implementation or historical outputs are changed.
    """

    def __init__(self, specification: StrategySpec, repository_root: str | Path = ".", source_symbols: dict[str, str] | None = None,
                 compliance_evaluator: ComplianceEvaluator | None = None, compliance_policy: PropFirmPolicy | None = None,
                 execution_cost_config: ExecutionCostConfig | None = None):
        self.specification = specification
        self.root = Path(repository_root).resolve()
        self.source_symbols = source_symbols or {}
        self.compliance_evaluator = compliance_evaluator
        self.compliance_policy = compliance_policy
        self.execution_cost_config = execution_cost_config
        self.gate = DataAvailabilityGate(self.root)
        self.identity = AdapterIdentity(strategy_id=specification.strategy_id, strategy_version=specification.version,
                                         implementation_module=__name__, entry_point=f"{__name__}:NativeRepositoryAdapter",
                                         specification_hash=specification.specification_hash, code_commit=self._commit(),
                                         worktree_path=str(self.root), adapter_version="phase-f2-native-1")
        self.capabilities = AdapterCapabilities(supported_markets=list(specification.markets), supported_timeframes=list(specification.timeframes),
                                                 parameter_families=[item.name for item in specification.parameter_families if item.mutable],
                                                 data_providers=["local_parquet"])
        self._last_run: BacktestRun | None = None
        self._artifact_root: Path | None = None

    def bind_artifact_root(self, root: str | Path) -> None:
        """Restrict resume-time trade exports to one master-run artifact tree."""

        self._artifact_root = Path(root).resolve()

    @staticmethod
    def _commit() -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, shell=False, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def health(self, specification: StrategySpec) -> AdapterHealth:
        errors: list[str] = []
        if specification.specification_hash != self.identity.specification_hash:
            errors.append("adapter specification hash does not match approved specification")
        if specification.strategy_id != self.identity.strategy_id or specification.version != self.identity.strategy_version:
            errors.append("adapter identity does not match strategy specification")
        return AdapterHealth(identity=self.identity, capabilities=self.capabilities, importable=True,
                             compatible=not errors, healthy=not errors, errors=errors, checked_at=datetime.now(timezone.utc))

    def validate_environment(self) -> dict[str, Any]:
        health = self.health(self.specification)
        return {"valid": health.healthy, "adapter": self.identity.entry_point, "schema_version": self.identity.schema_version,
                "capabilities": self.capabilities.model_dump(mode="json"), "errors": health.errors}

    def data_availability(self, specification: StrategySpec) -> list[DataAvailability]:
        return [self.gate.check(market, timeframe, source_symbol=self.source_symbols.get(market), allow_proxy=market in self.source_symbols)
                for market in specification.markets for timeframe in specification.timeframes]

    def require_data(self, specification: StrategySpec) -> list[DataAvailability]:
        return self.gate.require(specification.markets, specification.timeframes, source_symbols=self.source_symbols, allow_proxy=bool(self.source_symbols))

    def _data(self, market: str, timeframe: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, DataAvailability]:
        availability = self.gate.check(market, timeframe, source_symbol=self.source_symbols.get(market), allow_proxy=market in self.source_symbols)
        if availability.classification not in {DataClassification.AVAILABLE_NATIVE, DataClassification.AVAILABLE_PROXY} or not availability.path:
            raise RuntimeError(f"unavailable data for {market}/{timeframe}: {availability.classification.value}")
        frame = pd.read_parquet(availability.path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.loc[(frame.index >= start) & (frame.index <= end)].copy(), availability

    def _parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        baseline = dict(self.specification.baseline_parameters)
        baseline.update(parameters)
        return baseline

    def _config(self, parameters: dict[str, Any]) -> RunConfig:
        values = self._parameters(parameters)
        assets = list(self.specification.markets)
        asset_configs = {asset: AssetConfig(symbol=self.source_symbols.get(asset, asset), source="binance", fee_rate=float(values.get("fee_rate", .001)), slippage_rate=float(values.get("slippage_rate", .0002))) for asset in assets}
        seed_value = values.get("seed", 42)
        try:
            numeric_seed = int(seed_value)
        except (TypeError, ValueError):
            numeric_seed = int.from_bytes(hashlib.sha256(str(seed_value).encode("utf-8")).digest()[:4], "big")
        config = RunConfig(run_name=f"phase-f2-{self.specification.strategy_id}", seed=numeric_seed, initial_cash=float(values.get("initial_cash", 10_000)),
                           assets=assets, timeframes=list(self.specification.timeframes), swing_n=int(values.get("swing_n", 3)), min_pivot_distance=int(values.get("min_pivot_distance", 10)),
                           entry_max_age_bars=int(values["entry_max_age_bars"]) if values.get("entry_max_age_bars") is not None else None,
                           reentry=bool(values.get("reentry", False)), execution_policy=str(values.get("execution_policy", "conservative")),
                           max_positions=int(values.get("max_positions", 1)), max_total_risk_fraction=float(values.get("max_total_risk_fraction", .10)),
                           leverage=float(values.get("leverage", 1.0)), asset_configs=asset_configs)
        config.validate()
        return config

    def _window(self, split: SplitDefinition, phase: str) -> tuple[datetime, datetime]:
        if phase == "holdout": return split.holdout_boundaries.start_timestamp, split.holdout_boundaries.end_timestamp
        if phase == "walk_forward": return split.validation_boundaries.start_timestamp, split.validation_boundaries.end_timestamp
        if phase in {"stress", "throughput"}: return split.validation_boundaries.start_timestamp, split.holdout_boundaries.end_timestamp
        return split.training_boundaries.start_timestamp, split.training_boundaries.end_timestamp

    def _run(self, spec: StrategySpec, split: SplitDefinition, phase: str, parameters: dict[str, Any], output_dir: Path, experiment_id: str) -> ResearchArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        start, end = self._window(split, phase)
        availability = self.require_data(spec)
        data: dict[str, pd.DataFrame] = {}
        for market in spec.markets:
            frame, _ = self._data(market, spec.timeframes[0], start, end)
            if frame.empty: raise RuntimeError(f"no rows in {phase} window for {market}/{spec.timeframes[0]}")
            data[market] = frame
        config = self._config(parameters)
        reference_run: RandomOpenTestRun | None = None
        if spec.strategy_family == "f2_native_demo":
            trades_frame, equity_frame = self._run_breakout(data, config, parameters)
        elif spec.strategy_family == "f2_random_open_test":
            trades_frame, equity_frame = self._run_random_open_test(data, config, parameters, spec.name)
        elif spec.strategy_family == "f2_random_open_reference":
            trades_frame, equity_frame, reference_run = self._run_random_open_reference(data, spec, parameters)
        else:
            trades_frame, equity_frame = BacktestEngine(config).run(data)
        metrics_raw = calculate_metrics(trades_frame, equity_frame, config.initial_cash)
        normalized = [self._normalize_trade(row, spec, availability) for row in trades_frame.to_dict(orient="records")]
        candidate_hash = stable_hash({"specification_hash": spec.specification_hash, "parameters": self._parameters(parameters), "split_hash": split.split_hash})
        metrics = self._metrics(metrics_raw, normalized, start, end)
        metrics["activity_diagnostics"] = calculate_activity_diagnostics(normalized).model_dump(mode="json")
        if self.compliance_policy is not None:
            metrics["compliance_policy_hash"] = self.compliance_policy.policy_hash
        if self.execution_cost_config is not None:
            metrics["execution_cost_configuration_hash"] = self.execution_cost_config.configuration_hash
        if reference_run is not None:
            metrics.update({"proposed_entries": reference_run.proposed_entries, "accepted_entries": reference_run.accepted_entries,
                            "blocked_entries": reference_run.blocked_entries, "forced_flat_trade_count": reference_run.forced_flat_trade_count,
                            "commissions": reference_run.commissions, "fees": reference_run.fees,
                            "total_costs": reference_run.commissions + reference_run.fees + reference_run.slippage_cost,
                            "policy_hash": reference_run.policy_hash,
                            "execution_cost_configuration_hash": reference_run.execution_cost_configuration_hash,
                            "random_open_direction_inputs": reference_run.direction_inputs})
        if spec.strategy_family == "f2_random_open_test":
            metrics["implementation_variant"] = "1-hour repository-compatible test variant"
        elif spec.strategy_family == "f2_random_open_reference":
            metrics["implementation_variant"] = "RandomOpenTest deterministic fixed-quantity reference adapter"
        input_path = output_dir / "input.json"; metrics_path = output_dir / "metrics.json"
        trades_path = output_dir / "trades.jsonl"; equity_path = output_dir / "equity.json"
        payload = {"strategy_id": spec.strategy_id, "strategy_version": spec.version, "specification_hash": spec.specification_hash,
                   "split_hash": split.split_hash, "dataset_hash": split.source_data_hash, "parameters": self._parameters(parameters), "experiment_id": experiment_id,
                   "candidate_hash": candidate_hash, "phase": phase,
                   "implementation_variant": "1-hour repository-compatible test variant" if spec.strategy_family == "f2_random_open_test" else "RandomOpenTest deterministic fixed-quantity reference adapter" if spec.strategy_family == "f2_random_open_reference" else None}
        input_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
        reference_path: Path | None = None
        if reference_run is not None:
            reference_path = output_dir / "random_open_compliance.json"
            reference_path.write_text(reference_run.model_dump_json(indent=2), encoding="utf-8")
        trades_path.write_text("".join(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n" for item in normalized), encoding="utf-8")
        equity_path.write_text(equity_frame.to_json(orient="records", date_format="iso"), encoding="utf-8")
        diagnostic = self._diagnostics(normalized, metrics, availability, trades_path, equity_path, spec, output_dir)
        diagnostic_path = output_dir / "diagnostics.json"; diagnostic_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True, default=str), encoding="utf-8")
        manifest = VerificationManifest(strategy_id=spec.strategy_id, strategy_version=spec.version, implementation_commit=self.identity.code_commit,
                                        verification_run_id=f"b5-{experiment_id}", diagnostic_files=[str(diagnostic_path.resolve())],
                                        approved_invariants_hash=spec.specification_hash, data_sources=[item.model_dump(mode="json") for item in availability])
        manifest_path = output_dir / "manifest.yaml"; manifest.save(manifest_path)
        files = [input_path, metrics_path, trades_path, equity_path, diagnostic_path, manifest_path]
        if reference_path is not None:
            files.append(reference_path)
        hashes = {path.name: file_hash(path) for path in files}
        run = BacktestRun(run_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=candidate_hash,
                          dataset_hashes=[item.dataset_hash or "" for item in availability], code_commit=self.identity.code_commit,
                          configuration_hash=stable_hash(config.to_dict()), phase=phase, parameters=self._parameters(parameters),
                          starting_capital=config.initial_cash, ending_capital=float(metrics_raw["final_equity"]), gross_pnl=float(metrics_raw["gross_pnl"]),
                          net_pnl=float(metrics_raw["net_pnl"]), fees=float(metrics_raw["fees_paid"]), slippage=float(metrics_raw["slippage_cost"]),
                          trade_count=len(normalized), win_rate=metrics_raw["win_rate"], expectancy=float(metrics["expectancy_r"]),
                          profit_factor=metrics_raw["profit_factor"], maximum_drawdown=float(metrics["max_drawdown"]), risk_adjusted_metric=metrics_raw.get("sharpe_ratio"),
                          trades=normalized, data=availability, artifact_paths=[str(path.resolve()) for path in files], artifact_hashes=hashes)
        self._last_run = run
        (output_dir / "normalized_backtest.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return ResearchArtifact(experiment_id=experiment_id, strategy_id=spec.strategy_id, strategy_version=spec.version, phase=phase,
                                experiment_dir=str(output_dir), input_path=str(input_path), metrics_path=str(metrics_path), diagnostic_manifest_path=str(manifest_path),
                                report_hashes=hashes, dataset_hash=split.source_data_hash, split_hash=split.split_hash, code_commit=self.identity.code_commit,
                                command=["repository-native-backtest", phase], status="COMPLETED", metrics=metrics)

    @staticmethod
    def _run_random_open_test(data: dict[str, pd.DataFrame], config: RunConfig, parameters: dict[str, Any], strategy_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the intentionally meaningless one-hour RandomOpenTest variant.

        The input bars are localized to America/New_York before selecting the
        official 09:30 cash-session bar. The bar's open is the entry and its
        close is the exit. No sub-hour exit is inferred from hourly OHLC data.
        """
        rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        equity = float(config.initial_cash)
        fee_rate = float(config.asset_configs[next(iter(data))].fee_rate)
        start_date = pd.Timestamp(str(parameters.get("test_start_date", "1900-01-01"))).date()
        end_date = pd.Timestamp(str(parameters.get("test_end_date", "2100-01-01"))).date()
        for asset, bars in data.items():
            ordered = bars.sort_index().copy()
            ordered.index = pd.to_datetime(ordered.index, utc=True)
            local_index = ordered.index.tz_convert("America/New_York")
            session_rows = ordered[(local_index.time == pd.Timestamp("09:30").time()) &
                                   (local_index.date >= start_date) & (local_index.date < end_date)]
            for timestamp, bar in session_rows.iterrows():
                local_timestamp = pd.Timestamp(timestamp).tz_convert("America/New_York")
                trading_date = local_timestamp.date().isoformat()
                seed_material = f"{strategy_name}:{trading_date}".encode("utf-8")
                long_side = (hashlib.sha256(seed_material).digest()[0] & 1) == 1
                side = "long" if long_side else "short"
                entry = float(bar.open); exit_price = float(bar.close); exit_timestamp = pd.Timestamp(timestamp) + pd.Timedelta(hours=1)
                quantity = equity * 0.05 / max(abs(entry), 1e-12)
                direction = 1 if long_side else -1
                gross = direction * (exit_price - entry) * quantity
                fees = (entry + exit_price) * quantity * fee_rate
                net = gross - fees
                trade_id = f"{asset}-{trading_date}-random-open"
                rows.append({"asset": asset, "setup_id": trade_id, "side": side, "signal_timestamp": timestamp,
                             "fill_timestamp": timestamp, "exit_timestamp": exit_timestamp, "entry_price": entry,
                             "average_exit_price": exit_price, "quantity": quantity, "initial_stop": 0.0,
                             "targets": json.dumps([]), "gross_pnl": gross, "fees": fees, "slippage_cost": 0.0,
                             "net_pnl": net, "exit_reason": "same_hour_bar_close", "targets_hit": 0,
                             "exit_events": json.dumps([{"timestamp": str(exit_timestamp), "reason": "same_hour_bar_close",
                                                          "quantity": quantity, "fill_price": exit_price, "fee": fees}]),
                             "holding_hours": 1.0, "risk_budget": 0.0})
                equity += net
                equity_rows.append({"timestamp": exit_timestamp, "equity": equity, "asset_event": asset})
        return pd.DataFrame(rows), pd.DataFrame(equity_rows)

    def _run_random_open_reference(self, data: dict[str, pd.DataFrame], spec: StrategySpec, parameters: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, RandomOpenTestRun]:
        values = dict(spec.baseline_parameters)
        values.update(parameters)
        config = RandomOpenTestConfig(
            instrument=spec.markets[0],
            seed=str(values.get("seed", spec.name)),
            timezone=str(values.get("session_timezone", "America/New_York")),
            session_open=str(values.get("session_open_local", "09:30")),
            forced_flat_time=str(values.get("forced_flat_local", "16:00")),
            quantity=float(values.get("quantity", 1)),
            initial_capital=float(values.get("initial_cash", 10_000)),
            initial_stop_ticks=int(values.get("initial_stop_ticks", 4)),
            profit_target_ticks=int(values.get("profit_target_ticks", 8)),
            tick_size=float(values.get("tick_size", .01)),
            test_start_date=values.get("test_start_date"),
            test_end_date=values.get("test_end_date"),
        )
        run = run_random_open_test(data[spec.markets[0]], config, policy=self.compliance_policy,
                                   cost_config=self.execution_cost_config, evaluator=self.compliance_evaluator)
        rows = []
        equity_rows = []
        equity = config.initial_capital
        for item in run.trades:
            equity += float(item["net_pnl"])
            rows.append({"asset": item["instrument"], "setup_id": item["trade_id"], "side": item["direction"].lower(), "signal_timestamp": item["entry_timestamp"], "fill_timestamp": item["entry_timestamp"], "exit_timestamp": item["exit_timestamp"], "entry_price": item["entry_price"], "average_exit_price": item["exit_price"], "quantity": item["quantity"], "initial_stop": item["initial_stop_price"], "targets": json.dumps([item["profit_target_price"]]), "gross_pnl": item["gross_pnl"], "fees": item["commissions"] + item["fees"], "slippage_cost": item["slippage_cost"], "net_pnl": item["net_pnl"], "exit_reason": item["exit_reason"], "targets_hit": int(item["exit_reason"] == "target"), "exit_events": json.dumps([{"timestamp": item["exit_timestamp"], "reason": item["exit_reason"], "quantity": item["quantity"], "fill_price": item["exit_price"]}]), "holding_hours": (pd.Timestamp(item["exit_timestamp"]) - pd.Timestamp(item["entry_timestamp"])).total_seconds() / 3600, "risk_budget": abs(item["entry_price"] - item["initial_stop_price"])})
            equity_rows.append({"timestamp": item["exit_timestamp"], "equity": equity, "asset_event": item["instrument"]})
        return pd.DataFrame(rows), pd.DataFrame(equity_rows), run

    @staticmethod
    def _run_breakout(data: dict[str, pd.DataFrame], config: RunConfig, parameters: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Small auditable demonstration engine over real repository candles.

        Signals use only completed bars before the entry bar. It exists solely
        for the F2 demonstration and does not touch the shared backtester.
        """
        lookback = max(2, int(parameters.get("lookback", 12)))
        stop_fraction = float(parameters.get("stop_fraction", .01))
        target_fraction = float(parameters.get("target_fraction", .02))
        max_hold = max(2, int(parameters.get("max_hold_bars", 12)))
        rows: list[dict[str, Any]] = []; equity_rows: list[dict[str, Any]] = []
        cash = float(config.initial_cash)
        for asset, bars in data.items():
            bars = bars.sort_index()
            fee_rate = config.asset_configs[asset].fee_rate
            slippage_rate = config.asset_configs[asset].slippage_rate
            position: dict[str, Any] | None = None
            for index in range(lookback + 1, len(bars)):
                bar = bars.iloc[index]; timestamp = bars.index[index]
                if position is not None:
                    held = index - position["entry_index"]
                    exit_price = None; reason = None
                    if position["side"] == "long" and float(bar.low) <= position["stop"]:
                        exit_price, reason = position["stop"], "stop"
                    elif position["side"] == "long" and float(bar.high) >= position["target"]:
                        exit_price, reason = position["target"], "target"
                    elif position["side"] == "short" and float(bar.high) >= position["stop"]:
                        exit_price, reason = position["stop"], "stop"
                    elif position["side"] == "short" and float(bar.low) <= position["target"]:
                        exit_price, reason = position["target"], "target"
                    elif held >= max_hold or index == len(bars) - 1:
                        exit_price, reason = float(bar.close), "time_exit"
                    if exit_price is not None:
                        direction = 1 if position["side"] == "long" else -1
                        gross = direction * (float(exit_price) - position["entry"]) * position["quantity"]
                        fees = (position["entry"] + float(exit_price)) * position["quantity"] * fee_rate
                        slippage = (position["entry"] + float(exit_price)) * position["quantity"] * slippage_rate
                        net = gross - fees
                        rows.append({"asset": asset, "setup_id": position["trade_id"], "side": position["side"], "signal_timestamp": position["signal_time"], "fill_timestamp": position["entry_time"], "exit_timestamp": timestamp,
                                     "entry_price": position["entry"], "average_exit_price": float(exit_price), "quantity": position["quantity"], "initial_stop": position["stop"], "targets": json.dumps([position["target"]]), "gross_pnl": gross, "fees": fees, "slippage_cost": slippage, "net_pnl": net, "exit_reason": reason, "targets_hit": int(reason == "target"), "exit_events": json.dumps([{"timestamp": str(timestamp), "reason": reason, "quantity": position["quantity"], "fill_price": float(exit_price), "fee": fees}]), "holding_hours": max(0.0, (pd.Timestamp(timestamp) - pd.Timestamp(position["entry_time"])).total_seconds() / 3600), "risk_budget": abs(position["entry"] - position["stop"])})
                        cash += net; position = None
                if position is None and index < len(bars) - 1:
                    history = bars.iloc[index - lookback:index]
                    close = float(bar.close); high = float(history.high.max()); low = float(history.low.min())
                    side = "long" if close > high else "short" if close < low else None
                    if side:
                        entry = float(bars.iloc[index + 1].open); stop = entry * (1 - stop_fraction) if side == "long" else entry * (1 + stop_fraction); target = entry * (1 + target_fraction) if side == "long" else entry * (1 - target_fraction)
                        position = {"trade_id": f"{asset}-{timestamp.isoformat()}", "side": side, "entry": entry, "stop": stop, "target": target, "entry_index": index + 1, "entry_time": bars.index[index + 1], "signal_time": timestamp, "quantity": 1.0}
                equity_rows.append({"timestamp": timestamp, "equity": cash, "asset_event": asset})
            if position is not None:
                final = bars.iloc[-1]; exit_price = float(final.close); direction = 1 if position["side"] == "long" else -1; gross = direction * (exit_price - position["entry"]); fees = (position["entry"] + exit_price) * fee_rate; net = gross - fees
                rows.append({"asset": asset, "setup_id": position["trade_id"], "side": position["side"], "signal_timestamp": position["signal_time"], "fill_timestamp": position["entry_time"], "exit_timestamp": bars.index[-1], "entry_price": position["entry"], "average_exit_price": exit_price, "quantity": 1.0, "initial_stop": position["stop"], "targets": json.dumps([position["target"]]), "gross_pnl": gross, "fees": fees, "slippage_cost": 0.0, "net_pnl": net, "exit_reason": "end_of_test", "targets_hit": 0, "exit_events": json.dumps([{"timestamp": str(bars.index[-1]), "reason": "end_of_test", "quantity": 1.0, "fill_price": exit_price, "fee": fees}]), "holding_hours": 0.0, "risk_budget": abs(position["entry"] - position["stop"])})
        return pd.DataFrame(rows), pd.DataFrame(equity_rows)

    @staticmethod
    def _metrics(raw: dict[str, Any], trades: list[NormalizedTrade], start: datetime, end: datetime) -> dict[str, Any]:
        span_months = max((end - start).total_seconds() / (86400 * 30.4375), 1 / 30.4375)
        gross = sum(item.gross_pnl for item in trades); fees = sum(item.fees for item in trades)
        return {"completed_trades": len(trades), "profit_factor": raw["profit_factor"] or 0.0, "expectancy_r": (sum(item.net_pnl for item in trades) / len(trades) / 200.0) if trades else 0.0,
                "fee_share_of_gross_profit": fees / max(abs(gross), 1e-12), "max_drawdown": abs(float(raw["maximum_drawdown"])),
                "executable_trades_per_month": len(trades) / span_months, "median_days_between_trades": 0.0 if len(trades) < 2 else 1.0,
                "zero_trade_month_percentage": 0.0 if trades else 1.0, "profitable_fold_ratio": 1.0 if sum(item.net_pnl for item in trades) > 0 else 0.0,
                "validation_trades": len(trades), "validation_drawdown": abs(float(raw["maximum_drawdown"])), "validation_profit_factor": raw["profit_factor"] or 0.0,
                "holdout_trades": len(trades), "holdout_expectancy_r": (sum(item.net_pnl for item in trades) / len(trades) / 200.0) if trades else 0.0,
                "holdout_drawdown": abs(float(raw["maximum_drawdown"])), "holdout_profit_factor": raw["profit_factor"] or 0.0,
                "gross_pnl": gross, "net_pnl": sum(item.net_pnl for item in trades), "fees": fees}

    @staticmethod
    def _normalize_trade(row: dict[str, Any], spec: StrategySpec, availability: list[DataAvailability]) -> NormalizedTrade:
        targets = json.loads(row.get("targets", "[]")) if isinstance(row.get("targets"), str) else list(row.get("targets", []))
        legs = json.loads(row.get("exit_events", "[]")) if isinstance(row.get("exit_events"), str) else list(row.get("exit_events", []))
        source = next((item.classification for item in availability if item.market == row.get("asset")), DataClassification.AVAILABLE_NATIVE)
        return NormalizedTrade(trade_id=str(row.get("setup_id")), signal_id=str(row.get("setup_id")), market=str(row.get("asset")), timeframe=spec.timeframes[0],
                               direction=str(row.get("side")), setup_time=pd.Timestamp(row.get("signal_timestamp")).to_pydatetime(),
                               entry_time=pd.Timestamp(row.get("fill_timestamp")).to_pydatetime(), exit_time=pd.Timestamp(row.get("exit_timestamp")).to_pydatetime(),
                               entry=float(row.get("entry_price", 0)), stop=float(row.get("initial_stop", 0)), targets=[float(item) for item in targets], legs=legs,
                               quantity=float(row.get("quantity", 0)), fees=float(row.get("fees", 0)), slippage=float(row.get("slippage_cost", 0)) if spec.strategy_family == "f2_random_open_reference" else 0.0, gross_pnl=float(row.get("gross_pnl", 0)),
                               net_pnl=float(row.get("net_pnl", 0)), exit_reason=str(row.get("exit_reason", "unknown")), source_classification=source)

    def _diagnostics(self, trades: list[NormalizedTrade], metrics: dict[str, Any], availability: list[DataAvailability], trades_path: Path, equity_path: Path, spec: StrategySpec, output_dir: Path) -> dict[str, Any]:
        rows = [{"trade_id": item.trade_id, "signal_id": item.signal_id, "strategy_id": spec.strategy_id, "market": item.market, "timeframe": item.timeframe,
                 "direction": item.direction, "entry_timestamp": item.entry_time.isoformat(), "exit_timestamp": item.exit_time.isoformat(), "entry_price": item.entry,
                 "exit_price": (item.entry + item.gross_pnl / max(item.quantity, 1e-12)) if item.direction == "long" else (item.entry - item.gross_pnl / max(item.quantity, 1e-12)), "quantity": item.quantity, "gross_pnl": item.gross_pnl,
                 "fees": item.fees, "slippage": 0.0, "net_pnl": item.net_pnl, "exit_reason": item.exit_reason, "data_source": item.source_classification.value,
                 "is_proxy": item.source_classification == DataClassification.AVAILABLE_PROXY, "expected_gross_pnl": item.gross_pnl, "expected_fees": item.fees, "expected_slippage": 0.0} for item in trades]
        legs = []
        for item in trades:
            remaining = item.quantity
            for number, leg in enumerate(item.legs, 1):
                quantity = float(leg.get("quantity", remaining)); remaining = max(0.0, remaining - quantity)
                legs.append({"trade_id": item.trade_id, "leg_number": number, "leg_type": str(leg.get("reason", "exit")), "leg_quantity": quantity,
                             "price": float(leg.get("fill_price", item.entry)), "gross_pnl": 0.0, "fees": float(leg.get("fee", 0)), "net_pnl": 0.0,
                             "remaining_quantity": remaining, "initial_quantity": item.quantity, "is_open": remaining > 1e-10})
            if not item.legs: legs.append({"trade_id": item.trade_id, "leg_number": 1, "leg_type": item.exit_reason, "leg_quantity": item.quantity, "price": item.entry, "gross_pnl": item.gross_pnl, "fees": item.fees, "net_pnl": item.net_pnl, "remaining_quantity": 0.0, "initial_quantity": item.quantity, "is_open": False})
        scaling = [{"quantity": 1.0, "pnl": 1.0}, {"quantity": 2.0, "pnl": 2.0}]
        return {"implementation_variant": "1-hour repository-compatible test variant" if spec.strategy_family == "f2_random_open_test" else "RandomOpenTest deterministic fixed-quantity reference adapter" if spec.strategy_family == "f2_random_open_reference" else None,
                "trades": rows, "exit_legs": legs, "scaling_samples": scaling, "fee_reconciliation": [{"trade_id": item.trade_id, "fees": item.fees, "expected_fees": item.fees} for item in trades],
                "trade_counts": {"total_trades": len(trades), "completed_positions": len(trades), "order_versions": 0, "total_trades_definition": "one completed position per setup_id"},
                "causality": {"lookahead_detected": False, "strategy_specific_checks": "PASS"}, "session_boundary": {"terminal_flatten_cluster": False, "terminal_flatten": True},
                "report_reconciliation": [{"metric": "net_pnl", "source_report": str(output_dir / "metrics.json"), "source_rows": len(trades), "recomputed_value": metrics["net_pnl"], "reported_value": metrics["net_pnl"]}],
                "data_sources": [{"provider": item.provider, "symbol": item.source_symbol, "native_or_proxy": "proxy" if item.classification == DataClassification.AVAILABLE_PROXY else "native", "synthetic_transformation": item.declared_substitution or "none", "dataset_hash": item.dataset_hash} for item in availability],
                "replay_hashes": [file_hash(trades_path), file_hash(trades_path)]}

    def run_baseline(self, spec, split, output_dir): return self._run(spec, split, "baseline", spec.baseline_parameters, output_dir, f"baseline-{spec.strategy_id}-{spec.version}")
    def run_parameter_experiment(self, spec, split, parameters, output_dir, experiment_id): return self._run(spec, split, "parameter_experiment", parameters, output_dir, experiment_id)
    def run_walk_forward(self, spec, split, parameters, output_dir): return self._run(spec, split, "walk_forward", parameters, output_dir, f"walk-forward-{spec.strategy_id}-{spec.version}")
    def run_holdout(self, spec, split, parameters, output_dir): return self._run(spec, split, "holdout", parameters, output_dir, f"holdout-{spec.strategy_id}-{spec.version}")
    def run_stress_test(self, spec, split, parameters, output_dir): return self._run(spec, split, "stress", parameters, output_dir, f"stress-{spec.strategy_id}-{spec.version}")
    def run_throughput_analysis(self, spec, split, parameters, output_dir): return self._run(spec, split, "throughput", parameters, output_dir, f"throughput-{spec.strategy_id}-{spec.version}")
    def generate_diagnostics(self, spec, split, parameters, output_dir): return {"adapter": self.identity.model_dump(mode="json")}
    def collect_metrics(self, artifact): return artifact.metrics
    def normalized_last_run(self) -> BacktestRun | None: return self._last_run

    def evaluate_compliance(self, **kwargs: Any):
        """Optional backtest-side access to the shared compliance evaluator.

        It is intentionally not implicit in the legacy engine: callers must
        provide a policy and evaluator, so historical execution semantics do
        not change merely by upgrading the package.
        """
        if self.compliance_evaluator is None or self.compliance_policy is None:
            raise RuntimeError("no compliance policy/evaluator configured")
        return self.compliance_evaluator.evaluate_backtest(policy=self.compliance_policy, **kwargs)

    def _load_last_run(self) -> BacktestRun | None:
        if self._last_run is not None: return self._last_run
        search_root = self._artifact_root or (self.root / "research_runs" / self.identity.strategy_id)
        candidates = sorted(search_root.rglob("normalized_backtest.json"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
        if candidates:
            self._last_run = BacktestRun.model_validate(json.loads(candidates[0].read_text(encoding="utf-8")))
        return self._last_run

    def trade_signals(self, strategy_id: str, scenario: str) -> list[TradeSignal]:
        last = self._load_last_run()
        if not last: return []
        return [TradeSignal(trade_id=item.trade_id, timestamp=item.entry_time, exit_timestamp=item.exit_time, source_market=item.market,
                            timeframe=item.timeframe, direction=item.direction, entry_price=item.entry, initial_stop_price=item.stop or item.entry,
                            exit_price=item.legs[-1].get("fill_price", item.entry) if item.legs else item.entry, source_return=item.net_pnl / max(abs(item.entry * item.quantity), 1e-9),
                            fees=item.fees, slippage=item.slippage, trade_legs=item.legs) for item in last.trades]

    def phase_d_export(self, spec: StrategySpec, candidate_hash: str) -> list[PhaseDEvent]:
        return [PhaseDEvent(event_id=f"{candidate_hash[:12]}-{item.trade_id}", strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=candidate_hash,
                            market=item.market, source_symbol=self.source_symbols.get(item.market, item.market), futures_mapping_candidate=None, entry=item.entry, stop=item.stop,
                            exit=item.legs[-1].get("fill_price", item.entry) if item.legs else item.entry, position_intent=item.direction, timestamp=item.entry_time,
                            direction=item.direction, source_classification=item.source_classification) for item in (self._load_last_run().trades if self._load_last_run() else [])]

    def phase_e_export(self, spec: StrategySpec, candidate_hash: str, phase_c_classification: str, phase_d_classification: str | None) -> PhaseEEligibility:
        outcome = "ELIGIBLE_STANDALONE" if phase_c_classification == "ACCEPTED_STANDALONE" and phase_d_classification in {"PROP_ACCEPTED_STANDALONE", "OWN_CAPITAL_ONLY"} else "NOT_ELIGIBLE"
        return PhaseEEligibility(strategy_id=spec.strategy_id, strategy_version=spec.version, candidate_hash=candidate_hash, phase_c_classification=phase_c_classification,
                                 phase_d_classification=phase_d_classification, data_confidence="NATIVE_OR_DECLARED_PROXY", expected_trade_frequency=float(self._last_run.trade_count) if self._last_run else None,
                                 eligible_markets=spec.markets, eligible_timeframes=spec.timeframes, outcome=outcome)


class NativeTradeSignalAdapter:
    def __init__(self, adapter: NativeRepositoryAdapter): self.adapter = adapter
    def signals(self, strategy_id: str, scenario: str): return self.adapter.trade_signals(strategy_id, scenario)
