"""Deterministic local cross-period V3 versus frozen robust-candidate stress runner.

The preflight path inventories files and published artifacts without opening a
DBN.  The explicit ``--run`` path is the only path that performs heavy local
replays.  It never downloads data, searches parameters, or changes either
frozen portfolio contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import causal_master_tape as master
from . import historical_runner as historical
from . import january_holdout_v3_vs_robust as january
from . import v2_august_seen_replay as august_seen
from . import v2_extended_existing_data as extended
from . import v3_poc_april_retro_replay as april
from . import v3_poc_fresh_august_replay as fresh_august
from . import v3_poc_only as v3
from .model import L2Config, StructuralLevel
from .v2_quality050 import V2_CONFIG


STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD"
OUTPUT_RELATIVE = Path("research_runs/CMEOrderflowAbsorption.ES_L2_ROBUST_CANDIDATE_CROSS_PERIOD")
V3_HASH = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
CANDIDATE_HASH = "5d3d72cd378d0ac986670a3c48ee14344571e1f559746e3fa1f514429e80553a"
CANDIDATE_ID = "W02-02-07-02-07-Q40"
CANDIDATE_LABEL = "DECEMBER_SELECTED_ROBUST_STRESS_CANDIDATE_NOT_VALIDATED"
NO_AGGREGATE_OOS_LABEL = "MIXED_EVIDENCE_CROSS_PERIOD_DESCRIPTIVE_NOT_OOS"

V3_CONFIG = V2_CONFIG
CANDIDATE_CONFIG = replace(
    V2_CONFIG,
    min_quality_score=0.40,
    aggression_weight=0.10,
    restoration_weight=0.10,
    price_resistance_weight=0.35,
    persistence_weight=0.10,
    multi_level_support_weight=0.35,
    weights_label=CANDIDATE_LABEL,
)


class CrossPeriodError(RuntimeError):
    pass


@dataclass(frozen=True)
class Portfolio:
    portfolio_id: str
    config: L2Config
    contract_hash: str


V3_PORTFOLIO = Portfolio("V3", V3_CONFIG, V3_HASH)
CANDIDATE_PORTFOLIO = Portfolio(CANDIDATE_ID, CANDIDATE_CONFIG, CANDIDATE_HASH)
PORTFOLIOS = (V3_PORTFOLIO, CANDIDATE_PORTFOLIO)


@dataclass(frozen=True)
class Period:
    period_id: str
    dates: tuple[str, ...]
    evidence_label: str
    source_root: Path
    source_model: str
    mes_source: str
    prior_rth_source: str
    calendar_semantics: str
    exact_v3_artifact: Path | None
    existing_non_v3_artifact: Path | None
    causal_master: Path | None
    replay_mode: str
    source_tail_complete: bool


PERIODS: tuple[Period, ...] = (
    Period(
        "APRIL_2026", april.TARGET_DATES, "RETROSPECTIVE_CROSS_PERIOD_STRESS_TEST",
        april.DATA_ROOT, "NATIVE_DATABENTO_MBP10", "NATIVE_DATABENTO_MBP1",
        "DATABENTO_ES_TRADES_PRIOR_COMPLETED_RTH", "SUMMER_UTC_FROZEN_2245",
        april.OUTPUT_ROOT, None, None, "REUSE_V3_REPLAY_CANDIDATE_NATIVE", True,
    ),
    Period(
        "MAY_2026", historical.MAY_DATES, "DEVELOPMENT_SEEN_DATA_CROSS_PERIOD_STRESS_TEST",
        Path("data/cme_orderflow_absorption_v2/may_2026_cost_proxy"),
        "MBO_DERIVED_SYNTHETIC_MBP10", "NATIVE_DATABENTO_MBP1",
        "DATABENTO_ES_TRADES_PRIOR_COMPLETED_RTH", "SUMMER_UTC_FROZEN_2245",
        None, Path("research_runs/CMEOrderflowAbsorption.ES_L2_V2_MAY_DEVELOPMENT_REPLAY"),
        None, "PAIRED_V3_CANDIDATE_MBO", True,
    ),
    Period(
        "RETRO_JUNE_JULY_2026", extended.RETRO_DATES, "RETROSPECTIVE_CROSS_PERIOD_STRESS_TEST",
        Path("data/cme_orderflow_absorption_v2_holdout"),
        "MBO_DERIVED_SYNTHETIC_MBP10", "NATIVE_DATABENTO_MBP1",
        "DATABENTO_ES_TRADES_PRIOR_COMPLETED_RTH", "SOURCE_END_1600_FAIL_CLOSED_NO_FORCED_EXIT",
        None, Path("research_runs/CMEOrderflowAbsorption.ES_L2_V2_EXTENDED_EXISTING_DATA/retro_june_july"),
        None, "PAIRED_V3_CANDIDATE_MBO_INCOMPLETE_TAIL", False,
    ),
    Period(
        "AUGUST_03_06_2026", august_seen.TARGET_DATES, "SEEN_AUGUST_CROSS_PERIOD_STRESS_TEST",
        Path("data/cme_orderflow_absorption_l2_v2/august_completion"),
        "MBO_DERIVED_SYNTHETIC_MBP10_SHARED_FILE", "NATIVE_DATABENTO_MBP1",
        "DATABENTO_ES_TRADES_PRIOR_COMPLETED_RTH", "SUMMER_UTC_FROZEN_2245_AUGUST_7_EXCLUDED",
        None, august_seen.DEFAULT_OUTPUT_ROOT, None, "PAIRED_V3_CANDIDATE_SHARED_AUGUST_MBO", True,
    ),
    Period(
        "AUGUST_10_14_2026", fresh_august.TARGET_DATES,
        "PREVIOUSLY_SEEN_FOR_CANDIDATE_CROSS_PERIOD_STRESS_TEST",
        fresh_august.DATA_ROOT, "NATIVE_DATABENTO_MBP10", "NATIVE_DATABENTO_MBP1",
        "DATABENTO_ES_TRADES_PRIOR_COMPLETED_RTH", "PERIOD_CALENDAR_WITH_AUGUST_14_SCHEDULED_CLOSE",
        fresh_august.OUTPUT_ROOT, None, None, "REUSE_V3_REPLAY_CANDIDATE_NATIVE", True,
    ),
    Period(
        "DECEMBER_2025", tuple(), "DECEMBER_SELECTION_DATA_NOT_OOS_EVIDENCE",
        master.OUTPUT_ROOT, "CAUSAL_MASTER_COMPACT_PARQUET", "CAUSAL_MASTER_MES_EVENTS",
        "CAUSAL_MASTER_PRIOR_RTH_LEVELS", "PERIOD_CORRECT_WINTER_CALENDAR",
        Path("research_runs/CMEOrderflowAbsorption.ES_L2_WEIGHT_Q_RESEARCH_DEC2025"), None,
        master.OUTPUT_ROOT, "REUSE_PUBLISHED_V3_AND_CANDIDATE", True,
    ),
    Period(
        "JANUARY_2026", tuple(), "JANUARY_2026_INTERNAL_HOLDOUT_AFTER_DECEMBER_SELECTION",
        master.OUTPUT_ROOT, "CAUSAL_MASTER_COMPACT_PARQUET", "CAUSAL_MASTER_MES_EVENTS",
        "CAUSAL_MASTER_PRIOR_RTH_LEVELS", "PERIOD_CORRECT_WINTER_CALENDAR",
        january.OUTPUT_RELATIVE, None, master.OUTPUT_ROOT,
        "REUSE_PUBLISHED_V3_AND_CANDIDATE", True,
    ),
)

METRIC_FIELDS = (
    "sessions", "completed_interactions", "accepted_setups", "confirmations",
    "confirmation_expiries", "active_position_blocks", "completed_trades", "wins", "losses",
    "win_rate", "total_r", "average_r", "median_r", "net_pnl_usd", "profit_factor",
    "max_cumulative_drawdown_r", "es_trades", "mes_trades", "target_exits", "stop_exits",
    "hard_cutoff_exits", "unresolved",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossPeriodError(f"missing or invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise CrossPeriodError(f"JSON artifact is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise CrossPeriodError(f"missing CSV artifact: {path}") from exc


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _present(value: object) -> bool:
    return value not in (None, "", "None", "null")


def _assert_frozen_contracts() -> None:
    if v3.v3_contract_sha256() != V3_HASH or january.V3_CONTRACT_SHA256 != V3_HASH:
        raise CrossPeriodError("FROZEN_V3_CONTRACT_HASH_MISMATCH")
    if january.challenger_contract_sha256() != CANDIDATE_HASH:
        raise CrossPeriodError("FROZEN_CANDIDATE_CONTRACT_HASH_MISMATCH")
    if tuple(v3.ELIGIBLE_STRUCTURAL_LEVELS) != ("PRIOR_RTH_POC",):
        raise CrossPeriodError("POC_ONLY_ELIGIBILITY_CHANGED")
    expected_v3 = {
        "aggression_weight": 0.28, "restoration_weight": 0.25,
        "price_resistance_weight": 0.22, "persistence_weight": 0.12,
        "multi_level_support_weight": 0.13, "min_quality_score": 0.50,
    }
    expected_candidate = {
        "aggression_weight": 0.10, "restoration_weight": 0.10,
        "price_resistance_weight": 0.35, "persistence_weight": 0.10,
        "multi_level_support_weight": 0.35, "min_quality_score": 0.40,
    }
    for name, expected in expected_v3.items():
        if not math.isclose(float(getattr(V3_CONFIG, name)), expected, rel_tol=0, abs_tol=1e-12):
            raise CrossPeriodError(f"FROZEN_V3_CONFIGURATION_CHANGED:{name}")
    for name, expected in expected_candidate.items():
        if not math.isclose(float(getattr(CANDIDATE_CONFIG, name)), expected, rel_tol=0, abs_tol=1e-12):
            raise CrossPeriodError(f"FROZEN_CANDIDATE_CONFIGURATION_CHANGED:{name}")
    changed = {
        key for key, value in asdict(V3_CONFIG).items()
        if asdict(CANDIDATE_CONFIG)[key] != value
    }
    allowed = {
        "aggression_weight", "restoration_weight", "price_resistance_weight",
        "persistence_weight", "multi_level_support_weight", "min_quality_score", "weights_label",
    }
    if changed != allowed:
        raise CrossPeriodError(f"PORTFOLIO_DIFFERENCE_SCOPE_CHANGED:{sorted(changed)}")


def _existing_files_for_period(repository_root: Path, period: Period) -> list[Path]:
    root = repository_root / period.source_root
    files: list[Path] = []
    if period.period_id == "APRIL_2026":
        for day in period.dates:
            files.extend(april._paths(root, day))
    elif period.period_id == "MAY_2026":
        for day in period.dates:
            files.extend(root / value for value in historical._may_paths(day).values())
    elif period.period_id == "RETRO_JUNE_JULY_2026":
        for day in period.dates:
            prior = extended.RETRO_PRIOR_RTH[day]
            files.extend((
                root / "es_mbo" / f"ESU6_{day}_0000_1600_mbo.dbn.zst",
                root / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst",
                root / "es_rth_trades" / f"ESU6_{prior}_1330_2000_trades.dbn.zst",
            ))
    elif period.period_id == "AUGUST_03_06_2026":
        files.append(repository_root / august_seen.ES_MBO_RELATIVE)
        for day in period.dates:
            files.extend(august_seen._paths(root, day))
    elif period.period_id == "AUGUST_10_14_2026":
        for day in period.dates:
            files.extend(fresh_august._paths(root, day))
    elif period.causal_master is not None:
        files.extend((
            repository_root / period.causal_master / "interaction-master.parquet",
            repository_root / period.causal_master / "interaction-event-index.parquet",
            repository_root / period.causal_master / "calendar.json",
        ))
    return files


def _artifact_is_exact_v3(repository_root: Path, relative: Path | None) -> bool:
    if relative is None:
        return False
    summary_path = repository_root / relative / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
    except CrossPeriodError:
        return False
    if relative == january.OUTPUT_RELATIVE:
        gate = summary.get("v3_reproduction_gate", {})
        return (
            summary.get("status") == "JANUARY_INTERNAL_HOLDOUT_COMPLETE"
            and gate.get("status") == "PASS"
            and summary.get("candidate_contract_sha256") == CANDIDATE_HASH
        )
    if relative.name == "CMEOrderflowAbsorption.ES_L2_WEIGHT_Q_RESEARCH_DEC2025":
        gate = summary.get("v3_december_reproduction", {})
        return (
            summary.get("status") == "DECEMBER_WEIGHT_Q_RESEARCH_COMPLETE_NO_SELECTION"
            and int(gate.get("completed_trades", -1)) == 38
        )
    return summary.get("strategy_id") == v3.STRATEGY_ID and summary.get("v3_contract_sha256") == V3_HASH


def build_source_inventory(repository_root: Path) -> dict[str, Any]:
    """Inventory declared local paths and artifacts without opening any DBN."""
    _assert_frozen_contracts()
    repository_root = repository_root.resolve()
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        dates = period.dates
        if period.period_id in {"DECEMBER_2025", "JANUARY_2026"}:
            calendar = _read_json(repository_root / master.OUTPUT_ROOT / "calendar.json")
            prefix = "2025-12" if period.period_id == "DECEMBER_2025" else "2026-01"
            dates = tuple(day for day in calendar.get("target_sessions", []) if str(day).startswith(prefix))
        files = _existing_files_for_period(repository_root, period)
        missing = [str(path) for path in files if not path.is_file()]
        exact_v3 = _artifact_is_exact_v3(repository_root, period.exact_v3_artifact)
        old_v2_exists = bool(
            period.existing_non_v3_artifact
            and (repository_root / period.existing_non_v3_artifact / "summary.json").is_file()
        )
        if period.period_id in {"DECEMBER_2025", "JANUARY_2026"}:
            evaluable = not missing and exact_v3
        else:
            evaluable = not missing
        rows.append({
            "period_id": period.period_id,
            "dates": list(dates),
            "session_count": len(dates),
            "source_root": str(repository_root / period.source_root),
            "source_model": period.source_model,
            "mes_source": period.mes_source,
            "prior_rth_source": period.prior_rth_source,
            "calendar_semantics": period.calendar_semantics,
            "source_tail_complete": period.source_tail_complete,
            "evidence_label": period.evidence_label,
            "exact_v3_artifact": None if period.exact_v3_artifact is None else str(repository_root / period.exact_v3_artifact),
            "exact_v3_artifact_reusable": exact_v3,
            "older_all_level_v2_artifact": None if period.existing_non_v3_artifact is None else str(repository_root / period.existing_non_v3_artifact),
            "older_v2_poc_subset_is_exact_v3": False if old_v2_exists else None,
            "causal_master": None if period.causal_master is None else str(repository_root / period.causal_master),
            "causal_master_or_compact_available": period.causal_master is not None and not missing,
            "replay_mode": period.replay_mode,
            "heavy_local_replay_required": period.replay_mode.startswith(("PAIRED_", "REUSE_V3_REPLAY_CANDIDATE")),
            "candidate_evaluable_without_download": evaluable,
            "missing_paths": missing,
            "status": "READY_FROM_EXISTING_LOCAL_DATA" if evaluable else "INSUFFICIENT_EXISTING_LOCAL_DATA",
        })
    return {
        "strategy_id": STRATEGY_ID,
        "inventory_scope": "FILESYSTEM_AND_PUBLISHED_ARTIFACT_METADATA_ONLY_NO_DBN_OPEN",
        "v3_contract_sha256": V3_HASH,
        "candidate_id": CANDIDATE_ID,
        "candidate_contract_sha256": CANDIDATE_HASH,
        "aggregate_evidence_label": NO_AGGREGATE_OOS_LABEL,
        "periods": rows,
        "downloads_permitted": False,
        "databento_calls": 0,
        "candidate_reselection": False,
    }


def _select_poc(levels: Iterable[StructuralLevel]) -> tuple[StructuralLevel, ...]:
    selected = v3.filter_eligible_levels(levels)
    if len(selected) != 1 or selected[0].name != "PRIOR_RTH_POC":
        raise CrossPeriodError("PRIOR_RTH_PROFILE_MUST_PRODUCE_EXACTLY_ONE_POC")
    return selected


def _new_portfolio_runners(day: str, evidence_label: str, levels: Sequence[StructuralLevel]) -> dict[str, historical.HistoricalL2Runner]:
    selected = _select_poc(levels)
    runners = {
        portfolio.portfolio_id: historical.HistoricalL2Runner(
            date=day, evidence_label=evidence_label, levels=selected,
            config=portfolio.config, strategy_id=v3.STRATEGY_ID,
            require_native_mes_for_fallback=True,
        )
        for portfolio in PORTFOLIOS
    }
    if runners[V3_PORTFOLIO.portfolio_id] is runners[CANDIDATE_PORTFOLIO.portfolio_id]:
        raise CrossPeriodError("PORTFOLIO_STATE_NOT_INDEPENDENT")
    return runners


def _run_mbo_pair_session(
    *, day: str, es_path: Path, mes_path: Path, profile_path: Path,
    evidence_label: str, source_end_seconds: int | None,
) -> dict[str, historical.HistoricalL2Runner]:
    levels = historical._profile_levels_from_declared_trades(profile_path)
    runners = _new_portfolio_runners(day, evidence_label, levels)
    adapter = historical.HistoricalMBOToMBP10Adapter()
    es_iter = iter(historical._stream_private_mbo(es_path))
    mes_iter = iter(historical._stream_mes_quotes(mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    start_ns = historical._clock_ns(day, historical.RTH_START_SECONDS)
    cutoff_ns = historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS)
    closed = False
    records = 0
    while es is not None or mes is not None:
        es_ts = es.timestamp_ns if es is not None else 2**63 - 1
        mes_ts = mes[0] if mes is not None else 2**63 - 1
        timestamp = min(es_ts, mes_ts)
        if source_end_seconds is None and timestamp >= cutoff_ns:
            for runner in runners.values():
                runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
                runner.finish(cutoff_ns)
            closed = True
            break
        if mes_ts < es_ts:
            if mes_ts >= start_ns and adapter.state not in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                for runner in runners.values():
                    runner.observe_mes_quote(*mes)
            mes = historical._next(mes_iter)
            continue
        record, es = es, historical._next(es_iter)
        records += 1
        try:
            public = adapter.feed(record, materialize_public=record.timestamp_ns >= start_ns)
        except historical.L2ValidationError as exc:
            raise CrossPeriodError(f"INVALID_MBO_SOURCE:{day}:{record.timestamp_ns}") from exc
        if record.timestamp_ns >= start_ns:
            for runner in runners.values():
                historical.route_mbo_public_event(runner, adapter, public, record.timestamp_ns)
        if records % 5_000_000 == 0:
            completed = {name: len(runner.interaction_ledger) for name, runner in runners.items()}
            print(f"CROSS_PERIOD_MBO {day} records={records:,} completed={completed}", flush=True)
    if source_end_seconds is None and not closed:
        for runner in runners.values():
            runner.force_flat_from_last_causal_cutoff_quote(cutoff_ns)
            runner.finish(cutoff_ns)
    adapter.finish()
    diagnostics = adapter.source_integrity_diagnostics()
    if source_end_seconds is not None:
        source_end_ns = historical._clock_ns(day, source_end_seconds)
        for runner in runners.values():
            runner.mark_source_end_incomplete(source_end_ns)
    for runner in runners.values():
        runner.source_integrity_diagnostics = [dict(row) for row in diagnostics]
    return runners


def _run_august_pair(repository_root: Path, completion_root: Path, evidence_label: str) -> dict[str, list[historical.HistoricalL2Runner]]:
    august_seen.verify_august_inputs(repository_root=repository_root, completion_root=completion_root)
    runners: dict[str, dict[str, historical.HistoricalL2Runner]] = {}
    adapters: dict[str, historical.HistoricalMBOToMBP10Adapter] = {}
    mes_iters: dict[str, Any] = {}
    mes_next: dict[str, tuple[int, float, float] | None] = {}
    cutoffs = {day: historical._clock_ns(day, historical.HARD_CUTOFF_SECONDS) for day in august_seen.TARGET_DATES}
    closed: set[str] = set()
    for day in august_seen.TARGET_DATES:
        profile, mes = august_seen._paths(completion_root, day)
        runners[day] = _new_portfolio_runners(
            day, evidence_label, historical._profile_levels_from_declared_trades(profile),
        )
        adapters[day] = historical.HistoricalMBOToMBP10Adapter()
        mes_iters[day] = iter(historical._stream_mes_quotes(mes))
        mes_next[day] = historical._next(mes_iters[day])
    records = 0
    for record in historical._stream_private_mbo(repository_root / august_seen.ES_MBO_RELATIVE):
        day = august_seen._date_from_ns(record.timestamp_ns)
        if day > august_seen.TARGET_DATES[-1]:
            break
        if day not in runners or day in closed:
            continue
        while mes_next[day] is not None and mes_next[day][0] < record.timestamp_ns:
            if adapters[day].state not in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                for runner in runners[day].values():
                    runner.observe_mes_quote(*mes_next[day])
            mes_next[day] = historical._next(mes_iters[day])
        if record.timestamp_ns >= cutoffs[day]:
            for runner in runners[day].values():
                runner.force_flat_from_last_causal_cutoff_quote(cutoffs[day])
                runner.finish(cutoffs[day])
            closed.add(day)
            continue
        records += 1
        try:
            public = adapters[day].feed(
                record,
                materialize_public=record.timestamp_ns >= historical._clock_ns(day, historical.RTH_START_SECONDS),
            )
        except historical.L2ValidationError as exc:
            raise CrossPeriodError(f"INVALID_AUGUST_MBO_SOURCE:{day}:{record.timestamp_ns}") from exc
        if record.timestamp_ns >= historical._clock_ns(day, historical.RTH_START_SECONDS):
            for runner in runners[day].values():
                historical.route_mbo_public_event(runner, adapters[day], public, record.timestamp_ns)
        if records % 5_000_000 == 0:
            print(f"CROSS_PERIOD_AUGUST_MBO records={records:,} closed={len(closed)}/4", flush=True)
    if closed != set(august_seen.TARGET_DATES):
        raise CrossPeriodError("AUGUST_SOURCE_DID_NOT_REACH_ALL_FOUR_CUTOFFS")
    by_portfolio = {portfolio.portfolio_id: [] for portfolio in PORTFOLIOS}
    for day in august_seen.TARGET_DATES:
        adapters[day].finish()
        diagnostics = adapters[day].source_integrity_diagnostics()
        for portfolio in PORTFOLIOS:
            runner = runners[day][portfolio.portfolio_id]
            runner.source_integrity_diagnostics = [dict(row) for row in diagnostics]
            by_portfolio[portfolio.portfolio_id].append(runner)
    return by_portfolio


def _setup_terminal_rows(runners: Sequence[historical.HistoricalL2Runner]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for runner in runners:
        runner.refresh_setup_ledger()
        rows.extend(dict(row) for row in runner.setup_ledger)
    return rows


def _result_from_runners(runners: Sequence[historical.HistoricalL2Runner], *, provenance: str) -> dict[str, Any]:
    setups = _setup_terminal_rows(runners)
    trades = [dict(row) for runner in runners for row in runner.trade_ledger]
    trades.sort(key=lambda row: (int(row["exit_timestamp_ns"]), str(row["trade_id"])))
    interactions = [dict(row) for runner in runners for row in runner.interaction_ledger]
    performance = historical._performance(trades)
    accepted = [row for row in setups if _truth(row.get("accepted"))]
    unresolved_ids = {
        str(row.get("setup_id")) for runner in runners for row in runner.source_end_unresolved
        if _present(row.get("setup_id"))
    }
    metrics = {
        "sessions": len(runners),
        "completed_interactions": len(interactions),
        "accepted_setups": len(accepted),
        "confirmations": sum(_present(row.get("confirmation_timestamp_ns")) for row in accepted),
        "confirmation_expiries": sum(row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED" for row in accepted),
        "active_position_blocks": sum(row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for row in accepted),
        **performance,
        "hard_cutoff_exits": sum(str(row.get("exit_reason", "")).startswith("HARD_") for row in trades),
        "unresolved": len(unresolved_ids),
    }
    return {
        "metrics": metrics, "trades": trades, "setup_rows": setups,
        "provenance": provenance, "independent_chronological_portfolio_state": True,
    }


def _result_from_v3_artifact(root: Path, dates: Sequence[str]) -> dict[str, Any]:
    summary = _read_json(root / "summary.json")
    if summary.get("strategy_id") != v3.STRATEGY_ID or summary.get("v3_contract_sha256") != V3_HASH:
        raise CrossPeriodError(f"EXACT_V3_ARTIFACT_CONTRACT_MISMATCH:{root}")
    wanted = set(dates)
    setups = [row for row in _read_csv(root / "setup-ledger.csv") if row.get("date") in wanted]
    trades = [row for row in _read_csv(root / "trade-ledger.csv") if row.get("date") in wanted]
    interactions = [row for row in _read_csv(root / "interaction-features.csv") if row.get("date") in wanted]
    performance = historical._performance(trades)
    accepted = [row for row in setups if _truth(row.get("accepted"))]
    unresolved_reasons = {
        "SOURCE_TAIL_UNRESOLVED", "SOURCE_NON_EXECUTABLE_BEFORE_ENTRY",
        "EXECUTION_UNRESOLVED_SOURCE_INCOMPLETE", "CONFIRMATION_UNRESOLVED_SOURCE_INCOMPLETE",
        "UNRESOLVED_SOURCE_END",
    }
    metrics = {
        "sessions": len(dates), "completed_interactions": len(interactions),
        "accepted_setups": len(accepted),
        "confirmations": sum(_present(row.get("confirmation_timestamp_ns")) for row in accepted),
        "confirmation_expiries": sum(row.get("terminal_reason") == "CONFIRMATION_WINDOW_EXPIRED" for row in accepted),
        "active_position_blocks": sum(row.get("terminal_reason") == "COMPLIANCE_BLOCK_ACTIVE_POSITION" for row in accepted),
        **performance,
        "hard_cutoff_exits": sum(str(row.get("exit_reason", "")).startswith("HARD_") for row in trades),
        "unresolved": sum(row.get("terminal_reason") in unresolved_reasons for row in accepted),
    }
    return {
        "metrics": metrics, "trades": trades, "setup_rows": setups,
        "provenance": f"REUSED_EXACT_V3_ARTIFACT:{root}",
        "independent_chronological_portfolio_state": True,
    }


def _numeric_row(row: Mapping[str, Any], completed_interactions: int, sessions: int) -> dict[str, Any]:
    def integer(name: str) -> int:
        return int(float(row.get(name, 0) or 0))

    def number(name: str) -> float | None:
        value = row.get(name)
        return None if value in (None, "", "None") else float(value)

    return {
        "sessions": sessions, "completed_interactions": completed_interactions,
        "accepted_setups": integer("accepted_setups"), "confirmations": integer("confirmations"),
        "confirmation_expiries": integer("confirmation_expiries"),
        "active_position_blocks": integer("active_position_blocks"),
        "completed_trades": integer("trades" if "trades" in row else "completed_trades"),
        "wins": integer("wins"), "losses": integer("losses"), "win_rate": number("win_rate"),
        "total_r": number("total_r"), "average_r": number("average_r"), "median_r": number("median_r"),
        "net_pnl_usd": number("net_pnl_usd"), "profit_factor": number("profit_factor"),
        "max_cumulative_drawdown_r": number("max_cumulative_drawdown_r"),
        "es_trades": integer("es_trades"), "mes_trades": integer("mes_trades"),
        "target_exits": integer("target_exits"), "stop_exits": integer("stop_exits"),
        "hard_cutoff_exits": integer("hard_cutoff_exits"), "unresolved": integer("unresolved"),
    }


def _load_december(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    matrix_root = repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_WEIGHT_Q_RESEARCH_DEC2025"
    matrix_summary = _read_json(matrix_root / "summary.json")
    baseline = dict(matrix_summary.get("compact_engine_v3_reproduction", {}))
    if int(baseline.get("completed_trades", -1)) != 38:
        raise CrossPeriodError("PUBLISHED_DECEMBER_V3_GATE_MISSING")
    reference = _read_json(
        repository_root / "research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_DEC2025_JAN2026_RETRO/summary.json"
    )
    january_v3 = _read_json(repository_root / january.OUTPUT_RELATIVE / "v3-results.json")
    december_daily = [row for row in reference.get("period_results", []) if str(row.get("date", "")).startswith("2025-12-")]
    sessions = len(december_daily)
    completed = sum(int(row.get("interactions_completed", 0)) for row in december_daily)
    accepted = sum(int(row.get("accepted_setups", 0)) for row in december_daily)
    combined_metrics = reference.get("metrics", {})
    confirmations = int(combined_metrics.get("confirmations_passed", 0)) - int(january_v3.get("confirmations", 0))
    confirmation_expiries = int(combined_metrics.get("confirmation_expiries", 0)) - int(january_v3.get("confirmation_expiries", 0))
    active_position_blocks = int(combined_metrics.get("active_position_blocks", 0)) - int(january_v3.get("active_position_blocks", 0))
    if accepted != confirmations + confirmation_expiries:
        raise CrossPeriodError("PUBLISHED_DECEMBER_V3_FUNNEL_MISMATCH")
    v3_metrics = {
        **baseline, "sessions": sessions, "completed_interactions": completed,
        "accepted_setups": accepted, "confirmations": confirmations,
        "confirmation_expiries": confirmation_expiries,
        "active_position_blocks": active_position_blocks, "unresolved": 0,
    }
    candidates = _read_csv(matrix_root / "weight-q-results.csv")
    rows = [row for row in candidates if row.get("config_id") == CANDIDATE_ID]
    if len(rows) != 1:
        raise CrossPeriodError("FROZEN_DECEMBER_CANDIDATE_ROW_MISSING_OR_DUPLICATE")
    candidate_row = rows[0]
    if not (
        math.isclose(float(candidate_row["quality_threshold"]), 0.40, abs_tol=1e-12)
        and all(math.isclose(float(candidate_row[f"G{i}"]), value, abs_tol=1e-12)
                for i, value in enumerate((0.10, 0.10, 0.35, 0.10, 0.35), start=1))
    ):
        raise CrossPeriodError("FROZEN_DECEMBER_CANDIDATE_PARAMETERS_MISMATCH")
    candidate_metrics = _numeric_row(candidate_row, completed, sessions)
    aggregate_overlap = [{
        "period_id": "DECEMBER_2025", "record_type": "AGGREGATE_ONLY",
        "common_trades": int(candidate_row["common_trades_with_v3"]),
        "candidate_only_trades": int(candidate_row["unique_trades_vs_v3"]),
        "v3_only_trades": int(candidate_row["v3_trades_lost"]),
        "reason": "MATRIX_PERSISTED_COUNTS_WITHOUT_INDIVIDUAL_CANDIDATE_TRADE_LEDGER",
    }]
    return (
        {"metrics": v3_metrics, "trades": [], "setup_rows": [], "provenance": "REUSED_DECEMBER_COMPACT_V3"},
        {"metrics": candidate_metrics, "trades": [], "setup_rows": [], "provenance": "REUSED_DECEMBER_MATRIX_CANDIDATE"},
        aggregate_overlap,
    )


def _load_january(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = repository_root / january.OUTPUT_RELATIVE
    summary = _read_json(root / "summary.json")
    if (
        summary.get("status") != "JANUARY_INTERNAL_HOLDOUT_COMPLETE"
        or summary.get("candidate_contract_sha256") != CANDIDATE_HASH
        or summary.get("v3_reproduction_gate", {}).get("status") != "PASS"
    ):
        raise CrossPeriodError("PUBLISHED_JANUARY_HOLDOUT_CONTRACT_MISMATCH")
    v3_result = _read_json(root / "v3-results.json")
    candidate = _read_json(root / "candidate-results.json")
    overlap = []
    for row in _read_csv(root / "trade-comparison.csv"):
        overlap.append({"period_id": "JANUARY_2026", "record_type": "TRADE", **row})
    v3_metrics = {key: v3_result.get(key) for key in METRIC_FIELDS}
    candidate_metrics = {key: candidate.get(key) for key in METRIC_FIELDS}
    v3_metrics["sessions"] = int(v3_result.get("eligible_sessions", 0))
    candidate_metrics["sessions"] = int(candidate.get("eligible_sessions", 0))
    return (
        {"metrics": v3_metrics, "trades": v3_result.get("trades", []),
         "setup_rows": [], "provenance": "REUSED_PUBLISHED_JANUARY_V3"},
        {"metrics": candidate_metrics, "trades": candidate.get("trades", []),
         "setup_rows": [], "provenance": "REUSED_PUBLISHED_JANUARY_CANDIDATE"},
        overlap,
    )


def _terminal_by_interaction(result: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in result.get("setup_rows", []):
        interaction_id = str(row.get("interaction_id", ""))
        if not interaction_id or not _truth(row.get("accepted")):
            continue
        values[interaction_id] = str(row.get("terminal_reason") or row.get("confirmation_status") or "UNKNOWN")
    return values


def _trade_overlap(period_id: str, v3_result: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    v3_trades = {str(row["interaction_id"]): row for row in v3_result.get("trades", [])}
    candidate_trades = {str(row["interaction_id"]): row for row in candidate.get("trades", [])}
    v3_terminal, candidate_terminal = _terminal_by_interaction(v3_result), _terminal_by_interaction(candidate)
    rows: list[dict[str, Any]] = []
    for interaction_id in sorted(set(v3_trades).union(candidate_trades)):
        baseline, challenger = v3_trades.get(interaction_id), candidate_trades.get(interaction_id)
        v3_reason = v3_terminal.get(interaction_id, "NOT_ACCEPTED")
        candidate_reason = candidate_terminal.get(interaction_id, "NOT_ACCEPTED")
        chronology = "NONE"
        if challenger is not None and v3_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION":
            chronology = "CANDIDATE_TRADE_UNBLOCKED_RELATIVE_TO_V3"
        elif baseline is not None and candidate_reason == "COMPLIANCE_BLOCK_ACTIVE_POSITION":
            chronology = "V3_TRADE_BLOCKED_IN_CANDIDATE"
        rows.append({
            "period_id": period_id, "record_type": "TRADE", "interaction_id": interaction_id,
            "common_trade": baseline is not None and challenger is not None,
            "v3_trade_id": "" if baseline is None else baseline.get("trade_id", ""),
            "candidate_trade_id": "" if challenger is None else challenger.get("trade_id", ""),
            "v3_r": "" if baseline is None else baseline.get("r_multiple", ""),
            "candidate_r": "" if challenger is None else challenger.get("r_multiple", ""),
            "v3_net_pnl_usd": "" if baseline is None else baseline.get("net_pnl_usd", ""),
            "candidate_net_pnl_usd": "" if challenger is None else challenger.get("net_pnl_usd", ""),
            "v3_terminal_outcome": v3_reason, "candidate_terminal_outcome": candidate_reason,
            "one_position_chronology_difference": chronology,
        })
    return rows


def _overlap_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    aggregate = [row for row in rows if row.get("record_type") == "AGGREGATE_ONLY"]
    if aggregate:
        if len(aggregate) != 1:
            raise CrossPeriodError("DUPLICATE_AGGREGATE_OVERLAP_ROWS")
        row = aggregate[0]
        return {
            "common_trades": int(row.get("common_trades", 0)),
            "candidate_only_trades": int(row.get("candidate_only_trades", 0)),
            "v3_only_trades": int(row.get("v3_only_trades", 0)),
            "one_position_chronology_differences": 0,
        }
    trades = [row for row in rows if row.get("record_type") == "TRADE"]
    return {
        "common_trades": sum(_truth(row.get("common_trade")) for row in trades),
        "candidate_only_trades": sum(not _present(row.get("v3_trade_id")) and _present(row.get("candidate_trade_id")) for row in trades),
        "v3_only_trades": sum(_present(row.get("v3_trade_id")) and not _present(row.get("candidate_trade_id")) for row in trades),
        "one_position_chronology_differences": sum(
            str(row.get("one_position_chronology_difference", "NONE")) != "NONE" for row in trades
        ),
    }


def _delta(v3_metrics: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    def difference(name: str) -> float | int | None:
        left, right = v3_metrics.get(name), candidate.get(name)
        return None if left is None or right is None else right - left

    return {
        "trade_delta": difference("completed_trades"), "win_rate_delta": difference("win_rate"),
        "total_r_delta": difference("total_r"), "net_pnl_usd_delta": difference("net_pnl_usd"),
        "profit_factor_delta": difference("profit_factor"),
        "max_cumulative_drawdown_r_delta": difference("max_cumulative_drawdown_r"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_period(
    root: Path,
    period: Period,
    v3_result: Mapping[str, Any],
    candidate: Mapping[str, Any],
    overlap: Sequence[Mapping[str, Any]],
) -> None:
    temporary = root.with_name(root.name + ".building")
    if root.exists() or temporary.exists():
        raise FileExistsError(f"period checkpoint collision: {root}")
    temporary.mkdir(parents=True)
    _write_json(temporary / "v3-results.json", {"metrics": v3_result["metrics"], "provenance": v3_result["provenance"]})
    _write_json(temporary / "candidate-results.json", {"metrics": candidate["metrics"], "provenance": candidate["provenance"]})
    _write_csv(temporary / "v3-trades.csv", v3_result.get("trades", []))
    _write_csv(temporary / "candidate-trades.csv", candidate.get("trades", []))
    _write_csv(temporary / "v3-setups.csv", v3_result.get("setup_rows", []))
    _write_csv(temporary / "candidate-setups.csv", candidate.get("setup_rows", []))
    _write_csv(temporary / "trade-overlap.csv", overlap)
    _write_json(temporary / "checkpoint.json", {
        "status": "PERIOD_CHECKPOINT_COMPLETE",
        "period_id": period.period_id,
        "evidence_label": period.evidence_label,
        "source_model": period.source_model,
        "v3_contract_sha256": V3_HASH,
        "candidate_contract_sha256": CANDIDATE_HASH,
        "candidate_id": CANDIDATE_ID,
    })
    os.rename(temporary, root)


def _load_period_checkpoint(
    root: Path, period: Period,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    contract = _read_json(root / "checkpoint.json")
    expected = {
        "status": "PERIOD_CHECKPOINT_COMPLETE",
        "period_id": period.period_id,
        "evidence_label": period.evidence_label,
        "source_model": period.source_model,
        "v3_contract_sha256": V3_HASH,
        "candidate_contract_sha256": CANDIDATE_HASH,
        "candidate_id": CANDIDATE_ID,
    }
    if contract != expected:
        raise CrossPeriodError(f"PERIOD_CHECKPOINT_CONTRACT_MISMATCH:{period.period_id}")

    def result(name: str) -> dict[str, Any]:
        payload = _read_json(root / f"{name}-results.json")
        return {
            "metrics": payload["metrics"],
            "provenance": payload["provenance"],
            "trades": _read_csv(root / f"{name}-trades.csv"),
            "setup_rows": _read_csv(root / f"{name}-setups.csv"),
            "independent_chronological_portfolio_state": True,
        }

    return result("v3"), result("candidate"), _read_csv(root / "trade-overlap.csv")


def _gross_components(metrics: Mapping[str, Any]) -> tuple[float, float] | None:
    net, pf = metrics.get("net_pnl_usd"), metrics.get("profit_factor")
    if net is None or pf is None:
        return None
    net, pf = float(net), float(pf)
    if math.isclose(pf, 1.0, abs_tol=1e-15):
        return None
    loss = net / (pf - 1.0)
    profit = pf * loss
    return max(0.0, profit), max(0.0, loss)


def _aggregate(portfolio: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [row[portfolio]["metrics"] for row in results]
    sums = {
        name: sum(int(item.get(name) or 0) for item in metrics)
        for name in (
            "sessions", "completed_interactions", "accepted_setups", "confirmations",
            "confirmation_expiries", "active_position_blocks", "completed_trades", "wins", "losses",
            "es_trades", "mes_trades", "target_exits", "stop_exits", "hard_cutoff_exits", "unresolved",
        )
    }
    total_r = sum(float(item.get("total_r") or 0.0) for item in metrics)
    net_pnl = sum(float(item.get("net_pnl_usd") or 0.0) for item in metrics)
    gross = [_gross_components(item) for item in metrics]
    profit_factor = None
    if all(item is not None for item in gross):
        total_profit = sum(item[0] for item in gross if item is not None)
        total_loss = sum(item[1] for item in gross if item is not None)
        profit_factor = total_profit / total_loss if total_loss else None
    trades = sums["completed_trades"]
    return {
        **sums, "win_rate": sums["wins"] / trades if trades else 0.0,
        "total_r": total_r, "average_r": total_r / trades if trades else 0.0,
        "net_pnl_usd": net_pnl, "profit_factor": profit_factor,
        "worst_period_max_cumulative_drawdown_r": min(float(item.get("max_cumulative_drawdown_r") or 0.0) for item in metrics),
        "cross_period_cumulative_drawdown_not_claimed": True,
    }


def _classification(rows: Sequence[Mapping[str, Any]]) -> str:
    r_wins = sum(float(row["delta"]["total_r_delta"]) > 0 for row in rows)
    pf_wins = sum(
        row["delta"]["profit_factor_delta"] is not None and float(row["delta"]["profit_factor_delta"]) > 0
        for row in rows
    )
    deltas_by_family: dict[str, float] = {}
    for row in rows:
        delta = float(row["delta"]["total_r_delta"])
        source = str(row["source_model"])
        family = "NATIVE_MBP10" if source.startswith("NATIVE_") else "MBO_DERIVED" if source.startswith("MBO_") else "COMPACT"
        deltas_by_family[family] = deltas_by_family.get(family, 0.0) + delta
    directional_signs = {
        1 if value > 0 else -1 if value < 0 else 0
        for value in deltas_by_family.values()
    }
    if {-1, 1}.issubset(directional_signs):
        return "MIXED_SOURCE_MODEL_PREVENTS_CLEAN_CONCLUSION"
    if r_wins >= 5 and pf_wins >= 5:
        return "CANDIDATE_BROADLY_OUTPERFORMS_V3"
    if r_wins >= 4 and pf_wins >= 4:
        return "CANDIDATE_MODESTLY_OUTPERFORMS_V3"
    if r_wins <= 2 and pf_wins <= 2:
        return "CANDIDATE_BROADLY_WEAKER_THAN_V3"
    return "CANDIDATE_ROUGHLY_EQUIVALENT_TO_V3"


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen robust Weight x Quality candidate cross-period stress test", "",
        f"Classification: `{summary['classification']}`", "",
        "This aggregate is descriptive mixed-evidence research, not OOS evidence. No parameter was searched, changed, promoted, or selected here.", "",
        "## Period comparisons", "",
        "| Period | Source model | V3 trades / W-L / WR / R / PnL / PF / DD | Candidate trades / W-L / WR / R / PnL / PF / DD | Delta trades / R / PnL |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["periods"]:
        vm, cm, delta = row["v3"]["metrics"], row["candidate"]["metrics"], row["delta"]
        lines.append(
            f"| {row['period_id']} | {row['source_model']} | {vm['completed_trades']} / {vm['wins']}-{vm['losses']} / "
            f"{vm['win_rate']:.4f} / {vm['total_r']:.6f} / {vm['net_pnl_usd']:.2f} / {vm['profit_factor']} / "
            f"{vm['max_cumulative_drawdown_r']:.6f} | {cm['completed_trades']} / {cm['wins']}-{cm['losses']} / "
            f"{cm['win_rate']:.4f} / {cm['total_r']:.6f} / {cm['net_pnl_usd']:.2f} / {cm['profit_factor']} / "
            f"{cm['max_cumulative_drawdown_r']:.6f} | {delta['trade_delta']:+d} / {delta['total_r_delta']:+.6f} / "
            f"{delta['net_pnl_usd_delta']:+.2f} |"
        )
    questions = summary["robustness_questions"]
    lines.extend(["", "## Descriptive robustness answers", ""])
    for key, value in questions.items():
        lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    lines.extend([
        "", "December is selection data; January is an internal holdout; April and June/July are retrospective; May and Aug 3-6 are seen/development; Aug 10-14 was fresh only for V3 and previously seen for this later-selected candidate.",
        "No aggregate OOS claim is permitted.", "",
    ])
    return "\n".join(lines)


def _run_period(repository_root: Path, period: Period) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if period.period_id == "APRIL_2026":
        data_root = repository_root / period.source_root
        april.verify_acquisition_manifest(data_root)
        v3_result = _result_from_v3_artifact(repository_root / april.OUTPUT_ROOT, period.dates)
        runners = [
            april._run_session(
                day, data_root, config=CANDIDATE_CONFIG, strategy_id=v3.STRATEGY_ID,
                evidence_label=period.evidence_label,
            )
            for day in period.dates
        ]
        candidate = _result_from_runners(runners, provenance="FROZEN_CANDIDATE_NATIVE_APRIL_REPLAY")
    elif period.period_id == "MAY_2026":
        data_root = repository_root / period.source_root
        historical.verify_may_acquisition_manifest(data_root)
        by_portfolio = {portfolio.portfolio_id: [] for portfolio in PORTFOLIOS}
        for index, day in enumerate(period.dates, start=1):
            print(f"CROSS_PERIOD MAY {index:02d}/{len(period.dates):02d} {day}", flush=True)
            paths = historical._may_paths(day)
            paired = _run_mbo_pair_session(
                day=day, es_path=data_root / paths["ES_MBO_L3"],
                mes_path=data_root / paths["MES_NATIVE_EXECUTION"],
                profile_path=data_root / paths["ES_PRIOR_RTH_PROFILE"],
                evidence_label=period.evidence_label, source_end_seconds=None,
            )
            for portfolio in PORTFOLIOS:
                by_portfolio[portfolio.portfolio_id].append(paired[portfolio.portfolio_id])
        v3_result = _result_from_runners(by_portfolio["V3"], provenance="PAIRED_MAY_V3_REPLAY")
        candidate = _result_from_runners(by_portfolio[CANDIDATE_ID], provenance="PAIRED_MAY_CANDIDATE_REPLAY")
    elif period.period_id == "RETRO_JUNE_JULY_2026":
        data_root = repository_root / period.source_root
        by_portfolio = {portfolio.portfolio_id: [] for portfolio in PORTFOLIOS}
        for index, day in enumerate(period.dates, start=1):
            print(f"CROSS_PERIOD RETRO {index:02d}/{len(period.dates):02d} {day}", flush=True)
            prior = extended.RETRO_PRIOR_RTH[day]
            paired = _run_mbo_pair_session(
                day=day,
                es_path=data_root / "es_mbo" / f"ESU6_{day}_0000_1600_mbo.dbn.zst",
                mes_path=data_root / "mes_mbp1" / f"MESU6_{day}_1300_1600_mbp1.dbn.zst",
                profile_path=data_root / "es_rth_trades" / f"ESU6_{prior}_1330_2000_trades.dbn.zst",
                evidence_label=period.evidence_label, source_end_seconds=16 * 3600,
            )
            for portfolio in PORTFOLIOS:
                by_portfolio[portfolio.portfolio_id].append(paired[portfolio.portfolio_id])
        v3_result = _result_from_runners(by_portfolio["V3"], provenance="PAIRED_RETRO_V3_REPLAY_INCOMPLETE_TAIL")
        candidate = _result_from_runners(by_portfolio[CANDIDATE_ID], provenance="PAIRED_RETRO_CANDIDATE_REPLAY_INCOMPLETE_TAIL")
    elif period.period_id == "AUGUST_03_06_2026":
        paired = _run_august_pair(repository_root, repository_root / period.source_root, period.evidence_label)
        v3_result = _result_from_runners(paired["V3"], provenance="PAIRED_AUGUST_SEEN_V3_REPLAY")
        candidate = _result_from_runners(paired[CANDIDATE_ID], provenance="PAIRED_AUGUST_SEEN_CANDIDATE_REPLAY")
    elif period.period_id == "AUGUST_10_14_2026":
        data_root = repository_root / period.source_root
        fresh_august.verify_acquisition_manifest(data_root)
        v3_result = _result_from_v3_artifact(repository_root / fresh_august.OUTPUT_ROOT, period.dates)
        runners = [
            fresh_august._run_session(
                day, data_root, config=CANDIDATE_CONFIG, strategy_id=v3.STRATEGY_ID,
                evidence_label=period.evidence_label,
            )
            for day in period.dates
        ]
        candidate = _result_from_runners(runners, provenance="FROZEN_CANDIDATE_NATIVE_FRESH_AUGUST_REPLAY")
    elif period.period_id == "DECEMBER_2025":
        return _load_december(repository_root)
    elif period.period_id == "JANUARY_2026":
        return _load_january(repository_root)
    else:
        raise CrossPeriodError(f"UNSUPPORTED_PERIOD:{period.period_id}")
    return v3_result, candidate, _trade_overlap(period.period_id, v3_result, candidate)


def run_all(*, repository_root: Path, output_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    expected_output = (repository_root / OUTPUT_RELATIVE).resolve()
    output_root = output_root.resolve()
    if output_root != expected_output:
        raise CrossPeriodError(f"OUTPUT_ROOT_MUST_EQUAL:{expected_output}")
    if output_root.exists():
        raise FileExistsError(f"immutable cross-period output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    inventory = build_source_inventory(repository_root)
    unavailable = [row for row in inventory["periods"] if row["status"] != "READY_FROM_EXISTING_LOCAL_DATA"]
    if unavailable:
        raise CrossPeriodError(f"INSUFFICIENT_EXISTING_LOCAL_DATA:{[row['period_id'] for row in unavailable]}")
    if staging.exists():
        if _read_json(staging / "source-inventory.json") != inventory:
            raise CrossPeriodError("RESUME_SOURCE_INVENTORY_MISMATCH")
    else:
        staging.mkdir(parents=True)
        _write_json(staging / "source-inventory.json", inventory)
    (staging / "periods").mkdir(exist_ok=True)
    period_results: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for index, period in enumerate(PERIODS, start=1):
        checkpoint = staging / "periods" / period.period_id.lower()
        if checkpoint.is_dir():
            print(f"CROSS_PERIOD {index:02d}/{len(PERIODS):02d} {period.period_id} RESUME_CHECKPOINT", flush=True)
            v3_result, candidate, overlap = _load_period_checkpoint(checkpoint, period)
        else:
            print(f"CROSS_PERIOD {index:02d}/{len(PERIODS):02d} {period.period_id}", flush=True)
            v3_result, candidate, overlap = _run_period(repository_root, period)
            _checkpoint_period(checkpoint, period, v3_result, candidate, overlap)
        overlap_rows.extend(overlap)
        period_results.append({
            "period_id": period.period_id, "evidence_label": period.evidence_label,
            "source_model": period.source_model, "source_tail_complete": period.source_tail_complete,
            "v3": v3_result, "candidate": candidate,
            "trade_overlap": _overlap_summary(overlap),
            "delta": _delta(v3_result["metrics"], candidate["metrics"]),
        })
    v3_aggregate = _aggregate("v3", period_results)
    candidate_aggregate = _aggregate("candidate", period_results)
    classification = _classification(period_results)
    total_r_wins = sum(float(row["delta"]["total_r_delta"]) > 0 for row in period_results)
    pf_wins = sum(
        row["delta"]["profit_factor_delta"] is not None and float(row["delta"]["profit_factor_delta"]) > 0
        for row in period_results
    )
    dd_wins = sum(float(row["delta"]["max_cumulative_drawdown_r_delta"]) > 0 for row in period_results)
    periods_with_candidate_trades = sum(
        int(row["candidate"]["metrics"]["completed_trades"]) > 0 for row in period_results
    )
    positive_periods = sum(
        int(row["candidate"]["metrics"]["completed_trades"]) > 0
        and float(row["candidate"]["metrics"]["total_r"]) > 0
        for row in period_results
    )
    deltas = [float(row["delta"]["total_r_delta"]) for row in period_results]
    largest = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
    candidate_only_trade_rows = [
        row for row in overlap_rows
        if row.get("record_type") == "TRADE"
        and not _present(row.get("v3_trade_id"))
        and _present(row.get("candidate_trade_id"))
        and _present(row.get("candidate_r"))
    ]
    candidate_only_r = [float(row["candidate_r"]) for row in candidate_only_trade_rows]
    candidate_period_r = [float(row["candidate"]["metrics"]["total_r"]) for row in period_results]
    absolute_r_sum = sum(abs(value) for value in candidate_period_r)
    dominant_index = max(range(len(candidate_period_r)), key=lambda index: abs(candidate_period_r[index]))
    source_model_deltas: dict[str, float] = {}
    for row in period_results:
        source_model_deltas[row["source_model"]] = (
            source_model_deltas.get(row["source_model"], 0.0) + float(row["delta"]["total_r_delta"])
        )
    january_row = next(row for row in period_results if row["period_id"] == "JANUARY_2026")
    non_january_r_deltas = [
        float(row["delta"]["total_r_delta"]) for row in period_results if row["period_id"] != "JANUARY_2026"
    ]
    non_january_r_deltas.sort()
    middle = len(non_january_r_deltas) // 2
    non_january_median = (
        (non_january_r_deltas[middle - 1] + non_january_r_deltas[middle]) / 2
        if len(non_january_r_deltas) % 2 == 0 else non_january_r_deltas[middle]
    )
    questions = {
        "candidate_positive_periods_with_trades": f"{positive_periods}/{periods_with_candidate_trades}",
        "candidate_outperforms_v3_in_total_r": f"{total_r_wins}/{len(period_results)}",
        "candidate_outperforms_v3_in_profit_factor": f"{pf_wins}/{len(period_results)}",
        "candidate_improves_max_drawdown": f"{dd_wins}/{len(period_results)}",
        "q40_additional_trade_quality": {
            "trade_level_periods_only": True,
            "candidate_only_trades": len(candidate_only_r),
            "wins": sum(value > 0 for value in candidate_only_r),
            "losses": sum(value < 0 for value in candidate_only_r),
            "total_r": sum(candidate_only_r),
            "december_trade_level_attribution_unavailable": True,
        },
        "january_representative_or_exception": {
            "january_total_r_delta": january_row["delta"]["total_r_delta"],
            "median_non_january_total_r_delta": non_january_median,
            "january_used_for_reselection": False,
        },
        "native_vs_mbo_stability": source_model_deltas,
        "largest_absolute_total_r_delta_period": period_results[largest]["period_id"],
        "candidate_single_period_dominance": {
            "period_id": period_results[dominant_index]["period_id"],
            "absolute_r_share_of_period_absolute_r": (
                abs(candidate_period_r[dominant_index]) / absolute_r_sum if absolute_r_sum else 0.0
            ),
        },
        "broad_assessment": classification,
    }
    summary_periods = [
        {
            **{key: row[key] for key in ("period_id", "evidence_label", "source_model", "source_tail_complete", "trade_overlap", "delta")},
            "v3": {"metrics": row["v3"]["metrics"], "provenance": row["v3"]["provenance"]},
            "candidate": {"metrics": row["candidate"]["metrics"], "provenance": row["candidate"]["provenance"]},
        }
        for row in period_results
    ]
    summary = {
        "status": "CROSS_PERIOD_STRESS_TEST_COMPLETE", "strategy_id": STRATEGY_ID,
        "classification": classification, "aggregate_evidence_label": NO_AGGREGATE_OOS_LABEL,
        "aggregate_is_oos": False, "candidate_promoted": False, "candidate_reselected": False,
        "v3_contract_sha256": V3_HASH, "candidate_id": CANDIDATE_ID,
        "candidate_contract_sha256": CANDIDATE_HASH,
        "frozen_v3_configuration": asdict(V3_CONFIG),
        "frozen_candidate_configuration": asdict(CANDIDATE_CONFIG),
        "periods": summary_periods,
        "aggregate": {"v3": v3_aggregate, "candidate": candidate_aggregate,
                      "delta": _delta(v3_aggregate, candidate_aggregate)},
        "robustness_questions": questions, "databento_calls": 0, "downloads": 0,
    }
    period_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for row in summary_periods:
        for portfolio in ("v3", "candidate"):
            period_rows.append({
                "period_id": row["period_id"], "portfolio": portfolio.upper(),
                "evidence_label": row["evidence_label"], "source_model": row["source_model"],
                **row[portfolio]["metrics"],
            })
        comparison_rows.append({
            "period_id": row["period_id"], "evidence_label": row["evidence_label"],
            "source_model": row["source_model"],
            **{f"v3_{key}": value for key, value in row["v3"]["metrics"].items()},
            **{f"candidate_{key}": value for key, value in row["candidate"]["metrics"].items()},
            **{f"overlap_{key}": value for key, value in row["trade_overlap"].items()},
            **row["delta"],
        })
    _write_csv(staging / "period-results.csv", period_rows)
    _write_csv(staging / "v3-vs-candidate.csv", comparison_rows)
    _write_csv(staging / "trade-overlap.csv", overlap_rows)
    _write_json(staging / "aggregate-comparison.json", summary["aggregate"])
    _write_json(staging / "summary.json", summary)
    (staging / "diagnostic-report.md").write_text(_report(summary), encoding="utf-8")
    os.rename(staging, output_root)
    return {
        "status": summary["status"], "classification": classification,
        "output_root": str(output_root), "period_count": len(period_results),
        "v3_contract_sha256": V3_HASH, "candidate_contract_sha256": CANDIDATE_HASH,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Inventory only; opens no DBN and writes no output")
    mode.add_argument("--run", action="store_true", help="Execute all required local heavy replays")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            build_source_inventory(args.repository_root)
            if args.preflight
            else run_all(repository_root=args.repository_root, output_root=args.output_root)
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
