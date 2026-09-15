"""Reusable three-stage, offline-first research orchestration for CME ES L2.

Stage 1 screens the *existing causal tapes* with a fixed quality gate and one
common legal weight vector while varying only target R and the canonical zone
stop buffer.  Stage 2 freezes that geometry per strategy and evaluates the
existing legal five-weight grid without opening DBN files.  Both stages are
development research only; neither selects or promotes a production strategy.
Stage 3 replays exactly one persisted Stage-2 selection per strategy and builds
independent plus aggregate-research accounting journals from canonical trades.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import causal_master_tape as master
from . import causal_level_resolver as levels
from . import weight_q_research as matrix


SCHEMA_VERSION = 1
EVIDENCE_LABEL = "MULTI_STRATEGY_DEVELOPMENT_RETROSPECTIVE_RESEARCH_NOT_OOS"
STAGE1_Q = Decimal("0.50")
STAGE1_RR = (1.5, 2.0, 2.5, 3.0, 4.0)
STAGE1_STOP_TICKS = (3, 5, 7)
STAGE2_Q = tuple(Decimal(value) for value in (
    "0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75",
))
SCORE_FIELDS = master.SCORE_FIELDS

# Three relevant Europe cells use the frozen W04 vector.  The current NY POC
# contract uses the older non-grid V2 vector.  The component-wise median of
# those four actual vectors is W04, which is also a legal 0.05 grid point.
COMMON_BASELINE_WEIGHTS: dict[str, Decimal] = {
    "aggression_score": Decimal("0.20"),
    "restoration_score": Decimal("0.10"),
    "price_resistance_score": Decimal("0.30"),
    "persistence_score": Decimal("0.20"),
    "multi_level_support_score": Decimal("0.20"),
}
BASELINE_DERIVATION = {
    "method": "componentwise_median_of_four_existing_relevant_contracts",
    "sources": {
        "prior_europe_high": ["0.20", "0.10", "0.30", "0.20", "0.20"],
        "current_europe_high_sweep": ["0.20", "0.10", "0.30", "0.20", "0.20"],
        "prior_europe_vah": ["0.20", "0.10", "0.30", "0.20", "0.20"],
        "prior_new_york_poc": ["0.28", "0.25", "0.22", "0.12", "0.13"],
    },
    "result": ["0.20", "0.10", "0.30", "0.20", "0.20"],
    "legal_grid_vector": True,
    "fallback_used": False,
}


class MultiStrategyResearchError(RuntimeError):
    """The reusable orchestration contract could not be satisfied."""


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    session: str
    source_session: str
    reference_level: str
    reference_semantics: str
    causal_availability_rule: str
    long_short_behavior: str
    uses_shared_absorption_engine: bool
    implementation_status: str
    level_resolver_required: bool
    stage1_enabled: bool
    stage2_enabled: bool
    trigger_options: Mapping[str, Any]
    baseline_quality: Mapping[str, Any]
    allowed_research_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class PeriodSpec:
    period_id: str
    interaction_master: Path
    interaction_index: Path
    sessions: tuple[tuple[str, Path], ...]
    level_catalog: Path | None = None


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise MultiStrategyResearchError(f"invalid manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise MultiStrategyResearchError(f"manifest root must be an object: {path}")
    return value


def load_strategy_manifest(path: Path) -> tuple[StrategySpec, ...]:
    raw = _read_manifest(path)
    if raw.get("schema_version") != SCHEMA_VERSION or not isinstance(raw.get("strategies"), list):
        raise MultiStrategyResearchError("strategy manifest requires schema_version=1 and strategies")
    specs: list[StrategySpec] = []
    for item in raw["strategies"]:
        if not isinstance(item, Mapping):
            raise MultiStrategyResearchError("strategy declaration must be an object")
        required = ("strategy_id", "session", "reference_level", "long_short_behavior", "uses_shared_absorption_engine")
        if any(not item.get(name) for name in required):
            raise MultiStrategyResearchError(f"strategy lacks required fields: {item}")
        for field in ("trigger_options", "baseline_quality", "allowed_research_parameters"):
            if field in item and item[field] is not None and not isinstance(item[field], Mapping):
                raise MultiStrategyResearchError(f"strategy {field} must be an object: {item['strategy_id']}")
        specs.append(StrategySpec(
            strategy_id=str(item["strategy_id"]), session=str(item["session"]),
            source_session=str(item.get("source_session") or item["session"]),
            reference_level=str(item["reference_level"]), long_short_behavior=str(item["long_short_behavior"]),
            reference_semantics=str(item.get("reference_semantics") or "UNSPECIFIED"),
            causal_availability_rule=str(item.get("causal_availability_rule") or "UNSPECIFIED"),
            uses_shared_absorption_engine=bool(item["uses_shared_absorption_engine"]),
            implementation_status=str(item.get("implementation_status") or "UNSPECIFIED"),
            level_resolver_required=bool(item.get("level_resolver_required", False)),
            stage1_enabled=bool(item.get("stage1_enabled", True)),
            stage2_enabled=bool(item.get("stage2_enabled", True)),
            trigger_options=dict(item.get("trigger_options") or {}),
            baseline_quality=dict(item.get("baseline_quality") or {}),
            allowed_research_parameters=dict(item.get("allowed_research_parameters") or {}),
        ))
    if not specs or len({item.strategy_id for item in specs}) != len(specs):
        raise MultiStrategyResearchError("strategy IDs must be non-empty and unique")
    if any(not item.uses_shared_absorption_engine for item in specs):
        raise MultiStrategyResearchError("this workflow currently requires the shared absorption engine")
    if any(not item.stage1_enabled or not item.stage2_enabled for item in specs):
        raise MultiStrategyResearchError("selected strategies must enable both Stage 1 and Stage 2")
    return tuple(sorted(specs, key=lambda item: item.strategy_id))


def load_period_manifest(path: Path) -> PeriodSpec:
    raw = _read_manifest(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MultiStrategyResearchError("period manifest requires schema_version=1")
    root = path.parent
    try:
        interaction_master = (root / str(raw["interaction_master"])).resolve()
        interaction_index = (root / str(raw["interaction_index"])).resolve()
        sessions = tuple((str(row["date"]), (root / str(row["event_tape"])).resolve()) for row in raw["sessions"])
    except (KeyError, TypeError) as exc:
        raise MultiStrategyResearchError("period manifest requires interaction artifacts and sessions") from exc
    days = tuple(day for day, _path in sessions)
    if not sessions or len(days) != len(set(days)) or tuple(sorted(days)) != days:
        raise MultiStrategyResearchError("period sessions must be non-empty, unique, and chronologically sorted")
    level_catalog = ((root / str(raw["level_catalog"])).resolve()
                     if raw.get("level_catalog") not in (None, "") else None)
    required = (interaction_master, interaction_index, *(item[1] for item in sessions),
                *((level_catalog,) if level_catalog is not None else ()))
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise MultiStrategyResearchError(f"period causal artifacts missing: {missing[:3]}")
    return PeriodSpec(str(raw.get("period_id") or "unnamed-period"), interaction_master, interaction_index, sessions, level_catalog)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") + "-" + _sha(value)[:10]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], default_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(default_fields)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _identity(strategy: StrategySpec, period: PeriodSpec, *, stage: str, extra: Mapping[str, Any]) -> str:
    sources = {
        "interactions": _file_sha(period.interaction_master), "indexes": _file_sha(period.interaction_index),
        "sessions": {day: _file_sha(path) for day, path in period.sessions},
    }
    if period.level_catalog is not None:
        sources["level_catalog"] = _file_sha(period.level_catalog)
    return _sha({"stage": stage, "strategy": strategy.__dict__, "period": period.period_id, "sources": sources, "extra": extra})


def _signal_timestamp(row: Mapping[str, Any]) -> int:
    value = row.get("interaction_start_ns", row.get("interaction_end_ns"))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MultiStrategyResearchError(f"interaction lacks causal signal timestamp: {row.get('interaction_id')}") from exc


def _resolved_provenance(rows_by_day: Mapping[str, Sequence[Mapping[str, Any]]], strategy: StrategySpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day, interactions in rows_by_day.items():
        for interaction in interactions:
            provenance = interaction.get("level_resolution")
            if isinstance(provenance, Mapping):
                rows.append(dict(provenance))
            else:
                rows.append({
                    "strategy_id": strategy.strategy_id, "trading_date": day, "target_session": strategy.session,
                    "source_session": strategy.source_session, "source_date": None,
                    "level_type": str(interaction.get("level")), "level_value": interaction.get("level_price"),
                    "source_artifact": "interaction-master-legacy-level", "source_artifact_sha256": None,
                    "causal_availability_timestamp_ns": None, "prior_current_semantics": strategy.reference_semantics,
                    "observation_mode": "LEGACY_PRE_RESOLVED", "available": True, "unavailable_reason": None,
                })
    unique = {json.dumps(row, sort_keys=True, separators=(",", ":")): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def _load_population(period: PeriodSpec, strategy: StrategySpec) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    interactions = master._read_parquet_rows(period.interaction_master)
    indexes = {str(row["interaction_id"]): dict(row) for row in master._read_parquet_rows(period.interaction_index)}
    result: dict[str, list[dict[str, Any]]] = {day: [] for day, _path in period.sessions}
    if period.level_catalog is None and strategy.level_resolver_required:
        raise MultiStrategyResearchError(f"level catalog required for cross-session strategy: {strategy.strategy_id}")
    resolver = levels.CausalLevelResolver.from_path(period.level_catalog) if period.level_catalog is not None else None
    for row in interactions:
        day = str(row.get("session_date"))
        target_session = row.get("target_session")
        if (day in result and str(row.get("level")) == strategy.reference_level
                and (target_session in (None, "", strategy.session))):
            identifier = str(row["interaction_id"])
            if identifier not in indexes:
                raise MultiStrategyResearchError(f"interaction index missing: {strategy.strategy_id}/{identifier}")
            resolved_row = dict(row)
            if resolver is not None:
                resolution = resolver.resolve(strategy, trading_date=day, signal_timestamp_ns=_signal_timestamp(row))
                if not resolution.available:
                    continue
                if not math.isclose(float(row["level_price"]), float(resolution.level_value), rel_tol=0.0, abs_tol=1e-9):
                    # The prebuilt interaction must be tied to exactly the level
                    # known at its own signal time; a later profile/extremum is
                    # never substituted into its zone geometry.
                    continue
                resolved_row["level_resolution"] = resolution.provenance()
                resolved_row["level_resolution_catalog_sha256"] = resolver.catalog_sha256
            result[day].append(resolved_row)
    return result, indexes


def _accepted(rows: Iterable[Mapping[str, Any]], weights: Mapping[str, Decimal], threshold: Decimal) -> list[dict[str, Any]]:
    # The historical master intentionally accepts only its sealed six-Q grid.
    # This orchestration retains its score and legal-weight functions while
    # extending the *new* development sweep to the explicitly declared Q30–75
    # values, without changing any legacy replay contract.
    if threshold < Decimal("0") or threshold > Decimal("1") or threshold * 20 != (threshold * 20).to_integral_value():
        raise MultiStrategyResearchError(f"unsupported 0.05 quality threshold: {threshold}")
    return [dict(row) for row in rows if not str(row.get("non_quality_rejection_reasons") or "")
            and Decimal(str(master.recompute_quality(row, weights))) >= threshold]


def _metrics(
    *, strategy: StrategySpec, period: PeriodSpec, rows_by_day: Mapping[str, Sequence[Mapping[str, Any]]],
    indexes: Mapping[str, Mapping[str, Any]], weights: Mapping[str, Decimal], threshold: Decimal,
    rr: float, stop_ticks: int, tape_cache: dict[tuple[str, float, int], matrix.SessionCausalTape] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accumulator = matrix.ConfigurationAccumulator((4, 2, 6, 4, 4), threshold)
    all_trades: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    for day, tape_path in period.sessions:
        candidates = _accepted(rows_by_day[day], weights, threshold)
        day_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in candidates}
        cache_key = (day, rr, stop_ticks)
        tape = (tape_cache or {}).get(cache_key)
        if tape is None:
            tape = matrix.SessionCausalTape.from_parquet(day, tape_path, stop_buffer_ticks=stop_ticks, target_r=rr)
            if tape_cache is not None:
                tape_cache[cache_key] = tape
        session = matrix.simulate_independent_session(tape, candidates, day_indexes)
        accumulator.add(session, set())
        day_accumulator = matrix.ConfigurationAccumulator((4, 2, 6, 4, 4), threshold)
        day_accumulator.add(session, set())
        day_row = day_accumulator.row(0)
        daily.append({"strategy_id": strategy.strategy_id, "date": day, "rr": rr, "stop_ticks": stop_ticks, **day_row})
        all_trades.extend(session.trades)
    row = accumulator.row(0)
    sessions = len(period.sessions)
    long_trades = [trade for trade in all_trades if trade["direction"] == "LONG"]
    short_trades = [trade for trade in all_trades if trade["direction"] == "SHORT"]
    return {
        "strategy_id": strategy.strategy_id, "period_id": period.period_id,
        "reference_level": strategy.reference_level, "rr": rr, "stop_ticks": stop_ticks,
        **row, "sessions_evaluated": sessions, "trades_per_session": row["trades"] / sessions,
        "expectancy_r_per_trade": row["average_r"], "expectancy_r_per_session": row["total_r"] / sessions,
        "long_trades": len(long_trades), "short_trades": len(short_trades),
        "long_total_r": sum(float(item["r_multiple"] or 0) for item in long_trades),
        "short_total_r": sum(float(item["r_multiple"] or 0) for item in short_trades),
        "hard_flat_exits": row["hard_cutoff_exits"],
    }, daily


def _stage1_neighbors(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_geometry = {(float(row["rr"]), int(row["stop_ticks"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        rr, stop = float(row["rr"]), int(row["stop_ticks"])
        neighbors = [by_geometry[item] for item in ((rr - 0.5, stop), (rr + 0.5, stop), (rr, stop - 2), (rr, stop + 2)) if item in by_geometry]
        # 4.0 is deliberately not considered adjacent to 3.0 by a fixed 0.5 step.
        if rr == 4.0 and (3.0, stop) in by_geometry:
            neighbors.append(by_geometry[(3.0, stop)])
        totals = [float(item["total_r"]) for item in neighbors]
        output.append({**row, "immediate_geometry_neighbor_count": len(neighbors),
            "neighbor_profitable_fraction": sum(value > 0 for value in totals) / len(totals) if totals else 0.0,
            "neighbor_worst_total_r": min(totals) if totals else float("-inf"),
            "neighbor_median_total_r": sorted(totals)[len(totals) // 2] if totals else None,
            "neighbor_ids": ",".join(f"RR{item['rr']}-S{item['stop_ticks']}" for item in sorted(neighbors, key=lambda item: (item["rr"], item["stop_ticks"]))),
        })
    return output


def _finite_sort(value: object) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else -1_000_000_000.0


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise MultiStrategyResearchError(f"missing summary input: {path}") from exc


def _number(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _stage1_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _number(row, "neighbor_profitable_fraction"), _finite_sort(row.get("neighbor_worst_total_r")),
        _number(row, "expectancy_r_per_session"), _number(row, "total_r"),
        _number(row, "max_cumulative_drawdown_r"), _finite_sort(row.get("profit_factor")),
        _number(row, "trades"), str(row.get("config_id", "")),
    )


def _stage1_important_rows(strategy: StrategySpec, strategy_root: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(strategy_root / "stage1-matrix.csv")
    selection = json.loads((strategy_root / "selection.json").read_text(encoding="utf-8"))
    by_id = {str(row["config_id"]): row for row in rows}
    raw_id = str(selection["raw_best"]["config_id"])
    robust_id = str(selection["stage2_execution_configuration"]["config_id"])
    if raw_id not in by_id or robust_id not in by_id:
        raise MultiStrategyResearchError(f"Stage 1 summary selection is absent from matrix: {strategy.strategy_id}")
    chosen: dict[str, set[str]] = {}

    def add(identifier: str, role: str) -> None:
        if identifier in by_id:
            chosen.setdefault(identifier, set()).add(role)

    for rank, row in enumerate(sorted(rows, key=_stage1_rank_key, reverse=True)[:5], 1):
        add(str(row["config_id"]), f"TOP5_ROBUST_RANK_{rank}")
    add(raw_id, "RAW_BEST")
    add(robust_id, "ROBUST_BEST_STAGE2_GEOMETRY")
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted((by_id[key] for key in chosen), key=_stage1_rank_key, reverse=True), 1):
        output.append({
            "important_rank": rank, "summary_roles": ";".join(sorted(chosen[str(row["config_id"])])),
            "strategy_id": strategy.strategy_id, "raw_best_config_id": raw_id, "robust_best_config_id": robust_id,
            "raw_best_rr": selection["raw_best"]["rr"], "raw_best_stop_ticks": selection["raw_best"]["stop_ticks"],
            "robust_best_rr": selection["stage2_execution_configuration"]["rr"],
            "robust_best_stop_ticks": selection["stage2_execution_configuration"]["stop_ticks"],
            "Q": str(STAGE1_Q), "stage1_quality_threshold": str(STAGE1_Q),
            "baseline_weights": json.dumps({key: str(value) for key, value in COMMON_BASELINE_WEIGHTS.items()}, sort_keys=True),
            **row,
        })
    return output


def _generate_stage1_summaries(strategies: Sequence[StrategySpec], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in sorted(strategies, key=lambda item: item.strategy_id):
        strategy_root = output_root / "strategies" / _safe_name(strategy.strategy_id)
        if (strategy_root / "complete.json").is_file():
            rows.extend(_stage1_important_rows(strategy, strategy_root))
    rows.sort(key=lambda row: (str(row["strategy_id"]), int(row["important_rank"]), str(row["config_id"])))
    _write_csv(output_root / "stage1-important-summary.csv", rows, ("important_rank", "strategy_id", "config_id"))
    return rows


def _select_stage1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = max(rows, key=lambda item: (float(item["total_r"]), float(item["expectancy_r_per_session"]), str(item["config_id"])))
    robust = max(rows, key=lambda item: (
        float(item["neighbor_profitable_fraction"]), float(item["neighbor_worst_total_r"]),
        float(item["expectancy_r_per_session"]), float(item["total_r"]),
        float(item["max_cumulative_drawdown_r"]), _finite_sort(item["profit_factor"]), int(item["trades"]),
        str(item["config_id"]),
    ))
    return {
        "selection_rule": "rank by profitable-neighbor fraction, worst immediate-neighbor total R, expectancy/session, total R, lower drawdown, profit factor, then trade count; all configurations remain eligible for Stage 2",
        "raw_best": {"config_id": raw["config_id"], "rr": raw["rr"], "stop_ticks": raw["stop_ticks"]},
        "stage2_execution_configuration": {"config_id": robust["config_id"], "rr": robust["rr"], "stop_ticks": robust["stop_ticks"]},
        "automatic_strategy_elimination": False,
    }


def _complete_or_raise(root: Path, identity: str) -> dict[str, Any] | None:
    marker = root / "complete.json"
    if not marker.is_file():
        return None
    existing = json.loads(marker.read_text(encoding="utf-8"))
    if existing.get("input_identity") != identity:
        raise MultiStrategyResearchError(f"completed output identity differs: {root}")
    return dict(existing)


def _run_stage1_strategy(strategy: StrategySpec, period: PeriodSpec, root: Path) -> dict[str, Any]:
    identity = _identity(strategy, period, stage="stage1", extra={"q": str(STAGE1_Q), "weights": COMMON_BASELINE_WEIGHTS, "rr": STAGE1_RR, "stops": STAGE1_STOP_TICKS})
    strategy_root = root / "strategies" / _safe_name(strategy.strategy_id)
    reused = _complete_or_raise(strategy_root, identity)
    if reused:
        return {**reused, "strategy_id": strategy.strategy_id, "status": "REUSED", "root": str(strategy_root)}
    rows_by_day, indexes = _load_population(period, strategy)
    provenance = _resolved_provenance(rows_by_day, strategy)
    provenance_sha256 = _sha(provenance)
    _write_json(strategy_root / "level-provenance.json", {
        "strategy_id": strategy.strategy_id, "period_id": period.period_id,
        "level_catalog": None if period.level_catalog is None else str(period.level_catalog),
        "level_catalog_sha256": None if period.level_catalog is None else _file_sha(period.level_catalog),
        "provenance_sha256": provenance_sha256, "resolutions": provenance,
    })
    rows: list[dict[str, Any]] = []
    tape_cache: dict[tuple[str, float, int], matrix.SessionCausalTape] = {}
    for rr in STAGE1_RR:
        for stop in STAGE1_STOP_TICKS:
            row, daily = _metrics(strategy=strategy, period=period, rows_by_day=rows_by_day, indexes=indexes,
                                  weights=COMMON_BASELINE_WEIGHTS, threshold=STAGE1_Q, rr=rr, stop_ticks=stop,
                                  tape_cache=tape_cache)
            rows.append({**row, "config_id": f"RR{rr:.1f}-S{stop}-Q50-W04"})
            _write_csv(strategy_root / "daily" / f"RR{rr:.1f}-S{stop}.csv", daily, ("date", "trades", "total_r"))
    rows = _stage1_neighbors(rows)
    selection = _select_stage1(rows)
    _write_csv(strategy_root / "stage1-matrix.csv", rows, ("config_id",))
    _write_json(strategy_root / "selection.json", selection)
    complete = {"stage": "stage1", "status": "COMPLETE", "input_identity": identity, "strategy_id": strategy.strategy_id,
                "configuration_count": len(rows), "selection": selection, "level_provenance_sha256": provenance_sha256,
                "evidence_label": EVIDENCE_LABEL}
    _write_json(strategy_root / "complete.json", complete)
    return {"strategy_id": strategy.strategy_id, "status": "COMPLETE", "root": str(strategy_root), **complete}


def run_stage1(*, strategies_path: Path, period_path: Path, output_root: Path, workers: int = 1) -> dict[str, Any]:
    strategies, period = load_strategy_manifest(strategies_path), load_period_manifest(period_path)
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    # Each strategy owns an isolated directory; strategy-level process parallelism is deterministic after sorting.
    args = [(spec, period, output_root) for spec in strategies]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_stage1_worker, args))
    else:
        results = [_run_stage1_worker(item) for item in args]
    results.sort(key=lambda item: item["strategy_id"])
    important_rows = _generate_stage1_summaries(strategies, output_root)
    _write_csv(output_root / "cross-strategy-stage1-summary.csv", results, ("strategy_id", "status"))
    summary = {"stage": "stage1", "status": "COMPLETE", "evidence_label": EVIDENCE_LABEL, "stage1_quality_threshold": str(STAGE1_Q),
               "common_baseline_weights": {key: str(value) for key, value in COMMON_BASELINE_WEIGHTS.items()},
               "baseline_derivation": BASELINE_DERIVATION, "matrix_per_strategy": len(STAGE1_RR) * len(STAGE1_STOP_TICKS),
               "important_summary_rows": len(important_rows), "strategies": results, "workers": workers}
    _write_json(output_root / "stage1-summary.json", summary)
    return summary


def _run_stage1_worker(args: tuple[StrategySpec, PeriodSpec, Path]) -> dict[str, Any]:
    strategy, period, root = args
    try:
        return _run_stage1_strategy(strategy, period, root)
    except Exception as exc:
        failure_root = root / "strategies" / _safe_name(strategy.strategy_id)
        _write_json(failure_root / "failure.json", {"stage": "stage1", "strategy_id": strategy.strategy_id, "error": str(exc)})
        return {"strategy_id": strategy.strategy_id, "status": "FAILED", "error": str(exc), "root": str(failure_root)}


def _stage2_selection(rows: Sequence[Mapping[str, Any]], combined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    neighbors = {str(row["config_id"]): row for row in combined}
    raw = max(rows, key=lambda item: (float(item["total_r"]), float(item["expectancy_r_per_session"]), str(item["config_id"])))
    robust = max(rows, key=lambda item: (
        float(neighbors[str(item["config_id"])]["proportion_neighbors_profitable"] or 0),
        _finite_sort(neighbors[str(item["config_id"])]["worst_neighbor_total_r"]),
        _finite_sort(neighbors[str(item["config_id"])]["median_neighbor_total_r"]),
        float(item["expectancy_r_per_session"]), float(item["total_r"]),
        float(item["max_cumulative_drawdown_r"]), _finite_sort(item["profit_factor"]), int(item["trades"]), str(item["config_id"]),
    ))
    return {"selection_rule": "descriptive only: rank neighbor profitability, worst and median combined legal-neighbor R, expectancy/session, total R, drawdown, profit factor, and trade count",
            "raw_best": {key: raw[key] for key in ("config_id", "rr", "stop_ticks", "G1", "G2", "G3", "G4", "G5", "quality_threshold")},
            "robust_best": {key: robust[key] for key in ("config_id", "rr", "stop_ticks", "G1", "G2", "G3", "G4", "G5", "quality_threshold")},
            "automatic_production_promotion": False}


def _stage2_config_id(units: Sequence[int], threshold: Decimal) -> str:
    # Reuse the canonical legal-weight validation, while keeping the new
    # development Q range independent of the frozen historical six-Q registry.
    matrix.unit_weights(units)
    if threshold not in STAGE2_Q:
        raise MultiStrategyResearchError(f"Stage 2 threshold not in declared grid: {threshold}")
    return "W" + "-".join(f"{int(value):02d}" for value in units) + f"-Q{int(threshold * 100):02d}"


def _stage2_neighbor_ids(units: Sequence[int], threshold: Decimal) -> tuple[tuple[str, ...], tuple[str, ...]]:
    weights = tuple(_stage2_config_id(item, threshold) for item in matrix.weight_neighbors(units))
    position = STAGE2_Q.index(threshold)
    quality = tuple(_stage2_config_id(units, STAGE2_Q[index]) for index in (position - 1, position + 1) if 0 <= index < len(STAGE2_Q))
    return weights, quality


def _stage2_robustness(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["config_id"]): row for row in rows}
    weight_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for row in rows:
        units = tuple(int(round(float(row[f"G{i}"]) / 0.05)) for i in range(1, 6))
        threshold = Decimal(str(row["quality_threshold"]))
        weight_ids, quality_ids = _stage2_neighbor_ids(units, threshold)
        missing = [item for item in (*weight_ids, *quality_ids) if item not in by_id]
        if missing:
            raise MultiStrategyResearchError(f"complete legal Stage 2 grid is missing neighbor: {missing[0]}")
        key = {name: row[name] for name in ("config_id", "G1", "G2", "G3", "G4", "G5", "quality_threshold")}
        weight_stats = matrix._neighbor_statistics([by_id[item] for item in weight_ids])
        quality_stats = matrix._neighbor_statistics([by_id[item] for item in quality_ids])
        combined_ids = tuple(sorted(set(weight_ids) | set(quality_ids)))
        combined_stats = matrix._neighbor_statistics([by_id[item] for item in combined_ids])
        weight_rows.append({**key, "immediate_weight_neighbor_count": len(weight_ids), **weight_stats})
        quality_rows.append({**key, "adjacent_quality_neighbor_count": len(quality_ids), **quality_stats})
        combined_rows.append({**key, "immediate_weight_neighbor_count": len(weight_ids),
                              "adjacent_quality_neighbor_count": len(quality_ids), "combined_neighbor_count": len(combined_ids), **combined_stats})
    return weight_rows, quality_rows, combined_rows


def _quality_sensitivity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for threshold in STAGE2_Q:
        group = [row for row in rows if Decimal(str(row["quality_threshold"])) == threshold]
        output.append({"quality_threshold": float(threshold), "configuration_count": len(group),
                       "median_total_r": sorted(float(row["total_r"]) for row in group)[len(group) // 2] if group else None,
                       "profitable_configuration_fraction": sum(float(row["total_r"]) > 0 for row in group) / len(group) if group else 0.0})
    return output


def _plateau(rows: Sequence[Mapping[str, Any]], combined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["config_id"]): row for row in rows}; neighbors = {str(row["config_id"]): row for row in combined}
    eligible = {identifier for identifier, row in by_id.items() if float(row["total_r"]) > 0 and int(row["trades"]) > 0 and float(neighbors[identifier]["proportion_neighbors_profitable"] or 0) >= 0.5}
    components: list[list[str]] = []
    while eligible:
        seed = min(eligible); eligible.remove(seed); stack, component = [seed], []
        while stack:
            item = stack.pop(); component.append(item); row = by_id[item]
            units = tuple(int(round(float(row[f"G{i}"]) / 0.05)) for i in range(1, 6))
            weight_ids, quality_ids = _stage2_neighbor_ids(units, Decimal(str(row["quality_threshold"])))
            adjacent = (set(weight_ids) | set(quality_ids)) & eligible
            eligible.difference_update(adjacent); stack.extend(sorted(adjacent, reverse=True))
        components.append(sorted(component))
    components.sort(key=lambda value: (-len(value), value[0]))
    return {"status": "DESCRIPTIVE_PLATEAU_ANALYSIS_NO_SELECTION", "positive_neighbor_majority_floor": 0.5,
            "plateau_count": len(components), "plateaus": [{"plateau_id": f"PLATEAU-{i:04d}", "configuration_count": len(values), "config_ids": values} for i, values in enumerate(components, 1)]}


def _stage2_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The existing robust research ordering, expressed for summary exports."""
    return (
        _number(row, "proportion_neighbors_profitable"), _finite_sort(row.get("worst_neighbor_total_r")),
        _finite_sort(row.get("median_neighbor_total_r")), _number(row, "expectancy_r_per_session"),
        _number(row, "total_r"), _number(row, "max_cumulative_drawdown_r"),
        _finite_sort(row.get("profit_factor")), _number(row, "trades"), str(row.get("config_id", "")),
    )


def _stage2_summary_row(
    row: Mapping[str, Any], *, strategy_id: str, rank: int, roles: Sequence[str], plateau_by_config: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    plateau = plateau_by_config.get(str(row["config_id"]), {})
    return {
        "rank": rank, "summary_roles": ";".join(sorted(set(roles))), "strategy_id": strategy_id,
        "config_id": row["config_id"], "G1": row.get("G1"), "G2": row.get("G2"), "G3": row.get("G3"),
        "G4": row.get("G4"), "G5": row.get("G5"), "Q": row.get("quality_threshold"),
        "quality_threshold": row.get("quality_threshold"), "frozen_rr": row.get("rr"),
        "frozen_stop_ticks": row.get("stop_ticks"), "trades": row.get("trades"),
        "sessions_evaluated": row.get("sessions_evaluated"), "trades_per_session": row.get("trades_per_session"),
        "total_r": row.get("total_r"), "expectancy_r_per_trade": row.get("expectancy_r_per_trade"),
        "expectancy_r_per_session": row.get("expectancy_r_per_session"),
        "max_cumulative_drawdown_r": row.get("max_cumulative_drawdown_r"), "profit_factor": row.get("profit_factor"),
        "win_rate": row.get("win_rate"),
        "neighbor_profitability": row.get("proportion_neighbors_profitable"),
        "worst_neighbor_result": row.get("worst_neighbor_total_r"),
        "median_neighbor_result": row.get("median_neighbor_total_r"),
        "combined_neighbor_count": row.get("combined_neighbor_count"),
        "robustness_score_neighbor_profitability": row.get("proportion_neighbors_profitable"),
        "plateau_id": plateau.get("plateau_id"), "plateau_configuration_count": plateau.get("configuration_count"),
    }


def _plateau_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for plateau in payload.get("plateaus", []):
        if not isinstance(plateau, Mapping):
            continue
        for identifier in plateau.get("config_ids", []):
            output[str(identifier)] = plateau
    return output


def _stage2_important_ids(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], plateau_by_config: Mapping[str, Mapping[str, Any]], *, limit: int = 25,
) -> dict[str, set[str]]:
    """Preserve raw/robust winners while showing their local and plateau context."""
    by_id = {str(row["config_id"]): row for row in rows}
    chosen: dict[str, set[str]] = {}

    def add(identifier: str, role: str) -> None:
        if identifier in by_id and len(chosen) < limit:
            chosen.setdefault(identifier, set()).add(role)
        elif identifier in chosen:
            chosen[identifier].add(role)

    raw_id = str(selection["raw_best"]["config_id"])
    robust_id = str(selection["robust_best"]["config_id"])
    add(raw_id, "RAW_BEST")
    add(robust_id, "ROBUST_BEST")
    rank = {str(row["config_id"]): index for index, row in enumerate(sorted(rows, key=_stage2_rank_key, reverse=True), 1)}
    for anchor, role in ((raw_id, "RAW_BEST_NEIGHBOR"), (robust_id, "ROBUST_BEST_NEIGHBOR")):
        row = by_id.get(anchor)
        if row is None:
            continue
        units = tuple(int(round(_number(row, f"G{i}") / 0.05)) for i in range(1, 6))
        thresholds = _stage2_neighbor_ids(units, Decimal(str(row["quality_threshold"])))
        for identifier in sorted(set((*thresholds[0], *thresholds[1])), key=lambda value: (rank.get(value, 10**9), value)):
            add(identifier, role)
    # Preserve a comparable best row from every Q region before using remaining
    # space for plateau members. This prevents a large local plateau from hiding
    # the quality-threshold sensitivity that the compact export is meant to show.
    for threshold in STAGE2_Q:
        candidates = [row for row in rows if Decimal(str(row["quality_threshold"])) == threshold]
        if candidates:
            add(str(max(candidates, key=_stage2_rank_key)["config_id"]), "QUALITY_REGION_REPRESENTATIVE")
    for anchor, role in ((raw_id, "RAW_BEST_PLATEAU"), (robust_id, "ROBUST_BEST_PLATEAU")):
        plateau = plateau_by_config.get(anchor)
        if plateau is not None:
            for identifier in sorted((str(item) for item in plateau.get("config_ids", [])), key=lambda value: (rank.get(value, 10**9), value)):
                add(identifier, role)
    for plateau_id in sorted({str(value.get("plateau_id")) for value in plateau_by_config.values()}):
        members = [identifier for identifier, item in plateau_by_config.items() if str(item.get("plateau_id")) == plateau_id]
        if members:
            add(min(members, key=lambda value: (rank.get(value, 10**9), value)), "PLATEAU_REPRESENTATIVE")
    for row in sorted(rows, key=_stage2_rank_key, reverse=True):
        add(str(row["config_id"]), "ROBUST_TOP")
    return chosen


def _generate_stage2_strategy_summaries(strategy: StrategySpec, strategy_root: Path) -> dict[str, Any]:
    """Regenerate compact exports from complete result files without simulation."""
    rows = _read_csv_rows(strategy_root / "weight-q-results.csv")
    combined = {str(row["config_id"]): row for row in _read_csv_rows(strategy_root / "combined-neighbors.csv")}
    if set(combined) != {str(row["config_id"]) for row in rows}:
        raise MultiStrategyResearchError(f"Stage 2 neighbor summary does not match full results: {strategy.strategy_id}")
    merged = [{**row, **combined[str(row["config_id"])]} for row in rows]
    selection = json.loads((strategy_root / "selection.json").read_text(encoding="utf-8"))
    plateau_by_config = _plateau_index(json.loads((strategy_root / "plateau-analysis.json").read_text(encoding="utf-8")))
    ranked = sorted(merged, key=_stage2_rank_key, reverse=True)
    by_id = {str(row["config_id"]): row for row in merged}
    top = [_stage2_summary_row(row, strategy_id=strategy.strategy_id, rank=index, roles=("TOP1000_ROBUST",), plateau_by_config=plateau_by_config)
           for index, row in enumerate(ranked[:1000], 1)]
    important_ids = _stage2_important_ids(merged, selection, plateau_by_config)
    important = [_stage2_summary_row(row, strategy_id=strategy.strategy_id, rank=index,
                                     roles=tuple(important_ids[str(row["config_id"])]), plateau_by_config=plateau_by_config)
                 for index, row in enumerate(sorted((by_id[identifier] for identifier in important_ids),
                                                     key=_stage2_rank_key, reverse=True), 1)]
    _write_csv(strategy_root / "top-1000-configurations.csv", top, ("rank", "strategy_id", "config_id"))
    _write_csv(strategy_root / "important-summary.csv", important, ("rank", "strategy_id", "config_id"))
    return {"strategy_id": strategy.strategy_id, "root": str(strategy_root), "full_configuration_count": len(rows),
            "top_configuration_count": len(top), "important_configuration_count": len(important),
            "raw_best": _stage2_summary_row(by_id[str(selection["raw_best"]["config_id"])], strategy_id=strategy.strategy_id,
                                               rank=0, roles=("RAW_BEST",), plateau_by_config=plateau_by_config),
            "robust_best": _stage2_summary_row(by_id[str(selection["robust_best"]["config_id"])], strategy_id=strategy.strategy_id,
                                                  rank=0, roles=("ROBUST_BEST",), plateau_by_config=plateau_by_config),
            "important": important}


def _run_stage2_strategy(strategy: StrategySpec, period: PeriodSpec, root: Path, stage1_root: Path, *, grid: Sequence[tuple[int, int, int, int, int]] | None = None) -> dict[str, Any]:
    stage1_dir = stage1_root / "strategies" / _safe_name(strategy.strategy_id)
    selection_path = stage1_dir / "selection.json"
    if not selection_path.is_file():
        raise MultiStrategyResearchError(f"missing Stage 1 selection: {strategy.strategy_id}")
    execution = json.loads(selection_path.read_text(encoding="utf-8"))["stage2_execution_configuration"]
    stage1_complete_path = stage1_dir / "complete.json"
    if not stage1_complete_path.is_file():
        raise MultiStrategyResearchError(f"missing Stage 1 completion marker: {strategy.strategy_id}")
    stage1_complete = json.loads(stage1_complete_path.read_text(encoding="utf-8"))
    weights_grid = tuple(grid or matrix.generate_weight_grid())
    identity = _identity(strategy, period, stage="stage2", extra={"stage1_selection": execution, "q": [str(value) for value in STAGE2_Q], "weight_grid_sha": _sha(weights_grid)})
    strategy_root = root / "strategies" / _safe_name(strategy.strategy_id)
    reused = _complete_or_raise(strategy_root, identity)
    if reused:
        summary = _generate_stage2_strategy_summaries(strategy, strategy_root)
        return {**reused, "strategy_id": strategy.strategy_id, "status": "REUSED", "root": str(strategy_root),
                "important_summary_rows": summary["important_configuration_count"], "summary_regenerated": True}
    progress_path = strategy_root / "in-progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_identity") != identity:
            raise MultiStrategyResearchError(f"interrupted Stage 2 identity differs: {strategy.strategy_id}")
    else:
        _write_json(progress_path, {"stage": "stage2", "input_identity": identity, "status": "IN_PROGRESS"})
    rows_by_day, indexes = _load_population(period, strategy)
    provenance = _resolved_provenance(rows_by_day, strategy)
    provenance_sha256 = _sha(provenance)
    if stage1_complete.get("level_provenance_sha256") != provenance_sha256:
        raise MultiStrategyResearchError(f"Stage 2 level provenance differs from Stage 1: {strategy.strategy_id}")
    results: list[dict[str, Any]] = []
    rr, stop = float(execution["rr"]), int(execution["stop_ticks"])
    tape_cache: dict[tuple[str, float, int], matrix.SessionCausalTape] = {}
    for units in weights_grid:
        chunk_id = "-".join(f"{value:02d}" for value in units)
        chunk_path = strategy_root / "weight-q-checkpoints" / f"W{chunk_id}.json"
        if chunk_path.is_file():
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            if chunk.get("input_identity") != identity or tuple(chunk.get("units", ())) != tuple(units):
                raise MultiStrategyResearchError(f"invalid Stage 2 checkpoint: {chunk_path}")
            results.extend(dict(row) for row in chunk.get("rows", ()))
            continue
        weights = matrix.unit_weights(units)
        batch: list[dict[str, Any]] = []
        for threshold in STAGE2_Q:
            row, _daily = _metrics(strategy=strategy, period=period, rows_by_day=rows_by_day, indexes=indexes,
                                   weights=weights, threshold=threshold, rr=rr, stop_ticks=stop, tape_cache=tape_cache)
            batch.append({**row, "config_id": _stage2_config_id(units, threshold),
                          **{f"G{i}": float(weights[name]) for i, name in enumerate(SCORE_FIELDS, 1)}})
        _write_json(chunk_path, {"input_identity": identity, "units": list(units), "rows": batch})
        results.extend(batch)
    results.sort(key=lambda row: str(row["config_id"]))
    weight_neighbors, quality_neighbors, combined = _stage2_robustness(results)
    sensitivity, plateau = _quality_sensitivity(results), _plateau(results, combined)
    selection = _stage2_selection(results, combined)
    selected = next(row for row in results if row["config_id"] == selection["robust_best"]["config_id"])
    selected_weights = {name: Decimal(str(selected[f"G{i}"])) for i, name in enumerate(SCORE_FIELDS, 1)}
    _selected_metrics, daily = _metrics(strategy=strategy, period=period, rows_by_day=rows_by_day, indexes=indexes,
                                        weights=selected_weights, threshold=Decimal(str(selected["quality_threshold"])), rr=rr, stop_ticks=stop,
                                        tape_cache=tape_cache)
    _write_csv(strategy_root / "weight-q-results.csv", results, ("config_id",))
    _write_csv(strategy_root / "period-metrics.csv", results, ("config_id",))
    _write_csv(strategy_root / "weight-neighbors.csv", weight_neighbors, ("config_id",))
    _write_csv(strategy_root / "quality-neighbors.csv", quality_neighbors, ("config_id",))
    _write_csv(strategy_root / "combined-neighbors.csv", combined, ("config_id",))
    _write_csv(strategy_root / "quality-threshold-sensitivity.csv", sensitivity, ("quality_threshold",))
    _write_csv(strategy_root / "robust-best-daily-results.csv", daily, ("date",))
    _write_json(strategy_root / "selection.json", selection); _write_json(strategy_root / "plateau-analysis.json", plateau)
    summary = _generate_stage2_strategy_summaries(strategy, strategy_root)
    complete = {"stage": "stage2", "status": "COMPLETE", "input_identity": identity, "strategy_id": strategy.strategy_id,
                "configuration_count": len(results), "rr": rr, "stop_ticks": stop, "selection": selection,
                "level_provenance_sha256": provenance_sha256, "summary_configuration_count": summary["important_configuration_count"],
                "evidence_label": EVIDENCE_LABEL}
    _write_json(strategy_root / "complete.json", complete)
    return {**complete, "strategy_id": strategy.strategy_id, "status": "COMPLETE", "root": str(strategy_root),
            "important_summary_rows": summary["important_configuration_count"]}


def _run_stage2_worker(args: tuple[StrategySpec, PeriodSpec, Path, Path]) -> dict[str, Any]:
    strategy, period, root, stage1_root = args
    try:
        return _run_stage2_strategy(strategy, period, root, stage1_root)
    except Exception as exc:
        failure_root = root / "strategies" / _safe_name(strategy.strategy_id)
        _write_json(failure_root / "failure.json", {"stage": "stage2", "strategy_id": strategy.strategy_id, "error": str(exc)})
        return {"strategy_id": strategy.strategy_id, "status": "FAILED", "error": str(exc), "root": str(failure_root)}


def _period_input_hashes(period: PeriodSpec) -> dict[str, Any]:
    return {
        "interaction_master": {"path": str(period.interaction_master), "sha256": _file_sha(period.interaction_master)},
        "interaction_index": {"path": str(period.interaction_index), "sha256": _file_sha(period.interaction_index)},
        "event_tapes": [{"date": day, "path": str(path), "sha256": _file_sha(path)} for day, path in period.sessions],
        "level_catalog": None if period.level_catalog is None else {"path": str(period.level_catalog), "sha256": _file_sha(period.level_catalog)},
    }


def _generate_research_summaries(
    strategies: Sequence[StrategySpec], period: PeriodSpec, stage1_root: Path, output_root: Path,
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build global compact summaries from completed strategy result files only."""
    status_by_id = {str(row["strategy_id"]): dict(row) for row in statuses}
    important_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    strategy_json: list[dict[str, Any]] = []
    for strategy in sorted(strategies, key=lambda item: item.strategy_id):
        strategy_root = output_root / "strategies" / _safe_name(strategy.strategy_id)
        stage1_dir = stage1_root / "strategies" / _safe_name(strategy.strategy_id)
        status = status_by_id.get(strategy.strategy_id, {"strategy_id": strategy.strategy_id, "status": "NOT_RUN"})
        if not (strategy_root / "complete.json").is_file() or not (stage1_dir / "selection.json").is_file():
            strategy_rows.append({"strategy_id": strategy.strategy_id, "completion_status": status.get("status"),
                                  "output_directory": str(strategy_root), "missing_result_files": True})
            strategy_json.append({"strategy_id": strategy.strategy_id, "completion_status": status.get("status"),
                                  "output_directory": str(strategy_root), "missing_result_files": True})
            continue
        summary = _generate_stage2_strategy_summaries(strategy, strategy_root)
        stage1_selection = json.loads((stage1_dir / "selection.json").read_text(encoding="utf-8"))
        raw, robust = summary["raw_best"], summary["robust_best"]
        for row in summary["important"]:
            important_rows.append({**row, "output_directory": str(strategy_root)})
        strategy_row = {
            "strategy_id": strategy.strategy_id,
            "stage1_selected_rr": stage1_selection["stage2_execution_configuration"]["rr"],
            "stage1_selected_stop_ticks": stage1_selection["stage2_execution_configuration"]["stop_ticks"],
            "raw_best_G1": raw["G1"], "raw_best_G2": raw["G2"], "raw_best_G3": raw["G3"],
            "raw_best_G4": raw["G4"], "raw_best_G5": raw["G5"], "raw_best_Q": raw["Q"],
            "raw_best_total_r": raw["total_r"], "raw_best_expectancy_r_per_session": raw["expectancy_r_per_session"],
            "raw_best_max_cumulative_drawdown_r": raw["max_cumulative_drawdown_r"],
            "robust_best_G1": robust["G1"], "robust_best_G2": robust["G2"], "robust_best_G3": robust["G3"],
            "robust_best_G4": robust["G4"], "robust_best_G5": robust["G5"], "robust_best_Q": robust["Q"],
            "robust_best_total_r": robust["total_r"], "robust_best_expectancy_r_per_session": robust["expectancy_r_per_session"],
            "robust_best_max_cumulative_drawdown_r": robust["max_cumulative_drawdown_r"],
            "trade_count": robust["trades"], "sessions": robust["sessions_evaluated"],
            "profit_factor": robust["profit_factor"], "win_rate": robust["win_rate"],
            "robustness_neighbor_profitability": robust["neighbor_profitability"],
            "robustness_worst_neighbor_result": robust["worst_neighbor_result"],
            "robustness_median_neighbor_result": robust["median_neighbor_result"],
            "plateau_id": robust["plateau_id"], "plateau_configuration_count": robust["plateau_configuration_count"],
            "output_directory": str(strategy_root), "completion_status": status.get("status"),
        }
        strategy_rows.append(strategy_row)
        strategy_json.append({
            "strategy_id": strategy.strategy_id, "strategy": strategy.__dict__, "completion_status": status.get("status"),
            "output_directory": str(strategy_root), "stage1_selection": stage1_selection,
            "raw_best": raw, "robust_best": robust, "important_candidates": summary["important"],
        })
    important_rows.sort(key=lambda row: (str(row["strategy_id"]), int(row["rank"]), str(row["config_id"])))
    strategy_rows.sort(key=lambda row: str(row["strategy_id"]))
    _write_csv(output_root / "research-important-summary.csv", important_rows, ("rank", "strategy_id", "config_id"))
    _write_csv(output_root / "research-strategy-summary.csv", strategy_rows, ("strategy_id", "output_directory"))
    hashes = _period_input_hashes(period)
    run_identity = _sha({"period_id": period.period_id, "strategies": [item.__dict__ for item in strategies], "input_artifact_hashes": hashes})
    payload = {
        "run_identity": run_identity, "period_id": period.period_id,
        "strategy_list": [item.strategy_id for item in sorted(strategies, key=lambda item: item.strategy_id)],
        "session_dates": [day for day, _path in period.sessions],
        "sessions": [{"date": day, "event_tape": str(path)} for day, path in period.sessions],
        "input_artifact_hashes": hashes,
        "strategies": strategy_json,
        "completion_resume_status": [{"strategy_id": item.strategy_id, "status": status_by_id.get(item.strategy_id, {}).get("status", "NOT_RUN")}
                                     for item in sorted(strategies, key=lambda item: item.strategy_id)],
        "all_configurations_embedded": False,
        "important_summary_row_count": len(important_rows),
    }
    _write_json(output_root / "research-summary.json", payload)
    return {"run_identity": run_identity, "important_summary_rows": len(important_rows),
            "strategy_summary_rows": len(strategy_rows), "strategies": strategy_json}


def run_stage2(*, strategies_path: Path, period_path: Path, stage1_root: Path, output_root: Path, workers: int = 1) -> dict[str, Any]:
    strategies, period = load_strategy_manifest(strategies_path), load_period_manifest(period_path)
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    args = [(spec, period, output_root, stage1_root.resolve()) for spec in strategies]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_stage2_worker, args))
    else:
        results = [_run_stage2_worker(item) for item in args]
    results.sort(key=lambda item: item["strategy_id"])
    global_summaries = _generate_research_summaries(strategies, period, stage1_root.resolve(), output_root, results)
    _write_csv(output_root / "cross-strategy-stage2-summary.csv", results, ("strategy_id", "status"))
    summary = {"stage": "stage2", "status": "COMPLETE", "evidence_label": EVIDENCE_LABEL,
               "quality_thresholds": [str(value) for value in STAGE2_Q], "weight_count": len(matrix.generate_weight_grid()),
               "configuration_count_per_strategy": len(matrix.generate_weight_grid()) * len(STAGE2_Q), "strategies": results, "workers": workers,
               "global_summary": {key: value for key, value in global_summaries.items() if key != "strategies"},
               "automatic_production_promotion": False}
    _write_json(output_root / "stage2-summary.json", summary)
    return summary


STAGE3_STARTING_BALANCE_USD = 50_000.00
STAGE3_PORTFOLIO_TYPE = "AGGREGATED_RESEARCH_PORTFOLIO"
STAGE3_PORTFOLIO_WARNING = "NOT_A_SIMULTANEOUS_CAPITAL_CONSTRAINED_PORTFOLIO_BACKTEST"


def _stage3_selection_key(selection: str) -> str:
    normalized = str(selection).strip().lower()
    if normalized not in {"robust-best", "raw-best"}:
        raise MultiStrategyResearchError("Stage 3 selection must be robust-best or raw-best")
    return normalized


def _stage3_selected_config(
    strategy: StrategySpec, *, stage1_root: Path, stage2_root: Path, selection: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    stage1_dir = stage1_root / "strategies" / _safe_name(strategy.strategy_id)
    stage2_dir = stage2_root / "strategies" / _safe_name(strategy.strategy_id)
    if not (stage1_dir / "complete.json").is_file() or not (stage2_dir / "complete.json").is_file():
        raise MultiStrategyResearchError(f"Stage 3 requires completed Stage 1 and 2 results: {strategy.strategy_id}")
    stage1 = json.loads((stage1_dir / "selection.json").read_text(encoding="utf-8"))
    stage2_marker = json.loads((stage2_dir / "complete.json").read_text(encoding="utf-8"))
    stage2_selection = json.loads((stage2_dir / "selection.json").read_text(encoding="utf-8"))
    selected = dict(stage2_selection["robust_best" if selection == "robust-best" else "raw_best"])
    frozen_geometry = stage1["stage2_execution_configuration"]
    if (float(selected["rr"]), int(selected["stop_ticks"])) != (
        float(frozen_geometry["rr"]), int(frozen_geometry["stop_ticks"])
    ):
        raise MultiStrategyResearchError(f"Stage 2 geometry no longer matches Stage 1: {strategy.strategy_id}")
    result = next((row for row in _read_csv_rows(stage2_dir / "weight-q-results.csv")
                   if str(row["config_id"]) == str(selected["config_id"])), None)
    if result is None:
        raise MultiStrategyResearchError(f"Stage 2 selected configuration is absent from full result: {strategy.strategy_id}")
    for name in ("rr", "G1", "G2", "G3", "G4", "G5", "quality_threshold"):
        if Decimal(str(result[name])) != Decimal(str(selected[name])):
            raise MultiStrategyResearchError(f"Stage 2 selected configuration mismatch for {strategy.strategy_id}/{name}")
    if int(result["stop_ticks"]) != int(selected["stop_ticks"]):
        raise MultiStrategyResearchError(f"Stage 2 selected configuration mismatch for {strategy.strategy_id}/stop_ticks")
    config_identity = _sha({"stage2_input_identity": stage2_marker["input_identity"], "selection": selection, "configuration": selected})
    return selected, stage2_marker, config_identity


def _stage3_replay_strategy(
    *, strategy: StrategySpec, period: PeriodSpec, selected: Mapping[str, Any], config_identity: str, selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Replay exactly one persisted Stage-2 configuration using causal Parquet tapes only."""
    weights = {name: Decimal(str(selected[f"G{index}"])) for index, name in enumerate(SCORE_FIELDS, 1)}
    threshold, rr, stop_ticks = Decimal(str(selected["quality_threshold"])), float(selected["rr"]), int(selected["stop_ticks"])
    rows_by_day, indexes = _load_population(period, strategy)
    provenance_sha256 = _sha(_resolved_provenance(rows_by_day, strategy))
    trades: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    for day, tape_path in period.sessions:
        candidates = _accepted(rows_by_day[day], weights, threshold)
        candidate_indexes = {str(row["interaction_id"]): indexes[str(row["interaction_id"])] for row in candidates}
        tape = matrix.SessionCausalTape.from_parquet(day, tape_path, stop_buffer_ticks=stop_ticks, target_r=rr)
        session = matrix.simulate_independent_session(tape, candidates, candidate_indexes)
        by_source = {str(row["source_interaction_id"]): row for row in candidates}
        daily.append({
            "strategy_id": strategy.strategy_id, "trading_date": day, "sessions": 1,
            "unresolved_trades": session.unresolved, "accepted_setups": session.accepted_setups,
            "confirmation_expiries": session.confirmation_expiries,
            "active_position_blocks": session.active_position_blocks,
            "other_terminal_count": sum(session.other_terminal.values()),
        })
        for trade in session.trades:
            interaction = by_source.get(str(trade["interaction_id"]))
            if interaction is None:
                raise MultiStrategyResearchError(f"Stage 3 trade lacks its accepted interaction: {strategy.strategy_id}")
            resolution = interaction.get("level_resolution")
            trades.append({
                "strategy_id": strategy.strategy_id, "trading_date": day,
                "target_session": strategy.session, "source_session": strategy.source_session,
                "reference_level": strategy.reference_level,
                "reference_level_price": interaction.get("level_price"), "side": trade["direction"],
                "signal_timestamp": _signal_timestamp(interaction), "entry_timestamp": trade["entry_timestamp_ns"],
                "exit_timestamp": trade["exit_timestamp_ns"], "q_score": float(master.recompute_quality(interaction, weights)),
                "q_threshold": str(threshold),
                **{f"G{index}": str(weights[name]) for index, name in enumerate(SCORE_FIELDS, 1)},
                **{f"W{index}": str(weights[name]) for index, name in enumerate(SCORE_FIELDS, 1)},
                "rr": rr, "stop_ticks": stop_ticks, "instrument": trade["instrument"], "quantity": trade["contracts"],
                "entry_price": trade["entry"], "stop_price": trade["stop"], "target_price": trade["target"],
                "exit_price": trade["exit"], "exit_reason": trade["exit_reason"],
                "result_r": trade["r_multiple"], "pnl_usd": trade["net_pnl_usd"],
                "configuration_selection_type": selection, "stage2_config_id": selected["config_id"],
                "stage2_config_identity": config_identity,
                "source_tape_session_identity": f"{day}:{_file_sha(tape_path)}",
                "causal_level_provenance_identity": _sha(resolution) if isinstance(resolution, Mapping) else None,
                "canonical_trade_id": trade["trade_id"],
            })
    return trades, daily, provenance_sha256


def _stage3_entry_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(row["entry_timestamp"]), int(row["exit_timestamp"]), str(row["strategy_id"]),
            int(row.get("trade_number_global_for_strategy", 0)), str(row["canonical_trade_id"]))


def _stage3_realization_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(row["exit_timestamp"]), int(row["entry_timestamp"]), str(row["strategy_id"]),
            int(row["trade_number_global_for_strategy"]), str(row["canonical_trade_id"]))


def _stage3_apply_equity(rows: Sequence[dict[str, Any]], *, prefix: str, starting_balance: float) -> None:
    balance = peak = float(starting_balance)
    cumulative_pnl = cumulative_r = 0.0
    for row in rows:
        pnl, result_r = float(row["pnl_usd"]), float(row["result_r"] or 0.0)
        row[f"{prefix}_starting_balance_usd"] = starting_balance
        row[f"{prefix}_balance_before_trade"] = balance
        balance += pnl; cumulative_pnl += pnl; cumulative_r += result_r; peak = max(peak, balance)
        drawdown = peak - balance
        row[f"{prefix}_balance_after_trade"] = balance
        row[f"{prefix}_peak_balance"] = peak
        row[f"{prefix}_drawdown_usd"] = drawdown
        row[f"{prefix}_drawdown_pct"] = (drawdown / peak * 100.0) if peak else 0.0
        row[f"{prefix}_cumulative_pnl_usd"] = cumulative_pnl
        row[f"{prefix}_cumulative_r"] = cumulative_r


def _stage3_daily_rows(
    strategy: StrategySpec, journal: Sequence[Mapping[str, Any]], daily_inputs: Sequence[Mapping[str, Any]],
    *, starting_balance: float = STAGE3_STARTING_BALANCE_USD,
) -> list[dict[str, Any]]:
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in journal:
        by_day[str(row["trading_date"])].append(row)
    output: list[dict[str, Any]] = []
    balance = peak = float(starting_balance)
    for item in sorted(daily_inputs, key=lambda value: str(value["trading_date"])):
        day, rows = str(item["trading_date"]), sorted(by_day.get(str(item["trading_date"]), []), key=_stage3_entry_key)
        values = [float(row["pnl_usd"]) for row in rows]
        start = float(rows[0]["strategy_balance_before_trade"]) if rows else balance
        end = float(rows[-1]["strategy_balance_after_trade"]) if rows else balance
        latest = rows[-1] if rows else None
        peak = float(latest["strategy_peak_balance"]) if latest is not None else peak
        drawdown = peak - end
        output.append({
            "strategy_id": strategy.strategy_id, "trading_date": day, "trades": len(rows),
            "wins": sum(value > 0 for value in values), "losses": sum(value < 0 for value in values),
            "breakeven": sum(value == 0 for value in values), "total_r": sum(float(row["result_r"] or 0.0) for row in rows),
            "pnl_usd": sum(values), "strategy_starting_balance_for_day": start,
            "strategy_ending_balance_for_day": end,
            "strategy_peak_balance_to_date": peak,
            "strategy_drawdown_usd_to_date": drawdown,
            "strategy_drawdown_pct_to_date": (drawdown / peak * 100.0) if peak else 0.0,
            "win_rate": sum(value > 0 for value in values) / len(values) if values else 0.0,
            "target_exits": sum(row["exit_reason"] == "TARGET" for row in rows),
            "stop_exits": sum(row["exit_reason"] == "STOP" for row in rows),
            "hard_flat_exits": sum(str(row["exit_reason"]).startswith("HARD_") for row in rows),
            "unresolved_trades": item["unresolved_trades"],
        })
        balance = end
    return output


def _stage3_strategy_summary(
    *, strategy: StrategySpec, selected: Mapping[str, Any], journal: Sequence[Mapping[str, Any]],
    daily_rows: Sequence[Mapping[str, Any]], starting_balance: float, selection: str,
) -> dict[str, Any]:
    ordered = sorted(journal, key=_stage3_entry_key)
    pnl = [float(row["pnl_usd"]) for row in ordered]
    gross_profit, gross_loss = sum(value for value in pnl if value > 0), -sum(value for value in pnl if value < 0)
    ending = float(ordered[-1]["strategy_balance_after_trade"]) if ordered else starting_balance
    max_dd_usd = max((float(row["strategy_drawdown_usd"]) for row in ordered), default=0.0)
    max_dd_pct = max((float(row["strategy_drawdown_pct"]) for row in ordered), default=0.0)
    return {
        "strategy_id": strategy.strategy_id, "selection_type": selection, "RR": selected["rr"],
        "stop_ticks": selected["stop_ticks"], **{f"G{index}": selected[f"G{index}"] for index in range(1, 6)},
        "Q": selected["quality_threshold"], "sessions": len(daily_rows), "trades": len(ordered),
        "wins": sum(value > 0 for value in pnl), "losses": sum(value < 0 for value in pnl),
        "win_rate": sum(value > 0 for value in pnl) / len(pnl) if pnl else 0.0,
        "total_r": sum(float(row["result_r"] or 0.0) for row in ordered),
        "expectancy_r_per_trade": sum(float(row["result_r"] or 0.0) for row in ordered) / len(ordered) if ordered else 0.0,
        "expectancy_r_per_session": sum(float(row["result_r"] or 0.0) for row in ordered) / len(daily_rows) if daily_rows else 0.0,
        "total_pnl_usd": sum(pnl), "starting_balance_usd": starting_balance, "ending_balance_usd": ending,
        "return_pct": (ending - starting_balance) / starting_balance * 100.0,
        "max_drawdown_usd": max_dd_usd, "max_drawdown_pct": max_dd_pct,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "target_exits": sum(row["exit_reason"] == "TARGET" for row in ordered),
        "stop_exits": sum(row["exit_reason"] == "STOP" for row in ordered),
        "hard_flat_exits": sum(str(row["exit_reason"]).startswith("HARD_") for row in ordered),
        "unresolved_trades": sum(int(row["unresolved_trades"]) for row in daily_rows),
    }


def _run_stage3_strategy(
    strategy: StrategySpec, period: PeriodSpec, root: Path, stage1_root: Path, stage2_root: Path,
    *, selection: str, starting_balance: float,
) -> dict[str, Any]:
    selected, stage2_marker, config_identity = _stage3_selected_config(
        strategy, stage1_root=stage1_root, stage2_root=stage2_root, selection=selection,
    )
    identity = _identity(strategy, period, stage="stage3", extra={
        "selection": selection, "starting_balance_usd": starting_balance,
        "stage2_config_identity": config_identity, "stage2_input_identity": stage2_marker["input_identity"],
    })
    strategy_root = root / "strategies" / _safe_name(strategy.strategy_id)
    reused = _complete_or_raise(strategy_root, identity)
    if reused:
        return {**reused, "strategy_id": strategy.strategy_id, "status": "REUSED", "root": str(strategy_root)}
    trades, daily, provenance_sha256 = _stage3_replay_strategy(
        strategy=strategy, period=period, selected=selected, config_identity=config_identity, selection=selection,
    )
    _write_csv(strategy_root / "trades.csv", trades, ("strategy_id", "canonical_trade_id"))
    _write_json(strategy_root / "daily-input.json", daily)
    complete = {
        "stage": "stage3", "status": "COMPLETE", "input_identity": identity, "strategy_id": strategy.strategy_id,
        "selection_type": selection, "starting_balance_usd": starting_balance, "selected_configuration": selected,
        "stage2_config_identity": config_identity, "stage2_input_identity": stage2_marker["input_identity"],
        "trade_count": len(trades), "daily_inputs": daily, "level_provenance_sha256": provenance_sha256,
        "evidence_label": EVIDENCE_LABEL,
    }
    _write_json(strategy_root / "complete.json", complete)
    return {**complete, "strategy_id": strategy.strategy_id, "status": "COMPLETE", "root": str(strategy_root)}


def _run_stage3_worker(args: tuple[StrategySpec, PeriodSpec, Path, Path, Path, str, float]) -> dict[str, Any]:
    strategy, period, root, stage1_root, stage2_root, selection, starting_balance = args
    try:
        return _run_stage3_strategy(strategy, period, root, stage1_root, stage2_root,
                                    selection=selection, starting_balance=starting_balance)
    except Exception as exc:
        failure_root = root / "strategies" / _safe_name(strategy.strategy_id)
        _write_json(failure_root / "failure.json", {"stage": "stage3", "strategy_id": strategy.strategy_id, "error": str(exc)})
        return {"strategy_id": strategy.strategy_id, "status": "FAILED", "error": str(exc), "root": str(failure_root)}


def _write_stage3_markdown(path: Path, journal: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> None:
    lines = ["# Stage 3 trade journal", "", f"Portfolio type: `{STAGE3_PORTFOLIO_TYPE}`.",
             f"Warning: `{STAGE3_PORTFOLIO_WARNING}`.", ""]
    by_strategy: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in journal:
        by_strategy[str(row["strategy_id"])][str(row["trading_date"])].append(row)
    for strategy_id in sorted(by_strategy):
        lines.extend([f"# {strategy_id}", ""])
        for day in sorted(by_strategy[strategy_id]):
            lines.extend([f"## {day}", ""])
            for row in sorted(by_strategy[strategy_id][day], key=_stage3_entry_key):
                lines.extend([
                    f"Trade {row['trade_number_global_for_strategy']}", f"- {row['side']}",
                    f"- Entry: {row['entry_price']}", f"- Exit: {row['exit_price']} ({row['exit_reason']})",
                    f"- Result: {float(row['result_r'] or 0.0):+.6g}R", f"- PnL: ${float(row['pnl_usd']):+.2f}",
                    f"- Strategy balance: ${float(row['strategy_balance_before_trade']):,.2f} -> ${float(row['strategy_balance_after_trade']):,.2f}",
                    f"- Overall balance after realization: ${float(row['overall_balance_after_trade']):,.2f}", "",
                ])
    lines.extend([
        "# Overall Aggregated Research Portfolio", "", f"- Start: ${float(overall['starting_balance_usd']):,.2f}",
        f"- End: ${float(overall['ending_balance_usd']):,.2f}", f"- PnL: ${float(overall['total_pnl_usd']):+.2f}",
        f"- Max DD: ${float(overall['max_drawdown_usd']):,.2f}", f"- Total trades: {overall['total_trades']}", "",
        "This is aggregated research accounting, not a capital-constrained simultaneous portfolio simulation.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _generate_stage3_global(
    strategies: Sequence[StrategySpec], period: PeriodSpec, output_root: Path, statuses: Sequence[Mapping[str, Any]],
    *, selection: str, starting_balance: float,
) -> dict[str, Any]:
    status_by_id = {str(item["strategy_id"]): dict(item) for item in statuses}
    all_rows: list[dict[str, Any]] = []
    strategy_summaries: list[dict[str, Any]] = []
    strategy_payloads: list[dict[str, Any]] = []
    for strategy in sorted(strategies, key=lambda item: item.strategy_id):
        root = output_root / "strategies" / _safe_name(strategy.strategy_id)
        if not (root / "complete.json").is_file() or not (root / "trades.csv").is_file():
            status = status_by_id.get(strategy.strategy_id, {}).get("status", "NOT_RUN")
            strategy_summaries.append({"strategy_id": strategy.strategy_id, "selection_type": selection,
                                       "completion_status": status, "missing_result_files": True})
            strategy_payloads.append({"strategy_id": strategy.strategy_id, "status": status,
                                      "output_directory": str(root), "missing_result_files": True})
            continue
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        rows = _read_csv_rows(root / "trades.csv")
        rows.sort(key=_stage3_entry_key)
        for number, row in enumerate(rows, 1):
            row["trade_number_global_for_strategy"] = number
            row["trade_number_for_day"] = sum(1 for prior in rows[:number] if str(prior["trading_date"]) == str(row["trading_date"]))
        _stage3_apply_equity(rows, prefix="strategy", starting_balance=starting_balance)
        daily_rows = _stage3_daily_rows(strategy, rows, complete["daily_inputs"], starting_balance=starting_balance)
        summary = _stage3_strategy_summary(strategy=strategy, selected=complete["selected_configuration"], journal=rows,
                                           daily_rows=daily_rows, starting_balance=starting_balance, selection=selection)
        strategy_summaries.append(summary)
        strategy_payloads.append({"strategy_id": strategy.strategy_id, "status": status_by_id.get(strategy.strategy_id, {}).get("status", "NOT_RUN"),
                                  "selected_configuration": complete["selected_configuration"], "stage2_config_identity": complete["stage2_config_identity"],
                                  "summary": summary, "output_directory": str(root)})
        all_rows.extend(rows)
    entry_rows = sorted(all_rows, key=_stage3_entry_key)
    for sequence, row in enumerate(entry_rows, 1):
        row["global_trade_sequence"] = sequence
    realized_rows = sorted(entry_rows, key=_stage3_realization_key)
    for sequence, row in enumerate(realized_rows, 1):
        row["overall_realization_sequence"] = sequence
    _stage3_apply_equity(realized_rows, prefix="overall", starting_balance=starting_balance)
    daily_by_strategy: list[dict[str, Any]] = []
    for strategy in strategies:
        root = output_root / "strategies" / _safe_name(strategy.strategy_id)
        rows = [row for row in entry_rows if row["strategy_id"] == strategy.strategy_id]
        if not (root / "complete.json").is_file():
            continue
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        daily = _stage3_daily_rows(strategy, rows, complete["daily_inputs"], starting_balance=starting_balance)
        _write_csv(root / "trades.csv", rows, ("strategy_id", "canonical_trade_id"))
        _write_csv(root / "daily-summary.csv", daily, ("strategy_id", "trading_date"))
        daily_by_strategy.extend(daily)
    overall_daily: list[dict[str, Any]] = []
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in realized_rows:
        by_day[str(row["trading_date"])].append(row)
    overall_balance = overall_peak = float(starting_balance)
    for day, _path in period.sessions:
        rows = sorted(by_day[day], key=_stage3_realization_key); pnl = [float(row["pnl_usd"]) for row in rows]
        latest = rows[-1] if rows else None
        start = float(rows[0]["overall_balance_before_trade"]) if rows else overall_balance
        end = float(latest["overall_balance_after_trade"]) if latest is not None else overall_balance
        overall_peak = float(latest["overall_peak_balance"]) if latest is not None else overall_peak
        drawdown = overall_peak - end
        overall_daily.append({
            "portfolio_type": STAGE3_PORTFOLIO_TYPE, "trading_date": day,
            "strategies_with_trades": len({row["strategy_id"] for row in rows}), "total_trades": len(rows),
            "wins": sum(value > 0 for value in pnl), "losses": sum(value < 0 for value in pnl), "breakeven": sum(value == 0 for value in pnl),
            "total_r": sum(float(row["result_r"] or 0.0) for row in rows), "pnl_usd": sum(pnl),
            "overall_starting_balance_for_day": start, "overall_ending_balance_for_day": end,
            "overall_peak_balance_to_date": overall_peak, "overall_drawdown_usd_to_date": drawdown,
            "overall_drawdown_pct_to_date": (drawdown / overall_peak * 100.0) if overall_peak else 0.0,
            "target_exits": sum(row["exit_reason"] == "TARGET" for row in rows),
            "stop_exits": sum(row["exit_reason"] == "STOP" for row in rows),
            "hard_flat_exits": sum(str(row["exit_reason"]).startswith("HARD_") for row in rows),
            "unresolved_trades": sum(int(row["unresolved_trades"]) for row in daily_by_strategy if str(row["trading_date"]) == day),
        })
        overall_balance = end
    pnl = [float(row["pnl_usd"]) for row in realized_rows]
    gross_profit, gross_loss = sum(value for value in pnl if value > 0), -sum(value for value in pnl if value < 0)
    ending = float(realized_rows[-1]["overall_balance_after_trade"]) if realized_rows else starting_balance
    overall = {
        "portfolio_type": STAGE3_PORTFOLIO_TYPE, "warning": STAGE3_PORTFOLIO_WARNING,
        "starting_balance_usd": starting_balance, "ending_balance_usd": ending,
        "return_pct": (ending - starting_balance) / starting_balance * 100.0, "total_pnl_usd": sum(pnl),
        "total_r": sum(float(row["result_r"] or 0.0) for row in realized_rows), "total_trades": len(realized_rows),
        "wins": sum(value > 0 for value in pnl), "losses": sum(value < 0 for value in pnl),
        "win_rate": sum(value > 0 for value in pnl) / len(pnl) if pnl else 0.0,
        "max_drawdown_usd": max((float(row["overall_drawdown_usd"]) for row in realized_rows), default=0.0),
        "max_drawdown_pct": max((float(row["overall_drawdown_pct"]) for row in realized_rows), default=0.0),
        "profit_factor": gross_profit / gross_loss if gross_loss else None, "number_of_strategies": len(strategies),
        "strategies_with_at_least_one_trade": len({row["strategy_id"] for row in realized_rows}),
        "first_trade_timestamp": None if not entry_rows else entry_rows[0]["entry_timestamp"],
        "last_trade_timestamp": None if not realized_rows else realized_rows[-1]["exit_timestamp"],
        "realized_pnl_ordering": "exit_timestamp, entry_timestamp, strategy_id, trade_number_global_for_strategy",
    }
    _write_csv(output_root / "research-trades.csv", entry_rows, ("strategy_id", "global_trade_sequence"))
    _write_csv(output_root / "research-daily-summary.csv", daily_by_strategy, ("strategy_id", "trading_date"))
    _write_csv(output_root / "research-overall-daily-summary.csv", overall_daily, ("portfolio_type", "trading_date"))
    _write_csv(output_root / "research-stage3-strategy-summary.csv", strategy_summaries, ("strategy_id",))
    _write_csv(output_root / "research-stage3-overall-summary.csv", [overall], ("portfolio_type",))
    _write_stage3_markdown(output_root / "research-trade-journal.md", entry_rows, overall)
    hashes = _period_input_hashes(period)
    run_identity = _sha({"period_id": period.period_id, "strategies": [item.__dict__ for item in strategies],
                         "selection": selection, "starting_balance_usd": starting_balance, "input_artifact_hashes": hashes,
                         "stage2_config_identities": [item.get("stage2_config_identity") for item in strategy_payloads]})
    payload = {
        "run_identity": run_identity, "selection_type": selection, "strategy_list": [item.strategy_id for item in strategies],
        "session_dates": [day for day, _path in period.sessions], "sessions": [{"date": day, "event_tape": str(path)} for day, path in period.sessions],
        "starting_balance_usd": starting_balance, "input_artifact_hashes": hashes, "strategies": strategy_payloads,
        "overall_aggregated_research_portfolio": overall, "trade_journal_path": str(output_root / "research-trades.csv"),
        "strategy_daily_summary_path": str(output_root / "research-daily-summary.csv"),
        "overall_daily_summary_path": str(output_root / "research-overall-daily-summary.csv"),
        "completion_resume_status": [{"strategy_id": item.strategy_id, "status": status_by_id.get(item.strategy_id, {}).get("status", "NOT_RUN")} for item in strategies],
        "trades_embedded": False,
    }
    _write_json(output_root / "research-stage3-summary.json", payload)
    return {"run_identity": run_identity, "overall": overall, "strategy_summaries": strategy_summaries}


def run_stage3(*, strategies_path: Path, period_path: Path, stage1_root: Path, stage2_root: Path,
               output_root: Path, starting_balance_usd: float = STAGE3_STARTING_BALANCE_USD,
               selection: str = "robust-best", workers: int = 1) -> dict[str, Any]:
    selection = _stage3_selection_key(selection)
    starting_balance = float(starting_balance_usd)
    if not math.isfinite(starting_balance) or starting_balance <= 0:
        raise MultiStrategyResearchError("Stage 3 starting balance must be a positive finite USD amount")
    if workers < 1:
        raise MultiStrategyResearchError("workers must be at least one")
    strategies, period = load_strategy_manifest(strategies_path), load_period_manifest(period_path)
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    args = [(item, period, output_root, stage1_root.resolve(), stage2_root.resolve(), selection, starting_balance) for item in strategies]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_stage3_worker, args))
    else:
        results = [_run_stage3_worker(item) for item in args]
    results.sort(key=lambda item: item["strategy_id"])
    global_summary = _generate_stage3_global(strategies, period, output_root, results,
                                             selection=selection, starting_balance=starting_balance)
    summary = {"stage": "stage3", "status": "COMPLETE", "selection_type": selection,
               "starting_balance_usd": starting_balance, "portfolio_type": STAGE3_PORTFOLIO_TYPE,
               "portfolio_warning": STAGE3_PORTFOLIO_WARNING, "strategies": results,
               "global_summary": {key: value for key, value in global_summary.items() if key != "strategy_summaries"},
               "workers": workers, "evidence_label": EVIDENCE_LABEL}
    _write_json(output_root / "stage3-summary.json", summary)
    return summary


def audit_candidate_executability(*, strategies_path: Path) -> dict[str, Any]:
    """Perform a no-data capability audit of the selected strategy manifest."""
    rows = levels.candidate_executability_audit(load_strategy_manifest(strategies_path))
    return {
        "status": "COMPLETE", "historical_data_opened": False,
        "strategies": rows,
        "executable_with_supported_inputs": sum(row["classification"] == "EXECUTABLE_WITH_SUPPORTED_INPUTS" for row in rows),
        "not_executable": sum(row["classification"] == "NOT_EXECUTABLE" for row in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("multi-strategy-screen", help="Stage 1 RR/stop screening from causal tapes")
    screen.add_argument("--strategies", type=Path, required=True); screen.add_argument("--period", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True); screen.add_argument("--workers", type=int, default=1)
    optimize = sub.add_parser("multi-strategy-optimize", help="Stage 2 offline weight/Q optimization from causal tapes")
    optimize.add_argument("--strategies", type=Path, required=True); optimize.add_argument("--period", type=Path, required=True)
    optimize.add_argument("--stage1-results", type=Path, required=True); optimize.add_argument("--output", type=Path, required=True)
    optimize.add_argument("--workers", type=int, default=1)
    journal = sub.add_parser("multi-strategy-trade-journal", help="Stage 3 frozen-selection trade journal from causal tapes")
    journal.add_argument("--strategies", type=Path, required=True); journal.add_argument("--period", type=Path, required=True)
    journal.add_argument("--stage1-results", type=Path, required=True); journal.add_argument("--stage2-results", type=Path, required=True)
    journal.add_argument("--output", type=Path, required=True); journal.add_argument("--starting-balance-usd", type=float, default=STAGE3_STARTING_BALANCE_USD)
    journal.add_argument("--selection", choices=("robust-best", "raw-best"), default="robust-best"); journal.add_argument("--workers", type=int, default=1)
    audit = sub.add_parser("multi-strategy-level-audit", help="Offline structural-level capability audit; opens no market data")
    audit.add_argument("--strategies", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "workers", 1) < 1:
            raise MultiStrategyResearchError("workers must be at least one")
        if args.command == "multi-strategy-level-audit":
            result = audit_candidate_executability(strategies_path=args.strategies)
        elif args.command == "multi-strategy-screen":
            result = run_stage1(strategies_path=args.strategies, period_path=args.period, output_root=args.output, workers=args.workers)
        elif args.command == "multi-strategy-optimize":
            result = run_stage2(strategies_path=args.strategies, period_path=args.period, stage1_root=args.stage1_results, output_root=args.output, workers=args.workers)
        else:
            result = run_stage3(strategies_path=args.strategies, period_path=args.period, stage1_root=args.stage1_results,
                                stage2_root=args.stage2_results, output_root=args.output,
                                starting_balance_usd=args.starting_balance_usd, selection=args.selection, workers=args.workers)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
