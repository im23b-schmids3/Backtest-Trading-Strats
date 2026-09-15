"""Reusable two-stage, offline-first research orchestration for CME ES L2.

Stage 1 screens the *existing causal tapes* with a fixed quality gate and one
common legal weight vector while varying only target R and the canonical zone
stop buffer.  Stage 2 freezes that geometry per strategy and evaluates the
existing legal five-weight grid without opening DBN files.  Both stages are
development research only; neither selects or promotes a production strategy.
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
    reference_level: str
    long_short_behavior: str
    uses_shared_absorption_engine: bool
    trigger_options: Mapping[str, Any]
    baseline_quality: Mapping[str, Any]
    allowed_research_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class PeriodSpec:
    period_id: str
    interaction_master: Path
    interaction_index: Path
    sessions: tuple[tuple[str, Path], ...]


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
        specs.append(StrategySpec(
            strategy_id=str(item["strategy_id"]), session=str(item["session"]),
            reference_level=str(item["reference_level"]), long_short_behavior=str(item["long_short_behavior"]),
            uses_shared_absorption_engine=bool(item["uses_shared_absorption_engine"]),
            trigger_options=dict(item.get("trigger_options") or {}),
            baseline_quality=dict(item.get("baseline_quality") or {}),
            allowed_research_parameters=dict(item.get("allowed_research_parameters") or {}),
        ))
    if not specs or len({item.strategy_id for item in specs}) != len(specs):
        raise MultiStrategyResearchError("strategy IDs must be non-empty and unique")
    if any(not item.uses_shared_absorption_engine for item in specs):
        raise MultiStrategyResearchError("this workflow currently requires the shared absorption engine")
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
    required = (interaction_master, interaction_index, *(item[1] for item in sessions))
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise MultiStrategyResearchError(f"period causal artifacts missing: {missing[:3]}")
    return PeriodSpec(str(raw.get("period_id") or "unnamed-period"), interaction_master, interaction_index, sessions)


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
    return _sha({"stage": stage, "strategy": strategy.__dict__, "period": period.period_id, "sources": sources, "extra": extra})


def _load_population(period: PeriodSpec, strategy: StrategySpec) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    interactions = master._read_parquet_rows(period.interaction_master)
    indexes = {str(row["interaction_id"]): dict(row) for row in master._read_parquet_rows(period.interaction_index)}
    result: dict[str, list[dict[str, Any]]] = {day: [] for day, _path in period.sessions}
    for row in interactions:
        day = str(row.get("session_date"))
        if day in result and str(row.get("level")) == strategy.reference_level:
            identifier = str(row["interaction_id"])
            if identifier not in indexes:
                raise MultiStrategyResearchError(f"interaction index missing: {strategy.strategy_id}/{identifier}")
            result[day].append(dict(row))
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
        return {"strategy_id": strategy.strategy_id, "status": "REUSED", "root": str(strategy_root), **reused}
    rows_by_day, indexes = _load_population(period, strategy)
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
                "configuration_count": len(rows), "selection": selection, "evidence_label": EVIDENCE_LABEL}
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
    _write_csv(output_root / "cross-strategy-stage1-summary.csv", results, ("strategy_id", "status"))
    summary = {"stage": "stage1", "status": "COMPLETE", "evidence_label": EVIDENCE_LABEL, "stage1_quality_threshold": str(STAGE1_Q),
               "common_baseline_weights": {key: str(value) for key, value in COMMON_BASELINE_WEIGHTS.items()},
               "baseline_derivation": BASELINE_DERIVATION, "matrix_per_strategy": len(STAGE1_RR) * len(STAGE1_STOP_TICKS),
               "strategies": results, "workers": workers}
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


def _run_stage2_strategy(strategy: StrategySpec, period: PeriodSpec, root: Path, stage1_root: Path, *, grid: Sequence[tuple[int, int, int, int, int]] | None = None) -> dict[str, Any]:
    stage1_dir = stage1_root / "strategies" / _safe_name(strategy.strategy_id)
    selection_path = stage1_dir / "selection.json"
    if not selection_path.is_file():
        raise MultiStrategyResearchError(f"missing Stage 1 selection: {strategy.strategy_id}")
    execution = json.loads(selection_path.read_text(encoding="utf-8"))["stage2_execution_configuration"]
    weights_grid = tuple(grid or matrix.generate_weight_grid())
    identity = _identity(strategy, period, stage="stage2", extra={"stage1_selection": execution, "q": [str(value) for value in STAGE2_Q], "weight_grid_sha": _sha(weights_grid)})
    strategy_root = root / "strategies" / _safe_name(strategy.strategy_id)
    reused = _complete_or_raise(strategy_root, identity)
    if reused:
        return {"strategy_id": strategy.strategy_id, "status": "REUSED", "root": str(strategy_root), **reused}
    progress_path = strategy_root / "in-progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_identity") != identity:
            raise MultiStrategyResearchError(f"interrupted Stage 2 identity differs: {strategy.strategy_id}")
    else:
        _write_json(progress_path, {"stage": "stage2", "input_identity": identity, "status": "IN_PROGRESS"})
    rows_by_day, indexes = _load_population(period, strategy)
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
    complete = {"stage": "stage2", "status": "COMPLETE", "input_identity": identity, "strategy_id": strategy.strategy_id,
                "configuration_count": len(results), "rr": rr, "stop_ticks": stop, "selection": selection, "evidence_label": EVIDENCE_LABEL}
    _write_json(strategy_root / "complete.json", complete)
    return {"strategy_id": strategy.strategy_id, "status": "COMPLETE", "root": str(strategy_root), **complete}


def _run_stage2_worker(args: tuple[StrategySpec, PeriodSpec, Path, Path]) -> dict[str, Any]:
    strategy, period, root, stage1_root = args
    try:
        return _run_stage2_strategy(strategy, period, root, stage1_root)
    except Exception as exc:
        failure_root = root / "strategies" / _safe_name(strategy.strategy_id)
        _write_json(failure_root / "failure.json", {"stage": "stage2", "strategy_id": strategy.strategy_id, "error": str(exc)})
        return {"strategy_id": strategy.strategy_id, "status": "FAILED", "error": str(exc), "root": str(failure_root)}


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
    _write_csv(output_root / "cross-strategy-stage2-summary.csv", results, ("strategy_id", "status"))
    summary = {"stage": "stage2", "status": "COMPLETE", "evidence_label": EVIDENCE_LABEL,
               "quality_thresholds": [str(value) for value in STAGE2_Q], "weight_count": len(matrix.generate_weight_grid()),
               "configuration_count_per_strategy": len(matrix.generate_weight_grid()) * len(STAGE2_Q), "strategies": results, "workers": workers,
               "automatic_production_promotion": False}
    _write_json(output_root / "stage2-summary.json", summary)
    return summary


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
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise MultiStrategyResearchError("workers must be at least one")
        result = (run_stage1(strategies_path=args.strategies, period_path=args.period, output_root=args.output, workers=args.workers)
                  if args.command == "multi-strategy-screen" else
                  run_stage2(strategies_path=args.strategies, period_path=args.period, stage1_root=args.stage1_results, output_root=args.output, workers=args.workers))
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
