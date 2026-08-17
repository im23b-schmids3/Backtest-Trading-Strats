"""Read-only causal market-context diagnostic for frozen V1 PLUS setups.

All context fields are materialized at an interaction's recorded end timestamp.
Trade outcomes are joined only after those rows are frozen.  The module reads
no market data until its CLI is explicitly invoked.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import v2_aug_vs_retro_diagnostic as artifacts
from . import v2_aug_vs_retro_path_replay as paths
from .oos_backtest_runner import CausalMBOBook, is_snapshot_record, load_frozen
from .v2_retro_holdout_runner import RAW_PRICE_SCALE, _code
from .v2_target_matrix_runner import DBN as AUGUST_ES_MBO, action_value


DEFAULT_OUTPUT = artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V3_DIAGNOSTIC/long_short_regime"
MATRIX_ROOT = artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V3_RESEARCH/tick_trigger_target_matrix"
PRIMARY_CELL = "TICK_3_TARGET_2_5R"  # Predeclared, descriptive population only.
TICK = 0.25
WINDOW_NS = {"1m": 60_000_000_000, "5m": 300_000_000_000, "15m": 900_000_000_000, "30m": 1_800_000_000_000, "60m": 3_600_000_000_000}
ACTIVITY_NS = {"5s": 5_000_000_000, "15s": 15_000_000_000, "60s": 60_000_000_000}
PROGRESS_EVERY = 5_000_000


class RegimeDiagnosticError(RuntimeError):
    pass


def _points(raw: int | float) -> float:
    return float(raw) / RAW_PRICE_SCALE


def _long(direction: str) -> bool:
    if direction in {"BUYER_ABSORPTION", "LONG"}: return True
    if direction in {"SELLER_ABSORPTION", "SHORT"}: return False
    raise RegimeDiagnosticError(f"unknown direction: {direction!r}")


def direction_normalize(value: float | None, direction: str) -> float | None:
    return value if value is None or _long(direction) else -value


def _last_at_or_before(items: deque[tuple[int, float, int]], cutoff: int) -> tuple[int, float, int] | None:
    for item in reversed(items):
        if item[0] <= cutoff: return item
    return None


@dataclass
class ContextAccumulator:
    """One-pass bounded causal execution context; no future prices are retained."""
    executions: deque[tuple[int, float, int]] = field(default_factory=deque)
    rth_open: float | None = None
    rth_high: float | None = None
    rth_low: float | None = None
    vwap_volume: int = 0
    vwap_price_volume: float = 0.0

    def observe_execution(self, timestamp: int, raw_price: int, size: int) -> None:
        price = _points(raw_price)
        if self.rth_open is None: self.rth_open = price
        self.rth_high = price if self.rth_high is None else max(self.rth_high, price)
        self.rth_low = price if self.rth_low is None else min(self.rth_low, price)
        self.vwap_volume += size; self.vwap_price_volume += price * size
        self.executions.append((timestamp, price, size))
        floor = timestamp - WINDOW_NS["60m"]
        while self.executions and self.executions[0][0] < floor: self.executions.popleft()

    def snapshot(self, *, timestamp: int, end_price_raw: int, direction: str) -> dict[str, Any]:
        end_price = _points(end_price_raw)
        result: dict[str, Any] = {"interaction_end_price": end_price, "rth_open_price": self.rth_open, "session_high": self.rth_high, "session_low": self.rth_low}
        for label in ("5m", "15m", "30m", "60m"):
            prior = _last_at_or_before(self.executions, timestamp - WINDOW_NS[label])
            move = end_price - prior[1] if prior is not None else None
            result[f"price_move_{label}_points"] = move
            result[f"price_move_{label}_ticks"] = move / TICK if move is not None else None
            result[f"direction_normalized_price_move_{label}_ticks"] = direction_normalize(result[f"price_move_{label}_ticks"], direction)
        open_move = end_price - self.rth_open if self.rth_open is not None else None
        result["price_move_from_rth_open_points"] = open_move
        result["price_move_from_rth_open_ticks"] = open_move / TICK if open_move is not None else None
        result["direction_normalized_move_from_rth_open_ticks"] = direction_normalize(result["price_move_from_rth_open_ticks"], direction)
        vwap = self.vwap_price_volume / self.vwap_volume if self.vwap_volume else None
        result["session_vwap"] = vwap
        result["price_minus_vwap_points"] = end_price - vwap if vwap is not None else None
        result["price_minus_vwap_ticks"] = result["price_minus_vwap_points"] / TICK if result["price_minus_vwap_points"] is not None else None
        result["direction_normalized_price_minus_vwap_ticks"] = direction_normalize(result["price_minus_vwap_ticks"], direction)
        if self.rth_high is not None and self.rth_low is not None:
            session_range = self.rth_high - self.rth_low
            result.update({"session_range_points": session_range, "session_range_ticks": session_range / TICK, "distance_from_session_high_ticks": (self.rth_high - end_price) / TICK, "distance_from_session_low_ticks": (end_price - self.rth_low) / TICK, "session_range_position_0_1": (end_price - self.rth_low) / session_range if session_range else None})
        for label in ("1m", "5m", "15m"):
            recent = [price for ts, price, _ in self.executions if ts >= timestamp - WINDOW_NS[label]]
            span = max(recent) - min(recent) if recent else None
            result[f"recent_range_{label}_points"] = span
            result[f"recent_range_{label}_ticks"] = span / TICK if span is not None else None
        for label, window in ACTIVITY_NS.items():
            recent = [(ts, size) for ts, _, size in self.executions if ts >= timestamp - window]
            count, volume = len(recent), sum(size for _, size in recent)
            seconds = window / RAW_PRICE_SCALE
            result.update({f"execution_count_{label}": count, f"executed_volume_{label}": volume, f"executions_per_second_{label}": count / seconds, f"executed_volume_per_second_{label}": volume / seconds})
        return result


def previous_level_counts(signals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Causal prior PLUS visits per day/level, excluding the current setup."""
    counts: Counter[tuple[str, str]] = Counter(); output: dict[str, dict[str, Any]] = {}
    for row in sorted(signals, key=lambda item: (int(item["interaction_end"]), str(item["interaction_id"]))):
        key = (str(row["date"]), str(row["level"])); prior = counts[key]
        output[str(row["interaction_id"])] = {"previous_plus_events_same_level": prior, "level_touch_ordinal": "FIRST" if prior == 0 else "SECOND" if prior == 1 else "THIRD_PLUS"}
        counts[key] += 1
    return output


def previous_completed_interaction_counts(interactions: list[dict[str, str]], signals: list[dict[str, Any]]) -> dict[str, int]:
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in interactions: by_key[(row["date"], row["level"])].append(int(row["interaction_end"]))
    for values in by_key.values(): values.sort()
    return {str(row["interaction_id"]): bisect.bisect_left(by_key[(str(row["date"]), str(row["level"]))], int(row["interaction_end"])) for row in signals}


def _scan(path: Path, signals: list[dict[str, Any]], *, august: bool) -> dict[str, dict[str, Any]]:
    from databento import DBNStore
    expected = sorted(signals, key=lambda row: (int(row["interaction_end"]), str(row["interaction_id"])))
    output: dict[str, dict[str, Any]] = {}; cursor = 0; accumulator = ContextAccumulator(); book = CausalMBOBook()
    _, manifest = load_frozen() if august else (None, None)
    for count, record in enumerate(DBNStore.from_file(path), start=1):
        if august and is_snapshot_record(record, manifest):
            book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False); continue
        action = action_value(record) if august else _code(getattr(record, "action", "N"))
        if action in {"N", "NONE"}: continue
        timestamp = int(record.ts_recv) if august else int(record.ts_event)
        # A signal ending between executions must be frozen before the later
        # record is admitted; otherwise its context would leak a future price.
        while cursor < len(expected) and int(expected[cursor]["interaction_end"]) < timestamp:
            row = expected[cursor]
            output[str(row["interaction_id"])] = accumulator.snapshot(timestamp=int(row["interaction_end"]), end_price_raw=int(row["end_price"]), direction=str(row["direction"]))
            cursor += 1
        applied = book.apply(action=action, side=_code(getattr(record, "side", "")), price=int(record.price), size=int(record.size), order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv), channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
        if applied is not None and applied.executed:
            accumulator.observe_execution(timestamp, int(record.price), int(record.size))
        while cursor < len(expected) and timestamp >= int(expected[cursor]["interaction_end"]):
            row = expected[cursor]
            # Current record has been included only if it is at/before end.
            output[str(row["interaction_id"])] = accumulator.snapshot(timestamp=int(row["interaction_end"]), end_price_raw=int(row["end_price"]), direction=str(row["direction"]))
            cursor += 1
        if count % PROGRESS_EVERY == 0: print(f"[regime context] {path.name}: records={count:,} setups={cursor:,}", flush=True)
    if cursor != len(expected): raise RegimeDiagnosticError(f"source ended before context snapshot: {expected[cursor]['interaction_id']}")
    return output


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values: return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    values = sorted(values)
    def percentile(q: float) -> float:
        index = (len(values) - 1) * q; lower, upper = math.floor(index), math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "p25": percentile(.25), "p75": percentile(.75), "min": values[0], "max": values[-1]}


NUMERIC_FEATURES = ("absorption_score", "replenishment_score", "direction_normalized_price_move_5m_ticks", "direction_normalized_price_move_15m_ticks", "direction_normalized_price_move_30m_ticks", "direction_normalized_price_move_60m_ticks", "direction_normalized_price_minus_vwap_ticks", "session_range_ticks", "session_range_position_0_1", "recent_range_1m_ticks", "recent_range_5m_ticks", "recent_range_15m_ticks", "execution_count_5s", "executed_volume_5s", "execution_count_15s", "executed_volume_15s", "execution_count_60s", "executed_volume_60s", "previous_completed_interactions_same_level", "previous_plus_events_same_level", "entry_displacement_ticks_from_interaction_end", "stop_distance_ticks", "mfe_ticks", "mae_ticks", "r_multiple")


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {feature: _stats([float(row[feature]) for row in rows if row.get(feature) not in (None, "")]) for feature in NUMERIC_FEATURES}


def _trade_index() -> dict[str, dict[str, Any]]:
    ledger = artifacts._read_csv(MATRIX_ROOT / "trade-ledger.csv")
    rows = [row for row in ledger if row["cell_id"] == PRIMARY_CELL]
    return {f"{row['period']}:{row['interaction_id']}": row for row in rows}


def materialize(*, output_dir: Path = DEFAULT_OUTPUT, retro_data_root: Path = paths.DEFAULT_RETRO_DATA) -> dict[str, Any]:
    if output_dir.exists(): raise RegimeDiagnosticError(f"immutable output already exists: {output_dir}")
    from .v2_target_matrix_runner import load_seen_aug_plus
    august = [{**row, "period": "AUGUST_SEEN"} for row in load_seen_aug_plus()]
    retro = [{**row, "period": "RETRO_JUNE_JULY"} for row in artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "plus-signals.csv")]
    aug_context = _scan(AUGUST_ES_MBO, august, august=True)
    retro_context: dict[str, dict[str, Any]] = {}
    for date in sorted({str(row["date"]) for row in retro}):
        day_rows = [row for row in retro if row["date"] == date]
        es_path, _ = paths._retro_paths(retro_data_root, date); retro_context.update(_scan(es_path, day_rows, august=False))
    interactions_aug = [row for row in artifacts._read_csv(artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_RESEARCH/seen_15_rth/all-interactions.csv") if row.get("research_split") == "SEEN_OOS_AUG"]
    interactions_retro = artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "interactions.csv")
    level_prior = {"AUGUST_SEEN": previous_level_counts(august), "RETRO_JUNE_JULY": previous_level_counts(retro)}
    interaction_prior = {"AUGUST_SEEN": previous_completed_interaction_counts(interactions_aug, august), "RETRO_JUNE_JULY": previous_completed_interaction_counts(interactions_retro, retro)}
    trades = _trade_index(); rows: list[dict[str, Any]] = []
    for signal in august + retro:
        period, identifier = signal["period"], str(signal["interaction_id"])
        context = (aug_context if period == "AUGUST_SEEN" else retro_context)[identifier]
        trade = trades.get(f"{period}:{identifier}")
        row = {"period": period, "interaction_id": identifier, "date": signal["date"], "direction": signal["direction"], "level": signal["level"], "absorption_score": float(signal["absorption_score"]), "replenishment_score": float(signal["replenishment_score"]), **context, **level_prior[period][identifier], "previous_completed_interactions_same_level": interaction_prior[period][identifier], "primary_matrix_trade": trade is not None}
        if trade is not None:
            row.update({"trade_outcome": "WIN" if float(trade["r_multiple"]) > 0 else "LOSS" if float(trade["r_multiple"]) < 0 else "FLAT", "entry_displacement_ticks_from_interaction_end": float(trade["entry_displacement_ticks_from_interaction_end"]), "stop_distance_ticks": float(trade["stop_distance_ticks"]), "r_multiple": float(trade["r_multiple"]), "mfe_ticks": None, "mae_ticks": None})
        else: row.update({"trade_outcome": "NOT_TRADED", "entry_displacement_ticks_from_interaction_end": None, "stop_distance_ticks": None, "r_multiple": None, "mfe_ticks": None, "mae_ticks": None})
        rows.append(row)
    groups = {"period": {name: _group([row for row in rows if row["period"] == name]) for name in ("AUGUST_SEEN", "RETRO_JUNE_JULY")}, "direction": {name: _group([row for row in rows if row["direction"] == name]) for name in ("BUYER_ABSORPTION", "SELLER_ABSORPTION")}, "level": {name: _group([row for row in rows if row["level"] == name]) for name in sorted({row["level"] for row in rows})}, "outcome": {name: _group([row for row in rows if row["trade_outcome"] == name]) for name in ("WIN", "LOSS", "NOT_TRADED")}}
    payload = {"diagnostic_type": "READ_ONLY_CAUSAL_LONG_SHORT_REGIME_CONTEXT", "primary_population": f"{PRIMARY_CELL} predeclared matrix population; not selected", "strategy_semantics_changed": False, "pnl_optimization_performed": False, "selection_prohibited": True, "context_frozen_at": "interaction_end; outcomes joined afterward", "rows": len(rows), "groups": groups, "interpretation": {"august": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE", "retro": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST"}, "hypothesis": "DESCRIPTIVE_ONLY: assess group distributions; no context feature becomes a rule without untouched OOS validation."}
    output_dir.mkdir(parents=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "setup-context.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    definitions = {"feature_time": "Every context feature is computed with ES executions at or before interaction_end only.", "trend": "Price movement is interaction_end_price minus last causal ES execution at/before the declared lookback boundary.", "vwap": "Cumulative RTH executed ES trade price×volume / volume through interaction_end.", "range": "High-low of causal ES executions in the declared 1m/5m/15m window.", "activity": "Count and volume of causal ES executions in 5s/15s/60s windows.", "level_history": "Counts completed materialized interactions and PLUS events before current interaction_end at same date/level; current/future excluded."}
    (output_dir / "feature-definitions.json").write_text(json.dumps(definitions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "group-comparison.json").write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text("# Long/short and high/low-sweep regime diagnostic\n\nContext is frozen causally at each PLUS interaction end. Trade outcomes are attached only afterward for descriptive grouping. The report contains no optimal cutoff, no filter, and no selected rule.\n\nThe primary comparison is the predeclared +3 tick / 2.5R matrix population, retained only as a diagnostic population. August is seen data; retro is retrospective. Any observed BUYER/SELLER or LOW/HIGH asymmetry is a hypothesis requiring untouched OOS validation.\n\n```json\n" + json.dumps(groups, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only causal V2 long/short regime diagnostic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--retro-data-root", type=Path, default=paths.DEFAULT_RETRO_DATA)
    args = parser.parse_args()
    try: print(json.dumps(materialize(output_dir=args.output_dir, retro_data_root=args.retro_data_root), indent=2, sort_keys=True))
    except RegimeDiagnosticError as exc: print(f"ERROR: {exc}"); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
