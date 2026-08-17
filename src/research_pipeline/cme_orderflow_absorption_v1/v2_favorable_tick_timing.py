"""Read-only first-favorable-tick timing diagnostic for frozen V2 PLUS setups."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import v2_aug_vs_retro_diagnostic as artifacts
from . import v2_aug_vs_retro_path_replay as paths
from .oos_backtest_runner import CausalMBOBook, is_snapshot_record, load_frozen
from .v2_retro_holdout_runner import CONFIRMATION_NS, RAW_PRICE_SCALE, TICK, _code
from .v2_target_matrix_runner import DBN as AUGUST_ES_MBO, action_value


DEFAULT_OUTPUT = artifacts.ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_DIAGNOSTIC/favorable_tick_timing"
THRESHOLDS = (1, 2, 3, 4, 5)


class TimingError(RuntimeError):
    pass


def _raw_points(value: Any) -> float:
    return float(value) / RAW_PRICE_SCALE


def _long(direction: str) -> bool:
    if direction in {"BUYER_ABSORPTION", "LONG"}:
        return True
    if direction in {"SELLER_ABSORPTION", "SHORT"}:
        return False
    raise TimingError(f"unknown direction: {direction!r}")


@dataclass
class TimingState:
    row: dict[str, Any]
    timestamp_semantics: str
    confirmation_timestamp: int | None = None
    confirmation_price: float | None = None
    confirmation_favorable_ticks: float | None = None
    first_reach: dict[int, tuple[int, float] | None] = field(default_factory=lambda: {threshold: None for threshold in THRESHOLDS})

    @property
    def interaction_end(self) -> int:
        return int(self.row["interaction_end"])

    @property
    def interaction_end_price(self) -> float:
        return _raw_points(self.row["end_price"])

    @property
    def confirmation_due(self) -> int:
        return self.interaction_end + CONFIRMATION_NS

    def observe_execution(self, timestamp: int, raw_price: int) -> None:
        """Observe only causal ES executions from interaction end through +15s."""
        if timestamp < self.interaction_end or timestamp > self.confirmation_due:
            return
        price = _raw_points(raw_price)
        favorable = (price - self.interaction_end_price) / TICK
        if not _long(str(self.row["direction"])):
            favorable = -favorable
        for threshold in THRESHOLDS:
            if favorable >= threshold and self.first_reach[threshold] is None:
                self.first_reach[threshold] = (timestamp, price)
        if timestamp == self.confirmation_due and self.confirmation_timestamp is None:
            self.confirmation_timestamp, self.confirmation_price, self.confirmation_favorable_ticks = timestamp, price, favorable

    def observe_after_horizon_execution(self, timestamp: int, raw_price: int) -> None:
        """Record the frozen first ES execution at or after the horizon once."""
        if self.confirmation_timestamp is not None or timestamp < self.confirmation_due:
            return
        price = _raw_points(raw_price)
        favorable = (price - self.interaction_end_price) / TICK
        if not _long(str(self.row["direction"])):
            favorable = -favorable
        self.confirmation_timestamp, self.confirmation_price, self.confirmation_favorable_ticks = timestamp, price, favorable

    def materialize(self, trade: dict[str, Any] | None) -> dict[str, Any]:
        output: dict[str, Any] = {
            "period": self.row["period"], "interaction_id": self.row["interaction_id"], "date": self.row["date"],
            "direction": self.row["direction"], "level": self.row["level"], "interaction_end": self.interaction_end,
            "interaction_end_price": self.interaction_end_price, "confirmation_timestamp": self.confirmation_timestamp,
            "confirmation_price": self.confirmation_price, "confirmation_favorable_ticks": self.confirmation_favorable_ticks,
            "timestamp_semantics": self.timestamp_semantics,
            "confirmation_status": "OBSERVED" if self.confirmation_timestamp is not None else "NOT_OBSERVED_SOURCE_END",
            "passed_confirmation": self.confirmation_favorable_ticks is not None and self.confirmation_favorable_ticks >= 1,
        }
        if trade is not None:
            entry = float(trade["entry"])
            sign = 1.0 if _long(str(self.row["direction"])) else -1.0
            output.update({"entry": entry, "instrument": trade["instrument"], "entry_timestamp": int(trade["entry_timestamp"]), "entry_displacement_ticks_from_interaction_end": sign * (entry - self.interaction_end_price) / TICK})
        else:
            output.update({"entry": None, "instrument": None, "entry_timestamp": None, "entry_displacement_ticks_from_interaction_end": None})
        for threshold, occurrence in self.first_reach.items():
            prefix = f"plus_{threshold}_ticks"
            if occurrence is None:
                output.update({f"{prefix}_reached": False, f"{prefix}_timestamp": None, f"{prefix}_seconds_after_interaction_end": None, f"{prefix}_execution_price": None, f"seconds_from_{prefix}_to_actual_entry": None, f"directional_ticks_from_{prefix}_to_actual_entry": None})
                continue
            timestamp, price = occurrence
            sign = 1.0 if _long(str(self.row["direction"])) else -1.0
            output.update({
                f"{prefix}_reached": True, f"{prefix}_timestamp": timestamp,
                f"{prefix}_seconds_after_interaction_end": (timestamp - self.interaction_end) / RAW_PRICE_SCALE,
                f"{prefix}_execution_price": price,
                f"seconds_from_{prefix}_to_actual_entry": (int(trade["entry_timestamp"]) - timestamp) / RAW_PRICE_SCALE if trade is not None else None,
                f"directional_ticks_from_{prefix}_to_actual_entry": sign * (float(trade["entry"]) - price) / TICK if trade is not None else None,
            })
        return output


def _scan_es(path: Path, states: list[TimingState], *, august: bool) -> None:
    from databento import DBNStore
    book = CausalMBOBook()
    _, manifest = load_frozen() if august else (None, None)
    for count, record in enumerate(DBNStore.from_file(path), start=1):
        if august and is_snapshot_record(record, manifest):
            book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False)
            continue
        action = action_value(record) if august else _code(getattr(record, "action", "N"))
        if action in {"N", "NONE"}:
            continue
        applied = book.apply(action=action, side=_code(getattr(record, "side", "")), price=int(record.price), size=int(record.size), order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv), channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
        if applied is not None and applied.executed:
            timestamp = int(record.ts_recv) if august else int(record.ts_event)
            for state in states:
                state.observe_execution(timestamp, int(record.price))
                state.observe_after_horizon_execution(timestamp, int(record.price))
        if count % paths.PROGRESS_EVERY == 0:
            print(f"[favorable-tick timing] {path.name}: records={count:,}", flush=True)


def _retro_plus() -> list[dict[str, Any]]:
    signals = artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "plus-signals.csv")
    return [{**row, "period": "RETRO_JUNE_JULY"} for row in signals]


def _august_plus() -> list[dict[str, Any]]:
    return [{**row, "period": "AUGUST_SEEN"} for row in paths.load_seen_aug_plus()]


def _trades_by_id() -> dict[str, dict[str, Any]]:
    august = artifacts._read_csv(artifacts.DEFAULT_AUGUST_ROOT / "trades_3_0R.csv")
    retro = artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "trades.csv")
    return {row["interaction_id"]: row for row in august + retro}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _describe(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "mean": statistics.fmean(values) if values else None, "median": statistics.median(values) if values else None, "p25": _percentile(values, .25) if values else None, "p75": _percentile(values, .75) if values else None}


def _period_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row["passed_confirmation"]]
    thresholds: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        prefix = f"plus_{threshold}_ticks"
        reached = [row for row in passed if row[f"{prefix}_reached"]]
        thresholds[str(threshold)] = {
            "passed_confirmation_setups": len(passed), "reached_count": len(reached), "reached_percentage": len(reached) / len(passed) if passed else None,
            "seconds_to_first_reach": _describe([float(row[f"{prefix}_seconds_after_interaction_end"]) for row in reached]),
            "remaining_seconds_to_actual_entry": _describe([float(row[f"seconds_from_{prefix}_to_actual_entry"]) for row in reached if row[f"seconds_from_{prefix}_to_actual_entry"] is not None]),
            "remaining_directional_ticks_to_actual_entry": _describe([float(row[f"directional_ticks_from_{prefix}_to_actual_entry"]) for row in reached if row[f"directional_ticks_from_{prefix}_to_actual_entry"] is not None]),
        }
    return {"plus_setups": len(rows), "passed_confirmations": len(passed), "thresholds": thresholds}


def materialize(*, output_dir: Path = DEFAULT_OUTPUT, retro_data_root: Path = paths.DEFAULT_RETRO_DATA) -> dict[str, Any]:
    if output_dir.exists():
        raise TimingError(f"immutable diagnostic output already exists: {output_dir}")
    august_states = [TimingState(row, "ES_MBO_TS_RECV") for row in _august_plus()]
    _scan_es(AUGUST_ES_MBO, august_states, august=True)
    retro_states_by_day: dict[str, list[TimingState]] = {}
    for row in _retro_plus():
        retro_states_by_day.setdefault(str(row["date"]), []).append(TimingState(row, "ES_MBO_TS_EVENT"))
    for date, states in sorted(retro_states_by_day.items()):
        es_path, _ = paths._retro_paths(retro_data_root, date)
        _scan_es(es_path, states, august=False)
    trades = _trades_by_id()
    rows = [state.materialize(trades.get(state.row["interaction_id"])) for state in august_states + [state for states in retro_states_by_day.values() for state in states]]
    passed_rows = [row for row in rows if row["passed_confirmation"]]
    completed_ids = {row["interaction_id"] for row in artifacts._read_csv(artifacts.DEFAULT_RETRO_ROOT / "trades.csv")}
    missing_completed = sorted(identifier for identifier in completed_ids if not next((row for row in passed_rows if row["interaction_id"] == identifier), None))
    if missing_completed:
        raise TimingError(f"completed retro trades fail confirmation reconciliation: {','.join(missing_completed)}")
    periods = {"august": _period_summary([row for row in rows if row["period"] == "AUGUST_SEEN"]), "retro": _period_summary([row for row in rows if row["period"] == "RETRO_JUNE_JULY"])}
    summary = {
        "diagnostic_type": "READ_ONLY_FAVORABLE_TICK_TIMING",
        "strategy_semantics_changed": False, "pnl_optimization_performed": False, "new_strategy_rule_selected": False,
        "confirmation_reconciliation": {"retro_completed_trades": len(completed_ids), "retro_completed_trades_with_passed_es_confirmation": len(completed_ids) - len(missing_completed), "reconciles": not missing_completed},
        "periods": periods,
        "interpretation": {"august": "SEEN_AUG_DATA_NOT_FRESH_OOS_EVIDENCE", "retro": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST"},
        "questions": "Threshold timing is observability only. It neither calculates alternative-trigger PnL nor selects a confirmation rule.",
    }
    output_dir.mkdir(parents=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "favorable-tick-timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "period-timing-comparison.json").write_text(json.dumps(periods, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text(
        "# V2 favorable-tick timing diagnostic\n\n"
        "Read-only observability only. The frozen confirmation remains the first ES execution at or after interaction end plus 15 seconds, requiring at least +1 favorable ES tick. No alternative confirmation, PnL, or rule selection was evaluated.\n\n"
        "For LONG, favorable ticks are `(execution - interaction_end) / 0.25`; for SHORT, `(interaction_end - execution) / 0.25`. First reaches are immutable first occurrences from interaction end through +15 seconds inclusive. Confirmation is always recovered from ES execution data, including trades ultimately entered as MES.\n\n"
        "```json\n" + json.dumps(periods, indent=2, sort_keys=True) + "\n```\n\n"
        "August remains seen-data research; retro remains a non-strict-chronological retrospective robustness test. Any descriptive timing difference cannot become a rule without fresh validation.\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V2 favorable-tick timing diagnostic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retro-data-root", type=Path, default=paths.DEFAULT_RETRO_DATA)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(output_dir=args.output_dir, retro_data_root=args.retro_data_root), indent=2, sort_keys=True))
    except TimingError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
