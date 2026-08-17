"""Read-only post-entry path diagnostics for existing V2 artifacts.

No interactions, PLUS scores, confirmation decisions, or recorded trade outcomes
are changed.  The scanner merely replays declared local DBNs to mark each
already-recorded entry through its already-recorded exit, plus an explicitly
labelled August-only 2.5R diagnostic replay using the frozen target-matrix
signal/confirmation/order lifecycle.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import v2_aug_vs_retro_diagnostic as artifacts
from .oos_backtest_runner import CausalMBOBook, _valid, day_and_seconds, is_snapshot_record, load_frozen
from .v2_retro_holdout_runner import CONFIRMATION_NS, RAW_PRICE_SCALE, TICK, _code, _valid_es, _valid_mes
from .v2_target_matrix_runner import (
    CONFIRM_NS, DBN as AUGUST_ES_MBO, EXPECTED_SOURCE_SHA, LATENCY_NS,
    MIN_FAVORABLE_TICKS, action_value, close_trade, load_seen_aug_plus,
    size_trade_with_mes_fallback, target_for,
)


ROOT = artifacts.ROOT
DEFAULT_RETRO_DATA = ROOT / "data/cme_orderflow_absorption_v2_holdout"
DEFAULT_OUTPUT = ROOT / "research_runs/CMEOrderflowAbsorption.ES_V2_DIAGNOSTIC/aug_vs_retro_path_replay"
RAW_TICK = 250_000_000
PROGRESS_EVERY = 5_000_000
UNRESOLVED_TAIL_ID = "2026-07-13:PRIOR_RTH_VAL:7596000000000:0097"


class PathReplayError(RuntimeError):
    pass


@dataclass
class PathState:
    row: dict[str, Any]
    observation_count: int = 0
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0
    mfe_timestamp: int | None = None
    mae_timestamp: int | None = None
    reached: dict[float, bool] = field(default_factory=lambda: {value: False for value in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)})

    @property
    def direction(self) -> str:
        return str(self.row["direction"])

    @property
    def entry_ns(self) -> int:
        return int(self.row["entry_timestamp"])

    @property
    def exit_ns(self) -> int:
        return int(self.row["exit_timestamp"])

    @property
    def price_risk_ticks(self) -> float:
        entry, stop = float(self.row["entry"]), float(self.row["stop"])
        stop_exit = stop - TICK if self.direction == "BUYER_ABSORPTION" else stop + TICK
        return abs(entry - stop_exit) / TICK

    def observe(self, timestamp: int, bid: float, ask: float) -> None:
        """Use an executable adverse exit fill, only inside [entry, exit]."""
        if timestamp < self.entry_ns or timestamp > self.exit_ns:
            return
        exit_fill = bid - TICK if self.direction == "BUYER_ABSORPTION" else ask + TICK
        signed_ticks = ((exit_fill - float(self.row["entry"])) / TICK if self.direction == "BUYER_ABSORPTION" else (float(self.row["entry"]) - exit_fill) / TICK)
        self.observation_count += 1
        favorable, adverse = max(0.0, signed_ticks), max(0.0, -signed_ticks)
        if favorable > self.mfe_ticks:
            self.mfe_ticks, self.mfe_timestamp = favorable, timestamp
        if adverse > self.mae_ticks:
            self.mae_ticks, self.mae_timestamp = adverse, timestamp
        for milestone in self.reached:
            if favorable / self.price_risk_ticks >= milestone:
                self.reached[milestone] = True

    def observe_confirmation_execution(self, timestamp: int, execution_price_raw: int) -> None:
        """Capture the frozen first execution at/after the 15-second horizon.

        August ledgers already persist this information.  Retro ledgers do not,
        so it is recovered only for an existing executed trade from its own ES
        MBO stream, using the timestamp convention of its source runner.
        """
        if self.row.get("confirmation_timestamp") is not None:
            return
        interaction_end = self.row.get("interaction_end")
        if interaction_end is None or timestamp < int(interaction_end) + CONFIRM_NS:
            return
        end_price = self.row.get("interaction_end_price")
        if end_price is None:
            raise PathReplayError("recorded trade lacks interaction-end price")
        execution_points = execution_price_raw / RAW_PRICE_SCALE
        favorable = (execution_points - float(end_price)) / TICK
        if self.direction == "SELLER_ABSORPTION":
            favorable = -favorable
        self.row["confirmation_timestamp"] = timestamp
        self.row["confirmation_price"] = execution_points
        self.row["confirmation_favorable_ticks"] = favorable

    def materialize(self) -> dict[str, Any]:
        risk_ticks = self.price_risk_ticks
        output = dict(self.row)
        output.update({
            "mfe_ticks": self.mfe_ticks if self.observation_count else None,
            "mae_ticks": self.mae_ticks if self.observation_count else None,
            "maximum_favorable_r": self.mfe_ticks / risk_ticks if self.observation_count else None,
            "maximum_adverse_r": self.mae_ticks / risk_ticks if self.observation_count else None,
            "seconds_entry_to_exit": (self.exit_ns - self.entry_ns) / RAW_PRICE_SCALE,
            "seconds_entry_to_mfe": (self.mfe_timestamp - self.entry_ns) / RAW_PRICE_SCALE if self.mfe_timestamp is not None else None,
            "seconds_entry_to_mae": (self.mae_timestamp - self.entry_ns) / RAW_PRICE_SCALE if self.mae_timestamp is not None else None,
            "path_observation_count": self.observation_count,
            "path_status": "REPLAYED" if self.observation_count else "NO_EXECUTABLE_OBSERVATIONS",
        })
        output.update({f"reached_{str(milestone).replace('.', '_')}r": value for milestone, value in self.reached.items()})
        return output


def target_2p5(direction: str, entry: float, stop: float) -> float:
    """The frozen target-matrix formula, invoked only for the 2.5R diagnostic."""
    position = {"direction": "LONG" if direction == "BUYER_ABSORPTION" else "SHORT", "entry": entry, "stop": stop}
    return target_for(position, 2.5)


def _active(states: Iterable[PathState], timestamp: int) -> list[PathState]:
    return [state for state in states if state.entry_ns <= timestamp <= state.exit_ns]


def scan_es_mbo_path(
    path: Path,
    states: list[PathState],
    *,
    august_snapshot_contract: bool,
    confirmation_states: list[PathState] | None = None,
) -> None:
    """Scan one ES MBO file once; no interaction construction occurs here."""
    from databento import DBNStore
    _, manifest = load_frozen() if august_snapshot_contract else (None, None)
    book = CausalMBOBook()
    for count, record in enumerate(DBNStore.from_file(path), start=1):
        if august_snapshot_contract and is_snapshot_record(record, manifest):
            book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False)
            continue
        action = action_value(record) if august_snapshot_contract else _code(getattr(record, "action", "N"))
        if action in {"N", "NONE"}:
            continue
        applied = book.apply(action=action, side=_code(getattr(record, "side", "")), price=int(record.price), size=int(record.size), order_id=int(record.order_id), sequence=int(record.sequence), ts_recv=int(record.ts_recv), channel_id=int(record.channel_id), validate_sequence=False, mutate_execution=False)
        quote = _valid(book) if august_snapshot_contract else _valid_es(book)
        timestamp = int(record.ts_recv) if august_snapshot_contract else int(record.ts_event)
        if applied is not None and applied.executed:
            for state in (confirmation_states if confirmation_states is not None else states):
                state.observe_confirmation_execution(timestamp, int(record.price))
        if quote is not None:
            for state in _active(states, timestamp):
                state.observe(timestamp, *quote)
        if count % PROGRESS_EVERY == 0:
            print(f"[path replay] {path.name}: records={count:,} active_trades={len(_active(states, int(getattr(record, 'ts_event', 0))))}", flush=True)


def scan_mes_mbp1_path(path: Path, states: list[PathState]) -> None:
    """Scan native MES MBP-1 snapshots once for retro MES trade-path marks."""
    from databento import DBNStore
    for count, record in enumerate(DBNStore.from_file(path), start=1):
        quote = _valid_mes(record)
        if quote is not None:
            timestamp = int(record.ts_event)
            for state in _active(states, timestamp):
                state.observe(timestamp, *quote)
        if count % PROGRESS_EVERY == 0:
            print(f"[path replay] {path.name}: records={count:,}", flush=True)


def _retro_paths(data_root: Path, date: str) -> tuple[Path, Path]:
    return (
        data_root / "es_mbo" / f"ESU6_{date}_0000_1600_mbo.dbn.zst",
        data_root / "mes_mbp1" / f"MESU6_{date}_1300_1600_mbp1.dbn.zst",
    )


def _rows_by_day(rows: list[dict[str, Any]]) -> dict[str, list[PathState]]:
    by_day: dict[str, list[PathState]] = {}
    for row in rows:
        by_day.setdefault(str(row["date"]), []).append(PathState(row=row))
    return by_day


def replay_recorded_trade_paths(*, august_trades: list[dict[str, Any]], retro_trades: list[dict[str, Any]], retro_data_root: Path = DEFAULT_RETRO_DATA) -> list[dict[str, Any]]:
    """Replay known trade paths with frozen source/execution venue separation."""
    august_by_day, retro_by_day = _rows_by_day(august_trades), _rows_by_day(retro_trades)
    august_states = [state for states in august_by_day.values() for state in states]
    for day in august_by_day:
        if day < "2026-08-03" or day > "2026-08-07":
            raise PathReplayError(f"unexpected August trade date: {day}")
    if august_states:
        scan_es_mbo_path(AUGUST_ES_MBO, august_states, august_snapshot_contract=True)
    for day, states in retro_by_day.items():
        # Confirmation is always the first relevant ES execution; it does not
        # depend on the later ES-versus-MES execution-instrument decision.
        es_states = [state for state in states if state.row["instrument"] == "ES"]
        mes_states = [state for state in states if state.row["instrument"] == "MES"]
        es_path, mes_path = _retro_paths(retro_data_root, day)
        if states:
            scan_es_mbo_path(es_path, es_states, august_snapshot_contract=False, confirmation_states=states)
        if mes_states:
            scan_mes_mbp1_path(mes_path, mes_states)
    completed = [state.materialize() for states in august_by_day.values() for state in states] + [state.materialize() for states in retro_by_day.values() for state in states]
    missing_confirmation = [row["interaction_id"] for row in completed if row.get("confirmation_timestamp") is None or row.get("confirmation_favorable_ticks") is None]
    if missing_confirmation:
        raise PathReplayError(f"completed trades lack ES confirmation recovery: {','.join(sorted(missing_confirmation))}")
    return completed


def replay_august_2p5() -> list[dict[str, Any]]:
    """Exact target-matrix lifecycle with only TARGET_R fixed to 2.5 for diagnosis."""
    from databento import DBNStore
    from .oos_backtest_runner import digest
    if digest(AUGUST_ES_MBO) != EXPECTED_SOURCE_SHA:
        raise PathReplayError("August source SHA mismatch")
    _, manifest = load_frozen()
    waiting = [{**row, "confirmation_due_ns": row["interaction_end"] + CONFIRM_NS} for row in load_seen_aug_plus()]
    ready: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    book, position, cutoff_quote = CausalMBOBook(), None, None
    for count, record in enumerate(DBNStore.from_file(AUGUST_ES_MBO), start=1):
        if is_snapshot_record(record, manifest):
            book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False)
            continue
        if action_value(record) == "N":
            continue
        applied = book.apply(action=record.action, side=record.side, price=record.price, size=record.size, order_id=record.order_id, sequence=record.sequence, ts_recv=record.ts_recv, channel_id=record.channel_id, validate_sequence=False, mutate_execution=False)
        day, _ = day_and_seconds(record.ts_recv)
        from .oos_backtest_runner import _cutoff
        cutoff, quote = _cutoff(day), _valid(book)
        if quote and cutoff - 1_000_000_000 <= record.ts_recv <= cutoff:
            cutoff_quote = (record.ts_recv, quote)
        if position is not None and record.ts_recv > position["cutoff_ns"]:
            if cutoff_quote is None:
                raise PathReplayError("CUTOFF_EXECUTION_INTEGRITY_FAILURE")
            timestamp, quote_at_cutoff = cutoff_quote
            reference = quote_at_cutoff[0] if position["direction"] == "LONG" else quote_at_cutoff[1]
            fill = reference - TICK if position["direction"] == "LONG" else reference + TICK
            trades.append(close_trade(position, timestamp, fill, "CUTOFF_FORCED_FLAT", 2.5)); position = cutoff_quote = None
        waiting = [row for row in waiting if record.ts_recv < _cutoff(row["date"])]
        if applied is not None and applied.executed:
            unresolved: list[dict[str, Any]] = []
            for row in waiting:
                if row["confirmation_due_ns"] <= record.ts_recv:
                    favorable = ((applied.price - row["end_price"]) / RAW_TICK if row["direction"] == "BUYER_ABSORPTION" else (row["end_price"] - applied.price) / RAW_TICK)
                    if favorable >= MIN_FAVORABLE_TICKS:
                        ready.append({**row, "confirmation_timestamp": record.ts_recv, "confirmation_price": applied.price / RAW_PRICE_SCALE, "confirmation_favorable_ticks": favorable, "entry_ready_ns": record.ts_recv + LATENCY_NS})
                else:
                    unresolved.append(row)
            waiting = unresolved
            ready.sort(key=lambda row: (row["entry_ready_ns"], row["interaction_end"], row["interaction_id"]))
        remaining: list[dict[str, Any]] = []
        for row in ready:
            if row["entry_ready_ns"] > record.ts_recv:
                remaining.append(row); continue
            if record.ts_recv >= _cutoff(row["date"]) or quote is None or position is not None:
                continue
            from .oos_backtest_runner import prices
            prices_row = prices(row["direction"], quote[0], quote[1], row["zone_low"] / RAW_PRICE_SCALE, row["zone_high"] / RAW_PRICE_SCALE)
            sizing = size_trade_with_mes_fallback(prices_row)
            if not sizing["contracts"]:
                continue
            position = {**row, **prices_row, **sizing, "entry_timestamp": record.ts_recv, "cutoff_ns": _cutoff(row["date"])}
            position["target"] = target_2p5(row["direction"], position["entry"], position["stop"])
        ready = remaining
        if position is not None and quote:
            bid, ask = quote
            stop_hit = (position["direction"] == "LONG" and bid <= position["stop"]) or (position["direction"] == "SHORT" and ask >= position["stop"])
            target_hit = (position["direction"] == "LONG" and bid >= position["target"]) or (position["direction"] == "SHORT" and ask <= position["target"])
            if stop_hit or target_hit:
                reference = bid if position["direction"] == "LONG" else ask
                fill = reference - TICK if position["direction"] == "LONG" else reference + TICK
                trades.append(close_trade(position, record.ts_recv, fill, "STOP" if stop_hit else "TARGET", 2.5)); position = None
        if count % PROGRESS_EVERY == 0:
            print(f"[August 2.5R diagnostic] records={count:,}", flush=True)
    if position is not None:
        raise PathReplayError("open August 2.5R position remained after source end")
    return trades


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if float(row["r_multiple"]) > 0]
    losses = [row for row in rows if float(row["r_multiple"]) < 0]
    gross_profit = sum(float(row["r_multiple"]) for row in wins)
    gross_loss = abs(sum(float(row["r_multiple"]) for row in losses))
    return {"trades": len(rows), "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / len(rows) if rows else 0.0, "total_r": sum(float(row["r_multiple"]) for row in rows), "profit_factor": gross_profit / gross_loss if gross_loss else None}


def _path_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric(field: str) -> dict[str, float | int | None]:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return {"count": len(values), "mean": statistics.fmean(values) if values else None, "median": statistics.median(values) if values else None, "minimum": min(values) if values else None, "maximum": max(values) if values else None}
    return {
        **_outcome_summary(rows),
        "numeric": {field: numeric(field) for field in ("mfe_ticks", "mae_ticks", "maximum_favorable_r", "maximum_adverse_r", "seconds_entry_to_exit", "seconds_entry_to_mfe", "seconds_entry_to_mae")},
        "milestones": {str(level): sum(bool(row.get(f"reached_{str(level).replace('.', '_')}r")) for row in rows) for level in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)},
        "path_status": {status: sum(row.get("path_status") == status for row in rows) for status in sorted({str(row.get("path_status")) for row in rows})},
    }


def materialize(*, output_dir: Path = DEFAULT_OUTPUT, retro_data_root: Path = DEFAULT_RETRO_DATA) -> dict[str, Any]:
    if output_dir.exists():
        raise PathReplayError(f"immutable diagnostic output already exists: {output_dir}")
    august_source = artifacts.DEFAULT_AUGUST_ROOT
    retro_source = artifacts.DEFAULT_RETRO_ROOT
    august_trades = [artifacts._augment_trade(row, period="AUGUST_SEEN_3R", execution_model="MES_PROXY_EXECUTION_FROM_ES_MBO") for row in artifacts._read_csv(august_source / "trades_3_0R.csv")]
    retro_signals = {row["interaction_id"]: row for row in artifacts._read_csv(retro_source / "plus-signals.csv")}
    retro_trades = [artifacts._augment_trade(row, period="RETRO_JUNE_JULY_2P5R", execution_model="NATIVE_MES_MBP1_FALLBACK", interaction=retro_signals.get(row["interaction_id"])) for row in artifacts._read_csv(retro_source / "trades.csv")]
    path_rows = replay_recorded_trade_paths(august_trades=august_trades, retro_trades=retro_trades, retro_data_root=retro_data_root)
    august_2p5 = replay_august_2p5()
    period_path = {"august": _path_summary([row for row in path_rows if row["period"] == "AUGUST_SEEN_3R"]), "retro": _path_summary([row for row in path_rows if row["period"] == "RETRO_JUNE_JULY_2P5R"])}
    summary = {
        "diagnostic_type": "READ_ONLY_POST_ENTRY_PATH_REPLAY",
        "strategy_semantics_changed": False,
        "pnl_optimization_performed": False,
        "new_strategy_rule_selected": False,
        "august_interpretation": "SEEN_AUG_DIAGNOSTIC_NOT_FRESH_OOS",
        "retro_interpretation": "NOT_STRICT_CHRONOLOGICAL_OOS; FROZEN_PARAMETER_RETROSPECTIVE_ROBUSTNESS_TEST",
        "unresolved_tail": {"interaction_id": UNRESOLVED_TAIL_ID, "status": "UNRESOLVED_SOURCE_END_1600_NOT_INFERRED"},
        "execution_models": {"august_mes": "MES_PROXY_EXECUTION_FROM_ES_MBO", "retro_mes": "NATIVE_MES_MBP1_FALLBACK"},
        "august_2p5": _outcome_summary(august_2p5),
        "period_path": period_path,
        "path_rows": len(path_rows),
    }
    output_dir.mkdir(parents=True)
    fields = sorted({key for row in path_rows for key in row})
    with (output_dir / "trade-path-diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(path_rows)
    with (output_dir / "august-2p5-trades.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_2p5 = sorted({key for row in august_2p5 for key in row}) or ["interaction_id"]
        writer = csv.DictWriter(handle, fieldnames=fields_2p5, lineterminator="\n"); writer.writeheader(); writer.writerows(august_2p5)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "period-path-comparison.json").write_text(json.dumps(period_path, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "diagnostic-report.md").write_text(
        "# V2 August vs retro post-entry path replay\n\n"
        "Descriptive diagnosis only. August is seen-data research, not fresh OOS. June/July is retrospective robustness, not strict chronological OOS. No rule was selected and no parameter was optimized.\n\n"
        "MFE/MAE uses executable adverse exit-fill marks: long `bid - 1 tick`; short `ask + 1 tick`, observed only from recorded entry through recorded exit inclusive. August MES marks remain ES-MBO proxy marks; retro MES marks use native MES MBP-1. The July 13 tail remains unresolved and was not inferred.\n\n"
        "## Period path summaries\n\n```json\n" + json.dumps(period_path, indent=2, sort_keys=True) + "\n```\n\n"
        "## Diagnostic questions\n\n"
        "A–D are answered descriptively by the MFE/MAE, duration, and milestone summaries above; they do not create a rule. E is answered by `summary.json.august_2p5`. F remains descriptive because the execution-model difference is retained rather than normalized away.\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V2 August-versus-retro DBN path diagnostic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retro-data-root", type=Path, default=DEFAULT_RETRO_DATA)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(output_dir=args.output_dir, retro_data_root=args.retro_data_root), indent=2, sort_keys=True))
    except PathReplayError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
