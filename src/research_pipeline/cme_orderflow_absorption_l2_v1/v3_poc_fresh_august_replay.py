"""Local-only first fresh native-L2 replay for the sealed POC-only V3 contract.

Nothing in this module acquires data or contacts Databento services.  The CLI
is intentionally explicit and must be invoked manually after preflight.  It
does not publish an artifact directory until all five sessions have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import historical_runner as historical
from .model import Execution, MBP10Snapshot, MBP10Update, MBPLevel, StructuralLevel, TICK
from .v2_quality050 import V2_CONFIG
from .v3_poc_only import (
    ELIGIBLE_STRUCTURAL_LEVELS, EVIDENCE_LABEL, STRATEGY_ID, v3_contract, v3_contract_sha256,
)


DATA_ROOT = Path("data/cme_orderflow_absorption_l2_v3/aug10_14_fresh")
OUTPUT_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY_AUG10_14_FRESH")
TARGET_DATES = ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14")
PRIOR_RTH = {
    "2026-08-10": "2026-08-07", "2026-08-11": "2026-08-10", "2026-08-12": "2026-08-11",
    "2026-08-13": "2026-08-12", "2026-08-14": "2026-08-13",
}
RAW_PRICE_SCALE = 1_000_000_000
NOMINAL_HARD_FLAT_SECONDS = historical.HARD_CUTOFF_SECONDS
SCHEDULED_CLOSE_SECONDS = {"2026-08-14": 21 * 60 * 60}
GLOBEX_MAINTENANCE_PAUSE = (21 * 60 * 60, 22 * 60 * 60)
CALENDAR_CLARIFICATION = "EXECUTION_CALENDAR_CLARIFICATION"


class V3FreshReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_hard_flat_seconds(day: str) -> int:
    """The frozen 22:45 convention bounded only by a documented exchange close."""
    return min(NOMINAL_HARD_FLAT_SECONDS, SCHEDULED_CLOSE_SECONDS.get(day, NOMINAL_HARD_FLAT_SECONDS))


def liquidation_window_ns(day: str) -> tuple[int, int]:
    cutoff = historical._clock_ns(day, effective_hard_flat_seconds(day))
    return cutoff - historical.CUTOFF_QUOTE_LOOKBACK_NS, cutoff


def calendar_contract() -> dict[str, Any]:
    return {
        "classification": CALENDAR_CLARIFICATION,
        "normal_hard_flat_utc": "22:45:00",
        "rule": "effective_hard_flat = min(frozen_22_45_UTC, scheduled_market_close)",
        "scheduled_close_by_date": {"2026-08-14": "21:00:00 UTC"},
        "aug_14_effective_hard_flat_utc": "21:00:00",
        "liquidation_window": "[effective_hard_flat - 1 second, effective_hard_flat] inclusive",
        "no_invented_post_close_bbo": True,
        "open_position_behavior": "last causally valid executable native BBO inside the liquidation window; fail closed if unavailable",
        "strategy_contract_changed": False,
    }


def _expected_files() -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for day in TARGET_DATES:
        expected[f"es_mbp10/ESU6_{day}_130000_224501_mbp10.dbn.zst"] = {
            "label": "ES_MBP10_USD", "schema": "mbp-10", "symbol": "ESU6", "session_date": day,
            "start": f"{day}T13:00:00Z", "end": f"{day}T22:45:01Z",
        }
        expected[f"mes_mbp1/MESU6_{day}_133000_224501_mbp1.dbn.zst"] = {
            "label": "MES_MBP1_USD", "schema": "mbp-1", "symbol": "MESU6", "session_date": day,
            "start": f"{day}T13:30:00Z", "end": f"{day}T22:45:01Z",
        }
        prior = PRIOR_RTH[day]
        expected[f"es_prior_rth_trades/ESU6_{prior}_133000_200000_trades.dbn.zst"] = {
            "label": "PRIOR_RTH_TRADES_USD", "schema": "trades", "symbol": "ESU6", "session_date": prior,
            "start": f"{prior}T13:30:00Z", "end": f"{prior}T20:00:00Z",
        }
    return expected


def verify_acquisition_manifest(data_root: Path) -> dict[str, Any]:
    """Hash-and-identity preflight only; this function never opens a DBN."""
    if v3_contract_sha256() != "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4":
        raise V3FreshReplayError("frozen V3 strategy contract hash mismatch")
    path = data_root / "acquisition-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3FreshReplayError("missing or unreadable sealed V3 acquisition manifest") from exc
    identity = manifest.get("request_identity", {})
    if (manifest.get("manifest_kind") != "AUGUST_2026_L2_V3_POC_ONLY_FRESH_ACQUISITION" or
            manifest.get("data_acquired") is not True or manifest.get("strategy_replay_executed") is not False or
            manifest.get("outcomes_inspected") is not False or identity.get("strategy_id") != STRATEGY_ID or
            identity.get("v3_contract_sha256") != v3_contract_sha256() or
            identity.get("fresh_rth_dates") != list(TARGET_DATES) or identity.get("prior_rth_mapping") != PRIOR_RTH or
            identity.get("mbo_purchased") is not False):
        raise V3FreshReplayError("V3 acquisition manifest does not bind the frozen fresh package")
    expected, files = _expected_files(), manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise V3FreshReplayError("V3 acquisition manifest has missing, extra, or unordered declared inputs")
    verified: list[dict[str, Any]] = []
    for relative, required in expected.items():
        record, local = files.get(relative), data_root / relative
        if not isinstance(record, dict) or not local.is_file() or any(record.get(key) != value for key, value in required.items()):
            raise V3FreshReplayError(f"V3 input identity mismatch: {relative}")
        if local.stat().st_size <= 0 or local.stat().st_size != record.get("bytes") or _sha256(local) != record.get("sha256"):
            raise V3FreshReplayError(f"V3 input hash/size mismatch: {relative}")
        verified.append({"relative_path": relative, "bytes": local.stat().st_size, "sha256": record["sha256"]})
    counts = Counter(str(row["label"]) for row in files.values())
    if counts != Counter({"ES_MBP10_USD": 5, "MES_MBP1_USD": 5, "PRIOR_RTH_TRADES_USD": 5}):
        raise V3FreshReplayError("V3 acquisition manifest component cardinality is not 5/5/5")
    return {
        "preflight_only": True, "manifest_path": str(path), "manifest_sha256": _sha256(path),
        "files_verified": len(verified), "by_label": dict(counts), "target_dates": list(TARGET_DATES),
        "v3_contract_sha256": v3_contract_sha256(), "strategy_or_outcomes_accessed": False, "files": verified,
    }


def _code(value: object) -> str:
    return str(getattr(value, "value", value)).rsplit(".", 1)[-1]


def _timestamp(record: object) -> int:
    value = getattr(record, "ts_recv", None)
    return int(getattr(record, "ts_event") if value is None else value)


def _native_price(raw: object, *, context: str) -> float:
    value = int(raw)
    if value <= 0 or value >= historical.UNDEF_PRICE:
        raise V3FreshReplayError(f"invalid native MBP-10 fixed-point price: {context}")
    return value / RAW_PRICE_SCALE


class NativeMBP10Adapter:
    """Map native aggregate MBP-10 records directly into public L2 objects."""

    def __init__(self) -> None:
        self.previous: MBP10Snapshot | None = None
        self.first_valid_book_ns: int | None = None
        self.state = "UNINITIALIZED"
        self.temporary_non_executable_started_ns: int | None = None
        self.temporary_non_executable_records = 0
        self.last_transient: dict[str, int] | None = None

    @staticmethod
    def _side_levels(raw_levels: object, side: str) -> tuple[MBPLevel, ...]:
        rows: list[MBPLevel] = []
        for level in tuple(raw_levels):
            price = int(getattr(level, "bid_px" if side == "B" else "ask_px", 0))
            size = int(getattr(level, "bid_sz" if side == "B" else "ask_sz", 0))
            count = int(getattr(level, "bid_ct" if side == "B" else "ask_ct", 0))
            if price == 0 and size == 0 and count == 0:
                continue
            if price <= 0 or size < 0 or count < 0:
                raise V3FreshReplayError("malformed native MBP-10 aggregate level")
            rows.append(MBPLevel(price / RAW_PRICE_SCALE, size, count))
        return tuple(rows)

    def feed(self, record: object, *, expected_non_executable: bool = False) -> historical.PublicBookEvent | None:
        timestamp, action, side = _timestamp(record), _code(getattr(record, "action", "")), _code(getattr(record, "side", ""))
        if action not in {"A", "C", "M", "R", "T"} or side not in {"A", "B", "N"}:
            raise V3FreshReplayError(f"unsupported native MBP-10 action/side: {action}/{side}")
        snapshot = MBP10Snapshot(timestamp, self._side_levels(getattr(record, "levels", ()), "B"),
                                  self._side_levels(getattr(record, "levels", ()), "A"))
        executable = bool(snapshot.bids and snapshot.asks and snapshot.asks[0].price > snapshot.bids[0].price)
        if expected_non_executable:
            self.temporary_non_executable_started_ns = None
            self.temporary_non_executable_records = 0
            self.state, self.previous = "NON_EXECUTABLE_EXPECTED", None
            return None
        if not executable:
            # Native MBP-10 can publish bounded locked/crossed reconstruction
            # sequences under different ordinary update actions. A finite,
            # two-sided aggregate book identifies that source condition; the
            # action code does not. Once suspended, consume the complete native
            # sequence while exposing no BBO until a fresh uncrossed snapshot.
            two_sided = bool(snapshot.bids and snapshot.asks)
            if self.state == "TEMPORARILY_NON_EXECUTABLE" or (
                    self.state == "EXECUTABLE" and two_sided):
                if self.state != "TEMPORARILY_NON_EXECUTABLE":
                    self.temporary_non_executable_started_ns = timestamp
                    self.temporary_non_executable_records = 0
                self.temporary_non_executable_records += 1
                self.state, self.previous = "TEMPORARILY_NON_EXECUTABLE", None
                return None
            # MBP trade/control records need not carry a complete executable
            # book.  They can never use a stale BBO; wait for a valid reopen.
            if action in {"T", "R"}:
                self.state, self.previous = "WAITING_FOR_REOPEN_BOOK", None
                return None
            if self.state == "UNINITIALIZED":
                return None
            raise V3FreshReplayError("unexpected non-executable native MBP-10 book during active trading")
        if self.first_valid_book_ns is None:
            self.first_valid_book_ns = timestamp
        if self.state == "TEMPORARILY_NON_EXECUTABLE":
            assert self.temporary_non_executable_started_ns is not None
            self.last_transient = {
                "start_timestamp_ns": self.temporary_non_executable_started_ns,
                "reopen_timestamp_ns": timestamp,
                "non_executable_records": self.temporary_non_executable_records,
            }
            self.temporary_non_executable_started_ns = None
            self.temporary_non_executable_records = 0
        self.state = "EXECUTABLE"
        update: MBP10Update | None = None
        if action in {"A", "C", "M"} and side in {"A", "B"}:
            price = _native_price(getattr(record, "price", 0), context="book update")
            before = self.previous.level_at(side, price) if self.previous else None
            after = snapshot.level_at(side, price)
            size_delta = (after.size if after else 0) - (before.size if before else 0)
            count_delta = (after.order_count if after else 0) - (before.order_count if before else 0)
            update = MBP10Update(timestamp, side, price, size_delta, count_delta,
                                 {"A": "ADD", "C": "CANCEL", "M": "MODIFY"}[action])
        execution: Execution | None = None
        if action == "T":
            aggressor = "BUY" if side == "B" else "SELL" if side == "A" else "UNKNOWN"
            execution = Execution(timestamp, _native_price(getattr(record, "price", 0), context="trade"),
                                  int(getattr(record, "size", 0)), aggressor)  # type: ignore[arg-type]
        self.previous = snapshot
        return historical.PublicBookEvent(timestamp, snapshot, update, execution)

    def assert_executable_at_boundary(self) -> None:
        if self.first_valid_book_ns is not None and self.state != "EXECUTABLE":
            raise V3FreshReplayError(f"native MBP-10 remained {self.state} at source boundary")

    def assert_initialized_before(self, start_ns: int) -> None:
        if self.first_valid_book_ns is None or self.first_valid_book_ns >= start_ns:
            raise V3FreshReplayError("native MBP-10 lacks a valid two-sided top-10 initialization before 13:30 UTC")


def _stream_native_mbp10_records(path: Path) -> Iterator[object]:
    from databento import DBNStore
    for record in DBNStore.from_file(path):
        yield record


def _profile_poc(path: Path) -> StructuralLevel:
    levels = historical._profile_levels_from_declared_trades(path)
    selected = [level for level in levels if level.name in ELIGIBLE_STRUCTURAL_LEVELS]
    if len(selected) != 1 or selected[0].name != "PRIOR_RTH_POC":
        raise V3FreshReplayError("prior-RTH profile did not produce exactly one eligible POC level")
    return selected[0]


def _paths(data_root: Path, day: str) -> tuple[Path, Path, Path]:
    return (
        data_root / "es_mbp10" / f"ESU6_{day}_130000_224501_mbp10.dbn.zst",
        data_root / "mes_mbp1" / f"MESU6_{day}_133000_224501_mbp1.dbn.zst",
        data_root / "es_prior_rth_trades" / f"ESU6_{PRIOR_RTH[day]}_133000_200000_trades.dbn.zst",
    )


def _in_maintenance_pause(timestamp_ns: int, day: str) -> bool:
    seconds = (timestamp_ns - historical._clock_ns(day, 0)) // 1_000_000_000
    return GLOBEX_MAINTENANCE_PAUSE[0] <= seconds < GLOBEX_MAINTENANCE_PAUSE[1]


def _begin_non_executable_state(runner: historical.HistoricalL2Runner, timestamp_ns: int) -> None:
    """Remove executable quotes without inventing a pause-crossing decision."""
    if runner.signals.position is not None:
        raise V3FreshReplayError("MAINTENANCE_HALT_WITH_OPEN_POSITION_UNDEFINED_BY_FROZEN_CONTRACT")
    for setup in runner.signals.pending.values():
        if setup.state == "CONFIRMED" and setup.terminal_reason is None:
            setup.state, setup.terminal_reason = "FAILED", "SOURCE_NON_EXECUTABLE_BEFORE_ENTRY"
    runner.es_quote = runner.mes_quote = None
    runner.es_quote_timestamp_ns = runner.mes_quote_timestamp_ns = None
    runner.diagnostic_events.append({"event": "NON_EXECUTABLE_BOOK", "timestamp_ns": timestamp_ns})


def _begin_temporary_non_executable_state(runner: historical.HistoricalL2Runner, timestamp_ns: int) -> None:
    """Suspend all execution without interpreting one transient source row."""
    if getattr(runner, "_native_transient_started_ns", None) is None:
        runner._native_transient_started_ns = timestamp_ns
        runner._native_transient_position_open = runner.signals.position is not None
        runner.diagnostic_events.append({"event": "TEMPORARILY_NON_EXECUTABLE_BOOK", "timestamp_ns": timestamp_ns})
    runner.es_quote = runner.mes_quote = None
    runner.es_quote_timestamp_ns = runner.mes_quote_timestamp_ns = None


def _resume_temporary_non_executable_state(runner: historical.HistoricalL2Runner, timestamp_ns: int) -> None:
    started = getattr(runner, "_native_transient_started_ns", None)
    if started is None:
        return
    position_was_open = bool(getattr(runner, "_native_transient_position_open", False))
    del runner._native_transient_started_ns
    del runner._native_transient_position_open
    if position_was_open or runner.signals.position is not None:
        raise V3FreshReplayError("temporary native MBP-10 outage overlapped an open position")
    runner.diagnostic_events.append({
        "event": "EXECUTABLE_BOOK_REOPENED", "timestamp_ns": timestamp_ns,
        "non_executable_duration_ns": timestamp_ns - started,
    })


def _run_session(day: str, data_root: Path) -> historical.HistoricalL2Runner:
    es_path, mes_path, profile_path = _paths(data_root, day)
    runner = historical.HistoricalL2Runner(
        date=day, evidence_label=EVIDENCE_LABEL, levels=[_profile_poc(profile_path)],
        config=V2_CONFIG,
        strategy_id=STRATEGY_ID, require_native_mes_for_fallback=True,
    )
    adapter = NativeMBP10Adapter()
    es_iter, mes_iter = iter(_stream_native_mbp10_records(es_path)), iter(historical._stream_mes_quotes(mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    start_ns, cutoff_ns = historical._clock_ns(day, historical.RTH_START_SECONDS), historical._clock_ns(day, effective_hard_flat_seconds(day))
    initialized, closed, pause_active, records = False, False, False, 0
    while es is not None or mes is not None:
        next_es = _timestamp(es) if es is not None else None
        next_mes = mes[0] if mes is not None else None
        timestamp = min(item for item in (next_es, next_mes) if item is not None)
        if timestamp >= cutoff_ns:
            if not initialized:
                raise V3FreshReplayError("native MBP-10 initialization absent before RTH")
            adapter.assert_executable_at_boundary()
            runner.force_flat_from_last_causal_cutoff_quote(
                cutoff_ns,
                exit_reason="HARD_FLAT_SCHEDULED_CLOSE_2100" if day in SCHEDULED_CLOSE_SECONDS else "HARD_CUTOFF_2245",
            )
            runner.finish(cutoff_ns); closed = True; break
        if _in_maintenance_pause(timestamp, day):
            if not pause_active:
                _begin_non_executable_state(runner, timestamp); pause_active = True
            if next_mes is not None and next_mes <= (next_es if next_es is not None else next_mes):
                mes = historical._next(mes_iter)
            else:
                assert es is not None
                adapter.feed(es, expected_non_executable=True); es = historical._next(es_iter)
            continue
        if next_mes is not None and next_mes <= (next_es if next_es is not None else next_mes):
            if getattr(runner, "_native_transient_started_ns", None) is None:
                runner.observe_mes_quote(*mes)
            mes = historical._next(mes_iter)
            continue
        assert es is not None
        public = adapter.feed(es)
        if next_es < start_ns:
            initialized = adapter.first_valid_book_ns is not None and adapter.first_valid_book_ns < start_ns
            es = historical._next(es_iter); continue
        if not initialized:
            raise V3FreshReplayError("native MBP-10 initialization absent before first RTH event")
        if public is not None:
            _resume_temporary_non_executable_state(runner, next_es)
            runner.observe_public(public); records += 1
        elif adapter.state == "TEMPORARILY_NON_EXECUTABLE":
            _begin_temporary_non_executable_state(runner, next_es)
        else:
            _begin_non_executable_state(runner, next_es)
        if records % 1_000_000 == 0:
            print(f"  V3 native MBP-10 {day} records={records:,}", flush=True)
        es = historical._next(es_iter)
    if not closed:
        if not initialized:
            raise V3FreshReplayError("native MBP-10 initialization absent before RTH")
        adapter.assert_executable_at_boundary()
        runner.force_flat_from_last_causal_cutoff_quote(
            cutoff_ns, exit_reason="HARD_FLAT_SCHEDULED_CLOSE_2100" if day in SCHEDULED_CLOSE_SECONDS else "HARD_CUTOFF_2245",
        )
        runner.finish(cutoff_ns)
    return runner


def _rows(runners: list[historical.HistoricalL2Runner], name: str) -> list[dict[str, Any]]:
    return [row for runner in runners for row in getattr(runner, name)]


def _write_results(output_root: Path, runners: list[historical.HistoricalL2Runner], verification: dict[str, Any]) -> dict[str, Any]:
    base = historical.write_future_artifacts(output_root, runners, contract=v3_contract())
    trades, setups = _rows(runners, "trade_ledger"), _rows(runners, "setup_ledger")
    metrics = historical._performance(trades)
    result = {
        **base, "strategy_id": STRATEGY_ID, "evidence_label": EVIDENCE_LABEL,
        "v3_contract_sha256": v3_contract_sha256(), "input_verification": verification,
        "execution_calendar": calendar_contract(), "metrics": {
            "completed_interactions": sum(len(item.interaction_ledger) for item in runners),
            "accepted_setups": sum(bool(row["accepted"]) for row in setups),
            "confirmations_passed": sum(row.get("confirmation_timestamp_ns") is not None for row in setups),
            **metrics,
        }, "strategy_replay_completed_all_five_sessions": True,
        "first_run_policy": "FIRST_PREDECLARED_FRESH_L2_V3_POC_ONLY_REPLAY; NO_OUTCOME_BASED_PARAMETER_SELECTION",
        "outcome_parameter_selection": False,
    }
    (output_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "diagnostic-report.md").write_text(
        f"# {STRATEGY_ID} — first fresh validation replay\n\n"
        f"Evidence label: `{EVIDENCE_LABEL}`.\n\n"
        "This is the first predeclared fresh POC-only V3 block. No comparison, early stopping, or parameter adjustment is permitted.\n\n"
        f"Calendar classification: `{CALENDAR_CLARIFICATION}`; August 14 effective hard-flat is 21:00 UTC using the last valid executable BBO in its inclusive one-second liquidation window.\n",
        encoding="utf-8",
    )
    return result


def run(*, data_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("fresh V3 output directory already exists")
    verification = verify_acquisition_manifest(data_root)
    runners: list[historical.HistoricalL2Runner] = []
    for index, day in enumerate(TARGET_DATES, start=1):
        print(f"=== V3 FRESH {index:02d}/{len(TARGET_DATES)} {day} ===", flush=True)
        runners.append(_run_session(day, data_root))
    return _write_results(output_root, runners, verification)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the first sealed native-L2 V3 fresh POC-only replay locally")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true", help="hash-verify inputs only; never parse DBNs or run strategy")
    args = parser.parse_args(argv)
    try:
        result = verify_acquisition_manifest(args.data_root) if args.preflight_only else run(data_root=args.data_root, output_root=args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (V3FreshReplayError, historical.HistoricalReplayError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
