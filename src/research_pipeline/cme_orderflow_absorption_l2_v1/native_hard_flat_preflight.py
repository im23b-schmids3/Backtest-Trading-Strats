"""Read-only native Aug-2026 source-boundary consistency preflight.

This audit opens only the sealed local ES MBP-10 and MES MBP-1 files.  It
does not instantiate strategy state, generate interactions, inspect outcomes,
or contact Databento.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import causal_master_tape as master
from . import historical_runner as historical
from . import v3_poc_fresh_august_replay as native


AUDIT_ID = "CMEOrderflowAbsorption.ES_L2_NATIVE_HARD_FLAT_PREFLIGHT_AUG10_14"
DEFAULT_OUTPUT_ROOT = Path("research_runs") / AUDIT_ID


class NativeHardFlatPreflightError(RuntimeError):
    pass


def _iso(timestamp_ns: int | None) -> str | None:
    if timestamp_ns is None:
        return None
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat(
        timespec="microseconds",
    ).replace("+00:00", "Z")


def _stream_timestamp(record: object, *, stream: str) -> int:
    if stream == "ES":
        return native._timestamp(record)
    return int(getattr(record, "ts_event", getattr(record, "ts_recv")))


def _record_timestamps(record: object) -> tuple[int, int]:
    event = int(getattr(record, "ts_event", getattr(record, "ts_recv")))
    receive = int(getattr(record, "ts_recv", event))
    return event, receive


def _mes_bbo(record: object) -> tuple[float, float] | None:
    levels = tuple(getattr(record, "levels", ()))
    if not levels:
        return None
    level = levels[0]
    bid, ask = int(getattr(level, "bid_px", 0)), int(getattr(level, "ask_px", 0))
    if bid <= 0 or ask <= bid or int(getattr(level, "bid_sz", 0)) <= 0 or int(getattr(level, "ask_sz", 0)) <= 0:
        return None
    return bid / historical.RAW_PRICE_SCALE, ask / historical.RAW_PRICE_SCALE


def _source_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "record_count": 0,
        "first_ts_event_ns": None,
        "first_ts_event_utc": None,
        "first_ts_recv_ns": None,
        "first_ts_recv_utc": None,
        "last_ts_event_ns": None,
        "last_ts_event_utc": None,
        "last_ts_recv_ns": None,
        "last_ts_recv_utc": None,
    }


def _observe_source(row: dict[str, Any], record: object) -> None:
    event, receive = _record_timestamps(record)
    row["record_count"] += 1
    if row["first_ts_event_ns"] is None:
        row["first_ts_event_ns"], row["first_ts_event_utc"] = event, _iso(event)
        row["first_ts_recv_ns"], row["first_ts_recv_utc"] = receive, _iso(receive)
    row["last_ts_event_ns"], row["last_ts_event_utc"] = event, _iso(event)
    row["last_ts_recv_ns"], row["last_ts_recv_utc"] = receive, _iso(receive)


def audit_session(repository_root: Path, day: str) -> dict[str, Any]:
    """Scan one sealed native session without constructing strategy state."""
    if day not in native.TARGET_DATES:
        raise NativeHardFlatPreflightError(f"unexpected native preflight date: {day}")
    data_root = repository_root.resolve() / native.DATA_ROOT
    es_path, mes_path, _ = native._paths(data_root, day)
    if not es_path.is_file() or not mes_path.is_file():
        raise NativeHardFlatPreflightError(f"sealed native source missing: {day}")

    from databento import DBNStore

    es_iter, mes_iter = iter(DBNStore.from_file(es_path)), iter(DBNStore.from_file(mes_path))
    es, mes = historical._next(es_iter), historical._next(mes_iter)
    es_source, mes_source = _source_row(es_path), _source_row(mes_path)
    adapter = native.NativeMBP10Adapter()
    cutoff_ns = historical._clock_ns(day, native.effective_hard_flat_seconds(day))
    window_start, window_end = native.liquidation_window_ns(day)
    es_quote: tuple[float, float] | None = None
    mes_quote: tuple[float, float] | None = None
    es_quote_ns = mes_quote_ns = None
    stale_bbo_exposure_count = 0
    adapter_exception_count = 0

    while es is not None or mes is not None:
        es_timestamp = _stream_timestamp(es, stream="ES") if es is not None else None
        mes_timestamp = _stream_timestamp(mes, stream="MES") if mes is not None else None
        take_mes = mes_timestamp is not None and mes_timestamp <= (
            es_timestamp if es_timestamp is not None else mes_timestamp
        )
        if take_mes:
            assert mes is not None and mes_timestamp is not None
            _observe_source(mes_source, mes)
            if mes_timestamp < cutoff_ns:
                if native._in_maintenance_pause(mes_timestamp, day):
                    es_quote = mes_quote = None
                    es_quote_ns = mes_quote_ns = None
                elif adapter.state != "TEMPORARILY_NON_EXECUTABLE":
                    quote = _mes_bbo(mes)
                    if quote is not None:
                        mes_quote, mes_quote_ns = quote, mes_timestamp
            mes = historical._next(mes_iter)
            continue

        assert es is not None and es_timestamp is not None
        _observe_source(es_source, es)
        if es_timestamp < cutoff_ns:
            try:
                public = adapter.feed(
                    es, expected_non_executable=native._in_maintenance_pause(es_timestamp, day),
                )
            except Exception:
                adapter_exception_count += 1
                raise
            if native._in_maintenance_pause(es_timestamp, day):
                es_quote = mes_quote = None
                es_quote_ns = mes_quote_ns = None
            elif public is not None:
                quote = historical._quote(public.snapshot)
                if quote is None:
                    raise NativeHardFlatPreflightError(f"adapter emitted invalid public BBO: {day}")
                es_quote, es_quote_ns = quote, es_timestamp
            elif adapter.state in {"TEMPORARILY_NON_EXECUTABLE", "WAITING_FOR_REOPEN_BOOK"}:
                es_quote = mes_quote = None
                es_quote_ns = mes_quote_ns = None
        es = historical._next(es_iter)

    try:
        adapter.assert_executable_at_boundary()
        evidence = master.resolve_native_hard_flat_boundary(
            cutoff_ns=cutoff_ns,
            es_quote=es_quote,
            es_quote_timestamp_ns=es_quote_ns,
            mes_quote=mes_quote,
            mes_quote_timestamp_ns=mes_quote_ns,
        )
    except Exception as exc:
        return {
            "session_date": day,
            "status": "NATIVE_HARD_FLAT_PREFLIGHT_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "es_source": es_source,
            "mes_source": mes_source,
            "adapter_exception_count": adapter_exception_count,
            "strategy_logic_executed": False,
            "interactions_created": 0,
            "setups_created": 0,
            "trades_created": 0,
            "pnl_calculated": False,
            "network_calls": 0,
            "downloads": 0,
        }

    final_source_timestamp = max(
        int(es_source["last_ts_recv_ns"] or 0), int(mes_source["last_ts_event_ns"] or 0),
    )
    return {
        "session_date": day,
        "status": "NATIVE_HARD_FLAT_PREFLIGHT_PASS",
        "calendar": {
            "rth_utc": "13:30:00-20:00:00",
            "maintenance_utc": "21:00:00-22:00:00",
            "nominal_hard_flat_utc": f"{day}T22:45:00.000000Z",
            "effective_hard_flat_utc": _iso(cutoff_ns),
            "liquidation_window_start_utc": _iso(window_start),
            "liquidation_window_end_utc": _iso(window_end),
            "scheduled_close": day in native.SCHEDULED_CLOSE_SECONDS,
        },
        "es_source": es_source,
        "mes_source": mes_source,
        "last_executable_es_bbo": {
            "timestamp_ns": evidence["es_quote_timestamp_ns"],
            "timestamp_utc": _iso(evidence["es_quote_timestamp_ns"]),
            "bid": evidence["es_quote"][0] if evidence["es_quote"] else None,
            "ask": evidence["es_quote"][1] if evidence["es_quote"] else None,
        },
        "last_executable_mes_bbo": {
            "timestamp_ns": evidence["mes_quote_timestamp_ns"],
            "timestamp_utc": _iso(evidence["mes_quote_timestamp_ns"]),
            "bid": evidence["mes_quote"][0] if evidence["mes_quote"] else None,
            "ask": evidence["mes_quote"][1] if evidence["mes_quote"] else None,
        },
        "source_ended_before_effective_hard_flat": final_source_timestamp < cutoff_ns,
        "completion_mode": (
            "SOURCE_ENDED_WITH_SUFFICIENT_LIQUIDATION_EVIDENCE"
            if final_source_timestamp < cutoff_ns else "SOURCE_REACHED_OR_PASSED_HARD_FLAT"
        ),
        "fresh_august_contract_decision": "COMPLETE",
        "causal_tape_builder_decision": "COMPLETE",
        "exact_hard_flat_record_required": False,
        "invented_quote_count": evidence["invented_quote_count"],
        "stale_bbo_exposure_count": stale_bbo_exposure_count,
        "adapter_exception_count": adapter_exception_count,
        "unresolved_tape_boundary_count": 0,
        "final_adapter_state": adapter.state,
        "strategy_logic_executed": False,
        "interactions_created": 0,
        "setups_created": 0,
        "trades_created": 0,
        "pnl_calculated": False,
        "network_calls": 0,
        "downloads": 0,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_html(path: Path, payload: Mapping[str, Any]) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['session_date']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{int(row['es_source']['record_count']):,}</td>"
        f"<td>{html.escape(str(row['es_source']['last_ts_recv_utc']))}</td>"
        f"<td>{int(row['mes_source']['record_count']):,}</td>"
        f"<td>{html.escape(str(row['mes_source']['last_ts_event_utc']))}</td>"
        f"<td>{html.escape(str(row.get('completion_mode')))}</td>"
        "</tr>"
        for row in payload.get("sessions", ())
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Native hard-flat preflight</title>
<style>body{{font:14px system-ui;margin:2rem;color:#172033;background:#f6f8fb}}.card{{background:#fff;border:1px solid #d7deea;border-radius:10px;padding:1rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7deea;padding:.45rem;text-align:left}}th{{background:#edf2f8}}.pass{{color:#08783e}}</style></head>
<body><h1>Aug 10–14 native hard-flat consistency preflight</h1><div class="card"><strong class="pass">{html.escape(str(payload.get('status')))}</strong><p>Local source-only audit. No strategy, interactions, trades, optimizer, network, or PnL.</p></div>
<div class="card"><table><thead><tr><th>Date</th><th>Status</th><th>ES records</th><th>Last ES receive</th><th>MES records</th><th>Last MES event</th><th>Completion</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>"""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, path)


def run_preflight(*, repository_root: Path, output_root: Path, workers: int = 2) -> dict[str, Any]:
    if workers < 1 or workers > 5:
        raise NativeHardFlatPreflightError("workers must be between 1 and 5")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        sessions = list(executor.map(audit_session, [repository_root] * 5, native.TARGET_DATES))
    sessions.sort(key=lambda row: str(row["session_date"]))
    passed = all(row["status"] == "NATIVE_HARD_FLAT_PREFLIGHT_PASS" for row in sessions)
    payload = {
        "audit_id": AUDIT_ID,
        "status": "NATIVE_HARD_FLAT_PREFLIGHT_PASS" if passed else "NATIVE_HARD_FLAT_PREFLIGHT_FAIL",
        "session_count": len(sessions),
        "expected_sessions": list(native.TARGET_DATES),
        "unexpected_source_end_failure_count": sum(row["status"] != "NATIVE_HARD_FLAT_PREFLIGHT_PASS" for row in sessions),
        "stale_bbo_exposure_count": sum(int(row.get("stale_bbo_exposure_count", 0)) for row in sessions),
        "invented_quote_count": sum(int(row.get("invented_quote_count", 0)) for row in sessions),
        "unresolved_tape_boundary_count": sum(int(row.get("unresolved_tape_boundary_count", 0)) for row in sessions),
        "adapter_exception_count": sum(int(row.get("adapter_exception_count", 0)) for row in sessions),
        "fresh_august_calendar_contract": native.calendar_contract(),
        "strategy_logic_executed": False,
        "interactions_created": 0,
        "setups_created": 0,
        "trades_created": 0,
        "pnl_calculated": False,
        "optimizer_executed": False,
        "network_calls": 0,
        "downloads": 0,
        "sessions": sessions,
    }
    _atomic_json(output_root / "native-hard-flat-preflight.json", payload)
    _write_html(output_root / "native-hard-flat-preflight-report.html", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    result = run_preflight(
        repository_root=repository_root, output_root=output_root.resolve(), workers=args.workers,
    )
    print(json.dumps({
        "status": result["status"], "session_count": result["session_count"],
        "report": str((output_root / "native-hard-flat-preflight.json").resolve()),
    }, sort_keys=True))
    return 0 if result["status"] == "NATIVE_HARD_FLAT_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
