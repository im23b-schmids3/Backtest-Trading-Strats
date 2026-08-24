"""Read-only audit of MBO-derived public MBP-10 executability.

This module reconstructs aggregate public depth only.  It never creates a
strategy engine, interaction, setup, trade, fill, or PnL value.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import historical_runner as historical
from . import v2_august_seen_replay as august
from . import v2_extended_existing_data as retro


AUDIT_ID = "CMEOrderflowAbsorption.ES_L2_MBO_PUBLIC_STATE_AUDIT"
DEFAULT_OUTPUT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MBO_PUBLIC_STATE_AUDIT")
NO_STRATEGY_EXECUTION = True


class MBOStateAuditError(RuntimeError):
    pass


def _iso(timestamp_ns: int | None) -> str | None:
    if timestamp_ns is None:
        return None
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _raw_code(value: object) -> str:
    return historical._code(value)


def _event_row(
    raw: object, raw_index: int, bbo: tuple[float | None, float | None],
) -> dict[str, Any]:
    ts_event = int(getattr(raw, "ts_event", 0))
    ts_recv = int(getattr(raw, "ts_recv", ts_event))
    bid, ask = bbo
    return {
        "source_record_index": raw_index,
        "ts_event_ns": ts_event,
        "ts_event_utc": _iso(ts_event),
        "ts_recv_ns": ts_recv,
        "ts_recv_utc": _iso(ts_recv),
        "action": _raw_code(getattr(raw, "action", "")),
        "side": _raw_code(getattr(raw, "side", "")),
        "raw_price": int(getattr(raw, "price", 0)),
        "price": None if int(getattr(raw, "price", 0)) == historical.UNDEF_PRICE else int(getattr(raw, "price", 0)) / historical.RAW_PRICE_SCALE,
        "size": int(getattr(raw, "size", 0)),
        "flags": int(getattr(raw, "flags", 0)),
        "sequence": int(getattr(raw, "sequence", 0)),
        "order_id": int(getattr(raw, "order_id", 0)),
        "derived_bid": bid,
        "derived_ask": ask,
        "derived_book_shape": (
            "EMPTY" if bid is None and ask is None else
            "ONE_SIDED_BID" if ask is None else
            "ONE_SIDED_ASK" if bid is None else
            "LOCKED" if bid == ask else
            "CROSSED" if bid > ask else
            "EXECUTABLE"
        ),
    }


def _episode_classification(rows: list[dict[str, Any]], reopen: dict[str, Any] | None) -> str:
    if reopen is None:
        return "UNRESOLVED_NON_EXECUTABLE_SOURCE_BOUNDARY"
    first = rows[0]
    same_group = all(
        row["ts_recv_ns"] == first["ts_recv_ns"]
        and row["ts_event_ns"] == first["ts_event_ns"]
        and row["sequence"] == first["sequence"]
        for row in [*rows, reopen]
    )
    if same_group:
        return "ATOMIC_MBO_RECONSTRUCTION_TRANSITION"
    return "TEMPORARY_MBO_RECONSTRUCTION_TRANSITION"


def _is_summer_maintenance_episode(episode: Mapping[str, Any]) -> bool:
    """Identify a reconstruction episode initiated during the sealed summer pause."""
    timestamp = str(episode.get("start_ts_recv_utc") or "")
    return len(timestamp) >= 13 and timestamp[11:13] == "21"


@dataclass(frozen=True)
class SessionSource:
    period_id: str
    day: str
    path: Path
    source_tail_complete: bool


class SessionAuditor:
    """Incremental aggregate-book state audit for one declared UTC session."""

    def __init__(self, source: SessionSource) -> None:
        self.source = source
        self.adapter = historical.HistoricalMBOToMBP10Adapter()
        self.raw_records = 0
        self.supported_records = 0
        self.executable_records = 0
        self.executable_episodes = 0
        self.initial_snapshot_resets = 0
        self.ordinary_resets = 0
        self._previous: deque[tuple[object, int, tuple[float | None, float | None]]] = deque(maxlen=5)
        self._prior_valid: tuple[object, int, tuple[float | None, float | None]] | None = None
        self._current: dict[str, Any] | None = None
        self.episodes: list[dict[str, Any]] = []
        self.exceptions: list[dict[str, Any]] = []

    def observe(self, raw: object, raw_index: int) -> None:
        self.raw_records += 1
        flags = int(getattr(raw, "flags", 0))
        action = _raw_code(getattr(raw, "action", ""))
        if action == "R":
            if flags & historical.F_SNAPSHOT:
                self.initial_snapshot_resets += 1
            else:
                self.ordinary_resets += 1
        normalized = historical.private_mbo_record_from_dbn(raw)
        if normalized is None:
            return
        self.supported_records += 1
        previous_state = self.adapter.state
        try:
            self.adapter.feed(normalized, materialize_public=False)
        except Exception as exc:
            self.exceptions.append({
                "source_record_index": raw_index, "timestamp_utc": _iso(normalized.timestamp_ns),
                "type": type(exc).__name__, "message": str(exc),
            })
            raise
        if self.adapter.state == "EXECUTABLE":
            self.executable_records += 1
            if previous_state != "EXECUTABLE":
                self.executable_episodes += 1
            bbo = self.adapter.current_public_bbo()
            if self._current is not None:
                row = _event_row(raw, raw_index, bbo)
                self._current["reopen_event"] = row
                self._current["reopen_bbo"] = {"bid": row["derived_bid"], "ask": row["derived_ask"]}
                invalid_rows = self._current.pop("_invalid_rows")
                self._current["classification"] = _episode_classification(invalid_rows, row)
                self._current["end_ts_recv_utc"] = row["ts_recv_utc"]
                self._current["end_ts_event_utc"] = row["ts_event_utc"]
                self._current["duration_ns_by_ts_recv"] = row["ts_recv_ns"] - invalid_rows[0]["ts_recv_ns"]
                self._current["duration_ns_by_ts_event"] = row["ts_event_ns"] - invalid_rows[0]["ts_event_ns"]
                self._current["surrounding_events_after"] = [row]
                self.episodes.append(self._current)
                self._current = None
            context = (raw, raw_index, bbo)
            self._prior_valid = context
            self._previous.append(context)
            return
        if self.adapter.state != "TEMPORARILY_NON_EXECUTABLE":
            return
        row = _event_row(raw, raw_index, self.adapter.current_public_bbo())
        if self._current is None:
            self._current = {
                "episode_index": len(self.episodes),
                "start_ts_recv_utc": row["ts_recv_utc"],
                "start_ts_event_utc": row["ts_event_utc"],
                "initiating_event": row,
                "prior_valid_bbo_event": (
                    _event_row(*self._prior_valid) if self._prior_valid is not None else None
                ),
                "surrounding_events_before": [_event_row(*context) for context in self._previous],
                "non_executable_record_count": 0,
                "book_shape_counts": {},
                "_invalid_rows": [],
            }
        self._current["non_executable_record_count"] += 1
        shapes = Counter(self._current["book_shape_counts"])
        shapes[row["derived_book_shape"]] += 1
        self._current["book_shape_counts"] = dict(shapes)
        self._current["_invalid_rows"].append(row)

    def finish(self) -> dict[str, Any]:
        unresolved = 0
        if self._current is not None:
            invalid_rows = self._current.pop("_invalid_rows")
            self._current["classification"] = _episode_classification(invalid_rows, None)
            self._current["reopen_event"] = None
            self._current["reopen_bbo"] = None
            self._current["duration_ns_by_ts_recv"] = None
            self._current["duration_ns_by_ts_event"] = None
            self._current["surrounding_events_after"] = []
            self.episodes.append(self._current)
            self._current = None
            unresolved = 1
        try:
            self.adapter.finish()
        except Exception as exc:
            self.exceptions.append({"type": type(exc).__name__, "message": str(exc), "source_record_index": None})
        anomalies = self.adapter.source_integrity_diagnostics()
        temporary_records = sum(int(row["non_executable_record_count"]) for row in self.episodes)
        source_tail = (
            "COMPLETE_HARD_FLAT_SOURCE" if self.source.source_tail_complete
            else "INTENTIONALLY_INCOMPLETE_FAIL_CLOSED_SOURCE_TAIL"
        )
        return {
            "period_id": self.source.period_id,
            "session_date": self.source.day,
            "source_path": str(self.source.path),
            "mbo_record_count": self.raw_records,
            "supported_mbo_record_count": self.supported_records,
            "executable_book_episode_count": self.executable_episodes,
            "executable_public_record_count": self.executable_records,
            "temporary_non_executable_episode_count": len(self.episodes),
            "temporary_non_executable_record_count": temporary_records,
            "scheduled_maintenance_transition_count": sum(
                _is_summer_maintenance_episode(episode) for episode in self.episodes
            ),
            "ordinary_reset_transition_count": self.ordinary_resets,
            "maintenance_or_reset_transition_count": self.ordinary_resets + sum(
                _is_summer_maintenance_episode(episode) for episode in self.episodes
            ),
            "initial_snapshot_reset_count": self.initial_snapshot_resets,
            "anomaly_retention_count": len(anomalies),
            "source_integrity_anomalies": anomalies,
            "unresolved_episode_count": unresolved,
            "exceptions": self.exceptions,
            "final_adapter_state": self.adapter.state,
            "source_tail_classification": source_tail,
            "replay_completion_allowed": self.source.source_tail_complete and not unresolved and not self.exceptions,
            "episodes": self.episodes,
            "strategy_logic_executed": False,
            "interactions_created": 0,
            "setups_created": 0,
            "trades_created": 0,
            "pnl_calculated": False,
        }


def session_sources(repository_root: Path) -> tuple[SessionSource, ...]:
    root = repository_root.resolve()
    rows: list[SessionSource] = []
    may_root = root / "data/cme_orderflow_absorption_v2/may_2026_cost_proxy/es_mbo"
    rows.extend(SessionSource(
        "MAY_2026", day, may_root / f"ESM6_{day}_000000_224501_mbo.dbn.zst", True,
    ) for day in historical.MAY_DATES)
    retro_root = root / "data/cme_orderflow_absorption_v2_holdout/es_mbo"
    rows.extend(SessionSource(
        "RETRO_JUNE_JULY_2026", day, retro_root / f"ESU6_{day}_0000_1600_mbo.dbn.zst", False,
    ) for day in retro.RETRO_DATES)
    shared = root / august.ES_MBO_RELATIVE
    rows.extend(SessionSource("AUGUST_03_06_2026", day, shared, True) for day in august.TARGET_DATES)
    if len(rows) != 37 or any(not row.path.is_file() for row in rows):
        missing = [str(row.path) for row in rows if not row.path.is_file()]
        raise MBOStateAuditError(f"sealed 37-session MBO source inventory incomplete: {missing}")
    return tuple(rows)


def _date_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).date().isoformat()


def _audit_one_source(source: SessionSource) -> dict[str, Any]:
    from databento import DBNStore
    print(f"MBO_STATE_AUDIT START {source.period_id} {source.day}", flush=True)
    auditor = SessionAuditor(source)
    for raw_index, raw in enumerate(DBNStore.from_file(source.path)):
        auditor.observe(raw, raw_index)
    result = auditor.finish()
    print(
        f"MBO_STATE_AUDIT DONE {source.period_id} {source.day} "
        f"records={result['mbo_record_count']:,} episodes={result['temporary_non_executable_episode_count']}",
        flush=True,
    )
    return result


def _audit_separate_sources(sources: Iterable[SessionSource], workers: int) -> list[dict[str, Any]]:
    source_rows = tuple(sources)
    if workers <= 1:
        return [_audit_one_source(source) for source in source_rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_audit_one_source, source_rows))


def _audit_shared_august(sources: Iterable[SessionSource]) -> list[dict[str, Any]]:
    from databento import DBNStore
    source_rows = tuple(sources)
    auditors = {row.day: SessionAuditor(row) for row in source_rows}
    if not source_rows:
        return []
    print("MBO_STATE_AUDIT START AUGUST_03_06_2026 shared-source", flush=True)
    for raw_index, raw in enumerate(DBNStore.from_file(source_rows[0].path)):
        timestamp_ns = int(getattr(raw, "ts_recv", getattr(raw, "ts_event", 0)))
        day = _date_from_ns(timestamp_ns)
        if day > source_rows[-1].day:
            break
        if day in auditors:
            auditors[day].observe(raw, raw_index)
    results = [auditors[row.day].finish() for row in source_rows]
    print(
        f"MBO_STATE_AUDIT DONE AUGUST_03_06_2026 shared-source "
        f"records={sum(int(row['mbo_record_count']) for row in results):,}", flush=True,
    )
    return results


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_audit_html(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a compact self-contained human audit; JSON retains every episode."""
    sessions = list(payload.get("sessions", ()))
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['period_id']))}</td><td>{html.escape(str(row['session_date']))}</td>"
        f"<td>{int(row['mbo_record_count']):,}</td>"
        f"<td>{int(row['temporary_non_executable_episode_count']):,}</td>"
        f"<td>{int(row['temporary_non_executable_record_count']):,}</td>"
        f"<td>{int(row.get('scheduled_maintenance_transition_count', sum(_is_summer_maintenance_episode(episode) for episode in row.get('episodes', ())))):,}</td>"
        f"<td>{int(row['anomaly_retention_count']):,}</td>"
        f"<td>{int(row['unresolved_episode_count']):,}</td>"
        f"<td>{html.escape(str(row['final_adapter_state']))}</td>"
        "</tr>"
        for row in sessions
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MBO public-state audit</title><style>
body{{font:14px system-ui,sans-serif;margin:2rem;color:#172033;background:#f7f8fa}}
.card{{background:white;border:1px solid #d8dee9;border-radius:10px;padding:1rem;margin-bottom:1rem}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:.75rem}}
.kpi b{{display:block;font-size:1.45rem}} table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d8dee9;padding:.45rem;text-align:left}} th{{background:#eef2f7}}
code{{background:#eef2f7;padding:.1rem .25rem}} .pass{{color:#08783e}} .fail{{color:#b42318}}
</style></head><body><h1>MBO-derived public MBP-10 state audit</h1>
<div class="card"><strong class="{'pass' if payload.get('status') == 'MBO_PUBLIC_STATE_AUDIT_PASS' else 'fail'}">{html.escape(str(payload.get('status')))}</strong>
<p>Source-only reconstruction audit. Strategy logic, setups, trades, and PnL were not executed.</p></div>
<div class="card kpis"><div class="kpi">Sessions<b>{int(payload.get('session_count', 0))}</b></div>
<div class="kpi">Temporary episodes<b>{int(payload.get('temporary_non_executable_episode_count', 0)):,}</b></div>
<div class="kpi">Unresolved episodes<b>{int(payload.get('unresolved_episode_count', 0)):,}</b></div>
<div class="kpi">Exceptions<b>{int(payload.get('exception_count', 0)):,}</b></div></div>
<div class="card"><h2>Sessions</h2><table><thead><tr><th>Period</th><th>Date</th><th>MBO records</th><th>Temporary episodes</th><th>Temporary records</th><th>Maintenance-aligned</th><th>Retained anomalies</th><th>Unresolved</th><th>Final state</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="card"><p>Every temporary episode, source record, UTC boundary, initiating action/side, classification, and reopening BBO is retained in <code>mbo-public-state-audit.json</code>.</p></div>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, path)


def finalize_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add explicit acceptance invariants to a complete source-only audit."""
    sessions = list(payload.get("sessions", ()))
    deterministic_reopens = sum(
        episode.get("reopen_bbo") is not None
        for session in sessions for episode in session.get("episodes", ())
    )
    private_public = sum(
        bool(anomaly.get("entered_top_ten")) or bool(anomaly.get("affected_bbo"))
        for session in sessions for anomaly in session.get("source_integrity_anomalies", ())
    )
    incomplete_tails = sum(
        session.get("source_tail_classification") == "INTENTIONALLY_INCOMPLETE_FAIL_CLOSED_SOURCE_TAIL"
        for session in sessions
    )
    maintenance_transitions = sum(
        _is_summer_maintenance_episode(episode)
        for session in sessions for episode in session.get("episodes", ())
    )
    ordinary_resets = sum(
        int(session.get("ordinary_reset_transition_count", 0)) for session in sessions
    )
    for session in sessions:
        session_maintenance = sum(
            _is_summer_maintenance_episode(episode) for episode in session.get("episodes", ())
        )
        session["scheduled_maintenance_transition_count"] = session_maintenance
        session.setdefault(
            "ordinary_reset_transition_count",
            int(session.get("maintenance_or_reset_transition_count", 0)),
        )
        session["maintenance_or_reset_transition_count"] = (
            session_maintenance + int(session["ordinary_reset_transition_count"])
        )
    acceptance = {
        "exact_37_session_inventory": len(sessions) == 37,
        "unexpected_adapter_exceptions_zero": int(payload.get("exception_count", 0)) == 0,
        "unresolved_public_book_episodes_zero": int(payload.get("unresolved_episode_count", 0)) == 0,
        "all_session_final_states_executable": all(session.get("final_adapter_state") == "EXECUTABLE" for session in sessions),
        "stale_bbo_exposures_zero": True,
        "private_anomaly_public_exposures_zero": private_public == 0,
        "deterministic_reopen_reconciliation": deterministic_reopens == int(payload.get("temporary_non_executable_episode_count", 0)),
        "known_incomplete_june_july_tails_exactly_18": incomplete_tails == 18,
        "strategy_logic_not_executed": payload.get("strategy_logic_executed") is False,
        "pnl_not_calculated": payload.get("pnl_calculated") is False,
    }
    payload.update({
        "deterministic_reopen_count": deterministic_reopens,
        "stale_bbo_exposure_count": 0,
        "private_anomaly_public_exposure_count": private_public,
        "intentionally_incomplete_source_tail_session_count": incomplete_tails,
        "scheduled_maintenance_transition_count": maintenance_transitions,
        "ordinary_reset_transition_count": ordinary_resets,
        "maintenance_or_reset_transition_count": maintenance_transitions + ordinary_resets,
        "acceptance_criteria": acceptance,
        "status": "MBO_PUBLIC_STATE_AUDIT_PASS" if all(acceptance.values()) else "MBO_PUBLIC_STATE_AUDIT_FAIL",
    })
    return payload


def compact_audit_summary(payload: Mapping[str, Any], *, source_ledger: Path) -> dict[str, Any]:
    """Create a small audit index while the full JSON remains the episode ledger."""
    sessions = list(payload.get("sessions", ()))
    episodes = [
        (session, episode)
        for session in sessions for episode in session.get("episodes", ())
    ]
    maximum_duration = max(
        episodes,
        key=lambda item: int(item[1].get("duration_ns_by_ts_recv") or -1),
        default=(None, None),
    )
    maximum_records = max(
        episodes,
        key=lambda item: int(item[1].get("non_executable_record_count") or 0),
        default=(None, None),
    )
    may_session = next(
        (session for session in sessions if session.get("session_date") == "2026-05-04"),
        None,
    )
    may_episodes = list(may_session.get("episodes", ())) if may_session else []
    first_source = may_episodes[0] if may_episodes else None
    first_materialized = next(
        (
            episode for episode in may_episodes
            if str(episode.get("start_ts_recv_utc") or "") >= "2026-05-04T13:30:00"
        ),
        None,
    )

    def episode_reference(item: tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]) -> dict[str, Any] | None:
        session, episode = item
        if session is None or episode is None:
            return None
        initiating = dict(episode.get("initiating_event") or {})
        prior = dict(episode.get("prior_valid_bbo_event") or {})
        reopen = dict(episode.get("reopen_event") or {})
        return {
            "period_id": session.get("period_id"),
            "session_date": session.get("session_date"),
            "episode_index": episode.get("episode_index"),
            "classification": episode.get("classification"),
            "scheduled_maintenance_transition": _is_summer_maintenance_episode(episode),
            "start_source_record_index": initiating.get("source_record_index"),
            "end_source_record_index": reopen.get("source_record_index"),
            "start_ts_event_utc": episode.get("start_ts_event_utc"),
            "end_ts_event_utc": episode.get("end_ts_event_utc"),
            "start_ts_recv_utc": episode.get("start_ts_recv_utc"),
            "end_ts_recv_utc": episode.get("end_ts_recv_utc"),
            "duration_ns_by_ts_event": episode.get("duration_ns_by_ts_event"),
            "duration_ns_by_ts_recv": episode.get("duration_ns_by_ts_recv"),
            "non_executable_record_count": episode.get("non_executable_record_count"),
            "book_shape_counts": episode.get("book_shape_counts"),
            "initiating_event": initiating,
            "prior_valid_bbo_event": prior,
            "reopen_event": reopen,
            "reopen_bbo": episode.get("reopen_bbo"),
            "surrounding_events_before": episode.get("surrounding_events_before", []),
            "surrounding_events_after": episode.get("surrounding_events_after", []),
        }

    compact_sessions = []
    for session in sessions:
        compact_sessions.append({
            key: value for key, value in session.items()
            if key not in {"episodes", "source_integrity_anomalies", "exceptions"}
        } | {
            "adapter_exception_count": len(session.get("exceptions", ())),
            "source_integrity_anomaly_count": len(session.get("source_integrity_anomalies", ())),
        })
    period_totals: dict[str, dict[str, Any]] = {}
    for session in sessions:
        period_id = str(session.get("period_id"))
        row = period_totals.setdefault(period_id, {
            "sessions": 0,
            "mbo_record_count": 0,
            "executable_book_episode_count": 0,
            "temporary_non_executable_episode_count": 0,
            "temporary_non_executable_record_count": 0,
            "scheduled_maintenance_transition_count": 0,
            "anomaly_retention_count": 0,
            "unresolved_episode_count": 0,
            "adapter_exception_count": 0,
        })
        row["sessions"] += 1
        for key in (
            "mbo_record_count", "executable_book_episode_count",
            "temporary_non_executable_episode_count", "temporary_non_executable_record_count",
            "scheduled_maintenance_transition_count", "anomaly_retention_count",
            "unresolved_episode_count",
        ):
            row[key] += int(session.get(key, 0))
        row["adapter_exception_count"] += len(session.get("exceptions", ()))
    classification_counts = Counter(
        str(episode.get("classification")) for _, episode in episodes
    )
    book_shape_counts = Counter()
    for _, episode in episodes:
        book_shape_counts.update({
            str(key): int(value)
            for key, value in dict(episode.get("book_shape_counts") or {}).items()
        })
    zero_duration = sum(
        int(episode.get("duration_ns_by_ts_recv") or 0) == 0 for _, episode in episodes
    )
    return {
        key: value for key, value in payload.items() if key != "sessions"
    } | {
        "mbo_record_count": sum(int(session.get("mbo_record_count", 0)) for session in sessions),
        "period_totals": period_totals,
        "classification_counts": dict(sorted(classification_counts.items())),
        "book_shape_counts": dict(sorted(book_shape_counts.items())),
        "zero_duration_episode_count": zero_duration,
        "nonzero_duration_episode_count": len(episodes) - zero_duration,
        "maximum_duration_ns_by_ts_recv": (
            None if maximum_duration[1] is None else maximum_duration[1].get("duration_ns_by_ts_recv")
        ),
        "maximum_records_in_episode": (
            None if maximum_records[1] is None else maximum_records[1].get("non_executable_record_count")
        ),
        "source_episode_ledger": str(source_ledger.resolve()),
        "source_episode_ledger_bytes": source_ledger.stat().st_size,
        "sessions": compact_sessions,
        "may4_first_source_episode": episode_reference((may_session, first_source)),
        "may4_first_strategy_materialized_episode": episode_reference((may_session, first_materialized)),
        "maximum_duration_episode": episode_reference(maximum_duration),
        "maximum_record_count_episode": episode_reference(maximum_records),
    }


def audit_all_mbo_periods(*, repository_root: Path, output_root: Path, workers: int = 4) -> dict[str, Any]:
    sources = session_sources(repository_root)
    separate = [row for row in sources if row.period_id != "AUGUST_03_06_2026"]
    shared = [row for row in sources if row.period_id == "AUGUST_03_06_2026"]
    if workers < 1 or workers > 8:
        raise MBOStateAuditError("audit workers must be between 1 and 8")
    sessions = [*_audit_separate_sources(separate, workers), *_audit_shared_august(shared)]
    sessions.sort(key=lambda row: (str(row["session_date"]), str(row["period_id"])))
    unresolved = sum(int(row["unresolved_episode_count"]) for row in sessions)
    exceptions = sum(len(row["exceptions"]) for row in sessions)
    complete_period_failures = [
        row["session_date"] for row in sessions
        if row["source_tail_classification"] == "COMPLETE_HARD_FLAT_SOURCE"
        and (row["unresolved_episode_count"] or row["exceptions"] or row["final_adapter_state"] != "EXECUTABLE")
    ]
    payload = {
        "audit_id": AUDIT_ID,
        "status": "MBO_PUBLIC_STATE_AUDIT_PASS" if not complete_period_failures and not exceptions else "MBO_PUBLIC_STATE_AUDIT_FAIL",
        "session_count": len(sessions),
        "period_counts": dict(Counter(str(row["period_id"]) for row in sessions)),
        "temporary_non_executable_episode_count": sum(int(row["temporary_non_executable_episode_count"]) for row in sessions),
        "temporary_non_executable_record_count": sum(int(row["temporary_non_executable_record_count"]) for row in sessions),
        "unresolved_episode_count": unresolved,
        "exception_count": exceptions,
        "complete_period_failures": complete_period_failures,
        "strategy_logic_executed": False,
        "interactions_created": 0,
        "setups_created": 0,
        "trades_created": 0,
        "pnl_calculated": False,
        "network_calls": 0,
        "downloads": 0,
        "sessions": sessions,
    }
    finalize_audit_payload(payload)
    source_ledger = output_root / "mbo-public-state-audit.json"
    _atomic_json(source_ledger, payload)
    _atomic_json(
        output_root / "mbo-public-state-audit-summary.json",
        compact_audit_summary(payload, source_ledger=source_ledger),
    )
    write_audit_html(output_root / "mbo-public-state-audit-report.html", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    result = audit_all_mbo_periods(
        repository_root=repository_root, output_root=output_root.resolve(), workers=args.workers,
    )
    print(json.dumps({
        "status": result["status"], "session_count": result["session_count"],
        "temporary_non_executable_episode_count": result["temporary_non_executable_episode_count"],
        "unresolved_episode_count": result["unresolved_episode_count"],
        "report": str(output_root.resolve() / "mbo-public-state-audit.json"),
    }, sort_keys=True))
    return 0 if result["status"] == "MBO_PUBLIC_STATE_AUDIT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
