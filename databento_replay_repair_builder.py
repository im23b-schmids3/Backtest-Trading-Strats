"""Build verified local replay streams from ProjectX captures and DBN repairs.

This is deliberately an offline build step.  It never creates a Databento
client and never downloads data.  The source files are immutable; output is
written below ``<repair-root>/final-replay``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import databento_replay_repair_downloader as acquisition


APPROVED_DATES = acquisition.APPROVED_DATES
# CME equity-index futures had an early Labor Day halt on 2026-09-07.  The
# replay contract ends at the actual tradable halt, rather than treating the
# ordinary 20:00Z bound as required coverage for that holiday.
SESSION_END_OVERRIDES_UTC = {
    "2026-09-07": "2026-09-07T17:00:00Z",
}
SESSION_COVERAGE_NOTES = {
    "2026-09-07": "CME equity-index futures Labor Day halt; required coverage ends 17:00Z",
}
SESSION_NS = {
    day: (
        acquisition.session_bounds(day)[0],
        acquisition._parse_ns(SESSION_END_OVERRIDES_UTC[day])
        if day in SESSION_END_OVERRIDES_UTC else acquisition.session_bounds(day)[1],
    )
    for day in APPROVED_DATES
}
EXPECTED_SYMBOLS = {
    day: ("ESZ6", "MESZ6") if day in acquisition.Z26_DATES else ("ESU6", "MESU6")
    for day in APPROVED_DATES
}
OUTPUT_DIRECTORY = "final-replay"
STREAM_NAME = "causal-events.jsonl.gz"
MANIFEST_NAME = "replay-manifest.json"
FORMAT_VERSION = "cme-replay-causal-v1"


class ReplayBuildError(RuntimeError):
    """A replay source or output invariant failed."""


def _code(value: Any) -> str:
    return getattr(value, "name", str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_iso_ns(value: str) -> int:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text).astimezone(timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _iso_ns(value: int) -> str:
    return acquisition.ns_to_iso(value)


def _valid_source_ns(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = _parse_iso_ns(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    # ProjectX uses year 0001 for payloads without an event timestamp.
    return parsed if parsed > 1_000_000_000_000_000_000 else None


def _stable_digest(value: Any) -> int:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")


def _priority(event_type: str) -> int:
    return {"MES_BBO": 0, "ES_DEPTH": 1, "ES_TRADE": 2, "ES_BBO": 3}[event_type]


@dataclass
class SourceLedger:
    path: Path
    provider: str
    schema: str
    symbol: str
    bytes: int
    sha256: str | None = None
    rows_read: int = 0
    rows_emitted: int = 0

    def as_dict(self, root: Path) -> dict[str, Any]:
        try:
            path = self.path.relative_to(root).as_posix()
        except ValueError:
            path = str(self.path)
        return {
            "path": path,
            "provider": self.provider,
            "schema": self.schema,
            "symbol": self.symbol,
            "bytes": self.bytes,
            "sha256": self.sha256 or _sha256(self.path),
            "rows_read": self.rows_read,
            "rows_emitted": self.rows_emitted,
        }


@dataclass(frozen=True)
class FeedSource:
    event_type: str
    provider: str
    path: Path
    artifact: acquisition.VerifiedArtifact | None = None


@dataclass
class DateBuild:
    day: str
    session_start_ns: int
    session_end_ns: int
    expected_symbols: tuple[str, str]
    repair_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    ledgers: dict[str, SourceLedger] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    provider_counts: dict[str, int] = field(default_factory=dict)
    first_by_type: dict[str, int] = field(default_factory=dict)
    last_by_type: dict[str, int] = field(default_factory=dict)
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    duplicate_count: int = 0
    source_event_count: int = 0

    def observe(self, event: Mapping[str, Any], provider: str) -> None:
        event_type = str(event["event_type"])
        timestamp_ns = int(event["timestamp_ns"])
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        self.first_by_type[event_type] = min(self.first_by_type.get(event_type, timestamp_ns), timestamp_ns)
        self.last_by_type[event_type] = max(self.last_by_type.get(event_type, timestamp_ns), timestamp_ns)
        key = f"{provider}:{event_type}"
        self.provider_counts[key] = self.provider_counts.get(key, 0) + 1
        self.first_timestamp_ns = timestamp_ns if self.first_timestamp_ns is None else min(self.first_timestamp_ns, timestamp_ns)
        self.last_timestamp_ns = timestamp_ns if self.last_timestamp_ns is None else max(self.last_timestamp_ns, timestamp_ns)


def _source_log_paths(archive_root: Path, day: str) -> list[Path]:
    root = archive_root / "logs" / "topstep"
    return sorted(root.glob(f"{day}-session-auto*/market-user-capture*.jsonl"))


def _repair_artifacts(
    artifacts: Sequence[acquisition.VerifiedArtifact], day: str,
) -> dict[str, list[acquisition.VerifiedArtifact]]:
    result: dict[str, list[acquisition.VerifiedArtifact]] = {}
    for artifact in artifacts:
        if artifact.request.date != day:
            continue
        result.setdefault(artifact.request.purpose, []).append(artifact)
    for rows in result.values():
        rows.sort(key=lambda item: (item.request.window.start_ns, str(item.path)))
    return result


def _merged_intervals(rows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(rows):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _in_intervals(timestamp_ns: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= timestamp_ns < end for start, end in intervals)


def _json_payload(event: Mapping[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"


def _projectx_event(
    row: Mapping[str, Any], path: Path, line_number: int,
    expected_es_symbol: str, expected_mes_symbol: str,
) -> tuple[str, dict[str, Any]] | None:
    event_type = row.get("event_type")
    if event_type not in {"GatewayDepth", "GatewayTrade", "GatewayQuote"}:
        return None
    symbol = str(row.get("symbol"))
    if event_type in {"GatewayDepth", "GatewayTrade"}:
        if symbol != expected_es_symbol:
            raise ReplayBuildError(f"{path}:{line_number}: expected {expected_es_symbol}, got {symbol!r}")
        canonical_type = "ES_DEPTH" if event_type == "GatewayDepth" else "ES_TRADE"
    elif symbol == expected_mes_symbol:
        canonical_type = "MES_BBO"
    elif symbol == expected_es_symbol:
        # ES quote updates are not a required replay feed; ES depth and trades
        # are the authoritative ES inputs for this build.
        return None
    else:
        raise ReplayBuildError(f"{path}:{line_number}: expected {expected_mes_symbol}, got {symbol!r}")
    timestamp_ns = row.get("local_receipt_timestamp_ns")
    if timestamp_ns is None:
        raise ReplayBuildError(f"{path}:{line_number}: missing local_receipt_timestamp_ns")
    timestamp_ns = int(timestamp_ns)
    normalized = row.get("normalized") or {}
    source_timestamp = row.get("source_timestamp") or row.get("source_timestamp_raw")
    common = {
        "format_version": FORMAT_VERSION,
        "timestamp_ns": timestamp_ns,
        "timestamp_utc": _iso_ns(timestamp_ns),
        "provider": "PROJECTX",
        "source_file": str(path),
        "source_line": line_number,
        "contract": symbol,
        "contract_id": row.get("contract_id"),
        "provider_timestamp": source_timestamp,
        "provider_timestamp_ns": _valid_source_ns(source_timestamp),
        "batch_id": row.get("batch_id"),
        "batch_index": row.get("batch_index"),
    }
    if event_type == "GatewayDepth":
        price = normalized.get("price")
        volume = normalized.get("volume")
        payload = {
            "depth_event_type": normalized.get("depth_event_type_label") or normalized.get("depth_event_type"),
            "side": normalized.get("side"),
            "price": price,
            "volume": volume,
            "order_count": normalized.get("order_count"),
            "is_reset": bool(normalized.get("is_reset")),
            "sequence": normalized.get("sequence"),
        }
        identity = ["PROJECTX", "ES_DEPTH", row.get("batch_id"), row.get("batch_index"), payload]
        return canonical_type, {**common, "event_type": canonical_type, "stream": "ES", "payload": payload,
                             "dedup_key": _stable_digest(identity)}
    if event_type == "GatewayTrade":
        payload = {
            "price": normalized.get("price"),
            "size": normalized.get("volume"),
            "aggressor": normalized.get("aggressor_side"),
            "aggressor_verified": bool(normalized.get("aggressor_side_verified")),
            "trade_type": normalized.get("trade_type_label"),
            "sequence": normalized.get("sequence"),
        }
        identity = ["PROJECTX", "ES_TRADE", row.get("batch_id"), row.get("batch_index"), payload]
        return canonical_type, {**common, "event_type": canonical_type, "stream": "ES", "payload": payload,
                             "dedup_key": _stable_digest(identity)}
    bid = normalized.get("bid")
    ask = normalized.get("ask")
    if bid is None or ask is None or float(bid) <= 0 or float(ask) <= float(bid):
        return None
    payload = {"bid": bid, "ask": ask, "bid_size": normalized.get("bid_size"),
               "ask_size": normalized.get("ask_size"), "last_price": normalized.get("last_price"),
               "volume": normalized.get("volume")}
    identity = ["PROJECTX", "MES_BBO", source_timestamp, row.get("local_receipt_timestamp_ns"), payload]
    return "MES_BBO", {**common, "event_type": "MES_BBO", "stream": "MES", "payload": payload,
                       "dedup_key": _stable_digest(identity)}


def _iter_projectx(
    path: Path, day: str, expected_symbol: str, ledger: SourceLedger,
    excluded_intervals: Sequence[tuple[int, int]],
) -> Iterator[tuple[int, int, int, dict[str, Any], str]]:
    previous: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            ledger.rows_read += 1
            row = json.loads(line)
            parsed = _projectx_event(row, path, line_number, expected_symbol, expected_symbol)
            if parsed is None:
                continue
            event_type, event = parsed
            timestamp_ns = int(event["timestamp_ns"])
            if previous is not None and timestamp_ns < previous:
                raise ReplayBuildError(f"ProjectX capture is not ordered: {path}:{line_number}")
            previous = timestamp_ns
            start_ns, end_ns = SESSION_NS[day]
            if not start_ns <= timestamp_ns < end_ns or _in_intervals(timestamp_ns, excluded_intervals):
                continue
            ledger.rows_emitted += 1
            yield timestamp_ns, _priority(event_type), line_number, event, "PROJECTX"


def _dbn_level(level: Any) -> dict[str, Any]:
    return {
        "bid_price": int(level.bid_px) / 1_000_000_000 if int(level.bid_px) > 0 else None,
        "bid_size": int(level.bid_sz),
        "bid_orders": int(level.bid_ct),
        "ask_price": int(level.ask_px) / 1_000_000_000 if int(level.ask_px) > 0 else None,
        "ask_size": int(level.ask_sz),
        "ask_orders": int(level.ask_ct),
    }


def _dbn_event(record: Any, artifact: acquisition.VerifiedArtifact, index: int) -> tuple[str, dict[str, Any]]:
    request = artifact.request
    ts_recv = int(record.ts_recv)
    ts_event = int(getattr(record, "ts_event", ts_recv))
    common = {
        "format_version": FORMAT_VERSION,
        "timestamp_ns": ts_recv,
        "timestamp_utc": _iso_ns(ts_recv),
        "provider": "DATABENTO",
        "source_file": str(artifact.path),
        "source_record_index": index,
        "contract": request.symbol,
        "contract_id": int(getattr(record, "instrument_id", 0)),
        "provider_timestamp_ns": ts_event,
        "receive_timestamp_ns": ts_recv,
        "sequence": int(getattr(record, "sequence", 0)),
        "flags": int(getattr(record, "flags", 0)),
    }
    if request.schema == "trades":
        side = _code(getattr(record, "side", "N"))
        aggressor = "BUY" if side == "A" else "SELL" if side == "B" else "UNKNOWN"
        payload = {"price": int(record.price) / 1_000_000_000, "size": int(record.size),
                   "aggressor": aggressor, "side": side, "action": _code(record.action)}
        identity = ["DATABENTO", "ES_TRADE", ts_recv, ts_event, int(record.price), int(record.size), side,
                    int(getattr(record, "sequence", 0))]
        return "ES_TRADE", {**common, "event_type": "ES_TRADE", "stream": "ES", "payload": payload,
                             "dedup_key": _stable_digest(identity)}
    if request.schema == "mbp-1":
        levels = getattr(record, "levels", ())
        if not levels:
            raise ReplayBuildError(f"MES mbp-1 record has no level: {artifact.path}:{index}")
        payload = _dbn_level(levels[0])
        identity = ["DATABENTO", "MES_BBO", ts_recv, ts_event, payload, int(getattr(record, "sequence", 0))]
        return "MES_BBO", {**common, "event_type": "MES_BBO", "stream": "MES", "payload": payload,
                            "dedup_key": _stable_digest(identity)}
    if request.schema == "mbp-10":
        levels = getattr(record, "levels", ())
        payload = {"action": _code(record.action), "side": _code(record.side), "depth": int(record.depth),
                   "price": int(record.price) / 1_000_000_000, "size": int(record.size),
                   "levels": [_dbn_level(level) for level in levels]}
        identity = ["DATABENTO", "ES_DEPTH", ts_recv, ts_event, payload, int(getattr(record, "sequence", 0))]
        return "ES_DEPTH", {**common, "event_type": "ES_DEPTH", "stream": "ES", "payload": payload,
                             "dedup_key": _stable_digest(identity)}
    raise ReplayBuildError(f"unsupported repair schema: {request.schema}")


def _iter_dbn(
    artifact: acquisition.VerifiedArtifact, day: str, ledger: SourceLedger,
) -> Iterator[tuple[int, int, int, dict[str, Any], str]]:
    from databento import DBNStore

    start_ns, end_ns = SESSION_NS[day]
    previous: int | None = None
    for index, record in enumerate(DBNStore.from_file(artifact.path)):
        ledger.rows_read += 1
        timestamp_ns = int(record.ts_recv)
        if previous is not None and timestamp_ns < previous:
            raise ReplayBuildError(f"DBN source is not ordered: {artifact.path}:{index}")
        previous = timestamp_ns
        if timestamp_ns < start_ns:
            raise ReplayBuildError(f"DBN source starts before session bounds: {artifact.path}:{index}")
        # Databento tail requests may intentionally overfetch past a holiday
        # halt.  Keep the source artifact auditable, but do not emit or require
        # records outside the effective replay window.
        if timestamp_ns >= end_ns:
            continue
        event_type, event = _dbn_event(record, artifact, index)
        ledger.rows_emitted += 1
        yield timestamp_ns, _priority(event_type), index, event, "DATABENTO"


def _merge_streams(
    streams: Sequence[Iterator[tuple[int, int, int, dict[str, Any], str]]],
    output: Path, build: DateBuild,
) -> None:
    heap: list[tuple[int, int, int, int, dict[str, Any], str]] = []
    for source_index, iterator in enumerate(streams):
        try:
            timestamp_ns, priority, ordinal, event, provider = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (timestamp_ns, priority, source_index, ordinal, event, provider))
    seen: set[int] = set()
    previous_key: tuple[int, int, int] | None = None
    with gzip.open(output, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        while heap:
            timestamp_ns, priority, source_index, ordinal, event, provider = heapq.heappop(heap)
            dedup_key = int(event["dedup_key"])
            if dedup_key in seen:
                build.duplicate_count += 1
            else:
                seen.add(dedup_key)
                key = (timestamp_ns, priority, source_index)
                if previous_key is not None and key < previous_key:
                    raise ReplayBuildError(f"causal ordering decreased for {build.day}")
                previous_key = key
                build.observe(event, provider)
                build.source_event_count += 1
                handle.write(_json_payload(event))
            try:
                next_timestamp, next_priority, next_ordinal, next_event, next_provider = next(streams[source_index])
            except StopIteration:
                continue
            heapq.heappush(heap, (next_timestamp, next_priority, source_index, next_ordinal, next_event, next_provider))


def _capture_ledger(path: Path, expected_symbols: tuple[str, str]) -> SourceLedger:
    return SourceLedger(path, "PROJECTX", "projectx-capture-v1", "/".join(expected_symbols), path.stat().st_size)


def _dbn_ledger(artifact: acquisition.VerifiedArtifact) -> SourceLedger:
    return SourceLedger(artifact.path, "DATABENTO", artifact.request.schema, artifact.request.symbol, artifact.path.stat().st_size,
                        artifact.sha256)


def _recovery_trade_paths(archive_root: Path, day: str) -> list[Path]:
    return sorted((archive_root / "data" / "topstep-session-profile-recovery" / day).glob("*trades.dbn.zst"))


def _manifest_verified_artifacts(repair_root: Path) -> tuple[acquisition.VerifiedArtifact, ...]:
    """Load artifacts already verified by the acquisition manifests.

    The downloader has already validated these DBNs at acquisition time.  The
    replay merge below validates every record again while decoding it, so a
    second complete DBN pass here would only duplicate the expensive work.
    """
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for manifest_name in (acquisition.LEGACY_MANIFEST_NAME, acquisition.MANIFEST_NAME):
        manifest_path = repair_root / manifest_name
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("requests", {})
        if not isinstance(records, dict):
            raise ReplayBuildError(f"invalid acquisition manifest: {manifest_path}")
        for record in records.values():
            if isinstance(record, dict) and record.get("output_path"):
                records_by_path[str(record["output_path"])] = record

    artifacts: list[acquisition.VerifiedArtifact] = []
    for relative_path, record in sorted(records_by_path.items()):
        path = repair_root / relative_path
        if not path.is_file() or not path.name.endswith(".dbn.zst"):
            continue
        try:
            request = acquisition._request_from_record(record)
            size = path.stat().st_size
            recorded_size = record.get("size")
            if recorded_size is not None and int(recorded_size) != size:
                raise ReplayBuildError(f"size mismatch for {path}")
            digest = str(record.get("sha256") or _sha256(path))
            artifacts.append(acquisition.VerifiedArtifact(
                request=request, path=path, size=size, sha256=digest,
                quoted_cost_usd=None,
            ))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ReplayBuildError(f"invalid verified artifact record {relative_path}: {exc}") from exc

    # The four final-tail downloads are kept in their own acquisition manifest
    # and are not part of either historical eight-day manifest.  Include only
    # completed, verified final-tail files; an unfinished .part is ignored.
    tail_manifest_path = repair_root / "final-tails" / "final-tail-download-manifest.json"
    if tail_manifest_path.is_file():
        payload = json.loads(tail_manifest_path.read_text(encoding="utf-8"))
        records = payload.get("requests", {})
        if not isinstance(records, dict):
            raise ReplayBuildError(f"invalid final-tail manifest: {tail_manifest_path}")
        known_paths = {str(item.path) for item in artifacts}
        for request_id, record in sorted(records.items()):
            if not isinstance(record, dict) or record.get("status") not in {
                "DOWNLOADED_VERIFIED", "SKIPPED_VERIFIED", "PROMOTED_VERIFIED_PART",
            }:
                continue
            request_data = record.get("request")
            if not isinstance(request_data, dict):
                raise ReplayBuildError(f"invalid final-tail request: {request_id}")
            symbols = request_data.get("symbols")
            # The standalone final-tail downloader records its single target
            # as ``symbol``; the original four-request downloader used the
            # SDK-shaped ``symbols`` list.  Accept both manifest forms while
            # keeping the one-symbol invariant strict.
            if symbols is None and isinstance(request_data.get("symbol"), str):
                symbols = [request_data["symbol"]]
            if not isinstance(symbols, list) or len(symbols) != 1:
                raise ReplayBuildError(f"invalid final-tail symbols: {request_id}")
            schema = str(request_data.get("schema"))
            symbol = str(symbols[0])
            purpose = {
                ("mbp-10", "ESU6"): "ES_DEPTH",
                ("mbp-1", "MESU6"): "MES_BBO",
                ("trades", "ESU6"): "ES_TRADES",
            }.get((schema, symbol))
            if purpose is None:
                raise ReplayBuildError(f"unsupported final-tail request: {request_id}")
            path = Path(str(record.get("path", "")))
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.is_file() or str(path) in known_paths:
                continue
            date = str(request_data.get("start", ""))[:10]
            request = acquisition.Request(
                str(request_id), date, purpose, schema, symbol,
                acquisition.Window(
                    acquisition._parse_ns(str(request_data["start"])),
                    acquisition._parse_ns(str(request_data["end"])),
                    "verified final-tail download",
                ),
                1,
            )
            size = path.stat().st_size
            recorded_size = record.get("bytes")
            if recorded_size is not None and int(recorded_size) != size:
                raise ReplayBuildError(f"final-tail size mismatch: {path}")
            digest = str(record.get("sha256") or _sha256(path))
            artifacts.append(acquisition.VerifiedArtifact(
                request=request, path=path, size=size, sha256=digest,
                quoted_cost_usd=None,
            ))
            known_paths.add(str(path))
    return tuple(artifacts)


def _build_date(
    archive_root: Path, repair_root: Path, day: str,
    artifacts: Sequence[acquisition.VerifiedArtifact],
) -> dict[str, Any]:
    session_start, session_end = SESSION_NS[day]
    es_symbol, mes_symbol = EXPECTED_SYMBOLS[day]
    build = DateBuild(day, session_start, session_end, (es_symbol, mes_symbol))
    by_purpose = _repair_artifacts(artifacts, day)
    for purpose, rows in by_purpose.items():
        build.repair_intervals[purpose] = _merged_intervals(
            (item.request.window.start_ns, item.request.window.end_ns) for item in rows
        )

    log_paths = _source_log_paths(archive_root, day)
    for path in log_paths:
        ledger = _capture_ledger(path, (es_symbol, mes_symbol))
        build.ledgers[f"PROJECTX:{path}"] = ledger

    streams: list[Iterator[tuple[int, int, int, dict[str, Any], str]]] = []
    for key, ledger in list(build.ledgers.items()):
        if not key.startswith("PROJECTX:"):
            continue
        path = ledger.path
        streams.append(_iter_projectx_precise(path, day, es_symbol, mes_symbol, ledger, build.repair_intervals))

    # Historical ES trades are required for 2026-09-02/04, where no ProjectX
    # market capture is archived.  On 2026-09-07 ProjectX trades are present
    # through the live capture and the newly acquired suffix is the explicit
    # replacement source, so the older recovery file is not mixed in.
    if day in {"2026-09-02", "2026-09-04"}:
        for path in _recovery_trade_paths(archive_root, day):
            artifact = acquisition.VerifiedArtifact(
                request=acquisition.Request(
                    f"archive-trades-{day}", day, "ES_TRADES", "trades", es_symbol,
                    acquisition.Window(session_start, session_end, "archived Databento trade recovery"), 1,
                ),
                path=path, size=path.stat().st_size, sha256=_sha256(path), quoted_cost_usd=None,
            )
            ledger = _dbn_ledger(artifact)
            build.ledgers[f"DATABENTO_RECOVERY:{path}"] = ledger
            streams.append(_iter_dbn(artifact, day, ledger))

    for purpose, event_type in (("ES_DEPTH", "ES_DEPTH"), ("MES_BBO", "MES_BBO"), ("ES_TRADES", "ES_TRADE")):
        for artifact in by_purpose.get(purpose, []):
            ledger = _dbn_ledger(artifact)
            build.ledgers[f"DATABENTO:{artifact.path}"] = ledger
            streams.append(_iter_dbn(artifact, day, ledger))

    output_dir = repair_root / OUTPUT_DIRECTORY / day
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / STREAM_NAME
    _merge_streams(streams, stream_path, build)
    stream_sha = _sha256(stream_path)
    feed_summary = {}
    for event_type in ("ES_TRADE", "ES_DEPTH", "MES_BBO"):
        count = build.counts.get(event_type, 0)
        feed_summary[event_type] = {"event_count": count}
        if count:
            feed_summary[event_type].update({
                "first_timestamp_utc": _iso_ns(build.first_by_type[event_type]),
                "last_timestamp_utc": _iso_ns(build.last_by_type[event_type]),
            })

    failures: list[str] = []
    for event_type, label in (("ES_TRADE", "ES trades"), ("ES_DEPTH", "ES depth"), ("MES_BBO", "MES BBO")):
        summary = feed_summary[event_type]
        if summary["event_count"] == 0:
            failures.append(f"{label}: no canonical events")
            continue
        first_ns = _parse_iso_ns(summary["first_timestamp_utc"])
        last_ns = _parse_iso_ns(summary["last_timestamp_utc"])
        if first_ns > session_start + 60 * 1_000_000_000:
            failures.append(f"{label}: starts at {_iso_ns(first_ns)}, after session start {_iso_ns(session_start)}")
        if last_ns < session_end - 60 * 1_000_000_000:
            failures.append(f"{label}: no source after {_iso_ns(last_ns)}; missing [{_iso_ns(last_ns)},{_iso_ns(session_end)})")

    source_records = [ledger.as_dict(archive_root) for ledger in sorted(build.ledgers.values(), key=lambda item: str(item.path))]
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "FULL_LIVE_EQUIVALENT_REPLAY_READY" if not failures else "FAILED_INCOMPLETE_REPLAY",
        "date": day,
        "dataset": "GLBX.MDP3",
        "session": {"start_utc_inclusive": _iso_ns(session_start), "end_utc_exclusive": _iso_ns(session_end)},
        "session_coverage_basis": SESSION_COVERAGE_NOTES.get(day, "regular 07:00Z-20:00Z replay coverage"),
        "contracts": {"ES": es_symbol, "MES": mes_symbol},
        "schemas": {"ES_DEPTH": "mbp-10/projectx-depth", "MES_BBO": "mbp-1/projectx-quote", "ES_TRADES": "trades/projectx-trade"},
        "repair_intervals": {
            key: [{"start_utc_inclusive": _iso_ns(start), "end_utc_exclusive": _iso_ns(end)} for start, end in value]
            for key, value in sorted(build.repair_intervals.items())
        },
        "stream": {"path": STREAM_NAME, "bytes": stream_path.stat().st_size, "sha256": stream_sha,
                   "event_count": build.source_event_count, "duplicate_events_removed": build.duplicate_count},
        "feed_summary": feed_summary,
        "source_files": source_records,
        "verification": {"passed": not failures, "failures": failures, "no_mbo": True,
                          "timestamp_basis": "ProjectX local_receipt_timestamp_ns; Databento ts_recv",
                          "overlap_policy": "exact replay duplicates removed; explicit repair blocks replace ProjectX events",
                          "ordering": "timestamp, feed priority MES_BBO < ES_DEPTH < ES_TRADE < ES_BBO, source order"},
    }
    manifest_path = output_dir / MANIFEST_NAME
    fd, tmp_name = tempfile.mkstemp(prefix=".replay-manifest.", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, manifest_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return manifest


def _iter_projectx_precise(
    path: Path, day: str, expected_es_symbol: str, expected_mes_symbol: str, ledger: SourceLedger,
    repair_intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> Iterator[tuple[int, int, int, dict[str, Any], str]]:
    previous: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            ledger.rows_read += 1
            row = json.loads(line)
            parsed = _projectx_event(row, path, line_number, expected_es_symbol, expected_mes_symbol)
            if parsed is None:
                continue
            event_type, event = parsed
            timestamp_ns = int(event["timestamp_ns"])
            if previous is not None and timestamp_ns < previous:
                raise ReplayBuildError(f"ProjectX capture is not ordered: {path}:{line_number}")
            previous = timestamp_ns
            start_ns, end_ns = SESSION_NS[day]
            purpose = {"ES_DEPTH": "ES_DEPTH", "ES_TRADE": "ES_TRADES", "MES_BBO": "MES_BBO"}[event_type]
            excluded = repair_intervals.get(purpose, ()) if purpose else ()
            if not start_ns <= timestamp_ns < end_ns or _in_intervals(timestamp_ns, excluded):
                continue
            ledger.rows_emitted += 1
            yield timestamp_ns, _priority(event_type), line_number, event, "PROJECTX"


def _validate_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ReplayBuildError(f"manifest format mismatch: {path}")
    stream = path.parent / str(payload["stream"]["path"])
    if not stream.is_file() or _sha256(stream) != payload["stream"]["sha256"]:
        raise ReplayBuildError(f"stream integrity mismatch: {path}")
    if payload["stream"]["event_count"] <= 0:
        raise ReplayBuildError(f"empty replay stream: {path}")
    return payload


def build_all(archive_root: Path, repair_root: Path) -> tuple[list[str], list[str]]:
    artifacts = _manifest_verified_artifacts(repair_root)
    if not artifacts:
        raise ReplayBuildError("no verified repair DBNs found")
    ready: list[str] = []
    failed: list[str] = []
    for day in APPROVED_DATES:
        print(f"BUILD_REPLAY_DATE={day}")
        try:
            manifest = _build_date(archive_root, repair_root, day, artifacts)
            if manifest["verification"]["passed"]:
                _validate_manifest(repair_root / OUTPUT_DIRECTORY / day / MANIFEST_NAME)
                ready.append(day)
                print(f"REPLAY_DATE_READY={day}")
            else:
                failed.append(day)
                print(f"REPLAY_DATE_FAILED={day} failures={manifest['verification']['failures']}")
        except Exception as exc:
            failed.append(day)
            print(f"REPLAY_DATE_FAILED={day} error={exc}")
    print(f"FULL_REPLAY_READY_DATES={json.dumps(ready)}")
    print(f"FAILED_REPLAY_DATES={json.dumps(failed)}")
    return ready, failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, default=Path("data/live-replay-repair"))
    args = parser.parse_args(argv)
    ready, failed = build_all(args.archive_root.expanduser().resolve(), args.repair_root)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
