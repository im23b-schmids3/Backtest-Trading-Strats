"""Acquire the frozen eight-day Databento replay-repair plan.

This module is intentionally acquisition-only.  It never imports strategy code,
merges files with ProjectX data, or requests MBO.  The plan is built from the
approved archive evidence and is deliberately fail-closed when that evidence
is missing or malformed.

Examples::

    python databento_replay_repair_downloader.py \
        --archive-root /Users/sandro/Documents/Trading-Bot-Fib/Historical-Live-Data \
        --dry-run

    python databento_replay_repair_downloader.py \
        --archive-root /Users/sandro/Documents/Trading-Bot-Fib/Historical-Live-Data \
        --download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
import tempfile
import time
from http.client import RemoteDisconnected
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET = "GLBX.MDP3"
STYPE_IN = "raw_symbol"
BUDGET_USD = Decimal("9.40")
EXPECTED_QUOTED_TOTAL_USD = Decimal("8.450882")
OUTPUT_ROOT = Path("data/live-replay-repair")
APPROVED_DATES = (
    "2026-09-02", "2026-09-04", "2026-09-07", "2026-09-08",
    "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-17",
)
Z26_DATES = frozenset({"2026-09-17"})
FULL_SESSION_DATES = frozenset({"2026-09-02", "2026-09-04"})
MANIFEST_NAME = "practical-download-manifest.json"
PROGRESS_NAME = "practical-download-progress.json"
LEGACY_MANIFEST_NAME = "download-manifest.json"
LEGACY_PROGRESS_NAME = "download-progress.json"
PRACTICAL_MERGE_LEVELS_SECONDS = (86_400, 3_600, 1_800, 900, 300, 60, 0)
DEFAULT_REMAINING_BUDGET_USD = Decimal("0.95")
NANOSECONDS = 1_000_000_000
RETRY_DELAYS_SECONDS = (2, 4, 8, 16, 30, 60)
MAX_RETRY_ATTEMPTS = len(RETRY_DELAYS_SECONDS) + 1

try:
    from requests.exceptions import ChunkedEncodingError as RequestsChunkedEncodingError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout
except ImportError:  # pragma: no cover - requests is supplied by Databento's SDK
    RequestsChunkedEncodingError = RequestsConnectionError = RequestsTimeout = ()  # type: ignore[assignment]

try:
    from urllib3.exceptions import ProtocolError as Urllib3ProtocolError
except ImportError:  # pragma: no cover - urllib3 is supplied by Databento's SDK
    Urllib3ProtocolError = ()  # type: ignore[assignment]

try:
    from databento.common.error import BentoError
except ImportError:  # pragma: no cover - the downloader imports Databento at runtime
    BentoError = ()  # type: ignore[assignment]


class PlanError(RuntimeError):
    """The frozen repair plan or an acquisition invariant is invalid."""


def _http_status(exc: BaseException) -> int | None:
    for name in ("http_status", "status_code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _exception_text(exc: BaseException) -> str:
    values = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.append(str(current))
        for name in ("http_body", "json_body"):
            value = getattr(current, name, None)
            if value is not None:
                values.append(str(value))
        current = current.__cause__ or current.__context__
    return " ".join(values).lower()


def _is_hard_quota_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return any(marker in text for marker in (
        "monthly data returned limit exceeded",
        "monthly quota",
        "quota exhausted",
        "insufficient credit",
        "credit limit",
        "account quota",
    ))


def _is_transient_bento_error(exc: BaseException) -> bool:
    if not isinstance(BentoError, type) or not isinstance(exc, BentoError):
        return False
    text = _exception_text(exc)
    return any(marker in text for marker in (
        "read timed out",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "remote end closed connection",
        "temporary network",
        "temporarily unavailable",
        "streaming failure",
    ))


def is_retryable_exception(exc: BaseException) -> bool:
    """Classify only transient transport/server failures as retryable."""
    if _is_hard_quota_error(exc):
        return False
    status = _http_status(exc)
    if status is not None:
        if 500 <= status <= 599 or status in (408, 429):
            return True
        return False
    if _is_transient_bento_error(exc):
        return True
    network_types = tuple(
        item for item in (
            RequestsConnectionError,
            RequestsTimeout,
            RequestsChunkedEncodingError,
            Urllib3ProtocolError,
            RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
            TimeoutError,
        ) if isinstance(item, type)
    )
    return isinstance(exc, network_types)


def _transient_reason(exc: BaseException) -> str:
    reason = " ".join(str(exc).split())
    return reason[:240] if reason else "unspecified transport failure"


def _retry_call(*, operation: str, label: str, callback: Any,
                sleep_fn: Any = time.sleep, random_fn: Any = random.random) -> Any:
    """Run one quote/download call with bounded exponential backoff."""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return callback()
        except Exception as exc:
            if not is_retryable_exception(exc) or attempt == MAX_RETRY_ATTEMPTS:
                raise
            base_delay = RETRY_DELAYS_SECONDS[attempt - 1]
            delay = base_delay + (base_delay * 0.25 * float(random_fn()))
            print(
                f"RETRY label={label} operation={operation} attempt={attempt + 1}/"
                f"{MAX_RETRY_ATTEMPTS} exception={type(exc).__name__} "
                f"reason={_transient_reason(exc)!r} "
                f"next_delay_seconds={delay:.3f}"
            )
            sleep_fn(delay)
    raise AssertionError("retry loop terminated unexpectedly")


@dataclass(frozen=True)
class Window:
    start_ns: int
    end_ns: int
    source: str

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise PlanError(f"invalid half-open window: {self.start_ns}..{self.end_ns}")

    @property
    def start(self) -> str:
        return ns_to_iso(self.start_ns)

    @property
    def end(self) -> str:
        return ns_to_iso(self.end_ns)


@dataclass(frozen=True)
class Request:
    request_id: str
    date: str
    purpose: str
    schema: str
    symbol: str
    window: Window
    ordinal: int

    def api_request(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "schema": self.schema,
            "symbols": [self.symbol],
            "stype_in": STYPE_IN,
            "start": self.window.start,
            "end": self.window.end,
        }

    @property
    def relative_path(self) -> str:
        return f"{self.date}/{self.request_id}.dbn.zst"


@dataclass(frozen=True)
class VerifiedArtifact:
    request: Request
    path: Path
    size: int
    sha256: str
    quoted_cost_usd: Decimal | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_ns(value: Any) -> int:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        if "+" in text[10:] or text.endswith("+00:00"):
            main, offset = text.rsplit("+", 1)
            if ":" in offset:
                offset_seconds = int(offset[:2]) * 3600 + int(offset[3:5]) * 60
                if offset_seconds:
                    raise PlanError(f"non-UTC timestamp is not allowed: {value!r}")
            text = main
        elif text[10:].count("-") == 1:
            raise PlanError(f"non-UTC timestamp is not allowed: {value!r}")
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?", text)
        if not match:
            raise ValueError(text)
        base = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
        fraction = (match.group(2) or "").ljust(9, "0")[:9]
        return int(base.timestamp()) * NANOSECONDS + int(fraction or 0)
    except (ValueError, OverflowError) as exc:
        raise PlanError(f"invalid UTC timestamp: {value!r}") from exc


def ns_to_iso(value: int) -> str:
    seconds, nanos = divmod(int(value), NANOSECONDS)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "")
    return f"{base}.{nanos:09d}Z"


def session_bounds(day: str) -> tuple[int, int]:
    parsed = date.fromisoformat(day)
    start = datetime(parsed.year, parsed.month, parsed.day, 7, tzinfo=timezone.utc)
    end = datetime(parsed.year, parsed.month, parsed.day, 20, tzinfo=timezone.utc)
    return int(start.timestamp()) * NANOSECONDS, int(end.timestamp()) * NANOSECONDS


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise PlanError(f"JSON evidence is not an object: {path}")
    return payload


def _window_from_mapping(row: Mapping[str, Any], source: str) -> tuple[int, int] | None:
    start = row.get("start_utc_inclusive") or row.get("started_at") or row.get("start")
    end = row.get("end_utc_exclusive") or row.get("ended_at") or row.get("end")
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise PlanError(f"incomplete approved window in {source}: {row}")
    return _parse_ns(start), _parse_ns(end)


def _clip_and_merge(rows: Iterable[tuple[int, int]], day: str, source: str) -> list[Window]:
    session_start, session_end = session_bounds(day)
    clipped = []
    for start, end in rows:
        start = max(start, session_start)
        end = min(end, session_end)
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: list[list[int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [Window(start, end, source) for start, end in merged]


def _authoritative_windows(archive_root: Path, day: str) -> list[Window] | None:
    path = archive_root / "data" / "topstep-gap-repair" / day / "gap-repair-acquisition-manifest.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    plan = payload.get("plan")
    if not isinstance(plan, dict) or plan.get("trading_date") != day:
        raise PlanError(f"authoritative repair manifest identity mismatch: {path}")
    rows = plan.get("repair_windows")
    if not isinstance(rows, list) or not rows:
        raise PlanError(f"authoritative repair windows missing: {path}")
    pairs = []
    for row in rows:
        if not isinstance(row, dict):
            raise PlanError(f"invalid repair window in {path}")
        pair = _window_from_mapping(row, str(path))
        if pair:
            pairs.append(pair)
    return _clip_and_merge(pairs, day, str(path))


def _summary_windows(archive_root: Path, day: str) -> list[Window]:
    path = archive_root / "logs" / "topstep" / f"{day}-session-auto" / "summary.json"
    payload = _load_json(path)
    rows = payload.get("market_coverage_gaps")
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise PlanError(f"no market_coverage_gaps in approved source summary: {path}")
    pairs = []
    for row in rows:
        if not isinstance(row, dict):
            raise PlanError(f"invalid market coverage gap in {path}")
        pair = _window_from_mapping(row, str(path))
        if pair:
            pairs.append(pair)
    result = _clip_and_merge(pairs, day, str(path))
    if not result:
        raise PlanError(f"approved summary has no gaps inside 07:00Z-20:00Z: {path}")
    return result


def _existing_trade_end(archive_root: Path, day: str, symbol: str) -> int:
    path = archive_root / "data" / "topstep-session-profile-recovery" / day / "acquisition-manifest.json"
    payload = _load_json(path)
    request = payload.get("request")
    if not isinstance(request, dict) or request.get("symbol") != symbol:
        raise PlanError(f"existing trade-recovery identity mismatch: {path}")
    end = request.get("request_end_utc_exclusive")
    if not end:
        raise PlanError(f"existing trade-recovery end is missing: {path}")
    return _parse_ns(end)


def _windows_for_day(archive_root: Path, day: str) -> tuple[list[Window], list[Window]]:
    if day in FULL_SESSION_DATES:
        start, end = session_bounds(day)
        return [Window(start, end, "approved full-session window")], []

    depth_windows = _authoritative_windows(archive_root, day)
    if depth_windows is None:
        depth_windows = _summary_windows(archive_root, day)

    trade_windows: list[Window] = []
    if day == "2026-09-07":
        es_symbol = "ESZ6" if day in Z26_DATES else "ESU6"
        session_start, session_end = session_bounds(day)
        existing_end = _existing_trade_end(archive_root, day, es_symbol)
        start = max(session_start, existing_end)
        if start < session_end:
            trade_windows = [Window(start, session_end, "suffix after archived ES trade recovery")]
    return depth_windows, trade_windows


def build_plan(archive_root: Path) -> tuple[Request, ...]:
    if not archive_root.is_dir():
        raise PlanError(f"archive root does not exist: {archive_root}")
    requests: list[Request] = []
    for day in APPROVED_DATES:
        es_symbol, mes_symbol = (("ESZ6", "MESZ6") if day in Z26_DATES else ("ESU6", "MESU6"))
        depth_windows, trade_windows = _windows_for_day(archive_root, day)
        for ordinal, window in enumerate(depth_windows, 1):
            requests.append(Request(f"es-mbp-10-{ordinal:03d}-{window.start_ns}-{window.end_ns}", day,
                                    "ES_DEPTH", "mbp-10", es_symbol, window, ordinal))
            requests.append(Request(f"mes-mbp-1-{ordinal:03d}-{window.start_ns}-{window.end_ns}", day,
                                    "MES_BBO", "mbp-1", mes_symbol, window, ordinal))
        for ordinal, window in enumerate(trade_windows, 1):
            requests.append(Request(f"es-trades-{ordinal:03d}-{window.start_ns}-{window.end_ns}", day,
                                    "ES_TRADES", "trades", es_symbol, window, ordinal))
    if not requests:
        raise PlanError("approved request plan is empty")
    if any(item.schema == "mbo" for item in requests):
        raise PlanError("MBO is prohibited")
    return tuple(requests)


def _request_from_record(record: Mapping[str, Any]) -> Request:
    try:
        start = _parse_ns(record["start_utc_inclusive"])
        end = _parse_ns(record["end_utc_exclusive"])
        return Request(
            str(record["request_id"]),
            str(record["date"]),
            str(record["purpose"]),
            str(record["schema"]),
            str(record["symbol"]),
            Window(start, end, str(record.get("source", "manifest"))),
            int(record.get("ordinal", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"invalid request record in repair manifest: {record}") from exc


def inventory_verified_files(output_root: Path) -> tuple[VerifiedArtifact, ...]:
    """Inventory and revalidate all completed DBNs without touching them."""
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for manifest_name in (LEGACY_MANIFEST_NAME, MANIFEST_NAME):
        manifest_path = output_root / manifest_name
        if not manifest_path.is_file():
            continue
        payload = _load_json(manifest_path)
        records = payload.get("requests", {})
        if not isinstance(records, dict):
            raise PlanError(f"invalid requests section: {manifest_path}")
        for record in records.values():
            if isinstance(record, dict) and record.get("output_path"):
                records_by_path[str(record["output_path"])] = record

    artifacts: list[VerifiedArtifact] = []
    for path in sorted(output_root.rglob("*.dbn.zst")):
        relative = path.relative_to(output_root).as_posix()
        record = records_by_path.get(relative)
        if record is None:
            print(f"INVENTORY_UNRECORDED_SKIP path={path}")
            continue
        request = _request_from_record(record)
        try:
            verification = validate_dbn(path, request)
            size = path.stat().st_size
            digest = sha256_file(path)
            recorded_size = record.get("size")
            recorded_digest = record.get("sha256")
            if recorded_size is not None and int(recorded_size) != size:
                raise PlanError("manifest size mismatch")
            if recorded_digest is not None and str(recorded_digest) != digest:
                raise PlanError("manifest SHA-256 mismatch")
            quoted = record.get("quoted_cost_usd")
            artifacts.append(VerifiedArtifact(
                request=request,
                path=path,
                size=size,
                sha256=digest,
                quoted_cost_usd=Decimal(str(quoted)) if quoted is not None else None,
            ))
            print(
                f"INVENTORY_VERIFIED date={request.date} schema={request.schema} "
                f"symbol={request.symbol} start={request.window.start} end={request.window.end} "
                f"bytes={size} sha256={digest} range={verification['range_filter_field']}"
            )
        except (OSError, PlanError, ValueError) as exc:
            print(f"INVENTORY_UNVERIFIED_SKIP path={path} reason={exc}")
    return tuple(artifacts)


def _subtract_covered(window: Window, covered: Sequence[tuple[int, int]]) -> list[Window]:
    pieces: list[tuple[int, int]] = [(window.start_ns, window.end_ns)]
    for covered_start, covered_end in sorted(covered):
        remaining: list[tuple[int, int]] = []
        for start, end in pieces:
            if covered_end <= start or covered_start >= end:
                remaining.append((start, end))
                continue
            if start < covered_start:
                remaining.append((start, min(covered_start, end)))
            if covered_end < end:
                remaining.append((max(covered_end, start), end))
        pieces = remaining
    return [Window(start, end, window.source) for start, end in pieces if end > start]


def _merge_missing(windows: Sequence[Window], merge_gap_seconds: int) -> list[Window]:
    if not windows:
        return []
    threshold = merge_gap_seconds * NANOSECONDS
    merged: list[list[int]] = []
    for window in sorted(windows, key=lambda item: (item.start_ns, item.end_ns)):
        if not merged or window.start_ns - merged[-1][1] > threshold:
            merged.append([window.start_ns, window.end_ns])
        else:
            merged[-1][1] = max(merged[-1][1], window.end_ns)
    return [Window(start, end, f"practical block; merge_gap_seconds={merge_gap_seconds}")
            for start, end in merged]


def build_practical_plan(
    archive_root: Path,
    artifacts: Sequence[VerifiedArtifact],
    *,
    merge_gap_seconds: int,
) -> tuple[Request, ...]:
    """Build larger blocks around only still-missing source windows."""
    micro_requests = build_plan(archive_root)
    coverage: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for artifact in artifacts:
        key = (artifact.request.date, artifact.request.schema, artifact.request.symbol)
        coverage.setdefault(key, []).append((artifact.request.window.start_ns, artifact.request.window.end_ns))

    grouped: dict[tuple[str, str, str, str], list[Window]] = {}
    for request in micro_requests:
        key = (request.date, request.purpose, request.schema, request.symbol)
        covered = coverage.get((request.date, request.schema, request.symbol), [])
        grouped.setdefault(key, []).extend(_subtract_covered(request.window, covered))

    practical: list[Request] = []
    for (day, purpose, schema, symbol), windows in sorted(grouped.items()):
        for ordinal, window in enumerate(_merge_missing(windows, merge_gap_seconds), 1):
            request_id = f"practical-{purpose.lower()}-{ordinal:03d}-{window.start_ns}-{window.end_ns}"
            practical.append(Request(request_id, day, purpose, schema, symbol, window, ordinal))
    if any(item.schema == "mbo" for item in practical):
        raise PlanError("MBO is prohibited")
    return tuple(practical)


def practical_candidates(
    archive_root: Path, artifacts: Sequence[VerifiedArtifact],
) -> dict[int, tuple[Request, ...]]:
    return {
        level: build_practical_plan(archive_root, artifacts, merge_gap_seconds=level)
        for level in PRACTICAL_MERGE_LEVELS_SECONDS
    }


def _request_key(request: Request) -> str:
    return json.dumps(request.api_request(), sort_keys=True)


def _legacy_summary(
    micro_requests: Sequence[Request], artifacts: Sequence[VerifiedArtifact],
) -> tuple[int, Decimal, Decimal]:
    micro_ids = {request.request_id for request in micro_requests}
    verified_ids = {artifact.request.request_id for artifact in artifacts if artifact.request.request_id in micro_ids}
    paid = sum(
        (artifact.quoted_cost_usd or Decimal("0") for artifact in artifacts
         if artifact.request.request_id in micro_ids),
        Decimal("0"),
    )
    return len(micro_requests) - len(verified_ids), paid, EXPECTED_QUOTED_TOTAL_USD - paid


def _quote_requests(
    client: Any, requests: Sequence[Request], cache: dict[str, Decimal],
) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    for request in requests:
        key = _request_key(request)
        if key not in cache:
            cache[key] = _quote(client, request)
        result[request.request_id] = cache[key]
        print(f"QUOTE {request.request_id} ${_format_usd(cache[key])}")
    return result


def choose_practical_plan(
    client: Any,
    candidates: Mapping[int, Sequence[Request]],
    *,
    remaining_budget: Decimal,
) -> tuple[int, tuple[Request, ...], dict[str, Decimal], Decimal]:
    """Choose the fewest-request candidate whose fresh quote fits the budget."""
    quote_cache: dict[str, Decimal] = {}
    for merge_gap_seconds in sorted(candidates, reverse=True):
        requests = tuple(candidates[merge_gap_seconds])
        quotes = _quote_requests(client, requests, quote_cache)
        total = sum(quotes.values(), Decimal("0"))
        print(
            f"CANDIDATE merge_gap_seconds={merge_gap_seconds} "
            f"request_count={len(requests)} quoted_cost_usd={_format_usd(total)} "
            f"within_remaining_budget={str(total <= remaining_budget).lower()}"
        )
        if total <= remaining_budget:
            return merge_gap_seconds, requests, quotes, total
    raise PlanError(
        f"no practical candidate fits remaining budget ${remaining_budget}; "
        "no market-data download started"
    )


def _requests_from_manifest_plan(payload: Mapping[str, Any]) -> tuple[Request, ...] | None:
    rows = payload.get("plan")
    if not isinstance(rows, list) or not rows:
        return None
    requests = []
    for row in rows:
        if not isinstance(row, dict):
            raise PlanError("practical manifest contains an invalid plan row")
        requests.append(_request_from_record(row))
    return tuple(requests)


def _cost_decimal(value: Any) -> Decimal:
    if isinstance(value, Mapping):
        for key in ("cost", "cost_usd", "total", "amount"):
            if key in value:
                value = value[key]
                break
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise PlanError(f"Databento returned a non-numeric quote: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise PlanError(f"Databento returned an invalid quote: {value!r}")
    return result


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    if payload.get("dataset") != DATASET or payload.get("stype_in") != STYPE_IN:
        raise PlanError(f"existing manifest identity mismatch: {path}")
    return payload


def _manifest_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    return getattr(metadata, name, default)


def authoritative_range_timestamp(record: Any, *, record_index: int, path: Path) -> tuple[int, str]:
    """Return the timestamp Databento uses for ``get_range(start/end)``.

    Databento's Historical ``timeseries.get_range`` filters on ``ts_recv``
    whenever the schema contains it, otherwise it filters on ``ts_event``.
    Keeping this choice explicit prevents event-time latency from being
    mistaken for a request-boundary violation.
    """
    receive_timestamp = getattr(record, "ts_recv", None)
    if receive_timestamp is not None:
        return int(receive_timestamp), "ts_recv"
    event_timestamp = getattr(record, "ts_event", None)
    if event_timestamp is not None:
        return int(event_timestamp), "ts_event"
    raise PlanError(f"DBN record has no ts_recv or ts_event: {path}; record_index={record_index}")


def validate_record_bounds(record: Any, request: Request, *, record_index: int, path: Path) -> int:
    timestamp, range_field = authoritative_range_timestamp(record, record_index=record_index, path=path)
    if not request.window.start_ns <= timestamp < request.window.end_ns:
        event_timestamp = getattr(record, "ts_event", None)
        receive_timestamp = getattr(record, "ts_recv", None)
        raise PlanError(
            f"DBN record outside requested bounds: {path}; record_index={record_index}; "
            f"range_field={range_field}; ts_event={event_timestamp}; ts_recv={receive_timestamp}; "
            f"requested=[{request.window.start_ns},{request.window.end_ns})"
        )
    return timestamp


def validate_dbn(path: Path, request: Request) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PlanError(f"DBN target is missing or empty: {path}")
    try:
        from databento import DBNStore
        store = DBNStore.from_file(path)
        metadata = store.metadata
        dataset = str(_metadata_value(metadata, "dataset", ""))
        schema = str(_metadata_value(metadata, "schema", ""))
        symbols = tuple(str(item) for item in (_metadata_value(metadata, "symbols", ()) or ()))
        if dataset != DATASET or schema != request.schema or request.symbol not in symbols:
            raise PlanError(f"DBN metadata mismatch for {request.relative_path}")
        record_count = 0
        first_range_ns: int | None = None
        last_range_ns: int | None = None
        min_event_ns: int | None = None
        max_event_ns: int | None = None
        min_recv_ns: int | None = None
        max_recv_ns: int | None = None
        for record in store:
            timestamp = validate_record_bounds(record, request, record_index=record_count, path=path)
            event_timestamp = getattr(record, "ts_event", None)
            if event_timestamp is not None:
                event_timestamp = int(event_timestamp)
                min_event_ns = event_timestamp if min_event_ns is None else min(min_event_ns, event_timestamp)
                max_event_ns = event_timestamp if max_event_ns is None else max(max_event_ns, event_timestamp)
            receive_timestamp = getattr(record, "ts_recv", None)
            if receive_timestamp is not None:
                receive_timestamp = int(receive_timestamp)
                min_recv_ns = receive_timestamp if min_recv_ns is None else min(min_recv_ns, receive_timestamp)
                max_recv_ns = receive_timestamp if max_recv_ns is None else max(max_recv_ns, receive_timestamp)
            first_range_ns = timestamp if first_range_ns is None else min(first_range_ns, timestamp)
            last_range_ns = timestamp if last_range_ns is None else max(last_range_ns, timestamp)
            record_count += 1
        if record_count == 0:
            raise PlanError(f"DBN contains no records: {path}")
        return {
            "dataset": dataset,
            "schema": schema,
            "symbols": list(symbols),
            "record_count": record_count,
            "range_filter_field": "ts_recv" if min_recv_ns is not None else "ts_event",
            "first_timestamp_utc": ns_to_iso(first_range_ns),
            "last_timestamp_utc": ns_to_iso(last_range_ns),
            "min_ts_event_utc": ns_to_iso(min_event_ns) if min_event_ns is not None else None,
            "max_ts_event_utc": ns_to_iso(max_event_ns) if max_event_ns is not None else None,
            "min_ts_recv_utc": ns_to_iso(min_recv_ns) if min_recv_ns is not None else None,
            "max_ts_recv_utc": ns_to_iso(max_recv_ns) if max_recv_ns is not None else None,
        }
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError(f"DBN is unreadable: {path}") from exc


def _request_record(request: Request, quoted_cost: str | None = None) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "date": request.date,
        "purpose": request.purpose,
        "dataset": DATASET,
        "schema": request.schema,
        "symbol": request.symbol,
        "stype_in": STYPE_IN,
        "start_utc_inclusive": request.window.start,
        "end_utc_exclusive": request.window.end,
        "source": request.window.source,
        "quoted_cost_usd": quoted_cost,
        "output_path": request.relative_path,
    }


def _plan_hash(requests: Sequence[Request]) -> str:
    encoded = json.dumps([_request_record(item) for item in requests], sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_path(output_root: Path, request: Request) -> Path:
    return output_root / request.relative_path


def _partial_path(output_root: Path, request: Request) -> Path:
    destination = _target_path(output_root, request)
    return destination.with_suffix(destination.suffix + ".part")


def _existing_verified(output_root: Path, request: Request, record: Mapping[str, Any] | None) -> bool:
    path = _target_path(output_root, request)
    if not path.is_file():
        return False
    try:
        verification = validate_dbn(path, request)
        if record is not None and int(record.get("size", -1)) != path.stat().st_size:
            return False
        if record is not None and str(record.get("sha256")) != sha256_file(path):
            return False
        if not verification:
            return False
    except (OSError, PlanError, ValueError):
        return False
    return True


def _existing_partial_verified(output_root: Path, request: Request) -> dict[str, Any] | None:
    path = _partial_path(output_root, request)
    if not path.is_file():
        return None
    try:
        return validate_dbn(path, request)
    except (OSError, PlanError, ValueError):
        return None


def _quote(client: Any, request: Request, *, sleep_fn: Any = time.sleep,
           random_fn: Any = random.random) -> Decimal:
    return _cost_decimal(_retry_call(
        operation="quote",
        label=request.request_id,
        callback=lambda: client.metadata.get_cost(**request.api_request()),
        sleep_fn=sleep_fn,
        random_fn=random_fn,
    ))


def _format_usd(value: Decimal) -> str:
    return f"{value:.6f}"


def _print_plan(requests: Sequence[Request]) -> None:
    print(f"APPROVED_DATES={','.join(APPROVED_DATES)}")
    print(f"PLAN_REQUEST_COUNT={len(requests)}")
    for request in requests:
        print(json.dumps({**_request_record(request), "api_request": request.api_request()}, sort_keys=True))


def _prepare_manifest(requests: Sequence[Request], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_hash = _plan_hash(requests)
    existing = _load_manifest(output_root / MANIFEST_NAME)
    if existing is not None and existing.get("plan_sha256") != plan_hash:
        raise PlanError("existing download manifest belongs to a different approved request plan")
    progress = _load_manifest(output_root / PROGRESS_NAME)
    if progress is not None and progress.get("plan_sha256") != plan_hash:
        raise PlanError("existing progress manifest belongs to a different approved request plan")
    manifest = existing or {
        "status": "ACQUISITION_IN_PROGRESS",
        "dataset": DATASET,
        "stype_in": STYPE_IN,
        "approved_dates": list(APPROVED_DATES),
        "plan_sha256": plan_hash,
        "budget_usd": str(BUDGET_USD),
        "expected_quoted_total_usd": str(EXPECTED_QUOTED_TOTAL_USD),
        "plan": [_request_record(request) for request in requests],
        "requests": {},
    }
    progress = progress or {
        "status": "ACQUISITION_IN_PROGRESS",
        "dataset": DATASET,
        "stype_in": STYPE_IN,
        "approved_dates": list(APPROVED_DATES),
        "plan_sha256": plan_hash,
        "plan": [_request_record(request) for request in requests],
        "completed": {},
    }
    return manifest, progress


def _quote_pending(client: Any, requests: Sequence[Request], output_root: Path,
                   manifest: dict[str, Any], *, include_verified: bool = False,
                   sleep_fn: Any = time.sleep, random_fn: Any = random.random) -> dict[str, Decimal]:
    quotes: dict[str, Decimal] = {}
    pending = []
    for request in requests:
        existing = manifest.get("requests", {}).get(request.request_id)
        if not include_verified and (
            _existing_verified(output_root, request, existing)
            or _existing_partial_verified(output_root, request) is not None
        ):
            continue
        pending.append(request)
    cumulative = Decimal("0")
    for request in pending:
        cost = _quote(client, request, sleep_fn=sleep_fn, random_fn=random_fn)
        quotes[request.request_id] = cost
        cumulative += cost
        print(f"QUOTE {request.request_id} ${_format_usd(cost)} cumulative_pending=${_format_usd(cumulative)}")
    print(f"PENDING_QUOTED_TOTAL_USD={_format_usd(cumulative)}")
    if cumulative > BUDGET_USD:
        raise PlanError(f"projected pending total ${cumulative} exceeds hard cap ${BUDGET_USD}; no download started")
    return quotes


def _download(client: Any, requests: Sequence[Request], output_root: Path,
              manifest: dict[str, Any], progress: dict[str, Any],
              quotes: Mapping[str, Decimal], *, sleep_fn: Any = time.sleep,
              random_fn: Any = random.random) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for request in requests:
        destination = _target_path(output_root, request)
        existing_record = manifest.setdefault("requests", {}).get(request.request_id)
        if _existing_verified(output_root, request, existing_record):
            if existing_record is None:
                verification = validate_dbn(destination, request)
                manifest["requests"][request.request_id] = {
                    **_request_record(request),
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "verification": {"status": "VERIFIED_EXISTING", **verification},
                }
                progress["completed"][request.request_id] = manifest["requests"][request.request_id]
                _manifest_write(output_root / MANIFEST_NAME, manifest)
                _manifest_write(output_root / PROGRESS_NAME, progress)
            print(f"SKIP_VERIFIED {request.request_id} {destination}")
            continue
        partial = _partial_path(output_root, request)
        partial_verification = _existing_partial_verified(output_root, request)
        if partial_verification is not None:
            size = partial.stat().st_size
            digest = sha256_file(partial)
            os.replace(partial, destination)
            record = {
                **_request_record(request),
                "size": size,
                "sha256": digest,
                "verification": {"status": "PROMOTED_VERIFIED_PART", **partial_verification},
            }
            manifest["requests"][request.request_id] = record
            progress["completed"][request.request_id] = record
            _manifest_write(output_root / MANIFEST_NAME, manifest)
            _manifest_write(output_root / PROGRESS_NAME, progress)
            print(f"PROMOTED_VERIFIED_PART {request.request_id} {destination}")
            continue
        if partial.exists():
            partial.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            def download_attempt() -> Any:
                if partial.exists():
                    partial.unlink()
                return client.timeseries.get_range(**request.api_request(), path=str(partial))

            _retry_call(
                operation="download",
                label=request.request_id,
                callback=download_attempt,
                sleep_fn=sleep_fn,
                random_fn=random_fn,
            )
            verification = validate_dbn(partial, request)
            size = partial.stat().st_size
            digest = sha256_file(partial)
            os.replace(partial, destination)
            record = {
                **_request_record(request, _format_usd(quotes[request.request_id])),
                "size": size,
                "sha256": digest,
                "verification": {"status": "VERIFIED", **verification},
            }
            manifest["requests"][request.request_id] = record
            progress["completed"][request.request_id] = record
            _manifest_write(output_root / MANIFEST_NAME, manifest)
            _manifest_write(output_root / PROGRESS_NAME, progress)
            print(f"DOWNLOADED_VERIFIED {request.request_id} bytes={size} sha256={digest}")
        except Exception:
            # Preserve the partial for local forensic validation and a later
            # resume.  A subsequent resume removes it only after it has failed
            # the same validation and is about to issue that request again.
            raise
    manifest["status"] = "ACQUISITION_COMPLETE_VERIFIED"
    progress["status"] = "ACQUISITION_COMPLETE_VERIFIED"
    _manifest_write(output_root / MANIFEST_NAME, manifest)
    _manifest_write(output_root / PROGRESS_NAME, progress)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="quote metadata only; never download")
    mode.add_argument("--download", action="store_true", help="quote and acquire the approved plan")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--remaining-budget", type=Decimal, default=DEFAULT_REMAINING_BUDGET_USD)
    args = parser.parse_args(argv)

    archive_root = args.archive_root.expanduser().resolve()
    output_root = args.output_root
    artifacts = inventory_verified_files(output_root)
    micro_requests = build_plan(archive_root)
    old_pending_count, paid_cost, old_pending_cost = _legacy_summary(micro_requests, artifacts)
    print(f"OLD_MICRO_REQUEST_COUNT={len(micro_requests)}")
    print(f"OLD_MICRO_PENDING_REQUEST_COUNT={old_pending_count}")
    print(f"ALREADY_PAID_COST_USD={_format_usd(paid_cost)}")
    print(f"OLD_MICRO_PENDING_COST_USD={_format_usd(old_pending_cost)}")
    print(f"REMAINING_BUDGET_USD={_format_usd(args.remaining_budget)}")

    existing_manifest = _load_manifest(output_root / MANIFEST_NAME)
    requests = _requests_from_manifest_plan(existing_manifest) if existing_manifest else None
    selected_merge_gap: int | None = None
    initial_quotes: dict[str, Decimal] = {}

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise PlanError("DATABENTO_API_KEY is required for cost quotes and is never printed")
    try:
        import databento as db
    except ImportError as exc:
        raise PlanError("databento package is required") from exc
    client = db.Historical(key=api_key)
    if requests is None:
        candidates = practical_candidates(archive_root, artifacts)
        selected_merge_gap, requests, initial_quotes, new_cost = choose_practical_plan(
            client, candidates, remaining_budget=args.remaining_budget,
        )
        print(f"SELECTED_MERGE_GAP_SECONDS={selected_merge_gap}")
    else:
        print(f"RESUMING_PRACTICAL_PLAN_REQUEST_COUNT={len(requests)}")

    _print_plan(requests)
    manifest, progress = _prepare_manifest(requests, output_root)
    if selected_merge_gap is not None:
        manifest["selected_merge_gap_seconds"] = selected_merge_gap
        progress["selected_merge_gap_seconds"] = selected_merge_gap
    quotes = initial_quotes or _quote_pending(client, requests, output_root, manifest)
    new_cost = sum(quotes.values(), Decimal("0"))
    print(f"NEW_PENDING_REQUEST_COUNT={len(quotes)}")
    print(f"NEW_PENDING_COST_USD={_format_usd(new_cost)}")
    print(f"TOTAL_EXPECTED_FINAL_COST_USD={_format_usd(paid_cost + new_cost)}")
    print(f"WITHIN_REMAINING_BUDGET={str(new_cost <= args.remaining_budget).lower()}")
    if args.dry_run:
        print("DRY_RUN_NO_DOWNLOAD=true")
        return 0
    _manifest_write(output_root / MANIFEST_NAME, manifest)
    _manifest_write(output_root / PROGRESS_NAME, progress)
    _download(client, requests, output_root, manifest, progress, quotes)
    print(f"ACQUISITION_COMPLETE=true output_root={output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
