"""Resumable, fail-closed Algoseek REST acquisition for local CME L2 inputs.

This is deliberately an acquisition boundary: it never starts a research
stage, does not alter raw manual exports, and writes only to an explicitly
selected output root.  The public data API does not provide a cross-stream
sequence key; downloaded files therefore retain the existing
``ALGOSEEK_CAUSAL_VARIANT`` semantics when read by :mod:`algoseek_adapter`.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


PRODUCTION_API_BASE_URL = "https://api.algoseek.com/v1"
API_BASE_URL = PRODUCTION_API_BASE_URL
API_BASE_URL_ENV = "ALGOSEEK_API_BASE_URL"
API_KEY_ENV = "ALGOSEEK_API_KEY"
CSV_GZIP_FORMAT = "csv_gzip"
CSV_GZIP_MAX_LIMIT = 80_000
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 4
SOURCE_TIMEZONE = ZoneInfo("America/Chicago")
# Production probes show that EventDateTime filter values are evaluated in
# the Eastern wall-clock frame while returned futures EventDateTime values
# are Chicago wall-clock values. Keep this as a named, auditable assumption
# and use zoneinfo rather than a fixed one-hour arithmetic adjustment.
PROVIDER_FILTER_TIMEZONE = ZoneInfo("America/New_York")
PROVIDER_FILTER_TIMEZONE_NAME = "America/New_York"
PARTITION_END_COVERAGE_GRACE = timedelta(minutes=5)
MANIFEST_VERSION = 2

ACCOUNT_IDENTITY_ENDPOINT = "/account/my"
ACCOUNT_QUOTAS_ENDPOINT = "/account/my/quotas"
ACCOUNT_ACCESS_RULES_ENDPOINT = "/account/my/data-access-rules"
MY_DATASETS_ENDPOINT = "/meta/datasets/my"
FUTURES_DEPTH_ENDPOINT = "/data/us-futures/multiple-depth/{trade_date}/{ticker}"
FUTURES_TAQ_ENDPOINT = "/data/us-futures/taq/{trade_date}/{ticker}"

REQUIRED_ENTITLEMENTS = {
    "US6002": {"display_name": "US Futures Multiple Depth", "tickers": ("ESH3", "ESM3")},
    "US6011": {"display_name": "US Futures Trade and Quote", "tickers": ("ESH3", "MESH3", "ESM3", "MESM3")},
}
Q1_2023_START = date(2023, 1, 1)
Q1_2023_END = date(2023, 3, 31)

DEPTH = "es-multiple-depth"
ES_TAQ = "es-trade-and-quote"
MES_TAQ = "mes-trade-and-quote"
FEEDS = (DEPTH, ES_TAQ, MES_TAQ)
Q1_2023_CME_CLOSED = frozenset({date(2023, 1, 2), date(2023, 1, 16), date(2023, 2, 20)})


class AlgoseekAPIError(RuntimeError):
    """The remote API or locally persisted acquisition state is unsafe."""


class AlgoseekAuthenticationError(AlgoseekAPIError):
    pass


class AlgoseekEntitlementError(AlgoseekAPIError):
    pass


class AlgoseekDownloadError(AlgoseekAPIError):
    pass


@dataclass(frozen=True)
class ContractMap:
    es: str
    mes: str
    version: str = "q1-2023-front-month-v1"


@dataclass(frozen=True)
class SessionWindow:
    logical_date: date
    start: datetime
    end: datetime

    @property
    def provider_trade_dates(self) -> tuple[date, date]:
        return self.start.date(), self.end.date()

    @property
    def provider_trade_date_windows(self) -> tuple[tuple[date, datetime, datetime], tuple[date, datetime, datetime]]:
        """Exact [start, end) slices for the two calendar-day API partitions."""
        midnight = datetime.combine(self.end.date(), datetime.min.time(), SOURCE_TIMEZONE)
        return ((self.start.date(), self.start, midnight), (self.end.date(), midnight, self.end))


def hourly_session_partitions(session: SessionWindow) -> tuple[tuple[date, datetime, datetime], ...]:
    """Return every expected one-hour local [start, end) session partition."""
    partitions: list[tuple[date, datetime, datetime]] = []
    current = session.start
    while current < session.end:
        end = min(current + timedelta(hours=1), session.end)
        partitions.append((current.date(), current, end))
        current = end
    return tuple(partitions)


@dataclass(frozen=True)
class Response:
    body: Any
    headers: Mapping[str, str]
    status: int


def contract_map_for(session_date: date) -> ContractMap:
    """Return the explicitly approved Q1 2023 front-month mapping."""
    return ContractMap("ESH3", "MESH3") if session_date <= date(2023, 3, 12) else ContractMap("ESM3", "MESM3")


def logical_session_window(session_date: date) -> SessionWindow:
    start = datetime.combine(session_date - timedelta(days=1), datetime.min.time(), SOURCE_TIMEZONE).replace(hour=17)
    end = datetime.combine(session_date, datetime.min.time(), SOURCE_TIMEZONE).replace(hour=16)
    return SessionWindow(session_date, start, end)


def is_cme_closed_session(session_date: date) -> bool:
    """Q1 2023 full closures; early closes remain valid sessions."""
    return session_date.weekday() >= 5 or session_date in Q1_2023_CME_CLOSED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _short_http_body(error: HTTPError, *, limit: int = 512) -> str:
    """Return a bounded API-error body; credentials are never part of URLs or bodies."""
    try:
        body = error.read(limit + 1)
    except OSError:
        return ""
    text = body[:limit].decode("utf-8", errors="replace").strip()
    return text + ("…" if len(body) > limit else "")


def _iso_local(value: datetime) -> str:
    # API filter values and returned futures EventDateTime are wall timestamps;
    # the caller is responsible for selecting the relevant timezone frame.
    return value.strftime("%Y-%m-%d %H:%M:%S")


def api_filter_bounds_for_local_window(
    local_start: datetime, local_end: datetime, provider_trade_date: date,
) -> tuple[str, str]:
    """Translate desired Chicago bounds into the provider's filter time frame.

    The production API probes established that a filter value of 23:00 was
    applied as 22:00 in the returned Chicago EventDateTime stream. The
    observed frame is therefore Eastern wall time. Converting between named
    IANA zones keeps the translation explicit and DST-aware instead of
    scattering a fixed ``+1 hour`` adjustment through request construction.
    """
    if local_start.tzinfo is None or local_end.tzinfo is None:
        raise AlgoseekDownloadError("local EventDateTime bounds must be timezone-aware")
    desired_start = local_start.astimezone(SOURCE_TIMEZONE)
    desired_end = local_end.astimezone(SOURCE_TIMEZONE)
    if desired_start >= desired_end:
        raise AlgoseekDownloadError("local EventDateTime window must be non-empty")
    if desired_start.date() != provider_trade_date:
        raise AlgoseekDownloadError(
            f"provider TradeDate {provider_trade_date} does not own local partition {desired_start.date()}"
        )
    translated_start = desired_start.astimezone(PROVIDER_FILTER_TIMEZONE)
    translated_end = desired_end.astimezone(PROVIDER_FILTER_TIMEZONE)
    return _iso_local(translated_start), _iso_local(translated_end)


def _parse_local_timestamp(value: str) -> datetime:
    value = value.strip().replace("T", " ")
    if value.endswith("Z"):
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(SOURCE_TIMEZONE)
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(SOURCE_TIMEZONE) if parsed.tzinfo else parsed.replace(tzinfo=SOURCE_TIMEZONE)


def _dataset_name(feed: str) -> str:
    return "US Futures Multiple Depth" if feed == DEPTH else "US Futures Trade and Quote"


def _feed_spec(feed: str, contracts: ContractMap) -> tuple[str, str, str, str]:
    if feed == DEPTH:
        return FUTURES_DEPTH_ENDPOINT, contracts.es, "ES", "chunk"
    if feed == ES_TAQ:
        return FUTURES_TAQ_ENDPOINT, contracts.es, "ES", "es-chunk"
    if feed == MES_TAQ:
        return FUTURES_TAQ_ENDPOINT, contracts.mes, "MES", "mes-chunk"
    raise AlgoseekDownloadError(f"unknown feed: {feed}")


class AlgoseekAPIClient:
    """Small stdlib client with bounded retry and no credential persistence."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise AlgoseekAuthenticationError(f"{API_KEY_ENV} is required; set it in the environment")
        self.base_url = (base_url or os.environ.get(API_BASE_URL_ENV) or API_BASE_URL).rstrip("/")
        self.timeout_seconds, self.max_retries, self._opener, self._sleep = timeout_seconds, max_retries, opener, sleep

    def _url(self, endpoint: str, query: Mapping[str, Any] | None = None) -> str:
        query_string = urlencode({key: str(value) for key, value in (query or {}).items() if value is not None})
        return f"{self.base_url}{endpoint}" + (f"?{query_string}" if query_string else "")

    def open(self, endpoint: str, query: Mapping[str, Any] | None = None) -> Response:
        url = self._url(endpoint, query)
        request = Request(url, headers={"X-API-KEY": self.api_key, "User-Agent": "cme-l2-research/algoseek-api-v1"})
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._opener(request, timeout=self.timeout_seconds)
                return Response(raw, dict(raw.headers.items()), int(getattr(raw, "status", raw.getcode())))
            except HTTPError as exc:
                detail = f"method=GET url={url} status={exc.code}"
                body = _short_http_body(exc)
                if body:
                    detail += f" response_body={body!r}"
                if exc.code == 401:
                    raise AlgoseekAuthenticationError(f"Algoseek rejected ALGOSEEK_API_KEY ({detail})") from exc
                if exc.code == 403:
                    raise AlgoseekEntitlementError(f"Algoseek denied this account/IP/dataset request ({detail})") from exc
                retry_after = _header(dict(exc.headers.items()) if exc.headers else {}, "Retry-After")
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise AlgoseekAPIError(f"Algoseek HTTP error ({detail}; reason={exc.reason})") from exc
                self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0 ** attempt))
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise AlgoseekAPIError(f"Algoseek network failure after {attempt + 1} attempts: {exc}") from exc
                self._sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    def get_json(self, endpoint: str) -> Any:
        response = self.open(endpoint)
        try:
            return json.load(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AlgoseekAPIError(f"non-JSON response from {endpoint}") from exc
        finally:
            response.body.close()


def _normalize_dataset_name(value: Any) -> str:
    """Display-name fallback only; treat '&' and 'and' as equivalent."""
    words = str(value).lower().replace("&", " and ").split()
    return "".join(character for character in " ".join(words) if character.isalnum())


def _access_rule_records(value: Any) -> list[dict[str, Any]]:
    """Normalize the documented list-shaped entitlement response without guessing omissions."""
    entries = value if isinstance(value, list) else value.get("data", value.get("rules", [])) if isinstance(value, Mapping) else []
    result: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        result.append({key: entry.get(key) for key in ("dataset_id", "dataset_text_id", "dataset_name", "dataset_version", "start_date", "end_date", "universe_identifiers")})
    return result


def _required_entitlement_access(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate the two required feeds by stable dataset ID, then name fallback."""
    result: dict[str, dict[str, Any]] = {}
    for dataset_id, requirement in REQUIRED_ENTITLEMENTS.items():
        direct = [item for item in records if str(item.get("dataset_id", "")).upper() == dataset_id]
        fallback = [item for item in records if _normalize_dataset_name(item.get("dataset_name")) == _normalize_dataset_name(requirement["display_name"])]
        candidates, match = (direct, "dataset_id") if direct else (fallback, "normalized_name" if fallback else "none")
        summary: dict[str, Any] = {"dataset_id": dataset_id, "required_dataset": requirement["display_name"], "match": match,
                                   "access": "not_entitled", "date_range": None, "universe_access": "not_entitled"}
        for item in candidates:
            try:
                start = date.fromisoformat(str(item["start_date"])) if item.get("start_date") else None
                end = date.fromisoformat(str(item["end_date"])) if item.get("end_date") else None
            except ValueError:
                summary.update({"access": "unknown", "date_range": {"start_date": item.get("start_date"), "end_date": item.get("end_date")},
                                "universe_access": "unknown"})
                continue
            universe = item.get("universe_identifiers")
            allowed_tickers = {str(value).upper() for value in universe} if isinstance(universe, list) else set()
            universe_access = "accessible" if universe in (None, []) or all(ticker in allowed_tickers for ticker in requirement["tickers"]) else "restricted"
            date_access = (start is None or start <= Q1_2023_START) and (end is None or end >= Q1_2023_END)
            summary.update({"provider_dataset_name": item.get("dataset_name"), "date_range": {"start_date": item.get("start_date"), "end_date": item.get("end_date")},
                            "universe_access": universe_access,
                            "access": "accessible" if date_access and universe_access == "accessible" else "restricted"})
            if summary["access"] == "accessible":
                break
        result[dataset_id] = summary
    return result


def preflight(client: AlgoseekAPIClient) -> dict[str, Any]:
    """Read identity, quotas, catalog and access rules; never fetches market data."""
    identity = client.get_json(ACCOUNT_IDENTITY_ENDPOINT)
    quotas = client.get_json(ACCOUNT_QUOTAS_ENDPOINT)
    access_rules = client.get_json(ACCOUNT_ACCESS_RULES_ENDPOINT)
    # Identity validates the supplied key and IP allow-list. It is intentionally
    # not emitted: preflight output is limited to entitlement decisions.
    del identity
    rules = _access_rule_records(access_rules)
    required_datasets = _required_entitlement_access(rules)
    ready = all(item["access"] == "accessible" for item in required_datasets.values())
    return {
        "status": "ALGOSEEK_API_PREFLIGHT_READY" if ready else "ALGOSEEK_API_PREFLIGHT_NOT_READY",
        "ready": ready, "q1_2023_access": "accessible" if ready else "not_accessible",
        "required_datasets": required_datasets, "quota": quotas,
    }


def _validate_and_write_page(
    *, response: Response, destination: Path, expected_ticker: str, expected_trade_date: date,
    start: datetime, end: datetime, expected_header: Sequence[str] | None,
) -> tuple[list[str], int, int, str, int, str | None, str | None, str | None, str | None]:
    """Boundedly validate gzip CSV and atomically publish an independent page."""
    part = destination.with_suffix(destination.suffix + ".part")
    source_part = destination.with_suffix(destination.suffix + ".wire.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_part.open("wb") as wire:
            shutil.copyfileobj(response.body, wire, length=1_048_576)
        if source_part.stat().st_size == 0:
            raise AlgoseekDownloadError("empty gzip response")
        try:
            source = gzip.open(source_part, "rt", newline="", encoding="utf-8-sig")
            first_reader = csv.reader(source)
            first = next(first_reader, None)
        except (OSError, EOFError, UnicodeDecodeError, csv.Error) as exc:
            raise AlgoseekDownloadError("response is not a valid gzip CSV") from exc
        finally:
            try:
                source.close()
            except UnboundLocalError:
                pass
        header = list(first or ()) if expected_header is None else list(expected_header)
        if not header or "TradeDate" not in header or "EventDateTime" not in header or "Ticker" not in header:
            raise AlgoseekDownloadError("CSV schema lacks TradeDate, EventDateTime, or Ticker")
        has_header = list(first or ()) == header
        if expected_header is not None and has_header is False and len(first or ()) != len(header):
            raise AlgoseekDownloadError("subsequent CSV page has incompatible column count")
        with gzip.open(source_part, "rt", newline="", encoding="utf-8-sig") as input_handle, \
             gzip.open(part, "wt", newline="", encoding="utf-8") as output_handle:
            reader = csv.DictReader(input_handle, fieldnames=None if has_header else header)
            if has_header:
                if reader.fieldnames != header:
                    raise AlgoseekDownloadError("CSV page header changed during validation")
            writer = csv.DictWriter(output_handle, fieldnames=header, extrasaction="raise")
            writer.writeheader()
            returned = retained = 0
            first_returned_timestamp = last_returned_timestamp = None
            first_retained_timestamp = last_retained_timestamp = None
            for row in reader:
                returned += 1
                if row.get("Ticker", "").strip() != expected_ticker:
                    raise AlgoseekDownloadError(f"response ticker mismatch: {row.get('Ticker')!r}, expected {expected_ticker!r}")
                if row.get("TradeDate", "").strip() != expected_trade_date.isoformat():
                    raise AlgoseekDownloadError(f"response TradeDate mismatch: {row.get('TradeDate')!r}, expected {expected_trade_date}")
                timestamp = _parse_local_timestamp(row["EventDateTime"])
                raw_timestamp = row["EventDateTime"].strip()
                first_returned_timestamp = first_returned_timestamp or raw_timestamp
                last_returned_timestamp = raw_timestamp
                if start <= timestamp < end:
                    writer.writerow(row)
                    retained += 1
                    first_retained_timestamp = first_retained_timestamp or raw_timestamp
                    last_retained_timestamp = raw_timestamp
        if returned == 0:
            raise AlgoseekDownloadError("empty API page is not a successful market-data response")
        if retained == 0:
            raise AlgoseekDownloadError("API page has no rows inside the requested logical-session window")
        if returned != retained:
            raise AlgoseekDownloadError("API page returned rows outside the intended local partition")
        digest, size = _sha256(part), part.stat().st_size
        os.replace(part, destination)
        return (header, returned, retained, digest, size, first_returned_timestamp,
                last_returned_timestamp, first_retained_timestamp, last_retained_timestamp)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    finally:
        response.body.close()
        source_part.unlink(missing_ok=True)


def _manifest_path(root: Path, session_date: date) -> Path:
    return root / session_date.isoformat() / "algoseek-download-manifest.json"


def _complete_manifest_path(root: Path, session_date: date) -> Path:
    return root / session_date.isoformat() / "algoseek-complete-session.json"


def _new_manifest(session_date: date, contracts: ContractMap, output_root: Path, page_limit: int) -> dict[str, Any]:
    window = logical_session_window(session_date)
    return {"schema_version": MANIFEST_VERSION, "provider": "ALGOSEEK", "provider_semantics": "ALGOSEEK_CAUSAL_VARIANT",
            "logical_session_date": session_date.isoformat(), "session_start": _iso_local(window.start), "session_end": _iso_local(window.end),
            "provider_trade_dates": [value.isoformat() for value in window.provider_trade_dates],
            "contracts": {"ES": contracts.es, "MES": contracts.mes, "mapping_version": contracts.version},
            "output_root": str(output_root), "page_limit": page_limit, "response_format": CSV_GZIP_FORMAT,
            "partitioning": {"kind": "hourly-local", "timezone": "America/Chicago", "bounds": "[start,end)"},
            "filter_translation": {"timezone": PROVIDER_FILTER_TIMEZONE_NAME, "assumption": "provider filters use Eastern wall time"},
            "partitions": [], "pages": [], "session_complete": False, "audit": None}


def _load_resume_manifest(path: Path, session_date: date, contracts: ContractMap, page_limit: int) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlgoseekDownloadError(f"cannot read resume manifest: {path}") from exc
    if (manifest.get("schema_version"), manifest.get("logical_session_date"), manifest.get("contracts", {}).get("ES"),
        manifest.get("contracts", {}).get("MES"), manifest.get("page_limit")) != (MANIFEST_VERSION, session_date.isoformat(), contracts.es, contracts.mes, page_limit):
        raise AlgoseekDownloadError("resume manifest does not match this session, contracts, or page limit")
    for page in manifest.get("pages", []):
        if page.get("completion_state") != "COMPLETE":
            raise AlgoseekDownloadError("resume manifest contains incomplete page; remove only its .part file and retry")
        candidate = path.parent / page["filename"]
        if not candidate.is_file() or candidate.stat().st_size != page.get("file_size") or _sha256(candidate) != page.get("sha256"):
            raise AlgoseekDownloadError(f"resume manifest/file hash mismatch: {candidate}")
    return manifest


def _page_records(
    manifest: Mapping[str, Any], feed: str, trade_date: date, start: datetime, end: datetime,
) -> list[Mapping[str, Any]]:
    return [page for page in manifest.get("pages", [])
            if page.get("feed") == feed
            and page.get("provider_trade_date") == trade_date.isoformat()
            and page.get("partition_start") == _iso_local(start)
            and page.get("partition_end") == _iso_local(end)]


def _partition_summary(
    manifest: Mapping[str, Any], feed: str, trade_date: date, start: datetime, end: datetime,
) -> dict[str, Any]:
    pages = _page_records(manifest, feed, trade_date, start, end)
    if not pages:
        raise AlgoseekDownloadError(f"missing expected {feed} partition {start} -> {end}")
    if any(page.get("completion_state") != "COMPLETE" for page in pages):
        raise AlgoseekDownloadError(f"incomplete {feed} partition {start} -> {end}")
    if any(page.get("api_filter_start") != pages[0].get("api_filter_start") or
           page.get("api_filter_end") != pages[0].get("api_filter_end") for page in pages):
        raise AlgoseekDownloadError(f"inconsistent API filter bounds in {feed} partition {start} -> {end}")
    retained = sum(int(page.get("retained_row_count", 0)) for page in pages)
    returned = sum(int(page.get("returned_row_count", 0)) for page in pages)
    if retained <= 0:
        raise AlgoseekDownloadError(f"zero retained rows in active {feed} partition {start} -> {end}")
    first_values = [page.get("first_retained_timestamp") for page in pages if page.get("first_retained_timestamp")]
    last_values = [page.get("last_retained_timestamp") for page in pages if page.get("last_retained_timestamp")]
    first = min(first_values, key=_parse_local_timestamp)
    last = max(last_values, key=_parse_local_timestamp)
    if _parse_local_timestamp(first) < start or _parse_local_timestamp(last) >= end:
        raise AlgoseekDownloadError(f"retained rows escaped local partition {start} -> {end}")
    if _parse_local_timestamp(last) < end - PARTITION_END_COVERAGE_GRACE:
        raise AlgoseekDownloadError(
            f"{feed} partition stopped before its expected end coverage: last={last} end={_iso_local(end)}"
        )
    return {"feed": feed, "provider_trade_date": trade_date.isoformat(), "partition_start": _iso_local(start),
            "partition_end": _iso_local(end), "api_filter_start": pages[0].get("api_filter_start"),
            "api_filter_end": pages[0].get("api_filter_end"), "first_returned_timestamp": pages[0].get("first_returned_timestamp"),
            "last_returned_timestamp": pages[-1].get("last_returned_timestamp"), "first_retained_timestamp": first,
            "last_retained_timestamp": last, "returned_row_count": returned, "retained_row_count": retained,
            "page_count": len(pages), "completion_state": "COMPLETE"}


def _upsert_partition_summary(manifest: dict[str, Any], summary: Mapping[str, Any]) -> None:
    key = (summary["feed"], summary["partition_start"], summary["partition_end"])
    manifest.setdefault("partitions", [])
    manifest["partitions"] = [item for item in manifest["partitions"]
                                if (item.get("feed"), item.get("partition_start"), item.get("partition_end")) != key]
    manifest["partitions"].append(dict(summary))


def _download_feed(
    *, client: AlgoseekAPIClient, manifest: dict[str, Any], manifest_path: Path, root: Path, feed: str,
    contracts: ContractMap, session: SessionWindow, page_limit: int,
) -> None:
    endpoint_template, ticker, base_symbol, prefix = _feed_spec(feed, contracts)
    endpoint = endpoint_template.format(trade_date="{trade_date}", ticker=ticker)
    header: list[str] | None = None
    page_index = len([page for page in manifest["pages"] if page.get("feed") == feed])
    for trade_date, requested_start, requested_end in hourly_session_partitions(session):
        api_filter_start, api_filter_end = api_filter_bounds_for_local_window(
            requested_start, requested_end, trade_date,
        )
        prior = _page_records(manifest, feed, trade_date, requested_start, requested_end)
        offsets = [int(page["pagination_offset"]) for page in prior]
        if len(offsets) != len(set(offsets)):
            raise AlgoseekDownloadError(f"duplicate page offsets in manifest for {feed} {trade_date}")
        if prior:
            header = list(prior[0].get("csv_header", ())) or header
        offset = int(prior[-1]["next_offset"]) if prior and prior[-1].get("next_offset") is not None else None
        if prior and offset is None:
            _upsert_partition_summary(manifest, _partition_summary(manifest, feed, trade_date, requested_start, requested_end))
            continue
        offset = 0 if offset is None else offset
        while True:
            request_endpoint = endpoint.format(trade_date=trade_date.isoformat())
            query = {"limit": page_limit, "offset": offset, "response_format": CSV_GZIP_FORMAT,
                     "EventDateTime.ge": api_filter_start, "EventDateTime.lt": api_filter_end,
                     "sort": "+EventDateTime"}
            response = client.open(request_endpoint, query)
            next_offset_raw = _header(response.headers, "X-Pagination-Next-Offset")
            next_offset = int(next_offset_raw) if next_offset_raw not in (None, "") else None
            page_index += 1
            filename = f"{prefix}-{page_index:06d}.csv.gz"
            destination = root / session.logical_date.isoformat() / feed / filename
            (header, returned, retained, digest, size, first_returned, last_returned,
             first_retained, last_retained) = _validate_and_write_page(
                response=response, destination=destination, expected_ticker=ticker, expected_trade_date=trade_date,
                start=requested_start, end=requested_end, expected_header=header,
            )
            manifest["pages"].append({"provider": "ALGOSEEK", "endpoint": request_endpoint, "dataset": _dataset_name(feed),
                    "feed": feed, "ticker": ticker, "base_symbol": base_symbol, "logical_session_date": session.logical_date.isoformat(),
                    "provider_trade_date": trade_date.isoformat(), "requested_start": _iso_local(requested_start), "requested_end": _iso_local(requested_end),
                    "partition_start": _iso_local(requested_start), "partition_end": _iso_local(requested_end),
                    "api_filter_start": api_filter_start, "api_filter_end": api_filter_end,
                    "pagination_offset": offset, "next_offset": next_offset, "page_limit": page_limit, "returned_row_count": returned,
                    "retained_row_count": retained, "first_returned_timestamp": first_returned, "last_returned_timestamp": last_returned,
                    "first_retained_timestamp": first_retained, "last_retained_timestamp": last_retained,
                    "filename": str(destination.relative_to(manifest_path.parent)), "file_size": size, "sha256": digest,
                    "request_timestamp_utc": datetime.now().astimezone().isoformat(), "response_request_id": _header(response.headers, "X-Request-ID"),
                    "response_format": CSV_GZIP_FORMAT, "csv_header": header, "completion_state": "COMPLETE"})
            _atomic_json(manifest_path, manifest)
            if next_offset is None:
                break
            if next_offset <= offset:
                raise AlgoseekDownloadError(f"non-advancing pagination offset {offset} -> {next_offset}")
            offset = next_offset
        _upsert_partition_summary(manifest, _partition_summary(manifest, feed, trade_date, requested_start, requested_end))
        _atomic_json(manifest_path, manifest)


def _input_paths(root: Path, session_date: date) -> tuple[list[Path], list[Path], list[Path]]:
    session_root = root / session_date.isoformat()
    def paths(feed: str) -> list[Path]:
        result = sorted(session_root.joinpath(feed).glob("*.csv*"))
        if not result:
            raise AlgoseekDownloadError(f"no completed {feed} pages for {session_date}")
        return result
    return paths(DEPTH), paths(ES_TAQ), paths(MES_TAQ)


def _validate_complete_partition_set(root: Path, session_date: date) -> None:
    manifest_path = _manifest_path(root, session_date)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlgoseekDownloadError(f"cannot validate partition manifest: {manifest_path}") from exc
    expected = {(feed, start.date().isoformat(), _iso_local(start), _iso_local(end))
                for feed in FEEDS for provider_date, start, end in hourly_session_partitions(logical_session_window(session_date))
                if provider_date.isoformat() == start.date().isoformat()}
    recorded = {(item.get("feed"), item.get("provider_trade_date"), item.get("partition_start"), item.get("partition_end"))
                for item in manifest.get("partitions", [])}
    if recorded != expected:
        missing = sorted(expected - recorded)
        extra = sorted(recorded - expected)
        raise AlgoseekDownloadError(f"partition completeness mismatch; missing={missing[:3]} extra={extra[:3]}")
    for feed in FEEDS:
        for provider_date, start, end in hourly_session_partitions(logical_session_window(session_date)):
            _partition_summary(manifest, feed, provider_date, start, end)


def _run_session_audit(root: Path, session_date: date) -> dict[str, Any]:
    from .algoseek_adapter import AlgoseekAdapterError, audit_inputs, streaming_dry_build
    _validate_complete_partition_set(root, session_date)
    depth, es, mes = _input_paths(root, session_date)
    try:
        audit = audit_inputs(es_depth_paths=depth, es_taq_paths=es, mes_taq_paths=mes)
        dry_build = streaming_dry_build(es_depth_paths=depth, es_taq_paths=es, mes_taq_paths=mes)
    except (AlgoseekAdapterError, OSError, csv.Error) as exc:
        raise AlgoseekDownloadError(f"streaming audit failed: {exc}") from exc
    required = (audit["es_depth_rows"] > 0, audit["es_taq_rows"] > 0, audit["mes_taq_rows"] > 0,
                audit["canonical_es_depth_states"] > 0, audit["profile"].get("POC") is not None,
                dry_build["canonical_events_emitted"] > 0, dry_build["mes_bbo_coverage"] > 0.99)
    if not all(required):
        raise AlgoseekDownloadError("streaming audit did not meet complete-session acceptance conditions")
    return {"input_audit": audit, "streaming_dry_build": dry_build}


def download_session(
    *, client: AlgoseekAPIClient, logical_session_date: date, output_root: Path, page_limit: int = CSV_GZIP_MAX_LIMIT,
    resume: bool = False, audit: bool = True,
) -> dict[str, Any]:
    """Download one logical Globex session and publish completion only after audit."""
    if is_cme_closed_session(logical_session_date):
        raise AlgoseekDownloadError(f"{logical_session_date} is a CME closed session; no zero-row download is accepted")
    if not 1 <= page_limit <= CSV_GZIP_MAX_LIMIT:
        raise AlgoseekDownloadError(f"csv_gzip page limit must be in [1, {CSV_GZIP_MAX_LIMIT}]")
    contracts, session = contract_map_for(logical_session_date), logical_session_window(logical_session_date)
    manifest_path = _manifest_path(output_root, logical_session_date)
    session_root = manifest_path.parent
    if session_root.exists() and not manifest_path.exists():
        raise AlgoseekDownloadError(f"refusing to write into existing non-API session directory: {session_root}; use a separate output root")
    manifest = _load_resume_manifest(manifest_path, logical_session_date, contracts, page_limit) if resume and manifest_path.exists() else _new_manifest(logical_session_date, contracts, output_root, page_limit)
    if manifest_path.exists() and not resume:
        raise AlgoseekDownloadError(f"API manifest already exists: {manifest_path}; use --resume after hash verification")
    if resume and manifest.get("session_complete"):
        return {"manifest_path": str(manifest_path), "download_pages": len(manifest["pages"]), "audit": manifest.get("audit"),
                "complete_manifest_path": str(_complete_manifest_path(output_root, logical_session_date)), "session_complete": True,
                "contracts": {"ES": contracts.es, "MES": contracts.mes}}
    # Quotas are account-wide. Record the preflight snapshot for this attempt,
    # but never credentials, before any potentially expensive data request.
    manifest["quota_preflight"] = client.get_json(ACCOUNT_QUOTAS_ENDPOINT)
    _atomic_json(manifest_path, manifest)
    for feed in FEEDS:
        _download_feed(client=client, manifest=manifest, manifest_path=manifest_path, root=output_root, feed=feed,
                       contracts=contracts, session=session, page_limit=page_limit)
    result: dict[str, Any] = {"manifest_path": str(manifest_path), "download_pages": len(manifest["pages"]), "audit": None}
    if audit:
        result["audit"] = _run_session_audit(output_root, logical_session_date)
        manifest["audit"], manifest["session_complete"] = result["audit"], True
        _atomic_json(manifest_path, manifest)
        complete_path = _complete_manifest_path(output_root, logical_session_date)
        _atomic_json(complete_path, {"schema_version": MANIFEST_VERSION, "status": "ALGOSEEK_COMPLETE_SESSION",
            "session_complete": True, "logical_session_date": logical_session_date.isoformat(), "contracts": manifest["contracts"],
            "download_manifest": manifest_path.name, "page_count": len(manifest["pages"]), "audit": result["audit"]})
        result["complete_manifest_path"] = str(complete_path)
    return result | {"session_complete": bool(manifest.get("session_complete")), "contracts": {"ES": contracts.es, "MES": contracts.mes}}


def range_plan(start_date: date, end_date: date, output_root: Path) -> dict[str, Any]:
    if end_date < start_date:
        raise AlgoseekDownloadError("end date precedes start date")
    active, skipped = [], []
    current = start_date
    while current <= end_date:
        (skipped if is_cme_closed_session(current) else active).append(current.isoformat())
        current += timedelta(days=1)
    return {"status": "ALGOSEEK_API_DOWNLOAD_DRY_RUN", "sessions": [
        {"logical_session_date": value, "contracts": contract_map_for(date.fromisoformat(value)).__dict__,
         "provider_trade_dates": [item.isoformat() for item in logical_session_window(date.fromisoformat(value)).provider_trade_dates],
         "feeds": list(FEEDS), "output_directory": str(output_root / value), "estimated_requests": "unknown until pagination"}
        for value in active], "skipped_closed_dates": skipped, "page_limit": CSV_GZIP_MAX_LIMIT}


def download_range(
    *, client: AlgoseekAPIClient, start_date: date, end_date: date, output_root: Path, page_limit: int = CSV_GZIP_MAX_LIMIT,
    resume: bool = False, continue_on_audit_failure: bool = False,
) -> dict[str, Any]:
    plan = range_plan(start_date, end_date, output_root)
    completed, failures = [], []
    for entry in plan["sessions"]:
        session_date = date.fromisoformat(entry["logical_session_date"])
        try:
            completed.append(download_session(client=client, logical_session_date=session_date, output_root=output_root,
                                              page_limit=page_limit, resume=resume))
        except AlgoseekAPIError as exc:
            failures.append({"logical_session_date": session_date.isoformat(), "error": str(exc)})
            if not continue_on_audit_failure:
                break
    return {"status": "ALGOSEEK_API_RANGE_COMPLETE" if not failures else "ALGOSEEK_API_RANGE_FAILED",
            "completed": completed, "failures": failures, "skipped_closed_dates": plan["skipped_closed_dates"]}


def tiny_probe(client: AlgoseekAPIClient, *, logical_reference_date: date = date(2023, 1, 3), limit: int = 5) -> dict[str, Any]:
    if not 1 <= limit <= 10:
        raise AlgoseekDownloadError("tiny probe limit must be 1 through 10")
    contracts = contract_map_for(logical_reference_date)
    session = logical_session_window(logical_reference_date)
    results = []
    with tempfile.TemporaryDirectory(prefix="algoseek-api-probe-") as temporary:
        for feed in FEEDS:
            endpoint_template, ticker, _, prefix = _feed_spec(feed, contracts)
            dataset_id = "US6002" if feed == DEPTH else "US6011"
            endpoint = endpoint_template.format(trade_date=logical_reference_date.isoformat(), ticker=ticker)
            query = {"limit": limit, "offset": 0, "response_format": CSV_GZIP_FORMAT, "sort": "+EventDateTime"}
            url = client._url(endpoint, query)
            try:
                response = client.open(endpoint, query)
            except AlgoseekAPIError as exc:
                raise AlgoseekDownloadError(
                    f"tiny probe failed feed={feed} dataset_id={dataset_id} method=GET url={url}; {exc}"
                ) from exc
            next_offset = _header(response.headers, "X-Pagination-Next-Offset")
            destination = Path(temporary) / f"{prefix}.csv.gz"
            header, returned, retained, _, _, first_returned, last_returned, _, _ = _validate_and_write_page(response=response, destination=destination,
                expected_ticker=ticker, expected_trade_date=logical_reference_date,
                start=datetime.combine(logical_reference_date, datetime.min.time(), SOURCE_TIMEZONE),
                end=datetime.combine(logical_reference_date + timedelta(days=1), datetime.min.time(), SOURCE_TIMEZONE), expected_header=None)
            results.append({"feed": feed, "dataset_id": dataset_id, "ticker": ticker, "request_url": url, "csv_header": header, "returned_row_count": returned,
                            "validated_row_count": retained, "first_returned_timestamp": first_returned, "last_returned_timestamp": last_returned,
                            "response_compression": "gzip", "pagination_next_offset": next_offset})
    return {"status": "ALGOSEEK_API_TINY_PROBE_READY", "ready": True, "logical_reference_date": logical_reference_date.isoformat(), "results": results}


def _comparison_timestamp_utc_nanoseconds(value: str) -> int:
    text = value.strip().replace("T", " ")
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:?\d{2})?", text)
    if not match:
        raise AlgoseekDownloadError(f"invalid EventDateTime for comparison: {value!r}")
    base_naive = datetime.fromisoformat(match.group(1))
    suffix = match.group(3)
    if suffix == "Z":
        base = base_naive.replace(tzinfo=ZoneInfo("UTC"))
    elif suffix:
        offset = suffix[:3] + ":" + suffix[-2:] if len(suffix) == 5 and suffix[3] != ":" else suffix
        base = base_naive.replace(tzinfo=datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").tzinfo)
    else:
        base = base_naive.replace(tzinfo=SOURCE_TIMEZONE)
    base = base.astimezone(ZoneInfo("UTC")).replace(microsecond=0)
    epoch = datetime(1970, 1, 1, tzinfo=ZoneInfo("UTC"))
    delta = base - epoch
    seconds = delta.days * 86_400 + delta.seconds
    fraction = (match.group(2) or "").ljust(9, "0")
    return seconds * 1_000_000_000 + int(fraction or 0)


_COMPARISON_INTEGER_FIELDS = {"Quantity", "Orders", "Depth", "Flags", "TypeMask"}


def _canonical_comparison_value(field: str, value: Any) -> Any:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    if field == "EventDateTime":
        return _comparison_timestamp_utc_nanoseconds(text)
    if field == "TradeDate":
        return text
    if field in _COMPARISON_INTEGER_FIELDS:
        try:
            return int(Decimal(text))
        except (InvalidOperation, ValueError):
            return text
    if field == "Price" or re.fullmatch(r"L\d+Price", field):
        try:
            decimal = Decimal(text)
            ticks = decimal * Decimal(4)
            if ticks == ticks.to_integral_value():
                return {"quarter_point_ticks": int(ticks)}
            return {"decimal": format(decimal.normalize(), "f")}
        except (InvalidOperation, ValueError):
            return text
    return text


def canonical_comparison_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: _canonical_comparison_value(field, row.get(field)) for field in sorted(row)}


def normalized_row_multiset_fingerprint(paths: Iterable[Path]) -> dict[str, Any]:
    """Constant-memory normalized multiset signature, independent of page/chunk files."""
    count = total = exclusive = 0
    modulus = 1 << 256
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                canonical = json.dumps(canonical_comparison_row(row), separators=(",", ":"), sort_keys=True).encode("utf-8")
                value = int.from_bytes(hashlib.sha256(canonical).digest(), "big")
                count += 1; total = (total + value) % modulus; exclusive ^= value
    return {"row_count": count, "sha256_sum_mod_2_256": f"{total:064x}", "sha256_xor": f"{exclusive:064x}"}


def compare_session_roots(*, api_root: Path, manual_root: Path, logical_session_date: date) -> dict[str, Any]:
    """Read-only semantic comparison; different API page boundaries are ignored."""
    from .algoseek_adapter import audit_inputs, streaming_dry_build

    def metrics(root: Path) -> dict[str, Any]:
        depth, es, mes = _input_paths(root, logical_session_date)
        audit = audit_inputs(es_depth_paths=depth, es_taq_paths=es, mes_taq_paths=mes)
        dry = streaming_dry_build(es_depth_paths=depth, es_taq_paths=es, mes_taq_paths=mes)
        return {"row_counts": {key: audit[key] for key in ("es_depth_rows", "es_taq_rows", "mes_taq_rows", "es_trade_rows", "canonical_es_depth_states")},
                "profile": audit["profile"], "mes_bbo_coverage": dry["mes_bbo_coverage"], "canonical_event_count": dry["canonical_events_emitted"],
                "identities": audit["provider_provenance"],
                "normalized_raw_row_multiset": {DEPTH: normalized_row_multiset_fingerprint(depth), ES_TAQ: normalized_row_multiset_fingerprint(es), MES_TAQ: normalized_row_multiset_fingerprint(mes)}}
    api, manual = metrics(api_root), metrics(manual_root)
    compared = {"raw_semantic_multiset": api["normalized_raw_row_multiset"] == manual["normalized_raw_row_multiset"],
                "row_counts": api["row_counts"] == manual["row_counts"], "profile": api["profile"] == manual["profile"],
                "canonical_event_count": api["canonical_event_count"] == manual["canonical_event_count"],
                "exact_ordered_stream": False}
    return {"status": "ALGOSEEK_API_REFERENCE_COMPARISON_COMPLETE", "logical_session_date": logical_session_date.isoformat(),
            "api": api, "manual": manual, "matches": compared,
            "ordered_stream_note": "not required: same-timestamp source order is not provider-causal and remains a known limitation",
            "semantic_match": all(value for key, value in compared.items() if key != "exact_ordered_stream")}


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def add_cli_parsers(subparsers: Any) -> None:
    preflight_parser = subparsers.add_parser("algoseek-api-preflight", help="verify Algoseek account access; never downloads market data")
    preflight_parser.add_argument("--base-url", default=None, help=f"API base URL (default: ${API_BASE_URL_ENV} or {API_BASE_URL})")
    probe = subparsers.add_parser("algoseek-api-probe", help="download and validate only 1-10 rows per required Algoseek feed")
    probe.add_argument("--base-url", default=None, help=f"API base URL (default: ${API_BASE_URL_ENV} or {API_BASE_URL})"); probe.add_argument("--logical-reference-date", type=_date_argument, default=date(2023, 1, 3)); probe.add_argument("--limit", type=int, default=5)
    for name, help_text in (("algoseek-download-session", "download one Algoseek logical session and audit it"),
                            ("algoseek-download-range", "download Algoseek sessions one at a time and audit each")):
        parser = subparsers.add_parser(name, help=help_text)
        parser.add_argument("--base-url", default=None, help=f"API base URL (default: ${API_BASE_URL_ENV} or {API_BASE_URL})"); parser.add_argument("--output-root", type=Path, required=True)
        parser.add_argument("--page-limit", type=int, default=CSV_GZIP_MAX_LIMIT); parser.add_argument("--resume", action="store_true")
        if name.endswith("session"):
            parser.add_argument("--logical-session-date", type=_date_argument, required=True)
        else:
            parser.add_argument("--start-date", type=_date_argument, required=True); parser.add_argument("--end-date", type=_date_argument, required=True)
            parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--continue-on-audit-failure", action="store_true")
    compare = subparsers.add_parser("algoseek-compare-session", help="read-only semantic comparison of API and manual session roots")
    compare.add_argument("--api-root", type=Path, required=True); compare.add_argument("--manual-root", type=Path, required=True); compare.add_argument("--logical-session-date", type=_date_argument, required=True)


def dispatch_cli(args: Any) -> dict[str, Any]:
    if args.command == "algoseek-compare-session":
        return compare_session_roots(api_root=args.api_root, manual_root=args.manual_root, logical_session_date=args.logical_session_date)
    if args.command == "algoseek-download-range" and args.dry_run:
        return range_plan(args.start_date, args.end_date, args.output_root)
    client = AlgoseekAPIClient(base_url=args.base_url)
    if args.command == "algoseek-api-preflight":
        return preflight(client)
    if args.command == "algoseek-api-probe":
        return tiny_probe(client, logical_reference_date=args.logical_reference_date, limit=args.limit)
    if args.command == "algoseek-download-session":
        return download_session(client=client, logical_session_date=args.logical_session_date, output_root=args.output_root, page_limit=args.page_limit, resume=args.resume)
    return download_range(client=client, start_date=args.start_date, end_date=args.end_date, output_root=args.output_root,
                          page_limit=args.page_limit, resume=args.resume, continue_on_audit_failure=args.continue_on_audit_failure)
