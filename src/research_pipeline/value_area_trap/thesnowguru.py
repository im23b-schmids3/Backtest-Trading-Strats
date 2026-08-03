"""Read-only audit and tightly-gated importer for TheSnowGuru public data.

The audit never rewrites source files.  It streams CSV and ZIP members, writes
only content-addressed local reports below the ignored staging root, and makes
no claim that an index/CFD series is a CME futures contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator, TextIO

from pydantic import Field

from ..schemas.strategy_spec import StrictModel
from .data import AggregateTrade, AggregateTradeImporter


AUDIT_VERSION = "thesnowguru-read-only-audit-1"
SOURCE_NAME = "TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data"
CLASSIFICATIONS = {
    "VERIFIED_FUTURES_TICK",
    "VERIFIED_FUTURES_BAR",
    "INDEX_OR_CFD_BAR",
    "AMBIGUOUS",
    "UNSUITABLE",
}
CVD_RESULTS = {"EXACT_CVD_SUPPORTED", "CVD_REQUIRES_TICK_RULE_APPROXIMATION", "CVD_NOT_SUPPORTED"}
_CANDIDATE_TERMS = ("s&p", "sp500", "usa500", "nasdaq", "usatech")
_TIMESTAMP_COLUMNS = ("timestamp", "datetime", "local time", "time", "date")


class TheSnowGuruCandidate(StrictModel):
    relative_path: str
    source_path: str
    archive_type: str
    source_sha256: str
    extracted_sha256: str | None = None
    file_size_bytes: int
    instrument_inferred: str
    classification: str
    classification_reason: str
    date_start: str | None = None
    date_end: str | None = None
    row_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    timestamp_format: str | None = None
    timezone: str | None = None
    sampling_frequency: str | None = None
    bar_type: str
    volume_exists: bool = False
    bid_exists: bool = False
    ask_exists: bool = False
    aggressor_side_exists: bool = False
    contract_code_exists: bool = False
    rollover_information_exists: bool = False
    cvd_support: str
    quality: dict[str, Any]
    monthly_coverage: dict[str, int]
    extraction_status: str


class TheSnowGuruAuditResult(StrictModel):
    status: str
    source_repository: str
    source_root: str
    audit_root: str
    audit_hash: str
    source_manifest_path: str
    schema_manifest_path: str
    data_quality_report_path: str
    monthly_coverage_report_path: str
    candidates: list[TheSnowGuruCandidate]
    es_import_available: bool
    es_import_reason: str
    nq_verified_futures_present: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _is_candidate(path: Path) -> bool:
    text = path.as_posix().lower()
    if path.suffix.lower() not in {".csv", ".zip"}:
        return False
    if any(term in text for term in _CANDIDATE_TERMS):
        return True
    # Keep short futures symbols as path tokens.  A substring check would
    # accidentally classify unrelated paths such as ``indices`` as ES data.
    return bool(re.search(r"(?:^|[._/\\-])(es|nq)(?=$|[._/\\-])", text))


def _normal_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _column_map(names: list[str]) -> dict[str, str]:
    return {_normal_column(name): name for name in names}


class _ZipTextHandle:
    """Text stream whose close operation also releases its owning ZIP file."""

    def __init__(self, path: Path, member: str) -> None:
        self._bundle = zipfile.ZipFile(path)
        with self._bundle.open(member) as probe:
            encoding = _detect_encoding(probe.read(64 * 1024))
        self._text = TextIOWrapper(self._bundle.open(member), encoding=encoding, newline="")

    def __iter__(self) -> Iterator[str]:
        return iter(self._text)

    def __next__(self) -> str:
        return next(self._text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._text, name)

    def close(self) -> None:
        try:
            self._text.close()
        finally:
            self._bundle.close()


def _detect_encoding(sample: bytes) -> str:
    """Choose a deterministic, lossless encoding from known local exports."""

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("candidate CSV could not be decoded using UTF-8, Windows-1252, or Latin-1")


def _open_csv(path: Path, member: str | None = None) -> tuple[TextIO | _ZipTextHandle, str]:
    if member is None:
        with path.open("rb") as probe:
            encoding = _detect_encoding(probe.read(64 * 1024))
        return path.open("r", encoding=encoding, newline=""), "CSV"
    return _ZipTextHandle(path, member), "ZIP_CSV_MEMBER"


def _timestamp_from_row(row: dict[str, str], columns: dict[str, str]) -> tuple[datetime, str] | None:
    raw: str | None = None
    format_name = ""
    if "local_time" in columns:
        raw = row.get(columns["local_time"])
        if raw:
            value = raw.replace(" GMT", " ")
            for fmt in ("%d.%m.%Y %H:%M:%S.%f %z", "%d.%m.%Y %H:%M:%S %z"):
                try:
                    return datetime.strptime(value, fmt), "DD.MM.YYYY HH:MM:SS[.fff] GMT±HHMM"
                except ValueError:
                    pass
    if "date" in columns and "time" in columns:
        raw = f"{row.get(columns['date'], '')} {row.get(columns['time'], '')}".strip()
        # The documented S&P archive uses a fixed DD/MM/YYYY plus
        # HH:MM:SS[.fff] layout.  Parse it directly so a full streaming audit
        # stays practical for multi-hundred-megabyte members.
        date_value = row.get(columns["date"], "")
        time_value = row.get(columns["time"], "")
        if len(date_value) == 10 and len(time_value) >= 8 and date_value[2] == date_value[5] == "/" and time_value[2] == time_value[5] == ":":
            try:
                microsecond = int((time_value[9:].split(".", 1)[0] + "000000")[:6]) if len(time_value) > 8 and time_value[8] == "." else 0
                # The TheSnowGuru SP archive is US-formatted MM/DD/YYYY.
                return datetime(int(date_value[6:10]), int(date_value[0:2]), int(date_value[3:5]), int(time_value[0:2]), int(time_value[3:5]), int(time_value[6:8]), microsecond), "MM/DD/YYYY HH:MM:SS[.fff]"
            except ValueError:
                pass
        for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt), fmt
            except ValueError:
                pass
    for name in ("timestamp", "datetime", "time", "date"):
        if name not in columns:
            continue
        raw = row.get(columns[name])
        if not raw:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f"):
            try:
                return datetime.strptime(raw, fmt), fmt
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")), "ISO-8601"
        except ValueError:
            pass
    return None


def _decimal(row: dict[str, str], column: str | None) -> Decimal | None:
    if column is None:
        return None
    value = row.get(column)
    if value is None or not value.strip():
        return None
    try:
        return Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None


def _audit_number(row: dict[str, str], column: str) -> float | None:
    """Fast numeric validity check for the read-only audit only.

    Normalization remains Decimal-based in :func:`import_thesnowguru_es`; the
    audit only needs to identify absent, malformed, negative, or zero values.
    """

    value = row.get(column)
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _infer_instrument(relative: str, readme: str) -> tuple[str, bool, bool]:
    value = relative.lower()
    sp = any(token in value for token in ("s&p", "sp500", "usa500"))
    nasdaq = any(token in value for token in ("nasdaq", "usatech"))
    index = any(token in value for token in ("idxusd", "usa500", "usatech", "indices/s&p500", "indices/nasdaq100"))
    # The README's futures statement is evidence only for its separate
    # ``s&p-tick.zip`` archive.  It does not convert neighboring IDXUSD quote
    # files in the misspelled ``indicies`` folder into ES futures data.
    documented_sp_futures = "s&p-tick.zip" in value and "only s&p futures tick data" in readme.lower()
    if documented_sp_futures:
        return "S&P futures tick data (README-documented; contract code unspecified)", True, False
    if sp:
        return "S&P 500 index/CFD proxy", False, True
    if nasdaq:
        return "Nasdaq index/CFD proxy", False, True
    return "Unidentified candidate", False, index


def _classify(*, relative: str, readme: str, columns: list[str], bar_type: str) -> tuple[str, str, str]:
    instrument, documented_futures, index = _infer_instrument(relative, readme)
    normalized = {_normal_column(item) for item in columns}
    has_timestamp = bool(normalized.intersection({"timestamp", "datetime", "local_time", "time", "date"}))
    if not has_timestamp:
        return "UNSUITABLE", "no parseable timestamp column", instrument
    if documented_futures:
        return (
            "VERIFIED_FUTURES_TICK" if bar_type == "TICK" else "VERIFIED_FUTURES_BAR",
            "repository README explicitly documents only the S&P tick archive as futures data; exact contract/rollover is absent",
            instrument,
        )
    if index:
        return "INDEX_OR_CFD_BAR", "path/instrument labels identify an index or IDXUSD proxy, not a verified futures contract", instrument
    return "AMBIGUOUS", "candidate path does not establish exchange-traded futures provenance", instrument


def _bar_type(relative: str, columns: list[str]) -> str:
    value = relative.lower()
    normalized = {_normal_column(item) for item in columns}
    if "tick" in value or "ticks" in value:
        return "TICK"
    for suffix, label in (("_m30", "30-minute"), ("_m15", "15-minute"), ("_m5", "5-minute"), ("_m1", "1-minute"), ("_h4", "4-hour"), ("_h1", "1-hour"), ("_d1", "daily")):
        if suffix in value:
            return label
    if {"open", "high", "low", "close"}.issubset(normalized):
        return "BAR_UNSPECIFIED"
    return "OTHER"


def _sampling(relative: str, deltas: Counter[int]) -> str:
    value = relative.lower()
    for suffix, label in (("_m30", "30-minute"), ("_m15", "15-minute"), ("_m5", "5-minute"), ("_m1", "1-minute"), ("_h4", "4-hour"), ("_h1", "1-hour"), ("_d1", "daily")):
        if suffix in value:
            return label
    if "tick" in value:
        return "irregular tick"
    if deltas:
        seconds, _ = deltas.most_common(1)[0]
        return f"approximately {seconds} seconds"
    return "unknown"


def _audit_csv(
    *, source_path: Path, member: str | None, relative: str, source_hash: str, file_size: int, readme: str
) -> TheSnowGuruCandidate:
    handle, archive_type = _open_csv(source_path, member)
    try:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        columns = reader.fieldnames or []
        normalized = _column_map(columns)
        bar_type = _bar_type(relative, columns)
        classification, reason, instrument = _classify(relative=relative, readme=readme, columns=columns, bar_type=bar_type)
        price_column = next((normalized[key] for key in ("price", "trade_price", "last_price") if key in normalized), None)
        size_column = next((normalized[key] for key in ("size", "trade_size", "quantity", "volume") if key in normalized), None)
        quality_price_columns = [
            normalized[key]
            for key in ("price", "trade_price", "last_price", "bid", "bid_price", "ask", "ask_price", "open", "high", "low", "close")
            if key in normalized
        ]
        quality_size_columns = [
            normalized[key]
            for key in ("size", "trade_size", "quantity", "volume", "bid_volume", "ask_volume", "bidvolume", "askvolume")
            if key in normalized
        ]
        side_column = next((normalized[key] for key in ("aggressor_side", "side", "trade_side", "buyer_is_maker") if key in normalized), None)
        bid_column = next((normalized[key] for key in ("bid", "bid_price") if key in normalized), None)
        ask_column = next((normalized[key] for key in ("ask", "ask_price") if key in normalized), None)
        contract = any(key in normalized for key in ("contract", "contract_code", "symbol", "instrument"))
        rollover = any("roll" in key or "expiry" in key or "expiration" in key for key in normalized)
        row_count = parse_failures = null_values = duplicate_timestamps = non_monotonic = impossible_prices = impossible_sizes = zero_sizes = repeated_rows = weekend_activity = gap_count = positive_sizes = 0
        previous_timestamp: datetime | None = None
        previous_row: tuple[str, ...] | None = None
        start: datetime | None = None
        end: datetime | None = None
        time_formats: Counter[str] = Counter(); timezone_values: Counter[str] = Counter(); deltas: Counter[int] = Counter(); coverage: Counter[str] = Counter()
        expected_seconds = {"1-minute": 60, "5-minute": 300, "15-minute": 900, "30-minute": 1800, "1-hour": 3600, "4-hour": 14400, "daily": 86400}.get(bar_type, 300 if bar_type == "TICK" else None)
        for row in reader:
            row_count += 1
            values = tuple((row.get(name) or "") for name in columns)
            if previous_row == values:
                repeated_rows += 1
            previous_row = values
            null_values += sum(value is None or not value.strip() for value in row.values())
            parsed = _timestamp_from_row(row, normalized)
            if parsed is None:
                parse_failures += 1
                continue
            timestamp, fmt = parsed
            time_formats[fmt] += 1
            if timestamp.tzinfo is not None:
                timezone_values[timestamp.strftime("%z")] += 1
            else:
                timezone_values["GMT_DECLARED_BY_REPOSITORY_README"] += 1
            if timestamp.weekday() >= 5:
                weekend_activity += 1
            start = timestamp if start is None or timestamp < start else start
            end = timestamp if end is None or timestamp > end else end
            coverage[timestamp.strftime("%Y-%m")] += 1
            if previous_timestamp is not None:
                delta = (timestamp - previous_timestamp).total_seconds()
                if delta == 0:
                    duplicate_timestamps += 1
                elif delta < 0:
                    non_monotonic += 1
                else:
                    whole = int(delta)
                    deltas[whole] += 1
                    if expected_seconds is not None and delta > expected_seconds * 2:
                        gap_count += 1
            previous_timestamp = timestamp
            parsed_numbers = {column: _audit_number(row, column) for column in dict.fromkeys(quality_price_columns + quality_size_columns)}
            price = parsed_numbers.get(price_column) if price_column else None
            size = parsed_numbers.get(size_column) if size_column else None
            prices = [parsed_numbers[column] for column in quality_price_columns]
            sizes = [parsed_numbers[column] for column in quality_size_columns]
            if any(value is None for value in prices):
                parse_failures += 1
            elif any(value is not None and value <= 0 for value in prices):
                impossible_prices += 1
            if any(value is None for value in sizes):
                parse_failures += 1
            elif any(value is not None and value < 0 for value in sizes):
                impossible_sizes += 1
            if size is not None:
                if size == 0:
                    zero_sizes += 1
                elif size > 0:
                    positive_sizes += 1
        has_trade_price = price_column is not None and bar_type == "TICK"
        if has_trade_price and positive_sizes and (side_column is not None or (bid_column is not None and ask_column is not None)):
            cvd = "EXACT_CVD_SUPPORTED"
        elif has_trade_price and positive_sizes:
            cvd = "CVD_REQUIRES_TICK_RULE_APPROXIMATION"
        else:
            cvd = "CVD_NOT_SUPPORTED"
        return TheSnowGuruCandidate(
            relative_path=relative,
            source_path=str(source_path.resolve()),
            archive_type=archive_type,
            source_sha256=source_hash,
            file_size_bytes=file_size,
            instrument_inferred=instrument,
            classification=classification,
            classification_reason=reason,
            date_start=start.isoformat() if start else None,
            date_end=end.isoformat() if end else None,
            row_count=row_count,
            column_names=columns,
            timestamp_format=time_formats.most_common(1)[0][0] if time_formats else None,
            timezone=timezone_values.most_common(1)[0][0] if timezone_values else None,
            sampling_frequency=_sampling(relative, deltas),
            bar_type=bar_type,
            volume_exists=size_column is not None,
            bid_exists=bid_column is not None,
            ask_exists=ask_column is not None,
            aggressor_side_exists=side_column is not None,
            contract_code_exists=contract,
            rollover_information_exists=rollover,
            cvd_support=cvd,
            quality={
                "parse_failures": parse_failures,
                "duplicate_timestamps": duplicate_timestamps,
                "non_monotonic_timestamps": non_monotonic,
                "null_values": null_values,
                "impossible_prices": impossible_prices,
                "impossible_sizes": impossible_sizes,
                "zero_sizes": zero_sizes,
                "timestamp_gaps_over_threshold": gap_count,
                "weekend_activity_rows": weekend_activity,
                "repeated_adjacent_rows": repeated_rows,
                "quality_method": "streaming full-file scan; repeated rows means adjacent exact duplicates",
            },
            monthly_coverage=dict(sorted(coverage.items())),
            extraction_status="NOT_EXTRACTED_STREAMED_FROM_ARCHIVE" if member is not None else "NOT_APPLICABLE",
        )
    finally:
        handle.close()


def _write_once(path: Path, payload: Any) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"immutable TheSnowGuru audit artifact differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def audit_thesnowguru_data(
    *, repository_root: str | Path = ".", source_root: str | Path = "external_data/thesnowguru", staging_root: str | Path = "data/value_area_trap/staging"
) -> TheSnowGuruAuditResult:
    root = Path(repository_root).resolve()
    source = Path(source_root)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"TheSnowGuru source repository is missing: {source}")
    readme_path = source / "README.md"
    readme = readme_path.read_text(encoding="utf-8", errors="replace") if readme_path.is_file() else ""
    candidates: list[TheSnowGuruCandidate] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file() and _is_candidate(item)):
        relative = path.relative_to(source).as_posix()
        source_hash = _sha256_file(path)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as bundle:
                for info in sorted((item for item in bundle.infolist() if not item.is_dir() and item.filename.lower().endswith(".csv")), key=lambda item: item.filename):
                    candidates.append(_audit_csv(source_path=path, member=info.filename, relative=f"{relative}::{info.filename}", source_hash=source_hash, file_size=info.file_size, readme=readme))
        else:
            candidates.append(_audit_csv(source_path=path, member=None, relative=relative, source_hash=source_hash, file_size=path.stat().st_size, readme=readme))
    canonical = [{**item.model_dump(mode="json"), "source_path": item.relative_path} for item in candidates]
    audit_hash = _canonical_hash({"version": AUDIT_VERSION, "source_readme_hash": _sha256_file(readme_path) if readme_path.is_file() else None, "candidates": canonical})
    staging = Path(staging_root)
    if not staging.is_absolute():
        staging = root / staging
    audit_root = staging.resolve() / "thesnowguru" / audit_hash
    source_manifest = {"audit_version": AUDIT_VERSION, "source_repository": SOURCE_NAME, "source_root": str(source), "audit_hash": audit_hash, "candidates": [{key: value for key, value in item.model_dump(mode="json").items() if key not in {"quality", "monthly_coverage"}} for item in candidates]}
    schema_manifest = {"audit_hash": audit_hash, "schemas": [{"relative_path": item.relative_path, "columns": item.column_names, "timestamp_format": item.timestamp_format, "timezone": item.timezone, "sampling_frequency": item.sampling_frequency, "bar_type": item.bar_type, "classification": item.classification, "cvd_support": item.cvd_support} for item in candidates]}
    quality_report = {"audit_hash": audit_hash, "quality": [{"relative_path": item.relative_path, "classification": item.classification, "quality": item.quality} for item in candidates]}
    coverage_report = {"audit_hash": audit_hash, "coverage": [{"relative_path": item.relative_path, "monthly_coverage": item.monthly_coverage} for item in candidates]}
    _write_once(audit_root / "source-manifest.json", source_manifest)
    _write_once(audit_root / "schema-manifest.json", schema_manifest)
    _write_once(audit_root / "data-quality-report.json", quality_report)
    _write_once(audit_root / "monthly-coverage-report.json", coverage_report)
    eligible = [item for item in candidates if item.classification == "VERIFIED_FUTURES_TICK" and item.cvd_support == "EXACT_CVD_SUPPORTED"]
    reason = "verified S&P futures tick data with timestamp, trade price, positive trade size, and aggressor/bid-ask evidence is available" if eligible else "no verified S&P futures tick candidate provides exact CVD-required trade price, positive size, and aggressor/bid-ask evidence"
    audit_payload = {"status": "AUDIT_COMPLETE", "audit_hash": audit_hash, "source_repository": SOURCE_NAME, "source_root": str(source), "audit_root": str(audit_root), "source_manifest_path": str(audit_root / "source-manifest.json"), "schema_manifest_path": str(audit_root / "schema-manifest.json"), "data_quality_report_path": str(audit_root / "data-quality-report.json"), "monthly_coverage_report_path": str(audit_root / "monthly-coverage-report.json"), "candidates": [item.model_dump(mode="json") for item in candidates], "es_import_available": bool(eligible), "es_import_reason": reason, "nq_verified_futures_present": any(item.classification.startswith("VERIFIED_FUTURES") and "Nasdaq" in item.instrument_inferred for item in candidates)}
    _write_once(audit_root / "audit.json", audit_payload)
    return TheSnowGuruAuditResult.model_validate(audit_payload)


def _stage_zip_member(candidate: TheSnowGuruCandidate, staging_root: Path) -> Path:
    archive = Path(candidate.source_path)
    member = candidate.relative_path.split("::", 1)[1]
    if _sha256_file(archive) != candidate.source_sha256:
        raise ValueError("TheSnowGuru source archive hash changed after audit; rerun the audit before import")
    target = staging_root / "thesnowguru-extracted" / candidate.source_sha256 / Path(member).name
    target.parent.mkdir(parents=True, exist_ok=True)
    # A staged member is a derived, content-addressed audit artifact.  Never
    # overwrite it during a repeat import attempt; source archives remain
    # untouched in every case.
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise ValueError(f"audited ZIP member is missing from source archive: {member}")
        source_member_hash = hashlib.sha256()
        with bundle.open(member) as source:
            while chunk := source.read(1024 * 1024):
                source_member_hash.update(chunk)
        if not target.exists():
            with bundle.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if _sha256_file(target) != source_member_hash.hexdigest():
        raise ValueError("staged archive member hash differs from audited source archive member")
    if not target.is_file():  # pragma: no cover - defensive
        raise ValueError("staged archive member is unavailable")
    return target


def import_thesnowguru_es(*, audit_path: str | Path, cache_root: str | Path = "data/value_area_trap") -> dict[str, Any]:
    """Import only a previously audited exact-CVD S&P futures tick candidate."""

    audit_file = Path(audit_path).resolve()
    payload = json.loads(audit_file.read_text(encoding="utf-8"))
    audit = TheSnowGuruAuditResult.model_validate(payload)
    eligible = [item for item in audit.candidates if item.classification == "VERIFIED_FUTURES_TICK" and item.cvd_support == "EXACT_CVD_SUPPORTED"]
    if not eligible:
        raise ValueError("TheSnowGuru ES import is unavailable: audit did not establish exact CVD-supported verified futures tick data")
    candidate = eligible[0]
    source_path = Path(candidate.source_path)
    cache = Path(cache_root).resolve()
    if "::" in candidate.relative_path:
        staged = _stage_zip_member(candidate, cache / "staging")
        source_for_rows, member = staged, None
    else:
        source_for_rows, member = source_path, None
    handle, _ = _open_csv(source_for_rows, member)
    try:
        reader = csv.DictReader(handle)
        columns = _column_map(reader.fieldnames or [])
        price_column = next(columns[key] for key in ("price", "trade_price", "last_price") if key in columns)
        size_column = next(columns[key] for key in ("size", "trade_size", "quantity", "volume") if key in columns)
        side_column = next((columns[key] for key in ("aggressor_side", "side", "trade_side", "buyer_is_maker") if key in columns), None)
        if side_column is None:
            raise ValueError("TheSnowGuru ES import requires audited aggressor-side evidence")

        def records() -> Iterator[AggregateTrade]:
            for index, row in enumerate(reader, start=1):
                parsed = _timestamp_from_row(row, columns)
                if parsed is None:
                    raise ValueError(f"audited timestamp parse failure at row {index}")
                timestamp, _ = parsed
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                price, size = _decimal(row, price_column), _decimal(row, size_column)
                if price is None or size is None or price <= 0 or size <= 0:
                    raise ValueError(f"audited invalid price/size at row {index}")
                side = (row.get(side_column) or "").strip().upper()
                if side in {"BUY", "B", "1", "TRUE"}:
                    maker = False
                elif side in {"SELL", "S", "0", "FALSE"}:
                    maker = True
                else:
                    raise ValueError(f"audited unrecognized aggressor side at row {index}")
                yield AggregateTrade(event_time_utc=timestamp.astimezone(timezone.utc), trade_time_utc=timestamp.astimezone(timezone.utc), aggregate_trade_id=index, first_trade_id=None, last_trade_id=None, price=price, quantity_base=size, notional_quote=price * size, buyer_is_maker=maker, aggressor_side="SELL" if maker else "BUY", signed_quantity=-size if maker else size, source="thesnowguru_s_and_p_futures_tick", source_file=str(source_for_rows), source_hash=candidate.source_sha256)

        importer = AggregateTradeImporter(cache)
        parquet, manifest = importer.ingest_records(records(), symbol="ES_THESNOWGURU", source_files=[str(source_for_rows)])
    finally:
        handle.close()
    output_root = parquet.parent
    provenance = {"audit_path": str(audit_file), "audit_hash": audit.audit_hash, "candidate": candidate.model_dump(mode="json"), "staged_source": str(source_for_rows), "staged_source_sha256": _sha256_file(source_for_rows), "normalized_dataset_hash": manifest.normalized_dataset_hash, "parquet": str(parquet)}
    _write_once(output_root / "thesnowguru-source-provenance.json", provenance)
    return {"status": "IMPORTED", "parquet": str(parquet), "manifest_path": str(output_root / "manifest.json"), "dataset_hash": manifest.normalized_dataset_hash, "provenance_path": str(output_root / "thesnowguru-source-provenance.json")}
