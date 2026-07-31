from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, field_validator

from ..schemas.strategy_spec import StrictModel


IMPORTER_VERSION = "value-area-trap-aggregate-trades-1"
SCHEMA_VERSION = "binance-usdm-aggregate-trade-1"
BINANCE_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
ARCHIVE_COLUMN_ORDER = (
    "aggregate_trade_id",
    "price",
    "quantity_base",
    "first_trade_id",
    "last_trade_id",
    "trade_time",
    "buyer_is_maker",
    "is_best_match",
)
FIELD_ALIASES = {
    "aggregate_trade_id": "aggregate_trade_id",
    "agg_trade_id": "aggregate_trade_id",
    "a": "aggregate_trade_id",
    "price": "price",
    "p": "price",
    "quantity_base": "quantity_base",
    "quantity": "quantity_base",
    "q": "quantity_base",
    "first_trade_id": "first_trade_id",
    "f": "first_trade_id",
    "last_trade_id": "last_trade_id",
    "l": "last_trade_id",
    "trade_time": "trade_time",
    "trade_time_utc": "trade_time",
    "transact_time": "trade_time",
    "T": "trade_time",
    "event_time": "event_time",
    "event_time_utc": "event_time",
    "E": "event_time",
    "buyer_is_maker": "buyer_is_maker",
    "is_buyer_maker": "buyer_is_maker",
    "m": "buyer_is_maker",
    "is_best_match": "is_best_match",
}
REQUIRED_NORMALIZED_FIELDS = {
    "aggregate_trade_id",
    "price",
    "quantity_base",
    "trade_time",
    "buyer_is_maker",
}
PARQUET_SCHEMA = pa.schema(
    [
        ("event_time_utc", pa.string()),
        ("trade_time_utc", pa.string()),
        ("aggregate_trade_id", pa.int64()),
        ("first_trade_id", pa.int64()),
        ("last_trade_id", pa.int64()),
        ("price", pa.string()),
        ("quantity_base", pa.string()),
        ("notional_quote", pa.string()),
        ("buyer_is_maker", pa.bool_()),
        ("aggressor_side", pa.string()),
        ("signed_quantity", pa.string()),
        ("source", pa.string()),
        ("source_file", pa.string()),
        ("source_hash", pa.string()),
    ]
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_millis(value: Any) -> datetime:
    raw = int(str(value))
    # Binance archive timestamps are milliseconds; accept microseconds for
    # fixture compatibility without guessing a timezone.
    seconds = raw / (1_000_000 if raw > 10_000_000_000_000 else 1_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


class AggregateTrade(StrictModel):
    event_time_utc: datetime
    trade_time_utc: datetime
    aggregate_trade_id: int = Field(ge=0)
    first_trade_id: int | None = Field(default=None, ge=0)
    last_trade_id: int | None = Field(default=None, ge=0)
    price: Decimal = Field(gt=0)
    quantity_base: Decimal = Field(gt=0)
    notional_quote: Decimal = Field(gt=0)
    buyer_is_maker: bool
    aggressor_side: str
    signed_quantity: Decimal
    source: str = "binance_usdm_public_archive"
    source_file: str
    source_hash: str

    @field_validator("event_time_utc", "trade_time_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("aggregate-trade timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)


class AggregateTradeManifest(StrictModel):
    provider: str = "Binance USD-M Futures public data archive"
    product: str = "USD-M perpetual aggregate trades"
    symbol: str = "BTCUSDT"
    date_start: date
    date_end: date
    retrieved_at: datetime
    source_files: list[str]
    source_file_hashes: dict[str, str]
    normalized_dataset_hash: str
    row_count: int
    duplicate_count: int
    missing_interval_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    importer_version: str = IMPORTER_VERSION
    manifest_hash: str = "pending"


def _manifest_hash(manifest: AggregateTradeManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_hash", None)
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _normalize_columns(row: dict[str, Any], *, source_file: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    detected = [str(name).strip() for name in row]
    for name, value in row.items():
        clean_name = str(name).strip().lstrip("\ufeff")
        canonical = FIELD_ALIASES.get(clean_name)
        if canonical is not None:
            normalized[canonical] = value
    missing = sorted(
        field
        for field in REQUIRED_NORMALIZED_FIELDS
        if normalized.get(field) in {None, ""}
    )
    if missing:
        raise ValueError(
            f"invalid aggregate-trade schema in ZIP member {source_file!r}: "
            f"detected columns={detected}; missing normalized fields={missing}"
        )
    return normalized


def _parse_boolean(value: Any, *, field: str, source_file: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(
        f"invalid boolean for {field!r} in ZIP member {source_file!r}: "
        f"expected true/false or 1/0, got {value!r}"
    )


def parse_binance_aggregate_trade(row: dict[str, Any], *, source_file: str, source_hash: str, source: str = "binance_usdm_public_archive") -> AggregateTrade:
    """Normalize and parse documented archive/REST aggregate-trade fields."""
    normalized = _normalize_columns(row, source_file=source_file)
    price = Decimal(str(normalized["price"]))
    quantity = Decimal(str(normalized["quantity_base"]))
    maker = _parse_boolean(
        normalized["buyer_is_maker"],
        field="buyer_is_maker",
        source_file=source_file,
    )
    trade_time = normalized["trade_time"]
    return AggregateTrade(
        event_time_utc=_utc_millis(normalized.get("event_time", trade_time)),
        trade_time_utc=_utc_millis(trade_time),
        aggregate_trade_id=int(normalized["aggregate_trade_id"]),
        first_trade_id=int(normalized["first_trade_id"]) if normalized.get("first_trade_id") not in {None, ""} else None,
        last_trade_id=int(normalized["last_trade_id"]) if normalized.get("last_trade_id") not in {None, ""} else None,
        price=price,
        quantity_base=quantity,
        notional_quote=price * quantity,
        buyer_is_maker=maker,
        aggressor_side="SELL" if maker else "BUY",
        signed_quantity=-quantity if maker else quantity,
        source=source,
        source_file=source_file,
        source_hash=source_hash,
    )


class AggregateTradeImporter:
    """Public, content-addressed importer for Binance USD-M aggregate trades."""

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root).resolve()

    @staticmethod
    def archive_url(symbol: str, month: str) -> str:
        symbol = symbol.upper()
        if len(month) != 7 or month[4] != "-":
            raise ValueError("month must use YYYY-MM")
        return f"{BINANCE_ARCHIVE_ROOT}/{symbol}/{symbol}-aggTrades-{month}.zip"

    def download_month(self, symbol: str, month: str, *, allow_network: bool = False) -> Path:
        """Download one archive only when explicitly allowed; resumable .part files are retained."""
        if not allow_network:
            raise RuntimeError("network download is disabled; use a supplied archive or set allow_network=True")
        target_dir = self.cache_root / "downloads" / symbol.upper()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{symbol.upper()}-aggTrades-{month}.zip"
        if target.exists():
            return target
        partial = target.with_suffix(".zip.part")
        request = urllib.request.Request(self.archive_url(symbol, month), headers={"User-Agent": "research-pipeline/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        except Exception:
            # The partial file is intentionally preserved for an auditable retry.
            raise
        partial.replace(target)
        return target

    @staticmethod
    def _archive_rows(
        bundle: zipfile.ZipFile,
        member_name: str,
    ) -> Iterator[dict[str, Any]]:
        with bundle.open(member_name) as raw:
            reader = csv.reader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            try:
                first = next(reader)
            except StopIteration:
                raise ValueError(
                    f"aggregate-trade ZIP member {member_name!r} is empty"
                )
            detected = [item.strip().lstrip("\ufeff") for item in first]
            recognized = [FIELD_ALIASES.get(item) for item in detected]
            if any(item is not None for item in recognized):
                header = detected
                normalized_header = {
                    FIELD_ALIASES[item]
                    for item in header
                    if item in FIELD_ALIASES
                }
                missing = sorted(REQUIRED_NORMALIZED_FIELDS - normalized_header)
                if missing:
                    raise ValueError(
                        f"invalid aggregate-trade schema in ZIP member "
                        f"{member_name!r}: detected columns={header}; "
                        f"missing normalized fields={missing}"
                    )
            else:
                if len(first) not in {7, 8}:
                    raise ValueError(
                        f"cannot detect headerless aggregate-trade schema in "
                        f"ZIP member {member_name!r}: detected columns={detected}; "
                        f"missing normalized fields={sorted(REQUIRED_NORMALIZED_FIELDS)}"
                    )
                header = list(ARCHIVE_COLUMN_ORDER[:len(first)])
                yield dict(zip(header, first, strict=True))
            for values in reader:
                if not values or all(not item.strip() for item in values):
                    continue
                if len(values) != len(header):
                    raise ValueError(
                        f"invalid aggregate-trade row width in ZIP member "
                        f"{member_name!r}: detected columns={header}; "
                        f"expected {len(header)} values, got {len(values)}"
                    )
                yield dict(zip(header, values, strict=True))

    def records_from_archive(self, path: str | Path, *, source: str = "binance_usdm_public_archive") -> Iterator[AggregateTrade]:
        archive = Path(path)
        source_hash = _sha256_file(archive)
        def generate() -> Iterator[AggregateTrade]:
            try:
                with zipfile.ZipFile(archive) as bundle:
                    names = [
                        name for name in bundle.namelist()
                        if name.lower().endswith(".csv")
                    ]
                    if len(names) != 1:
                        raise ValueError(
                            "aggregate-trade archive must contain exactly one CSV"
                        )
                    member_name = names[0]
                    source_file = f"{archive.name}!{member_name}"
                    for row in self._archive_rows(bundle, member_name):
                        yield parse_binance_aggregate_trade(
                            row,
                            source_file=source_file,
                            source_hash=source_hash,
                            source=source,
                        )
            except zipfile.BadZipFile as exc:
                raise ValueError(
                    f"corrupted Binance aggregate-trade archive: {archive}"
                ) from exc
        return generate()

    def ingest_records(self, records: Iterable[AggregateTrade], *, symbol: str = "BTCUSDT", source_files: list[str] | None = None) -> tuple[Path, AggregateTradeManifest]:
        ordered: Iterable[AggregateTrade]
        if isinstance(records, Sequence):
            ordered = sorted(
                records,
                key=lambda item: (item.trade_time_utc, item.aggregate_trade_id),
            )
        else:
            ordered = records
        iterator = iter(ordered)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("aggregate-trade dataset is empty")
        staging_root = self.cache_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / (
            f"{symbol.upper()}-{first.source_hash[:16]}.parquet.part"
        )
        dataset_digest = hashlib.sha256()
        dataset_digest.update(b"[")
        writer = pq.ParquetWriter(staging, PARQUET_SCHEMA, compression="zstd")
        chunk: list[dict[str, Any]] = []
        row_count = 0
        duplicate_count = 0
        first_payload = True
        previous: AggregateTrade | None = None
        minimum_date: date | None = None
        maximum_date: date | None = None
        source_hashes: dict[str, str] = {}
        missing_intervals: list[dict[str, Any]] = []
        try:
            for item in (value for pair in ((first,), iterator) for value in pair):
                if previous is not None:
                    if item.trade_time_utc < previous.trade_time_utc:
                        raise ValueError(
                            "aggregate trades are not in timestamp order: "
                            f"{item.trade_time_utc.isoformat()} follows "
                            f"{previous.trade_time_utc.isoformat()}"
                        )
                    if item.aggregate_trade_id < previous.aggregate_trade_id:
                        raise ValueError(
                            "aggregate trade IDs are not in ascending order: "
                            f"{item.aggregate_trade_id} follows "
                            f"{previous.aggregate_trade_id}"
                        )
                    if item.aggregate_trade_id == previous.aggregate_trade_id:
                        duplicate_count += 1
                        continue
                    gap_seconds = (
                        item.trade_time_utc - previous.trade_time_utc
                    ).total_seconds()
                    if gap_seconds > 300 and len(missing_intervals) < 1000:
                        missing_intervals.append(
                            {
                                "start_utc": previous.trade_time_utc.isoformat(),
                                "end_utc": item.trade_time_utc.isoformat(),
                                "gap_seconds": gap_seconds,
                            }
                        )
                payload = item.model_dump(mode="json")
                encoded = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                if not first_payload:
                    dataset_digest.update(b",")
                dataset_digest.update(encoded)
                first_payload = False
                chunk.append(payload)
                if len(chunk) >= 100_000:
                    writer.write_table(
                        pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA)
                    )
                    chunk.clear()
                row_count += 1
                item_date = item.trade_time_utc.date()
                minimum_date = (
                    item_date if minimum_date is None
                    else min(minimum_date, item_date)
                )
                maximum_date = (
                    item_date if maximum_date is None
                    else max(maximum_date, item_date)
                )
                source_hashes[item.source_file] = item.source_hash
                previous = item
            if chunk:
                writer.write_table(
                    pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA)
                )
        finally:
            writer.close()
        dataset_digest.update(b"]")
        dataset_hash = dataset_digest.hexdigest()
        destination = self.cache_root / "normalized" / symbol.upper() / dataset_hash
        parquet = destination / "aggregate_trades.parquet"
        manifest_path = destination / "manifest.json"
        destination.mkdir(parents=True, exist_ok=True)
        if parquet.exists():
            staging.unlink()
        else:
            staging.replace(parquet)
        assert minimum_date is not None and maximum_date is not None
        manifest = AggregateTradeManifest(
            date_start=minimum_date,
            date_end=maximum_date,
            retrieved_at=datetime.now(timezone.utc),
            source_files=source_files or sorted(source_hashes),
            source_file_hashes=source_hashes,
            normalized_dataset_hash=dataset_hash,
            row_count=row_count,
            duplicate_count=duplicate_count,
            missing_interval_diagnostics=missing_intervals,
            manifest_hash="pending",
        )
        raw = manifest.model_dump(mode="python"); raw["manifest_hash"] = _manifest_hash(manifest)
        manifest = AggregateTradeManifest.model_validate(raw)
        if manifest_path.exists():
            existing = AggregateTradeManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
            if existing.normalized_dataset_hash != manifest.normalized_dataset_hash:
                raise RuntimeError("content-addressed manifest collision")
            return parquet, existing
        manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        return parquet, manifest
