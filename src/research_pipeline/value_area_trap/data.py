from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from urllib.parse import urlencode
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


IMPORTER_VERSION = "value-area-trap-aggregate-trades-2"
SCHEMA_VERSION = "binance-usdm-aggregate-trade-1"
BINANCE_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
BINANCE_USDM_AGGREGATE_TRADES_ENDPOINT = "https://fapi.binance.com/fapi/v1/aggTrades"
BINANCE_USDM_EXCHANGE_INFO_ENDPOINT = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SUPPORTED_CROSS_MARKET_TRADFI_SYMBOLS = frozenset({"XAUUSDT", "QQQUSDT", "SPYUSDT"})
SUPPORTED_CROSS_MARKET_CONTRACT_TYPE = "TRADIFI_PERPETUAL"
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


class AggregateTradePartition(StrictModel):
    month: str
    file_name: str
    parquet_hash: str
    normalized_dataset_hash: str
    source_archive: str
    source_archive_hash: str
    row_count: int
    duplicate_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    first_aggregate_trade_id: int
    last_aggregate_trade_id: int
    missing_interval_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    continuity_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    repair_status: str = "ARCHIVE_ONLY"
    repair_audit_path: str | None = None
    repair_audit_hash: str | None = None


class BinanceSymbolMetadata(StrictModel):
    """Pinned Binance USD-M filters; never infer these from BTC defaults."""

    symbol: str
    contract_type: str
    tick_size: Decimal = Field(gt=0)
    quantity_step: Decimal = Field(gt=0)
    minimum_quantity: Decimal = Field(gt=0)
    onboard_date: date | None = None
    pair: str | None = None
    status: str | None = None
    underlying_type: str | None = None
    underlying_sub_type: list[str] = Field(default_factory=list)
    margin_asset: str | None = None
    quote_asset: str | None = None
    delivery_date_epoch_ms: int | None = None
    raw_symbol_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_symbol_hash: str | None = None
    source: str = "Binance USD-M Futures exchangeInfo"
    source_hash: str


class BinanceSymbolMetadataArtifact(StrictModel):
    provider: str = "Binance USD-M Futures"
    retrieved_at: datetime
    symbols: list[BinanceSymbolMetadata]
    artifact_hash: str = "pending"


class MonthlyAggregateTradeManifest(StrictModel):
    """Immutable combined manifest for verified monthly aggregate partitions."""

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
    partitions: list[AggregateTradePartition]
    symbol_metadata: BinanceSymbolMetadata | None = None
    metadata_artifact_path: str | None = None
    metadata_artifact_hash: str | None = None
    schema_version: str = SCHEMA_VERSION
    importer_version: str = IMPORTER_VERSION
    manifest_hash: str = "pending"


def _manifest_hash(manifest: AggregateTradeManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_hash", None)
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _monthly_manifest_hash(manifest: MonthlyAggregateTradeManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_hash", None)
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _metadata_artifact_hash(artifact: BinanceSymbolMetadataArtifact) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("artifact_hash", None)
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def parse_binance_symbol_metadata(item: dict[str, Any], *, source_hash: str) -> BinanceSymbolMetadata:
    """Parse and pin one exact USD-M ``exchangeInfo`` symbol object.

    The raw object is kept in the immutable metadata artifact so eligibility
    decisions can be audited without another network call.  It intentionally
    includes Binance's full symbol object, including its exchange filters.
    """

    symbol = str(item["symbol"]).upper()
    filters = {entry.get("filterType"): entry for entry in item.get("filters", [])}
    raw_symbol = json.loads(json.dumps(item, sort_keys=True, separators=(",", ":")))
    try:
        return BinanceSymbolMetadata(
            symbol=symbol,
            contract_type=str(item["contractType"]),
            tick_size=Decimal(str(filters["PRICE_FILTER"]["tickSize"])),
            quantity_step=Decimal(str(filters["LOT_SIZE"]["stepSize"])),
            minimum_quantity=Decimal(str(filters["LOT_SIZE"]["minQty"])),
            onboard_date=_utc_millis(item["onboardDate"]).date() if item.get("onboardDate") else None,
            pair=str(item["pair"]) if item.get("pair") is not None else None,
            status=str(item["status"]) if item.get("status") is not None else None,
            underlying_type=str(item["underlyingType"]) if item.get("underlyingType") is not None else None,
            underlying_sub_type=[str(value) for value in item.get("underlyingSubType", [])],
            margin_asset=str(item["marginAsset"]) if item.get("marginAsset") is not None else None,
            quote_asset=str(item["quoteAsset"]) if item.get("quoteAsset") is not None else None,
            delivery_date_epoch_ms=int(item["deliveryDate"]) if item.get("deliveryDate") is not None else None,
            raw_symbol_metadata=raw_symbol,
            raw_symbol_hash=_sha256_bytes(json.dumps(raw_symbol, sort_keys=True, separators=(",", ":")).encode()),
            source_hash=source_hash,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Binance exchangeInfo has incomplete filters or metadata for {symbol}") from exc


def cross_market_symbol_diagnostic(metadata: BinanceSymbolMetadata) -> dict[str, Any]:
    """Return only the eligibility evidence needed in an error or report."""

    return {
        "symbol": metadata.symbol,
        "pair": metadata.pair,
        "contractType": metadata.contract_type,
        "status": metadata.status,
        "underlyingType": metadata.underlying_type,
        "underlyingSubType": metadata.underlying_sub_type,
        "marginAsset": metadata.margin_asset,
        "quoteAsset": metadata.quote_asset,
        "onboardDate": metadata.raw_symbol_metadata.get("onboardDate"),
        "deliveryDate": metadata.delivery_date_epoch_ms,
        "raw_symbol_hash": metadata.raw_symbol_hash,
    }


def validate_cross_market_symbol_eligibility(metadata: BinanceSymbolMetadata) -> None:
    """Enforce the deliberately narrow supported TradFi-perpetual contract.

    Binance labels these three instruments ``TRADIFI_PERPETUAL`` and gives
    them a far-future delivery timestamp.  The explicit contract type and the
    ``TradFi`` subtype, rather than the timestamp alone, establish that this
    is Binance's non-expiring TradFi perpetual representation.  No ordinary
    USD-M or spot symbol is admitted through this path.
    """

    reasons: list[str] = []
    if metadata.symbol not in SUPPORTED_CROSS_MARKET_TRADFI_SYMBOLS:
        reasons.append("symbol is outside the explicit frozen cross-market allowlist")
    if metadata.pair != metadata.symbol:
        reasons.append("pair does not match the requested USD-M symbol")
    if metadata.status != "TRADING":
        reasons.append("Binance status is not TRADING")
    if metadata.margin_asset != "USDT" or metadata.quote_asset != "USDT":
        reasons.append("quote and margin assets must both be USDT")
    if metadata.contract_type != SUPPORTED_CROSS_MARKET_CONTRACT_TYPE:
        reasons.append(f"contractType must be {SUPPORTED_CROSS_MARKET_CONTRACT_TYPE}")
    if "TradFi" not in metadata.underlying_sub_type:
        reasons.append("underlyingSubType must include TradFi")
    if not metadata.underlying_type:
        reasons.append("underlyingType is required")
    if not metadata.raw_symbol_metadata or not metadata.raw_symbol_hash:
        reasons.append("raw Binance exchangeInfo symbol metadata is not pinned")
    if reasons:
        raise ValueError(
            "cross-market symbol eligibility rejected: "
            + json.dumps(
                {
                    "reasons": reasons,
                    "metadata": cross_market_symbol_diagnostic(metadata),
                    "raw_symbol_metadata": metadata.raw_symbol_metadata,
                },
                sort_keys=True,
            )
        )


def _months_between(start_month: str, end_month: str) -> list[str]:
    def parse(value: str) -> tuple[int, int]:
        try:
            year, month = value.split("-", 1)
            parsed = int(year), int(month)
        except (AttributeError, ValueError) as exc:
            raise ValueError("months must use YYYY-MM") from exc
        if not 1 <= parsed[1] <= 12:
            raise ValueError("months must use YYYY-MM")
        return parsed

    first, last = parse(start_month), parse(end_month)
    if first > last:
        raise ValueError("start month must not follow end month")
    months: list[str] = []
    year, month = first
    while (year, month) <= last:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


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
        # A call exposes only the actions for that call.  It is deliberately
        # process-local; immutable partition metadata remains the audit record.
        self.last_ingestion_diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def archive_url(symbol: str, month: str) -> str:
        symbol = symbol.upper()
        if len(month) != 7 or month[4] != "-":
            raise ValueError("month must use YYYY-MM")
        return f"{BINANCE_ARCHIVE_ROOT}/{symbol}/{symbol}-aggTrades-{month}.zip"

    def download_month(self, symbol: str, month: str, *, allow_network: bool = False) -> Path:
        """Download one archive only when explicitly allowed; resumable .part files are retained."""
        target_dir = self.cache_root / "downloads" / symbol.upper()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{symbol.upper()}-aggTrades-{month}.zip"
        if target.exists():
            return target
        if not allow_network:
            raise RuntimeError("network download is disabled; use a supplied archive or set allow_network=True")
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

    def resolve_symbol_metadata(
        self,
        symbols: Sequence[str],
        *,
        artifact_path: str | Path | None = None,
        allow_network: bool = False,
    ) -> tuple[Path, BinanceSymbolMetadataArtifact]:
        """Load a hash-pinned exchange-info artifact or create one explicitly.

        Network access is opt-in and only obtains public exchange filters.  The
        resulting artifact becomes part of every cross-market manifest.
        """

        requested = [item.upper() for item in symbols]
        if len(set(requested)) != len(requested):
            raise ValueError("symbol metadata request contains duplicate symbols")
        if artifact_path is not None:
            path = Path(artifact_path).resolve()
            try:
                artifact = BinanceSymbolMetadataArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Binance symbol metadata artifact: {path}") from exc
            if _metadata_artifact_hash(artifact) != artifact.artifact_hash:
                raise ValueError(f"Binance symbol metadata artifact hash mismatch: {path}")
        else:
            if not allow_network:
                raise ValueError("a pinned symbol metadata artifact is required unless --allow-network is set")
            request = urllib.request.Request(BINANCE_USDM_EXCHANGE_INFO_ENDPOINT, headers={"User-Agent": "research-pipeline/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
            try:
                payload = json.loads(body.decode("utf-8"))
                source_symbols = {item["symbol"].upper(): item for item in payload["symbols"]}
            except (UnicodeDecodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Binance exchangeInfo response is invalid") from exc
            source_hash = _sha256_bytes(body)
            selected: list[BinanceSymbolMetadata] = []
            for symbol in requested:
                item = source_symbols.get(symbol)
                if item is None:
                    raise ValueError(
                        "cross-market symbol eligibility rejected: "
                        + json.dumps({"reasons": ["symbol does not exist in Binance USD-M exchangeInfo"], "metadata": {"symbol": symbol}})
                    )
                selected.append(parse_binance_symbol_metadata(item, source_hash=source_hash))
            unsigned = BinanceSymbolMetadataArtifact(
                retrieved_at=datetime.now(timezone.utc), symbols=selected, artifact_hash="pending"
            )
            artifact = unsigned.model_copy(update={"artifact_hash": _metadata_artifact_hash(unsigned)})
            path = self.cache_root / "metadata" / f"binance-usdm-exchange-info-{artifact.artifact_hash}.json"
            content = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"immutable Binance symbol metadata collision: {path}")
            if not path.exists():
                path.write_bytes(content)
        by_symbol = {item.symbol.upper(): item for item in artifact.symbols}
        missing = [symbol for symbol in requested if symbol not in by_symbol]
        if missing:
            raise ValueError(f"pinned Binance symbol metadata is missing: {missing}")
        for symbol in requested:
            validate_cross_market_symbol_eligibility(by_symbol[symbol])
        return path, artifact

    @staticmethod
    def validate_complete_calendar_months(
        manifest: MonthlyAggregateTradeManifest,
        *,
        start_month: str,
        end_month: str,
        edge_tolerance_seconds: int = 3600,
    ) -> list[dict[str, Any]]:
        """Require each requested archive to cover a calendar month to a small edge tolerance."""

        expected = _months_between(start_month, end_month)
        partitions = {item.month: item for item in manifest.partitions}
        diagnostics: list[dict[str, Any]] = []
        for month in expected:
            partition = partitions.get(month)
            if partition is None:
                raise ValueError(f"INCOMPLETE_CALENDAR_MONTH: missing requested partition {month}")
            year, number = (int(value) for value in month.split("-", 1))
            start = datetime(year, number, 1, tzinfo=timezone.utc)
            next_start = datetime(year + (number == 12), 1 if number == 12 else number + 1, 1, tzinfo=timezone.utc)
            first_gap = (partition.first_timestamp - start).total_seconds()
            last_gap = (next_start - partition.last_timestamp).total_seconds()
            if first_gap < 0 or last_gap < 0 or first_gap > edge_tolerance_seconds or last_gap > edge_tolerance_seconds:
                raise ValueError(
                    "INCOMPLETE_CALENDAR_MONTH: "
                    f"{month} first={partition.first_timestamp.isoformat()} last={partition.last_timestamp.isoformat()} "
                    f"edge_tolerance_seconds={edge_tolerance_seconds}"
                )
            diagnostics.append({
                "month": month,
                "first_timestamp": partition.first_timestamp.isoformat(),
                "last_timestamp": partition.last_timestamp.isoformat(),
                "first_edge_gap_seconds": first_gap,
                "last_edge_gap_seconds": last_gap,
                "status": "COMPLETE_WITHIN_EDGE_TOLERANCE",
            })
        return diagnostics

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

    @staticmethod
    def _same_trade_payload(left: AggregateTrade, right: AggregateTrade) -> bool:
        """Compare identity-bearing trade fields without source provenance."""

        fields = (
            "event_time_utc", "trade_time_utc", "aggregate_trade_id",
            "first_trade_id", "last_trade_id", "price", "quantity_base",
            "notional_quote", "buyer_is_maker", "aggressor_side", "signed_quantity",
        )
        return all(getattr(left, field) == getattr(right, field) for field in fields)

    def _fetch_api_aggregate_trades_page(
        self,
        symbol: str,
        from_aggregate_trade_id: int,
        *,
        limit: int = 1000,
    ) -> tuple[list[AggregateTrade], dict[str, Any]]:
        """Fetch one explicit historical-ID page from Binance USD-M Futures.

        This is intentionally separate from public-archive download.  Callers
        must opt in to gap repair and persist the request/response provenance.
        """

        query = urlencode({
            "symbol": symbol.upper(),
            "fromId": from_aggregate_trade_id,
            "limit": limit,
        })
        url = f"{BINANCE_USDM_AGGREGATE_TRADES_ENDPOINT}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "research-pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Binance aggregate-trade gap-repair response is not valid JSON") from exc
        if not isinstance(payload, list):
            raise ValueError("Binance aggregate-trade gap-repair response must be a JSON array")
        response_hash = _sha256_bytes(body)
        source_file = f"{url}#sha256={response_hash}"
        rows = [
            parse_binance_aggregate_trade(
                row,
                source_file=source_file,
                source_hash=response_hash,
                source="binance_usdm_gap_repair_api",
            )
            for row in payload
        ]
        return rows, {
            "url": url,
            "from_aggregate_trade_id": from_aggregate_trade_id,
            "limit": limit,
            "response_hash": response_hash,
            "response_row_count": len(rows),
        }

    def _repair_missing_ids(
        self,
        *,
        symbol: str,
        month: str,
        previous: AggregateTrade,
        following: AggregateTrade,
        allow_network: bool,
        allow_gap_repair: bool,
    ) -> tuple[list[AggregateTrade], dict[str, Any]]:
        """Return exactly the absent IDs between two archive records, or fail closed."""

        first_missing = previous.aggregate_trade_id + 1
        last_missing = following.aggregate_trade_id - 1
        diagnostic: dict[str, Any] = {
            "previous_aggregate_trade_id": previous.aggregate_trade_id,
            "previous_timestamp": previous.trade_time_utc.isoformat(),
            "next_aggregate_trade_id": following.aggregate_trade_id,
            "next_timestamp": following.trade_time_utc.isoformat(),
            "missing_id_start": first_missing,
            "missing_id_end": last_missing,
            "missing_id_count": last_missing - first_missing + 1,
            "ids_continuous": False,
            "repair_status": "DATA_GAP_UNREPAIRABLE",
            "api_requests": [],
            "fetched_row_count": 0,
        }
        if not allow_gap_repair or not allow_network:
            raise ValueError(
                "DATA_GAP_UNREPAIRABLE: aggregate-trade IDs are missing "
                f"between {previous.aggregate_trade_id} at {previous.trade_time_utc.isoformat()} "
                f"and {following.aggregate_trade_id} at {following.trade_time_utc.isoformat()} "
                f"(missing_id_count={diagnostic['missing_id_count']}); "
                "rerun only with --allow-network --allow-gap-repair to request those IDs"
            )
        repaired: list[AggregateTrade] = []
        expected = first_missing
        while expected <= last_missing:
            try:
                page, request = self._fetch_api_aggregate_trades_page(
                    symbol, expected, limit=min(1000, last_missing - expected + 1)
                )
            except (OSError, ValueError, urllib.error.URLError) as exc:
                raise ValueError(
                    "DATA_GAP_UNREPAIRABLE: Binance historical aggregate-trade API "
                    f"could not supply IDs {expected} through {last_missing}: {exc}"
                ) from exc
            diagnostic["api_requests"].append(request)
            if not page:
                raise ValueError(
                    "DATA_GAP_UNREPAIRABLE: Binance historical aggregate-trade API "
                    f"returned no rows beginning at missing ID {expected}"
                )
            last_seen: int | None = None
            for row in page:
                if last_seen is not None and row.aggregate_trade_id <= last_seen:
                    raise ValueError(
                        "DATA_GAP_UNREPAIRABLE: Binance gap-repair response has "
                        f"out-of-order or duplicate ID {row.aggregate_trade_id}"
                    )
                last_seen = row.aggregate_trade_id
                if row.aggregate_trade_id in {previous.aggregate_trade_id, following.aggregate_trade_id}:
                    counterpart = previous if row.aggregate_trade_id == previous.aggregate_trade_id else following
                    if not self._same_trade_payload(row, counterpart):
                        raise ValueError(
                            "DATA_GAP_UNREPAIRABLE: conflicting API/archive duplicate "
                            f"aggregate-trade ID {row.aggregate_trade_id}"
                        )
                    continue
                if row.aggregate_trade_id < first_missing or row.aggregate_trade_id > last_missing:
                    raise ValueError(
                        "DATA_GAP_UNREPAIRABLE: Binance gap-repair response returned "
                        f"unexpected aggregate-trade ID {row.aggregate_trade_id}"
                    )
                if row.aggregate_trade_id != expected:
                    raise ValueError(
                        "DATA_GAP_UNREPAIRABLE: Binance gap-repair response is missing "
                        f"aggregate-trade ID {expected}"
                    )
                expected += 1
                if (row.trade_time_utc.year, row.trade_time_utc.month) != tuple(int(value) for value in month.split("-", 1)):
                    raise ValueError(
                        "DATA_GAP_UNREPAIRABLE: Binance gap-repair timestamp is outside "
                        f"{month}: {row.trade_time_utc.isoformat()}"
                    )
                repaired.append(row)
        if expected != last_missing + 1:
            raise ValueError("DATA_GAP_UNREPAIRABLE: incomplete aggregate-trade ID repair")
        if any(
            right.trade_time_utc < left.trade_time_utc
            for left, right in zip([previous, *repaired], [*repaired, following], strict=True)
        ):
            raise ValueError("DATA_GAP_UNREPAIRABLE: repaired aggregate-trade timestamps are out of order")
        diagnostic.update({
            "repair_status": "API_GAP_FILLED",
            "fetched_row_count": len(repaired),
            "fetched_timestamp_start": repaired[0].trade_time_utc.isoformat(),
            "fetched_timestamp_end": repaired[-1].trade_time_utc.isoformat(),
        })
        return repaired, diagnostic

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

    def _partition_cache_path(self, symbol: str, month: str) -> Path:
        return self.cache_root / "normalized" / symbol.upper() / "monthly_partitions" / month / "partition.json"

    def _load_reusable_partition(self, symbol: str, month: str, archive: Path) -> AggregateTradePartition | None:
        index = self._partition_cache_path(symbol, month)
        if not index.is_file():
            return None
        try:
            payload = AggregateTradePartition.model_validate(json.loads(index.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if payload.source_archive_hash != _sha256_file(archive):
            return None
        parquet = index.parent / payload.file_name
        if not parquet.is_file() or _sha256_file(parquet) != payload.parquet_hash:
            return None
        return payload

    def _ingest_month_partition(
        self,
        symbol: str,
        month: str,
        archive: Path,
        *,
        allow_network: bool,
        allow_gap_repair: bool,
    ) -> AggregateTradePartition:
        reusable = self._load_reusable_partition(symbol, month, archive)
        if reusable is not None:
            self._last_partition_action = {
                "month": month,
                "action": "SKIPPED_HASH_VERIFIED",
                "reused": True,
                "repair_status": reusable.repair_status,
                "archive": str(archive),
            }
            return reusable
        expected_year, expected_month = (int(value) for value in month.split("-", 1))
        statistics: dict[str, Any] = {"first": None, "last": None, "first_id": None, "last_id": None}
        continuity_diagnostics: list[dict[str, Any]] = []
        repair_audits: list[dict[str, Any]] = []
        repair_status = "ARCHIVE_ONLY"

        def checked_records() -> Iterator[AggregateTrade]:
            nonlocal repair_status
            previous: AggregateTrade | None = None
            for record in self.records_from_archive(archive):
                timestamp = record.trade_time_utc
                if (timestamp.year, timestamp.month) != (expected_year, expected_month):
                    raise ValueError(
                        f"archive {archive.name} contains out-of-range timestamp "
                        f"{timestamp.isoformat()} for month {month}"
                    )
                if previous is not None:
                    if record.aggregate_trade_id < previous.aggregate_trade_id:
                        raise ValueError(
                            "aggregate trades are not in ascending ID order: "
                            f"{record.aggregate_trade_id} follows {previous.aggregate_trade_id}"
                        )
                    if record.aggregate_trade_id == previous.aggregate_trade_id:
                        if not self._same_trade_payload(record, previous):
                            raise ValueError(
                                "conflicting duplicate aggregate-trade ID in archive: "
                                f"{record.aggregate_trade_id}"
                            )
                        continuity_diagnostics.append({
                            "previous_aggregate_trade_id": previous.aggregate_trade_id,
                            "previous_timestamp": previous.trade_time_utc.isoformat(),
                            "next_aggregate_trade_id": record.aggregate_trade_id,
                            "next_timestamp": record.trade_time_utc.isoformat(),
                            "missing_id_count": 0,
                            "ids_continuous": True,
                            "repair_status": "DUPLICATE_ARCHIVE_ROW_DEDUPLICATED",
                        })
                        # Preserve the row for the established streaming
                        # deduplicator and its duplicate counter.
                        yield record
                        continue
                    missing_id_count = record.aggregate_trade_id - previous.aggregate_trade_id - 1
                    timestamp_gap_seconds = (record.trade_time_utc - previous.trade_time_utc).total_seconds()
                    if missing_id_count:
                        repaired, repair_diagnostic = self._repair_missing_ids(
                            symbol=symbol,
                            month=month,
                            previous=previous,
                            following=record,
                            allow_network=allow_network,
                            allow_gap_repair=allow_gap_repair,
                        )
                        continuity_diagnostics.append({
                            **repair_diagnostic,
                            "timestamp_gap_seconds": timestamp_gap_seconds,
                        })
                        repair_audits.append(repair_diagnostic)
                        repair_status = "API_GAP_FILLED"
                        yield from repaired
                    elif timestamp_gap_seconds > 300:
                        continuity_diagnostics.append({
                            "previous_aggregate_trade_id": previous.aggregate_trade_id,
                            "previous_timestamp": previous.trade_time_utc.isoformat(),
                            "next_aggregate_trade_id": record.aggregate_trade_id,
                            "next_timestamp": record.trade_time_utc.isoformat(),
                            "missing_id_count": 0,
                            "ids_continuous": True,
                            "timestamp_gap_seconds": timestamp_gap_seconds,
                            "repair_status": "CONTINUOUS_IDS_NO_TRADE_INTERVAL",
                        })
                        if repair_status == "ARCHIVE_ONLY":
                            repair_status = "ARCHIVE_CONTINUOUS_ID_TIMESTAMP_GAP"
                if statistics["first"] is None:
                    statistics["first"] = timestamp
                    statistics["first_id"] = record.aggregate_trade_id
                statistics["last"] = timestamp
                statistics["last_id"] = record.aggregate_trade_id
                yield record
                previous = record

        parquet, manifest = self.ingest_records(checked_records(), symbol=symbol, source_files=[str(archive.resolve())])
        if statistics["first"] is None:
            raise ValueError(f"aggregate-trade archive has no records: {archive}")
        index = self._partition_cache_path(symbol, month)
        partition_dir = index.parent
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_file = f"{manifest.normalized_dataset_hash}.parquet"
        destination = partition_dir / partition_file
        if destination.exists():
            if _sha256_file(destination) != _sha256_file(parquet):
                raise ValueError(f"immutable monthly partition collision: {destination}")
        else:
            shutil.copyfile(parquet, destination)
        audit_path: Path | None = None
        audit_hash: str | None = None
        if repair_audits:
            audit_path = partition_dir / "gap-repair-audit.json"
            audit = {
                "repair_status": "API_GAP_FILLED",
                "month": month,
                "source_archive": str(archive.resolve()),
                "source_archive_hash": _sha256_file(archive),
                "repairs": repair_audits,
                "output_hashes": {
                    "normalized_archive_parquet": _sha256_file(parquet),
                    "monthly_partition_parquet": _sha256_file(destination),
                    "normalized_dataset_hash": manifest.normalized_dataset_hash,
                },
            }
            audit_content = json.dumps(audit, indent=2, sort_keys=True).encode("utf-8")
            if audit_path.exists() and audit_path.read_bytes() != audit_content:
                raise ValueError(f"immutable gap-repair audit collision: {audit_path}")
            if not audit_path.exists():
                audit_path.write_bytes(audit_content)
            audit_hash = _sha256_file(audit_path)
        partition = AggregateTradePartition(
            month=month,
            file_name=partition_file,
            parquet_hash=_sha256_file(destination),
            normalized_dataset_hash=manifest.normalized_dataset_hash,
            source_archive=str(archive.resolve()),
            source_archive_hash=_sha256_file(archive),
            row_count=manifest.row_count,
            duplicate_count=manifest.duplicate_count,
            first_timestamp=statistics["first"],
            last_timestamp=statistics["last"],
            first_aggregate_trade_id=statistics["first_id"],
            last_aggregate_trade_id=statistics["last_id"],
            missing_interval_diagnostics=manifest.missing_interval_diagnostics,
            continuity_diagnostics=continuity_diagnostics,
            repair_status=repair_status,
            repair_audit_path=str(audit_path) if audit_path else None,
            repair_audit_hash=audit_hash,
        )
        content = json.dumps(partition.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
        if index.exists() and index.read_bytes() != content:
            raise ValueError(f"immutable monthly partition index collision: {index}")
        if not index.exists():
            index.write_bytes(content)
        self._last_partition_action = {
            "month": month,
            "action": "REPAIRED_API_GAP" if repair_audits else "NEWLY_PROCESSED",
            "reused": False,
            "repair_status": repair_status,
            "archive": str(archive),
            "continuity_diagnostics": continuity_diagnostics,
        }
        return partition

    def ingest_monthly_range(
        self,
        *,
        symbol: str,
        start_month: str,
        end_month: str,
        allow_network: bool = False,
        allow_gap_repair: bool = False,
        symbol_metadata: BinanceSymbolMetadata | None = None,
        metadata_artifact_path: str | Path | None = None,
        metadata_artifact_hash: str | None = None,
    ) -> tuple[Path, MonthlyAggregateTradeManifest]:
        """Build a resumable, immutable combined manifest from monthly archives.

        Completed partitions are reused only if both their archive and Parquet
        hashes still match the immutable index. Aggregate trades are streamed
        one archive at a time and never substituted with OHLCV data.
        """

        if allow_gap_repair and not allow_network:
            raise ValueError("--allow-gap-repair requires --allow-network")
        if symbol_metadata is not None and symbol_metadata.symbol.upper() != symbol.upper():
            raise ValueError("symbol metadata does not match monthly aggregate-trade symbol")
        months = _months_between(start_month, end_month)
        self.last_ingestion_diagnostics = []
        partitions: list[AggregateTradePartition] = []
        for month in months:
            archive = self.download_month(symbol, month, allow_network=allow_network)
            partitions.append(self._ingest_month_partition(
                symbol,
                month,
                archive,
                allow_network=allow_network,
                allow_gap_repair=allow_gap_repair,
            ))
            self.last_ingestion_diagnostics.append(dict(self._last_partition_action))
        previous: AggregateTradePartition | None = None
        for partition in partitions:
            if previous is not None:
                if partition.first_timestamp <= previous.last_timestamp:
                    raise ValueError(f"monthly partition overlap: {previous.month} -> {partition.month}")
                if partition.first_aggregate_trade_id <= previous.last_aggregate_trade_id:
                    raise ValueError(f"aggregate trade ID overlap: {previous.month} -> {partition.month}")
                if partition.first_aggregate_trade_id != previous.last_aggregate_trade_id + 1:
                    raise ValueError(
                        "DATA_GAP_UNREPAIRABLE: archive truncation or inter-month "
                        f"aggregate-trade ID gap between {previous.month} and {partition.month}"
                    )
            previous = partition
        identity = {
            "symbol": symbol.upper(),
            "months": [
                {"month": item.month, "normalized_dataset_hash": item.normalized_dataset_hash, "parquet_hash": item.parquet_hash}
                for item in partitions
            ],
        }
        dataset_hash = _sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        root = self.cache_root / "normalized" / symbol.upper() / dataset_hash
        root.mkdir(parents=True, exist_ok=True)
        combined_partitions: list[AggregateTradePartition] = []
        for item in partitions:
            source = self._partition_cache_path(symbol, item.month).parent / item.file_name
            name = f"{item.month}.parquet"
            target = root / name
            if target.exists():
                if _sha256_file(target) != item.parquet_hash:
                    raise ValueError(f"immutable combined partition collision: {target}")
            else:
                shutil.copyfile(source, target)
            combined_partitions.append(item.model_copy(update={"file_name": name, "parquet_hash": _sha256_file(target)}))
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = self.validate_monthly_manifest(manifest_path)
            if existing.normalized_dataset_hash != dataset_hash:
                raise ValueError(f"immutable combined manifest collision: {manifest_path}")
            if symbol_metadata is not None and existing.symbol_metadata != symbol_metadata:
                raise ValueError(f"immutable combined manifest metadata collision: {manifest_path}")
            return manifest_path, existing
        manifest = MonthlyAggregateTradeManifest(
            symbol=symbol.upper(),
            date_start=partitions[0].first_timestamp.date(),
            date_end=partitions[-1].last_timestamp.date(),
            retrieved_at=datetime.now(timezone.utc),
            source_files=[item.source_archive for item in partitions],
            source_file_hashes={item.source_archive: item.source_archive_hash for item in partitions},
            normalized_dataset_hash=dataset_hash,
            row_count=sum(item.row_count for item in combined_partitions),
            duplicate_count=sum(item.duplicate_count for item in combined_partitions),
            partitions=combined_partitions,
            symbol_metadata=symbol_metadata,
            metadata_artifact_path=str(Path(metadata_artifact_path).resolve()) if metadata_artifact_path else None,
            metadata_artifact_hash=metadata_artifact_hash,
            manifest_hash="pending",
        )
        raw = manifest.model_dump(mode="python")
        raw["manifest_hash"] = _monthly_manifest_hash(manifest)
        manifest = MonthlyAggregateTradeManifest.model_validate(raw)
        content = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
        if manifest_path.exists():
            if manifest_path.read_bytes() != content:
                raise ValueError(f"immutable combined manifest collision: {manifest_path}")
        else:
            manifest_path.write_bytes(content)
        return manifest_path, manifest

    def validate_monthly_manifest(self, path: str | Path) -> MonthlyAggregateTradeManifest:
        manifest_path = Path(path).resolve()
        manifest = MonthlyAggregateTradeManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        if _monthly_manifest_hash(manifest) != manifest.manifest_hash:
            raise ValueError(f"monthly aggregate-trade manifest hash mismatch: {manifest_path}")
        if manifest.symbol_metadata is not None:
            if manifest.symbol_metadata.symbol.upper() != manifest.symbol.upper():
                raise ValueError("monthly aggregate-trade manifest symbol metadata mismatch")
            if not manifest.metadata_artifact_path or not manifest.metadata_artifact_hash:
                raise ValueError("monthly aggregate-trade manifest metadata provenance is incomplete")
            metadata_path = Path(manifest.metadata_artifact_path)
            try:
                artifact = BinanceSymbolMetadataArtifact.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"monthly aggregate-trade metadata artifact is unavailable: {metadata_path}") from exc
            if artifact.artifact_hash != manifest.metadata_artifact_hash or _metadata_artifact_hash(artifact) != artifact.artifact_hash:
                raise ValueError(f"monthly aggregate-trade metadata artifact hash mismatch: {metadata_path}")
            if manifest.symbol_metadata not in artifact.symbols:
                raise ValueError(f"monthly aggregate-trade metadata symbol is not pinned: {manifest.symbol}")
        expected_months = _months_between(manifest.partitions[0].month, manifest.partitions[-1].month)
        if [item.month for item in manifest.partitions] != expected_months:
            raise ValueError("monthly aggregate-trade manifest has a missing or unordered month")
        for source, expected_hash in manifest.source_file_hashes.items():
            archive = Path(source)
            if not archive.is_file() or _sha256_file(archive) != expected_hash:
                raise ValueError(f"monthly aggregate-trade source archive hash mismatch: {archive}")
        previous: AggregateTradePartition | None = None
        rows = 0
        for partition in manifest.partitions:
            parquet = manifest_path.parent / partition.file_name
            if not parquet.is_file() or _sha256_file(parquet) != partition.parquet_hash:
                raise ValueError(f"monthly aggregate-trade partition hash mismatch: {parquet}")
            metadata = pq.ParquetFile(parquet).metadata
            if metadata.num_rows != partition.row_count:
                raise ValueError(f"monthly aggregate-trade partition row count mismatch: {parquet}")
            if previous and (partition.first_timestamp <= previous.last_timestamp or partition.first_aggregate_trade_id <= previous.last_aggregate_trade_id):
                raise ValueError(f"monthly aggregate-trade partition overlap: {previous.month} -> {partition.month}")
            if partition.repair_status == "API_GAP_FILLED":
                if not partition.repair_audit_path or not partition.repair_audit_hash:
                    raise ValueError(f"monthly aggregate-trade repair audit is missing: {partition.month}")
                audit_path = Path(partition.repair_audit_path)
                if not audit_path.is_file() or _sha256_file(audit_path) != partition.repair_audit_hash:
                    raise ValueError(f"monthly aggregate-trade repair audit hash mismatch: {audit_path}")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("source_archive_hash") != partition.source_archive_hash:
                    raise ValueError(f"monthly aggregate-trade repair audit source mismatch: {audit_path}")
                if audit.get("output_hashes", {}).get("monthly_partition_parquet") != partition.parquet_hash:
                    raise ValueError(f"monthly aggregate-trade repair audit output mismatch: {audit_path}")
            previous = partition
            rows += partition.row_count
        if rows != manifest.row_count:
            raise ValueError("monthly aggregate-trade manifest row count does not equal partition rows")
        return manifest
