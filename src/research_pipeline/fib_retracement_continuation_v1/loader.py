from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .manifests import ManifestError, load_manifest, verify_manifest
from .models import Bar


def parquet_schema_sha256(source: Path) -> str:
    """Stable schema identity used by the synthetic development contracts."""
    return hashlib.sha256(str(pq.ParquetFile(source).schema_arrow).encode("utf-8")).hexdigest()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ManifestError("FIB09_V1_TIMESTAMP_TYPE_INVALID")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ManifestError("FIB09_V1_TIMESTAMP_NOT_UTC")
    return value.astimezone(timezone.utc)


def load_development_bars(manifest_path: str | Path, *, development_start: datetime, development_end: datetime, chronology_claim: dict[str, object]) -> tuple[list[Bar], dict[str, object]]:
    """Read only row groups sealed as development; holdout groups are never decoded."""
    path, manifest = load_manifest(manifest_path)
    development_start_at = _timestamp(development_start)
    development_end_at = _timestamp(development_end)

    source = Path(manifest["absoluteSourcePath"])
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ["open", "high", "low", "close", "volume", "timestamp"]:
        raise ManifestError("FIB09_V1_PARQUET_SCHEMA_COLUMNS_MISMATCH")
    schema_hash = parquet_schema_sha256(source)

    development_groups: list[int] = []
    timestamp_index = parquet.schema_arrow.get_field_index("timestamp")
    for index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(index).column(timestamp_index).statistics
        if stats is None or not stats.has_min_max:
            raise ManifestError("FIB09_V1_CHRONOLOGY_ISOLATION_UNPROVABLE")
        minimum, maximum = _timestamp(stats.min), _timestamp(stats.max)
        if minimum >= development_start_at and maximum < development_end_at:
            development_groups.append(index)
        elif minimum >= development_end_at or maximum < development_start_at:
            continue
        else:
            raise ManifestError("FIB09_V1_CHRONOLOGY_ROW_GROUP_CROSSES_LOCK")
    if not development_groups:
        raise ManifestError("FIB09_V1_DEVELOPMENT_ROWS_MISSING")

    # The row-group proof above is deliberately stricter than pushdown alone:
    # a group crossing the lock is rejected before any value can be decoded.
    scanner = ds.dataset(source, format="parquet").scanner(
        columns=["open", "high", "low", "close", "volume", "timestamp"],
        filter=(ds.field("timestamp") >= development_start_at) & (ds.field("timestamp") < development_end_at),
    )
    table = scanner.to_table()
    rows = [dict(zip(table.column_names, values)) for values in zip(*(column.to_pylist() for column in table.columns))]
    verified = verify_manifest(path, mode="development", schema_hash=schema_hash)
    bars = [Bar(_timestamp(row["timestamp"]), *(Decimal(str(row[key])) for key in ("open", "high", "low", "close", "volume"))) for row in rows]
    if len(bars) != int(chronology_claim.get("development_row_count", -1)):
        raise ManifestError("FIB09_V1_CHRONOLOGY_DEVELOPMENT_COUNT_MISMATCH")
    if bars[0].timestamp != development_start_at or bars[-1].timestamp >= development_end_at:
        raise ManifestError("FIB09_V1_DEVELOPMENT_TIMESTAMP_COVERAGE_MISMATCH")
    if bars[-1].timestamp.isoformat() != str(chronology_claim.get("development_final_timestamp")):
        raise ManifestError("FIB09_V1_CHRONOLOGY_DEVELOPMENT_TIMESTAMP_MISMATCH")
    if any(not development_start_at <= bar.timestamp < development_end_at for bar in bars):
        raise ManifestError("FIB09_V1_CHRONOLOGY_ISOLATION_FAILED")
    return bars, {**verified, "development_end": development_end_at.isoformat().replace("+00:00", "Z"), "holdout_status": "LOCKED_NOT_OPENED"}
