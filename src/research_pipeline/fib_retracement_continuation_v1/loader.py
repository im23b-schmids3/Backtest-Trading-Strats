from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .manifests import ManifestError, load_manifest, verify_manifest
from .models import Bar


_SEALED_DEVELOPMENT_CLAIMS = {
    9928: (6576, "2024-12-31T20:00:00+00:00"),
    1654: (1096, "2024-12-31T00:00:00+00:00"),
}


def parquet_schema_sha256(source: Path) -> str:
    """Return the data-contract hash for the physical Arrow schema.

    The audit manifests canonically compact JSON-valued schema metadata (notably
    pandas metadata), whereas PyArrow preserves its original whitespace.
    """
    schema = pq.ParquetFile(source).schema_arrow

    def metadata(values):
        normalized = {}
        for key, value in (values or {}).items():
            text = value.decode("utf-8")
            try:
                text = json.dumps(json.loads(text), separators=(",", ":"))
            except json.JSONDecodeError:
                pass
            normalized[key.decode("utf-8")] = text
        return normalized

    payload = {
        "fields": [
            {
                "metadata": metadata(field.metadata),
                "name": field.name,
                "nullable": field.nullable,
                "type": str(field.type),
            }
            for field in schema
        ],
        "metadata": metadata(schema.metadata),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ManifestError("FIB09_V1_TIMESTAMP_TYPE_INVALID")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ManifestError("FIB09_V1_TIMESTAMP_NOT_UTC")
    return value.astimezone(timezone.utc)


def _development_mask(table, development_start: datetime, development_end: datetime):
    """Private boundary filter; its input table must never leave this module."""
    timestamps = table["timestamp"]
    return pc.and_kleene(
        pc.greater_equal(timestamps, development_start),
        pc.less(timestamps, development_end),
    )


def _validate_development_bars(bars: list[Bar], *, development_start: datetime, development_end: datetime, chronology_claim: dict[str, object], expected_interval: str, source_row_count: int) -> None:
    sealed_claim = _SEALED_DEVELOPMENT_CLAIMS.get(source_row_count)
    if sealed_claim is not None and (int(chronology_claim.get("development_row_count", -1)), str(chronology_claim.get("development_final_timestamp"))) != sealed_claim:
        raise ManifestError("FIB09_V1_SEALED_DEVELOPMENT_CLAIM_MISMATCH")
    if len(bars) != int(chronology_claim.get("development_row_count", -1)):
        raise ManifestError("FIB09_V1_CHRONOLOGY_DEVELOPMENT_COUNT_MISMATCH")
    if not bars or bars[0].timestamp != development_start or bars[-1].timestamp >= development_end:
        raise ManifestError("FIB09_V1_DEVELOPMENT_TIMESTAMP_COVERAGE_MISMATCH")
    if bars[-1].timestamp.isoformat() != str(chronology_claim.get("development_final_timestamp")):
        raise ManifestError("FIB09_V1_CHRONOLOGY_DEVELOPMENT_TIMESTAMP_MISMATCH")
    if expected_interval not in {"PT4H", "P1D"}:
        raise ManifestError("FIB09_V1_CADENCE_INVALID")
    for previous, current in zip(bars, bars[1:]):
        if current.timestamp <= previous.timestamp:
            raise ManifestError("FIB09_V1_NON_INCREASING_OR_DUPLICATE_TIMESTAMP")
    if any(not development_start <= bar.timestamp < development_end for bar in bars):
        raise ManifestError("FIB09_V1_CHRONOLOGY_ISOLATION_FAILED")


def load_development_bars(manifest_path: str | Path, *, development_start: datetime, development_end: datetime, chronology_claim: dict[str, object]) -> tuple[list[Bar], dict[str, object]]:
    """Return development Bars only; physical mixed groups are filtered privately."""
    path, manifest = load_manifest(manifest_path)
    development_start_at = _timestamp(development_start)
    development_end_at = _timestamp(development_end)

    source = Path(manifest["absoluteSourcePath"])
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ["open", "high", "low", "close", "volume", "timestamp"]:
        raise ManifestError("FIB09_V1_PARQUET_SCHEMA_COLUMNS_MISMATCH")
    schema_hash = parquet_schema_sha256(source)
    verified = verify_manifest(path, mode="development", schema_hash=schema_hash)

    physical_row_groups_read = 0
    mixed_row_groups_read = 0
    development_rows_returned = 0
    holdout_rows_discarded = 0
    timestamp_index = parquet.schema_arrow.get_field_index("timestamp")
    bars: list[Bar] = []
    for index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(index)
        stats = row_group.column(timestamp_index).statistics
        if stats is None or not stats.has_min_max:
            raise ManifestError("FIB09_V1_CHRONOLOGY_ISOLATION_UNPROVABLE")
        minimum, maximum = _timestamp(stats.min), _timestamp(stats.max)
        if maximum < development_start_at:
            raise ManifestError("FIB09_V1_DEVELOPMENT_TIMESTAMP_COVERAGE_MISMATCH")
        if minimum >= development_end_at:
            continue
        if minimum < development_start_at:
            raise ManifestError("FIB09_V1_DEVELOPMENT_TIMESTAMP_COVERAGE_MISMATCH")

        # This is the sole physical decode path.  A mixed group remains private
        # and is immediately reduced by the sealed timestamp predicate.
        physical_row_groups_read += 1
        is_mixed = maximum >= development_end_at
        mixed_row_groups_read += int(is_mixed)
        private_table = parquet.read_row_group(index, columns=["open", "high", "low", "close", "volume", "timestamp"])
        development_table = private_table.filter(_development_mask(private_table, development_start_at, development_end_at))
        development_rows_returned += development_table.num_rows
        holdout_rows_discarded += private_table.num_rows - development_table.num_rows
        for row in development_table.to_pylist():
            bars.append(Bar(_timestamp(row["timestamp"]), *(Decimal(str(row[key])) for key in ("open", "high", "low", "close", "volume"))))
    if not bars:
        raise ManifestError("FIB09_V1_DEVELOPMENT_ROWS_MISSING")

    _validate_development_bars(bars, development_start=development_start_at, development_end=development_end_at, chronology_claim=chronology_claim, expected_interval=str(manifest.get("expectedInterval")), source_row_count=int(manifest.get("rowCount", -1)))
    return bars, {
        **verified,
        "development_end": development_end_at.isoformat().replace("+00:00", "Z"),
        "holdout_status": "LOCKED_NOT_OPENED",
        "isolation_counters": {
            "physical_row_groups_read": physical_row_groups_read,
            "mixed_row_groups_read": mixed_row_groups_read,
            "development_rows_returned": development_rows_returned,
            "holdout_rows_discarded": holdout_rows_discarded,
        },
    }
