from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .artifacts import ArtifactContext, canonical_json, sha256_file, sha256_value, write_bytes_once
from .models import (
    DATASET_HASH,
    EVIDENCE_LABEL,
    FOOTPRINT_VERSION,
    SOURCE_MANIFEST_HASH,
)

RAW_COLUMNS = (
    "event_time_utc",
    "aggregate_trade_id",
    "price",
    "quantity_base",
    "notional_quote",
    "buyer_is_maker",
)
REQUIRED_RAW_COLUMNS = {
    "event_time_utc": "string",
    "aggregate_trade_id": "int64",
    "price": "string",
    "quantity_base": "string",
    "notional_quote": "string",
    "buyer_is_maker": "bool",
}


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _manifest_hash(raw: dict[str, Any]) -> str:
    payload = dict(raw)
    payload.pop("manifest_hash", None)
    return sha256_value(payload)


def validate_source_manifest(
    manifest_path: str | Path,
    *,
    require_pinned: bool = True,
    verify_parquet_hashes: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if _manifest_hash(raw) != raw.get("manifest_hash"):
        errors.append("source manifest hash mismatch")
    if require_pinned:
        expected = {
            "normalized_dataset_hash": DATASET_HASH,
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "symbol": "BTCUSDT",
            "date_start": "2024-01-01",
            "date_end": "2024-07-31",
            "row_count": 298_240_261,
        }
        for name, value in expected.items():
            if raw.get(name) != value:
                errors.append(f"pinned manifest {name} mismatch")
    partitions = raw.get("partitions") or []
    if len(partitions) != (7 if require_pinned else len(partitions)) or not partitions:
        errors.append("source manifest partition count is invalid")
    rows = 0
    previous_id: int | None = None
    previous_timestamp: str | None = None
    partition_reports: list[dict[str, Any]] = []
    for partition in partitions:
        parquet = path.parent / str(partition.get("file_name"))
        report = {"month": partition.get("month"), "path": str(parquet), "valid": True}
        try:
            parquet_file = pq.ParquetFile(parquet)
            schema = parquet_file.schema_arrow
            for name, type_name in REQUIRED_RAW_COLUMNS.items():
                if name not in schema.names or str(schema.field(name).type) != type_name:
                    raise ValueError(f"raw column {name} must have type {type_name}")
            actual_rows = parquet_file.metadata.num_rows
            if actual_rows != int(partition["row_count"]):
                raise ValueError("Parquet row count does not match manifest")
            if verify_parquet_hashes and sha256_file(parquet) != partition.get("parquet_hash"):
                raise ValueError("Parquet SHA-256 does not match manifest")
            first_id = int(partition["first_aggregate_trade_id"])
            last_id = int(partition["last_aggregate_trade_id"])
            first_timestamp = str(partition["first_timestamp"])
            last_timestamp = str(partition["last_timestamp"])
            if previous_id is not None and first_id != previous_id + 1:
                raise ValueError("partition aggregate-trade boundary is discontinuous")
            if previous_timestamp is not None and first_timestamp <= previous_timestamp:
                raise ValueError("partition timestamp boundary is not increasing")
            if last_id - first_id + 1 != actual_rows:
                raise ValueError("partition aggregate-trade IDs are not contiguous")
            previous_id, previous_timestamp = last_id, last_timestamp
            rows += actual_rows
            report.update({"row_count": actual_rows, "parquet_sha256": partition.get("parquet_hash")})
        except (KeyError, OSError, ValueError) as exc:
            report.update({"valid": False, "error": str(exc)})
            errors.append(f"{partition.get('month')}: {exc}")
        partition_reports.append(report)
    if rows != int(raw.get("row_count", -1)):
        errors.append("sum of partition rows does not equal manifest row_count")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_path": str(path),
        "manifest_hash": raw.get("manifest_hash"),
        "dataset_hash": raw.get("normalized_dataset_hash"),
        "row_count": rows,
        "partition_count": len(partitions),
        "partitions": partition_reports,
        "manifest": raw,
    }


def _bucket_minutes(raw_timestamp: pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    dates = pc.cast(pc.utf8_slice_codeunits(raw_timestamp, 0, 10), pa.date32())
    days = pc.cast(dates, pa.int32())
    hours = pc.cast(pc.utf8_slice_codeunits(raw_timestamp, 11, 13), pa.int32())
    minutes = pc.cast(pc.utf8_slice_codeunits(raw_timestamp, 14, 16), pa.int32())
    return pc.add(
        pc.add(pc.multiply(days, 1440), pc.multiply(hours, 60)),
        pc.multiply(pc.divide(minutes, 5), 5),
    )


def _utc_from_epoch_minute(value: int) -> datetime:
    return datetime.fromtimestamp(int(value) * 60, tz=timezone.utc)


def _batch_table(batch: pa.RecordBatch) -> pa.Table:
    table = pa.Table.from_batches([batch])
    raw_time = table["event_time_utc"]
    price_float = pc.cast(table["price"], pa.float64())
    return pa.table(
        {
            "bucket": _bucket_minutes(raw_time),
            # One integer bin tick is $0.10. The base footprint uses half-open
            # [floor, floor + $10) bins and can be losslessly coarsened to $20.
            "bin_ticks": pc.cast(pc.multiply(pc.floor(pc.divide(price_float, 10.0)), 100.0), pa.int64()),
            "buyer_is_maker": table["buyer_is_maker"],
            "quantity": pc.cast(table["quantity_base"], pa.decimal128(24, 8)),
            "price": pc.cast(table["price"], pa.decimal128(24, 8)),
            "notional": pc.cast(table["notional_quote"], pa.decimal128(30, 8)),
            "aggregate_trade_id": table["aggregate_trade_id"],
            "raw_price": table["price"],
            "raw_timestamp": raw_time,
        }
    )


def aggregate_trade_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    month: str,
    expected_first_id: int | None = None,
    expected_last_id: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate bounded raw batches while explicitly reconciling boundaries."""

    footprints: dict[tuple[int, int], list[Any]] = {}
    bars: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    previous_id: int | None = None
    previous_timestamp: str | None = None
    row_total = 0
    for batch_number, batch in enumerate(batches):
        if batch.num_rows == 0:
            continue
        table = _batch_table(batch)
        ids = table["aggregate_trade_id"].combine_chunks().to_numpy(zero_copy_only=False)
        if ids.size > 1 and not np.all(np.diff(ids) == 1):
            raise ValueError(f"non-contiguous aggregate-trade IDs within {month} batch {batch_number}")
        raw_times = table["raw_timestamp"].combine_chunks()
        first_id, last_id = int(ids[0]), int(ids[-1])
        first_timestamp, last_timestamp = raw_times[0].as_py(), raw_times[-1].as_py()
        if previous_id is not None and first_id != previous_id + 1:
            raise ValueError(f"aggregate-trade batch boundary gap in {month}")
        if previous_timestamp is not None and first_timestamp < previous_timestamp:
            raise ValueError(f"timestamp regression at batch boundary in {month}")
        if batch_number == 0 and expected_first_id is not None and first_id != expected_first_id:
            raise ValueError(f"first aggregate-trade ID mismatch in {month}")

        grouped = table.group_by(["bucket", "bin_ticks", "buyer_is_maker"]).aggregate(
            [("quantity", "sum"), ("aggregate_trade_id", "count")]
        )
        for row in grouped.to_pylist():
            key = (int(row["bucket"]), int(row["bin_ticks"]))
            current = footprints.setdefault(key, [Decimal(), Decimal(), 0])
            quantity = _decimal(row["quantity_sum"])
            # Binance buyer_is_maker=True means the buyer rested: the
            # aggressive participant was the seller.
            if bool(row["buyer_is_maker"]):
                current[1] += quantity
            else:
                current[0] += quantity
            current[2] += int(row["aggregate_trade_id_count"])

        bar_groups = table.group_by(["bucket"]).aggregate(
            [
                ("price", "min"),
                ("price", "max"),
                ("quantity", "sum"),
                ("notional", "sum"),
                ("aggregate_trade_id", "min"),
                ("aggregate_trade_id", "max"),
                ("aggregate_trade_id", "count"),
            ]
        )
        grouped_by_bucket = {int(row["bucket"]): row for row in bar_groups.to_pylist()}
        bucket_values = table["bucket"].combine_chunks().to_numpy(zero_copy_only=False)
        change_points = np.flatnonzero(np.diff(bucket_values)) + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [len(bucket_values)]))
        raw_prices = table["raw_price"].combine_chunks()
        for start, end in zip(starts, ends, strict=True):
            bucket = int(bucket_values[start])
            row = grouped_by_bucket[bucket]
            incoming = {
                "open": _decimal(raw_prices[int(start)].as_py()),
                "high": _decimal(row["price_max"]),
                "low": _decimal(row["price_min"]),
                "close": _decimal(raw_prices[int(end) - 1].as_py()),
                "volume": _decimal(row["quantity_sum"]),
                "notional": _decimal(row["notional_sum"]),
                "first_aggregate_trade_id": int(row["aggregate_trade_id_min"]),
                "last_aggregate_trade_id": int(row["aggregate_trade_id_max"]),
                "trade_count": int(row["aggregate_trade_id_count"]),
            }
            current = bars.get(bucket)
            if current is None:
                bars[bucket] = incoming
            else:
                if incoming["first_aggregate_trade_id"] != current["last_aggregate_trade_id"] + 1:
                    raise ValueError(f"bucket boundary ID discontinuity in {month}")
                current.update(
                    {
                        "high": max(current["high"], incoming["high"]),
                        "low": min(current["low"], incoming["low"]),
                        "close": incoming["close"],
                        "volume": current["volume"] + incoming["volume"],
                        "notional": current["notional"] + incoming["notional"],
                        "last_aggregate_trade_id": incoming["last_aggregate_trade_id"],
                        "trade_count": current["trade_count"] + incoming["trade_count"],
                    }
                )
        diagnostics.append(
            {
                "month": month,
                "batch_number": batch_number,
                "row_count": batch.num_rows,
                "first_aggregate_trade_id": first_id,
                "last_aggregate_trade_id": last_id,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "first_bucket_epoch_minute": int(bucket_values[0]),
                "last_bucket_epoch_minute": int(bucket_values[-1]),
            }
        )
        previous_id, previous_timestamp = last_id, last_timestamp
        row_total += batch.num_rows
    if expected_last_id is not None and previous_id != expected_last_id:
        raise ValueError(f"last aggregate-trade ID mismatch in {month}")

    footprint_rows: list[dict[str, Any]] = []
    bar_deltas: dict[int, Decimal] = defaultdict(Decimal)
    for (bucket, bin_ticks), (buy, sell, count) in sorted(footprints.items()):
        total, delta = buy + sell, buy - sell
        if total != buy + sell or delta != buy - sell:
            raise AssertionError("footprint arithmetic failed")
        bar_deltas[bucket] += delta
        start = _utc_from_epoch_minute(bucket)
        floor = Decimal(bin_ticks) / Decimal("10")
        footprint_rows.append(
            {
                "bar_start_utc": start,
                "bar_end_utc": start + timedelta(minutes=5),
                "month": month,
                "bin_floor": floor,
                "bin_upper_exclusive": floor + Decimal("10"),
                "buy_volume_btc": buy,
                "sell_volume_btc": sell,
                "total_volume_btc": total,
                "delta_btc": delta,
                "trade_count": count,
            }
        )

    bar_rows: list[dict[str, Any]] = []
    cvd = Decimal()
    current_day = None
    cumulative_volume = cumulative_notional = Decimal()
    for bucket, row in sorted(bars.items()):
        start = _utc_from_epoch_minute(bucket)
        day = start.date().isoformat()
        if day != current_day:
            current_day = day
            cumulative_volume = cumulative_notional = Decimal()
        delta = bar_deltas[bucket]
        cvd += delta
        cumulative_volume += row["volume"]
        cumulative_notional += row["notional"]
        daily_vwap = cumulative_notional / cumulative_volume if cumulative_volume else row["close"]
        bar_rows.append(
            {
                "bar_start_utc": start,
                "bar_end_utc": start + timedelta(minutes=5),
                "session_date": day,
                "month": month,
                **row,
                "delta_btc": delta,
                "cumulative_volume_delta_btc": cvd,
                "daily_vwap": daily_vwap,
            }
        )
    if sum(item["trade_count"] for item in bar_rows) != row_total:
        raise AssertionError("bar trade counts do not reconcile to streamed rows")
    if sum(item["trade_count"] for item in footprint_rows) != row_total:
        raise AssertionError("footprint trade counts do not reconcile to streamed rows")
    return footprint_rows, bar_rows, diagnostics


def _footprint_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "builder_version": FOOTPRINT_VERSION,
        "dataset_hash": source["normalized_dataset_hash"],
        "source_manifest_hash": source["manifest_hash"],
        "base_bin_size_usd": "10",
        "bar_interval": "5m",
        "aggressor_rule": "BUY_IFF_BUYER_IS_MAKER_FALSE",
        "bin_interval": "HALF_OPEN",
    }


def build_footprint_dataset(
    manifest_path: str | Path,
    cache_root: str | Path,
    *,
    batch_size: int = 1_000_000,
    require_pinned: bool = True,
    verify_source_hashes: bool = True,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    validation = validate_source_manifest(
        manifest_path,
        require_pinned=require_pinned,
        verify_parquet_hashes=verify_source_hashes,
    )
    if not validation["valid"]:
        raise ValueError("invalid immutable source manifest: " + "; ".join(validation["errors"]))
    source = validation["manifest"]
    identity = _footprint_identity(source)
    footprint_id = sha256_value(identity)
    root = Path(cache_root).resolve() / "BTCUSDT" / footprint_id
    manifest_output = root / "manifest.json"
    if manifest_output.exists() and not force_rebuild:
        existing = json.loads(manifest_output.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"immutable footprint manifest collision: {manifest_output}")
        for item in existing.get("parquet_files", []):
            target = root / item["relative_path"]
            if not target.is_file() or sha256_file(target) != item["sha256"]:
                raise ValueError(f"immutable footprint Parquet collision: {target}")
        return existing

    parameter_hash = sha256_value({"base_bin_size_usd": "10", "bar_interval": "5m"})
    context = ArtifactContext(
        run_id=footprint_id,
        dataset_hash=source["normalized_dataset_hash"],
        source_manifest_hash=source["manifest_hash"],
        specification_hash=sha256_value(identity),
        parameter_hash=parameter_hash,
        code_hash=sha256_value({"builder_version": FOOTPRINT_VERSION}),
        evidence_label=EVIDENCE_LABEL,
        timestamp=str(source.get("retrieved_at")),
    )
    parquet_files: list[dict[str, Any]] = []
    all_batches: list[dict[str, Any]] = []
    previous_last_id: int | None = None
    previous_last_timestamp: str | None = None
    total_footprints = total_bars = total_trades = 0
    cvd_carry = Decimal()
    for partition in source["partitions"]:
        if previous_last_id is not None and int(partition["first_aggregate_trade_id"]) != previous_last_id + 1:
            raise ValueError("source month aggregate-trade boundary is discontinuous")
        if previous_last_timestamp is not None and str(partition["first_timestamp"]) <= previous_last_timestamp:
            raise ValueError("source month timestamp boundary is not increasing")
        parquet = Path(manifest_path).resolve().parent / partition["file_name"]
        parquet_file = pq.ParquetFile(parquet)
        batches = parquet_file.iter_batches(batch_size=batch_size, columns=list(RAW_COLUMNS), use_threads=True)
        footprint_rows, bar_rows, batch_diagnostics = aggregate_trade_batches(
            batches,
            month=partition["month"],
            expected_first_id=int(partition["first_aggregate_trade_id"]),
            expected_last_id=int(partition["last_aggregate_trade_id"]),
        )
        for bar in bar_rows:
            bar["cumulative_volume_delta_btc"] += cvd_carry
        if bar_rows:
            cvd_carry = bar_rows[-1]["cumulative_volume_delta_btc"]
        footprint_path = root / "footprints" / f"{partition['month']}.parquet"
        bars_path = root / "bars" / f"{partition['month']}.parquet"
        context.write_parquet(footprint_path, footprint_rows)
        context.write_parquet(bars_path, bar_rows)
        for kind, target, count in (
            ("footprints", footprint_path, len(footprint_rows)),
            ("bars", bars_path, len(bar_rows)),
        ):
            parquet_files.append(
                {
                    "kind": kind,
                    "month": partition["month"],
                    "relative_path": target.relative_to(root).as_posix(),
                    "row_count": count,
                    "sha256": sha256_file(target),
                    "schema": str(pq.ParquetFile(target).schema_arrow),
                }
            )
        all_batches.extend(batch_diagnostics)
        total_footprints += len(footprint_rows)
        total_bars += len(bar_rows)
        total_trades += sum(row["trade_count"] for row in bar_rows)
        previous_last_id = int(partition["last_aggregate_trade_id"])
        previous_last_timestamp = str(partition["last_timestamp"])
    if total_trades != int(source["row_count"]):
        raise AssertionError("footprint dataset does not reconcile to source trade count")
    content_hash = sha256_value({item["relative_path"]: item["sha256"] for item in parquet_files})
    output = context.envelope(
        {
            "identity": identity,
            "footprint_dataset_hash": content_hash,
            "footprint_root": str(root),
            "source_manifest_path": str(Path(manifest_path).resolve()),
            "source_row_count": int(source["row_count"]),
            "streamed_trade_count": total_trades,
            "footprint_row_count": total_footprints,
            "five_minute_bar_count": total_bars,
            "batch_size": batch_size,
            "batch_count": len(all_batches),
            "batch_boundaries": all_batches,
            "parquet_files": parquet_files,
            "valid": True,
        }
    )
    content = json.dumps(output, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"
    write_bytes_once(manifest_output, content)
    return output


def validate_footprint_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest_path = root if root.name == "manifest.json" else root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    errors: list[str] = []
    rows: dict[str, int] = defaultdict(int)
    hashes: dict[str, str] = {}
    for item in manifest.get("parquet_files", []):
        target = root / item["relative_path"]
        try:
            actual_hash = sha256_file(target)
            actual_rows = pq.ParquetFile(target).metadata.num_rows
            if actual_hash != item["sha256"]:
                raise ValueError("SHA-256 mismatch")
            if actual_rows != item["row_count"]:
                raise ValueError("row-count mismatch")
            rows[item["kind"]] += actual_rows
            hashes[item["relative_path"]] = actual_hash
        except (OSError, ValueError) as exc:
            errors.append(f"{target}: {exc}")
    if rows["footprints"] != int(manifest.get("footprint_row_count", -1)):
        errors.append("footprint row total mismatch")
    if rows["bars"] != int(manifest.get("five_minute_bar_count", -1)):
        errors.append("bar row total mismatch")
    if sha256_value(hashes) != manifest.get("footprint_dataset_hash"):
        errors.append("content-addressed footprint dataset hash mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_path": str(manifest_path),
        "footprint_dataset_hash": manifest.get("footprint_dataset_hash"),
        "footprint_row_count": rows["footprints"],
        "five_minute_bar_count": rows["bars"],
        "parquet_sha256": hashes,
    }


def load_footprint_dataset(
    manifest: dict[str, Any] | str | Path,
    *,
    months: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
    else:
        raw = manifest
        root = Path(raw["footprint_root"])
    footprint_rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    for item in raw["parquet_files"]:
        if months is not None and item["month"] not in months:
            continue
        rows = pq.read_table(root / item["relative_path"]).to_pylist()
        if item["kind"] == "footprints":
            footprint_rows.extend(rows)
        elif item["kind"] == "bars":
            bar_rows.extend(rows)
    return footprint_rows, bar_rows
