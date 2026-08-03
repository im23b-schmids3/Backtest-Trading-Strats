from __future__ import annotations

import json
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ..value_area_trap.data import (
    ARCHIVE_COLUMN_ORDER,
    FIELD_ALIASES,
    PARQUET_SCHEMA,
    REQUIRED_NORMALIZED_FIELDS,
    AggregateTradeImporter,
    AggregateTradePartition,
    MonthlyAggregateTradeManifest,
    _monthly_manifest_hash,
)
from .artifacts import ArtifactContext, sha256_file, sha256_value, write_bytes_once
from .footprint import RAW_COLUMNS, _batch_table, _decimal, _utc_from_epoch_minute
from .v3_models import AUTHORIZED_MONTHS, EVIDENCE_LABEL

OFFICIAL_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
FOOTPRINT_BUILDER_VERSION = "imbalance-vwap-ride-v3-exact-footprints-1"
V3_NORMALIZER_VERSION = "imbalance-vwap-ride-v3-vectorized-normalizer-1"
EXACT_BIN_SIZES = (Decimal("30"), Decimal("50"), Decimal("75"))


def authorized_archive_url(month: str) -> str:
    if month not in AUTHORIZED_MONTHS:
        raise ValueError(f"month is outside the sealed V3 download allowlist: {month}")
    url = f"{OFFICIAL_ARCHIVE_ROOT}/BTCUSDT/BTCUSDT-aggTrades-{month}.zip"
    parsed = urlparse(url)
    expected_path = f"/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{month}.zip"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "data.binance.vision"
        or parsed.port is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.path != expected_path
    ):
        raise ValueError(f"invalid Binance V3 archive URL: {url}")
    return url


def validate_authorized_archive(path: str | Path, month: str) -> dict[str, Any]:
    archive = Path(path).resolve()
    expected_name = f"BTCUSDT-aggTrades-{month}.zip"
    expected_member = f"BTCUSDT-aggTrades-{month}.csv"
    if month not in AUTHORIZED_MONTHS or archive.name != expected_name:
        raise ValueError(f"archive is outside the sealed V3 allowlist: {archive}")
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [item for item in bundle.infolist() if not item.is_dir()]
            if len(members) != 1 or members[0].filename != expected_member:
                raise ValueError(
                    f"archive {archive.name} must contain exactly {expected_member}"
                )
            if Path(members[0].filename).name != members[0].filename:
                raise ValueError(f"archive member path is not flat: {members[0].filename}")
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise ValueError(f"archive CRC validation failed for {bad_member}")
            member_size = members[0].file_size
            member_crc = f"{members[0].CRC:08x}"
    except zipfile.BadZipFile as exc:
        raise ValueError(f"corrupt Binance archive: {archive}") from exc
    return {
        "month": month,
        "url": authorized_archive_url(month),
        "archive_path": str(archive),
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "csv_member": expected_member,
        "csv_uncompressed_size_bytes": member_size,
        "csv_crc32": member_crc,
        "zip_integrity_valid": True,
    }


def validate_v3_source_manifest(
    manifest_path: str | Path,
    *,
    verify_archives: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    importer = AggregateTradeImporter(path.parents[2])
    errors: list[str] = []
    archives: list[dict[str, Any]] = []
    try:
        manifest = importer.validate_monthly_manifest(path)
    except (OSError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)], "manifest_path": str(path)}
    months = tuple(item.month for item in manifest.partitions)
    if months != AUTHORIZED_MONTHS:
        errors.append(f"manifest months must equal the exact sealed V3 range: {months}")
    if manifest.symbol != "BTCUSDT":
        errors.append("manifest symbol must be BTCUSDT")
    if manifest.date_start != date(2024, 8, 1) or manifest.date_end != date(2025, 1, 31):
        errors.append("manifest date bounds must be 2024-08-01 through 2025-01-31")
    if manifest.duplicate_count != 0:
        errors.append("normalized V3 dataset contains duplicate aggregate-trade IDs")
    try:
        calendar_diagnostics = importer.validate_complete_calendar_months(
            manifest,
            start_month=AUTHORIZED_MONTHS[0],
            end_month=AUTHORIZED_MONTHS[-1],
        )
    except ValueError as exc:
        errors.append(str(exc))
        calendar_diagnostics = []

    previous = None
    rows = 0
    for partition in manifest.partitions:
        if partition.duplicate_count != 0:
            errors.append(f"{partition.month}: duplicate aggregate-trade rows are forbidden")
        if partition.last_aggregate_trade_id - partition.first_aggregate_trade_id + 1 != partition.row_count:
            errors.append(f"{partition.month}: aggregate-trade IDs are not contiguous")
        if previous is not None:
            if partition.first_aggregate_trade_id != previous.last_aggregate_trade_id + 1:
                errors.append(f"{previous.month}->{partition.month}: aggregate-trade ID gap or overlap")
            if partition.first_timestamp <= previous.last_timestamp:
                errors.append(f"{previous.month}->{partition.month}: timestamp overlap or regression")
        previous = partition
        rows += partition.row_count
        source = Path(partition.source_archive).resolve()
        expected_url = authorized_archive_url(partition.month)
        if AggregateTradeImporter.archive_url("BTCUSDT", partition.month) != expected_url:
            errors.append(f"{partition.month}: importer URL differs from sealed official URL")
        if verify_archives:
            try:
                report = validate_authorized_archive(source, partition.month)
                if report["archive_sha256"] != partition.source_archive_hash:
                    raise ValueError("archive hash differs from normalized partition provenance")
                archives.append(report)
            except (OSError, ValueError) as exc:
                errors.append(f"{partition.month}: {exc}")
    if rows != manifest.row_count:
        errors.append("partition row total does not equal combined manifest row_count")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_path": str(path),
        "manifest_file_sha256": sha256_file(path),
        "manifest_hash": manifest.manifest_hash,
        "normalized_dataset_hash": manifest.normalized_dataset_hash,
        "symbol": manifest.symbol,
        "months": list(months),
        "date_start": manifest.date_start.isoformat(),
        "date_end": manifest.date_end.isoformat(),
        "row_count": manifest.row_count,
        "duplicate_count": manifest.duplicate_count,
        "calendar_diagnostics": calendar_diagnostics,
        "archives": archives,
        "partitions": [item.model_dump(mode="json") for item in manifest.partitions],
        "manifest": manifest,
    }


def acquire_and_normalize_v3_data(
    cache_root: str | Path,
    *,
    allow_authorized_downloads: bool,
) -> tuple[Path, MonthlyAggregateTradeManifest, dict[str, Any]]:
    """Acquire only the six sealed archives and build one local content-addressed dataset."""

    root = Path(cache_root).resolve()
    importer = AggregateTradeImporter(root)
    actions: list[dict[str, Any]] = []
    for month in AUTHORIZED_MONTHS:
        url = authorized_archive_url(month)
        if importer.archive_url("BTCUSDT", month) != url:
            raise RuntimeError("Binance importer URL failed the sealed V3 allowlist")
        expected = root / "downloads" / "BTCUSDT" / f"BTCUSDT-aggTrades-{month}.zip"
        existed = expected.is_file()
        archive = importer.download_month(
            "BTCUSDT",
            month,
            allow_network=allow_authorized_downloads,
        )
        report = validate_authorized_archive(archive, month)
        report["action"] = "REUSED_HASH_AND_CRC_VALID_LOCAL_ARCHIVE" if existed else "DOWNLOADED_AUTHORIZED_ARCHIVE"
        actions.append(report)

    manifest_path, manifest, normalization_actions = _ingest_v3_months_vectorized(root, importer)
    validation = validate_v3_source_manifest(manifest_path, verify_archives=False)
    if not validation["valid"]:
        raise ValueError("V3 source validation failed: " + "; ".join(validation["errors"]))
    action_by_month = {item["month"]: item for item in actions}
    normalization = [
        {**item, "archive_validation": action_by_month[item["month"]]}
        for item in normalization_actions
    ]
    download_manifest = {
        "provider": "Binance USD-M Futures public data archive",
        "official_origin": "https://data.binance.vision",
        "symbol": "BTCUSDT",
        "authorized_months": list(AUTHORIZED_MONTHS),
        "network_scope": "ONLY_THE_SIX_LISTED_MONTHLY_ARCHIVES",
        "network_request_count": sum(item["action"] == "DOWNLOADED_AUTHORIZED_ARCHIVE" for item in actions),
        "archive_count": len(actions),
        "archives": actions,
        "normalization_actions": normalization,
        "normalized_manifest_path": str(manifest_path),
        "normalized_manifest_file_sha256": sha256_file(manifest_path),
        "normalized_manifest_hash": manifest.manifest_hash,
        "normalized_dataset_hash": manifest.normalized_dataset_hash,
        "raw_rows_transmitted_externally": False,
        "unrelated_assets_downloaded": False,
    }
    if len(actions) != 6:
        raise AssertionError("sealed V3 acquisition did not resolve exactly six archives")
    return manifest_path, manifest, download_manifest


def _archive_layout(archive: Path, month: str) -> tuple[str, list[str], bool]:
    expected_member = f"BTCUSDT-aggTrades-{month}.csv"
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(expected_member) as raw:
            first = raw.readline().decode("utf-8-sig").strip().split(",")
    detected = [item.strip().lstrip("\ufeff") for item in first]
    recognized = [FIELD_ALIASES.get(item) for item in detected]
    has_header = any(item is not None for item in recognized)
    if has_header:
        normalized = {FIELD_ALIASES[item] for item in detected if item in FIELD_ALIASES}
        missing = sorted(REQUIRED_NORMALIZED_FIELDS - normalized)
        if missing:
            raise ValueError(f"{archive.name}: header is missing normalized fields {missing}")
        headers = detected
    else:
        if len(first) not in {7, 8}:
            raise ValueError(f"{archive.name}: unsupported headerless row width {len(first)}")
        headers = list(ARCHIVE_COLUMN_ORDER[: len(first)])
    return expected_member, headers, has_header


def _canonical_header_map(headers: list[str]) -> dict[str, str]:
    mapping = {FIELD_ALIASES.get(name, name): name for name in headers}
    missing = REQUIRED_NORMALIZED_FIELDS - set(mapping)
    if missing:
        raise ValueError(f"aggregate-trade CSV schema is missing {sorted(missing)}")
    return mapping


def _vectorized_month_partition(
    root: Path,
    importer: AggregateTradeImporter,
    month: str,
    archive: Path,
) -> tuple[AggregateTradePartition, dict[str, Any]]:
    reusable = importer._load_reusable_partition("BTCUSDT", month, archive)
    if reusable is not None:
        return reusable, {
            "month": month,
            "action": "SKIPPED_HASH_VERIFIED",
            "reused": True,
            "repair_status": reusable.repair_status,
            "archive": str(archive),
            "normalizer_version": "EXISTING_VALID_PARTITION",
        }
    member, headers, has_header = _archive_layout(archive, month)
    mapping = _canonical_header_map(headers)
    column_types: dict[str, pa.DataType] = {}
    for canonical, raw_name in mapping.items():
        if canonical in {"aggregate_trade_id", "first_trade_id", "last_trade_id", "trade_time", "event_time"}:
            column_types[raw_name] = pa.int64()
        elif canonical == "buyer_is_maker" or canonical == "is_best_match":
            column_types[raw_name] = pa.bool_()
        else:
            column_types[raw_name] = pa.string()
    read_options = pacsv.ReadOptions(
        block_size=32 * 1024 * 1024,
        column_names=None if has_header else headers,
        skip_rows=0,
        use_threads=True,
    )
    convert_options = pacsv.ConvertOptions(column_types=column_types, strings_can_be_null=False)
    archive_hash = sha256_file(archive)
    staging_root = root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"v3-{month}-{archive_hash[:16]}.parquet.part"
    writer = pq.ParquetWriter(staging, PARQUET_SCHEMA, compression="zstd")
    previous_id: int | None = None
    previous_timestamp_ms: int | None = None
    first_id = last_id = None
    first_timestamp_ms = last_timestamp_ms = None
    row_count = 0
    timestamp_gaps: list[dict[str, Any]] = []
    expected_year, expected_month = (int(value) for value in month.split("-", 1))
    source_file = f"{archive.name}!{member}"
    try:
        with zipfile.ZipFile(archive) as bundle, bundle.open(member) as raw:
            reader = pacsv.open_csv(
                raw,
                read_options=read_options,
                convert_options=convert_options,
            )
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                table = pa.Table.from_batches([batch])
                ids_array = pc.cast(table[mapping["aggregate_trade_id"]], pa.int64()).combine_chunks()
                ids = ids_array.to_numpy(zero_copy_only=False)
                if ids.size > 1 and not np.all(np.diff(ids) == 1):
                    raise ValueError(f"{month}: duplicate, gap, or unordered aggregate-trade ID")
                batch_first_id, batch_last_id = int(ids[0]), int(ids[-1])
                if previous_id is not None and batch_first_id != previous_id + 1:
                    raise ValueError(f"{month}: aggregate-trade batch boundary gap or overlap")
                raw_times_array = pc.cast(table[mapping["trade_time"]], pa.int64()).combine_chunks()
                raw_times = raw_times_array.to_numpy(zero_copy_only=False)
                microseconds = int(raw_times[0]) > 10_000_000_000_000
                timestamp_ms = raw_times // (1000 if microseconds else 1)
                if timestamp_ms.size > 1 and np.any(np.diff(timestamp_ms) < 0):
                    raise ValueError(f"{month}: aggregate-trade timestamps regress")
                batch_first_ms, batch_last_ms = int(timestamp_ms[0]), int(timestamp_ms[-1])
                if previous_timestamp_ms is not None and batch_first_ms < previous_timestamp_ms:
                    raise ValueError(f"{month}: timestamp batch boundary regression")
                first_dt = datetime.fromtimestamp(batch_first_ms / 1000, tz=timezone.utc)
                last_dt = datetime.fromtimestamp(batch_last_ms / 1000, tz=timezone.utc)
                if (first_dt.year, first_dt.month) != (expected_year, expected_month) or (
                    last_dt.year,
                    last_dt.month,
                ) != (expected_year, expected_month):
                    raise ValueError(f"{month}: timestamp is outside the requested calendar month")
                remaining = 1000 - len(timestamp_gaps)
                if remaining > 0:
                    for position in np.flatnonzero(np.diff(timestamp_ms) > 300_000)[:remaining]:
                        before_ms = int(timestamp_ms[position])
                        after_ms = int(timestamp_ms[position + 1])
                        timestamp_gaps.append(
                            {
                                "start_utc": datetime.fromtimestamp(before_ms / 1000, tz=timezone.utc).isoformat(),
                                "end_utc": datetime.fromtimestamp(after_ms / 1000, tz=timezone.utc).isoformat(),
                                "gap_seconds": (after_ms - before_ms) / 1000,
                                "previous_aggregate_trade_id": int(ids[position]),
                                "next_aggregate_trade_id": int(ids[position + 1]),
                                "missing_id_count": 0,
                                "ids_continuous": True,
                                "repair_status": "CONTINUOUS_IDS_NO_TRADE_INTERVAL",
                            }
                        )

                price = pc.cast(table[mapping["price"]], pa.string()).combine_chunks()
                quantity = pc.cast(table[mapping["quantity_base"]], pa.string()).combine_chunks()
                price_decimal = pc.cast(price, pa.decimal128(18, 8))
                quantity_decimal = pc.cast(quantity, pa.decimal128(18, 8))
                notional = pc.cast(
                    pc.cast(
                        pc.multiply(price_decimal, quantity_decimal),
                        pa.decimal128(30, 8),
                        safe=False,
                    ),
                    pa.string(),
                )
                maker = pc.cast(table[mapping["buyer_is_maker"]], pa.bool_()).combine_chunks()
                timestamp_strings = pc.cast(
                    pc.cast(pa.array(timestamp_ms, type=pa.int64()), pa.timestamp("ms", tz="UTC")),
                    pa.string(),
                )
                signed = pc.cast(
                    pc.if_else(maker, pc.negate(quantity_decimal), quantity_decimal),
                    pa.string(),
                )
                count = batch.num_rows
                first_trade = (
                    pc.cast(table[mapping["first_trade_id"]], pa.int64()).combine_chunks()
                    if "first_trade_id" in mapping
                    else pa.nulls(count, type=pa.int64())
                )
                last_trade = (
                    pc.cast(table[mapping["last_trade_id"]], pa.int64()).combine_chunks()
                    if "last_trade_id" in mapping
                    else pa.nulls(count, type=pa.int64())
                )
                normalized = pa.Table.from_arrays(
                    [
                        timestamp_strings,
                        timestamp_strings,
                        ids_array,
                        first_trade,
                        last_trade,
                        price,
                        quantity,
                        notional,
                        maker,
                        pc.if_else(maker, pa.scalar("SELL"), pa.scalar("BUY")),
                        signed,
                        pa.array(["binance_usdm_public_archive"] * count),
                        pa.array([source_file] * count),
                        pa.array([archive_hash] * count),
                    ],
                    schema=PARQUET_SCHEMA,
                )
                writer.write_table(normalized)
                if first_id is None:
                    first_id = batch_first_id
                    first_timestamp_ms = batch_first_ms
                last_id = batch_last_id
                last_timestamp_ms = batch_last_ms
                previous_id = batch_last_id
                previous_timestamp_ms = batch_last_ms
                row_count += count
    finally:
        writer.close()
    if first_id is None or last_id is None or first_timestamp_ms is None or last_timestamp_ms is None:
        raise ValueError(f"{month}: aggregate-trade archive is empty")
    if last_id - first_id + 1 != row_count:
        raise ValueError(f"{month}: normalized row count and aggregate-trade ID span differ")
    parquet_hash = sha256_file(staging)
    dataset_hash = sha256_value(
        {
            "normalizer_version": V3_NORMALIZER_VERSION,
            "month": month,
            "archive_sha256": archive_hash,
            "parquet_sha256": parquet_hash,
            "row_count": row_count,
            "first_aggregate_trade_id": first_id,
            "last_aggregate_trade_id": last_id,
        }
    )
    partition_dir = root / "normalized" / "BTCUSDT" / "monthly_partitions" / month
    partition_dir.mkdir(parents=True, exist_ok=True)
    destination = partition_dir / f"{dataset_hash}.parquet"
    if destination.exists():
        if sha256_file(destination) != parquet_hash:
            raise ValueError(f"immutable V3 normalized partition collision: {destination}")
        staging.unlink(missing_ok=True)
    else:
        staging.replace(destination)
    first_timestamp = datetime.fromtimestamp(first_timestamp_ms / 1000, tz=timezone.utc)
    last_timestamp = datetime.fromtimestamp(last_timestamp_ms / 1000, tz=timezone.utc)
    repair_status = "ARCHIVE_CONTINUOUS_ID_TIMESTAMP_GAP" if timestamp_gaps else "ARCHIVE_ONLY"
    partition = AggregateTradePartition(
        month=month,
        file_name=destination.name,
        parquet_hash=parquet_hash,
        normalized_dataset_hash=dataset_hash,
        source_archive=str(archive.resolve()),
        source_archive_hash=archive_hash,
        row_count=row_count,
        duplicate_count=0,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        first_aggregate_trade_id=first_id,
        last_aggregate_trade_id=last_id,
        missing_interval_diagnostics=timestamp_gaps,
        continuity_diagnostics=timestamp_gaps,
        repair_status=repair_status,
    )
    index = partition_dir / "partition.json"
    write_bytes_once(
        index,
        json.dumps(partition.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8"),
    )
    return partition, {
        "month": month,
        "action": "NEWLY_PROCESSED_VECTORIZED",
        "reused": False,
        "repair_status": repair_status,
        "archive": str(archive),
        "normalizer_version": V3_NORMALIZER_VERSION,
        "row_count": row_count,
        "parquet_sha256": parquet_hash,
        "continuity_diagnostics": timestamp_gaps,
    }


def _ingest_v3_months_vectorized(
    root: Path,
    importer: AggregateTradeImporter,
) -> tuple[Path, MonthlyAggregateTradeManifest, list[dict[str, Any]]]:
    def process(month: str) -> tuple[AggregateTradePartition, dict[str, Any]]:
        archive = root / "downloads" / "BTCUSDT" / f"BTCUSDT-aggTrades-{month}.zip"
        return _vectorized_month_partition(root, importer, month, archive)

    # Months are independent immutable partitions. Three workers keep the
    # local Arrow/ZIP pipeline bounded while using available CPU and disk I/O.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="v3-normalize") as executor:
        results = dict(zip(AUTHORIZED_MONTHS, executor.map(process, AUTHORIZED_MONTHS), strict=True))
    partitions = [results[month][0] for month in AUTHORIZED_MONTHS]
    actions = [results[month][1] for month in AUTHORIZED_MONTHS]
    previous = None
    for partition in partitions:
        if previous is not None:
            if partition.first_aggregate_trade_id != previous.last_aggregate_trade_id + 1:
                raise ValueError(f"{previous.month}->{partition.month}: inter-month ID gap or overlap")
            if partition.first_timestamp <= previous.last_timestamp:
                raise ValueError(f"{previous.month}->{partition.month}: inter-month timestamp overlap")
        previous = partition
    identity = {
        "symbol": "BTCUSDT",
        "months": [
            {
                "month": item.month,
                "normalized_dataset_hash": item.normalized_dataset_hash,
                "parquet_hash": item.parquet_hash,
            }
            for item in partitions
        ],
    }
    dataset_hash = sha256_value(identity)
    combined_root = root / "normalized" / "BTCUSDT" / dataset_hash
    combined_root.mkdir(parents=True, exist_ok=True)
    combined: list[AggregateTradePartition] = []
    for item in partitions:
        source = root / "normalized" / "BTCUSDT" / "monthly_partitions" / item.month / item.file_name
        target = combined_root / f"{item.month}.parquet"
        if target.exists():
            if sha256_file(target) != item.parquet_hash:
                raise ValueError(f"immutable V3 combined partition collision: {target}")
        else:
            shutil.copyfile(source, target)
        combined.append(item.model_copy(update={"file_name": target.name}))
    manifest_path = combined_root / "manifest.json"
    if manifest_path.exists():
        existing = importer.validate_monthly_manifest(manifest_path)
        if existing.normalized_dataset_hash != dataset_hash:
            raise ValueError(f"immutable V3 combined manifest collision: {manifest_path}")
        return manifest_path, existing, actions
    unsigned = MonthlyAggregateTradeManifest(
        symbol="BTCUSDT",
        date_start=combined[0].first_timestamp.date(),
        date_end=combined[-1].last_timestamp.date(),
        retrieved_at=datetime.now(timezone.utc),
        source_files=[item.source_archive for item in combined],
        source_file_hashes={item.source_archive: item.source_archive_hash for item in combined},
        normalized_dataset_hash=dataset_hash,
        row_count=sum(item.row_count for item in combined),
        duplicate_count=0,
        partitions=combined,
        manifest_hash="pending",
    )
    manifest = unsigned.model_copy(update={"manifest_hash": _monthly_manifest_hash(unsigned)})
    write_bytes_once(
        manifest_path,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8"),
    )
    importer.validate_monthly_manifest(manifest_path)
    return manifest_path, manifest, actions


def _aggregate_month(
    parquet_file: pq.ParquetFile,
    *,
    month: str,
    expected_first_id: int,
    expected_last_id: int,
    batch_size: int,
) -> tuple[dict[Decimal, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    footprints: dict[Decimal, dict[tuple[int, int], list[Any]]] = {
        size: {} for size in EXACT_BIN_SIZES
    }
    bars: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    previous_id: int | None = None
    previous_timestamp_us: int | None = None
    row_total = 0
    batches = parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(RAW_COLUMNS),
        use_threads=True,
    )
    for batch_number, batch in enumerate(batches):
        if batch.num_rows == 0:
            continue
        table = _batch_table(batch)
        ids = table["aggregate_trade_id"].combine_chunks().to_numpy(zero_copy_only=False)
        if ids.size > 1 and not np.all(np.diff(ids) == 1):
            raise ValueError(f"non-contiguous aggregate-trade IDs in {month} batch {batch_number}")
        times = table["raw_timestamp"].combine_chunks()
        # Existing hash-verified normalized partitions may use the equivalent
        # ISO-8601 spellings ``.123+00:00`` and ``.123000+00:00``.  Lexical
        # comparison incorrectly reports a regression across those spellings,
        # so chronology is validated on parsed UTC microseconds instead.
        timestamp_values = pc.cast(
            pc.cast(times, pa.timestamp("us", tz="UTC")),
            pa.int64(),
        ).to_numpy(zero_copy_only=False)
        if timestamp_values.size > 1 and np.any(np.diff(timestamp_values) < 0):
            raise ValueError(f"timestamp regression in {month} batch {batch_number}")
        first_id, last_id = int(ids[0]), int(ids[-1])
        first_timestamp_us, last_timestamp_us = int(timestamp_values[0]), int(timestamp_values[-1])
        first_timestamp, last_timestamp = str(times[0].as_py()), str(times[-1].as_py())
        if previous_id is not None and first_id != previous_id + 1:
            raise ValueError(f"aggregate-trade batch gap in {month}")
        if previous_timestamp_us is not None and first_timestamp_us < previous_timestamp_us:
            raise ValueError(f"timestamp batch regression in {month}")
        if batch_number == 0 and first_id != expected_first_id:
            raise ValueError(f"first aggregate-trade ID mismatch in {month}")

        price_float = pc.cast(table["price"], pa.float64())
        for size in EXACT_BIN_SIZES:
            size_int = int(size)
            bin_floor = pc.cast(
                pc.multiply(pc.floor(pc.divide(price_float, float(size_int))), float(size_int)),
                pa.int64(),
            )
            grouped = pa.table(
                {
                    "bucket": table["bucket"],
                    "bin_floor": bin_floor,
                    "buyer_is_maker": table["buyer_is_maker"],
                    "quantity": table["quantity"],
                    "aggregate_trade_id": table["aggregate_trade_id"],
                }
            ).group_by(["bucket", "bin_floor", "buyer_is_maker"]).aggregate(
                [("quantity", "sum"), ("aggregate_trade_id", "count")]
            )
            target = footprints[size]
            for row in grouped.to_pylist():
                key = (int(row["bucket"]), int(row["bin_floor"]))
                current = target.setdefault(key, [Decimal(), Decimal(), 0])
                quantity = _decimal(row["quantity_sum"])
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
                    raise ValueError(f"bar boundary aggregate-trade gap in {month}")
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
            }
        )
        previous_id, previous_timestamp_us = last_id, last_timestamp_us
        row_total += batch.num_rows
    if previous_id != expected_last_id or row_total != expected_last_id - expected_first_id + 1:
        raise ValueError(f"aggregate-trade endpoint or row-count mismatch in {month}")

    footprint_rows: dict[Decimal, list[dict[str, Any]]] = {}
    bar_deltas: dict[int, Decimal] = defaultdict(Decimal)
    for size, grouped in footprints.items():
        output: list[dict[str, Any]] = []
        for (bucket, floor_int), (buy, sell, count) in sorted(grouped.items()):
            start = _utc_from_epoch_minute(bucket)
            floor = Decimal(floor_int)
            output.append(
                {
                    "bar_start_utc": start,
                    "bar_end_utc": start + timedelta(minutes=5),
                    "month": month,
                    "bin_size_usd": size,
                    "bin_floor": floor,
                    "bin_upper_exclusive": floor + size,
                    "buy_volume_btc": buy,
                    "sell_volume_btc": sell,
                    "total_volume_btc": buy + sell,
                    "delta_btc": buy - sell,
                    "trade_count": count,
                }
            )
            if size == EXACT_BIN_SIZES[0]:
                bar_deltas[bucket] += buy - sell
        if sum(row["trade_count"] for row in output) != row_total:
            raise AssertionError(f"{month} ${size} footprint does not reconcile to raw rows")
        footprint_rows[size] = output

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
        bar_rows.append(
            {
                "bar_start_utc": start,
                "bar_end_utc": start + timedelta(minutes=5),
                "session_date": day,
                "month": month,
                **row,
                "delta_btc": delta,
                "cumulative_volume_delta_btc": cvd,
                "daily_vwap": cumulative_notional / cumulative_volume,
            }
        )
    if sum(row["trade_count"] for row in bar_rows) != row_total:
        raise AssertionError(f"{month} five-minute bars do not reconcile to raw rows")
    return footprint_rows, bar_rows, diagnostics


def build_v3_footprint_dataset(
    source_manifest_path: str | Path,
    cache_root: str | Path,
    *,
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    source_validation = validate_v3_source_manifest(source_manifest_path, verify_archives=False)
    if not source_validation["valid"]:
        raise ValueError("invalid V3 normalized source: " + "; ".join(source_validation["errors"]))
    source: MonthlyAggregateTradeManifest = source_validation["manifest"]
    source_path = Path(source_manifest_path).resolve()
    identity = {
        "builder_version": FOOTPRINT_BUILDER_VERSION,
        "normalized_dataset_hash": source.normalized_dataset_hash,
        "normalized_manifest_hash": source.manifest_hash,
        "normalized_manifest_file_sha256": sha256_file(source_path),
        "symbol": "BTCUSDT",
        "months": list(AUTHORIZED_MONTHS),
        "bar_interval": "5m",
        "bin_sizes_usd": [str(value) for value in EXACT_BIN_SIZES],
        "bin_interval": "HALF_OPEN_FLOOR_FROM_RAW_PRICE",
        "aggressor_rule": "BUY_IFF_BUYER_IS_MAKER_FALSE",
        "daily_vwap_reset": "UTC_MIDNIGHT",
    }
    footprint_id = sha256_value(identity)
    root = Path(cache_root).resolve() / "BTCUSDT" / footprint_id
    output_manifest = root / "manifest.json"
    if output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"immutable V3 footprint identity collision: {output_manifest}")
        validation = validate_v3_footprint_dataset(output_manifest)
        if not validation["valid"]:
            raise ValueError("invalid existing V3 footprint: " + "; ".join(validation["errors"]))
        return existing

    context = ArtifactContext(
        run_id=footprint_id,
        dataset_hash=source.normalized_dataset_hash,
        source_manifest_hash=source.manifest_hash,
        specification_hash=sha256_value(identity),
        parameter_hash=sha256_value({"bin_sizes": identity["bin_sizes_usd"], "bar_interval": "5m"}),
        code_hash=sha256_value({"builder_version": FOOTPRINT_BUILDER_VERSION}),
        evidence_label=EVIDENCE_LABEL,
        timestamp=source.retrieved_at.isoformat(),
    )
    parquet_files: list[dict[str, Any]] = []
    batch_boundaries: list[dict[str, Any]] = []
    total_bars = 0
    total_footprints = {str(size): 0 for size in EXACT_BIN_SIZES}
    total_trades = 0
    for partition in source.partitions:
        parquet = source_path.parent / partition.file_name
        monthly, bars, diagnostics = _aggregate_month(
            pq.ParquetFile(parquet),
            month=partition.month,
            expected_first_id=partition.first_aggregate_trade_id,
            expected_last_id=partition.last_aggregate_trade_id,
            batch_size=batch_size,
        )
        bars_path = root / "bars" / f"{partition.month}.parquet"
        context.write_parquet(bars_path, bars)
        parquet_files.append(
            {
                "kind": "bars",
                "month": partition.month,
                "bin_size_usd": None,
                "relative_path": bars_path.relative_to(root).as_posix(),
                "row_count": len(bars),
                "trade_count": sum(row["trade_count"] for row in bars),
                "sha256": sha256_file(bars_path),
                "schema": str(pq.ParquetFile(bars_path).schema_arrow),
            }
        )
        total_bars += len(bars)
        total_trades += sum(row["trade_count"] for row in bars)
        for size, rows in monthly.items():
            target = root / "footprints" / str(size) / f"{partition.month}.parquet"
            context.write_parquet(target, rows)
            parquet_files.append(
                {
                    "kind": "footprints",
                    "month": partition.month,
                    "bin_size_usd": str(size),
                    "relative_path": target.relative_to(root).as_posix(),
                    "row_count": len(rows),
                    "trade_count": sum(row["trade_count"] for row in rows),
                    "sha256": sha256_file(target),
                    "schema": str(pq.ParquetFile(target).schema_arrow),
                }
            )
            total_footprints[str(size)] += len(rows)
        batch_boundaries.extend(diagnostics)
    if total_trades != source.row_count:
        raise AssertionError("V3 bars do not reconcile to the normalized source row count")
    for size in EXACT_BIN_SIZES:
        reconciled = sum(
            item["trade_count"]
            for item in parquet_files
            if item["kind"] == "footprints" and item["bin_size_usd"] == str(size)
        )
        if reconciled != source.row_count:
            raise AssertionError(f"V3 ${size} footprints do not reconcile to normalized rows")
    content_hash = sha256_value({item["relative_path"]: item["sha256"] for item in parquet_files})
    output = context.envelope(
        {
            "identity": identity,
            "footprint_dataset_hash": content_hash,
            "footprint_root": str(root),
            "source_manifest_path": str(source_path),
            "source_row_count": source.row_count,
            "streamed_trade_count": total_trades,
            "streamed_trade_count_by_bin_size": {
                str(size): source.row_count for size in EXACT_BIN_SIZES
            },
            "five_minute_bar_count": total_bars,
            "footprint_row_count_by_bin_size": total_footprints,
            "batch_size": batch_size,
            "batch_count": len(batch_boundaries),
            "batch_boundaries": batch_boundaries,
            "parquet_files": parquet_files,
            "valid": True,
        }
    )
    write_bytes_once(
        output_manifest,
        json.dumps(output, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n",
    )
    validation = validate_v3_footprint_dataset(output_manifest)
    if not validation["valid"]:
        raise RuntimeError("created V3 footprint failed validation: " + "; ".join(validation["errors"]))
    return output


def validate_v3_footprint_dataset(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if manifest_path.name != "manifest.json":
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    errors: list[str] = []
    hashes: dict[str, str] = {}
    rows: dict[str, int] = defaultdict(int)
    trade_counts: dict[str, int] = defaultdict(int)
    for item in manifest.get("parquet_files", []):
        target = root / item["relative_path"]
        key = item["kind"] if item["kind"] == "bars" else f"footprints:{item['bin_size_usd']}"
        try:
            actual_hash = sha256_file(target)
            actual_rows = pq.ParquetFile(target).metadata.num_rows
            if actual_hash != item["sha256"]:
                raise ValueError("SHA-256 mismatch")
            if actual_rows != item["row_count"]:
                raise ValueError("row-count mismatch")
            hashes[item["relative_path"]] = actual_hash
            rows[key] += actual_rows
            trade_counts[key] += int(item["trade_count"])
        except (OSError, ValueError) as exc:
            errors.append(f"{target}: {exc}")
    source_rows = int(manifest.get("source_row_count", -1))
    if trade_counts["bars"] != source_rows:
        errors.append("bar trade counts do not reconcile to source rows")
    for size in EXACT_BIN_SIZES:
        key = f"footprints:{size}"
        if trade_counts[key] != source_rows:
            errors.append(f"${size} footprint trade counts do not reconcile to source rows")
        if rows[key] != int(manifest.get("footprint_row_count_by_bin_size", {}).get(str(size), -1)):
            errors.append(f"${size} footprint row total mismatch")
    if rows["bars"] != int(manifest.get("five_minute_bar_count", -1)):
        errors.append("five-minute bar row total mismatch")
    if sha256_value(hashes) != manifest.get("footprint_dataset_hash"):
        errors.append("content-addressed V3 footprint hash mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_path": str(manifest_path),
        "footprint_dataset_hash": manifest.get("footprint_dataset_hash"),
        "source_row_count": source_rows,
        "rows": dict(rows),
        "trade_counts": dict(trade_counts),
        "parquet_sha256": hashes,
    }


def load_v3_bars(manifest: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    raw, root = _resolve_footprint_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for item in raw["parquet_files"]:
        if item["kind"] == "bars":
            rows.extend(pq.read_table(root / item["relative_path"]).to_pylist())
    return rows


def load_v3_footprints(
    manifest: dict[str, Any] | str | Path,
    bin_size_usd: Decimal,
) -> list[dict[str, Any]]:
    size = str(bin_size_usd)
    if Decimal(size) not in EXACT_BIN_SIZES:
        raise ValueError(f"unsupported V3 exact footprint bin size: {size}")
    raw, root = _resolve_footprint_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for item in raw["parquet_files"]:
        if item["kind"] == "footprints" and item["bin_size_usd"] == size:
            rows.extend(pq.read_table(root / item["relative_path"]).to_pylist())
    return rows


def _resolve_footprint_manifest(
    manifest: dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest, dict):
        return manifest, Path(manifest["footprint_root"])
    path = Path(manifest).resolve()
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path.parent
