from __future__ import annotations

import json
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
    PARQUET_SCHEMA,
    AggregateTradeImporter,
    AggregateTradePartition,
    MonthlyAggregateTradeManifest,
    _monthly_manifest_hash,
)
from .artifacts import ArtifactContext, sha256_file, sha256_value, write_bytes_once
from .footprint import RAW_COLUMNS, _batch_table, _decimal, _utc_from_epoch_minute
from .v3_data import (
    _archive_layout,
    _canonical_header_map,
)
from .v4_models import PHASE_A_MONTHS, PHASE_B_MONTHS, SELECTION_EVIDENCE

OFFICIAL_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
V4_NORMALIZER_VERSION = "imbalance-vwap-ride-v4-vectorized-normalizer-1"
V4_BAR_BUILDER_VERSION = "imbalance-vwap-ride-v4-exact-5m-footprints-1"
BIN_SIZE_USD = Decimal("50")
MAX_ARCHIVE_ORDER_LANES = 64


def phase_months(phase: str) -> tuple[str, ...]:
    normalized = phase.upper().replace("-", "_")
    if normalized in {"A", "PHASE_A"}:
        return PHASE_A_MONTHS
    if normalized in {"B", "PHASE_B"}:
        return PHASE_B_MONTHS
    raise ValueError(f"unsupported V4 data phase: {phase}")


def authorized_archive_url(
    month: str,
    *,
    phase: str | None = None,
    expected_months: tuple[str, ...] | None = None,
) -> str:
    """Return an official archive URL constrained to one sealed month set.

    ``expected_months`` is deliberately opt-in for successor studies.  V4
    callers retain the historical phase allowlists unchanged.
    """
    if expected_months is not None and phase is not None:
        raise ValueError("provide either phase or expected_months, not both")
    allowed = expected_months if expected_months is not None else (
        phase_months(phase) if phase is not None else PHASE_A_MONTHS + PHASE_B_MONTHS
    )
    if month not in allowed:
        raise ValueError(f"month is outside the sealed V4 download allowlist: {month}")
    url = f"{OFFICIAL_ARCHIVE_ROOT}/BTCUSDT/BTCUSDT-aggTrades-{month}.zip"
    parsed = urlparse(url)
    expected = f"/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{month}.zip"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "data.binance.vision"
        or parsed.port is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.path != expected
    ):
        raise ValueError(f"invalid Binance V4 archive URL: {url}")
    return url


def validate_authorized_archive(
    path: str | Path,
    month: str,
    *,
    phase: str | None = None,
    expected_months: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    archive = Path(path).resolve()
    authorized_archive_url(month, phase=phase, expected_months=expected_months)
    expected_name = f"BTCUSDT-aggTrades-{month}.zip"
    expected_member = f"BTCUSDT-aggTrades-{month}.csv"
    if archive.name != expected_name:
        raise ValueError(f"archive is outside the sealed V4 allowlist: {archive}")
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [item for item in bundle.infolist() if not item.is_dir()]
            if len(members) != 1 or members[0].filename != expected_member:
                raise ValueError(f"archive {archive.name} must contain exactly {expected_member}")
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
        "url": authorized_archive_url(month, phase=phase, expected_months=expected_months),
        "archive_path": str(archive),
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "csv_member": expected_member,
        "csv_uncompressed_size_bytes": member_size,
        "csv_crc32": member_crc,
        "zip_integrity_valid": True,
    }


def _vectorized_month_partition(
    root: str | Path,
    importer: AggregateTradeImporter,
    month: str,
    archive: str | Path,
    *,
    phase: str | None = None,
) -> tuple[AggregateTradePartition, dict[str, Any]]:
    """Normalize one official archive with bounded-memory order repair.

    Some official 2023 Binance monthly archives contain a bounded number of
    individually ordered ID/time lanes interleaved in the CSV.  V4 must not
    accept that source order as chronological, and it must not load a whole
    month in memory to sort it.  Rows are therefore streamed into monotonic
    temporary lane Parquets, the lane ranges are proved disjoint/contiguous,
    and the lanes are concatenated into one validated chronological output
    before its hash and ``partition.json`` are committed.
    """

    authorized_archive_url(month, phase=phase)
    cache_root = Path(root).resolve()
    archive_path = Path(archive).resolve()
    reusable = importer._load_reusable_partition("BTCUSDT", month, archive_path)
    if reusable is not None:
        _validate_normalized_partition(
            cache_root / "normalized" / "BTCUSDT" / "monthly_partitions" / month / reusable.file_name,
            reusable,
        )
        return reusable, {
            "month": month,
            "action": "SKIPPED_HASH_AND_OUTPUT_VERIFIED",
            "reused": True,
            "repair_status": reusable.repair_status,
            "archive": str(archive_path),
            "normalizer_version": "EXISTING_VALID_PARTITION",
            "output_validation": "SCHEMA_ROWS_IDS_TIMESTAMPS_MONTH_HASH_VALID",
        }

    member, headers, has_header = _archive_layout(archive_path, month)
    mapping = _canonical_header_map(headers)
    column_types: dict[str, pa.DataType] = {}
    for canonical, raw_name in mapping.items():
        if canonical in {
            "aggregate_trade_id",
            "first_trade_id",
            "last_trade_id",
            "trade_time",
            "event_time",
        }:
            column_types[raw_name] = pa.int64()
        elif canonical in {"buyer_is_maker", "is_best_match"}:
            column_types[raw_name] = pa.bool_()
        else:
            column_types[raw_name] = pa.string()

    archive_hash = sha256_file(archive_path)
    staging_root = cache_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    prefix = f"v4-{month}-{archive_hash[:16]}"
    final_staging = staging_root / f"{prefix}.parquet.part"
    lane_paths: list[Path] = []
    lane_writers: list[pq.ParquetWriter] = []
    lane_first_ids: list[int] = []
    lane_last_ids: list[int] = []
    lane_first_times: list[int] = []
    lane_last_times: list[int] = []
    lane_rows: list[int] = []
    timestamp_gaps: list[dict[str, Any]] = []
    global_rows = 0
    expected_year, expected_month = (int(value) for value in month.split("-", 1))
    source_file = f"{archive_path.name}!{member}"

    def add_lane() -> int:
        lane = len(lane_writers)
        if lane >= MAX_ARCHIVE_ORDER_LANES:
            raise ValueError(
                f"{month}: official archive requires more than "
                f"{MAX_ARCHIVE_ORDER_LANES} bounded order lanes"
            )
        lane_path = staging_root / f"{prefix}.lane-{lane:02d}.parquet.part"
        lane_path.unlink(missing_ok=True)
        lane_paths.append(lane_path)
        lane_writers.append(pq.ParquetWriter(lane_path, PARQUET_SCHEMA, compression="zstd"))
        lane_first_ids.append(-1)
        lane_last_ids.append(-1)
        lane_first_times.append(-1)
        lane_last_times.append(-1)
        lane_rows.append(0)
        return lane

    try:
        with zipfile.ZipFile(archive_path) as bundle, bundle.open(member) as raw:
            reader = pacsv.open_csv(
                raw,
                read_options=pacsv.ReadOptions(
                    block_size=32 * 1024 * 1024,
                    column_names=None if has_header else headers,
                    skip_rows=0,
                    use_threads=True,
                ),
                convert_options=pacsv.ConvertOptions(
                    column_types=column_types,
                    strings_can_be_null=False,
                ),
            )
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                table = pa.Table.from_batches([batch])
                ids_array = pc.cast(
                    table[mapping["aggregate_trade_id"]], pa.int64()
                ).combine_chunks()
                ids = ids_array.to_numpy(zero_copy_only=False)
                raw_times_array = pc.cast(
                    table[mapping["trade_time"]], pa.int64()
                ).combine_chunks()
                raw_times = raw_times_array.to_numpy(zero_copy_only=False)
                microseconds = int(raw_times[0]) > 10_000_000_000_000
                timestamp_ms = raw_times // (1000 if microseconds else 1)

                prior_lane_last_ids = list(lane_last_ids)
                assignments = np.empty(batch.num_rows, dtype=np.uint8)
                for position, aggregate_id_value in enumerate(ids):
                    aggregate_id = int(aggregate_id_value)
                    matching = [
                        lane
                        for lane, last_id in enumerate(lane_last_ids)
                        if aggregate_id == last_id + 1
                    ]
                    if len(matching) > 1:
                        raise ValueError(f"{month}: ambiguous duplicate aggregate-trade lane")
                    lane = matching[0] if matching else add_lane()
                    assignments[position] = lane
                    lane_last_ids[lane] = aggregate_id

                for lane in range(len(lane_writers)):
                    positions = np.flatnonzero(assignments == lane)
                    if not len(positions):
                        continue
                    lane_table = table.take(pa.array(positions, type=pa.int64()))
                    lane_ids = ids[positions]
                    lane_times = timestamp_ms[positions]
                    if lane_ids.size > 1 and not np.all(np.diff(lane_ids) == 1):
                        raise ValueError(f"{month}: duplicate or gapped aggregate-trade ID within order lane")
                    if lane_times.size > 1 and np.any(np.diff(lane_times) < 0):
                        raise ValueError(f"{month}: aggregate-trade timestamps regress within order lane")
                    first_id, last_id = int(lane_ids[0]), int(lane_ids[-1])
                    first_ms, last_ms = int(lane_times[0]), int(lane_times[-1])
                    if lane_rows[lane]:
                        if first_id != prior_lane_last_ids[lane] + 1:
                            raise ValueError(f"{month}: order-lane batch boundary ID gap")
                        if first_ms < lane_last_times[lane]:
                            raise ValueError(f"{month}: order-lane batch boundary timestamp regression")
                    else:
                        lane_first_ids[lane] = first_id
                        lane_first_times[lane] = first_ms
                    first_dt = datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc)
                    last_dt = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
                    if (first_dt.year, first_dt.month) != (expected_year, expected_month) or (
                        last_dt.year,
                        last_dt.month,
                    ) != (expected_year, expected_month):
                        raise ValueError(f"{month}: timestamp is outside the requested calendar month")

                    price = pc.cast(lane_table[mapping["price"]], pa.string()).combine_chunks()
                    quantity = pc.cast(
                        lane_table[mapping["quantity_base"]], pa.string()
                    ).combine_chunks()
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
                    maker = pc.cast(
                        lane_table[mapping["buyer_is_maker"]], pa.bool_()
                    ).combine_chunks()
                    timestamp_strings = pc.cast(
                        pc.cast(pa.array(lane_times, type=pa.int64()), pa.timestamp("ms", tz="UTC")),
                        pa.string(),
                    )
                    signed = pc.cast(
                        pc.if_else(maker, pc.negate(quantity_decimal), quantity_decimal),
                        pa.string(),
                    )
                    count = len(positions)
                    first_trade = (
                        pc.cast(
                            lane_table[mapping["first_trade_id"]], pa.int64()
                        ).combine_chunks()
                        if "first_trade_id" in mapping
                        else pa.nulls(count, type=pa.int64())
                    )
                    last_trade = (
                        pc.cast(
                            lane_table[mapping["last_trade_id"]], pa.int64()
                        ).combine_chunks()
                        if "last_trade_id" in mapping
                        else pa.nulls(count, type=pa.int64())
                    )
                    normalized = pa.Table.from_arrays(
                        [
                            timestamp_strings,
                            timestamp_strings,
                            pc.cast(lane_table[mapping["aggregate_trade_id"]], pa.int64()).combine_chunks(),
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
                    lane_writers[lane].write_table(normalized)
                    lane_last_ids[lane] = last_id
                    lane_last_times[lane] = last_ms
                    lane_rows[lane] += count
                global_rows += batch.num_rows
    finally:
        for writer in lane_writers:
            writer.close()

    if not global_rows or not lane_rows:
        raise ValueError(f"{month}: aggregate-trade archive is empty")
    ordered_lanes = sorted(range(len(lane_rows)), key=lambda lane: lane_first_ids[lane])
    previous_lane: int | None = None
    lane_gap_counts: list[int] = []
    for lane in ordered_lanes:
        if lane_last_ids[lane] - lane_first_ids[lane] + 1 != lane_rows[lane]:
            raise ValueError(f"{month}: duplicate or missing aggregate-trade IDs in order lane")
        if previous_lane is not None:
            if lane_first_ids[lane] <= lane_last_ids[previous_lane]:
                raise ValueError(f"{month}: duplicate or overlap across archive order lanes")
            lane_gap_counts.append(lane_first_ids[lane] - lane_last_ids[previous_lane] - 1)
            if lane_first_times[lane] < lane_last_times[previous_lane]:
                raise ValueError(f"{month}: timestamp regression across archive order lanes")
        previous_lane = lane

    final_staging.unlink(missing_ok=True)
    final_writer = pq.ParquetWriter(final_staging, PARQUET_SCHEMA, compression="zstd")
    try:
        for lane in ordered_lanes:
            for lane_batch in pq.ParquetFile(lane_paths[lane]).iter_batches(batch_size=1_000_000):
                final_writer.write_batch(lane_batch)
    finally:
        final_writer.close()

    first_lane, last_lane = ordered_lanes[0], ordered_lanes[-1]
    first_id, last_id = lane_first_ids[first_lane], lane_last_ids[last_lane]
    first_timestamp_ms = lane_first_times[first_lane]
    last_timestamp_ms = lane_last_times[last_lane]
    missing_id_count = last_id - first_id + 1 - global_rows
    if missing_id_count < 0 or missing_id_count != sum(lane_gap_counts):
        raise ValueError(f"{month}: normalized aggregate-trade ID gap accounting failed")
    parquet_hash = sha256_file(final_staging)
    dataset_hash = sha256_value(
        {
            "normalizer_version": V4_NORMALIZER_VERSION,
            "month": month,
            "archive_sha256": archive_hash,
            "parquet_sha256": parquet_hash,
            "row_count": global_rows,
            "first_aggregate_trade_id": first_id,
            "last_aggregate_trade_id": last_id,
            "archive_order_lane_count": len(ordered_lanes),
        }
    )
    partition_dir = cache_root / "normalized" / "BTCUSDT" / "monthly_partitions" / month
    partition_dir.mkdir(parents=True, exist_ok=True)
    destination = partition_dir / f"{dataset_hash}.parquet"
    provisional = AggregateTradePartition(
        month=month,
        file_name=destination.name,
        parquet_hash=parquet_hash,
        normalized_dataset_hash=dataset_hash,
        source_archive=str(archive_path),
        source_archive_hash=archive_hash,
        row_count=global_rows,
        duplicate_count=0,
        first_timestamp=datetime.fromtimestamp(first_timestamp_ms / 1000, tz=timezone.utc),
        last_timestamp=datetime.fromtimestamp(last_timestamp_ms / 1000, tz=timezone.utc),
        first_aggregate_trade_id=first_id,
        last_aggregate_trade_id=last_id,
        missing_interval_diagnostics=(
            timestamp_gaps
            + (
                [
                    {
                        "kind": "AGGREGATE_TRADE_ID_GAP",
                        "missing_id_count": missing_id_count,
                        "repair_status": "OFFICIAL_ARCHIVE_GAP_PRESERVED",
                    }
                ]
                if missing_id_count
                else []
            )
        ),
        continuity_diagnostics=[
            {
                "archive_source_order": "ORDERED" if len(ordered_lanes) == 1 else "INTERLEAVED_MONOTONIC_LANES",
                "archive_order_lane_count": len(ordered_lanes),
                "normalized_output_order": "STRICT_AGGREGATE_TRADE_ID_AND_NONDECREASING_TIMESTAMP",
                "duplicate_count": 0,
                "row_count": global_rows,
                "missing_aggregate_trade_id_count": missing_id_count,
            }
        ],
        repair_status=(
            "OFFICIAL_ARCHIVE_ORDER_AND_GAPS_NORMALIZED"
            if len(ordered_lanes) > 1 and missing_id_count
            else "OFFICIAL_ARCHIVE_ORDER_NORMALIZED"
            if len(ordered_lanes) > 1
            else "OFFICIAL_ARCHIVE_GAPS_PRESERVED"
            if missing_id_count
            else "ARCHIVE_ONLY"
        ),
    )
    _validate_normalized_partition(final_staging, provisional)
    if destination.exists():
        if sha256_file(destination) != parquet_hash:
            raise ValueError(f"immutable V4 normalized partition collision: {destination}")
        final_staging.unlink(missing_ok=True)
    else:
        final_staging.replace(destination)
    _validate_normalized_partition(destination, provisional)
    write_bytes_once(
        partition_dir / "partition.json",
        json.dumps(provisional.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8"),
    )
    for lane_path in lane_paths:
        lane_path.unlink(missing_ok=True)
    return provisional, {
        "month": month,
        "action": "NEWLY_PROCESSED_BOUNDED_ORDER_LANES",
        "reused": False,
        "repair_status": provisional.repair_status,
        "archive": str(archive_path),
        "normalizer_version": V4_NORMALIZER_VERSION,
        "row_count": global_rows,
        "parquet_sha256": parquet_hash,
        "archive_order_lane_count": len(ordered_lanes),
        "missing_aggregate_trade_id_count": missing_id_count,
        "output_validation": "SCHEMA_ROWS_IDS_TIMESTAMPS_MONTH_HASH_VALID",
        "continuity_diagnostics": provisional.continuity_diagnostics,
    }


def _validate_normalized_partition(
    path: Path,
    partition: AggregateTradePartition,
    *,
    batch_size: int = 1_000_000,
) -> None:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.remove_metadata().equals(PARQUET_SCHEMA):
        raise ValueError(f"{partition.month}: normalized output schema mismatch")
    if parquet.metadata.num_rows != partition.row_count:
        raise ValueError(f"{partition.month}: normalized output row-count mismatch")
    previous_id: int | None = None
    previous_time: int | None = None
    rows = 0
    first_id = last_id = first_time = last_time = None
    expected_year, expected_month = (int(value) for value in partition.month.split("-", 1))
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["aggregate_trade_id", "trade_time_utc"],
        use_threads=True,
    ):
        table = pa.Table.from_batches([batch])
        ids = table["aggregate_trade_id"].combine_chunks().to_numpy(zero_copy_only=False)
        times = pc.cast(
            pc.cast(table["trade_time_utc"].combine_chunks(), pa.timestamp("us", tz="UTC")),
            pa.int64(),
        ).to_numpy(zero_copy_only=False)
        if ids.size > 1 and np.any(np.diff(ids) <= 0):
            raise ValueError(f"{partition.month}: normalized output IDs are not strictly increasing")
        if times.size > 1 and np.any(np.diff(times) < 0):
            raise ValueError(f"{partition.month}: normalized output timestamps regress")
        batch_first_id, batch_last_id = int(ids[0]), int(ids[-1])
        batch_first_time, batch_last_time = int(times[0]), int(times[-1])
        if previous_id is not None and batch_first_id <= previous_id:
            raise ValueError(f"{partition.month}: normalized output batch ID overlap or regression")
        if previous_time is not None and batch_first_time < previous_time:
            raise ValueError(f"{partition.month}: normalized output batch timestamp regression")
        if first_id is None:
            first_id, first_time = batch_first_id, batch_first_time
        last_id, last_time = batch_last_id, batch_last_time
        previous_id, previous_time = batch_last_id, batch_last_time
        rows += batch.num_rows
    if None in {first_id, last_id, first_time, last_time}:
        raise ValueError(f"{partition.month}: normalized output is empty")
    first_dt = datetime.fromtimestamp(int(first_time) / 1_000_000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(last_time) / 1_000_000, tz=timezone.utc)
    if (first_dt.year, first_dt.month) != (expected_year, expected_month) or (
        last_dt.year,
        last_dt.month,
    ) != (expected_year, expected_month):
        raise ValueError(f"{partition.month}: normalized output timestamp coverage mismatch")
    if (
        rows != partition.row_count
        or first_id != partition.first_aggregate_trade_id
        or last_id != partition.last_aggregate_trade_id
        or first_dt != partition.first_timestamp
        or last_dt != partition.last_timestamp
        or sha256_file(path) != partition.parquet_hash
    ):
        raise ValueError(f"{partition.month}: normalized output metadata or hash mismatch")


def _atomic_copy_once(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_hash:
            raise ValueError(f"immutable V4 combined partition collision: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"V4 combined partition copy hash mismatch: {source}")
    temporary.replace(destination)


def _combine_phase_partitions(
    root: Path,
    phase: str,
    partitions: list[AggregateTradePartition],
) -> tuple[Path, MonthlyAggregateTradeManifest]:
    months = phase_months(phase)
    if tuple(item.month for item in partitions) != months:
        raise ValueError(f"{phase} partitions do not match the exact sealed month range")
    previous: AggregateTradePartition | None = None
    for partition in partitions:
        if partition.duplicate_count != 0:
            raise ValueError(f"{partition.month}: duplicate aggregate-trade rows are forbidden")
        if partition.last_aggregate_trade_id - partition.first_aggregate_trade_id + 1 < partition.row_count:
            raise ValueError(f"{partition.month}: aggregate-trade ID span is smaller than row count")
        if previous is not None:
            if partition.first_aggregate_trade_id <= previous.last_aggregate_trade_id:
                raise ValueError(f"{previous.month}->{partition.month}: inter-month ID overlap or regression")
            if partition.first_timestamp <= previous.last_timestamp:
                raise ValueError(f"{previous.month}->{partition.month}: timestamp overlap")
        previous = partition
    identity = {
        "normalizer_version": V4_NORMALIZER_VERSION,
        "phase": phase,
        "symbol": "BTCUSDT",
        "months": [
            {
                "month": item.month,
                "normalized_dataset_hash": item.normalized_dataset_hash,
                "parquet_hash": item.parquet_hash,
                "source_archive_hash": item.source_archive_hash,
            }
            for item in partitions
        ],
    }
    dataset_hash = sha256_value(identity)
    combined_root = root / "normalized" / "BTCUSDT" / "v4" / phase.lower() / dataset_hash
    combined: list[AggregateTradePartition] = []
    for item in partitions:
        source = root / "normalized" / "BTCUSDT" / "monthly_partitions" / item.month / item.file_name
        destination = combined_root / f"{item.month}.parquet"
        _atomic_copy_once(source, destination, item.parquet_hash)
        combined.append(item.model_copy(update={"file_name": destination.name}))
    manifest_path = combined_root / "manifest.json"
    if manifest_path.exists():
        existing = AggregateTradeImporter(root).validate_monthly_manifest(manifest_path)
        if existing.normalized_dataset_hash != dataset_hash:
            raise ValueError(f"immutable V4 phase manifest collision: {manifest_path}")
        return manifest_path, existing
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
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    AggregateTradeImporter(root).validate_monthly_manifest(manifest_path)
    return manifest_path, manifest


def acquire_and_normalize_v4_phase(
    cache_root: str | Path,
    *,
    phase: str,
    allow_authorized_downloads: bool,
) -> tuple[Path, MonthlyAggregateTradeManifest, dict[str, Any]]:
    """Resolve and normalize only one sealed phase from official public archives."""

    root = Path(cache_root).resolve()
    importer = AggregateTradeImporter(root)
    partitions: list[AggregateTradePartition] = []
    actions: list[dict[str, Any]] = []
    archive_reports: list[dict[str, Any]] = []
    for month in phase_months(phase):
        if importer.archive_url("BTCUSDT", month) != authorized_archive_url(month, phase=phase):
            raise RuntimeError("Binance importer URL failed the sealed V4 allowlist")
        archive_path = root / "downloads" / "BTCUSDT" / f"BTCUSDT-aggTrades-{month}.zip"
        existed = archive_path.is_file()
        archive = importer.download_month(
            "BTCUSDT", month, allow_network=allow_authorized_downloads
        )
        try:
            archive_report = validate_authorized_archive(archive, month, phase=phase)
        except ValueError:
            if not allow_authorized_downloads:
                raise
            quarantine = archive.with_name(f".{archive.name}.{sha256_file(archive)[:16]}.invalid")
            if quarantine.exists() and sha256_file(quarantine) != sha256_file(archive):
                raise ValueError(f"invalid archive quarantine collision: {quarantine}")
            if not quarantine.exists():
                archive.replace(quarantine)
            else:
                archive.unlink()
            archive = importer.download_month("BTCUSDT", month, allow_network=True)
            archive_report = validate_authorized_archive(archive, month, phase=phase)
            existed = False
        partition, action = _vectorized_month_partition(
            root, importer, month, archive, phase=phase
        )
        archive_report["action"] = (
            "REUSED_HASH_AND_CRC_VALID_LOCAL_ARCHIVE"
            if existed
            else "DOWNLOADED_AUTHORIZED_ARCHIVE"
        )
        archive_reports.append(archive_report)
        partitions.append(partition)
        actions.append({**action, "archive_validation": archive_report})
    manifest_path, manifest = _combine_phase_partitions(root, phase, partitions)
    validation = validate_v4_source_manifest(manifest_path, phase=phase, verify_archives=False)
    if not validation["valid"]:
        raise ValueError("V4 source validation failed: " + "; ".join(validation["errors"]))
    report = {
        "phase": phase,
        "provider": "Binance USD-M Futures public data archive",
        "official_origin": "https://data.binance.vision",
        "symbol": "BTCUSDT",
        "authorized_months": list(phase_months(phase)),
        "network_scope": "ONLY_SEALED_MONTHLY_ARCHIVES",
        "network_request_count": sum(
            item["action"] == "DOWNLOADED_AUTHORIZED_ARCHIVE" for item in archive_reports
        ),
        "archives": archive_reports,
        "normalization_actions": actions,
        "normalized_manifest_path": str(manifest_path),
        "normalized_manifest_file_sha256": sha256_file(manifest_path),
        "normalized_manifest_hash": manifest.manifest_hash,
        "normalized_dataset_hash": manifest.normalized_dataset_hash,
        "raw_aggregate_rows_transmitted": False,
        "unrelated_assets_downloaded": False,
    }
    return manifest_path, manifest, report


def validate_v4_source_manifest(
    manifest_path: str | Path,
    *,
    phase: str,
    verify_archives: bool = True,
    expected_months: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    errors: list[str] = []
    archives: list[dict[str, Any]] = []
    importer = AggregateTradeImporter(path.parents[4] if len(path.parents) > 4 else path.parent)
    try:
        manifest = importer.validate_monthly_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)], "manifest_path": str(path)}
    # V4 defaults to its immutable phase ranges.  Later studies may supply an
    # explicitly sealed range, never an open-ended month selector.
    expected = expected_months if expected_months is not None else phase_months(phase)
    if not expected or len(set(expected)) != len(expected) or tuple(sorted(expected)) != expected:
        return {"valid": False, "errors": ["expected_months must be a non-empty, unique chronological tuple"], "manifest_path": str(path)}
    actual = tuple(item.month for item in manifest.partitions)
    if actual != expected:
        errors.append(f"manifest months must equal the exact sealed {phase} range: {actual}")
    if manifest.symbol != "BTCUSDT":
        errors.append("manifest symbol must be BTCUSDT")
    if manifest.duplicate_count != 0:
        errors.append("normalized V4 dataset contains duplicate aggregate-trade IDs")
    try:
        calendar_diagnostics = importer.validate_complete_calendar_months(
            manifest,
            start_month=expected[0],
            end_month=expected[-1],
        )
    except ValueError as exc:
        errors.append(str(exc))
        calendar_diagnostics = []
    previous: AggregateTradePartition | None = None
    rows = 0
    for partition in manifest.partitions:
        if partition.duplicate_count != 0:
            errors.append(f"{partition.month}: duplicate aggregate-trade rows are forbidden")
        if partition.last_aggregate_trade_id - partition.first_aggregate_trade_id + 1 < partition.row_count:
            errors.append(f"{partition.month}: aggregate-trade ID span is smaller than row count")
        if previous is not None and partition.first_aggregate_trade_id <= previous.last_aggregate_trade_id:
            errors.append(f"{previous.month}->{partition.month}: aggregate-trade ID overlap or regression")
        previous = partition
        rows += partition.row_count
        if verify_archives:
            try:
                report = validate_authorized_archive(
                    partition.source_archive,
                    partition.month,
                    phase=None if expected_months is not None else phase,
                    expected_months=expected_months,
                )
                if report["archive_sha256"] != partition.source_archive_hash:
                    raise ValueError("archive hash differs from partition provenance")
                archives.append(report)
            except (OSError, ValueError) as exc:
                errors.append(f"{partition.month}: {exc}")
    if rows != manifest.row_count:
        errors.append("partition row total does not equal combined manifest row_count")
    temporary = [
        item.name
        for item in path.parent.rglob("*")
        if item.is_file() and (item.name.endswith(".tmp") or item.name.endswith(".part"))
    ]
    if temporary:
        errors.append(f"temporary output is not admissible: {temporary}")
    return {
        "valid": not errors,
        "errors": errors,
        "phase": phase,
        "manifest_path": str(path),
        "manifest_file_sha256": sha256_file(path),
        "manifest_hash": manifest.manifest_hash,
        "normalized_dataset_hash": manifest.normalized_dataset_hash,
        "months": list(actual),
        "row_count": manifest.row_count,
        "duplicate_count": manifest.duplicate_count,
        "archives": archives,
        "calendar_diagnostics": calendar_diagnostics,
        "temporary_outputs": temporary,
        "manifest": manifest,
    }


def _aggregate_month(
    parquet_file: pq.ParquetFile,
    *,
    month: str,
    expected_first_id: int,
    expected_last_id: int,
    expected_row_count: int | None = None,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream one month into exact $50 footprints and UTC five-minute bars.

    Aggregate-trade IDs must be unique and strictly increasing, but official
    archive gaps are preserved and reconciled by row count instead of being
    mistaken for duplicate data.  Only batch-sized raw Arrow arrays plus the
    bounded month-level bar/bin accumulators are retained.
    """

    footprints: dict[tuple[int, int], list[Any]] = {}
    bars: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    previous_id: int | None = None
    previous_timestamp_us: int | None = None
    row_total = 0
    missing_id_total = 0
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(RAW_COLUMNS),
            use_threads=True,
        )
    ):
        if batch.num_rows == 0:
            continue
        table = _batch_table(batch)
        ids = table["aggregate_trade_id"].combine_chunks().to_numpy(zero_copy_only=False)
        id_differences = np.diff(ids)
        if id_differences.size and np.any(id_differences <= 0):
            raise ValueError(f"duplicate or unordered aggregate-trade IDs in {month} batch {batch_number}")
        times = table["raw_timestamp"].combine_chunks()
        timestamp_values = pc.cast(
            pc.cast(times, pa.timestamp("us", tz="UTC")),
            pa.int64(),
        ).to_numpy(zero_copy_only=False)
        if timestamp_values.size > 1 and np.any(np.diff(timestamp_values) < 0):
            raise ValueError(f"timestamp regression in {month} batch {batch_number}")
        first_id, last_id = int(ids[0]), int(ids[-1])
        first_timestamp_us, last_timestamp_us = int(timestamp_values[0]), int(timestamp_values[-1])
        if previous_id is not None:
            if first_id <= previous_id:
                raise ValueError(f"aggregate-trade batch overlap or regression in {month}")
            missing_id_total += first_id - previous_id - 1
        if id_differences.size:
            missing_id_total += int(np.sum(id_differences[id_differences > 1] - 1))
        if previous_timestamp_us is not None and first_timestamp_us < previous_timestamp_us:
            raise ValueError(f"timestamp batch regression in {month}")
        if batch_number == 0 and first_id != expected_first_id:
            raise ValueError(f"first aggregate-trade ID mismatch in {month}")

        price_float = pc.cast(table["price"], pa.float64())
        size_int = int(BIN_SIZE_USD)
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
        for row in grouped.to_pylist():
            key = (int(row["bucket"]), int(row["bin_floor"]))
            current = footprints.setdefault(key, [Decimal(), Decimal(), 0])
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
                if incoming["first_aggregate_trade_id"] <= current["last_aggregate_trade_id"]:
                    raise ValueError(f"bar boundary aggregate-trade overlap in {month}")
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
                "first_timestamp": str(times[0].as_py()),
                "last_timestamp": str(times[-1].as_py()),
                "missing_aggregate_trade_id_count_to_date": missing_id_total,
            }
        )
        previous_id, previous_timestamp_us = last_id, last_timestamp_us
        row_total += batch.num_rows

    required_rows = (
        expected_row_count
        if expected_row_count is not None
        else expected_last_id - expected_first_id + 1
    )
    if previous_id != expected_last_id or row_total != required_rows:
        raise ValueError(f"aggregate-trade endpoint or row-count mismatch in {month}")

    footprint_rows: list[dict[str, Any]] = []
    bar_deltas: dict[int, Decimal] = defaultdict(Decimal)
    for (bucket, floor_int), (buy, sell, count) in sorted(footprints.items()):
        start = _utc_from_epoch_minute(bucket)
        floor = Decimal(floor_int)
        footprint_rows.append(
            {
                "bar_start_utc": start,
                "bar_end_utc": start + timedelta(minutes=5),
                "month": month,
                "bin_size_usd": BIN_SIZE_USD,
                "bin_floor": floor,
                "bin_upper_exclusive": floor + BIN_SIZE_USD,
                "buy_volume_btc": buy,
                "sell_volume_btc": sell,
                "total_volume_btc": buy + sell,
                "delta_btc": buy - sell,
                "trade_count": count,
            }
        )
        bar_deltas[bucket] += buy - sell
    if sum(row["trade_count"] for row in footprint_rows) != row_total:
        raise AssertionError(f"{month} $50 footprint does not reconcile to source rows")

    bar_rows: list[dict[str, Any]] = []
    current_day: str | None = None
    cumulative_volume = cumulative_notional = session_delta = Decimal()
    for bucket, row in sorted(bars.items()):
        start = _utc_from_epoch_minute(bucket)
        day = start.date().isoformat()
        if day != current_day:
            current_day = day
            cumulative_volume = cumulative_notional = session_delta = Decimal()
        delta = bar_deltas[bucket]
        session_delta += delta
        cumulative_volume += row["volume"]
        cumulative_notional += row["notional"]
        bar_rows.append(
            {
                "bar_start_utc": start,
                "bar_end_utc": start + timedelta(minutes=5),
                "session_date": day,
                "utc_session": day,
                "month": month,
                **row,
                "buy_volume_btc": (row["volume"] + delta) / Decimal("2"),
                "sell_volume_btc": (row["volume"] - delta) / Decimal("2"),
                "total_volume_btc": row["volume"],
                "delta_btc": delta,
                "cumulative_session_delta_btc": session_delta,
                "cumulative_volume_delta_btc": session_delta,
                "daily_vwap": cumulative_notional / cumulative_volume,
            }
        )
    if sum(row["trade_count"] for row in bar_rows) != row_total:
        raise AssertionError(f"{month} five-minute bars do not reconcile to source rows")
    return footprint_rows, bar_rows, diagnostics


def build_v4_phase_footprint_dataset(
    source_manifest_path: str | Path,
    cache_root: str | Path,
    *,
    phase: str,
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validation = validate_v4_source_manifest(
        source_manifest_path, phase=phase, verify_archives=False
    )
    if not validation["valid"]:
        raise ValueError("invalid V4 normalized source: " + "; ".join(validation["errors"]))
    source: MonthlyAggregateTradeManifest = validation["manifest"]
    source_path = Path(source_manifest_path).resolve()
    identity = {
        "builder_version": V4_BAR_BUILDER_VERSION,
        "phase": phase,
        "normalized_dataset_hash": source.normalized_dataset_hash,
        "normalized_manifest_hash": source.manifest_hash,
        "normalized_manifest_file_sha256": sha256_file(source_path),
        "symbol": "BTCUSDT",
        "months": list(phase_months(phase)),
        "bar_interval": "5m",
        "bin_sizes_usd": [str(BIN_SIZE_USD)],
        "bin_interval": "HALF_OPEN_FLOOR_FROM_RAW_PRICE",
        "aggressor_rule": "BUY_IFF_BUYER_IS_MAKER_FALSE",
        "daily_vwap_reset": "UTC_MIDNIGHT",
        "session_delta_reset": "UTC_MIDNIGHT",
        "lookahead": False,
    }
    dataset_id = sha256_value(identity)
    root = Path(cache_root).resolve() / "BTCUSDT" / phase.lower() / dataset_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"immutable V4 footprint identity collision: {manifest_path}")
        report = validate_v4_footprint_dataset(manifest_path)
        if not report["valid"]:
            raise ValueError("invalid existing V4 footprint: " + "; ".join(report["errors"]))
        return existing
    context = ArtifactContext(
        run_id=dataset_id,
        dataset_hash=source.normalized_dataset_hash,
        source_manifest_hash=source.manifest_hash,
        specification_hash=sha256_value(identity),
        parameter_hash=sha256_value({"bin_size_usd": str(BIN_SIZE_USD), "bar_interval": "5m"}),
        code_hash=sha256_value({"builder_version": V4_BAR_BUILDER_VERSION}),
        evidence_label=SELECTION_EVIDENCE,
        timestamp=source.retrieved_at.isoformat(),
    )
    files: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    total_bars = total_footprints = total_trades = 0
    for partition in source.partitions:
        parquet_path = source_path.parent / partition.file_name
        footprints, bars, diagnostics = _aggregate_month(
            pq.ParquetFile(parquet_path),
            month=partition.month,
            expected_first_id=partition.first_aggregate_trade_id,
            expected_last_id=partition.last_aggregate_trade_id,
            expected_row_count=partition.row_count,
            batch_size=batch_size,
        )
        for kind, rows in (("bars", bars), ("footprints", footprints)):
            target = root / kind / f"{partition.month}.parquet"
            context.write_parquet(target, rows)
            item = {
                "kind": kind,
                "month": partition.month,
                "bin_size_usd": None if kind == "bars" else str(BIN_SIZE_USD),
                "relative_path": target.relative_to(root).as_posix(),
                "row_count": len(rows),
                "trade_count": sum(int(row["trade_count"]) for row in rows),
                "sha256": sha256_file(target),
                "schema": str(pq.ParquetFile(target).schema_arrow),
            }
            files.append(item)
        total_bars += len(bars)
        total_footprints += len(footprints)
        total_trades += sum(int(row["trade_count"]) for row in bars)
        boundaries.extend(diagnostics)
        del footprints, bars
    if total_trades != source.row_count:
        raise AssertionError("V4 bars do not reconcile to normalized aggregate rows")
    footprint_trade_count = sum(
        item["trade_count"] for item in files if item["kind"] == "footprints"
    )
    if footprint_trade_count != source.row_count:
        raise AssertionError("V4 footprints do not reconcile to normalized aggregate rows")
    content_hash = sha256_value({item["relative_path"]: item["sha256"] for item in files})
    output = context.envelope(
        {
            "identity": identity,
            "phase": phase,
            "footprint_dataset_hash": content_hash,
            "footprint_root": str(root),
            "source_manifest_path": str(source_path),
            "source_row_count": source.row_count,
            "streamed_trade_count": total_trades,
            "five_minute_bar_count": total_bars,
            "footprint_row_count": total_footprints,
            "batch_size": batch_size,
            "batch_count": len(boundaries),
            "batch_boundaries": boundaries,
            "parquet_files": files,
            "raw_aggregate_rows_transmitted": False,
            "valid": True,
        }
    )
    write_bytes_once(
        manifest_path,
        json.dumps(output, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n",
    )
    report = validate_v4_footprint_dataset(manifest_path)
    if not report["valid"]:
        raise RuntimeError("created V4 footprint failed validation: " + "; ".join(report["errors"]))
    return output


def validate_v4_footprint_dataset(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if manifest_path.name != "manifest.json":
        manifest_path = manifest_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)], "manifest_path": str(manifest_path)}
    root = manifest_path.parent
    errors: list[str] = []
    hashes: dict[str, str] = {}
    trade_counts: dict[str, int] = defaultdict(int)
    row_counts: dict[str, int] = defaultdict(int)
    if manifest.get("identity", {}).get("bin_sizes_usd") != [str(BIN_SIZE_USD)]:
        errors.append("V4 footprint must contain exactly the fixed $50 bins")
    for item in manifest.get("parquet_files", []):
        target = root / item["relative_path"]
        try:
            actual_hash = sha256_file(target)
            actual_rows = pq.ParquetFile(target).metadata.num_rows
            if actual_hash != item["sha256"]:
                raise ValueError("SHA-256 mismatch")
            if actual_rows != item["row_count"]:
                raise ValueError("row-count mismatch")
            hashes[item["relative_path"]] = actual_hash
            trade_counts[item["kind"]] += int(item["trade_count"])
            row_counts[item["kind"]] += actual_rows
        except (OSError, ValueError) as exc:
            errors.append(f"{target}: {exc}")
    source_rows = int(manifest.get("source_row_count", -1))
    if trade_counts["bars"] != source_rows or trade_counts["footprints"] != source_rows:
        errors.append("bar/footprint trade counts do not reconcile to source rows")
    if row_counts["bars"] != int(manifest.get("five_minute_bar_count", -1)):
        errors.append("five-minute bar row total mismatch")
    if row_counts["footprints"] != int(manifest.get("footprint_row_count", -1)):
        errors.append("footprint row total mismatch")
    if sha256_value(hashes) != manifest.get("footprint_dataset_hash"):
        errors.append("content-addressed V4 footprint hash mismatch")
    temporary = [
        item.name
        for item in root.rglob("*")
        if item.is_file() and (item.name.endswith(".tmp") or item.name.endswith(".part"))
    ]
    if temporary:
        errors.append(f"temporary output is not admissible: {temporary}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_path": str(manifest_path),
        "footprint_dataset_hash": manifest.get("footprint_dataset_hash"),
        "source_row_count": source_rows,
        "rows": dict(row_counts),
        "trade_counts": dict(trade_counts),
        "parquet_sha256": hashes,
        "temporary_outputs": temporary,
    }


def _resolve_manifest(manifest: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest, dict):
        return manifest, Path(manifest["footprint_root"])
    path = Path(manifest).resolve()
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path.parent


def load_v4_bars(manifest: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    raw, root = _resolve_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for item in raw["parquet_files"]:
        if item["kind"] == "bars":
            rows.extend(pq.read_table(root / item["relative_path"]).to_pylist())
    return rows


def load_v4_footprints(manifest: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    raw, root = _resolve_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for item in raw["parquet_files"]:
        if item["kind"] == "footprints":
            rows.extend(pq.read_table(root / item["relative_path"]).to_pylist())
    return rows


build_v4_bar_dataset = build_v4_phase_footprint_dataset
build_v4_footprint_dataset = build_v4_phase_footprint_dataset
validate_v4_bar_dataset = validate_v4_footprint_dataset
acquire_and_normalize_v4_data = acquire_and_normalize_v4_phase
