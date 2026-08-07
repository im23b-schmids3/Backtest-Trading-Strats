"""Manifest-only contract checking.  This module never opens partition payloads."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    pass


EXPECTED_SCHEMA = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_manifest_hash(data: dict[str, Any]) -> str:
    value = dict(data)
    value.pop("manifestHash", None)
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash an archive before a caller is allowed to decode it."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _partition_path(part: dict[str, Any], root: Path) -> Path:
    """Resolve only the content-addressed location declared by the manifest."""
    relative = part.get("path")
    filename = part.get("file")
    if not isinstance(relative, str) or not isinstance(filename, str) or not relative or not filename:
        raise ManifestError("V2_MANIFEST_PARTITION_PATH_INVALID")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or Path(filename).name != filename:
        raise ManifestError("V2_MANIFEST_PARTITION_PATH_INVALID")
    expected = Path("binance_usdm") / part["symbol"] / "1m" / part["partition"] / part["sha256"]
    if rel.as_posix() != expected.as_posix() or filename not in {"source.zip", "source.json"}:
        raise ManifestError("V2_MANIFEST_PARTITION_PATH_INVALID")
    resolved_root = root.resolve()
    target = (resolved_root / rel / filename).resolve()
    if resolved_root not in target.parents:
        raise ManifestError("V2_MANIFEST_PARTITION_PATH_INVALID")
    return target


def _stamp(value: object) -> datetime:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("V2_MANIFEST_TIMESTAMP_INVALID") from exc
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ManifestError("V2_MANIFEST_TIMESTAMP_NOT_UTC")
    return stamp.astimezone(timezone.utc)


def load_manifest(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file():
        raise ManifestError("V2_MANIFEST_REQUIRED")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifestHash") != canonical_manifest_hash(data):
        raise ManifestError("V2_MANIFEST_SELF_HASH_MISMATCH")
    return path, data


def verify_manifest(path_value: str | Path, *, symbol: str | None = None, partition_root: str | Path | None = None) -> dict[str, Any]:
    path, data = load_manifest(path_value)
    expected = symbol or data.get("symbol")
    if expected not in {"ETHUSDT", "BTCUSDT"} or data.get("symbol") != expected:
        raise ManifestError("V2_MANIFEST_SYMBOL_MISMATCH")
    if data.get("immutable") is not True or data.get("exchange") != "BINANCE" or data.get("instrumentType") != "USD_M_PERPETUAL" or data.get("interval") != "1m":
        raise ManifestError("V2_MANIFEST_CONTRACT_MISMATCH")
    if data.get("schema") != EXPECTED_SCHEMA or data.get("schemaHash") != hashlib.sha256(canonical(EXPECTED_SCHEMA).encode("utf-8")).hexdigest():
        raise ManifestError("V2_MANIFEST_SCHEMA_MISMATCH")
    if data.get("chronology") != {"development": "[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)", "holdout": "[2025-01-01T00:00:00Z, coverage end)", "holdoutStrategyAccess": False}:
        raise ManifestError("V2_MANIFEST_CHRONOLOGY_MISMATCH")
    if data.get("integrity") != {"duplicateOrNonIncreasingCount": 0, "invalidOhlcCount": 0, "nonPositivePriceCount": 0, "nullOhlcvCount": 0, "unresolvedGapCount": 0}:
        raise ManifestError("V2_MANIFEST_INTEGRITY_MISMATCH")
    parts = data.get("partitions")
    if not isinstance(parts, list) or len(parts) != 61:
        raise ManifestError("V2_MANIFEST_PARTITION_COUNT_MISMATCH")
    if not _is_digest(data.get("aggregateDatasetHash")):
        raise ManifestError("V2_MANIFEST_DATASET_HASH_MISMATCH")
    previous = None; identity = []
    total = 0
    for part in parts:
        if not isinstance(part, dict):
            raise ManifestError("V2_MANIFEST_PARTITION_CONTRACT_MISMATCH")
        first, final = _stamp(part.get("firstUtcTimestamp")), _stamp(part.get("finalUtcTimestamp"))
        if first.second or first.microsecond or final.second or final.microsecond or previous is not None and first != previous + timedelta(minutes=1):
            raise ManifestError("V2_MANIFEST_PARTITION_ORDER_OR_GAP")
        if part.get("symbol") != expected or part.get("exchange") != "BINANCE" or part.get("instrumentType") != "USD_M_PERPETUAL" or part.get("interval") != "1m" or part.get("sealed") is not True or part.get("missingMinuteDistribution") != []:
            raise ManifestError("V2_MANIFEST_PARTITION_CONTRACT_MISMATCH")
        count = int(part.get("rowCount", -1))
        expected_partition = first.strftime("%Y-%m") if part.get("partitionKind") == "MONTH" else first.date().isoformat()
        expected_file = "source.zip" if part.get("partitionKind") == "MONTH" else "source.json"
        if part.get("partitionKind") not in {"MONTH", "DAY"} or count <= 0 or count != int((final - first).total_seconds() // 60) + 1 or not _is_digest(part.get("sha256")) or part.get("partition") != expected_partition or part.get("file") != expected_file or not isinstance(part.get("byteSize"), int) or part["byteSize"] <= 0:
            raise ManifestError("V2_MANIFEST_PARTITION_ROW_OR_HASH_MISMATCH")
        # Validate the declared content-addressed path even in manifest-only mode.
        _partition_path(part, Path.cwd())
        total += count
        identity.append({"path": part["path"], "sha256": part["sha256"]})
        previous = final
    if total != int(data.get("rowCount", -1)) or _stamp(data["coverage"]["startInclusive"]) != _stamp(parts[0]["firstUtcTimestamp"]) or _stamp(data["coverage"]["endExclusive"]) != _stamp(parts[-1]["finalUtcTimestamp"]) + timedelta(minutes=1):
        raise ManifestError("V2_MANIFEST_COVERAGE_OR_COUNT_MISMATCH")
    if hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest() != data["aggregateDatasetHash"]:
        raise ManifestError("V2_MANIFEST_DATASET_HASH_MISMATCH")
    payloads_verified = False
    if partition_root is not None:
        root = Path(partition_root)
        if not root.is_absolute() or not root.is_dir():
            raise ManifestError("V2_PARTITION_ROOT_REQUIRED")
        for part in parts:
            archive = _partition_path(part, root)
            if not archive.is_file():
                raise ManifestError("V2_SEALED_PARTITION_PAYLOAD_MISSING")
            if _sha256_file(archive) != part["sha256"]:
                raise ManifestError("V2_SEALED_PARTITION_HASH_MISMATCH")
        payloads_verified = True
    return {"manifest_path": str(path), "validated": True, "rows_read": False, "partition_archives_verified": payloads_verified, "symbol": expected, "partition_count": 61, "holdout_status": "LOCKED_NOT_OPENED"}
