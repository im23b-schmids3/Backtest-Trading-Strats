from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{period}.zip"
REST = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
START = date(2022, 1, 1)
SCHEMA = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]


class AcquisitionError(RuntimeError):
    pass


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def _months(start: date, end: date) -> Iterable[date]:
    current = start.replace(day=1)
    while current < end.replace(day=1):
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def _days(start: date, end_inclusive: date) -> Iterable[date]:
    while start <= end_inclusive:
        yield start
        start += timedelta(days=1)


@dataclass(frozen=True)
class CheckedRows:
    first: str
    final: str
    rows: int
    missing: list[dict[str, object]]


class BinanceUsdmKlineAcquirer:
    """Strict, stdlib-only, V2-scoped public archive/REST acquirer."""

    def __init__(self, root: Path, fetch: Callable[[str], bytes] | None = None) -> None:
        self.root = root.resolve()
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "fib-prospective-v2-acquirer/1"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise AcquisitionError(f"network error obtaining {url}: {exc.reason}") from exc
        except urllib.error.HTTPError as exc:
            raise AcquisitionError(f"HTTP {exc.code} obtaining {url}") from exc

    @staticmethod
    def _validate(rows: list[list[str]], start: datetime, end: datetime) -> CheckedRows:
        if not rows:
            raise AcquisitionError("empty kline response")
        expected = start
        prior: datetime | None = None
        missing: list[dict[str, object]] = []
        for row in rows:
            if len(row) < 6:
                raise AcquisitionError("kline schema has fewer than six columns")
            try:
                stamp = _utc(int(row[0]))
                values = [Decimal(str(row[n])) for n in range(1, 6)]
            except (ValueError, InvalidOperation) as exc:
                raise AcquisitionError("kline has invalid timestamp or OHLCV") from exc
            if stamp.second or stamp.microsecond or stamp.tzinfo != timezone.utc:
                raise AcquisitionError(f"minute alignment failure at {stamp.isoformat()}")
            if stamp < start or stamp >= end:
                raise AcquisitionError(f"kline outside requested partition at {stamp.isoformat()}")
            if any(not value.is_finite() for value in values):
                raise AcquisitionError(f"null/non-finite OHLCV at {stamp.isoformat()}")
            o, h, l, c, v = values
            if min(o, h, l, c) <= 0 or v < 0:
                raise AcquisitionError(f"non-positive price or negative volume at {stamp.isoformat()}")
            if h < max(o, l, c) or l > min(o, h, c):
                raise AcquisitionError(f"invalid OHLC at {stamp.isoformat()}")
            if prior is not None and stamp <= prior:
                raise AcquisitionError(f"duplicate/non-increasing timestamp at {stamp.isoformat()}")
            if stamp > expected:
                missing.append({"classification": "UNRESOLVED_GAP", "start": expected.isoformat().replace("+00:00", "Z"), "endExclusive": stamp.isoformat().replace("+00:00", "Z"), "missingMinutes": int((stamp - expected).total_seconds() // 60)})
            expected, prior = stamp + timedelta(minutes=1), stamp
        if rows and _utc(int(rows[0][0])) > start:
            # The first distribution item was inserted above only after a row;
            # keep its classification explicit for callers and manifests.
            pass
        if expected < end:
            missing.append({"classification": "UNRESOLVED_GAP", "start": expected.isoformat().replace("+00:00", "Z"), "endExclusive": end.isoformat().replace("+00:00", "Z"), "missingMinutes": int((end - expected).total_seconds() // 60)})
        if missing:
            raise AcquisitionError("missing-minute distribution: " + json.dumps(missing, separators=(",", ":")))
        return CheckedRows(_utc(int(rows[0][0])).isoformat().replace("+00:00", "Z"), _utc(int(rows[-1][0])).isoformat().replace("+00:00", "Z"), len(rows), missing)

    @staticmethod
    def _archive_rows(blob: bytes) -> list[list[str]]:
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(names) != 1:
                    raise AcquisitionError("archive must contain exactly one CSV")
                with archive.open(names[0]) as source:
                    rows = list(csv.reader(io.TextIOWrapper(source, encoding="utf-8", newline="")))
                    if rows and rows[0] and rows[0][0] == "open_time":
                        expected = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
                        if rows[0] != expected:
                            raise AcquisitionError("archive CSV header/schema is unexpected")
                        rows = rows[1:]
                    return rows
        except zipfile.BadZipFile as exc:
            raise AcquisitionError("archive is not a valid ZIP") from exc

    @staticmethod
    def _checksum(checksum: bytes, blob: bytes) -> None:
        expected = checksum.decode("utf-8").strip().split()[0].lower()
        if len(expected) != 64 or expected != _hash(blob):
            raise AcquisitionError("archive checksum mismatch")

    def _seal(self, symbol: str, period: str, kind: str, url: str, raw: bytes, checked: CheckedRows, checksum_url: str | None = None) -> dict[str, object]:
        digest = _hash(raw)
        base = self.root / "binance_usdm" / symbol / "1m" / period
        target = base / digest
        if base.exists():
            existing = [child.name for child in base.iterdir() if child.is_dir() and not child.name.startswith(".staging-")]
            if existing and existing != [digest]:
                raise AcquisitionError(f"conflicting sealed partition hash for {symbol} {period}: {existing}")
        metadata = {"exchange": "BINANCE", "instrumentType": "USD_M_PERPETUAL", "symbol": symbol, "interval": "1m", "partition": period, "partitionKind": kind, "remoteSourceUrl": url, "remoteChecksumUrl": checksum_url, "firstUtcTimestamp": checked.first, "finalUtcTimestamp": checked.final, "byteSize": len(raw), "sha256": digest, "rowCount": checked.rows, "missingMinuteDistribution": checked.missing, "sealed": True}
        content = _canonical(metadata)
        if target.exists():
            meta_path, raw_path = target / "metadata.json", target / ("source.zip" if kind == "MONTH" else "source.json")
            if not meta_path.is_file() or not raw_path.is_file() or raw_path.read_bytes() != raw or meta_path.read_bytes() != content:
                raise AcquisitionError(f"conflicting existing sealed partition: {target}")
            return {**metadata, "path": str(target.relative_to(self.root)).replace("\\", "/"), "file": raw_path.name}
        base.mkdir(parents=True, exist_ok=True)
        staging = base / f".staging-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            raw_name = "source.zip" if kind == "MONTH" else "source.json"
            (staging / raw_name).write_bytes(raw)
            (staging / "metadata.json").write_bytes(content)
            os.replace(staging, target)
        except Exception:
            if staging.exists(): shutil.rmtree(staging)
            raise
        return {**metadata, "path": str(target.relative_to(self.root)).replace("\\", "/"), "file": raw_name}

    def _reusable(self, symbol: str, period: str, kind: str) -> dict[str, object] | None:
        base = self.root / "binance_usdm" / symbol / "1m" / period
        if not base.is_dir(): return None
        candidates = [child for child in base.iterdir() if child.is_dir() and not child.name.startswith(".staging-")]
        if len(candidates) != 1: raise AcquisitionError(f"conflicting sealed partition hash for {symbol} {period}")
        target = candidates[0]; raw_name = "source.zip" if kind == "MONTH" else "source.json"
        try: metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: raise AcquisitionError(f"invalid sealed partition metadata: {target}") from exc
        raw = target / raw_name
        if metadata.get("sha256") != target.name or not raw.is_file() or _hash(raw.read_bytes()) != target.name:
            raise AcquisitionError(f"sealed partition hash mismatch: {target}")
        if metadata.get("symbol") != symbol or metadata.get("partition") != period or metadata.get("partitionKind") != kind or metadata.get("sealed") is not True:
            raise AcquisitionError(f"sealed partition identity mismatch: {target}")
        return {**metadata, "path": str(target.relative_to(self.root)).replace("\\", "/"), "file": raw_name}

    def acquire_partition(self, symbol: str, partition_start: date, latest_complete: date) -> dict[str, object]:
        month_end = (date(partition_start.year + (partition_start.month == 12), partition_start.month % 12 + 1, 1))
        if partition_start < latest_complete.replace(day=1):
            period = partition_start.strftime("%Y-%m")
            if reusable := self._reusable(symbol, period, "MONTH"): return reusable
            url = ARCHIVE.format(symbol=symbol, period=period)
            blob, checksum_url = self.fetch(url), url + ".CHECKSUM"
            self._checksum(self.fetch(checksum_url), blob)
            checked = self._validate(self._archive_rows(blob), datetime.combine(partition_start, datetime.min.time(), timezone.utc), datetime.combine(month_end, datetime.min.time(), timezone.utc))
            return self._seal(symbol, period, "MONTH", url, blob, checked, checksum_url)
        result: list[dict[str, object]] = []
        for day in _days(partition_start, latest_complete):
            if reusable := self._reusable(symbol, day.isoformat(), "DAY"):
                result.append(reusable); continue
            start = datetime.combine(day, datetime.min.time(), timezone.utc)
            end = start + timedelta(days=1)
            query = urllib.parse.urlencode({"symbol": symbol, "interval": "1m", "startTime": int(start.timestamp() * 1000), "endTime": int(end.timestamp() * 1000) - 1, "limit": 1500})
            url = REST + "?" + query
            raw = self.fetch(url)
            try: rows = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise AcquisitionError("REST response is not JSON") from exc
            if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows): raise AcquisitionError("REST kline response schema is invalid")
            result.append(self._seal(symbol, day.isoformat(), "DAY", url, raw, self._validate(rows, start, end)))
        return {"dailyPartitions": result}

    def acquire_symbol(self, symbol: str, latest_complete: date) -> list[dict[str, object]]:
        if symbol not in SYMBOLS or latest_complete < START: raise AcquisitionError("unsupported symbol or invalid coverage")
        partitions: list[dict[str, object]] = []
        next_month = date(latest_complete.year + (latest_complete.month == 12), latest_complete.month % 12 + 1, 1)
        for month in _months(START, next_month):
            item = self.acquire_partition(symbol, month, latest_complete)
            partitions.extend(item.get("dailyPartitions", [item]))
        return partitions

    def manifest(self, symbol: str, partitions: list[dict[str, object]], latest_complete: date, docs_root: Path) -> Path:
        if not partitions: raise AcquisitionError("cannot manifest absent data")
        identity = [{"path": p["path"], "sha256": p["sha256"]} for p in partitions]
        schema_hash = _hash(_canonical(SCHEMA))
        payload = {"source": "Binance USD-M perpetual futures 1m klines", "exchange": "BINANCE", "instrumentType": "USD_M_PERPETUAL", "symbol": symbol, "interval": "1m", "immutable": True, "schema": SCHEMA, "schemaHash": schema_hash, "coverage": {"startInclusive": "2022-01-01T00:00:00Z", "endExclusive": (latest_complete + timedelta(days=1)).isoformat() + "T00:00:00Z"}, "chronology": {"development": "[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)", "holdout": "[2025-01-01T00:00:00Z, coverage end)", "holdoutStrategyAccess": False}, "partitions": partitions, "aggregateDatasetHash": _hash(_canonical(identity)), "rowCount": sum(int(p["rowCount"]) for p in partitions), "integrity": {"unresolvedGapCount": 0, "duplicateOrNonIncreasingCount": 0, "nullOhlcvCount": 0, "invalidOhlcCount": 0, "nonPositivePriceCount": 0}}
        payload["manifestHash"] = _hash(_canonical(payload))
        path = docs_root / f"{symbol}_1M" / "manifest.json"; path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        if path.exists() and path.read_bytes() != content: raise AcquisitionError(f"immutable manifest collision: {path}")
        if not path.exists(): path.write_bytes(content)
        return path

    def validate_manifest(self, path: Path) -> dict[str, object]:
        """Verify immutable identities and chronology without strategy access."""
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcquisitionError(f"invalid manifest JSON: {path}") from exc
        claimed = manifest.pop("manifestHash", None)
        if claimed != _hash(_canonical(manifest)):
            raise AcquisitionError(f"manifest hash mismatch: {path}")
        if manifest.get("schemaHash") != _hash(_canonical(SCHEMA)) or manifest.get("immutable") is not True:
            raise AcquisitionError(f"manifest schema/immutability failure: {path}")
        expected = datetime(2022, 1, 1, tzinfo=timezone.utc)
        identity = []
        for item in manifest.get("partitions", []):
            part = self.root / str(item["path"])
            raw = part / str(item["file"])
            metadata = part / "metadata.json"
            if not raw.is_file() or not metadata.is_file() or _hash(raw.read_bytes()) != item["sha256"]:
                raise AcquisitionError(f"partition hash mismatch: {part}")
            if json.loads(metadata.read_text(encoding="utf-8")) != {key: item[key] for key in item if key != "path" and key != "file"}:
                raise AcquisitionError(f"partition metadata mismatch: {part}")
            first = datetime.fromisoformat(str(item["firstUtcTimestamp"]).replace("Z", "+00:00"))
            final = datetime.fromisoformat(str(item["finalUtcTimestamp"]).replace("Z", "+00:00"))
            if first != expected or final < first:
                raise AcquisitionError(f"partition chronology mismatch: {part}")
            expected = final + timedelta(minutes=1)
            identity.append({"path": item["path"], "sha256": item["sha256"]})
        if expected.isoformat().replace("+00:00", "Z") != manifest["coverage"]["endExclusive"]:
            raise AcquisitionError(f"manifest coverage chronology mismatch: {path}")
        if _hash(_canonical(identity)) != manifest.get("aggregateDatasetHash"):
            raise AcquisitionError(f"aggregate dataset hash mismatch: {path}")
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire only V2 Binance USD-M 1m raw partitions")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--latest-complete-day", help="UTC YYYY-MM-DD; defaults to yesterday UTC")
    args = parser.parse_args(argv)
    root = Path(args.repository_root).resolve()
    latest = date.fromisoformat(args.latest_complete_day) if args.latest_complete_day else datetime.now(timezone.utc).date() - timedelta(days=1)
    acquirer = BinanceUsdmKlineAcquirer(root / "data" / "fib_prospective_v2")
    try:
        paths = {symbol: acquirer.manifest(symbol, acquirer.acquire_symbol(symbol, latest), latest, root / "docs" / "research_pipeline" / "fib_prospective_v2" / "data-contracts") for symbol in SYMBOLS}
        for path in paths.values(): acquirer.validate_manifest(path)
    except AcquisitionError as exc:
        print(json.dumps({"status": "DATA_ACQUISITION_BLOCKED", "blocker": str(exc), "v2BacktestStarted": False, "holdoutStrategyAccessed": False}))
        return 2
    print(json.dumps({"status": "READY_FOR_V2_IMPLEMENTATION", "btcManifest": str(paths["BTCUSDT"]), "ethManifest": str(paths["ETHUSDT"]), "v2BacktestStarted": False, "holdoutStrategyAccessed": False}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
