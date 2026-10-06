"""Download only the frozen Sato missing overnight OHLCV-1h quote plan."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1h"
STYPE_IN = "raw_symbol"
EXPECTED_PLAN_REQUESTS = 54
EXPECTED_PLAN_BARS = 103
EXPECTED_QUOTED_COST_USD = "0.001020655016"
FROZEN_REQUEST_SET_SHA256 = "5e2a8e32066ac04fe52c446355023ebf5430d6fec31ca222d03b6aad1d19a127"
QUOTE_PLAN_PATH = Path("research_runs/CMEOrderflow_SATO_ES_OVERNIGHT_OHLCV1H_QUOTE_PREP_V1/missing-overnight-hourly-quote-plan.json")
DATA_ROOT = Path("data/databento/sato-es-overnight-ohlcv1h-patch-v1")
MANIFEST_PATH = Path("research_runs/CMEOrderflow_SATO_ES_OVERNIGHT_OHLCV1H_DOWNLOAD_V1/download-manifest.json")
CSV_FIELDS = ("ts_event", "symbol", "open", "high", "low", "close", "volume")
FINGERPRINT_FIELDS = (
    "CURRENT_RTH_DATE", "PATCH_START_UTC", "PATCH_END_UTC", "RAW_ES_SYMBOL", "STYPE_IN",
    "DATASET", "SCHEMA", "EXPECTED_HOURLY_BAR_COUNT", "EXPECTED_HOURLY_BAR_STARTS_UTC",
)


class DownloadError(RuntimeError):
    """Raised when frozen-plan download or verification fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_set_sha256(rows: list[dict[str, Any]]) -> str:
    frozen = [{key: row[key] for key in FINGERPRINT_FIELDS} for row in rows]
    payload = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"cannot read frozen quote plan: {path}: {exc}") from exc
    rows = plan.get("requests")
    if plan.get("status") != "PREPARED_LOCAL_EVIDENCE_NO_QUOTE_EXECUTED":
        raise DownloadError("DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: unexpected quote-plan status")
    if plan.get("dataset") != DATASET or plan.get("schema") != SCHEMA or plan.get("stype_in") != STYPE_IN:
        raise DownloadError("DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: dataset/schema/stype differs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PLAN_REQUESTS:
        raise DownloadError("DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: expected exactly 54 requests")
    if sum(int(row.get("EXPECTED_HOURLY_BAR_COUNT", -1)) for row in rows) != EXPECTED_PLAN_BARS:
        raise DownloadError("DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: expected exactly 103 quoted bars")
    for row in rows:
        if row.get("STATUS") != "APPROVED_FOR_COST_QUOTE" or row.get("SYMBOL_STATUS") != "VERIFIED_LOCAL_NATIVE_RAW_SYMBOL":
            raise DownloadError(f"DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: unapproved or ambiguous request {row.get('CURRENT_RTH_DATE')}")
        if row.get("DATASET") != DATASET or row.get("SCHEMA") != SCHEMA or row.get("STYPE_IN") != STYPE_IN:
            raise DownloadError(f"DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: request constants differ for {row.get('CURRENT_RTH_DATE')}")
        starts = row.get("EXPECTED_HOURLY_BAR_STARTS_UTC")
        if not isinstance(starts, list) or len(starts) != int(row["EXPECTED_HOURLY_BAR_COUNT"]):
            raise DownloadError(f"DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: bar list/count differs for {row.get('CURRENT_RTH_DATE')}")
        if row.get("PATCH_END_UTC") != row.get("SOURCE_FILE_START_UTC"):
            raise DownloadError(f"DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: native boundary differs for {row.get('CURRENT_RTH_DATE')}")
        if row.get("PATCH_END_UTC") != f"{row.get('CURRENT_RTH_DATE')}T00:00:00Z":
            raise DownloadError(f"DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: patch end differs for {row.get('CURRENT_RTH_DATE')}")
    fingerprint = request_set_sha256(rows)
    if fingerprint != FROZEN_REQUEST_SET_SHA256:
        raise DownloadError(
            "DOWNLOAD_ABORTED_QUOTE_PLAN_MISMATCH: frozen request fingerprint differs "
            f"(actual {fingerprint})"
        )
    return plan


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, str):
        if value.endswith("+00:00"):
            value = value[:-6] + "Z"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DownloadError(f"invalid timestamp string: {value}") from exc
        if parsed.tzinfo is None:
            raise DownloadError("DBN bar timestamp is timezone-naive")
        return value
    if isinstance(value, int):
        seconds, nanos = divmod(value, 1_000_000_000)
        instant = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return instant.strftime("%Y-%m-%dT%H:%M:%S") + (f".{nanos:09d}" if nanos else "") + "Z"
    if hasattr(value, "value") and isinstance(value.value, int):
        return _iso_timestamp(int(value.value))
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise DownloadError(f"unsupported timestamp representation: {type(value).__name__}")
    if value.tzinfo is None:
        raise DownloadError("DBN bar timestamp is timezone-naive")
    value = value.astimezone(timezone.utc)
    if value.microsecond:
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_bars(rows: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, str]]:
    """Validate OHLCV rows and return normalized, lossless decimal strings."""
    expected_times = request["EXPECTED_HOURLY_BAR_STARTS_UTC"]
    if len(rows) != int(request["EXPECTED_HOURLY_BAR_COUNT"]):
        raise DownloadError(
            f"bar count mismatch for {request['CURRENT_RTH_DATE']}: "
            f"expected {request['EXPECTED_HOURLY_BAR_COUNT']}, received {len(rows)}"
        )
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        timestamp = _iso_timestamp(row["ts_event"])
        if timestamp not in expected_times:
            raise DownloadError(f"out-of-window/unexpected hourly timestamp {timestamp} for {request['CURRENT_RTH_DATE']}")
        if timestamp in seen:
            raise DownloadError(f"duplicate hourly timestamp {timestamp} for {request['CURRENT_RTH_DATE']}")
        seen.add(timestamp)
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed_timestamp.minute or parsed_timestamp.second or parsed_timestamp.microsecond:
            raise DownloadError(f"timestamp is not hour-aligned: {timestamp}")
        prices: dict[str, Decimal] = {}
        for name in ("open", "high", "low", "close"):
            try:
                value = Decimal(str(row[name]))
            except (KeyError, InvalidOperation) as exc:
                raise DownloadError(f"invalid {name} in bar {timestamp}") from exc
            if not value.is_finite():
                raise DownloadError(f"non-finite {name} in bar {timestamp}")
            prices[name] = value
            if (value / Decimal("0.25")) % 1:
                raise DownloadError(f"ES {name} price is off the 0.25 tick grid: {value} at {timestamp}")
        if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
            raise DownloadError(f"OHLC high invariant failed at {timestamp}")
        if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
            raise DownloadError(f"OHLC low invariant failed at {timestamp}")
        try:
            volume = int(row["volume"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DownloadError(f"invalid volume at {timestamp}") from exc
        if volume < 0 or str(volume) != str(row["volume"]):
            raise DownloadError(f"negative or non-integral volume at {timestamp}")
        symbol = row.get("symbol")
        if symbol != request["RAW_ES_SYMBOL"]:
            raise DownloadError(f"unexpected symbol {symbol!r} for {request['CURRENT_RTH_DATE']}")
        normalized.append({
            "ts_event": timestamp,
            "symbol": symbol,
            **{key: format(value, "f") for key, value in prices.items()},
            "volume": str(volume),
        })
    normalized.sort(key=lambda bar: bar["ts_event"])
    if [bar["ts_event"] for bar in normalized] != expected_times:
        raise DownloadError(f"missing or reordered hourly bars for {request['CURRENT_RTH_DATE']}")
    return normalized


def _frame_rows(store: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = store.metadata
    if store.dataset != DATASET or str(store.schema) != SCHEMA:
        raise DownloadError(f"DBN dataset/schema mismatch for {request['CURRENT_RTH_DATE']}")
    if list(store.symbols) != [request["RAW_ES_SYMBOL"]]:
        raise DownloadError(f"DBN requested symbol mismatch for {request['CURRENT_RTH_DATE']}: {store.symbols}")
    if str(store.stype_in) != STYPE_IN:
        raise DownloadError(f"DBN input symbology mismatch for {request['CURRENT_RTH_DATE']}")
    if _iso_timestamp(store.start) != request["PATCH_START_UTC"]:
        raise DownloadError(f"DBN query start metadata mismatch for {request['CURRENT_RTH_DATE']}")
    if store.end is None or _iso_timestamp(store.end) != request["PATCH_END_UTC"]:
        raise DownloadError(f"DBN query end metadata mismatch for {request['CURRENT_RTH_DATE']}")
    if metadata.partial or metadata.not_found:
        raise DownloadError(f"DBN reports unresolved symbols: partial={metadata.partial}, not_found={metadata.not_found}")
    frame = store.to_df(price_type="fixed", pretty_ts=True, map_symbols=True)
    frame = frame.reset_index()
    if "ts_event" not in frame.columns or "symbol" not in frame.columns:
        raise DownloadError("DBN decoded frame lacks timestamp or mapped raw-symbol identity")
    result: list[dict[str, Any]] = []
    for item in frame.to_dict(orient="records"):
        # Fixed-price DBN integers use a 1e-9 scale; Decimal conversion retains exact ticks.
        result.append({
            "ts_event": item["ts_event"],
            "symbol": item["symbol"],
            **{name: Decimal(int(item[name])) / Decimal(1_000_000_000) for name in ("open", "high", "low", "close")},
            "volume": item["volume"],
        })
    return result


def validate_dbn_file(path: Path, request: dict[str, Any], store_factory: Callable[[Path], Any] | None = None) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise DownloadError(f"missing/empty DBN file: {path}")
    if store_factory is None:
        from databento.common.dbnstore import DBNStore
        store_factory = DBNStore.from_file
    try:
        store = store_factory(path)
        decoded = _frame_rows(store, request)
        return validate_bars(decoded, request)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"DBN unreadable or invalid at {path}: {exc}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_csv(path: Path, bars: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(bars)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _filenames(request: dict[str, Any], data_root: Path = DATA_ROOT) -> tuple[Path, Path, Path]:
    stem = f"{request['CURRENT_RTH_DATE']}_{request['RAW_ES_SYMBOL']}_ohlcv-1h"
    raw = data_root / "raw" / f"{stem}.dbn.zst"
    part = raw.with_name(raw.name + ".part")
    normalized = data_root / "normalized" / f"{stem}.csv"
    return raw, part, normalized


def api_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return the exact quoted request parameters, with no interval expansion."""
    return {
        "dataset": DATASET,
        "symbols": [request["RAW_ES_SYMBOL"]],
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "start": request["PATCH_START_UTC"],
        "end": request["PATCH_END_UTC"],
    }


def _new_manifest(plan: dict[str, Any], data_root: Path = DATA_ROOT) -> dict[str, Any]:
    rows = []
    for request in plan["requests"]:
        raw, _, normalized = _filenames(request, data_root)
        rows.append({
            "CURRENT_RTH_DATE": request["CURRENT_RTH_DATE"],
            "RAW_ES_SYMBOL": request["RAW_ES_SYMBOL"],
            "PATCH_START_UTC": request["PATCH_START_UTC"],
            "PATCH_END_UTC": request["PATCH_END_UTC"],
            "EXPECTED_BARS": request["EXPECTED_HOURLY_BAR_COUNT"],
            "EXPECTED_HOURLY_BAR_STARTS_UTC": request["EXPECTED_HOURLY_BAR_STARTS_UTC"],
            "LOCAL_FILE": str(raw),
            "NORMALIZED_CSV": str(normalized),
            "ACTUAL_BARS": None,
            "BAR_TIMESTAMPS_UTC": [],
            "BARS": [],
            "FILE_SIZE_BYTES": None,
            "SHA256": None,
            "NORMALIZED_CSV_SHA256": None,
            "STATUS": "PENDING",
            "NOTES": "",
        })
    return {
        "status": "IN_PROGRESS",
        "data_provider": "Databento",
        "dataset": DATASET,
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "purpose": "SATO_OVERNIGHT_ONH_ONL_MISSING_INTERVAL_REPAIR",
        "quote_plan_source": str(QUOTE_PLAN_PATH),
        "quote_plan_sha256": sha256_file(QUOTE_PLAN_PATH),
        "frozen_request_set_sha256": FROZEN_REQUEST_SET_SHA256,
        "quote_estimated_cost_usd": EXPECTED_QUOTED_COST_USD,
        "actual_charge_usd": None,
        "expected_request_count": EXPECTED_PLAN_REQUESTS,
        "expected_hourly_bar_count": EXPECTED_PLAN_BARS,
        "extra_hours_downloaded": 0,
        "extra_dates_downloaded": 0,
        "extra_symbols_downloaded": 0,
        "requests": rows,
    }


def _manifest_row_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["CURRENT_RTH_DATE"]: row for row in manifest.get("requests", [])}


def _file_entry(request: dict[str, Any], raw: Path, normalized: Path, bars: list[dict[str, str]], status: str) -> dict[str, Any]:
    return {
        "CURRENT_RTH_DATE": request["CURRENT_RTH_DATE"],
        "RAW_ES_SYMBOL": request["RAW_ES_SYMBOL"],
        "PATCH_START_UTC": request["PATCH_START_UTC"],
        "PATCH_END_UTC": request["PATCH_END_UTC"],
        "EXPECTED_BARS": request["EXPECTED_HOURLY_BAR_COUNT"],
        "ACTUAL_BARS": len(bars),
        "BAR_TIMESTAMPS_UTC": [bar["ts_event"] for bar in bars],
        "BARS": bars,
        "LOCAL_FILE": str(raw),
        "NORMALIZED_CSV": str(normalized),
        "FILE_SIZE_BYTES": raw.stat().st_size,
        "SHA256": sha256_file(raw),
        "NORMALIZED_CSV_SHA256": sha256_file(normalized) if normalized.is_file() else None,
        "STATUS": status,
        "NOTES": "DBN metadata, mapped raw symbol, bounds, OHLC, ES tick grid and bar count validated.",
    }


def _existing_verified(
    request: dict[str, Any], row: dict[str, Any] | None, raw: Path, normalized: Path,
) -> tuple[list[dict[str, str]], str] | None:
    successful_statuses = {"VERIFIED", "SKIP_ALREADY_VERIFIED", "RECOVERED_VALID_PART"}
    if row is not None and row.get("STATUS") in successful_statuses:
        if not raw.is_file() or row.get("SHA256") != sha256_file(raw) or row.get("FILE_SIZE_BYTES") != raw.stat().st_size:
            # A file previously marked complete but no longer matching its checkpoint is untrusted.
            return None
        bars = validate_dbn_file(raw, request)
        if len(bars) == row.get("ACTUAL_BARS") and [b["ts_event"] for b in bars] == row.get("BAR_TIMESTAMPS_UTC"):
            if not normalized.is_file() or row.get("NORMALIZED_CSV_SHA256") != sha256_file(normalized):
                _write_csv(normalized, bars)
            return bars, "SKIP_ALREADY_VERIFIED"
        return None
    # A complete valid local final/part can be adopted after interruption without billing again.
    for candidate, status in ((raw, "SKIP_ALREADY_VERIFIED"), (raw.with_name(raw.name + ".part"), "RECOVERED_VALID_PART")):
        if candidate.is_file():
            try:
                bars = validate_dbn_file(candidate, request)
            except DownloadError:
                continue
            if candidate != raw:
                raw.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, raw)
            _write_csv(normalized, bars)
            return bars, status
    return None


def execute_downloads(
    plan: dict[str, Any], client: Any, *, data_root: Path | None = None, manifest_path: Path | None = None,
) -> dict[str, Any]:
    data_root = DATA_ROOT if data_root is None else data_root
    manifest_path = MANIFEST_PATH if manifest_path is None else manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError(f"existing download manifest is unreadable: {exc}") from exc
        if manifest.get("frozen_request_set_sha256") != FROZEN_REQUEST_SET_SHA256:
            raise DownloadError("existing download manifest belongs to a different request set")
    else:
        manifest = _new_manifest(plan, data_root)
    entries = _manifest_row_lookup(manifest)
    downloaded_now = 0
    skipped_now = 0

    for request in plan["requests"]:
        raw, part, normalized = _filenames(request, data_root)
        prior = entries.get(request["CURRENT_RTH_DATE"])
        existing = _existing_verified(request, prior, raw, normalized)
        if existing:
            bars, status = existing
            entries[request["CURRENT_RTH_DATE"]] = _file_entry(request, raw, normalized, bars, status)
            skipped_now += 1
            _atomic_json(manifest_path, manifest)
            continue

        if client is None:
            raise DownloadError("DATABENTO_API_KEY is not set; no download request was made")

        raw.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        try:
            response = client.timeseries.get_range(**api_request(request))
            response.to_file(part, compression="zstd")
            _fsync_file(part)
            bars = validate_dbn_file(part, request)
            # Emit the normalized companion only after the raw DBN passes validation.
            _write_csv(normalized, bars)
            os.replace(part, raw)
            entries[request["CURRENT_RTH_DATE"]] = _file_entry(request, raw, normalized, bars, "VERIFIED")
            downloaded_now += 1
            _atomic_json(manifest_path, manifest)
            print(f"VERIFIED {request['CURRENT_RTH_DATE']} {request['RAW_ES_SYMBOL']} bars={len(bars)} sha256={entries[request['CURRENT_RTH_DATE']]['SHA256']}")
        except Exception as exc:
            entries[request["CURRENT_RTH_DATE"]] = {
                "CURRENT_RTH_DATE": request["CURRENT_RTH_DATE"],
                "RAW_ES_SYMBOL": request["RAW_ES_SYMBOL"],
                "PATCH_START_UTC": request["PATCH_START_UTC"],
                "PATCH_END_UTC": request["PATCH_END_UTC"],
                "EXPECTED_BARS": request["EXPECTED_HOURLY_BAR_COUNT"],
                "LOCAL_FILE": str(raw),
                "STATUS": "INVALID_DOWNLOAD" if part.exists() else "FAILED",
                "NOTES": f"{type(exc).__name__}: {exc}",
            }
            _atomic_json(manifest_path, manifest)
            raise DownloadError(f"request failed for {request['CURRENT_RTH_DATE']} {request['RAW_ES_SYMBOL']}: {exc}") from exc

    manifest["requests"] = [entries[row["CURRENT_RTH_DATE"]] for row in plan["requests"]]
    verified = [row for row in manifest["requests"] if row["STATUS"] in {"VERIFIED", "SKIP_ALREADY_VERIFIED", "RECOVERED_VALID_PART"}]
    manifest["status"] = "COMPLETE" if len(verified) == EXPECTED_PLAN_REQUESTS else "IN_PROGRESS"
    manifest["requests_downloaded_this_run"] = downloaded_now
    manifest["requests_skipped_already_verified_this_run"] = skipped_now
    manifest["total_actual_hourly_bars"] = sum(int(row.get("ACTUAL_BARS") or 0) for row in manifest["requests"])
    manifest["invalid_request_count"] = sum(row.get("STATUS") == "INVALID_DOWNLOAD" for row in manifest["requests"])
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=QUOTE_PLAN_PATH)
    args = parser.parse_args(argv)
    try:
        plan = load_and_validate_plan(args.plan)
        api_key = os.environ.get("DATABENTO_API_KEY", "")
        if api_key:
            import databento as db
            client = db.Historical(api_key)
        else:
            client = None
        result = execute_downloads(plan, client, data_root=DATA_ROOT, manifest_path=MANIFEST_PATH)
        print(f"REQUESTS_DOWNLOADED = {result['requests_downloaded_this_run']}")
        print(f"REQUESTS_SKIPPED_ALREADY_VERIFIED = {result['requests_skipped_already_verified_this_run']}")
        print(f"TOTAL_DOWNLOADED_HOURLY_BARS = {result['total_actual_hourly_bars']}")
        print(f"INVALID_REQUESTS = {result['invalid_request_count']}")
        print(f"DOWNLOAD_STATUS = {result['status']}")
        print(f"MANIFEST = {MANIFEST_PATH}")
        return 0 if result["status"] == "COMPLETE" else 1
    except (DownloadError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
