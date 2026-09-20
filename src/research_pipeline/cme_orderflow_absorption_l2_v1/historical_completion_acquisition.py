"""Resumable acquisition for the approved historical completion quote.

The command is inert unless ``--download`` is supplied.  Before any download
it re-quotes the exact 79-request plan using metadata only and enforces the
approved cost guard.  Existing Dec/Jan NY files are outside both output roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from . import historical_completion_quote as quote_plan
from databento_replay_repair_downloader import is_retryable_exception


APPROVED_TOTAL_USD = Decimal("25.139932006600")
MAX_QUOTE_DEVIATION_USD = Decimal("1.00")
HARD_QUOTE_CAP_USD = Decimal("30.00")
DEFAULT_QUOTE_JSON = Path("research_runs/L2_HISTORICAL_COMPLETION_QUOTE/quote.json")
DEFAULT_OUTPUT_ROOT = Path("data/cme_orderflow_absorption_l2_v1/historical_completion")
JUN_JUL_ROOT_NAME = "jun_jul_ny_tails"
DEC_JAN_ROOT_NAME = "dec_jan_asia_europe"
MANIFEST_NAME = "acquisition-manifest.json"
MAX_DOWNLOAD_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 2.0


class AcquisitionError(RuntimeError):
    """Fail-closed acquisition contract violation."""


@dataclass(frozen=True)
class Item:
    request_id: str
    family: str
    component: str
    session_date: str
    symbol: str
    schema: str
    start: str
    end: str
    approved_cost_usd: Decimal
    relative_path: str

    def api_request(self) -> dict[str, Any]:
        return {
            "dataset": quote_plan.DATASET,
            "schema": self.schema,
            "symbols": [self.symbol],
            "stype_in": "raw_symbol",
            "start": self.start,
            "end": self.end,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionError(f"missing quote artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid quote artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AcquisitionError("quote artifact root is not an object")
    return value


def _request_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("group")), str(row.get("component")), str(row.get("session_date")),
        str(row.get("symbol")), str(row.get("schema")), str(row.get("start")), str(row.get("end")),
    )


def _item_payload(item: Item) -> dict[str, Any]:
    return {
        "family": item.family, "component": item.component, "session_date": item.session_date,
        "symbol": item.symbol, "schema": item.schema, "start": item.start, "end": item.end,
        "relative_path": item.relative_path, "approved_cost_usd": str(item.approved_cost_usd),
    }


def _filename(row: Mapping[str, Any]) -> str:
    start = str(row["start"])[11:19].replace(":", "")
    end = str(row["end"])[11:19].replace(":", "")
    return f"{row['symbol']}_{row['session_date']}_{start}_{end}_{row['schema']}.dbn.zst"


def _load_approved_quote(path: Path) -> tuple[dict[str, Any], tuple[Item, ...], str]:
    payload = _load_json(path)
    if payload.get("status") != "QUOTE_COMPLETE_NO_DATA_ACQUIRED":
        raise AcquisitionError("approved quote is not quote-only and complete")
    if payload.get("market_data_downloaded") is not False or payload.get("strategy_outcomes_run") is not False:
        raise AcquisitionError("approved quote contains non-quote activity")
    if int(payload.get("request_count", -1)) != 79:
        raise AcquisitionError("approved quote request count is not 79")
    if Decimal(str(payload.get("grand_total_usd"))) != APPROVED_TOTAL_USD:
        raise AcquisitionError("approved quote total differs from frozen baseline")
    rows = payload.get("requests")
    if not isinstance(rows, list) or len(rows) != 79 or not all(isinstance(row, dict) for row in rows):
        raise AcquisitionError("approved quote request rows are invalid")
    keys = [_request_key(row) for row in rows]
    if len(set(keys)) != 79:
        raise AcquisitionError("approved quote contains duplicate requests")
    items: list[Item] = []
    for row in rows:
        group = str(row["group"])
        if group == "JUN_JUL":
            family = JUN_JUL_ROOT_NAME
            expected = {"JUN_JUL_ES_MBO", "JUN_JUL_MES_MBP1"}
        elif group == "DEC_JAN":
            family, expected = DEC_JAN_ROOT_NAME, {"DEC_JAN_ES_MBP10"}
        elif group == "NOV28_DEPENDENCY":
            family, expected = DEC_JAN_ROOT_NAME, {"NOV28_PROFILE_MBP10"}
        else:
            raise AcquisitionError(f"unexpected request group: {group}")
        component = str(row["component"])
        if component not in expected:
            raise AcquisitionError(f"component/group mismatch: {component}")
        schema = str(row["schema"])
        if group == "JUN_JUL" and schema not in {"mbo", "mbp-1"}:
            raise AcquisitionError("invalid June/July schema")
        if group in {"DEC_JAN", "NOV28_DEPENDENCY"} and schema != "mbp-10":
            raise AcquisitionError("invalid Dec/Jan schema")
        items.append(Item(
            request_id=_canonical_sha256({"request": _request_key(row)}), family=family, component=component,
            session_date=str(row["session_date"]), symbol=str(row["symbol"]), schema=schema,
            start=str(row["start"]), end=str(row["end"]),
            approved_cost_usd=Decimal(str(row["cost_usd"])),
            relative_path=f"{family}/{_filename(row)}",
        ))
    family_counts = {family: sum(item.family == family for item in items) for family in (JUN_JUL_ROOT_NAME, DEC_JAN_ROOT_NAME)}
    if family_counts != {JUN_JUL_ROOT_NAME: 36, DEC_JAN_ROOT_NAME: 43}:
        raise AcquisitionError(f"family cardinality mismatch: {family_counts}")
    return payload, tuple(items), _sha256(path)


def _fresh_quote(client: object, approved: Mapping[str, Any], items: Sequence[Item]) -> tuple[dict[str, Any], dict[str, str]]:
    fresh = quote_plan.quote(client)
    fresh_rows = fresh.get("requests")
    if not isinstance(fresh_rows, list) or len(fresh_rows) != 79:
        raise AcquisitionError("fresh quote did not return exactly 79 requests")
    approved_rows = approved.get("requests")
    assert isinstance(approved_rows, list)
    fresh_by_key = {_request_key(row): row for row in fresh_rows if isinstance(row, dict)}
    approved_by_key = {_request_key(row): row for row in approved_rows if isinstance(row, dict)}
    if set(fresh_by_key) != set(approved_by_key):
        raise AcquisitionError("fresh quote request identity differs from approved plan")
    cost_by_id: dict[str, str] = {}
    for item in items:
        row = fresh_by_key.get((item.family == JUN_JUL_ROOT_NAME and "JUN_JUL" or "DEC_JAN" if item.component != "NOV28_PROFILE_MBP10" else "NOV28_DEPENDENCY", item.component, item.session_date, item.symbol, item.schema, item.start, item.end))
        if row is None:
            raise AcquisitionError(f"fresh quote omitted {item.relative_path}")
        cost_by_id[item.request_id] = str(row["cost_usd"])
    total = sum((Decimal(value) for value in cost_by_id.values()), Decimal(0))
    if total > HARD_QUOTE_CAP_USD or abs(total - APPROVED_TOTAL_USD) > MAX_QUOTE_DEVIATION_USD:
        raise AcquisitionError(f"fresh quote {total} is outside approved guard around {APPROVED_TOTAL_USD}")
    return {"total_usd": str(total), "approved_total_usd": str(APPROVED_TOTAL_USD), "deviation_usd": str(abs(total - APPROVED_TOTAL_USD))}, cost_by_id


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise AcquisitionError(f"stale manifest partial exists: {partial}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _load_manifest(path: Path, *, plan_hash: str, quote_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid acquisition manifest: {path}") from exc
    if payload.get("plan_sha256") != plan_hash or payload.get("quote_sha256") != quote_hash:
        raise AcquisitionError(f"acquisition manifest identity mismatch: {path}")
    return payload


def _assert_tree(root: Path, items: Sequence[Item], manifest: Mapping[str, Any] | None) -> None:
    allowed = {MANIFEST_NAME} | {Path(item.relative_path).name for item in items}
    recorded = set((manifest or {}).get("files", {}))
    for path in root.glob("*") if root.exists() else ():
        if not path.is_file():
            continue
        if path.name.endswith(".part"):
            raise AcquisitionError(f"stale partial file exists: {path}")
        if path.name not in allowed or (path.name != MANIFEST_NAME and path.name not in recorded):
            raise AcquisitionError(f"unknown or unrecorded file exists: {path}")
    if recorded - allowed:
        raise AcquisitionError("manifest contains a file outside the frozen plan")


def _verified_existing(root: Path, item: Item, manifest: Mapping[str, Any]) -> bool:
    path = root / Path(item.relative_path).name
    record = manifest.get("files", {}).get(path.name)
    if not path.exists():
        if record is not None:
            raise AcquisitionError(f"manifest-recorded file missing: {path}")
        return False
    if record is None:
        raise AcquisitionError(f"existing file is not recorded: {path}")
    size, digest = path.stat().st_size, _sha256(path)
    if size <= 0 or size != int(record.get("bytes", -1)) or digest != record.get("sha256"):
        raise AcquisitionError(f"existing file hash/size mismatch: {path}")
    return True


def _initial_manifest(*, root: Path, family: str, items: Sequence[Item], quote_path: Path, quote_hash: str, plan_hash: str, fresh: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ACQUISITION_IN_PROGRESS", "family": family, "dataset": quote_plan.DATASET,
        "quote_path": str(quote_path), "quote_sha256": quote_hash, "plan_sha256": plan_hash,
        "approved_total_usd": str(APPROVED_TOTAL_USD), "fresh_quote": dict(fresh),
        "request_count": len(items), "downloaded_file_count": 0,
        "strategy_replay_executed": False, "outcomes_inspected": False,
        "constraints": {"no_existing_files_replaced": True, "no_strategy_replay": True, "no_backtest": True},
        "requests": [{"request_id": item.request_id, "component": item.component, "session_date": item.session_date,
                      "symbol": item.symbol, "schema": item.schema, "start": item.start, "end": item.end,
                      "local_path": Path(item.relative_path).name, "approved_cost_usd": str(item.approved_cost_usd)} for item in items],
        "files": {},
    }


def _parse_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AcquisitionError(f"non-UTC request boundary: {value}")
    return int(parsed.timestamp() * 1_000_000_000)


def _validate_downloaded_file(path: Path, item: Item) -> dict[str, Any]:
    """Parse the complete DBN and validate its request-bound source contract."""
    try:
        from databento import DBNStore
    except ImportError as exc:  # pragma: no cover - the production path has Databento installed
        raise AcquisitionError("databento package is required for DBN validation") from exc
    start_ns, end_ns = _parse_ns(item.start), _parse_ns(item.end)
    count = 0
    first_ns: int | None = None
    last_ns: int | None = None
    previous_ns: int | None = None
    structural_kind: str | None = None
    store = DBNStore.from_file(path)
    try:
        for record in store:
            timestamp_ns = int(getattr(record, "ts_recv", getattr(record, "ts_event", 0)))
            if timestamp_ns < start_ns or timestamp_ns >= end_ns:
                raise AcquisitionError(f"DBN timestamp outside requested range: {path.name}")
            if previous_ns is not None and timestamp_ns < previous_ns:
                raise AcquisitionError(f"DBN timestamps are not ordered: {path.name}")
            previous_ns = timestamp_ns
            record_kind: str | None = None
            if item.schema == "mbo":
                required = ("action", "side", "price", "size", "order_id")
                record_kind = "mbo" if all(hasattr(record, field) for field in required) else None
            elif item.schema == "mbp-1":
                levels = getattr(record, "levels", ())
                record_kind = "mbp-1" if levels is not None else None
            elif item.schema == "mbp-10":
                levels = getattr(record, "levels", ())
                record_kind = "mbp-10" if levels is not None else None
            if record_kind is None:
                raise AcquisitionError(f"DBN record shape does not match {item.schema}: {path.name}")
            structural_kind = record_kind
            first_ns = timestamp_ns if first_ns is None else min(first_ns, timestamp_ns)
            last_ns = timestamp_ns if last_ns is None else max(last_ns, timestamp_ns)
            count += 1
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    if count == 0 or first_ns is None or last_ns is None:
        raise AcquisitionError(f"DBN contains no records: {path.name}")
    return {"record_count": count, "first_timestamp_ns": first_ns, "last_timestamp_ns": last_ns, "validated_schema": structural_kind}


def _download_one(client: object, root: Path, item: Item, manifest: MutableMapping[str, Any], fresh_cost: str) -> str:
    destination = root / Path(item.relative_path).name
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise AcquisitionError(f"stale partial exists: {partial}")
    if _verified_existing(root, item, manifest):
        return "SKIPPED_VERIFIED"
    last_error: BaseException | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.timeseries.get_range(**item.api_request(), path=str(partial))  # type: ignore[attr-defined]
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise AcquisitionError(f"empty download: {destination.name}")
            size, digest = partial.stat().st_size, _sha256(partial)
            validation = _validate_downloaded_file(partial, item)
            partial.replace(destination)
            manifest.setdefault("files", {})[destination.name] = {
                "request_id": item.request_id, "component": item.component, "session_date": item.session_date,
                "symbol": item.symbol, "schema": item.schema, "start": item.start, "end": item.end,
                "local_path": destination.name, "bytes": size, "sha256": digest,
                "fresh_component_cost_usd": fresh_cost, "status": "DOWNLOADED_VERIFIED",
                "source_validation": validation,
            }
            return "DOWNLOADED_VERIFIED"
        except Exception as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if not is_retryable_exception(exc):
                break
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time_module.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise AcquisitionError(f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: {destination.name}") from last_error


def run(*, client: object, quote_path: Path = DEFAULT_QUOTE_JSON, output_root: Path = DEFAULT_OUTPUT_ROOT, download: bool = False) -> dict[str, Any]:
    approved, items, quote_hash = _load_approved_quote(quote_path)
    plan_hash = _canonical_sha256([_item_payload(item) for item in items])
    fresh, fresh_costs = _fresh_quote(client, approved, items)
    result: dict[str, Any] = {"status": "PREFLIGHT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED", "download_requested": download,
                              "download_api_invoked": False, "request_count": len(items), "fresh_quote": fresh,
                              "output_roots": [str(output_root / JUN_JUL_ROOT_NAME), str(output_root / DEC_JAN_ROOT_NAME)],
                              "strategy_replay_executed": False, "outcomes_inspected": False,
                              "no_download_performed": not download}
    if not download:
        return result
    statuses: list[str] = []
    for family in (JUN_JUL_ROOT_NAME, DEC_JAN_ROOT_NAME):
        family_items = [item for item in items if item.family == family]
        root = output_root / family
        manifest_path = root / MANIFEST_NAME
        existing = _load_manifest(manifest_path, plan_hash=plan_hash, quote_hash=quote_hash)
        _assert_tree(root, family_items, existing)
        manifest = existing or _initial_manifest(root=root, family=family, items=family_items, quote_path=quote_path, quote_hash=quote_hash, plan_hash=plan_hash, fresh=fresh)
        manifest["latest_fresh_quote"] = dict(fresh)
        _write_manifest(manifest_path, manifest)
        for item in family_items:
            statuses.append(_download_one(client, root, item, manifest, fresh_costs[item.request_id]))
            manifest["files"] = manifest.get("files", {})
            manifest["downloaded_file_count"] = len(manifest["files"])
            _write_manifest(manifest_path, manifest)
        manifest["status"] = "ACQUISITION_COMPLETE_VERIFIED"
        _write_manifest(manifest_path, manifest)
    result.update({"status": "ACQUISITION_COMPLETE_VERIFIED", "download_api_invoked": any(status == "DOWNLOADED_VERIFIED" for status in statuses), "downloaded_or_verified_file_count": len(statuses), "no_download_performed": False})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-json", type=Path, default=DEFAULT_QUOTE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download", action="store_true", help="explicitly authorize the 79-file acquisition")
    args = parser.parse_args(argv)
    if not os.environ.get("DATABENTO_API_KEY"):
        raise AcquisitionError("DATABENTO_API_KEY is required and will not be printed")
    try:
        import databento as db
    except ImportError as exc:
        raise AcquisitionError("databento package is required") from exc
    result = run(client=db.Historical(os.environ["DATABENTO_API_KEY"]), quote_path=args.quote_json, output_root=args.output_root, download=args.download)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
