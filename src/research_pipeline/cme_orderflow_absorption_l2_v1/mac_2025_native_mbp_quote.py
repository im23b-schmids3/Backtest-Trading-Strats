"""Plan, quote, and later acquire the MAC 2025 native ES/MES replay block.

The quote mode calls only Databento metadata endpoints.  It never calls
``timeseries.get_range``.  The download mode is intentionally separate and
requires a previously written quote artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import algoseek_adapter  # noqa: F401 - keeps this module in the package surface


DATASET = "GLBX.MDP3"
STYPE_IN = "raw_symbol"
ES_SCHEMA = "mbp-10"
MES_SCHEMA = "mbp-1"
OUTPUT_ROOT = Path("data/databento/mac-2025-native-mbp")
DEFAULT_QUOTE_PATH = Path("research_runs/mac-2025-native-mbp-quote.json")
DEFAULT_ES_ONLY_QUOTE_PATH = Path("research_runs/mac-2025-native-es-mbp10-final-reduced-quote.json")
FORMAT_VERSION = "mac-2025-native-mbp-plan-v1"
ES_ONLY_FORMAT_VERSION = "mac-2025-native-es-mbp10-plan-v1"

# Good Friday has no complete U.S. equity-index RTH session.  April 17 is a
# normal equity-index target session; the holiday closure is April 18.  The
# reduced plan deliberately ends TRAIN on April 21 and starts VALIDATION on
# October 7 to preserve chronology while removing the requested tail dates.
TRAIN_CLOSED_DATES = frozenset({date(2025, 4, 18)})
TRAIN_START = date(2025, 3, 1)
TRAIN_END = date(2025, 4, 21)
VALIDATION_START = date(2025, 10, 7)
VALIDATION_END = date(2025, 10, 31)
# Each dependency is the immediately preceding usable session for the first
# target in its block: Friday Feb 28 precedes Monday Mar 3, and Monday Oct 6
# supplies the prior-session profile for Tuesday Oct 7.
DEPENDENCY_DATES = (date(2025, 2, 28), date(2025, 10, 6))
ES_ONLY_EXECUTION_POLICY = "ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS"

# CME's published 2025 equity-index roll dates are March 17 and September 15.
# The mapping is explicit so no one contract month silently spans a roll.
CONTRACT_TRANSITIONS = (
    {"effective_date": "2025-03-17", "from": "ESH5/MESH5", "to": "ESM5/MESM5"},
    {"effective_date": "2025-06-16", "from": "ESM5/MESM5", "to": "ESU5/MESU5"},
    {"effective_date": "2025-09-15", "from": "ESU5/MESU5", "to": "ESZ5/MESZ5"},
)


class PlanError(RuntimeError):
    """The frozen research plan or an acquisition invariant is invalid."""


@dataclass(frozen=True)
class Request:
    request_id: str
    category: str
    session_date: str
    schema: str
    symbol: str
    start: str
    end: str
    ordinal: int

    def api_request(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "schema": self.schema,
            "symbols": [self.symbol],
            "stype_in": STYPE_IN,
            "start": self.start,
            "end": self.end,
        }

    @property
    def path(self) -> str:
        return f"{self.category}/{self.session_date}/{self.request_id}.dbn.zst"


def _iso(day: date, clock: time) -> str:
    return datetime.combine(day, clock, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _weekdays(start: date, end: date, *, excluded: frozenset[date] = frozenset()) -> tuple[date, ...]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in excluded:
            days.append(cursor)
        cursor += timedelta(days=1)
    return tuple(days)


def train_dates() -> tuple[date, ...]:
    return _weekdays(TRAIN_START, TRAIN_END, excluded=TRAIN_CLOSED_DATES)


def validation_dates() -> tuple[date, ...]:
    # The requested "late September + all October" target is frozen as the
    # final two September weekdays plus all October weekdays: exactly 25.
    return _weekdays(VALIDATION_START, VALIDATION_END)


def contract_for(day: date) -> tuple[str, str]:
    if day < date(2025, 3, 17):
        return "ESH5", "MESH5"
    if day < date(2025, 6, 16):
        return "ESM5", "MESM5"
    if day < date(2025, 9, 15):
        return "ESU5", "MESU5"
    return "ESZ5", "MESZ5"


def session_end_utc(day: date) -> time:
    """Return the UTC close of the 09:30-16:00 America/New_York RTH."""
    # U.S. daylight saving time in 2025 runs from March 9 through November 1.
    return time(20, 0) if date(2025, 3, 9) <= day < date(2025, 11, 2) else time(21, 0)


def _request(category: str, day: date, schema: str, symbol: str, ordinal: int) -> Request:
    return Request(
        request_id=f"{category.lower()}-{day.isoformat()}-{schema}-{symbol}",
        category=category,
        session_date=day.isoformat(),
        schema=schema,
        symbol=symbol,
        start=_iso(day, time(0, 0)),
        end=_iso(day, session_end_utc(day)),
        ordinal=ordinal,
    )


def build_requests() -> tuple[Request, ...]:
    rows: list[Request] = []
    dates = (
        [("TRAIN", day) for day in train_dates()]
        + [("VALIDATION", day) for day in validation_dates()]
        + [("DEPENDENCY", day) for day in DEPENDENCY_DATES]
    )
    for category, day in dates:
        es, mes = contract_for(day)
        ordinal = len(rows) + 1
        rows.extend((_request(category, day, ES_SCHEMA, es, ordinal),
                     _request(category, day, MES_SCHEMA, mes, ordinal + 1)))
    if len(rows) != 112 or any(item.schema not in {ES_SCHEMA, MES_SCHEMA} for item in rows):
        raise PlanError(f"unexpected MAC request plan: {len(rows)} requests")
    if {item.session_date for item in rows if item.category == "TRAIN"} & {item.session_date for item in rows if item.category == "VALIDATION"}:
        raise PlanError("train/validation date overlap")
    return tuple(rows)


def build_es_only_requests() -> tuple[Request, ...]:
    """Build the same frozen date/windows plan with native ES MBP-10 only."""
    rows: list[Request] = []
    dates = (
        [("TRAIN", day) for day in train_dates()]
        + [("VALIDATION", day) for day in validation_dates()]
        + [("DEPENDENCY", day) for day in DEPENDENCY_DATES]
    )
    for category, day in dates:
        es, _ = contract_for(day)
        rows.append(_request(category, day, ES_SCHEMA, es, len(rows) + 1))
    if len(rows) != 56 or any(item.schema != ES_SCHEMA for item in rows):
        raise PlanError(f"unexpected ES-only MAC request plan: {len(rows)} requests")
    if {item.session_date for item in rows if item.category == "TRAIN"} & {item.session_date for item in rows if item.category == "VALIDATION"}:
        raise PlanError("train/validation date overlap")
    return tuple(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _request_rows(requests: Sequence[Request]) -> list[dict[str, Any]]:
    return [{"request_id": item.request_id, "category": item.category, "session_date": item.session_date,
             "schema": item.schema, "symbol": item.symbol, "start": item.start, "end": item.end,
             "ordinal": item.ordinal, "path": item.path} for item in requests]


def plan_artifact() -> dict[str, Any]:
    requests = build_requests()
    return {
        "status": "PLAN_READY_NO_DATA_ACQUIRED",
        "format_version": FORMAT_VERSION,
        "quote_only": True,
        "market_data_downloaded": False,
        "strategy_outcomes_run": False,
        "dataset": DATASET,
        "stype_in": STYPE_IN,
        "schemas": {"ES": ES_SCHEMA, "MES": MES_SCHEMA},
        "mbo_used": False,
        "mes_proxy_used": False,
        "synthetic_mes_used": False,
        "train_dates": [day.isoformat() for day in train_dates()],
        "validation_dates": [day.isoformat() for day in validation_dates()],
        "dependency_dates": [day.isoformat() for day in DEPENDENCY_DATES],
        "contract_transitions": list(CONTRACT_TRANSITIONS),
        "coverage": {
            "start_utc": "00:00:00",
            "end_utc_exclusive_by_dst": {"standard_time": "21:00:00", "daylight_time": "20:00:00"},
            "semantics": "[start,end); end is the date-specific 16:00 America/New_York RTH close",
        },
        "requests": _request_rows(requests),
    }


def es_only_plan_artifact() -> dict[str, Any]:
    requests = build_es_only_requests()
    return {
        "status": "PLAN_READY_NO_DATA_ACQUIRED",
        "format_version": ES_ONLY_FORMAT_VERSION,
        "quote_only": True,
        "market_data_downloaded": False,
        "strategy_outcomes_run": False,
        "dataset_variant": "ES_MBP10_ONLY",
        "dataset": DATASET,
        "stype_in": STYPE_IN,
        "schemas": {"ES": ES_SCHEMA},
        "mes_market_data_included": False,
        "execution_policy": ES_ONLY_EXECUTION_POLICY,
        "native_mes_required": False,
        "mbo_used": False,
        "mes_proxy_used": False,
        "synthetic_mes_used": False,
        "train_dates": [day.isoformat() for day in train_dates()],
        "validation_dates": [day.isoformat() for day in validation_dates()],
        "dependency_dates": [day.isoformat() for day in DEPENDENCY_DATES],
        "contract_transitions": list(CONTRACT_TRANSITIONS),
        "coverage": {
            "start_utc": "00:00:00",
            "end_utc_exclusive_by_dst": {"standard_time": "21:00:00", "daylight_time": "20:00:00"},
            "semantics": "[start,end); end is the date-specific 16:00 America/New_York RTH close",
        },
        "requests": _request_rows(requests),
    }


def quote(client: Any) -> dict[str, Any]:
    artifact = plan_artifact()
    costs: list[dict[str, Any]] = []
    totals = {"ES_MBP10": Decimal("0"), "MES_MBP1": Decimal("0"), "TRAIN": Decimal("0"),
              "VALIDATION": Decimal("0"), "DEPENDENCY": Decimal("0")}
    for row in artifact["requests"]:
        request = dict(row)
        request.pop("request_id")
        request.pop("category")
        request.pop("session_date")
        request.pop("ordinal")
        request.pop("path")
        expected_es, expected_mes = contract_for(date.fromisoformat(row["session_date"]))
        expected_schema = ES_SCHEMA if row["schema"] == ES_SCHEMA else MES_SCHEMA if row["schema"] == MES_SCHEMA else None
        expected_symbol = expected_es if row["schema"] == ES_SCHEMA else expected_mes if row["schema"] == MES_SCHEMA else None
        if expected_schema is None or row["schema"] != expected_schema or row["symbol"] != expected_symbol:
            raise PlanError(f"schema/contract mismatch in quote plan: {row['request_id']}")
        value = Decimal(str(client.metadata.get_cost(dataset=DATASET, schema=row["schema"], symbols=[row["symbol"]],
                                                     stype_in=STYPE_IN, start=row["start"], end=row["end"])))
        if not value.is_finite() or value <= 0:
            raise PlanError(f"invalid Databento quote for {row['request_id']}: {value}")
        component = "ES_MBP10" if row["schema"] == ES_SCHEMA else "MES_MBP1"
        totals[component] += value
        totals[row["category"]] += value
        costs.append({**row, "quoted_cost_usd": str(value)})
        print(f"QUOTE {row['category']} {row['session_date']} {row['schema']} {row['symbol']} ${value:.6f}")
    artifact.update({
        "status": "QUOTE_COMPLETE_NO_DATA_ACQUIRED",
        "quote_only": True,
        "requests": costs,
        "totals_usd": {key: str(value) for key, value in totals.items()},
        "grand_total_usd": str(sum((Decimal(value) for value in (str(totals["TRAIN"]), str(totals["VALIDATION"]), str(totals["DEPENDENCY"]))), Decimal("0"))),
    })
    return artifact


def quote_es_only(client: Any) -> dict[str, Any]:
    """Quote exactly the ES-only plan through metadata.get_cost."""
    artifact = es_only_plan_artifact()
    costs: list[dict[str, Any]] = []
    totals = {"ES_MBP10": Decimal("0"), "TRAIN": Decimal("0"),
              "VALIDATION": Decimal("0"), "DEPENDENCY": Decimal("0")}
    for row in artifact["requests"]:
        expected_es, _ = contract_for(date.fromisoformat(row["session_date"]))
        if row["schema"] != ES_SCHEMA or row["symbol"] != expected_es:
            raise PlanError(f"schema/contract mismatch in ES-only quote plan: {row['request_id']}")
        value = Decimal(str(client.metadata.get_cost(
            dataset=DATASET, schema=row["schema"], symbols=[row["symbol"]],
            stype_in=STYPE_IN, start=row["start"], end=row["end"],
        )))
        if not value.is_finite() or value <= 0:
            raise PlanError(f"invalid Databento quote for {row['request_id']}: {value}")
        totals["ES_MBP10"] += value
        totals[row["category"]] += value
        costs.append({**row, "quoted_cost_usd": str(value)})
        print(f"QUOTE {row['category']} {row['session_date']} {row['schema']} {row['symbol']} ${value:.6f}")
    artifact.update({
        "status": "QUOTE_COMPLETE_NO_DATA_ACQUIRED",
        "quote_only": True,
        "requests": costs,
        "totals_usd": {key: str(value) for key, value in totals.items()},
        "grand_total_usd": str(sum((totals["TRAIN"], totals["VALIDATION"], totals["DEPENDENCY"]), Decimal("0"))),
    })
    return artifact


def _load_quote(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read quote artifact: {path}") from exc
    if payload.get("status") != "QUOTE_COMPLETE_NO_DATA_ACQUIRED":
        raise PlanError("quote artifact is not a completed MAC 2025 quote")
    if payload.get("format_version") == ES_ONLY_FORMAT_VERSION:
        expected = es_only_plan_artifact()
    elif payload.get("format_version") == FORMAT_VERSION:
        expected = plan_artifact()
    else:
        raise PlanError("quote artifact is not a completed MAC 2025 quote")
    if [(row["session_date"], row["schema"], row["symbol"], row["start"], row["end"]) for row in payload.get("requests", ())] != [(row["session_date"], row["schema"], row["symbol"], row["start"], row["end"]) for row in expected["requests"]]:
        raise PlanError("quote artifact request plan does not match the frozen MAC plan")
    return payload


def download(quote_path: Path, output_root: Path) -> None:
    payload = _load_quote(quote_path)
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise PlanError("DATABENTO_API_KEY is required for download mode")
    try:
        import databento as db
        from databento_replay_repair_downloader import _retry_call
    except ImportError as exc:
        raise PlanError("Databento SDK and downloader retry support are required") from exc
    client = db.Historical(key=api_key)
    manifest_path = output_root / "mac-2025-native-mbp-download-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "format_version": FORMAT_VERSION, "status": "IN_PROGRESS", "quote_artifact": str(quote_path), "requests": {}
    }
    for row in payload["requests"]:
        destination = output_root / row["path"]
        record = manifest["requests"].get(row["request_id"])
        if record and destination.is_file() and record.get("sha256") == _sha256(destination):
            print(f"SKIP_VERIFIED {row['request_id']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if not partial.is_file():
            def fetch() -> Any:
                partial.unlink(missing_ok=True)
                return client.timeseries.get_range(dataset=DATASET, schema=row["schema"], symbols=[row["symbol"]],
                                                   stype_in=STYPE_IN, start=row["start"], end=row["end"], path=str(partial))
            _retry_call(operation="download", label=row["request_id"], callback=fetch)
        from databento_replay_repair_downloader import Request as RepairRequest, Window, _parse_ns, validate_dbn
        repair_request = RepairRequest(row["request_id"], row["session_date"], row["schema"], row["schema"], row["symbol"],
                                       Window(_parse_ns(row["start"]), _parse_ns(row["end"]), "MAC2025"), row["ordinal"])
        verification = validate_dbn(partial, repair_request)
        digest = _sha256(partial)
        os.replace(partial, destination)
        manifest["requests"][row["request_id"]] = {**row, "sha256": digest, "bytes": destination.stat().st_size,
                                                     "verification": verification, "status": "VERIFIED"}
        _write_json(manifest_path, manifest)
        print(f"DOWNLOADED_VERIFIED {row['request_id']} sha256={digest}")
    manifest["status"] = "COMPLETE"
    _write_json(manifest_path, manifest)
    print("MAC_2025_NATIVE_MBP_DOWNLOAD_COMPLETE=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--quote", action="store_true")
    mode.add_argument("--quote-es-only", action="store_true")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_QUOTE_PATH)
    parser.add_argument("--es-only-output", type=Path, default=DEFAULT_ES_ONLY_QUOTE_PATH)
    parser.add_argument("--quote-artifact", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.plan_only:
        _write_json(args.output, plan_artifact())
        print(json.dumps(plan_artifact(), sort_keys=True))
        return 0
    if args.quote:
        api_key = os.environ.get("DATABENTO_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: DATABENTO_API_KEY is required for quote mode")
        import databento as db
        result = quote(db.Historical(key=api_key))
        _write_json(args.output, result)
        print(json.dumps({"status": result["status"], "quote_artifact": str(args.output),
                          "grand_total_usd": result["grand_total_usd"]}, sort_keys=True))
        return 0
    if args.quote_es_only:
        api_key = os.environ.get("DATABENTO_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: DATABENTO_API_KEY is required for ES-only quote mode")
        import databento as db
        result = quote_es_only(db.Historical(key=api_key))
        _write_json(args.es_only_output, result)
        print(json.dumps({"status": result["status"], "quote_artifact": str(args.es_only_output),
                          "grand_total_usd": result["grand_total_usd"]}, sort_keys=True))
        return 0
    if args.quote_artifact is None:
        raise SystemExit("ERROR: --quote-artifact is required for download mode")
    download(args.quote_artifact, args.output_root)
    return 0


if __name__ == "__main__":
    main()
