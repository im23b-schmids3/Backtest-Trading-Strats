"""Acquire only the frozen Dec-2025/Jan-2026 prior-RTH trade tails.

Without ``--download`` this module performs a metadata-only fresh quote.  It
never constructs the request calendar and never invokes strategy code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .v3_poc_dec2025_jan2026_calendar_audit import (
    AUDIT_JSON,
    SUPPLEMENT_QUOTE_JSON,
    V3_CONTRACT_SHA256,
    _canonical_sha256,
    validate_frozen_supplement_requests,
)


DATASET = "GLBX.MDP3"
SCHEMA = "trades"
STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY"
EXPECTED_REQUEST_COUNT = 40
EXPECTED_SYMBOL_COUNTS = {"ESZ5": 12, "ESH6": 28}
APPROVED_ESZ5_USD = Decimal("0.682852327824")
APPROVED_ESH6_USD = Decimal("1.713521361353")
APPROVED_TOTAL_USD = Decimal("2.396373689177")
HARD_CAP_USD = Decimal("3.00")
MAX_QUOTE_DEVIATION_USD = Decimal("0.50")
DEFAULT_OUTPUT_ROOT = Path(
    "data/cme_orderflow_absorption_l2_v3/dec2025_jan2026/es_prior_rth_trades_supplement"
)
MANIFEST_NAME = "supplemental-acquisition-manifest.json"
MAX_DOWNLOAD_ATTEMPTS = 3


class SupplementalAcquisitionError(RuntimeError):
    """Fail-closed supplemental acquisition contract violation."""


@dataclass(frozen=True)
class SupplementalRequest:
    index: int
    target_session: str
    prior_rth_date: str
    raw_symbol: str
    start_utc: str
    end_utc: str
    approved_cost_usd: Decimal

    @property
    def filename(self) -> str:
        return f"{self.raw_symbol}_{self.prior_rth_date}_200000_210000_trades.dbn.zst"

    def api_request(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": "raw_symbol",
            "symbols": [self.raw_symbol],
            "start": self.start_utc,
            "end": self.end_utc,
        }


@dataclass(frozen=True)
class FrozenSupplement:
    audit_path: Path
    quote_path: Path
    audit_sha256: str
    quote_sha256: str
    request_spec_sha256: str
    requests: tuple[SupplementalRequest, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementalAcquisitionError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SupplementalAcquisitionError(f"{label} root must be an object")
    return payload, raw


def load_frozen_supplement(
    audit_path: Path = AUDIT_JSON,
    quote_path: Path = SUPPLEMENT_QUOTE_JSON,
) -> FrozenSupplement:
    audit, audit_bytes = _load_json(audit_path, "calendar audit")
    quote, quote_bytes = _load_json(quote_path, "supplement quote")
    try:
        audit_requests = validate_frozen_supplement_requests(audit)
    except RuntimeError as exc:
        raise SupplementalAcquisitionError(str(exc)) from exc

    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    request_spec_sha256 = _canonical_sha256(audit_requests)
    if quote.get("status") != "SUPPLEMENT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED":
        raise SupplementalAcquisitionError("supplement quote status is not approved")
    if quote.get("calendar_audit_sha256") != audit_sha256:
        raise SupplementalAcquisitionError("supplement quote does not bind the current calendar audit")
    if quote.get("request_spec_sha256") != request_spec_sha256:
        raise SupplementalAcquisitionError("supplement quote request hash mismatch")
    if quote.get("v3_contract_sha256") != V3_CONTRACT_SHA256:
        raise SupplementalAcquisitionError("supplement quote V3 hash mismatch")
    if quote.get("request_count") != EXPECTED_REQUEST_COUNT:
        raise SupplementalAcquisitionError("supplement quote request count mismatch")
    if quote.get("dataset") != DATASET or quote.get("schema") != SCHEMA:
        raise SupplementalAcquisitionError("supplement quote dataset/schema mismatch")

    quote_rows = quote.get("requests")
    if not isinstance(quote_rows, list) or len(quote_rows) != EXPECTED_REQUEST_COUNT:
        raise SupplementalAcquisitionError("supplement quote must contain exactly 40 request rows")
    costs: list[Decimal] = []
    for index, (frozen, quoted) in enumerate(zip(audit_requests, quote_rows, strict=True)):
        if not isinstance(quoted, Mapping):
            raise SupplementalAcquisitionError(f"supplement quote row {index} is invalid")
        without_cost = {key: value for key, value in quoted.items() if key != "estimated_usd"}
        if without_cost != frozen:
            raise SupplementalAcquisitionError(f"supplement quote row {index} differs from frozen request")
        try:
            cost = Decimal(str(quoted["estimated_usd"]))
        except (KeyError, ArithmeticError) as exc:
            raise SupplementalAcquisitionError(f"supplement quote row {index} has invalid cost") from exc
        if not cost.is_finite() or cost < 0:
            raise SupplementalAcquisitionError(f"supplement quote row {index} has invalid cost")
        costs.append(cost)

    subtotals = {
        symbol: sum(
            (cost for request, cost in zip(audit_requests, costs, strict=True)
             if request["raw_symbol"] == symbol),
            Decimal(0),
        )
        for symbol in EXPECTED_SYMBOL_COUNTS
    }
    if subtotals != {"ESZ5": APPROVED_ESZ5_USD, "ESH6": APPROVED_ESH6_USD}:
        raise SupplementalAcquisitionError("supplement quote symbol subtotals differ from approval")
    if sum(costs, Decimal(0)) != APPROVED_TOTAL_USD:
        raise SupplementalAcquisitionError("supplement quote total differs from approval")
    if quote.get("symbol_subtotals_usd") != {key: str(value) for key, value in subtotals.items()}:
        raise SupplementalAcquisitionError("supplement quote reported subtotals do not reconcile")
    if Decimal(str(quote.get("total_usd"))) != APPROVED_TOTAL_USD:
        raise SupplementalAcquisitionError("supplement quote reported total does not reconcile")

    requests = tuple(
        SupplementalRequest(
            index=index,
            target_session=str(row["target_session"]),
            prior_rth_date=str(row["prior_rth_date"]),
            raw_symbol=str(row["raw_symbol"]),
            start_utc=str(row["start_utc"]),
            end_utc=str(row["end_utc"]),
            approved_cost_usd=costs[index],
        )
        for index, row in enumerate(audit_requests)
    )
    if len({item.filename for item in requests}) != EXPECTED_REQUEST_COUNT:
        raise SupplementalAcquisitionError("supplement filenames are not unique")
    return FrozenSupplement(
        audit_path=audit_path,
        quote_path=quote_path,
        audit_sha256=audit_sha256,
        quote_sha256=hashlib.sha256(quote_bytes).hexdigest(),
        request_spec_sha256=request_spec_sha256,
        requests=requests,
    )


def _validate_fresh_total(total: Decimal) -> None:
    if not total.is_finite() or total < 0:
        raise SupplementalAcquisitionError("fresh quote total is invalid")
    if total > HARD_CAP_USD:
        raise SupplementalAcquisitionError(f"fresh quote {total} exceeds hard cap {HARD_CAP_USD}")
    deviation = abs(total - APPROVED_TOTAL_USD)
    if deviation > MAX_QUOTE_DEVIATION_USD:
        raise SupplementalAcquisitionError(
            f"fresh quote deviation {deviation} exceeds {MAX_QUOTE_DEVIATION_USD}"
        )


def fresh_quote(client: object, frozen: FrozenSupplement) -> dict[str, Any]:
    costs: dict[str, str] = {}
    subtotals = {"ESZ5": Decimal(0), "ESH6": Decimal(0)}
    for item in frozen.requests:
        value = Decimal(str(client.metadata.get_cost(**item.api_request())))  # type: ignore[attr-defined]
        if not value.is_finite() or value < 0:
            raise SupplementalAcquisitionError(f"invalid fresh cost for {item.filename}")
        costs[item.filename] = str(value)
        subtotals[item.raw_symbol] += value
    total = sum(subtotals.values(), Decimal(0))
    _validate_fresh_total(total)
    return {
        "request_count": len(frozen.requests),
        "cost_by_filename_usd": costs,
        "symbol_subtotals_usd": {key: str(value) for key, value in subtotals.items()},
        "total_usd": str(total),
        "approved_total_usd": str(APPROVED_TOTAL_USD),
        "hard_cap_usd": str(HARD_CAP_USD),
        "deviation_usd": str(abs(total - APPROVED_TOTAL_USD)),
    }


def _request_record(item: SupplementalRequest, fresh_cost: str) -> dict[str, Any]:
    return {
        "index": item.index,
        "target_session": item.target_session,
        "prior_rth_date": item.prior_rth_date,
        "dataset": DATASET,
        "schema": SCHEMA,
        "stype_in": "raw_symbol",
        "raw_symbol": item.raw_symbol,
        "start_utc": item.start_utc,
        "end_utc": item.end_utc,
        "local_path": item.filename,
        "approved_component_cost_usd": str(item.approved_cost_usd),
        "fresh_component_cost_usd": fresh_cost,
    }


def _initial_manifest(frozen: FrozenSupplement, quote: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "SUPPLEMENT_ACQUISITION_IN_PROGRESS",
        "strategy_id": STRATEGY_ID,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "dataset": DATASET,
        "schema": SCHEMA,
        "request_count": EXPECTED_REQUEST_COUNT,
        "symbol_counts": dict(EXPECTED_SYMBOL_COUNTS),
        "calendar_audit": {"path": str(frozen.audit_path), "sha256": frozen.audit_sha256},
        "approved_quote": {
            "path": str(frozen.quote_path),
            "sha256": frozen.quote_sha256,
            "request_spec_sha256": frozen.request_spec_sha256,
            "ESZ5_subtotal_usd": str(APPROVED_ESZ5_USD),
            "ESH6_subtotal_usd": str(APPROVED_ESH6_USD),
            "total_usd": str(APPROVED_TOTAL_USD),
        },
        "latest_fresh_quote": dict(quote),
        "requests": [
            _request_record(item, quote["cost_by_filename_usd"][item.filename])
            for item in frozen.requests
        ],
        "files": {},
        "constraints": {
            "supplemental_tails_only": True,
            "no_overlapping_1430_2000_purchase": True,
            "no_mbp_10": True,
            "no_mbp_1": True,
            "no_mbo": True,
            "no_february": True,
            "strategy_executed": False,
            "outcomes_inspected": False,
        },
    }


def _atomic_write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / MANIFEST_NAME
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise SupplementalAcquisitionError(f"stale manifest partial exists: {partial}")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(destination)


def _load_existing_manifest(root: Path, frozen: FrozenSupplement) -> dict[str, Any] | None:
    path = root / MANIFEST_NAME
    if not path.exists():
        return None
    payload, _ = _load_json(path, "supplemental acquisition manifest")
    approved = payload.get("approved_quote", {})
    if (
        payload.get("strategy_id") != STRATEGY_ID
        or payload.get("v3_contract_sha256") != V3_CONTRACT_SHA256
        or approved.get("request_spec_sha256") != frozen.request_spec_sha256
        or approved.get("sha256") != frozen.quote_sha256
        or payload.get("request_count") != EXPECTED_REQUEST_COUNT
    ):
        raise SupplementalAcquisitionError("existing supplemental manifest identity mismatch")
    return payload


def _assert_output_tree(
    root: Path, requests: Sequence[SupplementalRequest], manifest: Mapping[str, Any] | None,
) -> None:
    if not root.exists():
        return
    allowed = {MANIFEST_NAME} | {item.filename for item in requests}
    recorded = set((manifest or {}).get("files", {}))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".part"):
            raise SupplementalAcquisitionError(f"stale partial exists: {relative}")
        if relative not in allowed:
            raise SupplementalAcquisitionError(f"unknown existing file: {relative}")
        if relative != MANIFEST_NAME and relative not in recorded:
            raise SupplementalAcquisitionError(f"unrecorded existing file: {relative}")
    if recorded - allowed:
        raise SupplementalAcquisitionError("manifest records files outside the frozen supplement")


def _verified_existing(
    root: Path, item: SupplementalRequest, manifest: Mapping[str, Any],
) -> bool:
    destination = root / item.filename
    record = manifest.get("files", {}).get(item.filename)
    if not destination.exists():
        if record is not None:
            raise SupplementalAcquisitionError(f"manifest-recorded file is missing: {item.filename}")
        return False
    if record is None:
        raise SupplementalAcquisitionError(f"existing file is not recorded: {item.filename}")
    size = destination.stat().st_size
    digest = _sha256(destination)
    if size <= 0 or size != int(record.get("bytes", -1)) or digest != record.get("sha256"):
        raise SupplementalAcquisitionError(f"existing file hash/size mismatch: {item.filename}")
    return True


def _download_one(
    *,
    client: object,
    root: Path,
    item: SupplementalRequest,
    manifest: MutableMapping[str, Any],
    fresh_cost: str,
) -> str:
    destination = root / item.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise SupplementalAcquisitionError(f"stale partial exists: {partial}")
    if _verified_existing(root, item, manifest):
        return "SKIPPED_VERIFIED"
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            client.timeseries.get_range(**item.api_request(), path=str(partial))  # type: ignore[attr-defined]
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise SupplementalAcquisitionError(f"empty download: {item.filename}")
            size = partial.stat().st_size
            digest = _sha256(partial)
            partial.replace(destination)
            manifest.setdefault("files", {})[item.filename] = {
                **_request_record(item, fresh_cost),
                "bytes": size,
                "sha256": digest,
                "status": "DOWNLOADED_VERIFIED",
            }
            _atomic_write_manifest(root, manifest)
            return "DOWNLOADED_VERIFIED"
        except Exception as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                break
    raise SupplementalAcquisitionError(
        f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: {item.filename}"
    ) from last_error


def run(
    *,
    client: object,
    audit_path: Path = AUDIT_JSON,
    quote_path: Path = SUPPLEMENT_QUOTE_JSON,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    download: bool = False,
) -> dict[str, Any]:
    frozen = load_frozen_supplement(audit_path, quote_path)
    quote = fresh_quote(client, frozen)
    result: dict[str, Any] = {
        "status": "SUPPLEMENT_PREFLIGHT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED",
        "download_requested": download,
        "download_api_invoked": False,
        "request_count": len(frozen.requests),
        "fresh_quote": quote,
        "output_root": str(output_root),
        "strategy_executed": False,
        "outcomes_inspected": False,
    }
    if not download:
        return result

    existing = _load_existing_manifest(output_root, frozen)
    _assert_output_tree(output_root, frozen.requests, existing)
    if existing is None:
        manifest: MutableMapping[str, Any] = _initial_manifest(frozen, quote)
    else:
        manifest = existing
        manifest["latest_fresh_quote"] = dict(quote)
    _atomic_write_manifest(output_root, manifest)

    statuses = [
        _download_one(
            client=client,
            root=output_root,
            item=item,
            manifest=manifest,
            fresh_cost=quote["cost_by_filename_usd"][item.filename],
        )
        for item in frozen.requests
    ]
    manifest["status"] = "SUPPLEMENT_ACQUISITION_COMPLETE_VERIFIED"
    manifest["verified_file_count"] = len(manifest.get("files", {}))
    _atomic_write_manifest(output_root, manifest)
    result.update({
        "status": "SUPPLEMENT_ACQUISITION_COMPLETE_VERIFIED",
        "download_api_invoked": any(status == "DOWNLOADED_VERIFIED" for status in statuses),
        "downloaded_or_verified_file_count": len(statuses),
    })
    return result


def _print_result(result: Mapping[str, Any]) -> None:
    quote = result["fresh_quote"]
    print(
        f"request_count={result['request_count']} ESZ5_subtotal_usd={quote['symbol_subtotals_usd']['ESZ5']} "
        f"ESH6_subtotal_usd={quote['symbol_subtotals_usd']['ESH6']} total_usd={quote['total_usd']}"
    )
    print(f"status={result['status']} output_root={result['output_root']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-audit", type=Path, default=AUDIT_JSON)
    parser.add_argument("--supplement-quote", type=Path, default=SUPPLEMENT_QUOTE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise SupplementalAcquisitionError("DATABENTO_API_KEY is required and will not be printed")
    try:
        import databento as db
    except ImportError as exc:
        raise SupplementalAcquisitionError("databento package is required") from exc
    client = db.Historical(api_key)
    result = run(
        client=client,
        audit_path=args.calendar_audit,
        quote_path=args.supplement_quote,
        output_root=args.output_root,
        download=args.download,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
