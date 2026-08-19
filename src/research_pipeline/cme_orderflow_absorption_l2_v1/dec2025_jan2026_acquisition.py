"""Acquisition-only runner for the frozen Dec-2025/Jan-2026 L2 V3 block.

The quote artifact is the source of truth for target sessions, symbols, prior
RTH mappings, and calendar-aware windows.  Strategy code is never invoked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from . import dec2025_feb2026_quote as quote_contract


DATASET = "GLBX.MDP3"
STRATEGY_ID = "CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY"
EXPECTED_V3_HASH = "a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4"
EVIDENCE_CLASSIFICATION = "RETROSPECTIVE_ROBUSTNESS_DEC2025_JAN2026_NOT_STRICT_OOS"
SCENARIO_NAME = "DEC_PLUS_JAN"
EXPECTED_SESSION_COUNT = 42
EXPECTED_FIRST_SESSION = "2025-12-01"
EXPECTED_LAST_SESSION = "2026-01-30"
APPROVED_TOTAL_USD = Decimal("110.049909532070")
APPROVED_DECEMBER_USD = Decimal("51.986632294954")
APPROVED_JANUARY_USD = Decimal("58.063277237116")
HARD_CAP_USD = Decimal("115.00")
MAX_QUOTE_DEVIATION_USD = Decimal("1.00")
DEFAULT_QUOTE_JSON = Path("research_runs/L2_V3_DEC2025_FEB2026_QUOTE/quote.json")
DEFAULT_OUTPUT_ROOT = Path("data/cme_orderflow_absorption_l2_v3/dec2025_jan2026")
MANIFEST_NAME = "acquisition-manifest.json"
MAX_DOWNLOAD_ATTEMPTS = 3


class AcquisitionError(RuntimeError):
    """Fail-closed acquisition contract violation."""


@dataclass(frozen=True)
class FrozenRow:
    target_session: date
    es_raw_symbol: str
    es_instrument_id: int
    mes_raw_symbol: str
    mes_instrument_id: int
    prior_rth_date: date
    prior_rth_es_raw_symbol: str
    effective_session_end: time
    shortened_session: bool
    approved_costs: Mapping[str, Decimal]


@dataclass(frozen=True)
class Component:
    target_session: date
    purpose: str
    schema: str
    raw_symbol: str
    start: str
    end: str
    relative_path: str
    approved_cost_usd: Decimal
    prior_rth_date: date | None = None

    def request(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "schema": self.schema,
            "symbols": [self.raw_symbol],
            "stype_in": "raw_symbol",
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class FrozenQuote:
    source_json: Path
    source_csv: Path
    rows: tuple[FrozenRow, ...]
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc_time(value: str, expected_day: date) -> time:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcquisitionError(f"invalid UTC session end: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AcquisitionError(f"session end is not UTC: {value!r}")
    if parsed.date() != expected_day:
        raise AcquisitionError(f"session end date disagrees with target: {value!r}")
    return parsed.time().replace(tzinfo=None)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionError(f"missing quote artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid quote artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionError("quote artifact root is not an object")
    return payload


def _scenario(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [item for item in payload.get("scenarios", [])
               if isinstance(item, dict) and item.get("name") == SCENARIO_NAME]
    if len(matches) != 1:
        raise AcquisitionError("quote artifact must contain exactly one DEC_PLUS_JAN scenario")
    scenario = matches[0]
    if (
        scenario.get("session_count") != EXPECTED_SESSION_COUNT
        or scenario.get("start_date") != EXPECTED_FIRST_SESSION
        or scenario.get("end_date") != EXPECTED_LAST_SESSION
        or Decimal(str(scenario.get("component_totals_usd", {}).get("TOTAL_USD"))) != APPROVED_TOTAL_USD
    ):
        raise AcquisitionError("DEC_PLUS_JAN scenario identity or approved total changed")
    return scenario


def _validate_csv(json_rows: Sequence[Mapping[str, Any]], csv_path: Path) -> None:
    if not csv_path.is_file():
        raise AcquisitionError(f"missing paired quote CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = {row["target_session"]: row for row in csv.DictReader(handle)}
    fields = (
        "target_session", "es_raw_symbol", "mes_raw_symbol", "prior_rth_date",
        "prior_rth_es_raw_symbol", "effective_session_end_utc", "session_total_usd",
    )
    for row in json_rows:
        target = str(row["target_session"])
        counterpart = csv_rows.get(target)
        if counterpart is None or any(str(row[field]) != counterpart[field] for field in fields):
            raise AcquisitionError(f"quote JSON/CSV disagreement for {target}")
        for component, value in row["costs_usd"].items():
            if str(value) != counterpart[component]:
                raise AcquisitionError(f"quote JSON/CSV cost disagreement for {target}/{component}")


def load_frozen_quote(path: Path = DEFAULT_QUOTE_JSON) -> FrozenQuote:
    """Load and validate only the artifact-selected DEC_PLUS_JAN rows."""
    payload = _load_json(path)
    _scenario(payload)
    if payload.get("status") != "QUOTE_COMPLETE_NO_DATA_ACQUIRED":
        raise AcquisitionError("quote artifact status is not complete/no-data")
    if payload.get("strategy_id") != STRATEGY_ID or payload.get("v3_contract_sha256") != EXPECTED_V3_HASH:
        raise AcquisitionError("quote artifact strategy identity changed")
    if payload.get("market_data_downloaded") is not False or payload.get("strategy_outcomes_run") is not False:
        raise AcquisitionError("quote artifact is not quote-only evidence")

    all_rows = payload.get("sessions")
    if not isinstance(all_rows, list):
        raise AcquisitionError("quote artifact sessions are missing")
    selected = [row for row in all_rows if isinstance(row, dict) and
                EXPECTED_FIRST_SESSION <= str(row.get("target_session", "")) <= EXPECTED_LAST_SESSION]
    if len(selected) != EXPECTED_SESSION_COUNT:
        raise AcquisitionError(f"expected exactly {EXPECTED_SESSION_COUNT} DEC_PLUS_JAN rows")
    _validate_csv(selected, path.with_suffix(".csv"))

    expected_sessions = quote_contract.build_sessions(
        date.fromisoformat(EXPECTED_FIRST_SESSION), date.fromisoformat(EXPECTED_LAST_SESSION)
    )
    expected_by_day = {item.session_date: item for item in expected_sessions}
    rows: list[FrozenRow] = []
    for raw in selected:
        day = date.fromisoformat(str(raw["target_session"]))
        prior = date.fromisoformat(str(raw["prior_rth_date"]))
        expected = expected_by_day.get(day)
        if expected is None or expected.previous_rth_date != prior:
            raise AcquisitionError(f"calendar/prior-RTH mismatch for {day}")
        if day.weekday() >= 5 or day.month == 2 or day.year not in (2025, 2026):
            raise AcquisitionError(f"prohibited target date in DEC_PLUS_JAN: {day}")
        effective_end = _parse_utc_time(str(raw["effective_session_end_utc"]), day)
        if effective_end != expected.effective_end or bool(raw["shortened_session"]) != expected.shortened:
            raise AcquisitionError(f"calendar-aware end mismatch for {day}")
        costs_raw = raw.get("costs_usd")
        if not isinstance(costs_raw, dict) or set(costs_raw) != {
            "ES_MBP10_USD", "MES_MBP1_USD", "PRIOR_RTH_TRADES_USD"
        }:
            raise AcquisitionError(f"invalid component costs for {day}")
        rows.append(FrozenRow(
            target_session=day,
            es_raw_symbol=str(raw["es_raw_symbol"]),
            es_instrument_id=int(raw["es_instrument_id"]),
            mes_raw_symbol=str(raw["mes_raw_symbol"]),
            mes_instrument_id=int(raw["mes_instrument_id"]),
            prior_rth_date=prior,
            prior_rth_es_raw_symbol=str(raw["prior_rth_es_raw_symbol"]),
            effective_session_end=effective_end,
            shortened_session=bool(raw["shortened_session"]),
            approved_costs={key: Decimal(str(value)) for key, value in costs_raw.items()},
        ))

    if [item.target_session for item in rows] != [item.session_date for item in expected_sessions]:
        raise AcquisitionError("quote rows are missing, unordered, duplicated, or extra")
    if rows[0].prior_rth_date != date(2025, 11, 28):
        raise AcquisitionError("December 1 prior-RTH must be November 28")
    for row in rows:
        expected_es, expected_mes = (("ESZ5", "MESZ5") if row.target_session <= date(2025, 12, 16)
                                     else ("ESH6", "MESH6"))
        if (row.es_raw_symbol, row.mes_raw_symbol) != (expected_es, expected_mes):
            raise AcquisitionError(f"frozen contract roll mismatch for {row.target_session}")
    roll = next(item for item in rows if item.target_session == date(2025, 12, 17))
    if roll.prior_rth_date != date(2025, 12, 16) or roll.prior_rth_es_raw_symbol != "ESZ5":
        raise AcquisitionError("December 17 prior-RTH roll transition changed")

    approved_dec = sum((sum(row.approved_costs.values(), Decimal(0)) for row in rows
                        if row.target_session.month == 12), Decimal(0))
    approved_jan = sum((sum(row.approved_costs.values(), Decimal(0)) for row in rows
                        if row.target_session.month == 1), Decimal(0))
    if approved_dec != APPROVED_DECEMBER_USD or approved_jan != APPROVED_JANUARY_USD:
        raise AcquisitionError("approved monthly totals do not reconcile")
    if approved_dec + approved_jan != APPROVED_TOTAL_USD:
        raise AcquisitionError("approved grand total does not reconcile")
    return FrozenQuote(path.resolve(), path.with_suffix(".csv").resolve(), tuple(rows), _sha256(path))


def _utc(day: date, clock: time, *, plus_one_second: bool = False) -> str:
    value = datetime.combine(day, clock, tzinfo=timezone.utc)
    if plus_one_second:
        from datetime import timedelta
        value += timedelta(seconds=1)
    return value.isoformat().replace("+00:00", "Z")


def _filename(symbol: str, day: date, start: str, end: str, schema: str) -> str:
    start_token = start[11:19].replace(":", "")
    end_token = end[11:19].replace(":", "")
    return f"{symbol}_{day.isoformat()}_{start_token}_{end_token}_{schema}.dbn"


def components(frozen: FrozenQuote) -> tuple[Component, ...]:
    """Construct exactly three artifact-bound components per target session."""
    result: list[Component] = []
    for row in frozen.rows:
        target_end = _utc(row.target_session, row.effective_session_end, plus_one_second=True)
        es_start = _utc(row.target_session, time(13, 0))
        mes_start = _utc(row.target_session, time(13, 30))
        prior_end_clock = min(quote_contract.PROFILE_END, quote_contract._session_end(row.prior_rth_date))
        prior_start = _utc(row.prior_rth_date, time(13, 30))
        prior_end = _utc(row.prior_rth_date, prior_end_clock)
        definitions = (
            ("ES_MBP10", "mbp-10", row.es_raw_symbol, row.target_session, es_start, target_end,
             "es_mbp10", row.approved_costs["ES_MBP10_USD"], None),
            ("MES_MBP1", "mbp-1", row.mes_raw_symbol, row.target_session, mes_start, target_end,
             "mes_mbp1", row.approved_costs["MES_MBP1_USD"], None),
            ("PRIOR_RTH_TRADES", "trades", row.prior_rth_es_raw_symbol, row.prior_rth_date,
             prior_start, prior_end, "es_prior_rth_trades", row.approved_costs["PRIOR_RTH_TRADES_USD"],
             row.prior_rth_date),
        )
        for purpose, schema, symbol, file_day, start, end, directory, cost, prior_day in definitions:
            result.append(Component(
                target_session=row.target_session,
                purpose=purpose,
                schema=schema,
                raw_symbol=symbol,
                start=start,
                end=end,
                relative_path=f"{directory}/{_filename(symbol, file_day, start, end, schema)}",
                approved_cost_usd=cost,
                prior_rth_date=prior_day,
            ))
    if len(result) != EXPECTED_SESSION_COUNT * 3:
        raise AcquisitionError("component cardinality is not exactly 126")
    if any(item.schema == "mbo" or item.target_session.month == 2 for item in result):
        raise AcquisitionError("prohibited MBO or February target request")
    return tuple(result)


def _verify_symbology(client: object, frozen: FrozenQuote) -> None:
    sessions = tuple(quote_contract.Session(
        session_date=row.target_session,
        previous_rth_date=row.prior_rth_date,
        effective_end=row.effective_session_end,
        previous_rth_end=min(quote_contract.PROFILE_END, quote_contract._session_end(row.prior_rth_date)),
    ) for row in frozen.rows)
    resolved = quote_contract.resolve_symbols(client, sessions)
    for row in frozen.rows:
        day, prior = row.target_session, row.prior_rth_date
        if (
            resolved.es_by_date[day] != row.es_raw_symbol
            or resolved.mes_by_date[day] != row.mes_raw_symbol
            or resolved.es_by_date[prior] != row.prior_rth_es_raw_symbol
            or resolved.es_instrument_id_by_date[day] != row.es_instrument_id
            or resolved.mes_instrument_id_by_date[day] != row.mes_instrument_id
        ):
            raise AcquisitionError(f"fresh symbology disagrees with quote artifact for {day}")


def _validate_fresh_total(total: Decimal) -> None:
    if not total.is_finite() or total < 0:
        raise AcquisitionError("fresh quote total is invalid")
    if total > HARD_CAP_USD:
        raise AcquisitionError(f"fresh quote {total} exceeds hard cap {HARD_CAP_USD}")
    deviation = abs(total - APPROVED_TOTAL_USD)
    if deviation > MAX_QUOTE_DEVIATION_USD:
        raise AcquisitionError(
            f"fresh quote deviation {deviation} exceeds {MAX_QUOTE_DEVIATION_USD} from {APPROVED_TOTAL_USD}"
        )


def fresh_quote(client: object, frozen: FrozenQuote, items: Sequence[Component]) -> dict[str, Any]:
    _verify_symbology(client, frozen)
    by_path: dict[str, str] = {}
    monthly = {
        "2025-12": {"ES_MBP10_USD": Decimal(0), "MES_MBP1_USD": Decimal(0),
                    "PRIOR_RTH_TRADES_USD": Decimal(0)},
        "2026-01": {"ES_MBP10_USD": Decimal(0), "MES_MBP1_USD": Decimal(0),
                    "PRIOR_RTH_TRADES_USD": Decimal(0)},
    }
    component_key = {"ES_MBP10": "ES_MBP10_USD", "MES_MBP1": "MES_MBP1_USD",
                     "PRIOR_RTH_TRADES": "PRIOR_RTH_TRADES_USD"}
    for item in items:
        value = Decimal(str(client.metadata.get_cost(**item.request())))  # type: ignore[attr-defined]
        if not value.is_finite() or value < 0:
            raise AcquisitionError(f"invalid fresh cost for {item.relative_path}: {value}")
        by_path[item.relative_path] = str(value)
        month = item.target_session.strftime("%Y-%m")
        monthly[month][component_key[item.purpose]] += value
    month_payload: dict[str, dict[str, str]] = {}
    for month, values in monthly.items():
        total = sum(values.values(), Decimal(0))
        month_payload[month] = {**{key: str(value) for key, value in values.items()}, "TOTAL_USD": str(total)}
    total = sum((Decimal(value) for value in by_path.values()), Decimal(0))
    _validate_fresh_total(total)
    return {
        "cost_by_relative_path_usd": by_path,
        "monthly_totals_usd": month_payload,
        "total_usd": str(total),
        "approved_total_usd": str(APPROVED_TOTAL_USD),
        "hard_cap_usd": str(HARD_CAP_USD),
        "deviation_usd": str(abs(total - APPROVED_TOTAL_USD)),
    }


def _component_record(item: Component, fresh_cost: str) -> dict[str, Any]:
    return {
        "target_session": item.target_session.isoformat(),
        "prior_rth_date": item.prior_rth_date.isoformat() if item.prior_rth_date else None,
        "purpose": item.purpose,
        "dataset": DATASET,
        "schema": item.schema,
        "raw_symbol": item.raw_symbol,
        "start_utc": item.start,
        "end_utc": item.end,
        "local_path": item.relative_path,
        "approved_component_cost_usd": str(item.approved_cost_usd),
        "fresh_component_cost_usd": fresh_cost,
    }


def _initial_manifest(frozen: FrozenQuote, items: Sequence[Component], quote: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_day = {row.target_session.isoformat(): row for row in frozen.rows}
    return {
        "status": "ACQUISITION_IN_PROGRESS",
        "strategy_id": STRATEGY_ID,
        "v3_contract_sha256": EXPECTED_V3_HASH,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "dataset": DATASET,
        "target_session_count": EXPECTED_SESSION_COUNT,
        "first_target_session": EXPECTED_FIRST_SESSION,
        "last_target_session": EXPECTED_LAST_SESSION,
        "prior_rth_by_target_session": {
            day: row.prior_rth_date.isoformat() for day, row in rows_by_day.items()
        },
        "quote_artifact": {
            "path": str(frozen.source_json),
            "sha256": frozen.source_sha256,
            "scenario": SCENARIO_NAME,
            "approved_total_usd": str(APPROVED_TOTAL_USD),
            "approved_december_usd": str(APPROVED_DECEMBER_USD),
            "approved_january_usd": str(APPROVED_JANUARY_USD),
        },
        "fresh_quote": dict(quote),
        "contract_rolls": [{
            "effective_target_session": "2025-12-17",
            "target_from": "ESZ5/MESZ5",
            "target_to": "ESH6/MESH6",
            "prior_rth_date": "2025-12-16",
            "prior_rth_es_raw_symbol": "ESZ5",
        }],
        "components": [
            _component_record(item, quote["cost_by_relative_path_usd"][item.relative_path]) for item in items
        ],
        "files": {},
        "constraints": {
            "no_mbo": True,
            "no_strategy_replay": True,
            "no_outcomes_inspected": True,
            "february_excluded": True,
            "acquisition_specification_frozen_before_outcomes": True,
            "future_uses_not_executed": [
                "frozen V3 quality >= 0.50",
                "predeclared V4 challenger quality >= 0.45",
                "pre-quality-gate POC master dataset for offline weight research",
            ],
        },
    }


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / MANIFEST_NAME
    temporary = destination.with_suffix(".json.part")
    if temporary.exists():
        raise AcquisitionError(f"stale manifest partial exists: {temporary}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _load_existing_manifest(root: Path) -> dict[str, Any] | None:
    path = root / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError("existing acquisition manifest is invalid") from exc
    if payload.get("strategy_id") != STRATEGY_ID or payload.get("v3_contract_sha256") != EXPECTED_V3_HASH:
        raise AcquisitionError("existing acquisition manifest identity mismatch")
    return payload


def _assert_output_tree(root: Path, items: Sequence[Component], manifest: Mapping[str, Any] | None) -> None:
    if not root.exists():
        return
    allowed_posix = {MANIFEST_NAME} | {item.relative_path for item in items}
    recorded = set((manifest or {}).get("files", {}))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".part"):
            raise AcquisitionError(f"stale partial file exists: {relative}")
        if relative not in allowed_posix:
            raise AcquisitionError(f"unknown existing file: {relative}")
        if relative != MANIFEST_NAME and relative not in recorded:
            raise AcquisitionError(f"unrecorded existing data file: {relative}")
    if recorded - allowed_posix:
        raise AcquisitionError("manifest records files outside the frozen acquisition set")


def _verified_existing(root: Path, item: Component, manifest: Mapping[str, Any]) -> bool:
    destination = root / Path(item.relative_path)
    record = manifest.get("files", {}).get(item.relative_path)
    if not destination.exists():
        if record is not None:
            raise AcquisitionError(f"manifest-recorded file is missing: {item.relative_path}")
        return False
    if record is None:
        raise AcquisitionError(f"existing file is not recorded: {item.relative_path}")
    actual_size, actual_hash = destination.stat().st_size, _sha256(destination)
    if actual_size <= 0 or actual_size != int(record.get("bytes", -1)) or actual_hash != record.get("sha256"):
        raise AcquisitionError(f"existing file hash/size mismatch: {item.relative_path}")
    return True


def _download_one(
    *, client: object, root: Path, item: Component, manifest: MutableMapping[str, Any], fresh_cost: str
) -> str:
    destination = root / Path(item.relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise AcquisitionError(f"stale partial file exists: {partial}")
    if _verified_existing(root, item, manifest):
        return "SKIPPED_VERIFIED"
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            client.timeseries.get_range(**item.request(), path=str(partial))  # type: ignore[attr-defined]
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise AcquisitionError(f"empty download for {item.relative_path}")
            size, digest = partial.stat().st_size, _sha256(partial)
            partial.replace(destination)
            manifest.setdefault("files", {})[item.relative_path] = {
                **_component_record(item, fresh_cost),
                "bytes": size,
                "sha256": digest,
                "status": "DOWNLOADED_VERIFIED",
            }
            _write_manifest(root, manifest)
            return "DOWNLOADED_VERIFIED"
        except Exception as exc:  # retry provider/file errors deterministically
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                break
    raise AcquisitionError(f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: {item.relative_path}") from last_error


def run(
    *, client: object, quote_json: Path = DEFAULT_QUOTE_JSON, output_root: Path = DEFAULT_OUTPUT_ROOT,
    download: bool = False,
) -> dict[str, Any]:
    frozen = load_frozen_quote(quote_json)
    items = components(frozen)
    fresh = fresh_quote(client, frozen, items)
    result: dict[str, Any] = {
        "status": "PREFLIGHT_QUOTE_COMPLETE_DOWNLOAD_NOT_REQUESTED",
        "download_requested": download,
        "download_api_invoked": False,
        "strategy_replay_executed": False,
        "outcomes_inspected": False,
        "target_session_count": len(frozen.rows),
        "component_count": len(items),
        "first_target_session": frozen.rows[0].target_session.isoformat(),
        "last_target_session": frozen.rows[-1].target_session.isoformat(),
        "fresh_quote": fresh,
        "output_root": str(output_root),
    }
    if not download:
        return result

    existing = _load_existing_manifest(output_root)
    _assert_output_tree(output_root, items, existing)
    manifest: MutableMapping[str, Any]
    if existing is None:
        manifest = _initial_manifest(frozen, items, fresh)
        _write_manifest(output_root, manifest)
    else:
        if existing.get("quote_artifact", {}).get("sha256") != frozen.source_sha256:
            raise AcquisitionError("resume quote artifact hash mismatch")
        if existing.get("fresh_quote") != fresh:
            raise AcquisitionError("resume fresh quote differs from sealed acquisition manifest")
        manifest = existing
    statuses = []
    for item in items:
        statuses.append(_download_one(
            client=client,
            root=output_root,
            item=item,
            manifest=manifest,
            fresh_cost=fresh["cost_by_relative_path_usd"][item.relative_path],
        ))
    manifest["status"] = "ACQUISITION_COMPLETE_VERIFIED"
    manifest["downloaded_file_count"] = len(manifest.get("files", {}))
    _write_manifest(output_root, manifest)
    result.update({
        "status": "ACQUISITION_COMPLETE_VERIFIED",
        "download_api_invoked": any(value == "DOWNLOADED_VERIFIED" for value in statuses),
        "downloaded_or_verified_file_count": len(statuses),
    })
    return result


def _print_preflight(result: Mapping[str, Any]) -> None:
    fresh = result["fresh_quote"]
    print(f"sessions={result['target_session_count']} components={result['component_count']}")
    for month, values in fresh["monthly_totals_usd"].items():
        print(f"{month} total_usd={values['TOTAL_USD']} components={json.dumps(values, sort_keys=True)}")
    print(
        f"fresh_total_usd={fresh['total_usd']} approved_total_usd={fresh['approved_total_usd']} "
        f"deviation_usd={fresh['deviation_usd']} hard_cap_usd={fresh['hard_cap_usd']}"
    )
    print(f"status={result['status']} output_root={result['output_root']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-json", type=Path, default=DEFAULT_QUOTE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download", action="store_true", help="explicitly authorize the frozen 126-file acquisition")
    args = parser.parse_args(argv)
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise AcquisitionError("DATABENTO_API_KEY is required and will not be printed")
    try:
        import databento as db
    except ImportError as exc:
        raise AcquisitionError("databento package is required") from exc
    client = db.Historical(api_key)
    result = run(client=client, quote_json=args.quote_json, output_root=args.output_root, download=args.download)
    _print_preflight(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
