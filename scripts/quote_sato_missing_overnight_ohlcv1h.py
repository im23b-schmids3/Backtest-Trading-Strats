#!/usr/bin/env python3
"""Quote the frozen Sato missing overnight intervals. No records are downloaded."""
from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any


QUOTE_ONLY = True
# THIS SCRIPT MUST NEVER DOWNLOAD DATA.
PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "research_runs/CMEOrderflow_SATO_ES_OVERNIGHT_OHLCV1H_QUOTE_PREP_V1"
    / "missing-overnight-hourly-quote-plan.json"
)


def _load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "PREPARED_LOCAL_EVIDENCE_NO_QUOTE_EXECUTED":
        raise ValueError("quote plan is not in the frozen prepared state")
    if plan.get("dataset") != "GLBX.MDP3" or plan.get("schema") != "ohlcv-1h" or plan.get("stype_in") != "raw_symbol":
        raise ValueError("quote plan dataset/schema/symbology mismatch")
    rows = plan.get("requests")
    if not isinstance(rows, list) or len(rows) != 54:
        raise ValueError("quote plan must contain exactly 54 approved requests")
    for row in rows:
        if row.get("STATUS") != "APPROVED_FOR_COST_QUOTE":
            raise ValueError("quote plan contains an unapproved/ambiguous request")
        if row.get("SYMBOL_STATUS") != "VERIFIED_LOCAL_NATIVE_RAW_SYMBOL":
            raise ValueError("quote plan contains an unresolved or ambiguous contract")
        if row.get("DATASET") != "GLBX.MDP3" or row.get("SCHEMA") != "ohlcv-1h" or row.get("STYPE_IN") != "raw_symbol":
            raise ValueError("request dataset/schema/symbology mismatch")
        if not row.get("RAW_ES_SYMBOL") or not row.get("EXPECTED_HOURLY_BAR_STARTS_UTC"):
            raise ValueError("request is missing symbol or expected hourly bars")
    return plan


def _print_plan(plan: dict[str, Any]) -> None:
    for row in plan["requests"]:
        print(
            f"{row['CURRENT_RTH_DATE']} {row['RAW_ES_SYMBOL']} "
            f"{row['PATCH_START_UTC']} -> {row['PATCH_END_UTC']} "
            f"EXPECTED_BARS={row['EXPECTED_HOURLY_BAR_COUNT']}"
        )
    print(f"REQUESTS_IN_PLAN = {len(plan['requests'])}")
    print(f"EXPECTED_HOURLY_BARS = {plan['total_missing_hours']}")
    print("NO_DATA_WAS_DOWNLOADED = true")
    print("NETWORK_CALLS = 0 (--show-plan)")


def _quote(plan: dict[str, Any]) -> int:
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("DATABENTO_API_KEY is not set.\nSet it in PowerShell first:\n\n$env:DATABENTO_API_KEY=\"<YOUR_KEY>\"")
        return 2

    import databento as db

    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    total = Decimal("0")
    for row in plan["requests"]:
        amount = Decimal(str(client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=[row["RAW_ES_SYMBOL"]],
            schema="ohlcv-1h",
            stype_in="raw_symbol",
            start=row["PATCH_START_UTC"],
            end=row["PATCH_END_UTC"],
        )))
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"invalid cost estimate for {row['CURRENT_RTH_DATE']}: {amount}")
        total += amount
        print(
            f"CURRENT_RTH_DATE={row['CURRENT_RTH_DATE']} RAW_ES_SYMBOL={row['RAW_ES_SYMBOL']} "
            f"START={row['PATCH_START_UTC']} END={row['PATCH_END_UTC']} "
            f"EXPECTED_BARS={row['EXPECTED_HOURLY_BAR_COUNT']} QUOTED_USD={amount:.12f}"
        )
    print(f"REQUESTS_QUOTED = {len(plan['requests'])}")
    print(f"EXPECTED_HOURLY_BARS = {plan['total_missing_hours']}")
    print(f"TOTAL_QUOTED_COST_USD = ${total:.12f}")
    print(f"Estimated total: ${total:.2f} USD")
    print("NO_DATA_WAS_DOWNLOADED = true")
    print("ENDPOINT_USED = Historical.metadata.get_cost")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-plan", action="store_true", help="print the frozen local request plan; no network or API key")
    args = parser.parse_args(argv)
    try:
        plan = _load_plan()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_plan:
        _print_plan(plan)
        return 0
    return _quote(plan)


if __name__ == "__main__":
    raise SystemExit(main())
