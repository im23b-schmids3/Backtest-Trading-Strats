"""Development-only, price-free Fib09 reconciliation diagnostics."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from .constants import CANDIDATES
from .loader import load_development_bars
from .manifests import verify_chronology_manifest
from .models import Candidate, ExecutionAssumptions
from .runner import run_candidate

ZERO = Decimal()

def _sum(rows, key): return sum((row.get(key, ZERO) for row in rows), ZERO)
def _duplicates(values):
 duplicates = sorted({value for value in values if values.count(value) > 1})
 return {"count": len(duplicates), "first_id": duplicates[0] if duplicates else None}

def _candidate_diagnostic(item: dict, candidate_id: str, opening_equity: Decimal) -> dict:
    setups, outcomes, orders, trades, events = (item[key] for key in ("setups", "setup_outcomes", "orders", "trades", "events"))
    legs = [leg for trade in trades for leg in trade["legs"]]
    setup_ids, outcome_ids = [row["setup_id"] for row in setups], [row.get("setup_id") for row in outcomes]
    order_ids, trade_ids = [row.get("order_id") for row in orders], [row.get("trade_id") for row in trades]
    exit_leg_ids = [row.get("exit_leg_id") for row in legs]
    terminal = {row["setup_id"]: row["disposition"] for row in outcomes}
    submitted, filled = [event for event in events if event["kind"] == "ORDER_SUBMITTED"], [event for event in events if event["kind"] == "ORDER_FILLED"]
    missing = {"outcome_setup_ids": sorted(set(setup_ids) - set(outcome_ids)), "order_setup_ids": sorted(set(row["setup_id"] for row in orders) - set(setup_ids)), "trade_setup_ids": sorted(set(row["setup_id"] for row in trades) - set(setup_ids)), "exit_leg_trade_ids": sorted(set(row["trade_id"] for row in legs) - set(trade_ids)), "executed_without_trade": sorted({sid for sid, disposition in terminal.items() if disposition == "TRADE_EXECUTED"} - set(row["setup_id"] for row in trades))}
    quantity_failures, pnl_failures = [], []
    for trade in trades:
        exited = _sum(trade["legs"], "quantity"); actual_quantity = exited + trade["remaining_quantity"]
        if trade["quantity"] != actual_quantity:
            quantity_failures.append({"candidate_id": candidate_id, "setup_id": trade["setup_id"], "order_id": trade["order_id"], "trade_id": trade["trade_id"], "invariant_name": "quantity_conservation", "expected": trade["quantity"], "actual": actual_quantity, "difference": trade["quantity"] - actual_quantity, "relevant_timestamps": [trade["entry_timestamp"]], "relevant_exit_legs": [{"exit_leg_id": leg["exit_leg_id"], "timestamp": leg["timestamp"], "reason": leg["reason"]} for leg in trade["legs"]]})
        expected_net = trade["gross_pnl"] - trade["entry_fee"] - _sum(trade["legs"], "fee")
        if expected_net != trade["net_pnl"]:
            pnl_failures.append({"candidate_id": candidate_id, "setup_id": trade["setup_id"], "order_id": trade["order_id"], "trade_id": trade["trade_id"], "invariant_name": "trade_net_pnl", "expected": expected_net, "actual": trade["net_pnl"], "difference": expected_net - trade["net_pnl"], "relevant_timestamps": [trade["entry_timestamp"]], "relevant_exit_legs": [{"exit_leg_id": leg["exit_leg_id"], "timestamp": leg["timestamp"], "reason": leg["reason"]} for leg in trade["legs"]]})
    trade_net = _sum(trades, "net_pnl"); equity_delta = item["metrics"]["final_equity"] - opening_equity
    failure = next(iter(quantity_failures + pnl_failures), None)
    duplicate_setup = _duplicates(setup_ids)
    duplicate_outcome = _duplicates(outcome_ids)
    if failure is None and duplicate_setup["first_id"]:
        matching = [row for row in setups if row["setup_id"] == duplicate_setup["first_id"]]
        failure = {"candidate_id": candidate_id, "setup_id": duplicate_setup["first_id"], "order_id": None, "trade_id": None, "invariant_name": "unique_proposed_setup_ids", "expected": 1, "actual": len(matching), "difference": len(matching) - 1, "relevant_timestamps": [value for row in matching for value in (row.get("anchor_timestamp"), row.get("extreme_timestamp")) if value is not None], "relevant_exit_legs": []}
    if failure is None and duplicate_outcome["first_id"]:
        failure = {"candidate_id": candidate_id, "setup_id": duplicate_outcome["first_id"], "order_id": None, "trade_id": None, "invariant_name": "unique_terminal_setup_ids", "expected": 1, "actual": 2, "difference": 1, "relevant_timestamps": [], "relevant_exit_legs": []}
    if failure is None:
        for invariant, expected, actual in (("one_terminal_outcome", len(setup_ids), len(outcomes)), ("unique_terminal_setup_ids", len(outcomes), len(set(outcome_ids))), ("trade_equity_delta", trade_net, equity_delta)):
            if expected != actual:
                failure = {"candidate_id": candidate_id, "setup_id": None, "order_id": None, "trade_id": None, "invariant_name": invariant, "expected": expected, "actual": actual, "difference": expected - actual, "relevant_timestamps": [], "relevant_exit_legs": []}; break
    entry_fees, exit_fees = _sum(trades, "entry_fee"), _sum(legs, "fee")
    entry_slippage = _sum(trades, "slippage_cost") - _sum(legs, "slippage_cost")
    return {"candidate_id": candidate_id, "counts": {"proposed_setups": len(setups), "terminal_outcomes": len(outcomes), "submitted_orders": len(submitted), "activated_orders": sum(row.get("active_timestamp") is not None for row in orders), "filled_orders": len(filled), "executed_trades": len(trades), "trade_records": len(trades), "exit_legs": len(legs), "event_count": len(events), "expected_event_count": len(submitted) + len(filled)}, "quantity": {"initial": _sum(trades, "quantity"), "exited": _sum(legs, "quantity"), "remaining": _sum(trades, "remaining_quantity"), "failure_count": len(quantity_failures)}, "pnl": {"gross_from_trade_records": _sum(trades, "gross_pnl"), "gross_from_exit_legs": _sum(legs, "gross_pnl"), "fees_from_fills": _sum(trades, "fees"), "fees_from_accounting": entry_fees + exit_fees, "entry_fees": entry_fees, "exit_leg_fees": exit_fees, "slippage_from_fills": _sum(trades, "slippage_cost"), "slippage_from_accounting": entry_slippage + _sum(legs, "slippage_cost"), "net_from_trades": trade_net, "net_from_exit_legs_after_entry_fees": _sum(legs, "net_pnl") - entry_fees, "net_from_equity_delta": equity_delta, "failure_count": len(pnl_failures)}, "ids": {"duplicate_setup_ids": duplicate_setup, "duplicate_outcome_setup_ids": duplicate_outcome, "duplicate_order_ids": _duplicates(order_ids), "duplicate_trade_ids": _duplicates(trade_ids), "duplicate_exit_leg_ids": _duplicates(exit_leg_ids), "missing": {key: {"count": len(value), "first_id": value[0] if value else None} for key, value in missing.items()}}, "lifecycle": {"order_parent_links_valid": all(row["setup_id"] in set(setup_ids) for row in orders), "trade_parent_links_valid": all(row["setup_id"] in set(setup_ids) and row["order_id"] in set(order_ids) for row in trades), "event_order_links_valid": all(event.get("order_id") in set(order_ids) for event in events), "event_trade_links_valid": all(event.get("kind") != "ORDER_FILLED" or event.get("trade_id") in set(trade_ids) for event in events), "trades_have_executed_terminal_outcome": all(terminal.get(row["setup_id"]) == "TRADE_EXECUTED" for row in trades)}, "reconciles": item["reconciliation"]["reconciles"], "RECONCILIATION_FIRST_FAILURE": failure}

def development_reconciliation_diagnostic(*, eth_manifest: str | Path, btc_manifest: str | Path, chronology_manifest: str | Path) -> dict:
    """Run candidates against sealed development bars only; never writes artifacts."""
    chronology = verify_chronology_manifest(chronology_manifest, eth_manifest=eth_manifest, btc_manifest=btc_manifest)
    eth_bars, _ = load_development_bars(eth_manifest, development_start=chronology["development_start"], development_end=chronology["development_end"], chronology_claim=chronology["assets"]["ETH"])
    btc_bars, _ = load_development_bars(btc_manifest, development_start=chronology["development_start"], development_end=chronology["development_end"], chronology_claim=chronology["assets"]["BTC"])
    assumptions, candidates = ExecutionAssumptions(), []
    for row in CANDIDATES:
        item = run_candidate(eth_bars if row["symbol"] == "ETH" else btc_bars, Candidate(**row), assumptions)
        candidates.append(_candidate_diagnostic(item, row["candidate_id"], assumptions.opening_equity))
    first = next((row["RECONCILIATION_FIRST_FAILURE"] for row in candidates if row["RECONCILIATION_FIRST_FAILURE"]), None)
    return {"mode": "DEVELOPMENT_ONLY_DIAGNOSTIC", "holdout_status": "LOCKED_NOT_OPENED", "artifact_root_created": False, "candidate_count": len(candidates), "reconciles": all(row["reconciles"] for row in candidates), "RECONCILIATION_FIRST_FAILURE": first, "candidates": candidates}
