from __future__ import annotations

from decimal import Decimal

from .constants import TERMINAL_OUTCOMES


TOLERANCE = Decimal("0.0000000001")


def _sum(rows: list[dict], key: str) -> Decimal:
 return sum((row.get(key, Decimal()) for row in rows), Decimal())


def _same(expected: Decimal, actual: Decimal) -> bool:
 return abs(expected - actual) <= TOLERANCE


def _unique(rows: list[dict], key: str) -> bool:
 values = [row.get(key) for row in rows]
 return None not in values and len(values) == len(set(values))


def reconcile(setups: list[dict], outcomes: list[dict], orders: list[dict], trades: list[dict], opening_equity: Decimal, *, final_equity: Decimal | None = None, events: list[dict] | None = None) -> dict:
 """Fail closed on every sealed setup, lifecycle, quantity, and cash identity."""
 events = events or []
 setup_ids = [row.get("setup_id") for row in setups]
 outcome_ids = [row.get("setup_id") for row in outcomes]
 order_ids = {row.get("order_id") for row in orders}
 trade_ids = {row.get("trade_id") for row in trades}
 executed = {row["setup_id"] for row in outcomes if row.get("disposition") == "TRADE_EXECUTED"}
 trade_setup_ids = {row.get("setup_id") for row in trades}
 unique_setup_ids = _unique(setups, "setup_id")
 unique_outcome_ids = _unique(outcomes, "setup_id")
 one_terminal = unique_setup_ids and unique_outcome_ids and len(setups) == len(outcomes) and set(setup_ids) == set(outcome_ids) and all(row.get("disposition") in TERMINAL_OUTCOMES for row in outcomes)
 order_links = _unique(orders, "order_id") and all(row.get("setup_id") in set(setup_ids) for row in orders)
 trade_links = _unique(trades, "trade_id") and all(row.get("setup_id") in set(setup_ids) and row.get("order_id") in order_ids for row in trades)
 event_links = all(event.get("setup_id") in set(setup_ids) and event.get("order_id") in order_ids and (event.get("kind") != "ORDER_FILLED" or event.get("trade_id") in trade_ids) for event in events)
 quantity = all(_same(trade["quantity"], _sum(trade["legs"], "quantity") + trade["remaining_quantity"]) for trade in trades)
 gross = all(_same(trade["gross_pnl"], _sum(trade["legs"], "gross_pnl")) for trade in trades)
 net = all(_same(trade["net_pnl"], trade["gross_pnl"] - trade["entry_fee"] - _sum(trade["legs"], "fee")) for trade in trades)
 expected_final = opening_equity + _sum(trades, "net_pnl")
 final = expected_final if final_equity is None else final_equity
 equity = _same(expected_final, final)
 checks = {"unique_setup_ids": unique_setup_ids, "one_terminal_outcome": one_terminal, "trade_outcomes_reconcile": executed == trade_setup_ids, "order_parent_links": order_links, "trade_parent_links": trade_links, "event_links": event_links, "quantity_conservation": quantity, "gross_pnl_reconcile": gross, "pnl_reconcile": net, "equity_reconcile": equity}
 return {"reconciles": all(checks.values()), **checks, "final_equity": final, "expected_final_equity": expected_final, "tolerance": TOLERANCE}
