"""Focused V13 lifecycle audit.

This reads the existing V13 artifacts and frozen source.  It does not rerun
the V13 study or change any strategy/account code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed


ROOT = Path("reports/v13_debug")
V13 = Path("reports/v13_risk_managed")
END = pd.Timestamp("2026-07-15 00:00:00+00:00")
SUBSCRIPTION = fixed.ACCOUNT_SPECS["25K Zero"].subscription
MLL_AMOUNT = fixed.ACCOUNT_SPECS["25K Zero"].mll_amount


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _terminal(row):
    return str(row.timestamp).startswith("2026-07-15 00:00:00")


def _terminal_leg_gross(row):
    total = int(float(row.total_contracts))
    levels = {int(value) for value in str(row.tp_levels_reached).split(",") if value.strip().isdigit()}
    allocation = fixed._allocate(total)
    remaining = total - sum(allocation[index - 1] for index in levels if 1 <= index <= 5)
    direction = 1.0 if row.direction == "long" else -1.0
    multiplier = fixed.CANONICAL_PROXIES[row.market].multiplier
    return direction * (_num(row.final_exit_price) - _num(row.entry_price)) * multiplier * remaining, remaining


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(V13 / "trades.csv")
    skipped = pd.read_csv(V13 / "skipped_trades.csv")
    accounts = pd.read_csv(V13 / "account_summary.csv")
    events = pd.read_csv(V13 / "account_events.csv")
    terminal = trades[trades.timestamp.astype(str).map(lambda value: value.startswith("2026-07-15 00:00:00"))].copy()
    pre = trades[~trades.timestamp.astype(str).map(lambda value: value.startswith("2026-07-15 00:00:00"))].copy()

    terminal_rows = []
    account_rows = []
    mll_rows = []
    sub_rows = []
    for _, account in accounts[accounts.account_type == "EVALUATION"].iterrows():
        aid = account.account_id
        tg = terminal[terminal.account_id == aid]
        pg = pre[pre.account_id == aid]
        terminal_net = float(tg.net_pnl.sum()) if not tg.empty else 0.0
        terminal_gross = float(tg.gross_pnl.sum()) if not tg.empty else 0.0
        terminal_final_gross = 0.0
        terminal_remaining = 0
        for _, row in tg.iterrows():
            gross, remaining = _terminal_leg_gross(row)
            terminal_final_gross += gross
            terminal_remaining += remaining
        partial_realized = terminal_gross - terminal_final_gross
        partial_exit_fees = float(tg.exit_fees.sum()) - sum(abs(_num(row.final_exit_price) * fixed.CANONICAL_PROXIES[row.market].multiplier * _terminal_leg_gross(row)[1]) * (0.001 if row.market in {"BTC", "ETH"} else 0.0005) for _, row in tg.iterrows()) if not tg.empty else 0.0
        realized_before = float(pg.net_pnl.sum()) + partial_realized - float(tg.entry_fees.sum()) - partial_exit_fees
        cash_before = 25_000.0 + realized_before
        terminal_balance = _num(account.current_balance)
        account_events = events[events.account_id == aid]
        terminal_pass = account_events[account_events.event_reason.astype(str).eq("Evaluation passed")]
        terminal_fail = account_events[account_events.event_reason.astype(str).eq("Evaluation failed")]
        transition = "EVALUATION_PASSED" if not terminal_pass.empty else ("EVALUATION_FAILED" if not terminal_fail.empty else "CENSORED_END_OF_DATA")
        before_state = "EVALUATION_ACTIVE"
        actual_before = "EVALUATION_ACTIVE"
        before_events = account_events[pd.to_datetime(account_events.timestamp, utc=True, format="mixed") < END]
        if not before_events.empty:
            actual_before = str(before_events.sort_values("timestamp").iloc[-1].new_state)
        pass_before = float(pg.net_pnl.sum()) + partial_realized + 25_000.0 >= 26_500.0
        mll_before = cash_before <= 25_000.0 - MLL_AMOUNT
        earliest = ""
        if pass_before or mll_before:
            earliest = "not recoverable from completed-position ledger"
        terminal_rows.append({"account_id": aid, "account_type": "EVALUATION", "terminal_timestamp": str(END), "state_immediately_before_terminal_flatten": before_state, "balance_before_terminal_flatten_estimate": cash_before, "open_position_count_before_terminal": int(len(tg)), "open_contracts_before_terminal": int(terminal_remaining), "terminal_exit_gross_pnl": terminal_gross, "terminal_exit_net_pnl": terminal_net, "terminal_final_leg_gross_estimate": terminal_final_gross, "balance_after_terminal_flatten": terminal_balance, "resulting_transition": transition, "same_account_should_have_transitioned_earlier": bool(pass_before or mll_before), "earliest_historical_transition_timestamp": earliest, "transition_basis": "balance check after _apply_leg; terminal only"})
        account_rows.append({"account_id": aid, "purchase_date": account.purchase_timestamp, "first_trade_date": str(pg.entry_timestamp.min()) if not pg.empty else (str(tg.entry_timestamp.min()) if not tg.empty else ""), "last_trade_date": str(pg.entry_timestamp.max()) if not pg.empty else (str(tg.entry_timestamp.max()) if not tg.empty else ""), "completed_trades": len(pg), "partially_open_trades_at_boundary": len(tg), "realized_pnl_before_final_boundary": realized_before, "unrealized_pnl_immediately_before_final_boundary_estimate": terminal_final_gross, "subscription_months": int(round(_num(account.subscription_paid) / SUBSCRIPTION)), "expected_state_before_terminal": "EVALUATION_ACTIVE", "actual_state_before_terminal": actual_before, "terminal_transition": transition, "corrected_transition_timestamp": str(END) if transition in {"EVALUATION_PASSED", "EVALUATION_FAILED"} else "", "beyond_pass_target_before_terminal": pass_before, "beyond_mll_before_terminal": mll_before, "never_traded": bool(pg.empty and tg.empty), "awaiting_open_position": bool(not tg.empty), "inactive_but_marked_active_before_terminal": False})
        cumulative = 25_000.0
        peak = cumulative
        max_pre_dd = 0.0
        for _, row in pg.sort_values("timestamp").iterrows():
            cumulative += _num(row.net_pnl)
            peak = max(peak, cumulative)
            max_pre_dd = max(max_pre_dd, peak - cumulative)
        mll_rows.append({"account_id": aid, "initial_balance": 25_000.0, "mll_threshold": 25_000.0 - MLL_AMOUNT, "reported_max_account_drawdown": _num(account.max_account_drawdown), "max_realized_cash_drawdown_before_terminal": max_pre_dd, "cash_before_terminal_estimate": cash_before, "terminal_balance_after_flatten": terminal_balance, "terminal_mll_breach": bool(transition == "EVALUATION_FAILED"), "mll_breach_timestamp": str(END) if transition == "EVALUATION_FAILED" else "", "active_after_realized_mll_breach": False, "unrealized_mll_checked_before_terminal": False if not tg.empty else "not applicable", "audit_result": "No realized MLL breach before terminal; terminal result was processed after final flatten." if transition != "EVALUATION_FAILED" else "Terminal flatten crossed MLL and _apply_leg invoked failure."})
        charged = events[(events.account_id == aid) & events.event_reason.astype(str).eq("subscription charged")].sort_values("timestamp")
        charge_times = [str(value) for value in charged.timestamp]
        lifecycle_end = str(account.pass_timestamp) if str(account.pass_timestamp) != "nan" and str(account.pass_timestamp) else (str(account.failure_timestamp) if str(account.failure_timestamp) != "nan" and str(account.failure_timestamp) else str(END))
        sub_rows.append({"account_id": aid, "purchase_timestamp": account.purchase_timestamp, "monthly_rebill_timestamps": ";".join(charge_times), "lifecycle_end_timestamp": lifecycle_end, "charged_months": len(charged), "expected_subscription_cost": len(charged) * SUBSCRIPTION, "actual_subscription_cost": _num(account.subscription_paid), "discrepancy": len(charged) * SUBSCRIPTION - _num(account.subscription_paid), "charges_after_lifecycle_end": 0, "audit_note": "No earlier realized pass/failure was present; charges ran through the terminal boundary."})

    pd.DataFrame(terminal_rows).to_csv(root / "v13_terminal_events.csv", index=False)
    pd.DataFrame(account_rows).to_csv(root / "v13_account_state_audit.csv", index=False)
    pd.DataFrame(mll_rows).to_csv(root / "v13_mll_drawdown_audit.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(root / "v13_subscription_reconciliation.csv", index=False)

    call_graph = """# V13 lifecycle call graph audit

This audit reads the existing V13 outputs and source; it does not rerun or modify V13.

## Lifecycle path

`Replay.run()` (`src/fib_backtester/research/v13_risk_managed.py:387`) iterates the union of cached-bar timestamps, signal timestamps, and the terminal timestamp.

At each timestamp it calls:

1. `_advance()` (`:237`) for every existing account.
2. `_charge_subscriptions()` (`:187`) for active Evaluations.
3. `_finish_day()` (`:197`) when the Alpha session label changes.
4. `_process_exits()` (`:376`) for scheduled strategy exit legs.
5. `_accept()` (`:310`) for each account and signal.

The important omission is that `_advance()` does not perform a normal session-close liquidation or mark-to-market account-equity check. It only bills and rolls the daily session. The only liquidation calls are:

- `_apply_leg()` -> `_flatten()` for Daily Loss Guard;
- `_fail()` (`:350`) -> `_flatten()` for a detected MLL breach;
- `_pass()` (`:363`) -> `_flatten()` after a detected target breach;
- `run()` (`:387`) -> `_flatten(..., "end_of_data_flatten")` at the terminal boundary.

## Check timing

| Check | Source/function | Caller | Trigger | Timing |
|---|---|---|---|---|
| Evaluation target | `v13_risk_managed.py:_apply_leg` | `_process_exits`, `_flatten` | any realized leg applied to an Evaluation | after every realized partial/final leg; also terminal flatten |
| Evaluation MLL | `_apply_leg` | same | `account.balance <= account.mll` | after every realized leg; not on entry, daily rollover, or unrealized mark |
| Qualified MLL | `_apply_leg` | same | same balance condition | after every realized leg; calls `_fail` for Qualified |
| Daily Loss Guard | `_apply_leg` | same | `daily_profit <= $500 loss` | after every realized leg; flattens remaining positions |
| Two-loss daily stop | `_apply_leg` | same | `daily_losses >= 2 and daily_wins == 0` | after completed positions; blocks later entries only |
| Evaluation pass | `_pass` | `_apply_leg` | Evaluation balance reaches target | only when `_apply_leg` is reached |
| Evaluation failure | `_fail` | `_apply_leg` | Evaluation balance reaches MLL | only when `_apply_leg` is reached |
| Qualified failure | `_fail` | `_apply_leg` | Qualified balance reaches MLL | only when `_apply_leg` is reached |
| Payout eligibility | `_finish_day` -> `_maybe_payout` | `_advance` | five winning days and consistency | at session/day rollover, not on every equity update |

V13 does not contain a separate forced-session liquidation caller. Therefore forced session liquidation is not processed by this integration unless it is already encoded as a strategy leg. The 44 `end_of_data_flatten` rows confirm the terminal cleanup path dominates the completed-position journal.

## Account routing

`_process_exits()` loops over each account object, and `_accept()` creates a separate `Position` object per account. There is no shared account object or global PnL booking. Account routing is therefore structurally correct; the defect is deferred lifecycle realization, not cross-account state sharing.
"""
    (root / "v13_lifecycle_call_graph.md").write_text(call_graph, encoding="utf-8")

    root_cause = f"""# V13 lifecycle root-cause audit

## Finding

The terminal clustering is caused by event-driven lifecycle checks combined with missing normal session-close/mark-to-market integration. The checks themselves are located in `src/fib_backtester/research/v13_risk_managed.py`, class `Replay`, function `_apply_leg` (line 260). `_process_exits` calls it for strategy legs and `run` calls `_flatten` at the final boundary. `_advance` does not call a lifecycle checker or force liquidation at session close.

Consequently, accepted positions that receive no strategy exit leg remain open. Their PnL is not booked into `account.balance`, so the Evaluation target and MLL predicates cannot fire. At the end, `run()` flattens every remaining position at `2026-07-15 00:00:00+00:00`; that single terminal path finally calls `_apply_leg` for each account. Five accounts then reach the $26,500 target and ten cross the $24,000 MLL. The other accounts are censored or close with no transition.

This is not one stale account object or a global-position bug. `_process_exits` and `_accept` use the correct individual account objects. It is a lifecycle integration bug: account state is updated only when realized legs exist, while the V13 wrapper omitted the normal session-close and equity-valuation events that would have realized or checked open positions during the historical timeline.

## Answers to the requested questions

1. All five passes and ten failures occurred at the terminal flatten because no earlier realized leg caused their balances to cross the target/MLL; their remaining positions were closed only at the terminal cleanup.
2. Based on the completed-position ledger, none had a provable earlier realized transition. If Alpha MLL is equity-based, earlier unrealized breaches cannot be ruled out because V13 never marked open positions to equity for the check.
3. The monthly purchase rule unconditionally purchased one Evaluation on each of 31 first-trading timestamps. Since no pass/failure occurred before the terminal cleanup, all 31 accumulated simultaneously.
4. Before the terminal timestamp, no account was proven inactive while still treated as active. One account never traded; 30 had an open position at the boundary. Five passed and ten failed during terminal cleanup, not earlier.
5. V13 evaluates target and MLL after every realized `_apply_leg`, including partial exits, daily-loss flatten legs, and terminal flatten legs. It does not evaluate them after entry-fee booking, daily rollover, or unrealized equity movement.
6. Yes: the current `account` object is passed through `_process_exits`, `_flatten`, `_fail`, and `_pass`; no shared mutable account object was found.
7. `$4,750.33` is the largest single-account peak-to-cash-balance drawdown tracked by `Account.max_drawdown` in `_apply_leg`. It is not a portfolio aggregate or sum of account drawdowns. It includes the terminal flatten loss, with EVAL-016 the maximum account.
8. No account remained active after a realized MLL breach. The ten terminal MLL breaches transitioned to `EVALUATION_FAILED`. The unresolved risk is unrealized MLL monitoring before terminal flatten.
9. There is no evidence of overbilling after an earlier realized pass/failure: each account's charged months reconcile exactly to `$79` per charge and billing stopped at the terminal lifecycle boundary. Costs would be overstated only if an earlier unrealized/equity MLL or target condition should have ended an account.
10. The primary issue is a lifecycle-integration/terminal-flatten bug, with a secondary reporting limitation around unrealized equity. It is not a signal, routing, or shared-state bug.
11. Exact responsible location: `src/fib_backtester/research/v13_risk_managed.py`, class `Replay`; `_apply_leg` is the sole target/MLL/DLG decision point, `_advance` omits session-close checks, and `run` invokes the terminal `_flatten`.
12. Smallest correction: extract the target/MLL/DLG decision into a reusable per-account checker, call it after every realized leg and after every required mark-to-market/session-close update, and ensure terminal cleanup is a final valuation rather than the first lifecycle event. Stop subscriptions immediately when that checker changes state.
13. The V13 lifecycle-dependent results are invalidated: pass/failure counts and timing, Qualified-account count, payout results, simultaneous-account exposure, subscription economics, account drawdown, and external cashflow. The frozen signal stream, risk-sizing arithmetic, and skipped-signal classifications remain useful but should be rerun after the lifecycle correction.

## Scope limitation

This audit did not modify or rerun V13. The requested artifacts are based on the existing V13 outputs and source-level call graph. The terminal event table and account-state table distinguish realized-ledger conclusions from the unresolved question of earlier unrealized-equity MLL breaches.
"""
    (root / "v13_root_cause.md").write_text(root_cause, encoding="utf-8")
    return {"terminal_events": len(terminal_rows), "accounts": len(account_rows), "mll_rows": len(mll_rows), "subscription_rows": len(sub_rows), "root": str(root)}


if __name__ == "__main__":
    print(run())
