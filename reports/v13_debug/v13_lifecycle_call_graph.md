# V13 lifecycle call graph audit

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
