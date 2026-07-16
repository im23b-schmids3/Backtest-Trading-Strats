# V13 lifecycle root-cause audit

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
