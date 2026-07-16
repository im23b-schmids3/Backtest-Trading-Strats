# Dead-code and alternate-engine audit

## Lifecycle functions in the deterministic simulator

The following functions are called by `SingleTrader` when their conditions are met:

- `_buy_evaluation` — called by `run()` and by replacement/new-Evaluation branches.
- `_handle_eval_pass` — called by `_exit_for_account` on the profit-target condition.
- `_handle_eval_failure` — called by `_exit_for_account` on the Evaluation MLL condition.
- `_start_qualified` — called by `_handle_eval_pass`.
- `_handle_qualified_failure` — called by `_exit_for_account` for a Qualified MLL breach.
- `_apply_exit` — called by `_exit_for_account` and `_flatten`.

No pass, failure, or Qualified-transition function that belongs to `v12_single_trader.py` is dead with respect to the simulator. The zero counts are conditional outcomes, not evidence that these methods were disconnected.

## Separate economics engine

`src/fib_backtester/research/v12_economics_fixed.py` has a separate `_run_stage()` implementation. It contains its own pass/failure checks and is called by that module's `_simulate_path()` for the economics study. `v12_single_trader.py` imports `v12_economics_fixed` only to prepare the retained trade streams; it does not call `_run_stage()` to run the deterministic account. This is intentional separation, not a missing call in the single-trader lifecycle.

## Reporting caveat

`SingleTrader.total_trades` is incremented in `_accept()` and therefore counts accepted entries, not fully closed trades. The representative report shows 23 accepted entries, while the trace has 8 positions that completed all their planned exit legs during the year. Other positions were force-flattened at the period boundary. This naming/definition issue can make the report look inconsistent, but it does not suppress pass/failure checks for normal exit legs.

