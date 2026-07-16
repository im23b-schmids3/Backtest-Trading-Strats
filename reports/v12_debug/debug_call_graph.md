# V12 deterministic single-trader call graph

## Result

Evaluation pass/failure detection is connected and is executed after each applicable exit leg. It is not evaluated only at month-end or only at simulation end.

The relevant path is:

```text
SingleTrader.run()
  -> _exit_for_account(account, signal_id, leg_index, trade, timestamp)
       -> _apply_exit(account, trade, leg, timestamp)
            -> account.balance / PnL / fees / daily_profit are updated
       -> Daily Loss Guard check
       -> MLL check
            -> _handle_eval_failure(...) for an Evaluation
            -> _handle_qualified_failure(...) for a Qualified account
       -> profit-target check
            -> _handle_eval_pass(timestamp)
```

The entry path is `run() -> _accept()`. Accepted entries are charged and stored in `account.positions`; subsequent exit events are sent to `_exit_for_account` for each eligible account. An exit event for a signal with no position is ignored by the first guard in `_exit_for_account`.

## Exact checks

In `src/fib_backtester/research/v12_single_trader.py`, `SingleTrader._exit_for_account` contains:

```python
if account.balance <= account.mll:
    if account.kind == "EVALUATION":
        self._handle_eval_failure(timestamp, "Maximum Loss Limit")
    else:
        self._handle_qualified_failure(timestamp)
elif account.kind == "EVALUATION" and account.balance >= self.spec.account_size + self.spec.target:
    self._handle_eval_pass(timestamp)
```

For the representative `25K Zero | ETH only | 1h | 2 micros | MIRRORED` replay, the instrumented run observed 46 calls to `_exit_for_account`, zero calls to `_handle_eval_pass`, and zero calls to `_handle_eval_failure`. This proves the checks were reached; neither threshold was met.

The monthly subscription path is separate: `_buy_evaluation()` calls `_charge_subscription()` immediately, and `_before_event()` calls it before subsequent events. `_charge_subscription()` rebills while the Evaluation state is active or daily-locked and stops when the Evaluation is closed by pass/failure.

