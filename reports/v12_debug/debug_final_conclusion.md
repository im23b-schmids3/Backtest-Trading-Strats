# V12 deterministic single-trader diagnostic conclusion

## Answers

1. **Is Evaluation pass detection executed?** Yes. It is the final branch of `SingleTrader._exit_for_account()` after `_apply_exit()` and the MLL check. The representative replay invoked the exit handler 46 times and the pass handler zero times because the balance never reached `$26,500`.

2. **Is Evaluation failure detection executed?** Yes, through the preceding `account.balance <= account.mll` branch. It was reached as a check 46 times, but the balance never fell to the `$24,000` MLL.

3. **What updates state after a completed trade?** `run()` dispatches each exit leg to `_exit_for_account()`. That calls `_apply_exit()` to update balance/PnL/fees, then performs DLG, MLL, and pass checks. A threshold hit calls the corresponding lifecycle handler.

4. **Is it called?** Yes. Instrumentation confirmed 46 exit-handler calls in the one representative replay.

5. **Why did the account stay ACTIVE?** Its realized trading PnL was only `-$2.8132`. It was far below the `$1,500` pass profit and far above the `$1,000` MLL loss. Subscription rebills therefore continued by design.

6. **Is this the reported implementation bug?** No. The zero-pass/zero-failure/one-active result is explained by the thresholds not being reached. There is, however, a separate reporting-definition issue: `total_trades` means accepted entries, not completed trades, and a boundary limitation because `_flatten()` does not itself re-run threshold checks.

7. **Exact responsible location:** `src/fib_backtester/research/v12_single_trader.py`, class `SingleTrader`, method `_exit_for_account` (threshold dispatch), with `_apply_exit`, `_handle_eval_pass`, and `_handle_eval_failure` implementing the update and transitions.

8. **Missing call or incorrect logic?** Neither explains the observed result. The call is present and the threshold logic is internally coherent. The smallest corrective change for the separate boundary limitation would be to centralize the DLG/MLL/pass evaluation in a helper and call it after `_flatten()` when a forced flatten changes the account balance; that change is not implemented here.

9. **Can this explain zero payouts?** Yes, indirectly and completely: no Evaluation passed, so no Qualified account was created; with no Qualified account, no Qualified payout could occur. The causal chain is “no threshold pass” → “no Qualified” → “zero payouts,” not “missing Evaluation call.”

10. **Smallest code change:** Do not change the strategy. For the observed behavior, no fix is required. For correctness at forced-flatten boundaries, add one shared post-balance threshold-check helper and invoke it after forced flattening (or explicitly document that forced/end flattening cannot complete an Evaluation). This is a lifecycle-engine change, not a trading-rule change.

## Classification

The deterministic report is internally explainable. The audit found no evidence that the Evaluation engine was forgotten or that pass/failure transitions were unreachable. The only confirmed discrepancies are metric naming (`total_trades`) and the un-rechecked forced-flatten boundary.
