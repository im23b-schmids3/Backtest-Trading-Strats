# V12 deterministic Evaluation state machine

## Evaluation transitions

| From | To | Function | Trigger |
|---|---|---|---|
| `NONE` | `EVALUATION_ACTIVE` | `SingleTrader._buy_evaluation` | Initial purchase, or replacement/new Evaluation after a terminal event |
| `EVALUATION_ACTIVE` | `EVALUATION_DAILY_LOCKED` | `SingleTrader._exit_for_account` | Daily Loss Guard after an exit leg |
| `EVALUATION_DAILY_LOCKED` | `EVALUATION_ACTIVE` | `SingleTrader._finish_session` | Next official trading session |
| `EVALUATION_ACTIVE` or daily-locked | `EVALUATION_FAILED` | `SingleTrader._handle_eval_failure` | `account.balance <= account.mll` |
| `EVALUATION_ACTIVE` | `EVALUATION_PASSED` | `SingleTrader._handle_eval_pass` via `_exit_for_account` | `account.balance >= account_size + target` |
| `EVALUATION_PASSED` | `CLOSED` | `SingleTrader._handle_eval_pass` | Lifecycle close after pass |
| `EVALUATION_FAILED` | no retained active object | `SingleTrader._handle_eval_failure` | `self.eval = None`, then replacement purchase |

The pass and failure handlers are not unreachable. They are conditional branches of `_exit_for_account`; the representative replay reached the enclosing method 46 times but met neither condition.

## Representative thresholds

For the 25K Zero account:

- Starting balance: `$25,000`
- Pass threshold: `$26,500` (`$1,500` target)
- Initial MLL: `$24,000` (`$1,000` below the starting balance)
- Lowest observed completed-trade balance in the trace: `$24,981.4031`
- Highest observed completed-trade balance in the trace: `$25,015.1290`
- Final evaluation trading PnL: `-$2.8132`

Therefore the representative account never passed because it was about `$1,485`–`$1,519` short of the pass threshold, and never failed because it stayed about `$981`–`$1,008` above the MLL. It remained `EVALUATION_ACTIVE` through year-end, so monthly rebills were expected.

## Important boundary

`_exit_for_account` performs the threshold checks after an exit leg. `_flatten()` itself only realizes the forced exit and removes positions; it does not independently invoke pass/failure checks. Thus a threshold crossed solely by a session-forced or end-of-period flatten would not be checked until another applicable exit event. That is a lifecycle-boundary limitation, but it did not cause the representative zero-pass/zero-failure result.

