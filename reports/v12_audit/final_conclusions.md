# Final Conclusions

1. **Is the Alpha simulator mathematically correct?** No, not as a complete Alpha lifecycle simulator. The realized-leg arithmetic is internally traceable, but pass termination is missing, open-equity behavior is limited, and the portfolio/economics layers use different definitions.

2. **Is there any implementation bug?** Yes. At the pass condition the code sets `passed=True` and records `pass_time`, but never breaks or transitions to a terminal funded-account object. The final status is then `CENSORED` unless the account later fails. The curve conflict helper also uses hard-coded conflict parameters and returns the same value twice.

3. **Is there any reporting bug?** Yes. `average_gross_revenue` is actually average gross payout request, not gross trading profit. The economics output omits account balance change and per-trade gross/fee reconciliation. Portfolio D’s curve and prop-account outputs are not the same simulation.

4. **Why does ETH show high returns while some account configurations show approximately $0 gross revenue?** ETH’s high return is on the separate frozen-strategy market equity path. Prop “gross revenue” is only a payout request. If the account never satisfies pass/winning-day/consistency/payout rules, it is zero even when strategy equity is positive. Fixed proxy micros and conflict skips further reduce the account path.

5. **Why are subscription costs so high?** Each monthly lifecycle charges `ceil(days_to_subscription_stop / 30.4375)` months, with a minimum of one. Long censored histories therefore accumulate many months: for example, a 25K account at $79/month over roughly nine years costs about $8.3K. Failed trader paths can add a new subscription after each reset/account purchase. Charges do not continue after the individual lifecycle’s pass or failure timestamp, but pass does not stop trading.

6. **Are account lifetimes calculated correctly?** Failure lifetimes stop at the failure timestamp. Historical-end lifetimes stop at the last historical timestamp. However, passed accounts are not terminated and are labeled censored, so pass-ending and post-pass lifetime statistics are not correct. Retained counts are: 4,632 history-end-before-pass, 2,818 history-end-after-pass, 2,031 failure-before-pass, and 1,119 failure-after-pass.

7. **Are subscriptions cancelled correctly?** The subscription-cost formula stops at pass, failure, or history end; it does not add charges after those timestamps within one lifecycle. But the account state machine is still wrong after pass, and realistic trader restarts intentionally create new subscriptions after failures. The audit cannot verify provider billing semantics beyond the code’s formula.

8. **Is Portfolio D simulated correctly?** Not consistently. Its normalized return is model B independent equal-weight aggregation; its prop account is a single event stream with shared account constraints. Both can be useful views, but they must not be treated as one identical Portfolio D result.

9. **Would you trust the current Alpha simulator for future research?** No. I would trust it only for a limited diagnostic of realized-leg proxy arithmetic. I would not use its pass rates, lifetime distributions, payouts, economics, or Portfolio D comparisons for further decisions.

10. **What must be corrected before additional research?** Without changing it in this audit: (a) retain one row per account trade/leg with timestamp, market, entry, exit, PnL, balance, daily drawdown, MLL, and target distances; (b) make pass a real lifecycle transition/termination according to the chosen Alpha interpretation; (c) distinguish evaluation, funded, failed, paid-out, and censored statuses; (d) report trading gross PnL separately from gross payout and include ending-balance reconciliation; (e) define one portfolio capital-allocation model and use it consistently; (f) replace hard-coded conflict diagnostics; and (g) make open-equity/guard/MLL treatment explicit and causal.

The requested complete trade-by-trade representative chronology is unavailable from retained V12 data because V12 deliberately generated no trade or order logs. That is a data-retention limitation, not evidence that the missing trades did not occur.
