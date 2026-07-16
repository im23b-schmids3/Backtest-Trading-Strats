# V12 Alpha Simulator Audit

## Scope

This is a static audit of the retained V12 source and outputs. No strategy, simulator, data download, or backtest was rerun. Inputs were `src/fib_backtester/research/v12_binance_proxy_prop_simulation.py` and the retained V12 CSV reports.

## Executive result

The simulator contains internally coherent arithmetic for the limited realized-leg account model, but it is **not mathematically/logically correct as a complete Alpha account lifecycle simulator**. The audit found one major lifecycle bug, one major interpretation/reporting problem, and one aggregation inconsistency:

1. `passed=True` does not terminate an account. The loop continues trading after the pass timestamp and returns `status=CENSORED` unless the account later fails.
2. `average_gross_revenue` is the average **gross payout request**, not gross trading profit. Therefore it can be `$0` while the separate strategy report is strongly profitable.
3. `v12_portfolios.csv` uses independent normalized equity curves, while prop-account Portfolio D uses one account event stream with cross-market conflicts. They are not the same portfolio simulation.

## Retained-output counts

| Quantity | Count/value |
|---|---:|
| Lifecycle rows | 10,600 |
| Passed flag | 3,937 |
| Failed | 3,150 |
| Censored flag | 7,450 |
| At least one payout | 2,184 |
| History end before pass | 4,632 |
| History end after pass (pass did not terminate) | 2,818 |
| Failure before pass | 2,031 |
| Failure after pass | 1,119 |
| Total retained subscriptions | $41,074,570.00 |
| Total retained resets | $291,350.00 |
| Mean lifetime, all rows | 1,408.9 days |
| Mean lifetime, 25K | 1,426.1 days |
| Mean lifetime, 50K | 1,391.8 days |
| Mean lifetime, non-censored rows only | 768.5 days |

The non-censored average is failure-biased because passed accounts are incorrectly labeled censored. A true "ended by pass" count is zero in the retained status field because pass never terminates the loop.

## Representative evidence

ETH 4H has 226 strategy trades and 42,923.63 strategy net PnL on the separate $10,000 strategy equity path. The first retained 25K / Portfolio A / ETH-only / 4H / 5-micros lifecycle has no payout, $8,374.00 of subscriptions, and a small account balance change. That is expected under the current code because strategy returns are not copied into prop revenue; only qualifying payout requests are reported as revenue.

Portfolio D 4H reports 499 trades and 569.68% return in the normalized curve report. The prop account path is a different simulation with fixed micros, contract mappings, and position-conflict skips.

See `economics_verification.md`, `portfolio_aggregation.md`, and `final_conclusions.md` for the equations and conclusions. The complete trade-by-trade lifecycle requested cannot be reconstructed because V12 deliberately did not retain trade/order logs; `account_lifecycle_example.csv` and `account_timeline.csv` mark that retention gap explicitly.
