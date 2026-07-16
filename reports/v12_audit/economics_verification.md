# Economics Verification

## Equations implemented by V12

For a prepared trade leg, with direction `d` (+1 long, -1 short), entry `E`, exit `X`, proxy multiplier `M`, and leg quantity `q`:

`gross_leg = d × (X - E) × M × q`

`exit_fee = abs(X × M × q) × fee_rate`

`entry_fee = abs(E × M × total_quantity) × fee_rate`

`net_trade = sum(gross_leg - exit_fee) - entry_fee`

The account starts at `B0 = account_size`. Each accepted entry subtracts `entry_fee`; each accepted exit leg adds its `leg.net`. Thus the retained balance change is the realized account path, subject to conflict skips and the realized-leg-only risk checks.

Pass condition:

`balance >= account_size + target`

Payout request after pass:

`request = min(0.50 × cycle_profit, payout_max)`

subject to five winning days of at least $200, the consistency test, request >= $200, and `balance - request > mll`.

`gross_payout = sum(request)`

`net_payout = 0.90 × gross_payout`

`subscription_months = max(1, ceil((subscription_stop - start_days)/(365.25/12)))`

where `subscription_stop = pass_time or failure_time or historical_end`.

`cash_economic_result = net_payout - subscription_cost - reset_cost`

`ROI = cash_economic_result / (subscription_cost + reset_cost)`

## Why ETH can be profitable while gross revenue is $0

The V12 economics function assigns `gross = group.gross_payout`; it does **not** assign `gross = trading gross PnL`. A row with no qualifying payout has `gross_payout = 0`, so `average_gross_revenue = 0` by definition. Trading gains remain in the simulated account balance and are not cash revenue until a payout is requested. This is expected from the function’s definition, but the column name is misleading and the report does not expose gross trading PnL per account.

There is also a sizing/path separation: the market result is a $10,000 strategy equity path with the frozen strategy’s own sizing. The prop layer reconstructs dollar PnL from fixed `size` micros and proxy contract multipliers. It then skips overlapping same-market/capacity-conflicting trades. Therefore the market return cannot be algebraically expected to equal prop-account payout revenue.

## Representative cash reconciliation

The representative account is 25K Zero / Portfolio D / 4H / 5 micros, first retained failed lifecycle. Its retained values are:

- ending balance: $24,149.62
- balance change from $25,000 starting balance: $-850.38
- gross payout: $0.00
- net payout: $0.00
- subscriptions: $474.00
- reset: $69.00

Thus the implementation’s trader cashflow is:

`$0.00 - $474.00 - $69.00 = $-543.00`

The account balance path separately moved by `$-850.38`. V12 economics does not include that balance change, nor does it expose per-trade gross trading PnL, so it cannot provide the requested gross trading profit reconciliation from retained outputs alone.

## Assessment

The payout/cost arithmetic is reproducible. The economic reporting is incomplete/misnamed if “gross revenue” is intended to mean trading profit. The lifecycle is not correct for a rule that requires termination at evaluation pass.
