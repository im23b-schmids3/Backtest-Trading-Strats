# Portfolio Aggregation Audit

## Curve report

`_portfolio_metrics` builds one equity series per market, normalizes each by its own first equity, forward-fills it on the union timestamp index, fills pre-listing values with `1.0`, and computes:

`combined_curve[t] = mean(normalized_market_curve_1[t], ..., normalized_market_curve_N[t])`

This is an independent-market/equal-weight curve aggregation (model B), not one account with shared cash, shared margin, or real-time position capacity. For Portfolio D 4H this produced a 569.68% return and `{"BTC": 0.2018819051710516, "ETH": 0.6131947749133305, "Gold": 0.012625205545343394, "QQQ": 0.0, "S&P proxy": 0.06512951421876391, "SPY": 0.0, "Silver": 0.021025648159571832}` contributions.

## Prop account report

`_portfolio_trades` concatenates each market’s prepared trades and sorts them by entry time. `_simulate_account` then applies one account balance, one MLL, one daily guard, and conflict rules. A same-market active trade or a total-contract limit skips the new trade. This is closer to model A, but it is not the same as the curve report because it uses fixed micros and proxy contract specs.

## Additional inconsistency

The curve report’s `_conflict_count` is called with hard-coded `max_contracts=10` and `contracts=2`, and returns `(conflicts, conflicts)`. Its conflict fields therefore do not describe the actual account-size/position-size path. In the account lifecycle, conflicts are counted using the account’s actual micros and max-micros rule.

## Conclusion

Portfolio D is simulated two different ways in the same V12 research: independent normalized market curves in `v12_portfolios.csv`, and a single aggregated account event stream in `v12_account_lifetimes.csv`. Calling both “Portfolio D” without a common capital-allocation definition is a reporting/model-consistency issue.
