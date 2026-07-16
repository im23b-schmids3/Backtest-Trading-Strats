# Risk sizing

Phase D models fixed contracts, fixed initial dollar risk, MLL-percentage risk,
volatility-capped risk, and account-buffer-aware risk. Each policy is a typed
`RiskPolicy`. Contract quantities are rounded down and then capped by both the
policy override and provider contract limit.

`SharedExposure` tracks open contracts per account and releases them at the
synthetic settlement timestamp. A native adapter must retain this protection
across overlapping signals. Zero legal contracts are recorded as a skip rather
than rounded up to one contract.

Sizing is policy analysis only. It is not an order router and does not place
orders or change existing backtest behavior.
