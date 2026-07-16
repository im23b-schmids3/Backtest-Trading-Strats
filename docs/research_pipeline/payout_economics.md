# Payout and external economics

Payout eligibility is evaluated from qualified-ledger winning days, minimum
cycle profit, payout minimum/maximum, split, and consistency rules. Subscription
costs, reset/activation costs, and trader payouts are kept outside trading PnL.
The final review separately reports gross futures PnL, fees/slippage, net
trading PnL, external cash flow, cost per pass, cost per payout, payout ratio,
and ROI on external costs.

The official Zero rule fixture was verified on 2026-07-16 from:

- [Zero account overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview)
- [Payout policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy)
- [Daily loss guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard)

Rules are time-sensitive. `verify_rules` rejects stale, un-hashed, unresolved,
or unverified rule records.
