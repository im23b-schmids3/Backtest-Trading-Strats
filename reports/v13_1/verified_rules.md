# Alpha Futures Zero 25K rules verified 2026-07-15

The V13.1 model uses the current published Zero rules and the existing
corrected V13 lifecycle implementation:

- 25K Zero Evaluation: **$79/month**.
- Profit target: **$1,500**.
- Maximum Loss Limit: **$1,000**. The official MLL page states that a breach
  can occur from floating equity or closed balance and liquidates the account.
  The existing V13 engine therefore checks marked equity causally and keeps
  its verified end-of-day trailing MLL update.
- Daily Loss Guard: **$500** for 25K Zero. It is a soft breach: unrealized,
  realized PnL, fees and commissions are included; positions are flattened and
  the account is locked until the next trading day.
- Evaluation position limit: **1 mini or 10 micros**.
- Monthly billing: the subscription rebills on the signup day each month until
  the Evaluation passes or the trader cancels. After a failed Evaluation, the
  official documentation says rebilling continues unless cancelled; after
  rebill the failed account is reset. V13.1 explicitly models the requested
  trader action of manually cancelling immediately after a simulated failure,
  so no future rebill is charged for that account.
- Qualified payout economics: up to 50% of profit per request after five
  winning days of at least $200; the trader receives 90% of the requested
  withdrawal. The existing V13 payout engine is unchanged.
- Compliance: Alpha prohibits all-or-nothing trading, maximum-leverage
  account rolling, account stacking, and gambling-like repeated account
  failures. Policies D and E are therefore labeled research-only and receive
  explicit confidence/compliance warnings.

Sources:

- [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview)
- [Maximum Loss Limit](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll)
- [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard)
- [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription)
- [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy)
- [Prohibited Trading Practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)

The official documentation is time-sensitive. This verification is a research
record, not legal or commercial advice; live account terms should be checked
again before trading.
