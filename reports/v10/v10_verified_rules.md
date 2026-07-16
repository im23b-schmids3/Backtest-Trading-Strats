# Alpha Futures Zero rules verified for V10

Verified against official Alpha Futures documentation on 2026-07-14.

| Account | Target | MLL | DLG | Subscription | Eval reset | Qualified reset | Max position | Payout max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero 25K | $1,500 | $1,000 | $500 | $79/month | $69 | $399 | 1 mini / 10 micros | $1,000 |
| Zero 50K | $3,000 | $2,000 | $1,000 | $119/month | $109 | $499 | 3 minis / 30 micros | $1,500 |
| Zero 100K | $6,000 | $3,000 | $2,000 | $239/month | $219 | $799 | 6 minis / 60 micros | $2,500 |

All Zero accounts have $0 activation fee, 90% profit split, no evaluation consistency rule, 40% qualified consistency, and payout eligibility after five non-consecutive $200+ winning days. Withdrawals are up to 50% of profit, with up to four requests per month. Qualified accounts do not pay the evaluation subscription. [Zero overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [payout policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy), [subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription), [reset](https://help.alpha-futures.com/en/articles/9492077-reset)

Qualified resets are available only for eligible Zero accounts, are limited to two before the first payout, and are not automatically assumed by V10. V10 treats failed-account replacement as a new evaluation purchase, not as an automatic paid reset.

The V10 lifecycle model charges subscriptions only through evaluation end, stops them at failure/pass/voluntary closure, and does not double-count commissions already embedded in frozen V9 trade PnL. Challenge fees and reset fees are zero in the base lifecycle because the official Zero product uses monthly evaluation pricing and V10 does not assume optional resets. A continuous replacement analysis is also provided.

Alpha permits five Qualified Zero accounts per user, subject to the official allocation rules. [Maximum allocation](https://help.alpha-futures.com/en/articles/9492088-maximum-allocation)

Important: Alpha prohibits AI, bots, and fully automated trading. This is an economics audit, not authorization for automated execution. [Prohibited practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)
