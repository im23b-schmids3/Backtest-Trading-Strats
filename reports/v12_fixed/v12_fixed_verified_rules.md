# V12 Fixed Alpha Futures Zero Rules

Verification date: **2026-07-15** (Europe/Zurich). Sources were rechecked against the official Alpha Futures Help Center only.

## Verified official rules

| Rule | 25K Zero | 50K Zero | Source |
|---|---:|---:|---|
| Evaluation subscription | $79/month | $119/month | [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription) |
| Profit target | $1,500 | $3,000 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Maximum Loss Limit | $1,000 | $2,000 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [MLL](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll) |
| Daily Loss Guard | $500 | $1,000 | [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard) |
| Evaluation max position | 10 micros | 30 micros | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Evaluation consistency | None | None | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Qualified consistency | 40% | 40% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy), [Consistency Rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule) |
| Winning days | 5 days of >= $200 | 5 days of >= $200 | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Maximum withdrawal request | $1,000 | $1,500 | [Maximum Withdrawal](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request) |
| Minimum withdrawal request | $200 | $200 | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Maximum profit withdrawn per request | 50% | 50% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Trader split | 90% | 90% | [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Qualified monthly subscription | $0 | $0 | [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription) |
| Zero activation fee | $0 | $0 | [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview), [Activation Fee](https://help.alpha-futures.com/en/articles/9492083-activation-fee) |
| Qualified reset | $399, explicit only | $499, explicit only | [Reset](https://help.alpha-futures.com/en/articles/9492077-reset) |
| Trading session | 6PM ET to 5PM ET; close positions before 4:20PM ET | 6PM ET to 5PM ET; close positions before 4:20PM ET | [What and When You Can Trade](https://help.alpha-futures.com/en/articles/9492096-what-and-when-you-can-trade) |

## Lifecycle and billing interpretation

The official activation page says the Evaluation account is closed after reaching the profit target: [Activating your Qualified Account](https://help.alpha-futures.com/en/articles/9820801-activating-your-qualified-account). The fixed simulator therefore records `EVALUATION_PASSED`, cancels remaining Evaluation exposure, closes that ledger, and starts a separate Qualified ledger.

Alpha's current trading-hours page confirms both MBT and MET are available. CME contract metadata used for the proxy conversion is documented by [CME Micro Bitcoin](https://www.cmegroup.com/trading/files/micro-bitcoin-futures-fact-card-retail-us.pdf) and [CME Micro Ether](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html). BTC maps to MBT; ETH maps to MET; SPY is excluded because SPX already occupies MES.

The subscription page says billing continues after an Evaluation breach until the trader cancels or the next rebill resets the failed Evaluation. The replay therefore reports three explicit scenarios:

- `A_CANCEL_ON_BREACH`: cancel at breach; no future subscription or reset charge.
- `B_REBILL_AFTER_BREACH`: keep billing until the next calendar rebill and start a fresh Evaluation with no separate reset charge.
- `C_EXPLICIT_EVALUATION_RESET`: cancel at breach and charge the documented Evaluation Reset fee for an explicitly purchased reset.

Qualified monthly subscription and Zero activation fees are always zero. Qualified resets are not automatic; they are reported as zero in the primary replay and only represent a cost in an explicitly purchased scenario.

## Important rule difference from the old simulator

The old V12 model applied the same account ledger after pass, charged economics from payout-only fields, and labeled passed histories censored. The fixed simulator separates Evaluation and Qualified stages, records state transitions, separates trading PnL from withdrawals, and excludes censored stages from uncensored lifetime medians.

## Known data limitations

The frozen strategy produces Binance proxy fills rather than native CME futures. The retained trade objects do not contain a complete account-level intrabar mark stream, so floating DLG/MLL liquidation is represented conservatively from available trade-leg prices and last prices. News restrictions require a historical high-impact calendar, which was not retained; the relevant Qualified rule is documented in [News Trading Policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy).
