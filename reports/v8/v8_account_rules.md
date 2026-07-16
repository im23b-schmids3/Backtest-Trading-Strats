# Alpha Futures Zero 25K rules used by V8

Verified against official Alpha Futures Help Center documentation on 2026-07-14. The simulation uses the current published rules below.

## Account parameters

| Rule | Implemented value | Difference from request assumptions |
|---|---:|---|
| Simulated account size | $25,000 | None |
| Evaluation profit target | $1,500 | None |
| Maximum Loss Limit | $1,000, end-of-day trailing; stops at $25,000 | The request did not specify trailing calculation |
| Daily Loss Guard | $500, soft lock; 2% of starting balance | None |
| Evaluation max position | 1 mini or 10 micros | None |
| Evaluation consistency | None on Zero Evaluation | The request asked to verify; it is not required |
| Qualified consistency | 40% since the last withdrawal request | The request did not specify the qualified value |
| Profit split | Trader receives 90% of the gross withdrawal request | None |
| Qualified payout timing | Up to 4 requests per month after 5 winning days of at least $200 | Not a fixed weekly payout schedule |
| Withdrawal amount | Minimum $200; maximum $1,000; request removes up to 50% of account profit | The request said weekly payouts but not these limits |
| Evaluation subscription | $79/month; ends after passing | Not included as a trading PnL deduction |
| Evaluation reset | $69; monthly rebill also resets a failed account | Not included in payout statistics |
| Qualified reset | Zero Qualified reset available twice before any payout, within 7 days of breach, for eligible post-2026-03-11 purchases | Not automatically simulated |

## Implemented behavior

- Evaluation passes when balance reaches the $1,500 target without an MLL breach.
- MLL is based on the highest end-of-day balance minus $1,000, capped at the $25,000 starting balance. A balance/equity breach terminates the account.
- Daily Loss Guard is treated as a soft breach: open trading is locked until the next 6PM ET trading day. The simulation records the violation; it does not count it as account failure unless MLL is also breached.
- Qualified payouts require five accumulated $200+ winning days and the 40% consistency test. A payout request is modeled at 50% of cycle profit, capped at $1,000, with 90% paid to the trader.
- Consistency failures block a payout cycle; they do not terminate the account.
- News restrictions are not modeled because the historical data has no official high-impact-news calendar. Officially, evaluations have no news restriction; Qualified Zero accounts prohibit execution within two minutes before or after high-impact news.
- Prohibited-practice review is not modeled from OHLC data. Alpha prohibits AI, bots and fully automated trading; the results therefore represent a manual/semi-automated execution assumption only, not authorization to run the repository as a trading bot.
- The simulation uses the repository's conservative OHLC execution and existing fees/slippage. It uses CME price specifications to convert price moves into contract dollars, rounds fills to the official tick, and retains repository fee rates because Alpha's account rules do not publish a universal commission schedule.
- Historical ETH/SOL bars are exchange-price proxies, not CME futures bars. Contract rolls, CME session holidays, margin, commissions, news windows and exact intrabar unrealized PnL are not available in the repository and are limitations.

## Official sources

- [Zero Account Overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview)
- [Maximum Loss Limit](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll)
- [Daily Loss Guard](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard)
- [Consistency Rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule)
- [Payout Policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy)
- [Maximum Withdrawal Request](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request)
- [Reset](https://help.alpha-futures.com/en/articles/9492077-reset)
- [Monthly Subscription](https://help.alpha-futures.com/en/articles/9492068-monthly-subscription)
- [Scaling Plan](https://help.alpha-futures.com/en/articles/9492025-scaling-plan)
- [News Trading Policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy)
- [Prohibited Trading Practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices)

## CME contract specifications

| Instrument | Multiplier | Tick size | Tick value | Point value |
|---|---:|---:|---:|---:|
| Micro Ether (MET) | 0.10 ETH | $0.50/ETH | $0.05 | $0.10 per $1 |
| Ether (ETH) | 50 ETH | $0.25/ETH | $12.50 | $50 per $1 |
| Micro Solana (MSL) | 25 SOL | $0.05/SOL | $1.25 | $25 per $1 |
| Solana (SOL) | 500 SOL | $0.05/SOL | $25.00 | $500 per $1 |

Sources: [CME Micro Ether specifications](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html), [CME Ether specifications](https://www.cmegroup.com/education/courses/introduction-to-ether/ether-futures-product-overview), [CME Micro SOL specifications](https://www.cmegroup.com/rulebook/CME/IV/400/440/440.pdf), and [CME SOL specifications](https://www.cmegroup.com/articles/2025/the-essential-guide-to-solana-futures.html).

The official Alpha 25K limit permits one mini or ten micros. V8 evaluates 2, 3, 5, 7 and 10 micros, plus one mini for comparison, without exceeding that limit.
