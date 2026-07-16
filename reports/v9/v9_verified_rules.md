# Alpha Futures Zero 25K rules verified for V9

Verified against official Alpha Futures and CME documentation on 2026-07-14.

## Rules used

| Rule | V9 value | Source / implementation note |
|---|---:|---|
| Account size | $25,000 | Official Zero overview |
| Evaluation target | $1,500 | Official Zero overview |
| Maximum Loss Limit | $1,000, end-of-day trailing and capped at initial balance | [Alpha MLL](https://help.alpha-futures.com/en/articles/9491999-maximum-loss-limit-mll) |
| Daily Loss Guard | $500, soft lock, 2% of starting balance | [Alpha DLG](https://help.alpha-futures.com/en/articles/9492014-daily-loss-guard) |
| Maximum size | 1 mini or 10 micros | [Zero overview](https://help.alpha-futures.com/en/articles/11771813-zero-account-overview) |
| Evaluation consistency | None | Zero Evaluation has no consistency rule |
| Qualified consistency | 40% since last withdrawal | [Consistency rule](https://help.alpha-futures.com/en/articles/9492048-consistency-rule) |
| Winning days | 5 non-consecutive days with at least $200 | [Payout policy](https://help.alpha-futures.com/en/articles/9492051-payout-policy) |
| Withdrawal | Up to 50% of profit, $200–$1,000 for Zero 25K | [Maximum withdrawal](https://help.alpha-futures.com/en/articles/10491202-maximum-withdrawal-request) |
| Profit split | Trader receives 90% of request | Official payout policy |
| Schedule | Up to 4 requests per month; no fixed weekly day | Official Zero overview and payout policy |
| Evaluation fee | $79/month; reset $69 | Subscription/reset rules are documented but not deducted from trading PnL |
| Qualified reset | Available only for eligible accounts under published conditions; not automatic | Not modeled as an optimization or recovery action |
| News | No evaluation restriction; Qualified Zero has a 2-minute before/after restriction | [News policy](https://help.alpha-futures.com/en/articles/9492063-news-trading-policy); no news calendar is available locally |
| Automation | Alpha prohibits AI, bots, and fully automated trading | [Prohibited practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices) |

## Session policy

V9 treats the requested local session policy as configurable: no new entries at or after **22:20 Europe/Berlin**, and all remaining positions are liquidated at **22:30 Europe/Berlin**. With 4H and daily source bars, the first available bar at or after the liquidation time is used as the causal price proxy; this is not an exact CME intrabar fill.

CME Ether and SOL futures publish Globex hours of Sunday–Friday 5:00 p.m.–4:00 p.m. Central Time with a daily break. See [Micro Ether hours](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html) and [SOL futures hours](https://www.cmegroup.com/articles/2025/the-essential-guide-to-solana-futures.html).

## Important limitations

- To keep the seven-policy/two-session comparison computationally tractable, V9 uses one feasible account start per calendar month. Each selected run replays every available bar from that start to the end; this is not an every-calendar-day start-date estimate.
- Historical inputs are ETH/SOL exchange-price proxies, not continuous CME futures contracts.
- The repository has no CME holiday/session calendar, news calendar, or live commission schedule.
- OHLC data cannot reveal exact intrabar unrealized equity. MLL and DLG checks therefore occur at observed execution/forced-exit points; DLG flattening is modeled conservatively at the triggering observed price.
- Subscription, reset, and account-activation economics are reported as limitations, not included in trading PnL.
- V9 changes only trade eligibility and size. Entries, exits, stops, TP allocation, swing logic, fees, slippage, and execution generation remain the frozen V8 logic.
