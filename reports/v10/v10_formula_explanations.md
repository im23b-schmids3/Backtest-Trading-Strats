# V10 formula explanations

## V9 audit

- Pass rate = `count(passed == true) / count(all account rows)`.
- First/second/third payout probability = `count(payout_count >= n) / count(all account rows)`.
- V9 expected yearly payout = `mean(row.payouts_received / max((end_date - start_date).days / 365.25, 1))`.
- V9 monthly payout = expected yearly payout divided by 12.

The V9 formulas are mathematically valid for the statistics they name. They are not net trader-income formulas. The V9 aggregate mixes all six sizes, seven policies, four streams, and monthly starts. It includes zero-payout failures, annualizes each account row over its own historical exposure, excludes subscription/reset costs, and does not purchase replacement accounts. A payout probability answers whether an account ever reaches a payout; it does not state the amount or timing of withdrawals.

## V10 lifecycle economics

- Subscription months = `max(1, ceil(evaluation_days / 30.4375))`; no subscription is charged after pass, failure, or voluntary closure.
- Net lifetime profit = `received payouts - subscription cost - challenge fees - reset fees - commissions`.
- Yearly income = `net lifetime profit / max(lifetime_days / 365.25, 1 / 365.25)`.
- ROI = `net lifetime profit / total lifetime costs`.
- Scenario B/C/D retain only the first/second/third received payouts and close at that payout timestamp.
- Scenario E closes when realized withdrawals do not exceed the subscription cost required to keep the account to its natural end; this is an explicit positive-value heuristic, not a new trading rule.
- Continuous trader results bootstrap complete V9 lifecycle rows until a one-year horizon. A failed account is immediately replaced; a non-failed historical account ends the simulated path.
- Multi-account results sum independent bootstrap paths for 1, 3, and 5 accounts.
- Break-even results show cumulative received payouts minus evaluation subscription costs by month.

Commissions are reported as zero incremental cost because frozen V9 net trade values already include the repository fee model; V10 does not double-count them. Optional reset fees are separately available in the verified-rules document but are not assumed in the base replacement model.
