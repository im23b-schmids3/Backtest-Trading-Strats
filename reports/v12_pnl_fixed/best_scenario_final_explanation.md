# Best corrected scenario: final compact reconciliation

Scenario: `50K Zero | All canonical Alpha exposures | 4h | 10 micros | ONE_ACCOUNT_ONLY_BASELINE`, from `2025-01-01` through `2026-01-01`.

## Account-level reconciliation

```text
Candidate entries                         90
Filled entries                           75
Fully closed positions                    75
Positions open at data end                 0

Gross admissible exit-leg PnL     $317,570.50000
Entry fees                          -$11,541.41125
Exit fees                           -$11,488.21750
Total fees                          -$23,029.62875
------------------------------------------------
Net trading PnL                    $294,540.87125
```

The gross figure is the exact sum of account-admissible exit legs captured while an Evaluation or Qualified account was active. It excludes 15 rejected/cancelled candidate entries and contains no post-termination PnL.

The lifecycle totals are:

- 3 Evaluations passed.
- 0 Evaluations failed.
- 3 Qualified accounts created.
- 2 Qualified accounts failed.
- 65 session-forced liquidation legs.
- 3 Evaluation-pass flatten closures.
- 0 account-limit violations.
- Maximum simultaneous exposure: 10 micros total across the shared account.

The PnL ledger reconciles exactly:

```text
Evaluation net trading PnL       $55,725.65800
Qualified net trading PnL       $238,815.21325
-----------------------------------------------
Net trading PnL                 $294,540.87125
Reported net trading PnL        $294,540.87125
Difference                                $0.00
```

## Market attribution

The market CSV sums to the account totals:

- BTC/MBT: `$719.405` net.
- ETH/MET: `-$32.260` net.
- Gold/MGC: `$5,132.445` net.
- S&P proxy/MES: `$288,721.28125` net.
- QQQ/MNQ: no cached 4H trades.
- Silver/SIL: no accepted trades in this scenario.

The S&P proxy contributes most of the result. Its largest winning position is `$37,490.6875` net and its largest losing position is `-$36,223.33125` net. The synthetic mapped MES entry range was `$2,710` to `$17,288.50`, with a maximum absolute mapped leg movement of 763 points. This is mathematically compatible with the $5 MES multiplier, but it is highly dependent on the synthetic proxy mapping and is not evidence of native MES performance.

## Validation of the requested invariants

1. Every counted position was accepted by `_accept()` while its account state was active. The account-level audit recorded 75 filled positions and no post-termination fills.

2. No PnL was counted after Evaluation pass, Evaluation failure, or Qualified failure. The audit found `$0.00` excluded-after-termination PnL.

3. The maximum-contract check sums `position["remaining"]` across the account’s complete active-position dictionary. It is not applied independently by market.

4. “10 micros” means 10 total micros across the shared account. The observed maximum simultaneous exposure was exactly 10 and there were zero limit violations.

5. All 75 positions reconciled their exit-leg quantities to the original 10-contract exposure. Partial exits reduce remaining quantity; they do not recreate 10 contracts at each target.

6. MES synthetic mapping uses a causal return-based reference. It does not directly multiply normalized proxy values near 1.0 by the MES multiplier. The resulting point changes are large in some cases, so native futures validation remains necessary.

7. The largest winners are compatible with the mapped multipliers: for example, MES uses `$5` per index point and MGC uses `$10` per gold price point. Their size is explained by the observed proxy return and 10-contract exposure, not by duplicate contract application.

8. Gross PnL equals the exact sum of admissible account exit legs: `$317,570.50`.

9. Evaluation plus Qualified net PnL equals reported net trading PnL exactly; there is no unexplained accounting difference.

10. The approximately `$295,000` net trading PnL produces only `$5,400` in trader payouts because trading PnL remains inside the Evaluation/Qualified account ledgers. It is not automatically external cashflow. Only one payout was completed: `$6,000` gross requested × 90% trader split = `$5,400`. Subscription costs were `$357`, so:

```text
Trader payouts                 $5,400
- subscriptions                  $357
------------------------------------
Net external cashflow          $5,043
```

Two Qualified accounts failed before producing a first payout. The payout rules, account lifecycle, consistency requirements, and payout limits—not missing trading PnL—limited external withdrawals.

## Trust assessment

- **$5,043 external cashflow:** trustworthy as an exact output of this deterministic proxy-based account model and its lifecycle rules.
- **$294,540.87 net trading PnL:** trustworthy as an exact reconciliation of this run’s admissible synthetic-proxy legs and fees.
- **Inflation:** no evidence of inflation from contract limits, partial exits, or lifecycle accounting. The shared 10-micro limit and quantities reconcile correctly.
- **Remaining material limitation:** the S&P/MES result is synthetic return-mapped Binance-proxy PnL. Its scale is mathematically consistent but not validated against native CME MES data. Therefore neither figure should be treated as executable live-futures performance.
