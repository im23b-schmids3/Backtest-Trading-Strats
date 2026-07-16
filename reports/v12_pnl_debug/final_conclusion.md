# V12 PnL reconciliation conclusion

## Root cause of the approximately -$2.81 result

For the representative `25K Zero | ETH only | 1h | 2 micros | MIRRORED` account, the result is fully reconciled:

```text
exit gross PnL       $24.0000
- exit-leg fees       $13.4106
- entry fees          $13.4026
--------------------------------
net account PnL       -$2.8132
```

The simulator did not lose a large futures PnL amount. This particular account traded only 2 MET contracts, had 23 accepted entries, and its small positive/negative ETH price moves produced only $24 gross after the account's conflicts and session liquidations. Fees consumed more than the gross result.

## Conversion finding

The deterministic V12 path in `src/fib_backtester/research/v12_fixed_alpha_lifecycle.py` uses:

```python
gross = direction * (price - entry) * spec.multiplier * quantity
```

For BTC, ETH, and PAXG/Gold this is a direct price-unit conversion whose multipliers are correct for MBT, MET, and MGC. It does not use a normalized return or apply tick value a second time. Position count is applied in `quantity`.

However, the conversion is not generally valid for every Binance proxy. The retained SPX proxy is normalized to prices around `1.0`, while MES is quoted in S&P 500 index points around thousands. Applying the normalized SPX move directly to the MES multiplier produces zero or near-zero PnL after the 0.25-point tick rounding. This is a proxy price-scale/data-layer defect, not a futures tick-value calculation defect.

The legacy `v12_binance_proxy_prop_simulation.PROXY_SPECS` also contains an obsolete BTC-to-MET mapping; the deterministic fixed lifecycle uses the corrected `CANONICAL_PROXIES` mapping BTC-to-MBT. The SIL tick in the fixed canonical map is `0.005`, while the current CME metals specification reports `0.01`; that is a separate Silver specification defect.

## Answers to the requested questions

1. The representative produced `-$2.8132` because its booked gross was only `$24.00` and total entry/exit fees were `$26.8132`.
2. The conversion is correct for price-level-aligned BTC/MBT, ETH/MET, and PAXG/MGC in this harness, but not for normalized index proxies in general.
3. MBT, MET, MGC, MES, and MNQ tick sizes/multipliers match the verified specifications used by the fixed simulator. SIL's retained tick is stale/incorrect at `0.005` versus `0.01`.
4. Micros are applied correctly. A single identical ETH trade scaled 2→5→7→10 micros at exactly 1.0×, 2.5×, 3.5×, and 5.0× gross PnL before fee effects.
5. Partial exits are scaled correctly: each leg uses price movement × multiplier × remaining quantity. The sampled legs sum exactly to the original position quantity.
6. Forced liquidations are booked correctly in the representative replay: 15 forced legs, 24 contracts, `$30.65` gross and `$23.14185` after their fees.
7. No percentage/decimal error or double tick-value application was found in the fixed deterministic PnL formula. The major scale error is the normalized SPX proxy being treated as MES index points.
8. Correct annual PnL for the representative scenario, under the current retained proxy path and execution assumptions, is `-$2.8132`; the report reconciles exactly. A native-futures-equivalent answer cannot be inferred for normalized proxies without a price-scale conversion or native futures data.
9. This is not an account PnL booking bug for the representative ETH scenario. It is a data/proxy conversion problem for normalized index markets, plus the stale SIL tick metadata and an obsolete legacy BTC mapping.
10. The smallest correction is to keep native/proxy price scales explicit and convert normalized proxy returns into mapped-futures index-price changes before applying multiplier/ticks; separately update SIL tick metadata and remove/avoid the obsolete legacy BTC mapping. No correction is implemented here.

## Trade-count definitions

The representative has 29 candidate signals, 23 accepted/filled entries, 23 positions fully closed by the end of the replay, 8 with normal exit completion and 15 requiring forced session liquidation, zero positions still open at data end, and 6 cancelled/skipped entries. `total_trades` should not be interpreted as 23 completed trades.

Official references: [CME Micro Bitcoin specifications](https://www.cmegroup.com/trading/files/micro-bitcoin-futures-fact-card-retail-us.pdf), [CME Micro Ether FAQ](https://www.cmegroup.com/articles/2021/micro-ether-futures-frequently-asked-questions.html), [CME Micro Metals specifications](https://www.cmegroup.com/markets/microsuite/metals.html), and [CME Micro E-mini specifications](https://www.cmegroup.com/articles/faqs/frequently-asked-questions-micro-e-mini-equity-index-futures.html).
