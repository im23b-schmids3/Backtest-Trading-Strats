# Walk-forward research conclusion

## Classification: weak and unstable evidence

The research process used two expanding chronological validation folds for 243 constrained core-parameter trials (9 feasible asset/timeframe series × 9 swing values × 3 distances). A final 365-day holdout was excluded from parameter and model selection.

The selected configurations produced 25 aggregate holdout trades. SOL 4h (N=7, distance=5) and XRP 4h (N=9, distance=15) contributed most of the reported holdout PnL. BTC 4h and BTC 1d selected configurations generated no holdout trades. These counts are too small to establish repeatability.

The interpretable ML comparison selected a depth-limited decision tree on validation (AUC 0.776), but its 0.55 threshold accepted all 25 holdout trades. It therefore showed **no genuine holdout improvement** over the unfiltered selected strategy. Permutation importance concentrated on 10-bar rate of change, but this is unstable because the training dataset has only 97 rows and must not be interpreted causally.

Rule-based regimes show most holdout PnL in bear-labelled entries, but all 25 holdout trades achieved TP1 and 24 were net winners. That combination is a strong warning that the sample is unusually favorable rather than proof that bear regimes cause performance. K-means clusters are included only as a descriptive comparison, not as a trading rule.

Cost stress was material for ETH 4h (double costs changed the holdout result from a small gain to a loss). SOL and XRP remained positive in that narrow test, but this does not address the lack of one-hour replay data, sparse trade counts, or selection bias. The Monte Carlo trade-order analysis is descriptive only; the terminal PnL is unchanged by reordering and drawdown estimates are limited by the small sample.

The research database records 243 trials. This multiple testing burden, the 25-trade final holdout, constrained data availability, and no observed ML holdout improvement rule out a claim of statistical significance or a robust edge.

Recommended next step: obtain an independent, gap-free crypto hourly source and a reliable gold intraday source; rerun the same frozen research protocol with a substantially larger final holdout before considering any new parameters or live use.
