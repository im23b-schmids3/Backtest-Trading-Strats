# Strategy V2 lifecycle comparison

## Updated lifecycle

Strategy V1 remains unchanged. Strategy V2 now keeps independent causal generations for each direction:

- a long generation has a fixed low anchor and a moving higher-high extreme;
- a short generation has a fixed high anchor and a moving lower-low extreme;
- a meaningful pullback creates a provisional higher-low/lower-high candidate;
- the existing distance and percentage-move filters promote that candidate to a new setup;
- each setup has its own identifier, Fibonacci versions, order replacements, and invalidation state.

An anchor is also bounded by `max_anchor_age_days`, configured by timeframe with defaults of 30 days for 1h, 60 days for 4h, and 180 days for 1d. A stale unfilled setup emits `anchor_max_age` and the strategy searches for a newer anchor. Execution, sizing, stops, targets, fees, slippage, and intrabar policy are unchanged.

## Research outputs

The complete cached-data run contains 270 V2 configurations across nine validated asset/timeframe series. Updated lifecycle diagnostics are in `reports/v2/v2_diagnostics_by_configuration.csv`; the requested previous-versus-updated comparison is in `reports/v2/v2_behavioral_comparison.csv` and is included in `reports/v2/v2_report.html`.

Across the nine series, updated V2 created substantially more anchors and setups per trend than the previous one-generation lifecycle. Updated trades per year increased on every series in this grid. The percentage of setups invalidated by maximum age before entry ranged from 26.4% to 43.6% across the asset/timeframe aggregates.

The previous report did not retain anchor age at entry, so the previous average-anchor-age cells are intentionally blank. Previous setup/trade-per-trend values are reconstructed from the old lifecycle funnel and should be treated as aggregate estimates.

## Interpretation

These changes are mechanically closer to discretionary Fibonacci trading: a trend can produce several independent retracement setups, and stale anchors are prevented from remaining active indefinitely. That is a better lifecycle model, but it is not evidence of profitability or robustness. The larger setup and trade counts also increase the opportunity for overtrading, so V2 should remain a research hypothesis until independent out-of-sample validation.

The same validated cache was used for comparison. Crypto 1h and lower-timeframe replay remain excluded because of the documented source gap; gold 1h/4h are unavailable from the documented free source.
