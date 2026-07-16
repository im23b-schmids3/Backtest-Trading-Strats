# Quantitative validation report

## Scope and result

The baseline data request begins at `2022-01-01` and ends at each source's latest fully completed candle. The conservative feasible grid completed on 2026-07-13:

| Item | Result |
| --- | --- |
| Automated tests | 22 passed |
| Feasible conservative runs | 243 |
| Parameter combinations | N = 2–10; pivot distance = 5, 10, 15 |
| Valid series | BTC, ETH, SOL, XRP: 4h and 1d; GC=F gold proxy: 1d |
| Lower-timeframe replay runs | 0; no valid 1h replay series exists |

The run matrix and tables are in `reports/`. The complete source outcome is in `reports/data_availability.csv`; missing matrix combinations are in `reports/skipped_combinations.csv`.

## Deterministic validation

The test suite proves delayed long/short pivot confirmation, exact long/short Fib formulas, next-candle order activation, long/short limit fills and stops, all five 15/20/20/25/20 partial exits, post-TP1 stop migration, conservative and optimistic ambiguous-candle policies, lower-timeframe replay ordering, invalidation, duplicate setup suppression, two-percent sizing, cost impact, gap detection, and frequency-aware risk statistics.

## Defects found and fixed

1. Missing-candle reporting raised `IndexError` instead of a clear data error. The validator now reports exact missing timestamps and has regression coverage.
2. Binance's response contains a real shared 1-hour outage at 2023-03-24 13:00–14:00 UTC. The validator rejects it; it is not silently interpolated.
3. Yahoo's current multi-index response and duplicate adjusted-close column broke gold normalization. The downloader now selects the OHLC column level and removes the duplicate safely.
4. A request whose cached range began after the newly requested start could reuse incomplete history. Cache reuse now verifies start and latest-completed coverage.
5. `max_positions` was configured but not enforced. The order processor now enforces it.
6. Total planned open risk used the original stop after TP1. It now uses the active stop.
7. Residual floating-point position amounts could remain after partial exits. Remaining quantity is clamped at zero.
8. Trade logs recorded an order submission at the confirmation candle label. They now record next-candle open, the first instant the resting order can exist.
9. Sharpe and Sortino used daily scaling for every timeframe. They now infer periods per year from the equity timestamps.

## Ranking and warnings

`reports/result_matrix.csv` ranks configurations by:

`net return - 0.5 × absolute maximum drawdown + capped Sharpe / 10 + capped profit factor / 10 - minimum-trade penalty`

Sharpe and profit factor are capped at 3 in the score. The penalty is `0.5 × max(0, 20 - trades) / 20`. This deliberately prevents raw return or an extreme profit factor from dominating the rank.

The highest-ranked configurations were XRP 4h (N=8, distance=15) and several SOL 4h N=7 variants. They should be treated as **fragile candidates**, not proven profitable: many runs have fewer than 20 trades, profit factors can be extreme because losses are small or sparse, and the result selection scanned 243 combinations. `confidence_warnings.csv` carries these flags per run. The gold daily series produced very few trades and is especially inconclusive.

The benchmark table includes equal-period buy-and-hold, zero-return cash (implicitly 0%), and a simple 200-bar close-above-moving-average trend filter. A positive strategy return alone is not interpreted as an edge.

## Remaining data and execution limits

* Binance omits the same hourly candle for all four requested crypto symbols, so strict 1h tests and 1h replay cannot be run without a different validated source.
* Yahoo Finance's free GC=F data cannot provide the requested 2022-to-present hourly history and has no reliable 4h interval. GC=F is a futures proxy, not physical spot gold.
* Lower-timeframe replay is implemented and tested, but has no validated historical replay source for this period; no replay results are presented as comparable to conservative results.
* OHLC conservative fills remain a deliberately adverse approximation. Funding, borrow, liquidity, contract multipliers, exchange minimums, and taxes are outside this baseline.
