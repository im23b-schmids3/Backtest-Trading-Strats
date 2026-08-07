# Source-selection report: FibRetracementContinuation.ETH_BTC_V2_INTRADAY_FORCE_FLAT_2245

## Owner-selected prospective source

**OWNER_SELECTED_PROSPECTIVE_VENUE.** The repository owner has selected **BINANCE**, instrument type **USD_M_PERPETUAL**, for the BTCUSDT and ETHUSDT perpetual instruments. This prospective V2 decision is explicit authorization for the dedicated V2 public-data acquisition contract only. It is **not evidence about V6 or V6.5**, does not amend any existing research artifact, and does not authorize strategy implementation, live order routing, funding, borrowing, or leverage.

## Acquisition identity

The admitted source is Binance USD-M Futures 1m klines only: official monthly archive URLs under `https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1m/`, followed only where necessary by the official public USD-M REST endpoint `https://fapi.binance.com/fapi/v1/klines` for complete current UTC days not yet represented by an archive. No credentials are required or permitted. Raw data may exist only below `data/fib_prospective_v2/binance_usdm/{SYMBOL}/1m/`.

## Boundaries

The retained V2 future candidate registry remains unchanged. Development is `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`; holdout begins at `2025-01-01T00:00:00Z`. Acquisition validation is allowed on all acquired raw data, but no strategy, signal, trade, candidate, optimization, or metric computation may access holdout.
