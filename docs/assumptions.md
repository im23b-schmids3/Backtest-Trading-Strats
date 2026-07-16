# Assumptions and data limitations

Crypto uses public Binance spot OHLCV through CCXT and needs no credentials. Gold is **GC=F (COMEX gold futures)** from Yahoo Finance by default; it is a futures proxy, not physical or spot gold. Yahoo commonly restricts 1-hour intraday history to a short rolling window. The downloader fails with a clear error if the requested interval cannot cover the requested start; it never substitutes another interval.

Prices use a resting limit at the stated Fib level. Adverse slippage is applied by increasing long entry / short-cover prices and decreasing long-sale / short-entry prices. Fees are charged on every executed notional and are **added on top of** the 2% initial-stop risk budget. Position sizing uses raw planned price distance, as specified. No leverage is enabled at the default `leverage: 1.0`; entries are rejected if notional exceeds available equity times leverage.

With OHLC data, intrabar sequence is unknowable. `conservative` prioritizes a stop over targets when both are reachable; `optimistic` prioritizes targets. `lower_timeframe_replay` replays explicitly supplied lower bars (the CLI loads cached `1h` bars for `4h`/`1d` tests) and fails clearly when they are missing. It is never silently approximated.

Forced end-of-test exits use the final completed candle close with adverse slippage. This research engine does not model funding, borrowing, exchange minimums, spread, tax, contract multipliers, liquidity/partial market fills, or corporate action adjustments. Results are not investment advice.
