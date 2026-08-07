# Fib prospective V1 holding-period audit

Audit date: 2026-08-07. Scope was restricted to the sealed V1 specification, current V1 implementation, and the completed development artifacts at `research_runs/FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION/development/run-20260807-01`. No holdout file, row, value, result, or path was opened; no strategy/backtest was started.

## Conclusion

**Classification: `OVERNIGHT_POSITIONS_ALLOWED`; `INTRADAY_DATA_RESOLUTION_REQUIRED`.** V1 has no 22:45 UTC force-flat mechanism, no in-position holding-period limit, and no per-day session liquidation. Its only forced position exit outside stop/targets is the final available development bar (`DATA_END_FORCE_CLOSE`). Consequently, V1 must not be patched in place; the separate V2 execution contract below is required for a 22:45 hard-flat policy.

## Evidence classification and exit audit

| Question | Result | Evidence |
|---|---|---|
| Explicit `22:45 UTC` force close | FALSE | **Code-proven:** V1 contains no `22:45`, cutoff, or UTC-session test in `runner.py`, `execution.py`, `strategy.py`, or `models.py`. `process_position` only tests stop then TP1..TP5. |
| No positions after 22:45 | FALSE | **Artifact-proven:** open-after-22:45 trades: BTC 173/173; ETH reference 259/963; ETH 0.830 262/951. |
| No overnight positions | FALSE | **Artifact-proven:** crossing-midnight trades equal those open after cutoff in each candidate. |
| Session-end liquidation | FALSE (except data end) | **Code-proven:** the loop never has a daily/session boundary. It appends `SESSION_OR_DATA_END` only to pending setups after the bars loop; an active position is force-closed only at `bars[-1]`. |
| Maximum holding period tied to UTC day end | FALSE | **Specification- and code-proven:** sealed spec says “There is no in-position time stop”; code checks only `anchor_age_days` for unfilled orders. |

The sealed specification’s “immutable boundary” is the data/session boundary in its end-of-data rule, not a daily 22:45 boundary. The implementation confirms this: `runner.py` calls `_exit(... bars[-1].close, ... "DATA_END_FORCE_CLOSE", ...)` only after all bars have been processed. Stop/target legs receive adverse fill and fees through `_exit`; this same normal exit cost model is also the applicable model for any future forced exit.

## Artifact inspection and calculation method

All three candidates’ `trades.json`, `events.json`, and `setup-outcomes.json` were inspected. Event records contain only `ORDER_SUBMITTED` and `ORDER_FILLED`; no session/force-flat event exists. Each executed trade record was calculated as follows, using its `trade_id`, `direction`, `entry_timestamp`, and the maximum timestamp among its exit legs as final exit: `duration = final_exit - entry`; `crossed_2245 = any(entry <= YYYY-MM-DDT22:45:00Z < final_exit)` over UTC dates in the interval; `crossed_midnight = entry_utc_date != exit_utc_date`; `utc_days_touched = (exit_utc_date - entry_utc_date) + 1`. `closed_by_2245_entry_day` means final exit was on the entry UTC date at or before 22:45:00Z. The complete calculation population is the trade-record count in the table; the per-trade source fields remain immutable in the cited artifacts.

| Candidate | Trade records / unique IDs | Events (submitted, filled) | Outcomes (active blocked, expired, session/data end, executed) |
|---|---:|---:|---:|
| FIB09-BTC-1D-POST0786 | 173 / 173 | 2,190 / 173 | 1,359 / 657 / 3 / 173 |
| FIB09-ETH-4H-POST0786-REFERENCE | 963 / 963 | 11,639 / 963 | 7,179 / 3,497 / 2 / 963 |
| FIB09-ETH-4H-POST0830 | 951 / 951 | 11,639 / 951 | 7,204 / 3,482 / 4 / 951 |

| Candidate | Executed | Closed by 22:45 entry day | Open after 22:45 | Crossed midnight | >24h | >48h | >72h | Median h | Mean h | Max h | Violation % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIB09-BTC-1D-POST0786 | 173 | 0 | 173 | 173 | 59 | 37 | 11 | 24.0000 | 40.0925 | 192.0000 | 100.0000% |
| FIB09-ETH-4H-POST0786-REFERENCE | 963 | 704 | 259 | 259 | 17 | 4 | 2 | 4.0000 | 6.9657 | 88.0000 | 26.8951% |
| FIB09-ETH-4H-POST0830 | 951 | 689 | 262 | 262 | 23 | 4 | 2 | 4.0000 | 7.2892 | 88.0000 | 27.5499% |

Violation percentage is `open_after_2245 / executed * 100`; it is not a performance metric. Every BTC daily trade is necessarily still open at the first 22:45 following its 00:00 entry because the next evaluable bar is 00:00 on a later day.

## Ten longest calculated trades per candidate

All timestamps are UTC. Each row gives `trade_id | direction | entry -> final exit | duration | final-exit reason | first crossed 22:45 boundary`; the entry/exit UTC dates, crossing-midnight flag, and days touched follow directly from the displayed timestamps and duration calculation above.

### FIB09-BTC-1D-POST0786

| Trade | Direction | Entry -> final exit | Hours | Reason | First 22:45 |
|---|---|---|---:|---|---|
| trade-b7bc99a30f97f41e | LONG | 2024-05-23T00:00Z -> 2024-05-31T00:00Z | 192 | STOP | 2024-05-23T22:45Z |
| trade-2f0309fc56ff16e9 | SHORT | 2022-09-21T00:00Z -> 2022-09-27T00:00Z | 144 | STOP | 2022-09-21T22:45Z |
| trade-ff75e6723c509174 | SHORT | 2023-10-25T00:00Z -> 2023-10-30T00:00Z | 120 | STOP | 2023-10-25T22:45Z |
| trade-f9bd0a6a3a1761bb | SHORT | 2023-08-09T00:00Z -> 2023-08-14T00:00Z | 120 | TP5 | 2023-08-09T22:45Z |
| trade-b2a800422f4b40e5 | LONG | 2023-12-13T00:00Z -> 2023-12-18T00:00Z | 120 | STOP | 2023-12-13T22:45Z |
| trade-fdd4ee86ee38d16a | SHORT | 2023-05-18T00:00Z -> 2023-05-22T00:00Z | 96 | TP5 | 2023-05-18T22:45Z |
| trade-f802a0a043137228 | LONG | 2023-10-12T00:00Z -> 2023-10-16T00:00Z | 96 | TP5 | 2023-10-12T22:45Z |
| trade-b8ab16341f0379f9 | LONG | 2022-06-22T00:00Z -> 2022-06-26T00:00Z | 96 | TP5 | 2022-06-22T22:45Z |
| trade-aea6a01c2dd62333 | SHORT | 2023-03-23T00:00Z -> 2023-03-27T00:00Z | 96 | TP5 | 2023-03-23T22:45Z |
| trade-60b5b6d6433376f2 | SHORT | 2023-06-06T00:00Z -> 2023-06-10T00:00Z | 96 | TP4 | 2023-06-06T22:45Z |

### FIB09-ETH-4H-POST0786-REFERENCE

| Trade | Direction | Entry -> final exit | Hours | Reason | First 22:45 |
|---|---|---|---:|---|---|
| trade-7cf19d157552d221 | LONG | 2024-08-05T04:00Z -> 2024-08-08T20:00Z | 88 | TP5 | 2024-08-05T22:45Z |
| trade-9801dc84f73286e7 | SHORT | 2022-11-10T16:00Z -> 2022-11-14T00:00Z | 80 | TP5 | 2022-11-10T22:45Z |
| trade-c6edbf523ad93293 | LONG | 2022-04-14T16:00Z -> 2022-04-16T20:00Z | 52 | TP5 | 2022-04-14T22:45Z |
| trade-ae594d032063a034 | LONG | 2023-07-24T12:00Z -> 2023-07-26T16:00Z | 52 | TP5 | 2023-07-24T22:45Z |
| trade-6aca0a5fcea09f83 | SHORT | 2023-04-01T00:00Z -> 2023-04-02T12:00Z | 36 | TP4 | 2023-04-01T22:45Z |
| trade-0a103df496d73ef8 | SHORT | 2023-05-13T00:00Z -> 2023-05-14T12:00Z | 36 | STOP | 2023-05-13T22:45Z |
| trade-9b23682e455db15f | LONG | 2022-08-17T16:00Z -> 2022-08-19T00:00Z | 32 | STOP | 2022-08-17T22:45Z |
| trade-55cf7c2516d0d530 | LONG | 2022-10-31T20:00Z -> 2022-11-02T04:00Z | 32 | STOP | 2022-10-31T22:45Z |
| trade-f76a11520f4abd53 | SHORT | 2023-08-03T16:00Z -> 2023-08-04T20:00Z | 28 | TP5 | 2023-08-03T22:45Z |
| trade-b0ade19a666cbcc6 | SHORT | 2023-05-20T16:00Z -> 2023-05-21T20:00Z | 28 | TP5 | 2023-05-20T22:45Z |

### FIB09-ETH-4H-POST0830

| Trade | Direction | Entry -> final exit | Hours | Reason | First 22:45 |
|---|---|---|---:|---|---|
| trade-433d4b8afbc95bf2 | LONG | 2024-08-05T04:00Z -> 2024-08-08T20:00Z | 88 | TP5 | 2024-08-05T22:45Z |
| trade-8cc1643169a96c83 | SHORT | 2022-11-10T16:00Z -> 2022-11-14T00:00Z | 80 | TP5 | 2022-11-10T22:45Z |
| trade-a703477b31e4ef6e | LONG | 2023-07-24T12:00Z -> 2023-07-26T16:00Z | 52 | TP5 | 2023-07-24T22:45Z |
| trade-744921b71d2e1964 | LONG | 2022-04-14T16:00Z -> 2022-04-16T20:00Z | 52 | TP5 | 2022-04-14T22:45Z |
| trade-0fb8d223be6dcf59 | SHORT | 2023-06-30T20:00Z -> 2023-07-02T20:00Z | 48 | STOP | 2023-06-30T22:45Z |
| trade-6592487c1f0f2467 | SHORT | 2024-08-30T16:00Z -> 2024-09-01T12:00Z | 44 | TP5 | 2024-08-30T22:45Z |
| trade-a5c32c2477afa078 | LONG | 2023-08-14T00:00Z -> 2023-08-15T12:00Z | 36 | STOP | 2023-08-14T22:45Z |
| trade-6583fbad0a07c240 | SHORT | 2023-05-13T00:00Z -> 2023-05-14T12:00Z | 36 | STOP | 2023-05-13T22:45Z |
| trade-5449e710b6d11b98 | SHORT | 2023-04-01T00:00Z -> 2023-04-02T12:00Z | 36 | TP4 | 2023-04-01T22:45Z |
| trade-ab033f0d395049aa | LONG | 2022-08-17T16:00Z -> 2022-08-19T00:00Z | 32 | STOP | 2022-08-17T22:45Z |

## Current behavior when exits do not occur

When neither TP nor stop occurs by 22:45, V1 does nothing: it processes the next available bar and the position remains active. At UTC day end it also does nothing; neither the V1 loop nor `process_position` compares dates. Across multiple days it keeps the single active position, blocks new orders with `ACTIVE_POSITION_BLOCKED`, and can only close the position on a later stop/target leg or at the final available bar with `DATA_END_FORCE_CLOSE`. The order anchor-age expiry applies before entry only and cannot close a live trade.

## Separate V2 execution contract (not implemented)

This is a proposed sealed V2-only contract; it makes no change to V1, its artifacts, candidate registry, or results.

1. Hard flat: at exactly `22:45:00 UTC`, force-close every remaining position quantity using reason `FORCED_SESSION_EXIT_2245`; no position may remain open after that instant.
2. Entry control: reject/cancel any unfilled entry at or after the cutoff; no entry may be created or filled at/after `22:45:00 UTC`. Eligibility resumes at `00:00:00 UTC` on the next UTC day only.
3. Partial quantity: retain every previously realized TP leg; force-close exactly `remaining_quantity` as one final leg, then set remaining quantity to zero and terminally close the trade. No partial quantity may be discarded or re-opened next day.
4. Cost/accounting: forced exit is a normal market exit using the existing adverse exit slippage, fee, quantity rounding, raw/fill price, gross/net PnL, cash/equity, and unique exit-leg-ID logic. Reconcile `initial_quantity = all TP/stop/forced quantities`, entry and all exit fees/slippage, leg PnL to trade PnL, each trade to cash/equity, and final equity to opening equity plus realized net PnL. Emit a force-flat event and preserve reason, timestamp, raw/fill price, and quantity in artifacts.
5. Conservative deterministic precedence: at the cutoff timestamp, first apply any previously known completed-bar state only; then evaluate a contemporaneous protective stop before targets; if still nonzero, execute `FORCED_SESSION_EXIT_2245` for all remainder; do not allow a target after the forced exit. An entry is never allowed at/after cutoff. This prevents same-bar optimism and makes the forced exit terminal.

## Data-resolution finding

Exact 22:45 execution is **not representable** by current ETH 4h or BTC 1d OHLC: their bar timestamps are 4-hour/daily boundaries and neither supplies a 22:45 observation. A 4h or daily close must not be substituted as an approximation. `INTRADAY_DATA_RESOLUTION_REQUIRED` is therefore required. The minimum practical deterministic bar input is aligned 1-minute ETH and BTC OHLCV; exact executable market-fill reconstruction would additionally require timestamped trade/quote data (OHLC alone cannot prove the exact fill). A read-only repository filename search, excluding holdout paths and performing no download, found only `docs/research_pipeline/fib_prospective_v1/data-contracts/ETH_4H/manifest.json`, `docs/research_pipeline/fib_prospective_v1/data-contracts/BTC_1D/manifest.json`, and `config/smoke_btc_daily.yaml`; it found no suitable lower-timeframe ETH/BTC OHLCV asset.

## Validation

`git diff --check` was run after this report was added. No production backtest or strategy execution was invoked.
