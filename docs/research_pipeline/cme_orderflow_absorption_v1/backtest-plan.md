# CMEOrderflowAbsorption.ES_V1_BACKTEST - OOS contract plan

## Status and chronology

This sealed OOS backtest contract is not an engine or a backtest. The July 20-31, 2026 pilot remains DEVELOPMENT/DESIGN evidence only and can never support an OOS profitability claim. The sealed OOS interval is `[2026-08-03T00:00:00Z, 2026-08-08T00:00:00Z)`; data is cost-quoted and not acquired.

## OOS data manifest (do not acquire)

Owner-obtained official Databento evidence resolves `GLBX.MDP3` / `raw_symbol` `ESU6` to instrument ID `42140870` for d0 `2026-08-03` through d1 `2026-08-08`, with `OK`, `partial=[]`, and `not_found=[]`. ESU6 alone covers all five OOS RTH dates; no rollover is required. One owner-obtained official `metadata.get_cost` quote for `GLBX.MDP3` / `mbo` / `raw_symbol` `ESU6` over `[2026-08-03T00:00:00Z, 2026-08-08T00:00:00Z)` is `5.736491556466` USD. This plan does not issue a network call, acquire data, scan a DBN, execute a strategy, or calculate PnL.

The target acquisition path is `data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn`. It remains absent: file hash, bytes, record count, and coverage gaps are null until separately authorized acquisition. July 31 is permitted only as warmup for frozen prior-RTH structural levels on August 3; it cannot supply an OOS interaction or trade.

## Frozen causal decision

Load `absorption_p95 = 0.7977986403366786` and `replenishment_p95 = 0.7785691162188411` literally. Never derive distributions, p95s, calibrations, or thresholds from OOS data. A candidate is tradable only when the same completed interaction is both HIGH and STRONG at a mandatory structural level. Response fields are prohibited from selection, eligibility, entry, stop, target, and sizing. `BUYER_ABSORPTION` is LONG and `SELLER_ABSORPTION` is SHORT.

## Execution and fixed-risk design

Use only `SINGLE_2R`. After 1 ms decision latency and 1 ms order latency, take the first valid executable observation strictly after `interaction_end`: marketable long at ask or short at bid. `ENTRY_SLIPPAGE=1 adverse ES tick`: long entry fill is ask + 0.25 and short entry fill is bid - 0.25. `EXIT_SLIPPAGE=1 adverse ES tick`: long exit fill is bid - 0.25 and short exit fill is ask + 0.25. ES is 0.25 points/tick, $12.50/tick, and $50/point. Commission is exactly $3.00 per contract per side.

The completed interaction zone is the sole structural stop anchor. With `stop_buffer_ticks = 5` and ES tick 0.25, LONG stop is `zone_low - (5 * 0.25) = zone_low - 1.25` points; SHORT stop is `zone_high + (5 * 0.25) = zone_high + 1.25` points. Never measure the buffer from entry, level center, or NBBO. Reject missing/invalid/non-positive stop distances. Target exactly 2R. Resolve exits from chronologically ordered MBO events only and fail closed on an ambiguous collision.

The sealed fixed risk budget is `250.00 USD`. `entry_fill` includes entry slippage and `stop_exit_fill_assumption` includes exit slippage. Calculate `one_contract_price_risk_usd = abs(entry_fill - stop_exit_fill_assumption) * 50.00`; commissions are `6.00`; initial risk is price risk plus 6.00; and `contracts = floor(250.00 / one_contract_initial_risk_usd)`. Do not double-count slippage. Contracts are integer ES only: no fractions, MES replacement, compounding, or budget overrun. Below one contract is `INSUFFICIENT_RISK_BUDGET_FOR_ONE_ES_CONTRACT` and no trade. Report raw price risk, slippage contribution, commissions, and net economics separately.

## Session and state machine

No entry is permitted at or after 22:45:00.000000000 UTC. Cancel all pending orders no later than cutoff. Force-flat an open position at the LAST valid executable ES market observation in the inclusive `22:44:59.000000000 through 22:45:00.000000000 UTC` liquidation window with exactly one adverse exit tick and $3.00 per-contract exit commission. With no valid observation, fail closed as `CUTOFF_EXECUTION_INTEGRITY_FAILURE`; do not invent a fill after cutoff. No position may exist after cutoff. Eligibility resets at 00:00:00.000000000 UTC; one position maximum; each interaction ID may be submitted only once.

## Contract-only tests

Pure synthetic fixtures must verify frozen p95/PLUS rules and no refit; exact five-tick zone stops; fixed $250 risk; fill-inclusive one-tick entry/exit slippage and $6 commissions; integer ES-only sizing and insufficient budget; unchanged cutoff/flat behavior; and the cost-quoted, not-acquired ESU6 manifest. Tests must not load or scan market data, run a strategy, backtest, or calculate PnL.
