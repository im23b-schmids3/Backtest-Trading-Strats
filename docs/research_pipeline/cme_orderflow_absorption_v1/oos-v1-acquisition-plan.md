# CMEOrderflowAbsorption.ES_V1_BACKTEST - OOS validation-data acquisition plan

## Status: COST_QUOTED_NOT_ACQUIRED

No validation data was acquired, no DBN was scanned, no strategy was executed, and no PnL was calculated. This plan records owner-obtained official Databento evidence and does not issue a new network call.

Official `symbology.resolve` evidence for `GLBX.MDP3` / `raw_symbol` resolves `ESU6` to instrument ID `42140870` for d0 `2026-08-03` through d1 `2026-08-08`, with `OK`, `partial=[]`, and `not_found=[]`. ESU6 alone covers every OOS RTH date; no rollover is required. One official owner-obtained `metadata.get_cost` quote for `GLBX.MDP3` / `mbo` / `raw_symbol` `ESU6` over `[2026-08-03T00:00:00Z, 2026-08-08T00:00:00Z)` is `estimated_usd=5.736491556466`, with `quote_count=1`.

The manifest target is `data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn`. It remains not acquired: `data_acquired=false`, and file SHA-256, bytes, record count, and coverage gaps are null. Acquisition requires separate authorization.

The interval is exactly `[2026-08-03T00:00:00Z, 2026-08-08T00:00:00Z)` for RTH dates August 3 through August 7. July 31 is warmup only for frozen prior-RTH structural levels on August 3; it cannot generate an OOS interaction or trade.

Frozen p95 literals remain `0.7977986403366786` and `0.7785691162188411`; selection remains same-interaction ABSORPTION_PLUS_REPLENISHMENT only with no refit. The sealed fixed risk budget is `250.00 USD`; stop buffer is exactly five ES ticks from the completed interaction zone. `SINGLE_2R`, one-tick entry/exit adverse slippage, $3 per-contract-per-side commission, and the 22:45 UTC cutoff/flat behavior remain unchanged.
