# LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION

Sealed V2 only. V1 and `2d1fef6f49c0e251a9ae6657` are immutable negative evidence.

- BTCUSDT USD-M perpetual; 5-minute bars; Phase A uses only the existing V5 Phase-A bar manifest.
- No downloads, data writes, real Phase A/B/Alpha execution, grids, retuning, or Phase-B access.
- Long/short; next-bar-open; one active position; no pyramiding/averaging; 36-bar time stop; UTC 23:55 force-flat; stop-first; existing verified costs/slippage/tick/quantity.
- Reference: confirmed prior 24-bar high/low, excluding current bar.
- Penetration: max(0.25% reference, 4 ticks). Reclaim inside prior range in at most 2 completed bars.
- Body/range >=65%; long close>open, short close<open. Volume >=1.50x median prior 20 completed bars.
- Abs 24-bar same-UTC-session VWAP slope <=0.15% close; reclaim close within 0.75% current daily VWAP. Cross-session means SESSION_CONTEXT_UNAVAILABLE.
- One unresolved setup per direction/reference; repeat updates extreme. Twelve completed-bar same-direction cooldown after any terminal outcome.
- Stop-distance limits 0.20%..1.25% entry.
- Candidates only: LSMR-V2-2P0R=2R, LSMR-V2-2P5R=2.5R, LSMR-V2-3P0R=3R.
- Phase A hard gates exactly as user-sealed: >=163 executed/13mo, >=150 annualized, PF>=1.30, positive PnL/avg R, DD<=20R, 8 profitable months, <=3 zero months, concentration 35% month/30% best five, long/short >=25%, neither < -0.15R average, bootstrap median>0/lower>=-0.025R, positive extra slippage and best-trade removal, 3/4 2023 subperiods nonnegative, full reconciliation. >350 annualized only warning.
- Terminal dispositions exactly: TRADE_EXECUTED, RECLAIM_WINDOW_EXPIRED, VOLUME_REJECTED, CANDLE_REJECTED, REGIME_REJECTED, VWAP_PROXIMITY_REJECTED, SESSION_CONTEXT_UNAVAILABLE, STOP_DISTANCE_REJECTED, DUPLICATE_REFERENCE_SUPPRESSED, COOLDOWN_BLOCKED, NO_EXECUTABLE_ENTRY, COMPLIANCE_BLOCKED, SESSION_ENDED.
- Every proposed setup has deterministic setup_id, structure_id, direction, reference, extreme, event history, exactly one terminal disposition and a trade_id if executed.
