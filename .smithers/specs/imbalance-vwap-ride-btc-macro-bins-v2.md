# Sealed contract: ImbalanceVWAPRide.BTC_MACRO_BINS_V2_EXPLORATORY

This is a new, post-hoc V2 development study. Preserve V1 run
`research_runs/ImbalanceVWAPRide.BTC_EXPLORATORY/ff0afc85ef4b46c0bf671cfb/`
byte-for-byte. Use the existing standalone engine and immutable BTCUSDT manifest
`data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json`
(dataset hash `c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc`).
Never download, renormalize, place live orders, expose secrets, or transmit raw
aggregate trades. Process raw Parquet locally in bounded batches.

Labels on every artifact: `POST_HOC_V2_DEVELOPMENT`, confirmation false,
optimization false, external holdout required. New immutable root:
`research_runs/ImbalanceVWAPRide.BTC_MACRO_BINS_V2_EXPLORATORY/<run_id>/`.

## Sealed V2 baseline

`bin_size_usd=50`, `min_imbalance_ratio=3`, `stacked_bins=3`,
`min_bin_volume_btc=35`, `vwap_slope_bars=24`, `move_away_bars=2`,
`zone_expiry_bars=36`, `stop_buffer_bins=2`, `target_r_multiple=2`,
`maximum_active_zones_per_direction=3`, `maximum_trades_per_utc_day=1`, and
`NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST`.

## Exact unique one-factor registry

Baseline plus unique OAT settings only: bins `30,50,100`; min volume
`20,35,50`; VWAP slope `12,24,36`; ratio `2.5,3,4`; stacked bins `2,3,4`;
move-away `1,2`; expiry `20,36,48`; stop buffer `1,2,3`; target R `1.5,2,2.5`.
No Cartesian/random/hidden optimization. Stable IDs/order.

## Required V2 corrections

Target and risk are calculated only after adversely slipped, quantized next-bar
actual entry: long risk `entry-(IZ_bottom-buffer*bin)` and target
`entry+risk*R`; short symmetric. Persist theoretical zone-edge risk, actual
risk, entry gap, target distance, fees, slippage, total costs, cost/risk,
gross/net R. Non-positive risk or unprofitable-side target is explicitly
invalid/non-executable.

Remove terminal zones from active matching immediately. Persist terminal time,
state/reason, lifetime, move-away and retest. Terminals: EXPIRED, INVALIDATED,
TRADED, SUPERSEDED. Invalidate long closes below IZ bottom, short closes above
IZ top, VWAP regime loss, time expiry, and after trade.

Report actual quantity/notional fee and cost diagnostics: median initial risk,
gross risk, round-trip costs, cost/risk, shares over 10/25/50%, gross/net PF
and gross/net average R. Distinguish pre-cost edge failure, cost-destroyed edge,
and restrictive-threshold sample insufficiency.

## Development and gates

Run Jan--Apr 2024 only. Per config write all funnel, zone, trade, gross/net,
cost, concentration, monthly and long/short metrics. Enforce exact funnel
reconciliation. Gate: >=40 trades, net PF>1.05, avg net R>0, monthly positive
PnL concentration <=60%, best five <70%, finite DD, reconciled funnel and
long/short reporting. If none pass, finish `DEVELOPMENT_EDGE_NOT_FOUND` and do
not open validation, July, or Alpha. Otherwise existing deterministic selection
may freeze max three then apply existing validation/locked/Alpha gates.

Alpha is strictly conditional; retain one MBT / 0.1 BTC, actual quantized risk,
versioned rule manifest, Binance-to-MBT limitations and no eligibility claim
unless every earlier gate passes.

Add tests for exact baseline/registry/no Cartesian product, actual-entry risks
and targets, entry gap, terminal-zone cleanup, quantity/notional fees,
cost-to-risk, gross/net, insufficient trades, V1 preservation, deterministic
rerun and immutable collision. Run focused tests, full research_pipeline,
compileall and diff check; repair then execute V2 automatically.
