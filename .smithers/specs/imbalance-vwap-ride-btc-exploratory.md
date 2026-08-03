# Sealed execution contract: ImbalanceVWAPRide.BTC_EXPLORATORY

Implement and execute this complete standalone exploratory study. Do not modify
ValueAreaTrap, ValueAreaAcceptance, ES, equity, cross-market, frozen, or
historical artifacts. No downloads, renormalization, live orders, secrets, or
raw aggregate-trade transmission to a model service. Process raw data locally.

## Identity and data

- Strategy ID: `ImbalanceVWAPRide.BTC_EXPLORATORY`
- Adapter ID: `imbalance-vwap-ride-btc-1`
- Source: Binance BTCUSDT USD-M aggTrades; exact `is_buyer_maker` aggressor
  classification; UTC 5-minute bars.
- Immutable manifest: `data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json`
- Dataset hash: `c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc`.
- Development Jan--Apr 2024; validation May--Jun; locked July test once, labelled
  `INTERNAL_LOCKED_TEST_NOT_PRISTINE_HOLDOUT` (never an untouched external holdout).

Create a separate module, adapter, runner, CLI, tests, and immutable root
`research_runs/ImbalanceVWAPRide.BTC_EXPLORATORY/<study_run_id>/`. Never import
or call ValueAreaTrapAdapter or ValueAreaAcceptanceAdapter. Do not use stop-run,
divergence, value-area entry, or prior-POC target logic. Generic manifest/hash,
fee/slippage/quantization/deterministic-ID utilities may be reused.

## Footprint / regime

Stream all seven partitions in bounded batches. Per trade, calculate
`bin_floor=floor(price/bin_size_usd)*bin_size_usd`, half-open bin interval,
aggressive buy when buyer_is_maker=false, otherwise aggressive sell. Per 5m bar
and bin persist exact Decimal buy/sell/total/delta, OHLCV and CVD. Require
total=buy+sell and delta=buy-sell. Preserve all batch/month/bucket boundaries.
Persist a content-addressed footprint/bar Parquet dataset, Parquet SHA-256,
dataset hash, counts, bounds, schema and source-manifest hash.

Reset Daily VWAP at UTC midnight. A completed-bar long regime requires close>
current daily VWAP and current VWAP>VWAP exactly `vwap_slope_bars` completed
bars earlier. Short is symmetric. Never use current/future information.

## Baseline rules

```
bin_size_usd=10.0; min_imbalance_ratio=3.0; stacked_bins=3;
min_bin_volume_btc=5.0; vwap_slope_bars=10; move_away_bars=2;
zone_expiry_bars=20; stop_buffer_bins=2; target_r_multiple=2.0;
maximum_active_zones_per_direction=3; maximum_trades_per_utc_day=1;
entry_execution=NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST
```

Long bin: volume >= minimum and buy >= ratio*sell, or sell=0 and buy>=minimum.
Short is symmetric. Form exactly one maximal (not sliding/overlapping) sequence
of at least stacked adjacent qualifying bins per direction/bar. Bounds are
lowest floor and highest floor+bin size. Persist IDs, lineage, volume/delta,
parameters, expiry and state. Cap each direction at 3. Merge overlapping same
direction zones deterministically (earliest creation, broadest geometry, full
lineage). Retest priority is newest still-valid zone. One trade per zone/day.

Long arms after `move_away_bars` completed lows above zone top; short after highs
below bottom. Long retest crosses/touches top, closes >= top and retains long
VWAP; short symmetric at bottom. Confirm at retest close, enter only next 5m
open; missing next bar is `NO_EXECUTABLE_ENTRY`; no same-bar fill. Stops:
long bottom-buffer*bin, short top+buffer*bin. Fixed R target after adverse
slippage and quantization. Use existing verified BTC fees, slippage, price tick
and quantity rules. Exit testing only post entry; same-bar stop first; force flat
at UTC day end. Invalidate for adverse zone-boundary close, expiry, VWAP loss
pre-retest, post-execution, or deterministic opposite-zone supersession.

Funnel must reconcile exactly:
`proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades`.
Also report imbalance sequences, zones created, VWAP-qualified zones,
move-away-confirmed zones and retest triggers. Fail a run if its funnel fails.

## Pre-registered diagnostics / gates

Development only, baseline plus UNIQUE one-factor variations only:

```
bin_size_usd: 10,20
min_imbalance_ratio: 2.5,3.0,4.0
stacked_bins: 2,3,4
min_bin_volume_btc: 2.5,5.0,10.0
vwap_slope_bars: 6,10,20
move_away_bars: 1,2
zone_expiry_bars: 12,20,36
stop_buffer_bins: 1,2,3
target_r_multiple: 1.5,2.0,2.5
```

No Cartesian/Bayesian/genetic/random search, hidden variants, or highest-PnL
selection. Report trades, net PnL, PF, average/median net R, win rate, DD,
losing streak, months, long/short split, costs, best-day/best-five contribution.
Development candidate gate: >=40 trades, PF>1.05, avg net R>0, no month >60%
positive PnL, best five <70%, finite DD, funnel and both directions reported.
If none: complete reports, `DEVELOPMENT_EDGE_NOT_FOUND`, no Alpha claim. If
several: freeze max 3 stable candidates with
`selection_method=PRE_REGISTERED_ROBUSTNESS_FILTER`, optimization false.

Validation gates: >=15 trades, PF>1, avg net R>0, positive PnL and funnel.
Freeze exactly one: highest PF, lower DD, higher count, baseline-nearest,
lexicographic ID. July passes only with >=8 trades, PF>1, avg net R>0, positive
PnL and funnel. Always confirmation false, external holdout required,
optimization false.

## Conditional Alpha proxy

Only if July passes. Version alpha rules with sources, retrieval date, costs and
limitations. It is BTCUSDT-to-MBT proxy only, never CME/Alpha compliance proof.
Starting 25k, target 1.5k, MLL 1k, DLG 500; no eval consistency; qualified 40%;
one MBT only. MBT 0.1 BTC, 5 points/$0.50 tick. Round to 5 point ticks; apply
versioned conservative fees/slippage; no weekend/out-of-availability entries;
flat 16:20 ET; day starts 18:00 ET, DST-correct.

Use final frozen trades only, chronological and block bootstrap, deterministic
>=10,000 paths. Report pass/breach/unfinished, MLL/DLG, days, rebills/resets,
costs, downside, one tick/double commission and 20/30/40% degradation. Under 30
trades: `pass_probability_insufficient_sample=true`. With pass paths, simulate
qualified sequentially: 40% consistency, five $200 days, 50% profit withdrawal,
$200 minimum, $1000 maximum, 90% share, ongoing MLL/DLG; report all stated
payout eligibility, days, buffer, survival, second payout, withdrawals/share
and blocking/breach metrics.

## Artifacts and tests

Every immutable artifact contains run/dataset/specification/parameter/code hashes,
evidence label, timestamp and collision checks. Write every named artifact from
the user request: data/footprint validation, baseline/ablation/development,
selection/frozen/validation/final/locked reports, events/zones/trades/months/
funnel and three Alpha/final reports.

Add focused tests for data/footprint/boundaries, VWAP no-lookahead, zone formation
and lifecycle, execution/IDs, exact ablation registry/gates/freeze/locked behavior
and all specified Alpha/Monte Carlo/qualified rules. Then focused tests, complete
`tests/research_pipeline`, compileall, diff check. Repair automatically. Keep
real execution fail-closed until all pass; then execute automatically.
