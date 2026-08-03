# Sealed execution contract: ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY

This is a separate post-hoc V3 study. Preserve all V1 and V2 artifact roots
unchanged. Labels: `POST_HOC_V3_LONG_ONLY`, confirmation false, optimization
false, external confirmation required, selection
`PRE_REGISTERED_LONG_ONLY_OAT`. Strategy/adapter IDs are
`ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY` and
`imbalance-vwap-ride-btc-long-only-v3-1`.

## Data

Download only official Binance USD-M BTCUSDT aggTrades archives for
2024-08, 09, 10, 11, 12 and 2025-01 (reuse valid local archives if present).
No other network calls or downloads. Validate URL, archive/parquet hashes,
integrity, symbol/schema, timestamp/month bounds, monotonic IDs/timestamps,
duplicates, gaps/overlaps, rows and first/last stamp. Locally normalize to a
new content-addressed Parquet dataset/manifest. Fail closed on an invalid or
incomplete month. Never transmit raw rows. The period is
`NEW_TEMPORAL_V3_DEVELOPMENT_PERIOD`, not a pristine holdout.

## Long-only strategy

Use existing standalone ImbalanceVWAPRide components, but never produce a
short setup/order/fill/trade/PnL/candidate. UTC 5m bars and midnight-reset daily
VWAP. Long regime: completed close > VWAP and VWAP > exactly
`vwap_slope_bars` earlier completed VWAP. Exact aggressor buy when
buyer_is_maker=false. Bins are half-open floors. Long bin total >= minimum and
buy >= ratio*sell, with zero-sell exception. One maximal adjacent stack per bar.
Move away arms after one completed low > IZ top. Retest touches top, closes >=
top and retains VWAP, confirmed at close and entered next open only.

Long stop is IZ bottom minus buffer*bins; actual entry is adversely slipped,
quantized next-bar open. Risk and 2.5R target are computed only from actual
entry, then quantized; nonpositive risk/unprofitable target is explicit
non-executable. No lookahead; stop first ambiguity; UTC force-flat; one
trade/zone/day. Remove terminal zones immediately: EXPIRED, INVALIDATED,
TRADED, SUPERSEDED. Invalidate close below IZ bottom, VWAP failure and expiry.

## Baseline and exact registry

Baseline: bins=50, min volume=35, VWAP slope=24, ratio=3, stacks=3,
move-away=1, expiry=36, buffer=2, target=2.5, max active zones=3,
max trades/day=1, max trades/zone=1, NEXT_BAR_OPEN_AFTER_CONFIRMED_RETEST.
Fixed: long only, move-away 1, target 2.5, ratio 3, stacks 3, expiry 36,
buffer 2, trade caps. Exactly seven OAT configurations, stable order:
baseline; bins=30; bins=75; min volume=20; min volume=50; VWAP slope=18;
VWAP slope=36. No Cartesian/random/hidden variations.

## Execution, gates and artifacts

Run all configurations over the full six months and report monthly plus fixed
Aug-Oct / Nov-Jan subperiod diagnostics. Enforce funnel equality. Report actual
quantity/notional fees, gross/net PnL/PF/R, cost-risk diagnostics, zone funnel,
long-only trade counts and sample classification. Promotion requires >=48 trades,
five active months, four months with >=4 trades, net PF>1.10, net R>0, net PnL
positive, finite DD, reconciled funnel, concentration gates, and gross/net
reporting. 36-47 is informative but cannot promote. If none pass, set
DEVELOPMENT_EDGE_NOT_FOUND and Alpha NOT_ELIGIBLE_DEVELOPMENT_FAILED. If any
pass, freeze max two via the stated deterministic net-PF/DD/activity/count/
baseline-distance/lexical order, prepare external validation only; do not run
Alpha from development data.

Write immutable V3-root artifacts named in the request: source download,
aggregate/normalized/footprint manifests, baseline YAML, registry, configuration
and comparison/monthly/subperiod reports, trades/zones/events/funnel/cost/gates,
frozen candidates, validation preparation and final report. Include all identity
and content hashes and reject collisions.

Add focused tests for data six-month validation/continuity/no Jan-Jul mutation;
long-only/no short; lifecycle/risk/target/no-lookahead; exact 7 OAT registry;
monthly/promotion logic; V1/V2 preservation; collision and deterministic rerun.
Run focused/full tests, compileall and diff check. Repair then execute the full
study automatically.
