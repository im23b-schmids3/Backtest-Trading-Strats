# CMEOrderflowAbsorption.ES_L2_V1 — independent L2 absorption contract

`CMEOrderflowAbsorption.ES_L2_V1` is an independent Level-2 strategy research
contract. It does not import signal labels, thresholds, weights, or calibration
from any Level-3 strategy. It detects the L2-observable economic signature:
aggressive executions consume displayed liquidity at a structural level,
aggregate displayed depth restores at the same price, and price makes limited
adverse progress.

This package is synthetic-validation only. It contains no data client, DBN
reader, market-data download, historical scan, PnL calculation, or backtest.

## Public source boundary

`MBOToMBP10View` is the only optional historical-source adapter. It uses order
ids privately to reconstruct aggregate book state, then emits only
`MBP10Snapshot` and `MBP10Update`:

- `bid_px`, `bid_sz`, `bid_ct`, and `ask_px`, `ask_sz`, `ask_ct`, each as an
  ordered ten-level tuple (unused levels are `null`);
- timestamp, aggregate book-change side/price/size delta/order-count delta;
- execution price, size, timestamp, and aggressor side.

No strategy-facing class has an `order_id` field. The L2 engine cannot inspect
individual queues, order lineage, native iceberg identity, or MBO IDs.

## Interaction and structural context

The permitted structural levels are prior RTH HIGH/LOW/POC/VAH/VAL and current
RTH HIGH/LOW sweeps. Only an executed trade within ±4 ES ticks opens or extends
an interaction. Passive book activity can enrich an active interaction but can
never start one.

Aggressive selling is relevant for `BUYER_ABSORPTION`; aggressive buying is
relevant for `SELLER_ABSORPTION`. All price-progress and confirmation measures
are direction symmetric.

## Genuine consume → restore cycle

At the defended price and side, the engine records the pre-attack displayed
size/order count. A relevant execution first makes a cycle eligible. A later
same-price snapshot must show reduced displayed depth to establish consumption.
Only a subsequent increase at that exact defended price counts as restoration.
The cycle records pre-size, minimum post-consumption size, cumulative execution
and restored volume, restoration timestamp, latency, order-count restoration,
and fixed 100/250/500/1000ms recovery ratios. A passive addition without the
preceding execution-plus-depth-reduction sequence is not a refill.

## Features

The explainable interaction record includes:

- directional/opposite aggressive volume, imbalance, count, execution rate,
  aggressive-volume rate, and executions at / within one / within two ticks;
- consume/restore cycles, restored volume, latency statistics, cumulative
  consumed/restored volume, restoration-to-consumption ratio, and displayed-
  depth execution ratios;
- maximum/final adverse through-level progress, rejection, adverse progress per
  100 aggressive contracts, and aggressive contracts per adverse tick;
- defended-depth present fraction, time-weighted mean/median, order-count
  restoration, and 1/3/5-level bid/ask depth and imbalance;
- a documented multi-level order-flow imbalance (OFI): for each of the first
  five bid levels, contribute `new_size` if price rises, `new_size-old_size` if
  unchanged, otherwise `-old_size`; for asks contribute `-new_size` if price
  falls, `-(new_size-old_size)` if unchanged, otherwise `old_size`. Sum all
  ten contributions cumulatively. Positive OFI is bid-supporting;
- false-refill guards: unexecuted adds, rapid add→cancel volume/ratio within
  250ms, additions away from the defended price, and continuing adverse price
  progress. These are descriptive false-positive guards, not spoof claims.

## Fixed quality model and qualification

All values use `L2_V1_PREDECLARED_RESEARCH_WEIGHTS`, selected without labels or
outcomes. Component scores are monotonic [0,1] transforms:

| component | weight | inputs |
|---|---:|---|
| aggression | 0.28 | relevant volume/count, directional imbalance, execution-to-initial-depth ratio |
| restoration | 0.25 | cycles, restored/consumed ratio, executed-supported restoration, latency |
| price resistance | 0.22 | inverse maximum adverse progress, interaction rejection |
| persistence | 0.12 | defended-depth presence and time-weighted depth |
| multi-level support | 0.13 | direction-aligned depth imbalance and OFI |
| false-refill penalty | −0.25 | unexecuted additions, rapid cancellation, adverse progress |

The L2 quality score is the weighted component sum minus penalty, clamped to
[0,1]. A setup requires all of: relevant aggressive volume/count, at least one
genuine consume→restore cycle, either limited adverse progress or minimum
rejection, and quality at least 0.55. Explicit reasons are retained for every
rejection. Thresholds live only in `L2Config`.

## Frozen execution layer

After an accepted interaction ends, no confirmation is possible before +5.000s;
exactly +5.000s through exactly +15.000s are eligible. The first causal ES
execution at least +3 favorable ticks from the interaction end confirms. There
is no early adverse invalidation. Entry follows exactly 2ms later.

Long stop is `zone_low - 5 × 0.25`; short stop is `zone_high + 5 × 0.25`.
Target is 3.0R. Risk is $250, with ES-first sizing, MES fallback, and 6 ES / 60
MES caps. This module models no live order placement or historical result.

## Sealed L2 V1 freeze

The current implementation is frozen under
`L2_V1_PREDECLARED_RESEARCH_WEIGHTS` before its first broad historical replay.
The exact component weights are: aggression 0.28, restoration 0.25,
price-resistance 0.22, persistence 0.12, multi-level support 0.13, and
false-refill penalty contribution 0.25. All `L2Config` thresholds, recovery
windows, the plus/minus four-tick vicinity, structural-level family,
consume-to-restore definition, OFI definition, and false-refill guards are
included in that freeze. There are no DEVELOPMENT_ONLY or unspecified L2 V1
parameters.

Frozen execution is: first qualifying ES execution in the inclusive
`[interaction_end + 5.000s, interaction_end + 15.000s]` window at least three
favorable ES ticks from `interaction_end_price`; two milliseconds of latency;
five-tick zone stop; 3.0R target; $250 fixed risk; ES-first then native MES
fallback; and caps of 6 ES and 60 MES.

## Historical source and runner boundary

`historical_runner.py` has a private `PrivateMBORecord` adapter that may retain
MBO `order_id` only while reconstructing aggregate depth. It emits
`PublicBookEvent` objects containing only timestamp, a public MBP-10 snapshot,
aggregate update, and a public execution. The runner asserts that no
strategy-facing model has `order_id`, `resting_order_id`, or `queue_id`.

The first MBO snapshot must begin with `R` and becomes valid only after an `A`
record carrying `F_LAST`. Snapshot records seed the book but cannot generate an
interaction. Any ordinary record before that point, incomplete snapshot, or
ordinary reset fails closed. Completed interactions are consumed by index from
`engine.completed`; historical feature dictionaries are never rebuilt in the
record loop.

The future immutable output directory contains `summary.json`,
`daily-results.csv`, `trade-ledger.csv`, `setup-ledger.csv`,
`interaction-features.csv`, and `diagnostic-report.md`. It must state
`FIRST_BROAD_HISTORICAL_L2_V1_REPLAY` and
`NO_OUTCOME_BASED_PARAMETER_SELECTION_BEFORE_RUN`.

The eventual explicit command is:

```powershell
python -m research_pipeline.cme_orderflow_absorption_l2_v1.historical_runner --session-manifest <ABSOLUTE_SESSION_MANIFEST.json> --output-dir <NEW_ABSOLUTE_OUTPUT_DIRECTORY>
```

The session manifest names every MBO file and exact structural level; the
runner never globs or acquires data. May, June/July, and August retain distinct
evidence labels and are never silently pooled as one OOS result.
