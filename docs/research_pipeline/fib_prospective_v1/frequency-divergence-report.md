# FibRetracementContinuation prospective V1 frequency-divergence forensic audit

## Scope, controls, and conclusion

This is a read-only aggregation of completed development artifact tree `research_runs/FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION/development/run-20260807-01`, committed V1 lifecycle code, and the forensic reconstruction evidence in `docs/research_pipeline/fib_reproduction_v1`. No holdout row or result was opened, and no backtest was started or rerun. The only modified file is this report.

Conclusion: `PROSPECTIVE_V1_FREQUENCY_CORRECT_HISTORICALLY_DIFFERENT` is supported. The completed artifacts reconcile to the committed V1 lifecycle. They prove that V1 deliberately emits a setup for each qualifying directional anchor-to-extreme condition and does not apply a rate/cooldown/deduplication gate. They do not prove that a particular historical frequency filter existed or caused the historical matrix counts. The historical matrices are owner-supplied comparative evidence, not authenticated execution provenance.

Evidence labels: PROVEN means completed-artifact or committed-code evidence; USER_SUPPLIED means a row in the supplied V6/V6.5 matrices; UNRESOLVED means the available evidence cannot decide it.

## Exact development funnel

Bars evaluated are the development manifest row counts: ETH 4H 6,576 and BTC 1D 1,096, from 2022-01-01 through 2025-01-01. `directional opportunities` is the exact terminal-outcome population, one per emitted V1 setup. `impulses proposed` and `impulses confirmed` have no separately persisted stage: `causal_setups` immediately appends the setup when the directional distance and move predicates pass (`strategy.py:29-34`). Therefore both equal directional opportunities by the same deterministic lifecycle count. Two ETH and two BTC end-of-data setups have `SESSION_OR_DATA_END` before the runner can create a next-bar order; they do not have an order-persisted fib-range ID. All remaining setup-to-order transitions are one-to-one.

`unique fib ranges` is the exact number of distinct `fib_range_id` values in `orders.json`; all are unique. `activated orders` is exact orders with a non-null `active_timestamp`; it equals activated/submitted orders because `submit_order` always assigns the next bar (`runner.py:40`). `reached entries`, `filled orders`, and `executed` have no separate artifact stages in this run: no rejection disposition occurred, `execute_order` returns a trade only after a touch, and every `ORDER_FILLED` has one trade and `TRADE_EXECUTED` outcome (`runner.py:47-49`, `reconciliation.py:33-45`). Consequently their same deterministic lifecycle count is proven by events, trades, and outcomes.

| Candidate | bars evaluated | directional opportunities | impulses proposed | impulses confirmed | unique fib ranges | activated orders | reached entries | filled orders | active blocked | expired | end of data | executed | long/short trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIB09-ETH-4H-POST0830 | 6,576 | 11,641 | 11,641 | 11,641 | 11,639 | 11,639 | 951 | 951 | 7,204 | 3,482 | 4 | 951 | 479 / 472 |
| FIB09-ETH-4H-POST0786-REFERENCE | 6,576 | 11,641 | 11,641 | 11,641 | 11,639 | 11,639 | 963 | 963 | 7,179 | 3,497 | 2 | 963 | 487 / 476 |
| FIB09-BTC-1D-POST0786 | 1,096 | 2,192 | 2,192 | 2,192 | 2,190 | 2,190 | 173 | 173 | 1,359 | 657 | 3 | 173 | 75 / 98 |

For every row, `active blocked + expired + end of data + executed = directional opportunities`; for example, ETH .830 is `7,204 + 3,482 + 4 + 951 = 11,641`. Reconciliation is true for all three candidates. This proves 951, 963, and 173 are not accidental event counts: each also has exactly that many filled-order events, trade records, and `TRADE_EXECUTED` outcomes.

## Yearly count and density aggregation

Yearly proposed and unique-range counts use `ORDER_SUBMITTED` event timestamps; executed uses `ORDER_FILLED` timestamps. This is the only consistent year attribution across the lifecycle. Days are 365, 365, and 366; ETH bars are 2,190, 2,190, and 2,196; BTC bars are 365, 365, and 366. Density format is proposed per bar / per day / per year, then executed per bar / per day / per year.

| Candidate | 2022 proposed/ranges/executed | 2023 proposed/ranges/executed | 2024 proposed/ranges/executed | proposed density bar/day/year | executed density bar/day/year |
| --- | --- | --- | --- | --- | --- |
| ETH .830 | 4,046 / 4,046 / 335 | 3,264 / 3,264 / 276 | 4,329 / 4,329 / 340 | 1.7702 / 10.6211 / 3,879.6667 | 0.1446 / 0.8676 / 317.0000 |
| ETH .786 | 4,046 / 4,046 / 335 | 3,264 / 3,264 / 282 | 4,329 / 4,329 / 346 | 1.7702 / 10.6211 / 3,879.6667 | 0.1464 / 0.8785 / 321.0000 |
| BTC .786 | 730 / 730 / 57 | 730 / 730 / 57 | 730 / 730 / 59 | 1.9982 / 1.9973 / 730.0000 | 0.1578 / 0.1577 / 57.6667 |

The table totals equal the exact activated orders and executed counts above.

## Anchor, extreme, pair, and impulse-family reuse

An anchor family groups orders by `anchor_timestamp`; an impulse family groups by directional anchor-to-extreme pair. Pair identity includes direction, anchor timestamp, and extreme timestamp. Exact average/max values are orders per anchor family or orders per pair family. There is exactly one order per pair and one unique impulse/range per order, so pair-family average and maximum are 1.000000 and 1.

| Candidate | unique anchors | unique extremes | unique pairs | avg/max setups per anchor | avg/max setups per impulse family | long: anchors/extremes/pairs, avg/max anchor | short: anchors/extremes/pairs, avg/max anchor |
| --- | ---: | ---: | ---: | --- | --- | --- |
| ETH .830 | 5,756 | 6,093 | 11,639 | 2.022064 / 4 | 1.000000 / 1 | 4,443 / 5,829 / 5,829, 1.311951 / 2 | 4,481 / 5,810 / 5,810, 1.296586 / 2 |
| ETH .786 | 5,756 | 6,093 | 11,639 | 2.022064 / 4 | 1.000000 / 1 | 4,443 / 5,829 / 5,829, 1.311951 / 2 | 4,481 / 5,810 / 5,810, 1.296586 / 2 |
| BTC .786 | 1,046 | 1,095 | 2,190 | 2.093690 / 4 | 1.000000 / 1 | 834 / 1,095 / 1,095, 1.312950 / 2 | 820 / 1,095 / 1,095, 1.335366 / 2 |

## Explicit lifecycle answers

| Question | TRUE/FALSE | Evidence-bound answer |
| --- | --- | --- |
| Does V1 generate a setup on every qualifying anchor-to-new-extreme pair? | TRUE | Each qualifying LONG and SHORT condition appends `create_setup` with no cooldown or duplicate gate (`strategy.py:29-34`). The emitted population exactly reconciles. |
| Can one anchor be reused for later qualifying extremes? | TRUE | The relevant anchor is reset to the qualifying extreme, permitting later directional reuse; artifact maxima are four orders per anchor family. |
| Can an extreme timestamp serve two directional setups? | TRUE | LONG and SHORT are evaluated independently on each bar; extreme multiplicity reaches two. |
| Can an impulse ID or fib-range ID be reused? | FALSE | Each persisted ID is unique: 11,639 ETH / 2,190 BTC order IDs, impulse IDs, and fib-range IDs. |
| Is every confirmed order next-bar activated? | TRUE | All persisted orders have an `active_timestamp`, assigned as `bars[index+1]` by `runner.py:40`. |
| Is an entry activated or filled on its extreme bar? | FALSE | Activation is next bar, and earlier execution is skipped (`runner.py:40-42`). |
| Can more than one pending order exist? | TRUE | `pending` is a list and is iterated after every submission (`runner.py:36,40-45`). |
| Can overlapping impulses exist? | TRUE | `causal_setups` emits each qualifying direction independently and the runner has no impulse-overlap exclusion; multiple pending orders are explicitly permitted. |
| Can nested impulses exist? | TRUE (permitted, not separately labelled) | V1 does not define or reject a nested-impulse relationship. A qualifying new anchor-to-extreme pair is emitted regardless of whether another pending lifecycle overlaps its interval. |
| Can repeated setups originate from the same anchor? | TRUE | Reused-anchor statistics show up to four orders per anchor family; the anchor-to-extreme identity makes each later impulse distinct. |
| Can a repeated setup be created when a new extreme occurs? | TRUE | The extreme coordinates are part of the deterministic setup identity and `causal_setups` appends each qualifying new directional condition. |
| Can simultaneous opposite-direction setup families exist? | TRUE | LONG and SHORT conditions are evaluated independently in the same bar loop (`strategy.py:29-34`); the only position constraint applies after proposal. |
| Can immediate re-entry occur after a trade closes? | FALSE on the closing bar; TRUE on the next eligible bar | Pending orders are evaluated before the active trade is processed and are blocked while `active` is non-null; the next bar has no cooldown gate. |
| Can multiple valid Fib ranges exist before prior ranges expire? | TRUE | The `pending` list can contain multiple orders with distinct fib-range IDs until they fill, expire, or are active-position blocked. |
| Can a setup regenerate from an already-used swing? | TRUE | There is no consumed-anchor or consumed-swing registry. Reused anchor families are observed directly in the completed orders. |
| Can more than one active position exist? | FALSE | `active` is scalar; pending orders are terminally blocked while it exists (`runner.py:36,44`). |
| Does active-position blocking occur after proposal/submission? | TRUE | Submission precedes the pending-order disposition loop (`runner.py:40-45`). |
| Can an entry survive age/range invalidation? | FALSE | `expire_reason` returns `ENTRY_EXPIRED` for age or range breach (`strategy.py:37-40`). |
| Are 951 and 963 ETH counts internally correct? | TRUE | For .830/.786 respectively, the completed artifacts contain exactly 951/963 fills, trades, and `TRADE_EXECUTED` outcomes, with reconciliation true. |
| Is the 173 BTC count internally correct? | TRUE | The completed BTC artifacts contain exactly 173 fills, trades, and `TRADE_EXECUTED` outcomes, with reconciliation true. |

## Limited historical matrix comparison

This comparison is deliberately limited to the supplied V6/V6.5 matrix rows. It establishes only their reported parameters and counts, not historical execution, chronology, data identity, or a causal frequency rule. The matrix manifest and reconstruction report state those bindings are absent.

| Matching row | supplied matrix parameters | supplied trades | V1 executed | What can be concluded |
| --- | --- | ---: | ---: | --- |
| ETH POST0786 | d16, m.0025, policy C, Fib .786 | 56 | 963 | V1 count is 17.1964 times the supplied total; historical mechanism is UNRESOLVED. |
| ETH POST0830 | d16, m.0025, policy C, Fib .830 | 56 | 951 | V1 count is 16.9821 times the supplied total; historical mechanism is UNRESOLVED. |
| BTC POST0786 | d7, m.0025, policy C, Fib .786 | 37 | 173 | V1 count is 4.6757 times the supplied total; historical mechanism is UNRESOLVED. |

`min_distance`, `min_move`, and one-position concurrency are PROVEN current V6/V6.5 matrix parameters. Minimum-trade thresholds, penalties, ranking, policy selection, and Gold exclusion are PROVEN matrix-selection controls, not in-run V1 lifecycle gates. A historical cooldown, deduplication rule, maximum setup rate, chronology/data segmentation, or entry-time filter remains UNRESOLVED because no authenticated historical trade/order log or invocation is available.

| Frequency-control field or rule | Required evidence classification | Audit finding |
| --- | --- | --- |
| `min_distance` and `min_move` | PROVEN_BY_MATRIX | Present in the supplied selected rows; current V1 uses the same d16/d7 and .0025 values. Historical execution use is not authenticated. |
| `policy`, `stop_policy`, and post-TP1 stop ratio | PROVEN_BY_MATRIX | Present in the supplied matrices. They affect selection or post-fill handling, not V1 proposal emission. |
| swing parameters, swing deduplication, consumed-range policy | UNKNOWN_HISTORICAL_BEHAVIOR | No authenticated historical code or trade/order artifact proves such a control. |
| entry filters, early return, late return, eligibility fields | UNKNOWN_HISTORICAL_BEHAVIOR | No supplied matrix field or authenticated historical execution record binds one to the selected rows. |
| one-position concurrency | PROVEN_BY_MATRIX | The supplied configuration identifies max positions one; V1's scalar active position implements that same prospective constraint. |
| minimum trades, penalties, robust rank, and policy selection | PROVEN_BY_HISTORICAL_CODE | The committed V6/V6.5 generator/ranking code proves these are ranking controls. It does not prove the original historical run used that exact code. |
| cooldown, one-trade-per-impulse, setup-rate cap, re-entry suppression | UNKNOWN_HISTORICAL_BEHAVIOR | These controls are absent from V1 and not established by authenticated historical evidence. |

## A-E conclusions and ranked causes

| Answer | TRUE/FALSE | Exact answer |
| --- | --- | --- |
| A. Is the approximately 17x ETH difference primarily a missing historical frequency filter? | FALSE | A missing historical filter is plausible but not proven; the evidence proves V1's dense lifecycle and does not authenticate the historical lifecycle. |
| B. Is the approximately 5x BTC difference the same historical-filter mechanism? | FALSE | V1 has the same dense setup-generation mechanism, but the historical BTC mechanism is likewise unproven. |
| C. Does V1 generate a setup on every qualifying anchor-to-new-extreme pair? | TRUE | See the direct code and reconciled setup population above. |
| D. Are the 951/963 ETH counts internally correct? | TRUE | See three-way fill/trade/outcome equality and reconciliation above. |
| E. Is the 173 BTC count internally correct? | TRUE | See three-way fill/trade/outcome equality and reconciliation above. |

| Rank | Cause | Evidence-bound finding |
| --- | --- | --- |
| CRITICAL | V1 causal anchor-to-extreme emission | PROVEN: every qualifying directional condition emits a lifecycle; repeated anchors are allowed. This is the direct source of V1's high proposed population. |
| HIGH | Historical provenance gap | PROVEN gap: supplied matrices lack execution invocation, data identity, chronology, and order/trade logs. It prevents assigning the difference to a historical filter. |
| MEDIUM | No V1 rate, cooldown, or duplicate suppression | PROVEN for V1 and UNRESOLVED historically. It explains why V1 does not reduce the emitted population before submission. |
| LOW | Post-TP1 stop placement | PROVEN post-fill behavior only. It can explain the 12 ETH execution difference between .830 and .786, not the proposal population. |

## Recommendation

Keep prospective V1 frozen as an internally reconciled but historically different development result. Do not tune, abandon, or reinterpret it merely because the historical fill counts are lower. A separately sealed V2 is appropriate only if authenticated or otherwise recoverable evidence establishes a specific historical frequency-control rule and its exact semantics without using performance-driven parameter selection. Otherwise the correct action is to retain the classification above and treat historical mechanism as unresolved.

## Validation record

`data-manifest.json`, `chronology-manifest.json`, `freeze.json`, `development-result.json`, and `final-report.json` record `LOCKED_NOT_OPENED` holdout status. No holdout path was accessed. No production backtest was started. No code, strategy behavior, data, artifacts, manifests, or settings were changed. `git diff --check` passed after this report correction.
