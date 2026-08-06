# FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION

Status: **READY_TO_IMPLEMENT_WITH_PROVENANCE_LIMITATION**. This is a forward-study specification, not a historical reproduction. No historical matrix return is a target, gate, calibration input, or repair reference.

## Scope lock and evidence

The fixed registry is exhaustive: (1) `FIB09-ETH-4H-POST0830`, ETH/4h, entry 0.900, post-TP1 0.830, `min_distance=16`, `min_move=0.0025`; (2) `FIB09-ETH-4H-POST0786-REFERENCE`, ETH/4h, entry 0.900, post-TP1 0.786, `min_distance=16`, `min_move=0.0025`; and (3) `FIB09-BTC-1D-POST0786`, BTC/1d, entry 0.900, post-TP1 0.786, `min_distance=7`, `min_move=0.0025`. ETH and BTC are independent: never pool trades, gates, ranking, selection, capital, or evidence. No other asset, timeframe, ratio, policy, threshold, target structure, grid, candidate, or alternative is permitted.

Forensic sources read before sealing: `forensic-reconstruction-report.md`, `reconstruction-source-index.json`, `matrix-evidence-manifest.json`, `historical-row-registry.json`, and `matrix-anchor-comparison.json` under `docs/research_pipeline/fib_reproduction_v1/`. V6/V6.5 matrices are owner-supplied, Git-pinned evidence of reported results only. They do not prove historical execution or data provenance.

Classification vocabulary: `PROVEN_BY_CODE` means direct current source/config/test evidence; `PROVEN_BY_MATRIX` means only the fixed row's current owner-supplied reported setting; `PROSPECTIVE_V1_DECISION` means a forward-only locked contract; `UNRESOLVED_BLOCKING` means implementation/execution cannot proceed.

## Rule classification and sealed contract

| Rule | Class | Locked definition |
|---|---|---|
| Registry settings | PROVEN_BY_MATRIX | Exactly the three settings above; matrix evidence is not a performance gate. |
| Directional setup and causal swings | PROVEN_BY_CODE | Use `active_wick_lifecycle`: long starts from a low and favorable high; short from a high and favorable low; a pullback candidate is promoted only on a next favorable extreme after the fixed distance and percentage move. Events occur at bar close and act from the following bar. |
| Fib orientation | PROVEN_BY_CODE | Range must satisfy `low < high`. Long price(r)=`high-r*(high-low)`; short price(r)=`low+r*(high-low)`. Long 0=high/1=low; short 0=low/1=high. |
| Entry and stop | PROVEN_BY_CODE | Limit is price(0.900); initial stop is price(1.020). Long touch `low<=limit`; short touch `high>=limit`; entry fill applies adverse slippage. |
| Activation | PROVEN_BY_CODE | A closed-bar activation/update submits a versioned order for the next bar only; it cannot fill on confirmation bar. Updates replace an unfilled order. |
| Targets/fractions | PROVEN_BY_CODE | Profile B: TP1..TP5 ratios `(0.786,0.618,0.500,0.236,0.050)` and fractions `(0.30,0.25,0.20,0.15,0.10)`. |
| Post-TP1 stop | PROVEN_BY_CODE | After TP1 fill, set stop to price(candidate post-TP1 ratio); effective only next candle. |
| Same-bar ordering | PROVEN_BY_CODE | Conservative: current stop is evaluated before any reachable target; stop wins. A TP1-moved stop is never retroactive to TP1's bar. |
| Entry expiry/time stop | PROSPECTIVE_V1_DECISION | `entry_max_age_bars=null`: no arbitrary bar-count expiry. An unfilled setup expires only upon anchor break, anchor-age invalidation (ETH 60 days; BTC 180 days), replacement, active-position blocking, session/data end, or data-contract block. There is no in-position time stop. This is the single forward-only expiry decision, chosen because it is deterministic and avoids adding a parameter. |
| End of data | PROVEN_BY_CODE | Force-close any active remainder at final available close with adverse exit slippage and fee; unresolved orders receive `SESSION_OR_DATA_END`. |
| Concurrency | PROSPECTIVE_V1_DECISION | One independent position maximum per candidate contract; no shared portfolio. While active, all same-contract unfilled orders terminalize as `ACTIVE_POSITION_BLOCKED`; opposite pending order is cancelled at fill. |
| Costs, quantity, accounting | PROSPECTIVE_V1_DECISION | Contract must declare venue, instrument type, precision, fees, slippage unit, and rounding before data acceptance. Use adverse fill: long entry/short exit `raw*(1+s)`, short entry/long exit `raw*(1-s)`; fee=`abs(q*fill)*f` at entry and every exit. Quantity is floor-to-contract step; a zero quantity is `STOP_DISTANCE_REJECTED`. Risk budget is 2% of immediately prior independent candidate equity; `q=floor_step(budget/abs(entryFill-initialStop))`; entry fee debits cash, exit PnL is direction*(exitFill-entryFill)*q-exitFee, and equity compounds after every terminal trade. |

Data-contract supersession (2026-08-06): the exact hash-bound files recorded in `docs/research_pipeline/fib_prospective_v1/data-contracts/ETH_4H/manifest.json` and `docs/research_pipeline/fib_prospective_v1/data-contracts/BTC_1D/manifest.json` are accepted for this prospective study only, based on hash-bound identity, owner-attested symbol/timeframe, independently verified cadence, UTC chronology, and OHLCV integrity. Source exchange and spot/perpetual classification are **UNKNOWN**. Results apply only to these exact hash-bound files; no transferability to another venue or instrument type and no historical V6/V6.5 reproduction claim are made. The unknown provenance does not block this prospective study for that stated limited purpose. The source's Binance BTC/USDT and ETH/USDT defaults (0.001 fee, 0.0002 slippage) remain current-code evidence, are not established by these manifests, and are not silently adopted.

## Deterministic lifecycle and identifiers

Canonical JSON serialization (UTF-8, sorted keys, no whitespace) is hashed SHA-256; IDs are `prefix + first 16 lowercase hex`: `setup_id=setup-` hash(candidate,side,anchor timestamp,anchor price); `impulse_id=impulse-` hash(setup_id,extreme timestamp,extreme price); `fib_range_id=fib-` hash(impulse_id,low,high); `order_id=order-` hash(fib_range_id,version,active timestamp); `trade_id=trade-` hash(order_id,entry fill timestamp); `exit_leg_id=exit-` hash(trade_id,leg ordinal); `event_id=event-` hash(candidate,timestamp,event kind,subject ID,ordinal).

Each setup must have exactly one terminal disposition, selected from `TRADE_EXECUTED`, `DIRECTION_REJECTED`, `IMPULSE_NOT_CONFIRMED`, `FIB_RANGE_INVALID`, `ENTRY_NOT_REACHED`, `ENTRY_EXPIRED`, `STOP_DISTANCE_REJECTED`, `ACTIVE_POSITION_BLOCKED`, `SESSION_OR_DATA_END`, and `DATA_CONTRACT_BLOCKED`. `TRADE_EXECUTED` terminalizes the setup on entry, even though its trade later closes. No outcome may be omitted or terminalized twice.

Pseudocode:

```
for each closed UTC bar in one candidate contract:
  update causal long and short lifecycle; reject invalid direction/range/impulse
  submit/replace eligible order for next bar only
  for active orders: invalidate/expire first; if limit touched, validate distance and capacity; fill once
  for position: current stop first; otherwise eligible TP legs in TP1..TP5 order; TP1 move applies next bar
at immutable boundary: force-close position; terminalize pending setups
assert one terminal setup outcome and all reconciliation identities
```

## Chronology, gates, and anti-overfitting

The sealed chronology contract is `docs/research_pipeline/fib_prospective_v1/chronology-manifest.json` (canonical self-hash `bb910f85e271a5f436c6203f279a3ef0d4bc4119344e409fdeefec1dd01e5794`) and is required for every development invocation. It fixes identical UTC boundaries for both assets: development is `timestamp >= 2022-01-01T00:00:00+00:00` and `< 2025-01-01T00:00:00+00:00`; holdout is `timestamp >= 2025-01-01T00:00:00+00:00`, with no exclusive end. The holdout is unopened: no reading, derived artifacts, metrics, diagnostics, or execution before development gates pass. No date may be chosen from returns; split is fixed from contract coverage before development execution.

Development gates, independently per candidate: positive net after costs; PF >=1.30; positive average net R; max DD <=20%; positive with one extra conservative slippage unit each entry/exit; positive after best-trade removal; full reconciliation. Evidence labels: `<30` `LOW_FREQUENCY_DEVELOPMENT_EVIDENCE`, `30-59` `MODERATE_DEVELOPMENT_EVIDENCE`, `>=60` `FULL_DEVELOPMENT_GATE_ELIGIBILITY`. Holdout labels: `<15` `INSUFFICIENT_HOLDOUT_SAMPLE`, `15-29` `PRELIMINARY_LOW_FREQUENCY_EVIDENCE`, `>=30` `FULL_HOLDOUT_GATE_ELIGIBILITY`.

Anti-overfitting rules (verbatim from the brief): **“no holdout access before development gates pass”** and **“no historical-matrix repair.”** Also: no changes to the fixed registry, no return-targeting, no pooled ETH/BTC gate, no replacement feed/proxy/resampling, no adaptive chronology, and failure does not authorize a new candidate.

## Reconciliation, artifacts, tests, and implementation checklist

Reconcile setup count to exactly one terminal outcome; orders to submitted/replaced/cancelled/filled; each fill to one trade; `initial_quantity = sum(partial exits)+remaining forced exit`; each leg's raw/fill price, fee and slippage; trade gross/net PnL; cash/equity transitions; and final compounded equity to opening equity plus realized net PnL. Reconciliation failure fails gates.

Future artifact layout is exactly: `sealed-specification.json`, `candidate-registry.json`, `evidence-classification.json`, `chronology-manifest.json`, `data-manifest.json`, `execution-assumptions.json`, `events.json`, `setup-outcomes.json`, `orders.json`, `trades.json`, `partial-exits.json`, `monthly-metrics.json`, `report.json`, `gates.json`, `freeze.json`, `integrity-manifest.json`, `final-report.json`.

Required tests before any permitted future execution: immutable data hash/manifest validation; UTC/schema/duplicate/gap rejection; ID determinism; next-bar non-lookahead; long/short orientation; limit-touch and adverse costs; stop-over-target; TP1 delayed stop; each partial fraction and rounding; expiry/end-of-data; one-terminal-outcome; independent ETH/BTC accounting; all reconciliation identities; chronology lock; holdout-access refusal; and integrity hashes.

Implementation checklist: accept immutable contracts for both required timeframes; write and hash the exact listed artifacts; lock split without opening holdout; implement only this sealed contract; run development only if unblocked; require all gates and reconciliation; only then unseal holdout; publish final report without changing any lock. Current blocker prevents every execution step.
