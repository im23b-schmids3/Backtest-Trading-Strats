# CMEOrderflowAbsorption.ES_V1_PILOT — source/schema comparison

## Sealed scope

- Instrument: CME Globex E-mini S&P 500 futures, concrete raw front-month symbol `ESU6` only.
- Dataset: `GLBX.MDP3`; provider raw symbol: `ESU6`; no NQ, synthetic, back-adjusted, or roll-crossing series.
- Deterministic UTC calendar: 2026-07-20 through 2026-07-31 inclusive. These are the ten weekdays in that range and are outside the anticipated US/CME major holiday closures; they were selected by calendar rule, not outcome.
- No market data has been downloaded, no holdout accessed, and no paid acquisition authorized.

## Official evidence

| Source | Evidence used | Status |
|---|---|---|
| https://databento.com/docs/schemas-and-data-formats/whats-a-schema | A schema defines the record fields and interpretation. | Provided official-current source; identity URL sealed, not independently fetched in this design-only run. |
| https://databento.com/docs/schemas-and-data-formats/mbo | MBO carries individual order events keyed by `order_id`. | Provided official-current source; identity URL sealed. |
| https://databento.com/docs/standards-and-conventions/symbology | Raw symbols and instrument definitions/symbology require confirmation for the requested dataset. | Provided official-current source; identity URL sealed. |
| https://databento.com/docs/reference-historical/basics/ | Historical metadata includes `metadata.get_cost` for a current estimate. | Provided official-current source; identity URL sealed. |

Evidence status: `DOCUMENTED_PROVIDER_EVIDENCE_PENDING_FREE_DEFINITION_AND_SYMBOLOGY_CONFIRMATION`. The required confirmation is free and must precede any paid acquisition.

## Comparison

| Schema option | Fields / aggressor side | Passive replenishment / order identity | Iceberg inference | Volume / cost | Benefits | Limits |
|---|---|---|---|---|---|---|
| `trades` | Executed trades; aggressor interpretation is limited to trade-side fields where supplied and must not be inferred beyond documentation. | No displayed-depth lifecycle; no order identity. | Cannot support a true or probable replenishment inference. | Lowest expected record volume/cost, but not quoted. | Simple executed-volume context. | Cannot observe passive replenishment or queue/order behavior. |
| `trades+tbbo` | Trades plus top-of-book quote context; aggressor classification may be derived only by a predeclared quote-matching rule, with ambiguity retained. | Best-level changes only; no order identity. | Cannot prove a true iceberg or robustly establish replenishment. | More records than trades; cost not quoted. | Adds touch context for execution classification. | No depth beyond touch and no individual order lifecycle. |
| `mbp-10` | Top-ten aggregated price depth plus trade-related records as documented; aggressor remains rule-based/ambiguous. | Can observe aggregated displayed-depth depletion/refill, but not individual orders. | Never proves a true iceberg; at most an aggregate refill pattern. | Higher volume/cost than touch data, unquoted. | Supports depth-at-level and aggregate replenishment features. | Aggregation prevents order identity and makes cancellation/replacement attribution uncertain. |
| `mbo` | Individual order events, including `order_id` per official MBO documentation; trade aggressor labeling still follows documented fields/rules and ambiguity policy. | Best available option for add/modify/cancel/execute sequences and order identity. | A true iceberg is unprovable from this feed design; only a probable/replenishment inference, even with MBO. | Largest expected volume/cost, unquoted. | Technically supports order-lifecycle and replenishment evidence. | Feed semantics, hidden quantity, queue priority, packet loss, and venue behavior still bound conclusions. |

## Decision

Recommended minimum: `mbo`, technically justified because the pilot asks whether visible passive liquidity replenishes against aggressing flow and requires individual order-event identity. It is economically conditional: only proceed after free definition/symbology confirmation and a current zero-cost `metadata.get_cost` estimate demonstrates an approved purchase scope. No dollar, byte, or record estimate is asserted here.

No schema proves a true iceberg. “Probable iceberg/replenishment” is an evidence label, not a fact: repeated displayed liquidity restoration at a level after executions, measured with explicit cancellation/replacement ambiguity. `mbp-10` must never be represented as proof of a true iceberg.
