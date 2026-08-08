# CMEOrderflowAbsorption.ES_V1_PILOT — sealed pilot plan

## Status: PILOT_BLOCKED

The recommended technical minimum is MBO, but the pilot is blocked from sealing a purchase scope because `DATABENTO_API_KEY` was checked for presence only and is absent. No secret was printed. Therefore no current quote, bytes, records, or dollars are asserted.

## Fixed population and source identity

`GLBX.MDP3`, raw symbol `ESU6`, MBO schema, start `2026-07-20T00:00:00Z`, end `2026-08-01T00:00:00Z`, `stype_in=raw_symbol`. The complete weekdays are 2026-07-20, 21, 22, 23, 24, 27, 28, 29, 30, and 31 UTC. This is a deterministic calendar selection outside US/CME major holiday closures, not an outcome selection. It is one front-month raw contract: no roll crossing, synthetic or back-adjusted series, NQ, or data modification.

## Required free gate and non-executed command designs

Before any paid action, obtain and preserve the free Databento definition/instrument and symbology confirmation that `ESU6` resolves as intended in `GLBX.MDP3` for the fixed interval. Record returned identity fields and SHA-256 hashes after that future response exists.

The exact non-executed quote design is:

`databento-historical metadata get-cost --dataset GLBX.MDP3 --schema mbo --symbols ESU6 --stype-in raw_symbol --start 2026-07-20T00:00:00Z --end 2026-08-01T00:00:00Z`

Cost-quote status: `NOT_QUOTED_NO_LOCAL_CREDENTIAL`. If credentials later exist, a zero-cost metadata/get-cost and free definition/symbology query are permitted; record the exact returned quote and all estimate inputs, but do not download. Only after approval may this still-non-executed acquisition design be considered:

`databento-historical timeseries get-range --dataset GLBX.MDP3 --schema mbo --symbols ESU6 --stype-in raw_symbol --start 2026-07-20T00:00:00Z --end 2026-08-01T00:00:00Z --output <approved_immutable_destination>`

## Analysis boundary

Retain pre-RTH data for levels; permit candidate event eligibility in RTH only. Use prior-RTH PDH/PDL and VAH/VAL/POC plus the sealed level universe and normalized features in the companion contracts. This plan has no strategy implementation, tests, backtest, threshold optimization, paid download, live order, or holdout access.

Any future execution implementation is prospectively flat at exactly 22:45:00 UTC, carries no position after that cutoff, and accepts new eligibility at 00:00:00 UTC. It must not alter unrelated strategies. Candidate stops/targets are a non-optimized research framework only; bid/ask, fees, slippage, and queue uncertainty remain explicit.

## Sources and evidence status

- https://databento.com/docs/schemas-and-data-formats/whats-a-schema
- https://databento.com/docs/schemas-and-data-formats/mbo
- https://databento.com/docs/standards-and-conventions/symbology
- https://databento.com/docs/reference-historical/basics/

Evidence status: provider-document URLs are sealed as supplied official current evidence; free definition/symbology confirmation and current cost metadata have not been obtained. MBO documentation establishes individual order events keyed by `order_id`; MBP-10 is top-ten aggregated price depth; historical metadata supplies `metadata.get_cost`. A true iceberg is unprovable, and even MBO supports only a probable/replenishment inference.
