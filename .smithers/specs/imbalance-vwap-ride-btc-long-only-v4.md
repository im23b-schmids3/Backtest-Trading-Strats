# ImbalanceVWAPRide.BTC_LONG_ONLY_V4_CANDIDATE_SELECTION

Build and execute a new, standalone, post-hoc research study. Preserve every V1,
V2, and V3 artifact without modification, rerun, reinterpretation, or deletion.
V3 (`ImbalanceVWAPRide.BTC_LONG_ONLY_V3_EXPLORATORY`, run
`cce6efb5e35004df8dd8b752`) is read-only context only.

## Identity and evidence

- Strategy: `ImbalanceVWAPRide.BTC_LONG_ONLY_V4_CANDIDATE_SELECTION`
- Adapter: `imbalance-vwap-ride-btc-long-only-v4-1`
- Selection evidence: `POST_HOC_V4_RETROSPECTIVE_CANDIDATE_SELECTION`
- Locked evidence: `STRATEGY_SPECIFIC_TEMPORAL_LOCKED_TEST`
- `confirmation_evidence: false`, `optimization_claimed: false`, and
  `requires_external_live_or_contract_accurate_confirmation: true`.
- Strict LONG_ONLY. Diagnostics reconcile zero short setups, trades, and PnL.
- No secrets, credentials, raw aggregate-row disclosure, live orders, unrelated
  downloads/files, ValueAreaTrap, ValueAreaAcceptance, or short execution.

Use new V4 specification/candidate/data hashes, deterministic study ID, and
immutable artifacts under
`research_runs/ImbalanceVWAPRide.BTC_LONG_ONLY_V4_CANDIDATE_SELECTION/<run_id>/`.
Prevent collisions and resume a healthy V4 run instead of creating duplicates.

## Data scope and integrity

Only official public Binance USD-M BTCUSDT perpetual aggTrades are in scope.
Phase A is exactly 2023-01..2023-12; Phase B is exactly 2025-02..2025-07.
Reuse only hash-verified archives/partitions; download only missing or invalid
months. For each month: validate zip/schema/symbol/month coverage/order,
duplicate IDs/rows, streamed normalization, temporary write, output validation,
hash, atomic commit, then `partition.json`. Never accept temporary output.
Persist all source and partition lineage. Create separately content-addressed
Phase A and B aggregate hashes/manifests and 5-minute bar/footprint hashes.

Streaming bars are UTC 5m OHLC, total/buy/sell BTC volume, delta, cumulative
session delta where supplied, daily VWAP, and UTC session. `is_buyer_maker=false`
means aggressive buy. Daily VWAP resets at 00:00 UTC and no calculation may use
future/uncompleted-bar data. Preserve state across chunks/month/day boundaries.

## Fixed strategy

All candidates: 50 USD price bins; `min_bin_volume_btc=35`, ratio 3.0, three
stacked contiguous bins, one move-away bar, expiry 36 bars, stop buffer two
bins, maximum three zones, one trade/zone and UTC day, next-bar-open after a
confirmed retest, STOP_FIRST ambiguity, and force-flat on UTC day end. Use
existing verified BTC tick, quantity step/quantization, adverse slippage and
fee model.

A long zone is a maximal non-overlapping contiguous run of at least three bins
with total >=35 and buy >=3*sell (or sell zero and buy >=35). It arms after a
completed low above zone top. A valid retest crosses/touches top and closes at
or above top while close > daily VWAP and daily VWAP > N completed bars earlier.
Enter only next-bar open. Stop is bottom minus two bins. Compute risk from actual
quantized/slipped entry; target is entry plus R*risk, then quantize. Reject
nonpositive risk, invalid structure, quantity, or missing next bar. Exit checks
begin after entry; stop wins ties. Remove zones when expired, VWAP-invalid,
close below bottom, traded, or superseded.

## Sealed registry

Persist and hash these exactly before Phase A results; never expand, mutate, or
Cartesian-search them:

| candidate_id | vwap_slope_bars | target_r_multiple |
| --- | ---: | ---: |
| V4-A-BASELINE-2P5R | 24 | 2.5 |
| V4-B-BASELINE-3P0R | 24 | 3.0 |
| V4-C-BASELINE-3P5R | 24 | 3.5 |
| V4-D-SLOW-VWAP-2P5R | 36 | 2.5 |

## Phase A selection

Execute all four exactly once on 2023. Persist activity, gross/net/cost/risk,
concentration, monthly/quarterly/half-year, long-only, zones/events/trades, and
the funnel. Enforce exactly:
`proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades`.

A candidate passes only if all are true: >=72 trades; >=10 active months; >=8
months with >=4 trades; <=2 zero months; net PnL >0; net PF >1.10; mean net R
>0; finite/reported drawdown; reconciliations valid; best-month <=60%, best-3
months <=85%, best-5 trades <65% positive PnL; >=3 nonnegative quarters; both
half-years positive; all monthly/gross/net/cost outputs valid.

If zero pass, set `PHASE_A_NO_ROBUST_CANDIDATE`, finalize reports, and never
open Phase B/Alpha. If one passes select it. If many pass rank in exact order:
positive quarters descending, both halves positive, lower best-month
concentration, higher net PF, lower drawdown, higher average net R, more trades,
higher net PnL, nearest baseline, lexicographically smallest ID. Persist full
rank trace with `PRE_REGISTERED_ROBUSTNESS_RANKING`, never PnL-only.

Freeze one selected candidate immutably with every rule/cost/slippage/quantity
assumption plus registry/result and frozen configuration hashes. Phase B must
match that hash and cannot rerun/replace the selection.

## Phase B and Alpha

Only a committed frozen candidate runs exactly once on 2025-02..2025-07. Pass
requires >=30 trades, >=5 active months, four months with >=4 trades, net PnL
>0, net PF >1.05, mean net R >0, finite drawdown, valid reconciliations/hash/
costs/months, best month <=70%, best five trades <75%, and fixed Feb-Apr and
May-Jul diagnostics with at least one positive and the other not worse than
negative half full positive PnL. Set a literal LOCKED_TEST status. This is not
confirmation evidence.

Run Alpha only after `LOCKED_TEST_PASSED`, exact one Phase B execution, frozen
artifact, >=30 proxy-eligible Phase B trades, and freshly retrieved consistent
official Alpha Futures 25K Zero rules. Model Phase-B-only signals as a clearly
limited one-MBT (0.1 BTC) proxy with versioned public MBT contract assumptions,
conservative costs, Eastern DST, no weekends/unavailable CME periods, force-flat
cutoff, >=20,000 deterministic chronological and daily-block bootstrap paths,
and required evaluation/qualified/payout/sensitivity reports. Otherwise write
`NOT_EXECUTED` with reason. Never claim confirmation or contract-accurate fills.

## Verification and artifacts

Implement all required source/data/bar, candidate, Phase A selection/freeze,
Phase B locked-test, Alpha, preservation, and final immutable artifacts with
identity, hashes, code version, evidence labels, timestamps, and collision
protection. Add focused tests for acquisition, bounded memory, bars, strategy,
exact registry, selection, freeze, Phase B isolation, Alpha eligibility, all
funnel equations, immutability, and no raw disclosure. Before real execution,
repair until focused tests, `tests/research_pipeline`, `python -m compileall
src/research_pipeline`, and `git diff --check` pass. Then execute automatically.
