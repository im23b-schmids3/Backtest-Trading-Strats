# ImbalanceVWAPRide.BTC_LONG_ONLY_V5_PRICE_SCALED_BINS

Implement and execute a new standalone post-hoc V5 study. Preserve V1-V4
artifacts, including V4 run `41b00cb85bc1afbd28cbb23b`, without modification,
rerun, deletion, or reinterpretation. V3/V4 are read-only derived comparison
evidence only. Never use raw aggregate rows outside local processing, secrets,
credentials, live orders, unrelated files/downloads, short execution, or any
ValueArea strategy adapter.

## Identity and evidence

- Strategy: `ImbalanceVWAPRide.BTC_LONG_ONLY_V5_PRICE_SCALED_BINS`
- Adapter: `imbalance-vwap-ride-btc-long-only-v5-1`
- Evidence: `POST_HOC_V5_PRICE_SCALED_BIN_RESEARCH`
- `confirmation_evidence: false`, `optimization_claimed: false`, and
  `external_confirmation_required: true`.
- Strict long-only: short setup/trade/PnL diagnostics must all be zero.
- New immutable spec/registry/data/footprint hashes, study ID, and root:
  `research_runs/ImbalanceVWAPRide.BTC_LONG_ONLY_V5_PRICE_SCALED_BINS/<run_id>/`.

## Exact data scope

Only official public Binance USD-M BTCUSDT perpetual aggTrades are allowed.
Phase A: 2023-01 through 2024-01 inclusive. Phase B: 2024-02 through 2024-07
inclusive. Reuse only fully validated local archives/partitions. Download only
missing/invalid authorized months. Every partition requires zip/schema/symbol/
exact coverage/order/duplicate-ID/duplicate-row validation, bounded streaming
normalization, temp output, validated hash, atomic commit, then metadata. No
partial output is accepted. Build separate content-addressed source, normalized,
5m UTC bar, and price-scaled-footprint data for both phases.

Bars have OHLC, total/buy/sell BTC volume, delta, daily VWAP, and UTC session;
daily VWAP resets at 00:00 UTC. `is_buyer_maker=false` is aggressive buy. Enforce
buy+sell=total and buy-sell=delta. Preserve state across batches, row groups,
bar/day/month boundaries. Persist monthly footprint diagnostics including volumes,
bars, footprint rows, bin/sequence rates, conversion, and bin-size distribution.

## Pre-registered scaled bins

At the start of a completed bar, use only the preceding completed bar close.
For the first/unavailable-reference bar produce no footprint zone. Calculate
`bin_size = clamp(20, 100, round_half_up(previous_close*0.001/5)*5)` USD.
One frozen bin size applies to every trade in that bar. Never use current bar
close/high/low or future data; no ATR/volatility or parameter changes. Persist
the source bar bin size on footprint rows, zones, events, and trades.
Bins are floor(price/bin_size)*bin_size. Adjacent qualifying bins differ by the
exact source-bar bin size; never combine adaptive bins across source bars.

## Fixed rules and sealed candidates

Long-only fixed parameters: minimum bin volume 35 BTC; imbalance ratio 3.0;
three stacked bins; move-away 1; expiry 36; stop buffer 2 source-bar bins;
maximum active zones 3; one trade per zone and UTC day; VWAP slope 24;
next-bar entry after retest; stop-first; UTC force-flat. Use existing verified
BTC fees, slippage, quantization, tick, quantity, and immutable reporting.

A qualifying bin is volume >=35 and buy >=3*sell, or zero sell/buy >=35. Merge
each maximal contiguous sequence of >=3 same-bar bins into one non-overlapping
zone. Long regime requires completed close > daily VWAP > daily VWAP 24
completed bars earlier. Arm after low > top; valid retest touches/crosses top,
closes >=top, and keeps regime. Enter only next bar open. Stop is bottom minus
2*source bin size. Risk derives from actual quantized/slipped entry; target is
actual entry plus candidate R*risk and then quantized. Evaluate exits after
entry only; stop wins ties; force-flat at UTC-day end.

Seal and hash exactly these candidates before results, with distinct candidate
configuration hashes and no Cartesian expansion:

| candidate_id | target_r_multiple |
| --- | ---: |
| V5-A-SCALED-BIN-1P5R | 1.5 |
| V5-B-SCALED-BIN-2P0R | 2.0 |
| V5-C-SCALED-BIN-2P5R | 2.5 |

Do not test any other target, bin multiplier, rounding/clamp, slope, threshold,
ratio, stacked bins, stops, trailing/breakeven/partials, or filters.

## Phase A selection

Run only the three candidates exactly once over 2023-01..2024-01. Persist all
required activity, gross/net, target/MFE/MAE, concentration/cost, monthly and
subperiod, trades/zones/events/funnel artifacts. Reconcile:
`proposed = invalid + non_executable + compliance_blocks + executed` and
`stacked_sequences = zones_created`, unless a documented deterministic maximal
merge formula applies.

Eligibility needs all: trades >=52; active months >=10/13; >=8 months with >=4
trades; <=3 zero months; net PnL>0; net PF>1.10; average net R>0; >=4 target
hits; positive hit rate; >=20% trades reach 1R MFE; finite drawdown; valid
funnel/long-only/costs/months/hash; best month <=60%, best three <=85%, best
five <65%; >=3 nonnegative 2023 quarters; both 2023 halves >0; January 2024
reported and <=50% of positive PnL. Do not weaken gates.

If none pass: `PHASE_A_NO_ROBUST_CANDIDATE`, no Phase B/Alpha. If one pass,
freeze it. Multiple pass ranking: nonnegative quarters desc, both halves,
best-month concentration asc, net PF desc, drawdown asc, average net R desc,
target-hit rate desc, trades desc, lower target R, candidate ID. Never PnL only.
Persist gates/ranks/tie trace.

Freeze a selected candidate including scaled formula/rounding/clamp/rules/costs,
registry/config/result hashes and frozen hash; it cannot mutate/reselect.

## Phase B and Alpha

Only the frozen candidate, exactly once, runs 2024-02..2024-07. Pass requires:
>=24 trades, >=5 active months, >=4 months with >=3 trades, net PnL>0, net
PF>1.05, average net R>0, >=2 target hits, >=15% reach 1R MFE, valid frozen
hash/funnel/long-only/cost/month integrity, best month <=70%, best five <75%,
and fixed Feb-Apr/May-Jul diagnostics with one positive and the other no worse
than negative half full positive PnL. Set literal locked status.

Only a Phase B pass with >=24 eligible Phase-B-only trades permits refreshed,
versioned official Alpha 25K Zero and MBT rules and >=20,000 deterministic
one-MBT (0.1 BTC) proxy simulations. Use conservative costs, DST, no weekend,
cutoff force-flat, evaluation/qualified-payout metrics and all required
sensitivity scenarios. Otherwise write `NOT_EXECUTED` with exact reason. It is
always Binance-to-MBT proxy evidence, never confirmation/contract-accurate.

## Gates, reports, execution

Write every root/per-candidate artifact listed in the V5 user contract: study,
registry/history; phase source/normalized/scaled manifests/diagnostics/reports/
gates/selection/freeze; Phase B reports/gates; Alpha reports; preservation and
integrity manifests/final report. Include all identities/hashes/code/evidence/
timestamps and collision protection.

Add focused tests for all literal scaled-bin, integrity, registry, execution,
selection/freeze, Phase B, Alpha, and V1-V4-preservation requirements. Repair
until focused V5 tests, full `tests/research_pipeline`, compileall, and diff
check pass before data/study execution. Execute automatically after gates.
