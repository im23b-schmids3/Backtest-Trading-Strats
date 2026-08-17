# L2 V3 POC-only predeclared validation plan

## Freeze

- Strategy: `CMEOrderflowAbsorption.ES_L2_V3_POC_ONLY`
- Parent: `CMEOrderflowAbsorption.ES_L2_V2`
- Contract hash: `a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4`.
- Evidence label: `POST_HOC_POC_ONLY_PREDECLARED_FUTURE_VALIDATION_NOT_VALIDATED`

V3 has exactly one strategy-semantic change: the eligible structural-level set
is `[PRIOR_RTH_POC]`. V2's seven structural levels are otherwise unchanged in
the underlying level engine, but they must not be passed to the V3 interaction
engine. The machine-readable `l2-v3-poc-only-contract-diff.json` and
`v2_to_v3_contract_diff()` prove that the V2
configuration, interaction features, score weights, confirmation, entries,
stops, targets, sizing, caps and one-position logic are inherited unchanged.

The retained V2 configuration includes minimum quality `0.50`, relevant
aggressive volume `50`, execution count `2`, consume/restore cycles `1`, and
rejection ticks `0.25`. Score weights remain aggression `0.28`, restoration
`0.25`, price resistance `0.22`, persistence `0.12`, multi-level support
`0.13`, and false-refill penalty `0.25`. The execution contract remains ±4
ticks, first ES execution reaching at least +3 favorable ticks in inclusive
`[interaction_end + 5s, interaction_end + 15s]`, 2 ms latency, a 5-tick zone
stop, 3R target, $250 fixed risk, ES-first/MES fallback with caps 6/60, and one
position. No early confirmation invalidation is introduced.

## Post-hoc context, not validation

POC-only is motivated post hoc. It is not validated and may not be treated as
a selected live rule. The only recorded context is: May 6 POC trades `+4.4091R`
(development), June/July eight POC trades `+2.6756R`, zero unresolved
(retrospective robustness), and August one accepted POC setup with zero
confirmations/trades (seen data). These periods have distinct labels and do
not support an out-of-sample performance claim.

## Native L2 acquisition requirement

The future V3 source package contains no ES MBO:

1. ES `GLBX.MDP3` `mbp-10`, `[13:00:00, 22:45:01)` UTC per test RTH: native
   aggregate top-ten book plus trade/execution observations for the signal,
   confirmation, ES entry, exits and hard-flat monitoring.
2. MES `GLBX.MDP3` `mbp-1`, `[13:30:00, 22:45:01)` UTC only: native fallback
   execution monitoring.
3. ES `GLBX.MDP3` `trades`, `[13:30:00, 20:00:00)` UTC for each immediately
   prior completed RTH: POC profile construction only. Do not acquire
   HIGH/LOW/VAH/VAL-specific data or a separate ES MBP-1/TBBO/trades path.

MBP-10 is technically sufficient to replace MBO for this V3 aggregate-L2
contract because it supplies top-ten aggregate price/size/order-count fields
and trade records, without order identities. A future native adapter must
normalize fixed-point prices once, map native trade side to aggressor, and fail
closed unless a valid two-sided top-ten book is observed before 13:30 UTC. The
13:00 UTC request is the minimum safe choice among 13:00/13:15/13:20/13:30: it
provides a 30-minute causal book warm-up while preserving zero pre-RTH signal
eligibility. This is an acquisition integrity condition, not a strategy
parameter.

## Fresh untouched block and contamination audit

The fresh block is `2026-08-10` through `2026-08-14`, labelled
`FRESH_UNTOUCHED_L2_V3_POC_ONLY_AUG10_14_2026`. A read-only textual and
filename audit must be repeated immediately before acquisition. The initial
audit searched repository source/test/docs text and `data`/`research_runs`
filenames for every date in this block and found no L3/L2 diagnostic, signal,
data or parameter-selection reference. It did not read DBNs, prices, or result
contents. Any future evidence of prior use fails the fresh-block seal.

## April cost proxy

The quote-only retrospective proxy starts Monday `2026-04-06` and supports 5,
10, and 15 completed RTH sessions. Good Friday (`2026-04-03`) is excluded;
therefore April 6 uses April 2 as the prior completed RTH. This is cost and
technical planning only, explicitly not strict OOS evidence.

## Calendar and Friday requirement

The frozen normal-session hard-flat remains 22:45 UTC. Before any acquisition
or replay a calendar check must establish whether each planned session has a
scheduled close before 22:45. For an early-close date, the runner must fail
closed until an explicit calendar execution rule is separately frozen; it must
not buy nonexistent post-close data or silently reinterpret `HARD_CUTOFF_2245`.

## Quote-only commands

```powershell
python .\quote_l2_v3_poc_only_cost.py --block august-fresh --sessions 5 --quote
python .\quote_l2_v3_poc_only_cost.py --block april --sessions 5 --quote
python .\quote_l2_v3_poc_only_cost.py --block april --sessions 10 --quote
python .\quote_l2_v3_poc_only_cost.py --block april --sessions 15 --quote
```

The commands permit symbology metadata and `Historical.metadata.get_cost` only.
They do not expose credentials and contain no time-series or download call.
Their output separately totals `ES_MBP10_USD`, `MES_MBP1_USD`,
`PRIOR_RTH_TRADES_USD`, `TOTAL_USD`, and `AVG_USD_PER_SESSION`.
