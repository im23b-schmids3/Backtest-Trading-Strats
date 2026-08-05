# VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V1_SPECIFICATION

**Status:** SEALED PRE-IMPLEMENTATION SPECIFICATION. **Study ID:** `VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V1_SPECIFICATION`. No results are asserted by this document.

## 1. Scope, chronology, and data lock

Instrument is BTCUSDT; both long and short are permitted; at most one position may be active. The sole analytical timeframe is **5m only**, using completed OHLCV bars. The only permitted Phase-A input is the supplied manifest:

- `data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json`
- SHA-256: `9fb7228ca074fc5a3b90e6fe82181d07d49f8c62ebd68274e859d0361df3cd6e`

Phase A is exactly `2023-01-01T00:00:00Z` through `2024-01-31T23:59:59.999999Z`, inclusive. No other Phase-A manifest, input source, derived data, or parameter source is permitted. Validate the manifest hash before reading bars; a mismatch fails closed.

Locked Phase B is exactly `2024-02-01T00:00:00Z` through `2024-07-31T23:59:59.999999Z`, inclusive. It is inaccessible before a complete Phase-A pass/freeze: do not list, open, hash, sample, inspect, query, derive from, or otherwise access Phase-B material before all Phase-A gates pass, candidate ranking is complete, and at most one candidate is frozen. Phase-B facts must not alter this specification.

Required bar schema is JSON/object rows with `timestamp_utc` (RFC3339 UTC, 5-minute bar-open timestamp), `open`, `high`, `low`, `close`, `volume`, and `daily_vwap`. Prices and `daily_vwap` are finite positive numbers; volume is finite and non-negative; `high >= max(open, close)`, `low <= min(open, close)`, and `high >= low`. Timestamps are strictly increasing and unique. `daily_vwap` is validated as an available existing field but is not an additional V1 trend family or filter. Missing bars are never fabricated; any required unavailable history makes the relevant bar ineligible and must be reconciled, and an input-schema/chronology failure is a gate failure. No daily, order-book, funding, liquidation, news, or non-OHLCV input is allowed.

## 2. Sealed candidates and frequency

Exactly these candidates are sealed, and **only target R differs**:

| Candidate ID | Target R |
|---|---:|
| `VBTC-V1-1P5R` | 1.5R |
| `VBTC-V1-2P0R` | 2.0R |
| `VBTC-V1-2P5R` | 2.5R |

frequency objective approximately 100-300 annualized executed trades. This is measured as `executed_trades * 365.25 / calendar_days_in_phase`. OVERFREQUENCY_WARNING above 350 annualized trades, not automatic failure. A Phase-A pass nevertheless requires at least 108 Phase-A trades and at least 100 annualized trades.

No candidate beyond these three, parameter grid, filter substitution, discretionary intervention, rerun-based choice, or Phase-B-informed change is allowed.

## 3. Exact deterministic design

The completed-bar range is exactly 36 prior completed bars (a fixed 24-48 completed-bar range). The one OHLCV trend family is EMA alignment and EMA slope: no other trend family is allowed. `EMA20` and `EMA50` are close EMAs with standard recursive update and simple-average seed. `TR[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))`; `ATR14[i]` is Wilder ATR seeded with the first 14 eligible TR values.

For each signal bar `t`, using no data later than its close:

- `RH = max(high[t-36] ... high[t-1])`; `RL = min(low[t-36] ... low[t-1])`.
- `A = ATR14[t-1]` is the sole prior-only ATR or price volatility threshold input.
- Long trend: `EMA20[t-1] > EMA50[t-1]` and `EMA50[t-1] > EMA50[t-6]`.
- Short trend: `EMA20[t-1] < EMA50[t-1]` and `EMA50[t-1] < EMA50[t-6]`.
- Breakout threshold: long `close[t] >= RH + 0.10*A`; short `close[t] <= RL - 0.10*A`. Equality qualifies.
- The only one expansion filter is `TR[t] >= 1.25 * A`. Equality qualifies.

There is one entry mode only: next-bar market entry at `open[t+1]`. No pullback entry exists. Entry session is `00:05:00Z` through `23:35:00Z`, inclusive, by signal-bar timestamp; positions are forcibly flat at the close of the `23:55:00Z` bar. The exact risk rules are: one position maximum; stop distance must be at least `0.0020 * entry` and at most `0.0125 * entry`; no re-entry, scale-in/out, trailing stop, or stop loosening; time stop at the close of the 24th completed bar after entry. Use the pre-existing verified BTC fee, slippage, sizing, lot-rounding, margin/liquidation, and accounting assumptions unchanged, identified by an immutable execution-assumption hash; absence or mismatch fails closed. In addition to baseline results, robustness applies one extra tick of adverse slippage on both entry AND exit.

## 4. No-lookahead pseudocode

```text
for each completed 5m bar t in strictly increasing timestamp order:
  require all bars t-50 through t and all indicator seed history; otherwise do not propose a setup
  RH = max(high[t-36..t-1]); RL = min(low[t-36..t-1]); A = ATR14[t-1]
  long_trend = EMA20[t-1] > EMA50[t-1] and EMA50[t-1] > EMA50[t-6]
  short_trend = EMA20[t-1] < EMA50[t-1] and EMA50[t-1] < EMA50[t-6]
  long_break = close[t] >= RH + 0.10*A
  short_break = close[t] <= RL - 0.10*A
  expansion = TR[t] >= 1.25*A                 # TR[t] is known only after t closes
  for each direction in [LONG, SHORT], create one proposed setup with planned entry bar t+1:
    if not direction's trend condition: terminal TREND_FILTER_REJECTED
    else if not direction's breakout: terminal BREAKOUT_THRESHOLD_REJECTED
    else if not expansion: terminal EXPANSION_FILTER_REJECTED
    else if t timestamp is 23:55:00Z: terminal SESSION_ENDED
    else if t is outside 00:05:00Z..23:35:00Z: terminal SESSION_ENTRY_BLOCKED
    else if active_position or scheduled_entry: terminal ACTIVE_POSITION_BLOCKED
    else schedule that direction for entry at open[t+1]

at open[t+1] for each scheduled setup:
  if bar t+1 is unavailable or its open is not executable under fixed assumptions:
     terminal NO_EXECUTABLE_ENTRY
  else if (long and open[t+1] < RH) or (short and open[t+1] > RL):
     terminal FALSE_BREAKOUT_INVALIDATED
  else if long:
     entry = open[t+1]; stop = low[t] - 0.10*A; d = entry - stop
  else:
     entry = open[t+1]; stop = high[t] + 0.10*A; d = stop - entry
  if d < 0.0020*entry or d > 0.0125*entry: terminal STOP_DISTANCE_REJECTED
  else execute at entry, target = entry +/- candidate_target_R*d, and terminal TRADE_EXECUTED

for each active executed trade, for each completed bar x after entry:
  if long and low[x] <= stop and high[x] >= target: exit at stop (stop-first)
  else if short and high[x] >= stop and low[x] <= target: exit at stop (stop-first)
  else if long and low[x] <= stop: exit at stop
  else if short and high[x] >= stop: exit at stop
  else if long and high[x] >= target: exit at target
  else if short and low[x] <= target: exit at target
  else if x is the 24th completed bar after entry: exit at close[x]
  else if x timestamp is 23:55:00Z: exit at close[x]
```

The false-break invalidation is deterministic and pre-entry: a scheduled long whose next-bar open is below its original `RH`, or a scheduled short whose next-bar open is above its original `RL`, is invalidated without a trade. It never creates a new setup. Intrabar collision is always stop-first. No same-bar signal fill is permitted.

## 5. Identifiers, setup outcomes, and reconciliation

Canonical JSON means UTF-8, sorted keys, no insignificant whitespace, RFC3339 UTC timestamps, and normalized decimal strings. IDs are lowercase SHA-256 hex:

```text
setup_id     = sha256({study_id, phase, symbol, signal_bar_timestamp, direction, range_high:RH, range_low:RL, breakout_level, atr:A, range_bars:36, trend_family:"EMA20_EMA50", threshold_atr:"0.10", expansion_multiple:"1.25"})
structure_id = sha256({setup_id, breakout_bar_timestamp, entry_bar_timestamp, entry_mode:"NEXT_OPEN_MARKET", proposed_stop_or_null, stop_distance_or_null, session:"00:05-23:35Z", time_stop_bars:24})
event_id     = sha256({structure_id, event_timestamp, event_type, bar_timestamp, direction, price_rule})
trade_id     = sha256({candidate_id, structure_id, entry_event_id, target_r, execution_assumption_hash})
```

Every proposed setup record contains deterministic `setup_id`, `structure_id`, and `event_id`, candidate ID, direction, range high/low, breakout level and breakout bar, entry bar, `history`, terminal disposition, and `trade_id` if executed. For a pre-entry rejection, `entry_bar` is the planned next bar and `proposed_stop_or_null`/`stop_distance_or_null` are canonical JSON null; it still has a structure ID. Every proposed setup has exactly one mutually exclusive terminal disposition, chosen only from:

`TREND_FILTER_REJECTED`, `BREAKOUT_THRESHOLD_REJECTED`, `EXPANSION_FILTER_REJECTED`, `FALSE_BREAKOUT_INVALIDATED`, `PULLBACK_WINDOW_EXPIRED` only if pullback mode was sealed (otherwise mark not-applicable and do not emit), `STOP_DISTANCE_REJECTED`, `NO_EXECUTABLE_ENTRY`, `ACTIVE_POSITION_BLOCKED`, `SESSION_ENTRY_BLOCKED`, `SESSION_ENDED`, `TRADE_EXECUTED`.

`TREND_FILTER_REJECTED` is emitted for an evaluated direction when its trend condition fails. `FALSE_BREAKOUT_INVALIDATED` is emitted only when the sealed next-bar-open test is back inside the original range boundary before entry. `SESSION_ENDED` is emitted only for a qualifying signal on the UTC session's final completed bar; an active trade flattened at 23:55Z remains `TRADE_EXECUTED` and records `SESSION_FLAT` as its trade exit reason. Pullback mode is not sealed: PULLBACK_WINDOW_EXPIRED is not emitted. Price exits, targets, time stops, and force-flats are trade exit fields, not additional setup dispositions.

For every candidate, `proposed_setups equals disposition sum`, and `TRADE_EXECUTED equals trades/report count`. Every event/trade refers to a valid setup ID; every executed setup has exactly one valid trade ID; IDs are unique in their applicable namespace. Funnel, proposed-setups, dispositions, executed trades, costs, net PnL, and report totals must reconcile exactly. Any mismatch fails the integrity gate.

## 6. Immutable artifact layout and schemas

The deterministic root is `research/volatility_breakout_trend_continuation/v1/{phase}/study-{specification_sha256}/candidate-{candidate_id}/`; study-level artifacts omit `candidate-{candidate_id}`. A future run creates each path once with create-new semantics. **No overwrite rule:** an existing file at an intended immutable path is a failure; never replace, append to, mutate, or delete an artifact. `integrity-manifest.json` hashes every artifact and records its relative path, byte length, SHA-256, specification hash, manifest hash, execution-assumption hash, code revision, phase, and creation timestamp.

All artifacts are UTF-8 JSON objects (arrays only where stated), schema-versioned, canonicalized for hashing, and include `study_id`, `phase`, `specification_sha256`, `data_manifest_sha256`, `candidate_id` when candidate-scoped, and `created_at_utc`.

| Artifact | Required schema/content |
|---|---|
| `sealed-specification.json` | exact specification text/hash, study ID, seal status |
| `candidate-registry.json` | exactly the three IDs and their target R; confirms only target differs |
| `data-manifest.json` | supplied Phase-A manifest path/hash, requested input schema, chronology, validation result; Phase B only after freeze |
| `configuration.json` | every fixed formula, parameter, execution-assumption hash, code revision, and canonical config hash |
| `events.json` | array of event records with event_id, valid setup_id, timestamp, type, direction, price rule |
| `trades.json` | array of executed trades with trade_id, valid setup_id/structure_id, entry/exit, gross/net PnL, net R, fees, slippage, exit reason |
| `setup_outcomes.json` | array of all proposed setups and their one terminal disposition, IDs, range, breakout/entry/history fields |
| `monthly_metrics.json` | calendar-month counts, gross/net PnL, net R, PF, direction metrics, and zero-trade status |
| `report.json` | candidate summary, funnel, trade/setup/cost reconciliation, frequency, robustness calculations |
| `gates.json` | literal gate names, inputs, pass/fail values, and reasons |
| `selection_report.json` | complete Phase-A pass set, ranking tuple, chosen candidate or `PHASE_A_NO_ROBUST_CANDIDATE` |
| `freeze.json` | at most one selected candidate, frozen configuration/manifest/specification hashes and UTC timestamp; or no-candidate disposition |
| `integrity-manifest.json` | immutable inventory and hashes described above |
| `final_report.json` | phase outcome, all gates, reconciliation, integrity status, and no-result-claim provenance |

## 7. Phase-A hard gates and selection

A candidate is a complete Phase-A pass only if every item below passes:

1. supplied manifest hash equals `9fb7228ca074fc5a3b90e6fe82181d07d49f8c62ebd68274e859d0361df3cd6e`, requested data schema/chronology pass, and complete funnel/trade/setup/cost/immutable artifact reconciliation passes;
2. positive net PnL after costs;
3. net PF >=1.30;
4. positive average net R;
5. maximum DD <=20R;
6. at least 108 Phase-A trades and at least 100 annualized trades;
7. >=8 profitable months;
8. <=3 zero-trade months;
9. best month <=35% positive PnL;
10. best five trades <=30% positive PnL;
11. long/short each >=25% trades;
12. neither direction worse than -0.15R average/trade;
13. positive with one extra tick on entry AND exit;
14. positive after best-trade removal;
15. bootstrap median mean net R >0;
16. bootstrap 95% lower bound >=-0.025R;
17. >=3/4 2023 calendar subperiods nonnegative;
18. no lookahead, data/chronology, configuration, ID, disposition, execution, reconciliation, or immutable-artifact breach.

For gate 15-16, bootstrap exactly 10,000 resamples with replacement of the candidate's Phase-A net-R trade sequence, deterministic PCG64 seed `20240131`, statistic mean net R, percentile interval at 2.5%/97.5%. For gate 17, use 2023 calendar quarters Q1, Q2, Q3, Q4 and each quarter's net PnL after costs; nonnegative includes zero. For gates 9-10, the denominator is total positive PnL; failure is required if denominator is non-positive.

Candidate ranking only among complete Phase-A passes, ordered by descending average net R, descending net PF, ascending maximum DD, then lexicographic candidate ID. Freeze at most one, otherwise `PHASE_A_NO_ROBUST_CANDIDATE`. The `freeze.json` must be created before any Phase-B access.

## 8. Literal locked Phase-B gates

Without accessing Phase-B data in this specification process, the one frozen candidate may be assessed once only after its complete Phase-A pass/freeze. It passes only when all literal gates below pass:

1. unchanged frozen config/manifest chronology: the frozen specification hash, configuration hash, execution-assumption hash, selected candidate ID, and Phase-B chronology exactly `2024-02-01T00:00:00Z` through `2024-07-31T23:59:59.999999Z` are unchanged and non-overlapping;
2. input/reconciliation integrity: requested schema, manifest capture after freeze, chronological validation, ID/disposition checks, and complete funnel/trade/setup/cost/immutable-artifact reconciliation pass;
3. positive net PnL/average net R after costs;
4. PF >=1.00;
5. DD <=20R;
6. direction coverage when trades occur: if any trade occurs, every direction represented by at least one executed trade must have its trade count, PnL, and average net R reported; if both directions occur, each must be reported separately;
7. extra-slippage and best-trade-removal robustness: net PnL remains positive with one extra tick on entry AND exit and remains positive after best-trade removal;
8. no gate/specification breach.

There is no re-ranking, replacement, recalibration, second Phase-B run, or modification after Phase-B access.

## 9. Anti-overfitting, pre-implementation checklist, unresolved decisions

Anti-overfitting controls: this is a one-structure, three-target test; all non-target parameters above are frozen; report all candidates, all evaluated directions, all dispositions, all executed trades, all months, all costs, and every gate failure; do not add exclusions or selectively delete records; do not inspect Phase B before freeze.

Pre-implementation checklist:

- [ ] Verify the exact supplied Phase-A manifest SHA-256 before reading bars.
- [ ] Verify input schema, UTC chronology, missing-bar handling, and no-lookahead indicator availability.
- [ ] Bind the pre-existing verified BTC execution assumptions unchanged and hash them.
- [ ] Implement only the exact pseudocode, stop-first ordering, IDs, and dispositions.
- [ ] Create every immutable artifact once and complete all reconciliation before gates.
- [ ] Apply every Phase-A hard gate, rank only complete passes, and freeze at most one.
- [ ] Do not access Phase B until the freeze artifact is complete.

**Unresolved decisions:** none. Any unavailable verified execution assumption, schema ambiguity, or conflict is a fail-closed breach, not authorization to select an alternative. Any change requires a new study ID and a new sealed specification.
