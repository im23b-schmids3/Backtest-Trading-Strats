# VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V2_STRICT_SELECTION

**Status:** SEALED PRE-IMPLEMENTATION SPECIFICATION. **Study ID:** `VolatilityBreakoutTrendContinuation.BTC_LONG_SHORT_V2_STRICT_SELECTION`. This is a separate, stricter design prompted by the negative and over-frequent V1 diagnostic. It asserts no performance result and must not modify V1.

## 1. Scope, chronology, and data lock

The instrument is BTCUSDT; both long and short are permitted; at most one position may be active. The sole analytical timeframe is **completed 5-minute OHLCV bars**. The only permitted Phase-A input is the existing validated manifest, used in place:

- `data/imbalance_vwap_ride/v5/bars/BTCUSDT/phase_a/6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f/manifest.json`
- manifest SHA-256: `9fb7228ca074fc5a3b90e6fe82181d07d49f8c62ebd68274e859d0361df3cd6e`

Phase A is exactly `2023-01-01T00:00:00Z` through `2024-01-31T23:59:59.999999Z`, inclusive. Validate that exact manifest hash before reading any bar. No other source, timeframe, derived dataset, parameter source, download, rebuilt data, or fabricated bar is permitted.

Locked Phase B is exactly `2024-02-01T00:00:00Z` through `2024-07-31T23:59:59.999999Z`, inclusive. It must not be listed, opened, hashed, sampled, queried, or otherwise accessed unless exactly one V2 candidate passes every Phase-A hard gate and is frozen first. This specification process has not accessed Phase B.

The accepted bar contract is the existing V5 manifest's canonical timestamp field (mapped to `timestamp_utc`) plus `open`, `high`, `low`, `close`, `volume`, and `daily_vwap`. Timestamps are RFC3339 UTC 5-minute bar-open values, strictly increasing and unique. Prices and `daily_vwap` are finite and positive; volume is finite and non-negative; `high >= max(open, close)`, `low <= min(open, close)`, and `high >= low`. `daily_vwap` is input-contract validation only, not a V2 signal filter. A missing required history, schema breach, or chronology breach fails the integrity gate; it does not authorize imputation.

## 2. Sealed candidate registry

Exactly these candidates are permitted. **Only target R differs.**

| Candidate ID | Fixed target |
|---|---:|
| `VBTC-V2-2P5R` | 2.5R |
| `VBTC-V2-3P0R` | 3.0R |
| `VBTC-V2-3P5R` | 3.5R |

There is no parameter grid, target beyond this registry, filter substitution, alternate range length, Phase-A retuning, discretionary exclusion, rerun-based choice, or Phase-B-informed adjustment.

## 3. Exact sealed parameters

| Family | Value |
|---|---|
| bar interval | 5 minutes, completed bars only |
| prior range | 48 completed bars, excluding signal bar |
| EMA trend | close EMA20 and EMA50; standard recursive EMA with simple-average seeds |
| trend slopes | both EMA20 and EMA50 compared with their values six completed bars earlier |
| ATR | Wilder ATR14; signal uses prior-only `ATR14[t-1]` |
| EMA separation | `abs(EMA20[t-1] - EMA50[t-1]) >= 0.20 * ATR14[t-1]` |
| breakout threshold | `0.20 * ATR14[t-1]` beyond the prior range |
| true-range expansion | `TR[t] >= 1.50 * ATR14[t-1]` |
| breakout close quality | long close in upper 25% / short close in lower 25% of breakout bar |
| volume confirmation | `volume[t] >= 1.25 * median(volume[t-20..t-1])` |
| duplicate policy | first admitted qualifying breakout per deterministic range structure only |
| directional cooldown | 12 completed bars after an executed trade, same direction only |
| entry | next-bar market entry at `open[t+1]` |
| false break | long `open[t+1] <= RH`; short `open[t+1] >= RL` |
| stop | long `low[t] - 0.10*A`; short `high[t] + 0.10*A` |
| stop bounds | `0.0020 * entry <= risk_distance <= 0.0125 * entry` |
| time stop | close of 24th completed bar after entry |
| force flat | close of UTC `23:55` bar |
| collision | stop first |

`TR[t] = max(high[t]-low[t], abs(high[t]-close[t-1]), abs(low[t]-close[t-1]))`. `A = ATR14[t-1]`. For the even twenty-bar reference, median is `(sorted_volumes[9] + sorted_volumes[10]) / 2`, using exactly the twenty completed bars `t-20` through `t-1`; the current breakout bar never appears in its own reference set.

For a non-zero breakout-bar range, long close quality is `(close[t]-low[t]) / (high[t]-low[t]) >= 0.75`; short close quality is `(high[t]-close[t]) / (high[t]-low[t]) >= 0.75`. A zero-range bar cannot satisfy breakout close quality and receives `BREAKOUT_CLOSE_QUALITY_REJECTED` if it reaches that check.

The V1 verified BTC execution assumptions remain mandatory and unchanged: BTC quantity and quantity-step, price tick, minimum quantity, fee rate, market/stop adverse slippage, lot rounding, margin/liquidation treatment, accounting, and `STOP_FIRST` policy are captured by an immutable execution-assumption hash. Their absence or hash mismatch fails closed. No pyramiding, averaging down, scale-in/out, trailing stop, stop loosening, or same-bar signal fill is permitted.

## 4. Deterministic structures, IDs, and duplicate suppression

Canonical JSON is UTF-8, sorted keys, no insignificant whitespace, RFC3339 UTC timestamps, and normalized decimal strings. All IDs are lowercase SHA-256 hex.

For signal bar `t`, define `RH = max(high[t-48..t-1])`, `RL = min(low[t-48..t-1])`, and `range_start = timestamp[t-48]`, `range_end = timestamp[t-1]`. The deterministic range structure is:

```text
structure_id = sha256({
  study_id, phase, symbol, direction,
  range_start, range_end, range_high: RH, range_low: RL,
  range_bars: 48, trend_family: "EMA20_EMA50_DUAL_SLOPE",
  threshold_atr: "0.20", expansion_multiple: "1.50",
  ema_separation_atr: "0.20", volume_lookback_bars: 20,
  volume_multiplier: "1.25"
})
```

`setup_id = sha256({structure_id, signal_bar_timestamp, atr:A, breakout_level, breakout_close, true_range, reference_volume_median})`.

`event_id = sha256({structure_id, setup_id, event_timestamp, event_type, bar_timestamp, direction, price_rule})`.

`trade_id = sha256({candidate_id, structure_id, setup_id, entry_event_id, target_r, execution_assumption_hash})`.

The first setup that passes all signal-quality checks and entry-admission checks for a `structure_id` claims that structure before next-bar execution. Later otherwise-qualifying breakouts with that same `structure_id` receive `DUPLICATE_STRUCTURE_BLOCKED`; they cannot enter, claim the structure, or alter the first setup. A structure blocked before duplicate admission (for session, active-position, or directional-cooldown reasons) is not claimed.

After an executed long trade with entry index `e`, planned long entries at indices `e+1` through `e+12`, inclusive, receive `DIRECTIONAL_COOLDOWN_BLOCKED`. The corresponding rule applies independently to shorts. Opposite-direction setups are still evaluated, but the one-active-position rule remains binding.

## 5. No-lookahead long and short pseudocode

```text
for each completed 5m bar t in strictly increasing UTC order:
  require completed indicator history through t-1, bars t-50..t,
          prior range t-48..t-1, and prior volume bars t-20..t-1;
  if unavailable: do not propose a setup.

  RH = max(high[t-48..t-1]); RL = min(low[t-48..t-1]); A = ATR14[t-1]
  tr = TR[t]
  ema20_up = EMA20[t-1] > EMA20[t-6]
  ema50_up = EMA50[t-1] > EMA50[t-6]
  ema20_down = EMA20[t-1] < EMA20[t-6]
  ema50_down = EMA50[t-1] < EMA50[t-6]

  for direction in [LONG, SHORT]:
    create one proposed setup with planned entry bar t+1.
    long trend = EMA20[t-1] > EMA50[t-1] and ema20_up and ema50_up
    short trend = EMA20[t-1] < EMA50[t-1] and ema20_down and ema50_down
    long break = close[t] >= RH + 0.20*A
    short break = close[t] <= RL - 0.20*A

    if direction trend fails: terminal TREND_FILTER_REJECTED
    else if abs(EMA20[t-1]-EMA50[t-1]) < 0.20*A: terminal EMA_SEPARATION_REJECTED
    else if direction break fails: terminal BREAKOUT_THRESHOLD_REJECTED
    else if tr < 1.50*A: terminal EXPANSION_FILTER_REJECTED
    else if breakout close quality fails: terminal BREAKOUT_CLOSE_QUALITY_REJECTED
    else if volume[t] < 1.25 * median(volume[t-20..t-1]): terminal VOLUME_CONFIRMATION_REJECTED
    else if t timestamp is 23:55Z: terminal SESSION_ENDED
    else if t timestamp is outside 00:05Z..23:35Z inclusive: terminal SESSION_ENTRY_BLOCKED
    else if an active position exists or a position is scheduled: terminal ACTIVE_POSITION_BLOCKED
    else if same-direction planned entry is within its 12-bar cooldown: terminal DIRECTIONAL_COOLDOWN_BLOCKED
    else if structure_id is already claimed: terminal DUPLICATE_STRUCTURE_BLOCKED
    else claim structure_id and schedule next-bar entry.

at open[t+1] for each scheduled setup:
  if t+1 is unavailable or open is not executable under the immutable BTC assumptions:
    terminal NO_EXECUTABLE_ENTRY
  else if long and open[t+1] <= RH: terminal FALSE_BREAKOUT_INVALIDATED
  else if short and open[t+1] >= RL: terminal FALSE_BREAKOUT_INVALIDATED
  else calculate the sealed direction-specific stop and risk distance.
  if risk distance is outside [0.0020*entry, 0.0125*entry]: terminal STOP_DISTANCE_REJECTED
  else execute at next-bar market open with immutable adverse slippage, quantity,
       tick/step quantization, and fees; target = entry +/- candidate_target_R*risk.
       terminal TRADE_EXECUTED.

for every active executed trade, only on completed bars after entry:
  if stop and target are both hit in a bar: exit at stop.
  else evaluate stop, target, 24th-completed-bar time stop, then 23:55 UTC flat.
```

An executed setup remains `TRADE_EXECUTED` even if its trade later exits by stop, target, time stop, or `SESSION_FLAT`; the exit is a trade field, never a second terminal setup disposition.

## 6. Terminal dispositions and corrected reconciliation

Every proposed setup has exactly one mutually exclusive terminal disposition:

```text
TREND_FILTER_REJECTED
EMA_SEPARATION_REJECTED
BREAKOUT_THRESHOLD_REJECTED
EXPANSION_FILTER_REJECTED
BREAKOUT_CLOSE_QUALITY_REJECTED
VOLUME_CONFIRMATION_REJECTED
DUPLICATE_STRUCTURE_BLOCKED
SESSION_ENDED
SESSION_ENTRY_BLOCKED
ACTIVE_POSITION_BLOCKED
DIRECTIONAL_COOLDOWN_BLOCKED
FALSE_BREAKOUT_INVALIDATED
STOP_DISTANCE_REJECTED
NO_EXECUTABLE_ENTRY
TRADE_EXECUTED
```

Event lifecycle:

- every setup emits one `PROPOSED_SETUP` event and exactly one event whose type is its terminal disposition;
- an executed setup additionally emits exactly one `ENTRY` event;
- exits are represented only in the corresponding trade record.

Therefore, for every candidate:

```text
event_count == 2 * proposed_setups + executed_trades
disposition_sum == proposed_setups
TRADE_EXECUTED outcomes == trades.json count == report.executed_trades
```

Every event must reference one valid `setup_id`; every setup must have one terminal disposition; every trade must reference exactly one `TRADE_EXECUTED` setup; identifiers are unique in their namespaces. Any mismatch fails the integrity gate.

## 7. Frequency, economic gates, selection, and Phase B lock

Annualized frequency is `executed_trades * 365.25 / 396` for this exact Phase-A chronology.

- `< 100`: `UNDERFREQUENCY_FAIL`
- `100` through `500`, inclusive: frequency gate passes
- `> 500`: `OVERFREQUENCY_FAIL`

The V2 frequency gate replaces V1's frequency requirement. Preserve every V1 economic, robustness, concentration, direction-mix, monthly, bootstrap, cost-stress, drawdown, reconciliation, data-integrity, configuration, and immutable-artifact hard gate, unchanged:

1. positive net PnL after costs;
2. net profit factor `>= 1.30`;
3. positive average net R;
4. maximum drawdown `<= 20R`;
5. at least 8 profitable months and no more than 3 zero-trade months;
6. best month no more than 35% of positive PnL and best five trades no more than 30% of positive PnL;
7. long and short each at least 25% of trades; neither direction average below `-0.15R`;
8. positive net PnL after one extra adverse tick on both entry and exit;
9. positive net PnL after best-trade removal;
10. deterministic bootstrap: exactly 10,000 replacement resamples, PCG64 seed `20240131`, statistic mean net R, median above zero and 2.5th percentile `>= -0.025R`;
11. at least 3 of 4 calendar 2023 quarters nonnegative after costs;
12. no lookahead, data, chronology, execution, ID, disposition, reconciliation, or immutable-artifact breach.

Only complete Phase-A passes are ranked: descending average net R, descending net PF, ascending maximum drawdown, then lexicographic candidate ID. Freeze at most one candidate. Otherwise final status is `PHASE_A_NO_ROBUST_CANDIDATE`. No Phase-B access occurs before `freeze.json` is written; Phase B runs at most once only for the frozen candidate, with the V1 literal locked Phase-B gates unchanged.

## 8. Immutable artifact layout

The V2 root is `research/volatility_breakout_trend_continuation/v2/{phase}/study-{specification_sha256}/`; candidate artifacts are under `candidate-{candidate_id}/`. All writes use create-new semantics. An existing intended artifact or run directory is an immutable collision; it must never be replaced, appended to, mutated, or deleted.

Every canonical UTF-8 JSON artifact is schema-versioned and contains study ID, phase, specification hash, data-manifest hash, candidate ID when applicable, execution-assumption hash when applicable, and creation time. Required artifacts are:

- `sealed-specification.json`
- `candidate-registry.json`
- `data-manifest.json`
- `configuration.json`
- `events.json`
- `trades.json`
- `setup_outcomes.json`
- `monthly_metrics.json`
- `report.json`
- `gates.json`
- `selection_report.json`
- `freeze.json`
- `integrity-manifest.json`
- `final_report.json`

`integrity-manifest.json` inventories the relative path, size, SHA-256, specification hash, data-manifest hash, execution-assumption hash, code revision, phase, and creation timestamp of every sealed artifact. Reports include the frequency result, all terminal-disposition counts, corrected event formula values, full costs/PnL/net-R reconciliation, monthly and direction metrics, robustness outputs, every gate result, ranking, and no-result-claim provenance.

## 9. Pre-implementation checklist

- [ ] Verify this exact V2 specification hash after sealing.
- [ ] Validate only the stated Phase-A manifest, SHA-256, schema, UTC chronology, and partition hashes before reading bars.
- [ ] Confirm Phase B has not been accessed.
- [ ] Bind and hash the pre-existing verified BTC execution assumptions without substitution.
- [ ] Implement the 48-bar, dual-EMA-slope, ATR, close-quality, volume, duplicate, and cooldown rules exactly in stated order.
- [ ] Implement no same-bar entry or lookahead and preserve stop-first ordering.
- [ ] Emit exactly one terminal disposition per setup and validate `events == 2 * proposed_setups + executed_trades`.
- [ ] Materialize every immutable artifact once, validate its inventory, then apply all Phase-A gates.
- [ ] Freeze at most one complete Phase-A pass before any Phase-B access.

**Unresolved decisions:** none. A missing verified assumption, unavailable required history, manifest/schema conflict, or reconciliation breach fails closed; it is not authorization to substitute a rule or data source. Any change requires a new study ID and a new sealed specification.
