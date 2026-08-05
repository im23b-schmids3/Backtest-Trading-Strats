# HTFLevelLiquidityFVG.BTC_LONG_SHORT_V1_SPECIFICATION

## Seal and Scope

This is the sealed, immutable specification for `HTFLevelLiquidityFVG.BTC_LONG_SHORT_V1`. It is a causal BTC long/short study specification. No optimization, parameter search, adaptive thresholding, look-ahead, retry after a terminal disposition, or cross-candidate leakage is permitted. A change requires a new versioned specification and SHA-256.

## Phase Lock and Unopened Input Contract

- **Phase A:** `2023-01-01T00:00:00Z` through `2024-01-31T23:59:59.999999Z`, inclusive.
- **Locked Phase B:** `2024-02-01T00:00:00Z` through `2024-07-31T23:59:59.999999Z`, inclusive.
- The unopened Phase-A input contract is exactly `C:\Users\sandr\Trading-Bot-Fib\data\imbalance_vwap_ride\v5\bars\BTCUSDT\phase_a\6c75fc621bdb83ed10e687013e5d675f46ab96fa041ef9fda19b435d9ec5a65f\manifest.json`. It is labelled verbatim in `input-contract.json`, validated only by a later implementation, and is not read, opened, inferred, or resolved while sealing this specification.
- No Phase B, historical run, research run, market-data file, or manifest is read by producing or sealing this document.

## Bars, Time, and Causality

- Source bars are UTC, interval-start, 5-minute OHLCV bars. Derive UTC 15-minute, 4-hour, and daily bars in memory: respectively 3, 48, and 288 complete source bars.
- A derived bar is usable only when all constituents are closed. Its open is first open, high is maximum high, low is minimum low, close is last close, and volume is summed.
- At each confirmed 5-minute close, finalize derived bars first, then evaluate only facts known through that close. An order activated at that close cannot fill before the next executable price event.
- `ATR14(T)` is the Wilder ATR of the prior 14 confirmed bars of timeframe `T`, excluding the bar being tested. A rule needing unavailable prior ATR fails its applicable rule without creating an executable setup.

## Daily Bias

On each confirmed daily bar, calculate EMA50 of daily closes and its five-day slope `EMA50[t] - EMA50[t-5]`.

- `BULLISH`: close > EMA50 and five-day slope > 0.
- `BEARISH`: close < EMA50 and five-day slope < 0.
- Otherwise `NEUTRAL`.

Only bullish bias permits long setups; only bearish bias permits short setups. A bias change prevents new progression but does not alter a previously activated order or filled position.

## Confirmed 4H Levels

A 4-hour swing low is a bar whose low is strictly below the lows of each of the prior three confirmed 4-hour bars and less than or equal to the lows of each of the next three confirmed 4-hour bars. A 4-hour swing high is strictly above the highs of each of the prior three confirmed 4-hour bars and greater than or equal to the highs of each of the next three confirmed 4-hour bars. It becomes confirmed only after the third right bar closes.

- A swing low creates a `SUPPORT` at its low; a swing high creates a `RESISTANCE` at its high.
- Each level has deterministic ID, source bar, confirmation time, price, side, and state.
- A level is eligible until consumed. A qualifying 15-minute sweep consumes exactly one level, irrevocably. If multiple same-side levels qualify in one 15-minute bar, select the nearest eligible level below the current price for a long sweep or above the current price for a short sweep; deterministic level ID breaks an exact price tie.

## 15-Minute Liquidity Sweep

Evaluate only newly confirmed 15-minute bars against eligible levels known before the bar opened, using prior `ATR14(15m)`.

- Long: low <= support - 0.10 ATR, close >= support, and lower wick `(min(open, close)-low)/(high-low)` >= 0.50.
- Short: high >= resistance + 0.10 ATR, close <= resistance, and upper wick `(high-max(open, close))/(high-low)` >= 0.50.
- A zero-range bar fails the sweep gate. The first qualifying sweep consumes the level and creates one directional setup with immutable sweep evidence.

## 5-Minute MSS, Displacement, and FVG

For each setup, inspect only later confirmed 5-minute bars. A two-sided fractal low/high is respectively a low strictly below / high strictly above the two bars on each side; it confirms after the second right bar closes.

- Within 12 completed 5-minute bars after the sweep bar, long requires a close above the most recent confirmed two-sided fractal high that existed before the MSS bar; short requires a close below the most recent confirmed two-sided fractal low that existed before the MSS bar. This is MSS.
- The MSS bar is the displacement bar. Long requires close > open, true range >= 1.50 prior `ATR14(5m)`, and close >= low + 0.75(high-low). Short mirrors this: close < open, true range >= 1.50 prior ATR, and close <= high - 0.75(high-low).
- The FVG is the first qualifying three-candle sequence that forms no earlier than the MSS/displacement sequence and no later than two completed 5-minute bars after the MSS bar. Bullish FVG: `low[candle3] > high[candle1]`; bearish FVG: `high[candle3] < low[candle1]`. Its width must be >= 0.10 prior `ATR14(5m)`.
- Bullish boundaries are lower=`high[candle1]`, upper=`low[candle3]`; bearish boundaries are lower=`high[candle3]`, upper=`low[candle1]`. Record all three bars and immutable boundaries.
- Failure to confirm MSS, displacement, and FVG in the stated order inside the 12-bar window is `MSS_WINDOW_EXPIRED`.

## FVG Orders, Entry, Expiry, and Invalidation

- Activate after full FVG confirmation at the bullish FVG upper boundary for a long and bearish FVG lower boundary for a short. No entry occurs on the FVG-confirmation bar; filling begins on the next completed 5-minute bar only.
- Long fills only if that later bar low reaches/passes the limit; short fills only if that later bar high reaches/passes the limit. Apply sealed limit-order slippage and execution assumptions.
- Before a long fills, invalidate if price trades below the sweep low, daily long bias becomes invalid, the swept 4H level becomes unavailable/invalid, or the order expires; mirror for short.
- An unfilled order expires after 12 completed 5-minute bars after activation. Pending orders are cancelled at 23:55 UTC; no new entry after 23:40 UTC.

## Stop, Targets, Partials, and Costs

- For a long, stop = sweep-bar low - 0.05 prior `ATR14(15m)`; for a short, stop = sweep-bar high + 0.05 prior 15-minute ATR. Reject the setup if absolute entry-to-stop distance is less than 0.25% or greater than 2.00% of entry.
- Define `R = abs(entry - stop)` before entry. TP1 is the nearest confirmed opposing 15-minute swing-liquidity level and TP2 the nearest confirmed opposing 4H swing-liquidity level, both existing before entry; never use future-confirmed levels.
- Exit 50% of original quantity at TP1 and 50% at TP2. On TP1, move the remaining stop to entry adjusted by all entry costs and projected remaining-exit costs, so a stop at that price realizes break-even after costs.
- Cost-adjusted break-even is calculated before activation from the configured deterministic fee/slippage schedule and recorded with the order. Every entry and exit fill records quantity, price, gross impact, fees, slippage, net P&L, and realized R.
- If stop and target are touched in one 5-minute bar, stop has precedence. A position still open at 23:55 UTC exits at that bar's close as `FORCED_TIME_EXIT`; otherwise the 96th completed 5-minute bar after entry is the maximum holding bar and exits at its close as `FORCED_TIME_EXIT`.

## Candidate Registry

Run exactly these candidate IDs and no others. Their only differentiator is the minimum available fixed TP2 R multiple.

| Candidate | Required TP2 multiple | Result |
|---|---:|---|
| `HTFLFVG-V1-MIN2P5R` | >= 2.5R | admitted |
| `HTFLFVG-V1-MIN3P0R` | >= 3.0R | admitted |
| `HTFLFVG-V1-MIN4P0R` | >= 4.0R | admitted |

Reject a setup when its fixed entry-to-TP2 projected R is below its candidate threshold. TP2 remains level-based, not a fixed-R target.

## Authoritative Execution Matrix

This matrix resolves all operational details. If general prose elsewhere conflicts with this section, this section controls.

### State, Risk, and Session Controls

- There may be at most one active position and one active pending order across both directions. There is no pyramiding, averaging down, simultaneous long/short exposure, or replacement order while another order is active.
- A qualifying 15-minute sweep creates the deterministic `setup_id` and the proposed-setup record. The record then progresses through the directional chain `MSS -> displacement -> first valid FVG -> candidate projected-R check -> order/fill`, or receives one terminal disposition. This is why early-stage rejection dispositions are included in the funnel.
- A pending order is cancelled at `23:55 UTC`; no new order may be activated after `23:40 UTC`. Any position open at `23:55 UTC` is force-flattened at that bar close. The maximum holding duration is 96 completed 5-minute bars after entry.
- Risk accounting is normalized to `1R` at entry. No leverage, liquidation, funding, or account-growth model is implied. Use the repository's verified BTCUSDT price tick, quantity step/minimum quantity, maker/taker fee schedule, adverse market-entry/exit slippage, and deterministic limit-fill assumptions when implementation is authorized.
- Quantization that prevents a valid executable quantity is a `COMPLIANCE_BLOCKED` terminal disposition with reason code `QUANTITY_NOT_EXECUTABLE`.

### Long Pseudocode

1. At each completed daily bar, determine `BULLISH` only when daily close is above EMA50 and EMA50 is higher than five completed daily bars earlier.
2. At each completed 15-minute bar, choose the nearest eligible confirmed 4H support below current price; require the sealed penetration, reclaim, and lower-wick conditions. Consume that exact support only if the sweep qualifies.
3. On subsequent completed 5-minute bars, require a bullish MSS above a pre-existing confirmed fractal high, bullish displacement, and the first valid bullish FVG within two completed bars after MSS and within the 12-bar post-sweep window.
4. Activate one long limit at the bullish FVG upper boundary after FVG confirmation. The earliest possible fill is the next executable 5-minute bar. Before fill, cancel for a trade below the recorded sweep low, invalid daily bias, invalid/unavailable swept support, session control, or 12-bar order expiry.
5. On fill, apply adverse entry slippage and quantization. Stop is sweep low minus `0.05 * prior ATR14(15m)`. Reject a quantized stop distance outside 0.25%-2.00%. TP1 is the nearest opposing confirmed 15m swing-liquidity level and TP2 the nearest opposing confirmed 4H swing-liquidity level, each known before entry. Reject if candidate projected entry-to-TP2 R is below its sealed minimum.
6. After entry only, process stop before targets if one bar reaches both. Exit half at TP1, move the remainder to cost-adjusted break-even, and exit its half at TP2 or through the time/session stop.

### Short Pseudocode

1. Require `BEARISH` daily bias: daily close below EMA50 and EMA50 lower than five completed daily bars earlier.
2. Choose the nearest eligible confirmed 4H resistance above current price; require the sealed penetration, reclaim, and upper-wick conditions, then consume it only when qualifying.
3. Require a bearish MSS below a pre-existing confirmed fractal low, bearish displacement, and the first valid bearish FVG in the sealed timing window.
4. Activate one short limit at the bearish FVG lower boundary. The earliest possible fill is the next executable 5-minute bar. Before fill, mirror every long cancellation/invalidation rule.
5. Apply adverse entry slippage and quantization. Stop is sweep high plus `0.05 * prior ATR14(15m)`; the same 0.25%-2.00% bounds and candidate projected-R test apply. TP1 is the nearest opposing confirmed 15m swing-liquidity level below entry; TP2 is the nearest opposing confirmed 4H swing-liquidity level below entry, each existing before entry.
6. After entry, stop has precedence over targets; partial, break-even, time-stop, and session-force-flat rules mirror the long process.

## Frequency Guardrails

- Desired Phase-A frequency is 50-300 annualized executed trades. This is descriptive, not a selection score.
- Fewer than 50 annualized trades is `UNDERFREQUENCY_FAIL`; 50-300 is `TARGET_FREQUENCY`; more than 300 through 500 is `HIGH_FREQUENCY_WARNING`; more than 500 is `OVERFREQUENCY_FAIL`.
- The executor must report observed executed trades, annualized rate, and the applicable frequency classification for each candidate.

## Gates and Terminal Dispositions

Apply gates in this order: contract/phase lock; bar completeness; time controls; daily bias; 4H level eligibility; 15-minute sweep; 5-minute MSS; displacement; FVG; candidate registry; stop bounds; order activation; fill/exit controls. The earliest failing gate wins and cannot be overwritten.

Phase-A hard gates apply per candidate, exactly as follows:

- annualized frequency is within 50-500;
- positive net P&L after fees and slippage;
- net profit factor `>= 1.30`;
- positive average net R;
- maximum drawdown `<= 20R`;
- at least 8 profitable months and no more than 3 zero-trade months;
- no more than 35% of positive P&L from the best calendar month;
- no more than 30% of positive P&L from the best five trades;
- long and short each comprise at least 25% of executed trades, and neither direction has average net R below `-0.15R`;
- one additional price tick on entry and every exit remains positive;
- best-trade-removal sensitivity remains positive;
- bootstrap median mean net R is positive and the bootstrap 95% lower bound is at least `-0.025R`;
- at least 3 of 4 calendar subperiods in 2023 are non-negative; and
- immutable-artifact and full funnel reconciliation succeeds.

Freeze exactly one candidate only when exactly one candidate passes every hard gate. If zero or more than one candidate passes, report `PHASE_A_NO_ROBUST_CANDIDATE`; no ranking, discretionary tiebreaker, or selection is permitted and Phase B remains unopened. The locked Phase-B interval is executed once only after a freeze, and Alpha is out of scope for this design.

Each proposed setup reaches exactly one mutually exclusive terminal disposition: `TRADE_EXECUTED`, `DAILY_BIAS_REJECTED`, `NO_ACTIVE_HTF_LEVEL`, `DUPLICATE_LEVEL_BLOCKED`, `SWEEP_DEPTH_REJECTED`, `SWEEP_RECLAIM_REJECTED`, `SWEEP_WICK_REJECTED`, `MSS_WINDOW_EXPIRED`, `MSS_STRUCTURE_REJECTED`, `DISPLACEMENT_REJECTED`, `FVG_NOT_FORMED`, `FVG_TOO_SMALL`, `PENDING_ORDER_EXPIRED`, `PRE_ENTRY_SWEEP_INVALIDATED`, `BIAS_INVALIDATED_BEFORE_ENTRY`, `STOP_DISTANCE_REJECTED`, `PROJECTED_RR_REJECTED`, `ACTIVE_POSITION_BLOCKED`, `ACTIVE_ORDER_BLOCKED`, `SESSION_ENTRY_BLOCKED`, or `SESSION_ENDED`.

- `STOPPED`, `TP1_STOPPED`, `TP2_COMPLETED`, `FORCED_TIME_EXIT`, and `FORCED_END_OF_DATA_EXIT` are execution exit reasons attached to a `TRADE_EXECUTED` setup, never additional setup terminal dispositions.
- A mandatory control cancellation resolves to its applicable terminal disposition (`ACTIVE_POSITION_BLOCKED`, `ACTIVE_ORDER_BLOCKED`, `SESSION_ENTRY_BLOCKED`, or `SESSION_ENDED`); it must not create a generic extra state.
- Any identity, count, cash, quantity, or state mismatch is `RECONCILIATION_ERROR` and invalidates the run.

## Audit, IDs, and Event Reconciliation

Use deterministic IDs: `run_id`, `candidate_id`, `source_bar_id`, `derived_bar_id`, `level_id`, `sweep_id`, `setup_id`, `mss_id`, `displacement_id`, `fvg_id`, `order_id`, `position_id`, `fill_id`, `exit_id`, and `event_id`. Audit events are append-only and contain UTC time, processing sequence, parent/entity IDs, candidate ID, specification hash, inputs, decision, reason code, and before/after state.

Emit append-only audit records for aggregation confirmation, bias, level lifecycle, sweep, MSS, displacement, FVG, gate transitions, order state, fill, exit, disposition, and reconciliation. Every setup record stores the source bars and snapshots that supported its decision: daily EMA50 and its five-day slope, selected level ID/price/confirmation, 15m prior ATR and wick calculation, pre-MSS fractal identity, 5m prior ATR, FVG three bars/boundaries/width, candidate projected R, entry/stop/TP1/TP2, quantity and cost assumptions, and terminal/exit reason.

The sealed setup-scoped event lifecycle is `SETUP_PROPOSED`, optional `MSS_CONFIRMED`, optional `DISPLACEMENT_CONFIRMED`, optional `FVG_CONFIRMED`, optional `ORDER_ACTIVATED`, optional `ENTRY_FILLED`, zero-to-two `EXIT_FILLED`, and exactly one `TERMINAL_DISPOSITION`. Gate decisions and snapshots are fields on the applicable lifecycle event, not extra setup-scoped events. Therefore:

`setup_scoped_event_count = proposed_setups + mss_confirmed + displacement_confirmed + fvg_confirmed + order_activated + entry_fills + exit_fills + terminal_dispositions`.

An executed setup is one with an entry fill. A `recorded_exit_fill` is one immutable exit-fill record in the candidate trade/audit artifacts; therefore exit-event count must equal exit-fill record count. Reconciliation must prove the formula for every setup and prove the aggregate equals the sum of per-setup event counts.

For each candidate, reconciliation additionally proves: proposed-setup count equals the sum of the 21 terminal-disposition counts; `TRADE_EXECUTED` equals entries and `trades.json` count; each event points to an existing setup where applicable; each fill points to the applicable order/position; partial quantities sum to the original quantized quantity; TP1 quantity plus TP2 or stop remainder equals original quantity; realized gross/net P&L reconciles to fills, fees, and slippage; trade totals reconcile to monthly totals; and monthly totals reconcile to candidate and study totals.

## Immutable Artifacts

Required immutable artifacts: `sealed-specification.json`, `candidate-registry.json`, `data-manifest.json`, `derived-timeframe-manifest.json`, `configuration.json`, `levels.json`, `events.json`, `trades.json`, `setup_outcomes.json`, `monthly_metrics.json`, `report.json`, `gates.json`, `selection_report.json`, `freeze.json`, `integrity-manifest.json`, and `final_report.json`. Never overwrite, mutate, or merge a run directory.

Reconcile source bars to derived bars; levels to sweeps; sweeps to setups; setups to confirmations/orders; orders to fills/positions; positions to exits; exits to candidate totals; candidates to run totals; and the event formula to setup and exit-fill records.

Each `setup_outcomes.json` record contains `setup_id`, `level_id`, `sweep_id`, `mss_id` when formed, `fvg_id` when formed, direction, daily-bias snapshot, 4H-level snapshot, 15m sweep data, 5m MSS structure, FVG boundaries, entry price when activated, sweep extreme, stop, TP1 level/price, TP2 level/price, projected R, ordered event history, exactly one terminal disposition, and `trade_id` only for `TRADE_EXECUTED`.

## Pre-Implementation Checklist

Before any future implementation or run, verify all of the following without changing this seal:

- the sealed specification hash, candidate registry, Phase-A/Phase-B chronology, and unopened input-contract path;
- source 5-minute bar validity, UTC interval-start timestamps, closed-bar completeness, and deterministic in-memory 15m/4h/daily aggregation with no derived-data cache;
- confirmed-only fractal availability, prior-only ATR/EMA inputs, and no access to a right-side bar before its completion;
- deterministic IDs, order/position exclusivity, candidate isolation, exact costs/quantization, and session/holding controls;
- all 21 terminal dispositions, setup/event/fill/quantity/P&L reconciliation, required artifacts, hash manifest, collision rejection, and fail-closed errors;
- every hard Phase-A gate, zero/multiple-pass handling, and Phase-B lock; and
- synthetic causality, long/short, partial-exit, same-bar ambiguity, time-control, and artifact-integrity regression tests before any real run.

## Unresolved Decisions

None. Any behavior not expressly sealed here, including source-manifest validation details, exact repository cost constants, and output-path implementation choices, requires a new sealed version before coding; it may not be silently inferred from a historical study.
