# CMEOrderflowAbsorption.ES_V1_BACKTEST — OOS contract plan

## Status and chronology

This is the first sealed OOS backtest design contract, not an engine or a backtest. The July 20–31, 2026 pilot is frozen development/design evidence only and can never be used to claim OOS profitability. Validation starts no earlier than `2026-08-03T00:00:00Z`; its final end date remains unset until a separately sealed validation-data manifest records it.

## Future data manifest (do not acquire now)

The future manifest must identify exact ES CME contract symbols/instrument IDs, rollover policy, dataset/provider/schema version, file paths, hashes, record counts, UTC coverage, gaps, session calendar and multiplier/tick metadata. It must cover ordered ES MBO book events and executed trades, bid/ask depth for one-contract marketability, all required prior-RTH data, eligible RTHs, and the cutoff observation. Missing, stale, crossed/invalid, or unreconstructable market state is fail-closed.

## Frozen causal decision

Load `absorption_p95 = 0.7977986403366786` and `replenishment_p95 = 0.7785691162188411` literally. Never derive distributions, p95s, calibrations, or thresholds from validation data. A candidate is tradable only when that same completed interaction is both HIGH and STRONG at a required structural level; response fields are prohibited from the decision path. `BUYER_ABSORPTION` is LONG and `SELLER_ABSORPTION` is SHORT.

## Execution and risk design

Use only `SINGLE_2R`. After 1 ms decision latency and 1 ms order latency, take the first valid executable observation strictly after `interaction_end`: marketable long at ask or short at bid. `ENTRY_SLIPPAGE=1 adverse ES tick`: long entry fill is ask + 0.25 and short entry fill is bid - 0.25. `EXIT_SLIPPAGE=1 adverse ES tick`: long exit fill is bid - 0.25 and short exit fill is ask + 0.25. ES is 0.25 points/tick, $12.50/tick, and $50/point. Commission is a conservative research assumption of $3.00 per contract per side. No midpoint, OHLC, interpolation, or queue-fill inference is allowed.

Set the stop before entry: long below completed zone low minus one tick, short above completed zone high plus one tick. Reject missing/invalid/non-positive stop distances. Target exactly 2R. Resolve exits from chronologically ordered MBO events only and fail closed on an ambiguous collision. Before any validation data is processed, the immutable future run manifest must predeclare `FIXED_USD_RISK_BUDGET`. `entry_fill` already includes the one adverse entry tick and `stop_exit_fill_assumption` already includes the one adverse exit tick. Compute `one_contract_price_risk_usd = abs(entry_fill - stop_exit_fill_assumption) * 50.00`, `round_trip_commission_risk_usd = 2 * 3.00`, and `one_contract_initial_risk_usd = one_contract_price_risk_usd + round_trip_commission_risk_usd`; then `contracts = floor(FIXED_USD_RISK_BUDGET / one_contract_initial_risk_usd)`. Do not add entry or exit slippage again as a USD sizing-denominator component. Contracts must be a positive integer; entered trades have `contracts >= 1`; if fewer than one, do not trade and record `INSUFFICIENT_RISK_BUDGET_FOR_ONE_ES_CONTRACT`. No fractional ES, MES replacement, risk-budget overrun, or compounding. Report raw price risk, slippage contribution, commissions, and net economics separately. Reconcile `raw_price_risk_usd + slippage_contribution_usd = one_contract_price_risk_usd` and `one_contract_price_risk_usd + round_trip_commission_risk_usd = one_contract_initial_risk_usd`; the reported slippage contribution must not be added again to initial risk.

## Session and state machine

Allow no new entry at or after 22:45:00.000000000 UTC. Cancel all pending orders no later than cutoff. Force-flat every open position by selecting the LAST valid executable ES market observation in the inclusive `22:44:59.000000000 through 22:45:00.000000000 UTC` liquidation window and applying `EXIT_SLIPPAGE=1 adverse ES tick` plus the $3.00 per-contract-per-side exit commission. If the window has no valid executable observation, fail closed with `CUTOFF_EXECUTION_INTEGRITY_FAILURE`; never invent or timestamp a fill after cutoff. No position may exist at any timestamp after 22:45:00.000000000 UTC. Reconcile maximum exit timestamp <= cutoff, zero positions and pending orders after cutoff, and explicitly counted cutoff integrity failures. Eligibility resets at 00:00:00.000000000 UTC. Permit one position only; ignore opposite, overlapping, and repeated-level signals while open; each interaction ID can be submitted once only. Explicit audit outcomes include latency not reached, stale/absent executable observation, invalid spread, insufficient displayed size, book failure, unexecutable order, cutoff cancellation, cutoff integrity failure, open position, duplicate interaction, invalid stop, and insufficient one-contract risk budget.

## Future implementation sequence (not performed here)

1. Seal the validation-data manifest and `FIXED_USD_RISK_BUDGET` before acquiring or processing validation data or running any engine.
2. Implement a read-only causal event processor that imports this contract and preserves immutable event/trade/audit records.
3. Enforce the contract gates, then generate R-first reporting and separate one-contract ES economics.
4. Reconcile aggregate, direction, level, and RTH-date totals against the immutable ledger. Do not tune thresholds, exits, or parameters from validation results.

## Required contract-only tests

Fixtures must prove literal frozen-threshold loading; PLUS-only selection with no response field; entry timestamp after `interaction_end`; forced exit at or before cutoff; post-cutoff forced-exit rejection; zero position/pending order after cutoff; missing valid liquidation-window observation fails closed; exact integer sizing from fill-inclusive price risk plus two commissions; insufficient budget; no fractional contracts; exact one-tick entry/exit slippage; no slippage double-counting; causal stop/target event ordering without OHLC; and each stated reconciliation identity. These tests exercise only pure synthetic contract fixtures and no data loader or backtest engine.
