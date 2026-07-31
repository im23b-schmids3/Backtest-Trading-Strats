# Configurable compliance and realistic execution layer

The research pipeline now includes an opt-in, provider-independent compliance
package at `src/research_pipeline/compliance/`. It is a decision and evidence
layer, not a broker integration and not a guarantee of prop-firm compliance.

`PropFirmPolicy` contains the firm, account type, evaluation state, policy
version, effective date, IANA timezone, evidence references, news/session/daily
loss/position/holding-time settings, and automation mode. The shipped
`unconfigured_policy()` has no firm-specific rules enabled and defaults to
`RESEARCH_ONLY`. No current Alpha Futures or other firm rule is copied into the
generic profile. The existing Alpha-specific Phase D fixtures remain separate.

The shared `ComplianceEvaluator` is used by the optional native backtest facade
and the alert facade. It combines news, session, daily-loss, position-limit, and
automation checks into a hashed `ComplianceDecision`. Research-only runs may
continue with explicit `DATA_UNAVAILABLE` or `DATA_STALE` classifications; alert,
semi-automated, and automated paths fail safe with manual review when required
evidence is unavailable.

Economic calendar data is supplied through `EconomicCalendarProvider`. The
fixture provider is deterministic and offline. Saved calendar artifacts contain
event data, source-data hashes, retrieval timestamps, and an artifact hash.
There is no live provider in this change.

`ExecutionCostConfig` requires instrument-specific tick values and fees. The
engine calculates commissions, exchange/regulatory fees, order-type slippage,
and configured multipliers exactly once and emits a configuration hash. It does
not silently replace the legacy backtester; callers must opt in to the new cost
engine, so historical execution semantics remain unchanged.

SQLite now persists policy profiles, compliance decisions, calendar/cost
artifacts, and compliance events under the additive compliance schema tables.
The exact policy and execution configuration hashes can therefore be attached
to future backtest reports without rewriting historical outputs.

All firm-specific values remain configuration. Before enabling a profile for a
real account, review current first-party documentation, effective dates,
account-type scope, time zone, and unresolved ambiguities.
