# Phase D classification

The deterministic classifier uses compliance, futures-data confidence, payout
evidence, external cash flow, and the persisted Phase C classification. It
emits `PROP_ACCEPTED_STANDALONE`, `PROP_ACCEPTED_PORTFOLIO_COMPONENT`,
`OWN_CAPITAL_ONLY`, `REJECTED_PROP_INCOMPATIBLE`,
`REJECTED_NEGATIVE_ECONOMICS`, `INSUFFICIENT_FUTURES_DATA`, or
`INSUFFICIENT_PROP_EVIDENCE`. Synthetic/proxy fixtures cannot claim native
futures deployment readiness.
