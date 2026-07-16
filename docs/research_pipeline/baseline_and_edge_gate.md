# Baseline and edge gate

The baseline uses exactly the approved specification baseline parameters and
the persisted chronological split. It must pass Phase B.5 verification before
the edge gate evaluates metrics. The configured baseline gates are thresholds,
not universal trading truths; strategy-level configuration can replace them.

The edge gate returns `CONTINUE`, `REJECT`, `INSUFFICIENT_EVIDENCE`, or
`MANUAL_REVIEW_REQUIRED`. Low sample support is classified as insufficient
evidence before a performance failure is treated as a no-edge rejection.
