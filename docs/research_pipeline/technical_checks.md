# Technical checks

The runner implements trade PnL, partial exits, quantity scaling, fee reconciliation, trade-count semantics, causality, session/terminal boundaries, report totals, source/proxy labeling, and deterministic replay checks. Prop lifecycle checks activate only when `prop` is listed in `applicable_capabilities`.

Numerical checks use strict absolute and relative tolerances from the manifest. Order versions cannot substitute for completed positions, and undocumented proxy/synthetic data is a manual-review blocker.
