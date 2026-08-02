# Phase D entry requirements

The controller requires a Phase C final classification of
`ACCEPTED_STANDALONE` or `ACCEPTED_PORTFOLIO_COMPONENT`, a frozen candidate
with a hash and persisted manifest, frozen parameters, a persisted split,
holdout access count no greater than one, and a verified Phase B.5 result.
Any mismatch blocks the prop run before rules or sizing execute.

Phase D owns a separate `PropPhase` run state and cannot reopen Phase C's
terminal `ACCEPTED` state. It does not combine strategies, optimize parameters,
or access untouched holdout data.
