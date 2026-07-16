# Approval workflow

There is exactly one normal mandatory approval node: `approve-spec`. It appears after deterministic validation and SQLite registration and before implementation planning. Its summary contains the strategy identity, hypothesis, markets, timeframes, rules, stops, exits, baseline parameters, mutable families, invariants, assumptions, ambiguities, specification path, and hash.

The only decisions are `APPROVE` and `REJECT`. Approval calls the Phase A controller and verifies the approved specification is immutable before entering `IMPLEMENTATION`. Rejection records the note, transitions the strategy to `REJECTED`, and never creates an implementation plan.

Material ambiguity is a separate deterministic stop: the workflow records `MANUAL_REVIEW_REQUIRED` without requesting the normal approval. Phase B adds no hidden or second mandatory approval.
