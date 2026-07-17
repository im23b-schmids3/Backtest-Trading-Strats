# Real Phase C execution

Real Phase C receives the real adapter through the existing `PhaseCService`.
It uses the approved split and parameter families only, consumes the existing
budgets, evaluates one family per round, freezes the candidate, and opens the
holdout once. No parameter grid is invented when no mutable family is declared.

B.5 must be `VERIFIED` before baseline research begins.
