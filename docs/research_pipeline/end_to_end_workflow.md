# End-to-end workflow

The canonical order is intake → specification → approval → implementation →
implementation verification → B.5 technical verification → baseline/research →
walk-forward → holdout → stress → prop → portfolio → report → archive.

`run` creates the specification and pauses at approval. `approve` applies the
single immutable specification approval. `resume` executes the remaining
deterministic phases and safely reuses completed phase records after an
interruption.
