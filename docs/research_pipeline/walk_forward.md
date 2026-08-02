# Walk-forward validation

Walk-forward runs only after a candidate manifest is frozen. The candidate
contains the approved specification hash, split hash, selected parameters,
research decisions, and budget usage. A failed walk-forward rejects the
candidate and prevents holdout access. Validation evidence is persisted with
fold metrics and a verification outcome.
