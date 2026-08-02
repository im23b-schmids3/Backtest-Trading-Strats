# Budgets and gates

Default hard budgets are six parameter families, three rounds per family,
five values per round, 500 total backtests, three repair attempts, 180 runtime
minutes per phase, 100 MB of reports, and one holdout access. Limits and usage
are stored separately in SQLite. Budget checks happen before usage is written,
so an overrun has no partial consumption.

Gates are configurable records with a metric, threshold, comparison, source
file, and an outcome. Baseline, validation, holdout, and throughput examples
are supplied in `configs/research_pipeline/defaults.yaml`; they are examples,
not universal claims about trading quality. Missing or non-numeric evidence is
`INSUFFICIENT_EVIDENCE`.

Configuration files may add a `strategies` mapping keyed by `strategy_id` with
`budgets` and `gates` entries. Those values override the defaults only for that
strategy when loaded by the CLI or `load_pipeline_config(..., strategy_id=...)`.
