# Phase C architecture

Phase C is a deterministic research policy engine behind a durable Smithers
workflow. Smithers supplies task durability and role boundaries; Python owns
state transitions, budgets, split hashes, citations, holdout locking, and
classification. `StrategyResearchAdapter` is the only execution boundary.

The synthetic adapter is the Phase C fixture. A production adapter must emit
machine-readable metrics, hashes, process results, and a Phase B.5 diagnostic
manifest. Phase C never calls a market-data provider directly and does not
change the existing backtester.

Artifacts live under `research_runs/<strategy_id>/<version>/` and registry
records are in SQLite. Baseline, parameter, walk-forward, holdout, stress,
and throughput evidence are journaled before downstream decisions consume it.
