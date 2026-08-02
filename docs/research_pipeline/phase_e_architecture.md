# Phase E architecture

Phase E is a deterministic multi-strategy portfolio layer over frozen Phase C
candidates and Phase D account evidence. `PortfolioService` owns eligibility,
candidate enumeration, causal signal merging, overlap/correlation analysis,
shared-account replay, ablation, stress, and classification. SQLite is the
durable audit registry; JSON artifacts under `research_runs/portfolios/` hold
large signal streams.

The phase deliberately has its own portfolio state machine and tables. It does
not change the Phase A-D single-strategy state machine or reopen holdout data.
The Smithers graph only sequences typed Python bridge calls and reviewer
artifacts; it does not authorize backtests or parameter optimization.
