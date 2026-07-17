# Phase E operations

Initialize with `python -m research_pipeline init`, then create a YAML portfolio
with `python -m research_pipeline portfolio create CONFIG.yaml`. Run the
commands in order: `generate-candidates`, `merge-signals`, `analyze-overlap`,
`analyze-correlation`, `run-risk`, `run-prop`, `run-ablation`, `run-stress`,
and `final-review`. `status` and `journal` are safe to use after interruption;
completed records are reused. Registry state is in
`research_registry/research_pipeline.sqlite3`; signal and review artifacts are
under `research_runs/portfolios/`.

Validate the graph with `bunx smithers-orchestrator graph .smithers/workflows/trading-research-phase-e.tsx`. Start with `bunx smithers-orchestrator up .smithers/workflows/trading-research-phase-e.tsx --input '<JSON>' --detach --format json`, monitor with `bunx smithers-orchestrator monitor RUN_ID`, inspect with `bunx smithers-orchestrator inspect RUN_ID --format json`, and resume with `bunx smithers-orchestrator up .smithers/workflows/trading-research-phase-e.tsx --run-id RUN_ID --resume RUN_ID --format json`. Deterministic portfolio state remains authoritative in SQLite.
