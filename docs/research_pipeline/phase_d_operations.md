# Phase D operations

Initialize and inspect the registry:

```text
python -m research_pipeline init
python -m research_pipeline prop status STRATEGY_ID
python -m research_pipeline prop journal STRATEGY_ID
```

Run deterministic phases:

```text
python -m research_pipeline prop start STRATEGY_ID
python -m research_pipeline prop verify-rules STRATEGY_ID --product "Alpha Futures Zero 25K"
python -m research_pipeline prop verify-contracts STRATEGY_ID
python -m research_pipeline prop reconcile STRATEGY_ID
python -m research_pipeline prop run-risk STRATEGY_ID
python -m research_pipeline prop run-scenarios STRATEGY_ID
python -m research_pipeline prop final-review STRATEGY_ID
```

Synthetic fixtures use `python -m research_pipeline prop dry-run ID --scenario
SCENARIO`. Scenarios are `profitable`, `negative-economics`,
`high-pass-zero-payout`, `unsupported-mapping`, and `noncompliant`.

The Smithers graph is checked with `bunx smithers-orchestrator graph
.smithers/workflows/trading-research-phase-d.tsx`. A run is started with
`bunx smithers-orchestrator up .smithers/workflows/trading-research-phase-d.tsx
--detach --input <JSON>`, inspected with `bunx smithers-orchestrator inspect
<RUN_ID> --format json`, resumed with `... up ... --run-id <RUN_ID> --resume
true`, and cancelled with `bunx smithers-orchestrator cancel <RUN_ID>`.
