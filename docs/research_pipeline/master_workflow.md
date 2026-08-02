# Master Smithers workflow

The workflow is `.smithers/workflows/trading-research-master.tsx`. It uses one
typed Smithers approval node and calls the Python master bridge with structured
JSON. Smithers persists task outputs and can resume the run; the Python registry
persists phase evidence and remains authoritative for deterministic state.

Validate it with `bunx smithers-orchestrator graph
.smithers/workflows/trading-research-master.tsx`.
