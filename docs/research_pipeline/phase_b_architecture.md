# Phase B architecture

Phase B is the durable orchestration layer above the deterministic Phase A controller. Smithers owns task lifecycle, the single approval gate, retries, and resume. Python owns the canonical strategy schema, state transitions, registry writes, worktree checks, and process-result contracts.

The workflow is `.smithers/workflows/trading-research-phase-b.tsx` and is named `trading-research-phase-b`. Tasks call `python -m research_pipeline workflow ...` with JSON input and consume only validated JSON output. The strategy-spec task uses the local Codex CLI in read-only mode. Implementation uses the Python worktree manager and Codex workspace-write mode only when both `dry_run=false` and `implementation_enabled=true`.

The normal path is:

`generate-spec -> validate -> register -> approve-spec -> apply-approval -> implementation-plan -> implement-strategy -> tests -> bounded repair loop -> IMPLEMENTATION_VERIFICATION`

Phase B deliberately stops before baseline backtesting, edge gates, parameter research, holdout access, and optimization.

## Durability and reconciliation

Smithers persists task outputs and approval decisions in its local run store. The Python registry persists strategy state, transitions, experiments, and failures in SQLite. Bridge calls are idempotent for draft registration and deterministic experiment IDs. On resume, completed Smithers nodes are reused and Python checks the current registry state before applying a transition.

The bridge does not expose Smithers internals or use SQLite as a substitute for Smithers state. A final summary reports whether the two state views reconciled.
