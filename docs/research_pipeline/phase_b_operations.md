# Phase B operations

Initialize the Phase A registry:

```powershell
python -m research_pipeline init
```

Start a default dry run. The helper uses the installed Smithers CLI with argument-safe Windows process handling. The JSON is intentionally fictional and does not run Codex, tests, backtests, or optimization:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\research_pipeline\phase_b_dry_run.ps1
```

For raw CLI invocation on shells that preserve JSON quotes, use `smithers up ... --input '<JSON>'`. On Windows PowerShell, `--input -` with a correctly redirected stdin or the helper above avoids native argument quoting issues.

The command prints a run ID. Monitor and inspect it with:

```powershell
& .smithers\node_modules\.bin\smithers.exe ps
& .smithers\node_modules\.bin\smithers.exe inspect <RUN_ID>
& .smithers\node_modules\.bin\smithers.exe logs <RUN_ID>
```

At the approval pause, approve or reject the one node:

```powershell
& .smithers\node_modules\.bin\smithers.exe approve <RUN_ID> --node approve-spec --by operator
& .smithers\node_modules\.bin\smithers.exe deny <RUN_ID> --node approve-spec --by operator --reason "specification needs revision"
```

Resume an interrupted run:

```powershell
& .smithers\node_modules\.bin\smithers.exe up .smithers\workflows\trading-research-phase-b.tsx --run-id <RUN_ID> --resume true
```

Retry a failed node, then resume:

```powershell
& .smithers\node_modules\.bin\smithers.exe retry-task <RUN_ID> --node <NODE_ID>
& .smithers\node_modules\.bin\smithers.exe up .smithers\workflows\trading-research-phase-b.tsx --run-id <RUN_ID> --resume true
```

Cancel and inspect Python state:

```powershell
& .smithers\node_modules\.bin\smithers.exe cancel <RUN_ID>
python -m research_pipeline list-strategies
python -m research_pipeline status <STRATEGY_ID>
python -m research_pipeline history <STRATEGY_ID>
```

Smithers artifacts remain under `.smithers/` according to the installed local store. Python drafts, reports, and SQLite state are under `research_registry/`.
# Phase B operations

The bridge transition ownership is intentionally narrow: `register-spec` enters `WAITING_FOR_SPEC_APPROVAL`, `apply-approval` enters `IMPLEMENTATION`, and `technical-verification` enters `IMPLEMENTATION_VERIFICATION`. Planning, Codex execution, and test nodes do not advance those phases. Repeated approval application is safe and preserves the already-completed state.

## Dry runs

Interactive dry runs use `scripts/research_pipeline/phase_b_dry_run.ps1`. Each run has a unique strategy and registry, pauses at the real approval gate, and prints the exact `smithers approve` and resume commands. Run those commands only after inspecting the specification. `-AutomatedTest` is a separate fixture mode using a temporary registry and no resumable Smithers run; it cannot modify the interactive registry.

After editing the workflow, start a new run. The historical failed run remains unchanged for audit. Archive or ignore failed records in the Smithers run store/export rather than repairing their history.
