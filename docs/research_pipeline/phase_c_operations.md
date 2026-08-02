# Phase C operations

Create and inspect a deterministic fixture:

```powershell
python -m research_pipeline --registry research_registry/phase-c-dry-run.sqlite3 research dry-run --strategy-id phase-c-dry-run --scenario strong-stable --repository-root . --registry-path research_registry/phase-c-dry-run.sqlite3
python -m research_pipeline --registry research_registry/phase-c-dry-run.sqlite3 research status phase-c-dry-run
python -m research_pipeline --registry research_registry/phase-c-dry-run.sqlite3 research journal phase-c-dry-run
```

Start the durable Smithers synthetic workflow with Windows-safe argument-array
handling:

```powershell
python scripts\research_pipeline\phase_c_smithers_dry_run.py --detach
```

Inspect, monitor, resume, retry, or cancel the run using the installed CLI:

```powershell
& .smithers\node_modules\.bin\smithers.exe ps
& .smithers\node_modules\.bin\smithers.exe inspect phase-c-smithers-dry-run
& .smithers\node_modules\.bin\smithers.exe monitor phase-c-smithers-dry-run
& .smithers\node_modules\.bin\smithers.exe up .smithers\workflows\trading-research-phase-c.tsx --run-id phase-c-smithers-dry-run --resume true
& .smithers\node_modules\.bin\smithers.exe retry-task phase-c-smithers-dry-run --node <NODE_ID>
& .smithers\node_modules\.bin\smithers.exe cancel phase-c-smithers-dry-run
```

The helper always supplies `dry_run=true`; it runs only the synthetic adapter
and never edits a strategy module.

The Smithers graph is checked with:

```powershell
& .smithers\node_modules\.bin\smithers.exe graph .smithers\workflows\trading-research-phase-c.tsx
```

Use a separate registry for each dry run. Do not point the fixture at the
historical research database. Production adapters must be reviewed before
they are enabled and must not access untouched holdout data early.
