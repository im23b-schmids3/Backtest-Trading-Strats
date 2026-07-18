# External Codex executor

Restricted Smithers runs must not call an authenticated Codex CLI. After the
single strategy approval, the Python controller creates an immutable external
implementation job. The operator runs it from the primary repository with:

```text
py -m research_pipeline codex-executor run <RUN_ID>
py -m research_pipeline codex-executor status <RUN_ID>
py -m research_pipeline implementation ingest <RUN_ID>
```

The executor verifies the request and input hashes, repeats worktree
preflight, creates or reuses the run-specific Git worktree, sends the prompt
over stdin to `codex exec --sandbox workspace-write`, applies a timeout, and
records process output and actual pytest results. It never uses unrestricted
execution and never advances Phase C or later.

If the local Codex executable is unavailable, times out, exits nonzero, or
produces an integrity/scope/test failure, the job records a typed completion
status. A successful completion is ingested exactly once by the controller.
