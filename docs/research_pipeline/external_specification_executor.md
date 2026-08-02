# External specification executor

The specification-generation boundary is intentionally split from Smithers.
Smithers creates a durable job and pauses when the tenant cannot execute the
local authenticated Codex CLI. The operator runs that job from the primary
checkout, then resumes the Smithers run. This is a read-only operation: Codex
uses `--sandbox read-only`, receives the prompt on stdin, and may emit only one
structured YAML/JSON mapping.

The executor verifies every input hash before execution and records the Codex
invocation, raw output, extraction report, output manifest, completion record,
and pause signal under `research_runs/<strategy>/<run>/specification/jobs/`.
Repository mutations observed during the invocation produce
`FAILED_ARTIFACT_INTEGRITY`; existing dirty files are preserved and are not
treated as executor mutations.

The executor never approves a specification and never runs implementation,
backtests, optimization, or research. A successful completion is ingested by
the deterministic Phase B service on Smithers resume.
