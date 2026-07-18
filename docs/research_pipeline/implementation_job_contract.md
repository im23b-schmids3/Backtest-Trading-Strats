# Implementation job contract

Each approved real-mode implementation is stored at:

`research_runs/<strategy>/<run>/implementation/jobs/job-001/`

The request contains the run and strategy identity, approved specification
hash, base commit, repository root, isolated worktree, allowed and forbidden
paths, required tests, timeout, model configuration, provenance, and hashes of
the prompt and manifests. It does not contain credentials.

The executor writes `result/implementation_manifest.json`,
`changed_files.json`, `scope_validation.json`, `test_results.json`,
`codex_invocation.json`, `output_hash_manifest.json`, and `completion.json`.
The completion is accepted only when identity, input hashes, output hashes,
and the canonical completion hash all match. Repeating a successful run
returns the existing completion instead of creating a second worktree or job.
