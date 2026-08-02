# Specification failure handling

There are three distinct failure classes:

1. `CODEX_EXECUTION_FAILURE`: the local Codex process was unavailable, timed out, or returned a nonzero result.
2. `STRUCTURED_OUTPUT_INVALID` or a semantic/Pydantic issue: Codex returned a payload that could not become an approval-ready specification.
3. `SPECIFICATION_GENERATION_FAILURE`: the bounded attempt budget was exhausted without a valid canonical specification.

All raw drafts and reports are kept as local artifacts; SQLite stores compact paths, hashes, summaries, ambiguity rows, and failure metadata. Secret redaction is applied to Codex command/output artifacts. Real mode never searches for or silently reuses a repository fixture. The only prebuilt specification path is an explicit caller option used by compatibility tests and must be visible in the workflow input.

The Smithers master graph exposes `master-start`, specification status, a bounded retry/status branch, and the single approval gate. Smithers output fields use `source_run_id` because `run_id`, `node_id`, and `iteration` are reserved persistence columns.
