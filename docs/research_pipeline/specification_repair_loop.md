# Specification repair loop

Specification generation has a bounded candidate loop: one generation attempt
followed by no more than two repairs. Each repair receives the exact prior
invalid draft and its structured validation reports. The prompt requires one
complete `StrategySpec` mapping and prohibits invented rules, prose, code,
backtests, and optimization.

For a tenant-compatible external run, the controller creates a new durable
repair job and pauses again. The operator runs the executor for that job and
resumes the same Smithers run. A valid candidate proceeds to the single human
approval gate. If the repair budget is exhausted, the controller records
`SPECIFICATION_GENERATION_FAILURE`; it does not guess or approve the last
invalid output. Material ambiguity remains `MANUAL_REVIEW_REQUIRED`.

The attempt count is stored in both SQLite and the artifact tree. Repeating a
completed executor job is idempotent and does not invoke Codex again; repeating
a pending job does not create a duplicate job for the same run and attempt.
