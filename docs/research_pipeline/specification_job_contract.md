# Specification job contract

Each candidate attempt has one immutable job request:

`research_runs/<strategy_id>/<run_id>/specification/jobs/job-001/request.json`

Repair attempts use `job-002` and `job-003`. The request includes the intake,
schema contract, prompt, prior invalid draft and validation reports where
applicable, all input hashes, the strategy identity, attempt number, and the
read-only execution requirements. The SQLite registry stores the same request
hash and job status.

Results are written beside the request in `result/`. `completion.json` is the
typed completion contract. A completion is accepted only when its identity,
result hash, and every listed artifact hash match. A stale, copied, or
cross-run completion is rejected.

There is one initial candidate and at most two repair candidates. Invalid
structured output is a candidate failure and may create the next repair job;
Codex process failure, input-integrity failure, or artifact-integrity failure
does not silently become a valid candidate. The Phase B Pydantic and semantic
validators remain the authority after extraction.
