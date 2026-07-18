# Smithers external execution handoff

The local Smithers workflow is `trading-research-master` in
`.smithers/workflows/trading-research-master.tsx`. In real mode it pauses at
the specification boundary with outcome
`WAITING_EXTERNAL_SPECIFICATION_GENERATION` or
`WAITING_EXTERNAL_SPECIFICATION_REPAIR`, or at the implementation boundary
with outcome `WAITING_EXTERNAL_CODEX`. The Python status includes the exact
executor command and job path.

The installed Smithers `0.28.0` commands are:

```text
cd .smithers
smithers workflow run trading-research-master --detach --input '{"intake_path":"<INTAKE_YAML>","repository_root":"<REPO>","registry_path":"<REGISTRY>","mode":"real_run","dry_run":false,"implementation_enabled":true}'
smithers inspect <SMITHERS_RUN_ID> --format json
smithers logs <SMITHERS_RUN_ID> --tail 100 --format json
py -m research_pipeline specification-executor inspect <PYTHON_RUN_ID>
py -m research_pipeline specification-executor run <PYTHON_RUN_ID>
smithers approve <SMITHERS_RUN_ID> --node specification-approval --note "approved"
smithers deny <SMITHERS_RUN_ID> --node specification-approval --note "rejected"
smithers workflow run trading-research-master --resume <SMITHERS_RUN_ID> --detach
```

Use `smithers workflow list` to confirm discovery and `smithers graph
.smithers/workflows/trading-research-master.tsx` to inspect the graph without
executing it.

Run the specification or implementation command outside the restricted
Smithers task sandbox. The specification executor signals the waiting run
with `external.codex.specification.completed`; the implementation executor
uses `external.codex.completed`. Both use the Python run ID as correlation ID.
Smithers then calls `master-resume`; the controller ingests and verifies the
result before approval or B.5 as appropriate. There is no second strategy
approval.

If Codex is denied by tenant or sandbox policy during specification intake,
the controller creates a durable external specification job, does not consume
a generation or repair attempt, and remains paused. Phase F2 does not silently
weaken that policy or claim a successful pipeline run.
