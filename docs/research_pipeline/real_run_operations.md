# Real-run operations

1. Run the repository preflight and resolve unsafe tracked paths in a clean
   clone. Do not delete historical outputs automatically.
2. Start the Smithers workflow with `smithers workflow run
   trading-research-master --detach --input '<JSON>'`; record its run ID and
   monitor with `smithers inspect <SMITHERS_RUN_ID> --format json`.
3. Approve or reject with `smithers approve <SMITHERS_RUN_ID> --node
   specification-approval` or `smithers deny ...`. Rejection prevents
   implementation.
4. When the run reports `WAITING_EXTERNAL_SPECIFICATION_GENERATION` or
   `WAITING_EXTERNAL_SPECIFICATION_REPAIR`, inspect the job and run the printed
   `py -m research_pipeline specification-executor run <RUN_ID>` command from
   the primary checkout. The command uses the locally authenticated Codex CLI
   in read-only mode.
5. When the run reports `WAITING_EXTERNAL_CODEX`, inspect the immutable
   implementation job and run the printed `py -m research_pipeline
   codex-executor run <RUN_ID>` command from the primary checkout.
6. Monitor specification jobs with `py -m research_pipeline
   specification-executor status <RUN_ID>` or inspect all request and result
   artifacts with `py -m research_pipeline specification-executor inspect
   <RUN_ID>`. Monitor implementation with
   `py -m research_pipeline codex-executor status <RUN_ID>` and
   `py -m research_pipeline status <RUN_ID>`.
7. Resume Smithers with `smithers workflow run trading-research-master
   --resume <SMITHERS_RUN_ID> --detach` after a successful completion. If the signal is unavailable,
   run the same Smithers resume command after checking the Python status; the
   bridge performs the identity and hash-checked ingestion. For a direct
   controller run, `py -m research_pipeline specification-executor resume
   <RUN_ID>` reruns the pending external job without creating a duplicate.

The executor worktree is never merged automatically. Cleanup requires an
explicit operator action using Git worktree commands after artifacts have been
archived.
