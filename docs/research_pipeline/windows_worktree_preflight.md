# Windows worktree preflight

Phase F2 performs a deterministic repository scan before creating an isolated
implementation worktree. It checks tracked names for Windows-invalid
characters, reserved device names, trailing spaces or periods, case-folding
collisions, long anticipated checkout paths, and tracked runtime artifacts.

Run it with:

```text
py -m research_pipeline repository worktree-preflight --format json
py -m research_pipeline repository worktree-preflight --probe --format json
```

An unsafe report exits with code `3`. The command never deletes or untracks a
file. For generated output, the report may recommend `git rm --cached --
<path>`; review that recommendation manually. Probe worktrees use a
deterministic repository-local temporary path and only their own path is
removed. Cleanup failures make the report unsafe.

Reports are stored under `research_registry/worktree_preflight/` when the
repository is writable.
