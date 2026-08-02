# Smithers setup

Smithers is project-local under `.smithers/`. The installed version is pinned by `.smithers/package.json` and currently provides the executable `.smithers/node_modules/.bin/smithers.exe` on Windows.

From the repository root:

```powershell
& .smithers\node_modules\.bin\smithers.exe graph .smithers\workflows\trading-research-phase-b.tsx
```

The graph command is a safe API compatibility check. It performs no strategy work.

The workflow uses Bun's argument-array process API to call Python. It does not use shell concatenation. Python then invokes the resolved Codex executable with `codex exec` using `--sandbox read-only` for specification work and `--sandbox workspace-write --ask-for-approval never` inside the isolated worktree for implementation. Set `CODEX_EXECUTABLE` when Bun's PATH cannot see npm-global binaries.

After workflow source changes, start a fresh Smithers run; do not resume an old failed run. Use `py -m research_pipeline workflow diagnose-tools` before a run. A failed run is retained as an audit artifact and may be archived/ignored by moving its exported record outside the active run directory; never mutate its persisted history.

No OpenAI API key is required by the workflow. Codex authentication is supplied by the user's existing local Codex CLI login/configuration; credentials are never put into prompts, command arguments, SQLite, or logs.
