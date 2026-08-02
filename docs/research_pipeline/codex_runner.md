# Codex runner

`src/research_pipeline/runners/codex_runner.py` is the Python adapter for local Codex execution. It accepts a prompt and working directory, builds an argument list (never a shell command), sets a timeout, captures stdout/stderr and exit status, and returns `CodexExecutionResult`.

On Windows resolution is: `CODEX_EXECUTABLE`, `codex`, `codex.cmd`, `%APPDATA%\npm\codex.cmd`, then a verified discovery result. `.ps1` is not selected for direct process spawning when a `.cmd`/real executable is available. Override it without putting a user-specific path in source:

```powershell
$env:CODEX_EXECUTABLE = "$env:APPDATA\npm\codex.cmd"
```

Run `py -m research_pipeline workflow diagnose-tools` for redacted structured JSON diagnostics. The output includes only executable paths and whether `codex --version` succeeded; it never includes credentials.

Dry-run mode records the intended command and starts no subprocess. Missing executable, timeout, operating-system failure, and nonzero exit are typed result states. Output and command fields pass through the local redactor for common bearer, API-key, token, and `sk-` patterns.

The runner never selects unrestricted execution automatically. The implementation task is the only task allowed to request workspace-write, and the Python service verifies protected paths in the resulting diff. Codex prose is not used as test evidence; `DeterministicTestRunner` uses process exit status and parsed pytest output.
