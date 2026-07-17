# Codex implementation contract

Real Codex implementation runs use `codex exec` with argument arrays,
`workspace-write`, bounded timeouts, and an isolated worktree. Prompts include
the approved specification, immutable invariants, allowed files, bounded
families, required tests, and the prohibition on backtests, optimization,
holdout access, and unrelated changes.

The included demonstration uses a validated prebuilt adapter, so no Codex
process is launched by its deterministic integration test.
