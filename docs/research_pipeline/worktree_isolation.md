# Worktree isolation

`WorktreeManager` requires the configured repository root to be the Git top-level and, for real implementation, requires a clean primary worktree. It derives a filesystem-safe branch and worktree name from `strategy_id` and `version`, places the worktree under the configured parent, records the base commit, and refuses an unrelated existing path.

The primary worktree is never deleted or changed by cleanup. Creating a worktree is idempotent when the requested path and branch already match. Cleanup is intentionally not automatic; an operator must remove an approved worktree using an explicit Git command after reviewing its commit or diff.

Dry-run planning validates the Git root and computes the path but does not create a worktree. Real implementation is never merged automatically.
