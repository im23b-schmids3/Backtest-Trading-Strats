# Worktree review

Successful implementations remain isolated and are never merged automatically.
Review with:

```powershell
git -C <worktree> status
git -C <worktree> diff <base-commit>..<implementation-commit>
git -C <worktree> log --oneline --decorate -n 10
```

Diffs touching historical strategies, data, reports, holdout artifacts,
providers, credentials, broker code, or live-trading code are blocked as
`IMPLEMENTATION_SCOPE_VIOLATION`.
