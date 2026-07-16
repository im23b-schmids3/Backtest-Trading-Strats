from __future__ import annotations

from .models import PropBudget, PropBudgetUsage


class PropBudgetExceeded(ValueError):
    """Raised before a Phase D operation would exceed a hard budget."""


class PropBudgetEnforcer:
    @staticmethod
    def consume(limits: PropBudget, usage: PropBudgetUsage, *, scenarios: int = 0, accounts: int = 0, replay_days: int = 0, artifact_size_mb: float = 0, policy_variants: int = 0, concurrent_evaluations: int = 0) -> PropBudgetUsage:
        values = [scenarios, accounts, replay_days, policy_variants, concurrent_evaluations]
        if min(values) < 0 or artifact_size_mb < 0:
            raise PropBudgetExceeded("Phase D budget consumption cannot be negative")
        next_usage = usage.model_copy(deep=True)
        checks = (("scenarios", scenarios, limits.max_scenarios), ("accounts", accounts, limits.max_accounts_per_scenario), ("replay_days", replay_days, limits.max_replay_duration_days), ("artifact_size_mb", artifact_size_mb, limits.max_artifact_size_mb), ("policy_variants", policy_variants, limits.max_policy_variants), ("concurrent_evaluations", concurrent_evaluations, limits.max_concurrent_evaluations))
        for name, amount, limit in checks:
            current = getattr(next_usage, name)
            if current + amount > limit:
                raise PropBudgetExceeded(f"maximum Phase D {name} exceeded")
            setattr(next_usage, name, current + amount)
        return next_usage
