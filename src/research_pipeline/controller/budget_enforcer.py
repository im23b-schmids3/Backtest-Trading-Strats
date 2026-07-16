from __future__ import annotations

from dataclasses import dataclass

from ..errors import BudgetExceededError, BudgetConfigurationError
from ..schemas.budgets import BudgetUsage, ResearchBudget


@dataclass(frozen=True)
class BudgetRequest:
    backtests: int = 0
    family: str | None = None
    rounds: int = 0
    values: int = 0
    codex_repairs: int = 0
    runtime_minutes: int = 0
    report_size_mb: float = 0.0
    round_id: int | None = None


class BudgetEnforcer:
    """Pure deterministic budget checks; persistence is supplied by the registry."""

    @staticmethod
    def check(limits: ResearchBudget, usage: BudgetUsage, request: BudgetRequest) -> BudgetUsage:
        if min(request.backtests, request.rounds, request.values, request.codex_repairs, request.runtime_minutes) < 0 or request.report_size_mb < 0:
            raise BudgetConfigurationError("budget consumption cannot be negative")
        next_usage = usage.model_copy(deep=True)
        if request.family is None and (request.rounds or request.values):
            raise BudgetConfigurationError("family is required for rounds or values")
        if request.family:
            if request.family not in next_usage.parameter_families:
                if len(next_usage.parameter_families) >= limits.max_parameter_families:
                    raise BudgetExceededError("maximum parameter families exceeded")
                next_usage.parameter_families.append(request.family)
                next_usage.parameter_families.sort()
            if request.round_id is not None and request.rounds != 1:
                raise BudgetConfigurationError("round_id can only be used with exactly one round")
            first_round = request.round_id if request.round_id is not None else next_usage.rounds_per_family.get(request.family, 0) + 1
            for round_number in range(first_round, first_round + request.rounds):
                key = str(round_number)
                assigned = next_usage.family_by_round.get(key)
                if assigned is not None and assigned != request.family:
                    raise BudgetExceededError(f"research round {round_number} already belongs to parameter family {assigned}")
                if next_usage.values_by_round.get(key, 0) + request.values > limits.max_values_per_round:
                    raise BudgetExceededError(f"maximum values exceeded in research round {round_number}")
                next_usage.family_by_round[key] = request.family
                next_usage.values_by_round[key] = next_usage.values_by_round.get(key, 0) + request.values
            rounds = next_usage.rounds_per_family.get(request.family, 0) + request.rounds
            if rounds > limits.max_rounds_per_family:
                raise BudgetExceededError(f"maximum rounds exceeded for parameter family {request.family}")
            if request.values > limits.max_values_per_round:
                raise BudgetExceededError("maximum values per round exceeded")
            next_usage.rounds_per_family[request.family] = rounds
            next_usage.values_per_round[request.family] = next_usage.values_per_round.get(request.family, 0) + request.values * request.rounds
        if next_usage.total_backtests + request.backtests > limits.max_total_backtests:
            raise BudgetExceededError("maximum total backtests exceeded")
        if next_usage.codex_repair_attempts + request.codex_repairs > limits.max_codex_repair_attempts:
            raise BudgetExceededError("maximum Codex repair attempts exceeded")
        current_runtime = next_usage.runtime_minutes_by_phase.get("current", 0) + request.runtime_minutes
        if current_runtime > limits.max_runtime_minutes_per_phase:
            raise BudgetExceededError("maximum runtime per phase exceeded")
        if next_usage.report_size_mb + request.report_size_mb > limits.max_report_size_mb:
            raise BudgetExceededError("maximum report size exceeded")
        next_usage.total_backtests += request.backtests
        next_usage.codex_repair_attempts += request.codex_repairs
        next_usage.runtime_minutes_by_phase["current"] = current_runtime
        next_usage.report_size_mb += request.report_size_mb
        return next_usage
