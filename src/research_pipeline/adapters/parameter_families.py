from __future__ import annotations

from ..schemas.strategy_spec import StrategySpec
from .errors import AdapterCompatibilityError
from .models import ResearchParameterFamily


def declared_parameter_families(specification: StrategySpec) -> list[ResearchParameterFamily]:
    result: list[ResearchParameterFamily] = []
    for family in specification.parameter_families:
        if not family.mutable:
            continue
        values = family.allowed_values or [family.baseline_value]
        if not values:
            raise AdapterCompatibilityError(f"parameter family {family.name} has no bounded values")
        result.append(ResearchParameterFamily(family_name=family.name, parameters=[family.name], value_type=family.value_type,
                                              legal_range={"min": family.allowed_min, "max": family.allowed_max} if family.allowed_min is not None and family.allowed_max is not None else None,
                                              bounded_candidate_values=values, baseline_value=family.baseline_value, optimization_direction="maximize_expectancy",
                                              maximum_evaluations=len(values) * max(1, family.maximum_rounds), enabled=True))
    return result
