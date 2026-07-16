from collections.abc import Mapping
from numbers import Real


def verify_metrics(metrics: Mapping, required: list[str] | None = None) -> tuple[bool, list[str]]:
    errors = []
    for key in required or []:
        if key not in metrics:
            errors.append(f"missing metric: {key}")
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"metric {key} must be numeric")
    return not errors, errors

