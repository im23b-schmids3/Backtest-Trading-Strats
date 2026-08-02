"""Durable external natural-language specification generation and repair."""

__all__ = ["ExternalSpecificationExecutor", "SpecificationCompletion", "SpecificationCompletionStatus", "SpecificationJobRequest", "SpecificationJobService", "SpecificationJobType"]


def __getattr__(name: str):
    if name == "ExternalSpecificationExecutor":
        from .executor import ExternalSpecificationExecutor
        return ExternalSpecificationExecutor
    if name == "SpecificationJobService":
        from .jobs import SpecificationJobService
        return SpecificationJobService
    if name in {"SpecificationCompletion", "SpecificationCompletionStatus", "SpecificationJobRequest", "SpecificationJobType"}:
        from .models import SpecificationCompletion, SpecificationCompletionStatus, SpecificationJobRequest, SpecificationJobType
        return locals()[name]
    raise AttributeError(name)
