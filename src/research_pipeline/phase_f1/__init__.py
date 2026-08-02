"""Phase F1 user-facing end-to-end orchestration.

The service export is lazy so lower-level implementation-job contracts can
reuse the F1 models without importing the orchestration service recursively.
"""

from .models import FinalReport, IntakeSpec, MasterRunOutcome, MasterRunStatus

__all__ = ["FinalReport", "IntakeSpec", "MasterPipelineService", "MasterRunOutcome", "MasterRunStatus"]


def __getattr__(name: str):
    if name == "MasterPipelineService":
        from .service import MasterPipelineService

        return MasterPipelineService
    raise AttributeError(name)
