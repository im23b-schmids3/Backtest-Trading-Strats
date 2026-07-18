class ResearchPipelineError(Exception):
    """Base class for expected, user-facing pipeline errors."""


class InvalidTransitionError(ResearchPipelineError):
    pass


class TerminalStateError(InvalidTransitionError):
    pass


class SpecificationValidationError(ResearchPipelineError):
    pass


class ImmutableSpecificationError(ResearchPipelineError):
    pass


class BudgetExceededError(ResearchPipelineError):
    pass


class BudgetConfigurationError(ResearchPipelineError):
    pass


class SplitConflictError(ResearchPipelineError):
    pass


class HoldoutAccessError(ResearchPipelineError):
    pass


class RegistryError(ResearchPipelineError):
    pass


class ExternalSpecificationRequired(ResearchPipelineError):
    """Smithers must pause while the authenticated local Codex runs externally."""

    def __init__(self, message: str, *, classification: str, run_id: str, job_id: str, command: str):
        super().__init__(message)
        self.classification = classification
        self.run_id = run_id
        self.job_id = job_id
        self.command = command
