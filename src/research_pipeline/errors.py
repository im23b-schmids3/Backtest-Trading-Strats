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

