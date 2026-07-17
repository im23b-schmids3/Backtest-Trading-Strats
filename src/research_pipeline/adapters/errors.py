class AdapterError(RuntimeError):
    pass


class RealAdapterRequired(AdapterError):
    code = "REAL_ADAPTER_REQUIRED"


class AdapterCompatibilityError(AdapterError):
    code = "ADAPTER_SCHEMA_INCOMPATIBLE"


class DataAvailabilityError(AdapterError):
    code = "INSUFFICIENT_MARKET_DATA"


class ImplementationScopeViolation(AdapterError):
    code = "IMPLEMENTATION_SCOPE_VIOLATION"


class ArtifactIntegrityError(AdapterError):
    code = "ARTIFACT_INTEGRITY_FAILURE"
