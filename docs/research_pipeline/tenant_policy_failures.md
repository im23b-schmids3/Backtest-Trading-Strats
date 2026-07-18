# Tenant and sandbox policy failures

Authenticated Codex access is an external capability. A restricted Smithers
environment may be unable to use it even when the local CLI is installed.
Recognized policy-denial evidence includes tenant-policy/network denial and
Windows socket error `10013`.

Such a failure is classified as `CODEX_EXECUTION_REQUIRES_EXTERNAL_EXECUTOR`.
It is not treated as a malformed specification, and no repair budget is
consumed. The recorded invocation contains redacted output only. Operators
must run the documented external executor in an environment where the local
Codex authentication is available; the repository does not bypass tenant
policy.
