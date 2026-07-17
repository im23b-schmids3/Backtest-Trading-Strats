# Real strategy adapter

Implementations expose `AdapterIdentity`, `AdapterCapabilities`, health and
data checks, bounded research methods, normalized backtest outputs, B.5
diagnostics, Phase D exports, and Phase E eligibility. Every boundary uses
Pydantic models. Free-form dictionaries are limited to evidence payloads inside
typed result objects.

An adapter must match the approved strategy ID, version, specification hash,
and schema version. Missing or incompatible adapters fail with
`REAL_ADAPTER_REQUIRED` or `ADAPTER_SCHEMA_INCOMPATIBLE`.
