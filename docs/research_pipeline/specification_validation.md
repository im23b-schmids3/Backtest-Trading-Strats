# Specification validation

Pydantic remains the canonical structural contract. Before it is applied, the intake bridge performs representation normalization and semantic validation. Normalization is limited to stable formatting such as market/timeframe casing, whitespace, date representation, and percentage/capital representations; it does not add rules.

Semantic validation blocks approval for material drift, including:

- reversed or empty date intervals;
- invalid IANA session zones or fixed-UTC session claims;
- an exact 10-minute claim on hourly data;
- unseeded randomness;
- daily trade-frequency ambiguity;
- conversion of current-equity allocation into risk-per-trade sizing;
- invented stops, targets, or mutable optimization families;
- unresolved ambiguities or missing information;
- SPY proxy or unsupported Phase D mapping disclosure that is absent.

Failures are persisted as structured issues with `error_code`, `field_path`, `received_value`, `expected_constraint`, `explanation`, and `repair_hint`. Approval is available only when structural, semantic, hash, and provenance checks all pass. A blocked ambiguity can be stored as a draft, but cannot be registered or approved.
