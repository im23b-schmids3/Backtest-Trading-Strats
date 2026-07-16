# Provider compliance

The rule model records account size, target, MLL, DLG, contract limits,
consistency, winning-day requirements, payout split, fees, cancellation and
billing behavior, session assumptions, prohibited practices, automation
restrictions, source URLs, verification date, and ambiguities.

The current official fixtures cover Alpha Futures Zero 25K and Zero 50K. There
is intentionally no default 100K product. The compliance reviewer consumes the
persisted rule hash and produces a typed result. Missing facts are warnings or
blocking errors; they are never filled by agent prose.
