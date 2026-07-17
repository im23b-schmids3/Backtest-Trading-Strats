# Specification repair loop

Repairs are deterministic retries, not an open-ended agent loop. The first invalid output is retained unchanged. Subsequent prompts contain the original intake, the invalid draft, and only the structured validation failures. The prompt explicitly prohibits invented defaults, prose, multiple candidates, implementation, backtesting, optimization, and silent ambiguity resolution.

The Python service enforces `max_generation_attempts <= 3` and `max_repair_attempts <= 2`. Every attempt is upserted into SQLite and has file-backed draft, invocation, validation, and optional repair-prompt artifacts. A process interruption can resume from the next missing attempt. A completed canonical attempt is idempotently reused.

Exhaustion produces `SPECIFICATION_GENERATION_FAILURE`, a final failure JSON artifact, a registry failure row, and a nonzero CLI result. No approval node is created for that outcome. A later explicit retry may continue an interrupted run; it cannot bypass the same structural and semantic gates.
