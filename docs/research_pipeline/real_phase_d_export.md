# Real Phase D export

After an accepted Phase C candidate, the adapter writes a candidate-hash-bound
chronological futures export. Phase D then validates repository mappings,
contracts, risk, and prop economics. Unsupported mappings are recorded as
`INSUFFICIENT_FUTURES_DATA` or own-capital-only evidence; they do not silently
become native futures data.
