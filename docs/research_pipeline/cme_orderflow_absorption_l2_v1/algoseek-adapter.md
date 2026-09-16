# Algoseek CME L2 adapter

`algoseek_adapter.py` is a local-file ingestion boundary for the early-2023
research period. Databento remains the normal provider for later periods.
Neither the adapter nor its audit command downloads data, selects parameters,
or runs Stage 1, Stage 2, or Stage 3.

## Semantics and ordering

Artifacts emitted from this source must carry:

- `provider: ALGOSEEK`
- `provider_semantics: ALGOSEEK_CAUSAL_VARIANT`
- `ordering_policy_id: algoseek-causal-order-v2`
- `depth_assembly_policy_id: algoseek-depth-assembly-v2-final-state-with-raw-provenance`

The deterministic research convention for equal normalized timestamps is:

1. MES BBO quote
2. ES Multiple Depth side update
3. ES aggressive trade
4. ES Trade & Quote BBO update
5. source file path and original row number

This convention is not an assertion of undocumented exchange ordering.
`EventDateTime` is converted from `America/Chicago` to UTC using timezone-aware
DST conversion. Ambiguous and nonexistent wall times fail closed.

Multiple Depth bid and ask rows are independently maintained internally. Rows
sharing an exact timestamp are then exposed as one final-state depth event,
after all rows in that timestamp group have been applied. This prevents a
synthetic half-updated, locked, or crossed book from becoming a strategy input.
The event retains every raw row, including repeated rows, in
`raw_provenance`; no provider observations are silently deduplicated. The
policy is deterministic and provider-specific, not a claim that the rows were
atomic on the exchange. The adapter also rejects a session with more than one
ES or MES `(Ticker, SecurityID)` identity, so rollover ownership must be
explicit in the input manifest.

Algoseek does not supply the MBO action provenance used for unexecuted-add and
rapid-cancel false-refill components. The adapter exposes no inferred
`MBP10Update`; those subcomponents are provider-limited and must be retained as
such in canonical-artifact metadata.

## Input audit

Use explicitly named, already-local files:

```sh
cme-l2-research algoseek-input-audit \
  --es-depth /path/ES_multiple_depth_2023-03-10.csv \
  --es-taq /path/ES_trade_quote_2023-03-10.csv \
  --mes-taq /path/MES_trade_quote_2023-03-10.csv
```

The command reports source hashes, row and trade counts, BBO/depth disagreement
counts, initialization/crossed-book states, deterministic same-timestamp tie
metrics, and causal-variant provenance. It is an input audit, not a replay.

An optional input manifest has `schema_version: 1` and explicit session rows.
Every row names `date` and `provider`; Algoseek rows also name `es_depth`,
`es_taq`, `mes_taq`, `es_ticker`, `es_security_id`, `mes_ticker`, and
`mes_security_id`. Exactly one provider owns each session.
