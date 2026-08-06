# FibRetracementContinuation.ETH_BTC_V1_REPRODUCTION_SPECIFICATION forensic reconstruction

Status: **MATRIX_EVIDENCE_ACCEPTED_BUT_EXECUTION_PROVENANCE_BLOCKED**. This documentation-only audit read the supplied CSV matrices completely, committed text/CSV evidence, source/configuration/tests, and Git metadata. It did not read, download, generate, cache, normalize, or modify market data; it did not run any backtest, optimization, holdout, Phase A/B, Alpha, prop, portfolio, live action, or candidate.

## Matrix evidence

The repository owner supplied `evidence/v6_ranked_matrix.csv` and `evidence/v6_5_ranked_matrix.csv`. They are currently Git-tracked in commit `0d87708f3a8d0e7794a925ef7fcba0b7e21b5f3c` (`2026-08-06T16:07:29+02:00`, `Add Fib 0.9 reconstruction evidence`), with blobs `28ec027b9aa3a3183416492c31dffd182388bc5a` and `c94e7e8d83f94254a470964cfcd877ca02fb15c9`. Current SHA-256/byte size are respectively `bede956d6e972606f5c30e3e758902db65a83069b701e3226e76147740866661`/1,245,845 and `c45c360c866b237df74028d5864911776947af97567bfb8de679699fc2c509dd`/71,047.

Evidence status is **USER_SUPPLIED_EVIDENCE**: authenticated as supplied by the repository owner and pinned to current bytes, but no historic hash/reference proves byte identity to historical generated files. Therefore each row proves only its current reported parameter/result, never the historical execution mechanics. Full coverage, columns/meanings, missing values, duplicate check, rank checks, and matrix differences are in `matrix-evidence-manifest.json`.

The fixed candidates are exactly: `FIB09-ETH-4H-POST0830` (V6.5 source line 2), `FIB09-ETH-4H-POST0786-REFERENCE` (line 8), and `FIB09-BTC-1D-POST0786` (line 27). Every source-column raw value is preserved in `historical-row-registry.json`; every supplied anchor comparison is preserved in `matrix-anchor-comparison.json`. All supplied metrics match the rows after numeric/display-precision comparison; the supplied six-decimal values are not raw-string-identical to the longer CSV decimals. `0.830` is numerically equal to the CSV's `0.83` but textually different.

## Source and Git search findings

Search of committed text/CSV/source/configuration/tests and Git metadata for every matrix filename stem, column-name families, distinctive values (`0.830`, `0.786`, `0.9`, both exact final-equity values), and all three candidate identifiers found: the two matrix generators/rankers (`src/fib_backtester/research/v6_post_tp1_stop.py`, `v6_5_post_tp1_stop_placement.py`); stop engines/strategies; V2 lifecycle/execution; metrics; default config; and relevant tests. Candidate identifiers occur only in the prior documentation/index, not code or historical run artifacts. No trade output, run manifest, data hash/identity, historical date range, generator invocation, or historical-generated-matrix hash was found. Detailed classifications are in the updated source index.

## Execution-critical sealing checklist

| Item | Current classification | Seal result |
| --- | --- | --- |
| Chronology / historical run provenance | UNRESOLVED | blocks |
| Data source, schema, identity/hash, date range | UNRESOLVED | blocks |
| Swing/Fibonacci anchors | PROVEN in current code; historical use UNRESOLVED | blocks |
| Entry activation/fill | PROVEN in current code; historical use UNRESOLVED | blocks |
| Initial stop | PROVEN in current code; historical use UNRESOLVED | blocks |
| TP fractions/levels | PROVEN in current code; historical use UNRESOLVED | blocks |
| Post-TP1 formula | PROVEN in current code; historical use UNRESOLVED | blocks |
| Same-bar ordering | PROVEN in current code; historical use UNRESOLVED | blocks |
| Costs/slippage | PROVEN in current code/default config; historical use UNRESOLVED | blocks |
| Sizing/compounding | PROVEN in current code; historical use UNRESOLVED | blocks |
| Concurrency | PROVEN in current code/default config; historical use UNRESOLVED | blocks |
| End-of-data handling | PROVEN in current code; historical use UNRESOLVED | blocks |

Current code establishes a possible contract: fixed entry 0.900; Fib initial stop 1.02; Profile-B fractions 30/25/20/15/10 at ratios 0.786/0.618/0.500/0.236/0.050; next-bar activation; conservative stop-before-target precedence; post-TP1 stop effective next candle; compounded 2% risk sizing; max_positions=1; force exit at final close; default Binance BTC/USDT and ETH/USDT, fee .001 and slippage .0002. This is **PROVEN current-code evidence**, not authenticated historical execution provenance.

No sealed specification was created. The shortest remaining materials to reach `RECONSTRUCTION_READY_TO_SEAL` are: (1) an authenticated historical run manifest binding both matrices to generator commit/config, data source/schema/hash and UTC start/end range; and (2) authenticated historical trade/order output (or equivalent deterministic execution log) binding the three rows to chronology, fills, costs, ordering, sizing/concurrency, and end-of-data handling.
