# Local ETH/BTC market-data audit

Audit date: 2026-08-06. Read-only prospective Fib V1 audit: no raw rows printed, no full dataset load, no archive extraction, no market-data change/download.

## Exhaustive supplemental scope

This corrects the stable-root audit. The inventory covers every ETH/BTC artifact under **all** `data/` and `external_data/` subtrees, including normalized, downloaded/archive, cached, and prior-study/Phase-A directories. The authoritative exhaustive list of all 388 matching paths is [local-data-inventory.json](local-data-inventory.json).

Only `.git/`, `.smithers/`, `.tmp/`, `.pytest_cache/`, `research_runs/`, worktrees, dependencies, and generated reports are out of scope. Normalized, archive, cache, and Phase-A paths are included, not excluded.

## Contract provenance and eligibility

Existing manifests are text-JSON metadata; their identity/hash claims are not promoted by inference. Existing normalized/derived prior-study material remains ineligible unless it is an immutable, adequately identified exact ETH 4h or BTC 1d OHLC input contract with a matching manifest. Archives are recorded without extraction. Aggregate trades and other intervals were not resampled.

| Asset | Required contract | Result |
|---|---|---|
| ETH | Immutable, adequately identified exact ETH 4h OHLC input with matching manifest | `DATA_CONTRACT_BLOCKED` |
| BTC | Immutable, adequately identified exact BTC 1d OHLC input with matching manifest | `DATA_CONTRACT_BLOCKED` |

No manifested exact ETH 4h/BTC 1d OHLC source was established. Cached or filename-labelled exact-shape files do not prove the required immutable identified input contract.

## Execution boundary

No backtest, optimization, candidates, Phase A, holdout, Phase B, Alpha, prop, portfolio, or live action was run. Holdout was not accessed.

## Supersession note — 2026-08-06

The prior `DATA_CONTRACT_BLOCKED` conclusion is superseded only for `FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION` by the hash-bound manifests at `data-contracts/ETH_4H/manifest.json` and `data-contracts/BTC_1D/manifest.json`. The owner attests the intended ETH 4-hour and BTC 1-day purpose solely for this prospective study; hash-bound file identity, independently verified cadence, UTC chronology, and OHLCV integrity support that limited use. Source exchange and spot/perpetual classification are **UNKNOWN**. Results apply only to these exact hash-bound files, with no transferability to another venue or instrument type and no historical V6/V6.5 reproduction claim. Unknown provenance does not block this prospective study for the stated limited purpose. No market data was modified or downloaded, and no strategy, backtest, optimization, candidate, Phase A, holdout, Phase B, Alpha, prop, portfolio, or live action was run.
