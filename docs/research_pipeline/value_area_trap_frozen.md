# Frozen ValueAreaTrap UTC workflow

`ValueAreaTrap.UTC_24H_SESSION` is packaged as a verified repository adapter.
The frozen command never creates a Codex implementation job, never asks for a
manual approval or resume, and never runs parameter research. It accepts only
the fixed UTC session specification and a hash-verified combined monthly
aggregate-trade manifest.

Every report is labelled: **Binance BTCUSDT perpetual proxy evidence; not CME
MBT or Alpha Futures performance.** April 2026 is reported separately as a
previously observed month and is never used for parameter selection.

## A. Download and normalize the 16 monthly archives

This is the only command that may use the network. It reuses completed archive
downloads and hash-verified monthly partitions. Omit `--allow-network` to
validate/reassemble already-downloaded archives without any network request.

```powershell
python -m research_pipeline value-area-trap ingest-range `
  --symbol BTCUSDT `
  --start-month 2025-01 `
  --end-month 2026-04 `
  --cache-root .\data\value_area_trap `
  --allow-network
```

If, and only if, the command reports a proven missing aggregate-trade ID range,
rerun with `--allow-gap-repair` as well. That opt-in requests only the missing
IDs; a continuous-ID timestamp interval such as August 2025 is recorded as a
no-trade diagnostic and does not use the API.

The resulting immutable layout is:

```text
data/value_area_trap/normalized/BTCUSDT/<combined_dataset_hash>/
  2025-01.parquet
  ...
  2026-04.parquet
  manifest.json
```

The importer streams one archive/Parquet partition at a time. It checks source
archive hashes, partition hashes and row counts, month continuity, timestamp
ordering, aggregate-trade ID ordering/overlap, and out-of-month records. A
timestamp interval longer than five minutes is retained as a diagnostic; it is
accepted only when its aggregate-trade IDs are continuous, which documents a
real no-trade interval. A proven missing-ID range fails closed unless the
operator additionally supplies `--allow-gap-repair`, which fetches only those
IDs from the Binance USD-M API and writes an immutable repair audit. It never
substitutes OHLCV.

## B. Verify the combined manifest

```powershell
python -m research_pipeline value-area-trap validate-manifest `
  .\data\value_area_trap\normalized\BTCUSDT\<combined_dataset_hash>\manifest.json
```

## C. Run the frozen strategy

```powershell
python -m research_pipeline --registry .\research_registry\research_pipeline.sqlite3 `
  value-area-trap run-frozen `
  --variant UTC_24H_SESSION `
  --data-manifest .\data\value_area_trap\normalized\BTCUSDT\<combined_dataset_hash>\manifest.json `
  --artifact-root .\research_runs `
  --repository-root . `
  --auto-approve `
  --reuse-verified-implementation
```

The command fails closed unless the manifest covers `2025-01-01` through
`2026-04-30`, the frozen parameters exactly match the packaged contract, and
the adapter identity is `value-area-trap-3`. It gives the run a deterministic
identity derived from the specification and combined dataset hashes.

Its comparison report has separate primary-holdout (`2025-01-01` through
`2026-03-31`), April-2026, full-period, and monthly sections. These include
the event funnel, trades, PnL/costs, profit factor, drawdown, losing streak,
exposure/frequency, invalid setups, compliance blocks, and executed trades.
