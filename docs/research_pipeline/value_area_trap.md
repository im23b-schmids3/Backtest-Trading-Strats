# ValueAreaTrap research baseline

ValueAreaTrap is a research-only `value_area_trap_reference` strategy family.
It consumes public Binance USD-M Futures `BTCUSDT` perpetual aggregate trades;
it never substitutes OHLCV data and has no authenticated exchange or broker path.

The session is labelled `US_CASH_WINDOW_PROXY` (09:30-16:00
`America/New_York`), not CME RTH.  Previous-session VAH, VAL and POC are fixed
for the whole current session.  A 70% base-volume profile uses deterministic
POC ties (nearest volume-weighted mean, then lower bucket) and expands one
larger adjacent bucket at a time, choosing the lower side on a volume tie.

Aggregate data is stored at a content-addressed path under
`data/value_area_trap/normalized/BTCUSDT/<dataset_hash>/`. Each parquet file
requires its adjacent `manifest.json`. Network fetching is disabled unless the
importer is explicitly called with `allow_network=True`.

Example intake and pipeline commands:

```powershell
python -m research_pipeline value-area-trap download 2026-04 --cache-root .\data\value_area_trap --allow-network
python -m research_pipeline value-area-trap import-archive .\data\value_area_trap\downloads\BTCUSDT\BTCUSDT-aggTrades-2026-04.zip --cache-root .\data\value_area_trap
python -m research_pipeline value-area-trap validate-data PATH_TO\aggregate_trades.parquet
python -m research_pipeline value-area-trap import-calendar .\calendar.csv .\research_registry\value_area_trap_calendar.json
python -m research_pipeline run .\examples\research_pipeline\value_area_trap_binance_intake.json --mode dry_run --dry-run
python -m research_pipeline status RUN_ID
python -m research_pipeline approve RUN_ID --decision APPROVE --note "reviewed fixed baseline"
python -m research_pipeline resume RUN_ID
python -m research_pipeline report RUN_ID
python -m research_pipeline artifacts RUN_ID
```

The Alpha Zero 25K component is a separate scenario simulator. It models
versioned assumptions, risk-normalized BTC quantities, EOD MLL, daily locks and
Qualified news/consistency checks. It is not Alpha Futures integration or a
claim of compliance. BTC quantities are `NOT_COMPARABLE_TO_BINANCE` contract
references, not CME mini/micro contracts.
