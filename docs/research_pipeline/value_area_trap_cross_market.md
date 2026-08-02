# Frozen ValueAreaTrap cross-market robustness evaluation

This command evaluates only `XAUUSDT`, `QQQUSDT`, and `SPYUSDT` for the three
complete common months **May through July 2026**. It is a descriptive robustness
check of the already frozen UTC strategy—not a parameter search, symbol ranking,
approval, or promotion mechanism.

All output is labelled: **Binance synthetic/TradFi perpetual proxy evidence; not
ownership of QQQ or SPY ETF shares; not native COMEX gold; not CME MBT; not Alpha
Futures performance.**

The cross-market path reuses the packaged ValueAreaTrap adapter and has no Codex
executor, broker client, or live-order path. Strategy settings are fixed across
symbols. The only symbol-specific values are Binance's pinned tick-size,
quantity-step, and minimum-quantity filters; the evaluation uses the minimum
valid fixed quantity rather than assuming BTC's `0.001` quantity is tradable.

Eligibility is intentionally narrow. Binance's live USD-M metadata identifies
all three supported instruments as `contractType: TRADIFI_PERPETUAL` (not
`PERPETUAL`), with `status: TRADING`, `marginAsset: USDT`, `quoteAsset: USDT`,
and `underlyingSubType: [TradFi]`. The ingestion command pins each complete raw
symbol object and its hash in the immutable metadata artifact. Any different
symbol, fixed-expiry/delivery contract, spot classification, unavailable status,
or incompatible settlement is rejected with that symbol's diagnostic fields.

## 1. Ingest and pin metadata

This command may download only the three requested monthly Binance archives and
the public USD-M `exchangeInfo` metadata when no pinned metadata artifact has
been supplied. Completed hash-verified monthly partitions are skipped.

```powershell
python -m research_pipeline value-area-trap ingest-cross-market `
  --symbols XAUUSDT QQQUSDT SPYUSDT `
  --start-month 2026-05 `
  --end-month 2026-07 `
  --cache-root .\data\value_area_trap `
  --allow-network
```

The output supplies three manifest paths and the generated immutable metadata
artifact. A first or last timestamp outside the one-hour calendar-edge tolerance
fails as `INCOMPLETE_CALENDAR_MONTH`; April cannot be requested.

## 2. Validate the manifests

```powershell
python -m research_pipeline value-area-trap validate-cross-market `
  --manifest XAUUSDT=./data/value_area_trap/normalized/XAUUSDT/<hash>/manifest.json `
  --manifest QQQUSDT=./data/value_area_trap/normalized/QQQUSDT/<hash>/manifest.json `
  --manifest SPYUSDT=./data/value_area_trap/normalized/SPYUSDT/<hash>/manifest.json
```

## 3. Run the descriptive frozen evaluation

```powershell
python -m research_pipeline value-area-trap run-frozen-cross-market `
  --manifest XAUUSDT=./data/value_area_trap/normalized/XAUUSDT/<hash>/manifest.json `
  --manifest QQQUSDT=./data/value_area_trap/normalized/QQQUSDT/<hash>/manifest.json `
  --manifest SPYUSDT=./data/value_area_trap/normalized/SPYUSDT/<hash>/manifest.json `
  --artifact-root .\research_runs `
  --repository-root .
```

The comparison has per-symbol May/June/July metrics, PnL-sign agreement,
profit-factor checks, exceptional-month flags, and pooled descriptive totals.
It deliberately writes `best_symbol: null`, `ranking: null`, `promotion: null`,
and `selection_prohibited: true`.
