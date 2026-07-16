# Infrastructure_V2_AlphaDataResearch

## Decision

This is a data-infrastructure plan only. No strategy, signal, execution, portfolio, or backtesting code was changed.

The recommended architecture is deliberately two-tier:

1. Databento is the operational historical API and normalized contract-data source for MET, the equity-index futures, and the FX futures.
2. CME DataMine is the first-party audit source for metals, energy, and interest-rate futures, and the independent verification source for the Databento downloads.
3. CME Reference Data is the authority for contract specifications, tick sizes, multipliers, trading schedules, expiration, first notice, last trade, and lifecycle dates.

Databento is preferred where a self-service Python/API workflow is most valuable. Its CME Globex dataset exposes hourly data, instrument definitions, continuous symbology, and unadjusted contract prices. Its continuous symbols select tradable contracts over time but do not hide roll jumps through back-adjustment; that is appropriate for an auditable research pipeline. Sources: [Databento quickstart](https://databento.com/docs/quickstart), [futures symbology](https://databento.com/docs/examples/futures/futures-introduction/special-conventions-for-futures-on-databento), and [continuous-contract conventions](https://databento.com/docs/standards-and-conventions/symbology).

CME DataMine is preferred as the authority where contract history, settlements, and lifecycle provenance matter more than implementation convenience. CME describes DataMine as the historical delivery point for futures data and CME Reference Data as the authoritative source for specifications and lifecycle dates. Sources: [CME DataMine](https://www.cmegroup.com/datamine.html), [CME futures data catalog](https://www.cmegroup.com/market-data/browse-data/catalog/futures-and-options-data.html), and [CME Reference Data](https://www.cmegroup.com/market-data/browse-data/catalog/reference-data.html).

## Provider selection by market family

| Family | Primary | Secondary/audit | Reason |
|---|---|---|---|
| MET | Databento | CME DataMine | CME Globex contract data with API-friendly symbology; no ETH spot proxy |
| MNQ, NQ, MES, ES, M2K, RTY, MYM, YM | Databento | CME DataMine | Deep liquid CME products and straightforward contract/roll mapping |
| MGC, GC, SIL, SI, MHG, HG | CME DataMine | Databento | First-party COMEX provenance and official contract lifecycle context |
| MCL, CL, MNG, NG, HO, RB | CME DataMine | Databento | First-party NYMEX data, especially important for delivery and roll behavior |
| M6E, 6E, M6A, 6A, M6B, 6B | Databento | CME DataMine | Consistent API and symbology for the FX family; validate sessions against CME |
| ZN, ZB, ZF, ZT | CME DataMine | Databento | First-party CBOT contract and delivery conventions |

The recommendations are not proxies. They refer to the official listed Alpha/CME contracts themselves.

## Implementation sequence

### 1. Entitlement and legal setup

- Create a Databento account and store the API key outside the repository.
- Request the CME Globex historical dataset and verify that MET, all requested CME/CBOT/NYMEX/COMEX roots, and the required historical dates are licensed.
- Open CME DataMine access and purchase a small representative sample for MET, MNQ, MGC, MCL, M6E, and ZN before buying the full archive.
- Obtain the required Information License Agreement and determine whether the repository is personal, internal, or commercial research.
- Never commit API keys, raw credentials, or provider tokens.

### 2. Contract catalog

Create a provider-neutral contract catalog with:

- Alpha symbol and exchange;
- provider symbol and instrument ID;
- contract month and year;
- tick size, tick value, multiplier, currency;
- first notice, last trade, expiration, and delivery dates;
- trading schedule and holiday calendar;
- source provider, dataset, retrieval timestamp, and checksum.

CME Reference Data should be the authority for specifications and lifecycle dates. Databento instrument definitions should be retained as a second copy and compared field-by-field.

### 3. Raw acquisition

Prefer native contract-level trades or 1-minute OHLCV. Store the raw response before any transformation. Each download should have a manifest containing:

- request parameters;
- provider and dataset;
- source symbol and resolved instrument IDs;
- UTC retrieval time;
- earliest/latest timestamp;
- row count and byte count;
- checksum;
- provider status and error responses.

Do not use Yahoo continuous data, ETF prices, spot prices, or unlabelled vendor continuous series as Alpha futures data.

### 4. Roll methodology

Maintain two separate representations:

- `contract_series`: unmodified individual contract bars;
- `continuous_series`: a documented lead-month or volume/open-interest roll series used only where a continuous research series is appropriate.

The default continuous series should use volume/open-interest selection and preserve unadjusted prices. A separate back-adjusted research view may be generated later, but it must never replace the raw contract series. Roll dates, old/new contracts, price gaps, and adjustment factors must be stored explicitly.

### 5. 1H and 4H construction

- Build 1H bars from provider 1H OHLCV where the bar semantics are documented, otherwise aggregate from 1-minute trades.
- Build 4H bars from the canonical 1H or 1-minute source inside the repository, never by requesting an opaque provider-specific 4H bar unless it has been reconciled.
- Use UTC as the storage timezone.
- Store exchange-local and Europe/Berlin timestamps only as derived fields.
- Preserve empty session gaps; do not forward-fill OHLC.
- Define bar boundaries, inclusion rules, and daylight-saving behavior in the manifest.

### 6. Data-quality gates

Every market must pass before becoming available to research:

- OHLC consistency: `high >= max(open, close, low)` and `low <= min(open, close, high)`;
- monotonic, timezone-aware timestamps;
- duplicate and out-of-order detection;
- tick-size alignment;
- non-negative volume and trade-count checks;
- exchange-session and holiday validation;
- expected-gap classification versus missing-data classification;
- contract rollover detection;
- comparison of Databento and CME DataMine samples;
- exact first/last timestamp and row-count report for both 1H and 4H.

The gate should fail closed. A market with missing contract metadata or unexplained gaps must not enter a backtest.

### 7. Repository layout

The future implementation should add provider adapters without changing the strategy/backtester:

```text
data/
  raw_contracts/{provider}/{exchange}/{root}/{contract}.parquet
  canonical/{market}/1h.parquet
  canonical/{market}/4h.parquet
  manifests/{market}.json
  quality/{market}.csv
src/fib_backtester/data/
  providers/databento.py
  providers/cme_datamine.py
  catalog.py
  rolls.py
  bars.py
  quality.py
```

The existing research engine should consume canonical OHLCV only after the data-quality gate. This plan intentionally does not implement those adapters yet.

## API-key and access checklist

| Provider | Key/credential | Free tier | Optional? | Anonymous download? |
|---|---|---|---|---|
| Databento | API key from Databento portal API Keys page | $125 new-account data credit | Required for use | No |
| CME DataMine | Account, dataset license, ILA, delivery credentials | None assumed | Required for official historical data | No |
| Rithmic | FCM/broker credentials and API/dev-kit approval | Simulator trial only | Required for production/history | No |
| Tradovate | OAuth/access token from Tradovate account | None assumed | Required | No |
| IBKR | IBKR account, TWS/Client Portal credentials, market-data subscriptions | API software is not a free futures entitlement | Required | No |
| dxFeed | Customer credentials after sales onboarding | None public | Required | No |
| Polygon | API key from Polygon dashboard | Futures Basic shown at $0 | Required | No |
| Alpaca | API key and secret | Free supported-stock/crypto plan | Required | No |
| Twelve Data | API key from account dashboard | Limited trial/free access varies | Required | No |
| Alpha Vantage | API key from support/API-key page | 25 requests/day for majority of free endpoints | Required | No |
| Financial Modeling Prep | API key from dashboard | Free plan with limited bandwidth | Required | No |
| Stooq | No key for public daily CSV | Free daily downloads | Not applicable | Daily CSV only |

## Five-year stack recommendation

If building this repository for the next five years, use Databento as the daily operational ingestion source, CME DataMine as the first-party audit and recovery source, and CME Reference Data as the contract-catalog authority. Retain raw contract data permanently, derive canonical 1H/4H bars locally, and use Rithmic or IBKR only for occasional live/parity checks.

This stack balances reproducibility, contract fidelity, API automation, roll transparency, and long-term maintainability. It also avoids repeating the central V11 problem: treating a convenient proxy or opaque continuous series as if it were native Alpha Futures history.

## Research limitations

- Public provider pages do not expose a universal price for every CME DataMine dataset; exact cost requires dataset selection and licensing review.
- Rithmic, Tradovate, IBKR, and dxFeed access depends on account, broker, feed, or commercial entitlements.
- Polygon’s futures product is promising but should be treated as provisional until actual requested contracts, history, and roll semantics are verified.
- 4H is recommended as a repository-derived bar for consistency, even when a provider exposes a 4H endpoint.
