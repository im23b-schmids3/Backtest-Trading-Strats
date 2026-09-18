# Algoseek REST downloader

`algoseek_api.py` is an acquisition boundary for the existing local Algoseek
adapter. It never runs Stage 1, Stage 2, Stage 3, or a strategy. It uses only
`ALGOSEEK_API_KEY`, sent in the `X-API-KEY` request header; credentials are not
printed or written to manifests.

## Safe first use

Run account access checks, then a 5-row compressed response probe. Neither
command downloads a meaningful historical slice.

```sh
cme-l2-research algoseek-api-preflight
cme-l2-research algoseek-api-probe --logical-reference-date 2023-01-03 --limit 5
```

Preflight validates the key/IP through the identity endpoint but emits only the
two required entitlement decisions, their provider date ranges and universe
status, current quota, Q1 status, and `ready`. Stable entitlement IDs are
primary: `US6002` (Multiple Depth) and `US6011` (Futures Trade and Quote).
The provider display name is a fallback only; `and` and `&` are equivalent.

Only after both pass, download the known reference session to a separate root
and compare its semantic metrics with the manually validated source:

```sh
cme-l2-research algoseek-download-session \
  --logical-session-date 2023-01-03 \
  --output-root data/algoseek/api-validation
cme-l2-research algoseek-compare-session \
  --logical-session-date 2023-01-03 \
  --api-root data/algoseek/api-validation \
  --manual-root data/algoseek/raw
```

The downloader refuses to write to an existing manual session directory that
has no API manifest. This protects the verified `data/algoseek/raw/2023-01-03`
and `2023-01-04` exports.

## Data protocol and output

The production endpoint base is `https://api.algoseek.com/v1`. Set
`ALGOSEEK_API_BASE_URL` only to use an explicitly supplied alternate platform;
the `--base-url` CLI option takes precedence over that environment setting.
The downloader calls the
account identity, quota, entitlement and catalog endpoints during preflight;
data calls use:

- `/data/us-futures/multiple-depth/{trade_date}/{ticker}` for ES depth;
- `/data/us-futures/taq/{trade_date}/{ticker}` for ES and MES TAQ.

Data calls request `response_format=csv_gzip`, ascending `EventDateTime`, the
causal `columns=` projection below, and explicit `limit`/`offset`. The
documented maximum gzip CSV page size is 80,000.

Multiple Depth requests retain:

```text
TradeDate,EventDateTime,Ticker,SecurityID,Side,Flags,
L1Price,L1Size,L1Orders,...,L10Price,L10Size,L10Orders
```

ES and MES Trade & Quote requests retain:

```text
TradeDate,EventDateTime,Ticker,SecurityID,EventType,Price,Quantity,Flags,TypeMask
```

These fields are the causal contract consumed by the adapter. Existing
full-schema CSV/GZIP pages remain readable because the adapter requires only
these fields. `BaseSymbol`, Multiple Depth `Depth`, and TAQ `Orders` are not
causal inputs and are no longer requested.
`X-Pagination-Next-Offset` controls progress; a missing value completes that
query. Every persisted page is independently readable gzip CSV, including the
CSV header which the provider sends only on offset zero.

For a logical `YYYY-MM-DD` session the input window is `[17:00 CT previous
date, 16:00 CT session date)`. The downloader creates 23 one-hour
`[start,end)` partitions in returned `America/Chicago` time and sends each
partition to its owning provider `TradeDate`. Production probes established
that the API applies `EventDateTime.ge`/`EventDateTime.lt` in an Eastern
wall-clock frame even though returned futures `EventDateTime` values are
Chicago wall-clock values. The downloader therefore converts the desired
Chicago bounds to `America/New_York` with IANA timezone rules; it does not
hardcode a one-hour offset. Returned rows are still filtered and validated
against the desired Chicago partition locally. The manifest records both
boundaries, first/last timestamps, row/page counts, and completion status for
every feed/hour. Q1's explicit mapping is ESH3/MESH3 through 2023-03-12 and
ESM3/MESM3 from 2023-03-13.

An active partition must contain rows through the final five minutes of its
requested local hour. This is a conservative fail-closed guard against a
terminal page that silently stops early; a genuine no-event interval requires
manual review rather than being silently accepted.

Pages are written beneath the selected root:

```text
<root>/<session>/es-multiple-depth/chunk-000001.csv.gz
<root>/<session>/es-trade-and-quote/es-chunk-000001.csv.gz
<root>/<session>/mes-trade-and-quote/mes-chunk-000001.csv.gz
<root>/<session>/algoseek-download-manifest.json
<root>/<session>/algoseek-complete-session.json
```

Each page first becomes `*.part`, is gzip/schema/ticker/TradeDate validated,
hashed, then atomically renamed. The manifest records requests, translated
filters, partition coverage, offsets, response IDs, filenames and hashes—but
never credentials. `--resume`
re-hashes every recorded page, fails on a mismatch, and continues only from the
documented next offset. HTTP 429 observes `Retry-After` when present; network,
timeout and 5xx failures use bounded exponential backoff. HTTP 401 and 403 do
not retry. DNS/name-resolution, timeout, connection-reset, and other transient
network failures retry up to ten times after the initial request with
`2,4,8,16,30,60...` second delays and bounded jitter. HTTP 429 honors
`Retry-After`; 5xx responses use the same transient schedule. Each retry emits
feed, logical session, local partition, offset, attempt, error class, and next
delay telemetry. A client-level sliding-window limiter counts retries as
requests and keeps the request rate at or below 50 per minute.

After each verified page the gzip file is fsynced, atomically published, and
the manifest is fsynced. A failed process therefore resumes from the last
persisted page's provider continuation offset without re-downloading completed
pages or restarting the partition.

## Range and audit behavior

```sh
cme-l2-research algoseek-download-range \
  --start-date 2023-01-05 --end-date 2023-01-31 \
  --output-root data/algoseek/api-q1 --dry-run
```

The dry run lists active logical sessions, Q1 closed dates/weekends, contracts,
two provider dates, feed directories and the page-size ceiling without any
market-data calls. A real range processes one session at a time and stops on
the first failed audit unless `--continue-on-audit-failure` is explicitly set.
After all three feeds are present it runs the existing bounded-memory input
audit and streaming dry-build. The session manifest gains `session_complete:
true` only when all 23 expected partitions for each feed are present and
validated, in addition to the existing profile, canonical-event, depth-state,
and MES BBO checks. MES BBO coverage is not used as proof that source
partitions were complete.

The read-only reference comparison uses normalized causal row multisets:
timestamps are UTC nanoseconds, quarter-point prices are integer ticks, numeric
counts are integers, and empty values remain distinct from zero. It compares
the fields that remain in the causal projection; exact full-schema raw-row
parity is not expected when one root contains legacy full-schema pages and the
other contains projected pages. The comparison remains separate from ordered-
stream comparison because Algoseek does not provide a stable cross-stream
causal sequence key.
