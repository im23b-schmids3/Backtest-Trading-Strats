# Fib V1 development-loader isolation

`FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION` has a sealed
development interval of `[2022-01-01T00:00:00+00:00,
2025-01-01T00:00:00+00:00)`. Holdout begins at
`2025-01-01T00:00:00+00:00` and remains locked.

The loader distinguishes a physical Parquet read from logical public exposure.
Row-group timestamp metadata is first validated. Fully holdout row groups are
skipped. If a physical row group crosses the sealed boundary, the private loader
may decode that group solely to apply the sealed timestamp predicate immediately.
Only the filtered development Bars leave the loader; no public DataFrame or
Arrow Table is returned.

The exact invariant is `NO_HOLDOUT_LOGICAL_EXPOSURE`.

The loader fails closed when row-group statistics cannot prove the predicate,
or when manifest hash, source hash, schema, chronology, development count, or
timestamp validation fails. Isolation metadata records only
`physical_row_groups_read`, `mixed_row_groups_read`,
`development_rows_returned`, and `holdout_rows_discarded`; it never records
discarded values.

For the sealed contracts, the returned development claims are exactly 6576 ETH
4-hour Bars ending `2024-12-31T20:00:00+00:00` and 1096 BTC daily Bars ending
`2024-12-31T00:00:00+00:00`. Both maxima are strictly before holdout start.
