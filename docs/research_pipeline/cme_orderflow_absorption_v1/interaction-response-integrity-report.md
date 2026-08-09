# Interaction and response integrity

## Root causes and repair
The prior interaction key included every current-session extreme price and lifecycle closure occurred on any out-of-vicinity MBO event. This fragmented visits into message-level interactions. Fixed prior levels are now keyed by immutable level and current sweeps retain one lifecycle while their extreme is revised; only executed ES trades create, close, or reset a visit.

Response lookup previously used the next arbitrary applied MBO record (including displayed-order prices). It now uses the first valid ES execution at or after each horizon. Prices remain DBN fixed-point integers internally, are converted by 1e9 only for presentation, and signed ticks are `(future ES price - reference ES price) / 0.25`.

## Cardinality and response integrity
Before repair RAW interactions: 3,664,178. After repair RAW interactions: 3089.
Tier counts: {'RAW_INTERACTION': 3089, 'HIGH_ABSORPTION': 155, 'STRONG_REPLENISHMENT': 155, 'ABSORPTION_PLUS_REPLENISHMENT': 33}. Per-RTH practical counts: {'RAW_INTERACTION': 308.9, 'HIGH_ABSORPTION': 15.5, 'STRONG_REPLENISHMENT': 15.5, 'ABSORPTION_PLUS_REPLENISHMENT': 3.3}.
Response integrity: {'pass': True, 'sanity_violation_count': 0, 'excluded_from_descriptive_distributions': True}.

## Sanity-violation audit
All `abs(response_ticks) > 500` observations are classified as `RESPONSE_SANITY_VIOLATION`, excluded from descriptive distributions, and retained here as bounded audit evidence. Count: 0.
```json
[]
```

## Final status
READY_FOR_SMALL_BACKTEST_DESIGN

No trading strategy, PnL, optimization, download, or market-data mutation was performed.
