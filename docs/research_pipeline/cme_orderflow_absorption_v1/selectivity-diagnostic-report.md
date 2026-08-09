# Selectivity diagnostic

Legacy `ABSORPTION_PLUS_REPLENISHMENT` was an incorrect union: PASS (repaired to same-interaction intersection).

## Frozen tier construction

Scores use contemporaneous interaction-end fields only; equal-weight full-pilot p95 thresholds were frozen before any response distribution was computed. Future response values are not score or threshold inputs.

Thresholds: `{'absorption_p95': 0.7977986403366786, 'replenishment_p95': 0.7785691162188411}`.
Tier totals: `{'RAW_INTERACTION': 3089, 'HIGH_ABSORPTION': 155, 'STRONG_REPLENISHMENT': 155, 'ABSORPTION_PLUS_REPLENISHMENT': 33}`. Per-RTH averages: `{'RAW_INTERACTION': 308.9, 'HIGH_ABSORPTION': 15.5, 'STRONG_REPLENISHMENT': 15.5, 'ABSORPTION_PLUS_REPLENISHMENT': 3.3}`.
All ten daily counts: `{'2026-07-20': {'RAW_INTERACTION': 186, 'HIGH_ABSORPTION': 11, 'STRONG_REPLENISHMENT': 2, 'ABSORPTION_PLUS_REPLENISHMENT': 1}, '2026-07-21': {'RAW_INTERACTION': 220, 'HIGH_ABSORPTION': 11, 'STRONG_REPLENISHMENT': 1, 'ABSORPTION_PLUS_REPLENISHMENT': 0}, '2026-07-22': {'RAW_INTERACTION': 407, 'HIGH_ABSORPTION': 16, 'STRONG_REPLENISHMENT': 0, 'ABSORPTION_PLUS_REPLENISHMENT': 0}, '2026-07-23': {'RAW_INTERACTION': 93, 'HIGH_ABSORPTION': 14, 'STRONG_REPLENISHMENT': 0, 'ABSORPTION_PLUS_REPLENISHMENT': 0}, '2026-07-24': {'RAW_INTERACTION': 278, 'HIGH_ABSORPTION': 12, 'STRONG_REPLENISHMENT': 8, 'ABSORPTION_PLUS_REPLENISHMENT': 3}, '2026-07-27': {'RAW_INTERACTION': 627, 'HIGH_ABSORPTION': 21, 'STRONG_REPLENISHMENT': 35, 'ABSORPTION_PLUS_REPLENISHMENT': 8}, '2026-07-28': {'RAW_INTERACTION': 267, 'HIGH_ABSORPTION': 16, 'STRONG_REPLENISHMENT': 30, 'ABSORPTION_PLUS_REPLENISHMENT': 8}, '2026-07-29': {'RAW_INTERACTION': 341, 'HIGH_ABSORPTION': 17, 'STRONG_REPLENISHMENT': 30, 'ABSORPTION_PLUS_REPLENISHMENT': 1}, '2026-07-30': {'RAW_INTERACTION': 353, 'HIGH_ABSORPTION': 15, 'STRONG_REPLENISHMENT': 7, 'ABSORPTION_PLUS_REPLENISHMENT': 2}, '2026-07-31': {'RAW_INTERACTION': 317, 'HIGH_ABSORPTION': 22, 'STRONG_REPLENISHMENT': 42, 'ABSORPTION_PLUS_REPLENISHMENT': 10}}`.
HIGH and PLUS direction counts: `{'HIGH_ABSORPTION': {'SELLER_ABSORPTION': 79, 'BUYER_ABSORPTION': 76}, 'ABSORPTION_PLUS_REPLENISHMENT': {'BUYER_ABSORPTION': 24, 'SELLER_ABSORPTION': 9}}`. PLUS counts by required structural level: `{'PRIOR_RTH_HIGH': 0, 'PRIOR_RTH_LOW': 0, 'PRIOR_RTH_POC': 0, 'PRIOR_RTH_VAH': 0, 'PRIOR_RTH_VAL': 2, 'CURRENT_RTH_HIGH_SWEEP': 8, 'CURRENT_RTH_LOW_SWEEP': 23}`.
Subset checks (global and daily): `{'global_pass': True, 'per_day_pass': {'2026-07-20': True, '2026-07-21': True, '2026-07-22': True, '2026-07-23': True, '2026-07-24': True, '2026-07-27': True, '2026-07-28': True, '2026-07-29': True, '2026-07-30': True, '2026-07-31': True}, 'pass': True}`.
Response sanity violations: `0`.

## Descriptive responses after freezing

Signed-tick summaries at 5s, 15s, 30s, 60s, and 120s in the refined machine-readable report are descriptive only; they were not used for tier construction or threshold choice.

Final status: `READY_FOR_SMALL_BACKTEST_DESIGN`.
