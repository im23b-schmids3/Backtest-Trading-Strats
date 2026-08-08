# Non-optimized feature diagnostics

Labels are descriptive only: PROBABLE_REPLENISHMENT, ABSORPTION_CANDIDATE, and NO_CLEAR_REPLENISHMENT. No final threshold was chosen; no true/confirmed iceberg inference is made. Prior-RTH high/low/POC context is frozen before each RTH; VAH/VAL unavailable without a sealed profile algorithm and is explicitly not invented. Execution-only T/F records retain unknown aggressor rather than forcing a side; queue position and hidden quantity are unavailable.

Bounded summary-only probable-replenishment examples:

```json
[
  {
    "date": "2026-07-31",
    "level": "CURRENT_RTH_HIGH_SWEEP",
    "price_es": 7540.5,
    "label": "PROBABLE_REPLENISHMENT_INTERACTION"
  },
  {
    "date": "2026-07-31",
    "level": "CURRENT_RTH_HIGH_SWEEP",
    "price_es": 7540.25,
    "label": "PROBABLE_REPLENISHMENT_INTERACTION"
  },
  {
    "date": "2026-07-31",
    "level": "CURRENT_RTH_HIGH_SWEEP",
    "price_es": 7540.0,
    "label": "PROBABLE_REPLENISHMENT_INTERACTION"
  },
  {
    "date": "2026-07-31",
    "level": "CURRENT_RTH_HIGH_SWEEP",
    "price_es": 7540.0,
    "label": "PROBABLE_REPLENISHMENT_INTERACTION"
  },
  {
    "date": "2026-07-31",
    "level": "CURRENT_RTH_HIGH_SWEEP",
    "price_es": 7540.0,
    "label": "PROBABLE_REPLENISHMENT_INTERACTION"
  }
]
```

Suitability conclusion: suitable for descriptive MBO lifecycle and displayed-replenishment diagnostics if reconstruction completes without fail-closed errors; unsuitable for hidden-liquidity proof, queue-position claims, threshold selection, or profitability claims.
