# Non-optimized feature diagnostics

Labels are descriptive only: PROBABLE_REPLENISHMENT, ABSORPTION_CANDIDATE, and NO_CLEAR_REPLENISHMENT. No final threshold was chosen; no true/confirmed iceberg inference is made. Prior-RTH high/low/POC context is frozen before each RTH; VAH/VAL unavailable without a sealed profile algorithm and is explicitly not invented. Execution-only T/F records retain unknown aggressor rather than forcing a side; queue position and hidden quantity are unavailable.

Bounded summary-only probable-replenishment examples:

```json
[
  {
    "date": "2026-07-20",
    "price": "integer DBN price scale; raw event identifiers omitted",
    "passive_side": "B",
    "executed_size_before_add": 5,
    "label": "PROBABLE_REPLENISHMENT"
  },
  {
    "date": "2026-07-20",
    "price": "integer DBN price scale; raw event identifiers omitted",
    "passive_side": "B",
    "executed_size_before_add": 1,
    "label": "PROBABLE_REPLENISHMENT"
  },
  {
    "date": "2026-07-20",
    "price": "integer DBN price scale; raw event identifiers omitted",
    "passive_side": "A",
    "executed_size_before_add": 2,
    "label": "PROBABLE_REPLENISHMENT"
  },
  {
    "date": "2026-07-20",
    "price": "integer DBN price scale; raw event identifiers omitted",
    "passive_side": "A",
    "executed_size_before_add": 1,
    "label": "PROBABLE_REPLENISHMENT"
  },
  {
    "date": "2026-07-20",
    "price": "integer DBN price scale; raw event identifiers omitted",
    "passive_side": "A",
    "executed_size_before_add": 1,
    "label": "PROBABLE_REPLENISHMENT"
  }
]
```

Suitability conclusion: suitable for descriptive MBO lifecycle and displayed-replenishment diagnostics if reconstruction completes without fail-closed errors; unsuitable for hidden-liquidity proof, queue-position claims, threshold selection, or profitability claims.
