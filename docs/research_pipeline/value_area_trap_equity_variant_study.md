# QQQUSDT/SPYUSDT exploratory in-sample variant study

This is a sealed, descriptive-only `EXPLORATORY_IN_SAMPLE_VARIANT_STUDY` for
the existing May--July 2026 Binance USD-M QQQUSDT and SPYUSDT manifests. It is
separate from the completed BTC frozen run and the frozen cross-market study.
It cannot select, recommend, promote, or confirm a variant and every result
requires future untouched holdout evidence.

The registry is fixed before execution to exactly six variants: A--F. The
runner rejects any missing, added, or altered variant. It emits one immutable
specification, immutable parameter manifest, and deterministic run root per
symbol/variant pair (twelve total), plus an immutable comparison report.

UTC variants use a completed UTC calendar-day profile. `US_CASH_SESSION` uses
`America/New_York` 09:30--16:00 boundaries via `zoneinfo`, so DST is resolved
from timezone rules rather than a fixed UTC offset. Weekends and the three
pinned full-day NYSE closures in this study window (2026-05-25, 2026-06-19,
2026-07-03) are excluded. Only sessions with observed 09:30 and 16:00
five-minute boundaries are retained; missing/incomplete sessions are reported.

The funnel uses the explicit valid alternative:

```
proposed_setups = invalid_setups + non_executable_setups + compliance_blocks + executed_trades
```

`non_executable_setups` covers an otherwise valid proposed setup whose required
next-bar entry is outside the completed session. This preserves all proposals
without silently treating them as invalid or executed.

Validate existing immutable manifests without network access:

```powershell
python -m research_pipeline value-area-trap validate-equity-variant-study `
  --manifest QQQUSDT=./data/value_area_trap/normalized/QQQUSDT/8c1c024335c2cfbebb0249e5c00ff3b3527127bd06048f3ba15db03d41ca997c/manifest.json `
  --manifest SPYUSDT=./data/value_area_trap/normalized/SPYUSDT/58d55e303a081e402ecb3b52d2158fac04efadd84f2ec7d088b875c27b8bf1a2/manifest.json
```

The execution command is intentionally explicit and performs no downloads or
normalization. It reuses only hash-verified partitions named by the manifests.
