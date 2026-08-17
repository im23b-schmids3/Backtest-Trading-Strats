# L2 V3 POC-only execution-calendar clarification

Classification: `EXECUTION_CALENDAR_CLARIFICATION`.

The V3 signal and score contract remains hash-bound to
`a0ce94eeb78dcbf865cf4464bdf97ebc3f014a8ec5e2f559f99798534dfcbcb4` and
is unchanged. This document only resolves unavailable-market execution on the
fresh validation block.

## Rule

```text
effective_hard_flat = min(frozen_22:45:00_UTC, scheduled_market_close)
liquidation_window = [effective_hard_flat - 1 second, effective_hard_flat]
```

For Monday–Thursday, effective hard-flat remains 22:45 UTC. For Friday
2026-08-14, the existing project Friday convention records the scheduled close
as 21:00 UTC (see the local August Friday source audit). Its effective
hard-flat is therefore 21:00 UTC and its liquidation window is inclusive
20:59:59–21:00:00 UTC.

An open position must exit from the last causally valid executable native BBO
inside that window, applying the existing adverse exit handling. With no such
BBO, the replay fails closed. It may not invent a post-close BBO, fill, price,
or timestamp. This is not a target, stop, confirmation, signal, or sizing
change; it applies only because the exchange cannot execute after its
scheduled market close.

## Native-book maintenance state

For the August 10–14 UTC dates, the project execution calendar also treats
21:00:00–22:00:00 UTC as the regular Globex maintenance/pause interval. This
is a source-executability condition, not a new signal or exit rule. Native
MBP-10/MES MBP-1 observations in that interval cannot publish an executable
BBO, create an interaction, confirm a setup, or enter a trade. A valid
two-sided native ES MBP-10 book after 22:00 UTC is required before normal
execution resumes. Quotes are cleared at the transition and are never carried
across it.

If a position is open when the maintenance pause begins, the frozen contract
does not state whether it may be held through the halt. The runner therefore
fails closed with `MAINTENANCE_HALT_WITH_OPEN_POSITION_UNDEFINED_BY_FROZEN_CONTRACT`;
it does not invent a pause exit or a new holding rule.
