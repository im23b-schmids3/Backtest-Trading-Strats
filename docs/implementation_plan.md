# Implementation plan

1. Define immutable configuration, data contracts, cache layout, and validation rules. **Complete**
2. Implement confirmed (delayed) swing detection and exact Fibonacci setup formulas with unit tests. **Complete**
3. Implement an event-driven portfolio engine with limit entries, sizing, partial exits, costs, and deterministic intrabar policies. **Complete; validation audit in progress**
4. Add CCXT/Yahoo acquisition, CLI, reporting, documentation, and end-to-end validation. **In progress**

The engine consumes only rows that have closed. A pivot at index `i` is emitted on index `i + N`; an order is considered no earlier than the next candle. All order lifecycle timestamps are retained in the trade log.
