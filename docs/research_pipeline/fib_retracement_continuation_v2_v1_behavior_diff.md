# Fib09 V1/V2 behavior diff

V1 is frozen.  V2 is now an execution-layer adapter only.

| Subsystem | Fault found in prior V2 | Repair | Economic impact |
| --- | --- | --- | --- |
| Signals | Reimplemented `causal_setups`, generated both directions from every extreme, and used V2 IDs. | Direct V1 `causal_setups`/`create_setup`/expiry and V1 candidate registry. | Explains BTC setup inflation (173 to 691): V2 emitted a setup on every qualifying extreme instead of V1's anchor-promotion lifecycle. |
| Activation | V2 scheduled every generated setup independently. | A setup activates at exactly the next completed V1 signal-bar timestamp; 1m is execution-only. | Removes 1m/over-frequent opportunity creation. |
| IDs and fills | V2 identities used V2-specific payloads and order version 2. | Reuses V1 IDs, order version 1, sizing, adverse fill, partials, and accounting. | Restores reproducible order/trade lineage and comparable P&L. |
| Costs and gates | V2 duplicated costs and omitted the sealed additional-slippage hard gate. | Reuses V1 execution/accounting/metrics/gates. | Restores the V1 fee, slippage, quantity and stress economics. |
| Session | Pending cancellation existed, but reconciliation did not prove session invariants. | 22:45 UTC first force-closes at 1m open with normal adverse cost, then cancels pending orders; reconciliation rejects overnight/cutoff/pending/invalid forced exits. | Prevents overnight exposure and makes the forced-close cost auditable. |
| Diagnostics | No V1/V2 signal comparison. | Artifact-free development-only parity diagnostic compares reference/derived bars and setup provenance. | Equal HTF bars must produce zero implementation differences; unequal bars are classified data-source differences. |

The prior overnight defect was not an entry-rule signal defect: reconciliation had no checks for a position surviving the cutoff, a cutoff entry, a pending order surviving the cutoff, or a forced-exit timestamp/open-price assertion.  The repaired runner applies the force close before intrabar processing and the repaired reconciliation makes any such lifecycle fail closed.
