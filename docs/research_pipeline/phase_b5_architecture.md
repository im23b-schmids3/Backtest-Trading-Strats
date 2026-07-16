# Phase B.5 architecture

Phase B.5 is the mandatory mechanical integrity gate between implementation verification and baseline research. It consumes compact JSON or CSV diagnostic artifacts, recomputes numerical results, persists evidence, and transitions only a `VERIFIED` result to `BASELINE_BACKTEST`. It does not run optimization, open holdout data, or change strategy parameters.

The state path is `IMPLEMENTATION_VERIFICATION -> TECHNICAL_INTEGRITY_VERIFICATION -> BASELINE_BACKTEST`. Proven deterministic defects become `TECHNICAL_REPAIR_REQUIRED`; semantic ambiguity becomes `MANUAL_REVIEW_REQUIRED`; missing artifacts become `INSUFFICIENT_DIAGNOSTIC_DATA`.
