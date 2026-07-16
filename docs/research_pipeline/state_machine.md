# State machine

The legal transitions are declared explicitly in
`research_pipeline.controller.state_machine.LEGAL_TRANSITIONS`. The principal
sequence is:

`STRATEGY_DRAFT -> WAITING_FOR_SPEC_APPROVAL -> IMPLEMENTATION ->
IMPLEMENTATION_VERIFICATION -> TECHNICAL_INTEGRITY_VERIFICATION -> BASELINE_BACKTEST -> EDGE_GATE ->
PARAMETER_RESEARCH -> CANDIDATE_FREEZE -> WALK_FORWARD -> HOLDOUT ->
STRESS_TESTS -> THROUGHPUT -> RISK_SIZING -> PROP_SIMULATION ->
MULTI_STRATEGY_PORTFOLIO -> FINAL_REVIEW`.

Gate and review states may terminate in `REJECTED`, `INSUFFICIENT_EVIDENCE`,
or `MANUAL_REVIEW_REQUIRED`. Technical phases may terminate in
`TECHNICAL_FAILURE`. All terminal states reject further transitions.
