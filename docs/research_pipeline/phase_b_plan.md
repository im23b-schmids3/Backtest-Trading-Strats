# Phase B plan

Phase B can connect Smithers to the Phase A controller as a durable workflow.
Smithers steps should call deterministic controller methods, persist runner
artifacts as experiments, and require human approval at specification,
holdout, and final-review boundaries. Codex can later produce a validated
`DecisionRecord` from inspected reports, but the controller should remain the
authority for legal transitions, budgets, split hashes, and holdout access.

The integration should add explicit runner adapters, retry and repair policies,
artifact manifests, and Smithers replay tests. It must not give an AI direct
authority to change invariants, rewrite approved specifications, open a holdout
twice, or bypass a hard budget. No Smithers workflow or autonomous agent is
implemented in Phase A.

