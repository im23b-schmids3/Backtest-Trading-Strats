# Phase D architecture

Phase D is an additive, deterministic futures and prop-account compatibility
layer. It consumes a Phase C accepted, frozen candidate and never changes its
parameters or reruns strategy research. The Phase C `ACCEPTED` state remains a
terminal state; Phase D has a separate durable `prop_runs` state machine so
prop classification cannot reopen or mutate Phase C evidence.

The service sequence is entry verification -> provider rules -> contract and
market mappings -> futures reconciliation/B5 evidence -> risk sizing -> prop
simulation -> economics review. Every artifact is JSON-hashed where applicable
and every phase writes SQLite records plus a prop journal.

Phase D does not combine strategies, trade accounts, access holdout data, or
run optimization. Synthetic and proxy scenarios are fixtures for lifecycle and
arithmetic tests, not performance evidence.
