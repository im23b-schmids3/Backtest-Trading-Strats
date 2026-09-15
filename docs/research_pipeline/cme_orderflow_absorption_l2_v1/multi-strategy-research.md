# Multi-strategy development research

`cme-l2-research` adds an offline-first two-stage workflow for strategies that
use the canonical L2 absorption engine. It is development/retrospective
research only and never promotes a strategy to production.

## Required inputs

The period manifest names completed causal-tape artifacts: one interaction
master Parquet file, one interaction-index Parquet file, and one event-tape
Parquet file per ordered session. It deliberately does not name DBN files.
The heavy raw-data replay must happen once before this workflow using the
existing canonical causal-tape builders. The new stages never download data or
open DBN files.

The strategy manifest identifies a strategy, its session, reference level,
trigger options, and whether it uses the shared absorption engine. Interactions
are filtered by `reference_level`; all strategies can share a period's event
tapes and interaction master.

For an auditable candidate universe, a strategy may additionally declare its
`source_session`, `reference_semantics`, `causal_availability_rule`,
`implementation_status`, and `level_resolver_required`. These fields bind the
provenance of a level; they do not relax causality. A cross-session row can run
only after a causal-tape builder has emitted that exact `reference_level` for
the target-session interaction population. The workflow will not construct a
profile from future target-session data.

Cross-session strategies additionally require one shared `level_catalog` JSON
sidecar in the period manifest. It contains generic `session_relationships`
(`target_session`, trading date, source session/date, and semantic relation)
and `level_observations` (family, value, causal availability timestamp, mode,
and source-artifact provenance). It is not a file per strategy. Prior-session
relationships are explicit rather than date arithmetic, so gaps and weekends
remain canonical. Dynamic high/low rows use the latest observation at or before
the interaction start; completed POC/VAH/VAL rows are unavailable until their
recorded profile-completion timestamp.

`multi-strategy-level-audit --strategies strategies.yaml` performs a no-data
capability audit. It reports supported resolver modes, not the presence of
historical market data or a performance result.

See the manifests in `examples/research_pipeline/` for the exact format.
The small `cme_l2_multi_strategy.example.yaml` remains suitable for focused
runs. `cme_l2_candidate_universe.example.yaml` is an explicit development
universe and is never chosen automatically; users must still pass it (or a
smaller copied subset) with `--strategies`.

The causal rationale, accepted/rejected cross-session references, and scale
estimate are recorded in `candidate-universe.md`.

## Commands

```text
cme-l2-research multi-strategy-screen --strategies strategies.yaml --period period.yaml --output stage1 --workers 2
cme-l2-research multi-strategy-optimize --strategies strategies.yaml --period period.yaml --stage1-results stage1 --output stage2 --workers 2
```

Stage 1 uses Q=0.50 and the legal common W04 vector
`0.20/0.10/0.30/0.20/0.20`. It evaluates RR `1.5, 2.0, 2.5, 3.0, 4.0` by stop
buffer `3, 5, 7` ticks. A long stop remains `stop_ticks` below the interaction
zone low; a short stop remains `stop_ticks` above the zone high. The target is
the selected R multiple of that same entry-to-stop distance. Entry, confirmation,
fill, sizing, and exit semantics are unchanged.

The W04 baseline is the component-wise median of the existing prior-Europe
high, current-Europe-high-sweep, and prior-Europe-VAH W04 vectors and the
current prior-New-York-POC V3 inherited vector. It is not an invented weight.

Stage 2 freezes each strategy's Stage-1 geometry and evaluates the existing
canonical 3,876 legal weight vectors against Q=0.30 through Q=0.75 in 0.05
increments: 38,760 configurations per strategy. It uses the causal tapes only.
Within each strategy it materializes each session tape once for the selected
RR/stop geometry and reuses that immutable in-memory index for every weight/Q
configuration.

Each strategy has an isolated output directory keyed by a stable ID hash.
`complete.json` binds a strategy result to source artifact hashes and parameter
identity. Matching completed work is reused; changed inputs are rejected rather
than mixed. Stage 2 also checkpoints one complete ten-Q batch per legal weight
vector, so an interrupted strategy resumes from verified batches rather than
discarding already-completed weight evaluations. Different strategies can run
in separate processes with stable, sorted combined summaries.
