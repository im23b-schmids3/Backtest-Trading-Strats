# Phase F1 architecture

Phase F1 is an additive master-run layer. It owns intake normalization,
one approval boundary, phase ordering, resume state, phase timings, artifact
references, journals, report generation, and archive manifests. Existing Phase
B, B.5, C, D, and E services remain the authorities for their own semantics.

Master records are stored in `master_runs`, `master_phase_results`,
`master_journal`, `master_artifacts`, and `master_reports`. Large or structured
phase evidence remains in the existing phase tables and deterministic output
folders.
