# Natural-language specification generation

Phase F1/F2 intake is a bounded, deterministic bridge from an `IntakeSpec` to a canonical `StrategySpec`. The Strategy Spec Agent receives only the normalized intake and is instructed to return exactly one YAML or JSON mapping. It does not implement code, run backtests, select performance parameters, or modify the repository.

Each intake has a deterministic run identifier (`spec-<strategy>-<hash>`) unless an explicit run identifier is supplied for a resumed workflow. Artifacts live under `research_runs/<strategy_id>/<run_id>/specification/`:

- `intake/natural_language.json`
- `attempts/attempt-NNN/draft.yaml`
- `attempts/attempt-NNN/codex_invocation.json`
- `attempts/attempt-NNN/validation.json`
- `attempts/attempt-NNN/semantic_validation.json`
- `attempts/attempt-NNN/repair_prompt.md` when repair is requested
- `canonical/specification.yaml` and `canonical/hash_manifest.json` after validation
- `failure/final_failure.json` after exhaustion

The default bound is three generation attempts: one initial call and at most two repairs. A valid canonical artifact is reused on resume; the Codex process is not called again.
