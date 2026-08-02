# Phase B.5 operations

Create a deterministic fixture:

```powershell
py -m research_pipeline verification fixture STRATEGY_ID --kind correct --output .tmp\b5-correct
py -m research_pipeline verification run STRATEGY_ID --manifest .tmp\b5-correct\manifest.yaml
```

Defect fixtures include `missing-multiplier`, `duplicate-fee`, `partial-exit`, `scaling`, `ambiguous-count`, `lookahead`, `terminal-flatten`, `report-mismatch`, `proxy-unlabeled`, `nondeterministic`, and `missing-diagnostics`. Inspect results with `verification status`, `show-failures`, or `export-defect-prompt`.

The Smithers workflow is rendered with `smithers graph .smithers/workflows/trading-research-phase-b5.tsx` and run with `smithers up .smithers/workflows/trading-research-phase-b5.tsx --input '{...}'`. It resumes durably and does not request a second normal approval.
