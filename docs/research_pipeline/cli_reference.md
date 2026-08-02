# Phase F1 CLI reference

```text
py -m research_pipeline run STRATEGY_FILE [--dry-run]
py -m research_pipeline status RUN_ID
py -m research_pipeline approve RUN_ID [--decision APPROVE|REJECT]
py -m research_pipeline resume RUN_ID
py -m research_pipeline report RUN_ID
py -m research_pipeline artifacts RUN_ID
py -m research_pipeline cancel RUN_ID
```

The registry is selected with the global `--registry PATH` option. The default
dry run uses the deterministic adapters and does not modify trading strategy
source files. `implementation_enabled` must be explicitly enabled, and live or
broker execution is not part of F1.
