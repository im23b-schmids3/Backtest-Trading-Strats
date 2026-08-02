# CLI usage

Install the project in editable mode, then initialize the default registry:

```text
pip install -e .
python -m research_pipeline init
```

Typical fixture inspection:

```text
python -m research_pipeline new-strategy examples/research_pipeline/fibonacci_compatibility.yaml
python -m research_pipeline list-strategies
python -m research_pipeline status fibonacci-compatibility
python -m research_pipeline validate-spec fibonacci-compatibility
python -m research_pipeline submit-spec fibonacci-compatibility
python -m research_pipeline approve-spec fibonacci-compatibility
python -m research_pipeline show-budget fibonacci-compatibility
python -m research_pipeline history fibonacci-compatibility
```

Mocked Phase A dry run (the transitions are registry-only and do not run a
backtest):

```text
python -m research_pipeline --registry research_registry/dry_run.sqlite3 init
python -m research_pipeline --registry research_registry/dry_run.sqlite3 new-strategy examples/research_pipeline/fibonacci_compatibility.yaml
python -m research_pipeline --registry research_registry/dry_run.sqlite3 submit-spec fibonacci-compatibility
python -m research_pipeline --registry research_registry/dry_run.sqlite3 approve-spec fibonacci-compatibility
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility IMPLEMENTATION_VERIFICATION --reason "mock verification"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility BASELINE_BACKTEST --reason "mock baseline"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility EDGE_GATE --reason "mock gate"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 create-split fibonacci-compatibility examples/research_pipeline/mock_split.yaml
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility PARAMETER_RESEARCH --reason "mock research"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 consume-budget fibonacci-compatibility --backtests 5 --family entry --rounds 1 --values 2
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility CANDIDATE_FREEZE --reason "mock freeze"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility WALK_FORWARD --reason "mock walk forward"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility HOLDOUT --reason "mock holdout"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 open-holdout fibonacci-compatibility --reason "final validation"
python -m research_pipeline --registry research_registry/dry_run.sqlite3 status fibonacci-compatibility
# Expected failure with exit code 2: terminal/invalid phase retry
python -m research_pipeline --registry research_registry/dry_run.sqlite3 transition fibonacci-compatibility HOLDOUT --reason "intentional invalid retry"
```

Use `--registry PATH` before the command for an isolated registry. `transition`
requires `--reason`; `record-decision` accepts either a JSON file path or an
inline JSON document. CLI errors return exit code 2 and do not invoke any
backtest or research code.
