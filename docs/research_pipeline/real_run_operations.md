# Real run operations

Start a real run with the normal F1 entry point:

```powershell
py -m research_pipeline --registry C:/temp/f2.sqlite3 run configs/research_pipeline/phase_f2_real_demo_intake.yaml --mode real_run --allow-proxy-data --prebuilt-spec research_registry/spec_drafts/F2-real-breakout-demo_vphase-b-1.yaml
py -m research_pipeline --registry C:/temp/f2.sqlite3 approve RUN_ID --decision APPROVE
py -m research_pipeline --registry C:/temp/f2.sqlite3 resume RUN_ID --repository-root .
py -m research_pipeline --registry C:/temp/f2.sqlite3 verify-data RUN_ID
py -m research_pipeline --registry C:/temp/f2.sqlite3 report RUN_ID
```

The prebuilt-spec option is for deterministic local integration. New strategy
families must use the Codex-backed specification and implementation path.

Smithers operations (installed CLI 0.28.0):

```powershell
bunx smithers-orchestrator graph .smithers/workflows/trading-research-master.tsx
bunx smithers-orchestrator up .smithers/workflows/trading-research-master.tsx --detach --input '<JSON matching .smithers/schemas/trading-research/master.ts>'
bunx smithers-orchestrator inspect RUN_ID
bunx smithers-orchestrator approve RUN_ID --node specification-approval --by OPERATOR --note 'approved after specification review'
bunx smithers-orchestrator up .smithers/workflows/trading-research-master.tsx --resume RUN_ID --input '<same JSON input>'
```

The workflow has one approval gate, `specification-approval`. Use
`bunx smithers-orchestrator deny RUN_ID --node specification-approval` to
reject it. Smithers persists task state; the Python registry remains the
authoritative source for strategy phase, adapter, artifact, and report state.
