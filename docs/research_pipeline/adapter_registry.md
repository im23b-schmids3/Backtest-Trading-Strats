# Adapter registry

Register an explicit strategy-family factory in
`research_pipeline.adapters.registry`. The built-in families are
`f2_native_demo` and `f2_native`. The registry performs import, identity,
capability, and health checks and never falls back to synthetic fixtures in
real mode.

```powershell
py -m research_pipeline adapters list
py -m research_pipeline adapters inspect F2-real-breakout-demo
py -m research_pipeline adapters validate F2-real-breakout-demo
py -m research_pipeline adapters capabilities F2-real-breakout-demo
```
