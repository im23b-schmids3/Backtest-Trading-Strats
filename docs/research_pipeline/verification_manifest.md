# Verification manifest

A manifest is YAML validated by `VerificationManifest`. It binds strategy/version, implementation commit, unique verification run, diagnostic files, tolerances, required checks, capability flags, data-source declarations, expected contracts/sessions, exemptions, and an approved-invariants hash. Its canonical contents are hashed in `manifest_hash`.

Create one with:

```powershell
py -m research_pipeline verification create-manifest STRATEGY_ID --diagnostic-dir PATH
```
