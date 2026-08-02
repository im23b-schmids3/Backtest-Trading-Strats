# Tenant-compatible specification intake

Some tenants permit the Smithers process but deny subprocess execution of the
locally authenticated Codex CLI. The pipeline treats this as a typed pause,
not as a specification failure and not as permission to bypass tenant policy.

The Smithers run reports one of:

- `WAITING_EXTERNAL_SPECIFICATION_GENERATION`
- `WAITING_EXTERNAL_SPECIFICATION_REPAIR`
- `WAITING_EXTERNAL_CODEX` for the existing implementation boundary

The Python status includes the exact external command and job identifier. Run
the printed specification executor command from the primary repository only
after inspecting the request. The executor uses the existing local Codex
authentication; no API key is added to prompts, SQLite, or logs. It runs in
read-only sandbox mode and communicates the prompt through stdin.

After completion, resume the same Smithers run. Smithers then calls the
deterministic Python controller, which validates and ingests the result. There
is no fixture fallback, synthetic specification shortcut, or automatic policy
override.
