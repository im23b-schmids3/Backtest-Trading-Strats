"""Compact summary-only report writers."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
from .analysis import Diagnostics
from .loader import MetadataSummary

def write_reports(out: Path, *, sha256: str, dbn_bytes: int, metadata: MetadataSummary,
                  diagnostics: Diagnostics, integrity: str) -> None:
    out.mkdir(parents=True, exist_ok=True); diagnostics.finalize()
    # The sealed pilot consists of ten UTC weekdays; any weekend session
    # records remain counted in the manifest but are excluded from daily study metrics.
    daily = [{"date":d, **vars(m)} for d,m in sorted(diagnostics.days.items()) if date.fromisoformat(d).weekday() < 5]
    manifest = {"study":"CMEOrderflowAbsorption.ES_V1_PILOT","read_only":True,"dbn_sha256":sha256,
      "dbn_bytes":dbn_bytes,"metadata":vars(metadata),"event_count":diagnostics.events,
      "execution_count":diagnostics.executions,"integrity":integrity,"trading_strategy_executed":False}
    (out / "mbo-pilot-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    contract = {"engine":"causal order-id MBO reconstruction","ordering":"provider iterator order is preserved exactly; sequence is retained as diagnostic metadata but is not a provider-wide total order",
      "state":"active order-id map and side/price displayed depth","fail_closed":["unsupported action","unknown order operation","negative depth","invalid state transition","metadata/instrument mismatch"],
      "semantics":{"T/F":"execution-only evidence in this pilot; no undocumented displayed-book decrement is made because F may coexist with a subsequent cancel of the same quantity","aggressor":"not forced for execution-only records","queue_position":"UNAVAILABLE; no queue-position claim","hidden_quantity":"UNAVAILABLE; no iceberg claim"}}
    (out / "orderbook-reconstruction-contract.json").write_text(json.dumps(contract, indent=2)+"\n", encoding="utf-8")
    rows = "\n".join(f"| {x['date']} | {x['events']} | {x['executions']} | {x['adds']} | {x['cancels']} | {x['modifies']} | {x['probable_replenishment']} | {x['absorption_candidates']} | {x['no_clear_replenishment']} | {x['structural_tags']} |" for x in daily)
    (out / "mbo-validation-report.md").write_text(f"# ESU6 MBO validation\n\nRead-only deterministic pilot. SHA-256: `{sha256}`; bytes: {dbn_bytes}. Metadata: {metadata.dataset}, {metadata.schema}, {metadata.symbol}->{metadata.instrument_id}, UTC [{metadata.start_ns}, {metadata.end_ns}). Integrity: {integrity}.\n\n| UTC date | events | executions | adds | cancels | modifies | PROBABLE_REPLENISHMENT | ABSORPTION_CANDIDATE | NO_CLEAR_REPLENISHMENT | prior-RTH tags |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n{rows}\n\nData-quality issues: {dict(diagnostics.issues)}. Provider reset records: {sum(x['resets'] for x in daily)}. No raw events are included.\n", encoding="utf-8")
    examples = [{**x, "passive_side": str(x["passive_side"]), "price":"integer DBN price scale; raw event identifiers omitted"} for x in diagnostics.examples]
    (out / "feature-diagnostic-report.md").write_text("# Non-optimized feature diagnostics\n\nLabels are descriptive only: PROBABLE_REPLENISHMENT, ABSORPTION_CANDIDATE, and NO_CLEAR_REPLENISHMENT. No final threshold was chosen; no true/confirmed iceberg inference is made. Prior-RTH high/low/POC context is frozen before each RTH; VAH/VAL unavailable without a sealed profile algorithm and is explicitly not invented. Execution-only T/F records retain unknown aggressor rather than forcing a side; queue position and hidden quantity are unavailable.\n\nBounded summary-only probable-replenishment examples:\n\n```json\n" + json.dumps(examples, indent=2) + "\n```\n\nSuitability conclusion: suitable for descriptive MBO lifecycle and displayed-replenishment diagnostics if reconstruction completes without fail-closed errors; unsuitable for hidden-liquidity proof, queue-position claims, threshold selection, or profitability claims.\n", encoding="utf-8")
