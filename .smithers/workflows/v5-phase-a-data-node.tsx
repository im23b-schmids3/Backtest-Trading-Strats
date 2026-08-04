/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task, Workflow } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const result = z.object({
  status: z.enum([
    "PHASE_A_DATA_COMPLETE",
    "PHASE_A_DATA_BLOCKED",
  ]),
  summary: z.string(),
  archiveCount: z.number().int().min(0).max(13),
  datasetHash: z.string().length(64).nullable(),
  footprintHash: z.string().length(64).nullable(),
  blocker: z.string().nullable(),
  model: z.literal("gpt-5.6-terra"),
});


const { outputs, smithers } = createSmithers({ result });
const root = "C:/Users/sandr/Trading-Bot-Fib";

export default smithers(() => (
  <Workflow name="v5-phase-a-data-node">
    <Task
      id="acquire-phase-a-data"
      output={outputs.result}
      agent={agents.midTier}
      retries={1}
      timeoutMs={12 * 60 * 60_000}
      heartbeatTimeoutMs={30 * 60_000}
    >
      {`You are a Codex Terra Medium V5 Phase A data-resume node.

Read ${root}/.smithers/specs/imbalance-vwap-ride-btc-long-only-v5.md completely.
Work only in ${root}.

Resume from verified monthly V5 Phase A checkpoints. Do not rebuild a month whose
published parquet, metadata, source identity, and SHA-256 are all valid.

Use only official BTCUSDT USD-M months 2023-01 through 2024-01.
Do not run candidates, Phase B, or Alpha.
Do not modify or invalidate V1-V4.
Do not transmit raw aggregate-trade rows, secrets, or credentials.

Finish every missing or invalid month, especially 2024-01. Atomically finalize the
13-month manifest and compute and validate a non-null 64-character footprint hash.

Return PHASE_A_DATA_COMPLETE only after all 13 months and hashes validate.
IN_PROGRESS_CHECKPOINTED is not an acceptable final response.

If completion is impossible, return PHASE_A_DATA_BLOCKED with one concrete blocker.
Do not claim completion from a partial manifest.
Return schema-valid JSON with model exactly gpt-5.6-terra.`}
    </Task>
  </Workflow>
));