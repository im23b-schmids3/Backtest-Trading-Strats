// smithers-source: project-local
// smithers-display-name: Imbalance VWAP Ride BTC exploratory study
// smithers-description: Durable no-approval implementation, verification, and sealed exploratory execution.
/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const result = z.object({ status: z.string(), summary: z.string(), finalReportPath: z.string().nullable(), testsPassed: z.boolean(), studyExecuted: z.boolean() });
const { Workflow, outputs, smithers } = createSmithers({ result });

export default smithers((ctx) => (
  <Workflow name="imbalance-vwap-ride-btc-exploratory">
    <Task id="implement-verify-and-execute" output={outputs.result} agent={agents.smart} retries={1} timeoutMs={90 * 60_000} heartbeatTimeoutMs={15 * 60_000}>
      {`You are the sole implementation owner. Read and obey C:/Users/sandr/Trading-Bot-Fib/.smithers/specs/imbalance-vwap-ride-btc-exploratory.md completely. Work directly in C:/Users/sandr/Trading-Bot-Fib; the immutable source is C:/Users/sandr/Trading-Bot-Fib/data/value_area_trap/normalized/BTCUSDT/c2028fdd21bb69943820d532a592f13cd43f4ab18cc7b170b1e2b091a00202fc/manifest.json. Implement all strategy, CLI, tests, reports, diagnostics, validation, locked-test, and conditional Alpha proxy requirements. Do not modify forbidden paths, download, renormalize, use live orders, or expose secrets. Process raw Parquet only with local tools; never transmit raw trades. Work continuously: implement all phases, run focused and full research-pipeline tests, compileall and diff check, repair failures, then execute the real sealed study. Preserve immutable collision behavior. Do not return a plan or stop at scaffolding. Finally return ONLY JSON matching {status,summary,finalReportPath,testsPassed,studyExecuted}; use COMPLETED, DEVELOPMENT_EDGE_NOT_FOUND, or FAILED truthfully.`}
    </Task>
  </Workflow>
));
