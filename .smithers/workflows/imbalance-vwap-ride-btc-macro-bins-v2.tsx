/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";

const result = z.object({ status: z.string(), summary: z.string(), finalReportPath: z.string().nullable(), testsPassed: z.boolean(), studyExecuted: z.boolean() });
const { Workflow, outputs, smithers } = createSmithers({ result });
export default smithers(() => (
  <Workflow name="imbalance-vwap-ride-btc-macro-bins-v2">
    <Task id="implement-verify-and-execute-v2" output={outputs.result} agent={agents.smart} retries={1} timeoutMs={90 * 60_000} heartbeatTimeoutMs={15 * 60_000}>
      {`Read and obey C:/Users/sandr/Trading-Bot-Fib/.smithers/specs/imbalance-vwap-ride-btc-macro-bins-v2.md completely. Implement and execute the separate V2 study locally. Preserve V1 and all forbidden paths. Never download, renormalize, use live orders/secrets, or transmit raw aggregate trades; only local bounded processing. Run focused and full tests, compileall and diff check before real V2 development execution. Return only validated JSON with truthful COMPLETED, DEVELOPMENT_EDGE_NOT_FOUND, or FAILED status.`}
    </Task>
  </Workflow>
));
