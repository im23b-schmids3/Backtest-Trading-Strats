/** @jsxImportSource smithers-orchestrator */
import { createSmithers, Task } from "smithers-orchestrator";
import { z } from "zod/v4";
import { agents } from "../agents";
const result = z.object({ status: z.string(), summary: z.string(), finalReportPath: z.string().nullable(), testsPassed: z.boolean(), studyExecuted: z.boolean() });
const { Workflow, outputs, smithers } = createSmithers({ result });
export default smithers(() => <Workflow name="imbalance-vwap-ride-btc-long-only-v3"><Task id="acquire-implement-verify-and-execute-v3" output={outputs.result} agent={agents.smart} retries={1} timeoutMs={120 * 60_000} heartbeatTimeoutMs={15 * 60_000}>{`Read C:/Users/sandr/Trading-Bot-Fib/.smithers/specs/imbalance-vwap-ride-btc-long-only-v3.md completely and implement/execute it. Work locally; preserve V1/V2. You are authorized to download exactly the six official Binance BTCUSDT USD-M aggTrades months stated in the spec, and nothing else. Never transmit raw rows, use secrets/live orders, or download unrelated assets. Validate/normalize locally, test, compile, diff-check, then run V3. Return only structured JSON status, summary, final report path, tests passed and study executed.`}</Task></Workflow>);
