// smithers-source: project-local
// smithers-display-name: Trading research Phase B.5 technical integrity verification
/** @jsxImportSource smthrs */
import { createSmithers, Sequence, Task } from "smthrs";
import { manifest, verificationInput, verificationResult } from "../schemas/trading-research/phase-b5";

const { Workflow, outputs, smithers } = createSmithers({ input: verificationInput, manifest, result: verificationResult });

async function bridge(command: string, payload: Record<string, unknown>, root: string): Promise<unknown> {
  const registryPath = payload.registry_path ?? process.env.RESEARCH_PIPELINE_REGISTRY ?? `${root}/research_registry/research_pipeline.sqlite3`;
  const child = Bun.spawn(["python", "-m", "research_pipeline", "workflow", command, "--input-json", JSON.stringify({ ...payload, registry_path: registryPath })], { cwd: root, env: { ...process.env, PYTHONPATH: `${root}/src`, RESEARCH_PIPELINE_REGISTRY: String(registryPath) }, stdout: "pipe", stderr: "pipe" });
  const stdout = await new Response(child.stdout).text(); const stderr = await new Response(child.stderr).text(); const code = await child.exited;
  if (code !== 0) throw new Error(`Phase B.5 bridge ${command} failed (${code}): ${stderr || stdout}`);
  return JSON.parse(stdout);
}

export default smithers((ctx) => {
  const input = ctx.input;
  const created = ctx.outputMaybe(outputs.manifest, { nodeId: "create-verification-manifest" });
  return <Workflow name="trading-research-phase-b5"><Sequence>
    <Task id="create-verification-manifest" output={outputs.manifest} retries={0}>
      {async () => await bridge("verification-create-manifest", { strategy_id: input.strategy_id, manifest_path: input.manifest_path, diagnostic_dir: input.manifest_path.replace(/[\\/][^\\/]+$/, ""), registry_path: input.registry_path }, input.repository_root)}
    </Task>
    {created ? <Task id="run-verification" output={outputs.result} dependsOn={["create-verification-manifest"]} retries={0}>
      {async () => await bridge("verification-run", { strategy_id: input.strategy_id, manifest_path: created.manifest_path, registry_path: input.registry_path }, input.repository_root)}
    </Task> : null}
  </Sequence></Workflow>;
});
