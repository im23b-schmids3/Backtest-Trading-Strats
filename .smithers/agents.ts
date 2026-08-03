// smithers-source: generated
import { type AgentLike } from "smithers-orchestrator";
import { CodexAgent as SmithersCodexAgent } from "smithers-orchestrator";
// import { ClaudeCodeAgent as SmithersClaudeCodeAgent } from "smithers-orchestrator";
// import { OpenAIAgent as SmithersOpenAIAgent } from "smithers-orchestrator";
// import { OpenCodeAgent as SmithersOpenCodeAgent } from "smithers-orchestrator";
// import { AntigravityAgent as SmithersAntigravityAgent } from "smithers-orchestrator";
// import { PiAgent as SmithersPiAgent } from "smithers-orchestrator";
// import { KimiAgent as SmithersKimiAgent } from "smithers-orchestrator";
// import { AmpAgent as SmithersAmpAgent } from "smithers-orchestrator";
// import { VibeAgent as SmithersVibeAgent } from "smithers-orchestrator";
// import { HermesCliAgent as SmithersHermesCliAgent } from "smithers-orchestrator";
// import { OpenClawAgent as SmithersOpenClawAgent } from "smithers-orchestrator";
// import { PoolAgent as SmithersPoolAgent } from "smithers-orchestrator";

// export { ClaudeCodeAgent } from "./agents/claude-code";
export { CodexAgent } from "./agents/codex";
// export { OpenCodeAgent } from "./agents/opencode";
// export { AntigravityAgent } from "./agents/antigravity";
// export { PoolAgent } from "./agents/pool";

// class SmithersOpenRouterAgent extends SmithersOpenAIAgent {
//   generate(args = {}) {
//     if (!process.env.OPENROUTER_API_KEY) {
//       throw new Error("Smithers generated an OpenRouter default agent, but OPENROUTER_API_KEY is not set. Set OPENROUTER_API_KEY, or run `smithers agent add` to configure another agent, then rerun this workflow.");
//     }
//     return super.generate(args);
//   }
// }
//
// function createOpenRouterAgent() {
//   return new SmithersOpenRouterAgent({
//     model: "openai/gpt-5.4-mini",
//     baseURL: "https://openrouter.ai/api/v1",
//     apiKey: process.env.OPENROUTER_API_KEY,
//   });
// }

const codexWorkerPath = [
  String.raw`C:\Users\sandr\.vscode\extensions\openai.chatgpt-26.727.40816-win32-x64\bin\windows-x86_64`,
  // Smithers 0.28 preflight invokes `which codex` even on Windows. Git for
  // Windows supplies the POSIX-compatible `which.exe`; the actual Codex binary
  // remains the first path entry above.
  String.raw`C:\Program Files\Git\usr\bin`,
  String.raw`C:\Users\sandr\AppData\Roaming\npm`,
  process.env.PATH ?? "",
].join(";");

const codexWorkerEnv = {
  PATH: codexWorkerPath,
  Path: codexWorkerPath,
};

export const providers = {
  codex: new SmithersCodexAgent({
    model: "gpt-5.6-luna",
    config: {
      model_reasoning_effort: "medium",
    },
    skipGitRepoCheck: true,
    env: codexWorkerEnv,
  }),

  codexSol: new SmithersCodexAgent({
    model: "gpt-5.6-sol",
    config: {
      model_reasoning_effort: "xhigh",
    },
    skipGitRepoCheck: true,
    env: codexWorkerEnv,
  }),

  codexTerra: new SmithersCodexAgent({
    model: "gpt-5.6-terra",
    config: {
      model_reasoning_effort: "medium",
    },
    skipGitRepoCheck: true,
    env: codexWorkerEnv,
  }),

  codexLuna: new SmithersCodexAgent({
    model: "gpt-5.6-luna",
    config: {
      model_reasoning_effort: "medium",
    },
    skipGitRepoCheck: true,
    env: codexWorkerEnv,
  }),
} as const;

export const agents = {
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  cheapFast: [
    providers.codexLuna,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.vibe,
    // providers.antigravity,
    // providers.openclaw,
    // providers.pi,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  research: [
    providers.codexLuna,
    // providers.kimi,
    // providers.antigravity,
    // providers.opencode,
    // providers.claudeSonnet,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  implement: [
    providers.codexLuna,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.antigravity,
    // providers.claude,
    // providers.opencode,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  midTier: [
    providers.codexTerra,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.antigravity,
    // providers.opencode,
    // providers.claude,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  smartTool: [
    providers.codexTerra,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.antigravity,
    // providers.opencode,
    // providers.claude,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  validate: [
    providers.codexTerra,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.antigravity,
    // providers.opencode,
    // providers.claude,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  smart: [
    providers.codexSol,
    // providers.claude,
    // providers.claudeOpus,
    // providers.opencode,
    // providers.openclaw,
    // providers.openrouter,
    // providers.antigravity,
    // providers.amp,
    // providers.kimi,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  review: [
    providers.codexSol,
    // providers.claude,
    // providers.claudeOpus,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.amp,
    // providers.opencode,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  planning: [
    providers.codexSol,
    // providers.claude,
    // providers.claudeOpus,
    // providers.claudeSonnet,
    // providers.kimi,
    // providers.opencode,
    // providers.openclaw,
    // providers.openrouter,
  ],
  // Codex runs first. Later entries are runtime fallbacks and are invoked only if every Codex attempt fails.
  orchestrator: [
    providers.codexSol,
    // providers.claude,
    // providers.claudeOpus,
    // providers.kimi,
    // providers.opencode,
    // providers.openclaw,
    // providers.openrouter,
  ],
} as const satisfies Record<string, AgentLike[]>;
