import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const runtime = "nodejs";

type UiMessage = {
  role: "assistant" | "user";
  content: string;
};

type AdvisorBridgeResponse = {
  error?: string;
  type?: string;
  answer?: string;
  selectedAgent?: string;
  model?: string;
  sources?: string[];
  evidence?: unknown[];
  retrieval?: unknown;
  mode?: string;
  confidence?: string;
  strategyGoal?: string;
  productLifecycle?: unknown;
  contextualQuery?: string;
  memoryUsed?: boolean;
  routingReason?: string;
  matchedTerms?: string[];
  toolTrace?: string[];
  availableTools?: string[];
  routingConfidence?: number;
  routingMethod?: string;
  routerModel?: string;
  normalizedQuery?: string;
  rewrittenQuery?: string;
  needsExternalResearch?: boolean;
  externalResearchFocus?: string[];
  externalEvidence?: unknown[];
  webResearch?: unknown;
  internalStrategyAnswer?: string;
  llmProvider?: string;
  llmFallbackUsed?: boolean;
  llmFallbackReason?: string;
  routerProvider?: string;
  routerFallbackUsed?: boolean;
  routerFallbackReason?: string;
  mlflowTraceId?: string;
};

function sanitizeError(message: string) {
  return message.replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");
}

function runPythonAdvisor(payload: { message: string; messages: UiMessage[] }) {
  return new Promise<AdvisorBridgeResponse>((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "src", "web_rag_bridge.py");
    const child = spawn("python", [scriptPath], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Advisor request timed out. Try again after the model finishes loading."));
    }, 240000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (code) => {
      clearTimeout(timeout);

      const trimmed = stdout.trim();
      const jsonLine = trimmed.split(/\r?\n/).filter(Boolean).at(-1);

      if (!jsonLine) {
        reject(new Error(sanitizeError(stderr || `Python advisor bridge exited with code ${code}.`)));
        return;
      }

      try {
        const parsed = JSON.parse(jsonLine) as AdvisorBridgeResponse;

        if (code !== 0 || parsed.error) {
          reject(new Error(sanitizeError(parsed.error || stderr || `Python advisor bridge exited with code ${code}.`)));
          return;
        }

        resolve(parsed);
      } catch {
        reject(new Error(sanitizeError(`Could not parse advisor bridge output. ${stderr || trimmed}`)));
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      message?: string;
      messages?: UiMessage[];
    };

    const message = body.message?.trim();

    if (!message) {
      return NextResponse.json({ error: "Message is required." }, { status: 400 });
    }

    const result = await runPythonAdvisor({
      message,
      messages: (body.messages ?? []).slice(-8),
    });

    return NextResponse.json({
      answer: result.answer,
      selectedAgent: result.selectedAgent,
      model: result.model,
      sources: result.sources,
      evidence: result.evidence,
      retrieval: result.retrieval,
      mode: result.mode,
      confidence: result.confidence,
      strategyGoal: result.strategyGoal,
      productLifecycle: result.productLifecycle,
      contextualQuery: result.contextualQuery,
      memoryUsed: result.memoryUsed,
      routingReason: result.routingReason,
      matchedTerms: result.matchedTerms,
      toolTrace: result.toolTrace,
      availableTools: result.availableTools,
      routingConfidence: result.routingConfidence,
      routingMethod: result.routingMethod,
      routerModel: result.routerModel,
      normalizedQuery: result.normalizedQuery,
      rewrittenQuery: result.rewrittenQuery,
      needsExternalResearch: result.needsExternalResearch,
      externalResearchFocus: result.externalResearchFocus,
      externalEvidence: result.externalEvidence,
      webResearch: result.webResearch,
      internalStrategyAnswer: result.internalStrategyAnswer,
      llmProvider: result.llmProvider,
      llmFallbackUsed: result.llmFallbackUsed,
      llmFallbackReason: result.llmFallbackReason,
      routerProvider: result.routerProvider,
      routerFallbackUsed: result.routerFallbackUsed,
      routerFallbackReason: result.routerFallbackReason,
      mlflowTraceId: result.mlflowTraceId,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown advisor error.";
    return NextResponse.json({ error: sanitizeError(message) }, { status: 500 });
  }
}
