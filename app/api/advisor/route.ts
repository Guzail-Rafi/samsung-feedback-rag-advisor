import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const runtime = "nodejs";

type UiMessage = {
  role: "assistant" | "user";
  content: string;
};

type RagBridgeResponse = {
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
  contextualQuery?: string;
  memoryUsed?: boolean;
};

function sanitizeError(message: string) {
  return message.replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");
}

function runPythonRag(payload: { message: string; messages: UiMessage[] }) {
  return new Promise<RagBridgeResponse>((resolve, reject) => {
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
      reject(new Error("RAG retrieval timed out. Try again after the model finishes loading."));
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
        reject(new Error(sanitizeError(stderr || `Python RAG bridge exited with code ${code}.`)));
        return;
      }

      try {
        const parsed = JSON.parse(jsonLine) as RagBridgeResponse;

        if (code !== 0 || parsed.error) {
          reject(new Error(sanitizeError(parsed.error || stderr || `Python RAG bridge exited with code ${code}.`)));
          return;
        }

        resolve(parsed);
      } catch {
        reject(new Error(sanitizeError(`Could not parse RAG bridge output. ${stderr || trimmed}`)));
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

    const result = await runPythonRag({
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
      contextualQuery: result.contextualQuery,
      memoryUsed: result.memoryUsed,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RAG advisor error.";
    return NextResponse.json({ error: sanitizeError(message) }, { status: 500 });
  }
}
