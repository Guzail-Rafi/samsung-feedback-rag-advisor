import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const runtime = "nodejs";

type Phase = {
  id: string;
  title: string;
  items: string[];
};

type RefinementBridgeResponse = {
  error?: string;
  roadmap?: Phase[];
  decision?: {
    request: string;
    verdict: "Accepted" | "Rejected" | "Alternative suggested";
    rationale: string;
    update: string;
  };
  model?: string;
  llmProvider?: string;
  llmFallbackUsed?: boolean;
  evidenceCount?: number;
  evidenceAvailable?: boolean;
};

function sanitizeError(message: string) {
  return message.replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");
}

function runPythonRefinement(payload: { request: string; roadmap: Phase[] }) {
  return new Promise<RefinementBridgeResponse>((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "src", "roadmap_refinement_bridge.py");
    const child = spawn("python", [scriptPath], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Roadmap refinement timed out while the model was loading."));
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
      const jsonLine = stdout.trim().split(/\r?\n/).filter(Boolean).at(-1);

      if (!jsonLine) {
        reject(new Error(sanitizeError(stderr || `Refinement bridge exited with code ${code}.`)));
        return;
      }

      try {
        const parsed = JSON.parse(jsonLine) as RefinementBridgeResponse;
        if (code !== 0 || parsed.error) {
          reject(new Error(sanitizeError(parsed.error || stderr || "Roadmap refinement failed.")));
          return;
        }
        resolve(parsed);
      } catch {
        reject(new Error(sanitizeError(`Could not parse refinement output. ${stderr || stdout}`)));
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      request?: string;
      roadmap?: Phase[];
    };
    const refinementRequest = body.request?.trim();

    if (!refinementRequest) {
      return NextResponse.json({ error: "A refinement request is required." }, { status: 400 });
    }
    if (!Array.isArray(body.roadmap) || body.roadmap.length === 0) {
      return NextResponse.json({ error: "The current roadmap is required." }, { status: 400 });
    }

    const result = await runPythonRefinement({
      request: refinementRequest,
      roadmap: body.roadmap,
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown roadmap refinement error.";
    return NextResponse.json({ error: sanitizeError(message) }, { status: 500 });
  }
}
