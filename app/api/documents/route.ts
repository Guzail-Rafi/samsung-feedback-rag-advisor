import { spawn } from "child_process";
import { readFile } from "fs/promises";
import { NextRequest, NextResponse } from "next/server";
import path from "path";

export const runtime = "nodejs";

type BridgeResponse = {
  error?: string;
  type?: string;
  [key: string]: unknown;
};

const MAX_FILE_BYTES = 10 * 1024 * 1024;

function sanitizeError(message: string) {
  return message.replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");
}

function runDocumentBridge(payload: Record<string, unknown>) {
  return new Promise<BridgeResponse>((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "src", "document_rag_bridge.py");
    const child = spawn("python", [scriptPath], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Document request timed out. The embedding or language model may still be loading."));
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
        reject(new Error(sanitizeError(stderr || `Document bridge exited with code ${code}.`)));
        return;
      }

      try {
        const parsed = JSON.parse(jsonLine) as BridgeResponse;
        if (code !== 0 || parsed.error) {
          reject(new Error(sanitizeError(String(parsed.error || stderr || "Document request failed."))));
          return;
        }
        resolve(parsed);
      } catch {
        reject(new Error(sanitizeError(`Could not parse document bridge output. ${stderr || stdout}`)));
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get("content-type") || "";
    let payload: Record<string, unknown>;

    if (contentType.includes("multipart/form-data")) {
      const form = await request.formData();
      const file = form.get("file");

      if (!(file instanceof File)) {
        return NextResponse.json({ error: "A document file is required." }, { status: 400 });
      }
      if (file.size > MAX_FILE_BYTES) {
        return NextResponse.json({ error: "Document is larger than the 10 MB upload limit." }, { status: 400 });
      }

      payload = {
        action: "ingest",
        filename: file.name,
        content_base64: Buffer.from(await file.arrayBuffer()).toString("base64"),
      };
    } else {
      payload = (await request.json()) as Record<string, unknown>;
    }

    if (payload.action === "list") {
      const manifestPath = path.join(process.cwd(), "data", "processed", "samsung_documents_manifest.json");
      try {
        const documents = JSON.parse(await readFile(manifestPath, "utf8")) as unknown[];
        return NextResponse.json({ action: "list", documents, collection: "samsung_documents" });
      } catch {
        return NextResponse.json({ action: "list", documents: [], collection: "samsung_documents" });
      }
    }

    return NextResponse.json(await runDocumentBridge(payload));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown document RAG error.";
    return NextResponse.json({ error: sanitizeError(message) }, { status: 500 });
  }
}
