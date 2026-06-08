import { spawn } from "child_process";
import { NextResponse } from "next/server";
import path from "path";

export const runtime = "nodejs";

function runStatusBridge() {
  return new Promise<Record<string, unknown>>((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "src", "mlflow_status_bridge.py");
    const child = spawn("python", [scriptPath], {
      cwd: process.cwd(),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("MLflow status request timed out."));
    }, 60000);

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
        reject(new Error(stderr || `MLflow status bridge exited with code ${code}.`));
        return;
      }
      try {
        const parsed = JSON.parse(jsonLine) as Record<string, unknown>;
        if (code !== 0 || parsed.error) {
          reject(new Error(String(parsed.error || stderr || "MLflow status request failed.")));
          return;
        }
        resolve(parsed);
      } catch {
        reject(new Error(stderr || "Could not parse MLflow status response."));
      }
    });
  });
}

export async function GET() {
  try {
    return NextResponse.json(await runStatusBridge());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown MLflow error." },
      { status: 500 },
    );
  }
}
