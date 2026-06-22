"use client";

import { FormEvent, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  GitPullRequestArrow,
  Lightbulb,
  LoaderCircle,
  RotateCcw,
  Send,
  XCircle,
} from "lucide-react";
import { refinementDecisions } from "../lib/mock";

type Phase = {
  id: string;
  title: string;
  items: string[];
};

type Decision = {
  request: string;
  verdict: "Accepted" | "Rejected" | "Alternative suggested";
  rationale: string;
  update: string;
};

type RefinementResponse = {
  roadmap?: Phase[];
  decision?: Decision;
  error?: string;
};

const baseRoadmap: Phase[] = [
  {
    id: "phase-1",
    title: "Phase 1: Restore Ultra trust",
    items: ["Battery endurance", "S-Pen restoration", "Camera consistency"],
  },
  {
    id: "phase-2",
    title: "Phase 2: Make AI practical",
    items: ["Creator AI tools", "On-device search", "Support assistant"],
  },
  {
    id: "phase-3",
    title: "Phase 3: Defend premium value",
    items: ["Display durability", "Trade-in messaging", "Launch proof points"],
  },
];

const promptChips = [
  "Phase 1 should also include camera improvement because creators care about camera quality.",
  "Move premium AI features earlier for profit.",
  "Add display durability and green-line prevention to Phase 1.",
  "Remove S-Pen work to save cost.",
];

function VerdictIcon({ verdict }: { verdict: Decision["verdict"] }) {
  if (verdict === "Accepted") return <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
  if (verdict === "Rejected") return <XCircle className="h-4 w-4 text-red-600" />;
  return <Lightbulb className="h-4 w-4 text-amber-600" />;
}

export function RoadmapRefinement() {
  const [roadmap, setRoadmap] = useState(baseRoadmap);
  const [request, setRequest] = useState(promptChips[0]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [decisions, setDecisions] = useState<Decision[]>([
    {
      request: refinementDecisions[0].feedback,
      verdict: "Accepted",
      rationale: refinementDecisions[0].reason,
      update: "Camera moves into Phase 1 beside battery and S-Pen restoration.",
    },
  ]);

  async function submitRequest(event: FormEvent) {
    event.preventDefault();
    const trimmed = request.trim();
    if (!trimmed || isSubmitting) return;

    setIsSubmitting(true);
    setError("");

    try {
      const response = await fetch("/api/refinement", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request: trimmed, roadmap }),
      });
      const result = (await response.json()) as RefinementResponse;

      if (!response.ok || !result.roadmap || !result.decision) {
        throw new Error(result.error || "The roadmap refinement service returned an incomplete response.");
      }

      setRoadmap(result.roadmap);
      setDecisions((current) => [result.decision as Decision, ...current].slice(0, 5));
      setRequest("");
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : "The roadmap could not be refined. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  function resetRoadmap() {
    setRoadmap(baseRoadmap);
    setDecisions([]);
    setRequest(promptChips[0]);
    setError("");
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-950">Live Roadmap</h2>
            <p className="mt-1 text-sm text-slate-500">Negotiation requests update the roadmap cards immediately.</p>
          </div>
          <button
            type="button"
            onClick={resetRoadmap}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </button>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {roadmap.map((phase) => (
            <div key={phase.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
                <GitPullRequestArrow className="h-4 w-4 text-[#1428A0]" />
                {phase.title}
              </div>
              <ul className="mt-4 space-y-2">
                {phase.items.map((item) => (
                  <li key={item} className="rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <form onSubmit={submitRequest} className="mt-5 rounded-lg border border-slate-200 p-4">
          <label htmlFor="refinement-request" className="text-sm font-semibold text-slate-950">
            Negotiate the roadmap
          </label>
          <textarea
            id="refinement-request"
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            rows={4}
            disabled={isSubmitting}
            className="mt-3 w-full resize-none rounded-lg border border-slate-200 px-3 py-3 text-sm outline-none transition focus:border-[#1428A0] focus:ring-4 focus:ring-[#1428A0]/10"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {promptChips.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => setRequest(prompt)}
                disabled={isSubmitting}
                className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600 transition hover:bg-[#EAF0FF] hover:text-[#1428A0]"
              >
                {prompt}
              </button>
            ))}
          </div>
          {error && (
            <div className="mt-3 flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700 ring-1 ring-red-200">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
          <button
            type="submit"
            disabled={isSubmitting || !request.trim()}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-[#1428A0] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#0F1F78] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSubmitting ? (
              <LoaderCircle className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            {isSubmitting ? "Interpreting request..." : "Submit negotiation"}
          </button>
        </form>
      </section>

      <aside className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-base font-semibold text-slate-950">System Decisions</h2>
        <p className="mt-1 text-sm text-slate-500">Accept, reject, or suggest alternative based on strategy evidence.</p>

        <div className="mt-5 space-y-3">
          {decisions.length === 0 && (
            <div className="rounded-lg bg-slate-50 p-4 text-sm text-slate-500 ring-1 ring-slate-200">
              No negotiation decisions yet.
            </div>
          )}

          {decisions.map((decision, index) => (
            <div key={`${decision.request}-${index}`} className="rounded-lg border border-slate-200 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
                <VerdictIcon verdict={decision.verdict} />
                {decision.verdict}
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">{decision.request}</p>
              <p className="mt-3 text-xs leading-5 text-slate-500">{decision.rationale}</p>
              <div className="mt-3 rounded-md bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 ring-1 ring-slate-200">
                {decision.update}
              </div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
