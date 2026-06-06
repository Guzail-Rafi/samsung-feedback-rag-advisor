"use client";

import { FormEvent, useState } from "react";
import { CheckCircle2, GitPullRequestArrow, Lightbulb, RotateCcw, Send, XCircle } from "lucide-react";
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

function addUnique(items: string[], item: string) {
  return items.includes(item) ? items : [...items, item];
}

function evaluateRequest(request: string, roadmap: Phase[]) {
  const text = request.toLowerCase();
  let updated = roadmap.map((phase) => ({ ...phase, items: [...phase.items] }));

  if (text.includes("remove") && (text.includes("s-pen") || text.includes("spen") || text.includes("pen"))) {
    return {
      roadmap: updated,
      decision: {
        request,
        verdict: "Rejected" as const,
        rationale: "The S-Pen is strongly tied to Ultra identity, so removing it conflicts with the strategy evidence.",
        update: "No roadmap change applied.",
      },
    };
  }

  if (text.includes("camera") || text.includes("creator")) {
    updated = updated.map((phase) =>
      phase.id === "phase-1"
        ? { ...phase, items: addUnique(phase.items, "Creator-grade camera improvements") }
        : phase,
    );

    return {
      roadmap: updated,
      decision: {
        request,
        verdict: "Accepted" as const,
        rationale: "Camera quality is a high-value satisfaction signal and appears in strategy refinement evidence.",
        update: "Creator-grade camera improvements added to Phase 1.",
      },
    };
  }

  if (text.includes("display") || text.includes("green") || text.includes("durability")) {
    updated = updated.map((phase) =>
      phase.id === "phase-1"
        ? { ...phase, items: addUnique(phase.items, "Display durability and green-line prevention") }
        : phase,
    );

    return {
      roadmap: updated,
      decision: {
        request,
        verdict: "Accepted" as const,
        rationale: "Display trust is a complaint and risk signal, so it belongs in the first reliability phase.",
        update: "Display durability moved into Phase 1.",
      },
    };
  }

  if (text.includes("profit") || text.includes("premium ai") || text.includes("ai")) {
    updated = updated.map((phase) =>
      phase.id === "phase-2"
        ? { ...phase, items: addUnique(phase.items, "Premium AI workflows after hardware trust") }
        : phase,
    );

    return {
      roadmap: updated,
      decision: {
        request,
        verdict: "Alternative suggested" as const,
        rationale: "AI can support profit, but the evidence says visible hardware value should come before AI-led monetization.",
        update: "Premium AI workflows added to Phase 2 instead of Phase 1.",
      },
    };
  }

  return {
    roadmap: updated,
    decision: {
      request,
      verdict: "Alternative suggested" as const,
      rationale: "The request needs clearer evidence. The system keeps the current roadmap and recommends checking RAG evidence first.",
      update: "No roadmap change applied.",
    },
  };
}

function VerdictIcon({ verdict }: { verdict: Decision["verdict"] }) {
  if (verdict === "Accepted") return <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
  if (verdict === "Rejected") return <XCircle className="h-4 w-4 text-red-600" />;
  return <Lightbulb className="h-4 w-4 text-amber-600" />;
}

export function RoadmapRefinement() {
  const [roadmap, setRoadmap] = useState(baseRoadmap);
  const [request, setRequest] = useState(promptChips[0]);
  const [decisions, setDecisions] = useState<Decision[]>([
    {
      request: refinementDecisions[0].feedback,
      verdict: "Accepted",
      rationale: refinementDecisions[0].reason,
      update: "Camera moves into Phase 1 beside battery and S-Pen restoration.",
    },
  ]);

  function submitRequest(event: FormEvent) {
    event.preventDefault();
    const trimmed = request.trim();
    if (!trimmed) return;

    const result = evaluateRequest(trimmed, roadmap);
    setRoadmap(result.roadmap);
    setDecisions((current) => [result.decision, ...current].slice(0, 5));
    setRequest("");
  }

  function resetRoadmap() {
    setRoadmap(baseRoadmap);
    setDecisions([]);
    setRequest(promptChips[0]);
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
            className="mt-3 w-full resize-none rounded-lg border border-slate-200 px-3 py-3 text-sm outline-none transition focus:border-[#1428A0] focus:ring-4 focus:ring-[#1428A0]/10"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {promptChips.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => setRequest(prompt)}
                className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600 transition hover:bg-[#EAF0FF] hover:text-[#1428A0]"
              >
                {prompt}
              </button>
            ))}
          </div>
          <button
            type="submit"
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-[#1428A0] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#0F1F78]"
          >
            <Send className="h-4 w-4" />
            Submit negotiation
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
