"use client";

import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  FileQuestion,
  Route,
  Target,
} from "lucide-react";
import AetherFlowHero from "@/components/ui/aether-flow-hero";

const architectureSteps = [
  {
    title: "Feedback RAG",
    subtitle: "User opinion Q&A",
    icon: FileQuestion,
    color: "#1428A0",
    inputs: ["Clean comments", "Sentiment labels", "Issue categories", "Topic names"],
    outputs: ["Question answers", "Retrieved comments", "Evidence scores", "Precision@5"],
    href: "/rag",
  },
  {
    title: "Strategy RAG",
    subtitle: "Product roadmap advisor",
    icon: Route,
    color: "#0F766E",
    inputs: ["Strategy evidence", "Customer signals", "Business impact", "Priority labels"],
    outputs: ["S27 roadmap", "Feature priorities", "Trade-offs", "Refinement loop"],
    href: "/strategy",
  },
];

export function LandingHero() {
  return <AetherFlowHero />;
}

export function TwoRagArchitecture() {
  return (
    <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-xs font-semibold uppercase text-[#1428A0]">Two-RAG Architecture</div>
          <h2 className="mt-2 text-2xl font-semibold text-slate-950">Two retrieval systems, two decision layers</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Feedback RAG answers what customers are saying. Strategy RAG turns those grounded signals into product roadmap decisions.
          </p>
        </div>
        <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600 ring-1 ring-slate-200">
          comments &gt; evidence &gt; answers &gt; roadmap
        </div>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1fr_170px_1fr]">
        {[architectureSteps[0]].map((step) => {
          const Icon = step.icon;
          return (
            <div key={step.title} className="rounded-xl border border-slate-200 p-5">
              <div className="flex items-center gap-3">
                <span className="flex h-11 w-11 items-center justify-center rounded-lg text-white" style={{ backgroundColor: step.color }}>
                  <Icon className="h-5 w-5" />
                </span>
                <div>
                  <h3 className="text-lg font-semibold text-slate-950">{step.title}</h3>
                  <p className="text-sm text-slate-500">{step.subtitle}</p>
                </div>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-500">Inputs</div>
                  <div className="mt-2 space-y-2">
                    {step.inputs.map((item) => (
                      <div key={item} className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase text-slate-500">Outputs</div>
                  <div className="mt-2 space-y-2">
                    {step.outputs.map((item) => (
                      <div key={item} className="rounded-md bg-[#F4F7FF] px-3 py-2 text-sm text-slate-700 ring-1 ring-[#DDE7FF]">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <Link
                href={step.href}
                className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-[#1428A0] transition hover:text-[#0F1F78]"
              >
                Open {step.title}
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          );
        })}

        <div className="flex items-center justify-center">
          <div className="flex w-full items-center gap-3 xl:flex-col">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-slate-950 text-white">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div className="h-px flex-1 bg-slate-200 xl:h-16 xl:w-px xl:flex-none" />
            <div className="rounded-lg bg-slate-50 px-3 py-2 text-center text-xs font-semibold uppercase text-slate-600 ring-1 ring-slate-200">
              Shared evidence layer
            </div>
            <div className="h-px flex-1 bg-slate-200 xl:h-16 xl:w-px xl:flex-none" />
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-[#0F766E] text-white">
              <Target className="h-5 w-5" />
            </div>
          </div>
        </div>

        {[architectureSteps[1]].map((step) => {
          const Icon = step.icon;
          return (
            <div key={step.title} className="rounded-xl border border-slate-200 p-5">
              <div className="flex items-center gap-3">
                <span className="flex h-11 w-11 items-center justify-center rounded-lg text-white" style={{ backgroundColor: step.color }}>
                  <Icon className="h-5 w-5" />
                </span>
                <div>
                  <h3 className="text-lg font-semibold text-slate-950">{step.title}</h3>
                  <p className="text-sm text-slate-500">{step.subtitle}</p>
                </div>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <div>
                  <div className="text-xs font-semibold uppercase text-slate-500">Inputs</div>
                  <div className="mt-2 space-y-2">
                    {step.inputs.map((item) => (
                      <div key={item} className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase text-slate-500">Outputs</div>
                  <div className="mt-2 space-y-2">
                    {step.outputs.map((item) => (
                      <div key={item} className="rounded-md bg-[#F4F7FF] px-3 py-2 text-sm text-slate-700 ring-1 ring-[#DDE7FF]">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <Link
                href={step.href}
                className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-[#1428A0] transition hover:text-[#0F1F78]"
              >
                Open {step.title}
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          );
        })}
      </div>
    </section>
  );
}
