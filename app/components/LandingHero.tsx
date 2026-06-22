"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Bot,
  BrainCircuit,
  Database,
  FileQuestion,
  GitBranch,
  MessageSquare,
  Route,
  ShieldCheck,
  Target,
} from "lucide-react";

const heroKpis = [
  { label: "Processed comments", value: "15,000", sub: "clean Samsung feedback", icon: MessageSquare },
  { label: "YouTube videos", value: "10", sub: "source videos", icon: Database },
  { label: "Manual Precision@5", value: "93%", sub: "RAG evaluation", icon: ShieldCheck },
  { label: "RAG systems", value: "2", sub: "feedback + strategy", icon: GitBranch },
];

const signalRows = [
  { label: "Battery / Charging", signal: "high-priority complaint", color: "#2488D8" },
  { label: "S-Pen / Features", signal: "Ultra identity risk", color: "#68A52E" },
  { label: "Camera Quality", signal: "creator satisfaction signal", color: "#C57B15" },
  { label: "Display Durability", signal: "trust and reliability concern", color: "#D85B32" },
];

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
  return (
    <section>
      <div className="grid items-center gap-8 py-5 lg:grid-cols-[1.08fr_0.92fr] lg:py-8">
        <div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="inline-flex w-fit items-center gap-2 rounded-full border border-[#C9D7FA] bg-[#EAF1FF] px-3 py-1.5 text-xs font-semibold text-[#245C9E]"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-[#2E87DC]" />
            Samsung Feedback Intelligence
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.06 }}
            className="mt-5 max-w-3xl text-4xl font-semibold leading-[1.04] tracking-[-0.04em] text-[#181818] sm:text-5xl lg:text-6xl"
          >
            Galaxy Insight RAG
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.12 }}
            className="mt-5 max-w-2xl text-lg font-medium leading-8 text-[#4F4F49]"
          >
            AI-powered feedback analysis and product strategy intelligence. Turn YouTube comments into grounded answers and actionable roadmap signals.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.18 }}
            className="mt-7 flex flex-wrap gap-3"
          >
            <Link
              href="/advisor"
              className="inline-flex items-center gap-2 rounded-lg bg-[#181818] px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#30302D]"
            >
              <Bot className="h-4 w-4" />
              Samsung Chat
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/refinement"
              className="inline-flex items-center gap-2 rounded-lg border border-[#CFCFC8] bg-white px-5 py-3 text-sm font-semibold text-[#343430] shadow-sm transition hover:border-[#A8A79F] hover:bg-[#FAFAF7]"
            >
              <Route className="h-4 w-4 text-[#1428A0]" />
              Roadmap Refinement
            </Link>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.48, delay: 0.12 }}
          className="rounded-xl border border-[#D5D4CD] bg-white p-5 shadow-[0_18px_50px_rgba(35,35,30,0.08)] md:p-6"
        >
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-semibold text-[#343430]">Live feedback signals</div>
            </div>
            <span className="rounded-full bg-[#ECF7E4] px-2.5 py-1 text-[11px] font-semibold text-[#568A2D]">
              Ready
            </span>
          </div>

          <div className="mt-4 divide-y divide-[#E1E0DA]">
            {signalRows.map((row, index) => (
              <motion.div
                key={row.label}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.35, delay: 0.22 + index * 0.06 }}
                className="grid grid-cols-[minmax(0,1fr)_16px_minmax(0,1fr)] items-center gap-3 py-3 text-sm"
              >
                <div className="flex min-w-0 items-center gap-3 font-medium text-[#343430]">
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: row.color }} />
                  <span>{row.label}</span>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#92928B]" />
                <span className="text-[#5F5F59]">{row.signal}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      <div className="grid gap-3 border-y border-[#DDDCD5] py-5 sm:grid-cols-2 xl:grid-cols-4">
        {heroKpis.map((kpi, index) => {
          const Icon = kpi.icon;
          return (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.38, delay: 0.24 + index * 0.05 }}
              className="rounded-xl border border-[#DDDCD5] bg-white p-5 shadow-sm"
            >
              <div className="flex items-center gap-2 text-sm text-[#66665F]">
                <Icon className="h-4 w-4 text-[#1428A0]" />
                {kpi.label}
              </div>
              <div className={`mt-2 text-3xl font-medium tracking-tight ${kpi.value === "93%" ? "text-[#5E942F]" : "text-[#242421]"}`}>
                {kpi.value}
              </div>
              <div className="mt-1 text-xs text-[#777770]">{kpi.sub}</div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
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
