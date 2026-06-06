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
  Sparkles,
  Target,
} from "lucide-react";

const heroKpis = [
  { label: "Processed comments", value: "15,000", sub: "clean Samsung feedback", icon: MessageSquare },
  { label: "YouTube videos", value: "10", sub: "source videos", icon: Database },
  { label: "Manual Precision@5", value: "93%", sub: "RAG evaluation", icon: ShieldCheck },
  { label: "RAG systems", value: "2", sub: "feedback + strategy", icon: GitBranch },
];

const signalRows = [
  "Battery / Charging -> high-priority complaint",
  "S-Pen / Features -> Ultra identity risk",
  "Camera Quality -> creator satisfaction signal",
  "Display Durability -> trust and reliability concern",
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
    <section className="relative overflow-hidden rounded-xl border border-slate-800 bg-[#07111F] text-white shadow-2xl">
      <div className="absolute inset-0 opacity-80 [background:radial-gradient(circle_at_18%_18%,rgba(20,40,160,0.42),transparent_32%),radial-gradient(circle_at_82%_18%,rgba(15,118,110,0.34),transparent_30%),linear-gradient(135deg,#07111F_0%,#0D1728_48%,#111827_100%)]" />
      <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px)] [background-size:42px_42px]" />

      <div className="relative grid min-h-[680px] gap-8 px-5 py-8 md:px-8 lg:grid-cols-[1.05fr_0.95fr] lg:px-10 lg:py-12">
        <div className="flex flex-col justify-center">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
            className="inline-flex w-fit items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-xs font-semibold uppercase text-blue-100 backdrop-blur"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Samsung Feedback Intelligence
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.08 }}
            className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] md:text-7xl"
          >
            Galaxy Insight RAG
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="mt-5 max-w-2xl text-xl leading-8 text-slate-200 md:text-2xl"
          >
            AI-powered Samsung feedback and product strategy intelligence.
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.24 }}
            className="mt-5 max-w-2xl text-sm leading-7 text-slate-300 md:text-base"
          >
            The system transforms YouTube comments into sentiment, issue categories, topics,
            grounded RAG answers, strategy recommendations, and an interactive roadmap refinement loop.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.32 }}
            className="mt-8 flex flex-wrap gap-3"
          >
            <Link
              href="/advisor"
              className="inline-flex items-center gap-2 rounded-lg bg-white px-5 py-3 text-sm font-semibold text-[#1428A0] shadow-lg shadow-blue-950/30 transition hover:bg-blue-50"
            >
              Open Strategy Chat
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/refinement"
              className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/10 px-5 py-3 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15"
            >
              Negotiate Roadmap
            </Link>
          </motion.div>
        </div>

        <div className="flex flex-col justify-center gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="rounded-xl border border-white/15 bg-white/10 p-4 backdrop-blur-md"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-white">Live feedback signals</div>
                <div className="mt-1 text-xs text-slate-300">pipeline output preview</div>
              </div>
              <span className="rounded-md bg-emerald-400/15 px-2 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-300/20">
                Ready
              </span>
            </div>

            <div className="mt-4 space-y-2">
              {signalRows.map((signal, index) => (
                <motion.div
                  key={signal}
                  initial={{ opacity: 0, x: 18 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.42, delay: 0.32 + index * 0.08 }}
                  className="flex items-center gap-3 rounded-lg border border-white/10 bg-slate-950/35 px-3 py-2 text-sm text-slate-200"
                >
                  <span className="h-2 w-2 rounded-full bg-emerald-300" />
                  {signal}
                </motion.div>
              ))}
            </div>
          </motion.div>

          <div className="grid gap-3 sm:grid-cols-2">
            {heroKpis.map((kpi, index) => {
              const Icon = kpi.icon;
              return (
                <motion.div
                  key={kpi.label}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.48, delay: 0.42 + index * 0.08 }}
                  className="rounded-xl border border-white/15 bg-white/10 p-4 backdrop-blur-md"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs font-medium uppercase text-slate-300">{kpi.label}</span>
                    <Icon className="h-4 w-4 text-blue-200" />
                  </div>
                  <div className="mt-3 text-3xl font-semibold text-white">{kpi.value}</div>
                  <div className="mt-1 text-xs text-slate-300">{kpi.sub}</div>
                </motion.div>
              );
            })}
          </div>
        </div>
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
