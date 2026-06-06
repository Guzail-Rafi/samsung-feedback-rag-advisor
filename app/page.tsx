import { BarChart3, Bot, FileQuestion, MessageSquare, ShieldCheck, Target } from "lucide-react";
import { IssueBarChart, SentimentDonut } from "./components/Charts";
import { LandingHero, TwoRagArchitecture } from "./components/LandingHero";
import { MetricCard, PageFrame, PageHeader, Panel, StatusPill } from "./components/PageParts";
import {
  agentRoutes,
  issueData,
  kpis,
  monitoringMetrics,
  ragQuestions,
  sentimentData,
  strategyPriorities,
} from "./lib/mock";

export default function OverviewPage() {
  return (
    <PageFrame>
      <LandingHero />
      <TwoRagArchitecture />

      <PageHeader
        eyebrow="Overview"
        title="Dashboard Summary"
        description="A working dashboard for YouTube comment analysis, retrieval, strategy recommendations, agent routing, and monitoring outputs."
        className="mt-8"
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Comments analyzed" value="15,000" sub="cleaned demo sample" icon={MessageSquare} />
        <MetricCard label="Videos sampled" value={`${kpis.videos}`} sub="YouTube sources" icon={BarChart3} />
        <MetricCard label="Top issue" value="Battery" sub={kpis.topIssue} icon={Target} />
        <MetricCard label="RAG Precision@5" value="93%" sub="manual average" icon={ShieldCheck} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1fr_1fr]">
        <Panel title="Sentiment Distribution">
          <SentimentDonut data={sentimentData} />
          <div className="mt-2 grid grid-cols-3 gap-2 text-sm">
            {sentimentData.map((item) => (
              <div key={item.name} className="rounded-lg bg-slate-50 p-3">
                <div className="flex items-center gap-2 text-slate-500">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                  {item.name}
                </div>
                <div className="mt-1 text-xl font-semibold text-slate-950">{item.value}%</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Top Issue Categories">
          <IssueBarChart data={issueData} />
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel title="Strategy Roadmap">
          <div className="space-y-3">
            {strategyPriorities.map((priority) => (
              <div key={priority.phase} className="rounded-lg border border-slate-200 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-950">{priority.phase}</div>
                    <div className="mt-1 text-sm text-slate-600">{priority.title}</div>
                  </div>
                  <StatusPill value="High" />
                </div>
                <div className="mt-3 text-sm font-medium text-slate-800">{priority.focus}</div>
                <p className="mt-1 text-sm leading-6 text-slate-500">{priority.rationale}</p>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Agent Activity">
          <div className="space-y-3">
            {agentRoutes.map((route) => (
              <div key={route.query} className="rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200">
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-[#EAF0FF] text-[#1428A0]">
                    <Bot className="h-4 w-4" />
                  </span>
                  <div>
                    <div className="text-sm font-medium text-slate-950">{route.query}</div>
                    <div className="mt-1 text-xs text-slate-500">{route.agent}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-2">
        <Panel title="RAG Questions">
          <div className="space-y-3">
            {ragQuestions.slice(0, 3).map((item) => (
              <div key={item.query} className="rounded-lg border border-slate-200 p-4">
                <div className="flex items-start gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <FileQuestion className="h-4 w-4" />
                  </span>
                  <div>
                    <div className="text-sm font-semibold text-slate-950">{item.query}</div>
                    <p className="mt-1 text-sm leading-6 text-slate-600">{item.answer}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Monitoring Summary">
          <div className="grid gap-3 sm:grid-cols-2">
            {monitoringMetrics.map((metric) => (
              <div key={metric.label} className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
                <div className="text-sm text-slate-500">{metric.label}</div>
                <div className="mt-2 text-2xl font-semibold text-slate-950">{metric.value}</div>
                <div className="mt-1 text-xs text-slate-500">{metric.sub}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </PageFrame>
  );
}
