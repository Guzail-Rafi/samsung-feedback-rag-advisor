import { BadgeDollarSign, Camera, Lightbulb, Route, ShieldCheck, Sparkles } from "lucide-react";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import { refinementExamples, strategyPriorities, strategyRecommendations } from "../lib/mock";

export default function StrategyPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Strategy RAG"
        title="Evidence-Backed Strategy Recommendations"
        description="Retrieved strategy evidence, goal-aware recommendations, and S27 Ultra product planning logic."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Primary goal" value="Satisfaction" sub="customer_satisfaction" icon={ShieldCheck} />
        <MetricCard label="Phase 1" value="Battery" sub="plus camera and S-Pen" icon={Camera} />
        <MetricCard label="Pricing lever" value="Value" sub="visible hardware proof" icon={BadgeDollarSign} />
        <MetricCard label="AI role" value="Support" sub="practical workflows" icon={Sparkles} />
      </div>

      <Panel title="Strategic Recommendations" className="mt-6">
        <div className="grid gap-4 lg:grid-cols-3">
          {strategyRecommendations.map((item) => (
            <div key={item.goal} className="rounded-lg border border-slate-200 p-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-sm font-semibold text-slate-950">{item.goal}</h2>
                <span className="rounded-md bg-[#EAF0FF] px-2 py-1 text-xs font-semibold text-[#1428A0]">
                  {(item.score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-700">{item.recommendation}</p>
              <p className="mt-3 text-xs leading-5 text-slate-500">{item.evidence}</p>
            </div>
          ))}
        </div>
      </Panel>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1fr_1fr]">
        <Panel title="Roadmap Phases">
          <div className="space-y-3">
            {strategyPriorities.map((item) => (
              <div key={item.phase} className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
                <div className="flex items-center gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-[#1428A0] ring-1 ring-slate-200">
                    <Route className="h-4 w-4" />
                  </span>
                  <div>
                    <div className="text-sm font-semibold text-slate-950">{item.phase}: {item.title}</div>
                    <div className="mt-1 text-sm text-slate-600">{item.focus}</div>
                  </div>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-500">{item.rationale}</p>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Strategy Refinement Hand-Off">
          <SimpleTable
            columns={["Negotiation signal", "Roadmap impact"]}
            rows={refinementExamples.map((item) => [item.feedback, item.result])}
          />
        </Panel>
      </div>

      <Panel title="Decision Frame" className="mt-6">
        <div className="flex items-start gap-3 rounded-lg bg-[#F4F7FF] p-4 text-sm leading-6 text-slate-700 ring-1 ring-[#DDE7FF]">
          <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-[#1428A0]" />
          The roadmap should treat camera improvement as Phase 1 with battery and S-Pen restoration, then use Galaxy AI to amplify those hardware gains.
        </div>
      </Panel>
    </PageFrame>
  );
}
