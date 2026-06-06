import { RefreshCw, Route, ShieldCheck, Sparkles } from "lucide-react";
import { RoadmapRefinement } from "../components/RoadmapRefinement";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import { refinementDecisions } from "../lib/mock";

export default function RefinementPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Roadmap Refinement"
        title="Strategy Negotiation Workspace"
        description="Negotiate product roadmap changes, see the system accept or reject requests, and watch the roadmap update live."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Interaction mode" value="Negotiate" sub="user feedback loop" icon={RefreshCw} />
        <MetricCard label="Decision types" value="3" sub="accept, reject, alternative" icon={ShieldCheck} />
        <MetricCard label="Roadmap phases" value="3" sub="live editable plan" icon={Route} />
        <MetricCard label="Evidence file" value="CSV" sub="strategy_refinement_results.csv" icon={Sparkles} />
      </div>

      <div className="mt-6">
        <RoadmapRefinement />
      </div>

      <Panel title="Saved Refinement Examples" className="mt-6">
        <SimpleTable
          columns={["User negotiation", "System verdict", "Evidence-based reason", "Affected phase"]}
          rows={refinementDecisions.map((item) => [item.feedback, item.verdict, item.reason, item.phase])}
        />
      </Panel>
    </PageFrame>
  );
}
