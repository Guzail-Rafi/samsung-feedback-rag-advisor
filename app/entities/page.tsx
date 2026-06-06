import { Building2, GitCompare, PenTool, Tags } from "lucide-react";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable, StatusPill } from "../components/PageParts";
import { entities, entityExamples } from "../lib/mock";

export default function EntitiesPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Entities"
        title="Named Entity Extraction"
        description="Brand, product, competitor, and feature mentions extracted from comment text."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Top brand" value="Samsung" sub="highest mention volume" icon={Building2} />
        <MetricCard label="Top competitor" value="iPhone" sub="comparison signal" icon={GitCompare} />
        <MetricCard label="Feature entity" value="S Pen" sub="negative risk" icon={PenTool} />
        <MetricCard label="Entity rows" value="11,165" sub="ner_entities.csv" icon={Tags} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel title="Entity Leaderboard">
          <SimpleTable
            columns={["Entity", "Type", "Mentions", "Sentiment"]}
            rows={entities.map((item) => [
              <span key="entity" className="font-medium text-slate-950">{item.entity}</span>,
              item.type,
              item.mentions.toLocaleString(),
              <StatusPill key="sentiment" value={item.sentiment} />,
            ])}
          />
        </Panel>

        <Panel title="Evidence Examples">
          <div className="space-y-3">
            {entityExamples.map((item) => (
              <div key={item.entity} className="rounded-lg border border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-slate-950">{item.entity}</div>
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600">{item.issue}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-600">{item.comment}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </PageFrame>
  );
}
