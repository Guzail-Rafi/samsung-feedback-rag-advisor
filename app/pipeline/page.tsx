import { CheckCircle2, Database, FileText, GitBranch } from "lucide-react";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable, StatusPill } from "../components/PageParts";
import { pipelineStages } from "../lib/mock";

export default function PipelinePage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Pipeline"
        title="NLP and RAG Processing Pipeline"
        description="Collection, preprocessing, enrichment, retrieval, strategy generation, and monitoring outputs."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Stages" value="12" sub="core and extended pipeline" icon={GitBranch} />
        <MetricCard label="Processed data" value="CSV" sub="data/processed" icon={Database} />
        <MetricCard label="Artifacts" value="MLflow" sub="mlruns/" icon={FileText} />
        <MetricCard label="Status" value="Ready" sub="all listed outputs present" icon={CheckCircle2} />
      </div>

      <Panel title="Stage Outputs" className="mt-6">
        <SimpleTable
          columns={["Stage", "Output", "Status"]}
          rows={pipelineStages.map((item) => [
            item.stage,
            <span key="output" className="font-mono text-xs text-slate-600">{item.output}</span>,
            <StatusPill key="status" value={item.status} />,
          ])}
        />
      </Panel>
    </PageFrame>
  );
}
