import { Activity, Box, Gauge, LineChart, Settings2, ShieldCheck } from "lucide-react";
import { PrecisionChart } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import {
  mlflowArtifacts,
  mlflowExperiment,
  mlflowLoggedMetrics,
  mlflowParams,
  monitoringMetrics,
  ragEvaluation,
  rerankerEvaluation,
} from "../lib/mock";

export default function MonitoringPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Evaluation & MLflow"
        title="MLflow Tracking and RAG Evaluation"
        description="Experiment metadata, logged metrics, logged parameters, artifacts, and model/provider settings."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label={monitoringMetrics[0].label} value={monitoringMetrics[0].value} sub={monitoringMetrics[0].sub} icon={Activity} />
        <MetricCard label={monitoringMetrics[1].label} value={monitoringMetrics[1].value} sub={monitoringMetrics[1].sub} icon={Gauge} />
        <MetricCard label={monitoringMetrics[2].label} value={monitoringMetrics[2].value} sub={monitoringMetrics[2].sub} icon={ShieldCheck} />
        <MetricCard label={monitoringMetrics[3].label} value={monitoringMetrics[3].value} sub={monitoringMetrics[3].sub} icon={LineChart} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel title="MLflow Experiment">
          <div className="grid gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
              <div className="text-xs uppercase tracking-[0.12em] text-slate-500">Experiment</div>
              <div className="mt-2 font-semibold text-slate-950">{mlflowExperiment.name}</div>
            </div>
            <div className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
              <div className="text-xs uppercase tracking-[0.12em] text-slate-500">Experiment ID</div>
              <div className="mt-2 font-mono text-xs text-slate-700">{mlflowExperiment.id}</div>
            </div>
            <div className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
              <div className="text-xs uppercase tracking-[0.12em] text-slate-500">Run ID</div>
              <div className="mt-2 font-mono text-xs text-slate-700">{mlflowExperiment.runId}</div>
            </div>
            <div className="rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200">
              <div className="text-xs uppercase tracking-[0.12em] text-slate-500">Tracking path</div>
              <div className="mt-2 font-mono text-xs text-slate-700">{mlflowExperiment.trackingPath}</div>
            </div>
          </div>
        </Panel>

        <Panel title="Model and Provider Settings">
          <div className="grid gap-3 sm:grid-cols-2">
            {mlflowParams.map((param) => (
              <div key={param.name} className="rounded-lg border border-slate-200 p-4">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                  <Settings2 className="h-3.5 w-3.5" />
                  {param.name}
                </div>
                <div className="mt-2 text-sm font-semibold text-slate-950">{param.value}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1fr_1fr]">
        <Panel title="Logged Metrics">
          <SimpleTable
            columns={["Metric", "Value", "Group"]}
            rows={mlflowLoggedMetrics.map((metric) => [metric.name, metric.value, metric.group])}
          />
        </Panel>

        <Panel title="Logged Artifacts">
          <div className="grid gap-2 sm:grid-cols-2">
            {mlflowArtifacts.map((artifact) => (
              <div key={artifact} className="flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200">
                <Box className="h-4 w-4 text-[#1428A0]" />
                <span className="font-mono text-xs">{artifact}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel title="Manual Precision@5">
          <PrecisionChart data={ragEvaluation} />
        </Panel>

        <Panel title="Reranker Evaluation">
          <SimpleTable
            columns={["Query", "Precision", "Avg score", "Max score"]}
            rows={rerankerEvaluation.map((item) => [
              item.query,
              item.precision.toFixed(2),
              item.score.toFixed(3),
              item.max.toFixed(3),
            ])}
          />
        </Panel>
      </div>

      <Panel title="Metric Details" className="mt-6">
        <SimpleTable
          columns={["Query", "Precision@5", "Weighted score", "Similarity", "Lexical"]}
          rows={ragEvaluation.map((item) => [
            item.query,
            item.precision.toFixed(2),
            item.weighted.toFixed(3),
            item.similarity.toFixed(3),
            item.lexical.toFixed(2),
          ])}
        />
      </Panel>
    </PageFrame>
  );
}
