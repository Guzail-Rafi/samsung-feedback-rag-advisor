import { PrecisionChart } from "../components/Charts";
import { MlflowLiveDashboard } from "../components/MlflowLiveDashboard";
import { PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import {
  ragEvaluation,
  rerankerEvaluation,
} from "../lib/mock";

export default function MonitoringPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Evaluation & MLflow"
        title="MLflow Tracking and RAG Evaluation"
        description="Real live advisor traces, nested RAG spans, pipeline runs, logged metrics, and retrieval evaluation."
      />

      <MlflowLiveDashboard />

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
