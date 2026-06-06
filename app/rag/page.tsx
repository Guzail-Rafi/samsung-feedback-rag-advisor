import { FileQuestion, SearchCheck, ShieldCheck, Sparkles } from "lucide-react";
import { PrecisionChart } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable, StatusPill } from "../components/PageParts";
import { ragEvaluation, ragQuestions, retrievedEvidence } from "../lib/mock";

export default function RagPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Feedback RAG"
        title="Grounded Question Answering"
        description="Question answering outputs paired with retrieved comment evidence, categories, topics, and retrieval scores."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Questions" value="5" sub="rag_answers.csv" icon={FileQuestion} />
        <MetricCard label="High confidence" value="4" sub="battery, S-Pen, AI, Apple" icon={ShieldCheck} />
        <MetricCard label="Avg precision" value="93%" sub="manual Precision@5" icon={SearchCheck} />
        <MetricCard label="Answer mode" value="RAG" sub="retrieved comment evidence" icon={Sparkles} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[1fr_380px]">
        <Panel title="Question Answers">
          <div className="space-y-4">
            {ragQuestions.map((item) => (
              <div key={item.query} className="rounded-lg border border-slate-200 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <h2 className="text-sm font-semibold text-slate-950">{item.query}</h2>
                  <StatusPill value={item.confidence} />
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-600">{item.answer}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {item.evidence.map((evidence) => (
                    <span key={evidence} className="rounded-md bg-slate-50 px-2 py-1 text-xs text-slate-500 ring-1 ring-slate-200">
                      {evidence}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Retrieved Evidence Side Panel">
          <div className="space-y-3">
            {retrievedEvidence.map((item) => (
              <div key={`${item.query}-${item.score}`} className="rounded-lg border border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs font-semibold uppercase tracking-[0.12em] text-[#1428A0]">{item.query}</span>
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700">
                    {item.score.toFixed(3)}
                  </span>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{item.comment}</p>
                <div className="mt-3 grid gap-2 text-xs">
                  <div className="rounded-md bg-slate-50 px-2 py-1 text-slate-600 ring-1 ring-slate-200">
                    Sentiment: <span className="font-medium text-slate-900">{item.sentiment}</span>
                  </div>
                  <div className="rounded-md bg-slate-50 px-2 py-1 text-slate-600 ring-1 ring-slate-200">
                    Issue: <span className="font-medium text-slate-900">{item.issue}</span>
                  </div>
                  <div className="rounded-md bg-slate-50 px-2 py-1 text-slate-600 ring-1 ring-slate-200">
                    Topic: <span className="font-medium text-slate-900">{item.topic}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel title="Precision@5 Evaluation">
          <PrecisionChart data={ragEvaluation} />
        </Panel>

        <Panel title="Retrieval Metrics">
          <SimpleTable
            columns={["Query", "Precision", "Weighted", "Similarity"]}
            rows={ragEvaluation.map((item) => [
              item.query,
              item.precision.toFixed(2),
              item.weighted.toFixed(3),
              item.similarity.toFixed(3),
            ])}
          />
        </Panel>
      </div>
    </PageFrame>
  );
}
