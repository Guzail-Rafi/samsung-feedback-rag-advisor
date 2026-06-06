import { Brain, Layers3, MessageCircleMore, SplitSquareHorizontal } from "lucide-react";
import { MetricCard, PageFrame, PageHeader, Panel } from "../components/PageParts";
import { sentimentTopics, topics } from "../lib/mock";

export default function TopicsPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Topics"
        title="Topic Modeling"
        description="General LDA topics and sentiment-specific topic clusters interpreted from top words."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="General topics" value="8" sub="topic_keywords.csv" icon={Brain} />
        <MetricCard label="Sentiment topics" value="8" sub="positive, negative, neutral" icon={SplitSquareHorizontal} />
        <MetricCard label="Model type" value="LDA" sub="interpreted labels" icon={Layers3} />
        <MetricCard label="Top signal" value="Upgrade" sub="model comparisons" icon={MessageCircleMore} />
      </div>

      <Panel title="General Topics" className="mt-6">
        <div className="grid gap-4 lg:grid-cols-2">
          {topics.map((topic) => (
            <div key={topic.id} className="rounded-lg border border-slate-200 p-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-sm font-semibold text-slate-950">{topic.name}</h2>
                <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600">
                  Topic {topic.id}
                </span>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">{topic.signal}</p>
              <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-500 ring-1 ring-slate-200">
                {topic.words}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Sentiment-Specific Topics" className="mt-6">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {sentimentTopics.map((topic) => (
            <div key={`${topic.sentiment}-${topic.topic}`} className="rounded-lg border border-slate-200 p-4">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: topic.color }} />
                {topic.sentiment}
              </div>
              <div className="mt-3 text-sm font-semibold text-slate-950">{topic.topic}</div>
              <p className="mt-2 text-xs leading-5 text-slate-500">{topic.words}</p>
            </div>
          ))}
        </div>
      </Panel>
    </PageFrame>
  );
}
