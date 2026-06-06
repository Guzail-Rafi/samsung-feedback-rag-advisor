import { Activity, MessageSquare, TrendingDown, TrendingUp } from "lucide-react";
import { SentimentDonut } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import { sentimentData, sentimentTopics } from "../lib/mock";

export default function SentimentPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Sentiment"
        title="Customer Sentiment Analysis"
        description="VADER sentiment output across cleaned YouTube comments with positive, neutral, and negative topic signals."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Positive" value="48%" sub="brand praise and upgrade interest" icon={TrendingUp} />
        <MetricCard label="Neutral" value="31%" sub="model and comparison chatter" icon={Activity} />
        <MetricCard label="Negative" value="21%" sub="feature and value concerns" icon={TrendingDown} />
        <MetricCard label="Source file" value="CSV" sub="comments_with_sentiment.csv" icon={MessageSquare} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Distribution">
          <SentimentDonut data={sentimentData} />
        </Panel>

        <Panel title="Sentiment Topics">
          <SimpleTable
            columns={["Sentiment", "Topic", "Top words"]}
            rows={sentimentTopics.map((item) => [
              <span key="sentiment" className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                {item.sentiment}
              </span>,
              item.topic,
              item.words,
            ])}
          />
        </Panel>
      </div>
    </PageFrame>
  );
}
