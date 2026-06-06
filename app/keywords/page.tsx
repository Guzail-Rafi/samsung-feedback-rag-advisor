import { Hash, ListFilter, Tags, TrendingUp } from "lucide-react";
import { KeywordBarChart } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import { categoryKeywords, topKeywords } from "../lib/mock";

export default function KeywordsPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Keywords"
        title="TF-IDF Keyword Extraction"
        description="Overall and category-level keywords from the cleaned Samsung feedback corpus."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Top keyword" value="AI" sub="highest TF-IDF score" icon={Hash} />
        <MetricCard label="Feature keyword" value="Pen" sub="S-Pen demand signal" icon={Tags} />
        <MetricCard label="Category keywords" value="10" sub="sampled rows" icon={ListFilter} />
        <MetricCard label="Strategy signal" value="Value" sub="buy, worth, better" icon={TrendingUp} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel title="Overall Keywords">
          <KeywordBarChart data={topKeywords} />
        </Panel>

        <Panel title="Category Keywords">
          <SimpleTable
            columns={["Category", "Keyword", "TF-IDF score"]}
            rows={categoryKeywords.map((item) => [item.category, item.keyword, item.score.toFixed(2)])}
          />
        </Panel>
      </div>
    </PageFrame>
  );
}
