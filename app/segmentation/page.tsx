import { ChartNoAxesCombined, Layers3, MessagesSquare, UsersRound } from "lucide-react";
import { PersonaScatterChart, PersonaSizeChart } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable } from "../components/PageParts";
import { loadSegmentationDashboard } from "../lib/segmentation";

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function SegmentationPage() {
  const data = loadSegmentationDashboard();

  if (!data) {
    return (
      <PageFrame>
        <PageHeader
          eyebrow="User Segmentation"
          title="Behavioral User Personas"
          description="Run python src/user_segmentation.py to generate the segmentation dashboard artifacts."
        />
      </PageFrame>
    );
  }

  return (
    <PageFrame>
      <PageHeader
        eyebrow="User Segmentation"
        title="Behavioral User Personas"
        description="Audience segments inferred from comment language using TF-IDF, TruncatedSVD, and KMeans. These are behavioral comment personas, not verified demographic profiles."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Comments segmented"
          value={data.metrics.comments_segmented.toLocaleString()}
          sub="unique English comments"
          icon={MessagesSquare}
        />
        <MetricCard
          label="Selected personas"
          value={String(data.method.selected_k)}
          sub="best tested silhouette score"
          icon={UsersRound}
        />
        <MetricCard
          label="SVD dimensions"
          value={String(data.method.svd_components)}
          sub={`${percent(data.metrics.explained_variance)} explained variance`}
          icon={Layers3}
        />
        <MetricCard
          label="Silhouette score"
          value={data.metrics.silhouette_score.toFixed(4)}
          sub="overlapping social-media language"
          icon={ChartNoAxesCombined}
        />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-2">
        <Panel
          title="Persona Sizes"
          description="Each comment is assigned to its nearest KMeans cluster after dimensionality reduction."
        >
          <PersonaSizeChart data={data.personas} />
        </Panel>
        <Panel
          title="TruncatedSVD Cluster Projection"
          description="A sampled two-dimensional view for interpretation. KMeans used all reduced dimensions."
        >
          <PersonaScatterChart data={data.scatter_points} />
        </Panel>
      </div>

      <Panel
        title="Interpreted Personas"
        description="Names and recommendations are derived from each cluster's distinctive TF-IDF terms, dominant issue, topic, and sentiment."
        className="mt-6"
      >
        <div className="grid gap-4 lg:grid-cols-2">
          {data.personas.map((persona) => (
            <article key={persona.cluster_id} className="rounded-lg border border-slate-200 p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.12em] text-[#1428A0]">
                    Cluster {persona.cluster_id}
                  </div>
                  <h3 className="mt-1 text-base font-semibold text-slate-950">{persona.persona}</h3>
                </div>
                <span className="shrink-0 rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                  {persona.size.toLocaleString()} ({percent(persona.share)})
                </span>
              </div>

              <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">Dominant sentiment</dt>
                  <dd className="mt-1 text-slate-800">{persona.dominant_sentiment} ({percent(persona.sentiment_share)})</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">Top issue</dt>
                  <dd className="mt-1 text-slate-800">{persona.top_issue} ({percent(persona.top_issue_share)})</dd>
                </div>
              </dl>

              <div className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600 ring-1 ring-slate-200">
                <span className="font-semibold text-slate-800">Distinctive terms:</span> {persona.top_terms}
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-700">
                <span className="font-semibold text-slate-950">Recommendation:</span> {persona.recommendation}
              </p>
              <p className="mt-3 border-l-2 border-slate-200 pl-3 text-xs italic leading-5 text-slate-500">
                {persona.representative_comments[0]}
              </p>
            </article>
          ))}
        </div>
      </Panel>

      <Panel
        title="KMeans Model Selection"
        description="The pipeline tests k=4 through k=8 and selects the viable result with the highest silhouette score."
        className="mt-6"
      >
        <SimpleTable
          columns={["k", "Silhouette", "Calinski-Harabasz", "Inertia", "Smallest segment", "Largest segment"]}
          rows={data.evaluation.map((item) => [
            item.k,
            item.silhouette_score.toFixed(4),
            item.calinski_harabasz_score.toFixed(3),
            item.inertia.toFixed(3),
            `${item.smallest_cluster.toLocaleString()} (${percent(item.smallest_cluster_share)})`,
            item.largest_cluster.toLocaleString(),
          ])}
        />
      </Panel>

      <Panel title="Method and Limitations" className="mt-6">
        <p className="text-sm leading-6 text-slate-600">
          TF-IDF represents distinctive unigrams, bigrams, and trigrams. TruncatedSVD compresses the sparse vectors,
          and KMeans groups comments by language similarity. Persona labels interpret the resulting clusters. Because
          YouTube comments overlap heavily and do not include reliable demographics, these segments describe audience
          behaviors and discussion patterns rather than confirmed individual user identities.
        </p>
      </Panel>
    </PageFrame>
  );
}
