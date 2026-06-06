import { AlertTriangle, BatteryCharging, Camera, PenLine, Target } from "lucide-react";
import { IssueBarChart } from "../components/Charts";
import { MetricCard, PageFrame, PageHeader, Panel, SimpleTable, StatusPill } from "../components/PageParts";
import { issueData } from "../lib/mock";

export default function IssuesPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Issues"
        title="Issue Classification"
        description="Hybrid keyword and embedding classification grouped into customer concern categories."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Top issue" value="Battery" sub="Battery / Charging" icon={BatteryCharging} />
        <MetricCard label="Feature risk" value="S-Pen" sub="removal concerns" icon={PenLine} />
        <MetricCard label="Creator issue" value="Camera" sub="Phase 1 candidate" icon={Camera} />
        <MetricCard label="Priority categories" value="6" sub="shown in this view" icon={Target} />
      </div>

      <div className="mt-6 grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel title="Category Volume">
          <IssueBarChart data={issueData} />
        </Panel>

        <Panel title="Priority Table">
          <SimpleTable
            columns={["Issue", "Count", "Share", "Sentiment", "Urgency", "Recommendation"]}
            rows={issueData.map((issue) => [
              issue.issue,
              issue.count.toLocaleString(),
              issue.share,
              issue.sentiment,
              <StatusPill key="urgency" value={issue.urgency} />,
              issue.recommendation,
            ])}
          />
        </Panel>
      </div>

      <Panel title="Classification Note" className="mt-6">
        <div className="flex items-start gap-3 rounded-lg bg-amber-50 p-4 text-sm leading-6 text-amber-900 ring-1 ring-amber-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          Battery, S-Pen, and camera categories should be treated as launch-planning inputs because they appear in issue labels, RAG answers, and strategy evidence.
        </div>
      </Panel>
    </PageFrame>
  );
}
