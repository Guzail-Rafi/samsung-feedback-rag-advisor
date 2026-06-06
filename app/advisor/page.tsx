import { AdvisorChat } from "../components/AdvisorChat";
import { PageFrame, PageHeader } from "../components/PageParts";

export default function AdvisorPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Strategy Chat"
        title="Strategy and Question Answering Chat"
        description="A ChatGPT-style interface for customer feedback questions, roadmap decisions, and evidence-backed strategy refinement."
      />
      <AdvisorChat />
    </PageFrame>
  );
}
