import { AdvisorChat } from "../components/AdvisorChat";
import { PageHeader } from "../components/PageParts";

export default function AdvisorPage() {
  return (
    <div className="flex h-[calc(100dvh-57px)] min-h-0 flex-col overflow-hidden px-4 py-4 md:px-8 lg:h-dvh lg:py-6">
      <PageHeader
        eyebrow="Strategy Chat"
        title="Samsung Intelligence Chat"
        description="One ChatGPT-style interface for feedback, analytics, product strategy, current market research, and uploaded Samsung documents."
        className="mb-4 shrink-0"
      />
      <AdvisorChat />
    </div>
  );
}
