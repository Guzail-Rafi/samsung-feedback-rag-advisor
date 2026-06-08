import { DocumentChat } from "../components/DocumentChat";
import { PageFrame, PageHeader } from "../components/PageParts";

export default function DocumentsPage() {
  return (
    <PageFrame>
      <PageHeader
        eyebrow="Samsung Document RAG"
        title="Upload and Chat with Samsung Documents"
        description="Add Samsung reports, product documents, research, or customer-feedback files. The assistant answers from the uploaded evidence with citations."
      />
      <DocumentChat />
    </PageFrame>
  );
}
