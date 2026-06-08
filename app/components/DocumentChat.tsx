"use client";

import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Bot, FileText, LoaderCircle, Send, Trash2, Upload, UserRound } from "lucide-react";
import { Panel } from "./PageParts";

type IndexedDocument = {
  document_id: string;
  filename: string;
  extension: string;
  size_bytes: number;
  character_count: number;
  chunk_count: number;
  uploaded_at: string;
  matched_samsung_signals?: string[];
};

type Evidence = {
  evidence_id: string;
  filename: string;
  chunk_index: number;
  content: string;
  similarity: number;
};

type ChatMessage = {
  role: "assistant" | "user";
  content: string;
  evidence?: Evidence[];
  confidence?: string;
  provider?: string;
  model?: string;
  fallback?: boolean;
};

type DocumentResponse = {
  error?: string;
  status?: string;
  documents?: IndexedDocument[];
  answer?: string;
  evidence?: Evidence[];
  confidence?: string;
  llmProvider?: string;
  model?: string;
  llmFallbackUsed?: boolean;
};

const welcomeMessage: ChatMessage = {
  role: "assistant",
  content:
    "Upload Samsung-related documents, then ask me to summarize, explain, compare, or find evidence. I will answer only from your indexed Samsung documents.",
};

function ReadableAnswer({ content }: { content: string }) {
  return (
    <div className="space-y-2">
      {content.split(/\r?\n/).map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} className="h-1" />;
        if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
          return (
            <h3 key={index} className="font-semibold text-slate-950">
              {trimmed.slice(2, -2)}
            </h3>
          );
        }
        if (/^#{1,4}\s+/.test(trimmed)) {
          return (
            <h3 key={index} className="font-semibold text-slate-950">
              {trimmed.replace(/^#{1,4}\s+/, "").replaceAll("**", "")}
            </h3>
          );
        }
        if (trimmed.startsWith("- ")) {
          return (
            <div key={index} className="flex gap-2">
              <span className="text-[#1428A0]">-</span>
              <span>{trimmed.slice(2).replaceAll("**", "")}</span>
            </div>
          );
        }
        if (/^\d+\.\s+/.test(trimmed)) {
          return (
            <div key={index} className="flex gap-2">
              <span className="font-semibold text-[#1428A0]">{trimmed.match(/^\d+\./)?.[0]}</span>
              <span>{trimmed.replace(/^\d+\.\s+/, "").replaceAll("**", "")}</span>
            </div>
          );
        }
        return <p key={index}>{trimmed.replaceAll("**", "")}</p>;
      })}
    </div>
  );
}

async function readResponse(response: Response) {
  const data = (await response.json()) as DocumentResponse;
  if (!response.ok || data.error) throw new Error(data.error || "Document request failed.");
  return data;
}

export function DocumentChat() {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([welcomeMessage]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [status, setStatus] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function refreshDocuments() {
    try {
      const response = await fetch("/api/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "list" }),
      });
      const data = await readResponse(response);
      setDocuments(data.documents ?? []);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not list documents.");
    }
  }

  useEffect(() => {
    void refreshDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, answering]);

  async function uploadDocuments() {
    const files = Array.from(fileRef.current?.files ?? []);
    if (!files.length) return;

    setUploading(true);
    setStatus("");
    try {
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        const data = await readResponse(await fetch("/api/documents", { method: "POST", body: form }));
        setStatus(
          data.status === "already_indexed"
            ? `${file.name} was already indexed.`
            : `${file.name} was indexed successfully.`,
        );
        setDocuments(data.documents ?? []);
      }
      if (fileRef.current) fileRef.current.value = "";
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    setStatus("");
    try {
      const data = await readResponse(
        await fetch("/api/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "delete", document_id: documentId }),
        }),
      );
      setDocuments(data.documents ?? []);
      setStatus("Document deleted.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Delete failed.");
    }
  }

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    const question = input.trim();
    if (!question || answering) return;

    const nextMessages = [...messages, { role: "user" as const, content: question }];
    setMessages(nextMessages);
    setInput("");
    setAnswering(true);
    setStatus("");

    try {
      const data = await readResponse(
        await fetch("/api/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "chat",
            message: question,
            messages: nextMessages.slice(-8).map(({ role, content }) => ({ role, content })),
          }),
        }),
      );
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer || "No answer was returned.",
          evidence: data.evidence,
          confidence: data.confidence,
          provider: data.llmProvider,
          model: data.model,
          fallback: data.llmFallbackUsed,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error instanceof Error ? error.message : "Document chat failed.",
        },
      ]);
    } finally {
      setAnswering(false);
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
      <div className="space-y-6">
        <Panel
          title="Add Samsung documents"
          description="Accepted: PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML. Maximum 10 MB each."
        >
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.csv,.json,.html,.htm"
            className="block w-full rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-[#1428A0] file:px-3 file:py-2 file:text-sm file:font-medium file:text-white"
          />
          <button
            type="button"
            onClick={() => void uploadDocuments()}
            disabled={uploading}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-[#1428A0] px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-60"
          >
            {uploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            {uploading ? "Reading and indexing..." : "Upload and index"}
          </button>
          {status && <p className="mt-3 text-xs leading-5 text-slate-600">{status}</p>}
        </Panel>

        <Panel
          title={`Indexed documents (${documents.length})`}
          description="Stored separately in the samsung_documents ChromaDB collection."
        >
          <div className="space-y-3">
            {documents.length === 0 && <p className="text-sm text-slate-500">No documents indexed yet.</p>}
            {documents.map((document) => (
              <div key={document.document_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <FileText className="h-4 w-4 shrink-0 text-[#1428A0]" />
                      <span className="truncate">{document.filename}</span>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      {document.chunk_count} chunks - {(document.size_bytes / 1024).toFixed(1)} KB
                    </p>
                    {document.matched_samsung_signals && document.matched_samsung_signals.length > 0 && (
                      <p className="mt-1 truncate text-xs text-slate-500">
                        Samsung signals: {document.matched_samsung_signals.join(", ")}
                      </p>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => void deleteDocument(document.document_id)}
                    className="rounded-md p-2 text-slate-400 hover:bg-red-50 hover:text-red-600"
                    aria-label={`Delete ${document.filename}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel className="flex min-h-[720px] flex-col p-0">
        <div className="border-b border-slate-200 px-5 py-4">
          <h2 className="font-semibold text-slate-950">Samsung Document Assistant</h2>
          <p className="mt-1 text-xs text-slate-500">Answers are grounded only in your uploaded Samsung documents.</p>
        </div>
        <div className="flex-1 space-y-5 overflow-y-auto p-5">
          {messages.map((message, index) => {
            const assistant = message.role === "assistant";
            return (
              <div key={index} className={`flex gap-3 ${assistant ? "" : "justify-end"}`}>
                {assistant && (
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#EAF0FF] text-[#1428A0]">
                    <Bot className="h-4 w-4" />
                  </span>
                )}
                <div className={`max-w-3xl rounded-lg px-4 py-3 text-sm leading-6 ${assistant ? "bg-slate-50 text-slate-800" : "bg-[#1428A0] text-white"}`}>
                  {assistant ? <ReadableAnswer content={message.content} /> : message.content}
                  {assistant && message.provider && (
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span className="rounded-md bg-white px-2 py-1 ring-1 ring-slate-200">
                        Answer LLM: {message.provider}{message.fallback ? " fallback" : ""}
                      </span>
                      {message.model && <span className="rounded-md bg-white px-2 py-1 ring-1 ring-slate-200">Model: {message.model}</span>}
                      {message.confidence && <span className="rounded-md bg-white px-2 py-1 ring-1 ring-slate-200">Confidence: {message.confidence}</span>}
                    </div>
                  )}
                  {assistant && message.evidence && message.evidence.length > 0 && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">
                        Retrieved document evidence ({message.evidence.length})
                      </summary>
                      <div className="mt-3 space-y-3">
                        {message.evidence.map((item) => (
                          <div key={`${item.evidence_id}-${item.chunk_index}`} className="rounded-md bg-slate-50 p-3">
                            <div className="font-semibold text-slate-800">
                              [{item.evidence_id}] {item.filename} - chunk {item.chunk_index}
                            </div>
                            <p className="mt-1 line-clamp-5 leading-5">{item.content}</p>
                            <p className="mt-1 text-slate-400">Similarity: {item.similarity}</p>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
                {!assistant && (
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-200 text-slate-600">
                    <UserRound className="h-4 w-4" />
                  </span>
                )}
              </div>
            );
          })}
          {answering && (
            <div className="flex items-center gap-3 text-sm text-slate-500">
              <LoaderCircle className="h-4 w-4 animate-spin text-[#1428A0]" />
              Retrieving document evidence and preparing an answer...
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <form onSubmit={submitQuestion} className="border-t border-slate-200 p-4">
          <div className="flex gap-3">
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={documents.length ? "Ask about your Samsung documents..." : "Upload a Samsung document first..."}
              disabled={answering || documents.length === 0}
              className="min-w-0 flex-1 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none focus:border-[#1428A0] focus:ring-2 focus:ring-[#1428A0]/10 disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={answering || !input.trim() || documents.length === 0}
              className="flex items-center gap-2 rounded-lg bg-[#1428A0] px-4 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
              Ask
            </button>
          </div>
        </form>
      </Panel>
    </div>
  );
}
