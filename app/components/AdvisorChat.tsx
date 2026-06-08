"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Bot, FileText, LoaderCircle, Paperclip, RotateCcw, Send, Sparkles, Trash2, UserRound } from "lucide-react";

type ChatMessage = {
  role: "assistant" | "user";
  content: string;
  sources?: string[];
  agent?: string;
  model?: string;
  mode?: string;
  confidence?: string;
  strategyGoal?: string;
  contextualQuery?: string;
  memoryUsed?: boolean;
  routingReason?: string;
  matchedTerms?: string[];
  toolTrace?: string[];
  availableTools?: string[];
  routingConfidence?: number;
  routingMethod?: string;
  routerModel?: string;
  normalizedQuery?: string;
  rewrittenQuery?: string;
  needsExternalResearch?: boolean;
  externalResearchFocus?: string[];
  llmProvider?: string;
  llmFallbackUsed?: boolean;
  routerProvider?: string;
  routerFallbackUsed?: boolean;
  mlflowTraceId?: string;
  externalEvidence?: Array<{
    title?: string;
    url?: string;
    snippet?: string;
    date?: string;
    retrieved_at?: string;
    relevance_reason?: string;
    source?: string;
  }>;
  webResearch?: {
    provider?: string;
    result_count?: number;
    retrieved_at?: string;
    errors?: string[];
    focus?: string[];
  };
  retrieval?: {
    vector_store?: string;
    collection?: string;
    embedding_model?: string;
    top_k?: number;
    score?: number;
    goal_relevance?: number;
  };
  evidence?: Array<{
    comment?: string;
    content?: string;
    evidence_id?: string;
    filename?: string;
    chunk_index?: number;
    similarity?: number;
    sentiment?: string;
    issue_category?: string;
    topic?: string;
    weighted_score?: number;
    strategy_score?: number;
    priority?: string;
  }>;
};

type IndexedDocument = {
  document_id: string;
  filename: string;
  size_bytes: number;
  chunk_count: number;
  uploaded_at: string;
};

const starterPrompts = [
  "Give me an overall summary of Samsung feedback.",
  "What is the sentiment distribution?",
  "What are the main complaint categories?",
  "What are the top discussion topics?",
  "What are the top keywords?",
  "Why are users unhappy about the S-Pen?",
  "How should Samsung design the S27 Ultra?",
  "How should Samsung price the next Ultra in the UAE using current offers?",
  "Summarize the uploaded Samsung document.",
];

const CHAT_STORAGE_KEY = "galaxy-insight-rag-chat";
const initialMessages: ChatMessage[] = [
  {
    role: "assistant",
    content:
      "Ask me about Samsung feedback, analytics, product strategy, current market research, or uploaded Samsung documents. A live router will select the appropriate specialist tool for each request.",
    sources: ["Live analytical tools", "ChromaDB feedback RAG", "ChromaDB strategy RAG", "ChromaDB document RAG"],
    availableTools: [
      "summarization_agent",
      "sentiment_agent",
      "issue_agent",
      "topic_agent",
      "keyword_agent",
      "feedback_rag_agent",
      "strategy_rag_agent",
      "web_augmented_strategy_rag",
      "samsung_document_rag",
    ],
  },
];

function renderInlineMarkdown(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold text-slate-950">
          {part.slice(2, -2)}
        </strong>
      );
    }

    return <span key={index}>{part}</span>;
  });
}

function MarkdownMessage({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const elements: ReactNode[] = [];
  let listType: "ul" | "ol" | null = null;
  let listItems: ReactNode[] = [];

  function flushList() {
    if (!listType) return;

    if (listType === "ul") {
      elements.push(
        <ul key={`list-${elements.length}`} className="ml-5 list-disc space-y-1">
          {listItems}
        </ul>,
      );
    } else {
      elements.push(
        <ol key={`list-${elements.length}`} className="ml-5 list-decimal space-y-1">
          {listItems}
        </ol>,
      );
    }

    listType = null;
    listItems = [];
  }

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      return;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushList();
      const level = heading[1].length;
      const text = heading[2];

      if (level === 1) {
        elements.push(
          <h2 key={index} className="pt-2 text-lg font-semibold leading-7 text-slate-950">
            {renderInlineMarkdown(text)}
          </h2>,
        );
      } else if (level === 2) {
        elements.push(
          <h3 key={index} className="pt-2 text-base font-semibold leading-6 text-slate-950">
            {renderInlineMarkdown(text)}
          </h3>,
        );
      } else {
        elements.push(
          <h4 key={index} className="pt-1 text-sm font-semibold leading-6 text-slate-950">
            {renderInlineMarkdown(text)}
          </h4>,
        );
      }

      return;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (listType !== "ul") {
        flushList();
        listType = "ul";
      }

      listItems.push(
        <li key={index} className="pl-1">
          {renderInlineMarkdown(bullet[1])}
        </li>,
      );
      return;
    }

    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) {
      if (listType !== "ol") {
        flushList();
        listType = "ol";
      }

      listItems.push(
        <li key={index} className="pl-1">
          {renderInlineMarkdown(numbered[1])}
        </li>,
      );
      return;
    }

    flushList();
    elements.push(
      <p key={index} className="leading-6">
        {renderInlineMarkdown(trimmed)}
      </p>,
    );
  });

  flushList();

  return <div className="space-y-2">{elements}</div>;
}

export function AdvisorChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [documentStatus, setDocumentStatus] = useState("");
  const [memoryLoaded, setMemoryLoaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const documentInputRef = useRef<HTMLInputElement | null>(null);

  const canSend = input.trim().length > 0 && !isSending;
  const userMessageCount = messages.filter((message) => message.role === "user").length;

  const latestSources = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    return lastAssistant?.sources ?? [];
  }, [messages]);

  const latestTools = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    return lastAssistant?.availableTools ?? [];
  }, [messages]);

  useEffect(() => {
    try {
      const storedMessages = window.localStorage.getItem(CHAT_STORAGE_KEY);
      if (storedMessages) {
        const parsed = JSON.parse(storedMessages) as ChatMessage[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
        }
      }
    } finally {
      setMemoryLoaded(true);
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
  }, []);

  useEffect(() => {
    if (!memoryLoaded) return;
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages.slice(-30)));
  }, [memoryLoaded, messages]);

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || isSending) return;

    const userMessage: ChatMessage = { role: "user", content: question };
    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);
    setInput("");
    setIsSending(true);

    try {
      const response = await fetch("/api/advisor", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Advisor request failed.");
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          agent: data.selectedAgent,
          model: data.model,
          mode: data.mode,
          confidence: data.confidence,
          strategyGoal: data.strategyGoal,
          contextualQuery: data.contextualQuery,
          memoryUsed: data.memoryUsed,
          routingReason: data.routingReason,
          matchedTerms: data.matchedTerms,
          toolTrace: data.toolTrace,
          availableTools: data.availableTools,
          routingConfidence: data.routingConfidence,
          routingMethod: data.routingMethod,
          routerModel: data.routerModel,
          normalizedQuery: data.normalizedQuery,
          rewrittenQuery: data.rewrittenQuery,
          needsExternalResearch: data.needsExternalResearch,
          externalResearchFocus: data.externalResearchFocus,
          externalEvidence: data.externalEvidence,
          webResearch: data.webResearch,
          llmProvider: data.llmProvider,
          llmFallbackUsed: data.llmFallbackUsed,
          routerProvider: data.routerProvider,
          routerFallbackUsed: data.routerFallbackUsed,
          mlflowTraceId: data.mlflowTraceId,
          retrieval: data.retrieval,
          evidence: data.evidence,
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown advisor error.";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `I could not get a live advisor response.\n\n${message}`,
          sources: ["/api/advisor", ".env"],
        },
      ]);
    } finally {
      setIsSending(false);
      textareaRef.current?.focus();
    }
  }

  async function refreshDocuments() {
    try {
      const response = await fetch("/api/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "list" }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not list documents.");
      setDocuments(data.documents ?? []);
    } catch (error) {
      setDocumentStatus(error instanceof Error ? error.message : "Could not list documents.");
    }
  }

  async function uploadDocuments() {
    const files = Array.from(documentInputRef.current?.files ?? []);
    if (!files.length || isUploading) return;

    setIsUploading(true);
    setDocumentStatus("");
    try {
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/api/documents", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Could not upload ${file.name}.`);
        setDocuments(data.documents ?? []);
        setDocumentStatus(
          data.status === "already_indexed"
            ? `${file.name} was already indexed.`
            : `${file.name} was indexed and is ready to chat with.`,
        );
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content:
              data.status === "already_indexed"
                ? `**${file.name}** is already in my Samsung document memory. You can ask questions about it naturally.`
                : `I read and indexed **${file.name}**. It is now part of my Samsung document memory, so you can ask questions about it naturally alongside feedback, strategy, and market research.`,
            agent: "samsung_document_rag",
            mode: "document_memory",
            sources: [file.name],
            availableTools: [
              "summarization_agent",
              "sentiment_agent",
              "issue_agent",
              "topic_agent",
              "keyword_agent",
              "feedback_rag_agent",
              "strategy_rag_agent",
              "web_augmented_strategy_rag",
              "samsung_document_rag",
            ],
          },
        ]);
      }
      if (documentInputRef.current) documentInputRef.current.value = "";
    } catch (error) {
      setDocumentStatus(error instanceof Error ? error.message : "Document upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    try {
      const response = await fetch("/api/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "delete", document_id: documentId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not delete document.");
      setDocuments(data.documents ?? []);
      setDocumentStatus("Document deleted.");
    } catch (error) {
      setDocumentStatus(error instanceof Error ? error.message : "Document delete failed.");
    }
  }

  function usePrompt(prompt: string) {
    setInput(prompt);
    textareaRef.current?.focus();
  }

  function clearConversation() {
    setMessages(initialMessages);
    window.localStorage.removeItem(CHAT_STORAGE_KEY);
    textareaRef.current?.focus();
  }

  return (
    <div className="grid min-h-0 flex-1 gap-5 overflow-hidden xl:grid-cols-[1fr_320px]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#1428A0] text-white">
              <Bot className="h-4 w-4" />
            </span>
            <div>
              <div className="text-sm font-semibold text-slate-950">Samsung Intelligence Advisor</div>
              <div className="text-xs text-slate-500">Feedback + strategy + web research + document RAG</div>
            </div>
          </div>
          <span className="rounded-md bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700 ring-1 ring-emerald-200">
            {isSending ? "Thinking" : "Ready"}
          </span>
        </div>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-5">
          {messages.map((message, index) => {
            const assistant = message.role === "assistant";
            return (
              <div key={index} className={`flex gap-3 ${assistant ? "" : "justify-end"}`}>
                {assistant && (
                  <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <Bot className="h-4 w-4" />
                  </span>
                )}
                <div
                  className={`max-w-3xl rounded-lg px-4 py-3 text-sm leading-6 ${
                    assistant ? "bg-slate-50 text-slate-800" : "bg-[#1428A0] text-white"
                  }`}
                >
                  {assistant ? <MarkdownMessage content={message.content} /> : message.content}
                  {assistant && (message.agent || message.model) && (
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      {[
                        message.mode,
                        message.agent,
                        message.llmProvider
                          ? `Answer LLM: ${message.llmProvider}${message.llmFallbackUsed ? " fallback" : ""}`
                          : undefined,
                        message.model ? `Model: ${message.model}` : undefined,
                        message.confidence,
                        message.strategyGoal,
                        message.memoryUsed ? "memory:on" : undefined,
                      ]
                        .filter(Boolean)
                        .map((item) => (
                          <span key={item} className="rounded-md bg-white px-2 py-1 ring-1 ring-slate-200">
                            {item}
                          </span>
                        ))}
                    </div>
                  )}
                  {assistant && message.routingReason && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">Routing decision</summary>
                      <p className="mt-2">{message.routingReason}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {message.routingMethod && <span>Method: {message.routingMethod}</span>}
                        {message.routerModel && <span>Router model: {message.routerModel}</span>}
                        {message.routerProvider && <span>Router provider: {message.routerProvider}</span>}
                        {message.needsExternalResearch && <span>External research: required</span>}
                        {message.routingConfidence !== undefined && (
                          <span>Confidence: {Math.round(message.routingConfidence * 100)}%</span>
                        )}
                      </div>
                      {message.toolTrace && message.toolTrace.length > 0 && (
                        <div className="mt-2 font-mono text-[11px] text-[#1428A0]">
                          {message.toolTrace.join(" -> ")}
                        </div>
                      )}
                      {message.mlflowTraceId && (
                        <div className="mt-2 font-mono text-[11px] text-slate-500">
                          MLflow trace: {message.mlflowTraceId}
                        </div>
                      )}
                      {message.matchedTerms && message.matchedTerms.length > 0 && (
                        <div className="mt-2">Matched: {message.matchedTerms.join(", ")}</div>
                      )}
                      {message.normalizedQuery && (
                        <div className="mt-2 rounded-md bg-slate-50 px-2 py-1 text-slate-500 ring-1 ring-slate-200">
                          Normalized query: {message.normalizedQuery}
                        </div>
                      )}
                      {message.externalResearchFocus && message.externalResearchFocus.length > 0 && (
                        <div className="mt-2">
                          Research focus: {message.externalResearchFocus.join(", ")}
                        </div>
                      )}
                    </details>
                  )}
                  {assistant && message.retrieval && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">RAG retrieval</summary>
                      <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
                        <span>Store: {message.retrieval.vector_store}</span>
                        <span>Collection: {message.retrieval.collection}</span>
                        <span>Model: {message.retrieval.embedding_model}</span>
                        <span>Top-k: {message.retrieval.top_k}</span>
                        <span>Avg score: {message.retrieval.score}</span>
                      </div>
                      {message.contextualQuery && (
                        <div className="mt-2 rounded-md bg-slate-50 px-2 py-1 text-slate-500 ring-1 ring-slate-200">
                          Memory query: {message.contextualQuery}
                        </div>
                      )}
                    </details>
                  )}
                  {assistant && message.evidence && message.evidence.length > 0 && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">
                        Retrieved evidence ({message.evidence.length})
                      </summary>
                      <div className="mt-3 space-y-3">
                        {message.evidence.slice(0, 3).map((item, evidenceIndex) => (
                          <div key={evidenceIndex} className="rounded-md bg-slate-50 p-3 ring-1 ring-slate-200">
                            <p className="leading-5 text-slate-700">{item.comment || item.content}</p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {item.evidence_id && <span>[{item.evidence_id}]</span>}
                              {item.filename && <span>File: {item.filename}</span>}
                              {item.chunk_index !== undefined && <span>Chunk: {item.chunk_index}</span>}
                              {item.similarity !== undefined && <span>Similarity: {item.similarity}</span>}
                              {item.sentiment && <span>Sentiment: {item.sentiment}</span>}
                              {item.issue_category && <span>Issue: {item.issue_category}</span>}
                              {item.topic && <span>Topic: {item.topic}</span>}
                              {item.priority && <span>Priority: {item.priority}</span>}
                              {(item.weighted_score || item.strategy_score) && (
                                <span>Score: {item.weighted_score ?? item.strategy_score}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {assistant && message.externalEvidence && message.externalEvidence.length > 0 && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">
                        External web evidence ({message.externalEvidence.length})
                      </summary>
                      <div className="mt-3 space-y-3">
                        {message.externalEvidence.map((item, evidenceIndex) => (
                          <div key={evidenceIndex} className="rounded-md bg-slate-50 p-3 ring-1 ring-slate-200">
                            {item.url ? (
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noreferrer"
                                className="font-semibold text-[#1428A0] hover:underline"
                              >
                                {item.title || item.url}
                              </a>
                            ) : (
                              <div className="font-semibold text-slate-900">{item.title}</div>
                            )}
                            {item.snippet && <p className="mt-2 leading-5 text-slate-700">{item.snippet}</p>}
                            <div className="mt-2 flex flex-wrap gap-2">
                              {item.source && <span>Source: {item.source}</span>}
                              {item.date && <span>Date: {item.date}</span>}
                              {item.retrieved_at && <span>Retrieved: {item.retrieved_at}</span>}
                            </div>
                            {item.relevance_reason && <p className="mt-2">{item.relevance_reason}</p>}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {assistant && message.sources && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.sources.map((source) => (
                        <span key={source} className="rounded-md bg-white px-2 py-1 text-xs text-slate-500 ring-1 ring-slate-200">
                          {source}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {!assistant && (
                  <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#EAF0FF] text-[#1428A0]">
                    <UserRound className="h-4 w-4" />
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <form onSubmit={submitMessage} className="shrink-0 border-t border-slate-200 bg-white p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <input
              ref={documentInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.md,.csv,.json,.html,.htm"
              onChange={() => void uploadDocuments()}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => documentInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-[#1428A0] hover:text-[#1428A0] disabled:opacity-60"
            >
              {isUploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Paperclip className="h-4 w-4" />}
              {isUploading ? "Indexing document..." : "Add document"}
            </button>
            {documents.length > 0 && <span className="text-xs text-slate-500">{documents.length} indexed</span>}
          </div>
          <div className="flex gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              rows={2}
              placeholder="Ask about Samsung feedback, strategy, research, or uploaded documents"
              disabled={isSending}
              className="min-h-12 flex-1 resize-none rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-[#1428A0] focus:ring-4 focus:ring-[#1428A0]/10"
            />
            <button
              type="submit"
              disabled={!canSend}
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-[#1428A0] text-white transition hover:bg-[#0F1F78] disabled:cursor-not-allowed disabled:bg-slate-300"
              aria-label="Send message"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </form>
      </section>

      <aside className="hidden min-h-0 space-y-5 overflow-hidden pr-1 xl:block">
        <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
              <FileText className="h-4 w-4 text-[#1428A0]" />
              Uploaded documents
            </div>
            <button
              type="button"
              onClick={() => documentInputRef.current?.click()}
              className="rounded-md border border-slate-200 px-2 py-1 text-xs font-semibold text-slate-600 hover:border-[#1428A0] hover:text-[#1428A0]"
            >
              Add
            </button>
          </div>
          {documentStatus && <p className="mt-2 text-xs leading-5 text-slate-500">{documentStatus}</p>}
          <div className="mt-3 space-y-2">
            {documents.length === 0 && <p className="text-xs text-slate-500">No Samsung documents indexed.</p>}
            {documents.map((document) => (
              <div key={document.document_id} className="flex items-start justify-between gap-2 rounded-md bg-slate-50 p-2 ring-1 ring-slate-200">
                <div className="min-w-0">
                  <div className="truncate text-xs font-semibold text-slate-700">{document.filename}</div>
                  <div className="mt-1 text-[11px] text-slate-500">{document.chunk_count} chunks</div>
                </div>
                <button
                  type="button"
                  onClick={() => void deleteDocument(document.document_id)}
                  className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                  aria-label={`Delete ${document.filename}`}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-slate-950">Conversation memory</div>
              <div className="mt-1 text-xs text-slate-500">
                {userMessageCount} user turns saved locally
              </div>
            </div>
            <button
              type="button"
              onClick={clearConversation}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-50"
              aria-label="Clear conversation memory"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
            <Sparkles className="h-4 w-4 text-[#1428A0]" />
            Prompts
          </div>
          <div className="mt-3 space-y-2">
            {starterPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => usePrompt(prompt)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-left text-sm text-slate-700 transition hover:border-[#1428A0] hover:bg-[#F4F7FF]"
              >
                {prompt}
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-sm font-semibold text-slate-950">Live specialist tools</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {latestTools.map((tool) => (
              <span key={tool} className="rounded-md bg-[#F4F7FF] px-2 py-1 text-xs text-[#1428A0] ring-1 ring-[#D9E2FF]">
                {tool}
              </span>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-sm font-semibold text-slate-950">Latest sources</div>
          <div className="mt-3 space-y-2">
            {latestSources.map((source) => (
              <div key={source} className="rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                {source}
              </div>
            ))}
          </div>
        </section>
      </aside>
    </div>
  );
}
