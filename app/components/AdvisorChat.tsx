"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Bot, RotateCcw, Send, Sparkles, UserRound } from "lucide-react";

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
    sentiment?: string;
    issue_category?: string;
    topic?: string;
    weighted_score?: number;
    strategy_score?: number;
    priority?: string;
  }>;
};

const starterPrompts = [
  "Give me an overall summary of Samsung feedback.",
  "What is the sentiment distribution?",
  "What are the main complaint categories?",
  "What are the top discussion topics?",
  "What are the top keywords?",
  "Why are users unhappy about the S-Pen?",
  "How should Samsung design the S27 Ultra?",
];

const CHAT_STORAGE_KEY = "galaxy-insight-rag-chat";
const initialMessages: ChatMessage[] = [
  {
    role: "assistant",
    content:
      "Ask me for summaries, sentiment, issues, topics, keywords, feedback evidence, or product strategy. A live router will select the appropriate specialist tool for each request.",
    sources: ["Live analytical tools", "ChromaDB feedback RAG", "ChromaDB strategy RAG"],
    availableTools: [
      "summarization_agent",
      "sentiment_agent",
      "issue_agent",
      "topic_agent",
      "keyword_agent",
      "feedback_rag_agent",
      "strategy_rag_agent",
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
  const [memoryLoaded, setMemoryLoaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

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
          content: `I could not get a live OpenAI response.\n\n${message}`,
          sources: ["/api/advisor", ".env"],
        },
      ]);
    } finally {
      setIsSending(false);
      textareaRef.current?.focus();
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
    <div className="grid min-h-[calc(100vh-4rem)] gap-5 xl:grid-cols-[1fr_320px]">
      <section className="flex min-h-[680px] flex-col rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#1428A0] text-white">
              <Bot className="h-4 w-4" />
            </span>
            <div>
              <div className="text-sm font-semibold text-slate-950">YouTube Intelligence Advisor</div>
              <div className="text-xs text-slate-500">Live multi-agent orchestration + RAG retrieval</div>
            </div>
          </div>
          <span className="rounded-md bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700 ring-1 ring-emerald-200">
            {isSending ? "Thinking" : "Ready"}
          </span>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-5">
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
                        message.model,
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
                    <div className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <div className="font-semibold text-slate-900">Routing decision</div>
                      <p className="mt-2">{message.routingReason}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {message.routingMethod && <span>Method: {message.routingMethod}</span>}
                        {message.routerModel && <span>Router model: {message.routerModel}</span>}
                        {message.routingConfidence !== undefined && (
                          <span>Confidence: {Math.round(message.routingConfidence * 100)}%</span>
                        )}
                      </div>
                      {message.toolTrace && message.toolTrace.length > 0 && (
                        <div className="mt-2 font-mono text-[11px] text-[#1428A0]">
                          {message.toolTrace.join(" -> ")}
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
                    </div>
                  )}
                  {assistant && message.retrieval && (
                    <div className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <div className="font-semibold text-slate-900">RAG retrieval</div>
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
                    </div>
                  )}
                  {assistant && message.evidence && message.evidence.length > 0 && (
                    <details className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 ring-1 ring-slate-200">
                      <summary className="cursor-pointer font-semibold text-slate-900">
                        Retrieved evidence ({message.evidence.length})
                      </summary>
                      <div className="mt-3 space-y-3">
                        {message.evidence.slice(0, 3).map((item, evidenceIndex) => (
                          <div key={evidenceIndex} className="rounded-md bg-slate-50 p-3 ring-1 ring-slate-200">
                            <p className="leading-5 text-slate-700">{item.comment}</p>
                            <div className="mt-2 flex flex-wrap gap-2">
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

        <form onSubmit={submitMessage} className="border-t border-slate-200 p-4">
          <div className="flex gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              rows={2}
              placeholder="Ask for analytics, feedback evidence, or product strategy"
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

      <aside className="space-y-5">
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
