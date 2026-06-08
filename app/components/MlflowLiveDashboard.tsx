"use client";

import { useEffect, useState } from "react";
import { Activity, Clock3, ExternalLink, RefreshCw, Route, ShieldCheck } from "lucide-react";

type TraceSpan = {
  name: string;
  type: string;
  status: string;
  duration_ms?: number;
};

type Trace = {
  trace_id: string;
  name: string;
  state: string;
  request_time?: string;
  duration_ms?: number;
  request_preview?: string;
  response_preview?: string;
  selected_agent?: string;
  llm_provider?: string;
  routing_method?: string;
  span_count: number;
  spans: TraceSpan[];
};

type MlflowStatus = {
  tracking_uri: string;
  ui_url: string;
  live_experiment: { name: string; id: string };
  pipeline_experiment: { name: string; id: string };
  summary: {
    trace_count: number;
    success_rate: number;
    average_latency_ms: number;
    fallback_count: number;
    route_counts: Record<string, number>;
    provider_counts: Record<string, number>;
  };
  traces: Trace[];
  pipeline_runs: Array<{
    run_id: string;
    name?: string;
    status: string;
    metrics: Record<string, number>;
  }>;
};

function shortPreview(value?: string) {
  if (!value) return "Not available";
  return value.length > 220 ? `${value.slice(0, 220)}...` : value;
}

export function MlflowLiveDashboard() {
  const [data, setData] = useState<MlflowStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/mlflow", { cache: "no-store" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Could not load MLflow data.");
      setData(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load MLflow data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  if (loading && !data) {
    return <div className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">Loading real MLflow traces...</div>;
  }

  if (error && !data) {
    return <div className="rounded-lg border border-red-200 bg-red-50 p-5 text-sm text-red-700">{error}</div>;
  }

  if (!data) return null;

  const cards = [
    { label: "Live traces", value: String(data.summary.trace_count), icon: Activity },
    { label: "Success rate", value: `${Math.round(data.summary.success_rate * 100)}%`, icon: ShieldCheck },
    { label: "Average latency", value: `${(data.summary.average_latency_ms / 1000).toFixed(1)}s`, icon: Clock3 },
    { label: "Fallbacks", value: String(data.summary.fallback_count), icon: Route },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div>
          <div className="text-sm font-semibold text-slate-950">Real MLflow tracking store</div>
          <div className="mt-1 font-mono text-xs text-slate-500">{data.tracking_uri}</div>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void refresh()}
            className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:border-[#1428A0] hover:text-[#1428A0]"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <a
            href={data.ui_url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-lg bg-[#1428A0] px-3 py-2 text-xs font-semibold text-white"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Open MLflow UI
          </a>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map(({ label, value, icon: Icon }) => (
          <div key={label} className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between text-sm text-slate-500">
              {label}
              <Icon className="h-4 w-4 text-[#1428A0]" />
            </div>
            <div className="mt-3 text-2xl font-semibold text-slate-950">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="text-base font-semibold text-slate-950">Recent live advisor traces</div>
          <p className="mt-1 text-sm text-slate-500">{data.live_experiment.name}</p>
          <div className="mt-4 space-y-3">
            {data.traces.length === 0 && <p className="text-sm text-slate-500">No live traces yet. Ask the advisor a question.</p>}
            {data.traces.map((trace) => (
              <details key={trace.trace_id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <summary className="cursor-pointer list-none">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">{trace.selected_agent || trace.name}</div>
                      <div className="mt-1 font-mono text-[11px] text-slate-500">{trace.trace_id}</div>
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs text-slate-600">
                      <span>{trace.state}</span>
                      <span>{trace.span_count} spans</span>
                      <span>{((trace.duration_ms || 0) / 1000).toFixed(1)}s</span>
                      {trace.llm_provider && <span>LLM: {trace.llm_provider}</span>}
                    </div>
                  </div>
                </summary>
                <div className="mt-4 grid gap-3 text-xs">
                  <div className="rounded-md bg-white p-3 ring-1 ring-slate-200">
                    <div className="font-semibold text-slate-800">Request</div>
                    <div className="mt-1 break-words text-slate-600">{shortPreview(trace.request_preview)}</div>
                  </div>
                  <div className="space-y-2">
                    {trace.spans.map((span, index) => (
                      <div key={`${span.name}-${index}`} className="flex flex-wrap justify-between gap-2 rounded-md bg-white px-3 py-2 ring-1 ring-slate-200">
                        <span className="font-semibold text-slate-700">{index + 1}. {span.name}</span>
                        <span className="text-slate-500">{span.type} - {span.status} - {span.duration_ms ?? 0}ms</span>
                      </div>
                    ))}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </section>

        <div className="space-y-5">
          <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="text-base font-semibold text-slate-950">Route usage</div>
            <div className="mt-4 space-y-2">
              {Object.entries(data.summary.route_counts).map(([route, count]) => (
                <div key={route} className="flex justify-between gap-3 rounded-md bg-slate-50 px-3 py-2 text-xs ring-1 ring-slate-200">
                  <span className="font-mono text-slate-700">{route}</span>
                  <span className="font-semibold text-[#1428A0]">{count}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="text-base font-semibold text-slate-950">Latest pipeline run</div>
            {data.pipeline_runs[0] ? (
              <div className="mt-4 space-y-2 text-xs text-slate-600">
                <div>Run: <span className="font-mono">{data.pipeline_runs[0].run_id}</span></div>
                <div>Status: {data.pipeline_runs[0].status}</div>
                <div>Metrics logged: {Object.keys(data.pipeline_runs[0].metrics).length}</div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-500">Run `python src\mlflow_monitoring.py` to create a pipeline monitoring run.</p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
