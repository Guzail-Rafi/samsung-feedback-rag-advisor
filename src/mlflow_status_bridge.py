import json
import sys
from collections import Counter
from datetime import datetime, timezone

from mlflow import MlflowClient

from mlflow_tracing import (
    LIVE_EXPERIMENT_NAME,
    PIPELINE_EXPERIMENT_NAME,
    configure_mlflow,
    flush_mlflow_traces,
    get_tracking_uri,
)


def span_duration_ms(span):
    if not span.end_time_ns:
        return None
    return round((span.end_time_ns - span.start_time_ns) / 1_000_000, 1)


def timestamp_to_iso(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).isoformat()


def trace_summary(trace):
    spans = trace.data.spans
    root = next((span for span in spans if span.parent_id is None), spans[0] if spans else None)
    attributes = root.attributes if root else {}

    return {
        "trace_id": trace.info.trace_id,
        "name": trace.info.tags.get("mlflow.traceName", "Trace"),
        "state": str(trace.info.state),
        "request_time": timestamp_to_iso(trace.info.request_time),
        "duration_ms": trace.info.execution_time_ms,
        "request_preview": trace.info.request_preview,
        "response_preview": trace.info.response_preview,
        "selected_agent": attributes.get("samsung.selectedAgent"),
        "llm_provider": attributes.get("samsung.llmProvider"),
        "llm_fallback_used": attributes.get("samsung.llmFallbackUsed"),
        "routing_method": attributes.get("samsung.routingMethod"),
        "span_count": len(spans),
        "spans": [
            {
                "name": span.name,
                "type": str(span.span_type),
                "status": str(span.status.status_code).split(".")[-1],
                "duration_ms": span_duration_ms(span),
            }
            for span in spans
        ],
    }


def run_summary(run):
    return {
        "run_id": run.info.run_id,
        "name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time": datetime.fromtimestamp(
            run.info.start_time / 1000,
            tz=timezone.utc,
        ).isoformat() if run.info.start_time else None,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }


def main():
    live_experiment = configure_mlflow(LIVE_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = list(
        client.search_traces(
            experiment_ids=[live_experiment.experiment_id],
            max_results=50,
            order_by=["timestamp_ms DESC"],
            include_spans=True,
            flush=True,
        )
    )
    summaries = [trace_summary(trace) for trace in traces]
    successful = [item for item in summaries if "OK" in item["state"]]
    durations = [item["duration_ms"] for item in summaries if item["duration_ms"] is not None]
    route_counts = Counter(item["selected_agent"] or "unknown" for item in summaries)
    provider_counts = Counter(item["llm_provider"] or "none" for item in summaries)

    pipeline_experiment = configure_mlflow(PIPELINE_EXPERIMENT_NAME)
    runs = list(
        client.search_runs(
            [pipeline_experiment.experiment_id],
            max_results=5,
            order_by=["start_time DESC"],
        )
    )

    response = {
        "tracking_uri": get_tracking_uri(),
        "ui_url": "http://127.0.0.1:5000",
        "live_experiment": {
            "name": LIVE_EXPERIMENT_NAME,
            "id": live_experiment.experiment_id,
        },
        "pipeline_experiment": {
            "name": PIPELINE_EXPERIMENT_NAME,
            "id": pipeline_experiment.experiment_id,
        },
        "summary": {
            "trace_count": len(summaries),
            "success_rate": round(len(successful) / len(summaries), 3) if summaries else 0,
            "average_latency_ms": round(sum(durations) / len(durations), 1) if durations else 0,
            "fallback_count": sum(bool(item["llm_fallback_used"]) for item in summaries),
            "route_counts": dict(route_counts),
            "provider_counts": dict(provider_counts),
        },
        "traces": summaries[:20],
        "pipeline_runs": [run_summary(run) for run in runs],
    }
    print(json.dumps(response, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error), "type": error.__class__.__name__}, ensure_ascii=True))
        sys.exit(1)
    finally:
        flush_mlflow_traces()
