import sys
import time
from pathlib import Path
from uuid import uuid4

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langsmith.utils import LangSmithAuthError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_router import route_with_langchain
from tracing_utils import tracing_status


def main():
    status = tracing_status()

    if not status["enabled"]:
        raise ValueError(
            "LangSmith tracing is disabled. Add LANGSMITH_TRACING=true, "
            "LANGSMITH_API_KEY, and LANGSMITH_PROJECT to .env."
        )

    query = "What should Samsung improve to increase customer satisfaction?"
    trace_id = uuid4()
    client = Client()

    try:
        list(client.list_projects(limit=1))

        decision = route_with_langchain(
            query,
            langsmith_extra={
                "run_id": trace_id,
                "metadata": {
                    "source": "langsmith_trace_test",
                    "evaluation_type": "router_smoke_test",
                },
                "tags": ["router-smoke-test"],
            },
        )

        wait_for_all_tracers()
        client.flush()
        runs = []

        for _ in range(30):
            runs = list(
                client.list_runs(
                    project_name=status["project"],
                    trace_id=trace_id,
                )
            )
            if runs:
                time.sleep(2)
                runs = list(
                    client.list_runs(
                        project_name=status["project"],
                        trace_id=trace_id,
                    )
                )
                break
            time.sleep(1)
    finally:
        client.close()

    if not runs:
        raise RuntimeError(
            "The router completed, but LangSmith did not return the trace."
        )

    print("LangSmith trace uploaded and read back successfully.")
    print("Project:", status["project"])
    print("Trace ID:", trace_id)
    print("Recorded spans:", len(runs))
    print("Span names:", ", ".join(run.name for run in runs))
    print("Query:", query)
    print("Selected agent:", decision["selected_agent"])
    print("Routing confidence:", decision["confidence"])
    print("Routing reason:", decision["reason"])


if __name__ == "__main__":
    try:
        main()
    except LangSmithAuthError:
        raise SystemExit(
            "LangSmith authentication failed. Replace LANGSMITH_API_KEY in .env "
            "with a valid key, then run this command again."
        )
