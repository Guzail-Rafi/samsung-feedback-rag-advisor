import functools
import inspect
import os
from pathlib import Path

from dotenv import load_dotenv

from tracing_utils import summarize_trace_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "mlflow.db"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "mlartifacts"
LIVE_EXPERIMENT_NAME = "Samsung_Live_Advisor_Tracing"
PIPELINE_EXPERIMENT_NAME = "Samsung_YouTube_RAG_Monitoring"

_configured_experiment = None


def mlflow_enabled():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    value = os.getenv("MLFLOW_TRACING_ENABLED", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_tracking_uri():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return f"sqlite:///{DEFAULT_DATABASE_PATH.resolve().as_posix()}"


def get_artifact_root():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    configured = os.getenv("MLFLOW_ARTIFACT_ROOT", "").strip()
    return configured or DEFAULT_ARTIFACT_PATH.resolve().as_uri()


def configure_mlflow(experiment_name=LIVE_EXPERIMENT_NAME):
    global _configured_experiment

    if not mlflow_enabled():
        return None

    import mlflow

    if _configured_experiment == experiment_name and mlflow.get_tracking_uri() == get_tracking_uri():
        return mlflow.get_experiment_by_name(experiment_name)

    DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(get_tracking_uri())

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=get_artifact_root(),
        )
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name)
    _configured_experiment = experiment_name
    return experiment


def _useful_attributes(output):
    if not isinstance(output, dict):
        return {}

    keys = [
        "selectedAgent",
        "selected_agent",
        "mode",
        "confidence",
        "model",
        "llmProvider",
        "llmFallbackUsed",
        "routingMethod",
        "routerProvider",
        "needsExternalResearch",
        "collection",
        "action",
        "status",
    ]
    return {
        f"samsung.{key}": output[key]
        for key in keys
        if key in output and isinstance(output[key], (str, int, float, bool))
    }


def mlflow_span(name, span_type="CHAIN", attributes=None):
    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not mlflow_enabled():
                return func(*args, **kwargs)

            import mlflow

            configure_mlflow(LIVE_EXPERIMENT_NAME)
            trace_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in {"langsmith_extra", "mlflow_extra"}
            }
            bound = signature.bind_partial(*args, **trace_kwargs)
            safe_inputs = summarize_trace_value(bound.arguments)

            with mlflow.start_span(
                name=name,
                span_type=span_type,
                attributes={"samsung.component": name, **(attributes or {})},
            ) as span:
                span.set_inputs(safe_inputs)
                try:
                    output = func(*args, **kwargs)
                except Exception as error:
                    span.record_exception(error)
                    span.set_status("ERROR")
                    raise

                span.set_outputs(summarize_trace_value(output))
                span.set_attributes(_useful_attributes(output))
                span.set_status("OK")
                return output

        return wrapper

    return decorator


def flush_mlflow_traces():
    if not mlflow_enabled():
        return

    import mlflow

    mlflow.flush_trace_async_logging(terminate=True)
    mlflow.flush_async_logging()


def mlflow_status():
    experiment = configure_mlflow(LIVE_EXPERIMENT_NAME)
    return {
        "enabled": mlflow_enabled(),
        "tracking_uri": get_tracking_uri(),
        "experiment": LIVE_EXPERIMENT_NAME,
        "experiment_id": experiment.experiment_id if experiment else None,
    }


def get_active_mlflow_trace_id():
    if not mlflow_enabled():
        return None

    import mlflow

    return mlflow.get_active_trace_id()
