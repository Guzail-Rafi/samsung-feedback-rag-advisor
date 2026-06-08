import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
TRACE_TEXT_LIMIT = 4000
TRACE_LIST_LIMIT = 10

load_dotenv(dotenv_path=ENV_PATH, override=True)


def tracing_enabled():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    tracing_value = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    return tracing_value in {"1", "true", "yes", "on"} and bool(
        os.getenv("LANGSMITH_API_KEY", "").strip()
    )


def trace_project_name():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return os.getenv("LANGSMITH_PROJECT", "youtube-intelligence-rag").strip()


def _summarize_dataframe(value):
    score_columns = [
        column
        for column in [
            "weighted_retrieval_score",
            "strategy_retrieval_score",
            "similarity_score",
            "strategy_similarity_score",
            "goal_relevance_score",
        ]
        if column in value.columns
    ]

    summary = {
        "type": "DataFrame",
        "rows": len(value),
        "columns": value.columns.tolist(),
    }

    if score_columns and not value.empty:
        summary["mean_scores"] = {
            column: round(float(value[column].mean()), 4)
            for column in score_columns
        }

    return summary


def summarize_trace_value(value, depth=0):
    if isinstance(value, str):
        if len(value) <= TRACE_TEXT_LIMIT:
            return value
        return value[:TRACE_TEXT_LIMIT] + f"... <{len(value) - TRACE_TEXT_LIMIT} chars omitted>"

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return str(value)

    if depth >= 4:
        return f"<{value.__class__.__name__}>"

    if isinstance(value, pd.DataFrame):
        return _summarize_dataframe(value)

    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "length": len(value),
            "name": value.name,
        }

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, dict):
        return {
            str(key): summarize_trace_value(item, depth + 1)
            for key, item in list(value.items())[:TRACE_LIST_LIMIT]
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        summary = [
            summarize_trace_value(item, depth + 1)
            for item in items[:TRACE_LIST_LIMIT]
        ]
        if len(items) > TRACE_LIST_LIMIT:
            summary.append(f"<{len(items) - TRACE_LIST_LIMIT} items omitted>")
        return summary

    if hasattr(value, "name") and hasattr(value, "count"):
        try:
            return {
                "type": value.__class__.__name__,
                "name": value.name,
                "count": value.count(),
            }
        except Exception:
            pass

    return f"<{value.__class__.__name__}>"


def sanitize_trace_inputs(inputs):
    return summarize_trace_value(inputs)


def sanitize_trace_outputs(outputs):
    return summarize_trace_value(outputs)


def tracing_status():
    return {
        "enabled": tracing_enabled(),
        "project": trace_project_name(),
        "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
    }
