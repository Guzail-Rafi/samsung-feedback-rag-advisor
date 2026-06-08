import os
import pandas as pd
import mlflow

from mlflow_tracing import (
    PIPELINE_EXPERIMENT_NAME,
    configure_mlflow,
    flush_mlflow_traces,
    get_tracking_uri,
)
from openai_client import (
    get_llama_model,
    get_openai_model,
    llama_fallback_enabled,
)


# =========================
# FILE PATHS
# =========================

COMMENTS_PATH = "data/processed/comments_with_ner.csv"
RAG_EVAL_PATH = "data/processed/rag_evaluation_results.csv"
BGE_EVAL_PATH = "data/processed/rag_bge_reranker_evaluation_results.csv"
RAG_ANSWERS_PATH = "data/processed/rag_answers.csv"
SUMMARIES_PATH = "data/processed/llm_summaries.csv"
AGENT_RESULTS_PATH = "data/processed/agent_router_results.csv"

EXPERIMENT_NAME = PIPELINE_EXPERIMENT_NAME


# =========================
# SAFE HELPERS
# =========================

def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: File not found: {path}")
    return pd.DataFrame()


def log_count_metrics(df, column_name, prefix, top_n=10):
    """
    Logs top category counts as MLflow metrics.
    Example:
    sentiment_positive = 14769
    issue_AI_Gemini = 4269
    """

    if df.empty or column_name not in df.columns:
        return

    counts = df[column_name].value_counts().head(top_n)

    for label, count in counts.items():
        clean_label = (
            str(label)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(",", "")
            .replace("__", "_")
        )

        metric_name = f"{prefix}_{clean_label}"
        mlflow.log_metric(metric_name, int(count))


def log_artifact_if_exists(path):
    if os.path.exists(path):
        mlflow.log_artifact(path)


# =========================
# MAIN MONITORING
# =========================

def main():
    configure_mlflow(EXPERIMENT_NAME)

    comments_df = safe_read_csv(COMMENTS_PATH)
    rag_eval_df = safe_read_csv(RAG_EVAL_PATH)
    bge_eval_df = safe_read_csv(BGE_EVAL_PATH)
    rag_answers_df = safe_read_csv(RAG_ANSWERS_PATH)
    summaries_df = safe_read_csv(SUMMARIES_PATH)
    agent_df = safe_read_csv(AGENT_RESULTS_PATH)

    with mlflow.start_run(
        run_name="Samsung_YouTube_RAG_Pipeline_Run",
        tags={"system": "samsung-rag", "run_type": "batch-monitoring"},
    ):

        # =========================
        # PARAMETERS
        # =========================

        mlflow.log_param("data_source", "YouTube Data API")
        mlflow.log_param("project_topic", "Samsung Galaxy User Feedback Analysis")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_primary_provider", "openai")
        mlflow.log_param("openai_model", get_openai_model())
        mlflow.log_param("llm_fallback_enabled", llama_fallback_enabled())
        mlflow.log_param("llm_fallback_provider", "ollama")
        mlflow.log_param("llm_fallback_model", get_llama_model())
        mlflow.log_param("reranker_model", "BAAI/bge-reranker-base")
        mlflow.log_param("rag_top_k", 5)

        # =========================
        # DATASET METRICS
        # =========================

        if not comments_df.empty:
            mlflow.log_metric("total_processed_comments", len(comments_df))

            if "video_id" in comments_df.columns:
                mlflow.log_metric("unique_videos", comments_df["video_id"].nunique())

            if "language" in comments_df.columns:
                english_count = len(comments_df[comments_df["language"] == "en"])
                mlflow.log_metric("english_comments", english_count)

            if "word_count" in comments_df.columns:
                mlflow.log_metric("avg_word_count", round(comments_df["word_count"].mean(), 3))
                mlflow.log_metric("max_word_count", int(comments_df["word_count"].max()))

            if "like_count" in comments_df.columns:
                mlflow.log_metric("avg_comment_likes", round(comments_df["like_count"].mean(), 3))

            if "reply_count" in comments_df.columns:
                mlflow.log_metric("avg_reply_count", round(comments_df["reply_count"].mean(), 3))

            # Distribution metrics
            log_count_metrics(comments_df, "sentiment_label", "sentiment", top_n=10)
            log_count_metrics(comments_df, "issue_category", "issue", top_n=15)
            log_count_metrics(comments_df, "topic_name", "topic", top_n=15)

        # =========================
        # RAG EVALUATION METRICS
        # =========================

        if not rag_eval_df.empty:
            if "precision_at_5" in rag_eval_df.columns:
                mlflow.log_metric(
                    "manual_checked_precision_at_5",
                    round(rag_eval_df["precision_at_5"].mean(), 3)
                )

            if "avg_weighted_score" in rag_eval_df.columns:
                mlflow.log_metric(
                    "avg_rag_weighted_score",
                    round(rag_eval_df["avg_weighted_score"].mean(), 3)
                )

            if "avg_similarity_score" in rag_eval_df.columns:
                mlflow.log_metric(
                    "avg_rag_similarity_score",
                    round(rag_eval_df["avg_similarity_score"].mean(), 3)
                )

            if "avg_category_relevance_score" in rag_eval_df.columns:
                mlflow.log_metric(
                    "avg_category_relevance_score",
                    round(rag_eval_df["avg_category_relevance_score"].mean(), 3)
                )

            if "avg_lexical_relevance_score" in rag_eval_df.columns:
                mlflow.log_metric(
                    "avg_lexical_relevance_score",
                    round(rag_eval_df["avg_lexical_relevance_score"].mean(), 3)
                )

        # =========================
        # BGE EVALUATION METRICS
        # =========================

        if not bge_eval_df.empty:
            if "precision_at_5" in bge_eval_df.columns:
                mlflow.log_metric(
                    "bge_reranker_precision_at_5",
                    round(bge_eval_df["precision_at_5"].mean(), 3)
                )

            if "avg_bge_reranker_score" in bge_eval_df.columns:
                mlflow.log_metric(
                    "avg_bge_reranker_score",
                    round(bge_eval_df["avg_bge_reranker_score"].mean(), 3)
                )

        # =========================
        # LLM OUTPUT METRICS
        # =========================

        if not rag_answers_df.empty:
            mlflow.log_metric("rag_answers_generated", len(rag_answers_df))

        if not summaries_df.empty:
            mlflow.log_metric("llm_summaries_generated", len(summaries_df))

        if not agent_df.empty:
            mlflow.log_metric("agent_router_test_queries", len(agent_df))

            if "selected_agent" in agent_df.columns:
                log_count_metrics(agent_df, "selected_agent", "agent", top_n=10)

        # =========================
        # LOG ARTIFACTS
        # =========================

        artifact_files = [
            COMMENTS_PATH,
            RAG_EVAL_PATH,
            BGE_EVAL_PATH,
            RAG_ANSWERS_PATH,
            SUMMARIES_PATH,
            AGENT_RESULTS_PATH,
            "data/processed/top_keywords_overall.csv",
            "data/processed/top_negative_keywords.csv",
            "data/processed/top_keywords_by_category.csv",
            "data/processed/topic_keywords.csv",
            "data/processed/comments_with_sentiment_topics.csv",
            "data/processed/topic_keywords_by_sentiment.csv",
            "data/processed/rag_retrieval_results.csv",
        ]

        for path in artifact_files:
            log_artifact_if_exists(path)

        print("MLflow monitoring completed!")
        print("Experiment:", EXPERIMENT_NAME)
        print("Tracking URI:", get_tracking_uri())

        if not rag_eval_df.empty and "precision_at_5" in rag_eval_df.columns:
            print("Manual/checked Precision@5:", round(rag_eval_df["precision_at_5"].mean(), 3))

        if not bge_eval_df.empty and "precision_at_5" in bge_eval_df.columns:
            print("BGE Reranker Precision@5:", round(bge_eval_df["precision_at_5"].mean(), 3))


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_mlflow_traces()
