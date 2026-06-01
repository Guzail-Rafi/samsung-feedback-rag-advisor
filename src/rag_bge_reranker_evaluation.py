import os

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder


INPUT_PATH = "data/processed/rag_retrieval_results.csv"
OUTPUT_PATH = "data/processed/rag_bge_reranker_evaluation_results.csv"
DETAILED_OUTPUT_PATH = "data/processed/rag_bge_reranker_detailed_results.csv"

BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.01


def sigmoid(scores):
    scores = np.asarray(scores, dtype=float)
    return 1 / (1 + np.exp(-scores))


def relevance_labels(scores, threshold=RELEVANCE_THRESHOLD):
    return [int(score >= threshold) for score in scores]


def precision_at_k(labels, k=TOP_K):
    labels_at_k = labels[:k]

    if len(labels_at_k) == 0:
        return 0

    return sum(labels_at_k) / k


def average_score_if_present(df, column):
    if column not in df.columns:
        return None

    return round(df[column].mean(), 3)


def main():
    df = pd.read_csv(INPUT_PATH)

    if "query" not in df.columns or "clean_comment" not in df.columns:
        raise ValueError("Input CSV must contain query and clean_comment columns.")

    model = CrossEncoder(BGE_RERANKER_MODEL_NAME)

    evaluation_rows = []
    detailed_rows = []

    queries = df["query"].dropna().unique()

    for query in queries:
        query_results = df[df["query"] == query].head(TOP_K).copy()

        if len(query_results) < TOP_K:
            print(f"Warning: Less than {TOP_K} results found for query: {query}")
            continue

        pairs = [
            [query, str(comment)]
            for comment in query_results["clean_comment"].fillna("").tolist()
        ]

        raw_scores = model.predict(pairs)

        # Use raw BGE reranker scores directly
        reranker_scores = np.asarray(raw_scores, dtype=float)

        labels = relevance_labels(reranker_scores)
        p_at_5 = precision_at_k(labels, k=TOP_K)

        query_results.insert(0, "retrieval_rank", range(1, len(query_results) + 1))
        query_results["bge_reranker_raw_score"] = raw_scores
        query_results["bge_reranker_score"] = reranker_scores
        query_results["bge_reranker_relevance_label"] = labels

        evaluation_rows.append({
            "query": query,
            "evaluation_method": "bge_reranker_precision_at_5",
            "reranker_model": BGE_RERANKER_MODEL_NAME,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "bge_reranker_labels_top_5": labels,
            "relevant_results": sum(labels),
            "total_results": TOP_K,
            "precision_at_5": round(p_at_5, 2),
            "avg_bge_reranker_score": round(query_results["bge_reranker_score"].mean(), 3),
            "min_bge_reranker_score": round(query_results["bge_reranker_score"].min(), 3),
            "max_bge_reranker_score": round(query_results["bge_reranker_score"].max(), 3),
            "avg_weighted_retrieval_score": average_score_if_present(
                query_results,
                "weighted_retrieval_score",
            ),
            "avg_similarity_score": average_score_if_present(
                query_results,
                "similarity_score",
            ),
            "avg_category_relevance_score": average_score_if_present(
                query_results,
                "category_relevance_score",
            ),
            "avg_lexical_relevance_score": average_score_if_present(
                query_results,
                "lexical_relevance_score",
            ),
        })

        detailed_rows.append(query_results)

    eval_df = pd.DataFrame(evaluation_rows)
    detailed_df = (
        pd.concat(detailed_rows, ignore_index=True)
        if detailed_rows
        else pd.DataFrame()
    )

    os.makedirs("data/processed", exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(DETAILED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("BGE reranker RAG evaluation completed!")
    print("Saved summary to:", OUTPUT_PATH)
    print("Saved detailed results to:", DETAILED_OUTPUT_PATH)

    if eval_df.empty:
        print("No evaluation rows were created.")
        return

    print("\nEvaluation results:")
    print(eval_df)

    print("\nOverall BGE Reranker Precision@5:")
    print(round(eval_df["precision_at_5"].mean(), 2))


if __name__ == "__main__":
    main()
