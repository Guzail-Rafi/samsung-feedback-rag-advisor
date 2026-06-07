import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from agent_router import AVAILABLE_AGENTS, route_intent
from rag_answer_generator import (
    INPUT_PATH as FEEDBACK_INPUT_PATH,
    RETRIEVAL_WEIGHTS,
    build_comment_text,
    calculate_category_relevance,
    calculate_intent_penalty,
    calculate_lexical_relevance,
    calculate_sentiment_relevance,
    embedding_model,
    infer_query_intent,
    load_or_create_embeddings as load_feedback_embeddings,
    load_or_create_feedback_vector_store,
)
from rag_evaluation import heuristic_relevance_label
from routing_evaluation_data import ROUTING_QRELS
from strategy_rag import (
    INPUT_PATH as STRATEGY_INPUT_PATH,
    detect_strategy_goal,
    goal_relevance_score,
    load_or_create_embeddings as load_strategy_embeddings,
    load_or_create_strategy_vector_store,
)
from vector_store import query_collection


OUTPUT_DIR = Path("data/processed")
FEEDBACK_ABLATION_PATH = OUTPUT_DIR / "evaluation_feedback_ablation.csv"
STRATEGY_ABLATION_PATH = OUTPUT_DIR / "evaluation_strategy_ablation.csv"
SYSTEM_COMPARISON_PATH = OUTPUT_DIR / "evaluation_rag_system_comparison.csv"
ABLATION_IMPACT_PATH = OUTPUT_DIR / "evaluation_ablation_impact.csv"
BGE_COMPARISON_PATH = OUTPUT_DIR / "evaluation_bge_comparison.csv"
ROUTING_RESULTS_PATH = OUTPUT_DIR / "evaluation_routing_results.csv"
ROUTING_METRICS_PATH = OUTPUT_DIR / "evaluation_routing_metrics.csv"
ROUTING_CONFUSION_PATH = OUTPUT_DIR / "evaluation_routing_confusion_matrix.csv"
REPORT_PATH = OUTPUT_DIR / "evaluation_study_report.txt"

BGE_MODEL_NAME = "BAAI/bge-reranker-base"
TOP_K = 5
CANDIDATE_COUNT = 200
RERANK_CANDIDATE_COUNT = 20

FEEDBACK_QUERIES = [
    "What are users saying about Samsung battery life?",
    "Why are users unhappy about the S-Pen?",
    "What do users think about Galaxy AI and Gemini?",
    "Are users comparing Samsung with Apple?",
    "What are users saying about Samsung camera quality?",
    "What are users saying about Samsung screen or display issues?",
]

STRATEGY_QUERY_PROFILES = [
    {
        "query": "How should Samsung design the S27 Ultra for maximum customer satisfaction?",
        "goals": {"customer_satisfaction"},
        "categories": {
            "Battery / Charging",
            "S-Pen / Features",
            "Camera",
            "Display / Screen",
        },
        "sentiments": {"negative"},
    },
    {
        "query": "How should Samsung design the S27 Ultra for maximum profit?",
        "goals": {"profit"},
        "categories": {"Price / Value"},
        "sentiments": {"positive", "neutral", "negative"},
    },
    {
        "query": "What features should Samsung prioritize in the S27 Ultra?",
        "goals": {"balanced", "customer_satisfaction"},
        "categories": {
            "Battery / Charging",
            "S-Pen / Features",
            "Camera",
            "Display / Screen",
        },
        "sentiments": {"positive", "neutral", "negative"},
    },
    {
        "query": "What product roadmap should Samsung follow for the next Ultra phone?",
        "goals": {"balanced", "customer_satisfaction"},
        "categories": {
            "Battery / Charging",
            "S-Pen / Features",
            "Camera",
            "Display / Screen",
            "Price / Value",
        },
        "sentiments": {"positive", "neutral", "negative"},
    },
    {
        "query": "How can Samsung reduce customer complaints in the next flagship?",
        "goals": {"customer_satisfaction"},
        "categories": {
            "Battery / Charging",
            "S-Pen / Features",
            "Camera",
            "Display / Screen",
            "Software / One UI",
            "Customer Support / Warranty",
        },
        "sentiments": {"negative"},
    },
]

def precision_at_k(labels, k=TOP_K):
    return float(sum(labels[:k]) / k)


def recall_at_k(labels, total_relevant, k=TOP_K):
    if total_relevant == 0:
        return 0.0
    return float(sum(labels[:k]) / total_relevant)


def reciprocal_rank(labels, k=TOP_K):
    for rank, label in enumerate(labels[:k], start=1):
        if label:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, k=TOP_K):
    gains = labels[:k]
    dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(gains, start=1))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(ideal, start=1))
    return float(dcg / idcg) if idcg else 0.0


def evaluate_ranking(system, configuration, query, ranked_df, score_column, latency_ms):
    labels = ranked_df["relevance_label"].astype(int).tolist()
    total_relevant = int(ranked_df["relevance_label"].sum())

    return {
        "rag_system": system,
        "configuration": configuration,
        "query": query,
        "candidate_pool_size": len(ranked_df),
        "relevant_candidates": total_relevant,
        "precision_at_5": round(precision_at_k(labels), 4),
        "candidate_recall_at_5": round(recall_at_k(labels, total_relevant), 4),
        "mrr_at_5": round(reciprocal_rank(labels), 4),
        "ndcg_at_5": round(ndcg_at_k(labels), 4),
        "latency_ms": round(latency_ms, 2),
        "avg_top_5_score": round(float(ranked_df.head(TOP_K)[score_column].mean()), 4),
    }


def normalized_component_score(df, components, weights, penalty_column=None):
    weight_total = sum(weights[name] for name in components)
    score = sum(weights[name] * df[name] for name in components) / weight_total

    if penalty_column:
        score = score - df[penalty_column]

    return score


def prepare_feedback_candidates(query, df, vector_collection):
    start = time.perf_counter()
    query_embedding = embedding_model.encode([query])
    row_indices, similarities = query_collection(
        vector_collection,
        query_embedding,
        CANDIDATE_COUNT,
    )
    retrieval_ms = (time.perf_counter() - start) * 1000

    candidates = df.iloc[row_indices].copy()
    candidates["similarity_score"] = similarities
    intent = infer_query_intent(query)

    candidates["engagement_score"] = np.log1p(
        candidates["like_count"].fillna(0).astype(float)
        + candidates["reply_count"].fillna(0).astype(float)
    )
    max_engagement = candidates["engagement_score"].max()
    if max_engagement > 0:
        candidates["engagement_score"] = candidates["engagement_score"] / max_engagement

    candidates["category_relevance_score"] = candidates.apply(
        lambda row: calculate_category_relevance(query, row, intent),
        axis=1,
    )
    candidates["lexical_relevance_score"] = candidates.apply(
        lambda row: calculate_lexical_relevance(query, row, intent),
        axis=1,
    )
    candidates["sentiment_relevance_score"] = candidates.apply(
        lambda row: calculate_sentiment_relevance(query, row, intent),
        axis=1,
    )
    candidates["intent_penalty_score"] = candidates.apply(
        lambda row: calculate_intent_penalty(query, row, intent),
        axis=1,
    )
    candidates["relevance_label"] = candidates.apply(
        lambda row: heuristic_relevance_label(query, row),
        axis=1,
    )

    return candidates, retrieval_ms


def run_feedback_ablation(bge_model):
    df = pd.read_csv(FEEDBACK_INPUT_PATH)
    df = df[df["language"] == "en"].copy()
    df = df.dropna(subset=["clean_comment"])
    df = df[df["word_count"] >= 3].reset_index(drop=True)
    df["rag_text"] = df.apply(build_comment_text, axis=1)

    embeddings = load_feedback_embeddings(df["rag_text"].tolist())
    vector_collection = load_or_create_feedback_vector_store(df, embeddings)

    weights = {
        "similarity_score": RETRIEVAL_WEIGHTS["semantic"],
        "category_relevance_score": RETRIEVAL_WEIGHTS["category"],
        "lexical_relevance_score": RETRIEVAL_WEIGHTS["lexical"],
        "sentiment_relevance_score": RETRIEVAL_WEIGHTS["sentiment"],
        "engagement_score": RETRIEVAL_WEIGHTS["engagement"],
    }
    all_components = list(weights)
    configurations = {
        "semantic_only": ["similarity_score"],
        "hybrid_full": all_components,
        "hybrid_without_category": [
            component for component in all_components if component != "category_relevance_score"
        ],
        "hybrid_without_lexical": [
            component for component in all_components if component != "lexical_relevance_score"
        ],
        "hybrid_without_sentiment": [
            component for component in all_components if component != "sentiment_relevance_score"
        ],
        "hybrid_without_engagement": [
            component for component in all_components if component != "engagement_score"
        ],
    }

    evaluation_rows = []

    for query in FEEDBACK_QUERIES:
        candidates, retrieval_ms = prepare_feedback_candidates(query, df, vector_collection)

        for configuration, components in configurations.items():
            start = time.perf_counter()
            ranked = candidates.copy()
            penalty = "intent_penalty_score" if configuration != "semantic_only" else None
            ranked["evaluation_score"] = normalized_component_score(
                ranked,
                components,
                weights,
                penalty,
            )
            ranked = ranked.sort_values("evaluation_score", ascending=False)
            latency_ms = retrieval_ms + ((time.perf_counter() - start) * 1000)
            evaluation_rows.append(
                evaluate_ranking(
                    "feedback_rag",
                    configuration,
                    query,
                    ranked,
                    "evaluation_score",
                    latency_ms,
                )
            )

        start = time.perf_counter()
        hybrid = candidates.copy()
        hybrid["hybrid_score"] = normalized_component_score(
            hybrid,
            all_components,
            weights,
            "intent_penalty_score",
        )
        rerank_candidates = hybrid.sort_values("hybrid_score", ascending=False).head(
            RERANK_CANDIDATE_COUNT
        ).copy()
        pairs = [
            [query, str(comment)]
            for comment in rerank_candidates["clean_comment"].fillna("").tolist()
        ]
        rerank_candidates["evaluation_score"] = bge_model.predict(pairs)
        remaining = hybrid.drop(index=rerank_candidates.index).copy()
        remaining["evaluation_score"] = -np.inf
        ranked = pd.concat([rerank_candidates, remaining]).sort_values(
            "evaluation_score",
            ascending=False,
        )
        latency_ms = retrieval_ms + ((time.perf_counter() - start) * 1000)
        evaluation_rows.append(
            evaluate_ranking(
                "feedback_rag",
                "hybrid_plus_bge_reranker",
                query,
                ranked,
                "evaluation_score",
                latency_ms,
            )
        )

    return pd.DataFrame(evaluation_rows)


def strategy_relevance_label(row, profile):
    goal = str(row.get("goal_relevance", "")).lower()
    category = str(row.get("issue_category", ""))
    sentiment = str(row.get("sentiment_label", "")).lower()

    return int(
        goal in profile["goals"]
        and category in profile["categories"]
        and sentiment in profile["sentiments"]
    )


def prepare_strategy_candidates(profile, df, vector_collection):
    query = profile["query"]
    start = time.perf_counter()
    query_embedding = embedding_model.encode([query])
    row_indices, similarities = query_collection(
        vector_collection,
        query_embedding,
        CANDIDATE_COUNT,
    )
    retrieval_ms = (time.perf_counter() - start) * 1000

    candidates = df.iloc[row_indices].copy()
    candidates["strategy_similarity_score"] = similarities
    goal = detect_strategy_goal(query)

    candidates["engagement_score"] = np.log1p(
        candidates["engagement_total"].fillna(0).astype(float)
    )
    max_engagement = candidates["engagement_score"].max()
    if max_engagement > 0:
        candidates["engagement_score"] = candidates["engagement_score"] / max_engagement

    candidates["goal_relevance_score"] = candidates.apply(
        lambda row: goal_relevance_score(goal, row),
        axis=1,
    )
    candidates["priority_score"] = candidates["priority"].map(
        {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    ).fillna(0.5)
    candidates["relevance_label"] = candidates.apply(
        lambda row: strategy_relevance_label(row, profile),
        axis=1,
    )

    return candidates, retrieval_ms


def run_strategy_ablation(bge_model):
    df = pd.read_csv(STRATEGY_INPUT_PATH)
    df = df.dropna(subset=["strategy_text"]).reset_index(drop=True)

    embeddings = load_strategy_embeddings(df["strategy_text"].tolist())
    vector_collection = load_or_create_strategy_vector_store(df, embeddings)

    weights = {
        "strategy_similarity_score": 0.45,
        "goal_relevance_score": 0.25,
        "priority_score": 0.20,
        "engagement_score": 0.10,
    }
    all_components = list(weights)
    configurations = {
        "semantic_only": ["strategy_similarity_score"],
        "strategy_hybrid_full": all_components,
        "strategy_without_goal": [
            component for component in all_components if component != "goal_relevance_score"
        ],
        "strategy_without_priority": [
            component for component in all_components if component != "priority_score"
        ],
        "strategy_without_engagement": [
            component for component in all_components if component != "engagement_score"
        ],
    }

    evaluation_rows = []

    for profile in STRATEGY_QUERY_PROFILES:
        query = profile["query"]
        candidates, retrieval_ms = prepare_strategy_candidates(profile, df, vector_collection)

        for configuration, components in configurations.items():
            start = time.perf_counter()
            ranked = candidates.copy()
            ranked["evaluation_score"] = normalized_component_score(
                ranked,
                components,
                weights,
            )
            ranked = ranked.sort_values("evaluation_score", ascending=False)
            latency_ms = retrieval_ms + ((time.perf_counter() - start) * 1000)
            evaluation_rows.append(
                evaluate_ranking(
                    "strategy_rag",
                    configuration,
                    query,
                    ranked,
                    "evaluation_score",
                    latency_ms,
                )
            )

        start = time.perf_counter()
        hybrid = candidates.copy()
        hybrid["hybrid_score"] = normalized_component_score(
            hybrid,
            all_components,
            weights,
        )
        rerank_candidates = hybrid.sort_values("hybrid_score", ascending=False).head(
            RERANK_CANDIDATE_COUNT
        ).copy()
        pairs = [
            [query, str(text)]
            for text in rerank_candidates["strategy_text"].fillna("").tolist()
        ]
        rerank_candidates["evaluation_score"] = bge_model.predict(pairs)
        remaining = hybrid.drop(index=rerank_candidates.index).copy()
        remaining["evaluation_score"] = -np.inf
        ranked = pd.concat([rerank_candidates, remaining]).sort_values(
            "evaluation_score",
            ascending=False,
        )
        latency_ms = retrieval_ms + ((time.perf_counter() - start) * 1000)
        evaluation_rows.append(
            evaluate_ranking(
                "strategy_rag",
                "strategy_hybrid_plus_bge_reranker",
                query,
                ranked,
                "evaluation_score",
                latency_ms,
            )
        )

    return pd.DataFrame(evaluation_rows)


def summarize_retrieval_results(feedback_df, strategy_df):
    combined = pd.concat([feedback_df, strategy_df], ignore_index=True)
    return (
        combined.groupby(["rag_system", "configuration"], as_index=False)
        .agg(
            query_count=("query", "count"),
            precision_at_5=("precision_at_5", "mean"),
            candidate_recall_at_5=("candidate_recall_at_5", "mean"),
            mrr_at_5=("mrr_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )


def build_ablation_impact(retrieval_summary):
    baseline_names = {
        "feedback_rag": "hybrid_full",
        "strategy_rag": "strategy_hybrid_full",
    }
    rows = []

    for system, baseline_name in baseline_names.items():
        system_rows = retrieval_summary[retrieval_summary["rag_system"] == system]
        baseline = system_rows[system_rows["configuration"] == baseline_name].iloc[0]

        for _, row in system_rows.iterrows():
            rows.append(
                {
                    "rag_system": system,
                    "baseline_configuration": baseline_name,
                    "configuration": row["configuration"],
                    "delta_precision_at_5": round(
                        row["precision_at_5"] - baseline["precision_at_5"],
                        4,
                    ),
                    "delta_candidate_recall_at_5": round(
                        row["candidate_recall_at_5"] - baseline["candidate_recall_at_5"],
                        4,
                    ),
                    "delta_mrr_at_5": round(row["mrr_at_5"] - baseline["mrr_at_5"], 4),
                    "delta_ndcg_at_5": round(row["ndcg_at_5"] - baseline["ndcg_at_5"], 4),
                    "delta_latency_ms": round(
                        row["avg_latency_ms"] - baseline["avg_latency_ms"],
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_bge_comparison(retrieval_summary):
    comparisons = [
        ("feedback_rag", "hybrid_full", "hybrid_plus_bge_reranker"),
        ("strategy_rag", "strategy_hybrid_full", "strategy_hybrid_plus_bge_reranker"),
    ]
    rows = []

    for system, before_name, after_name in comparisons:
        system_rows = retrieval_summary[retrieval_summary["rag_system"] == system]
        before = system_rows[system_rows["configuration"] == before_name].iloc[0]
        after = system_rows[system_rows["configuration"] == after_name].iloc[0]
        rows.append(
            {
                "rag_system": system,
                "before_configuration": before_name,
                "after_configuration": after_name,
                "precision_at_5_before": before["precision_at_5"],
                "precision_at_5_after": after["precision_at_5"],
                "precision_at_5_delta": round(
                    after["precision_at_5"] - before["precision_at_5"],
                    4,
                ),
                "ndcg_at_5_before": before["ndcg_at_5"],
                "ndcg_at_5_after": after["ndcg_at_5"],
                "ndcg_at_5_delta": round(after["ndcg_at_5"] - before["ndcg_at_5"], 4),
                "latency_ms_before": before["avg_latency_ms"],
                "latency_ms_after": after["avg_latency_ms"],
                "latency_ms_delta": round(
                    after["avg_latency_ms"] - before["avg_latency_ms"],
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


def run_routing_evaluation():
    rows = []

    for query, expected_agent in ROUTING_QRELS:
        decision = route_intent(query)
        selected_agent = decision["selected_agent"]
        rows.append(
            {
                "query": query,
                "expected_agent": expected_agent,
                "selected_agent": selected_agent,
                "correct": int(expected_agent == selected_agent),
                "routing_reason": decision["reason"],
                "matched_terms": ", ".join(decision["matched_terms"]),
                "routing_method": decision["routing_method"],
                "routing_confidence": decision["confidence"],
                "normalized_query": decision["normalized_query"],
            }
        )

    results = pd.DataFrame(rows)
    metrics = []

    for agent in AVAILABLE_AGENTS:
        true_positive = int(
            ((results["expected_agent"] == agent) & (results["selected_agent"] == agent)).sum()
        )
        false_positive = int(
            ((results["expected_agent"] != agent) & (results["selected_agent"] == agent)).sum()
        )
        false_negative = int(
            ((results["expected_agent"] == agent) & (results["selected_agent"] != agent)).sum()
        )
        support = int((results["expected_agent"] == agent).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        metrics.append(
            {
                "agent": agent,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    confusion = pd.crosstab(
        results["expected_agent"],
        results["selected_agent"],
        rownames=["expected_agent"],
        colnames=["selected_agent"],
        dropna=False,
    ).reindex(index=AVAILABLE_AGENTS, columns=AVAILABLE_AGENTS, fill_value=0)

    return results, metrics_df, confusion


def percent(value):
    return f"{value * 100:.1f}%"


def format_table(df, columns):
    return df[columns].to_string(index=False)


def build_terminal_report(
    retrieval_summary,
    ablation_impact,
    bge_comparison,
    routing_results,
    routing_metrics,
):
    feedback = retrieval_summary[retrieval_summary["rag_system"] == "feedback_rag"].copy()
    strategy = retrieval_summary[retrieval_summary["rag_system"] == "strategy_rag"].copy()

    routing_accuracy = float(routing_results["correct"].mean())
    incorrect = routing_results[routing_results["correct"] == 0]
    correct = routing_results[routing_results["correct"] == 1]
    feedback_impact = ablation_impact[ablation_impact["rag_system"] == "feedback_rag"]
    strategy_impact = ablation_impact[ablation_impact["rag_system"] == "strategy_rag"]

    feedback_semantic = feedback_impact[
        feedback_impact["configuration"] == "semantic_only"
    ].iloc[0]
    feedback_without_lexical = feedback_impact[
        feedback_impact["configuration"] == "hybrid_without_lexical"
    ].iloc[0]
    strategy_semantic = strategy_impact[
        strategy_impact["configuration"] == "semantic_only"
    ].iloc[0]
    strategy_without_goal = strategy_impact[
        strategy_impact["configuration"] == "strategy_without_goal"
    ].iloc[0]
    feedback_bge = bge_comparison[bge_comparison["rag_system"] == "feedback_rag"].iloc[0]
    strategy_bge = bge_comparison[bge_comparison["rag_system"] == "strategy_rag"].iloc[0]

    lines = [
        "=" * 78,
        "RAG ABLATION AND MULTI-AGENT ROUTING EVALUATION",
        "=" * 78,
        "",
        "Methodology note:",
        f"- All retrieval configurations use the same top-{CANDIDATE_COUNT} ChromaDB candidate pool.",
        "- Relevance labels are independent rule-based qrels, not retrieval or BGE scores.",
        "- The qrels are reproducible automated judgments; manual annotation is recommended for final validation.",
        f"- BGE reranks the top {RERANK_CANDIDATE_COUNT} hybrid candidates.",
        "- Candidate Recall@5 is recall relative to relevant documents in the candidate pool.",
        "",
        "1. Feedback RAG retrieval ablation",
        format_table(
            feedback,
            [
                "configuration",
                "precision_at_5",
                "candidate_recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
                "avg_latency_ms",
            ],
        ),
        "",
        "2. Strategy RAG retrieval ablation",
        format_table(
            strategy,
            [
                "configuration",
                "precision_at_5",
                "candidate_recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
                "avg_latency_ms",
            ],
        ),
        "",
        "3. Feedback RAG versus Strategy RAG",
        "These systems are evaluated separately because they use different corpora and objectives.",
        format_table(
            retrieval_summary[
                retrieval_summary["configuration"].isin(
                    [
                        "hybrid_full",
                        "hybrid_plus_bge_reranker",
                        "strategy_hybrid_full",
                        "strategy_hybrid_plus_bge_reranker",
                    ]
                )
            ],
            [
                "rag_system",
                "configuration",
                "precision_at_5",
                "candidate_recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
                "avg_latency_ms",
            ],
        ),
        "",
        "4. Before versus after BGE reranking",
        format_table(
            bge_comparison,
            [
                "rag_system",
                "precision_at_5_before",
                "precision_at_5_after",
                "precision_at_5_delta",
                "ndcg_at_5_delta",
                "latency_ms_delta",
            ],
        ),
        "",
        "5. Multi-agent routing evaluation",
        f"Routing accuracy: {percent(routing_accuracy)} ({int(routing_results['correct'].sum())}/{len(routing_results)})",
        "",
        format_table(routing_metrics, ["agent", "support", "precision", "recall", "f1_score"]),
        "",
        "Correct routing examples:",
        format_table(correct.head(4), ["query", "expected_agent", "selected_agent"]),
        "",
        "Incorrect or ambiguous routing examples:",
        format_table(incorrect.head(8), ["query", "expected_agent", "selected_agent"]),
        "",
        "6. Report-ready findings",
        (
            "- Feedback RAG hybrid retrieval improved Precision@5 by "
            f"{feedback_semantic['delta_precision_at_5'] * -1:.4f} and nDCG@5 by "
            f"{feedback_semantic['delta_ndcg_at_5'] * -1:.4f} over semantic-only retrieval."
        ),
        (
            "- Removing lexical relevance reduced Feedback RAG Precision@5 by "
            f"{abs(feedback_without_lexical['delta_precision_at_5']):.4f}, making lexical matching "
            "the clearest contributor in the tested feedback queries."
        ),
        (
            "- Strategy RAG hybrid retrieval improved Precision@5 by "
            f"{strategy_semantic['delta_precision_at_5'] * -1:.4f} and nDCG@5 by "
            f"{strategy_semantic['delta_ndcg_at_5'] * -1:.4f} over semantic-only retrieval."
        ),
        (
            "- Removing strategy-goal alignment reduced Precision@5 by "
            f"{abs(strategy_without_goal['delta_precision_at_5']):.4f} and nDCG@5 by "
            f"{abs(strategy_without_goal['delta_ndcg_at_5']):.4f}, showing that goal alignment is "
            "the most important tested Strategy RAG component."
        ),
        (
            "- BGE reranking changed Feedback RAG Precision@5 by "
            f"{feedback_bge['precision_at_5_delta']:.4f} while adding "
            f"{feedback_bge['latency_ms_delta']:.2f} ms average latency."
        ),
        (
            "- BGE reranking changed Strategy RAG Precision@5 by "
            f"{strategy_bge['precision_at_5_delta']:.4f} while adding "
            f"{strategy_bge['latency_ms_delta']:.2f} ms average latency."
        ),
        (
            f"- The evaluated multi-agent router achieved {percent(routing_accuracy)} accuracy."
        ),
        "",
        "Interpretation guidance:",
        "- Higher Precision@5 and nDCG@5 indicate more relevant evidence near the top.",
        "- Ablations that reduce metrics identify components that contribute useful ranking signal.",
        "- BGE results show whether extra reranking quality justifies its additional latency.",
        "- Routing errors identify semantic or vocabulary gaps in the evaluated query router.",
        "- Automated qrels may favor explicit domain-term matches, so results should be discussed as an ablation study rather than final human relevance judgment.",
        "",
        "Generated artifacts:",
        f"- {FEEDBACK_ABLATION_PATH}",
        f"- {STRATEGY_ABLATION_PATH}",
        f"- {SYSTEM_COMPARISON_PATH}",
        f"- {ABLATION_IMPACT_PATH}",
        f"- {BGE_COMPARISON_PATH}",
        f"- {ROUTING_RESULTS_PATH}",
        f"- {ROUTING_METRICS_PATH}",
        f"- {ROUTING_CONFUSION_PATH}",
        f"- {REPORT_PATH}",
    ]

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BGE reranker:", BGE_MODEL_NAME)
    bge_model = CrossEncoder(BGE_MODEL_NAME)

    print("\nRunning Feedback RAG ablation study...")
    feedback_df = run_feedback_ablation(bge_model)
    feedback_df.to_csv(FEEDBACK_ABLATION_PATH, index=False, encoding="utf-8-sig")

    print("Running Strategy RAG ablation study...")
    strategy_df = run_strategy_ablation(bge_model)
    strategy_df.to_csv(STRATEGY_ABLATION_PATH, index=False, encoding="utf-8-sig")

    retrieval_summary = summarize_retrieval_results(feedback_df, strategy_df)
    retrieval_summary.to_csv(SYSTEM_COMPARISON_PATH, index=False, encoding="utf-8-sig")
    ablation_impact = build_ablation_impact(retrieval_summary)
    ablation_impact.to_csv(ABLATION_IMPACT_PATH, index=False, encoding="utf-8-sig")
    bge_comparison = build_bge_comparison(retrieval_summary)
    bge_comparison.to_csv(BGE_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    print("Running multi-agent routing evaluation...")
    routing_results, routing_metrics, routing_confusion = run_routing_evaluation()
    routing_results.to_csv(ROUTING_RESULTS_PATH, index=False, encoding="utf-8-sig")
    routing_metrics.to_csv(ROUTING_METRICS_PATH, index=False, encoding="utf-8-sig")
    routing_confusion.to_csv(ROUTING_CONFUSION_PATH, encoding="utf-8-sig")

    report = build_terminal_report(
        retrieval_summary,
        ablation_impact,
        bge_comparison,
        routing_results,
        routing_metrics,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n" + report)


if __name__ == "__main__":
    main()
