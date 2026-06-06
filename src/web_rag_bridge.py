import contextlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(PROJECT_ROOT)


STRATEGY_TERMS = [
    "s27",
    "roadmap",
    "strategy",
    "design",
    "profit",
    "revenue",
    "sales",
    "satisfaction",
    "product plan",
    "prioritize",
    "next flagship",
    "next ultra",
    "reduce complaints",
    "what should samsung do",
    "how should samsung",
]

FOLLOW_UP_STARTERS = [
    "what about",
    "and",
    "also",
    "same",
    "then",
    "compare",
    "continue",
    "expand",
    "explain more",
    "why",
    "how about",
]


def detect_rag_mode(query):
    query_lower = query.lower()
    if any(term in query_lower for term in STRATEGY_TERMS):
        return "strategy"
    return "feedback"


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def get_model_name():
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import get_openai_model

        return get_openai_model()


def normalize_messages(messages):
    normalized = []

    for message in messages or []:
        role = clean_value(message.get("role")).strip()
        content = clean_value(message.get("content")).strip()

        if role in {"assistant", "user"} and content:
            normalized.append({"role": role, "content": content})

    return normalized[-10:]


def conversation_text(messages):
    if not messages:
        return "No prior conversation."

    lines = []
    for message in messages[-8:]:
        role = "User" if message["role"] == "user" else "Assistant"
        content = message["content"].replace("\n", " ").strip()
        lines.append(f"{role}: {content[:700]}")

    return "\n".join(lines)


def is_follow_up(query):
    query_lower = query.lower().strip()
    words = query_lower.split()

    if len(words) <= 5:
        return True

    return any(query_lower.startswith(starter) for starter in FOLLOW_UP_STARTERS)


def build_memory_query(query, messages):
    previous_user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user" and message["content"].strip().lower() != query.lower()
    ]

    if not previous_user_messages:
        return query

    recent_user_context = " ".join(previous_user_messages[-3:])

    if is_follow_up(query):
        return f"Follow-up question: {query}\nCurrent focus: {query}\nPrevious conversation context: {recent_user_context}"

    return f"{query}\nRecent conversation context: {recent_user_context}"


def generate_feedback_answer(query, retrieval_query, evidence_text, confidence, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are an academic NLP assistant for a university conversational RAG project.

You must answer using only:
1. The retrieved YouTube comment evidence.
2. The conversation history only for interpreting follow-up questions.

Do not invent facts.
Do not overgeneralize beyond the retrieved evidence.
If the evidence is mixed, clearly say it is mixed.
If the user's question is a follow-up, connect it to the previous turn.
Keep the answer concise, balanced, and professional.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Original user question:
{query}

Contextual retrieval query used for RAG:
{retrieval_query}

Retrieved YouTube Comment Evidence:
{evidence_text}

RAG Confidence:
{confidence}

Write:
1. A direct answer based only on the retrieved evidence.
2. Key evidence-based points.
3. A short confidence explanation.
4. If this was a follow-up, mention how it relates to the previous topic.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_completion_tokens=850,
    )


def generate_strategy_answer_with_memory(query, retrieval_query, goal, evidence_text, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are a product strategy advisor for a university conversational Strategy RAG project.

You must answer only using the retrieved strategy evidence.
Use conversation history only to interpret follow-up questions or negotiation context.
Do not invent unsupported product claims.
Clearly separate customer satisfaction logic from profit logic when needed.
Use professional business language.
Use phases such as Phase 1, Phase 2, Phase 3, and Phase 4 instead of exact calendar quarters unless timing evidence is provided.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Original user strategy question:
{query}

Contextual retrieval query used for Strategy RAG:
{retrieval_query}

Detected strategy goal:
{goal}

Retrieved Strategy Evidence:
{evidence_text}

Write:
1. Direct strategic recommendation.
2. Top feature/product priorities.
3. Reasoning from retrieved customer feedback evidence.
4. Expected impact on customer satisfaction or profit.
5. Risks or trade-offs.
6. Updated phased product roadmap.
7. If this is a follow-up or negotiation, explain what changed from the prior context.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_completion_tokens=1100,
    )


def feedback_evidence_rows(results):
    rows = []

    for _, row in results.head(5).iterrows():
        rows.append(
            {
                "comment": clean_value(row.get("clean_comment")),
                "sentiment": clean_value(row.get("sentiment_label")),
                "issue_category": clean_value(row.get("issue_category")),
                "topic": clean_value(row.get("topic_name")),
                "video": clean_value(row.get("video_title")),
                "weighted_score": float(row.get("weighted_retrieval_score", 0)),
                "similarity_score": float(row.get("similarity_score", 0)),
            }
        )

    return rows


def strategy_evidence_rows(results):
    rows = []

    for _, row in results.head(5).iterrows():
        rows.append(
            {
                "comment": clean_value(row.get("clean_comment")),
                "sentiment": clean_value(row.get("sentiment_label")),
                "issue_category": clean_value(row.get("issue_category")),
                "customer_signal": clean_value(row.get("customer_signal")),
                "customer_recommendation": clean_value(row.get("customer_recommendation")),
                "profit_recommendation": clean_value(row.get("profit_recommendation")),
                "business_impact": clean_value(row.get("business_impact")),
                "priority": clean_value(row.get("priority")),
                "strategy_score": float(row.get("strategy_retrieval_score", 0)),
                "goal_relevance_score": float(row.get("goal_relevance_score", 0)),
            }
        )

    return rows


def run_feedback_rag(query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from rag_answer_generator import (
            INPUT_PATH,
            RETRIEVAL_WEIGHTS,
            build_comment_text,
            calculate_category_relevance,
            calculate_rag_confidence,
            calculate_intent_penalty,
            calculate_lexical_relevance,
            calculate_sentiment_relevance,
            embedding_model,
            format_evidence,
            infer_query_intent,
            load_or_create_embeddings,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df[df["language"] == "en"].copy()
        df = df.dropna(subset=["clean_comment"])
        df = df[df["word_count"] >= 3].copy()
        df = df.reset_index(drop=True)
        df["rag_text"] = df.apply(build_comment_text, axis=1)

        embeddings = load_or_create_embeddings(df["rag_text"].tolist())
        query_embedding = embedding_model.encode([retrieval_query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        candidate_count = min(700, len(df))
        candidate_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
        candidates = df.iloc[candidate_indices].copy()
        candidates["similarity_score"] = similarities[candidate_indices]

        intent = infer_query_intent(retrieval_query)
        candidates["engagement_score"] = np.log1p(
            candidates["like_count"].fillna(0).astype(float) +
            candidates["reply_count"].fillna(0).astype(float)
        )
        max_engagement = candidates["engagement_score"].max()
        if max_engagement > 0:
            candidates["engagement_score"] = candidates["engagement_score"] / max_engagement

        candidates["category_relevance_score"] = candidates.apply(
            lambda row: calculate_category_relevance(retrieval_query, row, intent),
            axis=1,
        )
        candidates["lexical_relevance_score"] = candidates.apply(
            lambda row: calculate_lexical_relevance(retrieval_query, row, intent),
            axis=1,
        )
        candidates["sentiment_relevance_score"] = candidates.apply(
            lambda row: calculate_sentiment_relevance(retrieval_query, row, intent),
            axis=1,
        )
        candidates["intent_penalty_score"] = candidates.apply(
            lambda row: calculate_intent_penalty(retrieval_query, row, intent),
            axis=1,
        )
        candidates["weighted_retrieval_score"] = (
            RETRIEVAL_WEIGHTS["semantic"] * candidates["similarity_score"] +
            RETRIEVAL_WEIGHTS["category"] * candidates["category_relevance_score"] +
            RETRIEVAL_WEIGHTS["lexical"] * candidates["lexical_relevance_score"] +
            RETRIEVAL_WEIGHTS["sentiment"] * candidates["sentiment_relevance_score"] +
            RETRIEVAL_WEIGHTS["engagement"] * candidates["engagement_score"] -
            candidates["intent_penalty_score"]
        )

        for column in [
            "similarity_score",
            "engagement_score",
            "category_relevance_score",
            "lexical_relevance_score",
            "sentiment_relevance_score",
            "intent_penalty_score",
            "weighted_retrieval_score",
        ]:
            candidates[column] = candidates[column].round(3)

        results = candidates.sort_values(by="weighted_retrieval_score", ascending=False).head(8)
        confidence = calculate_rag_confidence(results)
        evidence_text = format_evidence(results)

    answer = generate_feedback_answer(query, retrieval_query, evidence_text, confidence, history_text)

    return {
        "mode": "feedback_rag",
        "selectedAgent": "feedback_rag_retriever",
        "answer": answer,
        "confidence": confidence,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["comments_with_ner.csv", "rag_comment_embeddings.npy"],
        "evidence": feedback_evidence_rows(results),
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 8,
            "score": round(float(results["weighted_retrieval_score"].mean()), 3),
        },
    }


def run_strategy_rag_live(query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from strategy_rag import (
            INPUT_PATH,
            detect_strategy_goal,
            format_strategy_evidence,
            goal_relevance_score,
            embedding_model,
            load_or_create_embeddings,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df.dropna(subset=["strategy_text"]).copy()
        df = df.reset_index(drop=True)

        embeddings = load_or_create_embeddings(df["strategy_text"].tolist())
        goal = detect_strategy_goal(retrieval_query)
        query_embedding = embedding_model.encode([retrieval_query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        candidate_count = min(900, len(df))
        candidate_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
        candidates = df.iloc[candidate_indices].copy()
        candidates["strategy_similarity_score"] = similarities[candidate_indices]

        candidates["engagement_score"] = np.log1p(candidates["engagement_total"].fillna(0).astype(float))
        max_engagement = candidates["engagement_score"].max()
        if max_engagement > 0:
            candidates["engagement_score"] = candidates["engagement_score"] / max_engagement

        candidates["goal_relevance_score"] = candidates.apply(
            lambda row: goal_relevance_score(goal, row),
            axis=1,
        )
        priority_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
        candidates["priority_score"] = candidates["priority"].map(priority_map).fillna(0.5)
        candidates["strategy_retrieval_score"] = (
            0.45 * candidates["strategy_similarity_score"] +
            0.25 * candidates["goal_relevance_score"] +
            0.20 * candidates["priority_score"] +
            0.10 * candidates["engagement_score"]
        )

        for column in [
            "strategy_similarity_score",
            "goal_relevance_score",
            "priority_score",
            "engagement_score",
            "strategy_retrieval_score",
        ]:
            candidates[column] = candidates[column].round(3)

        results = candidates.sort_values(by="strategy_retrieval_score", ascending=False).head(12)
        evidence_text = format_strategy_evidence(results)

    answer = generate_strategy_answer_with_memory(query, retrieval_query, goal, evidence_text, history_text)

    return {
        "mode": "strategy_rag",
        "selectedAgent": "strategy_rag_retriever",
        "answer": answer,
        "strategyGoal": goal,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["strategy_evidence.csv", "strategy_evidence_embeddings.npy"],
        "evidence": strategy_evidence_rows(results),
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 12,
            "score": round(float(results["strategy_retrieval_score"].mean()), 3),
            "goal_relevance": round(float(results["goal_relevance_score"].mean()), 3),
        },
    }


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    query = clean_value(payload.get("message")).strip()
    messages = normalize_messages(payload.get("messages", []))

    if not query:
        raise ValueError("message is required")

    retrieval_query = build_memory_query(query, messages)
    history_text = conversation_text(messages)
    mode = detect_rag_mode(retrieval_query)
    result = run_strategy_rag_live(query, retrieval_query, history_text) if mode == "strategy" else run_feedback_rag(query, retrieval_query, history_text)
    result["model"] = get_model_name()
    result["query"] = query

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "type": error.__class__.__name__,
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)
