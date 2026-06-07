import contextlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(PROJECT_ROOT)


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


def run_analytical_tool(agent_name, query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from agent_router import (
            issue_agent,
            keyword_agent,
            sentiment_agent,
            summarization_agent,
            topic_agent,
        )

        tool_handlers = {
            "summarization_agent": {
                "handler": summarization_agent,
                "sources": ["llm_summaries.csv"],
            },
            "sentiment_agent": {
                "handler": sentiment_agent,
                "sources": ["comments_with_ner.csv"],
            },
            "issue_agent": {
                "handler": issue_agent,
                "sources": ["comments_with_ner.csv"],
            },
            "topic_agent": {
                "handler": topic_agent,
                "sources": ["comments_with_ner.csv", "topic_keywords.csv"],
            },
            "keyword_agent": {
                "handler": keyword_agent,
                "sources": ["top_keywords_overall.csv"],
            },
        }

        tool = tool_handlers.get(agent_name)

        if tool is None:
            raise ValueError(f"Unsupported analytical agent: {agent_name}")

        answer = tool["handler"](retrieval_query)

    return {
        "mode": "analytical_tool",
        "selectedAgent": agent_name,
        "answer": answer,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": tool["sources"],
    }


def run_feedback_rag(query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from rag_answer_generator import (
            INPUT_PATH,
            build_comment_text,
            calculate_rag_confidence,
            format_evidence,
            load_or_create_embeddings,
            load_or_create_feedback_vector_store,
            retrieve_comments,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df[df["language"] == "en"].copy()
        df = df.dropna(subset=["clean_comment"])
        df = df[df["word_count"] >= 3].copy()
        df = df.reset_index(drop=True)
        df["rag_text"] = df.apply(build_comment_text, axis=1)

        embeddings = load_or_create_embeddings(df["rag_text"].tolist())
        vector_collection = load_or_create_feedback_vector_store(df, embeddings)
        results = retrieve_comments(
            query=retrieval_query,
            df=df,
            comment_embeddings=embeddings,
            top_k=8,
            vector_collection=vector_collection,
            candidate_count=700,
        )
        confidence = calculate_rag_confidence(results)
        evidence_text = format_evidence(results)

    answer = generate_feedback_answer(query, retrieval_query, evidence_text, confidence, history_text)

    return {
        "mode": "feedback_rag",
        "selectedAgent": "feedback_rag_agent",
        "answer": answer,
        "confidence": confidence,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["comments_with_ner.csv", "ChromaDB: feedback_comments"],
        "evidence": feedback_evidence_rows(results),
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "feedback_comments",
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
            load_or_create_embeddings,
            load_or_create_strategy_vector_store,
            retrieve_strategy_evidence,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df.dropna(subset=["strategy_text"]).copy()
        df = df.reset_index(drop=True)

        embeddings = load_or_create_embeddings(df["strategy_text"].tolist())
        vector_collection = load_or_create_strategy_vector_store(df, embeddings)
        goal = detect_strategy_goal(retrieval_query)
        results = retrieve_strategy_evidence(
            query=retrieval_query,
            goal=goal,
            df=df,
            embeddings=embeddings,
            top_k=12,
            vector_collection=vector_collection,
            candidate_count=900,
        )
        evidence_text = format_strategy_evidence(results)

    answer = generate_strategy_answer_with_memory(query, retrieval_query, goal, evidence_text, history_text)

    return {
        "mode": "strategy_rag",
        "selectedAgent": "strategy_rag_agent",
        "answer": answer,
        "strategyGoal": goal,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["strategy_evidence.csv", "ChromaDB: strategy_evidence"],
        "evidence": strategy_evidence_rows(results),
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "strategy_evidence",
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

    with contextlib.redirect_stdout(sys.stderr):
        from agent_router import AVAILABLE_AGENTS, route_intent

        routing = route_intent(retrieval_query)

    selected_agent = routing["selected_agent"]
    routed_query = routing.get("normalized_query") or retrieval_query

    if selected_agent == "strategy_rag_agent":
        result = run_strategy_rag_live(query, routed_query, history_text)
    elif selected_agent == "feedback_rag_agent":
        result = run_feedback_rag(query, routed_query, history_text)
    else:
        result = run_analytical_tool(selected_agent, query, routed_query, history_text)

    if selected_agent in {"feedback_rag_agent", "strategy_rag_agent"}:
        result["model"] = get_model_name()

    result["routingReason"] = routing["reason"]
    result["matchedTerms"] = routing["matched_terms"]
    result["routingConfidence"] = routing["confidence"]
    result["routingMethod"] = routing["routing_method"]
    result["routerModel"] = routing["router_model"]
    result["normalizedQuery"] = routed_query
    result["toolTrace"] = [routing["routing_method"], selected_agent]
    result["availableTools"] = AVAILABLE_AGENTS
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
