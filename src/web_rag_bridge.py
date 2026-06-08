import contextlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langsmith import traceable

from mlflow_tracing import (
    flush_mlflow_traces,
    get_active_mlflow_trace_id,
    mlflow_span,
    mlflow_status,
)
from text_cleanup import sanitize_text
from tracing_utils import (
    sanitize_trace_inputs,
    sanitize_trace_outputs,
    tracing_enabled,
    tracing_status,
)


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
    return sanitize_text(value)


def get_model_name():
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import get_openai_model

        return get_openai_model()


def get_generation_metadata():
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import get_last_llm_metadata

        return get_last_llm_metadata()


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


@mlflow_span("Generate Grounded Feedback Answer", "LLM")
@traceable(
    name="Generate Grounded Feedback Answer",
    run_type="chain",
    tags=["feedback-rag", "generation"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
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
Use plain ASCII punctuation and stay under 350 words.
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

    return sanitize_text(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.2,
        max_completion_tokens=600,
    ))


@mlflow_span("Generate Grounded Strategy Answer", "LLM")
@traceable(
    name="Generate Grounded Strategy Answer",
    run_type="chain",
    tags=["strategy-rag", "generation"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
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
Use plain ASCII punctuation and stay under 450 words.
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

    return sanitize_text(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.2,
        max_completion_tokens=700,
    ))


@mlflow_span("Strategy Synthesizer", "LLM")
@traceable(
    name="Strategy Synthesizer",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "synthesis"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def synthesize_web_augmented_strategy(
    query,
    internal_answer,
    internal_evidence,
    external_evidence,
    history_text,
):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are the Strategy Synthesizer for an optional university web-augmented RAG extension.

Use only the supplied internal YouTube strategy evidence and external web evidence.
Do not invent current prices, offers, dates, market shares, or competitor claims.
Treat external snippets as limited evidence and cite their source titles and URLs.
Clearly distinguish customer evidence from external market evidence.
If external evidence is unavailable or weak, say so and make the recommendation provisional.
Every external factual claim must cite its matching [Web N] evidence ID.
Never state a current price, offer, date, or product availability unless it
appears explicitly in that evidence item's snippet.
Do not infer willingness to pay, market demand, market share, or sales success
from promotional offer pages.
Use plain ASCII punctuation. Be concise and stay under 320 words.
Use no more than two bullets in each section.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Strategy question:
{query}

Customer Strategist Agent output based on internal YouTube evidence:
{internal_answer}

Internal YouTube strategy evidence:
{json.dumps(internal_evidence, ensure_ascii=False, indent=2)}

External web evidence:
{json.dumps(external_evidence, ensure_ascii=False, indent=2)}

Write exactly these Markdown headings without numbering:
**Internal YouTube Evidence**
**External Web Evidence**
**Final Validated Recommendation**
**Risks/Trade-offs**
**Confidence Level**

In External Web Evidence, include source titles and URLs. Explain whether the
external evidence validates, challenges, or adds context to the internal evidence.
Use [Web N] citations for every external claim.
Do not add a separate source register after the five requested sections.
"""

    return sanitize_text(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.15,
        max_completion_tokens=650,
    ))


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


def build_customer_strategy_brief(goal, evidence):
    lines = [f"Strategy goal: {goal}", "Top internal customer evidence:"]

    for item in evidence[:3]:
        category = item.get("issue_category") or "General feedback"
        signal = item.get("customer_signal") or item.get("comment") or "No signal available"
        recommendation = (
            item.get("customer_recommendation")
            or item.get("profit_recommendation")
            or "No recommendation available"
        )
        lines.append(f"- {category}: {signal[:220]} Recommendation: {recommendation[:220]}")

    return sanitize_text("\n".join(lines))


@mlflow_span("Run Analytical Agent", "TOOL")
@traceable(
    name="Run Analytical Agent",
    run_type="tool",
    tags=["analytical-agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
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


@mlflow_span("Feedback RAG Agent", "CHAIN")
@traceable(
    name="Feedback RAG Agent",
    run_type="chain",
    tags=["feedback-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
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
    llm_metadata = get_generation_metadata()

    return {
        "mode": "feedback_rag",
        "selectedAgent": "feedback_rag_agent",
        "answer": answer,
        "llm": llm_metadata,
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


@mlflow_span("Strategy RAG Agent", "CHAIN")
@traceable(
    name="Strategy RAG Agent",
    run_type="chain",
    tags=["strategy-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_strategy_rag_live(query, retrieval_query, history_text, generate_answer=True):
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

    evidence = strategy_evidence_rows(results)

    if generate_answer:
        answer = generate_strategy_answer_with_memory(query, retrieval_query, goal, evidence_text, history_text)
        llm_metadata = get_generation_metadata()
    else:
        answer = build_customer_strategy_brief(goal, evidence)
        llm_metadata = {}

    return {
        "mode": "strategy_rag",
        "selectedAgent": "strategy_rag_agent",
        "answer": answer,
        "llm": llm_metadata,
        "strategyGoal": goal,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["strategy_evidence.csv", "ChromaDB: strategy_evidence"],
        "evidence": evidence,
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "strategy_evidence",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 12,
            "score": round(float(results["strategy_retrieval_score"].mean()), 3),
            "goal_relevance": round(float(results["goal_relevance_score"].mean()), 3),
        },
    }


@mlflow_span("Customer Strategist Agent", "CHAIN")
@traceable(
    name="Customer Strategist Agent",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "customer-strategist", "internal-evidence"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_customer_strategist_agent(query, retrieval_query, history_text):
    return run_strategy_rag_live(query, retrieval_query, history_text, generate_answer=False)


@mlflow_span("Web-Augmented Strategy RAG", "CHAIN")
@traceable(
    name="Web-Augmented Strategy RAG",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "advanced-extension"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_web_augmented_strategy_rag(query, retrieval_query, history_text, research_focus):
    with contextlib.redirect_stdout(sys.stderr):
        from web_research import search_market_evidence

    internal_result = run_customer_strategist_agent(query, retrieval_query, history_text)
    web_research = search_market_evidence(retrieval_query, research_focus, max_results=3)
    external_evidence = [
        {"evidence_id": f"Web {index}", **item}
        for index, item in enumerate(web_research["evidence"], start=1)
    ]
    answer = synthesize_web_augmented_strategy(
        query=query,
        internal_answer=internal_result["answer"],
        internal_evidence=internal_result["evidence"][:3],
        external_evidence=external_evidence,
        history_text=history_text,
    )
    llm_metadata = get_generation_metadata()

    if len(external_evidence) >= 3:
        confidence = "High"
    elif external_evidence:
        confidence = "Medium"
    else:
        confidence = "Low"

    if llm_metadata.get("fallback_used") and confidence == "High":
        confidence = "Medium"

    return {
        "mode": "web_augmented_strategy_rag",
        "selectedAgent": "web_augmented_strategy_rag",
        "answer": answer,
        "llm": llm_metadata,
        "confidence": confidence,
        "strategyGoal": internal_result.get("strategyGoal"),
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": internal_result["sources"] + [
            item["url"] for item in external_evidence
        ],
        "evidence": internal_result["evidence"],
        "internalStrategyAnswer": internal_result["answer"],
        "externalEvidence": external_evidence,
        "webResearch": {
            "provider": web_research["provider"],
            "result_count": web_research["result_count"],
            "retrieved_at": web_research["retrieved_at"],
            "errors": web_research["errors"],
            "focus": research_focus,
        },
        "retrieval": internal_result["retrieval"],
        "toolTrace": [
            "web_augmented_strategy_rag",
            "customer_strategist_agent",
            "market_research_agent",
            "strategy_synthesizer",
        ],
    }


@mlflow_span("Samsung Document RAG Agent", "CHAIN")
@traceable(
    name="Samsung Document RAG Agent",
    run_type="chain",
    tags=["document-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_samsung_document_rag(query, messages):
    with contextlib.redirect_stdout(sys.stderr):
        from document_rag_bridge import answer_document_question

    document_result = answer_document_question(query, messages)
    evidence = document_result["evidence"]

    return {
        "mode": "document_rag",
        "selectedAgent": "samsung_document_rag",
        "answer": document_result["answer"],
        "llm": document_result.get("llm", {}),
        "confidence": document_result["confidence"],
        "contextualQuery": query,
        "memoryUsed": bool(messages),
        "sources": sorted({item["filename"] for item in evidence}),
        "evidence": evidence,
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "samsung_documents",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 6,
            "score": round(
                sum(item["similarity"] for item in evidence) / len(evidence),
                3,
            ) if evidence else 0,
        },
        "toolTrace": ["samsung_document_rag", "document_retrieval", "document_answer_generation"],
    }


@mlflow_span("Samsung Advisor Request", "CHAIN")
@traceable(
    name="YouTube Intelligence Advisor Request",
    run_type="chain",
    tags=["advisor", "multi-agent-rag"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_advisor(payload):
    query = clean_value(payload.get("message")).strip()
    messages = normalize_messages(payload.get("messages", []))

    if not query:
        raise ValueError("message is required")

    retrieval_query = build_memory_query(query, messages)
    with contextlib.redirect_stdout(sys.stderr):
        from web_strategy_policy import is_feedback_request

    routing_query = (
        retrieval_query
        if is_follow_up(query) and not is_feedback_request(query)
        else query
    )
    history_text = conversation_text(messages)

    with contextlib.redirect_stdout(sys.stderr):
        from agent_router import AVAILABLE_AGENTS, route_intent
        from document_rag_bridge import load_manifest

        document_names = [item.get("filename", "") for item in load_manifest() if item.get("filename")]
        routing = route_intent(routing_query, document_names=document_names)

    selected_agent = routing["selected_agent"]
    routed_query = (
        routing.get("rewritten_query")
        or routing.get("normalized_query")
        or retrieval_query
    )

    if selected_agent == "web_augmented_strategy_rag":
        result = run_web_augmented_strategy_rag(
            query,
            routed_query,
            history_text,
            routing.get("external_research_focus", []),
        )
    elif selected_agent == "strategy_rag_agent":
        result = run_strategy_rag_live(query, routed_query, history_text)
    elif selected_agent == "feedback_rag_agent":
        result = run_feedback_rag(query, routed_query, history_text)
    elif selected_agent == "samsung_document_rag":
        result = run_samsung_document_rag(query, messages)
    else:
        result = run_analytical_tool(selected_agent, query, routed_query, history_text)

    if selected_agent in {
        "feedback_rag_agent",
        "strategy_rag_agent",
        "web_augmented_strategy_rag",
        "samsung_document_rag",
    }:
        llm_metadata = result.get("llm", {})
        result["model"] = llm_metadata.get("model") or get_model_name()
        result["llmProvider"] = llm_metadata.get("provider")
        result["llmFallbackUsed"] = llm_metadata.get("fallback_used", False)
        result["llmFallbackReason"] = llm_metadata.get("fallback_reason")

    result["routingReason"] = routing["reason"]
    result["matchedTerms"] = routing["matched_terms"]
    result["routingConfidence"] = routing["confidence"]
    result["routingMethod"] = routing["routing_method"]
    result["routerModel"] = routing["router_model"]
    result["routerProvider"] = routing.get("router_provider")
    result["routerFallbackUsed"] = routing.get("router_fallback_used", False)
    result["routerFallbackReason"] = routing.get("router_fallback_reason")
    result["needsExternalResearch"] = routing.get("needs_external_research", False)
    result["externalResearchFocus"] = routing.get("external_research_focus", [])
    result["rewrittenQuery"] = routing.get("rewritten_query", routed_query)
    result["normalizedQuery"] = routed_query
    result["toolTrace"] = [routing["routing_method"]] + result.get(
        "toolTrace",
        [selected_agent],
    )
    result["availableTools"] = AVAILABLE_AGENTS
    result["query"] = query
    result["uploadedDocumentCount"] = len(document_names)
    result["langsmithTracing"] = tracing_status()
    result["mlflowTracing"] = mlflow_status()
    result["mlflowTraceId"] = get_active_mlflow_trace_id()

    return result


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    trace_id = uuid4()
    result = run_advisor(
        payload,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {
                "source": "live_advisor",
                "message_count": len(payload.get("messages", [])),
            },
            "tags": ["live-advisor-request"],
        },
    )

    if tracing_enabled():
        result["langsmithTraceId"] = str(trace_id)

    print(json.dumps(result, ensure_ascii=True))


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
                ensure_ascii=True,
            )
        )
        sys.exit(1)
    finally:
        flush_mlflow_traces()
