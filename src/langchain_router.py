from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from mlflow_tracing import mlflow_span
from openai_client import (
    fallback_reason,
    get_llama_model,
    get_ollama_base_url,
    get_openai_model,
    is_openai_fallback_error,
    llama_fallback_enabled,
)
from tracing_utils import sanitize_trace_inputs, sanitize_trace_outputs
from web_strategy_policy import apply_web_strategy_policy


AgentName = Literal[
    "summarization_agent",
    "sentiment_agent",
    "issue_agent",
    "topic_agent",
    "keyword_agent",
    "feedback_rag_agent",
    "strategy_rag_agent",
    "web_augmented_strategy_rag",
    "samsung_document_rag",
]

ExternalResearchFocus = Literal[
    "latest_samsung_news",
    "uae_pricing_offers",
    "competitor_iphone_offers",
    "current_market_trends",
    "regional_market_context",
    "positioning_and_pricing",
]


class RoutingDecision(BaseModel):
    """Validated decision returned by the LangChain LLM router."""

    selected_agent: AgentName = Field(
        description="The single specialist agent that should handle the user request."
    )
    needs_external_research: bool = Field(
        description="True only for web_augmented_strategy_rag."
    )
    external_research_focus: list[ExternalResearchFocus] = Field(
        description="Controlled external research topics; empty for all non-web routes."
    )
    rewritten_query: str = Field(
        description="A concise standalone rewrite preserving the user's intended request."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the selected route from 0 to 1.",
    )
    reason: str = Field(
        description="A short explanation of why this specialist agent was selected."
    )


ROUTER_SYSTEM_PROMPT = """
You are the semantic query router for a university YouTube intelligence system.
Choose exactly one specialist agent. Do not answer the user's question.

Available specialist agents:

1. summarization_agent
   Use when the user explicitly requests a summary, overview, or condensed
   explanation of feedback.

2. sentiment_agent
   Use for aggregate sentiment counts, percentages, distributions, or questions
   asking specifically about positive, negative, or neutral proportions.

3. issue_agent
   Use for aggregate issue categories, complaint categories, common concerns,
   or problem-frequency analysis. Do not use for evidence about one specific
   product issue; that belongs to feedback_rag_agent.

4. topic_agent
   Use for topic-model results, broad discussion themes, or aggregate trends.

5. keyword_agent
   Use for keywords, key phrases, common words, or TF-IDF terms.

6. feedback_rag_agent
   Use for grounded questions asking what users say, why users feel something,
   comparisons, supporting evidence, examples, comments that mention something,
   or details about a specific product, feature, complaint, or opinion.

7. strategy_rag_agent
   Use for recommendations, actions Samsung should take, future product design,
   roadmaps, priorities, profit, revenue, or customer-satisfaction strategy that
   can be answered from the internal YouTube strategy evidence.

8. web_augmented_strategy_rag
   Optional advanced route. Use only when BOTH conditions are true:
   A. The request asks for strategy, business action, roadmap, pricing, or
      positioning.
   B. The requested strategy explicitly needs current external context, such as
      latest news, UAE pricing, current offers, competitor offers, current
      market trends, regional conditions, or recent events.

9. samsung_document_rag
   Use when the request explicitly asks about an uploaded document, uploaded
   file, PDF, report, document evidence, or asks to summarize/explain/compare
   information from the user's uploaded Samsung documents.

Routing rules:
- Distinguish aggregate analytics from evidence-seeking questions.
- A request for "main complaints" is issue_agent; a request asking why users
  complain about the S-Pen is feedback_rag_agent.
- A request asking which complaints or comments mention a specific feature is
  feedback_rag_agent because it requires retrieved evidence, not aggregate counts.
- A request asking what Samsung should do about complaints is strategy_rag_agent.
- Do not use web_augmented_strategy_rag for normal internal feedback, complaint,
  feature-priority, or roadmap questions unless current external context is
  explicitly required.
- Questions asking what users or people think, feel, say, complain about, or
  discuss always belong to feedback_rag_agent, even when they mention UAE,
  pricing, offers, competitors, markets, "current", or "latest".
- Do not let prior conversation context turn a clearly stated feedback question
  into a strategy or web-augmented request.
- Use samsung_document_rag only when the user explicitly refers to uploaded
  documents/files/reports, or when uploaded documents are available and a
  natural follow-up clearly refers to them, such as "summarize it" or "what
  does it recommend?". Do not confuse YouTube comments with uploaded document
  evidence, and do not route unrelated questions to documents merely because
  documents are available.
- For web_augmented_strategy_rag, set needs_external_research=true and select
  only relevant values in external_research_focus.
- For every other route, set needs_external_research=false and
  external_research_focus=[].
- Treat conversation context included in the query as useful follow-up context.
- Preserve the user's meaning in rewritten_query without adding new claims.
"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT.strip()),
        (
            "human",
            "Uploaded Samsung documents currently available:\n{document_context}\n\n"
            "Route this request:\n\n{user_query}",
        ),
    ]
)

_OPENAI_ROUTER_CHAIN = None
_LLAMA_ROUTER_CHAIN = None


def get_openai_router_chain():
    global _OPENAI_ROUTER_CHAIN

    if _OPENAI_ROUTER_CHAIN is None:
        model = ChatOpenAI(
            model=get_openai_model(),
            temperature=0,
            reasoning_effort="none",
            verbosity="low",
            max_completion_tokens=300,
            timeout=45,
            max_retries=1,
        )
        structured_model = model.with_structured_output(
            RoutingDecision,
            method="json_schema",
        )
        _OPENAI_ROUTER_CHAIN = ROUTER_PROMPT | structured_model

    return _OPENAI_ROUTER_CHAIN


def get_llama_router_chain():
    global _LLAMA_ROUTER_CHAIN

    if _LLAMA_ROUTER_CHAIN is None:
        model = ChatOllama(
            model=get_llama_model(),
            base_url=get_ollama_base_url(),
            temperature=0,
            num_ctx=8192,
            num_predict=300,
            keep_alive="10m",
        )
        structured_model = model.with_structured_output(
            RoutingDecision,
            method="json_schema",
        )
        _LLAMA_ROUTER_CHAIN = ROUTER_PROMPT | structured_model

    return _LLAMA_ROUTER_CHAIN


def format_routing_result(decision, method, provider, model, fallback_used=False, reason=None):
    if isinstance(decision, RoutingDecision):
        result = decision.model_dump()
    else:
        result = RoutingDecision.model_validate(decision).model_dump()

    result["routing_method"] = method
    result["router_provider"] = provider
    result["router_model"] = model
    result["router_fallback_used"] = fallback_used
    result["router_fallback_reason"] = reason
    result["matched_terms"] = []
    result["normalized_query"] = result["rewritten_query"]
    return result


@mlflow_span("LangChain Semantic Router", "CHAIN")
@traceable(
    name="LangChain Semantic Router",
    run_type="chain",
    tags=["router", "langchain", "structured-output"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def route_with_langchain(user_query, document_names=None):
    document_context = (
        ", ".join(document_names[:10])
        if document_names
        else "No uploaded Samsung documents are currently available."
    )
    inputs = {"user_query": user_query, "document_context": document_context}

    try:
        decision = get_openai_router_chain().invoke(inputs)
        return apply_web_strategy_policy(
            format_routing_result(
                decision,
                "langchain_openai",
                "openai",
                get_openai_model(),
            ),
            user_query,
        )
    except Exception as error:
        if not llama_fallback_enabled() or not is_openai_fallback_error(error):
            raise

        decision = get_llama_router_chain().invoke(inputs)
        result = format_routing_result(
            decision,
            "langchain_ollama_fallback",
            "ollama",
            get_llama_model(),
            fallback_used=True,
            reason=fallback_reason(error),
        )
        result["rewritten_query"] = user_query
        result["normalized_query"] = user_query
        return apply_web_strategy_policy(result, user_query)
