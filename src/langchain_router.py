from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from openai_client import (
    fallback_reason,
    get_llama_model,
    get_ollama_base_url,
    get_openai_model,
    is_openai_fallback_error,
    llama_fallback_enabled,
)
from tracing_utils import sanitize_trace_inputs, sanitize_trace_outputs


AgentName = Literal[
    "summarization_agent",
    "sentiment_agent",
    "issue_agent",
    "topic_agent",
    "keyword_agent",
    "feedback_rag_agent",
    "strategy_rag_agent",
]


class RoutingDecision(BaseModel):
    """Validated decision returned by the LangChain LLM router."""

    selected_agent: AgentName = Field(
        description="The single specialist agent that should handle the user request."
    )
    normalized_query: str = Field(
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
   roadmaps, priorities, profit, revenue, or customer-satisfaction strategy.

Routing rules:
- Distinguish aggregate analytics from evidence-seeking questions.
- A request for "main complaints" is issue_agent; a request asking why users
  complain about the S-Pen is feedback_rag_agent.
- A request asking which complaints or comments mention a specific feature is
  feedback_rag_agent because it requires retrieved evidence, not aggregate counts.
- A request asking what Samsung should do about complaints is strategy_rag_agent.
- Treat conversation context included in the query as useful follow-up context.
- Preserve the user's meaning in normalized_query without adding new claims.
"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT.strip()),
        ("human", "Route this request:\n\n{user_query}"),
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
    return result


@traceable(
    name="LangChain Semantic Router",
    run_type="chain",
    tags=["router", "langchain", "structured-output"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def route_with_langchain(user_query):
    try:
        decision = get_openai_router_chain().invoke({"user_query": user_query})
        return format_routing_result(
            decision,
            "langchain_openai",
            "openai",
            get_openai_model(),
        )
    except Exception as error:
        if not llama_fallback_enabled() or not is_openai_fallback_error(error):
            raise

        decision = get_llama_router_chain().invoke({"user_query": user_query})
        result = format_routing_result(
            decision,
            "langchain_ollama_fallback",
            "ollama",
            get_llama_model(),
            fallback_used=True,
            reason=fallback_reason(error),
        )
        result["normalized_query"] = user_query
        return result
