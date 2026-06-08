import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

FOCUS_SIGNALS = {
    "latest_samsung_news": [
        "latest samsung",
        "recent samsung",
        "samsung news",
        "latest galaxy",
        "recent galaxy",
    ],
    "uae_pricing_offers": [
        "uae",
        "dubai",
        "abu dhabi",
        "aed",
        "uae price",
        "uae pricing",
        "uae offer",
    ],
    "competitor_iphone_offers": [
        "competitor",
        "iphone offer",
        "iphone price",
        "apple offer",
        "apple pricing",
        "competing offer",
    ],
    "current_market_trends": [
        "current market",
        "latest market",
        "market trend",
        "market trends",
        "current trend",
        "current trends",
        "smartphone market",
    ],
    "regional_market_context": [
        "regional market",
        "regional context",
        "middle east",
        "gulf market",
        "gcc market",
        "uae market",
    ],
    "positioning_and_pricing": [
        "how should samsung position",
        "how should samsung price",
        "position the next ultra",
        "price the next ultra",
        "positioning strategy",
        "pricing strategy",
        "next ultra price",
        "next flagship price",
        "offers",
        "discounts",
        "trade-in",
        "trade in",
    ],
}

STRATEGY_SIGNALS = [
    "should samsung",
    "how should",
    "strategy",
    "roadmap",
    "recommend",
    "strategic",
    "position samsung",
    "position the",
    "respond to",
    "pricing",
    "price the",
    "next ultra",
    "next flagship",
    "profit",
    "business",
]

FEEDBACK_SIGNALS = [
    "according to the comments",
    "according to comments",
    "what are users saying",
    "what do users think",
    "what users think",
    "what people think",
    "what people say",
    "what are people saying",
    "how do users feel",
    "how do people feel",
    "why are users unhappy",
    "why are people unhappy",
    "user opinion",
    "people's opinion",
    "customer opinion",
    "user feedback",
    "customer feedback",
    "users complain",
    "people complain",
    "users discuss",
    "people discuss",
    "show evidence",
    "what complaints mention",
]

ANALYTICAL_SIGNALS = [
    "summarize",
    "summary",
    "overview",
    "distribution",
    "percentage",
    "how many",
    "main complaints",
    "complaint categories",
    "top issues",
    "common problems",
    "most frequent",
    "top discussion",
    "main themes",
    "top topics",
    "keywords",
    "key phrases",
]


def web_augmented_strategy_enabled():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    value = os.getenv("WEB_AUGMENTED_STRATEGY_ENABLED", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def detect_external_research_focus(query):
    query_lower = query.lower()
    return [
        focus
        for focus, signals in FOCUS_SIGNALS.items()
        if any(signal in query_lower for signal in signals)
    ]


def is_feedback_request(query):
    query_lower = query.lower()
    has_feedback_signal = any(signal in query_lower for signal in FEEDBACK_SIGNALS)
    has_analytical_signal = any(signal in query_lower for signal in ANALYTICAL_SIGNALS)
    return has_feedback_signal and not has_analytical_signal


def has_strategy_intent(query):
    query_lower = query.lower()
    return any(signal in query_lower for signal in STRATEGY_SIGNALS)


def should_use_web_augmented_strategy(query):
    if not web_augmented_strategy_enabled():
        return False

    query_lower = query.lower()
    focus = detect_external_research_focus(query)

    if not focus:
        return False

    if is_feedback_request(query):
        return False

    return has_strategy_intent(query)


def apply_web_strategy_policy(result, original_query):
    if result.get("selected_agent") == "samsung_document_rag":
        result["needs_external_research"] = False
        result["external_research_focus"] = []
        result["rewritten_query"] = result.get("rewritten_query") or original_query
        result["normalized_query"] = result["rewritten_query"]
        return result

    if is_feedback_request(original_query):
        result["selected_agent"] = "feedback_rag_agent"
        result["needs_external_research"] = False
        result["external_research_focus"] = []
        result["rewritten_query"] = result.get("rewritten_query") or original_query
        result["normalized_query"] = result["rewritten_query"]
        result["reason"] = (
            "The request asks what users think, feel, say, complain about, or discuss; "
            "using Feedback RAG without external web research."
        )
        return result

    should_use_web = should_use_web_augmented_strategy(original_query)
    selected_agent = result.get("selected_agent")

    if should_use_web:
        result["selected_agent"] = "web_augmented_strategy_rag"
        result["needs_external_research"] = True
        result["external_research_focus"] = detect_external_research_focus(original_query)
        result["rewritten_query"] = result.get("rewritten_query") or original_query
        result["normalized_query"] = result["rewritten_query"]

        if selected_agent != "web_augmented_strategy_rag":
            result["reason"] = (
                f"{result.get('reason', '').strip()} "
                "Controlled routing policy detected a strategy request requiring current external context."
            ).strip()

        return result

    if selected_agent == "web_augmented_strategy_rag":
        result["selected_agent"] = "strategy_rag_agent"
        result["reason"] = (
            "Web-augmented Strategy RAG requires both strategy intent and explicit "
            "current external context; using internal Strategy RAG."
        )

    result["needs_external_research"] = False
    result["external_research_focus"] = []
    result["rewritten_query"] = result.get("rewritten_query") or original_query
    result["normalized_query"] = result["rewritten_query"]
    return result
