import os
import re
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

FOCUS_SIGNALS = {
    "product_lifecycle_verification": [],
    "competitor_apple_playbook": [],
    "competitor_pixel_playbook": [],
    "competitor_apple_sales_result": [],
    "competitor_pixel_sales_result": [],
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
    "uae_retail_events": [
        "uae holiday",
        "uae holidays",
        "uae event",
        "uae events",
        "retail event",
        "shopping festival",
        "dubai shopping festival",
        "ramadan",
        "eid",
        "white friday",
        "black friday",
        "promotion calendar",
        "promotional calendar",
        "seasonal discount",
        "holiday discount",
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
    "maximize price",
    "maximize profit",
    "promotion calendar",
    "discount strategy",
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


def get_current_galaxy_s_generation():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return int(os.getenv("CURRENT_GALAXY_S_GENERATION", "26"))


def classify_product_lifecycle(query):
    query_lower = query.lower()
    current_generation = get_current_galaxy_s_generation()
    model_match = re.search(
        r"\b(?:galaxy\s+)?s[\s-]?(\d{2})\s*(?:ultra|plus|\+|fe)?\b",
        query_lower,
    )
    requested_generation = int(model_match.group(1)) if model_match else None

    if requested_generation is not None:
        requested_model = f"Galaxy S{requested_generation} Ultra"

        if requested_generation > current_generation:
            lifecycle = "future_product"
            instruction = (
                f"Treat {requested_model} as an unannounced future planning concept. "
                f"Do not claim confirmed specifications, prices, launch dates, offers, "
                f"or Samsung commitments. Use the current Galaxy S{current_generation} "
                "Ultra only as a market benchmark."
            )
        elif requested_generation == current_generation:
            lifecycle = "current_product"
            instruction = (
                f"Treat {requested_model} as the current product. Analyze its verified "
                "positioning, offers, strengths, and gaps, then recommend enhancements "
                "to the product, pricing, promotion, or customer experience."
            )
        else:
            lifecycle = "previous_product"
            instruction = (
                f"Treat {requested_model} as an existing previous-generation product. "
                "Use verified historical/current evidence and recommend lifecycle, "
                "portfolio, pricing, or upgrade-path actions without describing it as new."
            )
    elif any(
        phrase in query_lower
        for phrase in ["next ultra", "next flagship", "future ultra", "future device", "future phone"]
    ):
        requested_model = "Future Galaxy Ultra"
        lifecycle = "future_product"
        instruction = (
            f"Treat the requested device as an unannounced future planning concept. "
            f"Use the current Galaxy S{current_generation} Ultra only as a benchmark "
            "and do not invent confirmed specifications, prices, dates, or offers."
        )
    else:
        requested_model = None
        lifecycle = "unspecified"
        instruction = (
            "The query does not identify a specific Galaxy S generation. Avoid assuming "
            "whether the product is current or future; state the interpretation used."
        )

    return {
        "requested_model": requested_model,
        "requested_generation": requested_generation,
        "current_generation": current_generation,
        "current_model": f"Galaxy S{current_generation} Ultra",
        "lifecycle": lifecycle,
        "instruction": instruction,
    }


def detect_external_research_focus(query):
    query_lower = query.lower()
    focus = [
        focus
        for focus, signals in FOCUS_SIGNALS.items()
        if signals and any(signal in query_lower for signal in signals)
    ]
    lifecycle = classify_product_lifecycle(query)

    if lifecycle["requested_generation"] is not None:
        focus.insert(0, "product_lifecycle_verification")

    if lifecycle["requested_generation"] is not None and has_strategy_intent(query):
        commercial_intent = any(
            term in query_lower
            for term in ["profit", "revenue", "sales", "margin", "position", "pricing", "price"]
        )

        if lifecycle["lifecycle"] == "future_product":
            if commercial_intent:
                focus.extend(
                    [
                        "competitor_apple_playbook",
                        "competitor_apple_sales_result",
                        "competitor_pixel_playbook",
                        "competitor_pixel_sales_result",
                    ]
                )
            focus.extend(["current_market_trends", "positioning_and_pricing"])
        elif lifecycle["lifecycle"] in {"current_product", "previous_product"}:
            focus.extend(
                [
                    "current_market_trends",
                    "positioning_and_pricing",
                ]
            )

        if commercial_intent and "competitor_apple_sales_result" not in focus:
            focus.extend(
                [
                    "competitor_apple_sales_result",
                    "competitor_pixel_sales_result",
                ]
            )

    return list(dict.fromkeys(focus))


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
            lifecycle = classify_product_lifecycle(original_query)
            if lifecycle["requested_generation"] is not None:
                result["reason"] = (
                    "The request names a specific Galaxy generation and asks for "
                    "strategy. The system automatically adds external research to "
                    "verify the product lifecycle and benchmark the current market."
                )
            else:
                result["reason"] = (
                    "Controlled routing policy detected a strategy request requiring "
                    "current external context."
                )

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
