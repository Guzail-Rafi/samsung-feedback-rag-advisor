import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_router import route_intent_rules
from web_rag_bridge import build_memory_query, is_follow_up
from web_strategy_policy import classify_product_lifecycle, is_feedback_request


CASES = [
    (
        "How should Samsung price the next Ultra in the UAE using current offers?",
        "web_augmented_strategy_rag",
    ),
    (
        "How should Samsung maximize profit with a UAE promotion calendar for Ramadan, Eid, and Dubai shopping events?",
        "web_augmented_strategy_rag",
    ),
    ("What people think about S Pen?", "feedback_rag_agent"),
    ("What do users think about current iPhone offers?", "feedback_rag_agent"),
    ("What are users saying about Samsung pricing in the UAE?", "feedback_rag_agent"),
    ("What do people complain about in the current market?", "feedback_rag_agent"),
    ("What are the main complaint categories?", "issue_agent"),
    ("Summarize what users say about battery.", "summarization_agent"),
    ("How should Samsung design the S27 Ultra?", "web_augmented_strategy_rag"),
    (
        "What should Samsung's strategy be for the S27 Ultra to maximize profit?",
        "web_augmented_strategy_rag",
    ),
    ("What strategy should Samsung use for the Galaxy S22 Ultra now?", "web_augmented_strategy_rag"),
    ("Summarize the uploaded Samsung document.", "samsung_document_rag"),
    ("What does the uploaded report say about battery?", "samsung_document_rag"),
]


def main():
    for query, expected in CASES:
        actual = route_intent_rules(query)["selected_agent"]
        assert actual == expected, f"{query!r}: expected {expected}, got {actual}"

    query = "What people think about S Pen?"
    messages = [
        {
            "role": "user",
            "content": "How should Samsung price the next Ultra in the UAE using current offers?",
        },
        {"role": "assistant", "content": "Use premium pricing and current offers."},
    ]
    retrieval_query = build_memory_query(query, messages)
    routing_query = (
        retrieval_query
        if is_follow_up(query) and not is_feedback_request(query)
        else query
    )
    actual = route_intent_rules(routing_query)["selected_agent"]
    assert actual == "feedback_rag_agent", actual

    document_follow_up = route_intent_rules(
        "What does it recommend?",
        document_names=["samsung-roadmap.pdf"],
    )["selected_agent"]
    assert document_follow_up == "samsung_document_rag", document_follow_up

    unrelated_with_document = route_intent_rules(
        "Why are users unhappy about the S-Pen?",
        document_names=["samsung-roadmap.pdf"],
    )["selected_agent"]
    assert unrelated_with_document == "feedback_rag_agent", unrelated_with_document

    future_product = classify_product_lifecycle(
        "How should Samsung price the Galaxy S27 Ultra in the UAE using current offers?"
    )
    assert future_product["lifecycle"] == "future_product", future_product
    assert future_product["current_model"] == "Galaxy S26 Ultra", future_product

    current_product = classify_product_lifecycle(
        "How should Samsung improve the Galaxy S26 Ultra now?"
    )
    assert current_product["lifecycle"] == "current_product", current_product

    inferred_focus = route_intent_rules(
        "What should Samsung's strategy be for the S27 Ultra to maximize profit?"
    )["external_research_focus"]
    assert "product_lifecycle_verification" in inferred_focus, inferred_focus
    assert "competitor_apple_playbook" in inferred_focus, inferred_focus
    assert "competitor_apple_sales_result" in inferred_focus, inferred_focus
    assert "competitor_pixel_playbook" in inferred_focus, inferred_focus
    assert "competitor_pixel_sales_result" in inferred_focus, inferred_focus
    assert "current_market_trends" in inferred_focus, inferred_focus
    assert "positioning_and_pricing" in inferred_focus, inferred_focus

    print(f"Passed {len(CASES) + 12} unified routing and lifecycle regression checks.")


if __name__ == "__main__":
    main()
