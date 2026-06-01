import os
import pandas as pd
from dotenv import load_dotenv
from openai_client import generate_chat_response, get_openai_client


# =========================
# CONFIG
# =========================

load_dotenv()

COMMENTS_PATH = "data/processed/comments_with_topics.csv"
RAG_ANSWERS_PATH = "data/processed/rag_answers.csv"
SUMMARIES_PATH = "data/processed/llm_summaries.csv"
OUTPUT_PATH = "data/processed/agent_router_results.csv"
STRATEGY_RAG_PATH = "data/processed/strategy_rag_results.csv"
STRATEGY_REFINEMENT_PATH = "data/processed/strategy_refinement_results.csv" 

client = get_openai_client()


# =========================
# ROUTER
# =========================

def detect_intent(user_query):
    """
    Rule-based agent router.
    Routes the user query to the correct analysis module.
    """

    query = user_query.lower()

    # Strategy/Product Roadmap RAG
    strategy_words = [
        "s27", "roadmap", "strategy", "design", "make samsung",
        "maximum profit", "profit", "revenue", "sales",
        "customer satisfaction", "satisfaction", "product plan",
        "prioritize", "next flagship", "next ultra", "reduce complaints",
        "what should samsung do", "how should samsung"
    ]

    if any(word in query for word in strategy_words):
        return "strategy_rag_agent"

    if any(word in query for word in ["summarize", "summary", "overview", "overall"]):
        return "summarization_agent"

    if any(word in query for word in ["sentiment", "positive", "negative", "neutral", "emotion"]):
        return "sentiment_agent"

    if any(word in query for word in ["issue", "problem", "complaint", "category", "concern"]):
        return "issue_agent"

    if any(word in query for word in ["topic", "theme", "discussion", "trend"]):
        return "topic_agent"

    if any(word in query for word in ["keyword", "word", "phrase", "terms"]):
        return "keyword_agent"

    return "rag_qa_agent"


# =========================
# AGENTS
# =========================

def summarization_agent(query):
    summaries = pd.read_csv(SUMMARIES_PATH)

    if "negative" in query.lower():
        row = summaries[summaries["summary_type"] == "negative_feedback"]
    elif "positive" in query.lower():
        row = summaries[summaries["summary_type"] == "positive_feedback"]
    elif "battery" in query.lower():
        row = summaries[summaries["summary_title"].str.contains("Battery", case=False, na=False)]
    elif "ai" in query.lower() or "gemini" in query.lower():
        row = summaries[summaries["summary_title"].str.contains("AI", case=False, na=False)]
    elif "s-pen" in query.lower() or "spen" in query.lower() or "pen" in query.lower():
        row = summaries[summaries["summary_title"].str.contains("S-Pen", case=False, na=False)]
    elif "camera" in query.lower():
        row = summaries[summaries["summary_title"].str.contains("Camera", case=False, na=False)]
    elif "screen" in query.lower() or "display" in query.lower():
        row = summaries[summaries["summary_title"].str.contains("Display", case=False, na=False)]
    else:
        row = summaries[summaries["summary_type"] == "overall_feedback"]

    if row.empty:
        return "No matching summary found."

    return row.iloc[0]["summary"]


def sentiment_agent(query):
    df = pd.read_csv(COMMENTS_PATH)

    sentiment_counts = df["sentiment_label"].value_counts()
    total = len(df)

    result = "Sentiment Analysis Result:\n\n"

    for label, count in sentiment_counts.items():
        percentage = round((count / total) * 100, 2)
        result += f"- {label}: {count} comments ({percentage}%)\n"

    result += "\nInterpretation:\n"
    result += "This shows the overall emotional pattern in Samsung-related YouTube comments."

    return result


def issue_agent(query):
    df = pd.read_csv(COMMENTS_PATH)

    query_lower = query.lower()

    # Remove categories that are not real complaints
    complaint_df = df.copy()

    if any(word in query_lower for word in ["complaint", "complaints", "problem", "problems", "concern", "issues"]):
        complaint_df = complaint_df[
            ~complaint_df["issue_category"].isin([
                "Other",
                "Positive Feedback",
                "Non-English"
            ])
        ]

        # Focus more on negative/neutral comments for complaint-related queries
        complaint_df = complaint_df[
            complaint_df["sentiment_label"].isin(["negative", "neutral"])
        ]

        title = "Main Complaint Categories"
    else:
        complaint_df = complaint_df[
            ~complaint_df["issue_category"].isin([
                "Non-English"
            ])
        ]

        title = "Top Issue Categories"

    issue_counts = complaint_df["issue_category"].value_counts().head(10)

    result = f"{title}:\n\n"

    total = len(complaint_df)

    for issue, count in issue_counts.items():
        percentage = round((count / total) * 100, 2) if total > 0 else 0
        result += f"- {issue}: {count} comments ({percentage}%)\n"

    result += "\nInterpretation:\n"

    if "complaint" in query_lower or "problem" in query_lower or "concern" in query_lower:
        result += (
            "These are the most common complaint-related categories after removing "
            "general, positive, and non-English comments. This gives a cleaner view "
            "of actual user concerns."
        )
    else:
        result += (
            "These categories show the main Samsung-related discussion areas across "
            "the analyzed YouTube comments."
        )

    return result


def topic_agent(query):
    df = pd.read_csv(COMMENTS_PATH)

    topic_counts = df["topic_name"].value_counts().head(10)

    result = "Top Discussion Topics:\n\n"

    for topic, count in topic_counts.items():
        result += f"- {topic}: {count} comments\n"

    result += "\nInterpretation:\n"
    result += "These topics were discovered using topic modeling and show the main discussion themes."

    return result


def keyword_agent(query):
    keyword_path = "data/processed/top_keywords_overall.csv"

    if not os.path.exists(keyword_path):
        return "Keyword file not found. Please run keyword_extraction.py first."

    keywords = pd.read_csv(keyword_path).head(20)

    result = "Top Keywords:\n\n"

    for _, row in keywords.iterrows():
        result += f"- {row['keyword']} ({round(row['tfidf_score'], 2)})\n"

    return result


def rag_qa_agent(query):
    """
    For now, this reads already generated RAG answers.
    If query is not found exactly, it asks OpenAI to produce a short fallback answer
    from the closest existing RAG answer.
    """

    answers = pd.read_csv(RAG_ANSWERS_PATH)

    # Exact match
    match = answers[answers["query"].str.lower() == query.lower()]

    if not match.empty:
        return match.iloc[0]["answer"]

    # Fallback: use closest existing answers as context
    context = "\n\n".join(
        [
            f"Question: {row['query']}\nAnswer: {row['answer']}"
            for _, row in answers.iterrows()
        ]
    )

    system_prompt = """
You are an academic NLP assistant.
Answer only using the available RAG answer context.
If the exact question is not covered, say the available evidence is limited.
Keep the answer concise and professional.
"""

    user_prompt = f"""
User question:
{query}

Available RAG answer context:
{context}

Write a concise answer.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2,
        max_completion_tokens=1100,
    )

def strategy_rag_agent(query):
    """
    Routes product strategy questions to the Strategy RAG outputs.
    For now, it uses saved strategy_rag_results.csv.
    """

    if not os.path.exists(STRATEGY_RAG_PATH):
        return "Strategy RAG results not found. Please run strategy_rag.py first."

    strategy_df = pd.read_csv(STRATEGY_RAG_PATH)

    query_lower = query.lower()

    # Try exact match first
    exact_match = strategy_df[strategy_df["query"].str.lower() == query_lower]

    if not exact_match.empty:
        return exact_match.iloc[0]["answer"]

    # Route based on strategy goal
    if any(word in query_lower for word in ["profit", "revenue", "sales", "margin"]):
        match = strategy_df[strategy_df["strategy_goal"] == "profit"]

    elif any(word in query_lower for word in ["satisfaction", "happy", "customer", "loyalty", "reduce complaints"]):
        match = strategy_df[strategy_df["strategy_goal"] == "customer_satisfaction"]

    else:
        match = strategy_df[strategy_df["strategy_goal"] == "balanced"]

    if not match.empty:
        return match.iloc[0]["answer"]

    # Fallback if no goal match
    return strategy_df.iloc[0]["answer"]


# =========================
# MAIN ROUTING FUNCTION
# =========================

def run_agent(query):
    intent = detect_intent(query)

    if intent == "summarization_agent":
        answer = summarization_agent(query)
    elif intent == "sentiment_agent":
        answer = sentiment_agent(query)
    elif intent == "issue_agent":
        answer = issue_agent(query)
    elif intent == "topic_agent":
        answer = topic_agent(query)
    elif intent == "keyword_agent":
        answer = keyword_agent(query)
    elif intent == "strategy_rag_agent":
        answer = strategy_rag_agent(query)
    else:
        answer = rag_qa_agent(query)

    return {
        "user_query": query,
        "selected_agent": intent,
        "answer": answer
    }


def main():
    test_queries = [
        "Give me an overall summary of Samsung feedback.",
        "What is the sentiment distribution?",
        "What are the main complaints?",
        "What are the top discussion topics?",
        "What are the top keywords?",
        "Why are users unhappy about the S-Pen?",
        "What are users saying about Samsung battery life?",
        "How should Samsung design the S27 Ultra for maximum customer satisfaction?",
        "How should Samsung design the S27 Ultra for maximum profit?",
        "What product roadmap should Samsung follow for the next Ultra phone?"
    ]

    results = []

    for query in test_queries:
        print("\n====================================")
        print("User Query:", query)

        result = run_agent(query)

        print("Selected Agent:", result["selected_agent"])
        print("Answer:")
        print(result["answer"])

        results.append(result)

    output_df = pd.DataFrame(results)

    os.makedirs("data/processed", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nAgent routing completed!")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
