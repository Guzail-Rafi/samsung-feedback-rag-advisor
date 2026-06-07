import os
import json
import hashlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai_client import generate_chat_response, get_openai_client
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vector_store import (
    STRATEGY_COLLECTION,
    dataframe_metadata,
    load_or_create_collection,
    query_collection,
)


# =========================
# CONFIG
# =========================

load_dotenv()

INPUT_PATH = "data/processed/strategy_evidence.csv"
OUTPUT_PATH = "data/processed/strategy_rag_results.csv"
EMBEDDINGS_CACHE_PATH = "data/processed/strategy_evidence_embeddings.npy"
EMBEDDINGS_META_PATH = "data/processed/strategy_evidence_embeddings_meta.json"

client = get_openai_client()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# HELPERS
# =========================

def get_embedding_cache_key(texts):
    hasher = hashlib.sha256()

    for text in texts:
        hasher.update(str(text).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def load_or_create_embeddings(texts):
    cache_key = get_embedding_cache_key(texts)

    if os.path.exists(EMBEDDINGS_CACHE_PATH) and os.path.exists(EMBEDDINGS_META_PATH):
        with open(EMBEDDINGS_META_PATH, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)

        if metadata.get("cache_key") == cache_key and metadata.get("row_count") == len(texts):
            print("Loading cached strategy embeddings...")
            return np.load(EMBEDDINGS_CACHE_PATH)

    print("Creating strategy evidence embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)

    with open(EMBEDDINGS_META_PATH, "w", encoding="utf-8") as meta_file:
        json.dump(
            {
                "cache_key": cache_key,
                "row_count": len(texts),
                "model": "all-MiniLM-L6-v2"
            },
            meta_file,
            indent=2
        )

    return embeddings


def load_or_create_strategy_vector_store(df, embeddings):
    metadata_columns = [
        "issue_category",
        "sentiment_label",
        "goal_relevance",
        "priority",
        "business_impact",
        "engagement_total",
    ]

    return load_or_create_collection(
        collection_name=STRATEGY_COLLECTION,
        texts=df["strategy_text"].tolist(),
        embeddings=embeddings,
        metadatas=dataframe_metadata(df, metadata_columns),
    )


def detect_strategy_goal(query):
    query = query.lower()

    if any(word in query for word in ["profit", "revenue", "sales", "margin", "premium pricing"]):
        return "profit"

    if any(word in query for word in ["satisfaction", "happy", "customer", "loyalty", "user satisfaction"]):
        return "customer_satisfaction"

    return "balanced"


def goal_relevance_score(goal, row):
    row_goal = str(row.get("goal_relevance", "")).lower()
    sentiment = str(row.get("sentiment_label", "")).lower()
    issue = str(row.get("issue_category", "")).lower()

    score = 0.0

    if goal == "customer_satisfaction":
        if row_goal == "customer_satisfaction":
            score += 0.5
        if sentiment == "negative":
            score += 0.3
        if issue in ["battery / charging", "s-pen / features", "display / screen", "camera"]:
            score += 0.2

    elif goal == "profit":
        if row_goal == "profit":
            score += 0.5
        if issue in ["price / value", "camera", "ai / gemini", "performance / processor", "design / build"]:
            score += 0.3
        if sentiment == "positive":
            score += 0.2

    else:
        if row_goal == "balanced":
            score += 0.4
        if issue in ["battery / charging", "s-pen / features", "camera", "display / screen", "price / value"]:
            score += 0.4
        score += 0.2

    return min(score, 1.0)


def retrieve_strategy_evidence(
    query,
    goal,
    df,
    embeddings=None,
    top_k=12,
    vector_collection=None,
    candidate_count=900,
):
    query_embedding = embedding_model.encode([query])

    if vector_collection is not None:
        row_indices, similarities = query_collection(
            vector_collection,
            query_embedding,
            candidate_count,
        )
        df = df.iloc[row_indices].copy()
        df["strategy_similarity_score"] = similarities
    else:
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        df = df.copy()
        df["strategy_similarity_score"] = similarities

    df["engagement_score"] = np.log1p(df["engagement_total"].fillna(0).astype(float))

    if df["engagement_score"].max() > 0:
        df["engagement_score"] = df["engagement_score"] / df["engagement_score"].max()

    df["goal_relevance_score"] = df.apply(
        lambda row: goal_relevance_score(goal, row),
        axis=1
    )

    priority_map = {
        "High": 1.0,
        "Medium": 0.6,
        "Low": 0.3
    }

    df["priority_score"] = df["priority"].map(priority_map).fillna(0.5)

    # Strategy RAG formula:
    # prioritizes semantic relevance + goal fit + business priority + engagement
    df["strategy_retrieval_score"] = (
        0.45 * df["strategy_similarity_score"] +
        0.25 * df["goal_relevance_score"] +
        0.20 * df["priority_score"] +
        0.10 * df["engagement_score"]
    )

    df["strategy_similarity_score"] = df["strategy_similarity_score"].round(3)
    df["goal_relevance_score"] = df["goal_relevance_score"].round(3)
    df["priority_score"] = df["priority_score"].round(3)
    df["engagement_score"] = df["engagement_score"].round(3)
    df["strategy_retrieval_score"] = df["strategy_retrieval_score"].round(3)

    return df.sort_values(
        by="strategy_retrieval_score",
        ascending=False
    ).head(top_k)


def format_strategy_evidence(results):
    evidence = []

    for idx, row in results.reset_index(drop=True).iterrows():
        evidence.append(
            f"Evidence {idx + 1}:\n"
            f"Issue Category: {row['issue_category']}\n"
            f"Sentiment: {row['sentiment_label']}\n"
            f"User Comment: {row['clean_comment']}\n"
            f"Customer Signal: {row['customer_signal']}\n"
            f"Customer Recommendation: {row['customer_recommendation']}\n"
            f"Profit Recommendation: {row['profit_recommendation']}\n"
            f"Business Impact: {row['business_impact']}\n"
            f"Priority: {row['priority']}\n"
            f"Strategy Retrieval Score: {row['strategy_retrieval_score']}\n"
        )

    return "\n".join(evidence)


def generate_strategy_answer(query, goal, evidence_text):
    system_prompt = """
You are a product strategy advisor for a university NLP/RAG project.

You must answer only using the retrieved strategy evidence.
Do not invent unsupported claims.
Give practical product recommendations.
Clearly separate customer satisfaction logic from profit logic when needed.
Use professional business language.
Your answer should be useful for Samsung product planning.
You may create a phased implementation roadmap, 
but do not present it as Samsung's actual internal timeline. 
Use phases such as Phase 1, Phase 2, Phase 3, and Phase 4 
instead of exact calendar quarters unless timing evidence is provided.
"""

    user_prompt = f"""
User Strategy Question:
{query}

Detected Strategy Goal:
{goal}

Retrieved Strategy Evidence:
{evidence_text}

Write:
1. Direct strategic recommendation.
2. Top 5 feature/product priorities.
3. Reasoning from customer feedback evidence.
4. Expected impact on customer satisfaction or profit.
5. Risks or trade-offs.
6. Final S27 Ultra phased product roadmap recommendation using Phase 1, Phase 2, Phase 3, and Phase 4.
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


def run_strategy_rag(query, df, embeddings, vector_collection=None):
    goal = detect_strategy_goal(query)

    results = retrieve_strategy_evidence(
        query=query,
        goal=goal,
        df=df,
        embeddings=embeddings,
        top_k=12,
        vector_collection=vector_collection,
    )

    evidence_text = format_strategy_evidence(results)
    answer = generate_strategy_answer(query, goal, evidence_text)

    return {
        "query": query,
        "strategy_goal": goal,
        "answer": answer,
        "top_issue_1": results.iloc[0]["issue_category"],
        "top_issue_2": results.iloc[1]["issue_category"],
        "top_issue_3": results.iloc[2]["issue_category"],
        "avg_strategy_retrieval_score": round(results["strategy_retrieval_score"].mean(), 3),
        "avg_goal_relevance_score": round(results["goal_relevance_score"].mean(), 3)
    }


def refine_strategy_answer(original_query, goal, evidence_text, previous_answer, user_feedback):
    system_prompt = """
You are a product strategy advisor for a university NLP/RAG project.

You must revise the product roadmap using:
1. The original user strategy question
2. The retrieved strategy evidence
3. The previous roadmap answer
4. The user's new feedback or negotiation request

Do not ignore the retrieved evidence.
Do not invent unsupported product claims.
If the user's request is supported by evidence, incorporate it.
If the user's request is weakly supported, include it as a trade-off or optional consideration.
If the user's request conflicts with the evidence, explain the conflict politely and recommend a balanced alternative.

Use a phased roadmap format:
Phase 1: Immediate Priority
Phase 2: Feature Enhancement
Phase 3: Premium Positioning
Phase 4: Launch & Feedback Monitoring

Keep the answer professional and business-oriented.
"""

    user_prompt = f"""
Original Strategy Question:
{original_query}

Detected Strategy Goal:
{goal}

Retrieved Strategy Evidence:
{evidence_text}

Previous Strategy Answer:
{previous_answer}

User Feedback / Negotiation Request:
{user_feedback}

Revise the strategy roadmap.

Write:
1. What changed based on the user's feedback.
2. Updated phased roadmap.
3. Evidence-based justification.
4. Trade-offs or risks.
5. Final recommendation.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2,
        max_completion_tokens=900,
    )

def test_strategy_refinement(query, user_feedback, df, embeddings, vector_collection=None):
    goal = detect_strategy_goal(query)

    results = retrieve_strategy_evidence(
        query=query,
        goal=goal,
        df=df,
        embeddings=embeddings,
        top_k=12,
        vector_collection=vector_collection,
    )

    evidence_text = format_strategy_evidence(results)

    original_answer = generate_strategy_answer(
        query=query,
        goal=goal,
        evidence_text=evidence_text
    )

    refined_answer = refine_strategy_answer(
        original_query=query,
        goal=goal,
        evidence_text=evidence_text,
        previous_answer=original_answer,
        user_feedback=user_feedback
    )

    return {
        "original_query": query,
        "strategy_goal": goal,
        "user_feedback": user_feedback,
        "original_answer": original_answer,
        "refined_answer": refined_answer,
        "avg_strategy_retrieval_score": round(results["strategy_retrieval_score"].mean(), 3),
        "avg_goal_relevance_score": round(results["goal_relevance_score"].mean(), 3)
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["strategy_text"]).copy()
    df = df.reset_index(drop=True)

    print("Total strategy evidence rows:", len(df))

    embeddings = load_or_create_embeddings(df["strategy_text"].tolist())
    vector_collection = load_or_create_strategy_vector_store(df, embeddings)

    strategy_queries = [
        "How should Samsung design the S27 Ultra for maximum customer satisfaction?",
        "How should Samsung design the S27 Ultra for maximum profit?",
        "What features should Samsung prioritize in the S27 Ultra?",
        "What product roadmap should Samsung follow for the next Ultra phone?",
        "How can Samsung reduce customer complaints in the next flagship?"
    ]

    rows = []

    for query in strategy_queries:
        print("\n====================================")
        print("Strategy Query:", query)
        print("====================================")

        result = run_strategy_rag(query, df, embeddings, vector_collection)

        print("Detected Goal:", result["strategy_goal"])
        print("Answer:")
        print(result["answer"])

        rows.append(result)

    output_df = pd.DataFrame(rows)

    os.makedirs("data/processed", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nStrategy RAG completed!")
    print("Saved to:", OUTPUT_PATH)

        # =========================
    # TEST STRATEGY REFINEMENT
    # =========================

    refinement_tests = [
        {
            "query": "How should Samsung design the S27 Ultra for maximum customer satisfaction?",
            "user_feedback": "No, Phase 1 should also include camera improvement because creators care about camera quality."
        },
        {
            "query": "How should Samsung design the S27 Ultra for maximum profit?",
            "user_feedback": "Make the roadmap more profit-focused and move premium AI features earlier."
        },
        {
            "query": "How can Samsung reduce customer complaints in the next flagship?",
            "user_feedback": "Add display durability and green line prevention to Phase 1."
        }
    ]

    refinement_rows = []

    for test in refinement_tests:
        print("\n====================================")
        print("Refinement Test Query:", test["query"])
        print("User Feedback:", test["user_feedback"])
        print("====================================")

        refined_result = test_strategy_refinement(
            query=test["query"],
            user_feedback=test["user_feedback"],
            df=df,
            embeddings=embeddings,
            vector_collection=vector_collection,
        )

        print("Refined Answer:")
        print(refined_result["refined_answer"])

        refinement_rows.append(refined_result)

    refinement_df = pd.DataFrame(refinement_rows)
    refinement_output_path = "data/processed/strategy_refinement_results.csv"
    refinement_df.to_csv(refinement_output_path, index=False, encoding="utf-8-sig")

    print("\nStrategy refinement completed!")
    print("Saved to:", refinement_output_path)


if __name__ == "__main__":
    main()
