import os
import json
import hashlib
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai_client import generate_chat_response, get_openai_client
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. CONFIG
# =========================

load_dotenv()

INPUT_PATH = "data/processed/comments_with_ner.csv"
OUTPUT_PATH = "data/processed/rag_answers.csv"
EMBEDDINGS_CACHE_PATH = "data/processed/rag_comment_embeddings.npy"
EMBEDDINGS_META_PATH = "data/processed/rag_comment_embeddings_meta.json"

client = get_openai_client()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

RETRIEVAL_WEIGHTS = {
    "semantic": 0.58,
    "category": 0.22,
    "lexical": 0.12,
    "sentiment": 0.05,
    "engagement": 0.03,
}

QUERY_INTENTS = [
    {
        "triggers": ["battery", "charging", "charge"],
        "categories": ["battery charging"],
        "terms": ["battery", "battery life", "charging", "charge", "charger", "capacity", "5000mah", "mah"],
    },
    {
        "triggers": ["s pen", "spen", "stylus"],
        "categories": ["s pen features"],
        "terms": ["s pen", "spen", "stylus", "pen", "bluetooth", "air action", "support", "remove", "removed"],
    },
    {
        "triggers": ["camera", "photo", "photos", "zoom", "lens", "portrait"],
        "categories": ["camera"],
        "terms": ["camera", "photo", "photos", "video", "zoom", "lens", "portrait", "quality", "front camera"],
    },
    {
        "triggers": ["screen", "display", "crease", "brightness"],
        "categories": ["display screen"],
        "terms": ["screen", "display", "crease", "inner screen", "brightness", "green line", "glass"],
    },
    {
        "triggers": ["galaxy ai", "gemini", "ai assistant", "assistant", "ai"],
        "categories": ["ai gemini"],
        "terms": ["galaxy ai", "gemini", "ai", "assistant", "bixby", "one ui", "google"],
    },
    {
        "triggers": ["apple", "iphone", "ios"],
        "categories": [],
        "terms": ["apple", "iphone", "ios", "android", "ecosystem", "duopoly", "copy", "copied"],
    },
    {
        "triggers": ["price", "expensive", "cost", "value"],
        "categories": ["price value"],
        "terms": ["price", "expensive", "cost", "value", "worth", "1200", "2000"],
    },
]

COMPLAINT_QUERY_TERMS = [
    "complaint", "complaints", "unhappy", "issue", "issues", "problem",
    "problems", "concern", "concerns", "bad", "hate", "negative",
]

COMPLAINT_SIGNAL_TERMS = [
    "abysmal", "awful", "bad", "broken", "concern", "dealbreaker",
    "disappointed", "damage", "difficult", "fake", "hard pass", "hate",
    "horrible", "issue", "lack", "missing", "no support", "no s pen",
    "not much", "problem", "red flag", "remove", "removed", "removing",
    "ridiculous", "terrible", "weakest", "without", "worse", "worst",
    "bezel", "bezels", "blind", "break", "can't tell", "cant tell",
    "chin", "crap", "scratch",
]

ANTI_COMPLAINT_PHRASES = [
    "not too terrible", "isn't too terrible", "isnt too terrible",
    "decent camera", "absolute win", "don't use the camera",
    "dont use the camera", "don't use cameras", "dont use cameras",
]


def normalize_text(value):
    text = str(value).lower()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_term(text, term):
    text = normalize_text(text)
    term = normalize_text(term)

    if not text or not term:
        return False

    pattern = r"(?<![a-z0-9])" + r"\s+".join(
        re.escape(part) for part in term.split()
    ) + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def contains_any(text, terms):
    return any(contains_term(text, term) for term in terms)


def infer_query_intent(query):
    query_text = normalize_text(query)
    categories = set()
    terms = set()

    for intent in QUERY_INTENTS:
        if contains_any(query_text, intent["triggers"]):
            categories.update(intent["categories"])
            terms.update(intent["terms"])

    return {
        "categories": categories,
        "terms": terms,
        "is_complaint_query": contains_any(query_text, COMPLAINT_QUERY_TERMS),
    }


def row_search_text(row):
    return " ".join([
        normalize_text(row.get("clean_comment", "")),
        normalize_text(row.get("issue_category", "")),
        normalize_text(row.get("topic_name", "")),
        normalize_text(row.get("video_title", "")),
    ])


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
            print("Loading cached comment embeddings...")
            return np.load(EMBEDDINGS_CACHE_PATH)

    print("Creating embeddings...")
    comment_embeddings = embedding_model.encode(texts, show_progress_bar=True)

    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    np.save(EMBEDDINGS_CACHE_PATH, comment_embeddings)

    with open(EMBEDDINGS_META_PATH, "w", encoding="utf-8") as meta_file:
        json.dump(
            {
                "cache_key": cache_key,
                "row_count": len(texts),
                "model": "all-MiniLM-L6-v2",
            },
            meta_file,
            indent=2,
        )

    return comment_embeddings


# =========================
# 2. BUILD RAG TEXT
# =========================

def build_comment_text(row):
    return (
        f"Comment: {row['clean_comment']} "
        f"Issue Category: {row['issue_category']} "
        f"Sentiment: {row['sentiment_label']} "
        f"Topic: {row['topic_name']} "
        f"Video: {row['video_title']}"
    )


# =========================
# 3. CATEGORY RELEVANCE
# =========================

def calculate_category_relevance(query, row, intent=None):
    """
    Gives a boost when the query matches issue category, topic name,
    or important Samsung-related terms inside the comment.
    """

    if intent is None:
        intent = infer_query_intent(query)

    issue = normalize_text(row["issue_category"])
    topic = normalize_text(row["topic_name"])
    text = normalize_text(row["clean_comment"])
    searchable = row_search_text(row)

    score = 0.0

    for category in intent["categories"]:
        if contains_term(issue, category):
            score += 0.65
        elif contains_term(topic, category):
            score += 0.45
        elif contains_term(searchable, category):
            score += 0.25

    if contains_any(query, ["apple", "iphone", "ios"]):
        if contains_any(topic, ["apple", "iphone", "ios"]):
            score += 0.55
        if contains_any(text, ["apple", "iphone", "ios"]):
            score += 0.35

    return min(score, 1.0)


def calculate_lexical_relevance(query, row, intent=None):
    if intent is None:
        intent = infer_query_intent(query)

    terms = sorted(intent["terms"])
    if not terms:
        return 0.0

    searchable = row_search_text(row)
    matches = sum(1 for term in terms if contains_term(searchable, term))
    target_matches = min(4, len(terms))

    return min(matches / target_matches, 1.0)


def calculate_sentiment_relevance(query, row, intent=None):
    if intent is None:
        intent = infer_query_intent(query)

    if not intent["is_complaint_query"]:
        return 0.5

    text = normalize_text(row["clean_comment"])
    sentiment = normalize_text(row["sentiment_label"])
    has_complaint_signal = contains_any(text, COMPLAINT_SIGNAL_TERMS)

    if sentiment == "positive" and contains_any(text, ANTI_COMPLAINT_PHRASES):
        return 0.0

    if sentiment == "negative" and has_complaint_signal:
        return 1.0

    if has_complaint_signal:
        return 0.8

    if sentiment == "negative":
        return 0.55

    if sentiment == "neutral":
        return 0.25

    return 0.0


def calculate_intent_penalty(query, row, intent=None):
    if intent is None:
        intent = infer_query_intent(query)

    if not intent["is_complaint_query"]:
        return 0.0

    sentiment_relevance = calculate_sentiment_relevance(query, row, intent)
    category_relevance = calculate_category_relevance(query, row, intent)
    lexical_relevance = calculate_lexical_relevance(query, row, intent)
    has_complaint_signal = contains_any(row["clean_comment"], COMPLAINT_SIGNAL_TERMS)

    if category_relevance >= 0.5 and lexical_relevance >= 0.25 and not has_complaint_signal:
        if sentiment_relevance == 0:
            return 0.12
        return 0.08

    return 0.0


# =========================
# 4. RETRIEVE EVIDENCE
# =========================

def retrieve_comments(query, df, comment_embeddings, top_k=8):
    intent = infer_query_intent(query)
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, comment_embeddings)[0]

    df = df.copy()
    df["similarity_score"] = similarities

    df["engagement_score"] = np.log1p(
        df["like_count"].fillna(0).astype(float) +
        df["reply_count"].fillna(0).astype(float)
    )

    max_engagement = df["engagement_score"].max()
    if max_engagement > 0:
        df["engagement_score"] = df["engagement_score"] / max_engagement

    df["category_relevance_score"] = df.apply(
        lambda row: calculate_category_relevance(query, row, intent),
        axis=1
    )

    df["lexical_relevance_score"] = df.apply(
        lambda row: calculate_lexical_relevance(query, row, intent),
        axis=1
    )

    df["sentiment_relevance_score"] = df.apply(
        lambda row: calculate_sentiment_relevance(query, row, intent),
        axis=1
    )

    df["intent_penalty_score"] = df.apply(
        lambda row: calculate_intent_penalty(query, row, intent),
        axis=1
    )

    df["weighted_retrieval_score"] = (
        RETRIEVAL_WEIGHTS["semantic"] * df["similarity_score"] +
        RETRIEVAL_WEIGHTS["category"] * df["category_relevance_score"] +
        RETRIEVAL_WEIGHTS["lexical"] * df["lexical_relevance_score"] +
        RETRIEVAL_WEIGHTS["sentiment"] * df["sentiment_relevance_score"] +
        RETRIEVAL_WEIGHTS["engagement"] * df["engagement_score"] -
        df["intent_penalty_score"]
    )

    df["similarity_score"] = df["similarity_score"].round(3)
    df["engagement_score"] = df["engagement_score"].round(3)
    df["category_relevance_score"] = df["category_relevance_score"].round(3)
    df["lexical_relevance_score"] = df["lexical_relevance_score"].round(3)
    df["sentiment_relevance_score"] = df["sentiment_relevance_score"].round(3)
    df["intent_penalty_score"] = df["intent_penalty_score"].round(3)
    df["weighted_retrieval_score"] = df["weighted_retrieval_score"].round(3)

    results = df.sort_values(
        by="weighted_retrieval_score",
        ascending=False
    ).head(top_k)

    return results


# =========================
# 5. CONFIDENCE SCORE
# =========================

def calculate_rag_confidence(results):
    avg_score = results["weighted_retrieval_score"].mean()

    if avg_score >= 0.55:
        return "High"
    elif avg_score >= 0.35:
        return "Medium"
    else:
        return "Low"


# =========================
# 6. FORMAT EVIDENCE
# =========================

def format_evidence(results):
    evidence_blocks = []

    for idx, row in results.reset_index(drop=True).iterrows():
        evidence_blocks.append(
            f"Evidence {idx + 1}:\n"
            f"Comment: {row['clean_comment']}\n"
            f"Sentiment: {row['sentiment_label']}\n"
            f"Issue Category: {row['issue_category']}\n"
            f"Topic: {row['topic_name']}\n"
            f"Video: {row['video_title']}\n"
            f"Weighted Score: {row['weighted_retrieval_score']}\n"
        )

    return "\n".join(evidence_blocks)


# =========================
# 7. GENERATE OPENAI ANSWER
# =========================

def generate_answer(query, evidence_text, confidence):
    system_prompt = """
You are an academic NLP assistant for a university project.

You must answer only using the provided YouTube comment evidence.
Do not invent facts.
Do not overgeneralize beyond the retrieved evidence.
Use phrases such as "based on the retrieved comments" or "the evidence suggests".
If the evidence is mixed, clearly say it is mixed.
If the evidence is limited or indirect, clearly say that.
Do not say the whole user community believes something unless most retrieved evidence supports it.
Keep the answer concise, balanced, and professional.
Do not use phrases like "generally positive" or "majority of users" unless the retrieved evidence clearly supports it.
"""

    user_prompt = f"""
Question:
{query}

Retrieved YouTube Comment Evidence:
{evidence_text}

RAG Confidence:
{confidence}

Write:
1. A direct answer based only on the retrieved evidence.
2. Key evidence-based points.
3. A short confidence explanation, including whether the evidence is strong, mixed, or limited.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )


# =========================
# 8. MAIN
# =========================

def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["language"] == "en"].copy()
    df = df.dropna(subset=["clean_comment"])
    df = df[df["word_count"] >= 3].copy()
    df = df.reset_index(drop=True)

    print("Total comments used for RAG:", len(df))

    print("Building RAG text...")
    df["rag_text"] = df.apply(build_comment_text, axis=1)

    comment_embeddings = load_or_create_embeddings(df["rag_text"].tolist())

    test_queries = [
        "What are users saying about Samsung battery life?",
        "Why are users unhappy about the S-Pen?",
        "What do users think about Galaxy AI and Gemini?",
        "Are users comparing Samsung with Apple?",
        "What are users saying about Samsung camera quality?",
        "What are users saying about Samsung screen or display issues?"
    ]

    answer_rows = []

    for query in test_queries:
        print("\n====================================")
        print("Question:", query)
        print("====================================")

        results = retrieve_comments(
            query=query,
            df=df,
            comment_embeddings=comment_embeddings,
            top_k=8
        )

        confidence = calculate_rag_confidence(results)
        evidence_text = format_evidence(results)
        answer = generate_answer(query, evidence_text, confidence)

        print("\nAnswer:")
        print(answer)

        answer_rows.append({
            "query": query,
            "rag_confidence": confidence,
            "answer": answer,
            "top_evidence_1": results.iloc[0]["clean_comment"],
            "top_evidence_2": results.iloc[1]["clean_comment"],
            "top_evidence_3": results.iloc[2]["clean_comment"],
            "avg_weighted_score": round(results["weighted_retrieval_score"].mean(), 3)
        })  

    output_df = pd.DataFrame(answer_rows)

    os.makedirs("data/processed", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nRAG answer generation completed!")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
