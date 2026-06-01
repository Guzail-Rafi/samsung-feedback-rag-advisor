import os
import re
import pandas as pd


INPUT_PATH = "data/processed/rag_retrieval_results.csv"
OUTPUT_PATH = "data/processed/rag_evaluation_results.csv"


# Manual relevance labels for the current top 5 retrieved comments per query.
# Keep these labels in sync whenever rag_retrieval_results.csv is regenerated.
# 1 = relevant, 0 = not relevant
MANUAL_LABELS = {
    "What are users saying about Samsung battery life?": [1, 1, 1, 1, 1],
    "Why are users unhappy about the S-Pen?": [1, 1, 1, 1, 1],
    "What do users think about Galaxy AI and Gemini?": [1, 1, 1, 1, 1],
    "Are users comparing Samsung with Apple?": [1, 1, 1, 1, 1],
    "What are users saying about Samsung camera quality?": [1, 0, 1, 1, 1],
    "What are users saying about Samsung screen or display issues?": [1, 1, 1, 0, 1],
}

QUERY_PROFILES = [
    {
        "triggers": ["battery", "charging", "charge"],
        "required_terms": ["battery", "charging", "charge", "charger", "mah"],
    },
    {
        "triggers": ["s pen", "spen", "stylus"],
        "required_terms": ["s pen", "spen", "stylus", "pen"],
        "requires_complaint_signal": True,
    },
    {
        "triggers": ["galaxy ai", "gemini", "ai assistant", "assistant", "ai"],
        "required_terms": ["galaxy ai", "gemini", "ai", "assistant", "bixby"],
    },
    {
        "triggers": ["apple", "iphone", "ios"],
        "required_terms": ["apple", "iphone", "ios", "android"],
    },
    {
        "triggers": ["camera", "photo", "photos", "zoom", "lens"],
        "required_terms": ["camera", "photo", "photos", "zoom", "lens"],
        "quality_terms": ["quality", "dslr", "better", "terrible", "fake", "decent", "worst"],
    },
    {
        "triggers": ["screen", "display", "brightness", "protector"],
        "required_terms": ["screen", "display", "brightness", "protector", "glass"],
        "requires_complaint_signal": True,
    },
]

COMPLAINT_SIGNAL_TERMS = [
    "abysmal", "bad", "blind", "break", "can't tell", "cant tell",
    "complain", "complaining", "crap", "damage", "dealbreaker",
    "difficult", "disappointed", "disgusting", "fake", "hard pass",
    "hate", "horrible", "issue", "pathetic", "problem", "red flag",
    "remove", "removed", "ridiculous", "scratch", "terrible", "weak",
    "without", "worse", "worst",
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


def row_text(row):
    return " ".join([
        normalize_text(row.get("clean_comment", "")),
        normalize_text(row.get("issue_category", "")),
        normalize_text(row.get("topic_name", "")),
    ])


def infer_query_profile(query):
    for profile in QUERY_PROFILES:
        if contains_any(query, profile["triggers"]):
            return profile

    return {"required_terms": []}


def heuristic_relevance_label(query, row):
    """
    Fallback only for new queries that do not have manual labels.

    This intentionally does not use weighted_retrieval_score because that score
    is produced by the retriever being evaluated.
    """

    profile = infer_query_profile(query)
    text = row_text(row)

    if not profile["required_terms"]:
        similarity_score = float(row.get("similarity_score", 0) or 0)
        lexical_score = float(row.get("lexical_relevance_score", 0) or 0)
        return int(similarity_score >= 0.70 and lexical_score >= 0.50)

    has_required_term = contains_any(text, profile["required_terms"])
    if not has_required_term:
        return 0

    if profile.get("quality_terms") and not contains_any(text, profile["quality_terms"]):
        return 0

    if profile.get("requires_complaint_signal") and not contains_any(text, COMPLAINT_SIGNAL_TERMS):
        return 0

    return 1


def precision_at_k(labels, k=5):
    labels_at_k = labels[:k]

    if len(labels_at_k) == 0:
        return 0

    return sum(labels_at_k) / k


def average_score_if_present(df, column):
    if column not in df.columns:
        return None

    return round(df[column].mean(), 3)


def get_labels(query, query_results):
    if query in MANUAL_LABELS:
        return MANUAL_LABELS[query][:5], "manual"

    labels = query_results.apply(
        lambda row: heuristic_relevance_label(query, row),
        axis=1
    ).tolist()

    return labels, "heuristic"


def main():
    df = pd.read_csv(INPUT_PATH)

    evaluation_rows = []
    queries = df["query"].dropna().unique()

    for query in queries:
        query_results = df[df["query"] == query].head(5).copy()

        if len(query_results) < 5:
            print(f"Warning: Less than 5 results found for query: {query}")
            continue

        labels, label_source = get_labels(query, query_results)
        p_at_5 = precision_at_k(labels, k=5)

        evaluation_rows.append({
            "query": query,
            "label_source": label_source,
            "relevance_labels_top_5": labels,
            "relevant_results": sum(labels),
            "total_results": 5,
            "precision_at_5": round(p_at_5, 2),
            "avg_weighted_score": round(query_results["weighted_retrieval_score"].mean(), 3),
            "avg_similarity_score": round(query_results["similarity_score"].mean(), 3),
            "avg_category_relevance_score": average_score_if_present(query_results, "category_relevance_score"),
            "avg_lexical_relevance_score": average_score_if_present(query_results, "lexical_relevance_score"),
            "avg_sentiment_relevance_score": average_score_if_present(query_results, "sentiment_relevance_score"),
            "avg_engagement_score": average_score_if_present(query_results, "engagement_score"),
        })

    eval_df = pd.DataFrame(evaluation_rows)

    os.makedirs("data/processed", exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("RAG evaluation completed!")
    print("Saved to:", OUTPUT_PATH)

    print("\nEvaluation results:")
    print(eval_df)

    print("\nOverall Precision@5:")
    print(round(eval_df["precision_at_5"].mean(), 2))


if __name__ == "__main__":
    main()
