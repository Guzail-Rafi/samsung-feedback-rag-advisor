import json
import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import Normalizer

from mlflow_tracing import PIPELINE_EXPERIMENT_NAME, configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "comments_with_ner.csv"
ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "user_segmentation_assignments.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "user_personas.csv"
EVALUATION_PATH = PROJECT_ROOT / "data" / "processed" / "user_segmentation_evaluation.csv"
DASHBOARD_PATH = PROJECT_ROOT / "data" / "processed" / "user_segmentation_dashboard.json"

RANDOM_STATE = 42
MAX_FEATURES = 7000
SVD_COMPONENTS = 50
K_VALUES = range(4, 9)
SILHOUETTE_SAMPLE = 5000

PERSONA_PATTERNS = [
    (
        "S-Pen Bluetooth Feature Loyalists",
        {"bluetooth pen", "pen bluetooth", "spen", "removed", "removing", "use pen"},
        "Protect distinctive Ultra productivity features, especially advanced S-Pen controls.",
    ),
    (
        "Camera-Focused Enthusiasts",
        {"camera", "photo", "zoom", "lens", "telephoto", "sensor"},
        "Prioritize camera consistency, zoom, low-light performance, and creator workflows.",
    ),
    (
        "Battery and Performance Seekers",
        {"battery", "charging", "charge", "performance", "processor", "snapdragon", "heat"},
        "Improve real-world endurance, charging, thermal efficiency, and sustained performance.",
    ),
    (
        "Value-Conscious Upgrade Sceptics",
        {"price", "value", "upgrade", "expensive", "cost", "deal", "worth", "same"},
        "Prove upgrade value clearly and use trade-in or bundle offers without weakening premium positioning.",
    ),
    (
        "iPhone Comparison Shoppers",
        {"iphone 17", "iphone 16", "iphone 15", "pro max", "better iphone", "iphone"},
        "Emphasize Samsung ecosystem advantages, switching ease, and meaningful competitor differentiation.",
    ),
    (
        "Samsung-Apple Brand Debaters",
        {"samsung apple", "apple samsung", "like apple", "love samsung", "samsung phones", "innovation"},
        "Preserve Samsung's distinct product identity and communicate meaningful differentiation from Apple.",
    ),
    (
        "Galaxy AI Subscription Sceptics",
        {"ai features", "galaxy ai", "ai phone", "pay", "free", "gemini"},
        "Demonstrate useful Galaxy AI outcomes and clarify future subscription pricing before asking users to commit.",
    ),
    (
        "Galaxy Upgrade-Cycle Followers",
        {"s26 ultra", "s25 ultra", "s24 ultra", "s23 ultra", "s22 ultra", "s26", "s25", "s24", "s23", "s22"},
        "Make generation-to-generation improvements explicit and avoid feature removals that make new Ultra models feel incremental.",
    ),
    (
        "Phone Buying and Value Seekers",
        {"best phone", "phone year", "new phone", "buy phone", "samsung phone"},
        "Translate flagship capabilities into clear buying reasons, value comparisons, and upgrade guidance.",
    ),
    (
        "Fake Product and Authenticity Watchers",
        {"fake samsung", "fake phone", "buy fake", "fake better", "fake actually", "fake"},
        "Improve authenticity education and explain why genuine Samsung devices justify their premium.",
    ),
]


def clean_text(value):
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_comments():
    df = pd.read_csv(INPUT_PATH)
    required = ["clean_comment", "sentiment_label", "issue_category", "topic_name"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required segmentation columns: {missing}")

    df = df.dropna(subset=["clean_comment"]).copy()
    df["clean_comment"] = df["clean_comment"].map(clean_text)
    df = df[df["clean_comment"].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=["clean_comment"]).reset_index(drop=True)
    return df


def evaluate_k(reduced, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(reduced)
    counts = np.bincount(labels)
    return {
        "k": k,
        "silhouette_score": round(
            float(
                silhouette_score(
                    reduced,
                    labels,
                    sample_size=min(SILHOUETTE_SAMPLE, len(reduced)),
                    random_state=RANDOM_STATE,
                )
            ),
            4,
        ),
        "calinski_harabasz_score": round(float(calinski_harabasz_score(reduced, labels)), 3),
        "inertia": round(float(model.inertia_), 3),
        "smallest_cluster": int(counts.min()),
        "largest_cluster": int(counts.max()),
        "smallest_cluster_share": round(float(counts.min() / len(labels)), 4),
        "model": model,
        "labels": labels,
    }


def choose_best_result(results):
    viable = [result for result in results if result["smallest_cluster_share"] >= 0.025]
    candidates = viable or results
    return max(candidates, key=lambda result: (result["silhouette_score"], -result["k"]))


def top_cluster_terms(tfidf_matrix, labels, feature_names, cluster_id, top_n=12):
    cluster_mean = np.asarray(tfidf_matrix[labels == cluster_id].mean(axis=0)).ravel()
    overall_mean = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    distinctive = cluster_mean - overall_mean
    indices = distinctive.argsort()[::-1]
    return [feature_names[index] for index in indices if distinctive[index] > 0][:top_n]


def persona_from_signals(top_terms, top_issue, top_topic, sentiment):
    term_text = " | ".join(top_terms).lower()
    context_text = f"{top_issue} | {top_topic}".lower()
    scored = []
    for name, signals, recommendation in PERSONA_PATTERNS:
        term_hits = sum(1 for signal in signals if signal in term_text)
        context_hits = sum(1 for signal in signals if signal in context_text)
        score = (term_hits * 4) + context_hits if term_hits >= 2 else 0
        scored.append((score, term_hits, name, recommendation))

    score, _, name, recommendation = max(scored)
    if score == 0:
        if top_issue != "Other":
            name = f"{top_issue} Experience Discussants"
            focus = top_issue.lower()
        else:
            name = "General Mobile Discussion Audience"
            focus = "general mobile experience"
        recommendation = f"Use this segment's {focus} signals as supporting context rather than treating them as a single strong product demand."

    if sentiment == "negative" and "Critics" not in name and "Sceptics" not in name:
        name = f"{name} with Unresolved Concerns"

    return name, recommendation


def ensure_unique_personas(summary_df):
    duplicate_names = summary_df["persona"].value_counts()
    for name, count in duplicate_names.items():
        if count <= 1:
            continue
        matching = summary_df.index[summary_df["persona"] == name]
        for index in matching:
            qualifier = summary_df.at[index, "top_issue"]
            if qualifier == "Other":
                qualifier = summary_df.at[index, "top_topic"]
            summary_df.at[index, "persona"] = f"{name} - {qualifier}"
    return summary_df


def representative_comments(df, reduced, model, cluster_id, count=3):
    indices = np.flatnonzero(df["cluster_id"].to_numpy() == cluster_id)
    distances = np.linalg.norm(reduced[indices] - model.cluster_centers_[cluster_id], axis=1)
    selected = indices[np.argsort(distances)[:count]]
    return [df.iloc[index]["clean_comment"][:350] for index in selected]


def build_cluster_summaries(df, tfidf_matrix, reduced, model, labels, vectorizer):
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    summaries = []
    total = len(df)

    for cluster_id in sorted(np.unique(labels)):
        cluster = df[df["cluster_id"] == cluster_id]
        terms = top_cluster_terms(tfidf_matrix, labels, feature_names, cluster_id)
        sentiment_counts = cluster["sentiment_label"].value_counts()
        issue_counts = cluster["issue_category"].value_counts()
        topic_counts = cluster["topic_name"].value_counts()
        top_sentiment = str(sentiment_counts.index[0])
        top_issue = str(issue_counts.index[0])
        top_topic = str(topic_counts.index[0])
        persona, recommendation = persona_from_signals(terms, top_issue, top_topic, top_sentiment)

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "persona": persona,
                "size": int(len(cluster)),
                "share": round(float(len(cluster) / total), 4),
                "dominant_sentiment": top_sentiment,
                "sentiment_share": round(float(sentiment_counts.iloc[0] / len(cluster)), 4),
                "top_issue": top_issue,
                "top_issue_share": round(float(issue_counts.iloc[0] / len(cluster)), 4),
                "top_topic": top_topic,
                "top_terms": ", ".join(terms),
                "recommendation": recommendation,
                "representative_comments": json.dumps(
                    representative_comments(df, reduced, model, cluster_id),
                    ensure_ascii=True,
                ),
            }
        )

    return ensure_unique_personas(pd.DataFrame(summaries))


def dashboard_payload(df, summary_df, evaluation_df, best_result, vectorizer, svd):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(1800, len(df))
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)
    points = [
        {
            "x": round(float(df.iloc[index]["svd_x"]), 4),
            "y": round(float(df.iloc[index]["svd_y"]), 4),
            "cluster_id": int(df.iloc[index]["cluster_id"]),
            "persona": str(df.iloc[index]["persona"]),
            "sentiment": str(df.iloc[index]["sentiment_label"]),
        }
        for index in sample_indices
    ]
    personas = []
    for row in summary_df.to_dict(orient="records"):
        row["representative_comments"] = json.loads(row["representative_comments"])
        personas.append(row)

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "method": {
            "representation": "TF-IDF unigrams, bigrams, and trigrams",
            "max_features": MAX_FEATURES,
            "dimensionality_reduction": "TruncatedSVD",
            "svd_components": int(svd.n_components),
            "clustering": "KMeans",
            "selected_k": int(best_result["k"]),
            "random_state": RANDOM_STATE,
        },
        "metrics": {
            "comments_segmented": int(len(df)),
            "tfidf_features": int(len(vectorizer.get_feature_names_out())),
            "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
            "silhouette_score": best_result["silhouette_score"],
            "calinski_harabasz_score": best_result["calinski_harabasz_score"],
        },
        "personas": personas,
        "evaluation": evaluation_df.to_dict(orient="records"),
        "scatter_points": points,
    }


def main():
    df = load_comments()
    print(f"Segmenting {len(df):,} unique English comments...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=MAX_FEATURES,
        min_df=5,
        max_df=0.92,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["clean_comment"])

    components = min(SVD_COMPONENTS, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
    reduced = Normalizer(copy=False).fit_transform(svd.fit_transform(tfidf_matrix))

    results = [evaluate_k(reduced, k) for k in K_VALUES]
    best_result = choose_best_result(results)
    model = best_result["model"]
    labels = best_result["labels"]
    df["cluster_id"] = labels
    df["svd_x"] = reduced[:, 0]
    df["svd_y"] = reduced[:, 1]

    summary_df = build_cluster_summaries(df, tfidf_matrix, reduced, model, labels, vectorizer)
    persona_map = summary_df.set_index("cluster_id")["persona"].to_dict()
    df["persona"] = df["cluster_id"].map(persona_map)

    evaluation_df = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key not in {"model", "labels"}}
            for result in results
        ]
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)
    evaluation_df.to_csv(EVALUATION_PATH, index=False)
    df[
        [
            "comment_id",
            "clean_comment",
            "sentiment_label",
            "issue_category",
            "topic_name",
            "cluster_id",
            "persona",
            "svd_x",
            "svd_y",
        ]
    ].to_csv(ASSIGNMENTS_PATH, index=False)
    DASHBOARD_PATH.write_text(
        json.dumps(
            dashboard_payload(df, summary_df, evaluation_df, best_result, vectorizer, svd),
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    configure_mlflow(PIPELINE_EXPERIMENT_NAME)
    with mlflow.start_run(
        run_name="TFIDF_SVD_KMeans_User_Segmentation",
        tags={"system": "samsung-rag", "run_type": "user-segmentation"},
    ):
        mlflow.log_param("representation", "TF-IDF")
        mlflow.log_param("ngram_range", "1-3")
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("dimensionality_reduction", "TruncatedSVD")
        mlflow.log_param("svd_components", components)
        mlflow.log_param("clustering", "KMeans")
        mlflow.log_param("selected_k", best_result["k"])
        mlflow.log_metric("comments_segmented", len(df))
        mlflow.log_metric("silhouette_score", best_result["silhouette_score"])
        mlflow.log_metric("calinski_harabasz_score", best_result["calinski_harabasz_score"])
        mlflow.log_metric("svd_explained_variance", float(svd.explained_variance_ratio_.sum()))
        for path in [SUMMARY_PATH, EVALUATION_PATH, DASHBOARD_PATH]:
            mlflow.log_artifact(str(path))

    print(f"Selected k={best_result['k']} with silhouette={best_result['silhouette_score']}.")
    print(summary_df[["cluster_id", "persona", "size", "dominant_sentiment", "top_issue"]].to_string(index=False))
    print("Saved:", SUMMARY_PATH)
    print("Saved:", EVALUATION_PATH)
    print("Saved:", ASSIGNMENTS_PATH)
    print("Saved:", DASHBOARD_PATH)


if __name__ == "__main__":
    main()
