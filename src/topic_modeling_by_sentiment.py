import os

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


INPUT_PATH = "data/processed/comments_with_categories.csv"
OUTPUT_COMMENTS_PATH = "data/processed/comments_with_sentiment_topics.csv"
OUTPUT_TOPICS_PATH = "data/processed/topic_keywords_by_sentiment.csv"


SENTIMENT_TOPIC_NAMES = {
    "positive": {
        0: "Positive Brand / Product Praise",
        1: "Positive Camera and Design Feedback",
        2: "Positive AI / Feature Appreciation",
        3: "Positive Upgrade or Buying Interest",
    },
    "negative": {
        0: "Negative Feature Removal Concerns",
        1: "Negative Battery / Charging Issues",
        2: "Negative Price / Value Concerns",
        3: "Negative Camera / Display Complaints",
    },
    "neutral": {
        0: "Neutral Product Discussion",
        1: "Neutral Samsung vs Apple Comparison",
        2: "Neutral Model / Upgrade Discussion",
        3: "Neutral AI / Feature Discussion",
    },
}


def build_vectorizer():
    return CountVectorizer(
        stop_words="english",
        max_df=0.90,
        min_df=3,
        ngram_range=(1, 2),
        max_features=800,
    )


def display_topics(model, feature_names, sentiment_label, no_top_words=12):
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_words = [
            feature_names[index]
            for index in topic.argsort()[:-no_top_words - 1:-1]
        ]

        topic_name = SENTIMENT_TOPIC_NAMES.get(sentiment_label, {}).get(
            topic_idx,
            f"{sentiment_label.title()} Topic {topic_idx}",
        )

        topics.append(
            {
                "sentiment_label": sentiment_label,
                "sentiment_topic_id": topic_idx,
                "top_words": ", ".join(top_words),
                "sentiment_topic_name": topic_name,
            }
        )

    return pd.DataFrame(topics)


def run_topic_model_for_sentiment(df, sentiment_label, n_topics=4):
    sentiment_df = df[df["sentiment_label"] == sentiment_label].copy()

    if len(sentiment_df) < 50:
        print(f"Skipping {sentiment_label}: not enough comments.")
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = build_vectorizer()
    document_term_matrix = vectorizer.fit_transform(sentiment_df["clean_comment"])

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )

    lda_output = lda_model.fit_transform(document_term_matrix)
    sentiment_df["sentiment_topic_id"] = lda_output.argmax(axis=1)
    sentiment_df["sentiment_topic_confidence"] = lda_output.max(axis=1).round(3)
    sentiment_df["sentiment_topic_name"] = sentiment_df["sentiment_topic_id"].map(
        SENTIMENT_TOPIC_NAMES.get(sentiment_label, {})
    )

    topics_df = display_topics(
        model=lda_model,
        feature_names=vectorizer.get_feature_names_out(),
        sentiment_label=sentiment_label,
        no_top_words=12,
    )

    return sentiment_df, topics_df


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].astype(str)

    df_main = df[
        (df["issue_category"] != "Non-English") &
        (df["word_count"] >= 3)
    ].copy()

    all_comment_outputs = []
    all_topic_outputs = []

    for sentiment_label in ["positive", "negative", "neutral"]:
        print("\n====================================")
        print(f"Running topic modeling for: {sentiment_label}")
        print("====================================")

        sentiment_comments, sentiment_topics = run_topic_model_for_sentiment(
            df=df_main,
            sentiment_label=sentiment_label,
            n_topics=4,
        )

        if not sentiment_comments.empty:
            all_comment_outputs.append(sentiment_comments)

        if not sentiment_topics.empty:
            all_topic_outputs.append(sentiment_topics)
            print("\nDiscovered topics:")
            print(sentiment_topics)
            print("\nTopic counts:")
            print(sentiment_comments["sentiment_topic_name"].value_counts())

    final_comments = (
        pd.concat(all_comment_outputs, ignore_index=True)
        if all_comment_outputs
        else pd.DataFrame()
    )
    final_topics = (
        pd.concat(all_topic_outputs, ignore_index=True)
        if all_topic_outputs
        else pd.DataFrame()
    )

    os.makedirs("data/processed", exist_ok=True)
    final_comments.to_csv(OUTPUT_COMMENTS_PATH, index=False, encoding="utf-8-sig")
    final_topics.to_csv(OUTPUT_TOPICS_PATH, index=False, encoding="utf-8-sig")

    print("\nSentiment-based topic modeling completed!")
    print("Saved comments to:", OUTPUT_COMMENTS_PATH)
    print("Saved topic keywords to:", OUTPUT_TOPICS_PATH)


if __name__ == "__main__":
    main()
