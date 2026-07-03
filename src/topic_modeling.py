import os

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


INPUT_PATH = "data/processed/comments_with_categories.csv"
OUTPUT_COMMENTS_PATH = "data/processed/comments_with_topics.csv"
OUTPUT_TOPICS_PATH = "data/processed/topic_keywords.csv"

N_TOPICS = 8

TOPIC_NAMES = {
    0: "General Samsung / Apple Discussion",
    1: "AI, Battery, Camera and Feature Discussion",
    2: "Galaxy Models / S23-S24-S25 Series",
    3: "Buying Decision / Value Discussion",
    4: "Fake Products / Screen / Camera Concerns",
    5: "S-Pen / Bluetooth Removal",
    6: "Upgrade Cycle / Yearly Phone Changes",
    7: "S25 vs S24 / iPhone Pro Max Comparison",
}


def build_vectorizer():
    return CountVectorizer(
        stop_words="english",
        max_df=0.90,
        min_df=3,
        ngram_range=(1, 2),
        max_features=800,
    )


def display_topics(model, feature_names, no_top_words=12):
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_words = [
            feature_names[index]
            for index in topic.argsort()[:-no_top_words - 1:-1]
        ]

        topics.append(
            {
                "topic_id": topic_idx,
                "topic_name": TOPIC_NAMES.get(topic_idx, f"Topic {topic_idx}"),
                "top_words": ", ".join(top_words),
            }
        )

    return pd.DataFrame(topics)


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].astype(str)

    df_main = df[
        (df["issue_category"] != "Non-English") &
        (df["word_count"] >= 3)
    ].copy()

    vectorizer = build_vectorizer()
    document_term_matrix = vectorizer.fit_transform(df_main["clean_comment"])

    lda_model = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="batch",
    )

    lda_output = lda_model.fit_transform(document_term_matrix)
    df_main["topic_id"] = lda_output.argmax(axis=1)
    df_main["topic_confidence"] = lda_output.max(axis=1).round(3)
    df_main["topic_name"] = df_main["topic_id"].map(TOPIC_NAMES)

    topics_df = display_topics(
        model=lda_model,
        feature_names=vectorizer.get_feature_names_out(),
        no_top_words=12,
    )

    os.makedirs("data/processed", exist_ok=True)
    df_main.to_csv(OUTPUT_COMMENTS_PATH, index=False, encoding="utf-8-sig")
    topics_df.to_csv(OUTPUT_TOPICS_PATH, index=False, encoding="utf-8-sig")

    print("General topic modeling completed!")
    print("Saved comments to:", OUTPUT_COMMENTS_PATH)
    print("Saved topic keywords to:", OUTPUT_TOPICS_PATH)
    print("\nDiscovered topics:")
    print(topics_df)
    print("\nTopic counts:")
    print(df_main["topic_name"].value_counts())


if __name__ == "__main__":
    main()
