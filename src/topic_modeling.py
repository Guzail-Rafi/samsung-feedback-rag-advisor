import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


INPUT_PATH = "data/processed/comments_with_categories.csv"
OUTPUT_COMMENTS_PATH = "data/processed/comments_with_topics.csv"
OUTPUT_TOPICS_PATH = "data/processed/topic_keywords.csv"


CUSTOM_STOP_WORDS = [
    "samsung", "phone", "phones", "galaxy", "ultra", "s25", "s24",
    "s23", "iphone", "apple", "just", "like", "really", "thing",
    "things", "people", "new", "use", "using", "got", "get", "make",
    "don", "doesn", "didn", "isn", "can", "will", "would", "want",
    "one", "yes", "no", "bro", "lol", "please"
]


def display_topics(model, feature_names, no_top_words=10):
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]
        ]

        topics.append({
            "topic_id": topic_idx,
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topics)


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].astype(str)

    # Focus on useful English comments
    df_main = df[
        (df["issue_category"] != "Non-English") &
        (df["word_count"] >= 3)
    ].copy()

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.90,
        min_df=5,
        ngram_range=(1, 2),
        max_features=1000
    )

    document_term_matrix = vectorizer.fit_transform(df_main["clean_comment"])

    lda_model = LatentDirichletAllocation(
        n_components=8,
        random_state=42,
        learning_method="batch"
    )

    lda_output = lda_model.fit_transform(document_term_matrix)

    # Assign dominant topic to each comment
    df_main["topic_id"] = lda_output.argmax(axis=1)
    df_main["topic_confidence"] = lda_output.max(axis=1).round(3)

    feature_names = vectorizer.get_feature_names_out()
    topics_df = display_topics(lda_model, feature_names, no_top_words=12)

    # Optional manual topic names based on expected Samsung themes
    topic_name_map = {
        0: "General Samsung / Apple Discussion",
        1: "AI, Battery, Camera and Feature Discussion",
        2: "Galaxy Models / S23-S24-S25 Series",
        3: "Buying Decision / Value Discussion",
        4: "Fake Products / Screen / Camera Concerns",
        5: "S-Pen / Bluetooth Removal",
        6: "Upgrade Cycle / Yearly Phone Changes",
        7: "S25 vs S24 / iPhone Pro Max Comparison"
    }

    df_main["topic_name"] = df_main["topic_id"].map(topic_name_map)
    topics_df["topic_name"] = topics_df["topic_id"].map(topic_name_map)

    os.makedirs("data/processed", exist_ok=True)

    df_main.to_csv(OUTPUT_COMMENTS_PATH, index=False, encoding="utf-8-sig")
    topics_df.to_csv(OUTPUT_TOPICS_PATH, index=False, encoding="utf-8-sig")

    print("Topic modeling completed!")
    print("Saved to:", OUTPUT_COMMENTS_PATH)
    print("Saved to:", OUTPUT_TOPICS_PATH)

    print("\nDiscovered topics:")
    print(topics_df)

    print("\nTopic counts:")
    print(df_main["topic_name"].value_counts())


if __name__ == "__main__":
    main()