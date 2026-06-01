import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = "data/processed/comments_with_categories.csv"

OVERALL_OUTPUT_PATH = "data/processed/top_keywords_overall.csv"
CATEGORY_OUTPUT_PATH = "data/processed/top_keywords_by_category.csv"
NEGATIVE_OUTPUT_PATH = "data/processed/top_negative_keywords.csv"


CUSTOM_STOP_WORDS = [
    "samsung", "phone", "phones", "galaxy", "ultra", "s25", "s24",
    "s23", "iphone", "apple", "just", "like", "really", "thing",
    "things", "people", "new", "use", "using", "got", "get", "make",
    "don", "doesn", "didn", "isn", "can", "will", "would", "want",
    "one", "yes", "no", "bro", "lol", "please"
]


def extract_keywords(texts, top_n=30):
    """
    Extracts top keywords/phrases using TF-IDF.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=800,
        ngram_range=(1, 3),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    keywords_df = pd.DataFrame({
        "keyword": feature_names,
        "tfidf_score": scores
    })

    # Remove custom generic words
    keywords_df = keywords_df[
        ~keywords_df["keyword"].isin(CUSTOM_STOP_WORDS)
    ]

    # Also remove phrases that are only generic words
    def is_useful_keyword(keyword):
        words = keyword.split()
        return not all(word in CUSTOM_STOP_WORDS for word in words)

    keywords_df = keywords_df[
        keywords_df["keyword"].apply(is_useful_keyword)
    ]

    keywords_df = keywords_df.sort_values(
        by="tfidf_score",
        ascending=False
    ).head(top_n)

    return keywords_df


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].astype(str)

    # Focus main keyword extraction on English/useful categories
    df_main = df[
        (df["issue_category"] != "Non-English") &
        (df["word_count"] >= 3)
    ].copy()

    os.makedirs("data/processed", exist_ok=True)

    # =========================
    # 1. Overall keywords
    # =========================

    overall_keywords = extract_keywords(df_main["clean_comment"], top_n=50)
    overall_keywords.to_csv(OVERALL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nTop overall keywords:")
    print(overall_keywords.head(20))

    # =========================
    # 2. Negative comment keywords
    # =========================

    negative_df = df_main[df_main["sentiment_label"] == "negative"]

    if len(negative_df) > 0:
        negative_keywords = extract_keywords(negative_df["clean_comment"], top_n=50)
        negative_keywords.to_csv(NEGATIVE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

        print("\nTop negative keywords:")
        print(negative_keywords.head(20))

    # =========================
    # 3. Keywords by issue category
    # =========================

    all_category_keywords = []

    for category in df_main["issue_category"].dropna().unique():

        if category in ["Non-English", "Other"]:
            continue

        category_df = df_main[df_main["issue_category"] == category]

        if len(category_df) < 10:
            continue

        try:
            keywords = extract_keywords(category_df["clean_comment"], top_n=20)
            keywords["issue_category"] = category
            keywords["category_comment_count"] = len(category_df)

            all_category_keywords.append(keywords)

        except ValueError:
            continue

    if all_category_keywords:
        category_keywords_df = pd.concat(all_category_keywords, ignore_index=True)
        category_keywords_df.to_csv(CATEGORY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

        print("\nTop keywords by category saved.")

    print("\nKeyword extraction completed!")
    print("Saved files:")
    print(OVERALL_OUTPUT_PATH)
    print(NEGATIVE_OUTPUT_PATH)
    print(CATEGORY_OUTPUT_PATH)


if __name__ == "__main__":
    main()