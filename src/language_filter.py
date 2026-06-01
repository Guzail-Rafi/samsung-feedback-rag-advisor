import os
import pandas as pd
from langdetect import detect, LangDetectException


INPUT_PATH = "data/processed/comments_with_sentiment.csv"
OUTPUT_ENGLISH_PATH = "data/processed/comments_with_sentiment_english.csv"
OUTPUT_NON_ENGLISH_PATH = "data/processed/non_english_comments.csv"


def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["language"] = df["clean_comment"].apply(detect_language)

    english_df = df[df["language"] == "en"].copy()
    non_english_df = df[df["language"] != "en"].copy()

    os.makedirs("data/processed", exist_ok=True)

    english_df.to_csv(OUTPUT_ENGLISH_PATH, index=False, encoding="utf-8-sig")
    non_english_df.to_csv(OUTPUT_NON_ENGLISH_PATH, index=False, encoding="utf-8-sig")

    print("Language detection completed!")
    print("Total comments:", len(df))
    print("English comments:", len(english_df))
    print("Non-English/unknown comments:", len(non_english_df))

    print("\nLanguage counts:")
    print(df["language"].value_counts().head(20))

    print("\nSaved English file to:", OUTPUT_ENGLISH_PATH)
    print("Saved non-English file to:", OUTPUT_NON_ENGLISH_PATH)


if __name__ == "__main__":
    main()