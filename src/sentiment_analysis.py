import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_PATH = "data/processed/clean_comments.csv"
OUTPUT_PATH = "data/processed/comments_with_sentiment.csv"


def get_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    df = pd.read_csv(INPUT_PATH)

    analyzer = SentimentIntensityAnalyzer()

    # Text sentiment using VADER
    sentiment_scores = df["clean_comment"].apply(analyzer.polarity_scores)

    df["text_negative"] = sentiment_scores.apply(lambda x: x["neg"])
    df["text_neutral"] = sentiment_scores.apply(lambda x: x["neu"])
    df["text_positive"] = sentiment_scores.apply(lambda x: x["pos"])
    df["text_sentiment_score"] = sentiment_scores.apply(lambda x: x["compound"])

    # Combine text sentiment with emoji sentiment
    # 80% text + 20% emoji
    df["final_sentiment_score"] = (
        0.8 * df["text_sentiment_score"] +
        0.2 * df["emoji_sentiment_score"]
    )

    # Round scores to make CSV cleaner
    df["emoji_sentiment_score"] = df["emoji_sentiment_score"].round(3)
    df["text_sentiment_score"] = df["text_sentiment_score"].round(3)
    df["final_sentiment_score"] = df["final_sentiment_score"].round(3)

    df["sentiment_label"] = df["final_sentiment_score"].apply(get_sentiment_label)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Sentiment analysis completed!")
    print("Saved to:", OUTPUT_PATH)

    print("\nSentiment counts:")
    print(df["sentiment_label"].value_counts())

    print("\nPreview:")
    print(df[[
        "original_comment",
        "clean_comment",
        "text_sentiment_score",
        "emoji_sentiment_score",
        "final_sentiment_score",
        "sentiment_label"
    ]].head(10))


if __name__ == "__main__":
    main()